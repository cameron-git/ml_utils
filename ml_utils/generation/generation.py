import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from ..cache import DynamicCache

__all__ = ["GenerationMixin"]


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


def sample(
    logits: torch.Tensor,
    top_k: int = 1,
    top_p: float = 0.0,
    min_p: float = 0.0,
    temperature: float = 1.0,
):
    if top_k == 1:
        return logits.argmax(dim=-1)
    else:
        raise NotImplementedError("only top_k=1 sampling is implemented.")


@torch.no_grad()
def decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_length: int,
    top_k: int = 1,
    top_p: float = 0.0,
    min_p: float = 0.0,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
    vocab_size: Optional[int] = None,
):
    batch_size, prompt_len = input_ids.shape
    inference_params = InferenceParams(
        max_seqlen=max_length,
        max_batch_size=batch_size,
    )
    cache = DynamicCache()

    def get_logits(input_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            pos_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            pos_ids = None
        output = model(
            input_ids,
            pos_ids=pos_ids,
            cache=cache,
        )
        logits = output["logits"]
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    sequences = [input_ids]
    while not should_stop(sequences[-1], inference_params):
        logits = get_logits(sequences[-1], inference_params)
        inference_params.seqlen_offset += sequences[-1].shape[1]
        sampled_tokens = sample(
            logits=logits,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
        )
        sequences.append(sampled_tokens)
    return torch.cat(sequences, dim=1)


class GenerationMixin:
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        top_k: int = 1,
        top_p: float = 0.0,
        min_p: float = 0.0,
        temperature: float = 1.0,
    ):
        output = decode(
            model=self,
            input_ids=input_ids,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
        )
        return output
