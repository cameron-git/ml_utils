import torch
import torch.nn as nn
from accelerate import Accelerator

__all__ = ["train_step"]

# NOTE: Gradient accumulation assumes that loss is calculated for all tokens in the batch.
# If padding is used another method should be used.
# https://github.com/huggingface/accelerate/blob/main/examples/by_feature/gradient_accumulation_for_autoregressive_models.py


def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_grad_norm: float | None = None,
    max_grad_value: float | None = None,
):
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs["loss"]
        accelerator.backward(loss)
        outputs["grad_norm"] = None
        if accelerator.sync_gradients:
            if max_grad_norm is not None and max_grad_norm > 0:
                outputs["grad_norm"] = accelerator.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            if max_grad_value is not None and max_grad_value > 0:
                accelerator.clip_grad_value_(model.parameters(), max_grad_value)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()
    return outputs
