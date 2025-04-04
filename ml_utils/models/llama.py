import torch
import torch.nn as nn
from torch.nn.attention import flex_attention
from typing import Callable, Optional

from ..functional.rope import generate_rope, generate_rope_dynamic
from ..functional.masks.flex import causal_block_mask
from ..cache import Cache
from ..modules.transformer import Transformer
from ..generation import GenerationMixin

__all__ = ["Llama"]


class Llama(nn.Module, GenerationMixin):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        embed_dim: int,
        head_dim: int,
        num_heads: int,
        mlp_dim: int,
        norm: nn.Module = nn.RMSNorm,
        mlp_activation: Callable = nn.functional.silu,
        max_seq_len: int = 128,
        mask: flex_attention.BlockMask | None = None,
        mask_block_size: int = 1,
    ):
        """Transformer model.
        Args:
            vocab_size (`int`): Size of the vocabulary.
            num_layers (`int`): Number of transformer layers.
            embed_dim (`int`): Hidden dimension of the transformer.
            head_dim (`int`): Dimension of each attention head.
            num_heads (`int`): Number of attention heads.
            mlp_dim (`int`): Dimension of the MLP.
            norm (`nn.Module`, *optional*):
                Normalization layer to be used. Defaults to `nn.RMSNorm`.
            activation (`Callable`, *optional*):
                Activation function for the MLP. Defaults to `nn.functional.silu`.
        """
        super().__init__()

        self.head_dim = head_dim
        self.mask = (
            causal_block_mask(
                q_len=max_seq_len,
                kv_len=max_seq_len,
                block_size=mask_block_size,
            )
            if mask is None
            else mask
        )

        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.model = nn.ModuleList(
            [
                Transformer(
                    embed_dim=embed_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    norm=norm,
                    mlp_activation=mlp_activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.Sequential(norm(embed_dim), nn.Linear(embed_dim, vocab_size))
        pos_emb = generate_rope(max_seq_len, head_dim)
        self.register_buffer("pos_emb_cos", pos_emb["cos"])
        self.register_buffer("pos_emb_sin", pos_emb["sin"])

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        num_last_tokens: int = 0,
        **kwargs,
    ):
        activated = lambda k, d: k in d and d[k] is True
        if activated("output_attentions", kwargs):
            raise ValueError(
                "output_attentions is not supported in this model. Please set it to False."
            )

        # Encode
        x = self.encoder(input_ids)
        hidden_states = (x,) if activated("output_hidden_states", kwargs) else None

        # Position embedding
        if pos_ids:
            pos_emb = {
                "cos": generate_rope_dynamic(pos_ids, self.head_dim)["cos"],
                "sin": generate_rope_dynamic(pos_ids, self.head_dim)["sin"],
            }
        else:
            pos_emb = {
                "cos": self.pos_emb_cos[None, None, : x.size(1)],
                "sin": self.pos_emb_sin[None, None, : x.size(1)],
            }

        # Layers
        for layer_id, layer in enumerate(self.model):
            out = layer(
                x=x,
                pos_emb=pos_emb,
                mask=self.mask if cache is None else None,
                cache=cache,
                cache_key=f"layer_{layer_id}",
            )
            x = out["x"]
            if activated("output_hidden_states", kwargs):
                hidden_states += (x,)

        if num_last_tokens > 0:
            x = x[:, -num_last_tokens:]

        # Decode
        logits = self.decoder(x)

        # Loss
        loss = (
            nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), target_ids.view(-1)
            )
            if target_ids is not None
            else None
        )

        return {
            "logits": logits,
            "pos_ids": pos_ids,
            "loss": loss,
            "perplexity": torch.exp(loss) if loss is not None else None,
            "cache": cache,
            "hidden_states": hidden_states,
        }


Llama
