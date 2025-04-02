import torch
import torch.nn as nn
from torch.nn.attention import flex_attention
from typing import Callable, Optional

from ..attention.flex import Attention
from ..mlp import MLP
from ...cache import Cache


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        num_heads: int,
        mlp_dim: int,
        norm: nn.Module = nn.RMSNorm,
        mlp_activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu,
    ):
        """Transformer Block.
        Args:
            embed_dim (`int`): Hidden dimension of the transformer block.
            head_dim (`int`): Dimension of each attention head.
            num_heads (`int`): Number of attention heads. Typically embed_dim // head_dim.
            mlp_dim (`int`): Hidden dimension of the MLP, or 'intermediate dimension'. Typically 4 * embed_dim.
            norm (`nn.Module`, *optional*):
                Normalization layer to be used. Defaults to `nn.RMSNorm`.
            mlp_activation (`Callable`, *optional*):
                Activation function for the MLP. Defaults to `nn.functional.silu`.
        """

        super().__init__()
        self.norm1 = norm(embed_dim)
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.norm2 = norm(embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            activation=mlp_activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[flex_attention.BlockMask] = None,
        cache: Optional[Cache] = None,
        cache_key: str = None,
        pos_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the transformer block.
        Args:
            x (`torch.Tensor`): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask (`flex_attention.BlockMask`, *optional*):
                Attention mask. Defaults to None.
            cache (`Cache`, *optional*):
                Cache for storing key and value tensors. Defaults to None.
            cache_key (`str`, *optional*):
                Key for the cache. Defaults to None.
            pos_emb (`tuple[torch.Tensor, torch.Tensor]`, *optional*):
                Positional embeddings. Defaults to None.
        Returns:
            `dict[str, torch.Tensor]`: Dictionary containing the activation tensor.
        """

        r = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask, cache=cache, cache_key=cache_key, pos_emb=pos_emb)[
            "x"
        ]
        x = x + r
        r = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + r
        return {"x": x}
