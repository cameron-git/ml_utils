import torch
import torch.nn as nn
from torch.nn.attention import flex_attention

from typing import Optional

from ....cache import Cache
from ....functional.rope import apply_rope


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads

        self.Q = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.K = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.V = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.O = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: flex_attention.BlockMask | None = None,
        cache: Cache = None,
        cache_key: str | None = None,
        pos_emb: dict[str, torch.Tensor] | None = None,
    ):
        B, T, D = query.shape

        key = key if key is not None else query
        value = value if value is not None else query

        query = self.Q(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.K(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.V(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if pos_emb is not None:
            query = apply_rope(query, **pos_emb)
            key = apply_rope(key, **pos_emb)

        if cache is not None:
            assert cache_key is not None
            key = cache.update(f"{cache_key}_key", key, dim=-2)
            value = cache.update(f"{cache_key}_value", value, dim=-2)

        out = flex_attention(
            query=query,
            key=key,
            value=value,
            block_mask=mask,
        )

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out = self.O(out)
        return {"x": out}
