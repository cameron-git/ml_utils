import torch

from ..misc import rotate_half

__all__ = ["generate_rope", "generate_rope_dynamic", "apply_rope"]


def generate_rope(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
):
    # Officially pos should start from 1 but no one seems to care
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    # pos += 1.0
    theta = 1.0 / (
        torch.pow(base, torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    f = pos[..., None] @ theta[None, :]
    f = f.repeat_interleave(2, dim=-1)
    cos = f.cos()
    sin = f.sin()
    return {"cos": cos, "sin": sin}


def generate_rope_dynamic(
    pos_ids: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
):
    theta = 1.0 / (
        torch.pow(base, torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    ).to(pos_ids.device)
    f = pos_ids[..., None].type(theta.dtype) @ theta[None, :]
    f = f.repeat_interleave(2, dim=-1).unsqueeze(1)
    cos = f.cos()
    sin = f.sin()
    return {"cos": cos, "sin": sin}


def apply_rope(x, cos, sin):
    dtype = x.dtype
    x = (x * cos) + (rotate_half(x) * sin)
    x = x.to(dtype=dtype)
    return x
