from torch.nn.attention import flex_attention


def causal_mask(
    q_len: int,
    kv_len: int,
):
    def mask_mod(batch_size, num_heads, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = flex_attention.create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device="cuda",
        _compile=True,
    )
    return block_mask


def causal_block_mask(
    q_len: int,
    kv_len: int,
    block_size: int,
):
    def mask_mod(batch_size, num_heads, q_idx, kv_idx):
        q_block_idx = q_idx // block_size
        kv_block_idx = kv_idx // block_size
        return q_block_idx >= kv_block_idx

    block_mask = flex_attention.create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device="cuda",
        _compile=True,
    )
    return block_mask
