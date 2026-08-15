# PyTorch native SDPA fallback for devices without flash-attn/natten
import torch
import torch.nn.functional as F
from torch import Tensor
from cosmos_framework.model.attention.masks import CausalType


def sdpa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    is_varlen: bool = False,
    deterministic: bool = False,
    scale: float | None = None,
    **kwargs,
) -> Tensor:
    # query: [B, S, H_Q, D], key/value: [B, S, H_KV, D]
    # Handle GQA: repeat K/V heads to match Q heads
    h_q = query.shape[2]
    h_kv = key.shape[2]
    if h_q != h_kv:
        repeat_factor = h_q // h_kv
        key = key.repeat_interleave(repeat_factor, dim=2)
        value = value.repeat_interleave(repeat_factor, dim=2)

    # [B, S, H, D] -> SDPA expects [B, H, S, D]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    out = F.scaled_dot_product_attention(
        q, k, v,
        is_causal=is_causal,
        scale=scale,
    )

    return out.transpose(1, 2)  # back to [B, S, H, D]
