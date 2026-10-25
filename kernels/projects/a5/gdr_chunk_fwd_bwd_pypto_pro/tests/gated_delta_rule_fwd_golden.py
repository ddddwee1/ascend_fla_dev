# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Float32 CPU reference for gated_delta_rule_fwd.

Chunk-parallel WY formulation, run entirely on CPU in float32: bfloat16 inputs
are promoted and the result is NOT cast back down, so it is the ground truth
the bfloat16 kernel output is judged against.

Conventions the kernel shares:
  * beta is passed as LOGITS; sigmoid is applied here, so golden and kernel
    compute identical math.
  * g is a log-space decay, semantically <= 0.
  * scale = 1 / sqrt(K); no q/k L2 norm; H == HV; no initial or final state.
"""

import torch
import torch.nn.functional as F

_CHUNK_SIZE = 64


# ─────────────────────────────────────────────
# Internal reference kernels (beta ALREADY activated; public layout [B,T,H,*]; fp32).
# ─────────────────────────────────────────────

def _recurrent(q, k, v, beta, g, scale):
    """Per-token ground-truth recurrence (verbatim naive_recurrent semantics, fp32)."""
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                        for x in (q, k, v, beta, g)]
    B, H, T, K = k.shape
    V = v.shape[-1]
    o = torch.zeros(B, H, T, V, dtype=torch.float32)
    h = torch.zeros(B, H, K, V, dtype=torch.float32)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = (b_q[..., None] * h).sum(-2)
    o = o.transpose(1, 2).contiguous()
    return o, h


def _chunk(q, k, v, g, beta, scale, chunk_size=_CHUNK_SIZE):
    """Chunk-parallel WY reformulation (verbatim naive_chunk semantics, fp32)."""
    BT = chunk_size
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                        for x in (q, k, v, beta, g)]
    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))
    decay = g
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % chunk_size == 0
    n = l // chunk_size

    q = q.reshape(b, h, n, chunk_size, d_k)
    k = k.reshape(b, h, n, chunk_size, d_k)
    v = v.reshape(b, h, n, chunk_size, d_v)
    k_beta = k_beta.reshape(b, h, n, chunk_size, d_k)
    decay = decay.reshape(b, h, n, chunk_size)

    decay = decay.cumsum(-1)
    decay_exp = decay.exp()[..., None]

    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()

    up_incl_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(up_incl_diag, 0)

    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=torch.float32)

    k_cumsum = attn @ v
    k_cumdecay = attn @ (k_beta * decay_exp)
    v = k_cumsum

    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    up_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    for i in range(0, n):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        A_i = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill(up_strict, 0)
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + A_i @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (
            k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new

    o = o.reshape(b, h, n * chunk_size, d_v)
    o = o[:, :, :T]
    o = o.transpose(1, 2).contiguous()
    return o, S


# ─────────────────────────────────────────────
# CPU Golden 参考实现（更高精度，FP32）
# ─────────────────────────────────────────────

def gated_delta_rule_fwd_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """CPU FP32 golden for Gated Delta Rule forward (chunk-parallel WY).

    Args:
        q, k : [B, T, H, K]  (any float dtype; promoted to fp32)
        v    : [B, T, H, V]
        g    : [B, T, H]     (log-space decay, <= 0)
        beta : [B, T, H]     (sigmoid LOGITS)

    Returns:
        o : [B, T, H, V]  fp32 (NOT downcast).
    """
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    g = g.to(torch.float32)
    beta = beta.to(torch.float32)

    beta_act = torch.sigmoid(beta)   # logits -> beta
    scale = 1.0 / (q.shape[-1] ** 0.5)
    o, _ = _chunk(q, k, v, g, beta_act, scale)
    return o  # fp32


# ==========================================
# 验证
