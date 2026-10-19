#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_fwd_train_wy CPU golden reference (higher precision, FP32).

OP-A of the gated_delta_rule TRAINING-FORWARD pair = the WY-build front-end
(Modules M1-M4, BHN-parallel). This is a MECHANICAL PROJECTION (a faithful slice),
NOT fresh math: it computes exactly the M1-M4 mathematics of the READ-ONLY delivered
forward golden

    gated_delta_rule_fwd_golden_cpu.py  (function _chunk)

up to (and including) `k_cumsum = attn @ v` / `k_cumdecay = attn @ (k_beta*decay_exp)`
-- i.e. the chunk-parallel WY build BEFORE the inter-chunk s12-s16 scan -- and exposes
the 17 GM boundary tensors the lean forward computes internally and discards.

Signature:  gdr_fwd_train_wy_golden_cpu(q, k, v, g, beta) -> (17 fp32 tensors)

Returned tuple order:
   0 q_s        [B,H,N,C,K]   scaled q               (= q * scale)
   1 k_f32      [B,H,N,C,K]   padded fp32 k
   2 v_b        [B,H,N,C,V]   v * beta_act
   3 g_f32      [B,H,N,C]      padded fp32 gate (pre-cumsum)
   4 beta_act   [B,H,N,C]      sigmoid(beta_logit), pad rows 0
   5 sig        [B,H,N,C]      == beta_act (kept for backward dbeta)
   6 q_t        [B,H,N,K,C]   q_s ^T
   7 decay      [B,H,N,C]      cumsum(g_f32) over C
   8 decay_exp  [B,H,N,C]      exp(decay)
   9 dfac       [B,H,N,C,K]   exp(decay_last - decay), broadcast over K
  10 q_de       [B,H,N,C,K]   q_s * decay_exp
  11 attn       [B,H,N,C,C]   (I - tril_strict(P1*L_mask))^{-1}  (fwd substitution + I)
  12 a_mat      [B,H,N,C,C]   tril_strict(P2*L_mask)   (== A)
  13 kbeta_t    [B,H,N,K,C]   kbeta_de ^T
  14 u          [B,H,N,C,V]   attn @ v_b        (= chunk k_cumsum, the new v)
  15 w          [B,H,N,C,K]   attn @ kbeta_de   (= chunk k_cumdecay)
  16 w_t        [B,H,N,K,C]   w ^T

Semantic notes (identical lineage to the forward golden):
  - beta is passed as sigmoid LOGITS; sigmoid applied inside, THEN pad-with-0 (so pad
    rows carry beta_act=0). g is log-space decay (<=0), padded with 0.
  - scale = 1/sqrt(K); no q/k l2norm; H==HV.
  - Every output is in internal [B,H,N,...] layout (N=(T+63)//64, C=64). NO [:T] slice is
    applied here -- the terminal slice belongs to OP-B's s16 (on the final output o).
  - All arithmetic fp32; the 17 outputs are all fp32 (no narrow).
  - k_beta, P1, P2, kbeta_de, L_mask are INTERNAL scratch, NOT materialised.
"""

import torch
import torch.nn.functional as F

_CHUNK_SIZE = 64


def gdr_fwd_train_wy_golden_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
):
    """CPU FP32 golden for OP-A (M1-M4 WY-build front-end).

    Args:
        q, k : [B, T, H, K]  (any float dtype; promoted to fp32)
        v    : [B, T, H, V]
        g    : [B, T, H]     (log-space decay, <= 0)
        beta : [B, T, H]     (sigmoid LOGITS)

    Returns:
        tuple of 17 float32 tensors, in the documented output order (internal [B,H,N,...] layout).
    """
    C = _CHUNK_SIZE

    # ---- promote to fp32 (verbatim fwd golden: sigmoid BEFORE transpose/pad) ----
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    g = g.to(torch.float32)
    beta_logit = beta.to(torch.float32)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    beta_act_full = torch.sigmoid(beta_logit)                      # [B,T,H]

    # ---- public [B,T,H,*] -> internal [B,H,T,*] ----
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()
    beta_a = beta_act_full.transpose(1, 2).contiguous()           # [B,H,T]

    # ---- zero-pad T -> Tp = N*C ----
    T = q.shape[-2]
    pad = (C - (T % C)) % C
    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        g = F.pad(g, (0, pad))
        beta_a = F.pad(beta_a, (0, pad))                          # pad rows: beta_act = 0
    b, h, Tp, K = q.shape
    V = v.shape[-1]
    N = Tp // C

    # ================= M1 preprocess (s0-s3, pure Vector) =================
    q_s_flat = q * scale                                          # [b,h,Tp,K]
    v_b_flat = v * beta_a[..., None]                             # v * beta_act
    k_beta_flat = k * beta_a[..., None]                         # internal scratch

    q_s = q_s_flat.reshape(b, h, N, C, K)
    k_f32 = k.reshape(b, h, N, C, K)
    v_b = v_b_flat.reshape(b, h, N, C, V)
    k_beta = k_beta_flat.reshape(b, h, N, C, K)                  # internal
    g_f32 = g.reshape(b, h, N, C)
    beta_act = beta_a.reshape(b, h, N, C)
    sig = beta_act                                               # sig == beta_act
    q_t = q_s.transpose(-1, -2).contiguous()                     # [b,h,N,K,C]

    # ================= M2 cumsum_qk (s5, s7mm, s11mm, Cube) =================
    decay = g_f32.cumsum(-1)                                     # [b,h,N,C]
    kf_t = k_f32.transpose(-1, -2)                              # [b,h,N,K,C]
    P1 = torch.matmul(k_beta, kf_t)                            # k_beta @ k^T   [b,h,N,C,C] internal
    P2 = torch.matmul(q_s, kf_t)                               # q_s   @ k^T   [b,h,N,C,C] internal

    # ================= M3 wy_build (s6, s7mask, s8, s9, s11mask, Vector) =================
    decay_exp = decay.exp()                                      # [b,h,N,C]
    q_de = q_s * decay_exp[..., None]                          # [b,h,N,C,K]
    kbeta_de = k_beta * decay_exp[..., None]                   # internal scratch
    dfac = (decay[..., -1:] - decay).exp()[..., None].expand(b, h, N, C, K).contiguous()
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp()).tril()   # [b,h,N,C,C]

    up_incl_diag = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
    attn = -(P1 * L_mask).masked_fill(up_incl_diag, 0)
    for i in range(1, C):                                        # s8 forward substitution
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(C, dtype=torch.float32)             # + I

    up_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    a_mat = (P2 * L_mask).masked_fill(up_strict, 0)             # tril_strict(P2*L_mask)  (== A)
    kbeta_t = kbeta_de.transpose(-1, -2).contiguous()           # [b,h,N,K,C]

    # ================= M4 uw (s10, Cube) =================
    u = torch.matmul(attn, v_b)                                 # chunk k_cumsum  [b,h,N,C,V]
    w = torch.matmul(attn, kbeta_de)                          # chunk k_cumdecay [b,h,N,C,K]
    w_t = w.transpose(-1, -2).contiguous()                     # [b,h,N,K,C]

    return (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
            dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t)


# ==========================================
# 验证
# ==========================================

_OUT_NAMES = ["q_s", "k_f32", "v_b", "g_f32", "beta_act", "sig", "q_t", "decay",
              "decay_exp", "dfac", "q_de", "attn", "a_mat", "kbeta_t", "u", "w", "w_t"]

# acceptance cases: (name, B,T,H,K,V)
_P0 = [
    ("aligned_b1_t128_h4_k64_v64", 1, 128, 4, 64, 64),
    ("ttail_b1_t100_h4_k64_v64",   1, 100, 4, 64, 64),
    ("t3c_b1_t192_h2_k128_v128",   1, 192, 2, 128, 128),
    ("tail_vlt_b1_t200_h3_k128_v96", 1, 200, 3, 128, 96),
]
_SEEDS = [42, 123, 456]


def _expected_shapes(B, T, H, K, V):
    N = (T + 63) // 64
    C = 64
    return {
        "q_s": (B, H, N, C, K), "k_f32": (B, H, N, C, K), "v_b": (B, H, N, C, V),
        "g_f32": (B, H, N, C), "beta_act": (B, H, N, C), "sig": (B, H, N, C),
        "q_t": (B, H, N, K, C), "decay": (B, H, N, C), "decay_exp": (B, H, N, C),
        "dfac": (B, H, N, C, K), "q_de": (B, H, N, C, K), "attn": (B, H, N, C, C),
        "a_mat": (B, H, N, C, C), "kbeta_t": (B, H, N, K, C), "u": (B, H, N, C, V),
        "w": (B, H, N, C, K), "w_t": (B, H, N, K, C),
    }


def _mk_case(B, T, H, K, V, seed, g_lo=-16.0, g_hi=0.0):
    torch.manual_seed(seed)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(g_lo, g_hi)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return q, k, v, g, beta


def _validate():
    print("=" * 64)
    print("gdr_fwd_train_wy_golden_cpu 验证报告")
    print("=" * 64)
    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    # 1. All P0 shapes: count / shape / dtype / finite.
    print("\n[1. P0 shape / dtype / finite]")
    for (name, B, T, H, K, V) in _P0:
        exp = _expected_shapes(B, T, H, K, V)
        for seed in _SEEDS:
            outs = gdr_fwd_train_wy_golden_cpu(*_mk_case(B, T, H, K, V, seed))
            ok = _check(f"{name} seed{seed}: 17 outputs", len(outs) == 17)
            for nm, t in zip(_OUT_NAMES, outs):
                ok &= (tuple(t.shape) == exp[nm]) and (t.dtype == torch.float32) \
                    and bool(torch.isfinite(t).all())
            _check(f"{name} seed{seed}: shapes/dtype(fp32)/finite", ok)

    # 2. Internal algebraic identities (self-consistency of the slice).
    print("\n[2. 内部代数恒等式]")
    q, k, v, g, beta = _mk_case(1, 128, 4, 64, 64, seed=42)
    (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
     dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t) = gdr_fwd_train_wy_golden_cpu(q, k, v, g, beta)
    _check("sig == beta_act", torch.equal(sig, beta_act))
    _check("q_t == q_s^T", torch.allclose(q_t, q_s.transpose(-1, -2), atol=0, rtol=0))
    _check("w_t == w^T", torch.allclose(w_t, w.transpose(-1, -2), atol=0, rtol=0))
    _check("decay_exp == exp(decay)", torch.allclose(decay_exp, decay.exp(), atol=1e-6, rtol=1e-5))
    _check("q_de == q_s*decay_exp", torch.allclose(q_de, q_s * decay_exp[..., None], atol=1e-6, rtol=1e-5))
    _check("u == attn@v_b", torch.allclose(u, torch.matmul(attn, v_b), atol=1e-4, rtol=1e-4))
    # a_mat is lower-triangular INCLUDING the diagonal (fwd code masks up_strict only,
    # i.e. masked_fill(triu diagonal=1) -> keeps diagonal); so only strict-upper is 0.
    C = 64
    up_str = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    _check("a_mat strict-upper == 0 (lower-tri incl diag)",
           a_mat.masked_select(up_str).abs().max().item() == 0.0)
    # attn is unit-lower-triangular (diag == 1, strict upper == 0).
    diag = torch.arange(C)
    _check("attn diag == 1",
           torch.allclose(attn[..., diag, diag], torch.ones_like(attn[..., diag, diag]), atol=1e-6, rtol=0))
    _check("attn strict-upper == 0", attn.masked_select(up_str).abs().max().item() == 0.0)

    # 3. Value ranges from the formula.
    print("\n[3. 值域检查]")
    _check("beta_act in [0,1]", (beta_act.min() >= 0.0) and (beta_act.max() <= 1.0))
    _check("decay_exp in (0,1]", (decay_exp.min() >= 0.0) and (decay_exp.max() <= 1.0 + 1e-6))
    _check("dfac in (0,1]", (dfac.min() >= 0.0) and (dfac.max() <= 1.0 + 1e-6))
    _check("g_f32 <= 0", g_f32.max().item() <= 1e-6)

    # 4. Padding rows (T not multiple of C): beta_act pad rows == 0.
    print("\n[4. padding 语义 (T=100)]")
    q, k, v, g, beta = _mk_case(1, 100, 4, 64, 64, seed=1)
    outs = gdr_fwd_train_wy_golden_cpu(q, k, v, g, beta)
    ba = outs[4]  # beta_act [B,H,N,C]; N=2, pad = 28 rows in chunk 1 (rows 36..63)
    # T=100 -> chunk0 rows 0..63 valid; chunk1 rows 0..35 valid, 36..63 pad.
    pad_slice = ba[:, :, 1, 36:]
    _check("T=100: beta_act pad rows == 0", pad_slice.abs().max().item() == 0.0)
    _check("T=100: all outputs finite", all(torch.isfinite(t).all() for t in outs))

    # 5. very-negative g -> exp underflow to 0, no NaN/Inf.
    print("\n[5. 数值稳定性]")
    q, k, v, g, beta = _mk_case(1, 128, 2, 64, 64, seed=3, g_lo=-16.0, g_hi=-8.0)
    outs = gdr_fwd_train_wy_golden_cpu(q, k, v, g, beta)
    _check("very-negative g: all finite", all(torch.isfinite(t).all() for t in outs))

    print("\n" + "=" * 64)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 64)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
