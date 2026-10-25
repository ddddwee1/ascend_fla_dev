#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_dv_local CPU golden reference (higher precision, FP32).

OP3 of gated_delta_rule_bwd = the intra-chunk (chunk-local) contribution to the dv
gradient (parent module M6, golden_step b1, BHN-parallel). This is a MECHANICAL
PROJECTION (a faithful M6 slice), NOT fresh math: it computes exactly the M6 arithmetic
of the READ-ONLY delivered cumulative staging golden

    gated_delta_rule_bwd_golden_stage123456.py  (_m6)

but consumes OP3's frozen GM contract directly — grad_o (primary, public layout) plus the
three residuals (k_f32, q_t, g_f32) that gdr_fwd_train_wy M1 exports — instead of the
primary q/k/v/g/beta the parent recomputes internally.

──────────────────────────────────────────────────────────────────────────────
PROJECTION MAP  (M6 `_m6` in the parent staging golden  ->  OP3 here)
  parent:  A_l      = matmul(k_f32, q_s.transpose(-1,-2))          # [C,K] @ [K,C] = [C,C]
  OP3:     A_l      = matmul(k_f32, q_t)                            # q_t IS q_s^T = [B,H,N,K,C]
                                                                     # (pre-transposed upstream;
                                                                     #  scale=1/sqrt(K) already
                                                                     #  folded into q_s -> q_t)
  parent:  decay    = g_f32.cumsum(-1)   (g_f32 = pre-cumsum gate)  # M2 computes cumsum
  OP3:     decay    = g_f32.cumsum(-1)   (g_f32 input = pre-cumsum gate, index 3 of
                                          gdr_fwd_train_wy outputs -> identical)
  both:    gdiff[...,i,j] = decay_j - decay_i;  A_l *= exp(gdiff)
  both:    mask strict-lower (i>j) -> 0   (keep UPPER triangle INCLUSIVE of diagonal, i<=j)
  parent:  do       = grad_o chunked [B,H,N,C,V]   (transpose(1,2), pad T->N*C, reshape)
  OP3:     do       = grad_o chunked identically inside this op
  both:    dv_local = matmul(A_l, do)                              # [C,C] @ [C,V] = [C,V]

Signature:  gdr_bwd_dv_local_golden_cpu(grad_o, k_f32, q_t, g_f32) -> dv_local
    grad_o : [B, T, H, V]      (any float dtype; promoted to fp32; PUBLIC layout, unpadded)
    k_f32  : [B, H, N, C, K]   fp32   (residual, internal per-chunk layout, N=(T+63)//64, C=64)
    q_t    : [B, H, N, K, C]   fp32   (residual; q_s^T PRE-TRANSPOSED, q_s = q*scale)
    g_f32  : [B, H, N, C]      fp32   (residual; PRE-cumsum log-space gate, padded rows g=0)
    ->  dv_local : [B, H, N, C, V]  fp32   (NOT narrowed; internal per-chunk layout)

All arithmetic fp32. Padding rows carry g=0 (inert); grad_o pad rows are 0 (pad columns
contribute 0 to any valid row's dv_local). No [:T] slice here (that is OP7's b7).
"""

import torch
import torch.nn.functional as F

_C = 64  # fixed internal chunk size


def gdr_bwd_dv_local_golden_cpu(
    grad_o: torch.Tensor,
    k_f32: torch.Tensor,
    q_t: torch.Tensor,
    g_f32: torch.Tensor,
) -> torch.Tensor:
    """CPU FP32 golden for OP3 (M6 dv_local, faithful slice of the parent b1 step).

    Args:
        grad_o : [B, T, H, V]     (any float dtype; promoted to fp32; public layout)
        k_f32  : [B, H, N, C, K]  fp32
        q_t    : [B, H, N, K, C]  fp32   (q_s^T pre-transposed; scale folded in q_s)
        g_f32  : [B, H, N, C]     fp32   (pre-cumsum log-space gate; padded rows 0)

    Returns:
        dv_local : [B, H, N, C, V]  fp32
    """
    C = _C
    grad_o_f = grad_o.to(torch.float32)
    k_f32 = k_f32.to(torch.float32)
    q_t = q_t.to(torch.float32)
    g_f32 = g_f32.to(torch.float32)

    B, T, H, V = grad_o_f.shape
    _, _, N, _, _ = k_f32.shape  # [B, H, N, C, K]

    # (1) cumulative decay over the C tokens of each chunk (g_f32 is PRE-cumsum).
    decay = g_f32.cumsum(-1)                                      # [B, H, N, C]

    # (2) Cube A: A_l = k @ q^T. q_t is already q_s^T ([B,H,N,K,C]); scale folded in q_s.
    A_l = torch.matmul(k_f32.float(), q_t.float())               # [B, H, N, C, C]

    # (3) Vector: intra-chunk gate-decay factor exp(g_col_j - g_row_i).
    gdiff = decay.unsqueeze(-2) - decay.unsqueeze(-1)            # [...,i,j] = decay_j - decay_i
    A_l = A_l * gdiff.exp()

    # (4) Vector: keep UPPER triangle INCLUSIVE of the diagonal (i<=j); zero strict-lower (i>j).
    low_strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=A_l.device), diagonal=-1)
    A_l = A_l.masked_fill(low_strict, 0.0)

    # (5) grad_o -> chunked do [B,H,N,C,V] (public [B,T,H,V] -> [B,H,T,V] -> pad T->N*C -> reshape).
    do = grad_o_f.transpose(1, 2).contiguous()                   # [B, H, T, V]
    pad = (C - (T % C)) % C
    if pad > 0:
        do = F.pad(do, (0, 0, 0, pad))
    do = do.reshape(B, H, N, C, V)                               # [B, H, N, C, V]

    # (6) Cube B: dv_local = A_l @ do.
    dv_local = torch.matmul(A_l.float(), do.float())             # [B, H, N, C, V]
    return dv_local


# ==========================================
# 验证
# ==========================================

def _mk_residuals(B, T, H, K, V, seed, g_lo=-4.0, g_hi=0.0):
    """Build OP3's GM inputs (grad_o + 3 residuals) directly, at contract shapes/dtypes.

    Random residuals of the right shape/dtype/value-range exercise shape/dtype/finite
    checks. Numeric ground truth is the mandatory cross-check below (residuals derived
    from the real forward + compared to the parent M6 staging golden).
    """
    torch.manual_seed(seed)
    N = (T + 63) // 64
    C = 64
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    k_f32 = torch.randn((B, H, N, C, K), dtype=torch.float32)
    q_t = torch.randn((B, H, N, K, C), dtype=torch.float32)
    g_f32 = torch.empty((B, H, N, C), dtype=torch.float32).uniform_(g_lo, g_hi)
    return grad_o, k_f32, q_t, g_f32


# acceptance cases: (name, B, T, H, K, V)
_P0 = [
    ("aligned_b1_t128_h4_k64_v64",   1, 128, 4, 64, 64),
    ("ttail_b1_t100_h4_k64_v64",     1, 100, 4, 64, 64),
    ("t3c_b1_t192_h2_k128_v128",     1, 192, 2, 128, 128),
    ("tail_vlt_b1_t200_h3_k128_v96", 1, 200, 3, 128, 96),
]
_SEEDS = [42, 123, 456]


def _expected_dv_shape(B, T, H, K, V):
    N = (T + 63) // 64
    return (B, H, N, 64, V)


def _load_cross_check_refs():
    """Import the forward-WY residual source and the parent M6 staging oracle."""
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    custom_root = os.path.dirname(here)
    sys.path.insert(0, os.path.join(custom_root, "gdr_fwd_train_wy"))
    sys.path.insert(0, os.path.join(custom_root, "gated_delta_rule_bwd", "modules"))
    from gdr_fwd_train_wy_golden_cpu import gdr_fwd_train_wy_golden_cpu
    from gated_delta_rule_bwd_golden_stage123456 import gated_delta_rule_bwd_golden_stage123456
    return gdr_fwd_train_wy_golden_cpu, gated_delta_rule_bwd_golden_stage123456


def _mk_primary(B, T, H, K, V, seed, g_lo=-4.0, g_hi=0.0):
    """Primary GDR inputs (grad_o,q,k,v,g,beta) for the two-way cross-check."""
    torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(g_lo, g_hi)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return grad_o, q, k, v, g, beta


def _validate():
    print("=" * 68)
    print("gdr_bwd_dv_local_golden_cpu 验证报告 (M6/b1 mechanical projection)")
    print("=" * 68)
    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    # 1. Contract P0: run on direct residuals; shape / fp32 / finite.
    print("\n[1. 合同 P0 用例: shape / fp32 / finite]")
    for (name, B, T, H, K, V) in _P0:
        exp = _expected_dv_shape(B, T, H, K, V)
        for seed in _SEEDS:
            grad_o, k_f32, q_t, g_f32 = _mk_residuals(B, T, H, K, V, seed)
            dv = gdr_bwd_dv_local_golden_cpu(grad_o, k_f32, q_t, g_f32)
            ok = (tuple(dv.shape) == exp) and (dv.dtype == torch.float32) \
                and bool(torch.isfinite(dv).all())
            _check(f"{name} seed{seed}: shape {tuple(dv.shape)}=={exp} / fp32 / finite", ok)

    # 2. MANDATORY two-way cross-check (a mismatch is a DEFECT, not a tolerance question):
    #    (a) residuals from the REAL forward gdr_fwd_train_wy -> OP3 projection
    #    (b) parent M6 staging golden run from the SAME primary (M1..M6)
    #    assert dv_local agrees to fp32 tolerance on every P0 shape x seed.
    print("\n[2. 两路交叉验证 (OP3 投影 vs 父 M6 staging golden，同一 primary)]")
    fwd_wy, stage123456 = _load_cross_check_refs()
    CROSS_TOL = 1e-4  # fp32 mechanical-equivalence tolerance (same math, same order-ish)
    worst_overall = 0.0
    for (name, B, T, H, K, V) in _P0:
        for seed in _SEEDS:
            grad_o, q, k, v, g, beta = _mk_primary(B, T, H, K, V, seed)
            # (a) real residuals from forward WY-build; extract k_f32(1), g_f32(3), q_t(6)
            outs_wy = fwd_wy(q, k, v, g, beta)
            k_f32 = outs_wy[1]
            g_f32 = outs_wy[3]
            q_t = outs_wy[6]
            dv_op3 = gdr_bwd_dv_local_golden_cpu(grad_o, k_f32, q_t, g_f32)
            # (b) parent M6 slice from primary
            (dv_ref,) = stage123456(grad_o, q, k, v, g, beta)
            diff = (dv_op3 - dv_ref).abs()
            max_abs = diff.max().item()
            worst_overall = max(worst_overall, max_abs)
            _check(f"{name} seed{seed}: max|Δdv_local|={max_abs:.3e} (< {CROSS_TOL:.0e})",
                   torch.isfinite(dv_op3).all().item() and max_abs < CROSS_TOL)
    print(f"    worst max|Δdv_local| across all shapes/seeds = {worst_overall:.3e}")

    # 3. Numerical edges.
    print("\n[3. 数值边界]")
    grad_o, k_f32, q_t, g_f32 = _mk_residuals(1, 128, 2, 64, 64, seed=7)
    grad_o = torch.zeros_like(grad_o)
    dv = gdr_bwd_dv_local_golden_cpu(grad_o, k_f32, q_t, g_f32)
    _check("grad_o == 0 -> dv_local == 0 (linear in grad_o)", dv.abs().max().item() == 0.0)

    grad_o, k_f32, q_t, g_f32 = _mk_residuals(1, 128, 2, 64, 64, seed=3, g_lo=-16.0, g_hi=-8.0)
    dv = gdr_bwd_dv_local_golden_cpu(grad_o, k_f32, q_t, g_f32)
    _check("very-negative g (-16,-8): dv_local finite (exp bounded)",
           torch.isfinite(dv).all().item())

    # 4. Magnitude ranges (downstream precision-criterion evidence).
    print("\n[4. dv_local 量级范围 (供下游精度判据)]")
    for (name, B, T, H, K, V) in _P0:
        grad_o, q, k, v, g, beta = _mk_primary(B, T, H, K, V, seed=42)
        outs_wy = fwd_wy(q, k, v, g, beta)
        dv = gdr_bwd_dv_local_golden_cpu(grad_o, outs_wy[1], outs_wy[6], outs_wy[3])
        print(f"  {name}: |dv_local|max={dv.abs().max().item():.3e}")

    print("\n" + "=" * 68)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 68)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
