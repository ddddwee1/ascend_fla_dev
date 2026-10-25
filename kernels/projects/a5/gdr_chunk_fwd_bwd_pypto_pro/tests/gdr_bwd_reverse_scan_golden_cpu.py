#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_reverse_scan (OP4) CPU golden reference (higher precision, FP32).

Ultimate ground truth for precision comparison in later stages. Same math as the NPU golden
(gdr_bwd_reverse_scan_golden.py) but pure CPU, no torch_npu / no device selection; all
arithmetic and outputs FP32.

MECHANICAL PROJECTION (a faithful M7 slice), NOT fresh math: it computes exactly the M7
arithmetic (golden_step b2, the reverse inter-chunk dS scan) of the READ-ONLY delivered
cumulative staging golden

    gated_delta_rule_bwd_golden_stage1234567.py  (_m7)

but consumes OP4's frozen GM contract directly.

──────────────────────────────────────────────────────────────────────────────
PROJECTION MAP  (M7 `_m7` in the parent staging golden  ->  OP4 here)
  decay    = g_f32.cumsum(-1)               # g_f32 is PRE-cumsum gate (parent M2 cumsum)
  g_last   = decay[..., -1]
  q_de^T   = q_t * decay_exp[..., None, :]   # q_de = q_s*decay_exp; q_t = q_s^T; term_q = q_de^T@do
  w^T      = w_t                             # pre-transposed upstream
  do       = grad_o chunked [B,H,N,C,V]
  reverse chunk scan i=N-1..0 carrying dS[K,V] (init 0):
    dh[.,i] = dS ; dvc = k_f32_i @ dS ; dv2_i = dv_local_i + dvc*exp(g_last_i - decay_i)
    dS = exp(g_last_i)*dS + (q_de^T_i @ do_i) - (w^T_i @ dv2_i)

WHERE THE SCALE IS APPLIED:
  scale=1/sqrt(K) is folded into q_s (q_s = scale*q) UPSTREAM in gdr_fwd_train_wy, so q_t
  and the reconstructed q_de^T already carry scale. The authoritative parent _m7 applies NO
  explicit scale multiply to term_q. OP4 therefore uses q_t directly with NO extra scale
  multiply; the probe stub's whole-delta `scale*delta` is WRONG (double-scale).

Signature:  gdr_bwd_reverse_scan_golden_cpu(q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local)
              -> (dh, dv2)
    q_t       : [B, H, N, K, C]  fp32   q_s^T PRE-TRANSPOSED (scale folded into q_s)
    w_t       : [B, H, N, K, C]  fp32   w^T PRE-TRANSPOSED
    k_f32     : [B, H, N, C, K]  fp32
    grad_o    : [B, T, H, V]     (any float dtype; promoted to fp32; PUBLIC layout)
    decay_exp : [B, H, N, C]     fp32   exp(cumsum(g_f32))
    g_f32     : [B, H, N, C]     fp32   PRE-cumsum log-space gate (padded rows 0)
    dv_local  : [B, H, N, C, V]  fp32
    ->  dh  : [B, H, N, K, V]  fp32   (NOT narrowed)
        dv2 : [B, H, N, C, V]  fp32   (NOT narrowed)

All arithmetic fp32. Padding rows carry g=0 (inert). No [:T] slice here (that is OP7's b7).
"""

import torch
import torch.nn.functional as F

_C = 64  # fixed internal chunk size


def gdr_bwd_reverse_scan_golden_cpu(
    q_t: torch.Tensor,
    w_t: torch.Tensor,
    k_f32: torch.Tensor,
    grad_o: torch.Tensor,
    decay_exp: torch.Tensor,
    g_f32: torch.Tensor,
    dv_local: torch.Tensor,
):
    """CPU FP32 golden for OP4 (M7 reverse dS scan, faithful slice of the parent b2 step)."""
    C = _C
    q_t = q_t.to(torch.float32)
    w_t = w_t.to(torch.float32)
    k_f32 = k_f32.to(torch.float32)
    grad_o = grad_o.to(torch.float32)
    decay_exp = decay_exp.to(torch.float32)
    g_f32 = g_f32.to(torch.float32)
    dv_local = dv_local.to(torch.float32)

    B, T, H, V = grad_o.shape
    _, _, N, K, _ = q_t.shape  # [B, H, N, K, C]

    # grad_o (public [B,T,H,V]) -> chunked do [B,H,N,C,V] (pad T -> N*C).
    do = grad_o.transpose(1, 2).contiguous()                      # [B, H, T, V]
    pad = (C - (T % C)) % C
    if pad > 0:
        do = F.pad(do, (0, 0, 0, pad))
    do = do.reshape(B, H, N, C, V)                                # [B, H, N, C, V]

    # cumulative decay (g_f32 is PRE-cumsum) and last-token decay per chunk.
    decay = g_f32.cumsum(-1)                                      # [B, H, N, C]
    g_last = decay[..., -1]                                       # [B, H, N]

    # q_de^T = (q_s * decay_exp)^T = q_t * decay_exp (broadcast over K); reconstructs parent term_q.
    q_deT = q_t * decay_exp[..., None, :]                         # [B, H, N, K, C]

    dS = torch.zeros(B, H, K, V, dtype=torch.float32)
    dh = torch.zeros(B, H, N, K, V, dtype=torch.float32)
    dv2 = torch.zeros(B, H, N, C, V, dtype=torch.float32)

    for i in range(N - 1, -1, -1):
        dh[:, :, i] = dS                                         # entering dS snapshot
        dvc = torch.matmul(k_f32[:, :, i].float(), dS.float())  # [C,K]@[K,V]=[C,V]
        gate = (g_last[:, :, i, None] - decay[:, :, i]).exp()    # exp(g_last-decay) [B,H,C]
        dv2_i = dv_local[:, :, i] + dvc * gate[..., None]        # [B,H,C,V]
        dv2[:, :, i] = dv2_i
        term_q = torch.matmul(q_deT[:, :, i].float(), do[:, :, i].float())  # [K,C]@[C,V]=[K,V]
        term_w = torch.matmul(w_t[:, :, i].float(), dv2_i.float())          # [K,C]@[C,V]=[K,V]
        # exp(g_last_i) == decay_exp[...,-1]; loop-carried dS update.
        dS = dS * decay_exp[:, :, i, -1, None, None] + term_q - term_w

    return dh, dv2


# ==========================================
# 验证
# ==========================================

_P0 = [
    ("aligned_b1_t128_h4_k64_v64",   1, 128, 4, 64, 64),
    ("ttail_b1_t100_h4_k64_v64",     1, 100, 4, 64, 64),
    ("t3c_b1_t192_h2_k128_v128",     1, 192, 2, 128, 128),
    ("tail_vlt_b1_t200_h3_k128_v96", 1, 200, 3, 128, 96),
]
_SEEDS = [42, 123, 456]


def _expected_shapes(B, T, H, K, V):
    N = (T + 63) // 64
    return {"dh": (B, H, N, K, V), "dv2": (B, H, N, 64, V)}


def _mk_residuals(B, T, H, K, V, seed, g_lo=-4.0, g_hi=0.0):
    """Build OP4's GM inputs directly at contract shapes/dtypes; decay_exp self-consistent."""
    torch.manual_seed(seed)
    N = (T + 63) // 64
    C = _C
    q_t = torch.randn((B, H, N, K, C), dtype=torch.float32)
    w_t = torch.randn((B, H, N, K, C), dtype=torch.float32)
    k_f32 = torch.randn((B, H, N, C, K), dtype=torch.float32)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g_f32 = torch.empty((B, H, N, C), dtype=torch.float32).uniform_(g_lo, g_hi)
    decay_exp = g_f32.cumsum(-1).exp()
    dv_local = torch.randn((B, H, N, C, V), dtype=torch.float32)
    return q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local


def _load_cross_check_refs():
    """Import forward-WY residual source, OP3 dv_local, and the parent M7 staging oracle."""
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    custom_root = os.path.dirname(here)
    sys.path.insert(0, os.path.join(custom_root, "gdr_fwd_train_wy"))
    sys.path.insert(0, os.path.join(custom_root, "gdr_bwd_dv_local"))
    sys.path.insert(0, os.path.join(custom_root, "gated_delta_rule_bwd", "modules"))
    from gdr_fwd_train_wy_golden_cpu import gdr_fwd_train_wy_golden_cpu
    from gdr_bwd_dv_local_golden_cpu import gdr_bwd_dv_local_golden_cpu
    from gated_delta_rule_bwd_golden_stage1234567 import gated_delta_rule_bwd_golden_stage1234567
    return (gdr_fwd_train_wy_golden_cpu, gdr_bwd_dv_local_golden_cpu,
            gated_delta_rule_bwd_golden_stage1234567)


def _mk_primary(B, T, H, K, V, seed, g_lo=-4.0, g_hi=0.0):
    torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(g_lo, g_hi)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return grad_o, q, k, v, g, beta


def _build_op4_residuals(fwd_wy, dv_local_op3, grad_o, q, k, v, g, beta):
    outs_wy = fwd_wy(q, k, v, g, beta)
    k_f32 = outs_wy[1]
    g_f32 = outs_wy[3]
    q_t = outs_wy[6]
    decay_exp = outs_wy[8]
    w_t = outs_wy[16]
    dv_local = dv_local_op3(grad_o, k_f32, q_t, g_f32)
    return q_t, w_t, k_f32, decay_exp, g_f32, dv_local


def _validate():
    print("=" * 68)
    print("gdr_bwd_reverse_scan_golden_cpu 验证报告 (M7/b2 mechanical projection)")
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
        exp = _expected_shapes(B, T, H, K, V)
        for seed in _SEEDS:
            args = _mk_residuals(B, T, H, K, V, seed)
            dh, dv2 = gdr_bwd_reverse_scan_golden_cpu(*args)
            ok = (tuple(dh.shape) == exp["dh"] and tuple(dv2.shape) == exp["dv2"]
                  and dh.dtype == torch.float32 and dv2.dtype == torch.float32
                  and torch.isfinite(dh).all().item() and torch.isfinite(dv2).all().item())
            _check(f"{name} seed{seed}: dh{tuple(dh.shape)}=={exp['dh']} "
                   f"dv2{tuple(dv2.shape)}=={exp['dv2']} / fp32 / finite", ok)

    # 2. MANDATORY three-operator cross-check (a mismatch is a GM-CONTRACT DEFECT):
    #    residuals from REAL forward gdr_fwd_train_wy + REAL gdr_bwd_dv_local -> OP4 projection,
    #    compared to the parent M7 staging golden (stage1234567) from the SAME primary.
    print("\n[2. 三算子交叉验证 (OP4 投影 vs 父 M7 staging golden gated_delta_rule_bwd_golden_stage1234567)]")
    fwd_wy, dv_local_op3, stage1234567 = _load_cross_check_refs()
    CROSS_TOL = 1e-3  # fp32 mechanical-equivalence (long reverse-scan accumulation)
    worst_dh = 0.0
    worst_dv2 = 0.0
    for (name, B, T, H, K, V) in _P0:
        for seed in _SEEDS:
            grad_o, q, k, v, g, beta = _mk_primary(B, T, H, K, V, seed)
            q_t, w_t, k_f32, decay_exp, g_f32, dv_local = _build_op4_residuals(
                fwd_wy, dv_local_op3, grad_o, q, k, v, g, beta)
            dh_op4, dv2_op4 = gdr_bwd_reverse_scan_golden_cpu(
                q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local)
            dh_ref, dv2_ref = stage1234567(grad_o, q, k, v, g, beta)
            d_dh = (dh_op4 - dh_ref).abs().max().item()
            d_dv2 = (dv2_op4 - dv2_ref).abs().max().item()
            worst_dh = max(worst_dh, d_dh)
            worst_dv2 = max(worst_dv2, d_dv2)
            ok = (torch.isfinite(dh_op4).all().item() and torch.isfinite(dv2_op4).all().item()
                  and d_dh < CROSS_TOL and d_dv2 < CROSS_TOL)
            _check(f"{name} seed{seed}: max|Δdh|={d_dh:.3e} max|Δdv2|={d_dv2:.3e} (< {CROSS_TOL:.0e})",
                   ok)
    print(f"    worst across all shapes/seeds: max|Δdh|={worst_dh:.3e}  max|Δdv2|={worst_dv2:.3e}")

    # 3. Numerical edges.
    print("\n[3. 数值边界]")
    # first reverse chunk (i=N-1): dS=0 -> dh[N-1]=0 exactly, dvc=0 -> dv2[N-1]=dv_local[N-1].
    args = _mk_residuals(1, 128, 4, 64, 64, seed=42)
    dh0, dv20 = gdr_bwd_reverse_scan_golden_cpu(*args)
    N0 = dh0.shape[2]
    _check("first reverse chunk dh[:, :, N-1] == 0 (dS init 0)",
           dh0[:, :, N0 - 1].abs().max().item() == 0.0)
    dvloc0 = args[6]
    _check("first reverse chunk dv2[:, :, N-1] == dv_local[:, :, N-1] (dvc=0)",
           (dv20[:, :, N0 - 1] - dvloc0[:, :, N0 - 1]).abs().max().item() == 0.0)
    # grad_o == 0 through the real forward+OP3 chain -> dh, dv2 == 0 end-to-end.
    grad_o, q, k, v, g, beta = _mk_primary(1, 128, 2, 64, 64, seed=7)
    grad_o = torch.zeros_like(grad_o)
    q_t, w_t, k_f32, decay_exp, g_f32, dv_local = _build_op4_residuals(
        fwd_wy, dv_local_op3, grad_o, q, k, v, g, beta)
    dh_z, dv2_z = gdr_bwd_reverse_scan_golden_cpu(q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local)
    _check("grad_o == 0 (end-to-end) -> dh == 0 and dv2 == 0",
           dh_z.abs().max().item() == 0.0 and dv2_z.abs().max().item() == 0.0)
    # very-negative g: exp bounded, finite.
    grad_o, q, k, v, g, beta = _mk_primary(1, 128, 2, 64, 64, seed=3, g_lo=-16.0, g_hi=-8.0)
    q_t, w_t, k_f32, decay_exp, g_f32, dv_local = _build_op4_residuals(
        fwd_wy, dv_local_op3, grad_o, q, k, v, g, beta)
    dh_n, dv2_n = gdr_bwd_reverse_scan_golden_cpu(q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local)
    _check("very-negative g (-16,-8): dh, dv2 finite (exp bounded)",
           torch.isfinite(dh_n).all().item() and torch.isfinite(dv2_n).all().item())

    # 4. Magnitude ranges (downstream precision-criterion evidence).
    print("\n[4. dh / dv2 量级范围 (供下游精度判据)]")
    for (name, B, T, H, K, V) in _P0:
        grad_o, q, k, v, g, beta = _mk_primary(B, T, H, K, V, seed=42)
        q_t, w_t, k_f32, decay_exp, g_f32, dv_local = _build_op4_residuals(
            fwd_wy, dv_local_op3, grad_o, q, k, v, g, beta)
        dh_m, dv2_m = gdr_bwd_reverse_scan_golden_cpu(
            q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local)
        print(f"  {name}: |dh|max={dh_m.abs().max().item():.3e}  |dv2|max={dv2_m.abs().max().item():.3e}")

    print("\n" + "=" * 68)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 68)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
