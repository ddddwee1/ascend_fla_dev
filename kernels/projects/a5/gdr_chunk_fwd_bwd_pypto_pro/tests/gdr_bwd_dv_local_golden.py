#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_dv_local golden reference implementation (NPU torch).

OP3 of gated_delta_rule_bwd = the intra-chunk (chunk-local) contribution to the dv
gradient (parent module M6, golden_step b1, BHN-parallel). MECHANICAL PROJECTION (a
faithful M6 slice), NOT fresh math: it computes exactly the M6 arithmetic of the
READ-ONLY delivered cumulative staging golden

    gated_delta_rule_bwd_golden_stage123456.py  (_m6)

but consumes OP3's frozen GM contract directly — grad_o (primary, public layout) plus the
three residuals (k_f32, q_t, g_f32) that gdr_fwd_train_wy M1 exports.

──────────────────────────────────────────────────────────────────────────────
PROJECTION MAP  (M6 `_m6`  ->  OP3 here)
  A_l      = matmul(k_f32, q_t)         # q_t IS q_s^T = [B,H,N,K,C] (pre-transposed upstream;
                                        #   scale=1/sqrt(K) already folded into q_s -> q_t),
                                        #   so this reproduces the parent's k @ q_s^T
  decay    = g_f32.cumsum(-1)           # g_f32 input = pre-cumsum log-space gate (padded)
  gdiff    = decay_j - decay_i;  A_l *= exp(gdiff)
  mask     strict-lower (i>j) -> 0      # keep UPPER triangle INCLUSIVE of diagonal (i<=j)
  do       = grad_o chunked [B,H,N,C,V] (transpose(1,2), pad T->N*C, reshape)
  dv_local = matmul(A_l, do)            # [C,C] @ [C,V] = [C,V]

Public signature:
    gdr_bwd_dv_local_golden(grad_o, k_f32, q_t, g_f32) -> dv_local
      grad_o : [B, T, H, V]     bf16   (upstream grad of o; PUBLIC layout, unpadded)
      k_f32  : [B, H, N, C, K]  fp32   (residual, internal per-chunk layout, N=(T+63)//64, C=64)
      q_t    : [B, H, N, K, C]  fp32   (residual; q_s^T pre-transposed, q_s = q*scale)
      g_f32  : [B, H, N, C]     fp32   (residual; PRE-cumsum log-space gate, padded rows g=0)
      dv_local : [B, H, N, C, V] fp32  (NOT narrowed; internal per-chunk layout)

  scale = 1/sqrt(K) already folded into q_t upstream; no q/k L2 norm; H == HV; C = 64.
  All arithmetic fp32 (only cast: grad_o bf16 -> fp32 on load). dv_local stays fp32.

torch_npu is required; computation runs on NPU. Falls back to CPU only when no NPU
hardware is available.
"""

import os
import torch
import torch.nn.functional as F

_DEVICE = None
_HAS_NPU = False

try:
    import torch_npu
    _HAS_NPU = torch.npu.is_available() and torch.npu.device_count() > 0
except ImportError:
    raise ImportError(
        "torch_npu is not installed. Please install it following the torch_npu official guide:\n"
        "  pip install torch_npu\n"
        "Refer to the CANN and PyPTO/PyPTO-Pro installation docs for the full NPU environment setup."
    )


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if not _HAS_NPU:
            _DEVICE = torch.device("cpu")
        else:
            device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
            torch.npu.set_device(device_id)
            _DEVICE = torch.device(f"npu:{device_id}")
    return _DEVICE


# ─────────────────────────────────────────────
# Canonical mathematical semantics.
# ─────────────────────────────────────────────
_CANONICAL_FORMULA = (
    "OP3 = module M6 (golden_step b1) of gated_delta_rule_bwd: intra-chunk dv gradient. "
    "A_l[i,j] = scale*(k_i . q_j)*exp(g_col_j - g_row_i) for j>=i (UPPER triangle INCLUSIVE "
    "of diagonal), else 0; dv_local = A_l @ grad_o_chunk. q_t = q_s^T pre-transposed "
    "(scale folded in q_s); g is cumulative decay = cumsum(g_f32) over the C=64 chunk tokens. "
    "Faithful M6 slice of gated_delta_rule_bwd_golden_stage123456 (_m6)."
)

_C = 64  # fixed internal chunk size


# ─────────────────────────────────────────────
# Public golden (M6 dv_local slice; NPU torch).
# ─────────────────────────────────────────────

def gdr_bwd_dv_local_golden(
    grad_o: torch.Tensor,
    k_f32: torch.Tensor,
    q_t: torch.Tensor,
    g_f32: torch.Tensor,
) -> torch.Tensor:
    """gdr_bwd_dv_local (OP3, M6 dv_local), NPU torch reference.

    Args:
        grad_o : [B, T, H, V]     bf16   (upstream grad of o; public layout)
        k_f32  : [B, H, N, C, K]  fp32
        q_t    : [B, H, N, K, C]  fp32   (q_s^T pre-transposed; scale folded in q_s)
        g_f32  : [B, H, N, C]     fp32   (pre-cumsum log-space gate; padded rows 0)

    Returns:
        dv_local : [B, H, N, C, V]  fp32
    """
    device = _get_device()
    C = _C

    grad_o_f = grad_o.to(device=device, dtype=torch.float32)
    k_f32 = k_f32.to(device=device, dtype=torch.float32)
    q_t = q_t.to(device=device, dtype=torch.float32)
    g_f32 = g_f32.to(device=device, dtype=torch.float32)

    B, T, H, V = grad_o_f.shape
    _, _, N, _, _ = k_f32.shape  # [B, H, N, C, K]

    # (1) cumulative decay over the C chunk tokens (g_f32 is PRE-cumsum).
    decay = g_f32.cumsum(-1)                                      # [B, H, N, C]

    # (2) Cube A: A_l = k @ q^T. q_t is already q_s^T ([B,H,N,K,C]); scale folded in q_s.
    A_l = torch.matmul(k_f32.float(), q_t.float())               # [B, H, N, C, C]

    # (3) Vector: intra-chunk gate-decay factor exp(g_col_j - g_row_i).
    gdiff = decay.unsqueeze(-2) - decay.unsqueeze(-1)            # [...,i,j] = decay_j - decay_i
    A_l = A_l * gdiff.exp()

    # (4) Vector: keep UPPER triangle INCLUSIVE of diagonal (i<=j); zero strict-lower (i>j).
    low_strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=A_l.device), diagonal=-1)
    A_l = A_l.masked_fill(low_strict, 0.0)

    # (5) grad_o -> chunked do [B,H,N,C,V].
    do = grad_o_f.transpose(1, 2).contiguous()                   # [B, H, T, V]
    pad = (C - (T % C)) % C
    if pad > 0:
        do = F.pad(do, (0, 0, 0, pad))
    do = do.reshape(B, H, N, C, V)                               # [B, H, N, C, V]

    # (6) Cube B: dv_local = A_l @ do  (fp32 output, NOT narrowed).
    dv_local = torch.matmul(A_l.float(), do.float())             # [B, H, N, C, V]
    return dv_local


# ==========================================
# 输入构造（供验证使用；启用性能采集时供 profiling --factory 复用）
# ==========================================

def _mk_residuals(device, B, T, H, K, V, seed=42, g_lo=-4.0, g_hi=0.0):
    """Build OP3's GM inputs [grad_o, k_f32, q_t, g_f32] at contract shapes/dtypes.

    grad_o bf16 (public); k_f32/q_t fp32 residuals; g_f32 fp32 pre-cumsum gate in a
    NEGATIVE range (log-space; keeps exp(decay diff) bounded, no overflow).
    """
    torch.manual_seed(seed)
    N = (T + 63) // 64
    C = _C
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)
    k_f32 = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
    q_t = torch.randn((B, H, N, K, C), dtype=torch.float32, device=device)
    g_f32 = torch.empty((B, H, N, C), dtype=torch.float32, device=device).uniform_(g_lo, g_hi)
    return [grad_o, k_f32, q_t, g_f32]


# Acceptance cases: (name, (B,T,H,K,V)).
_P0_CONFIGS = [
    ("aligned_b1_t128_h4_k64_v64",   (1, 128, 4, 64, 64)),
    ("ttail_b1_t100_h4_k64_v64",     (1, 100, 4, 64, 64)),
    ("t3c_b1_t192_h2_k128_v128",     (1, 192, 2, 128, 128)),
    ("tail_vlt_b1_t200_h3_k128_v96", (1, 200, 3, 128, 96)),
]


def _make_inputs(device):
    """Construct every contract P0 case for OP3.

    Returns [(case_name, [grad_o, k_f32, q_t, g_f32], {}), ...].
    """
    cases = []
    for name, (B, T, H, K, V) in _P0_CONFIGS:
        # Fixed seed per case so _validate and any profiling reuse the SAME inputs.
        args = _mk_residuals(device, B, T, H, K, V, seed=42)
        cases.append((name, args, {}))
    return cases


def _expected_dv_shape(B, T, H, K, V):
    N = (T + 63) // 64
    return (B, H, N, _C, V)


_P0_OUTPUT_SHAPES = {
    name: {"dv_local": _expected_dv_shape(B, T, H, K, V)}
    for name, (B, T, H, K, V) in _P0_CONFIGS
}


# ==========================================
# 验证
# ==========================================

def _validate():
    device = _get_device()
    print("=" * 68)
    print("gdr_bwd_dv_local_golden 验证报告 (M6/b1 mechanical projection)")
    print("=" * 68)
    print(f"Device: {device}")

    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        status = "PASS" if cond else "FAIL"
        if not cond:
            all_pass = False
        print(f"  [{status}] {name}")
        return cond

    # 1. Contract P0 cases: exact set, shape, dtype (fp32), finiteness.
    print("\n[1. 合同 P0 用例: shape / dtype / finite]")
    raw_cases = _make_inputs(device)
    assert isinstance(raw_cases, list) and raw_cases, "_make_inputs returned no cases"
    names = [c[0] for c in raw_cases]
    _check("no duplicate case names", len(names) == len(set(names)))
    expected = set(_P0_OUTPUT_SHAPES)
    _check(f"exact P0 set == {sorted(expected)}", set(names) == expected)

    for case_name, args, kwargs in raw_cases:
        dv = gdr_bwd_dv_local_golden(*args, **kwargs)
        exp = _P0_OUTPUT_SHAPES[case_name]["dv_local"]
        _check(f"{case_name}.dv_local: shape {tuple(dv.shape)} == {exp}", tuple(dv.shape) == exp)
        _check(f"{case_name}.dv_local: dtype {dv.dtype} == fp32", dv.dtype == torch.float32)
        _check(f"{case_name}.dv_local: finite", torch.isfinite(dv).all().item())

    # 2. NPU golden vs CPU fp32 golden agreement (per P0 shape, same inputs).
    print("\n[2. NPU vs CPU-fp32 golden 一致性]")
    _npu_vs_cpu_check(device, _check)

    # 3. MANDATORY two-way cross-check: OP3 (fed REAL forward residuals) vs parent M6
    #    staging golden (fed the SAME primary). A mismatch is a DEFECT, not a tolerance
    #    question. Inputs built on CPU (deterministic), residuals from the real forward.
    print("\n[3. 两路交叉验证 (OP3 投影 vs 父 M6 staging golden，同一 primary)]")
    _cross_check(device, _check)

    # 4. Numerical edge: all-zero grad_o -> dv_local == 0 (linear in grad_o).
    print("\n[4. 数值边界]")
    args = _mk_residuals(device, 1, 128, 2, 64, 64, seed=7)
    args[0] = torch.zeros_like(args[0])
    dv = gdr_bwd_dv_local_golden(*args)
    _check("grad_o == 0 -> dv_local == 0", dv.float().abs().max().item() == 0.0)
    args = _mk_residuals(device, 1, 128, 2, 64, 64, seed=3, g_lo=-16.0, g_hi=-8.0)
    dv = gdr_bwd_dv_local_golden(*args)
    _check("very-negative g (-16,-8): dv_local finite", torch.isfinite(dv).all().item())

    # 5. Magnitude ranges (downstream precision-criterion evidence).
    print("\n[5. dv_local 量级范围 (供下游精度判据)]")
    for case_name, args, kwargs in raw_cases:
        dv = gdr_bwd_dv_local_golden(*args, **kwargs)
        print(f"  {case_name}: |dv_local|max={dv.float().abs().max().item():.3e}")

    print("\n" + "=" * 68)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 68)
    if not all_pass:
        raise SystemExit(1)


def _npu_vs_cpu_check(device, _check):
    """Compare NPU golden against the CPU fp32 golden on identical inputs, per P0 shape."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gdr_bwd_dv_local_golden_cpu import gdr_bwd_dv_local_golden_cpu

    for name, (B, T, H, K, V) in _P0_CONFIGS:
        args_cpu = _mk_residuals(torch.device("cpu"), B, T, H, K, V, seed=456)
        dv_npu = gdr_bwd_dv_local_golden(*[a.clone() for a in args_cpu])
        dv_cpu = gdr_bwd_dv_local_golden_cpu(*[a.clone() for a in args_cpu])
        a = dv_npu.float().cpu()
        b = dv_cpu.float().cpu()
        diff = (a - b).abs()
        max_abs = diff.max().item()
        peak = b.abs().max().item()
        max_rel = (diff / b.abs().clamp_min(1e-3)).max().item()
        # fp32 both sides; NPU matmul roundoff only. peak-scaled OR relative.
        ok = torch.isfinite(a).all().item() and (max_abs <= 1e-2 * max(peak, 1.0) or max_rel <= 1e-2)
        _check(f"{name}: NPU vs CPU (abs={max_abs:.2e}, rel={max_rel:.2e}) within tol", ok)


def _cross_check(device, _check):
    """Two-way oracle: OP3(real forward residuals) vs parent M6 staging golden(primary)."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    custom_root = os.path.dirname(here)
    sys.path.insert(0, os.path.join(custom_root, "gdr_fwd_train_wy"))
    sys.path.insert(0, os.path.join(custom_root, "gated_delta_rule_bwd", "modules"))
    from gdr_fwd_train_wy_golden_cpu import gdr_fwd_train_wy_golden_cpu
    from gated_delta_rule_bwd_golden_stage123456 import gated_delta_rule_bwd_golden_stage123456

    def _mk_primary(B, T, H, K, V, seed, g_lo=-4.0, g_hi=0.0):
        torch.manual_seed(seed)
        grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
        q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
        k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
        v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
        g = torch.empty((B, T, H), dtype=torch.float32).uniform_(g_lo, g_hi)
        beta = torch.randn((B, T, H), dtype=torch.bfloat16)
        return grad_o, q, k, v, g, beta

    worst = 0.0
    for name, (B, T, H, K, V) in _P0_CONFIGS:
        for seed in (42, 123, 456):
            grad_o, q, k, v, g, beta = _mk_primary(B, T, H, K, V, seed)
            outs_wy = gdr_fwd_train_wy_golden_cpu(q, k, v, g, beta)
            k_f32, g_f32, q_t = outs_wy[1], outs_wy[3], outs_wy[6]  # forward WY residuals
            dv_op3 = gdr_bwd_dv_local_golden(grad_o, k_f32, q_t, g_f32).float().cpu()
            (dv_ref,) = gated_delta_rule_bwd_golden_stage123456(grad_o, q, k, v, g, beta)
            dv_ref = dv_ref.float().cpu()
            max_abs = (dv_op3 - dv_ref).abs().max().item()
            worst = max(worst, max_abs)
            # NPU fp32 matmul roundoff (vs CPU parent). Mechanical equivalence: must be tiny.
            _check(f"{name} seed{seed}: max|Δdv_local|={max_abs:.3e} (< 1e-2)",
                   torch.isfinite(dv_op3).all().item() and max_abs < 1e-2)
    print(f"    worst max|Δdv_local| across all shapes/seeds = {worst:.3e}")


if __name__ == "__main__":
    _validate()
