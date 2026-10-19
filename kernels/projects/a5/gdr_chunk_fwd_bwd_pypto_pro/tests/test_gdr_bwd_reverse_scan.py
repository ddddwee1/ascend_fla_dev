# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gdr_bwd_reverse_scan — acceptance tests against the float32 CPU golden.

The backward chunk recurrence. The kernel and its host wrapper live in
`../kernels/gdr_bwd_reverse_scan_impl.py`; this file holds only
the acceptance cases. Requires an Ascend a5 NPU (Ascend950PR); run it directly
to pick a subset by name, e.g. `python3 test_gdr_bwd_reverse_scan.py aligned`.
"""

import logging
import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_KERNELS_DIR = os.path.join(os.path.dirname(_HERE), "kernels")
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

logging.basicConfig(level=logging.INFO, format="%(message)s")

from gdr_bwd_reverse_scan_impl import (  # noqa: E402
    gdr_bwd_reverse_scan_wrapper,
    C,
)

# =============================================================================
#   Tests — DESIGN §8 cases vs modules/gdr_bwd_reverse_scan_golden_cpu.py
# =============================================================================
def _mk(B, T, H, K, V, seed, g_lo=-4.0, g_hi=0.0, damp=1.0):
    """Residual generator — byte-identical to the stage1 golden's _mk_residuals
    (same torch.manual_seed order and the same distributions), then moved to the
    NPU.  All four DESIGN §8 cases use it with damp=1.0, i.e. the golden's own
    generator unmodified.

    `damp` scales ONLY w_t and k_f32, and is used ONLY by the added
    production-scale utilization shape.  Reason: the reverse recurrence is
    dS_i ~ -w_t_i @ (dv_local_i + (k_i @ dS_{i+1})*gate), so with i.i.d. N(0,1)
    residuals its per-chunk gain is ~sqrt(K*C) ~ 90.  At the §8 chunk counts
    (N<=4) that is harmless, but at N=32 it reaches ~1e62 and overflows fp32 —
    peak 5.5e32, ||ref||2 = inf, rel_l2 = NaN — which makes ANY comparison at
    that shape vacuous rather than passing.  damp=K**-0.5 puts the gain at ~0.7
    so the recurrence is contracting and the numbers stay finite and meaningful.
    This is a property of synthetic i.i.d. test data, not of the kernel: real
    w_t is a WY weight, not white noise.
    """
    from gdr_bwd_reverse_scan_golden import _get_device
    device = _get_device()
    N = (T + 63) // 64
    torch.manual_seed(seed)
    q_t = torch.randn((B, H, N, K, C), dtype=torch.float32)
    w_t = torch.randn((B, H, N, K, C), dtype=torch.float32) * damp
    k_f32 = torch.randn((B, H, N, C, K), dtype=torch.float32) * damp
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g_f32 = torch.empty((B, H, N, C), dtype=torch.float32).uniform_(g_lo, g_hi)
    decay_exp = g_f32.cumsum(-1).exp()
    dv_local = torch.randn((B, H, N, C, V), dtype=torch.float32)
    cpu = (q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local)
    npu = tuple(t.to(device) for t in cpu)
    return cpu, npu


def _judge(name, actual, ref, label):
    """SPEC §7 / DESIGN §9 frozen mixed criterion (identical to the one the
    Stage 3.5 simulator run was judged under):

        elementwise peak-scaled  |Δ| <= atol*peak + rtol*|ref|     (atol=rtol=1e-2)
        OR global relative L2    ||Δ||2 / max(||ref||2, eps) <= 2e-2

    The workflow engine's float32 row is evaluated first and reported; its FIXED
    1e-2 absolute cap is unreachable for these gradients (peak ~1e4-1e5, where a
    single fp32 ULP is already ~1e-2), which is precisely why SPEC §7 freezes the
    peak-scaled / rel-L2 pair as the acceptance criterion for this operator.
    """
    from precision_compare import check_precision
    a = actual.cpu().float()
    assert tuple(a.shape) == tuple(ref.shape), \
        f"[{label}] {name} shape {tuple(a.shape)} != {tuple(ref.shape)}"
    assert torch.isfinite(a).all(), f"[{label}] {name} not finite"
    # A reference that has overflowed fp32 gives a vacuous verdict (any error is
    # under 1e-2*peak, and rel_l2 is NaN).  Refuse to call that a PASS.
    assert torch.isfinite(ref).all(), \
        f"[{label}] {name} REFERENCE not finite — degenerate test data, no verdict"

    engine_pass, engine_summary = check_precision(a, ref, dtype_str="float32")
    diff = (a - ref).abs()
    peak = ref.abs().max().item()
    max_abs = diff.max().item()
    ref_n2 = torch.linalg.vector_norm(ref.double())
    rel_l2 = (torch.linalg.vector_norm((a - ref).double()) /
              torch.clamp(ref_n2, min=1e-12)).item()
    assert peak == peak and rel_l2 == rel_l2, \
        f"[{label}] {name} NaN in the criterion itself — no verdict"
    elem_ok = bool((diff <= 1e-2 * peak + 1e-2 * ref.abs()).all().item())
    spec_ok = elem_ok or rel_l2 <= 2e-2

    msg = (f"[{label}] {name}: peak={peak:.3e} max_abs={max_abs:.3e} "
           f"rel_l2={rel_l2:.3e} elementwise_peak_scaled={'OK' if elem_ok else 'no'} "
           f"| engine_float32_row={'PASS' if engine_pass else 'below-fixed-cap'} "
           f"({engine_summary})")
    if not spec_ok:
        raise AssertionError("MISMATCH " + msg)
    logging.info("  PASS " + msg)


def _assert_precision(dh, dv2, cpu_args, label=""):
    from gdr_bwd_reverse_scan_golden_cpu import gdr_bwd_reverse_scan_golden_cpu
    ref_dh, ref_dv2 = gdr_bwd_reverse_scan_golden_cpu(*cpu_args)
    _judge("dh", dh, ref_dh, label)
    _judge("dv2", dv2, ref_dv2, label)


def _run(B, T, H, K, V, seed, label, damp=1.0):
    cpu, npu = _mk(B, T, H, K, V, seed, damp=damp)
    dh, dv2 = gdr_bwd_reverse_scan_wrapper(*npu)
    _assert_precision(dh, dv2, cpu, label=label)


def test_gdr_bwd_reverse_scan_aligned():
    """§8 case 1 — fully divisible: T=2C, N=2, K=V=64, v_tiles=1, H=4>1."""
    _run(1, 128, 4, 64, 64, seed=42, label="aligned T128 H4 K64 V64")


def test_gdr_bwd_reverse_scan_ttail():
    """§8 case 2 — T tail: T=100, N=2, first reverse chunk valid_c=36, H=4>1."""
    _run(1, 100, 4, 64, 64, seed=123, label="ttail T100 H4 K64 V64")


def test_gdr_bwd_reverse_scan_t3c_kv128():
    """§8 case 3 — multi-chunk N=3 at the K=V=128 bound, v_tiles=2, H=2>1."""
    _run(1, 192, 2, 128, 128, seed=456, label="t3c T192 H2 K128 V128")


def test_gdr_bwd_reverse_scan_tail_vlt():
    """§8 case 4 — two-axis tail: T=200 (N=4, valid_c=8) + V=96<K=128
    (v_tiles=2, last valid_v=32), H=3>1."""
    _run(1, 200, 3, 128, 96, seed=42, label="tail_vlt T200 H3 K128 V96")


def test_gdr_bwd_reverse_scan_h8_seeds():
    """Extra: H=8 with all three frozen seeds — the widest H, which is the axis
    the order=[1,3] public-layout fix (X1) is load-bearing on."""
    for s in (42, 123, 456):
        _run(1, 128, 8, 64, 64, seed=s, label=f"h8 T128 H8 K64 V64 seed{s}")


def test_gdr_bwd_reverse_scan_production():
    """Utilization shape (orchestrator ruling): B*H=32 saturates all 32 cube
    cores, N=32 reverse chunks, v_tiles=2.  Correctness is re-checked here so
    the profiled shape is a verified one.  damp=K**-0.5 keeps the N=32 reverse
    recurrence numerically meaningful (see _mk); it does not change the kernel
    path exercised."""
    _run(4, 2048, 8, 128, 128, seed=7, damp=128.0 ** -0.5,
         label="production B4 T2048 H8 K128 V128")


if __name__ == "__main__":
    _sel = sys.argv[1:] or None
    _all = [
        ("aligned", test_gdr_bwd_reverse_scan_aligned),
        ("ttail", test_gdr_bwd_reverse_scan_ttail),
        ("t3c", test_gdr_bwd_reverse_scan_t3c_kv128),
        ("tail_vlt", test_gdr_bwd_reverse_scan_tail_vlt),
        ("h8", test_gdr_bwd_reverse_scan_h8_seeds),
        ("production", test_gdr_bwd_reverse_scan_production),
    ]
    logging.info("gdr_bwd_reverse_scan Module 1 (reverse dS scan) — staged self-test")
    logging.info("=" * 68)
    for _nm, _fn in _all:
        if _sel is None or _nm in _sel:
            logging.info(f"[{_nm}]")
            _fn()
    logging.info("\nAll selected tests PASS!")
