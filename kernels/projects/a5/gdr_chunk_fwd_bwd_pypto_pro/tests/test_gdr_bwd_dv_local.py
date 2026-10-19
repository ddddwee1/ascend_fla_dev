# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gdr_bwd_dv_local — acceptance tests against the float32 CPU golden.

The within-chunk contribution to dv. The kernel and its host wrapper live in
`../kernels/gdr_bwd_dv_local_impl.py`; this file holds only
the acceptance cases. Requires an Ascend a5 NPU (Ascend950PR); run it directly
to pick a subset by name, e.g. `python3 test_gdr_bwd_dv_local.py aligned`.
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

from gdr_bwd_dv_local_impl import (  # noqa: E402
    gdr_bwd_dv_local_wrapper,
    C,
)

# =============================================================================
#   Tests -- DESIGN section 8 target cases vs gdr_bwd_dv_local_golden_cpu
# =============================================================================
_SEEDS = [42, 123, 456]


def _mk_residuals(B, T, H, K, V, seed, device, g_lo=-4.0, g_hi=0.0):
    """Input generation IDENTICAL to gdr_bwd_dv_local_golden_cpu._mk_residuals
    (same dtypes, same shapes, same value ranges, same seeding order)."""
    torch.manual_seed(seed)
    N = (T + C - 1) // C
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    k_f32 = torch.randn((B, H, N, C, K), dtype=torch.float32)
    q_t = torch.randn((B, H, N, K, C), dtype=torch.float32)
    g_f32 = torch.empty((B, H, N, C), dtype=torch.float32).uniform_(g_lo, g_hi)
    return (grad_o.to(device), k_f32.to(device), q_t.to(device), g_f32.to(device))


def _assert_precision(dv_out, grad_o, k_f32, q_t, g_f32, label=""):
    """SPEC section 7 mixed acceptance criterion (inherited from the parent
    DESIGN section 9 _cpu_vs_npu_check family), applied to fp32 dv_local:

        per-element   |d| <= atol + rtol*|g|        atol = rtol = 1e-2
                                                   (SPEC machine-contract `tolerance`)
        OR global     ||d||_2 / max(||g||_2, eps) <= 2e-2

    Reference is the CPU fp32 staging golden -- dev-only import, kept inside the
    function body so the delivered module stays importable without it.
    """
    from gdr_bwd_dv_local_golden_cpu import gdr_bwd_dv_local_golden_cpu

    ref = gdr_bwd_dv_local_golden_cpu(
        grad_o.cpu(), k_f32.cpu(), q_t.cpu(), g_f32.cpu()).to(torch.float32)
    act = dv_out.cpu().to(torch.float32)

    assert act.shape == ref.shape, f"[{label}] shape {tuple(act.shape)} != {tuple(ref.shape)}"
    assert torch.isfinite(act).all(), f"[{label}] non-finite values in dv_local"

    atol = 1e-2
    rtol = 1e-2
    diff = (act - ref).abs()
    bound = atol + rtol * ref.abs()
    bad = (diff > bound)
    n_bad = int(bad.sum())
    peak = float(diff.max())

    rel_l2 = float(torch.linalg.vector_norm(act - ref) /
                   torch.clamp(torch.linalg.vector_norm(ref), min=1e-12))

    ok = (n_bad == 0) or (rel_l2 <= 2e-2)
    assert ok, (f"[{label}] PRECISION FAIL: {n_bad} elements exceed "
                f"atol+rtol*|g| (peak |d|={peak:.3e}), rel_L2={rel_l2:.3e} > 2e-2")
    logging.info(f"  [{label}] PASS  peak|d|={peak:.3e}  rel_L2={rel_l2:.3e}  "
                 f"outliers={n_bad}/{act.numel()}")


def _run_case(name, B, T, H, K, V):
    from gdr_bwd_dv_local_golden import _get_device
    device = _get_device()
    logging.info(f"[{name}] B={B} T={T} H={H} K={K} V={V}")
    for seed in _SEEDS:
        grad_o, k_f32, q_t, g_f32 = _mk_residuals(B, T, H, K, V, seed, device)
        dv = gdr_bwd_dv_local_wrapper(grad_o, k_f32, q_t, g_f32)
        _assert_precision(dv, grad_o, k_f32, q_t, g_f32, label=f"{name} seed{seed}")


def test_gdr_bwd_dv_local_module1_aligned():
    """DESIGN section 8 case 1 -- aligned_b1_t128_h4_k64_v64 (T=2C, K=V=64, no tail)."""
    _run_case("aligned_b1_t128_h4_k64_v64", 1, 128, 4, 64, 64)


def test_gdr_bwd_dv_local_module1_ttail():
    """DESIGN section 8 case 2 -- ttail_b1_t100_h4_k64_v64 (T tail, valid_c=36)."""
    _run_case("ttail_b1_t100_h4_k64_v64", 1, 100, 4, 64, 64)


def test_gdr_bwd_dv_local_module1_k128v128():
    """DESIGN section 8 case 3 -- t3c_b1_t192_h2_k128_v128 (K=V=128 upper bound)."""
    _run_case("t3c_b1_t192_h2_k128_v128", 1, 192, 2, 128, 128)


def test_gdr_bwd_dv_local_module1_tail_vlt():
    """DESIGN section 8 case 4 -- tail_vlt_b1_t200_h3_k128_v96 (N=4 chunks, valid_c=8, V<K)."""
    _run_case("tail_vlt_b1_t200_h3_k128_v96", 1, 200, 3, 128, 96)


if __name__ == "__main__":
    logging.info("gdr_bwd_dv_local Module 1 (cube->vec->cube fusion) -- staged self-test")
    logging.info("=" * 68)
    _all = [
        ("aligned", test_gdr_bwd_dv_local_module1_aligned),
        ("ttail", test_gdr_bwd_dv_local_module1_ttail),
        ("k128v128", test_gdr_bwd_dv_local_module1_k128v128),
        ("tail_vlt", test_gdr_bwd_dv_local_module1_tail_vlt),
    ]
    _sel = sys.argv[1:] if len(sys.argv) > 1 else None
    for _nm, _fn in _all:
        if _sel is None or _nm in _sel:
            _fn()
    logging.info("\nAll selected tests PASS!")
