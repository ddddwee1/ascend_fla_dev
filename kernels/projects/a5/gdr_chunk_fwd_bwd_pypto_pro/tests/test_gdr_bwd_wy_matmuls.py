# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gdr_bwd_wy_matmuls — acceptance tests against the float32 CPU golden.

The matmul half of the WY adjoint. The kernel and its host wrapper live in
`../kernels/gdr_bwd_wy_matmuls_impl.py`; this file holds only
the acceptance cases. Requires an Ascend a5 NPU (Ascend950PR); run it directly
to pick a subset by name, e.g. `python3 test_gdr_bwd_wy_matmuls.py aligned`.
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

from gdr_bwd_wy_matmuls_impl import (  # noqa: E402
    gdr_bwd_wy_matmuls_wrapper,
    C,
)

# =============================================================================
#   Tests — DESIGN §8 target cases, compared vs gdr_bwd_wy_matmuls_golden_cpu
# =============================================================================
_OUT_NAMES = ["dkbg", "dvb", "dkb", "dkt", "d_a"]


def _assert_precision(outs, args, label=""):
    """Per-output precision gate via the workflow's shared precision engine.

    Thresholds are NOT chosen here: `precision_compare.check_precision` hard-codes the
    scheme-A mixed-tolerance table (fp32: rtol=2^-10, atol=2^-16,
    max_abs_error_limit = max(1e-2, 32*ULP), required matched_ratio >= 0.99) and refuses
    external tolerance parameters by design.
    """
    # dev-only imports kept INSIDE the function (delivery-time import safety)
    from gdr_bwd_wy_matmuls_golden_cpu import gdr_bwd_wy_matmuls_golden_cpu
    from precision_compare import check_precision

    ref = gdr_bwd_wy_matmuls_golden_cpu(*[a.cpu() for a in args])
    failures = []
    for nm, a, b in zip(_OUT_NAMES, outs, ref):
        a_cpu = a.cpu().float()
        b_cpu = b.float()
        assert tuple(a_cpu.shape) == tuple(b_cpu.shape), \
            f"[{label}] {nm} shape {tuple(a_cpu.shape)} != {tuple(b_cpu.shape)}"
        assert torch.isfinite(a_cpu).all(), f"[{label}] {nm} contains inf/nan"
        passed, summary = check_precision(a_cpu, b_cpu, dtype_str="float32")
        logging.info(f"[{label}] {nm}: {'PASS' if passed else 'FAIL'} {summary}")
        if not passed:
            failures.append(f"{nm}: {summary}")
    if failures:
        raise AssertionError(f"[{label}] precision FAIL -- " + " | ".join(failures))
    logging.info(f"[{label}] PASS (all {len(_OUT_NAMES)} outputs)")


def _mk(B, H, N, K, V, seed):
    """Input generator — identical distributions to the golden's own contract inputs.

    `_get_device()` calls `torch.npu.set_device(TILE_FWK_DEVICE_ID)`, which is mandatory
    on a non-zero card: PyPTO launches on the CURRENT torch NPU device, so without it
    every DMA would hit an unmapped address and trap with 507015/507035.
    """
    from gdr_bwd_wy_matmuls_golden import _get_device
    device = _get_device()
    torch.manual_seed(seed)
    attn = torch.randn((B, H, N, C, C), dtype=torch.float32, device=device)
    k_f32 = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
    v_b = torch.randn((B, H, N, C, V), dtype=torch.float32, device=device)
    kbeta_t = torch.randn((B, H, N, K, C), dtype=torch.float32, device=device)
    beta_act = torch.rand((B, H, N, C), dtype=torch.float32, device=device)
    g = torch.empty((B, H, N, C), dtype=torch.float32, device=device).uniform_(-4.0, 0.0)
    decay = g.cumsum(-1)
    dw = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
    dv2 = torch.randn((B, H, N, C, V), dtype=torch.float32, device=device)
    return [attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2]


def test_gdr_bwd_wy_matmuls_aligned_n2_k64_v64():
    # DESIGN §8 case 1: fully divisible, K=V=64, tasks = B*H*N = 8
    args = _mk(1, 4, 2, 64, 64, seed=42)
    _assert_precision(gdr_bwd_wy_matmuls_wrapper(*args), args, label="aligned_n2_k64_v64")


def test_gdr_bwd_wy_matmuls_ttail_n2_k64_v64():
    # DESIGN §8 case 2: T-tail invariance (chunk domain coincides with case 1)
    args = _mk(1, 4, 2, 64, 64, seed=123)
    _assert_precision(gdr_bwd_wy_matmuls_wrapper(*args), args, label="ttail_n2_k64_v64")


def test_gdr_bwd_wy_matmuls_k128_n3():
    # DESIGN §8 case 3: K=V=128 upper bound, 2 output-subblocks + 2 contraction steps
    args = _mk(1, 2, 3, 128, 128, seed=456)
    _assert_precision(gdr_bwd_wy_matmuls_wrapper(*args), args, label="k128_n3")


def test_gdr_bwd_wy_matmuls_vlt_n4_k128_v96():
    # DESIGN §8 case 4: V=96 < K=128 -> partial V output-subblock + V contraction tail
    args = _mk(1, 3, 4, 128, 96, seed=42)
    _assert_precision(gdr_bwd_wy_matmuls_wrapper(*args), args, label="vlt_n4_k128_v96")


def test_gdr_bwd_wy_matmuls_zerocore_h3_n11():
    # tasks = 33 over 32 cores -> cores 17..31 run ZERO iterations; exercises every
    # cross-core credit balance at n=0.
    args = _mk(1, 3, 11, 64, 64, seed=99)
    _assert_precision(gdr_bwd_wy_matmuls_wrapper(*args), args, label="zerocore_h3_n11_tasks33")


def test_gdr_bwd_wy_matmuls_production():
    # Production / utilization shape: B=1, T=2048, H=8, K=V=128 -> 256 tasks
    args = _mk(1, 8, 32, 128, 128, seed=7)
    _assert_precision(gdr_bwd_wy_matmuls_wrapper(*args), args,
                      label="production_b1_h8_n32_k128_v128")


if __name__ == "__main__":
    logging.info("gdr_bwd_wy_matmuls (OP6a) — final deliverable self-test")
    logging.info("=" * 64)
    test_gdr_bwd_wy_matmuls_aligned_n2_k64_v64()
    test_gdr_bwd_wy_matmuls_ttail_n2_k64_v64()
    test_gdr_bwd_wy_matmuls_k128_n3()
    test_gdr_bwd_wy_matmuls_vlt_n4_k128_v96()
    test_gdr_bwd_wy_matmuls_zerocore_h3_n11()
    test_gdr_bwd_wy_matmuls_production()
    logging.info("\nAll tests PASS!")
