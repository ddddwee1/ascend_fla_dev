# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gdr_fwd_train_wy — acceptance tests against the float32 CPU golden.

Within-chunk WY transform and the residuals the backward needs. The kernel and its host wrapper live in
`../kernels/gdr_fwd_train_wy_impl.py`; this file holds only
the acceptance cases. Requires an Ascend a5 NPU (Ascend950PR); run it directly
to pick a subset by name, e.g. `python3 test_gdr_fwd_train_wy.py aligned`.
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

from gdr_fwd_train_wy_impl import (  # noqa: E402
    gdr_fwd_train_wy_wrapper,
)

# =============================================================================
#   Tests — DESIGN §8 target cases + production, vs gdr_fwd_train_wy_golden_cpu
# =============================================================================
# module_interfaces.yaml `final_outputs` == gdr_fwd_train_wy_golden_cpu's order.
_OUT_NAMES = ["q_s", "k_f32", "v_b", "g_f32", "beta_act", "sig", "q_t", "decay",
              "decay_exp", "dfac", "q_de", "attn", "a_mat", "kbeta_t",
              "u", "w", "w_t"]

# Precision is judged by the workflow's own engine, `precision_compare.py`
# (shipped beside these tests),
# NOT by a bespoke allclose. Its fixed float32 policy is
#   elementwise |a-g| <= atol + rtol*|g|   with rtol=2^-10, atol=2^-16
#   overall     matched_ratio >= 0.99  AND  max_abs_error <= max(1e-2, 32*ULP)
# so the thresholds come from the engine, not from this file. All 19 outputs
# are fp32, so the float32 row applies with no `dtype_str` override.


def _assert_precision(outs, q, k, v, g, beta, label=""):
    """Compare all 17 public outputs against the CPU FP32 golden.

    dev-only imports stay INSIDE the function so the delivered module can be
    imported with only the kernel + wrapper present.
    """
    from precision_compare import check_precision
    from gdr_fwd_train_wy_golden_cpu import gdr_fwd_train_wy_golden_cpu
    ref = gdr_fwd_train_wy_golden_cpu(q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu())
    assert len(outs) == len(ref) == 17,         f"[{label}] expected 17 outputs, got {len(outs)} vs golden {len(ref)}"

    failures = []
    for nm, a, b in zip(_OUT_NAMES, outs, ref):
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        assert tuple(a_cpu.shape) == tuple(b_cpu.shape),             f"[{label}] {nm} shape {tuple(a_cpu.shape)} != {tuple(b_cpu.shape)}"
        passed, summary = check_precision(a_cpu, b_cpu)
        logging.info(f"  [{label}] {nm:10s} {'PASS' if passed else 'FAIL'}  {summary}")
        if not passed:
            failures.append(f"{nm}: {summary}")
    if failures:
        raise AssertionError(f"[{label}] precision FAIL -> " + " | ".join(failures))
    logging.info(f"[{label}] PASS (all 17 outputs, precision_compare fp32 policy)")


def _mk(B, T, H, K, V, seed):
    """Input generator mirroring gdr_fwd_train_wy_golden_cpu._mk_case."""
    from gdr_fwd_train_wy_golden import _get_device
    device = _get_device()
    if device.type == "npu":
        torch.npu.set_device(device)
    torch.manual_seed(seed)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)
    g = torch.empty((B, T, H), dtype=torch.float32, device=device).uniform_(-16.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device=device)
    return q, k, v, g, beta


def test_gdr_fwd_train_wy_aligned():
    """[1,128,4,64,64] — fully divisible: T=2C, K=V=64=SUB, single sub-block."""
    q, k, v, g, beta = _mk(1, 128, 4, 64, 64, seed=42)
    outs = gdr_fwd_train_wy_wrapper(q, k, v, g, beta)
    _assert_precision(outs, q, k, v, g, beta, label="aligned [1,128,4,64,64]")


def test_gdr_fwd_train_wy_ttail():
    """[1,100,4,64,64] — T single-axis tail: last chunk 36 valid rows, 28 padded."""
    q, k, v, g, beta = _mk(1, 100, 4, 64, 64, seed=123)
    outs = gdr_fwd_train_wy_wrapper(q, k, v, g, beta)
    _assert_precision(outs, q, k, v, g, beta, label="ttail [1,100,4,64,64]")


def test_gdr_fwd_train_wy_t3c_kv128():
    """[1,192,2,128,128] — 3 chunks + K/V=128: 2 sub-blocks, multi-tile loops."""
    q, k, v, g, beta = _mk(1, 192, 2, 128, 128, seed=456)
    outs = gdr_fwd_train_wy_wrapper(q, k, v, g, beta)
    _assert_precision(outs, q, k, v, g, beta, label="t3c_kv128 [1,192,2,128,128]")


def test_gdr_fwd_train_wy_tailv96():
    """[1,200,3,128,96] — dual-axis tail: T tail (8 rows) + partial V (96 -> 32)."""
    q, k, v, g, beta = _mk(1, 200, 3, 128, 96, seed=42)
    outs = gdr_fwd_train_wy_wrapper(q, k, v, g, beta)
    _assert_precision(outs, q, k, v, g, beta, label="tailv96 [1,200,3,128,96]")


def test_gdr_fwd_train_wy_production():
    """[1,2048,8,128,128] — production shape, BEYOND DESIGN §8 (which is a floor).

    B*H*N = 1*8*32 = 256 work items, so this is the shape that actually
    saturates the machine: block_dim clamps to core_num=32, giving 8 tids per
    block across 32 AIC + 64 AIV. Added so correctness and the utilization
    profile are measured at the same shape."""
    q, k, v, g, beta = _mk(1, 2048, 8, 128, 128, seed=7)
    outs = gdr_fwd_train_wy_wrapper(q, k, v, g, beta)
    _assert_precision(outs, q, k, v, g, beta, label="production [1,2048,8,128,128]")


if __name__ == "__main__":
    logging.info("gdr_fwd_train_wy — delivered operator self-test (Modules 1-4 fused)")
    logging.info("=" * 68)
    test_gdr_fwd_train_wy_aligned()
    test_gdr_fwd_train_wy_ttail()
    test_gdr_fwd_train_wy_t3c_kv128()
    test_gdr_fwd_train_wy_tailv96()
    test_gdr_fwd_train_wy_production()
    logging.info("=" * 68)
    logging.info("All tests PASS!")
