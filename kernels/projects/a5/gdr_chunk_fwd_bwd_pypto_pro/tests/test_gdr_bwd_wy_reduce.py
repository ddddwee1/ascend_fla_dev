# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gdr_bwd_wy_reduce — acceptance tests against the float32 CPU golden.

The reduction half of the WY adjoint. The kernel and its host wrapper live in
`../kernels/gdr_bwd_wy_reduce_impl.py`; this file holds only
the acceptance cases. Requires an Ascend a5 NPU (Ascend950PR); run it directly
to pick a subset by name, e.g. `python3 test_gdr_bwd_wy_reduce.py aligned`.
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

from gdr_bwd_wy_reduce_impl import (  # noqa: E402
    gdr_bwd_wy_reduce_wrapper,
    C,
)

# =============================================================================
#   Tests — compare (dk,dv,dbeta_wy,dg2) vs gdr_bwd_wy_reduce_golden_cpu.
# =============================================================================
_OUT_NAMES = ("dk", "dv", "dbeta_wy", "dg2")


def _assert_precision(outs, cpu_args, label=""):
    """DESIGN §9 precision judgement for the 4 fp32 outputs vs the CPU FP32 golden.

    Uses the workflow-fixed `precision_compare.check_precision` (float32 row:
    rtol=2^-10, atol=2^-16, max_abs_error_limit=max(1e-2, 32*ULP)) -- no custom
    thresholds. dk/dv are algebraically exact (elementwise-only paths); dbeta_wy/
    dg2 carry only fp32 reduction-order noise (~1e-5 abs at ~1e4 magnitude per
    kal/NOTES.md sim measurements), comfortably inside the float32 row.
    """
    from precision_compare import check_precision
    from gdr_bwd_wy_reduce_golden_cpu import gdr_bwd_wy_reduce_golden_cpu

    ref = gdr_bwd_wy_reduce_golden_cpu(*[a.cpu() for a in cpu_args])
    actual = tuple(o.cpu().float() for o in outs)
    for nm, a, r in zip(_OUT_NAMES, actual, ref):
        assert tuple(a.shape) == tuple(r.shape), \
            f"[{label}] {nm} shape {tuple(a.shape)} != {tuple(r.shape)}"
        assert torch.isfinite(a).all(), f"[{label}] {nm} not finite"
    passed, summary = check_precision(actual, ref, dtype_str="float32")
    if passed:
        logging.info(f"[{label}] PASS ({summary})")
    else:
        raise AssertionError(f"[{label}] precision mismatch: {summary}")


def _mk(B, H, N, K, V, T, seed):
    """Build OP6b's 9 contract inputs directly at contract shapes/dtypes (DESIGN §8),
    mirroring `gdr_bwd_wy_reduce_golden_cpu._mk_contract_inputs` exactly, moved to
    the test device."""
    from gdr_bwd_wy_reduce_golden import _get_device
    device = _get_device()
    torch.manual_seed(seed)
    dkbg = torch.randn((B, H, N, C, K), dtype=torch.float32)
    dvb = torch.randn((B, H, N, C, V), dtype=torch.float32)
    dkb = torch.randn((B, H, N, C, K), dtype=torch.float32)
    dkt = torch.randn((B, H, N, C, K), dtype=torch.float32)
    k_f32 = torch.empty((B, H, N, C, K), dtype=torch.float32).uniform_(-10.0, 10.0)
    beta_act = torch.rand((B, H, N, C), dtype=torch.float32)
    g = torch.empty((B, H, N, C), dtype=torch.float32).uniform_(-4.0, 0.0)
    decay_exp = g.cumsum(-1).exp()
    dk_p = torch.randn((B, H, N, C, K), dtype=torch.float32)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    cpu_args = [dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v]
    dev_args = [a.to(device) for a in cpu_args]
    return dev_args, cpu_args


def test_gdr_bwd_wy_reduce_aligned():
    """aligned_t128_h4_k64_v64 (DESIGN §8): full-divide, T=128=2*C, K=V=64 single register."""
    dev_args, cpu_args = _mk(1, 4, 2, 64, 64, 128, seed=42)
    outs = gdr_bwd_wy_reduce_wrapper(*dev_args)
    _assert_precision(outs, cpu_args, label="aligned_t128_h4_k64_v64")


def test_gdr_bwd_wy_reduce_ttail():
    """ttail_t100_h4_k64_v64 (DESIGN §8): primary-v T-tail, T=100, last chunk valid_rows=36."""
    dev_args, cpu_args = _mk(1, 4, 2, 64, 64, 100, seed=123)
    outs = gdr_bwd_wy_reduce_wrapper(*dev_args)
    _assert_precision(outs, cpu_args, label="ttail_t100_h4_k64_v64")


def test_gdr_bwd_wy_reduce_t3c():
    """t3c_t192_h2_k128_v128 (DESIGN §8): K=V=128 crosses 2 fp32 registers (rule 15), N=3."""
    dev_args, cpu_args = _mk(1, 2, 3, 128, 128, 192, seed=456)
    outs = gdr_bwd_wy_reduce_wrapper(*dev_args)
    _assert_precision(outs, cpu_args, label="t3c_t192_h2_k128_v128")


def test_gdr_bwd_wy_reduce_tail_vlt():
    """tail_vlt_t200_h3_k128_v96 (DESIGN §8): V=96<K=128 partial 2nd register + T=200 tail."""
    dev_args, cpu_args = _mk(1, 3, 4, 128, 96, 200, seed=42)
    outs = gdr_bwd_wy_reduce_wrapper(*dev_args)
    _assert_precision(outs, cpu_args, label="tail_vlt_t200_h3_k128_v96")


def test_gdr_bwd_wy_reduce_b2():
    """A batch greater than one: B=2, H=2, T=128, K=V=64.

    The four cases above are all B=1, so this is the case that exercises this
    operator's batch axis. It passes; the composed pipeline is still limited to
    B=1 by `gdr_bwd_dqkw`, which pins that axis in its tensor annotations.
    """
    dev_args, cpu_args = _mk(2, 2, 2, 64, 64, 128, seed=42)
    outs = gdr_bwd_wy_reduce_wrapper(*dev_args)
    _assert_precision(outs, cpu_args, label="b2_h2_t128_k64_v64")


def test_gdr_bwd_wy_reduce_prod_util():
    """NOT a DESIGN §8 case -- a utilization-profiling shape (orchestrator
    ruling): B=1,H=8,T=2048,K=V=128 -> N=32, B*H*N=256 tasks (>=256, so the
    BHN-parallel grid saturates the 64 vector cores instead of being
    launch-bound like the four DESIGN §8 shapes, which only give 6-12 tasks).
    Used for msprof utilization profiling; also checked for correctness."""
    dev_args, cpu_args = _mk(1, 8, 32, 128, 128, 2048, seed=42)
    outs = gdr_bwd_wy_reduce_wrapper(*dev_args)
    _assert_precision(outs, cpu_args, label="prod_util_b1_h8_t2048_k128_v128")


if __name__ == "__main__":
    logging.info("gdr_bwd_wy_reduce (OP6b, Module 1, L0) -- self-test")
    logging.info("=" * 60)
    _sel = sys.argv[1:] if len(sys.argv) > 1 else None
    _all = [
        ("aligned", test_gdr_bwd_wy_reduce_aligned),
        ("ttail", test_gdr_bwd_wy_reduce_ttail),
        ("t3c", test_gdr_bwd_wy_reduce_t3c),
        ("tail_vlt", test_gdr_bwd_wy_reduce_tail_vlt),
        ("b2", test_gdr_bwd_wy_reduce_b2),
        ("prod_util", test_gdr_bwd_wy_reduce_prod_util),
    ]
    for _nm, _fn in _all:
        if _sel is None or _nm in _sel:
            _fn()
    logging.info("\nAll selected tests PASS!")
