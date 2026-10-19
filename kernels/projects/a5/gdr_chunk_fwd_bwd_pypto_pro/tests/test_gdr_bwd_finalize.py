# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gdr_bwd_finalize — acceptance tests against the float32 CPU golden.

The backward epilogue: the five public gradients. The kernel and its host wrapper live in
`../kernels/gdr_bwd_finalize_impl.py`; this file holds only
the acceptance cases. Requires an Ascend a5 NPU (Ascend950PR); run it directly
to pick a subset by name, e.g. `python3 test_gdr_bwd_finalize.py aligned`.
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

from gdr_bwd_finalize_impl import (  # noqa: E402
    gdr_bwd_finalize_wrapper,
)

# =============================================================================
#   Tests — compare (dq,dk,dv,dg,dbeta) vs gdr_bwd_finalize_golden_cpu (fp32).
#   DESIGN Sec.8's 5 cases + prod_scale (utilization-profiling target) + B=2.
# =============================================================================
def _assert_precision(actual, *inputs, label="", **kwargs):
    """方案A精度校验（混合容差标准，阈值由 precision_compare 按各输出自身 dtype 查表）。

    `actual`/golden 均为 5 元组 (dq,dk,dv,dg,dbeta)；check_precision 逐张量按自身
    dtype（bf16 或 fp32）独立查表比较。dev-only import 留在函数体内 (precision_compare
    与 gdr_bwd_finalize_golden_cpu 不属于交付单元)。
    """
    from precision_compare import check_precision
    from gdr_bwd_finalize_golden_cpu import gdr_bwd_finalize_golden_cpu
    inputs_cpu = [i.cpu() if hasattr(i, "cpu") else i for i in inputs]
    golden = gdr_bwd_finalize_golden_cpu(*inputs_cpu, **kwargs)
    actual_cpu = tuple(a.cpu() if hasattr(a, "cpu") else a for a in actual)
    passed, summary = check_precision(actual_cpu, golden)
    if not passed:
        raise AssertionError(f"精度不达标: {summary}")
    if label:
        logging.info(f"[{label}] PASS ({summary})")
    return summary


def _mk(B, T, H, K, V, seed):
    from gdr_bwd_finalize_golden import _get_device
    device = _get_device()
    C = 64
    N = (T + 63) // 64
    torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)          # value unused
    dq_p = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
    dk_in = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
    dv_in = torch.randn((B, H, N, C, V), dtype=torch.float32, device=device)
    dg_p = torch.randn((B, H, N, C), dtype=torch.float32, device=device)
    dg2 = torch.randn((B, H, N, C), dtype=torch.float32, device=device)
    dbeta_wy = torch.randn((B, H, N, C), dtype=torch.float32, device=device)
    sig = torch.rand((B, H, N, C), dtype=torch.float32, device=device)               # [0,1]
    beta_act = sig.clone()                                                            # == sig
    return grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act


def test_gdr_bwd_finalize_aligned():
    """DESIGN Sec.8 aligned_b1_t128_h4_k64_v64: full divide, N=2, wide tile valid=64.
    HG0,H_TILE,HG = 4,1,4 (degenerate: 1 real head/group)."""
    args = _mk(1, 128, 4, 64, 64, seed=42)
    out = gdr_bwd_finalize_wrapper(*args)
    _assert_precision(out, *args, label="aligned_b1_t128_h4_k64_v64")


def test_gdr_bwd_finalize_ttail():
    """DESIGN Sec.8 ttail_b1_t100_h4_k64_v64: single-axis tail, N=2, last-chunk valid=36.
    HG0,H_TILE,HG = 4,1,4 (degenerate: 1 real head/group)."""
    args = _mk(1, 100, 4, 64, 64, seed=123)
    out = gdr_bwd_finalize_wrapper(*args)
    _assert_precision(out, *args, label="ttail_b1_t100_h4_k64_v64")


def test_gdr_bwd_finalize_t3c():
    """DESIGN Sec.8 t3c_b1_t192_h2_k128_v128: cross multi-tile, N=3, full-width K=V=128.
    HG0,H_TILE,HG = 2,1,2 (degenerate: 1 real head/group)."""
    args = _mk(1, 192, 2, 128, 128, seed=456)
    out = gdr_bwd_finalize_wrapper(*args)
    _assert_precision(out, *args, label="t3c_b1_t192_h2_k128_v128")


def test_gdr_bwd_finalize_tail_vlt():
    """DESIGN Sec.8 tail_vlt_b1_t200_h3_k128_v96: double boundary, N=4, tail valid=8, V<K.
    HG0,H_TILE,HG = 3,1,3 (degenerate: 1 real head/group)."""
    args = _mk(1, 200, 3, 128, 96, seed=42)
    out = gdr_bwd_finalize_wrapper(*args)
    _assert_precision(out, *args, label="tail_vlt_b1_t200_h3_k128_v96")


def test_gdr_bwd_finalize_hgtail():
    """DESIGN Sec.8 hgtail_b3_t512_h8_k64_v64: B*N=24, H=8 -- packs MULTIPLE real
    heads into one [16,64] group. Under the min-critical-path selector this now
    resolves to H_TILE=4/HG=2 (48 tasks, 1 wave, 85.7% fill) instead of the
    frozen formula's H_TILE=3/HG=3 (72 tasks, 2 waves, 64.3% fill). 8 is a
    multiple of 4, so the head-axis TAIL sub-path (h_valid < H_TILE) moved to
    test_gdr_bwd_finalize_hgtail_h5 below, which covers it under both
    selectors."""
    args = _mk(3, 512, 8, 64, 64, seed=17)
    out = gdr_bwd_finalize_wrapper(*args)
    _assert_precision(out, *args, label="hgtail_b3_t512_h8_k64_v64")


def test_gdr_bwd_finalize_hgtail_h5():
    """Head-axis TAIL coverage (added with the wave-quantization fix): B=3,T=512,
    H=5,K=V=64 -> B*N=24. H=5 is not a multiple of any H_TILE>1 the selector can
    pick, so `h_valid < H_TILE` is exercised under BOTH the frozen formula
    (H_TILE=2, HG=3 -> groups [0:2),[2:4),[4:5)) and the min-critical-path
    selector (H_TILE=3, HG=2 -> groups [0:3),[3:5), last group h_valid=2 < 3).
    Replaces the head-tail sub-path test_..._hgtail covered before the change."""
    args = _mk(3, 512, 5, 64, 64, seed=23)
    out = gdr_bwd_finalize_wrapper(*args)
    _assert_precision(out, *args, label="hgtail_h5_b3_t512_h5_k64_v64")


def test_gdr_bwd_finalize_prod_scale():
    """Production-scale case (beyond DESIGN Sec.8's minimum 4): B=1,T=2048,H=8,
    K=V=128, N=32, B*N=32. Min-critical-path selector -> H_TILE=1, HG=8 ->
    B*N*HG=256 tasks over 56 vector cores = 5 waves, occupancy 256/(5*56)=91.4%
    (the frozen formula gave HG=2 -> 64 tasks / 2 waves / 57.1% on this 56-core
    host). Used both as an additional correctness point and as the
    utilization-profiling target."""
    args = _mk(1, 2048, 8, 128, 128, seed=7)
    out = gdr_bwd_finalize_wrapper(*args)
    _assert_precision(out, *args, label="prod_scale_b1_t2048_h8_k128_v128")


def test_gdr_bwd_finalize_b2():
    """B=2 correctness case for declared-deviation D5's fix: the kernel's public
    Tensor type annotations previously declared the leading (B) dimension as the
    LITERAL 1 (inherited unchanged from the Stage 3.5 KAL transpile, which only
    ever simulated B=1 shapes), restricting real hardware delivery to B=1 even
    though SPEC's dynamic_axes_ranges declares B in [1,16]. Fixed by switching
    dim0 to pl.DYNAMIC on all 12 Tensor params -- the same fix already proven on
    sibling gdr_bwd_dv_local (bit-identical results there). This case is the
    correctness proof for that fix: B=2,T=192,H=2,K=V=128, N=3 -> B*N=6 (b,n)
    pairs, HG0=min(2,max(1,ceil(64/6)=11))=2, H_TILE=1, HG=2 -> B*N*HG=12 tasks
    spanning two distinct batch indices. Unaffected by ROUND 3's store-mechanism
    change (per-group store is still per-(b,n,hg), B-independent arithmetic)."""
    args = _mk(2, 192, 2, 128, 128, seed=99)
    out = gdr_bwd_finalize_wrapper(*args)
    _assert_precision(out, *args, label="b2_b2_t192_h2_k128_v128")


if __name__ == "__main__":
    import sys
    logging.info("gdr_bwd_finalize (OP7 / M10) — Stage 4 L0 delivery self-test (ROUND 3)")
    logging.info("=" * 60)
    _sel = sys.argv[1:] if len(sys.argv) > 1 else None
    _all = [
        ("aligned", test_gdr_bwd_finalize_aligned),
        ("ttail", test_gdr_bwd_finalize_ttail),
        ("t3c", test_gdr_bwd_finalize_t3c),
        ("tail_vlt", test_gdr_bwd_finalize_tail_vlt),
        ("hgtail", test_gdr_bwd_finalize_hgtail),
        ("hgtail_h5", test_gdr_bwd_finalize_hgtail_h5),
        ("prod_scale", test_gdr_bwd_finalize_prod_scale),
        ("b2", test_gdr_bwd_finalize_b2),
    ]
    for _nm, _fn in _all:
        if _sel is None or _nm in _sel:
            _fn()
    logging.info("\nAll selected tests PASS!")
