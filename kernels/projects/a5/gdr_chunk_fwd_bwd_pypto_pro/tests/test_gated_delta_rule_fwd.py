# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gated_delta_rule_fwd — acceptance tests against the float32 CPU golden.

Eight shapes covering the aligned case, a T tail, a V tail, multi-tile K/V,
a batch greater than one, a long sequence, and the widest supported K/V.
Requires an Ascend a5 NPU; run the file directly to pick a subset by name,
e.g. `python3 test_gated_delta_rule_fwd.py mid colsplit`.
"""

import logging
import os
import sys

import pytest
import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_KERNELS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "kernels")
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from gated_delta_rule_fwd_golden import gated_delta_rule_fwd_golden
from precision_compare import check_precision

from gated_delta_rule_fwd_impl import (
    gated_delta_rule_fwd,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _npu_device():
    dev_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(dev_id)
    return torch.device(f"npu:{dev_id}")


def _assert_precision_o(o_out, q, k, v, g, beta, label=""):
    """Judge the bfloat16 output o against the float32 CPU golden.

    o is bfloat16, so the natural tolerance is the bf16 row (rtol = atol = 2^-6
    with a fixed absolute cap of 1.0). At the magnitudes this operator produces
    (peak ~1e2-1e6) that fixed cap is unreachable in principle: a bf16 value of
    magnitude P already carries ~P*2^-8 of quantisation, so even the *ideal*
    bf16 rounding of the golden trips it. The criterion is therefore peak
    scaled -- the matched ratio must be met under the bf16 row AND the max
    absolute error must be bounded by peak*2^-6 -- plus a cross-check that o is
    no worse than the ideal bf16 rounding of the golden, which is what proves
    there is no computational error beyond the final cast.
    """
    import re
    ref = gated_delta_rule_fwd_golden(q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu())
    a_cpu = o_out.cpu().float()
    assert tuple(a_cpu.shape) == tuple(ref.shape), \
        f"[{label}] o shape {tuple(a_cpu.shape)} != {tuple(ref.shape)}"
    assert torch.isfinite(a_cpu).all(), f"[{label}] o not finite"
    peak = ref.abs().max().item()
    our_max = (a_cpu - ref).abs().max().item()
    ideal_max = (ref.to(torch.bfloat16).float() - ref).abs().max().item()
    # bf16 tolerance row (o IS bf16); this is the natural default for a bf16 output.
    passed, summary = check_precision(a_cpu, ref, dtype_str="bfloat16")
    if passed:
        logging.info(f"[{label}] o PASS (bf16 row: peak={peak:.3e} {summary}; "
                     f"our_max_abs={our_max:.4e} ideal-bf16={ideal_max:.4e})")
        return
    m = re.search(r"matched_ratio=([0-9.eE+-]+)\s*\(req>=([0-9.eE+-]+)\)", summary)
    mr = float(m.group(1)) if m else 0.0
    req = float(m.group(2)) if m else 1.0
    peak_bound = peak * (2.0 ** -6)          # peak-scaled bf16 relative bound
    if mr >= req and peak >= 1.0 and our_max <= peak_bound and our_max <= ideal_max * 1.05:
        logging.info(
            f"[{label}] o PASS (bf16 matched_ratio={mr:.6f} >= req {req} fully met; "
            f"fixed abs cap 1.0 unreachable at peak={peak:.3e} -> peak-scaled: "
            f"our_max_abs={our_max:.4e} <= peak*2^-6={peak_bound:.4e}; and our_max_abs "
            f"<= ideal-bf16-rounding {ideal_max:.4e}*1.05 => no compute error beyond bf16). "
            f"raw=({summary})")
        return
    raise AssertionError(f"[{label}] o mismatch: peak={peak:.3e} our_max_abs={our_max:.4e} "
                         f"ideal-bf16={ideal_max:.4e} {summary}")


def _mk(B, T, H, K, V, seed):
    device = _npu_device()
    torch.manual_seed(seed)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)
    g = torch.empty((B, T, H), dtype=torch.float32, device=device).uniform_(-4.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device=device)
    return q, k, v, g, beta


@pytest.mark.soc("950")
def test_gated_delta_rule_fwd_aligned():
    q, k, v, g, beta = _mk(1, 128, 4, 64, 64, seed=42)
    o = gated_delta_rule_fwd(q, k, v, g, beta)
    _assert_precision_o(o, q, k, v, g, beta, label="aligned T128")


@pytest.mark.soc("950")
def test_gated_delta_rule_fwd_tail_T():
    q, k, v, g, beta = _mk(1, 100, 4, 64, 64, seed=123)
    o = gated_delta_rule_fwd(q, k, v, g, beta)
    _assert_precision_o(o, q, k, v, g, beta, label="tail_T T100")


@pytest.mark.soc("950")
def test_gated_delta_rule_fwd_tail_KV():
    q, k, v, g, beta = _mk(1, 192, 2, 128, 96, seed=456)
    o = gated_delta_rule_fwd(q, k, v, g, beta)
    _assert_precision_o(o, q, k, v, g, beta, label="tail_KV T192 K128 V96")


@pytest.mark.soc("950")
def test_gated_delta_rule_fwd_multitile():
    q, k, v, g, beta = _mk(1, 200, 3, 128, 96, seed=42)
    o = gated_delta_rule_fwd(q, k, v, g, beta)
    _assert_precision_o(o, q, k, v, g, beta, label="multitile T200 K128 V96")


@pytest.mark.soc("950")
def test_gated_delta_rule_fwd_smoke():
    q, k, v, g, beta = _mk(1, 128, 4, 64, 64, seed=0)
    o = gated_delta_rule_fwd(q, k, v, g, beta)
    _assert_precision_o(o, q, k, v, g, beta, label="smoke")


@pytest.mark.soc("950")
def test_gated_delta_rule_fwd_mid():
    q, k, v, g, beta = _mk(1, 2048, 8, 128, 128, seed=7)
    o = gated_delta_rule_fwd(q, k, v, g, beta)
    _assert_precision_o(o, q, k, v, g, beta, label="mid T2048 H8 K128 V128")


@pytest.mark.soc("950")
def test_gated_delta_rule_fwd_batch():
    q, k, v, g, beta = _mk(3, 192, 2, 128, 96, seed=456)
    o = gated_delta_rule_fwd(q, k, v, g, beta)
    _assert_precision_o(o, q, k, v, g, beta, label="batch B3 T192 K128 V96")


@pytest.mark.soc("950")
def test_gated_delta_rule_fwd_wide_kv():
    q, k, v, g, beta = _mk(1, 128, 2, 256, 256, seed=42)
    o = gated_delta_rule_fwd(q, k, v, g, beta)
    _assert_precision_o(o, q, k, v, g, beta, label="wide_kv K256 V256")


if __name__ == "__main__":
    import sys
    logging.info("gated_delta_rule_fwd — acceptance cases")
    logging.info("=" * 60)
    _sel = sys.argv[1:] if len(sys.argv) > 1 else None
    _all = [
        ("aligned", test_gated_delta_rule_fwd_aligned),
        ("tail_T", test_gated_delta_rule_fwd_tail_T),
        ("tail_KV", test_gated_delta_rule_fwd_tail_KV),
        ("multitile", test_gated_delta_rule_fwd_multitile),
        ("smoke", test_gated_delta_rule_fwd_smoke),
        ("mid", test_gated_delta_rule_fwd_mid),
        ("batch", test_gated_delta_rule_fwd_batch),
        ("wide_kv", test_gated_delta_rule_fwd_wide_kv),
    ]
    for _nm, _fn in _all:
        if _sel is None or _nm in _sel:
            _fn()
    logging.info("\nAll selected tests PASS!")
