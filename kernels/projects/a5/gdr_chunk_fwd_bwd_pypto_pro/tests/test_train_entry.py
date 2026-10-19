# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Acceptance gate for the composed training entry point.

`gated_delta_rule_train.gated_delta_rule_fwd_bwd` issues the same eight
launches as `test_composition.py` chains by hand, but with one device
synchronisation per step instead of eight and with the two independent branches
on separate streams. Only the host-side ordering and the core budgets differ, so
the requirement here is exact equality rather than a tolerance:

  * the concurrent order must equal the sequential order;
  * both must equal the chain as the component wrappers ship it, each
    synchronising for itself;
  * every core budget must give the same answer, including budgets that
    under-commit and over-commit the part, since a budget changes only how an
    independent item space folds onto cores;
  * steps queued on alternating streams without a per-step synchronisation must
    give the same answer as a single synchronised step.

Numerical correctness of the operator itself is `test_composition.py`'s gate;
this file pins only that composing it concurrently is free.

    python3 test_train_entry.py
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_KERNELS_DIR = os.path.join(os.path.dirname(_HERE), "kernels")
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from gated_delta_rule_train import (  # noqa: E402
    gated_delta_rule_fwd_bwd,
    gated_delta_rule_fwd_bwd_pipelined,
    scan_core_budget,
    reverse_scan_core_budget,
    _STREAMS,
)
from gdr_fwd_train_wy_impl import gdr_fwd_train_wy_wrapper  # noqa: E402
from gdr_fwd_train_scan_impl import gdr_fwd_train_scan_wrapper  # noqa: E402
from gdr_bwd_dv_local_impl import gdr_bwd_dv_local_wrapper  # noqa: E402
from gdr_bwd_reverse_scan_impl import gdr_bwd_reverse_scan_wrapper  # noqa: E402
from gdr_bwd_dqkw_impl import gdr_bwd_dqkw_wrapper  # noqa: E402
from gdr_bwd_wy_matmuls_impl import gdr_bwd_wy_matmuls_wrapper  # noqa: E402
from gdr_bwd_wy_reduce_impl import gdr_bwd_wy_reduce_wrapper  # noqa: E402
from gdr_bwd_finalize_impl import gdr_bwd_finalize_wrapper  # noqa: E402
from gated_delta_rule_bwd_golden import _get_device  # noqa: E402  (also calls torch.npu.set_device)

_CASES = (
    ("aligned_b1_t128_h4_k64_v64", dict(B=1, T=128, H=4, K=64, V=64)),
    ("t3c_b1_t192_h2_k128_v128", dict(B=1, T=192, H=2, K=128, V=128)),
    ("production_b1_t2048_h8_k128_v128", dict(B=1, T=2048, H=8, K=128, V=128)),
)

# (scan, reverse) pairs that under-commit, exactly commit and over-commit the part.
_BUDGET_PAIRS = ((8, 8), (12, 16), (16, 28), (28, 28))

_OUT_NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")


def _mk_inputs(B, T, H, K, V, device, seed):
    torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(-4.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return tuple(t.to(device) for t in (q, k, v, g, beta, grad_o))


def _shipped_chain(q, k, v, g, beta, grad_o):
    """The eight wrappers chained as they ship: every one synchronises."""
    (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
     dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t) = gdr_fwd_train_wy_wrapper(q, k, v, g, beta)
    o, s_hist, v_new = gdr_fwd_train_scan_wrapper(u, w, q_de, a_mat, k_f32, dfac, decay_exp)
    dv_local = gdr_bwd_dv_local_wrapper(grad_o, k_f32=k_f32, q_t=q_t, g_f32=g_f32)
    dh, dv2 = gdr_bwd_reverse_scan_wrapper(
        q_t, w_t, k_f32, grad_o=grad_o, decay_exp=decay_exp, g_f32=g_f32, dv_local=dv_local)
    dq_p, dk_p, dw, dg_p = gdr_bwd_dqkw_wrapper(
        grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, g_f32=g_f32, decay=decay)
    dkbg, dvb, dkb, dkt, _d_a = gdr_bwd_wy_matmuls_wrapper(
        attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2)
    dk, dv_chunked, dbeta_wy, dg2 = gdr_bwd_wy_reduce_wrapper(
        dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v)
    dq, dk_out, dv_out, dg, dbeta = gdr_bwd_finalize_wrapper(
        grad_o, dq_p, dk, dv_chunked, dg_p, dg2, dbeta_wy, sig)
    return o, dq, dk_out, dv_out, dg, dbeta


def _assert_equal(reference, candidate, label):
    """Exact equality: the launches and their inputs are identical."""
    for name, want, got in zip(_OUT_NAMES, reference, candidate):
        want_cpu, got_cpu = want.float().cpu(), got.float().cpu()
        if not torch.isfinite(want_cpu).all():
            raise AssertionError(f"{label}: reference {name} is not finite")
        max_abs = (want_cpu - got_cpu).abs().max().item()
        assert torch.equal(want_cpu, got_cpu), (
            f"{label}: {name} differs from the sequential chain, max_abs={max_abs:.6e}")
    logging.info("  %-44s EQUAL (all six outputs bit-identical)", label)


def test_train_entry_equivalence():
    device = _get_device()
    for name, shape in _CASES:
        logging.info("CASE %s %s", name, shape)
        args = _mk_inputs(device=device, seed=2024, **shape)
        reference = _shipped_chain(*args)
        _assert_equal(reference, gated_delta_rule_fwd_bwd(*args, concurrent=False),
                      "entry point, sequential")
        _assert_equal(
            reference, gated_delta_rule_fwd_bwd(*args, concurrent=True),
            "concurrent, default budgets scan=%d reverse=%d" % (
                scan_core_budget(shape["B"], shape["H"], shape["V"]),
                reverse_scan_core_budget(shape["B"], shape["H"], shape["V"])))
        for scan_budget, reverse_budget in _BUDGET_PAIRS:
            _assert_equal(
                reference,
                gated_delta_rule_fwd_bwd(
                    *args, concurrent=True,
                    scan_core_budget_override=scan_budget,
                    reverse_scan_core_budget_override=reverse_budget),
                f"concurrent, scan={scan_budget} reverse={reverse_budget}")


def test_train_entry_pipelined():
    """Queued steps on alternating streams must give the same answer.

    This is the path a training loop takes: no per-step synchronisation, and
    successive steps on alternating streams so their device work overlaps. The
    steps are independent, so the only thing that could go wrong is ordering.
    """
    device = _get_device()
    shape = dict(B=1, T=192, H=2, K=128, V=128)
    args = _mk_inputs(device=device, seed=2024, **shape)
    reference = _shipped_chain(*args)

    step_streams = (torch.npu.Stream(), torch.npu.Stream())
    here = torch.npu.current_stream()
    for stream in step_streams:
        stream.wait_stream(here)
    outputs = []
    for i in range(4):
        with torch.npu.stream(step_streams[i % len(step_streams)]):
            outputs.append(gated_delta_rule_fwd_bwd(*args, sync=False))
    for stream in step_streams:
        here.wait_stream(stream)
    torch.npu.synchronize()

    logging.info("CASE pipelined x%d on %d alternating streams %s",
                 len(outputs), len(step_streams), shape)
    for i, out in enumerate(outputs):
        _assert_equal(reference, out, f"queued step {i}")


def test_pipelined_entry_point():
    """The shipped pipelined entry point, which is the path a loop should take.

    Distinct steps, so an ordering or stream-hazard bug cannot hide behind
    identical inputs: each step gets its own seed, and each result is checked
    against that step's own single-step reference. Also covers one step (the
    degenerate sequence), an empty sequence, and the single-device guard.
    """
    device = _get_device()
    shape = dict(B=1, T=192, H=2, K=128, V=128)
    seeds = (2024, 7, 99, 1234, 55)
    batches = [_mk_inputs(device=device, seed=seed, **shape) for seed in seeds]
    references = [gated_delta_rule_fwd_bwd(*batch) for batch in batches]

    for depth in (1, 2, 3):
        results = gated_delta_rule_fwd_bwd_pipelined(batches, step_streams=depth)
        logging.info("CASE pipelined entry point, %d distinct steps, %d stream(s) %s",
                     len(batches), depth, shape)
        assert len(results) == len(batches), "one result per batch, in order"
        for i, (reference, result) in enumerate(zip(references, results)):
            _assert_equal(reference, result, f"depth={depth} step {i}")

    assert gated_delta_rule_fwd_bwd_pipelined([]) == [], "an empty sequence yields nothing"
    _assert_equal(references[0],
                  gated_delta_rule_fwd_bwd_pipelined(batches[:1])[0],
                  "single-step sequence")

    if torch.npu.device_count() > 1:
        other = torch.device(f"npu:{(device.index or 0) ^ 1}")
        mixed = list(batches[0][:-1]) + [batches[0][-1].to(other)]
        try:
            gated_delta_rule_fwd_bwd_pipelined([tuple(mixed)])
        except ValueError:
            logging.info("  %-44s rejected, as it must", "tensors split across two devices")
        else:
            raise AssertionError("a batch spanning two devices must be rejected")
    else:
        logging.info("  %-44s skipped (one device visible)", "multi-device rejection")


def test_in_flight_steps_do_not_share_branch_streams():
    """Steps that can be in flight together must get different branch streams.

    Sharing one pair couples them: the pair waits on both steps' streams while
    both wait on the pair, so their branches serialise or overlap depending on
    where the device had got to -- a long tail rather than a slower median, and
    the kind of defect a correctness test cannot see. This pins the structure
    instead: with N streams in flight, N distinct branch-stream slots must have
    been allocated, and they must be distinct stream objects.
    """
    device = _get_device()
    shape = dict(B=1, T=128, H=4, K=64, V=64)
    batches = [_mk_inputs(device=device, seed=seed, **shape) for seed in (1, 2, 3, 4)]

    for in_flight in (2, 3):
        _STREAMS.clear()
        gated_delta_rule_fwd_bwd_pipelined(batches, concurrent=True,
                                           step_streams=in_flight)
        slots = sorted(slot for _index, slot in _STREAMS)
        assert slots == list(range(in_flight)), (
            f"{in_flight} steps in flight must use slots {list(range(in_flight))}, got {slots}")
        pairs = [_STREAMS[key] for key in _STREAMS]
        flat = [stream for pair in pairs for stream in pair]
        assert len({id(stream) for stream in flat}) == len(flat), (
            "every in-flight step needs its own branch streams; a stream is shared")
        logging.info("  %-44s %d slots, %d distinct streams",
                     f"in flight={in_flight}", len(slots), len(flat))

    _STREAMS.clear()


if __name__ == "__main__":
    test_train_entry_equivalence()
    test_train_entry_pipelined()
    test_pipelined_entry_point()
    test_in_flight_steps_do_not_share_branch_streams()
    logging.info("ALL TRAIN ENTRY TESTS PASS")
