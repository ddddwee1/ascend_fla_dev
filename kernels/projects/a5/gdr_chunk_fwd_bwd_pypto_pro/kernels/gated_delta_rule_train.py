# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The composed training step: one call for the forward and the backward.

The operator ships as nine single-kernel launches, each with its own host entry
point. This module is the composition of the eight that a training step needs,
and it exists because the way they are chained is itself part of the operator's
cost rather than a detail left to the caller.

It offers two entry points, both single-device:

* `gated_delta_rule_fwd_bwd_pipelined(batches)` runs a sequence of independent
  steps back to back and is **the one to use**: 828.5 us per step with sixteen
  steps queued, against 1617.5 us for the eight component wrappers chained by
  hand, so **1.95 times**. Queue as many steps as the accumulation window
  allows -- four steps a call gives only 1.77 times, because each call pays for
  filling and draining the pipeline.
* `gated_delta_rule_fwd_bwd(...)` runs one step and synchronises, 1027.0 us. It
  is the right call for a single isolated step, and it is what the pipelined
  form issues internally.

Both are bit-identical to the hand-written chain. Ratios are quoted rather than
absolute microseconds because they are what reproduces: this part's absolute
per-step medians drift several percent between runs, so every figure in
`docs/PERFORMANCE.md` comes from arms interleaved within a single run.

Two properties of the dependency graph drive the composition.

**The chain needs one device synchronisation, not eight.** Every component
wrapper synchronises before returning, which is correct when it is called on its
own and is a host-device round trip per launch when eight are chained. The
wrappers therefore take a keyword-only `sync`, whose default preserves their
standalone behaviour, and this entry point passes `sync=False` to all eight and
synchronises once.

**The graph has two independent branches.** `gdr_fwd_train_scan` and the pair
`gdr_bwd_dv_local -> gdr_bwd_reverse_scan` both read only the outputs of
`gdr_fwd_train_wy`, and nothing reads the scan's results until
`gdr_bwd_dqkw`. The two can therefore run concurrently, which this module
expresses by issuing them on separate streams and rejoining before `dqkw`. The
launch bracket is `kernel[stream, block_dim]` and every component passes `None`
there, which the runtime resolves through the current stream, so redirecting a
launch needs no change inside any kernel.

Concurrency alone is not enough, and the reason is the whole of the core
budgeting below: each component's tiling selector maximises its own core fill on
the assumption that it owns the part. Two launches that each claim all 28 cube
cores do not overlap -- they serialise, with the scheduler interleaving them --
so the overlap pays only when the co-running launches are sized so that their
block counts sum to the part rather than oversubscribing it.

Every setting of either budget, and the concurrent and sequential orders alike,
produce bit-identical results: a budget changes only how an independent item
space folds onto cores, never the arithmetic. `tests/test_train_entry.py` is the
acceptance gate for that claim.
"""

import logging
import os

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
from pypto_pro.runtime.platform import get_platform_info

from gdr_fwd_train_wy_impl import gdr_fwd_train_wy_wrapper
from gdr_fwd_train_scan_impl import gdr_fwd_train_scan_wrapper
from gdr_bwd_dv_local_impl import gdr_bwd_dv_local_wrapper
from gdr_bwd_reverse_scan_impl import gdr_bwd_reverse_scan_wrapper
from gdr_bwd_dqkw_impl import gdr_bwd_dqkw_wrapper
from gdr_bwd_wy_matmuls_impl import gdr_bwd_wy_matmuls_wrapper
from gdr_bwd_wy_reduce_impl import gdr_bwd_wy_reduce_wrapper
from gdr_bwd_finalize_impl import gdr_bwd_finalize_wrapper

REVERSE_SCAN_TV = 64    # V sub-block width of gdr_bwd_reverse_scan
MIN_SCAN_CORES = 8      # floor on the forward scan's share of the part

STEP_STREAMS = 2        # steps in flight; successive steps alternate over these

_LOG_DEDUPE = None


class _DropRepeatedRuntimeRecords(logging.Filter):
    """Pass the first occurrence of each runtime log message and drop repeats.

    The runtime emits a fixed pair of records on every kernel launch when its
    exception-dump callback cannot be registered, because registration is
    retried per launch and only a success is remembered. That is 24 records for
    one training step here, all on the root logger, and writing them costs
    host time on the critical path: measured over the eight launches of one
    step, 974 us of host dispatch with them against 879 us without.

    Host time matters at this shape precisely because a pipelined loop runs the
    host ahead of the device. The eight kernels are 909 us of device time, so
    974 us of host dispatch makes the loop host-bound and sets the per-step
    floor above the device; 879 us puts it back under, which is what lets the
    device be the limit and lets successive steps overlap. For a single
    synchronised step the same records are invisible, since the host finishes
    well inside the device's window.

    The filter is deliberately narrow. It only considers records emitted from
    inside the runtime package, so nothing a caller logs is ever touched, and it
    keys on the formatted message rather than the format string, so distinct
    messages from one call site are not collapsed. The first occurrence of each
    message is always passed: the condition it reports is real and worth knowing
    once.
    """

    _ORIGIN = os.path.join("pypto_pro", "runtime")

    def __init__(self):
        super().__init__()
        self._seen = set()

    def filter(self, record):
        if self._ORIGIN not in (record.pathname or ""):
            return True
        key = (record.name, record.levelno, record.getMessage())
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def install_runtime_log_dedupe():
    """Attach `_DropRepeatedRuntimeRecords` to the root logger, once per process.

    It must persist to be worth anything: a filter installed and removed around
    each step passes the first occurrence of every message each time, so the
    records are never actually dropped.
    """
    global _LOG_DEDUPE
    if _LOG_DEDUPE is None:
        _LOG_DEDUPE = _DropRepeatedRuntimeRecords()
        logging.getLogger().addFilter(_LOG_DEDUPE)
    return _LOG_DEDUPE

_STREAMS = {}
_STEP_STREAMS = {}


def _streams_for(device, slot=0):
    """The cached pair of branch streams for one device and one step slot.

    Streams are created once and reused; creating them per call would put stream
    construction on the step's critical path.

    `slot` is what keeps two steps in flight from sharing a pair, and it is not
    optional for correctness of the overlap. A step takes its branch streams,
    makes them wait on the stream it is itself running on, and then makes that
    stream wait on them. If a second step running on a different stream borrows
    the same pair, those waits cross: the pair ends up waiting on both steps'
    streams while both steps' streams wait on the pair, so neither step's
    branches can start until the other's queue drains. The device work still
    overlaps sometimes and serialises other times, depending on where the
    device had got to when each launch was queued -- which shows up as a long
    tail rather than a slower median. One pair per in-flight slot removes the
    sharing, and with it the coupling.
    """
    index = device.index if device.index is not None else torch.npu.current_device()
    key = (index, slot)
    pair = _STREAMS.get(key)
    if pair is None:
        pair = (torch.npu.Stream(device=index), torch.npu.Stream(device=index))
        _STREAMS[key] = pair
    return pair


def _step_streams_for(device, count):
    """The cached streams that successive steps alternate over, on one device.

    These are streams, not devices: the concurrency is inside a single device,
    and every stream here belongs to the device the inputs live on.
    """
    index = device.index if device.index is not None else torch.npu.current_device()
    key = (index, count)
    streams = _STEP_STREAMS.get(key)
    if streams is None:
        streams = tuple(torch.npu.Stream(device=index) for _ in range(count))
        _STEP_STREAMS[key] = streams
    return streams


def reverse_scan_core_budget(B, H, V):
    """The blocks to plan `gdr_bwd_reverse_scan` for: the fewest that keep its wave count.

    Its item space is `B * H * ceil(V / TV)` and the kernel strides over it, so
    the block count is free and only balance depends on it. Taking every core is
    right when the items divide evenly over the part. Where they do not, the
    remainder forms a final wave that holds only a few cores while the rest are
    idle for its whole duration, and the launch's own duration is unchanged by
    dividing the same items into the same number of waves over fewer blocks.
    Fewer blocks at an unchanged wave count is therefore free for this launch
    and leaves a constant number of cores free throughout it, which is what a
    concurrent co-runner can use.
    """
    cores = get_platform_info().core_num or 1
    items = B * H * ((V + REVERSE_SCAN_TV - 1) // REVERSE_SCAN_TV)
    if items <= cores:
        return items
    waves = -(-items // cores)
    return -(-items // waves)


def scan_core_budget(B, H, V):
    """The cores to plan `gdr_fwd_train_scan` for while the backward branch runs beside it.

    The branch's resident demand is `gdr_bwd_reverse_scan`'s block count, so
    what is left of the part is what the scan can expect. The floor guards the
    case of a head count large enough that the branch alone would leave nothing.

    Both counts are the balanced ones, which is what makes a single subtraction
    correct: an unbalanced co-runner holds a different number of cores in each
    of its waves, and no single budget fits them all.
    """
    cores = get_platform_info().core_num or 1
    return max(MIN_SCAN_CORES, cores - reverse_scan_core_budget(B, H, V))


def gated_delta_rule_fwd_bwd(q, k, v, g, beta, grad_o, *, concurrent=True,
                             sync=True, stream_slot=0,
                             scan_core_budget_override=None,
                             reverse_scan_core_budget_override=None):
    """Run the training forward and backward, and return the public results.

    Args:
        q, k: [B, T, H, K] bfloat16
        v: [B, T, H, V] bfloat16
        g: [B, T, H] float32, a log-space decay, non-positive
        beta: [B, T, H] bfloat16, sigmoid logits
        grad_o: [B, T, H, V] bfloat16, the incoming output gradient
        concurrent: issue the two independent branches on separate streams.
            False gives the plain sequential order, which is the reference for
            any comparison: the two agree bit for bit.
        sync: synchronise the device before returning. False hands that
            responsibility to the caller, who must synchronise before reading
            any result on the host; ordering on the device is already
            guaranteed. It also lets successive steps overlap, which is how a
            training loop should call this -- see below.
        stream_slot: which cached pair of branch streams this step may use.
            Steps that can be in flight at the same time must be given
            different slots, or their branch streams couple and the overlap
            becomes order-dependent -- see `_streams_for`. A caller running one
            step at a time can leave it at 0.
        scan_core_budget_override: the cores `gdr_fwd_train_scan` plans for
            while it runs concurrently, in place of `scan_core_budget`. Ignored
            when `concurrent` is False.
        reverse_scan_core_budget_override: the blocks
            `gdr_bwd_reverse_scan` plans for, in place of
            `reverse_scan_core_budget`.

    Returns:
        (o, dq, dk, dv, dg, dbeta) -- `o` and the three large gradients in
        bfloat16, `dg` and `dbeta` in float32, all in the public [B, T, H, *]
        layout.

    A training loop should not call this per step, because a synchronised step
    pays its host dispatch where the device cannot hide it. Use
    `gated_delta_rule_fwd_bwd_pipelined`, which issues this function with
    `sync=False` over a sequence of steps and is 168 us per step cheaper at the
    reference shape.
    """
    (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
     dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t) = gdr_fwd_train_wy_wrapper(
        q, k, v, g, beta, sync=False)

    def run_scan(budget):
        return gdr_fwd_train_scan_wrapper(u, w, q_de, a_mat, k_f32, dfac, decay_exp,
                                          sync=False, cube_core_budget=budget)

    def run_backward_branch(reverse_budget):
        dv_local = gdr_bwd_dv_local_wrapper(grad_o, k_f32=k_f32, q_t=q_t, g_f32=g_f32,
                                            sync=False)
        return gdr_bwd_reverse_scan_wrapper(
            q_t, w_t, k_f32, grad_o=grad_o, decay_exp=decay_exp, g_f32=g_f32,
            dv_local=dv_local, sync=False, cube_core_budget=reverse_budget)

    if concurrent:
        B, T, H, V = grad_o.shape
        scan_budget = (scan_core_budget_override if scan_core_budget_override is not None
                       else scan_core_budget(B, H, V))
        reverse_budget = (reverse_scan_core_budget_override
                          if reverse_scan_core_budget_override is not None
                          else reverse_scan_core_budget(B, H, V))
        scan_stream, backward_stream = _streams_for(grad_o.device, stream_slot)
        here = torch.npu.current_stream()
        # Both branches read train_wy's outputs, which were produced here.
        scan_stream.wait_stream(here)
        backward_stream.wait_stream(here)
        with torch.npu.stream(scan_stream):
            o, s_hist, v_new = run_scan(scan_budget)
        with torch.npu.stream(backward_stream):
            dh, dv2 = run_backward_branch(reverse_budget)
        # dqkw needs both branches, so rejoin before it.
        here.wait_stream(scan_stream)
        here.wait_stream(backward_stream)
        # These five were allocated on a side stream and are read from this one,
        # so the caching allocator has to know this stream still needs them.
        for tensor in (o, s_hist, v_new, dh, dv2):
            tensor.record_stream(here)
    else:
        o, s_hist, v_new = run_scan(None)
        dh, dv2 = run_backward_branch(None)

    dq_p, dk_p, dw, dg_p = gdr_bwd_dqkw_wrapper(
        grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, g_f32=g_f32, decay=decay,
        sync=False)
    dkbg, dvb, dkb, dkt, _d_a = gdr_bwd_wy_matmuls_wrapper(
        attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2, sync=False)
    dk, dv_chunked, dbeta_wy, dg2 = gdr_bwd_wy_reduce_wrapper(
        dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v, sync=False)
    dq, dk_out, dv_out, dg, dbeta = gdr_bwd_finalize_wrapper(
        grad_o, dq_p, dk, dv_chunked, dg_p, dg2, dbeta_wy, sig, sync=False)

    if sync:
        torch.npu.synchronize()
    return o, dq, dk_out, dv_out, dg, dbeta


def gated_delta_rule_fwd_bwd_pipelined(batches, *, concurrent=True,
                                       step_streams=STEP_STREAMS,
                                       dedupe_runtime_log=True,
                                       scan_core_budget_override=None,
                                       reverse_scan_core_budget_override=None):
    """Run a sequence of independent training steps on one device, and return their results.

    This is the cheapest per-step cost the operator reaches and the path a
    training loop should take.

    **How many steps you queue per call is the single biggest factor.** Every
    call fills the pipeline and drains it: the first step has nothing running
    ahead of it to overlap with and the last step's tail has nothing following,
    so those two edges are paid once per call however long the sequence is.
    Queue four steps and the edges are half of them. At B=1, T=2048, H=8,
    K=V=128, per step:

        the eight wrappers chained as they ship   1617.5 us   1.00x
        one synchronised step                     1027.0      1.58x
        pipelined, 4 steps queued                  912.7      1.77x
        pipelined, 8 steps queued                  864.7      1.87x
        pipelined, 16 steps queued                 828.5      1.95x

    So queue the whole accumulation window in one call rather than a few steps
    at a time. Longer is also steadier: at 16 the p90 is 831.4 us against a
    828.5 us median, where at 4 it is 922.4 against 912.7.

    828.5 us is well under the 909.14 us the eight kernels need as device time
    on their own, which is only possible because successive steps' device work
    overlaps. The cube is busy for almost all of a step -- the forward scan and
    the reverse scan are co-sized to fill the part exactly -- so what the next
    step overlaps with is chiefly the 66 us of pure-vector tail in `wy_reduce`
    and `finalize`, where the cube would otherwise be idle.

    Two overlaps produce that, and both live inside one device -- these are
    streams, not devices, and every launch goes to the device the inputs are
    already on:

    * **A step's host dispatch hides under the previous step's device work.**
      There is one synchronisation for the whole sequence rather than one per
      step, so the host runs ahead of the device instead of waiting for it.
      This only holds while host dispatch stays under the device's 909 us, so
      the loop is sensitive to host cost in a way a single step is not --
      `dedupe_runtime_log` is what keeps it there.
    * **Each in-flight step owns its streams.** A step takes a pair of branch
      streams for its two independent halves, and steps that can be in flight
      together are given different pairs. Sharing one pair couples them and the
      overlap becomes order-dependent; `_streams_for` has the mechanism.
    * **Successive steps' device work overlaps.** The steps are independent, so
      they are issued on alternating streams; one step's vector tail can run
      while the next step's cube head is already going. The per-step cost lands
      below the sum of the eight kernels' individually measured device times,
      which is only possible if the device itself is overlapped and not merely
      kept fed.

    Args:
        batches: an iterable of `(q, k, v, g, beta, grad_o)` tuples, one per
            step, with the dtypes and shapes `gated_delta_rule_fwd_bwd`
            documents. The steps must be independent of each other, which
            successive microbatches of an accumulation window are; this does
            **not** hold for steps separated by an optimiser update, which must
            be run in separate calls.
        concurrent: issue each step's two independent branches on separate
            streams, as `gated_delta_rule_fwd_bwd` does. On by default: each
            in-flight step is given its own pair of branch streams, so the two
            concurrency axes compose instead of interfering. Measured over 25
            rotated iterations, per-step medians with the tail:

                one branch pair for all steps   922.0  p90 1202.1  spread 377.0
                one pair per in-flight step     914.1  p90  971.7  spread  89.6
                no within-step concurrency      921.4  p90 1025.2  spread 185.8

            The middle row is this default: the best median and a spread four
            times tighter than sharing a pair, two times tighter than not
            splitting the step at all. See `_streams_for` for why sharing a pair
            across steps produces that tail.
        dedupe_runtime_log: drop repeats of the runtime's per-launch log
            records, which cost 95 us of host time per step here and decide
            whether this loop is host-bound or device-bound. See
            `_DropRepeatedRuntimeRecords`; the first occurrence of each message
            still reports. Pass False to leave the root logger untouched.
        step_streams: how many steps may be in flight. Two collects the
            overlap; three and four are a wash (832.5 and 836.1 us against
            833.2 at sixteen steps queued), so two is the default because it
            holds the least memory. This is not the lever -- the number of steps
            QUEUED is, as the table above shows.
        scan_core_budget_override, reverse_scan_core_budget_override: as on
            `gated_delta_rule_fwd_bwd`, applied to every step.

    Returns:
        A list of `(o, dq, dk, dv, dg, dbeta)` tuples, one per input batch, in
        the order the batches were given, ready to read on the host.

    Every result is bit-identical to running the same step on its own: this
    changes only when work is issued, never what is computed.

    Memory is the cost. Each step in flight holds its own intermediates, so
    that part scales with `step_streams` and not with the length of the
    sequence; but every step's six results are returned, so those scale with
    the number queued. An accumulation window per call is the right granularity:
    long enough to amortise the pipeline's edges, bounded by the window.

    One note for anyone benchmarking this rather than training with it: the
    pipeline needs about ten iterations to reach steady state, and reads roughly
    50% high before it does. Three warm-up iterations are not enough.
    """
    if dedupe_runtime_log:
        install_runtime_log_dedupe()

    batches = list(batches)
    if not batches:
        return []

    devices = {tensor.device for batch in batches for tensor in batch}
    if len(devices) != 1:
        raise ValueError(
            f"every tensor of every step must live on one device, got {sorted(map(str, devices))}")
    device = devices.pop()

    here = torch.npu.current_stream()
    streams = _step_streams_for(device, max(1, int(step_streams)))
    for stream in streams:
        stream.wait_stream(here)

    results = []
    for index, batch in enumerate(batches):
        slot = index % len(streams)
        with torch.npu.stream(streams[slot]):
            results.append(gated_delta_rule_fwd_bwd(
                *batch, concurrent=concurrent, sync=False, stream_slot=slot,
                scan_core_budget_override=scan_core_budget_override,
                reverse_scan_core_budget_override=reverse_scan_core_budget_override))

    # The results were produced on the step streams and are read from this one.
    for stream in streams:
        here.wait_stream(stream)
    for result in results:
        for tensor in result:
            tensor.record_stream(here)
    torch.npu.synchronize()
    return results
