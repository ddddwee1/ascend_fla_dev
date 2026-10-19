# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Chunk-parallel backward: dq, dk, dw and the chunk-parallel part of dg.

Third of the six backward operators, and the one that consumes the state trail.
Every chunk is independent here: the two serial scans have already turned the
cross-chunk dependencies into the per-chunk tensors s_hist and dh, so this
operator is pure chunk-parallel work with eleven matmuls per item.

Mathematics
-----------
Per chunk, writing S for the entering state s_hist[., i], dS for dh[., i],
decay for the within-chunk cumulative gate sum, g_last for its last element and
scale for 1/sqrt(K)::

    scores   = tril_incl((grad_o @ v_new^T) * exp(decay_row - decay_col) * scale)
    dq       = (grad_o @ S) * exp(decay) * scale  +  scores @ k_f32
    dk       = (v_new @ dS) * exp(g_last - decay) +  scores^T @ q_scaled
    dw       = -(dv2 @ S)
    dg       = sum over K of (dq * q)  -  sum over K of (dk * k_f32)

and the per-chunk scalar that closes the gate gradient is folded onto the last
token of the chunk::

    dg[C-1] += exp(g_last) * sum over K,V of (S * dS)
               + sum over K of ((v_new @ dS) * exp(g_last - decay) * k_f32)

`scores^T` is never computed. The masked transpose equals the transpose of the
masked matrix, because masking keeps the lower triangle including the diagonal
of one and the upper triangle including the diagonal of the other; so `scores`
is written once to GM and read back transposed for the second use.

q arrives pre-scaled as q_scaled = q * scale, so the dq reduction multiplies by
sqrt(K) to recover the raw q, rather than carrying a second copy of q.

Interface
---------
``gdr_bwd_dqkw_wrapper(grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, g_f32, decay)``

    grad_o        [B, T, H, V] bf16     incoming output gradient, public layout
    v_new         [B, H, N, C, V] f32   corrected value rows from the forward scan
    k_f32         [B, H, N, C, K] f32   k widened to float32
    q_s           [B, H, N, C, K] f32   q scaled by 1/sqrt(K)
    s_hist        [B, H, N, K, V] f32   entering state per chunk
    dh            [B, H, N, K, V] f32   entering state gradient per chunk
    dv2           [B, H, N, C, V] f32   dv after the reverse scan
    g_f32         [B, H, N, C] f32      accepted for signature parity, unused
    decay         [B, H, N, C] f32      within-chunk cumulative gate sum

    returns dq_p, dk_p, dw  [B, H, N, C, K] f32
            dg_p            [B, H, N, C] f32

v_new cannot be replaced by the primary v here: the delta correction is part of
the chain, and substituting v makes dq wrong by O(1), not by a tolerance.

Supported batch size
--------------------
This kernel pins the batch axis of every tensor to 1, so the operator, and
therefore the whole pipeline, supports B = 1 only. The body itself decodes the
batch index correctly; only the annotations are narrow.

Layout and scheduling
---------------------
One launch of one mixed cube/vector kernel. A work item is one
(batch, head, chunk) triple with the same eleven matmuls each, so contiguous
per-core runs balance themselves. The chain is vector (widen grad_o), cube
(scores), vector (gate and mask), cube (five output matmuls), vector (assemble
and reduce), with four handshakes.

L0A is the binding resource: the largest left operand is [C, V] in float32,
32 KB, and at most two are live at once, which is exactly the 64 KB capacity.
The two K blocks are written out rather than looped because a runtime loop costs
measurable trip and scalar overhead for two iterations.

Within a block the three matmuls that do not need `scores` are issued first, so
that the operand fetches and the MACs of dq_inter, dk_inter and dw cover the
vector core's mask latency, and the `scores` rendezvous is taken once per item
rather than once per block.

Two accumulations are deliberately not done in place in UB. A read-modify-write
on a single tile address inside a vector loop is not ordered across iterations
on this part, so both the per-chunk gate scalar and the state dot product are
written as one partial per lane and reduced once after the producing loop.
"""

import logging

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

C = 64          # tokens per chunk, a frozen constant of the operator family
TILE = 64       # K and V sub-block width
KMAX = 128      # supported K upper bound
VMAX = 128      # supported V upper bound

_FIX = pl.PipeType.FIX
_S = pl.PipeType.S
_V = pl.PipeType.V
_MTE3 = pl.PipeType.MTE3
_IB = pl.CrossCoreSyncMode.INTRA_BLOCK

# Cross-core event ids; each pair guards one buffer handover. The two the cube
# raises before the task loop start with one credit, so the first item's vector
# work is not blocked by a consumer that has nothing to consume yet.
E_GRAD_O_GM_FREE = 0    # cube -> vector: the widened grad_o bridge has been read
E_GRAD_O_GM_READY = 1   # vector -> cube: that bridge is filled
E_SCORES_IN_UB = 2      # cube -> vector: raw scores drained from L0C to UB
E_SCORES_UB_FREE = 3    # vector -> cube: the scores landing tile is consumed
E_SCORES_GM_FREE = 4    # cube -> vector: the masked-scores bridge has been read
E_SCORES_GM_READY = 5   # vector -> cube: that bridge is filled
E_RESULTS_IN_UB = 6     # cube -> vector: the five accumulators are in UB
E_RESULTS_UB_FREE = 7   # vector -> cube: their landing tiles are consumed

# Some PyPTO builds expose part of the DT_* enum as bare integers, which the
# vector-function layer rejects. Rebuild the affected ones at module scope: the
# parser inlines helper calls, so a decorated body must see plain names.
_DATA_TYPE = type(pl.DT_FP32)


def _as_data_type(value):
    return value if isinstance(value, _DATA_TYPE) else _DATA_TYPE(value)


_PL_FP32 = _as_data_type(pl.DT_FP32)
_PL_INT32 = _as_data_type(pl.DT_INT32)


@pl.vector_function
def widen_grad_o_vf(grad_o_bf16_tile, grad_o_fp32_tile, token_count, col_count):
    """Widen grad_o from bfloat16 to float32 and zero the rows past the tail."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for token in pl.range(0, token_count, 1):
        narrow = vf.load_align(grad_o_bf16_tile, token * TILE,
                               dist=pl.LoadDist.UNPK_B16)
        wide = vf.astype(narrow, all_lanes, dtype=_PL_FP32,
                         layout=pl.CastLayout.ZERO)
        vf.store_align(grad_o_fp32_tile + token * TILE, wide, all_lanes)
    for pad_token in pl.range(token_count, C, 1):
        vf.store_align(grad_o_fp32_tile + pad_token * TILE, zero, all_lanes)


@pl.vector_function
def scale_constants_vf(consts_tile, Kd):
    """Store sqrt(K) at lane 0 and 1/sqrt(K) at lane 1."""
    lanes_fp32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lanes_int32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    k_int = vf.full(Kd, lanes_int32, dtype=_PL_INT32)
    k_fp32 = vf.astype(k_int, lanes_fp32, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    sqrt_k = vf.sqrt(k_fp32, lanes_fp32)
    one = vf.full(1.0, lanes_fp32, dtype=_PL_FP32)
    scale = vf.div(one, sqrt_k, lanes_fp32)
    vf.store_align(consts_tile + 0, sqrt_k, lanes_fp32,
                   dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(consts_tile + 1, scale, lanes_fp32,
                   dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def decay_exp_vf(decay_tile, decay_exp_tile):
    """decay_exp = exp(decay), one chunk row."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    decay = vf.load_align(decay_tile, 0, dtype=_PL_FP32)
    decay_exp = vf.exp(decay, all_lanes)
    vf.store_align(decay_exp_tile + 0, decay_exp, all_lanes)


@pl.vector_function
def gate_and_mask_scores_vf(decay_tile, scores_tile, consts_tile):
    """scores[i, j] *= exp(decay_i - decay_j) * scale for j <= i, else 0.

    The mask is applied to the exponential before it reaches the score row, so
    the discarded upper triangle never produces an infinity to cancel later.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    all_lanes_i32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    col_index = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    decay_row = vf.load_align(decay_tile, 0, dtype=_PL_FP32)
    scale = vf.load_align(consts_tile, 1, dist=pl.LoadDist.BRC_B32)
    for row in pl.range(0, C, 1):
        decay_i = vf.load_align(decay_tile, row, dist=pl.LoadDist.BRC_B32)
        gate = vf.exp_sub(decay_i, decay_row, all_lanes)
        keep = vf.le(col_index, row, all_lanes_i32)
        gate = vf.select(gate, zero, keep)
        gate = vf.mul(gate, scale, all_lanes)
        scores = vf.load_align(scores_tile, row * C, dtype=_PL_FP32)
        scores = vf.mul(scores, gate, all_lanes)
        vf.store_align(scores_tile + row * C, scores, all_lanes)


@pl.vector_function
def assemble_gradients_vf(dq_inter_tile, dq_intra_tile, dk_inter_tile, dk_intra_tile,
                          dw_tile, q_tile, k_tile, decay_exp_tile, decay_tile,
                          consts_tile, dg_tile, dg_last_tile, k_count):
    """Combine the five matmul results into dq, dk, dw and reduce dg.

    dq and dk are written back over their intra-chunk tiles, dw is negated in
    place, dg accumulates into one lane per token, and the per-chunk gate scalar
    leaves one partial per lane for a later reduction.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    all_lanes_i32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    col_index = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    live_k_lanes = vf.lt(col_index, k_count, all_lanes_i32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    scale = vf.load_align(consts_tile, 1, dist=pl.LoadDist.BRC_B32)
    sqrt_k = vf.load_align(consts_tile, 0, dist=pl.LoadDist.BRC_B32)
    gate_last = vf.load_align(decay_tile, C - 1, dist=pl.LoadDist.BRC_B32)
    for token in pl.range(0, C, 1):
        offset = token * TILE
        q_row = vf.load_align(q_tile, offset, dtype=_PL_FP32)
        k_row = vf.load_align(k_tile, offset, dtype=_PL_FP32)

        # dq = dq_inter * exp(decay) * scale + dq_intra
        decay_exp = vf.load_align(decay_exp_tile, token, dist=pl.LoadDist.BRC_B32)
        decay_exp = vf.mul(decay_exp, scale, all_lanes)
        dq = vf.load_align(dq_inter_tile, offset, dtype=_PL_FP32)
        dq_intra = vf.load_align(dq_intra_tile, offset, dtype=_PL_FP32)
        dq = vf.mul(dq, decay_exp, all_lanes)
        dq = vf.add(dq, dq_intra, all_lanes)
        vf.store_align(dq_intra_tile + offset, dq, all_lanes)

        # dk = dk_inter * exp(g_last - decay) + dk_intra * sqrt(K)
        decay_i = vf.load_align(decay_tile, token, dist=pl.LoadDist.BRC_B32)
        gate = vf.exp_sub(gate_last, decay_i, all_lanes)
        dk_inter = vf.load_align(dk_inter_tile, offset, dtype=_PL_FP32)
        dk_inter = vf.mul(dk_inter, gate, all_lanes)
        vf.store_align(dk_inter_tile + offset, dk_inter, all_lanes)
        dk_intra = vf.load_align(dk_intra_tile, offset, dtype=_PL_FP32)
        dk_intra = vf.mul(dk_intra, sqrt_k, all_lanes)
        dk = vf.add(dk_inter, dk_intra, all_lanes)
        vf.store_align(dk_intra_tile + offset, dk, all_lanes)

        dw_row = vf.load_align(dw_tile, offset, dtype=_PL_FP32)
        dw_row = vf.neg(dw_row, all_lanes)
        vf.store_align(dw_tile + offset, dw_row, all_lanes)

        # dg = dg + sqrt(K) * sum(dq * q_scaled) - sum(dk * k)
        masked = vf.mul(dq, q_row, all_lanes)
        masked = vf.select(masked, zero, live_k_lanes)
        dq_dot_q = vf.reduce_sum(masked, all_lanes)
        dq_dot_q = vf.mul(dq_dot_q, sqrt_k, all_lanes)
        masked = vf.mul(dk, k_row, all_lanes)
        masked = vf.select(masked, zero, live_k_lanes)
        dk_dot_k = vf.reduce_sum(masked, all_lanes)
        dg = vf.load_align(dg_tile, token, dist=pl.LoadDist.BRC_B32)
        dg = vf.add(dg, dq_dot_q, all_lanes)
        dg = vf.sub(dg, dk_dot_k, all_lanes)
        vf.store_align(dg_tile + token, dg, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)

        # One partial per lane of the chunk scalar; reduced after this loop.
        masked = vf.mul(dk_inter, k_row, all_lanes)
        masked = vf.select(masked, zero, live_k_lanes)
        partial = vf.reduce_sum(masked, all_lanes)
        vf.store_align(dg_last_tile + token, partial, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def state_dot_dstate_vf(state_tile, dstate_tile, partials_tile, row_count, col_count):
    """One partial per row of sum over V of (S * dS), zeroing the unused lanes."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    all_lanes_i32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    col_index = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    live_v_lanes = vf.lt(col_index, col_count, all_lanes_i32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for row in pl.range(0, row_count, 1):
        state = vf.load_align(state_tile, row * TILE, dtype=_PL_FP32)
        dstate = vf.load_align(dstate_tile, row * TILE, dtype=_PL_FP32)
        masked = vf.mul(state, dstate, all_lanes)
        masked = vf.select(masked, zero, live_v_lanes)
        partial = vf.reduce_sum(masked, all_lanes)
        vf.store_align(partials_tile + row, partial, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)
    for pad_row in pl.range(row_count, C, 1):
        vf.store_align(partials_tile + pad_row, zero, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def accumulate_partials_vf(partials_tile, scalar_tile):
    """Reduce the per-lane partials and accumulate into lane 0 of `scalar_tile`.

    Called once per producing loop, so its single read-modify-write carries no
    cross-iteration ordering hazard.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    partials = vf.load_align(partials_tile, 0, dtype=_PL_FP32)
    total = vf.reduce_sum(partials, all_lanes)
    previous = vf.load_align(scalar_tile, 0, dist=pl.LoadDist.BRC_B32)
    total = vf.add(previous, total, all_lanes)
    vf.store_align(scalar_tile + 0, total, all_lanes,
                   dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def fold_chunk_gate_vf(dg_tile, dg_last_tile, state_dot_tile, decay_tile):
    """Fold the per-chunk gate scalar onto the last token of the chunk."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    gate_last = vf.load_align(decay_tile, C - 1, dist=pl.LoadDist.BRC_B32)
    gate_exp = vf.exp(gate_last, all_lanes)
    state_dot = vf.load_align(state_dot_tile, 0, dist=pl.LoadDist.BRC_B32)
    dg_last = vf.mul(gate_exp, state_dot, all_lanes)
    dg_last_partial = vf.load_align(dg_last_tile, 0, dist=pl.LoadDist.BRC_B32)
    dg_last = vf.add(dg_last, dg_last_partial, all_lanes)
    dg = vf.load_align(dg_tile, C - 1, dist=pl.LoadDist.BRC_B32)
    dg = vf.add(dg, dg_last, all_lanes)
    vf.store_align(dg_tile + (C - 1), dg, all_lanes,
                   dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def zero_row_vf(tile):
    """Zero one full register row of a tile."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    vf.store_align(tile + 0, zero, all_lanes)


# The batch axis is pinned to 1: this operator supports B = 1 only.
_B1_DYN4 = [1] + [pl.DYNAMIC] * 4
_B1_DYN3 = [1] + [pl.DYNAMIC] * 3


@pl.jit(auto_mutex=True)
def gdr_bwd_dqkw_kernel(
    v_new: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    k_f32: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    q_s: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    s_hist: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    dh: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    dv2: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    grad_o_gm: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    scores_gm: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    grad_o: pl.Tensor[_B1_DYN3, pl.DT_BF16],
    decay: pl.Tensor[_B1_DYN3, pl.DT_FP32],
    dq_p: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    dk_p: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    dw: pl.Tensor[_B1_DYN4, pl.DT_FP32],
    dg_p: pl.Tensor[_B1_DYN3, pl.DT_FP32],
    B: pl.DT_INT32, T: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32,
    Vd: pl.DT_INT32, N: pl.DT_INT32,
):
    # L0A and L0B each serve every phase from the same two addresses; the phases
    # are separated by vector work, so the slots are reused off peak and both
    # sit at their 64 KB cap.
    g_l0a_wide = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1]), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g_l0a_square = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1]), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g_l0b_tall = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1]), addrs=[0x0, 0x8000], mutex_ids=[2, 3])
    g_l0b_square = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1]), addrs=[0x0, 0x8000], mutex_ids=[2, 3])

    # UB, 238,944 B of the 253,952 B budget.
    g_grad_o_bf16_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x0, mutex_ids=[4])
    g_grad_o_fp32_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x2000, mutex_ids=[5])
    g_scores_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x6000, mutex_ids=[6])
    g_dstate_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0xa000, mutex_ids=[6])
    g_decay_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0xe000, mutex_ids=[7])
    g_decay_exp_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0xe100, mutex_ids=[8])
    g_consts_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0xe200, mutex_ids=[8])
    g_dq_inter_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0xe220, mutex_ids=[9, 10])
    g_dq_intra_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x16220, mutex_ids=[11, 12])
    g_dk_inter_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x1e220, mutex_ids=[9, 10])
    g_dk_intra_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x26220, mutex_ids=[11, 12])
    g_dw_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x2e220, mutex_ids=[11, 12])
    g_k_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x36220, mutex_ids=[13])
    g_dg_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x3a220, mutex_ids=[14])
    # Per-lane partials and their reduced scalars, for the two accumulations
    # that cannot be done in place; see the module docstring.
    g_dg_last_partials_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x3a320, mutex_ids=[8])
    g_state_dot_partials_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x3a420, mutex_ids=[8])
    g_dg_last_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x3a520, mutex_ids=[8])
    g_state_dot_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x3a540, mutex_ids=[8])

    # L1. The transposed operands are filled by transposing loads, so no
    # pre-transposed copy is kept in GM.
    g_v_new_t_l1 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x8000, mutex_ids=[15])
    g_grad_o_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x10000, mutex_ids=[15])
    g_state_t_l1 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x18000, mutex_ids=[17, 18])
    g_k_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x28000, mutex_ids=[15])
    g_v_new_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x30000, mutex_ids=[15])
    g_dstate_t_l1 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x38000, mutex_ids=[30, 31])
    g_dv2_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x48000, mutex_ids=[15])
    g_scores_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x50000, mutex_ids=[19])
    g_scores_t_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.ZN, valid_shape=[-1, -1]), addrs=0x54000, mutex_ids=[19])
    g_q_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x58000, mutex_ids=[15])

    # One accumulator per output, each with its own mutex pair so that a MAC and
    # another output's fixpipe drain do not serialise against each other.
    g_scores_acc = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x0, mutex_ids=[16])
    g_dq_inter_acc = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x8000, mutex_ids=[20, 21])
    g_dq_intra_acc = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x10000, mutex_ids=[22, 23])
    g_dk_inter_acc = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x18000, mutex_ids=[24, 25])
    g_dk_intra_acc = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x20000, mutex_ids=[26, 27])
    g_dw_acc = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x28000, mutex_ids=[28, 29])

    with pl.section_cube():
        pl.system.set_cross_core(pipe=_FIX, event_id=E_GRAD_O_GM_FREE, sync_mode=_IB)
        pl.system.set_cross_core(pipe=_FIX, event_id=E_SCORES_GM_FREE, sync_mode=_IB)
        scores_ub = g_scores_ub.next()
        v_new_t_l1 = g_v_new_t_l1.next()
        scores_acc = g_scores_acc.next()
        dv2_l1 = g_dv2_l1.next()
        scores_t_l1 = g_scores_t_l1.next()

        task_count = (B * H) * N
        tasks_per_core = (task_count + pl.get_block_num() - 1) // pl.get_block_num()
        first_task = tasks_per_core * (pl.get_block_idx() // pl.get_subblock_num())
        for tid in pl.range(first_task, pl.min(first_task + tasks_per_core, task_count), 1):
            # Decode once per item rather than at each of the 29 transfer sites.
            b = tid // (H * N)
            within_batch = tid % (H * N)
            h = within_batch // N
            n = within_batch % N

            # Rotate the per-item L1 operands so that the next item's fetch can
            # issue while this item's matmuls still read the other buffer.
            grad_o_l1 = g_grad_o_l1.next()
            k_l1 = g_k_l1.next()
            v_new_l1 = g_v_new_l1.next()
            scores_l1 = g_scores_l1.next()
            q_l1 = g_q_l1.next()

            pl.set_validshape(v_new_t_l1, [Vd, C])
            pl.load(v_new_t_l1, v_new, [b, h, n, 0, 0], order=[4, 3])
            pl.set_validshape(v_new_l1, [C, Vd])
            pl.load(v_new_l1, v_new, [b, h, n, 0, 0])
            # k, dv2 and q come straight from GM inputs with no vector
            # dependency, so they are issued before the first rendezvous.
            pl.set_validshape(k_l1, [C, Kd])
            pl.load(k_l1, k_f32, [b, h, n, 0, 0])
            pl.set_validshape(dv2_l1, [C, Vd])
            pl.load(dv2_l1, dv2, [b, h, n, 0, 0])
            pl.set_validshape(q_l1, [C, Kd])
            pl.load(q_l1, q_s, [b, h, n, 0, 0])

            # scores = grad_o @ v_new^T
            pl.system.wait_cross_core(pipe=_S, event_id=E_GRAD_O_GM_READY, sync_mode=_IB)
            pl.set_validshape(grad_o_l1, [C, Vd])
            pl.load(grad_o_l1, grad_o_gm, [b, h, n, 0, 0])
            g_l0a_square.next()
            grad_o_l0a = g_l0a_wide.next()
            g_l0b_square.next()
            v_new_t_l0b = g_l0b_tall.next()
            pl.move(grad_o_l0a, grad_o_l1, [0, 0])
            pl.move(v_new_t_l0b, v_new_t_l1, [0, 0])
            pl.set_validshape(grad_o_l0a, [C, Vd])
            pl.set_validshape(v_new_t_l0b, [Vd, C])
            pl.set_validshape(scores_acc, [C, C])
            pl.matmul(scores_acc, grad_o_l0a, v_new_t_l0b)
            pl.system.set_cross_core(pipe=_FIX, event_id=E_GRAD_O_GM_FREE, sync_mode=_IB)

            pl.system.wait_cross_core(pipe=_S, event_id=E_SCORES_UB_FREE, sync_mode=_IB)
            pl.set_validshape(scores_acc, [C, C])
            pl.move(scores_ub, scores_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
            pl.system.set_cross_core(pipe=_FIX, event_id=E_SCORES_IN_UB, sync_mode=_IB)

            # First K block. The three matmuls that do not need the masked
            # scores run first, then the scores rendezvous, then the two that do.
            if Kd > 0:
                state_t_l1 = g_state_t_l1.next()
                dstate_t_l1 = g_dstate_t_l1.next()
                pl.set_validshape(state_t_l1, [Vd, pl.min(Kd, TILE)])
                pl.load(state_t_l1, s_hist, [b, h, n, 0, 0], order=[4, 3])
                pl.set_validshape(dstate_t_l1, [Vd, pl.min(Kd, TILE)])
                pl.load(dstate_t_l1, dh, [b, h, n, 0, 0], order=[4, 3])

                dq_inter_acc = g_dq_inter_acc.next()
                g_l0a_square.next()
                grad_o_l0a = g_l0a_wide.next()
                g_l0b_square.next()
                state_t_l0b = g_l0b_tall.next()
                pl.move(grad_o_l0a, grad_o_l1, [0, 0])
                pl.move(state_t_l0b, state_t_l1, [0, 0])
                pl.set_validshape(grad_o_l0a, [C, Vd])
                pl.set_validshape(state_t_l0b, [Vd, C])
                pl.set_validshape(dq_inter_acc, [C, pl.min(Kd, TILE)])
                pl.matmul(dq_inter_acc, grad_o_l0a, state_t_l0b)

                dk_inter_acc = g_dk_inter_acc.next()
                g_l0a_square.next()
                v_new_l0a = g_l0a_wide.next()
                g_l0b_square.next()
                dstate_t_l0b = g_l0b_tall.next()
                pl.move(v_new_l0a, v_new_l1, [0, 0])
                pl.move(dstate_t_l0b, dstate_t_l1, [0, 0])
                pl.set_validshape(v_new_l0a, [C, Vd])
                pl.set_validshape(dstate_t_l0b, [Vd, C])
                pl.set_validshape(dk_inter_acc, [C, pl.min(Kd, TILE)])
                pl.matmul(dk_inter_acc, v_new_l0a, dstate_t_l0b)

                dw_acc = g_dw_acc.next()
                g_l0a_square.next()
                dv2_l0a = g_l0a_wide.next()
                g_l0b_square.next()
                state_t_l0b = g_l0b_tall.next()
                pl.move(dv2_l0a, dv2_l1, [0, 0])
                pl.move(state_t_l0b, state_t_l1, [0, 0])
                pl.set_validshape(dv2_l0a, [C, Vd])
                pl.set_validshape(state_t_l0b, [Vd, C])
                pl.set_validshape(dw_acc, [C, pl.min(Kd, TILE)])
                pl.matmul(dw_acc, dv2_l0a, state_t_l0b)

                # The masked scores arrive once per item, in both orientations:
                # the transposed read is what replaces a second matmul.
                pl.system.wait_cross_core(pipe=_S, event_id=E_SCORES_GM_READY, sync_mode=_IB)
                pl.set_validshape(scores_l1, [C, C])
                pl.load(scores_l1, scores_gm, [b, h, n, 0, 0])
                pl.set_validshape(scores_t_l1, [C, C])
                pl.load(scores_t_l1, scores_gm, [b, h, n, 0, 0], order=[4, 3])
                pl.system.set_cross_core(pipe=_FIX, event_id=E_SCORES_GM_FREE, sync_mode=_IB)

                dq_intra_acc = g_dq_intra_acc.next()
                g_l0a_wide.next()
                scores_l0a = g_l0a_square.next()
                g_l0b_tall.next()
                k_l0b = g_l0b_square.next()
                pl.move(scores_l0a, scores_l1, [0, 0])
                pl.move(k_l0b, k_l1, [0, 0])
                pl.set_validshape(scores_l0a, [C, C])
                pl.set_validshape(k_l0b, [C, C])
                pl.set_validshape(dq_intra_acc, [C, pl.min(Kd, TILE)])
                pl.matmul(dq_intra_acc, scores_l0a, k_l0b)

                dk_intra_acc = g_dk_intra_acc.next()
                g_l0a_wide.next()
                scores_t_l0a = g_l0a_square.next()
                g_l0b_tall.next()
                q_l0b = g_l0b_square.next()
                pl.move(scores_t_l0a, scores_t_l1, [0, 0])
                pl.move(q_l0b, q_l1, [0, 0])
                pl.set_validshape(scores_t_l0a, [C, C])
                pl.set_validshape(q_l0b, [C, C])
                pl.set_validshape(dk_intra_acc, [C, pl.min(Kd, TILE)])
                pl.matmul(dk_intra_acc, scores_t_l0a, q_l0b)

                pl.system.wait_cross_core(pipe=_S, event_id=E_RESULTS_UB_FREE, sync_mode=_IB)
                dq_inter_ub = g_dq_inter_ub.next()
                pl.set_validshape(dq_inter_acc, [C, pl.min(Kd, TILE)])
                pl.move(dq_inter_ub, dq_inter_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                dq_intra_ub = g_dq_intra_ub.next()
                pl.set_validshape(dq_intra_acc, [C, pl.min(Kd, TILE)])
                pl.move(dq_intra_ub, dq_intra_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                dk_inter_ub = g_dk_inter_ub.next()
                pl.set_validshape(dk_inter_acc, [C, pl.min(Kd, TILE)])
                pl.move(dk_inter_ub, dk_inter_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                dk_intra_ub = g_dk_intra_ub.next()
                pl.set_validshape(dk_intra_acc, [C, pl.min(Kd, TILE)])
                pl.move(dk_intra_ub, dk_intra_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                dw_ub = g_dw_ub.next()
                pl.set_validshape(dw_acc, [C, pl.min(Kd, TILE)])
                pl.move(dw_ub, dw_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                pl.system.set_cross_core(pipe=_FIX, event_id=E_RESULTS_IN_UB, sync_mode=_IB)

            # Second K block, only when K exceeds one tile. The masked scores
            # are already in L1, so all five matmuls run back to back.
            if Kd > TILE:
                state_t_l1 = g_state_t_l1.next()
                k_count_hi = pl.min(Kd - TILE, TILE)
                pl.set_validshape(state_t_l1, [Vd, k_count_hi])
                pl.load(state_t_l1, s_hist, [b, h, n, TILE, 0], order=[4, 3])
                dstate_t_l1 = g_dstate_t_l1.next()
                pl.set_validshape(dstate_t_l1, [Vd, k_count_hi])
                pl.load(dstate_t_l1, dh, [b, h, n, TILE, 0], order=[4, 3])

                dq_inter_acc = g_dq_inter_acc.next()
                g_l0a_square.next()
                grad_o_l0a = g_l0a_wide.next()
                g_l0b_square.next()
                state_t_l0b = g_l0b_tall.next()
                pl.move(grad_o_l0a, grad_o_l1, [0, 0])
                pl.move(state_t_l0b, state_t_l1, [0, 0])
                pl.set_validshape(grad_o_l0a, [C, Vd])
                pl.set_validshape(state_t_l0b, [Vd, C])
                pl.set_validshape(dq_inter_acc, [C, k_count_hi])
                pl.matmul(dq_inter_acc, grad_o_l0a, state_t_l0b)

                dq_intra_acc = g_dq_intra_acc.next()
                g_l0a_wide.next()
                scores_l0a = g_l0a_square.next()
                g_l0b_tall.next()
                k_l0b = g_l0b_square.next()
                pl.move(scores_l0a, scores_l1, [0, 0])
                pl.move(k_l0b, k_l1, [0, TILE])
                pl.set_validshape(scores_l0a, [C, C])
                pl.set_validshape(k_l0b, [C, C])
                pl.set_validshape(dq_intra_acc, [C, k_count_hi])
                pl.matmul(dq_intra_acc, scores_l0a, k_l0b)

                dk_inter_acc = g_dk_inter_acc.next()
                g_l0a_square.next()
                v_new_l0a = g_l0a_wide.next()
                g_l0b_square.next()
                dstate_t_l0b = g_l0b_tall.next()
                pl.move(v_new_l0a, v_new_l1, [0, 0])
                pl.move(dstate_t_l0b, dstate_t_l1, [0, 0])
                pl.set_validshape(v_new_l0a, [C, Vd])
                pl.set_validshape(dstate_t_l0b, [Vd, C])
                pl.set_validshape(dk_inter_acc, [C, k_count_hi])
                pl.matmul(dk_inter_acc, v_new_l0a, dstate_t_l0b)

                dk_intra_acc = g_dk_intra_acc.next()
                g_l0a_wide.next()
                scores_t_l0a = g_l0a_square.next()
                g_l0b_tall.next()
                q_l0b = g_l0b_square.next()
                pl.move(scores_t_l0a, scores_t_l1, [0, 0])
                pl.move(q_l0b, q_l1, [0, TILE])
                pl.set_validshape(scores_t_l0a, [C, C])
                pl.set_validshape(q_l0b, [C, C])
                pl.set_validshape(dk_intra_acc, [C, k_count_hi])
                pl.matmul(dk_intra_acc, scores_t_l0a, q_l0b)

                dw_acc = g_dw_acc.next()
                g_l0a_square.next()
                dv2_l0a = g_l0a_wide.next()
                g_l0b_square.next()
                state_t_l0b = g_l0b_tall.next()
                pl.move(dv2_l0a, dv2_l1, [0, 0])
                pl.move(state_t_l0b, state_t_l1, [0, 0])
                pl.set_validshape(dv2_l0a, [C, Vd])
                pl.set_validshape(state_t_l0b, [Vd, C])
                pl.set_validshape(dw_acc, [C, k_count_hi])
                pl.matmul(dw_acc, dv2_l0a, state_t_l0b)

                pl.system.wait_cross_core(pipe=_S, event_id=E_RESULTS_UB_FREE, sync_mode=_IB)
                dq_inter_ub = g_dq_inter_ub.next()
                pl.set_validshape(dq_inter_acc, [C, k_count_hi])
                pl.move(dq_inter_ub, dq_inter_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                dq_intra_ub = g_dq_intra_ub.next()
                pl.set_validshape(dq_intra_acc, [C, k_count_hi])
                pl.move(dq_intra_ub, dq_intra_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                dk_inter_ub = g_dk_inter_ub.next()
                pl.set_validshape(dk_inter_acc, [C, k_count_hi])
                pl.move(dk_inter_ub, dk_inter_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                dk_intra_ub = g_dk_intra_ub.next()
                pl.set_validshape(dk_intra_acc, [C, k_count_hi])
                pl.move(dk_intra_ub, dk_intra_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                dw_ub = g_dw_ub.next()
                pl.set_validshape(dw_acc, [C, k_count_hi])
                pl.move(dw_ub, dw_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                pl.system.set_cross_core(pipe=_FIX, event_id=E_RESULTS_IN_UB, sync_mode=_IB)

        pl.system.wait_cross_core(pipe=_S, event_id=E_SCORES_UB_FREE, sync_mode=_IB)
        pl.system.wait_cross_core(pipe=_S, event_id=E_RESULTS_UB_FREE, sync_mode=_IB)

    with pl.section_vector():
        pl.system.set_cross_core(pipe=_V, event_id=E_SCORES_UB_FREE, sync_mode=_IB)
        pl.system.set_cross_core(pipe=_V, event_id=E_RESULTS_UB_FREE, sync_mode=_IB)
        grad_o_bf16_ub = g_grad_o_bf16_ub.next()
        grad_o_fp32_ub = g_grad_o_fp32_ub.next()
        scores_ub = g_scores_ub.next()
        dstate_ub = g_dstate_ub.next()
        decay_ub = g_decay_ub.next()
        decay_exp_ub = g_decay_exp_ub.next()
        consts_ub = g_consts_ub.next()
        k_ub = g_k_ub.next()
        dg_ub = g_dg_ub.next()
        dg_last_partials_ub = g_dg_last_partials_ub.next()
        state_dot_partials_ub = g_state_dot_partials_ub.next()
        dg_last_ub = g_dg_last_ub.next()
        state_dot_ub = g_state_dot_ub.next()

        task_count = (B * H) * N
        tasks_per_core = (task_count + pl.get_block_num() - 1) // pl.get_block_num()
        first_task = tasks_per_core * (pl.get_block_idx() // pl.get_subblock_num())
        for tid in pl.range(first_task, pl.min(first_task + tasks_per_core, task_count), 1):
            pl.system.wait_cross_core(pipe=_S, event_id=E_GRAD_O_GM_FREE, sync_mode=_IB)
            b = tid // (H * N)
            within_batch = tid % (H * N)
            h = within_batch // N
            n = within_batch % N
            token_base = n * C
            token_count = pl.min(T - token_base, C)
            # All vector work runs on sub-block 0; this offset is zero there.
            subblock_base = pl.get_subblock_idx() * TILE

            # Widen grad_o into the GM bridge the cube reads.
            if subblock_base < TILE:
                for v_block in pl.range(0, (Vd + TILE - 1) // TILE, 1):
                    v_count = pl.min(Vd - v_block * TILE, TILE)
                    if token_count > 0:
                        # grad_o is the one public [B, T, H, V] input read with
                        # a 2-D tile: order= makes it walk (T, V) instead of the
                        # default trailing (H, V).
                        pl.set_validshape(grad_o_bf16_ub, [token_count, v_count])
                        pl.load(grad_o_bf16_ub, grad_o,
                                [b, token_base, h, v_block * TILE], order=[1, 3])
                    widen_grad_o_vf(grad_o_bf16_ub, grad_o_fp32_ub, token_count, v_count)
                    pl.set_validshape(grad_o_fp32_ub, [C, v_count])
                    pl.store(grad_o_gm, grad_o_fp32_ub, [b, h, n, 0, v_block * TILE])
            pl.system.set_cross_core(pipe=_MTE3, event_id=E_GRAD_O_GM_READY, sync_mode=_IB)

            if subblock_base < TILE:
                pl.set_validshape(decay_ub, [1, C])
                pl.load(decay_ub, decay, [b, h, n, 0])

            # Gate and mask the scores, and publish them for both orientations.
            pl.system.wait_cross_core(pipe=_S, event_id=E_SCORES_IN_UB, sync_mode=_IB)
            if subblock_base < TILE:
                scale_constants_vf(consts_ub, Kd)
                decay_exp_vf(decay_ub, decay_exp_ub)
                gate_and_mask_scores_vf(decay_ub, scores_ub, consts_ub)
                pl.set_validshape(scores_ub, [C, C])
                pl.store(scores_gm, scores_ub, [b, h, n, 0, 0])
            pl.system.set_cross_core(pipe=_V, event_id=E_SCORES_UB_FREE, sync_mode=_IB)

            pl.system.wait_cross_core(pipe=_S, event_id=E_SCORES_GM_FREE, sync_mode=_IB)
            if subblock_base < TILE:
                zero_row_vf(dg_ub)
                zero_row_vf(dg_last_partials_ub)
                zero_row_vf(state_dot_partials_ub)
                zero_row_vf(dg_last_ub)
                zero_row_vf(state_dot_ub)
            pl.system.set_cross_core(pipe=_MTE3, event_id=E_SCORES_GM_READY, sync_mode=_IB)

            # Assemble the outputs, one K block at a time. grad_o_fp32_ub is
            # reused here to hold q, its bridge contents having been consumed.
            if Kd > 0:
                if subblock_base < TILE:
                    pl.set_validshape(grad_o_fp32_ub, [C, pl.min(Kd, TILE)])
                    pl.load(grad_o_fp32_ub, q_s, [b, h, n, 0, 0])
                    pl.set_validshape(k_ub, [C, pl.min(Kd, TILE)])
                    pl.load(k_ub, k_f32, [b, h, n, 0, 0])
                pl.system.wait_cross_core(pipe=_S, event_id=E_RESULTS_IN_UB, sync_mode=_IB)
                dq_inter_ub = g_dq_inter_ub.next()
                dq_intra_ub = g_dq_intra_ub.next()
                dk_inter_ub = g_dk_inter_ub.next()
                dk_intra_ub = g_dk_intra_ub.next()
                dw_ub = g_dw_ub.next()
                if subblock_base < TILE:
                    assemble_gradients_vf(dq_inter_ub, dq_intra_ub, dk_inter_ub,
                                          dk_intra_ub, dw_ub, grad_o_fp32_ub, k_ub,
                                          decay_exp_ub, decay_ub, consts_ub,
                                          dg_ub, dg_last_partials_ub, pl.min(Kd, TILE))
                    accumulate_partials_vf(dg_last_partials_ub, dg_last_ub)
                    pl.set_validshape(dq_intra_ub, [C, pl.min(Kd, TILE)])
                    pl.store(dq_p, dq_intra_ub, [b, h, n, 0, 0])
                    pl.set_validshape(dk_intra_ub, [C, pl.min(Kd, TILE)])
                    pl.store(dk_p, dk_intra_ub, [b, h, n, 0, 0])
                    pl.set_validshape(dw_ub, [C, pl.min(Kd, TILE)])
                    pl.store(dw, dw_ub, [b, h, n, 0, 0])
                pl.system.set_cross_core(pipe=_V, event_id=E_RESULTS_UB_FREE, sync_mode=_IB)

            if Kd > TILE:
                if subblock_base < TILE:
                    pl.set_validshape(grad_o_fp32_ub, [C, pl.min(Kd - TILE, TILE)])
                    pl.load(grad_o_fp32_ub, q_s, [b, h, n, 0, TILE])
                    pl.set_validshape(k_ub, [C, pl.min(Kd - TILE, TILE)])
                    pl.load(k_ub, k_f32, [b, h, n, 0, TILE])
                pl.system.wait_cross_core(pipe=_S, event_id=E_RESULTS_IN_UB, sync_mode=_IB)
                dq_inter_ub = g_dq_inter_ub.next()
                dq_intra_ub = g_dq_intra_ub.next()
                dk_inter_ub = g_dk_inter_ub.next()
                dk_intra_ub = g_dk_intra_ub.next()
                dw_ub = g_dw_ub.next()
                if subblock_base < TILE:
                    assemble_gradients_vf(dq_inter_ub, dq_intra_ub, dk_inter_ub,
                                          dk_intra_ub, dw_ub, grad_o_fp32_ub, k_ub,
                                          decay_exp_ub, decay_ub, consts_ub,
                                          dg_ub, dg_last_partials_ub,
                                          pl.min(Kd - TILE, TILE))
                    accumulate_partials_vf(dg_last_partials_ub, dg_last_ub)
                    pl.set_validshape(dq_intra_ub, [C, pl.min(Kd - TILE, TILE)])
                    pl.store(dq_p, dq_intra_ub, [b, h, n, 0, TILE])
                    pl.set_validshape(dk_intra_ub, [C, pl.min(Kd - TILE, TILE)])
                    pl.store(dk_p, dk_intra_ub, [b, h, n, 0, TILE])
                    pl.set_validshape(dw_ub, [C, pl.min(Kd - TILE, TILE)])
                    pl.store(dw, dw_ub, [b, h, n, 0, TILE])
                pl.system.set_cross_core(pipe=_V, event_id=E_RESULTS_UB_FREE, sync_mode=_IB)

            # The state dot product, then fold the chunk scalar onto token C-1.
            # The scores tile stages the state here, its contents being spent.
            if subblock_base < TILE:
                for k_block in pl.range(0, (Kd + TILE - 1) // TILE, 1):
                    for v_block in pl.range(0, (Vd + TILE - 1) // TILE, 1):
                        k_count = pl.min(Kd - k_block * TILE, TILE)
                        v_count = pl.min(Vd - v_block * TILE, TILE)
                        pl.set_validshape(scores_ub, [k_count, v_count])
                        pl.load(scores_ub, s_hist,
                                [b, h, n, k_block * TILE, v_block * TILE])
                        pl.set_validshape(dstate_ub, [k_count, v_count])
                        pl.load(dstate_ub, dh,
                                [b, h, n, k_block * TILE, v_block * TILE])
                        state_dot_dstate_vf(scores_ub, dstate_ub,
                                            state_dot_partials_ub, k_count, v_count)
                        accumulate_partials_vf(state_dot_partials_ub, state_dot_ub)
                fold_chunk_gate_vf(dg_ub, dg_last_ub, state_dot_ub, decay_ub)
                pl.set_validshape(dg_ub, [1, C])
                pl.store(dg_p, dg_ub, [b, h, n, 0])

        pl.system.wait_cross_core(pipe=_S, event_id=E_GRAD_O_GM_FREE, sync_mode=_IB)
        pl.system.wait_cross_core(pipe=_S, event_id=E_SCORES_GM_FREE, sync_mode=_IB)


def gdr_bwd_dqkw_wrapper(grad_o, v_new, k_f32, q_s, s_hist, dh, dv2,
                         g_f32=None, decay=None, *, sync=True):
    """Compute dq, dk, dw and the chunk-parallel part of dg in one launch.

    `g_f32` is part of the operator's public signature but is not used by this
    operator's arithmetic, so it may be omitted.

    Returns (dq_p, dk_p, dw, dg_p), all float32 and all in chunk layout.
    """
    B, T, H, V = grad_o.shape
    K = k_f32.shape[-1]
    N = (T + C - 1) // C
    device = grad_o.device

    # Kernel-internal bridges: the widened grad_o and the masked scores, both
    # produced by the vector core and consumed as cube operands.
    grad_o_gm = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    scores_gm = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)

    dq_p = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    dk_p = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    dw = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    dg_p = torch.empty((B, H, N, C), dtype=torch.float32, device=device)

    max_blocks = get_platform_info().core_num or 1
    num_cores = max(1, min(B * H * N, max_blocks))

    gdr_bwd_dqkw_kernel[None, num_cores](
        v_new, k_f32, q_s, s_hist, dh, dv2,
        grad_o_gm, scores_gm,
        grad_o, decay,
        dq_p, dk_p, dw, dg_p,
        B, T, H, K, V, N)
    if sync:
        torch.npu.synchronize()
    return dq_p, dk_p, dw, dg_p
