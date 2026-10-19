# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Within-chunk contribution to dv.

First of the six backward operators and the only one that needs no state
recurrence: every chunk is independent. The cross-chunk part of dv is added
later by the reverse state scan.

Mathematics
-----------
Per chunk, over token indices i and j in [0, C), with decay the within-chunk
cumulative gate sum::

    scores[i, j] = (k_i . q_j) * exp(decay_j - decay_i)   for j >= i, else 0
    dv_local     = scores @ grad_o_chunk

The dot products come from k_f32 [C, K] @ q_t [K, C], where q_t is the
pre-transposed q already multiplied by 1/sqrt(K) in the training forward, so no
scale factor appears here. The upper-triangular mask includes the diagonal: a
token's own value gradient belongs to the within-chunk part.

The mask is applied before the exponential, so the discarded half never produces
an infinity to cancel later.

Interface
---------
``gdr_bwd_dv_local_wrapper(grad_o, k_f32, q_t, g_f32)``

    grad_o          [B, T, H, V] bf16     incoming output gradient, public layout
    k_f32           [B, H, N, C, K] f32   k widened to float32
    q_t             [B, H, N, K, C] f32   pre-transposed, pre-scaled q
    g_f32           [B, H, N, C] f32      per-token log gate

    returns dv_local [B, H, N, C, V] f32

Layout and scheduling
---------------------
One launch of a cube - vector - cube chain inside a single kernel. A work item is
one (batch, head, chunk) triple; items are split into contiguous per-core runs.
The first cube phase forms the score matrix, the vector phase applies the gate
and the causal mask, and the second cube phase multiplies by grad_o.

Two details shape the code. The vector core cannot write an L1 operand directly,
so the masked scores and the widened grad_o are re-laid out to the NZ fractal
form in UB and inserted into L1 in eight-column slabs; each slab needs its own
compile-time tile group, and the kernel body admits only `pl.range` loops over
traced indices, so those inserts are written out. And the two cube operands for
the next item are prefetched into the second slot of a depth-2 L1 group right
after the current item's matmul is issued; the prefetch index is clamped to the
last item instead of being guarded by a branch, because an indexed group access
may not sit inside a traced branch.
"""

import logging

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

C = 64          # tokens per chunk, a frozen constant of the operator family
CH = 32         # rows per vector sub-block after a DualModeSplitM drain
KMAX = 128      # supported K upper bound, and the physical width of K tiles
VMAX = 128      # supported V upper bound, and the physical width of V tiles
NZ_SLAB = 8     # columns per NZ fractal slab on the UB to L1 bridge

_FIX = pl.PipeType.FIX
_S = pl.PipeType.S
_V = pl.PipeType.V
_MTE3 = pl.PipeType.MTE3
_IB = pl.CrossCoreSyncMode.INTRA_BLOCK

# Cross-core event ids; each pair guards one buffer handover.
E_SCORES_IN_UB = 0      # cube -> vector: raw scores drained from L0C to UB
E_SCORES_UB_FREE = 1    # vector -> cube: the scores landing tile is consumed
E_L1_OPERANDS_FREE = 2  # cube -> vector: both L1 operands have been read
E_L1_OPERANDS_READY = 3 # vector -> cube: both L1 operands are filled
E_DV_IN_UB = 4          # cube -> vector: dv_local drained from L0C to UB
E_DV_UB_FREE = 5        # vector -> cube: the dv landing tile is consumed

# Some PyPTO builds expose part of the DT_* enum as bare integers, which the
# vector-function layer rejects. Rebuild the affected ones at module scope: the
# parser inlines helper calls, so a decorated body must see plain names.
_DATA_TYPE = type(pl.DT_FP32)


def _as_data_type(value):
    return value if isinstance(value, _DATA_TYPE) else _DATA_TYPE(value)


_PL_FP32 = _as_data_type(pl.DT_FP32)
_PL_INT32 = _as_data_type(pl.DT_INT32)


@pl.vector_function
def widen_grad_o_vf(grad_o_bf16_tile, grad_o_fp32_tile, row_count):
    """Widen grad_o from bfloat16 to float32 and zero the rows past the tail."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for row in pl.range(0, row_count, 1):
        for col in pl.range(0, VMAX, 64):
            narrow = vf.load_align(grad_o_bf16_tile, row * VMAX + col,
                                   dist=pl.LoadDist.UNPK_B16)
            wide = vf.astype(narrow, all_lanes, dtype=_PL_FP32,
                             layout=pl.CastLayout.ZERO)
            vf.store_align(grad_o_fp32_tile + row * VMAX + col, wide, all_lanes)
    for pad_row in pl.range(row_count, CH, 1):
        for pad_col in pl.range(0, VMAX, 64):
            vf.store_align(grad_o_fp32_tile + pad_row * VMAX + pad_col, zero, all_lanes)


@pl.vector_function
def cumsum_decay_vf(gate_tile, decay_tile):
    """decay[t] = sum over t' <= t of gate[t'], one chunk."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    running = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for token in pl.range(0, C, 1):
        gate = vf.load_align(gate_tile, token, dist=pl.LoadDist.BRC_B32)
        running = vf.add(running, gate, all_lanes)
        vf.store_align(decay_tile + token, running, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def gate_and_mask_scores_vf(decay_tile, scores_tile, row_base):
    """scores[i, j] *= exp(decay_j - decay_i) for j >= i, else 0.

    Runs over the CH rows of one vector sub-block; `row_base` is that
    sub-block's first row within the chunk. Both decay terms are negated before
    `exp_sub` so that the result is exp(decay_j - decay_i) with the row term as
    the subtrahend, and the mask is applied to the exponential before it reaches
    the score row.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    all_lanes_i32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    col_index = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    decay_row = vf.load_align(decay_tile, 0, dtype=_PL_FP32)
    neg_decay_row = vf.neg(decay_row, all_lanes)
    for row in pl.range(0, CH, 1):
        token = row_base + row
        decay_i = vf.load_align(decay_tile, token, dist=pl.LoadDist.BRC_B32)
        neg_decay_i = vf.neg(decay_i, all_lanes)
        gate = vf.exp_sub(neg_decay_i, neg_decay_row, all_lanes)
        keep = vf.ge(col_index, token, all_lanes_i32)
        gate = vf.select(gate, zero, keep)
        scores = vf.load_align(scores_tile, row * C, dtype=_PL_FP32)
        scores = vf.mul(scores, gate, all_lanes)
        vf.store_align(scores_tile + row * C, scores, all_lanes)


_DYN5 = [pl.DYNAMIC] * 5
_DYN4 = [pl.DYNAMIC] * 4


@pl.jit(auto_mutex=True)
def gdr_bwd_dv_local_module1_kernel(
    k_f32: pl.Tensor[_DYN5, pl.DT_FP32],
    q_t: pl.Tensor[_DYN5, pl.DT_FP32],
    grad_o: pl.Tensor[_DYN4, pl.DT_BF16],
    g_f32: pl.Tensor[_DYN4, pl.DT_FP32],
    dv_local: pl.Tensor[_DYN5, pl.DT_FP32],
    B: pl.DT_INT32, T: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32,
    Vd: pl.DT_INT32, N: pl.DT_INT32, Cc: pl.DT_INT32,
):
    # L0A and L0B each serve both cube phases from the same two addresses: the
    # phases are separated by the vector phase, so the slots are reused off peak.
    g_k_l0a = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g_scores_l0a = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g_q_t_l0b = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right), addrs=[0x0, 0x8000], mutex_ids=[2, 3])
    g_grad_o_l0b = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right), addrs=[0x0, 0x8000], mutex_ids=[2, 3])

    # The two score-matmul operands are double buffered so that the next item's
    # refill is in flight under the current item's MAC. L1 map: k 2x32 KB,
    # q_t 2x32 KB, masked scores 16 KB, widened grad_o 32 KB.
    g_k_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x0, mutex_ids=[4, 22])
    g_q_t_l1 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x10000, mutex_ids=[5, 23])
    g_scores_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x20000, mutex_ids=[14])
    g_grad_o_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x24000, mutex_ids=[15])

    g_scores_acc = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=0x0, mutex_ids=[6, 7])
    g_dv_acc = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x8000, mutex_ids=[16, 17])

    g_scores_ub = pl.make_tile_group(type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x0, mutex_ids=[8, 9])
    g_gate_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x4000, mutex_ids=[10])
    g_decay_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x4100, mutex_ids=[11])
    g_grad_o_bf16_ub = pl.make_tile_group(type=pl.TileType(shape=[32, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=0x4200, mutex_ids=[12])
    g_grad_o_fp32_ub = pl.make_tile_group(type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x6200, mutex_ids=[13])
    g_dv_ub = pl.make_tile_group(type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0xa200, mutex_ids=[18, 19])

    # NZ staging for the UB to L1 bridge: one whole-tile view plus one view per
    # eight-column slab, all aliasing the same addresses.
    g_scores_nz_ub = pl.make_tile_group(type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ), addrs=0x12200, mutex_ids=[20])
    g_scores_nz_slab0 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x12200, mutex_ids=[20])
    g_scores_nz_slab1 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x12600, mutex_ids=[20])
    g_scores_nz_slab2 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x12a00, mutex_ids=[20])
    g_scores_nz_slab3 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x12e00, mutex_ids=[20])
    g_scores_nz_slab4 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x13200, mutex_ids=[20])
    g_scores_nz_slab5 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x13600, mutex_ids=[20])
    g_scores_nz_slab6 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x13a00, mutex_ids=[20])
    g_scores_nz_slab7 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x13e00, mutex_ids=[20])
    g_grad_o_nz_ub = pl.make_tile_group(type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ), addrs=0x15200, mutex_ids=[21])
    g_grad_o_nz_slab0 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x15200, mutex_ids=[21])
    g_grad_o_nz_slab1 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x15600, mutex_ids=[21])
    g_grad_o_nz_slab2 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x15a00, mutex_ids=[21])
    g_grad_o_nz_slab3 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x15e00, mutex_ids=[21])
    g_grad_o_nz_slab4 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x16200, mutex_ids=[21])
    g_grad_o_nz_slab5 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x16600, mutex_ids=[21])
    g_grad_o_nz_slab6 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x16a00, mutex_ids=[21])
    g_grad_o_nz_slab7 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x16e00, mutex_ids=[21])
    g_grad_o_nz_slab8 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x17200, mutex_ids=[21])
    g_grad_o_nz_slab9 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x17600, mutex_ids=[21])
    g_grad_o_nz_slab10 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x17a00, mutex_ids=[21])
    g_grad_o_nz_slab11 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x17e00, mutex_ids=[21])
    g_grad_o_nz_slab12 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x18200, mutex_ids=[21])
    g_grad_o_nz_slab13 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x18600, mutex_ids=[21])
    g_grad_o_nz_slab14 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x18a00, mutex_ids=[21])
    g_grad_o_nz_slab15 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x18e00, mutex_ids=[21])

    with pl.section_cube():
        pl.system.set_cross_core(pipe=_FIX, event_id=E_L1_OPERANDS_FREE, sync_mode=_IB)
        scores_l1 = g_scores_l1.next()
        grad_o_l1 = g_grad_o_l1.next()

        task_count = (B * H) * N
        tasks_per_core = (task_count + pl.get_block_num() - 1) // pl.get_block_num()
        first_task = tasks_per_core * (pl.get_block_idx() // pl.get_subblock_num())
        task_end = pl.min(first_task + tasks_per_core, task_count)

        # Prologue: fill slot 0 with the first item's two operands.
        first_b = first_task // (H * N)
        first_h = (first_task % (H * N)) // N
        first_n = (first_task % (H * N)) % N
        pl.set_validshape(g_k_l1[0], [C, Kd])
        pl.set_validshape(g_k_l1[1], [C, Kd])
        pl.set_validshape(g_q_t_l1[0], [Kd, C])
        pl.set_validshape(g_q_t_l1[1], [Kd, C])
        pl.load(g_k_l1[0], k_f32, [first_b, first_h, first_n, 0, 0])
        pl.load(g_q_t_l1[0], q_t, [first_b, first_h, first_n, 0, 0])

        for tid in pl.range(first_task, task_end, 1):
            slot = (tid - first_task) % 2
            next_slot = ((tid - first_task) + 1) % 2
            next_tid = pl.min(tid + 1, task_end - 1)
            k_l1 = g_k_l1[slot]
            q_t_l1 = g_q_t_l1[slot]

            # scores = k_f32 @ q_t
            scores_acc = g_scores_acc.next()
            g_scores_l0a.next()
            k_l0a = g_k_l0a.next()
            g_grad_o_l0b.next()
            q_t_l0b = g_q_t_l0b.next()
            pl.move(k_l0a, k_l1, [0, 0])
            pl.move(q_t_l0b, q_t_l1, [0, 0])
            pl.set_validshape(k_l0a, [C, Kd])
            pl.set_validshape(q_t_l0b, [Kd, C])
            pl.set_validshape(scores_acc, [C, C])
            pl.matmul(scores_acc, k_l0a, q_t_l0b)

            # Refill the other slot for the next item, under this item's MAC.
            next_b = next_tid // (H * N)
            next_h = (next_tid % (H * N)) // N
            next_n = (next_tid % (H * N)) % N
            pl.load(g_k_l1[next_slot], k_f32, [next_b, next_h, next_n, 0, 0])
            pl.load(g_q_t_l1[next_slot], q_t, [next_b, next_h, next_n, 0, 0])

            pl.system.wait_cross_core(pipe=_S, event_id=E_SCORES_UB_FREE, sync_mode=_IB)
            scores_ub = g_scores_ub.next()
            pl.set_validshape(scores_acc, [C, C])
            pl.move(scores_ub, scores_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=_FIX, event_id=E_SCORES_IN_UB, sync_mode=_IB)

            pl.system.bar_all()
            pl.system.wait_cross_core(pipe=_S, event_id=E_L1_OPERANDS_READY, sync_mode=_IB)

            # dv_local = masked scores @ grad_o
            dv_acc = g_dv_acc.next()
            g_k_l0a.next()
            scores_l0a = g_scores_l0a.next()
            g_q_t_l0b.next()
            grad_o_l0b = g_grad_o_l0b.next()
            pl.move(scores_l0a, scores_l1, [0, 0])
            pl.move(grad_o_l0b, grad_o_l1, [0, 0])
            pl.set_validshape(scores_l0a, [C, C])
            pl.set_validshape(grad_o_l0b, [C, VMAX])
            pl.set_validshape(dv_acc, [C, Vd])
            pl.matmul(dv_acc, scores_l0a, grad_o_l0b)
            pl.system.set_cross_core(pipe=_FIX, event_id=E_L1_OPERANDS_FREE, sync_mode=_IB)

            pl.system.wait_cross_core(pipe=_S, event_id=E_DV_UB_FREE, sync_mode=_IB)
            dv_ub = g_dv_ub.next()
            pl.set_validshape(dv_acc, [C, Vd])
            pl.move(dv_ub, dv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=_FIX, event_id=E_DV_IN_UB, sync_mode=_IB)

        pl.system.wait_cross_core(pipe=_S, event_id=E_SCORES_UB_FREE, sync_mode=_IB)
        pl.system.wait_cross_core(pipe=_S, event_id=E_DV_UB_FREE, sync_mode=_IB)

    with pl.section_vector():
        pl.system.set_cross_core(pipe=_V, event_id=E_SCORES_UB_FREE, sync_mode=_IB)
        pl.system.set_cross_core(pipe=_V, event_id=E_DV_UB_FREE, sync_mode=_IB)
        g_k_l1.next()
        g_q_t_l1.next()
        gate_ub = g_gate_ub.next()
        decay_ub = g_decay_ub.next()
        grad_o_bf16_ub = g_grad_o_bf16_ub.next()
        grad_o_fp32_ub = g_grad_o_fp32_ub.next()
        scores_l1 = g_scores_l1.next()
        grad_o_l1 = g_grad_o_l1.next()

        task_count = (B * H) * N
        tasks_per_core = (task_count + pl.get_block_num() - 1) // pl.get_block_num()
        first_task = tasks_per_core * (pl.get_block_idx() // pl.get_subblock_num())
        task_end = pl.min(first_task + tasks_per_core, task_count)
        for tid in pl.range(first_task, task_end, 1):
            b = tid // (H * N)
            h = (tid % (H * N)) // N
            n = (tid % (H * N)) % N

            # The gate cumsum and the grad_o widening read no cube output, so
            # they run before the score handshake is awaited.
            pl.set_validshape(gate_ub, [1, C])
            pl.load(gate_ub, g_f32, [b, h, n, 0])
            cumsum_decay_vf(gate_ub, decay_ub)

            row_base = pl.get_subblock_idx() * CH
            token_start = n * C + row_base
            row_count = pl.min(pl.max(T - token_start, 0), CH)
            if row_count > 0:
                # grad_o is the one public [B, T, H, V] input read with a 2-D
                # tile: order= makes it walk (T, V) instead of the default
                # trailing (H, V), which would stride by t*H+h.
                pl.set_validshape(grad_o_bf16_ub, [row_count, Vd])
                pl.load(grad_o_bf16_ub, grad_o, [b, token_start, h, 0], order=[1, 3])
            widen_grad_o_vf(grad_o_bf16_ub, grad_o_fp32_ub, row_count)

            # The grad_o L1 operand needs only the L1 handshake, so it is built
            # before waiting for the scores.
            pl.system.wait_cross_core(pipe=_S, event_id=E_L1_OPERANDS_FREE, sync_mode=_IB)
            grad_o_nz = g_grad_o_nz_ub.next()
            pl.move(grad_o_nz, grad_o_fp32_ub)
            slab = g_grad_o_nz_slab0.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 0])
            slab = g_grad_o_nz_slab1.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 8])
            slab = g_grad_o_nz_slab2.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 16])
            slab = g_grad_o_nz_slab3.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 24])
            slab = g_grad_o_nz_slab4.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 32])
            slab = g_grad_o_nz_slab5.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 40])
            slab = g_grad_o_nz_slab6.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 48])
            slab = g_grad_o_nz_slab7.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 56])
            slab = g_grad_o_nz_slab8.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 64])
            slab = g_grad_o_nz_slab9.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 72])
            slab = g_grad_o_nz_slab10.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 80])
            slab = g_grad_o_nz_slab11.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 88])
            slab = g_grad_o_nz_slab12.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 96])
            slab = g_grad_o_nz_slab13.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 104])
            slab = g_grad_o_nz_slab14.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 112])
            slab = g_grad_o_nz_slab15.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(grad_o_l1, slab, [row_base, 120])

            pl.system.wait_cross_core(pipe=_S, event_id=E_SCORES_IN_UB, sync_mode=_IB)
            scores_ub = g_scores_ub.next()
            gate_and_mask_scores_vf(decay_ub, scores_ub, row_base)
            pl.system.set_cross_core(pipe=_V, event_id=E_SCORES_UB_FREE, sync_mode=_IB)

            scores_nz = g_scores_nz_ub.next()
            pl.move(scores_nz, scores_ub)
            slab = g_scores_nz_slab0.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(scores_l1, slab, [row_base, 0])
            slab = g_scores_nz_slab1.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(scores_l1, slab, [row_base, 8])
            slab = g_scores_nz_slab2.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(scores_l1, slab, [row_base, 16])
            slab = g_scores_nz_slab3.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(scores_l1, slab, [row_base, 24])
            slab = g_scores_nz_slab4.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(scores_l1, slab, [row_base, 32])
            slab = g_scores_nz_slab5.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(scores_l1, slab, [row_base, 40])
            slab = g_scores_nz_slab6.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(scores_l1, slab, [row_base, 48])
            slab = g_scores_nz_slab7.next()
            pl.set_validshape(slab, [CH, NZ_SLAB])
            pl.insert(scores_l1, slab, [row_base, 56])

            pl.system.bar_all()
            pl.system.set_cross_core(pipe=_MTE3, event_id=E_L1_OPERANDS_READY, sync_mode=_IB)

            pl.system.wait_cross_core(pipe=_S, event_id=E_DV_IN_UB, sync_mode=_IB)
            dv_ub = g_dv_ub.next()
            pl.set_validshape(dv_ub, [CH, Vd])
            pl.store(dv_local, dv_ub, [b, h, n, row_base, 0])
            pl.system.set_cross_core(pipe=_V, event_id=E_DV_UB_FREE, sync_mode=_IB)

        pl.system.wait_cross_core(pipe=_S, event_id=E_L1_OPERANDS_FREE, sync_mode=_IB)


def gdr_bwd_dv_local_wrapper(grad_o, k_f32=None, q_t=None, g_f32=None, *, sync=True):
    """Compute the within-chunk part of dv in one launch.

    The wrapper does pure-Python shape arithmetic, one output allocation and one
    kernel launch. No host-side arithmetic, cast, transpose, slice, pad or
    cumulative sum happens here.

    Returns dv_local [B, H, N, C, V] float32.
    """
    B, T, H, V = grad_o.shape
    N, K = k_f32.shape[2], k_f32.shape[4]

    dv_local = torch.empty((B, H, N, C, V), dtype=torch.float32, device=grad_o.device)

    max_blocks = get_platform_info().core_num or 1
    num_cores = max(1, min(B * H * N, max_blocks))

    gdr_bwd_dv_local_module1_kernel[None, num_cores](
        k_f32, q_t, grad_o, g_f32, dv_local, B, T, H, K, V, N, C)
    if sync:
        torch.npu.synchronize()
    return dv_local
