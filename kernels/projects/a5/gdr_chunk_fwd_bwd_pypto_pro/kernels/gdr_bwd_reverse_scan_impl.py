# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reverse inter-chunk scan of the state gradient dS.

The directional dual of the training forward's state scan: where the forward
carries S forward over chunks, the backward carries dS backward over them. Both
recurrences are serial in the chunk index, so this operator and the forward scan
are the two serial members of the eight-operator pipeline.

Mathematics
-----------
For each (batch, head), chunks are visited in reverse order i = N-1 .. 0 with
dS[K, V] initialised to zero. Writing decay for the within-chunk cumulative gate
sum and g_last for its last element::

    dh[., i]  = dS                                  entering-state snapshot
    dvc       = k_f32_i @ dS
    dv2[., i] = dv_local_i + dvc * exp(g_last_i - decay_i)
    dS        = dS * decay_exp_i[C-1]
                + q_t_i @ (decay_exp_i . grad_o_i)
                - w_t_i @ dv2_i

dh is the entering state, so it must be snapshotted before dS is updated, and
dh[., N-1] is zero. The 1/sqrt(K) scale is already folded into q_t upstream, so
no scale appears here.

Interface
---------
``gdr_bwd_reverse_scan_wrapper(q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local)``

    q_t, w_t      [B, H, N, K, C] f32   pre-transposed q*scale and WY w
    k_f32         [B, H, N, C, K] f32   k widened to float32
    grad_o        [B, T, H, V] bf16     incoming output gradient, public layout
    decay_exp     [B, H, N, C] f32      exp of the within-chunk cumulative gate
    g_f32         [B, H, N, C] f32      per-token log gate
    dv_local      [B, H, N, C, V] f32   within-chunk part of dv

    returns dh    [B, H, N, K, V] f32   entering state gradient per chunk
            dv2   [B, H, N, C, V] f32   dv with the cross-chunk part added

Layout and scheduling
---------------------
One launch of one mixed cube/vector kernel. A work item is one
(batch, head, V sub-block) triple: the chunk recurrence cannot be split across
cores, but the V columns are independent, so V is sub-blocked at width TV to
raise the item count. dS lives in UB as a single persistent buffer that never
rotates, and reaches the cube through a GM bridge, because an L1 operand cannot
be written by the vector core directly. Four cross-core handshakes per chunk
sequence the two cube phases (dvc, then the two dS update matmuls) against the
three vector phases (gate application, bridge stores, state update). All vector
work runs on sub-block 0; the guard keeps the second sub-block out of the
single-buffered state.
"""

import logging

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

C = 64      # tokens per chunk, a frozen constant of the operator family
TV = 64     # V sub-block width; keeps the transposed L0B operand within 8192
            # float32 elements

_FIX = pl.PipeType.FIX
_S = pl.PipeType.S
_V = pl.PipeType.V
_MTE3 = pl.PipeType.MTE3
_IB = pl.CrossCoreSyncMode.INTRA_BLOCK

# Cross-core event ids. Even ids are raised by the cube, odd ids by the vector
# core, and each pair guards one buffer handover. The two ids the cube raises
# before the task loop start with two credits each, so the first chunks of a
# sequence are not blocked by a consumer that has nothing to consume yet.
E_STATE_CONSUMED = 0    # cube -> vector: the dS bridge has been read into L1
E_STATE_READY = 1       # vector -> cube: dS is readable through its GM bridge
E_DVC_IN_UB = 2         # cube -> vector: dvc has been drained to UB
E_DVC_FREE = 3          # vector -> cube: the dvc landing tile is consumed
E_OPERANDS_CONSUMED = 4 # cube -> vector: the two GM operands have been read
E_OPERANDS_READY = 5    # vector -> cube: scaled grad_o and -dv2 are in GM
E_INCREMENT_IN_UB = 6   # cube -> vector: the dS increment has reached UB
E_INCREMENT_FREE = 7    # vector -> cube: the increment landing tile is consumed

# Some PyPTO builds expose part of the DT_* enum as bare integers, which the
# vector-function layer rejects. Rebuild the affected ones at module scope: the
# parser inlines helper calls, so a decorated body must see plain names.
_DATA_TYPE = type(pl.DT_FP32)


def _as_data_type(value):
    return value if isinstance(value, _DATA_TYPE) else _DATA_TYPE(value)


_PL_FP32 = _as_data_type(pl.DT_FP32)


@pl.vector_function
def cumsum_vf(gate_tile, decay_tile):
    """decay[t] = sum over t' <= t of gate[t'], one chunk in place."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    running = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for token in pl.range(0, C, 1):
        gate = vf.load_align(gate_tile, token, dist=pl.LoadDist.BRC_B32)
        running = vf.add(running, gate, all_lanes)
        vf.store_align(decay_tile + token, running, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def dstate_update_vf(dstate_tile, increment_tile, decay_exp_tile, row_count):
    """dS = dS * decay_exp[C-1] + increment, over `row_count` rows of width TV."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    chunk_decay = vf.load_align(decay_exp_tile, C - 1, dist=pl.LoadDist.BRC_B32)
    for row in pl.range(0, row_count, 1):
        state = vf.load_align(dstate_tile, row * TV, dtype=_PL_FP32)
        increment = vf.load_align(increment_tile, row * TV, dtype=_PL_FP32)
        state = vf.mul(state, chunk_decay, all_lanes)
        state = vf.add(state, increment, all_lanes)
        vf.store_align(dstate_tile + row * TV, state, all_lanes)


@pl.vector_function
def grad_o_scaled_vf(grad_o_bf16_tile, grad_o_fp32_tile, decay_exp_tile, token_count):
    """Widen grad_o to float32 and scale row t by decay_exp[t]; pad with zeros."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for token in pl.range(0, token_count, 1):
        decay = vf.load_align(decay_exp_tile, token, dist=pl.LoadDist.BRC_B32)
        narrow = vf.load_align(grad_o_bf16_tile, token * TV, dist=pl.LoadDist.UNPK_B16)
        wide = vf.astype(narrow, all_lanes, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
        wide = vf.mul(wide, decay, all_lanes)
        vf.store_align(grad_o_fp32_tile + token * TV, wide, all_lanes)
    for token in pl.range(token_count, C, 1):
        vf.store_align(grad_o_fp32_tile + token * TV, zero, all_lanes)


@pl.vector_function
def dv2_gate_vf(decay_tile, dvc_tile, dv_local_tile, dv2_tile):
    """dv2 = dv_local + dvc * exp(g_last - decay[t]), one chunk."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    chunk_decay = vf.load_align(decay_tile, C - 1, dist=pl.LoadDist.BRC_B32)
    for token in pl.range(0, C, 1):
        decay = vf.load_align(decay_tile, token, dist=pl.LoadDist.BRC_B32)
        gate = vf.exp_sub(chunk_decay, decay, all_lanes)
        dv2_row = vf.load_align(dvc_tile, token * TV, dtype=_PL_FP32)
        dv_local_row = vf.load_align(dv_local_tile, token * TV, dtype=_PL_FP32)
        dv2_row = vf.mul(dv2_row, gate, all_lanes)
        dv2_row = vf.add(dv2_row, dv_local_row, all_lanes)
        vf.store_align(dv2_tile + token * TV, dv2_row, all_lanes)


@pl.vector_function
def negate_vf(tile):
    """Negate one [C, TV] float32 tile in place."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    for row in pl.range(0, C, 1):
        value = vf.load_align(tile, row * TV, dtype=_PL_FP32)
        value = vf.neg(value, all_lanes)
        vf.store_align(tile + row * TV, value, all_lanes)


@pl.vector_function
def zero_dstate_vf(dstate_tile, row_count):
    """Zero `row_count` rows of width TV."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for row in pl.range(0, row_count, 1):
        vf.store_align(dstate_tile + row * TV, zero, all_lanes)


_DYN5 = [pl.DYNAMIC] * 5
_DYN4 = [pl.DYNAMIC] * 4


@pl.jit(auto_mutex=True)
def gdr_bwd_reverse_scan_module1_kernel(
    q_t: pl.Tensor[_DYN5, pl.DT_FP32],
    w_t: pl.Tensor[_DYN5, pl.DT_FP32],
    k_f32: pl.Tensor[_DYN5, pl.DT_FP32],
    dstate_gm: pl.Tensor[_DYN5, pl.DT_FP32],
    grad_o_scaled_gm: pl.Tensor[_DYN5, pl.DT_FP32],
    dv2_negated_gm: pl.Tensor[_DYN5, pl.DT_FP32],
    grad_o: pl.Tensor[_DYN4, pl.DT_BF16],
    decay_exp: pl.Tensor[_DYN4, pl.DT_FP32],
    g_f32: pl.Tensor[_DYN4, pl.DT_FP32],
    dv_local: pl.Tensor[_DYN5, pl.DT_FP32],
    dh: pl.Tensor[_DYN5, pl.DT_FP32],
    dv2: pl.Tensor[_DYN5, pl.DT_FP32],
    B: pl.DT_INT32, T: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32,
    Vd: pl.DT_INT32, N: pl.DT_INT32, Cc: pl.DT_INT32, VT: pl.DT_INT32,
):
    # L0A and L0B each hold two shapes at the same two addresses: the dvc matmul
    # takes [C, K] x [K, TV], the dS update takes [K, C] x [C, TV]. The two cube
    # phases are separated by a vector phase, so the slots are reused off peak
    # and L0A sits exactly at its 64 KB cap.
    g_l0a_rowwise = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g_l0a_colwise = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g_l0b_tall = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right), addrs=[0x0, 0x8000], mutex_ids=[2, 3])
    g_l0b_square = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right), addrs=[0x0, 0x8000], mutex_ids=[2, 3])

    g_dstate_ub = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x0, mutex_ids=[4])
    g_increment_ub = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x8000, mutex_ids=[5])
    g_dvc_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x10000, mutex_ids=[6])
    g_dv_local_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x14000, mutex_ids=[7])
    g_dv2_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x18000, mutex_ids=[8])
    g_grad_o_bf16_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=0x1c000, mutex_ids=[9])
    g_grad_o_fp32_ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x1e000, mutex_ids=[8])
    g_gate_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x22000, mutex_ids=[10])
    g_decay_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x22100, mutex_ids=[11])
    g_decay_exp_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x22200, mutex_ids=[10])

    g_k_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x0, mutex_ids=[12])
    g_dstate_l1 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x8000, mutex_ids=[13])
    g_q_t_l1 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x10000, mutex_ids=[17])
    g_w_t_l1 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x18000, mutex_ids=[18])
    g_grad_o_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x20000, mutex_ids=[15])
    g_dv2_l1 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x24000, mutex_ids=[19])

    g_dvc_acc = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=0x0, mutex_ids=[14])
    g_increment_acc = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=0x4000, mutex_ids=[16])

    with pl.section_cube():
        # Bootstrap credits: the landing tiles start free.
        pl.system.set_cross_core(pipe=_FIX, event_id=E_STATE_CONSUMED, sync_mode=_IB)
        pl.system.set_cross_core(pipe=_FIX, event_id=E_STATE_CONSUMED, sync_mode=_IB)
        pl.system.set_cross_core(pipe=_FIX, event_id=E_OPERANDS_CONSUMED, sync_mode=_IB)
        pl.system.set_cross_core(pipe=_FIX, event_id=E_OPERANDS_CONSUMED, sync_mode=_IB)

        increment_ub = g_increment_ub.next()
        dvc_ub = g_dvc_ub.next()
        k_l1 = g_k_l1.next()
        dstate_l1 = g_dstate_l1.next()
        dvc_acc = g_dvc_acc.next()
        q_t_l1 = g_q_t_l1.next()
        w_t_l1 = g_w_t_l1.next()
        grad_o_l1 = g_grad_o_l1.next()
        dv2_l1 = g_dv2_l1.next()
        increment_acc = g_increment_acc.next()

        v_blocks = (Vd + TV - 1) // TV
        task_count = (B * H) * v_blocks
        core_count = pl.get_block_num()
        core = pl.get_block_idx() // pl.get_subblock_num()
        for tid in pl.range(core, task_count, core_count):
            bh = tid // v_blocks
            v_block = tid - bh * v_blocks
            b = bh // H
            h = bh % H
            v_base = v_block * TV
            v_count = pl.min(Vd - v_base, TV)
            pl.system.bar_all()
            for reverse_index in pl.range(0, N, 1):
                chunk = (N - 1) - reverse_index

                # dvc = k_f32_i @ dS
                pl.set_validshape(k_l1, [C, Kd])
                pl.load(k_l1, k_f32, [b, h, chunk, 0, 0])
                pl.system.wait_cross_core(pipe=_S, event_id=E_STATE_READY, sync_mode=_IB)
                pl.set_validshape(dstate_l1, [Kd, v_count])
                pl.load(dstate_l1, dstate_gm, [b, h, v_block, 0, 0])
                g_l0a_colwise.next()
                k_l0a = g_l0a_rowwise.next()
                g_l0b_square.next()
                dstate_l0b = g_l0b_tall.next()
                pl.move(k_l0a, k_l1, [0, 0])
                pl.move(dstate_l0b, dstate_l1, [0, 0])
                pl.set_validshape(k_l0a, [C, Kd])
                pl.set_validshape(dstate_l0b, [Kd, TV])
                pl.set_validshape(dvc_acc, [C, v_count])
                pl.matmul(dvc_acc, k_l0a, dstate_l0b)
                pl.system.set_cross_core(pipe=_FIX, event_id=E_STATE_CONSUMED, sync_mode=_IB)

                pl.system.wait_cross_core(pipe=_S, event_id=E_DVC_FREE, sync_mode=_IB)
                pl.set_validshape(dvc_acc, [C, v_count])
                pl.move(dvc_ub, dvc_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                pl.system.set_cross_core(pipe=_FIX, event_id=E_DVC_IN_UB, sync_mode=_IB)

                # increment = q_t_i @ (decay_exp_i . grad_o_i) - w_t_i @ dv2_i,
                # the second term arriving already negated from the vector side.
                pl.set_validshape(q_t_l1, [Kd, C])
                pl.load(q_t_l1, q_t, [b, h, chunk, 0, 0])
                pl.set_validshape(w_t_l1, [Kd, C])
                pl.load(w_t_l1, w_t, [b, h, chunk, 0, 0])
                pl.system.wait_cross_core(pipe=_S, event_id=E_OPERANDS_READY, sync_mode=_IB)
                pl.set_validshape(grad_o_l1, [C, v_count])
                pl.load(grad_o_l1, grad_o_scaled_gm, [b, h, v_block, 0, 0])
                pl.set_validshape(dv2_l1, [C, v_count])
                pl.load(dv2_l1, dv2_negated_gm, [b, h, v_block, 0, 0])

                g_l0a_rowwise.next()
                q_t_l0a = g_l0a_colwise.next()
                g_l0b_tall.next()
                grad_o_l0b = g_l0b_square.next()
                pl.move(q_t_l0a, q_t_l1, [0, 0])
                pl.move(grad_o_l0b, grad_o_l1, [0, 0])
                pl.set_validshape(q_t_l0a, [128, C])
                pl.set_validshape(grad_o_l0b, [C, TV])
                pl.set_validshape(increment_acc, [128, v_count])
                pl.matmul(increment_acc, q_t_l0a, grad_o_l0b)

                g_l0a_rowwise.next()
                w_t_l0a = g_l0a_colwise.next()
                g_l0b_tall.next()
                dv2_l0b = g_l0b_square.next()
                pl.move(w_t_l0a, w_t_l1, [0, 0])
                pl.move(dv2_l0b, dv2_l1, [0, 0])
                pl.set_validshape(w_t_l0a, [128, C])
                pl.set_validshape(dv2_l0b, [C, TV])
                pl.set_validshape(increment_acc, [128, v_count])
                pl.matmul_acc(increment_acc, increment_acc, w_t_l0a, dv2_l0b)
                pl.system.set_cross_core(pipe=_FIX, event_id=E_OPERANDS_CONSUMED, sync_mode=_IB)

                pl.system.wait_cross_core(pipe=_S, event_id=E_INCREMENT_FREE, sync_mode=_IB)
                pl.set_validshape(increment_acc, [128, v_count])
                pl.move(increment_ub, increment_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                pl.system.set_cross_core(pipe=_FIX, event_id=E_INCREMENT_IN_UB, sync_mode=_IB)

        # Drain the bootstrap credits so the two sections end balanced.
        pl.system.wait_cross_core(pipe=_S, event_id=E_DVC_FREE, sync_mode=_IB)
        pl.system.wait_cross_core(pipe=_S, event_id=E_DVC_FREE, sync_mode=_IB)
        pl.system.wait_cross_core(pipe=_S, event_id=E_INCREMENT_FREE, sync_mode=_IB)
        pl.system.wait_cross_core(pipe=_S, event_id=E_INCREMENT_FREE, sync_mode=_IB)

    with pl.section_vector():
        pl.system.set_cross_core(pipe=_V, event_id=E_DVC_FREE, sync_mode=_IB)
        pl.system.set_cross_core(pipe=_V, event_id=E_DVC_FREE, sync_mode=_IB)
        pl.system.set_cross_core(pipe=_V, event_id=E_INCREMENT_FREE, sync_mode=_IB)
        pl.system.set_cross_core(pipe=_V, event_id=E_INCREMENT_FREE, sync_mode=_IB)

        dstate_ub = g_dstate_ub.next()
        increment_ub = g_increment_ub.next()
        dvc_ub = g_dvc_ub.next()
        dv_local_ub = g_dv_local_ub.next()
        dv2_ub = g_dv2_ub.next()
        grad_o_bf16_ub = g_grad_o_bf16_ub.next()
        grad_o_fp32_ub = g_grad_o_fp32_ub.next()
        gate_ub = g_gate_ub.next()
        decay_ub = g_decay_ub.next()
        decay_exp_ub = g_decay_exp_ub.next()
        g_k_l1.next()
        g_dstate_l1.next()
        g_q_t_l1.next()
        g_w_t_l1.next()
        g_grad_o_l1.next()
        g_dv2_l1.next()

        v_blocks = (Vd + TV - 1) // TV
        task_count = (B * H) * v_blocks
        core_count = pl.get_block_num()
        core = pl.get_block_idx() // pl.get_subblock_num()
        for tid in pl.range(core, task_count, core_count):
            bh = tid // v_blocks
            v_block = tid - bh * v_blocks
            b = bh // H
            h = bh % H
            v_base = v_block * TV
            v_count = pl.min(Vd - v_base, TV)
            # The state is single-buffered, so all vector work runs on
            # sub-block 0 and this offset is zero there.
            subblock_base = pl.get_subblock_idx() * TV
            if subblock_base < TV:
                zero_dstate_vf(dstate_ub, Kd)
            pl.system.bar_all()
            for reverse_index in pl.range(0, N, 1):
                chunk = (N - 1) - reverse_index
                token_base = chunk * C
                token_count = pl.min(T - token_base, C)

                pl.system.wait_cross_core(pipe=_S, event_id=E_STATE_CONSUMED, sync_mode=_IB)
                if subblock_base < TV:
                    # Snapshot the entering state, and republish it for the cube.
                    pl.set_validshape(dstate_ub, [Kd, v_count])
                    pl.store(dh, dstate_ub, [b, h, chunk, 0, v_base])
                    pl.set_validshape(dstate_ub, [Kd, v_count])
                    pl.store(dstate_gm, dstate_ub, [b, h, v_block, 0, 0])
                pl.system.set_cross_core(pipe=_MTE3, event_id=E_STATE_READY, sync_mode=_IB)

                if subblock_base < TV:
                    pl.set_validshape(dv_local_ub, [C, v_count])
                    pl.load(dv_local_ub, dv_local, [b, h, chunk, 0, v_base])
                    pl.set_validshape(gate_ub, [1, C])
                    pl.load(gate_ub, g_f32, [b, h, chunk, 0])
                    pl.set_validshape(decay_exp_ub, [1, C])
                    pl.load(decay_exp_ub, decay_exp, [b, h, chunk, 0])
                    if token_count > 0:
                        # grad_o is the one public [B, T, H, V] input read with a
                        # 2-D tile: order= makes it walk (T, V) instead of the
                        # default trailing (H, V), which would stride by t*H+h.
                        pl.set_validshape(grad_o_bf16_ub, [token_count, v_count])
                        pl.load(grad_o_bf16_ub, grad_o, [b, token_base, h, v_base], order=[1, 3])

                pl.system.wait_cross_core(pipe=_S, event_id=E_DVC_IN_UB, sync_mode=_IB)
                if subblock_base < TV:
                    cumsum_vf(gate_ub, decay_ub)
                    dv2_gate_vf(decay_ub, dvc_ub, dv_local_ub, dv2_ub)
                    pl.set_validshape(dv2_ub, [C, v_count])
                    pl.store(dv2, dv2_ub, [b, h, chunk, 0, v_base])
                    grad_o_scaled_vf(grad_o_bf16_ub, grad_o_fp32_ub, decay_exp_ub, token_count)
                    negate_vf(dv2_ub)
                pl.system.set_cross_core(pipe=_V, event_id=E_DVC_FREE, sync_mode=_IB)

                pl.system.wait_cross_core(pipe=_S, event_id=E_OPERANDS_CONSUMED, sync_mode=_IB)
                if subblock_base < TV:
                    pl.set_validshape(grad_o_fp32_ub, [C, v_count])
                    pl.store(grad_o_scaled_gm, grad_o_fp32_ub, [b, h, v_block, 0, 0])
                    pl.set_validshape(dv2_ub, [C, v_count])
                    pl.store(dv2_negated_gm, dv2_ub, [b, h, v_block, 0, 0])
                pl.system.set_cross_core(pipe=_MTE3, event_id=E_OPERANDS_READY, sync_mode=_IB)

                pl.system.wait_cross_core(pipe=_S, event_id=E_INCREMENT_IN_UB, sync_mode=_IB)
                if subblock_base < TV:
                    dstate_update_vf(dstate_ub, increment_ub, decay_exp_ub, Kd)
                pl.system.set_cross_core(pipe=_V, event_id=E_INCREMENT_FREE, sync_mode=_IB)

        pl.system.wait_cross_core(pipe=_S, event_id=E_STATE_CONSUMED, sync_mode=_IB)
        pl.system.wait_cross_core(pipe=_S, event_id=E_STATE_CONSUMED, sync_mode=_IB)
        pl.system.wait_cross_core(pipe=_S, event_id=E_OPERANDS_CONSUMED, sync_mode=_IB)
        pl.system.wait_cross_core(pipe=_S, event_id=E_OPERANDS_CONSUMED, sync_mode=_IB)


def gdr_bwd_reverse_scan_wrapper(q_t, w_t, k_f32, grad_o=None,
                                 decay_exp=None, g_f32=None, dv_local=None, *,
                                 sync=True, cube_core_budget=None):
    """Run the reverse dS scan in one launch.

    `cube_core_budget` caps the blocks this launch plans for. The item space is
    `B * H * ceil(V / TV)` and the kernel strides over it, so any block count
    computes the same result and the choice is one of balance alone. Taking
    every core is right when the item count is a multiple of the core count;
    where it is not, the remainder runs as a near-empty final wave, and asking
    for the fewest blocks that keep the same wave count costs this launch
    nothing while holding a constant number of cores throughout it.

    The wrapper does pure-Python shape arithmetic, allocates the two outputs and
    the three internal GM bridges, and launches once. No cast, permute,
    contiguous or slice happens on the host.

    Returns (dh [B, H, N, K, V], dv2 [B, H, N, C, V]), both float32.
    """
    B, T, H, V = grad_o.shape
    N, K = q_t.shape[2], q_t.shape[3]
    device = q_t.device

    dh = torch.empty((B, H, N, K, V), dtype=torch.float32, device=device)
    dv2 = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)

    # Kernel-internal bridges: dS from UB to L1, and the two operands of the
    # second cube phase, which the vector core produces.
    v_blocks = (V + TV - 1) // TV
    dstate_gm = torch.empty((B, H, v_blocks, K, TV), dtype=torch.float32, device=device)
    grad_o_scaled_gm = torch.empty((B, H, v_blocks, C, TV), dtype=torch.float32, device=device)
    dv2_negated_gm = torch.empty((B, H, v_blocks, C, TV), dtype=torch.float32, device=device)

    max_blocks = get_platform_info().core_num or 1
    if cube_core_budget is not None:
        max_blocks = max(1, min(max_blocks, int(cube_core_budget)))
    num_cores = max(1, min(B * H * v_blocks, max_blocks))

    gdr_bwd_reverse_scan_module1_kernel[None, num_cores](
        q_t, w_t, k_f32, dstate_gm, grad_o_scaled_gm, dv2_negated_gm,
        grad_o, decay_exp, g_f32, dv_local, dh, dv2,
        B, T, H, K, V, N, C, TV)
    if sync:
        torch.npu.synchronize()
    return dh, dv2
