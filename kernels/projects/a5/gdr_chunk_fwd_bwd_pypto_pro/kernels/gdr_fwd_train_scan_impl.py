# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Inter-chunk state scan of the training forward: the output and the state trail.

Second of the two training-forward operators. The first one builds everything
that is local to a chunk; this one walks the chunks in order, carries the
recurrent state S, and emits the output together with the two tensors the
backward pass needs from the recurrence.

Mathematics
-----------
For each (batch, head), chunks are visited in order i = 0 .. N-1 with S[K, V]
initialised to zero::

    s_hist[., i] = S                                  entering-state snapshot
    v_prime      = w_i @ S
    v_new_i      = u_i - v_prime
    o_inter      = q_decayed_i @ S
    o_i          = o_inter + a_mat_i @ v_new_i
    S            = S * decay_exp_i[C-1] + (k_f32_i * dfac_i)^T @ v_new_i

s_hist is the state as it enters a chunk, so it is snapshotted before the
update, and s_hist[., 0] is zero. v_new is the value sequence after the delta
correction; the backward pass cannot reconstruct it from the primary v, so it is
published here. Both come for free: one is a snapshot of state already resident,
the other is computed inside the recurrence anyway.

Everything runs in float32. The only narrowing is the output store, which casts
to bfloat16 on the way out.

Interface
---------
``gdr_fwd_train_scan_wrapper(u, w, q_de, a_mat, k_f32, dfac, decay_exp)``

    u                [B, H, N, C, V] f32   WY-transformed value rows
    w                [B, H, N, C, K] f32   WY-transformed key rows
    q_de             [B, H, N, C, K] f32   q scaled and gate-decayed
    a_mat            [B, H, N, C, C] f32   within-chunk attention, gated and masked
    k_f32            [B, H, N, C, K] f32   k widened to float32
    dfac             [B, H, N, C, K] f32   exp(decay_last - decay), broadcast along K
    decay_exp        [B, H, N, C] f32      exp of the within-chunk cumulative gate

    returns o        [B, C*N, H, V] bf16   padded public output; the caller slices
                                           the first T rows
            s_hist   [B, H, N, K, V] f32   entering state per chunk
            v_new    [B, H, N, C, V] f32   corrected value rows

Layout and scheduling
---------------------
One launch of one mixed cube/vector kernel. A work item is one
(batch, head, V-tile group): the chunk recurrence is serial and cannot cross
cores, but the V columns of all three outputs are independent, so V is tiled and
the tiles are grouped. Each item interleaves up to two V streams, named `_s0`
and `_s1` below, so that one stream's cube work overlaps the other's vector
work; the tile width and the stream count are chosen on the host by
`_pick_vt_streams`.

Three details shape the code.

Matmul and fixpipe always run on the full [C, VTP] tile and only the contraction
axis and the GM transfers are narrowed. A partially valid N axis leaves stale
rows in the destination of a DualModeSplitM drain, so the valid shape on that
axis is kept at the physical width and the narrow V tail is expressed in the
loads, the stores and the vector column masks instead.

k^T reaches L0A as a transposed load into an NZ-laid-out L1 tile, once per chunk
at full K width, which avoids both a pre-transposed copy in GM and a per-tile
re-load. The same holds for w and q_decayed: they are loaded once per chunk over
the whole contraction axis and sliced per K tile with move offsets.

The K-tile and stream loops are nested K-tile outermost on both sides, so that
consecutive credits of a counting semaphore land on different buffers of the
depth-one delta tile rather than twice on the same one.
"""

import logging

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

C = 64            # tokens per chunk, a frozen constant of the operator family
KT = 64           # K tile: the contraction block, and the delta/state row block
VTP = 64          # physical V tile width; narrower runtime tiles are expressed
                  # through valid shapes, not through a smaller tile
KMAX = 256        # supported K upper bound, and the width of the full-K L1 tiles
CH = 32           # rows per vector sub-block after a DualModeSplitM drain

_IB = pl.CrossCoreSyncMode.INTRA_BLOCK
_S = pl.PipeType.S
_V = pl.PipeType.V
_FIX = pl.PipeType.FIX
_MTE2 = pl.PipeType.MTE2
_MTE3 = pl.PipeType.MTE3

# Cross-core event ids, one per buffer handover.
E_S_READY = 0     # vector -> cube : s_hist[i] is readable
E_VP_READY = 1    # cube -> vector : the cube landing tile holds v_prime or o
E_VP_FREE = 2     # vector -> cube : that landing tile is consumed
E_VN_READY = 3    # vector -> cube : v_new and the scaled v are in GM
E_VN_FREE = 4     # cube -> vector : both have been read
E_D_READY = 5     # cube -> vector : the landing tile holds the state increment
E_D_FREE = 6      # vector -> cube : that landing tile is consumed


@pl.vector_function
def subtract_and_scale_vf(v_prime_tile, u_tile, decay_factor_tile, v_scaled_tile,
                          row_count: pl.DT_INT64, col_count: pl.DT_INT64):
    """v_new = u - v_prime, in place over u; v_scaled = v_new * dfac[row]."""
    for row in pl.range(0, row_count):
        factor = vf.load_align(decay_factor_tile, row * 8, dist=pl.LoadDist.BRC_B32)
        live_lanes = vf.update_mask(col_count, dtype=pl.DT_FP32)
        offset = row * VTP
        u_row = vf.load_align(u_tile, offset)
        v_prime_row = vf.load_align(v_prime_tile, offset)
        v_new_row = vf.sub(u_row, v_prime_row, live_lanes)
        v_scaled_row = vf.mul(v_new_row, factor, live_lanes)
        vf.store_align(u_tile + offset, v_new_row, live_lanes)
        vf.store_align(v_scaled_tile + offset, v_scaled_row, live_lanes)


@pl.vector_function
def state_update_vf(state_tile, delta_tile, decay_exp_tile,
                    row_count: pl.DT_INT64, col_count: pl.DT_INT64,
                    decay_index: pl.DT_INT64):
    """S = S * decay_exp[decay_index] + delta, over one row band of the state."""
    chunk_decay = vf.load_align(decay_exp_tile, decay_index, dist=pl.LoadDist.BRC_B32)
    for row in pl.range(0, row_count):
        live_lanes = vf.update_mask(col_count, dtype=pl.DT_FP32)
        offset = row * VTP
        state = vf.load_align(state_tile, offset)
        delta = vf.load_align(delta_tile, offset)
        state = vf.mul(state, chunk_decay, live_lanes)
        state = vf.add(state, delta, live_lanes)
        vf.store_align(state_tile + offset, state, live_lanes)


@pl.vector_function
def zero_state_vf(state_tile, row_count: pl.DT_INT64):
    """Zero `row_count` rows of width VTP."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, all_lanes, dtype=pl.DT_FP32)
    for row in pl.range(0, row_count):
        vf.store_align(state_tile + row * VTP, zero, all_lanes)


_DYN5 = [pl.DYNAMIC] * 5
_DYN4 = [pl.DYNAMIC] * 4


@pl.jit(auto_mutex=True)
def gdr_fwd_train_scan_module1_kernel(
    u: pl.Tensor[_DYN5, pl.DT_FP32],
    w: pl.Tensor[_DYN5, pl.DT_FP32],
    q_de: pl.Tensor[_DYN5, pl.DT_FP32],
    a_mat: pl.Tensor[_DYN5, pl.DT_FP32],
    k_f32: pl.Tensor[_DYN5, pl.DT_FP32],
    dfac: pl.Tensor[_DYN5, pl.DT_FP32],
    decay_exp: pl.Tensor[_DYN4, pl.DT_FP32],
    v_scaled_gm: pl.Tensor[_DYN4, pl.DT_FP32],
    o: pl.Tensor[_DYN4, pl.DT_BF16],
    s_hist: pl.Tensor[_DYN5, pl.DT_FP32],
    v_new: pl.Tensor[_DYN5, pl.DT_FP32],
    B: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32, Vd: pl.DT_INT32,
    N: pl.DT_INT32, VT: pl.DT_INT32, vtiles: pl.DT_INT32,
    vpairs: pl.DT_INT32, SR: pl.DT_INT32,
):
    # UB, 115,968 B of the 253,952 B budget. The state tile group holds one
    # buffer per K tile; the other groups are single-buffered because the scan
    # is serial. Two groups deliberately alias another group's address and
    # mutex id, their live ranges being disjoint: the bfloat16 output band over
    # the scaled v, and the state increment over v_prime.
    g_state_ub_s0 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=[0x00000, 0x1C800, 0x1E800, 0x20800], mutex_ids=[0, 27, 28, 29])
    g_v_new_ub_s0 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x02000, mutex_ids=[1])
    g_v_scaled_ub_s0 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x04000, mutex_ids=[2])
    g_cube_out_ub_s0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x06000, mutex_ids=[3])
    g_delta_ub_s0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x0A000, mutex_ids=[4])
    g_state_ub_s1 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=[0x0E000, 0x22800, 0x24800, 0x26800], mutex_ids=[5, 30, 31, 27])
    g_v_new_ub_s1 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x10000, mutex_ids=[6])
    g_v_scaled_ub_s1 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x12000, mutex_ids=[7])
    g_cube_out_ub_s1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x14000, mutex_ids=[8])
    g_delta_ub_s1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x18000, mutex_ids=[9])
    g_o_bf16_ub_s0 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x04000, mutex_ids=[2])
    g_o_bf16_ub_s1 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x12000, mutex_ids=[7])
    g_decay_factor_ub = pl.make_tile_group(type=pl.TileType(shape=[CH, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x1C000, mutex_ids=[10])
    g_decay_exp_ub = pl.make_tile_group(type=pl.TileType(shape=[1, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x1C400, mutex_ids=[11])

    # L1, 311,296 B of 524,288. w, q_decayed and k^T are held at full K width
    # for a whole chunk and sliced per K tile with move offsets.
    g_w_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x00000, mutex_ids=[12])
    g_q_decayed_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x10000, mutex_ids=[14])
    g_a_mat_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x30000, mutex_ids=[14])
    # Transposed left operand: declared [K, C] with an NZ layout and filled by a
    # transposing load, so no pre-transposed copy of k is needed in GM.
    g_k_t_l1 = pl.make_tile_group(type=pl.TileType(shape=[KMAX, C], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], layout=pl.ZN), addrs=0x20000, mutex_ids=[13])
    g_state_l1_s0 = pl.make_tile_group(type=pl.TileType(shape=[KT, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x34000, mutex_ids=[15])
    g_state_l1_s1 = pl.make_tile_group(type=pl.TileType(shape=[KT, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x38000, mutex_ids=[16])
    g_v_new_l1_s0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x3C000, mutex_ids=[15])
    g_v_new_l1_s1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x40000, mutex_ids=[16])
    g_v_scaled_l1_s0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x44000, mutex_ids=[17])
    g_v_scaled_l1_s1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x48000, mutex_ids=[18])

    g_l0a = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1]), addrs=[0x0000, 0x4000], mutex_ids=[19, 20])
    g_l0b = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1]), addrs=[0x0000, 0x4000], mutex_ids=[21, 22])
    # v_prime and the state increment share one accumulator per stream: their
    # live ranges within a chunk are disjoint.
    g_scan_acc_s0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x00000, mutex_ids=[23])
    g_o_acc_s0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x04000, mutex_ids=[24])
    g_scan_acc_s1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x08000, mutex_ids=[25])
    g_o_acc_s1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x0C000, mutex_ids=[26])

    with pl.section_cube():
        core_count = pl.get_block_num()
        core = pl.get_block_idx() // pl.get_subblock_num()
        k_tiles = (Kd + KT - 1) // KT
        task_count = B * H * vpairs
        tasks_per_core = (task_count + core_count - 1) // core_count
        first_task = tasks_per_core * core
        task_end = pl.min(first_task + tasks_per_core, task_count)

        for tid in pl.range(first_task, task_end, 1):
            bh = tid // vpairs
            v_pair = tid % vpairs
            b = bh // H
            h = bh % H
            v_tile_s0 = v_pair * SR
            v_tile_limit = pl.min(vtiles, v_tile_s0 + SR)
            second_stream = (v_tile_s0 + 1) < v_tile_limit
            v_base_s0 = v_tile_s0 * VT
            v_count_s0 = pl.min(VT, Vd - v_base_s0)
            v_base_s1 = (v_tile_s0 + 1) * VT
            v_count_s1 = pl.min(VT, Vd - v_base_s1)

            # One free credit per active stream for the v_new GM handover.
            pl.system.set_cross_core(pipe=_FIX, event_id=E_VN_FREE, sync_mode=_IB)
            if second_stream:
                pl.system.set_cross_core(pipe=_FIX, event_id=E_VN_FREE, sync_mode=_IB)

            for ni in pl.range(0, N, 1):
                # Hoist every tile handle once per chunk for the mutex analysis.
                state_l1_s0 = g_state_l1_s0.current()
                v_new_l1_s0 = g_v_new_l1_s0.current()
                v_scaled_l1_s0 = g_v_scaled_l1_s0.current()
                scan_acc_s0 = g_scan_acc_s0.current()
                o_acc_s0 = g_o_acc_s0.current()
                cube_out_ub_s0 = g_cube_out_ub_s0.current()
                delta_ub_s0 = g_delta_ub_s0.current()
                state_l1_s1 = g_state_l1_s1.current()
                v_new_l1_s1 = g_v_new_l1_s1.current()
                v_scaled_l1_s1 = g_v_scaled_l1_s1.current()
                scan_acc_s1 = g_scan_acc_s1.current()
                o_acc_s1 = g_o_acc_s1.current()
                cube_out_ub_s1 = g_cube_out_ub_s1.current()
                delta_ub_s1 = g_delta_ub_s1.current()
                q_decayed_l1 = g_q_decayed_l1.current()
                a_mat_l1 = g_a_mat_l1.current()
                k_t_l1 = g_k_t_l1.current()
                w_l1 = g_w_l1.current()

                # w, q_decayed and k^T do not depend on the state, so all three
                # full-K loads are issued before the state handshake.
                pl.set_validshape(w_l1, [C, Kd])
                pl.load(w_l1, w, [b, h, ni, 0, 0])
                pl.set_validshape(w_l1, [C, KMAX])
                pl.set_validshape(q_decayed_l1, [C, Kd])
                pl.load(q_decayed_l1, q_de, [b, h, ni, 0, 0])
                pl.set_validshape(q_decayed_l1, [C, KMAX])
                pl.set_validshape(k_t_l1, [Kd, C])
                pl.load(k_t_l1, k_f32, [b, h, ni, 0, 0], order=[4, 3])
                pl.set_validshape(k_t_l1, [KMAX, C])

                # v_prime = w @ S
                pl.system.wait_cross_core(pipe=_S, event_id=E_S_READY, sync_mode=_IB)
                for k_tile in pl.range(0, k_tiles):
                    k_base = k_tile * KT
                    k_count = pl.min(KT, Kd - k_base)
                    pl.set_validshape(state_l1_s0, [k_count, v_count_s0])
                    pl.load(state_l1_s0, s_hist, [b, h, ni, k_base, v_base_s0])
                    pl.set_validshape(state_l1_s0, [k_count, VTP])
                    w_l0a = g_l0a.next()
                    pl.set_validshape(w_l0a, [C, k_count])
                    pl.move(w_l0a, w_l1, offset=[0, k_base])
                    state_l0b = g_l0b.next()
                    pl.set_validshape(state_l0b, [k_count, VTP])
                    pl.move(state_l0b, state_l1_s0)
                    pl.set_validshape(scan_acc_s0, [C, VTP])
                    if k_tile == 0:
                        pl.matmul(scan_acc_s0, w_l0a, state_l0b)
                    else:
                        pl.matmul_acc(scan_acc_s0, scan_acc_s0, w_l0a, state_l0b)
                if second_stream:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_S_READY, sync_mode=_IB)
                    for k_tile in pl.range(0, k_tiles):
                        k_base = k_tile * KT
                        k_count = pl.min(KT, Kd - k_base)
                        pl.set_validshape(state_l1_s1, [k_count, v_count_s1])
                        pl.load(state_l1_s1, s_hist, [b, h, ni, k_base, v_base_s1])
                        pl.set_validshape(state_l1_s1, [k_count, VTP])
                        w_l0a = g_l0a.next()
                        pl.set_validshape(w_l0a, [C, k_count])
                        pl.move(w_l0a, w_l1, offset=[0, k_base])
                        state_l0b = g_l0b.next()
                        pl.set_validshape(state_l0b, [k_count, VTP])
                        pl.move(state_l0b, state_l1_s1)
                        pl.set_validshape(scan_acc_s1, [C, VTP])
                        if k_tile == 0:
                            pl.matmul(scan_acc_s1, w_l0a, state_l0b)
                        else:
                            pl.matmul_acc(scan_acc_s1, scan_acc_s1, w_l0a, state_l0b)

                pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
                pl.set_validshape(scan_acc_s0, [C, VTP])
                pl.move(cube_out_ub_s0, scan_acc_s0, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                pl.system.set_cross_core(pipe=_FIX, event_id=E_VP_READY, sync_mode=_IB)
                if second_stream:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
                    pl.set_validshape(scan_acc_s1, [C, VTP])
                    pl.move(cube_out_ub_s1, scan_acc_s1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=_FIX, event_id=E_VP_READY, sync_mode=_IB)

                # o_inter = q_decayed @ S
                for k_tile in pl.range(0, k_tiles):
                    k_base = k_tile * KT
                    k_count = pl.min(KT, Kd - k_base)
                    pl.set_validshape(state_l1_s0, [k_count, v_count_s0])
                    pl.load(state_l1_s0, s_hist, [b, h, ni, k_base, v_base_s0])
                    pl.set_validshape(state_l1_s0, [k_count, VTP])
                    q_l0a = g_l0a.next()
                    pl.set_validshape(q_l0a, [C, k_count])
                    pl.move(q_l0a, q_decayed_l1, offset=[0, k_base])
                    state_l0b = g_l0b.next()
                    pl.set_validshape(state_l0b, [k_count, VTP])
                    pl.move(state_l0b, state_l1_s0)
                    pl.set_validshape(o_acc_s0, [C, VTP])
                    if k_tile == 0:
                        pl.matmul(o_acc_s0, q_l0a, state_l0b)
                    else:
                        pl.matmul_acc(o_acc_s0, o_acc_s0, q_l0a, state_l0b)
                if second_stream:
                    for k_tile in pl.range(0, k_tiles):
                        k_base = k_tile * KT
                        k_count = pl.min(KT, Kd - k_base)
                        pl.set_validshape(state_l1_s1, [k_count, v_count_s1])
                        pl.load(state_l1_s1, s_hist, [b, h, ni, k_base, v_base_s1])
                        pl.set_validshape(state_l1_s1, [k_count, VTP])
                        q_l0a = g_l0a.next()
                        pl.set_validshape(q_l0a, [C, k_count])
                        pl.move(q_l0a, q_decayed_l1, offset=[0, k_base])
                        state_l0b = g_l0b.next()
                        pl.set_validshape(state_l0b, [k_count, VTP])
                        pl.move(state_l0b, state_l1_s1)
                        pl.set_validshape(o_acc_s1, [C, VTP])
                        if k_tile == 0:
                            pl.matmul(o_acc_s1, q_l0a, state_l0b)
                        else:
                            pl.matmul_acc(o_acc_s1, o_acc_s1, q_l0a, state_l0b)

                # o = o_inter + a_mat @ v_new
                pl.set_validshape(a_mat_l1, [C, C])
                pl.load(a_mat_l1, a_mat, [b, h, ni, 0, 0])
                pl.system.wait_cross_core(pipe=_S, event_id=E_VN_READY, sync_mode=_IB)
                pl.set_validshape(v_new_l1_s0, [C, v_count_s0])
                pl.load(v_new_l1_s0, v_new, [b, h, ni, 0, v_base_s0])
                pl.set_validshape(v_scaled_l1_s0, [C, v_count_s0])
                pl.load(v_scaled_l1_s0, v_scaled_gm, [b, h, 0, v_base_s0])
                pl.system.set_cross_core(pipe=_MTE2, event_id=E_VN_FREE, sync_mode=_IB)
                pl.set_validshape(v_new_l1_s0, [C, VTP])
                pl.set_validshape(v_scaled_l1_s0, [C, VTP])
                a_l0a = g_l0a.next()
                pl.set_validshape(a_l0a, [C, C])
                pl.move(a_l0a, a_mat_l1)
                v_new_l0b = g_l0b.next()
                pl.set_validshape(v_new_l0b, [C, VTP])
                pl.move(v_new_l0b, v_new_l1_s0)
                pl.set_validshape(o_acc_s0, [C, VTP])
                pl.matmul_acc(o_acc_s0, o_acc_s0, a_l0a, v_new_l0b)
                if second_stream:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VN_READY, sync_mode=_IB)
                    pl.set_validshape(v_new_l1_s1, [C, v_count_s1])
                    pl.load(v_new_l1_s1, v_new, [b, h, ni, 0, v_base_s1])
                    pl.set_validshape(v_scaled_l1_s1, [C, v_count_s1])
                    pl.load(v_scaled_l1_s1, v_scaled_gm, [b, h, 0, v_base_s1])
                    pl.system.set_cross_core(pipe=_MTE2, event_id=E_VN_FREE, sync_mode=_IB)
                    pl.set_validshape(v_new_l1_s1, [C, VTP])
                    pl.set_validshape(v_scaled_l1_s1, [C, VTP])
                    a_l0a = g_l0a.next()
                    pl.set_validshape(a_l0a, [C, C])
                    pl.move(a_l0a, a_mat_l1)
                    v_new_l0b = g_l0b.next()
                    pl.set_validshape(v_new_l0b, [C, VTP])
                    pl.move(v_new_l0b, v_new_l1_s1)
                    pl.set_validshape(o_acc_s1, [C, VTP])
                    pl.matmul_acc(o_acc_s1, o_acc_s1, a_l0a, v_new_l0b)

                pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
                pl.set_validshape(o_acc_s0, [C, VTP])
                pl.move(cube_out_ub_s0, o_acc_s0, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                pl.system.set_cross_core(pipe=_FIX, event_id=E_VP_READY, sync_mode=_IB)
                if second_stream:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
                    pl.set_validshape(o_acc_s1, [C, VTP])
                    pl.move(cube_out_ub_s1, o_acc_s1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=_FIX, event_id=E_VP_READY, sync_mode=_IB)

                # state increment = k^T @ v_scaled, one K tile at a time
                for k_tile in pl.range(0, k_tiles):
                    k_base = k_tile * KT
                    k_count = pl.min(KT, Kd - k_base)
                    k_t_l0a = g_l0a.next()
                    pl.set_validshape(k_t_l0a, [C, C])
                    pl.move(k_t_l0a, k_t_l1, offset=[k_base, 0])
                    v_scaled_l0b = g_l0b.next()
                    pl.set_validshape(v_scaled_l0b, [C, VTP])
                    pl.move(v_scaled_l0b, v_scaled_l1_s0)
                    pl.set_validshape(scan_acc_s0, [C, VTP])
                    pl.matmul(scan_acc_s0, k_t_l0a, v_scaled_l0b)
                    pl.system.wait_cross_core(pipe=_S, event_id=E_D_FREE, sync_mode=_IB)
                    pl.set_validshape(scan_acc_s0, [C, VTP])
                    pl.move(delta_ub_s0, scan_acc_s0, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=_FIX, event_id=E_D_READY, sync_mode=_IB)
                    if second_stream:
                        k_t_l0a = g_l0a.next()
                        pl.set_validshape(k_t_l0a, [C, C])
                        pl.move(k_t_l0a, k_t_l1, offset=[k_base, 0])
                        v_scaled_l0b = g_l0b.next()
                        pl.set_validshape(v_scaled_l0b, [C, VTP])
                        pl.move(v_scaled_l0b, v_scaled_l1_s1)
                        pl.set_validshape(scan_acc_s1, [C, VTP])
                        pl.matmul(scan_acc_s1, k_t_l0a, v_scaled_l0b)
                        pl.system.wait_cross_core(pipe=_S, event_id=E_D_FREE, sync_mode=_IB)
                        pl.set_validshape(scan_acc_s1, [C, VTP])
                        pl.move(delta_ub_s1, scan_acc_s1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=_FIX, event_id=E_D_READY, sync_mode=_IB)

            # Drain this item's outstanding credits, one set per active stream.
            pl.system.wait_cross_core(pipe=_S, event_id=E_S_READY, sync_mode=_IB)
            pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
            pl.system.wait_cross_core(pipe=_S, event_id=E_D_FREE, sync_mode=_IB)
            if second_stream:
                pl.system.wait_cross_core(pipe=_S, event_id=E_S_READY, sync_mode=_IB)
                pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
                pl.system.wait_cross_core(pipe=_S, event_id=E_D_FREE, sync_mode=_IB)

    with pl.section_vector():
        core_count = pl.get_block_num()
        core = pl.get_block_idx() // pl.get_subblock_num()
        subblock = pl.get_subblock_idx()
        row_base = subblock * CH
        k_tiles = (Kd + KT - 1) // KT
        task_count = B * H * vpairs
        tasks_per_core = (task_count + core_count - 1) // core_count
        first_task = tasks_per_core * core
        task_end = pl.min(first_task + tasks_per_core, task_count)

        for tid in pl.range(first_task, task_end, 1):
            bh = tid // vpairs
            v_pair = tid % vpairs
            b = bh // H
            h = bh % H
            v_tile_s0 = v_pair * SR
            v_tile_limit = pl.min(vtiles, v_tile_s0 + SR)
            second_stream = (v_tile_s0 + 1) < v_tile_limit
            v_base_s0 = v_tile_s0 * VT
            v_count_s0 = pl.min(VT, Vd - v_base_s0)
            v_base_s1 = (v_tile_s0 + 1) * VT
            v_count_s1 = pl.min(VT, Vd - v_base_s1)

            # Hoist every tile handle once per item for the mutex analysis.
            v_new_ub_s0 = g_v_new_ub_s0.current()
            v_scaled_ub_s0 = g_v_scaled_ub_s0.current()
            cube_out_ub_s0 = g_cube_out_ub_s0.current()
            delta_ub_s0 = g_delta_ub_s0.current()
            o_bf16_ub_s0 = g_o_bf16_ub_s0.current()
            v_new_ub_s1 = g_v_new_ub_s1.current()
            v_scaled_ub_s1 = g_v_scaled_ub_s1.current()
            cube_out_ub_s1 = g_cube_out_ub_s1.current()
            delta_ub_s1 = g_delta_ub_s1.current()
            o_bf16_ub_s1 = g_o_bf16_ub_s1.current()
            decay_factor_ub = g_decay_factor_ub.current()
            decay_exp_ub = g_decay_exp_ub.current()

            # Publish the zero entering state of chunk 0 for this item.
            for k_tile in pl.range(0, k_tiles):
                k_base = k_tile * KT
                k_count = pl.min(KT, Kd - k_base)
                row_count = pl.min(CH, pl.max(0, k_count - row_base))
                state_ub_s0 = g_state_ub_s0[k_tile]
                if row_count > 0:
                    pl.set_validshape(state_ub_s0, [row_count, v_count_s0])
                    zero_state_vf(state_ub_s0, row_count)
                    pl.set_validshape(state_ub_s0, [row_count, v_count_s0])
                    pl.store(s_hist, state_ub_s0, [b, h, 0, k_base + row_base, v_base_s0])
            if second_stream:
                for k_tile in pl.range(0, k_tiles):
                    k_base = k_tile * KT
                    k_count = pl.min(KT, Kd - k_base)
                    row_count = pl.min(CH, pl.max(0, k_count - row_base))
                    state_ub_s1 = g_state_ub_s1[k_tile]
                    if row_count > 0:
                        pl.set_validshape(state_ub_s1, [row_count, v_count_s1])
                        zero_state_vf(state_ub_s1, row_count)
                        pl.set_validshape(state_ub_s1, [row_count, v_count_s1])
                        pl.store(s_hist, state_ub_s1, [b, h, 0, k_base + row_base, v_base_s1])
            pl.system.set_cross_core(pipe=_MTE3, event_id=E_S_READY, sync_mode=_IB)
            pl.system.set_cross_core(pipe=_V, event_id=E_VP_FREE, sync_mode=_IB)
            pl.system.set_cross_core(pipe=_V, event_id=E_D_FREE, sync_mode=_IB)
            if second_stream:
                pl.system.set_cross_core(pipe=_MTE3, event_id=E_S_READY, sync_mode=_IB)
                pl.system.set_cross_core(pipe=_V, event_id=E_VP_FREE, sync_mode=_IB)
                pl.system.set_cross_core(pipe=_V, event_id=E_D_FREE, sync_mode=_IB)

            for ni in pl.range(0, N, 1):
                # v_new = u - v_prime, and the decay-scaled copy for the state.
                pl.set_validshape(v_new_ub_s0, [CH, v_count_s0])
                pl.load(v_new_ub_s0, u, [b, h, ni, row_base, v_base_s0])
                pl.set_validshape(decay_factor_ub, [CH, 8])
                pl.load(decay_factor_ub, dfac, [b, h, ni, row_base, 0])
                pl.system.wait_cross_core(pipe=_S, event_id=E_VP_READY, sync_mode=_IB)
                pl.set_validshape(cube_out_ub_s0, [CH, v_count_s0])
                subtract_and_scale_vf(cube_out_ub_s0, v_new_ub_s0, decay_factor_ub,
                                      v_scaled_ub_s0, CH, v_count_s0)
                pl.system.set_cross_core(pipe=_V, event_id=E_VP_FREE, sync_mode=_IB)
                if second_stream:
                    pl.set_validshape(v_new_ub_s1, [CH, v_count_s1])
                    pl.load(v_new_ub_s1, u, [b, h, ni, row_base, v_base_s1])
                    pl.set_validshape(decay_factor_ub, [CH, 8])
                    pl.load(decay_factor_ub, dfac, [b, h, ni, row_base, 0])
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VP_READY, sync_mode=_IB)
                    pl.set_validshape(cube_out_ub_s1, [CH, v_count_s1])
                    subtract_and_scale_vf(cube_out_ub_s1, v_new_ub_s1, decay_factor_ub,
                                          v_scaled_ub_s1, CH, v_count_s1)
                    pl.system.set_cross_core(pipe=_V, event_id=E_VP_FREE, sync_mode=_IB)

                pl.system.wait_cross_core(pipe=_S, event_id=E_VN_FREE, sync_mode=_IB)
                pl.set_validshape(v_new_ub_s0, [CH, v_count_s0])
                pl.store(v_new, v_new_ub_s0, [b, h, ni, row_base, v_base_s0])
                pl.set_validshape(v_scaled_ub_s0, [CH, v_count_s0])
                pl.store(v_scaled_gm, v_scaled_ub_s0, [b, h, row_base, v_base_s0])
                pl.system.set_cross_core(pipe=_MTE3, event_id=E_VN_READY, sync_mode=_IB)
                if second_stream:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VN_FREE, sync_mode=_IB)
                    pl.set_validshape(v_new_ub_s1, [CH, v_count_s1])
                    pl.store(v_new, v_new_ub_s1, [b, h, ni, row_base, v_base_s1])
                    pl.set_validshape(v_scaled_ub_s1, [CH, v_count_s1])
                    pl.store(v_scaled_gm, v_scaled_ub_s1, [b, h, row_base, v_base_s1])
                    pl.system.set_cross_core(pipe=_MTE3, event_id=E_VN_READY, sync_mode=_IB)

                # The output band, narrowed to bfloat16 on the way to GM.
                pl.set_validshape(decay_exp_ub, [1, C])
                pl.load(decay_exp_ub, decay_exp, [b, h, ni, 0])
                pl.system.wait_cross_core(pipe=_S, event_id=E_VP_READY, sync_mode=_IB)
                pl.set_validshape(cube_out_ub_s0, [CH, v_count_s0])
                pl.set_validshape(o_bf16_ub_s0, [CH, v_count_s0])
                pl.cast(o_bf16_ub_s0, cube_out_ub_s0, mode=pl.RoundMode.CAST_ROUND)
                pl.set_validshape(o_bf16_ub_s0, [CH, v_count_s0])
                # o is in the public [B, T, H, V] layout: order= makes the tile
                # walk (T, V) rather than the default trailing (H, V).
                pl.store(o, o_bf16_ub_s0, [b, ni * C + row_base, h, v_base_s0], order=[1, 3])
                pl.system.set_cross_core(pipe=_MTE3, event_id=E_VP_FREE, sync_mode=_IB)
                if second_stream:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VP_READY, sync_mode=_IB)
                    pl.set_validshape(cube_out_ub_s1, [CH, v_count_s1])
                    pl.set_validshape(o_bf16_ub_s1, [CH, v_count_s1])
                    pl.cast(o_bf16_ub_s1, cube_out_ub_s1, mode=pl.RoundMode.CAST_ROUND)
                    pl.set_validshape(o_bf16_ub_s1, [CH, v_count_s1])
                    pl.store(o, o_bf16_ub_s1, [b, ni * C + row_base, h, v_base_s1], order=[1, 3])
                    pl.system.set_cross_core(pipe=_MTE3, event_id=E_VP_FREE, sync_mode=_IB)

                # Advance the state, and publish it as the next chunk's entry.
                for k_tile in pl.range(0, k_tiles):
                    k_base = k_tile * KT
                    k_count = pl.min(KT, Kd - k_base)
                    row_count = pl.min(CH, pl.max(0, k_count - row_base))
                    pl.system.wait_cross_core(pipe=_S, event_id=E_D_READY, sync_mode=_IB)
                    state_ub_s0 = g_state_ub_s0[k_tile]
                    if row_count > 0:
                        pl.set_validshape(delta_ub_s0, [row_count, v_count_s0])
                        state_update_vf(state_ub_s0, delta_ub_s0, decay_exp_ub,
                                        row_count, v_count_s0, C - 1)
                        if ni + 1 < N:
                            pl.set_validshape(state_ub_s0, [row_count, v_count_s0])
                            pl.store(s_hist, state_ub_s0,
                                     [b, h, ni + 1, k_base + row_base, v_base_s0])
                    pl.system.set_cross_core(pipe=_V, event_id=E_D_FREE, sync_mode=_IB)
                    if second_stream:
                        pl.system.wait_cross_core(pipe=_S, event_id=E_D_READY, sync_mode=_IB)
                        state_ub_s1 = g_state_ub_s1[k_tile]
                        if row_count > 0:
                            pl.set_validshape(delta_ub_s1, [row_count, v_count_s1])
                            state_update_vf(state_ub_s1, delta_ub_s1, decay_exp_ub,
                                            row_count, v_count_s1, C - 1)
                            if ni + 1 < N:
                                pl.set_validshape(state_ub_s1, [row_count, v_count_s1])
                                pl.store(s_hist, state_ub_s1,
                                         [b, h, ni + 1, k_base + row_base, v_base_s1])
                        pl.system.set_cross_core(pipe=_V, event_id=E_D_FREE, sync_mode=_IB)
                pl.system.set_cross_core(pipe=_MTE3, event_id=E_S_READY, sync_mode=_IB)
                if second_stream:
                    pl.system.set_cross_core(pipe=_MTE3, event_id=E_S_READY, sync_mode=_IB)

            # Drain the cube's bootstrap credits for this item.
            pl.system.wait_cross_core(pipe=_S, event_id=E_VN_FREE, sync_mode=_IB)
            if second_stream:
                pl.system.wait_cross_core(pipe=_S, event_id=E_VN_FREE, sync_mode=_IB)


def _pick_vt_streams(B, H, V, cube_cores):
    """Choose the V tile width and the number of interleaved streams.

    The item space is (batch, head, V-tile group) and every item costs the same,
    so the critical path is the number of launch waves times the number of
    streams a single item serialises::

        cost = ceil(items / cores) * streams

    A narrower V tile does not make an item cheaper, because matmul and fixpipe
    always run on the full physical tile; it only creates more items. So the
    width is chosen by minimising that cost, breaking ties first towards filling
    more cube cores and then towards the widest tile, which issues the fewest
    full-width matmuls and moves the least data.
    """
    best = None
    for vt in (64, 48, 32):
        vtiles = (V + vt - 1) // vt
        for streams in (2, 1):
            vpairs = (vtiles + streams - 1) // streams
            tasks = B * H * vpairs
            cores = min(cube_cores, tasks)
            waves = (tasks + cores - 1) // cores
            candidate = ((waves * streams, -min(tasks, cube_cores), -vt), (cores, streams, vt))
            best = candidate if best is None or candidate < best else best
    return best[1]


def gdr_fwd_train_scan_wrapper(u, w, q_de, a_mat, k_f32, dfac, decay_exp,
                               *, _streams=None, _vt=None, sync=True,
                               cube_core_budget=None):
    """Run the forward state scan in one launch.

    `cube_core_budget` caps the cube cores this launch plans its V tiling for.
    The default plans for the whole part, which is correct when the launch owns
    the device. A caller that runs this launch concurrently with another passes
    the cores the other one leaves free, so that the two block counts sum to the
    part rather than oversubscribing it. The budget changes only how an
    independent item space folds onto cores, never the arithmetic.

    The wrapper validates the shapes, allocates the three outputs and one GM
    scratch, picks the tiling, and launches once.

    `_streams` and `_vt` override the tiling for A/B measurement only; they
    change neither the arithmetic nor any output.

    Returns (o [B, C*N, H, V] bfloat16, s_hist [B, H, N, K, V] float32,
    v_new [B, H, N, C, V] float32).
    """
    B, H, N, chunk, V = u.shape
    K = w.shape[-1]
    if chunk != C:
        raise ValueError(f"chunk dim must be {C}, got {chunk}")
    if K > KMAX or V > KMAX:
        raise ValueError(f"supported range is K, V <= {KMAX}; got K={K}, V={V}")
    device = u.device
    padded_tokens = C * N

    o = torch.empty((B, padded_tokens, H, V), dtype=torch.bfloat16, device=device)
    s_hist = torch.empty((B, H, N, K, V), dtype=torch.float32, device=device)
    v_new = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    # Carries the decay-scaled v_new from the vector core to the cube operand.
    v_scaled_gm = torch.empty((B, H, C, V), dtype=torch.float32, device=device)

    cube_cores = get_platform_info().core_num or 1
    if cube_core_budget is not None:
        cube_cores = max(1, min(cube_cores, int(cube_core_budget)))
    cores, streams, vt = _pick_vt_streams(B, H, V, cube_cores)
    if _vt is not None:
        vt = _vt
    if _streams is not None:
        streams = _streams
    vtiles = (V + vt - 1) // vt
    vpairs = (vtiles + streams - 1) // streams
    tasks = B * H * vpairs
    num_cores = max(1, min(cube_cores, tasks))

    gdr_fwd_train_scan_module1_kernel[None, num_cores](
        u, w, q_de, a_mat, k_f32, dfac, decay_exp, v_scaled_gm,
        o, s_hist, v_new,
        B, H, K, V, N, vt, vtiles, vpairs, streams)
    if sync:
        torch.npu.synchronize()
    return o, s_hist, v_new
