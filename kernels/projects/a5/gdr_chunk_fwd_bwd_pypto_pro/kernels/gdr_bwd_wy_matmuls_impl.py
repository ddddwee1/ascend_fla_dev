# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Matmul half of the WY adjoint.

The training forward replaces the sequential delta rule inside a chunk by a WY
transform: a unit lower-triangular factor attn is built once and applied to the
key and value rows. Differentiating that factor is this operator's job. It is
the matmul-dense member of the pipeline, with eight matmuls per chunk arranged
in three cube phases separated by two vector phases, because the adjoint of the
triangular factor has to be masked and gated between products.

Mathematics
-----------
Per chunk, all tensors chunk-local, writing A for attn::

    phase A   dA   = dw @ kbeta_t  +  dv2 @ v_b^T
              dkbg = A^T @ dw                          output
              dvb  = A^T @ dv2                         output
    mask      dA  <- strict_lower(dA)                  keep column < row
    phase B   dA  <- A^T @ (dA @ A^T)
    gate      d_a  = -(dA * exp(min(decay_row - decay_col, 0)))   output, strict lower
    phase C   dkb  = d_a @ k_f32                       output
              dkt  = d_a^T @ k_beta                    output

k_beta = k_f32 * beta_act is formed on chip rather than read from GM, being one
cheap vector product. The gate factor is masked before the exponential, so the
upper triangle never produces an infinity to cancel later. The reductions that
complete dbeta and dg do not need any matmul: they collapse into row and column
sums of dkb and dkt, which is why they live in the reduction operator instead.

Interface
---------
``gdr_bwd_wy_matmuls_wrapper(attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2)``

    attn           [B, H, N, C, C] f32   the WY factor from the training forward
    k_f32          [B, H, N, C, K] f32   k widened to float32
    v_b            [B, H, N, C, V] f32   v scaled by beta
    kbeta_t        [B, H, N, K, C] f32   pre-transposed k * beta
    beta_act       [B, H, N, C] f32      the activated gate weight
    decay          [B, H, N, C] f32      within-chunk cumulative gate sum
    dw             [B, H, N, C, K] f32   gradient of the WY key rows
    dv2            [B, H, N, C, V] f32   dv after the reverse scan

    returns dkbg, dkb, dkt  [B, H, N, C, K] f32
            dvb             [B, H, N, C, V] f32
            d_a             [B, H, N, C, C] f32

Every input is chunk-local: this operator reads no public [B, T, H, *] tensor.

Layout and scheduling
---------------------
One launch of one mixed cube/vector kernel. A work item is one
(batch, head, chunk) triple. Both sections walk the same contiguous task range,
because `pl.get_block_idx() // pl.get_subblock_num()` yields the same physical
core index on either side.

L0A and L0B are the binding resources. Declaring them at the per-matmul [64, 64]
shape and splitting the contraction axis keeps each at half of its 64 KB
capacity, where the naive full-width operand would sit exactly at the cap with
no room for the NZ staging a transposed operand needs.

Transposed operands never exist in GM. attn^T, v_b^T and d_a^T each reach L1
through a descending-order load, and attn^T is loaded once per item and reused
as the left operand of three matmuls and the right operand of a fourth, since
`pl.move` lays an L1 tile out for whichever L0 fractal the operand role needs.

Three hand-offs cross between the cores, and each is shaped by a hardware rule:

* The masked dA goes back to the cube through GM. It is a required output
  anyway, so the read costs no extra write. The mask itself is row-local, so the
  drain splits the matrix across both vector sub-blocks, each masking and
  storing its own 32-row band; nothing is written twice and no sub-block guard
  is needed.
* The bridge between the two phase-B matmuls bounces L0C to UB to L1, because
  there is no accumulator-to-L1 path on this part: the legal moves are
  Mat to Left/Right/Scaling/Bias, Acc to Vec, Vec to Vec and Vec to Mat. The UB
  relay tile is declared NZ so that both hops are plain byte copies. That copy
  is guarded to sub-block 0, while the event that releases it is not: a
  cross-core wait is satisfied only once every vector lane has signalled.
* The dA hand-off rotates over four UB slots. With a single slot the two cores
  strictly alternate; the cube needs about three items of slack before the
  hand-off stops gating it, and four slots cost 64 KB of UB.
"""

import logging

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

C = 64        # tokens per chunk: the row count of every tile, and both dims of attn
CH = C // 2   # rows per vector sub-block after a DualModeSplitM drain
TILE = 64     # contraction-axis split step, and output sub-block width
KMAX = 128    # supported K upper bound, and the physical L1 tile width
VMAX = 128    # supported V upper bound

# L1 address map.
L1_DW = 0x00000             # [C, KMAX]
L1_KBETA_T = 0x08000        # [KMAX, C]
L1_ATTN_T = 0x10000         # [C, C]
L1_DV2 = 0x14000            # [C, VMAX]
L1_V_B_T = 0x1C000          # [VMAX, C]
L1_D_ATTN_MASKED = 0x24000  # [C, C]   phase B left operand
L1_BRIDGE = 0x28000         # [C, C]   between the two phase-B matmuls
L1_D_A = 0x2C000            # [C, C]   phase C left operand
L1_D_A_T = 0x30000          # [C, C]   phase C left operand, transposed
L1_K = 0x34000              # [C, KMAX]
L1_K_BETA = 0x3C000         # [C, KMAX]   high-water 0x44000 = 278,528 B

# L0 address maps. Two [64, 64] float32 slots per operand buffer is half of the
# 64 KB capacity, which leaves room for the NZ staging of transposed operands.
L0A_SLOTS = (0x0000, 0x4000)
L0B_SLOTS = (0x0000, 0x4000)
L0C_D_ATTN = 0x0000         # [C, C]            the dA accumulator
L0C_OUT_A = 0x4000          # 2 x [C, C]        dkbg, then dvb
L0C_BRACKET = 0xC000        # [C, C]            dA @ A^T
L0C_TWO_SIDED = 0x10000     # [C, C]            A^T @ (dA @ A^T)
L0C_OUT_C = 0x14000         # 2 x [C, C]        dkb, then dkt
                            #                   high-water 0x1C000 = 114,688 B

# UB address map.
UB_D_ATTN_SLOTS = (0x00000, 0x04000, 0x08000, 0x0C000)   # 4 x [C, C]
UB_BRIDGE = 0x10000                                      # [C, C] NZ
UB_TWO_SIDED_SLOTS = (0x14000, 0x18000, 0x2D000, 0x31000)  # 4 x [C, C]
UB_DECAY = 0x1C000                                       # [1, C]
UB_BETA = 0x1C100                                        # [1, C]
UB_K = 0x1D000                                           # [C, KMAX]
UB_K_BETA = 0x25000                                      # [C, KMAX]
                                                         # high-water 0x35000 = 217,088 B
HANDOFF_DEPTH = 4

# Cross-core event ids.
EV_D_ATTN_READY = 0      # cube -> vector: dA drained to UB
EV_D_ATTN_FREE = 1       # vector -> cube: that UB slot is free again
EV_MASKED_IN_GM = 2      # vector -> cube: the masked dA is committed to GM
EV_BRIDGE_READY = 3      # cube -> vector: the phase-B intermediate is in UB
EV_BRIDGE_RELAYED = 4    # vector -> cube: it has been relayed into L1
EV_TWO_SIDED_READY = 5   # cube -> vector: the two-sided product is in UB
EV_TWO_SIDED_FREE = 6    # vector -> cube: that UB slot is free again
EV_D_A_IN_GM = 7         # vector -> cube: d_a is committed to GM
EV_K_BETA_IN_GM = 8      # vector -> cube: k_beta is committed to GM


@pl.vector_function
def mask_strict_lower_vf(d_attn, n_rows: pl.DT_INT64, row_base: pl.DT_INT64):
    """Keep dA[r, j] for j < row_base + r and zero the rest, over one row band.

    `row_base` biases the local row index to the global one, so the boundary is
    correct in each sub-block's band. One [C, C] float32 row is exactly one
    64-lane register, so a row is one load, one select and one store.
    """
    keep_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, keep_all, dtype=pl.DT_FP32)
    for row in pl.range(0, n_rows, 1):
        value = vf.load_align(d_attn, row * C, dtype=pl.DT_FP32)
        keep = vf.update_mask(pl.max(0, row_base + row), dtype=pl.DT_FP32)
        value = vf.select(value, zero, keep)
        vf.store_align(d_attn + row * C, value, keep_all)


@pl.vector_function
def gate_and_negate_vf(d_attn, decay, n_rows: pl.DT_INT64, row_base: pl.DT_INT64):
    """d_a[r, j] = -(dA[r, j] * exp(min(decay_row - decay_j, 0))) for j < row.

    Clamping the exponent at zero before the exponential keeps the upper
    triangle out of its range; the select then restores the strict-lower shape.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, all_lanes, dtype=pl.DT_FP32)
    decay_row = vf.load_align(decay, 0, dtype=pl.DT_FP32)
    for row in pl.range(0, n_rows, 1):
        decay_i = vf.load_align(decay, row_base + row, dist=pl.LoadDist.BRC_B32)
        exponent = vf.sub(decay_i, decay_row, all_lanes)
        exponent = vf.mins(exponent, 0.0, all_lanes)
        gate = vf.exp(exponent, all_lanes)
        value = vf.load_align(d_attn, row * C, dtype=pl.DT_FP32)
        value = vf.mul(value, gate, all_lanes)
        value = vf.muls(value, -1.0, all_lanes)
        keep = vf.update_mask(pl.max(0, row_base + row), dtype=pl.DT_FP32)
        value = vf.select(value, zero, keep)
        vf.store_align(d_attn + row * C, value, all_lanes)


@pl.vector_function
def form_k_beta_vf(k, beta, k_beta, n_rows: pl.DT_INT64, row_base: pl.DT_INT64):
    """k_beta[row_base + r] = k_f32[row_base + r] * beta_act[row_base + r].

    The result lands at local row r so that a sub-block can store its band from
    the tile base. A register holds 64 float32 lanes, so a row up to KMAX wide
    is processed as two halves; one full-width load would drop the upper lanes.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for row in pl.range(0, n_rows, 1):
        beta_row = vf.load_align(beta, row_base + row, dist=pl.LoadDist.BRC_B32)
        low_half = vf.load_align(k, (row_base + row) * KMAX, dtype=pl.DT_FP32)
        high_half = vf.load_align(k, (row_base + row) * KMAX + C, dtype=pl.DT_FP32)
        low_half = vf.mul(low_half, beta_row, all_lanes)
        high_half = vf.mul(high_half, beta_row, all_lanes)
        vf.store_align(k_beta + row * KMAX, low_half, all_lanes)
        vf.store_align(k_beta + row * KMAX + C, high_half, all_lanes)


_DYN5 = [pl.DYNAMIC] * 5
_DYN4 = [pl.DYNAMIC] * 4


@pl.jit(auto_mutex=True)
def gdr_bwd_wy_matmuls_kernel(
    attn: pl.Tensor[_DYN5, pl.DT_FP32],
    k_f32: pl.Tensor[_DYN5, pl.DT_FP32],
    beta_act: pl.Tensor[_DYN4, pl.DT_FP32],
    decay: pl.Tensor[_DYN4, pl.DT_FP32],
    v_b: pl.Tensor[_DYN5, pl.DT_FP32],
    kbeta_t: pl.Tensor[_DYN5, pl.DT_FP32],
    dw: pl.Tensor[_DYN5, pl.DT_FP32],
    dv2: pl.Tensor[_DYN5, pl.DT_FP32],
    dkbg: pl.Tensor[_DYN5, pl.DT_FP32],
    dvb: pl.Tensor[_DYN5, pl.DT_FP32],
    d_attn_raw: pl.Tensor[_DYN5, pl.DT_FP32],
    d_attn_masked: pl.Tensor[_DYN5, pl.DT_FP32],
    d_attn_two_sided: pl.Tensor[_DYN5, pl.DT_FP32],
    d_a: pl.Tensor[_DYN5, pl.DT_FP32],
    k_beta: pl.Tensor[_DYN5, pl.DT_FP32],
    dkb: pl.Tensor[_DYN5, pl.DT_FP32],
    dkt: pl.Tensor[_DYN5, pl.DT_FP32],
):
    g_dw_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_DW, mutex_ids=[9])
    g_kbeta_t_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[KMAX, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_KBETA_T, mutex_ids=[10])
    g_attn_t_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_ATTN_T, mutex_ids=[11])
    # dw, dv2 and v_b^T are three disjoint L1 operands loaded at the top of an
    # item, so they hold three mutex ids: sharing one would serialise 96 KB of
    # independent transfers on the pipe where sharing hurts most.
    g_dv2_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, VMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_DV2, mutex_ids=[26])
    g_v_b_t_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[VMAX, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_V_B_T, mutex_ids=[29])

    g_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[C, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1]),
        addrs=list(L0A_SLOTS), mutex_ids=[0, 1])
    g_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1]),
        addrs=list(L0B_SLOTS), mutex_ids=[2, 3])

    g_d_attn_acc = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]),
        addrs=L0C_D_ATTN, mutex_ids=[12])
    g_out_acc_a = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]),
        addrs=L0C_OUT_A, mutex_ids=[13, 14])

    g_d_attn_ub = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=list(UB_D_ATTN_SLOTS), mutex_ids=[4, 5, 6, 7])

    # Phase B holds its own L1 and accumulator tiles: a phase that shares them
    # with another serialises against it.
    g_d_attn_masked_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_D_ATTN_MASKED, mutex_ids=[15])
    g_bridge_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_BRIDGE, mutex_ids=[16])
    # Declared NZ so that the drain is Acc(NZ) to Vec(NZ) and the relay a legal
    # Vec(NZ) to Mat(NZ) byte copy: a UB tile holding ND may not be copied into
    # an NZ or ZN L1 tile, and a float32 L1 tile cannot be ND.
    g_bridge_ub = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=UB_BRIDGE, mutex_ids=[19])

    g_two_sided_ub = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=list(UB_TWO_SIDED_SLOTS), mutex_ids=[20, 21, 22, 23])
    g_decay_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=UB_DECAY, mutex_ids=[8])
    g_beta_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=UB_BETA, mutex_ids=[19])
    g_k_ub = pl.make_tile_group(
        type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=UB_K, mutex_ids=[24])
    g_k_beta_ub = pl.make_tile_group(
        type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=UB_K_BETA, mutex_ids=[25])

    g_d_a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_D_A, mutex_ids=[15])
    g_d_a_t_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_D_A_T, mutex_ids=[27])
    g_k_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_K, mutex_ids=[28])
    g_k_beta_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_K_BETA, mutex_ids=[28])
    g_out_acc_c = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]),
        addrs=L0C_OUT_C, mutex_ids=[30, 31])
    g_bracket_acc = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]),
        addrs=L0C_BRACKET, mutex_ids=[17])
    g_two_sided_acc = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]),
        addrs=L0C_TWO_SIDED, mutex_ids=[18])

    with pl.section_cube():
        core_count = pl.get_block_num()
        core = pl.get_block_idx() // pl.get_subblock_num()

        B = dw.shape[0]
        H = dw.shape[1]
        N = dw.shape[2]
        Kd = dw.shape[4]
        Vd = dv2.shape[4]

        chunks_per_batch = H * N
        task_count = B * chunks_per_batch
        k_blocks = (Kd + TILE - 1) // TILE
        v_blocks = (Vd + TILE - 1) // TILE

        tasks_per_core = (task_count + core_count - 1) // core_count
        first_task = tasks_per_core * core
        task_end = pl.min(first_task + tasks_per_core, task_count)

        dw_l1 = g_dw_l1.current()
        kbeta_t_l1 = g_kbeta_t_l1.current()
        attn_t_l1 = g_attn_t_l1.current()
        dv2_l1 = g_dv2_l1.current()
        v_b_t_l1 = g_v_b_t_l1.current()
        d_attn_acc = g_d_attn_acc.current()
        d_attn_masked_l1 = g_d_attn_masked_l1.current()
        bridge_l1 = g_bridge_l1.current()
        bracket_acc = g_bracket_acc.current()
        bridge_ub = g_bridge_ub.current()
        d_a_l1 = g_d_a_l1.current()
        d_a_t_l1 = g_d_a_t_l1.current()
        k_l1 = g_k_l1.current()
        k_beta_l1 = g_k_beta_l1.current()
        two_sided_acc = g_two_sided_acc.current()

        for tid in pl.range(first_task, task_end, 1):
            b = tid // chunks_per_batch
            within_batch = tid % chunks_per_batch
            h = within_batch // N
            n = within_batch % N

            pl.set_validshape(dw_l1, [C, Kd])
            pl.load(dw_l1, dw, [b, h, n, 0, 0])
            pl.set_validshape(kbeta_t_l1, [Kd, C])
            pl.load(kbeta_t_l1, kbeta_t, [b, h, n, 0, 0])
            # attn^T and v_b^T by descending-order loads; no transposed copy
            # of either exists in GM.
            pl.set_validshape(attn_t_l1, [C, C])
            pl.load(attn_t_l1, attn, [b, h, n, 0, 0], order=[4, 3])
            pl.set_validshape(dv2_l1, [C, Vd])
            pl.load(dv2_l1, dv2, [b, h, n, 0, 0])
            pl.set_validshape(v_b_t_l1, [Vd, C])
            pl.load(v_b_t_l1, v_b, [b, h, n, 0, 0], order=[4, 3])

            # dA = dw @ kbeta_t, splitting the contraction over K
            for k_block in pl.range(0, k_blocks):
                k_base = k_block * TILE
                contract_count = pl.min(TILE, Kd - k_base)
                dw_l0a = g_l0a.next()
                kbeta_t_l0b = g_l0b.next()
                pl.set_validshape(dw_l1, [C, contract_count])
                pl.move(dw_l0a, dw_l1, offset=[0, k_base])
                pl.set_validshape(dw_l0a, [C, contract_count])
                pl.set_validshape(kbeta_t_l1, [contract_count, C])
                pl.move(kbeta_t_l0b, kbeta_t_l1, offset=[k_base, 0])
                pl.set_validshape(kbeta_t_l0b, [contract_count, C])
                pl.set_validshape(d_attn_acc, [C, C])
                if k_block == 0:
                    pl.matmul(d_attn_acc, dw_l0a, kbeta_t_l0b)
                else:
                    pl.matmul_acc(d_attn_acc, d_attn_acc, dw_l0a, kbeta_t_l0b)

            # dA += dv2 @ v_b^T, splitting the contraction over V
            for v_block in pl.range(0, v_blocks):
                v_base = v_block * TILE
                contract_count = pl.min(TILE, Vd - v_base)
                dv2_l0a = g_l0a.next()
                v_b_t_l0b = g_l0b.next()
                pl.set_validshape(dv2_l1, [C, contract_count])
                pl.move(dv2_l0a, dv2_l1, offset=[0, v_base])
                pl.set_validshape(dv2_l0a, [C, contract_count])
                pl.set_validshape(v_b_t_l1, [contract_count, C])
                pl.move(v_b_t_l0b, v_b_t_l1, offset=[v_base, 0])
                pl.set_validshape(v_b_t_l0b, [contract_count, C])
                pl.set_validshape(d_attn_acc, [C, C])
                pl.matmul_acc(d_attn_acc, d_attn_acc, dv2_l0a, v_b_t_l0b)

            pl.set_validshape(d_attn_acc, [C, C])
            pl.store(d_attn_raw, d_attn_acc, [b, h, n, 0, 0])

            # Only here can the cube stall: every load and three of the four
            # matmul groups of this item are already issued.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_D_ATTN_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # Drain dA to both vector sub-blocks, rows 0..31 to one and 32..63
            # to the other, and release them at once so the mask overlaps dkbg.
            d_attn_ub = g_d_attn_ub.next()
            pl.set_validshape(d_attn_acc, [C, C])
            pl.move(d_attn_ub, d_attn_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_D_ATTN_READY,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # dkbg = attn^T @ dw, one output sub-block per K tile. It is placed
            # after the drain so that it covers the masking latency.
            for k_block in pl.range(0, k_blocks):
                col_base = k_block * TILE
                attn_t_l0a = g_l0a.next()
                dw_l0b = g_l0b.next()
                pl.set_validshape(attn_t_l1, [C, C])
                pl.move(attn_t_l0a, attn_t_l1, offset=[0, 0])
                pl.set_validshape(attn_t_l0a, [C, C])
                # Full-width move: the columns past K hold stale L1 and are
                # never stored, matmul and fixpipe running on the whole tile.
                pl.set_validshape(dw_l1, [C, TILE])
                pl.move(dw_l0b, dw_l1, offset=[0, col_base])
                pl.set_validshape(dw_l0b, [C, C])
                out_acc = g_out_acc_a.next()
                pl.set_validshape(out_acc, [C, C])
                pl.matmul(out_acc, attn_t_l0a, dw_l0b)
                pl.set_validshape(out_acc, [C, pl.min(TILE, Kd - col_base)])
                pl.store(dkbg, out_acc, [b, h, n, 0, col_base])

            # Phase B. The masked dA comes back from GM, where the vector core
            # had to write it as an output in any case.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_MASKED_IN_GM,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.set_validshape(d_attn_masked_l1, [C, C])
            pl.load(d_attn_masked_l1, d_attn_masked, [b, h, n, 0, 0])

            # bracket = dA @ A^T, reusing the attn^T tile already in L1.
            d_attn_l0a = g_l0a.next()
            attn_t_l0b = g_l0b.next()
            pl.set_validshape(d_attn_masked_l1, [C, C])
            pl.move(d_attn_l0a, d_attn_masked_l1, offset=[0, 0])
            pl.set_validshape(d_attn_l0a, [C, C])
            pl.set_validshape(attn_t_l1, [C, C])
            pl.move(attn_t_l0b, attn_t_l1, offset=[0, 0])
            pl.set_validshape(attn_t_l0b, [C, C])
            pl.set_validshape(bracket_acc, [C, C])
            pl.matmul(bracket_acc, d_attn_l0a, attn_t_l0b)

            # Bridge the result back to L1 through UB; there is no path from an
            # accumulator to L1 on this part.
            pl.set_validshape(bracket_acc, [C, C])
            pl.set_validshape(bridge_ub, [C, C])
            pl.move(bridge_ub, bracket_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_BRIDGE_READY,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # dvb = attn^T @ dv2, placed after that signal so that it covers the
            # relay latency.
            for v_block in pl.range(0, v_blocks):
                col_base = v_block * TILE
                attn_t_l0a = g_l0a.next()
                dv2_l0b = g_l0b.next()
                pl.set_validshape(attn_t_l1, [C, C])
                pl.move(attn_t_l0a, attn_t_l1, offset=[0, 0])
                pl.set_validshape(attn_t_l0a, [C, C])
                pl.set_validshape(dv2_l1, [C, TILE])
                pl.move(dv2_l0b, dv2_l1, offset=[0, col_base])
                pl.set_validshape(dv2_l0b, [C, C])
                out_acc = g_out_acc_a.next()
                pl.set_validshape(out_acc, [C, C])
                pl.matmul(out_acc, attn_t_l0a, dv2_l0b)
                pl.set_validshape(out_acc, [C, pl.min(TILE, Vd - col_base)])
                pl.store(dvb, out_acc, [b, h, n, 0, col_base])

            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_BRIDGE_RELAYED,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # two_sided = A^T @ bracket
            attn_t_l0a = g_l0a.next()
            bridge_l0b = g_l0b.next()
            pl.set_validshape(attn_t_l1, [C, C])
            pl.move(attn_t_l0a, attn_t_l1, offset=[0, 0])
            pl.set_validshape(attn_t_l0a, [C, C])
            pl.set_validshape(bridge_l1, [C, C])
            pl.move(bridge_l0b, bridge_l1, offset=[0, 0])
            pl.set_validshape(bridge_l0b, [C, C])
            pl.set_validshape(two_sided_acc, [C, C])
            pl.matmul(two_sided_acc, attn_t_l0a, bridge_l0b)

            pl.set_validshape(two_sided_acc, [C, C])
            pl.store(d_attn_two_sided, two_sided_acc, [b, h, n, 0, 0])

            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_TWO_SIDED_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            two_sided_ub = g_two_sided_ub.next()
            pl.set_validshape(two_sided_acc, [C, C])
            pl.move(two_sided_ub, two_sided_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_TWO_SIDED_READY,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # Phase C. k_f32 has no vector dependency, so its load covers part
            # of the wait for d_a.
            pl.set_validshape(k_l1, [C, Kd])
            pl.load(k_l1, k_f32, [b, h, n, 0, 0])
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_D_A_IN_GM,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # d_a and k_beta are both required outputs, so phase C reads them
            # straight back from GM instead of keeping a scratch copy.
            pl.set_validshape(d_a_l1, [C, C])
            pl.load(d_a_l1, d_a, [b, h, n, 0, 0])
            pl.set_validshape(d_a_t_l1, [C, C])
            pl.load(d_a_t_l1, d_a, [b, h, n, 0, 0], order=[4, 3])

            # dkb = d_a @ k_f32
            for k_block in pl.range(0, k_blocks):
                col_base = k_block * TILE
                d_a_l0a = g_l0a.next()
                k_l0b = g_l0b.next()
                pl.set_validshape(d_a_l1, [C, C])
                pl.move(d_a_l0a, d_a_l1, offset=[0, 0])
                pl.set_validshape(d_a_l0a, [C, C])
                pl.set_validshape(k_l1, [C, TILE])
                pl.move(k_l0b, k_l1, offset=[0, col_base])
                pl.set_validshape(k_l0b, [C, C])
                out_acc = g_out_acc_c.next()
                pl.set_validshape(out_acc, [C, C])
                pl.matmul(out_acc, d_a_l0a, k_l0b)
                pl.set_validshape(out_acc, [C, pl.min(TILE, Kd - col_base)])
                pl.store(dkb, out_acc, [b, h, n, 0, col_base])

            # Only the k_beta half of the hand-off is awaited here: dkb above
            # ran while the vector core was still forming k_beta.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_K_BETA_IN_GM,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.set_validshape(k_beta_l1, [C, Kd])
            pl.load(k_beta_l1, k_beta, [b, h, n, 0, 0])

            # dkt = d_a^T @ k_beta, d_a^T reaching L0A by a transposing load
            for k_block in pl.range(0, k_blocks):
                col_base = k_block * TILE
                d_a_t_l0a = g_l0a.next()
                k_beta_l0b = g_l0b.next()
                pl.set_validshape(d_a_t_l1, [C, C])
                pl.move(d_a_t_l0a, d_a_t_l1, offset=[0, 0])
                pl.set_validshape(d_a_t_l0a, [C, C])
                pl.set_validshape(k_beta_l1, [C, TILE])
                pl.move(k_beta_l0b, k_beta_l1, offset=[0, col_base])
                pl.set_validshape(k_beta_l0b, [C, C])
                out_acc = g_out_acc_c.next()
                pl.set_validshape(out_acc, [C, C])
                pl.matmul(out_acc, d_a_t_l0a, k_beta_l0b)
                pl.set_validshape(out_acc, [C, pl.min(TILE, Kd - col_base)])
                pl.store(dkt, out_acc, [b, h, n, 0, col_base])

        # Drain the vector core's bootstrap credits so both counters end at zero.
        for _slot in pl.range(0, HANDOFF_DEPTH, 1):
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_D_ATTN_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        for _slot in pl.range(0, HANDOFF_DEPTH, 1):
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_TWO_SIDED_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

    with pl.section_vector():
        core_count = pl.get_block_num()
        core = pl.get_block_idx() // pl.get_subblock_num()
        subblock = pl.get_subblock_idx()

        B = dw.shape[0]
        H = dw.shape[1]
        N = dw.shape[2]
        Kd = dw.shape[4]
        chunks_per_batch = H * N
        task_count = B * chunks_per_batch
        tasks_per_core = (task_count + core_count - 1) // core_count
        first_task = tasks_per_core * core
        task_end = pl.min(first_task + tasks_per_core, task_count)

        # This sub-block's band of every drained matrix.
        row_base = subblock * CH

        bridge_ub = g_bridge_ub.current()
        bridge_l1 = g_bridge_l1.current()
        decay_ub = g_decay_ub.current()
        beta_ub = g_beta_ub.current()
        k_ub = g_k_ub.current()
        k_beta_ub = g_k_beta_ub.current()

        # All hand-off slots start free, so the cube may run ahead.
        for _slot in pl.range(0, HANDOFF_DEPTH, 1):
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=EV_D_ATTN_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        for _slot in pl.range(0, HANDOFF_DEPTH, 1):
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=EV_TWO_SIDED_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        for tid in pl.range(first_task, task_end, 1):
            d_attn_ub = g_d_attn_ub.next()
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_D_ATTN_READY,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            b = tid // chunks_per_batch
            within_batch = tid % chunks_per_batch
            h = within_batch // N
            n = within_batch % N

            # Mask this sub-block's own rows and store its own band. The bands
            # are disjoint, so nothing is written twice and no guard is needed.
            pl.set_validshape(d_attn_ub, [CH, C])
            mask_strict_lower_vf(d_attn_ub, CH, row_base)
            pl.set_validshape(d_attn_ub, [CH, C])
            pl.store(d_attn_masked, d_attn_ub, [b, h, n, row_base, 0])

            # Released on MTE3: the store is the last reader of the UB slot.
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_D_ATTN_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_MASKED_IN_GM,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # Relay the phase-B intermediate from UB into L1. The drain was to
            # sub-block 0 only, so the copy is guarded; the signal is not,
            # because a cross-core wait needs every lane to have signalled.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_BRIDGE_READY,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            if pl.get_subblock_idx() < 1:
                pl.set_validshape(bridge_ub, [C, C])
                pl.set_validshape(bridge_l1, [C, C])
                pl.move(bridge_l1, bridge_ub)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_BRIDGE_RELAYED,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # d_a = -(two_sided * gate), banded like the mask.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_TWO_SIDED_READY,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            two_sided_ub = g_two_sided_ub.next()
            pl.set_validshape(decay_ub, [1, C])
            pl.load(decay_ub, decay, [b, h, n, 0])
            pl.set_validshape(two_sided_ub, [CH, C])
            gate_and_negate_vf(two_sided_ub, decay_ub, CH, row_base)
            pl.set_validshape(two_sided_ub, [CH, C])
            pl.store(d_a, two_sided_ub, [b, h, n, row_base, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_D_A_IN_GM,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_TWO_SIDED_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # k_beta = k_f32 * beta_act, row-local and so banded the same way.
            pl.set_validshape(k_ub, [C, Kd])
            pl.load(k_ub, k_f32, [b, h, n, 0, 0])
            pl.set_validshape(beta_ub, [1, C])
            pl.load(beta_ub, beta_act, [b, h, n, 0])
            pl.set_validshape(k_beta_ub, [CH, Kd])
            form_k_beta_vf(k_ub, beta_ub, k_beta_ub, CH, row_base)
            pl.set_validshape(k_beta_ub, [CH, Kd])
            pl.store(k_beta, k_beta_ub, [b, h, n, row_base, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_K_BETA_IN_GM,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


def gdr_bwd_wy_matmuls_wrapper(attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2, *, sync=True):
    """Differentiate the WY factor in one launch.

    Returns (dkbg, dvb, dkb, dkt, d_a), all float32 in chunk layout. The three
    intermediate forms of dA and the on-chip k_beta are allocated here because
    they cross between the cores through GM, but they are not results.
    """
    B, H, N, _, K = dw.shape
    V = dv2.shape[4]
    device = dw.device

    dkbg = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    dvb = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    dkb = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    dkt = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    d_a = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)

    d_attn_raw = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    d_attn_masked = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    d_attn_two_sided = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    k_beta = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)

    max_blocks = get_platform_info().core_num or 1
    num_cores = max(1, min(B * H * N, max_blocks))

    gdr_bwd_wy_matmuls_kernel[None, num_cores](
        attn, k_f32, beta_act, decay, v_b, kbeta_t, dw, dv2,
        dkbg, dvb, d_attn_raw, d_attn_masked, d_attn_two_sided, d_a, k_beta, dkb, dkt)
    if sync:
        torch.npu.synchronize()

    return dkbg, dvb, dkb, dkt, d_a
