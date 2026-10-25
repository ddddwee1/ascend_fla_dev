# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Within-chunk part of the training forward: the WY transform and its residuals.

First of the two training-forward operators. It turns the public token-major
inputs into the chunk-major form the rest of the pipeline uses, builds the WY
factor that replaces the sequential delta rule inside a chunk, and materialises
every tensor the backward pass will need. The inter-chunk scan that follows
consumes seven of its outputs; the backward pass consumes thirteen.

Mathematics
-----------
Per chunk of C tokens, with scale = 1/sqrt(K)::

    beta_act = sigmoid(beta)          k_beta    = k * beta_act
    q_s      = q * scale              v_b       = v * beta_act
    decay    = cumulative sum of g over the chunk
    L[i, j]  = exp(decay_i - decay_j) for j <= i, else 0
    attn     = (I - strict_lower((k_beta @ k^T) * L))^-1
    a_mat    = lower_incl((q_s @ k^T) * L)
    decay_exp = exp(decay)            dfac = exp(decay_{C-1} - decay)
    q_de     = q_s * decay_exp        kbeta_de = k_beta * decay_exp
    u        = attn @ v_b             w    = attn @ kbeta_de

The inverse is exact, not iterative: `I - strict_lower(.)` is unit lower
triangular, so forward substitution over the C rows inverts it in one pass.

`q_t`, `kbeta_t` and `w_t` are transposed copies, produced here so that the
operators downstream can load a left matmul operand without transposing.

Interface
---------
``gdr_fwd_train_wy_wrapper(q, k, v, g, beta)``

    q, k      [B, T, H, K] bf16
    v         [B, T, H, V] bf16
    g         [B, T, H] f32     per-token log gate, non-positive
    beta      [B, T, H] bf16    sigmoid logits, activated in the kernel

    returns 17 float32 tensors in chunk layout, none sliced to T:
      q_s, k_f32, dfac, q_de        [B, H, N, C, K]
      v_b, u                        [B, H, N, C, V]
      w                             [B, H, N, C, K]
      attn, a_mat                   [B, H, N, C, C]
      q_t, kbeta_t, w_t             [B, H, N, K, C]
      g_f32, beta_act, sig          [B, H, N, C]
      decay, decay_exp              [B, H, N, C]

Tokens past T are padded with g = 0 and beta_act = 0, so every product they
enter is exactly zero; the trim back to T happens in the backward epilogue.

Layout and scheduling
---------------------
One launch of one mixed cube/vector kernel, four phases alternating between the
two cores, with one event pair per hand-off:

    preprocess (vector)  ->  scores (cube)  ->  wy_build (vector)  ->  apply (cube)

The hand-offs are flat and non-overlapping: each event pair is bootstrapped once
by the section that owns its release half and drained once at that section's
end, so every set has exactly one matching wait. Nesting them deadlocks a
single-buffered pipeline.

A work item is one (batch, head, chunk) triple, split into contiguous per-core
runs; both sections compute the same range, since
`pl.get_block_idx() // pl.get_subblock_num()` yields the same physical core
index on either side.

`wy_build` splits across the two vector sub-blocks into two independent chains:
one builds decay_exp, L and attn, which holds the sequential forward
substitution; the other builds a_mat, q_de, kbeta_de, kbeta_t and dfac, which
moves four times as much data. Both need decay, decay_exp and L in their own UB,
so both compute those few register operations and only the first stores the two
that are outputs. Every GM region the two write is disjoint, and the three tiles
they both build receive bit-identical values either way.

Two layout rules shape the code. A transposed operand destined for L0A needs an
L1 tile declared ZN and filled by a descending-order load, while an
untransposed one needs the default NZ, so attn is held twice when it is needed
in both orientations. And `pl.move` requires the source and destination
physical shapes to match, which is why the square [C, C] L0 views exist
alongside the wide ones, aliasing the same addresses and mutex ids.

The transpose writes down a column at the odd row pitch TP rather than reading
across one at the natural pitch: a stride that is an even multiple of the 32-byte
UB bank width puts every lane of a register in one bank, which is the worst case
for that memory. Only the single destination tile needs the padding.
"""

import logging

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

C = 64        # tokens per chunk, a frozen constant of the operator family
SUB = 64      # K and V sub-block width, and the apply phase's strip width
HMAX = 32     # head-axis width of the staging tile for the [B, T, H] loads
KMAX = 128    # supported K upper bound, and the L1/L0 operand column width
MMIN = 16     # minimum physical row count of a float32 accumulator or left tile
TP = 72       # row pitch of the transpose destination: 288 B is an odd multiple
              # of the 32 B UB bank width, which keeps the strided write
              # conflict-free. The valid shape stays [C, C].

# UB addresses. The second slot of each double-buffered group sits above the
# single-buffered region.
UB_Q = 0x00000            # [C, SUB] bf16
UB_Q_PONG = 0x20000
UB_K = 0x02000            # [C, SUB] bf16
UB_K_PONG = 0x22000
UB_BETA_ROWS = 0x04000    # [C, HMAX] bf16
UB_GATE_ROWS = 0x05000    # [C, HMAX] f32
UB_BETA_ACT = 0x07000     # [1, C] f32
UB_TOKEN_ROW = 0x07100    # [1, C] f32
UB_Q_SCALED = 0x07200     # [C, SUB] f32
UB_Q_SCALED_PONG = 0x24000
UB_K_FP32 = 0x0b200       # [C, SUB] f32
UB_K_FP32_PONG = 0x28000
UB_K_BETA = 0x0f200       # [C, SUB] f32
UB_K_BETA_PONG = 0x2c000
UB_TRIANGLE = 0x13200     # [C, C] f32
UB_ATTN = 0x17200         # [C, C] f32
UB_TRANSPOSE = 0x1b200    # [C, TP] f32
UB_DECAY = 0x1fa00        # [1, C] f32,  top 0x1fb00

# L1 and L0 addresses. The apply phase holds its own L1 and accumulator tiles:
# sharing the single-buffered ones across the cube to vector to cube boundary
# never earns a reuse credit.
L1_Q = 0x00000            # [C, KMAX]
L1_K_T = 0x08000          # [KMAX, C]
L1_K_BETA = 0x10000       # [C, KMAX]
L1_GATE = 0x18000         # [MMIN, C]
L1_TRIANGLE = 0x19000     # [C, C]
L1_ATTN = 0x1d000         # [C, C]
L1_OPERAND = 0x21000      # [C, C]
L1_ATTN_T = 0x25000       # [C, C] ZN
L1_OPERAND_PONG = 0x29000  # [C, C],  top 0x2d000
L0A_SLOTS = (0x0000, 0x8000)
L0B_SLOTS = (0x0000, 0x8000)
L0C_DECAY = 0x0000
L0C_KK_SCORES = 0x1000
L0C_QK_SCORES = 0x5000
L0C_APPLY = 0x9000
L0C_APPLY_PONG = 0xd000   # top 0x11000

# Cross-core event ids, three pairs, one per hand-off.
E_PREP_FREE = 0      # cube -> vector: the preprocess outputs have been read
E_PREP_READY = 1     # vector -> cube: they are in GM
E_SCORES_FREE = 3    # vector -> cube: the score tensors have been read
E_SCORES_READY = 2   # cube -> vector: they are in GM
E_WY_FREE = 4        # cube -> vector: the WY tensors have been read
E_WY_READY = 5       # vector -> cube: they are in GM

# Some PyPTO builds expose part of the DT_* enum as bare integers, which the
# vector-function layer rejects. Rebuild the affected ones at module scope: the
# parser inlines helper calls, so a decorated body must see plain names.
_DATA_TYPE = type(pl.DT_FP32)


def _as_data_type(value):
    return value if isinstance(value, _DATA_TYPE) else _DATA_TYPE(value)


_PL_FP32 = _as_data_type(pl.DT_FP32)
_PL_INT32 = _as_data_type(pl.DT_INT32)
_PL_UINT32 = _as_data_type(pl.DT_UINT32)


@pl.vector_function
def activate_gates_vf(beta_rows, gate_rows, beta_act, gate, token_count, head):
    """beta_act = sigmoid(beta[.., head]); gate = g[.., head], tail rows zeroed.

    The rows past the sequence tail are written as zero for both outputs, so
    every downstream product involving them is exactly zero.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    one = vf.full(1.0, all_lanes, dtype=_PL_FP32)
    for token in pl.range(0, token_count, 1):
        logit_bf16 = vf.load_align(beta_rows, token * HMAX + head,
                                   dist=pl.LoadDist.BRC_B16)
        logit = vf.astype(logit_bf16, all_lanes, dtype=_PL_FP32,
                          layout=pl.CastLayout.ZERO)
        denominator = vf.exp_sub(zero, logit, all_lanes)
        denominator = vf.adds(denominator, 1.0, all_lanes)
        activated = vf.div(one, denominator, all_lanes)
        vf.store_align(beta_act + token, activated, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)
        gate_value = vf.load_align(gate_rows, token * HMAX + head,
                                   dist=pl.LoadDist.BRC_B32)
        vf.store_align(gate + token, gate_value, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)
    for pad_token in pl.range(token_count, C, 1):
        vf.store_align(beta_act + pad_token, zero, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)
        vf.store_align(gate + pad_token, zero, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def scale_qk_tile_vf(q_bf16, k_bf16, beta_act, q_scaled, k_fp32, k_beta,
                     token_count, scale):
    """q_s = q * scale; k_f32 = float32(k); k_beta = k_f32 * beta_act."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for token in pl.range(0, token_count, 1):
        beta = vf.load_align(beta_act, token, dist=pl.LoadDist.BRC_B32)
        k_narrow = vf.load_align(k_bf16, token * SUB, dist=pl.LoadDist.UNPK_B16)
        k_row = vf.astype(k_narrow, all_lanes, dtype=_PL_FP32,
                          layout=pl.CastLayout.ZERO)
        vf.store_align(k_fp32 + token * SUB, k_row, all_lanes)
        k_beta_row = vf.mul(k_row, beta, all_lanes)
        vf.store_align(k_beta + token * SUB, k_beta_row, all_lanes)
        q_narrow = vf.load_align(q_bf16, token * SUB, dist=pl.LoadDist.UNPK_B16)
        q_row = vf.astype(q_narrow, all_lanes, dtype=_PL_FP32,
                          layout=pl.CastLayout.ZERO)
        q_row = vf.muls(q_row, scale, all_lanes)
        vf.store_align(q_scaled + token * SUB, q_row, all_lanes)
    for pad_token in pl.range(token_count, C, 1):
        vf.store_align(q_scaled + pad_token * SUB, zero, all_lanes)
        vf.store_align(k_fp32 + pad_token * SUB, zero, all_lanes)
        vf.store_align(k_beta + pad_token * SUB, zero, all_lanes)


@pl.vector_function
def transpose_tile_vf(source, destination):
    """destination[j, i] = source[i, j], source [C, SUB] and destination [C, TP].

    One source row is read contiguously and scattered down one destination
    column at the odd pitch TP, which keeps the strided side of the transpose
    off a single UB bank. The cross-lane index on a float32 tile must be
    unsigned, and TP = 72 is built as (j << 6) + (j << 3).
    """
    index_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_UINT32)
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lane = vf.arange(0, dtype=_PL_UINT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    lane_times_64 = vf.shift_left(lane, 6, index_lanes)
    lane_times_8 = vf.shift_left(lane, 3, index_lanes)
    column_offset = vf.add(lane_times_64, lane_times_8, index_lanes)
    for row in pl.range(0, C, 1):
        source_row = vf.load_align(source, row * SUB, dtype=_PL_FP32)
        vf.scatter(destination + row, source_row, column_offset, all_lanes)


@pl.vector_function
def scale_v_tile_vf(v_bf16, beta_act, v_beta, token_count):
    """v_b = float32(v) * beta_act, tail rows zeroed."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for token in pl.range(0, token_count, 1):
        beta = vf.load_align(beta_act, token, dist=pl.LoadDist.BRC_B32)
        v_narrow = vf.load_align(v_bf16, token * SUB, dist=pl.LoadDist.UNPK_B16)
        v_row = vf.astype(v_narrow, all_lanes, dtype=_PL_FP32,
                          layout=pl.CastLayout.ZERO)
        v_row = vf.mul(v_row, beta, all_lanes)
        vf.store_align(v_beta + token * SUB, v_row, all_lanes)
    for pad_token in pl.range(token_count, C, 1):
        vf.store_align(v_beta + pad_token * SUB, zero, all_lanes)


@pl.vector_function
def build_lower_ones_vf(triangle):
    """triangle[r, j] = 1 for j <= r, else 0, at row pitch C.

    The scores phase loads this transposed to get the upper-triangular operand
    that turns one gate row into its running cumulative sum.
    """
    index_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    column = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    for row in pl.range(0, C, 1):
        ones = vf.full(1.0, all_lanes, dtype=_PL_FP32)
        keep = vf.lt(column, row + 1, index_lanes)
        ones = vf.select(ones, zero, keep)
        vf.store_align(triangle + row * C, ones, all_lanes)


@pl.vector_function
def decay_exp_vf(decay, decay_exp):
    """decay_exp[j] = exp(decay[j]), one [1, C] row in a single register."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    decay_row = vf.load_align(decay, 0, dtype=_PL_FP32)
    exponentiated = vf.exp_sub(decay_row, zero, all_lanes)
    vf.store_align(decay_exp + 0, exponentiated, all_lanes)


@pl.vector_function
def build_decay_mask_vf(decay, mask):
    """mask[i, j] = exp(decay_i - decay_j) for j <= i, else 0."""
    index_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    column = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    decay_row = vf.load_align(decay, 0, dtype=_PL_FP32)
    for row in pl.range(0, C, 1):
        decay_i = vf.load_align(decay, row, dist=pl.LoadDist.BRC_B32)
        mask_row = vf.exp_sub(decay_i, decay_row, all_lanes)
        keep = vf.lt(column, row + 1, index_lanes)
        mask_row = vf.select(mask_row, zero, keep)
        vf.store_align(mask + row * C, mask_row, all_lanes)


@pl.vector_function
def negate_masked_scores_vf(kk_scores, mask, system):
    """system = -(kk_scores * mask) on the strict lower triangle, j < i."""
    index_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    column = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    for row in pl.range(0, C, 1):
        scores_row = vf.load_align(kk_scores, row * C, dtype=_PL_FP32)
        mask_row = vf.load_align(mask, row * C, dtype=_PL_FP32)
        value = vf.mul(scores_row, mask_row, all_lanes)
        value = vf.neg(value, all_lanes)
        keep = vf.lt(column, row, index_lanes)
        value = vf.select(value, zero, keep)
        vf.store_align(system + row * C, value, all_lanes)


@pl.vector_function
def forward_substitute_vf(system):
    """Invert a unit lower-triangular system in place, then set its diagonal.

    Row i is updated from rows 0..i-1, so the recurrence over i is inherently
    sequential and cannot be split across vector sub-blocks. Columns at or past
    i stay zero because every row below i is zero there, so accumulating over
    the full register width equals restricting the sum to the first i columns.

    Rows are produced in pairs: each row still accumulates its contributions in
    the same order into its own accumulator, so no sum is reassociated, but the
    shared row is loaded once for both, the two accumulators break the
    multiply-add dependency chain two ways, and the second row of a pair takes
    the first row's contribution from a register rather than re-reading it. With
    C - 1 rows to update the pairs cover rows 1..C-2 and the last row is
    finished on its own.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    one = vf.full(1.0, all_lanes, dtype=_PL_FP32)
    for row in pl.range(1, C - 1, 2):
        acc_first = vf.full(0.0, all_lanes, dtype=_PL_FP32)
        acc_second = vf.full(0.0, all_lanes, dtype=_PL_FP32)
        for source in pl.range(0, row, 1):
            source_row = vf.load_align(system, source * C, dtype=_PL_FP32)
            weight_first = vf.load_align(system, row * C + source,
                                         dist=pl.LoadDist.BRC_B32)
            weight_second = vf.load_align(system, (row + 1) * C + source,
                                          dist=pl.LoadDist.BRC_B32)
            acc_first = vf.mul_add_dst(source_row, weight_first, all_lanes)
            acc_second = vf.mul_add_dst(source_row, weight_second, all_lanes)
        # Every read of this pair happens before either store.
        old_first = vf.load_align(system, row * C, dtype=_PL_FP32)
        old_second = vf.load_align(system, (row + 1) * C, dtype=_PL_FP32)
        coupling = vf.load_align(system, (row + 1) * C + row, dist=pl.LoadDist.BRC_B32)
        new_first = vf.add(old_first, acc_first, all_lanes)
        vf.store_align(system + row * C, new_first, all_lanes)
        acc_second = vf.mul_add_dst(new_first, coupling, all_lanes)
        new_second = vf.add(old_second, acc_second, all_lanes)
        vf.store_align(system + (row + 1) * C, new_second, all_lanes)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    acc_last = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for source in pl.range(0, C - 1, 1):
        source_row = vf.load_align(system, source * C, dtype=_PL_FP32)
        weight_last = vf.load_align(system, (C - 1) * C + source,
                                    dist=pl.LoadDist.BRC_B32)
        acc_last = vf.mul_add_dst(source_row, weight_last, all_lanes)
    old_last = vf.load_align(system, (C - 1) * C, dtype=_PL_FP32)
    new_last = vf.add(old_last, acc_last, all_lanes)
    vf.store_align(system + (C - 1) * C, new_last, all_lanes)
    vf.mem_bar(mode=pl.MemBarMode.VST_VST)
    for row in pl.range(0, C, 1):
        vf.store_align(system + row * C + row, one, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def mask_attn_weights_vf(qk_scores, mask, a_mat):
    """a_mat = qk_scores * mask on the lower triangle including the diagonal.

    `qk_scores` and `a_mat` are the same tile at pitch TP; `mask` is the [C, C]
    decay mask and keeps pitch C.
    """
    index_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    column = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    for row in pl.range(0, C, 1):
        scores_row = vf.load_align(qk_scores, row * TP, dtype=_PL_FP32)
        mask_row = vf.load_align(mask, row * C, dtype=_PL_FP32)
        value = vf.mul(scores_row, mask_row, all_lanes)
        keep = vf.lt(column, row + 1, index_lanes)
        value = vf.select(value, zero, keep)
        vf.store_align(a_mat + row * TP, value, all_lanes)


@pl.vector_function
def scale_rows_vf(source, row_scale, destination):
    """destination[i, :] = source[i, :] * row_scale[i]."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    for row in pl.range(0, C, 1):
        factor = vf.load_align(row_scale, row, dist=pl.LoadDist.BRC_B32)
        value = vf.load_align(source, row * SUB, dtype=_PL_FP32)
        value = vf.mul(value, factor, all_lanes)
        vf.store_align(destination + row * SUB, value, all_lanes)


@pl.vector_function
def decay_factor_vf(decay, dfac):
    """dfac[i, :] = exp(decay[C-1] - decay[i]), broadcast along the free axis."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    chunk_decay = vf.load_align(decay, C - 1, dist=pl.LoadDist.BRC_B32)
    for row in pl.range(0, C, 1):
        decay_i = vf.load_align(decay, row, dist=pl.LoadDist.BRC_B32)
        factor = vf.exp_sub(chunk_decay, decay_i, all_lanes)
        vf.store_align(dfac + row * SUB, factor, all_lanes)


_DYN5 = [pl.DYNAMIC] * 5
_DYN4 = [pl.DYNAMIC] * 4
_DYN3 = [pl.DYNAMIC] * 3
_DYN2 = [pl.DYNAMIC] * 2


@pl.jit(auto_mutex=True)
def gdr_fwd_train_wy_kernel(
    q: pl.Tensor[_DYN4, pl.DT_BF16],
    k: pl.Tensor[_DYN4, pl.DT_BF16],
    v: pl.Tensor[_DYN4, pl.DT_BF16],
    g: pl.Tensor[_DYN3, pl.DT_FP32],
    beta: pl.Tensor[_DYN3, pl.DT_BF16],
    lower_ones_gm: pl.Tensor[_DYN2, pl.DT_FP32],
    v_b_padded_gm: pl.Tensor[_DYN5, pl.DT_FP32],
    q_s_out: pl.Tensor[_DYN5, pl.DT_FP32],
    k_f32_out: pl.Tensor[_DYN5, pl.DT_FP32],
    v_b_out: pl.Tensor[_DYN5, pl.DT_FP32],
    g_f32_out: pl.Tensor[_DYN4, pl.DT_FP32],
    beta_act_out: pl.Tensor[_DYN4, pl.DT_FP32],
    sig_out: pl.Tensor[_DYN4, pl.DT_FP32],
    q_t_out: pl.Tensor[_DYN5, pl.DT_FP32],
    k_beta_out: pl.Tensor[_DYN5, pl.DT_FP32],
    decay_out: pl.Tensor[_DYN4, pl.DT_FP32],
    kk_scores_gm: pl.Tensor[_DYN5, pl.DT_FP32],
    qk_scores_gm: pl.Tensor[_DYN5, pl.DT_FP32],
    decay_exp_out: pl.Tensor[_DYN4, pl.DT_FP32],
    q_de_out: pl.Tensor[_DYN5, pl.DT_FP32],
    dfac_out: pl.Tensor[_DYN5, pl.DT_FP32],
    attn_out: pl.Tensor[_DYN5, pl.DT_FP32],
    a_mat_out: pl.Tensor[_DYN5, pl.DT_FP32],
    kbeta_t_out: pl.Tensor[_DYN5, pl.DT_FP32],
    kbeta_de_out: pl.Tensor[_DYN5, pl.DT_FP32],
    u_out: pl.Tensor[_DYN5, pl.DT_FP32],
    w_out: pl.Tensor[_DYN5, pl.DT_FP32],
    w_t_out: pl.Tensor[_DYN5, pl.DT_FP32],
    B: pl.DT_INT32,
    T: pl.DT_INT32,
    H: pl.DT_INT32,
    Kd: pl.DT_INT32,
    Vd: pl.DT_INT32,
    N: pl.DT_INT32,
    scale: pl.DT_FP32,
):
    # Preprocess UB tiles. The five that carry a K or V sub-block are double
    # buffered so that the next sub-block's load overlaps this one's work.
    g_q_ub = pl.make_tile_group(type=pl.TileType(shape=[C, SUB], dtype=pl.DT_BF16,
                                                 target_memory=pl.MemorySpace.Vec,
                                                 valid_shape=[-1, -1]),
                                addrs=[UB_Q, UB_Q_PONG], mutex_ids=[4, 16])
    g_k_ub = pl.make_tile_group(type=pl.TileType(shape=[C, SUB], dtype=pl.DT_BF16,
                                                 target_memory=pl.MemorySpace.Vec,
                                                 valid_shape=[-1, -1]),
                                addrs=[UB_K, UB_K_PONG], mutex_ids=[17, 18])
    g_beta_rows_ub = pl.make_tile_group(type=pl.TileType(shape=[C, HMAX], dtype=pl.DT_BF16,
                                                         target_memory=pl.MemorySpace.Vec,
                                                         valid_shape=[-1, -1]),
                                        addrs=UB_BETA_ROWS, mutex_ids=[5])
    g_gate_rows_ub = pl.make_tile_group(type=pl.TileType(shape=[C, HMAX], dtype=pl.DT_FP32,
                                                         target_memory=pl.MemorySpace.Vec,
                                                         valid_shape=[-1, -1]),
                                        addrs=UB_GATE_ROWS, mutex_ids=[6])
    g_beta_act_ub = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                                                        target_memory=pl.MemorySpace.Vec,
                                                        valid_shape=[-1, -1]),
                                       addrs=UB_BETA_ACT, mutex_ids=[7])
    # One [1, C] row, holding the per-token gate in the preprocess phase and
    # decay_exp in the WY phase; the two aliases make each use explicit.
    g_gate_ub = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                                                    target_memory=pl.MemorySpace.Vec,
                                                    valid_shape=[-1, -1]),
                                   addrs=UB_TOKEN_ROW, mutex_ids=[7])
    g_decay_exp_ub = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                                                         target_memory=pl.MemorySpace.Vec,
                                                         valid_shape=[-1, -1]),
                                        addrs=UB_TOKEN_ROW, mutex_ids=[7])
    g_q_scaled_ub = pl.make_tile_group(type=pl.TileType(shape=[C, SUB], dtype=pl.DT_FP32,
                                                        target_memory=pl.MemorySpace.Vec,
                                                        valid_shape=[-1, -1]),
                                       addrs=[UB_Q_SCALED, UB_Q_SCALED_PONG],
                                       mutex_ids=[8, 19])
    g_k_fp32_ub = pl.make_tile_group(type=pl.TileType(shape=[C, SUB], dtype=pl.DT_FP32,
                                                      target_memory=pl.MemorySpace.Vec,
                                                      valid_shape=[-1, -1]),
                                     addrs=[UB_K_FP32, UB_K_FP32_PONG],
                                     mutex_ids=[9, 20])
    g_k_beta_ub = pl.make_tile_group(type=pl.TileType(shape=[C, SUB], dtype=pl.DT_FP32,
                                                      target_memory=pl.MemorySpace.Vec,
                                                      valid_shape=[-1, -1]),
                                     addrs=[UB_K_BETA, UB_K_BETA_PONG],
                                     mutex_ids=[21, 22])
    # One [C, C] tile, holding the lower-triangular ones matrix in the
    # preprocess phase and the decay mask in the WY phase. Hoisting the ones
    # matrix out of the item loop is what makes that safe: it becomes a
    # write-once artefact that the cube reads from GM, never from UB.
    g_lower_ones_ub = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                          target_memory=pl.MemorySpace.Vec,
                                                          valid_shape=[-1, -1]),
                                         addrs=UB_TRIANGLE, mutex_ids=[10])
    g_decay_mask_ub = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                          target_memory=pl.MemorySpace.Vec,
                                                          valid_shape=[-1, -1]),
                                         addrs=UB_TRIANGLE, mutex_ids=[10])
    # The transpose destination, also the place where the query-key scores
    # become a_mat. Padded to pitch TP; the valid shape stays [C, C], so the
    # padding columns are never loaded or stored.
    g_transpose_ub = pl.make_tile_group(type=pl.TileType(shape=[C, TP], dtype=pl.DT_FP32,
                                                         target_memory=pl.MemorySpace.Vec,
                                                         valid_shape=[-1, -1]),
                                        addrs=UB_TRANSPOSE, mutex_ids=[8])
    # Holds the key-key scores on arrival and attn after the WY solve.
    g_attn_ub = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                    target_memory=pl.MemorySpace.Vec,
                                                    valid_shape=[-1, -1]),
                                   addrs=UB_ATTN, mutex_ids=[23])
    g_decay_ub = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                                                     target_memory=pl.MemorySpace.Vec,
                                                     valid_shape=[-1, -1]),
                                    addrs=UB_DECAY, mutex_ids=[24])

    g_q_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                                                 target_memory=pl.MemorySpace.Mat,
                                                 valid_shape=[-1, -1]),
                                addrs=L1_Q, mutex_ids=[11])
    g_k_t_l1 = pl.make_tile_group(type=pl.TileType(shape=[KMAX, C], dtype=pl.DT_FP32,
                                                   target_memory=pl.MemorySpace.Mat,
                                                   valid_shape=[-1, -1]),
                                  addrs=L1_K_T, mutex_ids=[11])
    g_k_beta_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                                                      target_memory=pl.MemorySpace.Mat,
                                                      valid_shape=[-1, -1]),
                                     addrs=L1_K_BETA, mutex_ids=[11])
    g_gate_l1 = pl.make_tile_group(type=pl.TileType(shape=[MMIN, C], dtype=pl.DT_FP32,
                                                    target_memory=pl.MemorySpace.Mat,
                                                    valid_shape=[-1, -1]),
                                   addrs=L1_GATE, mutex_ids=[12])
    g_lower_ones_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                          target_memory=pl.MemorySpace.Mat,
                                                          valid_shape=[-1, -1]),
                                         addrs=L1_TRIANGLE, mutex_ids=[13])

    # L0A and L0B each carry three views of the same two slots, with the same
    # mutex ids, so that an operand can be moved in at its own physical shape.
    g_l0a_gate_row = pl.make_tile_group(type=pl.TileType(shape=[MMIN, C], dtype=pl.DT_FP32,
                                                         target_memory=pl.MemorySpace.Left,
                                                         valid_shape=[-1, -1]),
                                        addrs=list(L0A_SLOTS), mutex_ids=[0, 1])
    g_l0a_wide = pl.make_tile_group(type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                                                     target_memory=pl.MemorySpace.Left,
                                                     valid_shape=[-1, -1]),
                                    addrs=list(L0A_SLOTS), mutex_ids=[0, 1])
    g_l0a_square = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                       target_memory=pl.MemorySpace.Left,
                                                       valid_shape=[-1, -1]),
                                      addrs=list(L0A_SLOTS), mutex_ids=[0, 1])
    g_l0b_lower_ones = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                           target_memory=pl.MemorySpace.Right,
                                                           valid_shape=[-1, -1]),
                                          addrs=list(L0B_SLOTS), mutex_ids=[2, 3])
    g_l0b_tall = pl.make_tile_group(type=pl.TileType(shape=[KMAX, C], dtype=pl.DT_FP32,
                                                     target_memory=pl.MemorySpace.Right,
                                                     valid_shape=[-1, -1]),
                                    addrs=list(L0B_SLOTS), mutex_ids=[2, 3])
    g_l0b_square = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                       target_memory=pl.MemorySpace.Right,
                                                       valid_shape=[-1, -1]),
                                      addrs=list(L0B_SLOTS), mutex_ids=[2, 3])

    g_decay_acc = pl.make_tile_group(type=pl.TileType(shape=[MMIN, C], dtype=pl.DT_FP32,
                                                      target_memory=pl.MemorySpace.Acc,
                                                      valid_shape=[-1, -1]),
                                     addrs=L0C_DECAY, mutex_ids=[14])
    # The two score accumulators hold separate mutex ids: sharing one would make
    # the drain of the first wait for the second's multiply to complete, even
    # though they touch disjoint addresses.
    g_kk_scores_acc = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                          target_memory=pl.MemorySpace.Acc,
                                                          valid_shape=[-1, -1]),
                                         addrs=L0C_KK_SCORES, mutex_ids=[15])
    g_qk_scores_acc = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                          target_memory=pl.MemorySpace.Acc,
                                                          valid_shape=[-1, -1]),
                                         addrs=L0C_QK_SCORES, mutex_ids=[31])

    # The apply phase: attn in both orientations, plus a rotating right operand
    # and accumulator so that one strip drains while the next multiplies.
    g_attn_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                    target_memory=pl.MemorySpace.Mat,
                                                    valid_shape=[-1, -1]),
                                   addrs=L1_ATTN, mutex_ids=[25])
    g_attn_t_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                      target_memory=pl.MemorySpace.Mat,
                                                      valid_shape=[-1, -1],
                                                      layout=pl.ZN),
                                     addrs=L1_ATTN_T, mutex_ids=[28])
    g_operand_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                       target_memory=pl.MemorySpace.Mat,
                                                       valid_shape=[-1, -1]),
                                      addrs=[L1_OPERAND, L1_OPERAND_PONG],
                                      mutex_ids=[26, 29])
    g_apply_acc = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                      target_memory=pl.MemorySpace.Acc,
                                                      valid_shape=[-1, -1]),
                                     addrs=[L0C_APPLY, L0C_APPLY_PONG],
                                     mutex_ids=[27, 30])

    with pl.section_cube():
        core_count = pl.get_block_num()
        core = pl.get_block_idx() // pl.get_subblock_num()
        task_count = B * H * N
        tasks_per_core = (task_count + core_count - 1) // core_count
        first_task = tasks_per_core * core
        task_end = pl.min(first_task + tasks_per_core, task_count)

        q_l1 = g_q_l1.next()
        k_t_l1 = g_k_t_l1.next()
        k_beta_l1 = g_k_beta_l1.next()
        gate_l1 = g_gate_l1.next()
        lower_ones_l1 = g_lower_ones_l1.next()
        decay_acc = g_decay_acc.next()
        kk_scores_acc = g_kk_scores_acc.next()
        qk_scores_acc = g_qk_scores_acc.next()
        attn_l1 = g_attn_l1.next()
        attn_t_l1 = g_attn_t_l1.next()
        v_tiles = (Vd + SUB - 1) // SUB
        k_tiles = (Kd + SUB - 1) // SUB

        # Let the vector core write the first item's preprocess outputs, and the
        # first item's WY outputs.
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=E_PREP_FREE,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=E_WY_FREE,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        for tid in pl.range(first_task, task_end, 1):
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=E_PREP_READY,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            b = tid // (H * N)
            within_batch = tid % (H * N)
            h = within_batch // N
            n = within_batch % N

            # decay = gate_row @ lower_ones^T, a cumulative sum as one matmul
            pl.set_validshape(gate_l1, [1, C])
            pl.load(gate_l1, g_f32_out, [b, h, n, 0])
            pl.set_validshape(lower_ones_l1, [C, C])
            pl.load(lower_ones_l1, lower_ones_gm, [0, 0], order=[1, 0])
            gate_l0a = g_l0a_gate_row.next()
            lower_ones_l0b = g_l0b_lower_ones.next()
            pl.move(gate_l0a, gate_l1, offset=[0, 0])
            pl.move(lower_ones_l0b, lower_ones_l1, offset=[0, 0])
            pl.set_validshape(gate_l0a, [MMIN, C])
            pl.set_validshape(lower_ones_l0b, [C, C])
            pl.set_validshape(decay_acc, [1, C])
            pl.matmul(decay_acc, gate_l0a, lower_ones_l0b)

            # kk_scores = k_beta @ k^T
            pl.set_validshape(k_beta_l1, [C, Kd])
            pl.load(k_beta_l1, k_beta_out, [b, h, n, 0, 0])
            pl.set_validshape(k_t_l1, [Kd, C])
            pl.load(k_t_l1, k_f32_out, [b, h, n, 0, 0], order=[4, 3])
            k_beta_l0a = g_l0a_wide.next()
            k_t_l0b = g_l0b_tall.next()
            pl.move(k_beta_l0a, k_beta_l1, offset=[0, 0])
            pl.move(k_t_l0b, k_t_l1, offset=[0, 0])
            pl.set_validshape(k_beta_l0a, [C, Kd])
            pl.set_validshape(k_t_l0b, [Kd, C])
            pl.set_validshape(kk_scores_acc, [C, C])
            pl.matmul(kk_scores_acc, k_beta_l0a, k_t_l0b)

            # qk_scores = q_s @ k^T, reusing the transposed key tile
            pl.set_validshape(q_l1, [C, Kd])
            pl.load(q_l1, q_s_out, [b, h, n, 0, 0])
            q_l0a = g_l0a_wide.next()
            k_t_l0b = g_l0b_tall.next()
            pl.move(q_l0a, q_l1, offset=[0, 0])
            pl.move(k_t_l0b, k_t_l1, offset=[0, 0])
            pl.set_validshape(q_l0a, [C, Kd])
            pl.set_validshape(k_t_l0b, [Kd, C])
            pl.set_validshape(qk_scores_acc, [C, C])
            pl.matmul(qk_scores_acc, q_l0a, k_t_l0b)

            # Every GM operand of this item is consumed.
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=E_PREP_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=E_SCORES_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.set_validshape(decay_acc, [1, C])
            pl.store(decay_out, decay_acc, [b, h, n, 0])
            pl.set_validshape(kk_scores_acc, [C, C])
            pl.store(kk_scores_gm, kk_scores_acc, [b, h, n, 0, 0])
            pl.set_validshape(qk_scores_acc, [C, C])
            pl.store(qk_scores_gm, qk_scores_acc, [b, h, n, 0, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=E_SCORES_READY,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # Apply phase: u, w and w_t, once the WY tensors are in GM.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=E_WY_READY,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # attn is invariant across all three loops below, so it is loaded
            # once per item in both orientations; nothing in the loops writes
            # either tile.
            pl.set_validshape(attn_l1, [C, C])
            pl.load(attn_l1, attn_out, [b, h, n, 0, 0])
            pl.set_validshape(attn_t_l1, [C, C])
            pl.load(attn_t_l1, attn_out, [b, h, n, 0, 0], order=[4, 3])

            # u = attn @ v_b, over V strips
            for v_tile in pl.range(0, v_tiles, 1):
                operand_l1 = g_operand_l1.next()
                apply_acc = g_apply_acc.next()
                pl.set_validshape(operand_l1, [C, C])
                # The padded copy exists only for a V that is not a multiple of
                # the strip width; otherwise the output itself has that pitch.
                if Vd % SUB == 0:
                    pl.load(operand_l1, v_b_out, [b, h, n, 0, v_tile * SUB])
                if Vd % SUB != 0:
                    pl.load(operand_l1, v_b_padded_gm, [b, h, n, 0, v_tile * SUB])
                attn_l0a = g_l0a_square.next()
                v_b_l0b = g_l0b_square.next()
                pl.move(attn_l0a, attn_l1, offset=[0, 0])
                pl.move(v_b_l0b, operand_l1, offset=[0, 0])
                pl.set_validshape(attn_l0a, [C, C])
                pl.set_validshape(v_b_l0b, [C, C])
                pl.set_validshape(apply_acc, [C, C])
                pl.matmul(apply_acc, attn_l0a, v_b_l0b)
                pl.set_validshape(apply_acc, [C, pl.min(Vd - v_tile * SUB, SUB)])
                pl.store(u_out, apply_acc, [b, h, n, 0, v_tile * SUB])

            # w = attn @ kbeta_de, over K strips
            for k_tile in pl.range(0, k_tiles, 1):
                operand_l1 = g_operand_l1.next()
                apply_acc = g_apply_acc.next()
                pl.set_validshape(operand_l1, [C, C])
                pl.load(operand_l1, kbeta_de_out, [b, h, n, 0, k_tile * SUB])
                attn_l0a = g_l0a_square.next()
                kbeta_de_l0b = g_l0b_square.next()
                pl.move(attn_l0a, attn_l1, offset=[0, 0])
                pl.move(kbeta_de_l0b, operand_l1, offset=[0, 0])
                pl.set_validshape(attn_l0a, [C, C])
                pl.set_validshape(kbeta_de_l0b, [C, C])
                pl.set_validshape(apply_acc, [C, C])
                pl.matmul(apply_acc, attn_l0a, kbeta_de_l0b)
                pl.set_validshape(apply_acc, [C, C])
                pl.store(w_out, apply_acc, [b, h, n, 0, k_tile * SUB])

            # w_t = kbeta_t @ attn^T, so that no transpose of w is needed
            for k_tile in pl.range(0, k_tiles, 1):
                operand_l1 = g_operand_l1.next()
                apply_acc = g_apply_acc.next()
                pl.set_validshape(operand_l1, [C, C])
                pl.load(operand_l1, kbeta_t_out, [b, h, n, k_tile * SUB, 0])
                kbeta_t_l0a = g_l0a_square.next()
                attn_t_l0b = g_l0b_square.next()
                pl.move(kbeta_t_l0a, operand_l1, offset=[0, 0])
                pl.move(attn_t_l0b, attn_t_l1, offset=[0, 0])
                pl.set_validshape(kbeta_t_l0a, [C, C])
                pl.set_validshape(attn_t_l0b, [C, C])
                pl.set_validshape(apply_acc, [C, C])
                pl.matmul(apply_acc, kbeta_t_l0a, attn_t_l0b)
                pl.set_validshape(apply_acc, [C, C])
                pl.store(w_t_out, apply_acc, [b, h, n, k_tile * SUB, 0])

            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=E_WY_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=E_SCORES_FREE,
                                  sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

    with pl.section_vector():
        core_count = pl.get_block_num()
        core = pl.get_block_idx() // pl.get_subblock_num()
        task_count = B * H * N
        tasks_per_core = (task_count + core_count - 1) // core_count
        first_task = tasks_per_core * core
        task_end = pl.min(first_task + tasks_per_core, task_count)
        k_tiles = (Kd + SUB - 1) // SUB
        v_tiles = (Vd + SUB - 1) // SUB

        beta_rows_ub = g_beta_rows_ub.next()
        gate_rows_ub = g_gate_rows_ub.next()
        beta_act_ub = g_beta_act_ub.next()
        gate_ub = g_gate_ub.next()
        decay_exp_ub = g_decay_exp_ub.next()
        lower_ones_ub = g_lower_ones_ub.next()
        decay_mask_ub = g_decay_mask_ub.next()
        transpose_ub = g_transpose_ub.next()
        attn_ub = g_attn_ub.next()
        decay_ub = g_decay_ub.next()

        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=E_SCORES_FREE,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        # The lower-triangular ones matrix is the same for every chunk, so it is
        # built once per core, before the first hand-off the cube waits on.
        # Every core writes the same bytes to the same address.
        if pl.get_subblock_idx() == 0:
            build_lower_ones_vf(lower_ones_ub)
            pl.set_validshape(lower_ones_ub, [C, C])
            pl.store(lower_ones_gm, lower_ones_ub, [0, 0])

        for tid in pl.range(first_task, task_end, 1):
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=E_PREP_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            b = tid // (H * N)
            within_batch = tid % (H * N)
            h = within_batch // N
            n = within_batch % N
            token_base = n * C
            token_count = pl.min(T - token_base, C)

            # The preprocess phase runs on sub-block 0; the hand-off events stay
            # unguarded so that both sub-blocks of the block signal.
            if pl.get_subblock_idx() == 0:
                pl.set_validshape(beta_rows_ub, [token_count, H])
                pl.load(beta_rows_ub, beta, [b, token_base, 0])
                pl.set_validshape(gate_rows_ub, [token_count, H])
                pl.load(gate_rows_ub, g, [b, token_base, 0])
                activate_gates_vf(beta_rows_ub, gate_rows_ub, beta_act_ub, gate_ub,
                                  token_count, h)

                pl.set_validshape(gate_ub, [1, C])
                pl.store(g_f32_out, gate_ub, [b, h, n, 0])
                pl.set_validshape(beta_act_ub, [1, C])
                pl.store(beta_act_out, beta_act_ub, [b, h, n, 0])
                pl.set_validshape(beta_act_ub, [1, C])
                pl.store(sig_out, beta_act_ub, [b, h, n, 0])

                for k_tile in pl.range(0, k_tiles, 1):
                    q_ub = g_q_ub.next()
                    k_ub = g_k_ub.next()
                    q_scaled_ub = g_q_scaled_ub.next()
                    k_fp32_ub = g_k_fp32_ub.next()
                    k_beta_ub = g_k_beta_ub.next()
                    # q, k and v are public [B, T, H, *] inputs read with 2-D
                    # tiles: order= makes them walk (T, K) instead of the
                    # default trailing (H, K), which absorbs the layout change
                    # into the load instead of a host transpose.
                    pl.set_validshape(q_ub, [token_count, SUB])
                    pl.load(q_ub, q, [b, token_base, h, k_tile * SUB], order=[1, 3])
                    pl.set_validshape(k_ub, [token_count, SUB])
                    pl.load(k_ub, k, [b, token_base, h, k_tile * SUB], order=[1, 3])
                    scale_qk_tile_vf(q_ub, k_ub, beta_act_ub,
                                     q_scaled_ub, k_fp32_ub, k_beta_ub,
                                     token_count, scale)
                    pl.set_validshape(q_scaled_ub, [C, SUB])
                    pl.store(q_s_out, q_scaled_ub, [b, h, n, 0, k_tile * SUB])
                    pl.set_validshape(k_fp32_ub, [C, SUB])
                    pl.store(k_f32_out, k_fp32_ub, [b, h, n, 0, k_tile * SUB])
                    pl.set_validshape(k_beta_ub, [C, SUB])
                    pl.store(k_beta_out, k_beta_ub, [b, h, n, 0, k_tile * SUB])
                    transpose_tile_vf(q_scaled_ub, transpose_ub)
                    pl.set_validshape(transpose_ub, [C, C])
                    pl.store(q_t_out, transpose_ub, [b, h, n, k_tile * SUB, 0])

                for v_tile in pl.range(0, v_tiles, 1):
                    v_ub = g_q_ub.next()
                    v_beta_ub = g_q_scaled_ub.next()
                    v_count = pl.min(Vd - v_tile * SUB, SUB)
                    pl.set_validshape(v_ub, [token_count, v_count])
                    pl.load(v_ub, v, [b, token_base, h, v_tile * SUB], order=[1, 3])
                    scale_v_tile_vf(v_ub, beta_act_ub, v_beta_ub, token_count)
                    # The strip-width copy the apply phase reads is written
                    # before the narrower output, so its valid shape does not
                    # clip it. It is only needed for a partial last strip.
                    if Vd % SUB != 0:
                        pl.set_validshape(v_beta_ub, [C, SUB])
                        pl.store(v_b_padded_gm, v_beta_ub, [b, h, n, 0, v_tile * SUB])
                    pl.set_validshape(v_beta_ub, [C, v_count])
                    pl.store(v_b_out, v_beta_ub, [b, h, n, 0, v_tile * SUB])

            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=E_PREP_READY,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # WY build phase. Sub-block 0 takes decay_exp, the decay mask and
            # attn, which holds the sequential solve; sub-block 1 takes a_mat,
            # q_de, kbeta_de, kbeta_t and dfac, which moves more data.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=E_SCORES_READY,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.set_validshape(decay_ub, [1, C])
            pl.load(decay_ub, decay_out, [b, h, n, 0])
            if pl.get_subblock_idx() == 0:
                pl.set_validshape(attn_ub, [C, C])
                pl.load(attn_ub, kk_scores_gm, [b, h, n, 0, 0])
            if pl.get_subblock_idx() == 1:
                pl.set_validshape(transpose_ub, [C, C])
                pl.load(transpose_ub, qk_scores_gm, [b, h, n, 0, 0])

            # The score tensors are resident in UB, so the cube may overwrite
            # them for the next item. Released before the WY work rather than
            # after it, and unguarded, so it fires once both sub-blocks have
            # finished their loads.
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=E_SCORES_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=E_WY_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # Both sub-blocks need decay_exp and the mask, so both build them.
            decay_exp_vf(decay_ub, decay_exp_ub)
            build_decay_mask_vf(decay_ub, decay_mask_ub)

            if pl.get_subblock_idx() == 0:
                pl.set_validshape(decay_exp_ub, [1, C])
                pl.store(decay_exp_out, decay_exp_ub, [b, h, n, 0])
                pl.set_validshape(decay_mask_ub, [C, C])

                negate_masked_scores_vf(attn_ub, decay_mask_ub, attn_ub)
                forward_substitute_vf(attn_ub)
                pl.set_validshape(attn_ub, [C, C])
                pl.store(attn_out, attn_ub, [b, h, n, 0, 0])

            if pl.get_subblock_idx() == 1:
                mask_attn_weights_vf(transpose_ub, decay_mask_ub, transpose_ub)
                pl.set_validshape(transpose_ub, [C, C])
                pl.store(a_mat_out, transpose_ub, [b, h, n, 0, 0])

                # dfac does not depend on the K strip: it is the same tile for
                # every one of them, so it is built once per item.
                dfac_ub = g_k_fp32_ub.next()
                decay_factor_vf(decay_ub, dfac_ub)

                for k_tile in pl.range(0, k_tiles, 1):
                    q_de_ub = g_q_scaled_ub.next()
                    kbeta_de_ub = g_k_beta_ub.next()
                    # Both loads of the iteration are issued before the first
                    # vector function, so one transfer overlaps the other's work.
                    pl.set_validshape(q_de_ub, [C, SUB])
                    pl.load(q_de_ub, q_s_out, [b, h, n, 0, k_tile * SUB])
                    pl.set_validshape(kbeta_de_ub, [C, SUB])
                    pl.load(kbeta_de_ub, k_beta_out, [b, h, n, 0, k_tile * SUB])

                    scale_rows_vf(q_de_ub, decay_exp_ub, q_de_ub)
                    pl.set_validshape(q_de_ub, [C, SUB])
                    pl.store(q_de_out, q_de_ub, [b, h, n, 0, k_tile * SUB])

                    scale_rows_vf(kbeta_de_ub, decay_exp_ub, kbeta_de_ub)
                    pl.set_validshape(kbeta_de_ub, [C, SUB])
                    pl.store(kbeta_de_out, kbeta_de_ub, [b, h, n, 0, k_tile * SUB])

                    transpose_tile_vf(kbeta_de_ub, transpose_ub)
                    pl.set_validshape(transpose_ub, [C, C])
                    pl.store(kbeta_t_out, transpose_ub, [b, h, n, k_tile * SUB, 0])

                    pl.set_validshape(dfac_ub, [C, SUB])
                    pl.store(dfac_out, dfac_ub, [b, h, n, 0, k_tile * SUB])

            # Unguarded, so it fires once both sub-blocks are done, which is
            # what makes the split safe for the apply phase: it reads attn from
            # one sub-block and kbeta_de and kbeta_t from the other.
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=E_WY_READY,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=E_PREP_FREE,
                                  sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=E_WY_FREE,
                                  sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


def gdr_fwd_train_wy_wrapper(q, k, v, g, beta, *, sync=True):
    """Build the WY transform and every residual the pipeline needs, in one launch.

    Returns the 17 outputs listed in the module docstring, in that order. The
    kernel additionally writes four tensors that only the phases pass between
    themselves, which are allocated here and not returned.
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q/k/v must be [B,T,H,*]"
    assert g.dim() == 3 and beta.dim() == 3, "g/beta must be [B,T,H]"
    B, T, H, K = q.shape
    V = v.shape[3]
    assert k.shape == q.shape, "k must match q"
    assert v.shape[:3] == q.shape[:3], "v must share [B,T,H] with q"
    assert g.shape == (B, T, H) and beta.shape == (B, T, H)
    assert K % SUB == 0 and K <= KMAX, f"K must be a multiple of {SUB}, <= {KMAX}; got {K}"
    assert H <= HMAX, f"H must be <= {HMAX} (staging tile width), got {H}"

    N = (T + C - 1) // C
    scale = 1.0 / (K ** 0.5)
    device = q.device

    q_s = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    k_f32 = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    v_b = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    g_f32 = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    beta_act = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    sig = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    q_t = torch.empty((B, H, N, K, C), dtype=torch.float32, device=device)
    decay = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    decay_exp = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    q_de = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    dfac = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    attn = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    a_mat = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    kbeta_t = torch.empty((B, H, N, K, C), dtype=torch.float32, device=device)
    u = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    w = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    w_t = torch.empty((B, H, N, K, C), dtype=torch.float32, device=device)

    # Passed between the phases, not returned.
    k_beta = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    kbeta_de = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    kk_scores = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    qk_scores = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    lower_ones = torch.empty((C, C), dtype=torch.float32, device=device)
    strip_width = ((V + SUB - 1) // SUB) * SUB
    v_b_padded = torch.empty((B, H, N, C, strip_width), dtype=torch.float32,
                             device=device)

    max_blocks = get_platform_info().core_num or 1
    num_cores = max(1, min(B * H * N, max_blocks))

    gdr_fwd_train_wy_kernel[None, num_cores](
        q, k, v, g, beta, lower_ones, v_b_padded,
        q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, k_beta,
        decay, kk_scores, qk_scores,
        decay_exp, q_de, dfac, attn, a_mat, kbeta_t, kbeta_de,
        u, w, w_t,
        B, T, H, K, V, N, scale)
    if sync:
        torch.npu.synchronize()

    return (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
            dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t)
