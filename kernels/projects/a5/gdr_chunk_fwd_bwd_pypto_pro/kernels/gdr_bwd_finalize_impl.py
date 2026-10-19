# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Backward epilogue: assemble and publish the five gradients of the gated delta rule.

Last of the eight operators in the backward pipeline. Its outputs are the
deliverable of the whole backward pass, so this kernel owns both the final
arithmetic on the gate gradients and the conversion from the internal chunked
layout back to the public token layout.

Mathematics
-----------
Per chunk of C tokens, with sig = sigmoid(beta_logit) carried over from the
training forward::

    dg_sum[t] = dg_partial[t] + dg_wy[t]
    dg[t]     = sum over t' >= t of dg_sum[t']      suffix scan within the chunk
    dbeta[t]  = dbeta_wy[t] * sig[t] * (1 - sig[t])

The suffix scan is the transpose dual of the forward's lower-triangular
cumulative gate sum, and it is exact without a cube matmul: one chunk occupies
the 64 lanes of a vector register, so a six-step Hillis-Steele scan over lane
strides 1, 2, 4, 8, 16, 32 reduces it in log2(C) steps.

dq, dk and dv arrive already complete. They are only republished: the chunked
[B, H, N, C, D] layout is trimmed to the true sequence length T, transposed to
the public [B, T, H, D] layout, and narrowed from float32 to bfloat16.

Interface
---------
``gdr_bwd_finalize_wrapper(grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig)``

    grad_o            [B, T, H, V] bf16   public layout and sequence-length
                                          anchor only; its values are never read
    dq_p              [B, H, N, C, K] f32 dq from the chunk-parallel backward
    dk_in, dv_in      [B, H, N, C, K|V]   dk and dv completed by the WY adjoint
    dg_p, dg2         [B, H, N, C] f32    gate gradient: chunk-parallel and WY parts
    dbeta_wy, sig     [B, H, N, C] f32    beta gradient before the sigmoid
                                          derivative, and the activation itself

    returns dq, dk    [B, T, H, K] bf16
            dv        [B, T, H, V] bf16
            dg, dbeta [B, T, H] f32

Layout and scheduling
---------------------
Pure vector, one launch. A work item is one (batch, chunk, head group) triple;
heads are grouped because dg and dbeta hold one scalar per token and head, so
storing them head by head would spend one descriptor on 64 floats. The group
width is chosen on the host by `_select_head_group`.

Writing dg and dbeta drives the shape of the whole kernel. `pl.store` maps a
tile's partition axis onto a destination dimension with an arbitrary stride, but
the free axis is always written as one physically contiguous run of bytes. In
[B, T, H] the contiguous dimension is H and T is strided, so the token axis must
be the tile's partition axis. The per-head results are therefore staged
head-major into [H_TILE_MAX, C], transposed once per group to token-major
[C, H_TILE_MAX], trimmed to the live (token_count, head_count) window and written
with a single store per group.
"""

import logging

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

C = 64              # tokens per chunk, a frozen constant of the operator family
H_TILE_MAX = 16     # compile-time head-group width; a multiple of 8 keeps
                    # pl.transpose on its 32-byte alignment

# Some PyPTO builds expose part of the DT_* enum as bare integers, which the
# vector-function layer rejects. Rebuild the affected ones at module scope: the
# parser inlines helper calls, so a decorated body must see plain names.
_DATA_TYPE = type(pl.DT_FP32)


def _as_data_type(value):
    return value if isinstance(value, _DATA_TYPE) else _DATA_TYPE(value)


_PL_FP32 = _as_data_type(pl.DT_FP32)
_PL_UINT32 = _as_data_type(pl.DT_UINT32)


@pl.vector_function
def dbeta_chain_rule_vf(dbeta_wy_tile, sigmoid_tile, dbeta_out_tile, head_slot):
    """dbeta = dbeta_wy * sig * (1 - sig) for one chunk, into row `head_slot`."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    one = vf.full(1.0, all_lanes, dtype=_PL_FP32)
    sigmoid = vf.load_align(sigmoid_tile, 0, dtype=_PL_FP32)
    dbeta = vf.load_align(dbeta_wy_tile, 0, dtype=_PL_FP32)
    derivative = vf.sub(one, sigmoid, all_lanes)
    derivative = vf.mul(sigmoid, derivative, all_lanes)
    dbeta = vf.mul(dbeta, derivative, all_lanes)
    vf.store_align(dbeta_out_tile, dbeta, all_lanes, [head_slot, 0])


@pl.vector_function
def dg_suffix_scan_vf(dg_partial_tile, dg_wy_tile, dg_out_tile, head_slot):
    """dg[r] = sum over c >= r of (dg_partial[c] + dg_wy[c]), into row `head_slot`.

    Six straight-line Hillis-Steele steps over lane strides 1, 2, 4, 8, 16, 32.
    They are written out rather than looped because the vector-function parser
    accepts only `pl.range` loops, and a runtime loop here would reintroduce the
    per-element cost that the register-level scan exists to avoid.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    all_lanes_u32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_UINT32)
    lane = vf.arange(0, dtype=_PL_UINT32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    dg_partial = vf.load_align(dg_partial_tile, 0, dtype=_PL_FP32)
    dg_wy = vf.load_align(dg_wy_tile, 0, dtype=_PL_FP32)
    dg_sum = vf.add(dg_partial, dg_wy, all_lanes)

    # Each step adds the value `stride` lanes further along, for the lanes that
    # still have such a neighbour; after six steps every lane holds its suffix.
    shifted_lane = vf.arange(1, dtype=_PL_UINT32)
    neighbour = vf.gather(dg_sum, shifted_lane)
    in_range = vf.lt(lane, C - 1, all_lanes_u32)
    neighbour = vf.select(neighbour, zero, in_range)
    dg_sum = vf.add(dg_sum, neighbour, all_lanes)

    shifted_lane = vf.arange(2, dtype=_PL_UINT32)
    neighbour = vf.gather(dg_sum, shifted_lane)
    in_range = vf.lt(lane, C - 2, all_lanes_u32)
    neighbour = vf.select(neighbour, zero, in_range)
    dg_sum = vf.add(dg_sum, neighbour, all_lanes)

    shifted_lane = vf.arange(4, dtype=_PL_UINT32)
    neighbour = vf.gather(dg_sum, shifted_lane)
    in_range = vf.lt(lane, C - 4, all_lanes_u32)
    neighbour = vf.select(neighbour, zero, in_range)
    dg_sum = vf.add(dg_sum, neighbour, all_lanes)

    shifted_lane = vf.arange(8, dtype=_PL_UINT32)
    neighbour = vf.gather(dg_sum, shifted_lane)
    in_range = vf.lt(lane, C - 8, all_lanes_u32)
    neighbour = vf.select(neighbour, zero, in_range)
    dg_sum = vf.add(dg_sum, neighbour, all_lanes)

    shifted_lane = vf.arange(16, dtype=_PL_UINT32)
    neighbour = vf.gather(dg_sum, shifted_lane)
    in_range = vf.lt(lane, C - 16, all_lanes_u32)
    neighbour = vf.select(neighbour, zero, in_range)
    dg_sum = vf.add(dg_sum, neighbour, all_lanes)

    shifted_lane = vf.arange(32, dtype=_PL_UINT32)
    neighbour = vf.gather(dg_sum, shifted_lane)
    in_range = vf.lt(lane, C - 32, all_lanes_u32)
    neighbour = vf.select(neighbour, zero, in_range)
    dg_sum = vf.add(dg_sum, neighbour, all_lanes)

    vf.store_align(dg_out_tile, dg_sum, all_lanes, [head_slot, 0])


_DYN5 = [pl.DYNAMIC] * 5
_DYN4 = [pl.DYNAMIC] * 4
_DYN3 = [pl.DYNAMIC] * 3


@pl.jit(auto_mutex=True)
def gdr_bwd_finalize_kernel(
    dq_p: pl.Tensor[_DYN5, pl.DT_FP32],
    dk_in: pl.Tensor[_DYN5, pl.DT_FP32],
    dv_in: pl.Tensor[_DYN5, pl.DT_FP32],
    dg_p: pl.Tensor[_DYN4, pl.DT_FP32],
    dg2: pl.Tensor[_DYN4, pl.DT_FP32],
    dbeta_wy: pl.Tensor[_DYN4, pl.DT_FP32],
    sig: pl.Tensor[_DYN4, pl.DT_FP32],
    dq: pl.Tensor[_DYN4, pl.DT_BF16],
    dk: pl.Tensor[_DYN4, pl.DT_BF16],
    dv: pl.Tensor[_DYN4, pl.DT_BF16],
    dg: pl.Tensor[_DYN3, pl.DT_FP32],
    dbeta: pl.Tensor[_DYN3, pl.DT_FP32],
    B: pl.DT_INT32, T: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32,
    Vd: pl.DT_INT32, N: pl.DT_INT32, Cc: pl.DT_INT32,
    H_TILE: pl.DT_INT32, HG: pl.DT_INT32,
):
    # Ten double-buffered tile groups, 133,120 B of the 253,952 B UB budget.
    g_chunk_fp32 = pl.make_tile_group(type=pl.TileType(shape=[C, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x00000, mutex_ids=[0, 1])
    g_chunk_bf16 = pl.make_tile_group(type=pl.TileType(shape=[C, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=0x10000, mutex_ids=[2, 3])
    g_dg_partial = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x18000, mutex_ids=[4, 5])
    g_dg_wy = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x18200, mutex_ids=[6, 7])
    g_dbeta_wy = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x18400, mutex_ids=[8, 9])
    g_sigmoid = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x18600, mutex_ids=[10, 11])
    g_dg_head_major = pl.make_tile_group(type=pl.TileType(shape=[H_TILE_MAX, C], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x18800, mutex_ids=[12, 13])
    g_dg_token_major = pl.make_tile_group(type=pl.TileType(shape=[C, H_TILE_MAX], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x1A800, mutex_ids=[14, 15])
    g_dbeta_head_major = pl.make_tile_group(type=pl.TileType(shape=[H_TILE_MAX, C], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x1C800, mutex_ids=[16, 17])
    g_dbeta_token_major = pl.make_tile_group(type=pl.TileType(shape=[C, H_TILE_MAX], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x1E800, mutex_ids=[18, 19])

    with pl.section_vector():
        # Contiguous block partitioning over (batch, chunk, head group) items.
        task_count = (B * N) * HG
        tasks_per_core = (task_count + pl.get_block_num() - 1) // pl.get_block_num()
        first_task = tasks_per_core * pl.get_block_idx()
        groups_per_batch = N * HG
        for tid in pl.range(first_task, pl.min(first_task + tasks_per_core, task_count), 1):
            b = tid // groups_per_batch
            within_batch = tid % groups_per_batch
            n = within_batch // HG
            head_group = within_batch % HG
            token_base = n * C
            token_count = pl.min(T - token_base, C)
            head_base = head_group * H_TILE
            head_count = pl.min(H - head_base, H_TILE)

            dg_head_major = g_dg_head_major.next()
            dbeta_head_major = g_dbeta_head_major.next()

            for head_slot in pl.range(0, head_count, 1):
                h = head_base + head_slot

                dg_partial = g_dg_partial.next()
                pl.set_validshape(dg_partial, [1, C])
                pl.load(dg_partial, dg_p, [b, h, n, 0])
                dg_wy = g_dg_wy.next()
                pl.set_validshape(dg_wy, [1, C])
                pl.load(dg_wy, dg2, [b, h, n, 0])
                dg_suffix_scan_vf(dg_partial, dg_wy, dg_head_major, head_slot)

                dbeta_wy_tile = g_dbeta_wy.next()
                pl.set_validshape(dbeta_wy_tile, [1, C])
                pl.load(dbeta_wy_tile, dbeta_wy, [b, h, n, 0])
                sigmoid_tile = g_sigmoid.next()
                pl.set_validshape(sigmoid_tile, [1, C])
                pl.load(sigmoid_tile, sig, [b, h, n, 0])
                dbeta_chain_rule_vf(dbeta_wy_tile, sigmoid_tile, dbeta_head_major, head_slot)

                # dq, dk, dv: cast to bfloat16 and republish in token layout.
                src_tile = g_chunk_fp32.next()
                pl.set_validshape(src_tile, [C, Kd])
                pl.load(src_tile, dq_p, [b, h, n, 0, 0])
                dst_tile = g_chunk_bf16.next()
                pl.set_validshape(dst_tile, [C, Kd])
                pl.cast(dst_tile, src_tile, mode=pl.RoundMode.CAST_ROUND)
                pl.set_validshape(dst_tile, [token_count, Kd])
                pl.store(dq, dst_tile, [b, token_base, h, 0], order=[1, 3])

                src_tile = g_chunk_fp32.next()
                pl.set_validshape(src_tile, [C, Kd])
                pl.load(src_tile, dk_in, [b, h, n, 0, 0])
                dst_tile = g_chunk_bf16.next()
                pl.set_validshape(dst_tile, [C, Kd])
                pl.cast(dst_tile, src_tile, mode=pl.RoundMode.CAST_ROUND)
                pl.set_validshape(dst_tile, [token_count, Kd])
                pl.store(dk, dst_tile, [b, token_base, h, 0], order=[1, 3])

                src_tile = g_chunk_fp32.next()
                pl.set_validshape(src_tile, [C, Vd])
                pl.load(src_tile, dv_in, [b, h, n, 0, 0])
                dst_tile = g_chunk_bf16.next()
                pl.set_validshape(dst_tile, [C, Vd])
                pl.cast(dst_tile, src_tile, mode=pl.RoundMode.CAST_ROUND)
                pl.set_validshape(dst_tile, [token_count, Vd])
                pl.store(dv, dst_tile, [b, token_base, h, 0], order=[1, 3])

            # One store per group: transpose to token-major so that the token
            # axis is the partition axis, then trim the token and head tails.
            dg_token_major = g_dg_token_major.next()
            pl.transpose(dg_token_major, dg_head_major)
            pl.set_validshape(dg_token_major, [token_count, head_count])
            pl.store(dg, dg_token_major, [b, token_base, head_base], order=[1, 2])

            dbeta_token_major = g_dbeta_token_major.next()
            pl.transpose(dbeta_token_major, dbeta_head_major)
            pl.set_validshape(dbeta_token_major, [token_count, head_count])
            pl.store(dbeta, dbeta_token_major, [b, token_base, head_base], order=[1, 2])


def _select_head_group(head_total, bn_tasks, vector_core_num, h_tile_max=H_TILE_MAX):
    """Pick the head-group width and count that minimise the critical path.

    A work item is one (batch, chunk, head group) triple and its cost is
    proportional to the number of heads it carries, while the ceil(items/cores)
    launch waves run back to back. Wall clock therefore tracks
    waves * heads_per_group, which is not the same as making the item count
    merely reach the core count: at 32 batch-chunk pairs, 8 heads and 56 cores,
    two heads per group yields 64 items over two waves and leaves 24 cores idle,
    while one head per group yields 256 items over five waves and fills them.

    Candidates are enumerated over the group width so that the group count
    ceil(H / width) is always realisable and never exceeds H; only canonical
    pairs survive. Ties go to the better core fill, then to fewer waves.
    """
    cores = max(1, vector_core_num or 1)
    best = None
    for heads_per_group in range(1, min(head_total, h_tile_max) + 1):
        groups = -(-head_total // heads_per_group)
        if -(-head_total // groups) != heads_per_group:
            continue
        items = bn_tasks * groups
        waves = max(1, -(-items // cores))
        key = (waves * heads_per_group, -items / (waves * cores), waves)
        if best is None or key < best[0]:
            best = (key, heads_per_group, groups)
    return best[1], best[2]


def gdr_bwd_finalize_wrapper(grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig,
                             beta_act=None, *, sync=True):
    """Publish the five gradients of the gated delta rule in one vector launch.

    `grad_o` supplies the public geometry only -- T and the [B, T, H, *] shape of
    the outputs -- and its values never enter the kernel. `beta_act` is part of
    the operator's public signature but is identical to `sig`, so it is unused
    here and may be omitted.

    Returns (dq, dk, dv, dg, dbeta): dq/dk/dv in bfloat16, dg/dbeta in float32,
    all in the public [B, T, H, *] layout.
    """
    B, T, H, _ = grad_o.shape
    N, K = dq_p.shape[2], dq_p.shape[4]
    V = dv_in.shape[-1]
    device = dq_p.device
    if device.type == "npu":
        torch.npu.set_device(device.index if device.index is not None else 0)

    dq = torch.empty((B, T, H, K), dtype=torch.bfloat16, device=device)
    dk = torch.empty((B, T, H, K), dtype=torch.bfloat16, device=device)
    dv = torch.empty((B, T, H, V), dtype=torch.bfloat16, device=device)
    dg = torch.empty((B, T, H), dtype=torch.float32, device=device)
    dbeta = torch.empty((B, T, H), dtype=torch.float32, device=device)

    vector_core_num = get_platform_info().vector_core_num
    heads_per_group, head_groups = _select_head_group(H, B * N, vector_core_num)
    num_cores = max(1, min(vector_core_num, (B * N) * head_groups))

    gdr_bwd_finalize_kernel[None, num_cores](
        dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig,
        dq, dk, dv, dg, dbeta,
        B, T, H, K, V, N, C, heads_per_group, head_groups)
    if sync:
        torch.npu.synchronize()
    return dq, dk, dv, dg, dbeta
