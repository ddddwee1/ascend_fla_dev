# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reduction half of the WY adjoint: dk, dv, dbeta and the WY part of dg.

The WY adjoint splits into a matmul half, which produces the four large
chunk-local gradients, and this reduction half, which contracts them along K and
V into the per-token gradients. No matmul is needed here, because every
contraction that the chain rule asks for over the triangular WY factor collapses
into a row or column sum of tensors the matmul half already produced. This
operator is therefore pure vector.

Mathematics
-----------
Per chunk, with a[:, None] meaning a per-token scalar broadcast along the free
axis::

    kbeta_decayed = k_f32 * beta_act[:, None] * decay_exp[:, None]
    dk_gated      = dkbg * (decay_exp * beta_act)[:, None]
    db1           = sum over K of (dkbg * k_f32 * decay_exp[:, None])
    db3           = sum over K of (dkb * k_f32)
    db2           = sum over V of (dvb * v_c)
    dg2_inner     = sum over K of (dkbg * kbeta_decayed)

    dv            = dvb * beta_act[:, None]
    dk            = dk_p + dkb * beta_act[:, None] + dkt + dk_gated
    dg2           = dg2_inner + beta_act * db3 - sum over K of (k_f32 * dkt)
    dbeta_wy      = db1 + db2 + db3

v_c is the primary v, widened to float32 and read in chunk layout. db3 is
reduced once and feeds both dbeta_wy and dg2.

Where each factor is applied
----------------------------
This is the easiest place in the pipeline to double-apply a factor, so the
ledger is explicit:

* 1/sqrt(K) is folded into q upstream and never appears here.
* dkb and dkt arrive already gated by the pairwise decay from the matmul half,
  and must not be gated again.
* beta_act appears on exactly five paths, once each: dv; dk through
  dkb * beta_act; dk through dk_gated; dg2 through kbeta_decayed; dg2 through
  beta_act * db3. dbeta_wy carries no beta_act.
* decay_exp appears on exactly three paths, once each: dk_gated, db1 and
  kbeta_decayed. db1 already carries it when it reaches dg2_inner.

Interface
---------
``gdr_bwd_wy_reduce_wrapper(dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp,
dk_p, v)``

    dkbg, dkb, dkt, k_f32, dk_p  [B, H, N, C, K] f32
    dvb                          [B, H, N, C, V] f32
    beta_act, decay_exp          [B, H, N, C] f32
    v                            [B, T, H, V] bf16  the primary value tensor

    returns dk        [B, H, N, C, K] f32
            dv        [B, H, N, C, V] f32
            dbeta_wy  [B, H, N, C] f32
            dg2       [B, H, N, C] f32

Layout and scheduling
---------------------
One vector launch. A work item is one (batch, head, chunk) triple, split into
contiguous per-core runs; chunks are independent and cost the same, so the split
balances itself. Each token occupies one register row and K or V the 64 lanes,
so a free axis wider than 64 is handled as two halves, suffixed `_lo` and `_hi`
below. Each half is a tile of its own rather than an offset into a wide tile, so
the two halves load as two independent transfers. When the free axis is 64 wide
the `_hi` half carries zero valid lanes: its reductions contribute nothing and
its store collapses to zero columns, so no stale data can reach an output.

The two passes run in sequence and reuse tile addresses, which keeps the live
set within UB. Two carriers between them are deliberate: the K-domain part of
dbeta_wy crosses through a GM scratch of its own, while db2 is read directly out
of the UB tile the V pass leaves it in. A tile written with
`dist=FIRST_ELEMENT` and read back in another vector function does not carry
reliably beyond its first row on this part, and a GM scratch stored and
immediately re-read within the same item is not ordered by the tile-group pipe
tracking. Every public output is written exactly once per item.
"""

import logging

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

C = 64          # tokens per chunk, and the float32 register lane count
LANES = 64      # free-axis width of one half, equal to the lane count

# Some PyPTO builds expose part of the DT_* enum as bare integers, which the
# vector-function layer rejects. Rebuild the affected ones at module scope: the
# parser inlines helper calls, so a decorated body must see plain names.
_DATA_TYPE = type(pl.DT_FP32)


def _as_data_type(value):
    return value if isinstance(value, _DATA_TYPE) else _DATA_TYPE(value)


_PL_FP32 = _as_data_type(pl.DT_FP32)


@pl.vector_function
def reduce_k_domain_vf(dkbg_lo, dkbg_hi, dkb_lo, dkb_hi, dkt_lo, dkt_hi,
                       k_lo, k_hi, dk_lo, dk_hi,
                       beta_tile, decay_tile, dbeta_partial_tile, dg2_tile, Kd):
    """K-domain pass: assemble dk in place and reduce dg2 and the dbeta partial.

    `dk_lo` and `dk_hi` arrive holding dk_p and are overwritten with the
    completed dk. The two halves are written out rather than looped because the
    accumulators stay in registers across both of them.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    for token in pl.range(0, C, 1):
        beta = vf.load_align(beta_tile, token, dist=pl.LoadDist.BRC_B32)
        decay = vf.load_align(decay_tile, token, dist=pl.LoadDist.BRC_B32)
        decay_beta = vf.mul(decay, beta, all_lanes)
        inner_dg = vf.full(0.0, all_lanes, dtype=_PL_FP32)
        db3 = vf.full(0.0, all_lanes, dtype=_PL_FP32)
        k_dot_dkt = vf.full(0.0, all_lanes, dtype=_PL_FP32)

        lanes_lo = vf.update_mask(pl.max(0, pl.min(LANES, Kd)), dtype=_PL_FP32)
        dkbg_row = vf.load_align(dkbg_lo, token * LANES, dtype=_PL_FP32)
        k_row = vf.load_align(k_lo, token * LANES, dtype=_PL_FP32)
        dkb_row = vf.load_align(dkb_lo, token * LANES, dtype=_PL_FP32)
        dkt_row = vf.load_align(dkt_lo, token * LANES, dtype=_PL_FP32)
        product = vf.mul(dkbg_row, k_row, lanes_lo)
        partial = vf.reduce_sum(product, lanes_lo)
        inner_dg = vf.add(inner_dg, partial, all_lanes)
        product = vf.mul(dkb_row, k_row, lanes_lo)
        partial = vf.reduce_sum(product, lanes_lo)
        db3 = vf.add(db3, partial, all_lanes)
        product = vf.mul(k_row, dkt_row, lanes_lo)
        partial = vf.reduce_sum(product, lanes_lo)
        k_dot_dkt = vf.add(k_dot_dkt, partial, all_lanes)
        dk_row = vf.load_align(dk_lo, token * LANES, dtype=_PL_FP32)
        dk_gated = vf.mul(dkbg_row, decay_beta, all_lanes)
        dk_sum = vf.mul(dkb_row, beta, all_lanes)
        dk_sum = vf.add(dk_sum, dkt_row, all_lanes)
        dk_sum = vf.add(dk_sum, dk_gated, all_lanes)
        dk_sum = vf.add(dk_sum, dk_row, all_lanes)
        vf.store_align(dk_lo + token * LANES, dk_sum, all_lanes)

        lanes_hi = vf.update_mask(pl.max(0, pl.min(LANES, Kd - LANES)), dtype=_PL_FP32)
        dkbg_row = vf.load_align(dkbg_hi, token * LANES, dtype=_PL_FP32)
        k_row = vf.load_align(k_hi, token * LANES, dtype=_PL_FP32)
        dkb_row = vf.load_align(dkb_hi, token * LANES, dtype=_PL_FP32)
        dkt_row = vf.load_align(dkt_hi, token * LANES, dtype=_PL_FP32)
        product = vf.mul(dkbg_row, k_row, lanes_hi)
        partial = vf.reduce_sum(product, lanes_hi)
        inner_dg = vf.add(inner_dg, partial, all_lanes)
        product = vf.mul(dkb_row, k_row, lanes_hi)
        partial = vf.reduce_sum(product, lanes_hi)
        db3 = vf.add(db3, partial, all_lanes)
        product = vf.mul(k_row, dkt_row, lanes_hi)
        partial = vf.reduce_sum(product, lanes_hi)
        k_dot_dkt = vf.add(k_dot_dkt, partial, all_lanes)
        dk_row = vf.load_align(dk_hi, token * LANES, dtype=_PL_FP32)
        dk_gated = vf.mul(dkbg_row, decay_beta, all_lanes)
        dk_sum = vf.mul(dkb_row, beta, all_lanes)
        dk_sum = vf.add(dk_sum, dkt_row, all_lanes)
        dk_sum = vf.add(dk_sum, dk_gated, all_lanes)
        dk_sum = vf.add(dk_sum, dk_row, all_lanes)
        vf.store_align(dk_hi + token * LANES, dk_sum, all_lanes)

        db1 = vf.mul(inner_dg, decay, all_lanes)
        dg2_row = vf.mul(db1, beta, all_lanes)
        beta_db3 = vf.mul(db3, beta, all_lanes)
        dg2_row = vf.add(dg2_row, beta_db3, all_lanes)
        dg2_row = vf.sub(dg2_row, k_dot_dkt, all_lanes)
        vf.store_align(dg2_tile + token, dg2_row, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)
        dbeta_partial = vf.add(db1, db3, all_lanes)
        vf.store_align(dbeta_partial_tile + token, dbeta_partial, all_lanes,
                       dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def reduce_v_domain_vf(dvb_lo, dvb_hi, v_lo, v_hi, beta_tile, db2_tile, Vd, token_count):
    """V-domain pass: turn dvb into dv in place and reduce db2.

    The rows past the sequence tail are zeroed first: the kernel-level cast only
    wrote the live rows, and the reduction below runs over all of them.
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, all_lanes, dtype=_PL_FP32)
    for pad_row in pl.range(token_count, C, 1):
        vf.store_align(v_lo + pad_row * LANES, zero, all_lanes)
        vf.store_align(v_hi + pad_row * LANES, zero, all_lanes)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for token in pl.range(0, C, 1):
        beta = vf.load_align(beta_tile, token, dist=pl.LoadDist.BRC_B32)
        db2 = vf.full(0.0, all_lanes, dtype=_PL_FP32)

        lanes_lo = vf.update_mask(pl.max(0, pl.min(LANES, Vd)), dtype=_PL_FP32)
        dvb_row = vf.load_align(dvb_lo, token * LANES, dtype=_PL_FP32)
        v_row = vf.load_align(v_lo, token * LANES, dtype=_PL_FP32)
        product = vf.mul(dvb_row, v_row, lanes_lo)
        partial = vf.reduce_sum(product, lanes_lo)
        db2 = vf.add(db2, partial, all_lanes)
        dv_row = vf.mul(dvb_row, beta, all_lanes)
        vf.store_align(dvb_lo + token * LANES, dv_row, all_lanes)

        lanes_hi = vf.update_mask(pl.max(0, pl.min(LANES, Vd - LANES)), dtype=_PL_FP32)
        dvb_row = vf.load_align(dvb_hi, token * LANES, dtype=_PL_FP32)
        v_row = vf.load_align(v_hi, token * LANES, dtype=_PL_FP32)
        product = vf.mul(dvb_row, v_row, lanes_hi)
        partial = vf.reduce_sum(product, lanes_hi)
        db2 = vf.add(db2, partial, all_lanes)
        dv_row = vf.mul(dvb_row, beta, all_lanes)
        vf.store_align(dvb_hi + token * LANES, dv_row, all_lanes)

        vf.store_align(db2_tile + token, db2, all_lanes, dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def combine_dbeta_vf(k_partial_tile, v_partial_tile, dbeta_tile):
    """dbeta_wy = (db1 + db3) + db2, one full-width row add."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    k_partial = vf.load_align(k_partial_tile, 0, dtype=_PL_FP32)
    v_partial = vf.load_align(v_partial_tile, 0, dtype=_PL_FP32)
    dbeta = vf.add(k_partial, v_partial, all_lanes)
    vf.store_align(dbeta_tile + 0, dbeta, all_lanes)


_DYN5 = [pl.DYNAMIC] * 5
_DYN4 = [pl.DYNAMIC] * 4


@pl.jit(auto_mutex=True)
def gdr_bwd_wy_reduce_kernel(
    dkbg: pl.Tensor[_DYN5, pl.DT_FP32],
    dvb: pl.Tensor[_DYN5, pl.DT_FP32],
    dkb: pl.Tensor[_DYN5, pl.DT_FP32],
    dkt: pl.Tensor[_DYN5, pl.DT_FP32],
    k_f32: pl.Tensor[_DYN5, pl.DT_FP32],
    beta_act: pl.Tensor[_DYN4, pl.DT_FP32],
    decay_exp: pl.Tensor[_DYN4, pl.DT_FP32],
    dk_p: pl.Tensor[_DYN5, pl.DT_FP32],
    v: pl.Tensor[_DYN4, pl.DT_BF16],
    dk: pl.Tensor[_DYN5, pl.DT_FP32],
    dv: pl.Tensor[_DYN5, pl.DT_FP32],
    dbeta_wy: pl.Tensor[_DYN4, pl.DT_FP32],
    dg2: pl.Tensor[_DYN4, pl.DT_FP32],
    dbeta_partial_gm: pl.Tensor[_DYN4, pl.DT_FP32],
    B: pl.DT_INT32, T: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32,
    Vd: pl.DT_INT32, N: pl.DT_INT32, Cc: pl.DT_INT32,
):
    # One [C, LANES] tile per operand half. 247,808 B of the 253,952 B UB
    # budget, the two bfloat16 tiles being half the size of the rest.
    g_dkbg_lo = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x0, mutex_ids=[0])
    g_dkbg_hi = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x4000, mutex_ids=[1])
    g_dkb_lo = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x8000, mutex_ids=[2])
    g_dkb_hi = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0xC000, mutex_ids=[3])
    g_dkt_lo = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x10000, mutex_ids=[4])
    g_dkt_hi = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x14000, mutex_ids=[5])
    g_k_lo = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x18000, mutex_ids=[6])
    g_k_hi = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x1C000, mutex_ids=[7])
    g_dk_lo = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x20000, mutex_ids=[8])
    g_dk_hi = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x24000, mutex_ids=[9])
    g_v_bf16_lo = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=0x28000, mutex_ids=[10])
    g_v_bf16_hi = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=0x2A000, mutex_ids=[11])
    g_dvb_lo = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x2C000, mutex_ids=[12])
    g_dvb_hi = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x30000, mutex_ids=[13])
    g_v_fp32_lo = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x34000, mutex_ids=[14])
    g_v_fp32_hi = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x38000, mutex_ids=[15])
    g_beta_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x3C000, mutex_ids=[16])
    g_decay_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x3C200, mutex_ids=[17])
    g_dbeta_partial_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x3C400, mutex_ids=[18])
    g_dg2_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x3C500, mutex_ids=[19])
    g_db2_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x3C600, mutex_ids=[20])
    g_dbeta_ub = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x3C700, mutex_ids=[21])

    with pl.section_vector():
        dkt_lo = g_dkt_lo.next()
        dkt_hi = g_dkt_hi.next()
        k_lo = g_k_lo.next()
        k_hi = g_k_hi.next()
        dk_lo = g_dk_lo.next()
        dk_hi = g_dk_hi.next()
        v_bf16_lo = g_v_bf16_lo.next()
        v_bf16_hi = g_v_bf16_hi.next()
        beta_ub = g_beta_ub.next()
        decay_ub = g_decay_ub.next()
        dbeta_partial_ub = g_dbeta_partial_ub.next()
        dg2_ub = g_dg2_ub.next()
        db2_ub = g_db2_ub.next()
        dbeta_ub = g_dbeta_ub.next()
        dvb_lo = g_dvb_lo.next()
        dvb_hi = g_dvb_hi.next()
        v_fp32_lo = g_v_fp32_lo.next()
        v_fp32_hi = g_v_fp32_hi.next()

        # Valid widths of the two halves. Kd and Vd are per-launch scalars, so
        # these are constant across every item this core processes.
        k_width_lo = pl.max(0, pl.min(LANES, Kd))
        k_width_hi = pl.max(0, pl.min(LANES, Kd - LANES))
        v_width_lo = pl.max(0, pl.min(LANES, Vd))
        v_width_hi = pl.max(0, pl.min(LANES, Vd - LANES))

        task_count = (B * H) * N
        tasks_per_core = (task_count + pl.get_block_num() - 1) // pl.get_block_num()
        first_task = tasks_per_core * pl.get_block_idx()
        for tid in pl.range(first_task, pl.min(first_task + tasks_per_core, task_count), 1):
            within_batch = tid % (H * N)
            b = tid // (H * N)
            n = within_batch % N
            h = within_batch // N

            pl.set_validshape(dkt_lo, [C, k_width_lo])
            pl.load(dkt_lo, dkt, [b, h, n, 0, 0])
            pl.set_validshape(dkt_hi, [C, k_width_hi])
            pl.load(dkt_hi, dkt, [b, h, n, 0, LANES])
            pl.set_validshape(k_lo, [C, k_width_lo])
            pl.load(k_lo, k_f32, [b, h, n, 0, 0])
            pl.set_validshape(k_hi, [C, k_width_hi])
            pl.load(k_hi, k_f32, [b, h, n, 0, LANES])
            pl.set_validshape(dk_lo, [C, k_width_lo])
            pl.load(dk_lo, dk_p, [b, h, n, 0, 0])
            pl.set_validshape(dk_hi, [C, k_width_hi])
            pl.load(dk_hi, dk_p, [b, h, n, 0, LANES])
            dkbg_lo = g_dkbg_lo.next()
            dkbg_hi = g_dkbg_hi.next()
            pl.set_validshape(dkbg_lo, [C, k_width_lo])
            pl.load(dkbg_lo, dkbg, [b, h, n, 0, 0])
            pl.set_validshape(dkbg_hi, [C, k_width_hi])
            pl.load(dkbg_hi, dkbg, [b, h, n, 0, LANES])
            dkb_lo = g_dkb_lo.next()
            dkb_hi = g_dkb_hi.next()
            pl.set_validshape(dkb_lo, [C, k_width_lo])
            pl.load(dkb_lo, dkb, [b, h, n, 0, 0])
            pl.set_validshape(dkb_hi, [C, k_width_hi])
            pl.load(dkb_hi, dkb, [b, h, n, 0, LANES])
            pl.set_validshape(beta_ub, [1, C])
            pl.load(beta_ub, beta_act, [b, h, n, 0])
            pl.set_validshape(decay_ub, [1, C])
            pl.load(decay_ub, decay_exp, [b, h, n, 0])

            reduce_k_domain_vf(dkbg_lo, dkbg_hi, dkb_lo, dkb_hi,
                               dkt_lo, dkt_hi, k_lo, k_hi, dk_lo, dk_hi,
                               beta_ub, decay_ub, dbeta_partial_ub, dg2_ub, Kd)

            pl.set_validshape(dk_lo, [C, k_width_lo])
            pl.store(dk, dk_lo, [b, h, n, 0, 0])
            pl.set_validshape(dk_hi, [C, k_width_hi])
            pl.store(dk, dk_hi, [b, h, n, 0, LANES])
            pl.set_validshape(dg2_ub, [1, C])
            pl.store(dg2, dg2_ub, [b, h, n, 0])
            pl.set_validshape(dbeta_partial_ub, [1, C])
            pl.store(dbeta_partial_gm, dbeta_partial_ub, [b, h, n, 0])

            pl.set_validshape(dvb_lo, [C, v_width_lo])
            pl.load(dvb_lo, dvb, [b, h, n, 0, 0])
            pl.set_validshape(dvb_hi, [C, v_width_hi])
            pl.load(dvb_hi, dvb, [b, h, n, 0, LANES])

            token_base = n * C
            token_count = pl.min(T - token_base, C)
            # v is the one public [B, T, H, V] input: order= makes the 2-D tile
            # walk (T, V) instead of the default trailing (H, V), which absorbs
            # the layout change into the load instead of a host transpose.
            pl.set_validshape(v_bf16_lo, [token_count, v_width_lo])
            pl.load(v_bf16_lo, v, [b, token_base, h, 0], order=[1, 3])
            pl.set_validshape(v_bf16_hi, [token_count, v_width_hi])
            pl.load(v_bf16_hi, v, [b, token_base, h, LANES], order=[1, 3])
            pl.set_validshape(v_fp32_lo, [token_count, v_width_lo])
            pl.cast(v_fp32_lo, v_bf16_lo, mode=pl.RoundMode.CAST_NONE)
            pl.set_validshape(v_fp32_hi, [token_count, v_width_hi])
            pl.cast(v_fp32_hi, v_bf16_hi, mode=pl.RoundMode.CAST_NONE)

            reduce_v_domain_vf(dvb_lo, dvb_hi, v_fp32_lo, v_fp32_hi,
                               beta_ub, db2_ub, Vd, token_count)

            pl.set_validshape(dvb_lo, [C, v_width_lo])
            pl.store(dv, dvb_lo, [b, h, n, 0, 0])
            pl.set_validshape(dvb_hi, [C, v_width_hi])
            pl.store(dv, dvb_hi, [b, h, n, 0, LANES])

            pl.set_validshape(dbeta_partial_ub, [1, C])
            pl.load(dbeta_partial_ub, dbeta_partial_gm, [b, h, n, 0])
            combine_dbeta_vf(dbeta_partial_ub, db2_ub, dbeta_ub)
            pl.set_validshape(dbeta_ub, [1, C])
            pl.store(dbeta_wy, dbeta_ub, [b, h, n, 0])


def gdr_bwd_wy_reduce_wrapper(dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v, *, sync=True):
    """Reduce the WY adjoint to dk, dv, dbeta_wy and dg2 in one launch.

    Outputs stay in chunk layout; trimming to the real sequence length is the
    epilogue's job.

    Returns (dk, dv, dbeta_wy, dg2), all float32.
    """
    B, H, N, _, K = dkbg.shape
    V = dvb.shape[-1]
    T = v.shape[1]
    device = dkbg.device

    dk = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    dv = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    dbeta_wy = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    dg2 = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    # Carries the K-domain part of dbeta_wy across the two passes; see the
    # module docstring on why it is a scratch of its own.
    dbeta_partial_gm = torch.empty((B, H, N, C), dtype=torch.float32, device=device)

    num_cores = max(1, min(get_platform_info().vector_core_num, B * H * N))
    gdr_bwd_wy_reduce_kernel[None, num_cores](
        dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v,
        dk, dv, dbeta_wy, dg2, dbeta_partial_gm,
        B, T, H, K, V, N, C)
    if sync:
        torch.npu.synchronize()
    return dk, dv, dbeta_wy, dg2
