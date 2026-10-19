# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gated_delta_rule_fwd — chunked gated delta rule, forward pass, one fused kernel.

Inputs are q, k, v, beta in bfloat16 and a per-token log-decay g in float32,
all laid out [B, T, H, D]. The output o is [B, T, H, V] bfloat16. All internal
arithmetic is float32.

Math
----
The sequence is cut into chunks of C = 64 tokens. Writing the per-chunk
quantities for one (batch, head) and using `decay = cumsum(g)` within the chunk:

    beta_act      = sigmoid(beta)
    q_s           = q / sqrt(K)
    k_beta        = k * beta_act            v_beta        = v * beta_act
    decay_exp[i]  = exp(decay[i])           decay_tail[i] = exp(decay[C-1] - decay[i])
    decay_mask    = tril(exp(decay_i - decay_j), incl. diagonal)
    kk_gram       = k_beta @ k^T            qk_gram       = q_s @ k^T
    wy_t          = (I + tril_strict(kk_gram * decay_mask))^-1      (WY transform)
    attn_intra    = tril(qk_gram * decay_mask, incl. diagonal)
    u             = wy_t @ v_beta           w             = wy_t @ (k_beta * decay_exp)
    q_decay       = q_s * decay_exp

and then a strictly sequential recurrence over the chunks, carrying the
[K, V] state S (S starts at zero for each (batch, head)):

    v_prime = w @ S_prev
    v_new   = u - v_prime
    o       = q_decay @ S_prev + attn_intra @ v_new
    S       = S_prev * decay_exp[C-1] + k^T @ (v_new * decay_tail)

Kernel structure
----------------
ONE `@pl.jit` kernel and ONE launch. Two phases separated by a single
grid-wide `sync_all(HARD, MIX)` barrier; both AIV sub-blocks of every launched
block compute in both phases.

  chunk-prep phase — parallel over all B*H*ceil(T/C) chunks, four steps that
  hand off to each other inside the block through an INTRA_BLOCK handshake
  (events e0..e3) rather than through separate launches:
      preprocess (Vec)  beta_act, q_s, k_f32, k_beta, v_beta, the g column,
                        the lower-triangular ones matrix, and k^T
      gram       (Cube) decay = g @ upper_ones, kk_gram, qk_gram
      WY         (Vec)  decay_exp, decay_mask, wy_t by forward substitution,
                        attn_intra, q_decay, k_beta_decay, decay_tail
      u/w        (Cube) u = wy_t @ v_beta, w = wy_t @ k_beta_decay

  scan phase — the work item is the flat pair (bh, v_tile), giving
  B*H*ceil(V/VT) independent items. The V tiles are independent because every
  step of the recurrence touches only the current V column block. The chunk
  loop inside an item is serial: it is a true recurrence in S. Cube and vector
  alternate through an eight-event depth-1 INTRA_BLOCK handshake (e0..e7).

The state S is carried through a GM workspace rather than kept resident in L1,
which is what lets K and V reach 256 without an L1 budget problem.

Hardware notes (measured on Ascend 950PR, DAV_3510 / NpuArch=3510, CANN 9.1)
---------------------------------------------------------------------------
Three build behaviours this kernel is shaped around; each was found by a wrong
numeric result, not by a compile error, so they are easy to reintroduce:

  * A matmul whose N is below 64 returns wrong results, so V is padded to a
    multiple of 64 in the host entry and every V tile is full width.
  * An L0A or L0C tile declared [128, .] but used with valid_shape [64, .]
    returns wrong results; the scan therefore uses 64-row K output blocks.
  * A transpose-loaded tile (order=[2,1]) is usable as a matmul RIGHT operand
    but not as a LEFT one, so k^T is materialised explicitly by a vector
    transpose during preprocess and loaded normally.
"""

import torch
import torch_npu  # noqa: F401  (registers NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info


# ── compile-time constants ──
C = 64            # chunk size (SPEC-frozen internal constant)
TILE_COL = 128    # K/V column tile width (covers K<=256 via column loop)
LANES = 64        # fp32 VF register width
BCOLW = 64        # g/beta column-tile width == LANES (row pitch = one register)

# ── UB address map (all 32B-aligned) ──
VA_INBF0 = 0x00000                 # in_bf  [C,TILE_COL] bf16
VA_INBF1 = 0x04000
VA_F32I0 = 0x08000                 # f32in  [C,TILE_COL] fp32  (WY aliases as attn [C,C])
VA_F32I1 = 0x10000                 # f32in slot1               (WY aliases as A    [C,C])
VA_OUT0 = 0x18000                  # out    [C,TILE_COL] fp32  (WY reuses for q_de/kbeta_de/dfac)
VA_OUT1 = 0x20000
VA_BETABF = 0x28000                # beta_bf [C,BCOLW] bf16
VA_BLF32 = 0x2A000                 # beta_logit fp32 [C,BCOLW]
VA_BACT = 0x2E000                  # beta_act  fp32 [C,BCOLW]
VA_GCOL = 0x32000                  # g column  fp32 [C,BCOLW]
VA_LTUB = 0x36000                  # LTones build UB [C,C] fp32 (WY reuses for L_mask) -> ends 0x3A000
# ── WY-only UB tiles at the free tail above preprocess's map (16 KiB free: 0x3A000..0x3E000) ──
VA_DEC = 0x3A000                   # decay row [1,C] fp32 (256 B)
VA_DEXP = 0x3A100                  # decay_exp row [1,C] fp32 (256 B) -> ends 0x3A200

# ── L1(Mat) address map (gram cube operands) ──
L1_GROW = 0x00000                  # g as row [MMIN,C] (decay)
L1_UPPER = 0x08000                 # upper-tri ones [C,C] (decay)
L1_KB = 0x10000                    # k_beta [C,TILE_COL]
L1_KF = 0x18000                    # k^T    [TILE_COL,C]
L1_QS = 0x20000                    # q_s    [C,TILE_COL]
L1_WY_T = 0x28000                  # u/w attn [C,C]
L1_VB = 0x2C000                    # u/w v_b  [C,TILE_COL]
L1_KBDE = 0x34000                  # u/w kbeta_de [C,TILE_COL] (ends 0x3C000)

# ── L0A/L0B/L0C address maps ──
L0A_0 = 0x0000
L0A_1 = 0x8000
L0B_0 = 0x0000
L0B_1 = 0x8000
L0C_DECAY = 0x00000
L0C_KK_GRAM = 0x08000
L0C_QK_GRAM = 0x10000
L0C_U = 0x14000                    # u/w u [C,TILE_COL] fp32 (gram-disjoint)
L0C_W = 0x1C000                    # u/w w [C,TILE_COL] fp32 (ends 0x24000 < 0x40000)
MMIN = 16                          # min physical M for an Acc/Left tile (fractal rows)


# =============================================================================
#   Vector functions — preprocess arithmetic + LTones
# =============================================================================
@pl.vector_function
def _sigmoid_col_vf(bl_tile, ba_tile, n_rows: pl.DT_INT64):
    """beta_act[m] = 1/(1+exp(-logit[m])) for each of n_rows rows (col-0 value)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        off = m * BCOLW
        x = vf.load_align(bl_tile, off)
        negx = vf.neg(x, preg)
        e = vf.exp(negx, preg)
        d = vf.adds(e, 1.0, preg)
        ba = vf.div(one, d, preg)
        vf.store_align(ba_tile + off, ba, preg)


@pl.vector_function
def _scale_vf(in_tile, out_tile, n_rows: pl.DT_INT64, n_cols: pl.DT_INT64,
             scale: pl.DT_FP32):
    """out = in * scale over [n_rows, n_cols] (row pitch = TILE_COL)."""
    n_regs = (n_cols + LANES - 1) // LANES
    for m in pl.range(0, n_rows):
        base = m * TILE_COL
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            off = base + r * LANES
            reg = vf.load_align(in_tile, off)
            out = vf.muls(reg, scale, mreg)
            vf.store_align(out_tile + off, out, mreg)


@pl.vector_function
def _mul_beta_vf(in_tile, ba_tile, out_tile, n_rows: pl.DT_INT64,
                 n_cols: pl.DT_INT64):
    """out[m,:] = in[m,:] * beta_act[m]  (per-row scalar broadcast from col-0)."""
    n_regs = (n_cols + LANES - 1) // LANES
    for m in pl.range(0, n_rows):
        beta_bc = vf.load_align(ba_tile, m * BCOLW, dist=pl.LoadDist.BRC_B32)
        base = m * TILE_COL
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            off = base + r * LANES
            reg = vf.load_align(in_tile, off)
            out = vf.mul(reg, beta_bc, mreg)
            vf.store_align(out_tile + off, out, mreg)


@pl.vector_function
def _zero_rows_vf(tile, r0: pl.DT_INT64, r1: pl.DT_INT64, n_cols: pl.DT_INT64):
    """Zero rows [r0, r1) of a [C, TILE_COL] fp32 tile (row pitch TILE_COL)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES - 1) // LANES
    for m in pl.range(r0, r1):
        base = m * TILE_COL
        for r in pl.range(0, n_regs):
            vf.store_align(tile + base + r * LANES, zero, preg)


@pl.vector_function
def _zero_rows_col_vf(tile, r0: pl.DT_INT64, r1: pl.DT_INT64):
    """Zero rows [r0, r1) of a [C, BCOLW] fp32 column-staging tile."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(r0, r1):
        vf.store_align(tile + m * BCOLW, zero, preg)


@pl.vector_function
def _build_lower_tri_ones_vf(lt_tile):
    """LTones[r, j] = 1 if j <= r else 0  (lower-tri incl diag), row pitch C."""
    mask_i = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    mask_f = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, mask_f, dtype=pl.DT_FP32)
    cols = vf.arange(0, dtype=pl.DT_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    for r in pl.range(0, C):
        one = vf.full(1.0, mask_f, dtype=pl.DT_FP32)
        m = vf.lt(cols, r + 1, mask_i)      # cols < r+1  <=>  cols <= r
        row = vf.select(one, zero, m)        # 1 where cols<=r else 0
        vf.store_align(lt_tile + r * C, row, mask_f)


# =============================================================================
#   Vector functions — WY construction
# =============================================================================
@pl.vector_function
def _decay_exp_vf(dec_tile, dexp_tile):
    """decay_exp[j] = exp(decay[j]) over the [1,C] decay row."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    d = vf.load_align(dec_tile, 0)
    de = vf.exp_sub(d, zero, preg)           # exp(d - 0) = exp(d)
    vf.store_align(dexp_tile + 0, de, preg)


@pl.vector_function
def _decay_mask_vf(dec_tile, lt_tile):
    """L_mask[i,j] = exp(decay_i - decay_j) for j<=i else 0 (row pitch C)."""
    maski = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    maskf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, maskf, dtype=pl.DT_FP32)
    cols = vf.arange(0, dtype=pl.DT_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    dvec = vf.load_align(dec_tile, 0)                       # decay[0..C-1]
    for li in pl.range(0, C):
        di = vf.load_align(dec_tile, li, dist=pl.LoadDist.BRC_B32)  # decay[li]
        lrow = vf.exp_sub(di, dvec, maskf)                 # exp(decay_i - decay_j)
        m = vf.lt(cols, li + 1, maski)                     # cols <= li
        lrow = vf.select(lrow, zero, m)                    # keep j<=i else 0
        vf.store_align(lt_tile + li * C, lrow, maskf)


@pl.vector_function
def _wy_lower_init_vf(kk_tile, lt_tile):
    """attn_pre = -tril_strict(kk_gram * L_mask), in place on kk_tile (row pitch C)."""
    maski = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    maskf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, maskf, dtype=pl.DT_FP32)
    cols = vf.arange(0, dtype=pl.DT_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    for ai in pl.range(0, C):
        p = vf.load_align(kk_tile, ai * C)
        l = vf.load_align(lt_tile, ai * C)
        r = vf.mul(p, l, maskf)
        r = vf.neg(r, maskf)
        m = vf.lt(cols, ai, maski)                         # cols < ai (strict lower)
        r = vf.select(r, zero, m)
        vf.store_align(kk_tile + ai * C, r, maskf)


@pl.vector_function
def _wy_forward_substitution_vf(kk_tile):
    """forward substitution, then +I, in place on the strict-lower attn tile.

    for i in 1..C-1:  acc = sum_{m<i} attn[i,m] * attn[m,:];  row_i += acc.
    Rows m<i are the ALREADY-updated rows from earlier iterations -> loop-carried
    in-tile dependency; a VST_VLD barrier after each row store makes step i+1 see
    row i.  Then the diagonal is set to 1 (pre-diagonal value is 0, so == +I).
    """
    maskf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, maskf, dtype=pl.DT_FP32)
    for fi in pl.range(1, C):
        acc = vf.full(0.0, maskf, dtype=pl.DT_FP32)
        for fm in pl.range(0, fi):
            sc = vf.load_align(kk_tile, fi * C + fm, dist=pl.LoadDist.BRC_B32)  # attn[fi,fm]
            row_m = vf.load_align(kk_tile, fm * C)                              # attn[fm,:]
            prod = vf.mul(row_m, sc, maskf)
            acc = vf.add(acc, prod, maskf)                                     # acc += row_m*sc
        old_i = vf.load_align(kk_tile, fi * C)
        newi = vf.add(old_i, acc, maskf)
        vf.store_align(kk_tile + fi * C, newi, maskf)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vf.mem_bar(mode=pl.MemBarMode.VST_VST)
    for fid in pl.range(0, C):
        vf.store_align(kk_tile + fid * C + fid, one, maskf,
                       dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def _intra_attn_build_vf(qk_tile, lt_tile):
    """A = tril_incl_diag(qk_gram * L_mask), in place on qk_tile (row pitch C)."""
    maski = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    maskf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, maskf, dtype=pl.DT_FP32)
    cols = vf.arange(0, dtype=pl.DT_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    for ai in pl.range(0, C):
        p = vf.load_align(qk_tile, ai * C)
        l = vf.load_align(lt_tile, ai * C)
        r = vf.mul(p, l, maskf)
        m = vf.lt(cols, ai + 1, maski)                     # cols <= ai (lower incl diag)
        r = vf.select(r, zero, m)
        vf.store_align(qk_tile + ai * C, r, maskf)


@pl.vector_function
def _rowscale_vf(src_tile, dexp_tile, dst_tile, n_cols: pl.DT_INT64):
    """dst[m,:] = src[m,:] * decay_exp[m]  (per-row scalar; row pitch TILE_COL)."""
    n_regs = (n_cols + LANES - 1) // LANES
    for m in pl.range(0, C):
        sc = vf.load_align(dexp_tile, m, dist=pl.LoadDist.BRC_B32)  # decay_exp[m]
        base = m * TILE_COL
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            off = base + r * LANES
            reg = vf.load_align(src_tile, off)
            out = vf.mul(reg, sc, mreg)
            vf.store_align(dst_tile + off, out, mreg)


@pl.vector_function
def _decay_tail_vf(dec_tile, dfac_tile, n_cols: pl.DT_INT64):
    """dfac[m,:] = exp(decay[C-1] - decay[m]) broadcast over n_cols (row pitch TILE_COL)."""
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES - 1) // LANES
    dlast = vf.load_align(dec_tile, C - 1, dist=pl.LoadDist.BRC_B32)  # decay[C-1]
    for m in pl.range(0, C):
        dm = vf.load_align(dec_tile, m, dist=pl.LoadDist.BRC_B32)     # decay[m]
        s = vf.exp_sub(dlast, dm, active)                            # exp(decay[C-1]-decay[m])
        base = m * TILE_COL
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            vf.store_align(dfac_tile + base + r * LANES, s, mreg)


# =============================================================================
#   Single @pl.jit kernel — preprocess (Vector) + gram (Cube) + WY (Vector), 1 launch
# =============================================================================

# =============================================================================
#   Scan phase constants + address map.  The two phases are separated
#   by a grid-wide sync_all(MIX) barrier, so aliasing chunk-prep addresses/mutex
#   ids across the barrier is safe (probed on A5).
#   VT = 64  : V-tile width.  L0A/L0B slots are 32768 B = [128,64]/[64,128];
#              a 64-wide V tile keeps every matmul operand within one slot.
#   KT = 128 : K-block width for the K-contraction phase (k_tiles).
# =============================================================================
VT = 64
KT = 128
# AccToVecMode.DualModeSplitM hands each AIV half of the L0C M rows.
# C is even, so aligned_M == C and each sub-block owns exactly HC rows.
HC = C // 2
# UB (Vector)
SCAN_STATE_UB = 0x00000   # state_ub [KT, VT] fp32 = 32768  (one K-block of S)
SCAN_DELTA_UB = 0x08000   # ub_delta [KT, VT] fp32 = 32768  (SHARED id 31)
SCAN_VP_UB    = 0x10000   # ub_vp    [C, VT]  fp32 = 16384  (v_prime then o) (SHARED id30)
SCAN_U_UB    = 0x14000   # ub_uv    [C, VT]  fp32
SCAN_VSCALED_UB    = 0x18000   # ub_vs    [C, VT]  fp32
SCAN_OBF_UB   = 0x1C000   # ub_obf   [C, VT]  bf16
SCAN_DECAYTAIL_UB  = 0x1E000   # ub_dfac  [C, 8]   fp32
SCAN_DEXP_UB  = 0x1E800   # ub_dexp  [1, C]   fp32
# L1 (Mat)
SCAN_STATE_L1 = 0x00000   # state_l1 [2*KT, VT]  (full [Kd, vcur])
SCAN_W_L1     = 0x10000   # w_l1     [C, 2*KT]
SCAN_QDECAY_L1   = 0x20000   # q_decay_l1   [C, 2*KT]
SCAN_KT_L1   = 0x30000   # kfT_l1   [2*KT, C]
SCAN_ATTN_L1     = 0x40000   # A_l1     [C, C]
SCAN_VNEW_L1  = 0x44000   # vnew_l1  [C, VT]
SCAN_VSCALED_L1    = 0x48000   # vscaled_l1[C, VT]
# L0A / L0B (double buffered, 32768 B slots) — reuse chunk-prep L0 addrs
SCAN_L0A_0 = 0x0000
SCAN_L0A_1 = 0x8000
SCAN_L0B_0 = 0x0000
SCAN_L0B_1 = 0x8000
# L0C
SCAN_VP_L0C    = 0x00000  # [C, VT]
SCAN_O_L0C     = 0x04000  # [C, VT]
SCAN_DELTA_L0C = 0x08000  # [KT, VT]


@pl.vector_function
def _scan_vnew_vf(vp_tile, uv_tile, dfac_tile, vs_tile, n_rows: pl.DT_INT64,
                 n_cols: pl.DT_INT64):
    """v_new = u - v_prime (-> uv_tile); v_scaled = v_new * dfac[row] (-> vs_tile).

    dfac is a per-row scalar (col-0 of ub_dfac, pitch 8).  Row pitch VT.  Loops
    n_rows rows; pad rows (u=0, v_prime=0) yield v_new=0 -> harmless.  n_cols<=VT.
    """
    for r in pl.range(0, n_rows):
        fac = vf.load_align(dfac_tile, r * 8, dist=pl.LoadDist.BRC_B32)
        mreg = vf.update_mask(n_cols, dtype=pl.DT_FP32)
        off = r * VT
        uu = vf.load_align(uv_tile, off)
        vp = vf.load_align(vp_tile, off)
        vn = vf.sub(uu, vp, mreg)
        vs = vf.mul(vn, fac, mreg)
        vf.store_align(uv_tile + off, vn, mreg)
        vf.store_align(vs_tile + off, vs, mreg)


@pl.vector_function
def _scan_state_update_vf(state_tile, delta_tile, dexp_tile, n_rows: pl.DT_INT64,
                     n_cols: pl.DT_INT64, didx: pl.DT_INT64):
    """S = S * decay_exp[didx] + delta  over [n_rows, n_cols] (pitch VT).

    didx = valid_c-1 (per-chunk decay_exp index; one decay_exp value per chunk).  n_cols<=VT.
    """
    dreg = vf.load_align(dexp_tile, didx, dist=pl.LoadDist.BRC_B32)
    for r in pl.range(0, n_rows):
        mreg = vf.update_mask(n_cols, dtype=pl.DT_FP32)
        off = r * VT
        s = vf.load_align(state_tile, off)
        d = vf.load_align(delta_tile, off)
        s = vf.mul(s, dreg, mreg)
        s = vf.add(s, d, mreg)
        vf.store_align(state_tile + off, s, mreg)


@pl.vector_function
def _zero_tile_vf(t, n_rows: pl.DT_INT64, n_cols: pl.DT_INT64):
    """Zero [n_rows, n_cols] of a fp32 tile (pitch VT).  n_cols<=VT."""
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    z = vf.full(0.0, active, dtype=pl.DT_FP32)
    for r in pl.range(0, n_rows):
        vf.store_align(t + r * VT, z, active)


# (o fp32->bf16 cast is done with the tile-op pl.cast in section_vector, mirroring
#  the fused kernel's pl.cast usage; no VF needed.)


@pl.jit(auto_mutex=True)
def gated_delta_rule_fwd_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],   # [B,T,H,1]
    beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [B,T,H,1]
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],   # [B,T,H,V]
    qs_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],   # [BHN,C,K]
    kf_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],   # [BHN,C,K]
    kb_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],   # [BHN,C,K]
    gf_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],   # [BHN,C,1]
    vb_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],   # [BHN,C,TILE_COL] (v_b, pad-col zeroed)
    tri_ones_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],               # [C,C]
    dec_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],              # [BHN,C]  (gram->WY scratch)
    kk_gram_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],   # [BHN,C,C] (scratch)
    qk_gram_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],   # [BHN,C,C] (scratch)
    decay_exp_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],             # [BHN,C]
    q_decay_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],      # [BHN,C,K]
    k_beta_decay_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [BHN,C,K]
    decay_tail_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],      # [BHN,C,K]
    decay_mask_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],     # [BHN,C,C]
    wy_t_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],      # [BHN,C,C]
    attn_intra_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],         # [BHN,C,C]
    u_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],         # [BHN,C,V]  (u/w final)
    w_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],         # [BHN,C,K]  (u/w final)
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],   # [B,T,H,V] out
    state_gm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],        # [BH,K,V] persistent S
    vnew_gm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],         # [BH,C,V]
    vscaled_gm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],      # [BH,C,V]
    kft_gm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],          # [BHN,K,C] = k^T (normal layout)
    scale: pl.DT_FP32,
):
    # ---- tile-type declarations (UB: preprocess) ----
    bf_type = pl.TileType(shape=[C, TILE_COL], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec,
                          valid_shape=[-1, -1], pad=pl.TilePad.zero)
    f32_type = pl.TileType(shape=[C, TILE_COL], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec,
                           valid_shape=[-1, -1], pad=pl.TilePad.zero)
    bcol_bf = pl.TileType(shape=[C, BCOLW], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec,
                          valid_shape=[-1, -1], pad=pl.TilePad.zero)
    bcol_f32 = pl.TileType(shape=[C, BCOLW], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec,
                           valid_shape=[-1, -1], pad=pl.TilePad.zero)
    ltub_type = pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec,
                            valid_shape=[-1, -1], pad=pl.TilePad.zero)
    # ---- tile-type declarations (UB: WY) ----
    cc_type = pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec,
                          valid_shape=[-1, -1], pad=pl.TilePad.zero)
    dec_type = pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec,
                           valid_shape=[-1, -1], pad=pl.TilePad.zero)

    in_bf = pl.make_tile_group(type=bf_type, addrs=[VA_INBF0, VA_INBF1], mutex_ids=[0, 1])
    f32in = pl.make_tile_group(type=f32_type, addrs=[VA_F32I0, VA_F32I1], mutex_ids=[2, 3])
    out_grp = pl.make_tile_group(type=f32_type, addrs=[VA_OUT0, VA_OUT1], mutex_ids=[4, 5])
    beta_bf_g = pl.make_tile_group(type=bcol_bf, addrs=[VA_BETABF], mutex_ids=[6])
    bl_g = pl.make_tile_group(type=bcol_f32, addrs=[VA_BLF32], mutex_ids=[7])
    ba_g = pl.make_tile_group(type=bcol_f32, addrs=[VA_BACT], mutex_ids=[8])
    g_g = pl.make_tile_group(type=bcol_f32, addrs=[VA_GCOL], mutex_ids=[9])
    lt_ub_g = pl.make_tile_group(type=ltub_type, addrs=[VA_LTUB], mutex_ids=[10])
    # WY [C,C] tiles ALIAS the two f32in slots (same addrs + same mutex_ids);
    # preprocess (f32in) and WY (attn/A) never overlap in time within one tid.
    kk_gram_ub_g = pl.make_tile_group(type=cc_type, addrs=[VA_F32I0], mutex_ids=[2])
    qk_gram_ub_g = pl.make_tile_group(type=cc_type, addrs=[VA_F32I1], mutex_ids=[3])
    # WY fresh row tiles at the free UB tail above the preprocess map.
    dec_ub_g = pl.make_tile_group(type=dec_type, addrs=[VA_DEC], mutex_ids=[23])
    dexp_ub_g = pl.make_tile_group(type=dec_type, addrs=[VA_DEXP], mutex_ids=[24])
    # ---- k^T is materialised INSIDE preprocess (sub-block 0).
    #      Square 64x64 transposes, identical to the deleted scan pre-pass;
    #      the three staging tiles ALIAS the preprocess in_bf / f32in slots (same addrs,
    #      same mutex ids), which is this kernel's established alias idiom --
    #      the k^T loop runs after the q_s loop inside the SAME sub-block-0
    #      branch, so auto_mutex serialises the reuse.  No new UB bytes, no new
    #      mutex ids (0..31 are already fully allocated in chunk-prep). ----
    kt_bf_t = pl.TileType(shape=[C, LANES], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec,
                          valid_shape=[-1, -1], pad=pl.TilePad.zero)
    kt_f32_t = pl.TileType(shape=[C, LANES], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec,
                           valid_shape=[-1, -1], pad=pl.TilePad.zero)
    kt_tr_t = pl.TileType(shape=[LANES, C], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec,
                          valid_shape=[-1, -1], pad=pl.TilePad.zero)
    kt_bf_g = pl.make_tile_group(type=kt_bf_t, addrs=[VA_INBF0], mutex_ids=[0])
    kt_f32_g = pl.make_tile_group(type=kt_f32_t, addrs=[VA_F32I0], mutex_ids=[2])
    kt_tr_g = pl.make_tile_group(type=kt_tr_t, addrs=[VA_F32I1], mutex_ids=[3])

    # ---- tile-type declarations (Cube: L1 / L0A / L0B / L0C) ----
    grow_l1_t = pl.TileType(shape=[MMIN, C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
    upper_l1_t = pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
    col_l1_t = pl.TileType(shape=[C, TILE_COL], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
    kt_l1_t = pl.TileType(shape=[TILE_COL, C], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
    l0a_dec_t = pl.TileType(shape=[MMIN, C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1])
    l0a_mm_t = pl.TileType(shape=[C, TILE_COL], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1])
    l0b_dec_t = pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1])
    l0b_mm_t = pl.TileType(shape=[TILE_COL, C], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1])
    acc_dec_t = pl.TileType(shape=[MMIN, C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1])
    acc_cc_t = pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1])

    grow_l1_g = pl.make_tile_group(type=grow_l1_t, addrs=[L1_GROW], mutex_ids=[11])
    upper_l1_g = pl.make_tile_group(type=upper_l1_t, addrs=[L1_UPPER], mutex_ids=[12])
    kb_l1_g = pl.make_tile_group(type=col_l1_t, addrs=[L1_KB], mutex_ids=[13])
    kf_l1_g = pl.make_tile_group(type=kt_l1_t, addrs=[L1_KF], mutex_ids=[14])
    qs_l1_g = pl.make_tile_group(type=col_l1_t, addrs=[L1_QS], mutex_ids=[15])
    l0a_dec_g = pl.make_tile_group(type=l0a_dec_t, addrs=[L0A_0, L0A_1], mutex_ids=[16, 17])
    l0a_g = pl.make_tile_group(type=l0a_mm_t, addrs=[L0A_0, L0A_1], mutex_ids=[16, 17])
    l0b_dec_g = pl.make_tile_group(type=l0b_dec_t, addrs=[L0B_0, L0B_1], mutex_ids=[18, 19])
    l0b_g = pl.make_tile_group(type=l0b_mm_t, addrs=[L0B_0, L0B_1], mutex_ids=[18, 19])
    acc_dec_g = pl.make_tile_group(type=acc_dec_t, addrs=[L0C_DECAY], mutex_ids=[20])
    acc_p1_g = pl.make_tile_group(type=acc_cc_t, addrs=[L0C_KK_GRAM], mutex_ids=[21])
    acc_p2_g = pl.make_tile_group(type=acc_cc_t, addrs=[L0C_QK_GRAM], mutex_ids=[22])

    # ---- u/w (Cube): u = wy_t @ v_beta, w = wy_t @ k_beta_decay ----
    attn_l1_t = pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
    uw_l1_t = pl.TileType(shape=[C, TILE_COL], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
    l0b_uw_t = pl.TileType(shape=[C, TILE_COL], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1])
    acc_uw_t = pl.TileType(shape=[C, TILE_COL], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1])
    l0a_attn_t = pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1])
    attn_l1_g = pl.make_tile_group(type=attn_l1_t, addrs=[L1_WY_T], mutex_ids=[25])
    vb_l1_g = pl.make_tile_group(type=uw_l1_t, addrs=[L1_VB], mutex_ids=[26])
    kbde_l1_g = pl.make_tile_group(type=uw_l1_t, addrs=[L1_KBDE], mutex_ids=[27])
    l0b_uw_g = pl.make_tile_group(type=l0b_uw_t, addrs=[L0B_0, L0B_1], mutex_ids=[18, 19])
    l0a_attn_g = pl.make_tile_group(type=l0a_attn_t, addrs=[L0A_0, L0A_1], mutex_ids=[16, 17])
    u_l0c_g = pl.make_tile_group(type=acc_uw_t, addrs=[L0C_U], mutex_ids=[28])
    w_l0c_g = pl.make_tile_group(type=acc_uw_t, addrs=[L0C_W], mutex_ids=[29])

    # ---- Scan phase kernel-scope SHARED UB tiles (cube writes via move,
    #      vector reads).  Fresh ids 30/31 (chunk-prep uses 0..29). ----
    scan_vp_ub_t = pl.TileType(shape=[C, VT], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1], pad=pl.TilePad.zero)
    scan_delta_ub_t = pl.TileType(shape=[C, VT], dtype=pl.DT_FP32,
                                target_memory=pl.MemorySpace.Vec,
                                valid_shape=[-1, -1], pad=pl.TilePad.zero)
    scan_vp_ub_g = pl.make_tile_group(type=scan_vp_ub_t, addrs=[SCAN_VP_UB], mutex_ids=[30])
    scan_delta_ub_g = pl.make_tile_group(type=scan_delta_ub_t, addrs=[SCAN_DELTA_UB], mutex_ids=[31])

    # =====================================================================
    #   Cube section (gram): reads GM ws, writes decay / kk_gram / qk_gram scratch
    # =====================================================================
    with pl.section_cube():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        B = q.shape[0]
        T = q.shape[1]
        H = q.shape[2]
        K = q.shape[3]
        N = (T + C - 1) // C
        HN = H * N
        total = B * HN
        k_tiles = (K + TILE_COL - 1) // TILE_COL
        per_core = (total + num_cores - 1) // num_cores
        lo = per_core * core_id
        hi = pl.min(lo + per_core, total)

        # bootstrap: give the Vector side permission to write the first tid's ws
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        for tid in pl.range(lo, hi, 1):
            # wait until the Vector side produced this tid's GM workspace
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=1,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # ---- decay[1,C] = g_row[1,C] @ Upper[C,C] (cumsum; M=1/N=C) ----
            gr_l1 = grow_l1_g.current()
            pl.set_validshape(gr_l1, [1, C])
            pl.load(gr_l1, gf_ws, [tid, 0, 0], order=[2, 1])   # g column -> row
            up_l1 = upper_l1_g.current()
            pl.set_validshape(up_l1, [C, C])
            pl.load(up_l1, tri_ones_ws, [0, 0], order=[1, 0])         # lower-tri -> upper
            la = l0a_dec_g.next()
            pl.move(la, gr_l1, offset=[0, 0])
            pl.set_validshape(la, [MMIN, C])
            lb = l0b_dec_g.next()
            pl.move(lb, up_l1, offset=[0, 0])
            pl.set_validshape(lb, [C, C])
            ad = acc_dec_g.current()
            pl.set_validshape(ad, [1, C])
            pl.matmul(ad, la, lb)
            pl.set_validshape(ad, [1, C])
            pl.store(dec_ws, ad, [tid, 0])

            # ---- kk_gram[C,C] = k_beta[C,K] @ k^T[K,C] (K-contraction split) ----
            acc_kk = acc_p1_g.current()
            for kkc in pl.range(0, k_tiles):
                kco = kkc * TILE_COL
                kcc = pl.min(TILE_COL, K - kco)
                kb_l1 = kb_l1_g.current()
                pl.set_validshape(kb_l1, [C, kcc])
                pl.load(kb_l1, kb_ws, [tid, 0, kco])
                kf_l1 = kf_l1_g.current()
                pl.set_validshape(kf_l1, [kcc, C])
                pl.load(kf_l1, kf_ws, [tid, 0, kco], order=[2, 1])   # k^T block
                la1 = l0a_g.next()
                pl.move(la1, kb_l1, offset=[0, 0])
                pl.set_validshape(la1, [C, kcc])
                lb1 = l0b_g.next()
                pl.move(lb1, kf_l1, offset=[0, 0])
                pl.set_validshape(lb1, [kcc, C])
                pl.set_validshape(acc_kk, [C, C])
                if kkc == 0:
                    pl.matmul(acc_kk, la1, lb1)
                else:
                    pl.matmul_acc(acc_kk, acc_kk, la1, lb1)
            pl.set_validshape(acc_kk, [C, C])
            pl.store(kk_gram_ws, acc_kk, [tid, 0, 0])

            # ---- qk_gram[C,C] = q_s[C,K] @ k^T[K,C] (K-contraction split) ----
            acc_qk = acc_p2_g.current()
            for kkc in pl.range(0, k_tiles):
                kco = kkc * TILE_COL
                kcc = pl.min(TILE_COL, K - kco)
                qs_l1 = qs_l1_g.current()
                pl.set_validshape(qs_l1, [C, kcc])
                pl.load(qs_l1, qs_ws, [tid, 0, kco])
                kf_l1b = kf_l1_g.current()
                pl.set_validshape(kf_l1b, [kcc, C])
                pl.load(kf_l1b, kf_ws, [tid, 0, kco], order=[2, 1])
                la2 = l0a_g.next()
                pl.move(la2, qs_l1, offset=[0, 0])
                pl.set_validshape(la2, [C, kcc])
                lb2 = l0b_g.next()
                pl.move(lb2, kf_l1b, offset=[0, 0])
                pl.set_validshape(lb2, [kcc, C])
                pl.set_validshape(acc_qk, [C, C])
                if kkc == 0:
                    pl.matmul(acc_qk, la2, lb2)
                else:
                    pl.matmul_acc(acc_qk, acc_qk, la2, lb2)
            pl.set_validshape(acc_qk, [C, C])
            pl.store(qk_gram_ws, acc_qk, [tid, 0, 0])

            # cube produced tid's decay/kk_gram/qk_gram scratch -> vector may read (e2)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # cube done reading tid's ws -> vector may overwrite for tid+1 (e0)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # ============ u/w (Cube): u = wy_t @ v_beta ; w = wy_t @ k_beta_decay ============
            # wait until vector produced this tid's WY outputs + v_b (e3)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            V = v.shape[3]
            # load attn (LEFT, [C,C]) once; reused for both matmuls
            at_l1 = attn_l1_g.current()
            pl.set_validshape(at_l1, [C, C])
            pl.load(at_l1, wy_t_ws, [tid, 0, 0])

            # ---- u[C,V] = attn[C,C] @ v_b[C,V] (N-split over V, TILE_COL blocks) ----
            v_tiles_uw = (V + TILE_COL - 1) // TILE_COL
            for nbu in pl.range(0, v_tiles_uw):
                nou = nbu * TILE_COL
                ncu = pl.min(TILE_COL, V - nou)
                vb_l1 = vb_l1_g.current()
                pl.set_validshape(vb_l1, [C, TILE_COL])
                pl.load(vb_l1, vb_ws, [tid, 0, nou])
                la_u = l0a_attn_g.next()
                pl.move(la_u, at_l1, offset=[0, 0])
                pl.set_validshape(la_u, [C, C])
                lb_u = l0b_uw_g.next()
                pl.move(lb_u, vb_l1, offset=[0, 0])
                pl.set_validshape(lb_u, [C, TILE_COL])
                u_acc = u_l0c_g.current()
                pl.set_validshape(u_acc, [C, TILE_COL])
                pl.matmul(u_acc, la_u, lb_u)
                pl.set_validshape(u_acc, [C, ncu])
                pl.store(u_ws, u_acc, [tid, 0, nou])

            # ---- w[C,K] = attn[C,C] @ kbeta_de[C,K] (N-split over K, TILE_COL blocks) ----
            for nbw in pl.range(0, k_tiles):
                now = nbw * TILE_COL
                ncw = pl.min(TILE_COL, K - now)
                kbde_l1 = kbde_l1_g.current()
                pl.set_validshape(kbde_l1, [C, TILE_COL])
                pl.load(kbde_l1, k_beta_decay_ws, [tid, 0, now])
                la_w = l0a_attn_g.next()
                pl.move(la_w, at_l1, offset=[0, 0])
                pl.set_validshape(la_w, [C, C])
                lb_w = l0b_uw_g.next()
                pl.move(lb_w, kbde_l1, offset=[0, 0])
                pl.set_validshape(lb_w, [C, TILE_COL])
                w_acc = w_l0c_g.current()
                pl.set_validshape(w_acc, [C, TILE_COL])
                pl.matmul(w_acc, la_w, lb_w)
                pl.set_validshape(w_acc, [C, ncw])
                pl.store(w_ws, w_acc, [tid, 0, now])

    # =====================================================================
    #   Vector section (preprocess -> workspaces + lower-tri ones; then WY construction)
    # =====================================================================

        # =================================================================
        #   SCAN PHASE — CUBE side of the fused inter-chunk scan.
        #   Barrier: make all chunk-prep GM writes grid-visible before scan reads.
        # =================================================================
        pl.system.sync_all(mode=pl.SyncAllMode.HARD, core_type=pl.SyncCoreType.MIX)

        # ---- scan cube tile groups (cube-only; reuse addrs/ids, barrier-safe) ----
        s_state_l1_t = pl.TileType(shape=[2 * KT, VT], dtype=pl.DT_FP32,
                                   target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
        s_wq_l1_t = pl.TileType(shape=[C, 2 * KT], dtype=pl.DT_FP32,
                                target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
        s_kft_l1_t = pl.TileType(shape=[2 * KT, C], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
        s_a_l1_t = pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
        s_col_l1_t = pl.TileType(shape=[C, VT], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
        s_state_l1_g = pl.make_tile_group(type=s_state_l1_t, addrs=[SCAN_STATE_L1], mutex_ids=[0])
        s_w_l1_g = pl.make_tile_group(type=s_wq_l1_t, addrs=[SCAN_W_L1], mutex_ids=[1])
        s_qde_l1_g = pl.make_tile_group(type=s_wq_l1_t, addrs=[SCAN_QDECAY_L1], mutex_ids=[2])
        s_kft_l1_g = pl.make_tile_group(type=s_kft_l1_t, addrs=[SCAN_KT_L1], mutex_ids=[3])
        s_a_l1_g = pl.make_tile_group(type=s_a_l1_t, addrs=[SCAN_ATTN_L1], mutex_ids=[4])
        s_vnew_l1_g = pl.make_tile_group(type=s_col_l1_t, addrs=[SCAN_VNEW_L1], mutex_ids=[5])
        s_vs_l1_g = pl.make_tile_group(type=s_col_l1_t, addrs=[SCAN_VSCALED_L1], mutex_ids=[6])

        # L0A variants (aliased at the two double-buffer slots): [C,KT] for w/qde,
        # [KT,C] for k^T, [C,C] for A.  L0B variants: [KT,VT] for S, [C,VT] for v.
        s_l0a_wq_t = pl.TileType(shape=[C, KT], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1])
        s_l0a_kft_t = pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                  target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1])
        s_l0a_a_t = pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1])
        s_l0b_s_t = pl.TileType(shape=[KT, VT], dtype=pl.DT_FP32,
                                target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1])
        s_l0b_v_t = pl.TileType(shape=[C, VT], dtype=pl.DT_FP32,
                                target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1])
        s_l0a_wq_g = pl.make_tile_group(type=s_l0a_wq_t, addrs=[SCAN_L0A_0, SCAN_L0A_1], mutex_ids=[7, 8])
        s_l0a_kft_g = pl.make_tile_group(type=s_l0a_kft_t, addrs=[SCAN_L0A_0, SCAN_L0A_1], mutex_ids=[7, 8])
        s_l0a_a_g = pl.make_tile_group(type=s_l0a_a_t, addrs=[SCAN_L0A_0, SCAN_L0A_1], mutex_ids=[7, 8])
        s_l0b_s_g = pl.make_tile_group(type=s_l0b_s_t, addrs=[SCAN_L0B_0, SCAN_L0B_1], mutex_ids=[9, 10])
        s_l0b_v_g = pl.make_tile_group(type=s_l0b_v_t, addrs=[SCAN_L0B_0, SCAN_L0B_1], mutex_ids=[9, 10])

        s_vp_l0c_t = pl.TileType(shape=[C, VT], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1])
        s_delta_l0c_t = pl.TileType(shape=[C, VT], dtype=pl.DT_FP32,
                                    target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1])
        s_vp_l0c_g = pl.make_tile_group(type=s_vp_l0c_t, addrs=[SCAN_VP_L0C], mutex_ids=[11])
        s_o_l0c_g = pl.make_tile_group(type=s_vp_l0c_t, addrs=[SCAN_O_L0C], mutex_ids=[12])
        s_delta_l0c_g = pl.make_tile_group(type=s_delta_l0c_t, addrs=[SCAN_DELTA_L0C], mutex_ids=[13])

        # the 2nd barrier is GONE -- kft_gm is now produced in chunk-prep
        # and published by the single chunk-prep -> scan barrier above.
        c2 = pl.get_block_idx() // pl.get_subblock_num()
        nc2 = pl.get_block_num()
        Bd = q.shape[0]
        Td = q.shape[1]
        Hd = q.shape[2]
        Kd = q.shape[3]
        Vd = v.shape[3]
        Nc = (Td + C - 1) // C
        BHd = Bd * Hd
        ktiles = (Kd + KT - 1) // KT
        nkob = (Kd + C - 1) // C
        vtiles = (Vd + VT - 1) // VT
        # ---- the scan work item is the FLAT pair (bh, vv). ----
        # The vv passes are independent: S is [K,V] and every scan step touches
        # only the current V column block (v_prime = w @ S reads all K rows but
        # only this V block; delta = k^T @ v_scaled writes only this V block; the
        # state update is elementwise in that block).  So distributing
        # BHd*vtiles items instead of BHd items multiplies scan parallelism
        # by vtiles.  Proportional split: core c2 owns [n_scan_items*c2//nc2, n_scan_items*(c2+1)//nc2).
        n_scan_items = BHd * vtiles
        lo2 = (n_scan_items * c2) // nc2
        hi2 = (n_scan_items * (c2 + 1)) // nc2

        for item2 in pl.range(lo2, hi2, 1):
            bh = item2 // vtiles
            vv = item2 - bh * vtiles
            voff = vv * VT
            vcur = VT   # full-width tiles; V padded to VT multiple in wrapper
            # cube bootstrap for this (bh,vv) pass: RELEASE_VN (e5)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=5,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            for ni in pl.range(0, Nc, 1):
                lin = bh * Nc + ni
                # hoist all tile handles once per iteration (auto_mutex SSA)
                vp_ub = scan_vp_ub_g.current()
                d_ub = scan_delta_ub_g.current()
                vp_acc = s_vp_l0c_g.current()
                o_acc = s_o_l0c_g.current()
                d_acc = s_delta_l0c_g.current()
                w_l1 = s_w_l1_g.current()
                pl.set_validshape(w_l1, [C, Kd])
                pl.load(w_l1, w_ws, [lin, 0, 0])
                q_decay_l1 = s_qde_l1_g.current()
                pl.set_validshape(q_decay_l1, [C, Kd])
                pl.load(q_decay_l1, q_decay_ws, [lin, 0, 0])
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0,
                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                st_l1 = s_state_l1_g.current()
                pl.set_validshape(st_l1, [Kd, vcur])
                pl.load(st_l1, state_gm, [bh, 0, voff])
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1,
                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                # ---- v_prime = w @ S  (K-phase) ----
                for kk in pl.range(0, ktiles):
                    koff = kk * KT
                    kcur = pl.min(KT, Kd - koff)
                    la_wq = s_l0a_wq_g.next()
                    pl.set_validshape(la_wq, [C, kcur])
                    pl.move(la_wq, w_l1, offset=[0, koff])
                    lb_s = s_l0b_s_g.next()
                    pl.set_validshape(lb_s, [kcur, vcur])
                    pl.move(lb_s, st_l1, offset=[koff, 0])
                    pl.set_validshape(vp_acc, [C, vcur])
                    if kk == 0:
                        pl.matmul(vp_acc, la_wq, lb_s)
                    else:
                        pl.matmul_acc(vp_acc, vp_acc, la_wq, lb_s)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3,
                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                # split v_prime across both AIVs.  dst UB validshape is the
                # PER-AIV row count and the ACC validshape is aligned_M (== C,
                # C is even) -- the convention used by pro_ops/fa/test_fa_with_mask.py.
                pl.set_validshape(vp_ub, [HC, vcur])
                pl.set_validshape(vp_acc, [C, vcur])
                pl.move(vp_ub, vp_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2,
                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                # ---- o_inter = q_de @ S  (K-phase) into o_l0c ----
                for kk in pl.range(0, ktiles):
                    koff = kk * KT
                    kcur = pl.min(KT, Kd - koff)
                    la_wq = s_l0a_wq_g.next()
                    pl.set_validshape(la_wq, [C, kcur])
                    pl.move(la_wq, q_decay_l1, offset=[0, koff])
                    lb_s = s_l0b_s_g.next()
                    pl.set_validshape(lb_s, [kcur, vcur])
                    pl.move(lb_s, st_l1, offset=[koff, 0])
                    pl.set_validshape(o_acc, [C, vcur])
                    if kk == 0:
                        pl.matmul(o_acc, la_wq, lb_s)
                    else:
                        pl.matmul_acc(o_acc, o_acc, la_wq, lb_s)
                attn_l1 = s_a_l1_g.current()
                pl.set_validshape(attn_l1, [C, C])
                pl.load(attn_l1, attn_intra_ws, [lin, 0, 0])
                kt_l1 = s_kft_l1_g.current()
                pl.set_validshape(kt_l1, [Kd, C])
                pl.load(kt_l1, kft_gm, [lin, 0, 0])   # k^T normal-layout LEFT operand
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=4,
                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                vnew_l1 = s_vnew_l1_g.current()
                pl.set_validshape(vnew_l1, [C, vcur])
                pl.load(vnew_l1, vnew_gm, [bh, 0, voff])
                vs_l1 = s_vs_l1_g.current()
                pl.set_validshape(vs_l1, [C, vcur])
                pl.load(vs_l1, vscaled_gm, [bh, 0, voff])
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=5,
                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                # ---- o += A @ v_new ----
                la_a = s_l0a_a_g.next()
                pl.set_validshape(la_a, [C, C])
                pl.move(la_a, attn_l1, offset=[0, 0])
                lb_v = s_l0b_v_g.next()
                pl.set_validshape(lb_v, [C, vcur])
                pl.move(lb_v, vnew_l1, offset=[0, 0])
                pl.set_validshape(o_acc, [C, vcur])
                pl.matmul_acc(o_acc, o_acc, la_a, lb_v)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3,
                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                # split o across both AIVs (same convention as v_prime).
                pl.set_validshape(vp_ub, [HC, vcur])
                pl.set_validshape(o_acc, [C, vcur])
                pl.move(vp_ub, o_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2,
                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                # ---- delta = k^T @ v_scaled  (per 64-row K output-block) ----
                for kob in pl.range(0, nkob):
                    koff = kob * C
                    kcur = pl.min(C, Kd - koff)
                    la_kft = s_l0a_kft_g.next()
                    pl.set_validshape(la_kft, [kcur, C])
                    pl.move(la_kft, kt_l1, offset=[koff, 0])
                    lb_v = s_l0b_v_g.next()
                    pl.set_validshape(lb_v, [C, vcur])
                    pl.move(lb_v, vs_l1, offset=[0, 0])
                    pl.set_validshape(d_acc, [kcur, vcur])
                    pl.matmul(d_acc, la_kft, lb_v)
                    pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=7,
                                              sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                    # split delta across both AIVs.  kcur may in principle be
                    # odd, so aligned_M = ceil(kcur/2)*2 and each AIV gets ceil(kcur/2)
                    # rows (Vec1's tail rows are simply not consumed, see krows).
                    kh_c = (kcur + 1) // 2
                    pl.set_validshape(d_ub, [kh_c, vcur])
                    pl.set_validshape(d_acc, [kh_c * 2, vcur])
                    pl.move(d_ub, d_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=6,
                                             sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # cube drains for this (bh,vv) pass
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=7,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        B = q.shape[0]
        T = q.shape[1]
        H = q.shape[2]
        K = q.shape[3]
        N = (T + C - 1) // C
        HN = H * N
        total = B * HN
        k_tiles = (K + TILE_COL - 1) // TILE_COL
        per_core = (total + num_cores - 1) // num_cores
        lo = per_core * core_id
        hi = pl.min(lo + per_core, total)

        for tid in pl.range(lo, hi, 1):
            # wait for permission to write this tid's ws (e0 from cube)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # ---- chunk-prep preprocess split between the two AIV sub-blocks. ----
            # This is a SUB-TASK split, not a row split: the index arithmetic is
            # pure scalar work and is done by both sub-blocks; sub-block 0 then
            # produces gf_ws / qs_ws / tri_ones_ws and sub-block 1 produces
            # beta_act -> kf_ws / kb_ws / vb_ws.  beta_act feeds only sub-block
            # 1's outputs, so nothing is recomputed.  Both sub-blocks still set
            # e1 exactly once, and an INTRA_BLOCK wait needs BOTH of them, so the
            # cube still sees a complete workspace before it reads any of it.
            b = tid // HN
            rem = tid - b * HN
            h = rem // N
            n = rem - h * N
            t_base = n * C
            valid_c = pl.min(C, T - t_base)


            if pl.get_subblock_idx() == 0:
                # ---- g_f32 = g (transpose + pad, cast-free) ----
                gt = g_g.current()
                pl.set_validshape(gt, [valid_c, 1])
                pl.load(gt, g, [b, t_base, h, 0], order=[1, 3])
                if valid_c < C:
                    _zero_rows_col_vf(gt, valid_c, C)
                pl.set_validshape(gt, [C, 1])
                pl.store(gf_ws, gt, [tid, 0, 0])

                # ---- q_s = q * scale ----
                for cb in pl.range(0, k_tiles):
                    col_off = cb * TILE_COL
                    vcols = pl.min(TILE_COL, K - col_off)
                    qbf = in_bf.next()
                    pl.set_validshape(qbf, [valid_c, vcols])
                    pl.load(qbf, q, [b, t_base, h, col_off], order=[1, 3])
                    qf = f32in.next()
                    pl.set_validshape(qf, [valid_c, vcols])
                    pl.cast(qf, qbf, mode=pl.RoundMode.CAST_NONE)
                    if valid_c < C:
                        _zero_rows_vf(qf, valid_c, C, vcols)
                    pl.set_validshape(qf, [C, vcols])
                    qs = out_grp.next()
                    pl.set_validshape(qs, [C, vcols])
                    _scale_vf(qf, qs, C, vcols, scale)
                    pl.store(qs_ws, qs, [tid, 0, col_off])

                # ---- LTones (chunk-independent lower-tri ones) -> GM ws ----
                ltub = lt_ub_g.current()
                pl.set_validshape(ltub, [C, C])
                _build_lower_tri_ones_vf(ltub)
                pl.set_validshape(ltub, [C, C])
                pl.store(tri_ones_ws, ltub, [0, 0])

                # ---- k^T for THIS tid, folded into chunk-prep preprocess. ----
                # Earlier revisions materialised k^T in a separate scan vector pre-pass
                # that re-read kf_ws from GM and needed its own grid-wide
                # sync_all to publish kft_gm.  Both are deleted: the transpose
                # is the same square 64x64 pl.transpose, but fed from k here.
                # Pad rows (>= valid_c) are zeroed BEFORE the transpose, so the
                # pad COLUMNS of k^T are 0 -- byte-identical to what the pre-pass
                # produced from the zero-padded kf_ws, which is what makes the
                # scan delta = k^T @ v_scaled contraction ignore pad rows.
                # Sub-block 0 owns it (sub-block 1 already owns k/k_beta/v_b);
                # it is published by the SAME chunk-prep -> scan barrier that
                # already publishes every other chunk-prep workspace.
                nkb1 = (K + LANES - 1) // LANES
                for kb1 in pl.range(0, nkb1):
                    koff1 = kb1 * LANES
                    kcols1 = pl.min(LANES, K - koff1)
                    ktbf = kt_bf_g.current()
                    pl.set_validshape(ktbf, [valid_c, kcols1])
                    pl.load(ktbf, k, [b, t_base, h, koff1], order=[1, 3])
                    ktn = kt_f32_g.current()
                    pl.set_validshape(ktn, [valid_c, kcols1])
                    pl.cast(ktn, ktbf, mode=pl.RoundMode.CAST_NONE)
                    if valid_c < C:
                        # [C,LANES] fp32 has row pitch LANES == BCOLW, so the
                        # column-staging zeroer is the pitch-correct one here.
                        _zero_rows_col_vf(ktn, valid_c, C)
                    pl.set_validshape(ktn, [C, LANES])
                    ktt = kt_tr_g.current()
                    pl.set_validshape(ktt, [LANES, C])
                    pl.transpose(ktt, ktn)
                    pl.set_validshape(ktt, [kcols1, C])
                    pl.store(kft_gm, ktt, [tid, koff1, 0])

            if pl.get_subblock_idx() == 1:
                # ---- beta_act = sigmoid(logit) (column layout) ----
                bbf = beta_bf_g.current()
                pl.set_validshape(bbf, [valid_c, 1])
                pl.load(bbf, beta, [b, t_base, h, 0], order=[1, 3])
                bl = bl_g.current()
                pl.set_validshape(bl, [valid_c, 1])
                pl.cast(bl, bbf, mode=pl.RoundMode.CAST_NONE)
                if valid_c < C:
                    _zero_rows_col_vf(bl, valid_c, C)
                pl.set_validshape(bl, [C, 1])
                beta_act = ba_g.current()
                pl.set_validshape(beta_act, [C, 1])
                _sigmoid_col_vf(bl, beta_act, C)

                # ---- k_f32 = k ; k_beta = k * beta_act ----
                for cb in pl.range(0, k_tiles):
                    col_off = cb * TILE_COL
                    vcols = pl.min(TILE_COL, K - col_off)
                    kbf = in_bf.next()
                    pl.set_validshape(kbf, [valid_c, vcols])
                    pl.load(kbf, k, [b, t_base, h, col_off], order=[1, 3])
                    kf = f32in.next()
                    pl.set_validshape(kf, [valid_c, vcols])
                    pl.cast(kf, kbf, mode=pl.RoundMode.CAST_NONE)
                    if valid_c < C:
                        _zero_rows_vf(kf, valid_c, C, vcols)
                    pl.set_validshape(kf, [C, vcols])
                    pl.store(kf_ws, kf, [tid, 0, col_off])
                    kb = out_grp.next()
                    pl.set_validshape(kb, [C, vcols])
                    _mul_beta_vf(kf, beta_act, kb, C, vcols)
                    pl.store(kb_ws, kb, [tid, 0, col_off])

                # ---- v_b = v * beta_act (mirror of k_beta; for u/w u=attn@v_b) ----
                V = v.shape[3]
                v_tiles = (V + TILE_COL - 1) // TILE_COL
                for cb in pl.range(0, v_tiles):
                    col_off = cb * TILE_COL
                    vcols = pl.min(TILE_COL, V - col_off)
                    vbf = in_bf.next()
                    pl.set_validshape(vbf, [valid_c, vcols])
                    pl.load(vbf, v, [b, t_base, h, col_off], order=[1, 3])
                    vff = f32in.next()
                    pl.set_validshape(vff, [valid_c, vcols])
                    pl.cast(vff, vbf, mode=pl.RoundMode.CAST_NONE)
                    if valid_c < C:
                        _zero_rows_vf(vff, valid_c, C, vcols)
                    pl.set_validshape(vff, [C, vcols])
                    vb = out_grp.next()
                    # pad handling: zero the WHOLE [C,TILE_COL] tile first so pad columns
                    # [vcols:TILE_COL] are 0; the valid cols are overwritten next.
                    pl.set_validshape(vb, [C, TILE_COL])
                    _zero_rows_vf(vb, 0, C, TILE_COL)
                    pl.set_validshape(vb, [C, vcols])
                    _mul_beta_vf(vff, beta_act, vb, C, vcols)
                    pl.set_validshape(vb, [C, TILE_COL])
                    pl.store(vb_ws, vb, [tid, 0, col_off])

            # signal cube that tid's ws is ready (both AIVs must set — INTRA_BLOCK)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # wait until cube produced tid's decay/kk_gram/qk_gram scratch (e2)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=2,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # ---- chunk-prep WY split between the two AIV sub-blocks. ----
            # forward substitution is C-1 SEQUENTIAL row steps and dominates
            # WY, and attn is independent of A and of the q_de/kbeta_de/
            # dfac row scaling.  So sub-block 0 owns attn and sub-block 1
            # owns everything else.  Both need decay and L_mask; those are cheap
            # and are rebuilt in each AIV's OWN UB, but only sub-block 1 STORES
            # decay_exp_ws / decay_mask_ws, so no GM byte is written by both.
            if pl.get_subblock_idx() == 0:
                dec = dec_ub_g.current()
                pl.set_validshape(dec, [1, C])
                pl.load(dec, dec_ws, [tid, 0])

                # ---- L_mask (reuse LTones tile) ----
                lm = lt_ub_g.current()
                pl.set_validshape(lm, [C, C])
                _decay_mask_vf(dec, lm)

                # ---- attn = fwdsubst(-tril_strict(kk_gram*L_mask)) + I ----
                kk_tile = kk_gram_ub_g.current()
                pl.set_validshape(kk_tile, [C, C])
                pl.load(kk_tile, kk_gram_ws, [tid, 0, 0])
                _wy_lower_init_vf(kk_tile, lm)
                _wy_forward_substitution_vf(kk_tile)
                pl.set_validshape(kk_tile, [C, C])
                pl.store(wy_t_ws, kk_tile, [tid, 0, 0])

            # NOTE: distinct Python names from the sub-block-0 branch above.
            # auto_mutex emits a `<pyname>__mutexid_N` per traced block, so
            # reusing `dec`/`lm` across two SIBLING runtime branches makes the
            # merge reference a variable declared only in the other branch
            # (bisheng: "use of undeclared identifier 'dec__mutexid_0'").
            if pl.get_subblock_idx() == 1:
                K = q.shape[3]
                k_tiles = (K + TILE_COL - 1) // TILE_COL

                dec_b1 = dec_ub_g.current()
                pl.set_validshape(dec_b1, [1, C])
                pl.load(dec_b1, dec_ws, [tid, 0])

                # ---- decay_exp = exp(decay) ----
                dexp = dexp_ub_g.current()
                pl.set_validshape(dexp, [1, C])
                _decay_exp_vf(dec_b1, dexp)
                pl.set_validshape(dexp, [1, C])
                pl.store(decay_exp_ws, dexp, [tid, 0])

                # ---- L_mask (reuse LTones tile) ----
                lm_b1 = lt_ub_g.current()
                pl.set_validshape(lm_b1, [C, C])
                _decay_mask_vf(dec_b1, lm_b1)
                pl.set_validshape(lm_b1, [C, C])
                pl.store(decay_mask_ws, lm_b1, [tid, 0, 0])

                # ---- A = tril_incl_diag(qk_gram*L_mask) ----
                qk_tile = qk_gram_ub_g.current()
                pl.set_validshape(qk_tile, [C, C])
                pl.load(qk_tile, qk_gram_ws, [tid, 0, 0])
                _intra_attn_build_vf(qk_tile, lm_b1)
                pl.set_validshape(qk_tile, [C, C])
                pl.store(attn_intra_ws, qk_tile, [tid, 0, 0])

                # ---- q_de / kbeta_de (row-scale by decay_exp) + dfac ----
                for cb in pl.range(0, k_tiles):
                    col_off = cb * TILE_COL
                    vcols = pl.min(TILE_COL, K - col_off)
                    qd = out_grp.next()
                    pl.set_validshape(qd, [C, vcols])
                    pl.load(qd, qs_ws, [tid, 0, col_off])
                    _rowscale_vf(qd, dexp, qd, vcols)
                    pl.set_validshape(qd, [C, vcols])
                    pl.store(q_decay_ws, qd, [tid, 0, col_off])

                    kd = out_grp.next()
                    # pad handling: zero the WHOLE [C,TILE_COL] tile so pad cols [K:TILE_COL]=0
                    # for the u/w full-width w matmul; valid [C,K] loaded/scaled next.
                    pl.set_validshape(kd, [C, TILE_COL])
                    _zero_rows_vf(kd, 0, C, TILE_COL)
                    pl.set_validshape(kd, [C, vcols])
                    pl.load(kd, kb_ws, [tid, 0, col_off])
                    _rowscale_vf(kd, dexp, kd, vcols)
                    pl.set_validshape(kd, [C, TILE_COL])
                    pl.store(k_beta_decay_ws, kd, [tid, 0, col_off])

                    df = out_grp.next()
                    pl.set_validshape(df, [C, vcols])
                    _decay_tail_vf(dec_b1, df, vcols)
                    pl.set_validshape(df, [C, vcols])
                    pl.store(decay_tail_ws, df, [tid, 0, col_off])

            # signal cube that tid's WY outputs (attn, kbeta_de) + v_b are ready (e3)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        # drain the final e0 that cube set after its last read
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0,
                                  sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        # =================================================================
        #   SCAN PHASE — VECTOR side of the fused inter-chunk scan.
        # =================================================================
        pl.system.sync_all(mode=pl.SyncAllMode.HARD, core_type=pl.SyncCoreType.MIX)

        s_state_ub_t = pl.TileType(shape=[C, VT], dtype=pl.DT_FP32,
                                   target_memory=pl.MemorySpace.Vec,
                                   valid_shape=[-1, -1], pad=pl.TilePad.zero)
        s_col_ub_t = pl.TileType(shape=[C, VT], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec,
                                 valid_shape=[-1, -1], pad=pl.TilePad.zero)
        s_obf_ub_t = pl.TileType(shape=[C, VT], dtype=pl.DT_BF16,
                                 target_memory=pl.MemorySpace.Vec,
                                 valid_shape=[-1, -1], pad=pl.TilePad.zero)
        s_dfac_ub_t = pl.TileType(shape=[C, 8], dtype=pl.DT_FP32,
                                  target_memory=pl.MemorySpace.Vec,
                                  valid_shape=[-1, -1], pad=pl.TilePad.zero)
        s_dexp_ub_t = pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                                  target_memory=pl.MemorySpace.Vec,
                                  valid_shape=[-1, -1], pad=pl.TilePad.zero)
        s_state_ub_g = pl.make_tile_group(type=s_state_ub_t, addrs=[SCAN_STATE_UB], mutex_ids=[14])
        s_uv_ub_g = pl.make_tile_group(type=s_col_ub_t, addrs=[SCAN_U_UB], mutex_ids=[15])
        s_vs_ub_g = pl.make_tile_group(type=s_col_ub_t, addrs=[SCAN_VSCALED_UB], mutex_ids=[16])
        s_obf_ub_g = pl.make_tile_group(type=s_obf_ub_t, addrs=[SCAN_OBF_UB], mutex_ids=[17])
        s_dfac_ub_g = pl.make_tile_group(type=s_dfac_ub_t, addrs=[SCAN_DECAYTAIL_UB], mutex_ids=[18])
        s_dexp_ub_g = pl.make_tile_group(type=s_dexp_ub_t, addrs=[SCAN_DEXP_UB], mutex_ids=[19])
        c2 = pl.get_block_idx() // pl.get_subblock_num()
        nc2 = pl.get_block_num()
        sub2 = pl.get_subblock_idx()
        Bd = q.shape[0]
        Td = q.shape[1]
        Hd = q.shape[2]
        Kd = q.shape[3]
        Vd = v.shape[3]
        Nc = (Td + C - 1) // C
        BHd = Bd * Hd
        ktiles = (Kd + KT - 1) // KT
        nkob = (Kd + C - 1) // C
        vtiles = (Vd + VT - 1) // VT
        # ---- the scan work item is the FLAT pair (bh, vv). ----
        # The vv passes are independent: S is [K,V] and every scan step touches
        # only the current V column block (v_prime = w @ S reads all K rows but
        # only this V block; delta = k^T @ v_scaled writes only this V block; the
        # state update is elementwise in that block).  So distributing
        # BHd*vtiles items instead of BHd items multiplies scan parallelism
        # by vtiles.  Proportional split: core c2 owns [n_scan_items*c2//nc2, n_scan_items*(c2+1)//nc2).
        n_scan_items = BHd * vtiles
        lo2 = (n_scan_items * c2) // nc2
        hi2 = (n_scan_items * (c2 + 1)) // nc2

        # the k^T pre-pass and its grid-wide barrier are GONE --
        # k^T is produced in preprocess (sub-block 0) straight from k.
        for item2 in pl.range(lo2, hi2, 1):
            bh = item2 // vtiles
            vv = item2 - bh * vtiles
            voff = vv * VT
            vcur = VT   # full-width tiles; V padded to VT multiple in wrapper
            # ---- BOTH AIV sub-blocks now compute the scan. ----
            # The cube moves every L0C result with AccToVecMode.DualModeSplitM, so
            # sub-block 0 holds rows [0,HC) and sub-block 1 rows [HC,C) of each
            # [C,*] result -- each in ITS OWN UB, starting at tile row 0 (UB is
            # per-AIV).  So each AIV runs the payload on HC rows and addresses GM
            # at the global row base rb = sub2*HC.  vnew_gm / vscaled_gm /
            # state_gm / o are all row-disjoint between the two sub-blocks.
            # EVERY set_cross_core / wait_cross_core below is executed by BOTH
            # sub-blocks exactly as many times as in r2a -- only payload moved.
            rb = sub2 * HC
            # bootstrap this (bh,vv) pass: zero S in GM, READY_S/RELEASE_VP/RELEASE_D
            st_ub_b = s_state_ub_g.current()
            for kob in pl.range(0, nkob):
                koff = kob * C
                kcur = pl.min(C, Kd - koff)
                kh = (kcur + 1) // 2            # per-AIV rows (DualModeSplitM)
                kbase = sub2 * kh
                krows = pl.min(kh, kcur - kbase)    # <=0 only if kcur==1
                if krows > 0:
                    pl.set_validshape(st_ub_b, [krows, vcur])
                    _zero_tile_vf(st_ub_b, krows, vcur)
                    pl.set_validshape(st_ub_b, [krows, vcur])
                    pl.store(state_gm, st_ub_b, [bh, koff + kbase, voff])
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=0,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=7,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            for ni in pl.range(0, Nc, 1):
                lin = bh * Nc + ni
                t_base = ni * C
                valid_c = pl.min(C, Td - t_base)
                # hoist all tile handles once per iteration (auto_mutex SSA)
                vp_ub = scan_vp_ub_g.current()
                d_ub = scan_delta_ub_g.current()
                uv_ub = s_uv_ub_g.current()
                vs_ub = s_vs_ub_g.current()
                dfac_ub = s_dfac_ub_g.current()
                dexp_ub = s_dexp_ub_g.current()
                obf_ub = s_obf_ub_g.current()
                st_ub = s_state_ub_g.current()
                # each AIV loads only ITS HC rows of u and of dfac
                pl.set_validshape(uv_ub, [HC, vcur])
                pl.load(uv_ub, u_ws, [lin, rb, voff])
                pl.set_validshape(dfac_ub, [HC, 8])
                pl.load(dfac_ub, decay_tail_ws, [lin, rb, 0])
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=2,
                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                _scan_vnew_vf(vp_ub, uv_ub, dfac_ub, vs_ub, HC, vcur)
                pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3,
                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=5,
                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.set_validshape(uv_ub, [HC, vcur])
                pl.store(vnew_gm, uv_ub, [bh, rb, voff])
                pl.set_validshape(vs_ub, [HC, vcur])
                pl.store(vscaled_gm, vs_ub, [bh, rb, voff])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=4,
                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                # decay_exp is a [1,C] row of per-chunk scalars: both AIVs need it
                pl.set_validshape(dexp_ub, [1, C])
                pl.load(dexp_ub, decay_exp_ws, [lin, 0])
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=2,
                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                # tail rows: sub-block 1 has nothing to store when valid_c <= HC
                orow = pl.min(HC, valid_c - rb)
                if orow > 0:
                    pl.set_validshape(vp_ub, [orow, vcur])
                    pl.set_validshape(obf_ub, [orow, vcur])
                    pl.cast(obf_ub, vp_ub, mode=pl.RoundMode.CAST_ROUND)
                    pl.set_validshape(obf_ub, [orow, vcur])
                    pl.store(o, obf_ub, [bh // Hd, t_base + rb, bh % Hd, voff],
                             order=[1, 3])
                pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3,
                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=1,
                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                for kob in pl.range(0, nkob):
                    koff = kob * C
                    kcur = pl.min(C, Kd - koff)
                    kh = (kcur + 1) // 2
                    kbase = sub2 * kh
                    krows = pl.min(kh, kcur - kbase)
                    pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=6,
                                              sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                    if krows > 0:
                        pl.set_validshape(st_ub, [krows, vcur])
                        pl.load(st_ub, state_gm, [bh, koff + kbase, voff])
                        _scan_state_update_vf(st_ub, d_ub, dexp_ub, krows, vcur, valid_c - 1)
                        pl.set_validshape(st_ub, [krows, vcur])
                        pl.store(state_gm, st_ub, [bh, koff + kbase, voff])
                    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=7,
                                             sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0,
                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # vector drain for this (bh,vv) pass: RELEASE_VN
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=5,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


# =============================================================================
#   Wrapper — HARD contract name gated_delta_rule_fwd
# =============================================================================
def gated_delta_rule_fwd(q, k, v=None, g=None, beta=None):
    """Host entry: allocate the GM workspaces and run the fused kernel once.

    Returns o [B, T, H, V] bfloat16.
    """
    B, T, H, K = q.shape
    V = v.shape[3]
    N = (T + C - 1) // C
    scale = 1.0 / (K ** 0.5)          # pure-Python scalar (allowed host op)
    dev = q.device
    total = B * H * N
    BH = B * H

    qs = torch.empty((total, C, K), dtype=torch.float32, device=dev)
    kf = torch.empty((total, C, K), dtype=torch.float32, device=dev)
    kb = torch.empty((total, C, K), dtype=torch.float32, device=dev)
    gf = torch.empty((total, C, 1), dtype=torch.float32, device=dev)
    V_col_pad = ((V + TILE_COL - 1) // TILE_COL) * TILE_COL   # v_b width (u/w u N-blocks)
    Kp = ((K + TILE_COL - 1) // TILE_COL) * TILE_COL    # kbeta_de width (u/w w N-blocks)
    vb = torch.empty((total, C, V_col_pad), dtype=torch.float32, device=dev)
    lt = torch.empty((C, C), dtype=torch.float32, device=dev)
    dec = torch.empty((total, C), dtype=torch.float32, device=dev)
    kk_gram = torch.empty((total, C, C), dtype=torch.float32, device=dev)
    qk_gram = torch.empty((total, C, C), dtype=torch.float32, device=dev)
    decay_exp = torch.empty((total, C), dtype=torch.float32, device=dev)
    q_de = torch.empty((total, C, K), dtype=torch.float32, device=dev)
    kbeta_de = torch.empty((total, C, Kp), dtype=torch.float32, device=dev)
    dfac = torch.empty((total, C, K), dtype=torch.float32, device=dev)
    lmask = torch.empty((total, C, C), dtype=torch.float32, device=dev)
    attn = torch.empty((total, C, C), dtype=torch.float32, device=dev)
    a = torch.empty((total, C, C), dtype=torch.float32, device=dev)
    Vp = ((V + 63) // 64) * 64          # pad V to a 64 multiple (full-width tiles)
    u = torch.empty((total, C, Vp), dtype=torch.float32, device=dev)
    w = torch.empty((total, C, K), dtype=torch.float32, device=dev)

    o = torch.empty((B, T, H, Vp), dtype=torch.bfloat16, device=dev)
    state_gm = torch.empty((BH, K, Vp), dtype=torch.float32, device=dev)
    vnew_gm = torch.empty((BH, C, Vp), dtype=torch.float32, device=dev)
    vscaled_gm = torch.empty((BH, C, Vp), dtype=torch.float32, device=dev)
    kft_gm = torch.empty((total, K, C), dtype=torch.float32, device=dev)

    g4 = g.unsqueeze(-1)
    beta4 = beta.unsqueeze(-1)

    max_blocks = get_platform_info().core_num or 1
    # ---- size the launch for BOTH phases.  Chunk-prep has total=B*H*N
    # items; the scan phase has BH*ceil(Vp/VT) items.  The old
    # min(total, max_blocks) starved scan whenever BH*vtiles > B*H*N
    # (e.g. colsplit B1 H2 K256 V256: total=4 but scan items=8).
    n_scan_items = BH * (Vp // 64)
    num_cores = max(1, min(max(total, n_scan_items), max_blocks))
    gated_delta_rule_fwd_kernel[None, num_cores](
        q, k, g4, beta4, v, qs, kf, kb, gf, vb, lt, dec, kk_gram, qk_gram,
        decay_exp, q_de, kbeta_de, dfac, lmask, attn, a, u, w,
        o, state_gm, vnew_gm, vscaled_gm, kft_gm, scale)
    torch.npu.synchronize()
    return o[..., :V]   # slice off the V padding (pure view)
