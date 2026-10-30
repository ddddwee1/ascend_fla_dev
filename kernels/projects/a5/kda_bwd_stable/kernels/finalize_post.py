"""KDA kernel port; ABI and source provenance are owned by contract.json.

The arithmetic and cube-group partition come from the reviewed source.
Whole-chunk vector work is owned only by subblock zero to eliminate duplicate
GM writes; this source ownership correction is specified in contract.json.
"""

# ── 本仓的改动（唯一的改动，见 ../README.md）────────────────────────────────
# 上游把 chunk 内成对衰减 exp(g_i − g_j) 分解成 rscale_i · cscale_j，锚点取 g_last：
#     rscale = exp((g − g_last)·ln2)      g − g_last ≥ 0  → 最大 exp(+span)
#     cscale = exp((g_last − g)·ln2)      ≤ 0             → 最小 exp(−span)
# span 是 chunk 内门控跨度。fla 默认初始化的 KDA 层给出 span ≈ 94，而 fp32 与 bf16 的
# 上溢线都在 ln(MAX) ≈ 88.72 —— rscale 直接变 inf，同时 cscale 下溢到 0，下游四个
# 矩阵乘里 inf×0 得到 NaN。**这是真上溢**，与前向那边的下溢是两个方向，别混。
#
# 锚点可以任选：rscale_i · cscale_j = exp(g_i − g_j) 与锚点无关，而 finalize_pair 的
# 四个矩阵乘全是成对的（Mqk@kg.T 后乘 rscale、Mqk.T@q_scaled 后乘 cscale、
# Mbase@kg.T 后乘 rscale、Mbeta.T@k_scaled 后乘 cscale），所以常数一定抵消。
# 改取中点 g_last/2，两个因子各压到 ±span/2，可用跨度正好翻倍到 ~177。
# 数学同义，矩阵乘结构、流水、掩码一概不变。
MID = 0.5


import math

from ascriptor.a5 import *

L = 64

D = 128

REGS_D = D // 64

LN2 = math.log(2.0)

SCALAR_PACK_BF16 = DT.bfloat16.C0

SCALAR_PACK_F32 = DT.float.C0

@vf()
def finalize_post_vf(
    g_ub: Tensor,
    qkl_ub: Tensor,
    qkr_ub: Tensor,
    sbase_ub: Tensor,
    tbeta_ub: Tensor,
    qhv_ub: Tensor,
    khv_ub: Tensor,
    beta_ub: Tensor,
    dqhv_in_ub: Tensor,
    dkhv_in_ub: Tensor,
    dbeta_in_ub: Tensor,
    dgcore_in_ub: Tensor,
    dqhv_out_ub: Tensor,
    dkhv_out_ub: Tensor,
    dbeta_out_ub: Tensor,
    dg_out_ub: Tensor,
    rows: Var,
):
    glast = RegList(DT.float, REGS_D)
    g = RegList(DT.float, REGS_D)
    rscale = RegList(DT.float, REGS_D)
    cscale = RegList(DT.float, REGS_D)
    gmid = RegList(DT.float, REGS_D)      # 本仓新增：锚点取 g_last/2
    qkl = RegList(DT.float, REGS_D)
    qkr = RegList(DT.float, REGS_D)
    sbase = RegList(DT.float, REGS_D)
    tbeta = RegList(DT.float, REGS_D)
    dqpair = RegList(DT.float, REGS_D)
    dkpair = RegList(DT.float, REGS_D)
    rowc = RegList(DT.float, REGS_D)
    colc = RegList(DT.float, REGS_D)
    qh = RegList(DT.float, REGS_D)
    kh = RegList(DT.float, REGS_D)
    dgqk = RegList(DT.float, REGS_D)
    dkkk = RegList(DT.float, REGS_D)
    dgkk = RegList(DT.float, REGS_D)
    dqhv = RegList(DT.float, REGS_D)
    dkhv = RegList(DT.float, REGS_D)
    dgp = RegList(DT.float, REGS_D)
    running = RegList(DT.float, REGS_D)
    tmp = RegList(DT.float, REGS_D)
    prod = RegList(DT.float, REGS_D)
    beta_r = Reg(DT.float)
    dbadd = Reg(DT.float)
    dbin = Reg(DT.float)
    dbout = Reg(DT.float)
    dbout_bf16 = Reg(DT.bfloat16)

    glast <<= g_ub[L - 1:L, 0:D]
    gmid <<= glast * MID
    running.fill(0.0)

    # Walk rows L-1 -> 0 so `running` is the per-chunk reverse cumsum of dg'.
    for rt in range(rows):
        r = rows - 1 - rt

        g <<= g_ub[r:r + 1, 0:D]
        tmp <<= g - gmid            # 上游：g - glast
        tmp <<= tmp * LN2
        rscale <<= tmp.exp()
        tmp <<= gmid - g            # 上游：glast - g
        tmp <<= tmp * LN2
        cscale <<= tmp.exp()

        qkl <<= qkl_ub[r:r + 1, 0:D]
        dqpair <<= rscale * qkl
        qkr <<= qkr_ub[r:r + 1, 0:D]
        dkpair <<= cscale * qkr
        sbase <<= sbase_ub[r:r + 1, 0:D]
        rowc <<= rscale * sbase
        tbeta <<= tbeta_ub[r:r + 1, 0:D]
        colc <<= cscale * tbeta

        qh <<= qhv_ub[r:r + 1, 0:D]
        kh <<= khv_ub[r:r + 1, 0:D]
        beta_r <<= beta_ub[r:r + 1, 0:1].single()

        # dg_qk = q_hv*dq_pair - k_hv*dk_pair
        dgqk <<= qh * dqpair
        prod <<= kh * dkpair
        dgqk <<= dgqk - prod

        # dk_kk = beta*row_contrib + col_contrib
        dkkk <<= rowc * beta_r
        dkkk <<= dkkk + colc

        # dbeta_add = (k_hv * row_contrib).sum(D)
        prod <<= kh * rowc
        dbadd <<= prod.cadd()

        # dg_kk = beta*k_hv*row_contrib - k_hv*col_contrib
        dgkk <<= kh * rowc
        dgkk <<= dgkk * beta_r
        prod <<= kh * colc
        dgkk <<= dgkk - prod

        # dq_hv' = dq_hv + dq_pair   (fp32 out)
        dqhv <<= dqhv_in_ub[r:r + 1, 0:D]
        dqhv <<= dqhv + dqpair
        dqhv_out_ub[r:r + 1, 0:D] <<= dqhv

        # dk_hv' = dk_hv + dk_pair + dk_kk   (fp32 out)
        dkhv <<= dkhv_in_ub[r:r + 1, 0:D]
        dkhv <<= dkhv + dkpair
        dkhv <<= dkhv + dkkk
        dkhv_out_ub[r:r + 1, 0:D] <<= dkhv

        # dbeta' = dbeta + dbeta_add ; cast fp32->bf16 for the public bf16 output
        dbin <<= dbeta_in_ub[r:r + 1, 0:1].single()
        dbout <<= dbin + dbadd
        dbout_bf16 <<= dbout.cast()       # fp32 -> bf16 (round-to-nearest-even)
        dbeta_out_ub[r:r + 1, 0:1] <<= dbout_bf16.single_value()

        # dg' = dg_core + dg_qk + dg_kk ; running += dg' ; dg[r] = running
        dgp <<= dgcore_in_ub[r:r + 1, 0:D]
        dgp <<= dgp + dgqk
        dgp <<= dgp + dgkk
        running <<= running + dgp
        dg_out_ub[r * D] <<= running[0]
        dg_out_ub[r * D + 64] <<= running[1]
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@kernel()
def finalize_post_stable_kernel(g_cumsum: GM[bf16, ('B', 'T', 'HV', 128)], qk_left: GM[bf16, ('B', 'HV', 'C', 64, 128)], qk_right: GM[bf16, ('B', 'HV', 'C', 64, 128)], s_base: GM[bf16, ('B', 'HV', 'C', 64, 128)], t_beta: GM[bf16, ('B', 'HV', 'C', 64, 128)], q: GM[bf16, ('B', 'T', 'H', 128)], k: GM[bf16, ('B', 'T', 'H', 128)], beta: GM[bf16, ('B', 'T', 'HV')], dq_hv_in: GM[bf16, ('B', 'T', 'HV', 128)], dk_hv_in: GM[bf16, ('B', 'T', 'HV', 128)], dbeta_in: GM[f32, ('B', 'T', 'HV')], dg_core_in: GM[bf16, ('B', 'T', 'HV', 128)], dq_hv_out: GM[f32, ('B', 'HV', 'C', 64, 128)], dk_hv_out: GM[f32, ('B', 'HV', 'C', 64, 128)], dbeta_out: GM[bf16, ('B', 'T', 'HV')], dg_out: GM[bf16, ('B', 'T', 'HV', 128)], B: i32, HV: i32, C: i32, H: i32, G: i32):
    g_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    qkl_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    qkr_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    sbase_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    tbeta_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    qhv_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    khv_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    beta_ub = Tensor(DT.bfloat16, [L, SCALAR_PACK_BF16], Position.UB)
    dqhv_in_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dkhv_in_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dbeta_in_ub = Tensor(DT.float, [L, SCALAR_PACK_F32], Position.UB)  # fp32 per-token scalar, 32B-row pack
    dgcore_in_ub = Tensor(DT.bfloat16, [L, D], Position.UB)

    dqhv_out_ub = Tensor(DT.float, [L, D], Position.UB)
    dkhv_out_ub = Tensor(DT.float, [L, D], Position.UB)
    dbeta_out_ub = Tensor(DT.bfloat16, [L, SCALAR_PACK_BF16], Position.UB)  # bf16 per-token scalar, 32B-row pack
    dg_out_ub = Tensor(DT.bfloat16, [L, D], Position.UB)

    # KDA-P1: no matmul here either, so parking subblock 1 idles half the vector
    # engine. Partition the (b, c) item space over GetVecNum() instead of
    # GetCubeNum(). A worker still owns a whole chunk -- ALL of its HV heads and
    # all 64 of its tokens -- which is what keeps the two public token-major
    # outputs safe: dbeta_out[b, t, 0:HV] is 2*HV bytes wide and dg_out's row is
    # 2*HV*D, so a 32-byte GM write granule never spans two chunks' token ranges.
    # The per-chunk reverse cumsum in finalize_post_vf stays whole (it is serial
    # over the 64 rows of one chunk and that loop is untouched), and the heads
    # stay on one worker, so no head or row is split across workers. Arithmetic,
    # buffers, pitches and store order within an item are unchanged: bitwise
    # identical outputs, twice the workers.
    work_count = Var(B * C)
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            b_idx = Var(work / C)
            # All heads of this chunk share scalar GM transaction blocks and stay
            # on this worker.
            for hv_idx in range(HV):

                row0 = Var(c_idx * L)
                h_idx = Var(hv_idx / G)
                # g_cumsum: cache, strided (pitch HV*D)
                gm_to_ub_pad(g_ub[0:L, 0:D], g_cumsum[b_idx, row0:row0 + L, hv_idx, 0:D], L, D, (HV - 1) * D, 0)
                # qk_*: intra-stage from finalize_pair, contiguous GM-native (keep <<=)
                qkl_ub[0:L, 0:D] <<= qk_left[b_idx, hv_idx, c_idx, 0:L, 0:D]
                qkr_ub[0:L, 0:D] <<= qk_right[b_idx, hv_idx, c_idx, 0:L, 0:D]
                sbase_ub[0:L, 0:D] <<= s_base[b_idx, hv_idx, c_idx, 0:L, 0:D]
                tbeta_ub[0:L, 0:D] <<= t_beta[b_idx, hv_idx, c_idx, 0:L, 0:D]
                # q/k: primaries, strided (pitch H*D) with on-device H->HV gather
                gm_to_ub_pad(qhv_ub[0:L, 0:D], q[b_idx, row0:row0 + L, h_idx, 0:D], L, D, (H - 1) * D, 0)
                gm_to_ub_pad(khv_ub[0:L, 0:D], k[b_idx, row0:row0 + L, h_idx, 0:D], L, D, (H - 1) * D, 0)
                # beta: primary, strided (pitch HV, burst_len=1)
                gm_to_ub_pad(beta_ub[0:L, 0:1], beta[b_idx, row0:row0 + L, hv_idx], L, 1, HV - 1, 0)
                # inverse-stage outputs, strided BTHVD/BTHV (dbeta_in is fp32 -> fp32 UB)
                gm_to_ub_pad(dqhv_in_ub[0:L, 0:D], dq_hv_in[b_idx, row0:row0 + L, hv_idx, 0:D], L, D, (HV - 1) * D, 0)
                gm_to_ub_pad(dkhv_in_ub[0:L, 0:D], dk_hv_in[b_idx, row0:row0 + L, hv_idx, 0:D], L, D, (HV - 1) * D, 0)
                gm_to_ub_pad(dbeta_in_ub[0:L, 0:1], dbeta_in[b_idx, row0:row0 + L, hv_idx], L, 1, HV - 1, 0)
                gm_to_ub_pad(dgcore_in_ub[0:L, 0:D], dg_core_in[b_idx, row0:row0 + L, hv_idx, 0:D], L, D, (HV - 1) * D, 0)

                finalize_post_vf(
                    g_ub, qkl_ub, qkr_ub, sbase_ub, tbeta_ub, qhv_ub, khv_ub, beta_ub,
                    dqhv_in_ub, dkhv_in_ub, dbeta_in_ub, dgcore_in_ub,
                    dqhv_out_ub, dkhv_out_ub, dbeta_out_ub, dg_out_ub, Var(L),
                )

                # dq_hv'/dk_hv': intra-stage to finalize_reduce, contiguous GM-native fp32 (keep)
                dq_hv_out[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= dqhv_out_ub[0:L, 0:D]
                dk_hv_out[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= dkhv_out_ub[0:L, 0:D]
                # dbeta/dg: public outputs, strided into token-major BTHV / BTHVD bf16
                ub_to_gm_pad(dbeta_out[b_idx, row0:row0 + L, hv_idx], dbeta_out_ub[0:L, 0:1], L, 1, 0, HV - 1)
                ub_to_gm_pad(dg_out[b_idx, row0:row0 + L, hv_idx, 0:D], dg_out_ub[0:L, 0:D], L, D, 0, (HV - 1) * D)

    return dq_hv_out, dk_hv_out, dbeta_out, dg_out
