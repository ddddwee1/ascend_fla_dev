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

@vf()
def finalize_pre_vf(
    q_ub: Tensor,
    k_ub: Tensor,
    g_ub: Tensor,
    beta_ub: Tensor,
    dAqk_ub: Tensor,
    dAkk_ub: Tensor,
    q_scaled_ub: Tensor,
    k_scaled_ub: Tensor,
    kg_ub: Tensor,
    mqk_ub: Tensor,
    mbase_ub: Tensor,
    mbeta_ub: Tensor,
    rows: Var,
):
    glast = RegList(DT.float, REGS_D)
    g = RegList(DT.float, REGS_D)
    rscale = RegList(DT.float, REGS_D)
    cscale = RegList(DT.float, REGS_D)
    gmid = RegList(DT.float, REGS_D)      # 本仓新增：锚点取 g_last/2
    qr = RegList(DT.float, REGS_D)
    kr = RegList(DT.float, REGS_D)
    prod = RegList(DT.float, REGS_D)
    tmp = RegList(DT.float, REGS_D)
    cols = Reg(DT.int)
    daqk = Reg(DT.float)
    dakk = Reg(DT.float)
    base = Reg(DT.float)
    out = Reg(DT.float)
    zero = Reg(DT.float)
    beta_r = Reg(DT.float)
    mask_le = MaskReg(DT.int, init_mode=MaskType.NONE)
    mask_lt = MaskReg(DT.int, init_mode=MaskType.NONE)

    glast <<= g_ub[L - 1:L, 0:D]
    gmid <<= glast * MID
    zero <<= 0.0

    for r in range(rows):
        g <<= g_ub[r:r + 1, 0:D]
        tmp <<= g - gmid            # 上游：g - glast
        tmp <<= tmp * LN2
        rscale <<= tmp.exp()
        tmp <<= gmid - g            # 上游：glast - g
        tmp <<= tmp * LN2
        cscale <<= tmp.exp()

        qr <<= q_ub[r:r + 1, 0:D]
        prod <<= qr * rscale
        q_scaled_ub[r * D] <<= prod[0]
        q_scaled_ub[r * D + 64] <<= prod[1]

        kr <<= k_ub[r:r + 1, 0:D]
        prod <<= kr * rscale
        k_scaled_ub[r * D] <<= prod[0]
        k_scaled_ub[r * D + 64] <<= prod[1]

        prod <<= kr * cscale
        kg_ub[r * D] <<= prod[0]
        kg_ub[r * D + 64] <<= prod[1]

        cols.arange(0)
        daqk <<= dAqk_ub[r:r + 1, 0:L]
        compare(mask_le, cols, Var(r + 1), CompareMode.LT)  # j < r+1  == j <= r
        select(out, daqk, zero, mask=mask_le)
        mqk_ub[r:r + 1, 0:L] <<= out

        dakk <<= dAkk_ub[r:r + 1, 0:L]
        compare(mask_lt, cols, r, CompareMode.LT)           # j < r
        select(base, dakk, zero, mask=mask_lt)
        mbase_ub[r:r + 1, 0:L] <<= base
        beta_r <<= beta_ub[r:r + 1, 0:1].single()
        base <<= base * beta_r
        mbeta_ub[r:r + 1, 0:L] <<= base
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@kernel()
def finalize_pre_stable_kernel(q: GM[bf16, ('B', 'T', 'H', 128)], k: GM[bf16, ('B', 'T', 'H', 128)], g_cumsum: GM[bf16, ('B', 'T', 'HV', 128)], beta: GM[bf16, ('B', 'T', 'HV')], dAqk: GM[bf16, ('B', 'T', 'HV', 64)], dAkk: GM[bf16, ('B', 'T', 'HV', 64)], q_scaled: GM[bf16, ('B', 'HV', 'C', 64, 128)], k_scaled: GM[bf16, ('B', 'HV', 'C', 64, 128)], kg: GM[bf16, ('B', 'HV', 'C', 64, 128)], m_qk: GM[bf16, ('B', 'HV', 'C', 64, 64)], m_base: GM[bf16, ('B', 'HV', 'C', 64, 64)], m_beta: GM[bf16, ('B', 'HV', 'C', 64, 64)], B: i32, HV: i32, C: i32, H: i32, G: i32):
    q_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    k_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    g_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    beta_ub = Tensor(DT.bfloat16, [L, SCALAR_PACK_BF16], Position.UB)
    dAqk_ub = Tensor(DT.bfloat16, [L, L], Position.UB)
    dAkk_ub = Tensor(DT.bfloat16, [L, L], Position.UB)

    q_scaled_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    k_scaled_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    kg_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    mqk_ub = Tensor(DT.bfloat16, [L, L], Position.UB)
    mbase_ub = Tensor(DT.bfloat16, [L, L], Position.UB)
    mbeta_ub = Tensor(DT.bfloat16, [L, L], Position.UB)

    # KDA-P1: this kernel has no matmul -- every instruction in it runs on a
    # vector subblock. Partitioning the item space over GetCubeNum() and then
    # parking subblock 1 leaves HALF the vector engine idle (GetVecNum() ==
    # 2 * GetCubeNum()). Partition over GetVecNum() instead: one worker still
    # owns a whole (b, hv, c) item, so the duplicate-GM-write hazard the old
    # subblock-0 guard existed to prevent cannot come back -- the six outputs
    # are disjoint contiguous BHCLD blocks (>=8 KiB each), nowhere near sharing
    # a 32-byte GM write granule between workers. Arithmetic, buffers, read
    # pitches, store order within an item and the emitted VF are unchanged, so
    # the outputs are bitwise identical; only the worker count doubles.
    work_count = Var(B * HV * C)
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            bhv = Var(work / C)
            hv_idx = Var(bhv % HV)
            b_idx = Var(bhv / HV)

            # Strided on-device reads from token-major BTHVD primaries/caches; the
            # H->HV gather (h_idx = hv_idx / G) happens here. q/k carry H heads
            # (pitch H*D); g/beta/dAqk/dAkk carry HV (pitch HV*D / HV / HV*L).
            row0 = Var(c_idx * L)
            h_idx = Var(hv_idx / G)
            gm_to_ub_pad(q_ub[0:L, 0:D], q[b_idx, row0:row0 + L, h_idx, 0:D], L, D, (H - 1) * D, 0)
            gm_to_ub_pad(k_ub[0:L, 0:D], k[b_idx, row0:row0 + L, h_idx, 0:D], L, D, (H - 1) * D, 0)
            gm_to_ub_pad(g_ub[0:L, 0:D], g_cumsum[b_idx, row0:row0 + L, hv_idx, 0:D], L, D, (HV - 1) * D, 0)
            gm_to_ub_pad(beta_ub[0:L, 0:1], beta[b_idx, row0:row0 + L, hv_idx], L, 1, HV - 1, 0)
            gm_to_ub_pad(dAqk_ub[0:L, 0:L], dAqk[b_idx, row0:row0 + L, hv_idx, 0:L], L, L, (HV - 1) * L, 0)
            gm_to_ub_pad(dAkk_ub[0:L, 0:L], dAkk[b_idx, row0:row0 + L, hv_idx, 0:L], L, L, (HV - 1) * L, 0)

            finalize_pre_vf(
                q_ub, k_ub, g_ub, beta_ub, dAqk_ub, dAkk_ub,
                q_scaled_ub, k_scaled_ub, kg_ub,
                mqk_ub, mbase_ub, mbeta_ub, Var(L),
            )

            q_scaled[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= q_scaled_ub[0:L, 0:D]
            k_scaled[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= k_scaled_ub[0:L, 0:D]
            kg[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= kg_ub[0:L, 0:D]
            m_qk[b_idx, hv_idx, c_idx, 0:L, 0:L] <<= mqk_ub[0:L, 0:L]
            m_base[b_idx, hv_idx, c_idx, 0:L, 0:L] <<= mbase_ub[0:L, 0:L]
            m_beta[b_idx, hv_idx, c_idx, 0:L, 0:L] <<= mbeta_ub[0:L, 0:L]

    return q_scaled, k_scaled, kg, m_qk, m_base, m_beta
