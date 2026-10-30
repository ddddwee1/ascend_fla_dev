"""chunk 内打分 —— 成对衰减改成**对称分解**，可用门控跨度翻倍。

改自 ascriptor 的 ``kda_fwd/kernels/intra.py``（只读引用，AGENTS.md §3）。矩阵乘的
结构、流水、掩码一概不变；唯一的差别在 ``score_preprocess_vf`` 里三行门控算术。

**原版的问题。** 它收已经指数化的 ``eg = exp(gc)``，然后构造

    qg[i,d]    =  q[i,d] · eg[i,d] · scale
    kbg[j,d]   = -k[j,d] · eg[j,d] · beta[j]
    kgneg[j,d] =  k[j,d] / eg[j,d]          ← 除法

于是 ``Aqk[i,j] = Σ_d qg[i,d]·kgneg[j,d] = Σ_d q k exp(gc_i − gc_j)·scale``。数学上对，
但这是把一个恒 ≤1 的量（i ≥ j 时 gc_i ≤ gc_j）拆成了 ``exp(gc_i)`` 与 ``exp(−gc_j)``
两个因子。第二个因子在 ``gc_j`` 很负时上溢；而 ``eg`` 本身先下溢到 0（fp32 非正规数被
flush，阈值 ``-ln(FLT_MIN_NORMAL) ≈ 87.3``），``k/0 = inf``，再乘上 ``qg = 0`` 就是
**NaN**。实测跨度 ≥88.67 时 ``Aqk``/``strict`` 出 NaN，跨度 ≤66.84 完全正常。

**这里的做法。** 收 log 空间的 ``g_cumsum``，按**每个通道**取中点

    m[d] = gc[L−1, d] / 2

（``gc`` 沿 chunk 单调不增，``gc[0] ≈ 0``、``gc[L−1] = −S``，故 ``m = −S/2``），再构造

    pos[i,d] = exp(gc[i,d] − m[d])   ∈ [exp(−S/2), exp(+S/2)]
    neg[j,d] = exp(m[d] − gc[j,d])   ∈ [exp(−S/2), exp(+S/2)]

两个因子的指数都被压到 ``±S/2``，而乘积 ``pos[i]·neg[j] = exp(gc_i − gc_j)`` 与原版
**逐位同义**（m 在乘积里抵消）。于是可用跨度从 ``~87`` 翻到 ``~174``。

这不是"完全不分解"—— 用 matmul 在通道维上求和就必须分解。能做到的最好是**对称地**
分解，让两个因子同时留在范围内，这是标准做法。要彻底去掉上限得把 64×64 的 tile 再按
行列分块（每对子块用各自的 m），那是更大的改动，等实测需要再做。

上三角的垃圾值无害：``apply_score_masks_vf`` 用的是 ``select`` 而不是乘零，所以 inf
是被**替换**掉的，不会变成 ``inf × 0 = NaN``。

**``set_hf32(False)`` 是必需的，不是保险。** 这个 kernel 的两个 ``matmul`` 的 L1 操作数
（``l1_qg`` / ``l1_kbg`` / ``l1_kgneg``）都是 ``DT.float`` —— 这是本仓三个 KDA 单元里
**唯一**一处 FP32 操作数的 cube matmul（``wy.py`` / ``recurrent.py`` / ``kda_bwd_stable``
的矩阵乘操作数全是 bf16，不受影响）。而 FP32 cube ``matmul`` 的精度服从 AI core 的 HF32
舍入位，kernel 发射不会重置它：不显式设置就会继承**这块核上一次跑的随便什么进程**留下的
状态。共享卡上这意味着别的租户能把我们的精度静默降到 HF32（相对误差约 2^-11.75 量级），
没有报错、没有 NaN、数值量级正常。根因与决定性复现见 PK-07 第二轮（``gaps.json`` 的
``a5-fp32-cube-matmul-inherits-hf32``，P0），处置模板是它的
``pkda_chunk_fwd/kernels/output.py``。

它同时也是**性能测量的前提**：HF32 的 cube 吞吐与 FP32 不同，不钉住这一位，这个 kernel
的耗时就随卡的历史状态变，profile 数字不可复现。KDA-P1 在真机上对 off/on 两份 build
做了 A/B（见 ``docs/research/kda_perf_fwd.md``）。

代价：pin 住的 ascriptor 参考解释器没有实现 ``core.set_hf32``，所以本单元的 ``sim`` /
``pipesim`` 两条非真机 launcher 跑不了这个 kernel（``native`` 与 ``reference`` 不受影响）
—— 与 PK-07 同一笔取舍。
"""

from ascriptor.a5 import *

L = 64

K_DIM = 128

K_BLOCK = 64

K_TILES = K_DIM // K_BLOCK

HALF_L = L // 2


@vf()
def score_preprocess_stable_vf(
    q_ub: Tensor,
    k_ub: Tensor,
    gc_ub: Tensor,
    glast_ub: Tensor,
    beta_ub: Tensor,
    qg_ub: Tensor,
    kbg_ub: Tensor,
    kgneg_ub: Tensor,
    rows: Var,
    scale: Var,
):
    q_row = Reg(DT.float)
    k_row = Reg(DT.float)
    gc_row = Reg(DT.float)
    mid = Reg(DT.float)
    shifted = Reg(DT.float)
    pos = Reg(DT.float)
    neg = Reg(DT.float)
    beta_val = Reg(DT.float)
    half = Reg(DT.float)
    out_row = Reg(DT.float)

    half <<= 0.5

    for r in range(rows):
        beta_val <<= beta_ub[0:1, r:r + 1].single()
        for tile in range(K_TILES):
            k_begin = tile * K_BLOCK
            k_end = k_begin + K_BLOCK
            q_row <<= q_ub[r:r + 1, k_begin:k_end]
            k_row <<= k_ub[r:r + 1, k_begin:k_end]
            gc_row <<= gc_ub[r:r + 1, k_begin:k_end]

            # m[d] = gc[L-1, d] / 2，逐通道的中点
            mid <<= glast_ub[0:1, k_begin:k_end]
            mid <<= mid * half
            shifted <<= gc_row + mid.neg()          # gc - m
            pos <<= shifted.exp()                   # exp(gc - m)
            neg <<= shifted.neg().exp()             # exp(m - gc)

            out_row <<= q_row * pos
            out_row <<= out_row * scale
            qg_ub[r:r + 1, k_begin:k_end] <<= out_row

            out_row <<= k_row * pos
            out_row <<= out_row * beta_val
            out_row <<= out_row.neg()
            kbg_ub[r:r + 1, k_begin:k_end] <<= out_row

            # 原版这里是 k / eg；改成乘 exp(m - gc)，不再有除法
            out_row <<= k_row * neg
            kgneg_ub[r:r + 1, k_begin:k_end] <<= out_row


@vf()
def apply_score_masks_vf(
    aqk_full_ub: Tensor,
    strict_full_ub: Tensor,
    aqk_out_ub: Tensor,
    strict_out_ub: Tensor,
    row_begin: Var,
    rows: Var,
):
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    aqk_row = Reg(DT.float)
    strict_row = Reg(DT.float)
    aqk_masked = Reg(DT.float)
    strict_masked = Reg(DT.float)
    aqk_bf16 = Reg(DT.bfloat16)
    lower_eq_mask = MaskReg(DT.int, init_mode=MaskType.NONE)
    strict_lower_mask = MaskReg(DT.int, init_mode=MaskType.NONE)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)

    zero <<= 0.0
    cols.arange(0)
    row_mask_bf16 <<= Var(L * 2, dtype=DT.uint32)

    for r in range(rows):
        abs_r = Var(row_begin + r)

        aqk_row <<= aqk_full_ub[r:r + 1, 0:L]
        compare(lower_eq_mask, cols, abs_r + 1, CompareMode.LT)
        select(aqk_masked, aqk_row, zero, mask=lower_eq_mask)
        aqk_bf16 <<= aqk_masked.astype(DT.bfloat16)
        reg_to_ub_downsample(aqk_out_ub[r:r + 1, 0:L], aqk_bf16, mask=row_mask_bf16)

        strict_row <<= strict_full_ub[r:r + 1, 0:L]
        compare(strict_lower_mask, cols, abs_r, CompareMode.LT)
        select(strict_masked, strict_row, zero, mask=strict_lower_mask)
        strict_out_ub[r:r + 1, 0:L] <<= strict_masked


@kernel()
def kda_sub2_score_stable_kernel(
    q: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k: GM[bf16, ('B', 'H', 'C', 64, 128)],
    g_cumsum: GM[f32, ('B', 'HV', 'C', 64, 128)],
    beta: GM[f32, ('B', 'HV', 'C', 64)],
    Aqk: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    strict: GM[f32, ('B', 'HV', 'C', 64, 64)],
    B: i32,
    H: i32,
    HV: i32,
    C: i32,
    length_per_chunk: i32,
    head_dim: i32,
    scale: f32,
):
    vcmutex = VcMutex(
        0,
        depth=2,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    score_cvmutex = CvMutex(
        1,
        depth=2,
        src_start_pipe=Pipe.FIX,
        dst_start_pipe=Pipe.V,
        src_end_pipe=Pipe.FIX,
        dst_end_pipe=Pipe.V,
    )
    l1_qg = DBuff(DT.float, [L, K_DIM], Position.L1)
    l1_kbg = DBuff(DT.float, [L, K_DIM], Position.L1)
    l1_kgneg = DBuff(DT.float, [L, K_DIM], Position.L1)
    l0c_aqk = DBuff(DT.float, [L, L], Position.L0C)
    l0c_strict = DBuff(DT.float, [L, L], Position.L0C)

    q_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    k_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    gc_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    glast_ub = DBuff(DT.float, [1, K_DIM], Position.UB)
    beta_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    qg_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    kbg_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    kgneg_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    aqk_full_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    strict_full_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    aqk_out_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    strict_out_ub = DBuff(DT.float, [HALF_L, L], Position.UB)

    work_count = B * HV * C
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    group = Var(HV // H)
    pre_cnt = Var(0)
    cube_cnt = Var(0)
    post_cnt = Var(0)

    # The two matmuls below take FP32 L1 operands, so they obey the AI core's
    # HF32 rounding bit. Nothing in a launch resets it -- pin it. See the module
    # docstring and gaps.json a5-fp32-cube-matmul-inherits-hf32.
    with cube_scope():
        set_hf32(False)

    with auto_sync():
        for pipe_work in range(work_begin, work_end + 2):
            row_begin = Var(GetSubBlockIdx() * HALF_L)
            row_end = Min(row_begin + HALF_L, L)
            rows_this = Var(row_end - row_begin)

            if pipe_work < work_end:
                c_idx = Var(pipe_work % C)
                tmp = Var(pipe_work // C)
                hv_idx = Var(tmp % HV)
                b_idx = Var(tmp // HV)
                h_idx = Var(hv_idx // group)

                q_ub[pre_cnt][0:rows_this, 0:K_DIM] <<= q[b_idx, h_idx, c_idx, row_begin:row_end, 0:K_DIM]
                k_ub[pre_cnt][0:rows_this, 0:K_DIM] <<= k[b_idx, h_idx, c_idx, row_begin:row_end, 0:K_DIM]
                gc_ub[pre_cnt][0:rows_this, 0:K_DIM] <<= g_cumsum[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM]
                glast_ub[pre_cnt][0:1, 0:K_DIM] <<= g_cumsum[b_idx, hv_idx, c_idx, L - 1:L, 0:K_DIM]
                beta_ub[pre_cnt][0:1, 0:rows_this] <<= beta[b_idx, hv_idx, c_idx, row_begin:row_end]
                score_preprocess_stable_vf(
                    q_ub[pre_cnt],
                    k_ub[pre_cnt],
                    gc_ub[pre_cnt],
                    glast_ub[pre_cnt],
                    beta_ub[pre_cnt],
                    qg_ub[pre_cnt],
                    kbg_ub[pre_cnt],
                    kgneg_ub[pre_cnt],
                    rows_this,
                    scale,
                )

                vcmutex.lock()
                l1_qg[pre_cnt][row_begin:row_end, 0:K_DIM] <<= qg_ub[pre_cnt][0:rows_this, 0:K_DIM]
                l1_kbg[pre_cnt][row_begin:row_end, 0:K_DIM] <<= kbg_ub[pre_cnt][0:rows_this, 0:K_DIM]
                l1_kgneg[pre_cnt][row_begin:row_end, 0:K_DIM] <<= kgneg_ub[pre_cnt][0:rows_this, 0:K_DIM]
                vcmutex.ready()
                pre_cnt += 1

            if (pipe_work > work_begin) and (pipe_work < work_end + 1):
                vcmutex.wait()
                matmul(l0c_aqk[cube_cnt], l1_qg[cube_cnt], l1_kgneg[cube_cnt], splitk=K_BLOCK, m=L, n=L, k=K_DIM)
                matmul(l0c_strict[cube_cnt], l1_kbg[cube_cnt], l1_kgneg[cube_cnt], splitk=K_BLOCK, m=L, n=L, k=K_DIM)
                vcmutex.free()
                score_cvmutex.lock()
                aqk_full_ub[cube_cnt] <<= l0c_aqk[cube_cnt]
                strict_full_ub[cube_cnt] <<= l0c_strict[cube_cnt]
                score_cvmutex.ready()
                cube_cnt += 1

            if pipe_work > work_begin + 1:
                work = Var(pipe_work - 2)
                c_idx = Var(work % C)
                tmp = Var(work // C)
                hv_idx = Var(tmp % HV)
                b_idx = Var(tmp // HV)

                score_cvmutex.wait()
                apply_score_masks_vf(
                    aqk_full_ub[post_cnt],
                    strict_full_ub[post_cnt],
                    aqk_out_ub[post_cnt],
                    strict_out_ub[post_cnt],
                    row_begin,
                    rows_this,
                )
                score_cvmutex.free()
                Aqk[b_idx, hv_idx, c_idx, row_begin:row_end, 0:L] <<= aqk_out_ub[post_cnt][0:rows_this, 0:L]
                strict[b_idx, hv_idx, c_idx, row_begin:row_end, 0:L] <<= strict_out_ub[post_cnt][0:rows_this, 0:L]
                post_cnt += 1

    return Aqk, strict
