"""A2 backward inverse_epilogue: un-gate the inverse_mm gradients into dq/dk/dv/dbeta/dg_core and k_exp.

Port of ascriptor ``a5/kda_bwd/kernels/inverse_epilogue.py`` to the A2 (c220) facade. Arithmetic and order
unchanged, per chunk and value head (gates are log2, hence the ln2 scaling before ``exp``)::

    exp_g   = exp(g * ln2)                 exp_lmg = exp((g_last - g) * ln2)
    dq      = d_qg * exp_g * D^-0.5        k_exp   = k * exp_g
    dk_kg   = d_kg * exp_lmg               dv      = d_v_beta * beta
    dbeta   = sum_D(d_v_beta * v) + sum_D(d_k_beta_g * k_exp)
    dk      = dk_kg + d_k_beta_g * beta * exp_g
    dg      = q * dq - k * dk_kg + d_k_beta_g * k_exp * beta
    dg[63] += sum_D(h * dh) * exp_g_last + sum_tokens(k * dk_kg)

A2 form: no ``@vf``; one tile op per row per term, the three reductions via a two-stage ``cadd`` plus a scalar
read, and the per-token ``beta``/``dbeta`` scalars through ``Var``. Public ``q``/``k``/``v``/``beta`` and the
public gradients are token-major 2-D views; the chain caches stay chunk-major.

The cumulative gate stays FP32; the other per-row inputs stay BF16 and are cast one row at a time, and the chunk is loaded in
two halves of 32 rows: b3 has 192 KB of UB, and a whole 64-row chunk of FP32 inputs does not fit. ``dg`` is
the exception that stays resident for all 64 rows, because its last row takes a correction that is only
complete once every row of the chunk has contributed to ``term2``.

D-PM-60: derived FP32 cumulative-gate ABI. Other arithmetic, ownership and synchronization are unchanged.
"""

import math

from ascriptor.a2 import *

L = 64

D = 128

HALF_L = L // 2

LN2 = math.log(2.0)

SCALE = 1.0 / (D ** 0.5)


@kernel(mode="vec")
def inverse_epilogue_a2_fp32gc_kernel(
    d_qg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    d_kg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    d_v_beta: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    d_k_beta_g: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    q: GM[bf16, ('BT', 'HK')],
    k: GM[bf16, ('BT', 'HK')],
    v: GM[bf16, ('BT', 'HVK')],
    g_cumsum: GM[f32, ('BT', 'HVK')],
    beta: GM[bf16, ('BT', 'HVB')],
    h: GM[bf16, ('B', 'C', 'HV', 128, 128)],
    dh: GM[bf16, ('B', 'C', 'HV', 128, 128)],
    dq_hv: GM[bf16, ('BT', 'HVK')],
    dk_hv: GM[bf16, ('BT', 'HVK')],
    dv: GM[bf16, ('BT', 'HVK')],
    dbeta: GM[f32, ('BT', 'HVB')],
    dg_core: GM[bf16, ('BT', 'HVK')],
    k_exp: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    B: i32,
    Hq: i32,  # named Hq, not H: the ACLNN API lowercases it and it would collide with the h tensor
    HV: i32,
    C: i32,
):
    g_f_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    q_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    k_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    v_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dqg_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dkg_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dvb_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dkbg_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dq_out_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dk_out_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dv_out_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    kexp_out_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dg_f_ub = Tensor(DT.float, [L, D], Position.UB)
    dg_b_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dbeta_f_ub = Tensor(DT.float, [L, 8], Position.UB)
    # beta is staged through UB and cast on the vector unit: a scalar load straight out of BF16 memory
    # needs a scalar bf16 -> float cast, which the device compiler rejects ("not support bf16 type cast").
    # The 32-byte row pack is what gm_to_ub_pad produces for a strided single-element column.
    beta_b_ub = Tensor(DT.bfloat16, [L, 16], Position.UB)
    beta_f_ub = Tensor(DT.float, [L, 16], Position.UB)
    glast_ub = Tensor(DT.float, [1, D], Position.UB)
    hrow_b_ub = Tensor(DT.bfloat16, [1, D], Position.UB)
    dhrow_b_ub = Tensor(DT.bfloat16, [1, D], Position.UB)
    hrow_ub = Tensor(DT.float, [1, D], Position.UB)
    dhrow_ub = Tensor(DT.float, [1, D], Position.UB)
    qrow_ub = Tensor(DT.float, [1, D], Position.UB)
    krow_ub = Tensor(DT.float, [1, D], Position.UB)
    vrow_ub = Tensor(DT.float, [1, D], Position.UB)
    dqgrow_ub = Tensor(DT.float, [1, D], Position.UB)
    dkgrow_ub = Tensor(DT.float, [1, D], Position.UB)
    dvbrow_ub = Tensor(DT.float, [1, D], Position.UB)
    dkbgrow_ub = Tensor(DT.float, [1, D], Position.UB)
    dqrow_ub = Tensor(DT.float, [1, D], Position.UB)
    kexprow_ub = Tensor(DT.float, [1, D], Position.UB)
    expg_ub = Tensor(DT.float, [1, D], Position.UB)
    explmg_ub = Tensor(DT.float, [1, D], Position.UB)
    expgl_ub = Tensor(DT.float, [1, D], Position.UB)
    sh_ub = Tensor(DT.float, [1, D], Position.UB)
    dkkg_ub = Tensor(DT.float, [1, D], Position.UB)
    acc_ub = Tensor(DT.float, [1, D], Position.UB)
    term2_ub = Tensor(DT.float, [1, D], Position.UB)
    dglast_ub = Tensor(DT.float, [1, D], Position.UB)
    tmp_ub = Tensor(DT.float, [1, D], Position.UB)
    dot_ub = Tensor(DT.float, [1, 64], Position.UB)
    dot2_ub = Tensor(DT.float, [1, 64], Position.UB)

    # Precautionary store fences. On the pinned library auto_sync emits the V -> MTE3 and MTE3 -> V guards
    # for some of this kernel's output staging buffers and not others (read off the generated vector
    # source). No divergence was traced to the gap: the device errors first suspected here turned out to
    # be the packed-column cast below. These two fences order the whole store group against the vector
    # work on either side of it, and are kept as insurance.
    store_ready = DEvent(Pipe.V, Pipe.MTE3)
    store_done = DEvent(Pipe.MTE3, Pipe.V)

    group = Var(HV // Hq)
    work_count = B * HV * C
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)
    beta_val = Var(0.0, dtype=DT.float)
    dot_val = Var(0.0, dtype=DT.float)
    sum_val = Var(0.0, dtype=DT.float)
    n_d = L * D
    n_hd = HALF_L * D

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            bhv = Var(work // C)
            hv_idx = Var(bhv % HV)
            b_idx = Var(bhv // HV)
            h_idx = Var(hv_idx // group)
            row0 = Var(b_idx * C * L + c_idx * L)
            qk_col = Var(h_idx * D)
            hv_col = Var(hv_idx * D)

            gm_to_ub_pad(beta_b_ub[0:L, 0:1], beta[row0:row0 + L, hv_idx:hv_idx + 1], L, 1, HV - 1, 0)
            # one cast per packed row. A single whole-tile cast over the pack lowers to a vconv with a zero
            # source block stride on c220, which replicates row 0's block into every row: on the device every
            # token then carried beta[0]. The functional simulator does not model the block strides.
            for pack_row in range(0, L):
                cast(beta_f_ub[pack_row:pack_row + 1, 0:1], beta_b_ub[pack_row:pack_row + 1, 0:1],
                     round_mode=RoundMode.NONE, count=1)
            glast_ub[0:1, 0:D] <<= g_cumsum[row0 + L - 1:row0 + L, hv_col:hv_col + D]

            # exp_g_last, and term1 of d_g_last = sum_D(h[row] * dh[row]) * exp_g_last[row]
            muls(sh_ub[0:1, 0:D], glast_ub[0:1, 0:D], LN2, count=D)
            exp(expgl_ub[0:1, 0:D], sh_ub[0:1, 0:D], count=D)
            for gk in range(0, D):
                hrow_b_ub[0:1, 0:D] <<= h[b_idx, c_idx, hv_idx, gk:gk + 1, 0:D]
                dhrow_b_ub[0:1, 0:D] <<= dh[b_idx, c_idx, hv_idx, gk:gk + 1, 0:D]
                cast(hrow_ub[0:1, 0:D], hrow_b_ub[0:1, 0:D], round_mode=RoundMode.NONE, count=D)
                cast(dhrow_ub[0:1, 0:D], dhrow_b_ub[0:1, 0:D], round_mode=RoundMode.NONE, count=D)
                mul(tmp_ub[0:1, 0:D], hrow_ub[0:1, 0:D], dhrow_ub[0:1, 0:D], count=D)
                cadd(dot_ub[0:1, 0:2], tmp_ub[0:1, 0:D], repeat=2, count_per_rep=64)
                cadd(dot2_ub[0:1, 0:1], dot_ub[0:1, 0:2], repeat=1, count_per_rep=2)
                dot_val.GetValueFrom(dot2_ub[0:1, 0:1])
                sum_val.GetValueFrom(expgl_ub[0:1, gk:gk + 1])
                sum_val.set(sum_val * dot_val)
                sum_val.SetValueTo(dglast_ub[0:1, gk:gk + 1])

            dup(term2_ub[0:1, 0:D], 0.0, count=D)
            for half in range(0, 2):
                r0 = Var(half * HALF_L)
                tok = Var(row0 + r0)

                g_f_ub[0:HALF_L, 0:D] <<= g_cumsum[tok:tok + HALF_L, hv_col:hv_col + D]
                q_b_ub[0:HALF_L, 0:D] <<= q[tok:tok + HALF_L, qk_col:qk_col + D]
                k_b_ub[0:HALF_L, 0:D] <<= k[tok:tok + HALF_L, qk_col:qk_col + D]
                v_b_ub[0:HALF_L, 0:D] <<= v[tok:tok + HALF_L, hv_col:hv_col + D]
                dqg_b_ub[0:HALF_L, 0:D] <<= d_qg[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D]
                dkg_b_ub[0:HALF_L, 0:D] <<= d_kg[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D]
                dvb_b_ub[0:HALF_L, 0:D] <<= d_v_beta[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D]
                dkbg_b_ub[0:HALF_L, 0:D] <<= d_k_beta_g[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D]

                for r in range(0, HALF_L):
                    cast(qrow_ub[0:1, 0:D], q_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    cast(krow_ub[0:1, 0:D], k_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    cast(vrow_ub[0:1, 0:D], v_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    cast(dqgrow_ub[0:1, 0:D], dqg_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    cast(dkgrow_ub[0:1, 0:D], dkg_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    cast(dvbrow_ub[0:1, 0:D], dvb_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    cast(dkbgrow_ub[0:1, 0:D], dkbg_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)

                    muls(sh_ub[0:1, 0:D], g_f_ub[r:r + 1, 0:D], LN2, count=D)
                    exp(expg_ub[0:1, 0:D], sh_ub[0:1, 0:D], count=D)
                    sub(sh_ub[0:1, 0:D], glast_ub[0:1, 0:D], g_f_ub[r:r + 1, 0:D], count=D)
                    muls(sh_ub[0:1, 0:D], sh_ub[0:1, 0:D], LN2, count=D)
                    exp(explmg_ub[0:1, 0:D], sh_ub[0:1, 0:D], count=D)
                    beta_val.GetValueFrom(beta_f_ub[r0 + r:r0 + r + 1, 0:1])

                    # dq = d_qg * exp_g * scale
                    mul(dqrow_ub[0:1, 0:D], dqgrow_ub[0:1, 0:D], expg_ub[0:1, 0:D], count=D)
                    muls(dqrow_ub[0:1, 0:D], dqrow_ub[0:1, 0:D], SCALE, count=D)
                    cast(dq_out_ub[r:r + 1, 0:D], dqrow_ub[0:1, 0:D], round_mode=RoundMode.TO_EVEN, count=D)
                    # dk_from_kg = d_kg * exp(g_last - g) ; k_exp = k * exp_g
                    mul(dkkg_ub[0:1, 0:D], dkgrow_ub[0:1, 0:D], explmg_ub[0:1, 0:D], count=D)
                    mul(kexprow_ub[0:1, 0:D], krow_ub[0:1, 0:D], expg_ub[0:1, 0:D], count=D)
                    cast(kexp_out_ub[r:r + 1, 0:D], kexprow_ub[0:1, 0:D], round_mode=RoundMode.TO_EVEN, count=D)
                    # dv = d_v_beta * beta
                    muls(tmp_ub[0:1, 0:D], dvbrow_ub[0:1, 0:D], beta_val, count=D)
                    cast(dv_out_ub[r:r + 1, 0:D], tmp_ub[0:1, 0:D], round_mode=RoundMode.TO_EVEN, count=D)

                    # dbeta = sum_D(d_v_beta * v) + sum_D(d_k_beta_g * k_exp)
                    mul(tmp_ub[0:1, 0:D], dvbrow_ub[0:1, 0:D], vrow_ub[0:1, 0:D], count=D)
                    cadd(dot_ub[0:1, 0:2], tmp_ub[0:1, 0:D], repeat=2, count_per_rep=64)
                    cadd(dot2_ub[0:1, 0:1], dot_ub[0:1, 0:2], repeat=1, count_per_rep=2)
                    sum_val.GetValueFrom(dot2_ub[0:1, 0:1])
                    mul(tmp_ub[0:1, 0:D], dkbgrow_ub[0:1, 0:D], kexprow_ub[0:1, 0:D], count=D)
                    cadd(dot_ub[0:1, 0:2], tmp_ub[0:1, 0:D], repeat=2, count_per_rep=64)
                    cadd(dot2_ub[0:1, 0:1], dot_ub[0:1, 0:2], repeat=1, count_per_rep=2)
                    dot_val.GetValueFrom(dot2_ub[0:1, 0:1])
                    sum_val.set(sum_val + dot_val)
                    sum_val.SetValueTo(dbeta_f_ub[r0 + r:r0 + r + 1, 0:1])

                    # dk = dk_from_kg + d_k_beta_g * beta * exp_g
                    muls(tmp_ub[0:1, 0:D], dkbgrow_ub[0:1, 0:D], beta_val, count=D)
                    mul(tmp_ub[0:1, 0:D], tmp_ub[0:1, 0:D], expg_ub[0:1, 0:D], count=D)
                    add(tmp_ub[0:1, 0:D], dkkg_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                    cast(dk_out_ub[r:r + 1, 0:D], tmp_ub[0:1, 0:D], round_mode=RoundMode.TO_EVEN, count=D)

                    # dg = q * dq - k * dk_from_kg + d_k_beta_g * k_exp * beta
                    mul(acc_ub[0:1, 0:D], qrow_ub[0:1, 0:D], dqrow_ub[0:1, 0:D], count=D)
                    mul(tmp_ub[0:1, 0:D], krow_ub[0:1, 0:D], dkkg_ub[0:1, 0:D], count=D)
                    sub(acc_ub[0:1, 0:D], acc_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                    add(term2_ub[0:1, 0:D], term2_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                    mul(tmp_ub[0:1, 0:D], dkbgrow_ub[0:1, 0:D], kexprow_ub[0:1, 0:D], count=D)
                    muls(tmp_ub[0:1, 0:D], tmp_ub[0:1, 0:D], beta_val, count=D)
                    add(dg_f_ub[r0 + r:r0 + r + 1, 0:D], acc_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)

                store_ready.set()
                store_ready.wait()
                dq_hv[tok:tok + HALF_L, hv_col:hv_col + D] <<= dq_out_ub[0:HALF_L, 0:D]
                dk_hv[tok:tok + HALF_L, hv_col:hv_col + D] <<= dk_out_ub[0:HALF_L, 0:D]
                dv[tok:tok + HALF_L, hv_col:hv_col + D] <<= dv_out_ub[0:HALF_L, 0:D]
                k_exp[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D] <<= kexp_out_ub[0:HALF_L, 0:D]
                store_done.set()
                store_done.wait()

            # dg[L-1] += term1 + term2 (complete only once every row has contributed)
            add(tmp_ub[0:1, 0:D], dglast_ub[0:1, 0:D], term2_ub[0:1, 0:D], count=D)
            add(dg_f_ub[L - 1:L, 0:D], dg_f_ub[L - 1:L, 0:D], tmp_ub[0:1, 0:D], count=D)
            cast(dg_b_ub[0:L, 0:D], dg_f_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_d)
            dg_core[row0:row0 + L, hv_col:hv_col + D] <<= dg_b_ub[0:L, 0:D]
            ub_to_gm_pad(dbeta[row0:row0 + L, hv_idx:hv_idx + 1], dbeta_f_ub[0:L, 0:1], L, 1, 0, HV - 1)

    return dq_hv, dk_hv, dv, dbeta, dg_core, k_exp
