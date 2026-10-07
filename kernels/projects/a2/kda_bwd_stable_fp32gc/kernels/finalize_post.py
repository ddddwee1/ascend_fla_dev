"""A2 backward finalize_post: fold the paired-decay contributions into dq/dk/dbeta/dg.

Port of this repository's ``a5/kda_bwd_stable/kernels/finalize_post.py`` (midpoint gate anchor) to the A2
(c220) facade. Arithmetic and order unchanged. Per chunk and value head, walking rows 63 -> 0 so that
``running`` is the reverse cumulative sum of ``dg'``::

    rscale = exp((g - g_last/2) * ln2)          cscale = exp((g_last/2 - g) * ln2)
    dq_pair = rscale * qk_left                  dk_pair = cscale * qk_right
    row_c   = rscale * s_base                   col_c   = cscale * t_beta
    dg_qk   = q_hv * dq_pair - k_hv * dk_pair
    dk_kk   = beta * row_c + col_c              dbeta_add = sum_D(k_hv * row_c)
    dg_kk   = beta * k_hv * row_c - k_hv * col_c
    dq_hv'  = dq_hv + dq_pair                   dk_hv' = dk_hv + dk_pair + dk_kk
    dbeta'  = bf16(dbeta + dbeta_add)           dg'    = dg_core + dg_qk + dg_kk ; dg[r] = running += dg'

A2 form: no ``@vf``; one tile op per row per term on 128-wide UB rows, the per-row dot product via ``cadd``
plus a scalar read, and per-row ``beta`` / ``dbeta`` through ``Var.GetValueFrom`` / ``SetValueTo``.
Public ``q``/``k``/``beta``/``g_cumsum`` and the public ``dbeta``/``dg`` are token-major 2-D views.

The chunk is walked in two halves of 32 rows, the upper half first, so that ``running`` still sweeps rows
63 -> 0. The cumulative gate stays FP32; other per-row inputs stay BF16 and are cast one row at a time: b3 has 192 KB of UB, and
holding a whole 64-row chunk of FP32 inputs does not fit.

D-PM-60: derived FP32 cumulative-gate ABI. Other arithmetic, ownership and synchronization are unchanged.
"""

import math

from ascriptor.a2 import *

L = 64

D = 128

HALF_L = L // 2

LN2 = math.log(2.0)

MID = 0.5


@kernel(mode="vec")
def finalize_post_a2_fp32gc_kernel(
    g_cumsum: GM[f32, ('BT', 'HVK')],
    qk_left: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    qk_right: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    s_base: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    t_beta: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    q: GM[bf16, ('BT', 'HK')],
    k: GM[bf16, ('BT', 'HK')],
    beta: GM[bf16, ('BT', 'HVB')],
    dq_hv_in: GM[bf16, ('BT', 'HVK')],
    dk_hv_in: GM[bf16, ('BT', 'HVK')],
    dbeta_in: GM[f32, ('BT', 'HVB')],
    dg_core_in: GM[bf16, ('BT', 'HVK')],
    dq_hv_out: GM[f32, ('B', 'HV', 'C', 64, 128)],
    dk_hv_out: GM[f32, ('B', 'HV', 'C', 64, 128)],
    dbeta_out: GM[bf16, ('BT', 'HVB')],
    dg_out: GM[bf16, ('BT', 'HVK')],
    B: i32,
    H: i32,
    HV: i32,
    C: i32,
):
    g_f_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    qkl_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    qkr_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    sbase_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    tbeta_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    qhv_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    khv_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dq_in_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dk_in_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dgc_in_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dq_out_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dk_out_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dg_f_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dg_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    gmid_ub = Tensor(DT.float, [1, D], Position.UB)
    rs_ub = Tensor(DT.float, [1, D], Position.UB)
    cs_ub = Tensor(DT.float, [1, D], Position.UB)
    sh_ub = Tensor(DT.float, [1, D], Position.UB)
    dqp_ub = Tensor(DT.float, [1, D], Position.UB)
    dkp_ub = Tensor(DT.float, [1, D], Position.UB)
    rowc_ub = Tensor(DT.float, [1, D], Position.UB)
    colc_ub = Tensor(DT.float, [1, D], Position.UB)
    qrow_ub = Tensor(DT.float, [1, D], Position.UB)
    krow_ub = Tensor(DT.float, [1, D], Position.UB)
    tmp_ub = Tensor(DT.float, [1, D], Position.UB)
    tmp2_ub = Tensor(DT.float, [1, D], Position.UB)
    acc_ub = Tensor(DT.float, [1, D], Position.UB)
    run_ub = Tensor(DT.float, [1, D], Position.UB)
    dot_ub = Tensor(DT.float, [1, 64], Position.UB)
    dot2_ub = Tensor(DT.float, [1, 64], Position.UB)
    # beta is staged through UB and cast on the vector unit: a scalar load straight out of BF16 memory
    # needs a scalar bf16 -> float cast, which the device compiler rejects ("not support bf16 type cast").
    # The 32-byte row pack is what gm_to_ub_pad produces for a strided single-element column.
    beta_b_ub = Tensor(DT.bfloat16, [L, 16], Position.UB)
    beta_f_ub = Tensor(DT.float, [L, 16], Position.UB)
    dbeta_f_ub = Tensor(DT.float, [L, 16], Position.UB)     # per-token scalar column, 32-byte row pack
    dbeta_b_ub = Tensor(DT.bfloat16, [L, 16], Position.UB)

    # Precautionary store fences. On the pinned library auto_sync emits the V -> MTE3 and MTE3 -> V guards
    # for some of this kernel's output staging buffers and not others (read off the generated vector
    # source). No divergence was traced to the gap: the device errors first suspected here turned out to
    # be the packed-column cast below. These two fences order the whole store group against the vector
    # work on either side of it, and are kept as insurance.
    store_ready = DEvent(Pipe.V, Pipe.MTE3)
    store_done = DEvent(Pipe.MTE3, Pipe.V)

    group = Var(HV // H)
    work_count = B * HV * C
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)
    beta_val = Var(0.0, dtype=DT.float)
    db_val = Var(0.0, dtype=DT.float)
    dot_val = Var(0.0, dtype=DT.float)
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
            gmid_ub[0:1, 0:D] <<= g_cumsum[row0 + L - 1:row0 + L, hv_col:hv_col + D]
            muls(gmid_ub[0:1, 0:D], gmid_ub[0:1, 0:D], MID, count=D)
            dup(run_ub[0:1, 0:D], 0.0, count=D)

            for rev_half in range(0, 2):
                half = Var(1 - rev_half)          # upper half first: running sweeps rows 63 -> 0
                r0 = Var(half * HALF_L)
                tok = Var(row0 + r0)

                g_f_ub[0:HALF_L, 0:D] <<= g_cumsum[tok:tok + HALF_L, hv_col:hv_col + D]
                qkl_b_ub[0:HALF_L, 0:D] <<= qk_left[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D]
                qkr_b_ub[0:HALF_L, 0:D] <<= qk_right[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D]
                sbase_b_ub[0:HALF_L, 0:D] <<= s_base[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D]
                tbeta_b_ub[0:HALF_L, 0:D] <<= t_beta[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D]
                qhv_b_ub[0:HALF_L, 0:D] <<= q[tok:tok + HALF_L, qk_col:qk_col + D]
                khv_b_ub[0:HALF_L, 0:D] <<= k[tok:tok + HALF_L, qk_col:qk_col + D]
                dq_in_b_ub[0:HALF_L, 0:D] <<= dq_hv_in[tok:tok + HALF_L, hv_col:hv_col + D]
                dk_in_b_ub[0:HALF_L, 0:D] <<= dk_hv_in[tok:tok + HALF_L, hv_col:hv_col + D]
                dgc_in_b_ub[0:HALF_L, 0:D] <<= dg_core_in[tok:tok + HALF_L, hv_col:hv_col + D]

                for rt in range(0, HALF_L):
                    r = HALF_L - 1 - rt

                    sub(sh_ub[0:1, 0:D], g_f_ub[r:r + 1, 0:D], gmid_ub[0:1, 0:D], count=D)
                    muls(sh_ub[0:1, 0:D], sh_ub[0:1, 0:D], LN2, count=D)
                    exp(rs_ub[0:1, 0:D], sh_ub[0:1, 0:D], count=D)
                    muls(cs_ub[0:1, 0:D], sh_ub[0:1, 0:D], -1.0, count=D)
                    exp(cs_ub[0:1, 0:D], cs_ub[0:1, 0:D], count=D)

                    cast(tmp_ub[0:1, 0:D], qkl_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    mul(dqp_ub[0:1, 0:D], rs_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                    cast(tmp_ub[0:1, 0:D], qkr_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    mul(dkp_ub[0:1, 0:D], cs_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                    cast(tmp_ub[0:1, 0:D], sbase_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    mul(rowc_ub[0:1, 0:D], rs_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                    cast(tmp_ub[0:1, 0:D], tbeta_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    mul(colc_ub[0:1, 0:D], cs_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                    cast(qrow_ub[0:1, 0:D], qhv_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    cast(krow_ub[0:1, 0:D], khv_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    beta_val.GetValueFrom(beta_f_ub[r0 + r:r0 + r + 1, 0:1])

                    # dg_qk = q_hv*dq_pair - k_hv*dk_pair
                    mul(acc_ub[0:1, 0:D], qrow_ub[0:1, 0:D], dqp_ub[0:1, 0:D], count=D)
                    mul(tmp_ub[0:1, 0:D], krow_ub[0:1, 0:D], dkp_ub[0:1, 0:D], count=D)
                    sub(acc_ub[0:1, 0:D], acc_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)

                    # dq_hv' = dq_hv + dq_pair
                    cast(tmp_ub[0:1, 0:D], dq_in_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    add(dq_out_ub[r:r + 1, 0:D], tmp_ub[0:1, 0:D], dqp_ub[0:1, 0:D], count=D)

                    # dk_kk = beta*row_c + col_c ; dk_hv' = dk_hv + dk_pair + dk_kk
                    muls(tmp_ub[0:1, 0:D], rowc_ub[0:1, 0:D], beta_val, count=D)
                    add(tmp_ub[0:1, 0:D], tmp_ub[0:1, 0:D], colc_ub[0:1, 0:D], count=D)
                    cast(tmp2_ub[0:1, 0:D], dk_in_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    add(dk_out_ub[r:r + 1, 0:D], tmp2_ub[0:1, 0:D], dkp_ub[0:1, 0:D], count=D)
                    add(dk_out_ub[r:r + 1, 0:D], dk_out_ub[r:r + 1, 0:D], tmp_ub[0:1, 0:D], count=D)

                    # dbeta_add = sum_D(k_hv * row_c): two-stage cadd then a scalar read
                    mul(tmp_ub[0:1, 0:D], krow_ub[0:1, 0:D], rowc_ub[0:1, 0:D], count=D)
                    cadd(dot_ub[0:1, 0:2], tmp_ub[0:1, 0:D], repeat=2, count_per_rep=64)
                    cadd(dot2_ub[0:1, 0:1], dot_ub[0:1, 0:2], repeat=1, count_per_rep=2)
                    dot_val.GetValueFrom(dot2_ub[0:1, 0:1])
                    db_val.GetValueFrom(dbeta_in[tok + r:tok + r + 1, hv_idx:hv_idx + 1])
                    db_val.set(db_val + dot_val)
                    db_val.SetValueTo(dbeta_f_ub[r0 + r:r0 + r + 1, 0:1])

                    # dg_kk = beta*k_hv*row_c - k_hv*col_c
                    mul(tmp_ub[0:1, 0:D], krow_ub[0:1, 0:D], rowc_ub[0:1, 0:D], count=D)
                    muls(tmp_ub[0:1, 0:D], tmp_ub[0:1, 0:D], beta_val, count=D)
                    add(acc_ub[0:1, 0:D], acc_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                    mul(tmp_ub[0:1, 0:D], krow_ub[0:1, 0:D], colc_ub[0:1, 0:D], count=D)
                    sub(acc_ub[0:1, 0:D], acc_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)

                    # dg' = dg_core + dg_qk + dg_kk ; running += dg' ; dg[r] = running
                    cast(tmp_ub[0:1, 0:D], dgc_in_b_ub[r:r + 1, 0:D], round_mode=RoundMode.NONE, count=D)
                    add(acc_ub[0:1, 0:D], acc_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                    add(run_ub[0:1, 0:D], run_ub[0:1, 0:D], acc_ub[0:1, 0:D], count=D)
                    muls(dg_f_ub[r:r + 1, 0:D], run_ub[0:1, 0:D], 1.0, count=D)  # exact copy of the running sum

                cast(dg_b_ub[0:HALF_L, 0:D], dg_f_ub[0:HALF_L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_hd)
                store_ready.set()
                store_ready.wait()
                dg_out[tok:tok + HALF_L, hv_col:hv_col + D] <<= dg_b_ub[0:HALF_L, 0:D]
                dq_hv_out[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D] <<= dq_out_ub[0:HALF_L, 0:D]
                dk_hv_out[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D] <<= dk_out_ub[0:HALF_L, 0:D]
                store_done.set()
                store_done.wait()

            for r in range(0, L):  # one cast per packed row: the column is 32-byte strided, not contiguous
                cast(dbeta_b_ub[r:r + 1, 0:1], dbeta_f_ub[r:r + 1, 0:1], round_mode=RoundMode.TO_EVEN, count=1)
            ub_to_gm_pad(dbeta_out[row0:row0 + L, hv_idx:hv_idx + 1], dbeta_b_ub[0:L, 0:1], L, 1, 0, HV - 1)

    return dq_hv_out, dk_hv_out, dbeta_out, dg_out
