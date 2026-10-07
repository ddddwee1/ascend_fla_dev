"""A2 backward finalize_pre: gate-anchored q/k scalings and the three masked dA matrices.

Port of this repository's ``a5/kda_bwd_stable/kernels/finalize_pre.py`` (midpoint anchor, which doubles the
usable gate span) to the A2 (c220) facade. Arithmetic and order unchanged, per chunk and value head::

    gmid   = g_last / 2
    rscale = exp((g - gmid) * ln2)      cscale = exp((gmid - g) * ln2)
    q_scaled = bf16(q * rscale)  k_scaled = bf16(k * rscale)  kg = bf16(k * cscale)
    M_qk   = tril_incl(dAqk)     M_base = tril_strict(dAkk)   M_beta = M_base * beta[row]

A2 form: no ``@vf``; whole-tile ops with one op per row where a row-broadcast or a per-row scalar is needed;
masking by an in-kernel column index with ``compare_scalar`` + ``select`` (never multiply by zero, so an
overflowed value cannot become NaN). Public ``q``/``k``/``beta`` and the chain caches are read token-major
through 2-D views; the GQA head map is in-kernel.

The chunk is processed in two halves of 32 rows. Every row here is independent (the only shared value is the
chunk's last gate row), and b3 has 192 KB of UB: holding a whole 64-row chunk in FP32 does not fit, while two
half passes leave room to spare. One BF16 staging buffer serves q/k loads; gates load directly as FP32 and become
``cscale`` in place.

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
def finalize_pre_a2_fp32gc_kernel(
    q: GM[bf16, ('BT', 'HK')],
    k: GM[bf16, ('BT', 'HK')],
    g_cumsum: GM[f32, ('BT', 'HVK')],
    beta: GM[bf16, ('BT', 'HVB')],
    dAqk: GM[bf16, ('BT', 'HVL')],
    dAkk: GM[bf16, ('BT', 'HVL')],
    q_scaled: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    k_scaled: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    kg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    m_qk: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    m_base: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    m_beta: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    B: i32,
    H: i32,
    HV: i32,
    C: i32,
):
    stage_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    out_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    q_f_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    k_f_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    rs_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    cs_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    gmid_ub = Tensor(DT.float, [1, D], Position.UB)
    stage_l_ub = Tensor(DT.bfloat16, [HALF_L, L], Position.UB)
    a_f_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    m_f_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    # beta is staged through UB and cast on the vector unit: a scalar load straight out of BF16 memory
    # needs a scalar bf16 -> float cast, which the device compiler rejects ("not support bf16 type cast").
    # The 32-byte row pack is what gm_to_ub_pad produces for a strided single-element column.
    beta_b_ub = Tensor(DT.bfloat16, [L, 16], Position.UB)
    beta_f_ub = Tensor(DT.float, [L, 16], Position.UB)
    col_ub = Tensor(DT.float, [1, L], Position.UB)
    zero_ub = Tensor(DT.float, [1, L], Position.UB)
    pred_ub = Tensor(DT.uint8, [1, 32], Position.UB)

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
    col_val = Var(0.0, dtype=DT.float)
    row_val = Var(0.0, dtype=DT.float)
    beta_val = Var(0.0, dtype=DT.float)
    n_hd = HALF_L * D
    n_hl = HALF_L * L

    with auto_sync():
        col_val.set(0.0)
        for j in range(0, L):
            col_val.SetValueTo(col_ub[0:1, j:j + 1])
            col_val.set(col_val + 1.0)
        dup(zero_ub[0:1, 0:L], 0.0, count=L)

        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            bhv = Var(work // C)
            hv_idx = Var(bhv % HV)
            b_idx = Var(bhv // HV)
            h_idx = Var(hv_idx // group)
            row0 = Var(b_idx * C * L + c_idx * L)
            qk_col = Var(h_idx * D)
            g_col = Var(hv_idx * D)
            l_col = Var(hv_idx * L)

            gm_to_ub_pad(beta_b_ub[0:L, 0:1], beta[row0:row0 + L, hv_idx:hv_idx + 1], L, 1, HV - 1, 0)
            # one cast per packed row. A single whole-tile cast over the pack lowers to a vconv with a zero
            # source block stride on c220, which replicates row 0's block into every row: on the device every
            # token then carried beta[0]. The functional simulator does not model the block strides.
            for pack_row in range(0, L):
                cast(beta_f_ub[pack_row:pack_row + 1, 0:1], beta_b_ub[pack_row:pack_row + 1, 0:1],
                     round_mode=RoundMode.NONE, count=1)
            gmid_ub[0:1, 0:D] <<= g_cumsum[row0 + L - 1:row0 + L, g_col:g_col + D]
            muls(gmid_ub[0:1, 0:D], gmid_ub[0:1, 0:D], MID, count=D)

            for half in range(0, 2):
                r0 = Var(half * HALF_L)
                tok = Var(row0 + r0)

                # rscale = exp((g - gmid) * ln2), cscale = exp((gmid - g) * ln2)
                cs_ub[0:HALF_L, 0:D] <<= g_cumsum[tok:tok + HALF_L, g_col:g_col + D]
                for r in range(0, HALF_L):
                    sub(cs_ub[r:r + 1, 0:D], cs_ub[r:r + 1, 0:D], gmid_ub[0:1, 0:D], count=D)
                muls(cs_ub[0:HALF_L, 0:D], cs_ub[0:HALF_L, 0:D], LN2, count=n_hd)
                exp(rs_ub[0:HALF_L, 0:D], cs_ub[0:HALF_L, 0:D], count=n_hd)
                muls(cs_ub[0:HALF_L, 0:D], cs_ub[0:HALF_L, 0:D], -1.0, count=n_hd)
                exp(cs_ub[0:HALF_L, 0:D], cs_ub[0:HALF_L, 0:D], count=n_hd)

                # q_scaled = q * rscale
                stage_ub[0:HALF_L, 0:D] <<= q[tok:tok + HALF_L, qk_col:qk_col + D]
                cast(q_f_ub[0:HALF_L, 0:D], stage_ub[0:HALF_L, 0:D], round_mode=RoundMode.NONE, count=n_hd)
                mul(q_f_ub[0:HALF_L, 0:D], q_f_ub[0:HALF_L, 0:D], rs_ub[0:HALF_L, 0:D], count=n_hd)
                cast(out_ub[0:HALF_L, 0:D], q_f_ub[0:HALF_L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_hd)
                store_ready.set()
                store_ready.wait()
                q_scaled[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D] <<= out_ub[0:HALF_L, 0:D]
                store_done.set()
                store_done.wait()

                # kg = k * cscale (consumes cscale), then k_scaled = k * rscale
                stage_ub[0:HALF_L, 0:D] <<= k[tok:tok + HALF_L, qk_col:qk_col + D]
                cast(k_f_ub[0:HALF_L, 0:D], stage_ub[0:HALF_L, 0:D], round_mode=RoundMode.NONE, count=n_hd)
                mul(cs_ub[0:HALF_L, 0:D], k_f_ub[0:HALF_L, 0:D], cs_ub[0:HALF_L, 0:D], count=n_hd)
                cast(out_ub[0:HALF_L, 0:D], cs_ub[0:HALF_L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_hd)
                store_ready.set()
                store_ready.wait()
                kg[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D] <<= out_ub[0:HALF_L, 0:D]
                store_done.set()
                store_done.wait()
                mul(k_f_ub[0:HALF_L, 0:D], k_f_ub[0:HALF_L, 0:D], rs_ub[0:HALF_L, 0:D], count=n_hd)
                cast(out_ub[0:HALF_L, 0:D], k_f_ub[0:HALF_L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_hd)
                store_ready.set()
                store_ready.wait()
                k_scaled[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:D] <<= out_ub[0:HALF_L, 0:D]
                store_done.set()
                store_done.wait()

                # M_qk = tril including the diagonal
                stage_l_ub[0:HALF_L, 0:L] <<= dAqk[tok:tok + HALF_L, l_col:l_col + L]
                cast(a_f_ub[0:HALF_L, 0:L], stage_l_ub[0:HALF_L, 0:L], round_mode=RoundMode.NONE, count=n_hl)
                row_val.set(r0)
                for r in range(0, HALF_L):
                    compare_scalar(pred_ub[0:1, 0:32], col_ub[0:1, 0:L], row_val, CompareMode.LE)
                    select(m_f_ub[r:r + 1, 0:L], pred_ub[0:1, 0:32], a_f_ub[r:r + 1, 0:L], zero_ub[0:1, 0:L],
                           SelectMode.TENSOR_SCALAR)
                    row_val.set(row_val + 1.0)
                cast(stage_l_ub[0:HALF_L, 0:L], m_f_ub[0:HALF_L, 0:L], round_mode=RoundMode.TO_EVEN, count=n_hl)
                store_ready.set()
                store_ready.wait()
                m_qk[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:L] <<= stage_l_ub[0:HALF_L, 0:L]
                store_done.set()
                store_done.wait()

                # M_base = strictly lower, M_beta = M_base * beta[row] (built in place over the consumed rows)
                stage_l_ub[0:HALF_L, 0:L] <<= dAkk[tok:tok + HALF_L, l_col:l_col + L]
                cast(a_f_ub[0:HALF_L, 0:L], stage_l_ub[0:HALF_L, 0:L], round_mode=RoundMode.NONE, count=n_hl)
                row_val.set(r0)
                for r in range(0, HALF_L):
                    compare_scalar(pred_ub[0:1, 0:32], col_ub[0:1, 0:L], row_val, CompareMode.LT)
                    select(m_f_ub[r:r + 1, 0:L], pred_ub[0:1, 0:32], a_f_ub[r:r + 1, 0:L], zero_ub[0:1, 0:L],
                           SelectMode.TENSOR_SCALAR)
                    beta_val.GetValueFrom(beta_f_ub[r0 + r:r0 + r + 1, 0:1])
                    muls(a_f_ub[r:r + 1, 0:L], m_f_ub[r:r + 1, 0:L], beta_val, count=L)
                    row_val.set(row_val + 1.0)
                cast(stage_l_ub[0:HALF_L, 0:L], m_f_ub[0:HALF_L, 0:L], round_mode=RoundMode.TO_EVEN, count=n_hl)
                store_ready.set()
                store_ready.wait()
                m_base[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:L] <<= stage_l_ub[0:HALF_L, 0:L]
                store_done.set()
                store_done.wait()
                cast(stage_l_ub[0:HALF_L, 0:L], a_f_ub[0:HALF_L, 0:L], round_mode=RoundMode.TO_EVEN, count=n_hl)
                store_ready.set()
                store_ready.wait()
                m_beta[b_idx, hv_idx, c_idx, r0:r0 + HALF_L, 0:L] <<= stage_l_ub[0:HALF_L, 0:L]
                store_done.set()
                store_done.wait()

    return q_scaled, k_scaled, kg, m_qk, m_base, m_beta
