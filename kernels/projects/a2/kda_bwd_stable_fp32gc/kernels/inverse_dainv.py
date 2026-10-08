"""A2 backward inverse_dainv: the gradient that flows back through the WY inverse.

Port of ``a5/kda_bwd/kernels/inverse_dainv.py`` to the A2 (c220) facade. Per chunk and value head::

    dtri[i, j] = beta[j] * (dv @ v^T + d_w @ k_exp^T)[i, j]   for j < i, else 0

``beta`` scales by **column**, not by row: the A5 vector body gathers beta into the lane vector and
multiplies elementwise along the row, i.e. ``dA_inv * beta[None, :]``.

A2 differences:

* no ``@vf``: the row loop is UB instructions, the beta lane vector is built once per chunk by reading 64
  scalars with ``Var.GetValueFrom`` into a UB row,
  and the strictly-lower mask is an in-kernel column index plus ``compare_scalar`` / ``select`` (the same form as
  the A2 forward ``intra``), not a host-made mask tensor.
* no ``dma.l0c_to_ub`` on b3: the two FP32 [64, 64] products cross to the vector side through a two-slot GM ring
  under ``CvMutex`` (FIX -> MTE2), keeping FP32 all the way to the BF16 output cast.
* ``dv`` / ``v`` / ``beta`` are read in their public token-major BF16 layout as 2-D views; ``d_w`` / ``k_exp`` / ``dtri``
  stay chain-internal ``[B, HV, C, 64, *]``.
* ``splitn`` only, ``is_init=True``: no accumulate chain, so no settle barrier is required here (A2-01).
"""

from ascriptor.a2 import *

L = 64

D = 128

HALF_L = L // 2


@kernel()
def inverse_dainv_a2_kernel(
    dv: GM[bf16, ('BT', 'HVK')],
    v: GM[bf16, ('BT', 'HVK')],
    d_w: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    k_exp: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    beta: GM[bf16, ('BT', 'HVB')],
    dtri: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    B: i32,
    HV: i32,
    C: i32,
):
    a_ws = GMBuff(DT.float, [L, L], slots=2, name="dainv_a_ws")
    b_ws = GMBuff(DT.float, [L, L], slots=2, name="dainv_b_ws")
    cvmutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1_dv = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_v = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_dw = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_kexp = DBuff(DT.bfloat16, [L, D], Position.L1)
    l0c_a = DBuff(DT.float, [L, L], Position.L0C)
    l0c_b = DBuff(DT.float, [L, L], Position.L0C)

    a_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    b_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    m_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    out_ub = Tensor(DT.bfloat16, [HALF_L, L], Position.UB)
    col_ub = Tensor(DT.float, [1, L], Position.UB)
    zero_ub = Tensor(DT.float, [1, L], Position.UB)
    # beta is staged through UB and cast on the vector unit: a scalar load straight out of BF16 memory
    # needs a scalar bf16 -> float cast, which the device compiler rejects ("not support bf16 type cast").
    # The 32-byte row pack is what gm_to_ub_pad produces for a strided single-element column.
    beta_b_ub = Tensor(DT.bfloat16, [L, 16], Position.UB)
    beta_f_ub = Tensor(DT.float, [L, 16], Position.UB)
    beta_ub = Tensor(DT.float, [1, L], Position.UB)
    pred_ub = Tensor(DT.uint8, [1, 32], Position.UB)

    # Precautionary: auto_sync emits the V -> S guard for beta_f_ub and the V -> S write-after-read guard
    # for beta_ub, but no S -> V read-after-write guard for the vector mul below. No divergence was traced
    # to this; the beta error seen on the device was the packed-column cast above.
    beta_ready = DEvent(Pipe.S, Pipe.V)

    work_count = B * HV * C
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    cube_cnt = Var(0)
    col_val = Var(0.0, dtype=DT.float)
    row_val = Var(0.0, dtype=DT.float)
    beta_val = Var(0.0, dtype=DT.float)

    with auto_sync():
        with vec_scope():
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
            chunk_tok = Var(b_idx * C * L + c_idx * L)
            hv_col = Var(hv_idx * D)
            row_begin = Var(GetSubBlockIdx() * HALF_L)
            row_end = Var(row_begin + HALF_L)

            l1_dv[cube_cnt][0:L, 0:D] <<= dv[chunk_tok:chunk_tok + L, hv_col:hv_col + D]
            l1_v[cube_cnt][0:L, 0:D] <<= v[chunk_tok:chunk_tok + L, hv_col:hv_col + D]
            l1_dw[cube_cnt][0:L, 0:D] <<= d_w[b_idx, hv_idx, c_idx, 0:L, 0:D]
            l1_kexp[cube_cnt][0:L, 0:D] <<= k_exp[b_idx, hv_idx, c_idx, 0:L, 0:D]

            matmul(l0c_a[cube_cnt], l1_dv[cube_cnt], l1_v[cube_cnt], m=L, n=L, k=D, splitn=L)
            matmul(l0c_b[cube_cnt], l1_dw[cube_cnt], l1_kexp[cube_cnt], m=L, n=L, k=D, splitn=L)
            cvmutex.lock()
            a_ws[work][0:L, 0:L] <<= l0c_a[cube_cnt]
            b_ws[work][0:L, 0:L] <<= l0c_b[cube_cnt]
            cvmutex.ready()
            cube_cnt += 1

            cvmutex.wait()
            a_ub[0:HALF_L, 0:L] <<= a_ws[work][row_begin:row_end, 0:L]
            b_ub[0:HALF_L, 0:L] <<= b_ws[work][row_begin:row_end, 0:L]
            cvmutex.free()
            add(a_ub[0:HALF_L, 0:L], a_ub[0:HALF_L, 0:L], b_ub[0:HALF_L, 0:L], count=HALF_L * L)
            gm_to_ub_pad(beta_b_ub[0:L, 0:1], beta[chunk_tok:chunk_tok + L, hv_idx:hv_idx + 1], L, 1, HV - 1, 0)
            # one cast per packed row. A single whole-tile cast over the pack lowers to a vconv with a zero
            # source block stride on c220, which replicates row 0's block into every row: on the device every
            # token then carried beta[0]. The functional simulator does not model the block strides.
            for pack_row in range(0, L):
                cast(beta_f_ub[pack_row:pack_row + 1, 0:1], beta_b_ub[pack_row:pack_row + 1, 0:1],
                     round_mode=RoundMode.NONE, count=1)
            with vec_scope():
                for j in range(0, L):
                    beta_val.GetValueFrom(beta_f_ub[j:j + 1, 0:1])
                    beta_val.SetValueTo(beta_ub[0:1, j:j + 1])
            beta_ready.set()
            beta_ready.wait()
            row_val.set(row_begin)
            for r in range(0, HALF_L):
                mul(a_ub[r:r + 1, 0:L], a_ub[r:r + 1, 0:L], beta_ub[0:1, 0:L], count=L)
                compare_scalar(pred_ub[0:1, 0:32], col_ub[0:1, 0:L], row_val, CompareMode.LT)
                select(m_ub[r:r + 1, 0:L], pred_ub[0:1, 0:32], a_ub[r:r + 1, 0:L], zero_ub[0:1, 0:L],
                       SelectMode.TENSOR_SCALAR)
                row_val.set(row_val + 1.0)
            cast(out_ub[0:HALF_L, 0:L], m_ub[0:HALF_L, 0:L], round_mode=RoundMode.TO_EVEN, count=HALF_L * L)
            dtri[b_idx, hv_idx, c_idx, row_begin:row_end, 0:L] <<= out_ub[0:HALF_L, 0:L]

    return dtri
