"""A2 backward inverse_dakk_fused: push ``dtri`` back through the two inverse factors.

Port of ``a5/kda_bwd/kernels/inverse_dakk_fused.py`` to the A2 (c220) facade. Per chunk and value head::

    tmp   = dtri @ Akk^T                 (FP32 product, rounded to BF16 for the second matmul)
    raw   = Akk^T @ tmp^T
    dAkk  = -tril(raw, -1)               (negate the strictly-lower triangle, zero the rest)

A2 differences:

* no ``@vf``: the cast is a tile ``cast`` and the mask is an in-kernel column index with ``compare_scalar`` /
  ``select`` against a zero row, as in the A2 forward ``intra``.
* neither ``dma.l0c_to_ub`` nor ``dma.ub_to_l1`` exists on b3, so the round trip that A5 does inside the core
  (L0C -> UB -> L1) goes through two GM rings here: FP32 out on ``CvMutex`` (FIX -> MTE2), BF16 back on
  ``VcMutex`` (MTE3 -> MTE2). A third ring carries the second product back to the vector side.
* ``Akk`` and ``dAkk`` are read and written on the token-major ``[B*T, HV*64]`` seam (``dAkk`` feeds
  ``finalize_pre``); ``dtri`` is chain-internal ``[B, HV, C, 64, 64]`` from ``inverse_dainv``.
* ``splitn`` only, ``is_init=True``: no accumulate chain, so no settle barrier is required here (A2-01).
"""

from ascriptor.a2 import *

L = 64

HALF_L = L // 2


@kernel()
def inverse_dakk_fused_a2_kernel(
    Akk: GM[bf16, ('BT', 'HVL')],
    dtri: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    dAkk: GM[bf16, ('BT', 'HVL')],
    B: i32,
    HV: i32,
    C: i32,
):
    tmp_ws = GMBuff(DT.float, [L, L], slots=2, name="dakk_tmp_ws")
    tmpb_ws = GMBuff(DT.bfloat16, [L, L], slots=2, name="dakk_tmpb_ws")
    raw_ws = GMBuff(DT.float, [L, L], slots=2, name="dakk_raw_ws")
    tmp_mx = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    bridge_mx = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    raw_mx = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1_akk = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_dtri = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_tmp = DBuff(DT.bfloat16, [L, L], Position.L1)
    l0c_tmp = DBuff(DT.float, [L, L], Position.L0C)
    l0c_raw = DBuff(DT.float, [L, L], Position.L0C)

    tmp_f_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    tmp_b_ub = Tensor(DT.bfloat16, [HALF_L, L], Position.UB)
    raw_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    out_f_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    out_b_ub = Tensor(DT.bfloat16, [HALF_L, L], Position.UB)
    col_ub = Tensor(DT.float, [1, L], Position.UB)
    zero_ub = Tensor(DT.float, [1, L], Position.UB)
    pred_ub = Tensor(DT.uint8, [1, 32], Position.UB)

    work_count = B * HV * C
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    cube_cnt = Var(0)
    col_val = Var(0.0, dtype=DT.float)
    row_val = Var(0.0, dtype=DT.float)
    n_half = HALF_L * L

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
            l_col = Var(hv_idx * L)
            row_begin = Var(GetSubBlockIdx() * HALF_L)
            row_end = Var(row_begin + HALF_L)

            l1_akk[cube_cnt][0:L, 0:L] <<= Akk[chunk_tok:chunk_tok + L, l_col:l_col + L]
            l1_dtri[cube_cnt][0:L, 0:L] <<= dtri[b_idx, hv_idx, c_idx, 0:L, 0:L]
            matmul(l0c_tmp[cube_cnt], l1_dtri[cube_cnt], l1_akk[cube_cnt], m=L, n=L, k=L, splitn=L)
            tmp_mx.lock()
            tmp_ws[work][0:L, 0:L] <<= l0c_tmp[cube_cnt]
            tmp_mx.ready()

            # vector: round the FP32 product to BF16 for the second matmul
            tmp_mx.wait()
            tmp_f_ub[0:HALF_L, 0:L] <<= tmp_ws[work][row_begin:row_end, 0:L]
            tmp_mx.free()
            cast(tmp_b_ub[0:HALF_L, 0:L], tmp_f_ub[0:HALF_L, 0:L], round_mode=RoundMode.TO_EVEN, count=n_half)
            bridge_mx.lock()
            tmpb_ws[work][row_begin:row_end, 0:L] <<= tmp_b_ub[0:HALF_L, 0:L]
            bridge_mx.ready()

            bridge_mx.wait()
            l1_tmp[cube_cnt][0:L, 0:L] <<= tmpb_ws[work][0:L, 0:L]
            bridge_mx.free()
            matmul(l0c_raw[cube_cnt], l1_akk[cube_cnt].T, l1_tmp[cube_cnt].T, m=L, n=L, k=L, splitn=L)
            raw_mx.lock()
            raw_ws[work][0:L, 0:L] <<= l0c_raw[cube_cnt]
            raw_mx.ready()
            cube_cnt += 1

            # vector: dAkk = -tril(raw, -1)
            raw_mx.wait()
            raw_ub[0:HALF_L, 0:L] <<= raw_ws[work][row_begin:row_end, 0:L]
            raw_mx.free()
            muls(raw_ub[0:HALF_L, 0:L], raw_ub[0:HALF_L, 0:L], -1.0, count=n_half)
            row_val.set(row_begin)
            for r in range(0, HALF_L):
                compare_scalar(pred_ub[0:1, 0:32], col_ub[0:1, 0:L], row_val, CompareMode.LT)
                select(out_f_ub[r:r + 1, 0:L], pred_ub[0:1, 0:32], raw_ub[r:r + 1, 0:L], zero_ub[0:1, 0:L],
                       SelectMode.TENSOR_SCALAR)
                row_val.set(row_val + 1.0)
            cast(out_b_ub[0:HALF_L, 0:L], out_f_ub[0:HALF_L, 0:L], round_mode=RoundMode.TO_EVEN, count=n_half)
            dAkk[chunk_tok + row_begin:chunk_tok + row_end, l_col:l_col + L] <<= out_b_ub[0:HALF_L, 0:L]

    return dAkk
