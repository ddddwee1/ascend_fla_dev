"""A2 backward inverse_mm: the five WY-inverse gradient products.

Port of this repository's ``a5/kda_bwd_stable/kernels/inverse_mm.py`` (the A5K-02 / D-PM-24 bounded variant)
to the A2 (c220) facade. Products, per chunk and value head::

    d_vh    = dv  @ h            d_w        = -d_vh              (the negation is in-kernel here, see below)
    d_qg    = do  @ h  (``grad_out``)            d_v_beta   = Akk^T @ dv
    d_kg    = vnew @ dh          d_k_beta_g = Akk^T @ d_w

A2 differences:

* ``dma.l0c_to_ub`` does not exist on b3, so the A5 drain of every product through UB is replaced by direct
  ``L0C -> GM`` writes (the fixpipe does the FP32 -> BF16 conversion) for ``d_qg`` / ``d_v_beta`` / ``d_kg``,
  and by a two-slot GM ring for ``d_vh``, which the vector side needs.
* **D-PM-37**: the A5 unit negated this kernel's ``d_vh`` output on the host (``-result[2]`` in
  ``stages.inverse_mm``). Here the vector side negates it, writes ``d_w`` as the kernel's own output, and
  bridges the same tile back to the cube for the last product; no host arithmetic.
* Five FP32 [64, 128] L0C tiles would be 160 KB, over the 128 KB L0C of b3, so the products are sequenced
  through a two-slot L0C buffer instead of living in five separate ones.
* All four ``matmul`` are ``splitn`` with ``is_init=True``: no accumulate chain, so no settle barrier is
  required here (A2-01).
* ``do`` / ``vnew`` / ``dv`` / ``Akk`` are read from their token-major public layout as 2-D views.
"""

from ascriptor.a2 import *

L = 64

D = 128

HALF_L = L // 2


@kernel()
def inverse_mm_a2_kernel(
    grad_out: GM[bf16, ('BT', 'HVK')],  # named grad_out, not do: the ACLNN API emits the name as a C++ identifier
    vnew: GM[bf16, ('BT', 'HVK')],
    dv: GM[bf16, ('BT', 'HVK')],
    h: GM[bf16, ('B', 'C', 'HV', 128, 128)],
    dh: GM[bf16, ('B', 'C', 'HV', 128, 128)],
    Akk: GM[bf16, ('BT', 'HVL')],
    d_qg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    d_kg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    d_w: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    d_v_beta: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    d_k_beta_g: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    B: i32,
    HV: i32,
    C: i32,
):
    dvh_ws = GMBuff(DT.float, [L, D], slots=2, name="dvh_ws")
    dw_ws = GMBuff(DT.bfloat16, [L, D], slots=2, name="dw_ws")
    dvh_mx = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    dw_mx = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)

    l1_do = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_vnew = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_dv = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_h = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_dh = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_akk = Tensor(DT.bfloat16, [L, L], Position.L1)
    l1_dw = Tensor(DT.bfloat16, [L, D], Position.L1)
    l0c = DBuff(DT.float, [L, D], Position.L0C)

    dvh_f_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dw_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)

    # Precautionary L1 load fence. On the pinned library auto_sync's MTE2 -> MTE1 ready guard does not
    # name every L1 buffer of this kernel (read off the generated cube source), so in principle a matmul
    # could read a tile before its GM load landed. A controlled A/B on a 910B3 (same case, same build,
    # only these two fences removed) produced bitwise identical output, so no consequence has been
    # observed; the fence is kept as insurance, not as a fix for a measured defect.
    l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)

    work_count = B * HV * C
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    l0c_cnt = Var(0)
    n_half = HALF_L * D

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            bhv = Var(work // C)
            hv_idx = Var(bhv % HV)
            b_idx = Var(bhv // HV)
            row0 = Var(b_idx * C * L + c_idx * L)
            hv_col = Var(hv_idx * D)
            l_col = Var(hv_idx * L)
            rowl = Var(GetSubBlockIdx() * HALF_L)

            l1_dv <<= dv[row0:row0 + L, hv_col:hv_col + D]
            l1_do <<= grad_out[row0:row0 + L, hv_col:hv_col + D]
            l1_vnew <<= vnew[row0:row0 + L, hv_col:hv_col + D]
            l1_akk <<= Akk[row0:row0 + L, l_col:l_col + L]
            l1_h <<= h[b_idx, c_idx, hv_idx, 0:D, 0:D]
            l1_dh <<= dh[b_idx, c_idx, hv_idx, 0:D, 0:D]
            l1_ready.set()
            l1_ready.wait()

            # d_qg = do @ h
            matmul(l0c[l0c_cnt], l1_do, l1_h, m=L, n=D, k=D, splitn=D)
            d_qg[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= l0c[l0c_cnt]
            l0c_cnt += 1
            # d_kg = vnew @ dh
            matmul(l0c[l0c_cnt], l1_vnew, l1_dh, m=L, n=D, k=D, splitn=D)
            d_kg[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= l0c[l0c_cnt]
            l0c_cnt += 1
            # d_v_beta = Akk^T @ dv
            matmul(l0c[l0c_cnt], l1_akk.T, l1_dv.T, m=L, n=D, k=L, splitn=D)
            d_v_beta[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= l0c[l0c_cnt]
            l0c_cnt += 1
            # d_vh = dv @ h, handed to the vector side in FP32
            matmul(l0c[l0c_cnt], l1_dv, l1_h, m=L, n=D, k=D, splitn=D)
            dvh_mx.lock()
            dvh_ws[work][0:L, 0:D] <<= l0c[l0c_cnt]
            dvh_mx.ready()
            l0c_cnt += 1

            # vector: d_w = -d_vh, written out and bridged back to the cube
            dvh_mx.wait()
            dvh_f_ub[0:HALF_L, 0:D] <<= dvh_ws[work][rowl:rowl + HALF_L, 0:D]
            dvh_mx.free()
            muls(dvh_f_ub[0:HALF_L, 0:D], dvh_f_ub[0:HALF_L, 0:D], -1.0, count=n_half)
            cast(dw_b_ub[0:HALF_L, 0:D], dvh_f_ub[0:HALF_L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_half)
            d_w[b_idx, hv_idx, c_idx, rowl:rowl + HALF_L, 0:D] <<= dw_b_ub[0:HALF_L, 0:D]
            dw_mx.lock()
            dw_ws[work][rowl:rowl + HALF_L, 0:D] <<= dw_b_ub[0:HALF_L, 0:D]
            dw_mx.ready()

            # d_k_beta_g = Akk^T @ d_w
            dw_mx.wait()
            l1_dw <<= dw_ws[work][0:L, 0:D]
            dw_mx.free()
            l1_ready.set()
            l1_ready.wait()
            matmul(l0c[l0c_cnt], l1_akk.T, l1_dw.T, m=L, n=D, k=L, splitn=D)
            d_k_beta_g[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= l0c[l0c_cnt]
            l0c_cnt += 1

    return d_qg, d_kg, d_w, d_v_beta, d_k_beta_g
