"""A2 backward finalize_reduce: sum the GQA group's dq/dk and write them token-major in BF16.

Port of ascriptor ``a5/kda_bwd/kernels/finalize_reduce.py`` to the A2 (c220) facade. Arithmetic unchanged:
per chunk and query head, accumulate the G value-head contributions in FP32, then one BF16 cast.

A2 form: no ``@vf``; the per-chunk accumulate is one whole-tile ``add`` and the cast one whole-tile ``cast``.
The public outputs are declared as the 2-D token-major views ``[B*T, H*128]`` and written with a strided
window, as in the A2 forward unit (host does only ``view``).
"""

from ascriptor.a2 import *

L = 64

D = 128


@kernel(mode="vec")
def finalize_reduce_a2_kernel(
    dq_hv: GM[f32, ('B', 'HV', 'C', 64, 128)],
    dk_hv: GM[f32, ('B', 'HV', 'C', 64, 128)],
    dq_out: GM[bf16, ('BT', 'HK')],
    dk_out: GM[bf16, ('BT', 'HK')],
    B: i32,
    HV: i32,
    H: i32,
    C: i32,
):
    dqacc_ub = Tensor(DT.float, [L, D], Position.UB)
    dkacc_ub = Tensor(DT.float, [L, D], Position.UB)
    dq_g_ub = Tensor(DT.float, [L, D], Position.UB)
    dk_g_ub = Tensor(DT.float, [L, D], Position.UB)
    dq_b_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dk_b_ub = Tensor(DT.bfloat16, [L, D], Position.UB)

    group = Var(HV // H)
    work_count = B * H * C
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)
    n = L * D

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            bh = Var(work // C)
            h_idx = Var(bh % H)
            b_idx = Var(bh // H)
            hv0 = Var(h_idx * group)
            row0 = Var(b_idx * C * L + c_idx * L)
            col0 = Var(h_idx * D)

            dqacc_ub[0:L, 0:D] <<= dq_hv[b_idx, hv0, c_idx, 0:L, 0:D]
            dkacc_ub[0:L, 0:D] <<= dk_hv[b_idx, hv0, c_idx, 0:L, 0:D]
            for g in range(1, group):
                hv = Var(hv0 + g)
                dq_g_ub[0:L, 0:D] <<= dq_hv[b_idx, hv, c_idx, 0:L, 0:D]
                dk_g_ub[0:L, 0:D] <<= dk_hv[b_idx, hv, c_idx, 0:L, 0:D]
                add(dqacc_ub[0:L, 0:D], dqacc_ub[0:L, 0:D], dq_g_ub[0:L, 0:D], count=n)
                add(dkacc_ub[0:L, 0:D], dkacc_ub[0:L, 0:D], dk_g_ub[0:L, 0:D], count=n)
            cast(dq_b_ub[0:L, 0:D], dqacc_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n)
            cast(dk_b_ub[0:L, 0:D], dkacc_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n)
            dq_out[row0:row0 + L, col0:col0 + D] <<= dq_b_ub[0:L, 0:D]
            dk_out[row0:row0 + L, col0:col0 + D] <<= dk_b_ub[0:L, 0:D]

    return dq_out, dk_out
