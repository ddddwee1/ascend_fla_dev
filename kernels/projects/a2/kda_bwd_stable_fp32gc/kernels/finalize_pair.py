"""A2 backward finalize_pair: the four paired-decay matmuls of the finalize stage.

Port of ascriptor ``a5/kda_bwd/kernels/finalize_pair.py`` to the A2 (c220) facade::

    dq_pair = M_qk   @ kg        dk_pair = M_qk^T   @ q
    s_base  = M_base @ kg        t_beta  = M_beta^T @ k

All four are ``splitn`` with ``is_init=True``, so each L0C tile is written by exactly one MMAD: no accumulate
chain, hence no settle barrier needed here (A2-01). Operands are BF16 and all tensors are chain-internal
``[B, HV, T, *]``, so this kernel is a straight facade port; only the import and the double-buffer counter
handling differ from the A5 source.
"""

from ascriptor.a2 import *

L = 64

D = 128


@kernel(mode="cube")
def finalize_pair_a2_kernel(
    Mqk: GM[bf16, ('B', 'HV', 'T', 64)],
    Mbase: GM[bf16, ('B', 'HV', 'T', 64)],
    Mbeta: GM[bf16, ('B', 'HV', 'T', 64)],
    q_hv: GM[bf16, ('B', 'HV', 'T', 128)],
    k_hv: GM[bf16, ('B', 'HV', 'T', 128)],
    kg: GM[bf16, ('B', 'HV', 'T', 128)],
    dq_pair: GM[bf16, ('B', 'HV', 'T', 128)],
    dk_pair: GM[bf16, ('B', 'HV', 'T', 128)],
    s_base: GM[bf16, ('B', 'HV', 'T', 128)],
    t_beta: GM[bf16, ('B', 'HV', 'T', 128)],
    B: i32,
    HV: i32,
    T: i32,
):
    l1_mqk = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_mbase = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_mbeta = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_q = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_k = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_kg = DBuff(DT.bfloat16, [L, D], Position.L1)
    # Four [64, 128] FP32 accumulators fill b3's 128 KB of L0C exactly, so nothing has to be shared. An
    # earlier version ran the four products in two rounds against one pair; that was changed while chasing
    # a device error which turned out to be the packed-column cast elsewhere in this unit, and the separate
    # accumulators are kept because they fit and leave no repeated MMAD against one L0C block to reason
    # about (A2-01 / ascriptor M10-081).
    l0c_dq = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dk = Tensor(DT.float, [L, D], Position.L0C)
    l0c_s = Tensor(DT.float, [L, D], Position.L0C)
    l0c_t = Tensor(DT.float, [L, D], Position.L0C)

    # Precautionary L1 load fence. On the pinned library auto_sync's MTE2 -> MTE1 ready guard does not
    # name every L1 buffer of this kernel (read off the generated cube source), so in principle a matmul
    # could read a tile before its GM load landed. A controlled A/B on a 910B3 (same case, same build,
    # only these two fences removed) produced bitwise identical output, so no consequence has been
    # observed; the fence is kept as insurance, not as a fix for a measured defect.
    l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)

    chunk_count = Var(T // L)
    work_count = B * HV * chunk_count
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    slot = Var(0)

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % chunk_count)
            bhv = Var(work // chunk_count)
            hv_idx = Var(bhv % HV)
            b_idx = Var(bhv // HV)
            row0 = Var(c_idx * L)
            row1 = Var(row0 + L)

            l1_mqk[slot][0:L, 0:L] <<= Mqk[b_idx, hv_idx, row0:row1, 0:L]
            l1_q[slot][0:L, 0:D] <<= q_hv[b_idx, hv_idx, row0:row1, 0:D]
            l1_kg[slot][0:L, 0:D] <<= kg[b_idx, hv_idx, row0:row1, 0:D]
            l1_ready.set()
            l1_ready.wait()
            matmul(l0c_dq, l1_mqk[slot], l1_kg[slot].T, m=L, n=D, k=L, splitn=D)
            matmul(l0c_dk, l1_mqk[slot].T, l1_q[slot].T, m=L, n=D, k=L, splitn=D)
            dq_pair[b_idx, hv_idx, row0:row1, 0:D] <<= l0c_dq
            dk_pair[b_idx, hv_idx, row0:row1, 0:D] <<= l0c_dk

            l1_mbase[slot][0:L, 0:L] <<= Mbase[b_idx, hv_idx, row0:row1, 0:L]
            l1_mbeta[slot][0:L, 0:L] <<= Mbeta[b_idx, hv_idx, row0:row1, 0:L]
            l1_k[slot][0:L, 0:D] <<= k_hv[b_idx, hv_idx, row0:row1, 0:D]
            l1_ready.set()
            l1_ready.wait()
            matmul(l0c_s, l1_mbase[slot], l1_kg[slot].T, m=L, n=D, k=L, splitn=D)
            matmul(l0c_t, l1_mbeta[slot].T, l1_k[slot].T, m=L, n=D, k=L, splitn=D)
            s_base[b_idx, hv_idx, row0:row1, 0:D] <<= l0c_s
            t_beta[b_idx, hv_idx, row0:row1, 0:D] <<= l0c_t
            # Four independent MMAD -> fixpipe pairs per chunk overrun auto_sync's M -> FIX event, whose
            # capacity is two tokens: pipesim reports a temporal flag hazard on ev_m_fix_ready_1 at
            # HV=4, C=2 without this. bar_all closes the cube pipeline at the chunk boundary, which is
            # what the A5 kernel's _phase_matmul does around every matmul.
            bar_all()
            slot += 1

    return dq_pair, dk_pair, s_base, t_beta
