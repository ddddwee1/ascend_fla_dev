"""A2 backward chunk scan: the reverse state recurrence and everything that hangs off it.

Port of ``a5/kda_bwd/kernels/scan_fused.py`` to the A2 (c220) facade. Per (batch, value head), walking the
chunks in reverse with ``dstate`` [128, 128] carried in FP32 on the vector side::

    dh[c]     = bf16(dstate)                                  (snapshot before the chunk is folded in)
    dAqk[c]   = (do @ v_new^T) * D^-0.5
    dv[c]     = Aqk^T @ do  +  kg @ dstate
    seed      = qg^T @ do                 corr = w^T @ dv
    dstate    = seed * D^-0.5  +  dstate * exp(g_last * ln2)  -  corr     (decay is per key row)
    dh0       = bf16(dstate)                                  (after the last chunk)

A2 differences:

* no ``@vf``: the six A5 vector functions become tile-vector ops on UB, and the per-key-row decay is one
  ``muls`` per row with the scalar read by ``Var.GetValueFrom``. The A5 register forms (``deinterleave``,
  ``reinterpret``, ``.nz()`` UB -> L1 stores) have no A2 spelling.
* neither ``dma.ub_to_l1`` nor ``dma.l0c_to_ub`` exists on b3, so all seven cross-side handoffs are two-slot GM
  rings indexed by the single chunk beat, each guarded by its own mutex: ``VcMutex`` (MTE3 -> MTE2) for the
  state and ``dv`` bridges, ``CvMutex`` (FIX -> MTE2) for the four cube products.
* **D-PM-37**: A5 took ``g_last`` as its own input, which the host built with a strided slice and copy of the
  cached ``g_cumsum``. Here the kernel reads the chunk's last gate row straight out of ``g_cumsum``, and
  computes ``exp(g_last * ln2)`` itself (the gates are log2, as everywhere else in this unit).
* the ``1/sqrt(D)`` that A5 folded into a host-side ``qg`` pre-scale is applied to ``seed`` in the kernel.
* L0C on b3 is 128 KB, so the five products share three accumulators (``[128,128]``, ``[64,128]``, ``[64,64]``
  = 112 KB); every reuse gets an explicit ``barrier(Pipe.M)`` first, per the A2-01 discipline for repeated
  short MMADs against one L0C block.
* ``kg`` / ``qg`` / ``w`` / ``grad_out`` / ``v_new`` / ``Aqk`` / ``g_cumsum`` and the ``dv`` / ``dAqk`` outputs
  are token-major 2-D views; ``dh`` and ``dh0`` keep the chain-internal block layout.

D-PM-60: derived FP32 cumulative-gate ABI. Other arithmetic, ownership and synchronization are unchanged.
"""

import math

from ascriptor.a2 import *

L = 64

D = 128

HALF_L = L // 2

HALF_D = D // 2

SPLIT_N = 64

QG_SCALE = 1.0 / (D ** 0.5)

DAQK_SCALE = 1.0 / (D ** 0.5)

LN2 = math.log(2.0)


@kernel()
def scan_fused_a2_fp32gc_kernel(
    kg: GM[bf16, ('BT', 'HVK')],
    qg: GM[bf16, ('BT', 'HVK')],
    w: GM[bf16, ('BT', 'HVK')],
    g_cumsum: GM[f32, ('BT', 'HVK')],
    grad_out: GM[bf16, ('BT', 'HVK')],
    Aqk: GM[bf16, ('BT', 'HVL')],
    v_new: GM[bf16, ('BT', 'HVK')],
    dht: GM[bf16, ('B', 'HV', 128, 128)],
    dAqk: GM[bf16, ('BT', 'HVL')],
    dh: GM[bf16, ('B', 'C', 'HV', 128, 128)],
    dv: GM[bf16, ('BT', 'HVK')],
    dh0: GM[bf16, ('B', 'HV', 128, 128)],
    B: i32,
    HV: i32,
    C: i32,
):
    state_ws = GMBuff(DT.bfloat16, [D, D], slots=2, name="state_ws")
    seed_ws = GMBuff(DT.float, [D, D], slots=2, name="seed_ws")
    dv0_ws = GMBuff(DT.float, [L, D], slots=2, name="dv0_ws")
    daqk_ws = GMBuff(DT.float, [L, L], slots=2, name="daqk_ws")
    dvd_ws = GMBuff(DT.float, [L, D], slots=2, name="dvd_ws")
    dv_ws = GMBuff(DT.bfloat16, [L, D], slots=2, name="dv_ws")
    corr_ws = GMBuff(DT.float, [D, D], slots=2, name="corr_ws")

    state_mx = VcMutex(0, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    prod_mx = CvMutex(1, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    dvd_mx = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    dv_mx = VcMutex(3, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    corr_mx = CvMutex(4, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1_qg = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_w = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_do = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_aqk = Tensor(DT.bfloat16, [L, L], Position.L1)
    l1_vnew = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_kg = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_dstate = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_dv = Tensor(DT.bfloat16, [L, D], Position.L1)

    l0c_dd = Tensor(DT.float, [D, D], Position.L0C)     # seed, then corr
    l0c_ld = Tensor(DT.float, [L, D], Position.L0C)     # dv0, then dv_delta
    l0c_ll = Tensor(DT.float, [L, L], Position.L0C)     # dAqk

    dstate_ub = Tensor(DT.float, [HALF_D, D], Position.UB)
    state_h_ub = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)
    seed_ub = Tensor(DT.float, [HALF_D, D], Position.UB)
    dv_sum_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dv_delta_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dv_b_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    daqk_f_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    daqk_b_ub = Tensor(DT.bfloat16, [HALF_L, L], Position.UB)
    expg_ub = Tensor(DT.float, [1, HALF_D], Position.UB)

    # Precautionary L1 load fence. On the pinned library auto_sync's MTE2 -> MTE1 ready guard does not
    # name every L1 buffer of this kernel (read off the generated cube source), so in principle a matmul
    # could read a tile before its GM load landed. A controlled A/B on a 910B3 (same case, same build,
    # only these two fences removed) produced bitwise identical output, so no consequence has been
    # observed; the fence is kept as insurance, not as a fix for a measured defect.
    l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)

    bhv_count = B * HV
    bhv_per_cube = CeilDiv(bhv_count, GetCubeNum())
    bhv_begin = Var(bhv_per_cube * GetCubeIdx())
    bhv_end = Min(bhv_begin + bhv_per_cube, bhv_count)
    beat = Var(0)
    decay = Var(0.0, dtype=DT.float)
    n_dd = HALF_D * D
    n_ld = HALF_L * D
    n_ll = HALF_L * L

    with auto_sync():
        for bhv in range(bhv_begin, bhv_end):
            hv_idx = Var(bhv % HV)
            b_idx = Var(bhv // HV)
            rowd0 = Var(GetSubBlockIdx() * HALF_D)
            rowl0 = Var(GetSubBlockIdx() * HALF_L)
            hv_col = Var(hv_idx * D)
            l_col = Var(hv_idx * L)

            # dstate <- dht; the BF16 staging buffer is already the snapshot bf16(dstate).
            state_h_ub[0:HALF_D, 0:D] <<= dht[b_idx, hv_idx, rowd0:rowd0 + HALF_D, 0:D]
            cast(dstate_ub[0:HALF_D, 0:D], state_h_ub[0:HALF_D, 0:D], round_mode=RoundMode.NONE, count=n_dd)

            for rev_c in range(0, C):
                c_idx = Var(C - 1 - rev_c)
                ctok0 = Var(b_idx * C * L + c_idx * L)

                # vector: per-chunk decay exp(g_last * ln2), read straight from g_cumsum's last row
                expg_ub[0:1, 0:HALF_D] <<= g_cumsum[ctok0 + L - 1:ctok0 + L, hv_col + rowd0:hv_col + rowd0 + HALF_D]
                muls(expg_ub[0:1, 0:HALF_D], expg_ub[0:1, 0:HALF_D], LN2, count=HALF_D)
                exp(expg_ub[0:1, 0:HALF_D], expg_ub[0:1, 0:HALF_D], count=HALF_D)

                # vector: publish the pre-chunk state snapshot to dh and to the cube
                dh[b_idx, c_idx, hv_idx, rowd0:rowd0 + HALF_D, 0:D] <<= state_h_ub[0:HALF_D, 0:D]
                state_mx.lock()
                state_ws[beat][rowd0:rowd0 + HALF_D, 0:D] <<= state_h_ub[0:HALF_D, 0:D]
                state_mx.ready()

                # cube: the three products that need only this chunk's inputs
                l1_qg <<= qg[ctok0:ctok0 + L, hv_col:hv_col + D]
                l1_w <<= w[ctok0:ctok0 + L, hv_col:hv_col + D]
                l1_do <<= grad_out[ctok0:ctok0 + L, hv_col:hv_col + D]
                l1_aqk <<= Aqk[ctok0:ctok0 + L, l_col:l_col + L]
                l1_vnew <<= v_new[ctok0:ctok0 + L, hv_col:hv_col + D]
                l1_kg <<= kg[ctok0:ctok0 + L, hv_col:hv_col + D]
                l1_ready.set()
                l1_ready.wait()

                matmul(l0c_dd, l1_qg.T, l1_do.T, m=D, n=D, k=L, splitn=SPLIT_N)
                matmul(l0c_ld, l1_aqk.T, l1_do.T, m=L, n=D, k=L, splitn=SPLIT_N)
                matmul(l0c_ll, l1_do, l1_vnew, m=L, n=L, k=D, splitn=SPLIT_N)
                prod_mx.lock()
                seed_ws[beat][0:D, 0:D] <<= l0c_dd
                dv0_ws[beat][0:L, 0:D] <<= l0c_ld
                daqk_ws[beat][0:L, 0:L] <<= l0c_ll
                prod_mx.ready()

                # cube: dv_delta = kg @ dstate (reuses the [64,128] accumulator)
                state_mx.wait()
                l1_dstate <<= state_ws[beat][0:D, 0:D]
                state_mx.free()
                l1_ready.set()
                l1_ready.wait()
                barrier(Pipe.M)
                matmul(l0c_ld, l1_kg, l1_dstate.T, m=L, n=D, k=D, splitn=SPLIT_N)
                dvd_mx.lock()
                dvd_ws[beat][0:L, 0:D] <<= l0c_ld
                dvd_mx.ready()

                # vector: dAqk, and the first half of dv
                prod_mx.wait()
                seed_ub[0:HALF_D, 0:D] <<= seed_ws[beat][rowd0:rowd0 + HALF_D, 0:D]
                dv_sum_ub[0:HALF_L, 0:D] <<= dv0_ws[beat][rowl0:rowl0 + HALF_L, 0:D]
                daqk_f_ub[0:HALF_L, 0:L] <<= daqk_ws[beat][rowl0:rowl0 + HALF_L, 0:L]
                prod_mx.free()
                muls(daqk_f_ub[0:HALF_L, 0:L], daqk_f_ub[0:HALF_L, 0:L], DAQK_SCALE, count=n_ll)
                cast(daqk_b_ub[0:HALF_L, 0:L], daqk_f_ub[0:HALF_L, 0:L], round_mode=RoundMode.TO_EVEN, count=n_ll)
                dAqk[ctok0 + rowl0:ctok0 + rowl0 + HALF_L, l_col:l_col + L] <<= daqk_b_ub[0:HALF_L, 0:L]

                # vector: dv = dv0 + dv_delta, written out and bridged back to the cube
                dvd_mx.wait()
                dv_delta_ub[0:HALF_L, 0:D] <<= dvd_ws[beat][rowl0:rowl0 + HALF_L, 0:D]
                dvd_mx.free()
                add(dv_sum_ub[0:HALF_L, 0:D], dv_sum_ub[0:HALF_L, 0:D], dv_delta_ub[0:HALF_L, 0:D], count=n_ld)
                cast(dv_b_ub[0:HALF_L, 0:D], dv_sum_ub[0:HALF_L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_ld)
                dv[ctok0 + rowl0:ctok0 + rowl0 + HALF_L, hv_col:hv_col + D] <<= dv_b_ub[0:HALF_L, 0:D]
                dv_mx.lock()
                dv_ws[beat][rowl0:rowl0 + HALF_L, 0:D] <<= dv_b_ub[0:HALF_L, 0:D]
                dv_mx.ready()

                # cube: corr = w^T @ dv (reuses the [128,128] accumulator)
                dv_mx.wait()
                l1_dv <<= dv_ws[beat][0:L, 0:D]
                dv_mx.free()
                l1_ready.set()
                l1_ready.wait()
                barrier(Pipe.M)
                matmul(l0c_dd, l1_w.T, l1_dv.T, m=D, n=D, k=L, splitn=SPLIT_N)
                corr_mx.lock()
                corr_ws[beat][0:D, 0:D] <<= l0c_dd
                corr_mx.ready()

                # vector: dstate = seed/sqrt(D) + dstate*decay - corr, in the A5 operation order
                muls(seed_ub[0:HALF_D, 0:D], seed_ub[0:HALF_D, 0:D], QG_SCALE, count=n_dd)
                with vec_scope():
                    for r in range(0, HALF_D):
                        decay.GetValueFrom(expg_ub[0:1, r:r + 1])
                        muls(dstate_ub[r:r + 1, 0:D], dstate_ub[r:r + 1, 0:D], decay, count=D)
                add(seed_ub[0:HALF_D, 0:D], seed_ub[0:HALF_D, 0:D], dstate_ub[0:HALF_D, 0:D], count=n_dd)
                corr_mx.wait()
                dstate_ub[0:HALF_D, 0:D] <<= corr_ws[beat][rowd0:rowd0 + HALF_D, 0:D]
                corr_mx.free()
                sub(dstate_ub[0:HALF_D, 0:D], seed_ub[0:HALF_D, 0:D], dstate_ub[0:HALF_D, 0:D], count=n_dd)
                cast(state_h_ub[0:HALF_D, 0:D], dstate_ub[0:HALF_D, 0:D], round_mode=RoundMode.TO_EVEN, count=n_dd)
                beat += 1

            dh0[b_idx, hv_idx, rowd0:rowd0 + HALF_D, 0:D] <<= state_h_ub[0:HALF_D, 0:D]

    return dAqk, dh, dv, dh0
