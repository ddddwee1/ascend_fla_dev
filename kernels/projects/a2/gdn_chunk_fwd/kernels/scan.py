"""A2 GDN chunk-forward stage 4: scan (chunk recurrence).

The two matrix contractions run on the cube (matmul) with the L0C->GM->UB GMBuff-ring
handoff. An earlier pure-vector port did step-1 `wy@S` as a `_spread8`+strided-mul+
tree-reduce matvec, but it faulted on b3 ("UB VEC address out of bounds") for a
kernel-context reason (the byte-identical op runs fine in the backward
`reverse_rec._kreduce`), so the cube form — proven on b3 by the chunk-parallel unit —
is the shipped path.

Per (b, value-head), carry the recurrent state S[K,V] across the N chunks:

  delta_n = u_n - wy_n @ S                 (cube: wy[C,D] @ S[D,D], contract D)
  knd_n   = kn_n * exp(-gc_n)              (vec; gc is GDR's per-token scalar prefix)
  S       = exp(gc_last_n) * (S + knd_n^T @ delta_n)
                                           (cube: knd[C,D]^T @ delta[C,D], contract C)

This equals the preweighted form `k*=exp(gc_last-gc); S = S*exp(gc_last) + k^T@delta`
since k_preweighted = exp(gc_last)*knd. Outputs `states` (S entering each chunk) and
`final_state` (S after the last chunk); `delta` is the per-chunk solve residual.
"""
from ascriptor.a2 import *

D = 128
C = 64


@kernel()
def gdn_chunk_scan_a2_kernel(
    kn: GM[f32, ("B", "N", "HV", 64, 128)],
    gc: GM[f32, ("B", "N", "HV", 64, 128)],
    u: GM[f32, ("B", "N", "HV", 64, 128)],
    wy: GM[f32, ("B", "N", "HV", 64, 128)],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    states: GM[f32, ("B", "N", "HV", 128, 128)],
    delta: GM[f32, ("B", "N", "HV", 64, 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    s_ws = GMBuff(DT.float, [D, D], slots=2, name="s_ws")        # UB->GM->L1 (state)
    wys_ws = GMBuff(DT.float, [C, D], slots=2, name="wys_ws")    # L0C->GM->UB (wy@S)
    dl_ws = GMBuff(DT.float, [C, D], slots=2, name="dl_ws")      # UB->GM->L1 (delta)
    knd_ws = GMBuff(DT.float, [C, D], slots=2, name="knd_ws")    # UB->GM->L1 (knd, .T-consumed)
    s2_ws = GMBuff(DT.float, [D, D], slots=2, name="s2_ws")      # L0C->GM->UB (knd^T@delta)

    s_mx = VcMutex(0, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    wys_mx = CvMutex(1, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    dl_mx = VcMutex(2, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    knd_mx = VcMutex(3, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    s2_mx = CvMutex(4, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1_wy = Tensor(DT.float, [C, D], Position.L1)
    l1_s = Tensor(DT.float, [D, D], Position.L1)
    l1_knd = Tensor(DT.float, [C, D], Position.L1)
    l1_dl = Tensor(DT.float, [C, D], Position.L1)
    l0c_wys = Tensor(DT.float, [C, D], Position.L0C)
    l0c_s2 = Tensor(DT.float, [D, D], Position.L0C)

    s_ub = Tensor(DT.float, [D, D], Position.UB)
    u_ub = Tensor(DT.float, [C, D], Position.UB)
    wys_ub = Tensor(DT.float, [C, D], Position.UB)   # exp(-gc) temp, then wy@S / delta
    knd_ub = Tensor(DT.float, [C, D], Position.UB)
    el_ub = Tensor(DT.float, [1, 8], Position.UB)
    l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)

    bhv = B * HV
    per = CeilDiv(bhv, GetCubeNum())
    begin = Var(per * GetCubeIdx())
    end = Min(begin + per, bhv)
    beat = Var(0)
    eplast = Var(0.0, dtype=DT.float)

    with auto_sync():
        for item in range(begin, end):
            hv = Var(item % HV)
            bb = Var(item // HV)
            s_ub[0:D, 0:D] <<= initial_state[bb, hv, 0:D, 0:D]
            for nn in range(N):
                states[bb, nn, hv, 0:D, 0:D] <<= s_ub[0:D, 0:D]

                # exp(gc_last): scalar read in a vec_scope (mixed kernel)
                with vec_scope():
                    el_ub[0:1, 0:8] <<= gc[bb, nn, hv, C - 1:C, 0:8]
                    exp(el_ub[0:1, 0:8], el_ub[0:1, 0:8], count=8)
                    eplast.GetValueFrom(el_ub[0:1, 0:1])

                # knd = kn * exp(-gc) -> L1 via GM ring (.T-consumed by the state matmul)
                knd_ub[0:C, 0:D] <<= kn[bb, nn, hv, 0:C, 0:D]
                wys_ub[0:C, 0:D] <<= gc[bb, nn, hv, 0:C, 0:D]
                muls(wys_ub[0:C, 0:D], wys_ub[0:C, 0:D], -1.0, count=C * D)
                exp(wys_ub[0:C, 0:D], wys_ub[0:C, 0:D], count=C * D)
                mul(knd_ub[0:C, 0:D], knd_ub[0:C, 0:D], wys_ub[0:C, 0:D], count=C * D)
                knd_mx.lock(); knd_ws[beat][0:C, 0:D] <<= knd_ub[0:C, 0:D]; knd_mx.ready()

                # delta = u - wy @ S
                s_mx.lock(); s_ws[beat][0:D, 0:D] <<= s_ub[0:D, 0:D]; s_mx.ready()
                l1_wy <<= wy[bb, nn, hv, 0:C, 0:D]
                s_mx.wait(); l1_s <<= s_ws[beat][0:D, 0:D]; s_mx.free()
                l1_ready.set(); l1_ready.wait()
                matmul(l0c_wys, l1_wy, l1_s.T, m=C, n=D, k=D, splitn=D)
                wys_mx.lock(); wys_ws[beat][0:C, 0:D] <<= l0c_wys; wys_mx.ready()
                u_ub[0:C, 0:D] <<= u[bb, nn, hv, 0:C, 0:D]
                wys_mx.wait(); wys_ub[0:C, 0:D] <<= wys_ws[beat][0:C, 0:D]; wys_mx.free()
                sub(wys_ub[0:C, 0:D], u_ub[0:C, 0:D], wys_ub[0:C, 0:D], count=C * D)   # delta
                delta[bb, nn, hv, 0:C, 0:D] <<= wys_ub[0:C, 0:D]

                # S = exp(gc_last) * (S + knd^T @ delta)
                dl_mx.lock(); dl_ws[beat][0:C, 0:D] <<= wys_ub[0:C, 0:D]; dl_mx.ready()
                knd_mx.wait(); l1_knd <<= knd_ws[beat][0:C, 0:D]; knd_mx.free()
                dl_mx.wait(); l1_dl <<= dl_ws[beat][0:C, 0:D]; dl_mx.free()
                l1_ready.set(); l1_ready.wait()
                barrier(Pipe.M)
                matmul(l0c_s2, l1_knd.T, l1_dl.T, m=D, n=D, k=C, splitn=D)
                s2_mx.lock(); s2_ws[beat][0:D, 0:D] <<= l0c_s2; s2_mx.ready()
                # add knd^T@delta to S in two half-row loads (reuse u_ub as [64,D] temp)
                s2_mx.wait()
                u_ub[0:64, 0:D] <<= s2_ws[beat][0:64, 0:D]
                add(s_ub[0:64, 0:D], s_ub[0:64, 0:D], u_ub[0:64, 0:D], count=64 * D)
                u_ub[0:64, 0:D] <<= s2_ws[beat][64:128, 0:D]
                add(s_ub[64:128, 0:D], s_ub[64:128, 0:D], u_ub[0:64, 0:D], count=64 * D)
                s2_mx.free()
                muls(s_ub[0:64, 0:D], s_ub[0:64, 0:D], eplast, count=64 * D)
                muls(s_ub[64:128, 0:D], s_ub[64:128, 0:D], eplast, count=64 * D)
                beat += 1
            final_state[bb, hv, 0:D, 0:D] <<= s_ub[0:D, 0:D]

    return states, delta, final_state
