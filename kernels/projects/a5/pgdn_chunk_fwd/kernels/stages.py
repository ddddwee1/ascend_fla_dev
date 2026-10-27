"""Five FP32 Vector stages following the independent PGDN ATK stage.

The generic causal-score/WY/scan/output mechanics are adapted from this
repository's GDN-2 chunk unit at 8786683. PGDN uses normalized read keys for the correction/WY RHS and distinct
preconditioned write keys for causal scores and state writes. This module has no GDN-2 runtime dependency.

All GM edges are fully written and consumed by a subsequent launch. Local
buffers have one slot; auto_sync handles DMA/VF ownership within each launch.
"""
from ascriptor.a5 import *

D = 128
C = 64
SCALE = 1.0 / 11.313708498984761


@vf()
def zero_row(x: Tensor):
    z = RegList(DT.float, 2)
    z <<= 0.0
    x[0:1, 0:D] <<= z


@vf()
def prepare_tile(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
                 bk: Tensor, prefix: Tensor):
    qr = RegList(DT.float, 2)
    kr = RegList(DT.float, 2)
    vr = RegList(DT.float, 2)
    pr = RegList(DT.float, 2)
    gv = Reg(DT.float)
    bv = Reg(DT.float)
    pr <<= 0.0
    # The FP32 prefix still advances in token order. Keeping it in registers
    # removes materialize/reload only; no sum reassociation is introduced.
    for i in range(C):
        qr <<= q[i:i+1, 0:D]
        kr <<= k[i:i+1, 0:D]
        vr <<= v[i:i+1, 0:D]
        gv <<= g[i:i+1, 0:1].single()
        bv <<= beta[i:i+1, 0:1].single()
        pr <<= pr + gv
        qr <<= qr * SCALE
        kr <<= kr * bv
        vr <<= vr * bv
        q[i:i+1, 0:D] <<= qr
        bk[i:i+1, 0:D] <<= kr
        v[i:i+1, 0:D] <<= vr
        prefix[i:i+1, 0:D] <<= pr
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def pgdn_chunk_prepare(
    q_norm: GM[f32, ("B", "T", "H", 128)], k_read: GM[f32, ("B", "T", "H", 128)],
    k_write: GM[f32, ("B", "T", "H", 128)], v: GM[f32, ("B", "T", "HV", 128)], g: GM[f32, ("B", "T", "HV")],
    beta: GM[f32, ("B", "T", "HV")],
    qn: GM[f32, ("B", "N", "HV", 64, 128)], kw: GM[f32, ("B", "N", "HV", 64, 128)],
    gc: GM[f32, ("B", "N", "HV", 64, 128)], bk: GM[f32, ("B", "N", "HV", 64, 128)],
    wv: GM[f32, ("B", "N", "HV", 64, 128)], B: i32, T: i32, H: i32, HV: i32, N: i32,
):
    # 196 KiB total UB including the distinct preconditioned key tile. Two complete slots would exceed the 256 KiB profile;
    # use one item in flight and retire all MTE3 readers before the next load.
    qu = Tensor(DT.float, [C, D], Position.UB)
    ku = Tensor(DT.float, [C, D], Position.UB)
    kwu = Tensor(DT.float, [C, D], Position.UB)
    vu = Tensor(DT.float, [C, D], Position.UB)
    gu = Tensor(DT.float, [C, 8], Position.UB)
    bu = Tensor(DT.float, [C, 8], Position.UB)
    bku = Tensor(DT.float, [C, D], Position.UB)
    pu = Tensor(DT.float, [C, D], Position.UB)
    per = CeilDiv(B * N * HV, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, B * N * HV)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % HV)
            cc = Var((item // HV) % N)
            bb = Var(item // (N * HV))
            tt = Var(cc * C)
            kh = Var(hh // (HV // H))
            # Explicit row gaps preserve the distinct key/value-head strides.
            gm_to_ub_pad(qu, q_norm[bb, tt:tt+C, kh, :], C, D, (H - 1) * D, 0)
            gm_to_ub_pad(ku, k_read[bb, tt:tt+C, kh, :], C, D, (H - 1) * D, 0)
            gm_to_ub_pad(kwu, k_write[bb, tt:tt+C, kh, :], C, D, (H - 1) * D, 0)
            gm_to_ub_pad(vu, v[bb, tt:tt+C, hh, :], C, D, (HV - 1) * D, 0)
            gu[:, 0:1] <<= g[bb, tt:tt+C, hh:hh+1]
            bu[:, 0:1] <<= beta[bb, tt:tt+C, hh:hh+1]
            prepare_tile(qu, ku, vu, gu, bu, bku, pu)
            qn[bb, cc, hh, :, :] <<= qu[:, :]
            kw[bb, cc, hh, :, :] <<= kwu[:, :]
            gc[bb, cc, hh, :, :] <<= pu[:, :]
            bk[bb, cc, hh, :, :] <<= bku[:, :]
            wv[bb, cc, hh, :, :] <<= vu[:, :]
    return qn, kw, gc, bk, wv


@vf()
def scores_vf(q: Tensor, k: Tensor, g: Tensor, b: Tensor, lower: Tensor,
              score: Tensor):
    qr = RegList(DT.float, 2)
    kr = RegList(DT.float, 2)
    gi = RegList(DT.float, 2)
    gj = RegList(DT.float, 2)
    br = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    dot = Reg(DT.float)
    z = Reg(DT.float)
    z <<= 0.0
    for i in range(C):
        lower[i:i+1, 0:C] <<= z
        score[i:i+1, 0:C] <<= z
    for i in range(C):
        qr <<= q[i:i+1, 0:D]
        gi <<= g[i:i+1, 0:D]
        br <<= b[i:i+1, 0:D]
        for j in range(C):
            if j <= i:
                kr <<= k[j:j+1, 0:D]
                gj <<= g[j:j+1, 0:D]
                gj <<= gi - gj
                gj <<= gj.exp()
                kr <<= kr * gj
                tmp <<= qr * kr
                dot <<= tmp.cadd()
                score[i:i+1, j:j+1] <<= dot.single_value()
                if j < i:
                    tmp <<= br * kr
                    dot <<= tmp.cadd()
                    lower[i:i+1, j:j+1] <<= dot.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def pgdn_chunk_scores(
    qn: GM[f32, ("B", "N", "HV", 64, 128)], kw: GM[f32, ("B", "N", "HV", 64, 128)],
    gc: GM[f32, ("B", "N", "HV", 64, 128)], bk: GM[f32, ("B", "N", "HV", 64, 128)],
    lower: GM[f32, ("B", "N", "HV", 64, 64)], score: GM[f32, ("B", "N", "HV", 64, 64)],
    B: i32, T: i32, HV: i32, N: i32,
):
    qu = Tensor(DT.float, [C, D], Position.UB)
    ku = Tensor(DT.float, [C, D], Position.UB)
    gu = Tensor(DT.float, [C, D], Position.UB)
    bu = Tensor(DT.float, [C, D], Position.UB)
    lu = Tensor(DT.float, [C, C], Position.UB)
    au = Tensor(DT.float, [C, C], Position.UB)
    per = CeilDiv(B * N * HV, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, B * N * HV)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % HV)
            cc = Var((item // HV) % N)
            bb = Var(item // (N * HV))
            qu[:, :] <<= qn[bb, cc, hh, :, :]
            ku[:, :] <<= kw[bb, cc, hh, :, :]
            gu[:, :] <<= gc[bb, cc, hh, :, :]
            bu[:, :] <<= bk[bb, cc, hh, :, :]
            scores_vf(qu, ku, gu, bu, lu, au)
            lower[bb, cc, hh, :, :] <<= lu[:, :]
            score[bb, cc, hh, :, :] <<= au[:, :]
    return lower, score


@vf()
def wy_vf(lower: Tensor, g: Tensor, b: Tensor, v: Tensor,
          u: Tensor, wy: Tensor):
    ur = RegList(DT.float, 2)
    wr = RegList(DT.float, 2)
    un = RegList(DT.float, 2)
    wn = RegList(DT.float, 2)
    decay = RegList(DT.float, 2)
    prev = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    weight = Reg(DT.float)
    next_weight = Reg(DT.float)
    zero = RegList(DT.float, 2)
    zero <<= 0.0
    for i in range(C):
        u[i:i+1, 0:D] <<= zero
        wy[i:i+1, 0:D] <<= zero
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    # The public ABI has full 64-row chunks; even i keeps i+1 inside C.
    # Constant VF loop bounds avoid CCE mixed-width condition operands.
    for i in range(0, C, 2):
        ur <<= v[i:i+1, 0:D]
        wr <<= b[i:i+1, 0:D]
        decay <<= g[i:i+1, 0:D]
        decay <<= decay.exp()
        wr <<= wr * decay
        un <<= v[i+1:i+2, 0:D]
        wn <<= b[i+1:i+2, 0:D]
        decay <<= g[i+1:i+2, 0:D]
        decay <<= decay.exp()
        wn <<= wn * decay
        # CANN 9.2 / 950PR native control: a VF loop bounded directly by
        # the enclosing VF induction variable omitted the last contribution.
        # A constant trip count with an explicit guard preserves the complete
        # ordered solve. Stronger barriers and native float loads did not fix
        # the dependent-bound form; this is a scoped source workaround.
        for j in range(C):
            if j < i:
                weight <<= lower[i:i+1, j:j+1].single()
                next_weight <<= lower[i+1:i+2, j:j+1].single()
                prev <<= u[j:j+1, 0:D]
                tmp <<= prev * weight
                ur <<= ur - tmp
                tmp <<= prev * next_weight
                un <<= un - tmp
                prev <<= wy[j:j+1, 0:D]
                tmp <<= prev * weight
                wr <<= wr - tmp
                tmp <<= prev * next_weight
                wn <<= wn - tmp
        # The last contribution of row i+1 uses completed FP32 row i.
        # All per-row products/subtractions retain their original order.
        next_weight <<= lower[i+1:i+2, i:i+1].single()
        tmp <<= ur * next_weight
        un <<= un - tmp
        tmp <<= wr * next_weight
        wn <<= wn - tmp
        u[i:i+1, 0:D] <<= ur
        wy[i:i+1, 0:D] <<= wr
        u[i+1:i+2, 0:D] <<= un
        wy[i+1:i+2, 0:D] <<= wn
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def pgdn_chunk_wy(
    lower: GM[f32, ("B", "N", "HV", 64, 64)], gc: GM[f32, ("B", "N", "HV", 64, 128)],
    bk: GM[f32, ("B", "N", "HV", 64, 128)], wv: GM[f32, ("B", "N", "HV", 64, 128)],
    u: GM[f32, ("B", "N", "HV", 64, 128)], wy: GM[f32, ("B", "N", "HV", 64, 128)],
    B: i32, T: i32, HV: i32, N: i32,
):
    lu = Tensor(DT.float, [C, C], Position.UB)
    gu = Tensor(DT.float, [C, D], Position.UB)
    bu = Tensor(DT.float, [C, D], Position.UB)
    vu = Tensor(DT.float, [C, D], Position.UB)
    uu = Tensor(DT.float, [C, D], Position.UB)
    wu = Tensor(DT.float, [C, D], Position.UB)
    per = CeilDiv(B * N * HV, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, B * N * HV)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % HV)
            cc = Var((item // HV) % N)
            bb = Var(item // (N * HV))
            lu[:, :] <<= lower[bb, cc, hh, :, :]
            gu[:, :] <<= gc[bb, cc, hh, :, :]
            bu[:, :] <<= bk[bb, cc, hh, :, :]
            vu[:, :] <<= wv[bb, cc, hh, :, :]
            wy_vf(lu, gu, bu, vu, uu, wu)
            u[bb, cc, hh, :, :] <<= uu[:, :]
            wy[bb, cc, hh, :, :] <<= wu[:, :]
    return u, wy


@vf()
def scan_vf(state: Tensor, k: Tensor, g: Tensor, u: Tensor,
            wy: Tensor, delta: Tensor):
    row = RegList(DT.float, 2)
    acc = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    weight = Reg(DT.float)
    last = Reg(DT.float)
    decay = Reg(DT.float)
    zero = RegList(DT.float, 2)
    zero <<= 0.0
    for i in range(C):
        delta[i:i+1, 0:D] <<= zero
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    # All deltas read S_in before any state row is updated.
    for i in range(C):
        acc <<= u[i:i+1, 0:D]
        for d in range(D):
            weight <<= wy[i:i+1, d:d+1].single()
            row <<= state[d:d+1, 0:D]
            tmp <<= row * weight
            acc <<= acc - tmp
        delta[i:i+1, 0:D] <<= acc
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    # This private write-key tile has no later consumer. Preweight the complete channel row
    # once, avoiding an exponential broadcast for every (row, channel).
    acc <<= g[C-1:C, 0:D]
    for i in range(C):
        row <<= g[i:i+1, 0:D]
        row <<= acc - row
        row <<= row.exp()
        tmp <<= k[i:i+1, 0:D]
        tmp <<= tmp * row
        k[i:i+1, 0:D] <<= tmp
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    for d in range(D):
        last <<= g[C-1:C, d:d+1].single()
        decay <<= last.exp()
        row <<= state[d:d+1, 0:D]
        row <<= row * decay
        for i in range(C):
            weight <<= k[i:i+1, d:d+1].single()
            tmp <<= delta[i:i+1, 0:D]
            tmp <<= tmp * weight
            row <<= row + tmp
        state[d:d+1, 0:D] <<= row
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def pgdn_chunk_scan(
    kw: GM[f32, ("B", "N", "HV", 64, 128)], gc: GM[f32, ("B", "N", "HV", 64, 128)],
    u: GM[f32, ("B", "N", "HV", 64, 128)], wy: GM[f32, ("B", "N", "HV", 64, 128)],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    states: GM[f32, ("B", "N", "HV", 128, 128)], delta: GM[f32, ("B", "N", "HV", 64, 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)], B: i32, T: i32, HV: i32, N: i32,
):
    su = Tensor(DT.float, [D, D], Position.UB)
    ku = Tensor(DT.float, [C, D], Position.UB)
    gu = Tensor(DT.float, [C, D], Position.UB)
    uu = Tensor(DT.float, [C, D], Position.UB)
    wu = Tensor(DT.float, [C, D], Position.UB)
    du = Tensor(DT.float, [C, D], Position.UB)
    per = CeilDiv(B * HV, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, B * HV)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % HV)
            bb = Var(item // HV)
            su[:, :] <<= initial_state[bb, hh, :, :]
            for cc in range(N):
                states[bb, cc, hh, :, :] <<= su[:, :]
                ku[:, :] <<= kw[bb, cc, hh, :, :]
                gu[:, :] <<= gc[bb, cc, hh, :, :]
                uu[:, :] <<= u[bb, cc, hh, :, :]
                wu[:, :] <<= wy[bb, cc, hh, :, :]
                scan_vf(su, ku, gu, uu, wu, du)
                delta[bb, cc, hh, :, :] <<= du[:, :]
            final_state[bb, hh, :, :] <<= su[:, :]
    return states, delta, final_state


@vf()
def output_vf(q: Tensor, g: Tensor, score: Tensor, state: Tensor,
              delta: Tensor, output: Tensor):
    acc = RegList(DT.float, 2)
    row = RegList(DT.float, 2)
    weight = Reg(DT.float)
    # q is private UB storage; retire the raw query role before its weighted
    # scalar consumers. FP32 products and subsequent sum order are unchanged.
    for i in range(C):
        acc <<= q[i:i+1, 0:D]
        row <<= g[i:i+1, 0:D]
        row <<= row.exp()
        acc <<= acc * row
        q[i:i+1, 0:D] <<= acc
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    for i in range(C):
        acc <<= 0.0
        for d in range(D):
            weight <<= q[i:i+1, d:d+1].single()
            row <<= state[d:d+1, 0:D]
            row <<= row * weight
            acc <<= acc + row
        # The acausal half of `score` is exactly +-0: `pgdn_chunk_scores` fills the
        # whole 64x64 tile with 0.0 before writing only the causal entries, and the
        # independent stage reference multiplies by exp(-inf) there. A finite
        # `delta` therefore makes every acausal term 0*delta = +-0, and acc only
        # ever holds +0 or a nonzero value (it starts at +0 and IEEE cancellation
        # yields +0), so acc + (+-0) == acc exactly. Running the full constant
        # trip count keeps the increasing-j sum order and the stored values
        # bitwise unchanged, and removes the data-dependent branch: inside
        # __VEC_SCOPE__ a guard in this loop body costs far more per iteration
        # than the body itself (GDA-05: this kernel's own unguarded 128-trip d
        # loop above runs an identical 7-instruction body ~10x cheaper per
        # iteration than the guarded j loop did).
        for j in range(C):
            weight <<= score[i:i+1, j:j+1].single()
            row <<= delta[j:j+1, 0:D]
            row <<= row * weight
            acc <<= acc + row
        output[i:i+1, 0:D] <<= acc
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def pgdn_chunk_output(
    qn: GM[f32, ("B", "N", "HV", 64, 128)], gc: GM[f32, ("B", "N", "HV", 64, 128)],
    score: GM[f32, ("B", "N", "HV", 64, 64)], states: GM[f32, ("B", "N", "HV", 128, 128)],
    delta: GM[f32, ("B", "N", "HV", 64, 128)], o: GM[f32, ("B", "T", "HV", 128)],
    B: i32, T: i32, HV: i32, N: i32,
):
    qu = Tensor(DT.float, [C, D], Position.UB)
    gu = Tensor(DT.float, [C, D], Position.UB)
    au = Tensor(DT.float, [C, C], Position.UB)
    su = Tensor(DT.float, [D, D], Position.UB)
    du = Tensor(DT.float, [C, D], Position.UB)
    ou = Tensor(DT.float, [C, D], Position.UB)
    per = CeilDiv(B * N * HV, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, B * N * HV)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % HV)
            cc = Var((item // HV) % N)
            bb = Var(item // (N * HV))
            qu[:, :] <<= qn[bb, cc, hh, :, :]
            gu[:, :] <<= gc[bb, cc, hh, :, :]
            au[:, :] <<= score[bb, cc, hh, :, :]
            su[:, :] <<= states[bb, cc, hh, :, :]
            du[:, :] <<= delta[bb, cc, hh, :, :]
            output_vf(qu, gu, au, su, du, ou)
            for ii in range(C):
                tt = Var(cc * C + ii)
                o[bb, tt:tt+1, hh, :] <<= ou[ii:ii+1, :]
    return o


STAGES = (pgdn_chunk_prepare, pgdn_chunk_scores, pgdn_chunk_wy,
          pgdn_chunk_scan, pgdn_chunk_output)
