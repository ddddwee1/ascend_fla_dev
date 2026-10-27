"""PKDA FP32 causal score, triangular solve and state scan.

The output stage lives in ``output.py``: it is two plain GEMMs and runs on the
cube engine, like the BF16 sibling unit's output kernel.

Derived from the repository GDN-2 Vector implementation at 375d8f9, applying
KDA's stable five-stage decomposition. Here kn is the preconditioned WRITE
key while bk is beta times the original READ key; they are not interchangeable.
All local buffers are single-slot; original repositories are read-only.
"""
from ascriptor.a5 import *

D = 128
C = 64

@vf()
def scores_vf(q: Tensor, k: Tensor, g: Tensor, b: Tensor, lower: Tensor,
              score: Tensor, count: Var):
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
    for i in range(count):
        qr <<= q[i:i+1, 0:D]
        gi <<= g[i:i+1, 0:D]
        br <<= b[i:i+1, 0:D]
        for j in range(i + 1):
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
def pkda_chunk_scores(
    qn: GM[f32, ("B", "N", "H", 64, 128)], kn: GM[f32, ("B", "N", "H", 64, 128)],
    gc: GM[f32, ("B", "N", "H", 64, 128)], bk: GM[f32, ("B", "N", "H", 64, 128)],
    lower: GM[f32, ("B", "N", "H", 64, 64)], score: GM[f32, ("B", "N", "H", 64, 64)],
    B: i32, T: i32, H: i32, N: i32,
):
    qu = Tensor(DT.float, [C, D], Position.UB)
    ku = Tensor(DT.float, [C, D], Position.UB)
    gu = Tensor(DT.float, [C, D], Position.UB)
    bu = Tensor(DT.float, [C, D], Position.UB)
    lu = Tensor(DT.float, [C, C], Position.UB)
    au = Tensor(DT.float, [C, C], Position.UB)
    per = CeilDiv(B * N * H, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, B * N * H)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % H)
            cc = Var((item // H) % N)
            bb = Var(item // (N * H))
            count = Var(Min(C, T - cc * C))
            qu[:, :] <<= qn[bb, cc, hh, :, :]
            ku[:, :] <<= kn[bb, cc, hh, :, :]
            gu[:, :] <<= gc[bb, cc, hh, :, :]
            bu[:, :] <<= bk[bb, cc, hh, :, :]
            scores_vf(qu, ku, gu, bu, lu, au, count)
            lower[bb, cc, hh, :, :] <<= lu[:, :]
            score[bb, cc, hh, :, :] <<= au[:, :]
    return lower, score


@vf()
def wy_vf(lower: Tensor, g: Tensor, b: Tensor, v: Tensor,
          u: Tensor, wy: Tensor, count: Var):
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
    # Two rows share every earlier U/W load. Even i keeps i+1 inside C;
    # an odd-count partner is initialized zero padding in b/v/lower.
    for i in range(0, count, 2):
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
def pkda_chunk_wy(
    lower: GM[f32, ("B", "N", "H", 64, 64)], gc: GM[f32, ("B", "N", "H", 64, 128)],
    bk: GM[f32, ("B", "N", "H", 64, 128)], wv: GM[f32, ("B", "N", "H", 64, 128)],
    u: GM[f32, ("B", "N", "H", 64, 128)], wy: GM[f32, ("B", "N", "H", 64, 128)],
    B: i32, T: i32, H: i32, N: i32,
):
    lu = Tensor(DT.float, [C, C], Position.UB)
    gu = Tensor(DT.float, [C, D], Position.UB)
    bu = Tensor(DT.float, [C, D], Position.UB)
    vu = Tensor(DT.float, [C, D], Position.UB)
    uu = Tensor(DT.float, [C, D], Position.UB)
    wu = Tensor(DT.float, [C, D], Position.UB)
    per = CeilDiv(B * N * H, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, B * N * H)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % H)
            cc = Var((item // H) % N)
            bb = Var(item // (N * H))
            count = Var(Min(C, T - cc * C))
            lu[:, :] <<= lower[bb, cc, hh, :, :]
            gu[:, :] <<= gc[bb, cc, hh, :, :]
            bu[:, :] <<= bk[bb, cc, hh, :, :]
            vu[:, :] <<= wv[bb, cc, hh, :, :]
            wy_vf(lu, gu, bu, vu, uu, wu, count)
            u[bb, cc, hh, :, :] <<= uu[:, :]
            wy[bb, cc, hh, :, :] <<= wu[:, :]
    return u, wy


@vf()
def scan_vf(state: Tensor, k: Tensor, g: Tensor, u: Tensor,
            wy: Tensor, delta: Tensor, count: Var):
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
    for i in range(count):
        acc <<= u[i:i+1, 0:D]
        for d in range(D):
            weight <<= wy[i:i+1, d:d+1].single()
            row <<= state[d:d+1, 0:D]
            tmp <<= row * weight
            acc <<= acc - tmp
        delta[i:i+1, 0:D] <<= acc
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    # Raw keys have no later consumer. Preweight the complete channel row
    # once, avoiding an exponential broadcast for every (row, channel).
    acc <<= g[C-1:C, 0:D]
    for i in range(count):
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
        for i in range(count):
            weight <<= k[i:i+1, d:d+1].single()
            tmp <<= delta[i:i+1, 0:D]
            tmp <<= tmp * weight
            row <<= row + tmp
        state[d:d+1, 0:D] <<= row
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def pkda_chunk_scan(
    kn: GM[f32, ("B", "N", "H", 64, 128)], gc: GM[f32, ("B", "N", "H", 64, 128)],
    u: GM[f32, ("B", "N", "H", 64, 128)], wy: GM[f32, ("B", "N", "H", 64, 128)],
    initial_state: GM[f32, ("B", "H", 128, 128)],
    states: GM[f32, ("B", "N", "H", 128, 128)], delta: GM[f32, ("B", "N", "H", 64, 128)],
    final_state: GM[f32, ("B", "H", 128, 128)], B: i32, T: i32, H: i32, N: i32,
):
    su = Tensor(DT.float, [D, D], Position.UB)
    ku = Tensor(DT.float, [C, D], Position.UB)
    gu = Tensor(DT.float, [C, D], Position.UB)
    uu = Tensor(DT.float, [C, D], Position.UB)
    wu = Tensor(DT.float, [C, D], Position.UB)
    du = Tensor(DT.float, [C, D], Position.UB)
    per = CeilDiv(B * H, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, B * H)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % H)
            bb = Var(item // H)
            su[:, :] <<= initial_state[bb, hh, :, :]
            for cc in range(N):
                count = Var(Min(C, T - cc * C))
                states[bb, cc, hh, :, :] <<= su[:, :]
                ku[:, :] <<= kn[bb, cc, hh, :, :]
                gu[:, :] <<= gc[bb, cc, hh, :, :]
                uu[:, :] <<= u[bb, cc, hh, :, :]
                wu[:, :] <<= wy[bb, cc, hh, :, :]
                scan_vf(su, ku, gu, uu, wu, du, count)
                delta[bb, cc, hh, :, :] <<= du[:, :]
            final_state[bb, hh, :, :] <<= su[:, :]
    return states, delta, final_state


STAGES = (pkda_chunk_scores, pkda_chunk_wy, pkda_chunk_scan)
