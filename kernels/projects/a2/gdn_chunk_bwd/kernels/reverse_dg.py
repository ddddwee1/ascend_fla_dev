"""A2 GDN chunk-backward reverse, kernel: dg (per-token, no recurrence).

  dz    = (back_t . k)_K ; dr = beta * dz
  dD    = back_t - k ^ dr
  dg[t] = (dD . d_t)_{K,V}            (scalar full reduce)

One over-K `_kreduce` + one `muladddst` (dD, in place on the loaded back_t) + the dg
full reduce (tree-add over rows then a scalar cadd). No over-V row-build. Split out
of the combined dbeta+dg kernel to fit the sim's per-kernel sync budget. GVA k from
i_h.
"""
from ascriptor.a2 import *

D = 128
GROUP = 64
NGROUP = D // GROUP
ROWBLK = D // 8
HALFBLK = GROUP // 8
FULL_MASK = (1 << 64) - 1


def _spread8(sp8, x_row):
    brcb(sp8, x_row[0:1, 0:D], repeat=D // 8, dst_blk_stride=1, dst_rep_stride=8)


def _tree_k(scr):
    add(scr[0:64, 0:GROUP], scr[0:64, 0:GROUP], scr[64:128, 0:GROUP])
    add(scr[0:32, 0:GROUP], scr[0:32, 0:GROUP], scr[32:64, 0:GROUP])
    add(scr[0:16, 0:GROUP], scr[0:16, 0:GROUP], scr[16:32, 0:GROUP])
    add(scr[0:8, 0:GROUP], scr[0:8, 0:GROUP], scr[8:16, 0:GROUP])
    add(scr[0:4, 0:GROUP], scr[0:4, 0:GROUP], scr[4:8, 0:GROUP])
    add(scr[0:2, 0:GROUP], scr[0:2, 0:GROUP], scr[2:4, 0:GROUP])
    add(scr[0:1, 0:GROUP], scr[0:1, 0:GROUP], scr[1:2, 0:GROUP])


def _kreduce(dstrow, mat, sp8, scr):
    for vh in range(NGROUP):
        vs = vh * GROUP
        mul(scr[0:D, 0:GROUP], mat[0:D, vs:vs + GROUP], sp8,
            repeat=D, count_per_rep=GROUP,
            dst_blk_stride=1, dst_rep_stride=HALFBLK,
            src1_blk_stride=1, src1_rep_stride=ROWBLK,
            src2_blk_stride=0, src2_rep_stride=1)
        _tree_k(scr)
        adds(dstrow[0:1, vs:vs + GROUP], scr[0:1, 0:GROUP], 0.0, count=GROUP)


@kernel(mode="vec", block_dim=40)
def gdn_chunk_bwd_reverse_dg_a2_kernel(
    k: GM[f32, ("BT", "HD")],
    beta: GM[f32, ("BT", "HV")],
    tape_d: GM[f32, ("B", "HV", "T", 128, 128)],
    back_tape: GM[f32, ("B", "HV", "T", 128, 128)],
    dg: GM[f32, ("BT", "HV")],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    back = Tensor(DT.float, [D, D], Position.UB)         # back_t (becomes dD)
    su = Tensor(DT.float, [D, D], Position.UB)            # d_t
    scr = Tensor(DT.float, [D, GROUP], Position.UB)
    sp8 = Tensor(DT.float, [D, 8], Position.UB)
    krow = Tensor(DT.float, [1, D], Position.UB)
    dzrow = Tensor(DT.float, [1, D], Position.UB)
    tmp = Tensor(DT.float, [1, D], Position.UB)
    sc1 = Tensor(DT.float, [1, 8], Position.UB)

    work = B * HV
    per = CeilDiv(work, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, work)

    with auto_sync():
        set_mask(FULL_MASK, FULL_MASK)
        hvh = Var(HV // H)
        for item in range(begin, end):
            hv = Var(item % HV)
            bb = Var(item // HV)
            ih = Var(hv // hvh)
            for t in range(T):
                rr = Var(bb * T + t)
                krow[0:1, 0:D] <<= k[rr:rr + 1, ih * D:ih * D + D]
                bval = Var(0.0, dtype=DT.float); bval.GetValueFrom(beta[rr:rr + 1, hv:hv + 1])
                su[0:D, 0:D] <<= tape_d[bb, hv, t, 0:D, 0:D]
                back[0:D, 0:D] <<= back_tape[bb, hv, t, 0:D, 0:D]
                # dz = (back.k)_K ; dr = beta*dz
                _spread8(sp8, krow)
                _kreduce(dzrow, back, sp8, scr)
                muls(dzrow[0:1, 0:D], dzrow[0:1, 0:D], bval, count=D)          # dr
                # dD = back - k ^ dr  (in place)
                muls(tmp[0:1, 0:D], dzrow[0:1, 0:D], -1.0, count=D)
                for vh in range(NGROUP):
                    vs = vh * GROUP
                    muladddst(back[0:D, vs:vs + GROUP], sp8, tmp[0:1, vs:vs + GROUP],
                              repeat=D, count_per_rep=GROUP,
                              dst_blk_stride=1, dst_rep_stride=ROWBLK,
                              src1_blk_stride=0, src1_rep_stride=1,
                              src2_blk_stride=1, src2_rep_stride=0)
                # dg = sum_{K,V} dD * d
                dgacc = Var(0.0, dtype=DT.float)
                for gi in range(NGROUP):
                    s = gi * GROUP
                    mul(scr[0:D, 0:GROUP], back[0:D, s:s + GROUP], su[0:D, s:s + GROUP],
                        repeat=D, count_per_rep=GROUP,
                        dst_blk_stride=1, dst_rep_stride=HALFBLK,
                        src1_blk_stride=1, src1_rep_stride=ROWBLK,
                        src2_blk_stride=1, src2_rep_stride=ROWBLK)
                    _tree_k(scr)
                    dup(sc1[0:1, 0:1], 0.0, count=1)
                    cadd(sc1[0:1, 0:1], scr[0:1, 0:GROUP], repeat=1, count_per_rep=GROUP,
                         src_blk_stride=1, src_rep_stride=1, dst_rep_stride=1)
                    gpart = Var(0.0, dtype=DT.float); gpart.GetValueFrom(sc1[0:1, 0:1])
                    dgacc.set(dgacc + gpart)
                dgacc.SetValueTo(sc1[0:1, 0:1])
                dg[rr:rr + 1, hv:hv + 1] <<= sc1[0:1, 0:1]

    return dg
