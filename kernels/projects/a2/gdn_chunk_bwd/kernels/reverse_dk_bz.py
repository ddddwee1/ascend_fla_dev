"""A2 GDN chunk-backward reverse, kernel: dk term 1 = (back_t . z)_V.

  z = beta * (v - (k . d_t)_K) ;  dkbz[t] = (back_t . z)_V

One over-K `_kreduce` + one over-V `_rowdot` (same shape as the validated dq term).
dk = dkbz - dkddr is finished in group_reduce. Split from the monolithic dk
(2 kreduce + 2 _rowdot), which exceeded the sim budget. GVA k from i_h.
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


def _rowdot(dstrow, dtmp, mat, brow, scr):
    dup(dstrow[0:1, 0:D], 0.0, count=D)
    dup(dtmp[0:1, 0:D], 0.0, count=D)
    for gi in range(NGROUP):
        s = gi * GROUP
        mul(scr[0:D, 0:GROUP], mat[0:D, s:s + GROUP], brow[0:1, s:s + GROUP],
            repeat=D, count_per_rep=GROUP,
            dst_blk_stride=1, dst_rep_stride=HALFBLK,
            src1_blk_stride=1, src1_rep_stride=ROWBLK,
            src2_blk_stride=1, src2_rep_stride=0)
        if gi == 0:
            cadd(dstrow[0:1, 0:D], scr[0:D, 0:GROUP], repeat=D, count_per_rep=GROUP,
                 src_blk_stride=1, src_rep_stride=HALFBLK, dst_rep_stride=1)
        else:
            cadd(dtmp[0:1, 0:D], scr[0:D, 0:GROUP], repeat=D, count_per_rep=GROUP,
                 src_blk_stride=1, src_rep_stride=HALFBLK, dst_rep_stride=1)
    add(dstrow[0:1, 0:D], dstrow[0:1, 0:D], dtmp[0:1, 0:D], count=D)


@kernel(mode="vec", block_dim=40)
def gdn_chunk_bwd_reverse_dk_bz_a2_kernel(
    k: GM[f32, ("BT", "HD")],
    v: GM[f32, ("BT", "HVD")],
    beta: GM[f32, ("BT", "HV")],
    tape_d: GM[f32, ("B", "HV", "T", 128, 128)],
    back_tape: GM[f32, ("B", "HV", "T", 128, 128)],
    dkbz_parts: GM[f32, ("BT", "HVD")],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    back = Tensor(DT.float, [D, D], Position.UB)
    su = Tensor(DT.float, [D, D], Position.UB)
    scr = Tensor(DT.float, [D, GROUP], Position.UB)
    sp8 = Tensor(DT.float, [D, 8], Position.UB)
    krow = Tensor(DT.float, [1, D], Position.UB)
    vrow = Tensor(DT.float, [1, D], Position.UB)
    zrow = Tensor(DT.float, [1, D], Position.UB)
    outrow = Tensor(DT.float, [1, D], Position.UB)
    dtmp = Tensor(DT.float, [1, D], Position.UB)

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
                vrow[0:1, 0:D] <<= v[rr:rr + 1, hv * D:hv * D + D]
                bval = Var(0.0, dtype=DT.float); bval.GetValueFrom(beta[rr:rr + 1, hv:hv + 1])
                su[0:D, 0:D] <<= tape_d[bb, hv, t, 0:D, 0:D]
                back[0:D, 0:D] <<= back_tape[bb, hv, t, 0:D, 0:D]
                _spread8(sp8, krow)
                _kreduce(zrow, su, sp8, scr)
                sub(zrow[0:1, 0:D], vrow[0:1, 0:D], zrow[0:1, 0:D], count=D)
                muls(zrow[0:1, 0:D], zrow[0:1, 0:D], bval, count=D)            # z
                _rowdot(outrow, dtmp, back, zrow, scr)                        # (back.z)_V
                dkbz_parts[rr:rr + 1, hv * D:hv * D + D] <<= outrow[0:1, 0:D]

    return dkbz_parts
