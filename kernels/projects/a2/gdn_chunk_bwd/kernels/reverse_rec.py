"""A2 GDN chunk-backward reverse, kernel 1 of the decomposition: the recurrence.

The monolithic reverse overruns both the sim's auto_sync budget and the board
build time, so it is split into small kernels each with a per-token body as simple
as the validated checkpoints/replay. This one carries the state cotangent `back`
across the reverse scan and emits only what needs the recurrence -- dv, dh0, and the
per-token cotangent tape `back_t` (post q^do, pre dD) that the per-token dq/dbeta/dg/
dk kernels consume:

  back += scale * q ^ do ;  back_tape[t] = back
  dz = (back . k)_K ; dr = beta * dz ; dv[t] = dr
  back = exp(g) * (back - k ^ dr)

No over-V `cadd(dst_rep_stride=1)` row-builds and no per-token dg reduce here (dg is
a separate kernel), so this stays within the sim budget (validated: dv relL2 1.3e-7,
back_tape/dh0 finite). GVA k/q from i_h = hv//(HV//H). Pure vector. `dh0` = final back.
"""
from ascriptor.a2 import *

D = 128
C = 64
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
def gdn_chunk_bwd_reverse_rec_a2_kernel(
    q: GM[f32, ("BT", "HD")],
    k: GM[f32, ("BT", "HD")],
    g: GM[f32, ("BT", "HV")],
    beta: GM[f32, ("BT", "HV")],
    dout: GM[f32, ("BT", "HVD")],
    dht: GM[f32, ("B", "HV", 128, 128)],
    dv: GM[f32, ("BT", "HVD")],
    dh0: GM[f32, ("B", "HV", 128, 128)],
    back_tape: GM[f32, ("B", "HV", "T", 128, 128)],
    scale: f32,
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    back = Tensor(DT.float, [D, D], Position.UB)
    scr = Tensor(DT.float, [D, GROUP], Position.UB)
    sp8 = Tensor(DT.float, [D, 8], Position.UB)
    krow = Tensor(DT.float, [1, D], Position.UB)
    qrow = Tensor(DT.float, [1, D], Position.UB)
    dorow = Tensor(DT.float, [1, D], Position.UB)
    drrow = Tensor(DT.float, [1, D], Position.UB)
    tmp = Tensor(DT.float, [1, D], Position.UB)

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
            back[0:D, 0:D] <<= dht[bb, hv, 0:D, 0:D]
            for ccx in range(N):
                cc = Var(N - 1 - ccx)
                r0 = Var(bb * T + cc * C)
                for ix in range(C):
                    i = C - 1 - ix
                    rr = Var(r0 + i)
                    tt = Var(cc * C + i)
                    krow[0:1, 0:D] <<= k[rr:rr + 1, ih * D:ih * D + D]
                    qrow[0:1, 0:D] <<= q[rr:rr + 1, ih * D:ih * D + D]
                    dorow[0:1, 0:D] <<= dout[rr:rr + 1, hv * D:hv * D + D]
                    gval = Var(0.0, dtype=DT.float); gval.GetValueFrom(g[rr:rr + 1, hv:hv + 1])
                    bval = Var(0.0, dtype=DT.float); bval.GetValueFrom(beta[rr:rr + 1, hv:hv + 1])

                    # back += scale*q ^ do ; tape back_t
                    muls(tmp[0:1, 0:D], qrow[0:1, 0:D], scale, count=D)
                    _spread8(sp8, tmp)
                    for vh in range(NGROUP):
                        vs = vh * GROUP
                        muladddst(back[0:D, vs:vs + GROUP], sp8, dorow[0:1, vs:vs + GROUP],
                                  repeat=D, count_per_rep=GROUP,
                                  dst_blk_stride=1, dst_rep_stride=ROWBLK,
                                  src1_blk_stride=0, src1_rep_stride=1,
                                  src2_blk_stride=1, src2_rep_stride=0)
                    back_tape[bb, hv, tt, 0:D, 0:D] <<= back[0:D, 0:D]

                    # dz = (back.k)_K ; dr = beta*dz ; dv = dr
                    _spread8(sp8, krow)
                    _kreduce(drrow, back, sp8, scr)
                    muls(drrow[0:1, 0:D], drrow[0:1, 0:D], bval, count=D)
                    dv[rr:rr + 1, hv * D:hv * D + D] <<= drrow[0:1, 0:D]

                    # back = exp(g) * (back - k ^ dr)
                    muls(tmp[0:1, 0:D], drrow[0:1, 0:D], -1.0, count=D)
                    _spread8(sp8, krow)
                    for vh in range(NGROUP):
                        vs = vh * GROUP
                        muladddst(back[0:D, vs:vs + GROUP], sp8, tmp[0:1, vs:vs + GROUP],
                                  repeat=D, count_per_rep=GROUP,
                                  dst_blk_stride=1, dst_rep_stride=ROWBLK,
                                  src1_blk_stride=0, src1_rep_stride=1,
                                  src2_blk_stride=1, src2_rep_stride=0)
                    dup(tmp[0:1, 0:D], 0.0, count=D)
                    adds(tmp[0:1, 0:D], tmp[0:1, 0:D], gval, count=D)
                    exp(tmp[0:1, 0:D], tmp[0:1, 0:D], count=D)
                    egv = Var(0.0, dtype=DT.float); egv.GetValueFrom(tmp[0:1, 0:1])
                    muls(back[0:64, 0:D], back[0:64, 0:D], egv, count=64 * D)
                    muls(back[64:128, 0:D], back[64:128, 0:D], egv, count=64 * D)
            dh0[bb, hv, 0:D, 0:D] <<= back[0:D, 0:D]

    return dv, dh0, back_tape
