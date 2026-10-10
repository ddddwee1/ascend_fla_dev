"""A2 GDN chunk-backward stage 1: replay (tape the per-token decayed state).

Replays the GDN primal recurrence per (b, value-head) and writes every token's
decayed state d_t = exp(g[t]) * S_{t-1} to `tape_d[B,HV,T,128,128]`, so the reverse
pass reads d_t directly (a plain GM producer->consumer flow; no in-kernel GM
round-trip, which deadlocks the sim). Same recurrence and idioms as the validated
`checkpoints` kernel; only the save point differs (per-token d instead of
chunk-boundary S). Pure vector. GVA k from i_h = hv // (HV//H).
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


@kernel(mode="vec", block_dim=40)
def gdn_chunk_bwd_replay_a2_kernel(
    k: GM[f32, ("BT", "HD")],
    v: GM[f32, ("BT", "HVD")],
    g: GM[f32, ("BT", "HV")],
    beta: GM[f32, ("BT", "HV")],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    tape_d: GM[f32, ("B", "HV", "T", 128, 128)],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    su = Tensor(DT.float, [D, D], Position.UB)
    scr = Tensor(DT.float, [D, GROUP], Position.UB)
    sp8 = Tensor(DT.float, [D, 8], Position.UB)
    krow = Tensor(DT.float, [1, D], Position.UB)
    vrow = Tensor(DT.float, [1, D], Position.UB)
    zrow = Tensor(DT.float, [1, D], Position.UB)
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

            su[0:D, 0:D] <<= initial_state[bb, hv, 0:D, 0:D]
            for t in range(T):
                rr = Var(bb * T + t)
                krow[0:1, 0:D] <<= k[rr:rr + 1, ih * D:ih * D + D]
                vrow[0:1, 0:D] <<= v[rr:rr + 1, hv * D:hv * D + D]
                gval = Var(0.0, dtype=DT.float); gval.GetValueFrom(g[rr:rr + 1, hv:hv + 1])
                bval = Var(0.0, dtype=DT.float); bval.GetValueFrom(beta[rr:rr + 1, hv:hv + 1])
                dup(tmp[0:1, 0:D], 0.0, count=D)
                adds(tmp[0:1, 0:D], tmp[0:1, 0:D], gval, count=D)
                exp(tmp[0:1, 0:D], tmp[0:1, 0:D], count=D)
                egv = Var(0.0, dtype=DT.float); egv.GetValueFrom(tmp[0:1, 0:1])
                muls(su[0:64, 0:D], su[0:64, 0:D], egv, count=64 * D)
                muls(su[64:128, 0:D], su[64:128, 0:D], egv, count=64 * D)
                tape_d[bb, hv, t, 0:D, 0:D] <<= su[0:D, 0:D]           # d_t
                _spread8(sp8, krow)
                for vh in range(NGROUP):
                    vs = vh * GROUP
                    mul(scr[0:D, 0:GROUP], su[0:D, vs:vs + GROUP], sp8,
                        repeat=D, count_per_rep=GROUP,
                        dst_blk_stride=1, dst_rep_stride=HALFBLK,
                        src1_blk_stride=1, src1_rep_stride=ROWBLK,
                        src2_blk_stride=0, src2_rep_stride=1)
                    _tree_k(scr)
                    sub(zrow[0:1, vs:vs + GROUP], vrow[0:1, vs:vs + GROUP],
                        scr[0:1, 0:GROUP], count=GROUP)               # r half
                muls(zrow[0:1, 0:D], zrow[0:1, 0:D], bval, count=D)   # z = beta*r
                for vh in range(NGROUP):
                    vs = vh * GROUP
                    muladddst(su[0:D, vs:vs + GROUP], sp8, zrow[0:1, vs:vs + GROUP],
                              repeat=D, count_per_rep=GROUP,
                              dst_blk_stride=1, dst_rep_stride=ROWBLK,
                              src1_blk_stride=0, src1_rep_stride=1,
                              src2_blk_stride=1, src2_rep_stride=0)

    return tape_d
