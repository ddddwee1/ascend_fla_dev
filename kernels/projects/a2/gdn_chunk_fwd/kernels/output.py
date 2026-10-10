"""A2 GDN chunk-forward stage 5: output.

Tensor-vector port of a5 `gdn_chunk_fwd/kernels/stages.py::output_vf`. Per
(b, chunk, value-head), with S the state entering the chunk (from the scan):

  o[i,:] = (qn[i,:] * exp(gc[i])) @ S  +  sum_{j<=i} score[i,j] * delta[j,:]

qn is already scale-applied (prepare) and HV-indexed. The causal `score` from
stage 2 is zero above the diagonal, so the second term is a full matvec
score[i,:] @ delta (the zeros enforce j<=i). Both terms are matvecs computed with
the exact `_spread8`+rowscale+tree-reduce idiom the (validated) scan stage uses
for wy @ S: spread the per-row scalar, rowscale the matrix in two 64-lane value
halves, tree-reduce over the contracted rows, and accumulate into `uu`. Phase A
does qd @ S (reduce over the D key rows of S); phase B reuses wu for delta and
adds score @ delta (reduce over the C rows). The reduce and accumulate are kept
inline with explicit adds (folding them into a called helper mistraces on a2).
The query preweight is a plain elementwise tile op (g is a per-token scalar, gc
broadcast across D). Output is written in public token-major layout [B*T, HV*D].
"""
from ascriptor.a2 import *

D = 128
C = 64
GROUP = 64
NGROUP = D // GROUP           # 2
ROWBLK = D // 8              # 16
HALFBLK = GROUP // 8         # 8
FULL_MASK = (1 << 64) - 1


def _spread8(sp8, x_row, width):
    """sp8[j, 0:8] = x_row[0, j] for j in 0..width-1 (read from aligned offset 0)."""
    brcb(sp8, x_row[0:1, 0:width], repeat=width // 8, dst_blk_stride=1, dst_rep_stride=8)


@kernel(mode="vec", block_dim=40)
def gdn_chunk_output_a2_kernel(
    qn: GM[f32, ("B", "N", "HV", 64, 128)],
    gc: GM[f32, ("B", "N", "HV", 64, 128)],
    score: GM[f32, ("B", "N", "HV", 64, 64)],
    states: GM[f32, ("B", "N", "HV", 128, 128)],
    delta: GM[f32, ("B", "N", "HV", 64, 128)],
    o: GM[f32, ("BT", "HVD")],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    su = Tensor(DT.float, [D, D], Position.UB)       # state
    uu = Tensor(DT.float, [C, D], Position.UB)        # output accumulator
    wu = Tensor(DT.float, [C, D], Position.UB)        # qd (phase A); delta (phase B)
    au = Tensor(DT.float, [C, C], Position.UB)        # score (phase B)
    scr = Tensor(DT.float, [D, GROUP], Position.UB)   # reduce scratch (half V)
    sp8 = Tensor(DT.float, [D, 8], Position.UB)        # per-row scalar spread

    work = B * N * HV
    per = CeilDiv(work, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, work)

    with auto_sync():
        set_mask(FULL_MASK, FULL_MASK)
        for item in range(begin, end):
            hv = Var(item % HV)
            cc = Var((item // HV) % N)
            bb = Var(item // (N * HV))
            r0 = Var((bb * N + cc) * C)                # token row base (B*T major)
            hcol = Var(hv * D)

            # preweight query: wu = qd = qn * exp(gc) (uu is the exp scratch here)
            wu[0:C, 0:D] <<= qn[bb, cc, hv, 0:C, 0:D]
            uu[0:C, 0:D] <<= gc[bb, cc, hv, 0:C, 0:D]
            exp(uu[0:C, 0:D], uu[0:C, 0:D], count=C * D)
            mul(wu[0:C, 0:D], wu[0:C, 0:D], uu[0:C, 0:D], count=C * D)

            # phase A: uu = qd @ S  (reduce over the D key rows of S)
            su[0:D, 0:D] <<= states[bb, cc, hv, 0:D, 0:D]
            dup(uu[0:C, 0:D], 0.0, count=C * D)
            for i in range(C):
                _spread8(sp8, wu[i:i + 1, 0:D], D)
                for vh in range(NGROUP):
                    vs = vh * GROUP
                    mul(scr[0:D, 0:GROUP], su[0:D, vs:vs + GROUP], sp8,
                        repeat=D, count_per_rep=GROUP,
                        dst_blk_stride=1, dst_rep_stride=HALFBLK,
                        src1_blk_stride=1, src1_rep_stride=ROWBLK,
                        src2_blk_stride=0, src2_rep_stride=1)
                    add(scr[0:64, 0:GROUP], scr[0:64, 0:GROUP], scr[64:128, 0:GROUP], count=64 * GROUP)
                    add(scr[0:32, 0:GROUP], scr[0:32, 0:GROUP], scr[32:64, 0:GROUP], count=32 * GROUP)
                    add(scr[0:16, 0:GROUP], scr[0:16, 0:GROUP], scr[16:32, 0:GROUP], count=16 * GROUP)
                    add(scr[0:8, 0:GROUP], scr[0:8, 0:GROUP], scr[8:16, 0:GROUP], count=8 * GROUP)
                    add(scr[0:4, 0:GROUP], scr[0:4, 0:GROUP], scr[4:8, 0:GROUP], count=4 * GROUP)
                    add(scr[0:2, 0:GROUP], scr[0:2, 0:GROUP], scr[2:4, 0:GROUP], count=2 * GROUP)
                    add(scr[0:1, 0:GROUP], scr[0:1, 0:GROUP], scr[1:2, 0:GROUP], count=1 * GROUP)
                    add(uu[i:i + 1, vs:vs + GROUP], uu[i:i + 1, vs:vs + GROUP],
                        scr[0:1, 0:GROUP])

            # phase B: uu += score @ delta  (reduce over the C rows of delta)
            wu[0:C, 0:D] <<= delta[bb, cc, hv, 0:C, 0:D]
            au[0:C, 0:C] <<= score[bb, cc, hv, 0:C, 0:C]
            for i in range(C):
                _spread8(sp8, au[i:i + 1, 0:C], C)
                for vh in range(NGROUP):
                    vs = vh * GROUP
                    mul(scr[0:C, 0:GROUP], wu[0:C, vs:vs + GROUP], sp8,
                        repeat=C, count_per_rep=GROUP,
                        dst_blk_stride=1, dst_rep_stride=HALFBLK,
                        src1_blk_stride=1, src1_rep_stride=ROWBLK,
                        src2_blk_stride=0, src2_rep_stride=1)
                    add(scr[0:32, 0:GROUP], scr[0:32, 0:GROUP], scr[32:64, 0:GROUP], count=32 * GROUP)
                    add(scr[0:16, 0:GROUP], scr[0:16, 0:GROUP], scr[16:32, 0:GROUP], count=16 * GROUP)
                    add(scr[0:8, 0:GROUP], scr[0:8, 0:GROUP], scr[8:16, 0:GROUP], count=8 * GROUP)
                    add(scr[0:4, 0:GROUP], scr[0:4, 0:GROUP], scr[4:8, 0:GROUP], count=4 * GROUP)
                    add(scr[0:2, 0:GROUP], scr[0:2, 0:GROUP], scr[2:4, 0:GROUP], count=2 * GROUP)
                    add(scr[0:1, 0:GROUP], scr[0:1, 0:GROUP], scr[1:2, 0:GROUP], count=1 * GROUP)
                    add(uu[i:i + 1, vs:vs + GROUP], uu[i:i + 1, vs:vs + GROUP],
                        scr[0:1, 0:GROUP])

            for i in range(C):
                o[r0 + i:r0 + i + 1, hcol:hcol + D] <<= uu[i:i + 1, 0:D]

    return o
