"""A2 GDN chunk-forward stage 2: scores.

Tensor-vector port of a5 `gdn_chunk_fwd/kernels/stages.py::gdn_chunk_scores`.
Per (b, chunk, value-head), for j <= i (causal):

  score[i, j] = exp(gc[i] - gc[j]) * (qn[i] . kn[j])          (j <= i)
  lower[i, j] = exp(gc[i] - gc[j]) * (bk[i] . kn[j])          (j <  i)

GDN's g is a per-token scalar, so gc[i] is constant across channels and the decay
factors out of the dot: exp(gc[i]-gc[j]) = exp(gc[i]) * exp(-gc[j]). We therefore
pre-scale whole tiles once — qnd = qn*exp(gc), knd = kn*exp(-gc), bkd = bk*exp(gc)
— and the scores are plain dot products qnd.knd^T / bkd.knd^T (no decay matrix, no
[C,1]<->[1,C] transpose). The causal triangle is applied by writing only the valid
rows of each key-column into a pre-zeroed tile (no compare/select).

NOTE (Batch A scope): this split-factor form is exact but its intermediate
exp(-gc[j]) grows with the gate span; A2-K1 validates only at small, safe spans
(<= 10, far from the ~88.7 line). The midpoint-anchored decomposition for large
spans is Batch B (gate calibration), out of this task.
"""
from ascriptor.a2 import *

D = 128
C = 64
GROUP = 64
NGROUP = D // GROUP        # 2
ROWBLK = D // 8            # 16
FULL_MASK = (1 << 64) - 1


def _coldot(dst_col, aa, bb_row, tmp):
    """dst_col[i, 0] = sum_d aa[i, d] * bb_row[0, d]  (bb_row broadcast down rows)."""
    for gi in range(NGROUP):
        s = gi * GROUP
        mul(tmp[0:C, s:s + GROUP], aa[0:C, s:s + GROUP], bb_row[0:1, s:s + GROUP],
            repeat=C, count_per_rep=GROUP,
            dst_blk_stride=1, dst_rep_stride=ROWBLK,
            src1_blk_stride=1, src1_rep_stride=ROWBLK,
            src2_blk_stride=1, src2_rep_stride=0)
    add(tmp[0:C, 0:GROUP], tmp[0:C, 0:GROUP], tmp[0:C, GROUP:D],
        repeat=C, count_per_rep=GROUP,
        dst_blk_stride=1, dst_rep_stride=ROWBLK,
        src1_blk_stride=1, src1_rep_stride=ROWBLK,
        src2_blk_stride=1, src2_rep_stride=ROWBLK)
    # cadd's dst_rep_stride counts elements (not 8-elem blocks): scol row stride
    # is 8, so result i lands at scol[i, 0].
    cadd(dst_col[0:C, 0:1], tmp[0:C, 0:GROUP],
         repeat=C, count_per_rep=GROUP,
         src_blk_stride=1, src_rep_stride=ROWBLK, dst_rep_stride=8)


@kernel(mode="vec", block_dim=40)
def gdn_chunk_scores_a2_kernel(
    qn: GM[f32, ("B", "N", "HV", 64, 128)],
    kn: GM[f32, ("B", "N", "HV", 64, 128)],
    gc: GM[f32, ("B", "N", "HV", 64, 128)],
    bk: GM[f32, ("B", "N", "HV", 64, 128)],
    lower: GM[f32, ("B", "N", "HV", 64, 64)],
    score: GM[f32, ("B", "N", "HV", 64, 64)],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    qu = Tensor(DT.float, [C, D], Position.UB)
    ku = Tensor(DT.float, [C, D], Position.UB)
    bu = Tensor(DT.float, [C, D], Position.UB)
    ex = Tensor(DT.float, [C, D], Position.UB)
    tmp = Tensor(DT.float, [C, D], Position.UB)
    scol = Tensor(DT.float, [C, 8], Position.UB)   # cadd writes one 8-block per row
    au = Tensor(DT.float, [C, C], Position.UB)

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

            qu[0:C, 0:D] <<= qn[bb, cc, hv, 0:C, 0:D]
            ku[0:C, 0:D] <<= kn[bb, cc, hv, 0:C, 0:D]
            bu[0:C, 0:D] <<= bk[bb, cc, hv, 0:C, 0:D]

            # knd = kn * exp(-gc)
            ex[0:C, 0:D] <<= gc[bb, cc, hv, 0:C, 0:D]
            muls(ex[0:C, 0:D], ex[0:C, 0:D], -1.0, count=C * D)   # explicit count (device vec repeat)
            exp(ex, ex, count=C * D)
            mul(ku, ku, ex, count=C * D)
            # qnd = qn * exp(gc) ; bkd = bk * exp(gc)
            ex[0:C, 0:D] <<= gc[bb, cc, hv, 0:C, 0:D]
            exp(ex, ex, count=C * D)
            mul(qu, qu, ex, count=C * D)
            mul(bu, bu, ex, count=C * D)

            # score = qnd . knd^T, causal (keep i >= j). A strided sub-column UB
            # store faults on the vector core, so scatter the valid entries with
            # scalar Var copies (which also apply the triangular mask).
            dup(au[0:C, 0:C], 0.0, count=C * C)
            for j in range(C):
                _coldot(scol, qu, ku[j:j + 1, 0:D], tmp)
                for i in range(j, C):
                    val = Var(0.0, dtype=DT.float)
                    val.GetValueFrom(scol[i:i + 1, 0:1])
                    val.SetValueTo(au[i:i + 1, j:j + 1])
            score[bb, cc, hv, 0:C, 0:C] <<= au[0:C, 0:C]

            # lower = bkd . knd^T, strictly causal (keep i > j)
            dup(au[0:C, 0:C], 0.0, count=C * C)
            for j in range(C - 1):
                _coldot(scol, bu, ku[j:j + 1, 0:D], tmp)
                for i in range(j + 1, C):
                    val = Var(0.0, dtype=DT.float)
                    val.GetValueFrom(scol[i:i + 1, 0:1])
                    val.SetValueTo(au[i:i + 1, j:j + 1])
            lower[bb, cc, hv, 0:C, 0:C] <<= au[0:C, 0:C]

    return lower, score
