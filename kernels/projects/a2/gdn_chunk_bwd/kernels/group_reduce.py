"""A2 GDN chunk-backward stage: group-reduce dq/dk parts (GVA ratio-sum + dk combine).

The reverse forms per value-head (HV) parts; the qk-side gradients are per qk-head
(H). This sums each qk-head's `ratio = HV//H` value-head parts, and finishes dk from
its two split terms (dk = (back.z)_V - (d.dr)_V):

  dq[t, h, :] = sum_{r<ratio} dq_parts[t, h*ratio+r, :]
  dk[t, h, :] = sum_{r<ratio} ( dkbz_parts[..] - dkddr_parts[..] )

A no-op-ish copy when HV == H. One core per (b, t, qk-head); plain [1,128]
accumulation, low complexity. Pure vector.
"""
from ascriptor.a2 import *

D = 128
FULL_MASK = (1 << 64) - 1


@kernel(mode="vec", block_dim=40)
def gdn_chunk_bwd_group_reduce_a2_kernel(
    dq_parts: GM[f32, ("BT", "HVD")],
    dkbz_parts: GM[f32, ("BT", "HVD")],
    dkddr_parts: GM[f32, ("BT", "HVD")],
    dq: GM[f32, ("BT", "HD")],
    dk: GM[f32, ("BT", "HD")],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    accq = Tensor(DT.float, [1, D], Position.UB)
    acck = Tensor(DT.float, [1, D], Position.UB)
    segq = Tensor(DT.float, [1, D], Position.UB)
    segb = Tensor(DT.float, [1, D], Position.UB)
    segd = Tensor(DT.float, [1, D], Position.UB)

    work = B * T * H
    per = CeilDiv(work, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, work)

    with auto_sync():
        set_mask(FULL_MASK, FULL_MASK)
        ratio = Var(HV // H)
        for item in range(begin, end):
            h = Var(item % H)
            bt = Var(item // H)
            hv0 = Var(h * ratio)

            accq[0:1, 0:D] <<= dq_parts[bt:bt + 1, hv0 * D:hv0 * D + D]
            segb[0:1, 0:D] <<= dkbz_parts[bt:bt + 1, hv0 * D:hv0 * D + D]
            segd[0:1, 0:D] <<= dkddr_parts[bt:bt + 1, hv0 * D:hv0 * D + D]
            sub(acck[0:1, 0:D], segb[0:1, 0:D], segd[0:1, 0:D], count=D)
            for r in range(1, ratio):
                hv = Var(hv0 + r)
                segq[0:1, 0:D] <<= dq_parts[bt:bt + 1, hv * D:hv * D + D]
                add(accq[0:1, 0:D], accq[0:1, 0:D], segq[0:1, 0:D], count=D)
                segb[0:1, 0:D] <<= dkbz_parts[bt:bt + 1, hv * D:hv * D + D]
                segd[0:1, 0:D] <<= dkddr_parts[bt:bt + 1, hv * D:hv * D + D]
                sub(segb[0:1, 0:D], segb[0:1, 0:D], segd[0:1, 0:D], count=D)
                add(acck[0:1, 0:D], acck[0:1, 0:D], segb[0:1, 0:D], count=D)
            dq[bt:bt + 1, h * D:h * D + D] <<= accq[0:1, 0:D]
            dk[bt:bt + 1, h * D:h * D + D] <<= acck[0:1, 0:D]

    return dq, dk
