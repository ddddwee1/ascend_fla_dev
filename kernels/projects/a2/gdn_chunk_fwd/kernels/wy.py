"""A2 GDN chunk-forward stage 3: WY (ordered triangular solve).

Tensor-vector port of a5 `gdn_chunk_fwd/kernels/stages.py::wy_vf`. Per
(b, chunk, value-head), solve the unit-lower-triangular system for the WY
representation of the chunk update:

  u[i]  = wv[i]           - sum_{j<i} lower[i,j] * u[j]
  wy[i] = bk[i]*exp(gc[i]) - sum_{j<i} lower[i,j] * wy[j]

where `lower` is the strictly-lower score matrix from stage 2, `wv = v*beta`,
`bk = k*beta`, and `exp(gc[i])` is the per-token scalar decay (gc broadcast).

Unlike the a5 `@vf` version (which processes rows in pairs to reuse register
loads and adds a special i+1<-i coupling term), a2 has no `@vf`; we solve rows
strictly in increasing order, so every `u[j]/wy[j]` with j<i is already final
when row i is formed. No two-row coupling term is needed. Per-row scalars
lower[i,j] use the proven Var GetValueFrom idiom (kda_bwd_stable). T % 64 == 0
is required upstream (gap §1.7), so count is always C and there is no tail path.
"""
from ascriptor.a2 import *

D = 128
C = 64
FULL_MASK = (1 << 64) - 1


@kernel(mode="vec", block_dim=40)
def gdn_chunk_wy_a2_kernel(
    lower: GM[f32, ("B", "N", "HV", 64, 64)],
    gc: GM[f32, ("B", "N", "HV", 64, 128)],
    bk: GM[f32, ("B", "N", "HV", 64, 128)],
    wv: GM[f32, ("B", "N", "HV", 64, 128)],
    u: GM[f32, ("B", "N", "HV", 64, 128)],
    wy: GM[f32, ("B", "N", "HV", 64, 128)],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    lu = Tensor(DT.float, [C, C], Position.UB)
    gu = Tensor(DT.float, [C, D], Position.UB)
    bu = Tensor(DT.float, [C, D], Position.UB)
    uu = Tensor(DT.float, [C, D], Position.UB)   # u   (init = wv)
    wu = Tensor(DT.float, [C, D], Position.UB)   # wy  (init = bk*exp(gc))
    tmp = Tensor(DT.float, [1, D], Position.UB)

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

            lu[0:C, 0:C] <<= lower[bb, cc, hv, 0:C, 0:C]
            gu[0:C, 0:D] <<= gc[bb, cc, hv, 0:C, 0:D]
            bu[0:C, 0:D] <<= bk[bb, cc, hv, 0:C, 0:D]
            uu[0:C, 0:D] <<= wv[bb, cc, hv, 0:C, 0:D]

            # wy init = bk * exp(gc)  (per-row scalar decay, gc broadcast)
            exp(gu, gu, count=C * D)
            mul(wu, bu, gu, count=C * D)

            # ordered lower-triangular solve, rows in increasing order
            for i in range(1, C):
                for j in range(i):
                    w = Var(0.0, dtype=DT.float)
                    w.GetValueFrom(lu[i:i + 1, j:j + 1])
                    muls(tmp[0:1, 0:D], uu[j:j + 1, 0:D], w, count=D)
                    sub(uu[i:i + 1, 0:D], uu[i:i + 1, 0:D], tmp[0:1, 0:D], count=D)
                    muls(tmp[0:1, 0:D], wu[j:j + 1, 0:D], w, count=D)
                    sub(wu[i:i + 1, 0:D], wu[i:i + 1, 0:D], tmp[0:1, 0:D], count=D)

            u[bb, cc, hv, 0:C, 0:D] <<= uu[0:C, 0:D]
            wy[bb, cc, hv, 0:C, 0:D] <<= wu[0:C, 0:D]

    return u, wy
