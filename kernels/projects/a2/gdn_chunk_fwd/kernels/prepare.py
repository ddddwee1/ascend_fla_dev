"""A2 GDN chunk-forward stage 1: prepare.

Tensor-vector port of the a5 `gdn_chunk_fwd/kernels/stages.py::gdn_chunk_prepare`
(@vf) to the A2 (c220) facade. Same arithmetic per (b, chunk, value-head):

  qn = q * scale            (q from the qk-head i_h = hv // (HV//H); GVA gap §1.1)
  kn = k                    (raw k, unmodified — carried for the scores stage)
  bk = k * beta             (beta is the per-token scalar update gate)
  wv = v * beta
  gc[i, :] = sum_{j<=i} g[j]   (running FP32 prefix of the per-token scalar g,
                                broadcast across the 128 channels)

Changes vs the a5 source: no `@vf` (a2 has none). Per-token scalars (`beta`, `g`)
are read into a scalar `Var` with `GetValueFrom` and applied one row at a time with
`muls`/`adds`, the proven a2 idiom from `kda_fwd_stable/kernels/intra.py`. `scale`
is a kernel scalar (gap §1.7, absorbed at q). Inputs are read in public token-major
layout via 2-D `[B*T, .]` views (host does no permute; D-PM-35/37). Outputs keep the
chunked head-major `[B, N, HV, 64, 128]` layout the downstream stages consume.
"""
from ascriptor.a2 import *

D = 128
C = 64
FULL_MASK = (1 << 64) - 1


@kernel(mode="vec", block_dim=40)
def gdn_chunk_prepare_a2_kernel(
    q: GM[f32, ("BT", "HD")],
    k: GM[f32, ("BT", "HD")],
    v: GM[f32, ("BT", "HVD")],
    g: GM[f32, ("BT", "HV")],
    beta: GM[f32, ("BT", "HV")],
    qn: GM[f32, ("B", "N", "HV", 64, 128)],
    kn: GM[f32, ("B", "N", "HV", 64, 128)],
    gc: GM[f32, ("B", "N", "HV", 64, 128)],
    bk: GM[f32, ("B", "N", "HV", 64, 128)],
    wv: GM[f32, ("B", "N", "HV", 64, 128)],
    scale: f32,
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    qu = Tensor(DT.float, [C, D], Position.UB)
    ku = Tensor(DT.float, [C, D], Position.UB)
    vu = Tensor(DT.float, [C, D], Position.UB)
    bku = Tensor(DT.float, [C, D], Position.UB)
    wvu = Tensor(DT.float, [C, D], Position.UB)
    gcu = Tensor(DT.float, [C, D], Position.UB)

    work = B * N * HV
    per = CeilDiv(work, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, work)

    with auto_sync():
        set_mask(FULL_MASK, FULL_MASK)
        hvh = Var(HV // H)
        for item in range(begin, end):
            hv = Var(item % HV)
            cc = Var((item // HV) % N)
            bb = Var(item // (N * HV))
            ih = Var(hv // hvh)
            r0 = Var(bb * T + cc * C)

            qu[0:C, 0:D] <<= q[r0:r0 + C, ih * D:ih * D + D]
            ku[0:C, 0:D] <<= k[r0:r0 + C, ih * D:ih * D + D]
            vu[0:C, 0:D] <<= v[r0:r0 + C, hv * D:hv * D + D]

            # kn = raw k
            kn[bb, cc, hv, 0:C, 0:D] <<= ku[0:C, 0:D]
            # qn = q * scale
            muls(qu[0:C, 0:D], qu[0:C, 0:D], scale, count=C * D)   # explicit count (device vec repeat)
            qn[bb, cc, hv, 0:C, 0:D] <<= qu[0:C, 0:D]

            # per-token scalar beta (bk = k*beta, wv = v*beta) and prefix of g
            gsum = Var(0.0, dtype=DT.float)
            for rr in range(C):
                bval = Var(0.0, dtype=DT.float)
                bval.GetValueFrom(beta[r0 + rr:r0 + rr + 1, hv:hv + 1])
                muls(bku[rr:rr + 1, 0:D], ku[rr:rr + 1, 0:D], bval, count=D)
                muls(wvu[rr:rr + 1, 0:D], vu[rr:rr + 1, 0:D], bval, count=D)
                gval = Var(0.0, dtype=DT.float)
                gval.GetValueFrom(g[r0 + rr:r0 + rr + 1, hv:hv + 1])
                gsum.set(gsum + gval)
                # gc[rr, :] = gsum  (broadcast the scalar prefix to all channels)
                dup(gcu[rr:rr + 1, 0:D], 0.0, count=D)
                adds(gcu[rr:rr + 1, 0:D], gcu[rr:rr + 1, 0:D], gsum, count=D)

            bk[bb, cc, hv, 0:C, 0:D] <<= bku[0:C, 0:D]
            wv[bb, cc, hv, 0:C, 0:D] <<= wvu[0:C, 0:D]
            gc[bb, cc, hv, 0:C, 0:D] <<= gcu[0:C, 0:D]

    return qn, kn, gc, bk, wv
