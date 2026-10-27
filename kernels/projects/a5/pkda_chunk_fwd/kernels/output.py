"""FP32 cube output products with FP32 L0C accumulation.

The stage computes, per (batch, chunk, head)::

    o = (qn * exp(gc)) @ states + score @ delta

Both terms are plain GEMMs — ``[64,128] @ [128,128]`` and ``[64,64] @
[64,128]`` — so the whole stage is two ``matmul`` calls accumulating into one
FP32 L0C tile. The previous implementation (``stages.py::output_vf``, kept in
git history) evaluated exactly the same two products as nested vector loops:
128 broadcast-multiply-add rounds for the state term plus ``i+1`` rounds for
the score term, per row. PK-07 round 1 measured ``aic_mac_ratio = 0.000`` for
this kernel on real A5 hardware — the cube engine was idle while the vector
engine issued ~70k instructions per (batch, chunk, head) item.

Structure, buffer sizing and the vector→cube→vector hand-off are taken from
this repository's BF16 sibling ``pkda_chunk_fwd_bf16/kernels/output.py``, which
is already native-qualified on Ascend950PR and solves the same problem for the
same stage. Two differences follow from FP32 operands:

* every L1 tile is twice as wide in bytes, so the ``q @ states`` product uses
  ``splitk=64``. Its L0B tile would otherwise be ``[128,128]`` FP32 = 64 KB,
  the whole of L0B, leaving the compiler no room to rotate slots. With the
  split both operand tiles stay at 16 KB / 32 KB, which is the same sizing
  ``kda_fwd_stable/kernels/intra.py`` uses for its FP32 cube products.
* no cast is needed on either side, so ``weighted_query`` writes FP32 directly
  and the L0C drain is copied out by a plain vector move.

Padded rows need no masking. ``prepare`` writes exact zeros into ``qn`` beyond
the live token count and ``scores``/``scan`` zero-fill every ``score``/``delta``
row and column they do not compute, so rows ``>= count`` of both products are
exactly zero and the upper triangle of ``score`` contributes nothing. ``gc``
carries the last live prefix on padded rows, which is finite and non-positive,
so ``exp(gc)`` cannot overflow there. Only live rows reach ``o``.
"""
from ascriptor.a5 import *

D = 128
C = 64
HALF = C // 2
SPLIT_K = 64


@vf()
def weighted_query(q: Tensor, gate: Tensor, out: Tensor):
    """``out = q * exp(gate)`` — the same FP32 product order as before."""
    r = RegList(DT.float, 2)
    g = RegList(DT.float, 2)
    for rr in range(HALF):
        r <<= q[rr:rr+1, 0:D]
        g <<= gate[rr:rr+1, 0:D]
        g <<= g.exp()
        r <<= r * g
        out[rr:rr+1, 0:D] <<= r


@vf()
def copy_rows(src: Tensor, out: Tensor):
    """Release the drained L0C tile before the GM stores, as the BF16 unit does."""
    r = RegList(DT.float, 2)
    for rr in range(HALF):
        r <<= src[rr:rr+1, 0:D]
        out[rr:rr+1, 0:D] <<= r


@kernel(mode='mix')
def pkda_chunk_output(
    qn: GM[f32, ("B", "N", "H", 64, 128)], gc: GM[f32, ("B", "N", "H", 64, 128)],
    score: GM[f32, ("B", "N", "H", 64, 64)], states: GM[f32, ("B", "N", "H", 128, 128)],
    delta: GM[f32, ("B", "N", "H", 64, 128)], o: GM[f32, ("B", "T", "H", 128)],
    B: i32, T: i32, H: i32, N: i32,
):
    qu = Tensor(DT.float, [HALF, D], Position.UB)
    gu = Tensor(DT.float, [HALF, D], Position.UB)
    qw = Tensor(DT.float, [HALF, D], Position.UB)
    au = Tensor(DT.float, [HALF, C], Position.UB)
    su = Tensor(DT.float, [C, D], Position.UB)
    du = Tensor(DT.float, [HALF, D], Position.UB)
    middle = Tensor(DT.float, [HALF, D], Position.UB)
    out = Tensor(DT.float, [HALF, D], Position.UB)
    ql = Tensor(DT.float, [C, D], Position.L1)
    al = Tensor(DT.float, [C, C], Position.L1)
    sl = Tensor(DT.float, [D, D], Position.L1)
    dl = Tensor(DT.float, [C, D], Position.L1)
    product = Tensor(DT.float, [C, D], Position.L0C)
    q_ready = VcMutex(0, guards=ql, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    a_ready = VcMutex(1, guards=al, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    s_ready = VcMutex(2, guards=sl, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    d_ready = VcMutex(3, guards=dl, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    result_ready = CvMutex(4, guards=middle, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    per = CeilDiv(B * N * H, GetCubeNum())
    begin = Var(per * GetCubeIdx())
    end = Min(begin + per, B * N * H)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % H)
            cc = Var((item // H) % N)
            bb = Var(item // (N * H))
            rr = Var(GetSubBlockIdx() * HALF)
            sr = Var(GetSubBlockIdx() * C)
            q_ready.lock()
            qu[:, :] <<= qn[bb, cc, hh, rr:rr+HALF, :]
            gu[:, :] <<= gc[bb, cc, hh, rr:rr+HALF, :]
            weighted_query(qu, gu, qw)
            ql[rr:rr+HALF, :] <<= qw[:, :]
            q_ready.ready()
            a_ready.lock()
            au[:, :] <<= score[bb, cc, hh, rr:rr+HALF, :]
            al[rr:rr+HALF, :] <<= au[:, :]
            a_ready.ready()
            s_ready.lock()
            su[:, :] <<= states[bb, cc, hh, sr:sr+C, :]
            sl[sr:sr+C, :] <<= su[:, :]
            s_ready.ready()
            d_ready.lock()
            du[:, :] <<= delta[bb, cc, hh, rr:rr+HALF, :]
            dl[rr:rr+HALF, :] <<= du[:, :]
            d_ready.ready()

            q_ready.wait()
            s_ready.wait()
            matmul(product, ql, sl.T, m=C, n=D, k=D, splitk=SPLIT_K)
            a_ready.wait()
            d_ready.wait()
            matmul(product, al, dl.T, m=C, n=D, k=C, is_init=False)
            result_ready.lock()
            middle[:, :] <<= product[:, :]
            result_ready.ready()
            q_ready.free()
            a_ready.free()
            s_ready.free()
            d_ready.free()

            result_ready.wait()
            copy_rows(middle, out)
            result_ready.free()
            for ii in range(HALF):
                tt = Var(cc * C + rr + ii)
                if tt < T:
                    o[bb, tt:tt+1, hh, :] <<= out[ii:ii+1, :]
    return o
