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
* and because no cast is needed, ``score``, ``states`` and ``delta`` are used
  **verbatim**: they go straight from GM into L1 with one ``gm_to_l1_nd2nz``
  each on the cube side, never touching UB. The BF16 sibling has to route them
  through UB because it casts them. Only the weighted query is vector-computed
  and therefore still staged UB -> L1. The first FP32 revision staged all four
  operands through UB and measured ``aiv_mte3_ratio = 0.473`` on the real
  device with ``aic_mac_ratio = 0.444``: the stage was bound by the
  ``ub_to_l1.nd2nz`` multi-burst move (one MTE3 burst per NZ fractal column,
  D-084), which the compiler's own performance lint names. Three of the four
  operands do not need it at all.

``set_hf32(False)`` is load-bearing. An FP32 cube ``matmul`` obeys the AI
core's HF32 rounding bit, and nothing in a kernel launch resets it, so a kernel
that does not set it inherits whatever ran on that core before — including
another process on the same card. One real-device run of ``t4096_h8`` came back
at relative L2 3.06e-04 against the CPU reference (budget 1e-04) with all 13
input stages bitwise identical to the 20+ passing runs of the same case; a
build with ``set_hf32(True)`` reproduces that run's ``o`` SHA-256 and its
2.1152198314666748e-05 maximum absolute error exactly. Pinning the bit is the
only thing standing between this stage and an intermittent silent precision
downgrade (AGENTS.md section 7). Cost: the pinned Ascriptor reference
interpreter has no ``core.set_hf32``, so the ``sim`` and ``pipesim`` launchers
cannot execute this kernel until it gains one — see
``docs/research/pkda_chunk_fwd_gate_range.md``.

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
    middle = Tensor(DT.float, [HALF, D], Position.UB)
    out = Tensor(DT.float, [HALF, D], Position.UB)
    ql = Tensor(DT.float, [C, D], Position.L1)
    al = Tensor(DT.float, [C, C], Position.L1)
    sl = Tensor(DT.float, [D, D], Position.L1)
    dl = Tensor(DT.float, [C, D], Position.L1)
    product = Tensor(DT.float, [C, D], Position.L0C)
    q_ready = VcMutex(0, guards=ql, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    result_ready = CvMutex(1, guards=middle, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    per = CeilDiv(B * N * H, GetCubeNum())
    begin = Var(per * GetCubeIdx())
    end = Min(begin + per, B * N * H)
    with cube_scope():
        set_hf32(False)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % H)
            cc = Var((item // H) % N)
            bb = Var(item // (N * H))
            rr = Var(GetSubBlockIdx() * HALF)
            q_ready.lock()
            qu[:, :] <<= qn[bb, cc, hh, rr:rr+HALF, :]
            gu[:, :] <<= gc[bb, cc, hh, rr:rr+HALF, :]
            weighted_query(qu, gu, qw)
            ql[rr:rr+HALF, :] <<= qw[:, :]
            q_ready.ready()

            gm_to_l1_nd2nz(sl[0:D, 0:D], states[bb, cc, hh, 0:D, 0:D], D, D, D, D)
            gm_to_l1_nd2nz(al[0:C, 0:C], score[bb, cc, hh, 0:C, 0:C], C, C, C, C)
            gm_to_l1_nd2nz(dl[0:C, 0:D], delta[bb, cc, hh, 0:C, 0:D], C, D, D, C)
            q_ready.wait()
            matmul(product, ql, sl.T, m=C, n=D, k=D, splitk=SPLIT_K)
            matmul(product, al, dl.T, m=C, n=D, k=C, is_init=False)
            result_ready.lock()
            middle[:, :] <<= product[:, :]
            result_ready.ready()
            q_ready.free()

            result_ready.wait()
            copy_rows(middle, out)
            result_ready.free()
            for ii in range(HALF):
                tt = Var(cc * C + rr + ii)
                if tt < T:
                    o[bb, tt:tt+1, hh, :] <<= out[ii:ii+1, :]
    return o
