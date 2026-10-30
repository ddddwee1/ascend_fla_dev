# DR-P1 · DeltaNet (gated_delta_rule) backward: independent real-hardware re-verification and pipe utilisation

Task `DR-P1`, A5 real hardware. This document covers the **backward** half of
`kernels/projects/a5/gdr_chunk_fwd_bwd_pypto_pro` — the six kernels `gdr_bwd_dv_local`,
`gdr_bwd_reverse_scan`, `gdr_bwd_dqkw`, `gdr_bwd_wy_matmuls`, `gdr_bwd_wy_reduce`,
`gdr_bwd_finalize`.

**Everything shared lives in `dr_perf_fwd.md` and is not repeated here**: the environment and its
identity (§1 there), the measurement discipline (§2), the CPU semantic validation and the composed
gates (§3.1–§3.3), the `fla` comparison (§6), the HF32 verdict (§7), the PyPTO-Pro and
orchestrator-plugin situation (§8), the new real-shape correctness evidence (§9), and the list of
decisions left to the PM (§10). Read that document first; this one is the backward-specific
analysis.

Short version. The backward is **66% of the pipeline's device time** and reproduces as faithfully as
the forward. On AGENTS.md §6's 85% line, **five of the six backward kernels are short at every shape
measured**; the sixth (`gdr_bwd_finalize`) is the only kernel in the whole unit that meets the line,
and it meets it on **movement** (`aiv_mte2` 0.896–0.928), not compute. The binding pipe and the gap
are given per kernel in §2, and §3 explains why two of them cannot be moved from the host side.

## 1. Reproduction of the published backward numbers

Production shape `B=1, T=2048, H=8, K=V=128`, one `msprof` PipeUtilization capture, warm medians,
three warm-up occurrences discarded per op.

| # | kernel | `PERFORMANCE.md` µs | measured µs | Δ | doc `aic_mac` | measured | doc cube % | measured | blocks |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 3 | `gdr_bwd_dv_local` | 37.35 | 37.57 | +0.6% | 0.705 | 0.697 | 87.9 | 88.62 | 28 |
| 4 | `gdr_bwd_reverse_scan` | 216.99 | 216.58 | −0.2% | 0.570 | 0.570 | 56.7 | 56.75 | 16 |
| 5 | `gdr_bwd_dqkw` | 161.02 | 159.81 | −0.8% | 0.716 | 0.720 | 89.0 | 89.16 | 28 |
| 6 | `gdr_bwd_wy_matmuls` | 111.99 | 112.89 | +0.8% | 0.812 | 0.810 | 88.0 | 87.91 | 28 |
| 7 | `gdr_bwd_wy_reduce` | 41.64 | 40.83 | −1.9% | — | pure vector | — | 0 | 56 |
| 8 | `gdr_bwd_finalize` | 27.99 | 26.87 | −4.0% | — | pure vector | — | 0 | 56 |
| | **backward subtotal** | **596.98** | **594.55** | **−0.41%** | | | | | |

Every kernel is within 1 µs of its published figure except `finalize` (1.1 µs) and `wy_reduce`
(0.8 µs), both of which are the smallest two and therefore the most sensitive in relative terms.
`gdr_bwd_reverse_scan`'s block count of 16 — the one the document singles out and explains — matches
exactly.

Derived claims that follow and also hold. `PERFORMANCE.md`'s "forward is 34.2% of training; backward
is 1.92 times forward" is stated in its forward-only section and is reconstructible only from the
**large** shape (4873.3 / 14240 = 34.22%), so that is the shape to check it on. From the large-shape
capture, forward in-pipeline is 2052.67 + 2826.32 = 4878.99 µs of a 14232.13 µs total:
**34.28%** of training, and backward/forward = (14232.13 − 4878.99)/4878.99 = **1.917x**. Both
within 0.2% of the published values. At the production shape the same two ratios are 314.29/908.85 =
34.58% and 1.892x.

Correctness of the five backward public outputs reproduces to four significant figures at the
production shape; the table is in `dr_perf_fwd.md` §3.2 (`dq` 1.6976e-03, `dk` 1.6440e-03,
`dv` 1.5119e-03, `dg` 1.2271e-06, `dbeta` 1.1678e-06 relative L2, all `matched_ratio` 1.000000,
0 FAIL across 180 comparisons and all five shipped shapes). The backward golden is
`torch.autograd.grad` of the forward oracle, so `dr_perf_fwd.md` §3.1's exact bit-identity of the
forward chunk path against `fla`'s `naive_chunk` is what underwrites it.

**New backward correctness evidence.** `VALIDATION.md` records that the largest validated shape is
T=2048, H=8. DR-P1 ran the unit's own composition gate — which checks all five gradients plus
`dh`, `dv2`, `dq_p`, `dk_p`, `dw`, `dg_p`, `dk`, `dv`, `dbeta_wy`, `dg2` against the staging
goldens — at the real Qwen3-Next layer shape: `B=1, T=1024, H=32` (H = `HMAX`) and `B=1, T=1024,
H=16`, **both PASS, all boundaries clean**. Details in `dr_perf_fwd.md` §9.

## 2. The 85% line, per backward kernel

Real shape `B=1, T=4096, H=32, K=V=128` — the GQA-expanded Qwen3-Next layer from
`docs/matrix/models.json`; see `dr_perf_fwd.md` §4 for why the real shape can only be expressed with
H=32 in this unit. Bold marks the binding pipe. Backward subtotal 4277.09 µs of the 6467.60 µs
pipeline (**66.1%**).

| # | kernel | µs | share of pipeline | blocks | `aic_mac` | `aic_mte1` | `aic_mte2` | `aic_fix` | `aiv_vec` | `aiv_mte2` | `aiv_mte3` | cube % | binding pipe | gap to 85% |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| 3 | `gdr_bwd_dv_local` | 271.63 | 4.2% | 28 | **0.696** | 0.081 | 0.559 | 0.056 | 0.225 | 0.305 | 0.128 | 98.46 | `aic_mac` | **15.4 pts** |
| 4 | `gdr_bwd_reverse_scan` | 1277.13 | 19.7% | 28 | **0.577** | 0.080 | 0.293 | 0.044 | 0.046 | 0.067 | 0.076 | 75.99 | `aic_mac` | **27.3 pts** |
| 5 | `gdr_bwd_dqkw` | 1207.15 | 18.7% | 28 | **0.692** | 0.102 | 0.346 | 0.070 | 0.079 | 0.112 | 0.063 | 98.34 | `aic_mac` | **15.8 pts** |
| 6 | `gdr_bwd_wy_matmuls` | 789.63 | 12.2% | 28 | **0.829** | 0.135 | 0.481 | 0.324 | 0.030 | 0.087 | 0.073 | 97.96 | `aic_mac` | **2.1 pts** |
| 7 | `gdr_bwd_wy_reduce` | 507.41 | 7.8% | 56 | 0 | 0 | 0 | 0 | 0.175 | **0.816** | 0.332 | 0 | `aiv_mte2` | **3.4 pts** |
| 8 | `gdr_bwd_finalize` | 224.14 | 3.5% | 56 | 0 | 0 | 0 | 0 | 0.277 | **0.896** | 0.397 | 0 | `aiv_mte2` | **MET** |

The same six kernels at the other two shapes, binding pipe only:

| kernel | production T=2048 H=8 | real T=4096 H=32 | large T=16384 H=16 |
|---|--:|--:|--:|
| `gdr_bwd_dv_local` | `aic_mac` 0.697 | `aic_mac` 0.696 | `aic_mac` 0.682 |
| `gdr_bwd_reverse_scan` | `aic_mac` 0.570 | `aic_mac` 0.577 | `aic_mac` 0.581 |
| `gdr_bwd_dqkw` | `aic_mac` 0.720 | `aic_mac` 0.692 | `aic_mac` 0.695 |
| `gdr_bwd_wy_matmuls` | `aic_mac` 0.810 | `aic_mac` 0.829 | `aic_mac` 0.831 |
| `gdr_bwd_wy_reduce` | `aiv_mte2` 0.611 | `aiv_mte2` 0.816 | `aiv_mte2` 0.813 |
| `gdr_bwd_finalize` | `aiv_mte2` 0.675 | `aiv_mte2` 0.896 **MET** | `aiv_mte2` 0.928 **MET** |

Reading this honestly, kernel by kernel, as AGENTS.md §6 requires ("say which pipe, and what its
utilisation is, before deciding whether to keep optimising or to record the gap"):

- **`gdr_bwd_finalize` — MET, on movement, at the two larger shapes.** `aiv_mte2` 0.896 at the real
  shape and 0.928 at the large shape. It is a pure-vector epilogue that reads six chunk-major
  tensors and republishes three large gradients in the public token-major layout, so a load-pipe
  bound is the expected and correct place for it to land. At the production shape it is only 0.675,
  because at T=2048/H=8 there is not enough work to keep 56 vector cores' load pipes saturated. So
  this kernel is **at its bound on the shapes that matter** and nothing should be done to it.
- **`gdr_bwd_wy_reduce` — 3.4 points short, also on movement**, 0.816 at both larger shapes. It has
  the tightest UB budget in the set (247,808 of 253,952 B) and is the other pure-vector kernel.
  3.4 points is close enough that the remaining headroom is unlikely to be worth a restructure of
  the most UB-constrained kernel in the unit; recorded as near-bound.
- **`gdr_bwd_wy_matmuls` — 2.1 points short on `aic_mac`**, 0.829 at the real shape and 0.831 at the
  large one, the highest MAC occupancy of any cube kernel here. It already runs L0A and L0B at half
  capacity specifically to leave room for the NZ staging a transposed operand needs, and
  `aic_fix` 0.324 says the fixpipe is doing real work alongside. **This is the closest any cube
  kernel in the unit gets to the line, and it is the one where a small scheduling win could cross
  it.** It is also 12% of the pipeline, so crossing the line here buys at most a few percent overall.
- **`gdr_bwd_dqkw` — 15.8 points short.** Eleven matmuls per chunk against the two state trails;
  32 of 32 mutex ids and both L0A and L0B at capacity, i.e. it is already out of on-chip resources
  to restructure with. `aic_mte2` 0.346 is the second pipe. At 18.7% of the pipeline it is the
  second-largest consumer, so it is the most valuable target in absolute terms and the hardest in
  practical terms.
- **`gdr_bwd_dv_local` — 15.4 points short**, but only 4.2% of the pipeline. `aic_mte2` 0.559 is
  high for its size, consistent with its documented NZ staging through UB eight columns at a time
  because the vector core cannot write an L1 operand directly. Low value.
- **`gdr_bwd_reverse_scan` — 27.3 points short, the weakest of the six, and 19.7% of the
  pipeline — the single largest consumer in the unit.** §3 explains why its number is what it is and
  why no host-side knob moves it.

## 3. Why `gdr_bwd_reverse_scan` is where it is, in closed form

This is the kernel worth understanding, because it is the biggest single consumer and the furthest
from the line, and because the obvious reading of its numbers is wrong.

Its `cube_utilization` is **75.99%** at the real shape and **56.75%** at the production shape, which
invites "it is not filling the cores, give it more blocks". That is a symptom rule, and AGENTS.md
§6.5 says to find the exact cause first. The exact cause is **wave quantisation**, and it predicts
the measurement to within a quarter of a point:

Its independent work is `B * H * ceil(V/64)` items, and the kernel strides over the item space, so
the block count is a free host-side choice that affects balance only. Wall time is set by the **most
loaded** core, so with `n` items on `c` blocks the occupancy ceiling is
`(n/c) / ceil(n/c)`:

| shape | items `B·H·⌈V/64⌉` | blocks | mean per core | max per core | predicted cube % | measured cube % |
|---|--:|--:|--:|--:|--:|--:|
| real, H=32 | 64 | 28 | 2.286 | 3 | **76.2%** | **75.99%** |
| production, H=8 | 16 | 16 | 1.000 | 1 | 100% (but only 16 of 28 cores used → 57.1% of the part) | **56.75%** |
| large, H=16 | 32 | 28 | 1.143 | 2 | **57.1%** | **57.14%** |

All three match. The production and large shapes are the case the document already explains — the
reachable item counts are 8, 16, 32, … and none lands in (16, 28], so at H=8 it claims 16 blocks and
at H=16 it gets 32 items over 28 blocks, four of which do two items while 24 do one. The real shape
(H=32) is the new row: 64 items over 28 cores, 8 cores doing three.

**So the low occupancy is fully accounted for and is not removable from the host side.** Two things
that look like fixes are not:

- **Fewer blocks raises the ratio without changing the time.** At 64 items, 22 blocks also needs 3
  waves, so the wall clock is identical while `cube_utilization` rises to ~97%. That is moving a
  metric, not making anything faster.
- **More, cheaper items is what the forward scan's A/B already measured, and it is slower.**
  `dr_perf_fwd.md` §5 sweeps the sibling kernel's exposed `_vt`/`_streams` surface on real hardware:
  every refinement of the item space is monotonically worse (up to 1.69x slower), because matmul and
  fixpipe always run on the full physical tile, so a narrower tile creates more items of the *same*
  cost. The two scans share that design constraint and that item-space structure.

That same A/B produced the result that matters most for judging this kernel against the 85% bar:
**on the forward scan, two configurations reach `aic_mac` 0.911 and 0.917 — over the line — while
running 13% and 69% slower than the default, with bit-identical output.** `aic_mac_ratio` is busy
MAC time over the kernel's window, and issuing more redundant full-width matmuls raises it while
making the kernel slower. The backward scan is structurally the same kernel in the other direction.
**Reading `gdr_bwd_reverse_scan`'s 0.577 as "27 points of available speed-up" would therefore be
wrong**; part of that gap is the serial chunk recurrence and the wave quantisation above, and the
part that a tiling change could "fix" shows up as a slower kernel with a better number.

Removing the quantisation needs items that are genuinely cheaper, i.e. a matmul over a partially
valid tile — which this unit documents as a hardware behaviour it is deliberately shaped around
(a partially valid N axis leaves stale rows in the destination of a split-M drain; the module
docstrings record three such behaviours, each found through a wrong numeric result rather than a
compile error). **Recorded as at the structural limit of the current design, with the loss
quantified at 23.8% of the part at the real shape**, rather than as an open tuning opportunity.

One more structural note from the source, relevant to anyone who does try: `dS` lives in UB as a
single persistent buffer reaching the cube through a GM bridge, and **all vector work runs on
sub-block 0, guarded, because the state is single-buffered**. That is why `aiv_vec` is only 0.046 —
half the vector capability is idle by construction. Its L0A is exactly at its 64 KB capacity, so
there is no room to double-buffer the state without giving something else up.

## 4. What binds the backward as a whole

The backward's 594.55 µs (production) / 4277.09 µs (real shape) is spread, not concentrated: the
three largest kernels are 19.7%, 18.7% and 12.2% of the pipeline, so halving any one of them buys
roughly a tenth. The document's own observation — "the load is spread: halving any single one buys at
most an eighth of the pipeline" — reproduces.

Across the six, the binding resource is `aic_mac` for the four cube kernels and `aiv_mte2` for the two
vector ones. No kernel is `mte3`-bound; `aic_mte3_ratio` is **0.000 for every kernel in the unit at
every shape**, which is consistent with a design that writes its results through the fixpipe rather
than through the cube's store pipe.

And the same ceiling that caps the forward caps the backward: **all arithmetic is FP32**, and this
part's own back-to-back MMAD probe measures 623 ns per `[64,64]×[64,64]` MMAD in FP32 against 42 ns
in BF16 — **14.8x**. Four of the six backward kernels are MAC-bound at 0.58–0.83, so they are
genuinely spending their time in the cube; that cube is just running the slow dtype. Moving the
backward to BF16 arithmetic is the only ~10x-class lever here, it would break the bit-identity that
`dr_perf_fwd.md` §3.1 confirms against `fla`, and it is a semantics decision for the PM/user. Not
attempted.

## 5. Backward-specific items for the PM

The full decision list is `dr_perf_fwd.md` §10. Backward-specific:

1. **`gdr_bwd_wy_matmuls` at `aic_mac` 0.829 is 2.1 points from the line** and is the only cube
   kernel in the unit within reach. If the PM wants one kernel taken to 85%, it is that one — but it
   is 12% of the pipeline, so the overall win is small, and §3's warning applies: the change has to
   be *faster*, not merely higher-ratio.
2. **`gdr_bwd_reverse_scan` should be recorded as structurally bound, not as 27 points of
   headroom** (§3). Its quantisation loss is quantified and its binding constraints
   (single-buffered `dS`, sub-block 0 only, L0A at capacity) are in the source.
3. **No backward kernel was modified.** The gating-discipline gaps DR-P1 booked
   (`docs/matrix/gaps.json`) include one that lives in the backward chain —
   `gdr-b1-enforced-mid-chain`, whose root is `gdr_bwd_dqkw`'s `_B1_DYN4`/`_B1_DYN3` static shape
   annotations at `kernels/gdr_bwd_dqkw_impl.py:335-336`. Note for whoever fixes it: **that
   annotation is the real constraint and should stay**; the gap is that the chain's *first* host
   entry does not reject `B > 1`, so four kernels run before it fails.
