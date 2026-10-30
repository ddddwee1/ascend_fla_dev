# DR-P1 · DeltaNet (gated_delta_rule) forward: independent real-hardware re-verification and pipe utilisation

Task `DR-P1`, A5 real hardware. This document covers the **forward** half of
`kernels/projects/a5/gdr_chunk_fwd_bwd_pypto_pro` — the two training-forward kernels
(`gdr_fwd_train_wy`, `gdr_fwd_train_scan`) and the standalone inference forward
(`gated_delta_rule_fwd`). The backward half is `dr_perf_bwd.md`; §1 and §2 here are shared
and that document does not repeat them.

The unit was merged by `DR-01` (PR #145, external contribution, `f16c1225`) with two things
explicitly left open in the board record: **its hardware numbers had never been reproduced by
anyone but the contributor**, and three gating-discipline statements in its `README.md` did not
match the code. DR-P1's first job was the numbers; the gating gaps are booked in
`docs/matrix/gaps.json` and deliberately **not** fixed here (AGENTS.md §6.5).

**Headline.** Every device-time and correctness figure the unit publishes reproduces, most of
them to three or four significant figures — see §3. Every *host wall-clock* figure that compares
against `fla` does **not**: the published 13.2x speed-up at the production shape measures 7.8x
here, because `fla`'s own reference ran 36% faster on this machine than on the contributor's
(§6). And on AGENTS.md §6's 85% pipe-utilisation line, **8 of the 9 kernels are short at every
shape measured**; the one that meets it is movement-bound, not compute-bound (§4). §5 reports a
result that matters beyond this operator: on `gdr_fwd_train_scan` the 85% bar can be *met* today,
with no source change, by a configuration that is **13% slower** — `aic_mac_ratio` is not
monotone with speed.

## 1. Environment, and what identifies it

Measured on `machine_specs.md`'s 238 host, inside the pre-existing task container, physical card 3
mapped to logical 0. All writes confined to this task's own root under `/data/home/l00276318`;
nothing in the shared CANN, driver or image was modified.

| | |
|---|---|
| part | Ascend950PR, `Ascend950PR_957b`, `DAV_3510`, 28 cube / 56 vector cores |
| CANN | `9.1.0-beta.1` (toolkit and opp both report `Version=9.1.0-beta.1`) |
| opp kernel packages | `ascend950`, `ascend910b`, `ascend910_93` — `ascend950` present (AGENTS.md §5 boot check) |
| Python / torch | 3.12.13 / `2.12.0+cpu`, `torch_npu 2.12.0` |
| PyPTO-Pro | `pypto` dist `0.2.1`, `pypto_pro` from `/usr/local/Ascend/cann/python/site-packages` (`__init__.py` md5 `d83c776f6949925c38e3e75f8400685d`); `pl.jit`, `pl.vector_function` and `pl.Vf` all present — this is genuine PyPTO-Pro, not the `pypto` 0.2.0 that blocked the DR-01 reviewer |
| `fla` | `0.5.2`, the version the unit's `README.md` records as verified |
| unit revision | the merged tree, `kernels/projects/a5/gdr_chunk_fwd_bwd_pypto_pro` at `f16c1225`, transferred by tarball and md5-verified after transfer |

This is the **same part and the same CANN** the unit's `docs/PERFORMANCE.md` names ("Ascend 950PR,
28 cube and 56 vector cores, DAV_3510, CANN 9.1.0-beta.1"), so the comparisons in §3 are like for
like and the §6 divergence is not a SoC or toolchain difference.

**Card choice, recorded because AGENTS.md §6 asks for the reason.** The user's priority is physical
4 then 3. Card 4 was healthy but carried a foreign tenant cycling short-lived `python3` jobs
(~311 MB, PIDs 447106 → 453440 → 472026 inside two minutes), which would contend with any timing
run, so every measurement here is on physical 3 (health OK, no process, before and after each job).
No foreign process was stopped, inspected further or reset; no card was reset. Both of this
machine's shared per-card lock conventions were held for the duration of every job.

## 2. Measurement discipline

- **One build, one process, one capture** (AGENTS.md §6 iron law 2). Every number below comes from
  its own process and its own `msprof --output` directory. No invocation measures two shapes or two
  configurations.
- **`msprof --aic-metrics=PipeUtilization`**, device time read from `op_summary`'s
  `Task Duration(us)`. Three warm-up iterations then five timed ones; the first three occurrences of
  each `Op Name` are discarded, because the first launch of a PyPTO-Pro kernel includes its JIT
  compilation. Reported figure is the **median of the warm occurrences**. Nothing here is a
  simulator estimate.
- **Device time and host wall time are reported separately and never averaged.** They answer
  different questions and §6 is where the difference bites.
- **Host wall-clock arms are run without `msprof`.** The unit's pipelined per-step figures are
  wall-clock numbers from unprofiled runs; comparing them against a figure taken under the profiler
  would be comparing two different things. (My first pipelined capture *was* under `msprof` and read
  848.3 µs/step; the unprofiled runs read 835.7–882.7. Recorded so the mistake is not repeated.)
- **Correctness is judged in FP32** against the unit's own two references, by the unit's own judge
  (`tests/precision_compare.py`), at the unit's own thresholds. I did not relax or re-derive any
  tolerance.

## 3. Independent reproduction of the published numbers

### 3.1 Semantics, on CPU — the strong claims are exact

`tests/validate_against_fla.py`, `fla` 0.5.2, FP32, CPU, four shapes. `docs/VALIDATION.md` states
two **strong** claims (bit-identity) and two weak ones (intervals).

| comparison | published | measured here | verdict |
|---|---|---|---|
| our chunk path vs `naive_chunk_gated_delta_rule` | `0.000e+00` / `0.000e+00` | `0.000e+00` / `0.000e+00`, all 4 shapes | **exact, reproduced** |
| our recurrent final state vs `naive_recurrent_gated_delta_rule` | `0.000e+00` / `0.000e+00` | `0.000e+00` / `0.000e+00`, all 4 shapes | **exact, reproduced** |
| our recurrent `o` vs `fla` | max abs 2.1e-4 – 4.2e-4; rel L2 2.0e-7 – 4.6e-7 | max abs **3.815e-05** – 4.158e-04; rel L2 2.042e-07 – 4.579e-07 | rel L2 interval reproduced; **max-abs lower bound does not** |
| our chunk path vs `fla` recurrent | max abs 1.7e-4 – 3.1e-3; rel L2 2.1e-6 – 3.7e-6 | max abs 1.717e-04 – 3.082e-03; rel L2 2.136e-06 – 3.652e-06 | reproduced |

The one discrepancy: at `B1T192H2K128V128` the recurrent-`o` max abs is `3.815e-05`, below the
published interval's `2.1e-4` floor. The other three shapes land inside it (2.136e-4, 2.136e-4,
4.158e-4). The script seeds with `torch.manual_seed(0)` and runs on CPU, so this is deterministic:
the published interval simply omits its own smallest member. The DR-01 reviewer saw the same thing
on A2 and attributed it to float32 accumulation order; the deterministic-CPU reproduction here says
it is a **tabulation error in the document**, not environment noise. Cosmetic, but it is the kind
of interval a later reader would use as a regression bound.

### 3.2 The composed training gate, on device — numbers match to four significant figures

`tests/test_composition.py`, all five shipped shapes, unmodified: **all 5 PASS, 180 comparisons,
0 FAIL** (36 per case: 6 public outputs + 30 intermediate boundaries). At the production shape
`B=1, T=2048, H=8, K=V=128`:

| output | `VALIDATION.md` rel L2 / max abs / peak | measured here | |
|---|---|---|---|
| `o` | 4.568e-03 / 1.979e+03 / 4.628e+05 | 4.5675e-03 / 1.9786e+03 / 4.6278e+05 | match |
| `dq` | 1.698e-03 / 9.883e+02 / 3.483e+05 | 1.6976e-03 / 9.8834e+02 / 3.4834e+05 | match |
| `dk` | 1.644e-03 / 2.285e+03 / 1.079e+06 | 1.6440e-03 / 2.2846e+03 / 1.0791e+06 | match |
| `dv` | 1.512e-03 / 9.460e+02 / 3.943e+05 | 1.5119e-03 / 9.4603e+02 / 3.9432e+05 | match |
| `dg` | 1.227e-06 / 2.250e+00 / 1.489e+06 | 1.2271e-06 / 2.2500e+00 / 1.4886e+06 | match |
| `dbeta` | 1.168e-06 / 9.141e-01 / 7.736e+05 | 1.1678e-06 / 9.1406e-01 / 7.7356e+05 | match |

Every row agrees to the published precision. The document's two side claims also hold exactly:
`o` and `dg` are the two that pass on the peak-scaled criterion rather than plain `allclose`
(`nominal_allclose=False` for exactly those two, `True` for the other four), and the intermediate
boundaries span `0.0000e+00` to `4.5675e-03` relative L2.

Two counting nits. `VALIDATION.md` says "185 comparisons clean"; the run emits **180**
(36 × 5), and I cannot construct 185 from what the harness prints. It says "five of them exactly
zero (`q_s`, `k_f32`, `g_f32`, `q_t` and their derived rows)"; I count **four** exactly-zero
boundaries, named exactly those four. Both are bookkeeping, not evidence.

### 3.3 The composed entry point — 38/38 bit-identical

`tests/test_train_entry.py` unmodified: **38 `EQUAL` results, 0 failures**, matching
`VALIDATION.md`'s "Thirty-eight equality checks in total" exactly. This covers the sequential and
concurrent orders, four `(scan, reverse)` core-budget pairs including ones that over-commit the
part, four steps queued without a per-step synchronisation, and five distinct-input steps at one,
two and three streams in flight. So the claim that **stream assignment and core budgets change no
arithmetic** is independently confirmed bitwise — this is the local analogue of AGENTS.md §6's
"`bd=1` and `bd=4` must be bitwise identical" hard criterion, and it passes.

### 3.4 Device time — the published per-kernel table reproduces at 0.34%

Production shape `B=1, T=2048, H=8, K=V=128`, one capture, warm medians.

| # | kernel | `PERFORMANCE.md` µs | measured µs | Δ | doc `aic_mac` | measured | doc cube % | measured |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | `gdr_fwd_train_wy` | 132.29 | 131.80 | −0.4% | 0.502 | 0.508 | 89.1 | 89.00 |
| 2 | `gdr_fwd_train_scan` | 182.71 | 182.49 | −0.1% | 0.796 | 0.793 | 84.1 | 84.41 |
| 3 | `gdr_bwd_dv_local` | 37.35 | 37.57 | +0.6% | 0.705 | 0.697 | 87.9 | 88.62 |
| 4 | `gdr_bwd_reverse_scan` | 216.99 | 216.58 | −0.2% | 0.570 | 0.570 | 56.7 | 56.75 |
| 5 | `gdr_bwd_dqkw` | 161.02 | 159.81 | −0.8% | 0.716 | 0.720 | 89.0 | 89.16 |
| 6 | `gdr_bwd_wy_matmuls` | 111.99 | 112.89 | +0.8% | 0.812 | 0.810 | 88.0 | 87.91 |
| 7 | `gdr_bwd_wy_reduce` | 41.64 | 40.83 | −1.9% | — | pure vector | — | 0 |
| 8 | `gdr_bwd_finalize` | 27.99 | 26.87 | −4.0% | — | pure vector | — | 0 |
| | **total** | **911.98** | **908.85** | **−0.34%** | | | | |

Block counts match the document exactly, including the two it singles out: `gdr_fwd_train_scan`
launches 24 and `gdr_bwd_reverse_scan` 16 of the 28 cube blocks at this shape. The document also
reports a re-profile of the same chain at 909.14 µs; my 908.85 µs is **0.03%** from that. The
device side of this unit is reproducible by a third party on a different card of a different host.

Large shape `B=1, T=16384, H=16, K=V=128`: **14232.13 µs**, against the published **14.24 ms** —
exact. Per chunk and head that is 14232.13/4096 = **3.475 µs** against the published 3.477, and the
production shape gives 908.85/256 = 3.550 against 3.563; so the document's "scaling is slightly
sub-linear, 3.563 → 3.477" reproduces.

Forward alone, `B=1, T=16384, H=16`: `gdr_fwd_train_wy` 2050.12 + `gdr_fwd_train_scan` 2824.89 =
**4875.01 µs**, against the published 2047.4 + 2825.9 = **4873.3 µs** — 0.035% apart. Inside the
full chain the same two kernels cost 2052.67 + 2826.32 = 4878.99 µs, so forward alone and forward
in-pipeline differ by **0.08%**, confirming the document's "0.09% apart, so the two halves do not
interfere through L2".

Forward alone at the production shape: 129.79 + 182.60 = **312.39 µs**, against the published
316.6 µs — 1.3% apart.

### 3.5 One published occupancy claim does not reproduce

`PERFORMANCE.md` says of the large shape: "Six of the eight kernels clear 93% occupancy at that
shape, against two of eight at the reference shape." Measured `cube_utilization(%)` at
`T=16384, H=16`: 98.26, 56.97, 99.03, 57.14, 99.17, 98.87, 0, 0 — **four** of eight clear 93%, not
six. The two scans sit at ~57% and the two pure-vector kernels have no cube occupancy at all. At
the production shape the same metric gives 89.00, 84.41, 88.62, 56.75, 89.16, 87.91, 0, 0 — **none**
clears 93%, against the document's "two of eight". I cannot reconstruct either count from
`cube_utilization`; if the document meant a vector-side occupancy for the last two rows it does not
say so. Recorded as not reproduced rather than guessed at.

### 3.6 The second oracle

`tests/fla_npu_baseline.py` — `fla`'s `naive_chunk` under `torch_npu`, i.e. an independent
implementation of the same mathematics on the same part. Its correctness line reproduces **exactly**:
at `(1,192,2,128,128)`, `max_abs=5.615e-03 rel_l2=1.903e-06 peak=2.783e+03`, character for
character what `VALIDATION.md` publishes. So the double-oracle mechanism is real and its number is
reproducible, which was the one thing the DR-01 reviewer could check on A2 and it still holds here.

**A documentation defect found while running it.** `README.md` documents the invocation as
`python tests/fla_npu_baseline.py`. Run that way the script prints the correctness line and then
dies with `IndexError: list index out of range` at `WARM,REP = int(sys.argv[1]), int(sys.argv[2])`:
it requires two positional arguments plus one `B,T,H,K,V` string per shape (`sys.argv[3:]`), none
of which the README mentions. The timing half of the second oracle is unreachable by following the
documentation.

## 4. The 85% line: where the forward kernels actually bind

AGENTS.md §6's acceptance line is that a kernel counts as optimised only when some hardware
resource — compute (`aic_mac_ratio` / `aiv_vec_ratio`) or movement (`mte2` / `mte3` share) —
reaches **85%** on real hardware. Before this task the unit had no `ops.json` entry, so under that
section's own rule its status was "unverified". It is now measured.

Shapes. `docs/matrix/models.json` gives the `gated_delta_rule` target as **Qwen3-Next-80B-A3B**:
16 key heads, 32 value heads, `head_k_dim = head_v_dim = 128`, with `qwen3_next_layer` at T=1024 and
`long_context` at T=4096. This unit has a **single head axis and no separate value-head axis**, so
the real shape can only be expressed GQA-expanded, as `H = 32` — which is also exactly `HMAX`, the
unit's asserted head ceiling. `T=4096, H=32, K=V=128` is therefore "the real shape" below, and the
contributor's `T=2048, H=8` "production shape" is kept alongside it for comparability with the
published table.

`aic_mac` is the fraction of the kernel's window in which the cube MAC pipe is busy; `aiv_mte2` the
vector core's load pipe. Bold marks each kernel's binding pipe.

**Real shape, `B=1, T=4096, H=32, K=V=128`** (total 6467.60 µs device):

| # | kernel | µs | share | blocks | `aic_mac` | `aic_mte2` | `aic_fix` | `aiv_vec` | `aiv_mte2` | cube % | best pipe | 85% |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| 1 | `gdr_fwd_train_wy` | 1052.00 | 16% | 28 | **0.469** | 0.257 | 0.279 | 0.275 | 0.212 | 96.94 | `aic_mac` 0.469 | **SHORT** |
| 2 | `gdr_fwd_train_scan` | 1138.51 | 18% | 28 | **0.761** | 0.547 | 0.046 | 0.057 | 0.233 | 75.37 | `aic_mac` 0.761 | **SHORT** |
| 3 | `gdr_bwd_dv_local` | 271.63 | 4% | 28 | **0.696** | 0.559 | 0.056 | 0.225 | 0.305 | 98.46 | `aic_mac` 0.696 | **SHORT** |
| 4 | `gdr_bwd_reverse_scan` | 1277.13 | 20% | 28 | **0.577** | 0.293 | 0.044 | 0.046 | 0.067 | 75.99 | `aic_mac` 0.577 | **SHORT** |
| 5 | `gdr_bwd_dqkw` | 1207.15 | 19% | 28 | **0.692** | 0.346 | 0.070 | 0.079 | 0.112 | 98.34 | `aic_mac` 0.692 | **SHORT** |
| 6 | `gdr_bwd_wy_matmuls` | 789.63 | 12% | 28 | **0.829** | 0.481 | 0.324 | 0.030 | 0.087 | 97.96 | `aic_mac` 0.829 | **SHORT** |
| 7 | `gdr_bwd_wy_reduce` | 507.41 | 8% | 56 | 0 | 0 | 0 | 0.175 | **0.816** | 0 | `aiv_mte2` 0.816 | **SHORT** |
| 8 | `gdr_bwd_finalize` | 224.14 | 3% | 56 | 0 | 0 | 0 | 0.277 | **0.896** | 0 | `aiv_mte2` 0.896 | **MET** |
| 9 | `gated_delta_rule_fwd` (inference, separate capture) | 2532.41 | — | 28 | **0.461** | 0.239 | 0.076 | 0.407 | 0.207 | 89.50 | `aic_mac` 0.461 | **SHORT** |

**Production shape, `B=1, T=2048, H=8`** (total 908.85 µs): best pipes 0.508 / 0.793 / 0.697 /
0.570 / 0.720 / 0.810 / `aiv_mte2` 0.611 / `aiv_mte2` 0.675 — **0 of 8 meet the line**.

**Large shape, `B=1, T=16384, H=16`** (total 14232.13 µs): 0.474 / 0.809 / 0.682 / 0.581 / 0.695 /
0.831 / `aiv_mte2` 0.813 / `aiv_mte2` **0.928** — **1 of 8 meets the line** (`finalize`).

**Verdict for the forward kernels, stated as AGENTS.md §6 requires — which pipe, and by how much.**

- `gdr_fwd_train_wy` is **nowhere near any single bound**: at the real shape its largest single
  pipe is `aic_mac` 0.469, and `aic_fix` 0.279 and `aiv_vec` 0.275 are close behind. No pipe is
  within 38 points of 85%. The cube pipes sum past 1.0 (0.469+0.078+0.257+0.279+0.062 = 1.145), so
  the work *is* overlapped; it is simply a four-phase mixed kernel in which no one resource
  dominates. Reaching 85% on any pipe here requires restructuring the phase split, not tuning.
- `gdr_fwd_train_scan` binds on `aic_mac` at 0.761 (real shape) / 0.793 (production) / 0.809
  (large) — **6 to 9 points short**, and its `cube_utilization` of 75.4% is a separate, fully
  explained loss (§5).
- `gated_delta_rule_fwd`, the inference forward, binds on `aic_mac` 0.461 — **39 points short**,
  the weakest of the nine. It is a two-phase kernel with a grid-wide barrier between the phases,
  so its window includes the barrier; that is the first thing to look at if it is ever optimised.

**The dominant term is the dtype, and the unit says so.** All arithmetic is FP32. The unit's own
back-to-back MMAD probe measures 623 ns per `[64,64]×[64,64]` MMAD in FP32 against 42 ns in
BF16 — a **14.8x** gap. So even a perfectly scheduled, zero-redundancy FP32 forward is an order of
magnitude off what the part can do in BF16. Moving the arithmetic to BF16 would break the
bit-identity with `fla`'s `naive_chunk` that §3.1 just confirmed, so it is a semantics decision for
the PM/user, not a tuning change, and I did not attempt it.

## 5. The finding that matters beyond this operator: `aic_mac_ratio` is not monotone with speed

`gdr_fwd_train_scan`'s host entry ships `_vt` and `_streams` overrides, documented as "override the
tiling for A/B measurement only; they change neither the arithmetic nor any output", and its
selector `_pick_vt_streams` chooses the V-tile width that minimises `ceil(items/cores) * streams`,
justified by a docstring claim: "a narrower V tile does not make an item cheaper, because matmul and
fixpipe always run on the full physical tile; it only creates more items."

That is a performance inference in a docstring, and AGENTS.md §6 iron law 1 says not to believe one
before profiling it. Five configurations, **one process and one capture each**, real shape
`T=4096, H=32, V=128`:

| config | selector cost `waves×streams` | device µs | vs default | `aic_mac` | `aic_mte2` | cube % | 85% verdict | output hash |
|---|--:|--:|--:|--:|--:|--:|---|---|
| **default** `(cores 28, streams 1, vt 64)` | 3 | **1086.97** | 1.00x | 0.792 | 0.521 | 75.69 | SHORT | `cc851b9d…` |
| `vt=64, streams=2` | 4 | 1231.41 | 1.13x | **0.911** | 0.372 | 56.99 | **MET** | `cc851b9d…` |
| `vt=48, streams=1` | 4 | 1568.58 | 1.44x | 0.736 | 0.566 | 84.66 | SHORT | `cc851b9d…` |
| `vt=32, streams=1` | 5 | 1801.64 | 1.66x | 0.795 | 0.495 | 90.86 | SHORT | `cc851b9d…` |
| `vt=32, streams=2` | 6 | 1834.89 | 1.69x | **0.917** | 0.327 | 75.99 | **MET** | `cc851b9d…` |

Three things come out of this.

**The author's claim and selector are correct.** Narrowing the V tile is monotonically worse, the
default is the fastest of all five, and the selector's cost ordering (3 < 4 ≤ 5 ≤ 6) is respected.
`vt=48` is the outlier within cost 4 because V=128 needs three 48-wide tiles and the third carries
only 32 valid columns while still paying a full 48-wide matmul. So there is **no host-side knob that
makes this kernel faster**, and the sub-85% `aic_mac` is not reachable by tuning the exposed surface.

**The overrides really are output-neutral.** All five configurations produce the identical SHA256
`cc851b9dadb47ff2e00e23e9139fdf29b4cbbea320340a698d5e37c4562d10b5` over `(o, s_hist, v_new)` —
bit-identical, confirming the docstring's second claim too.

**And the 85% bar can be met by a 13% regression.** `vt=64, streams=2` reaches `aic_mac` **0.911**
— comfortably over the line — while being **13% slower** than the default, which is SHORT at 0.792.
`vt=32, streams=2` reaches 0.917 while being 69% slower. The mechanism is not subtle:
`aic_mac_ratio` is *busy MAC time divided by the kernel's window*, and splitting V into more
same-cost items issues more full-width matmuls over a window that grows less than the MAC work
does. Both numerator and denominator move, and the ratio goes up while the wall-clock goes up too.

This is worth escalating as a methodology point, not just an observation about one kernel. The
repo-wide 85% acceptance line (AGENTS.md §6, user, 2026-10-27) is a stopping rule: "optimise until
some resource's pipe utilisation reaches 85%". On this kernel that rule is **satisfiable by making
the kernel slower**, with no source change and with bit-identical output. The line still does what
it was written for — it stops people declaring victory on "looks much faster now" — but it needs a
companion clause, something like *"and the configuration that meets the line must not be slower than
the best configuration measured"*. Otherwise a future task can pass the gate by regressing.
`cube_utilization` has the same problem in the other direction: `vt=32, streams=1` reads 90.9%
occupancy while running 66% slower. **That is a PM/user decision about the acceptance rule, not
mine to make.** I measured it and am reporting it.

### 5.1 Why the two scans sit at ~75% cube occupancy, in closed form

Not a symptom rule (AGENTS.md §6.5) — the arithmetic matches the measurement. `gdr_bwd_reverse_scan`'s
independent work is `B * H * ceil(V/64)` items; at the real shape that is `1 × 32 × 2 = 64` items on
28 blocks. Wall time is set by the **most loaded** core, so with 64 items over 28 cores, 8 cores take
3 and 20 take 2: mean 64/28 = 2.286, max 3. Predicted occupancy = 2.286/3 = **76.2%**; measured
**75.99%**. `gdr_fwd_train_scan` at `(cores 28, streams 1, vt 64)` has the same 64-item space and
measures **75.69%**.

So the residual is **wave quantisation**, and it is not removable from the host side: fewer blocks
would raise the ratio without changing the 3-wave wall time (pure metric gaming — see above), and
more items of the same cost is what §5 just measured as slower. Removing it needs items that are
genuinely cheaper, i.e. a matmul over a partially valid tile — which the unit documents as a
hardware constraint it is shaped around ("a partially valid N axis leaves stale rows in the
destination of a split-M drain"). **Recorded as at the structural limit of the current design, with
the 23.8% quantisation loss quantified**, rather than as an open tuning opportunity.

## 6. Against the framework: the published speed-ups do not transfer

Same session, same card, `fla`'s `naive_chunk_gated_delta_rule` under `torch_npu`, FP32 inputs,
`scale = 1/sqrt(K)`, 3 warm-up + 5 timed, median — the unit's own baseline script with the
arguments it actually needs.

| shape | `fla` naive_chunk published | `fla` measured here | ours wall published | ours wall measured | published speed-up | **measured speed-up** |
|---|--:|--:|--:|--:|--:|--:|
| `B=1, T=2048, H=8, K=V=128` | 9161.3 µs | **5845.8 µs** | 695.1 µs | **748.8 µs** | 13.2x | **7.81x** |
| `B=1, T=16384, H=16, K=V=128` | 32361.7 µs | **28872.1 µs** | 5391 µs | **5152.2 µs** | 6.0x | **5.60x** |

The large shape roughly holds (5.60x against 6.0x). **The production-shape headline does not**: 7.81x
against a published 13.2x. The cause is on the *reference* side — `fla`'s naive_chunk ran **36%
faster** here than the contributor measured (5845.8 vs 9161.3 µs), while this unit's own forward ran
7.7% slower in wall clock (748.8 vs 695.1 µs) with its device time within 1.3% (§3.4). Since the
device side is reproducible and the reference side is not, the divergence is in `fla`'s eager-mode
dispatch, which is exactly what `PERFORMANCE.md`'s own first caveat warns about ("its triangular
inverse is a 63-iteration Python loop doing in-place slice assignment, so much of its cost is
eager-mode launch overhead rather than arithmetic").

So the document's disclosure was right and its numbers are not portable. **The defensible statement
after this task is "5.6x to 7.8x faster than `fla`'s own naive reference at the shapes stated, on
this host" — not 6.0x to 13.2x.** The arithmetic-floor comparison (§3.4's 2.41x over the FP32 floor)
is the one that does transfer, and that is the comparison `PERFORMANCE.md` itself recommends relying
on.

### 6.1 The composed entry point

Host wall per step, no profiler, 12 warm-up iterations discarded and 24 measured — the document's own
prescription, which it arrived at after finding that three warm-ups read ~50% high.

| arm | published µs/step | measured here | runs |
|---|--:|--:|--:|
| the eight wrappers chained as they ship | 1617.5 | **1574.8, 1560.2** | 2 |
| pipelined, 4 steps queued | 912.7 | **911.9** | 1 |
| pipelined, 8 steps queued | 864.7 | **858.5** | 1 |
| pipelined, 16 steps queued | **828.5** (p90 831.4) | **835.7, 842.2, 857.2, 882.7** | 4 |

The 4-step row reproduces to 0.09% and the 8-step row to 0.7%. The shipped-chain row is 2.6–3.5%
*faster* here. The **16-step headline does not cleanly reproduce**: four independent unprofiled runs
span 835.7–882.7 µs/step, with the published 828.5 below all four. The document's within-run
steadiness claim holds (its p90 is 0.3% over its median; my best run's spread is comparable), but
**between** runs I see 5.6% of drift, which its p90 figure does not capture. Taking my own best
pipelined run against my own best chain run gives **1560.2 / 835.7 = 1.87x**, against the published
1.95x.

None of this is a defect — it is one host's numbers versus another's, and the ordering of the arms
(more steps queued is better, and the pipelined path is below the 908.85 µs device sum, so steps
really do overlap device work) reproduces in full. It does mean **"828.5 µs, 1.95x" should be quoted
as a range, not a point**, and that the figure is more host-sensitive than the device-time table.

## 7. HF32: measured, not assumed — this unit is not exposed

`gaps.json` `a5-fp32-cube-matmul-inherits-hf32` is a repo-level **P0**: an FP32 cube matmul follows
the AI core's HF32 rounding bit, kernel dispatch does not reset it, so on a shared card a co-tenant
can silently halve the mantissa. DR-P1's spec says to verify whether PyPTO-Pro's execution model
carries the same risk, assuming neither way.

**Static.** PyPTO-Pro exposes no HF32 control at all: `pl.matmul`'s signature is
`(dst_tile, lhs_tile, rhs_tile, bias_tile=None, *, phase=None)` — no precision argument — and the
string `hf32`/`HF32` appears **nowhere** in `pypto`/`pypto_pro`'s Python sources, nor in any of
their shared objects (`pypto_impl…so`, `libtile_fwk_{runtime,adapter,platform,simulation_pv}.so`;
the only near-match is the unrelated dtype `hifloat8_t`). So there is no `set_hf32(False)` to add,
and the ascriptor-style fix does not exist here. Also noted: `msprof`'s `op_summary` reports
`HF32 Eligible = YES` for **all nine** kernels, at every shape — so eligibility alone says nothing.

**Empirical, in one process, same card, with a positive control.** Baseline chain → determinism
control → force HF32 on via `torch.npu.matmul.allow_hf32 = True` plus four 4096×4096 FP32 matmuls →
chain again → force it off → chain again. SHA256 over all six public outputs:

```
A1 chain baseline                      776bf9be93bc3e11edf909c43bb1ead203ad5a0326f6a3935f6b3f1e9be7b21c
A2 chain determinism control           776bf9be…  equal_to_A1=True
PC torch fp32 matmul allow_hf32=True   556b3b81f88ba153c0edf673ca7a79ee1f4192a5f25b8797013e7c0da6a84c81
PC torch fp32 matmul allow_hf32=False  e91b724094fe8733cb09a076dc1915fdd7d752ad53b18e631ebf79088fc836cb
   positive control moved              True
B  chain after HF32=True was in force  776bf9be…  equal_to_A1=True
C  chain after HF32=False was in force 776bf9be…  equal_to_A1=True
```

The positive control **did** move (the torch matmul's own hash changes with the flag), so the flag
reaches the hardware and an unchanged chain is evidence rather than a dead probe.

**Empirical, cross-process — the actual co-tenant threat model.** The gap is about a bit left behind
by *another* process, so the two halves were run as separate processes on the same card: chain
(baseline) → a separate process that sets `allow_hf32=True`, runs eight 4096×4096 FP32 matmuls and
exits → a fresh chain process.

```
CHAIN hash  (baseline)                 776bf9be93bc3e11edf909c43bb1ead203ad5a0326f6a3935f6b3f1e9be7b21c
SETTER hf32=True matmul hash           c6d51980ed45e156d5a9c906fb34c7f3a59a9f16ae75eb8bdfdc38a6924579b3
CHAIN hash  (fresh process, after)     776bf9be93bc3e11edf909c43bb1ead203ad5a0326f6a3935f6b3f1e9be7b21c
```

Bit-identical. **Verdict: this unit is not exposed to `a5-fp32-cube-matmul-inherits-hf32` by either
the within-process or the cross-process mechanism, on this part and this CANN, with a positive
control that moved.** No `set_hf32` guard is needed and none is available.

Stated limits, so nobody over-reads this. It establishes that *this* lever — the only one reachable
from `torch_npu` — does not perturb these kernels; it does not prove that no conceivable way of
leaving the bit set could. And it is an A5/CANN-9.1.0-beta.1 result, which under AGENTS.md §6
("conclusions do not inherit across SoCs") says nothing about A2 or A3.

## 8. Tooling: PyPTO-Pro and the official orchestrator plugin

AGENTS.md §6 requires the latest PyPTO-Pro framework plus the official
[`pypto-pro-op-orchestrator`](https://gitcode.com/cann/pypto-gym/tree/master/cannbot-skills/plugins-official/pypto-pro-op-orchestrator)
before optimisation work. `KDA-P1` and `GDN-P1` both found the plugin inapplicable because their
kernels are ascriptor/CCE. This unit is native PyPTO-Pro, so it is the first in the repo where the
requirement bites. What actually happened, reported as the task asks:

**Framework.** Genuine PyPTO-Pro was already installed and is what the unit runs on: `pypto` dist
`0.2.1` with `pypto_pro` under `/usr/local/Ascend/cann/python/site-packages`, `pl.jit` /
`pl.vector_function` / `pl.Vf` all present, `get_platform_info()` returning the correct part. I did
**not** upgrade it, for two deliberate reasons. First, it is read-only shared CANN
(`/usr/local` is read-only in the task container and AGENTS.md §5 forbids writing there), so an
upgrade would mean a private build of `pypto` — and building it needs Python development headers
that the container's system Python 3.11 lacks. Second and more important: **DR-P1's primary job was
to reproduce the contributor's numbers, which were taken on this exact framework and CANN.**
Upgrading first would have destroyed the comparison. Re-measuring on a newer PyPTO-Pro is a sensible
follow-up and is now cheap — the whole harness is in place — but it is a *different* experiment and
needs its own task. Flagging it as a deliberate deviation for the PM.

**Plugin.** Obtained and used. Its installable surface is six agent definitions (methodology
prompts), reference documents, an OpenCode-only state machine, and one mechanically useful piece:
the `pypto_pro_op_lint` hard-gate checker with 13 rules. I did **not** run `init.sh` against this
repository, because it installs skills and agent files into the project tree and DR-P1's write set
does not include them; I ran the lint directly from the plugin checkout instead, and wrote nothing
into the repo.

The lint resolves an "op directory" as one holding `test_{op}.py` beside `{op}_golden.py` /
`{op}_golden_cpu.py`. This unit keeps all nine operators' tests in one shared `tests/`, so `{op}` is
ambiguous and the lint cannot infer it; I built a per-operator view of **symlinks** (nothing copied
or modified) and linted each of the nine.

| rule | what it checks | result |
|---|---|---|
| PL11 | goldens must not `import pypto` / `pypto_pro` (golden purity) | **PASS, 9/9** |
| PL15 | `test_{op}.py` must contain `_assert_precision`, must not use `assert_close` | PASS 8/9 |
| PL16 | `test_{op}.py` must import `{op}_golden_cpu` | PASS 8/9 |
| PL01 | test/staged file must `import pypto_pro.language as pl`, must not import non-Pro `pypto` | FAIL 9/9 — **layout mismatch, not a defect** |
| PL02 | exactly one `@pl.jit` per test/staged file (single-kernel rule) | FAIL 9/9 — **same** |
| PL03–PL10, PL13 | gate artefacts (`SPEC.md`, `DESIGN.md`, `module_interfaces.yaml`, `.orchestrator_state.json`) | not applicable to this layout |

**PL01/PL02's uniform failure is the plugin's workflow assumption, not a finding.** In the
orchestrator's Stage 4 the kernel lives *inside* `test_{op}.py`; this unit separates
`kernels/{op}_impl.py` from `tests/test_{op}.py`, so the test file legitimately contains no
`@pl.jit`. The impl files do import `pypto_pro.language as pl` and do hold exactly one `@pl.jit`
each. I am recording this as "the lint has no verdict on this layout" rather than as nine failures.

**PL11 passing 9/9 is a real result**: no golden reaches for the framework it is supposed to be
independent of, which is what makes the double-oracle claim in §3.6 meaningful.

**PL15/PL16 failing on exactly one operator is a real signal, and it points at the same kernel the
gating gaps do.** `gated_delta_rule_fwd` — the standalone inference forward, the ninth kernel, the
one with zero domain validation — is the only one whose precision assertion is named
`_assert_precision_o` rather than `_assert_precision` and whose golden is `gated_delta_rule_fwd_golden.py`
rather than `..._golden_cpu.py`. **These two are naming deviations, not a missing CPU reference**:
`tests/test_gated_delta_rule_fwd.py:50-92` does call the FP32 golden on `.cpu()` tensors and does use
the shared `precision_compare.check_precision`. But chasing them turned up a genuine documentation
error: `VALIDATION.md` says "Per kernel there is a float32 CPU golden and an NPU golden", and this
kernel has **one** golden file, no NPU golden. Booked in the gaps entry for that kernel.

So: the plugin is applicable to this unit — it is the first unit in the repo where it is — and it
earned its place by isolating the ninth kernel on content rules, independently of the source reading
that found the same kernel's missing gate. Its gate rules (PL03–PL10) assume the orchestrator's own
directory contract and have nothing to say about an externally-structured unit.

## 9. New correctness evidence: the real model shape, never validated before

`VALIDATION.md`'s "What is not validated" lists as its first gap: "**Correctness has never been
checked at T=16384, H=16** … The largest validated shape is T=2048, H=8." AGENTS.md §6 is blunt about
why that matters — "performance is measured at real shapes, so precision must be too", and the (C, HV)
lesson is that two axes each green separately does not make their combination green.

So the unit's own composition gate, its own judge, its own staging goldens, unmodified, at the real
model shapes:

| case | B | T | H | K | V | result |
|---|--:|--:|--:|--:|--:|---|
| `qwen3next_layer_expanded_b1_t1024_h32_k128_v128` | 1 | 1024 | **32** | 128 | 128 | **PASS** |
| `qwen3next_keyheads_b1_t1024_h16_k128_v128` | 1 | 1024 | **16** | 128 | 128 | **PASS** |

Both clean on all 36 comparisons. **H=32 is `HMAX` itself**, the unit's asserted head ceiling, and
H=16 is the head count whose latency `PERFORMANCE.md` quotes and whose correctness it admits it never
checked. Neither had been validated before this task. This closes the H axis up to its declared
ceiling at a real T; it does **not** close `T=16384` — the CPU oracle at 16× the work was out of this
task's budget, and that gap stays open.

The ninth kernel's own acceptance tests also pass on real hardware, all 8 cases including
`wide_kv K256 V256` (matched_ratio 1.000000, `our_max_abs` 5.7158e+01 ≤ `peak·2⁻⁶` = 3.1678e+02 and
≤ 1.05× the ideal bf16 rounding of the golden) and `batch B3 T192 K128 V96` (matched_ratio 1.000000).
Those two numbers are the measured evidence that this kernel's legal domain really is wider than the
training chain's — which is why its complete absence of domain validation is booked as a gap rather
than shrugged off.

## 10. What DR-P1 changed, and what it decided not to

**Changed:** three entries added to `docs/matrix/gaps.json`
(`gdr-v-domain-never-validated`, `gdr-b1-enforced-mid-chain`,
`gdr-inference-fwd-no-domain-validation`), all `requires_kernel_change: true` with
`kernel_change_note`, plus the summary counts and `kernel_fix_queue` they imply; and
`docs/matrix/README.md` regenerated by `tools/gen_matrix.py` because it is generated from that JSON
and `--check` enforces the pair. These two documents.

**Not changed: no line of kernel source, and no document inside the unit.** Every discrepancy §3,
§6 and §8 found in the unit's own `README.md`, `PERFORMANCE.md` and `VALIDATION.md` — the recurrent-`o`
interval, the 185-vs-180 count, the five-vs-four zero boundaries, the six-of-eight occupancy claim,
the unreachable `fla_npu_baseline.py` invocation, the "per kernel … and an NPU golden" line, the
non-portable 13.2x — is reported here and left in place. They are documentation accuracy issues in
externally-contributed code; correcting them is a small, safe, separate change, and AGENTS.md §6's
judgement discipline says not to mix it into a measurement task. Recommended as its own follow-up.

**Open for the PM or the user, not for me:**

1. **The 85% acceptance rule needs a companion clause** (§5). As written it is satisfiable on
   `gdr_fwd_train_scan` by a 13%-slower configuration with bit-identical output and no source change.
2. **This unit still has no `ops.json` entry**, so by AGENTS.md §6's own rule its status reads
   "unverified" even now that it is measured. `docs/matrix/ops.json` is not in DR-P1's write set, so I
   did not add one; everything an entry needs is in §3, §4 and §9.
3. **The forward's real ceiling is the FP32 dtype, not its scheduling** (§4). A BF16 arithmetic path
   is a ~14.8x lever and would break the bit-identity with `fla` that §3.1 confirms. Semantics
   decision.
4. **`T=16384` correctness** remains unvalidated (§9), now the only shape gap left on the axes this
   unit declares.
5. **A PyPTO-Pro framework upgrade was deliberately not done** (§8), to keep the reproduction
   comparable. If the PM wants the newer framework measured, the harness is in place.
