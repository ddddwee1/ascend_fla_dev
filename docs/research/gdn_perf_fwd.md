# GDN-P1 · GDN chunk forward: real-hardware pipe utilisation and the block_dim ceiling

Task `GDN-P1`, A5 real hardware. Environment, oracles and measurement discipline
are the ones in `gdn_perf_decode.md` §1; every number below comes from a separate
process with its own vendor tree and compile cache, run under the device lock on
one physically assigned card, with `torch.npu.matmul.allow_hf32` pinned `False`
and recorded per run.

Shapes are `docs/matrix/models.json`: `qwen3_next_layer` (`B1/T1024/H16/HV32/K=V=128`)
and `long_context` (`B1/T4096/H16/HV32`). Accuracy is judged in FP32 against the
unit's independent CPU block-solve reference at a 1e-4 relative-L2 budget, on the
same real shapes as the timings (§6: "性能在真实形状上测，精度也必须在真实形状上测").

## 0. First, which code is the GDN forward

`ascend_fla/ops/gdn_chunk_fwd.py:112` builds the production pipeline from
`_native_pipeline()`, which is
`kernels/projects/a5/gdn_chunk_fwd_bf16/kernels` — **for FP32 as well as BF16**.
`_pipeline()` (`kernels/projects/a5/gdn_chunk_fwd/kernels`) is labelled in the
source as "Read-only legacy unit access for existing diagnostic callers/tests"
and is not on the public path. Inside that one directory,
`stages.py:513-514` declares

```python
STAGES_F32  = (gdn_bf01_prepare_f32,  gdn_bf01_scores, gdn_bf01_wy, gdn_bf01_scan, gdn_bf01_output_f32)
STAGES_BF16 = (gdn_bf01_prepare_bf16, gdn_bf01_scores, gdn_bf01_wy, gdn_bf01_scan, gdn_bf01_output_bf16)
```

so `scores`, `wy` and `scan` are the *same kernel objects* for both dtypes, and
`output_vf` is shared by both output kernels. The profiler confirms it: the FP32
run's native entries are `gdn_bf01_prepare_f32 / gdn_bf01_scores / gdn_bf01_wy /
gdn_bf01_scan / gdn_bf01_output_f32`, exactly the names GDA-05 recorded.

This matters for scope and is recorded here rather than acted on: the GDN-P1
write set names `kernels/projects/a5/gdn_fwd/**` (a directory that does not
exist) and says not to touch the BF16 write set, but **the FP32 forward has no
kernel source of its own** — any forward kernel change necessarily edits
`gdn_chunk_fwd_bf16/**` and necessarily affects BF16. §3 below measures two such
changes on real hardware without committing them, so the write-set decision can
be made on numbers instead of a prediction.

## 1. Baseline: where the time goes and what binds

`long_context` (T=4096), FP32, block_dim 1. Device kernel time is the median of
three profiler captures; shares are of the device kernel total.

| native entry | device us | share | `aiv_vec_ratio` | `aiv_mte2_ratio` | `aiv_mte3_ratio` | `aiv_scalar_ratio` | `aic_mac_ratio` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gdn_bf01_scores`      | 55337.894 | 35.39 % | **0.977** | 0.024 | 0.006 | 0.001 | 0.000 |
| `gdn_bf01_wy`          | 45300.040 | 28.97 % | **0.974** | 0.026 | 0.011 | 0.001 | 0.000 |
| `gdn_bf01_scan`        | 31800.207 | 20.34 % | **0.959** | 0.040 | 0.021 | 0.001 | 0.000 |
| `gdn_bf01_output_f32`  | 21222.263 | 13.57 % | **0.899** | 0.076 | 0.042 | 0.012 | 0.000 |
| `gdn_bf01_prepare_f32` |  2719.709 |  1.74 % | 0.188 | 0.520 | 0.549 | 0.039 | 0.000 |
| total | 156379.760 | | | | | | |

Worst relative L2 vs the CPU block-solve reference: 4.503e-07 on `o`, 4.270e-07
on `final_state`; budget 1e-4.

`qwen3_next_layer` (T=1024) is the same picture at a quarter of the cost
(total 39136.644 us; scores 13849.094 / wy 11333.607 / scan 7949.083 /
output 5322.317 / prepare 675.027 us; vec ratios 0.976 / 0.974 / 0.960 / 0.896 / 0.190),
worst relative L2 4.444e-07.

Two facts follow immediately.

**`aic_mac_ratio = 0.000` in all five stages, `cube_utilization ≤ 0.004 %`.** The
GDN forward contains no `matmul`: all five stages are `@vf()` nests of
`RegList`/`Reg` element-wise operations and scalar broadcasts. **Task item 4 is
therefore answered negatively: the repository-level P0
`a5-fp32-cube-matmul-inherits-hf32` has no reachable site in the GDN forward.**
This is the same "cube engine idle" structure PK-07 measured in PKDA, arrived at
independently here from the device counters.

**`prepare` is the one data-movement-bound stage** (`mte2` 0.520 + `mte3` 0.549,
which overlap) and it is 1.74 % of the total. It is not a target.

## 2. Change A (committed): the block_dim ceiling was 2, the chip has 56 vector cores

`ascend_fla/ops/gdn_chunk_fwd.py` and `gdn_chunk_bwd.py` accepted only
`block_dim in (1, 2)`, while the sibling `gdn_fused_recurrent` has accepted
`(1,2,4,8,16,28)` since GDA-04. The cap was the qualified scope, not a limit:
`GetVecNum() == 2 * block_dim`, so bd2 reaches 4 of Ascend950PR's 56 vector cores
— 7 % of the machine. Every launch partitions **independent** GM slices with no
cross-core reduction:

| launch | partitioned over | items at B1/T4096/H16/HV32 |
|---|---|---|
| `prepare`, `scores`, `wy`, `output` | `B*N*HV` | 2048 |
| `scan` | `B*HV` | 32 |

Change A widens both grids to `(1,2,4,8,16,28)`. 28 is the ceiling because a
block_dim above the physical core count deadlocks on the hardware barrier
(AGENTS.md §5). No kernel source is touched.

### Measured, `qwen3_next_layer` (T=1024), FP32

| block_dim | device kernel total us | vs bd1 | vs bd2 (old maximum) |
|---:|---:|---:|---:|
| 1  | 39144.036 | 1.000x | 0.501x |
| 2  | 19607.914 | 1.996x | 1.000x |
| 4  |  9840.322 | 3.978x | 1.993x |
| 8  |  4958.516 | 7.894x | 3.954x |
| 16 |  2480.180 | 15.783x | **7.906x** |
| 28 |  1820.993 | 21.496x | **10.767x** |

### Control: the ops-layer change is inert on the old grid

Same card, baseline build `758468cd` vs the change-A build, T=1024:

| block_dim | base build | change A | drift |
|---:|---:|---:|---:|
| 1 | 39136.644 us | 39144.036 us | +0.019 % |
| 2 | 19597.633 us | 19607.914 us | +0.052 % |

Both well inside run-to-run noise, as required: widening an accepted-value set
must not change what bd1/bd2 do.

### Correctness and bitwise neutrality

- Worst relative L2 is **identical at every block_dim** — 4.444e-07 (T=1024) and
  4.503e-07 (T=4096) — against the CPU FP32 reference, budget 1e-4. The budget
  was not touched.
- **The public `o` and `final_state` SHA-256 form a single equivalence class per
  shape across both builds and all six block_dims** (`o = 2c540d8f6d15…` and
  `final_state = 11d813fb11bc…` at T=4096). This is §6's reference-free hard
  criterion: core splitting must not change the arithmetic, and it does not.

### Where the scaling stops, and why it was predictable

At bd28 the per-stage breakdown shows `gdn_bf01_scan` flat against bd16 —
512.964 us at bd16 vs 512.558 us at bd28 — while the other four keep scaling
(scores 886.301 → 558.423, wy 723.297 → 457.818, output 355.860 → 232.896 us).
That is exactly the structural ceiling the source predicts: `scan` partitions
`B*HV = 32` items and `GetVecNum()` is already 32 at bd16, so 24 of the 56 cores
have `begin == end` and no work. The prediction was made from the source before
the bd28 run and the device confirmed it, which is the one direction §6 iron
rule 1 permits (predict, then measure, then report both).

Pipe ratios at bd16, T=1024: scores 0.961, wy 0.961, scan 0.943, output 0.865 —
all still ≥ 0.85 — and prepare 0.133 vec / 0.649 mte2 / 0.380 mte3 at 2.68 % of
the total. Raising block_dim does not change the pipe mix materially, which is
the expected result: it divides the same per-item work over more cores.

## 3. The remaining forward headroom is real, measured, and write-set blocked

`aiv_vec_ratio` of 0.96–0.98 on `scores` and `wy` reads like "nothing left", and
that reading is wrong. **`aiv_vec_ratio` is pipe *occupancy*, not issue
efficiency: a dependency-stalled vector pipe still counts as busy.** The proof is
already in this repository, merged, on real hardware:

PK-08 (merged `3541bcd2`) rewrote PGDN's structurally identical `scores_vf` from a
flat guarded triangle to a `BLK x BLK` blocked triangle built from compile-time
`unroll()` and constant inner trip counts, and measured
**112.274 ms → 30.026 ms** at `t4096_h16_hv32`, bd1, FP32.

GDN's `scores` at the identical shape, block_dim and dtype measures **55.338 ms**
here (55.320 ms in GDA-05). Both forms perform exactly the same number of
128-wide `cadd` reductions per 64-token tile — 4096 (GDN: 2080 for `score` plus
2016 for `lower`; PGDN post-fix: 3584 in the sub-diagonal blocks plus 512 in the
diagonal blocks) — and the PGDN form executes *more* inner iterations (2304 vs
2080). It is nevertheless **1.84x faster** than GDN's.

The only structural difference is the loop form. GDN uses
`for i in range(count)` with `for j in range(i + 1)`: a dynamic outer bound and
an inner bound derived from the enclosing induction variable. PGDN's merged form
has no dynamic bound anywhere.

`count` in GDN is provably always 64: the public entry rejects `T` that is not a
multiple of 64 (`gdn_chunk_fwd.py:68`), and `count = Min(C, T - cc*C)` with
`cc < N = T/64`, so no accepted input ever makes it smaller. The dynamic bound
buys nothing.

`wy` is the other candidate and the only GDN stage already written with a
constant trip count — but with a guard over all of it:

```python
for j in range(C):
    if j < i:
        ...
```

The source comment records why the constant trip count is there: on this
toolchain a VF loop bounded directly by the enclosing VF induction variable
"omitted the last contribution", and constant-trip + guard was the scoped
workaround. Blocking the triangle **preserves that property** — every trip count
stays constant, no dependent bound is reintroduced — while dropping the guard
from the sub-diagonal blocks, where `j < i` is unconditionally true. Executed
inner iterations per tile fall from 2048, all guarded, to 896 unguarded plus 256
guarded, with the same 992 doing work.

Both rewrites keep the per-`(i, j)` expression, its single destination address
and (for `wy`) the ascending `j` accumulation order, so both are bitwise
neutral by construction.

**Both changes live in `kernels/projects/a5/gdn_chunk_fwd_bf16/kernels/stages.py`
and affect the BF16 path identically, because the stages are shared objects
(§0).** They are therefore *measured* below but **not committed**: the GDN-P1
write set excludes that directory. PK-08's write set was expanded for exactly
this class of discovery; this is the same situation, and the decision belongs to
the PM. The exact patches are reproduced in §5.

### Measured on real hardware (not predicted)

Builds: `candA` = change A only (the committed ops-layer change); `candB` =
candA + blocked `scores_vf`; `candD` = candA + blocked `wy_vf`; `candC` = candA +
both. Same card, same session, one process and one vendor tree per row.

#### `scores_vf` (candB), per-stage device us, median of 3 captures

| stage | base, T1024 bd1 | candB, T1024 bd1 | ratio | base, T4096 bd1 | candB, T4096 bd1 | ratio |
|---|---:|---:|---:|---:|---:|---:|
| `gdn_bf01_scores`      | 13849.094 |  **7533.391** | **1.838x** | 55337.894 | **30056.191** | **1.841x** |
| `gdn_bf01_wy`          | 11333.607 | 11333.615 | 1.0000x | 45300.040 | 45302.448 | 1.0000x |
| `gdn_bf01_scan`        |  7949.083 |  7953.026 | 0.9995x | 31800.207 | 31794.009 | 1.0002x |
| `gdn_bf01_output_f32`  |  5322.317 |  5318.275 | 1.0008x | 21222.263 | 21226.451 | 0.9998x |
| `gdn_bf01_prepare_f32` |   675.027 |   680.131 | 0.9925x |  2719.709 |  2710.270 | 1.0035x |
| **kernel total**       | **39136.644** | **32822.435** | **1.192x** | **156379.760** | **131094.411** | **1.193x** |

At T=1024 block_dim 2 the same change gives 19597.633 → 16439.217 us (**1.192x**),
so the gain is independent of block_dim as expected.

Three things to note.

1. **The 1.84x was predicted before the run** from the GDN-vs-PGDN cross-unit
   comparison above, and measured at 1.838x / 1.841x. Independently, candB's
   `scores` at `t4096_h16_hv32` bd1 FP32 measures **30056.191 us** against
   PK-08's merged PGDN `scores` at the identical shape/bd/dtype, **30026 us** —
   a 0.10 % match. The two units' score stages now perform identically, which is
   what "same arithmetic, same loop form" should produce.
2. **The four untouched stages move by at most 0.35 %**, and `wy` is identical to
   five significant figures (11333.607 → 11333.615 us). The change's blast radius
   is exactly the one kernel.
3. **`aiv_vec_ratio` for `scores` went DOWN, 0.977 → 0.955, while the stage got
   1.84x faster.** This is the local, same-kernel proof of §4's methodological
   point: the higher occupancy belonged to the slower code.

#### CCE emission diff, candB (scope proof)

Generated CCE sources under `<build>/<kernel>-<hash>/project/op_kernel/`,
base build vs candB, SHA-256 (first 16 hex digits):

| kernel | file | base | candB | |
|---|---|---|---|---|
| `gdn_bf01_scores` | `scores_vf.h`            | `1c76f4cd5de9ede8` | `e5b0d693ea3a0e26` | **differs** |
| `gdn_bf01_scores` | `gdn_bf01_scores_vec.h`  | `75e6a6caa69f8110` | `4026fe33d66a1550` | **differs** |
| `gdn_bf01_scores` | `gdn_bf01_scores_cube.h` | `6bdebb495d6a996d` | `67285aac00543b89` | **differs** |
| `gdn_bf01_scores` | `gdn_bf01_scores.cpp`    | `8c6a63a716e7c323` | `8c6a63a716e7c323` | same |
| `gdn_bf01_wy` | `wy_vf.h`                    | `350fa06d332f1090` | `350fa06d332f1090` | same |
| `gdn_bf01_wy` | `gdn_bf01_wy_vec.h`          | `0ad806ecab5d9dd0` | `0ad806ecab5d9dd0` | same |
| `gdn_bf01_scan` | `scan_vf.h`                | `e7444ec5ab74d569` | `e7444ec5ab74d569` | same |
| `gdn_bf01_scan` | `gdn_bf01_scan_vec.h`      | `61967a73850c25ac` | `61967a73850c25ac` | same |
| `gdn_bf01_output_f32` | `output_vf.h`        | `a8100d36fe5b1e87` | `a8100d36fe5b1e87` | same |
| `gdn_bf01_output_f32` | `gdn_bf01_output_f32_vec.h` | `705839500ebb7818` | `705839500ebb7818` | same |
| `gdn_bf01_prepare_f32` | `prepare_tile.h`    | `db9cf952491ca64e` | `db9cf952491ca64e` | same |
| `gdn_bf01_prepare_f32` | `gdn_bf01_prepare_f32_vec.h` | `c89392c2e010690b` | `c89392c2e010690b` | same |

Every generated file of the four untouched kernels is byte-identical, and only
the three files belonging to `scores` change. Note that the per-kernel build
*directory* hash changes for all seven kernels, because it digests the whole
`stages.py` module — the directory name is therefore **not** a usable emission
hash, and the per-file comparison above is the one that answers the question.

#### Bitwise neutrality and accuracy, candB

Public output SHA-256, base build vs candB, identical at every measured
block_dim and both shapes:

| shape | `o` | `final_state` |
|---|---|---|
| T=1024 | `cd985f2170bf2745429e47f992608ed6e8f7329c2b9bb13dbb35c502216ecef6` | `87d92c42ae7f174f9cbfcd04b05a5b8fcb0d1fa19666c043ef74e821b915ea31` |
| T=4096 | `2c540d8f6d1563053993e21324143ecdc2fc0162ed0a0e570b7c9df7168ae11f` | `11d813fb11bcf212418b9d582721af418d2d5dc4297f911e2b1e2279f4917a12` |

Worst relative L2 against the CPU block-solve reference is unchanged to all
printed digits: 4.444e-07 (T=1024) and 4.503e-07 (T=4096). Budget 1e-4, untouched.

#### `wy_vf` (candD), per-stage device us, median of 3 captures

| stage | base, T1024 bd1 | candD, T1024 bd1 | ratio | base, T4096 bd1 | candD, T4096 bd1 | ratio |
|---|---:|---:|---:|---:|---:|---:|
| `gdn_bf01_wy`          | 11333.607 |  **3290.507** | **3.444x** | 45300.040 | **13119.799** | **3.453x** |
| `gdn_bf01_scores`      | 13849.094 | 13849.771 | 1.0000x | 55337.894 | 55337.759 | 1.0000x |
| `gdn_bf01_scan`        |  7949.083 |  7950.486 | 0.9998x | 31800.207 | 31790.380 | 1.0003x |
| `gdn_bf01_output_f32`  |  5322.317 |  5329.377 | 0.9987x | 21222.263 | 21226.958 | 0.9998x |
| `gdn_bf01_prepare_f32` |   675.027 |   675.431 | 0.9994x |  2719.709 |  2717.567 | 1.0008x |
| **kernel total**       | **39136.644** | **31096.228** | **1.259x** | **156379.760** | **124180.263** | **1.259x** |

**3.44x on `wy`, where iteration counting alone predicted only 1.78x.** The
prediction (2048 → 1152 executed inner iterations) is written down in the patch
header and was wrong in magnitude, in the favourable direction: the measured
factor requires the *guarded* iterations to have been several times more
expensive than unguarded ones, which is exactly the effect PK-08 fitted on PGDN
(≈19.5 ns guarded vs ≈2.0 ns unguarded for a 7-instruction body). Recording the
miss rather than the post-hoc rationalisation: the iteration count is a lower
bound on this kind of gain, not an estimate of it.

`aiv_vec_ratio` for `wy` went 0.974 → **0.907** while the stage got 3.44x faster —
the second independent instance of higher occupancy belonging to slower code
inside this one unit.

CCE emission diff, base vs candD: only `wy_vf.h`, `gdn_bf01_wy_vec.h` and
`gdn_bf01_wy_cube.h` differ; every generated file of the other four kernels is
byte-identical. Public output SHA-256 equals the baseline's at both shapes
(`o = cd985f2170bf…` / `2c540d8f6d15…`), and worst relative L2 is unchanged
(4.444e-07 / 4.503e-07).

#### Both together (candC)

<!-- MEASURED_KERNEL_TABLE -->

## 4. Verdict against the 85 % line

- The forward is **vector-arithmetic bound**: four of five stages sit at
  0.865–0.977 `aiv_vec_ratio` across bd1…bd28, i.e. above the 85 % line, and the
  fifth (`prepare`, 1.7–3.2 % of the total) is MTE-bound at 0.52–0.66 `mte2`.
- **The cube engine is 0.000 % used.** There is no FP32 cube matmul in the GDN
  forward, so the HF32 inheritance P0 does not apply, and conversely the single
  largest remaining lever is algorithmic: `scores`, `wy`, `scan` and `output` are
  all GEMV/GEMM-shaped computations executed with scalar-broadcast vector loops.
  PK-07's round-2 cube port of one PKDA stage reached `aic_mac_ratio = 0.635`, so
  that path is a multi-session design task per stage, not a tuning step.
- **Methodological finding for the acceptance line itself:** on these kernels
  `aiv_vec_ratio ≥ 0.85` is *not* sufficient evidence of optimality. GDN `scores`
  measures 0.977 and is still 1.84x slower than a bitwise-equivalent formulation
  of the same arithmetic. The line should be read as "no single pipe is starved",
  not as "no work is left"; a same-arithmetic cross-implementation comparison,
  where one exists, is the stronger test.

## 5. Patches as measured (not committed)

<!-- PATCH_TEXT -->

## 6. Reproduction

Per-run harness, logs and exit codes are in the task's isolated device workspace
(host-local; machine binding per `machine_specs.md`). Lost profiler captures are
retried and recorded; only captures with a complete `kernel_details.csv`
containing the expected native entries are kept.
