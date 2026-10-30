# GDN-P1 · `gdn_fused_recurrent` (decode): real-hardware pipe utilisation

Task `GDN-P1`, A5 real hardware. This document records what binds the GDN decode
kernel, measured with AI Core performance counters enabled, and the verdict
against the 85 % pipe-utilisation acceptance line (AGENTS.md §6).

GDA-05 (`docs/research/gda05_profile.md`) already measured this family's device
and host wall times at block_dim 1/2/4/8/16/28. It did **not** measure pipe
utilisation: that report explicitly disabled the AI Core performance counters
("使用Level1并显式关闭AI Core性能计数器"), so it contains no `aiv_vec_ratio`,
`aic_mac_ratio`, `mte2` or `mte3` figures at all. The numbers below are therefore
new measurements, not a re-run of something already answered — and they double
as an independent reproduction of GDA-05's device times (see §4).

## 1. Environment

| item | value |
|---|---|
| SoC | A5 / Ascend950PR, 28 cube / 56 vector cores, `Health=OK` |
| CANN / opp | 9.1.0, `built-in/op_impl/ai_core/tbe/kernel/` carries `ascend950` |
| driver / npu-smi | 25.7.rc1 |
| Python / torch / torch_npu | 3.12.14 / 2.12.0+cu130 / 2.12.0 |
| ascriptor | 0.1.0, library pin `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5` |
| FLA oracle | pin `e52dbc0e…`, source SHA-256 `d1cf1799…` (identical to GDA-05's `FLA_GDN_NAIVE`) |
| repo revision | `758468cd` (baseline) |
| `torch.npu.matmul.allow_hf32` | pinned `False` in every process, recorded per run |
| profiler | `ProfilerLevel.Level1`, `aic_metrics=PipeUtilization`, explicit `wait=0/warmup=1/active=1/repeat=1` |
| device | one physically assigned card, held through an advisory `flock`; health and
  occupancy recorded before and after every run |

Shapes are the `qwen3_next_layer` head configuration from `docs/matrix/models.json`
(`B=1, H=16, HV=32, K=V=128`) at the two decode lengths the contract declares,
`S=16` and `S=1`, with a non-zero random initial state. Correctness is checked
against **two** independent references before each timed call: the unit's own CPU
FP32 recurrence and the pinned FLA `naive_recurrent_gated_delta_rule`.

## 2. Measured

Device kernel time is the median of three profiler captures; `gdn_recurrent_fp32`
is the only device kernel, so it is 100 % of the total.

### S = 16

| block_dim | device us | speedup vs bd1 | `aiv_vec_ratio` | `aiv_mte2_ratio` | `aiv_mte3_ratio` | `aiv_scalar_ratio` | `aic_mac_ratio` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 395.611 | 1.00x  | **0.907** | 0.056 | 0.034 | 0.007 | 0.000 |
| 2  | 198.481 | 1.99x  | **0.906** | 0.057 | 0.034 | 0.008 | 0.000 |
| 4  | 107.403 | 3.68x  | 0.839 | 0.096 | 0.058 | 0.010 | 0.000 |
| 8  |  54.542 | 7.25x  | 0.836 | 0.097 | 0.055 | 0.015 | 0.000 |
| 16 |  28.255 | **14.00x** | 0.819 | 0.108 | 0.052 | 0.025 | 0.000 |
| 28 |  28.805 | 13.73x | 0.798 | 0.110 | 0.049 | 0.047 | 0.000 |

bd16 is the optimum, not bd28: `B*HV = 32` work items split over
`GetVecNum() = 2*block_dim`, so bd16 gives exactly one item per core with none
idle, while bd28 leaves 24 of the 56 cores with `begin == end` and pays their
launch cost (`aiv_scalar_ratio` nearly doubles, 0.025 → 0.047).

### S = 1

| block_dim | device us | speedup vs bd1 | `aiv_vec_ratio` | `aiv_mte2_ratio` | `aiv_mte3_ratio` | `aiv_scalar_ratio` | `aic_mac_ratio` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 53.043 | 1.00x | 0.448 | 0.296 | 0.234 | 0.054 | 0.000 |
| 2  | 27.188 | 1.95x | — | — | — | — | 0.000 |
| 4  | 17.598 | 3.01x | — | — | — | — | 0.000 |
| 8  | 10.032 | 5.29x | — | — | — | — | 0.000 |
| 16 |  5.835 | **9.09x** | 0.290 | 0.356 | 0.245 | 0.133 | 0.000 |
| 28 |  6.114 | 8.68x | 0.254 | 0.332 | 0.208 | 0.225 | 0.000 |

Accuracy, every block_dim, unchanged across the whole sweep:

| reference | S=16 worst relative L2 | S=1 worst relative L2 |
|---|---|---|
| unit CPU FP32 recurrence | 1.362e-07 (`o`), 9.155e-08 (`final_state`) | 7.859e-08, 3.398e-08 |
| FLA naive (pinned) | 1.537e-07 (`o`), 9.705e-08 (`final_state`) | 9.027e-08, 3.672e-08 |

Budget is 1e-4. The two oracles differ from each other by ~1.2e-08, which is the
useful extra signal §6 asks for: the kernel sits between them, not outside.

**Cross-block_dim bitwise identity.** The public `o` / `final_state` SHA-256 is a
single value per shape across all six block_dims:
`o = ce4265e13dcd20104ae1869d6e3f1d6870bb09023722161108dfabfac98bb0b7`,
`final_state = 998630635aa1aca6e002e0235dfddfb40e815f72bb7c1c3d8f94f613d86449ef` (S=16);
`o = 5d298948d745dc589fe445e2724630633e79db01b7cf5a7f994c7f4c54053780` (S=1).
Core splitting does not change the arithmetic, which is the reference-free hard
criterion in §6.

## 3. Verdict against the 85 % line

- **`aic_mac_ratio = 0.000` and `cube_utilization ≈ 0 %` at every block_dim and
  both lengths.** `gdn_recurrent_{fp32,bf16}` are declared `@kernel(mode="vec")`
  and contain no `matmul` call, so this is structural, not a tuning artefact.
  **Consequence for the HF32 question (task item 4): the repository-level P0
  `a5-fp32-cube-matmul-inherits-hf32` cannot affect this unit — there is no FP32
  cube matmul to inherit a rounding mode.** The runs nevertheless pin
  `allow_hf32=False` and record it, so the numbers are not conditional on what a
  co-tenant left behind.
- **S=16 at block_dim 1–2: `aiv_vec_ratio` 0.907 / 0.906 ≥ 0.85 → the vector
  arithmetic pipe is the bottleneck and the kernel is at the acceptance line.**
- **S=16 at block_dim 4–28: 0.839 → 0.819 → 0.798, just below the line.** Nothing
  got worse; the per-core *work* shrank while the per-item fixed cost (a 64 KiB
  state load and a 64 KiB state store per value head) did not, so the mix shifts
  toward data movement. At bd16 the three pipes together are
  0.819 + 0.108 + 0.052 = 0.979.
- **S=1 misses the line on every pipe: 0.448 vec + 0.296 mte2 + 0.234 mte3 at bd1
  (0.978 summed), 0.290 + 0.356 + 0.245 at bd16, 0.254 + 0.332 + 0.208 at bd28.** This is arithmetic intensity,
  not a kernel defect. One decode token performs O(D²) work against an O(D²)
  state, i.e. ~1 arithmetic operation per state byte moved, so the kernel is
  state-bandwidth bound by construction. **No kernel-level change can lift any
  single pipe to 85 % at S=1**; only an algorithmic change that amortises the
  state transfer over more tokens (i.e. larger S, or batching value heads
  differently) would, and that is an ABI change, not an optimisation.

So: **S=16 passes at the shipped default region, S=1 cannot pass and the gap is
explained and bounded.** Honest statement of the gap: at S=1/bd16 (the fastest
point, 5.835 us) the best single pipe is `aiv_mte2_ratio = 0.356`, i.e. **49.4
percentage points below the line**, and the sum over pipes is 0.891. At S=16/bd16
the vector pipe is 0.819, **3.1 points below the line**; only bd1/bd2 clear it.

## 3b. Uninstrumented same-card sandwich, and why the 14x shrinks to 5.7x

Everything in §2 is profiler-instrumented *device* time. With the profiler off,
three processes back to back on one card, 3 rounds x (10 warmup + 50 timed
samples), `perf_counter_ns` around the public call plus the completion wait,
S=16, FP32:

| order | block_dim | round medians (ms) | median |
|---|---:|---|---:|
| A | 1  | 0.4554 / 0.4521 / 0.4495 | 0.4521 |
| B | 28 | 0.0815 / 0.0792 / 0.0790 | **0.0792** |
| C | 1  | 0.4516 / 0.4497 / 0.4480 | 0.4497 |

**5.71x host-inclusive**, baseline drift −0.53 %, identical output SHA-256 in all
three processes.

Device-only the same pair is 13.73x (395.611 → 28.805 us). The difference is
host call overhead: GDA-05 measured the public call's return at ~0.051 ms for
this family, and that does not shrink with block_dim. At bd28 the device work is
0.029 ms, so **roughly two thirds of a bd28 decode call is host time**.

This is the trap AGENTS.md §6 iron rule 1 describes — a device-side improvement
disappearing into host overhead — and it cuts both ways here. It does not
invalidate the device measurement, but it does mean the realistic benefit of
raising the decode default is ~5.7x end to end at S=16, not 14x, and that
reducing the per-call host cost is a comparably large lever at these sizes. Both
numbers are reported; neither alone is the answer.

## 4. Independent reproduction of GDA-05

Same family, same shapes, different session and card, two weeks apart. GDA-05's
`gdn_decode` `kernel 总和 ms` against this report's device kernel medians:

| case | GDA-05 | here | delta |
|---|---:|---:|---:|
| s16 bd1 | 0.394180 ms | 0.395611 ms | +0.36 % |
| s16 bd2 | 0.197493 ms | 0.198481 ms | +0.50 % |
| s16 bd4 | 0.105775 ms | 0.107403 ms | +1.54 % |
| s16 bd8 | 0.053522 ms | 0.054542 ms | +1.91 % |
| s16 bd16 | 0.027484 ms | 0.028255 ms | +2.81 % |
| s16 bd28 | 0.028336 ms | 0.028805 ms | +1.66 % |
| s1 bd1 | 0.053819 ms | 0.053043 ms | −1.44 % |
| s1 bd28 | 0.006000 ms | 0.006114 ms | +1.90 % |

Agreement within 2.9 % throughout, and within 0.5 % on the two slowest rows where
the relative resolution is best. GDA-05's numbers stand; this report adds the pipe
decomposition they deliberately left out.

## 5. What this leaves on the table

The only unexploited lever is **the default `block_dim` of the public entry**.
`ascend_fla/ops/gdn_fused_recurrent.py` defaults `prepare()` and
`fused_recurrent_gdn()` to `block_dim=1`, while the contract has qualified
1/2/4/8/16/28 since GDA-04 and this report confirms bitwise-identical output at
all of them. A caller that does not pass `block_dim` therefore runs the decode on
2 of 56 vector cores and pays **13.7x** (S=16) or **8.7x** (S=1) more device time
than the same build at bd16, for no accuracy difference whatsoever.

This is a behaviour change rather than a bug fix, so it is reported rather than
taken unilaterally, for two concrete reasons beyond caution:

1. **No single value is right for all shapes.** bd16 is optimal at `B*HV = 32`
   (28.255 us) and bd28 is worse there (28.805 us) because 24 cores idle, but
   bd28 wins once `B*HV > 32`. A shape-aware choice is not expressible in the
   current design: block_dim is a *build* parameter and one process may hold only
   one build per operator name (§6 iron rule 2).
2. **It interacts with the test session.** `tests/conftest.py:56` deliberately
   pins `prepare(block_dim=1, …)` with the comment that one operator name allows
   only one build per process. Changing a public default without auditing which
   `prepare()` the pytest session performs first risks the "already executed an
   aclnn operator" failure mode described in AGENTS.md §5, which cannot be
   verified without a full real-hardware test run.

The measured table above is what the decision needs; the decision itself belongs
to the PM.

## 6. Reproduction

Harness, environment script and per-run logs live in the task's isolated device
workspace (host-local, not committed; machine binding per `machine_specs.md`).
Each row above is one process with its own vendor tree and compile cache, run
under the device lock, with unfiltered logs and an explicit exit code. Lost
profiler captures — CANN intermittently fails to associate ACL and NPU flow
events for a short kernel and then writes no `kernel_details.csv`, the same
failure GDA-05 documented — are retried and recorded as `CAPTURE_LOST`; only
captures with a complete CSV and the expected native entry present are kept.
