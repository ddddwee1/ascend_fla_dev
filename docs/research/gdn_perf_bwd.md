# GDN-P1 · GDN chunk backward: real-hardware pipe utilisation

Task `GDN-P1`, A5 real hardware. Environment, oracles and measurement discipline
are the ones in `gdn_perf_decode.md` §1: one process per row, its own vendor tree
and compile cache, device lock held, `torch.npu.matmul.allow_hf32` pinned `False`
and recorded, `ProfilerLevel.Level1` with `aic_metrics=PipeUtilization`.

Unlike the forward, the FP32 backward **does** have its own kernel directory:
`ascend_fla/ops/gdn_chunk_bwd.py` dispatches FP32 to `_pipeline()` =
`kernels/projects/a5/gdn_chunk_bwd/kernels` and BF16 to a separate
`gdn_chunk_bwd_bf16`, and the public entry rejects BF16 outright (BF-02 /
D-PM-35). So the FP32 backward is entirely inside the GDN-P1 write set.

Shapes are `docs/matrix/models.json`: `qwen3_next_layer` (`B1/T1024/H16/HV32/K=V=128`)
and `long_context` (`B1/T4096/H16/HV32`), with both cotangents present
(`mode='both'`). Accuracy is judged in FP32 against the unit's independent
analytical CPU adjoint (`ref/reference.py:analytical`, no FLA import, no
autograd) at a 1e-4 relative-L2 budget, per gradient.

## 1. Baseline: where the time goes and what binds

`qwen3_next_layer` (T=1024), FP32, block_dim 16. Device kernel time is the median
of three profiler captures.

| native entry | device us | share | `aiv_vec_ratio` | `aiv_mte2_ratio` | `aiv_mte3_ratio` | `aiv_scalar_ratio` | `aic_mac_ratio` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gdn_bwd_reverse`       | 4752.320 | 76.75 % | 0.652 | 0.300 | 0.160 | 0.017 | 0.000 |
| `gdn_bwd_checkpoints`   | 1018.826 | 16.45 % | 0.624 | 0.354 | 0.012 | 0.019 | 0.000 |
| `gdn_bwd_group_reduce`  |  422.030 |  6.82 % | 0.100 | **0.889** | 0.246 | 0.109 | 0.000 |
| total | 6192.010 | | | | | | |

Per-gradient relative L2 against the CPU analytical adjoint, identical at every
block_dim: `dq` 1.23e-07, `dk` 1.18e-07, `dv` 2.24e-07, `dg` 2.45e-07,
`dbeta` 2.657e-07. Budget 1e-4, untouched.

`aic_mac_ratio = 0.000` and `cube_utilization ≤ 0.35 %` in all three launches:
the backward, like the forward and the decode, contains no `matmul` — `primal`
and `adjoint` are `@vf()` loops of element-wise operations and scalar broadcasts.
**Task item 4 is answered negatively here too: the repository-level P0
`a5-fp32-cube-matmul-inherits-hf32` has no reachable site in the GDN backward.**

## 2. Verdict against the 85 % line

This is the one GDN unit that **does not** reach the line, and the gap is in the
stage that matters.

- **`gdn_bwd_reverse`, 76.75 % of the time, reaches no single pipe's line:**
  `aiv_vec_ratio = 0.652`, `aiv_mte2_ratio = 0.300`, `aiv_mte3_ratio = 0.160`.
  Honest gap: **19.8 percentage points short of 85 % on the vector pipe**, which
  is the closest resource. Summed over pipes it is 1.112 (they overlap), so the
  core is not idle — it is split roughly two-to-one between arithmetic and data
  movement, and neither side dominates.
- **`gdn_bwd_checkpoints`, 16.45 %: also short** — vec 0.624, mte2 0.354.
  Gap 22.6 points on the vector pipe.
- **`gdn_bwd_group_reduce`, 6.82 %: `aiv_mte2_ratio = 0.889` ≥ 0.85, so this one
  is data-movement bound and at the line.** It is a pure gather-and-sum of
  `HV/H` consecutive value-head contributions into the key-head gradients, so
  being MTE2-bound is the correct terminal state for it.

Weighted over the three launches, 6.82 % of the backward is at the line and
93.18 % is not.

### Identified mechanism for the gap (source + counters, not symptom-fitting)

`gdn_bwd_reverse` replays each 64-token chunk forward from its checkpoint while
**writing every intermediate state to a GM tape**, then walks the chunk backwards
reading the tape:

```python
tape: GM[f32, ("B","HV",64,128,128)]
...
for token in range(C):
    tape[bb,hh,token,:,:] <<= state[:,:]          # 64 KiB per token
    ...
    primal(state,ku,vu,gu,bu)
ready.set(); ready.wait()
for reverse_token in range(C):
    state[:,:] <<= tape[bb,hh,token,:,:]          # 64 KiB per token
```

That is 64 KiB written plus 64 KiB read per token: 4 MiB + 4 MiB per chunk per
`(batch, value head)`. At `B1/T1024/H16/HV32` — 16 chunks, 32 owners — the tape
alone moves ~4 GiB of GM traffic per call, against ~2 MiB of q/k/v/do/gradient
traffic per chunk-owner. This is why `reverse` shows `mte2 0.300 / mte3 0.160`
while the *forward's* comparable stages show 0.02–0.04: the forward has no tape.

`gdn_bwd_checkpoints` has a second, independent reason for `mte2 0.354`: it reads
the inputs **one token at a time** —
`ku[:,:] <<= k[bb,tt,kh:kh+1,:]` is a 512-byte transfer, four of them per token,
256 small MTE2 transactions per chunk. The forward's `prepare` loads a whole
64-token tile in one `gm_to_ub_pad` instead.

Neither observation is acted on in this task; both are candidate changes with
their trade-offs stated in §4, because each needs its own measured verification
cycle and §6.5 forbids making several kernel changes at once.

## 3. Change A: block_dim, and where the backward's scaling stops

`ascend_fla/ops/gdn_chunk_bwd.py` accepted only `block_dim in (1,2)`, i.e. at
most 4 of the 56 vector cores. Change A widens the grid to `(1,2,4,8,16,28)`;
28 is the ceiling because a block_dim above the physical core count deadlocks on
the hardware barrier. No kernel source is touched.

How each launch partitions, and therefore where it saturates:

| launch | partitioned over | items at B1/T1024/H16/HV32 | saturates at |
|---|---|---:|---|
| `checkpoints` | `B*HV` | 32 | `GetVecNum() = 32`, i.e. **bd16** |
| `reverse`     | `B*HV` | 32 | `GetVecNum() = 32`, i.e. **bd16** |
| `group_reduce`| `B*T*H` | 16384 | beyond bd28 |

Measured, T=1024, FP32, device kernel total (median of 3 captures):

| block_dim | total us (change A build) | `checkpoints` | `reverse` | `group_reduce` | vs bd1 | vs bd2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 91896.419 | — | — | — | 1.00x | 0.50x |
| 2  | 45876.280 | — | — | — | 2.00x | 1.00x |
| 4  | 23138.396 | — | — | — | 3.97x | 1.98x |
| 8  | 11844.123 | — | — | — | 7.76x | 3.87x |
| 16 |  6192.010 | 1018.826 | 4752.320 | 422.030 | **14.84x** | **7.41x** |
| 28 |  6131.463 | 1019.230 | 4864.412 | 247.024 | 14.99x | 7.48x |

T=4096, FP32: bd2 183111.737 us → bd16 24527.631 us, **7.47x**.

**Control — the ops-layer change is inert on the old grid.** Same card, baseline
build vs change-A build:

| block_dim | base build | change A | drift |
|---:|---:|---:|---:|
| 1 | 91813.627 us | 91896.419 us | +0.090 % |
| 2 | 45777.156 us | 45876.280 us | +0.217 % |

Both inside run-to-run noise. The per-stage split for bd1 on the baseline build
is `checkpoints` 15944.312 / `reverse` 69505.197 / `group_reduce` 6374.338 us,
with vec/mte2 ratios 0.636/0.346, 0.708/0.253 and 0.104/0.884 respectively — the
same balance as bd16, so the verdict in §2 is not an artefact of one block_dim.

The bd16 → bd28 rows confirm the prediction made from the source *before* the
bd28 run: `checkpoints` is flat (1018.826 → 1019.230 us) and `reverse` is
slightly **worse** (4752.320 → 4864.412 us, +2.4 %), because both already own one
item per core at bd16 and bd28 only adds 24 cores with `begin == end` plus their
launch cost. Only `group_reduce` keeps scaling (422.030 → 247.024 us). Net total
6192.010 → 6131.463 us, a 1.0 % gain that is not worth the extra build.

**For the backward at `B*HV = 32`, bd16 is the operating point**, not bd28.

### Uninstrumented same-card sandwich

Profiler off, three processes back to back on one card, 3 rounds x (10 warmup +
50 timed samples) each, `perf_counter_ns` around the public call plus the
completion wait. `qwen3_next_layer`, FP32.

| order | build | block_dim | round medians (ms) | median |
|---|---|---:|---|---:|
| A | base | 2 | 45.8563 / 45.8260 / 45.8427 | 45.8427 |
| B | change A | 16 | 6.2435 / 6.2521 / 6.2677 | **6.2521** |
| C | base | 2 | 45.7158 / 45.6680 / 45.6738 | 45.6738 |

**7.33x** host-inclusive, with the repeated baseline drifting −0.368 % between A
and C. All three processes produced the same five gradient SHA-256 values
(`dq = c96ffc390e33…`).

### Correctness and bitwise neutrality

- Per-gradient relative L2 is **identical at every block_dim**, as listed in §1.
- **The five returned gradients' SHA-256 form a single equivalence class across
  both builds and all measured block_dims** for each shape
  (`dq = c96ffc390e33…`, `dk = 40017a0fcfcf…`, `dv = 70f863435617…`,
  `dg = 5e55acd322a8…`, `dbeta = 9c7408b60cdb…` at T=1024). Core splitting does
  not change the arithmetic.

## 4. Candidate changes identified but not made

Each is bitwise-neutral or explicitly not, and each needs its own verification
cycle (CCE emission diff, simulator bitwise check, real-hardware double-oracle,
same-card sandwich). None was applied, because applying more than one at a time
is exactly what AGENTS.md §6.5 forbids and because none of them can be claimed
without its own measurement.

1. **Tile-load the inputs in `gdn_bwd_checkpoints`** (bitwise neutral). Replace
   the four 512-byte per-token reads with one `gm_to_ub_pad` per 64-token tile,
   as the forward's `prepare` already does, and index rows out of UB. UB budget
   fits: two 64x128 FP32 tiles (32 KiB each) plus the small gate rows plus the
   128x128 state (64 KiB) is ~130 KiB against the 256 KiB profile. Target is the
   `mte2 = 0.354` in a stage that is 16.45 % of the backward. Does **not** fit
   `gdn_bwd_reverse` as-is: that launch already holds `state` and `back`
   (128 KiB) and would need five tiles (160 KiB) on top.
2. **Halve the replay tape** (not bitwise neutral in cost, bitwise neutral in
   value): store every second token's state and recover the odd ones with one
   extra `primal` step during the reverse sweep. Tape traffic halves; `primal`
   work rises by 50 %. Since the mix is 0.652 vec / 0.300 mte2, this trades the
   resource that is *not* binding for the one that is — the wrong direction — so
   it is recorded as analysed-and-rejected rather than as a candidate.
3. **Hoist the two `SCALE` scalar multiplies out of `adjoint`'s second loop**
   (bitwise neutral: the same FP32 multiply, once per element either way).
   `qi <<= query[...]; qi <<= qi * SCALE` can pre-scale the whole `query` row
   once per token, and `scalar <<= scalar * SCALE` for `dq` can be deferred to
   one vector multiply over the finished `dq` row. Each turns 128 scalar
   operations per token into one vector operation. Expected effect is small:
   `aiv_scalar_ratio` is only 0.017 in `reverse`, so this bounds the gain at
   under 2 % of that stage.
4. **Reuse the forward's `states` output instead of recomputing checkpoints.**
   `chunk_gdn` already produces exactly the chunk-start states that
   `gdn_bwd_checkpoints` recomputes from scratch; accepting them would delete
   16.45 % of the backward outright. This is an **ABI change**, not an
   optimisation: it adds an input to the public `chunk_gdn_bwd` and gives up the
   unit's advertised "self-contained checkpoint recomputation" property, and it
   interacts with the existing `fwd-caches-not-emitted` gap. It needs an explicit
   decision, so it is reported, not taken (§7, §6.5).
5. **Port `primal`/`adjoint` to the cube engine.** The two state sweeps in
   `primal` and the three in `adjoint` are GEMV-shaped: 128 scalar-broadcast
   multiply-accumulates over 128-wide rows. `aic_mac_ratio = 0.000` means the
   cube is entirely unused. This is the same finding PK-07 made for PKDA, where a
   round-2 cube port of a single stage reached `aic_mac_ratio = 0.635` — a
   multi-session design task per stage, not a tuning step, and the only route to
   85 % on the arithmetic pipe for this unit.

## 5. Reproduction

Per-run harness, unfiltered logs and exit codes are in the task's isolated device
workspace (host-local; machine binding per `machine_specs.md`). The CPU
analytical adjoint is token-serial and is computed before the device call and
outside every timed region.
