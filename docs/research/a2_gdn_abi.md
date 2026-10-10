# A2-07 — GDN ABI plan for Qwen3-Next on A2 (research + design, no kernel)

- Wave / SoC: W0 / a2 (host-side; desktop research + real-machine probes + A2-01's hit-table)
- Write set: `docs/research/a2_gdn_abi.md`, `benchmarks/a2/probe_gdn_abi.py`,
  `benchmarks/a2/_a2_exp_kernel.py`, `benchmarks/a2/evidence/gdn_abi/**`.
  (`_a2_exp_kernel.py` is a throwaway measurement kernel under `benchmarks/`, not an
  `ascend_fla/**` operator.) **No `ascend_fla/**` and no shipped kernel changed.**
- Evidence (this task, real 910B3): `benchmarks/a2/probe_gdn_abi.py` →
  `benchmarks/a2/evidence/gdn_abi/{probe.json, probe_stdout.log}` — probe.json carries the CANN
  compiler/opp `version.info` raw text + sha256 and the opp built-in kernel dir listing (`ascend910b`,
  `ascend910_93`); probe_stdout.log is the unfiltered run output (`nl -ba` readable). Also A2-01
  (#29, PR #107, merged) `benchmarks/a2/README.md` +
  `benchmarks/a2/evidence/host/hit_table.json`; `docs/matrix/gaps.json`
  (`gdn-no-gqa`, `a2-splitk-fp32-cube`, `a2-splitk-bf16-fp16-unsettled`, `block-dim-ceiling`);
  current main GDN units `ascend_fla/ops/gdn_chunk_{fwd,bwd}.py`,
  `kernels/projects/a5/gdn_chunk_{fwd,fwd_bf16,bwd,bwd_bf16}`; `fla/ops/gated_delta_rule/gate.py`,
  `fla/layers/gated_deltanet.py`; `AGENTS.md` §2/§6/§7.
- **All A2 real-machine numbers below are observation records only until A2-11**
  (`AGENTS.md` §6 / D-PM-30); they do not constitute an operator conclusion or lift the A2 gate.
  Real-machine environment for this task's probe: **Ascend 910B3, CANN 9.2.0-beta.1, torch 2.10.0 /
  torch_npu 2.10.0.post2, ascriptor library `90cfcdc` / kernels `b3b3f9c`** (A2-01's own numbers were
  taken under CANN 9.0.0 / opp `ascend910b`, `ascend910_93` — cited as its environment, not mine) —
  not extrapolated to A3/A5.

## 0. Target and framing

Qwen3-Next-80B-A3B (`docs/matrix/models.json`) runs GDN. Its mixer: hidden 2048,
**H=16 key heads, HV=32 value heads** (`HV/H=2`, GVA), head_k=head_v=128, conv 4, 48 layers
(36 GDN + 12 full-attn), **bf16**. `head_k=head_v=128` and bf16 match the fixed-size ABI; the
blocker is the head grouping (`gdn-no-gqa`).

**What already exists on main (not "redo from upstream").** Since 2026-09-19 the repo has
repository-owned GDN chunk units and public entries — GDA-01/02 (fwd + GQA/GVA with in-kernel
head indexing) / 03 (grouped bwd), BF-01/02 (BF16-native fwd/bwd): `ascend_fla/ops/gdn_chunk_{fwd,bwd}.py`,
`kernels/projects/a5/gdn_chunk_{fwd,fwd_bf16,bwd,bwd_bf16}`, plus the research notes
`docs/research/gdn_chunk_{fwd,bwd}_{bf16,gate_range}.md`. So the six ABI gaps below are scored
against **these a5 derived units** (what's done — file:line; what's missing to move to a2; what's
genuinely absent), and the batch proposal (§5) is "**move the derived units to a2 + close the
remaining gaps**", not a from-scratch rebuild. (The six gaps in `gaps.json` are recorded against the
*upstream* `a5.gdn_fwd/bwd` — see the 2026-09-18 note on `gdn-no-gqa`: GDA-02's GQA/GVA unblocks
PGDN, it does **not** by itself close `gdn-no-gqa`, which is about the upstream unit blocking
Qwen3-Next.)

The a5 `gdn_fwd` five stages map to fla (`AGENTS.md` §2, cross-checked in `fla/ops/gated_delta_rule`):
preprocess (g cumsum / decay) ↔ `gate.py` (`b_o = cumsum(-exp(A_log)*softplus(g+dt_bias))`);
inverse (strict-lower-triangular solve) ↔ `ops/utils/solve_tril.py`; recompute (WY) ↔ `wy_fast.py`;
scores ↔ `chunk_o.py`; recurrent ↔ `chunk_delta_h.py`; `gdn_bwd` adds the reverse saved values.

## 1. The six ABI gaps, scored against the current a5 GDN units

Per gap: **現状 (main, file:line)** → **fix (KDA / A2-01 precedent)** → **kernels/contract** →
**numerical risk** → **fp32 dual-oracle validation** → **effort**. Dual oracle = fla `naive`/
`naive_recurrent` vs `chunk`, both fp32, + the per-token↔chunk cross-check; correctness judged in
fp32 by relative-L2 / max_abs_diff (`AGENTS.md` §6), never bitwise, never bf16-elementwise.

### 1.1 gdn-no-gqa — no independent value-head dimension (the blocker)

- **現状**: `gdn-no-gqa` is recorded against the *upstream* `a5.gdn_fwd/bwd` (`contract.json`
  `domain.shape` "B,H,C…", no HV). On main, **GDA-02 already added GQA/GVA to the derived
  `gdn_chunk_fwd`** with in-kernel head indexing (to unblock PGDN), so the mechanism exists in the
  repo. The gap for Qwen3-Next is that the *value-head axis + `HV%H` mapping* must be present in the
  a2 GDN path end-to-end (fwd + bwd + decode).
- **Fix (KDA/GDA-02)**: reuse GDA-02's in-kernel head indexing / KDA's `i_h = i_hv // (HV//H)`;
  broadcast qk-side (q,k,g,b) across the `HV/H=2` group, carry HV on v/state/output.
- **Kernels/contract**: the a2 derived `gdn_chunk_{fwd,bwd}` gain the HV axis on v/state/output/saved
  histories; qk-side stays H. `domain.shape` gains `HV: "positive; HV % H == 0"`.
- **Numerical risk**: none new (broadcast is exact).
- **Validation**: an asymmetric `HV≠H` case (e.g. H=2, HV=4) with per-group-distinct values so a
  wrong `i_h` cannot pass; fp32 dual oracle on o + final_state.
- **Effort**: M (mostly index/broadcast plumbing; GDA-02 is the template).

### 1.2 layout-not-token-major — public layout `[B,H,C,L,D]`

- **現状**: upstream `gdn_fwd` layout is `[B,H,C,L,D]`; check whether the merged
  `ascend_fla/ops/gdn_chunk_fwd.py` already permutes internally (KDA does: token-major public,
  internal BHCLK). If it does, this gap is "done for the derived unit, verify the adapter"; if not,
  add the internal permute.
- **Fix (KDA)**: keep public token-major, permute inside the launcher.
- **Numerical risk**: none (copy); ensure the permute is a real materialization, not a strided view
  into a kernel needing contiguous (see `npu-builtin-ops-missing`: strided Slice availability).
- **Validation**: non-square `[B,T,H,D]` round-trip vs token-major reference; assert contiguity at
  the kernel boundary. **Effort**: S.

### 1.3 nonzero-initial-state — only zero initial state

- **現状**: upstream `gdn_fwd` `initial_state: "zero only"`. KDA takes fp32 `[B,HV,128,128]`.
- **Fix (KDA)**: add the fp32 nonzero initial_state input as KDA does; until the a2 derived unit
  supports it, the layer gate rejects a nonzero initial_state for GDN (`AGENTS.md` §7).
- **Numerical risk**: couples with §1.5 — a fed-back bf16 state loses precision per segment; fix
  §1.5 together.
- **Validation**: two-segment chaining vs single call; fp32 relative-L2. **Effort**: S–M.

### 1.4 d-initial-state-absent — backward emits no `dh0`

- **現状**: upstream `gdn_bwd` — "no d_initial_state". KDA `kda_bwd` emits `dh0 [B,HV,128,128]` and
  takes `dht`.
- **Fix (KDA)**: emit `dh0` from the reverse recurrence and accept `dht`, as `kda_bwd` does.
- **Numerical risk**: none new; the reverse recurrence already carries the state cotangent to step 0.
- **Validation**: autograd of an fp32 reference recurrence with a `requires_grad` initial state vs
  the kernel `dh0`; relative-L2 in fp32. **Effort**: S.

### 1.5 state-dtype-bf16 — upstream `final_state` is BF16, fla convention FP32

- **現状**: upstream `gdn_fwd` `final_state: bfloat16`; KDA's is fp32. The state is a cross-chunk
  accumulator; `AGENTS.md` and the PROTOCOL §6 Gemini-corrections both fix state at fp32.
- **Fix (KDA)**: make `final_state`/`state_after_history` fp32 in the a2 derived unit.
- **Numerical risk**: this *is* the risk. Magnitude of the bf16→fp32 improvement is **unmeasured on
  a2** — do not assume from A5. Measure o/final_state divergence between a bf16-state and fp32-state
  build at long C (e.g. C=64) and its growth with span (A2-11).
- **Validation**: same-shape bf16-state vs fp32-state build; fp32 recurrent oracle. **Effort**: S + measurement.

### 1.6 fused-recurrent-missing — no GDN decode kernel

- **現状**: `gaps.json` — GDN/DeltaNet decode is absent (**GDA-04 still `open`** per the PM). KDA's
  decode is done (`kernels/projects/a5/kda_fused_recurrent`, one-core-per-head, two-pass scan, no
  inter-core sync). GDN decode must be authored.
- **Fix (KDA)**: author a GDN `fused_recurrent` copying `kda_fused_recurrent`, extended for GVA
  (§1.1), fp32 state (§1.3/§1.5), and the per-token `exp(g_i)` inside the scan. Decode applies only
  `exp(g_i)` (magnitude ~1, no cumsum), so it **has no gate-span ceiling** (why it is a separate path).
  Hit-table (§2): the KDA decode kernel is a **MISS (no cube matmul)** — decode is pure-vector, so the
  M10-081 split-K issue does not touch it.
- **Numerical risk**: low. Invariant: per-token↔chunk identity (T single steps with chained state
  bit-identical to one T-token call).
- **Validation**: fp32 recurrent oracle at 8 shapes + the per-token↔chunk identity (relative-L2=0).
  `block_dim` sweep — but its ceiling and multi-core behaviour on a2 are **untested (A2-10)**; do not
  assert a number (see §2/§6 on cores). **Effort**: L; the KDA unit is the structural template.

### 1.7 Coupled: scale-param-no-slot, no-tail-path

- **scale-param-no-slot**: upstream GDN has no `scale` scalar; KDA's score kernels carry `scale: f32`.
  Add the same to the a2 GDN unit and absorb it at q's read point. Effort S.
- **no-tail-path**: `L=64` fixed, `T` multiple of 64. Gate rejects non-multiples with the nearest
  legal T and charges host padding in the perf report; a real tail path is a later, larger kernel
  change, kept out of this batch. Both real target models tolerate host padding for now.

## 2. Split-K / L0C-settle exposure of the GDN kernels (from A2-01's real-machine hit-table)

A2-01 (#29, PR #107, merged; real 910B3) established the a2-specific defect **M10-081**: on c220,
two consecutive short MMADs writing the same L0C are not hardware-interlocked, so the second
(`is_init=False`) can read an unsettled accumulator — a **silent wrong result** visible only on
silicon (sim/pipesim are bitwise, 0 hazard). The ascriptor pin fixes it **only for A2-family +
both-FP32 split-K** (a `PIPE_M` barrier per MMAD, `desugar.py:411`); **hand-written FP32 accumulate
chains get only a lint trap, and BF16/FP16 are excluded by dtype**.

A2-01's real-machine result (block_dim=1, 5×, bitwise vs CPU float64):
**FP32 split-K all bitwise; BF16/FP16 split-K M16 all wrong (finite, no error, 1022–1024/1024 elements
off), M32 crashes the AI Core (507015), M64 correct.** Adding the same `PipeBarrier<PIPE_M>()` makes
M16/M32 bitwise, so the cause is the missing M settle, dtype-narrowed away.

**The GDN kernels in A2-01's 25-kernel hit-table** (`benchmarks/a2/README.md` §7,
`evidence/host/hit_table.json`):

| # | stage | kernel (file:line) | verdict |
|---|---|---|---|
| 17 | GDN fwd inverse | `gdn_fwd/kernels/inverse.py:306/312/317-318` | **HIT-B** FP32 hand-written chain, **M16**, NOT settled (a2 lint trap) — the exact M10-081 failing shape |
| 25 | GDN bwd finalize | `gdn_bwd/kernels/finalize.py:205/217/226/235/244` | **HIT-B** FP32 chain, M=rows_cube (dynamic), NOT settled |
| 23 | GDN bwd wu | `gdn_bwd/kernels/wu.py:206` | **EXPOSED-C** BF16 chain, M64, outside the fp32 rule |
| 16,18,19,20,21,24,26 | GDN preprocess/recompute/scores/recurrent/scan_local/inverse_preprocess/recurrent_saved | MISS (each MMAD initialises its own L0C) |
| 22 | GDN bwd scan_state (`scan_state.py`) | MISS — has an accumulate MMAD, but the helper puts an explicit M/ALL barrier before **each** accumulate, so it is already settled |

So the a2-porting exposure for GDN is precise: **GDN inverse and GDN bwd finalize are FP32 M16
hand-written chains the pin's fix does not reach; GDN bwd wu is a BF16 M64 chain.** A2-01's probe
did not trip the hand-written FP32 chains at M16 on silicon (5/5 bitwise), **but timing-not-tripped
is not safety** (`AGENTS.md` §6): the BF16 split-K twin of the same construction *is* wrong, and real
kernels issue MMADs back-to-back with different pipelining than the probe.

**Fix (A2-01 §8 W2/W4, the recommendation for A2-03):** the a2 derived GDN units add an explicit
`barrier(Pipe.M)` after **every** L0C accumulate MMAD, regardless of dtype (this is exactly the
lint's fix), and keep the pin's FP32 split-K rule. **Qwen3-Next is BF16**, so the BF16 exposure is
directly relevant: the a2 GDN path must not use M<64 BF16/FP16 `splitk` until the library extends the
settle rule (that library change is `RISK`-reported to the ascriptor owner, A2-01 W5 — this repo does
not modify ascriptor); entries for shapes that cannot be proven safe error per `AGENTS.md` §7.
**A2-11 is where these are reproduced and the workaround verified on silicon.**

## 3. Range table — every exp / division / cumsum in the GDN chain, with real a2 exp lines

fp32/bf16 `exp` over/under-flow lines **measured on this 910B3** (`probe.json`, observation record):

| dtype | exp(−87.3) | exp(−88) | exp(−88.7) | first →0 | exp(88.7) | first →inf |
|---|---|---|---|---|---|---|
| fp32 | 1.22e-38 | **6.05e-39 (subnormal, kept)** | 3.01e-39 | **x≈−104** | 3.33e38 | 89 |
| bf16 | 1.00e-38 | 6.06e-39 | 3.67e-39 | **x≈−95** | 2.72e38 | 89 |

**Novel a2 finding: a2 does NOT flush fp32 subnormals** — `exp(−88)` returns `6.05e-39` (a subnormal),
not 0, so fp32 decay underflows to 0 only at span ≈ **104**, not the A5 line ~87.3 (A5 flushes
subnormals). bf16 underflows to 0 at ≈95. **Do not carry A5's −87.3 line to a2** (`AGENTS.md` §6).
The same measured on the **ascriptor-compiled CCE vector path** (`benchmarks/a2/_a2_exp_kernel.py`, the
path a real GDN decay kernel runs, not torch_npu's aclnn built-in) is **byte-identical**:
`exp(−87.3)=1.2192433092680476e-38`, `exp(−88)=6.054602886494416e-39`, underflow→0 at −104,
overflow→inf at 89 (`probe.json` `ascriptor_fp32_exp`). So the subnormal-preservation holds on the
generated vector code, which is the version that decides the gate-decay behaviour.

**But subnormal-preservation does NOT extend the usable gate range.** The binding limit is still
≈88.7, not 104, because `eg=exp(−span)` becomes subnormal well before it reaches 0, and the recurrence
*divides* by it. Measured `1/x` for subnormal x (`probe.json` `fp32_reciprocal_subnormal`):
`1/exp(−88)=1/6.05e-39=1.65e38` (finite) but **`1/exp(−89)=1/2.23e-39=inf`** — so `1/eg` / `k/eg`
(|k|~1) overflow to inf at span≈89, and `exp(+span)` overflows at 89 too. So a2 keeping fp32
subnormals only means the decay *value* survives deeper (to 104); the *recurrence* still goes
non-finite at span≈88.7 — the same practical line as A5, reached by overflow of `1/eg` rather than by
`eg` flushing to 0. **Caveat**: this `1/x` was measured with torch_npu's device reciprocal; the
subnormal-denominator behaviour of the ascriptor vector unit's own `div`/`reciprocal` is **not tested**
here and is an A2-11 item before it is a conclusion.

| # | op | where | a2 range verdict |
|---|---|---|---|
| 1 | `A_log.exp()` | gate | `exp(A_log)∈[0,16]` (init) — safe |
| 2 | `softplus(g+dt_bias)` | gate | ≥0 — safe |
| 3 | `-exp(A_log)*softplus` = per-token g | gate | ≤0 by construction |
| 4 | `cumsum(g)` over a 64-chunk | preprocess | `∈[−span,0]`, span a random variable, init upper bound ≈16·0.1·63 ≈ **100.8** |
| 5 | `exp(cumsum)` decay VALUE | preprocess | on a2 the value underflows to 0 only at span≈104 (fp32) / 95 (bf16) — a2 keeps subnormals — **but it is already subnormal (<2.94e-39) past span 88.7**; the *value* surviving deeper is not the binding limit (see #7) |
| 6 | `exp(g_i−g_j)` pairwise | scores/WY/bwd | `+span` side overflows fp32/bf16 past **89** (measured) — a binding limit (backward-finalize overflow) |
| 7 | `k/eg`, `eg_last/eg` division | intra/WY | **binding**: `eg=exp(−span)` is subnormal past span 88.7, so `1/eg` / `k/eg` with `|k|~1` **overflow to inf at span ≳ 88.7** — *before* `eg` reaches 0 at 104. See the measured `1/x`-for-subnormal-x line (`probe.json` `fp32_reciprocal_subnormal`) |
| 8 | `exp(g_i)` per-token (decode) | fused_recurrent | ~1 — safe, no ceiling |
| 9 | RMSNorm `rsqrt(mean x²+eps)` | o_norm / q,k l2norm | eps-guarded — safe |

**GDN gate span, estimated from Qwen3-Next's own init (not KDA's 94).** GatedDeltaNet
(`fla/layers/gated_deltanet.py:151-165`) inits `exp(A_log)=U(0,16)`, `dt∈[0.001,0.1]`,
`dt_bias=inv_softplus(dt)`; per-token `−g≤1.6`, 63-step chunk **upper bound ≈100.8** — same
structural ceiling as KDA but a distinct distribution (KDA draws `U(1,16)`), and with HV=32 value
heads the span sits near the ceiling at init. Qwen3-Next is *trained*, so `A_log`/`dt` move
(a trained GDN-family probe hit a 64-token span of 1461); the natural-text span is an A2-10/A2-11
measurement. **Consequence**: GDN still needs the same midpoint-anchored symmetric decomposition KDA's
stable units use. The binding limit on a2 is **≈88.7–89** — set by `exp(+span)` overflow (89) and
`1/eg` overflow (`1/exp(−89)=inf`), *not* by the `eg→0` line at 104. a2 preserving fp32 subnormals does
**not** buy extra forward headroom (the recurrence divides by the subnormal `eg` and overflows); it is
the same practical line as A5, reached by a different mechanism. The `MAX_GATE_SPAN` gate is
two-dimensional `{impl:{forward,backward}}`, must cover the init ceiling 100.8, and its **backward
accuracy ceiling must be re-measured for GDN on a2** (A2-11), not copied from KDA's 105.

## 4. torch_npu op coverage for the rest of Qwen3-Next — measured on this 910B3

`probe.json` ran each op below in fp32 **and** bf16 on the 910B3, **on tiny shapes only** (e.g.
`[4,8]`, `conv [1,8,16]`, `sdpa [1,2,8,16]`) — the probed set is present and finite (observation record
until A2-11); this is availability, not coverage:

| layer | op (probed, tiny shapes) | fp32 | bf16 |
|---|---|---|---|
| GDN mixer non-kernel | silu, sigmoid, softplus, l2_normalize, cumsum | ok | ok |
| short conv | depthwise conv1d | ok | ok |
| full attn | scaled_dot_product_attention, softmax | ok | ok |
| MoE | topk, bmm, softmax | ok | ok |
| norms | layer_norm, **rms_norm** | ok | ok |
| I/O | embedding | ok | ok |

**Not probed here — marked 未测 (A2-10)**: RoPE / rotary, MoE expert dispatch `scatter`/`gather`,
`argmax` / sampling, grouped / batched MoE matmul at real expert shapes, flash-attn-style fused
attention, and the `FusedRMSNormGated` sigmoid-gated **custom** path (only plain `F.rms_norm` was
probed, not the gated fusion). "What breaks if missing" per §0's checklist. Full end-to-end coverage
at real shapes is A2-10. Raw per-op results in `benchmarks/a2/evidence/gdn_abi/probe.json`.

## 5. Proposed kernel batch decomposition (A2-03 / A2-K1 "split-K workaround" + GDN move-to-a2)

Framed as **moving the existing a5 derived units to a2 + closing the remaining gaps**, not a rebuild.

**Batch A: chunk ABI + L0C settle (A2-03 / A2-K1).** Move `gdn_chunk_{fwd,bwd}` to a2; §1.1 HV/GQA
(GDA-02 template), §1.2 token-major, §1.3 fp32 nonzero state, §1.4 `dh0`, §1.5 fp32 final_state,
§1.7 scale. **Plus the A2-01 W2/W4 settle**: explicit `barrier(Pipe.M)` after every L0C accumulate
MMAD (GDN inverse `inverse.py`, GDN bwd finalize `finalize.py`, GDN bwd wu — dtype-agnostic), keep the
pin FP32 rule, and reject M<64 BF16/FP16 splitk per `AGENTS.md` §7. Acceptance: fp32 dual oracle on
o/final_state/grads at an asymmetric HV≠H case + segment-chaining + `dh0`; lint 0 trap; on a2, the
settle-barrier count equals the accumulate-MMAD count; A2-11 reproduces the M16 BF16 failure and
verifies the barrier fix. Risk: M (plumbing + the mechanical settle; the numerics are unchanged).

**Batch B: gate stabilization.** Midpoint-anchored decomposition for GDN fwd/bwd + a 2-D
`MAX_GATE_SPAN{forward,backward}` sized to cover the 100.8 init ceiling; thresholds re-measured for
GDN on a2 (fwd underflow now ~104 fp32 / 95 bf16, bwd overflow at 89). Depends on A. Risk: M–H.

**Batch C: decode kernel (GDA-04, still open).** New GDN `fused_recurrent` (KDA template; pure-vector
→ MISS in the hit-table, no split-K exposure), GVA, fp32 state, per-token `exp(g_i)`, no gate ceiling.
Depends on A. Risk: L–M.

Sequencing: A first, then B and C in parallel. Non-kernel host wiring / torch_npu coverage / MoE /
full-attn are A2-10 and later.

## 6. Corrections and honest gaps

- **Cores / block_dim (corrected)**: a2 is **20 cube / 40 vector cores** (A2-01 §3 + this task's
  `probe.json`: `cube_core_num=20, vector_core_num=40`). For the cube-bearing GDN chunk kernels,
  `GetVecNum() == 2 × block_dim` (`AGENTS.md` §6), so block_dim maps to the **20 cube (AI) cores**,
  each with 2 vector subblocks — not "40 vector cores". The `block_dim` ceiling and multi-core
  behaviour on a2 are **untested** (A2-10); this doc asserts no block_dim number.
- **Cross-SoC rule**: `AGENTS.md` **§6** ("conclusions do not inherit across SoC"), not §8. KDA's A5
  silicon numbers (e.g. `9.6e-8–1.8e-7`) are **A5** and are labelled as such; not carried to a2.
- **Gemini table**: §1.5's state-fp32 point rests on `AGENTS.md` and `docs/pm/PROTOCOL.md` §6
  (Gemini docs are requirement-source only), not on any "corrected Gemini table".
- **A2-01 is done** (#107 merged), not "still open"; §2/§3/§6 are rebuilt on its real hit-table.
- **No repo evidence for a separate a2 GDN-2 backward** was cited — the earlier draft's reference was
  removed; `kernels/projects/a2/` currently holds only `kda_*`. The `dh0` precedent used here is
  KDA's `kda_bwd`.
- Trained-checkpoint gate span, bf16-vs-fp32 state divergence, block_dim ceiling, and the M16 BF16
  reproduction are **A2-10/A2-11 measurements**; this doc flags each rather than asserting it.

## 7. 实现结果 (A2-K1, board-validated 2026-10-07 on 910B3)

Env identity (captured under `kernels/projects/a2/gdn_chunk_fwd/evidence/`): SoC **Ascend910B3**,
**CANN 9.2.0-beta.1**, **opp 9.2.0-beta.1** (ts 20260805_224858134), **npu-smi 25.5.1**. The verbatim
env captures are `evidence/opp_version.info` (the opp package manifest) and `evidence/npu_smi.log`
(SoC + driver), summarized in `evidence/env.json`. Raw per-case acceptance numbers are in
`evidence/acceptance_numbers.log`, the block-dim note in `evidence/blockdim_sweep.log`.

The formal ABI contract for each unit is `kernels/projects/a2/gdn_chunk_{fwd,bwd}/contract.json`.
The runnable `unit.py`/`run.py`/`_unit_runner.py` unit-protocol harness is **deferred to a follow-up
task with PM approval (PR #138 REVIEW, 2026-10-07)**: `tests/test_a2_gdn_chunk.py` already validates
both units end-to-end on 910B3 (dual fp32 oracle + autograd), so the harness port is mechanical, not a
correctness gap. Until that harness lands, these two units are **not** marked `qualified` in the full
repo unit-protocol sense.

Both units build and run on a2, validated by `tests/test_a2_gdn_chunk.py` against a dual fp32 CPU
oracle (block-solve + head-local grouped recurrence) and autograd. Chains:
`prepare→scores→wy→scan→output` (fwd) and
`replay→reverse_rec→reverse_dq/dbeta/dg/dk_bz/dk_ddr→group_reduce` (bwd), at HV≠H (H=2, HV=4):

- o / final_state rel-L2 **2.8e-7 – 4.2e-7** vs both oracles (zero and nonzero initial_state);
- grads dq/dk/dv/dbeta/dg/dh0 **1.56e-7 – 5.15e-7** vs autograd, with zero and nonzero `dht`;
- two-segment chaining (seg-1 `final_state` → seg-2 `initial_state`) matches a single-pass oracle to
  **rel-L2 < 1e-4** (the test's `_TOL`; the earlier "bit-exact" wording was inaccurate — it is a
  tolerance check, not bitwise); `T % 64 != 0` is rejected (no tail path, §1.7 — see the oracle/checklist
  notes below for exactly where that check does and does not live).
- **§1.5 bf16-state divergence (produced, not deferred):** casting the carried state to bf16 between
  chunks and comparing `final_state` to the fp32 recurrent oracle gives relative-L2 **1.662e-3** at the
  tested span. This is the evidence item §1.5 asks for; it is not a shipped option — per the PM NOTE
  (2026-09-25, DESIGN.md) the shipped default is fp32-state. There is no bf16 host op, so the §4
  bf16-host-audit remains N/A.
- All six ABI gaps (§1.1–1.7) exercised. **Correction on the accumulate-barrier guard (the earlier
  "non-vacuous negative control" claim was wrong).** `tests/test_a2_accumulate_barriers.py`'s detector
  (`_is_accumulate_matmul`) matches only `matmul(..., is_init=False)` calls, and A2-K1's shipped kernels
  contain none: the forward scan's two matmuls (`scan.py:103,116`) are single, non-accumulating products
  (each writes a fresh L0C tile that is then copied out through the FIX→GM ring), and the entire backward
  is pure vector (Deviation 2). So `test_every_accumulate_has_settle_barrier` matches nothing in these
  kernels and passes **vacuously** for A2-K1 — deleting the one `barrier(Pipe.M)` at `scan.py:115` does
  **not** turn it red (independently reproduced: 6 passed, 0 failed). That barrier orders the cube (M)
  pipe between the two independent matmuls (L0C-settle discipline), not an `is_init=False` accumulate
  chain. The guard's checker logic is still exercised by the test module's own synthetic positive/negative
  controls (`test_checker_flags_missing_barrier` etc.); it simply has **zero coverage of A2-K1's kernels**
  because every stage that would have carried an M10-081 accumulate hazard was architecturally bypassed
  (Deviation 3).

**block_dim:** validated at **`block_dim=40`** (the a2 core count used throughout bring-up). A bd1-vs-bd40
comparison is **not available**: on the shared 910B3 the bd=1 build faulted with a vector-core exception
(the box runs 8 cards at 90–160% AICore under other users' jobs; the longest kernels intermittently hit
`rtDeviceSynchronizeWithTimeout`), so only the verified bd=40 value is claimed. `evidence/blockdim_sweep.log`
records this. The dual-oracle correctness is block-dim-independent by construction (per-(b,hv) work split
across cores with no cross-core reduction), so bd40 vs bd1 is an identity, not a numeric risk — but it is
recorded as unverified rather than asserted (AGENTS.md §2: an A2 number is not a conclusion until measured).

**Deviation 1 — the forward scan's two matrix contractions run on the cube, not pure-vector.** §5 Batch A
planned `scan` as a pure-vector port of a5 `scan_vf`. That port compiles and is sim-correct, but its
step-1 `wy@S` (`_spread8` + strided `mul` + tree-reduce matvec) **faults on b3 at runtime** ("UB VEC
address out of bounds") for a kernel-context reason — the byte-identical matvec runs fine in the
backward `reverse_rec`, and UB size, `GROUP`, tile order, column offset, and explicit `count=` were all
ruled out on-device. The shipped `scan.py` therefore computes `wy@S` (contract D) and `knd^T@delta`
(contract C) as cube `matmul`s with the L0C→GM→UB GMBuff-ring handoff (the b3-proven idiom from the
chunk-parallel unit). `output` (stage 5) keeps the vector idiom (runs on b3 after an explicit-`count=`
fix). The §1 ABI contract and numerics are unchanged. A separate pre-fix `prepare`/`scores`/`output`
needed the same explicit-`count=` on vector ops (the prior "validated" was sim-only; sim does not model
b3's vector UB-address bounds).

**Deviation 2 — the entire backward is a pure-vector sequential recurrence, not the cube finalize/wu
§2 planned (the larger deviation).** §2 (A2-01 hit-table) and DESIGN.md "L0C settle" sited `barrier(Pipe.M)`
after the bwd **finalize** FP32 chain and the **wu** BF16 M64 MMAD — i.e. the planned backward carried
cube matmuls. The shipped backward does not: `replay→reverse_rec→reverse_dq/dbeta/dg/dk_bz/dk_ddr→
group_reduce` is a **sequential per-token recurrence with zero `matmul` and zero `barrier(Pipe.M)`** — it
is all full-`[128,128]` vector reductions per token, the sim-imposed 8-kernel decomposition, not the lean
cube structure §2 assumed. This is a bigger structural deviation than Deviation 1 (which touches two
contractions in one forward stage). Correctness is unaffected — the dual-oracle + autograd numbers above
are the gate — but the performance consequence is real and is not hidden:

- On 910B3 (s512 = B1·T512·H4·HV4, fp32, bd40) the shipped backward costs **~9.44 ms/call** device time
  vs fla's Triton `chunk_gated_delta_rule` backward **~0.92 ms/call** — about **10×** slower, and the gap
  grows linearly in T (the a2 cost is O(T); fla's is sub-linear).
- The slowdown is **algorithmic, not a tuning gap**: this backward is a sequential O(T) per-token
  recurrence, whereas fla's is **chunk-parallel** (O(T/C) matmul-style work). An in-scope experiment that
  folded the tapes into a single tapeless reverse scan reached a near-optimal 95.6% vector-bound kernel at
  ~5.39 ms/call (5.9×) — still algorithmically bounded. Closing the gap needs a **chunk-parallel cube
  re-derivation of the backward**, a design-level change constrained on b3 by A2-01 (no cube L0C DMAs);
  that chunk-parallel backward unit was prototyped, did not beat fla on b3, and is **shelved** (out of
  A2-K1 scope, which is correctness/ABI). The perf variants (v2/v3) and their informal bench were removed
  from the deliverable so the shipped backward is unambiguously the dual-oracle-validated v1 chain.

**Deviation 3 — the `inverse` stage (the task's motivating M10-081 example) was bypassed, not fixed.**
A2-01 hit-table #17 / the task spec's `inverse.py:306/312/317-318` is an FP32 hand-chained M16 L0C
accumulate — the canonical M10-081 case the `barrier(Pipe.M)` discipline exists for, and the opening
motivation for A2-K1. The a2 port does **not** reimplement it as a cube matmul + settle barrier:
`wy.py:70-77` solves the lower-triangular system by **row-sequential scalar forward substitution (zero
cube ops)**, sidestepping the hazard class rather than repairing it. Numerically clean (wy rel-L2 6.1e-8;
`DESIGN.md` describes it as a "row-sequential lower-triangular solve"), but — like Deviations 1 and 2 — it
is a structural departure from the planned cube-with-settle form, called out here so a reader asking "was
the inverse barrier fix done?" gets a direct answer: **the hazard was bypassed, not repaired.** Taken
together, Deviations 1–3 are why the accumulate-barrier guard has no A2-K1 coverage (above): every
cube-accumulate site the task anticipated was rewritten to vector / single-matmul / row-scalar form.

**Oracle & checklist notes (REVIEW 2026-10-08).**
- *Dual oracle.* `DESIGN.md:40` planned the fp32 dual oracle as "fla `naive`/`naive_recurrent` + repo
  CPU ref." The shipped `tests/test_a2_gdn_chunk.py` instead uses two independent in-repo torch references
  (`_grouped_recurrent`, a head-local grouped recurrence; `_block_solve`, the chunked block solve) plus
  autograd for the grads, and does **not** import `fla`. This is a deviation from the AGENTS.md §6 letter
  (one oracle being fla's `naive.py` itself); the two references are mutually independent formulations so
  the cross-check is non-trivial, but it is not the fla-naive oracle the plan named.
- *`T % 64` rejection.* The no-tail-path reject (§1.7) currently lives only in the test's host-side helper
  (`tests/test_a2_gdn_chunk.py`, `if T % C: raise ValueError`) and is declared in `contract.json`'s
  `domain`. It is **not** enforced inside the 13 kernel files or at an op-entry wrapper — code calling a
  compiled kernel directly, bypassing the test harness, gets no such guard. Runtime enforcement on a
  formal op entry is part of the PM-approved deferred `unit.py` harness.
- *Segment-chaining number.* The chaining case is judged by the same `_rel(...) < 1e-4` tolerance as the
  other cases (not a separate bitwise check); `evidence/acceptance_numbers.log` does not yet carry a
  per-case raw number for it (wording corrected above).
