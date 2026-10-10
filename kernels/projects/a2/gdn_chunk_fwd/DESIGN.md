# A2-K1 — GDN chunk fwd+bwd move-to-a2 (design / plan)

Batch A of `docs/research/a2_gdn_abi.md` §5. Move the a5 derived units
`kernels/projects/a5/gdn_chunk_{fwd,bwd}` to a2, close six ABI gaps, add the
L0C-settle barriers. Batch B (gate-span calibration) and Batch C (decode /
GDA-04) are explicitly out of scope.

## Key finding: the port is a tensor-vector rewrite, not a file-move

The a5 `gdn_chunk_{fwd,bwd}` stages are written in the **`@vf` register
value-function model** (`gdn_chunk_fwd/kernels/stages.py`: 6 `@vf()` stages, 16
`@vf`/`VfPipe`/`vf_barrier` lines; `gdn_chunk_bwd/kernels/stages.py`: 13). **a2
(c220) has no `@vf`** — verified: the shipped a2 units `kda_fwd_stable/kernels/*`
use **zero** `@vf` (they are tensor-vector rewrites), and the a2 `gdn2_recurrent`
port's own docstring records "@vf … c310-only; A2 has no @vf, so this rewrite uses
the tensor-vector free functions."

So Batch A carries the a5 **algorithm** unchanged but **reimplements each stage in
the a2 tensor-vector model**, following `kda_fwd_stable` / `kda_bwd_stable` as the
pattern (the same rewrite KDA and gdn2-recurrent already did). `a2_gdn_abi.md` §5
frames this as "move … not a rebuild" with risk "M (plumbing + mechanical settle)";
the numerics/ABI framing holds, but the vector-stage rewrite is more than plumbing,
so the **ETA revises up from 34h** (to be re-estimated after the first stage).
Raise to PM in the next STATUS.

## Port sources (verified present on main)

- `kernels/projects/a5/gdn_chunk_fwd/` — `kernels/{pipeline.py,stages.py}`, `contract.json`,
  `unit.py`, `ref/`. GDA-02 added the GQA/GVA in-kernel head indexing here.
- `kernels/projects/a5/gdn_chunk_bwd/` — reverse recurrence + saved histories.
- Public entries `ascend_fla/ops/gdn_chunk_{fwd,bwd}.py`.

## Iron rules that shape the port (AGENTS.md)

- **One operator name per process per build** (§6 rule 2): the new a2 units get
  **distinct suffixed operator names** — do not reuse the a5 unit names (BF-05
  hit a silent wrong-binary on exactly this).
- **C++-keyword-safe kernel param names**: no `do` / `in` / `class` (flagged 3×
  in the spec — GDA-03, PK-05, and A2-K1).
- **fp32 dual-oracle** correctness (§6): fla `naive`/`naive_recurrent` + repo CPU
  ref; judge in fp32 by rel-L2 / max_abs_diff, never bitwise, never bf16-elementwise.
- **Gate-on-unsupported, no silent fallback** (§7).
- **A2 is 20 cube / 40 vector**, `block_dim` ceiling untested — assert no number.

## Six ABI gaps → approach (from a2_gdn_abi.md §1.1–1.7)

1. **§1.1 GQA/GVA**: carry `HV` on v/state/output/saved histories, keep qk-side on
   `H`; in-kernel `i_h = i_hv // (HV//H)` (GDA-02 template), broadcast q/k/g/b across
   the `HV/H` group. `domain.shape` gains `HV: HV % H == 0`.
2. **§1.2 token-major**: public token-major, permute to kernel layout **inside** the
   launcher as a real materialization (not a strided view — A2 strided Slice is
   unavailable, AGENTS §5). Assert contiguity at the kernel boundary.
3. **§1.3 nonzero initial_state**: accept fp32 `[B,HV,128,128]` (KDA precedent).
4. **§1.4 dh0**: backward emits `dh0`, accepts `dht` (kda_bwd precedent).
5. **§1.5 fp32 final_state**: the only numeric-risk item. Build bf16-state and
   fp32-state variants; **report** o/final_state rel-L2 divergence vs the fp32
   recurrent oracle at long C as *evidence* (not a threshold). **Open decision (PM):**
   whether the shipped default is fp32-state only, or both retained.
6. **§1.7 scale + no-tail-path**: add `scale: f32` absorbed at q's read; require
   `T % 64 == 0`, else reject with the nearest legal T (no real tail path).

## L0C settle (a2_gdn_abi.md §2, A2-01 hit-table)

`barrier(Pipe.M)` after **every** L0C-accumulate MMAD, dtype-agnostic, at:
- fwd **inverse** stage (a5 `gdn_fwd/inverse.py:306/312/317-318` → the corresponding
  inverse block in `gdn_chunk_fwd/kernels/{stages,pipeline}.py`), FP32 M16 hand-chain.
- bwd **finalize** stage (`finalize.py:205/217/226/235/244`), FP32 chain.
- bwd **wu** stage (`wu.py:206`), BF16 M64 — barrier added for uniform discipline,
  not because M64 is unsafe.
Keep the pin's FP32 split-K rule; **reject M<64 bf16/fp16 splitk** per AGENTS §7
(Qwen3-Next is bf16, so this exposure is live). The existing
`tests/test_a2_accumulate_barriers.py` static guard auto-covers the new files.

## Test plan (`tests/test_a2_gdn_chunk.py`, new)

- fp32 dual-oracle on o / final_state / 7 grads.
- **Asymmetric `HV≠H`** (H=2, HV=4) with per-group-distinct values so a wrong
  `i_h` cannot pass.
- Two-segment chaining (nonzero initial_state + `dh0`) vs single call.
- `bd1 == bd4` bitwise where `B*HV` is large enough to exercise core splitting.
- Barrier negative control: delete one `barrier(Pipe.M)` → reproduce the M16 bf16
  failure (A2-11 method); accumulate-barrier count == accumulate-MMAD count.
- Small, safe gate span only (≤10, far from the 88.7 line) — span calibration is Batch B.

## Resolved decisions (PM NOTE, 2026-09-25, issue #44)

- **§1.5 shipped default = fp32-state.** The bf16-state build exists **only** to
  report the divergence evidence vs the fp32 recurrent oracle (into evidence, not a
  threshold, not a switchable shipping option).
- **a2 `MAX_GATE_SPAN` GDN field = blank / "untested (Batch B)".** Span calibration
  is Batch B (separate task). This batch's kernel tests use only small, safe gate
  span (≤10, far from the 88.7 line); no boundary testing, no specific number.

## Status

**All five forward stages implemented (a2 tensor-vector, no `@vf`) and validated
per-stage on 910B3 in fp32** vs a torch reference (GVA HV=4/H=2, N=2, small gate
span), under `kernels/`:

- `prepare.py` — qn/kn/gc/bk/wv, exact / ≤1e-7 (per-token beta/g via Var idiom).
- `scores.py` — score 1.45e-7, lower 1.43e-7 (decay-factored, no transpose).
- `wy.py` — u bit-exact, wy 6.1e-8 (row-sequential lower-triangular solve).
- `scan.py` — states 1.9e-7, delta 2.9e-7, final_state 3.0e-7 (chunk recurrence,
  nonzero initial_state; pure-vector, no cube — b3 L0C DMAs unavailable, A2-01).
- `output.py` — o 1.4e-7 (q·exp(gc)@S + causal score@delta).

`scan` was additionally re-validated with the **real prepare→scores→wy pipeline
outputs** (not just random inputs).

### a2 tensor-vector gotchas found during bring-up (all fixed in the kernels)

- `cadd` `dst_rep_stride` counts **elements**, not 8-elem blocks (unlike
  `mul`/`add`/`muladddst`). The scores reduction landed contiguously until this
  was set to the tile row stride in elements.
- `dup` on a **slice** needs an explicit `count=` (KDA convention); without it
  only one 64-lane repeat is zeroed, leaving the rest of the tile uninitialized.
- Folding a mul+tree-reduce+accumulate into a **called helper mistraces on a2**
  (wrong values, or OOB faults in tight layouts). Inline the sequence.
- The intrinsic `repeatTimes` (count/64) is capped at 255 — a full [128,128]
  `muls` is 256 repeats; split into two row halves.
- `while` is rejected in kernel code even at trace time; use `for range(...)`.

### End-to-end status (open)

Per-stage correctness is established. A true end-to-end run of all five stages
is **blocked on harness/environment, not kernel correctness**:

1. **One op per process/build (§6 rule 2, broader than naming).** Compiling all
   five kernels in one Python process with `compile_kernel` and then running even
   a single one faults (silent wrong-binary class, cf. BF-05) — even though the
   op names are distinct. The chained pipeline must run each stage through the
   proper unit runner / separate builds (as the a5 `run.py`/`_unit_runner` do),
   not an ad-hoc multi-`compile_kernel` script.
2. **Shared-box contention.** All 8 910B3 cards run at 90–160% AICore with 34–64
   GB HBM held by other users' jobs; the longest kernel (`scan`) intermittently
   hits `rtDeviceSynchronizeWithTimeout` (surfaced as a vector-core exception).
   Per-stage validations passed in free scheduling windows.

Next: scaffold the a2 unit (`contract.json`, `unit.py`, `pipeline.py`, `ref/`
mirroring a5, suffixed op name, HV axis, barrier sites) and run the dual-oracle
(`grouped_recurrent` + `block_solve`) end-to-end through the unit runner; then the
backward unit (dh0 §1.4). Re-estimate the ETA after the backward — the port is a
full vector-stage rewrite, not the file-move the 34h in a2_gdn_abi.md §5 assumed.

## Board bring-up result (2026-10-07)

Validated end-to-end on 910B3 by `tests/test_a2_gdn_chunk.py` (dual fp32 oracle +
autograd); numbers and the full checklist are in `docs/research/a2_gdn_abi.md` §7.
Device bring-up (sim did not model b3's vector UB-address bounds) required explicit
`count=` on the vector ops in `prepare`/`scores`/`output`, and — the one design
deviation — `scan`'s two matrix contractions (`wy@S`, `knd^T@delta`) now run on the
**cube** (matmul + L0C→GM→UB GMBuff ring) instead of the planned pure-vector matvec,
which faults on b3 for a kernel-context reason (byte-identical to the working backward
`reverse_rec`). ABI contract and numerics unchanged; see §7 for the rationale. Raw
acceptance numbers, the block_dim=40 justification and the captured env identity are in
`evidence/{acceptance_numbers,blockdim_sweep}.log` and `evidence/env.json`; the formal
ABI contract is `contract.json`. The backward's larger pure-vector deviation (zero cube,
zero `barrier(Pipe.M)`, vs the §2 cube finalize/wu) is disclosed in `a2_gdn_abi.md` §7
Deviation 2 and `../gdn_chunk_bwd/DESIGN.md`.
