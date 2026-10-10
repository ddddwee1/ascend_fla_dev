# A2-K1 — GDN chunk backward move-to-a2 (design / plan)

Batch A of `docs/research/a2_gdn_abi.md` §5, backward half. Ports the a5
`kernels/projects/a5/gdn_chunk_bwd` unit to the A2 (c220) tensor-vector model,
closing the same six ABI gaps as the forward and adding `dh0` (§1.4). The forward
half (`../gdn_chunk_fwd`) is implemented and validated per-stage; this document
plans the backward, to be built the same way (per-stage validation on 910B3 vs the
analytical adjoint, then wired through the unit runner).

> **Shipped chain (2026-10-07, board-validated).** The §40 "three-kernel structure"
> below is the *planned* decomposition. What actually ships — and what
> `tests/test_a2_gdn_chunk.py` validates on 910B3 against the dual oracle + autograd —
> is the 8-kernel v1 chain `replay → reverse_rec → reverse_dq/dbeta/dg/dk_bz/dk_ddr →
> group_reduce` (a sim-imposed decomposition; `replay` is the former `checkpoints`
> stage, save point aside). It is **pure vector — zero `matmul`, zero
> `barrier(Pipe.M)`**; the §2-planned cube finalize/wu stages do not exist in the
> shipped chain. That structural deviation and its ~10× perf gap vs fla's chunk-parallel
> Triton backward are disclosed in `docs/research/a2_gdn_abi.md` §7 (Deviation 2). The
> earlier `checkpoints`/fused-`reduce`/tapeless-`reverse_rec_fused` (v2/v3) perf variants
> were removed so the deliverable is unambiguously this validated v1 chain.

## The adjoint math (device-independent oracle)

`a5/gdn_chunk_bwd/ref/reference.py::analytical` is the ground truth (no FLA, no
autograd; finite-difference-qualified). The primal recurrence per token t, per
(b, value-head hv), with K=V=128, scale = 128**-0.5, GVA ratio = HV//H:

  d     = exp(g[t]) * S                       # decayed prior state [K,V]
  r     = v[t] - (k[t] ⊙ d).sum_K            # residual [V]
  z     = beta[t] * r                         # [V]
  S     = d + k[t] ⊗ z                        # rank-1 update [K,V]
  o[t]  = scale * (q[t] ⊙ S).sum_K           # (forward output)

The reverse scan carries the state cotangent `back` [K,V] (initialized to `dht`,
or 0), and per token in reverse:

  dq[t]  = scale * (S ⊙ do[t]).sum_V          # [K]  (per value-head; GVA-reduced later)
  back  += scale * q[t] ⊗ do[t]               # [K,V]
  dz     = (back ⊙ k[t]).sum_K                # [V]
  dr     = beta[t] * dz                        # [V]
  dk[t]  = (back ⊙ z).sum_V - (d ⊙ dr).sum_V # [K]  (per value-head; GVA-reduced later)
  dv[t]  = dr                                  # [V]
  dbeta[t] = (dz ⊙ r).sum_V                    # scalar
  dD     = back - k[t] ⊗ dr                    # [K,V]
  dg[t]  = (dD ⊙ d).sum_{K,V}                  # scalar
  back   = exp(g[t]) * dD                       # [K,V]  (propagate to t-1)

`dh0` = `back` after the whole reverse scan (the cotangent w.r.t. the initial
state; kda_bwd precedent, §1.4). GVA: dq/dk are formed per value-head (HV) and
summed over the ratio into the qk-head (H) — `dq.reshape(B,T,H,ratio,K).sum(3)`.

## Three-kernel structure (mirrors a5)

The a5 unit is three kernels; the a2 port keeps the split (all **pure vector**,
no cube — b3 has no L0C DMAs, A2-01):

1. **`gdn_chunk_bwd_checkpoints_a2`** — replay the primal forward per (b,hv),
   saving the state at each chunk boundary (every 64 tokens) plus the final
   state. This is the forward scan (identical recurrence to `../gdn_chunk_fwd`
   scan step, minus the delta/output) writing `checkpoints[B,HV,N+1,128,128]`.
   Reuses the forward `_spread8` + `muladddst` rank-1 (S += k⊗z) and the
   rowscale+tree-reduce matvec ((k⊙d).sum_K), per-token scalar `beta`/`g` via the
   Var idiom.
2. **`gdn_chunk_bwd_reverse_a2`** — per (b,hv), for each chunk in reverse: reload
   the boundary state (checkpoints), replay the chunk forward writing the decayed
   state `d_t` to a GM tape `tape_d[B,HV,C,128,128]` and keeping `r_t`,`z_t` in UB
   `[C,V]` tiles, then reverse-scan producing `dq_parts[B,T,HV,128]`,
   `dk_parts[B,T,HV,128]`, `dv`, `dg`, `dbeta`, threading `back` (init `dht`, final
   -> `dh0[B,HV,128,128]`).

   UB-fitting derivations (so only two [128,128] tiles -- `back` and `d_t` -- plus
   a half scratch are live, ~160 KiB):
   - Do NOT materialize `S_after_t`. Since `S_after = d + k(x)z`,
     `dq[t] = (d_t . do_t) over V + k_t * (z_t . do_t)` -- taped `d_t` contracted
     over V, plus the key scaled by the scalar `z_t . do_t`.
   - `dk[t] = (back . z_t) over V - (d_t . dr_t) over V` -- two [K,V]-over-V sums.
   - Reductions come in BOTH orientations: over V (rows keep K) for dq/dk/dbeta --
     the scores `mul`+`cadd` row-reduce; over K (rows keep V) for
     `dz = (back . k_t) over K` -- the scan/output `_spread8`+rowscale+tree-add.
     `dg` is a full [K,V] reduce (row-reduce then reduce the [K,1] column).
   - The two rank-1s are `muladddst`: `back += (scale*q_t)(x)do_t`, and
     `dD = back - k_t(x)dr_t` (negate+accumulate), then `back = exp(g_t)*dD`.
   Reverse loops count down with `cc = N-1-ccx`, `i = C-1-ix` (runtime N; C unrolled).

   **Transpose decision (no cube on b3, no in-kernel transpose op).** dq/dk are
   indexed by K and fall out of the over-V reductions as **columns** `[K,1]` (one
   value per K-row), but the token-major public layout wants them as **rows**
   `[1,K]`. b3 has no cube to transpose freely, and a per-token scalar scatter
   (K GetValueFrom/SetValueTo per column × 2 × C × N) is millions of scalar ops.
   Chosen approach: the reverse kernel writes dq/dk **K-major** —
   `dq_parts[B,HV,128,T]` / `dk_parts[B,HV,128,T]` (each token's column stored with
   `cadd`'s natural `[K,1]` layout, `dst_rep_stride` in elements) — and the unit
   launcher does the K↔T permute + the GVA ratio-sum to the public `dq/dk`
   `[B,T,H,128]` as a boundary **materialization** (permitted by §1.2; it is a pure
   reshape/permute + a sum over the ratio, no per-element host arithmetic beyond the
   declared group-reduce). `group_reduce` therefore folds into that boundary step,
   or stays a kernel that reads/writes the K-major layout. This keeps every
   in-kernel store contiguous and avoids both the strided sub-column store (which
   faults, see scores) and the scalar-scatter cost.
3. **`gdn_chunk_bwd_group_reduce_a2`** — sum `dq_parts`/`dk_parts` over the GVA
   ratio into `dq[B,T,H,128]`/`dk[B,T,H,128]`. Trivial strided add reduction;
   a no-op copy when HV==H.

## ABI (gaps §1.1–1.7, as forward)

- Inputs `q,k,v,g,beta,do,dht`; outputs `dq,dk,dv,dg,dbeta,dh0`. Public token-major
  `[B,T,H,128]` (q/k), `[B,T,HV,128]` (v/do), `[B,T,HV]` (g/beta); read via 2-D
  `[B*T,·]` views (D-PM-35/37). `dht` fp32 `[B,HV,128,128]`; `dh0` emitted same.
- **§1.1 GVA**: v/g/beta/do/state on HV; q/k on H; in-kernel `i_h = i_hv//(HV//H)`;
  dq/dk formed on HV then group-reduced. `HV % H == 0`.
- **§1.5 fp32 state/cotangent** throughout (no bf16-state variant needed for
  correctness; report divergence only if a bf16 build is later requested).
- **§1.6 scale** absorbed at q's read; **§1.7** `T % 64 == 0`, no tail path.
- **C-keyword-safe kernel params**: no `do`/`in` — the `do` GM edge is named
  `dout` in the kernel signature (GDA-03/PK-05/A2-K1 flag).

## a2 tensor-vector constraints carried from the forward (see ../gdn_chunk_fwd/DESIGN.md)

- Keep the mul + tree-reduce + accumulate **inline**; folding into a called helper
  mistraces on a2.
- `dup` on a slice needs an explicit `count=`; `cadd` `dst_rep_stride` counts
  elements; intrinsic `repeatTimes` (count/64) ≤ 255 (split full-tile ops); no
  `while` in kernel code.
- Reductions over the K rows use the 7-step (128→1) explicit tree; over C rows the
  6-step (64→1) tree; value dim split into two 64-lane halves for the [128,64]
  scratch when a second [128,128] tile will not fit.

## L0C settle (a2_gdn_abi.md §2)

Not applicable to this pure-vector port — the a5 backward's `barrier(Pipe.M)`
sites (finalize/wu) are cube-MMAD accumulate points, and this port uses no cube.
The static guard `tests/test_a2_accumulate_barriers.py` still auto-covers the new
files (zero accumulate-MMADs ⇒ zero required barriers).

## Test plan (`tests/test_a2_gdn_chunk_bwd.py`, new)

- fp32 dual-oracle on dq/dk/dv/dg/dbeta/dh0 vs `analytical` (rel-L2 ≤ 1e-4, the
  a5 bwd bar) and finite-difference qualification at tiny sizes.
- Asymmetric `HV≠H` (H=2, HV=4) with per-group-distinct values so a wrong `i_h`
  or a missing group-reduce cannot pass.
- Two-segment chaining: nonzero `dht` in, `dh0` out chained across a split vs a
  single call (§1.4).
- Nonzero `dht` control (dht=0 vs dht≠0 changes dq/dk/dg as predicted).
- Small, safe gate span only (≤10, far from 88.7) — span calibration is Batch B.

## Status

**Backward VALIDATED end-to-end on the functional simulator** (CPU, device-independent
— the 910B3 cards are saturated by other users) vs the analytical adjoint, GVA
HV=4/H=2, N=2:  dq 1.4e-7, dk 1.2e-7, dv 1.3e-7, dg 1.7e-7, dbeta 1.7e-7. (The
forward likewise validated end-to-end on the sim, o/final_state ~3e-7.)

The monolithic reverse deadlocked the sim's auto_sync credit budget AND timed out the
board cce build — both because its per-token adjoint body was too large. The fix was
to **decompose** into small kernels, each with a per-token body as simple as the
validated checkpoints/replay (fast to build, within the sim budget):

- `replay.py`        — per-token decayed state `d_t` -> `tape_d[B,HV,T,128,128]`.
- `reverse_rec.py`   — the recurrence: `back += scale*q^do`; tape `back_t`; dv;
                       `back = exp(g)*(back - k^dr)`; `dh0` = final back.
- `reverse_dq.py`    — dq = scale*((d.do)_V + k*(z.do)).   [1 kreduce + 1 rowdot]
- `reverse_dbeta.py` — dbeta = (dz.r)_V.                    [2 kreduce + dotscalar]
- `reverse_dg.py`    — dg = ((back - k^dr).d)_{K,V}.        [kreduce + muladddst + reduce]
- `reverse_dk_bz.py` / `reverse_dk_ddr.py` — the two dk over-V terms (back.z, d.dr).
- `group_reduce.py`  — GVA ratio-sum of dq, and dk = ratio_sum(dkbz - dkddr).

`checkpoints.py` (boundary-state variant, validated ~5e-8) is legacy — the decomposition
uses `replay` (full per-token tape) instead.

Two sim-sync rules learned and applied throughout: (1) every buffer element written must
be consumed or the sim deadlocks ("a ready published more often than it is waited for");
(2) each kernel must stay under a per-kernel sync-credit ceiling (~1-2 over-V row-build
`cadd` reductions plus a couple of kreduce/muladddst), which the small kernels respect.

Remaining: scaffold the fwd+bwd units (`contract.json`/`unit.py`/`pipeline.py`, suffixed
op names, barrier sites) and run the board dual-oracle e2e through the unit runner when a
card frees. The math and every kernel are validated on the sim; the board pass is the
only step still gated on hardware.
