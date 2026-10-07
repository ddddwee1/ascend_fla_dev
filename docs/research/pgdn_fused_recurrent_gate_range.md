# PK-06 PGDN fused recurrent: ABI and numerical range

Status: candidate under validation. No hardware qualification is claimed by this
initial ledger. Task #95; public entry `ascend_fla.ops.pgdn_fused_recurrent`.

## Frozen ABI

| Item | Contract |
|---|---|
| Target | A5, CCE, single pure-vector launch |
| q/k | Contiguous `[B,S,H,128]`, matching FP32 or BF16 |
| v/o | Contiguous `[B,S,HV,128]`, same dtype as q/k |
| g/beta | Contiguous FP32 `[B,S,HV]` |
| g_atk/beta_atk | Contiguous FP32 `[B,S,H]` |
| Initial/final main state | FP32 `[B,HV,128,128]`, K before V |
| Initial/final ATK state | FP32 `[B,H,128]`, one state per key head |
| Initials | Independent None (zero), explicit zero or nonzero tensors |
| Outputs | Fresh contiguous tensors, inputs unchanged; both states None when output_final_state=False |
| Grouping | Positive H and HV; HV is a multiple of H; consecutive HV/H heads share one key head |
| Short sequence | S=1..16, no alignment constraint; S_MAX=16 follows UB row allocations |
| block_dim | Candidate set 1,2,4,8,16,28; qualification pending actual device grid |
| Constants | scale=128**-.5, x=1.5, ATK eps=1e-6, center=-.2 (None selects -.2) |
| Normalization | Mandatory FP32 `x/max(norm(x),1e-12)`, then q scaling |
| Host work | Metadata checks, allocation, prepared dispatch and launch; no input arithmetic or casts |
| Exclusions | Backward, varlen, context parallel, head_first, V-major state, configurable constants |

NPU callers provide finite operands, nonnegative initial ATK state,
g/g_atk<=0 and beta/beta_atk in[0,1], with finite FP32 intermediate squares,
reductions and state updates. CPU diagnostic launchers check values. Metadata
validation alone does not establish those value preconditions. Numerical
acceptance describes the measured domain, not every finite FP32 operand.

The production host never normalizes, expands heads, casts q/k/v or repairs
contiguity. Both BF16 widening and output RNE narrowing are kernel operations.
BF16 inputs therefore follow the same FP32 arithmetic as their exact FP32
widening. Public outputs retain the native input dtype; both states remain FP32.

## Ownership, storage and synchronization

One vector participant owns each `(B,H)` item. It normalizes all short-sequence
q/k rows and computes a single ATK trajectory, retaining `q_scaled`, `k_read`
and `k_write` for each token. It then sequentially advances the associated
HV/H main states. ATK lookahead is valid because ATK does not depend on v,
main gates or main state. It must retain every k_write row, not only final A.
The correction reads k_read; the rank-one update writes k_write.

FP32 uses 109088 bytes of explicitly allocated UB: one 128x128 main state,
five 16x128 rows, one 128-element ATK state, four 16x8 scalar-gate buffers,
and one 8-element reduction staging buffer. BF16 adds four 16x128 BF16
buffers for a total125472 bytes and16 allocations. These are source-level
counts; emitted addresses and device behavior remain separate checks.

MTE2/V/MTE3 edges use owner auto_sync. Explicit STORE->LOAD barriers order
scalar reduction staging/broadcast, the two main-state sweeps, consecutive
tokens and final stores. Each output region has exactly one writer. No cube,
inter-participant state sharing or speculative buffer reuse is introduced.

## Runtime preparation

`prepare(block_dim=..., chunk=True, chunk_block_dim=2)` prepares all14 entries
before the first aclnn call. One process must use one build per operator name.
Decode-only preparation registers the two native typed entries.

| Family | Entries |
|---|---|
| Chunk FP32 | pgdn_chunk_atk, pgdn_chunk_prepare, pgdn_chunk_scores, pgdn_chunk_wy, pgdn_chunk_scan, pgdn_chunk_output |
| Chunk BF16 | pgdn_bf03_atk, pgdn_bf03_prepare, pgdn_bf03_scores, pgdn_bf03_wy, pgdn_bf03_scan, pgdn_bf03_output |
| Decode | pgdn_recurrent_fp32, pgdn_recurrent_bf16 |

Chunk starts from zero and cannot consume either nonzero initial state. The
validated integration direction is chunk prefix -> decode suffix, passing
both returned NPU states directly. No reverse handoff is advertised. Width1
checks expose both states after every token; wider calls expose both states
at each return. Whole chunk only returns terminal states, so intermediate
states compare to independent CPU B and terminal states to both B and chunk.

## Sources and reference calibration

- library: `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`
- kernels: `b3b3f9c16df7c4626ed3c081032a1be5a753d0b1`
- FLA: `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`
- Literal A: `fla/ops/precond_gated_delta_rule/naive.py`, SHA256
  `3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8`

A is loaded unchanged with a full-file hash check. B independently owns H
ATK states and uses grouped batched matrix products for the main recurrence;
it does not import A. FP64 calibration is explicitly a lifted A, not original
naive FP64 support. The lift inventories four dtype nodes at source lines95,
116,118,120: input cast, scalar center, tensor center and log(x). All four are
lifted; fixed scalar center makes the tensor-center branch unreachable here.

| CPU environment | Cases | Max FP32 A/B relative L2 | Max lifted FP64 relative L2 |
|---|---:|---:|---:|
| Python3.11.15 / Torch2.10.0+cpu |112|1.506501779e-7|2.663937766e-16|
| Python3.12.14 / Torch2.12.0 |112|1.658265999e-7|2.412628766e-16|

Each of o, final_state and final_A_state is separately compared to A and B.
FP32 relative L2 budget is1e-4; a zero reference norm requires zero residual.
BF16 o is a quality/storage boundary, while both FP32 states retain their
budgets. Native FP32 companions check exact BF16 RNE output and both state
identities. No tolerance is adjusted after observing device results.

## Separate main and ATK range arguments

For nonnegative A0 and normalized k, exact arithmetic gives
`0 <= A_t <= A0+t` componentwise. Zero-initial chunk's bound `A<=t` therefore
does not cover decode with nonzero A0. The fixed squash satisfies
`2/3 < M_j < 3/2` and k_write=diag(M)k_read.

Main propagation is `P=I-beta*diag(M)*k_read*k_read^T`. It is similar to a
symmetric matrix through sqrt(diag(M)); a fixed weighted norm is nonexpansive,
but ordinary Euclidean norm need not be. A conservative step bound is
`||S_t||F <=1.5*exp(g)*||S_prev||F+1.5*beta*||v||2`.
The weighting changes per token, so g<=0 alone does not prove global stability.
An embedded two-coordinate FP64 example with A0=(0,1e30), normalized equal
key coordinates and beta1 yields a singular norm1.0751617745 although its
eigenvalues are approximately(-.0647414,1). This is an algebraic range example,
not a device result.

A tiny exponential times large A0 can matter independently of o: literal CPU
FP32 exp(-100)*FP32(1e30) is about3.78350576e-14. Hypothetically flushing the
exponential would lose all that ATK value while the fixed squash can remain
byte-identical because A is below the eps addition resolution. The grid
includes this diagnostic; its current device outcome is pending. A failure
must be retained and repaired or reported without relaxing the ATK budget.

Input cases cover grouping1/2/4/8, independent initial combinations, short S
boundaries, nonzero A up to scale1e30, main-state scales1e-6/.25/16, zero and
clamped q/k, gate/update endpoints and worker reuse B3/H28/HV224. Exact
runtime-generated inputs and per-output metrics belong to retained evidence.

## Verification status

Static checks: both entries0errors/0warnings. Standalone FP32 reference56/56.
Public ABI/reference tests130passed. Vendor compilation, native grid,
functional/pipe simulation, state continuation, host audit and same-card
three-round actual torch_npu baseline/candidate/baseline are tracked separately
and are pending acceptance. No CUDA/Triton or model-weight execution is claimed.
