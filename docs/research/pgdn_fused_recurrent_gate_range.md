# PK-06 PGDN fused recurrent: ABI and numerical range

Status: final-source native acceptance completed; submitted for PM review. Task #95; public entry `ascend_fla.ops.pgdn_fused_recurrent`.

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
| block_dim | 1,2,4,8,16,28; each qualified by 126 native cases |
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
| Python3.11.15 / Torch2.10.0+cpu |126|1.506501779e-7|2.663937766e-16|
| Python3.12.14 / Torch2.12.0+cu130 |126|1.658265999e-7|2.412628766e-16|

Each of o, final_state and final_A_state is separately compared to A and B.
FP32 relative L2 budget is1e-4; a zero reference norm requires zero residual.
BF16 o is a quality/storage boundary, while both FP32 states retain their
budgets. Native FP32 companions check exact BF16 RNE output and both state
identities. No tolerance is adjusted after observing device results.

## Oracle environment boundary (PM decision)

[PM NOTE6033949597](https://github.com/ddddwee1/ascend_fla_dev/issues/95#issuecomment-6033949597)
freezes the CPU reference used throughout device acceptance: Python3.12.14,
Torch2.12.0+cu130, torch git `7661cd9c6b841b62b7f411aa52ec51f05457263b`.
The references execute on CPU; the version suffix is not a CUDA execution claim.
The literal file, fixed constants and1e-4 budget remain unchanged.

The same literal source under host Torch2.10.0 differs at these recorded ATK
boundaries (B1/S1/H16/HV32, nonzero A0 scale1e30, beta_atk0). They are known
cross-environment differences, outside the task's frozen oracle identity; they
are not removed from the evidence or silently relabeled as identical.

| g_atk | Native vs frozen Docker A | Native vs host Torch2.10 A | Host A vs frozen A |
|---|---:|---:|---:|
| -94.10157775878906 | bit-identical |1.033497057e-4|1.033603880e-4|
| -95.60408020019531 | bit-identical |4.642522414e-4|4.644678717e-4|

At the second point, the references differ by more than the sum of two1e-4
budgets. PM explicitly does not require simultaneous agreement with both CPU
environments. Additional native checks at g_atk=-100/-90 with A0<=1 returned
positive subnormal ATK bit patterns identical to the frozen reference; a zero
reference norm did not hide an error. These observations do not prove accuracy
for every possible subnormal midpoint.

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
includes this diagnostic. The first candidate failed with ATK relativeL2=1 while
o and main state passed. The focused native exp probe and original returned
tensors are retained. The repair uses an RNE integer subnormal mantissa and
normal power-of-two multiplication factors, entirely inside the ATK kernel.
The unchanged failing case subsequently matched both references bit for bit.

Input cases cover grouping1/2/4/8, independent initial combinations, short S
boundaries, nonzero A up to scale1e30, main-state scales1e-6/.25/16, zero and
clamped q/k, gate/update endpoints and worker reuse B3/H28/HV224. Exact
runtime-generated inputs and per-output metrics belong to retained evidence.

## Executed acceptance

The implementation is commit `b6f2613`; the evidence receipts bind complete source
and artifact hashes. Later documentation and evidence-audit changes do not alter
those executed sources. A5 CANN/OPP9.1.0-beta.1, timestamp20260509_173000235;
installed kernel packages include ascend950, ascend910b and ascend910_93. Native
Python3.12.14/Torch2.12.0+cu130/torch_npu2.12/Ascriptor0.1.0 is separate from the
CPU-only host test environment Python3.11.15/Torch2.10.0+cpu.

| Stage | Observed result |
|---|---|
| Static check | FP32 343 ops and BF16 397 ops, each 0 errors/0 warnings |
| Independent host reference | 63/63 canonical FP32 cases |
| Public tests / full host suite | 132 passed / 1582 passed,10 skipped,5 known importorskip deprecations |
| Vendor compilation | Both entries at all six block dimensions; joint 14-entry package for integration |
| Native full workload first | B1/S16/H16/HV32, both dtypes; also rerun after ATK repair and final source cleanup |
| Native grid | 126 cases × six block dimensions =756; input and all three output hashes identical across bd |
| Maximum budgeted relative L2 vs A or B | FP32 o 2.349866029e-7; main state 1.936438754e-7; ATK state 2.419267506e-7 |
| BF16 storage | 378 companion checks: exact RNE o plus byte-identical FP32 main/ATK states |
| Chunk→decode | 14 input cases,56 continuation chains, widths1/3/7/16, prefixes64/128/4032 |
| Max FP32 token / main / ATK vs whole chunk | 6.883051439e-7 /5.470309742e-7 /0 |
| Max FP32 token / main / ATK vs independent B | 6.064010427e-7 /1.460629519e-7 /1.199243389e-7 |
| Returned intermediate states vs B | Both checked at every return; maximum4.550687813e-7 |
| Decode split→single call | 150 comparisons, o and both states byte-identical |
| Host operation audit | Eight native cases; allowed allocations only, both output states hidden when flagFalse |
| Loop-bound diagnosis | Every S=1..16 × both dtypes, B2/H3/HV12:32 cases passed |
| Precision boundary diagnosis | Four additional native ATK cases, including two positive subnormal returned states |
| Focused sim and pipesim | FP32 B1/S1/H3/HV6 and BF16 B1/S3/H2/HV4, bd1; no hazards/event imbalance/deadlock |

Models were run after full hardware workloads and before a two-blank-line EOF
cleanup. All emitted file hashes were unchanged; final native acceptance was
rerun with the exact delivered source bytes. Models do not qualify the full grid
or the hardware transcendental implementation. Standalone board/aclnn harnesses
were not used for acceptance; the public inprocess path was.

## Same-card measurements

Baseline is the actually executed torch_npu PGDN recurrence in this unit's
`ref/native_baseline.py`, with both nonzero initial states. Candidate is the
public inprocess API, bd4. Both include allocation/dispatch, synchronize before
and after each call, exclude compile and data transfer, and use identical
runtime-generated inputs. Each leg has10 warmups and50 samples; three B/C/B
rounds over four cases produce1800 retained raw samples. No speed threshold.

Median microseconds, each cell is baseline-before / candidate / baseline-after:

| B1/H16/HV32 | Round1 | Round2 | Round3 |
|---|---|---|---|
| S1 FP32 |514.497 /79.670 /317.186|316.686 /78.909 /316.546|315.509 /78.388 /318.743|
| S1 BF16 |355.814 /80.586 /361.038|357.021 /80.090 /348.929|349.865 /79.194 /348.023|
| S16 FP32 |2482.652 /170.466 /2489.071|2485.861 /170.325 /2467.894|2494.309 /169.690 /2464.735|
| S16 BF16 |2484.179 /169.745 /2491.344|2480.739 /168.493 /2476.853|2468.595 /168.378 /2474.960|

The first S1 FP32 baseline-before leg drifts materially relative to its after
leg. It is retained, not discarded or used to advertise a stable speedup.
No CUDA/Triton execution, weight validation or comparison with a GPU kernel
is claimed. Shared locks, fresh health checks and sampled process ownership
support these observed device windows, with machine details kept private.

## Evidence and limitations

The [evidence index](../../kernels/projects/a5/pgdn_fused_recurrent/evidence/index.json)
links exact stages. Native result files retain per-case A/B metrics, per-token
continuations, both state chains, hashes and all timing samples. Original ATK
failure, exp probe, corrected output bits and CPU environment differences remain
visible. [Warning dispositions](../../kernels/projects/a5/pgdn_fused_recurrent/evidence/warnings/README.md)
locate every vendor/native diagnostic and bound the resulting claims.

`python kernels/projects/a5/pgdn_fused_recurrent/verify_evidence.py` independently
recomputes recorded norms/budgets, state-chain links, block-dimension identities,
source hashes and timing medians. It audits published records; it is not another
kernel run. The ABI's value preconditions and measured range remain essential:
no arbitrary finite-magnitude or universal subnormal-midpoint guarantee is made.
