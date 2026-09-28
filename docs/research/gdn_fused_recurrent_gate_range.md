# GDN recurrent ABI and qualification ledger

Task GDA-04, specification at `e8a845b`. This is an implementation-preflight
record, not a completed qualification. Kernel/device evidence is pending.
The CPU A identity clarification is tracked in issue #94 comment 5869386476.

## ABI

| Field | Supported form |
|---|---|
| Entry | `fused_recurrent_gdn(q,k,v,g,beta, *, initial_state=None, output_final_state=False, scale=128**-0.5, head_first=False, device='a5', block_dim=1, launcher='inprocess', ...)` |
| q/k | Contiguous `[B,S,H,128]`, matching FP32 or BF16 |
| v/output | Contiguous `[B,S,HV,128]`, same dtype as q/k |
| g/beta | Contiguous `[B,S,HV]`, FP32 scalar log decay/update rate |
| state | Contiguous `[B,HV,128,128]`, FP32, K before V |
| Initial state | None means zero; a tensor may be nonzero and is never modified |
| Returned state | Fresh tensor when requested, otherwise None |
| Shape | Positive B/H, positive HV divisible by H, S=1..16; each GM element count <=2**31-1 |
| Grouping | Consecutive HV/H value heads share one q/k head |
| Scale/normalization | Fixed scale, raw unnormalized q/k |
| Candidate bd grid | 1/2/4/8/16/28, **not yet qualified** |
| Native host work | Metadata validation, allocation, cached compilation/dispatch |
| Diagnostic launchers | `aclnn`/`board` accept CPU tensors and explicitly transfer them |
| Exclusions | Other dtypes, normalization, head-first, varlen, alternate state layout, raw gate flags, backward/autograd and undeclared options |

S_MAX=16 comes from fixed 16-row staging. No range expansion is inferred from
an upstream optional argument. All non-state operands and state must be on the
same device. Invalid metadata is rejected before loading or launching kernels.
CPU diagnostics also reject nonfinite values, positive g and beta outside[0,1].
Native callers satisfy these value preconditions and avoid intermediate
FP32 overflow; public execution does not copy inputs to CPU for validation.

## Mathematical target and operation boundaries

For each value head, select its contiguous group's q/k head. Exact BF16-to-FP32
widening belongs to the kernel. In FP32:

```
q_scaled = q * 128**-0.5
D = exp(g) * S_previous
prediction = sum_K(k * D)
delta = beta * (v - prediction)
S_next = D + k * delta
o = sum_K(q_scaled * S_next)
```

The state never narrows to BF16. Output storage narrows only at the final BF16
store using RNE. No input normalization, clipping, head copies or casts occur
in the public host path. In particular beta multiplies delta before the state
write; a beta*k reassociation is not silently borrowed from KDA decode.

## CPU references and pending identity clarification

FLA pin: `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`.
The pinned `fused_recurrent_gated_delta_rule` calls a Triton kernel; there is
no CPU branch. The proposed A therefore loads the literal
`naive_recurrent_gated_delta_rule` from the same pin, verified against SHA256
`d1cf17992349fd3e94af999b22e3d3a81be4a2d1881ce5b70a3457257166e0cb`.
Only test-side contiguous group expansion adapts its equal-head signature.
This source selection remains subject to the explicit PM clarification above.
No CUDA/Triton execution is claimed.

B is `ref/reference.py`: independent per-token batched matrix products, with
explicit state decay/read/update/output and independently enumerated head
selection. It imports neither A nor any device implementation. Calibration
also compares FP64 B with a clearly labeled lift of A: exactly its two explicit
FP32 cast nodes change to FP64. The original naive API is not claimed to be
FP64-preserving.

| CPU environment | Cases | Max A/B relative L2 | Small FP64 cases | Max FP64 relative L2 |
|---|---:|---:|---:|---:|
| Python3.11.15 / Torch2.10.0+cpu |88|1.8991640892534126e-7|3|2.082554894892825e-16|
| Python3.12.14 / Torch2.12.0+cu130 |88|2.0076726144241469e-7|3|2.174836294858946e-16|

These are preliminary reference-calibration results, not device acceptance.
Each environment generated and consumed its own actual input tensors.
The full grid and deterministic seeds are defined before kernel source in
`ref/reference.py`. A/B must stay below1e-5 before kernel development; the
fixed FP32 device budget is1e-4 for each output against each reference.

Metric norms accumulate FP32 semantic values in FP64 for measurement. No
adjustable denominator floor is used: relativeL2=norm(error)/norm(reference),
with exact zero difference required if reference norm is zero. BF16 output
quality/storage observations do not relax the FP32 arithmetic budget. Negative
controls include zero, sign inversion,1.25x output, nonfinite results, wrong
shape/dtype and missing outputs.

## Range analysis

| Expression | Range and consequence |
|---|---|
| `exp(g)` | g<=0 gives [0,1]; very negative g may underflow to zero. This does not make the complete recurrence stable. |
| `I-beta*k*k^T` | Spectral norm is max(1,abs(1-beta*norm(k)^2)); arbitrary finite keys can amplify state. |
| State propagation | Norm is bounded by exp(g)*the preceding factor per step, plus the beta*k*v term; initial state magnitude matters. |
| k/state and q/state reductions | Products, partial sums, cancellation and FP32 representability require explicit numerical cases; finite inputs alone are insufficient. |
| Output narrowing | BF16 storage incurs RNE error; FP32 state retains the accumulation target. |

The88-case grid includes a full B1/S16/H16/HV32 workload first in both dtypes;
ratios1/2/4/8 and S1/2/3/7/15/16; multi-batch and72 work items; None/zero/random
state; g0/-.001/-30/-1000; beta0/1; zero q/k/all inputs; initial state scales16
and1e-6; key norms1/2 and signed spikes. Key normalization in the seeded input
generator defines a test profile only; it is not an operator transform.
Numerical support claims will be limited to the measured domain.

## Integration and measurement plan

Chunk starts from zero and emits K-major state. Prefixes64/128/4032 with a
64-token decode suffix produce legal whole-chunk lengths128/192/4096. Split
suffixes into short calls and compare every returned token and final state;
also compare one S16 decode with state-chained shorter calls. State cannot be
fed back into chunk because chunk rejects nonzero initial_state.

All chunk and decode kernels must be prepared before first aclnn use. Use one
process and build per operator name and bd; cross-bd comparison uses identical
input hashes and actual output bytes. BF16/FP32 preparation is kernel-side.

The intended timing baseline is a native torch_npu composition of the same
GDN recurrence with raw q/k and fresh state. It must actually run and be checked
before any performance claim. Measure S1 fixed cost and S16 separately on the
same card, three baseline/candidate/baseline rounds,10 warmup and50 synchronized
samples per leg. No speed threshold or CUDA/Triton comparison is promised.

## Evidence still required

Kernel source/static checks, vendor compilation, complete native workload,
bd qualification, small bounded model diagnostics, prefill/decode integration,
public host-work audit, native BF16 storage checks and same-card measurements
are pending. The branch is not ready for DONE or PR acceptance.
