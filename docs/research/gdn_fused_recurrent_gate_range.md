# GDN recurrent ABI and qualification ledger

Task GDA-04, specification at `ad2184a`. Native inprocess qualification covers
the measured FP32/BF16 domain below. Other launchers and arbitrary finite
magnitudes are not inferred from this evidence.
PM approved the exact CPU A identity in [issue #94 comment 5870217647](https://github.com/ddddwee1/ascend_fla_dev/issues/94#issuecomment-5870217647).

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
| Qualified native bd grid | 1/2/4/8/16/28; all88 cases at each, output/state bytes identical across bd |
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

## CPU references and approved identity

FLA pin: `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`.
The pinned `fused_recurrent_gated_delta_rule` calls a Triton kernel; there is
no CPU branch. A loads the literal
`naive_recurrent_gated_delta_rule` from the same pin, verified against SHA256
`d1cf17992349fd3e94af999b22e3d3a81be4a2d1881ce5b70a3457257166e0cb`.
Only test-side contiguous group expansion adapts its equal-head signature.
PM explicitly accepted the existing88-case calibration without rerunning it.
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

These are approved reference-calibration results, not device acceptance.
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
Numerical support claims are limited to this measured domain.

## Integration and measurement method

Chunk starts from zero and emits K-major state. Prefixes64/128/4032 with a
64-token decode suffix produce legal whole-chunk lengths128/192/4096. Split
suffixes into short calls and compare every returned token and final state;
also compare one S16 decode with state-chained shorter calls. State cannot be
fed back into chunk because chunk rejects nonzero initial_state.

All chunk and decode kernels must be prepared before first aclnn use. Use one
process and build per operator name and bd; cross-bd comparison uses identical
input hashes and actual output bytes. BF16/FP32 preparation is kernel-side.

The timing baseline is an actually executed native torch_npu composition of
the same GDN recurrence with raw q/k and fresh state, checked against A/B.
S1 fixed cost and S16 are measured separately on the same card, three
baseline/candidate/baseline rounds,10 warmup and50 synchronized samples per leg.
No speed threshold or CUDA/Triton comparison is used.

## Observed results

Both typed entries passed static checks with0 errors and0 warnings and emitted
pure-vector CCE. Lowered UB allocation is97KiB for FP32 and113KiB for BF16;
the latter includes an explicit FP32 output stage before RNE narrowing.
The44 canonical FP32 CPU reference cases passed. Task tests:85 on host,84 on
Docker; the extra final host test rejects the signed32 state-index limit using
meta inputs before allocation or kernel loading. Full CPU host regression before
that test-only addition:1449 passed,10 skipped,5 warnings.

The first complete native B1/S16/H16/HV32 workload passed for both dtypes, with
actual torch_npu baseline outputs also checked. FP32 output/state maximum
relative L2 against A/B was1.5653452511426124e-7. BF16 state was within the
FP32 budget, and BF16 output/state matched the same-input native FP32 output
after RNE/state bytes respectively. BF16 output quality relative L2 was about
0.0016592; this is not a FP32 arithmetic tolerance.

The initial88-case bd1 grid had a foreign task observed during its window;
its numerical results are retained separately. An isolated repeat passed all88
cases, maximum budgeted relative L2=2.170130340355715e-7, with all44 BF16
storage comparisons byte-identical. No foreign context was observed in that
repeat. All six isolated bd grids subsequently passed528 total cases with the
same maximum budgeted error,264 BF16 storage checks, and identical input and
returned output/state hashes across all block dimensions. Each isolated run
had zero sampled foreign contexts and empty contexts before and after.

After the full hardware run, bounded FP32 B1/S2/H3/HV3 and BF16 B1/S2/H1/HV4
with None state passed functional simulation and pipe simulation at bd1.
The latter reported no event imbalance, hazard or deadlock. These are model
diagnostics of repeated state reuse and cast footprints, not native evidence.

Actual chunk→decode integration passed14 cases/56 chains, with state passed
directly between NPU calls. Against whole native chunk, maximum FP32 suffix
token relative L2 was6.803415911187458e-7 and final-state error was
5.530971869996005e-7. Against independent CPU B, the corresponding maxima were
6.551472638786963e-7 and1.2300183594765942e-7. All132 S16 split comparisons
matched native one-call output/state bytes. The first integration collection
window observed a foreign test near its end; a fresh isolated repeat produced
the same metrics with no foreign contexts observed.

Same-card public-call latency, pooled medians of150 candidate and300 baseline
samples per workload (three B/C/B rounds):

| B/S/H/HV | dtype | Candidate μs | torch_npu baseline μs | Baseline / candidate |
|---|---|---:|---:|---:|
|1/1/16/32|FP32|66.455|120.416|1.812|
|1/1/16/32|BF16|66.615|176.335|2.647|
|1/16/16/32|FP32|157.206|1138.519|7.242|
|1/16/16/32|BF16|155.614|1177.632|7.568|

These synchronized wall times include public allocation and dispatch, with
input generation, CPU references, H2D and compilation outside the timed region.
No foreign contexts were observed. They are not device-only kernel times,
CUDA/Triton comparisons, weight validation or a model-level speed claim.

Actual ATen dispatch auditing covered both dtypes, None/nonzero initial state,
and both output-state flags. The prepared public path issued only
`aten.empty.memory_format`: two output allocations, plus one unread placeholder
when initial state is None. Each typed binding additionally allocated its uint8
workspace on first use, then reused it. Source call sites and all operations
are retained in `native/host-audit-bd4-v2`; unexpected operations were empty.
Source inspection additionally confirms all value checks/NaN poisoning are
restricted to explicit CPU diagnostic paths and no runtime reference fallback
is present. No kernel or shared runtime changes were needed for this audit.

Environment: Python3.12.14, Torch2.12.0+cu130 with torch_npu2.12.0,
Ascriptor0.1.0 at library pin90cfcdc, CANN/compiler/OPP9.1.0-beta.1,
timestamp20260509_173000235, Ascend950PR_9579V100 for the isolated grids,
integration and timing. SoC variants are recorded per execution; support is
not inferred for other hardware. Exact hashes, all per-case metrics, every
integration suffix token and raw timing samples are under
`kernels/projects/a5/gdn_fused_recurrent/evidence/`.

The CCE loop-type warnings are bounded by validated S1..16 and chunk bounds
0..64; the allocator's32-byte compatibility warning concerns physical capacity.
Warnings remain visible with source-based assessments in the evidence README.
Diagnostic board/aclnn launchers remain unqualified. Sim/pipesim evidence is
limited to the two documented cases; full model-grid acceptance is not claimed.
