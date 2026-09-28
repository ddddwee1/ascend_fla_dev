# PGDN FP32 backward: ABI, classification and range contract

## Historical dense-clamp failure and repair qualification

Commit `c8e707337e0734113fa1c3a457c5318b4413dd48` fails a supported
normalization boundary omitted from the original grid. At B1/T64/H1/HV2,
FP32, all three cotangents, random dense q/k rows are normalized and scaled
in FP32 to the clamp neighborhood. The frozen A/B classifier is unchanged.
`diagnose_dense_clamp.py` reproduces q-only, k-only and both variants; it is a
located diagnostic after the original complete native workloads, not a pass
condition or a substitute for full acceptance.

| Case | q branch differences / 64 | k branch differences / 64 | Ordinary dq relative L2 vs A | Ordinary dk relative L2 vs A |
| --- | ---: | ---: | ---: | ---: |
| q | 5 | 0 | 0.030137952029148613 | within budget |
| k | 0 | 6 | within budget | 0.01377006123545162 |
| both | 5 | 6 | 0.03013795074261903 | 0.013770060136453276 |

A and B satisfy the frozen ordinary budget on all three cases. Public and
composition outputs are byte-identical. Every actual gradient remains finite;
ordinary errors exceed 1e-4 and are not reclassified as disclosure-only.
Evidence: `evidence/native/dense-clamp-bd2-v3/diagnostic.json`, complete indexed
public disclosures, source manifest and execution receipt under the unit.
The driver returned zero and health checks were good, but the post-run empty
condition was not met. During execution, 83 context samples included 73 with
the owned process, two known owned-context retirement samples and zero
unexplained foreign-context samples. This is a failed numerical diagnostic,
not an exclusive acceptance or performance claim.

The located first boundary is raw norm reduction in `kernels/atk.py`, whose
`cadd` order differs from the selected literal CPU implementation before the
`>= 1e-12` VJP branch. The literal Torch source at commit
`7661cd9c6b841b62b7f411aa52ec51f05457263b`,
[`ReduceOpsKernel.cpp`](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp#L222),
uses vector-lane accumulation followed by sequential lane addition. The
accepted CPU environment reports AVX2. A separate CPU diagnostic over 8,192
rows reproduces every literal norm bit with eight lanes and separately rounded
multiply/add; fused or four/sixteen-lane variants differ. This is source and CPU
evidence for repair design, not a native repair claim. Budgets, classifiers,
near-zero input support, and the six frozen reference files remain unchanged.

The current repair replaces only the task-owned ATK kernel's raw-norm
reduction with the literal eight-lane FP32 schedule. `Tensor.brcb` emits
`E2B_B32`, reading eight values and broadcasting each to its 32-byte block.
Sixteen separately rounded multiply/add steps precede eight sequential scalar
loads/additions and square root. Its 64-element scratch is fully initialized
and published before scalar loads; stage1 UB is 140032 bytes / 10 buffers.
All four stages emitted for bd1/bd2, with empty static event-balance diagnostics.

The repaired first B1/T4096/H=HV8/mask7 case at bd1 satisfies all required
checks; ordinary maximum relative L2 is 6.551130529696604e-7. The three original
dense q/k/both failures at bd2 now have zero branch differences and satisfy
ordinary budgets against both references. The evidence labels are
`norm8-full-first-bd1-v1` and `norm8-dense-clamp-bd2-v1`. These were intermediate checks; complete qualification is recorded below. Both runs were healthy with empty before/after checks;
recorded during-run context sampling contains no unexplained foreign process.
These runs used the earlier task lock convention. Subsequent complete matrices
and measurements also acquire the canonical shared lock located in the ignored
machine configuration. Lock identifiers are not public artifact data.

The literal comparison here uses the explicitly recorded Torch2.12 AVX2 builds;
bitwise norm-branch agreement is not claimed for unmeasured binary builds or
CPU dispatch implementations. A matching Torch git and reported AVX2 capability
alone do not establish the same literal floating-point execution. A new 73-case supplemental grid covers all seven cotangent
subsets, dense q/k/both modes, multiple seeds, grouped heads, multiple chunks,
one-ULP-adjacent clamp radii and a full T4096 case. It uses the same classifier,
budgets, actual public output, independent-leaf and stage checks as the original
grid. Complete bd1/bd2 byte comparison rejects missing or mixed grids. The six
frozen reference files are unchanged. Host regressions: 1358 passed, 10 skipped,
5 existing CPU-stub warnings; targeted task regressions: 119 passed.

Earlier qualification and timing records remain under their original evidence
labels and source revision; they are historical records. The repaired artifact has now completed both full grids and new same-card measurement, as recorded below.

PK-05 implements a standalone A5 backward entry; forward dispatch and autograd
wiring are outside this task. The source authority is FLA
`e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`,
`fla/ops/precond_gated_delta_rule/naive.py`, SHA256
`3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8`.
The literal FP32 oracle is never modified. Native qualification is complete with numerical disclosures (nonPASS).

## Frozen public ABI

`chunk_pgdn_bwd(q,k,v,g_atk,g,beta_atk,beta,do=None,dht=None,dA_T=None,...)`
returns `(dq,dk,dv,dg_atk,dg,dbeta_atk,dbeta)`, in differentiated-input order.
Every input, cotangent and gradient is contiguous FP32 on one device. BF16 is
explicitly rejected; there is no host widening or fallback.

| Item | Supported value |
|---|---|
| Shape | B=1; positive T divisible by64, T<=4096; K=V=128; H>0, HV>0, HV%H=0 |
| q/k, v/do | [B,T,H,K], [B,T,HV,V] |
| g_atk/beta_atk, g/beta | [B,T,H], [B,T,HV] |
| dht, dA_T | [B,HV,K,V], [B,H,K] |
| Cotangents | All seven nonempty subsets of do/dht/dA_T; absent means zero |
| Initial states | Both None; zero main and ATK states; no initial-state gradients |
| Constants | scale=128**(-.5), x=1.5, eps_atk=1e-6, center=-.2 |
| Normalization | Mandatory FP32 naive x/max(norm(x),1e-12), including zero/near-zero rows |
| Layout | Token-major inputs, K-major main state; no varlen, CP or transpose mode |
| Device | A5 CCE, block_dim1/2, explicit inprocess/aclnn/board launchers |
| Values | Finite inputs/cotangents; g/g_atk<=0 and finite FP32 chunk sums; beta/beta_atk in[0,1] |

Value validation synchronizes NPU predicates and belongs to public API timing.
The host only validates, allocates, binds and launches. Normalization, state
preparation, grouping and absent-cotangent initialization belong to kernels.

## Independent adjoint and local scales

For one token and value head, let `p=scale*normalized(q)`, `u=normalized(k)`,
`w=u*M`, `a=exp(g)`, `D=a*Sprev`, `r=v-u^T D`, `z=beta*r`, `S=D+w*z^T`.
For incoming state adjoint and output cotangent, `G=dS+p*do^T`:

- `dp=S*do`, `dw=G*z`, `dz=G^T*w`, `dr=beta*dz`.
- `du_read=-D*dr`, `dv=dr`, `dbeta=dot(dz,r)`.
- `GD=G-u*dr^T`, `dg=sum(GD*D)`, `dSprev=a*GD`.

The read key `u` remains in the correction; the write key `w` remains in `dz`.
Consecutive value-head contributions are summed in ascending order before the
shared ATK adjoint. With `s=log(A+eps_atk)-center` and
`M=exp(-log(x)*s/(1+abs(s)))`, its injected adjoint is
`dM*(-log(x)*M)/((A+eps_atk)*(1+abs(s))^2)`, where `dM=u*sum_group(dw)`.
After adding it to incoming `Z`,
`dg_atk=sum(Z*exp(g_atk)*Aprev)`, `dbeta_atk=sum(Z*u^2)`,
`du=sum_group(du_read)+M*sum_group(dw)+2*beta_atk*u*Z` and
`Zprev=exp(g_atk)*Z`. Final `dA_T` is seeded once per key head.

The following local scales are evaluated in the independent FP64 analytical
reference, without any candidate output. Each sum uses the displayed terms,
not a global maximum or a bound propagated from unrelated output coordinates.

| Gradient element | Local scale L_i |
|---|---|
| dq/dk, selected norm branch | `(|h_i| + |y_i*dot(y,h)|) / norm(x)`; h is its complete grouped pre-normalization cotangent, y=x/norm(x) |
| dq/dk, below clamp | `|h_i| / eps_norm` |
| dv_j | `sum_k |beta * G_kj * w_k|` |
| dbeta | `sum_j |dz_j * r_j|` |
| dg | `sum_kj |GD_kj * D_kj|` |
| dg_atk | `sum_k |Z_k * exp(g_atk) * Aprev_k|` |
| dbeta_atk | `sum_k |Z_k * u_k^2|` |

For q, h already includes scale exactly once. For k, h includes both read/write
paths and ATK. These formulas and the ascending group order are frozen in
`ref/reference.py`; `ref/classification.py` applies the following rule.

## Element classification and evidence

The executable interpretation is approved by
[PM NOTE5842101164](https://github.com/ddddwee1/ascend_fla_dev/issues/93#issuecomment-5842101164).
Each element with algebraically zero derivative, or
`abs(reference64_i) <= 64 * 2^-23 * L_i`, is disclosure-only. All others are
ordinary. Algebraic reason codes cover normalization's axis radial nullspace,
zero initial-state gate derivatives, missing main cotangents and missing output
cotangents. Other computed zeros fall under the explicit local-threshold rule;
a floating-point zero alone is not labelled an algebraic proof.

For the classifier, the active normalization branch is selected using the
literal FP32 `norm>=1e-12`. FP64 arithmetic evaluates that selected branch,
using norm64 on the active side and the lifted FP32 epsilon on the clamped side.
It does not select the branch again in FP64. In particular the stored FP32
`1e-12` is slightly below decimal FP64 `1e-12`, but its literal FP32 branch is
active. Known algebraic zeros are canonicalized to zero with their raw FP64
residual retained. FP64 mathematical gradcheck separately uses the all-FP64
formula away from the kink; it does not prove a derivative by crossing the kink.

For each gradient, ordinary elements keep relative L2<=1e-4 against both
literal FP32 A and independent FP32 B. The same mask applies to both. Empty
ordinary subsets are identified by their zero count. Raw whole-gradient
relative L2/max_abs remain available, including their original failures.

Disclosure has no pass threshold. For every disclosed element the evidence
records position, candidate value, FP64 reference and raw reference, local scale,
threshold, reason, absolute candidate value, absolute error and
`absolute_error/norm(entire_FP64_reference_gradient)`. A zero global denominator
is recorded as undefined for zero residual or infinity for nonzero residual;
no denominator epsilon is inserted. Presence of disclosure is reported as
**numerical disclosure complete, nonPASS**, separately from the ordinary budget.

The JSONL disclosure format is lossless: each gradient header gives shape and
row-major ordering; each `start,count` record supplies the exact common payload
for every index in that interval. It merges only adjacent identical payloads,
including identical floating-point bits and reason. It is not sampled evidence.
Signed zero, complete coordinate coverage and zero-denominator handling have
explicit regression checks. Receipts carry record count, byte size and SHA256.

## Range analysis and validation boundary

Real-arithmetic normalized key norm is at most1. With zero initial ATK state,
nonpositive gates and beta_atk in[0,1], `0<=A_t<=t<=4096`. Thus A+eps_atk is
positive, M stays within `(2/3,1.5)`, and the squash derivative has no singularity
at s=0. Main/ATK reverse only multiply by nonpositive-gate exponentials; they
never invert a decay, beta or a rank-one state update. The normalization
Jacobian norm is bounded by1e12 in real arithmetic and can have an exactly zero
radial component. This bound explains large valid derivatives and cancellation;
it does not license clipping them or excluding small input norms.

These facts are not a proof that every combination of arbitrary finite FP32
magnitudes yields finite states/gradients. The complete measured range and
native evidence must be reported separately before declaring acceptance.

Current qualification: six independent A-lift/B-analytical FP64 gradchecks
passed (ratios1/2; normal, below-clamp and above-clamp). Complete256-case classified CPU calibration satisfies all ordinary A/B budgets;
maximum relativeL2 is2.9902082636512436e-6 (c3_h1_hv2_m7, dg_atk, B_against_A).
All six frozen reference/classifier source hashes are unchanged. Complete512
A/B disclosure lists are retained as plain JSONL with content hashes in the unit
evidence directory; overall numerical status remains nonPASS.

Four CCE stages emitted for each of bd1/bd2 and static balance checks report
no problems. Repaired UB allocations are140032/67136/135808/4832bytes. Native results and observed whole-chain ranges are reported below, separately
from these CPU/emission results. No simulator, CUDA/Triton or weight validation
is claimed.

The separate FP32 stage reference measured seven cases: full T4096 equal-head
and ratio8, full gate-zero/weak-decay, and the three axis normalization clamp
cases, each with all three cotangents. These are CPU observations, not device
ranges or bounds for arbitrary input magnitude. Every observed intermediate
was finite; the complete per-case table is `evidence/cpu-ranges.json`.

| Observed quantity | Range across those seven cases |
|---|---|
| Raw q norm | 4.99999998e-13 to0.71367657 |
| ATK A | 0 to19.513712 |
| Multiplier M | 0.73472625 to1.45895875 |
| Decayed main state D | -0.17093094 to0.16487035 |
| Residual v-u^T D | -0.28586960 to0.27640030 |
| Main adjoint G including output cotangent | -0.44471529 to0.47014689 |
| ATK adjoint Z including write-key injection | -0.27805093 to0.36575362 |
| Query cotangent before normalization | -0.01474613 to0.01592194 |
| Key cotangent before normalization | -0.30100784 to0.23552959 |
| Reciprocal clamped q norm | 1.40119493 to9.99999995904e11 |
| Final dq absolute maximum | 6.938435584e9 |
| Final dk absolute maximum | 2.35529584640e11 |

The large finite q/k derivatives arise from the supported clamped normalization
region. They are retained without clipping; the ordinary/disclosure rule is
unchanged. Native stage ranges are reported separately from actual returned
tensors in each device case's report.

Input receipts distinguish environments: the host calibration used Torch2.10,
and actual device qualification generates its inputs and both CPU references
inside the accepted Torch2.12/torch_npu2.12 environment. For the first full case,
normal-random tensor hashes differ between the two environments while the
uniform-derived gate/beta hashes agree. A seed is therefore not treated as a
cross-environment byte identity. Every native result is compared to freshly
computed A and B for its actual inputs; bd1/bd2 comparison separately requires
identical input hashes before asserting identical output bytes.

## Repaired artifact: complete native matrices

The earlier repaired-source bd1 run (`norm8-matrix-bd1-v1`) completed numerical
checks, but1482 context samples included570 unexplained foreign samples from
unrelated processes. Both pre/post idle and health checks succeeded; its
`qualification-scope.json` excludes exclusive acceptance and performance.
The first repaired-source bd2 run (`norm8-matrix-bd2-v1`) also completed
numerical checks, but19 unexplained foreign samples began before its final
report and post-run idle failed. It is likewise excluded from acceptance and
performance. Two initial secondary attempts failed before device execution
because the compiler Python environment lacked a vendor-declared dependency;
those source receipts and the first errors are retained as diagnostics. The
next attempts found a header-definition conflict in CANN9.2; the independently
installed CANN9.1 toolchain was selected without changing pinned or generated
source. That attempt compiled and packaged the first kernel but stopped at its installer's
read-only log directory; new device containers then failed the DCMI precheck.
All failed attempts have explicit non-acceptance scope. The qualified artifact
is built in a device-free Docker and executed in the existing device Docker
using the unchanged pinned source-cache validation. Every installed vendor
file, harness executable and source stamp is hashed before and after actual
execution; build and device receipts remain separate.

The following qualification uses only independently repeated secondary runs:
`norm8-secondary-matrix-bd1-v5`, `norm8-secondary-matrix-bd2-v5`, and
`norm8-secondary-clamp-bd1-v1` / `norm8-secondary-clamp-bd2-v1`. Their actual
compiler, chip, Python and Torch identities are recorded in each receipt.
The secondary CPU precheck originally used elementwise Tensor.sqrt to model
norm's internal scalar sqrt. `norm8-cpu-environment-diagnostic.json` and the
reproducible `diagnostics/norm8_cpu.py` retain that mismatch and locate the
probe error:8192 rows have identical actual literal norm and scalar sqrtf bits
in the two measured builds. No production/reference/budget change followed
from that probe correction. CUDA was not executed.

Actual A5/CCE execution completed the original256 and supplemental73 cases at
each block dimension1/2. All20 stages, independent leaves, composition, actual
public output, clamp branch bits and input immutability meet the required
checks. Both complete grids have identical input hashes and byte-identical
returned stages and public gradients across bd1/bd2. The largest ordinary
relativeL2 against either A/B reference is 4.1356284098271835e-06.

| Public gradient | Maximum against literal FP32 A | Maximum against analytical FP32 B |
| --- | ---: | ---: |
| dq | 3.94661822e-07 | 3.86331842e-07 |
| dk | 4.27346014e-07 | 3.32822648e-07 |
| dv | 4.29040179e-07 | 4.28473293e-07 |
| dg_atk | 4.13562841e-06 | 2.90040605e-06 |
| dg | 1.19273243e-06 | 1.20249031e-06 |
| dbeta_atk | 1.17760375e-06 | 6.05400248e-07 |
| dbeta | 9.72447074e-07 | 8.92385031e-07 |

Public disclosure elements: bd1=67662570, bd2=67662570. Every element
position and payload is retained in the complete per-case JSONL. No relative
pass threshold is applied to disclosures, and overall status remains
**numerical disclosure complete, nonPASS**. The four norm8 summary JSON files
include complete per-gradient/reason counts and observed device ranges; these
are observed ranges on the listed grids, not bounds for arbitrary FP32 values.
`norm8-source-consistency.json` links delivered production/frozen-reference
hashes to every repaired execution. Legacy evidence is retained with its
original source identity. The explicit aclnn/board transport launchers remain
untested; these results use the actual inprocess native launcher.

Observed device values across both complete grids and block dimensions:
these are measured ranges, not bounds for arbitrary finite inputs. Every listed
tensor has zero observed nonfinite elements. Exact per-case values remain in
the numerical reports.

| Stage tensor | Observed minimum | Observed maximum |
| --- | ---: | ---: |
| q_norm | -0.443489342928 | 1 |
| k_read | -0.423209398985 | 1 |
| k_write | -0.531184971333 | 1.20781362057 |
| A_history | 0 | 19.5137138367 |
| q_raw_norm | 0 | 2 |
| k_raw_norm | 0 | 2 |
| final_A_state | 0 | 19.5137138367 |
| checkpoints | -0.166130647063 | 0.15358954668 |
| final_state | -0.121090173721 | 0.120301358402 |
| tape | -0.130987972021 | 0.133448287845 |
| dq_norm_parts | -0.014746136032 | 0.0159219373018 |
| dk_read_parts | -0.104225963354 | 0.109108775854 |
| dk_write_parts | -0.226547166705 | 0.195417672396 |
| dv | -0.405002593994 | 0.371921092272 |
| dg | -0.578243970871 | 0.837932705879 |
| dbeta | -0.195531517267 | 0.225399181247 |
| dq | -9935822848 | 13162290176 |
| dk | -326637846528 | 402086494208 |
| dg_atk | -26.8955955505 | 16.469461441 |
| dbeta_atk | -3.18423056602 | 2.5711171627 |

## Repaired artifact: same-card measurement

B1/H=HV8/K=V128, FP32, bd2. Each leg uses10 warmups and50 samples, synchronized
immediately before and after each public call. Wall latency includes input
validation and allocation. Canonical per-device and task-private locks plus before/after
healthy/idle checks cover each run, with no unexplained foreign context in
recorded samples. CPU references finish before timing; all rounds retain
before/after classified numerical checks and disclosures.

The baseline caches this candidate's normalization/ATK tapes and main
checkpoints. It excludes stage1/2 device computation but retains validation
and all allocations, including unused forward outputs. The candidate performs
all four stages. This records regeneration cost, not another complete backward
backend or a speedup claim. Every sample is retained in
`evidence/native/norm8-secondary-timing-bd2-v1/measurement.json`.

| T | Round | Baseline before median ms | Complete backward median ms | Baseline after median ms |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 11.062093 | 12.980858 | 11.069886 |
| 1024 | 2 | 11.080801 | 12.980252 | 11.071163 |
| 1024 | 3 | 11.083341 | 12.980973 | 11.080091 |
| 4096 | 1 | 46.451332 | 55.668111 | 46.430295 |
| 4096 | 2 | 46.450776 | 55.677931 | 46.446940 |
| 4096 | 3 | 46.447205 | 55.720745 | 46.434281 |
