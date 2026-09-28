# A2-13 KDA backward qualification — accuracy-domain gap

This evidence concerns direct A2 units, not public autograd dispatch.
`CAPABILITIES["a2"]["qualified"]` remains false. No backward gate-span limit has
been selected: actual-cache real-shape tests passed, but a uniform gate-span
limit cannot currently satisfy the accuracy budget and initialization lower bound.

## Artifact and environment

- Base: `2c5e8f9a342711f99587afed51965713d40bf94f`; task branch `task/A2-13`.
- Runtime pin: library `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`, kernels
  `b3b3f9c16df7c4626ed3c081032a1be5a753d0b1`. Every process checks the imported
  library location, revision, and all 204 tracked runtime source files.
- Ascend910B3, 20 cube / 40 vector cores; CANN 9.0.0 compiler and OPP timestamp
  `20260428_134817545`; installed built-in OPP packages `ascend910b`,
  `ascend910_93`, `config`.
- Python 3.11.15, Torch 2.10.0+cpu, torch_npu 2.10.0; HF32 disabled.
- bd=1 and bd=2 run in separate processes. Each process registers all five
  forward and nine backward vendors before the first custom launch. Device
  health was checked before and after each completed run, under scoped locks.
- Full CPU host baseline: 1454 passed / 6 skipped. This is host regression,
  not device qualification.
- Full host regression after the test/benchmark changes: **1454 passed / 6
  skipped**, 241.89 seconds; raw logs and hashes are under `host/`.

## Independent backward fixtures — completed on hardware

All 26 declared cases passed: 12 at bd=1 and 14 at bd=2, with **156 gradient
checks and 858 intermediate checks** (33 checkpoints per case). Each case
executes the complete nine-kernel backward chain once; final and intermediate
outputs come from that same execution.

| Gradient | Worst relative L2 | Original fixture budget |
|---|---:|---:|
| dq | 0.038291209 | 0.05 |
| dk | 0.111544023 | 0.15 |
| dv | 0.003886758 | 0.05 |
| dbeta | 0.007456483 | 0.05 |
| dg | 0.177647183 | 0.25 |
| dh0 | 0.005292979 | 0.05 |

The fixture oracle uses CPU FP32 autograd with the contract's BF16 returned
gradients. These numbers **do not establish actual-forward-cache end-to-end
accuracy**. The latter has a separate 0.05 budget for every gradient, including
dk and dg. Fixture allclose and relative-L2 checks are both required; all original
stage overrides are retained, including the four finalize_pair products.

`fixture/bd*/events.jsonl` is the exact raw process stdout/stderr, without filtering
(each line is a JSON event). `identity.json` records toolchain and source hashes;
`receipt.json` contains every result. The matching probe source is archived by
SHA256 in `fixture/`; `fixture/manifest.json` checks those bytes.

## Actual forward cache protocol

The initial actual-cache suite passed **18/18 hardware cases** (nine per bd):
four small cache foundations, H=HV=32 at T=64/128/512/4096, and GQA H=16/HV=32
at T=192. The foundations include C=1, C=2, odd C=3, multiple GQA groups and B=2.
Both block dimensions produced identical bits for all **54 gradient pairs,
54 raw-forward-cache pairs and 81 assembled-cache pairs**.

| Gradient | Worst actual-cache relative L2 | End-to-end budget |
|---|---:|---:|
| dq | 0.006032422 | 0.05 |
| dk | 0.009714919 | 0.05 |
| dv | 0.003728860 | 0.05 |
| dbeta | 0.003833977 | 0.05 |
| dg | 0.005193044 | 0.05 |
| dh0 | 0.002825413 | 0.05 |

`actual-v1/` contains per-case CPU FP32 and Torch NPU FP32 comparisons, raw
events, exact source identities, source snapshots and the cross-bd summary.
These results establish the listed cases; they do not yet establish the full
gate-span domain. In particular, the range sweep additionally uses a distribution
where all heads have substantial decay, alongside calibrated layer initialization.

PM approved test-only D2H/CPU/H2D assembly in issue #40 comment 5875827454.
The test reads complete forward device outputs before transforming their layout.
It records separate hashes for the six original outputs and nine assembled
caches, and verifies a bitwise roundtrip for each of the six direct values.
It never invokes the A5 layout runtime.

- Aqk, Akk, w, u, unscaled qg and kg: actual A2 forward outputs, followed only
  by deterministic CPU head-major to token-major layout conversion.
- g_cumsum: actual gate FP32 natural-log cumulative values multiplied by
  `1/ln(2)`, then stored as BF16. No `log2(eg)` reconstruction after underflow.
- h and v_new: the explicitly permitted FP32 `_scan_states` reconstruction
  from actual w/u/kg/eg, with BF16 stored outputs. These are not literal
  captures of the recurrent kernel's BF16 ring states.

Assembly and upload costs are identified as test fixture costs. No timing from
this path is a claim about public runtime performance or zero-copy training.
The end-to-end golden uses the original input values and CPU FP32 recurrence
and autograd. Segment replay applies the full reverse chain rule and has been
cross-checked against one full autograd graph. A separate Torch NPU FP32 execution
checks the reference on device; no BF16 cumsum-difference oracle is substituted.

## Initialization lower bound — generated inputs, not hardware acceptance

`initialization-distribution.json` records eight freshly generated untrained
Kimi-layer initialization seeds at B=1, T=4096, H=HV=32, K=V=128.
The maximum FP32 span is **106.38469696044922**, and the maximum after the
backward input ABI's declared BF16 fixture quantization is **106.3828125**.
These are distribution measurements only; an acceptable backward limit must
also pass actual-cache device accuracy checks above this lower bound.

## Range results — finite does not imply accurate

`range-coarse-v2/` completed 36 native test points (18 per bd), H=HV=32, T=128,
seed=0, using both calibrated layer initialization and uniform negative gates.
There are **22 passing points, 10 finite gradient-budget failures, and four
nonfinite forward-cache failures**. Both block dimensions have identical input,
raw cache, assembled cache and available gradient byte hashes for all 18 pairs.

| Target span | Uniform dg relative L2 | Calibrated initialization dg relative L2 |
|---:|---:|---:|
| 1 | 0.004841 | 0.003824 |
| 8 | 0.017072 | 0.004014 |
| 16 | 0.032551 | 0.004021 |
| 32 | **0.063174** | 0.004275 |
| 64 | **0.122838** | 0.004522 |
| 96 | **0.137940** | 0.004888 |
| 128 | **0.244574** | 0.005222 |
| 160 | **0.263902** | 0.005596 |
| 192 | nonfinite forward caches; backward not launched | same failure stage |

The budget is 0.05 for every end-to-end gradient. The uniform span-32 point
has actual BF16 input span **32.017578125** and dg error **0.06317371366342159**;
all six gradients and 33 backward checkpoints remain finite. A max-span gate
covering the measured initialization lower bound would also admit this failing
input. Successful initialization samples therefore do not justify filling a
single backward limit. `qualified` stays false and no DONE is claimed.

The first coarse run (`range-coarse-v1/`) stopped at span 192 because its test
layout roundtrip used value equality, which treats NaNs as unequal. The original
exception and preceding numerical failures are retained. The corrected checker
compares contiguous bytes, including NaN payloads and signed zero; invalid
forward caches are recorded as failures before the backward unit's finite-input
boundary. This retains the failed point and allows subsequent cases to execute.

## Located precision experiment — not a repaired device kernel

`precision-v1/` and `precision-v2/` rerun the complete native forward/backward
chain at uniform target spans 24/28/32/64/96/128, at both block dimensions.
The pure CPU stage replay first consumes the exact native-produced BF16 caches.
A second CPU-only replay changes just g_cumsum to the actual forward FP32 value
times `1/ln(2)`, without its BF16 narrowing. All other caches remain identical.

| Target span | Native dg error | Same-cache CPU replay dg error | CPU-only FP32-gc replay dg error |
|---:|---:|---:|---:|
| 24 | 0.036355 | 0.036941 | 0.014029 |
| 28 | 0.053919 | 0.054420 | 0.014947 |
| 32 | 0.063174 | 0.063779 | 0.016146 |
| 64 | 0.122838 | 0.123686 | 0.022070 |
| 96 | 0.137940 | 0.139080 | 0.026545 |
| 128 | 0.244574 | 0.245634 | 0.030554 |

Every CPU-only FP32-gc replay gradient is within 0.05 at these six points. This
locates BF16 cumulative-gate narrowing as a major contributor and motivates a
mixed-precision cache ABI investigation. It **does not** establish that changing
only that ABI will pass on hardware: no modified kernel was compiled or executed.
The current write-set excludes kernel/unit/contract changes.

The same-cache native/replay intermediate checks also expose failures of the
existing finalize_pair 1e-5 bound at normalized-input spans 24/28/32 (allclose
still passes). Worst among these is t_beta relative L2 9.2949e-5 at span 32.
They are retained as failures, not waived; the original 26 fixture cases remain
fully passing. At spans 64/96/128 all 33 replay checkpoint checks pass. The
observed span-24 success and span-28 failure are individual measured points,
not proof of a monotone boundary or permission to set a lower gate.

The four native cumulative-gate consumers are scan_fused, inverse_epilogue,
finalize_pre and finalize_post. A prospective precision repair must explicitly
settle their GM/UB types, unit/cache ABI and contract/reference behavior, retain
all existing budgets, and repeat full hardware qualification. Such a repair
requires the kernel owner's assigned scope; these qualification changes do not
silently alter those frozen sources.

## Torch NPU format notification

Actual-cache/range raw logs include a Torch NPU notification from `zeros_like`
in the independent reference. All lines are retained in `raw.redacted.log`;
only private paths/process identifiers are redacted. No warning filter or
configuration change was used.

An isolated device probe reproduced the notification with **zero custom launches**:
input/output format codes were both 0, shape/dtype were preserved, inputs were
unchanged and the returned zeros matched CPU bytes. The package-reported revision
locates the notification in allocation code that selects a base format, before
allocating the tensor ([Torch NPU source](https://github.com/Ascend/pytorch/blob/94f8a8e6b523d7ba553e1b80d5b5248478391526/torch_npu/csrc/aten/common/TensorFactories.cpp#L301-L319)).
`format-warning/` records the probe, source identity and the measured CPU/NPU
reference agreement. This is an allocation-format notification in the reference;
it is neither a hidden custom-kernel fallback nor a performance qualification.
