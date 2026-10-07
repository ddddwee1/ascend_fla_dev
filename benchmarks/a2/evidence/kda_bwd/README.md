# A2-13 KDA backward evidence

D-PM-60 approved a new FP32 cumulative-gate derived unit on 2026-10-07.
Its implementation and fresh native results are in [fp32gc-v1](fp32gc-v1/README.md).
The new unit's real-cache gradients pass the listed cases, while original strict
checkpoint failures remain recorded. Its actual-cache end-to-end limit is128;
formal review is pending and public dispatch remains unqualified. One existing
forward host assertion needs the explicitly requested one-line write-set addition;
the actual checkout retains that failure and the isolated proposal passes.

The results below belong to the **frozen BF16 predecessor**, measured before
D-PM-60. Its accuracy-domain failure and CPU-only repair hypothesis remain
historical evidence and are not reused as qualification of the new unit.

This evidence concerns direct A2 units, not public autograd dispatch.
`CAPABILITIES["a2"]["qualified"]` remains false. For this historical BF16
predecessor, no backward gate-span limit could satisfy both the accuracy budget
and initialization lower bound. The new FP32-gate limit above does not apply
to this predecessor.

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

`precision-seeds-v1/` additionally executes **12 fresh full native chains**:
seeds1/2/3, uniform H32/T128 gates at target spans106.5/128, both block dimensions.
All 12 retain the original gradient failure; all CPU-only FP32-gc replays satisfy
the six whole-gradient budgets. This extends the repair hypothesis beyond seed0,
without claiming that a modified device kernel has passed.

| Seed | Target span | Native dg relative L2 | CPU-only FP32-gc dg relative L2 |
|---:|---:|---:|---:|
| 1 | 106.5 | 0.158323 | 0.029287 |
| 1 | 128 | 0.251085 | 0.032172 |
| 2 | 106.5 | 0.151497 | 0.029257 |
| 2 | 128 | 0.259012 | 0.031614 |
| 3 | 106.5 | 0.149237 | 0.028286 |
| 3 | 128 | 0.229074 | 0.030384 |

All six cross-bd case pairs match in **36 gradient, 36 raw-cache and 54
assembled-cache byte comparisons**. `seed-npu-reference/` separately recomputes
the six full FP32 autograd references with Torch NPU: maximum relative L2 against
CPU FP32 is **2.341916e-7**. The same previously investigated allocation-format
warning is retained in the raw log, with only the project path redacted.

Additional checkpoint failures remain explicit: seed1/span106.5 fails s_base's
allclose check, and seed1/span128 fails k_scaled and s_base allclose checks, at
both bd values. Their relative L2 errors are tiny (1.02e-8, 3.66e-9, 1.87e-11),
but **allclose and relative L2 are conjunctive**; a small normwise error does
not waive those failures. The other four case pairs pass all 33 stage checks.
`allclose-localization/` locates one failing element in each affected stage:
span106.5 s_base native130 versus CPU50.25; span128 s_base native-1.09375 versus
CPU-0.1513671875; and span128 k_scaled differs by one BF16 ULP near an exponent
scaling midpoint. The large intermediate norms mask these local differences in
relative L2. Their exact coordinates, values and original allclose allowances
remain recorded failures; the optional FP64 scalar calculation is diagnostic only.

## Pair-matmul replay diagnosis

`pair-replay/` analyzes the retained bd=1 native tensors from `precision-v1/`.
It separately recomputes each finalize_pair product from its **actual native
finalize_pre inputs**, removing predecessor differences from that comparison.
All computation in this follow-up is on CPU; it launches no custom kernel.

| Target span / product | Native vs actual-input CPU FP32 | Native vs actual-input CPU FP64 diagnostic |
|---|---:|---:|
| 24 / qk_right | 4.768162e-5 | 0 (identical BF16 values) |
| 28 / t_beta | 1.435331e-5 | 8.967541e-7 |
| 32 / t_beta | 9.293656e-5 | 2.836270e-9 |

Numbers are relative L2 after the product's declared BF16 materialization.
The full CPU replay also differs from native finalize_pre in M_qk/M_base/M_beta
by respectively 9/107/98 elements at span24, 1/42/40 at span28 and 6/52/49 at
span32. This separates propagated predecessor rounding from sensitivity to
FP32 matrix accumulation/cancellation. The higher-precision diagnostic is
consistent with accumulation sensitivity contributing to these discrepancies;
it does not prove the absence of a hardware defect or waive the original 1e-5
checkpoint failures. **CPU FP32 remains the acceptance golden.**

The exact analysis source, raw stdout, results and hashes of its retained native
tensor inputs are recorded. Tensor archives stay in private scratch and can be
regenerated with the published full-chain diagnostic below.

`slice-precision/` further localizes the saved bd=1 gradients by head and
64-token chunk. It uses 0.05 only as a diagnostic marker per slice; the assigned
whole-tensor acceptance rule is unchanged.

| Target span | Worst native dg slice | Worst CPU FP32-gc replay dg slice | CPU replay slices over 0.05 / 64 |
|---:|---:|---:|---:|
| 24 | 0.043291 | 0.018648 | 0 |
| 28 | 0.074723 | 0.021057 | 0 |
| 32 | 0.081040 | 0.021917 | 0 |
| 64 | 0.160728 | 0.038550 | 0 |
| 96 | 0.219073 | 0.055960 | 2 |
| 128 | 0.357063 | 0.085526 | 5 |

At span128 all 64 native dg slices exceed the marker. The FP32-gc CPU experiment
improves them substantially but still has localized residual error; its passing
whole-tensor result cannot establish a universal safe domain. The raw summary
stdout reports worst slices and counts for all six gradients. Full per-slice
metrics remain in private scratch with their hash and reproducible source.
`prospective-fp32-gc/` records the proposed ABI, storage and lifetime preflight;
it is a read-only design pending kernel-owner scope, with no implementation,
new compilation or repaired-device acceptance claim.

`pair-npu/` then runs fresh Torch NPU FP32 and BF16 matmuls on the retained
actual finalize_pre inputs at spans24/28/32, with HF32 disabled. All **12 BF16
products are byte-identical to the retained custom-kernel products**. The separate
Torch NPU FP32 products also differ from CPU FP32 in the sensitive cases: e.g.
span32/t_beta relative L2 9.293657e-5 after BF16 materialization. The run completed
24 Torch matmuls, zero custom launches, with healthy pre/post device checks.
These comparisons support an accumulation/rounding explanation for the located
pair-stage discrepancy, rather than uniquely attributing it to custom code.
They do not change CPU FP32 acceptance or resolve the end-to-end BF16-gc error.
The complete raw stream, source, identity and receipt are retained without line
filtering.

## Reproduction

Use the accepted Docker/Python/CANN environment and pins above, with private
machine configuration, fresh health checks, the required device lock and isolated
cache/output directories. Set `PYTHONPATH` to this checkout and the accepted
library checkout; source the machine's CANN environment. Machine paths and device
selection are intentionally external. Run every block dimension in a separate
process; the worker precompiles/registers all 14 vendors before a custom launch.

From the repository root, with `A213_OUT` set to an ignored scratch directory:

```bash
python benchmarks/probe_bwd_span.py --soc a2 --block-dim 1 \
  --a2-suite fixture --a2-out "$A213_OUT/fixture-bd1"
python benchmarks/probe_bwd_span.py --soc a2 --block-dim 1 \
  --a2-suite actual --a2-out "$A213_OUT/actual-bd1"
python benchmarks/probe_bwd_span.py --soc a2 --block-dim 1 \
  --a2-suite span --a2-lengths 128 --a2-seeds 0 \
  --a2-gates uniform fla_initialization \
  --a2-spans 1 8 16 32 64 96 128 160 192 --a2-out "$A213_OUT/span-bd1"
```

Repeat with `--block-dim 2` and distinct `*-bd2` output/cache directories. The
span suite is expected to return nonzero because the recorded accuracy and
finite-cache failures are real. Inspect every receipt row, not only the exit code.
For the located failing case, use `--a2-gates uniform --a2-spans 32`.

After the full workload, reproduce the precision diagnosis and retained tensors:

```bash
python benchmarks/a2/evidence/kda_bwd/precision-v1/precision_diagnostic.py \
  --bd 1 --spans 24 28 32 \
  --out "$A213_OUT/native/precision-v1-bd1/receipts"
cp benchmarks/a2/evidence/kda_bwd/pair-replay/analyze_pair_replay.py "$A213_OUT/"
python "$A213_OUT/analyze_pair_replay.py"
```

The last command reads only those saved tensors and writes CPU diagnosis results
beside the copied script. Run it with the accepted CPU Torch environment and
checkout on `PYTHONPATH`; it does not need a device. Spans64/96/128 are reproduced
by the same full-chain diagnostic with those `--spans` and a fresh output path.
The diagnostic's successful exit means the measurement completed, not that all
gradient/checkpoint comparisons passed; the JSON keeps each failed comparison.
To reproduce slice localization, also generate spans64/96/128 into
`$A213_OUT/native/precision-v2-bd1/receipts`, copy
`slice-precision/analyze_slice_precision.py` beside the other analysis script and
run it in the same CPU environment. It reads both sets of saved native tensors.
For the Torch NPU comparison, copy `pair-npu/pair_npu_diagnostic.py` beside the
analysis scripts and run it in the accepted native environment with a fresh
health check and lock, `--bd 1 --out "$A213_OUT/pair-npu"`. The bd argument
identifies the retained custom run; this diagnostic launches no custom kernel.
For the three-new-seed experiment use
`precision-seeds-v1/precision_seed_diagnostic.py --bd 1 --seeds 1 2 3
--spans 106.5 128 --out "$A213_OUT/native/precision-seeds-v1-bd1/receipts"`,
then repeat in a separate bd2 process with a distinct output/cache directory.
Copy `seed-npu-reference/seed_npu_reference.py` beside the other scratch scripts;
after the full native run completes, invoke it in the accepted native environment
under a health check and lock with `--bd 1 --out "$A213_OUT/seed-npu-reference"`.
It reads the retained bd1 inputs and CPU golden tensors and executes the full
Torch NPU FP32 reference, with no custom kernel launch.
Copy and run `allclose-localization/analyze_allclose_localization.py` beside the
other scratch analysis scripts in the accepted CPU environment to locate those
preserved seed1 checkpoint failures.

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
