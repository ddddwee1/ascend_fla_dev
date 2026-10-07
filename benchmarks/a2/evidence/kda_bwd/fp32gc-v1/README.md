# D-PM-60 FP32 cumulative-gate candidate

The new `kernels/projects/a2/kda_bwd_stable_fp32gc` unit removes BF16
materialization of `saved.g_cumsum` in four consumers. The other five kernels,
eight cached values, math order, synchronization protocol, CPU FP32 golden and
numerical thresholds are unchanged. The frozen BF16 unit remains available.
D-PM-60 authorization is issue #40 comment 6033288361.

This is an in-progress qualification with retained failures. Public dispatch
remains unqualified, no backward capability limit is set, and no DONE is claimed.
The working candidate is identified by `candidate-source-manifest.json` and each
run's source manifest; the earlier PR head alone does not identify this artifact.

## Completed stages

| Experiment | Observed result | Scope |
|---|---|---|
| Host regression | 1454 passed / 6 skipped | Host only |
| Vendor compilation | 5 forward + 9 backward at each bd=1,2 | Compilation only, zero custom launches |
| CPU ABI checks | FP32 gc accepted; wrong gc/cache dtype and NaN rejected | Other eight fixture caches unchanged bytewise |
| Independent fixture | 26 executed; 23 cases pass all checks | All 156 final-gradient checks pass; 3 intermediate checks fail |
| Actual-cache complete chain | 18/18 pass | CPU FP32 six-gradient budget 0.05 plus separate Torch NPU FP32 reference |
| Actual-cache gate sweep | 48 executed: 44 pass, 4 nonfinite-forward failures | Two distributions, both bd, seed0, T128, targets 1–192 |
| Long-sequence span128 | 12/12 pass | Both profiles, T64/512/4096, both bd; worst dg0.0385710521 |
| Seed0 checkpoint diagnostics | 14/14 end-to-end pass; 0/14 pass all original checkpoints | Seven spans24–128, both bd; failures retained |
| New seeds | 12/12 end-to-end pass | Uniform seeds1/2/3, T128, targets106.5/128; 6/12 pass all original stage checks |
| Reduced leaf pipesim | 10/10 pass | Two small shapes, five leaves, bd1; balance, hazards and deadlock checked |
| Reduced leaf functional sim | 10/10 pass after driver correction | Original list-return handling error and rerun retained separately |

Native environment is Ascend910B3 (20 cube / 40 vector cores), CANN9.0.0
compiler and OPP timestamp20260428_134817545, Python3.11.15,
Torch2.10.0+cpu/torch_npu2.10.0, Ascriptor0.1.0, HF32=False. Accepted pins are
library90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5 and
kernelsb3b3f9c16df7c4626ed3c081032a1be5a753d0b1; each native process checks
204 imported runtime files. All native runs use separate bd processes, complete
5+9 vendor preparation before custom launch, and fresh health checks/scoped locks.

## Actual-cache evidence

The 18-case suite covers cache foundations (C1/C2/odd C3, GQA and B2),
H=HV32 at T64/128/512/4096, and H16/HV32 at T192. All 54 gradient, 54 raw-six
cache and 81 assembled-nine cache cross-bd comparisons are byte-identical.
Worst six-gradient relative L2 values are dq0.0041884305, dk0.0043337526,
dv0.0037101211, dbeta0.0037606116, dg0.0049404286, dh00.0024671953.

Six caches are actual forward outputs with the approved test-side layout
conversion. The other three retain their explicit provenance: cumulative gates
are actual FP32 natural-log gates multiplied once by 1/ln2 and retained FP32;
h/v_new are FP32 `_scan_states` reconstructions with BF16 materialization.
These are test assembly costs, not a claim of literal recurrent-ring capture,
zero-copy public training, or kernel performance.

The 48-point sweep uses uniform gates and calibrated FLA initialization.
Uniform dg errors at target spans32/106.5/128/160 are respectively
0.0140062652/0.0199057010/0.0205367667/0.0495373367. Span160 has little budget
margin. At192 both distributions and both bd produce nonfinite forward caches;
backward is rejected before launch. All cross-bd bytes agree, including 132
available gradient, 144 raw-six and 216 assembled-nine comparisons. Detailed
input spans and each gradient's finite/accuracy fields remain in the receipts.

The new-seed suite has worst dg0.0213835663. All 36 gradient, 198 checkpoint,
36 raw-six and 54 assembled-nine comparisons agree bytewise across bd.
Its fresh Torch NPU FP32 autograd reference differs from CPU by at most
2.341915836e-7 relative L2. The independently rerun eight-seed initialization
lower requirement is FP32 span106.38469696044922 (BF16 input ABI106.3828125).
Input distributions alone are not device qualification.

## Original checkpoint failures remain failures

- `gentle_decay`, both bd: `finalize_pair.qk_left` has one differing BF16
  element, index[0,0,47,9], native4.380941390991211e-6 versus
  CPU4.351139068603516e-6. The residual is2.9802322387695312e-8 and relative
  L2 is1.6591333747983417e-5, exceeding1e-5. Torch NPU BF16 and FP32 matmul
  from the same native operands match the custom output bytewise. This
  localizes an accumulation/rounding difference without replacing the golden.
- `grid_c1_hv2`, seed20260903: `finalize_pre.kg` index[0,0,0,41,122] is
  native2032 versus CPU2040. Whole-output relative L2 is4.396917174e-13,
  but allclose fails; the small global ratio does not waive that check.
- The same two inputs were repeated twice at each bd. All132 intermediate
  output comparisons across bd agree bytewise. The original grid cases use
  different seeds across bd; the focused comparison deliberately holds input
  bytes fixed.
- Seed1/span106.5 fails k_scaled/qk_left/s_base checkpoints; seed3/span106.5
  fails q_scaled/k_scaled/kg/s_base; seed3/span128 fails q_scaled/kg. These
  failures recur at both bd. All original metrics and local head/chunk
  diagnostic markers are retained; whole-gradient passes are not presented
  as checkpoint passes.

## Located exponential rounding boundary

A fresh Torch NPU experiment makes zero custom launches and replays the three
scaling products from the exact stored gate/input values in fourteen retained
cases. All 42 built-in results match the native scaling outputs bytewise.
For the fixture `kg` failure, exponent10.3681640625 gives CPU FP32
exp31829.986328125 and NPU FP32 exp31829.984375. Multiplying by
0.06396484375 gives CPU2036.0001220703125 versus NPU2036.0; the BF16
rounding boundary produces2040 versus2032. This locates the difference in
FP32 exponential rounding followed by BF16 materialization. It does not
justify weakening allclose or replacing the CPU golden. The raw inputs,
intermediate numbers and other cases are in `fp32gc-exp-localization-v1-bd1/`.

## Model scope and remaining work

Models run only after the complete native workloads. Reduced captures are
B1/H1/HV1/C2 and B1/H1/HV2/C1 at bd1, preserving chunk reuse or repeated heads,
for scan_fused, inverse_mm, inverse_epilogue, finalize_pre and finalize_post.
The captured actual arguments and outputs are retained privately with hashes.
Independent CPU FP32 fixture stages are regenerated during each model run.
Model timings/cycles are not hardware performance or a full-domain proof.
The first functional driver incorrectly treated a list of outputs as one
tensor; its raw exceptions remain in models-v1, and the corrected list handling
is isolated in models-v2. No kernel or comparison threshold changed.

All listed native suites have completed. Long-sequence span128 gives worst
relative L2 dg0.0385710521, with36 gradient,36 raw-six and54 assembled-nine
byte-identical cross-bd comparisons. Seed0 checkpoint diagnostics retain
strict failures at every tested span24/28/32/64/96/106.5/128, while all six
end-to-end gradients pass. Their42 gradient and231 checkpoint comparisons
are byte-identical across bd. Contract support now records compile/reference
passed, full fixture board failed, and full-unit model stages untested
(the reduced leaf model scope is separate). This post-validation metadata
change leaves code, ABI, cases and all comparison numbers unchanged; the
contract-status attestation maps the before/after hashes. Strict checkpoint
mismatches still prevent a blanket unit-pass claim. Capability disposition
and formal review remain pending. Reproduction uses the checked-in harness:

```bash
python benchmarks/probe_bwd_span.py --soc a2 --block-dim 1 --a2-gc-dtype fp32 --a2-suite fixture --a2-out <fresh-fixture-output>
python benchmarks/probe_bwd_span.py --soc a2 --block-dim 1 --a2-gc-dtype fp32 --a2-suite actual --a2-out <fresh-actual-output>
python benchmarks/probe_bwd_span.py --soc a2 --block-dim 1 --a2-gc-dtype fp32 --a2-suite span --a2-spans 1 8 16 24 28 32 64 96 106.5 128 160 192 --a2-out <fresh-range-output>
```

Repeat bd2 in a separate process under the external machine configuration.
Raw logs preserve warning/error lines with private locations redacted.
