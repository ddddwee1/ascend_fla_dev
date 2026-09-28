# PGDN FP32 chunk backward (PK-05)

This standalone A5/CCE unit computes `dq, dk, dv, dg_atk, dg, dbeta_atk,
dbeta` for the loss `<do,o> + <dht,final_state> + <dA_T,final_A_state>`.
At least one cotangent must be supplied. The public adapter is
`ascend_fla.ops.pgdn_chunk_bwd.chunk_pgdn_bwd`; no autograd wiring is provided.

The frozen domain is B=1, T a positive multiple of64 no larger than4096,
K=V=128, positive H/HV with HV divisible by H, FP32 contiguous inputs,
zero initial states (only `None`), token-major layout, mandatory FP32 q/k
normalization with norm clamp1e-12, scale128**-.5, x1.5, ATK eps1e-6 and
log_atk_scale=-.2. Gates must be finite/nonpositive with finite chunk sums;
beta and beta_atk belong to[0,1]. All inputs and cotangents must be finite.
BF16 is explicitly rejected. Near-zero q/k rows are accepted.

The independent implementation has no runtime import of the forward, GDN or
GDN-2 units. Host production work is input validation, allocation and pointer
binding. All normalization, state replay, grouped reductions and derivatives
execute in the four task-owned device kernels.

| Stage | Work | UB bytes / buffers |
| --- | --- | --- |
| pgdn_bwd_atk | Normalize q/k and save every per-H ATK state | 140032 / 10 |
| pgdn_bwd_checkpoints | Save the main state before each64-token chunk | 67136 / 6 |
| pgdn_bwd_reverse | Replay one chunk, then reverse its main recurrence perHV | 135808 / 15 |
| pgdn_bwd_atk_reverse | Sum consecutive value-head contributions, reverse ATK and normalization perH | 4832 / 16 |

The main reverse uses `k_read` in the residual and read-key correction,
`k_write` in the state update and write-key derivative. The ATK terminal
cotangent is seeded once perH, after summing all its value-head contributions.
Each owner traverses the complete time axis. GM replay publication uses an
explicit MTE3-to-MTE2 event; all tape readers drain before slot reuse.
All20 named stage outputs are independently observable in the native harness.

## Numerical contract

A is literal FLA PGDN naive at pin
`e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`, differentiated by CPU FP32 autograd.
B is a separate analytical reverse with chunk checkpoints; it imports neither
FLA nor the device implementation. Both passed the six small FP64 gradchecks
away from the normalization kink. The external literal file is selected by
`FLA_PGDN_NAIVE` and hash-checked. Production execution does not depend on it.

Classification is per gradient element, generated from independent FP64
analytical truth and local cancellation scale before examining the candidate.
Algebraic zeros and elements with `abs(reference64)<=64*2^-23*local_scale`
are disclosure-only. Each ordinary subset separately requires relativeL2<=1e-4
against both A and B. The FP32 normalization clamp branch is retained when
lifting the reference at the exact boundary. Actual native branch bits are
also compared, independently of the numerical classifier.

Disclosure-only elements have no pass threshold. Complete records include
position, candidate, canonical/raw FP64 reference, local scale, threshold,
classification reason, absolute output and contribution relative to the full
FP64 gradient norm. Zero denominators remain undefined/infinite as appropriate;
no denominator epsilon is added. Complete indexed JSONL uses `start/count`
only for adjacent, bit-identical records and therefore restores every element,
including signed zeros. The concrete formulas are in
[the range study](../../../../docs/research/pgdn_chunk_bwd_gate_range.md).

**The overall numerical status is “numerical disclosure complete, nonPASS”
whenever disclosures exist.** Ordinary-budget satisfaction is reported
separately. `run.py` retains the canonical raw comparison as a diagnostic; its
reference stage only checks schema/finiteness and is not classified acceptance.

## Reproduction

Use the accepted Python environment and Ascriptor0.1.0 source selected by
`compatibility.json` (library90cfcdc / kernelsb3b3f9c). Machine selection stays
in ignored external configuration via `ASCRIPTOR_MACHINE_SPECS` and
`ASCRIPTOR_BOARDS`. Native runs require a fresh healthy/idle-device check,
the shared lock, isolated outputs, and separate processes/builds for bd1/bd2.
Run the complete T4096 workload first. Simulators are diagnostic tools only.

From this unit directory, with the selected library and repository importable:

```bash
python -m ref.gradcheck --output <fresh-gradcheck-json>
python -m ref.calibrate --output <fresh-calibration-directory> --workers 4
python run.py reference --case full_r1_m7_bd2 --output <fresh-reference-directory>
python benchmark.py --block-dim 1 --output <fresh-bd1-directory> --oracle-workers 4
python benchmark.py --block-dim 2 --output <fresh-bd2-directory> --oracle-workers 4
python compare_runs.py <bd1-qualification.json> <bd2-qualification.json> --output <comparison.json>
python benchmark.py --supplemental-clamp --block-dim 1 --output <fresh-clamp-bd1-directory> --oracle-workers 4
python benchmark.py --supplemental-clamp --block-dim 2 --output <fresh-clamp-bd2-directory> --oracle-workers 4
python compare_runs.py <clamp-bd1-qualification.json> <clamp-bd2-qualification.json> --output <clamp-comparison.json>
python measure.py --block-dim 2 --output <fresh-measurement-directory>
```

The public benchmark exercises actual `None` cotangents, poisoned outputs,
input immutability, independent leaf launches and composition, public output
bytes, internal stage values, clamp branch bits and classified A/B comparison.
The bd comparison requires the complete selected grid (original256 or
supplemental73) and identical inputs, 20 stage hashes and seven public output
hashes. Missing, duplicate or mixed grids are rejected.

Native evidence in this task uses the `inprocess` launcher inside the accepted
Docker environment. The explicit CPU-input `aclnn` and `board` transport paths
have host validation coverage but remain untested on hardware for this task.
The contract's `board` stage means device qualification; it does not imply that
the identically named transport launcher was exercised.

Measurement uses T1024/4096, B1/H=HV8, bd2, 10warmups, 50samples and three
baseline/candidate/baseline rounds. Each timed call synchronizes before/after;
CPU references finish beforehand. The baseline reuses this candidate's saved
normalization/ATK tapes and main checkpoints, excludes stage1/2 device work,
and retains public validation and all allocations (including unused forward
outputs). It is an adjoint cost baseline, not another full backward or a
CUDA/Triton comparison. Numerical checks and disclosures bracket every round.

## Evidence status

The unchanged frozen256-case CPU calibration satisfies every ordinary A/B
budget, with maximum relativeL2=2.9902082636512436e-6. Both references passed
six FP64 gradchecks away from the normalization kink. The complete512 A/B
disclosure lists and six frozen reference hashes remain in the evidence.

The repaired artifact completed the original256 and supplemental73 cases at
both bd1 and bd2, with all required actual-output checks. Identical runtime
inputs give byte-identical values for all20 stages and seven public gradients
in each complete grid. The maximum ordinary-subset relativeL2 against either
independent reference is 4.1356284098271835e-06, within the unchanged1e-4 budget.
Public disclosure elements are bd1=67,662,570 and bd2=67,662,570;
every indexed payload is retained. Overall numerical status is **numerical disclosure complete, nonPASS**.

See `evidence/norm8-qualification-summary.json`, both norm8 byte comparisons,
the four norm8 matrix summaries and `evidence/norm8-source-consistency.json`.
The original c8e7073 dense-clamp failure and older successful-grid evidence are
historical records; they do not stand in for the repaired artifact's runs.
The six frozen references, classification rules and budgets remain unchanged.

The earlier repaired-source `norm8-matrix-bd1-v1` and `norm8-matrix-bd2-v1`
numerical runs observed unrelated device contexts despite both locks. Their
complete numerical evidence is retained with `qualification-scope.json`
explicitly excluding exclusive acceptance and performance. The qualified
original grids are the secondary v5 runs, followed by secondary supplemental
v1 runs. Builds use device-free Docker; actual execution uses the existing
device Docker. Each run hashes all100 installed vendor files, harnesses and
source stamps before and after execution. Build and device receipts are separate.

Three new same-card sandwiches for each of T1024/4096 use10 warmups and50
synchronized samples per leg, with classified checks/disclosures before and
after every round. All samples are in `evidence/native/norm8-secondary-timing-bd2-v1/`.
This is public-call wall latency including validation/allocation, compared to
the precisely declared cached-forward-state baseline, with no speedup claim.

Host regressions:1358 passed,10 skipped,5 existing CPU-stub warnings. Actual
Docker CPU task regressions:119 passed. Vendor diagnostics are investigated
and retained without suppression in `evidence/norm8-vendor-warning-assessment.json`.
No simulator, pipesim, CUDA/Triton or weight
validation is claimed. Machine configuration remains external and ignored.
