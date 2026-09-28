# GDA-04 qualification evidence

The implementation starts at `1de6f78`; `922caf0` merges the PM's corrected
specification. Each native receipt contains exact executed source hashes,
compiler/OPP identity, runtime versions and compiled-artifact hashes checked
both before and after execution. Build-relative artifact paths identify files,
not machine locations. Machine bindings, absolute paths and process details
remain in ignored configuration. `log-excerpts.json` preserves original log
line numbers and the SHA256 of the private original.

`index.json` lists executions. The first native call was the complete
B1/S16/H16/HV32 workload in both dtypes. Native grids at bd1/2/4/8/16/28 each
contain all88 cases, with independent CPU A/B metrics and BF16 storage checks.
`native/cross-bd.json` checks identical inputs and output/state bytes across
all six grids. The initial `grid-bd1-v1` and `integration-bd4-v1` had a foreign
task observed during their collection windows; retain their numerical results
but use the separately labeled isolated repeats for acceptance. Context counts
are sampled observations, not a claim of absolute exclusion between samples.

The integration reports contain every suffix token's metrics against actual
whole-chunk execution and independent CPU B. State passes directly between NPU
calls. The132 decode partitions must match one-call outputs/state byte for
byte. CPU-only integration calibration is supplementary and retains an example
state-drop negative control; it is not substituted for native integration.

`native/timing-bd4-v1` contains four workloads, three B/C/B rounds each,
10 warmups and50 synchronized samples per leg. Baseline is actual torch_npu
equation composition. Candidate measurements include allocation, cached runtime
dispatch and synchronization; input generation, reference evaluation, transfers
and compilation are excluded. These are public-call wall times, not device-only
cycles, CUDA/Triton comparisons or a model-weight validation.

Original CPU calibration reports predate PM acceptance and therefore retain
the wording "proposed / RISK pending". The [PM decision](https://github.com/ddddwee1/ascend_fla_dev/issues/94#issuecomment-5870217647)
explicitly accepts those exact88-case results without rerunning. A is the
literal pinned naive source; FP64 A is explicitly lifted. Model records mark
`backend_executed=false` and cover only two focused bd1 cases after hardware
execution. They do not claim all88 cases or all block dimensions in a model.

The canonical runner and board/aclnn diagnostic launchers are distinct from
the qualified public NPU inprocess path. Board/aclnn remain unqualified here.
See the research ledger for the support domain and warning assessment below.

## Warnings and retained failures

- CCE `-Wcce-compat` reports uint16 loop counters compared with int32 bounds.
  Decode bounds are S1..16 (validated before launch); increments stay0..16.
  Constant K loops are128. The unchanged chunk kernels used for integration
  have `count=Min(64,T-64*chunk)` and `j<i+1` with i<64, hence all warned bounds
  and increments stay0..64. For the tested legal T multiples, count is64.
  No wrap, negative conversion or truncation is reachable in these domains.
  Original warning locations are retained in `warnings/vendor-log-excerpts.json`.
  No warning suppression or kernel cast workaround was applied.
- `NPUCachingAllocator.cpp:201` reports32-byte compatibility padding with
  torch_npu git `fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d` and CANN9.1.0-beta.1.
  The [owner implementation](https://github.com/Ascend/pytorch/blob/fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp)
  `AddPadSize` (192–207) chooses capacity padding; `round_size` (2119–2132)
  adds it before allocator rounding. It changes allocation capacity, not
  logical tensor shapes, values or stream ordering. Source SHA256 is
  `e582e9099555b54a9c0c41d89152dab4c2ed72759eae6851c19a53bdcd284527`.
  The matching version parser SHA256 is
  `bd9e9b06f110389ec1d1a2ef2613463dd37ef11f9ea1f44a597331b791a1eeef`;
  beta sorts below final9.1.0, explaining this branch. No setting changed.
- CMake reports only unused `CMAKE_CROSS_PLATFORM_COMPILER`; actual opapi
  libraries compiled, linked and executed. The vendor Python3.12 helper's
  `SyntaxWarning` concerns a regex escape. Exact contexts are retained under
  `warnings/`. Driver-info/loglevel error4 messages come from deliberately
  device-free compiler Docker, not failed physical execution.
- Full host regression has five existing `pytest.importorskip` deprecations
  because the accepted CPU environment intentionally refuses torch_npu import.
  Those modules are skipped and do not count as hardware acceptance. Warnings
  were neither suppressed nor repaired outside the task's write set.
- An early CPU calibration attempt lacked einops; installing it only in the
  private environment resolved that environment prerequisite. A source-only
  host CLI emission attempt lacked backend entry-point metadata; the documented
  `runtime.compile_kernel` CCE path emitted the same selected source without
  changing the library or backend. Raw failures stay in ignored task evidence.
- The first dispatch audit expected only public output/placeholder allocations
  and detected one additional empty allocation. `runtime/binding.py:269–275`
  owns the cached uint8 aclnn workspace. The follow-up audit identifies each
  call site and checks subsequent reuse, without permitting tensor arithmetic.
