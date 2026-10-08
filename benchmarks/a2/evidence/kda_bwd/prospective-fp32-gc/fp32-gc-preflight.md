# Read-only prospective FP32 cumulative-gate repair preflight

Status: proposal, not an implementation or hardware qualification. PM scope is
pending under issue40 RISK5876471263. No kernel/unit/contract source was modified.
Session01a0b7ce-3daf-7550-a7bf-31c67eb91ca3, A2-13, PR137.

## Objective and evidence boundary

Repair the cumulative-gate precision seam that produces finite but inaccurate
real-cache gradients. The measured BF16-gc uniform H32/T128/seed0 span32 case has
dg relativeL2 .063173714 against the unchanged CPU FP32 end-to-end budget .05.
CPU replay with only gc promoted to the actual unrounded forward FP32 values
reduces all six whole-tensor errors below .05 at six measured spans24..128.
This is a hypothesis for a device repair, not a proof. Per-head/per-chunk CPU
analysis still finds dg slices above .05 at high spans; that marker localizes
error, it does not invent a new acceptance contract or waive whole-tensor failures.

Proposed delivery is a separately named A2 derived backward unit, retaining the
frozen BF16 unit. Exact unit name and write-set require PM routing. No public
dispatch or qualified flag changes; no product host arithmetic. Forward/runtime
and the five backward kernels not consuming gc remain read-only.

## Semantics, ABI and precision

Only saved.g_cumsum becomes contiguous CPU/GM FP32 [B,T,HV,128], in log2 units.
Actual-forward assembly remains direct natural-log FP32 gc * 1/ln2, but without
BF16 narrowing. No difference of rounded cumulative values, no log2(eg), no
modified original g or gradient golden. Other eight caches, public inputs and
returned gradients retain their current BF16 ABI; existing FP32 internal outputs
retain their types. B>0,T%64==0,H>0,HV%H==0,K=V128 and bd1/2 remain explicit.
Nonfinite inputs still reject. The selected unit's HostSpec must bind the FP32
parameter; never send FP32 storage through a BF16 descriptor.

Reference fixture construction must intentionally retain only gc as FP32.
The independent FP32 autograd golden remains unchanged; fixture old budgets and
actual-cache six-gradient .05 budgets remain distinct. No stage threshold changes.
This moves an existing rounding boundary and therefore needs a new precise
contract; it is not a metadata-only change.

## Tiling, ownership and storage rationale

All original work partitioning and loop orders stay unchanged. scan_fused owns
B*HV heads split across cube blocks, two vector subblocks own 64 state rows and
32 token rows; each head scans chunks in reverse. The three vector consumers own
B*HV*C work items over GetVecNum(), and preserve half-tile32x128 order (post in
reverse order for its running gradient sum). Full shapes must exercise repeated
chunk/head reuse, not just one work item per core.

Proposed exact data movement, before any source edit:

- scan_fused: load the FP32 last 64-element row slice directly into existing
  expg_ub; remove glast_b_ub and its BF16 cast. Keep muls/exp and all scan bridges.
- finalize_pre: load the FP32 last128 row directly into existing gmid_ub; load
  each FP32 half directly into cs_ub before the same subtraction/exp sequence.
  stage_ub remains BF16 for q/k. No new staging tile is necessary.
- inverse_epilogue: FP32 32x128 gate staging (+8192 bytes), direct FP32 last-row
  load into glast_ub (-256 bytes), consume each read-only staged gate row directly
  in existing muls/sub operations and remove the old cast-only grow_ub (-512).
  These operations still take FP32 operands and retain arithmetic order.
- finalize_post: FP32 32x128 gate staging (+8192), direct last-row gmid_ub load
  (-256), feed a staged FP32 row directly to the existing sub(sh,row,gmid),
  removing the preceding BF16 cast. Preserve reverse row traversal and running sum.

Static declared UB bytes, excluding compiler temporaries (not compile proof):

| Kernel | Existing | Proposed | Remaining under192KiB |
|---|---:|---:|---:|
| scan_fused |135552|135424|61184|
| inverse_epilogue |168192|175616|20992|
| finalize_pre |109856|109600|87008|
| finalize_post |159488|167424|29184|

Detailed source-hashed allocation accounting is fp32-gc-storage-preflight.json.
All row widths128 FP32 are512B, scan half-width64 is256B; offsets and pitches
are multiples of32B for every allowed HV. No tail or packed single-element gate
load is introduced. Existing packed BF16 beta row-casts remain untouched.
The logical GM cache B1/T4096/HV32 grows32MiB to64MiB; this is byte arithmetic,
not measured peak memory or a performance claim. Emitted allocation and vendor
compilation still must confirm real UB usage and all footprint limits.

## Synchronization and lifetime obligations

Every gate staging tile remains single-slot. MTE2 publishes the complete half
before the first V consumer. In inverse_epilogue the last V read is the last
row's second gate subtraction; in finalize_post it is the final reverse row's
subtraction. The next half's MTE2 overwrite must wait for that last read. No
manual event depth increase or new lookahead is justified. Direct gmid/glast loads
require MTE2->V readiness and last-reader->MTE2 reuse at the next work item.
scan expg's final scalar reader belongs to its state-row decay loop and must retire
before the next chunk overwrite. Inspect generated auto_sync, do not infer it.

scan_fused's five existing workspace mutex families, two physical slots, beat
increment, acquire/publish/wait/free points, L1 fences and warmup/drain are
unchanged. Gate loading is vector-local and does not grant any extra cross-side
credit. Preserve frozen inverse_mm resources and the32-ID limit. Original A2
BF16/FP16 M<64 split-K prohibition remains; no new cube call is proposed.
No deletion of precautionary fences. Inspect emitted event IDs and physical
buffer lifetime after compilation, then bounded model diagnostics after full
native workloads. Neither static accounting nor unchanged source structure
proves a newly emitted schedule safe.

## Sources and forms inspected

Standalone compatibility pins: library90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5,
kernelsb3b3f9c16df7c4626ed3c081032a1be5a753d0b1. A2 remains an experimental facade
in library0.1.0; only actual A2 results establish this unit's scope.
Read Chinese router/common-language/debug/preflight/precision/synchronization;
owner docs/api/{README,storage,a2-vectors}.md, frontend/dsl.pyi DMA declaration,
and selected examples/api/a2_vectors.py (FP32 GM->UB + counted muls/add on A2).
The example is read-only source evidence, not a new executed qualification.
FP32 native gate loads and vector arithmetic use owner-declared forms; actual
generated stride/event/buffer footprints must be reverified for this unit.

## Required execution once scope is assigned

Recheck environment/pins/source hashes, Docker/device health and lock; one process
per bd, distinct cache/vendor symbols for derived and frozen variants. Compile
all5forward+9derivedbackward vendors before first custom launch, keeping frozen
variant in a separate process if a comparison needs both. No late registration.
Run original full fixtures, actual-cache small/GQA/B2/C1/odd-C boundaries and
H32T64/128/512/4096. Compare full six-gradient CPU FP32 autograd and Torch NPU
reference, all33 checkpoints, raw6/assembled9 digests, per-head/chunk localization
and cross-bd bytes. Preserve all original budgets and failures.
Then range curves must include uniform and initialization profiles, the existing
known failures and the eight-seed initialization lower bound. No threshold from
the six CPU-only counterfactual points. Finally run reduced leaf sim/pipesim with
repeated slots and guards, negative metric controls, host regression, sanitized
receipts and archive restoration. A task is complete only after its assigned
acceptance and PM delivery gates actually pass.
