# A5 PGDN fused recurrent

Single vector launch, native BF16/FP32 short-step forward, independently supplied
main and ATK initial states. Public API and frozen domain are documented in
`docs/research/pgdn_fused_recurrent_gate_range.md` at the repository root.

Set `FLA_PGDN_NAIVE` to the pinned FLA literal naive source before validation.
`python run.py reference` runs the canonical FP32 cases; the unit owns its inputs,
independent B, literal A loader, kernels and runners. `python -m ref.calibrate
--output <ignored-output>` compares both CPU references and the labeled FP64 lift.
Native verification uses `verify_native.py`, state continuation uses
`verify_integration.py`, host-dispatch observation uses `audit_host_work.py`, and
actual same-card B/C/B timing uses `measure.py`, all as package modules.

Final-source inprocess acceptance:126 cases at each block_dim1/2/4/8/16/28,
with all three outputs byte-identical across dimensions. Native FP32 o and both
FP32 states remain below1e-4 vs literal A and independent B. BF16 output passes
exact RNE checks against native FP32 companions; both states retain FP32 budgets.
Direct chunk→decode validation covers14 cases/56 chains and150 split comparisons.
The evidence includes actual same-card three-round torch_npu baseline timing.

Start at [evidence/index.json](evidence/index.json), then the repository's
[ABI/range ledger](../../../../docs/research/pgdn_fused_recurrent_gate_range.md).
[Warning dispositions](evidence/warnings/README.md) retain the original ATK
failure, compiler diagnostics and source/cache reconciliation. CPU oracle identity
is frozen by PM NOTE6033949597; cross-Torch subnormal boundaries are explicit.
No CUDA/Triton or model-weight execution is claimed.

Run `python verify_evidence.py` to audit published numeric records and source
hashes without NPU execution. This delivery helper was added after device runs;
it is separate from their immutable runtime source manifests. Machine config and
unredacted raw logs remain in ignored task storage.
