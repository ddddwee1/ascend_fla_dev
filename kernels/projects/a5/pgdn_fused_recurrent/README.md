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

The unit is under validation. An emitted or vendor-compiled artifact is not a
hardware qualification. Temporary runners, machine configuration and raw private
logs belong in ignored task storage; only sanitized final evidence is shareable.
