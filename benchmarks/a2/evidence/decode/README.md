# A2-14 independent decode evidence

See [the report](../../../../docs/research/a2_decode.md).

- `bd1.log` / `bd2.log`: complete final native raw JSON-line logs. Each begins with source/environment identity, records all six precompiled vendors, then all actual workload measurements.
- `bd1.json` / `bd2.json`: parsed receipts, input/output byte digests and per-case/per-token metrics.
- `cross-bd.json`: strict identical-identity/input checks and 19 output/state byte comparisons.
- `reference.json`, `sim.json`, `pipesim.json` and matching logs: separately scoped post-hardware model checks. Only absolute source paths are redacted.
- `host.log`: full host regression before qualification metadata. `host-metadata-pending.log` preserves the first stale platform expectation failure; `host-final-pending.log` records the second stale expectation in the forward evidence test. Both are isolated host assertions, with the second one awaiting PM scope approval.
- `source-correspondence.json`, `ast-audit.json`, `a5-preservation.json`, `model-environment.json` and `health.json`: supporting scope and identity checks.

The primary native golden is independent Torch CPU FP32; unit FP64-internal reference is only an additional diagnostic. Kernel/ABI/domain/comparison code is unchanged. Public A2 dispatch remains unqualified.

Host failure logs retain every line, with trailing whitespace normalized for Git; exact originals remain in ignored task scratch. Native raw logs are byte-preserved.
