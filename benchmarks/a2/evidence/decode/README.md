# A2-14 independent decode evidence

See [the report](../../../../docs/research/a2_decode.md).

- `bd1.log` / `bd2.log`: complete final native raw JSON-line logs. Each begins with source/environment identity, records all six precompiled vendors, then all actual workload measurements.
- `bd1.json` / `bd2.json`: parsed receipts, input/output byte digests and per-case/per-token metrics.
- `cross-bd.json`: strict identical-identity/input checks and 19 output/state byte comparisons.
- `reference.json`, `sim.json`, `pipesim.json` and matching logs: separately scoped post-hardware model checks. Only absolute source paths are redacted.
- `host.log` and `host-final.json`: final full host regression, 1369 passed / 6 skipped, after both PM-approved capability expectation corrections. `host-before-metadata.log` retains the earlier green run. `host-metadata-pending.log` and `host-final-pending.log` retain the two historical stale-dictionary failures, both now resolved; all other assertions are unchanged.
- `source-correspondence.json`, `ast-audit.json`, `a5-preservation.json`, `model-environment.json` and `health.json`: supporting scope and identity checks.

The primary native golden is independent Torch CPU FP32; unit FP64-internal reference is only an additional diagnostic. Kernel/ABI/domain/comparison code is unchanged. Public A2 dispatch remains unqualified.

Host failure logs retain every line, with trailing whitespace normalized for Git; exact originals remain in ignored task scratch. Native raw logs are byte-preserved.
