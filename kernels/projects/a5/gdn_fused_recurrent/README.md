# A5 GDN fused recurrent

One pure-vector launch for scalar-gated GDN with raw q/k, contiguous value-head
groups, S1..16, K=V128, native FP32/BF16 inputs, and fresh FP32 K-major state.
Each vector participant owns the complete state for a value head. All scaling,
casts, grouping and recurrence arithmetic occur inside the kernel.

The public entry is `ascend_fla.ops.gdn_fused_recurrent.fused_recurrent_gdn`.
See `docs/research/gdn_fused_recurrent_gate_range.md` for the ABI, numerical
domain, frozen budgets and qualification status. Development is still in
progress; do not interpret a candidate block dimension as completed acceptance.

Select library `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5` and kernels
`b3b3f9c16df7c4626ed3c081032a1be5a753d0b1` through the accepted environment.
Set `FLA_GDN_NAIVE` in ignored configuration to the exact pinned FLA naive
source; its hash is checked before importing the literal CPU oracle.

From the repository root, using the accepted Python:

```bash
python -m kernels.projects.a5.gdn_fused_recurrent.ref.calibrate --output tmp/GDA-04/calibration.json
python kernels/projects/a5/gdn_fused_recurrent/run.py reference --output tmp/GDA-04/reference
python -m kernels.projects.a5.gdn_fused_recurrent.verify_native --block-dim 1 --output tmp/GDA-04/native
python -m kernels.projects.a5.gdn_fused_recurrent.verify_integration --block-dim 4 --output tmp/GDA-04/integration
python -m kernels.projects.a5.gdn_fused_recurrent.measure --block-dim 4 --output tmp/GDA-04/timing
```

Hardware commands require Docker, fresh health/occupancy checks, the canonical
shared lock, and isolated caches/output directories selected by ignored machine
configuration. Prepare every operator used by a process before its first aclnn
call. Each block dimension uses a separate process and build. The first native
grid cases are the full B1/S16/H16/HV32 workload in both storage dtypes.

The unchanged canonical runner has a static output dtype and therefore covers
the44 FP32 cases. The native verifier covers all88 FP32/BF16 cases, including
same-input FP32 companions for BF16 output-RNE/state byte checks. CPU references,
functional simulation, pipe simulation, emitted source, vendor compilation,
native execution and same-card timing are separate evidence stages.

`ref/integration.py` is a CPU calibration of the integration checks, using
independent recurrence and chunk block-solve references. Only
`verify_integration.py` executes actual chunk→decode calls with device state
passed directly between them. `measure.py` compares real public calls with a
torch_npu composition of the same recurrence; it reports synchronized wall
latency including allocation and Python dispatch, with no speed threshold.
