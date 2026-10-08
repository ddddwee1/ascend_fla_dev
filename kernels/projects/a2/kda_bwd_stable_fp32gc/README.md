# A2 stable KDA backward with FP32 cumulative gates

D-PM-60 authorizes this derived unit. Only `saved.g_cumsum` changes to FP32
(log2 units); the other eight caches and all returned gradients remain BF16.
The four consumer kernels load FP32 gates directly. Five other kernels are
byte-for-byte copies of the frozen predecessor. Ownership, loop order,
mutex depths, barriers and original comparison thresholds are unchanged.

Vendor compilation passed at bd1/2. Original fixture qualification failed
3 of26 cases at strict intermediate checkpoints; all final-gradient checks
and the listed actual-cache end-to-end runs passed. See
[the A2-13 report](../../../../benchmarks/a2/evidence/kda_bwd/fp32gc-v1/README.md).
The measured actual-cache end-to-end backward limit is128; strict fixture
qualification remains failed. The original BF16 unit and its historical evidence
remain untouched; this limit must not be used for that predecessor. Real-cache acceptance uses CPU FP32 full-chain autograd
with relative L2 <= 0.05 for each of six gradients; fixture/stage comparisons
remain separate. No public dispatch or `qualified` flag is changed.

Use `run.py` for the unit protocol. The A2-13 hardware harness selects this
unit with `benchmarks/probe_bwd_span.py --soc a2 --a2-gc-dtype fp32`; all five
forward and nine backward vendors must compile before the first custom launch.
