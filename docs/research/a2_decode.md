# A2 KDA decode qualification — A2-14

The existing A2 BF16/FP32 decode unit passes fresh hardware qualification at `block_dim=1,2`. Public KDA dispatch remains unqualified (`CAPABILITIES["a2"]["qualified"] == False`); no public routing, kernel, dtype, ABI or tolerance changes are included.

## Artifact and environment

Ascend910B3 (20 cube / 40 vector cores), CANN 9.0.0 compiler and OPP timestamp `20260428_134817545`; installed kernel packages `ascend910b` and `ascend910_93`. Python 3.11.15, Torch 2.10.0+cpu, torch_npu 2.10.0, HF32 disabled. Device health was normal before and after execution.

Accepted source identities are library `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5` and kernels `b3b3f9c16df7c4626ed3c081032a1be5a753d0b1`, selected from the current compatibility pin. The verifier checked the actual imported library bytes (204 tracked package files) before compilation. Each native receipt includes source SHA-256 values and CANN/OPP identity. All kernel, runtime and verifier files still match the measured bytes; the subsequent contract change affects only the board row, and the README changes qualification prose. See [source correspondence](../../benchmarks/a2/evidence/decode/source-correspondence.json).

## Fresh device results

Each bd runs in its own process and precompiles **all six** decode/chunk vendors before the first custom launch. No A2-03 observation is relabelled as new evidence. The final run executes the actual `unit.execute` entry and the continuation launcher on identical inputs; their outputs and states match byte for byte in all 32 accuracy runs. Device inputs remain unchanged.

| Check | Executed scope | Result |
| --- | --- | --- |
| Accuracy | Eight original cases plus H/HV=32/32 and 16/32 at T=1,4,8,16; each at bd1 and bd2 | 32/32 pass both allclose and relative-L2 rules |
| Segmented decode | 16 single-token calls vs one T16 call; both head configurations, both bd values | Output and final state bitwise equal |
| Chunk → decode | H/HV=32/32, prefix128 followed by 64 single-token steps vs whole chunk192; both bd values | Every token passes; every state pointer is preserved on NPU |
| Cross-bd | 16 accuracy cases, two segmented cases and full continuation | All 19 output/state pairs bitwise equal; source/environment and input hashes also checked |

The main golden is an independent Torch **CPU FP32** recurrence. The original unit reference computes internally in FP64 and returns FP32; it is retained as an explicitly labelled extra diagnostic, following the PM specification correction. Torch NPU FP32 composed recurrence is the second oracle and executes successfully.

| Comparison | Maximum output relative L2 | Maximum state relative L2 |
| --- | ---: | ---: |
| Custom vs CPU FP32 | 0.00172454061 | 1.01513436e-07 |
| Torch NPU FP32 vs CPU FP32 | 2.50069263e-07 | 9.67778959e-08 |
| Unit FP64-internal vs CPU FP32 | 2.23311645e-07 | 1.06183244e-07 |

Frozen decode rules are unchanged: BF16 output `rtol=1e-2, atol=1e-4, relL2<=0.005`; FP32 state `rtol=1e-5, atol=1e-6, relL2<=1e-5`. The continuation golden starts from a CPU copy of the actual chunk-returned state solely for independent comparison; the custom chain consumes that original NPU state directly. All 128 step-wise state checks pass, with maximum relative L2 `1.792070066e-7`.

The separate **chunk-vs-decode** comparison uses the existing chunk rule (`rtol=0.02, atol=0.02, relL2<=0.05`), because the chunk path has BF16 intermediate arithmetic. Its overall output/state relative L2 is `0.003589344211 / 0.002944529289`; the worst individual token output relative L2 is `0.003996430896`. This does not loosen the decode rule: every decode step is also checked against the conditioned CPU FP32 golden under its original strict budget. A 64-token suffix makes whole length192 legal for the chunk path, which requires a multiple of64.

## A2-11 premises

1. The accepted pins and actual SoC/CANN/OPP identities are verified and embedded in these fresh receipts.
2. The full A2 static accumulate guard passes; it scans all A2 kernels and has no decode exemption in the test implementation.
3. The selected library contains the FP32 split-K `PIPE_M` repair established by A2-11. This decode kernel is vector-only and never exercises split-K; no new split-K hardware claim is made here.
4. A fresh AST scan finds **zero `matmul` calls** in decode `kernels/step.py`. Neither BF16/FP16 short split-K nor a handwritten FP32 MMAD chain is reached. The companion chunk unit retains its explicit barriers and separately passes the whole-chain hardware comparison. This is an applicability finding, not a claim that the hardware defects have disappeared.

## Timing

Three synchronized custom / Torch NPU / custom sandwiches per T and bd; warmup3, repeat10, synchronization around every sample. Values below are median microseconds per call. All raw samples are in the native JSON receipts. Fixture generation and H2D are excluded. Custom output buffers are preallocated; the Torch FP32 composed baseline allocates intermediate/output tensors and performs its BF16-to-FP32 input casts. These are measured launch/composition costs, **not isolated device kernel latency**, model throughput or a claim about an optimized baseline. No slowdown samples were discarded.

| bd | T | Round | Custom before µs | Torch NPU µs | Custom after µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 805.7 | 853.5 | 804.0 |
| 1 | 1 | 2 | 803.7 | 857.1 | 805.9 |
| 1 | 1 | 3 | 798.8 | 843.2 | 802.7 |
| 1 | 16 | 1 | 7214.3 | 8486.1 | 7209.5 |
| 1 | 16 | 2 | 7210.8 | 8239.7 | 7199.3 |
| 1 | 16 | 3 | 7210.7 | 8063.7 | 7202.4 |
| 2 | 1 | 1 | 585.2 | 837.3 | 578.8 |
| 2 | 1 | 2 | 575.7 | 838.1 | 583.4 |
| 2 | 1 | 3 | 576.3 | 840.0 | 578.6 |
| 2 | 16 | 1 | 3769.6 | 8025.3 | 3775.4 |
| 2 | 16 | 2 | 3775.3 | 7727.6 | 3834.2 |
| 2 | 16 | 3 | 3760.3 | 7635.6 | 3774.3 |

## Other validation and reproduction

After the complete hardware workload: the original unit reference generated eight cases (FP64-internal diagnostic); functional simulation passed `speculative_4` at bd1; pipe simulation passed `single_token_gva` at bd1, with empty event balance/hazards and no deadlock. These bounded model checks do not substitute for the real-shape hardware grid. The source checkout has no installed distribution metadata, so model summaries record `ascriptor: null`; [model environment](../../benchmarks/a2/evidence/decode/model-environment.json) records the imported module version0.1.0 and selected pin.

Host full suite before capability metadata: **1369 passed / 6 skipped**. A first run had six missing-reference-environment failures; setting the existing hash-verified FLA oracle file variables resolved all six without changing tests. PM approved the one-line platform-test expectation update. The subsequent full suite is **1368 passed / 6 skipped / 1 failed**: `tests/test_kda_fwd_a2.py:171` contains the second exact chunk-only capability dictionary. All-tests search and the full run identify this as the only remaining conflict. Its one-line expectation update is requested in issue41/comment5862263134; the PR stays draft until approval and a green final full suite. Neither failure changes the hardware result or justifies changing any forward gate/budget. All eleven original A5 verifier function bodies remain unchanged.

```bash
python benchmarks/verify_decode.py --soc a2 --shapes kimi --a2-out <fresh-ignored-directory> --a2-performance
pytest tests/ -q
```

Configure CANN, the pinned `ASCRIPTOR_WORKSPACE`/`PYTHONPATH`, device visibility and writable caches using the ignored private machine configuration. The full host suite also needs the pinned FLA naive source variables for its existing independent-oracle tests. No machine values are embedded in committed scripts or evidence.

[Evidence directory](../../benchmarks/a2/evidence/decode/) contains the complete native JSON-line logs, per-case receipts, exact cross-bd comparison, model results and host log. Native raw logs contain only measurement records and required no line removal. Model receipts redact only absolute source paths; original private receipts are retained. The private archive retains 3163 raw build/run/helper files; all3163 were independently restored and hash-verified. The [public artifact restoration receipt](../../benchmarks/a2/evidence/decode/archive-restoration.json) records the exact manifest and successful restored hashes; archives remain outside Git.
