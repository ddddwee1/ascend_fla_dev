# A2 KDA layout

This unit copies a five-dimensional strided view into contiguous storage. It supports BF16 and FP32 copies, exact widening, round-to-nearest-even narrowing, an optional FP32 multiplier, and positive zero fill.

The runtime performs validation, allocation and metadata views. The kernels perform all data movement and dtype conversion. Input and output allocations must not overlap. The last dimension must be divisible by 64. Strides must be nonnegative, and the complete storage span must fit signed int32 addressing.

Six vector entries implement the operations. Each copy work item owns 64 output elements. A strided load places each element in a separate 32-byte UB block. Its gather uses byte offsets with a complete physical allocation. Scalar offset stores have an explicit local barrier; gather explicitly selects normal mask mode.

The declared block dimensions are 1 and 2. Independent device probes observed 1 and 2 vector participants, respectively, on Ascend 910B3. The device reported 20 cube cores and 40 vector cores. These observations cover vector-only launches and do not establish a larger block-dimension limit.

Run the standalone protocol with the pinned library available:

```sh
python run.py reference --case strided_bf16_bf16_bd1
python run.py check --device a2 --backend cce --launcher sim --case strided_bf16_bf16_bd1
python run.py check --device a2 --backend cce --launcher pipesim --case strided_bf16_bf16_bd1
```

Device acceptance used the project's resident aclnn bridge. Each block dimension ran in a separate process. All six vendors and the geometry probe were compiled before the first custom launch. The 56 contract cases and 58 additional checks passed bitwise. All 28 matching case pairs had identical input and output hashes across block dimensions. Full-shape cases include T=4096 and HV=32. See `evidence/qualification.json` and `evidence/native/`.

The final source passed a reduced strided BF16 case in functional and pipe simulation. Those checks cover that case only. Earlier source versions passed the same model checks but failed native gather with error 507035. A device diagnostic verified the scalar offsets, then reproduced the error with offsets loaded through DMA. Explicit normal mask selection removed that failure; both original full workloads subsequently passed. The model checks alone did not detect the native failure.

These results qualify the layout leaf only. They do not qualify preprocessing, KDA backward, the public layer, or performance.
