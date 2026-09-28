<!-- 由 tools/gen_matrix.py 生成，请勿手改。改 docs/matrix/*.json 后重新运行。 -->

# 支持矩阵

记录于 2026-09-14。本文件由 `docs/matrix/*.json` 生成。状态词汇沿用 ascriptor：`passed` / `untested` / `gap` / `failed`。

## 目标模型形状

ascriptor A5 定尺 ABI：`[B, H, C, L, D]`，L=64，D=128，q/k/v `bfloat16`，beta/g `float32`。

| 模型 | 算子族 | 优先级 | 目标期 | H | HV | head_k | head_v | dtype | 定尺匹配 | 阻塞缺口 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B-Instruct | gated_delta_rule | secondary | 第 4 期 | 16 | 32 | 128 | 128 | bfloat16 | ✅ ✅ ❌ | `gdn-no-gqa` |
| GDN-2 1.3B FineWeb-Edu 100B（95B checkpoint） | gdn2 | investigation | — | 16 | 16 | 128 | 128 | bfloat16 | ✅ ✅ ✅ | `gdn2-chunk-gate-range` |
| Kimi-Linear-48B-A3B-Instruct | kda | primary | 第 1 期 | 32 | 32 | 128 | 128 | bfloat16 | ✅ ✅ ✅ | — |
| fla GatedDeltaNetConfig 默认值 | gated_delta_rule | reference-only | — | 6 | 6 | 256 | 512 | — | ❌ ❌ ✅ | `fixed-kv-128`, `asymmetric-kv-dim` |
| fla KDAConfig 默认值 | kda | reference-only | — | 16 | 16 | 128 | 128 | — | ✅ ✅ ✅ | — |
| fla DeltaNetConfig 默认值 | delta_rule | reference-only | — | 16 | 16 | 128 | 128 | — | ✅ ✅ ✅ | — |

定尺匹配三格依次为 head_k / head_v / head 分组。

### 算子测试应覆盖的形状

- **smoke** — B=1, H=1, C=1 · ascriptor 现有 case 的规模，仅用于接线冒烟
- **qwen3_next_layer** — B=1, H=16, HV=32, C=16, T=1024 · 单层真实形状，第一期精度验收目标
- **gdn2_1_3b_layer** — B=1, H=16, HV=16, C=16, T=1024 · GDN-2 1.3B 单层真实形状；输入槽语义与现有 a5.gdn_fwd 不同
- **kimi_linear_layer** — B=1, H=32, HV=32, C=16, T=1024 · KDA 单层真实形状
- **long_context** — B=1, H=16, HV=32, C=64, T=4096 · 覆盖 chunk 边界与 state 传递，不是为了测误差累积

> 算子测试矩阵应覆盖的形状。T 必须是 64 的倍数（L=64 无 tail 路径），C = T / 64。

### 待核实的内部规格

- **qwen3.5-9b**（gated_delta_rule）：32 heads / head_dim 128 / chunk 64, bf16, 24 layers — 规格需从权威 config 核实后再补入 models[]；当前仅作参考，不要当作已验证形状。

## 算子支持状态

ascriptor pin：`0.1.0.dev1` · library `77619116f9b3` · 支持硬件 a5 · deferred a2, a3

> ⚠️ 这个 library 修订 **unreachable** —— 2026-09-17 在权威来源上核过：该对象不存在（git cat-file 取不到）。本条自建仓 346cd8a 起未改动过，且从未被验证。推测上游历史重建过（版本号 0.1.0.dev1 → 0.1.0）。后果：第一、二期在 A5 上的精度数字目前无法按此 pin 复现 —— 数字本身有效，但『在哪个编译器修订上测的』这一维已经断了。重新钉 pin 需要所有者指认与之对应的修订，或接受在新 pin 上重跑回归。

> 动手用 `agent/compatibility.json` 的 pin（2026-09-17 读取）：release `0.1.0` · library `90cfcdc720bb` · kernels `b3b3f9c16df7`。来源 https://gitcode.com/ddddwe/ascriptor.git / https://gitcode.com/ddddwe/ascriptor-kernels.git / https://gitcode.com/ddddwe/ascriptor-agent.git

| 算子 | 族 | 方向 | reference | sim | pipesim | emit | **compile** | board(cce) | 本仓接线 |
|---|---|---|---|---|---|---|---|---|---|
| `a5.gdn_fwd` | gated_delta_rule | forward | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ 未开始 |
| `a5.gdn_bwd` | gated_delta_rule | backward | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ 未开始 |
| **`a5.kda_fwd`** ★ | kda | forward | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ✅ | ✅ 完成 |
| **`a5.kda_fwd_stable`** ★ | kda | forward | ✅ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ 完成 |
| **`a5.kda_bwd`** ★ | kda | backward | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ✅ | ✅ 完成 |
| **`a5.kda_bwd_stable`** ★ | kda | backward | ✅ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ 完成 |
| `a5.delta_rule_fwd` | delta_rule | forward | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ 未开始 |
| `a5.delta_rule_bwd` | delta_rule | backward | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ 未开始 |
| `a5.kda_fused_recurrent` | kda | forward | ✅ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ 完成 |
| `a5.gdn2_fused_recurrent` | gdn2 | forward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| `a5.gdn2_fused_decode` | gdn2 | forward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| `a5.gdn2_short_conv_decode` | gdn2 | forward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| `a5.gdn2_norm2_w12_swiglu` | gdn2 | forward | ✅ | ✅ | ✅ | ✅ | ⬜ | ⬜ | done-explicit-opt-in |

★ 标记第一期的首个目标。

> **compile** 一列指 `ascriptor compile` CLI（纯源码发射+编译、不执行），**不是** aclnn launcher —— unit runner 把 board / aclnn / pypto 都记为 `board` stage。本仓的 aclnn 本地编译与零拷贝调用已独立实测通过，见下方 `our_runtime_bridge`。

### 缺失的算子

- **kda_fused_recurrent**（kda）— decode 路径（逐 token 递推 + state 传递）。**KDA 那半已经做完**（2026-09-11）：本仓自写 `a5.kda_fused_recurrent` 并接进 `layers/kda.py`，prefill/decode 一致性验过（见该条目的 our_status）。剩下的是 GDN / DeltaNet 的 decode（随各自扩族，第四期），以及 decode 的性能 —— 整层一步 458µs、瓶颈在层侧不在算子，见 gaps.json 的 decode-layer-overhead。chunk↔recurrent 互验这个 oracle 现在 KDA 上**已经有了**。
- **gdn_fused_recurrent**（gated_delta_rule）— decode 路径。同上。随 GDN 扩族（第四期）再补。
- **gdn2_chunk_fwd_bwd**（gdn2）— GDN-2 训练与长 prefill；channel-wise erase/write/decay。形状 K=V=128、H=HV=16 可复用定尺与分块经验，但输入必须新增长度 K 的 b/g 和长度 V 的 w；见 gdn2-abi-not-gdn。数值算法不能照搬 KDA：真实权重的有效 T=4096 stress case 已观测 64-token 累计衰减跨度 1461，见 gdn2-chunk-gate-range。

### 可复用原语

| 原语 | 对应 fla | 本仓状态 |
|---|---|---|
| `chunk_row_scan` | fla 的 chunk cumsum（ops/utils/cumsum.py 的 chunk_local_cumsum） | ⬜ 未开始 |
| `matrix_normalization.row_l2` | fla.modules.l2norm | ⬜ 未开始 |
| `gated_approximations` | fla.modules.activations / fused swiglu | ⬜ 未开始 |

### 性能基线

> 两条基线。基线一：torch_npu 原生算子拼出的同语义 KDA（ascend_fla/reference/kda.py 的 kda_chunk_vectorized），它同时是第二 oracle。基线二：本仓经 runtime 桥调用的 ascriptor 自编译算子（ascend_fla/ops/kda/chunk.py）。两者**同机同卡同输入**测量，否则加速比无意义。报数必须带形状/dtype/warmup/iters/是否同步 —— 见 AGENTS.md §6。

机器：8-card Ascend950PR docker host, CANN 9.2.0 (V100R001C25B046), torch 2.10.0+cpu / torch_npu 2.10.0.post2, python 3.11.16, NPU 0 · 记录于 2026-09-11

条件：dtype=bfloat16 K=V=128 chunk=64 warmup=3 iters=10 synchronized=yes forward_only=yes

> 与 ascriptor_self_compiled 同一批运行测得（每个 block_dim 的子进程各自重测一次基线），所以两张表可以直接比。先前在 NPU 7 / iters=5 下测的 5.866 / 6.497 / 7.040 / 14.469ms 已被这组取代 —— 条件不同的数不能混用。

| 形状 | B/H/HV/T | 四次测量 (ms) | 中位数 | o relL2 vs CPU |
|---|---|---|---|---|
| smoke | B1/H1/HV1/T64 | 6.633 / 6.473 / 4.581 / 4.684 | 5.579 | 1.898e-05 |
| kimi_linear_layer | B1/H32/HV32/T1024 | 7.920 / 5.445 / 5.439 / 5.811 | 5.628 | 2.721e-05 |
| qwen3_next_layer | B1/H16/HV32/T1024 | 8.051 / 5.524 / 5.115 / 5.697 | 5.611 | 2.765e-05 |
| long_context | B1/H16/HV32/T4096 | 17.668 / 12.155 / 11.816 / 12.223 | 12.189 | 2.622e-05 |

**波动**：四次测量里**第一个子进程始终最慢**（6.633 / 7.920 / 8.051 / 17.668），应是设备初始化与缓存冷启的成本漏进了它的计时。中位数比均值更能代表稳态。


> 每个 block_dim 一个子进程 —— 同名算子多 build 在一进程内会互相覆盖，见 gaps.json 的 op-name-collision-in-process。四个 block_dim 的 relL2 完全相同。

| 形状 | B/H/HV/T | bd=1 | bd=2 | bd=3 | bd=4 | bd1→4 | o relL2 |
|---|---|---|---|---|---|---|---|
| smoke | B1/H1/HV1/T64 | 0.334 | 0.238 | 0.222 | 0.238 | 1.40x | 3.288e-03 |
| kimi_linear_layer | B1/H32/HV32/T1024 | 4.955 | 2.516 | 1.724 | 1.311 | 3.78x | 3.962e-03 |
| qwen3_next_layer | B1/H16/HV32/T1024 | 4.923 | 2.491 | 1.713 | 1.302 | 3.78x | 3.964e-03 |
| long_context | B1/H16/HV32/T4096 | 19.622 | 9.971 | 6.794 | 5.138 | 3.82x | 4.168e-03 |

**block_dim=4 下 vs torch_npu 基线**：smoke 19.7x · kimi_linear_layer 4.43x · qwen3_next_layer 4.38x · long_context 2.38x

> benchmarks/profile_bridge_overhead.py --sync-each 的设备耗时归因（ms/次）。同步会破坏流水，所以总和大于流水模式下的总耗时。

| 段 | bd=1 (ms) | bd=4 (ms) |
|---|---|---|
| `KdaSub2ScoreKernel` | 1.583 | 0.407 |
| `TrilInverse64V2StrictBf16Kernel` | 1.354 | 0.361 |
| `KdaSub45FusedKernel` | 1.311 | 0.329 |
| `KdaSub3WyKernel` | 0.486 | 0.129 |
| `KdaSub1GateKernel` | 0.198 | 0.058 |
| `layout_to_bhcld_x5` | 0.318 | 0.276 |
| `layout_from_bhcld` | 0.095 | 0.064 |
| `total` | 5.838 | 1.901 |

**怎么读**：layout 重排只占 7%（bd1）—— 我先前猜它是瓶颈，错了。大头是三个 kernel，而它们正是 ascriptor lint 里 15 处 ub_to_l1.nd2nz 的所在（见 gaps.json 的 kernel-nd2nz-suboptimal）。bd=4 下 layout 占比升到 18%，因为 kernel 侧随核数缩短而重排不随。


#### 训练步（fwd+bwd）

> 完整一步训练（fwd + bwd）对照 torch_npu 组合版 + autograd。benchmarks/bench_kda_train_step.py，warmup 2 / iters 5，同步，layout_device=npu，每个 block_dim 一个子进程。fwd 一列**不含**门控跨度检查（与第一期的数可比），检查的代价单列。

记录于 2026-09-11

| 形状 | B/H/HV/T | bd | fwd | +检查点 | fwd+bwd | torch_npu | 加速 |
|---|---|---|---|---|---|---|---|
| smoke | B1/H1/HV1/T64 | 1 | 0.194 | 0.409 | 1.504 | 15.085 | 10.03x |
| smoke | B1/H1/HV1/T64 | 4 | 0.240 | 0.558 | 1.732 | 17.251 | 9.96x |
| kimi_linear_layer | B1/H32/HV32/T1024 | 1 | 4.898 | 5.952 | 15.864 | 21.731 | 1.37x |
| kimi_linear_layer | B1/H32/HV32/T1024 | 4 | 1.314 | 2.379 | 5.069 | 24.463 | 4.83x |
| qwen3_next_layer | B1/H16/HV32/T1024 | 1 | 4.914 | 5.955 | 15.893 | 25.549 | 1.61x |
| qwen3_next_layer | B1/H16/HV32/T1024 | 4 | 1.335 | 2.446 | 5.154 | 26.089 | 5.06x |
| long_context | B1/H16/HV32/T4096 | 1 | 19.567 | 23.809 | 63.792 | 106.711 | 1.67x |
| long_context | B1/H16/HV32/T4096 | 4 | 5.134 | 9.485 | 20.275 | 114.935 | 5.67x |

kimi_linear_layer / bd=4 的拆分（ms）：fwd_kernels 1.314 · caches_host_side 1.065 · bwd_kernels 2.690 · gate_range_check 0.212 · total 5.069

**怎么读**：① 训练步的加速比（bd=4 下 4.83x~5.67x）**高于**仅前向的（4.4x）——torch_npu 侧的反向要穿过它那张 python 循环图，被 launch 开销支配得更厉害。② host 侧补检查点 1.065ms，占训练步 21%（bd=1 时只占 7%）。它是 torch 算子，**不随核数缩短**，所以 block_dim 越高占比越大 —— 抬高 block-dim-ceiling 之后这一项才真正凸显（见 fwd-caches-not-emitted）。③ 九个反向 kernel 2.690ms 占 53%，是训练步里最大的一块。④ 门控跨度检查 0.212ms —— 占训练步 4%、占仅前向 16%，可用 check_gate_range=False 关掉，但关掉后越界就是 NaN 而不是报错。


#### GDN-2 packed-inference

> 真实 95B checkpoint，B=1/H=HV=16/K=V=128，BF16；prompt T=6，decode T=1；warmup=100、iters=200、每轮末同步；host 打包后 H2D，canonical/packed 顺序加载并用相反顺序复测。绝对延迟对加载顺序敏感，因此报区间并取较小加速比作保守值

环境：Ascend950PR / CANN 9.1.0 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 · 记录于 2026-09-14

| 加载顺序 | canonical (ms/token) | packed (ms/token) | 加速 |
|---|---:|---:|---:|
| canonical-first | 11.625 | 9.822 | 1.184x |
| packed-first | 10.184 | 9.212 | 1.106x |

**数值一致性**：正式闸为同 dtype relative-L2 + cache offset：FP32 T=6 预算 1e-5，BF16 T=6 与确定性 T=64 预算 1e-2；prompt/下一 token logits、18 层 recurrent state、canonical 三份 conv cache 拼接 vs packed cache 均 passed。bitwise 与 argmax 仅作诊断。参数数均为 1,450,096,416；canonical state_dict 399 项，packed 在加入SwiGLU W1/W2 packing后为255项（此前只打包mixer时273项）。最新原生RMSNorm+W1/W2版本的真实BF16 canonical↔packed为prompt/step logits relative-L2=1.252e-3/3.937e-3，最差step recurrent/conv cache=7.239e-3/4.282e-3，仍过1e-2。另有 tiny FP32 CPU↔torch_npu数值测试，最大relative-L2=5.740e-4（预算1e-3）。

**端到端冒烟**：最终 host-packed loader 以同 prompt 与 greedy 配置生成预期的 32 token；单次 0.502s（63.7 tok/s）。这里含首次执行影响，只作端到端 smoke，不作为稳态加速比。

**打包边界**：q/k/v/b/w 五个无 bias 投影按输出行合成一次 F.linear；q/k/v depthwise short-conv 与 cache 沿 channel 合并。f/output-gate 低秩首层仅在 B=T=1 时合并：T>1 改用同一packed weight的两个连续row slice。SwiGLU W1/W2同样在host按行打包且不保留重复参数：T=1用一个[2304→12416] GEMV，prompt/T>1用两个连续row slice保持原尺寸路径。block/final RMSNorm weight保留模型dtype供原生NPU fused op；fused-decode的o_norm weight继续一次扩宽为FP32。所有打包在host完成后再H2D。

**内存**：同参数数证明没有常驻 canonical+packed 双份权重；host packing 后实测 forward allocated 为 canonical 2,980,944,896 bytes、packed 3,007,683,584 bytes，reserved 为 3,221,225,472 / 3,141,533,696 bytes，消除了目标设备逐层 cat 的临时 reserve。

**证据**：benchmarks/verify_gdn2_packed.py、benchmarks/verify_gdn2_cpu_npu.py；tmp/gdn2-95b/cpu-npu-fp32-final.json、verify-packed-host-pack-{bf16,fp32,t64-bf16}.json、verify-packed-steady-{canonical,packed}-first.json、packed-cache-chain.json、text-greedy-packed-host-pack-final.json 及原始日志（git-ignored）


**观察**：torch_npu 基线：数据量从 smoke 到 kimi_linear_layer 差 512 倍，耗时只差 ~15% —— 它完全被 kernel launch 开销支配（向量化后仍有 63 次求逆迭代 + NT 次 chunk 迭代的 python 循环），**不是硬件算力上限**。自编译侧相反：耗时随工作量近线性（T 从 1024 到 4096，bd4 下 1.311→5.138ms，正好 3.9 倍），是真正的算力账。这也解释了 smoke 上 19.7x 的加速 —— 那里 torch_npu 在付固定开销而我们不付。

**跨 CANN 版本一致性**：kda_fwd 经 runtime 桥在 CANN 9.1.0 与 9.2.0 两台机器上的 relL2 逐位相同（smoke 3.288e-03 / multi_chunk 3.383e-03 / gva 3.359e-03），说明这个偏差来自算子自身的数值路径（见 gaps.json 的 kda-fwd-bwd-dtype-mismatch），与 CANN 版本无关。

**测量注意**：torch_npu 基线的跑间波动约 20%（kimi_linear_layer 在四个子进程里测到 7.920 / 5.445 / 5.439 / 5.811ms），所以加速比带同等量级的不确定度。每行的比值用的是该子进程自己测的基线，不是跨进程平均。

**下一步**：① 抬高 block_dim 上限（gaps.json 的 block-dim-ceiling，已升 P1）—— 扩展性到 4 仍线性，物理上有 28 cube。② kernel 侧的 nd2nz 返工（kernel-nd2nz-suboptimal）。③ 反向的同类测量，第二期随 kda_bwd 一起做。

### 全链路三层

> 全链路的上三层。窄切片原则：按算子倒推，用到哪个做哪个。首个目标是 KDA 链路，其依赖面比 GDN 少一个 module。

> ``done-torch`` 表示功能完成但核心实现是 torch 原生算子拼的；``done-cce-inference`` 表示自编译 CCE 核心已接入推理、但训练/长 prefill 仍有显式缺口；``done`` 给范围内前后向均完整的自编译算子。

**modules**

- `causal_conv1d` — 🔶 完成（torch 实现） · none — 需新写
  - 证据：2026-09-11 第二期：ascend_fla/modules/convolution.py ShortConvolution。2026-09-14 GDN-2 架构优化新增 PackedShortConvolution：q/k/v 三路 depthwise conv 与 cache 沿 channel 合并；CPU 覆盖无 cache、短序列补零、分段 cache 与规格拒绝，tests/test_modules.py 共 13 项。真实 95B 权重 T=6/T=64 的 packed conv state 与 canonical 三份 state 拼接后 relative-L2=0；本次也逐位相同，但只作诊断。
- `fused_rms_norm_gated` — 🔶 完成（torch 实现） · partial — matrix_normalization 可借
  - 证据：2026-09-11 第二期：ascend_fla/modules/fused_norm_gated.py FusedRMSNormGated；tests/test_modules.py 纯 CPU 11 项通过。2026-09-15：GDN-2 真实 B1T1H16 BF16 packed 的 swish output-norm/gate 已并入 a5.gdn2_fused_decode，通用模块与 KDA sigmoid 变体仍是 torch；不能把模型专用融合外推为整项 CCE 完成。
- `rms_norm` — 🔶 完成（torch 实现） · partial — matrix_normalization 可借
  - 证据：2026-09-14：ascend_fla/models/gdn2.py 的 RMSNorm 在 fp32 归一化、回写输入 dtype；GDN-2 CPU 测试与 NPU 冒烟均经过它。2026-09-15 packed inference 将已按请求dtype量化的55个不变norm参数一次性扩到FP32，消掉每token weight Cast，但37个block/final RMSNorm本体仍是torch_npu。
- `l2norm` — 🔶 完成（torch 实现） · matrix_normalization.row_l2
  - 证据：2026-09-11：layers/kda.py 里用 F.normalize 在 fp32 下做。2026-09-14：models/gdn2.py 按训练 recurrent kernel 的 sum(x²)+1e-6 语义在 fp32 做，并保留随后 q/sqrt(K) 缩放。

**layers**

- `kda` — 🔶 完成（torch 实现）
  - 证据：2026-09-11 第二期：ascend_fla/layers/kda.py KimiDeltaAttention，参数名与 fla 逐项对齐（KDA 算子自编译，周边 modules 是 torch —— modules-are-torch-not-kernels）。层级梯度实测：三个形状下输出相对 L2 4.9e-03，全部 17 个参数的梯度在 3.6e-03~2.2e-02，预算 0.1（A_log/dt_bias/f_proj 用 0.25，因为它们的梯度直接由 dg 来）。参考是同一份权重的 CPU 层，只把 KDA 算子换成 fp32 逐 token 递推版。承担了 fla 放在 kernel 里的三件事（q/k 的 l2norm、门控变换、beta sigmoid）。默认初始化（跨度 ~94）另有两项：前向对递推 oracle 相对 L2 4.697e-03（已测）；整层反向（门控跨度校准到 94）对同一份权重的 CPU 层逐参数比对，18 项全在预算 0.25 内（output 4.694e-03、dx 9.024e-03、A_log 1.551e-01、dt_bias 6.542e-02、f_proj 5.0e-02/5.2e-02，其余 4.5e-03~1.1e-02），由 test_deep_gate_backward_matches_cpu_reference 盯，已在有 ascend950 算子包的机器上跑通。梯度对齐那三个形状是在 exp(A_log)=1 下测的 —— 为的是把「接线对不对」和「深衰减下 bf16 本来就糙」分开，不是因为默认初始化跑不了。decode 路径未接（fused-recurrent-missing）。 **decode 已接线**：prefill 走 chunk + 空 cache，之后每步 fused_recurrent 传同一个 cache；prefill 128 + 逐 token 解码 5 步对整段 CPU 参考 4.26e-03~5.03e-03，由 test_prefill_then_decode_matches_one_shot_reference 盯。性能另见 decode-layer-overhead。
- `gated_deltanet` — ⬜ 未开始
- `gdn2` — ✅ 完成（CCE 推理）
  - 证据：2026-09-14：先完成 layers/reference/checkpoint/packed-inference 架构，再新增本仓 Ascriptor CCE `a5.gdn2_fused_recurrent` 并通过显式 core_backend 接入；未知 backend、CCE 求导、T>16 与 packed 求导均报错，不静默 fallback。canonical 保持发布 checkpoint 的399 keys；packed-inference加入SwiGLU W1/W2 host packing后为255个内部entry，参数数仍为1,450,096,416。算子五个aclnn contract case全过，真实95B的BF16/FP32整网torch_npu↔CCE logits与两类cache均在relative-L2预算内，cache offset/argmax一致。随后fused decode、short-conv、stateful Graph、原生RMSNorm与W1/W2 packing把纯模型decode推进到375.37~376.86 token/s，并保留相同64-token greedy序列。训练和长prefill仍缺gdn2_chunk_fwd_bwd。

**models**

- `kimi-linear` — ⬜ 未开始 · 注入 — 用上游模型定义，替换 linear attention layer
- `qwen3-next` — ⬜ 未开始 · 注入 — 用 HF transformers 的模型定义，替换 linear attention layer
- `gdn2-1.3b-fineweb-edu-100b` — ✅ 完成（CCE 推理） · 独立 LitGPT-compatible canonical 基线 + inference-only packed 布局；core backend 显式分派，后续只替换 GDN-2 算子边界
  - 证据：2026-09-14 实权重完成：95B checkpoint 17,401,727,659 bytes / sha256 4ac729c6…f6d；canonical 399 项 strict load，packed-inference 参数数不变。通用 CCE recurrent 的 BF16/FP32 数值与文本均通过。2026-09-15 追加 a5.gdn2_fused_decode：真实 BF16 T1 把 raw gates、recurrence 与 output norm 合成一launch，Cast309→74、设备约5108.8→4020us/token；反序paired speedup 1.410x~1.650x。静态双槽进一步把kernel即时对照4.733/4.716us降到合并4.247/4.256us，四AIV pipe利用率和86.84%→111.55%，证明成对流水已生效。step logits=3.436e-3、最差recurrent cache=9.521e-3（预算1e-2），32-token续写token ids不变。单pipe约80%的目标仍受erase→delta全局join限制（最忙MTE2平均37.52%）；训练与长prefill仍被gdn2_chunk_fwd_bwd阻塞。证据：tmp/gdn2-recurrent/ 与 tmp/gdn2-cast-fusion/（git-ignored）。

## 缺口

P0 2 项 · P1 24 项 · P2 16 项 · 已解决 13 项 · 共 55 项

**第一期里程碑**：第一期五项已全部有结论，并补齐了同机性能对比：aclnn 编译、runtime 桥、kda_fwd 接线、KDA 本地基线均实测通过；自编译算子在 block_dim=4 下比 torch_npu 组合快 4.43x（kimi_linear_layer）/ 2.38x（long_context T=4096）/ 19.7x（smoke）。过程中修掉两个自己的 bug（bridge-per-call-overhead、op-name-collision-in-process），它们先后让 block_dim 的效果被完全掩盖。当前最大的性能项是 block-dim-ceiling（已升 P1）：扩展性一路线性到契约上限 4，而硬件有 28 cube。第二期的前置障碍 kda-fwd-bwd-dtype-mismatch 已量化（降 P2）。

**第二期里程碑**：第二期已完成：kda_bwd 九 kernel 链 + 九个前向检查点 + autograd + KDA layer（含 2 modules），训练步在 bd=4 下比 torch_npu 组合版快 4.83x~5.67x。
过程中发现并根治了第二期最关键的一个缺口：**按 fla 的默认初始化（chunk 内门控跨度 ~94），上游 kernel 在前向与反向各有一处量程失效，方向相反** —— 前向在 87.3 下溢、反向在 88.72 上溢。两处都在本仓建了派生单元（kda_fwd_stable / kda_bwd_stable），把成对衰减分解的锚点从区间端点改到中点，数学同义。
门控闸因此做成二维 `{impl: {forward, backward}}`：前向 155（受有限性约束，到 155.97 精度完全不退化）、反向 100（受**精度**约束，它先于有限性到来 —— 梯度到 169.8 都有限但 dq 在 130 处超预算）。**「不吐 NaN」不等于「能用」，两者要分别测、闸按更严的定。**
剩一项端到端确认（默认初始化下整层反向的精度）卡在机器可用性上 —— 算子层面的同一件事已由 test_kda_bwd_deep_npu.py 在跨度 94 测过，六项梯度全在契约预算内。
当前最大的性能项仍是 block-dim-ceiling（P1）。

**建议的首个目标**：KDA（Kimi-Linear / fla-kda-default 形状）。其 ABI 已是 token-major BTHK、GQA 原生支持、initial_state 与 final_state 均为 FP32、backward 产出 dh0 —— 上述多数 ABI 缺口对它都不适用。唯一需要前置补齐的是本地验证证据（kda-no-local-evidence）。

### 为什么首个目标是 KDA

> 选择 KDA 作为首个目标的依据：下列能力 KDA 有、GDN 没有。

| 仅 KDA 具备 | 仅 GDN 具备 |
|---|---|
| GQA 分组 (HV % H == 0) | 本地 validation.json 证据齐全 |
| token-major 公开布局 |  |
| 非零 initial_state 输入 |  |
| backward 产出 dh0 |  |
| final_state 为 FP32 |  |
| block_dim 上限 4 而非 2 |  |

### 按算子族速查

| 算子族 | P0 | P1 | P2 |
|---|---|---|---|
| KDA | `c1-multihead-o-corrupt`<br>`ascriptor-gm-transfer-two-slice-row-gap` | `decode-call-overhead`<br>`decode-layer-overhead`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`no-tail-path`<br>`block-dim-ceiling`<br>`qk-l2norm-not-in-kernel`<br>`state-layout-k-first`<br>`kda-bwd-inverse-mm-mutex-over-budget`<br>`a2-splitk-fp32-cube`<br>`a2-splitk-bf16-fp16-unsettled`<br>`a2-cast-blkstride-sim-blind`<br>`kda-bwd-scan-dh0-nondeterministic`<br>`kda-prep-backward-training-step-slowdown`<br>`a2-kda-bwd-gcumsum-bf16-precision` | `kda-fwd-bwd-dtype-mismatch`<br>`npu-builtin-ops-missing`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`kernel-nd2nz-suboptimal`<br>`fwd-caches-not-emitted`<br>`modules-are-torch-not-kernels`<br>`stable-unit-no-harness`<br>`gate-span-still-bounded`<br>`a2-sim-vs-toolchain-blind-spots`<br>`a2-autosync-missing-cross-pipe-guards`<br>`a2-kda-bwd-pair-checkpoint-thin-margin`<br>`kda-prep-nearzero-output-underflow`<br>`kda-prep-backward-endpoint-disclosures`<br>`pkda-fp32-host-domain-validation-unregistered`<br>`soc-defaults-a5-outside-kda` |
| GDN | `ascriptor-gm-transfer-two-slice-row-gap` | `gdn-no-gqa`<br>`layout-not-token-major`<br>`nonzero-initial-state`<br>`d-initial-state-absent`<br>`state-dtype-bf16`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`scale-param-no-slot`<br>`no-tail-path`<br>`block-dim-ceiling`<br>`a2-splitk-fp32-cube`<br>`a2-splitk-bf16-fp16-unsettled` | `npu-builtin-ops-missing`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`soc-defaults-a5-outside-kda` |
| GDN-2 | `ascriptor-gm-transfer-two-slice-row-gap` | `no-varlen`<br>`no-tail-path`<br>`gdn2-abi-not-gdn`<br>`gdn2-chunk-gate-range`<br>`gdn2-decode-fragmentation` | `npu-builtin-ops-missing`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`modules-are-torch-not-kernels`<br>`soc-defaults-a5-outside-kda` |
| DeltaNet | `ascriptor-gm-transfer-two-slice-row-gap` | `layout-not-token-major`<br>`nonzero-initial-state`<br>`d-initial-state-absent`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`scale-param-no-slot`<br>`no-tail-path` | `npu-builtin-ops-missing`<br>`fixed-kv-128`<br>`asymmetric-kv-dim` |

### 待统一修复的 kernel 问题

> **kernel 源码层面的问题统一修一轮，不零散改。** 这是 2026-09-11 定的：发现一条就去改一条，会在 ascriptor 侧留下一串互相干扰的小改动，而且每改一次都要重跑全部 case。做法：发现时把它记进本表并打 `requires_kernel_change`，本仓侧先按 AGENTS.md §7 **装闸报错或记为声明限制**，保证不静默出错；等攒够一批再统一进 ascriptor 侧（§3：本仓不改那个仓，要改走那一侧的流程或建本仓派生单元）。当前队列 28 项，见下表。

| 缺口 | 级别 | 要在 kernel 侧改什么 |
|---|---|---|
| `c1-multihead-o-corrupt` | P0 | **根因已定位**（A2-04 / PR #60，joshjms 诊断，PM 独立复算）：`kda_fwd/kernels/recurrent.py` 的 `Aqk` L1 交接是**两信用配固定槽** —— `aqk_l1_valid = DEvent(Pipe.MTE1, Pipe.MTE2, preset=True)`（:130）给两个信用，而槽是 `aqk_slot = Var(c_idx % 2)`（:243 写、:373 读），按 chunk 取。一个头最后一个 chunk 的槽是 `(C-1)%2`，下一个头第一个 chunk 的槽是 `0` —— **当且仅当 C 为奇数时两者相撞**，写方领先一周期踩进还没被读走的槽（:245 的 MTE2 写 与 :398 的 MTE1 读无序）。每核最后一个头后面没有写，所以恰好是对的。**不是漏了某次 DEvent/Mutex 调用**，是信用数与实际轮转的槽数不匹配 —— 与 ascriptor `library/docs/defects/M10-076-mutex-credits-and-handoff-slots.md` 同型（那一条在 autosync 里已修成『depth <= j 才算有序』，但本 kernel 是手写同步，不过 autosync）。**修法**：让槽按每核周期序号轮转 `((pair_idx - pair_begin) * C + c_idx) % 2`，两信用配两槽；或把两个 DEvent 降成 SEvent（少一周期 run-ahead）。 **补充（来自 A2-04 的 delta）**：`l1_Aqk` 是两槽 `DBuff`（:146）。备选修法是把 `aqk_l1_valid`/`aqk_l1_ready` 改 `SEvent`（去掉一拍 run-ahead，**性能未测**）。落地按 AGENTS.md §3 走本仓派生单元、进 kernel 批次。A5 上的硬判据：失效表 12 格全对、偶数 C 与未修版逐位相同、bd=1 与 bd=4 逐位相同。 **2026-09-19**：本仓派生单元已落地并接入公开调度（见 proposed_action）；上游 ascriptor 的 `kda_fwd/kernels/recurrent.py` 源码仍未改，`requires_kernel_change` 因此保持 true（指上游）。 |
| `a2-kda-bwd-gcumsum-bf16-precision` | P1 | 只改本仓 A2 派生单元（`kernels/projects/a2/kda_bwd_stable/**`），不碰 A5 源码、不碰上游 ascriptor：新建派生单元（不是原地改现有冻结单元），四个消费者 `unit.py:112/121/135/149/172` 的 `gc2` 输入类型从 BF16 改 FP32（GM/UB 声明 + 载入/转换），`scan_fused`/`inverse_epilogue`/`finalize_pre`/`finalize_post` 四个 kernel 跟着改，其余五个 kernel 与其余 8 项缓存不动。需要新 UB/ABI/同步 preflight、全部 vendor 预编译、完整真机验收。属 kernel 批次候选，要用户批准（同 `a2-splitk-fp32-cube`/`kda-bwd-scan-dh0-nondeterministic` 的先例）。 |
| `a2-splitk-fp32-cube` | P1 | 只改 A2 派生单元（A2-03 前向 + decode 与 A2-09 反向的 `kernels/projects/a2/**`），A5 源码不动：`kda_fwd/kernels/triangular_inverse.py:273/279/284-285`、`gdn_fwd/kernels/inverse.py:306/312/317-318`、`gdn_bwd/kernels/finalize.py:205/217/226/235/244` 的 FP32 `is_init=False` matmul 之前补 `barrier(Pipe.M)`；`kda_fwd_stable/kernels/recurrent.py:401`、`gdn_bwd/kernels/wu.py:206` 的 BF16 累加链同样补（不分 dtype，A2-01 的 W4）。以 a2 lint 0 trap 作闸，验证按 `benchmarks/a2/README.md` §8 的 W2。属 kernel 批次 A2-K1，要用户批准。 |
| `block-dim-ceiling` | P1 | kda_fwd/kda_bwd 的 contract domain.block_dim 上限由 4 抬高并补 case。物理 28 cube / 56 vec，实测到 4 仍是线性扩展，所以这是当前最大的单点性能头寸。 |
| `d-initial-state-absent` | P1 | gdn / delta_rule 的 backward 产出 dh0。 |
| `decode-call-overhead` | P1 | 若要消掉 host 侧 15.4µs 的布局转换：kda_fused_recurrent 改成直接吃 token-major [B,T,H/HV,128] 并在 kernel 内按 hv//groups 取 q/k 的头。桥侧那 25µs 不用改 kernel。 |
| `fused-recurrent-missing` | P1 | KDA 的 decode kernel 已自写完（kernels/projects/a5/kda_fused_recurrent）。剩下的是 GDN / DeltaNet 的 decode kernel，照 KDA 这个的结构做。 |
| `gdn-no-gqa` | P1 | gdn_fwd/bwd 加独立 value-head 维度。 |
| `gdn2-abi-not-gdn` | P1 | recurrent 已由本仓 a5.gdn2_fused_recurrent 解决；队列中只剩独立 gdn2 chunk fwd/bwd，正式 ABI 需要 channel-wise g[B,T,H,K]、b[B,T,H,K]、w[B,T,H,V]。 |
| `gdn2-chunk-gate-range` | P1 | 新建 gdn2 chunk fwd/bwd 时必须从一开始采用覆盖至少已观测 1461 局部跨度的数值表示；禁止直接复制 KDA stable 的 105/155 跨度实现。 |
| `gdn2-decode-fragmentation` | P1 | BF16 raw-gate recurrent+output-norm、qkv short-conv+cache、RMSNorm2+paired W1/W2+SiLU×Mul三个模型专用CCE单元均已通过整网/profile。后续大步优化需新建weight-only低精度GEMV及质量验证链。 |
| `kda-bwd-inverse-mm-mutex-over-budget` | P1 | 要改 ascriptor 的 `a5.kda_bwd/kernels/inverse_mm.py`：`l0c_dvh`、`l0c_dvbeta` 由 `DBuff` 改为单槽 `Tensor`（并去掉这两块在使用点的 `[pipe_work]` 下标），互斥锁 34→32；算式、事件信用、循环、lookahead 不动。本仓派生单元已落地并通过原生验证（见 proposed_action），上游源码仍未改。 |
| `kda-bwd-scan-dh0-nondeterministic` | P1 | 疑在上游只读 `kda_bwd/kernels/scan_fused.py`：`seed_ub`（241 行声明）在每个 BHV 边界重置槽 0，与 `seed_mutex`（212 行，`CvMutex(3, depth=2)`）的跨 BHV 连续信用流转冲突，在奇数 chunk 数时对下一 BHV 首次发布（277 行）与上一 BHV 最终 VF 读取（358 行，实际 load 160-161 行）之间产生 WAR。A5K-03 已给出精确机制与两个可证伪预测的正向证据（详见 evidence 字段），修法方向已提出但未验收（维持跨 BHV 槽相位连续，或设计归还协议；信用不能简单降到 1）。修法只能是本仓派生 scan 单元，不改上游；要用户批准 kernel 批次。 |
| `layout-not-token-major` | P1 | gdn / delta_rule 的公开布局改 token-major。 |
| `no-tail-path` | P1 | kda kernel 加 partial chunk 的 tail 路径，解除 T % 64 == 0。 |
| `no-varlen` | P1 | kda kernel 支持 cu_seqlens（变长序列打包）。 |
| `nonzero-initial-state` | P1 | gdn / delta_rule 支持非零初始 state。 |
| `qk-l2norm-not-in-kernel` | P1 | 把 q/k 的 L2 归一化、门控变换、beta sigmoid 融进 kernel（fla 的 KDA 在 kernel 内做）。 |
| `scale-param-no-slot` | P1 | 给 scale 参数开 kernel 标量入口。 |
| `state-dtype-bf16` | P1 | gdn 的 final_state 由 bf16 改 fp32。 |
| `state-layout-k-first` | P1 | state 布局支持 v-first（fla 的 KDA layer 用 state_v_first=True）。 |
| `asymmetric-kv-dim` | P2 | 拆开 K 与 V 的维度，支持 head_k != head_v。 |
| `fixed-kv-128` | P2 | 解除 K=V=128 定尺。 |
| `fwd-caches-not-emitted` | P2 | kda_fwd 的 gate / wy / recurrent 各多写一个 GM 输出：g_cumsum（stable 已有）、v_new、h。搬进 kernel 后 _scan_states 与两处 CPU 绕行可以一起删掉。 |
| `gate-span-still-bounded` | P2 | 若要继续抬门控跨度：把 64×64 tile 按行列分块、每对子块用各自的中点（等价分块 log-sum-exp），有限性上限随分块数线性增长。**但反向的约束是精度不是有限性，这一项可能帮不上忙** —— 先做 kda-fwd-bwd-dtype-mismatch。 |
| `kda-fwd-bwd-dtype-mismatch` | P2 | kda_bwd 九个 kernel 的 g_cumsum 入参由 bf16 改 fp32（前向入口的 g 本来就是 fp32）。这是降低反向精度曲线的主要候选 —— 但**是推测，要测**。 |
| `kernel-nd2nz-suboptimal` | P2 | ascriptor lint 标出的访存低效点：nd2nz 展开、偶数 block stride 撞 UB bank。lint 信息里带了板上实测倍数与具体改法，照着做即可。 |
| `modules-are-torch-not-kernels` | P2 | GDN-2 B1T1 BF16 output norm/gate、packed qkv short-conv/cache与RMSNorm2+W1/W2+SiLU×Mul三个模型专用CCE单元均已通过整网/profile；通用causal_conv1d、FusedRMSNormGated、其他shape与训练路径仍是torch算子。 |

### P0

#### `c1-multihead-o-corrupt` — 【P0·静默错误】C 为奇数且一个 cube 核要处理多个头时，kda_sub45_fused_kernel 的 Aqk L1 交接竞争，写出内容错误的 o

- **类别** correctness · **适用于** KDA · **阻塞** —
- **依据** **这是真实形状精度验收的第一个产出，而且是最坏的一类缺陷：没有 NaN、没有报错、范数还正常。** 发现路径：按 models.json 的 kimi 形状扫 C=1…16，C=1 那档 `o` 的相对 L2 是 **1.06**（其余档 3.2e-03），而同一次运行的 `final_state` 正常（2.46e-03）。
**失效规律**（2026-09-11，Ascend950PR / CANN 9.2.0，B=1、H=1、跨度 46，正确 = 相对 L2 < 0.01）：
| block_dim | HV=2 | HV=4 | HV=8 | HV=16 |
|---|---|---|---|---|
| 1 | 仅头 1 | 仅头 3 | 仅头 7 | 仅头 15 |
| 2 | 全对 | 头 1,3 | 头 3,7 | 头 7,15 |
| 4 | 全对 | 全对 | 头 1,3,5,7 | 头 3,7,11,15 |
正确的恰好是**每个 cube 核分到的最后一个头**。kernel 的 `pair_begin = (B*HV * GetCubeIdx()) // GetCubeNum()` 按 cube 核切 `B*HV`，而 `GetCubeNum() == block_dim` —— 所以安全条件是 **`B*HV <= block_dim`**。
**只在 C=1 出现**：C=2/3/16/64 下全对（HV 到 32 都试过）。chunk 循环跑第二遍时补上了缺的那次同步。kernel 源码里 `recurrent.py:176` 写着 `auto sync is not used here because the nested for loops interfere with it` —— 同步是手写的，而手写的那份假设了 C≥2。
**范围已逐条核实**：① `upstream` 与 `stable` 的错误值**逐位相同**（都 2.721e-01）→ 是共享的 `kda_sub45_fused_kernel`，不是本仓的 stable 派生引入的；② `final_state` 不受影响；③ **反向不受影响** —— C=1/HV=8 的六项梯度 dq 2.48e-02 / dk 3.94e-02 / dv 3.22e-03 / dbeta 3.35e-03 / dg 7.16e-02 / dh0 2.40e-03，与 C=2 同量级，因为反向不消费 `o`，只消费 `do` 与九个检查点。
**为什么契约的 case 测不到**：`kda_fwd` 四个 case 里 C=1 的三个都是 HV=1，唯一 HV=2 的那个是 C=2 —— **`C=1 且 HV≥2` 一个 case 都没覆盖**。这正是 toy-case-shapes 说的那件事。

**2026-09-17 补：模型定位 + 逐格复算（a5 pipesim，library 90cfcdc / kernels b3b3f9c）。**
PM 在权威 workspace 上独立跑了 `benchmarks/diag_c1_multihead.py`，**上表被逐格复现**：bd=1/HV=2→仅头 1、bd=1/HV=4→仅头 3、bd=2/HV=2→全对、bd=2/HV=4→头 1,3。
**并且暴露面比上表宽：是奇数 C，不是只有 C=1。** B=1/HV=2/bd=1 扫 C=1…6，冒险条数 2 / 0 / 2 / 0 / 2 / 0，错头恒为头 0：
| C | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| pipesim 冒险 | 2 | 0 | 2 | 0 | 2 | 0 |
| 回放错头 | 头 0 | 无 | 头 0 | 无 | 头 0 | 无 |
候选补丁（槽按周期轮转）在 bd∈{1,2}×HV∈{2,4}×C∈{1,3,5} 共 12 格上冒险归零、全头逐位正确；负对照（只轮转 q/qg）缺陷原样保留。
**与真机记录的冲突要并排看**：真机 2026-09-11 测过 C=3 且『全对』。两者不矛盾 —— 无序 ≠ 必然发生，那一次时序没踩到。但**结构上 C=3/5/7… 同样暴露**，而闸只拦 C=1。对 kimi（C = T/64）来说 T=192、320 都是奇数 C。

**根因定位（A2-04，#30 / PR #60，library 627f55f / kernels c89f69b，a5 管线模型值）**：pipesim 报出的冒险全部是"本头的 Aqk L1 读（:398）↔ 同核下一头的 Aqk L1 写（:245）"无序，条数与归因逐格相等、无其它冒险。按调度回放 o：真机失效表 12/12 格逐头吻合（逐位正确的头 = 表中对的头）。用 verify_real_shapes.py 同款输入（跨度 46）预测：kimi H32/HV32/C1/bd4 整体 o 相对 L2 1.060（本条记录 1.06），H1/HV8/C1/bd1 为 2.721e-01（本条记录 2.721e-01），kimi final_state 2.458e-03（本条记录 2.46e-03）；错头 |o|/|ref| 0.9254~1.1319、全部有限。**C≥2 全对的原因不是"chunk 循环第二遍补上了同步"**：C 为偶数时相邻两次写的槽交替，信用数与槽数匹配；C 为奇数时每次换头相撞一次。L0C 输出握手各头一致，与缺陷无关。修补（aqk-only 槽轮转）在模型中 43/43 格与奇数 C 12/12 格冒险 0、全头正确，偶数 C 格与上游逐位相同；负对照（只改 q/qg 槽）不起作用。
**一条被推翻的旧解释**：本条原来写着 C≥2 全对是因为『chunk 循环跑第二遍时补上了缺的那次同步』。两半都错 —— 根因不是漏同步（是信用数与槽数不匹配），C≥2 也不全对（奇数 C 照样撞）。那句话是从『C=1 坏、C=2 好』这个症状规律倒推的。AGENTS.md §6 已同步更正。
2026-09-18 真机首次证实（A5K-01，#76，950PR_9589 V100，CANN 9.2.0，BF16 q/k/v/o、FP32 gate/state，B1 H=HV=32 K=V=128，span46，bd4）：T64/C1 output relL2=1.06026316（与历史记录 1.06 吻合），T192/C3 output relL2=0.689781666——C=3 在真机上确实错，不只是 pipesim 模型预测，此前奇数 C 结构性暴露的结论首次拿到真机证据支撑。T4096/C64（偶数）relL2=0.00326641649，在预算内。错误的输出逐元素 atol=.02 仍能通过（有限值、量级正常），现有 .05 相对 L2 闸正确拒绝了 C1/C3——闸没有失效，此前记录的闸只拦 C=1、奇数 C≥3 结构上同样暴露，现在有真机数据佐证。
- **影响** **2026-09-19 更新（A5K-02，#83，合并提交 2589942）——先读这一段，下面 ①~④ 描述的是修复前的行为，以及仍未修复的 upstream 路径：**
默认公开路径（`impl="stable"`，含 `chunk_kda_fwd` 与带 cache 的前向/训练前向）已改用修复后的 recurrent，奇数 C 且多头不再写出错误的 `o`。A5 真机（950PR_9589 V100 / CANN 9.2.0 / 算子包 ascend950,ascend910b,ascend910_93）经公开 API 验证：C=1..6 × 7 组 H/HV × 零/随机 state × bd=1..4 共 336 例、9240 个 head/chunk 行，对两个 CPU FP32 oracle 在未改的预算内（最差相对 L2：输出 .003904、状态 .004269）；Kimi 真实形状 B1/T4096/H32/HV32/bd4 的前向与完整 backward 通过；bd1..4 的输出/cache/prefix-state 字节完全一致；偶数 C 与 9 个 cache 相对原始逐位不变。真机数字由 assignee 报告，PM 无法复跑，按证据审。
`impl="upstream"` 仍是原始有缺陷的 kernel，但奇数 C 且 B*HV>block_dim 在布局/编译/发射之前报错（闸由 `C==1` 收紧为奇数 C），不再静默出错。
① **T=64 的前向输出是错的**（HV>block_dim 时），错得没有任何信号：有限值、量级正常、`final_state` 还对。短 prompt 的 prefill 正好落在这里 —— kimi 形状 HV=32、bd=4 时 32 个头里只有 4 个对。
② 训练同样中招：梯度本身没问题，但**前向输出错 → loss 错**，所以 T=64 的训练步是垃圾。
③ 之前所有精度结论都不受影响 —— 它们用的形状要么 HV=1（安全），要么 C≥2（安全）。这也是它藏了两期没被发现的原因。
④ **实测到的一个具体后果（decode 接线时撞上）**：64 token 粒度的 prefill 用不了 —— T=64 就是 C=1，HV=2 且 bd=1 时就已经越界。prefill 要么一次 ≥128 个 token，要么按头分批。写 prefill→decode 的测试时被闸拦下，只能把 prefill 从 64 改成 128。
- **建议** **已做**：`ops/kda/chunk.py` 的 `_check` 里加了 `_check_single_chunk_heads`，`C==1 and B*HV > block_dim` 直接报错并给出两条绕法（AGENTS.md §7：绝不静默降级）。闸的边界照实测表逐个钉在 tests/test_kda_gating.py 里 —— 这类缺陷一旦闸被改松，没有别的东西会报警。
**闸目前是漏的（2026-09-17 发现，待所有者决定）**：`_check_single_chunk_heads` 只拦 `C==1 and B*HV > block_dim`，而模型显示奇数 C 全都暴露。收紧成 `C % 2 == 1 and B*HV > block_dim` 会拒掉一些现在能跑、真机上也确实跑过的形状（C=3 测过且过了），属于收紧可用面 —— 按 AGENTS.md §1「改变范围要显式决策」，等所有者放行再改。在那之前**这条缺口的影响面按奇数 C 记**，不要按 C=1。
**要做（按代价排序）**：
① 按头分批调用：C=1 时把 `B*HV` 切成每批 ≤ block_dim 个头，多发几次 kernel。数学完全不变（头之间独立，已由头独立性检查证明），代价是多几次发射。**这是可用性修复，但会悄悄改变性能特征，要显式声明而不是默默做掉。**
② 建本仓派生单元修手写同步（照 kda_fwd_stable / kda_bwd_stable 的先例，AGENTS.md §3 不改 ascriptor 仓）。要先读懂 `recurrent.py` 的 DEvent/Mutex 配对 —— 目前只掌握了**症状规律**（每核最后一个头对）而不是确切缺哪一次同步，动手前必须先把那个找出来，否则改了也不知道为什么好。
③ 上游补 case：`C=1 且 HV≥2`。这条不管我们怎么修都该做，否则上游下次改这个 kernel 还会踩。
⑤ **上游补 case**（A2-04 建议）：`kda_fwd` 契约要加 `C=1 且 HV≥2`，以及 `奇数 C≥3 且 B*HV > block_dim`。现有四个 case 一个都盖不到。
⑥ **真机侧要补的**：本条记的『C=3 全对』在仓里**没有对应的运行日志**。闸的范围要按奇数 C 定的话，得先在 A5 上补 C=3 / C=5 且 `B*HV > block_dim` 的逐 chunk 比对。
2026-09-18 修复已落地（A5K-01，#78）：kernels/projects/a5/kda_fwd_stable/kernels/recurrent.py 把槽表达式改成 Var(((pair_idx - pair_begin) * C + c_idx) % 2)（写读两处都改），与 A2-04 pipesim 验证过的方案一致。真机确认：336/336 case（C=1..6 x 7 组 H/HV x 零/随机 state x bd=1..4，含 Kimi 真实形状 H=HV=32）全过，负对照（原始/qg-only）保留、真机上仍复现奇数 C 冒险；偶数 C 与未修改基线逐位相同，跨 bd=1..4 也逐位相同。PM 独立复算了不依赖真机的部分：repair_diagnostics.py 的 12 格缩小 pipesim 回归——负对照在奇数 C=1,3,5 冒险数=2、回放错误，修复版全部 C 冒险数=0、回放正确，与报告逐字一致。
**重要边界（未解决）**：修复只落在这个独立单元里，`ascend_fla/ops/kda/chunk.py` 的公开调度仍然选择原始（有缺陷的）recurrent——那个文件不在 A5K-01 的写集内，接线是一个需要仓库所有者另外决定的独立步骤。**在接线完成之前，公开 API 的用户仍然会撞上这个缺陷**，本条修复目前只是证明了可行，没有改变任何用户能感知到的行为。
**2026-09-19 接线已落地（A5K-02，#83，合并提交 2589942；上面『重要边界（未解决）』那一条现已解决）**：`ops/kda/chunk.py` 的 stable 选修复后的 recurrent（含 cached-forward）；upstream 的闸收紧为奇数 C 且 B*HV>block_dim；`repair_runtime.py` 的 baseline 显式钉回原始 recurrent，A5K-01 的负对照保留；新增 `tests/test_kda_aqk_dispatch.py` 盯住『公开调度确实选中修复版』。**仍未解决**：(a) stable 路径没有奇数 C 闸——C=1..6 与 T4096（C=64）已原生验证，其余 C 只有『槽位按全局计数取模』的构造性论证，未原生测；(b) 上游 ascriptor 的 kernel 源码本身仍有缺陷（本仓不改，AGENTS.md §3），上游补 case（C=1 且 HV≥2、奇数 C≥3 且 B*HV>block_dim）仍未做；(c) 只覆盖 A5、block_dim 1..4，A2/A3 未建立；(d) A5K-02 新测的原始输出失败计数是 36/30/30/12（bd1..4），与 A5K-01 记录的 bd3=26 不同，口径未统一，两者不要互相替代；(e) 13 个 backward 回归里 dq 的最坏值 .04782 离预算 .05 只有约 4.4% 余量（预算未改，非本次引入）。

#### `ascriptor-gm-transfer-two-slice-row-gap` — 【P0·静默错误·ascriptor 库缺陷】gm_transfer 的双切片分支漏算夹在中间的标量索引维，行间距算成 0

- **类别** correctness · **适用于** 全部 · **阻塞** —
- **依据** GDA-01（#73）第一轮 bulk-prepare 优化候选，源表达式 `qu[:,:] <<= q[bb, tt:tt+64, hh, :]`（4 维 GM 张量 [B,T,H,D]，B/H 是标量索引，T/D 是切片）。缩小形状 T64/H3/block_dim1 的 pipesim 报告有限但错误的 qn：24135/24576 个元素不匹配，max_abs=0.0234730169。生成的搬运调用是 `gm_to_ub_pad(...,64,512,0,0)`，行间距（第 3 个参数之后那个 gap）算成了 0，而正确值应为 `(H-1)*512` 字节。
**PM 已独立核实**：`ascriptor/passes/device_lower.py` 的 `gm_transfer` 函数，`len(sliced) == 2` 分支（对应本例：T 与 D 两个维度被切片，B 与 H 是标量索引，H 恰好夹在两个切片维之间）算 `burst = prod(extents[second:])`、`gap = prod(shape[second:]) - burst`——这两个式子只看第二个切片维（D）之后的形状，**没有把夹在第一个切片维（T）和第二个切片维（D）之间的标量索引维（H）的形状乘进去**。对比同文件 `len(sliced) == 1` 分支（101-102 行）用的是 `shape[d+1:]`，正确地把 d 之后所有维度（含标量索引维）都算了进去——`len(sliced)==2` 分支的这个不对称就是漏算的根源。
- **影响** 任何 kernel 里出现『4 维（或更高维）GM 张量、切两个维、且有标量索引维夹在这两个切片维中间』这个具体模式，都会静默算错行间距——不报错、不崩溃，产出有限但内容错误的值，和本仓已经踩过的 `c1-multihead-o-corrupt` 是同一类（有限值、量级看着正常、只有逐元素比值乱）。GDA-01 的 bulk-prepare 候选没有被采纳，已知的其它本仓 kernel 暂未复查是否命中这个模式——这条要单独扫一遍。
- **建议** 本仓侧（AGENTS.md §3：不改 ascriptor 仓）：在自己的派生单元里，遇到这个 4 维双切片模式一律换成显式的 `gm_to_ub_pad(..., 正确的 gap)`，不依赖库的隐式推断。GDA-01 已经验证这条绕法：`qn/kn/bk/wv` 换成显式 gap 后与 CPU 参考完全一致，event balance/hazards 均空、无死锁。
仓库所有者侧：这是 ascriptor 库本身的缺陷，不是本仓能修的范围，建议原样报给 ascriptor 的维护方。
待办：扫一遍本仓现有 kernel（kda_fwd_stable、gdn2_chunk_fwd 等）里有没有已经踩中这个模式但还没被测出来的地方——形状够小或凑巧连续时（比如 H=1）这个 bug 不会触发，容易被现有测试漏过。

### P1

#### `decode-call-overhead` — decode 的瓶颈是每次调用约 48µs 的固定成本，不是 kernel 也不是带宽

- **类别** performance · **适用于** KDA · **阻塞** `phase 3`
- **依据** kimi decode 形状 B1/HV32/T1（warmup 5 / iters 50 / 同步）：整次 **53~67 µs**，其中 host 布局转换 15.4~15.7 µs（26~29%）；T=1→16 的**设备侧边际 2.7~4.8 µs/token**（两次运行 T=16 整次分别 101.0 与 125.4 µs —— 报区间而不是单点），于是**每次调用的固定成本约 48~58 µs**。边际与按「4 头/核 × 2 趟 × 128 行」估的设备时间同量级。block_dim 1/2/4/8/16/28 的总时长 54~67 µs **没有趋势** —— 固定成本主导的征兆。
**我原本预测 decode 是带宽瓶颈（按 state 64KB×2/头估下限约 2.5µs），那个预测错了。**而且如果不拆开量，bd 无效会被误判成「扩展性不行」—— AGENTS.md §6 铁律一说的就是这个。
复现：`benchmarks/verify_decode.py --check split`。
- **影响** 48 层模型按 55µs/层算是 **2.6ms/token**，不可接受（decode 一步的预算是几十 µs 量级）。所以 decode 算子虽然正确，但还不能用；这是 fused-recurrent-missing 接层之前必须先解的。
- **建议** 按占比从大到小：
① **桥侧（约 25µs）**：decode 期间形状固定，`aclCreateTensor` 的描述符与 workspace 查询可以按 (算子, 形状签名) 缓存复用，每步只换 data_ptr。要先 profile 确认 25µs 花在哪一段 —— **别再先推断后看数据**。
② **host 布局（15.4µs）**：让 kernel 直接吃 token-major 并在 kernel 内做 GQA 取头，就不用 permute/repeat_interleave/contiguous。这会改 ABI，归入 kernel 修复队列。
③ **输出（约 7µs）**：同理，让 kernel 直接写 token-major。
④ 设备侧 4.8µs/token 先不动 —— 它已经是最小的一项。

#### `decode-layer-overhead` — 整层 decode 一步 458µs，其中 KDA 算子只占 18%，层里那十几个小算子占 67%

- **类别** performance · **适用于** KDA · **阻塞** `phase 3`
- **依据** `benchmarks/bench_kda_decode_layer.py`，hidden=2048 / H16 / HV32 / bd1 / prefill 128，warmup 10 / iters 50 / 同步（2026-09-11，CANN 9.2.0）：
| 项 | 耗时 | 占比 |
|---|---|---|
| 整层一步 | 458.0 µs | 100% |
| 其中 KDA 算子 | 83.1 µs | 18% |
| 其中层里其余部分 | 305.1 µs | 67% |
**48 层外推 22 ms/token。**
层里 T=1 时要走的调用：7 个投影（q/k/v/f/b/g/o）、3 个短卷积、softplus、sigmoid、两次 fp32 l2norm（含 bf16↔fp32 往返）、FusedRMSNormGated —— 每个都几乎没有计算量却各要一次 launch。所以这 305µs 基本是**启动开销之和**，不是算力。
拆法是逐段替换而不是推断（AGENTS.md §6 铁律一）：`without_op` 那条跑完层里除 KDA 算子以外的全部步骤，`t_op` 只跑算子。

**A2-40 更正（issue #54 / PR #85，2026-09-18）**：上面的测量形状是 `hidden=2048/H16/HV32`，**不是** Kimi-Linear 的真实形状（`hidden=2304/H=HV=32`，见 models.json）；引用这组数只用来说明"为什么要做"，不用来预测 A2 上的收益。另外层里实际是 **9 个** `nn.Linear`（f_proj/g_proj 各拆两段），不是 7 个——`packed_projection` 的打包方案要按 9 个算。
- **影响** decode 现在**功能可用但性能不可用**：22 ms/token 意味着 45 token/s，而同级别模型的目标是几十到上百倍于此。
注意优化的着力点：算子侧只占 18%，且其中设备时间只有几 µs —— **把 kernel 再优化一倍，整层只快 9%**。该动的是层这一侧。
- **建议** 按占比排序，且都要先 profile 再动手：
① **图捕获**（最大头）：decode 每步的形状完全固定，整层可以用 ACL/torch_npu 的 graph capture 录一次重放，把十几次 launch 压成一次。这是业界对 decode 的标准解法，但要确认 torch_npu 在本版本支持、且我们的自定义算子能进图。
② **合投影**：q/k/v 三个投影可以并成一个 `[hidden, 3*key_dim]`（权重拼接，数学不变）；f/b/g 同理。能把 7 次降到 3 次。
③ **去掉 l2norm 的 dtype 往返**：现在是 bf16→fp32→normalize→bf16。T=1 时这两次 cast 各是一次 launch。若 `qk-l2norm-not-in-kernel` 做掉（搬进 kernel），这一段连带消失。
④ 算子侧的 `decode-call-overhead` 只值 18% × 其中的固定成本，排在后面。

#### `gdn-no-gqa` — gdn_fwd/bwd 无独立 value-head 维度，不支持 GQA 分组

- **类别** abi · **适用于** GDN · **阻塞** `qwen3-next-80b-a3b`, `phase 4`
- **依据** gdn_fwd contract.json domain.shape："B,H,C are positive runtime dimensions" —— 没有 HV。对比 kda_fwd domain 有 "H_HV": "positive; HV % H == 0"，kda_bwd 亦然。
- **影响** 阻塞 Qwen3-Next（16 key heads / 32 value heads）。
- **建议** 第四期做 GDN 扩族时一并解决，实现可借鉴 kda 已有的 HV%H==0 分区方式。严重度因第一期改走 KDA 而由 P0 降至 P1。
2026-09-18：GDA-02（排上看板，D-PM-22）会给本仓派生单元 kernels/projects/a5/gdn_chunk_fwd 加上 GQA/GVA 分组，目的是解锁 PK-03（PGDN 前向），**不是解决这条缺口本身**——这条缺口记的是上游 ascriptor 单元 a5.gdn_fwd/bwd 缺独立 value-head 维度，阻塞的是 Qwen3-Next（第四期）。GDA-02 完成后不要误以为这条缺口已关闭；两者是不同的单元、不同的下游目标，只是解决方式（借鉴 KDA 的 HV%H==0）可能相通。

#### `layout-not-token-major` — gdn 与 delta_rule 的公开布局是 [B,H,C,L,D]，与 fla 的 token-major 不一致

- **类别** abi · **适用于** GDN / DeltaNet · **阻塞** `phase 4`
- **依据** gdn_fwd contract domain.layout："Contiguous CPU tensors in [B,H,C,L,D]"。对比 kda_fwd："Contiguous token-major public tensors; explicit local permutation to BHCLK/BHVCLK kernel tensors"，kda_bwd："Public tensors are contiguous BTHK/BTHV"。
- **影响** 接入 GDN 时调用方需要 permute，带来额外访存开销。KDA 不受影响 —— 它的公开接口已是 token-major，内部自行 permute。
- **建议** GDN 接线时让 kernel 内部做 permute（照 kda 的做法），而不是把转换推给调用方。

#### `nonzero-initial-state` — gdn 与 delta_rule 只支持零初始 state

- **类别** abi · **适用于** GDN / DeltaNet · **阻塞** `phase 4`
- **依据** gdn_fwd / delta_rule_fwd contract.json domain.initial_state: "zero only"。kda_fwd 则以 float32 [B,HV,128,128] 作为正式输入，case 中有 random 与 zero 两种。
- **影响** GDN/DeltaNet 无法做 state 传递 —— 长序列分段训练、prefill→decode 交接、chunked prefill 都做不了。KDA 不受影响。
- **建议** GDN 扩族时照 kda_fwd 的方式加非零初始 state 入口。在此之前，门控必须对 GDN 传入非零 initial_state 的调用报错。

#### `d-initial-state-absent` — gdn 与 delta_rule 的 backward 不产出初始 state 的梯度

- **类别** abi · **适用于** GDN / DeltaNet · **阻塞** `phase 4`
- **依据** gdn_bwd / delta_rule_bwd contract.json："no d_initial_state in the preserved backward production ABI"。对比 kda_bwd 有 dh0 输出（[B,HV,128,128]）与 dht 输入。
- **影响** GDN/DeltaNet 无法支持可训练初始 state、序列并行（CP）与分段反向。KDA 不受影响。
- **建议** 记录为 GDN/DeltaNet 的已知限制。门控应在 initial_state.requires_grad 时报错。

#### `state-dtype-bf16` — gdn 的 final_state 为 BF16，fla 惯例为 FP32

- **类别** precision · **适用于** GDN · **阻塞** —
- **依据** gdn_fwd contract outputs.final_state: bfloat16 [B,H,128,128]。对比 kda_fwd 的 final_state 为 float32。
- **影响** state 是跨 chunk 累积量，BF16 存储的误差会进入下一段递推。影响幅度未在 A5 上测过。KDA 不受影响。
- **建议** GDN 接线时测：同形状下 BF16 state 与 FP32 state 的输出差异，长 C（如 C=64）下是否放大。不要沿用 A2 上的结论。

#### `fused-recurrent-missing` — decode 路径：KDA 已自写 kernel 并接进 layer，性能不可用；GDN/DeltaNet 仍整族缺失

- **类别** coverage · **适用于** KDA / GDN / DeltaNet · **阻塞** `phase 3`
- **依据** ascriptor kernels catalog 中无 fused_recurrent 类单元；六个 a5 单元均为 chunk 路径。
**KDA 这半已经补上**（2026-09-11）：本仓自写 `kernels/projects/a5/kda_fused_recurrent`，state 常驻 UB、两趟扫、按 B*HV 切给向量核（每头一核，核间无同步）。`ascriptor check` 0 error / 0 warning / 156 ops；真机八个形状下 o 的相对 L2 9.575e-08~1.789e-07、final_state 3.199e-08~1.440e-07（对 fp32 递推参考）；**逐 token 调 T 次并串接 state 与一次调 T 个 token 逐位相同**（decode 正确性就靠这条）；block_dim 1~28 全通（28 即 56 个向量核的物理上限）。
**它还顺带消掉一个约束**：逐 token 只用 `exp(g_i)`（~1.5），所以**门控跨度没有上限** —— 对比 chunk 路径的前向 155 / 反向 105（gate-span-still-bounded）。
- **影响** **KDA 的 decode 现在功能上可用了**（2026-09-11 接进 `layers/kda.py`）：prefill 走 `mode="chunk"` 并传一个空 `cache`，之后每步 `mode="fused_recurrent"` 传同一个 `cache`（原地更新，装 `recurrent_state` 与三份 `conv_state`）。实测 prefill 128 token 后逐 token 解码 5 步，对整段 CPU fp32 参考的相对 L2：prefill 4.595e-03、decode 4.628e-03 / 5.030e-03 / 4.650e-03 / 4.257e-03 / 4.783e-03 —— **逐 token 报数**而不是报平均，因为 conv_state 漏传只会坏前 conv_size−1 个 token，平均会掩盖它。
剩下三条：
① **性能不可用**：整层一步 458µs、48 层 22ms/token，见 decode-layer-overhead（层侧 67%）与 decode-call-overhead（算子侧的固定成本）。
② **不可求导**：本仓只有 chunk 的反向 kernel。层里会直接报错而不是静默不建图。
③ GDN / DeltaNet 的 decode 仍整族缺失，随各自扩族再补（第四期）。
（**此前我在这里写过「ShortConvolution 只有整段前向」，那是错的** —— `modules/convolution.py` 本来就支持 `cache` + `output_final_state` 的单步解码，还处理了 T < kernel_size 的补零。接线时直接用上了。）
- **建议** ① 先解 decode-layer-overhead（层侧占 67%），再看 decode-call-overhead（算子侧）。**别先去优化 kernel** —— 它只占 18%，其中设备时间才几 µs。
② GDN / DeltaNet 的 decode 随扩族做，照 KDA 这个单元的结构抄（两趟扫 + 每头一核）。
③ 接 HF/fla 模型时要一层 cache 适配器：我们的 `cache` 是本仓自己的两键字典，而 fla 的 KDA layer 用 `state_v_first=True`（V 在前），见 state-layout-k-first。

#### `no-varlen` — 无变长序列（cu_seqlens）支持

- **类别** coverage · **适用于** 全部 · **阻塞** —
- **依据** kda_fwd contract domain.scope 明确写 "Fixed length; no cu_seqlens, cp_context, safe_gate, gate fusion or state_v_first"。其余单元 ABI 亦均为规整形状。
- **影响** 训练常用的 sequence packing 无法使用，等长 padding 会浪费算力。顺带：gate fusion 不支持，意味着 g 必须在 kernel 外算好再传入。
- **建议** 列为开放问题（见 docs/plan.md §7）。短期门控拒绝。gate 在外计算对 KDA layer 是自然的（f_proj 本就是 Linear），不构成阻塞。

#### `scale-param-no-slot` — fla 的 scale 参数在 ascriptor ABI 中无入口

- **类别** abi · **适用于** GDN / DeltaNet · **阻塞** `phase 4`
- **依据** 各单元 inputs 中均无 scale 标量。gdn/delta_rule 的 case parameters 里的 "scale": 0.05 是输入生成幅度（delta_rule_fwd domain.input_values: "generated q/k/v scale 0.05"；kda_fwd domain.input_generation: "q/k/v stddev 0.04"），不是算子参数。 【2026-09-11 修正】KDA 不受影响：kda_sub2_score_kernel 与 kda_sub45_fused_kernel 都有 `scale: f32` 标量参数（单元固定传 128**-0.5），kernel 层面有入口，不需要 host 预乘。本仓 chunk_kda_fwd 已把 scale 作为可选参数直通 kernel。
- **影响** fla 语义下 q 要乘 scale（默认 head_dim**-0.5 ≈ 0.0884）。无入口则只能 host 侧预乘，多一次 elementwise 全量遍历，与性能目标冲突。
- **建议** 优先在 kernel 内吸收 scale（已有 q 的读取点可顺带乘）。第一期若先用 host 预乘打通，必须在性能报告中标注这部分开销。

#### `no-tail-path` — L=64 固定且无 tail 路径，T 必须是 64 的整数倍

- **类别** abi · **适用于** 全部 · **阻塞** —
- **依据** kda_fwd domain.tails："T=C*64; partial chunks, K/V tails, fp16, and L=32 are rejected by this authored kernel unit." 其余单元同为 "No L/D tails"。
- **影响** 任意序列长度的推理与训练都需要 host 侧 padding，或拒绝。另：fp16 与 L=32 同样被拒绝。
- **建议** 门控显式报错并给出最近的合法 T。padding 方案要在性能报告中算进开销。

#### `block-dim-ceiling` — kda 的 block_dim 上限 4 是当前最大的性能天花板：扩展性到 4 仍是线性的

- **类别** performance · **适用于** KDA / GDN · **阻塞** `phase 4`
- **依据** kda_fwd / kda_bwd domain.block_dim: [1,2,3,4]；gdn domain.block_dim: [1,2]；delta_rule_fwd domain.block_dim: {min:1, max:32}。 | 2026-09-11 实测（Ascend950PR / CANN 9.2.0，每个 block_dim 一个进程，bf16，warmup 3 / iters 10，同步，仅前向）：kimi_linear_layer 4.955 / 2.516 / 1.724 / 1.311ms（bd=1/2/3/4，bd1→4 提速 3.78x）；long_context 19.622 / 9.971 / 6.794 / 5.138ms（3.82x）。单 kernel 级（sync-each 归因）：三个重 kernel 从 1.583/1.354/1.311ms 降到 0.407/0.361/0.329ms。四个 block_dim 的 relL2 完全相同，分区不改变数值结果。
- **影响** **扩展性一路线性到声明上限，说明这是声明限制而不是实现限制。** Ascend950PR 物理上有 28 cube / 56 vec，我们只用到 4 —— 按线性外推还有 ~7 倍空间。这让它成为比 kernel-nd2nz-suboptimal 更靠前的优化项：后者是常数因子，前者是可用核数。
- **建议** 已有答案：是声明限制。下一步是在 ascriptor 侧把 kda_fwd/kda_bwd 的 domain.block_dim 上限抬高并补 cases 覆盖（contract 的 core_ownership 说 gate 按 B*HV*C 切、scores/WY/inverse 按 cube 组切、融合尾部按 B*HV 头对切 —— kimi 形状下 B*HV=32、B*HV*C=512，工作量足够喂满 28 核）。kernel 源码归 ascriptor 仓所有（AGENTS.md §3：只读），改动要走那一侧。本仓的 SUPPORTED_BLOCK_DIM 与 contract 的声明由 tests/test_kda_gating.py 锁在一起，抬高后会同时提醒。

#### `qk-l2norm-not-in-kernel` — fla 的 KDA 在 kernel 内做 q/k 的 L2 归一化、门控变换与 beta sigmoid，ascriptor 的不做

- **类别** abi · **适用于** KDA · **阻塞** —
- **依据** fla 的 KimiDeltaAttention 调 chunk_kda 时传 use_qk_l2norm_in_kernel=True、use_gate_in_kernel=True、use_beta_sigmoid_in_kernel=True，即三步都在它的 kernel 里：g 的变换是 -exp(A_log) * softplus(g + dt_bias.view(HV,K))，beta 过 sigmoid，q/k 沿头维 L2 归一化。ascriptor 的 kda_fwd/kda_bwd 的 inlet 只收已经变换好的 q/k/g/beta，contract 里没有 A_log / dt_bias 这两个入口。

**A2-40 复核收窄（issue #54 / PR #85，2026-09-18）**：照抄 fla 的调用方式（传 `use_qk_l2norm_in_kernel=True` 等 flags）在本仓会直接 `TypeError`，不是静默——`chunk_kda`/`fused_recurrent_kda` 签名没有 `**kwargs`。真正的静默面是调用方为了消掉这个报错、把不认识的 kwargs 删掉后传入未归一化/未激活的原始值：这在数值上完全合法，门控闸也未必拦得住（原始值尺度小时跨度也小）。

**2026-09-19 收窄（A2-44，PR #99 → afafe03）**：公开 `chunk_kda` / `fused_recurrent_kda` 现在显式支持 fla 式 raw flags（`use_qk_l2norm_in_kernel` / `use_gate_in_kernel` / `use_beta_sigmoid_in_kernel`，`A_log` / `dt_bias`）：FP32 预处理、保留梯度，层已改走这条路，调用方不再需要自己做三步，缺参与不支持的组合显式报错。**但这三步仍在 kernel 外（PyTorch 算子），没有融合进 kernel——不要把这条标成融合完成。**`check_domain` 对 prepared 输入的检查是启发式，检不出小范数的 raw q/k，decode 不做逐步检查。
- **影响** **门控挡不住这一条** —— 没做归一化的 q/k 在数值上完全合法，算子会照算并给出一个静静地错的结果。这是本仓目前唯一"错了不报错"的语义缺口，因此列 P1。另外这三步的梯度也落在调用方这边，autograd 链要从层级算起。
- **建议** 当前处置：ascend_fla/layers/kda.py 显式做这三步并在 fp32 下做，ops/kda 的 docstring 与 __init__ 写明"调用方需已做"。直接调算子的人要自己负责。若要彻底消除风险，得在 ascriptor 侧给 kernel 加 A_log/dt_bias 入口与 l2norm —— 那是第四期的事，收益还包括省掉几趟 elementwise 的访存。

**A2-40 补充**：语义修正（加 fla 式原始输入 API + 定义域检查，`g≤0`/`beta∈(0,1)`/`‖q‖,‖k‖≤1+tol` 的启发式校验，不动 kernel）可以先做、不必等 kernel 融合——建议新任务 A2-44。真正的 kernel 融合仍是 A2-42（`kernel-batch-approval`）。

#### `state-layout-k-first` — 我们的 state 是 [B,HV,K,V]，fla 的 KDA layer 用 state_v_first=True

- **类别** abi · **适用于** KDA · **阻塞** `phase 3`
- **依据** ascriptor kda_fwd 的 initial_state / final_state 均为 (B,HV,128,128) 且 K 在前（kernel 内 next_state = state * exp2(g_last)[:,None] + kg^T @ v_new，按 K 维缩放行）。fla 的 KimiDeltaAttention 调 chunk_kda 与 fused_recurrent_kda 时都传 state_v_first=True。
- **影响** 第三期把本仓的 layer 注入 Kimi-Linear 时，与上游 cache 交接要转置，否则 decode 第一步就会用错的状态起算。K=V=128 让形状相同，**转置错了不会报形状错** —— 又是一个静默失败面。
- **建议** 第三期在注入层里做转置并加一个显式断言（比如用非对称测试值验证方向）。不要在算子里改布局 —— 算子的布局由 kernel 决定，改它等于改 kernel。

#### `gdn2-abi-not-gdn` — GDN-2 的 channel-wise erase/write/decay ABI 与现有 GDN kernel 不同

- **类别** abi · **适用于** GDN-2 · **阻塞** `gdn2_chunk_fwd_bwd`
- **依据** 目标模型 revision 86327354 的 gdn2_1.3B 配置是 H=HV=16、K=V=128，形状本身匹配 A5 定尺；但训练实现传给 recurrence 的张量是 q/k/v [B,T,H,128]、g [B,T,H,128] fp32、b [B,T,H,128]、w [B,T,H,128]，其中 b 是 key-channel erase gate，w 是独立的 value-channel write gate，g 也是 key-channel decay。现有 a5.gdn_fwd 的正式 ABI 只有 beta [B,H,C,64] 与 g [B,H,C,64] 两个 per-token 标量槽，没有 w，也没有 channel-wise b/g。
**recurrent 这一半已按独立 ABI 做完**（2026-09-14）：本仓新增 `a5.gdn2_fused_recurrent`，直接吃 token-major q/k/v/g/b/w 与非零 FP32 state，kernel 内做 q/k L2 norm 和 scale；Ascriptor unit runner 的 CCE/aclnn 五个 case 全过，o 最大 relative-L2=4.169e-6、state 最大 9.108e-8。真实 95B 的 BF16/FP32 整网 torch_npu↔CCE logits/cache 均在预算内，并由 CCE backend greedy 生成 32 个可读 token。现有 a5.gdn_* 仍未被误用或改名。
- **影响** 现有 a5.gdn_fwd/bwd 仍不能用于这个 checkpoint；强接会在形状层面丢掉 128 倍门控信息并无法表达独立 write gate，是确定性的语义错误。独立 recurrent kernel 已解除短 prompt/decode 的阻塞；剩余影响收窄为 T>16 的长 prefill 与训练反向，它们仍缺 gdn2_chunk_fwd_bwd。
- **建议** recurrent 已完成，不再改。下一步只做独立 gdn2 chunk fwd+bwd：L=64、K=V=128、直接吃 token-major；数值表示先解决 gdn2-chunk-gate-range 的 1461 跨度反例。g 变换与 b/w sigmoid 是否继续融合，等整层 profile 后决定；不能因名字相近复用 a5.gdn_*。
**2026-09-17**：forward-only 部分已排上看板 GD2-01（主机侧：量程设计 + kernel 单元）/ GD2-02（gated，A5 真机数 + 整网验证），D-PM-16 批准。backward 仍未排期。
**2026-09-17 补充（GD2-01 assignee 的 RISK，PM 已核实）**：fla 0.5.2 的 fla.ops.gdn2 下不只有 naive.py，还有完整的 Triton chunk 前向/反向（chunk.py/chunk_fwd.py/chunk_bwd.py/chunk_intra.py/wy_fast.py），门控数值稳定复用 fla.ops.kda.gate.kda_gate_chunk_cumsum。此前记录里没提这个，容易让人以为 GDN-2 chunk 完全没有可参考的实现——有，是设计参考（AGENTS.md §1 的语义权威），但 Triton 在 CCE 上不能假设直接成立，量程仍需在本仓执行栈上重测。

#### `gdn2-chunk-gate-range` — GDN-2 的真实 chunk 衰减跨度远超 KDA stable 已验证域，chunk 算法必须单独做量程设计

- **类别** numerics · **适用于** GDN-2 · **阻塞** `gdn2_chunk_fwd_bwd`
- **依据** 2026-09-14 用真实 95B checkpoint、B=1/T=4096、合法 vocab token id 的确定性随机输入（seed=20260914）逐层抓取 g=-exp(A_log)*softplus(f+dt_bias)。18 层合计观测：单 token 的 -g 最大 60.926；64-token 局部累计跨度最大 1461.214，各层 chunk-span 均值再平均为 69.041；完整 4096-token 累计跨度最大 83768.180。logits 仍全有限，说明逐 token recurrence 本身可用；证据在 tmp/gdn2-95b/gate-domains-t4096.json 与原始日志（git-ignored）。该输入不是自然语料分布，不能拿均值外推任务分布，但它是模型公开 token 域内的有效反例，足以否定“照搬 KDA 155 的量程即可”。
**2026-09-17 补充（GD2-01，PR #64）**：assignee 用真实 95B checkpoint 重跑随机 token 回放（截至本次汇报 18 层中 14 层完成），观测到的 64-token 局部累计跨度最大到**1520.91455**，超过 2026-09-14 记的 1461.214——按要求如实报了这个差异，没有为了对齐旧数而裁剪。五档子块（4/8/16/32/64）在已测的层上相对 L2 均 ≤1e-4，对两个 oracle（fla naive 与本仓 CPU 实现）都过。全部 18 层跑完后以最终数字为准，**闸的边界要按新的最大观测值定，不是 1461**。
- **影响** 现有 KDA stable chunk 的前向已验证跨度约 155、反向约 105，不能直接作为 GDN-2 的设计上限。任何在 64-token 区间端点或中点形成 exp(±span) / exp(±span/2) 的实现，在观测到的 1461 跨度下都可能出现 0、inf 或 inf×0；有限输出也不能替代梯度精度验证。decode recurrence 每步只使用 exp(g_t)，g_t≤0，向 0 下溢与数学极限一致，不受这个跨 token 配对量程问题影响。
- **建议** 冻结 q/k/v/g/b/w 的公共语义 ABI，但 chunk 实现暂缓。写 kernel 前先把 forward/backward 全链路所有 exp(g_i-g_j) 逐处列出，按 4/8/16/32/64 子块扫描实测量程，并同时对 fp32 recurrent oracle 验有限性与相对 L2；算法应采用不会构造大正指数的分段/归一化表示，而不是只把端点锚改成中点。测试至少覆盖本次 1461 反例、自然文本样本、chunk↔recurrent 与梯度。

#### `gdn2-decode-fragmentation` — GDN-2 host/elementwise fragmentation 基本收敛，BF16 decode 转为 GEMV 权重读取瓶颈

- **类别** performance · **适用于** GDN-2 · **阻塞** —
- **依据** 2026-09-15 初始真实95B/BF16/packed/B1T1 trace：torch_npu 6043.4us/1523 kernels；旧 CCE recurrent 5108.8us/1163 kernels，Cast 565.9us/309。去掉不变weight cast并新增 `a5.gdn2_fused_decode` 后，kernel内完成raw gate、q/k norm、FP32 recurrence/state、output RMSNorm+swish和最终BF16舍入；Cast降到74次/token、147.355us，launch 1163→640，反序整网paired speedup为1.410x~1.650x。
**动态双缓冲反例**：16/32/64-row DBuff虽无hazard/deadlock，pipesim却从整state 9444退化到14801/11917/10463 cycles；DB64真机4.804us且四AIV pipe和98.17%。Lowered IR显示auto_sync把动态 `slot[n]` 与 `slot[n-1]` 保守判为同root，插入read1→compute0和write0→compute1假依赖，因此只有存储双槽、没有流水。
**接受实现**：把两个64-row槽静态命名并展开，保持公式、VF算术、流量、block_dim和ABI不变。check 0 error/0 warning/404 surface ops，CCE emit 415；3/3 sim与3/3 pipesim数值通过且无hazard/deadlock，pipesim=9613 cycles，同核多pipe重叠3.68%→11.44%。两次独立真机profile各54样本，kernel mean=4.259/4.236us，合并mean/median=4.247/4.256us；夹心整state即时对照=4.733/4.716us，候选快1.114x/1.108x。候选Vector/Scalar/MTE2/MTE3合并平均34.73/17.60/37.52/21.70%，四pipe和111.55%（对照86.84%），证明真实重叠。aclnn与真实95B整网数值复验均过原预算；两轮profile与整网任务所有kernel/PID只在物理7、物理0活跃0次。另一次canonical aclnn fresh build在actual launcher前捕捉到物理0的python/0MB瞬时条目，时间对齐CANN opc编译；test_aclnnop只在7，0未留驻，后续严格单卡窗口不再fresh build。证据：tmp/gdn2-cast-fusion/{tile64-profile-v1,static2-v4-profile-v1,static2-control-profile-v1,static2-v4-profile-v2,static2-v4-fullmodel-v1,static2-v4-canonical-aclnn-v1}/（git-ignored）。
**NPU Graph 实测**：torch_npu 2.10 的 `NPUGraph` 能同时捕获整网内置算子与本仓 ctypes→aclnn custom op。固定prompt-cache对照为6735.3→3834.9us/token（148.47→260.76 token/s，1.756x）；可递推版本在图尾把18层新recurrent/conv cache拷回固定输入地址，v3~v5同轮eager为5911.9~6323.7us/token，graph稳定在3908.2~3911.9us（255.63~255.87 token/s，1.512x~1.617x）；逐token同步的graph为3915.6~3926.0us（254.71~255.39 token/s，对eager加速1.554x~1.736x）。四个不同token连续replay的logits、两类cache均与eager逐位相同。eager提交约5.67~6.78ms/token，graph提交仅2.8~7.4us/token，说明host发射已不再配速；stateful graph每token额外回写19,759,104 bytes，实测只比fixed graph多约75us。清空eager allocator历史后，捕获常驻allocated/reserved增量约39.6/130.0MB，一次捕获约13.6ms。证据：`benchmarks/bench_gdn2_graph_capture.py` 与 tmp/gdn2-graph-capture/{fixed-v1,stateful-v3,stateful-v4,stateful-v5}.json（git-ignored）。
**真实生成接线**：`GDN2NPUGraphDecodeRunner` 与 `generate_tokens(decode_backend="npu-graph")` 已接入。真实95B、64-token greedy在graph-first/eager-second与eager-first/graph-second两组独立进程中，四次token ids逐项相同。含logits D2H、CPU argmax、token H2D的graph decode为244.21/244.79 token/s，对应eager为173.02/148.84，两个顺序加速1.411x/1.645x；含冷prefill和每请求setup的64-token总吞吐graph为99.11/104.23、eager为96.73/91.60，两个顺序加速1.025x/1.138x。四个接受窗口物理0活跃均为0，物理7各只有一个本任务PID。证据：tmp/gdn2-graph-generation/{graph-greedy64-v2,eager-greedy64-v1,eager-greedy64-v2,graph-greedy64-v3}.json（git-ignored）；一轮与外部作业竞争的smoke已隔离为rejected，不作性能证据。
- **影响** host gap与大部分elementwise fragmentation已经消除：低内存默认纯模型Graph约2.659ms/376.11 token/s；保守mixed显式opt-in为2.589ms/386.32 token/s。其最新device时间90.175%是native GEMV+mixed W12权重流，mixed AIC MTE2平均91.88%。gamma-fold v1的56.24us/token投影已因递推精度作废；保守v2以1,029,832,704 bytes派生权重换取clean Graph约68.20us/token，必须连同内存决策。若要下一次大步收益，需要weight-only低精度/量化及独立任务质量实验。
- **建议** 精度、同卡stateful Graph A/B和profile均已完成。现在先决定：①保持vendor低内存默认；②用1.030GB派生布局换2.6%吞吐；③设计prompt/decode共用布局去掉重复权重。若继续追求大步收益，再决策weight-only低精度及真实生成/任务质量预算。明确不选：单独W1/W2 kernel、N=64小tile、减小block_dim、增加冗余工作，或已证伪的细粒度DMA双缓冲。W3融合不减少约85.8MB/layer的BF16 W12+W3权重字节，只是ceiling候选。

#### `kda-bwd-inverse-mm-mutex-over-budget` — 上游 kda_bwd 的 inverse_mm_kernel 需要 34 个 cube 互斥锁 ID，超过 A5 局部预算 32，pin 版 kernels 下带 cache 的公开前向无法编译

- **类别** runtime · **适用于** KDA · **阻塞** —
- **依据** pin 版 `a5.kda_bwd/kernels/inverse_mm.py`（kernels b3b3f9c16df7）在 A5 上降级失败：`PassError: cube needs 34 mutex IDs (maximum 32) at kda_bwd/kernels/inverse_mm.py:65:15`。D-PM-24 记录 PM 已独立复现；2026-09-19 PM 在 pin 版 library（90cfcdc720bb）+ pin 版 kernels 上再次复现，并落成 `tests/test_kda_aqk_dispatch.py::test_cached_forward_dependency_fits_mutex_budget_and_keeps_upstream_control`：原版 `lower_kernel` 仍报 34>32，本仓派生版降级后 cube 32 / vec 15。根因（A5K-02 定位，PM 对着上游原文件逐行 diff 核过）：`l0c_dvh`、`l0c_dvbeta` 两块 L0C 是 `DBuff`（双槽），改成单槽 `Tensor` 即由 34 降到 32；互斥锁、事件信用、循环、lookahead、算式一行没动。
**2026-09-21 补充对照（FMT-01，joshjms，自述，PM 未独立复现）**：同一台 A5（CANN 9.3.0 / torch 2.7.1）上固定 kernels 解出件（b3b3f9c）不变、只换 library——90cfcdc 在 block_dim = 1 与 2 都报同一个 34 > 32（与核组数无关），627f55f（D-PM-13 时的 pin）能编能跑；`ascriptor/passes/local_mutex.py` 在 627f55f..90cfcdc 之间才出现，是新 pass 拒绝了此前能编的上游 kernel。PM 的处置：本条早已登记、派生单元已绕开；`impl="upstream"` 的带 cache 前向 / 反向在 pin 版下编不过是已知且被接受的现状，不需要另立任务，也不因此退役 `impl="upstream"`（公共 API 取舍，无必要）。
- **影响** A5K-02 的 contract 记录：带 cache 的公开前向（`chunk_kda_fwd_with_caches`，训练前向）会预编译 backward 的全部 9 个 kernel，pin 版上游 `inverse_mm` 在这一步降级失败，所以训练路径编不出来。**现状**：本仓派生单元已绕开（A5K-02，合并提交 2589942），公开 stable 路径不再触发；上游 kernel 本身仍是坏的，任何直接用上游 `inverse_mm` 的路径仍会撞上。
- **建议** **已做**：A5K-02（#83）在派生单元 `kernels/projects/a5/kda_bwd_stable/kernels/inverse_mm.py` 落地受限版（入口 `inverse_mm_bounded_kernel`），`ops/kda/chunk_bwd.py` 的 stable 覆盖表选它。原生验证：完整 inverse leaf B1/HV32/C64/bd4 + 3 个小 case、13 个完整 backward 回归（含 Kimi T4096、跨度 46/94/104/精确 105）都在未改的预算内，leaf 相对 L2 ≈ .00164–.00168；缩小 sim/pipesim 8/8 无 hazard/死锁。**要做**：上游（ascriptor）应改 `kda_bwd/kernels/inverse_mm.py`，本仓不改（AGENTS.md §3），进上游 kernel 批次；届时可删掉本仓派生，并把 `test_aqk_dispatch` 里『原版 34>32』的对照测试改成『原版通过』。

#### `a2-splitk-fp32-cube` — A2 系 FP32 cube 累加（M10-081）：split-K 在 pin 里已有修复，FP32 手写累加链仍未 settle

- **类别** numerics · **适用于** KDA / GDN · **阻塞** `A2 波次：A2-03 / A2-09 / A2-11，以及 A2 上的任何算子结论`
- **依据** **来源：A2-01（#29，PR #107，DONE 于 2026-09-19T16:13Z）。原始回执（8 份 JSON、15 份日志）随 PR，合入后在 `benchmarks/a2/evidence/`；PM 从原始回执逐项复算了真机数字、抽查了命中表的源码行、并在 CPU 上独立复跑了 reference / sim 与设备日志的输出哈希对得上；A2 真机部分 PM 无 A2 真机、未复现，按证据审。结论只对 910B3 / CANN 9.0.0 成立，A2-11 之前只作观测，不构成算子结论。**
- 库里的记录：ascriptor `docs/defects/M10-081-a2-family-fp32-mmad-settle.md`（A2 系两次短 FP32 MMAD 写同一块 L0C，硬件不互锁，第二次 `is_init=False` 读到未落定的累加器）。
- pin 版修复（PM 读源码核实）：`ascriptor/passes/desugar.py:411`——`family == "a2"` 且 A/B 均为 `f32` 时，split-K 展开在两次 MMAD 之间插 `PIPE_M` barrier；`desugar.py` 的 sha256 前缀 900610ea92dc，与申领人所报一致。手写 MMAD 链只有 lint trap，不自动修。
- 真机（申领人自述；a2 / 910B3 / CANN 9.0.0 / block_dim=1 / 每 case 5 次，逐位对 CPU float64 参考）：FP32 split-K 的 M16 原 case、`splitk_f32_m16_n64_k32_s16`、KDA intra 形状 `splitk_f32_m64_n64_k128_s64` 均 5/5 逐位；手写 FP32 链 `chain_f32_m16_n16_k16_t2/t3_nobar`（inverse 形状，无 barrier）目前 5/5 逐位，**但时序没踩到不等于安全**。
- 命中表初版（25 个在用 kernel，自述）：FP32 split-K 1 个（KDA scores/intra，pin 已自动 settle）；**FP32 手写累加链未 settle 3 个：KDA triangular inverse、GDN triangular inverse、GDN bwd finalize，正是 M16 失效形状**；BF16 累加链 2 个见 `a2-splitk-bf16-fp16-unsettled`；其余 19 个未命中。
- 模型看不出：sim / pipesim 在 a2、a5 两个 profile 下全部逐位、0 hazard——模型不表达这个同管线 L0C RAW，复现只能靠真机。
- 手写 FP32 链（M16/32/64，2~3 项；M16/N64 的 8 项孪生）共 45 个独立 case 在卡 A/B 上全部 5/5 逐位：发射节奏没有背靠背，不是安全证据（BF16 的 split-K 孪生就错，同样写法的手写 BF16 链不错）。

**A2-11 更新（2026-09-23，PR #130 → 652dfc0，用户已裁定接受结论）**：手写 FP32 累加链在真机上加了时序扰动（--jitter：两目标 MMAD 间插独立 scratch-L0C MMAD）跑了 2000 次（t2+t8 各 1000），仍**未复现出错**——比 A2-01 最初的 45 次样本量大得多，但结论不变：**「未复现」不是「安全」**，`barrier(Pipe.M)` 绕行保守保留。绕行本身在真机上 2000 次逐位稳定（多 bd、多卡）。已合入的 a2 单元现在有静态 AST 守卫（`tests/test_a2_accumulate_barriers.py`：每个 `is_init=False` matmul 前必须紧邻 `barrier(Pipe.M)`），main 上 0 违规，未来改动漏加 barrier 会被测试拦住（负对照：PM 删掉 `kda_fwd_stable/kernels/recurrent.py:187` 的 barrier，测试按预期变红，还原后复绿）。A2-11 是本条目在「A2 结论算数」条件下的总闸，已完成，见 `docs/research/a2_splitk_gate.md`。
- **影响** A2 上凡是写同一块 L0C 的 FP32 累加链（KDA / GDN 的三角求逆、GDN 反向 finalize）在 M=16 形状下可能读到未落定的累加器，输出有限、量级正常但内容错（真机尚未复现出错，时序没踩到不等于安全）。A5 现有路径不受影响（这是 A2 系的硬件行为）。A2 派生单元（A2-03）要逐个核对，A2-11 才是真机复现与绕行验证。**AGENTS.md §2 原先写的「没解决的 split-K FP32 cube 缺陷」太窄**：FP32 split-K 在 pin 里已有修复，没解决的是 FP32 手写链与 BF16/FP16 split-K，已于 2026-09-19 改写（用户同意）。
- **建议** 1. 转给 ascriptor 所有者（本仓不改 ascriptor）：把 M10-081 的 settle 扩到手写累加链，或提供自动检测。2. 本仓侧：a2 派生单元对所有写同一块 L0C 的累加链显式 `barrier(Pipe.M)`，不分 dtype 与写法（A2-03 / A2-09 起；属 kernel 批次 A2-K1 的「split-K 绕行」）；ascriptor 修好前，对不能证明安全的形状入口按 `AGENTS.md` §7 显式报错。3. A2-11 在真机上做复现与绕行验证（含手写链的时序压力）。

#### `a2-splitk-bf16-fp16-unsettled` — A2 系 BF16/FP16 split-K matmul 在 910B3 上输出有限但全错（M16），M32 触发 AI Core 异常：M10-081 的 settle 规则按 dtype 排除了它们（静默错误类）

- **类别** numerics · **适用于** KDA / GDN · **阻塞** `A2 波次：A2-03 / A2-09 / A2-11，以及 A2 上的任何算子结论`
- **依据** **来源：A2-01（#29，PR #107，DONE 于 2026-09-19T16:13Z）。原始回执（8 份 JSON、15 份日志）随 PR，合入后在 `benchmarks/a2/evidence/`；PM 从原始回执逐项复算了真机数字、抽查了命中表的源码行、并在 CPU 上独立复跑了 reference / sim 与设备日志的输出哈希对得上；A2 真机部分 PM 无 A2 真机、未复现，按证据审。结论只对 910B3 / CANN 9.0.0 成立，A2-11 之前只作观测，不构成算子结论。**
- 静态（PM 核实）：`desugar.py:411` 的条件是 `dta.name == dtb.name == "f32"`，其上注释写明其他 dtype 不在这次「board-proven workaround」范围内——BF16/FP16 split-K 不插 barrier。
- 真机（自述；a2 / 910B3 / CANN 9.0.0 / block_dim=1，逐位对 CPU float64 参考，输入为有界二进分数、FP32 下精确）：M16：K32/48/64/128 的 BF16/FP16 split-K 全部错，输出有限、无报错，1022~1024/1024 个元素不对（例：`splitk_bf16_m16_n64_k128_s16` max_abs 15.67 / rel_l2 1.35，`splitk_f16_m16_n64_k128_s16` 15.88 / 1.17），5 次输出哈希相同，三张卡上复现；M32：每次触发 AI Core 异常（`aclrtSynchronizeStream` 507015，retCode 0x26），两张卡复现；M64：K32~128 全部逐位。
- 对照：只在该次运行的生成 CCE 里、MMAD 分支后补一行 `PipeBarrier<PIPE_M>()`（位置与 FP32 规则生成的相同），M16 5/5、M32 2/2 的 BF16/FP16 全部逐位；只插注释的 sham 版照样错。
- 手写孪生（8 次 K16 matmul 写同一 L0C，不加 barrier）真机 5/5 逐位：split-K 循环里 MMAD **背靠背发出**才踩到。
- 功能模拟与 pipesim 在 a2/a5 两个 profile 下全部逐位、0 hazard，模型看不出。
- 运行时日志（`m32_bf16_k32_cann_runtime_errors.log`）明写 `L0C read/write conflict` 与 `retCode=0x26 [aicore exception]`，与 M10-081 的机理一致；错误输出不是任何 K 分片或交叉分片乘积的组合（最小二乘残差约 0.97~0.99），输出范数约为参考的 5 倍，5 次输出哈希相同（确定性错误）。

**A2-11 更新（2026-09-23，PR #130 → 652dfc0，用户已裁定接受结论）**：真机大样本复现确认了危险性质——BF16 split-K @M16：**500/500 次全错，500 个输出哈希两两不同**（完美非确定 = 硬件时序竞争，非确定性 miscompile，不是可复现的固定错误；最小二乘分解显示「隔片丢弃」的陈旧累加器系数特征，15/16 行系数精确落在 [0,1,0,1,...] 附近）；BF16/FP16 split-K @M32：**200/200 次触发 AI Core 异常**（`aclrtSynchronizeStream failed: 507015`，设备故障，不是数值错误）；BF16 split-K @M64：199/200 逐位正确（唯一 1 次是瞬时设备故障，非错误结果）。**AGENTS.md §7 已加入永久禁止条款**：A2 系 kernel 里 BF16/FP16 `splitk` 在 `M<64` 时必须显式报错，不接受任何绕过；`tests/test_a2_accumulate_barriers.py` 的 `test_no_splitk_below_m64` 是这条的静态守卫。详见 `docs/research/a2_splitk_gate.md`。
- **影响** **静默错误类**（输出有限、无报错）。本仓现有 25 个在用 kernel 都不用 BF16/FP16 split-K，A5 现有路径不受影响；KDA recurrent 与 GDN bwd wu 有 BF16 累加链（是否满足踩踏条件未定，手写孪生没踩到）；A2 派生单元（A2-03 / A2-09）若因 L0 容量改用 BF16/FP16 `splitk` 就会中招。D-PM-35 的 BF16 优先叠加 A2 优先，BF16 的 KDA 在 910B 上要先过这一关。
- **建议** 1. 转给 ascriptor 所有者：M10-081 的 settle 规则按 dtype 收窄在 910B3 上不成立，应扩到 BF16/FP16，并修 M32 的异常。2. 本仓侧（A2-03 / A2-09 起）：a2 路径不用 M<64 的 BF16/FP16 `splitk`，入口按 `AGENTS.md` §7 显式报错；a2 派生单元对所有 L0C 累加链显式 barrier，不分 dtype。3. A2-11 真机复现（含 M32 异常）。

#### `a2-cast-blkstride-sim-blind` — 【静默错误·自述】A2（c220）上对『32 字节行包』的列整块 cast 被降解成 srcBlk=0 的 vconv，真机整块拿到第 0 行；功能模拟器与 pipesim 都不建模 block stride，全绿

- **类别** correctness · **适用于** KDA · **阻塞** `A2 波次：A2-09 及之后所有含 BF16 → FP32 整块 cast 的 A2 单元`
- **依据** **来源：A2-09 的 RISK sim-device-divergence（#112，2026-09-20T04:28Z，申领人自述；PM 无 A2 真机、未复现；结论只对 910B3 / CANN 9.0.0 / ascriptor pin 90cfcdc 成立，A2-11 之前只作观测，不构成算子结论；最小复现与原始输出要随 A2-09 的 DONE 提交，届时按证据复算并更新本条。**
- 构造（自述）：BF16 的 beta 是 `[B*T, HV]`，一个头的 64 个值在 GM 里按 HV 跨步；`gm_to_ub_pad(beta_b_ub[0:64, 0:1], beta[row0:row0+64, hv:hv+1], 64, 1, HV-1, 0)` 之后 `cast(beta_f_ub[0:64, 0:16], beta_b_ub[0:64, 0:16], count=64*16)`。生成的 c220 代码 `vconv_bf162f32(dst, src, repeat=16, dstBlk=1, srcBlk=0, dstRep=8, srcRep=4)`——`srcBlk=0` 使第 0 行那个 block 被复制到每一行，整个 chunk 的每个 token 都拿到 beta[0]。改成逐行 `count=1` 的 cast 后真机与参考逐位一致。
- 静态（PM 核实的只有这一点）：pin 版 CCE 发射器的向量指令确实由 IR 属性 `src_blk_stride` / `dst_blk_stride` 决定 block stride（`ascriptor/backends/cce/emit_vec.py:211-212` 等），所以 srcBlk 出现在生成代码里这件事成立；具体哪种构造把它降成 0，PM 没复现。
- 模型看不出（自述）：功能模拟器与 pipesim 都不建模 block stride，两者都判通过。
- 影响面（自述）：A2-09 单元里 dv、D_tri、dAkk、M_base、M_beta、s_base、t_beta 以及下游六项梯度全错而 sim 全绿；`finalize_pair.s_base` 的 max_abs 一度到 6.2e+08。修完后同一台 910B3 上 33 个逐 kernel checkpoint 与 CPU 参考逐位一致，余下只是 FP32 舍入（最大相对 L2 1.2e-02，在 dg 上，预算 0.25）。
- **影响** **静默错误类**（输出有限、无报错、sim 与 pipesim 全绿）。只在 A2（c220）上；A5 不受影响这一点没有验证，按 §6『结论不跨 SoC 继承』既不外推也不否认。A2-03 已合入的前向与 decode 单元里 beta 是 FP32，申领人称没踩到（自述）；凡是后续 A2 单元里对『跨步 / 行包』列做整块 cast 的，同样暴露。A2 上的算子结论在 A2-11 之前本来就不算数，这条是给 A2 单元的写法加一条硬约束。
- **建议** 1. 本仓侧（A2-09 起）：a2 单元里对行包列的 cast 逐行 `count=1`（或改成不依赖 srcBlk 的写法），并在单元 README 登记每一处这类构造；A2-09 的 DONE 带上最小复现（构造、生成的 c220 行、错误版与逐行版的真机原始输出，文本）。2. 转给 ascriptor 所有者：cast 降解在行包视图上产出 `srcBlk=0` 应报错或正确展开；功能模拟器 / pipesim 应建模 block stride，或对 `srcBlk=0` 且 `repeat>1` 的向量指令给出 lint。3. A2-10 / A2-11 做真机核对时把『sim 判过而真机错』列成必查类别，A2 上 sim 通过不再作为任何单元的通过依据。

#### `kda-bwd-scan-dh0-nondeterministic` — 【静默错误候选·自述·环境相关】KDA 反向的上游 scan_fused 在 CANN 9.1.0-beta.1 的一张卡上 dh0 输出不确定（12 次直接重放 11 次不同），旧公共路径 h0 梯度 4/12 超 0.05 预算；换一张卡 12/12 逐位相同

- **类别** correctness · **适用于** KDA · **阻塞** `FMT-02 的 dh0 / h0 梯度端到端逐位验收（按 D-PM-44 改读）`
- **依据** **来源：FMT-02 的 RISK `kernel-change-needed`（#109，2026-09-20T05:36Z）与 STATUS（05:47Z），申领人自述；PM 无该环境、未复现；结论只对其实测环境成立——只读既有 CANN 9.1.0-beta.1（compiler timestamp 20260509_173000235）、Python 3.12.14、Torch 2.12.0+cu130、torch_npu 2.12.0，library 90cf / kernels b3b3——不跨环境推断（此前 A5 证据来自 CANN 9.2.0）。**2026-09-20T06:05Z PM 更新：PM 已从 PR #116 头 964cc8e 里提交的原始 JSON（`evidence/diagnosis/old-public-repeats.json`、`scan-probe-comparison.json`）自己复算，下面的统计与自述一致（旧路径 12 次 h0 五个哈希 7/2/1/1/1、其余 7 项各 1 个哈希、h0 相对 L2 0.002337 ×7 / 0.008439 / 0.153270 / 0.357131 / 0.358643 ×2、超 0.05 的 4 次、超 3F=0.004971 的 5 次；候选 12 次同一哈希；第一设备 12 次直接重放 11 次 dh0 不同、差异元素 1917–8184、dAqk / dh / dv 全同）；设备 / 环境标识（哪张卡、CANN 9.1.0-beta.1）与第二设备的 12/12 仍是自述，PM 没有该环境、未复现。**
- 直接重放（自述）：固定完全相同的输入与 bd4 / B2 / HV4 / C3，直接调用已编译的原始（上游只读）scan vendor 重复 12 次（每次独立输出、NaN 预填、设备同步）：**第一张卡上后 11 次的 dh0 字节都偏离第一次**（差异元素数 1917–8184），dAqk / dh / dv 的 12 次结果全部逐位相同；**另一张健康空闲的物理设备上（同宿主、同 CANN、同 vendor、同生成输入）12 次直接重放全部逐位相同**。
- 旧公共路径（自述，同一 B2/T192/H2/HV4 输入、完整旧 autograd 路径重复 12 次、不加内部同步）：FP32 h0 梯度出现 5 个不同哈希（频次 2/1/7/1/1），其余 7 项输出 / 梯度各只有 1 个哈希；h0 对 CPU FP32 的相对 L2 是 0.002337 ×7、0.008439、0.153270、0.357131、0.358643 ×2，**4 次超过原 0.05 预算**。BF16 dh0 在任何 widen 之前就已不同；新 kernel widen 与旧 Torch widen 各自逐位等于 CPU FP32 转换；候选 12 次同哈希（等于旧路径的多数哈希），但同步 / 捕获边界会改变复现概率，不能当修复。
- 定位（自述，机制未定位）：首个差异隔离到 `kda_bwd/kernels/scan_fused.py:358-362`（最终 state 更新 / 发布区域）的 dh0 输出。

**A5K-03 更新（2026-09-23，PR #133 → 6619475，squash 合入，只定位不修，PM 独立复核关键数字）**：机制已定位——`seed_ub` 缓冲在每个 BHV 边界重置回槽 0，但 `CvMutex(3, depth=2)` 的信用协议跨 BHV 连续流转；奇数 chunk 数时，下一 BHV 的首次 seed 发布（277 行）可以用上一 BHV 归还的信用覆盖槽 0，而上一 BHV 的最终 VF 读取（358 行，实际 load 在 160-161 行）还没读完——WAR 冒险，不是缺 barrier（本地 32-ID 自动同步命名空间本来就建立不了这条跨侧边）。危险对数按 block_dim（B2/HV4/C3）：14/12/10/8。**方法论**：执行前记录可证伪预测（H1/H2），单变量干预（只把 4 处 seed 索引改成连续相位，缓冲/信用/算术不变）+ 无关注释负对照（AST/CCE 验证与 baseline 相同）：连续槽号模型在原模型报 14/12/10/8/8/0/2 处 WAR 的地方全部归零。反事实替换（不拟合系数）解释了 12 次历史重放里 51990/52001 个异常 BF16 元素（PM 从原始 `counterfactual.json` 的 12 条 trial 记录独立加总复现，逐位相同）；受控时序干预（三张卡、两套 CANN）在大 SPINS 值下稳定复现 50/50 次 dh0 错误，连续槽号版本 0/50 且与未扰动结果逐位一致，但健康卡上的自然复现（9,900 次无扰动重放）仍未出现——如实区分「受控因果证据」与「自然复现」，不混为一谈，不声称已修复。详见 `docs/research/kda_scan_dh0_nondeterminism.md` 与 `kernels/projects/a5/kda_scan_diag/evidence/`。
- **影响** **静默错误候选**（无报错、输出有限，只是 dh0 有时是错的，最差相对 L2 0.36）。目前只知道：KDA 反向的 dh0（初始 state 的梯度）；只在 CANN 9.1.0-beta.1 环境里的一张卡上复现，另一张卡与此前 CANN 9.2.0 环境的历史证据里没有观察到——**这不等于没有问题**：竞态类缺陷的复现概率依赖时序（与 `c1-multihead-o-corrupt` 同类，也藏过）。也可能是那张卡 / 那个 CANN beta 环境本身的问题（申领人自己管理的机器，D-PM-28）——机制未定位前两种都不能排除。
- **建议** 1. **先定位**（§6.5，不许照症状改）：FMT-02 交最小复现包（固定输入、直接重放脚本、两张卡的哈希分布、原始张量哈希）后，由后续任务定位 `scan_fused.py:358-362` 最终 state 发布区域缺哪一次同步，或证明是环境问题。2. 在 CANN 9.2.0 环境或别的卡上复核同一组直接重放，区分环境 / 卡与 kernel 竞态（机器归各申领人自己管，D-PM-28）。3. 若确认是上游 scan_fused 的缺陷：修在本仓派生 scan 单元（上游不改，AGENTS §3），进 kernel 批次，**要用户批准**；批准前不动 scan、不用 host 同步掩盖。4. 结论之前，依赖 dh0 的调用方应知道它在该环境下可能不确定。

**2026-09-20 用户指示「scan fused 需要定位」（D-PM-46）：已立项 A5K-03（只定位、不修，实验性派生单元、不接公开调度；机制 + 可证伪预测 + 干预实验；结论可能是 kernel 竞态也可能是环境 / 卡问题）。**修复仍要在定位出精确机制之后由用户另批 kernel 批次（本条仍是 `requires_kernel_change`，仍在队列里）。

**2026-09-23 A5K-03 完成定位**：机制已查明（见 evidence 字段），不是环境/卡问题（跨卡、跨 CANN 的受控干预复现同一机制）。修法方向：维持跨 BHV 的 seed 槽相位连续，或设计明确归还全部在途槽的边界协议；不能简单把信用降到 1（预取在消费之前，可能死锁，需另行分析）。**修复仍需用户批准 kernel 批次**，本条继续在 `kernel_fix_queue`。

#### `kda-prep-backward-training-step-slowdown` — KDA raw flags 训练路径（BF-08 合入后）的完整前向 + 反向训练步比旧 host 图慢 11.3 ×：一个 gate backward 第一阶段 kernel 的补偿算术占 375 ms（T4096）

- **类别** performance · **适用于** KDA · **阻塞** —
- **依据** **来源：BF-08（#117，PR #122 → 6c4aaca）；PM 从 perf-v2-bd4 的 36 条原始样本复算；申领人的 A5 真机（CANN 9.1.0-beta.1），结论只对该环境成立。**
- 同卡 bd4、B1 / H = HV32、BF16 raw / FP32 参数、三轮 baseline / candidate / baseline：T1024 旧 host 图 9.5750 → 106.7511 ms（11.149 ×）、Torch NPU 训练基线 39.9378 → 106.8193（2.675 ×）；T4096 37.5994 → 425.7246（11.323 ×）、140.5361 → 425.7781（3.030 ×）。这是**完整前向 + 反向训练步**（含 raw 前处理），不是单独前处理。
- 逐 kernel（Level1 profiler，含扰动、不可与 clean 墙钟相加）：`kda_prep_backward_gate_bf16_f32_f32_kernel` 375.466 ms（独立事件窗口 370.907 ms）；`kda_prep_backward_norm_bf16_kernel` 两次共 19.566 ms；其余为既有 kernel。申领人的推断（源码 + 上述实测，非逐操作微基准）：补偿 exp / log1p / sigmoid 与乘积残差的指令数、每 64 lane 重复计算 / UB 物化与固定五层部分归约；第二阶段全局归约很小。
- 为什么这么做：冻结的逐元素与 L2 精度预算（3 × 校准地板）在完整 Kimi 人口上连旧 host 图自己都过不了（k 的最差元素上候选与旧 host 逐位相同、相对误差 2.166，条件数 6.0e7），首次未补偿的候选因此失败，改用补偿算术后才通过——精度换来的性能代价。
- **影响** `ascend_fla/layers/kda.py` 的 KDA 层在 chunk 与 decode 上都用 use_qk_l2norm_in_kernel / use_gate_in_kernel / use_beta_sigmoid_in_kernel 三个 raw flag，所以 **KDA 层的训练路径直接受影响**（推理 / decode 是 BF-07 的前向，不受影响；默认无 flag 路径不受影响）。规格不设速度门槛，用户在 D-PM-57 知情后选择先合入、另立 P0 性能任务。
- **建议** BF-09（P0，open）：在冻结预算字节不变、跨 bd 逐位相同、固定归约次序的前提下优化 gate 第一阶段与 norm backward——减少补偿超越函数 / 乘积的指令数（证明同样的误差界）、减少重复的 64 lane 临时 load / store、改善流水与 tile 复用、减少冗余 launch；目标线（PM 建议，不是门槛）T4096 完整训练步 ≤ 旧 host 图的 1.25 ×。若守住约束仍做不到，如实报瓶颈与下界，由用户决定是否放宽逐元素判据（同一人口对旧 host 成对比较，或按项幅值归一）。

#### `a2-kda-bwd-gcumsum-bf16-precision` — 【P1·真实缓存资格化中发现】A2 kda_bwd_stable 的 gc2（累积门控）四个消费者只吃 BF16，真实默认初始化跨度下 dg 可超 0.05 预算

- **类别** numerics · **适用于** KDA · **阻塞** `A2-13 的宽域精度曲线资格化（backward 闸的上下边界目前互相打架：预算安全域上限低于默认初始化真实会用到的跨度下界）`, `A2-15（A2 KDA 层：前反向 + prefill→decode + torch_npu 基线，依赖 A2-13）`
- **依据** 来源：A2-13（issue #40）用真实 kda_fwd_stable 产出的九个检查点（不是独立 fixture）驱动 kda_bwd_stable 做端到端资格化时发现，RISK kernel-change-needed（评论 5876471263，2026-09-28T18:56Z）与后续 STATUS（PR #137 评论 5876604634，2026-09-28T19:04Z）；申领人自述，PM 未复现（无 A2 真机）。
两个 bd 完整粗扫 36 点：22 通过、10 有限梯度超预算、4 前向非有限（span192，按 finite-input 契约正确拒绝，不是 bug）。定位实验：uniform H=HV=32/T128/seed0，span 24/28/32/64/96/128，两 bd 逐位一致，dg 依次 0.036355/0.053919/0.063174/0.122838/0.137940/0.244574（预算 0.05，前两点在预算内，从 span32 起全部超）。CPU stage replay 消费完全相同的真机 BF16 缓存复现同量级误差（64/96/128 为 0.123686/0.139080/0.245634）；**只把 g_cumsum 一项换成真实前向的 FP32 值（gc*1/ln2），其余 8 项缓存不动**，同一 CPU 对照的六梯度在全部六个 span 点都回到0.05 预算内（依次 0.014029/0.014947/0.016146/0.022070/0.026545/0.030554）——干净地把责任定位到 g_cumsum 的 BF16 narrowing，不是其它缓存或反向算式本身。
定位到的消费者：`kernels/projects/a2/kda_bwd_stable/unit.py:112/121/135/149/172` 的 `gc2` 只有四个消费者——`scan_fused`、`inverse_epilogue`、`finalize_pre`、`finalize_post`；其余五个反向 kernel 不直接吃 gc。这四处的 GM/UB 累积门控输入定义为 BF16，是精度损失的位置。
该定位对照本身**不是**"改后 kernel 真机通过"的声明——现有 kernel 的 GM 输入与 `unit.validate_inputs` 强制 g_cumsum 为 BF16，测试侧不能绕过 ABI 直接喂 FP32 值给现有kernel，所以这不是能在现有写集内修的东西。另附带发现（如实记录，不隐藏）：normalized-input 的 span24/28/32 在 `finalize_pair` 中间检查上也超出既有 1e-5 界（最坏 t_beta 9.2949e-5）；FP64 诊断显示该残差远小于 FP32（0/8.97e-7/2.84e-9），提示是 FP32 累加/前级舍入敏感，但FP64 不替代 golden，1e-5 失败原样保留，不构成"已修复"的证据。
- **影响** A2 上 `kda_bwd_stable`（KDA 反向的正式 backward kernel）在真实默认初始化会产生的门控跨度（fla 默认初始化 8 seed 实测下界 max≈106.4，本条在 span32 起就已超预算）下，`dg` 梯度相对 CPU FP32 完整自动微分的误差可能静默超出 0.05 验收预算而不报任何错误（有限值，量级正常，只是数值偏了）——这正是 §7 要求装闸防的那类问题。目前 A2-13 尚未给 A2 的 `max_gate_span["stable"]["backward"]` 填任何数字、`qualified` 仍是 `False`，公开 `ascend_fla.ops.kda` 未接 A2，所以**当前没有产品路径会静默吃到这个错误**——风险仅存在于"如果不管这个发现直接填一个只看有限性、不看这条精度曲线的闸值"这种假设路径，实际未发生。
- **建议** 申领人提议（评论 5876471263）：保留现有 BF16 `kda_bwd_stable` 单元及其历史资格不动，新增一个 A2 派生的 FP32-g_cumsum 反向单元——只改四个消费者（`scan_fused`、`inverse_epilogue`、`finalize_pre`、`finalize_post`）的 GM/UB 类型与相应载入/转换，新 unit.py/reference.py/contract.json，其余 8 项缓存与返回梯度仍 BF16，五个非 gc 消费者复用冻结实现。gc 缓存会从 32MiB 增到 64MiB（B1/T4096/HV32，仅张量字节数，非性能数字）。需要全新 UB/ABI/同步 preflight、全部 vendor 预编译、完整原始+真实工作负载+范围真机验收。**这是新建 kernel 代码（不是本仓不改 ascriptor 那条边界内的事——是本仓自己在 `kernels/projects/a2/` 下建派生单元，符合 AGENTS.md §3 的路径，但超出 A2-13 当前写集与"单元级资格化、不碰 kernel"的范围），按 §6.5 记账，攒批或获批准后再动手；PM 已把"批准新建这个 kernel 单元 / 是否纳入一次 kernel 批次"这个决定转给用户，assignee 在裁定前保持kernel 只读，继续独立定位与文档化。**另一条不冲突的路径：A2-13 本身可以不等这个修复，以"资格化未通过、根因已定位并归档"收尾，把 kernel 修复列为独立的后续候选（由用户决定是否、何时纳入 kernel 批次）。

### P2

#### `kda-fwd-bwd-dtype-mismatch` — kda 的 fwd 与 bwd 对同名张量声明了不同 dtype

- **类别** abi · **适用于** KDA · **阻塞** —
- **依据** kda_fwd inputs：beta float32、initial_state float32、g_raw float32，final_state 输出 float32。kda_bwd inputs：beta bfloat16、initial_state bfloat16、g bfloat16、dht bfloat16。 | 2026-09-11 量化（benchmarks/quantify_bwd_dtype_mismatch.py，CPU fp32 参考 + autograd，四个形状 smoke/multi_chunk/gva/kimi_shaped 结论一致）。把三个效应拆开，基准为全程 fp32：A 保存值走 bf16 往返（bwd kernel 实际看到的输入）：dq/dk/dv/dg 相对 L2 1.4e-03~1.6e-03，dbeta 1.6e-04，dinitial_state 3.3e-04。B 梯度输出舍到 bf16（bwd ABI 的输出精度）：一律 ~1.65e-03。C 两者叠加：2.17e-03~2.37e-03，且 C ≈ √(A²+B²)（1.633²+1.686² → 2.347 vs 实测 2.370）—— 两个效应相互独立，没有病态放大。
- **影响** **降精度不免费，但与契约已声明的输出精度同量级。** B（输出舍到 bf16）是 bwd ABI 规定的，无论如何都要付；A（保存值降精度）把总误差从 1.65e-03 的地板抬到 2.3e-03，即 ×1.4。没有引入新性质的误差，因此第二期可以按 .to(bfloat16) 组装 autograd，把这个数记在案。注意这是**算子级**相对 L2，按 AGENTS.md §6 的纪律不能外推成任务级精度结论 —— 要声称对训练的影响，得跑任务级实验并拆出"替换实现"与"改精度"两个对照组。
- **建议** 已量化，不再阻塞第二期。组装 autograd 时照 bwd ABI 降到 bf16，但**显式**做、在 docstring 里写明代价，不要当成无害的类型适配。若后续任务级实验显示不可接受，再回头推动 bwd ABI 接受 FP32（那是 ascriptor 侧的改动）。

#### `npu-builtin-ops-missing` — 内置算子包的覆盖随机器而异：部分 Ascend950PR 机器上 torch_npu 的计算算子不可用

- **类别** environment · **适用于** 全部 · **阻塞** —
- **依据** CANN 9.1.0 的那台 Ascend950PR（见 machine_specs.md）上 $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/ 只有 ascend910_93 与 ascend910b 两个 SoC 目录。torch.randn(device='npu') 报 aclnnInplaceNormal_1_StatelessNormalAiCore 找不到 JSON 配置；torch.zeros、bf16->fp32 Cast 同样失败。torch 本身是 2.10.0+cpu。 【2026-09-11 补充】另一台 8 卡 Ascend950PR 机器（CANN 9.2.0，innerversion V100R001C25B046）的 opp 下有 **ascend950** 算子包，SoC 报 Ascend950PR_9579，实测 randn / zeros / fp32+bf16 matmul / bf16↔fp32 cast / permute+contiguous / einsum / cumsum 全部可用。所以这不是 SoC 级缺陷，而是**算子包安装差异**：CANN 9.1.0 的 opp 只装了 910 系列。
【2026-09-11 再补充】缺算子包时**跨步视图的 D2H 也不可用** —— 它要走 NPU 侧的 `Slice`。最小复现（形状 [1,128,2,8]，对应 C=2 的 g_cumsum）：
  `dev[:, 63::64].cpu()` → RuntimeError：`Op Slice does not has any binary` / `launch failed for Slice, errno:561000`
  `dev.cpu()[:, 63::64]` → 一致
  同一块张量按 C=1 切（`[:, 127::128]`，只取一行）→ 两种写法**都一致**，因为切片等效连续。
这条 C=1/C≥2 的分界正好解释了一次真实失败：`chunk_kda_bwd` 里 `g_last = g_cumsum[:, 63::64]` 绕 CPU 时，single_chunk 与 grouped_idle_cores（都是 C=1）通过，multi_chunk / grouped_heads / gentle_decay（都是 C≥2）全挂。**是硬报错不是静默出错** —— 我最初写成「静默给出错误数据」，最小复现证伪了。修法：先整块 D2H 再在 CPU 上切。
- **影响** 选机器决定能做什么：装了 ascend950 算子包的机器上 torch_npu 基线与 layer 级验证都可做；没装的机器上只能跑自编译 kernel（empty/H2D/D2H/data_ptr/stream 可用，计算算子全不可用）。runtime 桥在两种机器上都工作 —— 这正是它的价值。 【2026-09-11 修正】「runtime 桥在两种机器上都工作」只对**前向**成立。训练路径要在 host 侧补三个反向检查点（`fwd-caches-not-emitted`），那一段是 torch 算子 —— 在缺算子包的机器上原本直接失败。已加 CPU 绕行；层级验证仍然只能在有 ascend950 算子包的机器上做（层里的投影/卷积/softplus/RMSNorm 全是 torch_npu 算子）。 另外：算子入口现在**要求输入连续**并在不满足时报错（`chunk.py` 与 `chunk_bwd.py` 的 `_check`）。在这种机器上「悄悄 .contiguous() 一下」根本做不到 —— device 上要 d2d copy，跨步 D2H 要 Slice，两条都缺，所以只能报错。

**2026-09-20（FMT-02 已合入，#116）**：`layout_device="cpu"` 与「探测内置 d2d copy 后自动绕 CPU」的路径已删除（D-PM-37 不许 host CPU 布局转换），`_resolve_layout` 恒返回 False。所以缺 ascend950 算子包的机器上：布局 / dtype 转换 / 零填充本身不再依赖内置算子（改走自编译 kernel），纯前向在 `check_gate_range=False` 时可用；但**默认的门控跨度检查（device 上的 cumsum / 规约）、带缓存前向与反向里的 `_scan_states`（Cast / matmul）、`dw = -d_vh`（Neg）、`log2(eg)` 分支**这些存量 host 算术没有 CPU 兜底了，会因缺内置算子而失败——这是 D-PM-42（存量例外保留）叠加 D-PM-37 的后果，要等 BF-07 / kernel 批次把它们搬进 kernel 才消除；`AGENTS.md` §5 里「layout_device=auto 会自动探测并绕路」一句因此已过时（等用户同意再改）。
- **建议** 三条路：① 性能基线改用 ascriptor 自己的 profile 子命令 + 自编译 kernel 之间的对比；② 在有完整算子包的机器上做 torch_npu 基线（a2/910B3 有 ascend910b）；③ 确认是否存在 950PR 的算子包可安装。选哪条取决于基线要回答的问题 —— 要对比 ascriptor vs torch_npu 就必须有内置算子，换机器是最直接的。

#### `fixed-kv-128` — K=V=128 固定，不支持其他 head_dim

- **类别** coverage · **适用于** 全部 · **阻塞** `fla-gated-deltanet-default`
- **依据** 各单元 domain 固定 D=128 / K=128 V=128。
- **影响** fla GatedDeltaNetConfig 默认值（head_k=256 / head_v=512）不被支持。
- **建议** 不去支持它 —— 那是 fla 的参考默认值，不是落地模型规格。两个真实目标模型都是 128。门控明确拒绝并在错误信息中说明。

#### `asymmetric-kv-dim` — K 与 V 共用同一个 D 维，不支持 head_k != head_v

- **类别** abi · **适用于** 全部 · **阻塞** `fla-gated-deltanet-default`
- **依据** 各单元 inputs 中 q/k/v 的末维同为 128，domain 只声明单一 D=128（kda 分别声明 K=128 与 V=128，但两者都固定）。
- **影响** expand_v != 1.0 的配置不被支持（fla GatedDeltaNetConfig 默认 expand_v=2.0 → head_v 是 head_k 的两倍）。
- **建议** 与 fixed-kv-128 同样处置：两个真实目标模型都是 head_k == head_v == 128，不去支持非对称维度。门控拒绝并说明。

#### `kernel-nd2nz-suboptimal` — kda 的前后向 kernel 里有一批 ascriptor lint 标出的访存低效点（nd2nz 展开、偶数 block stride）

- **类别** performance · **适用于** KDA · **阻塞** —
- **依据** 编译 kda_fwd 时 ascriptor lint 在 intra.py(2)、triangular_inverse.py(8)、wy.py(2)、recurrent.py(3) 共 15 处报同一条 warning：ub_to_l1.nd2nz 会展开成多次 MTE3 burst（每个 NZ fractal 列一次），不是单条指令，板上实测比 compact-NZ move 慢 10 倍（D-084）。lint 还给了第二条：staging tile 的自然 block stride 是 tile 行数（天然 16 对齐，正好是 bank 阶梯最差的一档，~8.7 cycle/store vs ~1.1），补一行 padding 让它变奇数；D-226 里两个返工的 kernel 上这步收益 -12.3us / -10.8us（总量 -17.7us / -28.2us），比换 move 本身更大。 | 反向侧同族线索：scan_fused.py:53 的 snapshot_and_cast_state_vf 报"strided block store with an even block stride (64)" —— 连续 datablock 撞同一组 UB bank、store 口串行化，板上实测比奇数 stride 慢 ~2x、在 16 的倍数上慢 ~8x（同一条 D-084）；处置同样是把目标行补 1 让 stride 变奇数（65）。
- **影响** 这 15 处在 kernel 源码里，不在本仓。按 D-226 的比例，单 kernel 量级的收益在数十微秒 —— 但要先确认设备侧耗时占比（见 bridge-per-call-overhead），占比低的话改了也看不出来。 反向侧的占比尚未测 —— 当前反向的瓶颈在 host 侧补检查点（fwd-caches-not-emitted），要先换掉那一段，kernel 级优化才看得出效果。
- **建议** 第四期。kernel 源码归 ascriptor 仓所有（AGENTS.md §3：只读），所以这里只登记线索，改动要走 ascriptor 侧。动手前先用 benchmarks/profile_bridge_overhead.py 确认设备侧占比足够大，否则是在优化一个不在关键路径上的东西。

#### `fwd-caches-not-emitted` — kda_bwd 要九个前向检查点，前向 kernel 只直接给出六个

- **类别** abi · **适用于** KDA · **阻塞** `phase 2 性能`, `phase 3`
- **依据** kda_bwd 的 unit.py validate_inputs 要求 saved 恰好含九项：g_cumsum/w/u/qg/kg/v_new 为 (B,T,HV,128)、Aqk/Akk 为 (B,T,HV,64)、h 为 (B,C,HV,128,128)，全 bf16、token-major。前向 kernel 直接产出的只有 w/u/qg/kg（kda_sub3_wy_kernel）、Aqk（kda_sub2_score_kernel）、Akk（tril_inverse64_v2_strict_bf16_kernel）六项。gate kernel 只写 eg = 2**g_cumsum，不写 cumsum 本身；kda_sub45_fused_kernel 内部算了逐 chunk 状态 h 与 v_new = u - w @ h，但只写出 o 与 final_state。单元自己是用 CPU 参考 build_saved_forward() 造 saved 的，不是用 kernel。 | **代价已实测**（benchmarks/bench_kda_train_step.py，Ascend950PR / CANN 9.2.0，bf16，warmup 2 / iters 5，同步）：kimi_linear_layer 形状下补检查点 1.053ms（bd=1）/ 1.065ms（bd=4），long_context 4.242 / 4.351ms —— **随 block_dim 基本不变**，因为它是 torch 算子而不是我们的 kernel。于是它的占比随 block_dim 上升：fwd+bwd 的 7%（bd=1）→ 21%（bd=4）。
- **影响** autograd 的前向必须补齐这三项。g_cumsum 可由 log2(eg) 得到（一次 elementwise）；h 与 v_new 只能重跑 chunk 递推，当前在 host 侧用 torch 做（C 次迭代 × 2 次 bmm）。
**修正先前的判断**：我曾写它是"反向链的性能瓶颈"，实测不是 —— kimi_linear_layer / bd=4 下它占 21%，而九个反向 kernel 占 53%（2.690ms / 5.069ms）。它是一笔确定的、值得收的账，但不是主因。**真正要紧的是它不随核数缩短**：block_dim 上限若被抬高（block-dim-ceiling），kernel 侧会继续变快而这一段不会，占比会继续涨。
**2026-09-11 新发现的第二个后果：它让训练路径依赖内置算子包，而纯前向路径不依赖。**`_scan_states` 与检查点的降 bf16 用的是 Cast / bmm / stack，在只装了 910 算子包的机器上全部不可用 —— 实测表现为 `copy_d2d_baseformat_opapi … error code is 561103` + `Cast ADD_TO_LAUNCHER_LIST_AICORE failed`。这推翻了「我们自己编译的 kernel 在两种机器上都不受影响」这句话的适用范围：它对**前向**成立，对**训练**不成立，因为训练要补的三项检查点不在 kernel 里。已加 `on_cpu` 绕行（`_scan_states(on_cpu=)`、`chunk_kda_bwd` 的 `layout_device`），把检查点生产和那一次 strided `contiguous()` 整段搬到 CPU —— 这是**可用性**开关不是性能开关。把三项挪进 kernel 之后这些绕行可以删掉。
**2026-09-11 顺带修掉的一条**：`chunk_kda` 此前**无条件**走 autograd.Function，于是`no_grad` 下的推理也照样产那九个检查点（纯浪费，占训练步的 21%），而且被**反向**那条更严的门控闸（stable 下 100）挡着 —— 推理本来只受前向的 155 约束。现在不需要梯度时直接走 `chunk_kda_fwd`；`o` / `final_state` 逐位相同（共用同一次 kernel 调用），钉在 tests/test_kda_gating.py::test_chunk_kda_skips_caches_when_no_grad_is_needed。

**2026-09-20（FMT-02 已合入，#116）**：`_scan_states(on_cpu=)` 与 `chunk_kda_bwd(layout_device=)` 的 CPU 绕行随之失效（`_resolve_layout` 恒返回 False）——检查点降 BF16 与布局搬运已在 kernel 里，但 h / v_new 的 host 复算仍在（D-PM-42 存量例外），所以训练路径在缺内置算子包的机器上现在不可用，直到这三个检查点搬进 kernel（本条的 kernel 批次）。
- **建议** 按 AGENTS.md §3 在本仓 kernels/ 下建自己的单元：做一个 kda_sub45_fused_kernel 的变体，额外写出 h 与 v_new（两个 GM 输出 + store，内部量已有），再做一个 gate 变体直接写 g_cumsum。改 ascriptor 仓是不允许的。优先级排在 block-dim-ceiling 之后 —— 先抬核数上限，那一项的收益更大，而且抬完之后这一项的占比才真正凸显。
做完之后顺带删掉 `_scan_states(on_cpu=…)` 与 `chunk_kda_bwd(layout_device=…)` 两处绕行。

#### `modules-are-torch-not-kernels` — modules 层是 torch 原生算子实现，不是本仓自编译的算子

- **类别** performance · **适用于** KDA / GDN-2 · **阻塞** —
- **依据** 通用 `ascend_fla/modules/convolution.py` 仍用 F.conv1d+F.silu，`fused_norm_gated.py` 仍用rsqrt/mean/激活；KDA的投影/l2norm/门控也仍是torch算子。GDN-2真实B1T1 BF16 packed路径的模型专用CCE例外现在有 `a5.gdn2_fused_decode`、`a5.gdn2_short_conv_decode`，以及显式opt-in的 `a5.gdn2_norm2_w12_swiglu`。后者v1整网已拒绝；v2 reference2/2与random sim/pipesim通过，relative-L2=3.416e-7、无hazard/deadlock；`mlp_backend='cce'`才启用。通用卷积、其他projection、W3、prompt MLP及训练路径仍依赖torch_npu。
- **影响** ① 这些步骤在内置算子包不全的机器上不可用（需要 conv1d/silu/matmul），而自编译的 kda 算子本身不受影响 —— 所以层级验证比算子级验证对机器挑剔。② 层级耗时里有一部分不归本仓的"高效率算子"管，报层级性能数时必须拆开说，否则会把 torch 的开销算进算子账上。
- **建议** KDA保持原计划。GDN-2混合MLP的kernel、显式接线、真实95B递推、同卡Graph夹心与profile均已完成。下一步先决定是否用约1.030GB派生布局换2.6%吞吐，或设计不重复权重的prompt/decode统一布局；追求大幅提升则先决策weight-only低精度与任务质量预算。通用causal_conv1d与FusedRMSNormGated仍需独立算子，不能从模型专用T1 kernel外推为已完成。

#### `stable-unit-no-harness` — 本仓自有的三个单元都还不能用 ascriptor harness 独立跑

- **类别** verification · **适用于** KDA · **阻塞** —
- **依据** kernels/projects/a5/kda_fwd_stable/ 目前有 contract.json、README.md 与三个 kernel 文件，但缺 unit 协议要求的 unit.py（make_inputs/reference/execute）与 run.py —— 它们要对接 ascriptor 的 _unit_runner。现在的验证全部经本仓 runtime 桥 + pytest 做。
（2026-09-11 起这条覆盖三个单元：`kda_fwd_stable`、`kda_bwd_stable`、以及本仓自写的 `kda_fused_recurrent`。三者的 `ascriptor check` 与 runtime 桥都过了，缺的是 `unit.py` + `run.py` 对接 `_unit_runner`。）
- **影响** ① 拿不到 ascriptor harness 的 sim / pipesim / cannsim 几个 stage 的证据，也就用不上它的逐 stage checkpoint 比对（那对定位 kernel 内部错误很有用）。② 这个单元不能被 ascriptor 侧的人独立复现，不利于把修法推回上游。contract.json 的 support 里已如实标注证据来源，没有假装有 harness 证据。
（2026-09-11 起这条覆盖三个单元：`kda_fwd_stable`、`kda_bwd_stable`、以及本仓自写的 `kda_fused_recurrent`。三者的 `ascriptor check` 与 runtime 桥都过了，缺的是 `unit.py` + `run.py` 对接 `_unit_runner`。）
- **建议** 补 unit.py 与 run.py。reference 可以直接用 ascend_fla/reference/kda.py 的逐 token 递推版（它没有跨度上限，正是宽域下唯一可用的 oracle）。做完后把 contract.json 的 support 按 harness 实际结果更新。
（2026-09-11 起这条覆盖三个单元：`kda_fwd_stable`、`kda_bwd_stable`、以及本仓自写的 `kda_fused_recurrent`。三者的 `ascriptor check` 与 runtime 桥都过了，缺的是 `unit.py` + `run.py` 对接 `_unit_runner`。）

#### `gate-span-still-bounded` — 稳定化把门控跨度上限从 80 抬到前向 155 / 反向 105，但没有去掉上限

- **类别** numerics · **适用于** KDA · **阻塞** —
- **依据** `kda_fwd_stable` / `kda_bwd_stable` 走的是**对称分解**：把 `exp(a_i − a_j)` 拆成两个以中点为锚的因子，各压到 ±span/2，有限性的理论上限正好翻倍 —— 前向 `2 × -ln(FLT_MIN_NORMAL) ≈ 174.7`，反向 `2 × ln(BF16_MAX) ≈ 177.4`。
**但两条链的闸不是同一回事，这是实测出来的**：
* 前向的约束是**有限性**。实测跨度到 155.97 时 `o` 的相对 L2 仍稳定在 2.85e-03~3.19e-03，完全不随跨度退化 —— 所以闸就设在实测最深点 155。
* 反向的约束是**精度，而且它先于有限性到来**。梯度到 169.76 都还是有限值，但对 fp32 递推参考的相对 L2 随跨度单调上升：dq 在 46/94/105/110/130/169 处是 2.89e-02 / 4.55e-02 / 4.86e-02 / 4.99e-02 / 6.17e-02 / 6.88e-02，**130 处越过契约预算 0.05**；dg 在 169 处崩到 6.49e-01（预算 0.25）。所以反向的闸设在 105。
**105 的下界是 fla 初始化本身的上界，这点此前被我写错了。** 跨度不是常量 ~94，它是**随机变量** —— 跨度 ∝ `max_hv exp(A_log)`，而 fla 取 `A_log = log(U(1,16))`，所以 `exp(A_log) ∈ [1,16]`。实测 12 个 seed：HV=1 给 21.5~96.2、HV=2 给 15.3~94.8、**HV=8 给 63.1~100.9**（头数越多越稳地顶到上界，因为取 max 的样本更多）。甚至同一个 seed 下，建层与抽 x 的先后顺序不同就从 64.6 变成 94.0（RNG 消耗顺序不同）。上界是 `exp(A_log)≤16 × dt≤0.1 × 63 步 ≈ 100.8`。**闸必须覆盖 100.8**，否则默认初始化的层会被我们自己的门控拒掉 —— 这就排除了 100。上限那头是契约预算还成立的最深实测点：110 处 dq=4.99e-02 只剩 0.2% 余量，不取；105 处 dq=4.861e-02，余量 2.8%。
于是 `MAX_GATE_SPAN` 是二维的：`{impl: {forward, backward}}`，纯推理用前向那条、训练用反向那条。fla 默认初始化的层跨度约 94，两条都满足。
**跨度不随 T 增长** —— 它是 chunk 内（64 token）的量，cumsum 每 chunk 重置。推高它的是 `exp(A_log)` 与 `dt` 的乘积。
- **影响** ① fla 默认初始化的跨度上界是 100.8，反向的闸 105 只剩 4% 余量，而实测 HV=8 时 8 个 seed 里就有一个到 100.6。`A_log` 与 `dt_bias` 都是可训练参数，训练中 `dt` 变大就会撞上限 —— 届时是**报错**（设计如此），但会中断训练。要继续得显式 `check_gate_range=False` 并接受超预算的梯度，或者压 `dt`。
② 门控检查默认开，每次前向对 g 做一次 cumsum + 两次规约，实测 0.212ms / 步（bd=4 / kimi_linear_layer，占训练步 4%）。
③ **有限性上限（反向 169.76）比声明上限宽**。需要更深跨度又能接受精度退化的调用方，可以自己关掉检查 —— 曲线在 `kda_bwd_stable/contract.json` 的 `accuracy_vs_span` 里，照着选，不要瞎试。
- **建议** 现在不做。真撞上限时按代价排序：
① **先查 dq 为什么是约束项**。它在跨度 46 时就已用掉 58% 预算，说明深衰减下 `dq` 的主项（`d_qg · exp(g) · scale`，见 inverse_epilogue）对 bf16 的 `g` 最敏感。若把 `g_cumsum` 检查点从 bf16 升到 fp32（kda_bwd 的 ABI 问题，见 kda-fwd-bwd-dtype-mismatch），这条曲线可能整体下移 —— **这是推测，要测**。
② 把 64×64 的 tile 再按行列分块，每对子块用各自的中点（等价于分块 log-sum-exp），有限性上限随分块数线性增长。但若约束是精度而不是有限性，这一项帮不上忙。
③ fla 的 `lower_bound` / `safe_gate`：给门控设下界。那**会改变数学**，属于模型侧决策，不能当数值修补悄悄加上（本仓目前显式拒绝这两个开关）。

#### `a2-sim-vs-toolchain-blind-spots` — A2 单元『sim 通过』不等于能编译 / 放得下：BF16 GM 标量读在 bisheng 编译期失败，功能模拟器不校验物理地址分配

- **类别** verification · **适用于** KDA · **阻塞** —
- **依据** **来源：同 `a2-cast-blkstride-sim-blind`（A2-09 的 RISK，#112，自述，PM 未复现；只对 910B3 / CANN 9.0.0 / pin 90cfcdc 成立，A2-11 之前只作观测）。**
- 从 BF16 内存直接做标量读（`Var.GetValueFrom(bf16_gm[...])`）在真机编译期失败：bisheng `fatal error: error in backend: not support bf16 type cast`；check / sim / pipesim 都不报（自述）。A2-03 的前向没踩到，因为那边 beta 是 FP32。绕行：先 `gm_to_ub_pad` 进 UB，向量侧 cast（逐行，见 `a2-cast-blkstride-sim-blind`），再从 FP32 UB 读标量。
- 功能模拟器不校验物理地址分配；pipesim 与真机会：四个 kernel 在 sim 下全绿，换 pipesim 才报 `addr_alloc: UB overflow` / `L0C overflow`（自述：finalize_pair 要 192KB L0C，finalize_pre / finalize_post / inverse_epilogue 要 208~229KB UB）。
- **影响** 都是『响』的失败（编译期或 pipesim / 真机报错），不是静默错误；代价是 sim 全绿会让人误以为 kernel 可用。只对 A2 系成立；A5 单元的 compile / pipesim 路径不在此列。
- **建议** 1. 本仓侧：A2 单元报『sim 通过』之前，每个 kernel 至少跑一次 `ascriptor compile --backend cce` 与 pipesim（已写进 A2-09 的验收）；BF16 GM 不做标量读。2. 转给 ascriptor 所有者：功能模拟器校验物理地址分配、BF16 标量读在 check 阶段就报错。

#### `a2-autosync-missing-cross-pipe-guards` — A2 上 auto_sync 生成的 MTE2→MTE1 ready 列表不完整（静态观察；申领人已撤回「导致 C=2 出错」的因果，未观察到后果）

- **类别** verification · **适用于** KDA · **阻塞** —
- **依据** **来源：A2-09 的 STATUS（#112，2026-09-20T05:43Z）与其后的更正 RISK（07:16Z，申领人自述；PM 无 A2 真机、未复现；只对 910B3 / CANN 9.0.0 / ascriptor pin 90cfcdc 成立，A2-11 之前只作观测）。**
- **撤回**：申领人 05:43Z 报「auto_sync 漏跨 pipe 保护 → 第二个 chunk 的 qk_right 整体错（max_abs 8.0）」，07:16Z 撤回该因果。他按 A2-01 的 settle / sham 口径做了 A/B（同一台 910B3、同一 case `multi_chunk_bd1` 即 C=2、同一份代码，只差 `finalize_pair` 里两处 `DEvent(Pipe.MTE2, Pipe.MTE1)` 栅栏的有无，两个副本各跑一张空闲卡）：两个变体的 `finalize_pair.qk_right` 输出**逐位相同**（max_abs=8.000000e+00、rel_l2=1.056e-07，chunk 行 64–127 的 max_abs=8.0，chunk 行 0–63 为 7.6e-06），栅栏什么都没改变。max_abs=8.0 自始至终是**同一个元素**的舍入差（16384 个元素里 1 个，1408 对 1416，约 1.45 个 BF16 ulp，FP32 累加顺序），不是同步缺陷。那一轮之所以从「失败」变「通过」，是因为同一个提交里还改了逐 kernel checkpoint 的判据（加 `stage_outputs`），起作用的是判据、不是栅栏；当时没有把两个改动分开验证。
- **仍成立的静态事实（申领人照生成的 `*_cube.h` 念的，不是推断）**：`DEvent<PIPE_MTE2, PIPE_MTE1, 0, 0, 1> ev_mte2_mte1_ready_1;  // guards l1_k, l1_kg, l1_mbase, l1_mbeta, l1_mqk` 里没有 `l1_q`；`inverse_mm` 的列表缺 `l1_akk` / `l1_dv` / `l1_vnew`，`scan_fused` 的缺 `l1_w`。**没有证据说它会导致错误结果**：26 个真机 case 没有一格因此出错，A/B 也无差异。栅栏留在代码里按「预防」注明，不再声称它修了什么。
- 过程教训：这正是 `AGENTS.md` §6.5「按症状规律倒推根因」——同一次改动里混了两个变量就把通过归因给了其中一个。PM 要最小复现，A/B 把因果证伪了，这是证据流程按设计起作用。
- **影响** 观察性：没有观察到错误结果。潜在风险是竞态类（生成代码的 ready 列表缺缓冲，复现概率可能依赖时序），所以留着预防性栅栏；但 PM 不再把它记成静默错误类。（对比 `a2-cast-blkstride-sim-blind`：那条有独立证据——`got/ref * beta[col]` 恒为 0.5547 = beta[0]，改逐行 cast 后逐位一致。）
- **建议** 1. 本仓侧：a2 单元里保留栅栏作预防，单元 README 注明「预防、未观察到后果、不声称修了什么」；(C, HV) 网格里 C≥2 照扫。2. 转给 ascriptor 所有者：若确认 auto_sync 的 ready 列表本应包含这些缓冲，补全（这条只是静态观察）。3. A2-09 的 DONE 带上这次 A/B 的负结果原始输出（文本），不再为它要求「最小复现」。

#### `a2-kda-bwd-pair-checkpoint-thin-margin` — A2 上 kda_bwd 的 finalize_pair 四个积：checkpoint 相对 L2 1e-5 的分辨率就是「有没有一次 BF16 翻转」——26 例里 2 处（各是一个小元素差 1 个 BF16 ulp）真机独有，sim / pipesim 为 0，判据靠固定种子过

- **类别** numerics · **适用于** KDA · **阻塞** —
- **依据** **来源：A2-09 的 DONE（#112，PR #119 头 276d53d，2026-09-20T13:50Z；PM 从 `evidence/unit/aclnn.json` 原始回执复算，sim / pipesim 回执对照；真机数字自述、PM 无 A2 真机，只对 910B3 / CANN 9.0.0 / pin 90cfcdc 成立，A2-11 之前只作观测。）**
- 契约给 `finalize_pair.{qk_left,qk_right,s_base,t_beta}` 的判据是相对 L2 1e-5，`reason` 与 README 的依据是「四个 case（C=1/2/3、HV=1/2/4）实测最差 1.06e-07，比实测宽两个数量级」。26 个真机 case 的实数：`qk_left` 在 `gentle_decay_bd1` / `bd2` 是 8.296e-06、`t_beta` 在 `grid_c2_hv4_bd1` 是 7.000e-06（预算的 83% / 70%），其余 23 例都 ≤ 1.06e-07（大多是 0 或 1e-9 以下）；ulp 采样的四个 case 没含这两个。
- `gentle_decay_bd1` 与 `grid_c2_hv4_bd1` 在 sim / pipesim 回执里这两项都是 0.0：只有真机偏离。
- kernel 里四个积各写自己的 L0C tile，`splitn` + `is_init=True`，每个 tile 只有一次 MMAD（源码注释与 PM 读源码一致），所以**不是** `a2-splitk-fp32-cube` 那类累加链问题。
- **已复算（PM 在 CPU 上用 `reference_stages` 对参考张量算「单元素差 1 个 BF16 ulp 的相对 L2」= `ulp(x)/||ref||`；申领人 14:45Z 补跑的 `ulp_stats.py` 说这两项「逐元素最多 1 ulp、超 1 ulp 占 0%」）**：`gentle_decay_bd1` 的 `qk_left`：`||ref||=1.796191e-3`，指数 −19 的元素（2548 个，`|x|` 为 `|ref|max` 的 2.05%~4.09%）差 1 ulp = 2^-26，得 8.2960e-6，与实测 8.296e-6 五位一致；`grid_c2_hv4_bd1` 的 `t_beta`：`||ref||=2.857535e5`，指数 8 的元素（276 个，为 `|ref|max` 的 0.17%~0.33%）差 1 ulp = 2，得 6.999e-6，与实测 7.000e-6 吻合。所以两处偏离**各自就是一个（小）元素差 1 个 BF16 ulp**，其余元素逐位相同；申领人 README 里「落在占张量范数很大份额的元素上」的说法对这两处都不成立，已请其改。
- **判据分辨率**：`gentle_decay` 的 `qk_left`（N=16384）里中位数元素差 1 ulp 就是 1.66e-5，64% 的元素单独差 1 ulp 就超 1e-5；`t_beta`（N=65536、动态范围大）只有 0.62% 的元素会超。所以 1e-5 这条相对 L2 判据的分辨率就是「有没有一次翻转」，不是连续的 1.2× / 1.4× 余量：翻转落在小元素上才过。固定种子下确定性（bd1 = bd2 逐位相同），换种子 / 形状 / 卡可能翻过来。
- 求和顺序不同能不能解释「真机为什么与 torch 在这两个元素上舍到不同的一侧」——没有确立（sim / pipesim 在这两项上恰好 0.0，`gentle_decay` 的操作数不跨数量级）。
- **影响** 只影响 A2 的 stage checkpoint 判据；六项输出梯度的预算（沿用 a5）不受影响（最差占预算 77%，dq @ zero_initial_state）。失败方式是「响」的（stage 检查报 failed），不是静默错误。代价是：换种子 / 形状 / 卡时，若 1 ulp 翻转落在中位数级元素上（`gentle_decay` 类张量里 64% 的元素），checkpoint 会超 1e-5 而误报；T 更大的真实形状里翻转的元素数会随之增加。**A2-09 的 README / contract.reason「比实测宽两个数量级」只对采样的四个 case 成立**，已要求 README 改成 26 例的实数与上面的分辨率说明。
- **建议** 1. A2-09（已要求，README-only）：机制改成实数、写明分辨率 ≈ 一次翻转；「≥95×」改「≥94×」；「58 ulp」写明只在 6 个 case 上测过。`contract.json` 的 `reason` 因为在 unit digest 里，留到下一次因别的原因动 contract 时同改。2. 下一次动 A2 契约时，把 `finalize_pair` 这一档的 stage 判据改成对翻转敏感的写法（超 1 ulp 的元素数 = 0、被翻转元素数有上限），不要只用相对 L2——相对 L2 的 1e-5 在 N 万级元素的 BF16 张量上就是单次翻转的量级。3. A2-11 定性 A2 的 cube FP32 / BF16 数值时对照，不据此单独下结论。

#### `kda-prep-nearzero-output-underflow` — KDA raw 前处理（BF-07）的 nearzero 输出落在极端下溢区（|golden|max ≲ 3.2e-37）时数值判据失败：candidate = 旧 NPU，逻辑上无解的比较口径，用户批准只披露、不设通过线（D-PM-52）

- **类别** numerics · **适用于** KDA · **阻塞** —
- **依据** **来源：BF-07（#106，PR #118 已合入 381b452）；PM 从原始回执与逐位置记录复算；申领人的 A5 真机（CANN 9.1.0-beta.1），结论只对该环境成立。**
- 失败范围：chunk 每 bd 16 例（40 个 head/chunk 位置）、decode 每 bd 44 例（148 个位置），六个 bd 的失败 ID / 位置 / 指标完全一致；触发条件是 q / k ×~1e-20 这类近零输入使输出 |golden|max ≲ 3.2e-37（chunk 最大 3.2323e-37 = 27.50 × FP32 最小正规数，decode 最大 1.1480e-38 = 0.977 ×）。
- 与旧 host 路径的关系：candidate 与旧 NPU 逐位相同（chunk 35/40、decode 115/148）或差 ≤ 1 BF16 ULP（chunk）/ ≤ 4 FP32 ULP（decode）——没有回退，旧 host 路径同样失败；这不是 CPU 正确性通过。
- 必要下界（PM 从回执的范数与距离重算）：decode 148 个位置里 124 个是「两套 CPU FP32 参考的允许误差球无交集」（两套参考在 ~1e-43 量级彼此就差 ~33%，最大距离 / 半径 17,930.9）、24 个是「BF16 最近舍入误差已超阈值」；chunk 40 个位置里 16 个 BF16 最近舍入 L2 > 0.05；其余 chunk 位置是真实的 CPU-normal 数值差异，不证明不可修。
- 另有一类：q / k ×1e-20 时 decode 的 final_state 对旧 prep 的 CPU reference 相差 4.3e-4~9.8e-4（1e-5 判据），原因是 eps 主导下商恰在 BF16 中点、candidate 与旧 FP32 商落在两侧（一次舍入翻转，q / k 约 1% 元素差 1 ulp）；D-PM-51（用户批准）只对 `nearzero_*` decode 组改判为「对实际 prep 输出的 reference ≤ 1e-5」（全部 384 例 state 最大 9.438e-7）。
- **影响** 只影响输出量级 ≲ 3.2e-37 的极端近零输入（绝对误差同量级，数值上无实际意义）；失败是「响」的（contract board stage 保持 failed，逐切片表公开），不是静默错误；普通域 25,920 个 case 与完整 Kimi workload 不受影响。不影响 A2 / A3；BF-08（梯度链）会在自己的端点披露里再出现同类情形。
- **建议** 1. 已由用户批准（D-PM-52）：只披露、不设通过线，类定义 = 切片 |golden|max ≤ 32 × FP32 最小正规数 且 candidate 与旧 NPU 逐位相同或 ≤ 1 BF16 / 4 FP32 ULP；类外失败仍是失败。2. 要闭合这些失败必须先改比较口径（现有口径对 decode 148/148、chunk 16/40 逻辑上无解），单改 kernel 不可能过；不建议改 inherited kernel 或收窄域。3. BF-08 沿用同一披露框架；新的类别或需要新通过线的，先 RISK 给 PM、由 PM 转用户。

#### `kda-prep-backward-endpoint-disclosures` — KDA raw 前处理反向（BF-08）的端点披露：每 bd 482 个 native 下溢输出不是 CPU 正确性通过、30 个 gate A_log = 88 输出记录留作未资格化观察

- **类别** numerics · **适用于** KDA · **阻塞** —
- **依据** **来源：BF-08（#117，PR #122 → 6c4aaca）；PM 从端点跨 bd 汇总与原始四列回执核过；申领人的 A5 真机（CANN 9.1.0-beta.1），结论只对该环境成立。**
- native 下溢（D-PM-56 (2) / D-PM-50 元素分类 (b)）：每 bd 482 个输出——gate A_log = −100（BF16 239、FP32 241）与 beta = −88（两 dtype 各 1）：golden 为 FP32 subnormal、candidate 为 +0、旧 NPU 为 −0（数值 ULP = 0 但字节不同）、CPU FP32 为 subnormal；四列数值与位跨 bd 全同；**不是 CPU 正确性通过**；公共入口在 beta = −88 / 16 / 20 与 A_log = −120 / −100 可达。
- gate A_log = 88（D-PM-56 (3)）：每 bd 30 个输出记录（BF16 dg 2、FP32 dg 13 与对应 dt_bias 15）上 candidate 既不等于 CPU 也不等于旧 NPU 且涉及非有限值——14 个两旧路径均 +Inf、16 个 CPU +Inf / 旧 NPU 有限（与 candidate 有有限舍入差）；candidate 有限且更接近 FP64（例：BF16 dg[0,0,0,72] candidate 2.1002e38、旧 NPU = CPU = +Inf、FP64 2.1011e38）；candidate 独有 NaN = 0；默认公共 gate 检查在 A_log 80 / 88 / 89 / 100 两 dtype 都提前拒绝（只有 check_gate_range = False 才走得到）。
- contract 的 board 阶段保持 failed、merge_authorized = false；这些是披露，不是通过。另一条发现：冻结的逐元素限值在完整 Kimi 人口上被旧 host 图自己超过（q 至少 0.0464、k 2.166，限值 0.0588），旧图在 raw beta = 16 处相对 FP64 约 5.8% / 9 BF16 ULP 之外。
- **影响** 只影响：亚正规 / 上溢端点的输入（默认公共闸就会拒绝 A_log ≥ 80 一类）与 native 下溢的输出；这些输出的绝对量级极小或输入被默认闸拒绝，不是静默错误；普通域 58,320 次公开训练调用与完整 Kimi 全部满足冻结预算。
- **建议** 用户在 D-PM-57 知情后授权合入，端点按披露处理，不需要动作；若将来要把这两类改成有通过线的口径（例如 A_log = 88 的候选有限而旧路径 Inf 的一类），需要用户另行裁定新端点类（D-PM-52 的「只披露」口径不外推）。BF-09 沿用同一披露框架，新类别先 RISK。

#### `pkda-fp32-host-domain-validation-unregistered` — PKDA FP32 公共入口的域校验（key 行 L2 范数与域判定）还在 host 上算，写在未登记的函数里；BF16 兄弟单元已把同样的校验放在 kernel 侧状态字

- **类别** validation · **适用于** KDA · **阻塞** —
- **依据** **来源：FMT-01（#108，PR #124，A5 真机，CANN 9.3.0 / torch 2.7.1；PM 从原始 JSON 复算）。** `chunk_precond_kda` FP32 路径 host 上发出 6 次算术（pow / sum / neg），全在 `ascend_fla/ops/pkda_chunk_fwd.py:126` 的 `ref.reference.validate_inputs`（算 key 行 L2 范数与域判定，产出只是判定）；BF16 路径在 `:102` 直接进 `_bf16_module().execute(...)`，域校验由 kernel 侧状态字完成（`kernels/projects/a5/pkda_chunk_fwd_bf16/kernels/pipeline.py` 的 `ERRORS` 1–7，含 "key row L2 norm must be <=1+1e-5"），host 上一个校验算子都没有——两份 trace 的差别就是证据。这是 D-PM-42 ① 意义下的「只读校验」（暂列、不是 D-PM-37 的 dtype / 格式 / 算术违规），只是写在未登记的函数里、在 host 上算。
- **影响** 不影响结果，只是 host 侧多一次同步与若干算术；审计工具把它按算术报、需要人工判断它是校验。BF16 已有先例，改法范围小。
- **建议** P2：PKDA FP32 路径的域校验搬成 kernel 侧状态字 + host 回读，与 BF16 兄弟单元一致（`ascend_fla/ops/pkda_chunk_fwd.py:126` 一处调用 + 可能的派生单元改动）；验收：审计报 clean、拒绝面不变（同样的非法输入仍然报错，错误信息等价）。也可以选择把 `validate_inputs` 登记为只读校验（D-PM-42），二选一，PM 待有空闲任务槽再立项。

#### `soc-defaults-a5-outside-kda` — SoC 显式化只做了 KDA 一侧：GDN / PGDN / PKDA / GDN-2 各族的入口仍写死 device="a5" 默认，各族 kernels/projects/a5 单元路径也是字面量，能力表的 unit_root 还没有任何算子读它

- **类别** validation · **适用于** KDA / GDN / GDN-2 · **阻塞** —
- **依据** **来源：A2-02（#31，PR #126，squash 33259af；PM 在合入后的 main 上 grep 与读源码，没有在 A2 机器上试过这些入口）。** A2-02 在 `ascend_fla/platform.py` 建了 SoC 解析（显式实参 → ASCEND_FLA_SOC → 设备名探测）与能力表，并把 `runtime/compile.py`、`ops/kda/**` 的 `device` 默认改成 None、对外入口在编译前按 SoC 验收。范围只到 `ops/kda/**` + `runtime/` + `layers/kda.py`（其余不在 A2-02 写集，`ops/gdn2/**` 在 reserved_paths）。剩下的：`ascend_fla/ops/gdn_chunk_{fwd,bwd}.py`、`pgdn_chunk_fwd.py`、`pkda_chunk_fwd.py`、`gdn2_chunk_fwd.py`、`ops/gdn2/*.py` 里仍有 `device='a5'` 默认与 `if device != 'a5'` 检查；各族与 KDA 的 `kernels/projects/a5/<unit>` 路径都是字面量（KDA 侧的路径路由按 PM 裁定延后到第一个 a2 单元出现时），`CAPABILITIES[soc]['unit_root']` 目前只被测试读。
- **影响** 不构成已知的静默错误：这些入口要么对非 a5 的 device 显式报错，要么按 a5 profile 编译，都没有像 KDA 那样在编译前按 SoC 报「未验收」；在 A2 机器上直接调用它们的后果没有实测（PM 只读了源码）。等对应的 A2 任务接 a2 时才会变成真问题。
- **建议** P2：各族在接 a2 时用 A2-02 的同一个模式——`device` 默认 None → `resolve_soc`、对外入口先 `require_qualified`；单元路径接 `platform.unit_root(soc)`（第一个 a2 单元出现时一并做 KDA 侧的路径路由）。落点：GDN 的 A2 任务（A2-20 / kernel 批次 A2-K1）、PGDN / PKDA 的 a2 派生；`gdn2` 在 reserved_paths 内由仓主轨道自行处理。
