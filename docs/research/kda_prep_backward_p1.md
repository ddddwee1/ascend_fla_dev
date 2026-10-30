# KDA raw preparation backward — BF-09-P1：按 85% 管线利用率线回头复核

本文件是 BF-09-P1 的交付，**不覆盖** BF-09 的 `kda_prep_backward.md`。那一份记录
BF-09 当时的优化与收尾；这一份只做一件事：把 `kda_prep` 每个 kernel 的真机
`PipeUtilization` 量出来，按 AGENTS.md §6（2026-10-27 用户给出）的 85% 线逐个判定，
并核实 `a5-fp32-cube-matmul-inherits-hf32`（P0）在这个单元上是否暴露。

结论先放前面，三条：

1. **反向的三个重 kernel 本来就在 85% 线之上**（vector 管线 0.924~0.997），
   `gate_reduce` 与 beta 前向辅助 kernel 在线下（最高管线占比 0.467~0.75），但两者合计只占
   总设备耗时的 **0.07%**。**前向三个 kernel 全部在线下**（最高 0.575）。
2. **85% 线在这个 kernel 上失去判别力，而且比 DR-P1 报的那个例子更极端。**
   `gate` 反向 kernel 的 `aiv_vec_ratio` 在 `block_dim` 1 / 4 / 28 上**都是 0.996**，
   而同一份逐位相同的输出对应的设备耗时是 **645.8ms / 164.8ms / 24.4ms**——
   占用率读数完全不动，真机耗时差 **26.5 倍**。DR-P1 测到的是 1.13~1.69 倍。
3. **`kda_prep` 没有任何 cube matmul**，因此不暴露于那条 P0。真机 A/B（带有效正对照）
   + 13 份 capture 的 `aic_mac_ratio` / `cube_utilization` 全 0 + 源码零 `matmul(`，三路证据一致。

**本任务没有改动任何生产代码。** 唯一改动是 `tests/test_kda_prep_backward.py` 里加了一条
NPU-free 守卫（见 §5 末）。原因同在 §5：真正的瓶颈已经定位、也已量化，但它落在
`ascend_fla/ops/kda/autograd.py` 与 `ascend_fla/platform.py`——本任务写集之外
（前者是 A2-15 的写集），按 D-PM-33 要先问用户。

`backward_budgets.json` 字节未变，SHA-256 仍为
`af053b4d3987eb9baf8ff8636b57f421f1f6531bd57f2fe10e176e9dba306b31`。


## 1. 环境与口径

| 项 | 值 |
|---|---|
| SoC / 设备名 | Ascend950PR，Torch 报 `Ascend950PR_957b` |
| CANN | 9.1.0（`/usr/local/Ascend/cann-9.1.0`；OPP 含 `ascend950`/`ascend910b`/`ascend910_93`） |
| Python / Torch / torch_npu | 3.12.14 / 2.12.0+cu130 / 2.12.0 |
| ascriptor library | pin `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`，156 个 `.py` 的排序清单哈希 `4da507e8cdf6f6add1a56f84a4c32ca64022667b0b44129fd2b14564811a71d1`（与本仓 pin 逐文件一致） |
| 被测源码 | `kernels/backward.py` `ba38f200…cba7dbb56c`（= BF-09 收尾冻结的那一份）、`kernels/forward.py` `135b1fa2…c9b2b455`、`runtime.py` `981feab5…f7b1d577a36a` |
| profiler | `torch_npu.profiler`，Level1，`aic_metrics=PipeUtilization`，`schedule(wait=0, warmup=1, active=1, repeat=1)` |
| 形状 | `docs/matrix/models.json` 的真实形状，不是契约玩具形状（见下） |
| 机器 | 共享主机，用户已授权共享健康卡（见 `machine_specs.md`）。每次启动前在卡锁下复查 `npu-smi` 健康与占用，发现他人进程就**拒绝启动**并等待，不 kill、不复位 |

形状口径：

- `kimi_t4096` = B1/T4096/H32/HV32。**这是 BF-09 自己 profile 用的那个形状**
  （研究文档写的"完整 Kimi B1/T4096/H=HV32"），gate 阶段元素数 16,777,216，
  与 BF-09 文中那个数字对得上。
- `long_context` = B1/T4096/H16/HV32（`models.json`）。
- `kimi_linear_layer` = B1/T1024/H32/HV32（`models.json`）。

输入分布**照抄** `ref/backward_calibrate.py` 的 `full_kimi` 口径（`g ~ randn`、
`dt_bias = linspace(-.2,.2,HV*128)`、`A_log = linspace(-3,2.7,HV)`、`gy ~ randn`），
这样冻结预算可以直接套用，不需要重新解释。

一个进程 = 一份 build = 一个 `block_dim` = 一次 capture（§6 铁律 2）。每个 `block_dim`
有自己的 `ASCEND_FLA_CACHE`；每个进程在第一次 custom launch 之前编完全部
**30 个 vendor**（14 前向 + 16 反向，§5）。

### 工具链：`pypto-pro-op-orchestrator` 在这里用不上

AGENTS.md §6 要求优化前装 PyPTO-Pro 框架与官方 `pypto-pro-op-orchestrator` 插件。
**本任务核实后判定不适用，照 KDA-P1 / GDN-P1 的先例如实记录。**
`kda_prep` 的 7 个 kernel 全部是 ascriptor 的 `@kernel(mode='vec')` + `@vf()`
（`kernels/forward.py`、`kernels/backward.py`），经本仓 `ascend_fla/runtime/compile.py`
编成 CANN 自定义算子；整条链里没有 PyPTO/PyPTO-Pro 的 DSL 或 schedule。
这跟 DR-P1 那个 `gdr_chunk_fwd_bwd_pypto_pro` 原生 PyPTO-Pro 单元不是一回事。
插件那套"规划/编码/调试/验证/调优/工具"子代理作用在 PyPTO-Pro 的算子上，
对 ascriptor/CCE 发射没有抓手，所以没有硬装。


## 2. 逐 kernel 的 `PipeUtilization` 与 85% 判定

判定口径按 AGENTS.md §6：计算类取 `aic_mac_ratio`/`aiv_vec_ratio` 的较大者，
搬运类取 `aic_mte2_ratio`/`aiv_mte2_ratio`/`aiv_mte3_ratio`/`aic_mte1_ratio` 的较大者，
两者再取较大者与 0.85 比。全部是真机硬件计数器，不是模拟器的 `model_cycles`。

### 2.1 反向，`kimi_t4096`，`block_dim=4`（BF-09 的那一档）

| kernel | 次数 | 设备 µs | 瓶颈 | 该管线占比 | ≥85% |
|---|---:|---:|---|---:|:--:|
| `KdaPrepBackwardGateBf16F32F32Kernel` | 1 | 164801.80 | 计算 (vec) | **0.9960** | ✅ |
| `KdaPrepBackwardNormBf16Kernel` | 2 | 17747.04 | 计算 (vec) | **0.9400** | ✅ |
| `KdaPrepBackwardBetaBf16Kernel` | 1 | 301.27 | 计算 (vec) | **0.9520** | ✅ |
| `KdaPrepBackwardGateReduceF32F32Kernel` | 1 | 116.27 | 计算 (vec) 0.530 / 搬运 (mte2) 0.459 | 0.5300 | ❌ |
| `KdaPrepChunkBetaBf16Kernel`（前向 sigmoid，反向要用它的 `probability`） | 1 | 10.14 | 搬运 (mte2) | 0.6820 | ❌ |
| **合计** | | **182976.53** | | | |

gate 反向那一行的完整读数：`aic_mac_ratio` 0.0、`aiv_vec_ratio` 0.996、
`aiv_mte2_ratio` 0.004、`aiv_mte3_ratio` 0.002、`aic_*` 全 0、`cube_utilization(%)` 0.0。

同一进程的同步墙钟（7 轮，card 2）是 190.83ms，设备 kernel 累计 182.98ms——
**profiler 在这里几乎不扰动**（差 4%）。这跟 BF-08 时期"Level1 把 37.6/425.7ms 抬成
81.9/476.1ms"的情形不同，所以本文的设备时间可以直接当量级用。

### 2.2 反向，其它形状与 `block_dim`

| 配置 | gate 反向 µs | norm 反向 µs (×2) | beta 反向 µs | gate_reduce µs | 合计 µs |
|---|---:|---:|---:|---:|---:|
| `kimi_t4096` / bd1 | 645813.07 | 70969.10 | 1184.41 | 482.61 | 718480.93 |
| `kimi_t4096` / bd4 | 164801.80 | 17747.04 | 301.27 | 116.27 | 182976.53 |
| `kimi_t4096` / bd28 | 24379.79 | 2557.84 | 77.89 | 122.80 | 27142.44 |
| `long_context` / bd4 | 171507.60 | 8884.02 | 300.45 | 118.58 | 180820.20 |
| `long_context` / bd28 | 24967.29 | 1291.45 | 77.93 | 122.54 | 26463.54 |
| `kimi_linear_layer` / bd4 | 41429.32 | 4382.22 | 76.93 | 23.70 | 45915.38 |

对应的管线占比（`aiv_vec_ratio`）：

| 配置 | gate 反向 | norm 反向 | beta 反向 | gate_reduce |
|---|---:|---:|---:|---:|
| `kimi_t4096` / bd1 | 0.996 | 0.940 | 0.960 | 0.502 |
| `kimi_t4096` / bd4 | 0.996 | 0.940 | 0.952 | 0.530 |
| `kimi_t4096` / bd28 | 0.996 | 0.937 | 0.924 | 0.480 |
| `long_context` / bd4 | 0.996 | 0.939 | 0.953 | 0.521 |
| `long_context` / bd28 | 0.996 | 0.936 | 0.924 | 0.467 |
| `kimi_linear_layer` / bd4 | 0.997 | 0.956 | 0.938 | 0.726 |

三个重 kernel 在**所有** 6 个配置上都过线；`gate_reduce` 在所有配置上都不过线
（`aiv_vec_ratio` 0.467~0.726），`long_context`/bd28 那一档它甚至变成搬运侧更高
（`aiv_mte2_ratio` 0.489 对 `aiv_vec_ratio` 0.467）——
两边都低，就是个没喂饱的小 kernel。它和 beta 前向辅助合计 126.41µs，
占 bd4 那 182976.53µs 的 **0.069%**。**把它们优化到 85% 对墙钟的影响在噪声以下**，
所以本任务不建议动；这不是"没报告问题所以达标"，是量过之后判定不值得。

### 2.3 前向：三个 kernel 全部不过线

| 配置 | kernel | 次数 | 设备 µs | 瓶颈 | 占比 | ≥85% |
|---|---|---:|---:|---|---:|:--:|
| `kimi_t4096` / bd4 | `KdaPrepChunkNormBf16Bf16Kernel` | 2 | 1640.56 | 搬运 (mte2 0.575) | 0.5750 | ❌ |
| `kimi_t4096` / bd4 | `KdaPrepChunkGateBf16F32F32Kernel` | 1 | 1024.58 | 计算 (vec 0.524) | 0.5240 | ❌ |
| `kimi_t4096` / bd4 | `KdaPrepChunkBetaBf16Kernel` | 1 | 9.47 | 搬运 (mte2 0.731) | 0.7310 | ❌ |
| `long_context` / bd4 | `KdaPrepChunkGateBf16F32F32Kernel` | 1 | 1014.80 | 计算 (vec 0.528) | 0.5280 | ❌ |
| `long_context` / bd4 | `KdaPrepChunkNormBf16Bf16Kernel` | 2 | 811.37 | 搬运 (mte2 0.5515) | 0.5515 | ❌ |
| `long_context` / bd4 | `KdaPrepChunkBetaBf16Kernel` | 1 | 9.58 | 搬运 (mte2 0.728) | 0.7280 | ❌ |

前向 norm 是**搬运 bound 且只到 0.575**：它读 bf16、写 bf16、每 128 通道只做一次
平方和与一次除法，算术密度本来就低。前向 gate 是**计算 bound 但只到 0.524**，
与反向 gate 同一个 `_dd_*` 家族却差一倍——前向只算 softplus 一次，没有反向那套
补偿 exp + 20 项 log 多项式 + 8 次补偿平方。

**但前向 prep 整体只有 2.70ms**（bd4 同步墙钟），对比反向的 190.83ms。所以
"前向三个 kernel 都没到 85%" 是真的、要如实记，而它的绝对权重是反向的 1.4%。

#### 一个必须说明的 capture 缺陷

`KdaPrepChunkNormBf16Bf16Kernel` 在 **bd≥16 的 capture 里记录不全**：bd4 两次 launch
都在 CSV 里，bd16 只记到 1 次，bd28 **3 份独立 capture 全部 0 次**。
同一进程的正确性检查通过、`q`/`k` 的 relative-L2 与 bd4 逐位一致，所以 kernel
**确实执行了**，是 profiler 的记录丢了。因此本文**不给**该 kernel 在 bd≥16 的
`PipeUtilization`，前向的 `block_dim` 伸缩改用同步墙钟（§4）。
如实记为"profiler 记录丢失"，不编机制。


## 3. 85% 线在这个 kernel 上失去判别力（本任务最重要的发现）

把 §2.2 的两列并排看：

| `block_dim` | `GetVecNum()` | gate 反向设备 µs | `aiv_vec_ratio` | 按 85% 线 | 输出 |
|---:|---:|---:|---:|:--:|---|
| 1 | 2 | 645813.07 | **0.996** | ✅ 过线 | 逐位相同 |
| 4 | 8 | 164801.80 | **0.996** | ✅ 过线 | 逐位相同 |
| 28 | 56 | 24379.79 | **0.996** | ✅ 过线 | 逐位相同 |

**同一份源码、同一个形状、逐位相同的输出，设备耗时差 26.5 倍，占用率读数一位都不动。**

原因不难，但正是这条线的结构性盲区：`aiv_vec_ratio` 是**每个在用核**上向量管线的
忙闲比，分母只含**被这次 launch 实际启动的核**。`kda_prep` 的 kernel 内部用
`GetVecIdx()/GetVecNum()` 自己切工作项，而 `GetVecNum() == 2 * block_dim`。
bd=4 时只启动 8 个向量核，剩下 48 个（Ascend950PR 物理 56 个向量核）**完全空闲**，
而这 8 个核确实 99.6% 时间在跑向量指令——于是读数是 0.996，"达标"。

> 核数的来源要说清：`GetVecNum() == 2 * block_dim` 与"Ascend950PR 物理 28 cube /
> 56 vec"两条都**引自 AGENTS.md §6/§5 已确立的结论，本任务没有从硬件寄存器读核数**。
> 本任务提供的是与之一致的独立实测旁证：§4.1 里 bd1→bd28 的加速是 **26.361×**
> （若关系成立则核数比正好 2→56 = 28×，实测伸缩效率 94.1%），而在 bd=28 之后
> 没有继续试（§8）。这一列当作"按该关系推出的核数"读，不是直接测量值。

这跟 DR-P1 今天测到的是同一类问题，但方向和幅度都更坏：

- DR-P1：找到一个逐位相同、慢 1.13~1.69 倍、占用率反而更高的配置。
- 这里：占用率**完全相同**（0.996 三处一致），真机耗时差 **26.5 倍**，
  而且"过线"的那一档恰好是**最慢**的可用档之一。

**方法论后果**：`aiv_vec_ratio`/`aic_mac_ratio` 达到 85% 只能说明
"**已启动的那些核**上那条管线喂饱了"，不能说明"**这块芯片**喂饱了"。
要让这条线有判别力，必须同时看一个"用了多少核"的量。本次 capture 的
`kernel_details.csv` 列里没有直接的核数列（可用列见 §1 的 receipts），
最可靠的办法是**把 `block_dim` 当自变量扫一遍、看墙钟是否还在降**——
降得动就说明没到物理上限，不管占用率读数多高。BF-09 给出的
"profiler 条件下界 164.428501ms" 本身是对的，但它**只是 `block_dim=4` 的下界**，
不是这个 kernel 的下界；本任务在 bd=28 上实测到 24.38ms。


## 4. `block_dim` 伸缩：实测

### 4.1 反向，同卡串行（card 2），`kimi_t4096`，7 轮同步墙钟

| `block_dim` | 向量核 | 中位数 ms | min / max ms | 对 bd1 | 理想 | 伸缩效率 | 对 bd4 |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 2 | 713.722 | 713.648 / 713.804 | 1.000× | 1× | — | 0.267× |
| 2 | 4 | 379.038 | 379.026 / 379.072 | 1.883× | 2× | 94.2% | 0.503× |
| 3 | 6 | 248.490 | 248.390 / 248.728 | 2.872× | 3× | 95.7% | 0.768× |
| 4 | 8 | 190.834 | 190.786 / 190.922 | 3.741× | 4× | 93.5% | 1.000× |
| 8 | 16 | 92.209 | 92.170 / 92.215 | 7.740× | 8× | 96.8% | 2.069× |
| 16 | 32 | 46.344 | 46.336 / 46.371 | 15.400× | 16× | 96.2% | 4.118× |
| 28 | 56 | 27.076 | 27.060 / 27.094 | 26.361× | 28× | 94.1% | **7.047×** |

样本极稳（每档 min/max 相对中位数 ≤0.1%）。**伸缩到声明上限 4 之后仍然是线性的，
一路线性到 28。** bd4 → bd28 是 **7.047×**（核数比正好 7×；略微超过是同卡噪声，
不声称超线性）。

### 4.2 前向，同卡串行（card 5），`kimi_t4096`，7 轮

| `block_dim` | 中位数 ms | min / max ms | 对 bd4 |
|---:|---:|---|---:|
| 4 | 2.7023 | 2.7003 / 2.7177 | 1.000× |
| 8 | 1.4034 | 1.3953 / 1.4113 | 1.925× |
| 16 | 0.7493 | 0.7447 / 0.7598 | 3.607× |
| 28 | 0.4514 | 0.4498 / 0.4671 | **5.986×** |

前向也线性，效率略低（5.99× 对 7×），与 §2.3 "前向是搬运 bound、算术密度低"一致。

### 4.3 逐位相同

7 个 `block_dim`（1/2/3/4/8/16/28）的 6 个梯度
（`dq`/`dk`/`dg`/`dA_log`/`ddt_bias`/`dbeta`）**逐位相同**：
每档的 SHA-256 集合的 md5 都是 `a58e2e32c4a06ca68a72…`。
同一组哈希在**另外 6 张卡**上独立跑出来的一轮并行预热里也一致，
所以不是同卡复用缓存造成的假相同。

对 FP64 解析参考的 relative-L2 在 7 个档上**完全相同**（`dq` 1.655621e-03、
`dk` 1.656255e-03、`dg` 1.657792e-03、`dA_log` 1.936679e-08、
`ddt_bias` 3.588245e-08、`dbeta` 1.657175e-03）。

这是结构上就该成立的：`gate_backward` 的工作项按 `work ∈ [0, HV*chunks)` 切，
每个工作项只读自己那 32 行、只写自己那 32 行 `destination` 和自己那段
`partials[work*512 : work*512+512]`，kernel 内那 5 层固定归约发生在工作项**内部**
（UB 上的 `terms`），跨工作项的归约交给第二个 kernel `gate_reduce`。
所以输出是工作项的纯函数，与工作项怎么分配到核上无关。
`norm_backward`/`beta_backward`（按 4096 元素切）、`gate_reduce`
（按 `CeilDiv(HV, heads_per_work)` 切）同理。**实测与这个结构判断一致。**

### 4.4 双 oracle 与冻结预算（bd4，`kimi_t4096` 真实形状）

| 梯度 | 预算组 | 对 FP64 解析 relL2 | 预算 | 占预算 | >1ULP | 对 torch autograd relL2 | 判定 |
|---|---|---:|---:|---:|---:|---:|:--:|
| `dq` | `norm:bf16:dx` | 1.655621e-03 | 5.087975e-03 | 0.3254 | 0 | 2.083e-05 | ✅ |
| `dk` | `norm:bf16:dx` | 1.656255e-03 | 5.087975e-03 | 0.3255 | 0 | 1.781e-05 | ✅ |
| `dg` | `gate:bf16_f32_f32:dg` | 1.657792e-03 | 5.006649e-03 | 0.3311 | 0 | 1.638e-05 | ✅ |
| `dA_log` | `gate:bf16_f32_f32:dA_log` | 1.936679e-08 | 2.584491e-06 | 0.0075 | 0 | 2.045e-06 | ✅ |
| `ddt_bias` | `gate:bf16_f32_f32:ddt_bias` | 3.588245e-08 | 4.215515e-07 | 0.0851 | 0 | 1.484e-07 | ✅ |
| `dbeta` | `beta:bf16:dbeta` | 1.657175e-03 | 4.990506e-03 | 0.3321 | 0 | 0.000e+00 | ✅ |

两个 oracle：① `ref/backward_calibrate.py` 的独立 FP64 解析导数；
② torch autograd（CPU，FP32，另一条实现路径）。两者一致，最差占预算 0.3321。
预算取自冻结的 `backward_budgets.json`，**没有放宽任何一项**。

**注意这组数字落在模型真实会用的形状上**（B1/T4096/HV32），不是 H=1/C≤2 的玩具形状——
这正是 AGENTS.md §6"性能在真实形状上测，精度也必须在真实形状上测"要求的那一格。


## 5. 瓶颈定位，以及为什么本任务不改生产代码

按 §6.5"先定位到确切的原因，不是症状规律"：

**确切原因**：`kda_prep` 的 `block_dim` 上限不是这些 kernel 的限制，是**声明限制**，
而且是一条**重复声明**。

- `kernels/projects/a5/kda_prep/runtime.py` 有自己的一份
  `_BLOCK_DIMS = {'chunk': (1,2,3,4), 'decode': (1,2,4,8,16,28)}`。
- `ascend_fla/platform.py` 的 `_A5_CAPABILITY["supported_block_dim"]` 是同样的两组值。
- `ascend_fla/ops/kda/autograd.py` 把**主链的** `block_dim` 原样传给
  `runtime.prepare_backward()` 与 `_RawGate.apply(...)`，而主链的
  `SUPPORTED_BLOCK_DIM` 来自 `platform` 的 `chunk` 那一组。

于是：`kda_prep` 的 gate 反向 kernel 被**另一族 kernel 的契约**限制在 8 个向量核上。
`gaps.json` 的 `block-dim-ceiling`（P1）记的是上游 ascriptor `kda_fwd`/`kda_bwd`
的 `domain.block_dim` 上限 4，而那是 AGENTS.md §3 规定只读的上游代码。
**`kda_prep` 不是上游单元，它是本仓自建的（`kernels/projects/a5/kda_prep/`）**，
所以它这一半的天花板**不需要动 ascriptor 就能抬**。

还有一条现成的旁证，但要说准它的范围：**同一个单元的 decode namespace 已经在
生产里跑到 28**——`ascend_fla/ops/kda/fused_recurrent.py` 的 `SUPPORTED_BLOCK_DIM`
取 `capability("a5")["supported_block_dim"]["decode"]`（含 28），并以
`namespace="decode"` 调同一个 `_prep_runtime()`，用的是 `make_norm`/`make_gate`/`make_beta`
**同样的工厂**。
**但 decode 是只前向的（T≤16），所以这条先例只覆盖前向三个 prep kernel。**
反向四个 kernel 在 bd>4 上没有生产先例，它们的证据就是本任务 §4 的实测
（逐位相同 + 预算内 + 线性伸缩），加上 §4.3 那条结构论证（工作项之间无依赖、
跨工作项归约在第二个 kernel 里）。两者要分开记，不能用 decode 的先例替反向背书。

**为什么本任务仍然不改生产代码**：

把这 7 倍真正拿到手，需要 `ascend_fla/ops/kda/autograd.py` 给 prep 传一个独立于
主链的 `block_dim`（以及 `ascend_fla/platform.py` 声明它）。这两个文件**都不在
BF-09-P1 的写集里**，`ops/kda/*.py` 更是 A2-15 正在改的写集，任务文件明确要求不碰。

只改 `kda_prep/runtime.py` 的 `_BLOCK_DIMS['chunk']` 是**无效的半步**：
放宽之后没有任何调用方能走到那些值（ops 层先拒），等于在声明里写上一个
"支持但不可达"的域，还会让这个单元的声明与 `platform.py` 不一致。
按 §6.5"记账 + 不静默出错 + 攒够一批统一修"的流程，正确做法是**记账 + 装守卫 + 上报**，
不是零散改一笔。

### 本任务唯一的改动（测试守卫，不碰生产代码）

`tests/test_kda_prep_backward.py` 加
`test_prep_block_dim_ceiling_tracks_the_platform_capability`：把
`kda_prep/runtime.py` 的 `_BLOCK_DIMS['chunk']` / `['decode']` 钉在
`platform.supported_block_dim('a5', ...)` 上。

动机是 §6.5 第 2 步。此前 `tests/test_kda_gating.py` 把 `ops/kda/chunk.py` 的
`SUPPORTED_BLOCK_DIM` 钉在上游契约上、`tests/test_platform.py` 把 platform 钉在 ops 上，
但**没有任何东西**把 `kda_prep` 自己那份重复字面量钉住。于是下一个人抬 ops 层天花板时，
prep 单元会被静默落在后面（反过来也一样）。现在哪一边先动都会撞上这条测试，
测试信息直接指向另一边和本文件。

负对照：把 `_BLOCK_DIMS['chunk']` 临时改成 `(1,2,3,4,8,16,28)` 后该测试
如期失败（`assert (1, 2, 3, 4, 8, 16, ...) == (1, 2, 3, 4)`），随后文件按字节还原。
`tests/test_kda_prep_backward.py` 全量：**102 passed / 3 skipped**（新增 2 个参数化用例，
无回归）。没有 CCE 发射改动，所以没有发射 diff 要贴。


## 6. FP32 cube matmul 与 HF32（P0 `a5-fp32-cube-matmul-inherits-hf32`）

**结论：`kda_prep` 完全没有 cube matmul，不暴露于这条 P0，不需要任何 `set_hf32` 守卫。**

三路独立证据，其中第一路是任务要求的真机 A/B（不是假设）：

### 6.1 真机跨进程 A/B（带有效正对照）

这条 P0 的威胁模型是"**别的进程**在这块卡的核上留下的舍入位"，所以两半必须是
**同一张卡上的不同进程**（口径同 DR-P1 的探针）。card 6、bd4，顺序执行五个进程：

```
chain(基线) → setter(allow_hf32=True) → chain(疑似) → setter_off(allow_hf32=False) → chain(钉回)
```

`kda_prep` 的 **8 个输出**（3 前向 `fwd_norm`/`fwd_gate`/`fwd_beta`
+ 5 反向 `bwd_norm`/`bwd_gate_dg`/`bwd_gate_dA_log`/`bwd_gate_ddt_bias`/`bwd_beta`）
在**三次 chain 里逐位完全相同**：

| 输出 | 三次 chain 的 SHA-256 |
|---|---|
| `fwd_norm` | `2c24188f1032d5efddd44c8da4303f1dffa757b10d59599edf699eca1cb91b44` |
| `fwd_gate` | `71c0ba02edd6c802b98147c0cf2940941fa771cce0d088e72de15c3ab3bef64d` |
| `fwd_beta` | `5e64f2ac6b3ef382352e9e68b55394ae9f5626f356c78ec2a7c67637199e1f65` |
| `bwd_norm` | `94efe941c7f2676f2516f6975b1b18b32f0300bceee4be5b2c0ac1fec8d0d5a6` |
| `bwd_gate_dg` | `72466791c1682f32b8619da6732089fdcce2da5d8828373c40cf5247bc718b1b` |
| `bwd_gate_dA_log` | `356b11bffef574019903ceea6c82e9e965126c96949963eb5547f34ac53da185` |
| `bwd_gate_ddt_bias` | `0411c803f7230fb178c9dfd6f4fd0ad2ec769e750387690dc0f4c269fb984b88` |
| `bwd_beta` | `0c35ca086bd3c017bf641bac1601b39c94e151a6e62650b603dc79a3adbe8dad` |

**正对照成立**：两个 setter 进程自己那个 4096×4096 FP32 matmul 的哈希**确实变了**——
`allow_hf32=True` 给 `b23e462ed3bc14ad84039b37325d8bdf517029b055f15d3a3fe1a08ecb06c7c2`，
`allow_hf32=False` 给 `15b9ccc3434f37f590fc10699edcca03bda4cf9077a46dbb54905a698a8c3171`。
也就是说这张卡上那个位**真的被翻动过**，而 `kda_prep` 的输出对它毫无反应。
（如果没有这个正对照，"哈希没变"也可能只是因为 setter 根本没生效——
HF-01 的教训就是暴露面不能靠想当然判定。）

### 6.2 硬件计数器

本任务 13 份带 `per_kernel` 的 capture（3 个形状 × bd1/4/16/28 × 前向/反向）里，
**每一个** `kda_prep` kernel 的 `aic_mac_ratio`、`aic_scalar_ratio`、`aic_mte1_ratio`、
`aic_mte2_ratio`、`aic_mte3_ratio`、`aic_fixpipe_ratio` 与 `cube_utilization(%)`
**全部为 0.000**——而且这是**逐次 launch 取 max** 之后仍为 0.000，不只是均值为 0
（均值为 0 也可能是少数非零被平均掉；机械检查输出 `NONZERO aic/cube entries: NONE`）。
cube 侧完全没有被触及——口径与 GDN-P1 判定 GDN 那条线不需要 HF32 守卫时用的一致。

### 6.3 源码

`kernels/forward.py`（217 行）与 `kernels/backward.py`（777 行）里
`matmul` / `mmad` / `splitk` / `set_hf32` 的出现次数都是 **0**。
7 个 kernel 全是 `@kernel(mode='vec')`，计算体是 `@vf()` 向量函数
（`_dd_add`/`_dd_mul`/`_dd_div`/`_dd_exp_negative`/`_dd_softplus_sigmoid` 这套
double-double 算术 + `cadd`/`cmax`/`gather`/`vxor` 这类向量原语）。
`backward.py` 里两处出现 "FP32" 字样（L78 注释讲 fused product residual、
L274 注释讲 D-PM-55 的 FP32 溢出边界）都与 cube 无关。

> 这一条单独不足以下结论——HF-01 的更正正是"靠 grep `matmul(` 判暴露面"
> 犯的错（把 BF16 操作数的 `wy.py` 误判为暴露、把零暴露的单元误判为全暴露）。
> 所以这里以 §6.1 的真机 A/B 为主、§6.2 的硬件计数器为辅，源码只做一致性印证。


## 7. 需要 PM / 用户裁定的事

1. **要不要把 `kda_prep` 的 chunk `block_dim` 天花板抬到 28。**
   实测：反向 7.047×、前向 5.986×（真实形状、同卡、7 轮、逐位相同、预算内）。
   代价：写集要扩到 `ascend_fla/ops/kda/autograd.py`（**A2-15 的写集，当前冲突**）
   与 `ascend_fla/platform.py`，再加 `kda_prep/contract.json` 的 domain 与
   `tests/test_kda_prep_backward.py`、`tests/test_platform.py` 的声明。
   按 D-PM-33 属于扩写集，本任务没有自行决定。
   需要注意两点：
   - prep 的 `block_dim` 与主 KDA 链的是**同一个参数**。要么给 prep 单独一个参数
     （ABI 变更，调用方可见），要么等上游把 `kda_fwd`/`kda_bwd` 的契约一起抬
     （那就回到 `block_dim-ceiling` 那条 P1，要走 ascriptor 侧流程或建派生单元）。
     **这是个 ABI / 架构选择，不是性能调参。**
   - 本任务只在三个真实形状上验证了 bd>4。要把 28 写进声明域，按 §7
     应当把 `contract.json` 的 case 矩阵（含 (C, HV) 的奇偶边界，§6）在新的
     `block_dim` 上过一遍，不能只凭真实形状这一格。
2. **`gaps.json` 要不要加一条**：`block-dim-ceiling` 现在只讲上游
   `kda_fwd`/`kda_bwd`，没提"本仓自建的 `kda_prep` 也被同一条天花板连带限住，
   而它这一半不需要改 ascriptor"。`docs/matrix/gaps.json` 不在本任务写集里，
   所以没有代改。
3. **85% 验收线本身要不要补一个"用了多少核"的维度**（§3）。
   今天 KDA-P1 / GDN-P1 / DR-P1 加上本任务，已经是**四个**独立任务各自撞到
   "占用率达标但不是最优"。本任务这一例是其中最极端的：占用率读数三处完全相同，
   真机耗时差 26.5 倍。这条线的措辞是用户 2026-10-27 直接给的，怎么改要用户定。


## 8. 复核入口

原始回执（`pipe.json`：含每次 launch 的 `kernel_details.csv` 全部行、
`PipeUtilization` 全部列、正确性度量、源码与 harness 哈希、环境、
`block_dim_gate_widened_for_measurement` 标记）留在真机任务目录下，
位置见 `machine_specs.md`，未提交（§10：不提交 profiling trace 与构建产物）。

共 **36 个进程**，一个进程一份独立 receipt：

| 组 | 进程数 | 内容 |
|---|---:|---|
| profiler capture | 13 | §2 的全部 `PipeUtilization`（机械检查：13 份都有 `per_kernel`） |
| 同步墙钟 bench | 17 | §4 的 card2 反向 7 档 + card5 前向 4 档 + 另外 6 张卡的并行预热（用于 §4.3 的跨卡逐位复核） |
| HF32 跨进程探针 | 5 | §6.1 的 chain / setter / chain / setter_off / chain |
| 正确性冒烟（不开 profiler） | 1 | 首轮 bd4 |

其中 bd1 的并行预热被我自己的占用守卫**拒绝启动**（card 1 在健康检查与启动之间来了
别人的进程，rc=73），bd1 的数据改在 card 2 的串行队列里取；另有一次 bd1 编译在
7 份并发编译下撞到 CMake 依赖文件的瞬时错误（`opening dependency file … No such
file or directory`），清掉 build 目录重试即过——**如实记为瞬时构建错误，不是 kernel 问题**。

本文所有 `block_dim>4` 的测量都带 `block_dim_gate_widened_for_measurement: true` 标记：
那是**测量工具**在自己进程里放宽 `kda_prep/runtime.py` 的 `_BLOCK_DIMS['chunk']`，
**没有改仓库源码**（`runtime.py` 的 SHA-256 在所有 25 个进程里都是
`981feab5813c9330fefd10001f23c85c8c53d298b4509d6b1f03f7b1d577a36a`）。

未做、如实记：

- **没有测 `block_dim > 28`**。AGENTS.md §5 说超过物理核数会在硬件 barrier 上死锁，
  共享卡上不试；28 已经是 56/56 向量核。
- **没有重测 BF-09 的完整训练步墙钟**（428.86 → 223.06ms 那条）。本任务量的是
  `kda_prep` 单元自身，不是整个训练步；上面的 7.047× 是 prep 反向链的加速，
  **不能直接当成训练步的加速**，整步里还有 `kda_bwd_stable` 等别的 kernel。
- **`gate_reduce` 与 beta 前向辅助 kernel 没有优化**（§2.2，合计占 0.069%）。
- **前向三个 kernel 没有优化**（§2.3，整体 2.70ms 对反向 190.83ms）。
