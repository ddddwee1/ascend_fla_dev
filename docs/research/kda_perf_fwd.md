# KDA-P1 · `a5.kda_fwd_stable` 的真机管线利用率

本文只报真机实测的 `PipeUtilization`，不报模拟器 `model_cycles`。形状全部取自
`docs/matrix/models.json` 的 `test_case_shapes`，不是契约的玩具形状。

## 按验收线（对应资源 `PipeUtilization` ≥85%）的逐 kernel 结论

本单元四个 kernel，**两个已达标、两个没达标**：

**报区间不报单点。** 同一个 kernel 的耗时在多次采集间很稳（例如 `wy` 136.65~146.50 µs），
但管线占空比有 ±0.05 量级的采集噪声，所以下面给**六次 `kimi_linear_layer` / bd4 采集
加上 bd1 与 `long_context` 各一次**的实测区间，单点数值在后面的明细表里。

| kernel | 绑定资源 | 实测区间（kimi bd4，6 次） | bd1 / long_context | 达标 | 本轮动作 |
|---|---|--:|--:|---|---|
| `kda_sub1_gate_stable_kernel` | 搬运 `aiv_mte3` | **0.954 ~ 0.966** | 0.989 / 0.984 | **是** | 不动（已到位） |
| `kda_sub2_score_stable_kernel` | 计算 `aic_mac` | **0.980 ~ 0.983** | 0.994 / 0.993 | **是** | 只钉 HF32 位（见下，不改算术） |
| `kda_sub3_wy_stable_kernel` | `aic_fixpipe` | 0.623 ~ 0.716 | `aiv_mte2` 0.584 / 0.605 | 否，差 **13 ~ 23** 点 | 未改 —— 定位到位、改法给出，而且算过一遍发现"显而易见的那条"不是净收益 |
| `kda_sub45_aqk_repaired_kernel` | 搬运 `aiv_mte3` | 0.619 ~ 0.639 | 0.633 / 0.660 | 否，差 **19 ~ 23** 点 | 未改 —— 同上 |

两点口径说明：

- `scores` 那一行**只在 HF32 位被钉成 off 时成立**。同一个 kernel 在 HF32=on 的进程里
  量到的是 `aic_mac` 0.227~0.229、最高管线变成 `aiv_mte3` 0.877，结论会反过来。
  见下面「HF32」一节 —— 这是本轮必须先钉住那一位的直接原因。
- "不动"不是省事：`gate` 与 `scores` 的最高管线在两个真实形状、两个 `block_dim`、
  六次采集上都 ≥0.95，**没有可挤的空隙**，按 AGENTS.md §6「profile 之前不要相信任何
  性能推断」的反面 —— profile 之后也不要制造没有依据的改动 —— 就该如实报数。
  （另见本文末尾对 `G3` 那句"高占比本身不证明吞吐最优"的保留。）

## 环境身份

| 项 | 值 |
|---|---|
| SoC | Ascend950PR（`npu-smi` 25.7.rc1，Health=OK，独占共享锁，启动前后卡上无外来进程） |
| CANN / OPP | **9.1.0**（`compiler/version.info` 与 `opp/version.info` 均 `Version=9.1.0`） |
| 内置算子包 | `opp/built-in/op_impl/ai_core/tbe/kernel/` 含 **`ascend950`**（另有 ascend910b、ascend910_93） |
| ascriptor | 0.1.0，library pin `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`；kernels 源码核查 pin `b3b3f9c16df7c4626ed3c081032a1be5a753d0b1` |
| Python / Torch | 3.12.14 / 2.12.0+cu130 / torch_npu 2.12.0 |
| 采集 | `torch_npu.profiler` Level1 + `AiCMetrics.PipeUtilization`，`wait=0/warmup=1/active=1/repeat=1`，**每进程只采一次**，先过正确性再采 |
| 一进程一 build | `ASCEND_FLA_CACHE` 按 build 分目录；每个 `block_dim` 一个进程（AGENTS.md §6 铁律二） |

正确性判据沿用 `kda_fwd_stable/contract.json` 的 `comparison`，**没有放宽**：
`rtol=atol=0.02`、`max_relative_l2=0.05`，对两个独立 CPU FP32 oracle
（`ascend_fla/reference/kda.py` 的 `kda_chunk_ref` 与 `kda_recurrent_ref`）各判一次。

## 基线：`kimi_linear_layer`（B1 / T1024 / H32 / HV32 / C16），`block_dim=4`

公共入口 `chunk_kda_fwd` 一次调用的全部设备 kernel，设备侧合计 **2501.75 µs**。

| # | kernel | µs | 占比 | 最高管线 | 值 | 判定 |
|--:|---|--:|--:|---|--:|---|
| 1 | `KdaLayoutBf16Bf16Kernel` | 964.75 | 38.6% | `aiv_mte2` | 0.752 | 搬运 bound，差 9.8 点 |
| 2 | `TrilInverse64V2StrictBf16Kernel` | 384.95 | 15.4% | `aiv_vec` | 0.444 | 未达标，差 40.6 点 |
| 3 | **`KdaSub2ScoreStableKernel`**（`intra.py`） | 325.75 | 13.0% | **`aic_mac`** | **0.981** | **计算 bound，达标** |
| 4 | **`KdaSub45AqkRepairedKernel`**（`recurrent.py`） | 321.40 | 12.8% | `aiv_mte3` | 0.623 | 未达标，差 22.7 点 |
| 5 | `KdaLayoutF32F32Kernel` | 238.02 | 9.5% | `aiv_mte2` | 0.785 | 搬运 bound，差 6.5 点 |
| 6 | **`KdaSub3WyStableKernel`**（`wy.py`） | 145.05 | 5.8% | `aic_fixpipe` | 0.671 | 未达标，差 17.9 点 |
| 7 | **`KdaSub1GateStableKernel`**（`gate.py`） | 51.50 | 2.1% | **`aiv_mte3`** | **0.957** | **搬运 bound，达标** |
| 8 | `aclnnCumsum`（门控跨度检查） | 48.58 | 1.9% | `aiv_vec` | 0.594 | 不是 kernel，是 host 侧检查 |
| 9 | `aclnnAmin`/`Amax`/`Max`/`Sub`（同上） | 21.76 | 0.9% | — | — | 同上 |

逐 kernel 的全部管线（均值）：

| kernel | `aic_mac` | `aiv_vec` | `aic_mte1` | `aic_mte2` | `aiv_mte2` | `aiv_mte3` | `aic_fixpipe` | `cube_util%` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `KdaSub1GateStable` | 0.000 | 0.345 | 0.002 | 0.002 | 0.477 | **0.957** | 0.004 | 0.18 |
| `KdaSub2ScoreStable` | **0.981** | 0.210 | 0.507 | 0.000 | 0.224 | 0.935 | 0.066 | 14.23 |
| `TrilInverse64V2Strict` | 0.409 | 0.444 | 0.113 | 0.000 | 0.165 | 0.307 | 0.421 | 13.73 |
| `KdaSub3WyStable` | 0.257 | 0.342 | 0.199 | 0.501 | 0.552 | 0.496 | **0.671** | 13.17 |
| `KdaSub45AqkRepaired` | 0.284 | 0.307 | 0.501 | 0.322 | 0.376 | **0.623** | 0.418 | 14.19 |
| `KdaLayoutBf16Bf16` | 0.000 | 0.000 | 0.000 | 0.000 | **0.752** | 0.233 | 0.000 | 0.00 |
| `KdaLayoutF32F32` | 0.000 | 0.000 | 0.000 | 0.000 | **0.785** | 0.178 | 0.000 | 0.00 |

正确性（同一进程，采集之前）：`o` 相对 L2 **3.962e-03**（对 `kda_chunk_ref`）/
**3.963e-03**（对 `kda_recurrent_ref`）；`final_state` **3.245e-03** / **3.245e-03**；
两个 oracle 自身一致到 3.904e-05。预算 0.05，`allclose(rtol=atol=0.02)` 通过。

## 同一组数在另外两个点上复算

`block_dim=1`（同形状，独立进程、独立 build）合计 **9294.42 µs**，bd1→bd4 **3.71×**
（与 `gaps.json` 的 `block-dim-ceiling` 记的 3.78× 同量级）。**每个 kernel 的管线比值
几乎不随 `block_dim` 变**，说明这些比值是 kernel 的固有性质，不是分区的偶然：

| kernel | `aic_mac` bd1 / bd4 | 最高管线 bd1 / bd4 |
|---|---|---|
| `KdaSub2ScoreStable` | 0.994 / 0.981 | `aic_mac` 0.994 / 0.981 |
| `KdaSub1GateStable` | — | `aiv_mte3` 0.989 / 0.957 |
| `KdaSub3WyStable` | 0.242 / 0.257 | `aic_fixpipe` 0.578 / 0.671 |
| `KdaSub45AqkRepaired` | 0.282 / 0.284 | `aiv_mte3` 0.633 / 0.623 |

`long_context`（B1 / T4096 / H16 / HV32 / C64），`block_dim=4`，合计 **9154.39 µs**：
`KdaSub2ScoreStable` `aic_mac` **0.993**、`KdaSub1GateStable` `aiv_mte3` **0.984**、
`KdaSub3WyStable` `aic_fixpipe` 0.550、`KdaSub45AqkRepaired` `aiv_mte3` 0.660。
`o` 相对 L2 4.169e-03、`final_state` 3.238e-03，预算内。
**两个达标的 kernel 在两个真实形状、两个 `block_dim` 上都达标。**

## 两个 kernel 已经到位，怎么读这件事

- **`gate.py`（95.7% / 98.4% `aiv_mte3`）是搬运 bound。** 它读 `g_raw`（64×128 FP32）
  写出 `g_cumsum` 与 `eg` 两份同尺寸 FP32 —— 写出的字节是读入的两倍，而算术只有一次
  `exp` 和一次加法。向量存储口打满是这个 kernel 的正确形态，不是缺陷。
- **`intra.py`（98.1% / 99.3% `aic_mac`）是计算 bound。** 它的两个 `matmul` 的 L1 操作数
  是 **FP32**（`l1_qg` / `l1_kbg` / `l1_kgneg` 都是 `DT.float`），`splitk=64`、
  `m=n=64`、`k=128`。FP32 cube 的 MAC 吞吐远低于 BF16，所以 MAC 口先满。
  注意它的 `aiv_mte3` 同时是 0.935 —— **向量存储口几乎同时打满**，那正是编译器
  perf lint 在 `intra.py:238-240` 指出的三处 `ub_to_l1.nd2nz`（每个 NZ fractal 列一次
  MTE3 burst，板上实测比 compact-NZ move 慢 10 倍，D-084）。两条口同时 >0.93，
  这个 kernel 没有可用的空隙。

### 一条必须说清楚的口径

`aic_mac_ratio` 是**单核**的 MAC 口占空比，`cube_utilization(%)` 才是**核占用率**。
上表里所有 cube kernel 的 `cube_utilization` 都是 **13~14%** —— 因为 `block_dim=4`，
而 Ascend950PR 物理上有 **28 cube / 56 vector** 核（4/28 = 14.3%）。
**"`intra.py` 计算 bound" 说的是"它占的那 4 个核的 MAC 口打满了"，不是"这颗芯片
的算力用满了"。** 芯片级还有约 7 倍头寸，那是 `gaps.json` 的 `block-dim-ceiling`
（契约声明上限 4，实测到 4 仍线性），不是本任务能在 kernel 实现层解决的
（`SUPPORTED_BLOCK_DIM` 住在 `ascend_fla/platform.py`，不在本任务写集内）。
按用户给的验收线（对应资源的 `PipeUtilization` ≥85%）这两个 kernel 已经到位；
按"芯片算力用满"这条**没有**到位。两条都报，不混。

## HF32：既是 P0 正确性问题，也是性能测量的前提

`intra.py` 的两个 `matmul` 是**本任务三个单元里唯一**一处 FP32 操作数的 cube matmul。
逐处核对过（操作数的 `Position.L1` / `Position.L0A|B` 声明 dtype）：

| 单元 / 文件 | cube 乘法 | 操作数 dtype | HF32 暴露 |
|---|---|---|---|
| `kda_fwd_stable/kernels/intra.py` | 2 × `matmul` | **FP32** | **是** |
| `kda_fwd_stable/kernels/wy.py` | 2 × `matmul` | BF16 | 否 |
| `kda_fwd_stable/kernels/recurrent.py` | 4 × `mmad` | BF16 | 否 |
| `kda_fwd_stable/kernels/gate.py` | 无 | — | 否 |
| `kda_bwd_stable/kernels/inverse_mm.py` | 5 × `matmul` | BF16 | 否 |
| `kda_bwd_stable/kernels/finalize_pre.py` / `finalize_post.py` | 无 | — | 否 |
| `kda_fused_recurrent/kernels/step.py` | 无 | — | 否 |

> **这更正了 `gaps.json` 的 `a5-fp32-cube-matmul-inherits-hf32` 里一句过宽的话。**
> 那条的 impact 写的是"`kda_fwd_stable/kernels/{intra,wy}.py`、`kda_bwd_stable`、
> `kda_scan_diag` 全部暴露"。按操作数 dtype 逐处核对，**`wy.py` 与 `kda_bwd_stable`
> 不暴露** —— 它们的矩阵乘操作数全是 bf16，HF32 舍入位只管 FP32 cube 乘法。
> 暴露面在本任务范围内只有 `intra.py` 一处。（`kda_scan_diag` 不在本任务写集，未核。）
> 把暴露面说宽了的代价不是漏修，是下一个人要重查一遍 —— 和 AGENTS.md §6
> 「那个判断也要写下来」是同一条纪律。

> **与 HF-01 的对账（HF-01 已完成，结论直接采用，不重测）。** 两边独立得到同一个
> 结论：`kda_fwd_stable` 的暴露点就是 `intra.py` 的 L246/L247（`l1_qg` / `l1_kgneg` /
> `l1_kbg`，`splitk=64`），`kda_bwd_stable` 侧零处（HF-01 静态审计了 5 个 cube kernel，
> 本任务按操作数 dtype 核了 `inverse_mm` 的 5 个 `matmul` 与两个 finalize 的零个，
> 两边一致）。**HF-01 另外查出一处本任务没查的**：上游只读的
> `kda_fwd/kernels/triangular_inverse.py` 有 11 处 —— 见下面「意外的连带证据」，
> 本任务的实测数据正好能独立印证它，但按 AGENTS.md §3 不动那个文件。
>
> HF-01 点名未重测的几格（C=8 / C=16 / GQA / 长上下文），本任务顺手补上了，
> **全部在 HF32 位已确认为 off 的状态下测得**：`kimi_linear_layer`
> （C=16、H=HV=32、无 GQA）`o` 相对 L2 **3.962497e-03**、`final_state` **3.245049e-03**，
> 由 `hf32off` 与 `base` 两份 build 给出**逐位相同**的 SHA-256 所以状态可判；
> `long_context`（C=64、H=16/HV=32，即 GQA 2:1）`o` **4.168813e-03**、
> `final_state` **3.237635e-03**，该次采集的 `scores` `aic_mac` = 0.993 落在 FP32 档
> （HF32 档是 0.227~0.229），据此判为 off。layer 级仍未覆盖。
>
> HF-01 提到"改 `kda_fwd_stable` 的 FP32 cube 位点理论上会联动影响反向"。本轮实测给出
> 量级：位状态在 off/off 之间时前向输出**逐位相同**（零扰动）；位从 off 翻到 on 时前向
> `o` 的相对 L2 从 3.962497e-03 变 3.969099e-03（绝对动 6.6e-06，相对动 0.17%），
> 而六个梯度的 SHA-256 在跨越这次翻转的两次采集之间**没有变**
> （`cand1-bwd-kimi-bd4-r1` 在 `hf32on` 之前、`-r1b` 在它之后，`dq c923cd09…` 等六项全同），
> 与 HF-01 强制开关下"六个梯度逐位相同"的结论一致。所以是"远小于预算余量"，不是硬阻塞。

对性能的影响是独立于正确性的第二件事：**HF32 的 cube 吞吐与真 FP32 不同**，不钉住
这一位，`intra.py` 的耗时就随"这块核上一次跑了什么"变，profile 数字不可复现。
所以本任务在真机上做了 base / `set_hf32(False)` / `set_hf32(True)` 三份 build 的 A/B/C。

### 实测：同一张卡、同一个会话里**现场复现**了这个 P0

三份 build 各一个进程、各一份 `ASCEND_FLA_CACHE`，同一张卡、同一把设备锁，按下表的
**时间顺序**跑。`base` 是未改的 main（**完全没有** `set_hf32`），`hf32off` 是本轮的改动，
`hf32on` 是故意写成 `set_hf32(True)` 的**负控**（不是候选）。

| # | 时刻（UTC） | build | `o` SHA-256 前 16 位 | `o` 相对 L2 | `scores` µs | `aic_mac` | `aiv_mte3` | `aic_mte1` |
|--:|---|---|---|--:|--:|--:|--:|--:|
| 0 | 10:00 | `base`（未钉） | `ded5953ee7e3b477` | 3.962497e-03 | 325.75 | 0.981 | 0.935 | 0.507 |
| 1 | 10:05 | `base`（未钉） | `ded5953ee7e3b477` | 3.962497e-03 | 325.96 | 0.980 | 0.945 | 0.515 |
| 2 | 10:18 | **`hf32off`（钉 off）** | `ded5953ee7e3b477` | 3.962497e-03 | 324.78 | 0.983 | 0.952 | 0.512 |
| 3 | 10:30 | **`hf32on`（钉 on，负控）** | **`7590af999f19ebe2`** | 3.969099e-03 | **295.60** | **0.229** | 0.877 | 0.818 |
| 4 | 10:31 | **`base`（未钉）** | **`7590af999f19ebe2`** | 3.969099e-03 | **299.87** | **0.227** | 0.896 | 0.792 |
| 5 | 10:32 | **`hf32off`（钉 off）** | `ded5953ee7e3b477` | 3.962497e-03 | 325.69 | 0.981 | 0.953 | 0.516 |

读这张表：

1. **第 4 行是现场复现。** 未改的 `base` 在第 1 行给 FP32 的结果，在第 4 行给出
   **与 `hf32on` 逐位相同**的 `o` / `final_state` SHA-256 和几乎相同的管线画像 ——
   中间唯一发生的事是第 3 行那个 `set_hf32(True)` 的进程在这块核上跑过。
   **HF32 位被上一个进程留在了 on，下一个不钉它的进程就继承了。**
   PK-07 推断的跨进程继承机制，在这里由"自己的前一个进程"直接演示了一遍；
   换成别的租户的进程，效果完全一样，而我们看不见。
2. **`hf32off` 在第 2 行和第 5 行都给 FP32 的结果 —— 钉住之后对两个方向都免疫**，
   包括紧跟在 `hf32on` 后面的第 5 行。
3. **它是静默的，预算抓不到。** HF32 状态下 `o` 的相对 L2 是 3.969e-03，
   预算是 0.05 —— **过**；`allclose(rtol=atol=0.02)` 也**过**。
   在这个形状上，HF32 与 FP32 的差别只有 SHA-256 和管线画像能看见。
   （这正是 `gaps.json` 里"预算本身留了裕度，HF32 的降级未必每次都会把原本通过的
   case 顶出预算"那句话的实例。）
4. **钉住它不要钱。** `hf32off` 的 `scores` 耗时 324.78 / 325.69 µs 对
   未钉但处于 FP32 态的 `base` 325.75 / 325.96 µs —— 差在 0.4% 以内（run-to-run 噪声
   量级），`aic_mac` 0.983 / 0.981 对 0.981 / 0.980。IR op 数 276 → 278。

### 意外的连带证据：那一位管的不止 `intra.py`，`inverse` 也跟着动

`hf32on` 那份 build **唯一的源码差别是 `intra.py` 里 `set_hf32(True)`**。可是同一次采集里
**另一个** kernel —— 上游只读的 `TrilInverse64V2StrictBf16Kernel`
（`kda_fwd/kernels/triangular_inverse.py`）—— 的画像也跟着变了：

| 位状态 | run | `inverse` µs | `aic_mac` | `aic_fixpipe` |
|---|---|--:|--:|--:|
| off | `base` r1（任何 hf32 build 之前） | 357.87 | 0.425 | 0.420 |
| **off（钉住）** | `hf32off` r1 | 385.96 | 0.406 | 0.423 |
| **on（钉住）** | `hf32on` r1 | **318.92** | **0.312** | 0.499 |
| **on（继承来的）** | `base` r2 | **326.97** | **0.309** | 0.499 |
| **off（钉住）** | `hf32off` r2 | 370.51 | 0.421 | 0.424 |

三次 off 聚在 357.87~385.96 µs / `aic_mac` 0.406~0.425，两次 on 聚在 318.92~326.97 µs /
`aic_mac` 0.309~0.312 —— **分得很干净**。推论有两条：

1. **`triangular_inverse.py` 确实有 FP32 cube matmul 暴露。** 这是从时序与管线数据得到的
   独立印证，与 HF-01 静态审计出的"11 处"对上了（两种方法、同一结论）。
   它意味着这个 kernel 的精度同样一直暴露在共享卡的历史状态下。
2. **那一位是按核的控制位，跨 kernel 发射持续有效。** 前向链的启动顺序是
   gate（纯向量）→ scores(`intra`) → inverse → wy → recurrent，所以 `intra.py` 里钉的那
   一次，**顺带把它之后发射的全部 cube kernel 也钉住了** —— 本轮测到的 `inverse` 稳定
   落在 off 档就是这个的证据。

**但这不能当作 `triangular_inverse` 的修复。** 它依赖"`intra` 先于 `inverse` 发射"这个
**没有写进任何契约**的顺序：调度顺序一变、或者将来有带 FP32 cube 的 kernel 排到 `intra`
之前，这层连带保护就没了。HF-01 给的处置（建派生单元或走 ascriptor 侧流程）仍然是正解，
是否现在做由用户决定；本任务按 AGENTS.md §3 不动那个文件，只把这条可观测事实记下来。

### 一条比正确性更直接影响本任务的后果

**如果这张卡当时继承的是 HF32=on，本任务对 `scores` 的结论会是错的。** 看第 3/4 行：
HF32 下 `aic_mac_ratio` 掉到 **0.227**（因为 HF32 的 MAC 吞吐高，同样的乘法占满不了
MAC 口），最高管线变成 `aiv_mte3` 0.877~0.896 与 `aic_mte1` 0.792~0.818，耗时也少 8%。
按那组数会写下"`scores` 不是计算 bound，是 MTE3 搬运 bound，差 85% 线约 5 点"
—— 而真实情况是 `aic_mac` 0.98、**已经达标的计算 bound**。
**两个相反的工程结论，取决于别人的进程在这块核上留了什么，而预算一声不响。**
这就是把 `set_hf32(False)` 当作性能测量前提、而不是可选保险的理由。

## 未达标的三项与各自的卡点

### `recurrent.py` — 321.40 µs，`aiv_mte3` 0.623（差 22.7 点）

AIV 侧各口忙时之和 `0.307 + 0.376 + 0.623 + 0.159 = 1.465`、AIC 侧
`0.284 + 0.322 + 0.501 + 0.418 + 0.068 = 1.593`，**都大于 1，说明两侧内部的口已经
互相重叠得不错**，不是"完全串行"那种病。瓶颈是向量存储口：这个 kernel 每个
(head-pair, chunk) 要把 `h`（bf16 8KB）、`qg`（bf16 16KB）、`v_new`（bf16 8KB）
先在 UB 里写出来、再 `ub_to_l1` 搬进 L1 —— **同一批字节过两次 MTE3**，而其中的
`ub_to_l1` 正是 lint 标的 nd2nz 多 burst 形式（`gaps.json` 的
`kernel-nd2nz-suboptimal` 记了 recurrent.py 三处）。
要把 `aiv_mte3` 从 0.623 推到 0.85 有两条路，都**没有**在本轮做：
① 让 `@vf` 直接按 compact-NZ 写出 staging tile，`ub_to_l1` 就退化成单条 plain move
（lint 自带的改法，并要求给 staging tile 补一行 padding 让 block stride 变奇数）；
② 把 `state` 的 bf16 副本常驻 L1、不每 chunk 重新搬。
两条都要改这个 kernel 二十多个手写 `DEvent` 的配对关系，按 AGENTS.md §6.5
「动手修之前必须先定位到确切的原因」，本轮只给出定位和实测数，不照着规律瞎改。

### `wy.py` — 145.05 µs，`aic_fixpipe` 0.671（差 17.9 点）

最高口是 cube 侧的 FIX（L0C 读出）。这个 kernel 的两个结果 `w` / `u` 都是
`l0c_to_gm` 直出 GM，而同一个仓里 `kda_bwd_stable/kernels/inverse_mm.py:123-125`
已经写下过实测结论：**"FIX is L0C read bound, so `l0c_to_ub` (138cy) + MTE3 store
beats five `l0c_to_gm` stores (1324cy each)"**。照这条改，FIX 忙时能降一个数量级。

**但算了一遍发现这不一定是净收益，所以本轮没做。** 现在 145 µs 窗口里 FIX 忙
97 µs；改掉后 FIX 降到约 20 µs，可是那 2×16KB/item 的字节要改走向量 MTE3，而
`aiv_mte3` 现在已经忙 72 µs（`qg`/`kg` 的两次 store），再加等量字节就是 ~144 µs
—— 瓶颈从 FIX 平移到 MTE3，窗口几乎不变。**把搬运从一条口挪到另一条口不是优化。**
这条记在这里是为了下一个人不要照着 inverse_mm 的那句注释直接改 wy。

真正的空隙是另一处：`wy.py` 是一个**扁平循环**（load → vf → L1 → matmul → drain），
而同单元的 `intra.py` 用的是 `for pipe_work in range(work_begin, work_end + 2)` 的
**三段软流水**，`inverse_mm.py` 用 `work_end + 2` 的两段前瞻 —— 两个 lint 零告警的
高利用率 kernel 都有软流水，`wy.py` 没有。145 − 97 = 48 µs 的 FIX 空闲正是它的候选
来源。这是下一轮的首选改动（改法：按 `intra.py` 的形状把 vf 段与 cube 段错开一次
迭代，`l1_abeta` / `l1_kexp` 的 `vcmutex` 深度已经是 2，正好配两个轮转槽；算术、
GM 读写顺序、`matmul` 参数一概不动，**预期输出逐位相同**，可以直接用跨 build 的
SHA-256 判等来验证改动没有改数学）。

### 上游的两项（不在本任务写集）

- `TrilInverse64V2StrictBf16Kernel`（384.95 µs，`aiv_vec` 0.444）：ascriptor 仓的
  `kda_fwd/kernels/triangular_inverse.py`，按 AGENTS.md §3 只读。lint 在它上面报 8 处
  nd2nz（`kernel-nd2nz-suboptimal`）。
- `kda_layout` 的两个 kernel（`kimi_linear_layer` 合计 **1174.58~1202.77 µs = 48.1%**，
  `long_context` **4022.18 µs = 43.9%**；`aiv_mte2` 0.749~0.849）：
  纯布局转换，是 `kernels/projects/a5/kda_layout` 单元，不在本任务写集。
  **它们是前向最大的单项头寸**：接近一半的设备时间花在 token-major ↔ BHCLD 的搬运上，
  而且已经接近搬运 bound（75~85%），所以在"这一层"里没多少可挤的 ——
  要省只能去掉这一层本身，即让 kernel 直接吃 token-major 并在 kernel 内取头
  （与 `gaps.json` 的 `decode-call-overhead` 第 ② 条同一种改法，那是 ABI 级改动）。

### 门控跨度检查的设备成本（顺带量到的）

`chunk_kda_fwd` 默认开 `check_gate_range`，它在 device 上走 torch_npu 的
`cumsum`/`amin`/`amax`/`max`/`sub`。实测 `kimi_linear_layer` 69.66~70.34 µs（占 2.9%）、
`long_context` **281.44 µs（占 3.1%）**。这段不在任何 kernel 里，是 `ops/kda/chunk.py`
的 host 算术（`D-PM-42` 登记的例外之一，且该文件是 A2-15 的写集，本任务不碰）。
登记在这里是因为它和 kernel 时间混在同一个 profile 里，不说明就会被误读成 kernel 成本。

## 工具链那一项：`pypto-pro-op-orchestrator` 取到了、读了、没装

验收线要求"开始优化前先装最新 PyPTO-Pro 框架 + 官方插件
[`pypto-pro-op-orchestrator`](https://gitcode.com/cann/pypto-gym/tree/master/cannbot-skills/plugins-official/pypto-pro-op-orchestrator)"，
同时说"如果装不上或者发现它对这次任务不是必需的，如实记录下来，不要为了'装了'而硬套"。
如实记录如下。

`cann/pypto-gym` 从这台机器**可达**（`git ls-remote` 通，HEAD `d2604896`），已 sparse
clone 到任务工作区并读了 `quickstart.md`、`AGENTS.md`、六个 agent 定义与
`references/{kernel-core-principles,kernel-perf-pypto-pro-a5,performance-constraints,multi-agent-workflow}.md`。
**没有安装**，理由是适用面不对，不是装不上：

- 它的 `quickstart.md` 开门第一句就是"本插件用于**原生 PyPTO-Pro 算子**开发、验证、
  调试与性能优化"，开发内容是"PyPTO-Pro kernel + golden + test"。本任务的三个单元是
  **ascriptor / CCE** kernel（`from ascriptor.a5 import *`、`@kernel()` / `@vf()`），
  不是 PyPTO-Pro DSL。
- 它的门禁 `hooks/pypto-pro-op-lint` 按 PyPTO-Pro 的 API 形状（`make_tile_group`、
  `pl.set_validshape`、`auto_mutex` 等）做 AST 检查，对 ascriptor 源码没有可判项；
  装它的 D1–D4 检查会对本任务的文件全部空转。
- 安装入口是把六个 agent 与 hook 装进 **OpenCode** 宿主（`scripts/install_opencode.py`），
  那是换 agent 运行环境，不是本会话能在任务范围内完成的事。

**真正拿来用了的是它的方法论参考**，那部分与框架无关：
`references/kernel-perf-pypto-pro-a5.md` §2 的 `G1`–`G6` 自检清单当作"还有哪里没看"的
清单走了一遍，本文与 `kda_perf_bwd.md` / `kda_perf_decode.md` 的结论结构就是按它组织的 ——
`G1` 正确性前置（每次 profile 之前先过预算）、`G3`"`max(pipe_ratio) < 0.8` 时检查同步、
依赖链与启动/排空"（正是 `wy` / `recurrent` / 两个 finalize / decode T=1 的落点）、
`G4` 多核均衡（`finalize_post` 在 `B*C=16` 上 8 个 worker 整除，`η_tail = 1.0`，
所以 1.55× 的缺口不是尾部不均衡）、`M5`/`M6`（估计值要用可复现实测校准；没有证据路径的
按 `unknown` 计，不当已验证）。

### 读它的时候发现一条与验收线的张力，不自行处置

同一份参考在 `G` 表下面写着：**"这些阈值仅是诊断参考，可随平台、工作负载和计量口径
调整，不能作为验收或停止调优的依据"**；`G3` 还补了一句 **"高占比本身不证明吞吐最优
或已经重叠"**。而用户 2026-10-27 给的验收线正是把 `PipeUtilization` ≥85% 当作停止线。

本任务**按用户给的线执行**（用户指示优先），但把这条张力如实记下来，因为它影响怎么读
上面"已达标"那两格：`gate` 的 `aiv_mte3` 0.957 与 `scores` 的 `aic_mac` 0.981 **满足
用户给的停止线**；它们**不等于**"这两个 kernel 已到硬件吞吐上限" —— 后者还要 Roofline
对照（`G6`），本轮没做。要不要把插件那句话纳入验收线的表述，是用户的决定，不是 assignee 的。

## 复现

```sh
# 一进程一 build 一 block_dim；输出目录必须是新的（复用会留下两份 capture）
./run.sh <build> <tag> ./profile_kda.py --unit fwd --case kimi_linear_layer \
         --block-dim 4 --output out/<tag>
```

`profile_kda.py` 先对两个 CPU FP32 oracle 判预算，再做**唯一一次** Level1 +
`PipeUtilization` 采集；报告里带 harness 自身与四个单元全部 kernel 源码的 SHA-256。
原始 `kernel_details.csv`、逐 kernel 全列数值与运行日志留在任务的私有证据目录
（机器坐标只存在 git-ignored 的配置里，不进仓）。
