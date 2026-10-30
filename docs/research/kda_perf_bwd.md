# KDA-P1 · `a5.kda_bwd_stable` 的真机管线利用率

环境身份、采集口径、"一进程一 build 一 `block_dim`" 的纪律与
[`kda_perf_fwd.md`](kda_perf_fwd.md) 同一份，不重复。

## 按验收线（对应资源 `PipeUtilization` ≥85%）的逐 kernel 结论

本单元三个 kernel（反向链共九个，另外六个在上游单元里，本任务只读）：

| kernel | 绑定资源 | base | 本轮改后 | 达标 | 本轮动作 |
|---|---|--:|--:|---|---|
| `inverse_mm_bounded_kernel` | 搬运 `aic_mte2` | **0.941** | 0.948（未改动） | **是** | 不动（已到位） |
| `finalize_pre_stable_kernel` | `aiv_vec` | 0.535 | 0.523 | 否，差 32.7 点 | **改了**：墙上时间 **1.90×** |
| `finalize_post_stable_kernel` | `aiv_vec` → `aiv_mte2` | 0.571 | 0.536 | 否，差 31.4 点 | **改了**：墙上时间 **1.55×** |

**两个 finalize 变快了但没达标，这两件事都要说。** 原因是本轮解掉的是"一半向量核
在空转"（并行度），而 85% 这条线量的是**单核占空比**；后者由"取数与计算不重叠"
决定，那是第二处缺陷，UB 容量挡着，留给下一轮。详见下面 ① 与 ②。

正确性参考用 **`kda_bwd` 单元自己的 `ref.oracle.oracle`** —— 不是自拟参考：契约
`comparison` 的逐输出预算（默认 0.05、`dk` 0.15、`dg` 0.25）就是相对它定的，换参考
等于换判据。`tests/test_kda_bwd_npu.py` 用的也是它。巧的是这个单元的
`ref.types.REPRESENTATIVE_SHAPE` 本来就是 `{B1, T1024, H32, HV32, K=V=128}` ——
正是 `models.json` 的 `kimi_linear_layer`，所以**没有自拟任何输入分布**。

## 基线：`kimi_linear_layer`（B1 / T1024 / H32 / HV32 / C16），`block_dim=4`

`chunk_kda_fwd_with_caches` 造好九个检查点后调 `chunk_kda_bwd` 一次，设备侧合计
**2413.70 µs**。

| # | kernel | µs | 占比 | 最高管线 | 值 | 写集 | 判定 |
|--:|---|--:|--:|---|--:|---|---|
| 1 | `InverseEpilogueKernel` | 557.44 | 23.1% | `aiv_vec` | 0.596 | 上游 | 未达标，差 25.4 |
| 2 | `ScanFusedKernel` | 489.14 | 20.3% | `aic_mte2` | 0.406 | 上游 | 未达标，差 44.4 |
| 3 | **`FinalizePostStableKernel`** | 470.08 | 19.5% | `aiv_vec` | 0.571 | **本任务** | 未达标，差 27.9 |
| 4 | **`FinalizePreStableKernel`** | 241.89 | 10.0% | `aiv_vec` | 0.535 | **本任务** | 未达标，差 31.5 |
| 5 | **`InverseMmBoundedKernel`** | 180.13 | 7.5% | **`aic_mte2`** | **0.941** | **本任务** | **搬运 bound，达标** |
| 6 | `FinalizePairKernel` | 160.35 | 6.6% | **`aic_fixpipe`** | **0.984** | 上游 | **达标** |
| 7 | `FinalizeReduceKernel` | 132.30 | 5.5% | `aiv_mte2` | 0.713 | 上游 | 未达标，差 13.7 |
| 8 | `InverseDainvKernel` | 90.11 | 3.7% | **`aic_mte2`** | **0.980** | 上游 | **达标** |
| 9 | `InverseDakkFusedKernel` | 78.58 | 3.3% | `aic_mte2` | 0.829 | 上游 | 未达标，差 2.1 |
| — | `aclnnNeg`（host 侧 `dw` 取负） | 7.70 | 0.3% | — | — | `ops/kda` | 非 kernel |
| — | `KdaLayoutBf16Bf16Kernel` | 5.97 | 0.2% | — | — | 另一单元 | — |

本任务三个 kernel 的全部管线（均值）：

| kernel | `aic_mac` | `aiv_vec` | `aic_mte1` | `aic_mte2` | `aiv_mte2` | `aiv_mte3` | `aic_fixpipe` | `aiv_scalar` | `cube_util%` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `FinalizePreStable` | 0.000 | **0.535** | 0.001 | 0.001 | 0.441 | 0.272 | 0.002 | 0.060 | 0.07 |
| `FinalizePostStable` | 0.000 | **0.571** | 0.001 | 0.001 | 0.416 | 0.173 | 0.002 | 0.023 | 0.04 |
| `InverseMmBounded` | 0.554 | 0.235 | 0.518 | **0.941** | 0.001 | 0.582 | 0.827 | 0.121 | 14.18 |

正确性（采集之前，同一进程）：`dq` **3.368e-02**（预算 0.05）、`dk` **7.507e-02**（0.15）、
`dv` **3.054e-03**（0.05）、`dbeta` / `dg` / `dh0` 同样在各自预算内，六项全部有限且非零。

`block_dim=1` 复算（合计 **9389.93 µs**，bd1→bd4 **3.89×**）：
`FinalizePostStable` `aiv_vec` 0.582、`FinalizePreStable` 0.543、
`InverseMmBounded` `aic_mte2` 0.934 —— **比值不随 `block_dim` 变**，与前向同样的结论。

## `InverseMmBoundedKernel` 为什么已经到位

`aic_mte2 = 0.941`：它每个 (b, hv, c) 要把六块 GM 搬进 L1（`dv`/`do`/`v_new` 各 16KB、
`h`/`dh` 各 32KB、`Akk` 8KB = 120KB），然后做五个 BF16 `matmul`。cube 侧的 GM→L1 口
先满，这是搬运 bound 的正确形态。

它还顺带印证了一条本仓自己量过的结论：注释里写的
"FIX is L0C read bound, so `l0c_to_ub` (138cy) + MTE3 store beats five `l0c_to_gm`
stores (1324cy each)" —— 它把 `dqg`/`dvbeta`/`dkg` 三个结果走 `l0c_to_ub` + 向量 store
出去，只留 `d_vh` 与 `d_k_beta_g` 两个 `l0c_to_gm`，于是 `aic_fixpipe` 停在 0.827
而不是撞上限。**这条经验是本轮评估 `wy.py` 时的参照**（见 `kda_perf_fwd.md`）。

## 两个 finalize kernel 的卡点：半个向量引擎在空转，而且取数与计算完全串行

两件事，必须分开说，因为**它们各自解决其中一半，谁都不能单独把比值推到 85%**。

### ① 一半的向量核没在用（本轮改掉了）

`finalize_pre` 与 `finalize_post` 里**没有一条 `matmul`** —— `cube_utilization` 实测
0.07% / 0.04%，cube 核启动了但只跑 scalar 同步代码。可是它们的工作划分是：

```python
work_per_core = CeilDiv(work_count, GetCubeNum())      # 按 cube 核数切
...
if GetSubBlockIdx() == 0:                              # 然后让另一个向量子块空转
```

`GetVecNum() == 2 * GetCubeNum()`，所以 `block_dim=4` 时只有 4 个向量子块在干活，
另外 4 个整程空转。那个 `if GetSubBlockIdx() == 0` 的来历见两个文件的 docstring：
它是**正确性修复** —— 原来两个子块算同一份数据、重复写 GM。用"关掉一个"修对了，
但代价是一半吞吐。

本轮改成按 `GetVecNum()` / `GetVecIdx()` 切同一个 item 空间，**一个 worker 仍然整份
拥有一个 item**，所以那个重复写的隐患不可能回来：

- `finalize_pre` 的 item 是 `(b, hv, c)`，六个输出都是 BHCLD 布局下 ≥8KiB 的连续块，
  两个 worker 之间离"共用一个 32 字节 GM 写粒度"差好几个数量级。
- `finalize_post` 的 item 是 `(b, c)`，**一个 worker 拥有这个 chunk 的全部 HV 个头和
  全部 64 个 token**。这一条是关键：两个公共输出是 token-major 的，
  `dbeta_out[b, t, 0:HV]` 一行宽 `2*HV` 字节、`dg_out` 一行宽 `2*HV*D` 字节，只要整个
  `hv` 区间归一个 worker、且不同 worker 拿的是不同 chunk 的 token 区间，32 字节写粒度
  就不可能跨两个 worker。（这正是原注释 "All heads share scalar GM transaction blocks
  and stay on this core" 要保护的东西 —— 本轮**保留**了这个性质，没有按 `hv` 切。）
- `finalize_post_vf` 里那条按行**倒序**走的 `running`（chunk 内 `dg'` 的反向 cumsum）
  是串行的，本轮**没有碰那个循环**：worker 仍然整份处理一个 chunk 的 64 行。

算术、缓冲、读取 pitch、item 内的 store 顺序、`@vf` 的发射一概不动，所以**输出逐位
相同**，只有 worker 数翻倍。

#### 实测（`kimi_linear_layer`，`block_dim=4`，同一张卡，各自独立进程与独立 build）

| kernel | base µs | cand µs（两次） | 提速 | base 最高管线 | cand 最高管线 |
|---|--:|--:|--:|---|---|
| `FinalizePreStableKernel` | 241.89 | **124.96 / 130.29** | **1.90×** | `aiv_vec` 0.535 | `aiv_vec` 0.523 / 0.507 |
| `FinalizePostStableKernel` | 470.08 | **295.44 / 310.83** | **1.55×** | `aiv_vec` 0.571 | **`aiv_mte2` 0.536 / 0.551** |
| 其余七个 kernel | — | 逐项在 ±2% 内 | — | — | — |
| **反向设备时间合计** | **2413.70** | **2128.89 / 2143.18** | **1.13×** | | |

`finalize_pre` 拿到 **1.90×**，接近 worker 数翻倍的理论值 2×。
`finalize_post` 只有 **1.55×**，而且**瓶颈换了口**：从 `aiv_vec` 0.571 变成
`aiv_mte2` 0.536~0.551。原因是它的 item 空间只有 `B*C = 16` 个，`block_dim=4` 下
8 个 worker 各拿 2 个 —— 粒度粗、尾部不均衡，而每个 item 要读 32 个头 × 十个 16KiB
tile，取数口先顶上来。（item 空间的粒度本身是这个 kernel 的 ABI 性质：按 `hv` 再切
会碰上 token-major 公共输出的 32 字节写粒度，见上。）

整条反向链 **1.13×**（−284.8 µs / −270.5 µs），因为被改的两个 kernel 原本占 29.5%。

**验证证据，逐项：**

- **CCE 发射范围精确**：对两份 build 的 48 个 kernel 逐个 hash
  `project/op_kernel/` 下的全部 `.cpp` / `.h`（即 vendor 编译器看到的 CCE），
  **46 个逐字节相同，只有 `finalize_pre_stable_kernel` 与 `finalize_post_stable_kernel`
  这两个不同**（含 `kda_sub1/2/3/45`、`scan_fused`、`inverse_*`、`finalize_pair`、
  `finalize_reduce`、`tril_inverse64`、六个 `kda_layout`、二十多个 `kda_prep`）。
  差异内容就是预期的那几行：`GetCubeNum()` → `GetVecNum()`、
  `GetCubeIdx()` → `GetVecIdx()`、以及 `if (GetSubBlockIdx() == 0)` 那层括号消失。
- **静态检查**：`ascriptor check` 对两个 kernel 在 base 与 cand 下都是
  **0 error / 0 warning**；IR op 数 `finalize_pre` 251 → 248、`finalize_post` 372 → 369
  （少的三个就是被删掉的 `GetSubBlockIdx()==0` 分支）。
- **逐位相同**：六个梯度的 SHA-256 与 base **完全一致**
  （`dq` `c923cd09a388f146…`、`dk` `886ba1ae1566f112…`，其余同），
  相对 L2 也完全一致（`dq` 3.368313291042882e-02、`dk` 7.506592746641179e-02、
  `dv` 3.054e-03）。**不是"在预算内"，是"一个字节都没变"** —— 这正是这类
  纯分区改动应有的判据（AGENTS.md §6：核切分不改算式，不同就是切分 bug）。
- **预算没放宽**：仍用 `kda_bwd` 单元自己的 `ref.oracle`、契约的
  0.05 / `dk` 0.15 / `dg` 0.25。
- **跨 `block_dim`**：cand 的 bd1 与 bd4 六个梯度 SHA-256 **逐对相同**
  （`dq c923cd09…`、`dk 886ba1ae…`、`dv 9934d32b…`、`dbeta ca4ce598…`、
  `dg 9f589abd…`、`dh0 2ea51247…`），且都与 base 相同。
- **第二个 `block_dim` 上同比例**：bd1 下 `finalize_pre` 943.79 → **498.09 µs（1.89×）**、
  `finalize_post` 1842.29 → **1166.68 µs（1.58×）**、合计 9389.93 → **8280.45 µs（1.13×）**
  —— 与 bd4 的 1.90× / 1.55× / 1.13× 一致，说明收益来自 worker 数而不是某个 bd 的巧合。
- **第二个真实形状上同比例**：`long_context`（B1/T4096/H16/HV32/C64），bd4：

  | kernel | base µs | cand µs | 提速 | cand 最高管线 |
  |---|--:|--:|--:|---|
  | `FinalizePreStableKernel` | 976.62 | **554.89** | **1.76×** | `aiv_mte2` 0.504 |
  | `FinalizePostStableKernel` | 1877.54 | **1191.26** | **1.58×** | `aiv_mte2` 0.541 |
  | 反向合计 | 9563.10 | **8429.65** | **1.13×** | |

  六个梯度在这个形状上同样**逐位相同**（`dq eafdb4c9…`、`dk 4125a245…`、
  `dv d42a5e8b…`、`dbeta c46bce20…`、`dg 5177d770…`、`dh0 ae95e1c5…`），
  相对 L2 也完全一致（`dq` 3.722987e-02、`dk` 7.390429e-02、`dg` 1.354975e-01，
  全部在预算内）。

**没有达标，差多少说清楚：**

| kernel | 改前最高管线 | 改后最高管线 | 离 85% 差 |
|---|--:|--:|--:|
| `FinalizePreStableKernel` | `aiv_vec` 0.535 | `aiv_vec` 0.523 | **32.7 点** |
| `FinalizePostStableKernel` | `aiv_vec` 0.571 | `aiv_mte2` 0.536 | **31.4 点** |

**这是预期的，不是意外。** 管线利用率是**单核**的忙时 ÷ 窗口；worker 数翻倍让窗口
和每核的工作量同时减半，比值当然不变。这一步买到的是**墙上时间**（1.90× / 1.55×），
不是占空比。要把占空比推上去必须做下面的 ②。
**所以：本轮改动在"比基线快"这条上成立且有逐位证据，在"≥85% 管线利用率"这条上
不成立，差 31~33 点。** 两件事分开报，不用前者糊过后者（用户 2026-10-27 的验收线）。

### ② 取数与计算完全串行（本轮没改，这才是 85% 的拦路者）

把忙时加起来就看出来了。`finalize_pre` 在 241.89 µs 窗口里：
`aiv_vec 0.535 + aiv_mte2 0.441 = 0.976`；`finalize_post`：`0.571 + 0.416 = 0.987`。
**两条口的占空比之和都几乎正好是 1，也就是说向量计算与它的 GM 取数几乎没有重叠。**

原因在源码里是明摆着的：两个 kernel 的 UB tile 全部是单槽 `Tensor(...)`，不是 `DBuff`。
配上 `auto_sync()`，每次迭代的十来个 `gm_to_ub_pad` 必须全部落地才能跑 `@vf`，而
`@vf` 必须跑完才能让下一次迭代覆写同一块 UB。于是时间线是
`load → compute → store → load → compute → …`，一条口忙的时候另一条必然闲。
（同一个症状在 decode 的 T=1 上也出现，见 `kda_perf_decode.md` —— 那边三条口之和
是 0.963。**这是本仓两个独立单元上同一个结构性毛病**。）

**所以 ① 的收益是"同样的串行时间线，并行跑两倍份数"：墙上时间减半，而每核的占空比
基本不变。**要把 `aiv_vec` 推到 0.85 以上必须做 ②：让 MTE2 与 V 重叠。为什么本轮没做：

| kernel | 输入 tile | 输出 tile | 合计 | 双缓冲后 | UB 容量 |
|---|--:|--:|--:|--:|--:|
| `finalize_pre` | 66 KiB | 72 KiB | 138 KiB | 276 KiB | **256 KiB** |
| `finalize_post` | 164 KiB | 82 KiB | 246 KiB | 492 KiB | **256 KiB** |

（`ascriptor/devices/profiles/950.json` 的 `capacities_kb.ub = 256`。）

- `finalize_pre` 整体双缓冲**不够**（276 > 256）。只双缓冲输入（66→132，输出仍 72）
  能放下（204 KiB），但那只让 MTE2 重叠、MTE3 仍串行，按忙时估算
  `aiv_vec` 只能到约 0.66，**仍不达标**。
- `finalize_post` 现在就已经占了 246 KiB —— **一个槽都加不了**。它要双缓冲必须先把
  每个 tile 从 64 行砍到 32 行（footprint → 123 KiB，双缓冲 246 KiB 刚好放下），
  而那要把 `finalize_post_vf` 的反向 cumsum 拆成两半并把 `running` 在两半之间
  落到 UB 再读回 —— 数学上完全等价（本来就是顺序的），但那是对这个 `@vf` 的
  重写，不是一处改动。

按 AGENTS.md §6.5「一次只改一处」，②留给下一轮，并且**两条路都要先量再改**：
砍成 32 行会把每个 `gm_to_ub_pad` 的 burst 数翻倍，有可能把 MTE2 忙时推上去，
净收益不是算出来的、要测。

## 上游侧最大的两项（不在本任务写集）

- `InverseEpilogueKernel` 557.44 µs（23.1%），`aiv_vec` 0.596 + `aiv_mte2` 0.469
  = 1.065 —— **和两个 finalize 同一个毛病的味道**（占空比之和贴近 1）。它是
  ascriptor 仓 `kda_bwd/kernels/inverse_epilogue.py`，按 AGENTS.md §3 只读。
- `ScanFusedKernel` 489.14 µs（20.3%），最高口 `aic_mte2` 0.406，**所有口都很低**
  —— 它是反向的串行 state 扫描，天然依赖链长；lint 另外在它上面报了两处
  "even block stride"（`gaps.json` 的 `kernel-nd2nz-suboptimal` 已登记）。

两项合计 1046.58 µs = **43.4% 的反向设备时间**，都在上游单元里。
也就是说：**即使本任务把两个 finalize 做到理论极限，反向整体的上限也被上游这两项
按住。**要动它们得走 ascriptor 侧流程或在本仓建派生单元（§3），那是独立任务。

## 复现

```sh
./run.sh <build> <tag> ./profile_kda.py --unit bwd --case kimi_linear_layer \
         --block-dim 4 --output out/<tag>
# CCE 发射范围（改动 blast radius）：
$KP_PYTHON ./ccediff.py --a build/base --b build/cand1 --diff-text \
         --output out/<tag>/cce-diff.json
```
