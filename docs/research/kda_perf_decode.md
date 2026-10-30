# KDA-P1 · KDA decode 的真机管线利用率

环境身份与采集口径同 [`kda_perf_fwd.md`](kda_perf_fwd.md)。

## 按验收线（对应资源 `PipeUtilization` ≥85%）的结论

公共 decode 入口只产生**一个**设备 kernel，所以这条线只有一格要判 —— 而它**按形状分档**：

| 形状 | 绑定资源 | 实测 | 达标 | 本轮动作 |
|---|---|--:|---|---|
| T=16（投机解码），bd4 | 计算 `aiv_vec` | **0.879** | **是** | 不动（已到位） |
| T=1（纯 decode），bd4 | 无（三条口都 ≤0.413，之和 0.963 = 完全串行） | 0.413 | 否，差 43.7 点 | **未改，被写集挡住**（见下） |
| T=1，bd28 | 同上 | 0.310 | 否，差 54.0 点 | 同上 |

## 先说一件会影响派单的事：公共 decode 路径跑的不是 `kda_fused_recurrent`

KDA-P1 的写集写的是 `kernels/projects/a5/kda_fused_recurrent/**`。实测与读源码一致地
表明，**公共入口 `ascend_fla.ops.kda.fused_recurrent_kda` 执行的不是这个单元**：

`ascend_fla/ops/kda/fused_recurrent.py:49-53` 的 `_native_kernel()` 从
`kernels/projects/a5/kda_fused_recurrent_bf16/kernels/step.py` 取
`kda_decode_fp32_kernel` / `kda_decode_bf16_kernel`；`_compiled_pair()` 编译的就是它们。
同文件 34 行的 `kda_fused_recurrent_kernel()`（指向 `kda_fused_recurrent/kernels/step.py`
的 `kda_fused_recurrent_kernel`）**在整个仓里没有任何调用点**（只有定义处、
`docs/matrix/ops.json` 的清单、以及 `benchmarks/a2` 的静态审计表引用它）。

真机 profile 直接证实了这一点：`kimi` decode 形状下一次公共调用**只有一个**设备
kernel，名字是 `KdaDecodeFp32Kernel`。

两个单元的 ABI 也确实不同 —— `kda_fused_recurrent` 收 BHV-major
`[B, HV, T, 128]`（要 host 侧先重排），`kda_fused_recurrent_bf16` 直接收公共的
token-major `[B, T, H/HV, 128]` 并在 kernel 内按 `hi // (HV//H)` 取 q/k 的头。
后者是前者的继任实现（BF-06 交付，已 done），**而它不在本任务写集里**，也不在
`board.json` 的 `reserved_paths` 里，当前没有活跃负责人。

> **因此本节只给实测结论与定位，不改代码。** 要把下面这条优化落地需要把
> `kernels/projects/a5/kda_fused_recurrent_bf16/**` 纳入写集 —— 按 D-PM-33
> 「写集扩大」要用户/PM 决定，assignee 不自行扩大（AGENTS.md §6.5 / PROTOCOL）。
> 在 `kda_fused_recurrent`（本任务写集内）里做同一处改动是可行的，但那个 kernel
> 公共入口到不了，等于优化一段取不到的代码、也无法用公共路径做真机验证。

## 基线（公共入口 `fused_recurrent_kda`，FP32，`kimi` 形状 B1 / HV32 / H32）

| case | `block_dim` | 设备 µs | `aiv_vec` | `aiv_mte2` | `aiv_mte3` | `aiv_scalar` | 三条口之和 | 判定 |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| T=1（纯 decode） | 4 | **19.09** | 0.413 | 0.325 | 0.225 | 0.065 | **0.963** | 未达标，差 43.7 点 |
| T=1 | 28 | **6.66** | 0.310 | 0.295 | 0.198 | 0.221 | 0.803 | 未达标，差 54.0 点 |
| T=16（投机解码） | 4 | **131.47** | **0.879** | 0.076 | 0.040 | 0.009 | 0.995 | **计算 bound，达标** |

`cube_utilization` 全部 0.000% —— 这个 kernel 是 `@kernel(mode="vec")`，不碰 cube，
所以判定只看向量侧的计算口（`aiv_vec`）与搬运口（`aiv_mte2`/`aiv_mte3`）。

正确性（每次采集之前，同一进程，预算取 `kda_fused_recurrent/contract.json` 的
`rtol=1e-5 / atol=1e-6 / max_relative_l2=1e-5`，**没有放宽**）：

| case | `o` 相对 L2 | `final_state` 相对 L2 | `allclose` |
|---|--:|--:|---|
| T=1，bd4 | 8.272e-08 | 2.841e-08 | 通过 |
| T=1，bd28 | 8.272e-08 | 2.841e-08 | 通过 |
| T=16，bd4 | 1.670e-07 | 1.358e-07 | 通过 |

bd4 与 bd28 的 `o` / `final_state` **SHA-256 逐位相同**
（`0669f418…` / `284bd4fa…`）—— 核切分不改算式，这是不需要参考的硬判据。

## T=16 已经达标；T=1 卡在"取数与计算完全不重叠"

**T=16：`aiv_vec = 0.879`，向量算术管线 87.9%，过线。** 这不是调出来的，是现成的：
T=16 时每个头的 state 只读写一次（64KiB 进、64KiB 出），却要跑 16 个 token × 2 趟 ×
128 行的递推，算术完全压住了那份固定搬运。

**T=1：三条口的占空比之和 0.963，而最高一条只有 0.413。** 这个形状几乎是教科书式的
"完全串行"：向量忙的时候 MTE2/MTE3 闲，反之亦然。按 19.09 µs 窗口折算忙时约
`V 7.9 µs + MTE2 6.2 µs + MTE3 4.3 µs = 18.4 µs ≈ 19.09 µs`，三段几乎首尾相接。

源码上的原因是单槽：`kda_decode_fp32_kernel` 里

```python
state = Tensor(DT.float, [D, D], Position.UB)      # 64 KiB，单槽，不是 DBuff
...
for work in range(begin, end):                      # 每核 B*HV/GetVecNum() 个头
    state[0:D, 0:D] <<= initial_state[bi, hi, …]     # MTE2 64 KiB
    …                                                # 四个 gm_to_ub_pad
    recurrent_steps(state, …)                        # V
    final_state[bi, hi, …] <<= state[0:D, 0:D]       # MTE3 64 KiB
```

配 `auto_sync()`，第 i+1 个头的 `state` 载入必须等第 i 个头的 `state` 写回完成
（同一块 UB），于是 `load → compute → store` 三段按头串成一条链。T=16 时 compute
那一段长 8 倍，串行的代价被摊薄到看不见；T=1 时它就是全部。

**改法与预算（本轮未实施，写集之外）**：把 `state` 做成两槽 `DBuff`，让第 i+1 个头的
GM→UB 与第 i 个头的 compute、第 i−1 个头的 UB→GM 三段重叠。UB 预算算得过来：
state 双槽 128 KiB + `qu/ku/vu/gu/ou`（各 `T_MAX=16 × 128` FP32 = 8 KiB）40 KiB +
`bu` 0.5 KiB ≈ **168.5 KiB < 256 KiB**（`950.json` 的 `capacities_kb.ub`）。
按忙时估算，窗口可从 19.09 µs 降到约 `max(7.9, 6.2, 4.3) ≈ 8 µs`（**约 2.4×**），
`aiv_vec` 随之推到约 0.95 —— **这一条能把 T=1 也送过 85% 线**。
BF16 入口 `kda_decode_bf16_kernel` 是同一个结构，同样适用。

这与 `gaps.json` 的 `decode-call-overhead`（P1）不是同一件事，必须分开记：
那条说的是**每次调用约 48~58 µs 的 host 侧固定成本**（桥的 acl 描述符 / workspace
查询 + 布局），本节量的是**设备侧的 19.09 µs**。两件事叠加才是用户看到的延迟；
本轮只动得了（也只量了）设备侧这一半。顺带校正一个数：那条 gap 记的设备侧边际是
"2.7~4.8 µs/token"，本轮实测 T=1→T=16 的边际是 `(131.47 − 19.09)/15 = 7.49 µs/token`
（FP32、bd4、单次 Level1 采集）—— 比原记录高，原因之一是这里是 FP32 而非 BF16，
之二是原值是从"整次调用"反算的、混了 host 侧。**设备侧的数要用设备侧的采集去取。**

## `block_dim` 在 decode 上确实有效（更正一条旧结论）

`gaps.json` 的 `decode-call-overhead` 写过 "`block_dim` 1/2/4/8/16/28 的总时长
54~67 µs **没有趋势**"。那句话对**整次调用的 wall time** 是对的（固定成本主导），
但对**设备时间**不对：bd4 → bd28 实测 19.09 → **6.66 µs（2.87×）**。
B*HV = 32 个头，bd4 给 8 个向量子块（4 头/核）、bd28 给 56 个（1 头/核，24 核空转），
所以 2.87× 已经接近 4 头→1 头能拿到的全部。**"设备侧没有趋势"是把 host 固定成本
算进去造成的错觉** —— AGENTS.md §6 铁律一说的那类错。

## 复现

```sh
./run.sh base <tag> ./profile_kda.py --unit decode --case decode_t1  --block-dim 4  --output out/<tag>
./run.sh base <tag> ./profile_kda.py --unit decode --case decode_t1  --block-dim 28 --output out/<tag>
./run.sh base <tag> ./profile_kda.py --unit decode --case decode_t16 --block-dim 4  --output out/<tag>
```
