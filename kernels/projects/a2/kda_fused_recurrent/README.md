# a2.kda_fused_recurrent：A2（c220）上的 KDA decode

> 结论范围：Ascend 910B3 / CANN 9.0.0 / ascriptor library `90cfcdc`。
> A2-14 在 A2-11 之后独立重测通过：bd1/2、真实 Kimi/GQA 形状与状态续接。
> [报告](../../../../docs/research/a2_decode.md)与[原始证据](../../../../benchmarks/a2/evidence/decode/)记录本次源码与环境；公共 dispatch 仍未接线。

## 这是什么

逐 token 的 KDA 递推，覆盖 decode（T=1）与投机解码（T ≤ 16）。state 常驻 UB，每个 token 做两趟行扫描：

```
state <- state * exp(g)            # 沿 K 的逐行衰减
delta  = v - k^T state
state <- state + (beta k) (x) delta
o      = (scale q)^T state         # 用更新后的 state
```

由本仓 `kernels/projects/a5/kda_fused_recurrent/kernels/step.py` 改写，只有一个 kernel：`kernels/step.py` 的 `kda_fused_recurrent_a2_kernel`。

## ABI（与 A5 版不同）

A5 版的 ABI 是 FP32、BHV-major、q 预乘 scale、GQA 在 host 侧展开。这些 host 侧转换现在都被禁止（D-PM-35/37）。
A2 版直接吃公共张量：

| 名称 | dtype | 形状 |
|---|---|---|
| `q`、`k` | BF16 | `[B, T, H, 128]` |
| `v` | BF16 | `[B, T, HV, 128]` |
| `g` | FP32 | `[B, T, HV, 128]`（log 空间） |
| `beta` | FP32 | `[B, T, HV]` |
| `initial_state` | FP32 | `[B, HV, 128, 128]`，key-major |
| `o`（输出） | BF16 | `[B, T, HV, 128]` |
| `final_state`（输出） | FP32 | `[B, HV, 128, 128]` |

kernel 把 token-major 张量当 2-D 视图读写。BF16→FP32 转换、`scale`、GQA 头映射都在 kernel 里做，`o` 用 round-to-nearest-even 转成 BF16。
host 只做两件事：NaN 预填的分配，以及不拷贝的 `view`。

## 向量体

A2 没有 `@vf`。每个 state 行用一条 `muls` 加一条 `add`，标量由 `Var.GetValueFrom` 从 UB 读出，累加顺序同 A5：先乘进临时量，再加。

## 预算

在跑之前写定：
- `final_state` 沿用 A5 decode 的 1e-5。真机实测 4e-8 ~ 1e-7。
- `o` 是 BF16，地板是 FP32 结果本身舍入到 BF16 的误差，实测 1.63e-3 ~ 1.71e-3。预算取 `max_relative_l2` 5e-3，约为地板的 3 倍。

## 运行

```bash
PYTHONPATH=<ascriptor>/library python run.py reference
PYTHONPATH=<ascriptor>/library python run.py check --device a2 --backend cce --launcher sim
ASCEND_RT_VISIBLE_DEVICES=<card> PYTHONPATH=<ascriptor>/library python run.py check --device a2 --backend cce --launcher aclnn
```

契约 8 个 case：单 token；GVA；T=4；T=16；Kimi 形状 H=HV=32；GQA H16/HV32；B2 T8 GQA；零初始 state。

## 没有确立的

- `block_dim` 只跑过 1 与 2。
- A2-14 已测 T=1/16 的三轮同步 Torch NPU 对照，逐轮样本和测量边界见报告；不代表已优化的 decode 性能。

## A2-14 资格化

本次每个 bd 独立进程先编完 decode 与 chunk 共六个 vendor，再执行完整真机 workload。
CPU FP32 递推是主 golden，Torch NPU FP32 是第二 oracle；原有 `reference.py` 内部 FP64
算术作为额外诊断保留。输出与状态预算未变。实际单元入口与续接 launcher 逐位一致，
两种 bd 的 19 组输出/状态逐位一致，128-token prefill 后连续解码 64 步逐 token 验收。

AST 扫描 `kernels/step.py` 中所有 Call 节点，`matmul` 调用数为 0。因此 A2 的 BF16/FP16
split-K 与手写 FP32 MMAD 累加链均不在此 kernel 的执行路径上；全量
`tests/test_a2_accumulate_barriers.py` 仍覆盖它与其它 A2 单元，没有跳过守卫。

```bash
python benchmarks/verify_decode.py --soc a2 --shapes kimi --a2-out <fresh-ignored-directory> --a2-performance
```

该命令从仓库根目录运行；CANN、固定修订的 ascriptor workspace、设备可见性和缓存目录
由外部私有配置提供。decode 不受 chunk 门控跨度字段约束。
