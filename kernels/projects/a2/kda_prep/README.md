# A2 KDA 输入准备与原始输入梯度

本单元用 A2 UB 向量指令实现归一化、门控变换和 sigmoid。生产 runtime 只分配输出、检查元数据和启动 kernel。独立参考在 CPU 上运行。运行时不导入 A5 单元或其参考实现。

`runtime.py` 提供 chunk/decode 各 14 个前向入口，以及 16 个原始输入梯度入口。`backward/gate` 包含逐块部分和与跨块规约两次启动。BF16 输出使用 RNE；门控前向输出保持 FP32。比较保留 `budgets.json` 和 `backward_budgets.json` 中原有数值阈值。这两个文件的历史设备、gate span 和资格描述不是 A2 资格记录。A2 的实际状态见 `evidence/qualification.json`。

## 已执行的前向检查

选定环境为 Ascend910B3、Python 3.11.15、Torch 2.10.0+cpu、torch_npu 2.10.0、CANN 9.0.0（时间戳 `20260428_134817545`）。library 固定为 `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`，kernels 固定为 `b3b3f9c16df7c4626ed3c081032a1be5a753d0b1`。各回执另记录源文件哈希和算子包身份。

- bd1、bd2 各在独立进程运行 14 个 T4096/HV32 完整形状检查，均满足原冻结误差预算；14 对输出哈希相同。
- 两路共 56 个契约案例通过；28 对输入与输出哈希相同。BF16 输出另检查距两种舍入参考不超过 1 ULP。
- bd1 的 100 个补充案例中，60 个数值案例通过。其余 40 个是溢出、下溢及非有限值观测，未作为数值通过项。
- 新单元独立测得物理 20 个 cube、40 个 vector 核。vec-only 启动的 `GetVecNum` 在 bd1/bd2 分别为 1/2。未测试更高 block_dim。
- 完整真机检查之后，三个缩小的 norm/gate/beta 案例分别通过 sim 和 pipesim。模型结果不代表完整形状或硬件性能。

FP32 norm 仍有精度审查项：补充案例距正确舍入 FP64 最大 2 ULP，距 CPU FP32 最大 3 ULP。相对误差在冻结预算内，但不能据此跳过既有的 ULP 审查要求。极端值观测也不继承 A5 的资格例外。

## 反向实现边界

A2 `muladddst` 的真机探针没有返回补偿算法需要的融合乘积残差。本实现改用 FP32 尾数位拆分恢复残差，保留补偿算术与固定规约顺序。该原语的 4096 项逐位通过只证明原语，不证明完整梯度。

编译器按分配累计占用 UB，不自动复用内联 helper 的局部缓冲。反向 helper 因此使用显式、互不重叠的临时区，映射在 `kernels/scratch-layout.json`。主缓冲、系数表和索引表各自分配。每次 DMA 前后执行本地 ALL barrier；无预取或跨核共享，也没有新增事件 ID。该保守排程的性能尚未测量。

原始反向在 bd1/2 各通过 28 项完整 T4096/HV32 输出检查，共 48 个契约案例、112 项输出比较通过。24 对契约输入、56 对输出及 16 对 gate 部分和哈希跨 bd 一致。反向三个缩小案例分别通过 sim/pipesim；gate pipesim 首次 60 秒超时，同源同案例的 180 秒有界复跑通过，原失败日志保留。bd1 的额外反向边界中，26 个数值案例通过，16 个端点案例保留逐点 candidate/CPU FP32/Torch NPU/FP64 四列观测。没有 candidate-only NaN；上溢、下溢及舍入差异仍待范围审查。公开低层前向/cached-forward/完整反向已在三种真实形状 × bd1/2 通过，原层的 prefill 128 token 加 5 步 decode 也通过。整份层训练测试和三轮训练性能三明治仍待验收。当前证据不支持翻转公共 `qualified` 状态。

## 单元运行

先按外部机器配置选择受支持环境及设备。完整硬件 workload 必须在模型诊断之前执行；首次自定义启动前注册该进程所需的全部 vendor。

```bash
python run.py reference --output <ignored-output>/reference
python run.py check --launcher sim --case norm_bf16_bf16_reduced_bd1 --output <ignored-output>/sim
python run.py check --launcher pipesim --case gate_bf16_f32_f32_reduced_bd1 --output <ignored-output>/pipesim
python run.py check --launcher board --probe-gaps --board <private-board-alias> --output <ignored-output>/board
```

上述命令描述运行方式。只有 evidence 中明确记录的 stage 和 case 已获检查，不把可运行命令列作执行证明。
