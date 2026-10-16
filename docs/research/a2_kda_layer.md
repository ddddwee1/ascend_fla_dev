# A2 KDA 公开路径与层级验收

当前 A2-15 尚未完成层级验收，**不建议翻转 `CAPABILITIES["a2"]["qualified"]`**。提交值保持 `False`。新增 layout 和 prep 的完整形状真机结果不能代替公开层、prefill→decode 或训练性能验收。

本报告的公开整链与 prefill→decode 数字对应提交 `bc25e7f2402cee98738206e5e577a8aa4ec54bf5`，基于任务起点 `63ce595617c2c2416b8ba0c09bf94586107fc76e`。当前 PR 为 draft；最终验收和恢复归档仍待补齐。范围来自 [A2-15 派单](https://github.com/ddddwee1/ascend_fla_dev/issues/42#issuecomment-6194692557)。

## 实现范围

三个公开入口按实际 A2 ABI 选择前向、FP32 累积门控反向和 BF16 decode kernel。A2 前向输入为 token-major；反向保留 FP32 `g_cumsum`，其余八个缓存保持 BF16。A2 decode 沿用已有 BF16 输入域，对 FP32 输入明确报错。

新增 `a2/kda_layout` 实现布局搬运、转换和零填充；新增 `a2/kda_prep` 实现归一化、门控、sigmoid 及原始输入梯度。完整公开路径需要 65 个 vendor：layout 6、chunk prep 14、decode prep 14、prep backward 16、核心前向 5、核心反向 9、decode 1。每个 block_dim 使用独立进程，首次自定义执行前注册全部依赖。

A5 分支的专门主机对比覆盖 15 个分支/核数组合、320 次编译请求。请求顺序、参数、返回键和 kernel 源码与任务起点一致。这是源码及路由对比，不是新的 A5 真机验收。

## 本次环境

| 项目 | 已核实身份 |
| --- | --- |
| SoC | Ascend910B3，物理 20 cube / 40 vector |
| Python / Torch / torch_npu | 3.11.15 / 2.10.0+cpu / 2.10.0 |
| CANN compiler / OPP | 9.0.0，时间戳 `20260428_134817545` |
| 内置算子包 | `ascend910b`、`ascend910_93`、`config` |
| ascriptor | 0.1.0；library `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5` |
| kernels | `b3b3f9c16df7c4626ed3c081032a1be5a753d0b1` |
| HF32 | 关闭 |

设备连接、映射和锁仅在忽略的外部机器配置中。每次运行重新检查健康，保持任务锁和独立输出目录；未复位设备或干预其他任务。

## 已执行的检查

| 范围 | 实际结果 | 结论边界 |
| --- | --- | --- |
| layout | 56 个契约案例、58 个补充检查逐位通过；28 对跨 bd 输入/输出哈希一致 | 仅 layout 叶子 |
| prep forward 完整形状 | bd1/2 各 14 项 T4096/HV32 检查通过；14 对输出哈希一致 | 原相对误差预算通过；FP32 norm ULP 审查仍开放 |
| prep forward 契约 | 56 项数值检查通过；28 对跨 bd 输入/输出哈希一致 | 不含公开层验收 |
| prep forward 补充边界 | bd1 的 60 个数值项通过；另 40 个极端值观测单列 | 不继承 A5 的端点资格 |
| prep backward 完整形状 | bd1/2 各 28 项 T4096/HV32 梯度检查通过；28 对输出哈希一致 | 原冻结梯度预算不变；另 48 个契约案例、112 项输出比较通过 |
| prep backward 补充边界 | 26 个数值案例通过，另 16 个端点观测单列 | 逐点四列保留；端点不计入 CPU 精度通过 |
| 公开完整链路 | 3 种真实形状 × bd1/2 共 6 项通过，24 对返回输出、27 对缓存哈希跨 bd 一致；6,144 项逐头/逐 chunk 输出与192项逐头末态比较通过 | 普通前向、完整 cached-forward、9 个核心反向；未代表完整层训练 |
| 原层 prefill→decode | prefill 128 token 与随后 5 个逐 token decode 均通过原 0.1 预算 | 仅原文件这一项无梯度测试，完整层尚未通过 |
| 原始整份层测试 | 5 passed / 6 failed，0 setup error；65 vendor 预编完成，35 次自定义 kernel 执行 | 5 项失败来自 autograd 选择 A5 prep，另 1 项为 A5 报错断言；未通过验收 |
| 新单元核数 | layout、prep 各自实测 vec-only 在 bd1/2 的参与数为 1/2 | 更高 block_dim 未测试 |
| A2 累加静态守卫 | 6 passed | 无新增 cube 累加或事件 ID |
| 主机全套 | 1585 passed / 10 skipped / 1 failed | 上游 README 看板生成不一致，尚未全绿 |

独立参考在运行时生成。完整链路的 golden 使用 Torch CPU FP32。孤立准备操作另保留已有 FP64 精度参考和 BF16 ULP 判据；没有放宽阈值。反向比较器对 84 个逐输出零值、反号及放大错误对照均拒绝，这也不构成设备验收。

完整硬件检查之后，prep forward 的三个缩小案例分别通过 sim/pipesim；prep backward 的三个缩小案例通过 sim。反向 pipesim 的三个缩小案例也已通过。gate 首次在 60 秒限制下报告 `SimDeadlock`；同源同案例在 180 秒有界复跑中通过，无 hazard、deadlock 或事件余额错误。owner 实现会在工作线程超过时限时使用同一异常；首次空 blocked 列表及复跑结果均保留，不据此扩大模型覆盖。

## 保留的故障和审查项

BF16 strided gather 的早期 layout 候选在硬件发生 507035；补充 barrier 未解决。显式设置 normal mask 后，完整真机重跑通过。早期小模型没有捕获该故障。

A2 `muladddst` 不提供本次补偿算法需要的融合残差，4096 项负对照已复现。位拆分残差探针逐位通过，随后完整准备梯度通过。早期反向编译还暴露了累计 UB 分配超过 196608 字节、自动同步分析耗时及 BF16 标量转换问题。当前候选使用显式临时区复用、串行本地 barrier 和 FP32 标量 staging；没有修改上游或 32-ID 上限。

FP32 norm 正常数补充案例距正确舍入 FP64 最大 2 ULP，距 CPU FP32 最大 3 ULP。八项逐阶段真机探针逐位复现产品结果：平方项一致，差异位于规约、开方和除法舍入。该结果已提交 [精度审查](https://github.com/ddddwee1/ascend_fla_dev/issues/42#issuecomment-6211127376)，不能仅凭相对误差通过就跳过原有审查规则。

`autograd.py` 仍有 17 处 helper 调用默认选择 A5。该文件不在已列出的写集中，最小 device 接线正在等待 [PM 范围回复](https://github.com/ddddwee1/ascend_fla_dev/issues/42#issuecomment-6209806081)。未修改该文件。上游主机检查失败已另报 [README 看板问题](https://github.com/ddddwee1/ascend_fla_dev/issues/42#issuecomment-6210793134)。

在提交 `147237d8` 上，原始 `tests/test_kda_layer_npu.py` 整份运行结果为 5 passed / 6 failed（718.08 秒，包含完整预编）。三个浅跨度梯度案例、默认初始化前向、深跨度反向都在 `autograd.py:118` 调到 A5 `prepare_backward` 后失败。深跨度实现选择案例则收到 `A2 KDA declares only impl='stable'`，与原测试期待的 A5“门控跨度”报错不符。已按实际 A2 支持域修正这一断言，保留 A5 原断言和后续 stable 有限性检查；没有改任何数值预算。修改后的整份测试仍待接线获准后重跑，不计为通过。

原始 11 项的逐项 pytest 回执、未过滤而仅脱敏的日志、源码哈希、65 项编译签名和健康回执位于 `kernels/projects/a2/kda_prep/evidence/integration/original-layer-diagnostic-v1/`。控制器正确返回失败码 1，运行后健康为 0。

## Torch NPU 基线

下面是修改后的 benchmark 在同一 A2 环境执行的独立前向基线。每项 warmup 3、repeat 10，计时前后同步。CPU 参考使用原量化输入升为 FP32，参考输出也保持 FP32。

| 形状 B/H/HV/T | Torch NPU vectorized ms | 输出相对 L2 | 末态相对 L2 |
| --- | ---: | ---: | ---: |
| 1/1/1/64 | 20.150 | 0.00164449 | 1.41406e-7 |
| 1/32/32/1024 | 23.453 | 0.00165891 | 1.77703e-7 |
| 1/16/32/1024 | 22.934 | 0.00166110 | 1.78997e-7 |
| 1/16/32/4096 | 49.635 | 0.00165865 | 1.79982e-7 |

以上是独立基线。随后完成的公开算子三轮夹测见下；完整层与训练性能尚未完成。

## 公开前向三轮性能夹测

提交 `8e0c9b6d` 的产品源码与 `bc25e7f2` 相同。本次在同卡、独立 bd1 进程先注册全部 65 个 vendor，再按 Torch NPU 基线 → 普通前向 → 带门控检查前向 → cached-forward → Torch NPU 基线执行三轮。每段 warmup 3、重复 10 次，计时前后同步；输入在运行时按 benchmark 固定种子生成，各实现使用同一组输入，q/k/v 为 BF16，g/beta/初态为 FP32。下表为三轮中位数，单位 ms。

| KDA 形状 B/H/HV/T | Torch NPU 前后均值 | 普通前向 | 带门控检查前向 | cached-forward | 前向加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1/16/32/4096 | 47.560 | 64.837 | 65.136 | 367.778 | 0.734× |
| 1/32/32/1024 | 22.463 | 16.029 | 16.471 | 84.757 | 1.401× |
| 1/16/32/1024 | 21.338 | 16.038 | 16.498 | 86.053 | 1.330× |

加速比取各轮“前后基线均值 / 普通前向耗时”的中位数。T4096 普通前向比 Torch NPU 慢；cached-forward 包含构造九份反向缓存的成本。这里没有测完整层或训练步，不据此给出训练加速结论，也不开展本任务范围外的性能优化。

原始九轮记录、环境和源码身份、65 项编译签名、退出与健康回执位于 `kernels/projects/a2/kda_prep/evidence/integration/forward-performance-v1/`。全部 vendor 签名与此前完整公开链路验收一致。Torch NPU 的 base-format 分配警告保留在原始日志；定位为 `TensorFactories.cpp:340` 的格式选择，未抑制警告。可移植入口 `reproduce_forward_performance.py` 只核对语法与 CLI，实际数据来自原隔离运行；外部机器配置负责环境、锁和独立缓存。


## 公开整链与逐 token 结果

每个进程先注册全部 65 个 vendor，之后才执行自定义 kernel。下面各形状在 bd1/2 均通过，普通前向与 cached-forward 的输出和末态逐位相同。参考为运行时生成的 Torch CPU FP32 逐 token 递推及梯度；九个缓存使用独立 CPU FP32 构造。

| KDA 形状 B/H/HV/T | 输出相对 L2 | 末态相对 L2 | dg 相对 L2 | 九缓存最大相对 L2 |
| --- | ---: | ---: | ---: | ---: |
| 1/16/32/4096 | 0.00327451 | 0.00245604 | 0.02171016 | 0.00321225 |
| 1/32/32/1024 | 0.00326434 | 0.00256218 | 0.01917324 | 0.00321154 |
| 1/16/32/1024 | 0.00326325 | 0.00253628 | 0.01932947 | 0.00320915 |

层测试没有改动：prefill 相对 L2 为 0.00447696；token 128–132 分别为 0.00370432、0.00422260、0.00431989、0.00307665、0.00428091。每步均按原预算 0.1 判定，并检查 recurrent state 与三份卷积状态交接。

回执在 `kernels/projects/a2/kda_prep/evidence/integration/public-v3/`。其中 `reproduce_public.py` 是同一比较流程的可移植重放入口；它去除了私有缓存路径。先通过外部机器配置选择环境、设备锁和独立缓存，再在独立进程执行每个 block_dim：

```bash
python kernels/projects/a2/kda_prep/evidence/integration/public-v3/reproduce_public.py \
  --repo <repo> --output <ignored-output>/public-bd1 --block-dim 1
```

该脚本也保留完整反向预编。bd1 会执行原层的 prefill→decode 测试。重放入口本身只完成语法/CLI核对；已执行数字来自同一比较流程的隔离运行，源码哈希与构建签名分别保存，不把可运行命令当作新执行证据。

## 尚未完成

整份 11 项层测试、完整训练性能三明治、主机全绿、最终脱敏归档及恢复检查仍未完成。公开普通前向、完整 cached-forward、九个核心反向、原层逐 token prefill→decode 和前向三轮性能夹测已有本次真机结果。全部完成并经 PM 审核后，才能建议翻转资格、转正式评审和提交 DONE。
