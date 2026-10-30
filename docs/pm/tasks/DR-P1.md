# DR-P1 DeltaNet（gated_delta_rule）真机独立复验 + 性能优化：先把数字坐实，再谈 85%

- 波次 / SoC：W0 / a5
- 优先级：P2 · 时限：24h · 依赖：无（DR-01 已合入，但其真机数字是申领人自述，从未被独立复现）
- 需要：A5 真机（D-PM-34）、**真正的 PyPTO-Pro**（不是 ascriptor——这是本仓目前唯一一个原生
  PyPTO-Pro 单元）、[`pypto-pro-op-orchestrator`](https://gitcode.com/cann/pypto-gym/tree/master/cannbot-skills/plugins-official/pypto-pro-op-orchestrator)
- 写集：`kernels/projects/a5/gdr_chunk_fwd_bwd_pypto_pro/**`、
  `docs/research/dr_perf_fwd.md`、`docs/research/dr_perf_bwd.md`（新建研究文档）、
  `docs/matrix/gaps.json`（记录门控纪律缺口，不在本任务里修）
  ——**不包含** `ascend_fla/ops/*`、`ascend_fla/layers/*`、`ascend_fla/models/*`：
  这个单元目前完全没有接公开 API 层（`ascend_fla/ops/` 下没有任何 delta_rule 相关文件，
  `tests/` 下也没有），本任务不做这件事——那是另一个决定（建 ops 层 wrapper），不是性能优化。

## 这条任务是怎么来的

DR-01（PR #145，外部贡献，2026-10-25 已合入）的 PM 复核明确记录了两件没解决的事，
在 AGENTS.md §6"性能优化任务的验收线"（覆盖所有 kernel，包括已经声称"达标"的历史记录）下
变得需要处理：

1. **PR 自己的硬件数字完全是自述，从未被独立复现**——复核子代理当时的沙箱只有 A2/910B3
   且系统 `pypto` 是 0.2.0（没有 `pl.jit`/`pl.vector_function`，不是真正的 `pypto_pro`），
   装不了申领人用的框架，只能用不需要 pypto_pro 的旁证脚本（`fla_npu_baseline.py`）验证
   oracle 机制是真的，验证不了性能数字本身。
2. **门控纪律与文档不符**（AGENTS.md §7 的精神：文档过度承诺比缺失能力更糟）：
   README 声称"域约束不满足就报错"，但 V 的"64 的倍数"从未被 assert/raise 校验过
   （且与 PR 自己验证通过的 case V=96 矛盾）；B=1 只在 9 个 kernel 里的 1 个
   （`gdr_bwd_dqkw`）靠类型标注生效，其余 7 个训练 kernel 对 B 没有限制；独立推理 kernel
   （第 9 个）完全没有域校验。这些细节记在 `VALIDATION.md` 里但没有体现在 README 和
   实际的报错行为上。

这个单元在 `docs/matrix/ops.json` 里**完全没有条目**——按 AGENTS.md §6 的新纪律，
"没记的，按'未验证'处理"。

## 目标（按优先级）

### 1. 先把数字坐实（这是本任务的前提，不是可选项）

在真正的 A5 + PyPTO-Pro 环境里，重新跑一遍 DR-01/README/`docs/PERFORMANCE.md` 声称的
真机性能数字（前向+反向，`tests/validate_against_fla.py` 的强声明也顺手复核一遍——
应为精确 0.000e+00，不是"在预算内"）。如果这次独立复现的数字与自述数字不一致，
如实记录差异，不要替申领人找理由。

**升级到最新 PyPTO-Pro 框架 + 装官方插件 `pypto-pro-op-orchestrator`**（这是本仓第一个
真正用得上这个插件的单元——KDA-P1/GDN-P1 都发现自己的 ascriptor/CCE kernel 用不上它，
这次不一样，它的 lint/检查应该对这个单元的 PyPTO-Pro DSL 源码有实际判项）。

### 2. 门控纪律缺口：记账，不在本任务里改代码

按 AGENTS.md §6.5"先记账、本仓侧先保证不静默出错、攒够一批再统一修"：把 DR-01 复核发现的
三处门控缺口（V 倍数未校验、B>1 未在前 4 个 kernel 前门拦、独立推理 kernel 零校验）逐条
写进 `docs/matrix/gaps.json`（`requires_kernel_change: true`，写清楚要改哪个文件）。
**不在本任务里直接改**——这是外部贡献的代码，改之前先确认复验结论站得住，而且这类改动
应该独立于性能改动验证（§6 判定纪律：不要混在一起测）。

### 3. 真机 profile 到 85% 管线利用率（同 KDA-P1/GDN-P1 的方法论）

对 9 个 kernel（7 个训练 + 1 个独立推理 + 看 `docs/KERNELS.md` 的确切清单）：

1. 真机 `PipeUtilization` 实测，真实形状（`docs/matrix/models.json`，不是契约玩具形状）。
2. 一次只改一处，每次改动：CCE/PyPTO-Pro 发射 diff 确认改动范围、双 oracle 精度不变
   （FP32 判定）、真机同卡同步三明治测真实提升。
3. 量化停止线：每个被改的 kernel 最终要到计算 bound 或搬运 bound（85%）。没打到如实写
   差距，不要用"比基线快"模糊过去。
4. **FP32 cube matmul 检查 HF32 设置**——`a5-fp32-cube-matmul-inherits-hf32`（P0）现在
   确认只覆盖 ascriptor/CCE kernel 的那一类 HF32 寄存器行为；PyPTO-Pro 的执行模型是否有
   同类风险需要本任务自己核实（不要假设有，也不要假设没有）。

## 范围边界

- 不碰 `ascend_fla/ops/*`、`layers/*`、`models/*`（建 ops 层是另一个决定）。
- 不碰 `reserved_paths`（GDN-2 仓主轨道）。
- 如果发现许可证头 / 版权主体的问题（DR-01 当时已经发现并上报给用户、用户已批准合入），
  这件事已经处理过，不需要本任务重新纠结，除非发现新的、DR-01 没报过的实例。

## 验收

- [ ] 独立复现（或如实指出复现不了的差异）DR-01/PERFORMANCE.md 的真机数字。
- [ ] `tests/validate_against_fla.py` 的强声明精确复现为 0.000e+00（真机，不是自述）。
- [ ] 三处门控缺口记入 `docs/matrix/gaps.json`，不在本任务里改代码。
- [ ] 9 个 kernel 的真机 PipeUtilization 分别记录，达标/未达标各自写清楚。
- [ ] PyPTO-Pro 框架升级 + 官方插件安装情况如实记录（这次预期能真正用上）。
- [ ] HF32（或 PyPTO-Pro 执行模型的同类风险）核实结论。
- [ ] `git diff --stat` 对写集之外（尤其 `ascend_fla/ops/*`、`reserved_paths`）为空。
