# GDN-P1 GDN 核心三单元性能优化：先 profile 到底，按 85% 管线利用率验收

- 波次 / SoC：W0 / a5
- 优先级：P1 · 时限：24h · 依赖：无（三个单元均已真机功能验收通过）
- 需要：A5 真机（D-PM-34）、ascriptor workspace、最新 PyPTO-Pro 框架 + 官方插件
  `pypto-pro-op-orchestrator`
- 写集：`kernels/projects/a5/{gdn_fwd,gdn_bwd,gdn_fused_recurrent}/**`、
  对应 `ascend_fla/ops/gdn_*.py`、`tests/test_gdn_*.py`、`docs/research/gdn_perf_*.md`
  （新建研究文档）

## 这条任务是怎么来的

同 KDA-P1：用户要求对已经功能验收通过、但从未系统性优化过的算子族启动性能优化，这次是
GDN 的三个核心真机单元：`gdn_fwd`（前向）、`gdn_bwd`（反向）、`gdn_fused_recurrent`
（decode）。按 AGENTS.md §6"性能优化任务的验收线"执行。

**与 PK-08 的关系**：GDA-05 已经对 `gdn_fwd`/`gdn_bwd` 做过 stage 级 device/host 耗时拆分
profile（不是本次新测），PK-08 的根因分析方法（从 GDA-05 已有数字反推瓶颈，不需要重新
profile）同样适用于这里——**先去读 `docs/research/gda05_profile.md`，不要假设需要从零
profile**。`gdn_fused_recurrent`（decode）不在 GDA-05 范围内，这部分需要新测。

## 开始之前

同 KDA-P1：先升级到最新 PyPTO-Pro 框架 + 官方插件
[`pypto-pro-op-orchestrator`](https://gitcode.com/cann/pypto-gym/tree/master/cannbot-skills/plugins-official/pypto-pro-op-orchestrator)
（如果发现不是必需的，如实记录，不要硬套）。

## 范围

1. **先读 GDA-05 已有数字**（`gdn_fwd`/`gdn_bwd`），比照 PK-08 的方法论找有没有类似"某个
   stage 比预期慢很多"的异常——PK-08 发现的根因（三角循环守卫破坏向量流水线）是 PK-03
   引入的，GDN 这边的实现路径不同，不要假设同样的根因，但方法论（从已有数字反推，不急着
   重新测）值得复用。
2. `gdn_fused_recurrent`（decode）需要新的真机 profile（真实 decode 形状，不是契约玩具
   形状）。
3. 对每个识别出的瓶颈，**一次只改一处**，验证方式同 KDA-P1（CCE diff + 模拟器/真机双重
   逐位核验 + 正确性预算不变 + 真机三明治）。
4. **FP32 cube matmul 检查 HF32 设置**——GDN 的单元是否有这个问题需要单独确认（HF-01 没有
   覆盖 GDN，只覆盖了 KDA 的三个单元）。
5. **量化停止线**：同 KDA-P1，≥85% 管线利用率，没打到就如实写差距。

## 验收

- [ ] 先确认有没有复用 GDA-05 已有数字（不重复 profile `gdn_fwd`/`gdn_bwd`），`gdn_fused_recurrent`
      的新 profile 数字（真实形状，真机，按阶段）。
- [ ] 每处改动独立验证（CCE diff + 模拟器/真机双重逐位核验 + 正确性预算不变）。
- [ ] 真机同卡同步三明治性能数字（带 SoC/CANN/opp 身份、原始日志）。
- [ ] 每个改动过的 kernel 报告是否达到 85% 利用率；没达到的写清楚差距。
- [ ] FP32 cube matmul 的 HF32 设置状态逐一确认。
- [ ] `git diff --stat` 对 `reserved_paths`（GDN-2 仓主轨道）为空；不碰 BF16 独立写集。
