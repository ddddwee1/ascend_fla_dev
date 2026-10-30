# KDA-P1 KDA 核心三单元性能优化：先 profile 到底，按 85% 管线利用率验收

- 波次 / SoC：W0 / a5
- 优先级：P1 · 时限：24h · 依赖：无（三个单元均已真机功能验收通过）
- 需要：A5 真机（D-PM-34）、ascriptor workspace、最新 PyPTO-Pro 框架 + 官方插件
  `pypto-pro-op-orchestrator`（见下）
- 写集：`kernels/projects/a5/{kda_fwd_stable,kda_bwd_stable,kda_fused_recurrent}/**`、
  `tests/test_kda_{fwd_npu,bwd_npu,decode}.py`、`docs/research/kda_perf_*.md`（新建研究文档）
  ——**不包含** `ascend_fla/ops/kda/*.py`（A2-15 正在改这几个文件做 A2 设备透传，写集冲突，
  已避开）。本任务只动 kernel 实现层，不碰公开调度层；如果优化需要改调度层签名，先问用户。

## 这条任务是怎么来的

用户要求对已经功能验收通过、但从未系统性做过性能优化的算子族启动优化，覆盖 KDA 的三个
核心真机单元：`kda_fwd_stable`（前向）、`kda_bwd_stable`（反向）、`kda_fused_recurrent`
（decode）。这是 AGENTS.md §6"性能优化任务的验收线"（2026-10-27 用户给出，2026-10-27
用户澄清覆盖所有 kernel）定下规则后，第一批正式按这条线执行的任务之一。

## 开始之前

**先把工具链升级到最新 PyPTO-Pro 框架，并安装官方插件**
[`pypto-pro-op-orchestrator`](https://gitcode.com/cann/pypto-gym/tree/master/cannbot-skills/plugins-official/pypto-pro-op-orchestrator)
（`cann/pypto-gym` 仓库，不是 `cann/cannbot` 社区版）。如果装不上或者发现它对这次任务不是
必需的（PK-08/PK-07 的经验是：真正的编译阻塞往往是 CANN 版本问题，不是 PyPTO-Pro 本身），
如实记录下来，不要为了"装了"而硬套。

## 范围（三个单元分别处理，一次只改一处）

对每个单元：

1. **先用真机 profile 定位瓶颈**（`PipeUtilization`，真机实测，不是模拟器 `model_cycles`）：
   每个阶段/kernel 的 `aic_mac_ratio`（计算）和 `mte2`/`mte3` 占比（搬运），用
   `docs/matrix/models.json` 的真实形状（不是契约的玩具形状）。
2. **读源码定设计空间**：是否已经在用 cube matmul（KDA 的 `intra.py`/`wy.py` 大概率已经
   是 cube 实现，这次更多是确认利用率有没有打满，不是从零做 cube port）；如果某个阶段是
   vector-only，参考 PK-07 的 cube port 经验（`pkda_chunk_fwd_bf16/kernels/output.py` 的
   `mode='mix'` 模式）。
3. **一次只改一处**，每次改动都要：
   - CCE 发射 diff 确认改动范围精确；
   - 功能模拟器/真机双重验证逐位一致（跨 `block_dim`）；
   - 正确性预算不变（FP32 判定，1e-4，双 oracle）；
   - 真机同卡同步三明治测真实速度提升。
4. **FP32 cube matmul 一定要检查 `set_hf32` 是否已设置**（见 HF-01 的发现，这三个单元正是
   HF-01 在诊断的对象）——如果 HF-01 已经有结论，直接用；如果还没有，本任务自己先确认
   目标 kernel 有没有这个问题，不要在不知道精度基线干不干净的情况下去优化。
5. **量化停止线**：每个被改动的 kernel 最终要达到计算 bound 或搬运 bound（对应资源真机
   `PipeUtilization` ≥85%）。没打到就如实写清楚卡在哪、差多少，不要用"比基线快"模糊过去。


## 写集边界（避免和 A2-15 冲突）

A2-15（进行中）的写集包含 `ascend_fla/ops/kda/{chunk.py,chunk_bwd.py,fused_recurrent.py,autograd.py}`
（给公开调度加 A2 设备透传）。本任务不碰这些文件，只在 `kernels/projects/a5/` 下的 kernel
实现层做改动——公开 API 调用这些 kernel 的方式不变，不需要改调度层代码。

## 验收

- [ ] 三个单元各自的 profile 结果（真实形状，真机，按阶段）。
- [ ] 每处改动独立验证（CCE diff + 模拟器/真机双重逐位核验 + 正确性预算不变）。
- [ ] 真机同卡同步三明治性能数字（带 SoC/CANN/opp 身份、原始日志）。
- [ ] 每个改动过的 kernel 报告是否达到 85% 利用率；没达到的写清楚差距。
- [ ] FP32 cube matmul 的 HF32 设置状态逐一确认（不留空白）。
- [ ] `git diff --stat` 对 `reserved_paths` 为空；不碰 BF16 独立写集。
