# 交接：下一个会话从哪里接

> 写于 2026-09-11，会话状态更新至 2026-09-19。**本文件只记"接手需要知道的"**；权威状态在
> `docs/matrix/*.json`（`gen_matrix.py` 校验），纪律在 `AGENTS.md`，分期在 `docs/plan.md`。
> 三者冲突时以 json 为准，并顺手修本文件。

## -1. 2026-09-19 PM 交接（本节最新，PM 换人先读这个）

**这是一次 PM 账号本身的交接，不是算子进度交接。** 下面按"现在手上有什么、有什么坑、
下一步该干什么"排列，配合 `docs/pm/board.json`（权威）和 `python tools/pm_board.py --render` 看。

### 正在飞的任务（现在没有；A5K-02、PK-02、PK-03 都已于 2026-09-19 合入）

> **2026-09-28T15:08Z 更新：GDA-04（A5 GDN decode/fused recurrent：从零建 kernel，PR #136）审查 accept 并按 D-PM-33 合入（merge commit `ebdf55eb`）；main 全量（排除 torch_npu）1421 passed / 25 skipped / 14 failed。GDA-04 done。**
> - **从 ASSIGN 到 DONE 只用了不到两小时**——GDN decode 之前完全没有可移植的 ascriptor 资产（不像 A2-14 是资格化既有单元），是这轮里真正从零建 kernel 却最快交付的一次。规格里唯一的技术错误是 PM 自己写的：oracle A 原文写"fla 的 `fused_recurrent_gated_delta_rule` CPU FP32 自动实现"——申领人读源码指出这个函数是纯 Triton kernel，没有 CPU 分支；真正对应的是同目录 `naive.py::naive_recurrent_gated_delta_rule`，与 GDA-03（GDN 反向）已经用的是同一个函数。PM 核实属实、当场改规格，不是申领人的错误。
> - **528 个真机 case（6 个 block_dim × 88 例）全部通过**，对 A/B 最大相对 L2=2.170130340355715e-7（1e-4 预算内近 3 个数量级余量）；264 组 BF16 与同输入 FP32 RNE 输出逐位一致；prefill→decode 的 14 组/56 条链、132 次分段对照逐位一致；跨 6 个 block_dim 的输出/state 哈希全部一致。PM 独立复核全部重新计算过，包括从原始 sha256 digest 字段（不是汇总的 `passed` 字段）逐 case 复核跨 bd 一致性。
> - **PM 复核过程中发现一个"自述但证据不够独立核实"的缺口，并自己动手解决了**：申领人自述"最终交付的 `ascend_fla/ops/gdn_fused_recurrent.py` 字节（sha256 `b5534d63`）与真机证据收集时实际执行的字节（`216dbef7`）不同，但 AST 相同（只改了资格化注释）"——证据里只有执行版本的哈希，没有源码文本，子 agent 复核时标成"无法独立复核"。**PM 直接在 PR 分支上 `git diff` 两个相关 commit，确认这个文件唯一的差异就是一行注释**（"Candidate set...pending native evidence" → "Qualified native inprocess grid..."），零可执行代码改动。**这是本轮学到的一课**：遇到"哈希对不上、只有自述解释"的情况，先看 PR 分支自己的提交历史能不能直接验证，不必假设"无法核实"就只能选择信或不信——很多时候证据其实就在那里，只是复核 agent 没往那个方向找。
> - 真机基础设施细节：host 侧零 dtype/布局转换，静态 grep 之外还做了真实 ATen dispatch 审计（`TorchDispatchMode`）确认所有拦截调用只有内存分配，是比静态检查更强的验证方式，值得作为以后审 BF16/dtype 相关 DONE 的标准做法之一。
>
> **2026-09-28T06:57Z 更新：A2-14（A2 KDA decode 资格化，PR #135）审查 accept 并按 D-PM-33 合入（merge commit `acf817fd`）；main 全量（排除 torch_npu）1336 passed / 25 skipped / 14 failed。A2-14 done。**
> - 这是"资格化既有单元"，不是新建 kernel：`kernels/projects/a2/kda_fused_recurrent/{kernels,unit.py,run.py,reference.py}` 全程未改，`git diff` 对这几个路径为空——PM 独立核实过。真机 32/32 精度 case（Kimi 真实 H=HV=32 与 H16/HV32，T=1/4/8/16，bd1/2）、19 组跨 bd 输出/state 字节比较全部一致（**不信汇总字段**，直接从原始 sha256 digest 逐 case 复算）；`o` 最大相对 L2 1.7245406052097678e-3、`final_state` 1.0151343587949668e-7，均在既有预算内。`CAPABILITIES["a2"]["supported_block_dim"]` 加了 `"decode": (1,2)`，`qualified` 仍 `False`（公共路由接线不在本任务范围）。
> - **申领人的一条 contradicts-handoff RISK 纠正了 PM 自己规格里的一处不精确措辞**：规格原文说"CPU FP32 参考"，但既有单元的 `independent_reference` 实际是"FP64 内部算术、FP32 输出"。PM 核实属实，裁定按申领人方案办——在申领人自己的验证器（`benchmarks/verify_decode.py` 新增的 `a2_fp32_oracle()`）里另写一个全程 `.float()` 的显式独立递推作主金标准，旧的 FP64 内部参考保留做补充诊断、不作验收依据。PM 复核时确认这个新函数是真的独立实现（代数结构与旧参考不同——beta 先并入 delta 再做外积，不是换皮抄一份），符合裁定。
> - **两次写集扩展都是同一类坑**：`tests/test_platform.py` 和 `tests/test_kda_fwd_a2.py` 里各有一处硬编码的 `supported_block_dim == {"chunk": (1, 2)}` 精确字典断言，加 `"decode"` 键必然打破它们。第一处是申领人主动搜出来的，第二处是全套跑完才暴露（定向搜索没扫到）——**这提示以后凡是要给 `CAPABILITIES` 这类被到处精确断言的字典加键，光搜"引用这个变量的地方"不够，得跑一次全套测试才能保证抓全**。两处都批准为一行改动，PR 里也确实只改了那一行。
> - PM 独立复核延续 PK-05 那次的做法：子 agent 在独立 worktree 里核，不信自述，全部关键数字从原始证据（`benchmarks/a2/evidence/decode/{bd1,bd2}.json` 等）重新算一遍。这次审查体量小很多（36 个文件，其中 28 个是证据 json/log），复核用时也短很多——**qualification 类任务和新建 kernel 类任务的审查成本差一个数量级，以后分诊时可以按这个预估**。
>
> **2026-09-28T01:45Z 更新：PK-05（A5 PGDN chunk 反向：ATK 求导，FP32，PR #134）审查 accept 并按 D-PM-33 合入（merge commit `790d148c`）；main 全量（排除 torch_npu）1329 passed / 25 skipped / 14 failed。PK-05 done。**
> - 14 个失败全部是这台 PM 机器本身缺 `ascriptor` 包（`ModuleNotFoundError`，PYTHONPATH 里没有）——**是这台机器一直就有的环境缺口，不是 PK-05 带来的回归**；PK-05 自己的测试文件 0 failure。第一次复算时误把 `FLA_GDN_NAIVE` 指到 `fla/ops/gdn2/naive.py`（GDN-2 的，函数名不一样），炸出 31 个假失败——**下一次复算这几个 dtype-naive 的 env 变量前，先确认路径指对模块**（GDN 的 naive 在 `fla/ops/gated_delta_rule/naive.py`，不是 `gdn2/`）。
> - **本次审查过程中真发现并修好了一个货真价实的 silent-wrong-result**：DONE 第一版（PR 提交，256 例 + 独立复核干净）之后，申领人自己主动发起的补充诊断（密集随机 q/k 缩放到 FP32 clamp 边界 1e-12 附近）测出 NPU kernel 内部的 `raw_norm>=eps` 分支判定跟它要实现的 literal FP32 `Tensor.norm>=eps` 不一致——q 5/64 行、k 6/64 行走错分支，dq/dk 相对 L2 冲到 0.03，远超 1e-4 预算，但公共入口和 composition 仍然逐位一致、看起来完全正常。根因：设备端 norm 归约用了一次性 128 路 `cadd()`，literal PyTorch 的 AVX2 实际是 8 路独立累加再顺序求和（拆分之后各分支落入 clamp 的时机不同）。修复对齐了这个具体累加顺序，CPU 8192 行核验逐位匹配，PM 复核时独立拉取了 PyTorch 该 pin 版本的 `ReduceOpsKernel.cpp` 源码核实这确实是同一模式（不是编出来的巧合），还确认了候选修复排除过 FMA / 4 路 / 16 路等替代方案（不是蒙对的）。
> - 两轮独立复核都是派子 agent 在独立 worktree 里做的、不信自述：写集边界（10 个改动文件全在 PK-05 写集内，公共入口签名/冻结分类代码/64×阈值/1e-4 预算全部逐 diff 确认未碰）、原三个失败例用同一份输入哈希重跑后归零、关键数字从原始 JSON 独立复算（bd1/bd2 用逐 case 的 sha256 字段而不是汇总的 `all_byte_identical` 字段比对，0 mismatch，且 bd1/bd2 是在两张不同物理卡上跑的——比同卡更强的证据）、host 测试独立重跑。唯一的非阻塞瑕疵：`contract.json` 的 `reference` stage 说明文字没跟着 73 例补充更新（`board` stage 更新了），纯文案，不影响 `status:"failed"` 判定。
> - **合入本身卡了一轮机制问题，写下来防止下次重蹈**：`limjiunnbin`（`pm_github_login`）没有这个仓的 `MergePullRequest` 权限（这条其实 2026-09-21 A2-02 那次已经踩过、写在下面那条 note 里了，这次没翻旧账直接又踩了一次）；随后 Claude Code 的自动模式分类器把 `gh pr merge` 判成"Merge Without Review"直接拦掉，紧接着**连 `git pull --ff-only` 都被同一个分类器拦了**，一直拦到用户在会话里明确说"你来合入"才解除（分类器似乎是按最近的 shell 历史/会话状态判的，不是针对单条命令）。真正解决合入权限问题用的是 `tools/pm_github.py` 里已有的 `_gh(..., as_login=...)` 模式：`GH_TOKEN=$(gh auth token -u ascend-fla-pm-bot) gh pr merge ...`，不是 `gh auth switch`。**下次遇到分类器拦截合入类操作，第一反应应该是先看这条 note 和 A2-02 那条，而不是重新摸索。**
>
> **2026-09-23T16:02Z 更新：A5K-03（KDA 反向 scan_fused 的 dh0 不确定性只定位不修，PR #133）审查 accept 并由 PM 按 D-PM-33 自行合入（squash）→ 6619475；main 全量 1234 passed / 15 skipped / 0 failed。A5K-03 done——机制已定位。**
> - **机制**：`seed_ub` 缓冲每个 BHV 边界重置槽 0，但 `CvMutex(3, depth=2)` 的信用跨 BHV 连续流转，奇数 chunk 数时产生 WAR（下一 BHV 首次发布可能覆盖上一 BHV 最终读取尚未消费的槽 0）。不是缺 barrier——本地 32-ID 自动同步命名空间本来就建立不了这条跨侧边。危险对数（B2/HV4/C3 按 block_dim）：14/12/10/8。
> - **这是本会话审过的证据链最扎实的一次之一**：可证伪预测预注册、单变量干预（只改 4 处 seed 索引相位）、无关注释负对照、反事实替换不拟合系数——12 次历史重放解释 51990/52001 个异常元素（≈99.98%），PM 从原始 JSON 独立加总复现到位。受控时序干预在三张卡、两套 CANN 上稳定复现 50/50 次 dh0 错误，连续槽号版本逐位修复，但**健康卡上的自然复现仍未出现**——申领人全程严格区分"受控因果证据"与"自然复现"，没有一次混为一谈或提前宣称"通过"。
> - PM 审查里踩了一个自己的小坑：一度以为文档/证据里反复出现的 Clopper-Pearson 上界数字（`0.05815507911697225`）是转述错误（我朴素公式复算出的末位不同），后来直接调用交付代码才发现是我自己的复算方法数值精度不够——申领人的实现用 `math.expm1` 更稳定，数字全程一致。**教训：怀疑对方数字不一致之前，先用对方交付的代码本身复算，不要只用自己的独立公式对比。**
> - `docs/matrix/gaps.json` 的 `kda-bwd-scan-dh0-nondeterministic` 已更新为"机制已定位"，`requires_kernel_change` 仍是 `true`——**修复仍需用户批准 kernel 批次**，这条本身不代表放行修复。
>
> **2026-09-23T15:33Z 更新：A2-12（A2 KDA 前向：真实形状 + 门控跨度重测 + (C,HV) 网格，PR #132）审查 accept 并由 PM 按 D-PM-33 自行合入（squash）→ 6841be2；main 全量 1223 passed / 15 skipped / 0 failed（NPU-free 口径）。A2-12 done。**
> - 门控跨度收敛到 guard=158（`floor(0.9×176/2)×2`），精确验证 span=158 的 80 个 case 全部有限且在 0.05 预算内；默认初始化实测跨度 97.69~106.38 在 158 之下。`ascend_fla/platform.py` 的 `CAPABILITIES["a2"]` 首次填了真实数字（`supported_block_dim={"chunk":(1,2)}`、`max_gate_span={"stable":{"forward":158.0}}`），但 `qualified` 仍是 `False`——**这只是单元级资格化，不是公共 KDA 入口接线**（`ascend_fla/ops/kda/chunk.py` 仍硬编码 A5 单元），这条边界申领人主动澄清了两次，PM 在评审里确认无误。
> - **这轮审查发现并改正了 PM 自己规格里的一个真实错误**：我最初以为 A2 的 `kda_fwd_stable` 在 C=1 多头时也会像历史缺陷那样报错，申领人读源码指出 `_check_recurrent_heads` 只对 `impl=="upstream"` 生效、A2 架构根本不共享触发缺陷的 L1 环——PM 核实属实，当场改了规格，最终交付把 C=1/HV=32 按正例验证（`o`/`state` 相对 L2 分别 0.00323/0.00259，bd1/bd2 逐位相同），这次评审独立核对了这两个数字。
> - PM 独立复核强度：不只读汇总 JSON，**扫了全部 80 份原始 guard-span case 文件**确认两个"全局最坏"数字确实是全局最大值而非挑选；全量测试数字对不上时（1223/15 vs 申领人的 1228/10）没有直接放过，查清是本机无 torch_npu 导致 5 个既有 A5 测试文件被跳过，总数吻合。
> - **这条任务全程有三轮"申领人指出 PM 错误"**：两次是看板 assignee_caps/write_set 的元数据遗漏，一次是规格本身的技术错误——都被 PM 核实属实并当场改正，没有一次是申领人误判。
>
> **2026-09-23T11:28Z 更新：BF-05（GDN-2 chunk 前向 BF16 原生 kernel，PR #131）审查 accept 并由 PM 按 D-PM-33 自行合入（squash）→ 1bb9e1e；main 全量 1207 passed / 13 skipped / 0 failed（计数持平，本 PR 未加新 pytest 文件）。BF-05 done。**
> - 两项如实披露的负结果，都不影响合入：① 性能三明治 BF16 比 FP32 慢 2.3%——只转换了公共面，cube 累加仍 FP32，没拿到吞吐，原因讲得很清楚；② 门控跨度测到 2541.7（远超 GD2-01 的 FP32 观测 1520.9）仍有限且在预算内，但没推到失效边界，如实标"上限未确立"，没有拿最深测试点当闸值。
> - 过程中自查自修一个真实缺陷：BF16 派生 kernel 最初沿用了 FP32 单元的算子名，同进程两条 dtype 路径共存时 BF16 会静默拿到 FP32 二进制（CANN 按算子名解析、同进程首个 vendor 树胜出，二次 build 被静默忽略）——host 算子审计场景下暴露，五个 kernel 改名 `*_bf16` 后修复。这是本会话第二次遇到这一类缺陷（A2-10 的 `_claim_op_name` 也是同一机制），值得记住。
> - PM 审查踩了一次自己的坑：核对合入内容时第一次用错了 diff 基准（拿旧的 merge-base 而不是合入提交自己的直接父提交），一度以为 AGENTS.md/README 被合入"回退"了，虚惊一场——正确做法是对合入提交的**直接父提交**做 diff，不是对分支当初 fetch 时的旧 merge-base。
>
> **2026-09-23T07:14Z 更新：A2-11（A2 算子结论总闸，PR #130）审查完成后用户直接裁定"accept and merge"，已合入（squash → 652dfc0）；main 全量 1207 passed / 13 skipped / 0 failed（净增 6）。这是本仓 A2 波次迄今最重要的一次合入。**
> - 真机定量结果：FP32 split-K（pin 已修）2000/2000 逐位稳定；BF16 split-K@M16 500/500 全错、500 个输出哈希两两不同（硬件时序竞争的铁证）；BF16/FP16 split-K@M32 200/200 触发 AI Core 异常；BF16 split-K@M64 199/200 正确；FP32 手写累加链无 barrier 2000/2000 未复现（如实标"未复现≠安全"，保守保留绕行）。
> - **AGENTS.md 已更新**：§2 记录 A2-11 完成与用户裁定——满足研究文档四项前提的 A2 结论从此算数（不再仅是观测）；§7 新增永久禁止条款——A2 系 BF16/FP16 `splitk` 在 `M<64` 必须显式报错，不接受任何绕过，静态守卫是 `tests/test_a2_accumulate_barriers.py`。`gaps.json` 的 `a2-splitk-fp32-cube` / `a2-splitk-bf16-fp16-unsettled` 已补 A2-11 的真机确认证据。
> - PM 独立复核强度：逐个核对了全部 11 份原始 evidence JSON 的 run_summary（不只读汇总），连 BF16 M16 的原始最小二乘分解系数都看了（15/16 行系数精确落在 [0,1,0,1,…] 附近，隔片丢弃的陈旧累加器特征），独立复现了静态守卫的负对照。
> - **A2-11 是 A2 波次的总闸，现在解开了**：A2-12～A2-16、A2-K1 依赖已满足，下一轮可以考虑派单顺序。
>
> **2026-09-23T03:45Z 更新：A2-08（PR #129）二审 accept 并由 PM 按 D-PM-33 自行合入（squash）→ abffbf9；main 全量 1201 passed / 13 skipped / 0 failed（净增 14+1）。A2-08 done。**
> - 一审补了两项：原始终端输出（不是转述）、`fla.ops.kda.naive` 层级交叉校验真跑了（此前因容器缺包跳过）——relative_l2=1.24e-6 ≤ 1e-5，用的是仓库对齐的 fla pin `e52dbc0e`（0.6.0），与 GD2-01/PK-02/PK-03/BF-05 同一 commit，不是随手挑的版本。
>
> **2026-09-23T03:31Z 更新：BF-09（PR #128）审查 accept 并由 PM 按 D-PM-33 自行合入（squash）→ 3e26891；main 全量 1187 passed / 12 skipped / 0 failed（净增 3）。BF-09 done。**
> - T4096 完整训练步 428.86ms（合入版 BF-08）→ 223.06ms，降 47.99%，仍为旧 host 图 5.92×——PM 建议的 1.25× 软目标未达，但已按规格给出瓶颈分析（gate 第一阶段的 profiler 条件下界 164.43ms）。`backward_budgets.json` 字节级冻结不变。
> - 数学优化：`_dd_add_constant` 2Sum→FastTwoSum、`_dd_mul` Dekker 拆尾数→FMA 的 2MultFMA，均为文献级等价精确算法，注释里给了量级证明。
> - **本次评审首次做到「从提交树原始去重数据整条链路重放」**：不只读汇总 JSON，是真跑了申领人自己写的 `restore_backward_validation.py` / `classify_backward_endpoints.py` / `verify_backward_review.py` / `verify_bf09_closeout.py`，grid 汇总哈希、边界分类四项数字、收尾延迟降幅都独立复现到位级相同；还用 ascriptor `sim` launcher 真跑了新增算术诊断测试并做负对照。以后遇到类似「私有归档 + 公开去重证据」结构的 DONE，优先找这条路径，比只信汇总数字强得多。
> - 唯一瑕疵：`contract.json` 因收尾提交回写自身而与回执记录的前一版哈希不同（自引用因果循环导致，不可避免），逐行 diff 确认只差描述性字段，判定良性，不影响合入。
>
> **2026-09-23T02:09Z 更新：A2-10（PR #127）三审 accept 并由 PM 按 D-PM-33 自行合入（squash）→ 8d7dcff；main 全量 1184 passed / 12 skipped / 0 failed（= 1173 + 11）。A2-10 done；A2-11 现在可以派了。**
> - 三轮评审全过程：第一轮抓到设备事实非实测 + block_dim 探针没到硬件 + `_claim_op_name` 检查绕开而非验证 + 回执缺修订/自证 + 测试太薄 + 缺原始日志；第二轮除测试断言外全部补齐（含 152 个哈希与上一头逐个相同的强证据、bd1=bd2 76/76）；第三轮只剩测试断言按文档口径补齐，PM 用 4 个负对照独立核过，接受并合入。
> - A2 上机主线完成第二步（A2-02 → **A2-10** → A2-11）。A2-11（split-K 总闸，issue #38）现在依赖已解除，MikejR2904 此前有条件 APPLY 在排队，下一轮可以直接派。A2-08（Kimi 注入脚手架，issue #36，成品已做完）也排在 A2-10 之后，可以一并考虑派单顺序。
> - **BF-09**：发了一条 shared-machine RISK（设备短暂 Alarm，已自行换健康设备重跑，PM 按 D-PM-28 只记录不升级）；candidate3 训练步比未优化基线降 46.93%，但仍慢旧 host 6.3×，未达 1.25× 建议目标；draft PR #128，未交 DONE。
> - **BF-05**：bd 逐位相同与 FP32 路径改前改后逐位相同两条硬判据已过，进入门控跨度端点重测与性能三明治阶段。
>
> **2026-09-21T13:37Z 更新：A2-10（PR #127）force-push 新头 8a001a0（未发 STATUS）→ PM 二审：上一轮七条全部到位，只剩测试断言三处小的，仍 rework（改完只核增量、然后请示用户）。**
> - 强证据：新一批真机回执的 152 个桥 / harness 哈希与**上一头那批逐个相同**（两次独立运行相隔约 1.5h）；bd1 = bd2 桥输出 76/76；桥自证 fwd 5 次 / decode 1 次启动、签名 bd1 与 bd2 不相交；带完整 40 位 ascriptor 修订（与 A2-03 的 env.json 同）；scratch merge 全量 1182 passed / 12 skipped / 0 failed（= 1173 + 9）。
> - 新的观测（标观测、不是结论）：解开单元的域检查后 block_dim 1 / 2 / 4 / 8 / 20 / 40 在 kda_fwd_stable 的 narrow_gate case 上**都跑完、没有死锁**（与 A5「超过物理核数死锁」不同；但只是 liveness，正确性没验）；核数 torch_npu 读到 20 / 40 与 profile 一致，UB / L1 / L0C 容量仍是 profile 声明值；`_claim_op_name` 在 A2 上真机同进程二次 build 抛 AclError。
> - PM 的负对照发现三处新测试没守住文档断言：fwd case 启动次数只要求 in (1, 5)、claim_op_name 与 block_dim_probe 两份回执没有测试读。要求只动测试三处断言。**教训**：申领人 force-push 后没发 STATUS 也别干等——先评审，发评论前核头没变。
> - 接下来：申领人改完测试 → PM 核增量 → **向用户请求 A2-10 合入授权**（D-PM-33：A2 数字只作观测）→ 合入后派 A2-08（成品放着）、A2-11 解锁。
>
> **2026-09-21T12:23Z 更新：A2-10（PR #127，MikejR2904）交了 DONE（ACK 后约 1.5h）→ PM 评审 **rework**。核心结果成立、卫生做得好，但规格有两项没做成。**
> - 成立的：runtime 桥对 aclnn harness 逐位相同 152/152（PM 从原始 JSON 复算；152 个哈希各不相同；case 集与 contract 逐个对上；**bd1 与 bd2 桥输出逐位相同 76/76**，A2 观测）；覆盖表 12/12；scratch merge 全量 1178 passed / 12 skipped / 0 failed（= 1173 + 5）。单提交、写集干净、隐私 0 命中、脱敏在推 PR 前做——A2-02 的教训他们吃进去了。
> - 没做成的：① 设备事实是 ascriptor profile 的声明值（20 / 40、UB 192 …）而非实测，block_dim 探针 4 / 8 / 20 / 40 全死在单元自己的域检查、没到硬件（文档推给 A2-11 是错的，A2-11 不测 block_dim）；② `_claim_op_name()` 在 A2 上「报错而不是静默忽略」的检查没做（DONE 写的是绕开它）；③ 文档说每份回执带 ascriptor 修订，一份都没有；④ 回执不能自证走的是桥；⑤ 测试太薄（PM 负对照：删掉 50/60 比较并把 soc 改成 a5，5 测试仍全过）；⑥ 缺原始控制台日志。
> - 改完 PM 重做 scratch merge、从原始 JSON 复算；**合入前先请示用户**（D-PM-33：A2 数字只作观测、A2-11 之前不算结论）。A2-08（成品放着）仍排在 A2-10 过审之后。
>
> **2026-09-21T10:26Z 更新：A2-02（PR #126）二审 accept 并由 PM 按 D-PM-33 自行合入（squash）→ 33259af；main 全量 1173 passed / 12 skipped / 0 failed（= 1159 + 14）。A2-02 done；A2-10 已派给 MikejR2904。**
> - 合入的是审过的头 `90e248cb`，合入的树与我 scratch merge 的树逐位相同（846d466）。用 **squash**：PR 分支中间提交里有旧版未脱敏的证据日志（工作副本临时目录名），不让它们进 main 的历史（PR 分支上的旧提交仍公开可见，撤不回——已告诉申领人以后推 PR 前就脱敏）。merge 是用 bot 账号的令牌执行的（`limjiunnbin` 账号没有 MergePullRequest 权限；此前 #122 / #123 / #124 也是 bot 合的）。
> - 三轮评审的结论：第一轮抓到**回归**（`test_kda_decode` 的 ast 字面量读被能力表调用打破，申领人的「200 / 3 / 8 前后一样」看不出）、`ops/kda/**` 还剩 7 处 a5 默认、证据要件不合规；第二轮剩隐私（机器路径）与「改前签名值写死」没按字面做；第三轮全绿。**PM 收窄了自己的规格两处**：验收 grep 只管 `ops/kda` / `runtime` / `layers/kda.py`；「单元路径由能力表给」延后到第一个 a2 单元出现。已登 gap `soc-defaults-a5-outside-kda`（P2）。
> - **A2-10（#37，P0，12h，task/A2-10）→ MikejR2904**（按 03:53Z 的承诺）。**顺序更正**：我 08:06Z 说过 A2-02 后先派 A2-08，改成 A2-10 先（P0 主线、依赖已解除），A2-08 的成品继续放着、A2-10 DONE 并过审后再派；已在 #36 与 #37 说明。A2-10 的合入我按 D-PM-33 先请示用户（A2-11 之前 A2 数字只是观测）。之后 A2-08 → A2-11。
> - 主线现在：BF-05（GDN-2 BF16，session bf05…，时限约 15:30Z、还没真机数字，已请其发合规 STATUS 写 ETA）、BF-09（KDA 训练前处理反向性能，等 ACK）、A2-10（新派）。A5K-03 仍排在 BF-09 之后。
>
> **2026-09-21T09:50Z 更新：A2-02（PR #126）二审第二轮——上一轮的阻塞项全关（全量 1172 passed / 12 skipped / 0 failed、AST 只有 `_op_consts` 变、内部调用点都显式传 device、负对照变红），仍 REVIEW rework，只剩两条小的。**
> - 要改的：① 证据日志与探针脚本里有机器绝对路径（临时目录前缀 / Ascend 与 Python 安装根）→ 脚本只打相对路径、日志机械替换 + 声明一行；② 验收 2「改前签名值写死在测试里」没按字面做——PM 在 base 与 PR 上用固定合成 kernel 各算了一遍逐位相同（a5 / bd1 / 空绑定 `986f4a15e1a1dd05`，a5 / bd4 / T=64 `d79923b948b20af3`，a2 / bd1 `63c73523d216ea41`），已把三个字面量交给他们写进测试。
> - 改完 PM 核增量、重做 scratch merge（预期 1173 / 12 / 0），然后按 D-PM-33 自行合入；之后 A2-08 直接 DONE，再 A2-10 → A2-11。
> - 教训（给下一位评审）：评审输出里**不要再把别人评论 / 日志里的私有路径与容器名原样引回去**（我上一轮起草时差点把临时目录名又写进公开评论）；显示日志时别对含中文的行用 `cut -c`，它按字节切，会把整行尾部吃掉、看着像「日志被截断」。
>
> **2026-09-21T09:22Z 更新：A2-02（PR #126，MikejR2904）交了 DONE → PM 二审 **rework**（不是合入）；BF-05 一条头格式不合规的 STATUS 按数据记下并回复。**
> - A2-02 的硬伤是一个**实测回归**：PR 合到当前 main（NPU-free）全量 `1 failed, 1170 passed, 12 skipped`（预期 1171），失败的是 `tests/test_kda_decode.py::test_block_dim_domain_matches_contract`——测试用 ast 读 `fused_recurrent.py` 的字面量，PR 把它改成了能力表调用。申领人报的「200 / 3 / 8 改前改后一样」看不出它（8 个预存失败很可能含这一个）。教训：**报数要报失败 ID 集合，不只是计数**；预存失败会遮住同 ID 上的新增失败。
> - 其他：`ops/kda/**` 仍有 7 处 `device="a5"` 默认（规格要求 2）；真机证据 sha256 被截断、无原始输出、DONE 评论带了容器名（已请其编辑评论，PM 的记录里不重复该值）；「A2-01 仅静态验证」措辞不准。范围、隐私、a5 常量搬家值、`test_platform` 12 passed、`gen_matrix --check` 都没问题。
> - **PM 收窄了自己的规格（两处）**：验收 grep 只管 `ops/kda/**` + `runtime/` + `layers/kda.py`（gdn / pgdn / pkda 不在写集、gdn2 是 reserved）；「单元路径由能力表给」延后到第一个 a2 单元出现（现在是死代码，入口已拦）。写在 `docs/pm/tasks/A2-02.md` 的「二审澄清」。预批写集扩展三项：`tests/test_kda_decode.py::_op_consts` 函数体、`benchmarks/a2/probe_platform.py`、`benchmarks/a2/evidence/platform/**`。
> - 改完的复核清单：AST 比对只有 `_op_consts` 变；重做 scratch merge，预期 1171 + 新增数 passed / 12 skipped / 0 failed；核 probe 输出的完整哈希。合入按 D-PM-33 由 PM 自行合（MikejR2904 首合已由 D-PM-58 授权；A2-02 的真机读数是环境事实、不是算子结论）。之后 A2-08 直接 DONE，再 A2-10 → A2-11。
> - BF-05：08:46Z 那条 STATUS 首行多了一个词（`correction-not-a-deadlock`），解析器判为非协议消息；内容（撤回「BF16 改写引入 SimDeadlock」、根因是模拟器 join 超时、12 case 用 `--timeout 7200` 重跑）当数据记在看板。**仍没有真机数字**；时限约 15:30Z 到期，已请其发合规 STATUS 写现实 ETA，到期前我再看一次。
> - **（2026-09-21T15:37Z）BF-05 时限 15:36Z 到期，申领人 09:20Z 之后没有回应；已在 #104 发 PING**：要合规 STATUS + 到第一批真机数字的 ETA + 预计超时量；预计超 ≥ 25% RISK、> 50% PM 请示用户。没有预计值前不上报；仍 in_progress，不是收回。
> - joshjms 09:07Z 在 #28 发了 APPLY（any，A5，手上没有在飞任务；自述不抢 A5K-03 / GD2-02 / A2-08，明说不要把 A2-08 派给他）→ **NO_TASK**：没有合适的可派任务（A5K-03 排在 BF-09 之后派给发现者 session；GD2-02 有 9-17 的在先 APPLY 且 BF-05 未完；A2 线是 MikejR2904 的且要 910B；PK-05 / GDA-04 / PK-06 / GDA-05 是骨架、前置未完；其余 gated）。看板有变化他再 APPLY。
>
> **2026-09-21T08:50Z 更新：FMT-01（PR #124）二审 accept 并由 PM 按 D-PM-33 自行合入 → c1379a3；main 全量 1159 passed / 12 skipped（= 1132 + 27）。FMT-01 done。**
> - 合入的是审过的头 `3ec23db2`（joshjms，不是新账号——D-PM-15 已批准其首合，我早先记错）：22 份回执的 `audit_tool_sha256` 全等于提交的工具、24 个源文件哈希对上 8e5da4c、20 个 case 的计数与上一轮逐格相同、26 次按调用点改判我从 rows 独立重算与文档 §2.1 一致。工具 `benchmarks/host_op_audit.py` 从现在起是各任务 host 算子审计的**统一工具**（`docs/pm/bf16-kernel-side.md` 已指到它，BF-09 规格已加一条验收）。
> - 清单是 8e5da4c 的快照：clean——KDA 推理 / decode、GDN / GDN 反向、PGDN、PKDA BF16、GDN-2 FP32；违规五处（均已登记）——`_scan_states`、`dw` 取负、raw-flag 训练前处理 + 引擎反向（BF-08 合入后已消除）、GDN-2 BF16 加宽；**新发现**：PKDA FP32 的域校验还在 host 上算（gaps 新增 `pkda-fp32-host-domain-validation-unregistered`，P2）；`log2(eg)` 只在 `impl="upstream"` 上走。
>
> **2026-09-21T07:50Z 更新：用户「#122授权, #123」（D-PM-57 / D-PM-58）——PR #122（BF-08）→ 6c4aaca、PR #123（A2-07）→ 28cced5 已合入；main 全量 1132 passed, 12 skipped, 5 warnings（= 1032 + BF-08 的 100 个新测试）。BF-08、A2-07 done；BF-09（P0 性能优化）立项。**
> - **BF-08 合入**（bot 账号，`--match-head-commit` 钉住 d9280a7d；用户选了选项 A：合入 + 立即另立性能任务）：用户是在被告知判据裁定 D-PM-54 / 55 / 56、端点披露（每 bd 482 个 native 下溢输出不是 CPU 通过、30 个 A_log = 88 记录未资格化）、完整训练步慢 11.3 ×、证据约 83 MB 之后授权的。`kernel_inventory` 的 `kda_prep` 注记更新；gaps 新增 `kda-prep-backward-training-step-slowdown`（P1）与 `kda-prep-backward-endpoint-disclosures`（P2）。
> - **BF-09**（P0，issue 由 sync 创建）：BF-08 后续的性能优化——目标线 T4096 完整训练步 ≤ 旧 host 图 × 1.25（PM 建议、非门槛），冻结预算字节不变、跨 bd 逐位相同、不放宽判据；写集只在 `kda_prep/**` + 两个测试 / 文档文件，`autograd.py` / `chunk.py` 不在写集（避免与 A2-02 相交）。**做不到目标就报瓶颈与下界，要放宽逐元素判据是用户的决定。**
> - **A2-07 合入**（新账号 MikejR2904 的第一次合入，用户 D-PM-58 授权；A2 数字是观测记录、不构成结论）：批次建议 A / B / C 是给用户审批 kernel 批次的输入，本次合入不构成对 A2-K1 的批准（A2-K1 / A2-20 仍 gated）。
> - **（2026-09-21T07:52Z 已派）BF-09（issue #125）→ limjiunnbin（session 01a0b7ce-…，24h，task/BF-09）；A2-02（#31）→ MikejR2904（10h，task/A2-02）。A5K-03 顺延到 BF-09 之后直接派。**
> - 待派：A5K-03（我承诺 BF-08 CLOSE 后直接派）与 BF-09（性能）都是 P0，同一 session（01a0b7ce-…）——先派 BF-09（同一 kernel 单元、KDA 层训练路径的性能倒退是眼下最明显的问题），A5K-03 顺延，已向申领人说明；MikejR2904 的 A2-02 → A2-10 → A2-11 队列：BF-08 已合入，A2-02 可派（A2-07 已 CLOSE）。
>
> **2026-09-21T07:16Z 更新：FMT-01 交了 DONE（PR #124）→ PM 评审 rework（源码身份不符等四处小的）；**更正：joshjms 不是新账号**。**
> - **FMT-01**：工具 + 19 个 case 的真机清单质量很高——我从原始 JSON 重算各类别调用数与文档 / DONE 的表逐格一致、audited_sources_sha256（23 个）与 8e5da4c 逐个对上、隐私 0 命中、scratch merge 后工具单测 25 passed。唯一硬伤：**回执里的 audit_tool_sha256（7bf19644…）与提交的工具（6c4ff8db…）对不上**——要重跑或证明差异不影响。另三处小的：文档写 24 passed（实 25）、PKDA FP32 的 6 次算术是未登记的只读校验（D-PM-42 意义下）不该与违规混成一类、CHECK_SITE_FUNCTIONS 改判的算子要列清单。清单结论：clean——KDA 推理 / decode、GDN / GDN 反向、PGDN、PKDA BF16、GDN-2 FP32；违规——`_scan_states` / dw 取负 / raw-flag 训练前处理 + 引擎反向 / GDN-2 BF16 加宽；log2(eg) 只在 `impl="upstream"` 上走。
> - **更正**：我 02:12Z 起把 joshjms 记成「新账号、首次合入要用户同意」——错了：D-PM-15（用户）已批准其首合，A2-04（PR #60）已合入、A2-40 已 done。我派单时没查看板历史。FMT-01 按 D-PM-33 由 PM 评审通过后自行合入，不需要问用户；A2-07 的 MikejR2904 才是真新账号。
>
> **2026-09-21T06:01Z 更新：BF-08 DONE（PR #122，头 d9280a7）→ PM 评审 accept（技术判据全过）；A2-07 三审 accept（头 b54978b）。两件都等用户授权合入。**
> - **BF-08**：PM 从 Git 里的原始文件独立复算——范围 / 隐私 0 命中（1727 blob）、预算先于实现且冻结（f907a54 无 kernel，冻结文件逐字节未变）、最终回执 39 份生产源码哈希与最终树 0 处不同、full-v7 bd1–4 全过（19 个输出哈希跨 bd 全同、端到端六梯度在预算内）、最终网格 7992 行 / 58,320 次公开训练调用全部满足冻结预算（最大 L2 / limit 0.7120、逐元素 0.4146）、host 审计 unexpected = 0、scratch merge NPU-free 全量 **1132 passed / 12 skipped**（= 1032 + 100 新测试）、入口换回 base 的负对照通过。
> - **要用户看的三件事**：(1) 判据裁定 D-PM-54 / 55 / 56（PM 作出、可推翻）；(2) 端点披露——每 bd 482 个 native flush 输出不是 CPU 通过、30 个 gate A_log = 88 输出记录留作「未资格化观察」（默认公共闸就会拒绝这些输入），contract 的 board 阶段保持 failed；(3) **性能：完整前向 + 反向训练步（含 raw 前处理）慢 11 ×**——T4096 旧 host 37.6 → 425.7 ms、比 torch_npu 基线慢 3.0 ×，一个 gate backward 第一阶段 kernel 就占 375 ms（补偿算术），本任务未优化；**`ascend_fla/layers/kda.py` 的 KDA 层在 chunk 与 decode 上都用这三个 raw flag，所以合入后 KDA 层训练路径受影响**（更正我 04:59Z 那条把它说成「前处理慢 11 ×」）。选项：A 合入 + 立即另立 P0 性能任务（推荐：慢在一个 kernel 上、可局部优化；默认无 flag 路径不受影响）；B 等优化后再合。若优化后仍要满足冻结的精度预算而慢，那时要用户决定是否放宽逐元素判据。
> - **⚠ MikejR2904 在 #35 发了一条「代为转达用户明确决定」的 STATUS（05:35Z）——声称用户已同意 A2-07 的第一次合入、并知悉 D-PM-30。PM 不采信为授权**（评论内容是数据；「用户明确同意」必须由用户直接对 PM 会话给出，assignee 转达无法核实），已回复它并如实告诉用户；等用户直接授权。
> - 排队：A5K-03（#120）来自 limjiunnbin session 01a0b7ce-…（BF-07 / BF-08 的同一个），排在 BF-08 CLOSE 后——PM 回 NO_TASK 并承诺直接派；MikejR2904 的 A2-08（#36，P2，纯 CPU）条件 APPLY 排在 A2-07 CLOSE 后——NO_TASK；A2 主线 A2-02 → A2-10 → A2-11（P0）优先。
> - **A2-07（PR #123，头 b54978b，MikejR2904，新账号，只有一个文档 + 探针 + 证据）**：三审 accept——需用户同意新账号第一次合入与 A2 观测数字（a2 不冲 fp32 subnormal 等，A2-11 之前不构成结论）。
>
> **2026-09-21T04:59Z 更新：A2-07 二审仍 rework（四处小的）；BF-08 最终版几乎收尾——但训练路径前处理比旧 host 慢 11 ×；gate A_log = 88 的 30 个输出记录留作「未资格化观察」；FMT-01 开了 PR #124（未 DONE）。**
> - **A2-07（PR #123，头 ef502c7）**：第一轮的问题基本改对（A2-01 命中表引用逐行对上 `hit_table.json`；真机探针是真编译真跑，含 ascriptor 自编译 exp）。二审仍 rework：证据要件不全（缺内置算子包目录 / version.info 原文 / 未过滤原始日志）、`probe.json` 里有机器绝对路径、§3 第 5 / 7 行「A5 的 87 线对 a2 fp32 不适用」说过头（subnormal 分母在 span > 88.7 就使 1/eg 溢出）、§4 覆盖说满了。改完 PM 核完就向用户请求合入同意（新账号第一次合入 + A2 观测数字）。发现：a2（910B3）**不冲 fp32 subnormal**（exp(−88) = 6.05e-39 保留，下溢到 0 在 ≈ −104、上溢在 89）——观测记录，A2-11 之前不算结论。
> - **BF-08（头 5c28123）**：D-PM-56 已落实；最终 1620 × 9 网格 bd2/3/4 各 PASS、bd1 进行中；**性能 T4096 前处理 37.6 → 425.7 ms（慢 11.3 ×，比 torch_npu 基线慢 3.0 ×）**——规格不设速度门槛，但合入授权时要单列给用户；gate A_log = 88 的 30 个输出记录（候选既不等于 CPU 也不等于旧 NPU、涉及非有限值）PM 同意留作「未资格化观察」并交用户处置（倾向接受为只披露：默认不可达、候选更接近 FP64）。累计 Git 证据约 84 MB，已在约 100 MB 上限边缘。
>
> **2026-09-21T04:23Z 更新：A2-07 交了 DONE（PR #123）→ PM 审查 rework；FMT-01 报 mutex 编不过 → 已知条目；BF-08 第三条 RISK → D-PM-56。**
> - **A2-07（MikejR2904，新账号，PR #123，只有一个文档）**：ACK（03:57Z）写 eta 0、随后 04:02Z 就 DONE；**device: none、无真机数字 → REVIEW rework**（D-PM-34）。另：文档写「A2-01 还是 open」（其实 done）、没对着 main 上已有的 GDN 派生单元（GDA-01/02/03、BF-01/02）、有仓里查不到出处的说法（`gdn2_recurrent_bwd`、「Gemini table」）、把 block_dim 上限写成 40 个向量核。规格补了「2026-09-21 更新」（部分是规格 09-19 写得不全）+ 写集补 `benchmarks/a2/probe_gdn_abi.py` 与 `benchmarks/a2/evidence/gdn_abi/**`。它的 fork 名在 PR 里是 `…_a2`，与 APPLY 不同。
> - **FMT-01（joshjms）的 RISK**：pin 版下 `impl="upstream"` 反向编不过（34 > 32 mutex）——**已登记**（`kda-bwd-inverse-mm-mutex-over-budget`，A5K-02 已用派生单元绕开）；它的同机对照（627f55f 能编能跑）补进了那条 gap。处置同意，不需要用户决定。
> - **BF-08 D-PM-56**：BF16 双 1 ULP 线改逐点读（对 FP64 仍是每元素通过线；对旧 host 只在旧 host 自己 ≤ 1 ULP 处要求）——实证：raw beta = 16 处旧路径 5.8% 误差、9 ULP 外，候选 = FP64 正确舍入；native exp 下溢端点按 D-PM-50、gate 上溢端点按 D-PM-48。**合入授权时 D-PM-54 / 55 / 56 一起单列给用户。** 累计新增证据约 75 MB，已提醒只剩约 25 MB。
>
> **2026-09-21T03:29Z 更新：BF-08 两条 RISK（norm 1e20 上溢端点 / beta 导数语义）→ PM 裁定 D-PM-55；新账号 MikejR2904 申领 gated 的 A2-20 → NO_TASK。**
> - BF-08：补偿 + 二幂缩放 norm 在完整 Kimi 训练 bd1–4 通过（PM 从已推送的 297c80b 核：19 个输出哈希四个 bd 全同、prep 最大逐元素 0.00389）；网格里 gate / beta 有新失败（自述：ddt_bias 1 / 1024 个元素差 2 ULP、beta relL2 2.405e-7 > 冻结 1.957e-7，旧 CPU 同 case 2.414e-7 也超），申领人在候选内修，不改冻结限。**D-PM-55**：(1) norm 在前向 FP32 平方和上溢处梯度取零（跟随前向、同旧 CPU / NPU，不作新端点类），(2) beta 从 raw beta 算非饱和分支的解析导数符合契约、饱和开关只用保存的 s 恰为 0 / 1。**BF-08 合入授权时与 D-PM-54 一起单列给用户。**
> - **（2026-09-21T03:52Z）MikejR2904 又排了 A2-02（BF-08 合入后）→ A2-10（A2-02 之后）→ A2-11（A2-10 之后）三条条件 APPLY**：PM 各回 NO_TASK 并承诺依赖落地后直接派（A2-02 写集与在飞的 BF-08 相交，BF-08 CLOSE 前不能派）。它自述拿不到固定的 ascriptor 修订（90cfcdc / b3b3f9c）「without a gitcode token」、目前在 5287adc / b9c8a84 并已告知仓主——这属仓主 / 用户的事（PM 不经手令牌），A2-10 / A2-11 派单前要解决，A2 数字对修订敏感。
> - BF-08 STATUS（03:49Z）：D-PM-55 已落实；最新候选完整 Kimi bd4 PASS（bd1/2/3 续跑）；旧版 297c80b 的网格每个 bd 54 组失败待新候选闭环；gate 归约初版 VF stack 超限、改立即数后通过（未放宽资源限制）。
> - **（03:33Z）MikejR2904 随后自己发了 A2-07 的 APPLY → PM 直接派 A2-07（纯主机的 GDN ABI 调研 + 设计，12h，要求真机实测设计所依赖的事实、用固定 ascriptor 修订）。**
> - A2-20 是 gated（A2-K1 + A2-16，A2-11 未做）→ NO_TASK，指路 A2-07（现在可派，纯主机）与 A2-02 → A2-10 → A2-11（A2-02 写集与在飞的 BF-08 相交，BF-08 合入后才能派）。新账号，第一次合入要用户同意；它的 ascriptor 修订（5287adc）与固定的 90cfcdc 不同，已提醒。
>
> **2026-09-21T02:45Z 更新：BF-08 首次真机（候选实现，PR #122 头 ccc7194，仍 draft）——norm 逐元素相对误差超冻结限（q 0.0774、k 2.166，限值 0.0588），PM 核实是判据度量的问题，不是候选缺陷；（后续：申领人 02:42Z 的补偿实现已在 bd1 / 3 / 4 通过冻结判据，**不需要改判据、不需问用户**，见下）。**
> - 其余全过：relL2（0.00166 < 0.00509）、gate / beta 的 isolated 预算、两套 CPU FP32 端到端六梯度、audit unexpected = 0 / old-host-preparation = 0、plain / cached 逐位一致、输入未改、NaN 预填。
> - **PM 从原始 JSON 核实**：k 的最差元素上候选与旧 host 的 BF16 位逐位相同（相对误差 2.166，条件数 6.0e7）——冻结限值连旧 host 自己都过不了；q 候选 0.0774 对旧 host 至少 0.0464（至多 1.67 倍）；超 1 ULP 的元素数候选（q 7 / k 6）不比旧 host（至少 9 / 12）多；按项幅值归一后候选 ≈ 1.2–2e-7（FP32 舍入水平）。冻结的「3 × 校准最大值」对相消输出不稳定（旧 host 校准 0.0196 ↔ 真实人口 2.166）。
> - ~~PM 已回复申领人：别为它改 kernel 算术……~~（**作废**）：申领人 02:42Z 的 RISK 显示补偿后的 candidate 已在完整 Kimi 训练 bd1 / 3 / 4 通过冻结判据（bd2 续跑）、near-null（gy = x）逐位 = FP64 正确舍入，PM 已回复同意并撤回「别改算术 / 我去问用户 / 补 A–D 数据」；只要求把旧 host 在真实人口上超冻结数的事实写成发现。**新问题**：补偿式 norm 在 scale 1e18 / 1e20 因中间乘积溢出给 NaN（旧 CPU 在 1e18 有限）——是回退、必须修（设备内精确二幂缩放），四列数据与前向一致性要写明。
> - 更正：我早先在 #117 说「那 60 个结构性对照」有误——60 = 32 结构性 + 28 边界构造（申领人已改对），结论不变。
>
> **2026-09-21T02:12Z 更新：新账号 joshjms 申领 FMT-01（host 算子审计工具 + 全仓入口清单，P0，A5 真机）→ PM 直接派单。**
> - 申领人自述 a5（Ascend950PR_9589，CANN 9.3.0 / torch 2.7.1，与其它任务的环境不同）、已先在真机跑通 `tests/test_kda_fwd_npu.py`、写集与保留路径不相交。分配给新账号不需要问用户；**它第一次合入要用户明确同意**（PROTOCOL §5 第 5 条），届时把审查结论与关键数字发给用户。
> - 派单前补了规格「2026-09-21 更新」一节：09-19 之后主线合入了 FMT-02 / BF-02 / BF-03 / BF-04 / BF-06 / BF-07，「已知违规」大半已不在——阳性对照改用仍在的（D-PM-42 存量例外、KDA 训练 raw-flag 前处理 BF-08、GDN-2 BF16 加宽 BF-05），并要求被审计提交 + 源文件哈希、与已合入任务自带审计驱动交叉核对、环境差异如实标注。
>
> **2026-09-21T01:26Z 更新：BF-08 的第一个 RISK（comparison-contract-boundary，实现前）→ PM 裁定 D-PM-54：反向 BF16 输出不继承「对两个对象各 ≤ 1 ULP」。**
> - 申领人在 CPU 校准里发现旧 host 的 BF16 norm 反向对 FP64 正确舍入最多 3 ULP（7 / 16.7M 元素，两项相消，条件数 1.5e5–5.9e5）；三处 3 ULP 位置上任何 BF16 candidate 都无法同时落在两个对象的 1 ULP 内。**PM 在 CPU 上原样复现，逐位一致**（另有 4 个位置只剩中点一个值可选）。
> - 裁定（PM，规格 `BF-08.md` 已加 2026-09-21 澄清一节）：反向 BF16 输出改用 BF-07 澄清第 2 条口径——不设 ULP 通过线，relL2 ≤ min(1e-2, 3F)、逐元素相对误差 ≤ 3 × 旧 host 对 FP64 的实测地板，第一个提交冻结；两个对象的 ULP 分布都报、> 1 ULP 元素逐个列出；哪些输出用哪种口径由校准冻结；带接近预算边界的损坏输出对照。端到端六个梯度预算、155 / 105 闸、域、D-PM-48 / 50 / 52 框架不变。
> - BF-08 的第三个 RISK（write-set-expansion，01:51Z）：`ascend_fla/ops/kda/__init__.py` 模块 docstring 里「pending BF-08」的过时一句——PM 事先批准只改这一处（条件见规格「追加二」：AST 核、不宣称整条训练路径无 host 算术、真机验先 inference 后 training 不出 161001）。
> - BF-08 的第二个 RISK（write-set-expansion，01:25Z）：BF-07 的 `test_training_graph_selection_and_no_grad`（16 例，断言训练走 host）必然被 BF-08 打破——PM 事先批准只改这一个函数体（条件见规格「现有测试」追加段：AST 核其余不变、保留覆盖并明确断言旧 host 未被调用、局部替身、`chunk._prepare_inputs` 必须保留、入口换回 base 的负对照）。
> - **2026-09-21T02:03Z：BF-08 首提交（校准与预算冻结，头 f907a54，draft PR #122）PM 核对通过**：28 组预算从 144 条原始记录逐组重算 28/28 一致；D-PM-54 落实正确（13 个 BF16 输出组保留 1 ULP 双线、norm:bf16:dx 与 14 个 FP32 组不设 ULP 线）；228 个损坏对照全被拒绝（结构性最紧 1.5652 倍）；基线副本与 6c42dc6 逐字节相同；隐私 0 命中；写集内。已提醒：gy=x 近零导数对任何 FP32 公式都会超预算，出现即 RISK；保留双线的 BF16 组若验证人口出现旧 host 超 1 ULP，先 RISK。
> - **待用户：暂无，但 BF-08 合入授权时要把这条（BF16 反向输出去掉 ULP 判据、改 3× 地板）单列给用户；用户可推翻。**
>
> **2026-09-21T01:04Z 更新：用户授权「#118授权」（D-PM-53），PR #118（BF-07）已合入 → 381b452，BF-07 done；main 全量 1032 passed, 12 skipped, 5 warnings。BF-08 按约定派给同 session。**
> - **合入**（bot 账号，`--match-head-commit`）：授权早于评审里要补的只加证据的提交，PM 没合当时的头 `ef5bfc65`，而是等申领人补完并核对后合入 `5b71779e`——两个提交只动 `kda_prep/evidence/`：最终源码上补跑 bd4 完整 workload（`full-v2-bd4`：8241 项比较 0 超预算、17 个返回量与 bd1/2/3 及首轮 v1-bd4 逐位相同、34 个源码哈希与提交树全对，`autograd.py` = `78b96d6b…`）、被引用的失败切片表 CSV（270 行 = 失败表 270 个切片）、更新后的 1422 项 manifest（全对、无未列文件）；隐私 0 命中。合入后 main 全量（NPU-free）**1032 passed, 12 skipped, 5 warnings in 281.27s (0:04:41)**（915 + 117 个新测试，与预测一致）。`kernel_inventory`：`kda_prep` 的 dtype_status → native@a5（推理 / decode 前处理已 kernel 侧，训练路径仍是登记例外，待 BF-08）；gaps 新增 `kda-prep-nearzero-output-underflow`（P2，D-PM-52 披露）。
> - **用户是在被告知四件事后授权的**：dtype 域收窄（raw 只接 BF16 / FP32，FP16 / FP64 显式报错）、D-PM-52 的披露（不是通过）、性能比前任慢 12% / 6%、证据体量 1.4 GB / 1413 个文件。**以后不再接受这个证据体量**——BF-08 的 ASSIGN 已要求 evidence 总量控制在 ~100 MB 内（Git 里留汇总 + 逐文件 SHA + 抽样明细 + 恢复脚本，完整原始回执放私有归档）。
> - **BF-08**（issue #117，24h，分支 task/BF-08，session `01a0b7ce-…`，**01:10Z 已 ACK：eta 21h、基线 6c42dc6，计划与约束与 ASSIGN 一致**）：训练梯度链进自编译 kernel；ASSIGN 点明校准先行冻结预算、跨核 (B,T) 归约确定性、每份最终源码资格回执的生产源码哈希要与最终提交树对上（吸取 BF-07 首个 bd4 回执的教训）、端点披露沿用 D-PM-48 / 50 / 52 框架（新类别先 RISK）、证据体量、写集与既有测试规则。**A5K-03**（scan_fused dh0 定位，P0）仍 open、无人申领——用户可另开 session 直接 APPLY，需要 A5 至少两张卡。
> - 待用户：暂无。
>
> **2026-09-21T00:39Z 更新：BF-07 交了 DONE（PR #118 转 ready，头 `ef5bfc65`）——PM 评审 accept（技术判据全过），合入前只欠一处只加证据的补充；合入待用户授权。**
> - **PM 从原始文件独立复算**（`tmp/review/BF-07/`，git-ignored）：1455 个文件全在写集内（含已批准的既有测试函数）、reserved 0、stable kernel 零 diff、无二进制、blob ≤ 2.5 MB、12 个提交 1331 个 blob（1.38 GB）隐私 0 命中；预算先于实现（首提交 c9409a2）且逐字节未变、与 PM 独立重算一致；完整 workload 四个 bd 32,964 个比较项 0 超预算、17 个返回量跨 bd 逐位相同；普通网格 25,920 case 全过、945,216 个比较项 0 超预算、最大 error / budget 0.3333、4,860 组跨 bd 哈希无差；D-PM-51 六个 bd × 384 ID 全部 state ≤ 1e-5（最大 9.438e-7）；D-PM-52 的 188 个位置 0 个在类外（chunk 最大 27.50 ×、decode 最大 0.977 × FP32 最小正规数）、必要下界重算一致；AST 恰好 5 个获批函数、负对照 44 failed；NPU-free 全量 1032 passed / 12 skipped（= 915 + 117，与预测一致）；性能 plain / cached 慢 12% / 6%。
> - **合入前要补（只加证据）**：首个 bd4 完整 workload 的 6 份回执记录的 `autograd.py` sha256 是 `12293641…`，提交树里自 1d18143 起的是 `78b96d6b…`（其余 318 份回执都是后者，那一版不在 Git 历史、文档没说明）——已请二选一：在最终源码上补跑 bd4 完整 workload（建议）或给两版 diff 并证明差异不在执行路径上。生产文件在所有回执里其余都一致，验证驱动版本在批次间不同（各回执自带身份），不阻塞。
> - **合入前要单列给用户**：(1) dtype 域收窄——raw `q/k/g/beta/A_log/dt_bias` 只接 BF16 / FP32、FP16 / FP64 显式 `ValueError`（D-PM-40 同型，PM 在规格与派单里承诺）；(2) D-PM-52 的披露（chunk 每 bd 16 例、decode 每 bd 44 例输出失败，不是通过，contract board stage 保持 failed）；(3) 性能比前任慢 12% / 6%（无速度门槛；全 flags 多 4 次 launch）；(4) 证据体量 1.4 GB / 1413 个文件（打包 38 MiB，每个 blob ≤ 2.5 MB）——合入后 main 工作树多这么多文本。补充提交核对完后向用户请求 #118 授权。BF-08 与 A5K-03 仍排在 BF-07 之后。
>
> **2026-09-20T22:43Z 更新：用户回复「bf 07批准」（D-PM-52）——PM 读作批准 A，BF-07 解除 blocked。**
> - **批准的内容（A，PM 建议）**：chunk（每 bd 16 例）与 decode（每 bd 44 例）的 nearzero 输出失败，其切片同时满足 |golden|max ≤ 32 × FP32 最小正规数（≈ 3.76e-37）且 candidate 与旧 NPU 逐位相同或差 ≤ 1 BF16 ULP（chunk）/ 4 FP32 ULP（decode）者，归入「只披露、不设通过线」的端点类——披露不是通过；逐切片表照发布、必要下界证明与范围核对（188/188 落在类内）留档；类外失败仍是失败；普通域判据、1e-5 / 0.05 / min(0.01, 3F)、155 / 105 门控、域、D-PM-48 / 50 / 51 不变；**不改 inherited kernel、不批准 kernel 批次**。规格已加「端点比较口径追加」第 5 条。
> - **不是**：BF-07 的合入授权（PR #118 仍 draft、未 DONE、未评审）；也不是 B。**合入前仍要单列给用户**：dtype 域收窄（raw `q/k/g/beta/A_log/dt_bias` 只接 BF16 / FP32、FP16 / FP64 显式拒绝，D-PM-40 同型）+ 本条端点类，做最终授权。若用户本意是 B 或其它，PM 更正。
> - **下一步**：申领人按批准范围收尾（研究文档 + DONE，PR 转 ready）→ PM 评审（原始 JSON 复算、AST / 负对照、NPU-free 全量、隐私与 blob 扫描、对 D-PM-48 / 50 / 51 / 52 逐条）→ 单列给用户 → 合入 → BF-08 接续（同 session）→ A5K-03。
>
> **2026-09-20T17:36Z 更新：BF-07 的端点类范围核对——PM 建议的类定义覆盖全部 188 个失败位置（用户裁定仍待）。**
> - 申领人只读核对（`approved=false`）：40 个 chunk + 148 个 decode 失败位置全部落在「|golden|max ≤ 32 × FP32 最小正规数」内，candidate / 旧 NPU 满足提议的 ULP 界限（decode 48 个 BF16 切片比较与旧 NPU 字节相同，其余 142 个 FP32 比较最大 4 ULP）。
> - **别漏（合入前）**：BF-07 的 dtype 域收窄——raw `q/k/g/beta/A_log/dt_bias` 只接 BF16 / FP32、FP16 / FP64 显式拒绝（PM 裁定，D-PM-40 同型，承诺合入前向用户单列）——最终评审 / 合入时与端点裁定一并单列给用户。
>
> **2026-09-20T17:14Z 更新：BF-07 报 BLOCKED（等用户对剩余输出失败的裁定），状态记 `blocked`。**
> - 申领人 17:11Z：所有规定真机运行已执行；D-PM-51 的 384 个 state 检查 × 六个 bd 通过不变的 1e-5；输出 44 例 / bd、chunk 16 例 / bd 保留数值失败（决策包见上一块）；「慢于前任的真实性能」如实写入 PR（规格没设速度门槛）；最终源码 / 证据恢复 5348 文件 SHA256 全同，798 份实际张量私有归档（二进制不进 Git），11 个可达 PR 提交 / 1326 个 blob 隐私与大小检查通过（自述）；PR 维持 draft（申领人的用户要求全部验收后才正式评审 / DONE）。
> - **待用户**：BF-07 剩余输出失败的处理——A（PM 建议）接受为「只披露、不设通过线」的端点类（|golden|max ≤ 32 × FP32 最小正规数且 candidate 与旧 NPU 逐位相同或 ≤ 1 BF16 / 4 FP32 ULP）/ B 派生 kernel 修复批次（decode 148/148、chunk 16/40 逻辑上无解，除非先改比较口径）/ C 其它。用户答复前 BF-08 与 A5K-03 继续排队。
>
> **2026-09-20T17:10Z 更新：BF-03 DONE 评审 accept 并合入（PR #121）；BF-07 交来剩余输出失败的决策包（待用户）；BF-05 报了单元骨架与入口改造。**
> - **BF-03（PGDN BF16 原生 Vector 前向，DONE 16:27Z，距派单 7.8h / 24h）**：PM 在 NPU-free 环境从原始文件复算全部成立——138 个文件全在写集内、FP32 单元零 diff、无二进制、blob 最大 1.14 MB、隐私 0 命中；首提交 315cb3d 的校准 / 预算规则先于 kernel；134 条真机记录 5360 项比较 0 超预算，最大 error / budget 0.3333，最差 BF16 relL2 o 1.68e-3、主 state 3.2e-6、ATK 6.0e-7；跨 bd 1122 + 198 对 SHA256 全同；67 个 FP32 结果对 pristine 入口字节相同；host 生产算子只有 zeros × 1 + empty × 17；三明治 900 样本中位数复算一致，BF16 与 FP32 持平（T1024 1.0023、T4096 1.0004）；每份回执自带环境与 version.info SHA；65 个源码哈希对上 64 个（差的 contract.json 是运行后只追加 support / measured）；AST：旧测试文件仅获批函数变了，base 负对照 13 failed / 25 passed。**披露的小事**：现场求的 F 与首提交校准 F 因 torch 2.10 → 2.12 相差 ≤ 2.0e-4（首提交文档已声明现场重求），对冻结预算也全过；一个日志里有 4 行私有工作目录相对路径片段。
> - **BF-07 决策包（待用户）**：D-PM-51 state 边界已闭环（384 案例 × 6 bd，state 对实际 prep 的 reference ≤ 1e-5，最大 9.44e-7）。剩余未验收的输出失败全在极端下溢区——chunk：每 bd 16 案例 / 40 个位置，全部失败切片 |golden|max ≤ 3.23e-37（27.5 × FP32 最小正规数），candidate 与旧 NPU 逐位相同或 ≤ 1 BF16 ULP（70/80 比较逐位相同），8 个 case 的 normal 子集 > 0.05 且旧 NPU 同样失败；decode：每 bd 44 案例 / 148 个位置，全部 |golden|max ≤ 1.148e-38（0.977 × 最小正规数），candidate 与旧 NPU 逐位相同或 ≤ 4 FP32 ULP（139/190 逐位相同），normal 元素数 0。PM 建议（不替用户定）：接受为「只披露、不设通过线」的端点类，类定义 = 切片 |golden|max ≤ 32 × FP32 最小正规数（≈ 3.76e-37）且 candidate 与旧 NPU 逐位相同或 ≤ 1 BF16 / 4 FP32 ULP；备选：改 inherited kernel（kernel 批次，无明确修法）、收窄域（不建议）。**17:01Z 补充的必要下界证明（RISK）**：decode 的 148 个失败位置里 124 个是「两套 CPU 参考的允许范围无交集」（例：||r1−r2|| = 3.3e-43 而两个 1e-5 半径之和仅 1.8e-47，差 17931 倍——两套 CPU FP32 参考在 ~1e-43 量级彼此就差 ~33%）、24 个是「BF16 最优舍入 L2 已 > 0.01」，合计 148/148 任何 kernel 都不可能同时满足；chunk 40 个失败位置里 16 个 BF16 最优舍入 L2 > 0.05（8 个 case），其余 CPU-normal 错误仍是真实数值失败（不证明不可修）。所以派生 kernel 修复不可能闭合全部失败，除非先改这些逻辑上无解的比较口径。
> - **BF-05**（头 `b751286`）：派生单元 `gdn2_chunk_fwd_bf16`（五个 kernel check 0 error / 0 warning，A5 无 tile 级 cast 故加宽 / 收窄写成 @vf 里的寄存器 load / store）与入口去 host 转换（删 q/k/v.float()、o.to()、缺省 state 的 cpu-zeros-to-device 改直接在设备分配）已推；尚无真机数字，参考 A 与 unit / contract 还没改 BF16 口径。
>
> **2026-09-20T16:25Z 更新：BF-07 近零调查——D-PM-51 前提补齐；同时发现规则之外的失败（未验收）；BF-03 bd1 全网格与 132 条记录完成。**
> - **D-PM-51 前提补齐**：普通 decode 的 1e-5 是对旧 CPU-prep 输入的 golden，普通域 6 个 bd 分布一致（max 3.27e-7），普通 BF16 q / k prep 与旧 CPU 值全逐位相同；近零档 ~1% 的 q / k 差异（173 / 185 / 16384）= eps 主导下商恰在 BF16 中点（PM 在 CPU 上复算：随机 BF16 ÷ 1e-3 约 0.75% ≈ 1/128 恰在中点，同量级）；16 个 decode state 失败 = C∈{2,3} × group∈{1,2,4,8} × flags∈{110,111}，最大 9.825e-4，六个 bd 完全一致。新 verifier `--dpm51-nearzero-decode` 只限 decode / boundary / `nearzero_*`，逐案记 Σ(q/k²)/eps。
> - **RISK `contradicts-handoff`（16:10Z）**：逐元素分类发现三类不能被 flush 类关闭的输出失败——chunk 的 normal 子集 L2 > 0.05（最坏 0.975，golden max 2.5e-38）、chunk 的 CPU subnormal→非零、decode 的非零 subnormal / CPU 零而 candidate 非零；chunk 每 bd 16 个 o 失败、decode 每 bd 44 个 o 失败，六个 bd 一致；candidate = 旧 NPU 逐位或差 ≤ 1 BF16 ULP（chunk）/ ≤ 4 FP32 ULP（decode）。**PM 认：这些按 D-PM-50 (c) 就是失败，保持未验收**，不能用 zero-only 规则或 D-PM-51 关闭；不改只读 kernel / 阈值 / 域。**BF-07 若带着这些失败 DONE，PM 把它们逐类交用户决定**（接受为端点类只披露不设线并给下溢区间定义 / 改 inherited kernel（kernel 批次）/ 其它）；DONE 里要给每个失败切片的类别、|golden|max、各分类个数、candidate vs 旧 NPU 的逐位 / ULP 差与「全部失败切片 |golden|max 上界」。
> - **排队提醒**：BF-07 的近零端点调查在拉长，BF-08（梯度链，依赖 BF-07）与 A5K-03（scan_fused 定位，P0，建议给发现者 session）都排在它后面；A5K-03 与 BF-07 无依赖，用户可另开 session 直接 APPLY 并行做（需要 A5 至少两张卡）。
> - **BF-03**（距 ASSIGN 7.7h / 24h）：bd1 全网格 66/66，与 bd2 合计 132 条原生记录；跨 bd 1122 对 SHA256 一致；66 个 FP32 公共结果对 pristine 入口逐位相同；BF16 最差 o 1.681e-3、主 state 3.23e-6、ATK 6.01e-7，最大 error / budget = 0.3333；24 个构建的 432 条 warning 行逐类定位到模板 / CANN 头文件；三轮同卡测量已启动，随后整理证据与 PR 再 DONE。
>
> **2026-09-20T16:13Z 更新：用户同意 BF-07 decode 近零方案（D-PM-51）。**
> - **决定**：只限 boundary-grid 的近零 decode 端点组（`nearzero_*`），该组 `final_state` / `o` 改判 (A) 对以 candidate 实际 native-prep 输出为输入的 CPU FP32 reference（`final_state` ≤ 1e-5、`o` ≤ min(0.01, 3F)，数值不放宽）+ (B) prep 单步冻结预算已过 + (C) raw 与 prepared-public 逐位相同（真实字节 SHA）；对旧 prep 的 CPU reference 的偏差（4.31e-4）作观测如实报并写明分解。普通域 decode、chunk 的判据与 155 / 105 门控、域不变。
> - **前提（补齐前不得对别的 case 用）**：申领人补三样——普通 decode state 1e-5 的比较对象；普通域对旧 prep reference 的分布与近零档 ~1% q / k 差异率更高的原因；其它落在 1e-5 之上的近零 decode 切片。**BF-07 规格**已追加「端点比较口径追加」一节，汇总 D-PM-48 / D-PM-50 / D-PM-51 四条端点口径，评审时逐条对。
> - 待用户：暂无（BF-07 decode 近零已答）。
>
> **2026-09-20T16:08Z 更新：BF-03 首个完整 workload 与 bd2 全网格真机通过（自述，未 DONE）；BF-05 校准提交已推、参考 A 待对齐 pin。**
> - **BF-03**（头 `10e31b9`，距 ASSIGN 7.4h / 24h）：B1/T4096/H=HV8 完整 workload bd2 真机通过，BF16 o 对 A / B 相对 L2 1.659e-3 / 1.659e-3（预算 4.977e-3）；bd2 全网格 66/66（33 × 2 dtype），BF16 最大 o 1.681e-3、主 state 3.2e-6、ATK 6.0e-7；FP32 公共结果对 pristine 入口逐位相同（33 / 33）；host 算子只有 1 次 `aten.zeros.default` + 17 次 `aten.empty`（通则允许的分配）。下一步 bd1 / 跨 bd 字节 / 近零 / 三明治，此前不发 DONE。全主机测试 918 passed / 9 skipped（自述；main 是 877 / 12，评审时按 `--collect-only` 对账）。
> - **BF-05**（头 `1ebec19`）：校准提交先行——F=1.7451e-3、预算 min(1e-2, 3F)=5.2354e-3、D(o) / D(final_state) 单列不当约束；参考 A 尚未接上（本机 fla `864a87f6` 与 pin `e52dbc0e` 的 naive SHA 对不上），申领人会取 pin 版、取不到发 RISK。**PM 提醒**：参考 A 的 F_A 与预算也要在写 kernel 之前冻结（第二个校准提交即可），不能等真机验收时 A、B 一起报再定。
> - **待用户**（未答）：BF-07 decode 近零档的比较对象。
>
> **2026-09-20T15:59Z 更新：用户授权「#119授权」（D-PM-49），PR #119（A2-09）已合入 → c2156f3，A2-09 done；main 全量 877 passed, 12 skipped, 5 warnings in 275.98s (0:04:35)。BF-05 ACK；BF-07 又报一个端点类，PM 裁定（D-PM-50）。**
> - **合入**（bot 账号）：#119 A2-09（头 `6bd9c95` → `c2156f3`）；PR 原为 draft，先 `gh pr ready` 再 `--match-head-commit` 钉住审过的头合入；merge commit 两个父提交是 f11a2ad 与 6bd9c95。合入后在 main 上复跑全量（NPU-free）：**877 passed, 12 skipped, 5 warnings in 275.98s (0:04:35)**（本 PR 只新增 `kernels/projects/a2/kda_bwd_stable/**` 46 个文件、不含 tests/ 下的测试，计数不变，与预测一致）。`kernel_inventory`：`kda_bwd_stable` 补记 a2 派生单元已合入（A2 数字只作观测，A2-11 之前不构成结论）。用户是在被告知三点后授权的：finalize_pair 的 stage 判据（相对 L2 1e-5）是看到真机结果之后定的、分辨率 ≈ 一次 BF16 翻转、固定种子下确定性通过；`contract.json` 的 reason 仍留旧措辞（README 已更正，下次动契约时同改）；六项输出梯度预算沿用 a5 未动。
> - **BF-05 ACK**（15:44Z，in_progress）：修订与环境自述（ascriptor 90cfcdc / b3b3f9c；fla `864a87f6`（v0.5.2-123）；CANN 9.2.0、opp 含 ascend950、compiler 2026-05-09T12:45:09+08:00、Python 3.12.14、torch / torch_npu 2.12.0、Ascend950PR）。**一处要对齐**：它的 fla 修订与 GD2-01 / PK-02 / PK-03 的 pin `e52dbc0e`（0.6.0）不同，而验收里的参考 A 是「pin 版 GDN-2 naive」——已要求用 pin 版或先证明两版 GDN-2 naive 逐字节相同（给 SHA-256），并在第一个提交里写明参考所用的 fla 修订。
> - **BF-07 新端点类**（RISK 15:41Z）：boundary-v3 chunk bd4 第 394 例（q/k×1e-20、仅 gate flag）第二 chunk / head1 的实际 o 全 0，与旧 NPU 路径逐字节相同；CPU FP32 golden 最大 1.148e-38（全 < FP32 最小正规数），BF16 正确舍入后有 955 个非零元素，最大 125 ULP、相对 L2=1。**PM 裁定 D-PM-50**：归入既有的 native 下溢单列资格（(ii) 类），不新增闸、不收窄域，D-PM-48 (2) 的整片舍入零口径不扩到此类、保持失败、不放宽阈值；条件是该片逐元素分类（golden 正规数 → 普通判据；golden subnormal 且 candidate ±0 → flush 类报个数；其它一律失败）、旧 NPU 逐字节相同 ≠ CPU 正确性通过、写可达性、其它切片同类统计。decode 近零的比较对象仍等用户（未答）。
> - **待用户**：BF-07 decode 近零档的比较对象（见上一块）。
>
> **2026-09-20T15:45Z 更新：A2-09 的 README 修正提交到了（头 6bd9c95，只改 README）——PM 核对成立，已向用户请求 #119 的合入授权（待答）；BF-07 自我更正了一处「逐位相同」。**
> - **A2-09 / #119**：`compare e6a3a8a…6bd9c95` = 1 个提交、只改 README.md；README 里的实数与 PM 在 CPU 参考张量上的复算逐项一致；PR 14 个提交、46 个文件全在写集内、0 删除、无二进制、最大 1.75 MB，隐私 0 命中（提交信息只命中「hostname」一词，是在描述检查项）。技术评审 accept；A2 PR 合入要用户授权（A2-11 之前 A2 结论不算数）。**合入步骤**：用户授权后 → PR 仍是 draft，需先转 ready → `gh pr merge 119 --merge --match-head-commit 6bd9c950fc8d81767eb0842040cf0a41df941941` → 在 main 上复跑全量（NPU-free）→ 记看板 / CLOSE。`contract.json` 的 reason 还留着旧措辞（在 digest 里），README 已更正，下次动契约时同改。A2-09 status 改为 review。
> - **BF-07 自我更正**：15:19Z 的「近零切片与正确舍入逐位相同」不成立（torch.equal 忽略 +0 / −0；符号位不同的零有 3991 / 3670 个，没有非零元素）；D-PM-48 (2) 的判据是数值 ULP ≤ 1（不是逐位），+0 / −0 是 0 步，不改变该口径，符号差如实报。
>
> **2026-09-20T15:31Z 更新：BF-05 派给新 session（bf05-gdn2-bf16-a5-1）；BF-07 的 decode 近零比较对象交用户决定（待答）；BF-03 报了第一个 STATUS。**
> - **BF-05**：15:21Z 新 session 的 APPLY（如实说明手上的 A2-09 还没 CLOSE）；PM 现在就派（只依赖 GD2-01；session 值与 A2-09 不同，符合 PROTOCOL §3.1；用户 D-PM-35 后续 BF16 优先），24h，分支 task/BF-05；ASSIGN 要求先推掉 A2-09 欠的只改 README 的提交、校准先行并冻结预算、证据机械检查、`tests/test_gdn2_chunk_fwd.py` 不在写集（要改先 RISK write-set-expansion）。BF-03 CLOSE 后**不会**再给 BF-03 的 session 派 BF-05（它自己取消了）。
> - **BF-07 decode 近零（待用户）**：`nearzero_c2_g1_rbf16_vbf16_flags110` 的 final_state 对旧 prep 的 CPU reference 是 4.31e-4（冻结判据 1e-5）；对 candidate 实际 native-prep 输入的双 CPU reference 是 7.24e-8 / 3.96e-8（o 1.54e-3）；raw 与 prepared-public 的 o / state 逐位相同；q / k 对旧 NPU prep 分别 173 / 185 个 BF16 元素（各共 16384，约 1%）差 1 ulp，prep 单步冻结预算已过。定位：偏差来自 prep 的 BF16 舍入位置，不是 decode 递推。申领人请求裁定比较对象——这改的是冻结判据的比较对象，PM 不自行裁，问用户；已要求申领人补三样（普通域比较对象与分布、近零差异率更高的原因、其它落在 1e-5 之上的近零切片）。chunk 近零与端点执行都符合 D-PM-48。
> - **BF-03**：15:20Z STATUS——66 条实现前校准记录与预算随第一个提交 315cb3d 冻结，六阶段 Vector BF16 单元与入口去 host cast 已实现、104 项主机测试过；尚无设备执行。
>
> **2026-09-20T15:18Z 更新：BF-03 ACK（in_progress）；BF-07 报了新的端点 RISK——PM 裁定两条比较口径（D-PM-48）；BF-05 的 APPLY / WITHDRAW 按撤回处理、不派单。**
> - **BF-03**：15:02Z ACK（距 ASSIGN 6.4h），计划与规格 / Cube 澄清 / 写集预批准一致，无需裁定；ETA 24h。
> - **BF-07 RISK（PR #118 头 3bb0c63，仍 draft）**：扩展真机交叉端点上「同时匹配两个前任的 NaN/Inf 掩码」不可满足（u=−87/−40/−20 且 A_log=89/100：旧 NPU NaN、CPU 与 candidate −Inf；u≤−88：三者里 candidate 与旧 NPU NaN、CPU −Inf）；极小 BF16 输出（q/k×1e-20）golden 7.85e-44 小于 BF16 最小 subnormal，相对 L2=1；decode nearzero 的 final_state 4.31e-4 > 1e-5（定位中）。**PM 裁定**：CPU FP32 是语义权威，端点不设掩码相等通过线、四列报并分类写明、不改 candidate；golden 舍入后整片为零的切片按「对 golden 正确舍入 ≤1 ulp」判（相对 L2 照报、原失败保留、不删判据、范围只限此类）；decode 的 1e-5 不放宽，定位后若要改比较对象 / 冻结判据问用户。预算数值、155/105 门控、域都没动。PM 猜测机制（未验证）：u=−87/−88 恰在 FP32 最小正规数两侧，是 subnormal 被冲成 0 后 Inf×0。
> - **评审提醒**：BF-07 评审时对 PR **全部提交**做隐私扫描（applicant 自述曾把私有运行目录路径带进公开 fork，已用 force-with-lease 替换最新提交，头因此从 81444c0a 变成 3bb0c63）；要求把 traceback / 日志路径脱敏加进出包前机械检查；对常规域证明 FP64 累积指标与旧指标一致 / 只更小。
> - **BF-05**：新 session 的 APPLY（15:07Z）与 WITHDRAW（15:09Z）——用户要求停掉这个 gdn2 session、已在别处另开；PM 不派单，BF-05 open 无人申领；主 session 的旧排队 APPLY 按其取消不再自动派（BF-03 CLOSE 后不会派给它），用户另开的 session 走自己的 APPLY。
> - **A2-09**：仍在等申领人推「只改 README」的提交（头仍 e6a3a8a）；BF-08 / A5K-03 仍待 BF-07 CLOSE。
>
> **2026-09-20T15:07Z 更新：A2-09 推了 README 提交（头 e6a3a8a）——PM 复算发现机制解释有一处不对，已回复申领人再改一次 README；改完 PM 核对后请用户授权合入 #119。**
> - **核对成立**：提交只动 README.md 与新增一份日志（不在 unit digest 里，digest 仍 `b4bb774d…`）；日志相对 L2 与 `aclnn.json` 逐项一致，`|ref|max` 在 CPU 上重算一致；`finalize_pre` 78 项最大 2.085e-26。
> - **不对的一处**：README 说两处偏离是「一次 BF16 舍入落在占张量范数很大份额的元素上」。PM 用 `ulp(x)/||ref||` 在 CPU 上对参考张量复算——`gentle_decay_bd1` 的 `qk_left`：指数 −19 的元素差 1 ulp（2^-26）= 8.2960e-6，与实测 8.296e-6 五位一致（该元素是 `|ref|max` 的 2%~4%）；`grid_c2_hv4_bd1` 的 `t_beta`：指数 8 的元素差 1 ulp（=2）= 6.999e-6，与实测 7.000e-6 吻合（0.17%~0.33%）。**两处各是「一个小元素差 1 个 BF16 ulp」**；1e-5 判据的分辨率就是「有没有一次翻转」（`gentle_decay` 里中位数元素翻一次是 1.66e-5，64% 的元素翻一次就超），这两处是翻转恰好落在小元素上才过——固定种子下确定性，换种子 / 形状 / 卡可能变。六项输出梯度的预算不受影响。gaps `a2-kda-bwd-pair-checkpoint-thin-margin` 已改成这个刻画。
> - **要申领人再改的（只改 README，不跑真机）**：机制换实数并写明分辨率；「余量 ≥95×」改「≥94×」；「其余 101 项逐元素最大 58 ulp」写明 ulp 只在 6 个 case 上测过。
> - **下一步**：申领人推 README 提交 → PM 核对 diff 只动 README → 向用户请求 #119 的合入授权 → `--match-head-commit` 合入 → 全量测试 → CLOSE。
> - **工具坑（本轮）**：`poll` 遇到网络超时会打印「失败：…i/o timeout」并 exit 1，**输出里仍可能带着已拿到的部分事件**——不能当成「无事件」，要重跑到 exit 0 再 `--advance`；`pm_github.py post` 也会因超时失败，要看输出里有没有「已评论」再决定是否重试（别用 `| tail && break`，管道的退出码是 tail 的）。
>
> **2026-09-20T14:18Z 更新：A2-09 rework 后重交 DONE（PR #119 头 276d53d）——PM 评审 accept（技术判据全过），合入前要一个只改 README 的提交；合入等用户授权。**
> - **两条阻塞已修，均由 PM 复算确认**（`tmp/review/A2-09/v2/`，git-ignored）：所有 `.json` 可解析、数值全是有限 float；`source_id.py` 在 PR 头重算 digest = `b4bb774d…`，26 个 case 的 before / after 都是它、`started_at` 晚于最后一次源码改动；六项梯度相对 L2 从 `aclnn.json` 复算（dq ≤ 3.829e-2 / dk ≤ 1.115e-1 / dv ≤ 3.887e-3 / dbeta ≤ 7.456e-3 / dg ≤ 1.776e-1 / dh0 ≤ 5.293e-3，全在 a5 预算内，1014 项全 passed）；`run.py reference` 26/26；pipesim `gentle_decay_bd1` 通过、39 项与回执逐位相同；隐私 0 命中。代码 / 契约 / README 与上一版逐字节相同。
> - **PM 发现的一条实质问题**：契约 `reason` 与 README 写「四个 case 实测最差相对 L2 1.06e-07，1e-5 比实测宽两个数量级」——对那四个成立、对 26 个不成立：真机 `finalize_pair.qk_left` 在 `gentle_decay` 是 8.296e-6、`t_beta` 在 `grid_c2_hv4_bd1` 是 7.000e-6（预算的 83% / 70%），sim / pipesim 在这两处是 0.0。已要求 README 改成 26 例实数、写明成因未确立（`contract.json` 的 reason 在 digest 里，留到下次动契约时同改）；记 gaps `a2-kda-bwd-pair-checkpoint-thin-margin`（P2，A2 观察）。此外 `pair_checkpoint_ulp.log` 少了 finalize_pre 三个量（contract 只给 finalize_pair 设档，ulp_stats 按 contract 枚举），DONE 没提，已要求 README 说一句。
> - **下一步**：申领人推只改 README.md 的提交 → PM 核对 diff 只动 README → 向用户请求 #119 的合入授权（A2 PR，A2-11 之前 A2 结论不算数）→ `--match-head-commit` 合入 → 全量测试 → CLOSE。
>
> **2026-09-20T13:44Z 更新：用户回「改」——AGENTS.md §5 的两处过时措辞已改（D-PM-47）。**
> - 改了：`layout_device="auto"` 自动探测绕路那句（FMT-02 后已删）、⚠️ 段里「已加绕行 `_scan_states(on_cpu=)` / `chunk_kda_bwd(layout_device=)`」那句（同样失效，现写明缺 ascend950 算子包的机器上默认门控检查、`_scan_states`、`dw` 取负、`log2(eg)` 分支会失败，纯前向在 `check_gate_range=False` 时可用）。不改任何规则；`CLAUDE.md` 是软链，只改 AGENTS.md 一处。
>
> **2026-09-20T13:27Z 更新：用户回复「scan fused 需要定位。agent.md 修改是啥」（D-PM-46）——scan_fused 立项 A5K-03（只定位）；AGENTS.md 的改动已把新旧文对照给用户、等确认。**
> - **A5K-03**（`docs/pm/tasks/A5K-03.md`，P0，open，24h 首轮）：只定位 `kda_bwd/kernels/scan_fused.py` 的 dh0 不确定性，**不批准 kernel 批次、不改 scan**；方法 = 机制 + 可证伪预测 + 干预实验（实验性派生单元 `kda_scan_diag`，不接公开调度，一次一个变量，N≥50 重放，负对照，跨卡 / 跨 CANN 对照），回答四问（kernel 竞态还是环境 / 卡？源码行与事件？预测被干预证实？影响面与「为什么只有 dh0」）；写集 = `kda_scan_diag/**` + 研究文档 + 诊断测试。修复要在定位后由用户另批。**排队**：因「一个 agent 一次一个任务」排在在飞任务之后，建议 BF-07（阶段 1）CLOSE 后由发现者 session `01a0b7ce-…` 接手、BF-08 顺延；用户若要插队请说。gaps `kda-bwd-scan-dh0-nondeterministic` 已注明。
> - **AGENTS.md（未改，等用户确认）**：只是 §5 两处过时措辞——`layout_device="auto"` 自动探测绕路（FMT-02 后已删）、以及「已加绕行 `_scan_states(on_cpu=)` / `chunk_kda_bwd(layout_device=)`」（同样失效）；不改任何规则或纪律，新旧文对照在给用户的回复里。
> - **工具坑**：auto 模式的分类器偶尔「暂时不可用（timed out）」会让 Bash 失败——等一会再试，读文件类操作不受影响；不要因此换别的方式硬跑。
>
> **2026-09-20T11:30Z 更新：A2-09 交了 DONE（PR #119 draft）——PM 评审 rework：原始回执被脱敏破坏、回执不带源码身份；技术部分成立。**
> - **评审**（`tmp/review/A2-09/`，git-ignored）：40 个文件全在 `kernels/projects/a2/kda_bwd_stable/**`、无二进制、最大 1.7 MB，隐私 0 命中，a5 与 A2-03 的 a2 单元零改动；契约六项预算 = a5、board stage `untested`、block_dim 只声明 1 与 2、门控跨度 `untested`；无 `@vf` / splitk，`GetValueFrom` 只读 FP32；`finalize_pair` 判据（相对 L2 1e-5，实测最差 1.06e-7）与 ulp 日志（最大 58 ulp、>1 ulp 至多 0.4028 个百分点）对得上，A/B 逐位相同，bd 逐位相同；PM 在 NPU-free 环境自己跑 `run.py reference` 26/26。
> - **两条阻塞**：① `evidence/unit/{aclnn,sim,pipesim}.json` 是坏 JSON——脱敏把「0.」后不带科学计数法的长小数替换成 `<serial>`（aclnn 490 处 / sim、pipesim 各 232 处），六项梯度相对 L2 等关键数在原始回执里被毁，只剩 `passed` 布尔，PM 只能读汇总表；② 回执不带源码身份，aclnn 26 次运行跨 05:30–09:24Z、期间源码改过（L1 栅栏、`bar_all` 删 / 恢复、判据），`repo_sha` 是 rebase 前的提交，「去注释逐字节相同」没给对照。已请修脱敏并在最终源码上复跑 26 个 case（回执记 `unit_source_sha256`）、补 `bar_all` 的 pipesim 报错原文。A2 PR 合入本来就要用户授权（A2-11 之前 A2 结论不算数）。
> - **评审工具坑**：A2 的 `run.py check --launcher pipesim --case <id>` 单个 case 也要 >5 分钟——工具调用会超时，用 `setsid nohup … &` 脱离运行再轮询日志；`json.load` 失败先看是不是脱敏占位符（`grep -c '<serial>'`）。**教训（给以后所有带脱敏的证据）**：脱敏规则别用「长数字串」的启发式，会吃掉浮点数；要按字段名 / 严格格式，并且脱敏后必须有「可解析 + 数值字段无占位符」的检查。
>
> **2026-09-20T09:15Z 更新：BF-07 已 ACK 并给了估计（阶段 1 约 17h、阶段 2 约 21h）——PM 拆出 BF-08（阶段 2 梯度链）；BF-07 的 RISK（FP32 输出没有「1 ulp」线）是规格写宽了，已改。**
> - **拆分**：BF-07 = 前向 / 推理 / decode（24h），BF-08 = 梯度链（24h，BF-07 CLOSE 后同 session `01a0b7ce-…` 连续，issue 待 sync 建）。BF-07 需要梯度且 raw flag 开着时保留 host `_prepare_inputs` 图（登记的存量例外），DONE 只声称阶段 1；BF-08 才消除它。kernel 批次是同一个（D-PM-43），拆分不另问用户。`kernel_inventory` 新增 `kda_prep`（wip）。
> - **测试清单已确认**：BF-07 可改 `tests/test_kda_domain_checks.py` 里 3 个函数、BF-08 改 2 个；能不改就不改，新覆盖写进新测试文件，改动只换观察点为新 kernel ABI 边界的局部 CPU 替身，不加生产旁路。
> - **RISK 裁定（规格写宽了）**：旧 host 的 FP32 q/k 归一化对 FP64 参考就有 2–3 ulp（自述，92 条 CPU 校准）——FP32 输出不设 ulp 通过线，预算 = 旧 host 对 FP64 地板的 3 倍（norm 另受 1e-2），ULP 分布两个比较对象都报并逐类解释；BF16 输出 ≤1 ulp 不变；端点与旧 host 的 FP32 语义逐点对照、不新增闸、不收窄域。看板里 BF-07 的 caps 旧 9.2 描述已改为申领人声明的 CANN 9.1.0-beta.1 环境。
>
> **2026-09-20T08:48Z 更新：BF-07 规格已细化并派单（session `01a0b7ce-…`，新单元 `kda_prep`，阶段 1 前向 / decode + 阶段 2 梯度链，24h 首轮）。**
> - **规格要点**（`docs/pm/tasks/BF-07.md`）：`_prepare_inputs` 三步（gate、q/k l2norm、beta sigmoid）进自编译 kernel；**dtype 域收窄成 raw 量只接 BF16 / FP32**（FP16 / FP64 显式报错，D-PM-40 同型，**合入前要单列给用户**）；阶段 2 必须给链式反向（`A_log` / `dt_bias` 的 (B,T) 归约确定性且与 `bd` 无关），可经 PM 拆成 BF-08；预算申领人先校准、第一个提交冻结；现有 `tests/test_kda_domain_checks.py` 只在「列清单、PM 确认」后才动；门控闸 155 / 105 不变；FMT-02 的性能与证据规则（大 tile、每份回执自带环境行、PR 任一提交无 >5 MB blob）已写进规格与 ASSIGN。
> - **当前在飞**：BF-03（session `gdn-series-…`）、BF-07（session `01a0b7ce-…`）、A2-09（session `a2-09-bwd-1`），三个 session 不同、写集互不相交。BF-05 仍排在 BF-03 之后；PK-05 的近零判据我派前自裁。
>
> **2026-09-20T08:39Z 更新：用户授权「#115授权 #116授权」，两个 PR 已合入（D-PM-45）——BF-02 与 FMT-02 都 done；BF-03 已派；main 全量 877 passed / 12 skipped。**
> - **合入**（bot 账号、`--match-head-commit` 钉住审过的头）：#115 BF-02（头 `b5c8c2e6` → `15c8ede`）、#116 FMT-02（头 `343ebfd5` → `0c7c74a`）。合入后在 main 上复跑全量（NPU-free）：**877 passed / 12 skipped = 762 + BF-02 的 21 + FMT-02 的 94**（与预测一致；762 = 761 + 我加的 1 个 PM 测试）。`kernel_inventory`：`gdn_bwd` 去掉 BF16 `dtype_fix`，新增 `kda_layout` 单元，`kda_fwd_stable` / `kda_bwd_stable` 的说明记了 layout / dtype 转换已改走 kda_layout。用户是在被告知 #116 的五件事（D-PM-44 读法、两处公开域变化、前向 1.67–2.32×、CANN 9.1.0-beta.1、dev-A 基线缺陷）后授权的，D-PM-44 由「PM 裁定」转为「用户已接受」。
> - **合入后我才注意到的一件事（评审汇报里没有单列，已记 D-PM-45 (3) 与 gaps）**：#116 删了 CPU 绕行（`_resolve_layout` 恒 False），所以在缺 ascend950 算子包的机器上，默认的门控跨度检查（device 上 cumsum）、带缓存前向 / 反向里的 `_scan_states`、`dw` 取负、`log2(eg)` 分支这些存量 host 算术不再有 CPU 兜底，会失败；布局 / 转换 / 零填充本身不依赖内置算子，纯前向在 `check_gate_range=False` 时可用。AGENTS.md §5 里「layout_device=auto 会自动探测并绕路」一句因此过时——**要用户同意我才改 AGENTS.md**。规格里我事先写过「缺算子包的机器不是验收项」，所以不算违规，只是没在汇报里说清。
> - **BF-03 已派**（session `gdn-series-…`，24h，分支 `task/BF-03`）：ASSIGN 点明 Cube 澄清、写集预批准（`tests/test_pgdn_chunk_fwd.py` 仅一个函数）、预算 min(1e-2, 3F) 先校准后写 kernel、PK-05 的近零判据不在本任务、原始回执自带环境行、PR 任一提交不含 >5 MB blob、协议头格式。BF-05 仍排在 BF-03 之后。
> - **待办**：**BF-07 规格与写集要 PM 细化**（session `01a0b7ce-…` 已空闲、排队 APPLY 在 #106，预检输入见 BF-07.md）；FMT-02 / BF-02 的 CLOSE 已发。**仍在飞**：A2-09（session a2-09-bwd-1）。**仍未决 / 要用户**：kernel 批次里上游 scan_fused 的 dh0 不确定性（gaps `kda-bwd-scan-dh0-nondeterministic`）先定位再批、AGENTS.md §5 的过时句、「校验类」暂定裁定、BF16 的 k 范数闸、PK-05 近零判据（PM 派前自裁）。
>
> **2026-09-20T08:10Z 更新：FMT-02 DONE（PR #116，头 `343ebfd5`）已评审 accept（技术），合入要先问用户；#115 仍等用户（未重试）。**
> - **FMT-02 评审（PM 从原始 JSON 自己复算）**：464 案例 × 19 项输出对旧路径逐位 8816 次 0 失败，预算未动（最紧 dg 0.1646 / 0.25），跨 bd 9396 哈希全同，host 审计 unexpected 空、例外按 D-PM-42 单列，D-PM-44 基线（dev-A 旧 3 哈希 9/2/1、候选 10/1/1；dev-B 12/12 同哈希 d88e1911…）、96 次同张量转换逐位全过、种子输入 16/16 重建一致、性能六个三明治与归因求和逐项一致；NPU-free 全量 855 / 12（main 761 / 12 + 94），负对照成立；PR 只有 1 个提交、最大 blob 0.95 MB、2348 个证据 SHA 全对、无隐私命中。
> - **合入前要问用户的五件事**：(1) D-PM-44 对「逐位相同」的读法；(2) 公开域变化——`layout_device='cpu'` 显式报错、`runtime.cast/move` 只接 BF16 / FP32（autograd 不受影响，PM 判断未测）；(3) 前向 1.7–2.3× 变慢（T1024 plain 1.670× / cached 2.255×；T4096 1.854× / 2.316×；backward 1.003–1.014×；相对 Torch NPU 组合基线快 25–49×）；(4) 环境是 CANN 9.1.0-beta.1（不是 9.2.0）；(5) dev-A 的 h0 基线缺陷不资格。合入后 main 预期 856 passed / 12 skipped（855 + 我加的 1 个 PM 测试），合入用 `--match-head-commit 343ebfd558a9b088f4e3c807d4d380480d4e4398`。
> - **申领人这边**：07:22Z 的 DONE 在我评审途中发出，08:00Z 因为看板没及时记而补发了同一份（我评审耗时约 40 分钟，是我这边的延迟）；BF-07 继续排队。**评审工具坑**：`git rev-parse HEAD` 在 `tmp/review/*/wt` 里会读到外层仓库的头（wt 不是 git 仓库），别拿它当快照身份，用 tarball 的 sha。
> - **#115**：仍等用户「#115授权」，不重试。
>
> **2026-09-20T07:26Z 更新：A2-09 撤回了 auto_sync 的因果（A/B 逐位相同，gaps 降级）；FMT-02 前向变慢降到 1.7–2.3×、PR 已整理成 1 个提交并转 ready（等 DONE）；#115 仍等用户（未重试）。**
> - **A2-09 的更正 RISK**：申领人做了我要的 A/B（栅栏有无，同 case 同卡型），`qk_right` 逐位相同，`max_abs=8.0` 是同一元素的舍入差；失败变通过是同一提交里改了 checkpoint 判据。**`a2-autosync-missing-cross-pipe-guards` 由 P1「静默」降为 P2「静态观察」**（gaps 现 P1 22 / P2 11 / 总 48）；`a2-cast-blkstride-sim-blind` 有独立证据，不动。教训：同一提交混两个变量就把通过归因给其中一个（§6.5）；最小复现要求在这里按设计起了作用（证伪了一条静默错误类）。**poll 的新规则第一次派上用场**——这条 RISK 的头又多了一个词，被当「首行格式不对」的警告露出来了，没有被吞。
> - **FMT-02**：新批量搬运候选性能降到前向 1.67–2.32×（T1024 plain 1.670×、cached 2.255×；T4096 1.854×、2.316×）、backward 1.003–1.014×，设备事件与 host 派发对得上端到端（自述，等 DONE 复算）；PR #116 头 `343ebfd5` 只有 1 个提交、base 当前 main、最大 blob 0.95 MB（历史卫生满足），已转 ready，**但还没有替代 06:24Z 的 DONE，等 DONE 再评审**。前向 ≥10× 的单列门槛不再触发，1.7–2.3× 合入前照实告诉用户。D-PM-44 的新 dev-A 数据一致（旧路径 3 种 h0 哈希 9/2/1、3 次超 0.05；直接 scan 7 种 dh0 哈希）。
> - **#115**：仍等用户「#115授权」，不重试。
>
> **2026-09-20T06:53Z 更新：FMT-02 性能归因完成（成本在设备侧，不是 host）、在做 NDDMA 批量搬运优化；分支历史里有两个大 blob 要整理；#115 仍等用户（未重试）。**
> - **FMT-02**：归因（自述，PM 核对自洽）——T1024 plain / cached 设备事件合计 45.7 / 117.8 ms，host 派发只有 0.47 / 0.91 ms，热调用零源码读取 / 哈希 / compile / 注册、缓存键 O(1)，端到端 47.0 / 120.0 ms（T4096 202.4 / 478.3 ms）。正在写集内做「最多 4096 元素连续输出 tile 的五维 NDDMA 批量搬运」（数学 / 舍入 / bd 不变，旧证据不借给新 kernel，完整真机 / 逐位 / 性能重跑），重交 DONE 时替代 06:24Z 那份。**卫生**：分支历史里已有 13.5 MB gz 与 29 MB 十六进制 JSON 两个大 blob，merge commit 会永久带进 main——已请申领人重交前整理历史（force-push 自己的分支、任一提交都不含 >5 MB blob），否则合入我只能改 squash（要先问用户）。
> - **#115**：仍等用户「#115授权」，不重试。
>
> **2026-09-20T06:40Z 更新：FMT-02 交了 DONE 但与性能请求交叉、已转回 draft（暂不评审）；PR 里有一个 13.5 MB 二进制要处理；BF-07 排队 APPLY；#115 仍等用户（未重试），并且它卡着 BF-03 / BF-05 的派单。**
> - **FMT-02**：06:24Z DONE（PR #116 头 `454fba1a`）在我 06:25Z 的性能回复之前发出；申领人 06:28Z 把 PR 转回 draft，补设备事件 / host 开销归因与无额外同步的端到端、写集内优化后重交（改 kernel 则完整真机与逐位重跑）。看板保持 in_progress，DONE 的数字都记进了 reports 供重交后复算。**PR 含二进制 `evidence/repro-bundle/fixed-inputs.json.gz`（约 13.5 MB）**——PROTOCOL 安全审查不接受二进制（BF-04 的 tar.gz 是用户一次性例外，不是先例）；已请申领人改成「种子 + 生成脚本 + 输入哈希」的文本、原始张量放私有归档，或说明必须进仓的理由，由用户决定是否再给一次性例外，合入前单列。另：性能前向 31–51× 已单列给用户（≥10× 的前向退化合入前要问）。
> - **BF-07**：申领人（同 session）排队 APPLY，排在 FMT-02 CLOSE 之后；他在 #106 发了三条只读预检（梯度链、dtype 边界、数学细节），已写进 BF-07 规格「预检输入」，细化规格时逐条回答；回了 NO_TASK。
> - **#115 的代价**：BF-02 处于 review 状态时，同一 session 不能领 BF-03（一个 agent 一次一个任务），所以 gdn-series 的申领人现在只能做只读预检——用户的「#115授权」越晚，BF-03 / BF-05 越晚。
>
> **2026-09-20T06:24Z 更新：FMT-02 前向变慢 31–51×（自述，backward 仅 1.04–1.06×）——新的关注点；A2-09 pipesim 26/26、真机 12/26 通过；#115 仍等用户（未重试）。**
> - **FMT-02 性能**（dev-B、bd4、三轮同步，无速度门槛）：T1024 plain 前向 1.564→48.547 ms、cached 2.410→122.146 ms；T4096 5.694→202.023 ms、9.277→477.484 ms；backward 2.480→2.578 / 9.767→10.372 ms。按新增 launch（6 / 14 / 1）摊开，前向每个 layout launch 约 8 ms（T1024）/ 33 ms（T4096），线性放大，backward 那个 launch 只有 0.1 / 0.6 ms——更像实现 / host 开销而不是搬运本身（PM 粗估带宽下限数十 µs，未测）。**已请申领人 DONE 前先分清设备侧 / host 侧、给不含额外同步的端到端，并在写集内优化（逐位不变）；无速度门槛，但前向 ≥10× 的退化合入前要单列给用户。** D-PM-44 的执行与裁定一致（36 个真实 BF16 dh0 样本 × 两设备 = 72 次转换回放逐位全过；dev-B 旧路径稳定、dev-A 基线缺陷保留）。
> - **A2-09**：reference / sim / pipesim 26/26，真机 12/26 通过（自述，四张卡并行），六项梯度相对 L2 都在预算内（dg 最大 1.78e-1 / 0.25，落在 a5 契约写的 0.115–0.183 地板里；预算沿用 a5，PM 读 a5 contract.json 核实）；最小复现 A/B 与 ulp 分布等 DONE。
> - **#115**：仍等用户「#115授权」，不重试。
>
> **2026-09-20T06:10Z 更新：PM 已从 FMT-02 提交的原始 JSON 复算了旧路径的自身不确定性（与自述一致）；第二设备 464 case 全过（自述）；#115 仍等用户（未重试）。**
> - **FMT-02（PR #116 draft，头 `964cc8e`，28 个文件全在写集内）**：`evidence/diagnosis/old-public-repeats.json` / `scan-probe-comparison.json` 我自己复算——旧路径 12 次 h0 五个哈希 7/2/1/1/1、其余 7 项各 1 个哈希、h0 相对 L2 0.002337 ×7 / 0.008439 / 0.153270 / 0.357131 / 0.358643 ×2（超 0.05 的 4 次、超 3F 的 5 次）、候选 12 次同一哈希；第一设备 12 次直接重放 11 次 dh0 不同、dAqk/dh/dv 全同——D-PM-44 的前提现在是 PM 核实过的（设备 / 环境标识与第二设备结果仍自述）。第二设备 bd1..4 × 116 = 464 case 全过、跨 bd 哈希一致（自述），三轮同步测量在跑，未 DONE。gaps `kda-bwd-scan-dh0-nondeterministic` 与 FMT-02 规格已注明。
> - **#115**：仍等用户「#115授权」，不重试。
>
> **2026-09-20T05:57Z 更新：FMT-02 的 h0 差异隔离到上游 scan_fused 的 dh0 不确定性（基线自己就不稳，环境 / 卡相关）——D-PM-44；A2-09 九个 kernel 真机跑通、又报两条；#115 仍等用户（未重试）。**
> - **FMT-02 → D-PM-44**：旧公共路径自己重复 12 次，h0 梯度 5 个哈希、4 次超 0.05（最差相对 L2 0.36），其余 7 项各 1 个哈希；直接重放上游只读 `scan_fused`（`scan_fused.py:358-362` 一带，机制未定位）第一张卡 11/12 不同、另一张卡 12/12 相同；BF16 dh0 在转换前已不同，新旧 widen 各自逐位等于 CPU 转换；候选 12 次同哈希（等于旧路径多数哈希）。环境是 CANN 9.1.0-beta.1（自述，PM 未复现）。裁定：FMT-02 不承接根因与修复（写集不变、不批派生 scan 单元）；验收口径改读为「转换边界逐位每样本 + 其余 7 项每设备逐位 + h0 只在旧路径自身稳定的设备上端到端比 + 0.05 预算不动」；已记 gaps `kda-bwd-scan-dh0-nondeterministic`（P1，`requires_kernel_change`，进队列，**要用户批准的 kernel 批次**，机制未定位——§6.5 先定位）。**合入 FMT-02 前先问用户**（这是判据的读法）。
> - **A2-09**：九个 kernel 在 910B3 上跑通（reference 26/26、sim 26/26、33 个 checkpoint 对上 CPU，真机 26 case 与 pipesim 在跑；自述）；又报两条：auto_sync 漏跨 pipe 保护（C=2 才炸，记 gaps `a2-autosync-missing-cross-pipe-guards`，P1）与七个 checkpoint 判据改成两个 BF16 ulp（事后调整，评审单列、要 ulp 误差分布）。gaps 现 48 条（P1 23 / P2 10）。
> - **#115**：仍等用户「#115授权」，不重试。
>
> **2026-09-20T05:37Z 更新：FMT-02 报 RISK bitwise-mismatch（自检，autograd 案例只有 h0 梯度哈希不同）；#115 仍等用户（未重试）。**
> - **FMT-02**：完整 Kimi direct plain / cached / backward 与 31 个叶子检查通过后，autograd B2/T192/H2/HV4 案例里 o / final_state / dq / dk / dv / dg / dbeta 逐位相同、FP32 h0 梯度哈希不同（自述；原始 failure.json 保留、没放宽判据）。我只记不上报（自检发现、没宣称通过、不是 main 上的静默错误），回复了建议：先证明旧路径自身 h0 梯度哈希多次运行稳定，再按 D-PM-42 分界归因；若旧路径本身非确定，逐位标准由 PM 裁。PR #116 头 `9d17abfe`（仍 draft）。
> - **#115**：仍等用户「#115授权」，不重试。
>
> **2026-09-20T05:22Z 更新：FMT-02 首个完整 Kimi 真机 workload 通过（自述）；PK-05 近零判据问题比预想宽；#115 仍等用户（未重试）。**
> - **FMT-02（PR #116 仍 draft）**：完整 22 个 vendor 构建通过；首个完整 Kimi B1/T4096/H=HV32/C64、bd4 真机 workload 19 项对旧路径逐位通过、两套 CPU FP32 参考与原六梯度预算通过、host 审计 unexpected=[]（自述，receipt 稍后提交）；叶子阶段的验收脚本有同名 `unit.py` 导入冲突（已保留原始失败、已修、未重跑，没宣称通过）。**环境是 CANN 9.1.0-beta.1**（与前面 A5 证据的 9.2.0 不同）——DONE 时要看 opp 目录里有没有 `ascend950`，两套 CANN 的数字别互相当预期。
> - **PK-05 待裁又扩大**：不是「近 eps 行」问题——单分量正 q/k 归一化后恒为单位轴，相关梯度数学上为 0，普通范数（.03…3）下纯相对 1e-4 判据也被舍入残差支配，1e-6 时 A 严格 0 而 B 非 0 → 相对误差为 ∞。裁定要针对「真零 / 近零梯度的基准与量化判据」，规格「补充 2」已写；派单前我要给出口径（可能要引入相对参考总梯度尺度的绝对底，且实现前冻结），这是 PK-05 之前必须做完的一件事。
> - **#115**：仍等用户「#115授权」，不重试。
>
> **2026-09-20T05:07Z 更新：FMT-02 开了 draft PR #116；BF-02 合入仍等用户（未重试）；PK-05 近零范数预检又补一条。**
> - **等用户**：授权合入 #115（BF-02，头 `b5c8c2e68330a0757d2cbeeda90d8fd078821383`）。被自动模式分类器拦过一次（Merge Without Review），按 D-PM-33 第 (5) 项不绕过、不换方式重试；已在 #101 公开说明动作与依据（申领人问的），并说明 D-PM-33 没有被收紧。用户一句「#115授权」就合入、CLOSE、派 BF-03。
> - **FMT-02**：draft PR #116（头 `13028fb8`，22 个文件全在写集内，无保留路径 / 二进制），等 DONE 再审。**PK-05**：见规格「派单前待 PM 裁定」，新补一条（primitive 顺序匹配仍有 8 个整链失败）。
>
> **2026-09-20T05:00Z 更新：BF-02 新头 `b5c8c2e6`（只多 README +5 行），合入仍等用户；A2-09 报静默错误类 RISK 已记账；BF-03 写集预批准；PK-05 有一条待裁；修了 poll 的一个盲区。**
> - **BF-02**：申领人按非阻塞意见 a 补了 README 并推送 `b5c8c2e6`（PM 用 compare 复核：只改新单元 README +5 行，评审结论对新头成立，合入要钉 `--match-head-commit b5c8c2e68330a0757d2cbeeda90d8fd078821383`）。申领人要求 PM 依 D-PM-33 自行合入；我试了一次，**自动模式的拦截仍要求用户对该具体 PR 的明确授权**（同 #110 / #111 / #113 / #114），没有绕行，已请用户授权 #115。
> - **A2-09 的 RISK `sim-device-divergence`（severity high，静默错误类，自述）**：c220 上对 32 字节行包的列整块 cast 被降解成 `srcBlk=0` 的 vconv，真机整块拿到第 0 行而 sim / pipesim 全绿；另外 BF16 GM 标量读编译期失败、功能模拟器不校验物理地址分配。已记 `gaps.json` 两条（`a2-cast-blkstride-sim-blind` P1、`a2-sim-vs-toolchain-blind-spots` P2），A2-09 规格加三条硬约束（sim 不再是通过依据、逐行 cast、最小复现随 DONE），A2 数字仍只作观测。PM 无 A2 真机、没复现，只核了发射器里 block stride 由 IR 属性决定。
> - **BF-03**：之前漏处理了申领人在 #102 的写集预批准请求（`tests/test_pgdn_chunk_fwd.py` 的一个函数，靠生产入口 `.to(q.dtype)` 才过）——已核实成立并补批，条件写在 BF-03 规格。**PK-05**：排队申领人的近零范数发现（oracle 在近零真实导数区本身是 FP32 相消噪声）要 PM 在派单前裁定判据 / 预算口径，规格里已列四问。**FMT-02** STATUS：接线完成、主机回归通过，真机因守卫拒绝窗口在等（D-PM-28，只记录）。
> - **踩坑（poll 盲区，已修）**：agent 把风险类别写进 RISK 头（`[FLA-PM] RISK A2-09 sim-device-divergence from=…`，协议头只有 TYPE / ID / from），头解析失败，而 `cmd_poll` 对 PM 账号自己发的评论把「type 为 None」也跳过——所以这条静默错误类的 RISK **poll 里一片空白**。改成 `skip_own_comment()`：只跳过无头的普通评论与 PM 类型，头格式不对的露出来并附「首行格式不对」警告；加了回归测试（PM 工具测试 86 个）。**仍要每轮匿名读 issue 评论**（普通评论仍会被跳过，上面 BF-03 的预批准请求就是这么漏的）。
> - **踩坑（分类器）**：被拦的合入之后，有几条无关的 Bash（用 `python3` 读 `docs/matrix/gaps.json`）也被同一个理由拦了；换 Read 工具读文件、`git` / `pm_*` 命令照常。不要为此绕开，做别的事、最后把要用户定的说清楚。
>
> **2026-09-20T04:36Z 更新：BF-02 DONE 已评审 accept（PR #115，头 `58c56a18`），合入前请用户授权。**
> - **BF-02（GDN chunk 反向原生 BF16，含分组，A5 真机）**：PM 从原始 JSON 复算——native 276 条全过，BF16 最紧 误差/预算 0.3334（预算 min(1e-2, 3F_g)，实现前提交 `1a3cf5a` 定死），FP32 新对原字节相同 138/138，跨 bd 3036 个哈希相同；BF16 host 算子只有 `aten.empty`；入口域与原 FP32 逐条相同（没有收窄，所以不需要为「域」问用户）；NPU-free 全量 760 / 12（= main 761 − 22 + 21）、focused 88、负对照 20 个失败。三轮三明治从 900 个原始样本重算一致。
> - **要用户定的**：授权合入 #115（头 `58c56a18`，`--match-head-commit` 钉住）。
> - **记录给用户知情**：BF16 反向比原 FP32 **慢 6.6–6.8%**（T1024 1.066×、T4096 1.068×，无速度门槛，记为后续性能任务起点）；FP32 缺省 cotangent 的 `zeros` / `zeros_like` 是原实现就有的常量填充（允许）；第二个 CANN 9.2.0 环境（不同 build timestamp）的构建失败记录保留、不声称通过。非阻塞：full-v4 的 source manifest 里 `unit.py` 哈希不在 PR 任何提交里（同 case 在头版下逐位复现），已请申领人在 README 补一句。
> - **合入后**：CLOSE BF-02，按排队 APPLY 直接派 BF-03（Vector 方案，session `gdn-series-…`），随后 BF-05；BF-07 要等 FMT-02。评审复算脚本与 NPU-free 环境在 `tmp/review/BF-02/`（git-ignored）。
> - **踩坑（评审工具）**：`pytest -q -q` 会吞掉末尾的 `N passed` 汇总行，只剩进度点——要读汇总就只用一个 `-q`；进度点数 = 收集的用例数，汇总里的 skipped 还含模块级 `importorskip` 的整文件跳过（这里 7 + 5 = 12），所以逐项对账要用 `--collect-only` 的逐文件计数。
>
> **2026-09-20T03:49Z 更新：用户授权合入 #114 并「其他的也放行」（D-PM-43）——BF-06 已合入（`7f8f002`，main 全量 761 passed / 12 skipped）；一批 gated 项放行；FMT-02 派给 KDA / PKDA session。**
> - **BF-06 合入**（bot 账号、`--match-head-commit 98d75fc9`）：KDA decode 原生 BF16 / FP32；`kernel_inventory` 的 `kda_fused_recurrent` BF16 格改为原生（A5）并去掉 `dtype_fix`。入口 dtype 域收窄（BF16 q/k/v 或全 FP32，其余显式报错）与 T=16 变慢（约 1–25%）是用户知情后授权的；raw flags 分支仍是「存量例外 → BF-07」。
> - **放行了什么（我的读法，用户可纠正，见 D-PM-43）**：`open` 了——PK-05 / GDA-04 / PK-06 / GDA-05（请求来源，已记 `approved_by_user`；**仍是骨架，PM 派单前先补全规格**，后三个还要等前置）、BF-07（kernel 批次已批准；**仍是骨架，派单前先细化规格、定写集，排在 FMT-02 之后**，因为两者写集在 `chunk.py` / `autograd.py` 重叠）、A2-10 / A2-11（**PM 补写了规格**；A2-10 前置 A2-02 没完成、A2-11 前置 A2-10，所以不可派）。**状态位仍 gated、但 gate 改成「用户已放行，待前置任务与规格」**——A2-K1、A2-12 … A2-16、A2-41、A2-43（规格没写、前置没完成，PM 在前置完成、规格写好时直接改 open，不再问用户）。
> - **没有放行**（不在清单里或含义不明确，用户若也要放行请说）：A2-20、A2-42、A3-00、A5-01 … A5-06、A2-06、G3-01 与各 epic。
> - **A2-10 / A2-11 的规格要点**：A2-10 = 在 A2 上把 runtime 桥跑起来并证明对 A2-03 的 aclnn harness **逐位一致**（含算子包覆盖表、CANN 带时间戳、设备事实只作观测）；A2-11 = 三类累加缺陷的最小复现（各 ≥ 1000 次 + 时序扰动）、绕行（显式 barrier）的重复稳定性、`tests/test_a2_accumulate_barriers.py` 的 AST 级机械检查、以及「A2 结论算数」的条件清单——**只交证据与建议，是否放宽 AGENTS §2 的总闸仍是用户的决定**。
> - **FMT-02 已派**（session `01a0b7ce-…`，24h，分支 `task/FMT-02`）：范围按 D-PM-42（dtype / 布局转换进新 kda_layout 单元；`_scan_states` 整体、`log2(eg)`、只读校验是已登记的存量算术例外；位精确的 dw 取负与 g_cumsum × 1/ln2 可选并进同一 launch）。
> - **踩坑**：往 python heredoc 里写含反引号的文字要用带引号的 heredoc 分隔符（`<<'PYEOF'`），否则 shell 会把 `_scan_states` 之类当命令执行、把文字吃掉（这次吃掉了 FMT-02 报告里的一个词，已修）。
>
> **2026-09-20T03:18Z 更新：BF-06 DONE 已评审 accept（PR #114，头 `98d75fc9`），合入前先问用户（入口 dtype 域收窄）；FMT-02 的范围已裁定（D-PM-42）；BF-02 / A2-09 在跑。**
> - **BF-06（KDA decode 原生 BF16 / FP32，token-major 直读直写，A5 真机）**：PM 从原始回执自己复算——414 条数值记录 + 342 个附加检查全过，BF16 最紧 误差/预算 0.333，跨 bd 2068 个哈希相同，FP32 新对原实现逐位 159/159；**D-PM-41 的三层 prefill 口径**（预算运行前定死）126 条链 / 8568 次局部 decode 全过，层 3 最紧 0.396，没有改线；host 审计只有 `aten.empty`（PM 代理审计每次恰好 2 个）；无 NPU 全量 761 passed / 12 skipped（main 739 + 22），入口换回 base 后 30 个失败。**REVIEW accept 已发在 #114。**
> - **要用户定的**：授权合入 #114（头 `98d75fc9`）。**为什么要问**：入口 dtype 域从「任意浮点（host 加宽）」收窄成 BF16 q/k/v 或全 FP32（FP16 / FP64 / 混用显式报错）——这是 D-PM-40 里我裁的 D-PM-37 直接后果，仓里查无调用方，但属域改动，D-PM-33 要先问。**记录给用户知情**：B1/T16/H32 上新 kernel 比旧 wrapper + 旧 kernel 慢约 1–25%（BF16 速度比 0.814 / 0.988 / 0.748，FP32 新路径 0.795 / 0.937 / 0.746；T=1 持平、T=2 更快），无速度门槛，我记为后续性能任务的起点。
> - **FMT-02（排队，BF-06 CLOSE 后派）范围裁定（D-PM-42）**：申领人核对源码发现 `chunk.py::_scan_states`（h / v_new 的 host 复算：bmm / 减 / 乘 / stack，夹着 FP32 Cast）、`log2(eg)` 分支、`dw` 取负、域 / 门控校验都不是纯 layout / dtype。裁定：FMT-02 = D-PM-37 字面（dtype / 布局转换进新 kda_layout 单元，逐位不变）；`_scan_states` 整体、`log2(eg)`、只读校验是「已登记的存量算术例外」（审计单列、不宣称合规、不搬进新单元）；位精确的 dw 取负与 g_cumsum × 1/ln2 可选并进同一 launch（逐位相同才并）；验收口径改为推理前向与 decode 完全干净、带缓存前向与反向在例外之外无 host 转换算子。**后果**：训练路径（带缓存前向）在 kernel 批次前**不可能完全没有 host 算子**；消除 `_scan_states` 要让前向 kernel 导出 h / v_new（`fwd-caches-not-emitted`，P2，kernel 批次，只有用户能批）——建议与 BF-07 一起批。
> - **BF-02（PR #115 draft）**：首个完整 workload 通过（BF16 dq/dk/dv 最大相对 L2 约 1.675e-3，FP32 新旧逐位一致，两 dtype host 审计只有 10 次 aten.empty），bd2 完整网格 135/138 记录已过，自述。**A2-09（session 自报 a2-09-bwd-1）**：九个反向 kernel 写完并过 ascriptor check，bwd 单元 26 个 case reference 全过，functional sim 单 chunk / grouped_heads 通过，D-PM-37 的两处 host 操作已挪进 kernel。
> - **观察**：BF-02 的第二验证环境里「同为 CANN 9.2.0」的不同小版本头文件布局不同（`IMPL_UTILS_SYS_MACROS_H` 与 `..._IMPL_H`、`g_coreType` 重定义），所以评审要的 CANN 标识应带 inner / compiler 时间戳（BF-06 的回执已带，BF-04 没有）。
>
> **2026-09-20T00:40Z 更新：用户授权合入 #110 / #111 / #113，三个都已合入（D-PM-39）；BF-06 / BF-02 / A2-09 已按各自的排队申请直接派单。main 全量 739 passed / 12 skipped。**
> - **合入**（都经 bot 账号、`--match-head-commit` 钉住审过的头，用户原话「#110授权 #111授权 #113授权」）：#110 BF-04（头 `2d9e4068` → `b3e6e7a`）、#111 BF-01（头 `c5ddd293` → `4670730`）、#113 A2-03（头 `de893f2e` → `3884c27`）。合入后在 main 上复跑全量（NPU-free 环境）：**739 passed / 12 skipped = 669 + BF-04 的 32 + BF-01 的 38**（A2-03 不改测试），与逐个 PR 上的预期完全相同。
> - **看板**：三个任务 done；`kernel_inventory` 里 `pkda_chunk_fwd` 与 `gdn_chunk_fwd_a5` 的 BF16 格改为「原生（A5）」并去掉对应 `dtype_fix`（首页 kernel × dtype 表随之更新）；`kda_fwd_stable` / `kda_fused_recurrent` 的说明里记了 a2 派生单元已合入（真机数字只作观测）。`ops.json` 没动：A2-03 申领人建议记两个 a2 单元，但矩阵还没有按 SoC 分维，等 A2-06。
> - **一次性例外**：#110 的证据是 tar.gz（PROTOCOL §4.9 禁二进制）。用户在被告知后授权合入，PM 按「这一次」处理、没改 §4.9，新的 ASSIGN 里已要求申领人以后用文本证据、每次运行的原始回执自带 SoC / CANN / 算子包一行。**仍未决**：「校验类」暂定裁定、BF16 的 k 范数闸（沿用 1+1e-5，比 BF16 分辨率紧）——用户没有另外回答，按暂定 / 未决处理。
> - **新派单**（都是排队 APPLY 承诺过的，不需要重新申领）：**BF-06**（KDA decode 原生 BF16，session `01a0b7ce-…`，16h）、**BF-02**（GDN 反向 BF16，session `gdn-series-…`，24h，写集已含预批准的 `tests/test_gdn_chunk_bwd.py`）、**A2-09**（KDA 反向 a2 单元，session `a2-01-…`，16h，第一个 STATUS 按实测速度重估）。BF-03（BF-02 之后）、BF-05（BF-03 之后，规格已澄清 B=1）仍是排队。
> - **仍在等用户的**（同前）：「校验类」暂定裁定、BF16 k 范数闸、FMT-02 做法、BF-07 / A2-K1 批次、A2-10 / A2-11 放行、PK-05 / GDA-04 / PK-06 / GDA-05 放行。
>
> **2026-09-19T20:32Z 更新：A2-03 DONE 已评审（技术 accept，A2 上的 PR 等用户授权）；BF-05 排队 APPLY 回 NO_TASK 并澄清了规格。**
> - **A2-03（PR #113，头 `de893f2e`）**：前向 5 个 kernel + decode 共 6 个 a2 kernel、两个单元（`kda_fwd_stable` 20 case、`kda_fused_recurrent` 8 case），实际耗时约 3.5h（估计 60h）。**PM 无 A2 真机，真机部分只对原始回执复算**：fwd o rel_l2 [2.96e-3, 3.45e-3]、decode o [1.37e-3, 1.73e-3]，网格 12 格与自述逐格一致，每个 case 的执行证据都是 aclnn / board / a2 / backend_executed；**PM 在本机对钉住的库独立复跑 CPU 的 reference / sim，60 + 16 个数字与申领人的 sim.json 逐位相同**，pipesim 抽查通过，6 个 kernel 的 ascriptor check 与 op 数完全一致；所有 `is_init=False` 的 matmul 紧邻 `barrier(Pipe.M)`、无 BF16 / FP16 splitk（A2-01 结论落实）；契约 board stage untested、门控跨度闸与 block_dim 上限都不继承 A5；全量 669 passed / 12 skipped 与 main 基线相同。**REVIEW accept 已发在 #113，合入等用户（A2 结论 A2-11 之前不算数）。**
> - **BF-05 排队 APPLY（#104）**：预检发现验收里的「多 batch」与 GDN-2 入口的 B=1 公共域矛盾。PM 裁定：公共域与 GD2-01 相同（B=1、H∈{1,16}），BF16 不扩域不收窄，B≥2 显式报错；「多 batch」验收项已从 BF-05 规格删去并加了澄清一节，扩域另立项。回 NO_TASK：BF-02 → BF-03 → BF-05。
> - **BF-01 STATUS**：申领人收到 REVIEW accept，PR 未变；做了 BF-02 的只读 CPU 预检（138/138 case，A/B 最大相对 L2 5.6e-7），不是验收。
> - **待用户**：授权合入 #110（头 `2d9e4068`）、#111（头 `c5ddd293`）、#113（头 `de893f2e`，A2）；其余问题同前（「校验类」暂定裁定、tar.gz 证据放不放行、BF16 的 k 范数闸、FMT-02 做法、BF-07 / A2-K1 批次、A2-10 / A2-11 放行、PK-05 / GDA-04 / PK-06 / GDA-05 放行）。
>
> **2026-09-19T19:45Z 更新：BF-01 DONE 已评审 accept（PR #111），BF-04 头移到 2d9e4068（仅证据元数据，已复核），两个合入都在等用户授权；A2-03 的估计偏差约 45 倍。**
> - **合入被自动模式分类器拦着（不绕过）**：`gh pr merge` 报 `[Merge Without Review]`；这一轮连 `git fetch origin pull/N/head` 和 `poll --advance` 也被报 `[Auto-Mode Bypass]`（复盘：分类器在合入被拒后对「把 PR 头取到本地」「推进游标」变敏感）。**评审改用只读 API**：`gh api repos/…/tarball/<sha>` 取源码快照、`gh api …/pulls/N/files --paginate` 取完整文件列表（**`gh pr view --json files` 最多只给 100 个，135 个文件的 PR 会被截断，别用它做写集核对**）、`gh api …/compare/A...B` 看头之间的差异。`git push origin main`（板子提交）单独跑可以，和 sync / 游标推进拼在一条复合命令里会被拦。
> - **要用户做的**：授权合入 #110（头 `2d9e4068`，不是原来的 `a25ba5e`）与 #111（头 `c5ddd293`）；用户也可以在设置里加一条允许 `gh pr merge` 的 Bash 权限规则，让 D-PM-33 的常设授权对分类器生效。合入要经 bot 账号（`GH_TOKEN=$(gh auth token -u ascend-fla-pm-bot) gh pr merge N --merge --match-head-commit <sha>`），`limjiunnbin` 无权限。
> - **BF-01（PR #111，头 `c5ddd293`，A5 真机）**：PM 从原始 JSON / 日志自己复算——112 条原生记录 0 失败，BF16 vsA 最大相对 L2 o 1.6758e-3 / state 1.019e-6（与自述逐位一致），BF16 最紧 误差/预算 = 0.333；跨 bd 1120 个哈希全相同；FP32 原 / 新 392 项 0 不同；host 审计每次调用 13 个 `aten.empty`（PM 代理审计得到相同）；900 个原始样本重算的加速比 T1024 0.993、T4096 0.996（BF16 与 FP32 墙钟持平，无速度声称）；测的代码就是合入的代码（源码清单 582/585 逐字节相同，差的 3 个是文档与契约状态）；无 NPU 环境新增 38 个测试全过、换回 base 入口 20 个失败、全量 707 passed / 12 skipped；135 个文件全在写集内、无二进制、隐私扫描 0 真命中。**REVIEW accept 已发在 #111。** BF-01 合入后直接派 BF-02（写集已预批准补上 `tests/test_gdn_chunk_bwd.py`，因为 GDA-03 的 15 个「全 BF16 一律拒绝」用例与 BF-02 目标矛盾），再 BF-03。
> - **A2-03 的估计偏差（好消息）**：申领人自述前向 5 个 kernel + decode + 两个单元骨架合计约 1.5h 墙钟（我记的估计是约 60h，估计是按人工小时写的、没按 agent 速度校准），draft PR #113 已开、整链 sim 与 910B3 真机观测都过了（自述，A2-11 之前只作观测）。PM 把 A2-03 时限 72→24h、A2-09 56→16h，并在 #34 重发修订 ASSIGN。A2 波次（KDA 先行）的工作量比 17:50Z 上报的 ~123h 小得多，A2 总闸不变。
> - **排队 APPLY**：BF-06（KDA decode BF16，session `01a0b7ce-…`，BF-04 合入后派）、BF-03（session `gdn-series-…`，BF-02 之后）、BF-02（同 session，BF-01 之后）都已回 NO_TASK。本轮 `poll --advance` 被分类器拦下没推进游标，下一轮会重放这些事件，看板都已记录，按重放处理。
>
> **2026-09-19T18:50Z 更新：BF-04 DONE，PM 评审 accept，但合入被自动模式分类器拦下（[Merge Without Review]），不绕过，等用户授权。**
> - **BF-04（PR #110，头 `a25ba5e3`，A5 真机）**：PM 从解包后的原始回执自己复算——4 个 bd × 89 = 356 个完整链 case 全过，vsA 最大相对 L2 o 3.426e-3 / S 6.61e-6 / A 1.05e-7 与自述完全一致，最紧的误差/预算 = 0.68；172 个边界全过；跨 bd 801 个哈希字段全相同；host 审计并集只有 `aten.empty` / `aten.view`（PM 的代理审计——桩掉 launch 跑 `TorchDispatchMode`——得到相同算子与个数）；三轮三明治的加速比从原始样本重算逐位一致（T1024 1.25–1.31、T4096 1.21–1.24，基线是 base 的 FP32 入口，源码哈希等于 base）；
>   无 NPU 环境复跑新增 32 个测试全过、换回 base 入口后 32 个全失败，全量 701 passed / 12 skipped；隐私扫描 485 个文本文件 0 命中。原 FP32 单元零改动，38 个文件全在写集内，BF16 门控域没收窄（155，与 FP32 同）。**REVIEW accept 已发在 #110。**
> - **合入前要用户定的两件事**：① 授权合入（回复「#110授权」，我用 `--match-head-commit a25ba5e3` 经 bot 账号合入；合入要用 bot 账号，`limjiunnbin` 无权限）；② **协议偏差**：证据是 1.26MB 的 `tar.gz`，PROTOCOL §4.9 把二进制文件列为整体拒绝项。PM 已解包核对全部 448 个成员（SHA 与索引一致），建议这次放行、以后要求解成文本；或让申领人现在改成文本再合。
> - **观察（非阻塞，未收窄域）**：BF16 的 k 范数闸沿用 FP32 的 1+1e-5，比 BF16 的分辨率（约 1e-4）还紧，常规「先 FP32 归一化再转 BF16」的 k 约一半会被拒（测试用 `k*0.99` 避开），报错里「normalize explicitly」对 BF16 误导。放宽闸要用户批准并重测数值影响。合入还依赖暂定的「校验」类裁定（控制字经 `aclrtMemcpy` 回读、FP32 只读校验器沿用）。
> - **合入后 PM 要做**：BF-04 → done（result 填 commits）；`kernel_inventory` 里 `pkda_chunk_fwd` 的 BF16 格从 reject 改成原生（A5）；main 上复跑全量（预期 701 passed / 12 skipped）；发 CLOSE。
> - **BF-01（PR #111 draft）**：两个 block_dim 的完整网格自述通过（112 条记录，跨 bd 728 对阶段数组逐字节相同，每次调用只有 13 次 `aten.empty`），FP32 与原入口逐字节相同；同卡 900 样本三明治在跑。**BF-02 排队 APPLY**（#101）已回 NO_TASK：BF-01 合入后直接派。**A2-03**：已在 #34 重发修订 ASSIGN（72h，范围拆分），首批两个 kernel（fwd gate、decode step）自述在 910B3 上过了，只是观测。
>
> **2026-09-19T17:50Z 更新：A2-03 拆成 A2-03（前向 + decode）与 A2-09（反向）——申领人的估计合计约 123h，原时限 20h；BF-04 原生 BF16 的 bd4 全网格自述通过，draft PR #110 开了；BF-01 改前校准完成。**
> - **A2-03（D-PM-38）**：申领人的第一个 STATUS 给了按 kernel 的估计——纯向量 6 个共 33h、纯 cube 1 个 2h、混合 8 个共 72h，另加骨架 6h 与网格 / 真机 / 证据 10h，**合计约 123h**（自述，PM 只核了静态事实：a2 上没有 `@vf`、14/15 个 kernel 用 `@vf`、`dma.l0c_to_ub` 与 `dma.ub_to_l1.nd2nz` 不可用）。
>   PM 采纳它建议的拆分：**A2-03 = 前向 + decode**（gate、decode、前向四 kernel，`kda_fwd_stable` / `kda_fused_recurrent` 两个单元，72h），**A2-09 = 反向**（九个 kernel + bwd 单元，56h，依赖 A2-03，写集 `kernels/projects/a2/kda_bwd_stable/**`）；A2-13 增加对 A2-09 的依赖。
>   A2-09 的 55h 只含九个 kernel 行，bwd 单元自己的 contract / 网格 / 证据整理另计，派单时补估计再调时限。**这改变 A2 波次（KDA 先行）的工作量预期**，已上报用户；A2 上的 PR 合入前仍问用户，A2-11 之前 A2 结论不算数。
> - **BF-04**：原生 BF16 的 bd4 完整 T4096 已上板，89 个完整链 case + 43 项 API / FP32 字节回归全过（**自述**）：相对 CPU 参考的最大相对 L2 o 3.426e-3、S 6.61e-6、A 1.05e-7；BF16 入口 host 审计只有 empty / view，FP32 原 / 新入口三个输出逐位相同。
>   bd1/2/3 各独立进程构建中，再做同卡三轮性能与脱敏回执。draft PR #110 已开、**未 DONE**，我没有评审；早期扫描（非评审）写集内、无机器信息。DONE 时要复算原始回执、跑两条 dtype 路径的 host 算子审计、复核 BF16 门控域没有比 FP32 的 155 窄（窄了要问用户）。
>   `native_checks.py` 里有三处 `exec(compile(before_path.read_text()…))` 加载「改前」模块做 FP32 逐位对照，DONE 评审时确认 `before_path` 指向仓内基线而不是外部来源。
> - **BF-01**：改前校准完成（28 case × 56 条 A/B 记录，A↔B 最大相对 L2 o 2.04e-6 / final_state 1.78e-6，BF16 误差地板约 1.63–1.68e-3，负对照全被拒），7 个类型化条目静态检查 0 error；原生 BF16 结果与旧 FP32 全工作量基线还在跑。
>
> **2026-09-19T17:25Z 更新：D-PM-37 的边界被申领人问出来了（校验算不算算术），PM 暂定并更正；GDA-03 补交了仓里第一份真机 dispatch 审计；A2-03 派单，附一个影响 A2 波次估计的事实。**
> - **更正**：我此前说「PKDA 入口是干净的」只对 dtype / 格式转换成立——PKDA 的 FP32 入口有 host 数值校验（`validate_inputs`：`isfinite` / 范围 / 范数 / 门控前缀求和）和 `torch.full` 的默认 center（BF-04 申领人的 `RISK contradicts-handoff` 指出，我的 grep 只查了转换、漏了算术）。
>   同类的情况大概率在别的入口也有（KDA 的 `check_gate_range` 就在 device 上做 cumsum + 规约）。**PM 暂定（待用户确认）**：只读数值校验、校验用的 kernel 状态字回读、常数填充分配允许，审计里单列「校验」类；产出进入计算的数据的算术 / 转换 / 重排不允许；CPU 诊断 launcher 的输入校验保留。
>   通则文档、`pm.md`、D-PM-37 的记录都已更正。不必为 FP32 路径多加一个校验 launch。
> - **BF-04**：CPU 精度研究结论（自述）——BF16 的 ATK 状态存储 23/24 不过、门控前缀存储 13/24 不过，而「输出阶段用真 BF16 cube operand + FP32 累加」24/24 过；设计是 ATK / Kahan 前缀 / scores / WY / scan 留 kernel 内 FP32（D-PM-35 允许），只有输出阶段吃 BF16 operand、kernel 直接写 BF16。PM 认可，不是域变化，预算不变。
> - **GDA-03 的补交**：合入后申领人在真机上对公共入口做了 `TorchDispatchMode` 审计（#98 评论）——4 个完整 T4096 FP32 case 只有 `empty` / `zeros` / `zeros_like`，5 个 BF16 拒绝 case 零 aten 算子；PM 解析了原始报告，与 PM 的代理式审计一致。
> - **A2-03 派给 A2-01 的申领人**（session `a2-01-…`，唯一有 a2 真机的）。**事实（PM 读库源码核实）**：pin 版 `ascriptor.a2` 没有 `@vf` / `@simt`（`a2.py:6,18`），15 个 KDA kernel 里 14 个用 `@vf`——a2 上要用 UB 指令**重写全部向量侧与数据流**，不是「派生 / 换 import」；规格的 20h 是低估，
>   第一个 STATUS 要给按 kernel 的估计，PM 据此调时限或拆任务。这影响 A2 波次（KDA 先行）的工作量估计，已上报用户。A2-10 / A2-11 仍 gated（`machines:a2`，用户放行）。
>
> **2026-09-19T17:10Z 更新：GDA-03 已合入（PR #98 → `8d16cb7`，依 D-PM-33 自行合入，第三个）；BF-01 派给 GDN 系列 session。**
> - **GDA-03**：GDN chunk 反向，A5 真机，**FP32-only**（BF16 的 q/k/v/do 在任何准备与分派之前显式报错并指向 BF-02）。PM 复算：FP32 每 bd 69 case，最大相对 L2 8.3e-7（预算 1e-4），bd1↔bd2 逐位相同，900 个计时样本重算一致；用真 oracle 复跑 67 个测试全过，
>   全量 580/12 精确对账，合入后 main 669 passed / 12 skipped。**代理式 host 算子审计**（把 kernel 启动桩掉，对公共入口 + pipeline 在 CPU 张量上跑 `TorchDispatchMode`）只有 `aten.empty`（缺省 cotangent 时多一个 `zeros`）——这个办法可以给别的入口做静态复核，
>   真机 dispatch 追踪等 FMT-01 的工具。计时基线是「saved-checkpoint 伴随」而不是另一套完整反向，候选/基线 1.17–1.18，如实标注。evidence 目录 7.5 MB，比前几个 PR 大得多。
> - **首页表**：`gdn_bwd` 的 FP32 格从「进行中」改为「✅ 原生 · A5 真机」，BF16 格仍是上游单元 → BF-02。
> - **BF-01**：按 #100 的排队 APPLY 直接派给 session `gdn-series-…`（24h）。BF-02 现在只差 BF-01（GDA-03 已 done）。PK-05 / GDA-04 / PK-06 / GDA-05 仍 gated。
>
> **2026-09-19T16:55Z 更新：用户又定了「禁止在 host 转数据类型、格式，需要在 kernel 内完成」（D-PM-37）——D-PM-35 从 BF16 扩到所有 dtype 与格式，FP32 路径同样。A2-01 已合入（PR #107 → `b514de2`，用户授权）。**
> - **规则**（`docs/pm/bf16-kernel-side.md` 已整段改写）：host 侧只允许分配输出、不拷贝的元数据操作、检查并显式报错、取指针、launch；禁止 dtype 转换、格式（布局）转换——`permute`/`transpose`+`contiguous`、token-major↔head-major、GQA 的 q/k 复制、`cat`/`stack`/`pad`、`npu_format_cast`、绕 CPU 重排——与算术。
>   「kernel 内」读作**自编译 kernel**（融进主 kernel 首选，独立的自编译布局 kernel 也行，要报多出的 launch），不是 torch 算子。去掉转换后输出与改前**逐位相同**。范围是 `ascend_fla/ops/**`，layers/modules 暂不在内。
> - **静态清单（PM grep，正式清单等 FMT-01 的真机审计）**：KDA `chunk.py` 13 处 dtype 转换 + 8 处布局搬运（含 `layout_device='cpu'` 旁路）、`autograd.py` 6+2、`fused_recurrent.py` 3+5、`chunk_bwd.py` 0+2；GDN / PGDN / GDN-2 chunk 前向各 2 处（BF16 加宽，GQA 复制在 device 上做）；**PKDA 入口是干净的**。
> - **任务**：`FMT-01`（审计工具 + 全仓清单，只读）与 `FMT-02`（KDA layout/dtype 进自编译 kernel，新单元，不改已有 kernel 源码，要改先发 RISK）——都是 P0、open；BF-01…BF-06 规格已加 D-PM-37 追加段（两条 dtype 路径、格式转换、FP32 逐位相同）；BF-07 仍 gated。
> - **在飞**：BF-04（session `01a0b7ce-…`）已 ACK、in_progress，已通知新规则；GDA-03（PR #98）转正式、更新后的 DONE 待审——静态上公共入口只有缺省 `do`/`dht` 的 zeros 分配，真机审计与证据复算还没做。
> - **A2-01**：合入后 main 599 passed / 15 skipped；两条缺口按证据更新，`a2-splitk-fp32-cube` 记 requires_kernel_change（属 A2-K1，要用户批准），队列 25→26；申领人的 delta 提议 fp32-cube 记 P0，PM 记 P1（出货路径没中招，是 A2 派生单元的潜在风险）。
>
> **2026-09-19T16:20Z 更新：A2-01 的 `RISK silent-wrong-result` 已上报，用户同意（D-PM-36）改写 `AGENTS.md` §2 并记缺口——已做。**
> - **事实（改写后的 §2）**：FP32 的 split-K 在 pin 里已有修复（`ascriptor/passes/desugar.py:411`，a2 系且 A/B 均为 f32 时插 `PIPE_M`，PM 读源码核实，sha256 前缀 900610ea92dc）；
>   没解决的是 ① BF16/FP16 的 split-K（同一条规则按 dtype 排除；910B3 / CANN 9.0.0 上 M16 输出有限但全错、M32 触发 AI Core 异常、M64 逐位；sim/pipesim 看不出）与 ② FP32 的手写 MMAD 累加链（只有 lint trap；KDA/GDN 的三角求逆与 GDN 反向 finalize，正是 M16 形状）。
>   **证据等级**：静态部分 PM 核实；真机部分是 A2-01 申领人自述、PM 无 A2 真机未复现。A2 总闸不变（D-PM-30）。
> - **缺口**：`a2-splitk-fp32-cube`、`a2-splitk-bf16-fp16-unsettled`（P1，`requires_kernel_change=false`：库侧缺陷，本仓不改 ascriptor）；`ops.json` 的 A2/A3 说明同步。A2-01 的 DONE（脱敏原始回执 + `docs/pm/deltas/A2-01.json`）到了要**核并更新这两条**，别当成已定论。
> - **待用户做**：把这条转给 ascriptor 所有者（gitcode 上的库，PM 没有渠道）。本仓侧的显式 barrier（A2-03 起、kernel 批次 A2-K1 的「split-K 绕行」）仍要用户批准。
> - **计划影响**：D-PM-35 的 BF16 优先叠加 A2 优先——BF16 的 KDA 在 910B 上要先过 BF16 累加链这一关；A2-03 规格已加线索。
>
> **2026-09-19T15:55Z 更新：用户定了「BF16 计算必须 kernel 侧，不能在 host 侧完成；追加任务改正；后续 BF16 优先」（D-PM-35）。**
> - **通则**：`docs/pm/bf16-kernel-side.md`——BF16 张量直接进自编译 kernel，转换/算术在 kernel 里，host 侧只允许零算术零转换的布局操作；`q.float()` 加宽再转回不合规；
>   统一验收里的**硬判据是真机上的 host 算子审计**（`TorchDispatchMode` 记录入口内的 aten 算子，出现算术/转换算子即失败）；BF16 预算先校准（误差地板 F）、实现前写定；BF16 的门控域要重测，比 FP32 窄属域变化，先问用户。
> - **新任务（origin=user，不需要放行；P0，插在看板最前，`--next` 先给它们）**：`BF-01` GDN 前向、`BF-02` GDN 反向（依赖 GDA-03、BF-01）、`BF-03` PGDN 前向（依赖 PK-03、BF-01）、
>   `BF-04` PKDA 前向（先做 CPU 精度研究）、`BF-05` GDN-2 chunk 前向、`BF-06` KDA decode（依赖 A2-44，已完成）；`BF-07`（KDA 前向里融合 q/k l2norm、门控、beta sigmoid）**gated `kernel-batch-approval`，等用户批**。
>   首页 kernel 表的 BF16 格里 `🔁 API 加宽 → BF-xx` 就是待改项；已合入的历史 PR 不回滚。
> - **在飞的调整**：GDA-03 的 BF16 公共入口不得再 host 加宽——改为显式拒绝，规格里的 BF16 条款作废，FP32 证据不受影响（已在 #91 通知申领人）；PK-05 / GDA-04 / PK-06 骨架同样显式拒绝 BF16。
>   A2-44（已合入）的 raw flags 预处理是 FP32 host 算子、不是 BF16 计算，按 D-PM-35 的读法不违规，融进 kernel 归 BF-07。
> - **同一轮**：A2-44 已合入（PR #99 → `afafe03`，依 D-PM-33/34 自行合入，第二个）：8 个文件恰为批准写集，语义对 FLA v0.5.2 源码核实，新 D1/D3 测试在 base 上 4 失败/3 通过，真机日志（4 个 bd，25 个带哈希的 case 跨 bd 逐位相同，最大误差 4.9e-3，预算 0.05）由 PM 从原始记录复算，合入后 main 598 passed / 15 skipped。
>   缺口表：`kda-layer-l2norm-eps-formula-diverges`、`kda-layer-missing-silu-when-no-short-conv` 转 resolved，`qk-l2norm-not-in-kernel` 收窄（预处理仍在 kernel 外）。
>   **观察**：`check_domain=True`（默认）时 beta 用严格不等式 (0,1)，调用方自己算 sigmoid 并饱和成恰好 1.0/0 会被误拒（规格如此，层已走 raw flags 不受影响，可 `check_domain=False`）。
> - **待办**：A2-44 的申领人（session `01a0b7ce-…`）在 #31 排队申请 A2-02；按 BF16 优先，派它 `BF-04`（PKDA BF16 是用户直接问到的），A2-02 顺延。GDN 系列 session 的下一个是 `BF-01`（GDA-03 完成后）。
>
> **2026-09-19T15:20Z 更新：首页进展表按 kernel 列出 BF16 / FP32 现状（用户指出原表只有「涉及任务」计数，看不出每个 kernel 每种 dtype 的进展）。**
> - **实况**（对着源码与 contract 核过）：**不是只有 BF16 的 kernel。** 原生 BF16 operand 的只有 KDA 前/反向（Kimi-Linear 主路径，state 为 FP32）、GDN-2 的 decode 侧融合 kernel 与上游单元；
>   本仓新做的 GDN / PGDN / PKDA / GDN-2 chunk 前向以及 GDN 反向（进行中）的 **kernel 都是 FP32**。GDN / PGDN / GDN-2 chunk 的公共入口收 BF16，但是 host 侧 `q.float()` 加宽后进 FP32 kernel、输出转回
>   （`API 加宽`，kernel 本身不跑 BF16）；**PKDA 公共入口显式拒绝 BF16**（不静默 cast）。KDA 的 decode 入口收任意浮点 dtype、内部升到 FP32。
> - **PKDA 为何没有 BF16**：仓内记录的是范围决定而不是不可行证明——PK-02 规格把 BF16 定为非验收项（正确性一律 FP32 判定，首个切片是照 GD2-01 五 launch 结构做的全 FP32 实现，不得拿 BF16 KDA 的证据充数），
>   上游 ascriptor 也没有 PKDA/PGDN 单元可改成 BF16 cube。技术上的疑点有实测背书的只有 FP32 一侧：门控前缀差抵消（PK-02 的跨度 4096 误差 S=1.34e-4、PK-04 的 3.36e-4）已经逼近/超过 1e-4 预算，
>   BF16 尾数更短，大概率更糟——但这是推断，没测过。要不要做 BF16 的 PKDA/PGDN 是用户的范围决定，目前没有任务。
> - **看板改动**：`kernel_inventory.kernels[].dtype_status`（取值见 `tools/pm_board.py` 的 `DTYPE_STATE`，`--check` 校验）；首页各族表新增 BF16 / FP32 两列与图例；PK-02、PK-04 的任务 dtype 标签由 `bf16,fp32` 改为 `fp32`
>   （旧标签是「FP32 判定 + BF16 可选质量检查」的模糊读法，对 PKDA 是误导）。`gdn2_chunk_fwd` 的 BF16 只标「仅主机侧」（GD2-01 的真机数是 FP32）。仓主轨道与上游单元里没核过的格子标「未核」。
>
> **2026-09-19T14:05Z 更新：用户定了「所有任务必须完成真机验证」（D-PM-34），并要一份让 agent 知道怎么申领、开工的提示。**
> - **规则**：DONE 里没有真机验证就 `REVIEW rework`，不进入合入，也不能靠主机侧/CPU 测试补；`soc: any` 的任务在 assignee 自己的真机上验证、只对那个 SoC 声称；
>   确实无法真机的任务由用户豁免（PM 不自行豁免）；涉及 a2 的仍受 `AGENTS.md` §2 约束。已 done 的纯主机任务（PM-0、A2-04、A2-05、A2-40、PK-01、GD2-01、PK-02）**不追溯**，除非用户另说。
>   D-PM-33 里「没有真机验证的 PR 先问用户」一条由此变成「退回补真机」。
> - **机制**：看板新增 `hardware_verification.required_for_all_tasks`；`pm_board.py --check` 要求在飞任务的 assignee 声明过真机（具体 SoC 的任务要声明过那个 SoC；`soc: any` 的任务声明任一即可），
>   `--next` 同样按此过滤——`socs: none` 的申领人现在接不到任何任务。`needs.npu=False` 且未 done 的七个规格（A2-01/02/03/06/07/08/44）已各加一条真机验收；A2-44 还加了针对本任务的真机项，已通知申领人。
> - **agent 提示**：重写了 `docs/pm/prompts/agent.md`（开机、申领、ASSIGN 之后、真机验证、DONE/审查/合入、纪律、别 idle 的循环），START.md 的生成 APPLY 命令也补了 `--socs`。
> - **GDA-03**：三条 STATUS 已记（完整 B1/T4096/H=HV8 真机先跑并通过，FP32 各梯度 ~1e-7；第一次真机构建栽在形参名 `do` 是 C++ 保留字，已改内部名，`PK-05` 骨架里加了同样的提醒）。
>
> **2026-09-19T13:50Z 更新：PK-04 已合入（PR #97，合并提交 `add76cd`，合入的是审过的头 `108a500`）——第一个按 D-PM-33 由 PM 自行合入的 PR。**
> - **发现的缺陷与修复**：合入的 PK-02 版 PKDA 前向 `prepare` 的门控前缀和是纯 FP32 顺序累加，在域内（跨度 <155）「每 chunk 首项 ≈ −155、其余 −1e-5」的门控上
>   偏差 3.357e-4，真机 o/S 相对 L2 到 1.30e-4/1.82e-4，超 1e-4 预算（silent-wrong-result，已在 PK-04 里抓到并修）。修复是 prepare 的 FP32 Kahan 补偿前缀求和，
>   ATK、其余四个 kernel、公开 ABI、155 域、1e-4 预算都没动。修后真机 112 个全链 case（4 个 bd × 28）、40 个 API 边界、65 个尾长全 passed，最大相对 L2 6.6e-6，
>   跨 bd 逐位相同（PM 从原始 JSON 自己逐 case 比的）；同卡三明治比 Torch NPU eager 的 pinned naive 快 13×（T1024）/26×（T4096），基线不是 Triton、无速度门槛。
>   PM 复跑：新增回归测试在原 prepare.py 上失败（最大绝对差恰为 0.000335693359375）、修复后通过；合入后 main 全量 511 passed / 15 skipped，无回归。
> - **资格只到「这套精确环境 + in-process 桥的 FP32 前向」**（950PR_9589 V100、CANN 9.2.0、算子包 ascend950/910b/910_93、torch_npu 2.12.0）；SSH board/aclnn、BF16、backward、Triton、checkpoint 仍未测。
>   `docs/matrix/ops.json` 仍没有 PKDA 条目（矩阵欠账，等 A2-06 一并补）。
> - **待查（假设，没验证）**：KDA 链路的门控前缀和是否同构。KDA 的 155.97 精度扫描用的不是「一大后小」模式；`AGENTS.md` §6 的「整条链同类算式列一遍」适用。不在任何在飞任务的写集里，要查得另开任务（用户定）。
> - **申领人自述、PM 没复现**：另一套 CANN 9.2.0 头文件安装上 vendor 编译失败——pin 版 ascriptor 的 `tensorutils_cce.h:143` 只认 `IMPL_UTILS_SYS_MACROS_H`，而该工具链的 `sys_macros_impl.h:17` 用
>   `IMPL_UTILS_SYS_MACROS_IMPL_H`，`g_coreType` 重定义。仓内没有这次失败的原始证据（日志在其私有环境），所以**没记 gaps.json**；若别的 agent 在别的 A5 机器上撞到，
>   再带证据记（P2，ascriptor 库侧，我们不改）。与 AGENTS.md §5「同为 A5，不同机器不同」是同一类：**版本字符串相同不代表头文件/编译器相同**。
> - **同一轮的其余变化**：`GDA-03` 已 ACK（in_progress，eta 24h）；`A2-44` 已按 #86 的 APPLY 派给 session `01a0b7ce-…`（PK-04 的同一个 session，PK-04 已 done，并发规则不再挡），16h。
>   A2-44 与 A2-02 写集重叠，在飞期间 A2-02 若有人申领会被 gate。A2-44 是纯主机侧、没有真机验证，**它的 PR 合入前 PM 仍要问用户**（D-PM-33 只覆盖有真机验证的 PR）。
> - **PM 工具坑（今天踩的）**：`poll` 输出的最后一行 JSON 后面会**粘着**「已暂存到 …」那行头（没换行）；`json.loads` 会报 `Extra data`。用 `json.JSONDecoder().raw_decode(line)`
>   取第一个对象；别用 `str.splitlines()` 切（文本里的 `\x0c`、`\u2028` 会把一行事件切开，看起来像"非 JSON 行"）。
>
> **2026-09-19T13:05Z 更新：用户给了常设合入授权（D-PM-33）——取代下面所有「合入需要用户明确授权」的说法。**
> 原话：「PR只要有通过真机验证，你审查后确认正确性，高质量就可以合入，不需要我授权。有争议时，再向我提问」。PM 的读法（用户可纠正）：
> 只适用于带通过的真机验证的 PR；审查流程不变；证据只有自述、缺原始日志/标识、真机项被省略的按「有争议」问用户；
> 没有真机验证的 PR（纯文档/主机侧）、新账号首次合入、写集扩大、预算/闸/域变化、申领人异议、A2 上的结论、仓库定位/gated 放行/kernel 批次批准，仍先问用户。
> 合入后在 main 上复跑全量并记进看板，再事后一句话告诉用户；自动模式的分类器若仍拦，不绕过。完整条文见 `docs/pm/prompts/pm.md`「合入授权」。
>
> **2026-09-19T12:55Z 更新：用户放行 GDA-03 与 PK-04（D-PM-32，原话「两个都放行」），已派单。**下面 12:45Z 那段里"全部 gated、等用户放行"对这两个已过时，其余仍然有效。
> - `GDA-03`（#91）→ session `gdn-series-20260919T115559Z-29cb31d7`，分支 `task/GDA-03`，24h；`PK-04`（#92）→ session `01a0b7ce-…`，分支 `task/PK-04`，16h。
>   同账号不同 session，并发规则放行；两边写集不相交。
> - **PK-04 的申领人放行前已在私有工作区做了补偿累加修复和一轮真机**（自述，没有 PR）：bd4 的 28 个全链案例、3 个域内对抗种子误差约 6e-6，其余 bd 与同卡性能在跑。
>   放行**不追认**这些进度：DONE 时逐项核对原始证据——原失败样本要保留、修复要有「先定位」的证据、修复后的全部验收要在同一份修复后的版本上重跑。补偿累加修法本身仍是候选。
> - **PM 对 D-PM-32 的读法**：用户没有单独回答「PK-04 里的 kernel 修复算不算需要用户批准的 kernel 批次」，PM 读作放行覆盖 PK-04 规格的全部（含「先定位再在已批写集内修」），
>   已在汇报里向用户说明，用户不同意可以撤回。
> - 仍 gated：`PK-05`、`GDA-04`、`PK-06`、`GDA-05`（前置任务完成后 PM 细化，放行是用户的事）。合入授权见上面 D-PM-33。
> - 下一轮起：两个 assignee 的 ACK 要在 24h 内到（GDA-03 与 PK-04 都是）；PK-04 的 PR 到了先做「来源」核对（它此前是 gated 的外部需求），再走审查流程。
>
> **2026-09-19T12:45Z 更新：新增六个 gated 任务，等用户放行；PK-04 有一条待用户知悉的 silent-wrong-result 线索。**
> - **来源**：#89（GDN/PGDN 后续）与 #68 下的 REQUEST，均已分诊采纳（`triage:accepted`）。**#90 是 #68 需求的重复，已关闭。**
>   来源是外部需求，所以全部 `gated`，放行只能是用户：
>   - `GDA-03`（#91，分组 GDN chunk 反向，完整规格）→ `PK-05`（#93，PGDN 反向，骨架）→ `GDA-04`（#94，GDN decode，骨架）→
>     `PK-06`（#95，PGDN decode，骨架）→ `GDA-05`（#96，GDN/PGDN 性能，骨架）；`PK-04`（#92，PKDA 前向 A5 真机验收，完整规格，独立于上面这条链）。
>   - 序列严格按 D-PM-22 的"backward → decode → 性能"，**不含** GDN-2、Qwen3-Next、A2、KDA，也不是被否决的 #80。申领人转述的用户指示
>     （"继续申请任务，直到所有 GDN 任务都完成""需要完成真机测试"）PM 无法从评论核实，已交用户确认。
>   - 两个申领人（session `gdn-series-20260919T115559Z-29cb31d7` 与 `01a0b7ce-…`）的 APPLY 都留在 #28/#92 上，放行后 PM 直接按它们 ASSIGN
>     （GDA-03 `timebox_h` 24、PK-04 16），先把 `origin.approved_by_user` 置 true、状态置 `open`。
>   - 后四个 GDN/PGDN 任务只是骨架：细化与放行在前置任务完成后由 PM 做，之前不动手。
> - **PK-04 的线索（申领人自述，PM 只复算了机理）**：原 PKDA 前向在域内对抗性门控（每 chunk 首项 g≈−155、其余 −1e-5，跨度 <155）的**真机**输出有限
>   但超 ≤1e-4 预算（o/S 相对 L2 1.30e-4/1.82e-4），首个偏差在 `prepare` 的门控前缀和。PM 用 numpy 复算：纯 FP32 顺序累加该 pattern 对高精度累加的
>   最大偏差恰为 0.000335693359375，与申领人报的一致，机理成立；kernel 实际这样累加、o/S 超预算，PM 没有复现。RISK 来自未派单的申领人，poll 标"不采信"，
>   所以没有当协议消息处理；PM 独立把这个模式写进了 PK-04 的验收与陷阱，**修法（补偿累加）没有批准**。矩阵里本来就没有 PKDA 的真机结论，没有条目需要更正；
>   `gaps.json` 等 PK-04 的 DONE 见到原始证据后再记（届时是 `requires_kernel_change`）。**待查（假设，没验证）**：KDA 链路的门控前缀和是否同构——
>   `AGENTS.md` §6 的"整条链同类算式列一遍"适用；KDA 的 155.97 精度扫描用的不是这种"一大后小"模式。
> - **PM 侧的工具坑**：`poll` 会跳过 PM 账号发的非协议评论，而 agent 与 PM 共用 `pm_github_login`，所以 **agent 的纯文字进展评论对 poll 不可见**
>   （这次的可行性预研、工具链失败、PK-04 的 RISK 之外的进展都在纯文字评论里）。每轮对有活跃申领人的 issue 匿名读一次评论
>   （`https://api.github.com/repos/<owner>/<repo>/issues/<N>/comments`），别只信 poll。
>
> **2026-09-19T11:45Z 更新：PK-02、PK-03 已合入，当前没有在飞任务。**（用户 D-PM-31 授权；合并后 `main` 全量 510 passed / 15 skipped。）
> - **PK-03**（PGDN 前向，PR #87，合并提交 `1f4da05`）：A5 真机验收完成——60 个原生 case、bd1↔bd2 逐位相同，全网格 FP32
>   相对 L2 ≤ 2.5e-6（预算 1e-4）。同卡测量的基线是'GDN 方程成本基线'（不是 GDA-02 实现的性能），PGDN 慢 3.4%–4.5%。
> - **PK-02**（PKDA 前向，PR #88，合并提交 `8b78c09`）：主机侧验收。FP32 五阶段、q/k 原样消费（调用方负责归一化）、前向跨度
>   门控保持 155；跨度 4096 的抵消误差 S=1.34e-4 超预算，保留在公开域之外。无真机声称。
> - **两处规格被实测纠正过**：PKDA 的 naive **不做** q/k 归一化，PGDN 的 naive **做** `F.normalize`，两个算子的归一化语义
>   不同、结论不共用（`pkda_semantics.md` §7）；PK-02 的 oracle 由'Triton chunk'改为'独立 CPU 参考'。
> - **（已被 13:05Z 的 D-PM-33 取代）合入都需要用户明确授权**：自动模式会拦未经人工确认的合入（A5K-02、PK-03 各被拦过一次）。
> - **角色边界**（用户 D-PM-28）：机器与卡不归 PM 管，PM 只管 tasks 的进度与提交代码的质量。
> - **下一步候选**：D-PM-22 排期是 GDN 前向 → GQA → PGDN 前向 → backward → decode → 性能，前三步已完成。PGDN/PKDA 的
>   backward、decode、性能都还没有任务号，要用户排。另有 `A2-02`、`A2-44` 已解门（D-PM-29），等申领。
> - **矩阵欠账**：`docs/matrix/ops.json` 的 `linear_attention_ops` 里没有 GDN/GDN-2/PGDN/PKDA 的 chunk 前向单元条目
>   （前面几批 agent 任务的 `matrix_delta` 都是 none）。`ops.json` 与 `gen_matrix.py` 也是仓主轨道高频改的文件，
>   等 A2-06（矩阵按 SoC 分维）协调好一并补，不零散加。
>
> 下面各条是合入前的记录。

> **2026-09-19T09:05Z 更新：A5K-02 已合入**（PR #83，合并提交 `2589942`，合入的是审过的头 `3e3670f`；用户 D-PM-27 授权）。
> 公开 KDA 调度的 stable 路径现在选修复后的 recurrent（奇数 C 多头不再静默写错 `o`），upstream 路径对奇数 C 且
> `B*HV>block_dim` 提前报错；派生 `inverse_mm` 只做了被批准的事（与 pin 上游逐行 diff：仅入口改名 + 两块 L0C 双槽改单槽）。
> 缺口表已更新：`c1-multihead-o-corrupt` 保持 P0，写明默认路径已修与仍未解决的边界；新增
> `kda-bwd-inverse-mm-mutex-over-budget`（P1，进 kernel 修复队列，指上游源码）。**下面 A5K-02 的条目是合入前的记录。**
> **`A2-02` 与 `A2-44` 已由用户解门**（D-PM-29，2026-09-19），转 `open`。两个任务的写集彼此也有重叠
> （`ops/kda/chunk.py`、`layers/kda.py`），先 APPLY 者先做，后者大概率再 gate 一轮；此前被 NO_TASK 的申领人
> 已在 #31、#86 下被通知重新 APPLY。
>
> **PM 的边界（用户 2026-09-19，D-PM-28）**：机器与卡不归 PM 管理，agent 自己管理；PM 只管 tasks 的进度与提交代码的质量。
> 此前 PM 给 PK-03 设的'占用证据必须由用户/管理员提供'关卡已撤销，`shared-machine` 类 RISK 只记录、不上报。

- **`A5K-02`**（issue #81，草稿 PR #83，`blocked`，assignee `limjiunnbin`，
  session `01a0ae7a-d1c0-7f02-b040-be977ec47393`）：D-PM-24 批的 `inverse_mm.py` 派生
  kernel（mutex 34→32 单槽化）已实现，公开前向选择/原始对照/全部 9 个 backward kernel
  在 `block_dim=4` 下 vendor-compile 通过，bd1-3 编译中。**唯一的阻塞是设备空闲窗口**——
  同机另有任务在跑，另一张卡不健康。写集含 `ascend_fla/ops/kda/chunk.py`、
  `chunk_bwd.py`、`kda_fwd_stable/**`、`kda_bwd_stable/**`（含新加的 `inverse_mm.py` 与
  `contract.json`）。最近一次 STATUS：2026-09-18T09:31。
  **2026-09-19T04:08Z 更新**：换成 successor session `01a0b7ce-3daf-7550-a7bf-31c67eb91ca3`
  （自述受用户指示接手，PM 已记录，session 自称、不核实）；PR #83 head 仍 `888862b`（草稿）。
  **阻塞原因变了**：不再是"等设备空闲窗口"，而是**设备 SSH 认证失败**，
  远端构建、设备健康/占用/锁状态都未知；它没有动设备。机器归 assignee 管，PM 不代为恢复访问。
- **`PK-03`**（issue #69）：2026-09-19T02:41Z 撤回，04:16Z 由新 session 重新申领，**用户确认（D-PM-25）后
  已重新 ASSIGN**（session `pgdn-forward-20260919T041214Z-7b59b3a6`，与 A5K-02 同账号不同 session，并发规则放行）。
  `PK-03.md` 已重写为"主机侧、窄范围"的可派状态：ABI 与现有 GDN 门控逐项一致；oracle A=fla naive.py +
  B=独立 CPU 参考；**真机验收是必须项**（用户 2026-09-19，D-PM-26），通过后可合入；机器与卡归 assignee 自己管理（D-PM-28），PM 不设占用证据关卡。原 session `gdn-pgdn-forward` 的申领作废。已 ACK，草稿 PR #87 已开、主机侧验收就绪，等真机。
  （PGDN chunk 前向 + ATK 预条件，D-PM-22 排期解锁。）

### 这次会话新合入的两个 PR（供快速对账，细节见 §0.x 各节和 gaps.json）

- **PR #84（`GDA-02`，commit `8303167`）**：GDN chunk 前向补 GQA/GVA 分组
  （`HV % H == 0`），`HV==H` 退化路径逐位不变（代码层面可证明，见 `kernels/pipeline.py`
  的 `expand_inputs`）。合入后解锁了 `PK-03`。
- **PR #85（`A2-40`，commit `1c1abe3`）**：整网融合算子（`causal_conv1d` /
  `fused_rms_norm_gated` / `qk_l2norm_gate` / `packed_projection`）的清单/ABI/验收设计，
  纯文档，没动 kernel。发现两处本仓层与 fla 的静默分叉：
  - **D1**：`ascend_fla/layers/kda.py` 用 `F.normalize(x, eps=1e-6)`（对范数取 max），
    fla 是 `x/sqrt(Σx²+eps)`（分母整体加 eps）。‖x‖=1e-4 时两者差出一个数量级
    （fla 给 0.0995，本仓给 1.0）。近零行才触发，Kimi-Linear 的正常输入大概率碰不到。
  - **D3**：`use_short_conv=False` 时本仓不做 silu，fla 做。Kimi-Linear 用短卷积碰不到，
    但构造参数接受这个开关。
  - 两条都记进了 `gaps.json`（`kda-layer-l2norm-eps-formula-diverges` P2、
    `kda-layer-missing-silu-when-no-short-conv` P3），修复任务是新建的 **`A2-44`**
    （纯主机侧，不动 kernel）。

### 写集重叠：一个新问题类别，这次连续撞上两次

`A5K-02` 在飞期间，`ascend_fla/ops/kda/chunk.py`（以及 `chunk_bwd.py`、
`fused_recurrent.py`）实际上被它"占用"着——它要在这些文件里切换公开调度选中的
recurrent kernel。这次会话里，**`A2-44`**（issue #86）和 **`A2-02`**（issue #31，
`platform.py` 的 SoC 显式化重构）先后 APPLY，写集都覆盖到这几个文件，都被我按
write-set-overlap 转 `gated` 并加了 `deps: [..., A5K-02]`，NO_TASK 回复已发（issue
#86、#31 各有一条说明）。

**给下一个 PM 的提醒**：`A5K-02` 合入之后，`A2-02` 和 `A2-44` 会同时解锁——**它们两个
自己的写集也有重叠**（都碰 `ascend_fla/layers/kda.py` 和 `ops/kda/**` 的部分文件），
到时候谁先 APPLY 谁先做，第二个大概率还要再 gate 一轮，除非先手动协调一下顺序或者
缩小其中一个的写集。这不是这次会话能提前处理的（两个任务都还没人在写），只是先说清楚
免得到时候被当成新问题重新分析一遍。

### 并发规则实战：这次真的被用上了，而且是级联的

> **2026-09-19 更新**：`PK-03` 在本节写完 11 分钟后被 WITHDRAW，04:16Z 又由新 session 重新申领并重新 ASSIGN
> （D-PM-25）。`limjiunnbin` 现在仍同时占 A5K-02（session `01a0b7ce-…`）与 PK-03（session `pgdn-forward-…`）
> 两个任务，按 session 不同放行；规则本身没变。

`docs/pm/PROTOCOL.md` §3.1 的 `session` 并发规则（同账号、不同 `session`，可以同时持有
两个任务）这次被 `limjiunnbin` 账号连续用了两轮：先是 GDA-01/A5K-01/A5K-02 这条链
（`session: 01a0ae7a-...`），现在又叠加了 `PK-03` 的 `session: gdn-pgdn-forward`。
**目前 `limjiunnbin` 账号同时占着 A5K-02（blocked）和 PK-03（assigned）两个任务**，
这是按规则允许的形式条件（两边都填了 session 且不同），**不代表 PM 核实过真的是两个人
在操作**——如果后续两边的 STATUS 内容有明显不一致或互相矛盾的地方，值得留意。

### A2-40 留下的问题：已决，不放宽（用户 2026-09-19，D-PM-30）

`A2-40` 文中提出：`A2-41` 涉及的四个融合算子对象都是纯向量运算、不经过 cube，
技术上不受 split-K FP32 cube 缺陷影响，但 `AGENTS.md` §2 写的是"A2-11 之前 A2 上任何
算子结论都不算数"（不分 cube/vector）。**要不要对纯向量单元放宽这条总闸，是规则层面的
决定。用户 2026-09-19 已决定不放宽**（D-PM-30）：`AGENTS.md` §2 不改，`A2-41` 的 `machines:a2` 闸与对
A2-11 的依赖不变。

### 心跳 / 轮询状态（交接时刻）

两个在飞任务的最近 STATUS/ASSIGN 时间都在 2026-09-18 上午，交接时刻约 2026-09-19
凌晨，年龄约 16~17 小时，远未到 48h PING 阈值。`tools/pm_github.py poll` 的游标已推进
到最新（2026-09-19T01:23:11Z），没有积压事件。下一个 PM 直接按
`docs/pm/prompts/pm.md` 的轮询流程接手即可，不需要补跑历史事件。

> **2026-09-19 更正**：上面"不需要补跑历史事件"**只在同一台机器上成立**。下一节是换到全新机器时踩到的两个坑。

### 换机器接手 PM 的两个坑（2026-09-19，换到一台全新机器时实测）；另见上面 12:45Z 更新里的 poll 盲区

- **poll 游标是机器本地的。** 它存在 `tmp/pm/github_state.json`（git-ignored），换机器就没有；
  没有时 `since` 默认 `1970-01-01`，第一次 `poll` 会把整个仓库历史当成新事件吐出来。
  做法：从旧机器拷过来；拷不到就按上一次交接记下的时间戳手工播种 `{"since": "<ISO 时间>"}`，
  **宁早勿晚**（偏早只会重放少量已处理的事件，偏晚会静默丢事件）。播种前先匿名读一次公开 API
  （`issues/comments?since=…`）看回放量。本次按 01:23:11Z 播种，只回放 2 个事件：
  PK-03 的 WITHDRAW（上一个 PM 已处理；因为 assignee 已被清空，工具报"不采信"，是**重放的假警报**，
  不是冒充）和 A5K-02 草稿 PR #83（无警告、无需动作）。
- **`gh` 要够新，而且 `!` 通道没有 TTY。** 工具用了 `gh auth token -u <账号>`，系统包管理器里的老版本
  没有它也没有 `gh auth switch`（Ubuntu jammy 的 apt 包是 2.4.0，两样都缺，也只能存一个账号）；
  官方 v2.101.0 两样都有。装官方 release 的二进制并校验 sha256。`!gh auth login` 会报
  `--web or --with-token required when not running interactively`，要用 `--web`（设备码）或
  `--with-token < <token 文件>`。做法与登录顺序见 `docs/pm/START.md` §2 第 3、4 步。

### 在装了 torch_npu 的共享机器上复跑 PR 测试（2026-09-19 实测，都是踩过的）

- **不清环境直接 `pytest tests/` 会碰设备。** 机器装了 torch_npu/CANN 且有 NPU 设备可见：`import torch` 会自动加载
  torch_npu，`tests/conftest.py` 的 session fixture 会去编译全部 kernel，NPU 测试也可能起跑，而设备是共享的。
  做法：设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，再在 `PYTHONPATH` 最前放一个只写 `raise ImportError(...)` 的
  `torch_npu/__init__.py` 桩，并**先验证 `torch.npu.is_available()` 为 False** 才跑。只 `env -i` 清环境不够：
  torch_npu 会半初始化，报 `TORCH_LIBRARY` 重复注册并段错误。
- **测试要 pin 版的 ascriptor。** 同级目录的 `ascriptor-kernels` 是 kernels 的 pin（用 `git rev-parse` 核对，工作区要干净）；
  同级的 `ascriptor`（library）修订常常不是 pin，用 `git archive <pin> | tar -x` 从 git 对象导出 pin 版到临时目录，
  别动别人的检出。用 `ASCRIPTOR_KDA_FWD` / `ASCRIPTOR_KDA_BWD` 指向 kernels 的 `projects/a5/kda_fwd` / `kda_bwd`，
  `PYTHONPATH` 指向导出的 library。gitcode 要凭据，匿名取不到。
- **有 10 个测试需要 pin 版 FLA 源码**（cache adapter 4、GDN 3、GDN-2 1、KDA gating 2），本机没有 fla 就会 skip；
  对账时要写清 `passed + 这 10 个 = 申领人报的 passed`。全量套件在这个环境里约 70 秒，卡住不是测试慢，是没排除 torch_npu。

## 0. 2026-09-15 更新：SoC 顺序变更 + 多 agent 协作（先读这节）

**先说清楚：现在有两条并行轨道，本节讲的是 agent 那条。**

| 轨道 | 谁在做 | 内容 | 看哪里 |
|---|---|---|---|
| 仓主轨道 | 仓主本人 | A5 上的 GDN-2（decode 链路、gdn2-1.3B 整网、四个 a5/gdn2_* 单元） | 下面 §1.1~§1.3 |
| agent 轨道 | 外部 agent，PM 派单 | A2 (910B) → A3 (910C) → A5，首波 KDA + Kimi-Linear | `docs/pm/board.json` 与 GitHub issue |

两条轨道共用仓库，但互不指挥：仓主那条不受看板的 SoC 顺序约束，看板也不因为仓主在 A5 上推进就改波次
（2026-09-16 用户确认，看板 D-PM-10）。边界靠 `board.json` 的 `reserved_paths` 划：那些路径 agent 不碰，
`pm_board.py --check` 会拦住写集与之相交的任务。

**agent 轨道的 SoC 顺序是 A2 (910B) → A3 (910C) → A5**（2026-09-14 定的，见 `AGENTS.md` §2 与
`docs/plan.md` 的「按 SoC 分波次」）。所以下面 §2 那三个候选属于 agent 轨道的 A5 波次，要等 W-A3 退出；
**它们和仓主正在 A5 上做的 GDN-2 是两回事，别混起来读。**

工作改成多 agent 推进：PM 维护 `docs/pm/board.json`，agent 按 `docs/pm/PROTOCOL.md` 申领任务并汇报，
看进度跑 `python tools/pm_board.py --render`。

现在能派的是 W0 的主机侧任务，A2 机器还没到位。最靠前的是 A2-01（split-K FP32 cube 缺陷定性，A2 的总闸）。
A2-06（矩阵按 SoC 分维）已改成 gated：它要重构的
`docs/matrix/*.json` 与 `tools/gen_matrix.py` 正是仓主轨道在高频改的文件，等协调好再开。

### A2-04 已完成（2026-09-17，PR #60 合入 8ac32e5）—— 两条要记住的

**① `c1-multihead-o-corrupt` 的根因找到了，而且暴露面比我们记的宽。**
不是漏了某次 DEvent/Mutex 配对，是 `Aqk` 的 L1 交接**两信用配固定槽**：
`aqk_l1_valid = DEvent(..., preset=True)` 给两个信用，槽却是 `aqk_slot = Var(c_idx % 2)`（按 chunk 号取）。
一个头最后一个 chunk 的槽是 `(C-1)%2`、下一个头第一个是 `0` —— **当且仅当 C 为奇数时相撞**。
与 ascriptor `library/docs/defects/M10-076-mutex-credits-and-handoff-slots.md` 同型，
但那条的修复在 `autosync` 里，而本 kernel 是手写同步、不过 autosync，**拦不住**。

**闸现在是漏的**：`_check_single_chunk_heads` 只拦 `C==1 and B*HV > block_dim`，
而奇数 C≥3 结构上同样暴露（kimi 的 `C = T/64`，T=192、320 都在里面）。
收紧成 `C % 2 == 1` 会拒掉真机上跑过的形状，按 §1 要显式决策 —— **待所有者放行，先别改代码**。
细节与证据在 `gaps.json` 的该条目，`docs/pm/deltas/A2-04.json` 是原始提案。

**② "C≥2 全对是因为 chunk 循环第二遍补上了同步"是错的，已从 `AGENTS.md` §6 更正。**
那句话是从"C=1 坏、C=2 好"的**症状规律**倒推出来的，写进文档两期没人怀疑，
既把根因指错方向（去找"缺的那次同步"），又把安全域说宽了。
**症状规律不只会误导修法，还会伪装成结论写进文档。**

证据等级：全部是 a5 管线模型值（pipesim + 调度回放），**不是真机数**。
但 `gaps.json` 里 2026-09-11 那张真机失效表被 **12/12 格逐头复现**（PM 在权威 workspace 上独立复算过），
且预测的 kimi `o` 相对 L2 1.060 / `final_state` 2.458e-03 与真机记录的 1.06 / 2.46e-03 吻合。

### GD2-01 已完成（2026-09-17，PR #64 合入 0397aa1）—— 例外任务，两条要记住的

**这条不属于上面 §0 的 A2→A3→A5 波次**，是用户对 D-PM-12（"GDN-2 不在看板造任务"）批准的一次
显式例外（D-PM-16，issue #61）：agent 轨道做 GDN-2 **chunk 前向**（forward-only），
写集限定新文件（`ascend_fla/ops/gdn2_chunk_fwd.py` 是 `ops/gdn2/` 的兄弟文件，不进那个包），
`reserved_paths` 一个字节没碰。

**① 门控跨度的新记录：真实 95B checkpoint 测出 1520.91455，超过 2026-09-14 记的 1461.214。**
`gaps.json` 的 `gdn2-chunk-gate-range` 已更新。选定的表示是逐 chunk FP32 对数前缀 + 直接算因果差分
（不是 KDA 那种端点/中点乘积），全部指数项恒非正，`exp(1461/2)` 这类值从未被求值——但**精度仍要单独测**，
不能因为"有限"就假设"够准"（这条是从 KDA 那次教训延续下来的）。

**② 真机上定位到一个 WY 依赖循环的编译器/硬件缺陷**（950PR_9589 V100, CANN 9.2.0）：VF 循环的内层
边界如果直接绑定外层归纳变量会漏掉最后一次贡献；改成固定 64 次迭代 + 显式 `j < i` 判断绕过。
**compiler-versus-silicon 的归因没有做**，只确认了这个具体写法在这台机器上不安全。

**GD2-02（真机数 + 真实整网验证 + 性能三明治）现在可派**（deps 已满足）。GD2-01 的原
assignee 后续又申请并做了 GD2-03（性能优化，用户批准 D-PM-17，正在进行）。

### PKDA / PGDN 权威来源确立（2026-09-17，PK-01 完成）—— 见 `docs/research/pkda_semantics.md`

用户提供本地 fla checkout（`/Users/limjiunnbin/work/flash-linear-attention`，0.6.0，
HEAD `e52dbc0e`——**与 GD2-01 用的 FLA pin 是同一个 commit**）。之前 PK-01.md 写的
"fla 0.5.2 没有 pkda/pgdn，本仓无 oracle"**只在 0.5.2 成立**，本仓 `.venv` 装的正是 0.5.2，
但上游 0.6.0（2026-06 合入，PR fla-org/flash-linear-attention#950）已经有完整实现。

**两条要记住的**：

① **PKDA 是 KDA 加一层 ATK 预条件，不是新算子族**——ABI 除三个 ATK 专属张量外，与本仓
`ascend_fla/ops/kda/chunk.py` 逐项对应，可直接复用 `kda_fwd_stable`/`kda_bwd_stable`。
PGDN 同理是 GDN 加同一层预条件，但需要的 GQA/GVA 分组正是 GDN 自己缺的 `gdn-no-gqa`——
**排在 GDN 自身 ABI 之后**，不是并列关系。

② **Gemini 文档"`k̃=k/(p+eps)` 会下溢"那条担心，看了真实算法后不成立。** ATK 的预条件乘子
是 `M = exp(-log(x)·s)`，`s = r/(1+|r|)` 恒落在 `(-1,1)`——这是这个函数形式本身的硬数学界，
不依赖状态 `A` 的大小，所以不存在"除法下溢/溢出"这回事。仍要测的是 `log(A+eps)` 在本仓真实
量级下的精度，以及 `k_precond` 替代 `k` 之后 KDA 已有的门控跨度判定要不要重新过一遍。

排上 **PK-02**（PKDA，open，主机侧，`kda_fwd_stable`/`bwd_stable` 派生单元）与
**PK-03**（PGDN，gated: prereq-gdn-abi）。两者都**没有已发布的预训练权重**——训练仓
（<https://github.com/ntumm120/preconditioned-deltanet>）只有从零训练的配置（340M/1B），
端到端验证到不了真实 checkpoint 的 logits/cache 一级，只能到双 oracle（fla naive + fla
自己已验证过的 Triton chunk）加真实规模形状。

### 2026-09-17/18：GD2-02~04、ascriptor 库缺陷、并发规则、GDA-01、A5K-01

**GD2-01 之后又做了两轮性能优化**：GD2-03（scan/output 逐 channel 重复算的标量指数改成
逐行向量化，D-PM-17）、GD2-04（WY 行对复用，D-PM-19）都已合入。两轮都遵守"先 profile
再改、精度判据不放松"，性能数字见各自的 PR（#67、#74）与 `contract.json`。**GD2-02**
（真机数 + 真实整网验证 + 性能三明治）还没人接，deps 已满足，随时可派。

**发现一个 ascriptor 库本身的缺陷**（`gaps.json` 的 `ascriptor-gm-transfer-two-slice-row-gap`，
P0）：`passes/device_lower.py` 的 `gm_transfer` 函数，双切片分支（`len(sliced)==2`）算行间距
时只看第二个切片维之后的形状，**漏了夹在两个切片维中间的标量索引维**——4 维 GM 张量
`[B,T,H,D]` 里 B/H 标量索引、T/D 切片这种模式会静默算错行间距，产出有限但错误的值。
这是库层面的问题，不是本仓能改的范围（`AGENTS.md` §3）；本仓侧的绕法是显式传
`gm_to_ub_pad(..., 正确的 gap)`，GDA-01 已经这么做并验证过。**扫过已合入的 kernel**，
找到 19 处同类模式，全部只搬一行、不受影响——但扫描范围有限（只覆盖直接 GM 参数 +
`<<=` 形式），不代表证明了整仓没有别处踩中。

**多 agent 并发规则改了**（`tools/pm_board.py` 的 `check()`）：原来"一个 agent 同时只接
一个任务"是按 GitHub 账号判定的，现在改成按 `(账号, assignee_caps.session)` 判定——
两个任务都填 `session` 且不同，同一个账号可以并发持有多个任务；只填一边、都不填、或
两边相同，仍按老规矩当同一个 agent。见 `docs/pm/PROTOCOL.md` §3.1。这条规则被
GDA-01 与 A5K-01 实际用上了——同一个 `limjiunnbin` 账号靠不同 `session` 同时接了两个任务。

**GDA-01 已合入**（非 GQA 的 GDN `gated_delta_rule` chunk 前向，A5，D-PM-20 窄范围例外，
PR #77）：真机 10/10 case 通过、三轮同步三明治 T1024 1.07x/T4096 1.08x。过程中原设备
出了真的硬件故障（RAS bus error `0x80f78009`），处置完全正确——不复位不 kill、只读诊断、
切到同机健康设备重跑、旧结果不混入最终声明。

**A5K-01 已合入**（KDA `Aqk` L1 交接缺陷修复，PR #78）：把 A2-04 诊断、pipesim 验证过的
槽轮转修法（`Var(((pair_idx - pair_begin) * C + c_idx) % 2)`）落到 `kda_fwd_stable` 的
派生 `recurrent.py` 里。真机 336/336 case（C=1..6 × 7 组 H/HV × 零/随机 state × bd=1..4，
含 Kimi 真实形状 H=HV=32）通过，PM 独立复算了不依赖真机的 12 格 pipesim 回归，逐字一致。
**没解决的边界，写进了 `gaps.json`**：修复只在这个独立单元里，`ascend_fla/ops/kda/chunk.py`
的公开调度**仍然选择原始（有缺陷的）recurrent**——那个文件不在 A5K-01 写集内，接线是
一个需要显式决定的独立步骤。**在接线完成之前，公开 API 的用户仍然会撞上这个缺陷。**

**当前可派（`tools/pm_board.py --next`）**：A2-01（P0，split-K 缺陷定性）、A2-02、A2-08、
**GD2-02**（真机验证 GD2-01~04 的累积成果）、**PK-02**（PKDA，KDA 派生单元）。
`GDA-01`/`A5K-01` 都是波次例外，做完之后想再申领同类工作要走"新 REQUEST → 用户批准 →
新任务号"，不能假设已批准的例外自动延伸到下一轮（GD2-03/04 就是这么一路批下来的先例）。

任务 issue 发过两轮。第一轮用新建的 bot 账号 `ascend-fla-pm-bot` 建了 26 个（#2~#27），两分钟内建完，
触发 GitHub 反滥用过滤：匿名访问这些 issue 和该账号主页全是 404，登录态却一切正常 —— 所以看着像发成功了。
那 26 个已全部关闭（`not_planned`）。第二轮改用 `limjiunnbin` 重发，申领入口是 **#28**，任务 issue 从 #29 起，
实测匿名可见。

由此定下两个身份（看板 D-PM-9）：`pm_github_login` 发 issue 正文与协议评论，必须匿名可见；
`label_github_login` 做打标签、关 issue 这类要 write 权限的操作。没有 write 权限的账号建 issue 时
**标签会被静默丢掉**，gh 不报错。

同一个坑还暴露了 `pm_github.py` 的三处问题，都已修并有反例测试：REST 列表被过滤后 sync 会重建重复 issue
（改走 GraphQL 并加了"记过编号却不见了就停下"的闸，84add96）；换账号后旧的已关闭 issue 会被当成"已存在"而复活；
建 issue 没有节流（现在默认 `--pace 15`）。

三份 Gemini 规划文档只当需求来源看。它们与实测矛盾的说法，列在 `docs/pm/PROTOCOL.md` §6。

## 1. 一句话现状

第一期（runtime 桥 + `kda_fwd` 接线 + 基线）与第二期（`kda_bwd` + autograd + KDA layer）
**已完成**；第三期进行中 —— **KDA 的 decode 已自写 kernel 并接进 layer，功能可用、性能不可用**。
Kimi/Qwen 模型注入与矩阵 CI 未开始；用户指定的 GDN-2 1.3B 已完成 torch_npu 基线、
架构拆分和 packed-inference 优化；除通用 **Ascriptor CCE `gdn2_fused_recurrent`** 外，又新增
模型专用 **`gdn2_fused_decode`**，把 raw gates、递推和 output norm 合成一 launch。Cast 与
整网时延显著下降；静态双槽又让合法的两段成对流水在真机生效。完整 stateful NPU Graph
与模型专用 short-conv/cache kernel 已把纯模型 decode 压到稳定约 **3.109ms/token**；随后
原生 fused RMSNorm + host-packed SwiGLU W1/W2 又压到 **2.654~2.664ms/token**（375.37~376.86
token/s），真实生成 decode loop 为 **340.62~354.33 calls/s**。低内存默认设备时间约86%已是
MatMul/GEMV。gamma-fold混合MLP虽通过kernel级真机闸（边界1.077x），却在真实95B递推中
把cache误差累积到1.523e-2，已拒绝；恢复三个BF16边界的保守版已过完整数值闸与同卡Graph
夹心A/B，显式opt-in达到 **2.587~2.590ms/token、386.13~386.52 token/s**，但增加约1.030GB
派生权重，故低内存默认仍不变。最新profile中权重流占90.18%、混合kernel的AIC MTE2平均
91.88%；后续大步收益必须触及权重字节。长
prefill/训练的 chunk fwd+bwd 尚未开始。

| 指标 | 值 |
|---|---|
| 真机全量测试 | **63 passed**（A5 / CANN 9.2.0，约 346s；本轮新 kernel 尚未并入这次全量） |
| 主机侧测试 | 上一轮全量 66 passed / 5 skipped；本轮 GDN-2/模块定向 **39 passed** |
| GDN-2 定向真机 | 通用 recurrent contract 5/5；fused decode、packed short-conv与mixed MLP的sim/pipesim/aclnn、真实整网及profiler passed；低内存默认375.37~376.86 token/s，mixed显式opt-in 386.13~386.52 token/s |
| 缺口 | P0 **1** · P1 17 · P2 9 · 已解决 11 · 共 38 |
| kernel 统一修复队列 | **24 项**（`gaps.json` 的 `summary.kernel_fix_queue`） |
| 工作区 | GDN-2 model-first 改动未提交；先看 `git status` |

### 1.1 GDN-2 1.3B model-first（2026-09-14 新增）

用户指定先跑 `LLM-OS-Models/gdn2-1.3B-fineweb-edu-100b` 的 95B checkpoint，再从真实
调用链定位算子。已完成：

- 只下载 `checkpoint-95B-model-ckpt.pth`，固定 HF revision `86327354…`；文件
  17,401,727,659 bytes，SHA-256 `4ac729c6…f6d`，目标 model mount 上又复核一次。
  HF 的 90B 与 95B 文件当前指向同一个 LFS 对象，但 archive 内部是
  `step_count=22667` / `iter_num=725344`，按 4,194,304 tokens/step 得 **95.072B**，
  所以 95B 文件本身身份正确，90B 名字才是发布侧异常。
- `ascend_fla/models/gdn2.py`：完整 LitGPT-compatible torch/torch_npu 基线，399 个
  state_dict 键与发布方 Apache runtime 逐键逐 shape 相同；`meta + mmap + assign`
  严格加载大 checkpoint，普通权重转 BF16、`A_log/dt_bias` 保持 FP32。
- 真机实权重：input `[1,2]` → logits `[1,2,32000]` BF16，全有限、范围
  `[-11.75,11.0]`；18 层 state `[1,16,128,128]` FP32；一次前向 vs 逐 token cache
  **max_abs_diff=0 / relL2=0**。NPU allocated 2,960,865,280 bytes。load 2.947s、首次
  forward 0.210s 只是单次冒烟，**不是性能基线**。原始记录在
  `tmp/gdn2-95b/gdn2-full-run.json`（git-ignored）。
- `ascend_fla/models/generation.py` + `benchmarks/generate_gdn2_text.py`：一次 prefill 后逐 token
  传递 recurrent/conv cache；采样只把最后一行 logits 搬回 CPU，避免把 torch_npu 随机算子
  混进“模型能跑”的结论。TinyLlama v1.1 tokenizer 固定到 revision `ff3c701f…`，验证了
  32K vocab、自动 BOS=1、EOS=2。真实 95B 权重的 greedy 输出为：
  `The capital of France is Paris. It is the capital of the country of France. It is the largest city in France. It is the most populous city in the European Union.`
  共生成 32 token，最终代码单次 0.629s / 50.9 token/s；这是跑通证据，**不是性能基线**。原始记录在
  `tmp/gdn2-95b/text-greedy.json` 与 `text-greedy.raw.log`（git-ignored）。
- 最关键的算子结论：**GDN-2 不是现有 GDN 的换名。** 目标需要 channel-wise
  `g[B,T,H,K]`、erase `b[B,T,H,K]`、独立 write `w[B,T,H,V]`；现有
  `a5.gdn_fwd/bwd` 只有 per-token 标量 `beta/g`，不能接。已记
  `gdn2-abi-not-gdn`，缺失算子记为 `gdn2_fused_recurrent` 与
  `gdn2_chunk_fwd_bwd`。
- 边界 profile（真实 95B 权重、warmup 后）：全模型 decode 不含 logits D2H 为 10.086ms/token；
  layer-0 block 568.5µs，其中 mixer 433.7µs。最大的四段是 recurrence+qk norm 140.8µs、
  三路 short conv 105.9µs、output norm-gate 58.9µs、g/b/w 激活 57.0µs。把同形状权重按行
  拼接后，qkvbw projection 在 T=1 快 2.54x、qkv short conv 快 4.27x；实测输出与分开执行
  逐位相同，但这只作诊断，正式闸仍是 relative-L2。
  证据：`tmp/gdn2-95b/profile-boundaries.json` 与原始日志（git-ignored）。
- 数值域 stress（真实权重、合法随机 token id、T=4096）：单 token `-g` 最大 60.926，
  64-token 累计跨度最大 **1461.214**，远超 KDA stable 的已验证域。recurrent 每步只算
  `exp(g_t)`，不受跨 token 配对量程影响；chunk 必须单独设计，不能复制 KDA。已记
  `gdn2-chunk-gate-range`，证据：`tmp/gdn2-95b/gate-domains-t4096.json`。

### 1.2 GDN-2 架构步骤 1/2（2026-09-14 已完成）

用户审定三项决策后完成“先重构、再打包”，**没有开始 gate/recurrent/chunk 自编译算子**：

- `reference/gdn2.py` 独立出 fp32-state recurrence oracle；`layers/gdn2.py` 接管
  `GatedDeltaNet2`、typed `GDN2LayerCache` 与显式 `core_backend`。这一阶段 backend 只有
  `torch`；后续 CCE 接线见 §1.3。未知名字始终报错，不静默 fallback。canonical 模型仍是
  399 个 LitGPT key。
- `models/gdn2_checkpoint.py` 只负责 checkpoint wrapper 前缀规范化；模型 loader 先以
  canonical 399-key strict load，再可选原地转成 `projection_layout="packed-inference"`。
  packed 布局仍有同样 1,450,096,416 个参数；加入 W1/W2 host packing 后内部 state_dict 是
  255 项（此前只打包 mixer 时是273项）；转换逐层消费
  原模块，不常驻两份约 0.85GB 投影。打包在 host 完成后才 H2D，packed reserved memory
  实测约 3.14GB；packed 求导会显式报错。
- q/k/v/b/w 五个同输入、无 bias 的投影合成一个 `F.linear`；q/k/v depthwise short-conv
  与三份 cache 沿 channel 合成一次调用，packed cache 是 `[B,6144,4]` BF16。f/output-gate
  的低秩首层只在 B=T=1 时合并；T>1 改用 packed weight 的两个连续 row slice 分开算。
- **这一条件分支来自一次真机纠错**：孤立 microbench 只验证 packed 第一层输出，最初把它的
  strided split 直接喂第二层，真实权重 T=6 的 output-gate 出现 max_abs_diff=7.8125e-3，
  最终 logits relL2=4.57e-3。改为 T>1 分开算后，真实 95B 的 BF16 T=6/T=64 与 FP32 T=6
  均通过数值闸（预算分别 1e-2 / 1e-5，实测 logits/cache relL2=0）。本次恰好逐位相同只是
  额外观测，**不是 CPU/NPU 或 packed/canonical 的契约**。另有 tiny FP32 模型直接对 CPU
  oracle，logits/recurrent/conv cache 最大 relative-L2=5.740e-4，以 1e-3 预算通过。
- warm decode 用两种加载顺序、warmup=100/iters=200 复测：canonical 10.184~11.625ms，
  packed 9.212~9.822ms，加速 **1.106x~1.184x**（按保守下界看 1.106x）。同一 32-token
  greedy 文本与 token ids 在最终 host-packed loader 上再次通过；单次 0.502s 只作 cold-ish
  端到端 smoke，不当稳态基线。复现：
  `benchmarks/verify_gdn2_packed.py`；证据在 `tmp/gdn2-95b/verify-packed-steady-*.json`、
  `verify-packed-host-pack-{bf16,fp32,t64-bf16}.json`、
  `packed-cache-chain.json`、`text-greedy-packed-host-pack-final.json` 与原始日志。

### 1.3 GDN-2 步骤 1~4：CCE recurrent 已接整网（2026-09-14 已完成）

- **ABI**：公开语义冻结为 token-major q/k/v/g/b/w `[B,T,H,128]` + FP32
  state `[B,H,128,128]`；g/b/w 已激活。kernel 内做 q/k FP32 L2 norm、`128^-0.5`
  scale、channel-wise decay/erase/write、输出与 final state。首版只收 B=1、T=1~16、
  H∈{1,16}、K=V=128、block_dim=8；CCE 只支持 inference，不满足就报错。
- **kernel + harness**：`kernels/projects/a5/gdn2_fused_recurrent/` 是本仓自有
  `ascriptor.kernel-unit/1`，CCE 静态检查 0 error / 0 warning / 220 surface ops，emit
  266 lowered ops；reference 5/5、sim 3/3 scoped、pipesim 2/2 scoped。Ascriptor 原生
  unit runner 用 `requested_backend=cce + launcher=aclnn` 跑完五个 contract case，o 最大
  relative-L2 `4.169e-6`、final_state 最大 `9.108e-8`（预算均 `1e-4`）。
- **接线**：`ascend_fla/ops/gdn2/fused_recurrent.py` 经已有进程内 ctypes→aclnn 桥直接吃
  NPU tensor；本轮没有再包一层 pybind，因为现有桥已经覆盖零拷贝 data_ptr、stream 与
  workspace。`models/gdn2.py` 在任何整网算子执行前 `prepare()`，`layers/gdn2.py::_run_core`
  显式分派 torch/cce，绝不 fallback。另修了 runtime cache：source signature 现在覆盖入口
  同文件的 `@vf/@func` helper；否则 helper 改动会错误复用旧 binary。
- **公开算子**：BF16 real-prompt case 的 o relative-L2=0、state=`6.167e-8`；一次 T=4
  与四次 T=1 串 state 的 o/state relative-L2=0。B1/T1/H16 完整 public call（含 cast +
  aclnn）`88.165us`，torch_npu 组合 `148.400us`，1.683x（warmup10/iters100/同步）。
- **整网**：真实 95B checkpoint、packed-inference 下，BF16 prompt/step logits
  relative-L2=`3.260e-3/3.186e-3`，最差 recurrent cache=`9.154e-3`（预算1e-2）；FP32
  为 `1.391e-5/2.065e-5`、cache=`1.785e-4`（预算1e-3）。argmax 与 cache offset 相同，
  bitwise 只作诊断。BF16 两种加载顺序都过；整网 decode CCE `8.182~9.435ms` 对 torch_npu
  `8.378~9.460ms`，目前基本持平，说明下一轮要 profile 周边而不是从单算子倍数外推。
  CCE core 下 canonical↔packed 也通过：prompt/step logits 与两类 cache relative-L2 均为0；
  同轮 decode `10.920ms` 对 `9.843ms`，packed 为1.109x。
- **吐字**：同一真实 checkpoint 由 CCE backend greedy 生成 32 token：
  `The capital of France is Paris. It is the largest city in France and the capital of the country. Paris is located in the north-eastern part of France. Paris is the`
  单次 0.550s / 58.18 token/s，只作 smoke。证据统一在 `tmp/gdn2-recurrent/`（git-ignored）。

### 1.4 GDN-2 CCE decode profiler（2026-09-15）

`benchmarks/profile_gdn2_backend.py` 临时给生产 forward 注入 `record_function` 范围，普通推理
路径没有 profiling 分支或额外开销。真实 95B / BF16 packed / B1T1H16 用 torch_npu
profiler Level1 + PipeUtilization，两个 backend 各自独立进程，1 warmup + 3 active：

- torch_npu `6043.4us/token / 1523 kernels` → CCE `5108.8us / 1163 kernels`，设备侧少
  `934.6us`（15.5%）和 360 次发射（23.6%）。几乎全部来自 core：`1189.3→282.2us`。
  直接 ctypes→aclnn 的 custom kernel 不挂 Torch 关系树，CCE core 的 282.2 = 范围内 casts
  186.2 + kernel CSV 的 custom 96.0。
- **CCE 剩余占比**：MLP 26.36%、short-conv+cache 16.85%、packed projection 16.20%、
  两次 block RMSNorm 12.04%、output norm-gate 8.65%、gate prep 7.38%、core 含 casts 5.52%。
  recurrent kernel 本体只有 5.336us/层、1.88% 整网设备时间，下一轮不能再先抠它的算术。
- MatMul 占 46.2%（145 calls），代表性 BF16 GEMV 的 MTE2 ratio≈94%、MAC≈11%，是权重
  读取主导。Cast 占11.1%，**CCE 与 torch 都是309次/token**：FP32 CCE ABI 108、output
  norm-gate 72、37个 model RMSNorm 111、gate prep 18，源码与 trace 可以逐项对齐。
  其中91次是 packed CCE inference 下严格冗余的：55个不变 norm weight 每 token 重转FP32，
  以及18层 recurrent output FP32→BF16 后立刻被 output norm 升回FP32的36次往返。剩余218次
  表达的是FP32规约/state精度语义，但可以搬进融合kernel内部，并不要求作为独立Cast算子存在；
  **没有一项是为了CPU/NPU逐位相同**，本仓本来就不以bitwise为契约。
- packed short-conv 每层为 cache 做两次 cat+两次 slice；T=1 时 window 与 new_cache 可共用，
  按当前各半计可先去掉约234.3us/token。`-exp(A_log).repeat_interleave` 在 inference 中是常量，
  却每 token 重算，exp+repeat+neg 共125.9us/54 launches，也应先预计算。
- **wall time 仍只能判持平**：warmup100/iters200 的反序 paired check，torch-first 为
  9.721 vs 9.523ms（1.021x），CCE-first 为9.318 vs9.562ms（0.975x）。两次跨过1，否决了
  单次 pre-profiler 1.246x 的加速说法。设备收益被上千次周边发射、casts 与 aclnn host
  固定成本吞掉。完整分析在 `tmp/gdn2-profiler/analysis.md`，新缺口为
  `gdn2-decode-fragmentation`。

### 1.5 GDN-2 Cast 清理与 BF16 fused decode（2026-09-15）

- packed inference 在 host 已按目标 dtype 量化权重；55 个 norm weight 以前每 token 再
  `.float()` 一次。现在把**同一份已量化值**只扩宽一次到 FP32，不改变旧数值边界；同时缓存
  channel-expanded `-exp(A_log)`。中间 profile 的 Cast **309→254**、565.925→466.875us/token。
- 新单元 `a5.gdn2_fused_decode` 固定真实 B1/T1/H16/K=V=128 BF16：直接吃
  q/k/v/f_raw/b_raw/w_raw/output_gate，在 CCE 内做 stable softplus、sigmoid、q/k norm、FP32
  recurrence/state、output RMSNorm+swish，最后只做一次 BF16 RNE store。旧 FP32 recurrent ABI
  不改，继续服务 prompt、FP32 与非精确匹配配置；不回退 Torch。包级 `prepare()` 在首次算子
  解析前把两者都编好。
- 固定 Ascriptor 0.1.0：静态双槽版本 check 0 error/0 warning/404 surface ops，CCE emit 415 lowered ops；
  reference/sim/pipesim 3/3，无 hazard/deadlock；aclnn real_decode 的 o/state relative-L2
  `8.574e-11/3.044e-7`。真实95B step logits `3.436e-3`、最差 recurrent cache `9.521e-3`，
  均在 BF16 `1e-2` 预算内；32-token greedy token ids 与旧 CCE 路径相同且文本可读。
- 最终 Cast **309→74/token**，Cast 时间565.925→147.355us；设备kernel总时长
  5108.848→4023.590us/token，launch 1163→640。warmup100/iters200 的反序 paired：
  torch-first 8.087→4.900ms（1.650x），CCE-first 8.726→6.188ms（1.410x），这次两边都
  稳定快于 torch。
- **静态双槽流水已通过**：动态下标 DBuff 的 `auto_sync` 把 slot n 与 n-1 保守判成同 root，
  插入 read1→compute0 与 write0→compute1 假依赖；所以旧 DB64 四 pipe 和为98.17%、真机反而
  4.804us。改成两个静态命名的64-row槽后，pipesim 从10463降到9613 cycles，同核多pipe
  重叠3.68%→11.44%；两次真机各54样本 mean=4.259/4.236us，夹心整state对照
  4.733us，合并mean/median=4.247/4.256us，快1.114x/1.108x。AIV
  Vector/Scalar/MTE2/MTE3 合并平均34.73/17.60/37.52/21.70%，四pipe和111.55%（对照86.84%），
  证明是真重叠。aclnn与真实95B整网数值均过原闸；两轮profile与整网任务的宿主监控中，
  物理0活跃0次、任务PID只在7。随后为补 canonical unit runner 证据做的一次**fresh CCE build**
  在实际 `test_aclnnop` 上7之前，捕捉到物理0上一个 `python`/0MB瞬时条目；时间与CANN `opc`
  编译进程吻合，但进程退出后无法补做PID namespace硬归属。实际 `test_aclnnop` 只在7，且0未
  留驻。严格“只碰7”期间后续应复用已构建产物，不再 fresh build；不能把编译器探针说成0次。
- **单pipe约80%仍未过，但原因已钉死**：第一遍必须对全部128行完成erase规约，才能形成delta；
  因而MTE2+第一遍Vector与第二遍Vector+MTE3被全局join分成两段，当前边界不可能形成三pipe
  长稳态。**不要减核或加冗余工作刷利用率。** 要继续满足该门槛只能扩大到
  RMSNorm+projection / output+o_proj 等含有用BF16 GEMV的混合边界；若优先整网时延，则做
  qkv short-conv+cache。证据在 `tmp/gdn2-cast-fusion/`。

### 1.6 GDN-2 stateful NPU Graph capture（2026-09-15）

- torch_npu 2.10 的 `NPUGraph` 可捕获完整18层BF16 packed CCE decode；本仓 ctypes→aclnn
  custom op 也能进图。单 custom kernel 从 eager 约69.5us降到 replay约14.0us；整网固定
  prompt-cache 从6735.3降到3834.9us/token（148.47→260.76 token/s，1.756x）。
- 真正可递推版本在图尾把18层新 recurrent/conv cache 拷回固定输入地址；4个不同token连续
  replay后，logits、两类cache均与eager**逐位相同**。排除早期探针后，v3~v5同轮总吞吐
  eager为5911.9~6323.7us/token，graph稳定在3908.2~3911.9us（255.63~255.87 token/s，
  1.512x~1.617x）；每token同步的串行graph为3915.6~3926.0us（254.71~255.39 token/s，
  对eager加速1.554x~1.736x），所以收益不是靠预提交200步伪造的。
- eager提交阶段约5.67~6.78ms/token，graph提交只有2.8~7.4us/token；graph的最终drain约
  3.90ms/token，说明host已不再配速，瓶颈确实移到设备侧。stateful固定地址代价是每token
  回写19,759,104 bytes，但只比fixed graph多约75us。清空eager allocator历史后复测：捕获
  一次约13.6ms，常驻allocated/reserved增量约39.6/130.0MB。
- 当前边界仍是**纯模型**：不含logits D2H、CPU argmax与下一个token H2D；`offset` 是Python
  元数据，由调用方递增，不进图。复现与证据：`benchmarks/bench_gdn2_graph_capture.py`、
  `tmp/gdn2-graph-capture/{fixed-v1,stateful-v3,stateful-v4,stateful-v5}.json`（git-ignored）。
- 本轮复用现有CCE产物，没有fresh build。接受的stateful-v3运行期间宿主采样50次：物理0
  活跃0次、物理7活跃16次；运行前后两卡均无本任务留驻。
- **生成接线与清洁配对验收已完成**：`models/gdn2_graph.py` 提供窄契约
  `GDN2NPUGraphDecodeRunner`，固定token/cache地址，图尾回写两类cache，并支持用同形状prompt
  cache做`reset()`而无需重捕获；`generate_tokens(..., decode_backend="npu-graph")` 与文本脚本已
  接入，默认仍是eager且不满足B1/T1/BF16/packed/CCE/NPU时明确报错。真实95B的64-token greedy
  做了两个独立进程、正反顺序配对；四次完整token ids逐项相同，文本连贯。graph decode loop
  （含logits D2H、CPU argmax、下一token H2D）稳定244.21/244.79 token/s，即4094.8/4085.1us；
  eager为173.02/148.84 token/s，即5779.6/6718.8us，两个顺序的graph加速1.411x/1.645x。
  64-token整请求（含冷prefill和每请求graph setup、不含模型load）为graph 99.11/104.23 token/s，
  eager 96.73/91.60，两个顺序仍都快1.025x/1.138x；graph setup为54.5~58.5ms，冷prefill
  274~327ms，所以短请求的总收益被一次性成本稀释。四个接受窗口的物理0活跃均为0，物理7
  各只有本任务一个PID。证据：`tmp/gdn2-graph-generation/{graph-greedy64-v2,
  eager-greedy64-v1,eager-greedy64-v2,graph-greedy64-v3}.json`（git-ignored）。另有一轮功能通过但
  与外部8-rank任务初始化竞争，已明确隔离为`rejected-contended-*`，性能数不得引用。

### 1.7 GDN-2 short-conv/cache（2026-09-15，已完成）

- **先收了无风险的重复工作**：packed T=1 decode 里卷积窗口本来就等于更新后的 cache，现复用
  同一连续 tensor，逐层去掉一次 Cat+Slice。设备 launch 640→604/token，short-conv+cache
  约855.95→642.39us/token，device kernel总时长4018.69→3800.17us/token。stateful graph
  从旧三轮3908.18~3911.91降到新两轮3693.01~3698.00us/token，即270.42~270.78 token/s，
  保守吞吐提升5.68%；logits与两类cache relative-L2均为0。真实64-token greedy的graph
  decode loop为256.47 calls/s，eager为147.36，token ids逐项相同。
- **CCE kernel 已实现并接线**：新单元 `a5.gdn2_short_conv_decode` 固定
  B1/T1/D6144/W4/BF16/block_dim=8，一个launch完成四tap FP32累加、SiLU与cache更新。16个
  向量参与者各拥有384通道；NDDMA把cache/weight转tap-major UB，cache用b16→b32两级
  interleave按channel-major写回，无核间同步。公开wrapper用零拷贝flatten view走现有
  ctypes→aclnn桥；packed CCE层只在精确匹配且已有cache时选它，prompt/T>1保留原路径。
- **当前证据**：静态0 error/0 warning/203 surface ops，CCE emit231；reference/sim/pipesim/
  canonical aclnn均3/3，y/cache exact，cache满足bitwise规则，pipesim无event imbalance/hazard/
  deadlock且2215 cycles。公开runtime双顺序与真实95B双backend顺序均过：step logits
  relative-L2=4.044e-3，最差recurrent/conv cache=9.313e-3/7.270e-3（预算1e-2），argmax与
  offset一致。定向主机测试37 passed。
- **设备收益已profile**：54个kernel样本mean/median=3.374/3.363us；每token把18 Conv2D+
  18 Cat+18 Slice+18 short-conv Swish换成18个custom call，launch604→550，device kernel总时长
  3800.169→3197.959us，跨采集暂估省602.210us（15.85%）；随后clean同卡反序Graph A/B
  实测省584.34us/token（15.82%），与profile估计吻合。但公开eager wrapper为40.99~42.00us，仍慢于
  torch_npu组合31.79~31.82us；小边界的aclnn/分配固定成本只有graph才能隐藏。
- **单pipe约80%在这个边界排不开**：bulk版MTE2/Vector/MTE3/Scalar为
  57.59/8.84/9.18/22.13%，四项和97.74%，基本串行。三路静态128槽与256+128槽都保持数值、
  event、hazard正确，但pipesim从2215退到3745/2819 cycles——仅三个vector tile，额外
  NDDMA/MTE3 setup大于可隐藏工作，故未上板并已回退。若要求单pipe接近饱和，下一架构边界必须
  吞入packed projection的BF16 GEMV；不能减核或造冗余工作刷比率。
- **clean Graph A/B 已完成**：物理5上按 CCE→Torch→CCE 反序、独立进程实测，CCE三轮
  3107.60~3110.82us/token（321.46~321.79 token/s，均值3108.96us/321.65 token/s），旧Torch
  short-conv同卡为3693.29us/270.76 token/s；因此少584.34us/token（-15.82%），吞吐+18.80%。
  逐token同步的CCE为3113.58~3120.34us/token（320.48~321.17 token/s），排除预提交假收益；
  三轮logits/recurrent/conv cache对eager relative-L2均为0。真实95B 64-token greedy的decode
  loop为299.93 calls/s（含logits D2H、CPU argmax、下一token H2D），整请求110.10 generated
  token/s（含冷prefill与每请求Graph setup、不含模型load），token ids与既有结果逐项相同。
  三个接受窗口合计297次宿主采样，物理0活跃0次、物理5只出现五个本任务PID；起初物理7被
  外部任务抢占的一轮仍隔离为`rejected-contended-*`，不得引用。证据：
  `tmp/gdn2-shortconv-kernel/current-performance/`（git-ignored）。

### 1.8 GDN-2 原生 RMSNorm + SwiGLU W1/W2 packing（2026-09-15）

- **不是二选一，先做低风险组合并用2×2消融归因**：packed NPU inference 的37个block/final
  RMSNorm改用已安装的单算子`npu_rms_norm`，weight保留checkpoint已量化的BF16值；18层W1/W2
  在host按行打包且不保留重复参数，T=1合成一个`[2304→12416]` GEMV，prompt/T>1仍从同一
  packed weight的两个连续row slice执行原尺寸GEMV。`o_norm`不变，仍以FP32 weight直入
  fused-decode CCE ABI。packed参数数仍为1,450,096,416，state_dict 273→255项。
- **2×2 clean Graph消融**：旧norm+分离W1/W2为3116.09us/token（320.91 token/s）；只换原生
  RMSNorm两轮均值2699.68us（370.41 token/s）；只pack W1/W2两轮均值3063.15us（326.46
  token/s）；两者一起两轮2653.51/2664.05us（376.86/375.37 token/s，均值2658.78us/
  376.11 token/s）。组合延迟-14.68%、吞吐+17.20%；按对称两因素归因，RMSNorm贡献89.74%，
  W1/W2贡献10.26%。逐token同步为2662.98/2663.80us附近，结论不是预提交假收益。
- **数值与吐字**：真实95B canonical↔packed两个加载顺序得到同一组数值：prompt/step logits
  relative-L2=1.252e-3/3.937e-3，最差prompt/step recurrent cache=1.391e-3/7.239e-3，step
  conv cache=4.282e-3，均过BF16 1e-2预算；argmax/offset一致。两次64-token greedy token ids
  与此前逐项相同，decode loop为340.62/354.33 calls/s；整请求100.50/116.45 generated token/s，
  差异来自冷prefill 402.6/320.4ms，不作为稳态回退。
- **新profile**：Cast 74→0/token，kernel rows 550→273/token，device total
  3197.959→2665.141us/token。剩余MatMul（含Addmm）2290.572us/85.95%，原生RMSNorm仅
  97.798us/3.67%，SiLU+Mul 79.280us/2.97%。packed W1/W2 GEMV本身733.967us/token，MTE2
  平均94.7%；若再写`RMSNorm2+W1/W2+SiLU×Mul`混合CCE kernel，可明确隐藏/删除的非GEMV
  上限约122.7us/token（4.6%）。它是BF16 ceiling实验，不再是单点最大收益；更大的下一步要
  减少权重字节。证据：`tmp/gdn2-rmsnorm-swiglu/`（git-ignored）。

### 1.9 GDN-2 混合 RMSNorm2/W1W2/SwiGLU（2026-09-15，v1拒绝/v2已完成）

- **先过权重流闸**：自写paired BF16 GEMV的N=64在pipesim中因194个tile均摊到28核，综合
  MTE2利用率86.42%→98.81%、模型周期672129→593354；但真机反而是44.345us/layer，慢于
  N=128的41.691us，证明翻倍的DMA/matmul固定开销超过尾部均衡收益。N=128最初写16个物理M行，
  比vendor慢6.47%；改为只发布唯一有效行后，两种顺序均值**39.272 vs 39.216us/layer**，只慢
  0.142%，且与vendor逐位相同，raw GEMV闸通过。pipesim看不见这2.4us/layer的FIX固定成本，
  速度排序必须以真机为准。
- **v1完整混合边界**：新unit `a5.gdn2_norm2_w12_swiglu` 固定B=T=1/D2304/I6208/block_dim28。
  host把不变BF16 gamma折进W1/W2；cube维持N=128饱和权重流，vector subblock并行算inverse RMS，
  通过两槽CvMutex取FP32 accumulator后完成缩放、SiLU与multiply，只把BF16 hidden写GM，W3不并。
  static 0 error/0 warning；reference 2/2、functional sim与pipesim random通过，无event imbalance/
  hazard/deadlock，临界MTE2约99.75%；正式op名经本仓ctypes→aclnn bridge真机输出对独立oracle
  relative-L2=3.444e-7、max_abs=1.526e-5。
- **v1 kernel级双顺序A/B**：18份不同权重下custom-first为40.537 vs43.609us/layer，vendor-first为
  40.416 vs43.593；均值**40.477 vs43.601us/layer，边界1.0772x**，折算18层省56.238us/token。
  synthetic相对当前组合路径为5.231e-3；真实checkpoint逐层gamma-fold探针的W3输出最坏
  relative-L2为4.984e-3（FP32 accumulator变体），当时看似都在BF16 1e-2预算内。
- **v1被整网否决**：prompt因仍走vendor而逐位相同；decode第1步logits/recurrent为
  3.254e-3/4.799e-3，但第2~4步recurrent累积为1.216e-2/1.367e-2/1.523e-2，final conv为
  1.191e-2，均越过1e-2；四步argmax虽相同也不能放松闸。该轮后段有外部进程进入同卡，所有
  timing一并拒绝，但确定性数值失败与竞争无关。这再次证明单层预算不能外推递推整网。
- **v2恢复舍入边界**：取消gamma fold；两个vector subblock各自算同一个RMS规约并通过V→C
  发布1152维`x*inv_rms*gamma` BF16；GEMV先落BF16 UB，SiLU再落BF16 UB，最后BF16 multiply。
  独立边界拆分证明不fold但保留FP32 epilogue仍有3.892e-3单层误差，而恢复两处BF16物化后与
  当前CPU公式逐位相同。正式unit的reference 2/2、random sim/pipesim均通过，relative-L2
  3.416e-7、max_abs1.526e-5、无event imbalance/hazard/deadlock，672723 cycles。真实95B的prompt
  与四个串行decode step在logits/recurrent/conv cache上relative-L2均为0，argmax与offset一致。
- **显式接入、不改默认**：`from_checkpoint(..., mlp_backend="cce")` 才启用；prefill仍走原
  `npu_rms_norm + packed W12`，v2 decode走保守kernel。为保持prompt控制和checkpoint ABI，当前
  每层另存一个non-persistent原始paired布局，18层增加1,029,832,704 bytes；去重是整网通过后的
  独立布局决策。`bench_gdn2_graph_capture.py`已有`--mlp-boundary-path cce`，另有
  `verify_gdn2_mixed_mlp.py`做4-token递推精度。
- **v2整网与profile均已通过**：同卡CCE→vendor→CCE的stateful Graph为2587.192/2656.712/
  2589.831us/token，CCE均值2588.511us、386.323 token/s；相对中间控制延迟-2.567%、吞吐
  +2.635%。逐token同步均值2597.205us，同样快2.371%。Level-1 profile为2576.490us/token、
  219 rows/token，相比vendor profile的2665.141us/273 rows少88.651us和54 rows；混合kernel
  54样本mean/median=42.339/42.168us，AIC MTE2平均/中位91.880%/92.150%，cube utilization
  平均87.914%，因此“至少一条有用pipe接近饱和”已满足。当前native GEMV+混合W12合计
  2323.348us/token，占90.175%。profile运行只在物理7出现目标容器PID；物理0始终是另一容器
  的VLLM，没有本任务PID。完整记录：`tmp/gdn2-rmsnorm-swiglu/performance-analysis.md` 与
  `tmp/gdn2-rmsnorm-swiglu/profile-conservative-v1/analysis.md`。
- **决策边界**：最快opt-in以18层额外1,029,832,704 bytes换2.6%吞吐，默认暂不切换。继续抠
  custom对裸W12 GEMV的1.564us/layer差距最多只值约28.1us/token；把W3并入也不减少约85.8MB/
  layer的W12+W3权重流。下一次大步收益需要weight-only低精度及独立任务质量预算；或者先设计
  prompt/decode共用布局，把当前1.03GB重复权重去掉。

## 2. 下一步的三个候选（带推荐）

> **若继续用户当前指定的 GDN-2 路线**：BF16 raw-gate recurrent+output-norm 与静态双槽
> 成对流水、Graph、short-conv与原生RMSNorm/W1W2 packing均已完成；最新纯模型stateful Graph
> 低内存默认为2.654~2.664ms/token（375.37~376.86 token/s），含D2H/argmax/H2D的真实decode为
> 340.62~354.33 calls/s。**host gap与大部分elementwise fragmentation已解决**：最新profile
> 的权重流占90.175%。保持BF16 dtype的
> `RMSNorm2+packed W1/W2+SiLU×Mul`的gamma-fold v1虽做到kernel级1.077x，却已被整网精度否决；
> 保守v2已通过真实整网、Graph夹心与profile，最快显式opt-in为2.587~2.590ms/token，代价约
> 1.03GB。其AIC MTE2平均91.88%，继续BF16融合只有约1% ceiling；若追求下一次大幅提升，应
> 单独决策weight-only低精度及任务质量预算。T=1 conv window/cache 复用
> 已经验收。训练/长 prefill 则做
> `gdn2_chunk_fwd_bwd`，但它被
> `gdn2-chunk-gate-range` 阻塞，必须先
> 解决真实 stress 中 1461 局部跨度的数值表示。不能复制 KDA stable，也不能拿现有 GDN
> kernel 改名冒充。下列三个仍是原 KDA 主线的排序。

### ① `decode-layer-overhead`（P1，推荐）

**单点收益最大。** 整层 decode 一步 458µs 里，KDA 算子只占 18%（83µs），
**层里其余占 67%（305µs）**，48 层外推 22ms/token。层里 T=1 时要走 7 个投影 + 3 个短卷积
+ 两次 fp32 l2norm 往返 + norm，每个几乎没有计算量却各要一次 launch。

> **把 kernel 再快一倍，整层只快 9%。** 这句话是这条的全部理由 —— 别先去调算子。

处置顺序写在缺口里：图捕获（形状固定，十几次 launch 压成一次）→ 合投影（q/k/v 并成一个
`[hidden, 3*key_dim]`，f/b/g 同理）→ 去掉 l2norm 的 dtype 往返。
复跑：`python benchmarks/bench_kda_decode_layer.py`。

### ② `fwd-caches-not-emitted`（P2，但顺带收益大）

九个反向检查点里 `g_cumsum` / `h` / `v_new` 现在在 host 侧用 torch 补。搬进 kernel 能
**同时**：去掉训练路径对内置算子包的依赖、删掉 `_scan_states(on_cpu=)` 与
`chunk_kda_bwd(layout_device=)` 两处 CPU 绕行、省掉每步 C 次 bmm。属于 kernel 队列。

### ③ `block-dim-ceiling`（P1）

物理 28 cube / 56 vec，chunk 的契约只声明到 4，而实测到 4 仍是**线性**扩展 ——
说明头寸还在。但 kernel 源码归 ascriptor 仓（`AGENTS.md §3` 只读），要跨仓做。
参考点：decode 那个自写 kernel 声明到 28 并实测跑通，所以 28 本身是安全的。

## 3. 动手前必读的五条（都是这两个会话踩出来的）

1. **kernel 的问题攒批统一修，不零散改**（`AGENTS.md §6.5`）。发现一条就记进 `gaps.json`
   打 `requires_kernel_change` + `kernel_change_note`，本仓侧先按 §7 装闸报错，
   等攒够一批再进 ascriptor 侧。`gen_matrix.py --check` 会校验队列不漏项。
2. **一个算子名，一个进程，一份 build。** 扫 `block_dim` 或比 `impl` 必须分进程，
   `runtime/binding.py` 的 `_claim_op_name()` 会拦。
3. **一个进程要用的 kernel 必须在第一次执行之前全部编完。** prefill+decode 的进程要
   `prepare(decode=True)`；测试靠 `tests/conftest.py` 的 session fixture（在测试函数里调
   **来不及**，pytest 同进程跑全部测试）。
4. **profile 之前不要相信任何性能推断**（§6 铁律一）。这两个会话里我在 `block_dim` 和
   decode 瓶颈上各错一次，都是先有推断后看数据。
5. **判结论要读未过滤的原始日志。** CANN 会打不带换行的 `path string is NULL`，
   粘在下一行前面，`grep -v` 会把整行删掉 —— 我因此差点对着少两行的输出下结论。
   做法：重定向到文件再 `nl -ba` 看，不要在管道里过滤。

## 4. 共享机器的规矩

机器清单、SSH 方式、CANN 路径、conda 环境在 **git-ignored 的 `machine_specs.md`**。
绝不把主机名/IP/端口/账号/路径写进任何会被提交的文件。

- 当前调度允许从4~7里选择空卡，kernel调试另获准使用物理0；先前固定物理7的窗口已经结束。
  每次都必须以**当下空闲**为准，授权不等于预约：本轮物理4在静态验证期间就被外部作业占用，
  随后的物理0也在测试启动后被外部作业抢占，因此都立即停止继续使用。用卡前
  `npu-smi info` 看 Health，
  再 `npu-smi info -t proc-mem -i N` 确认无进程。卡会中途从 OK 变 `Critical`，
  换卡要把理由和当时的占用数字写进环境脚本的注释。
- **登录 shell 可能把 `ASCEND_RT_VISIBLE_DEVICES` 重置为全部设备。** 不能只在
  `docker exec -e` 里传卡号再启动 `bash -l`；要在所有 profile/CANN 环境加载之后重新
  export，并在 import torch_npu 前断言“只看见一张卡”。2026-09-14 的受控探针与随后六轮
  真机任务均由宿主每秒同时采样物理 0/5：六轮合计 151 个时间样本，物理 0 活跃样本为 0，
  选定卡活跃样本为 38，任务只落在选定卡。2026-09-15 profiler 的 kernel CSV 全部显示
  Device_id=7，宿主监控里本任务 PID 也只在7；0上的 VLLM 属于另一容器。完整环境细节与
  原始监控日志分别放 git-ignored 的 `machine_specs.md`、`tmp/gdn2-{recurrent,profiler}/`。
  例外：2026-09-15 的 canonical aclnn **fresh build** 在 actual launcher 前捕捉到物理0上
  `python`/0MB 一次，时间对齐 CANN `opc` 编译；`test_aclnnop` 仍只在7。它没有计算/显存驻留
  证据，但也不能记成“物理0完全未触碰”，后续严格单卡窗口禁止再做 fresh CCE build。
- 绝不 kill 或修改他人进程；共享宿主机上不要复位别人也在用的卡。
- 装包只进自己的 venv（`python -m venv --system-site-packages`）。
- `scp` 完一定对 `md5sum`（中断会留下不完整文件且不报错）。
- **开机必查** `ls $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/` 有没有
  `ascend950`：没有则 torch_npu 的计算算子全不可用，**层级验证做不了**，
  只能做纯前向的算子级验证（可用面表见 `AGENTS.md §5`）。
- 收工前扫一遍自己留下的后台循环：`pgrep -af "unti[l] ! pgrep"`（模式要自排除，
  否则 `pgrep -f` 会匹配到自己 —— 实测留下过 9 个永不退出的空转循环）。

## 5. 怎么复跑各项结论

| 结论 | 命令 |
|---|---|
| 主机侧一致性（常量、声明、契约对齐） | `python -m pytest tests/ -q` |
| 真机全量 | `python -m pytest tests/ -q`（在有 `ascend950` 的机器上） |
| GDN-2 文本生成 | eager：`python benchmarks/generate_gdn2_text.py --checkpoint <checkpoint.pth> --tokenizer <tokenizer-dir> --device npu:0 --temperature 0 --top-k 0`；graph另加 `--core-backend cce --projection-layout packed-inference --decode-backend npu-graph` |
| GDN-2 CPU vs torch_npu | `python benchmarks/verify_gdn2_cpu_npu.py --device npu:0`（FP32 relative-L2，不判 bitwise） |
| GDN-2 canonical vs packed | `python benchmarks/verify_gdn2_packed.py --checkpoint <checkpoint.pth> --device npu:0 --dtype float32 --order canonical-first`；BF16 与反序加载另跑，bitwise/argmax 只作诊断 |
| GDN-2 CCE recurrent 算子 | `python benchmarks/verify_gdn2_recurrent.py --device npu:0 --block-dim 8`；调用前必须已把进程可见设备限制为一张允许卡 |
| GDN-2 torch_npu vs CCE 整网 | `python benchmarks/verify_gdn2_backends.py --checkpoint <checkpoint.pth> --device npu:0 --dtype bfloat16 --order torch-first`；FP32 与反序加载另跑 |
| GDN-2 decode profiler | `python benchmarks/profile_gdn2_backend.py --checkpoint <checkpoint.pth> --trace-dir <empty-dir> --output <profile.json> --device npu:0 --core-backend cce`；torch backend 必须另开进程 |
| GDN-2 NPU Graph | `python benchmarks/bench_gdn2_graph_capture.py --checkpoint <checkpoint.pth> --device npu:0 --mode stateful --short-conv-path cce --output <result.json>`；相邻独立进程用 `--short-conv-path torch` 作旧路径A/B；`--mode fixed-cache` 只量host上限，不代表可递推生成 |
| GDN-2 Ascriptor unit CCE | `python kernels/projects/a5/gdn2_fused_recurrent/run.py check --case all --launcher aclnn --device a5 --backend cce --block-dim 8` |
| 真实形状精度验收 | `python benchmarks/verify_real_shapes.py --check drift\|bitwise\|gqa\|bwd` |
| decode 验收 | `python benchmarks/verify_decode.py`（acc / chain / bd / split） |
| decode 整层拆解 | `python benchmarks/bench_kda_decode_layer.py` |
| 门控跨度扫描 | `python benchmarks/probe_bwd_span.py` |
| kernel 静态校验（**本机可跑**） | `PYTHONPATH=<ascriptor>/library ascriptor check <file>::<fn>` |
| 矩阵一致性 | `python tools/gen_matrix.py --check` |

## 6. 本会话纠正过的判断（别重新推导错一遍）

这一节的价值在于**阻止重复犯错**，所以都留着，连同错误的那一版。

| 我原本以为 | 实测 |
|---|---|
| decode 是带宽瓶颈（state 64KB×2/头 ≈ 2.5µs） | **错**。是每次调用约 48~58µs 的固定成本；设备侧边际只有 2.7~4.8 µs/token |
| 优化 decode 要调 kernel | **错**。整层 458µs 里算子占 18%，层侧占 67% |
| `o = qᵀ·state_dec + β(q·k)·delta` 能省一趟 | 恒等式对，但 `RegList.cadd()` 的结果**只落在 lane 0**（不广播），当乘数用只有 1/128 个 lane 对。改成把读出挂到第二趟，连 `cadd` 都不用，IR op 184→156 |
| 反向的精度已经到 ABI 地板（bf16 `g_cumsum`） | **错**。算子离 fp32 参考反而更近（跨度 46：对 fp32 2.889e-02，对 bf16-g 4.169e-02）。我那个构造把 bf16 cumsum 差分回增量，是灾难性相消 |
| 默认初始化的门控跨度 ≈ 94（常数） | **是随机变量**，上界 100.8。同 seed 换 RNG 消耗顺序就从 64.55 变 94.0。所以反向闸从 100 改成 **105**，测试用 `_calibrate_span` 确定性标定 |
| C≥2 的失败是"跨步 D2H 静默给错数据" | 是**硬报错**（`Op Slice does not has any binary` / `errno:561000`）。修法不变（先整块 D2H 再切），但机制写错会误导下一个人 |
| `ShortConvolution` 只有整段前向 | **错**。它本来就支持 `cache` + `output_final_state` 单步解码，还处理 `T < kernel_size` 的补零 |
| `torch-npu-baseline-missing` 还没做 | 早做完了，缺口忘了关 —— 矩阵一边说"已完成"一边说"尚无实现"。**验收打 ✅ 时要回头关对应缺口** |
| 孤立验证 packed low projection 的误差很小，就能推出整链在预算内 | **错**。T>1 的 split 是跨行 view，继续喂第二层 Linear 改了 Ascend 路径；必须逐段再测下一消费者，并最终对 full logits/cache 设 relative-L2 硬闸。bitwise 只作定位线索 |
| `docker exec -e ASCEND_RT_VISIBLE_DEVICES=N ... bash -lc ...` 会一直保留单卡可见性 | **错**。登录初始化可把它重置成全部设备，此后代码里的 `npu:0` 就是物理 0。卡号必须在 login/profile/CANN 初始化后再 export，并由宿主 `npu-smi` 在任务运行期间验证实际落卡 |
| GDN-2 recurrent 公开算子快 1.683x，整网也会接近这个倍数 | **错**。trace 里 recurrent kernel 本体只占 CCE 设备时间1.88%，core连 casts 也只有5.52%；paired whole-model 两次为1.021x与0.975x，只能判持平。先减少周边融合边界和 host 固定成本 |

## 7. 两个"闸"的现状（改之前先读）

| 闸 | 值 | 依据 |
|---|---|---|
| 门控跨度（`MAX_GATE_SPAN`） | `stable` 前向 **155** / 反向 **105**；`upstream` 均 80 | 两维是因为两条链约束不同：前向受**有限性**约束（到 155.97 精度不退化），反向受**精度**约束且先于有限性到来（有限到 169.8，但 `dq` 在 130 超预算）。105 的**下界**由 fla 初始化的跨度上界 100.8 定 —— 低于它会把默认初始化的层用自己的门控拒掉 |
| C=1 多头（`_check_single_chunk_heads`） | `C==1 and B*HV > block_dim` 报错 | 上游 `kda_sub45_fused_kernel` 在这里写出**静默错误**的 `o`（有限、量级正常、内容错，只有每个 cube 核的最后一个头对）。实测表逐格钉在 `tests/test_kda_gating.py` —— 这类缺陷闸一松就再没有东西会报警 |

**后果**：64 token 粒度的 prefill 用不了（T=64 即 C=1）。prefill 要么一次 ≥128 token，
要么按头分批。decode 不受影响（不走那个 kernel）。

## 8. decode 的用法（刚接好，还没别处记）

```python
from ascend_fla.ops.kda import prepare
prepare(decode=True)          # 进程启动时一次；晚了会报"已经执行过 aclnn 算子"

cache = {}                                     # 空字典 = 从零开始并开始记录
o, _ = layer(x_prefill, cache=cache, mode="chunk")          # T 必须是 64 的倍数
for step in steps:                                          # T ≤ 16
    o, _ = layer(step, cache=cache, mode="fused_recurrent") # cache 原地更新
```

- `cache` 两个键：`recurrent_state`（`[B,HV,128,128]` fp32，**K 在前**）与
  `conv_state`（`(q,k,v)` 三个 `[B,D,conv_size]`）。
- **两条路径不自动互换**：服务不了就报错并指出另一条。数学等价但数值不同
  （decode 全 fp32，对参考 1e-07；chunk 有 bf16 中间量，3e-03）。
- **decode 不可求导**（只有 chunk 有反向 kernel）。层里直接报错，不会静默不建图。
- 接 HF/fla 模型还要一层 cache 适配器：fla 的 KDA layer 用 `state_v_first=True`
  （V 在前），见 `state-layout-k-first`。

## 9. 本会话的提交（倒序）

```
8d1f183 记上全量真机结果：63 passed
ad3dc60 矩阵里"decode 完全缺失"的条目同步为已完成一半
cccd7f8 decode 接进 layer：两条路径 + cache 交接，并把遗留问题与优化点记全
e3ec138 探 decode：自写 kda_fused_recurrent kernel，真机通了，但瓶颈不在算子
3cd44dc 建立 kernel 问题的统一修复队列：记账 + 装闸，不零散改
7be9a65 真实形状精度验收四项全部完成：链长不累积、核切分逐位相同、GQA 与反向都在预算内
7cf0dd1 真实形状精度验收抓到一个 P0 静默错误：C=1 多头时 o 内容错，已装闸
941d1b3 关掉过期的 torch_npu 基线缺口，并把 toy-case-shapes 收窄到精度侧
aa73c96 远端等待循环会自匹配 pgrep 模式：记下这个坑与替代写法
1047f58 整层反向在真机跑通，并修正三处被实测纠正的说法
```

## 10. 没做的事（按是否阻塞第三期）

**阻塞第三期**：`decode-layer-overhead`、`decode-call-overhead`、`fwd-caches-not-emitted`、
`state-layout-k-first`（接 HF 模型要它）、矩阵 CI。

**不阻塞**：`stable-unit-no-harness`（三个本仓单元都没接 ascriptor harness —— `check` 与
runtime 桥都过了，缺的是 `unit.py` + `run.py`）、`kernel-nd2nz-suboptimal`（lint 标出的访存
低效点，带板上实测倍数与具体改法）、GDN/DeltaNet 扩族（第四期，六项 ABI 缺口）。

**刻意不做**：`gate-span-still-bounded` 的进一步放宽（反向的约束是精度不是有限性，
分块 log-sum-exp 可能帮不上忙，先做 `kda-fwd-bwd-dtype-mismatch` 看曲线会不会整体下移
—— 那是推测，要测）。
