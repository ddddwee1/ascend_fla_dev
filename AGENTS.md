# AGENTS.md

本仓库是 **fla 系列线性注意力算子在昇腾 NPU 上的高效实现库**，后端用
[ascriptor](https://github.com/ddddwee1/ascriptor)（指令级 Python 编译器）。

> **接手工作先读 `docs/handoff.md`** —— 那里有当前状态、下一步的候选与推荐、
> 以及"这两个会话里被实测纠正过的判断"（别重新推导错一遍）。本文件讲的是长期纪律，
> 那份讲的是此刻进度。
>
> **如果你是来申领任务的 agent**（任何账号、任何模型）：再读 `docs/pm/PROTOCOL.md`、`docs/pm/START.md`
> 与你的 `docs/pm/tasks/<ID>.md`。协作走本仓（公开）的 GitHub issue 与 PR：在 `task/<ID>` 分支上工作、以 PR 交付，
> 只采信 `docs/pm/board.json` 里 `pm_github_login` 账号发的派单。SoC 顺序已于 2026-09-14 改为 A2 → A3 → A5（§2）。

## 1. 定位

- **是什么**：独立的昇腾算子库。实现 GDN / KDA / DeltaNet 等 fla 系列算子的
  forward + backward，以 torch 可调用算子对外暴露，目标是**比现有方案更快**。
- **不是什么**：不是 fla 的 fork，也**不是 fla 的后端插件**。我们不接 fla 的
  `@dispatch` / `BackendRegistry` 机制，不在 `fla.ops.*.backends` 下注册。
  公共 API 由本仓自己定义。
- **fla 在这里的两个角色**：① 算子语义的权威定义；② 各算子族的 `naive.py`
  充当 CPU fp32 oracle。仅测试期依赖，运行时不依赖。

> 架构纪律：有人提议"顺便注册进 fla 的 dispatch"时，这是在改变仓库定位，
> 需要显式决策，不要顺手做。往 fla 方向回摆的代价是跟随上游的长期维护成本。

## 2. 四条已定的决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 目标 SoC | **A2 (910B) → A3 (910C) → A5**（2026-09-14 改） | 用户决定首波算子与整模型先在 A2/A3 落地；A5 上已完成的第一、二期保留，下一波再回来。风险见下 |
| 与 fla 关系 | **纯算子库** | 定尺约束远窄于 fla 公共 API 的承诺范围；做独立库才能把约束写进契约，而不是塞进 verifier 的拒绝理由 |
| 首期范围 | **fwd + bwd** | 面向训练；反向资产已有，不做等于浪费 |
| 首个算子族 | **KDA**（非 GDN） | 第 0 期 ABI 对比的结论，见下 |

**首个目标是 KDA 链路，不是 GDN。** 第 0 期把两者 ABI 逐项对出来后发现，KDA 在六项
能力上都更接近 fla 语义：GQA 分组、token-major 公开布局、非零 `initial_state`、
backward 产出 `dh0`、`final_state` 为 FP32、`block_dim` 上限 4。GDN 只在"本地验证证据
齐全"一项上占优，而那是可补的。对照表见 `docs/matrix/README.md` 的"为什么首个目标是
KDA"，依据见 `gaps.json` 的 `summary.kda_vs_gdn`。

首要目标模型相应是 **Kimi-Linear-48B-A3B**；第二个是 Qwen3-Next（GDN），它受 `gdn-no-gqa`
等六项 ABI 缺口阻塞，要先过 kernel 批次。**不要因为 GDN 更知名就调换顺序** —— KDA 先。

> **例外（D-PM-20，2026-09-17，用户批准）**：agent 轨道在 A5 上开一条**窄范围**的 GDN
> 前向任务（`GDA-01`），显式跳过 `gdn-no-gqa`（不支持的 GQA 组合直接报错，不是解决它），
> 复用现有 `a5.gdn_fwd` 契约做只读起点。这**不是**把 GDN 调回第一优先级——KDA 仍是首个
> 目标算子族，Qwen3-Next 仍卡在六项 ABI 缺口上不动。这条例外只覆盖"非 GQA 的 GDN 前向
> 在 A5 上跑通 + 优化"这一件窄事，理由是有申领人已配好 A5 真机访问、愿意做。跟
> GDN-2（D-PM-16）一样，是范围内的显式例外，不改变上面这两条决策本身。
>
> **例外的排期（D-PM-22，2026-09-18，用户直接给出，不是 agent 推断）**：GDA-01 完成后，
> 用户明确批准了后续顺序——**GDN 前向（已完成）→ 给 GDN 补 GQA/GVA 分组（GDA-02）→
> PGDN 前向（PK-03，此时解锁）→ backward → decode → 性能**。这条排期仍然**只覆盖
> GDN/PGDN 这一条线**，不代表 KDA 首位或 A2→A3→A5 波次顺序发生了任何变化——同一时期
> 一个更宽泛的"GDN 全族推进"提案（issue #80）被用户明确否决，只批准了这条具体顺序。
> `PK-03` 的 `gate: prereq-gdn-abi` 在 GDA-02 完成后由 PM 解除，不由 assignee 自行判断。

### SoC 顺序（2026-09-14 用户决策，取代原来的"A5 优先、不要分叉去做 A2"）

**首波在 A2 (910B) 上做 KDA 算子 + Kimi-Linear 整模型，然后 GDN + Qwen3-Next；再做 A3 (910C)；
最后回到 A5。** 工具链仍只用 ascriptor。第一、二期在 A5 上的全部结论保留，但**只属于 A5**。

接 A2 的活之前，先知道这两条风险。

ascriptor 侧 A2/A3 在 2026-09-06 被 deferred（D-250），并带着一类 cube 累加数值缺陷（库里编号 M10-081：A2 系上两次短 MMAD 写同一块 L0C，硬件不互锁，第二次读到没落定的累加器）。
**2026-09-19 按 A2-01 的调查改写**（此前这里写的是"一个没解决的 split-K FP32 cube 缺陷"，那个说法太窄；用户同意改写）：

- **FP32 的 split-K 在 pin 里已有修复**：pin 版库（90cfcdc）在 a2 系且 A/B 都是 FP32 时，于 split-K 展开处自动插 `PIPE_M` barrier（`ascriptor/passes/desugar.py:411`，PM 读源码核实）。
  申领人在 910B3 / CANN 9.0.0 真机上复现，FP32 split-K 逐位（自述，证据随 A2-01 的 DONE）；**A2-11 在 CANN 9.2.0-beta.1 上用独立 CPU fp64 参考重新复现，2000/2000 逐位一致，结论确认**。
- **没解决的有两类**（A2-01 自述、A2-11 真机定量复现）：① **BF16/FP16 的 split-K**——同一条修复按 dtype 把它们排除在外，M16 在 910B3 上 **500/500 次全错、500 个输出哈希两两不同**（完美非确定 = 硬件时序竞争，非确定性miscompile；最小二乘分解显示"隔片丢弃"的陈旧累加器系数特征）、M32 **200/200 次触发 AI Core 异常**（`aclrtSynchronizeStream failed: 507015`）、M64 199/200 逐位正确（`gaps.json` 的 `a2-splitk-bf16-fp16-unsettled`，A2-11 证据见 `docs/research/a2_splitk_gate.md`）；② **FP32 的手写 MMAD 累加链**——库里只有 lint 告警、不自动修，KDA / GDN 的三角求逆与 GDN 反向 finalize 各有一处，正是 M16 形状
  （`a2-splitk-fp32-cube`；A2-11 加时序扰动跑了 2000 次未复现出错，**如实记为"未复现"，不是"安全"**，保守保留 `barrier(Pipe.M)` 绕行）。
- **A5 不受影响**（现有 25 个在用 kernel 都不用 BF16/FP16 split-K，且这是 A2 系的硬件行为）；**A2 派生单元要逐个核对**，线索见 `docs/pm/tasks/A2-03.md`。
  D-PM-35 的「BF16 优先」叠加 A2 优先：BF16 的 KDA 在 910B 上要先过②这一关。

**A2-11（PR #130 → `652dfc0`，2026-09-23）已完成真机复现与绕行验证，用户已裁定接受其结论与建议**（含研究文档列出的四项前提：数字均出自 pin 修订 `90cfcdc`/`b3b3f9c` 并带 SoC/CANN/算子包身份；已合入的 a2 单元过静态守卫 `tests/test_a2_accumulate_barriers.py`；FP32 split-K 修复真机确认逐位；FP32 手写链的"未复现"残余风险按显式守卫携带、不当作缺陷不存在）——**A2 算子结论从此不再仅是观测，满足以上四项前提的可以算数**。`barrier(Pipe.M)` 绕行本身逐位稳定（2000/2000，多 `bd`、多卡），也已在真机验证。

A2 的核数、UB、`block_dim` 上限、门控跨度上限、内置算子包覆盖，眼下一个都不知道，全要重测（见 §6「结论不跨 SoC 继承」）。
A5 上 profile 声明与物理核数对不上会死锁，A2 同样要先查实这一条。

协作方式：本仓用多 agent 推进，派单与汇报协议见 `docs/pm/PROTOCOL.md`，看板是 `docs/pm/board.json`。

## 3. 与 ascriptor workspace 的关系

ascriptor 是 `library/` + `kernels/` + `agent/` 三个同级 checkout 组成的 workspace。
本仓通过 `$ASCRIPTOR_WORKSPACE` 定位它（未设置时按同级目录 `../ascriptor` 查找）。
进入那边工作前先读它自己的 `AGENTS.md`，
并按 `agent/compatibility.json` 选定 library/kernels 修订。

**权威来源在 gitcode，不在 GitHub**（2026-09-17 由仓库所有者给出）。三个 checkout 分别来自：

```bash
git clone https://gitcode.com/ddddwe/ascriptor.git         "$ASCRIPTOR_WORKSPACE/library"
git clone https://gitcode.com/ddddwe/ascriptor-kernels.git "$ASCRIPTOR_WORKSPACE/kernels"
git clone https://gitcode.com/ddddwe/ascriptor-agent.git   "$ASCRIPTOR_WORKSPACE/agent"
```

本文件开头那个 `github.com/ddddwee1/ascriptor` 链接**取不到**（匿名与登录都 404）。
两个外部 agent 因此各自去 fetch 了别处的镜像、拿到互不相同的修订，其中一个还据此报了风险。
**别再走 GitHub。**

- 我们**复用**它的算子单元（`kernels/projects/a5/{gdn,kda,delta_rule}_{fwd,bwd}`）
  和算法单元（`chunk_row_scan`、`matrix_normalization`、`gated_approximations`）。
- 我们**不修改** ascriptor 仓。需要改 kernel 时，在本仓 `kernels/` 下按它的
  unit 协议建自己的单元（`unit.py` 导出 `make_inputs/reference/execute` +
  `contract.json` + `run.py`），保持可独立运行。
- 本仓实际使用的 ascriptor 修订记录在 `docs/matrix/ops.json` 的 `ascriptor_pin`。

## 4. 本仓要自建的关键能力：runtime 桥

这是第一期唯一的真实技术风险，也是整个仓的技术护城河。

ascriptor 现在的执行模型全是"**落盘 + 独立进程**"：`aclnn` launcher 写二进制参数
文件后跑独立的 `test_aclnnop`；`board`/`pypto` 通过 SSH 推源码和输入到远端；返回值
是 `torch.frombuffer` 重建的 **CPU** tensor。**它是 kernel 开发/验证框架，不是可嵌入
的运行时算子库。**

`ascend_fla/runtime/` 要补的就是这一段：把 ascriptor 生成的 CANN 自定义算子编译成
常驻 `.so`，经 `torch.library` 在进程内调用，直吃 NPU device tensor、零拷贝。
地基是 ascriptor 的 `runtime/aclnn/template/`（op_host + op_kernel + CMake 工具链）
和 `build_custom_op()`。

> ⚠️ 已知风险：六个 a5 单元的 `compile` 与 `cannsim` stage 全是 `untested`
> ——**真机 passed 走的是 SSH board 路径，不是我们要用的本地 aclnn 编译路径**。
> 第一期要先证明这条路通，再谈算子接线。

## 5. 硬件与远程环境

**本机是 macOS，没有 NPU。所有真机验证都在远程 Ascend 机器上执行。**

主机清单、SSH 方式、CANN 路径、conda 环境写在 git-ignored 的 `machine_specs.md`
（本仓尚未建立时，参照 ascriptor `agent/machine_specs.md` 与 ascriptor 的
`boards.json`）。

> 绝不把主机名、IP、端口、账号、路径写进任何会被提交的文件 —— 包括本文件、
> 脚本、注释、commit message。需要引用时写"见 `machine_specs.md`"。

- 按目标 SoC 选机器。**不要把一台机器的 CANN/torch 版本或结论套用到另一台。**
- 共享机器上跑任务前先看 `npu-smi info`；有别人的活跃任务就等，**绝不 kill
  或修改他人进程**。
- 远程工作副本通常是 `rsync` 的普通拷贝，不是 git checkout —— 不要在远端 `git pull`。
- **装包用自己的 venv**。共享 conda 环境可能同时服务别的项目，往里 pip install
  会污染别人。做法：`python -m venv --system-site-packages <工作区>/venv`，
  复用宿主的 torch/torch_npu，自己的包只进 venv。
- 传大文件要留意带宽：整包 `git archive` 往往有大量 examples/docs，
  只打包 `ascriptor` 包 + `pyproject.toml` 能把 29MB 压到 2.7MB。
  `scp` 中断会留下**不完整**的文件且不报错 —— 传完一定对 `md5sum`。
- 等远端长任务结束时，**`pgrep -f <模式>` 会匹配到等待循环自己的命令行**，
  于是 `until ! pgrep -f "pytest tests/foo"; do sleep 10; done` 条件恒真、永不退出。
  实测在容器里留下 9 个空转循环（测试早就跑完了），表现为本地的后台等待命令超时退出
  （exit 124），很容易误读成"测试失败"。**判失败前先看日志有没有正常收尾。**
  写法：盯文件而不是盯进程 —— `until grep -q "passed\|failed\|error" <log>; do sleep 10; done`，
  或给模式加 `[p]ytest` 这类自排除。收工前 `pgrep -af "until ! pgrep"` 扫一遍自己的残留
  （**只清自己容器里自己起的**）。
- 同理，**按噪声模式 `grep -v` 过滤远端日志会连正经输出一起删** —— CANN 会打不带换行的
  `path string is NULL`，它粘在下一行前面，于是那一整行被滤掉。我因此差点对着少了两行的
  输出下结论（8 个张量只列出 7 个）。**要判结论就读未过滤的原始日志**：重定向到文件再
  `nl -ba` 看，不要在管道里过滤。

### 开机必查：opp 有没有 `ascend950` 算子包

**同为 Ascend950PR，不同机器的内置算子包覆盖不同 —— 这决定了你能做什么。**
进任何 A5 机器先跑：

```bash
ls $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/
```

- 有 `ascend950` → torch_npu 的计算算子可用（实测 randn / zeros / fp32+bf16 matmul /
  cast / contiguous / einsum / cumsum 全通）。性能基线与 layer 级验证都能做。
- 只有 `ascend910*` → **torch_npu 的计算算子全不可用**，下表为实测可用面。

实测：CANN 9.2.0（innerversion V100R001C25B046）的机器**有** `ascend950`；
CANN 9.1.0 的机器**只有** 910 系列。这是算子包安装差异，不是 SoC 级缺陷。

### 内置算子包缺失时的可用面（实测）

| 操作 | 可用 | 说明 |
|---|---|---|
| `torch.empty(device="npu")` | ✅ | 纯分配，不走算子 |
| `.to("npu")` / `.cpu()` | ✅ | H2D / D2H memcpy |
| `data_ptr()` / `current_stream()` | ✅ | runtime 桥需要的就是这些 |
| `torch.zeros` / `randn` | ❌ | 需要 ZerosLike / StatelessNormal |
| 任何 dtype 转换（`.float()`、bf16↔fp32） | ❌ | 需要 Cast |
| `permute().contiguous()`（NPU 上） | ❌ | 需要 d2d copy |
| 任何 matmul / einsum | ❌ | |

**实践后果**（仅在缺 `ascend950` 的机器上）：取值、比较、layout 重排一律**先 D2H
再做**（`t.cpu().float()`，不是 `t.float().cpu()`）。造零张量在 CPU 上造再 H2D。
（2026-09-20 起）`ops/kda` 的布局 / dtype 转换 / 零填充已在自编译 kernel 里，不再依赖内置算子；
`layout_device` 只接受 `auto` / `npu`（等价），`cpu` 显式报错（D-PM-37 不许 host 侧 CPU 布局转换）。
上面「先 D2H 再做」只对测试 / 诊断脚本里的 host 代码有效，不再是 ops 层的绕行办法。

**还有一条更隐蔽的：跨步视图的 D2H 也不可用**，它要走 NPU 侧的 `Slice`。

```python
dev[:, 63::64].cpu()   # ❌ Op Slice does not has any binary / errno:561000
dev.cpu()[:, 63::64]   # ✅ 整块 D2H 是纯 memcpy，切和 contiguous 都在 CPU 上做
```

**规则：先整块 D2H 再切，不要先切再 D2H。** 这条特别容易漏，因为**切片只取一行时
（等效连续）两种写法都能过** —— 我就是这样让 C=1 的两个 case 通过、C≥2 的三个全挂，
绕了一圈才定位。凡是绕 CPU 的代码，**按形状参数取极端值各跑一遍**（这里是 C=1 与 C=2），
不要只试一个。

相应地：算子入口**要求输入连续、不满足就报错**，不"悄悄 contiguous 一下" ——
在这种机器上那件事根本做不到（device 上要 d2d copy，跨步 D2H 要 Slice，两条都缺）。

**我们自己编译的 kernel 在两种机器上都不受影响** —— 计算都在自编译算子里。
这正是 runtime 桥的价值：它让算子在内置算子包不全的机器上照样可用。

> ⚠️ **但这句话只对「纯前向」成立，对「训练」不成立。** 反向要九个前向检查点，其中
> `g_cumsum` / `h` / `v_new` 当前在 host 侧用 torch 补（`fwd-caches-not-emitted`），
> 那一段是 Cast / bmm / stack —— 缺算子包的机器上全不可用，报
> `copy_d2d_baseformat_opapi … 561103` + `Cast ADD_TO_LAUNCHER_LIST_AICORE failed`。
> 这些 CPU 绕行（`_scan_states(on_cpu=)`、`chunk_kda_bwd(layout_device=)`）已随 FMT-02
> （2026-09-20，D-PM-37）删除：缺 `ascend950` 算子包的机器上，默认的门控跨度检查
> （device 上的 cumsum）、带缓存前向与反向里的 `_scan_states`（Cast / matmul）、`dw`
> 取负、`log2(eg)` 分支这些存量 host 算术（D-PM-42 登记的例外）会因缺内置算子而失败，
> 训练路径在这类机器上不可用；纯前向在 `check_gate_range=False` 时可用。要等 BF-07 /
> kernel 批次把它们搬进 kernel 才消除。**层级验证在这种机器上做不了**（投影/卷积/softplus/RMSNorm
> 全是 torch_npu 算子），要换有 `ascend950` 算子包的机器。
>
> 一般教训：**「我们的计算都在自编译 kernel 里」这种论断，要按调用链逐段核对，**
> 不能从"主算子是自编译的"推出"整条链不依赖内置算子"。

**aclnn 相关的硬事实**（写 runtime 代码时会用到）：

- `aclCreateTensor` / `aclDestroyTensor` 在 **`libnnopbase.so`**，
  `libascendcl.so` 里没有这个符号。
- aclnn 参数顺序 = `inputs… + scalars… + outputs… + &wsSize + &executor`。
- aclnn 接口层**放宽** attr 类型：整型 attr 一律 `int64_t`，浮点 attr 是
  **`double`**（不是 `float`）。按 `c_float` 传 4 字节会让被调方从 8 字节槽里读到
  垃圾值 —— 实测表现为 `scale` 近 0，于是**只有用到它的输出归零、别的输出照常正确**，
  极其隐蔽。判类型一律看生成的 `aclnn_*.h`，不要照搬 ascriptor 的 `SCALAR_C`
  （那是 kernel 侧的 C 类型）。
- ACL dtype 枚举：f32=0、f16=1、i32=3、i64=9、bool=12、bf16=27；`ACL_FORMAT_ND=2`。
- aclnn 的 **HostSpec 标量列表包含 GM 形状里出现的全部符号维**，不只是 kernel 签名里
  显式声明的标量。`kda_bwd` 的九个 kernel 都因此多一个 `T`；`kda_fwd` 的五个恰好把符号
  都显式声明了，所以没踩到。**标量名一律以 `CompiledKernel.scalar_names` 为准，不要从
  kernel 签名推断** —— 漏传会报"缺少参数"（这个还算好查），多传或错序则不一定报错。
- 必须让 `ASCEND_CUSTOM_OPP_PATH` 指向 vendor 树，CANN 才找得到算子的 JSON 配置。
  **而且它只在首次算子解析时被读一次** —— 之后追加的路径 CANN 看不见，调用时报
  `rc=161001`，plog 里说的却是"SoC version ascend950 verification failed / 算子包未安装"。
  **这个报错是误导的**：构建产物完好也会这样。所以一个进程要用到的 kernel 必须在第一次
  执行之前全部编译完（`ascend_fla.ops.kda.prepare()`）。
- `ascriptor` 的 `a5` → `950` profile（32 cube / 64 vec），而 Ascend950PR 物理上
  只有 **28 cube / 56 vec**。`block_dim` 超过物理核数会在硬件 barrier 上死锁。
- 由上一条派生的一个实践约束：**一个进程要用的 kernel 必须在第一次执行之前全部编完。**
  既要 prefill（chunk）又要 decode（fused_recurrent）的进程，启动时调一次
  `ascend_fla.ops.kda.prepare(decode=True)`；否则 decode 的 kernel 晚于 chunk 第一次执行
  才注册 vendor 树，报"已经执行过 aclnn 算子"。**测试里在测试函数内部调 prepare 来不及** ——
  pytest 把所有测试跑在同一进程里，所以放在 `tests/conftest.py` 的 session 级 autouse fixture。

## 6. 验证方法论

### 双 oracle

每个算子的精度判定都对**两个**独立参考：

1. **fla 的 `naive.py`**（纯 torch，CPU fp32）—— 语义权威。
2. **torch_npu 组合实现**（同形状在 NPU 上用原生算子拼出来）—— 同时是性能基线。

两个 oracle 之间的差异本身就是有用信息，不要只报一个。

### 判定纪律

- **算子正确性一律在 fp32 下判定。** bf16 的逐元素比对没有判别力，只适合做端到端
  输出质量检查，不要用它判断算子对错。
- **chunk 与 recurrent 两条路径在数学上等价，互为最好的 oracle。** prefill/decode
  一致性（一次前向 vs 逐 token 递推 + state 传递）同理。
- **报数字，不报 "OK"**：给 `max_abs_diff` / 相对误差 / 相对 L2 残差。
- **算子级精度指标不能外推到任务精度**，反之亦然。要声称任务级影响，就得跑任务级
  实验，并且拆出中间对照组（"替换实现"与"改精度"是两件事，混在一起测会把账记错）。
- **失败要留证据**：贴真实输出和报错，不要用"应该没问题"收尾。日志留在 `tmp/<task>/`。

### 性能测量的三条铁律（都是踩出来的）

1. **profile 之前不要相信任何性能推断。** 我在 KDA 上对 `block_dim` 连续判断错两次：
   第一次结论"无效"是因为桥的缓存键每次调用都算一遍（`inspect.getsource` + sha256，
   5 个 kernel 共 ~10ms），把设备侧 0.23ms 淹没在 95% 的 host 开销里；第二次结论
   "无效"是因为同进程的四份 build 互相覆盖。两次都是先有推断、后看数据。
2. **一个算子名，一个进程，一份 build。** `ASCEND_CUSTOM_OPP_PATH` 是搜索路径，CANN
   按算子名查，第一个命中的 vendor 树胜出，解析每进程只发生一次 —— 第二份 build 被
   **静默**忽略。扫 `block_dim` 或标量绑定必须一份 build 一个进程。`runtime/binding.py`
   的 `_claim_op_name()` 会在越界时报错。
3. **缓存键不能比缓存贵。** 凡是放在每次前向热路径上的缓存查询，键的计算必须是 O(1)
   的字典查找级别。读源码、算 hash、遍历文件系统都不行。

`block_dim` 对 KDA 的实际效果（分进程实测，kimi_linear_layer）：三个重 kernel 从
bd=1 的 1.583/1.354/1.311ms 降到 bd=4 的 0.407/0.361/0.329ms，**近乎完美的 4 倍扩展**。
kernel 内部用 `GetVecIdx()/GetVecNum()` 自行切分，而 `GetVecNum() == 2 * block_dim`，
所以 bd=1 只用到 2 个向量核。契约只声明到 4。

### 性能优化任务的验收线（用户 2026-10-27 直接给出）

**每个算子的性能优化不是"比基线快就行"，有一条量化的停止线**：优化到某一类硬件资源
（计算，即 cube/vector 的算术管线；或搬运，即 MTE2/MTE3 这类数据搬运管线）成为瓶颈——
即该资源的 pipe 利用率达到 **85% 以上**——才算到位。利用率算法沿用 PK-07 第二轮已经在用
的 `aic_mac_ratio`/`aiv_vec_ratio`（计算）与 `mte2`/`mte3` 占比（搬运）这类 `PipeUtilization`
指标，真机实测，不是模拟器的 `model_cycles`。没打到 85% 之前，"看起来已经快很多了"不算
验收理由——先说清楚卡在哪条 pipe、那条 pipe 的利用率是多少，再决定是继续优化还是如实
记录"已到物理上限，差距是 xx%"。

**开始优化之前，先把工具链升级到位**：装最新的 PyPTO-Pro 框架，并装
[`pypto-pro-op-orchestrator`](https://gitcode.com/cann/pypto-gym/tree/master/cannbot-skills/plugins-official/pypto-pro-op-orchestrator)
（`cann/pypto-gym` 仓库下的官方插件，不是 `cann/cannbot` 下的社区版——两者容易混，按仓库
路径区分）。这是方法论工具，不是解决某个具体编译兼容问题的手段——2026-10-27 的 PK-08
真机验证已经证实，这台机器上 `library90cfcdc` 对 CANN 的那个编译不兼容问题，换 CANN
版本（9.1.0-beta.1，不是 9.2.0）就解决了，跟 pypto_pro 无关；这条记录留着是为了不让
下一个人把这两件事搞混——装插件是为了用它那套"规划/编码/调试/验证/调优/工具"六个子代理
的分工做优化方法论，不是因为编译需要它。

### 按「量程」失效的缺陷：要把整条链同类算式列一遍

KDA 的门控跨度那件事：`exp()` 的参数超出 fp32/bf16 量程。前向根治之后我差点收工，
反向那一处是**读源码时顺手发现的，不是测出来的** —— 契约的 case 跨度 ≤1.92，离失效线
46 倍远，永远测不到。而且两处方向相反（前向下溢、反向上溢），修法的细节也不同。

**做法**：把整条链上所有同类算式逐处列成表，逐个判量程，再动手。这次列了 14 个 kernel，
命中 5 个（前向 gate/intra/wy，反向 finalize_pre/post），另外 9 个判定为「指数恒 ≤1，
下溢到 0 就是正确结果」—— 那个判断也要写下来，否则下一个人还得重查一遍。

成对量 `exp(a_i − a_j)` 用 matmul 求和时必须分解成两个单边因子，分解的**锚点可以任选**
（配对时抵消）。上游两处都把锚点放在区间端点，于是一个因子顶到 `exp(±span)`；
取中点则两个因子各压到 `exp(±span/2)`，可用量程正好翻倍。改锚点**不改数学**，
但会改 bf16 的舍入位置 —— 实测逐项差在第四位有效数字，要如实说成「同义但不逐位相同」。

**扩了可用域之后，"有限"和"准"要分别测，闸按更严的那个定。** KDA 反向的有限性上限是
跨度 169.8，而精度（对 fp32 递推参考的相对 L2）在 130 就超出契约预算 —— 我最初按有限性把闸
写成 160，那会让调用方在 130~170 之间拿到**有限但超预算**的梯度且毫无提示，正是 §7 要避免的
静默降级。最后闸取 105，并且因为两条链的约束不同（前向受有限性约束、
到 155.97 精度完全不退化），`MAX_GATE_SPAN` 做成 `{impl: {forward, backward}}` 两维 ——
**一个数字表达不了两条链的约束，硬并成一个就会说谎。**

**闸是双边约束，两边都要有实测依据。** 上边界是"预算还成立的最深实测点"（反向 105），
下边界是**调用方真实会送进来的值** —— fla 的 KDA 初始化算出来的跨度上界是 100.8，闸低于它
就会把默认初始化的层用我们自己的门控拒掉。我第一版写 100 就是只看了上边界。而且那个跨度
**是随机变量不是常数**（`A_log = log(U(1,16))`，同 seed 换一下 RNG 消耗顺序就从 64.6 变 94.0，
HV=8 实测 8 个 seed 落在 54.2~100.6），所以测试里要**确定性地把它标定到目标值**
（`_calibrate_span` 平移 `A_log`），不能假设"默认初始化就是 ~94"再去断言。

### 形状维度同理：(C, HV) 这种组合要逐格扫，不能只扫单轴

量程那条讲完了，同一句话在形状上也成立，而且更阴。实测到的 P0
（`c1-multihead-o-corrupt`）：`kda_sub45_fused_kernel` 在 **C=1 且一个 cube 核要连续处理
多个头**时写出**内容错误**的 `o` —— 有限值、`|got| ≈ |ref|` 范数正常、`final_state` 还对，
只有逐元素比值是乱的。正确的恰好是每个核分到的最后一个头（`pair_begin/pair_end` 按
`GetCubeIdx()/GetCubeNum()` 切 `B*HV`，而 `GetCubeNum() == block_dim`，所以安全条件是
`B*HV ≤ block_dim`）。

> **2026-09-17 更正（A2-04，#30 / PR #60，模型定位后 PM 复算）。** 上面这段原来还有一句
> "C≥2 全对 —— chunk 循环第二遍补上了缺的那次同步"。**两半都是错的。**
> ① 根因不是漏了某次同步，是 `Aqk` 的 L1 交接**两信用配固定槽**：
> `aqk_l1_valid` 是 `DEvent(preset=True)`（两信用），而槽 `aqk_slot = Var(c_idx % 2)` 按 chunk 号取。
> ② 因此 C≥2 并非全对 —— 一个头最后一个 chunk 的槽是 `(C-1)%2`、下一个头第一个是 `0`，
> **当且仅当 C 为奇数时相撞**。偶数 C 安全是因为槽恰好交替，不是因为多跑了一遍。
> 真机 C=3 那次"全对"是时序没踩到（pipesim 在 C=3/5 上照样报无序对）。
>
> **这条本身就是教训**：原来的解释是从"C=1 坏、C=2 好"这个**症状规律**倒推出来的，
> 听上去合理、写进文档两期没人怀疑，而它把根因指错了方向（去找"缺的那次同步"），
> 也把安全域说宽了（"C≥2 全对"）。§6.5 那句"动手修之前必须先定位到确切的原因，不是症状规律"
> 说的就是这个 —— 而且**症状规律不只会误导修法，还会伪装成结论写进文档**。

契约的 case 在 (C, HV) 平面上只覆盖 (1,1) 与 (2,2)，而 (1,2) 就是坏的。
**两个维度各自全绿，不等于它们的组合全绿。** 做法：把定尺参数列成表，**对每一对可能交互的
参数取笛卡尔积的边界点**（这里是 C∈{1,2} × HV∈{1,2,…}），而不是每个参数各扫一遍。
**而且边界点要覆盖参数的"奇偶"这类结构性分档** —— C∈{1,2} 两个点看不出奇数 C 是一整档，
扫到 C=3 才看得见。

两条配套的：

- **性能在真实形状上测，精度也必须在真实形状上测。** 我们的性能数一直用 `models.json` 的
  真实形状，而精度数全是 H=1/C≤2 的玩具形状 —— 于是"精度在预算内"这句话没有一个点落在
  模型会用的形状上，缺陷就藏在那个空隙里。
- **`bd=1` 与 `bd=4` 的输出应当逐位相同**（核切分不改算式），这是个不需要参考的硬判据，
  而且只有在 `B*HV` 足够大时才真正压到切分逻辑。不同就是切分 bug，不是精度问题。

### 卡会中途挂掉

其中一台共享主机的 NPU 7 在本次调试中途从 OK 变成 `Critical` / 0.0W，表现为 `TsdOpen failed,
devId=7` + `error code is 507033`。**先看 `npu-smi info` 的 Health 列再怀疑自己的代码。**
共享主机上不要尝试复位别人也在用的卡；换一张 Health=OK 且 `npu-smi info -t proc-mem -i N`
无进程的卡，并把换卡理由写进环境脚本的注释。

### 结论不跨 SoC 继承（两个方向都是）

同级的 `fla_infer` 工作区有 A2/910B3 上的 GDN 精度与 HF32 实测结论；本仓第一、二期的数全部来自 A5。
**一个 SoC 上的数字不能当另一个 SoC 的预期** —— 不论是把 `fla_infer` 的 A2 数搬到本仓的 A2 工作，
还是把本仓的 A5 门控上限、`block_dim` 上限、(C, HV) 边界行为、算子包覆盖搬到 A2/A3。
方法论可以借，阈值和结论必须在目标 SoC 上重测。多 agent 协作里违反这一条要报 `RISK soc-assumption`。

## 6.5 kernel 源码的问题统一修一轮，不零散改

**发现一条就去改一条是错的做法。** 那会在 ascriptor 侧留下一串互相干扰的小改动，
而且每改一次都要重跑全部 case。流程固定成三步：

1. **记账**：写进 `docs/matrix/gaps.json`，打 `requires_kernel_change: true`
   并写 `kernel_change_note`（要改哪个文件的什么）。`gen_matrix.py --check` 会校验
   `summary.kernel_fix_queue` 与这些标记一致 —— 这张表的用处全在"没漏项"上。
2. **本仓侧先保证不静默出错**：按 §7 装闸报错，或记为声明限制。
   闸的边界要**照实测数据逐个钉进测试** —— 静默错误类的缺陷一旦闸被改松，
   没有别的东西会报警。
3. **攒够一批再统一修**（§3：本仓不改 ascriptor 仓，要么走那一侧的流程，
   要么建本仓派生单元）。

动手修之前必须**先定位到确切的原因**，不是症状规律。`c1-multihead-o-corrupt` 现在只掌握
"每个 cube 核的最后一个头是对的"这条规律，还没找出缺的是哪一次 DEvent/Mutex 配对 ——
照规律瞎改可能让 case 变绿而原因没除。

## 7. 门控：不满足就报错

现成算子有硬性定尺限制（`L=64`、`K=V=128`、零初始 state、无 varlen、GDN 无 GQA 分组）。

- 这些限制必须在 `platform.py` / 算子入口**显式声明并在不满足时报错**，
  错误信息要说清哪一条约束没满足、实际值是多少。
- **绝不静默降级到 torch 兜底**，也不要用"近似等价"的路径悄悄替换。
  隐藏缺失能力比缺失能力本身更糟 —— 它让支持矩阵说谎。

**A2 专属·永久禁止（2026-09-23，A2-11 真机定量证实，用户已裁定接受，§2）**：A2 (910B) 系
kernel 里，**BF16 / FP16 的 `splitk` 在 `M < 64` 时必须显式报错，不能出现在任何可达路径里**
——M16 在真机上 500/500 次可靠出错（硬件时序竞争，输出每次不同）、M32 直接触发 AI Core
异常（拖垮进程，不是数值错误）。这不是"暂不支持、以后再补"的缺口，是**结构性硬件缺陷**，
不接受任何绕过（近似、重试、静默改路径）；lint / 静态守卫见
`tests/test_a2_accumulate_barriers.py`。同一份守卫要求 A2 kernel 里每个 `is_init=False`
的累加 `matmul` 前必须紧邻 `barrier(Pipe.M)`（FP32 手写累加链的绕行，M10-081，见 §2）。

## 8. 支持矩阵是单一事实源

`docs/matrix/` 下的 json 是**唯一权威**，markdown 由 `tools/gen_matrix.py` 生成。

- `models.json` — 目标模型的真实形状（带来源 URL 与获取日期）
- `ops.json` — 算子 ABI、定尺约束、各 stage 验证状态
- `gaps.json` — 缺口表，每条带影响面与建议处置

规则：**不要手写 `docs/matrix/*.md`**（手写的矩阵必然腐烂）。每一格的状态都应能
指向一次真实的 `check` / `profile` 运行记录。状态值沿用 ascriptor 的词汇：
`passed` / `untested` / `gap` / `failed`。

## 9. 目录约定

```
ascend_fla/
├── ascend_fla/
│   ├── platform.py      # SoC/CANN 探测、能力门控（不满足即报错）
│   ├── runtime/         # ★ ascriptor → 常驻 torch 可调用算子
│   │   ├── compile.py   #   kernel → CANN custom op .so
│   │   ├── cache.py     #   按 (kernel, 形状签名, SoC, 版本) 缓存
│   │   ├── binding.py   #   torch.library 注册，device tensor 零拷贝
│   │   └── autograd.py  #   fwd/bwd → autograd.Function
│   ├── ops/             # 算子层（目录名对齐 fla.ops 便于对照）
│   ├── modules/         # 窄切片：causal_conv1d / RMSNorm / FusedRMSNormGated
│   ├── layers/          # 窄切片：GatedDeltaNet 等
│   ├── models/          # 注入式：不重写 modeling_*.py，只替换 layer
│   ├── compat/          # 可选：fla 风格签名 wrapper（布局转换）
│   └── reference/       # torch oracle
├── kernels/projects/<soc>/  # 本仓自有的 ascriptor 单元（unit 协议），按 SoC 分目录
├── tests/  benchmarks/
├── docs/handoff.md      # ★ 会话交接：现状 + 下一步 + 已纠正的判断
├── docs/plan.md         # 构建规划
├── docs/matrix/         # ★ 支持矩阵（json 权威，md 生成）
├── docs/pm/             # ★ 多 agent 协作：PROTOCOL.md、board.json（PM 唯一写者）、tasks/<ID>.md
├── docs/research/       # 调研与设计文档（如 GDN ABI 方案），由 PM 合入
├── tools/               # gen_matrix.py、pm_board.py 等
└── tmp/                 # 构建产物、日志、profiling（git-ignored）
```

**窄切片原则**：fla 有 43 个 layers、41 个 models，**一个都不要照搬**。按算子倒推，
用到哪个做哪个。`models/` 用注入而非重写 —— 直接用 HF/fla 的模型定义，只把我们的
layer 换进去，这样规格自动跟上游对齐、零维护成本。

新增目录时同步更新这一节。

## 10. 提交卫生

- 不提交：`machine_specs.md`、`boards.json`、模型权重、数据集、构建产物、
  profiling trace、`tmp/` 下任何东西。规则见 `.gitignore`。
- 提交前 `git status` 确认没有大文件和机器信息混入。
- commit message 中英文皆可，但不要写入主机或账号信息。
