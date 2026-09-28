# GDA-04 GDN decode（fused recurrent）：从零建 kernel + A5 真机验收

- 波次 / SoC：W0 / a5 · 优先级：`-`（D-PM-22 排期任务，不靠优先级）· 时限：24h（真机部分若因机器/卡
  受阻，assignee 在 STATUS 里如实说明，PM 据此调整，不当超时）· 依赖：`PK-05`（已完成）
- 需要：A5 真机、ascriptor workspace（`AGENTS.md` §3 的 gitcode 来源，按 `agent/compatibility.json` 的 pin）
- 写集：`kernels/projects/a5/gdn_fused_recurrent/**`、`ascend_fla/ops/gdn_fused_recurrent.py`、
  `tests/test_gdn_fused_recurrent.py`、`docs/research/gdn_fused_recurrent_gate_range.md`
- **2026-09-28 PM 补全并放行**：原骨架依赖 PK-05，已完成；本规格是 PM 照 GDA-03（反向）与 A2-14
  （decode 资格化）的先例写就，放行后不再另问用户批准这个任务本身。

## 这条任务是怎么来的，和 A2-14/GDA-03 的关键区别

D-PM-22 排期里 decode 阶段的第二个切片（GDN，接在 GDA-03 反向之后）。来自 #89（外部需求，分诊后
采纳，用户批准）。`docs/matrix/ops.json` 的缺口表原文写"随 GDN 扩族（第四期）再补"，`ascriptor_asset:
none`——**这条和 A2-14 不一样：A2-14 是资格化一个已经存在的 A2 KDA decode 单元，这条是从零建 kernel**，
`kernels/projects/a5/gdn_fused_recurrent/` 目前不存在，也没有可参照移植的上游 ascriptor 资产（同
`gdn_fused_recurrent` 在缺口表里 `ascriptor_asset: "none"` 一致）。工作量、风险与 GDA-01（GDN chunk
前向，同样是从零建）更接近，不要按 A2-14 那种"资格化"的轻量预期来做。

`fla` 对应实现是 `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`（语义权威）。

## 现有 GDN chunk 前向 ABI（只读参照，不是要复用的代码）

`ascend_fla/ops/gdn_chunk_fwd.py` 的 `chunk_gdn`（PM 已读源码核实，行号以当前 main 为准）：

- `q,k`: `[B,T,H,128]`；`v`: `[B,T,HV,128]`；`g,beta`: `[B,T,HV]`（**标量门控/更新率，不是 KDA 那种
  按 128 维展开的逐通道门控**——GDN 与 KDA 的 ABI 差别就在这里，不要照抄 KDA decode 的形状）。
- `q/k/v` 同为 FP32 或 BF16；`g/beta` 恒 FP32。
- `scale` 固定 `128**-0.5`，**不可配置**（`_validate` 对非此值直接报错）。
- `T` 必须是 64 的倍数、`≤4096`；`HV % H == 0`（连续分组，同 KDA/GDA-02 的约定）。
- **`initial_state` 只接受 `None`（零初始态），传张量直接报错**——chunk 前向现在不支持非零初始 state。
  这意味着"chunk→decode 续接"的验证只能是：chunk 从零开始跑一段前缀（`initial_state=None`）产出
  `final_state`，再把这个 `final_state` 喂给**本任务的 decode**继续递推；不能反过来验证 decode 的
  state 喂回 chunk（chunk 侧目前没有这个入口）。
- `final_state`（chunk 契约 `kernels/projects/a5/gdn_chunk_fwd/contract.json`）：`[B, HV, 128, 128]`
  FP32，K 在前——本任务的 `initial_state`/输出 state 必须用同一个布局，否则 prefill→decode 接不上。
- `head_first` 不支持，`launcher` 只认 `inprocess`/`aclnn`/`board`。

## 范围

1. **域声明与 ABI 冻结**（先做这步，再动手，`AGENTS.md` §7）：
   - 公共入口 `fused_recurrent_gdn(q, k, v, g, beta, *, initial_state=None, output_final_state=False,
     scale=SCALE, device='a5', block_dim=?, ...)`，形状/dtype 逐项对齐上面的 chunk ABI 表（S 换成
     短步数，不要求 64 的倍数——decode 场景 S 很小，允许 `S=1..S_MAX`，`S_MAX` 由 kernel 内部缓冲行数
     决定，参照 KDA decode 的 `T_MAX=16` 先例，具体数字由实现定并写进 ABI 冻结表，不要凭空假设）。
   - `initial_state`：**必须支持非零**，`[B,HV,128,128]` FP32，K 在前，与 chunk 的 `final_state`
     布局一致；`None` 表示零初始态。`output_final_state=True` 时返回更新后的同形状 state（新张量，
     不原地改 `initial_state`）。
   - `block_dim` 的合法取值**由实测决定，不假设等于 chunk 的 `{1,2}`**（同 KDA 的先例：decode 走纯
     向量核，物理核数上限与 chunk 不同，A5 KDA decode 实测到 28）。
   - **BF16 不需要显式拒绝**——`q/k/v` 原生支持 BF16 或 FP32（同 chunk），dtype 转换全部在 kernel 内
     完成（D-PM-35/37：host 侧只做校验/分配/launch，不做 `.float()`/`.to(dtype)`）。`g/beta` 恒 FP32。
   - 归一化：同 GDA-01/03，GDN 的 naive 与本仓前向都不对 `q/k` 归一化，decode 同样不做，不支持
     `use_qk_l2norm_in_kernel` 这类可选项，需要就显式拒绝。
2. **双 oracle**：**A** = fla 的 `fused_recurrent_gated_delta_rule` CPU FP32 自动实现（语义权威，
   pin 的确切版本写进 ABI 冻结表）；**B** = 本任务独立写的 CPU FP32 逐 token 递推（不复用 A 的实现，
   不 import A）。两者先在小形状上过一次 FP64 数值 gradcheck 或至少 FP64 参考互相对照，证明 A、B 自己
   是对的，再往下走（同 GDA-03 口径 1）。**吸取 A2-14 的教训**：如果打算像 A2-14 那样另写一个"显式
   FP32 逐元素算术"的独立参考当主金标准，就把这条明确写进冻结表，不要事后被指出"实现细节和声称的不
   一致"才补。
3. **prefill→decode 一致性**（`AGENTS.md` §6 硬判据，A2-14 已验证过同类模式，可以照抄测试结构）：
   - 用 `chunk_gdn` 跑一段前缀（长度取 64 的倍数，例如 128），`initial_state=None`，拿到 `final_state`；
   - 把这个 `final_state` 喂给 `fused_recurrent_gdn` 继续单步/短步递推若干步；
   - 与"整段一次性用 `chunk_gdn` 算完"（前缀+续接总长仍需 64 的倍数，例如凑到 192）的结果比对，
     `o`/`final_state` 相对 L2 报数，**逐 token 报，不只报平均**（AGENTS.md §6 的教训：state 漏传只
     坏前几个 token，平均会掩盖）；
   - 另外验证"decode 分段调用 + 串接 state"与"decode 一次调用相同总步数"两种方式逐位或近似一致
     （同 A2-14 的分段 T16 vs 一次性对照）。
4. **bd / 分组网格**：
   - `block_dim` 网格扫描，确定安全集合，`bd` 之间输出必须逐位相同（`B*HV` 足够大才能真正压到切分
     逻辑，参照 §6 铁律，用真实 Kimi 形状或至少 `HV≥4` 的形状测，不要只测 `HV=1`）。
   - `HV/H` 分组比至少覆盖 `{1,2,4,8}`（同 GDA-02/03 的口径），含非对称 `HV≠H`。
   - 数值量程：`g≤0` 只保证 `exp(g)≤1`，不保证递推稳定（更新矩阵含 `I-beta*k*k^T`），把这条列进量程
     复核表，不要想当然假设"g 恒 ≤0 就万事大吉"。
5. **精度预算**：FP32 判定，`o`/`final_state` 对 A、B 各自相对 L2 ≤ 1e-4（先在 CPU 上校准 A↔B
   一致性，若二者自己就接近或超过这个数量级，先报 PM 调整，不能事后放宽，同 GDA-03 口径 3）。
   BF16 只作质量/存储边界检查（不设 FP32 级别的通过阈值），不放宽 FP32 预算。
6. **同卡测量**：三轮 baseline/candidate/baseline（不设速度门槛，如实标基线身份，同 GD2-02 最近
   踩的坑——basline 必须是任务真正要比的那个东西，不是抄别的任务的口径）。
7. **入口门控**：范围外组合（不支持的 `block_dim`、形状不匹配、`initial_state` 形状/dtype 不对、
   `head_first`、任何未声明的可选项）在 `ascend_fla/ops/gdn_fused_recurrent.py` 显式报错，测试覆盖
   每一条拒绝路径。

## 已知陷阱

- **不要把 KDA decode 的 ABI 直接套过来**：KDA 的 `g` 是按 128 维展开的逐通道门控，GDN 的 `g/beta`
  是标量；形状不一样，语义也不一样。
- **`chunk_gdn` 不支持非零 `initial_state`**——decode 的验证链路只能是"chunk 从零起→decode 续接"，
  不能反过来。
- `final_state`/`initial_state` 的 K-V 轴顺序：K 在前（`[B,HV,128,128]`），与 chunk 契约一致；
  接线时如果发现和这个不一致，先发 RISK 不要自己悄悄改布局。
- 机器与卡归 assignee 自己管理（D-PM-28）；真机证据一行不能少：SoC/CANN/算子包/原始日志/逐项数字。
- **写集扩展走标准流程**：证据/报告类路径（如需要独立的 `docs/research/` 或 `benchmarks/a5/evidence/`
  子目录）不在当前写集里，需要就先发 `RISK write-set-expansion`，PM 查完冲突批准后再用——A2-14 刚
  踩过两次"临时才发现要改写集外文件"的坑，能提前想到的就提前在 ABI 冻结阶段一并申请。

## 验证命令

```bash
PYTHONPATH=<ascriptor>/library ascriptor check kernels/projects/a5/gdn_fused_recurrent/kernels/<file>.py::<fn>
PYTHONPATH=<ascriptor>/library python kernels/projects/a5/gdn_fused_recurrent/run.py reference
PYTHONPATH=<ascriptor>/library python kernels/projects/a5/gdn_fused_recurrent/run.py check --launcher sim
pytest tests/test_gdn_fused_recurrent.py -q
```

## 验收

- [ ] `docs/research/gdn_fused_recurrent_gate_range.md`：ABI 冻结表（含 `S_MAX`、`block_dim` 合法值、
      state 布局）、A↔B 一致性数字、量程复核表。
- [ ] `ascriptor check` 0 error；`run.py reference` 全过；sim/pipesim 在缩小形状边界点上过。
- [ ] **真机**：完整选定 workload（真实 Kimi 或至少 `HV≥4` 的形状）先跑；`HV/H∈{1,2,4,8}`；FP32 相对
      L2 ≤ 1e-4（对 A、B 各一组）；`bd` 之间输出逐位相同；输入未被修改。
- [ ] prefill→decode 一致性：chunk 前缀 + decode 续接 vs 整段 chunk，逐 token 报数；decode 分段调用 vs
      一次性调用一致。
- [ ] 同卡三轮三明治，如实标注 baseline 身份与原始样本。
- [ ] 入口拒绝范围外组合，测试覆盖每一条。
- [ ] `git diff --stat` 对写集之外的路径为空，不改 `gdn_chunk_fwd/**`、`gdn_chunk_bwd/**`。
- [ ] **没有声称**：CUDA/Triton、任何没跑过的真机结果、把 A2/其它 SoC 的结论当本任务结论。
