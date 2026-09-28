# ascend_fla

fla 系列线性注意力算子在昇腾 NPU 上的高效实现库。后端用
[ascriptor](https://github.com/ddddwee1/ascriptor)（指令级 Python 编译器），
以 torch 可调用算子对外暴露，目标是比现有方案更快。

这不是 fla 的 fork，也不是它的后端插件 —— 公共 API 由本仓自己定义，
`fla` 只在测试期充当语义权威（`naive.py` 作 CPU fp32 oracle）。详见 `AGENTS.md`。

| 想看什么 | 去哪 |
|---|---|
| 长期工作纪律 | [`AGENTS.md`](AGENTS.md) |
| 此刻进度与已纠正的判断 | [`docs/handoff.md`](docs/handoff.md) |
| 支持矩阵（json 权威，md 生成） | [`docs/matrix/`](docs/matrix/) |
| English | [README.en.md](README.en.md) |
| 想接活 / 提需求 | [`docs/pm/START.md`](docs/pm/START.md)、[申领入口 #28](https://github.com/ddddwee1/ascend_fla_dev/issues/28) |

> 仓库里有两条并行轨道：仓主在 A5 上做 GDN-2（见 `docs/handoff.md` §1），
> 外部 agent 走下面这张表里的任务（A2 → A3 → A5）。两条共用仓库、互不指挥，
> 边界由 `docs/pm/board.json` 的 `reserved_paths` 划定。

## 算子进展

<!-- fla-pm:kernels start -->
<!-- 由 tools/pm_board.py --readme 生成，请勿手改。改 docs/pm/board.json 后重新运行。 -->

| 算子族 | id | 归属 | kernel 数 | 任务进度 |
|---|---|---|---|---|
| KDA（Kimi Delta Attention） | `kda` | 可申领 | 7 | 13/23 |
| GDN（Gated DeltaNet） | `gated_delta_rule` | 可申领 | 4 | 7/10 |
| DeltaNet | `delta_rule` | 尚未排任务 | 2 | 2/2 单元有验证记录 |
| GDN-2（Gated DeltaNet 2） | `gdn2` | 仓主轨道 | 6 | 4/5 |
| 整网融合算子（模块 / 层级） | `fusion` | 可申领 | 4 | 1/4 |
| Mamba-1/2/3 | `mamba` | 未排期 (G4) | — | — |
| GLA（Gated Linear Attention） | `gla` | 未排期 (G4) | — | — |
| PKDA / PGDN（预条件） | `pkda` | 可申领 | 2 | 5/6 |
| Log-Linear Attention | `log_linear` | 未排期 (G4) | — | — |
| DLA（动态线性注意力） | `dla` | 未排期 (G4) | — | — |
| StateX（宽状态） | `statex` | 未排期 (G4) | — | — |
| NAtS-L（动态路由混合） | `natsl` | 未排期 (G4) | — | — |
| NHA（Native Hybrid Attention） | `nha` | 未排期 (G4) | — | — |
| Local Linear Attention | `local_linear` | 未排期 (G4) | — | — |
| DeltaStack（栈式） | `deltastack` | 未排期 (G4) | — | — |
| TTT / MesaNet | `ttt` | 未排期 (G4) | — | — |
| NSA / 稀疏注意力 | `nsa` | 未排期 (G4) | — | — |

### 展开看细节（算子族 → kernel → 任务）

<details><summary><b>KDA（Kimi Delta Attention） —— 7 个 kernel，13/23 完成</b></summary>

_首个目标算子族，Kimi-Linear 用它_

| kernel | 归属 | BF16 | FP32 | 进度 | 下一步 |
|---|---|---|---|---|---|
| `kda_fwd_stable` | 可申领 | ✅ 原生 · A5 真机 | ⛔ 拒绝 | 6/11 | [#30](https://github.com/ddddwee1/ascend_fla_dev/issues/30) A2-04 |
| `kda_bwd_stable` | 可申领 | ✅ 原生 · A5 真机 | — | 3/10 | [#109](https://github.com/ddddwee1/ascend_fla_dev/issues/109) FMT-02 |
| `kda_fused_recurrent` | 可申领 | ✅ 原生 · A5 真机 | ✅ 原生 · A5 真机 | 3/10 | [#105](https://github.com/ddddwee1/ascend_fla_dev/issues/105) BF-06 |
| `kda_fwd` | 上游单元 | ✅ 原生 · A5 真机 | — | 5/9 passed | _上游 ascriptor 单元，本仓用 kda_fwd_stable 取代_ |
| `kda_bwd` | 上游单元 | ✅ 原生 · A5 真机 | — | 4/9 passed，1 项 gap | _上游 ascriptor 单元，本仓用 kda_bwd_stable 取代_ |
| `kda_layout` | 可申领 | ✅ 原生 · A5 真机 | ✅ 原生 · A5 真机 | — | _本仓自写的搬运 / 转换单元（FMT-02，已合入）：token-major ↔ head/chunk-major 布局、BF16 / FP32 转换与缺省 state / dht 零填充，六个自编译 vector kernel（≤4096 元素 tile 的五维 NDDMA）；只在 CANN 9.1.0-beta.1 的 A5 环境验证，前向比原 host 路径慢 1.67–2.32×；存量 host 算术（_scan_states 等）仍在，见 D-PM-42_ |
| `kda_prep` | 可申领 | ✅ 原生 · A5 真机 | ✅ 原生 · A5 真机 | 3/3 | [#106](https://github.com/ddddwee1/ascend_fla_dev/issues/106) BF-07 |

<details><summary>kda_fwd_stable —— 6/11 完成，起点 A2-04、FMT-02、A2-03、A5K-01、A5-04</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| FMT-02 | [#109](https://github.com/ddddwee1/ascend_fla_dev/issues/109) | `a5` | bf16 | ✅ done | ★ 起点 |
| A2-04 | [#30](https://github.com/ddddwee1/ascend_fla_dev/issues/30) | `any` | bf16 | ✅ done | ★ 起点 |
| A2-03 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) | `a2` | bf16、fp32 | ✅ done | ★ 起点 |
| A5-04 | [#50](https://github.com/ddddwee1/ascend_fla_dev/issues/50) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A5K-01 | [#76](https://github.com/ddddwee1/ascend_fla_dev/issues/76) | `a5` | bf16、fp32 | ✅ done | ★ 起点 |
| A5K-02 | [#81](https://github.com/ddddwee1/ascend_fla_dev/issues/81) | `a5` | bf16、fp32 | ✅ done |  |
| A2-12 | [#39](https://github.com/ddddwee1/ascend_fla_dev/issues/39) | `a2` | bf16、fp32 | ✅ done |  |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔵 in_progress |  |
| A2-15 | [#42](https://github.com/ddddwee1/ascend_fla_dev/issues/42) | `a2` | bf16、fp32 | 🔒 gated (用户已放行（D-PM-43），待前置任务与规格) |  |
| A2-16 | [#43](https://github.com/ddddwee1/ascend_fla_dev/issues/43) | `a2` | bf16、fp32 | 🔒 gated (用户已放行（D-PM-43），待前置任务与规格) |  |
| A5-06 | [#52](https://github.com/ddddwee1/ascend_fla_dev/issues/52) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) |  |

</details>

<details><summary>kda_bwd_stable —— 3/10 完成，起点 FMT-02、A2-09、A2-K1、A5-04、A5-05</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| FMT-02 | [#109](https://github.com/ddddwee1/ascend_fla_dev/issues/109) | `a5` | bf16 | ✅ done | ★ 起点 |
| A2-09 | [#112](https://github.com/ddddwee1/ascend_fla_dev/issues/112) | `a2` | bf16、fp32 | ✅ done | ★ 起点 |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔵 in_progress | ★ 起点 |
| A5-04 | [#50](https://github.com/ddddwee1/ascend_fla_dev/issues/50) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A5-05 | [#51](https://github.com/ddddwee1/ascend_fla_dev/issues/51) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A2-13 | [#40](https://github.com/ddddwee1/ascend_fla_dev/issues/40) | `a2` | bf16、fp32 | 🔒 gated (用户已放行（D-PM-43），待前置任务与规格) |  |
| A5K-03 | [#120](https://github.com/ddddwee1/ascend_fla_dev/issues/120) | `a5` | bf16 | ✅ done |  |
| A2-15 | [#42](https://github.com/ddddwee1/ascend_fla_dev/issues/42) | `a2` | bf16、fp32 | 🔒 gated (用户已放行（D-PM-43），待前置任务与规格) |  |
| A2-16 | [#43](https://github.com/ddddwee1/ascend_fla_dev/issues/43) | `a2` | bf16、fp32 | 🔒 gated (用户已放行（D-PM-43），待前置任务与规格) |  |
| A5-06 | [#52](https://github.com/ddddwee1/ascend_fla_dev/issues/52) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) |  |

</details>

<details><summary>kda_fused_recurrent —— 3/10 完成，起点 BF-06、A2-03、A2-K1、A5-01、A5-02、A5-03</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| BF-06 | [#105](https://github.com/ddddwee1/ascend_fla_dev/issues/105) | `a5` | bf16 | ✅ done | ★ 起点 |
| A2-03 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) | `a2` | bf16、fp32 | ✅ done | ★ 起点 |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔵 in_progress | ★ 起点 |
| A5-01 | [#47](https://github.com/ddddwee1/ascend_fla_dev/issues/47) | `a5` | fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A5-02 | [#48](https://github.com/ddddwee1/ascend_fla_dev/issues/48) | `a5` | fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A5-03 | [#49](https://github.com/ddddwee1/ascend_fla_dev/issues/49) | `a5` | fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A2-14 | [#41](https://github.com/ddddwee1/ascend_fla_dev/issues/41) | `a2` | bf16、fp32 | ✅ done |  |
| A2-15 | [#42](https://github.com/ddddwee1/ascend_fla_dev/issues/42) | `a2` | bf16、fp32 | 🔒 gated (用户已放行（D-PM-43），待前置任务与规格) |  |
| A2-16 | [#43](https://github.com/ddddwee1/ascend_fla_dev/issues/43) | `a2` | bf16、fp32 | 🔒 gated (用户已放行（D-PM-43），待前置任务与规格) |  |
| A5-06 | [#52](https://github.com/ddddwee1/ascend_fla_dev/issues/52) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) |  |

</details>

<details><summary>kda_prep —— 3/3 完成，起点 BF-07</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| BF-07 | [#106](https://github.com/ddddwee1/ascend_fla_dev/issues/106) | `a5` | bf16 | ✅ done | ★ 起点 |
| BF-08 | [#117](https://github.com/ddddwee1/ascend_fla_dev/issues/117) | `a5` | bf16 | ✅ done |  |
| BF-09 | [#125](https://github.com/ddddwee1/ascend_fla_dev/issues/125) | `a5` | bf16、fp32 | ✅ done |  |

</details>

</details>

<details><summary><b>GDN（Gated DeltaNet） —— 4 个 kernel，7/10 完成</b></summary>

_Qwen3-Next 用它，六项 ABI 缺口待补（第四期，与本条无关）。**GDN/PGDN 例外线**（D-PM-20/22）：GDA-01（非 GQA 前向）与 GDA-02（GQA/GVA 分组）均已合入，PK-03（PGDN 前向）已解锁，排期继续走 backward → decode → 性能_

| kernel | 归属 | BF16 | FP32 | 进度 | 下一步 |
|---|---|---|---|---|---|
| `gdn_fwd` | 可申领 | ✅ 原生 · A5 真机 | — | 4/7 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) A2-07 |
| `gdn_bwd` | 可申领 | ✅ 原生 · A5 真机 | ✅ 原生 · A5 真机 | 3/6 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) A2-07 |
| `gdn_fused_recurrent` | 可申领 | — | — | 1/2 | [#94](https://github.com/ddddwee1/ascend_fla_dev/issues/94) GDA-04 |
| `gdn_chunk_fwd_a5` | 可申领 | ✅ 原生 · A5 真机 | ✅ 原生 · A5 真机 | — | _窄范围例外（D-PM-20/22）：非 GQA 前向 + GQA/GVA 分组均已合入（GDA-01/02）；BF16 原生 + 分组由 kernel 内 head 索引完成、FP32 公共入口也改走新单元（输出逐字节不变）见 BF-01，已合入_ |

<details><summary>gdn_fwd —— 4/7 完成，起点 A2-07、GDA-01</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-07 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) | `a2` | bf16、fp32 | ✅ done | ★ 起点 |
| GDA-01 | [#73](https://github.com/ddddwee1/ascend_fla_dev/issues/73) | `a5` | bf16、fp32 | ✅ done | ★ 起点 |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔵 in_progress |  |
| GDA-02 | [#82](https://github.com/ddddwee1/ascend_fla_dev/issues/82) | `a5` | bf16、fp32 | ✅ done |  |
| BF-01 | [#100](https://github.com/ddddwee1/ascend_fla_dev/issues/100) | `a5` | bf16 | ✅ done |  |
| A2-20 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |
| GDA-05 | [#96](https://github.com/ddddwee1/ascend_fla_dev/issues/96) | `a5` | bf16、fp32 | ⬜ open |  |

</details>

<details><summary>gdn_bwd —— 3/6 完成，起点 A2-07、GDA-03</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-07 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) | `a2` | bf16、fp32 | ✅ done | ★ 起点 |
| GDA-03 | [#91](https://github.com/ddddwee1/ascend_fla_dev/issues/91) | `a5` | bf16、fp32 | ✅ done | ★ 起点 |
| BF-02 | [#101](https://github.com/ddddwee1/ascend_fla_dev/issues/101) | `a5` | bf16 | ✅ done |  |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔵 in_progress |  |
| GDA-05 | [#96](https://github.com/ddddwee1/ascend_fla_dev/issues/96) | `a5` | bf16、fp32 | ⬜ open |  |
| A2-20 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |

</details>

<details><summary>gdn_fused_recurrent —— 1/2 完成，起点 GDA-04、A2-20</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-20 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) | ★ 起点 |
| GDA-04 | [#94](https://github.com/ddddwee1/ascend_fla_dev/issues/94) | `a5` | bf16、fp32 | ✅ done | ★ 起点 |

</details>

</details>

<details><summary><b>DeltaNet —— 2 个 kernel</b></summary>

_上游单元已存在，本仓尚未排任务_

| kernel | 归属 | BF16 | FP32 | 进度 | 下一步 |
|---|---|---|---|---|---|
| `delta_rule_fwd` | 尚未排任务 | ✅ 原生 · A5 真机 | — | 7/9 passed | _上游单元已存在，本仓尚未排任务_ |
| `delta_rule_bwd` | 尚未排任务 | ✅ 原生 · A5 真机 | — | 7/9 passed | _上游单元已存在，本仓尚未排任务_ |

</details>

<details><summary><b>GDN-2（Gated DeltaNet 2） —— 6 个 kernel，4/5 完成</b></summary>

_仓主并行轨道，见 docs/handoff.md §1。**例外**（D-PM-16，2026-09-17）：chunk 前向单独开放给 agent 轨道，见 GD2-01（前向，已合入）/GD2-02（真机+整网验证，待派）/GD2-03（性能优化，已合入），写集限定新文件、不碰 reserved_paths_

| kernel | 归属 | BF16 | FP32 | 进度 | 下一步 |
|---|---|---|---|---|---|
| `gdn2_fused_recurrent` | 仓主轨道 | ❓ 未核 | ✅ 原生 · A5 真机 | 6/6 passed | _仓主轨道：通用 CCE recurrent_ |
| `gdn2_fused_decode` | 仓主轨道 | ✅ 原生 · A5 真机 | ❓ 未核 | 6/6 passed | _仓主轨道：模型专用融合 decode_ |
| `gdn2_short_conv_decode` | 仓主轨道 | ✅ 原生 · A5 真机 | ❓ 未核 | 6/6 passed | _仓主轨道：打包短卷积 decode_ |
| `gdn2_norm2_w12_swiglu` | 仓主轨道 | ✅ 原生 · A5 真机 | ❓ 未核 | 4/6 passed | _仓主轨道：融合 RMSNorm + SwiGLU_ |
| `gdn2_chunk_fwd` | 可申领 | 🔁 API 加宽 · 仅主机侧 → BF-05 | ✅ 原生 · A5 真机 | 4/5 | [#62](https://github.com/ddddwee1/ascend_fla_dev/issues/62) GD2-01 |
| `gdn2_chunk_fwd_bwd` | 仓主轨道 | ❓ 未核 | ❓ 未核 | — | _反向仍未排期；前向已由 agent 轨道完成，见 gdn2_chunk_fwd_ |

<details><summary>gdn2_chunk_fwd —— 4/5 完成，起点 GD2-01</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| GD2-01 | [#62](https://github.com/ddddwee1/ascend_fla_dev/issues/62) | `a5` | bf16、fp32 | ✅ done | ★ 起点 |
| BF-05 | [#104](https://github.com/ddddwee1/ascend_fla_dev/issues/104) | `a5` | bf16 | ✅ done |  |
| GD2-02 | [#63](https://github.com/ddddwee1/ascend_fla_dev/issues/63) | `a5` | bf16、fp32 | 🔵 in_progress |  |
| GD2-03 | [#66](https://github.com/ddddwee1/ascend_fla_dev/issues/66) | `a5` | bf16、fp32 | ✅ done |  |
| GD2-04 | [#72](https://github.com/ddddwee1/ascend_fla_dev/issues/72) | `a5` | bf16、fp32 | ✅ done |  |

</details>

</details>

<details><summary><b>整网融合算子（模块 / 层级） —— 4 个 kernel，1/4 完成</b></summary>

_算子本身之外，整网跑起来还要的那些：短卷积、门控 RMSNorm、q/k l2norm 与门控变换、合投影。现在全是 torch 原生算子（gaps: modules-are-torch-not-kernels / qk-l2norm-not-in-kernel / decode-layer-overhead）_

| kernel | 归属 | BF16 | FP32 | 进度 | 下一步 |
|---|---|---|---|---|---|
| `causal_conv1d` | 可申领 | — | — | 1/2 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) A2-40 |
| `fused_rms_norm_gated` | 可申领 | — | — | 1/2 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) A2-40 |
| `qk_l2norm_gate` | 可申领 | — | — | 1/2 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) A2-40 |
| `packed_projection` | 可申领 | — | — | 1/2 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) A2-40 |

<details><summary>causal_conv1d —— 1/2 完成，起点 A2-40</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-40 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) | `any` | bf16、fp32 | ✅ done | ★ 起点 |
| A2-41 | [#55](https://github.com/ddddwee1/ascend_fla_dev/issues/55) | `a2` | bf16、fp32 | 🔒 gated (用户已放行（D-PM-43），待前置任务与规格) |  |

</details>

<details><summary>fused_rms_norm_gated —— 1/2 完成，起点 A2-40</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-40 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) | `any` | bf16、fp32 | ✅ done | ★ 起点 |
| A2-41 | [#55](https://github.com/ddddwee1/ascend_fla_dev/issues/55) | `a2` | bf16、fp32 | 🔒 gated (用户已放行（D-PM-43），待前置任务与规格) |  |

</details>

<details><summary>qk_l2norm_gate —— 1/2 完成，起点 A2-40</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-40 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) | `any` | bf16、fp32 | ✅ done | ★ 起点 |
| A2-42 | [#56](https://github.com/ddddwee1/ascend_fla_dev/issues/56) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |

</details>

<details><summary>packed_projection —— 1/2 完成，起点 A2-40</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-40 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) | `any` | bf16、fp32 | ✅ done | ★ 起点 |
| A2-43 | [#57](https://github.com/ddddwee1/ascend_fla_dev/issues/57) | `a2` | bf16、fp32 | 🔒 gated (用户已放行（D-PM-43），待前置任务与规格) |  |

</details>

</details>

<details><summary><b>Mamba-1/2/3 —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>GLA（Gated Linear Attention） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>PKDA / PGDN（预条件） —— 2 个 kernel，5/6 完成</b></summary>

_权威来源已确立（2026-09-17，见 docs/research/pkda_semantics.md）：论文《Preconditioned DeltaNet》(arXiv:2604.21100, ICML 2026)，已合入上游 fla（PR fla-org/flash-linear-attention#950，0.6.0）。PKDA 是 KDA 加一层 ATK预条件，可复用 kda_fwd_stable/kda_bwd_stable，见 PK-02。PGDN 是 GDN 加同一层预条件，但 GDN 自身缺 GQA 分组（gdn-no-gqa），PGDN 排在其后，见 PK-03（gated）。两者都没有已发布的预训练权重，端到端验证到不了真实 logits 一级_

| kernel | 归属 | BF16 | FP32 | 进度 | 下一步 |
|---|---|---|---|---|---|
| `pkda_chunk_fwd` | 可申领 | ✅ 原生 · A5 真机 | ✅ 原生 · A5 真机 | 3/3 | [#68](https://github.com/ddddwee1/ascend_fla_dev/issues/68) PK-02 |
| `pgdn_chunk_fwd` | 可申领 | ✅ 原生 · A5 真机 → BF-03 | ✅ 原生 · A5 真机 | 2/3 | [#69](https://github.com/ddddwee1/ascend_fla_dev/issues/69) PK-03 |

<details><summary>pkda_chunk_fwd —— 3/3 完成，起点 PK-02</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| PK-02 | [#68](https://github.com/ddddwee1/ascend_fla_dev/issues/68) | `a5` | fp32 | ✅ done | ★ 起点 |
| PK-04 | [#92](https://github.com/ddddwee1/ascend_fla_dev/issues/92) | `a5` | fp32 | ✅ done |  |
| BF-04 | [#103](https://github.com/ddddwee1/ascend_fla_dev/issues/103) | `a5` | bf16 | ✅ done |  |

</details>

<details><summary>pgdn_chunk_fwd —— 2/3 完成，起点 PK-03</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| PK-03 | [#69](https://github.com/ddddwee1/ascend_fla_dev/issues/69) | `a5` | bf16、fp32 | ✅ done | ★ 起点 |
| BF-03 | [#102](https://github.com/ddddwee1/ascend_fla_dev/issues/102) | `a5` | bf16 | ✅ done |  |
| GDA-05 | [#96](https://github.com/ddddwee1/ascend_fla_dev/issues/96) | `a5` | bf16、fp32 | ⬜ open |  |

</details>

</details>

<details><summary><b>Log-Linear Attention —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>DLA（动态线性注意力） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>StateX（宽状态） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>NAtS-L（动态路由混合） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>NHA（Native Hybrid Attention） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>Local Linear Attention —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>DeltaStack（栈式） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>TTT / MesaNet —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>NSA / 稀疏注意力 —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>


### 数据类型

| dtype | 归属 | 涉及任务 | 说明 |
|---|---|---|---|
| `bf16` | 可申领 | 46 | 当前 ABI：q/k/v/o 与多数中间量 |
| `fp32` | 可申领 | 44 | 当前 ABI：state、门控累加、精度判定一律 fp32 |
| `fp16` | 未排期 (G3) | — | 契约明确拒绝（见 ops.json 的 no-tail-path 一条） |
| `int8` | 未排期 (G3) | 1 | 未排期；state 累积漂移需先有方案 |
| `mxfp8` | 未排期 (G3) | 1 | 未排期；950 原生解码待核实 |
| `hif8` | 未排期 (G3) | 1 | 未排期 |
| `mxfp4` | 未排期 (G3) | 1 | 未排期；需配 RHT 抑制离群值 |
| `hif4` | 未排期 (G3) | 1 | 未排期；三级微指数解码未核实 |

> 图例：⬜ 可申领 · 🔵 进行中 · 🔒 有前置条件未满足 · ✅ 已完成 · — 还没有任务。
> ★ 起点 = 该 kernel 链里传递依赖全在组外的任务，也就是要让这个 kernel 动起来先做哪一条。
> 任务顺序由依赖关系算出，不是手写的。**仓主轨道**不派给 agent（见 `docs/handoff.md` §1）；
> **未排期 (G4)** 的算子族本仓没有 kernel，按窄切片原则要有目标模型才开工。
> BF16 / FP32 两列 = 该 kernel 的 q/k/v 算子数据类型：✅ 原生 = kernel 以该 dtype 计算且已在对应 SoC 真机验证；
> 🔁 API 加宽 = 公共入口收该 dtype，但在 host 侧加宽成 FP32 再进 FP32 kernel、输出转回，**kernel 本身不跑该 dtype**；**不合规（用户 D-PM-35：BF16 必须 kernel 侧），`→ BF-xx` 是改正它的任务**；
> ⛔ 拒绝 = 公共入口显式报错；🔵 进行中；— 未实现；❓ 未核（仓主轨道 / 上游单元的细节 PM 没核）。
> 「数据类型」表里的「涉及任务」只是任务数，不是进度；某个 kernel 的 BF16 / FP32 进展看各族表里的这两列。
<!-- fla-pm:kernels end -->
