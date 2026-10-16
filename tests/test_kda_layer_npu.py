"""KDA 层的端到端梯度对齐（需要已选定 SoC 的 NPU + CANN）。

第二期的验收项。检查的是**整层**：投影、短卷积、门控变换（``-exp(A_log)*softplus``）、
q/k 的 L2 归一化、beta 的 sigmoid、KDA 算子、``FusedRMSNormGated``、输出投影，以及
这一整条链的反向。

参考怎么造：用**同一份权重**在 CPU 上跑同一个层，只把 KDA 算子换成 fp32 的 CPU 参考
（``reference.kda.kda_recurrent_ref``，逐 token 递推）。于是两边的差异只来自算子实现
与 bf16，层本身的数学是同一份代码 —— 层里写错的公式不会被「参考也写错」抵消掉，因为
参考用的就是它。换言之这个测试守的是**算子在层里接对了**，而算子本身的正确性由
``test_kda_bwd_npu.py`` 对单元 oracle 守。

用递推版而不是向量化分块版，是因为真实层的门控跨度超出了后者的 fp32 安全范围 ——
见 ``test_layer_gate_span_exceeds_declared_domain``。

**梯度对齐测试里把 ``exp(A_log)`` 压到了 1**，为的是把"接线对不对"和"深衰减下 bf16 本来
就糙"分开测，不是因为默认初始化跑不了 —— 默认初始化（跨度 ~94）由
``test_default_init_forward_matches_recurrent_oracle`` 与
``test_deep_gate_backward_matches_cpu_reference`` 专门盯。

预算：参数梯度比单个算子输出更敏感（经过了投影的放大）。算子侧 ``dg`` 的契约预算是
0.25，而 ``A_log`` / ``dt_bias`` 的梯度直接由 ``dg`` 来，所以它们用同一档；其余参数
用 0.1。这些是按契约反推的，不是调到能过为止 —— 超了要去查，不要放宽。

    pytest tests/test_kda_layer_npu.py -v
"""
from __future__ import annotations

import copy
import pathlib
import sys
import math
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

torch_npu = pytest.importorskip("torch_npu", reason="需要 torch_npu")
if not torch.npu.is_available():  # pragma: no cover
    pytest.skip("没有可用的 NPU", allow_module_level=True)

from ascend_fla.layers.kda import KimiDeltaAttention  # noqa: E402
from ascend_fla.ops.kda.chunk import HEAD_DIM  # noqa: E402
from ascend_fla.platform import resolve_soc  # noqa: E402
from ascend_fla.reference.kda import kda_recurrent_ref  # noqa: E402

# dg 的契约预算是 0.25，A_log/dt_bias 的梯度直接由它来
BUDGET_DEFAULT = 0.1
BUDGET_GATE = 0.25
GATE_PARAMS = ("A_log", "dt_bias", "f_proj")


def _budget(name: str) -> float:
    return BUDGET_GATE if any(p in name for p in GATE_PARAMS) else BUDGET_DEFAULT


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _cpu_reference_attn(q, k, v, g, beta, *args, **kwargs):
    """替换 ``chunk_kda`` 的 CPU fp32 参考。签名与它兼容。

    用**逐 token 递推**版而不是向量化分块版。原因：随机初始化的 KDA 层产生的 chunk 内
    门控跨度约 94（见 ``test_layer_gate_span_exceeds_declared_domain``），而向量化版要算
    ``exp(g_max - g)``，跨度超过 80 就在 fp32 上溢。递推版按 per-token 增量做
    ``exp(g_i)``（量级 ~1.5），任何跨度都安全。
    """
    # Independent raw-input adapter; do not call production preprocessing.
    if kwargs.get("use_qk_l2norm_in_kernel", False):
        q = (q.float() / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16()
        k = (k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16()
    if kwargs.get("use_gate_in_kernel", False):
        hv, kd = g.shape[-2:]
        g = -kwargs["A_log"].float().exp().reshape(hv, 1) * torch.nn.functional.softplus(
            g.float() + kwargs["dt_bias"].float().reshape(hv, kd))
    if kwargs.get("use_beta_sigmoid_in_kernel", False):
        beta = beta.float().sigmoid()
    o, state = kda_recurrent_ref(
        q.float(), k.float(), v.float(), g, beta,
        initial_state=kwargs.get("initial_state"),
        output_final_state=kwargs.get("output_final_state", False),
    )
    return o.to(v.dtype), state


@pytest.mark.parametrize("b,t,num_heads,num_v_heads,hidden", [
    pytest.param(1, 64, 1, 1, 256, id="smoke"),
    pytest.param(1, 128, 2, 2, 512, id="multi_chunk"),
    pytest.param(1, 128, 1, 2, 256, id="gva"),
])
def test_layer_grads_match_cpu_reference(b, t, num_heads, num_v_heads, hidden):
    torch.manual_seed(2026)
    kw = dict(hidden_size=hidden, head_dim=128, num_heads=num_heads,
              num_v_heads=num_v_heads, conv_size=4, norm_eps=1e-5)
    layer_cpu = KimiDeltaAttention(**kw).float()
    # ⚠️ 把 exp(A_log) 从 fla 初始化的 ~15 压到 1，让 chunk 内门控跨度落在 ~6 而不是 ~94。
    # **不是为了让测试通过** —— stable 实现在 94 下也能跑（见
    # test_deep_gate_backward_matches_cpu_reference）。这里压低是为了把两件事分开：
    # 本测试验"算子在层里接对了"（17 个参数的梯度逐个对账），深衰减下 bf16 对 log2 累积
    # 门控的舍入会让 dk/dg 的误差涨一个数量级（契约里 dk 0.15 / dg 0.25 就是为它留的），
    # 混在一起测会让"接线错了"和"深衰减本来就糙"分不开。
    with torch.no_grad():
        layer_cpu.A_log.zero_()
    layer_npu = KimiDeltaAttention(**kw).float()
    layer_npu.load_state_dict(layer_cpu.state_dict())
    layer_npu = layer_npu.to("npu")

    x = torch.randn(b, t, hidden) * 0.5
    # 固定的上游梯度，避免 loss=o.sum() 把符号抵消掉
    seed_grad = torch.randn(b, t, hidden, generator=torch.Generator().manual_seed(7))

    x_cpu = x.clone().requires_grad_(True)
    with mock.patch("ascend_fla.layers.kda.chunk_kda", _cpu_reference_attn):
        o_cpu, _ = layer_cpu(x_cpu)
    (o_cpu.float() * seed_grad).sum().backward()

    x_npu = x.clone().to("npu").requires_grad_(True)
    o_npu, _ = layer_npu(x_npu)
    (o_npu.float() * seed_grad.to("npu")).sum().backward()

    fwd_err = _rel_l2(o_npu.detach().cpu(), o_cpu.detach())
    errors = {"[output]": fwd_err, "[dx]": _rel_l2(x_npu.grad.cpu(), x_cpu.grad)}
    for (name, p_cpu), (_, p_npu) in zip(layer_cpu.named_parameters(),
                                         layer_npu.named_parameters()):
        assert p_cpu.grad is not None, f"CPU 侧 {name} 没有梯度，参考路径没走全"
        assert p_npu.grad is not None, f"NPU 侧 {name} 没有梯度，反向链断了"
        errors[name] = _rel_l2(p_npu.grad.cpu(), p_cpu.grad)

    bad = {n: (e, _budget(n)) for n, e in errors.items() if not (e < _budget(n))}
    report = "\n  ".join(f"{n:<24} {e:.3e} / {_budget(n)}" for n, e in errors.items())
    assert not bad, f"超出预算：{bad}\n  全部：\n  {report}"
    print(f"\nB{b} T{t} H{num_heads} HV{num_v_heads} hidden{hidden}\n  {report}")


def test_layer_rejects_unsupported_config():
    """层级门控：不支持的上游开关必须报错，不静默忽略（AGENTS.md §7）。"""
    base = dict(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=1)

    with pytest.raises(ValueError, match="head_k_dim=head_v_dim=128"):
        KimiDeltaAttention(**{**base, "head_dim": 64})
    with pytest.raises(ValueError, match="head_k_dim=head_v_dim=128"):
        KimiDeltaAttention(**{**base, "expand_v": 2.0})
    with pytest.raises(ValueError, match="整数倍"):
        KimiDeltaAttention(**{**base, "num_heads": 2, "num_v_heads": 3})
    with pytest.raises(ValueError, match="allow_neg_eigval"):
        KimiDeltaAttention(**base, allow_neg_eigval=True)
    with pytest.raises(ValueError, match="safe_gate"):
        KimiDeltaAttention(**base, safe_gate=True)
    with pytest.raises(ValueError, match="lower_bound"):
        KimiDeltaAttention(**base, lower_bound=0.5)
    with pytest.raises(ValueError, match="mode"):
        KimiDeltaAttention(**base, mode="recurrent")      # 不存在的 mode
    # 两条路径都该被接受（decode 已接线）
    for m in ("chunk", "fused_recurrent"):
        KimiDeltaAttention(**base, mode=m)

    layer = KimiDeltaAttention(**base).to("npu")
    with pytest.raises(ValueError, match="64 的整数倍"):
        layer(torch.randn(1, 100, 256, device="npu"))     # 既不是 64 的倍数也 >16
    with pytest.raises(ValueError, match="cu_seqlens|varlen"):
        layer(torch.randn(1, 64, 256, device="npu"), cu_seqlens=torch.tensor([0, 64]))


def _calibrate_span(layer, x, target: float) -> float:
    """把层的门控跨度确定性地平移到 ``target``，返回实际值。

    为什么必须校准而不是靠默认抽样：跨度 ∝ ``max_hv exp(A_log)``，而 fla 取
    ``A_log = log(U(1,16))``，所以它是**随机变量** —— 实测同一配置换 seed 能从 15 跳到 96，
    甚至**同一个 seed 下只要建层与抽 x 的先后顺序不同就从 64.6 变成 94.0**（RNG 消耗顺序不同）。
    靠抽样碰深跨度的测试必然时好时坏。``g ∝ exp(A_log)``，所以加 ``log(target/current)`` 即可。
    """
    with torch.no_grad():
        layer.A_log += math.log(target / _gate_span_of(layer, x))
    return _gate_span_of(layer, x)


def test_deep_gate_needs_stable_impl():
    """深门控跨度下 ``upstream`` 必须被挡住报错、``stable`` 必须跑得通。

    跨度校准到 94 —— fla 的初始化确实能产生这个量级（理论上界 ``16 × 0.1 × 63 ≈ 100.8``，
    实测 HV=8 时 12 个 seed 的中位数 93.1、最大 100.9）。实测两套实现的分界：

    * ``upstream``：跨度 ≤66.84 正常（relL2 2.9e-03），≥89.12 全 NaN。根因是 gate 只写
      ``eg = exp(gc)``，它在 ``-ln(FLT_MIN_NORMAL) ≈ 87.3`` 下溢到 0，下游的 ``k/eg`` 与
      ``eg_last/eg`` 变成 ``0×inf`` 与 ``0/0``。
    * ``stable``：跨度到 155.97 仍全部有限，relL2 稳定 2.85e-03~3.19e-03。

    ``upstream`` 要**报错**而不是返回 NaN（AGENTS.md §7）。见 ``docs/matrix/gaps.json`` 的
    ``gate-range-beyond-declared``。
    """
    torch.manual_seed(2026)
    kw = dict(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=1)
    x_cpu = torch.randn(1, 64, 256) * 0.5
    x = x_cpu.to("npu")

    up = KimiDeltaAttention(**kw, impl="upstream").float()
    span = _calibrate_span(up, x_cpu, 94.0)
    assert abs(span - 94.0) < 1.0, f"跨度校准失败：{span:.1f}"
    up = up.to("npu")
    if resolve_soc() == "a2":
        # A2 只声明 stable，加载入口在门控跨度检查之前拒绝 upstream。
        with pytest.raises(ValueError, match="A2 KDA declares only impl='stable'"):
            up(x)
    else:
        with pytest.raises(ValueError, match="门控跨度") as e:
            up(x)
        assert "upstream" in str(e.value), f"报错没指明是哪套实现：{e.value}"

    # 默认就是 stable —— 同样的跨度应当跑得通
    st = KimiDeltaAttention(**kw).float()
    assert st.impl == "stable", "默认实现应当是 stable"
    _calibrate_span(st, x_cpu, 94.0)
    o, _ = st.to("npu")(x)
    assert o.float().isfinite().all(), f"stable 在跨度 {span:.1f} 下仍不是有限值"


def test_default_init_forward_matches_recurrent_oracle():
    """默认初始化下整层前向对逐 token 递推 oracle 的精度。

    这是 stable 实现的意义所在：上一个测试只证明"不是 NaN"，这里要求它**准**。
    预算取 0.05，与算子契约的 max_relative_l2 同档。

    **不固定跨度**：用 fla 原样的初始化，跨度是多少就是多少（会打印出来）。深跨度下的
    精度由 :func:`test_deep_gate_backward_matches_cpu_reference` 校准到 94 专门测。
    """
    torch.manual_seed(2026)
    kw = dict(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=2, conv_size=4)
    layer_cpu = KimiDeltaAttention(**kw).float()
    layer_npu = KimiDeltaAttention(**kw).float()
    layer_npu.load_state_dict(layer_cpu.state_dict())
    layer_npu = layer_npu.to("npu")

    x = torch.randn(1, 128, 256) * 0.5
    with mock.patch("ascend_fla.layers.kda.chunk_kda", _cpu_reference_attn):
        o_cpu, _ = layer_cpu(x)
    o_npu, _ = layer_npu(x.to("npu"))
    err = _rel_l2(o_npu.detach().cpu(), o_cpu.detach())
    print(f"\n默认初始化（本 seed 下跨度 {_gate_span_of(layer_cpu, x):.1f}）"
          f"整层前向相对 L2 = {err:.3e}")
    assert err < 0.05, f"相对 L2 {err:.3e} 超出 0.05"


def test_deep_gate_backward_matches_cpu_reference():
    """门控跨度校准到 94 时**整层反向**对 CPU 参考的精度。

    94 是 fla 初始化能产生的量级（理论上界 ``16 × 0.1 × 63 ≈ 100.8``，实测 HV=8 时 12 个
    seed 的中位数 93.1、最大 100.9），但**不是常量** —— 所以这里确定性地校准，
    见 :func:`_calibrate_span`。

    为什么单独立一个：前向修好不等于能训练。ascriptor 的反向在 ``finalize_pre`` /
    ``finalize_post`` 里把成对衰减分解成 ``exp(g − g_last) · exp(g_last − g)``，前者在
    ``ln(MAX) ≈ 88.72`` 处**上溢**到 inf（输出是 bf16 GM），配对的因子同时下溢到 0，
    下游矩阵乘得 ``inf × 0 = NaN``。方向与前向那处下溢相反，是独立的一处 —— 见
    ``docs/matrix/gaps.json`` 的 ``bwd-gate-range-overflow``。

    所以这个测试盯两件事，顺序不能换：

    1. **有限性**：跨度 ~94 下六类梯度全部有限。这是 ``upstream`` 会直接挂掉的地方。
    2. **精度**：逐参数对 CPU 参考在预算内。预算用深衰减档（``dg`` 的契约预算 0.25），
       因为 bf16 对 log2 累积门控的舍入在深衰减下确实会放大 —— 那是
       ``kda-fwd-bwd-dtype-mismatch`` 已量化过的成因，不是接线错。

    与 ``test_layer_grads_match_cpu_reference`` 的分工：那个把 ``exp(A_log)`` 压到 1（浅跨度）
    只验接线，这个在深跨度验同一条链 —— 深衰减下 bf16 的舍入会让 ``dk``/``dg`` 涨一个数量级，
    混在一起测就分不开"接线错了"和"深衰减本来就糙"。
    """
    torch.manual_seed(2026)
    kw = dict(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=2, conv_size=4)
    layer_cpu = KimiDeltaAttention(**kw).float()
    layer_npu = KimiDeltaAttention(**kw).float()
    layer_npu.load_state_dict(layer_cpu.state_dict())
    layer_npu = layer_npu.to("npu")

    x = torch.randn(1, 128, 256) * 0.5
    # 默认初始化的跨度是**随机变量**：它 ∝ max_hv exp(A_log)，而 fla 取 A_log = log(U(1,16))，
    # 所以同一配置换个 seed 能从 15 跳到 96（实测 12 个 seed：HV=1 给 21.5~96.2、
    # HV=8 给 63.1~100.9）。靠抽样碰一个深跨度会让这个测试时好时坏，所以**确定性地**
    # 把 A_log 整体平移到目标跨度 —— g ∝ exp(A_log)，加 log(target/current) 即可。
    # 目标取 94（实测过的深档，也在反向闸 100 以内）。
    target = 94.0
    with torch.no_grad():
        shift = math.log(target / _gate_span_of(layer_cpu, x))
        layer_cpu.A_log += shift
        layer_npu.A_log += shift
    span = _gate_span_of(layer_cpu, x)
    assert abs(span - target) < 1.0, f"跨度校准失败：想要 {target}，得到 {span:.1f}"

    seed_grad = torch.randn(1, 128, 256, generator=torch.Generator().manual_seed(7))

    x_cpu = x.clone().requires_grad_(True)
    with mock.patch("ascend_fla.layers.kda.chunk_kda", _cpu_reference_attn):
        o_cpu, _ = layer_cpu(x_cpu)
    (o_cpu.float() * seed_grad).sum().backward()

    x_npu = x.clone().to("npu").requires_grad_(True)
    o_npu, _ = layer_npu(x_npu)
    (o_npu.float() * seed_grad.to("npu")).sum().backward()

    # ① 有限性先判 —— NaN 下比精度没有意义，而且 rel_l2 会给出 nan 让断言信息难读
    nonfinite = {"[output]": o_npu, "[dx]": x_npu.grad}
    nonfinite |= {n: p.grad for n, p in layer_npu.named_parameters()}
    broken = [n for n, t in nonfinite.items()
              if t is None or not t.detach().cpu().float().isfinite().all()]
    assert not broken, (
        f"跨度 {span:.1f} 下这些量不是有限值：{broken}。"
        f"若 impl 是 upstream 这是预期的（bwd-gate-range-overflow）；"
        f"若 impl 是 stable 说明 kda_bwd_stable 没生效或没修对"
    )

    # ② 再判精度
    errors = {"[output]": _rel_l2(o_npu.detach().cpu(), o_cpu.detach()),
              "[dx]": _rel_l2(x_npu.grad.cpu(), x_cpu.grad)}
    for (name, p_cpu), (_, p_npu) in zip(layer_cpu.named_parameters(),
                                         layer_npu.named_parameters()):
        errors[name] = _rel_l2(p_npu.grad.cpu(), p_cpu.grad)
    bad = {n: (e, BUDGET_GATE) for n, e in errors.items() if not (e < BUDGET_GATE)}
    report = "\n  ".join(f"{n:<24} {e:.3e}" for n, e in errors.items())
    assert not bad, f"超出预算 {BUDGET_GATE}：{bad}\n  全部：\n  {report}"
    print(f"\n默认初始化（跨度 {span:.1f}）下整层反向，预算 {BUDGET_GATE}：\n  {report}")


def _gate_span_of(layer, x) -> float:
    """层在这份输入下产生的 chunk 内门控跨度。与算子的门控检查同一个定义。"""
    b, t = x.shape[0], x.shape[1]
    g = layer._gate(x, b, t)                      # _gate(hidden_states, b, t) -> [B,T,HV,KD]
    cum = g.view(b, t // 64, 64, layer.num_v_heads, layer.head_k_dim).cumsum(2)
    return (cum.amax(dim=2) - cum.amin(dim=2)).max().item()


def test_layer_gate_span_is_a_random_variable_with_a_ceiling():
    """把"层产生的门控跨度是随机变量、且有上界"钉成测试 —— 纯 CPU。

    两件此前被我写错的事：

    1. **它不是常量 ~94。** 跨度 ∝ ``max_hv exp(A_log)``，而 fla 取 ``A_log = log(U(1,16))``，
       所以 ``exp(A_log) ∈ [1,16]``，跨度能差 16 倍。实测 12 个 seed：HV=1 给 21.5~96.2、
       HV=2 给 15.3~94.8、HV=8 给 63.1~100.9。**头数越多，最大值越稳地顶到上界**
       （取 max 的样本更多）。甚至同一个 seed 下，建层与抽 ``x`` 的先后顺序不同就能从 64.6
       变成 94.0 —— RNG 消耗顺序不同。
    2. **它有上界**：``exp(A_log) ≤ 16``、``dt ≤ 0.1``、63 步 → ``16 × 0.1 × 63 ≈ 100.8``。
       这个数**恰好顶在反向闸 100 上** —— 即 fla 默认初始化的最坏情形会被我们的门控挡住。
       记在 ``docs/matrix/gaps.json`` 的 ``gate-span-still-bounded``。

    对比：``kda_fwd`` contract 的 ``input_generation`` 是 ``g_raw ∈ [-0.03, 0]``，即 64 token
    跨度 ≤1.89 —— 算子被验证过的域比真实层窄 50 倍。
    """
    kw = dict(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=8, conv_size=4)
    spans = []
    for seed in range(8):
        torch.manual_seed(seed)
        layer = KimiDeltaAttention(**kw).float()
        torch.manual_seed(1000 + seed)
        spans.append(_gate_span_of(layer, torch.randn(1, 128, kw["hidden_size"]) * 0.5))
    ceiling = 16.0 * 0.1 * 63
    print(f"\nHV=8 下 8 个 seed 的 chunk 内门控跨度："
          + " / ".join(f"{v:.1f}" for v in sorted(spans))
          + f"   （fla 初始化的理论上界 {ceiling:.1f}）")
    assert max(spans) / min(spans) > 1.2, (
        f"8 个 seed 的跨度几乎一样（{spans}），与「它是随机变量」矛盾 —— 初始化被改过？"
    )
    assert max(spans) <= ceiling * 1.05, (
        f"最大跨度 {max(spans):.1f} 超过理论上界 {ceiling:.1f}，说明上界推导错了"
    )
    # 输入尺度几乎不影响它 —— 这条是原先那个测试的有效部分，保留
    torch.manual_seed(2026)
    layer = KimiDeltaAttention(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=1).float()
    by_scale = [_gate_span_of(layer, torch.randn(1, 64, 256) * sc) for sc in (0.5, 0.05, 0.01)]
    print(f"输入 scale 0.5/0.05/0.01 下的跨度："
          + " / ".join(f"{v:.1f}" for v in by_scale) + "   （应当几乎相同）")
    assert max(by_scale) / min(by_scale) < 1.2, (
        f"跨度随输入尺度明显变化（{by_scale}），与「由初始化决定」的结论矛盾"
    )


# ------------------------------------------------------- prefill → decode 一致性
def _layer_and_input(b=1, t=64, num_heads=2, num_v_heads=2, hidden=256, seed=11):
    torch.manual_seed(seed)
    layer = KimiDeltaAttention(
        hidden_size=hidden, num_heads=num_heads, num_v_heads=num_v_heads,
        head_dim=HEAD_DIM, layer_idx=0, block_dim=1,
    ).to("npu").eval()
    x = torch.randn(b, t, hidden, generator=torch.Generator().manual_seed(seed + 1))
    return layer, x


@torch.no_grad()
def test_prefill_then_decode_matches_one_shot_reference():
    """prefill 走 chunk + cache，再逐 token 走 fused_recurrent —— 对整段 CPU 参考。

    **这是 decode 接线的验收判据**（AGENTS.md §6：chunk 与 recurrent 互为最好的 oracle，
    prefill/decode 一致性同理）。它同时盯三件事：① 两条路径数学一致；
    ② ``cache`` 把 recurrent_state 与三份 conv_state 都正确交接了 ——
    **漏掉 conv_state 的表现是只有前几个 decode token 错**，整段平均误差会掩盖它，
    所以下面逐 token 报数；③ 层里 l2norm / 门控 / o_norm 在 T=1 上与 T=64 行为一致。
    """
    from ascend_fla.layers.kda import RECURRENT_T_MAX
    from ascend_fla.ops.kda import prepare

    # **必须先 prepare(decode=True)**：CANN 只在首次算子解析时读 ASCEND_CUSTOM_OPP_PATH，
    # 所以 decode 的 kernel 必须在 chunk 第一次执行**之前**编译好。不这么做会报
    # "已经执行过 aclnn 算子" —— 实测踩过，而且任何"prefill 完再解码"的进程都会遇到。
    prepare(decode=True)

    # PREFILL 取 128 而不是 64：64 个 token 就是 C=1，而 C=1 且 B*HV > block_dim 时
    # 上游 kda_sub45_fused_kernel 的 o 是错的（c1-multihead-o-corrupt），会被门控拦下。
    # **这就是那个 P0 在真实用法里的后果：64 token 粒度的 prefill 用不了。**
    PREFILL, DECODE = 128, 5
    assert DECODE <= RECURRENT_T_MAX
    layer, x_all = _layer_and_input(t=PREFILL + DECODE)
    xd = x_all.to("npu")

    cache: dict = {}
    o_pre, _ = layer(xd[:, :PREFILL], cache=cache, mode="chunk")
    assert set(cache) == {"recurrent_state", "conv_state"}, f"cache 的键不对：{sorted(cache)}"
    assert cache["recurrent_state"] is not None
    assert all(c is not None for c in cache["conv_state"]), "conv_state 没被写回"

    steps = []
    for i in range(DECODE):
        oi, _ = layer(xd[:, PREFILL + i:PREFILL + i + 1], cache=cache,
                      mode="fused_recurrent")
        steps.append(oi)
    o_dec = torch.cat(steps, dim=1)

    # 整段参考：同一份权重的 CPU 层，KDA 算子换成逐 token 递推
    # 参考要一次过 133 个 token，而层里 T%64 那个闸是**算子**的约束（无 tail 路径），
    # CPU 参考没有这个约束 —— chunk_kda 已经被换成逐 token 递推了。所以把 L_PER_CHUNK
    # 临时打成 1 只是绕开那个闸，不改任何算术。
    ref = copy.deepcopy(layer).cpu().float()
    with mock.patch("ascend_fla.layers.kda.chunk_kda", _cpu_reference_attn), \
            mock.patch("ascend_fla.layers.kda.L_PER_CHUNK", 1):
        o_ref, _ = ref(x_all.float(), mode="chunk")

    o_pre, o_dec = o_pre.cpu(), o_dec.cpu()      # 先 D2H 再比，_rel_l2 不跨设备
    e_pre = _rel_l2(o_pre, o_ref[:, :PREFILL])
    print(f"\nprefill（chunk，{PREFILL} token）相对 L2 = {e_pre:.3e}")
    per_step = [_rel_l2(o_dec[:, i:i + 1], o_ref[:, PREFILL + i:PREFILL + i + 1])
                for i in range(DECODE)]
    print("decode 逐 token 相对 L2：" + "  ".join(f"t{PREFILL + i}={e:.3e}"
                                                for i, e in enumerate(per_step)))
    budget = _budget("output")
    assert e_pre < budget, f"prefill 段就不对：{e_pre:.3e} ≥ {budget}"
    # 逐 token 判，而不是判平均 —— conv_state 漏传只会坏前 conv_size-1 个 token
    bad = [(i, e) for i, e in enumerate(per_step) if not e < budget]
    assert not bad, f"decode 这些步超预算（预算 {budget}）：{bad}"


@torch.no_grad()
def test_layer_refuses_to_switch_paths_silently():
    """请求的路径服务不了输入时必须报错并指出另一条，不悄悄换（AGENTS.md §7）。"""
    from ascend_fla.layers.kda import RECURRENT_T_MAX

    layer, x = _layer_and_input(t=64)
    xd = x.to("npu")
    with pytest.raises(ValueError, match="fused_recurrent"):
        layer(xd[:, :1], mode="chunk")                      # T=1 不是 64 的倍数
    with pytest.raises(ValueError, match="chunk"):
        layer(xd, mode="fused_recurrent")                   # T=64 > T_MAX
    assert RECURRENT_T_MAX < 64


def test_decode_path_refuses_to_run_under_grad():
    """decode 没有反向 kernel。不拦的话不报错、只是不建图 —— 梯度静默消失。"""
    layer, x = _layer_and_input(t=64)
    with pytest.raises(ValueError, match="不可求导"):
        layer(x[:, :1].to("npu"), mode="fused_recurrent")   # 默认 grad 开着，参数 requires_grad
