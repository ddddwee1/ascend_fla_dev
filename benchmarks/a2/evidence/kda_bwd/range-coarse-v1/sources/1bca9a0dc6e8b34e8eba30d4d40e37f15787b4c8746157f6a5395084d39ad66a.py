"""宽门控跨度下的反向精度（需要 A5 NPU + CANN）。

**为什么单独一个文件。** `test_kda_bwd_npu.py` 对的是 `a5.kda_bwd` 单元自己的
`ref.oracle`，而那个 oracle 在宽域下**自己就溢出**：实测跨度 93.84 时它的 `dq` 有 3286 个、
`dk` 6507 个、`dg` 16320 个非有限值（它按 `exp2(g_i − g_j)` 算整个成对矩阵再掩码，
非因果那一半先变 inf，掩码后成 NaN）。所以宽域下它不能当参考。

这里换成 **fp32 逐 token 递推参考 + autograd**（`reference.kda.kda_recurrent_ref`）。它只用
per-token 的 `exp(g_i)`（量级 ~1.5），任何跨度都安全 —— 与前向那边用它当宽域 oracle 同理。

**实验设计**：同一份输入扫几个跨度，其中**必须包含契约 case 所在的 ~46 档**。那一档起
标定作用：如果它给出的相对 L2 与 `test_kda_bwd_npu.py` 对单元 oracle 测到的同量级
（`dk` ~9e-02、`dg` ~1.6e-01），就说明两个 oracle 在窄域一致，于是 ~94 档的数可以直接和
契约预算比。没有这一档，宽域的数就无从判断是"算子变糟"还是"换了参考"。

预算用契约的 `comparison`（默认 0.05 / `dk` 0.15 / `dg` 0.25），**不为了让它过而放宽**。

    pytest tests/test_kda_bwd_deep_npu.py -v -s
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

torch_npu = pytest.importorskip("torch_npu", reason="需要 torch_npu")
if not torch.npu.is_available():  # pragma: no cover
    pytest.skip("没有可用的 NPU", allow_module_level=True)

from ascend_fla.ops.kda import chunk_kda_fwd_with_caches  # noqa: E402
from ascend_fla.ops.kda.chunk import (  # noqa: E402
    L_PER_CHUNK,
    MAX_GATE_SPAN,
    _gate_span,
)
from ascend_fla.ops.kda.chunk_bwd import chunk_kda_bwd  # noqa: E402
from ascend_fla.reference.kda import kda_recurrent_ref  # noqa: E402

B, T, H, HV, D = 1, 128, 1, 1, 128          # C=2 —— 单 chunk 测不到 chunk 间的状态传递
BUDGET = {"dq": 0.05, "dk": 0.15, "dv": 0.05, "dbeta": 0.05, "dg": 0.25, "dh0": 0.05}

#: 目标跨度。46 是 a5.kda_bwd 契约 case 所在的档（标定用），94 是 fla 默认初始化常见的档
#: （它是随机变量，上界 ≈100.8 —— 见 test_kda_layer_npu.py 的
#: ``test_layer_gate_span_is_a_random_variable_with_a_ceiling``）。
#: 上限取 :data:`~ascend_fla.ops.kda.chunk.MAX_GATE_SPAN` 的 ``stable`` 值 —— 门控声明为
#: 安全的跨度，就必须在契约预算内，否则门控等于在默许静默降级（AGENTS.md §7）。
#: 更深的档精度会继续退化（到 169.76 都还是**有限值**，但 ``dq`` 在 130 处越过预算 0.05），
#: 曲线记在 ``kda_bwd_stable/contract.json`` 的 ``domain.gate_span.accuracy_vs_span``；
#: 要用更深的跨度就得显式 ``check_gate_range=False``，那是调用方自己的决定。
#: 最深档取闸减 1 —— 正好等于闸时，按比例缩放出的跨度可能是 105.0000x 而被门控拒掉，
#: 那测的就是浮点噪声而不是精度。
TARGET_SPANS = (46.0, 94.0, MAX_GATE_SPAN["stable"]["backward"] - 1.0)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _inputs(span: float, seed: int = 2026):
    """造一份 chunk 内跨度约为 ``span`` 的输入。全在 CPU 上造 —— 缺内置算子包的机器也能跑。"""
    gen = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, generator=gen), dim=-1)
    g_unit = -torch.rand(B, T, HV, D, generator=gen) * 0.03
    # 单位 g 的跨度 ≈ 0.03 * 63 / 2（均匀分布的期望），直接按比例缩放到目标
    scale = span / _gate_span(g_unit.float(), T // 64, on_cpu=True)
    return dict(
        q=q.bfloat16(), k=k.bfloat16(),
        v=(torch.randn(B, T, HV, D, generator=gen) * 0.04).bfloat16(),
        beta=torch.rand(B, T, HV, generator=gen) * 0.45 + 0.05,
        g=(g_unit * scale).float(),
        h0=(torch.randn(B, HV, D, D, generator=gen) * 0.01),
        do=(torch.randn(B, T, HV, D, generator=gen) * 0.04).bfloat16(),
        dht=(torch.randn(B, HV, D, D, generator=gen) * 0.01).bfloat16(),
    )


def _through_bf16_cumsum(g: torch.Tensor) -> torch.Tensor:
    """per-token 增量 → chunk 内 cumsum → bf16 → 回 fp32 → 差分回增量。

    这正是反向 kernel 能看到的门控信息：``kda_bwd`` 的 ABI 把 ``g_cumsum`` 定为 **bf16**
    （前向入口的 g 是 fp32）。所以喂了这个 g 的 fp32 参考，才是**任何实现都只能做到**的目标。
    """
    b, t, hv, d = g.shape
    c = t // L_PER_CHUNK
    cum = g.view(b, c, L_PER_CHUNK, hv, d).cumsum(2).bfloat16().float()
    prev = torch.cat([torch.zeros_like(cum[:, :, :1]), cum[:, :, :-1]], dim=2)
    return (cum - prev).view(b, t, hv, d)


def _reference_grads(x: dict, g: torch.Tensor | None = None) -> dict:
    """fp32 递推参考的梯度。用 ``autograd.grad`` 而不是 ``.backward()``，避免污染叶子。

    ``g`` 不传时用 ``x["g"]``（fp32，比反向 ABI 更准）；传 :func:`_through_bf16_cumsum`
    的结果则得到**可达目标**。
    """
    leaves = {n: x[n].clone().float().requires_grad_(True)
              for n in ("q", "k", "v", "beta", "h0")}
    gl = (x["g"] if g is None else g).clone().float().requires_grad_(True)
    o, ht = kda_recurrent_ref(
        leaves["q"], leaves["k"], leaves["v"], gl, leaves["beta"],
        initial_state=leaves["h0"], output_final_state=True,
    )
    # 与喂给 kernel 的上游梯度完全一致：o 配 do、final_state 配 dht
    loss = (o * x["do"].float()).sum() + (ht * x["dht"].float()).sum()
    grads = torch.autograd.grad(
        loss, [leaves["q"], leaves["k"], leaves["v"], leaves["beta"], gl, leaves["h0"]])
    return dict(zip(("dq", "dk", "dv", "dbeta", "dg", "dh0"), grads))


@pytest.mark.parametrize("span", TARGET_SPANS)
def test_bwd_matches_recurrent_reference(span):
    x = _inputs(span)
    got_span = _gate_span(x["g"], T // 64, on_cpu=True)
    assert abs(got_span - span) < 1.0, f"造出来的跨度 {got_span:.2f} 与目标 {span} 不符"

    npu = lambda t: t.to("npu")  # noqa: E731
    _, _, caches = chunk_kda_fwd_with_caches(
        npu(x["q"]), npu(x["k"]), npu(x["v"]), npu(x["g"]), npu(x["beta"]),
        None, npu(x["h0"]),
    )
    got = chunk_kda_bwd(
        q=npu(x["q"]), k=npu(x["k"]), v=npu(x["v"]), beta=npu(x["beta"].bfloat16()),
        do=npu(x["do"]), dht=npu(x["dht"]), caches=caches,
    )
    want = _reference_grads(x)

    # 有限性先判 —— NaN 下相对 L2 没有判别力，而且会让断言信息难读
    broken = [n for n in BUDGET if not got[n].cpu().float().isfinite().all()]
    assert not broken, f"跨度 {got_span:.2f} 下这些梯度不是有限值：{broken}"
    # 参考自己也得有限，否则这一档的比对无效（单元 oracle 正是在这里失效的）
    bad_ref = [n for n in BUDGET if not want[n].isfinite().all()]
    assert not bad_ref, f"递推参考在跨度 {got_span:.2f} 下自己不是有限值：{bad_ref}，本档无效"


    # 判据是**对 fp32 递推参考**的相对 L2 —— 它是语义权威。
    #
    # ⚠️ 我试过换一个"更公平"的参考：把 g 经 bf16 `g_cumsum` 往返（反向 ABI 把该检查点定为
    # bf16），以为那才是可达目标。**那个想法是错的**，实测证伪：算子离 fp32 参考反而更近
    # （跨度 46：对 fp32 的 dq 2.889e-02，对 bf16-g 的 4.169e-02）。原因是那个构造把 bf16
    # cumsum **差分回** per-token 增量，属于灾难性相消；而 kernel 是**直接用** cumsum 去算
    # `exp(g_i − g_j)`，从不差分回去。所以那不是 ABI 的地板，是构造方式引入的误差。
    # 留着这段诊断是因为它解释了**为什么 kernel 必须直接用 cumsum**。
    errors = {n: _rel_l2(got[n].cpu(), want[n]) for n in BUDGET}
    want_diffed = _reference_grads(x, _through_bf16_cumsum(x["g"]))
    diffed = {n: _rel_l2(got[n].cpu(), want_diffed[n]) for n in BUDGET}
    print(f"\n跨度 {got_span:.2f}")
    print("  对 fp32 递推参考（判据）：        "
          + "  ".join(f"{n}={errors[n]:.3e}/{BUDGET[n]}" for n in BUDGET))
    print("  对「bf16 cumsum 差分回」的参考：  "
          + "  ".join(f"{n}={diffed[n]:.3e}" for n in BUDGET)
          + "   ← 更差，见上方注释")
    over = {n: (e, BUDGET[n]) for n, e in errors.items() if not (e < BUDGET[n])}
    assert not over, f"跨度 {got_span:.2f} 下超出契约预算：{over}"


def test_recurrent_reference_survives_where_unit_oracle_does_not():
    """把「宽域下只有递推参考可用」钉成一个会说话的测试。

    单元的 `ref.oracle` 在跨度 93.84 下实测有 3286/6507/16320 个非有限值（dq/dk/dg）。
    递推参考必须在同一档全程有限 —— 否则上面那组测试的参考本身就站不住。
    纯 CPU，不需要 NPU。
    """
    for span in (94.0, 150.0, 170.0):
        want = _reference_grads(_inputs(span))
        bad = {n: int((~t.isfinite()).sum()) for n, t in want.items() if not t.isfinite().all()}
        assert not bad, f"递推参考在跨度 {span} 下出现非有限值：{bad}"
    # 顺带固定住成因：单元 oracle 的成对矩阵在非因果半边会上溢
    span94_log2 = 94.0 / math.log(2.0)
    assert torch.exp2(torch.tensor(span94_log2)).isinf(), (
        "exp2(94/ln2) 居然没有上溢，说明 fp32 的量程假设变了，本测试的前提失效"
    )


def a2_fp32_end_to_end(inputs, device="cpu", segment=64):
    """Full FP32 recurrence and autograd, with explicit reverse chain rule.

    Save FP32 boundary states in a no-grad forward, then replay each segment with
    autograd and pass its dh0 to the preceding segment as dht. This computes the
    complete sequence derivative without retaining T copies of the 128x128 state.
    No forward cache, BF16 cumsum roundtrip, or custom kernel enters this oracle.
    CPU is the golden. The same Torch expression on NPU is a separate baseline.
    """
    names = ("q", "k", "v", "g", "beta")
    if any(inputs[n].device.type != "cpu" for n in (*names, "initial_state", "do", "dht")):
        raise ValueError("oracle inputs must be the original complete CPU fixtures")
    t = inputs["q"].shape[1]
    if t % segment:
        raise ValueError("oracle segment must divide T")
    source = {n: inputs[n].detach().float().to(device) for n in
              (*names, "initial_state", "do", "dht")}
    boundaries, outputs = [], []
    state = source["initial_state"]
    with torch.no_grad():
        for start in range(0, t, segment):
            boundaries.append(state)
            o, state = kda_recurrent_ref(
                *(source[n][:, start:start + segment] for n in names),
                initial_state=state, output_final_state=True)
            outputs.append(o)
    forward = {"o": torch.cat(outputs, 1).detach(), "final_state": state.detach()}
    gradients = {"d" + n: torch.empty_like(source[n]) for n in names}
    dstate = source["dht"]
    for ci in reversed(range(t // segment)):
        start, stop = ci * segment, (ci + 1) * segment
        leaves = [source[n][:, start:stop].detach().requires_grad_(True) for n in names]
        h0 = boundaries[ci].detach().requires_grad_(True)
        o, ht = kda_recurrent_ref(*leaves, initial_state=h0, output_final_state=True)
        grads = torch.autograd.grad((o, ht), (*leaves, h0),
                                    grad_outputs=(source["do"][:, start:stop], dstate))
        for n, value in zip(names, grads[:-1]):
            gradients["d" + n][:, start:stop] = value.detach()
        dstate = grads[-1].detach()
    gradients["dh0"] = dstate
    return forward, gradients


def test_a2_segmented_oracle_matches_full_autograd():
    """Validate segment-boundary adjoints against one full graph, including GQA."""
    from benchmarks.verify_decode import _a2_unit
    _, ref = _a2_unit("kda_bwd_stable")
    inputs = ref.make_inputs({"B": 1, "H": 1, "HV": 2, "C": 2,
                              "seed": 213, "gate_multiplier": .03})
    names = ("q", "k", "v", "g", "beta", "initial_state")
    leaves = [inputs[n].float().requires_grad_(True) for n in names]
    output, ht = kda_recurrent_ref(*leaves[:5], initial_state=leaves[5], output_final_state=True)
    grads = torch.autograd.grad((output, ht), leaves,
                               grad_outputs=(inputs["do"].float(), inputs["dht"].float()))
    actual_fwd, actual_grad = a2_fp32_end_to_end(inputs)
    torch.testing.assert_close(actual_fwd["o"], output, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_fwd["final_state"], ht, rtol=1e-6, atol=1e-7)
    for name, expected in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), grads):
        torch.testing.assert_close(actual_grad[name], expected, rtol=1e-6, atol=1e-7)
