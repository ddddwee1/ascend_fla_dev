"""kda_bwd 九 kernel 链经 runtime 桥的梯度回归（需要 A5 NPU + CANN）。

参考是 ``kda_bwd`` 单元自己的 ``ref.oracle.oracle`` —— bf16 输入、fp32 前向公式上
autograd、输出降回 bf16。用它而不是自拟参考：那是这个单元被验证时用的同一个 oracle。

**预算取自 contract.json 的 ``comparison``，不要自拟。** 它逐输出给了不同上限：

* 默认 ``max_relative_l2 = 0.05``
* ``dk`` = 0.15 —— 契约的理由：bf16 的 log2 累积门控缓存本身会把参考实现的相对 L2
  推到 0.046~0.096，因为把 log2 门控舍到 0.25 以内再 exp2 会改变乘性导数。
* ``dg`` = 0.25 —— 同样的原因，参考实现自身就是 0.115~0.183。

也就是说 ``dk`` / ``dg`` 的宽预算是**算子精度 ABI 的一部分**，不是我们放水。
契约同时要求这个界"足以拒绝零梯度或严重错误的梯度"，所以下面另外检查非零。

    pytest tests/test_kda_bwd_npu.py -v

形状与 ``block_dim`` 直接用 contract.json 的五个 case，包括 ``gentle_decay``
（``gate_multiplier=0.03``，浅衰减）与 ``grouped_idle_cores``（``block_dim=3`` 而
工作量只有 2 个头对，故意让部分核空转）。

**每个 case 起一个子进程。** 这些 case 的 ``block_dim`` 不同，而同一算子名的多份
build 在一个进程里会互相覆盖（``runtime/binding.py`` 的 ``_claim_op_name`` 会直接
报错）。所以本文件兼作 worker：``python tests/test_kda_bwd_npu.py <case_id>`` 跑单个
case 并把误差打成 JSON，pytest 侧只负责派进程与判预算。
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

torch_npu = pytest.importorskip("torch_npu", reason="需要 torch_npu")
if not torch.npu.is_available():  # pragma: no cover
    pytest.skip("没有可用的 NPU", allow_module_level=True)

from ascend_fla.ops.kda.chunk import chunk_kda_fwd_with_caches  # noqa: E402
from ascend_fla.ops.kda.chunk_bwd import _bwd_kernels_root, chunk_kda_bwd  # noqa: E402

# contract.json 的 comparison：默认 0.05，dk/dg 另有更宽的上限（见 docstring）
BUDGET = {"dq": 0.05, "dk": 0.15, "dv": 0.05, "dbeta": 0.05, "dg": 0.25, "dh0": 0.05}

# contract.json 的 cases，逐项照搬：(B, H, HV, C, gate_multiplier, block_dim)
CASES = {
    "single_chunk": (1, 1, 1, 1, 1.0, 1),
    "multi_chunk": (1, 1, 1, 2, 1.0, 1),
    "grouped_heads": (1, 1, 2, 2, 1.0, 2),
    "gentle_decay": (1, 1, 1, 2, 0.03, 1),
    "grouped_idle_cores": (1, 1, 2, 1, 1.0, 3),
}


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _load_ref():
    """导入 kda_bwd 单元的 ``ref`` 包。worker 与 pytest 两侧共用。"""
    root = _bwd_kernels_root()
    sys.path.insert(0, str(root))
    from ref.inputs import make_inputs as source_make_inputs
    from ref.oracle import oracle
    from ref.types import KDAOracleInputs
    return oracle, source_make_inputs, KDAOracleInputs


def run_case(case_id: str) -> dict:
    """跑一个 case，返回每个梯度的相对 L2 与是否退化为全零。

    worker 的主体。pytest 侧一律派子进程调它，不在自己进程里跑。
    """
    b, h, hv, c, gate_mul, block_dim = CASES[case_id]
    oracle, source_make_inputs, Inputs = _load_ref()

    # 用单元自己的 make_inputs：同分布、同 seed 语义，省得我们复刻它的分布假设
    src = source_make_inputs(B=b, H=h, HV=hv, C=c, K=128, V=128, chunk_size=64, seed=2026)
    x = dict(vars(src))
    if gate_mul != 1.0:
        x["g"] = (x["g"].float() * gate_mul).to(torch.bfloat16)

    ref = oracle(Inputs(**{n: x[n] for n in Inputs.__dataclass_fields__}))
    want = dict(vars(ref))

    # 前向按 kda_fwd 的 ABI 收 fp32 的 g/beta/initial_state；单元的输入是 bf16，
    # 这里升回 fp32 —— 信息量不变，只是匹配 fwd 的入口类型
    npu = {n: x[n].to("npu") for n in ("q", "k", "v")}
    npu |= {n: x[n].float().to("npu") for n in ("g", "beta", "initial_state")}
    _, _, caches = chunk_kda_fwd_with_caches(**npu, block_dim=block_dim)

    got = chunk_kda_bwd(
        q=npu["q"], k=npu["k"], v=npu["v"], beta=x["beta"].to("npu"),
        do=x["do"].to("npu"), dht=x["dht"].to("npu"), caches=caches, block_dim=block_dim,
    )

    out = {"case": case_id, "errors": {}, "shape_mismatch": {}, "all_zero": []}
    for name in BUDGET:
        exp = want[name]
        if tuple(got[name].shape) != tuple(exp.shape):
            out["shape_mismatch"][name] = [list(got[name].shape), list(exp.shape)]
            continue
        out["errors"][name] = _rel_l2(got[name].cpu(), exp)
        # 契约要求预算"足以拒绝零梯度"。relL2 对全零会给 1.0，但若参考本身也近零就可能
        # 漏过，所以显式查一次。
        # 先 D2H 再 cast —— 缺内置算子包的机器上 NPU 侧的 .float() 不可用（AGENTS.md §5）
        if got[name].cpu().float().abs().max().item() == 0.0:
            out["all_zero"].append(name)
    return out


@pytest.mark.parametrize("case_id", list(CASES))
def test_bwd_matches_unit_oracle(case_id):
    """九 kernel 反向链的梯度对齐单元 oracle。每个 case 一个子进程，理由见模块 docstring。"""
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), case_id],
                       capture_output=True, text=True)
    payload = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("{")), None)
    assert r.returncode == 0 and payload, (
        f"worker 失败 exit={r.returncode}\n--- stdout ---\n{r.stdout[-3000:]}"
        f"\n--- stderr ---\n{r.stderr[-3000:]}"
    )
    out = json.loads(payload)
    assert not out["shape_mismatch"], f"形状不符：{out['shape_mismatch']}"
    errors = out["errors"]
    msg = "  ".join(f"{n}={errors[n]:.3e}/{BUDGET[n]}" for n in BUDGET if n in errors)
    assert not out["all_zero"], f"梯度全零：{out['all_zero']}；误差 {msg}"
    bad = {n: (e, BUDGET[n]) for n, e in errors.items() if not (e < BUDGET[n])}
    assert not bad, f"超出契约预算：{bad}；全部 {msg}"
    print(f"\n{case_id}: {msg}")


def test_bwd_gate_rejects_bad_inputs():
    """门控必须报错而不是静默降级（AGENTS.md §7）。"""
    _, source_make_inputs, _ = _load_ref()
    src = source_make_inputs(B=1, H=1, HV=1, C=1, K=128, V=128, chunk_size=64, seed=2026)
    x = dict(vars(src))
    npu = {n: x[n].to("npu") for n in ("q", "k", "v")}
    npu |= {n: x[n].float().to("npu") for n in ("g", "beta", "initial_state")}
    _, _, caches = chunk_kda_fwd_with_caches(**npu)
    base = dict(q=npu["q"], k=npu["k"], v=npu["v"], beta=x["beta"].to("npu"),
                do=x["do"].to("npu"), dht=x["dht"].to("npu"), caches=caches)

    with pytest.raises(ValueError, match="block_dim"):
        chunk_kda_bwd(**base, block_dim=5)

    with pytest.raises(ValueError, match="bfloat16"):       # beta 用了 fp32
        # 在 CPU 上 cast 再 H2D —— NPU 侧的 .float() 在缺内置算子包的机器上不可用，
        # 那会让这条门控测试死在构造输入上而不是验到门控（AGENTS.md §5）
        chunk_kda_bwd(**{**base, "beta": x["beta"].float().to("npu")})

    with pytest.raises(ValueError, match="caches 必须恰好是"):
        chunk_kda_bwd(**{**base, "caches": {k: v for k, v in caches.items() if k != "h"}})

    with pytest.raises(ValueError, match="必须是连续张量"):
        # 非连续输入必须报错而不是悄悄修正 —— 缺内置算子包的机器上既不能在 device 上
        # contiguous()，也不能对跨步视图 D2H（要 Slice，实测 errno 561000）。
        # 造法：在 CPU 上把 T 翻倍、H2D（连续），再取 ::2 的视图 —— 形状对得上、仅改 stride，
        # 纯元数据操作不需要任何 NPU 算子，所以这条门控在两种机器上都测得到。
        strided = x["do"].repeat(1, 2, 1, 1).to("npu")[:, ::2]
        assert not strided.is_contiguous() and strided.shape == base["do"].shape
        chunk_kda_bwd(**{**base, "do": strided})

    with pytest.raises(ValueError, match=r"caches\['h'\]"):  # h 形状错
        bad = dict(caches)
        bad["h"] = caches["h"][:, :1] if caches["h"].shape[1] > 1 else caches["h"].squeeze(1)
        chunk_kda_bwd(**{**base, "caches": bad})


def a2_qualification_cases():
    """Real model shapes, odd chunks, multiple GQA groups and B>1."""
    small = [(1, 64, 1, 1), (1, 128, 1, 2), (1, 192, 2, 4), (2, 128, 2, 4)]
    cases = [dict(id=f"cache_b{b}_t{t}_h{h}_hv{hv}", B=b, T=t, H=h, HV=hv,
                  seed=213, span=1.0, gate="uniform") for b, t, h, hv in small]
    cases += [dict(id=f"kimi_t{t}_default", B=1, T=t, H=32, HV=32,
                   seed=0, span=None, gate="fla_initialization") for t in (64, 128, 512, 4096)]
    cases.append(dict(id="gqa_t192_h16_hv32", B=1, T=192, H=16, HV=32,
                      seed=1, span=None, gate="fla_initialization"))
    return cases


def a2_qualification_inputs(case):
    """CPU BF16 fixtures satisfy both unit ABIs without changing their values.

    Forward g/beta/h0 are promoted exactly to FP32 when calling the forward unit.
    All oracles use these same BF16-representable input values. Gate initialization
    is labelled before and after ABI quantization, never presented as trained data.
    """
    from ascend_fla.ops.kda.chunk import _gate_span
    from benchmarks.verify_real_shapes import a2_default_gate
    b, t, h, hv = (case[n] for n in ("B", "T", "H", "HV"))
    gen = torch.Generator().manual_seed(case["seed"] + 21300)
    q, k = (torch.nn.functional.normalize(torch.randn(b, t, h, 128, generator=gen), dim=-1)
            .bfloat16() for _ in range(2))
    if case["gate"] == "fla_initialization":
        if b != 1 or hv != 32:
            raise ValueError("Kimi initialization fixtures require B=1, HV=32")
        gate, metadata = a2_default_gate(t, case["seed"], case["span"])
    else:
        gate = -torch.rand(b, t, hv, 128, generator=gen) * .03
        gate *= case["span"] / _gate_span(gate, t // 64, on_cpu=True)
        metadata = {"target_span": case["span"], "measured_span": _gate_span(gate, t // 64, on_cpu=True)}
    result = {"q": q, "k": k, "g": gate.bfloat16(),
              "v": (torch.randn(b, t, hv, 128, generator=gen) * .04).bfloat16(),
              "beta": (torch.rand(b, t, hv, generator=gen) * .45 + .05).bfloat16(),
              "initial_state": (torch.randn(b, hv, 128, 128, generator=gen) * .01).bfloat16(),
              "do": (torch.randn(b, t, hv, 128, generator=gen) * .04).bfloat16(),
              "dht": (torch.randn(b, hv, 128, 128, generator=gen) * .01).bfloat16()}
    metadata.update(actual_bf16_input_span=_gate_span(result["g"], t // 64, on_cpu=True),
                    gate_fixture=case["gate"], input_dtype="BF16, promoted exactly for forward and FP32 oracles")
    return {n: v.contiguous() for n, v in result.items()}, metadata


if __name__ == "__main__":
    # worker 模式：python tests/test_kda_bwd_npu.py <case_id>
    if len(sys.argv) != 2 or sys.argv[1] not in CASES:
        print(f"用法：{sys.argv[0]} <{'|'.join(CASES)}>", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(run_case(sys.argv[1]), ensure_ascii=False))
