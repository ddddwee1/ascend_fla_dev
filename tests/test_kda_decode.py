"""decode 路径的主机侧检查（不需要 NPU）。

真机精度与 state 串接由 `benchmarks/verify_decode.py` 验（要 NPU）；这里盯的是
**两边会漂移的常量与门控** —— 那些东西一旦不一致，真机上表现为越界读而不是报错。
"""
from __future__ import annotations

import ast
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
UNIT = ROOT / "kernels/projects/a5/kda_fused_recurrent"


def _kernel_source() -> str:
    return (UNIT / "kernels/step.py").read_text(encoding="utf-8")


def _module_consts(path: pathlib.Path) -> dict:
    """用 ast 读模块级常量赋值 —— **不 import**。

    `fused_recurrent.py` 用相对 import（`from .chunk import ...`），单文件加载不了；
    而 import 整个包会把 torch_npu 拖进来，主机上没有。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        pass
    return out


def _op_consts() -> dict:
    # SUPPORTED_BLOCK_DIM is no longer a module-level literal (它从 platform 能力表读，A2-02)，
    # 所以走 import 取值；T_MAX 等仍是字面量，继续走 ast。import 在无 torch_npu 的主机上可用。
    consts = _module_consts(ROOT / "ascend_fla/ops/kda/fused_recurrent.py")
    import ascend_fla.ops.kda.fused_recurrent as fused_recurrent
    consts["SUPPORTED_BLOCK_DIM"] = fused_recurrent.SUPPORTED_BLOCK_DIM
    return consts


def _kernel_calls() -> set[str]:
    """kernel 源码里**实际调用**的方法名（不含注释与 docstring 里的文字）。"""
    names = set()
    for node in ast.walk(ast.parse(_kernel_source())):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                names.add(f.attr)
            elif isinstance(f, ast.Name):
                names.add(f.id)
    return names


def test_t_max_matches_between_kernel_and_wrapper():
    """``T_MAX`` 在 kernel 与算子入口里各写了一遍 —— 必须相等。

    不等的后果是**越界读**而不是报错：入口放过 T=24，kernel 的流式缓冲只有 16 行。
    """
    kernel_t_max = _module_consts(UNIT / "kernels/step.py").get("T_MAX")
    assert kernel_t_max is not None, "kernel 里找不到 T_MAX 的定义"
    assert kernel_t_max == _op_consts()["T_MAX"]
    # 缓冲行数必须用这个常量，不能再写字面量
    assert "DBuff(DT.float, [16," not in _kernel_source(), \
        "kernel 里还有写死的 16，应当用 T_MAX"


def test_block_dim_domain_matches_contract():
    """声明的 block_dim 必须与契约一致，而且每个值都得是实测过的。"""
    contract = json.loads((UNIT / "contract.json").read_text(encoding="utf-8"))
    assert list(_op_consts()["SUPPORTED_BLOCK_DIM"]) == contract["domain"]["block_dim"]
    # 28 是物理上限（56 个向量核），不能再往上声明
    assert max(contract["domain"]["block_dim"]) == 28


def test_no_cross_core_sync_in_kernel():
    """本 kernel 刻意不手写跨核同步 —— 手写同步正是 c1-multihead-o-corrupt 的成因。

    盯两件事：① 用 ``auto_sync()``；② 没有手写的 DEvent / Mutex。
    """
    src = _kernel_source()
    assert "auto_sync()" in src, "没有用 auto_sync()"
    for bad in ("DEvent(", "CvMutex(", "VcMutex("):
        assert bad not in src, (
            f"kernel 里出现了手写同步 {bad} —— 本单元的设计前提是每头一核、核间无同步，"
            f"若确实需要手写同步，要先解释为什么不会重犯 c1-multihead-o-corrupt"
        )


def test_cadd_is_not_used_as_a_broadcast_multiplier():
    """``cadd()`` 的结果只落在 lane 0，不是广播 —— 当乘数用会只有 1/128 个 lane 正确。

    这是实测踩过的（o 相对 L2 8.5e-02 而 final_state 3.2e-08）。现在的实现根本不用
    ``cadd``；这个测试防止有人"优化"时把它加回来而不经过 UB 往返广播。
    """
    assert "cadd" not in _kernel_calls(), (
        "kernel 里出现了 cadd() —— 它的结果只在 lane 0。要当乘数用必须先经 UB 存一次、"
        "再用 .single() 读回广播（见 ascriptor gdn_bwd 的 broadcast_scalar_vf）；"
        "本单元的两趟扫法本来就不需要它，见 README"
    )


def test_decode_gap_records_the_call_overhead_finding():
    gaps = json.loads((ROOT / "docs/matrix/gaps.json").read_text(encoding="utf-8"))
    by = {g["id"]: g for g in gaps["gaps"]}
    assert "decode-call-overhead" in by, "gaps.json 里没有 decode-call-overhead"
    assert by["decode-call-overhead"]["severity"] == "P1"
    # decode 缺口不能再说"算子完全缺失" —— KDA 那半已经做了
    assert "完全缺失" not in by["fused-recurrent-missing"]["title"]


def test_contract_declares_no_gate_span_limit():
    """recurrent 形式结构性地没有门控跨度上限 —— 这一点必须写在契约里。

    chunk 路径有前向 155 / 反向 105 的闸；decode 没有，因为它只用 exp(g_i)。
    调用方据此选路，所以不能只写在 README 里。
    """
    contract = json.loads((UNIT / "contract.json").read_text(encoding="utf-8"))
    span = contract["domain"]["gate_span"]
    assert "无上限" in span and "exp(g_i)" in span


# A2-14: verifier regression tests. All remain CPU-only, with no kernel compilation.
def _a2_verifier():
    import importlib.util
    path = ROOT / "benchmarks/verify_decode.py"
    spec = importlib.util.spec_from_file_location("_test_decode_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a2_fp32_oracle_preserves_gqa_decay_and_nonzero_state():
    """Analytic one-hot case distinguishes K-axis decay, GQA mapping and state reuse."""
    import torch
    verifier = _a2_verifier()
    q = torch.zeros(1, 2, 2, 128, dtype=torch.bfloat16)
    q[:, :, 0, 0] = 1
    q[:, :, 1, 1] = 1
    x = dict(q=q, k=q.clone(), v=torch.zeros(1, 2, 4, 128, dtype=torch.bfloat16),
             g=torch.zeros(1, 2, 4, 128), beta=torch.full((1, 2, 4), .5),
             initial_state=torch.zeros(1, 4, 128, 128))
    x["initial_state"][0, :, 0, 3] = 2
    x["initial_state"][0, :, 1, 4] = 4
    result = verifier.a2_fp32_oracle(x)
    assert all(t.dtype == torch.float32 and t.device.type == "cpu" for t in result.values())
    expected_state = x["initial_state"].clone()
    expected_state[:, :2, 0, :] *= .25
    expected_state[:, 2:, 1, :] *= .25
    torch.testing.assert_close(result["final_state"], expected_state, rtol=0, atol=0)
    expected_o = torch.zeros(1, 2, 4, 128)
    expected_o[0, 0, :2, 3] = 128 ** -.5
    expected_o[0, 1, :2, 3] = .5 * 128 ** -.5
    expected_o[0, 0, 2:, 4] = 2 * 128 ** -.5
    expected_o[0, 1, 2:, 4] = 128 ** -.5
    torch.testing.assert_close(result["o"], expected_o, rtol=0, atol=0)
    # A gate on one K row must not decay all rows or a V column.
    x["g"][:, :, :, 0] = -1
    decayed = verifier.a2_fp32_oracle(x)
    torch.testing.assert_close(decayed["final_state"][:, :2, 0, 3],
                               expected_state[:, :2, 0, 3] * torch.exp(torch.tensor(-2.)))
    torch.testing.assert_close(decayed["final_state"][:, :, 1, :], expected_state[:, :, 1, :])


def test_a2_metrics_reject_nan_and_small_l2_with_bad_element():
    import torch
    verifier = _a2_verifier()
    contract = json.loads((ROOT / "kernels/projects/a2/kda_fused_recurrent/contract.json").read_text())
    rule = verifier._a2_rules(contract)["final_state"]
    expected = torch.ones(10000)
    actual = expected.clone()
    actual[0] += 1e-4
    metrics = verifier.a2_metrics(actual, expected, rule)
    assert metrics["rel_l2"] < rule["max_relative_l2"]
    assert not metrics["allclose"] and not metrics["passed"]
    actual[0] = float("nan")
    assert not verifier.a2_metrics(actual, expected, rule)["passed"]


def test_a2_metrics_also_enforce_l2_and_exact_shape():
    import pytest
    import torch
    verifier = _a2_verifier()
    rule = dict(rtol=1., atol=1., max_relative_l2=1e-5)
    metric = verifier.a2_metrics(torch.ones(2) * 1.01, torch.ones(2), rule)
    assert metric["allclose"] and not metric["passed"]
    with pytest.raises(ValueError, match="shape mismatch"):
        verifier.a2_metrics(torch.ones(1, 2), torch.ones(2), rule)


def test_a2_reference_loaders_do_not_share_bare_reference_module():
    import sys
    verifier = _a2_verifier()
    before = sys.modules.get("reference")
    decode, ref_decode = verifier._a2_unit("kda_fused_recurrent")
    chunk, ref_chunk = verifier._a2_unit("kda_fwd_stable")
    assert decode.independent_reference is ref_decode.independent_reference
    assert chunk.independent_reference is ref_chunk.independent_reference
    assert decode.independent_reference is not chunk.independent_reference
    assert sys.modules.get("reference") is before


def test_a2_cross_bd_rejects_mismatch_missing_case_and_identity():
    import copy
    import pytest
    import torch
    verifier = _a2_verifier()
    outputs = {f"real_h{h}_t{t}": {"o": verifier._a2_digest(torch.zeros(1, t, 32, 128, dtype=torch.bfloat16)),
                                 "final_state": verifier._a2_digest(torch.zeros(1, 32, 128, 128))}
               for h in (32, 16) for t in (1, 4, 8, 16)}
    left = dict(block_dim=1, identity={"source": "same"}, passed=True, outputs=outputs,
                inputs={name: {"digest": "same_input"} for name in outputs})
    right = copy.deepcopy(left)
    right["block_dim"] = 2
    assert verifier.a2_compare_block_dims(left, right)["passed"]
    right["outputs"]["real_h16_t16"]["o"]["sha256"] = "different"
    assert not verifier.a2_compare_block_dims(left, right)["passed"]
    del right["outputs"]["real_h16_t16"]
    with pytest.raises(ValueError, match="cases missing"):
        verifier.a2_compare_block_dims(left, right)
    right["identity"] = {"source": "other"}
    with pytest.raises(ValueError, match="identities differ"):
        verifier.a2_compare_block_dims(left, right)
    right = copy.deepcopy(left)
    right["block_dim"] = 2
    right["inputs"]["real_h32_t1"]["digest"] = "different_input"
    with pytest.raises(ValueError, match="input identities"):
        verifier.a2_compare_block_dims(left, right)
    right["inputs"] = copy.deepcopy(left["inputs"])
    right["outputs"]["real_h32_t1"] = {}
    with pytest.raises(ValueError, match="both outputs"):
        verifier.a2_compare_block_dims(left, right)


def test_a2_precompiles_all_six_vendors_before_launch(monkeypatch):
    from types import SimpleNamespace
    import ascend_fla.runtime.compile as compiler
    verifier = _a2_verifier()
    decode = SimpleNamespace(HERE=ROOT / "kernels/projects/a2/kda_fused_recurrent", _kernel=lambda: "decode")
    chunk = SimpleNamespace(HERE=ROOT / "kernels/projects/a2/kda_fwd_stable",
                            _kernels=lambda: {name: name for name in ("gate", "intra", "triangular_inverse", "wy", "recurrent")})
    monkeypatch.setattr(verifier, "_a2_unit", lambda name: (decode if name == "kda_fused_recurrent" else chunk, None))
    built = []
    def compile_stub(kernel, **kwargs):
        assert kwargs == dict(device="a2", block_dim=2, backend="cce")
        built.append(kernel)
        return SimpleNamespace(signature=kernel, scalar_names=[])
    monkeypatch.setattr(compiler, "compile_kernel", compile_stub)
    events = []
    native = verifier._A2Native(2, events.append)
    assert len(set(built)) == 6 and len(native.compiled) == 6
    assert not native.trace
    assert events[-1] == dict(stage="precompile_complete", count=6, custom_launches=0)


def test_a2_decode_has_no_matmul_and_keeps_declared_launch_domain():
    """No matmul means neither split-K nor handwritten MMAD accumulation is exercised."""
    unit = ROOT / "kernels/projects/a2/kda_fused_recurrent"
    tree = ast.parse((unit / "kernels/step.py").read_text())
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert "matmul" not in calls
    contract = json.loads((unit / "contract.json").read_text())
    assert contract["domain"]["block_dim"] == [1, 2]
    assert "no chunk-span limit" in contract["domain"]["gate_span"]
    assert _module_consts(unit / "kernels/step.py")["T_MAX"] == 16
