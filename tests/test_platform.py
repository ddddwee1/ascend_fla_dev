"""ascend_fla.platform 的主机侧测试：SoC 解析、能力表、入口检查。

不需要 NPU。a5 的实测值在这里**写死**（A2-02 只搬家不改值，所以这些常量改了就算回归）。
"""
from __future__ import annotations

import pytest

from ascend_fla import platform


@pytest.fixture(autouse=True)
def _clear_soc_cache(monkeypatch):
    """每个用例前清进程内备忘并抹掉环境变量，用例自己按需设。"""
    monkeypatch.delenv(platform.SOC_ENV, raising=False)
    platform._reset_cache()
    yield
    platform._reset_cache()


# ---- a5 实测值写死（改前的值；变了就是回归） ----
A5_BLOCK_DIM_CHUNK = (1, 2, 3, 4)
A5_BLOCK_DIM_DECODE = (1, 2, 4, 8, 16, 28)
A5_GATE_SPAN = {
    "upstream": {"forward": 80.0, "backward": 80.0},
    "stable": {"forward": 155.0, "backward": 105.0},
}


def test_a5_capability_values_are_frozen():
    cap = platform.capability("a5")
    assert cap["qualified"] is True
    assert cap["supported_block_dim"]["chunk"] == A5_BLOCK_DIM_CHUNK
    assert cap["supported_block_dim"]["decode"] == A5_BLOCK_DIM_DECODE
    assert cap["max_gate_span"] == A5_GATE_SPAN
    assert cap["unit_root"] == "kernels/projects/a5"


def test_a5_accessors_match_ops_kda_constants():
    # 与 ops/kda 的现有常量必须逐字一致 —— platform 是它们的唯一事实源之后的别名来源。
    from ascend_fla.ops.kda.chunk import MAX_GATE_SPAN, SUPPORTED_BLOCK_DIM
    from ascend_fla.ops.kda.fused_recurrent import (
        SUPPORTED_BLOCK_DIM as DECODE_BLOCK_DIM,
    )

    assert platform.supported_block_dim("a5", "chunk") == SUPPORTED_BLOCK_DIM
    assert platform.supported_block_dim("a5", "decode") == DECODE_BLOCK_DIM
    assert platform.max_gate_span("a5") == MAX_GATE_SPAN


@pytest.mark.parametrize("soc", ["a2", "a3"])
def test_unqualified_socs_raise_on_use(soc):
    cap = platform.capability(soc)
    assert cap["qualified"] is False
    if soc == "a2":
        assert cap["supported_block_dim"] == {"chunk": (1, 2), "decode": (1, 2)}
        assert cap["max_gate_span"] == {"stable": {"forward": 158.0}}
    else:
        assert cap["supported_block_dim"] == {}
        assert cap["max_gate_span"] == {}
    with pytest.raises(RuntimeError, match="未验收"):
        platform.require_qualified(soc)
    with pytest.raises(RuntimeError, match="未验收"):
        platform.supported_block_dim(soc, "chunk")
    with pytest.raises(RuntimeError, match="未验收"):
        platform.max_gate_span(soc)


def test_require_qualified_passes_for_a5():
    platform.require_qualified("a5")  # 不抛


def test_resolve_from_env(monkeypatch):
    monkeypatch.setenv(platform.SOC_ENV, "a2")
    assert platform.resolve_soc() == "a2"


def test_explicit_wins_and_is_not_cached(monkeypatch):
    monkeypatch.setenv(platform.SOC_ENV, "a5")
    assert platform.resolve_soc("a2") == "a2"      # 显式实参优先
    assert platform.resolve_soc() == "a5"          # 且不污染缓存


def test_resolution_is_memoized(monkeypatch):
    monkeypatch.setenv(platform.SOC_ENV, "a5")
    assert platform.resolve_soc() == "a5"
    monkeypatch.setenv(platform.SOC_ENV, "a2")     # 缓存后改环境变量不再生效
    assert platform.resolve_soc() == "a5"
    platform._reset_cache()
    assert platform.resolve_soc() == "a2"          # 清缓存后重读


def test_unknown_soc_rejected(monkeypatch):
    monkeypatch.setenv(platform.SOC_ENV, "a9")
    with pytest.raises(ValueError, match="未知 SoC"):
        platform.resolve_soc()


def test_unresolvable_raises_with_how_to_set(monkeypatch):
    # 无环境变量、无 torch_npu：报错里必须带 ASCEND_FLA_SOC 的设法。
    monkeypatch.setattr(platform, "_probe_from_device_name", lambda: None)
    with pytest.raises(RuntimeError) as ei:
        platform.resolve_soc()
    assert platform.SOC_ENV in str(ei.value)


def test_unit_root_points_under_soc(monkeypatch):
    root = platform.unit_root("a2")
    assert root.name == "a2"
    assert root.parent.name == "projects"


def test_a5_compile_signature_unchanged(monkeypatch):
    """``device=None`` 解析成 a5 后，编译签名必须与显式 ``device="a5"`` 逐字相同 ——
    否则所有 a5 缓存失效，也让"改没改算式"变得难判（A2-02：只搬家不改值）。"""
    from ascend_fla.runtime import compile as _compile

    def _sig_probe():  # 一个有源码可读的假 kernel
        return None

    _sig_probe.name = "a2_02_sig_probe"
    sig_explicit = _compile._signature(_sig_probe, "a5", 1, {}, "cce")
    monkeypatch.setenv(platform.SOC_ENV, "a5")
    platform._reset_cache()
    sig_resolved = _compile._signature(_sig_probe, platform.resolve_soc(), 1, {}, "cce")
    assert sig_resolved == sig_explicit


def test_compile_signature_frozen_literals(monkeypatch):
    """Pin the compile signature to literals computed on base main (before A2-02), so a future
    change to the signing logic is caught, not just a self-consistency between two new codepaths.
    The first two freeze a5; the third keeps device in the key (a2 differs). Synthetic kernel so
    a real kernel's source edits never touch this test.
    """
    from ascend_fla.runtime import compile as _compile

    monkeypatch.setattr(_compile, "_kernel_source",
                        lambda kernel: "def pm_sig_probe():\n    return 0\n")

    def pm_sig_probe():
        return 0

    pm_sig_probe.name = "pm_sig_probe"
    assert _compile._signature(pm_sig_probe, "a5", 1, {}, "cce") == "986f4a15e1a1dd05"
    assert _compile._signature(pm_sig_probe, "a5", 4, {"T": 64}, "cce") == "d79923b948b20af3"
    assert _compile._signature(pm_sig_probe, "a2", 1, {}, "cce") == "63c73523d216ea41"


def test_op_entries_gate_before_compile_on_unqualified_soc(monkeypatch):
    """Every device-taking KDA op entry must raise "未验收" before any compile when the SoC is
    unqualified. Runs on the host (no NPU): the gate is the first statement, before the tensors
    are touched, so zero CPU inputs reach it. A5 resolution is covered by the signature test above.
    """
    import torch

    from ascend_fla.ops.kda import prepare
    from ascend_fla.ops.kda.autograd import chunk_kda
    from ascend_fla.ops.kda.chunk import chunk_kda_fwd, chunk_kda_fwd_with_caches
    from ascend_fla.ops.kda.chunk_bwd import chunk_kda_bwd
    from ascend_fla.ops.kda.fused_recurrent import fused_recurrent_kda

    monkeypatch.setenv(platform.SOC_ENV, "a2")
    platform._reset_cache()
    z = lambda *shape: torch.zeros(*shape)
    q_k_v_g_beta = (z(1, 64, 1, 128), z(1, 64, 1, 128), z(1, 64, 1, 128), z(1, 64, 1, 128), z(1, 64, 1))
    entries = {
        "prepare": lambda: prepare(),
        "chunk_kda": lambda: chunk_kda(*q_k_v_g_beta),
        "chunk_kda_fwd": lambda: chunk_kda_fwd(*q_k_v_g_beta),
        "chunk_kda_fwd_with_caches": lambda: chunk_kda_fwd_with_caches(*q_k_v_g_beta),
        "fused_recurrent_kda": lambda: fused_recurrent_kda(*q_k_v_g_beta),
        "chunk_kda_bwd": lambda: chunk_kda_bwd(
            z(1, 64, 1, 128), z(1, 64, 1, 128), z(1, 64, 1, 128), z(1, 64, 1),
            z(1, 64, 1, 128), z(1, 1, 128, 128), {}),
    }
    for name, call in entries.items():
        with pytest.raises(RuntimeError, match="未验收"):
            call()
