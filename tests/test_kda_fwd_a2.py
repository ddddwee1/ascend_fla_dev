"""A2 KDA qualification checks, independent of the A5 NPU session fixture."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location("a212_benchmark_tests", ROOT / "benchmarks/verify_real_shapes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a2_comparator_rejects_finite_wrong_head_and_nonfinite(benchmark):
    expected = torch.tensor([[.02, .03], [-.05, .08]])
    wrong = expected.flip(0)
    assert torch.equal(wrong.norm(), expected.norm())
    assert benchmark.a2_metrics(expected, expected)["ok"]
    result = benchmark.a2_metrics(wrong, expected)
    assert result["finite"] and not result["ok"] and result["relative_l2"] > .05
    assert not benchmark.a2_metrics(expected * float("nan"), expected)["finite"]
    assert not benchmark.a2_metrics(expected * float("inf"), expected)["ok"]


def test_a2_comparator_does_not_hide_zero_reference_error(benchmark):
    zero = torch.zeros(2)
    assert benchmark.a2_metrics(zero, zero)["relative_l2"] == 0
    for amplitude in (1e-20, 1e-30):
        bad = benchmark.a2_metrics(torch.tensor([amplitude, 0.]), zero)
        assert bad["finite"] and bad["relative_l2"] is None and not bad["ok"]


def test_a2_gate_calibration_preserves_real_shape_and_rng(benchmark):
    before = torch.get_rng_state().clone()
    gate, calibration = benchmark.a2_default_gate(128, 0, 32.)
    assert torch.equal(before, torch.get_rng_state())
    assert gate.shape == (1, 128, 32, 128)
    assert gate.dtype == torch.float32 and gate.is_contiguous()
    assert bool(torch.isfinite(gate).all()) and bool((gate <= 0).all())
    assert calibration["hidden_size"] == 2304
    assert calibration["measured_span"] == pytest.approx(32., rel=2e-6)
    assert calibration["initialization_span"] != calibration["measured_span"]


@pytest.mark.parametrize("target", [0., -1., float("inf"), float("nan")])
def test_a2_gate_calibration_rejects_invalid_target(benchmark, target):
    with pytest.raises(ValueError, match="finite and positive"):
        benchmark.a2_default_gate(64, 1, target)


def test_a2_public_entry_stays_unqualified_before_compile(monkeypatch):
    from ascend_fla import platform
    from ascend_fla.ops.kda import chunk_kda_fwd
    from ascend_fla.runtime import compile as compiler

    def forbidden(*args, **kwargs):
        pytest.fail("Unqualified A2 entry attempted compilation")

    monkeypatch.setattr(compiler, "compile_kernel", forbidden)
    assert platform.capability("a2")["qualified"] is False
    with pytest.raises(RuntimeError, match="未验收"):
        chunk_kda_fwd(None, None, None, None, None, device="a2", block_dim=1)


@pytest.mark.parametrize("bd", [1, 2])
def test_a2_unit_c1_hv32_is_declared_positive(benchmark, bd):
    if not os.environ.get("ASCRIPTOR_WORKSPACE"):
        pytest.skip("Set ASCRIPTOR_WORKSPACE to the selected owner source bundle")
    unit, _, _ = benchmark._a2_unit()
    x = benchmark.make_inputs(1, 32, 32, 1, 8.)
    inputs = dict(q=x["q"], k=x["k"], v=x["v"], g_raw=x["g"],
                  beta=x["beta"], initial_state=x["h0"])
    unit.validate_inputs(inputs, {"block_dim": bd})
    with pytest.raises(ValueError, match="block_dim"):
        unit.validate_inputs(inputs, {"block_dim": 3})
    for name in ("q", "k", "v"):
        with pytest.raises(ValueError, match="bfloat16"):
            unit.validate_inputs({**inputs, name: inputs[name].float()}, {"block_dim": bd})


EVIDENCE = ROOT / "benchmarks/a2/evidence/kda_fwd"
STAGES = {"o", "final_state", "g_cumsum", "eg", "Aqk", "strict", "Akk", "w", "u", "qg", "kg"}
LENGTHS = {64, 128, 256, 512, 4096}


def _records(name):
    return json.loads((EVIDENCE / "runs" / name / "results.json").read_text())


def _assert_accepted(row):
    assert row["shape"] == [1, row["shape"][1], 32, 32, 128, 128]
    assert row["shape"][1] in LENGTHS
    assert row["unchanged"] is True
    assert set(row["output_sha256"]) == STAGES
    assert set(row["finite_stages"]) == STAGES and all(row["finite_stages"].values())
    assert len(row["execution"]) == 5 and len({x["signature"] for x in row["execution"]}) == 5
    assert [x["kernel"] for x in row["execution"]] == [
        "KdaSub1GateA2Kernel", "KdaSub2ScoreA2Kernel", "TrilInverse64A2Kernel",
        "KdaSub3WyA2Kernel", "KdaSub45A2Kernel"]
    for name in ("o", "final_state"):
        metric = row["comparison"][name]
        assert metric["finite"] is True and metric["ok"] is True
        assert 0 <= metric["relative_l2"] <= .05 and metric["max_abs_diff"] >= 0
        for oracle in ("cpu_vs_fla", "independent_vs_fla"):
            assert row[oracle][name]["finite"] is True
            assert 0 <= row[oracle][name]["relative_l2"] <= 1e-5


@pytest.mark.parametrize("bd", [1, 2])
def test_a2_native_real_shapes_and_default_initialization_receipts(bd):
    real = _records(f"real-bd{bd}")
    assert len(real) == 5 and {r["shape"][1] for r in real} == LENGTHS
    defaults = _records(f"defaults-bd{bd}")
    assert {r["id"] for r in defaults} == {f"t4096_seed{s}_spanNone" for s in range(8)}
    assert len(defaults) == 8
    for row in real + defaults:
        _assert_accepted(row)
        assert row["block_dim"] == bd
    for row in defaults:
        calibration = row["calibration"]
        assert calibration["target_span"] is None
        assert calibration["initialization_span"] == calibration["measured_span"]


def _range_guard():
    import math

    bounds = [b for bd in (1, 2) for b in json.loads(
        (EVIDENCE / "runs" / f"refine-bd{bd}" / "bounds.json").read_text())]
    assert {(b["t"], b["seed"], b["block_dim"]) for b in bounds} == {
        (t, s, bd) for bd in (1, 2) for t, s in [(4096, 0), (4096, 1), (4096, 2), (128, 0), (64, 0)]}
    assert len(bounds) == 10
    for bound in bounds:
        assert bound["first_bad"] - bound["last_good"] == 2
        observations = {}
        for step in bound["history"]:
            row = json.loads((EVIDENCE / "runs" / step["record"]).read_text())
            assert row["id"] == f"t{bound['t']}_seed{bound['seed']}_span{float(step['target'])}"
            assert row["calibration"]["target_span"] == step["target"]
            assert row["block_dim"] == bound["block_dim"]
            passed = all(m["finite"] and m["relative_l2"] is not None and 0 <= m["relative_l2"] <= .05
                         for m in row["comparison"].values()) and all(row["finite_stages"].values())
            assert passed == (all(m["ok"] for m in row["comparison"].values()) and all(row["finite_stages"].values()))
            assert passed == step["good"]
            observations[step["target"]] = passed
        assert observations[bound["last_good"]] is True
        assert observations[bound["first_bad"]] is False
    return min(math.floor(.9 * min(b["first_bad"] for b in bounds) / 2) * 2,
               min(b["last_good"] for b in bounds))


def test_a2_guard_has_measured_upper_and_lower_bounds():
    from ascend_fla.platform import CAPABILITIES

    coarse = _records("range-coarse-bd1")
    assert {r["id"] for r in coarse} == {
        f"t4096_seed{s}_span{float(span)}" for s in range(3) for span in (8, 16, 32, 64, 96, 128, 160, 192)}
    assert len(coarse) == 24
    assert sum(not all(m["ok"] for m in r["comparison"].values()) for r in coarse) == 3
    guard = _range_guard()
    assert CAPABILITIES["a2"]["qualified"] is False
    assert CAPABILITIES["a2"]["supported_block_dim"] == {"chunk": (1, 2), "decode": (1, 2)}
    assert CAPABILITIES["a2"]["max_gate_span"]["stable"]["forward"] == float(guard)
    for bd in (1, 2):
        defaults = _records(f"defaults-bd{bd}")
        assert max(r["calibration"]["initialization_span"] for r in defaults) < guard
        rows = _records(f"guard-bd{bd}")
        assert {r["id"] for r in rows} == {
            f"t{t}_seed{s}_span{float(guard)}" for t in LENGTHS for s in range(8)}
        assert len(rows) == 40
        for row in rows:
            _assert_accepted(row)
            assert row["block_dim"] == bd
            assert row["calibration"]["measured_span"] == pytest.approx(guard, rel=2e-6)


def test_a2_block_dimensions_preserve_all_eleven_tensors():
    index = json.loads((EVIDENCE / "runs.json").read_text())
    assert index["real-bd1"]["same_card_group"] == index["real-bd2"]["same_card_group"]
    for suite in ("real", "defaults", "guard"):
        a = {r["id"]: r for r in _records(f"{suite}-bd1")}
        b = {r["id"]: r for r in _records(f"{suite}-bd2")}
        assert a.keys() == b.keys()
        for key in a:
            assert a[key]["input_sha256"] == b[key]["input_sha256"]
            assert a[key]["output_sha256"] == b[key]["output_sha256"]


@pytest.mark.parametrize("bd", [1, 2])
def test_a2_same_card_timing_preserves_raw_sandwich_samples(bd):
    import math

    rows = _records(f"performance-bd{bd}")
    assert len(rows) == 2 and {r["shape"][1] for r in rows} == {128, 4096}
    for row in rows:
        _assert_accepted(row)
        timing = row["timing"]
        assert timing["synchronized"] is True
        assert (timing["warmup_per_phase"], timing["repeat_per_phase"], timing["rounds"]) == (3, 20, 3)
        assert [(s["round"], s["position"]) for s in timing["samples"]] == [
            (r, p) for r in range(3) for p in ("baseline_before", "candidate", "baseline_after")]
        for sample in timing["samples"]:
            assert len(sample["wall_ms"]) == 20
            assert all(math.isfinite(x) and x > 0 for x in sample["wall_ms"])
        for path in ("candidate", "baseline"):
            assert all(m["finite"] and m["ok"] and 0 <= m["relative_l2"] <= .05
                       for m in timing["correctness"][path].values())


def test_a2_receipts_match_frozen_sources_and_publish_manifest(benchmark):
    import hashlib

    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    index = json.loads((EVIDENCE / "runs.json").read_text())
    for name, run in index.items():
        environments = list((EVIDENCE / "runs" / name).rglob("environment.json"))
        assert len(environments) == (run["case_count"] if name.startswith("refine-") else 1)
    for path in (EVIDENCE / "runs").rglob("environment.json"):
        env = json.loads(path.read_text())
        assert env["soc"] == "a2" and env["chip"] == "Ascend910B3"
        assert env["source_pins"] == benchmark.A2_PINS
        assert {"ascend910b", "ascend910_93"} <= set(env["opp_packages"])
        for package in ("compiler/version.info", "opp/version.info"):
            assert env["cann"][package]["fields"]["Version"] == "9.0.0"
            assert env["cann"][package]["fields"]["timestamp"] == "20260428_134817545"
        for source, expected in env["source_sha256"].items():
            if source == "benchmarks/verify_real_shapes.py":
                assert digest(EVIDENCE / "sources" / f"{expected}.py") == expected
                if "range-coarse-bd1" not in path.parts:
                    assert digest(ROOT / source) == expected
            else:
                assert digest(ROOT / source) == expected
    manifest = json.loads((EVIDENCE / "redaction-manifest.json").read_text())
    assert manifest
    for entry in manifest:
        assert digest(EVIDENCE / entry["path"]) == entry["published_sha256"]
    compiled = json.loads((EVIDENCE / "compiled-artifact-hashes.json").read_text())
    for bd in (1, 2):
        for launch in _records(f"real-bd{bd}")[0]["execution"]:
            binaries = [a for a in compiled if f"-{launch['signature']}/custom_op/" in a["path"]
                        and a["path"].endswith(".o") and "/ascend910b/" in a["path"]]
            assert len(binaries) == 1 and binaries[0]["bytes"] > 0
            assert len(binaries[0]["sha256"]) == 64


def test_a2_native_batches_have_completed_device_leases():
    index = json.loads((EVIDENCE / "runs.json").read_text())
    assert set(index) == {"range-coarse-bd1", "refine-bd1", "refine-bd2"} | {
        f"{suite}-bd{bd}" for suite in ("real", "defaults", "guard", "performance") for bd in (1, 2)}
    leases = json.loads((EVIDENCE / "leases.json").read_text())
    covered = []
    for lease in leases:
        assert lease["shared_lock_held_through_children"] is True
        assert lease["result"]["exit_code"] == 0
        assert {s["phase"] for s in lease["statuses"]} == {"before", "after"}
        assert all(s["healthy"] and s["idle"] for s in lease["statuses"])
        for name in lease["batches"]:
            assert index[name]["same_card_group"] == lease["same_card_group"]
            covered.append(name)
    assert len(covered) == len(set(covered)) and set(covered) == set(index)
