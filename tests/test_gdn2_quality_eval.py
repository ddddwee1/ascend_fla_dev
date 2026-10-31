"""Independent score, masking, branching-cache and noninferiority checks."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "benchmarks/eval_gdn2_quality.py"
SPEC = importlib.util.spec_from_file_location("gdn2_quality_eval", PATH)
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


class NextTokenModel:
    """The exact next label is the input token plus one, with immutable caches."""
    config = SimpleNamespace(n_layer=1, num_heads=1)

    def __init__(self):
        self.lengths = []

    def __call__(self, ids, cache=None, return_cache=True):
        self.lengths.append(ids.shape[1])
        logits = torch.zeros(1, ids.shape[1], 32)
        logits.scatter_(-1, ((ids + 1) % 32).unsqueeze(-1), math.log(31))
        offset = (0 if cache is None else cache[0]["offset"]) + ids.shape[1]
        state = torch.zeros(1, 1, 128, 128)
        conv = tuple(torch.zeros(1, 128, 4) for _ in range(3))
        return logits, [{"offset": offset, "recurrent_state": state, "conv_state": conv}]


class RecordingRunner(quality.ModelRunner):
    def __init__(self):
        super().__init__(NextTokenModel(), torch.float32, "cpu", ".")
        self.captures = []

    def capture(self, tag, step, logits, cache):
        self.captures.append((step, cache[0]["offset"]))


@pytest.mark.parametrize("length,context,stride", [(2, 4, 2), (29, 8, 3), (33, 8, 4), (31, 16, 7)])
def test_sliding_targets_are_a_partition(length, context, stride):
    scored = []
    for begin, end, target in quality.windows(length, context, stride):
        assert 0 <= begin < target < end <= length
        assert end - begin <= context
        scored.extend(range(target, end))
    assert scored == list(range(1, length))


@pytest.mark.parametrize("args", [(1, 8, 4), (8, 1, 1), (8, 4, 0), (8, 4, 4)])
def test_invalid_window_protocol_rejected(args):
    with pytest.raises(ValueError):
        list(quality.windows(*args))


def test_manual_next_token_probability_and_masking():
    runner = RecordingRunner()
    tokens = torch.arange(29) % 32
    records = [quality.ppl_example(runner, tokens, w) for w in quality.windows(len(tokens), 8, 3)]
    # exp(log(31)) / (31 + 31) = 1/2 for every correct next token.
    count = sum(r["n_tokens"] for r in records)
    assert count == 28
    assert sum(r["sum_nll"] for r in records) == pytest.approx(28 * math.log(2), abs=1e-5)
    logits, _ = runner.forward(tokens[:8])
    wrong_shift = quality.token_nll(logits[:, :-1], tokens[:7])
    assert wrong_shift["sum_nll"] > 7 * 3


def test_stateful_scores_decode_outputs_against_the_next_true_token():
    runner = RecordingRunner()
    tokens = torch.arange(20) % 32
    record = quality.stateful_example(runner, tokens, 4, "test", [0, 1, 4])
    assert runner.model.lengths == [15, 1, 1, 1, 1]
    assert runner.captures == [(0, 15), (1, 16), (4, 19)]
    assert record["n_tokens"] == 4 and record["final_offset"] == 19
    assert record["sum_nll"] == pytest.approx(4 * math.log(2), abs=1e-5)


def example(prefix):
    return {"context_ids": prefix, "gold": 0,
            "choices": [{"ids": [3, 4], "normalization_length": 2},
                        {"ids": [4], "normalization_length": 1},
                        {"ids": [5, 6, 7], "normalization_length": 3},
                        {"ids": [6, 7], "normalization_length": 2}]}


def test_hellaswag_shared_prefix_and_truncation_have_correct_alignment():
    for context in [4, 16]:
        runner = RecordingRunner()
        result = quality.hellaswag_example(runner, example([0, 1, 2]), context)
        assert result["acc"] == result["acc_norm"] == 1
        assert result["loglikelihoods"][0] == pytest.approx(-2 * math.log(2), abs=1e-5)
        assert result["shared_prefix"] is (context == 16)
        assert max(runner.model.lengths) <= context


def test_character_length_normalization_can_change_the_winner():
    result = quality.choice_result([-1, -1.5, -4, -8], [1, 3, 1, 1], 1)
    assert result["prediction"] == 0
    assert result["prediction_norm"] == 1 and result["acc_norm"] == 1


@pytest.mark.parametrize("defect", ["nan", "label", "length", "empty"])
def test_invalid_choice_scores_rejected(defect):
    scores, lengths, gold = [-1., -2., -3., -4.], [1, 2, 3, 4], 0
    if defect == "nan": scores[0] = float("nan")
    if defect == "label": gold = -1
    if defect == "length": lengths[2] = 0
    if defect == "empty": scores = []
    with pytest.raises(ValueError): quality.choice_result(scores, lengths, gold)


@pytest.mark.parametrize("defect", ["offset", "state", "conv", "nan", "dtype", "heads", "channels"])
def test_cache_protocol_negative_controls(defect):
    _, cache = NextTokenModel()(torch.tensor([[0, 1, 2]]))
    if defect == "offset": cache[0]["offset"] = 2
    if defect == "state": cache[0]["recurrent_state"] = None
    if defect == "conv": cache[0]["conv_state"] = None
    if defect == "nan": cache[0]["recurrent_state"][0, 0, 0, 0] = float("nan")
    if defect == "dtype": cache[0]["conv_state"] = tuple(t.bfloat16() for t in cache[0]["conv_state"])
    if defect == "heads": cache[0]["recurrent_state"] = torch.zeros(1, 2, 128, 128)
    if defect == "channels": cache[0]["conv_state"] = tuple(t[:, :64] for t in cache[0]["conv_state"])
    with pytest.raises(ValueError): quality.check_cache(cache, 1, 3, torch.float32, heads=1)


def test_branching_does_not_advance_the_shared_prefix_cache():
    runner = RecordingRunner()
    _, cache = runner.forward([0, 1, 2])
    original = cache[0]["recurrent_state"].clone()
    first, one = runner.forward([3, 4], cache)
    second, two = runner.forward([3, 4], cache)
    assert cache[0]["offset"] == 3 and one[0]["offset"] == two[0]["offset"] == 5
    assert torch.equal(first, second)
    assert torch.equal(original, cache[0]["recurrent_state"])


def test_real_model_forked_choice_scores_match_independent_cold_forwards(tmp_path):
    from ascend_fla.models import GDN2Config, GDN2ForCausalLM

    config = GDN2Config(vocab_size=32, padding_multiple=8, block_size=32,
                       n_layer=2, n_embd=16, intermediate_size=24,
                       num_heads=1, head_dim=128, conv_size=4)
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(146)
        model = GDN2ForCausalLM(config).eval()
        runner = quality.ModelRunner(model, torch.float32, "cpu", tmp_path)
        item = example([0, 1, 2])
        cached = quality.hellaswag_example(runner, item, 32)
        cold_scores = []
        for choice in item["choices"]:
            ids = choice["ids"]
            logits = model(torch.tensor([item["context_ids"] + ids[:-1]]))
            logp = logits[0, -len(ids):].float().log_softmax(-1)
            selected = logp.gather(1, torch.tensor(ids).reshape(-1, 1))
            cold_scores.append(float(selected.double().sum()))
        assert cached["loglikelihoods"] == pytest.approx(cold_scores, abs=1e-5, rel=1e-6)
        reordered = {**item, "choices": list(reversed(item["choices"])), "gold": 3 - item["gold"]}
        reverse = quality.hellaswag_example(runner, reordered, 32)
        assert reverse["loglikelihoods"] == pytest.approx(list(reversed(cold_scores)), abs=1e-5, rel=1e-6)

        _, prefix_cache = runner.forward(item["context_ids"])
        before = {p: value.clone() if isinstance(value, torch.Tensor) else value
                  for p, value in quality.bench.flatten(prefix_cache)}
        runner.forward([3, 4], prefix_cache)
        runner.forward([5, 6, 7], prefix_cache)
        after = dict(quality.bench.flatten(prefix_cache))
        for path, value in before.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(after[path], value), path
            else:
                assert after[path] == value, path


def test_weighted_nll_and_noninferiority_detect_clear_regression():
    plan = {**quality.PLAN, "bootstrap_replicates": 500}
    baseline = [{"id": str(i), "sum_nll": float(n), "n_tokens": n} for i, n in enumerate([1, 2, 10, 20])]
    equal = quality.bootstrap(baseline, baseline, "ppl", .01, plan)
    assert equal["degradation"] == equal["upper95_one_sided"] == 0
    assert equal["verdict"] == "pass"
    candidate = [{**r, "sum_nll": 2 * r["sum_nll"]} for r in baseline]
    bad = quality.bootstrap(baseline, candidate, "ppl", .01, plan)
    assert bad["degradation"] == pytest.approx(math.e - 1)
    assert bad["verdict"] == "fail"


def test_sampling_uncertainty_is_not_a_pass():
    plan = {**quality.PLAN, "bootstrap_replicates": 1000}
    baseline = [{"id": str(i), "gold": 0, "acc_norm": 1} for i in range(20)]
    candidate = [{**r, "acc_norm": int(i != 0)} for i, r in enumerate(baseline)]
    result = quality.bootstrap(baseline, candidate, "acc_norm", .005, plan)
    assert result["verdict"] == "inconclusive"
    only_one = quality.bootstrap(baseline[:1], baseline[:1], "acc_norm", .005, plan)
    assert only_one["verdict"] == "inconclusive"


def test_mismatched_or_duplicate_paired_results_rejected():
    row = {"id": "x", "sum_nll": 1., "n_tokens": 2}
    for candidate in [[{**row, "id": "y"}], [{**row, "n_tokens": 3}], [row, row]]:
        with pytest.raises(ValueError): quality.bootstrap([row], candidate, "ppl", .01)


def test_sampling_panel_is_deterministic_and_spans_the_dataset():
    assert quality.sample_indices(100, 4) == [0, 33, 66, 99]
    assert quality.sample_indices(3, 16) == [0, 1, 2]


@pytest.mark.parametrize("pilot", [True, False])
def test_formal_comparison_rejects_pilots_and_missing_parts(tmp_path, monkeypatch, pilot):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(quality, "load_inputs", lambda _: ({}, torch.arange(100), [{"id": "x"}]))
    for group in quality.GROUPS:
        directory = tmp_path / group
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps({
            "complete": True, "pilot_items": 1 if pilot else None, "group": group,
            "manifest_sha256": quality.bench.sha256(manifest),
            "evaluator_sha256": quality.bench.sha256(quality.__file__), "parts": {}}))
    with pytest.raises(ValueError):
        quality.compare(SimpleNamespace(manifest=manifest, output=tmp_path))
