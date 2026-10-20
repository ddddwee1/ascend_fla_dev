"""Offline A5 GDN-2 quality evaluation for the protocol in issue #146.

Inputs are local, pretokenized, checksummed files. Each A/B/C backend runs in
its own process. Pilot outputs cannot be used for formal noninferiority tests.
No dataset, tokenizer, credential or network access occurs in this entry point.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
import bench_gdn2_chunk_fwd_a5 as bench

SCHEMA = "ascend-fla.gdn2-quality/1"
GROUPS = {"A": ("torch", torch.float32), "B": ("torch", torch.bfloat16),
          "C": ("chunk", torch.bfloat16)}
PLAN = {"contexts": [1024, 4096], "stride": 512, "decode_steps": 64,
        "stateful_windows": 16, "capture_steps": [0, 1, 32, 64],
        "ppl_block_windows": 8, "bootstrap_replicates": 10000,
        "bootstrap_seed": 14620261020, "confidence": 0.95,
        "ppl_tolerance": 0.01, "acc_norm_tolerance": 0.005}
MODEL_FILES = ("ascend_fla/models/gdn2.py", "ascend_fla/layers/gdn2.py",
               "ascend_fla/reference/gdn2.py", "ascend_fla/ops/gdn2_chunk_fwd.py",
               "ascend_fla/ops/gdn2/fused_recurrent.py",
               "kernels/projects/a5/gdn2_chunk_fwd/kernels/stages.py",
               "kernels/projects/a5/gdn2_chunk_fwd_bf16/kernels/stages.py",
               "kernels/projects/a5/gdn2_fused_recurrent/kernels/step.py")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def windows(length, context, stride):
    """Yield [begin,end) inputs and the first global target, counting each once."""
    if length < 2 or context < 2 or not 1 <= stride < context:
        raise ValueError("need length/context>=2 and 1<=stride<context")
    previous_end, end = 1, min(context, length)
    while True:
        begin = max(0, end - context)
        yield begin, end, max(previous_end, begin + 1)
        if end == length:
            break
        previous_end, end = end, min(length, end + stride)


def sample_indices(length, count):
    if min(length, count) < 1:
        raise ValueError("sampling needs nonempty inputs")
    count = min(length, count)
    return [0] if count == 1 else [i * (length - 1) // (count - 1) for i in range(count)]


def token_nll(logits, targets):
    """Score aligned next-token rows in FP32, then accumulate on CPU in FP64."""
    targets = torch.as_tensor(targets, dtype=torch.int64, device=logits.device).reshape(-1)
    logits = logits.reshape(-1, logits.shape[-1])
    if not targets.numel() or logits.shape[0] != targets.numel():
        raise ValueError("one logit row is required for every scored target")
    if bool(((targets < 0) | (targets >= logits.shape[-1])).any()):
        raise ValueError("target outside vocabulary")
    values = F.cross_entropy(logits.float(), targets, reduction="none").detach().cpu().double()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite token likelihood")
    return {"sum_nll": float(values.sum()), "n_tokens": values.numel()}


def check_cache(cache, layers, offset, dtype=None, finite=True, heads=None):
    if not isinstance(cache, (list, tuple)) or len(cache) != layers:
        raise ValueError("wrong cache layer count")
    tensors = []
    for entry in cache:
        if not isinstance(entry, Mapping) or type(entry.get("offset")) is not int:
            raise ValueError("cache must be a Mapping with an integer offset")
        if entry["offset"] != offset:
            raise ValueError(f"cache offset must be {offset}, got {entry['offset']}")
        state = entry.get("recurrent_state")
        if (not isinstance(state, torch.Tensor) or state.dtype != torch.float32 or
                state.ndim != 4 or state.shape[0] != 1 or tuple(state.shape[-2:]) != (128, 128)
                or (heads is not None and state.shape[1] != heads)):
            raise ValueError("recurrent cache must be FP32 [1,H,128,128]")
        conv = entry.get("conv_state")
        conv = [conv] if isinstance(conv, torch.Tensor) else conv
        if not isinstance(conv, (tuple, list)) or len(conv) not in (1, 3):
            raise ValueError("convolution cache is missing")
        for value in conv:
            if (not isinstance(value, torch.Tensor) or value.ndim != 3 or
                    value.shape[0] != 1 or value.shape[-1] != 4 or
                    (heads is not None and value.shape[1] != heads * 128 * (3 if len(conv) == 1 else 1)) or
                    (dtype is not None and value.dtype != dtype)):
                raise ValueError("wrong convolution cache shape/dtype")
        tensors.extend([state, *conv])
    if finite and not bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all().cpu()):
        raise ValueError("nonfinite cache")


def ppl_example(runner, tokens, window):
    begin, end, target = window
    logits, _ = runner.forward(tokens[begin:end])
    local = target - begin
    return {"begin": begin, "end": end, "target_start": target,
            **token_nll(logits[:, local - 1:end - begin - 1], tokens[target:end])}


def choice_result(scores, lengths, gold):
    if (len(scores) != 4 or len(lengths) != 4 or type(gold) is not int or gold not in range(4)
            or not all(math.isfinite(s) for s in scores)
            or not all(type(n) is int and n > 0 for n in lengths)):
        raise ValueError("need four finite scores, positive character lengths and a valid label")
    norm = [score / length for score, length in zip(scores, lengths)]
    # Python max, like NumPy argmax, selects the first entry of an exact tie.
    pred = max(range(4), key=scores.__getitem__)
    pred_norm = max(range(4), key=norm.__getitem__)
    return {"loglikelihoods": scores, "normalization_lengths": lengths, "gold": gold,
            "prediction": pred, "prediction_norm": pred_norm,
            "acc": int(pred == gold), "acc_norm": int(pred_norm == gold)}


def hellaswag_example(runner, example, context):
    prefix, choices = example["context_ids"], example["choices"]
    if not prefix or len(choices) != 4 or any(not c["ids"] for c in choices):
        raise ValueError("empty context/choice or wrong choice count")
    if any(len(c["ids"]) > context for c in choices):
        raise ValueError("continuation exceeds the scoring context")
    shared = len(prefix) + max(len(c["ids"]) for c in choices) - 1 <= context
    scores = []
    if shared:
        logits, cache = runner.forward(prefix)
        first = logits[:, -1:]
        for choice in choices:
            ids = choice["ids"]
            loss = token_nll(first, ids[:1])["sum_nll"]
            if len(ids) > 1:
                tail, _ = runner.forward(ids[:-1], cache)
                loss += token_nll(tail, ids[1:])["sum_nll"]
                check_cache(cache, runner.layers, len(prefix), finite=False)
            scores.append(-loss)
    else:
        for choice in choices:
            ids = choice["ids"]
            sequence = (prefix + ids)[-(context + 1):]
            logits, _ = runner.forward(sequence[:-1])
            scores.append(-token_nll(logits[:, -len(ids):], ids)["sum_nll"])
    return {"shared_prefix": shared, **choice_result(
        scores, [c["normalization_length"] for c in choices], example["gold"])}


def stateful_example(runner, tokens, decode_steps, tag, capture_steps):
    prefix = len(tokens) - decode_steps - 1
    if prefix < 1:
        raise ValueError("stateful example needs a prefix plus decode_steps+1 tokens")
    logits, cache = runner.forward(tokens[:prefix])
    runner.capture(tag, 0, logits, cache)
    total = 0.0
    for step in range(1, decode_steps + 1):
        index = prefix + step - 1
        logits, cache = runner.forward(tokens[index:index + 1], cache)
        total += token_nll(logits[:, -1:], tokens[index + 1:index + 2])["sum_nll"]
        if step in capture_steps:
            runner.capture(tag, step, logits, cache)
    return {"prefix_tokens": prefix, "decode_steps": decode_steps,
            "final_offset": len(tokens) - 1, "sum_nll": total, "n_tokens": decode_steps}


class ModelRunner:
    def __init__(self, model, dtype, device, output):
        self.model, self.dtype, self.device, self.output = model, dtype, device, Path(output)
        self.layers, self.calls = model.config.n_layer, 0
        self.heads = model.config.num_heads

    def forward(self, ids, cache=None):
        ids = torch.as_tensor(ids, dtype=torch.int64).reshape(1, -1).to(self.device)
        offset = 0 if cache is None else cache[0]["offset"]
        if cache is not None:
            check_cache(cache, self.layers, offset, self.dtype, finite=False, heads=self.heads)
        logits, updated = self.model(ids, cache=cache, return_cache=True)
        if logits.shape[:2] != ids.shape:
            raise ValueError("model output and input lengths disagree")
        if logits.dtype != self.dtype or not bool(torch.isfinite(logits).all().cpu()):
            raise ValueError("nonfinite logits or wrong model output dtype")
        check_cache(updated, self.layers, offset + ids.shape[1], self.dtype, heads=self.heads)
        self.calls += 1
        return logits, updated

    def capture(self, tag, step, logits, cache):
        path = self.output / "captures" / f"{tag}-step{step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"logits": bench.snapshot(logits[:, -1:]),
                    "cache": bench.checked_cache(cache, self.layers)}, temporary)
        temporary.replace(path)


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate evaluation record")
    return rows


def paired_rows(reference, candidate):
    left, right = ({r["id"]: r for r in rows} for rows in (reference, candidate))
    if len(left) != len(reference) or len(right) != len(candidate) or left.keys() != right.keys() or not left:
        raise ValueError("paired evaluation IDs differ, repeat or are empty")
    return [(left[key], right[key]) for key in left]


def bootstrap(reference, candidate, metric, tolerance, plan=PLAN, block_size=1):
    pairs = paired_rows(reference, candidate)
    if metric == "ppl":
        if any(a["n_tokens"] != b["n_tokens"] or a["n_tokens"] < 1 for a, b in pairs):
            raise ValueError("paired target counts differ")
        values = [(b["sum_nll"] - a["sum_nll"], a["n_tokens"]) for a, b in pairs]
    elif metric in ("acc", "acc_norm"):
        if any(a["gold"] != b["gold"] or a[metric] not in (0, 1) or b[metric] not in (0, 1)
               for a, b in pairs):
            raise ValueError("paired labels/correctness disagree")
        values = [(a[metric] - b[metric], 1) for a, b in pairs]
    else:
        raise ValueError("unknown bootstrap metric")
    grouped = [tuple(sum(v[j] for v in values[i:i + block_size]) for j in range(2))
               for i in range(0, len(values), block_size)]
    tensor = torch.tensor(grouped, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("nonfinite paired scores")
    transform = torch.expm1 if metric == "ppl" else lambda x: x
    point = float(transform(tensor[:, 0].sum() / tensor[:, 1].sum()))
    generator = torch.Generator(device="cpu").manual_seed(plan["bootstrap_seed"])
    samples = []
    for start in range(0, plan["bootstrap_replicates"], 128):
        size = min(128, plan["bootstrap_replicates"] - start)
        index = torch.randint(len(grouped), (size, len(grouped)), generator=generator, device="cpu")
        selected = tensor[index].sum(1)
        samples.append(transform(selected[:, 0] / selected[:, 1]))
    samples = torch.cat(samples)
    upper = float(torch.quantile(samples, plan["confidence"], interpolation="higher"))
    lower = float(torch.quantile(samples, 1 - plan["confidence"], interpolation="lower"))
    # A single bootstrap unit cannot establish a sampling confidence bound.
    verdict = ("inconclusive" if len(grouped) < 2 else
               "pass" if upper <= tolerance and point <= tolerance else
               "fail" if lower > tolerance else "inconclusive")
    return {"metric": metric, "degradation": point, "upper95_one_sided": upper,
            "lower95_one_sided": lower, "tolerance": tolerance, "verdict": verdict,
            "bootstrap_units": len(grouped), "paired_records": len(pairs),
            "bootstrap_replicates": plan["bootstrap_replicates"], "seed": plan["bootstrap_seed"]}


def package_digest():
    import ascriptor
    root = Path(ascriptor.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def load_inputs(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if (data["schema"] != SCHEMA or data["plan"] != PLAN or
            data["checkpoint_sha256"] != bench.CHECKPOINT_SHA256 or
            data["tokenizer_revision"] != "ff3c701f2424c7625fdefb9dd470f45ef18b02d6"):
        raise ValueError("manifest is outside the approved frozen protocol")
    files = {}
    for kind in ("wikitext", "hellaswag"):
        record = data["files"][kind]
        file = (path.parent / record["file"]).resolve()
        if not file.is_relative_to(path.parent) or bench.sha256(file) != record["sha256"]:
            raise ValueError("local data checksum or relative path is invalid")
        files[kind] = file
    tokens = torch.load(files["wikitext"], map_location="cpu", weights_only=True)
    if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or tokens.ndim != 1
            or tokens.numel() < 2 or not bool(((tokens >= 0) & (tokens < 32000)).all())):
        raise ValueError("invalid WikiText token stream")
    examples = read_rows(files["hellaswag"])
    if len(examples) != data["counts"]["hellaswag"] or tokens.numel() != data["counts"]["wikitext_tokens"]:
        raise ValueError("manifest input counts differ")
    for row in examples:
        ids = [*row["context_ids"], *(i for c in row["choices"] for i in c["ids"])]
        if not ids or any(type(i) is not int or i < 0 or i >= 32000 for i in ids):
            raise ValueError("invalid HellaSwag token IDs")
    return data, tokens, examples


def execute(args):
    import torch_npu  # noqa: F401
    from ascend_fla.models.gdn2 import GDN2ForCausalLM
    # FP32 is a full-precision control; do not silently enable reduced-mantissa math.
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    precision = {"matmul_allow_hf32": torch.npu.matmul.allow_hf32,
                 "conv_allow_hf32": torch.npu.conv.allow_hf32,
                 "allow_internal_format": torch.npu.config.allow_internal_format}
    data, tokens, examples = load_inputs(args.manifest)
    if bench.sha256(args.checkpoint) != bench.CHECKPOINT_SHA256:
        raise ValueError("wrong checkpoint")
    if package_digest() != data["ascriptor_package_sha256"]:
        raise ValueError("Ascriptor package differs from the frozen pin")
    backend, dtype = GROUPS[args.group]
    environment = bench.runtime_metadata()
    if environment != {**data["expected_environment"],
                       "benchmark_sha256": environment["benchmark_sha256"]}:
        raise ValueError("device/software identity differs from the preflight")
    if backend == "chunk":
        from ascend_fla.ops.gdn2_chunk_fwd import prepare
        from ascend_fla.ops.gdn2.fused_recurrent import prepare as prepare_decode
        prepare(block_dim=8, dtypes=(torch.bfloat16,))
        prepare_decode()
    model = GDN2ForCausalLM.from_checkpoint(
        args.checkpoint, device="npu", dtype=dtype, core_backend="torch",
        projection_layout="canonical", strict=True, mmap=True, weights_only=True)
    calls = {"chunk_prefill": 0, "cce_decode": 0}
    if backend == "chunk":
        bench.install_chunk_core(model, 8, calls)
    identity = {"schema": SCHEMA, "group": args.group, "backend": backend,
                "dtype": str(dtype), "manifest_sha256": bench.sha256(args.manifest),
                "evaluator_sha256": bench.sha256(__file__), "environment": environment,
                "model_sources": {p: bench.sha256(ROOT / p) for p in sorted(set(MODEL_FILES) | {
                    str(p.relative_to(ROOT)) for p in (ROOT / "ascend_fla").rglob("*.py")})},
                "precision": precision,
                "checkpoint_sha256": bench.CHECKPOINT_SHA256, "pilot_items": args.limit}
    args.output.mkdir(parents=True, exist_ok=True)
    header = args.output / "identity.json"
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("output directory belongs to a different run")
    write_json(header, identity)
    runner = ModelRunner(model, dtype, "npu", args.output)
    started = time.perf_counter()
    reports, expected = {}, {}

    def run_part(name, items, evaluate):
        expected[name] = len(items)
        file = args.output / f"{name}.jsonl"
        rows = read_rows(file)
        done = {row["id"] for row in rows}
        requested = items if args.limit is None else items[:args.limit]
        if not done.issubset({key for key, _ in requested}):
            raise ValueError("unexpected saved evaluation IDs")
        with file.open("a") as stream:
            for key, item in requested:
                if key in done:
                    continue
                before = time.perf_counter()
                before_forwards, before_core = runner.calls, dict(calls)
                record = {"id": key, **evaluate(item),
                          "wall_seconds": time.perf_counter() - before,
                          "forward_calls": runner.calls - before_forwards,
                          "core_calls": {k: v - before_core[k] for k, v in calls.items()}}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                rows.append(record)
                if len(rows) % 10 == 0 or len(rows) == len(requested):
                    progress = {"group": args.group, "part": name, "done": len(rows),
                                "total": len(requested), "formal_total": len(items),
                                "elapsed_seconds": time.perf_counter() - started,
                                "last_wall_seconds": record["wall_seconds"]}
                    write_json(args.output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        reports[name] = {"records": len(rows), "expected": len(items), "sha256": bench.sha256(file),
                         "forward_calls": sum(r["forward_calls"] for r in rows),
                         "core_calls": {k: sum(r["core_calls"][k] for r in rows) for k in calls}}
        if rows and "sum_nll" in rows[0]:
            count = sum(r["n_tokens"] for r in rows)
            nll = sum(r["sum_nll"] for r in rows)
            reports[name].update(n_tokens=count, sum_nll=nll, nll_per_token=nll / count,
                                 perplexity=math.exp(nll / count))
        elif rows:
            reports[name].update({k: sum(r[k] for r in rows) / len(rows) for k in ("acc", "acc_norm")})

    with torch.inference_mode():
        for context in PLAN["contexts"]:
            grid = list(windows(len(tokens), context, PLAN["stride"]))
            run_part(f"ppl-{context}", [(str(i), w) for i, w in enumerate(grid)],
                     lambda w: ppl_example(runner, tokens, w))
            selected = sample_indices(len(grid), PLAN["stateful_windows"])
            run_part(f"stateful-{context}", [(str(i), i) for i in selected],
                     lambda i, context=context, grid=grid: stateful_example(
                         runner, tokens[grid[i][0]:grid[i][1]], PLAN["decode_steps"],
                         f"stateful-{context}-{i}", PLAN["capture_steps"]))
        run_part("hellaswag", [(str(e["id"]), e) for e in examples],
                 lambda e: hellaswag_example(runner, e, max(PLAN["contexts"])))
    if backend == "chunk" and sum(calls.values()) != runner.calls * runner.layers:
        raise ValueError("not every candidate layer used the requested custom operators")
    complete = args.limit is None and all(r["records"] == r["expected"] for r in reports.values())
    if complete and any(reports[f"ppl-{c}"]["n_tokens"] != len(tokens) - 1 for c in PLAN["contexts"]):
        raise ValueError("sliding windows did not score every eligible target exactly once")
    all_forwards = sum(r["forward_calls"] for r in reports.values())
    all_core = {k: sum(r["core_calls"][k] for r in reports.values()) for k in calls}
    if backend == "chunk" and sum(all_core.values()) != all_forwards * runner.layers:
        raise ValueError("saved candidate records do not demonstrate the required custom route")
    result = {**identity, "complete": complete, "parts": reports,
              "all_forward_calls": all_forwards, "all_core_calls": all_core,
              "this_process_forward_calls": runner.calls, "this_process_core_calls": calls,
              "elapsed_seconds": time.perf_counter() - started,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "summary.json", result)
    return result


def panel_comparison(reference, candidate, expected=None):
    reference, candidate = Path(reference), Path(candidate)
    left = {p.name: p for p in (reference / "captures").glob("*.pt")}
    right = {p.name: p for p in (candidate / "captures").glob("*.pt")}
    if not left or left.keys() != right.keys():
        raise ValueError("diagnostic panel captures differ or are missing")
    if expected is not None and left.keys() != expected:
        raise ValueError("diagnostic panel does not cover the frozen window/step grid")
    worst, finite, scalar_count, tensor_count = {}, True, 0, 0
    for name in sorted(left):
        a, b = (dict(bench.flatten(torch.load(p, map_location="cpu", weights_only=True)))
                for p in (left[name], right[name]))
        if a.keys() != b.keys():
            raise ValueError("diagnostic cache paths differ")
        for key, want in a.items():
            got = b[key]
            if isinstance(want, torch.Tensor):
                if not isinstance(got, torch.Tensor):
                    raise ValueError("diagnostic tensor disappeared")
                row = bench.diff(got, want)
                row.update(reference_dtype=str(want.dtype), candidate_dtype=str(got.dtype),
                           capture=name, path=key)
                kind = "logits" if key.startswith("logits") else "recurrent_state" if "recurrent_state" in key else "conv_state"
                if kind not in worst or (row["relative_l2"] or 0) > (worst[kind]["relative_l2"] or 0):
                    worst[kind] = row
                finite = finite and row["finite"]
                tensor_count += 1
            else:
                if type(got) is not type(want) or got != want:
                    raise ValueError("cache offset/scalar metadata differs")
                scalar_count += 1
    return {"finite": finite, "captures": len(left), "tensor_comparisons": tensor_count,
            "equal_scalars": scalar_count, "worst_relative_l2": worst,
            "role": "diagnostic tensor distances; quality verdict uses predeclared task metrics"}


def compare(args):
    _, tokens, examples = load_inputs(args.manifest)
    expected = {}
    captures = set()
    for context in PLAN["contexts"]:
        count = len(list(windows(len(tokens), context, PLAN["stride"])))
        expected[f"ppl-{context}"] = {str(i) for i in range(count)}
        expected[f"stateful-{context}"] = {str(i) for i in sample_indices(count, PLAN["stateful_windows"])}
        captures.update(f"stateful-{context}-{i}-step{step}.pt"
                        for i in expected[f"stateful-{context}"] for step in PLAN["capture_steps"])
    expected["hellaswag"] = {str(e["id"]) for e in examples}
    directories = {group: args.output / group for group in GROUPS}
    summaries = {group: json.loads((path / "summary.json").read_text()) for group, path in directories.items()}
    for group, record in summaries.items():
        if not record["complete"] or record["pilot_items"] is not None or record["group"] != group:
            raise ValueError("formal comparison requires three complete non-pilot groups")
        if (record["manifest_sha256"] != bench.sha256(args.manifest) or
                record["evaluator_sha256"] != bench.sha256(__file__) or record["parts"].keys() != expected.keys()):
            raise ValueError("completed run does not cover this manifest/evaluator and every required part")
        for key in ("schema", "manifest_sha256", "evaluator_sha256", "environment", "model_sources", "checkpoint_sha256", "precision"):
            if record[key] != summaries["A"][key]:
                raise ValueError(f"paired run identity differs: {key}")
        if record["dtype"] != str(GROUPS[group][1]) or record["backend"] != GROUPS[group][0]:
            raise ValueError("run has the wrong backend/dtype")
        for name, part in record["parts"].items():
            file = directories[group] / f"{name}.jsonl"
            rows = read_rows(file)
            if (bench.sha256(file) != part["sha256"] or {r["id"] for r in rows} != expected[name]
                    or part["expected"] != len(expected[name]) or part["records"] != part["expected"]):
                raise ValueError("completed result records are missing or changed")
            if name.startswith("ppl-") and sum(r["n_tokens"] for r in rows) != len(tokens) - 1:
                raise ValueError("completed PPL does not score every target once")
            if name.startswith("stateful-") and any(r["n_tokens"] != PLAN["decode_steps"] for r in rows):
                raise ValueError("completed stateful records have the wrong decode length")
    pairs = {}
    for reference, candidate in [("B", "C"), ("A", "B"), ("A", "C")]:
        name = reference + "_to_" + candidate
        metrics = {}
        for context in PLAN["contexts"]:
            for kind in ("ppl", "stateful"):
                file = f"{kind}-{context}.jsonl"
                metrics[file] = bootstrap(read_rows(directories[reference] / file),
                    read_rows(directories[candidate] / file), "ppl", PLAN["ppl_tolerance"],
                    block_size=PLAN["ppl_block_windows"] if kind == "ppl" else 1)
        metrics["hellaswag"] = bootstrap(read_rows(directories[reference] / "hellaswag.jsonl"),
            read_rows(directories[candidate] / "hellaswag.jsonl"), "acc_norm", PLAN["acc_norm_tolerance"])
        pairs[name] = {"role": "primary" if name == "B_to_C" else "dtype/total-change diagnostic",
                       "metrics": metrics,
                       "tensors": panel_comparison(directories[reference], directories[candidate], captures)}
    primary = pairs["B_to_C"]
    result = {"schema": SCHEMA, "plan": PLAN, "pairs": pairs,
              "passed": primary["tensors"]["finite"] and all(m["verdict"] == "pass" for m in primary["metrics"].values()),
              "note": "Existing BF16 logits/cache L2 receipts remain diagnostics; FP32 qualification is separate."}
    write_json(args.output / "comparison.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "compare"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--group", choices=tuple(GROUPS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="pilot-only item cap per part; never a formal pass")
    args = parser.parse_args(argv)
    if args.mode == "run" and (args.manifest is None or args.checkpoint is None or args.group is None):
        parser.error("run requires manifest, checkpoint and group")
    if args.mode == "compare" and args.manifest is None:
        parser.error("compare requires the frozen manifest")
    if args.limit is not None and args.limit < 1:
        parser.error("pilot limit must be positive")
    result = execute(args) if args.mode == "run" else compare(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
