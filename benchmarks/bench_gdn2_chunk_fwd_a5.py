"""A5 GDN-2 chunk checks, model snapshots and synchronized performance sandwiches.

Use a separate process per block_dim or model backend. All custom operators are
prepared before the first launch. Snapshots contain CPU tensors, including the
model's Mapping caches, and can be compared without CANN or an NPU.

Operator validation and timing: --shapes real --sandwich
Cross-bd checks: --check-only --seq-len 65 --block-dim 1
Model snapshots: --mode model --backend torch|chunk --checkpoint FILE --dump FILE
Compare snapshots: --mode compare --reference FILE --candidate FILE --json FILE

Model runs use canonical weights with the same inference dtype on both sides.
The chunk backend explicitly runs chunk prefill followed by CCE recurrent decode.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
QK_EPS, Q_SCALE = 1e-6, 128 ** -0.5
CHECKPOINT_SHA256 = "4ac729c627febc431bf4b2011a9cb7ad9c30cc1d017ff74aeca15f76efb2df6d"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(value):
    """Copy complete tensors before slicing; recognize typed Mapping caches."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, Mapping):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    if value is None or type(value) in (int, float, bool, str):
        return value
    raise TypeError(f"unsupported snapshot value: {type(value).__name__}")


def flatten(value, path=()):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from flatten(item, path + (str(key),))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from flatten(item, path + (str(index),))
    else:
        yield "/".join(path), value


def diff(got, reference):
    if got.shape != reference.shape:
        raise ValueError(f"shape mismatch: {tuple(got.shape)} vs {tuple(reference.shape)}")
    a, b = got.detach().cpu().double(), reference.detach().cpu().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    residual = a - b
    return {"shape": list(a.shape), "dtype": str(got.dtype), "finite": finite,
            "max_abs_diff": float(residual.abs().max()) if finite else None,
            "relative_l2": float(residual.norm() / b.norm().clamp_min(1e-30)) if finite else None,
            "bitwise_equal": got.dtype == reference.dtype and torch.equal(got.cpu(), reference.cpu())}


def compare_trees(got, reference, budget):
    actual, expected = dict(flatten(got)), dict(flatten(reference))
    if actual.keys() != expected.keys():
        raise ValueError(f"snapshot paths differ: missing={sorted(expected.keys()-actual.keys())}, "
                         f"extra={sorted(actual.keys()-expected.keys())}")
    rows, scalar_paths = {}, []
    for path, want in expected.items():
        value = actual[path]
        if isinstance(want, torch.Tensor):
            if not isinstance(value, torch.Tensor) or value.dtype != want.dtype:
                raise ValueError(f"tensor type/dtype mismatch at {path}")
            rows[path] = diff(value, want)
        elif type(value) is not type(want) or value != want:
            raise ValueError(f"metadata differs at {path}")
        else:
            scalar_paths.append(path)
    if not rows:
        raise ValueError("comparison contains no tensors")
    passed = all(row["finite"] and row["relative_l2"] <= budget for row in rows.values())
    return {"passed": passed, "budget": budget, "tensors": rows, "equal_scalar_paths": scalar_paths}


def compare_operator(got, reference, dtype):
    """Apply the unit's operator tolerances, separately from model budgets."""
    unit = "gdn2_chunk_fwd" + ("_bf16" if dtype == "bfloat16" else "")
    contract_path = ROOT / "kernels/projects/a5" / unit / "contract.json"
    comparison = json.loads(contract_path.read_text())["comparison"]
    rules = [dict(comparison["default"], **comparison.get("outputs", {}).get(name, {}))
             for name in ("o", "final_state")]
    result = compare_trees(got, reference, max(r["max_relative_l2"] for r in rules))
    for index, rule in enumerate(rules):
        if rule["mode"] != "allclose":
            raise ValueError("operator checks require the declared allclose contract")
        row = result["tensors"][str(index)]
        row["comparison"] = rule
        row["allclose"] = bool(torch.allclose(got[index].double(), reference[index].double(),
                                             atol=rule["atol"], rtol=rule["rtol"]))
        row["passed"] = (row["finite"] and row["allclose"] and
                         row["relative_l2"] <= rule["max_relative_l2"])
    result["passed"] = all(row["passed"] for row in result["tensors"].values())
    result["contract_sha256"] = sha256(contract_path)
    return result


def tensor_digest(tensor):
    return hashlib.sha256(snapshot(tensor).view(torch.uint8).numpy().tobytes()).hexdigest()


def make_inputs(dtype, seq_len, heads, seed, device="npu"):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, seq_len, heads, 128)
    q, k, v = ((torch.randn(shape, generator=generator) * 0.5).to(dtype) for _ in range(3))
    gate = -torch.rand(shape, generator=generator) * 3.0
    erase = torch.rand(shape, generator=generator)
    write = torch.rand(shape, generator=generator)
    state = torch.randn(1, heads, 128, 128, generator=generator) * 0.1
    return tuple(t.to(device) for t in (q, k, v, gate, erase, write, state))


def torch_npu_composed(q, k, v, gate, erase_gate, w, state):
    """Naive token-at-a-time native-operator baseline, with FP32 recurrence."""
    q32, k32, v32 = q.float(), k.float(), v.float()
    qn = q32 * torch.rsqrt(q32.square().sum(-1, keepdim=True) + QK_EPS) * Q_SCALE
    kn = k32 * torch.rsqrt(k32.square().sum(-1, keepdim=True) + QK_EPS)
    current, outputs = state.clone(), []
    for index in range(q.shape[1]):
        k_t = kn[:, index]
        current = current * torch.exp(gate[:, index]).unsqueeze(-1)
        erase = torch.matmul((erase_gate[:, index] * k_t).unsqueeze(-2), current).squeeze(-2)
        delta = w[:, index] * v32[:, index] - erase
        current = current + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(torch.matmul(qn[:, index].unsqueeze(-2), current).squeeze(-2))
    return torch.stack(outputs, dim=1).to(q.dtype), current


def timed(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    ordered = sorted(samples)
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "min_ms": ordered[0], "p90_ms": ordered[max(0, int(0.9 * len(ordered)) - 1)],
            "repeat": repeats, "warmup": warmup, "synchronize_each_iteration": True,
            "peak_extra_allocated_bytes_including_outputs":
                max(0, torch.npu.max_memory_allocated() - before)}


def runtime_metadata():
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("CANN_PATH")
    if not home:
        raise ValueError("set the CANN environment before collecting device evidence")
    opp = Path(os.environ.get("ASCEND_OPP_PATH", str(Path(home) / "opp")))

    def identity(path):
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        if not all(values.get(key) for key in ("Version", "timestamp")):
            raise ValueError("CANN compiler and OPP identities require Version and timestamp")
        return {key: values[key] for key in ("Version", "timestamp")}

    return {"soc": torch.npu.get_device_name(0), "python": platform.python_version(),
            "torch": torch.__version__, "torch_npu": version("torch-npu"),
            "ascriptor": version("ascriptor"), "benchmark_sha256": sha256(__file__),
            "cann_compiler": identity(Path(home) / "compiler/version.info"),
            "cann_opp": identity(opp / "version.info"),
            "opp_packages": sorted(p.name for p in (opp / "built-in/op_impl/ai_core/tbe/kernel").iterdir()
                                   if p.is_dir() and p.name.startswith("ascend"))}


def run_operator(args):
    from ascend_fla.ops.gdn2_chunk_fwd import chunk_gdn2, prepare
    from ascend_fla.reference.gdn2 import gdn2_recurrent_reference

    environment = runtime_metadata()
    prepare(block_dim=args.block_dim, dtypes=(DTYPES[args.dtype],))
    cases = []
    for length in ([1024, 4096] if args.shapes else [args.seq_len]):
        tensors = make_inputs(DTYPES[args.dtype], length, args.heads, args.seed)
        q, k, v, gate, erase, write, state = tensors

        def candidate():
            return chunk_gdn2(q, k, v, gate, erase, write, initial_state=state,
                              output_final_state=True, block_dim=args.block_dim)

        cpu = tuple(snapshot(t) for t in tensors)
        cpu_golden = gdn2_recurrent_reference(*cpu, scale=Q_SCALE,
                                             use_qk_l2norm=True, qk_norm_eps=QK_EPS)
        native_golden = torch_npu_composed(*tensors)
        actual = candidate()
        checks = {name: compare_operator(snapshot(actual), snapshot(want), args.dtype)
                  for name, want in [("cpu_fp32_recurrence", cpu_golden),
                                     ("torch_npu_composition", native_golden)]}
        checks["oracle_agreement"] = compare_operator(snapshot(native_golden), cpu_golden, args.dtype)
        row = {"seq_len": length, "heads": args.heads, "dtype": args.dtype,
               "block_dim": args.block_dim, "checks": checks,
               "output_sha256": tensor_digest(actual[0]),
               "final_state_sha256": tensor_digest(actual[1]), "sandwiches": []}
        if not all(check["passed"] for check in checks.values()):
            row["passed"] = False
            cases.append(row)
            continue
        del actual, native_golden, cpu_golden, cpu
        if not args.check_only:
            for number in range(args.sandwiches):
                before = timed(lambda: torch_npu_composed(*tensors), args.reps, args.warmup)
                measured = timed(candidate, args.reps, args.warmup)
                after = timed(lambda: torch_npu_composed(*tensors), args.reps, args.warmup)
                baseline = min(before["median_ms"], after["median_ms"])
                row["sandwiches"].append({
                    "number": number + 1, "baseline_before": before, "candidate": measured,
                    "baseline_after": after, "ratio_candidate_over_baseline": measured["median_ms"] / baseline,
                    "baseline_reference": "faster adjacent baseline median",
                    "baseline_drift": abs(before["median_ms"] - after["median_ms"]) / baseline})
        row["passed"] = True
        row["performance_passed"] = (None if args.check_only else
            all(s["ratio_candidate_over_baseline"] < 1 for s in row["sandwiches"]))
        from ascend_fla.ops.gdn2_chunk_fwd import _compiled
        row["retained_aclnn_workspace_capacity_bytes"] = {
            compiled.op.op_name: sum(t.numel() * t.element_size()
                                     for t in compiled.op._ws_cache.values())
            for compiled in _compiled(args.block_dim, DTYPES[args.dtype])}
        cases.append(row)
        print(json.dumps(row), flush=True)
    return {"mode": "operator", "environment": environment, "cases": cases,
            "baseline_identity": "naive token-at-a-time torch_npu composition on identical inputs",
            "passed": all(c["passed"] and c.get("performance_passed") is not False for c in cases)}


def install_chunk_core(model, block_dim, calls):
    """Test-only adapter; production routing and reserved sources stay intact.

    Materializing model views and widening already rounded gates are explicit
    harness operations. Operator sandwiches time the public operator directly,
    excluding this model adapter.
    """
    from ascend_fla.ops.gdn2_chunk_fwd import chunk_gdn2
    from ascend_fla.ops.gdn2 import fused_recurrent_gdn2

    def core(layer, q, k, v, g, b, w, initial_state):
        q, k, v, g, b, w = (t.contiguous() for t in (q, k, v, g, b, w))
        if q.shape[1] > 1:
            calls["chunk_prefill"] += 1
            return chunk_gdn2(q, k, v, g, b.float(), w.float(), initial_state=initial_state,
                              output_final_state=True, block_dim=block_dim)
        calls["cce_decode"] += 1
        return fused_recurrent_gdn2(q, k, v, g, b, w, initial_state=initial_state,
                                   output_final_state=True, block_dim=8)

    for block in model.transformer.h:
        block.attn._run_core = MethodType(core, block.attn)


def checked_cache(cache, layers):
    if len(cache) != layers:
        raise ValueError("wrong number of cache layers")
    for entry in cache:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("recurrent_state"), torch.Tensor):
            raise ValueError("cache is missing recurrent-state tensors")
        state = entry["recurrent_state"]
        if state.dtype != torch.float32 or state.ndim != 4 or tuple(state.shape[-2:]) != (128, 128):
            raise ValueError("recurrent cache must be FP32 [B,H,128,128]")
        if type(entry.get("offset")) is not int or entry["offset"] < 0:
            raise ValueError("cache offset must be a nonnegative integer")
        conv = entry.get("conv_state")
        if not isinstance(conv, torch.Tensor) and not (
            isinstance(conv, (tuple, list)) and len(conv) == 3 and
            all(isinstance(t, torch.Tensor) for t in conv)
        ):
            raise ValueError("cache is missing convolution-state tensors")
    return snapshot(cache)


def run_model(args):
    if args.checkpoint is None or args.dump is None:
        raise ValueError("model mode requires --checkpoint and --dump")
    if args.model_tokens < 2:
        raise ValueError("model prefill requires at least two tokens to exercise chunk")
    environment = runtime_metadata()
    checkpoint_digest = sha256(args.checkpoint)
    if checkpoint_digest != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 does not match models.json")
    from ascend_fla.models.gdn2 import GDN2ForCausalLM
    if args.backend == "chunk":
        from ascend_fla.ops.gdn2_chunk_fwd import prepare
        from ascend_fla.ops.gdn2.fused_recurrent import prepare as prepare_decode
        prepare(block_dim=args.block_dim, dtypes=(DTYPES[args.dtype],))
        prepare_decode()
    model = GDN2ForCausalLM.from_checkpoint(
        args.checkpoint, device="npu", dtype=DTYPES[args.dtype], core_backend="torch",
        projection_layout="canonical", strict=True, mmap=True, weights_only=True)
    calls = {"chunk_prefill": 0, "cce_decode": 0}
    if args.backend == "chunk":
        install_chunk_core(model, args.block_dim, calls)
    layers = model.config.n_layer
    ids = (torch.arange(args.model_tokens, dtype=torch.int64).reshape(1, -1) + 1)
    with torch.inference_mode():
        logits, cache = model(ids.to("npu"), return_cache=True)
        data = {"prefill": {"logits": snapshot(logits), "cache": checked_cache(cache, layers)},
                "decode": []}
        for index in range(args.decode_steps):
            token = torch.tensor([[args.model_tokens + index + 1]], device="npu")
            logits, cache = model(token, cache=cache, return_cache=True)
            data["decode"].append({"logits": snapshot(logits), "cache": checked_cache(cache, layers)})
    if args.backend == "chunk" and calls != {
        "chunk_prefill": layers, "cce_decode": layers * args.decode_steps
    }:
        raise ValueError(f"wrong custom-operator route: {calls}")
    payload = {"dtype": args.dtype, "backend": args.backend, "checkpoint_sha256": checkpoint_digest,
               "prefill_tokens": args.model_tokens, "decode_steps": args.decode_steps,
               "layers": layers, "data": data, "core_calls": calls}
    args.dump.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.dump)
    return {"mode": "model", "backend": args.backend, "dtype": args.dtype,
            "checkpoint_sha256": checkpoint_digest, "prefill_tokens": args.model_tokens,
            "decode_steps": args.decode_steps, "core_calls": calls,
            "snapshot_tensor_count": sum(isinstance(v, torch.Tensor) for _, v in flatten(data)),
            "environment": environment,
            "note": "Snapshot production alone is not a cross-backend accuracy pass; run compare."}


def run_compare(args):
    if args.reference is None or args.candidate is None:
        raise ValueError("compare requires --reference and --candidate snapshots")
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    for name in ["dtype", "checkpoint_sha256", "prefill_tokens", "decode_steps", "layers"]:
        if reference[name] != candidate[name]:
            raise ValueError(f"model metadata differs: {name}")
    if candidate["dtype"] not in DTYPES or candidate["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("snapshot dtype/checkpoint is outside this model validation contract")
    if reference["backend"] != "torch" or candidate["backend"] != "chunk":
        raise ValueError("comparison requires torch reference and chunk candidate")
    if candidate["core_calls"] != {"chunk_prefill": candidate["layers"],
                                    "cce_decode": candidate["layers"] * candidate["decode_steps"]}:
        raise ValueError("snapshot does not demonstrate chunk-prefill/CCE-decode execution")
    for payload in [reference, candidate]:
        states = [payload["data"]["prefill"], *payload["data"]["decode"]]
        if len(states) != payload["decode_steps"] + 1:
            raise ValueError("snapshot has missing decode steps")
        for state in states:
            checked_cache(state["cache"], payload["layers"])
    budget = 1e-3 if candidate["dtype"] == "float32" else 1e-2
    result = compare_trees(candidate["data"], reference["data"], budget)
    return {"mode": "compare", "dtype": candidate["dtype"], "core_calls": candidate["core_calls"],
            "checkpoint_sha256": candidate["checkpoint_sha256"], **result}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("operator", "model", "compare"), default="operator")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--backend", choices=("torch", "chunk"), default="torch")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--heads", type=int, choices=(1, 16), default=16)
    parser.add_argument("--block-dim", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sandwiches", type=int, default=3)
    parser.add_argument("--sandwich", action="store_true", help="baseline/candidate/baseline acquisition")
    parser.add_argument("--shapes", choices=("real",), help="validate and time T1024 and T4096")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-tokens", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if min(args.reps, args.warmup, args.sandwiches, args.seq_len, args.decode_steps) < 1:
        parser.error("repeat, warmup, sandwiches, length and decode steps must be positive")
    if args.seq_len > 4096 or not 2 <= args.model_tokens <= 4096:
        parser.error("operator length must be <=4096 and model prefill length must be 2..4096")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "compare":
        result = run_compare(args)
    else:
        import torch_npu  # noqa: F401
        with torch.inference_mode():
            result = run_model(args) if args.mode == "model" else run_operator(args)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
