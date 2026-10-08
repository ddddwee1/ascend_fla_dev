#!/usr/bin/env python3
"""反向链在宽门控跨度下还有限吗？—— 前向已用 stable 根治，反向是独立的一处。

**为什么要单独测反向。** 读 ascriptor ``kda_bwd`` 的 ``finalize_pre.py`` /
``finalize_post.py``，它把 chunk 内成对衰减分解成::

    rscale = exp((g − g_last)·ln2)   # g − g_last ≥ 0 → 最大 exp(+span)
    cscale = exp((g_last − g)·ln2)   # ≤ 0            → 最小 exp(−span)

``rscale`` 是**真上溢**（fp32 与 bf16 的上溢线都在 ``ln(MAX) ≈ 88.72``，而
``q_scaled`` / ``k_scaled`` 是 bf16 GM 输出），配对的 ``kg`` 同时下溢到 0，下游四个矩阵乘
里 ``inf × 0 = NaN``。方向与前向那边的下溢**相反**，所以前向修好不代表反向好了。

本脚本逐档扫门控跨度，报：

1. 六个梯度的**有限性**（非有限元素个数）——这是主判据；
2. ``stable`` 与 ``upstream`` 在**重叠域**内的相对 L2 —— 应当 ≈0，用来证明改锚点是
   恒等变形（锚点常数在四个成对矩阵乘里逐项抵消）。

**不经 autograd**，直接调 ``chunk_kda_fwd_with_caches`` + ``chunk_kda_bwd``，``do`` / ``dht``
在 CPU 上造好再 H2D —— 这样在**缺内置算子包**的机器上也能跑（见 AGENTS.md §5 的可用面表：
那种机器上 NPU 侧的 Cast / sum / zeros 全不可用）。

一个进程只跑一个 ``impl``：同名算子多 build 会互相覆盖（AGENTS.md §6 铁律二），而两套
实现虽然改了名，``prepare`` 的链也是按 impl 选的。多 impl 由本脚本自动派子进程。

用法::

    python benchmarks/probe_bwd_span.py                       # 两个 impl 各一个子进程
    python benchmarks/probe_bwd_span.py --impl stable --block-dim 1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

B, T, H, HV, D = 1, 64, 1, 1, 128
GRADS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
#: 门控倍数 → 跨度约为 0.0315 × 64 × mult（g_raw ∈ [-0.03,0] 的期望）
MULTS = (1.0, 10.0, 45.0, 67.0, 80.0, 90.0, 112.0, 134.0, 156.0)


def _inputs(seed: int = 2026):
    """在 CPU 上造一份输入。不用 NPU 算子 —— 缺算子包的机器上造不了。"""
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, generator=g), dim=-1)
    return dict(
        q=q.bfloat16(), k=k.bfloat16(),
        v=(torch.randn(B, T, HV, D, generator=g) * 0.04).bfloat16(),
        beta=(torch.rand(B, T, HV, generator=g) * 0.45 + 0.05),
        g_unit=-torch.rand(B, T, HV, D, generator=g) * 0.03,
        h0=(torch.randn(B, HV, D, D, generator=g) * 0.01),
        do=(torch.randn(B, T, HV, D, generator=g) * 0.04).bfloat16(),
        dht=(torch.randn(B, HV, D, D, generator=g) * 0.01).bfloat16(),
    )


def _bad(t: torch.Tensor) -> tuple[int, int, int]:
    """(非有限数, nan 数, inf 数)。一律先 D2H 再判 —— NPU 上 isfinite 要算子。"""
    f = t.cpu().float()
    return int((~f.isfinite()).sum()), int(f.isnan().sum()), int(f.isinf().sum())


def run(impl: str, block_dim: int, mults=None) -> list[dict]:
    from ascend_fla.ops.kda import chunk_kda_fwd_with_caches, prepare
    from ascend_fla.ops.kda.chunk import _gate_span
    from ascend_fla.ops.kda.chunk_bwd import chunk_kda_bwd

    prepare("a5", block_dim, backward=True, impl=impl)
    x = _inputs()
    npu = lambda t: t.to("npu")  # noqa: E731

    rows = []
    for mult in (mults or MULTS):
        gf = (x["g_unit"] * mult).float()
        span = _gate_span(gf, T // 64, on_cpu=True)
        _, _, caches = chunk_kda_fwd_with_caches(
            npu(x["q"]), npu(x["k"]), npu(x["v"]), npu(gf), npu(x["beta"]),
            None, npu(x["h0"]),
            block_dim=block_dim, check_gate_range=False, impl=impl,
        )
        grads = chunk_kda_bwd(
            q=npu(x["q"]), k=npu(x["k"]), v=npu(x["v"]),
            beta=npu(x["beta"].bfloat16()), do=npu(x["do"]), dht=npu(x["dht"]),
            caches=caches, block_dim=block_dim, impl=impl,
        )
        row = dict(impl=impl, block_dim=block_dim, mult=mult, span=span)
        for name in GRADS:
            bad, nan, inf = _bad(grads[name])
            row[name] = dict(bad=bad, nan=nan, inf=inf,
                             n=grads[name].numel(),
                             vals=grads[name].cpu().float().flatten()[:4].tolist())
        # 全量存下来太大，只留一份可比的指纹：每个梯度的 L2 与前 64 个元素
        row["fingerprint"] = {n: dict(l2=grads[n].cpu().float().norm().item(),
                                      head=grads[n].cpu().float().flatten()[:64].tolist())
                              for n in GRADS}
        rows.append(row)
        # 不用嵌套 f-string：远端是 python 3.11，同引号嵌套在那儿是 SyntaxError
        flags = "  ".join(
            "{}:{}".format(n, "OK" if row[n]["bad"] == 0 else "BAD{}".format(row[n]["bad"]))
            for n in GRADS)
        print(f"{impl:>8} span={span:>7.2f}  {flags}", flush=True)
    return rows


def _rel_l2(a: list[float], b: list[float]) -> float:
    ta, tb = torch.tensor(a), torch.tensor(b)
    return ((ta - tb).norm() / tb.norm().clamp_min(1e-30)).item()


def compare(rows: list[dict]) -> None:
    """重叠域内 stable 与 upstream 的逐梯度相对 L2。应当 ≈0（恒等变形）。"""
    by = {}
    for r in rows:
        by.setdefault(r["mult"], {})[r["impl"]] = r
    print(f"\n{'span':>8} " + " ".join(f"{n:>10}" for n in GRADS) + "   ← stable vs upstream 相对 L2")
    for mult in sorted(by):
        pair = by[mult]
        if len(pair) != 2:
            continue
        st, up = pair.get("stable"), pair.get("upstream")
        if any(up["fingerprint"][n]["l2"] != up["fingerprint"][n]["l2"] for n in GRADS):
            print(f"{st['span']:>8.2f} upstream 侧含 NaN，无法比对")
            continue
        if any(up[n]["bad"] for n in GRADS):
            print(f"{st['span']:>8.2f} " + " ".join(f"{'upNaN':>10}" for _ in GRADS))
            continue
        errs = [_rel_l2(st["fingerprint"][n]["head"], up["fingerprint"][n]["head"])
                for n in GRADS]
        print(f"{st['span']:>8.2f} " + " ".join(f"{e:>10.3e}" for e in errs))


def a2_gradient_metrics(actual, expected, rule):
    """Whole-output readback; no denominator floor, NaN masking or zero-gradient pass.

    The FP32 oracle's tensor values are unchanged. FP64 is used only to accumulate
    error norms, as in the unit contract runner. Fixture allclose and relative-L2
    requirements are conjunctive; end-to-end uses the separately specified .05.
    """
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    a, e = actual.double(), expected.double()
    finite = bool(a.isfinite().all() and e.isfinite().all())
    residual = float((a - e).norm()) if finite else None
    norm = float(e.norm()) if bool(e.isfinite().all()) else None
    relative = (residual / norm if norm else (0.0 if residual == 0 else None)) if finite else None
    nonzero = bool(a.count_nonzero()) or not bool(e.count_nonzero())
    close = None
    if "rtol" in rule:
        close = bool(torch.allclose(a, e, rtol=rule["rtol"], atol=rule["atol"]))
    return {"finite": finite, "actual_nonfinite": int((~a.isfinite()).sum()),
            "expected_nonfinite": int((~e.isfinite()).sum()),
            "relative_l2": relative, "expected_norm": norm, "residual_norm": residual,
            "max_abs": float((a - e).abs().max()) if finite else None,
            "nonzero_when_expected": nonzero, "allclose": close,
            "budget": rule["max_relative_l2"],
            "passed": finite and nonzero and close is not False and relative is not None
                      and relative <= rule["max_relative_l2"]}


class A2BackwardNative:
    """Direct frozen A2 units; ALL five forward and nine backward vendors first.

    The common launch binder preserves device intermediates and binds every scalar
    from the compiled signature. Public dispatch and its qualification gate are
    not involved. A worker owns exactly one block_dim for its lifetime.
    """
    def __init__(self, bd, emit):
        from benchmarks.verify_decode import _A2Native, _a2_load, _a2_unit
        from ascend_fla.runtime.compile import compile_kernel
        if bd not in (1, 2):
            raise ValueError("A2 backward currently declares block_dim 1 and 2")
        self.bd = bd
        self.forward, self.forward_ref = _a2_unit("kda_fwd_stable")
        self.backward, self.backward_ref = _a2_unit("kda_bwd_stable")
        self.contract = json.loads((self.backward.HERE / "contract.json").read_text())
        self.forward_contract = json.loads((self.forward.HERE / "contract.json").read_text())
        plan = [(f"fwd.{name}", kernel) for name, kernel in self.forward._kernels().items()]
        plan += [(f"bwd.{name}", kernel) for name, kernel in self.backward._kernels().items()]
        if len(plan) != 14 or sum(n.startswith("bwd.") for n, _ in plan) != 9:
            raise RuntimeError("incomplete forward/backward precompile plan")
        self.names = {id(kernel): name for name, kernel in plan}
        self.compiled, self.trace = {}, []
        self.launch = _A2Native.launch.__get__(self, type(self))
        for name, kernel in plan:
            emit({"stage": "compile_start", "kernel": name, "block_dim": bd})
            obj = compile_kernel(kernel, device="a2", block_dim=bd, backend="cce")
            self.compiled[name] = obj
            emit({"stage": "vendor_compiled", "kernel": name, "signature": obj.signature,
                  "scalar_names": obj.scalar_names, "block_dim": bd})
        if self.trace:
            raise RuntimeError("custom kernel launched before precompile completed")
        emit({"stage": "precompile_complete", "forward_count": 5,
              "backward_count": 9, "custom_launches": 0, "block_dim": bd})
        self.runner = _a2_load(self.backward.HERE / "_unit_runner.py", "_a213_runner")

    def chain(self, unit, inputs):
        previous = sys.modules.get("_unit_runner")
        self.runner.launch_kernel = self.launch
        try:
            sys.modules["_unit_runner"] = self.runner
            result = unit._execute_chain(
                inputs, {"device": "a2", "backend": "cce", "block_dim": self.bd})
            torch.npu.synchronize()
            return result
        finally:
            if previous is None:
                sys.modules.pop("_unit_runner", None)
            else:
                sys.modules["_unit_runner"] = previous

    def backward_call(self, inputs):
        outputs, stages = self.chain(self.backward, inputs)
        b, t, h, _ = inputs["q"].shape
        dims = {"B": b, "C": t // 64, "H": h, "HV": inputs["v"].shape[2]}
        # Exactly the contract's metadata-only views, from the same nine launches.
        def extent(value):
            if isinstance(value, int):
                return value
            result = 1
            for factor in value.split("*"):
                result *= int(factor) if factor.isdecimal() else dims[factor]
            return result
        if set(stages) != set(self.contract["stages"]):
            raise RuntimeError("missing or unexpected backward stage outputs")
        return outputs, {name: value.view(*(extent(d) for d in self.contract["stages"][name]["shape"]))
                         for name, value in stages.items()}


def a2_fixture_cases(native, emit):
    """Existing independent mathematical fixtures, NOT actual-cache qualification."""
    from benchmarks.verify_decode import _a2_digest
    comparison = native.contract["comparison"]
    rows = []
    for case in native.contract["cases"]:
        if case["block_dim"] != native.bd:
            continue
        emit({"stage": "fixture_start", "case": case["id"], "block_dim": native.bd})
        inputs = native.backward.make_inputs(case)
        started = time.monotonic()
        trace_start = len(native.trace)
        actual, stages = native.backward_call(inputs)
        device_s = time.monotonic() - started
        # References are independently generated at run time, after the full chain.
        expected = native.backward.reference(inputs)
        expected_stages = native.backward.reference_stages(inputs)
        if set(actual) != set(expected) or set(stages) != set(expected_stages):
            raise RuntimeError("incomplete fixture output comparison")
        metrics = {name: a2_gradient_metrics(value, expected[name],
                   {**comparison["default"], **comparison.get("outputs", {}).get(name, {})})
                   for name, value in actual.items()}
        stage_metrics = {name: a2_gradient_metrics(value, expected_stages[name],
                         {**comparison["default"], **comparison.get("stage_outputs", {}).get(name, {})})
                         for name, value in stages.items()}
        row = {"stage": "independent_fixture", "case": case["id"], "block_dim": native.bd,
               "parameters": case["parameters"], "seed": case["seed"],
               "oracle": "CPU FP32 fixture autograd, declared BF16 outputs",
               "end_to_end_actual_cache_acceptance": False,
               "gradients": metrics, "checkpoints": stage_metrics,
               "output_digests": {n: _a2_digest(v) for n, v in actual.items()},
               "saved_input_digests": {n: _a2_digest(v) for n, v in inputs["saved"].items()},
               "launches": native.trace[trace_start:], "chain_wall_s": device_s,
               "passed": all(m["passed"] for m in (*metrics.values(), *stage_metrics.values()))}
        rows.append(row)
        emit(row)
    return rows


def a2_worker(bd, out, suite):
    """One process / one block_dim. Caller supplies an isolated output directory."""
    import hashlib
    from benchmarks.verify_decode import _a2_identity
    out.mkdir(parents=True, exist_ok=False)
    def emit(row):
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
    identity = _a2_identity()
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [pathlib.Path(__file__).resolve(), root / "ascend_fla/reference/kda.py"]
    sources += [p for p in (root / "kernels/projects/a2/kda_bwd_stable").rglob("*")
                if p.is_file() and (p.suffix == ".py" or p.name == "contract.json")
                and "evidence" not in p.parts and "__pycache__" not in p.parts]
    identity["backward_qualification_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    emit({"stage": "identity_verified", "block_dim": bd, "soc": identity["soc"],
          "cann": identity["cann"], "opp_packages": identity["opp_packages"]})
    native = A2BackwardNative(bd, emit)
    if suite != "fixture":
        raise ValueError("unsupported A2 suite")
    rows = a2_fixture_cases(native, emit)
    receipt = {"block_dim": bd, "suite": suite, "cases": len(rows),
               "passed": all(row["passed"] for row in rows),
               "actual_cache_qualified": False, "rows": rows}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    emit({key: value for key, value in receipt.items() if key != "rows"})
    return 0 if receipt["passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=("stable", "upstream"), default=None,
                    help="不传则两个都跑（各一个子进程）")
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--mults", type=float, nargs="+", default=None,
                    help="门控倍数列表，默认 MULTS。本脚本的输入分布下跨度 ≈ 1.117 × 倍数")
    ap.add_argument("--json-out", type=pathlib.Path)
    ap.add_argument("--soc", choices=("a5", "a2"), default="a5")
    ap.add_argument("--a2-suite", choices=("fixture",), default="fixture")
    ap.add_argument("--a2-out", type=pathlib.Path)
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("本脚本只在 NPU 上有意义", file=sys.stderr)
        return 2

    if args.soc == "a2":
        if args.impl not in (None, "stable") or args.mults or args.json_out or args.a2_out is None:
            ap.error("A2 requires --a2-out and its own suite; A5 impl/mults/json-out do not apply")
        return a2_worker(args.block_dim, args.a2_out, args.a2_suite)

    if args.impl is not None:
        rows = run(args.impl, args.block_dim, args.mults)
        if args.json_out:
            args.json_out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return 0

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        for impl in ("upstream", "stable"):
            out = pathlib.Path(td) / f"{impl}.json"
            r = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).resolve()),
                 "--impl", impl, "--block-dim", str(args.block_dim), "--json-out", str(out),
                 *(["--mults", *map(str, args.mults)] if args.mults else [])],
                capture_output=True, text=True)
            sys.stdout.write("\n".join(l for l in r.stdout.splitlines()
                                       if not l.startswith("[lint]")) + "\n")
            if r.returncode != 0:
                print(f"impl={impl} 子进程失败 exit={r.returncode}", file=sys.stderr)
                print(r.stderr[-3000:], file=sys.stderr)
                return 1
            rows += json.loads(out.read_text(encoding="utf-8"))
    compare(rows)
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
