#!/usr/bin/env python3
"""decode 路径的真机验收（需要 A5 NPU）。

四件事，每件对应 decode 成立的一个前提：

``acc``
    对 fp32 递推参考的精度。decode 全程 fp32、无 bf16 中间量，所以预算取
    **1e-05 量级**而不是 chunk 路径的 5e-02 —— 松预算会让真实的退化藏进去。

``chain``
    **state 串接逐位相同**：逐 token 调 T 次并串接 state，必须与一次调 T 个 token
    比特一致。decode 的正确性就建立在这条上 —— 不是"差不多"，是逐位。

``bd``
    声明的 ``block_dim`` 每个值都要实测。本 kernel 只用向量核、不碰 cube，
    ``GetVecNum() == 2*block_dim`` 而物理上有 56 个向量核，所以上限 28 是**推断**，
    必须测出来（超过物理核数会在硬件 barrier 死锁，见 AGENTS.md §5）。

``split``
    把一次调用的耗时拆成 host 布局 / 桥+设备 / 输出重排，并用 T=1 与 T=16 的差
    求设备侧的边际。**不拆就别下"谁是瓶颈"的结论** —— 我先推断 decode 是带宽瓶颈，
    实测是每次调用的固定成本，差了一个数量级（AGENTS.md §6 铁律一）。

**一个 bd 一个进程**（铁律二：一个算子名一份 build），``bd`` 那项自动派子进程。

用法::

    python benchmarks/verify_decode.py                    # 四件全做
    python benchmarks/verify_decode.py --check acc chain

A2-14 qualifies the existing mixed BF16/FP32 A2 unit, independently of public
dispatch. ``--soc a2 --a2-out <fresh scratch>`` runs the full real-shape grid,
chunk/decode continuation and a separate process for each of bd=1,2. A2 ``split``
means GQA head grouping; the A5 timing decomposition above is unchanged.
``--a2-performance`` additionally measures three synchronized Torch NPU sandwiches.
The CPU golden computes in FP32; the unit's FP64 reference is a labelled diagnostic.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ascend_fla.ops.kda.chunk import HEAD_DIM, VALUE_DIM          # noqa: E402
from ascend_fla.ops.kda.fused_recurrent import (                  # noqa: E402
    SUPPORTED_BLOCK_DIM,
    T_MAX,
    fused_recurrent_kda,
)
from ascend_fla.reference.kda import kda_recurrent_ref            # noqa: E402

D = 128
#: 全 fp32 路径的预算。比 chunk 路径（5e-02）紧三个数量级 —— 见模块文档。
BUDGET = 1e-5
#: decode 形状：kimi_linear_layer 的头配置，T=1。
KIMI = dict(B=1, H=32, HV=32)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def make(B, T, H, HV, seed=2026):
    gen = torch.Generator().manual_seed(seed)
    f = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
    return dict(
        q=torch.nn.functional.normalize(f(B, T, H, D), dim=-1),
        k=torch.nn.functional.normalize(f(B, T, H, D), dim=-1),
        v=f(B, T, HV, D) * 0.04,
        g=-torch.rand(B, T, HV, D, generator=gen) * 0.03,
        beta=torch.rand(B, T, HV, generator=gen) * 0.45 + 0.05,
        h0=f(B, HV, D, D) * 0.01,
    )


def npu(x: torch.Tensor) -> torch.Tensor:
    return x.to("npu")


def reference(x: dict):
    return kda_recurrent_ref(
        x["q"].float(), x["k"].float(), x["v"].float(), x["g"].float(), x["beta"].float(),
        initial_state=x["h0"].float(), output_final_state=True)


def call(x: dict, bd: int, state=None, sl: slice | None = None):
    g = (lambda n: x[n][:, sl]) if sl is not None else (lambda n: x[n])
    return fused_recurrent_kda(
        npu(g("q")), npu(g("k")), npu(g("v")), npu(g("g")), npu(g("beta")),
        initial_state=npu(x["h0"]) if state is None else state,
        output_final_state=True, block_dim=bd)


# ------------------------------------------------------------------------------ acc
def check_acc(bd: int) -> int:
    print(f"\n=== acc：对 fp32 递推参考的相对 L2（预算 {BUDGET:.0e}，bd={bd}）===")
    cases = ((1, 1, 1, 1), (1, 1, 1, 2), (1, 4, 1, 1), (1, 1, 2, 4),
             (1, T_MAX, 1, 1), (1, 1, 16, 32), (1, 1, 32, 32), (1, 8, 32, 32))
    bad = []
    for B, T, H, HV in cases:
        x = make(B, T, H, HV)
        o, ht = call(x, bd)
        o_ref, ht_ref = reference(x)
        eo, es = rel_l2(o, o_ref), rel_l2(ht, ht_ref)
        flag = "" if max(eo, es) < BUDGET else "  ← 超预算"
        if flag:
            bad.append((B, T, H, HV, eo, es))
        print(f"  B{B} T{T:<3d} H{H:<3d} HV{HV:<3d}  o={eo:.3e}  final_state={es:.3e}{flag}")
    print("  判定：" + ("全部在预算内" if not bad else f"超预算 {bad}"))
    return 1 if bad else 0


# ---------------------------------------------------------------------------- chain
def check_chain(bd: int) -> int:
    """逐 token 串接必须与整段调用**逐位相同**。decode 的正确性就是这条。"""
    print(f"\n=== chain：state 串接（{T_MAX} 次单 token vs 一次 {T_MAX} token，bd={bd}）===")
    bad = []
    for B, H, HV in ((1, 4, 8), (1, 32, 32)):
        x = make(B, T_MAX, H, HV)
        o_all, ht_all = call(x, bd)
        state, outs = npu(x["h0"]), []
        for i in range(T_MAX):
            oi, state = call(x, bd, state=state, sl=slice(i, i + 1))
            outs.append(oi)
        o_step = torch.cat(outs, dim=1)
        same_o = torch.equal(o_step.cpu(), o_all.cpu())
        same_s = torch.equal(state.cpu(), ht_all.cpu())
        if not (same_o and same_s):
            bad.append((B, H, HV, rel_l2(o_step, o_all), rel_l2(state, ht_all)))
        print(f"  B{B} H{H:<3d} HV{HV:<3d}  o 逐位相同={same_o}  final_state 逐位相同={same_s}"
              f"   （不同则 relL2 o={rel_l2(o_step, o_all):.2e} s={rel_l2(state, ht_all):.2e}）")
    print("  判定：" + ("逐位一致" if not bad else f"**不一致** {bad} —— decode 的前提不成立"))
    return 1 if bad else 0


# ------------------------------------------------------------------------------- bd
def _one_bd(bd: int, t: int) -> int:
    x = make(1, t, KIMI["H"], KIMI["HV"], seed=7)
    d = {k: npu(v) for k, v in x.items()}
    fn = lambda: fused_recurrent_kda(  # noqa: E731
        d["q"], d["k"], d["v"], d["g"], d["beta"],
        initial_state=d["h0"], output_final_state=True, block_dim=bd)
    o, ht = fn()
    o_ref, ht_ref = reference(x)
    us = timed(fn)
    print(f"  bd={bd:<3d} T={t:<3d} {us:8.1f} µs/次 ({us / t:7.1f} µs/token)   "
          f"o={rel_l2(o, o_ref):.2e} state={rel_l2(ht, ht_ref):.2e}")
    return 0 if max(rel_l2(o, o_ref), rel_l2(ht, ht_ref)) < BUDGET else 1


def timed(fn, warm: int = 5, iters: int = 50) -> float:
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def check_bd(_bd: int) -> int:
    print(f"\n=== bd：声明的 {SUPPORTED_BLOCK_DIM} 每个值都要实测（kimi decode 形状）===")
    print("  28 → 56 个向量核，是物理上限；超过会在硬件 barrier 死锁，所以这一项是安全检查")
    rc = 0
    for bd in SUPPORTED_BLOCK_DIM:
        r = subprocess.run([sys.executable, __file__, "--check", "bd", "--_one", str(bd)],
                           text=True, capture_output=True)
        sys.stdout.write("".join(ln for ln in r.stdout.splitlines(keepends=True)
                                 if "bd=" in ln) or f"  bd={bd} 无输出\n")
        if r.returncode:
            sys.stderr.write(r.stderr[-600:])
            rc = 1
    print("  判定：" + ("全部跑通且在预算内" if not rc else "有档失败，见上"))
    return rc


# ---------------------------------------------------------------------------- split
def check_split(bd: int) -> int:
    """拆开量：谁是瓶颈。不拆就别下结论。"""
    print(f"\n=== split：一次调用的耗时拆解（kimi decode 形状，bd={bd}）===")
    x = make(1, 1, KIMI["H"], KIMI["HV"], seed=7)
    d = {k: npu(v) for k, v in x.items()}
    sc = HEAD_DIM ** -0.5
    bhv = lambda t: t.to(torch.float32).permute(0, 2, 1, 3).contiguous()  # noqa: E731

    def host_only():
        bhv(d["q"] * sc), bhv(d["k"]), bhv(d["v"]), bhv(d["g"])
        d["beta"].to(torch.float32).permute(0, 2, 1).contiguous().view(1, KIMI["HV"], 1, 1)
        torch.empty(1, KIMI["HV"], 1, VALUE_DIM, dtype=torch.float32, device="npu")
        torch.empty(1, KIMI["HV"], HEAD_DIM, VALUE_DIM, dtype=torch.float32, device="npu")

    full1 = timed(lambda: fused_recurrent_kda(
        d["q"], d["k"], d["v"], d["g"], d["beta"],
        initial_state=d["h0"], output_final_state=True, block_dim=bd))
    host = timed(host_only)

    xt = make(1, T_MAX, KIMI["H"], KIMI["HV"], seed=7)
    dt = {k: npu(v) for k, v in xt.items()}
    fullt = timed(lambda: fused_recurrent_kda(
        dt["q"], dt["k"], dt["v"], dt["g"], dt["beta"],
        initial_state=dt["h0"], output_final_state=True, block_dim=bd))
    marginal = (fullt - full1) / (T_MAX - 1)

    print(f"  T=1 整次      {full1:8.1f} µs")
    print(f"  其中 host 布局 {host:8.1f} µs   （{host / full1 * 100:.0f}%）")
    print(f"  T={T_MAX} 整次     {fullt:8.1f} µs")
    print(f"  设备侧边际     {marginal:8.1f} µs/token   （(T={T_MAX} − T=1) / {T_MAX - 1}）")
    print(f"  固定成本       {full1 - marginal:8.1f} µs/次")
    print("  判定：固定成本" + ("主导 —— 先解 decode-call-overhead，不是调 kernel"
                              if (full1 - marginal) > 2 * marginal else "不再主导，可以回头看 kernel"))
    return 0


# ---------------------------------------------------------------- A2 unit qualification
# Deliberately independent of the public A5 dispatcher. A2 public routing is unqualified.
def _a2_load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _a2_unit(name):
    """Load units without leaking their legacy bare `reference` import into another unit."""
    root = pathlib.Path(__file__).resolve().parents[1] / "kernels/projects/a2" / name
    ref = _a2_load(root / "reference.py", f"_verify_{name}_reference")
    previous = sys.modules.get("reference")
    try:
        sys.modules["reference"] = ref
        unit = _a2_load(root / "unit.py", f"_verify_{name}_unit")
    finally:
        if previous is None:
            sys.modules.pop("reference", None)
        else:
            sys.modules["reference"] = previous
    return unit, ref


def a2_fp32_oracle(x):
    """Independent mathematical recurrence, FP32 throughout, on the input device.

    CPU is the primary golden; the same composed Torch expression on NPU is the
    second oracle. This is test computation, never a fallback for the custom path.
    The unit's separate FP64 reference is also measured, explicitly labelled FP64.
    """
    q, k, v, g, beta, state = (x[n].float() for n in
                              ("q", "k", "v", "g", "beta", "initial_state"))
    b, t, h, _ = q.shape
    hv = v.shape[2]
    groups = torch.arange(hv, device=q.device) // (hv // h)
    state = state.clone()
    outputs = []
    for i in range(t):
        qt, kt = q[:, i][:, groups], k[:, i][:, groups]
        decayed = state * g[:, i].exp().unsqueeze(-1)
        prediction = torch.einsum("bhk,bhkv->bhv", kt, decayed)
        delta = (v[:, i] - prediction) * beta[:, i].unsqueeze(-1)
        state = decayed + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", qt * D ** -.5, state))
    return {"o": torch.stack(outputs, dim=1), "final_state": state}


def a2_metrics(actual, expected, rule):
    """Check allclose AND relative L2, with whole-tensor D2H before casts/slicing."""
    a, b = actual.detach().cpu(), expected.detach().cpu()
    if a.shape != b.shape:
        raise ValueError(f"comparison shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
    a, b = a.float(), b.float()
    finite = bool(a.isfinite().all() and b.isfinite().all())
    diff = (a - b).abs()
    relative = float((a - b).norm() / b.norm().clamp_min(1e-30)) if finite else None
    close = bool(torch.allclose(a, b, rtol=rule["rtol"], atol=rule["atol"]))
    return {"finite": finite, "rel_l2": relative,
            "max_abs": float(diff.max()) if finite else None,
            "allclose": close,
            "out_of_tolerance": int((diff > rule["atol"] + rule["rtol"] * b.abs()).sum()),
            "passed": finite and close and relative <= rule["max_relative_l2"]}


def _a2_digest(tensor):
    import hashlib
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def _a2_rules(contract):
    default = contract["comparison"]["default"]
    overrides = contract["comparison"].get("outputs", {})
    return {name: {**default, **overrides.get(name, {})} for name in ("o", "final_state")}


def _a2_compare(actual, expected, rules):
    return {name: a2_metrics(actual[name], expected[name], rules[name])
            for name in ("o", "final_state")}


def _a2_slice(x, start, stop):
    # Runtime fixture creation on CPU only, before transfer to the tested path.
    if any(v.device.type != "cpu" for v in x.values()):
        raise ValueError("fixture slicing requires CPU tensors")
    return {k: v if k == "initial_state" else v[:, start:stop].contiguous()
            for k, v in x.items()}


class _A2Native:
    """Six precompiled vendors, actual unit launches and zero-copy decode state."""
    def __init__(self, bd, emit):
        import json
        from ascend_fla.runtime.compile import compile_kernel
        if bd not in (1, 2):
            raise ValueError("A2-14 only qualifies block_dim 1 and 2")
        self.bd = bd
        self.decode, self.decode_ref = _a2_unit("kda_fused_recurrent")
        self.chunk, _ = _a2_unit("kda_fwd_stable")
        self.contract = json.loads((self.decode.HERE / "contract.json").read_text())
        self.chunk_contract = json.loads((self.chunk.HERE / "contract.json").read_text())
        kernels = {"decode": self.decode._kernel(), **self.chunk._kernels()}
        self.compiled = {}
        self.names = {id(kernel): name for name, kernel in kernels.items()}
        self.trace = []
        for name, kernel in kernels.items():
            emit({"stage": "compile_start", "kernel": name, "block_dim": bd})
            obj = compile_kernel(kernel, device="a2", block_dim=bd, backend="cce")
            self.compiled[name] = obj
            emit({"stage": "vendor_compiled", "kernel": name, "signature": obj.signature,
                  "scalar_names": obj.scalar_names, "block_dim": bd})
        assert len(self.compiled) == 6
        emit({"stage": "precompile_complete", "count": 6, "custom_launches": 0})
        self.runner = _a2_load(self.chunk.HERE / "_unit_runner.py", "_verify_a2_runner")

    def launch(self, kernel, args, options):
        compiled = self.compiled[self.names[id(kernel)]]
        ni, no = len(compiled.input_names), len(compiled.output_names)
        tensors = [x if x.device.type == "npu" else x.to("npu") for x in args[:ni + no]]
        dims = {}
        for entry, tensor in zip(compiled.spec.inputs + compiled.spec.outputs, tensors):
            for symbol, extent in zip(entry["dims"], tensor.shape):
                if isinstance(symbol, str):
                    dims[symbol] = int(extent)
        explicit = args[ni + no:]
        scalars = dict(zip(compiled.scalar_names[:len(explicit)], explicit))
        for name in compiled.scalar_names[len(explicit):]:
            scalars[name] = dims[name]
        trace = {"kernel": self.names[id(kernel)], "signature": compiled.signature}
        if self.names[id(kernel)] == "decode":
            trace["state_input_preserved"] = (args[5].device.type == "npu"
                                              and tensors[5].data_ptr() == args[5].data_ptr())
        self.trace.append(trace)
        compiled(dict(zip(compiled.input_names, tensors[:ni])), scalars,
                 dict(zip(compiled.output_names, tensors[ni:])))
        return tensors[ni] if no == 1 else tuple(tensors[ni:])

    def prepare_decode(self, x):
        self.decode.validate_inputs(x, {"block_dim": self.bd})
        return {k: v.to("npu") for k, v in x.items()}

    def decode_call(self, x, state=None, outputs=None):
        # NPU tensors retain their original dtype/layout. Only allocation and view here.
        q, k, v, g, beta = (x[n] for n in ("q", "k", "v", "g", "beta"))
        b, t, h, _ = q.shape
        hv = v.shape[2]
        if not 1 <= t <= 16:
            raise ValueError("decode T must be in 1..16")
        h0 = x["initial_state"] if state is None else state
        if h0.device.type != "npu":
            raise ValueError("decode continuation requires the original NPU state")
        if outputs is None:
            outputs = (torch.full((b * t, hv * D), float("nan"), dtype=torch.bfloat16).to("npu"),
                       torch.full((b, hv, D, D), float("nan"), dtype=torch.float32).to("npu"))
        o, ht = self.launch(self.decode._kernel(),
                            (q.view(b * t, h * D), k.view(b * t, h * D),
                             v.view(b * t, hv * D), g.view(b * t, hv * D),
                             beta.view(b * t, hv), h0, *outputs, b, t, h, hv, D ** -.5), {})
        return {"o": o.view(b, t, hv, D), "final_state": ht}

    def unit_call(self, unit, x):
        """Exercise the actual unit.execute entry, replacing only its launcher."""
        previous = sys.modules.get("_unit_runner")
        self.runner.launch_kernel = self.launch
        try:
            sys.modules["_unit_runner"] = self.runner
            return unit.execute(x, {"device": "a2", "backend": "cce", "block_dim": self.bd})
        finally:
            if previous is None:
                sys.modules.pop("_unit_runner", None)
            else:
                sys.modules["_unit_runner"] = previous

    def chunk_call(self, x):
        return self.unit_call(self.chunk, x)


def _a2_identity():
    """Verify actual imported sources before compilation; receipts contain no host paths."""
    import hashlib
    import json
    import os
    import platform
    import ascriptor
    import torch_npu
    root = pathlib.Path(__file__).resolve().parents[1]
    workspace = pathlib.Path(os.environ["ASCRIPTOR_WORKSPACE"])
    pin = json.loads((root / "docs/matrix/ops.json").read_text())["ascriptor_pin"]["current_compatibility_pin"]
    revisions = {}
    library = pathlib.Path(ascriptor.__file__).resolve().parents[1]
    if library != (workspace / "library").resolve():
        raise RuntimeError("imported ascriptor is not the selected workspace library")
    for component in ("library", "kernels"):
        owner = workspace / component
        revision = subprocess.check_output(["git", "-C", str(owner), "rev-parse", "HEAD"], text=True).strip()
        # Source identities come from the repository's active pin, not historical evidence.
        expected = pin[component]["commit"] if isinstance(pin[component], dict) else pin[component]
        if revision != expected:
            raise RuntimeError(f"selected {component} revision differs from the active pin")
        revisions[component] = revision
    files = subprocess.check_output(["git", "-C", str(library), "ls-files", "ascriptor"], text=True).splitlines()
    library_hashes = {}
    for name in files:
        path = library / name
        expected = subprocess.check_output(["git", "-C", str(library), "show", f"HEAD:{name}"])
        if path.read_bytes() != expected:
            raise RuntimeError(f"imported library file differs from pin: {name}")
        library_hashes[name] = hashlib.sha256(expected).hexdigest()
    patterns = ("kernels/projects/a2/kda_fused_recurrent", "kernels/projects/a2/kda_fwd_stable",
                "ascend_fla/runtime", "benchmarks/verify_decode.py")
    source_files = subprocess.check_output(["git", "-C", str(root), "ls-files", *patterns], text=True).splitlines()
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files
              if not name.startswith("kernels/") or "/evidence/" not in name}
    cann = pathlib.Path(os.environ["ASCEND_HOME_PATH"])
    versions = {}
    for name in ("compiler", "opp", "runtime"):
        path = cann / name / "version.info"
        if path.is_file():
            raw = path.read_text()
            # Version files contain version/timestamp fields, no install path is needed.
            versions[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "fields": [line for line in raw.splitlines()
                                         if line.startswith(("Version=", "version=", "version_dir=", "timestamp=", "innerversion="))]}
    opp = pathlib.Path(os.environ["ASCEND_OPP_PATH"]) / "built-in/op_impl/ai_core/tbe/kernel"
    properties = torch.npu.get_device_properties(0)
    return {"soc": torch.npu.get_device_name(0), "python": platform.python_version(),
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "torch_npu_allow_hf32": torch.npu.matmul.allow_hf32,
            "device_geometry": {"cube_cores": properties.cube_core_num,
                                "vector_cores": properties.vector_core_num},
            "ascriptor": ascriptor.__version__, "revisions": revisions,
            "cann": versions, "opp_packages": sorted(p.name for p in opp.iterdir() if p.is_dir()),
            "library_files_checked": len(library_hashes),
            "library_manifest_sha256": hashlib.sha256(json.dumps(library_hashes, sort_keys=True).encode()).hexdigest(),
            "source_sha256": hashes}


def _a2_perf(native, x, emit):
    """Synchronized end-to-end launch time; fixtures/transfers excluded from both paths."""
    device = native.prepare_decode(x)
    first = native.decode_call(device)
    allocated = (first["o"].view(-1, x["v"].shape[2] * D), first["final_state"])
    custom = lambda: native.decode_call(device, outputs=allocated)
    baseline = lambda: a2_fp32_oracle(device)
    def measure(fn):
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        samples = []
        for _ in range(10):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            fn()
            torch.npu.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1000)
        return samples
    for turn in range(3):
        emit({"stage": "performance", "T": x["q"].shape[1], "round": turn + 1,
              "synchronized": True, "warmup": 3, "repeat": 10,
              "order": "custom/torch_npu/custom", "custom_before_us": measure(custom),
              "torch_npu_us": measure(baseline), "custom_after_us": measure(custom),
              "scope": "custom preallocated outputs; Torch composed FP32 recurrence allocates intermediates"})


def _a2_worker(bd, out, performance=False):
    import json
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    out.mkdir(parents=True, exist_ok=True)
    records, hashes, input_hashes = [], {}, {}
    journal = out / f"bd{bd}.jsonl"
    def emit(row):
        row = {"block_dim": bd, **row}
        records.append(row)
        line = json.dumps(row, sort_keys=True)
        with journal.open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
    if journal.exists():
        raise ValueError("use a fresh output directory; an earlier receipt must not be overwritten")
    identity = _a2_identity()
    emit({"stage": "environment", "identity": identity})
    native = _A2Native(bd, emit)
    rules, chunk_rules = _a2_rules(native.contract), _a2_rules(native.chunk_contract)
    emit({"stage": "comparison_contract", "decode": rules, "chunk": chunk_rules})
    cases = list(native.contract["cases"])
    cases += [{"id": f"real_h{h}_t{t}", "seed": 214, "parameters": {"B": 1, "T": t, "H": h, "HV": 32}}
              for h in (32, 16) for t in (1, 4, 8, 16)]
    failed = False
    for case in cases:
        x = native.decode.make_inputs(case)
        input_hashes[case["id"]] = {k: _a2_digest(v) for k, v in x.items()}
        device = native.prepare_decode(x)
        native.trace.clear()
        result = native.decode_call(device)
        unit_result = native.unit_call(native.decode, x)
        torch.npu.synchronize()
        same_as_unit = {k: _a2_digest(v) == _a2_digest(unit_result[k]) for k, v in result.items()}
        failed |= not all(same_as_unit.values())
        inputs_unchanged = input_hashes[case["id"]] == {k: _a2_digest(v) for k, v in device.items()}
        failed |= not inputs_unchanged
        cpu = a2_fp32_oracle(x)
        second = a2_fp32_oracle(device)
        old_reference = native.decode_ref.independent_reference(x)
        measured = _a2_compare(result, cpu, rules)
        failed |= not all(m["passed"] for m in measured.values())
        oracle_diff = _a2_compare(second, cpu, rules)
        failed |= not all(m["passed"] for m in oracle_diff.values())
        hashes[case["id"]] = {k: _a2_digest(v) for k, v in result.items()}
        emit({"stage": "accuracy", "case": case["id"], "parameters": case["parameters"],
              "seed": case["seed"], "inputs": input_hashes[case["id"]], "inputs_unchanged": inputs_unchanged,
              "unit_execute_bitwise": same_as_unit,
              "custom_vs_cpu_fp32": measured, "torch_npu_vs_cpu_fp32": oracle_diff,
              "custom_vs_torch_npu": _a2_compare(result, second, rules),
              "unit_fp64_vs_cpu_fp32": _a2_compare(old_reference, cpu, rules),
              "outputs": hashes[case["id"]], "launches": list(native.trace)})
        if case["id"] in ("real_h32_t16", "real_h16_t16"):
            state, parts = device["initial_state"], []
            for i in range(16):
                step = native.decode_call(native.prepare_decode(_a2_slice(x, i, i + 1)), state=state)
                state = step["final_state"]
                parts.append(step["o"].cpu())
                metric = a2_metrics(parts[-1], result["o"].cpu()[:, i:i+1], rules["o"])
                emit({"stage": "segmented_token", "case": case["id"], "token": i, "o": metric})
            segmented = {"o": torch.cat(parts, dim=1), "final_state": state.cpu()}
            same = {k: _a2_digest(segmented[k]) == _a2_digest(result[k]) for k in segmented}
            failed |= not all(same.values())
            hashes[case["id"] + "_segmented"] = {k: _a2_digest(v) for k, v in segmented.items()}
            emit({"stage": "segmented", "case": case["id"], "bitwise": same,
                  "metrics": _a2_compare(segmented, result, rules)})
        if performance and case["id"] in ("real_h32_t1", "real_h32_t16"):
            _a2_perf(native, x, emit)

    # Full prefix + suffix length must remain a multiple of the chunk size (64).
    case = {"id": "chain_128_64", "seed": 214, "parameters":
            {"B": 1, "C": 3, "H": 32, "HV": 32, "gate_multiplier": 1}}
    chunk_x = native.chunk.make_inputs(case)
    whole = native.chunk_call(chunk_x)
    prefix = native.chunk_call(_a2_slice(chunk_x, 0, 128))
    torch.npu.synchronize()
    x = {("g" if k == "g_raw" else k): v for k, v in chunk_x.items()}
    cpu_whole = a2_fp32_oracle(x)
    whole_metrics = _a2_compare(whole, cpu_whole, chunk_rules)
    failed |= not all(m["passed"] for m in whole_metrics.values())
    emit({"stage": "whole_chunk", "T": 192, "metrics": whole_metrics})
    state, parts = prefix["final_state"], []
    # Copy for the independent oracle only; the custom chain keeps the original NPU state.
    suffix = _a2_slice(x, 128, 192)
    suffix["initial_state"] = state.cpu()
    cpu_suffix = a2_fp32_oracle(suffix)
    cpu_state = suffix["initial_state"]
    for i in range(64):
        fixture = _a2_slice(suffix, i, i + 1)
        one = native.prepare_decode(fixture)
        result = native.decode_call(one, state=state)
        zero_copy = native.trace[-1]["state_input_preserved"]
        state = result["final_state"]
        cpu_step = a2_fp32_oracle({**fixture, "initial_state": cpu_state})
        cpu_state = cpu_step["final_state"]
        output = result["o"].cpu()
        parts.append(output)
        metric = a2_metrics(output, cpu_suffix["o"][:, i:i+1], rules["o"])
        whole_metric = a2_metrics(output, whole["o"].cpu()[:, i+128:i+129], chunk_rules["o"])
        state_metric = a2_metrics(state, cpu_state, rules["final_state"])
        failed |= not all((metric["passed"], whole_metric["passed"], state_metric["passed"], zero_copy))
        emit({"stage": "chunk_decode_token", "token": 128 + i, "zero_copy_state": zero_copy,
              "vs_cpu_fp32_conditioned_on_prefix": metric, "state_vs_cpu_fp32": state_metric,
              "vs_whole_chunk": whole_metric})
    tail = {"o": torch.cat(parts, dim=1), "final_state": state.cpu()}
    strict = _a2_compare(tail, cpu_suffix, rules)
    chunk_cmp = _a2_compare(tail, {"o": whole["o"].cpu()[:, 128:], "final_state": whole["final_state"]}, chunk_rules)
    failed |= not all(m["passed"] for m in (*strict.values(), *chunk_cmp.values()))
    hashes["chain"] = {k: _a2_digest(v) for k, v in tail.items()}
    emit({"stage": "chunk_decode", "prefix": 128, "single_token_steps": 64,
          "decode_vs_cpu_fp32_conditioned_on_prefix": strict, "vs_whole_chunk": chunk_cmp})
    emit({"stage": "summary", "passed": not failed, "accuracy_cases": len(cases), "outputs": hashes})
    receipt = {"schema": "a2-decode-qualification/1", "block_dim": bd, "identity": identity,
               "records": records, "outputs": hashes, "inputs": input_hashes, "passed": not failed}
    (out / f"bd{bd}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return int(failed)


def a2_compare_block_dims(left, right):
    """Byte digests include dtype/shape; reject missing cases and mismatched identities."""
    if left["block_dim"] != 1 or right["block_dim"] != 2:
        raise ValueError("cross-bd comparison requires bd1 and bd2 receipts")
    if left["identity"] != right["identity"]:
        raise ValueError("cross-bd source/environment identities differ")
    required = {f"real_h{h}_t{t}" for h in (32, 16) for t in (1, 4, 8, 16)}
    if not required <= left["outputs"].keys() or left["outputs"].keys() != right["outputs"].keys():
        raise ValueError("cross-bd receipt cases missing or different")
    if left.get("inputs") != right.get("inputs") or not required <= left.get("inputs", {}).keys():
        raise ValueError("cross-bd input identities missing or different")
    for receipt in (left, right):
        if any(set(outputs) != {"o", "final_state"} for outputs in receipt["outputs"].values()):
            raise ValueError("cross-bd receipt must contain both outputs for every case")
    comparisons = {case: left["outputs"][case] == right["outputs"][case] for case in left["outputs"]}
    return {"passed": left["passed"] and right["passed"] and all(comparisons.values()),
            "bitwise": comparisons}


def _a2_main(args):
    import json
    if args.a2_out is None:
        raise ValueError("A2 qualification requires --a2-out in a fresh ignored scratch directory")
    out = pathlib.Path(args.a2_out).resolve()
    if args.a2_worker_bd is not None:
        return _a2_worker(args.a2_worker_bd, out, args.a2_performance)
    out.mkdir(parents=True, exist_ok=True)
    for bd in (1, 2):
        command = [sys.executable, __file__, "--soc", "a2", "--a2-worker-bd", str(bd), "--a2-out", str(out)]
        if args.a2_performance:
            command.append("--a2-performance")
        with (out / f"bd{bd}.log").open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(f"A2 bd{bd} failed: preserve and inspect bd{bd}.log and bd{bd}.jsonl", flush=True)
            return result.returncode
    receipts = [json.loads((out / f"bd{bd}.json").read_text()) for bd in (1, 2)]
    comparison = a2_compare_block_dims(*receipts)
    (out / "cross-bd.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison), flush=True)
    return int(not comparison["passed"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", nargs="+", default=["acc", "chain", "bd", "split"],
                    choices=("acc", "chain", "bd", "split"))
    ap.add_argument("--block-dim", type=int, default=4)
    ap.add_argument("--_one", type=int, help="内部：bd 子进程只测这一个值")
    ap.add_argument("--soc", choices=("a5", "a2"), default="a5")
    ap.add_argument("--shapes", choices=("kimi",), default="kimi", help="A2 real shape grid")
    ap.add_argument("--a2-out", help="fresh ignored receipt directory (A2 only)")
    ap.add_argument("--a2-worker-bd", type=int, choices=(1, 2), help=argparse.SUPPRESS)
    ap.add_argument("--a2-performance", action="store_true", help="three synchronized Torch NPU sandwiches")
    args = ap.parse_args()
    if args.soc == "a2":
        return _a2_main(args)


    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("本脚本只在 NPU 上有意义", file=sys.stderr)
        return 2

    if args._one is not None:
        return _one_bd(args._one, 1)
    rc = 0
    for name in args.check:
        rc |= {"acc": check_acc, "chain": check_chain,
               "bd": check_bd, "split": check_split}[name](args.block_dim)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
