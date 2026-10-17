#!/usr/bin/env python3
"""KDA 的 torch_npu 组合基线 —— 性能基线 + 第二 oracle。

两个用途（见 AGENTS.md §6 的双 oracle）：

1. **性能基线**：用 torch_npu 原生算子拼出同语义 KDA，给 ascriptor 算子一个对照。
2. **第二 oracle**：NPU 上的结果与 CPU fp32 参考对比，差异本身就是有用信息。

用法（在有 NPU 的机器上，先 source 环境脚本）::

    python bench_kda_torch_npu.py --shape kimi_linear_layer
    python bench_kda_torch_npu.py --shape all --dtype bfloat16
    python bench_kda_torch_npu.py --shape smoke --device cpu   # 无 NPU 时自检

形状取自 ``docs/matrix/models.json`` 的 ``test_case_shapes``，不要自拟 —— 玩具形状
下多核分区与 UB 压力根本不会被触发（见 gaps.json 的 ``toy-case-shapes``）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ascend_fla.platform import CAPABILITIES, resolve_soc
from ascend_fla.reference.kda import kda_chunk_ref, kda_chunk_vectorized, kda_recurrent_ref  # noqa: E402

# 与 docs/matrix/models.json 的 test_case_shapes 对应。B/H/HV/T 之外的 K=V=128 是定尺约束。
SHAPES = {
    "smoke": dict(B=1, H=1, HV=1, T=64),
    "kimi_linear_layer": dict(B=1, H=32, HV=32, T=1024),
    "qwen3_next_layer": dict(B=1, H=16, HV=32, T=1024),
    "long_context": dict(B=1, H=16, HV=32, T=4096),
}
K_DIM = V_DIM = 128


def sync(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def make_inputs(B, H, HV, T, dtype, device, seed=2026):
    """按 ascriptor kda_fwd contract 的 input_generation 分布造输入。

    q/k/v stddev 0.04、initial state stddev 0.01、g_raw ∈ [-0.03, 0]、beta ∈ [0.05, 0.5]。
    用同分布才能让性能数和精度数落在契约声明的范围内。
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    f = dict(generator=gen, dtype=torch.float32)
    x = dict(
        q=torch.randn(B, T, H, K_DIM, **f) * 0.04,
        k=torch.randn(B, T, H, K_DIM, **f) * 0.04,
        v=torch.randn(B, T, HV, V_DIM, **f) * 0.04,
        g=torch.empty(B, T, HV, K_DIM, dtype=torch.float32).uniform_(-0.03, 0.0, generator=gen),
        beta=torch.empty(B, T, HV, dtype=torch.float32).uniform_(0.05, 0.5, generator=gen),
        initial_state=torch.randn(B, HV, K_DIM, V_DIM, **f) * 0.01,
    )
    # q/k/v 按目标 dtype 存储；g/beta/initial_state 保持 FP32（与 kda_fwd ABI 一致）
    for name in ("q", "k", "v"):
        x[name] = x[name].to(dtype)
    return {k: v.to(device) for k, v in x.items()}


def timed(fn, kwargs, device, warmup, iters):
    """同步计时。返回每次迭代的毫秒数。"""
    for _ in range(warmup):
        fn(**kwargs)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(**kwargs)
    sync(device)
    return (time.perf_counter() - t0) / iters * 1e3


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="kimi_linear_layer", choices=[*SHAPES, "all"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default="npu", choices=["npu", "cpu", "cuda"])
    ap.add_argument("--soc", choices=sorted(CAPABILITIES), help="Explicit NPU SoC; otherwise use the configured resolver")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--accuracy", action="store_true", help="同时对 CPU fp32 参考做精度比对（第二 oracle）")
    ap.add_argument("--json-out", type=pathlib.Path, help="把结果写成 json，便于进支持矩阵")
    args = ap.parse_args()

    if args.device == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            print("torch_npu 不可用；用 --device cpu 自检", file=sys.stderr)
            return 2
        if not torch.npu.is_available():
            print("torch.npu.is_available() 为 False", file=sys.stderr)
            return 2

    soc = resolve_soc(args.soc) if args.device == "npu" else None
    dtype = getattr(torch, args.dtype)
    names = list(SHAPES) if args.shape == "all" else [args.shape]

    print(f"device={args.device} soc={soc} dtype={args.dtype} warmup={args.warmup} iters={args.iters} "
          f"K=V={K_DIM} chunk=64 synchronized=yes fwd_only=yes")
    print(f"{'shape':<20} {'B/H/HV/T':<18} {'vectorized(ms)':>15} {'ref_loop(ms)':>14}  {'speedup':>8}")
    print("-" * 84)

    records = []
    for name in names:
        shp = SHAPES[name]
        x = make_inputs(**shp, dtype=dtype, device=args.device)
        dims = f"{shp['B']}/{shp['H']}/{shp['HV']}/{shp['T']}"

        kw = dict(**x, output_final_state=True)
        ms_vec = timed(kda_chunk_vectorized, kw, args.device, args.warmup, args.iters)
        # 循环版只在小形状上测，大形状下 O(BT) 的 python 循环会慢到没有参考价值
        ms_ref = float("nan")
        if shp["T"] * shp["HV"] <= 64 * 32:
            ms_ref = timed(kda_chunk_ref, kw, args.device, 1, max(1, args.iters // 5))
        speed = f"{ms_ref / ms_vec:.1f}x" if ms_ref == ms_ref else "—"
        print(f"{name:<20} {dims:<18} {ms_vec:>15.3f} {ms_ref:>14.3f}  {speed:>8}")

        rec = dict(shape=name, dims=shp, dtype=args.dtype, device=args.device, soc=soc,
                   ms_vectorized=ms_vec, ms_ref_loop=(ms_ref if ms_ref == ms_ref else None),
                   warmup=args.warmup, iters=args.iters, synchronized=True, forward_only=True)

        if args.accuracy:
            o_dev, s_dev = kda_chunk_vectorized(**kw)
            # Widen the exact device inputs so the CPU golden also returns FP32.
            x_cpu = {k: v.cpu().float() for k, v in x.items()}
            rec["cpu_reference_dtype"] = "torch.float32"
            # CPU 参考按形状选实现：循环版（kda_chunk_ref / kda_recurrent_ref）在 T=1024、
            # HV=32 下要跑上千次 python 循环，实测 wall 1.5min 就烧掉 59min CPU time。
            # 大形状只用向量化 CPU 版当参考（它已与 fla 上游 naive 对齐到 1.1e-07）。
            small = shp["T"] * shp["HV"] <= 64 * 32
            if small:
                o_cpu, s_cpu = kda_chunk_ref(**x_cpu, output_final_state=True)
                o_rec, _ = kda_recurrent_ref(**x_cpu, output_final_state=True)
                rec["rel_l2_cpu_chunk_vs_recurrent"] = rel_l2(o_cpu, o_rec)
                rec["cpu_reference"] = "kda_chunk_ref (loop)"
            else:
                o_cpu, s_cpu = kda_chunk_vectorized(**x_cpu, output_final_state=True)
                rec["cpu_reference"] = "kda_chunk_vectorized (大形状下循环版过慢)"
            rec["rel_l2_vs_cpu"] = rel_l2(o_dev.cpu(), o_cpu)
            rec["rel_l2_state_vs_cpu"] = rel_l2(s_dev.cpu(), s_cpu)
            extra = (f" | cpu chunk vs recurrent relL2={rec['rel_l2_cpu_chunk_vs_recurrent']:.3e}"
                     if "rel_l2_cpu_chunk_vs_recurrent" in rec else "")
            print(f"{'':<20} 精度[{rec['cpu_reference'].split()[0]}]: o relL2={rec['rel_l2_vs_cpu']:.3e} "
                  f"state={rec['rel_l2_state_vs_cpu']:.3e}{extra}")
        records.append(rec)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
