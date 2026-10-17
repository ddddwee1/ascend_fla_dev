#!/usr/bin/env python3
"""KDA 训练步（fwd+bwd）的耗时，对照 torch_npu 组合版 —— 第二期的性能数。

拆成四档，好看出代价花在哪：

1. ``fwd`` —— 只前向、不产检查点、**不做门控检查**（与第一期的数可比）。门控检查的
   代价单独列一项。
2. ``fwd+caches`` —— 前向并产出九个反向检查点。与 ① 的差额就是补
   ``g_cumsum`` / ``h`` / ``v_new`` 的代价（其中 ``h`` / ``v_new`` 是 host 侧 C 次
   迭代的 torch 递推，见 ``docs/matrix/gaps.json`` 的 ``fwd-caches-not-emitted``）。
3. ``fwd+bwd`` —— 完整一步训练（:func:`chunk_kda` 的 autograd）。与 ② 的差额是九个
   反向 kernel 本身。
4. ``torch_npu fwd+bwd`` —— 同语义的 torch_npu 组合版 + autograd，作为对照基线。

报数纪律见 AGENTS.md §6：形状、dtype、是否含 bwd、warmup 与重复次数、是否同步，
都要声明。**一个进程只用一个 block_dim**（同名算子多 build 会互相覆盖），多值时自动
派子进程。

用法::

    python bench_kda_train_step.py --shape kimi_linear_layer --block-dim 4
    python bench_kda_train_step.py --shape all --block-dim 1 4 --json-out out.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import statistics
import sys
import tempfile
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ascend_fla.ops.kda import (  # noqa: E402
    chunk_kda,
    chunk_kda_fwd,
    chunk_kda_fwd_with_caches,
    prepare,
)
from ascend_fla.platform import CAPABILITIES, resolve_soc  # noqa: E402
from ascend_fla.reference.kda import kda_chunk_vectorized  # noqa: E402
from benchmarks.bench_kda_torch_npu import SHAPES, make_inputs  # noqa: E402


def timed(fn, warmup: int, iters: int) -> float:
    """同步计时，返回每次迭代的毫秒数。warmup 吃掉首次编译与分配。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _grad_inputs(x: dict) -> dict:
    """把需要梯度的张量换成 leaf。q/k/v 是 bf16，g/beta/initial_state 是 fp32。"""
    return {n: t.clone().requires_grad_(True) for n, t in x.items()}


def run_shape(name: str, block_dim: int, warmup: int, iters: int,
              soc: str | None = None, sandwich_rounds: int = 3) -> dict:
    # 必须先 prepare：本脚本第一档是纯前向（它不会预编译反向链），等跑到带缓存那一档
    # 再注册反向的 vendor 树就晚了 —— CANN 只在首次算子解析时读 ASCEND_CUSTOM_OPP_PATH。
    # 不调 prepare 会被 runtime/binding.py 的门控拦住并报错，而不是静默失效。
    soc = resolve_soc(soc)
    prepare(soc, block_dim, backward=True)
    shp = SHAPES[name]
    x = make_inputs(**shp, dtype=torch.bfloat16, device="npu")
    common = dict(device=soc, block_dim=block_dim, layout_device="npu")

    def train_step():
        leaves = _grad_inputs(x)
        o, state = chunk_kda(**leaves, output_final_state=True, **common)
        (o.float().sum() + state.float().sum()).backward()

    def ref_step():
        leaves = _grad_inputs(x)
        o, state = kda_chunk_vectorized(**leaves, output_final_state=True)
        (o.float().sum() + state.float().sum()).backward()

    rounds = []
    for index in range(sandwich_rounds):
        before = timed(ref_step, warmup, iters)
        fwd = timed(lambda: chunk_kda_fwd(**x, output_final_state=True,
                                         check_gate_range=False, **common), warmup, iters)
        checked = timed(lambda: chunk_kda_fwd(**x, output_final_state=True, **common), warmup, iters)
        cached = timed(lambda: chunk_kda_fwd_with_caches(**x, **common), warmup, iters)
        train = timed(train_step, warmup, iters)
        after = timed(ref_step, warmup, iters)
        rounds.append(dict(round=index+1, baseline_before_ms=before, baseline_after_ms=after,
                           fwd_ms=fwd, checked_fwd_ms=checked, cached_fwd_ms=cached,
                           train_ms=train, speedup=((before+after)/2)/train))
    ms_fwd = statistics.median(r['fwd_ms'] for r in rounds)
    ms_fwd_checked = statistics.median(r['checked_fwd_ms'] for r in rounds)
    ms_caches = statistics.median(r['cached_fwd_ms'] for r in rounds)
    ms_train = statistics.median(r['train_ms'] for r in rounds)
    ms_ref = statistics.median((r['baseline_before_ms']+r['baseline_after_ms'])/2 for r in rounds)

    return dict(
        shape=name, dims=shp, soc=soc, block_dim=block_dim, dtype="bfloat16",
        sandwich_rounds=rounds, timing_summary="median across synchronized rounds",
        ms_fwd=ms_fwd, ms_fwd_with_gate_check=ms_fwd_checked,
        ms_fwd_with_caches=ms_caches, ms_fwd_bwd=ms_train,
        ms_torch_npu_fwd_bwd=ms_ref,
        gate_check_ms=ms_fwd_checked - ms_fwd,
        cache_overhead_ms=ms_caches - ms_fwd,
        bwd_only_ms=ms_train - ms_caches,
        speedup_vs_torch_npu=ms_ref / ms_train,
        warmup=warmup, iters=iters, synchronized=True, layout_device="npu",
    )


def _print_table(rows: list[dict]) -> None:
    print(f"{'shape':<20} {'bd':>3} {'fwd':>8} {'+caches':>9} {'fwd+bwd':>9} "
          f"{'torch_npu':>10} {'加速':>7}")
    print("-" * 74)
    for r in rows:
        print(f"{r['shape']:<20} {r['block_dim']:>3} {r['ms_fwd']:>8.3f} "
              f"{r['ms_fwd_with_caches']:>9.3f} {r['ms_fwd_bwd']:>9.3f} "
              f"{r['ms_torch_npu_fwd_bwd']:>10.3f} {r['speedup_vs_torch_npu']:>6.2f}x")
        share = r["cache_overhead_ms"] / r["ms_fwd_bwd"] * 100
        print(f"{'':<20} {'':>3} 其中 检查点 {r['cache_overhead_ms']:.3f}ms"
              f"（占 fwd+bwd 的 {share:.0f}%）  反向 kernel {r['bwd_only_ms']:.3f}ms"
              f"  门控检查 {r['gate_check_ms']:.3f}ms")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="all", choices=[*SHAPES, "all"])
    ap.add_argument("--block-dim", type=int, nargs="+", default=[1])
    ap.add_argument("--soc", choices=sorted(CAPABILITIES), help="Explicit SoC; otherwise use the configured resolver")
    ap.add_argument("--sandwich-rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--json-out", type=pathlib.Path)
    args = ap.parse_args()
    if args.sandwich_rounds < 1 or args.iters < 1 or args.warmup < 0:
        ap.error("rounds/iters must be positive and warmup nonnegative")

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("本脚本只在 NPU 上有意义", file=sys.stderr)
        return 2
    if not torch.npu.is_available():
        print("torch.npu.is_available() 为 False", file=sys.stderr)
        return 2
    soc = resolve_soc(args.soc)
    supported = CAPABILITIES[soc]["supported_block_dim"].get("chunk", ())
    bad = [b for b in args.block_dim if b not in supported]
    if bad:
        print(f"block_dim {bad} 超出契约声明的 {list(supported)} ({soc})", file=sys.stderr)
        return 2

    names = list(SHAPES) if args.shape == "all" else [args.shape]
    print(f"soc={soc} dtype=bfloat16 K=V=128 chunk=64 warmup={args.warmup} iters={args.iters} "
          f"synchronized=yes layout_device=npu 含反向=yes")

    if len(args.block_dim) > 1:
        rows = []
        with tempfile.TemporaryDirectory() as td:
            for bd in args.block_dim:
                out = pathlib.Path(td) / f"bd{bd}.json"
                r = subprocess.run(
                    [sys.executable, str(pathlib.Path(__file__).resolve()),
                     "--shape", args.shape, "--block-dim", str(bd), "--soc", soc,
                     "--sandwich-rounds", str(args.sandwich_rounds),
                     "--warmup", str(args.warmup), "--iters", str(args.iters),
                     "--json-out", str(out)],
                    capture_output=True, text=True)
                if r.returncode != 0 or not out.is_file():
                    print(f"block_dim={bd} 的子进程失败 exit={r.returncode}", file=sys.stderr)
                    print(r.stdout[-2000:], file=sys.stderr)
                    print(r.stderr[-2000:], file=sys.stderr)
                    return 1
                rows += json.loads(out.read_text(encoding="utf-8"))
    else:
        rows = [run_shape(n, args.block_dim[0], args.warmup, args.iters, soc, args.sandwich_rounds) for n in names]

    _print_table(rows)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"\n已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
