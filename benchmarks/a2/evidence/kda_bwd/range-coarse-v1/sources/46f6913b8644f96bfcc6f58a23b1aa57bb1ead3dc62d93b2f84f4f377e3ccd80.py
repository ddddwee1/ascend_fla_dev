#!/usr/bin/env python3
"""真实形状下的精度验收（需要 A5 NPU + 带 ``ascend950`` 算子包的 CANN）。

**为什么要有这个脚本。** 到目前为止本仓**所有**精度数字的形状都是玩具规模 ——
`kda_fwd` 接线 H=1/HV=1~2/C=1~2、`kda_bwd` 契约五 case 最大 HV2/C2、层级验证
B1/T128/H1/HV2、宽门控跨度 B1/T64。**最大的 C 是 2，最大的 H/HV 是 2。**
而性能一直在真实形状上测（`kimi_linear_layer` B1/H32/HV32/C16/T1024 等）。
于是"算子精度在预算内"这句话的依据里，没有一个点落在模型真会用的形状上。

这与刚修完的门控跨度是**同一类风险**：契约声明的域 ≠ 模型会用的域，窄域全绿不代表可用
（AGENTS.md §6"按量程失效的缺陷"）。那次是量程维度，这次是形状维度。

真实形状多出三件窄形状**结构上**测不到的东西，本脚本逐一对应一个 check：

``drift``
    C=16 意味着 chunk 间 state 传递串 16 层深。C≤2 上看不出 `final_state` 的误差
    是**随链长累积**还是**恒定**。做法：固定每 chunk 的输入分布，扫 C=1…16，看曲线形状。
    判的是趋势而不是单点 —— 单点在预算内但斜率为正，说明长序列会出问题。

``bitwise``
    H/HV=32 才真正喂满 `block_dim=4` 的核切分（contract 的 core_ownership 按 `B*HV` 与
    `B*HV*C` 切，kimi 形状下是 32 与 512；窄形状下是 1~4，连一个核组都喂不满）。
    **核切分只是把同样的算式分给不同核，bd=1 与 bd=4 的输出应当逐位相同** —— 这是个
    不需要参考的硬判据：不同就是切分 bug，不是精度问题。
    一个算子名一个进程一份 build（AGENTS.md §6 铁律二），所以本 check 自动派子进程。

``gqa``
    GQA 分组至今只跨过一组（HV/H=2 且 H=1）。qwen 形状是 H16/HV32，16 组。
    除了整体误差，还做**头独立性**检查：KDA 的头彼此独立，所以把第 r 组单独切出来
    按 H=1/HV=2 跑，结果应当与全量跑的对应切片一致。它同时验分组映射和核切分 ——
    映射错了会表现为"整体误差不大但某组明显更差"，那种错在 H=1 上根本不可能出现。

``bwd``
    反向在真实形状上的精度。参考是 fp32 逐 token 递推 + autograd，但 1024 步的图
    在 HV=32 下要十几 GB，所以**按 chunk 做 gradient checkpointing** ——
    checkpoint 是重算而不是近似，fp32 的精确性不变，峰值内存降到一个 chunk 的量级。

**实验设计：门控跨度固定在 46**（契约 case 所在的档），这样任何差异都只能归因于形状。
形状与跨度会不会交互，用 ``--span`` 另外跑一档看。

用法::

    python benchmarks/verify_real_shapes.py --check drift
    python benchmarks/verify_real_shapes.py --check bitwise          # 自动派 bd=1/bd=4 两个子进程
    python benchmarks/verify_real_shapes.py --check gqa
    python benchmarks/verify_real_shapes.py --check bwd --span 46

A2 单元资格化（先在私有配置中选定健康空闲卡，并在共享设备锁内执行）::

    python benchmarks/verify_real_shapes.py --check a2-unit --block-dim 1 \
        --a2-launcher bridge --a2-output tmp/a2-real-bd1
    python benchmarks/verify_real_shapes.py --check a2-unit --block-dim 2 \
        --a2-launcher bridge --a2-suite defaults --a2-seeds 0 1 2 3 4 5 6 7 \
        --a2-lengths 4096 --a2-output tmp/a2-defaults-bd2

A2 使用已合入的 a2.kda_fwd_stable 单元，保留公共入口的 qualified=False。
range 模式逐点保留有限性与数值失败，不把观察到的失败当成支持范围。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ascend_fla.ops.kda import prepare                                   # noqa: E402
from ascend_fla.ops.kda.chunk import (                                   # noqa: E402
    L_PER_CHUNK,
    MAX_GATE_SPAN,
    _gate_span,
    chunk_kda_fwd,
    chunk_kda_fwd_with_caches,
)
from ascend_fla.ops.kda.chunk_bwd import chunk_kda_bwd                   # noqa: E402
from ascend_fla.reference.kda import kda_recurrent_ref                   # noqa: E402

D = 128
#: 来自 docs/matrix/models.json 的 test_case_shapes —— 不在这里另造形状
SHAPES = {
    "kimi_linear_layer": dict(B=1, H=32, HV=32, C=16),
    "qwen3_next_layer": dict(B=1, H=16, HV=32, C=16),
    "long_context": dict(B=1, H=16, HV=32, C=64),
}
BUDGET = {"o": 0.05, "final_state": 0.05,
          "dq": 0.05, "dk": 0.15, "dv": 0.05, "dbeta": 0.05, "dg": 0.25, "dh0": 0.05}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def make_inputs(B, H, HV, C, span, seed=2026, *, want_grads=False):
    """在 CPU 上造一份 chunk 内门控跨度约为 ``span`` 的真实形状输入。

    每 chunk 的 g 分布固定（先造单位 g 再整体缩放），所以扫 C 时"每 chunk 有多深"不变 ——
    这正是 ``drift`` 要的控制变量：只有链长在变。
    """
    T = C * L_PER_CHUNK
    gen = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, generator=gen), dim=-1)
    g_unit = -torch.rand(B, T, HV, D, generator=gen) * 0.03
    scale = span / _gate_span(g_unit.float(), C, on_cpu=True)
    x = dict(
        q=q.bfloat16().contiguous(), k=k.bfloat16().contiguous(),
        v=(torch.randn(B, T, HV, D, generator=gen) * 0.04).bfloat16().contiguous(),
        g=(g_unit * scale).float().contiguous(),
        beta=(torch.rand(B, T, HV, generator=gen) * 0.45 + 0.05).contiguous(),
        h0=(torch.randn(B, HV, D, D, generator=gen) * 0.01).contiguous(),
    )
    if want_grads:
        x["do"] = (torch.randn(B, T, HV, D, generator=gen) * 0.04).bfloat16().contiguous()
        x["dht"] = (torch.randn(B, HV, D, D, generator=gen) * 0.01).bfloat16().contiguous()
    return x


def to_npu(x: dict) -> dict:
    return {k: v.to("npu") for k, v in x.items()}


def ref_fwd(x: dict):
    """fp32 递推参考。**必须把 v 升到 fp32 再传** —— `kda_recurrent_ref` 的返回 dtype
    跟 ``v`` 走（`reference/kda.py:76`），喂 bf16 的 v 会让"fp32 参考"自己先舍到 bf16，
    那就不是语义权威而是第二个被测对象了。
    """
    return kda_recurrent_ref(
        x["q"].float(), x["k"].float(), x["v"].float(), x["g"].float(), x["beta"].float(),
        initial_state=x["h0"].float(), output_final_state=True)


# --------------------------------------------------------------------------- drift
def check_drift(args) -> int:
    """`final_state` / `o` 的误差是随 chunk 链长累积，还是恒定？"""
    shape = dict(SHAPES[args.shapes[0]])
    shape.pop("C")
    print(f"\n=== drift：{args.shapes[0]} 的头配置 H{shape['H']}/HV{shape['HV']}，"
          f"跨度 {args.span}，bd={args.block_dim} ===")
    print("  每 chunk 的门控深度固定，只有 C 在变 —— 看的是斜率，不是单点")
    rows = []
    for C in args.cs:
        x = make_inputs(**shape, C=C, span=args.span)
        assert x["q"].shape[1] == C * L_PER_CHUNK
        dev = to_npu(x)
        try:
            o, ht = chunk_kda_fwd(dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"],
                                  initial_state=dev["h0"], output_final_state=True,
                                  block_dim=args.block_dim)
        except ValueError as exc:
            # C=1 多头被闸拦下是**预期行为**（c1-multihead-o-corrupt）。如实打印并继续，
            # 不要当成失败 —— 但也不要静静跳过，否则表里会凭空少一行而没人知道为什么。
            print(f"  C={C:<3d} T={C * L_PER_CHUNK:<5d} 被门控拒绝：{str(exc)[:60]}…")
            continue
        o_ref, ht_ref = ref_fwd(x)
        rows.append((C, rel_l2(o, o_ref), rel_l2(ht, ht_ref)))
        print(f"  C={C:<3d} T={C * L_PER_CHUNK:<5d} o={rows[-1][1]:.3e}  "
              f"final_state={rows[-1][2]:.3e}")
    if not rows:
        print("  没有一档跑成功")
        return 1
    first, last = rows[0], rows[-1]
    print(f"  C={first[0]} → C={last[0]}：o ×{last[1] / first[1]:.2f}，"
          f"final_state ×{last[2] / first[2]:.2f}")
    over = [(n, C, e) for C, eo, eh in rows
            for n, e in (("o", eo), ("final_state", eh)) if not e < BUDGET[n]]
    print("  判定：" + ("全部在预算内" if not over else f"超预算 {over}"))
    return 1 if over else 0


# ------------------------------------------------------------------------- bitwise
def _dump_for_bitwise(args) -> int:
    """子进程：按给定 block_dim 跑前向+反向，把每个输出的 sha256 与张量落盘。"""
    out = pathlib.Path(args._dump)
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in args.shapes:
        shape = SHAPES[name]
        x = make_inputs(**shape, span=args.span, want_grads=True)
        dev = to_npu(x)
        o, ht, caches = chunk_kda_fwd_with_caches(
            dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"], None, dev["h0"],
            block_dim=args.block_dim)
        got = dict(o=o, final_state=ht)
        got.update(chunk_kda_bwd(
            q=dev["q"], k=dev["k"], v=dev["v"], beta=dev["beta"].bfloat16(),
            do=dev["do"], dht=dev["dht"], caches=caches, block_dim=args.block_dim))
        for tname, t in got.items():
            cpu = t.cpu().contiguous()
            key = f"{name}.{tname}"
            digests[key] = hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            torch.save(cpu, out / f"{key}.bd{args.block_dim}.pt")
        print(f"  [bd={args.block_dim}] {name} 落盘 {len(got)} 个张量")
    (out / f"digests.bd{args.block_dim}.json").write_text(
        json.dumps(digests, indent=2), encoding="utf-8")
    return 0


def check_bitwise(args) -> int:
    """bd=1 与 bd=4 的输出应当逐位相同 —— 核切分不改算式。"""
    print(f"\n=== bitwise：bd=1 vs bd={args.max_bd}，形状 {args.shapes}，跨度 {args.span} ===")
    print("  核切分只是把同样的算式分给不同核 → 逐位相同。不同即切分 bug，不是精度问题")
    print("  一个算子名一份 build，所以两个 bd 各派一个子进程（AGENTS.md §6 铁律二）")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="afla_bitwise_"))
    try:
        for bd in (1, args.max_bd):
            cmd = [sys.executable, __file__, "--check", "bitwise", "--_dump", str(tmp),
                   "--block-dim", str(bd), "--span", str(args.span),
                   "--shapes", *args.shapes]
            r = subprocess.run(cmd, text=True, capture_output=True)
            sys.stdout.write(r.stdout)
            if r.returncode:
                sys.stderr.write(r.stderr[-3000:])
                return r.returncode
        d1 = json.loads((tmp / "digests.bd1.json").read_text())
        d4 = json.loads((tmp / f"digests.bd{args.max_bd}.json").read_text())
        assert d1.keys() == d4.keys()
        bad = []
        for key in d1:
            same = d1[key] == d4[key]
            note = "逐位相同" if same else "**不同**"
            if not same:
                a = torch.load(tmp / f"{key}.bd1.pt").float()
                b = torch.load(tmp / f"{key}.bd{args.max_bd}.pt").float()
                n_diff = int((a != b).sum())
                note += (f" 差 {n_diff}/{a.numel()} 个元素，max_abs_diff="
                         f"{(a - b).abs().max().item():.3e}，相对 L2={rel_l2(b, a):.3e}")
                bad.append(key)
            print(f"  {key:<34} {note}")
        print("  判定：" + ("全部逐位相同" if not bad else f"这些量受 block_dim 影响：{bad}"))
        return 1 if bad else 0
    finally:
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()


# ----------------------------------------------------------------------------- gqa
def check_gqa(args) -> int:
    """qwen 形状 H16/HV32：逐组误差 + 头独立性。"""
    shape = SHAPES["qwen3_next_layer"]
    H, HV, C = shape["H"], shape["HV"], shape["C"]
    groups = HV // H
    print(f"\n=== gqa：qwen 形状 H{H}/HV{HV}（{H} 个 q/k 头 × {groups} 个 v 头），"
          f"C={C}，跨度 {args.span}，bd={args.block_dim} ===")
    x = make_inputs(**shape, span=args.span)
    dev = to_npu(x)
    o, ht = chunk_kda_fwd(dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"],
                          initial_state=dev["h0"], output_final_state=True,
                          block_dim=args.block_dim)
    o_ref, ht_ref = ref_fwd(x)
    print(f"  整体：o={rel_l2(o, o_ref):.3e}  final_state={rel_l2(ht, ht_ref):.3e}")

    # 逐 v 头的误差 —— 分组映射错会表现为某些头明显更差，而不是整体变差
    o_c, oref_c = o.cpu().float(), o_ref.float()
    per_head = [rel_l2(o_c[:, :, i], oref_c[:, :, i]) for i in range(HV)]
    print(f"  逐 v 头 o 误差：min={min(per_head):.3e} max={max(per_head):.3e} "
          f"max/min={max(per_head) / max(min(per_head), 1e-30):.2f}")
    worst = sorted(range(HV), key=lambda i: -per_head[i])[:3]
    print(f"  最差三个头：{[(i, f'{per_head[i]:.3e}') for i in worst]}")

    # 头独立性：第 r 组单独按 H=1/HV=2 跑，应与全量跑的对应切片一致
    print("  头独立性（第 r 组单独跑 vs 全量跑的切片）：")
    ok = True
    for r in (0, H // 2, H - 1):
        vsl = slice(r * groups, (r + 1) * groups)
        sub = chunk_kda_fwd(
            dev["q"][:, :, r:r + 1].contiguous(), dev["k"][:, :, r:r + 1].contiguous(),
            dev["v"][:, :, vsl].contiguous(), dev["g"][:, :, vsl].contiguous(),
            dev["beta"][:, :, vsl].contiguous(),
            initial_state=dev["h0"][:, vsl].contiguous(), output_final_state=True,
            block_dim=args.block_dim)
        e_o = rel_l2(sub[0], o[:, :, vsl])
        e_h = rel_l2(sub[1], ht[:, vsl])
        # 这是"同一算式、不同并行规模"，本应逐位相同；放一个极小容差给核切分留余地
        flag = "" if max(e_o, e_h) < 1e-6 else "  ← 超出 1e-6，分组映射或核切分可疑"
        ok &= max(e_o, e_h) < 1e-6
        print(f"    组 {r:<3d} v 头 {vsl.start}..{vsl.stop - 1}：o={e_o:.3e} "
              f"final_state={e_h:.3e}{flag}")
    over = [n for n, e in (("o", rel_l2(o, o_ref)), ("final_state", rel_l2(ht, ht_ref)))
            if not e < BUDGET[n]]
    print("  判定：" + ("整体在预算内，头独立" if not over and ok
                      else f"超预算 {over}；头独立={ok}"))
    return 0 if (not over and ok) else 1


# ----------------------------------------------------------------------------- bwd
def _ref_grads_checkpointed(x: dict, C: int) -> dict:
    """fp32 递推参考的梯度，按 chunk 做 gradient checkpointing。

    1024 步的完整图在 HV=32 下要十几 GB。checkpoint 是**重算**不是近似 —— fp32 的
    精确性不变，峰值内存降到一个 chunk（64 步）的量级。
    """
    from torch.utils.checkpoint import checkpoint

    leaves = {n: x[n].clone().float().requires_grad_(True)
              for n in ("q", "k", "v", "beta", "g", "h0")}

    def seg(q, k, v, g, beta, state):
        return kda_recurrent_ref(q, k, v, g, beta, initial_state=state,
                                 output_final_state=True)

    state, outs = leaves["h0"], []
    for c in range(C):
        s = slice(c * L_PER_CHUNK, (c + 1) * L_PER_CHUNK)
        o_c, state = checkpoint(
            seg, leaves["q"][:, s], leaves["k"][:, s], leaves["v"][:, s],
            leaves["g"][:, s], leaves["beta"][:, s], state, use_reentrant=False)
        outs.append(o_c)
    o = torch.cat(outs, dim=1)
    loss = (o * x["do"].float()).sum() + (state * x["dht"].float()).sum()
    names = ("q", "k", "v", "beta", "g", "h0")
    grads = torch.autograd.grad(loss, [leaves[n] for n in names])
    return dict(zip(("dq", "dk", "dv", "dbeta", "dg", "dh0"), grads))


def check_bwd(args) -> int:
    shape = SHAPES[args.shapes[0]]
    C = shape["C"]
    print(f"\n=== bwd：{args.shapes[0]} H{shape['H']}/HV{shape['HV']}/C{C}，"
          f"跨度 {args.span}，bd={args.block_dim} ===")
    x = make_inputs(**shape, span=args.span, want_grads=True)
    got_span = _gate_span(x["g"], C, on_cpu=True)
    print(f"  实际跨度 {got_span:.2f}（闸 {MAX_GATE_SPAN['stable']['backward']}）")
    dev = to_npu(x)
    _, _, caches = chunk_kda_fwd_with_caches(
        dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"], None, dev["h0"],
        block_dim=args.block_dim)
    got = chunk_kda_bwd(q=dev["q"], k=dev["k"], v=dev["v"], beta=dev["beta"].bfloat16(),
                        do=dev["do"], dht=dev["dht"], caches=caches,
                        block_dim=args.block_dim)
    broken = [n for n in got if not got[n].cpu().float().isfinite().all()]
    if broken:
        print(f"  ✗ 这些梯度不是有限值：{broken}")
        return 1
    want = _ref_grads_checkpointed(x, C)
    errs = {n: rel_l2(got[n], want[n]) for n in want}
    print("  " + "  ".join(f"{n}={errs[n]:.3e}/{BUDGET[n]}" for n in errs))
    over = {n: e for n, e in errs.items() if not e < BUDGET[n]}
    print("  判定：" + ("全部在预算内" if not over else f"超预算 {over}"))
    return 1 if over else 0


# ---------------------------------------------------------------------- A2 unit
# A2 qualification uses the A2-03 unit directly. The public KDA entry is still
# unqualified and selects A5 definitions; this tool never changes that gate.
A2_PINS = {
    "library": "90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5",
    "kernels": "b3b3f9c16df7c4626ed3c081032a1be5a753d0b1",
    "fla": "e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2",
}


def _a2_load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _a2_unit():
    root = pathlib.Path(__file__).resolve().parents[1]
    directory = root / "kernels/projects/a2/kda_fwd_stable"
    workspace = pathlib.Path(os.environ["ASCRIPTOR_WORKSPACE"])
    runner_path = workspace / "kernels/tools/unit_runner.py"
    expected = json.loads((directory / "runner-source.json").read_text())["sha256"]
    if hashlib.sha256(runner_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("A2 unit runner does not match its owner source receipt")
    runner = _a2_load("_unit_runner", runner_path)
    previous_reference = sys.modules.get("reference")
    reference = _a2_load("reference", directory / "reference.py")
    try:
        unit = _a2_load("_a212_kda_unit", directory / "unit.py")
    finally:
        if previous_reference is None:
            sys.modules.pop("reference", None)
        else:
            sys.modules["reference"] = previous_reference
    return unit, runner, reference


def a2_metrics(actual, expected):
    """FP32 comparison; finite and in-budget are deliberately separate facts."""
    a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    if not finite:
        return dict(finite=False, relative_l2=None, max_abs_diff=None, ok=False)
    # Values are judged after FP32 conversion; FP64 norm accumulation prevents
    # a tiny nonzero error against an exact zero reference from underflowing.
    residual, denominator = float((a.double() - e.double()).norm()), float(e.double().norm())
    relative = residual / denominator if denominator else (0.0 if residual == 0 else None)
    return dict(finite=True, relative_l2=relative, max_abs_diff=float((a - e).abs().max()),
                ok=relative is not None and relative <= BUDGET["o"])


def _a2_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _a2_model_shape():
    root = pathlib.Path(__file__).resolve().parents[1]
    model = next(m for m in json.loads((root / "docs/matrix/models.json").read_text())["models"]
                 if m["id"] == "kimi-linear-48b-a3b")
    shape = model["shape"]
    assert (shape["num_key_heads"], shape["num_value_heads"],
            shape["head_k_dim"], shape["head_v_dim"]) == (32, 32, 128, 128)
    return shape


def a2_default_gate(t, seed, target=None):
    """CPU-generated layer initialization, never claimed to be trained weights.

    Use the repository's Kimi layer at the recorded real hidden/head dimensions.
    Its A_log and inverse-softplus dt initialization match the pinned FLA source.
    Calibration shifts A_log, preserving the raw-gate distribution for that seed.
    """
    from ascend_fla.layers.kda import KimiDeltaAttention

    shape = _a2_model_shape()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        layer = KimiDeltaAttention(hidden_size=shape["hidden_size"], head_dim=128,
                                   num_heads=32, num_v_heads=32, dtype=torch.float32)
        generator = torch.Generator().manual_seed(1000 + seed)
        hidden = torch.randn(1, t, shape["hidden_size"], generator=generator) * .5
        gate = layer._gate(hidden, 1, t).contiguous()
        original = _gate_span(gate, t // 64, on_cpu=True)
        if target is not None:
            if not math.isfinite(target) or target <= 0:
                raise ValueError("A2 target span must be finite and positive")
            layer.A_log.add_(math.log(target / original))
            gate = layer._gate(hidden, 1, t).contiguous()
        measured = _gate_span(gate, t // 64, on_cpu=True)
        if target is not None and not math.isclose(measured, target, rel_tol=2e-6):
            raise AssertionError(f"A_log calibration missed target: {measured} vs {target}")
        return gate, {"initialization_span": original, "measured_span": measured,
                      "target_span": target, "hidden_size": shape["hidden_size"]}


def _a2_environment(unit):
    import ascriptor
    import torch_npu

    workspace = pathlib.Path(os.environ["ASCRIPTOR_WORKSPACE"]).resolve()
    if not pathlib.Path(ascriptor.__file__).resolve().is_relative_to(workspace / "library"):
        raise RuntimeError("Actual ascriptor import is outside the selected library")
    for name, expected in A2_PINS.items():
        actual = subprocess.check_output(["git", "-C", str(workspace / name), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(workspace / name), "status", "--porcelain"], text=True)
        if actual != expected or dirty:
            raise RuntimeError(f"Selected {name} source identity is not the clean task pin")
    oracle = pathlib.Path(os.environ["FLA_KDA_NAIVE"]).resolve()
    if oracle != workspace / "fla/fla/ops/kda/naive.py":
        raise RuntimeError("Actual FLA oracle must be the selected clean pin source")
    cann = pathlib.Path(os.environ["ASCEND_HOME_PATH"])
    versions = {}
    for name in ("compiler/version.info", "opp/version.info"):
        path = cann / name
        raw = path.read_bytes()
        fields = dict(line.split("=", 1) for line in raw.decode().splitlines() if "=" in line)
        # Version/time fields are shareable; installation and host paths are not.
        versions[name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                          "fields": {k: v for k, v in fields.items()
                                     if any(tag in k.lower() for tag in ("version", "timestamp"))}}
    source_files = [pathlib.Path(unit.__file__), pathlib.Path(__file__),
                    *sorted((pathlib.Path(unit.__file__).parent / "kernels").glob("*.py"))]
    root = pathlib.Path(__file__).resolve().parents[1]
    source_files += [pathlib.Path(unit.__file__).parent / "reference.py",
                     root / "ascend_fla/reference/kda.py", root / "ascend_fla/runtime/compile.py",
                     root / "ascend_fla/runtime/binding.py", root / "benchmarks/a2/bringup.py"]
    chip = torch.npu.get_device_name(0)
    if "910B" not in chip.upper():
        raise RuntimeError("A2 qualification requires a measured 910B device")
    return {"soc": "a2", "chip": chip, "python": sys.version.split()[0],
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "repo_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "source_pins": A2_PINS, "cann": versions,
            "fla_oracle_sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
            "opp_packages": sorted(p.name for p in (cann / "opp/built-in/op_impl/ai_core/tbe/kernel").iterdir()
                                   if p.is_dir() and p.name.startswith("ascend")),
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in source_files}}


def _a2_measure_sandwich(unit, inputs, options, x, *, warmup=3, repeat=20):
    """Same-card, synchronized wall time of the two CPU-input/device-output tools.

    Both timed paths include allocation and H2D. The candidate is the existing
    five-launch unit adapter, not the unqualified public API or a kernel-only
    latency. Generation, CPU oracle, D2H comparison and compilation are outside.
    """
    from ascend_fla.reference.kda import kda_chunk_vectorized

    def candidate():
        got = unit._execute_chain(inputs, options)
        return got["o"], got["final_state"]

    def baseline():
        dev = to_npu(x)
        return kda_chunk_vectorized(
            *(dev[n] for n in ("q", "k", "v", "g", "beta")),
            initial_state=dev["h0"], output_final_state=True)

    reference = ref_fwd(x)
    checks = {}
    for name, call in (("candidate", candidate), ("baseline", baseline)):
        got = call()
        torch.npu.synchronize()
        checks[name] = {key: a2_metrics(value, expected)
                        for key, value, expected in zip(("o", "final_state"), got, reference)}
        assert all(v["ok"] for v in checks[name].values()), f"{name} timing baseline is numerically invalid"
    samples = []
    for round_id in range(3):
        for position, call in (("baseline_before", baseline), ("candidate", candidate), ("baseline_after", baseline)):
            for _ in range(warmup):
                call()
            torch.npu.synchronize()
            values = []
            for _ in range(repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                call()
                torch.npu.synchronize()
                values.append((time.perf_counter() - start) * 1000)
            samples.append({"round": round_id, "position": position, "wall_ms": values})
    return {"scope": "CPU prepared inputs to device outputs; allocations and H2D included; not public-op latency",
            "baseline": "torch_npu kda_chunk_vectorized", "synchronized": True,
            "warmup_per_phase": warmup, "repeat_per_phase": repeat, "rounds": 3,
            "correctness": checks, "samples": samples}


def check_a2_unit(args):
    """Real A2 unit validation, separate from the unqualified public op entry.

    Execute under an external shared device lease with isolated build/cache paths.
    Each block_dim is a different process. Inputs and all references are generated
    on this machine; CPU tensor preparation belongs to the test input generator.
    """
    if args.block_dim not in (1, 2):
        raise ValueError("A2-12 only qualifies block_dim 1 and 2")
    if args.a2_output is None:
        raise ValueError("--a2-output is required for retained raw receipts")
    if any(t <= 0 or t > 4096 or t % 64 for t in args.a2_lengths):
        raise ValueError("A2-12 lengths must be positive multiples of 64, at most 4096")
    if args.a2_suite == "performance" and args.a2_launcher != "bridge":
        raise ValueError("A2 timing requires the bridge; harness file transport is not timed")
    directory = pathlib.Path(args.a2_output)
    directory.mkdir(parents=True, exist_ok=True)
    unit, runner, independent = _a2_unit()
    environment = _a2_environment(unit)
    print("A2_ENVIRONMENT", json.dumps(environment), flush=True)
    (directory / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    oracle = _a2_load("_a212_fla_naive", pathlib.Path(os.environ["FLA_KDA_NAIVE"]))
    if args.a2_launcher == "bridge":
        from ascend_fla.runtime.compile import compile_kernel
        adapter = _a2_load("_a212_bridge", pathlib.Path(__file__).parent / "a2/bringup.py")
        for kernel in unit._kernels().values():
            compile_kernel(kernel, device="a2", block_dim=args.block_dim, backend="cce")
        runner.launch_kernel = adapter._bridge_launch_kernel
    else:
        adapter = None
    results = []
    for t in args.a2_lengths:
        for seed in args.a2_seeds:
            targets = [None] if args.a2_suite == "defaults" else args.a2_spans
            for span in targets:
                case_id = f"t{t}_seed{seed}_span{span}"
                gate, calibration = a2_default_gate(t, seed, span)
                x = make_inputs(1, 32, 32, t // 64, 8., seed=seed)
                x["g"] = gate
                inputs = dict(q=x["q"], k=x["k"], v=x["v"], g_raw=x["g"],
                              beta=x["beta"], initial_state=x["h0"])
                unit.validate_inputs(inputs)
                before = {n: _a2_hash(v) for n, v in inputs.items()}
                options = dict(device="a2", backend="cce", block_dim=args.block_dim,
                               launcher="aclnn", board=None, timeout=900,
                               out_dir=str(directory / "build"))
                if adapter:
                    adapter._BRIDGE_TRACE.clear()
                print("A2_CASE_START", case_id, json.dumps(calibration), flush=True)
                started = time.perf_counter()
                got = unit._execute_chain(inputs, options)
                torch.npu.synchronize()
                wall = time.perf_counter() - started
                want_o, want_h = ref_fwd(x)
                fla_o, fla_h = oracle.naive_recurrent_kda(
                    *(x[n].float() for n in ("q", "k", "v", "g", "beta")),
                    initial_state=x["h0"].float(), output_final_state=True)
                comparisons = {"o": a2_metrics(got["o"], want_o),
                               "final_state": a2_metrics(got["final_state"], want_h)}
                oracle_checks = {"o": a2_metrics(want_o, fla_o), "final_state": a2_metrics(want_h, fla_h)}
                transposed = independent.independent_reference(x)
                independent_checks = {"o": a2_metrics(transposed["o"], fla_o),
                                      "final_state": a2_metrics(transposed["final_state"], fla_h)}
                unchanged = before == {n: _a2_hash(v) for n, v in inputs.items()}
                row = dict(id=case_id, shape=[1, t, 32, 32, 128, 128], block_dim=args.block_dim,
                           input_dtype="qkv=bf16;g,beta,h0=fp32", calibration=calibration,
                           comparison=comparisons, cpu_vs_fla=oracle_checks,
                           independent_vs_fla=independent_checks, unchanged=unchanged,
                           input_sha256=before, output_sha256={n: _a2_hash(v) for n, v in got.items()},
                           finite_stages={n: bool(v.cpu().isfinite().all()) for n, v in got.items()},
                           launcher=args.a2_launcher, wall_including_allocation_copy_build_s=wall,
                           execution=adapter._BRIDGE_TRACE.copy() if adapter else options.get("_execution_evidence"))
                (directory / f"{case_id}.json").write_text(json.dumps(row, indent=2) + "\n")
                print("A2_CASE_RESULT", json.dumps(row), flush=True)
                results.append(row)
                assert unchanged, "Unit modified an input"
                assert all(v["ok"] and v["relative_l2"] <= 1e-5 for v in oracle_checks.values()), "CPU oracle disagreement"
                assert all(v["ok"] and v["relative_l2"] <= 1e-5 for v in independent_checks.values()), "Independent transpose-state oracle disagreement"
                if args.a2_suite != "range":
                    assert all(v["ok"] for v in comparisons.values()), "A2 numeric acceptance failed"
                if args.a2_suite == "performance":
                    row["timing"] = _a2_measure_sandwich(unit, inputs, options, x)
                    assert before == {n: _a2_hash(v) for n, v in inputs.items()}, "Timing mutated inputs"
                    (directory / f"{case_id}.json").write_text(json.dumps(row, indent=2) + "\n")
                    print("A2_TIMING", case_id, json.dumps(row["timing"]), flush=True)
    (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", required=True,
                    choices=("drift", "bitwise", "gqa", "bwd", "a2-unit"))
    ap.add_argument("--block-dim", type=int, default=4)
    ap.add_argument("--max-bd", type=int, default=4, help="bitwise 对比的上限 bd")
    ap.add_argument("--span", type=float, default=46.0,
                    help="门控跨度。默认 46 = 契约 case 所在的档，固定它才能把差异归因到形状")
    ap.add_argument("--shapes", nargs="+", default=["kimi_linear_layer"],
                    choices=list(SHAPES))
    ap.add_argument("--cs", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="drift 扫的 C 列表。C=1 会被 C=1 多头闸拦下（那是预期的）")
    ap.add_argument("--_dump", help="内部：bitwise 子进程的落盘目录")
    ap.add_argument("--a2-output", help="A2 private output/build directory; one directory per block_dim")
    ap.add_argument("--a2-launcher", choices=("aclnn", "bridge"), default="aclnn")
    ap.add_argument("--a2-suite", choices=("real", "defaults", "range", "performance"), default="real")
    ap.add_argument("--a2-lengths", type=int, nargs="+", default=[4096, 128, 256, 512, 64])
    ap.add_argument("--a2-seeds", type=int, nargs="+", default=[2026])
    ap.add_argument("--a2-spans", type=float, nargs="+", default=[8.])
    args = ap.parse_args()

    if args.check == "a2-unit":
        return check_a2_unit(args)

    try:
        import torch_npu  # noqa: F401  —— 注册 "npu" 设备类型，不导入 .to("npu") 就报错
    except ImportError:
        print("本脚本只在 NPU 上有意义", file=sys.stderr)
        return 2

    if args._dump:
        prepare("a5", args.block_dim, backward=True)
        return _dump_for_bitwise(args)
    if args.check == "bitwise":
        return check_bitwise(args)

    prepare("a5", args.block_dim, backward=(args.check == "bwd"))
    return {"drift": check_drift, "gqa": check_gqa, "bwd": check_bwd}[args.check](args)


if __name__ == "__main__":
    raise SystemExit(main())
