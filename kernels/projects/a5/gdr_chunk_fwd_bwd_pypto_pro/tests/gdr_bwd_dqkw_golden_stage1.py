#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""Staging golden for gdr_bwd_dqkw: the whole operator in one stage.

Pure torch, self-contained, and numerically identical to
gdr_bwd_dqkw_golden_cpu.py -- the operator is a single fused cube and vector
program, so its cumulative staging golden is also its complete golden.

Signature:
    (grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, g_f32, decay)
Returns (dq_p, dk_p, dw, dg_p), all float32.
"""

import torch
import torch.nn.functional as F

_C = 64


def gdr_bwd_dqkw_golden_stage1(
    grad_o: torch.Tensor,
    v_new: torch.Tensor,
    k_f32: torch.Tensor,
    q_s: torch.Tensor,
    s_hist: torch.Tensor,
    dh: torch.Tensor,
    dv2: torch.Tensor,
    g_f32: torch.Tensor,
    decay: torch.Tensor,
):
    C = _C

    do_pub = grad_o.to(torch.float32)
    v_new = v_new.to(torch.float32)
    k_f32 = k_f32.to(torch.float32)
    q_s = q_s.to(torch.float32)
    s_hist = s_hist.to(torch.float32)
    dh = dh.to(torch.float32)
    dv2 = dv2.to(torch.float32)
    decay = decay.to(torch.float32)
    _ = g_f32  # parent-fidelity contract input; unused by the core math

    B, T, H, V = do_pub.shape
    N, K = v_new.shape[2], k_f32.shape[-1]
    scale = 1.0 / (K ** 0.5)

    do = do_pub.transpose(1, 2).contiguous()
    pad = (C - (T % C)) % C
    if pad > 0:
        do = F.pad(do, (0, 0, 0, pad))
    do = do.reshape(B, H, N, C, V)

    q_raw = q_s / scale

    decay_exp = decay.exp()
    g_last = decay[..., -1]

    ds = torch.matmul(do, v_new.transpose(-1, -2))
    gdiff = decay.unsqueeze(-1) - decay.unsqueeze(-2)
    ds = ds * gdiff.exp() * scale
    up_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    ds = ds.masked_fill(up_strict, 0.0)

    h_T = s_hist.transpose(-1, -2)
    dh_T = dh.transpose(-1, -2)

    dq_inter = torch.matmul(do, h_T) * decay_exp[..., None] * scale
    dq_intra = torch.matmul(ds, k_f32)
    dq_p = dq_inter + dq_intra

    dk_inter = torch.matmul(v_new, dh_T)
    gate_k = (g_last[..., None] - decay).exp()
    dk_inter = dk_inter * gate_k[..., None]
    dk_intra = torch.matmul(ds.transpose(-1, -2), q_raw)
    dk_p = dk_inter + dk_intra

    dw = -torch.matmul(dv2, h_T)

    dg_last = g_last.exp() * (s_hist * dh).sum(dim=(-2, -1)) \
        + (dk_inter * k_f32).sum(dim=(-2, -1))

    dg_p = (dq_p * q_raw).sum(-1) - (dk_p * k_f32).sum(-1)
    dg_corr = F.pad(dg_last.unsqueeze(-1), (C - 1, 0))
    dg_p = dg_p + dg_corr

    return (dq_p.to(torch.float32), dk_p.to(torch.float32),
            dw.to(torch.float32), dg_p.to(torch.float32))


_P0 = [
    ("aligned_b1_t128_h4_k64_v64", 1, 128, 4, 64, 64),
    ("ttail_b1_t100_h4_k64_v64", 1, 100, 4, 64, 64),
    ("t3c_b1_t192_h2_k128_v128", 1, 192, 2, 128, 128),
    ("tail_vlt_b1_t200_h3_k128_v96", 1, 200, 3, 128, 96),
]
_SEEDS = [42, 123, 456]
_OUT_NAMES = ("dq_p", "dk_p", "dw", "dg_p")


def _upstream():
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))

    def _load(subdir, modname):
        # the upstream CPU goldens are flat siblings of this file
        path = os.path.join(here, modname + ".py")
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    wy = _load("gdr_fwd_train_wy", "gdr_fwd_train_wy_golden_cpu")
    scan = _load("gdr_fwd_train_scan", "gdr_fwd_train_scan_golden_cpu")
    dvl = _load("gdr_bwd_dv_local", "gdr_bwd_dv_local_golden_cpu")
    rev = _load("gdr_bwd_reverse_scan", "gdr_bwd_reverse_scan_golden_cpu")
    return wy, scan, dvl, rev


def _build_op5_inputs(B, T, H, K, V, seed, g_lo=-4.0, g_hi=0.0):
    torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(g_lo, g_hi)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)

    wy, scan, dvl, rev = _upstream()
    (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
     dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t) = wy.gdr_fwd_train_wy_golden_cpu(q, k, v, g, beta)
    o, s_hist, v_new = scan.gdr_fwd_train_scan_golden_cpu(u, w, q_de, a_mat, k_f32, dfac, decay_exp)
    dv_local = dvl.gdr_bwd_dv_local_golden_cpu(grad_o, k_f32, q_t, g_f32)
    dh, dv2 = rev.gdr_bwd_reverse_scan_golden_cpu(q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local)

    inputs = [grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, g_f32, decay]
    return inputs


def _load_golden_cpu():
    import importlib.util
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "gdr_bwd_dqkw_golden_cpu.py")
    spec = importlib.util.spec_from_file_location("gdr_bwd_dqkw_golden_cpu", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _validate():
    print("=" * 70)
    print("gdr_bwd_dqkw_golden_stage1 验证报告 (Module 1, TERMINAL == full op)")
    print("=" * 70)
    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    print("\n[1. sanity: shape / dtype / finite over P0]")
    for name, B, T, H, K, V in _P0:
        N = (T + 63) // 64
        exp = {"dq_p": (B, H, N, 64, K), "dk_p": (B, H, N, 64, K),
               "dw": (B, H, N, 64, K), "dg_p": (B, H, N, 64)}
        args = _build_op5_inputs(B, T, H, K, V, seed=42)
        outs = gdr_bwd_dqkw_golden_stage1(*args)
        ok = _check(f"{name}: 4 outputs", len(outs) == 4)
        for nm, t in zip(_OUT_NAMES, outs):
            ok &= (tuple(t.shape) == exp[nm]) and (t.dtype == torch.float32) \
                and bool(torch.isfinite(t).all())
        _check(f"{name}: shapes/dtype(fp32)/finite", ok)

    print("\n[2. HARD 验收: 精确复现 gdr_bwd_dqkw_golden_cpu.py]")
    gc = _load_golden_cpu()
    worst = {nm: 0.0 for nm in _OUT_NAMES}
    for name, B, T, H, K, V in _P0:
        for seed in _SEEDS:
            args = _build_op5_inputs(B, T, H, K, V, seed)
            outs_s = gdr_bwd_dqkw_golden_stage1(*args)
            outs_r = gc.gdr_bwd_dqkw_golden_cpu(*args)
            errs = {}
            ok = True
            for nm, s, r in zip(_OUT_NAMES, outs_s, outs_r):
                d = (s - r).abs().max().item()
                errs[nm] = d
                worst[nm] = max(worst[nm], d)
                ok = ok and (d == 0.0)
            _check(f"{name} s{seed}: " + " ".join(f"{nm}={errs[nm]:.2e}" for nm in _OUT_NAMES), ok)
    print("    worst across all shapes/seeds: "
          + " ".join(f"{nm}={worst[nm]:.3e}" for nm in _OUT_NAMES))

    print("\n" + "=" * 70)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
