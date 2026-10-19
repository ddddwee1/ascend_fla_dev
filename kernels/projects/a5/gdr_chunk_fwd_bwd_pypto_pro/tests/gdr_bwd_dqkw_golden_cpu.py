#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_dqkw (OP5) CPU golden reference (higher precision, FP32).

Ultimate ground truth for precision comparison in later stages. Same math as the NPU
golden (gdr_bwd_dqkw_golden.py) but pure CPU, no torch_npu / no device selection; all
arithmetic and outputs FP32.

MECHANICAL PROJECTION (a faithful M8 slice), NOT fresh math: it computes exactly the M8
arithmetic (golden_step b3, the chunk-parallel dqkwg backward) of the READ-ONLY delivered
cumulative staging golden

    gated_delta_rule_bwd_golden_stage12345678.py  (_m8)

but consumes OP5's frozen GM contract directly.

──────────────────────────────────────────────────────────────────────────────
WHERE EACH FACTOR IS APPLIED, matched
against the AUTHORITATIVE `_m8` (not the one-line summary):
  * scale=1/sqrt(K) : folded into ds AND into dq_inter; nowhere else.
  * decay_exp = exp(decay) : RECOMPUTED internally from the `decay` GM input; NOT a GM
    input. Used only in dq_inter.
  * exp(g_last - decay) gate on dk_inter : g_last = decay[..., -1].
  * dg_last last-row correction (token C-1) : exp(g_last)*sum_KV(h*dh) + sum_KV(dk_inter*
    k_f32), dk_inter being the ALREADY-GATED inter term.
  * raw g_f32 : NOT read by the core math (g_last from decay[...,-1], decay_exp from
    decay). Carried in the GM input set for parent fidelity only. CONFIRMED vs `_m8`.
  * q operand : `_m8` uses RAW (unscaled) q in dk_intra and the dg reduce. OP5's GM input
    is q_s = q*scale (scale folded in by upstream gdr_fwd_train_wy M1). This golden RECOVERS
    q_raw = q_s / scale internally. Feeding q_s directly breaks dk/dg (relL2 ~0.9);
    recovering q_raw matches `_m8` to fp32.

Signature:
    gdr_bwd_dqkw_golden_cpu(grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, g_f32, decay)
        -> (dq_p, dk_p, dw, dg_p)   all fp32
    grad_o : [B, T, H, V]     (any float dtype; promoted to fp32; PUBLIC layout)
    v_new  : [B, H, N, C, V]  fp32   (delta-corrected value; from gdr_fwd_train_scan)
    k_f32  : [B, H, N, C, K]  fp32   (raw k)
    q_s    : [B, H, N, C, K]  fp32   (SCALED q = q*scale; scale recovered internally)
    s_hist : [B, H, N, K, V]  fp32   (per-chunk entering state; s_hist[:,:,0]=0)
    dh     : [B, H, N, K, V]  fp32   (reverse-scan entering dS; dh[:,:,N-1]=0)
    dv2    : [B, H, N, C, V]  fp32
    g_f32  : [B, H, N, C]     fp32   (raw pre-cumsum gate; carried for parent fidelity, UNUSED)
    decay  : [B, H, N, C]     fp32   (within-chunk cumulative gate)
    ->  dq_p, dk_p, dw : [B, H, N, C, K]  fp32   (NOT narrowed)
        dg_p           : [B, H, N, C]     fp32   (NOT narrowed)

All arithmetic fp32. BHN-parallel, stateless. Padding rows carry g=0 (inert). No [:T]
slice here (that is OP7's b7).
"""

import os

import torch
import torch.nn.functional as F

_C = 64  # fixed internal chunk size


def gdr_bwd_dqkw_golden_cpu(
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
    """CPU FP32 golden for OP5 (M8 dqkwg, faithful slice of the parent b3 step).

    Returns (dq_p, dk_p, dw, dg_p), all fp32.  g_f32 is accepted per contract but UNUSED.
    """
    C = _C

    do_pub = grad_o.to(torch.float32)
    v_new = v_new.to(torch.float32)
    k_f32 = k_f32.to(torch.float32)
    q_s = q_s.to(torch.float32)
    s_hist = s_hist.to(torch.float32)
    dh = dh.to(torch.float32)
    dv2 = dv2.to(torch.float32)
    decay = decay.to(torch.float32)
    _ = g_f32  # parent-fidelity contract input; not used by the core math

    B, T, H, V = do_pub.shape
    N, K = v_new.shape[2], k_f32.shape[-1]
    scale = 1.0 / (K ** 0.5)

    # grad_o public [B,T,H,V] -> chunked do [B,H,N,C,V] (pad T -> N*C).
    do = do_pub.transpose(1, 2).contiguous()                          # [B, H, T, V]
    pad = (C - (T % C)) % C
    if pad > 0:
        do = F.pad(do, (0, 0, 0, pad))
    do = do.reshape(B, H, N, C, V)

    # recover raw q (q_s = q*scale upstream; _m8 uses raw q for dk_intra / dg reduce).
    q_raw = q_s / scale

    decay_exp = decay.exp()
    g_last = decay[..., -1]

    # ds = tril_incl_diag( (do @ v_new^T) * exp(decay_i - decay_j) * scale )
    ds = torch.matmul(do.float(), v_new.transpose(-1, -2).float())
    gdiff = decay.unsqueeze(-1) - decay.unsqueeze(-2)                # [.,i,j] = decay_i - decay_j
    ds = ds * gdiff.exp() * scale
    up_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    ds = ds.masked_fill(up_strict, 0.0)                             # keep lower incl diag (i>=j)

    h_T = s_hist.transpose(-1, -2)                                  # [B,H,N,V,K]
    dh_T = dh.transpose(-1, -2)                                     # [B,H,N,V,K]

    # dq_p = (do @ h) * decay_exp * scale + ds @ k
    dq_inter = torch.matmul(do.float(), h_T.float()) * decay_exp[..., None] * scale
    dq_intra = torch.matmul(ds.float(), k_f32.float())
    dq_p = dq_inter + dq_intra

    # dk_p = (v_new @ dh) * exp(g_last - decay) + ds^T @ q_raw
    dk_inter = torch.matmul(v_new.float(), dh_T.float())
    gate_k = (g_last[..., None] - decay).exp()
    dk_inter = dk_inter * gate_k[..., None]
    dk_intra = torch.matmul(ds.transpose(-1, -2).float(), q_raw.float())
    dk_p = dk_inter + dk_intra

    # dw = -(dv2 @ h)
    dw = -torch.matmul(dv2.float(), h_T.float())

    # dg_last scalar per chunk = exp(g_last)*sum_KV(h*dh) + sum_KV(dk_inter*k)
    dg_last = g_last.exp() * (s_hist * dh).sum(dim=(-2, -1)) \
        + (dk_inter * k_f32).sum(dim=(-2, -1))

    # dg_p per token = sum_K(dq_p*q_raw) - sum_K(dk_p*k); dg_last folded onto row C-1.
    dg_p = (dq_p * q_raw).sum(-1) - (dk_p * k_f32).sum(-1)
    dg_corr = F.pad(dg_last.unsqueeze(-1), (C - 1, 0))
    dg_p = dg_p + dg_corr

    return (dq_p.to(torch.float32), dk_p.to(torch.float32),
            dw.to(torch.float32), dg_p.to(torch.float32))


# ==========================================
# 验证
# ==========================================

_P0 = [
    ("aligned_b1_t128_h4_k64_v64", 1, 128, 4, 64, 64),
    ("ttail_b1_t100_h4_k64_v64", 1, 100, 4, 64, 64),
    ("t3c_b1_t192_h2_k128_v128", 1, 192, 2, 128, 128),
    ("tail_vlt_b1_t200_h3_k128_v96", 1, 200, 3, 128, 96),
]
_SEEDS = [42, 123, 456]
_OUT_NAMES = ("dq_p", "dk_p", "dw", "dg_p")


def _expected_shapes(B, T, H, K, V):
    N = (T + 63) // 64
    C = 64
    return {"dq_p": (B, H, N, C, K), "dk_p": (B, H, N, C, K),
            "dw": (B, H, N, C, K), "dg_p": (B, H, N, C)}


def _upstream():
    """Import the upstream CPU goldens (torch-only) by file path."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    custom_root = os.path.dirname(here)

    def _load(subdir, modname):
        path = os.path.join(custom_root, subdir, modname + ".py")
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    wy = _load("gdr_fwd_train_wy", "gdr_fwd_train_wy_golden_cpu")
    scan = _load("gdr_fwd_train_scan", "gdr_fwd_train_scan_golden_cpu")
    dvl = _load("gdr_bwd_dv_local", "gdr_bwd_dv_local_golden_cpu")
    rev = _load("gdr_bwd_reverse_scan", "gdr_bwd_reverse_scan_golden_cpu")
    m8_path = os.path.join(custom_root, "gated_delta_rule_bwd", "modules",
                           "gated_delta_rule_bwd_golden_stage12345678.py")
    spec = importlib.util.spec_from_file_location("gated_delta_rule_bwd_golden_stage12345678", m8_path)
    m8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m8)
    return wy, scan, dvl, rev, m8


def _build_op5_inputs(B, T, H, K, V, seed, g_lo=-4.0, g_hi=0.0, zero_grad_o=False):
    """Build the 9 OP5 GM inputs jointly from a common primary (all CPU fp32/bf16)."""
    torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    if zero_grad_o:
        grad_o = torch.zeros_like(grad_o)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(g_lo, g_hi)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)

    wy, scan, dvl, rev, _ = _upstream()
    (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
     dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t) = wy.gdr_fwd_train_wy_golden_cpu(q, k, v, g, beta)
    o, s_hist, v_new = scan.gdr_fwd_train_scan_golden_cpu(u, w, q_de, a_mat, k_f32, dfac, decay_exp)
    dv_local = dvl.gdr_bwd_dv_local_golden_cpu(grad_o, k_f32, q_t, g_f32)
    dh, dv2 = rev.gdr_bwd_reverse_scan_golden_cpu(q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local)

    inputs = [grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, g_f32, decay]
    prim = dict(q=q, k=k, v=v, g=g, beta=beta, grad_o=grad_o)
    return inputs, prim


def _rel_l2(a, b):
    return (torch.linalg.vector_norm(a - b)
            / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()


def _validate():
    print("=" * 68)
    print("gdr_bwd_dqkw_golden_cpu 验证报告 (OP5, M8/b3 mechanical projection, CPU FP32)")
    print("=" * 68)

    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    # 1. Contract P0: shape / fp32 / finite.
    print("\n[1. 合同 P0 用例: shape / fp32 / finite]")
    for name, B, T, H, K, V in _P0:
        exp = _expected_shapes(B, T, H, K, V)
        args, _ = _build_op5_inputs(B, T, H, K, V, seed=42)
        outs = gdr_bwd_dqkw_golden_cpu(*args)
        _check(f"{name}: returns 4 outputs", len(outs) == 4)
        for nm, t in zip(_OUT_NAMES, outs):
            _check(f"{name}.{nm}: shape {tuple(t.shape)} == {exp[nm]}", tuple(t.shape) == exp[nm])
            _check(f"{name}.{nm}: dtype fp32", t.dtype == torch.float32)
            _check(f"{name}.{nm}: finite", torch.isfinite(t).all().item())

    # 2. MANDATORY five-operator cross-check (a mismatch is a GM-CONTRACT DEFECT):
    #    real gdr_fwd_train_wy + gdr_fwd_train_scan (v_new,s_hist) + gdr_bwd_dv_local +
    #    gdr_bwd_reverse_scan (dh,dv2) -> OP5 projection, vs the parent M8 staging golden
    #    gated_delta_rule_bwd_golden_stage12345678 from the SAME primary. All four outputs
    #    must agree to fp32 tolerance across 4 shapes x seeds 42/123/456.
    print("\n[2. 五算子交叉验证 (OP5 投影 vs 父 M8 staging golden stage12345678)]")
    _, _, _, _, m8 = _upstream()
    CROSS_TOL = 2e-2  # rel-L2 acceptance criterion
    worst = {nm: 0.0 for nm in _OUT_NAMES}
    for name, B, T, H, K, V in _P0:
        for seed in _SEEDS:
            args, prim = _build_op5_inputs(B, T, H, K, V, seed)
            outs = gdr_bwd_dqkw_golden_cpu(*args)
            ref = m8.gated_delta_rule_bwd_golden_stage12345678(
                prim["grad_o"], prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
            errs, ok = {}, True
            for nm, o, r in zip(_OUT_NAMES, outs, ref):
                e = _rel_l2(o, r)
                errs[nm] = e
                worst[nm] = max(worst[nm], e)
                ok = ok and (e <= CROSS_TOL) and torch.isfinite(o).all().item()
            _check(f"{name} s{seed}: " + " ".join(f"{nm}={errs[nm]:.2e}" for nm in _OUT_NAMES)
                   + f" (<= {CROSS_TOL:.0e})", ok)
    print("    worst rel-L2 across all shapes/seeds: "
          + " ".join(f"{nm}={worst[nm]:.3e}" for nm in _OUT_NAMES))

    # 3. Numerical edges.
    print("\n[3. 数值边界]")
    args_z, _ = _build_op5_inputs(1, 128, 2, 64, 64, seed=7, zero_grad_o=True)
    outs_z = gdr_bwd_dqkw_golden_cpu(*args_z)
    _check("grad_o == 0 (end-to-end) -> all four outputs == 0",
           all(t.abs().max().item() == 0.0 for t in outs_z))
    args_e, _ = _build_op5_inputs(1, 200, 3, 128, 96, seed=42)
    dh_in = args_e[5]
    N_ = dh_in.shape[2]
    _check("dh[:,:,N-1] == 0 (reverse-scan first)", dh_in[:, :, N_ - 1].abs().max().item() == 0.0)
    args_n, _ = _build_op5_inputs(1, 128, 2, 64, 64, seed=3, g_lo=-16.0, g_hi=-8.0)
    outs_n = gdr_bwd_dqkw_golden_cpu(*args_n)
    _check("very-negative g (-16,-8): all outputs finite",
           all(torch.isfinite(t).all().item() for t in outs_n))

    # 4. bf16 & fp32 grad_o both accepted; output always fp32.
    print("\n[4. bf16 与 fp32 grad_o -> fp32 输出]")
    for tag in ("bf16", "fp32"):
        args, _ = _build_op5_inputs(1, 128, 4, 64, 64, seed=11)
        if tag == "fp32":
            args[0] = args[0].to(torch.float32)
        outs = gdr_bwd_dqkw_golden_cpu(*args)
        _check(f"{tag} grad_o: all outputs fp32", all(t.dtype == torch.float32 for t in outs))
        _check(f"{tag} grad_o: all outputs finite", all(torch.isfinite(t).all().item() for t in outs))

    # 5. Magnitude ranges (downstream precision-criterion evidence).
    print("\n[5. 梯度量级范围 (供下游精度判据)]")
    for name, B, T, H, K, V in _P0:
        args, _ = _build_op5_inputs(B, T, H, K, V, seed=42)
        outs = gdr_bwd_dqkw_golden_cpu(*args)
        mags = {nm: t.abs().max().item() for nm, t in zip(_OUT_NAMES, outs)}
        print(f"  {name}: " + ", ".join(f"|{nm}|max={mags[nm]:.3e}" for nm in _OUT_NAMES))

    print("\n" + "=" * 68)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 68)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
