#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_dqkw (OP5) NPU golden reference (torch / torch_npu).

OP5 of gated_delta_rule_bwd = the chunk-parallel dqkwg backward (parent module M8,
golden_step b3, BHN-parallel). This is a MECHANICAL PROJECTION (a faithful M8 slice),
NOT fresh math: it computes exactly the M8 arithmetic of the READ-ONLY delivered
cumulative staging golden

    gated_delta_rule_bwd_golden_stage12345678.py  (_m8)

but consumes OP5's frozen GM contract directly — 9 GM inputs already in internal
per-chunk fp32 layout (plus the public bf16 grad_o) — instead of the primary
(q,k,v,g,beta) the parent recomputes internally.

──────────────────────────────────────────────────────────────────────────────
PROJECTION MAP  (M8 `_m8` in the parent staging golden  ->  OP5 here)
  do        = grad_o chunked [B,H,N,C,V]   (transpose(1,2), pad T->N*C, reshape, ->fp32)
  h  = s_hist [B,H,N,K,V] ; dh [B,H,N,K,V] ; inter terms contract V -> h^T/dh^T = [V,K]
  decay     = cumulative gate GM input (parent M2 cumsum); decay_exp = exp(decay) RECOMPUTED
  g_last    = decay[..., -1]                (chunk-total decay; NOT read from raw g_f32)
  scale     = 1/sqrt(K)
  ds        = tril_incl_diag( (do @ v_new^T) * exp(decay_i - decay_j) * scale )
  dq_p      = (do @ h) * decay_exp * scale  +  ds @ k_f32
  dk_inter  = (v_new @ dh) * exp(g_last - decay)
  dk_p      = dk_inter  +  ds^T @ q_raw
  dw        = -(dv2 @ h)
  dg_last   = exp(g_last)*sum_KV(h*dh) + sum_KV(dk_inter*k_f32)   (scalar per chunk)
  dg_p[c]   = sum_K(dq_p*q_raw) - sum_K(dk_p*k_f32) ; dg_last folded onto LAST row (c=C-1)

──────────────────────────────────────────────────────────────────────────────
WHERE EACH FACTOR IS APPLIED
  Matched against the AUTHORITATIVE `_m8`, not the one-line summary:
  * scale  : folded into ds (`* scale`) AND into dq_inter (`* scale`); NOT applied
             again to dq_intra (already carried by ds), to any dk term, to dw, or to
             the dg reduce.
  * decay_exp = exp(decay) : RECOMPUTED internally from the `decay` GM input (a cheap
             vec exp); NOT a GM input. Used only in dq_inter.
  * exp(g_last - decay) gate on dk_inter : g_last = decay[..., -1]; gate = exp(g_last -
             decay) broadcast over K; multiplies the (v_new @ dh) inter term.
  * dg_last last-row (token C-1) correction : exp(g_last)*sum_KV(h*dh) + sum_KV(dk_inter
             *k_f32), where dk_inter is the ALREADY-GATED inter term. Added only to c=C-1.
  * raw g_f32 : NOT read by the core math. `_m8` derives g_last from decay[...,-1] and
             recomputes decay_exp from decay. g_f32 is carried in the GM input set for
             parent fidelity only (redundant here). CONFIRMED against `_m8`.

  * q operand (LOAD-BEARING GM-contract subtlety):
      `_m8` uses the RAW (unscaled) q in dk_intra (`ds^T @ q_c`) and in the dg reduce
      (`sum_K(dq_p * q_c)`). OP5's GM input is `q_s = q * scale` (scale folded in by the
      upstream gdr_fwd_train_wy M1). This golden therefore RECOVERS the raw q internally:
          q_raw = q_s / scale        (= q_s * sqrt(K))
      Feeding q_s directly (unrecovered) breaks dk/dg (relL2 ~0.87-0.91); recovering q_raw
      matches `_m8` to fp32 (K=64 exact; K=128 ~1e-8 round-trip). dq and dw do not use q.

──────────────────────────────────────────────────────────────────────────────
Signature:
    gdr_bwd_dqkw_golden(grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, g_f32, decay)
        -> (dq_p, dk_p, dw, dg_p)   all fp32
    grad_o : [B, T, H, V]     bf16 (public layout; sliced+cast to fp32 internally)
    v_new  : [B, H, N, C, V]  fp32   (delta-corrected value; from gdr_fwd_train_scan)
    k_f32  : [B, H, N, C, K]  fp32   (raw k; from gdr_fwd_train_wy)
    q_s    : [B, H, N, C, K]  fp32   (SCALED q = q*scale; from gdr_fwd_train_wy)
    s_hist : [B, H, N, K, V]  fp32   (per-chunk entering state; s_hist[:,:,0]=0)
    dh     : [B, H, N, K, V]  fp32   (reverse-scan entering dS; dh[:,:,N-1]=0)
    dv2    : [B, H, N, C, V]  fp32   (from gdr_bwd_reverse_scan)
    g_f32  : [B, H, N, C]     fp32   (raw pre-cumsum gate; carried for parent fidelity, UNUSED)
    decay  : [B, H, N, C]     fp32   (within-chunk cumulative gate = cumsum(g) over C)
    ->  dq_p, dk_p, dw : [B, H, N, C, K]  fp32   (NOT narrowed)
        dg_p           : [B, H, N, C]     fp32   (NOT narrowed)

All arithmetic fp32. BHN-parallel, stateless (no loop-carried tensor). Padding rows carry
g=0 (inert). No [:T] slice here (that is OP7's b7).
"""

import os

import torch

try:
    import torch_npu  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch_npu is required for gdr_bwd_dqkw_golden. Install torch_npu / configure CANN."
    ) from e

import torch.nn.functional as F

_C = 64  # fixed internal chunk size


# ─────────────────────────────────────────────
# Device selection (TILE_FWK_DEVICE_ID; fall back to CPU when NPU unavailable).
# ─────────────────────────────────────────────

def _select_device():
    if hasattr(torch, "npu") and torch.npu.is_available() and torch.npu.device_count() > 0:
        dev_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
        return torch.device(f"npu:{dev_id}")
    return torch.device("cpu")


# ─────────────────────────────────────────────
# NPU Golden 参考实现（torch / torch_npu, fp32 math）
# ─────────────────────────────────────────────

def gdr_bwd_dqkw_golden(
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
    """NPU golden for OP5 (M8 dqkwg, faithful slice of the parent b3 step).

    Returns (dq_p, dk_p, dw, dg_p), all fp32.  g_f32 is accepted per contract but UNUSED
    (g_last / decay_exp are derived from `decay`); see module docstring.
    """
    device = grad_o.device
    C = _C

    # ---- promote / move to device (grad_o bf16 -> fp32, the only cast boundary) ----
    do_pub = grad_o.to(device=device, dtype=torch.float32)             # [B, T, H, V]
    v_new = v_new.to(device=device, dtype=torch.float32)
    k_f32 = k_f32.to(device=device, dtype=torch.float32)
    q_s = q_s.to(device=device, dtype=torch.float32)
    s_hist = s_hist.to(device=device, dtype=torch.float32)
    dh = dh.to(device=device, dtype=torch.float32)
    dv2 = dv2.to(device=device, dtype=torch.float32)
    decay = decay.to(device=device, dtype=torch.float32)
    # g_f32 accepted for parent-fidelity contract; not used by the core math.
    _ = g_f32

    B, T, H, V = do_pub.shape
    N, K = v_new.shape[2], k_f32.shape[-1]
    scale = 1.0 / (K ** 0.5)

    # ---- grad_o public [B,T,H,V] -> chunked do [B,H,N,C,V] (pad T -> N*C) ----
    do = do_pub.transpose(1, 2).contiguous()                          # [B, H, T, V]
    pad = (C - (T % C)) % C
    if pad > 0:
        do = F.pad(do, (0, 0, 0, pad))
    do = do.reshape(B, H, N, C, V)                                    # [B, H, N, C, V]

    # ---- recover raw q (q_s = q*scale upstream; _m8 uses raw q) ----
    q_raw = q_s / scale                                               # [B, H, N, C, K]

    # ---- derived gate quantities (from `decay`; NOT from raw g_f32) ----
    decay_exp = decay.exp()                                          # [B, H, N, C]
    g_last = decay[..., -1]                                          # [B, H, N]

    # ---- ds = tril_incl_diag( (do @ v_new^T) * exp(decay_i - decay_j) * scale ) ----
    ds = torch.matmul(do.float(), v_new.transpose(-1, -2).float())   # [C,V]@[V,C] -> [C,C]
    gdiff = decay.unsqueeze(-1) - decay.unsqueeze(-2)                # [.,i,j] = decay_i - decay_j
    ds = ds * gdiff.exp() * scale
    up_strict = torch.triu(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=1)
    ds = ds.masked_fill(up_strict, 0.0)                              # keep lower incl diag (i>=j)

    # inter-chunk terms contract over V -> use h^T / dh^T ([V,K]).
    h_T = s_hist.transpose(-1, -2)                                   # [B,H,N,V,K]
    dh_T = dh.transpose(-1, -2)                                      # [B,H,N,V,K]

    # ---- dq_p = (do @ h) * decay_exp * scale + ds @ k ----
    dq_inter = torch.matmul(do.float(), h_T.float()) * decay_exp[..., None] * scale
    dq_intra = torch.matmul(ds.float(), k_f32.float())
    dq_p = dq_inter + dq_intra                                      # [B,H,N,C,K]

    # ---- dk_p = (v_new @ dh) * exp(g_last - decay) + ds^T @ q_raw ----
    dk_inter = torch.matmul(v_new.float(), dh_T.float())            # [C,V]@[V,K] -> [C,K]
    gate_k = (g_last[..., None] - decay).exp()                      # exp(g_last - decay_i) [B,H,N,C]
    dk_inter = dk_inter * gate_k[..., None]
    dk_intra = torch.matmul(ds.transpose(-1, -2).float(), q_raw.float())
    dk_p = dk_inter + dk_intra                                      # [B,H,N,C,K]

    # ---- dw = -(dv2 @ h) ----
    dw = -torch.matmul(dv2.float(), h_T.float())                   # [C,V]@[V,K] -> [C,K]

    # ---- dg_last scalar per chunk = exp(g_last)*sum_KV(h*dh) + sum_KV(dk_inter*k) ----
    dg_last = g_last.exp() * (s_hist * dh).sum(dim=(-2, -1)) \
        + (dk_inter * k_f32).sum(dim=(-2, -1))                     # [B,H,N]

    # ---- dg_p per token = sum_K(dq_p*q_raw) - sum_K(dk_p*k); dg_last folded onto row C-1 ----
    dg_p = (dq_p * q_raw).sum(-1) - (dk_p * k_f32).sum(-1)          # [B,H,N,C]
    dg_corr = F.pad(dg_last.unsqueeze(-1), (C - 1, 0))             # place scalar at token C-1
    dg_p = dg_p + dg_corr

    return (dq_p.to(torch.float32), dk_p.to(torch.float32),
            dw.to(torch.float32), dg_p.to(torch.float32))


# ─────────────────────────────────────────────
# Jointly-constructed inputs — built from a common primary through the upstream chain
# (gdr_fwd_train_wy -> gdr_fwd_train_scan -> gdr_bwd_dv_local -> gdr_bwd_reverse_scan),
# so the 9 OP5 inputs carry their real inter-tensor dependencies (v_new = u - w@S,
# s_hist entering state, dh reverse-scan snapshot, decay = cumsum(g), ...). Random
# per-tensor init would violate these couplings.
# ─────────────────────────────────────────────

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
    return wy, scan, dvl, rev


def _build_op5_inputs(B, T, H, K, V, seed, device, g_lo=-4.0, g_hi=0.0):
    """Build the 9 OP5 GM inputs jointly from a common primary, on `device`."""
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

    def _d(t):
        return t.to(device)

    inputs = [_d(grad_o), _d(v_new), _d(k_f32), _d(q_s), _d(s_hist),
              _d(dh), _d(dv2), _d(g_f32), _d(decay)]
    return inputs, {}, dict(q=q, k=k, v=v, g=g, beta=beta, grad_o=grad_o)


def _make_inputs(device):
    """Contract P0 inputs (jointly constructed). Multi-case format (one entry per P0)."""
    cases = []
    for name, (B, T, H, K, V) in _P0_CONFIGS:
        args, kwargs, _ = _build_op5_inputs(B, T, H, K, V, seed=42, device=device)
        cases.append((name, args, kwargs))
    return cases


# ==========================================
# 验证
# ==========================================

_P0_CONFIGS = [
    ("aligned_b1_t128_h4_k64_v64", (1, 128, 4, 64, 64)),
    ("ttail_b1_t100_h4_k64_v64", (1, 100, 4, 64, 64)),
    ("t3c_b1_t192_h2_k128_v128", (1, 192, 2, 128, 128)),
    ("tail_vlt_b1_t200_h3_k128_v96", (1, 200, 3, 128, 96)),
]
_SEEDS = [42, 123, 456]
_OUT_NAMES = ("dq_p", "dk_p", "dw", "dg_p")


def _expected_shapes(B, T, H, K, V):
    N = (T + 63) // 64
    C = 64
    return {"dq_p": (B, H, N, C, K), "dk_p": (B, H, N, C, K),
            "dw": (B, H, N, C, K), "dg_p": (B, H, N, C)}


def _rel_l2(a, b):
    return (torch.linalg.vector_norm((a - b).float())
            / torch.linalg.vector_norm(b.float()).clamp_min(1e-12)).item()


def _validate():
    print("=" * 68)
    print("gdr_bwd_dqkw_golden 验证报告 (OP5, M8/b3 mechanical projection)")
    print("=" * 68)

    device = _select_device()
    print(f"device = {device}")

    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    # 1. Contract P0: _make_inputs coverage, shape / fp32 / finite.
    print("\n[1. 合同 P0 用例: _make_inputs / shape / fp32 / finite]")
    cases = _make_inputs(device)
    seen = set()
    for (name, args, kwargs), (cfg_name, (B, T, H, K, V)) in zip(cases, _P0_CONFIGS):
        _check(f"{name}: matches P0 order '{cfg_name}'", name == cfg_name)
        seen.add(name)
        outs = gdr_bwd_dqkw_golden(*args, **kwargs)
        exp = _expected_shapes(B, T, H, K, V)
        _check(f"{name}: returns 4 outputs", len(outs) == 4)
        for nm, t in zip(_OUT_NAMES, outs):
            _check(f"{name}.{nm}: shape {tuple(t.shape)} == {exp[nm]}", tuple(t.shape) == exp[nm])
            _check(f"{name}.{nm}: dtype fp32", t.dtype == torch.float32)
            _check(f"{name}.{nm}: finite", torch.isfinite(t).all().item())
    _check("all 4 P0 cases present, no dup/missing",
           seen == {c[0] for c in _P0_CONFIGS} and len(cases) == 4)

    # 2. MANDATORY five-operator cross-check (a mismatch is a GM-CONTRACT DEFECT):
    #    real forward gdr_fwd_train_wy + gdr_fwd_train_scan (v_new,s_hist) + gdr_bwd_dv_local
    #    + gdr_bwd_reverse_scan (dh,dv2) -> OP5 projection, vs the parent M8 staging golden
    #    gated_delta_rule_bwd_golden_stage12345678 from the SAME primary. All four outputs
    #    must agree to fp32 tolerance across 4 shapes x seeds 42/123/456.
    print("\n[2. 五算子交叉验证 (OP5 投影 vs 父 M8 staging golden stage12345678)]")
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    parent_mod = os.path.join(os.path.dirname(here), "gated_delta_rule_bwd", "modules",
                              "gated_delta_rule_bwd_golden_stage12345678.py")
    spec = importlib.util.spec_from_file_location("gated_delta_rule_bwd_golden_stage12345678", parent_mod)
    m8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m8)

    CROSS_TOL = 2e-2  # rel-L2 acceptance criterion for the fp32 gradient outputs
    worst = {nm: 0.0 for nm in _OUT_NAMES}
    for name, (B, T, H, K, V) in _P0_CONFIGS:
        for seed in _SEEDS:
            args, kwargs, prim = _build_op5_inputs(B, T, H, K, V, seed, device)
            outs = gdr_bwd_dqkw_golden(*args, **kwargs)
            ref = m8.gated_delta_rule_bwd_golden_stage12345678(
                prim["grad_o"], prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
            errs = {}
            ok = True
            for nm, o, r in zip(_OUT_NAMES, outs, ref):
                e = _rel_l2(o.cpu(), r)
                errs[nm] = e
                worst[nm] = max(worst[nm], e)
                ok = ok and (e <= CROSS_TOL) and torch.isfinite(o).all().item()
            _check(f"{name} s{seed}: " + " ".join(f"{nm}={errs[nm]:.2e}" for nm in _OUT_NAMES)
                   + f" (<= {CROSS_TOL:.0e})", ok)
    print("    worst rel-L2 across all shapes/seeds: "
          + " ".join(f"{nm}={worst[nm]:.3e}" for nm in _OUT_NAMES))

    # 3. Numerical edges (structural properties of the b3 slice).
    print("\n[3. 数值边界]")
    # (a) grad_o == 0 end-to-end -> all four outputs == 0. Build the whole chain with a
    #     zeroed grad_o so downstream dv2/dh also reflect it.
    B, T, H, K, V = 1, 128, 2, 64, 64
    wy, scan, dvl, rev = _upstream()
    torch.manual_seed(7)
    grad_o = torch.zeros((B, T, H, V), dtype=torch.bfloat16)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(-4.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
     dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t) = wy.gdr_fwd_train_wy_golden_cpu(q, k, v, g, beta)
    o, s_hist, v_new = scan.gdr_fwd_train_scan_golden_cpu(u, w, q_de, a_mat, k_f32, dfac, decay_exp)
    dv_local = dvl.gdr_bwd_dv_local_golden_cpu(grad_o, k_f32, q_t, g_f32)
    dh, dv2 = rev.gdr_bwd_reverse_scan_golden_cpu(q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local)
    outs_z = gdr_bwd_dqkw_golden(grad_o.to(device), v_new.to(device), k_f32.to(device),
                                 q_s.to(device), s_hist.to(device), dh.to(device),
                                 dv2.to(device), g_f32.to(device), decay.to(device))
    _check("grad_o == 0 (end-to-end) -> all four outputs == 0",
           all(t.abs().max().item() == 0.0 for t in outs_z))

    # (b) last reverse chunk dh[:,:,N-1]==0 -> dk_inter and dg_last's dh terms vanish there.
    args, kwargs, prim = _build_op5_inputs(1, 200, 3, 128, 96, seed=42, device=device)
    dh_in = args[5]
    N_ = dh_in.shape[2]
    _check("dh[:,:,N-1] == 0 (reverse-scan first)", dh_in[:, :, N_ - 1].abs().max().item() == 0.0)

    # (c) very-negative g: exp bounded -> finite outputs.
    args, kwargs, prim = _build_op5_inputs(1, 128, 2, 64, 64, seed=3, device=device, g_lo=-16.0, g_hi=-8.0)
    outs_n = gdr_bwd_dqkw_golden(*args, **kwargs)
    _check("very-negative g (-16,-8): all outputs finite",
           all(torch.isfinite(t).all().item() for t in outs_n))

    # 4. Magnitude ranges (downstream precision-criterion evidence).
    print("\n[4. 梯度量级范围 (供下游精度判据)]")
    for name, (B, T, H, K, V) in _P0_CONFIGS:
        args, kwargs, _ = _build_op5_inputs(B, T, H, K, V, seed=42, device=device)
        outs = gdr_bwd_dqkw_golden(*args, **kwargs)
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
