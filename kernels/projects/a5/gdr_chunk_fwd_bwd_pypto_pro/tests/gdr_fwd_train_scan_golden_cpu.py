#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_fwd_train_scan CPU golden reference (higher precision, FP32).

OP-B of the gated_delta_rule TRAINING-FORWARD pair (OP-A = gdr_fwd_train_wy → OP-B).
This is the M5 inter-chunk scan (golden steps s12–s16) of the delivered, READ-ONLY
lean forward `gated_delta_rule_fwd`, NARROWED to expose BOTH:

  - `o`  : the real public forward output (bf16 in the NPU contract; FP32 here, per the
           CPU-golden convention of not downcasting), padded layout [B, 64*N, H, V].
           `o[:, :T]` reproduces the lean forward `o` EXACTLY (identical math).
  - `s_hist` : the per-chunk ENTERING state S_i, fp32 [B, H, N, K, V]. `s_hist[:,:,i]`
           is the snapshot of S taken BEFORE chunk-i's update (s15). `s_hist[:,:,0]=0`.
           This store-side promotion is a store-side addition over the lean-forward M5
           scan (zero arithmetic change). Convention matches the delivered backward's
           reverse scan (NOTES_module12345.md: `S_hist[i]=S`).
  - `v_new`  : the per-chunk delta-corrected chunk value, fp32 [B, H, N, C, V].
           `v_new[:,:,i] = u_i - w_i @ S` with S the ENTERING state s_hist[:,:,i] (s12).
           Already computed internally in the M5 scan; here PROMOTED to a public output
           (same store-side promotion pattern as s_hist, zero arithmetic change). This is
           the exact quantity the delivered backward b3/M8 consumes (its `_m5` recomputes
           v_new = u - w@S identically): 
           gated_delta_rule_bwd_golden_stage12345678.py::_m5/_m8. OP5 (gdr_bwd_dqkw)
           receives this instead of reconstructing it.

Math source (READ-ONLY, copied not modified): 
gated_delta_rule_fwd_golden_cpu.py (its _chunk M5 loop, s12–s16) and the per-Module
golden modules/gated_delta_rule_fwd_golden_stage12345.py (_state_through_4 → scan).

Public signature:
    gdr_fwd_train_scan_golden_cpu(u, w, q_de, a_mat, k_f32, dfac, decay_exp)
        -> (o, s_hist, v_new)   all fp32
"""

import torch
import torch.nn.functional as F

_CHUNK_SIZE = 64


# ─────────────────────────────────────────────
# OP-B M5 inter-chunk scan (s12–s16 + s_hist store-side promotion).
# Verbatim transcription of the read-only fwd golden's _chunk M5 loop, plus the
# entering-state snapshot. NO arithmetic change to the o branch.
# ─────────────────────────────────────────────

def gdr_fwd_train_scan_golden_cpu(
    u: torch.Tensor,
    w: torch.Tensor,
    q_de: torch.Tensor,
    a_mat: torch.Tensor,
    k_f32: torch.Tensor,
    dfac: torch.Tensor,
    decay_exp: torch.Tensor,
):
    """CPU FP32 golden for the gated_delta_rule training-forward inter-chunk scan.

    Args (all OP-A M1–M4 handoffs, fp32, chunk-padded to Tp = 64*N):
        u         : [B, H, N, 64, V]  (M4 k_cumsum, the new v)
        w         : [B, H, N, 64, K]  (M4 k_cumdecay)
        q_de      : [B, H, N, 64, K]  (M3 q_s * decay_exp)
        a_mat     : [B, H, N, 64, 64] (M3 intra-chunk strictly-lower A)
        k_f32     : [B, H, N, 64, K]  (M1)
        dfac      : [B, H, N, 64, K]  (M3 exp(decay_last - decay), broadcast over K)
        decay_exp : [B, H, N, 64]     (M3 exp(cumsum(g)))

    Returns:
        o      : [B, 64*N, H, V]  fp32 (padded public layout, NOT downcast to bf16;
                 the public forward output is the external view o[:, :T]).
        s_hist : [B, H, N, K, V]  fp32 (per-chunk entering state; s_hist[:,:,0]=0).
        v_new  : [B, H, N, C, V]  fp32 (per-chunk delta-corrected value u_i - w_i@S,
                 S = entering state; the exact quantity backward b3/M8 consumes).
    """
    u = u.to(torch.float32)
    w = w.to(torch.float32)
    q_de = q_de.to(torch.float32)
    a_mat = a_mat.to(torch.float32)
    k_f32 = k_f32.to(torch.float32)
    dfac = dfac.to(torch.float32)
    decay_exp = decay_exp.to(torch.float32)

    C = _CHUNK_SIZE
    b, h, n, _, dv = u.shape
    dk = w.shape[-1]

    S = u.new_zeros(b, h, dk, dv)
    o = torch.zeros_like(u)                    # [b, h, n, C, dv]
    s_hist = u.new_zeros(b, h, n, dk, dv)      # [b, h, n, K, V]
    v_new_hist = u.new_zeros(b, h, n, C, dv)   # [b, h, n, C, V]

    for i in range(0, n):
        # s_hist snapshot: ENTERING state, taken BEFORE the chunk-i update (s15).
        # For i=0 this is the initial S = 0.
        s_hist[:, :, i] = S

        A_i = a_mat[:, :, i]
        u_i = u[:, :, i]
        w_i = w[:, :, i]
        v_prime = w_i @ S                       # s12  [C,V]
        v_new = u_i - v_prime                    # s12  [C,V]
        v_new_hist[:, :, i] = v_new              # promote per-chunk v_new (store-side)
        o_inter = q_de[:, :, i] @ S              # s13  [C,V]
        o[:, :, i] = o_inter + A_i @ v_new       # s14  [C,V]
        S = S * decay_exp[:, :, i, -1, None, None] + (
            k_f32[:, :, i] * dfac[:, :, i]).transpose(-1, -2) @ v_new   # s15  [K,V]

    # s16: reshape [b,h,n,C,V] -> [b,h,64N,V], transpose to public [b,64N,h,V].
    # NO [:T] slice here: OP-B emits the padded layout; the public output is the
    # external view o[:, :T] applied at composition.
    o = o.reshape(b, h, n * C, dv)
    o = o.transpose(1, 2).contiguous()          # [b, 64N, h, V]
    return o, s_hist, v_new_hist


# ─────────────────────────────────────────────
# M1–M4 projection (READ-ONLY math copied from the fwd golden's
# modules/gated_delta_rule_fwd_golden_stage12345.py::_state_through_4).
# Produces the 7 handoffs from PRIMARY inputs (q,k,v,g,beta) so the golden can be
# fed consistent, jointly-constructed inputs and cross-checked end-to-end against
# the read-only lean forward. OP-A (gdr_fwd_train_wy) golden is not present, so we
# project M1–M4 here directly from the read-only fwd golden math.
# ─────────────────────────────────────────────

def _project_m1_m4(q, k, v, g, beta):
    C = _CHUNK_SIZE
    q = q.to(torch.float32); k = k.to(torch.float32); v = v.to(torch.float32)
    g = g.to(torch.float32); beta = torch.sigmoid(beta.to(torch.float32))
    scale = 1.0 / (q.shape[-1] ** 0.5)
    q = q.transpose(1, 2).contiguous(); k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous(); g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    T = q.shape[-2]
    pad = (C - (T % C)) % C
    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad)); g = F.pad(g, (0, pad)); beta = F.pad(beta, (0, pad))
    b, h, Tp, dk = q.shape
    dv = v.shape[-1]
    n = Tp // C
    # Module 1
    q_s = (q * scale).reshape(b, h, n, C, dk)
    k_f32 = k.reshape(b, h, n, C, dk)
    v_b = (v * beta[..., None]).reshape(b, h, n, C, dv)
    k_beta = (k * beta[..., None]).reshape(b, h, n, C, dk)
    g_f32 = g.reshape(b, h, n, C)
    # Module 2
    decay = g_f32.cumsum(-1)
    P1 = k_beta @ k_f32.transpose(-1, -2)
    P2 = q_s @ k_f32.transpose(-1, -2)
    # Module 3
    decay_exp = decay.exp()
    q_de = q_s * decay_exp[..., None]
    kbeta_de = k_beta * decay_exp[..., None]
    dfac = (decay[..., -1:] - decay).exp()[..., None].expand(b, h, n, C, dk).contiguous()
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp()).tril()
    up_incl_diag = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
    attn = -((P1) * L_mask).masked_fill(up_incl_diag, 0)
    for i in range(1, C):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(C, dtype=torch.float32)
    up_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    A = ((P2) * L_mask).masked_fill(up_strict, 0)
    # Module 4
    u = attn @ v_b
    w = attn @ kbeta_de
    # OP-B signature order: (u, w, q_de, a_mat, k_f32, dfac, decay_exp)
    return u, w, q_de, A, k_f32, dfac, decay_exp


# ==========================================
# 验证
# ==========================================

def _mk_primary(B, T, H, K, V, seed):
    torch.manual_seed(seed)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(-4.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return q, k, v, g, beta


def _load_fwd_golden_cpu():
    """Load the READ-ONLY lean forward CPU golden by file path (no edit, no copy)."""
    import importlib.util, os
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "gated_delta_rule_fwd", "gated_delta_rule_fwd_golden_cpu.py")
    spec = importlib.util.spec_from_file_location("_gdr_fwd_golden_cpu", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _validate():
    print("=" * 66)
    print("gdr_fwd_train_scan_golden_cpu 验证报告 (OP-B, CPU FP32)")
    print("=" * 66)

    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        status = "PASS" if cond else "FAIL"
        if not cond:
            all_pass = False
        print(f"  [{status}] {name}")
        return cond

    # ---- 1. Contract P0 cases: shape / dtype / finite ----
    print("\n[1. 机器合同 P0: shape / dtype / finite]")
    p0 = [
        # (name, B, T, H, K, V, N)  -- T chosen so ceil(T/64)=N
        ("smoke_b1_h4_n2_k64_v64", 1, 128, 4, 64, 64, 2),
        ("mid_b1_h2_n3_k128_v128", 1, 192, 2, 128, 128, 3),
    ]
    for name, B, T, H, K, V, N in p0:
        q, k, v, g, beta = _mk_primary(B, T, H, K, V, seed=0)
        inputs = _project_m1_m4(q, k, v, g, beta)
        o, s_hist, v_new = gdr_fwd_train_scan_golden_cpu(*inputs)
        _check(f"{name}: o shape [B,64N,H,V]={ (B, 64*N, H, V)}",
               tuple(o.shape) == (B, 64 * N, H, V))
        _check(f"{name}: s_hist shape [B,H,N,K,V]={(B, H, N, K, V)}",
               tuple(s_hist.shape) == (B, H, N, K, V))
        _check(f"{name}: v_new shape [B,H,N,C,V]={(B, H, N, 64, V)}",
               tuple(v_new.shape) == (B, H, N, 64, V))
        _check(f"{name}: o dtype fp32", o.dtype == torch.float32)
        _check(f"{name}: s_hist dtype fp32", s_hist.dtype == torch.float32)
        _check(f"{name}: v_new dtype fp32", v_new.dtype == torch.float32)
        _check(f"{name}: o finite", torch.isfinite(o).all().item())
        _check(f"{name}: s_hist finite", torch.isfinite(s_hist).all().item())
        _check(f"{name}: v_new finite", torch.isfinite(v_new).all().item())
        _check(f"{name}: s_hist[:,:,0]==0", s_hist[:, :, 0].abs().max().item() == 0.0)

    # ---- 2. HARD acceptance: o[:, :T] reproduces the READ-ONLY lean forward o ----
    #        4 shapes × seeds 42/123/456, mixed peak-scaled allclose OR rel-L2 criterion.
    print("\n[2. HARD 验收: o[:, :T] 复现只读 lean forward o]")
    fwd = _load_fwd_golden_cpu()
    shapes = [(1, 128, 4, 64, 64), (1, 100, 4, 64, 64),
              (1, 192, 2, 128, 128), (1, 200, 3, 128, 96)]
    seeds = [42, 123, 456]
    worst_scaled = 0.0
    for (B, T, H, K, V) in shapes:
        for seed in seeds:
            q, k, v, g, beta = _mk_primary(B, T, H, K, V, seed)
            inputs = _project_m1_m4(q, k, v, g, beta)
            o_pad, s_hist, v_new = gdr_fwd_train_scan_golden_cpu(*inputs)
            o_pub = o_pad[:, :T]                              # [B,T,H,V] fp32
            o_ref = fwd.gated_delta_rule_fwd_golden_cpu(q, k, v, g, beta)  # [B,T,H,V] fp32
            d = (o_pub - o_ref).abs()
            max_abs = d.max().item()
            peak = o_ref.abs().max().clamp_min(1e-6).item()
            scaled = max_abs / peak
            worst_scaled = max(worst_scaled, scaled)
            ok_allclose = torch.allclose(o_pub, o_ref, rtol=1e-2, atol=1e-2 * peak)
            rel_l2 = (torch.linalg.vector_norm(o_pub - o_ref) /
                      torch.linalg.vector_norm(o_ref).clamp_min(1e-12)).item()
            crit = ok_allclose or rel_l2 <= 1e-2
            _check(f"o-repro B{B}T{T}H{H}K{K}V{V} s{seed}: "
                   f"scaled_err={scaled:.3e} rel_l2={rel_l2:.3e} peak={peak:.2e}", crit)
    print(f"  -> worst peak-scaled o error across 12 cases = {worst_scaled:.3e}")

    # ---- 3. s_hist ENTERING-state convention ----
    #     s_hist[:,:,i] must equal the state accumulated over chunks 0..i-1 (before
    #     chunk-i update). Independently re-run the scan and snapshot at loop top.
    print("\n[3. s_hist 进入态语义 (snapshot before chunk-i update)]")
    q, k, v, g, beta = _mk_primary(1, 200, 3, 128, 96, seed=42)
    u, w, q_de, a_mat, k_f32, dfac, decay_exp = _project_m1_m4(q, k, v, g, beta)
    o_pad, s_hist, v_new = gdr_fwd_train_scan_golden_cpu(u, w, q_de, a_mat, k_f32, dfac, decay_exp)
    C = _CHUNK_SIZE
    b, h, n, _, dv = u.shape
    dk = w.shape[-1]
    S_ref = u.new_zeros(b, h, dk, dv)
    entering = []
    v_new_ref = []
    for i in range(n):
        entering.append(S_ref.clone())                 # entering state for chunk i
        v_prime = w[:, :, i] @ S_ref
        v_new_i = u[:, :, i] - v_prime
        v_new_ref.append(v_new_i.clone())
        S_ref = S_ref * decay_exp[:, :, i, -1, None, None] + (
            k_f32[:, :, i] * dfac[:, :, i]).transpose(-1, -2) @ v_new_i
    enter_match = all(
        torch.equal(s_hist[:, :, i], entering[i]) for i in range(n))
    _check("s_hist[:,:,i] == entering state (before chunk-i update), all i", enter_match)
    _check("s_hist[:,:,0] == 0 (initial state)",
           s_hist[:, :, 0].abs().max().item() == 0.0)
    # s_hist[:,:,i] for i>=1 is a nonzero accumulated state (sanity: not still 0)
    if n > 1:
        _check("s_hist[:,:,1] != 0 (accumulated after chunk 0)",
               s_hist[:, :, 1].abs().max().item() > 0.0)

    # ---- 4. v_new correctness: v_new[:,:,i] == u_i - w_i @ s_hist[:,:,i] ----
    #     Path-A reconstruction from the EXPOSED entering state s_hist.
    print("\n[4. v_new 正确性 (== u_i - w_i @ s_hist[:,:,i], Path-A 重构)]")
    vnew_scan_match = all(
        torch.equal(v_new[:, :, i], v_new_ref[i]) for i in range(n))
    _check("v_new[:,:,i] == independent scan recompute (u_i - w_i@S), all i",
           vnew_scan_match)
    # Path-A: reconstruct v_new purely from exposed (u, w, s_hist) — no internal state.
    v_new_pathA = u - torch.matmul(w, s_hist)          # [b,h,n,C,K]@[b,h,n,K,V]->[b,h,n,C,V]
    pathA_max = (v_new - v_new_pathA).abs().max().item()
    _check(f"v_new == u - w @ s_hist (Path-A, max_abs={pathA_max:.2e})",
           pathA_max == 0.0)
    _check("v_new[:,:,0] == u[:,:,0] (first chunk entering S=0 -> v_new=u)",
           torch.equal(v_new[:, :, 0], u[:, :, 0]))

    print("\n" + "=" * 66)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 66)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        exit(1)
