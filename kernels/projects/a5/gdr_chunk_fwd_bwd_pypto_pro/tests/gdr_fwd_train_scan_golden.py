#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_fwd_train_scan NPU golden reference (torch + torch_npu).

OP-B of the gated_delta_rule TRAINING-FORWARD pair (OP-A = gdr_fwd_train_wy → OP-B).
The M5 inter-chunk scan (golden steps s12–s16) of the delivered READ-ONLY lean
forward `gated_delta_rule_fwd`, NARROWED to expose BOTH outputs:

  - `o`      : real public forward output, bf16, padded layout [B, 64*N, H, V].
               `o[:, :T]` reproduces the lean forward `o` EXACTLY (math unchanged).
  - `s_hist` : per-chunk ENTERING state S_i, fp32 [B, H, N, K, V]. `s_hist[:,:,i]`
               is S snapshotted BEFORE chunk-i's update (s15); `s_hist[:,:,0]=0`.
               Convention matches the delivered backward reverse scan
               (NOTES_module12345.md: `S_hist[i]=S`).
  - `v_new`  : per-chunk delta-corrected value, fp32 [B, H, N, C, V].
               `v_new[:,:,i] = u_i - w_i @ S`, S the ENTERING state s_hist[:,:,i] (s12).
               Already computed in the M5 scan; PROMOTED to a public output (same
               store-side promotion pattern as s_hist, zero arithmetic change). Exact
               quantity backward b3/M8 consumes (
               gated_delta_rule_bwd_golden_stage12345678.py::_m5/_m8); OP5 (gdr_bwd_dqkw)
               receives it instead of reconstructing it.

All compute is fp32; the only narrow boundary is the s16 store of `o` (fp32->bf16).
`s_hist` and `v_new` carry no cast. Math source (READ-ONLY, copied not modified):
gated_delta_rule_fwd_golden_cpu.py (its _chunk M5 loop)
and modules/gated_delta_rule_fwd_golden_stage12345.py (_state_through_4 → scan).

Public signature:
    gdr_fwd_train_scan_golden(u, w, q_de, a_mat, k_f32, dfac, decay_exp)
        -> (o (bf16), s_hist (fp32), v_new (fp32))
"""

import os
import torch

try:
    import torch_npu  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch_npu is required for the NPU golden. Install torch_npu, or run the "
        "CPU golden gdr_fwd_train_scan_golden_cpu.py instead.") from e

_CHUNK_SIZE = 64


def _resolve_device():
    """Select NPU by TILE_FWK_DEVICE_ID (default 0); fall back to CPU if unavailable."""
    if torch.npu.is_available() and torch.npu.device_count() > 0:
        dev_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
        return torch.device(f"npu:{dev_id}")
    return torch.device("cpu")


# ─────────────────────────────────────────────
# OP-B M5 inter-chunk scan (s12–s16 + s_hist store-side promotion).
# ─────────────────────────────────────────────

def gdr_fwd_train_scan_golden(
    u: torch.Tensor,
    w: torch.Tensor,
    q_de: torch.Tensor,
    a_mat: torch.Tensor,
    k_f32: torch.Tensor,
    dfac: torch.Tensor,
    decay_exp: torch.Tensor,
):
    """NPU golden for the gated_delta_rule training-forward inter-chunk scan.

    Args (OP-A M1–M4 handoffs, fp32, chunk-padded to Tp = 64*N):
        u         : [B, H, N, 64, V]
        w         : [B, H, N, 64, K]
        q_de      : [B, H, N, 64, K]
        a_mat     : [B, H, N, 64, 64]
        k_f32     : [B, H, N, 64, K]
        dfac      : [B, H, N, 64, K]
        decay_exp : [B, H, N, 64]

    Returns:
        o      : [B, 64*N, H, V]  bf16  (padded public layout; public output is o[:, :T]).
        s_hist : [B, H, N, K, V]  fp32  (per-chunk entering state; s_hist[:,:,0]=0).
        v_new  : [B, H, N, C, V]  fp32  (per-chunk delta-corrected value u_i - w_i@S,
                 S = entering state; the exact quantity backward b3/M8 consumes).
    """
    device = _resolve_device()
    u = u.to(device=device, dtype=torch.float32)
    w = w.to(device=device, dtype=torch.float32)
    q_de = q_de.to(device=device, dtype=torch.float32)
    a_mat = a_mat.to(device=device, dtype=torch.float32)
    k_f32 = k_f32.to(device=device, dtype=torch.float32)
    dfac = dfac.to(device=device, dtype=torch.float32)
    decay_exp = decay_exp.to(device=device, dtype=torch.float32)

    C = _CHUNK_SIZE
    b, h, n, _, dv = u.shape
    dk = w.shape[-1]

    S = u.new_zeros(b, h, dk, dv)
    o = torch.zeros_like(u)                    # [b, h, n, C, dv] fp32
    s_hist = u.new_zeros(b, h, n, dk, dv)      # [b, h, n, K, V] fp32
    v_new_hist = u.new_zeros(b, h, n, C, dv)   # [b, h, n, C, V] fp32

    for i in range(0, n):
        # s_hist snapshot: ENTERING state, BEFORE chunk-i update (s15). i=0 -> S=0.
        s_hist[:, :, i] = S

        A_i = a_mat[:, :, i]
        u_i = u[:, :, i]
        w_i = w[:, :, i]
        # matmul operands promoted to fp32 (they already are).
        v_prime = torch.matmul(w_i.float(), S.float())              # s12
        v_new = u_i - v_prime                                        # s12
        v_new_hist[:, :, i] = v_new                                  # promote per-chunk v_new
        o_inter = torch.matmul(q_de[:, :, i].float(), S.float())    # s13
        o[:, :, i] = o_inter + torch.matmul(A_i.float(), v_new.float())  # s14
        delta = torch.matmul(
            (k_f32[:, :, i] * dfac[:, :, i]).transpose(-1, -2).float(),
            v_new.float())                                          # s15 (contract C)
        S = S * decay_exp[:, :, i, -1, None, None] + delta          # s15

    # s16: [b,h,n,C,V] -> [b,h,64N,V] -> public [b,64N,h,V], cast fp32->bf16.
    o = o.reshape(b, h, n * C, dv)
    o = o.transpose(1, 2).contiguous()
    o = o.to(torch.bfloat16)
    return o, s_hist, v_new_hist


# ─────────────────────────────────────────────
# M1–M4 projection (READ-ONLY math copied from the fwd golden's
# modules/gated_delta_rule_fwd_golden_stage12345.py::_state_through_4).
# Builds the 7 jointly-consistent handoffs from PRIMARY inputs so the golden can be
# fed valid inputs and cross-checked against the read-only lean forward. OP-A golden
# (gdr_fwd_train_wy) is not present; M1–M4 are projected here from the read-only fwd.
# ─────────────────────────────────────────────

def _project_m1_m4(q, k, v, g, beta):
    import torch.nn.functional as F
    C = _CHUNK_SIZE
    dev = q.device
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
    q_s = (q * scale).reshape(b, h, n, C, dk)
    k_f32 = k.reshape(b, h, n, C, dk)
    v_b = (v * beta[..., None]).reshape(b, h, n, C, dv)
    k_beta = (k * beta[..., None]).reshape(b, h, n, C, dk)
    g_f32 = g.reshape(b, h, n, C)
    decay = g_f32.cumsum(-1)
    P1 = torch.matmul(k_beta, k_f32.transpose(-1, -2))
    P2 = torch.matmul(q_s, k_f32.transpose(-1, -2))
    decay_exp = decay.exp()
    q_de = q_s * decay_exp[..., None]
    kbeta_de = k_beta * decay_exp[..., None]
    dfac = (decay[..., -1:] - decay).exp()[..., None].expand(b, h, n, C, dk).contiguous()
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp()).tril()
    up_incl_diag = torch.triu(torch.ones(C, C, dtype=torch.bool, device=dev), diagonal=0)
    attn = -((P1) * L_mask).masked_fill(up_incl_diag, 0)
    for i in range(1, C):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(C, dtype=torch.float32, device=dev)
    up_strict = torch.triu(torch.ones(C, C, dtype=torch.bool, device=dev), diagonal=1)
    A = ((P2) * L_mask).masked_fill(up_strict, 0)
    u = torch.matmul(attn, v_b)
    w = torch.matmul(attn, kbeta_de)
    return u, w, q_de, A, k_f32, dfac, decay_exp


# ─────────────────────────────────────────────
# _make_inputs(device): the exact machine-contract P0 set, jointly constructed by
# projecting M1–M4 from primary inputs (T chosen so ceil(T/64) = N). Reused by
# _validate(). Multi-case format.
# ─────────────────────────────────────────────

def _make_inputs(device):
    def _one(B, T, H, K, V, seed):
        torch.manual_seed(seed)
        q = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
        k = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
        v = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)
        g = torch.empty((B, T, H), dtype=torch.float32, device=device).uniform_(-4.0, 0.0)
        beta = torch.randn((B, T, H), dtype=torch.bfloat16, device=device)
        u, w, q_de, a_mat, k_f32, dfac, decay_exp = _project_m1_m4(q, k, v, g, beta)
        return [u, w, q_de, a_mat, k_f32, dfac, decay_exp], {}

    # P0 machine-contract cases:
    #   smoke: B1 H4 N2 K64 V64  (T=128 -> N=2)
    #   mid  : B1 H2 N3 K128 V128 (T=192 -> N=3)
    return [
        ("smoke_b1_h4_n2_k64_v64", *_one(1, 128, 4, 64, 64, seed=0)),
        ("mid_b1_h2_n3_k128_v128", *_one(1, 192, 2, 128, 128, seed=0)),
    ]


# ==========================================
# 验证
# ==========================================

def _load_fwd_golden_cpu():
    """Load the READ-ONLY lean forward CPU golden by file path (no edit, no copy)."""
    import importlib.util
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "gated_delta_rule_fwd", "gated_delta_rule_fwd_golden_cpu.py")
    spec = importlib.util.spec_from_file_location("_gdr_fwd_golden_cpu", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _validate():
    print("=" * 66)
    print("gdr_fwd_train_scan_golden 验证报告 (OP-B, NPU torch/torch_npu)")
    print("=" * 66)
    device = _resolve_device()
    print(f"device = {device}")

    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        status = "PASS" if cond else "FAIL"
        if not cond:
            all_pass = False
        print(f"  [{status}] {name}")
        return cond

    # ---- 1. Machine-contract P0 exact set: shape / dtype / finite / s_hist[0]=0 ----
    print("\n[1. 机器合同 P0 (exact set via _make_inputs): shape / dtype / finite]")
    cases = _make_inputs(device)
    seen = set()
    expected = {
        "smoke_b1_h4_n2_k64_v64": ((1, 128, 4, 64), (1, 4, 2, 64, 64), (1, 4, 2, 64, 64)),
        "mid_b1_h2_n3_k128_v128": ((1, 192, 2, 128), (1, 2, 3, 128, 128), (1, 2, 3, 64, 128)),
    }
    _check("P0 count == 2", len(cases) == 2)
    for name, args, kwargs in cases:
        _check(f"{name}: not duplicate", name not in seen)
        seen.add(name)
        o, s_hist, v_new = gdr_fwd_train_scan_golden(*args, **kwargs)
        exp_o, exp_s, exp_vn = expected[name]
        _check(f"{name}: o shape {exp_o}", tuple(o.shape) == exp_o)
        _check(f"{name}: s_hist shape {exp_s}", tuple(s_hist.shape) == exp_s)
        _check(f"{name}: v_new shape {exp_vn}", tuple(v_new.shape) == exp_vn)
        _check(f"{name}: o dtype bf16", o.dtype == torch.bfloat16)
        _check(f"{name}: s_hist dtype fp32", s_hist.dtype == torch.float32)
        _check(f"{name}: v_new dtype fp32", v_new.dtype == torch.float32)
        _check(f"{name}: o finite", torch.isfinite(o.float()).all().item())
        _check(f"{name}: s_hist finite", torch.isfinite(s_hist).all().item())
        _check(f"{name}: v_new finite", torch.isfinite(v_new).all().item())
        _check(f"{name}: s_hist[:,:,0]==0", s_hist[:, :, 0].abs().max().item() == 0.0)
    _check("P0 exact set (no missing/extra names)", seen == set(expected.keys()))

    # ---- 2. HARD acceptance: o[:, :T] reproduces the READ-ONLY lean forward o ----
    #     4 shapes × seeds 42/123/456; mixed peak-scaled allclose OR rel-L2 criterion.
    #     bf16 o cast tolerated by atol = 1e-2 * peak. Reference computed on CPU.
    print("\n[2. HARD 验收: o[:, :T] 复现只读 lean forward o (mixed §8/§9 criterion)]")
    fwd = _load_fwd_golden_cpu()
    shapes = [(1, 128, 4, 64, 64), (1, 100, 4, 64, 64),
              (1, 192, 2, 128, 128), (1, 200, 3, 128, 96)]
    seeds = [42, 123, 456]
    worst_scaled = 0.0
    for (B, T, H, K, V) in shapes:
        for seed in seeds:
            torch.manual_seed(seed)
            q = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
            k = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
            v = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)
            g = torch.empty((B, T, H), dtype=torch.float32, device=device).uniform_(-4.0, 0.0)
            beta = torch.randn((B, T, H), dtype=torch.bfloat16, device=device)
            inputs = _project_m1_m4(q, k, v, g, beta)
            o_pad, s_hist, v_new = gdr_fwd_train_scan_golden(*inputs)
            o_pub = o_pad[:, :T].float().cpu()                     # [B,T,H,V]
            o_ref = fwd.gated_delta_rule_fwd_golden_cpu(
                q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu())     # fp32 CPU reference
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

    # ---- 3. s_hist ENTERING-state convention (before chunk-i update) ----
    print("\n[3. s_hist 进入态语义 (snapshot before chunk-i update)]")
    torch.manual_seed(42)
    B, T, H, K, V = 1, 200, 3, 128, 96
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)
    g = torch.empty((B, T, H), dtype=torch.float32, device=device).uniform_(-4.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device=device)
    u, w, q_de, a_mat, k_f32, dfac, decay_exp = _project_m1_m4(q, k, v, g, beta)
    o_pad, s_hist, v_new = gdr_fwd_train_scan_golden(u, w, q_de, a_mat, k_f32, dfac, decay_exp)
    C = _CHUNK_SIZE
    b, h, n, _, dv = u.shape
    dk = w.shape[-1]
    S_ref = u.new_zeros(b, h, dk, dv)
    entering = []
    v_new_ref = []
    for i in range(n):
        entering.append(S_ref.clone())
        v_prime = torch.matmul(w[:, :, i], S_ref)
        v_new_i = u[:, :, i] - v_prime
        v_new_ref.append(v_new_i.clone())
        S_ref = S_ref * decay_exp[:, :, i, -1, None, None] + torch.matmul(
            (k_f32[:, :, i] * dfac[:, :, i]).transpose(-1, -2), v_new_i)
    enter_match = all(torch.equal(s_hist[:, :, i], entering[i]) for i in range(n))
    _check("s_hist[:,:,i] == entering state (before chunk-i update), all i", enter_match)
    _check("s_hist[:,:,0] == 0", s_hist[:, :, 0].abs().max().item() == 0.0)
    if n > 1:
        _check("s_hist[:,:,1] != 0", s_hist[:, :, 1].abs().max().item() > 0.0)

    # ---- 4. v_new correctness: v_new[:,:,i] == u_i - w_i @ s_hist[:,:,i] (Path-A) ----
    print("\n[4. v_new 正确性 (== u_i - w_i @ s_hist[:,:,i], Path-A 重构)]")
    vnew_scan_match = all(torch.equal(v_new[:, :, i], v_new_ref[i]) for i in range(n))
    _check("v_new[:,:,i] == independent scan recompute (u_i - w_i@S), all i",
           vnew_scan_match)
    v_new_pathA = u - torch.matmul(w, s_hist)          # [b,h,n,C,K]@[b,h,n,K,V]
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
