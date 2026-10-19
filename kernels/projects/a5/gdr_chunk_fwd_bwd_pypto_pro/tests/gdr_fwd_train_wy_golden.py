#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_fwd_train_wy golden reference (NPU, torch + torch_npu).

OP-A of the gated_delta_rule TRAINING-FORWARD pair = the WY-build front-end
(Modules M1-M4, BHN-parallel). MECHANICAL PROJECTION (a faithful slice), NOT fresh math:
it computes exactly the M1-M4 mathematics of the READ-ONLY delivered forward golden
gated_delta_rule_fwd_golden_cpu.py (function _chunk), up to
`k_cumsum`/`k_cumdecay` (BEFORE the inter-chunk s12-s16 scan), and exposes the 17 GM
boundary tensors the lean forward computes internally and discards.

Signature:  gdr_fwd_train_wy_golden(q, k, v, g, beta) -> (17 fp32 tensors)
Return order:
  q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp, dfac,
  q_de, attn, a_mat, kbeta_t, u, w, w_t   (all fp32, internal [B,H,N,...] layout, C=64).

Semantics (identical lineage to the forward golden):
  - beta is sigmoid LOGITS; sigmoid applied inside, THEN pad-with-0 (pad rows beta_act=0).
  - g is log-space decay (<=0), padded with 0. scale = 1/sqrt(K); no q/k l2norm; H==HV.
  - Every output in internal [B,H,N,...] layout (N=(T+63)//64); NO [:T] slice applied here.
  - All arithmetic fp32; 17 outputs all fp32 (no narrow).
"""

import os
import torch
import torch.nn.functional as F

_CHUNK_SIZE = 64

_DEVICE = None
_HAS_NPU = False

try:
    import torch_npu
    _HAS_NPU = torch.npu.is_available() and torch.npu.device_count() > 0
except ImportError:
    raise ImportError(
        "torch_npu is not installed. Please install it following the torch_npu official guide:\n"
        "  pip install torch_npu\n"
        "Refer to the CANN and PyPTO/PyPTO-Pro installation docs for the full NPU environment setup."
    )


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if not _HAS_NPU:
            _DEVICE = torch.device("cpu")
        else:
            device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
            torch.npu.set_device(device_id)
            _DEVICE = torch.device(f"npu:{device_id}")
    return _DEVICE


# ─────────────────────────────────────────────
# Golden 参考实现（NPU torch）
# ─────────────────────────────────────────────

def gdr_fwd_train_wy_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
):
    """NPU FP32 golden for OP-A (M1-M4 WY-build front-end).

    Args:
        q, k : [B, T, H, K]  (bf16/fp32; promoted to fp32)
        v    : [B, T, H, V]
        g    : [B, T, H]     (log-space decay, <= 0)
        beta : [B, T, H]     (sigmoid LOGITS)

    Returns:
        tuple of 17 float32 tensors, in the documented output order (internal [B,H,N,...] layout).
    """
    device = _get_device()
    C = _CHUNK_SIZE

    q = q.to(device).to(torch.float32)
    k = k.to(device).to(torch.float32)
    v = v.to(device).to(torch.float32)
    g = g.to(device).to(torch.float32)
    beta_logit = beta.to(device).to(torch.float32)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    beta_act_full = torch.sigmoid(beta_logit)                     # [B,T,H]

    # public [B,T,H,*] -> internal [B,H,T,*]
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()
    beta_a = beta_act_full.transpose(1, 2).contiguous()          # [B,H,T]

    # zero-pad T -> Tp = N*C
    T = q.shape[-2]
    pad = (C - (T % C)) % C
    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        g = F.pad(g, (0, pad))
        beta_a = F.pad(beta_a, (0, pad))                         # pad rows: beta_act = 0
    b, h, Tp, K = q.shape
    V = v.shape[-1]
    N = Tp // C

    # ===== M1 preprocess (s0-s3, Vector) =====
    q_s = (q * scale).reshape(b, h, N, C, K)
    k_f32 = k.reshape(b, h, N, C, K)
    v_b = (v * beta_a[..., None]).reshape(b, h, N, C, V)
    k_beta = (k * beta_a[..., None]).reshape(b, h, N, C, K)      # internal scratch
    g_f32 = g.reshape(b, h, N, C)
    beta_act = beta_a.reshape(b, h, N, C)
    sig = beta_act                                              # sig == beta_act
    q_t = q_s.transpose(-1, -2).contiguous()                    # [b,h,N,K,C]

    # ===== M2 cumsum_qk (s5, s7mm, s11mm, Cube) =====
    decay = g_f32.cumsum(-1)                                     # [b,h,N,C]
    kf_t = k_f32.transpose(-1, -2)
    P1 = torch.matmul(k_beta, kf_t)                           # k_beta @ k^T  internal
    P2 = torch.matmul(q_s, kf_t)                              # q_s   @ k^T  internal

    # ===== M3 wy_build (s6, s7mask, s8, s9, s11mask, Vector) =====
    decay_exp = decay.exp()                                      # [b,h,N,C]
    q_de = q_s * decay_exp[..., None]
    kbeta_de = k_beta * decay_exp[..., None]                   # internal scratch
    dfac = (decay[..., -1:] - decay).exp()[..., None].expand(b, h, N, C, K).contiguous()
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp()).tril()

    up_incl_diag = torch.triu(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=0)
    attn = -(P1 * L_mask).masked_fill(up_incl_diag, 0)
    for i in range(1, C):                                        # s8 forward substitution
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(C, dtype=torch.float32, device=device)

    up_strict = torch.triu(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=1)
    a_mat = (P2 * L_mask).masked_fill(up_strict, 0)            # tril(P2*L_mask) incl diag (== A)
    kbeta_t = kbeta_de.transpose(-1, -2).contiguous()           # [b,h,N,K,C]

    # ===== M4 uw (s10, Cube) =====
    u = torch.matmul(attn, v_b)                                # chunk k_cumsum   [b,h,N,C,V]
    w = torch.matmul(attn, kbeta_de)                         # chunk k_cumdecay [b,h,N,C,K]
    w_t = w.transpose(-1, -2).contiguous()                    # [b,h,N,K,C]

    return (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
            dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t)


# ==========================================
# 输入构造（供验证使用；启用性能采集时供 profiling --factory 复用）
# ==========================================

# acceptance cases: (name, B,T,H,K,V)
_P0 = [
    ("aligned_b1_t128_h4_k64_v64",   1, 128, 4, 64, 64),
    ("ttail_b1_t100_h4_k64_v64",     1, 100, 4, 64, 64),
    ("t3c_b1_t192_h2_k128_v128",     1, 192, 2, 128, 128),
    ("tail_vlt_b1_t200_h3_k128_v96", 1, 200, 3, 128, 96),
]


def _make_inputs(device):
    """The four acceptance cases. Signature order q,k,v,g,beta -> positional args."""
    cases = []
    for (name, B, T, H, K, V) in _P0:
        q = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
        k = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
        v = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)
        g = torch.empty((B, T, H), dtype=torch.float32, device=device).uniform_(-16.0, 0.0)
        beta = torch.randn((B, T, H), dtype=torch.bfloat16, device=device)
        cases.append((name, [q, k, v, g, beta], {}))
    return cases


# ==========================================
# 验证
# ==========================================

_OUT_NAMES = ["q_s", "k_f32", "v_b", "g_f32", "beta_act", "sig", "q_t", "decay",
              "decay_exp", "dfac", "q_de", "attn", "a_mat", "kbeta_t", "u", "w", "w_t"]


def _expected_shapes(B, T, H, K, V):
    N = (T + 63) // 64
    C = 64
    return {
        "q_s": (B, H, N, C, K), "k_f32": (B, H, N, C, K), "v_b": (B, H, N, C, V),
        "g_f32": (B, H, N, C), "beta_act": (B, H, N, C), "sig": (B, H, N, C),
        "q_t": (B, H, N, K, C), "decay": (B, H, N, C), "decay_exp": (B, H, N, C),
        "dfac": (B, H, N, C, K), "q_de": (B, H, N, C, K), "attn": (B, H, N, C, C),
        "a_mat": (B, H, N, C, C), "kbeta_t": (B, H, N, K, C), "u": (B, H, N, C, V),
        "w": (B, H, N, C, K), "w_t": (B, H, N, K, C),
    }


def _validate():
    device = _get_device()
    print("=" * 64)
    print("gdr_fwd_train_wy_golden 验证报告")
    print("=" * 64)
    print(f"Device: {device}")
    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    # -- 1. 全部 P0：count / shape / dtype / finite --
    print("\n[1. P0 shape / dtype / finite]")
    cases = _make_inputs(device)
    p0_meta = {name: (B, T, H, K, V) for (name, B, T, H, K, V) in _P0}
    seen = set()
    for (name, args, kwargs) in cases:
        _check(f"{name}: no duplicate case", name not in seen)
        seen.add(name)
        B, T, H, K, V = p0_meta[name]
        exp = _expected_shapes(B, T, H, K, V)
        outs = gdr_fwd_train_wy_golden(*args, **kwargs)
        ok = _check(f"{name}: 17 outputs", len(outs) == 17)
        for nm, t in zip(_OUT_NAMES, outs):
            ok &= (tuple(t.shape) == exp[nm]) and (t.dtype == torch.float32) \
                and bool(torch.isfinite(t).all())
        _check(f"{name}: shapes/dtype(fp32)/finite", ok)
    _check("all 4 P0 cases present, none missing", seen == set(p0_meta.keys()))

    # -- 2. 内部代数恒等式 --
    print("\n[2. 内部代数恒等式]")
    torch.manual_seed(42)
    B, T, H, K, V = 1, 128, 4, 64, 64
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)
    g = torch.empty((B, T, H), dtype=torch.float32, device=device).uniform_(-16.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device=device)
    (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
     dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t) = gdr_fwd_train_wy_golden(q, k, v, g, beta)
    _check("sig == beta_act", torch.equal(sig, beta_act))
    _check("q_t == q_s^T", torch.allclose(q_t, q_s.transpose(-1, -2), atol=0, rtol=0))
    _check("w_t == w^T", torch.allclose(w_t, w.transpose(-1, -2), atol=0, rtol=0))
    _check("q_de == q_s*decay_exp", torch.allclose(q_de, q_s * decay_exp[..., None], atol=1e-4, rtol=1e-4))
    _check("u == attn@v_b", torch.allclose(u, torch.matmul(attn, v_b), atol=1e-3, rtol=1e-3))
    C = 64
    up_str = torch.triu(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=1)
    _check("a_mat strict-upper == 0", a_mat.masked_select(up_str).abs().max().item() == 0.0)
    _check("attn strict-upper == 0", attn.masked_select(up_str).abs().max().item() == 0.0)
    diag = torch.arange(C, device=device)
    _check("attn diag == 1",
           torch.allclose(attn[..., diag, diag], torch.ones_like(attn[..., diag, diag]), atol=1e-5, rtol=0))

    # -- 3. 值域检查 --
    print("\n[3. 值域检查]")
    _check("beta_act in [0,1]", (beta_act.min().item() >= 0.0) and (beta_act.max().item() <= 1.0))
    _check("decay_exp in (0,1]", (decay_exp.min().item() >= 0.0) and (decay_exp.max().item() <= 1.0 + 1e-5))
    _check("dfac in (0,1]", (dfac.min().item() >= 0.0) and (dfac.max().item() <= 1.0 + 1e-5))
    _check("g_f32 <= 0", g_f32.max().item() <= 1e-5)

    # -- 4. padding 语义 (T=100) + 数值稳定性 --
    print("\n[4. padding 语义 + 数值稳定性]")
    torch.manual_seed(1)
    q = torch.randn((1, 100, 4, 64), dtype=torch.bfloat16, device=device)
    k = torch.randn((1, 100, 4, 64), dtype=torch.bfloat16, device=device)
    v = torch.randn((1, 100, 4, 64), dtype=torch.bfloat16, device=device)
    g = torch.empty((1, 100, 4), dtype=torch.float32, device=device).uniform_(-16.0, 0.0)
    beta = torch.randn((1, 100, 4), dtype=torch.bfloat16, device=device)
    outs = gdr_fwd_train_wy_golden(q, k, v, g, beta)
    ba = outs[4]  # beta_act [B,H,N,C]; T=100 -> chunk1 rows 36..63 are pad
    _check("T=100: beta_act pad rows == 0", ba[:, :, 1, 36:].abs().max().item() == 0.0)
    _check("T=100: all outputs finite", all(torch.isfinite(t).all() for t in outs))

    torch.manual_seed(3)
    q = torch.randn((1, 128, 2, 64), dtype=torch.bfloat16, device=device)
    k = torch.randn((1, 128, 2, 64), dtype=torch.bfloat16, device=device)
    v = torch.randn((1, 128, 2, 64), dtype=torch.bfloat16, device=device)
    g = torch.empty((1, 128, 2), dtype=torch.float32, device=device).uniform_(-16.0, -8.0)
    beta = torch.randn((1, 128, 2), dtype=torch.bfloat16, device=device)
    outs = gdr_fwd_train_wy_golden(q, k, v, g, beta)
    _check("very-negative g: all finite", all(torch.isfinite(t).all() for t in outs))

    print("\n" + "=" * 64)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 64)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
