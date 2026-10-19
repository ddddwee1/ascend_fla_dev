#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gated_delta_rule_bwd CPU golden reference (higher precision, FP32).

Ultimate ground truth for precision comparison in later stages. Same math as the NPU
golden (gated_delta_rule_bwd_golden.py) but:
  - pure CPU, no torch_npu / no device selection
  - all five gradients returned in FP32 (NOT narrowed to the bf16 contract dtypes);
    this is the higher-precision reference the kernels are compared against.

──────────────────────────────────────────────────────────────────────────────
GOLDEN ORACLE LINEAGE
  The backward golden differentiates the EXACT oracle the FORWARD golden reproduces:
  the fp32 per-token recurrence `_recurrent(q, k, v, beta, g, scale)` transcribed
  VERBATIM from gated_delta_rule_fwd_golden_cpu.py. That
  oracle expects a PRE-ACTIVATED beta; the operator contract is
  use_beta_sigmoid_in_kernel=True (allow_neg_eigval=False), so we apply
  beta = sigmoid(beta_logit) INSIDE the differentiated graph. Therefore:
    - autograd's grad w.r.t. the beta LEAF is the grad w.r.t. the LOGIT (dbeta),
      NOT d/d sigmoid(beta).                              (confirmed by FD below)
    - g enters as exp(g) (log-space decay), so autograd's grad w.r.t. g is w.r.t.
      LOG-SPACE g (dg), NOT grad w.r.t. exp(g).           (confirmed by FD below)
  Ground truth:  (dq,dk,dv,dg,dbeta) = torch.autograd.grad(o, [q,k,v,g,beta_logit], grad_o)
  where  o, _ = _recurrent(q, k, v, sigmoid(beta_logit), g, scale=1/sqrt(K)).

Public signature:
    gated_delta_rule_bwd_golden_cpu(grad_o, q, k, v, g, beta) -> (dq, dk, dv, dg, dbeta)
      grad_o : [B, T, H, V]   (any float dtype; promoted to fp32)
      q, k   : [B, T, H, K]
      v      : [B, T, H, V]
      g      : [B, T, H]      (log-space decay, <= 0)
      beta   : [B, T, H]      (sigmoid LOGITS)
    Returns all five gradients as FP32 (dq,dk,dv,dg,dbeta); dbeta is w.r.t. the LOGIT,
    dg is w.r.t. LOG-SPACE g.
"""

import torch

_CHUNK_SIZE = 64


# ─────────────────────────────────────────────
# Forward oracle — transcribed VERBATIM from the forward golden `_recurrent`.
# `beta` is PRE-ACTIVATED on entry (sigmoid applied by the caller). Public layout
# [B, T, H, *]; all math in float32. Autograd-safe (explicit .clone() at every carry).
# ─────────────────────────────────────────────

def _recurrent(q, k, v, beta, g, scale):
    """Per-token ground-truth recurrence (returns o[B,T,H,V] fp32, S[B,H,K,V] fp32).

    Verbatim semantics of naive_recurrent_gated_delta_rule; `beta` is pre-activated.
    einsum replaced by an explicit broadcast-sum for portability across backends.
    """
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                        for x in (q, k, v, beta, g)]
    B, H, T, K = k.shape
    V = v.shape[-1]
    o = torch.zeros(B, H, T, V, device=q.device, dtype=torch.float32)
    h = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        # o[i] = einsum('bhd,bhdm->bhm', b_q, h)
        o[:, :, i] = (b_q[..., None] * h).sum(-2)
    o = o.transpose(1, 2).contiguous()
    return o, h


def _oracle_o(q, k, v, g, beta_logit, scale):
    """The EXACT forward oracle output o, wired for the backward VJP.

    sigmoid(beta_logit) applied INSIDE the differentiated graph -> grad w.r.t. beta_logit
    is dbeta (grad w.r.t. logit); g enters as exp(g) -> grad w.r.t. g is dg (log-space).
    Returns o [B,T,H,V] fp32.
    """
    beta_act = torch.sigmoid(beta_logit)
    o, _ = _recurrent(q, k, v, beta_act, g, scale)
    return o


# ─────────────────────────────────────────────
# CPU Golden 参考实现（更高精度，FP32）
# ─────────────────────────────────────────────

def gated_delta_rule_bwd_golden_cpu(
    grad_o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
):
    """CPU FP32 golden for Gated Delta Rule backward (VJP of the forward oracle).

    Args:
        grad_o : [B, T, H, V]  (any float dtype; promoted to fp32)
        q, k   : [B, T, H, K]
        v      : [B, T, H, V]
        g      : [B, T, H]     (log-space decay, <= 0)
        beta   : [B, T, H]     (sigmoid LOGITS)

    Returns:
        (dq, dk, dv, dg, dbeta), ALL fp32 (NOT narrowed):
            dq, dk : [B, T, H, K]
            dv     : [B, T, H, V]
            dg     : [B, T, H]     (grad w.r.t. log-space g)
            dbeta  : [B, T, H]     (grad w.r.t. the sigmoid LOGIT)
    """
    qf = q.to(torch.float32).detach().requires_grad_(True)
    kf = k.to(torch.float32).detach().requires_grad_(True)
    vf = v.to(torch.float32).detach().requires_grad_(True)
    gf = g.to(torch.float32).detach().requires_grad_(True)
    betaf = beta.to(torch.float32).detach().requires_grad_(True)
    grad_o_f = grad_o.to(torch.float32)

    scale = 1.0 / (qf.shape[-1] ** 0.5)
    o = _oracle_o(qf, kf, vf, gf, betaf, scale)

    dq, dk, dv, dg, dbeta = torch.autograd.grad(
        o, [qf, kf, vf, gf, betaf], grad_outputs=grad_o_f)

    return (dq.to(torch.float32), dk.to(torch.float32), dv.to(torch.float32),
            dg.to(torch.float32), dbeta.to(torch.float32))


# ==========================================
# 验证
# ==========================================

def _mk_bwd_case(B, T, H, K, V, g_lo=-4.0, g_hi=0.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(g_lo, g_hi)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return [grad_o, q, k, v, g, beta]


_P0_CONFIGS = [
    ("aligned_b1_t128_h4_k64_v64", (1, 128, 4, 64, 64)),
    ("ttail_b1_t100_h4_k64_v64",   (1, 100, 4, 64, 64)),
    ("t3c_b1_t192_h2_k128_v128",   (1, 192, 2, 128, 128)),
    ("tail_vlt_b1_t200_h3_k128_v96", (1, 200, 3, 128, 96)),
]


def _validate():
    print("=" * 60)
    print("gated_delta_rule_bwd_golden_cpu 验证报告")
    print("=" * 60)

    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        status = "PASS" if cond else "FAIL"
        if not cond:
            all_pass = False
        print(f"  [{status}] {name}")
        return cond

    out_names = ("dq", "dk", "dv", "dg", "dbeta")

    # 1. All P0 shapes: runs, output is fp32, shapes correct, finite.
    print("\n[1. 合同 P0 用例: shape / fp32 / finite]")
    for name, (B, T, H, K, V) in _P0_CONFIGS:
        args = _mk_bwd_case(B, T, H, K, V, seed=42)
        outs = gated_delta_rule_bwd_golden_cpu(*args)
        exp = {"dq": (B, T, H, K), "dk": (B, T, H, K), "dv": (B, T, H, V),
               "dg": (B, T, H), "dbeta": (B, T, H)}
        _check(f"{name}: returns 5 grads", len(outs) == 5)
        for nm, t in zip(out_names, outs):
            _check(f"{name}.{nm}: shape {tuple(t.shape)} == {exp[nm]}", tuple(t.shape) == exp[nm])
            _check(f"{name}.{nm}: dtype fp32", t.dtype == torch.float32)
            _check(f"{name}.{nm}: finite", torch.isfinite(t).all().item())

    # 2. Both bf16 and fp32 inputs accepted; output always fp32.
    print("\n[2. bf16 与 fp32 输入 -> fp32 输出]")
    for tag, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        torch.manual_seed(0)
        B, T, H, K, V = 1, 64, 2, 32, 32
        grad_o = torch.randn((B, T, H, V), dtype=dtype)
        q = torch.randn((B, T, H, K), dtype=dtype)
        k = torch.randn((B, T, H, K), dtype=dtype)
        v = torch.randn((B, T, H, V), dtype=dtype)
        g = torch.empty((B, T, H), dtype=torch.float32).uniform_(-4.0, 0.0)
        beta = torch.randn((B, T, H), dtype=dtype)
        outs = gated_delta_rule_bwd_golden_cpu(grad_o, q, k, v, g, beta)
        _check(f"{tag}: all outputs fp32", all(t.dtype == torch.float32 for t in outs))
        _check(f"{tag}: all outputs finite", all(torch.isfinite(t).all().item() for t in outs))

    # 3. Finite-difference lineage check (dbeta w.r.t LOGIT, dg w.r.t LOG-SPACE g).
    print("\n[3. 有限差分血统验证 (fp32; 确认 dbeta 对 logit, dg 对 log-space g)]")
    _fd_check(_check)

    # 4. Numerical edge: grad_o == 0 -> all grads == 0.
    print("\n[4. 数值边界]")
    args = _mk_bwd_case(1, 96, 2, 64, 64, seed=7)
    args[0] = torch.zeros_like(args[0])
    outs = gated_delta_rule_bwd_golden_cpu(*args)
    _check("grad_o == 0 -> all grads == 0",
           all(t.abs().max().item() == 0.0 for t in outs))
    # very-negative g finite.
    args = _mk_bwd_case(1, 128, 2, 64, 64, g_lo=-16.0, g_hi=-8.0, seed=3)
    outs = gated_delta_rule_bwd_golden_cpu(*args)
    _check("very-negative g (-16,-8): all grads finite",
           all(torch.isfinite(t).all().item() for t in outs))

    # 5. Gradient magnitude ranges (downstream precision-criterion evidence).
    print("\n[5. 梯度量级范围 (供下游精度判据)]")
    for name, (B, T, H, K, V) in _P0_CONFIGS:
        args = _mk_bwd_case(B, T, H, K, V, seed=42)
        outs = gated_delta_rule_bwd_golden_cpu(*args)
        mags = {nm: t.abs().max().item() for nm, t in zip(out_names, outs)}
        print(f"  {name}: " + ", ".join(f"|{nm}|max={mags[nm]:.3e}" for nm in out_names))

    print("\n" + "=" * 60)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 60)
    return all_pass


def _fd_check(_check):
    """Central finite-difference cross-check of all five gradients on a small shape."""
    torch.manual_seed(123)
    B, T, H, K, V = 1, 6, 1, 5, 4
    scale = 1.0 / (K ** 0.5)
    q = torch.randn((B, T, H, K), dtype=torch.float32)
    k = torch.randn((B, T, H, K), dtype=torch.float32)
    v = torch.randn((B, T, H, V), dtype=torch.float32)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(-2.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.float32)  # logits
    grad_o = torch.randn((B, T, H, V), dtype=torch.float32)

    def loss(qq, kk, vv, gg, bb):
        o = _oracle_o(qq, kk, vv, gg, bb, scale)
        return (o * grad_o).sum()

    qa = q.clone().requires_grad_(True)
    ka = k.clone().requires_grad_(True)
    va = v.clone().requires_grad_(True)
    ga = g.clone().requires_grad_(True)
    ba = beta.clone().requires_grad_(True)
    L = loss(qa, ka, va, ga, ba)
    gq, gk, gv, gg_, gb = torch.autograd.grad(L, [qa, ka, va, ga, ba])
    analytic = {"dq": gq, "dk": gk, "dv": gv, "dg": gg_, "dbeta": gb}
    tensors = {"dq": q, "dk": k, "dv": v, "dg": g, "dbeta": beta}

    eps = 1e-3
    worst = {}
    for nm in ("dq", "dk", "dv", "dg", "dbeta"):
        base = tensors[nm]
        flat = base.reshape(-1)
        n = flat.numel()
        # Probe the coords with the LARGEST analytic gradient: central FD is most
        # meaningful and least corrupted by catastrophic cancellation there.
        idxs = torch.topk(analytic[nm].reshape(-1).abs(), k=min(6, n)).indices.tolist()
        max_rel = 0.0
        for idx in idxs:
            pert = flat.clone(); pert[idx] += eps
            tp = pert.reshape(base.shape)
            pert2 = flat.clone(); pert2[idx] -= eps
            tm = pert2.reshape(base.shape)
            args_p = {"dq": [tp, k, v, g, beta], "dk": [q, tp, v, g, beta],
                      "dv": [q, k, tp, g, beta], "dg": [q, k, v, tp, beta],
                      "dbeta": [q, k, v, g, tp]}[nm]
            args_m = {"dq": [tm, k, v, g, beta], "dk": [q, tm, v, g, beta],
                      "dv": [q, k, tm, g, beta], "dg": [q, k, v, tm, beta],
                      "dbeta": [q, k, v, g, tm]}[nm]
            with torch.no_grad():
                num = (loss(*args_p).item() - loss(*args_m).item()) / (2 * eps)
            ana = analytic[nm].reshape(-1)[idx].item()
            denom = max(abs(ana), abs(num), 1e-4)
            max_rel = max(max_rel, abs(num - ana) / denom)
        worst[nm] = max_rel
        _check(f"FD {nm}: max_rel={max_rel:.3e} (thr < 2e-2)", max_rel < 2e-2)
    print(f"    lineage confirmed: dbeta w.r.t sigmoid LOGIT (FD rel={worst['dbeta']:.2e}), "
          f"dg w.r.t LOG-SPACE g (FD rel={worst['dg']:.2e})")
    return worst


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        exit(1)
