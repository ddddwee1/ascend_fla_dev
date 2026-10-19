#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gated_delta_rule_bwd golden reference implementation (NPU torch).

Backward (gradient) pass of Gated Delta Rule (GDN) linear attention. SELF-CONTAINED and
BACKWARD-ONLY: consumes only grad_o and the PRIMARY inputs (q,k,v,g,beta); emits gradients
only (dq,dk,dv,dg,dbeta); does NOT read any residual from the forward op.

──────────────────────────────────────────────────────────────────────────────
GOLDEN ORACLE LINEAGE
  The backward golden differentiates the EXACT oracle the FORWARD golden reproduces:
  the fp32 per-token recurrence `_recurrent(q, k, v, beta, g, scale)` transcribed
  VERBATIM from gated_delta_rule_fwd_golden.py. That oracle
  expects a PRE-ACTIVATED beta; the forward operator contract is
  use_beta_sigmoid_in_kernel=True (allow_neg_eigval=False), so we apply
  beta = sigmoid(beta_logit) INSIDE the differentiated graph. Because of that:
    - autograd's grad w.r.t. the beta LEAF is automatically the grad w.r.t. the LOGIT
      (dbeta) — NOT d/d sigmoid(beta).  (confirmed by finite difference below)
    - g is used as exp(g) inside the recurrence (log-space decay), so autograd's grad
      w.r.t. g is w.r.t. LOG-SPACE g (dg) — NOT grad w.r.t. exp(g).  (confirmed by FD)
  Ground truth:  (dq,dk,dv,dg,dbeta) = torch.autograd.grad(o, [q,k,v,g,beta_logit], grad_o)
  where  o, _ = _recurrent(q, k, v, sigmoid(beta_logit), g, scale=1/sqrt(K)).
  This is the exact reverse-mode VJP of the exact forward oracle.

Public signature:
    gated_delta_rule_bwd_golden(grad_o, q, k, v, g, beta) -> (dq, dk, dv, dg, dbeta)
      grad_o : [B, T, H, V]  bf16   (upstream grad of o)
      q, k   : [B, T, H, K]  bf16
      v      : [B, T, H, V]  bf16
      g      : [B, T, H]     fp32    (log-space decay, semantically <= 0)
      beta   : [B, T, H]     bf16    (sigmoid LOGITS)
      dq, dk : [B, T, H, K]  bf16
      dv     : [B, T, H, V]  bf16
      dg     : [B, T, H]     fp32
      dbeta  : [B, T, H]     fp32
  scale = 1/sqrt(K); no q/k L2 norm; H == HV; no initial/final state (v1). C=64 is a
  Stage-3 internal tiling constant only — the numeric ground truth is the natural-exp VJP
  above, which is per-token and needs no chunking.

  All arithmetic is fp32 (inputs promoted to fp32 leaves so autograd accumulates in fp32);
  only dq/dk/dv are narrowed fp32 -> bf16 on return; dg/dbeta stay fp32.

torch_npu is required; computation runs on NPU. Falls back to CPU only when no NPU
hardware is available.
"""

import os
import torch

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
# Canonical mathematical semantics.
# ─────────────────────────────────────────────
_CANONICAL_FORMULA = (
    "Backward (VJP) of gated_delta_rule. Differentiate the EXACT forward oracle "
    "o = _recurrent(q,k,v,sigmoid(beta_logit),g,scale=1/sqrt(K)) and contract with grad_o: "
    "(dq,dk,dv,dg,dbeta) = autograd.grad(o, [q,k,v,g,beta_logit], grad_o). "
    "dbeta is w.r.t. the sigmoid LOGIT; dg is w.r.t. LOG-SPACE g."
)

_CHUNK_SIZE = 64


# ─────────────────────────────────────────────
# Forward oracle — transcribed VERBATIM from gated_delta_rule_fwd_golden.py `_recurrent`.
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

    sigmoid(beta_logit) is applied INSIDE the differentiated graph, so grad w.r.t.
    beta_logit == dbeta (grad w.r.t. the logit). g enters as exp(g), so grad w.r.t. g ==
    dg (grad w.r.t. log-space g). Reuses _recurrent verbatim. Returns o [B,T,H,V] fp32.
    """
    beta_act = torch.sigmoid(beta_logit)
    o, _ = _recurrent(q, k, v, beta_act, g, scale)
    return o


# ─────────────────────────────────────────────
# Public golden (reverse-mode VJP of the forward oracle).
# ─────────────────────────────────────────────

def gated_delta_rule_bwd_golden(
    grad_o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
):
    """Gated Delta Rule backward (VJP), NPU torch reference.

    Args:
        grad_o : [B, T, H, V] bf16  (upstream grad of o)
        q, k   : [B, T, H, K] bf16
        v      : [B, T, H, V] bf16
        g      : [B, T, H]    fp32  (log-space decay, <= 0)
        beta   : [B, T, H]    bf16  (sigmoid LOGITS)

    Returns:
        (dq, dk, dv, dg, dbeta):
            dq, dk : [B, T, H, K] bf16
            dv     : [B, T, H, V] bf16
            dg     : [B, T, H]    fp32
            dbeta  : [B, T, H]    fp32   (grad w.r.t. the sigmoid LOGIT)
    """
    device = _get_device()

    # Promote primary inputs to fp32 leaves so autograd accumulates in fp32.
    qf = q.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    kf = k.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    vf = v.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    gf = g.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    betaf = beta.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    grad_o_f = grad_o.to(device=device, dtype=torch.float32)

    scale = 1.0 / (qf.shape[-1] ** 0.5)
    o = _oracle_o(qf, kf, vf, gf, betaf, scale)

    dq, dk, dv, dg, dbeta = torch.autograd.grad(
        o, [qf, kf, vf, gf, betaf], grad_outputs=grad_o_f)

    # Output contract: dq/dk/dv -> bf16; dg/dbeta stay fp32.
    return (dq.to(torch.bfloat16), dk.to(torch.bfloat16), dv.to(torch.bfloat16),
            dg.to(torch.float32), dbeta.to(torch.float32))


# ==========================================
# 输入构造（供验证使用；启用性能采集时供 profiling --factory 复用）
# ==========================================

def _mk_bwd_case(device, B, T, H, K, V, g_lo=-4.0, g_hi=0.0, seed=None):
    """Build one backward input tuple [grad_o, q, k, v, g, beta].

    Value ranges:
      grad_o,q,k,v : bf16 (randn, well within [-10,10])
      g            : fp32 in a NEGATIVE sub-range [g_lo,g_hi] (log-space forget gate, MUST
                     be <= 0; positive g would overflow exp() over the recurrence)
      beta         : bf16 sigmoid LOGITS (randn; sigmoid applied inside the golden)
    """
    if seed is not None:
        torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16, device=device)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)
    g = torch.empty((B, T, H), dtype=torch.float32, device=device).uniform_(g_lo, g_hi)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device=device)
    return [grad_o, q, k, v, g, beta]


# Acceptance cases: (name, (B,T,H,K,V)).
_P0_CONFIGS = [
    ("aligned_b1_t128_h4_k64_v64", (1, 128, 4, 64, 64)),
    ("ttail_b1_t100_h4_k64_v64",   (1, 100, 4, 64, 64)),
    ("t3c_b1_t192_h2_k128_v128",   (1, 192, 2, 128, 128)),
    ("tail_vlt_b1_t200_h3_k128_v96", (1, 200, 3, 128, 96)),
]


def _make_inputs(device):
    """Construct every contract P0 case for the backward op.

    Returns [(case_name, [grad_o,q,k,v,g,beta], {}), ...].
    """
    cases = []
    for name, (B, T, H, K, V) in _P0_CONFIGS:
        # Fixed seed per case so _validate and any profiling reuse the SAME inputs.
        args = _mk_bwd_case(device, B, T, H, K, V, seed=42)
        cases.append((name, args, {}))
    return cases


# Expected output shapes per P0 case: (dq, dk, dv, dg, dbeta).
_P0_OUTPUT_SHAPES = {
    name: {
        "dq": (B, T, H, K), "dk": (B, T, H, K), "dv": (B, T, H, V),
        "dg": (B, T, H), "dbeta": (B, T, H),
    }
    for name, (B, T, H, K, V) in _P0_CONFIGS
}


# ==========================================
# 验证
# ==========================================

def _validate():
    device = _get_device()
    print("=" * 60)
    print("gated_delta_rule_bwd_golden 验证报告")
    print("=" * 60)
    print(f"Device: {device}")

    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        status = "PASS" if cond else "FAIL"
        if not cond:
            all_pass = False
        print(f"  [{status}] {name}")
        return cond

    out_names = ("dq", "dk", "dv", "dg", "dbeta")
    out_dtypes = {
        "dq": torch.bfloat16, "dk": torch.bfloat16, "dv": torch.bfloat16,
        "dg": torch.float32, "dbeta": torch.float32,
    }

    # ---------------------------------------------------------------
    # 1. Contract P0 cases: exact set, shape, dtype, finiteness
    # ---------------------------------------------------------------
    print("\n[1. 合同 P0 用例: shape / dtype / finite]")
    raw_cases = _make_inputs(device)
    assert isinstance(raw_cases, list) and raw_cases, "_make_inputs returned no cases"
    names = [c[0] for c in raw_cases]
    _check("no duplicate case names", len(names) == len(set(names)))
    expected = set(_P0_OUTPUT_SHAPES)
    _check(f"exact P0 set == {sorted(expected)}", set(names) == expected)

    for case_name, args, kwargs in raw_cases:
        outs = gated_delta_rule_bwd_golden(*args, **kwargs)
        _check(f"{case_name}: returns 5 grads", len(outs) == 5)
        exp = _P0_OUTPUT_SHAPES[case_name]
        for nm, t in zip(out_names, outs):
            _check(f"{case_name}.{nm}: shape {tuple(t.shape)} == {exp[nm]}",
                   tuple(t.shape) == exp[nm])
            _check(f"{case_name}.{nm}: dtype {t.dtype} == {out_dtypes[nm]}",
                   t.dtype == out_dtypes[nm])
            _check(f"{case_name}.{nm}: finite", torch.isfinite(t).all().item())

    # ---------------------------------------------------------------
    # 2. Finite-difference sanity (lineage: dbeta w.r.t LOGIT, dg w.r.t LOG-SPACE g)
    #    Small shape, central differences of L = sum(o * grad_o) vs analytic grads.
    # ---------------------------------------------------------------
    print("\n[2. 有限差分血统验证 (fp32; 确认 dbeta 对 logit, dg 对 log-space g)]")
    fd_report = _fd_check(device, _check)

    # ---------------------------------------------------------------
    # 3. CPU-fp32 vs NPU golden agreement, per output per P0 shape
    # ---------------------------------------------------------------
    print("\n[3. CPU-fp32 vs NPU golden 一致性 (每输出每 shape)]")
    _cpu_vs_npu_check(device, _check)

    # ---------------------------------------------------------------
    # 4. Numerical edge: all-zero grad_o -> all grads == 0 (linear in grad_o)
    # ---------------------------------------------------------------
    print("\n[4. 数值边界: grad_o == 0 -> 全部梯度为 0]")
    args = _mk_bwd_case(device, 1, 96, 2, 64, 64, seed=7)
    args[0] = torch.zeros_like(args[0])  # zero grad_o
    outs = gated_delta_rule_bwd_golden(*args)
    _check("all grads == 0 when grad_o == 0",
           all(t.float().abs().max().item() == 0.0 for t in outs))

    # very-negative g stays finite (exp underflow, no NaN)
    args = _mk_bwd_case(device, 1, 128, 2, 64, 64, g_lo=-16.0, g_hi=-8.0, seed=3)
    outs = gated_delta_rule_bwd_golden(*args)
    _check("very-negative g (-16,-8): all grads finite",
           all(torch.isfinite(t).all().item() for t in outs))

    # ---------------------------------------------------------------
    # 5. Gradient magnitude ranges (downstream precision-criterion evidence)
    # ---------------------------------------------------------------
    print("\n[5. 梯度量级范围 (供下游精度判据)]")
    for case_name, args, kwargs in raw_cases:
        outs = gated_delta_rule_bwd_golden(*args, **kwargs)
        mags = {nm: t.float().abs().max().item() for nm, t in zip(out_names, outs)}
        print(f"  {case_name}: " +
              ", ".join(f"|{nm}|max={mags[nm]:.3e}" for nm in out_names))

    print("\n" + "=" * 60)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 60)
    if not all_pass:
        raise SystemExit(1)


def _fd_check(device, _check):
    """Central finite-difference cross-check of all five gradients on a small shape."""
    torch.manual_seed(123)
    B, T, H, K, V = 1, 6, 1, 5, 4
    scale = 1.0 / (K ** 0.5)
    q = torch.randn((B, T, H, K), dtype=torch.float32, device=device)
    k = torch.randn((B, T, H, K), dtype=torch.float32, device=device)
    v = torch.randn((B, T, H, V), dtype=torch.float32, device=device)
    g = torch.empty((B, T, H), dtype=torch.float32, device=device).uniform_(-2.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.float32, device=device)  # logits
    grad_o = torch.randn((B, T, H, V), dtype=torch.float32, device=device)

    def loss(qq, kk, vv, gg, bb):
        o = _oracle_o(qq, kk, vv, gg, bb, scale)
        return (o * grad_o).sum()

    # analytic grads (fp32)
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
        # meaningful and least corrupted by catastrophic cancellation there (near-zero
        # gradient coords give a noisy relative error under NPU fp32, not a real defect).
        idxs = torch.topk(analytic[nm].reshape(-1).abs(), k=min(6, n)).indices.tolist()
        max_rel = 0.0
        for idx in idxs:
            pert = flat.clone()
            pert[idx] += eps
            tp = pert.reshape(base.shape)
            pert2 = flat.clone()
            pert2[idx] -= eps
            tm = pert2.reshape(base.shape)
            args_p = {"dq": [tp, k, v, g, beta], "dk": [q, tp, v, g, beta],
                      "dv": [q, k, tp, g, beta], "dg": [q, k, v, tp, beta],
                      "dbeta": [q, k, v, g, tp]}[nm]
            args_m = {"dq": [tm, k, v, g, beta], "dk": [q, tm, v, g, beta],
                      "dv": [q, k, tm, g, beta], "dg": [q, k, v, tm, beta],
                      "dbeta": [q, k, v, g, tm]}[nm]
            with torch.no_grad():
                lp = loss(*args_p).item()
                lm = loss(*args_m).item()
            num = (lp - lm) / (2 * eps)
            ana = analytic[nm].reshape(-1)[idx].item()
            denom = max(abs(ana), abs(num), 1e-4)
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
        worst[nm] = max_rel
        _check(f"FD {nm}: max_rel={max_rel:.3e} (thr < 2e-2)", max_rel < 2e-2)
    print(f"    lineage confirmed: dbeta is w.r.t. sigmoid LOGIT (FD rel={worst['dbeta']:.2e}), "
          f"dg is w.r.t. LOG-SPACE g (FD rel={worst['dg']:.2e})")
    return worst


def _cpu_vs_npu_check(device, _check):
    """Compare NPU golden (bf16 outputs upcast) against CPU fp32 golden, per output/shape."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gated_delta_rule_bwd_golden_cpu import gated_delta_rule_bwd_golden_cpu

    out_names = ("dq", "dk", "dv", "dg", "dbeta")
    for name, (B, T, H, K, V) in _P0_CONFIGS:
        # Build inputs on CPU (deterministic), feed the SAME tensors to both goldens.
        args_cpu = _mk_bwd_case(torch.device("cpu"), B, T, H, K, V, seed=456)
        outs_npu = gated_delta_rule_bwd_golden(*[a.clone() for a in args_cpu])
        outs_cpu = gated_delta_rule_bwd_golden_cpu(*[a.clone() for a in args_cpu])
        line = [f"  {name}:"]
        ok_all = True
        for nm, on, oc in zip(out_names, outs_npu, outs_cpu):
            a = on.float().cpu()
            b = oc.float().cpu()
            diff = (a - b).abs()
            max_abs = diff.max().item()
            max_rel = (diff / b.abs().clamp_min(1e-3)).max().item()
            # dq/dk/dv narrowed to bf16 on NPU -> allow bf16 roundoff; dg/dbeta tighter.
            thr = 5e-2 if nm in ("dq", "dk", "dv") else 5e-3
            peak = b.abs().max().item()
            # peak-scaled: accept when max_abs small relative to peak OR rel small.
            ok = torch.isfinite(a).all().item() and (
                max_abs <= thr * max(peak, 1.0) or max_rel <= thr)
            ok_all = ok_all and ok
            line.append(f"{nm}(abs={max_abs:.2e},rel={max_rel:.2e})")
        print(" ".join(line))
        _check(f"{name}: CPU-vs-NPU agreement within tolerance", ok_all)


if __name__ == "__main__":
    _validate()
