#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_finalize (OP7) golden reference — NPU (torch + torch_npu).

OP7 is the LAST operator of gated_delta_rule_bwd. Its five public outputs (dq,dk,dv,dg,dbeta)
ARE the whole backward deliverable, so the Stage-2 cross-check verifies ALL FIVE gradients.

MECHANICAL PROJECTION (a faithful `_m10` slice), NOT fresh math. This computes exactly the
finalize+store arithmetic (golden_steps b4b_dg, b5, b6, b7 — module M10) of the READ-ONLY
delivered cumulative staging golden

    gated_delta_rule_bwd_golden_stage12345678910.py  (_m10)

but consumes OP7's frozen GM contract directly. `_m10` is the full backward and equals the
fp32 CPU autograd VJP ground truth gated_delta_rule_bwd_golden_cpu.

──────────────────────────────────────────────────────────────────────────────
PROJECTION MAP  (M10 `_m10`  ->  OP7 here)

  b4b_dg  dg_sum   = dg_p + dg2                                  # [B,H,N,C]
  b5      dg_full  = triu_incl @ dg_sum                          # per-chunk reverse cumsum over C
                    triu_incl[r,c] = 1 iff c >= r  (upper-tri ones INCL diag, [C,C])
  b6      dbeta    = dbeta_wy * sig * (1 - sig)                  # chain through sigmoid logit; sig ONLY
  b7      un-pad + transpose + cast:
              reshape [B,H,N,C,*] -> [B,H,N*C,*]; slice [:, :, :T]; transpose (1,2) -> [B,T,H,*].
              dq/dk/dv cast fp32 -> bf16 at store; dg/dbeta stay fp32.

Mixed precision: all arithmetic fp32; the ONLY narrow is dq/dk/dv fp32->bf16 at store
(dg/dbeta fp32). The b7 transpose is a PURE REINDEX (reshape / slice / transpose — no value
change). grad_o is the primary [B,T,H,V] seed threaded ONLY as the sequence-length (T) /
public-geometry anchor: OP7's arithmetic does NOT read grad_o VALUES.

Public signature:
    gdr_bwd_finalize_golden(grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act)
      -> (dq, dk, dv, dg, dbeta)
    grad_o   : [B, T, H, V]        bf16  (VALUE UNUSED; only shape -> T / public geometry)
    dq_p     : [B, H, N, C, K] fp32  (<- OP5 gdr_bwd_dqkw)
    dk_in    : [B, H, N, C, K] fp32  (<- OP6b gdr_bwd_wy_reduce; == OP6b.dk)
    dv_in    : [B, H, N, C, V] fp32  (<- OP6b gdr_bwd_wy_reduce; == OP6b.dv)
    dg_p     : [B, H, N, C]    fp32  (<- OP5 gdr_bwd_dqkw)
    dg2      : [B, H, N, C]    fp32  (<- OP6b gdr_bwd_wy_reduce)
    dbeta_wy : [B, H, N, C]    fp32  (<- OP6b gdr_bwd_wy_reduce)
    sig      : [B, H, N, C]    fp32  (<- gdr_fwd_train_wy; sigmoid(beta_logit))
    beta_act : [B, H, N, C]    fp32  (<- gdr_fwd_train_wy; == sig; DECLARED but UNUSED)
    ->  dq  : [B, T, H, K] bf16   dk : [B, T, H, K] bf16   dv : [B, T, H, V] bf16
        dg  : [B, T, H]    fp32   dbeta : [B, T, H] fp32
"""

import os
import torch

_C = 64  # fixed internal chunk size

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
# Golden 参考实现（NPU torch）— M10 finalize_store mechanical projection
# ─────────────────────────────────────────────

def gdr_bwd_finalize_golden(
    grad_o: torch.Tensor,
    dq_p: torch.Tensor,
    dk_in: torch.Tensor,
    dv_in: torch.Tensor,
    dg_p: torch.Tensor,
    dg2: torch.Tensor,
    dbeta_wy: torch.Tensor,
    sig: torch.Tensor,
    beta_act: torch.Tensor,
):
    """NPU reference for OP7 (M10 finalize_store: b4b_dg, b5, b6, b7).

    Args:
        grad_o   : [B, T, H, V]        bf16  (VALUE UNUSED; only shape -> T / public geometry)
        dq_p     : [B, H, N, C, K] fp32  (<- OP5 gdr_bwd_dqkw)
        dk_in    : [B, H, N, C, K] fp32  (<- OP6b; == OP6b.dk)
        dv_in    : [B, H, N, C, V] fp32  (<- OP6b; == OP6b.dv)
        dg_p     : [B, H, N, C]    fp32  (<- OP5 gdr_bwd_dqkw)
        dg2      : [B, H, N, C]    fp32  (<- OP6b)
        dbeta_wy : [B, H, N, C]    fp32  (<- OP6b)
        sig      : [B, H, N, C]    fp32  (<- gdr_fwd_train_wy; sigmoid(beta_logit))
        beta_act : [B, H, N, C]    fp32  (<- gdr_fwd_train_wy; == sig; UNUSED)

    Returns:
        (dq, dk, dv, dg, dbeta):
            dq : [B, T, H, K] bf16   dk : [B, T, H, K] bf16   dv : [B, T, H, V] bf16
            dg : [B, T, H]    fp32   dbeta : [B, T, H] fp32
    """
    device = _get_device()
    C = _C

    # Only grad_o's SHAPE is read (T / public geometry); the value never enters the math.
    B, T, H, _V_pub = grad_o.shape

    dq_p = dq_p.to(torch.float32).to(device)
    dk_in = dk_in.to(torch.float32).to(device)
    dv_in = dv_in.to(torch.float32).to(device)
    dg_p = dg_p.to(torch.float32).to(device)
    dg2 = dg2.to(torch.float32).to(device)
    dbeta_wy = dbeta_wy.to(torch.float32).to(device)
    sig = sig.to(torch.float32).to(device)
    _ = beta_act  # parent-fidelity contract input; == sig; not used by the core math

    N = dq_p.shape[2]
    K = dq_p.shape[-1]
    V = dv_in.shape[-1]

    # ---- b4b_dg: dg_sum = dg_p + dg2 ----
    dg_sum = dg_p + dg2                                                    # [B,H,N,C]

    # ---- b5: dg_full = triu_incl @ dg_sum  (per-chunk reverse cumsum over C) ----
    triu_incl = torch.triu(torch.ones(C, C, dtype=torch.float32, device=device), diagonal=0)
    dg_full = torch.matmul(triu_incl, dg_sum.unsqueeze(-1)).squeeze(-1)    # [B,H,N,C]

    # ---- b6: dbeta = dbeta_wy * sigma * (1 - sigma) ----
    dbeta = dbeta_wy * sig * (1.0 - sig)                                   # [B,H,N,C]

    # ---- b7: un-pad ([:T]) + transpose [B,H,N,C,*] -> public [B,T,H,*] (pure reindex) + cast ----
    def _unpad_t(x):
        d = x.shape[-1]
        x = x.reshape(B, H, N * C, d)[:, :, :T]                            # drop N*C-T pad rows
        return x.transpose(1, 2).contiguous()                             # [B,T,H,d]

    def _unpad_s(x):
        x = x.reshape(B, H, N * C)[:, :, :T]
        return x.transpose(1, 2).contiguous()                             # [B,T,H]

    dq = _unpad_t(dq_p).to(torch.bfloat16)                                # narrow at store
    dk = _unpad_t(dk_in).to(torch.bfloat16)
    dv = _unpad_t(dv_in).to(torch.bfloat16)
    dg = _unpad_s(dg_full).to(torch.float32)                              # fp32 store
    dbeta_o = _unpad_s(dbeta).to(torch.float32)

    return (dq, dk, dv, dg, dbeta_o)


# ==========================================
# 输入构造（供验证使用；启用性能采集时供 profiling --factory 复用）
# ==========================================

def _make_inputs(device):
    """Construct every acceptance case.

    Chunk-domain gradient intermediates built directly at contract shapes (sig/beta_act in
    [0,1] and identical); primary grad_o is bf16 [B,T,H,V] in raw sequence layout (value
    unused). Numeric ground truth is the composition cross-check in _validate(), NOT randoms.
    """
    # (name, B, T, H, K, V)
    p0 = [
        ("aligned_b1_t128_h4_k64_v64", 1, 128, 4, 64, 64),
        ("ttail_b1_t100_h4_k64_v64", 1, 100, 4, 64, 64),
        ("t3c_b1_t192_h2_k128_v128", 1, 192, 2, 128, 128),
        ("tail_vlt_b1_t200_h3_k128_v96", 1, 200, 3, 128, 96),
    ]
    C = _C
    cases = []
    for idx, (name, B, T, H, K, V) in enumerate(p0):
        N = (T + 63) // 64
        torch.manual_seed(4000 + idx)
        grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)      # value unused
        dq_p = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
        dk_in = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
        dv_in = torch.randn((B, H, N, C, V), dtype=torch.float32, device=device)
        dg_p = torch.randn((B, H, N, C), dtype=torch.float32, device=device)
        dg2 = torch.randn((B, H, N, C), dtype=torch.float32, device=device)
        dbeta_wy = torch.randn((B, H, N, C), dtype=torch.float32, device=device)
        sig = torch.rand((B, H, N, C), dtype=torch.float32, device=device)           # [0,1]
        beta_act = sig.clone()                                                        # == sig
        args = [grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act]
        cases.append((name, args, {}))
    return cases


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
_P0_NAMES = tuple(name for name, *_ in _P0)
_OUT_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
_EXP_DTYPE = {"dq": torch.bfloat16, "dk": torch.bfloat16, "dv": torch.bfloat16,
              "dg": torch.float32, "dbeta": torch.float32}


def _expected_shapes(B, T, H, K, V):
    return {"dq": (B, T, H, K), "dk": (B, T, H, K), "dv": (B, T, H, V),
            "dg": (B, T, H), "dbeta": (B, T, H)}


def _load_cross_check_refs():
    """Import upstream CPU goldens (torch-only, CPU) + the parent full-backward staging module."""
    import importlib.util
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    custom_root = os.path.dirname(here)

    def _load(subdir, modname):
        path = os.path.join(custom_root, subdir, modname + ".py")
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    op6a = _load("gdr_bwd_wy_matmuls", "gdr_bwd_wy_matmuls_golden_cpu")
    op6b = _load("gdr_bwd_wy_reduce", "gdr_bwd_wy_reduce_golden_cpu")
    wy = _load("gdr_fwd_train_wy", "gdr_fwd_train_wy_golden_cpu")
    dqkw = _load("gdr_bwd_dqkw", "gdr_bwd_dqkw_golden_cpu")
    full = _load(os.path.join("gated_delta_rule_bwd", "modules"),
                 "gated_delta_rule_bwd_golden_stage12345678910")
    return op6a, op6b, wy, dqkw, full


def _build_op7_inputs_from_upstream(op6a, op6b, wy, dqkw, B, T, H, K, V, seed):
    """Build OP7's 9 inputs from the ACTUAL emitted outputs of every upstream golden (same primary)."""
    op5_in, prim = dqkw._build_op5_inputs(B, T, H, K, V, seed)
    dq_p, _dk_p, _dw, dg_p = dqkw.gdr_bwd_dqkw_golden_cpu(*op5_in)
    op6b_args, _prim6b = op6b._build_op6b_inputs_from_upstream(op6a, wy, dqkw, B, T, H, K, V, seed)
    dk_in, dv_in, dbeta_wy, dg2 = op6b.gdr_bwd_wy_reduce_golden_cpu(*op6b_args)
    outs_wy = wy.gdr_fwd_train_wy_golden_cpu(prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
    beta_act = outs_wy[4]
    sig = outs_wy[5]
    grad_o = prim["grad_o"]
    args = [grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act]
    return args, prim


def _rel_l2(a, b):
    return (torch.linalg.vector_norm(a - b)
            / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()


def _peak_scaled(a, b):
    max_abs = (a - b).abs().max().item()
    peak = b.abs().max().clamp_min(1e-6).item()
    return max_abs, max_abs / peak


def _mixed_ok(o, r, atol=1e-2, rtol=1e-2, rl2_tol=2e-2):
    """mixed acceptance criterion: allclose(atol,rtol) OR peak-scaled OR global rel-L2."""
    if torch.allclose(o, r, atol=atol, rtol=rtol):
        return True
    _max_abs, rel_peak = _peak_scaled(o, r)
    if rel_peak <= rtol:
        return True
    if _rel_l2(o, r) <= rl2_tol:
        return True
    return False


def _validate():
    device = _get_device()
    print("=" * 80)
    print("gdr_bwd_finalize_golden 验证报告 (OP7 = LAST op, M10 b4b_dg/b5/b6/b7 mechanical projection)")
    print("=" * 80)
    print(f"Device: {device}")
    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    # 1. Contract P0: _make_inputs cases run; case-set exact; shape / dtype / finite.
    print("\n[1. 合同 P0 用例: 集合完整性 / shape / dtype (dq/dk/dv=bf16, dg/dbeta=fp32) / finite]")
    cases = _make_inputs(device)
    observed = [c[0] for c in cases]
    _check(f"P0 case set == contract ({len(_P0_NAMES)})", observed == list(_P0_NAMES))
    for (name, B, T, H, K, V), (cname, args, kwargs) in zip(_P0, cases):
        exp = _expected_shapes(B, T, H, K, V)
        _check(f"{name}: primary grad_o is bf16", args[0].dtype == torch.bfloat16)
        outs = gdr_bwd_finalize_golden(*args, **kwargs)
        ok = len(outs) == 5
        for nm, t in zip(_OUT_NAMES, outs):
            ok = ok and (tuple(t.shape) == exp[nm] and t.dtype == _EXP_DTYPE[nm]
                         and torch.isfinite(t.float()).all().item())
        _check(f"{name}: 5 outputs / shapes {[tuple(t.shape) for t in outs]} / dtype / finite", ok)

    # 2. MANDATORY five-gradient cross-check (the LAST per-operator check).
    #    The COMPOSED full backward chain (OP5 + OP6a->OP6b + train_wy residuals -> OP7 (NPU))
    #    must reproduce the parent full-backward autograd oracle (CPU)
    #    gated_delta_rule_bwd_golden_stage12345678910 (== gated_delta_rule_bwd_golden_cpu), from
    #    the SAME primary. ALL FIVE gradients (dq,dk,dv,dg,dbeta) must agree to the mixed
    #    criterion — allclose(1e-2,1e-2) OR peak-scaled OR rel-L2<=2e-2 — across 4 shapes x seeds
    #    42/123/456. (dq/dk/dv are bf16 here -> ~1e-2 relative narrow, absorbed by the criterion.)
    #    A mismatch on ANY grad is a GM-contract / projection defect — fix the cause, never widen.
    print("\n[2. 五梯度交叉验证 (完整反向链 -> OP7 (NPU) 复现父 full-backward oracle (CPU); 混合判据)]")
    op6a, op6b, wy, dqkw, full = _load_cross_check_refs()
    worst_relpeak = {nm: 0.0 for nm in _OUT_NAMES}
    worst_rel_l2 = {nm: 0.0 for nm in _OUT_NAMES}
    for name, B, T, H, K, V in _P0:
        for seed in _SEEDS:
            args, prim = _build_op7_inputs_from_upstream(op6a, op6b, wy, dqkw, B, T, H, K, V, seed)
            outs = gdr_bwd_finalize_golden(*args)
            ref = full.gated_delta_rule_bwd_golden_stage12345678910(
                prim["grad_o"], prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
            line, ok = [], True
            for nm, o, r in zip(_OUT_NAMES, outs, ref):
                o = o.detach().cpu().float()   # bf16 dq/dk/dv widened for comparison
                r = r.float()
                if tuple(o.shape) != tuple(r.shape):
                    print(f"    shape mismatch {nm}: {tuple(o.shape)} vs {tuple(r.shape)}")
                    ok = False
                    continue
                max_abs, rel_peak = _peak_scaled(o, r)
                rl2 = _rel_l2(o, r)
                worst_relpeak[nm] = max(worst_relpeak[nm], rel_peak)
                worst_rel_l2[nm] = max(worst_rel_l2[nm], rl2)
                g_ok = _mixed_ok(o, r) and torch.isfinite(o).all().item()
                ok = ok and g_ok
                line.append(f"{nm}(rpk={rel_peak:.2e},rL2={rl2:.2e})")
            _check(f"{name} s{seed}: " + " ".join(line), ok)
    print("    worst peak-scaled : " + " ".join(f"{nm}={worst_relpeak[nm]:.3e}" for nm in _OUT_NAMES))
    print("    worst rel-L2      : " + " ".join(f"{nm}={worst_rel_l2[nm]:.3e}" for nm in _OUT_NAMES))

    # 3. b7 transpose is a PURE REINDEX for dq/dk/dv (bf16(source) reindexed; no value change
    #    beyond the store narrow). Compare against bf16-narrowed chunk-domain source.
    print("\n[3. b7 transpose = 纯重索引 + store narrow (dq/dk/dv == bf16(unpad(source)))]")
    cases = _make_inputs(device)
    for (name, B, T, H, K, V), (cname, args, kwargs) in zip(_P0, cases):
        grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act = args
        dq, dk, dv, dg, dbeta = gdr_bwd_finalize_golden(*args)
        C = _C
        N = (T + 63) // 64
        src_dq = dq_p.reshape(B, H, N * C, K)[:, :, :T].transpose(1, 2).contiguous().to(torch.bfloat16)
        src_dv = dv_in.reshape(B, H, N * C, V)[:, :, :T].transpose(1, 2).contiguous().to(torch.bfloat16)
        src_dk = dk_in.reshape(B, H, N * C, K)[:, :, :T].transpose(1, 2).contiguous().to(torch.bfloat16)
        ok = (torch.equal(dq.cpu(), src_dq.cpu())
              and torch.equal(dv.cpu(), src_dv.cpu())
              and torch.equal(dk.cpu(), src_dk.cpu()))
        _check(f"{name}: dq/dk/dv == bf16(pure-reindex(chunk source))", ok)

    # 4. Numerical edges.
    print("\n[4. 数值边界]")
    cases = _make_inputs(device)
    # all-zero gradient inputs -> all-zero public gradients (linearity).
    _, args_z, _ = cases[1]
    args_z = list(args_z)
    for j in (1, 2, 3, 4, 5, 6):   # dq_p,dk_in,dv_in,dg_p,dg2,dbeta_wy -> 0
        args_z[j] = torch.zeros_like(args_z[j])
    outs_z = gdr_bwd_finalize_golden(*args_z)
    _check("all grad inputs == 0 -> all 5 public grads == 0",
           all(t.float().abs().max().item() == 0.0 for t in outs_z))
    # grad_o VALUE unused: perturbing grad_o value leaves all 5 outputs unchanged.
    _, args_g0, _ = cases[0]
    args_g0 = list(args_g0)
    outs_ref = gdr_bwd_finalize_golden(*args_g0)
    args_g1 = list(args_g0)
    args_g1[0] = torch.randn_like(args_g1[0])
    outs_pert = gdr_bwd_finalize_golden(*args_g1)
    _check("grad_o VALUE is UNUSED (outputs invariant to grad_o value perturbation)",
           all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(outs_ref, outs_pert)))
    # beta_act UNUSED: perturbing beta_act (keeping sig) leaves all 5 outputs unchanged.
    args_b = list(args_g0)
    args_b[8] = torch.randn_like(args_b[8])
    outs_b = gdr_bwd_finalize_golden(*args_b)
    _check("beta_act value is UNUSED (outputs invariant to beta_act perturbation)",
           all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(outs_ref, outs_b)))

    # 5. Magnitude ranges (downstream precision-criterion evidence; heterogeneous grads).
    print("\n[5. 梯度量级范围 (供下游精度判据; 量级异质)]")
    for name, B, T, H, K, V in _P0:
        args, prim = _build_op7_inputs_from_upstream(op6a, op6b, wy, dqkw, B, T, H, K, V, seed=42)
        outs = gdr_bwd_finalize_golden(*args)
        mags = {nm: t.float().abs().max().item() for nm, t in zip(_OUT_NAMES, outs)}
        print(f"  {name}: " + ", ".join(f"|{nm}|max={mags[nm]:.3e}" for nm in _OUT_NAMES))

    print("\n" + "=" * 80)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 80)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
