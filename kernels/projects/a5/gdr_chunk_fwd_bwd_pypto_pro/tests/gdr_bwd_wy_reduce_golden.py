#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_wy_reduce (OP6b) golden reference — NPU (torch + torch_npu).

MECHANICAL PROJECTION (a faithful slice), NOT fresh math. This computes exactly the
PURE-VECTOR tail (b4 reductions + b4b_dk) of module M9 (golden_step b4, prepare_wy_repr_bwd)
of the READ-ONLY delivered cumulative staging golden

    gated_delta_rule_bwd_golden_stage123456789.py  (_m9)

but consumes OP6b's frozen GM/public contract directly. `_m9` computes the 4 cube adjoint
products (dkbg/dvb/dkb/dkt) via cube matmuls (that half is OP6a) then reduces them by
pure elementwise-mul + row-reduction into (dk,dv,dbeta_wy,dg2). OP6b IS that tail.

──────────────────────────────────────────────────────────────────────────────
PROJECTION MAP  (M9 `_m9` vector tail  ->  OP6b here)

  S0 (v_c self-serve, M1-style prelude, pure vector; the ONLY cast: bf16->fp32):
      v_c = reshape( pad_T( fp32( transpose(v,[B,T,H,V]->[B,H,T,V]) ) ), [B,H,N,C,V] )

  kbeta_de  = k_f32 * beta_act * decay_exp                 # on-chip recompute of k*beta*exp(decay)
  dk_term1  = dkbg * (decay_exp * beta_act)                # gate (decay_exp*beta_act) ONCE
  db1       = sum_K(dkbg * k_f32 * decay_exp)              # decay_exp gate ONCE
  dg2_loop1 = sum_K(dkbg * kbeta_de)
  dv        = dvb * beta_act                               # *** OUTPUT dv ***  (beta_act ONCE)
  db2       = sum_V(dvb * v_c)                             # *** RAW v_c *** (from primary v; NOT v_b)
  db3       = sum_K(dkb * k_f32)
  dk2       = dkb * beta_act + dkt + dk_term1              # beta_act on dkb ONCE
  dg2       = dg2_loop1 + beta_act*db3 - sum_K(k_f32*dkt)  # *** OUTPUT dg2 *** (A_kk reduced)
  dbeta_wy  = db1 + db2 + db3                              # *** OUTPUT dbeta_wy ***
  dk        = dk_p + dk2                                   # *** OUTPUT dk *** (b4b_dk cross-stage)

A_kk=k@k^T (the 9th _m9 matmul) is algebraically reduced to vector reductions of dkb/dkt
(OP6a §2 finding; verified vs _m9 L249-252):
  AdA.sum(-1) = beta_act*sum_K(dkb*k) = beta_act*db3 ;  AdA.sum(-2) = sum_K(k*dkt).
So OP6b needs NO dA and NO matmul — it is exactly pure-vector.

SCALE LEDGER (honored): scale=1/sqrt(K) NOT applied (folded upstream); pairwise decay gate
exp(g_i-g_j) NOT re-applied to dkb/dkt (baked into them by OP6a glue-2); beta_act on five
paths / decay_exp on three paths, each exactly ONCE; db3 reused by dbeta_wy and dg2.

All arithmetic fp32 (the sole cast is the bf16->fp32 widen of primary v); 4 outputs emitted
fp32 (NOT narrowed). BHN-parallel, stateless. No [:T] slice (outputs are chunk-domain).

Public signature:
    gdr_bwd_wy_reduce_golden(dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v)
      -> (dk, dv, dbeta_wy, dg2)
"""

import os
import torch
import torch.nn.functional as F

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
# Golden 参考实现（NPU torch）— M9 vector-tail mechanical projection
# ─────────────────────────────────────────────

def gdr_bwd_wy_reduce_golden(
    dkbg: torch.Tensor,
    dvb: torch.Tensor,
    dkb: torch.Tensor,
    dkt: torch.Tensor,
    k_f32: torch.Tensor,
    beta_act: torch.Tensor,
    decay_exp: torch.Tensor,
    dk_p: torch.Tensor,
    v: torch.Tensor,
):
    """NPU reference for OP6b (M9 vector tail: b4 reductions + b4b_dk).

    Args:
        dkbg      : [B, H, N, C, K]  fp32   (<- OP6a)  A^T @ dw
        dvb       : [B, H, N, C, V]  fp32   (<- OP6a)  A^T @ du
        dkb       : [B, H, N, C, K]  fp32   (<- OP6a)  dA @ k
        dkt       : [B, H, N, C, K]  fp32   (<- OP6a)  dA^T @ k_beta
        k_f32     : [B, H, N, C, K]  fp32   (<- gdr_fwd_train_wy)  raw k
        beta_act  : [B, H, N, C]     fp32   (<- gdr_fwd_train_wy)  sigmoid(beta_logit)
        decay_exp : [B, H, N, C]     fp32   (<- gdr_fwd_train_wy)  exp(cumsum(g)) in (0,1]
        dk_p      : [B, H, N, C, K]  fp32   (<- OP5 gdr_bwd_dqkw)
        v         : [B, T, H, V]     bf16   (primary public input; self-served to v_c)

    Returns:
        (dk, dv, dbeta_wy, dg2):
            dk       : [B, H, N, C, K]  fp32
            dv       : [B, H, N, C, V]  fp32
            dbeta_wy : [B, H, N, C]     fp32
            dg2      : [B, H, N, C]     fp32
    """
    device = _get_device()
    C = _C

    dkbg = dkbg.to(torch.float32).to(device)
    dvb = dvb.to(torch.float32).to(device)
    dkb = dkb.to(torch.float32).to(device)
    dkt = dkt.to(torch.float32).to(device)
    k_f32 = k_f32.to(torch.float32).to(device)
    beta_act = beta_act.to(torch.float32).to(device)
    decay_exp = decay_exp.to(torch.float32).to(device)
    dk_p = dk_p.to(torch.float32).to(device)
    v = v.to(device)

    B, H, N = dkbg.shape[0], dkbg.shape[1], dkbg.shape[2]
    V = dvb.shape[-1]

    # ---- S0: v_c self-serve (M1-style prelude; sole cast bf16 -> fp32 widen) ----
    v_f = v.to(torch.float32).transpose(1, 2).contiguous()             # [B,H,T,V]
    T = v_f.shape[-2]
    pad = (C - (T % C)) % C
    if pad > 0:
        v_f = F.pad(v_f, (0, 0, 0, pad))                               # pad token rows (inert)
    v_c = v_f.reshape(B, H, N, C, V)                                   # [B,H,N,C,V] fp32

    # ---- on-chip recompute k*beta*exp(decay) (pure vector) ----
    kbeta_de = k_f32 * beta_act[..., None] * decay_exp[..., None]

    # ---- K-loop1 reductions (from dkbg) ----
    dk_term1 = dkbg * (decay_exp * beta_act)[..., None]                # gate ONCE
    db1 = (dkbg * k_f32 * decay_exp[..., None]).sum(-1)
    dg2_loop1 = (dkbg * kbeta_de).sum(-1)

    # ---- V-loop reductions (from dvb; db2 uses RAW v_c) ----
    dv = dvb * beta_act[..., None]                                     # *** OUTPUT dv ***
    db2 = (dvb * v_c).sum(-1)

    # ---- K-loop2 reductions (from dkb / dkt; NO extra decay gate) ----
    db3 = (dkb * k_f32).sum(-1)                                        # reused
    dk2 = dkb * beta_act[..., None] + dkt + dk_term1                   # beta_act on dkb ONCE

    # ---- dg2 (A_kk reduced to dkb/dkt vector reductions) ----
    dg2 = dg2_loop1 + beta_act * db3 - (k_f32 * dkt).sum(-1)           # *** OUTPUT dg2 ***

    # ---- dbeta_wy / dk ----
    dbeta_wy = db1 + db2 + db3                                         # *** OUTPUT dbeta_wy ***
    dk = dk_p + dk2                                                    # *** OUTPUT dk ***

    return (dk.to(torch.float32), dv.to(torch.float32),
            dbeta_wy.to(torch.float32), dg2.to(torch.float32))


# ==========================================
# 输入构造（供验证使用；启用性能采集时供 profiling --factory 复用）
# ==========================================

def _make_inputs(device):
    """Construct every acceptance case.

    Chunk-domain intermediates built directly at contract shapes (decay_exp in (0,1], beta_act
    in [0,1]); primary v is bf16 [B,T,H,V] in raw sequence layout. Numeric ground truth is the
    composition cross-check in _validate() (composed OP6a->OP6b vs parent _m9), NOT these randoms.
    """
    # (name, B, H, N, K, V, T)
    p0 = [
        ("aligned_t128_h4_k64_v64", 1, 4, 2, 64, 64, 128),
        ("ttail_t100_h4_k64_v64",   1, 4, 2, 64, 64, 100),
        ("t3c_t192_h2_k128_v128",   1, 2, 3, 128, 128, 192),
        ("tail_vlt_t200_h3_k128_v96", 1, 3, 4, 128, 96, 200),
    ]
    C = _C
    cases = []
    for idx, (name, B, H, N, K, V, T) in enumerate(p0):
        torch.manual_seed(3000 + idx)
        dkbg = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
        dvb = torch.randn((B, H, N, C, V), dtype=torch.float32, device=device)
        dkb = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
        dkt = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
        k_f32 = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device).uniform_(-10.0, 10.0)
        beta_act = torch.rand((B, H, N, C), dtype=torch.float32, device=device)               # [0,1]
        g = torch.empty((B, H, N, C), dtype=torch.float32, device=device).uniform_(-4.0, 0.0)
        decay_exp = g.cumsum(-1).exp()                                                         # (0,1]
        dk_p = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
        v = torch.randn((B, T, H, V), dtype=torch.bfloat16, device=device)                    # primary
        args = [dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v]
        cases.append((name, args, {}))
    return cases


# ==========================================
# 验证
# ==========================================

# (name, B, H, N, K, V, T)  -- T chosen so ceil(T/64)==N (campaign 4-shape set).
_P0 = [
    ("aligned_t128_h4_k64_v64", 1, 4, 2, 64, 64, 128),
    ("ttail_t100_h4_k64_v64",   1, 4, 2, 64, 64, 100),
    ("t3c_t192_h2_k128_v128",   1, 2, 3, 128, 128, 192),
    ("tail_vlt_t200_h3_k128_v96", 1, 3, 4, 128, 96, 200),
]
_SEEDS = [42, 123, 456]
_P0_NAMES = tuple(name for name, *_ in _P0)
_OUT_NAMES = ("dk", "dv", "dbeta_wy", "dg2")


def _expected_shapes(B, H, N, K, V):
    C = _C
    return {"dk": (B, H, N, C, K), "dv": (B, H, N, C, V),
            "dbeta_wy": (B, H, N, C), "dg2": (B, H, N, C)}


def _load_cross_check_refs():
    """Import OP6a golden_cpu (reuse its upstream builders) + wy/dqkw/m9 modules (torch-only, CPU)."""
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
    wy = _load("gdr_fwd_train_wy", "gdr_fwd_train_wy_golden_cpu")
    dqkw = _load("gdr_bwd_dqkw", "gdr_bwd_dqkw_golden_cpu")
    m9 = _load(os.path.join("gated_delta_rule_bwd", "modules"),
               "gated_delta_rule_bwd_golden_stage123456789")
    return op6a, wy, dqkw, m9


def _build_op6b_inputs_from_upstream(op6a, wy, dqkw, B, T, H, K, V, seed):
    """Build OP6b's 9 inputs from the ACTUAL emitted outputs of every upstream golden (same primary)."""
    op6a_args, prim = op6a._build_op6a_inputs_from_upstream(wy, dqkw, B, T, H, K, V, seed)
    dkbg, dvb, dkb, dkt, _d_a = op6a.gdr_bwd_wy_matmuls_golden_cpu(*op6a_args)
    outs_wy = wy.gdr_fwd_train_wy_golden_cpu(prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
    k_f32 = outs_wy[1]; beta_act = outs_wy[4]; decay_exp = outs_wy[8]
    op5_in, _prim5 = dqkw._build_op5_inputs(B, T, H, K, V, seed)
    _dq_p, dk_p, _dw, _dg_p = dqkw.gdr_bwd_dqkw_golden_cpu(*op5_in)
    v = prim["v"]
    args = [dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v]
    return args, prim


def _rel_l2(a, b):
    return (torch.linalg.vector_norm(a - b)
            / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()


def _max_abs(a, b):
    return (a - b).abs().max().item()


def _validate():
    device = _get_device()
    print("=" * 78)
    print("gdr_bwd_wy_reduce_golden 验证报告 (OP6b, M9 vector-tail mechanical projection)")
    print("=" * 78)
    print(f"Device: {device}")
    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    # 1. Contract P0: _make_inputs cases run; shape / dtype fp32 / finite; case-set exact.
    print("\n[1. 合同 P0 用例: 集合完整性 / shape / fp32 / finite]")
    cases = _make_inputs(device)
    observed = [c[0] for c in cases]
    _check(f"P0 case set == contract ({len(_P0_NAMES)})", observed == list(_P0_NAMES))
    for (name, B, H, N, K, V, T), (cname, args, kwargs) in zip(_P0, cases):
        exp = _expected_shapes(B, H, N, K, V)
        _check(f"{name}: primary v is bf16", args[8].dtype == torch.bfloat16)
        outs = gdr_bwd_wy_reduce_golden(*args, **kwargs)
        ok = len(outs) == 4
        for nm, t in zip(_OUT_NAMES, outs):
            ok = ok and (tuple(t.shape) == exp[nm] and t.dtype == torch.float32
                         and torch.isfinite(t).all().item())
        _check(f"{name}: 4 outputs / shapes {[tuple(t.shape) for t in outs]} / fp32 / finite", ok)

    # 2. MANDATORY composition cross-check (a mismatch is a GM-CONTRACT / SCALE-LEDGER DEFECT):
    #    the COMPOSED OP6a->OP6b — OP6b (NPU) fed with OP6a's ACTUAL emitted products +
    #    gdr_fwd_train_wy residuals + OP5 dk_p + primary v — vs the parent _m9 FINAL outputs
    #    (dk,dv,dbeta_wy,dg2) on CPU, from the SAME primary. All 4 outputs must agree to fp32
    #    tolerance across 4 shapes x seeds 42/123/456. Fix the cause, never widen tolerances.
    print("\n[2. 组合交叉验证 (OP6a->OP6b (NPU) 复现父 _m9 最终 dk/dv/dbeta_wy/dg2 (CPU))]")
    op6a, wy, dqkw, m9 = _load_cross_check_refs()
    CROSS_TOL = 2e-2  # rel-L2 acceptance criterion (inherited upstream-chain drift)
    worst = {nm: 0.0 for nm in _OUT_NAMES}
    worst_abs = {nm: 0.0 for nm in _OUT_NAMES}
    for name, B, H, N, K, V, T in _P0:
        for seed in _SEEDS:
            args, prim = _build_op6b_inputs_from_upstream(op6a, wy, dqkw, B, T, H, K, V, seed)
            outs = gdr_bwd_wy_reduce_golden(*args)
            ref = m9.gated_delta_rule_bwd_golden_stage123456789(
                prim["grad_o"], prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
            errs, ok = {}, True
            for nm, o, r in zip(_OUT_NAMES, outs, ref):
                o = o.detach().cpu()
                if tuple(o.shape) != tuple(r.shape):
                    print(f"    shape mismatch {nm}: {tuple(o.shape)} vs {tuple(r.shape)}")
                    ok = False
                    continue
                e = _rel_l2(o, r)
                errs[nm] = e
                worst[nm] = max(worst[nm], e)
                worst_abs[nm] = max(worst_abs[nm], _max_abs(o, r))
                ok = ok and (e <= CROSS_TOL) and torch.isfinite(o).all().item()
            _check(f"{name} s{seed}: " + " ".join(f"{nm}={errs.get(nm, float('nan')):.2e}"
                   for nm in _OUT_NAMES) + f" (relL2 <= {CROSS_TOL:.0e})", ok)
    print("    worst rel-L2 : " + " ".join(f"{nm}={worst[nm]:.3e}" for nm in _OUT_NAMES))
    print("    worst |Δ|max : " + " ".join(f"{nm}={worst_abs[nm]:.3e}" for nm in _OUT_NAMES))

    # 3. Numerical edges.
    print("\n[3. 数值边界]")
    cases = _make_inputs(device)
    # zero OP6a products -> dk=dk_p, dv=0, dbeta_wy=0, dg2=0.
    _, args_z, _ = cases[0]
    args_z = list(args_z)
    for j in range(4):  # dkbg, dvb, dkb, dkt
        args_z[j] = torch.zeros_like(args_z[j])
    dk_z, dv_z, db_z, dg_z = gdr_bwd_wy_reduce_golden(*args_z)
    _check("all OP6a products == 0 -> dk == dk_p",
           _max_abs(dk_z.detach().cpu(), args_z[7].detach().cpu()) == 0.0)
    _check("all OP6a products == 0 -> dv == 0", dv_z.abs().max().item() == 0.0)
    _check("all OP6a products == 0 -> dbeta_wy == 0", db_z.abs().max().item() == 0.0)
    _check("all OP6a products == 0 -> dg2 == 0", dg_z.abs().max().item() == 0.0)
    # very-negative decay (decay_exp -> 0) -> finite.
    _, args_n, _ = cases[2]
    args_n = list(args_n)
    g = torch.empty((1, 2, 3, _C), dtype=torch.float32, device=device).uniform_(-16.0, -8.0)
    args_n[6] = g.cumsum(-1).exp()
    outs_n = gdr_bwd_wy_reduce_golden(*args_n)
    _check("very-negative decay: all 4 outputs finite",
           all(torch.isfinite(t).all().item() for t in outs_n))

    # 4. Magnitude ranges (downstream precision-criterion evidence).
    print("\n[4. 输出量级范围 (供下游精度判据)]")
    for name, B, H, N, K, V, T in _P0:
        args, prim = _build_op6b_inputs_from_upstream(op6a, wy, dqkw, B, T, H, K, V, seed=42)
        outs = gdr_bwd_wy_reduce_golden(*args)
        mags = {nm: t.abs().max().item() for nm, t in zip(_OUT_NAMES, outs)}
        print(f"  {name}: " + ", ".join(f"|{nm}|max={mags[nm]:.3e}" for nm in _OUT_NAMES))

    print("\n" + "=" * 78)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 78)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
