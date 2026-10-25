#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_wy_reduce (OP6b) CPU golden reference (higher precision, FP32).

Ultimate ground truth for precision comparison in later stages. Pure CPU, torch-only, no
torch_npu / no device selection; all arithmetic FP32 (the ONLY cast is the bf16->fp32
widen of the primary `v`). All four outputs FP32.

MECHANICAL PROJECTION (a faithful slice), NOT fresh math: it computes exactly the
PURE-VECTOR tail (b4 reductions + b4b_dk) of module M9 (golden_step b4, prepare_wy_repr_bwd)
of the READ-ONLY delivered cumulative staging golden

    gated_delta_rule_bwd_golden_stage123456789.py  (_m9)

but consumes OP6b's frozen GM/public contract directly. `_m9` computes the 4 cube adjoint
products (dkbg/dvb/dkb/dkt) with cube matmuls (that half is OP6a) and then reduces them by
pure elementwise-mul + row-reduction into (dk,dv,dbeta_wy,dg2). OP6b is exactly that
row-reduction tail.

──────────────────────────────────────────────────────────────────────────────
PROJECTION MAP  (M9 `_m9` vector tail  ->  OP6b here)

`_m9` cube products dkbg/dvb/dkb/dkt arrive as GM inputs (<- OP6a). k_f32/beta_act/decay_exp
are training-forward residuals (<- gdr_fwd_train_wy). dk_p (<- OP5 gdr_bwd_dqkw). PRIMARY v
[B,T,H,V] bf16 is self-served (M1-style prelude) into v_c[B,H,N,C,V] fp32 for db2.

  S0 (v_c self-serve, M1-style prelude, pure vector):
      v_c = reshape( pad_T( cast_fp32( transpose(v,[B,T,H,V]->[B,H,T,V]) ) ), [B,H,N,C,V] )
      (pad token rows are inert: they pair with upstream-padded dvb=0 -> contribute 0 to db2)

  kbeta_de  = k_f32 * beta_act * decay_exp                 # on-chip recompute of k*beta*exp(decay)
  dk_term1  = dkbg * (decay_exp * beta_act)                # gate (decay_exp*beta_act) ONCE
  db1       = sum_K(dkbg * k_f32 * decay_exp)              # decay_exp gate ONCE
  dg2_loop1 = sum_K(dkbg * kbeta_de)
  dv        = dvb * beta_act                               # *** OUTPUT dv ***  (beta_act ONCE)
  db2       = sum_V(dvb * v_c)                             # *** RAW v_c *** (from primary v; NOT v_b)
  db3       = sum_K(dkb * k_f32)
  dk2       = dkb * beta_act + dkt + dk_term1              # beta_act on dkb ONCE; dkt/dk_term1 no extra gate
  dg2       = dg2_loop1 + beta_act*db3 - sum_K(k_f32*dkt)  # *** OUTPUT dg2 *** (A_kk=k@k^T algebraically reduced)
  dbeta_wy  = db1 + db2 + db3                              # *** OUTPUT dbeta_wy ***
  dk        = dk_p + dk2                                   # *** OUTPUT dk *** (b4b_dk cross-stage accumulate)

A_kk REDUCTION (OP6a §2 frozen finding; verified vs _m9 L249-252):
  In _m9, dg2 = dg2_loop1 + AdA.sum(-1) - AdA.sum(-2) with AdA[i,j] = dA[i,j]*beta_i*(k_i.k_j).
    AdA.sum(-1)_i = beta_i * sum_j dA[i,j] (k_i.k_j) = beta_i * sum_p k[i,p] (dA@k)[i,p]
                  = beta_i * sum_K(dkb*k)_i = beta_act * db3
    AdA.sum(-2)_j = sum_i dA[i,j] beta_i (k_i.k_j) = sum_p k[j,p] (dA^T@k_beta)[j,p]
                  = sum_K(k*dkt)_j
  => dg2 needs NO dA and NO matmul; OP6b is exactly pure-vector.

SCALE LEDGER (honored here; raw for Stage-3):
  * scale=1/sqrt(K) is NOT applied (folded upstream). Not present in this golden.
  * The pairwise decay gate exp(g_i-g_j) is NOT re-applied to dkb/dkt (it lives entirely in
    OP6a glue-2, baked into dkb/dkt). This golden touches dkb/dkt only via db3, dk2, dg2 —
    no extra gate.
  * beta_act on FIVE distinct paths (dv; dk via dk_term1; dk via dkb; dg2 via kbeta_de; dg2
    via db3) — each exactly ONCE. decay_exp on THREE paths (dk_term1; db1; kbeta_de) — once each.
  * db3 = sum_K(dkb*k_f32) is REUSED by dbeta_wy and dg2 (beta_act*db3).

Signature:
    gdr_bwd_wy_reduce_golden_cpu(dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v)
      -> (dk, dv, dbeta_wy, dg2)   all fp32
    dkbg      : [B, H, N, C, K]  fp32   (<- OP6a)  A^T @ dw
    dvb       : [B, H, N, C, V]  fp32   (<- OP6a)  A^T @ du
    dkb       : [B, H, N, C, K]  fp32   (<- OP6a)  dA @ k
    dkt       : [B, H, N, C, K]  fp32   (<- OP6a)  dA^T @ k_beta
    k_f32     : [B, H, N, C, K]  fp32   (<- gdr_fwd_train_wy)  raw k
    beta_act  : [B, H, N, C]     fp32   (<- gdr_fwd_train_wy)  sigmoid(beta_logit)
    decay_exp : [B, H, N, C]     fp32   (<- gdr_fwd_train_wy)  exp(cumsum(g)) in (0,1]
    dk_p      : [B, H, N, C, K]  fp32   (<- OP5 gdr_bwd_dqkw)
    v         : [B, T, H, V]     bf16   (primary public input; self-served to v_c)
    ->  dk       : [B, H, N, C, K]  fp32
        dv       : [B, H, N, C, V]  fp32
        dbeta_wy : [B, H, N, C]     fp32
        dg2      : [B, H, N, C]     fp32

All arithmetic fp32. BHN-parallel, stateless (no loop-carried tensor, no cross-chunk dep).
No [:T] slice here (outputs are chunk-domain; the terminal slice is OP7's b7).
"""

import os

import torch
import torch.nn.functional as F

_C = 64  # fixed internal chunk size


def gdr_bwd_wy_reduce_golden_cpu(
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
    """CPU FP32 golden for OP6b (M9 vector tail: b4 reductions + b4b_dk).

    Returns (dk, dv, dbeta_wy, dg2), all fp32.
    """
    C = _C

    dkbg = dkbg.to(torch.float32)
    dvb = dvb.to(torch.float32)
    dkb = dkb.to(torch.float32)
    dkt = dkt.to(torch.float32)
    k_f32 = k_f32.to(torch.float32)
    beta_act = beta_act.to(torch.float32)
    decay_exp = decay_exp.to(torch.float32)
    dk_p = dk_p.to(torch.float32)

    B, H, N = dkbg.shape[0], dkbg.shape[1], dkbg.shape[2]
    V = dvb.shape[-1]

    # ---- S0: v_c self-serve (M1-style prelude; the ONLY cast: bf16 -> fp32 widen) ----
    # primary v [B,T,H,V] -> transpose [B,H,T,V] -> cast fp32 -> pad T->Tp=N*C -> reshape.
    v_f = v.to(torch.float32).transpose(1, 2).contiguous()             # [B,H,T,V]
    T = v_f.shape[-2]
    pad = (C - (T % C)) % C
    if pad > 0:
        v_f = F.pad(v_f, (0, 0, 0, pad))                               # pad token rows (inert; dvb=0 there)
    v_c = v_f.reshape(B, H, N, C, V)                                   # [B,H,N,C,V] fp32

    # ---- on-chip recompute k*beta*exp(decay) (pure vector) ----
    kbeta_de = k_f32 * beta_act[..., None] * decay_exp[..., None]      # [B,H,N,C,K]

    # ---- K-loop1 reductions (from dkbg) ----
    dk_term1 = dkbg * (decay_exp * beta_act)[..., None]                # gate ONCE
    db1 = (dkbg * k_f32 * decay_exp[..., None]).sum(-1)                # sum_K(dkbg*k*exp(decay))
    dg2_loop1 = (dkbg * kbeta_de).sum(-1)                             # sum_K(dkbg*kbeta_de)

    # ---- V-loop reductions (from dvb; db2 uses RAW v_c) ----
    dv = dvb * beta_act[..., None]                                     # *** OUTPUT dv ***
    db2 = (dvb * v_c).sum(-1)                                          # sum_V(dvb*v_c)

    # ---- K-loop2 reductions (from dkb / dkt; NO extra decay gate) ----
    db3 = (dkb * k_f32).sum(-1)                                        # sum_K(dkb*k) — reused
    dk2 = dkb * beta_act[..., None] + dkt + dk_term1                   # beta_act on dkb ONCE

    # ---- dg2 (A_kk reduced to dkb/dkt vector reductions) ----
    dg2 = dg2_loop1 + beta_act * db3 - (k_f32 * dkt).sum(-1)           # *** OUTPUT dg2 ***

    # ---- dbeta_wy / dk (cross-stage) ----
    dbeta_wy = db1 + db2 + db3                                         # *** OUTPUT dbeta_wy ***
    dk = dk_p + dk2                                                    # *** OUTPUT dk *** (b4b_dk)

    return (dk.to(torch.float32), dv.to(torch.float32),
            dbeta_wy.to(torch.float32), dg2.to(torch.float32))


# ==========================================
# 验证
# ==========================================

# acceptance cases in chunk domain, plus the matching primary (B,T,H,K,V) for the cross-check.
# (name, B, H, N, K, V, T)  -- T chosen so ceil(T/64)==N (campaign 4-shape set).
_P0 = [
    ("aligned_t128_h4_k64_v64", 1, 4, 2, 64, 64, 128),
    ("ttail_t100_h4_k64_v64",   1, 4, 2, 64, 64, 100),
    ("t3c_t192_h2_k128_v128",   1, 2, 3, 128, 128, 192),
    ("tail_vlt_t200_h3_k128_v96", 1, 3, 4, 128, 96, 200),
]
_SEEDS = [42, 123, 456]
_OUT_NAMES = ("dk", "dv", "dbeta_wy", "dg2")


def _expected_shapes(B, H, N, K, V):
    C = _C
    return {"dk": (B, H, N, C, K), "dv": (B, H, N, C, V),
            "dbeta_wy": (B, H, N, C), "dg2": (B, H, N, C)}


def _mk_contract_inputs(B, H, N, K, V, T, seed):
    """Build OP6b's 9 inputs directly at contract shapes/dtypes (chunk domain + primary v).

    Values are inside the declared range; decay_exp in (0,1]; beta_act in [0,1]; v bf16.
    Numeric ground truth is the composition cross-check in _validate(), NOT these randoms.
    """
    C = _C
    torch.manual_seed(seed)
    dkbg = torch.randn((B, H, N, C, K), dtype=torch.float32)
    dvb = torch.randn((B, H, N, C, V), dtype=torch.float32)
    dkb = torch.randn((B, H, N, C, K), dtype=torch.float32)
    dkt = torch.randn((B, H, N, C, K), dtype=torch.float32)
    k_f32 = torch.empty((B, H, N, C, K), dtype=torch.float32).uniform_(-10.0, 10.0)
    beta_act = torch.rand((B, H, N, C), dtype=torch.float32)                        # [0,1]
    g = torch.empty((B, H, N, C), dtype=torch.float32).uniform_(-4.0, 0.0)
    decay_exp = g.cumsum(-1).exp()                                                  # (0,1]
    dk_p = torch.randn((B, H, N, C, K), dtype=torch.float32)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)                             # primary raw layout
    return [dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v]


# ---- cross-check helpers (composed OP6a->OP6b vs parent _m9 final outputs, same primary) ----

def _load_cross_check_refs():
    """Import OP6a golden_cpu (reuse its upstream builders) + wy/dqkw/m9 modules (torch-only)."""
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
    """Build OP6b's 9 inputs from the ACTUAL emitted outputs of every upstream golden.

    dkbg,dvb,dkb,dkt <- OP6a (gdr_bwd_wy_matmuls) on inputs assembled from gdr_fwd_train_wy +
    OP5 dw + OP4 dv2 ; k_f32,beta_act,decay_exp <- gdr_fwd_train_wy ; dk_p <- OP5 gdr_bwd_dqkw ;
    v <- the shared primary. Everything on the SAME primary (seed-deterministic).
    """
    # OP6a's 8 inputs + the shared primary.
    op6a_args, prim = op6a._build_op6a_inputs_from_upstream(wy, dqkw, B, T, H, K, V, seed)
    dkbg, dvb, dkb, dkt, _d_a = op6a.gdr_bwd_wy_matmuls_golden_cpu(*op6a_args)

    # training-forward residuals from the SAME primary.
    outs_wy = wy.gdr_fwd_train_wy_golden_cpu(prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
    k_f32 = outs_wy[1]; beta_act = outs_wy[4]; decay_exp = outs_wy[8]

    # dk_p from OP5 (same primary/seed -> identical primary as op6a_args build).
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
    print("=" * 78)
    print("gdr_bwd_wy_reduce_golden_cpu 验证报告 (OP6b, M9 vector-tail projection, CPU FP32)")
    print("=" * 78)
    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    # 1. Contract P0: shape / fp32 / finite (direct chunk-domain + primary-v random inputs).
    print("\n[1. 合同 P0 用例: shape / fp32 / finite]")
    for name, B, H, N, K, V, T in _P0:
        exp = _expected_shapes(B, H, N, K, V)
        args = _mk_contract_inputs(B, H, N, K, V, T, seed=42)
        outs = gdr_bwd_wy_reduce_golden_cpu(*args)
        _check(f"{name}: returns 4 outputs", len(outs) == 4)
        for nm, t in zip(_OUT_NAMES, outs):
            ok = (tuple(t.shape) == exp[nm] and t.dtype == torch.float32
                  and torch.isfinite(t).all().item())
            _check(f"{name}.{nm}: shape {tuple(t.shape)}=={exp[nm]} / fp32 / finite", ok)

    # 2. MANDATORY composition cross-check (a mismatch is a GM-CONTRACT / SCALE-LEDGER DEFECT):
    #    the COMPOSED OP6a->OP6b — OP6b fed with OP6a's ACTUAL emitted products + gdr_fwd_train_wy
    #    residuals + OP5 dk_p + primary v — vs the parent _m9 FINAL outputs (dk,dv,dbeta_wy,dg2),
    #    from the SAME primary. All 4 outputs must agree to fp32 tolerance across 4 shapes x
    #    seeds 42/123/456. Fix the cause (double-applied gate / wrong sign / v_b-vs-v), never widen.
    print("\n[2. 组合交叉验证 (OP6a->OP6b 复现父 _m9 最终 dk/dv/dbeta_wy/dg2)]")
    op6a, wy, dqkw, m9 = _load_cross_check_refs()
    CROSS_TOL = 2e-2  # rel-L2 acceptance criterion (inherited upstream-chain drift)
    worst = {nm: 0.0 for nm in _OUT_NAMES}
    worst_abs = {nm: 0.0 for nm in _OUT_NAMES}
    for name, B, H, N, K, V, T in _P0:
        for seed in _SEEDS:
            args, prim = _build_op6b_inputs_from_upstream(op6a, wy, dqkw, B, T, H, K, V, seed)
            outs = gdr_bwd_wy_reduce_golden_cpu(*args)
            ref = m9.gated_delta_rule_bwd_golden_stage123456789(
                prim["grad_o"], prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
            errs, ok = {}, True
            for nm, o, r in zip(_OUT_NAMES, outs, ref):
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

    # 3. v_c self-serve EXACTNESS: the S0 prelude reproduces _m9's internal v_c bit-for-bit
    #    (transpose+cast+pad+reshape of the SAME primary v). Proves db2 uses the exact raw v.
    print("\n[3. v_c self-serve 精确性 (S0 prelude == _m9 内部 v_c)]")
    for name, B, H, N, K, V, T in _P0:
        _, prim = _build_op6b_inputs_from_upstream(op6a, wy, dqkw, B, T, H, K, V, seed=42)
        st = m9._prep(prim["grad_o"], prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
        v_c_ref = st["v_c"]                                            # _m9 internal v_c
        vv = prim["v"].to(torch.float32).transpose(1, 2).contiguous()
        Tp = vv.shape[-2]; pad = (_C - (Tp % _C)) % _C
        if pad > 0:
            vv = F.pad(vv, (0, 0, 0, pad))
        v_c_ours = vv.reshape(B, H, N, _C, V)
        _check(f"{name}: v_c bit-exact vs _m9 (max|Δ|={_max_abs(v_c_ours, v_c_ref):.1e})",
               torch.equal(v_c_ours, v_c_ref))

    # 4. NO double gate / NO scale double-count — direct algebraic identities of the tail.
    print("\n[4. 无重复门 / 无 scale double-count (代数恒等)]")
    args = _mk_contract_inputs(1, 2, 3, 128, 128, 192, seed=9)
    dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v = args
    dk, dv, dbeta_wy, dg2 = gdr_bwd_wy_reduce_golden_cpu(*args)
    # dv = dvb*beta_act (beta_act applied ONCE, no decay).
    _check("dv == dvb*beta_act (beta once, no decay)",
           torch.allclose(dv, dvb * beta_act[..., None], atol=0, rtol=0))
    # dk with dk_p removed == dk2 = dkb*beta_act + dkt + dkbg*decay_exp*beta_act.
    dk2_ref = dkb * beta_act[..., None] + dkt + dkbg * (decay_exp * beta_act)[..., None]
    _check("dk - dk_p == dkb*beta + dkt + dkbg*(decay_exp*beta) (each gate ONCE)",
           torch.allclose(dk - dk_p, dk2_ref, atol=1e-5, rtol=1e-5))
    # dbeta_wy == db1+db2+db3 ; db3 reused; verify component sum.
    v_f = v.to(torch.float32).transpose(1, 2).contiguous()
    Tp = v_f.shape[-2]; pad = (_C - (Tp % _C)) % _C
    if pad > 0:
        v_f = F.pad(v_f, (0, 0, 0, pad))
    v_c = v_f.reshape(1, 2, 3, _C, 128)
    db1 = (dkbg * k_f32 * decay_exp[..., None]).sum(-1)
    db2 = (dvb * v_c).sum(-1)
    db3 = (dkb * k_f32).sum(-1)
    _check("dbeta_wy == db1+db2+db3", torch.allclose(dbeta_wy, db1 + db2 + db3, atol=1e-4, rtol=1e-4))
    # dg2 == dg2_loop1 + beta*db3 - sum_K(k*dkt) (A_kk reduced; db3 reused).
    dg2_loop1 = (dkbg * (k_f32 * beta_act[..., None] * decay_exp[..., None])).sum(-1)
    dg2_ref = dg2_loop1 + beta_act * db3 - (k_f32 * dkt).sum(-1)
    _check("dg2 == dg2_loop1 + beta*db3 - sum_K(k*dkt) (A_kk reduced)",
           torch.allclose(dg2, dg2_ref, atol=1e-4, rtol=1e-4))

    # 5. Numerical edges.
    print("\n[5. 数值边界]")
    # zero OP6a products -> dk=dk_p, dv=0, dbeta_wy=0, dg2=0.
    args = _mk_contract_inputs(1, 4, 2, 64, 64, 128, seed=7)
    args[0] = torch.zeros_like(args[0])   # dkbg=0
    args[1] = torch.zeros_like(args[1])   # dvb=0
    args[2] = torch.zeros_like(args[2])   # dkb=0
    args[3] = torch.zeros_like(args[3])   # dkt=0
    dk_z, dv_z, db_z, dg_z = gdr_bwd_wy_reduce_golden_cpu(*args)
    _check("all OP6a products == 0 -> dk==dk_p", torch.allclose(dk_z, args[7], atol=0, rtol=0))
    _check("all OP6a products == 0 -> dv==0", dv_z.abs().max().item() == 0.0)
    _check("all OP6a products == 0 -> dbeta_wy==0", db_z.abs().max().item() == 0.0)
    _check("all OP6a products == 0 -> dg2==0", dg_z.abs().max().item() == 0.0)
    # bf16 primary v accepted; outputs fp32 & finite.
    args = _mk_contract_inputs(1, 3, 4, 128, 96, 200, seed=11)
    _check("v is bf16 input", args[8].dtype == torch.bfloat16)
    outs = gdr_bwd_wy_reduce_golden_cpu(*args)
    _check("bf16 v -> all outputs fp32", all(t.dtype == torch.float32 for t in outs))
    _check("bf16 v -> all outputs finite", all(torch.isfinite(t).all().item() for t in outs))
    # very-small decay_exp (deep underflow region) -> finite.
    args = _mk_contract_inputs(1, 2, 3, 128, 128, 192, seed=13)
    g = torch.empty((1, 2, 3, _C), dtype=torch.float32).uniform_(-16.0, -8.0)
    args[6] = g.cumsum(-1).exp()
    outs_n = gdr_bwd_wy_reduce_golden_cpu(*args)
    _check("very-negative decay (decay_exp->0): all outputs finite",
           all(torch.isfinite(t).all().item() for t in outs_n))

    # 6. Magnitude ranges (downstream precision-criterion evidence).
    print("\n[6. 输出量级范围 (供下游精度判据)]")
    for name, B, H, N, K, V, T in _P0:
        args, prim = _build_op6b_inputs_from_upstream(op6a, wy, dqkw, B, T, H, K, V, seed=42)
        outs = gdr_bwd_wy_reduce_golden_cpu(*args)
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
