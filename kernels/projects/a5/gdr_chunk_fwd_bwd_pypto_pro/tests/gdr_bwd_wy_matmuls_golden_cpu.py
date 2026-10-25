#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_wy_matmuls (OP6a) CPU golden reference (higher precision, FP32).

Ultimate ground truth for precision comparison in later stages. Pure CPU, torch-only, no
torch_npu / no device selection; all arithmetic and outputs FP32.

MECHANICAL PROJECTION (a faithful slice), NOT fresh math: it computes exactly the CUBE half
(the 8 matmuls b4.1..b4.8 + the two vector-glue masks) of module M9 (golden_step b4,
prepare_wy_repr_bwd) of the READ-ONLY delivered cumulative staging golden

    gated_delta_rule_bwd_golden_stage123456789.py  (_m9)

but consumes OP6a's frozen GM contract directly.

──────────────────────────────────────────────────────────────────────────────
PROJECTION MAP  (M9 `_m9` cube half in the parent staging golden  ->  OP6a here)

`_m9` computes dkbg/dvb/dkb/dkt/dA internally but returns only M9's FINAL reductions
(dk,dv,dbeta_wy,dg2). OP6a is exactly the cube-product slice — the five raw adjoint
matmul products the vector half (OP6b) later reduces. The projection is verbatim:

  AT          = attn.transpose(-1,-2)                      # A^T (FLA loads A transposed)
  k_beta      = k_f32 * beta_act[..., None]                # formed ON-CHIP (not a GM input)
  kbeta_t     = kbeta_de^T                                 # PRE-transposed GM input (== _m9 kbeta_de.T)

  [cube A]  b4.1  dA    = dw @ kbeta_t                     # dw[C,K] @ kbeta_t[K,C] -> [C,C]
            b4.2  dkbg  = AT @ dw                          # [C,C]@[C,K] -> [C,K]   *** OUTPUT ***
            b4.3  dA   += dv2 @ v_b^T                      # accumulate onto dA
            b4.4  dvb   = AT @ dv2                         # [C,C]@[C,V] -> [C,V]   *** OUTPUT ***
  [vec glue-1]    dA    = strict_low * dA                  # tril(dA, diag=-1): keep row>col
  [cube B]  b4.5  dA    = dA @ AT                          # [C,C]@[C,C] -> [C,C]
            b4.6  dA    = AT @ dA                          # [C,C]@[C,C] -> [C,C]
  [vec glue-2]    decay_fac = exp(decay_i-decay_j)*strict_low ; dA = -(dA*decay_fac)   *** OUTPUT dA ***
  [cube C]  b4.7  dkb   = dA @ k_f32                       # [C,C]@[C,K] -> [C,K]   *** OUTPUT ***
            b4.8  dkt   = dA^T @ k_beta                    # [C,C]@[C,K] -> [C,K]   *** OUTPUT ***

dkt is emitted the mathematically-natural [C,K]=[B,H,N,64,K] (fixed by the contract; the alternative
[K,C] pre-transposed storage is a separate design choice, not this contract).

The 9th `_m9` matmul A_kk = k@k^T (feeding dg2) does NOT belong to OP6a — it reduces
algebraically to pure-vector reductions of dkb/dkt in OP6b. So OP6a carries exactly 8
matmuls and OP6b carries no cube.

Signature:
    gdr_bwd_wy_matmuls_golden_cpu(attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2)
      -> (dkbg, dvb, dkb, dkt, d_a)
    attn     : [B, H, N, C, C]  fp32   WY-inverse matrix A  (C=64)
    k_f32    : [B, H, N, C, K]  fp32   raw k
    v_b      : [B, H, N, C, V]  fp32   v * beta_act
    kbeta_t  : [B, H, N, K, C]  fp32   kbeta_de^T = (k*beta*decay_exp)^T (PRE-transposed)
    beta_act : [B, H, N, C]     fp32   sigmoid(beta_logit)
    decay    : [B, H, N, C]     fp32   within-chunk cumulative gate (=cumsum(g), <=0)
    dw       : [B, H, N, C, K]  fp32   (<- OP5 gdr_bwd_dqkw)
    dv2      : [B, H, N, C, V]  fp32   (<- OP4 gdr_bwd_reverse_scan)
    ->  dkbg : [B, H, N, C, K]  fp32
        dvb  : [B, H, N, C, V]  fp32
        dkb  : [B, H, N, C, K]  fp32
        dkt  : [B, H, N, C, K]  fp32
        d_a  : [B, H, N, C, C]  fp32   (== dA final, the WY-adjoint matrix)

All arithmetic fp32. BHN-parallel, stateless (no loop-carried tensor, no cross-chunk dep).
No [:T] slice here (upstream residuals already padded to Tp=N*C; T-tail invisible to OP6a).
"""

import os

import torch

_C = 64  # fixed internal chunk size


def gdr_bwd_wy_matmuls_golden_cpu(
    attn: torch.Tensor,
    k_f32: torch.Tensor,
    v_b: torch.Tensor,
    kbeta_t: torch.Tensor,
    beta_act: torch.Tensor,
    decay: torch.Tensor,
    dw: torch.Tensor,
    dv2: torch.Tensor,
):
    """CPU FP32 golden for OP6a (M9 cube half, faithful slice of the parent b4 step).

    Returns (dkbg, dvb, dkb, dkt, d_a), all fp32.
    """
    C = _C

    attn = attn.to(torch.float32)
    k_f32 = k_f32.to(torch.float32)
    v_b = v_b.to(torch.float32)
    kbeta_t = kbeta_t.to(torch.float32)
    beta_act = beta_act.to(torch.float32)
    decay = decay.to(torch.float32)
    dw = dw.to(torch.float32)
    dv2 = dv2.to(torch.float32)

    AT = attn.transpose(-1, -2)                                # A^T (loaded transposed)
    k_beta = k_f32 * beta_act[..., None]                       # ON-CHIP: k*beta (for b4.8)

    # ---- cube A : b4.1-b4.4 ----
    dA = torch.matmul(dw.float(), kbeta_t.float())             # b4.1  dw @ kbeta_de^T -> [C,C]
    dkbg = torch.matmul(AT.float(), dw.float())                # b4.2  A^T @ dw        *** OUTPUT ***
    dA = dA + torch.matmul(dv2.float(), v_b.transpose(-1, -2).float())  # b4.3  += dv2 @ v_b^T
    dvb = torch.matmul(AT.float(), dv2.float())                # b4.4  A^T @ dv2       *** OUTPUT ***

    # ---- vec glue-1 : tril_strict(dA) ----
    strict_low = torch.tril(torch.ones(C, C, dtype=torch.bool, device=attn.device), diagonal=-1)  # i>j
    dA = dA * strict_low

    # ---- cube B (dedicated) : b4.5-b4.6 ----
    dA = torch.matmul(dA.float(), AT.float())                  # b4.5  dA @ A^T
    dA = torch.matmul(AT.float(), dA.float())                  # b4.6  A^T @ (dA@A^T)

    # ---- vec glue-2 : decay factor mask + negate ----
    gg = decay
    gdiff = gg.unsqueeze(-1) - gg.unsqueeze(-2)                # [.,i,j] = decay_i - decay_j
    slf = strict_low.to(torch.float32)
    decay_fac = (gdiff * slf).exp() * slf                      # mask-before-exp: upper -> 0 (no inf)
    dA = dA * decay_fac
    dA = -dA                                                   # *** OUTPUT d_a (final) ***

    # ---- cube C (dedicated) : b4.7-b4.8 ----
    dkb = torch.matmul(dA.float(), k_f32.float())              # b4.7  dA @ k          *** OUTPUT ***
    dkt = torch.matmul(dA.transpose(-1, -2).float(), k_beta.float())  # b4.8  dA^T @ k_beta   *** OUTPUT ***

    return (dkbg.to(torch.float32), dvb.to(torch.float32), dkb.to(torch.float32),
            dkt.to(torch.float32), dA.to(torch.float32))


# ==========================================
# 验证
# ==========================================

# acceptance cases in chunk domain, plus the matching primary (B,T,H,K,V) for the cross-check.
# (name, B, H, N, K, V, T)  -- T chosen so ceil(T/64)==N (campaign 4-shape set).
_P0 = [
    ("aligned_n2_k64_v64", 1, 4, 2, 64, 64, 128),
    ("ttail_n2_k64_v64",   1, 4, 2, 64, 64, 100),
    ("k128_n3",            1, 2, 3, 128, 128, 192),
    ("vlt_n4_k128_v96",    1, 3, 4, 128, 96, 200),
]
_SEEDS = [42, 123, 456]
_OUT_NAMES = ("dkbg", "dvb", "dkb", "dkt", "d_a")


def _expected_shapes(B, H, N, K, V):
    C = _C
    return {"dkbg": (B, H, N, C, K), "dvb": (B, H, N, C, V), "dkb": (B, H, N, C, K),
            "dkt": (B, H, N, C, K), "d_a": (B, H, N, C, C)}


def _mk_contract_inputs(B, H, N, K, V, seed):
    """Build OP6a's 8 GM inputs directly at contract shapes/dtypes (chunk domain).

    Values are inside the declared range; decay <= 0 monotone-non-increasing within a chunk
    (a legal cumulative gate). Numeric ground truth is the cross-check in _validate(), NOT
    these random inputs.
    """
    C = _C
    torch.manual_seed(seed)
    attn = torch.randn((B, H, N, C, C), dtype=torch.float32)
    k_f32 = torch.randn((B, H, N, C, K), dtype=torch.float32)
    v_b = torch.randn((B, H, N, C, V), dtype=torch.float32)
    kbeta_t = torch.randn((B, H, N, K, C), dtype=torch.float32)
    beta_act = torch.rand((B, H, N, C), dtype=torch.float32)                     # [0,1]
    g = torch.empty((B, H, N, C), dtype=torch.float32).uniform_(-4.0, 0.0)       # log-space gate
    decay = g.cumsum(-1)                                                          # <=0 monotone
    dw = torch.randn((B, H, N, C, K), dtype=torch.float32)
    dv2 = torch.randn((B, H, N, C, V), dtype=torch.float32)
    return [attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2]


# ---- cross-check helpers (parent _m9 cube-half reference from the SAME primary) ----

def _load_cross_check_refs():
    """Import upstream CPU goldens (torch-only) and the parent M9 staging module."""
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

    wy = _load("gdr_fwd_train_wy", "gdr_fwd_train_wy_golden_cpu")
    dqkw = _load("gdr_bwd_dqkw", "gdr_bwd_dqkw_golden_cpu")   # provides _build_op5_inputs (full chain)
    m9 = _load(os.path.join("gated_delta_rule_bwd", "modules"),
               "gated_delta_rule_bwd_golden_stage123456789")
    return wy, dqkw, m9


def _m9_cube_ref(m9, prim):
    """Replicate `_m9`'s cube half (b4.1..b4.8 + glue masks) from the SAME primary.

    Builds the parent state st via _prep + _m1.._m8, then computes the FIVE internal cube
    products verbatim from `_m9`. This is the authoritative oracle for OP6a's 5 outputs.
    """
    st = m9._prep(prim["grad_o"], prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
    m9._m1(st); m9._m2(st); m9._m3(st); m9._m4(st)
    m9._m5(st); m9._m6(st); m9._m7(st); m9._m8(st)
    C = st["C"]
    attn, k_f32, v_b, beta_act = st["attn"], st["k_f32"], st["v_b"], st["beta_act"]
    decay, kbeta_de, k_beta = st["decay"], st["kbeta_de"], st["k_beta"]
    dw, dv2 = st["dw"], st["dv2"]
    AT = attn.transpose(-1, -2)
    dA = torch.matmul(dw, kbeta_de.transpose(-1, -2))          # b4.1
    dkbg = torch.matmul(AT, dw)                                # b4.2
    dA = dA + torch.matmul(dv2, v_b.transpose(-1, -2))         # b4.3
    dvb = torch.matmul(AT, dv2)                                # b4.4
    strict_low = torch.tril(torch.ones(C, C, dtype=torch.bool), diagonal=-1)
    dA = dA * strict_low                                       # glue-1
    dA = torch.matmul(dA, AT)                                  # b4.5
    dA = torch.matmul(AT, dA)                                  # b4.6
    gdiff = decay.unsqueeze(-1) - decay.unsqueeze(-2)
    slf = strict_low.to(torch.float32)
    decay_fac = (gdiff * slf).exp() * slf
    dA = dA * decay_fac
    dA = -dA                                                   # glue-2
    dkb = torch.matmul(dA, k_f32)                              # b4.7
    dkt = torch.matmul(dA.transpose(-1, -2), k_beta)          # b4.8
    # OP6a inputs extracted from the SAME st (for the exact-projection check).
    op6a_in = [st["attn"], st["k_f32"], st["v_b"], st["kbeta_T"], st["beta_act"],
               st["decay"], st["dw"], st["dv2"]]
    return (dkbg, dvb, dkb, dkt, dA), op6a_in


def _build_op6a_inputs_from_upstream(wy, dqkw, B, T, H, K, V, seed):
    """Build OP6a's 8 GM inputs from the ACTUAL emitted outputs of the upstream goldens.

    attn,k_f32,v_b,kbeta_t,beta_act,decay <- gdr_fwd_train_wy ; dw <- OP5 gdr_bwd_dqkw ;
    dv2 <- OP4 gdr_bwd_reverse_scan (assembled inside dqkw._build_op5_inputs). Same primary.
    """
    op5_in, prim = dqkw._build_op5_inputs(B, T, H, K, V, seed)
    dv2 = op5_in[6]                                            # OP4 emitted dv2
    _, _, dw, _ = dqkw.gdr_bwd_dqkw_golden_cpu(*op5_in)        # OP5 emitted dw
    outs_wy = wy.gdr_fwd_train_wy_golden_cpu(
        prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
    attn = outs_wy[11]; k_f32 = outs_wy[1]; v_b = outs_wy[2]
    kbeta_t = outs_wy[13]; beta_act = outs_wy[4]; decay = outs_wy[7]
    args = [attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2]
    return args, prim


def _rel_l2(a, b):
    return (torch.linalg.vector_norm(a - b)
            / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()


def _validate():
    print("=" * 74)
    print("gdr_bwd_wy_matmuls_golden_cpu 验证报告 (OP6a, M9 cube-half projection, CPU FP32)")
    print("=" * 74)
    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    # 1. Contract P0: shape / fp32 / finite (direct chunk-domain random inputs).
    print("\n[1. 合同 P0 用例: shape / fp32 / finite]")
    for name, B, H, N, K, V, T in _P0:
        exp = _expected_shapes(B, H, N, K, V)
        args = _mk_contract_inputs(B, H, N, K, V, seed=42)
        outs = gdr_bwd_wy_matmuls_golden_cpu(*args)
        _check(f"{name}: returns 5 outputs", len(outs) == 5)
        for nm, t in zip(_OUT_NAMES, outs):
            ok = (tuple(t.shape) == exp[nm] and t.dtype == torch.float32
                  and torch.isfinite(t).all().item())
            _check(f"{name}.{nm}: shape {tuple(t.shape)}=={exp[nm]} / fp32 / finite", ok)

    # 2. MANDATORY cross-check (a mismatch is a GM-CONTRACT / PROJECTION DEFECT):
    #    OP6a fed with ACTUAL upstream-emitted inputs (gdr_fwd_train_wy + OP5 dw + OP4 dv2),
    #    vs the parent _m9 CUBE-HALF internal products, from the SAME primary. All 5 outputs
    #    must agree to fp32 tolerance across 4 shapes x seeds 42/123/456.
    print("\n[2. 交叉验证 (OP6a 上游真实输入投影 vs 父 _m9 cube-half 内部乘积)]")
    wy, dqkw, m9 = _load_cross_check_refs()
    CROSS_TOL = 2e-2  # rel-L2 acceptance criterion (mixed grad magnitudes)
    worst = {nm: 0.0 for nm in _OUT_NAMES}
    for name, B, H, N, K, V, T in _P0:
        for seed in _SEEDS:
            args, prim = _build_op6a_inputs_from_upstream(wy, dqkw, B, T, H, K, V, seed)
            outs = gdr_bwd_wy_matmuls_golden_cpu(*args)
            (ref, _op6a_in) = _m9_cube_ref(m9, prim)
            errs, ok = {}, True
            for nm, o, r in zip(_OUT_NAMES, outs, ref):
                if tuple(o.shape) != tuple(r.shape):
                    print(f"    shape mismatch {nm}: {tuple(o.shape)} vs {tuple(r.shape)}")
                    ok = False
                    continue
                e = _rel_l2(o, r)
                errs[nm] = e
                worst[nm] = max(worst[nm], e)
                ok = ok and (e <= CROSS_TOL) and torch.isfinite(o).all().item()
            _check(f"{name} s{seed}: " + " ".join(f"{nm}={errs.get(nm, float('nan')):.2e}"
                   for nm in _OUT_NAMES) + f" (<= {CROSS_TOL:.0e})", ok)
    print("    worst rel-L2 across all shapes/seeds: "
          + " ".join(f"{nm}={worst[nm]:.3e}" for nm in _OUT_NAMES))

    # 3. EXACT projection check: OP6a fed with the parent's OWN st intermediates must
    #    reproduce the _m9 cube half to near machine precision (pure refactor, no chain drift).
    print("\n[3. 精确投影核对 (OP6a(st 内部量) vs _m9 cube-half, 近机器精度)]")
    EXACT_TOL = 1e-4
    worst_exact = {nm: 0.0 for nm in _OUT_NAMES}
    for name, B, H, N, K, V, T in _P0:
        for seed in _SEEDS:
            _, prim = _build_op6a_inputs_from_upstream(wy, dqkw, B, T, H, K, V, seed)
            (ref, op6a_in) = _m9_cube_ref(m9, prim)
            outs = gdr_bwd_wy_matmuls_golden_cpu(*op6a_in)
            ok = True
            for nm, o, r in zip(_OUT_NAMES, outs, ref):
                e = _rel_l2(o, r)
                worst_exact[nm] = max(worst_exact[nm], e)
                ok = ok and (e <= EXACT_TOL)
            _check(f"{name} s{seed}: exact-projection rel-L2 <= {EXACT_TOL:.0e}", ok)
    print("    worst exact-projection rel-L2: "
          + " ".join(f"{nm}={worst_exact[nm]:.3e}" for nm in _OUT_NAMES))

    # 4. Numerical edges.
    print("\n[4. 数值边界]")
    # zero dw & dv2 -> cube-A dA/dkbg/dvb all zero -> linear chain -> all 5 outputs zero.
    args = _mk_contract_inputs(1, 4, 2, 64, 64, seed=7)
    args[6] = torch.zeros_like(args[6])   # dw = 0
    args[7] = torch.zeros_like(args[7])   # dv2 = 0
    outs_z = gdr_bwd_wy_matmuls_golden_cpu(*args)
    _check("dw==0 and dv2==0 -> all 5 outputs == 0",
           all(t.abs().max().item() == 0.0 for t in outs_z))
    # d_a is strictly lower-triangular (glue masks) -> strict-upper incl diag == 0.
    args = _mk_contract_inputs(1, 4, 2, 64, 64, seed=9)
    outs = gdr_bwd_wy_matmuls_golden_cpu(*args)
    d_a = outs[4]
    up_incl_diag = torch.triu(torch.ones(_C, _C, dtype=torch.bool), diagonal=0)
    _check("d_a strict-upper (incl diag) == 0 (glue masks strict-lower)",
           d_a.masked_select(up_incl_diag).abs().max().item() == 0.0)
    # very-negative decay: decay_fac in (0,1], finite (mask-before-exp, no inf).
    args = _mk_contract_inputs(1, 2, 3, 128, 128, seed=11)
    g = torch.empty((1, 2, 3, _C), dtype=torch.float32).uniform_(-16.0, -8.0)
    args[5] = g.cumsum(-1)
    outs_n = gdr_bwd_wy_matmuls_golden_cpu(*args)
    _check("very-negative decay: all 5 outputs finite",
           all(torch.isfinite(t).all().item() for t in outs_n))

    # 5. Magnitude ranges (downstream precision-criterion evidence).
    print("\n[5. 输出量级范围 (供下游精度判据)]")
    for name, B, H, N, K, V, T in _P0:
        args, prim = _build_op6a_inputs_from_upstream(wy, dqkw, B, T, H, K, V, seed=42)
        outs = gdr_bwd_wy_matmuls_golden_cpu(*args)
        mags = {nm: t.abs().max().item() for nm, t in zip(_OUT_NAMES, outs)}
        print(f"  {name}: " + ", ".join(f"|{nm}|max={mags[nm]:.3e}" for nm in _OUT_NAMES))

    print("\n" + "=" * 74)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 74)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
