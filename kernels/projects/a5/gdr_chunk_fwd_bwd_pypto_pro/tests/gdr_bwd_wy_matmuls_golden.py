#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_wy_matmuls (OP6a) golden reference — NPU (torch + torch_npu).

MECHANICAL PROJECTION (a faithful slice), NOT fresh math. This computes exactly the CUBE
half (the 8 matmuls b4.1..b4.8 + two vector-glue masks) of module M9 (golden_step b4,
prepare_wy_repr_bwd) of the READ-ONLY delivered cumulative staging golden

    gated_delta_rule_bwd_golden_stage123456789.py  (_m9)

but consumes OP6a's frozen GM contract directly. `_m9` computes dkbg/dvb/dkb/dkt/dA
internally and returns only M9's final reductions (dk,dv,dbeta_wy,dg2); OP6a is exactly the
cube-product slice — the five raw adjoint matmul products the vector half (OP6b) reduces.

──────────────────────────────────────────────────────────────────────────────
PROJECTION MAP  (M9 `_m9` cube half  ->  OP6a here)
  AT       = attn.transpose(-1,-2)                        # A^T (FLA loads A transposed)
  k_beta   = k_f32 * beta_act[..., None]                  # formed ON-CHIP (not a GM input)
  kbeta_t  = kbeta_de^T                                   # PRE-transposed GM input (== _m9 kbeta_de.T)

  [cube A]  b4.1  dA   = dw @ kbeta_t          # dw[C,K]@kbeta_t[K,C]->[C,C]
            b4.2  dkbg = AT @ dw               # [C,C]@[C,K]->[C,K]   *** OUTPUT ***
            b4.3  dA  += dv2 @ v_b^T           # accumulate onto dA
            b4.4  dvb  = AT @ dv2              # [C,C]@[C,V]->[C,V]   *** OUTPUT ***
  [vec glue-1]    dA   = strict_low * dA        # tril(dA, diag=-1): keep row>col
  [cube B]  b4.5  dA   = dA @ AT               # [C,C]@[C,C]->[C,C]
            b4.6  dA   = AT @ dA               # [C,C]@[C,C]->[C,C]
  [vec glue-2]    decay_fac = exp(decay_i-decay_j)*strict_low ; dA = -(dA*decay_fac)   *** OUTPUT dA ***
  [cube C]  b4.7  dkb  = dA @ k_f32            # [C,C]@[C,K]->[C,K]   *** OUTPUT ***
            b4.8  dkt  = dA^T @ k_beta         # [C,C]@[C,K]->[C,K]   *** OUTPUT ***

dkt is emitted the mathematically-natural [C,K]=[B,H,N,64,K] (fixed by the contract; the alternative
[K,C] pre-transposed storage is a separate design choice, not this contract). The 9th
`_m9` matmul A_kk=k@k^T (feeding dg2) belongs to OP6b (reduces to vector reductions of
dkb/dkt), so OP6a carries exactly 8 matmuls.

All arithmetic fp32; every torch.matmul promotes both operands to fp32; 5 outputs emitted
fp32 (NOT narrowed). BHN-parallel, stateless. No [:T] slice (residuals already padded to N*C).

Public signature:
    gdr_bwd_wy_matmuls_golden(attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2)
      -> (dkbg, dvb, dkb, dkt, d_a)
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
# Golden 参考实现（NPU torch）— M9 cube-half mechanical projection
# ─────────────────────────────────────────────

def gdr_bwd_wy_matmuls_golden(
    attn: torch.Tensor,
    k_f32: torch.Tensor,
    v_b: torch.Tensor,
    kbeta_t: torch.Tensor,
    beta_act: torch.Tensor,
    decay: torch.Tensor,
    dw: torch.Tensor,
    dv2: torch.Tensor,
):
    """NPU reference for OP6a (M9 cube half, faithful slice of the parent b4 step).

    Args:
        attn     : [B, H, N, C, C]  fp32   WY-inverse matrix A (C=64)
        k_f32    : [B, H, N, C, K]  fp32   raw k
        v_b      : [B, H, N, C, V]  fp32   v * beta_act
        kbeta_t  : [B, H, N, K, C]  fp32   kbeta_de^T (PRE-transposed)
        beta_act : [B, H, N, C]     fp32   sigmoid(beta_logit)
        decay    : [B, H, N, C]     fp32   within-chunk cumulative gate (=cumsum(g), <=0)
        dw       : [B, H, N, C, K]  fp32   (<- OP5 gdr_bwd_dqkw)
        dv2      : [B, H, N, C, V]  fp32   (<- OP4 gdr_bwd_reverse_scan)

    Returns:
        (dkbg, dvb, dkb, dkt, d_a):
            dkbg : [B, H, N, C, K]  fp32
            dvb  : [B, H, N, C, V]  fp32
            dkb  : [B, H, N, C, K]  fp32
            dkt  : [B, H, N, C, K]  fp32
            d_a  : [B, H, N, C, C]  fp32   (== dA final)
    """
    device = _get_device()
    C = _C

    attn = attn.to(torch.float32).to(device)
    k_f32 = k_f32.to(torch.float32).to(device)
    v_b = v_b.to(torch.float32).to(device)
    kbeta_t = kbeta_t.to(torch.float32).to(device)
    beta_act = beta_act.to(torch.float32).to(device)
    decay = decay.to(torch.float32).to(device)
    dw = dw.to(torch.float32).to(device)
    dv2 = dv2.to(torch.float32).to(device)

    AT = attn.transpose(-1, -2)                                # A^T (loaded transposed)
    k_beta = k_f32 * beta_act[..., None]                       # ON-CHIP: k*beta (for b4.8)

    # ---- cube A : b4.1-b4.4 ----
    dA = torch.matmul(dw.float(), kbeta_t.float())             # b4.1  dw @ kbeta_de^T -> [C,C]
    dkbg = torch.matmul(AT.float(), dw.float())                # b4.2  A^T @ dw        *** OUTPUT ***
    dA = dA + torch.matmul(dv2.float(), v_b.transpose(-1, -2).float())  # b4.3  += dv2 @ v_b^T
    dvb = torch.matmul(AT.float(), dv2.float())                # b4.4  A^T @ dv2       *** OUTPUT ***

    # ---- vec glue-1 : tril_strict(dA) ----
    strict_low = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=-1)  # i>j
    dA = dA * strict_low

    # ---- cube B (dedicated) : b4.5-b4.6 ----
    dA = torch.matmul(dA.float(), AT.float())                  # b4.5  dA @ A^T
    dA = torch.matmul(AT.float(), dA.float())                  # b4.6  A^T @ (dA@A^T)

    # ---- vec glue-2 : decay factor mask + negate ----
    gdiff = decay.unsqueeze(-1) - decay.unsqueeze(-2)          # [.,i,j] = decay_i - decay_j
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
# 输入构造（供验证使用；启用性能采集时供 profiling --factory 复用）
# ==========================================

def _make_inputs(device):
    """Construct every acceptance case.

    Inputs are built directly at contract shapes: decay = cumsum(legal log-space gate) is a
    legal (<=0, within-chunk monotone) cumulative gate; beta_act in [0,1]. Numeric ground
    truth is the cross-check in _validate() (vs the parent _m9 cube half), NOT these randoms.
    """
    # (name, B, H, N, K, V)
    p0 = [
        ("aligned_n2_k64_v64", 1, 4, 2, 64, 64),
        ("ttail_n2_k64_v64",   1, 4, 2, 64, 64),
        ("k128_n3",            1, 2, 3, 128, 128),
        ("vlt_n4_k128_v96",    1, 3, 4, 128, 96),
    ]
    C = _C
    cases = []
    for idx, (name, B, H, N, K, V) in enumerate(p0):
        torch.manual_seed(2000 + idx)
        attn = torch.randn((B, H, N, C, C), dtype=torch.float32, device=device)
        k_f32 = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
        v_b = torch.randn((B, H, N, C, V), dtype=torch.float32, device=device)
        kbeta_t = torch.randn((B, H, N, K, C), dtype=torch.float32, device=device)
        beta_act = torch.rand((B, H, N, C), dtype=torch.float32, device=device)               # [0,1]
        g = torch.empty((B, H, N, C), dtype=torch.float32, device=device).uniform_(-4.0, 0.0)
        decay = g.cumsum(-1)                                                                   # <=0
        dw = torch.randn((B, H, N, C, K), dtype=torch.float32, device=device)
        dv2 = torch.randn((B, H, N, C, V), dtype=torch.float32, device=device)
        args = [attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2]
        cases.append((name, args, {}))
    return cases


# ==========================================
# 验证
# ==========================================

# (name, B, H, N, K, V, T)  -- T chosen so ceil(T/64)==N (campaign 4-shape set).
_P0 = [
    ("aligned_n2_k64_v64", 1, 4, 2, 64, 64, 128),
    ("ttail_n2_k64_v64",   1, 4, 2, 64, 64, 100),
    ("k128_n3",            1, 2, 3, 128, 128, 192),
    ("vlt_n4_k128_v96",    1, 3, 4, 128, 96, 200),
]
_SEEDS = [42, 123, 456]
_P0_NAMES = tuple(name for name, *_ in _P0)
_OUT_NAMES = ("dkbg", "dvb", "dkb", "dkt", "d_a")


def _expected_shapes(B, H, N, K, V):
    C = _C
    return {"dkbg": (B, H, N, C, K), "dvb": (B, H, N, C, V), "dkb": (B, H, N, C, K),
            "dkt": (B, H, N, C, K), "d_a": (B, H, N, C, C)}


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
    dqkw = _load("gdr_bwd_dqkw", "gdr_bwd_dqkw_golden_cpu")
    m9 = _load(os.path.join("gated_delta_rule_bwd", "modules"),
               "gated_delta_rule_bwd_golden_stage123456789")
    return wy, dqkw, m9


def _m9_cube_ref(m9, prim):
    """Replicate `_m9`'s cube half from the SAME primary (CPU fp32) — authoritative oracle."""
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
    return (dkbg, dvb, dkb, dkt, dA)


def _build_op6a_inputs_from_upstream(wy, dqkw, B, T, H, K, V, seed):
    """Build OP6a's 8 GM inputs from ACTUAL upstream-emitted outputs (same primary, CPU fp32)."""
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
    device = _get_device()
    print("=" * 74)
    print("gdr_bwd_wy_matmuls_golden 验证报告 (OP6a, M9 cube-half mechanical projection)")
    print("=" * 74)
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
        outs = gdr_bwd_wy_matmuls_golden(*args, **kwargs)
        ok = len(outs) == 5
        for nm, t in zip(_OUT_NAMES, outs):
            ok = ok and (tuple(t.shape) == exp[nm] and t.dtype == torch.float32
                         and torch.isfinite(t).all().item())
        _check(f"{name}: 5 outputs / shapes {[tuple(t.shape) for t in outs]} / fp32 / finite", ok)

    # 2. MANDATORY cross-check (a mismatch is a GM-CONTRACT / PROJECTION DEFECT):
    #    OP6a (NPU) fed with ACTUAL upstream-emitted inputs (gdr_fwd_train_wy + OP5 dw + OP4
    #    dv2), vs the parent _m9 CUBE-HALF internal products (CPU), from the SAME primary. All
    #    5 outputs must agree to fp32 tolerance across 4 shapes x seeds 42/123/456.
    print("\n[2. 交叉验证 (OP6a 上游真实输入投影 vs 父 _m9 cube-half 内部乘积)]")
    wy, dqkw, m9 = _load_cross_check_refs()
    CROSS_TOL = 2e-2  # rel-L2 acceptance criterion (mixed grad magnitudes)
    worst = {nm: 0.0 for nm in _OUT_NAMES}
    for name, B, H, N, K, V, T in _P0:
        for seed in _SEEDS:
            args, prim = _build_op6a_inputs_from_upstream(wy, dqkw, B, T, H, K, V, seed)
            outs = gdr_bwd_wy_matmuls_golden(*args)
            ref = _m9_cube_ref(m9, prim)
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
                ok = ok and (e <= CROSS_TOL) and torch.isfinite(o).all().item()
            _check(f"{name} s{seed}: " + " ".join(f"{nm}={errs.get(nm, float('nan')):.2e}"
                   for nm in _OUT_NAMES) + f" (<= {CROSS_TOL:.0e})", ok)
    print("    worst rel-L2 across all shapes/seeds: "
          + " ".join(f"{nm}={worst[nm]:.3e}" for nm in _OUT_NAMES))

    # 3. Numerical edges.
    print("\n[3. 数值边界]")
    # zero dw & dv2 -> all 5 outputs zero (linear chain).
    cases = _make_inputs(device)
    _, args_z, _ = cases[0]
    args_z = list(args_z)
    args_z[6] = torch.zeros_like(args_z[6])   # dw = 0
    args_z[7] = torch.zeros_like(args_z[7])   # dv2 = 0
    outs_z = gdr_bwd_wy_matmuls_golden(*args_z)
    _check("dw==0 and dv2==0 -> all 5 outputs == 0",
           all(t.abs().max().item() == 0.0 for t in outs_z))
    # d_a strictly lower-triangular (glue masks) -> strict-upper incl diag == 0.
    _, args_t, _ = cases[2]
    outs_t = gdr_bwd_wy_matmuls_golden(*args_t)
    d_a = outs_t[4].detach().cpu()
    up_incl_diag = torch.triu(torch.ones(_C, _C, dtype=torch.bool), diagonal=0)
    _check("d_a strict-upper (incl diag) == 0 (glue masks strict-lower)",
           d_a.masked_select(up_incl_diag).abs().max().item() == 0.0)
    # very-negative decay -> finite (mask-before-exp).
    _, args_n, _ = cases[2]
    args_n = list(args_n)
    g = torch.empty((1, 2, 3, _C), dtype=torch.float32, device=device).uniform_(-16.0, -8.0)
    args_n[5] = g.cumsum(-1)
    outs_n = gdr_bwd_wy_matmuls_golden(*args_n)
    _check("very-negative decay: all 5 outputs finite",
           all(torch.isfinite(t).all().item() for t in outs_n))

    # 4. Magnitude ranges (downstream precision-criterion evidence).
    print("\n[4. 输出量级范围 (供下游精度判据)]")
    for name, B, H, N, K, V, T in _P0:
        args, prim = _build_op6a_inputs_from_upstream(wy, dqkw, B, T, H, K, V, seed=42)
        outs = gdr_bwd_wy_matmuls_golden(*args)
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
