#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro gdr_bwd_finalize (OP7) CPU golden reference (higher precision, FP32).

Ultimate ground truth for precision comparison in later stages. Pure CPU, torch-only,
no torch_npu / no device selection; all arithmetic FP32.

OP7 is the LAST operator of gated_delta_rule_bwd. Its five public outputs ARE the whole
backward deliverable, so the Stage-2 cross-check below verifies ALL FIVE gradients.

MECHANICAL PROJECTION (a faithful `_m10` slice), NOT fresh math: it computes exactly the
finalize+store arithmetic (golden_steps b4b_dg, b5, b6, b7 — module M10) of the READ-ONLY
delivered cumulative staging golden

    gated_delta_rule_bwd_golden_stage12345678910.py  (_m10)

but consumes OP7's frozen GM contract directly. `_m10` is the full backward and equals
gated_delta_rule_bwd_golden_cpu (fp32 autograd VJP ground truth).

──────────────────────────────────────────────────────────────────────────────
PROJECTION MAP  (M10 `_m10` (L258-285)  ->  OP7 here)

  b4b_dg  dg_sum   = dg_p + dg2                                  # [B,H,N,C]
  b5      dg_full  = triu_incl @ dg_sum                          # reverse cumsum over C
                    triu_incl[r,c] = 1 iff c >= r  (upper-tri ones INCL diag, [C,C])
                    dg_full[i][r] = sum_{c>=r} dg_sum[i][c]      # per-chunk ([C,C]); N chunks independent
  b6      dbeta    = dbeta_wy * sig * (1 - sig)                  # chain through sigmoid logit; uses sig ONLY
  b7      un-pad + transpose (+ cast at store):
              reshape [B,H,N,C,*] -> [B,H,N*C,*]; slice [:, :, :T] (drop N*C-T pad rows);
              transpose axes (1,2) -> public [B,T,H,*]  (token t = i*C + r -> [b,t,h,:]).
              dq/dk/dv cast fp32->bf16 in the public op; dg/dbeta stay fp32.

CPU-GOLDEN CONVENTION: this fp32 reference keeps dq/dk/dv in fp32 (no bf16 narrow at store)
— it is the high-precision oracle. The narrow to bf16 is a store-dtype detail exercised by
the NPU golden (gdr_bwd_finalize_golden.py) and the kernel. The b7 transpose is a PURE
REINDEX (reshape / slice / transpose only — no arithmetic, no value change).

grad_o is the primary [B,T,H,V] seed threaded ONLY as the sequence-length (T) / public-layout
anchor. OP7's arithmetic does NOT read grad_o VALUES — only grad_o.shape defines the valid T
and the [B,T,H,*] output geometry for the tail-safe un-pad+transpose store.

Signature:
    gdr_bwd_finalize_golden_cpu(grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act)
      -> (dq, dk, dv, dg, dbeta)   all fp32
    grad_o   : [B, T, H, V]        any float (VALUE UNUSED; only shape -> T / public geometry)
    dq_p     : [B, H, N, C, K] fp32  (<- OP5 gdr_bwd_dqkw)
    dk_in    : [B, H, N, C, K] fp32  (<- OP6b gdr_bwd_wy_reduce; == OP6b.dk)
    dv_in    : [B, H, N, C, V] fp32  (<- OP6b gdr_bwd_wy_reduce; == OP6b.dv)
    dg_p     : [B, H, N, C]    fp32  (<- OP5 gdr_bwd_dqkw)
    dg2      : [B, H, N, C]    fp32  (<- OP6b gdr_bwd_wy_reduce)
    dbeta_wy : [B, H, N, C]    fp32  (<- OP6b gdr_bwd_wy_reduce)
    sig      : [B, H, N, C]    fp32  (<- gdr_fwd_train_wy; sigmoid(beta_logit))
    beta_act : [B, H, N, C]    fp32  (<- gdr_fwd_train_wy; == sig; DECLARED but UNUSED, parent-fidelity)
    ->  dq  : [B, T, H, K] fp32   dk : [B, T, H, K] fp32   dv : [B, T, H, V] fp32
        dg  : [B, T, H]    fp32   dbeta : [B, T, H] fp32

All arithmetic fp32. BH per-(b,h); each (b,h) processes its N=(T+63)//64 chunks. C=64.
No scale factor in M10. b7 slices [:T] to drop tail pad rows.
"""

import os

import torch

_C = 64  # fixed internal chunk size


def gdr_bwd_finalize_golden_cpu(
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
    """CPU FP32 golden for OP7 (M10 finalize_store; faithful slice of the parent b4b_dg/b5/b6/b7).

    Returns (dq, dk, dv, dg, dbeta), all fp32. beta_act accepted per contract but UNUSED
    (== sig; kept for parent M10 fidelity). grad_o VALUE unused (only its shape -> T).
    """
    C = _C

    # Only grad_o's SHAPE is read (T / public geometry); value never enters the math.
    B, T, H, V_pub = grad_o.shape

    dq_p = dq_p.to(torch.float32)
    dk_in = dk_in.to(torch.float32)
    dv_in = dv_in.to(torch.float32)
    dg_p = dg_p.to(torch.float32)
    dg2 = dg2.to(torch.float32)
    dbeta_wy = dbeta_wy.to(torch.float32)
    sig = sig.to(torch.float32)
    _ = beta_act  # parent-fidelity contract input; == sig; not used by the core math

    Bd, Hd, N = dq_p.shape[0], dq_p.shape[1], dq_p.shape[2]
    K = dq_p.shape[-1]
    V = dv_in.shape[-1]

    # ---- b4b_dg: dg_sum = dg_p + dg2  (elementwise over the C token axis) ----
    dg_sum = dg_p + dg2                                                    # [B,H,N,C]

    # ---- b5: dg_full = triu_incl @ dg_sum  (per-chunk reverse cumsum over C) ----
    #   triu_incl[r,c] = 1 iff c >= r (upper-tri ones INCL diagonal); [C,C] constant.
    triu_incl = torch.triu(torch.ones(C, C, dtype=torch.float32), diagonal=0)
    dg_full = torch.matmul(triu_incl, dg_sum.unsqueeze(-1)).squeeze(-1)    # [B,H,N,C]

    # ---- b6: dbeta = dbeta_wy * sigma * (1 - sigma)  (chain through sigmoid logit) ----
    dbeta = dbeta_wy * sig * (1.0 - sig)                                   # [B,H,N,C]

    # ---- b7: un-pad ([:T]) + transpose [B,H,N,C,*] -> public [B,T,H,*]  (pure reindex) ----
    def _unpad_t(x):
        d = x.shape[-1]
        x = x.reshape(B, H, N * C, d)[:, :, :T]                            # drop N*C-T pad rows
        return x.transpose(1, 2).contiguous()                             # [B,T,H,d]

    def _unpad_s(x):
        x = x.reshape(B, H, N * C)[:, :, :T]
        return x.transpose(1, 2).contiguous()                             # [B,T,H]

    dq = _unpad_t(dq_p)                                                    # [B,T,H,K]
    dk = _unpad_t(dk_in)                                                   # [B,T,H,K]
    dv = _unpad_t(dv_in)                                                   # [B,T,H,V]
    dg = _unpad_s(dg_full)                                                 # [B,T,H]
    dbeta_o = _unpad_s(dbeta)                                              # [B,T,H]

    return (dq.to(torch.float32), dk.to(torch.float32), dv.to(torch.float32),
            dg.to(torch.float32), dbeta_o.to(torch.float32))


# ==========================================
# 验证
# ==========================================

# acceptance cases: (name, B, T, H, K, V).  N=(T+63)//64.
_P0 = [
    ("aligned_b1_t128_h4_k64_v64", 1, 128, 4, 64, 64),
    ("ttail_b1_t100_h4_k64_v64", 1, 100, 4, 64, 64),
    ("t3c_b1_t192_h2_k128_v128", 1, 192, 2, 128, 128),
    ("tail_vlt_b1_t200_h3_k128_v96", 1, 200, 3, 128, 96),
]
_SEEDS = [42, 123, 456]
_OUT_NAMES = ("dq", "dk", "dv", "dg", "dbeta")


def _expected_shapes(B, T, H, K, V):
    return {"dq": (B, T, H, K), "dk": (B, T, H, K), "dv": (B, T, H, V),
            "dg": (B, T, H), "dbeta": (B, T, H)}


def _mk_contract_inputs(B, T, H, K, V, seed):
    """Build OP7's 9 inputs directly at contract shapes/dtypes (chunk domain + primary grad_o).

    Values are inside the declared range; sig/beta_act in [0,1] and identical. Numeric ground
    truth is the composition cross-check in _validate(), NOT these random inputs.
    """
    C = _C
    N = (T + 63) // 64
    torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)                     # value unused
    dq_p = torch.randn((B, H, N, C, K), dtype=torch.float32)
    dk_in = torch.randn((B, H, N, C, K), dtype=torch.float32)
    dv_in = torch.randn((B, H, N, C, V), dtype=torch.float32)
    dg_p = torch.randn((B, H, N, C), dtype=torch.float32)
    dg2 = torch.randn((B, H, N, C), dtype=torch.float32)
    dbeta_wy = torch.randn((B, H, N, C), dtype=torch.float32)
    sig = torch.rand((B, H, N, C), dtype=torch.float32)                          # [0,1]
    beta_act = sig.clone()                                                       # == sig
    return [grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act]


# ---- cross-check helpers (full chain vs parent full-backward oracle, same primary) ----

def _load_cross_check_refs():
    """Import upstream CPU goldens (torch-only) + the parent full-backward staging module."""
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
    """Build OP7's 9 inputs from the ACTUAL emitted outputs of every upstream golden.

    dq_p,dg_p        <- OP5 gdr_bwd_dqkw
    dk_in,dv_in,dbeta_wy,dg2 <- OP6b gdr_bwd_wy_reduce (fed by OP6a products + residuals + OP5 dk_p)
    sig,beta_act     <- gdr_fwd_train_wy
    grad_o           <- the shared primary
    Everything on the SAME primary (seed-deterministic; every builder reseeds identically).
    """
    # OP5 -> dq_p, dg_p (and the shared primary).
    op5_in, prim = dqkw._build_op5_inputs(B, T, H, K, V, seed)
    dq_p, _dk_p, _dw, dg_p = dqkw.gdr_bwd_dqkw_golden_cpu(*op5_in)

    # OP6b -> dk_in, dv_in, dbeta_wy, dg2 (built from the full upstream chain, same primary).
    op6b_args, _prim6b = op6b._build_op6b_inputs_from_upstream(op6a, wy, dqkw, B, T, H, K, V, seed)
    dk_in, dv_in, dbeta_wy, dg2 = op6b.gdr_bwd_wy_reduce_golden_cpu(*op6b_args)

    # gdr_fwd_train_wy -> sig, beta_act (same primary).
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
    """Per-output peak-scaled error (max|Δ| / peak|b|)."""
    max_abs = (a - b).abs().max().item()
    peak = b.abs().max().clamp_min(1e-6).item()
    return max_abs, max_abs / peak


def _mixed_ok(o, r, atol=1e-2, rtol=1e-2, rl2_tol=2e-2):
    """mixed acceptance criterion: allclose(atol,rtol) OR peak-scaled OR global rel-L2."""
    if torch.allclose(o, r, atol=atol, rtol=rtol):
        return True
    max_abs, rel_peak = _peak_scaled(o, r)
    if rel_peak <= rtol:
        return True
    if _rel_l2(o, r) <= rl2_tol:
        return True
    return False


def _validate():
    print("=" * 80)
    print("gdr_bwd_finalize_golden_cpu 验证报告 (OP7 = LAST op, M10 b4b_dg/b5/b6/b7 projection, CPU FP32)")
    print("=" * 80)
    all_pass = True

    def _check(name, cond):
        nonlocal all_pass
        if not cond:
            all_pass = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        return cond

    # 1. Contract P0: shape / fp32 / finite.
    print("\n[1. 合同 P0 用例: shape / fp32 / finite]")
    for name, B, T, H, K, V in _P0:
        exp = _expected_shapes(B, T, H, K, V)
        args = _mk_contract_inputs(B, T, H, K, V, seed=42)
        outs = gdr_bwd_finalize_golden_cpu(*args)
        _check(f"{name}: returns 5 outputs", len(outs) == 5)
        for nm, t in zip(_OUT_NAMES, outs):
            ok = (tuple(t.shape) == exp[nm] and t.dtype == torch.float32
                  and torch.isfinite(t).all().item())
            _check(f"{name}.{nm}: shape {tuple(t.shape)}=={exp[nm]} / fp32 / finite", ok)

    # 2. MANDATORY five-gradient cross-check (the LAST per-operator check).
    #    The COMPOSED full backward chain (OP5 + OP6a->OP6b + train_wy residuals -> OP7)
    #    must reproduce the parent full-backward autograd oracle
    #    gated_delta_rule_bwd_golden_stage12345678910 (== gated_delta_rule_bwd_golden_cpu),
    #    from the SAME primary. ALL FIVE gradients (dq,dk,dv,dg,dbeta) must agree to the mixed
    #    criterion across 4 shapes x seeds 42/123/456. A mismatch on ANY grad is a GM-contract
    #    / projection defect — fix the cause, never widen tolerances.
    print("\n[2. 五梯度交叉验证 (完整反向链 OP5+OP6a->OP6b+train_wy -> OP7 复现父 full-backward oracle)]")
    op6a, op6b, wy, dqkw, full = _load_cross_check_refs()
    worst_relpeak = {nm: 0.0 for nm in _OUT_NAMES}
    worst_rel_l2 = {nm: 0.0 for nm in _OUT_NAMES}
    worst_abs = {nm: 0.0 for nm in _OUT_NAMES}
    for name, B, T, H, K, V in _P0:
        for seed in _SEEDS:
            args, prim = _build_op7_inputs_from_upstream(op6a, op6b, wy, dqkw, B, T, H, K, V, seed)
            outs = gdr_bwd_finalize_golden_cpu(*args)
            ref = full.gated_delta_rule_bwd_golden_stage12345678910(
                prim["grad_o"], prim["q"], prim["k"], prim["v"], prim["g"], prim["beta"])
            line, ok = [], True
            for nm, o, r in zip(_OUT_NAMES, outs, ref):
                if tuple(o.shape) != tuple(r.shape):
                    print(f"    shape mismatch {nm}: {tuple(o.shape)} vs {tuple(r.shape)}")
                    ok = False
                    continue
                max_abs, rel_peak = _peak_scaled(o, r)
                rl2 = _rel_l2(o, r)
                worst_relpeak[nm] = max(worst_relpeak[nm], rel_peak)
                worst_rel_l2[nm] = max(worst_rel_l2[nm], rl2)
                worst_abs[nm] = max(worst_abs[nm], max_abs)
                g_ok = _mixed_ok(o, r) and torch.isfinite(o).all().item()
                ok = ok and g_ok
                line.append(f"{nm}(rpk={rel_peak:.2e},rL2={rl2:.2e})")
            _check(f"{name} s{seed}: " + " ".join(line), ok)
    print("    worst peak-scaled : " + " ".join(f"{nm}={worst_relpeak[nm]:.3e}" for nm in _OUT_NAMES))
    print("    worst rel-L2      : " + " ".join(f"{nm}={worst_rel_l2[nm]:.3e}" for nm in _OUT_NAMES))
    print("    worst |Δ|max      : " + " ".join(f"{nm}={worst_abs[nm]:.3e}" for nm in _OUT_NAMES))

    # 3. b7 transpose is a PURE REINDEX (no value change) — every output value appears in the
    #    chunk-domain source; verify token t=i*C+r maps to [b,t,h,:] bit-for-bit.
    print("\n[3. b7 transpose = 纯重索引 (值不变; token t=i*C+r -> [b,t,h,:] 逐元素相等)]")
    for name, B, T, H, K, V in _P0:
        args = _mk_contract_inputs(B, T, H, K, V, seed=7)
        grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act = args
        dq, dk, dv, dg, dbeta = gdr_bwd_finalize_golden_cpu(*args)
        C = _C
        N = (T + 63) // 64
        # dq[b,t,h,:] must equal dq_p[b,h,i,r,:] with i=t//C, r=t%C, for every t<T.
        src_dq = dq_p.reshape(B, H, N * C, K)[:, :, :T].transpose(1, 2)
        src_dv = dv_in.reshape(B, H, N * C, V)[:, :, :T].transpose(1, 2)
        src_dg = dg_p  # only checks reindex for dv/dq/dk here; dg goes through b5 matmul
        ok_reindex = (torch.equal(dq, src_dq.contiguous())
                      and torch.equal(dv, src_dv.contiguous())
                      and torch.equal(dk, dk_in.reshape(B, H, N * C, K)[:, :, :T].transpose(1, 2).contiguous()))
        _check(f"{name}: dq/dk/dv are pure reindex of chunk-domain source (no value change)", ok_reindex)

    # 4. b5 reverse-cumsum identity: dg_full[r] == sum_{c>=r} dg_sum[c]  (triu_incl @ dg_sum).
    print("\n[4. b5 reverse-cumsum 恒等 (triu_incl @ dg_sum == reversed-cumsum over C)]")
    args = _mk_contract_inputs(1, 192, 2, 128, 128, seed=9)
    grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act = args
    dg_sum = dg_p + dg2
    # reference reverse cumsum via flip-cumsum-flip.
    ref_rev = torch.flip(torch.cumsum(torch.flip(dg_sum, dims=[-1]), dim=-1), dims=[-1])
    C = _C
    triu_incl = torch.triu(torch.ones(C, C, dtype=torch.float32), diagonal=0)
    ours_rev = torch.matmul(triu_incl, dg_sum.unsqueeze(-1)).squeeze(-1)
    _check("triu_incl@dg_sum == flip-cumsum-flip(dg_sum) (reverse cumsum over C)",
           torch.allclose(ours_rev, ref_rev, atol=1e-4, rtol=1e-4))

    # 5. b6 dbeta identity + sigmoid-derivative range; and beta_act is UNUSED.
    print("\n[5. b6 dbeta = dbeta_wy*sig*(1-sig); sig*(1-sig) in [0,0.25]; beta_act UNUSED]")
    args = _mk_contract_inputs(1, 128, 4, 64, 64, seed=11)
    grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act = args
    _, _, _, _, dbeta = gdr_bwd_finalize_golden_cpu(*args)
    # rebuild expected dbeta in public layout.
    dbeta_chunk = dbeta_wy * sig * (1.0 - sig)
    B, T, H = 1, 128, 4
    N = (T + 63) // 64
    dbeta_exp = dbeta_chunk.reshape(B, H, N * C)[:, :, :T].transpose(1, 2).contiguous()
    _check("dbeta == unpad(dbeta_wy*sig*(1-sig))", torch.allclose(dbeta, dbeta_exp, atol=0, rtol=0))
    sd = (sig * (1.0 - sig))
    _check("sig*(1-sig) in [0,0.25]", (sd.min().item() >= 0.0) and (sd.max().item() <= 0.25 + 1e-6))
    # beta_act unused: perturbing beta_act (keeping sig) leaves all 5 outputs unchanged.
    args_p = list(args)
    args_p[8] = torch.randn_like(args_p[8])   # garbage beta_act
    outs_a = gdr_bwd_finalize_golden_cpu(*args)
    outs_b = gdr_bwd_finalize_golden_cpu(*args_p)
    _check("beta_act value is UNUSED (outputs invariant to beta_act perturbation)",
           all(torch.equal(a, b) for a, b in zip(outs_a, outs_b)))
    # grad_o value unused: perturbing grad_o value leaves all 5 outputs unchanged.
    args_g = list(args)
    args_g[0] = torch.randn_like(args_g[0])
    outs_c = gdr_bwd_finalize_golden_cpu(*args_g)
    _check("grad_o VALUE is UNUSED (outputs invariant to grad_o value perturbation)",
           all(torch.equal(a, b) for a, b in zip(outs_a, outs_c)))

    # 6. Numerical edges.
    print("\n[6. 数值边界]")
    # all-zero gradient inputs -> all-zero public gradients (linearity).
    args_z = _mk_contract_inputs(1, 100, 4, 64, 64, seed=5)
    for j in (1, 2, 3, 4, 5, 6):  # dq_p,dk_in,dv_in,dg_p,dg2,dbeta_wy -> 0
        args_z[j] = torch.zeros_like(args_z[j])
    outs_z = gdr_bwd_finalize_golden_cpu(*args_z)
    _check("all grad inputs == 0 -> all 5 public grads == 0",
           all(t.abs().max().item() == 0.0 for t in outs_z))
    # tail chunk: dq/dk/dv pad rows are DROPPED by b7's [:T] slice (pure reindex) -> poisoning
    #   them must NOT pollute the public output.
    #   NOTE (faithful _m10): dg's b5 reverse-cumsum (triu_incl@dg_sum) reads ALL C rows of a
    #   chunk INCLUDING the tail pad rows, so dg_p/dg2/dbeta_wy pad rows are NOT ignored — they
    #   must carry the padded-pipeline values (which match the oracle, per the §2 cross-check).
    #   Hence this test poisons ONLY dq_p/dk_in/dv_in pad rows and leaves the gate inputs intact.
    args_c = _mk_contract_inputs(1, 100, 4, 64, 64, seed=6)
    outs_clean = gdr_bwd_finalize_golden_cpu(*args_c)
    args_poison = [t.clone() if torch.is_tensor(t) else t for t in args_c]
    C = _C
    N = (100 + 63) // 64   # =2
    last_valid = 100 - (N - 1) * C   # rows [last_valid:C] of last chunk are pad (tokens >= T)
    for j in (1, 2, 3):   # dq_p, dk_in, dv_in
        args_poison[j][:, :, N - 1, last_valid:, :] = 1e30
    outs_poison = gdr_bwd_finalize_golden_cpu(*args_poison)
    _check("tail pad rows in dq/dk/dv do NOT pollute public [:T] output (b7 [:T] drops them)",
           all(torch.equal(a, b) for a, b in zip(outs_clean, outs_poison)))

    # 7. Magnitude ranges (downstream precision-criterion evidence; heterogeneous grads).
    print("\n[7. 梯度量级范围 (供下游精度判据; 量级异质)]")
    for name, B, T, H, K, V in _P0:
        args, prim = _build_op7_inputs_from_upstream(op6a, op6b, wy, dqkw, B, T, H, K, V, seed=42)
        outs = gdr_bwd_finalize_golden_cpu(*args)
        mags = {nm: t.abs().max().item() for nm, t in zip(_OUT_NAMES, outs)}
        print(f"  {name}: " + ", ".join(f"|{nm}|max={mags[nm]:.3e}" for nm in _OUT_NAMES))

    print("\n" + "=" * 80)
    print("所有验证通过" if all_pass else "存在验证失败项")
    print("=" * 80)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
