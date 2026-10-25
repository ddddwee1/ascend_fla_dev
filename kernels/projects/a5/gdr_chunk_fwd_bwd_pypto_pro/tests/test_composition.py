# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end validation of the composed training forward and backward.

This is the acceptance gate for the operator as a whole, not for any one
kernel: eight correct kernels can still be wired together wrongly. It chains
the eight host entry points, materialising every boundary in GM, and checks

  (a) the five public gradients against the autograd oracle
      `gated_delta_rule_bwd_golden_cpu`, and `o` against the forward
      recurrence `_oracle_o`;
  (b) every intermediate tensor against the cumulative staging golden
      `gated_delta_rule_bwd_golden_stage12345678910.py`, whose `_m1 .. _m10`
      advance a shared dict using the same field names as the GM contract, so
      that a divergence localises to the first boundary that breaks rather
      than to "the chain is wrong".

The host only does pure-Python shape arithmetic, threads GM tensor handles
between the eight launches, and builds the CPU references. No arithmetic on
device tensors happens here outside the launches themselves.

A public output passes if either

  (A) `torch.allclose(actual_fp32, ref_fp32, rtol=1e-2, atol=1e-2)`, or
  (B) the peak-scaled criterion: `|delta| <= atol_floor + rtol*|ref|`
      element-wise with at least 99% of elements matched, and global relative
      L2 residual `<= 2e-2`. For the bfloat16-stored outputs o, dq, dk and dv,
      `rtol = 5e-2` and `atol_floor = 5e-2 * peak`; for the float32 outputs dg
      and dbeta, `1e-2` and `1e-2 * peak`.

Boundary tensors are all float32 and are judged by the same criterion with the
float32 thresholds.
"""

import importlib.util
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

_HERE = os.path.dirname(os.path.abspath(__file__))   # tests/ops/pypto_pro/gated_delta_rule
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_KERNELS_DIR = os.path.join(os.path.dirname(_HERE), "kernels")
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)


# =============================================================================
#   The eight component wrappers.  Each one lives beside its own single
#   @pl.jit kernel in ../kernels/<op>_impl.py,
#   so the delivered operators are imported as modules here rather than loaded
#   out of their test files.
# =============================================================================
from gdr_fwd_train_wy_impl import (  # noqa: E402
    gdr_fwd_train_wy_wrapper,
)
from gdr_fwd_train_scan_impl import (  # noqa: E402
    gdr_fwd_train_scan_wrapper,
)
from gdr_bwd_dv_local_impl import (  # noqa: E402
    gdr_bwd_dv_local_wrapper,
)
from gdr_bwd_reverse_scan_impl import (  # noqa: E402
    gdr_bwd_reverse_scan_wrapper,
)
from gdr_bwd_dqkw_impl import (  # noqa: E402
    gdr_bwd_dqkw_wrapper,
)
from gdr_bwd_wy_matmuls_impl import (  # noqa: E402
    gdr_bwd_wy_matmuls_wrapper,
)
from gdr_bwd_wy_reduce_impl import (  # noqa: E402
    gdr_bwd_wy_reduce_wrapper,
)
from gdr_bwd_finalize_impl import (  # noqa: E402
    gdr_bwd_finalize_wrapper,
)

# The scan component's wrapper is a single fusion module; keep the historical
# alias the composition sketch uses.
gdr_fwd_train_scan_wrapper_module1 = gdr_fwd_train_scan_wrapper


# =============================================================================
#   Loader -- isolated module namespaces for the CPU oracles, which reuse
#   module-scope names (_prep, _m1..._m10, C) across files.
# =============================================================================
def _load(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


# Oracles: autograd VJP (formal acceptance) + lean-forward recurrence (for `o`).
_golden_cpu_mod = _load("_comp_golden_cpu", os.path.join(_HERE, "gated_delta_rule_bwd_golden_cpu.py"))
gated_delta_rule_bwd_golden_cpu = _golden_cpu_mod.gated_delta_rule_bwd_golden_cpu
_oracle_o = _golden_cpu_mod._oracle_o

# Cumulative-staging golden (bisection): _prep/_m1..._m10 populate a shared `st` dict
# with names matching DECOMPOSITION_PLAN.md sec4's GM contract.
_stage_final = _load(
    "_comp_golden_stage_final",
    os.path.join(_HERE, "gated_delta_rule_bwd_golden_stage12345678910.py"),
)

from gated_delta_rule_bwd_golden import _get_device  # noqa: E402  (also calls torch.npu.set_device)


# =============================================================================
#   Host composition -- allocation + sequencing only, NO host arithmetic.
#   Mirrors DECOMPOSITION_PLAN.md sec6.  Returns the public outputs PLUS every
#   intermediate GM tensor (as a dict, for boundary bisection).
# =============================================================================
def gated_delta_rule_fwd_bwd(q, k, v, g, beta, grad_o, *, return_intermediates=False):
    # OP1: gdr_fwd_train_wy (BHN) -- recompute + WY build.
    (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
     dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t) = gdr_fwd_train_wy_wrapper(q, k, v, g, beta)

    # OP2: gdr_fwd_train_scan (BH-serial) -- state scan, emits o + s_hist + v_new.
    o, s_hist, v_new = gdr_fwd_train_scan_wrapper_module1(u, w, q_de, a_mat, k_f32, dfac, decay_exp)

    # OP3: gdr_bwd_dv_local (BHN) -- independent of OP2.
    dv_local = gdr_bwd_dv_local_wrapper(grad_o, k_f32=k_f32, q_t=q_t, g_f32=g_f32)

    # OP4: gdr_bwd_reverse_scan (BH-serial reverse) -- needs OP3 + OP1.
    dh, dv2 = gdr_bwd_reverse_scan_wrapper(
        q_t, w_t, k_f32, grad_o=grad_o, decay_exp=decay_exp, g_f32=g_f32, dv_local=dv_local)

    # OP5: gdr_bwd_dqkw (BHN) -- needs OP2 + OP4 + OP1.
    dq_p, dk_p, dw, dg_p = gdr_bwd_dqkw_wrapper(
        grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, g_f32=g_f32, decay=decay)

    # OP6a: gdr_bwd_wy_matmuls (BHN) -- needs OP5 + OP4 + OP1.
    dkbg, dvb, dkb, dkt, d_a = gdr_bwd_wy_matmuls_wrapper(
        attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2)

    # OP6b: gdr_bwd_wy_reduce (BHN) -- needs OP6a + OP5 + OP1 + primary v.
    dk, dv_, dbeta_wy, dg2 = gdr_bwd_wy_reduce_wrapper(
        dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v)

    # OP7: gdr_bwd_finalize (BHN) -- public outputs.
    dq, dk_out, dv_out, dg, dbeta = gdr_bwd_finalize_wrapper(
        grad_o, dq_p, dk, dv_, dg_p, dg2, dbeta_wy, sig)

    if not return_intermediates:
        return o, dq, dk_out, dv_out, dg, dbeta

    intermediates = dict(
        q_s=q_s, k_f32=k_f32, v_b=v_b, g_f32=g_f32, beta_act=beta_act, sig=sig, q_t=q_t,
        decay=decay, decay_exp=decay_exp, dfac=dfac, q_de=q_de, attn=attn, a_mat=a_mat,
        kbeta_t=kbeta_t, u=u, w=w, w_t=w_t,
        s_hist=s_hist, v_new=v_new,
        dv_local=dv_local,
        dh=dh, dv2=dv2,
        dq_p=dq_p, dk_p=dk_p, dw=dw, dg_p=dg_p,
        dk=dk, dv=dv_, dbeta_wy=dbeta_wy, dg2=dg2,
    )
    return (o, dq, dk_out, dv_out, dg, dbeta), intermediates


# =============================================================================
#   Timing variant -- identical wiring to `gated_delta_rule_fwd_bwd` above, but
#   brackets EACH of the 8 wrapper launches with torch.npu.synchronize() and
#   records device wall-clock per component. Used only by `--timing` mode;
#   the correctness path above never calls this. No host arithmetic beyond the
#   plain Python shape/dict bookkeeping already present in the untimed version.
# =============================================================================
def _sync():
    torch.npu.synchronize()


def _time_call(fn, *args, **kwargs):
    _sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _sync()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1e6  # microseconds


_PIPELINE_ORDER = [
    "gdr_fwd_train_wy", "gdr_fwd_train_scan", "gdr_bwd_dv_local",
    "gdr_bwd_reverse_scan", "gdr_bwd_dqkw", "gdr_bwd_wy_matmuls",
    "gdr_bwd_wy_reduce", "gdr_bwd_finalize",
]


def gated_delta_rule_fwd_bwd_timed(q, k, v, g, beta, grad_o):
    """Same 8-wrapper chain as `gated_delta_rule_fwd_bwd`, instrumented for timing.

    Returns ((o, dq, dk_out, dv_out, dg, dbeta), timings_us) where timings_us is a
    dict keyed by `_PIPELINE_ORDER` name -> microseconds for that single launch
    (device-synchronized before and after, per CLI task's timing-mode spec).
    """
    timings = {}

    (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
     dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t), timings["gdr_fwd_train_wy"] = _time_call(
        gdr_fwd_train_wy_wrapper, q, k, v, g, beta)

    (o, s_hist, v_new), timings["gdr_fwd_train_scan"] = _time_call(
        gdr_fwd_train_scan_wrapper_module1, u, w, q_de, a_mat, k_f32, dfac, decay_exp)

    dv_local, timings["gdr_bwd_dv_local"] = _time_call(
        gdr_bwd_dv_local_wrapper, grad_o, k_f32=k_f32, q_t=q_t, g_f32=g_f32)

    (dh, dv2), timings["gdr_bwd_reverse_scan"] = _time_call(
        gdr_bwd_reverse_scan_wrapper, q_t, w_t, k_f32,
        grad_o=grad_o, decay_exp=decay_exp, g_f32=g_f32, dv_local=dv_local)

    (dq_p, dk_p, dw, dg_p), timings["gdr_bwd_dqkw"] = _time_call(
        gdr_bwd_dqkw_wrapper, grad_o, v_new, k_f32, q_s, s_hist, dh, dv2,
        g_f32=g_f32, decay=decay)

    (dkbg, dvb, dkb, dkt, d_a), timings["gdr_bwd_wy_matmuls"] = _time_call(
        gdr_bwd_wy_matmuls_wrapper, attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2)

    (dk, dv_, dbeta_wy, dg2), timings["gdr_bwd_wy_reduce"] = _time_call(
        gdr_bwd_wy_reduce_wrapper, dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v)

    (dq, dk_out, dv_out, dg, dbeta), timings["gdr_bwd_finalize"] = _time_call(
        gdr_bwd_finalize_wrapper, grad_o, dq_p, dk, dv_, dg_p, dg2, dbeta_wy, sig)

    return (o, dq, dk_out, dv_out, dg, dbeta), timings


def _run_timing(shape=None, seed=2024, warmup=3, repeats=5):
    """`--timing` entry point: warm up the untimed pipeline, then time the 8
    wrapper launches `repeats` times (median reported) at `shape` (default:
    production B=1,T=2048,H=8,K=V=128). CPU oracle is never touched here."""
    if shape is None:
        shape = dict(B=1, T=2048, H=8, K=128, V=128)
    device = _get_device()
    grad_o_c, q_c, k_c, v_c, g_c, beta_c = _mk_case_cpu(seed=seed, **shape)
    grad_o_d = grad_o_c.to(device)
    q_d, k_d, v_d = q_c.to(device), k_c.to(device), v_c.to(device)
    g_d, beta_d = g_c.to(device), beta_c.to(device)

    logging.info("=" * 78)
    logging.info("TIMING MODE: shape=%s warmup=%d repeats=%d", shape, warmup, repeats)
    logging.info("=" * 78)

    for _ in range(warmup):
        gated_delta_rule_fwd_bwd(q_d, k_d, v_d, g_d, beta_d, grad_o_d)
    _sync()

    samples = [gated_delta_rule_fwd_bwd_timed(q_d, k_d, v_d, g_d, beta_d, grad_o_d)[1]
               for _ in range(repeats)]

    logging.info("%-24s %12s", "component", "median_us")
    medians = {}
    for name in _PIPELINE_ORDER:
        vals = sorted(s[name] for s in samples)
        medians[name] = vals[len(vals) // 2]
        logging.info("%-24s %12.2f", name, medians[name])
    total = sum(medians.values())
    logging.info("%-24s %12.2f", "TOTAL", total)
    logging.info("(per-sample totals: %s)",
                 ", ".join(f"{sum(s.values()):.2f}" for s in samples))
    return medians, total


# =============================================================================
#   Precision judgement -- DECOMPOSITION_PLAN.md sec9 acceptance.
# =============================================================================
def _judge(name, actual, ref, is_bf16_class):
    a = actual.detach().to(torch.float32).cpu()
    r = ref.detach().to(torch.float32).cpu()
    if a.shape != r.shape:
        return False, f"{name}: SHAPE MISMATCH actual={tuple(a.shape)} ref={tuple(r.shape)}"
    diff = (a - r).abs()
    peak = max(r.abs().max().item(), 1e-12)

    nominal_ok = torch.allclose(a, r, rtol=1e-2, atol=1e-2)

    if is_bf16_class:
        rtol, atol_coef, l2_thr = 5e-2, 5e-2, 2e-2
    else:
        rtol, atol_coef, l2_thr = 1e-2, 1e-2, 2e-2
    atol_floor = atol_coef * peak
    elem_ok = diff <= (atol_floor + rtol * r.abs())
    matched_ratio = elem_ok.float().mean().item()
    l2_denom = max(r.norm().item(), 1e-3 * peak, 1e-12)
    rel_l2 = diff.norm().item() / l2_denom
    mixed_ok = (matched_ratio >= 0.99) and (rel_l2 <= l2_thr)

    passed = bool(nominal_ok or mixed_ok)
    max_abs = diff.max().item()
    rel_denom = torch.clamp(r.abs(), min=1e-3 * peak)
    max_rel = (diff / rel_denom).max().item()
    summary = (
        f"{name}: {'PASS' if passed else 'FAIL'} "
        f"(nominal_allclose={nominal_ok}, matched_ratio={matched_ratio:.6f}, "
        f"rel_l2={rel_l2:.4e}(thr={l2_thr}), max_abs={max_abs:.4e}, max_rel={max_rel:.4e}, "
        f"peak={peak:.4e})"
    )
    return passed, summary


# GM contract name -> the golden `st` dict's field name (sec4 GM contract naming vs the
# cumulative-staging golden's internal naming; most match, these five don't).
_ALIAS_TO_GOLDEN = {"q_t": "q_T", "a_mat": "A", "kbeta_t": "kbeta_T", "w_t": "w_T", "s_hist": "S_hist"}

_BOUNDARIES = [
    ("B1 gdr_fwd_train_wy", [
        "q_s", "k_f32", "v_b", "g_f32", "beta_act", "sig", "q_t",
        "decay", "decay_exp", "dfac", "q_de", "attn", "a_mat", "kbeta_t",
        "u", "w", "w_t",
    ]),
    ("B2 gdr_fwd_train_scan", ["v_new", "s_hist"]),
    ("B3 gdr_bwd_dv_local", ["dv_local"]),
    ("B4 gdr_bwd_reverse_scan", ["dh", "dv2"]),
    ("B5 gdr_bwd_dqkw", ["dq_p", "dk_p", "dw", "dg_p"]),
    ("B6 gdr_bwd_wy_matmuls+wy_reduce", ["dk", "dv", "dbeta_wy", "dg2"]),
]


def _bisect(actual_intermediates, grad_o_cpu, q_cpu, k_cpu, v_cpu, g_cpu, beta_cpu):
    """Compare every intra-chain boundary tensor against the cumulative-staging golden.

    Returns (first_failure_boundary_or_None, [report lines]).
    """
    st = _stage_final._prep(grad_o_cpu, q_cpu, k_cpu, v_cpu, g_cpu, beta_cpu)
    _stage_final._m1(st)
    _stage_final._m2(st)
    _stage_final._m3(st)
    _stage_final._m4(st)
    _stage_final._m5(st)
    _stage_final._m6(st)
    _stage_final._m7(st)
    _stage_final._m8(st)
    _stage_final._m9(st)

    lines = []
    first_failure = None
    for boundary_name, fields in _BOUNDARIES:
        boundary_ok = True
        for f in fields:
            golden_key = _ALIAS_TO_GOLDEN.get(f, f)
            actual_t = actual_intermediates[f]
            ref_t = st[golden_key]
            ok, summary = _judge(f"{boundary_name}::{f}", actual_t, ref_t, is_bf16_class=False)
            lines.append("  " + summary)
            boundary_ok = boundary_ok and ok
        if not boundary_ok and first_failure is None:
            first_failure = boundary_name
    return first_failure, lines


# =============================================================================
#   Cases -- DESIGN sec8 / SPEC p0_cases / composition_verification shapes+seeds.
# =============================================================================
_CASES = [
    ("aligned_b1_t128_h4_k64_v64", dict(B=1, T=128, H=4, K=64, V=64), 42),
    ("ttail_b1_t100_h4_k64_v64", dict(B=1, T=100, H=4, K=64, V=64), 123),
    ("t3c_b1_t192_h2_k128_v128", dict(B=1, T=192, H=2, K=128, V=128), 456),
    ("tail_vlt_b1_t200_h3_k128_v96", dict(B=1, T=200, H=3, K=128, V=96), 42),
    # Production shape -- matches the single-group PipeUtilization baselines in
    # PERF_NEWHOST.md (B1,T2048,H8,K128,V128). Added per follow-up task: the
    # harness previously had no production-shape case and no timing.
    ("production_b1_t2048_h8_k128_v128", dict(B=1, T=2048, H=8, K=128, V=128), 2024),
]


def _mk_case_cpu(B, T, H, K, V, seed, g_lo=-4.0, g_hi=0.0):
    torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(g_lo, g_hi)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return grad_o, q, k, v, g, beta


def _run_case(name, shape, seed):
    logging.info("=" * 78)
    logging.info("COMPOSITION CASE: %s  shape=%s  seed=%d", name, shape, seed)
    logging.info("=" * 78)
    device = _get_device()

    grad_o_c, q_c, k_c, v_c, g_c, beta_c = _mk_case_cpu(seed=seed, **shape)
    grad_o_d = grad_o_c.to(device)
    q_d, k_d, v_d = q_c.to(device), k_c.to(device), v_c.to(device)
    g_d, beta_d = g_c.to(device), beta_c.to(device)

    (o, dq, dk, dv, dg, dbeta), inter = gated_delta_rule_fwd_bwd(
        q_d, k_d, v_d, g_d, beta_d, grad_o_d, return_intermediates=True)

    # ---- Formal acceptance oracle: autograd VJP + lean-forward o ----
    dq_ref, dk_ref, dv_ref, dg_ref, dbeta_ref = gated_delta_rule_bwd_golden_cpu(
        grad_o_c, q_c, k_c, v_c, g_c, beta_c)
    scale = 1.0 / (shape["K"] ** 0.5)
    o_ref_full = _oracle_o(q_c, k_c, v_c, g_c, beta_c, scale)  # [B,T,H,V] fp32
    T = shape["T"]
    o_actual_t = o[:, :T].to(torch.float32).cpu()

    results = []
    results.append(_judge("o", o_actual_t, o_ref_full, is_bf16_class=True))
    results.append(_judge("dq", dq, dq_ref, is_bf16_class=True))
    results.append(_judge("dk", dk, dk_ref, is_bf16_class=True))
    results.append(_judge("dv", dv, dv_ref, is_bf16_class=True))
    results.append(_judge("dg", dg, dg_ref, is_bf16_class=False))
    results.append(_judge("dbeta", dbeta, dbeta_ref, is_bf16_class=False))

    all_pass = True
    for ok, summary in results:
        logging.info("  %s", summary)
        all_pass = all_pass and ok

    # ---- Bisection against the cumulative-staging golden ----
    first_failure, boundary_lines = _bisect(inter, grad_o_c, q_c, k_c, v_c, g_c, beta_c)
    logging.info("-- boundary bisection (vs gated_delta_rule_bwd_golden_stage12345678910) --")
    for line in boundary_lines:
        logging.info(line)
    if first_failure is not None:
        logging.info("  FIRST FAILING BOUNDARY: %s", first_failure)
        all_pass = False
    else:
        logging.info("  all component boundaries PASS")

    if not all_pass:
        raise AssertionError(f"composition case {name} FAILED (see log above)")
    logging.info("CASE %s: PASS", name)


def test_composition_aligned():
    _run_case(*_CASES[0])


def test_composition_ttail():
    _run_case(*_CASES[1])


def test_composition_t3c_kv128():
    _run_case(*_CASES[2])


def test_composition_tail_vlt():
    _run_case(*_CASES[3])


def test_composition_production():
    _run_case(*_CASES[4])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true",
                         help="run the device-timing pipeline instead of correctness "
                              "(off by default; keeps the correctness run free of the "
                              "CPU oracle / timing overhead)")
    parser.add_argument("--warmup", type=int, default=3,
                         help="untimed warm-up launches before timing (default 3)")
    parser.add_argument("--repeats", type=int, default=5,
                         help="timed repeats; median per component is reported (default 5)")
    args = parser.parse_args()

    if args.timing:
        _run_timing(warmup=args.warmup, repeats=args.repeats)
    else:
        logging.info("gated_delta_rule_bwd -- end-to-end composition validation")
        logging.info("=" * 78)
        test_composition_aligned()
        test_composition_ttail()
        test_composition_t3c_kv128()
        test_composition_tail_vlt()
        test_composition_production()
        logging.info("=" * 78)
        logging.info("ALL COMPOSITION CASES PASS")
