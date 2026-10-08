"""A2 stable KDA backward unit: nine kernels, FP32 cumulative-gate cache and otherwise BF16 ABI, no host-side arithmetic.

``reference.py`` imports no compiler. ``execute`` launches the nine A2 kernels in order; the host only
allocates NaN-filled outputs and takes metadata-only ``view``s of the contiguous token-major tensors
(D-PM-35/37). Two operations the A5 unit did on the host are folded into the kernels here: the strided slice
that built ``g_last`` from ``g_cumsum`` (now read inside ``scan_fused``) and the negation of the ``inverse_mm``
``d_vh`` output (now ``d_w``, negated inside ``inverse_mm``).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from reference import (build_saved_forward, independent_reference, make_inputs as _make_inputs,
                       reference_stages as _reference_stages)

HERE = Path(__file__).resolve().parent
BLOCK_DIMS = (1, 2)
L, D = 64, 128
SAVED_NAMES = ("g_cumsum", "Aqk", "Akk", "w", "u", "qg", "kg", "v_new", "h")
KERNELS = (("scan_fused", "scan_fused_a2_fp32gc_kernel"), ("inverse_mm", "inverse_mm_a2_kernel"),
           ("inverse_epilogue", "inverse_epilogue_a2_fp32gc_kernel"), ("inverse_dainv", "inverse_dainv_a2_kernel"),
           ("inverse_dakk_fused", "inverse_dakk_fused_a2_kernel"), ("finalize_pre", "finalize_pre_a2_fp32gc_kernel"),
           ("finalize_pair", "finalize_pair_a2_kernel"), ("finalize_post", "finalize_post_a2_fp32gc_kernel"),
           ("finalize_reduce", "finalize_reduce_a2_kernel"))


def make_inputs(case):
    p = case["parameters"]
    if (p.get("L", 64), p.get("K", 128), p.get("V", 128)) != (64, 128, 128):
        raise ValueError("This unit requires L=64 and K=V=128")
    values = _make_inputs({**p, "seed": case["seed"]})
    values["saved"] = build_saved_forward(*(values[n] for n in ("q", "k", "v", "g", "beta", "initial_state")))
    return values


def validate_inputs(inputs, case=None):
    q = inputs["q"]
    if q.ndim != 4:
        raise ValueError(f"q must be [B,T,H,128]; got rank {q.ndim}")
    b, t, h, width = q.shape
    hv = inputs["v"].shape[2]
    if min(b, t, h, hv) <= 0 or t % L or width != D or hv % h:
        raise ValueError(f"Require positive T % 64 == 0, K=V=128 and HV % H == 0; "
                         f"got B={b} T={t} H={h} HV={hv} K={width}")
    c = t // L
    specs = {"q": (b, t, h, D), "k": (b, t, h, D), "v": (b, t, hv, D), "g": (b, t, hv, D), "beta": (b, t, hv),
             "initial_state": (b, hv, D, D), "do": (b, t, hv, D), "dht": (b, hv, D, D)}
    saved_specs = {name: (b, t, hv, D) for name in ("g_cumsum", "w", "u", "qg", "kg", "v_new")}
    saved_specs.update(Aqk=(b, t, hv, L), Akk=(b, t, hv, L), h=(b, c, hv, D, D))
    if not isinstance(inputs.get("saved"), dict) or set(inputs["saved"]) != set(SAVED_NAMES):
        raise ValueError(f"saved must contain exactly the nine declared forward caches {SAVED_NAMES}")
    for name, shape in {**specs, **{f"saved.{k}": v for k, v in saved_specs.items()}}.items():
        value = inputs["saved"][name.split(".", 1)[1]] if name.startswith("saved.") else inputs[name]
        dtype = torch.float32 if name == "saved.g_cumsum" else torch.bfloat16
        if (tuple(value.shape) != shape or value.dtype != dtype or value.device.type != "cpu"
                or not value.is_contiguous() or not bool(value.isfinite().all())):
            raise ValueError(f"{name} requires a finite contiguous CPU {dtype} tensor of shape {shape}; "
                             f"got {value.dtype} {tuple(value.shape)}")
    if case is not None and case.get("block_dim", 1) not in BLOCK_DIMS:
        raise ValueError(f"block_dim must be one of {BLOCK_DIMS} (the A2 values measured so far); "
                         f"got {case.get('block_dim')}")


def reference(inputs):
    validate_inputs(inputs)
    return independent_reference(inputs)


def reference_stages(inputs):
    validate_inputs(inputs)
    return _reference_stages(inputs)


def _kernels():
    out = {}
    for module, fn in KERNELS:
        name = f"_a2_kda_bwd_fp32gc_{module}"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, HERE / "kernels" / f"{module}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        out[module] = getattr(sys.modules[name], fn)
    return out


def _nan(*shape, dtype=torch.bfloat16):
    return torch.full(shape, float("nan"), dtype=dtype)


def _execute_chain(inputs, options):
    from _unit_runner import launch_kernel

    validate_inputs(inputs)
    if options["device"] != "a2" or options["backend"] != "cce" or options["block_dim"] not in BLOCK_DIMS:
        raise ValueError(f"Only a2/cce and block_dim in {BLOCK_DIMS} are declared; got "
                         f"{options['device']}/{options['backend']}/{options['block_dim']}")
    kern = _kernels()
    saved = inputs["saved"]
    b, t, h, _ = inputs["q"].shape
    hv, c, bt = inputs["v"].shape[2], t // L, b * t

    # host work: metadata-only views of the contiguous token-major tensors, and NaN-seeded allocation
    def tok(x, width):
        return x.view(bt, width)

    q2, k2 = tok(inputs["q"], h * D), tok(inputs["k"], h * D)
    v2, do2, beta2 = tok(inputs["v"], hv * D), tok(inputs["do"], hv * D), tok(inputs["beta"], hv)
    gc2 = tok(saved["g_cumsum"], hv * D)
    aqk2, akk2 = tok(saved["Aqk"], hv * L), tok(saved["Akk"], hv * L)
    kg2, qg2, w2, vnew2 = (tok(saved[n], hv * D) for n in ("kg", "qg", "w", "v_new"))

    stage = {}
    d_aqk, d_h = _nan(bt, hv * L), _nan(b, c, hv, D, D)
    d_v_scan, d_h0 = _nan(bt, hv * D), _nan(b, hv, D, D)
    d_aqk, d_h, d_v_scan, d_h0 = launch_kernel(
        kern["scan_fused"],
        (kg2, qg2, w2, gc2, do2, aqk2, vnew2, inputs["dht"], d_aqk, d_h, d_v_scan, d_h0, b, hv, c), options)
    stage.update({"scan.dAqk": d_aqk, "scan.dh": d_h, "scan.dv": d_v_scan, "scan.dh0": d_h0})

    mm_out = tuple(_nan(b, hv, c, L, D) for _ in range(5))
    d_qg, d_kg, d_w, d_v_beta, d_k_beta_g = launch_kernel(
        kern["inverse_mm"],
        (do2, vnew2, d_v_scan, saved["h"], d_h, akk2, *mm_out, b, hv, c), options)
    stage.update(zip(("inverse_mm.d_qg", "inverse_mm.d_kg", "inverse_mm.d_w", "inverse_mm.d_v_beta",
                      "inverse_mm.d_k_beta_g"), (d_qg, d_kg, d_w, d_v_beta, d_k_beta_g)))

    ep_out = (_nan(bt, hv * D), _nan(bt, hv * D), _nan(bt, hv * D), _nan(bt, hv, dtype=torch.float32),
              _nan(bt, hv * D), _nan(b, hv, c, L, D))
    dq_hv, dk_hv, dv_out, dbeta, dg_core, k_exp = launch_kernel(
        kern["inverse_epilogue"],
        (d_qg, d_kg, d_v_beta, d_k_beta_g, q2, k2, v2, gc2, beta2, saved["h"], d_h, *ep_out, b, h, hv, c), options)
    stage.update(zip(("inverse_epilogue.dq_hv", "inverse_epilogue.dk_hv", "inverse_epilogue.dv",
                      "inverse_epilogue.dbeta", "inverse_epilogue.dg_core", "inverse_epilogue.k_exp"),
                     (dq_hv, dk_hv, dv_out, dbeta, dg_core, k_exp)))

    dtri = launch_kernel(kern["inverse_dainv"],
                         (d_v_scan, v2, d_w, k_exp, beta2, _nan(b, hv, c, L, L), b, hv, c), options)
    stage["inverse_dainv.D_tri"] = dtri
    d_akk = launch_kernel(kern["inverse_dakk_fused"], (akk2, dtri, _nan(bt, hv * L), b, hv, c), options)
    stage["inverse_dakk.dAkk"] = d_akk

    pre_out = (_nan(b, hv, c, L, D), _nan(b, hv, c, L, D), _nan(b, hv, c, L, D),
               _nan(b, hv, c, L, L), _nan(b, hv, c, L, L), _nan(b, hv, c, L, L))
    q_scaled, k_scaled, kg_f, m_qk, m_base, m_beta = launch_kernel(
        kern["finalize_pre"], (q2, k2, gc2, beta2, d_aqk, d_akk, *pre_out, b, h, hv, c), options)
    stage.update(zip(("finalize_pre.q_scaled", "finalize_pre.k_scaled", "finalize_pre.kg", "finalize_pre.M_qk",
                      "finalize_pre.M_base", "finalize_pre.M_beta"),
                     (q_scaled, k_scaled, kg_f, m_qk, m_base, m_beta)))

    def flat(x):
        return x.view(b, hv, t, x.shape[-1])

    pair_out = tuple(_nan(b, hv, t, D) for _ in range(4))
    qk_left, qk_right, s_base, t_beta = launch_kernel(
        kern["finalize_pair"],
        (flat(m_qk), flat(m_base), flat(m_beta), flat(q_scaled), flat(k_scaled), flat(kg_f), *pair_out,
         b, hv, t), options)
    stage.update(zip(("finalize_pair.qk_left", "finalize_pair.qk_right", "finalize_pair.s_base",
                      "finalize_pair.t_beta"), (qk_left, qk_right, s_base, t_beta)))

    def chunked(x):
        return x.view(b, hv, c, L, x.shape[-1])

    post_out = (_nan(b, hv, c, L, D, dtype=torch.float32), _nan(b, hv, c, L, D, dtype=torch.float32),
                _nan(bt, hv), _nan(bt, hv * D))
    dq_post, dk_post, dbeta_out, dg_out = launch_kernel(
        kern["finalize_post"],
        (gc2, chunked(qk_left), chunked(qk_right), chunked(s_base), chunked(t_beta), q2, k2, beta2,
         dq_hv, dk_hv, dbeta, dg_core, *post_out, b, h, hv, c), options)
    stage.update(zip(("finalize_post.dq_hv", "finalize_post.dk_hv", "finalize_post.dbeta", "finalize_post.dg"),
                     (dq_post, dk_post, dbeta_out, dg_out)))

    dq, dk = launch_kernel(kern["finalize_reduce"],
                           (dq_post, dk_post, _nan(bt, h * D), _nan(bt, h * D), b, hv, h, c), options)
    stage.update({"finalize_reduce.dq": dq, "finalize_reduce.dk": dk})

    outputs = {"dq": dq.view(b, t, h, D), "dk": dk.view(b, t, h, D), "dv": dv_out.view(b, t, hv, D),
               "dbeta": dbeta_out.view(b, t, hv), "dg": dg_out.view(b, t, hv, D), "dh0": d_h0}
    return outputs, stage


def execute(inputs, options):
    return _execute_chain(inputs, options)[0]


def execute_stages(inputs, options):
    """The nine kernel seams, reshaped to the contract's declared stage shapes (views, no copies)."""
    validate_inputs(inputs)
    stage = _execute_chain(inputs, options)[1]
    b, t, h, _ = inputs["q"].shape
    hv = inputs["v"].shape[2]
    token = {"scan.dAqk": (hv, L), "scan.dv": (hv, D), "inverse_epilogue.dq_hv": (hv, D),
             "inverse_epilogue.dk_hv": (hv, D), "inverse_epilogue.dv": (hv, D),
             "inverse_epilogue.dg_core": (hv, D), "inverse_dakk.dAkk": (hv, L),
             "finalize_post.dg": (hv, D), "finalize_reduce.dq": (h, D), "finalize_reduce.dk": (h, D)}
    for name, (heads, width) in token.items():
        stage[name] = stage[name].view(b, t, heads, width)
    for name in ("inverse_epilogue.dbeta", "finalize_post.dbeta"):
        stage[name] = stage[name].view(b, t, hv)
    return stage
