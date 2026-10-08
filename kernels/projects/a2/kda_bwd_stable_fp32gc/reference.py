"""Self-contained CPU oracle for the A2 KDA backward unit.

Imports nothing from a compiler. Three things live here:

* ``make_inputs`` -- the A5 backward unit's input distribution and seed order, so cases are comparable;
* ``build_saved_forward`` -- the nine forward checkpoints the backward consumes, computed in FP32 and stored
  BF16 except g_cumsum, retained FP32 under D-PM-60 (``g_cumsum`` is the chunk-local cumulative gate in
  **log2** space: the natural-log cumsum divided by ln2);
* ``independent_reference`` -- the gradients, by autograd through an FP32 chunked forward written from the KDA
  definition. It is independent of how the nine kernels split the work.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

RCP_LN2 = 1.0 / math.log(2.0)
IO_DTYPE = torch.bfloat16
INPUT_SCALE = 0.01
INPUT_K_SCALE = 0.05
CHUNK = 64


def _exp2(x):
    return torch.exp2(x)


def _bounds(total, chunk):
    return [(s, min(total, s + chunk)) for s in range(0, total, chunk)]


def _chunk_cumsum(x, chunk):
    return torch.cat([x[:, s:e].float().cumsum(dim=1) for s, e in _bounds(x.shape[1], chunk)], dim=1)


def _pairwise_decay(g_blk):
    return _exp2(g_blk[:, None, :] - g_blk[None, :, :])


def make_inputs(params):
    """The A5 backward unit's generator: same distributions, same draw order."""
    b, h, hv, c = (int(params[name]) for name in ("B", "H", "HV", "C"))
    k_dim, v_dim = int(params.get("K", 128)), int(params.get("V", 128))
    t = c * CHUNK
    gen = torch.Generator(device="cpu").manual_seed(int(params["seed"]))

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    q = (randn(b, t, h, k_dim) * INPUT_SCALE).to(IO_DTYPE)
    k = (randn(b, t, h, k_dim) * INPUT_K_SCALE).to(IO_DTYPE)
    v = (randn(b, t, hv, v_dim) * INPUT_SCALE).to(IO_DTYPE)
    g = (-F.softplus(randn(b, t, hv, k_dim))).to(IO_DTYPE)
    beta = torch.sigmoid(randn(b, t, hv)).to(IO_DTYPE)
    initial_state = (randn(b, hv, k_dim, v_dim) * INPUT_SCALE).to(IO_DTYPE)
    do = (randn(b, t, hv, v_dim) * INPUT_SCALE).to(IO_DTYPE)
    dht = (randn(b, hv, k_dim, v_dim) * INPUT_SCALE).to(IO_DTYPE)

    multiplier = params.get("gate_multiplier", 1)
    if multiplier != 1:
        g = (g.float() * multiplier).to(IO_DTYPE)
    if params.get("initial_state", "random") == "zero":
        initial_state = torch.zeros_like(initial_state)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": initial_state, "do": do, "dht": dht}


def build_saved_forward(q, k, v, g, beta, initial_state, chunk=CHUNK):
    """The nine forward checkpoints, FP32 arithmetic; g_cumsum retained FP32, other eight caches stored BF16."""
    b, t, h, k_dim = q.shape
    hv, v_dim = v.shape[2], v.shape[3]
    group = hv // h
    scale = k_dim ** -0.5
    qf, kf, vf, betaf = q.float(), k.float(), v.float(), beta.float()
    gf = _chunk_cumsum(g.float(), chunk) * RCP_LN2
    bounds = _bounds(t, chunk)

    out = {name: torch.empty(b, t, hv, k_dim, dtype=IO_DTYPE) for name in ("w", "qg", "kg")}
    out.update({name: torch.empty(b, t, hv, v_dim, dtype=IO_DTYPE) for name in ("u", "v_new")})
    out.update({name: torch.zeros(b, t, hv, chunk, dtype=IO_DTYPE) for name in ("Aqk", "Akk")})
    out["h"] = torch.empty(b, len(bounds), hv, k_dim, v_dim, dtype=IO_DTYPE)
    states = initial_state.float().clone()

    for bi in range(b):
        for ci, (s, e) in enumerate(bounds):
            n = e - s
            for hvi in range(hv):
                hq = hvi // group
                q_blk, k_blk = qf[bi, s:e, hq], kf[bi, s:e, hq]
                v_blk, g_blk, beta_blk = vf[bi, s:e, hvi], gf[bi, s:e, hvi], betaf[bi, s:e, hvi]
                decay = _pairwise_decay(g_blk)
                score = torch.tril((q_blk[:, None, :] * k_blk[None, :, :] * decay).sum(-1)) * scale
                kk = (k_blk[:, None, :] * k_blk[None, :, :] * decay).sum(-1)
                a_blk = torch.linalg.inv(torch.eye(n) + torch.tril(kk * beta_blk[:, None], diagonal=-1))
                exp_g = _exp2(g_blk)
                g_last = g_blk[n - 1]
                state = states[bi, hvi]
                w_blk = a_blk @ (k_blk * beta_blk[:, None] * exp_g)
                u_blk = a_blk @ (v_blk * beta_blk[:, None])
                kg_blk = k_blk * _exp2(g_last[None, :] - g_blk)
                v_new_blk = u_blk - w_blk @ state
                out["Aqk"][bi, s:e, hvi, :n] = score.to(IO_DTYPE)
                out["Akk"][bi, s:e, hvi, :n] = a_blk.to(IO_DTYPE)
                out["w"][bi, s:e, hvi] = w_blk.to(IO_DTYPE)
                out["u"][bi, s:e, hvi] = u_blk.to(IO_DTYPE)
                out["qg"][bi, s:e, hvi] = (q_blk * exp_g).to(IO_DTYPE)
                out["kg"][bi, s:e, hvi] = kg_blk.to(IO_DTYPE)
                out["v_new"][bi, s:e, hvi] = v_new_blk.to(IO_DTYPE)
                out["h"][bi, ci, hvi] = state.to(IO_DTYPE)
                states[bi, hvi] = state * _exp2(g_last)[:, None] + kg_blk.transpose(-1, -2) @ v_new_blk
    out["g_cumsum"] = gf
    return {name: out[name] for name in
            ("g_cumsum", "Aqk", "Akk", "w", "u", "qg", "kg", "v_new", "h")}


def _forward_f32(q, k, v, g, beta, initial_state, chunk=CHUNK):
    """Differentiable FP32 chunked forward; the oracle differentiates this."""
    b, t, h, k_dim = q.shape
    hv = v.shape[2]
    group = hv // h
    scale = k_dim ** -0.5
    gf = _chunk_cumsum(g, chunk) * RCP_LN2
    bounds = _bounds(t, chunk)
    outs, finals = [], []
    for bi in range(b):
        hv_out, hv_state = [], []
        for hvi in range(hv):
            hq = hvi // group
            state = initial_state[bi, hvi]
            blocks = []
            for s, e in bounds:
                n = e - s
                q_blk, k_blk = q[bi, s:e, hq], k[bi, s:e, hq]
                v_blk, g_blk, beta_blk = v[bi, s:e, hvi], gf[bi, s:e, hvi], beta[bi, s:e, hvi]
                decay = _pairwise_decay(g_blk)
                score = torch.tril((q_blk[:, None, :] * k_blk[None, :, :] * decay).sum(-1))
                kk = (k_blk[:, None, :] * k_blk[None, :, :] * decay).sum(-1)
                a_blk = torch.linalg.inv(torch.eye(n) + torch.tril(kk * beta_blk[:, None], diagonal=-1))
                exp_g = _exp2(g_blk)
                g_last = g_blk[n - 1]
                w_blk = a_blk @ (k_blk * beta_blk[:, None] * exp_g)
                u_blk = a_blk @ (v_blk * beta_blk[:, None])
                kg_blk = k_blk * _exp2(g_last[None, :] - g_blk)
                v_new_blk = u_blk - w_blk @ state
                blocks.append(torch.tril(score * scale) @ v_new_blk + scale * ((q_blk * exp_g) @ state))
                state = state * _exp2(g_last)[:, None] + kg_blk.transpose(-1, -2) @ v_new_blk
            hv_out.append(torch.cat(blocks, dim=0))
            hv_state.append(state)
        outs.append(torch.stack(hv_out, dim=1))
        finals.append(torch.stack(hv_state, dim=0))
    return torch.stack(outs, dim=0), torch.stack(finals, dim=0)


def independent_reference(inputs):
    """BF16 inputs -> the six BF16 gradients, by autograd through the FP32 forward."""
    leaves = {name: inputs[name].detach().clone().float().requires_grad_(True)
              for name in ("q", "k", "v", "g", "beta", "initial_state")}
    output, final_state = _forward_f32(*(leaves[name] for name in
                                         ("q", "k", "v", "g", "beta", "initial_state")))
    loss = (output * inputs["do"].float()).sum() + (final_state * inputs["dht"].float()).sum()
    dq, dk, dv, dg, dbeta, dh0 = torch.autograd.grad(
        loss, tuple(leaves[name] for name in ("q", "k", "v", "g", "beta", "initial_state")))
    return {"dq": dq.detach().to(IO_DTYPE), "dk": dk.detach().to(IO_DTYPE), "dv": dv.detach().to(IO_DTYPE),
            "dbeta": dbeta.detach().to(IO_DTYPE), "dg": dg.detach().to(IO_DTYPE), "dh0": dh0.detach().to(IO_DTYPE)}


def replay_forward(saved, chunk=CHUNK):
    """Rebuild o / final_state from the checkpoints alone; used to validate the cache set."""
    b, t, hv, k_dim = saved["g_cumsum"].shape
    v_dim = saved["v_new"].shape[-1]
    scale = k_dim ** -0.5
    bounds = _bounds(t, chunk)
    out = torch.empty(b, t, hv, v_dim, dtype=torch.float32)
    final = torch.empty(b, hv, k_dim, v_dim, dtype=torch.float32)
    for bi in range(b):
        for hvi in range(hv):
            state = None
            for ci, (s, e) in enumerate(bounds):
                n = e - s
                aqk = torch.tril(saved["Aqk"][bi, s:e, hvi, :n].float())
                qg = saved["qg"][bi, s:e, hvi].float()
                kg = saved["kg"][bi, s:e, hvi].float()
                h_blk = saved["h"][bi, ci, hvi].float()
                v_new = saved["v_new"][bi, s:e, hvi].float()
                out[bi, s:e, hvi] = aqk @ v_new + scale * (qg @ h_blk)
                g_last = saved["g_cumsum"][bi, e - 1, hvi].float()
                state = h_blk * _exp2(g_last)[:, None] + kg.transpose(-1, -2) @ v_new
            final[bi, hvi] = state
    return out, final


# --------------------------------------------------------------------------------------------------
# Per-kernel checkpoints.
#
# The oracle above differentiates the unrounded forward and is the acceptance reference. These
# checkpoints instead express the BF16 storage seams between the nine kernels, so a mismatch can be
# attributed to one kernel. They follow the seams of ascriptor's a5.kda_bwd staged reference: the
# FP32 beta-gradient seam, the dense (un-masked) scan scores, and FP32 accumulation before the final
# head reduction.
# --------------------------------------------------------------------------------------------------

def _pack(x):
    """[B, T, heads, W] -> [B, heads, C, 64, W]."""
    b, t, heads, w = x.shape
    return x.reshape(b, t // CHUNK, CHUNK, heads, w).permute(0, 3, 1, 2, 4).contiguous()


def _unpack(x):
    """[B, heads, C, 64, W] -> [B, T, heads, W]."""
    b, heads, c, n, w = x.shape
    return x.permute(0, 2, 3, 1, 4).reshape(b, c * n, heads, w).contiguous()


def _bf(x):
    return x.to(IO_DTYPE)


def _tr(x):
    return x.transpose(-1, -2)


def reference_stages(inputs):
    def ex2(x):
        # the kernels spell exp2 as an FP32 multiply by ln2 followed by the natural exp
        return torch.exp(x * math.log(2.0))

    saved = inputs["saved"]
    b, t, heads, width = inputs["q"].shape
    hv, c = inputs["v"].shape[2], t // CHUNK
    group, scale = hv // heads, width ** -0.5
    q = _pack(inputs["q"].repeat_interleave(group, dim=2)).float()
    k = _pack(inputs["k"].repeat_interleave(group, dim=2)).float()
    v, grad = _pack(inputs["v"]).float(), _pack(inputs["do"]).float()
    beta = inputs["beta"].reshape(b, c, CHUNK, hv).permute(0, 3, 1, 2).float()
    gate = _pack(saved["g_cumsum"]).float()
    aqk, akk = _pack(saved["Aqk"]).float(), _pack(saved["Akk"]).float()
    kg, qg, w = (_pack(saved[name]).float() for name in ("kg", "qg", "w"))
    new = _pack(saved["v_new"]).float()
    h = saved["h"].permute(0, 2, 1, 3, 4).float()

    # scan_fused
    daqk = _bf((grad @ _tr(new)) * scale)
    dv0 = _tr(aqk) @ grad
    dv = torch.empty_like(new, dtype=IO_DTYPE)
    dh = torch.empty_like(h, dtype=IO_DTYPE)
    state = inputs["dht"].float().clone()
    for ci in range(c - 1, -1, -1):
        dh[:, :, ci] = _bf(state)
        current = _bf(dv0[:, :, ci] + kg[:, :, ci] @ _bf(state).float())
        dv[:, :, ci] = current
        state = (state * ex2(gate[:, :, ci, -1, :, None])
                 + (_tr(qg[:, :, ci]) @ grad[:, :, ci]) * scale - _tr(w[:, :, ci]) @ current.float())
    dh0 = _bf(state)

    # inverse_mm (d_w carries the negation, which this unit does in the kernel)
    dqg = _bf(grad @ _tr(h))
    dkg = _bf(new @ _tr(dh.float()))
    dw = _bf(-(dv.float() @ _tr(h)))
    dvbeta = _bf(_tr(akk) @ dv.float())
    dkbetag = _bf(_tr(akk) @ dw.float())

    # inverse_epilogue
    eg = ex2(gate)
    glast = gate[..., -1:, :]
    elast = ex2(glast - gate)
    dq = dqg.float() * eg * scale
    dkkg = dkg.float() * elast
    kexp = k * eg
    dvout = _bf(dvbeta.float() * beta[..., None])
    dbeta = (dvbeta.float() * v).sum(-1) + (dkbetag.float() * kexp).sum(-1)
    dk = _bf(dkkg + dkbetag.float() * beta[..., None] * eg)
    dglast = (h * dh.float()).sum(-1) * ex2(gate[..., -1, :]) + (k * dkkg).sum(-2)
    dgcore = _bf(q * dq - k * dkkg + dkbetag.float() * kexp * beta[..., None])
    dgcore[..., -1, :] = _bf(dgcore[..., -1, :].float() + dglast)
    dq, kexp = _bf(dq), _bf(kexp)

    # inverse_dainv and inverse_dakk_fused (beta scales by column)
    dainv = dv.float() @ _tr(v) + dw.float() @ _tr(kexp.float())
    dtri = _bf(torch.tril(dainv * beta[..., None, :], diagonal=-1))
    temp = _bf(dtri.float() @ _tr(akk))
    dakk = _bf(torch.tril(-(_tr(akk) @ temp.float()), diagonal=-1))

    # finalize_pre / finalize_pair / finalize_post / finalize_reduce
    # the stable finalize anchors the pairwise decay at the chunk midpoint (g_last / 2), not at the
    # endpoint: the anchor cancels when the pair is recombined, but it halves the exponent range and it
    # changes these intermediate seams, so the checkpoints must use the same anchor as the kernels.
    gmid = glast * 0.5
    rowscale, colscale = ex2(gate - gmid), ex2(gmid - gate)
    qscaled, kscaled, scaled_k = _bf(q * rowscale), _bf(k * rowscale), _bf(k * colscale)
    mqk, mbase = torch.tril(daqk), torch.tril(dakk, diagonal=-1)
    mbeta = _bf(mbase.float() * beta[..., :, None])
    qkl = _bf(mqk.float() @ scaled_k.float())
    qkr = _bf(_tr(mqk.float()) @ qscaled.float())
    sbase = _bf(mbase.float() @ scaled_k.float())
    tbeta = _bf(_tr(mbeta.float()) @ kscaled.float())

    dqpair, dkpair = rowscale * qkl.float(), colscale * qkr.float()
    row_contrib, col_contrib = rowscale * sbase.float(), colscale * tbeta.float()
    dqpost = dq.float() + dqpair
    dkpost = dk.float() + dkpair + (row_contrib * beta[..., None] + col_contrib)
    dbpost = _bf(dbeta + (k * row_contrib).sum(-1))
    dg = dgcore.float() + (q * dqpair - k * dkpair) + (k * row_contrib * beta[..., None] - k * col_contrib)
    dg = _bf(torch.flip(torch.cumsum(torch.flip(dg, dims=(-2,)), dim=-2), dims=(-2,)))
    dqfinal = _bf(_unpack(dqpost).reshape(b, t, heads, group, width).sum(3))
    dkfinal = _bf(_unpack(dkpost).reshape(b, t, heads, group, width).sum(3))

    def scalar_public(x):
        return x.permute(0, 2, 3, 1).reshape(b, t, hv).contiguous()

    def token_native(x):
        return x.reshape(b, hv, t, x.shape[-1])

    return {
        "scan.dAqk": _unpack(daqk), "scan.dh": dh.permute(0, 2, 1, 3, 4).contiguous(),
        "scan.dv": _unpack(dv), "scan.dh0": dh0,
        "inverse_mm.d_qg": dqg, "inverse_mm.d_kg": dkg, "inverse_mm.d_w": dw,
        "inverse_mm.d_v_beta": dvbeta, "inverse_mm.d_k_beta_g": dkbetag,
        "inverse_epilogue.dq_hv": _unpack(dq), "inverse_epilogue.dk_hv": _unpack(dk),
        "inverse_epilogue.dv": _unpack(dvout), "inverse_epilogue.dbeta": scalar_public(dbeta),
        "inverse_epilogue.dg_core": _unpack(dgcore), "inverse_epilogue.k_exp": kexp,
        "inverse_dainv.D_tri": dtri, "inverse_dakk.dAkk": _unpack(dakk),
        "finalize_pre.q_scaled": qscaled, "finalize_pre.k_scaled": kscaled, "finalize_pre.kg": scaled_k,
        "finalize_pre.M_qk": mqk, "finalize_pre.M_base": mbase, "finalize_pre.M_beta": mbeta,
        "finalize_pair.qk_left": token_native(qkl), "finalize_pair.qk_right": token_native(qkr),
        "finalize_pair.s_base": token_native(sbase), "finalize_pair.t_beta": token_native(tbeta),
        "finalize_post.dq_hv": dqpost, "finalize_post.dk_hv": dkpost,
        "finalize_post.dbeta": scalar_public(dbpost), "finalize_post.dg": _unpack(dg),
        "finalize_reduce.dq": dqfinal, "finalize_reduce.dk": dkfinal,
    }
