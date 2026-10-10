"""A2-K1 acceptance test for the a2 GDN chunk forward + backward units.

Hardware test (a2 / 910B3): chains the forward (prepare -> scores -> wy -> scan_cube
-> output) and the backward (replay -> reverse_rec -> reverse_dq/dbeta/dg/dk_bz/dk_ddr
-> group_reduce), and checks every A2-K1 ABI gap (docs/pm/tasks/A2-K1.md §1.1-1.7, §2)
against a dual fp32 CPU oracle (head-local grouped recurrence + block-solve) and
autograd grads. Skipped automatically when torch_npu / an a2 device is unavailable.

The forward's chunk recurrence (`scan.py`) does its two matrix contractions on the
cube (matmul + GMBuff ring) because the earlier pure-vector matvec faulted on b3
(vector UB-address fault); the cube form is the shipped path.
"""
import importlib.util
import pathlib

import pytest

try:
    import torch
    import torch_npu  # noqa: F401
    _HAS_NPU = torch.npu.is_available()
except Exception:  # pragma: no cover - import guard
    _HAS_NPU = False

pytestmark = pytest.mark.skipif(not _HAS_NPU, reason="requires an a2 (910B3) NPU + torch_npu")

D, C = 128, 64
SCALE = D ** -0.5
DEV = "npu:0"
BLOCK_DIM = 40
_REPO = pathlib.Path(__file__).resolve().parents[1]
_FKD = _REPO / "kernels/projects/a2/gdn_chunk_fwd/kernels"
_BKD = _REPO / "kernels/projects/a2/gdn_chunk_bwd/kernels"


def _ld(kd, stem, fn):
    from ascend_fla.runtime.compile import compile_kernel
    s = importlib.util.spec_from_file_location(stem, kd / (stem + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return compile_kernel(getattr(m, fn), device="a2", block_dim=BLOCK_DIM, backend="cce")


def _compile():
    fwd = {
        "prep": _ld(_FKD, "prepare", "gdn_chunk_prepare_a2_kernel"),
        "scor": _ld(_FKD, "scores", "gdn_chunk_scores_a2_kernel"),
        "wy": _ld(_FKD, "wy", "gdn_chunk_wy_a2_kernel"),
        "scan": _ld(_FKD, "scan", "gdn_chunk_scan_a2_kernel"),
        "out": _ld(_FKD, "output", "gdn_chunk_output_a2_kernel"),
    }
    bwd = {stem: _ld(_BKD, stem, fn) for stem, fn in [
        ("replay", "gdn_chunk_bwd_replay_a2_kernel"),
        ("reverse_rec", "gdn_chunk_bwd_reverse_rec_a2_kernel"),
        ("reverse_dq", "gdn_chunk_bwd_reverse_dq_a2_kernel"),
        ("reverse_dbeta", "gdn_chunk_bwd_reverse_dbeta_a2_kernel"),
        ("reverse_dg", "gdn_chunk_bwd_reverse_dg_a2_kernel"),
        ("reverse_dk_bz", "gdn_chunk_bwd_reverse_dk_bz_a2_kernel"),
        ("reverse_dk_ddr", "gdn_chunk_bwd_reverse_dk_ddr_a2_kernel"),
        ("group_reduce", "gdn_chunk_bwd_group_reduce_a2_kernel"),
    ]}
    return fwd, bwd


def _rel(a, b):
    a = a.reshape(-1).float().cpu(); b = b.reshape(-1).float().cpu()
    return (a - b).norm().item() / max(b.norm().item(), 1e-30)


def _dev(x):
    return x.detach().to(DEV).float().contiguous()


def _mk(B, T, H, HV, seed=11, gate=0.1, distinct=False):
    gg = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, D, generator=gg) * 0.05
    k = torch.randn(B, T, H, D, generator=gg) * 0.05
    v = torch.randn(B, T, HV, D, generator=gg) * 0.05
    beta = torch.rand(B, T, HV, generator=gg)
    g = -torch.rand(B, T, HV, generator=gg) * gate
    if distinct:  # distinct per-value-head stats so a wrong i_h = i_hv // (HV//H) is caught
        for j in range(HV):
            v[:, :, j] *= (1.0 + 0.5 * j); beta[:, :, j] *= (0.3 + 0.2 * j); g[:, :, j] *= (0.5 + 0.3 * j)
    return q, k, v, beta, g


def _run_forward(fwd, q, k, v, beta, g, h0):
    """token-major in; T must be a multiple of the chunk size (gap §1.7)."""
    B, T, H = q.shape[:3]; HV = v.shape[2]
    if T % C != 0:
        raise ValueError(f"T={T} must be a multiple of chunk size {C} (no tail path in Batch A)")
    N = T // C
    sc = {"scale": SCALE, "B": B, "T": T, "H": H, "HV": HV, "N": N,
          "BT": B * T, "HD": H * D, "HVD": HV * D}
    qd = _dev(q).view(B * T, H * D); kd = _dev(k).view(B * T, H * D); vd = _dev(v).view(B * T, HV * D)
    gd = _dev(g).view(B * T, HV); btd = _dev(beta).view(B * T, HV)
    z5 = lambda: torch.zeros(B, N, HV, C, D, device=DEV)
    qn, kn, gc, bk, wv = z5(), z5(), z5(), z5(), z5()
    fwd["prep"]({"q": qd, "k": kd, "v": vd, "g": gd, "beta": btd}, sc,
                {"qn": qn, "kn": kn, "gc": gc, "bk": bk, "wv": wv})
    lower = torch.zeros(B, N, HV, C, C, device=DEV); score = torch.zeros(B, N, HV, C, C, device=DEV)
    fwd["scor"]({"qn": qn, "kn": kn, "gc": gc, "bk": bk}, sc, {"lower": lower, "score": score})
    u, wy = z5(), z5()
    fwd["wy"]({"lower": lower, "gc": gc, "bk": bk, "wv": wv}, sc, {"u": u, "wy": wy})
    states = torch.zeros(B, N, HV, D, D, device=DEV); delta = z5(); fs = torch.zeros(B, HV, D, D, device=DEV)
    fwd["scan"]({"kn": kn, "gc": gc, "u": u, "wy": wy, "initial_state": _dev(h0)}, sc,
                {"states": states, "delta": delta, "final_state": fs})
    o = torch.zeros(B * T, HV * D, device=DEV)
    fwd["out"]({"qn": qn, "gc": gc, "score": score, "states": states, "delta": delta}, sc, {"o": o})
    torch.npu.synchronize()
    return o.view(B, T, HV, D), fs


def _run_backward(bwd, q, k, v, beta, g, h0, dout, dht):
    B, T, H = q.shape[:3]; HV = v.shape[2]; N = T // C
    dims = {"B": B, "T": T, "H": H, "HV": HV, "N": N, "BT": B * T, "HD": H * D, "HVD": HV * D}
    x = {"q": _dev(q).view(B * T, H * D), "k": _dev(k).view(B * T, H * D), "v": _dev(v).view(B * T, HV * D),
         "dout": _dev(dout).view(B * T, HV * D), "g": _dev(g).view(B * T, HV), "beta": _dev(beta).view(B * T, HV),
         "h0": _dev(h0), "dht": _dev(dht)}
    tape = lambda: torch.zeros(B, HV, T, D, D, device=DEV)
    z2 = lambda c: torch.zeros(B * T, c, device=DEV); zhv = lambda: torch.zeros(B * T, HV, device=DEV)
    tape_d, back_tape = tape(), tape()
    dv = z2(HV * D); dh0 = torch.zeros(B, HV, D, D, device=DEV)
    dqp, dkbz, dkddr = z2(HV * D), z2(HV * D), z2(HV * D); dbeta = zhv(); dg = zhv()
    dq = z2(H * D); dk = z2(H * D)
    bwd["replay"]({"k": x["k"], "v": x["v"], "g": x["g"], "beta": x["beta"], "initial_state": x["h0"]},
                  dims, {"tape_d": tape_d})
    bwd["reverse_rec"]({"q": x["q"], "k": x["k"], "g": x["g"], "beta": x["beta"], "dout": x["dout"], "dht": x["dht"]},
                       {"scale": SCALE, **dims}, {"dv": dv, "dh0": dh0, "back_tape": back_tape})
    bwd["reverse_dq"]({"k": x["k"], "v": x["v"], "beta": x["beta"], "dout": x["dout"], "tape_d": tape_d},
                      {"scale": SCALE, **dims}, {"dq_parts": dqp})
    bwd["reverse_dbeta"]({"k": x["k"], "v": x["v"], "tape_d": tape_d, "back_tape": back_tape}, dims, {"dbeta": dbeta})
    bwd["reverse_dg"]({"k": x["k"], "beta": x["beta"], "tape_d": tape_d, "back_tape": back_tape}, dims, {"dg": dg})
    bwd["reverse_dk_bz"]({"k": x["k"], "v": x["v"], "beta": x["beta"], "tape_d": tape_d, "back_tape": back_tape},
                         dims, {"dkbz_parts": dkbz})
    bwd["reverse_dk_ddr"]({"k": x["k"], "beta": x["beta"], "tape_d": tape_d, "back_tape": back_tape},
                          dims, {"dkddr_parts": dkddr})
    bwd["group_reduce"]({"dq_parts": dqp, "dkbz_parts": dkbz, "dkddr_parts": dkddr}, dims, {"dq": dq, "dk": dk})
    torch.npu.synchronize()
    return (dq.view(B, T, H, D), dk.view(B, T, H, D), dv.view(B, T, HV, D),
            dbeta.view(B, T, HV), dg.view(B, T, HV), dh0)


def _grouped_recurrent(q, k, v, beta, g, h0):
    """head-local fp32 recurrence oracle; nonzero init supported, differentiable."""
    B, T, HV, d = v.shape; H = q.shape[2]; ratio = HV // H
    out, finals = [], []
    for j in range(HV):
        state = h0[:, j].clone(); os = []
        for i in range(T):
            key = k[:, i, j // ratio]
            state = state * g[:, i, j].exp()[:, None, None]
            residual = (v[:, i, j] - torch.einsum('bk,bkv->bv', key, state)) * beta[:, i, j, None]
            state = state + key[:, :, None] * residual[:, None, :]
            os.append(torch.einsum('bk,bkv->bv', q[:, i, j // ratio] * (d ** -0.5), state))
        out.append(torch.stack(os, 1)); finals.append(state)
    return torch.stack(out, 2), torch.stack(finals, 1)


def _block_solve(q, k, v, beta, g, h0):
    h, hv = q.shape[2], v.shape[2]
    if hv != h:
        q, k = (torch.stack([x[:, :, j // (hv // h)] for j in range(hv)], dim=2) for x in (q, k))
    q, k, v = (x.transpose(1, 2).float() for x in (q, k, v))
    beta, g = (x.transpose(1, 2).float() for x in (beta, g))
    state = h0.clone().float(); output = torch.empty_like(v)
    eye = torch.eye(64); causal = torch.ones(64, 64, dtype=torch.bool).tril()
    for start in range(0, q.shape[2], 64):
        sl = slice(start, start + 64)
        kc, qc, vc, bc = k[:, :, sl], q[:, :, sl], v[:, :, sl], beta[:, :, sl]
        prefix = g[:, :, sl].cumsum(-1); dd = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
        decay = dd.masked_fill(~causal, -torch.inf).exp()
        lower = ((kc @ kc.transpose(-1, -2)) * decay * bc.unsqueeze(-1)).tril(-1)
        rhs = bc.unsqueeze(-1) * (vc - prefix.exp().unsqueeze(-1) * (kc @ state))
        upd = torch.linalg.solve_triangular(eye + lower, rhs, upper=False, unitriangular=True)
        output[:, :, sl] = ((qc @ state) * prefix.exp().unsqueeze(-1)
                            + ((qc @ kc.transpose(-1, -2)) * decay) @ upd) / (128 ** 0.5)
        weights = (prefix[..., -1:] - prefix).exp()
        state = state * prefix[..., -1:].exp().unsqueeze(-1) + (kc * weights.unsqueeze(-1)).transpose(-1, -2) @ upd
    return output.transpose(1, 2).contiguous(), state


_TOL = 1e-4  # fp32 cube/vec chain vs fp32 CPU oracle


def test_a2_gdn_chunk_forward_gva_and_state():
    """§1.1 GQA/GVA (HV!=H), §1.2 token-major, §1.3 nonzero init, §1.7 scale — forward o/final_state."""
    fwd, _ = _compile()
    B, T, H, HV = 1, 128, 2, 4  # HV != H, N=2
    q, k, v, beta, g = _mk(B, T, H, HV, distinct=True)
    for tag, h0 in [("zero", torch.zeros(B, HV, D, D)),
                    ("nonzero", torch.randn(B, HV, D, D, generator=torch.Generator().manual_seed(5)) * 0.02)]:
        o, fs = _run_forward(fwd, q, k, v, beta, g, h0)
        o_blk, fs_blk = _block_solve(q, k, v, beta, g, h0)
        o_rec, fs_rec = _grouped_recurrent(q, k, v, beta, g, h0)
        assert _rel(o, o_blk) < _TOL and _rel(o, o_rec) < _TOL, f"{tag} o: {_rel(o,o_blk):.2e}/{_rel(o,o_rec):.2e}"
        assert _rel(fs, fs_blk) < _TOL and _rel(fs, fs_rec) < _TOL, f"{tag} fs: {_rel(fs,fs_blk):.2e}/{_rel(fs,fs_rec):.2e}"


def test_a2_gdn_chunk_segment_chaining():
    """§2 segment chaining: seg1 final_state -> seg2 initial_state matches a single-pass oracle."""
    fwd, _ = _compile()
    B, H, HV = 1, 2, 4
    qL, kL, vL, betaL, gL = _mk(B, 256, H, HV, seed=31, distinct=True)
    z = torch.zeros(B, HV, D, D)
    oL, fsL = _grouped_recurrent(qL, kL, vL, betaL, gL, z)
    a, b = slice(0, 128), slice(128, 256)
    o1, fs1 = _run_forward(fwd, qL[:, a], kL[:, a], vL[:, a], betaL[:, a], gL[:, a], z)
    o2, fs2 = _run_forward(fwd, qL[:, b], kL[:, b], vL[:, b], betaL[:, b], gL[:, b], fs1.cpu())
    o_cat = torch.cat([o1.cpu(), o2.cpu()], dim=1)
    assert _rel(o_cat, oL) < _TOL and _rel(fs2, fsL) < _TOL


def test_a2_gdn_chunk_backward_grads():
    """§1.4 backward dh0 + accepts dht; all grads vs autograd through the oracle (HV!=H)."""
    _, bwd = _compile()
    B, T, H, HV = 1, 128, 2, 4
    q, k, v, beta, g = _mk(B, T, H, HV, distinct=True)
    h0 = torch.zeros(B, HV, D, D)
    tens = [x.clone().requires_grad_(True) for x in (q, k, v, beta, g)]
    h0r = h0.clone().requires_grad_(True)
    o_t, fs_t = _grouped_recurrent(*tens, h0r)
    dout = torch.randn(B, T, HV, D, generator=torch.Generator().manual_seed(23)) * 0.05
    for tag, dht in [("zero_dht", torch.zeros(B, HV, D, D)),
                     ("nonzero_dht", torch.randn(B, HV, D, D, generator=torch.Generator().manual_seed(7)) * 0.05)]:
        go = torch.autograd.grad([o_t, fs_t], tens + [h0r], grad_outputs=[dout, dht], retain_graph=True)
        got = _run_backward(bwd, q, k, v, beta, g, h0, dout, dht)
        for nm, a, b in zip(("dq", "dk", "dv", "dbeta", "dg", "dh0"), got, go):
            assert _rel(a, b) < _TOL, f"{tag} {nm}: {_rel(a, b):.2e}"


def test_a2_gdn_chunk_rejects_non_multiple_T():
    """§1.7 no-tail-path: T not a multiple of 64 is explicitly rejected."""
    fwd, _ = _compile()
    q, k, v, beta, g = _mk(1, 100, 2, 4)  # 100 % 64 != 0
    with pytest.raises(ValueError):
        _run_forward(fwd, q, k, v, beta, g, torch.zeros(1, 4, D, D))
