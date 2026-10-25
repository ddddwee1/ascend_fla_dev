#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""Per-Module accumulative golden — Modules 1..10 (stage12345678910) of gated_delta_rule_bwd.

Self-contained, pure-torch fp32 CPU reference. Implements the chunk-parallel BACKWARD
of Gated Delta Rule from PRIMARY inputs (grad_o,q,k,v,g,beta) and returns Module 10's
`outputs` as a tuple, in the documented order.

Derived from gated_delta_rule_bwd_golden_cpu.py (CPU FP32 autograd VJP ground truth) and
the chunk-parallel backward the kernels implement. Recompute
M1-M5 mirror the forward chunk intermediates; M6-M10 are the backward-adjoint stages.

g is log-space (<=0); beta are sigmoid LOGITS (sigmoid applied inside); dbeta is w.r.t. the
LOGIT, dg is w.r.t. LOG-SPACE g. scale = 1/sqrt(K); C=64 chunk; no q/k l2norm; H==HV.
"""

import math
import torch
import torch.nn.functional as F

_C = 64

def _prep(grad_o, q, k, v, g, beta):
    """Primary [B,T,H,*] -> chunked fp32 [B,H,N,C,*]; pad T->N*C (C=64).

    Mirrors the forward golden: sigmoid(beta_logit) THEN pad-with-0 (pad rows: beta_act=0,
    g=0 so decay is flat, k=0 -> no state change; identical lineage to the fwd staging golden).
    """
    C = _C
    q = q.to(torch.float32); k = k.to(torch.float32); v = v.to(torch.float32)
    g = g.to(torch.float32); do = grad_o.to(torch.float32)
    beta_logit = beta.to(torch.float32)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    beta_act_full = torch.sigmoid(beta_logit)                       # [B,T,H]
    # [B,T,H,*] -> [B,H,T,*]
    q = q.transpose(1, 2).contiguous(); k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous(); do = do.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()
    beta_act = beta_act_full.transpose(1, 2).contiguous()          # [B,H,T]
    T = q.shape[-2]
    pad = (C - (T % C)) % C
    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad)); do = F.pad(do, (0, 0, 0, pad))
        g = F.pad(g, (0, pad)); beta_act = F.pad(beta_act, (0, pad))
    b, h, Tp, K = q.shape
    V = v.shape[-1]
    N = Tp // C
    st = dict(b=b, h=h, N=N, C=C, K=K, V=V, T=T, scale=scale)
    st['q_c'] = q.reshape(b, h, N, C, K)
    st['k_c'] = k.reshape(b, h, N, C, K)
    st['v_c'] = v.reshape(b, h, N, C, V)
    st['do'] = do.reshape(b, h, N, C, V)
    st['g_c'] = g.reshape(b, h, N, C)
    st['beta_c'] = beta_act.reshape(b, h, N, C)
    return st

def _m1(st):
    """M1 recompute_preprocess (b0 s0-s3)."""
    b, h, N, C, K, V = st['b'], st['h'], st['N'], st['C'], st['K'], st['V']
    scale = st['scale']
    q_c, k_c, v_c = st['q_c'], st['k_c'], st['v_c']
    beta = st['beta_c']                                            # [b,h,N,C]
    q_s = q_c * scale                                             # [b,h,N,C,K]
    k_f32 = k_c
    k_beta = k_c * beta[..., None]
    v_b = v_c * beta[..., None]
    g_f32 = st['g_c']
    beta_act = beta
    sig = beta                                                    # sigma == beta_act
    q_T = q_s.transpose(-1, -2).contiguous()                      # [b,h,N,K,C]
    kf_T = k_f32.transpose(-1, -2).contiguous()                   # [b,h,N,K,C]
    st.update(q_s=q_s, k_f32=k_f32, k_beta=k_beta, v_b=v_b, g_f32=g_f32,
              beta_act=beta_act, sig=sig, q_T=q_T, kf_T=kf_T)
    return (q_s, k_f32, k_beta, v_b, g_f32, beta_act, sig, q_T, kf_T)

def _m2(st):
    """M2 recompute_cumsum_qk (b0 s5,s7mm,s11mm)."""
    g_f32, k_beta, k_f32, q_s = st['g_f32'], st['k_beta'], st['k_f32'], st['q_s']
    decay = g_f32.cumsum(-1)                                      # [b,h,N,C]
    kf_t = k_f32.transpose(-1, -2)                               # [b,h,N,K,C]
    P1 = torch.matmul(k_beta, kf_t)                              # [b,h,N,C,C] k_beta@k^T
    P2 = torch.matmul(q_s, kf_t)                                 # [b,h,N,C,C] q_s@k^T
    st.update(decay=decay, P1=P1, P2=P2)
    return (decay, P1, P2)

def _m3(st):
    """M3 recompute_wy (b0 s6,s7mask,s8,s9,s11mask)."""
    b, h, N, C, K = st['b'], st['h'], st['N'], st['C'], st['K']
    decay, P1, P2 = st['decay'], st['P1'], st['P2']
    q_s, k_beta = st['q_s'], st['k_beta']
    decay_exp = decay.exp()                                       # [b,h,N,C]
    q_de = q_s * decay_exp[..., None]
    kbeta_de = k_beta * decay_exp[..., None]
    dfac = (decay[..., -1:] - decay).exp()[..., None].expand(b, h, N, C, K).contiguous()
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp()).tril()   # [b,h,N,C,C]
    up_incl_diag = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
    attn = -((P1) * L_mask).masked_fill(up_incl_diag, 0)
    for i in range(1, C):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(C, dtype=torch.float32)
    up_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    A = ((P2) * L_mask).masked_fill(up_strict, 0)
    kbeta_T = kbeta_de.transpose(-1, -2).contiguous()            # [b,h,N,K,C]
    st.update(decay_exp=decay_exp, q_de=q_de, kbeta_de=kbeta_de, dfac=dfac,
              L_mask=L_mask, attn=attn, A=A, kbeta_T=kbeta_T)
    return (decay_exp, dfac, q_de, kbeta_de, L_mask, attn, A, kbeta_T)

def _m4(st):
    """M4 recompute_uw (b0 s10)."""
    attn, v_b, kbeta_de = st['attn'], st['v_b'], st['kbeta_de']
    u = torch.matmul(attn, v_b)                                  # [b,h,N,C,V]
    w = torch.matmul(attn, kbeta_de)                            # [b,h,N,C,K]
    w_T = w.transpose(-1, -2).contiguous()                      # [b,h,N,K,C]
    st.update(u=u, w=w, w_T=w_T)
    return (u, w, w_T)

def _m5(st):
    """M5 recompute_state_scan (b0 s12-s15). Stores per-chunk ENTERING state S_hist."""
    b, h, N, C, K, V = st['b'], st['h'], st['N'], st['C'], st['K'], st['V']
    u, w = st['u'], st['w']
    k_f32, dfac, decay_exp = st['k_f32'], st['dfac'], st['decay_exp']
    S = u.new_zeros(b, h, K, V)
    v_new = torch.zeros(b, h, N, C, V, dtype=torch.float32)
    S_hist = torch.zeros(b, h, N, K, V, dtype=torch.float32)
    for i in range(N):
        S_hist[:, :, i] = S
        v_prime = torch.matmul(w[:, :, i], S)                    # [b,h,C,V]
        vn = u[:, :, i] - v_prime
        v_new[:, :, i] = vn
        S = S * decay_exp[:, :, i, -1, None, None] + torch.matmul(
            (k_f32[:, :, i] * dfac[:, :, i]).transpose(-1, -2), vn)
    st.update(v_new=v_new, S_hist=S_hist)
    return (v_new, S_hist)

def _m6(st):
    """M6 dv_local (b1). A_l = tril_up(scale*k@q^T*exp(g_col-g_row)); dv_local = A_l@do."""
    C = st['C']
    k_f32, q_s, do, decay = st['k_f32'], st['q_s'], st['do'], st['decay']
    gg = decay                                                   # cumulative decay ("g")
    A_l = torch.matmul(k_f32, q_s.transpose(-1, -2))            # (k@q^T)*scale (scale folded in q_s)
    gdiff = gg.unsqueeze(-2) - gg.unsqueeze(-1)                 # [.,i,j] = g_j - g_i
    A_l = A_l * gdiff.exp()
    low_strict = torch.tril(torch.ones(C, C, dtype=torch.bool), diagonal=-1)  # i>j
    A_l = A_l.masked_fill(low_strict, 0.0)                      # keep upper incl diag (i<=j)
    dv_local = torch.matmul(A_l, do)                           # [C,C]@[C,V]->[C,V]
    st.update(dv_local=dv_local)
    return (dv_local,)

def _m7(st):
    """M7 reverse_dS_scan (b2). Reverse chunk scan carrying dS[K,V]."""
    b, h, N, C, K, V = st['b'], st['h'], st['N'], st['C'], st['K'], st['V']
    k_f32, q_de, w, do = st['k_f32'], st['q_de'], st['w'], st['do']
    decay, dv_local = st['decay'], st['dv_local']
    gg = decay
    g_last = gg[..., -1]                                        # [b,h,N]
    dS = torch.zeros(b, h, K, V, dtype=torch.float32)
    dh = torch.zeros(b, h, N, K, V, dtype=torch.float32)
    dv2 = torch.zeros(b, h, N, C, V, dtype=torch.float32)
    for i in range(N - 1, -1, -1):
        dh[:, :, i] = dS                                       # snapshot (entering dS = dl/dS_{i+1})
        dvc = torch.matmul(k_f32[:, :, i], dS)                # k@dS [C,K]@[K,V]->[C,V]
        gate = (g_last[:, :, i, None] - gg[:, :, i]).exp()     # exp(g_last-g_i) [b,h,C]
        dv2_i = dv_local[:, :, i] + dvc * gate[..., None]
        dv2[:, :, i] = dv2_i
        term_q = torch.matmul(q_de[:, :, i].transpose(-1, -2), do[:, :, i])   # q_de^T@do [K,V]
        term_w = torch.matmul(w[:, :, i].transpose(-1, -2), dv2_i)           # w^T@dv2 [K,V]
        dS = dS * g_last[:, :, i, None, None].exp() + term_q - term_w
    st.update(dh=dh, dv2=dv2)
    return (dh, dv2)

def _m8(st):
    """M8 dqkwg (b3). dq_p, dk_p, dw, dg_p (chunk-parallel)."""
    C = st['C']
    scale = st['scale']
    do, v_new, S_hist, dh, dv2 = st['do'], st['v_new'], st['S_hist'], st['dh'], st['dv2']
    k_f32, q_c, decay, decay_exp = st['k_f32'], st['q_c'], st['decay'], st['decay_exp']
    gg = decay
    g_last = gg[..., -1]                                        # [b,h,N]
    # ds = tril_low(do@v_new^T * exp(g_i-g_j) * scale)
    ds = torch.matmul(do, v_new.transpose(-1, -2))            # [C,V]@[V,C]->[C,C]; ds[i,j]=do_i.vnew_j
    gdiff = gg.unsqueeze(-1) - gg.unsqueeze(-2)               # [.,i,j] = g_i - g_j
    ds = ds * gdiff.exp() * scale
    up_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)   # i<j
    ds = ds.masked_fill(up_strict, 0.0)                        # keep lower incl diag (i>=j)
    # h=S_hist, dh stored [K,V]; inter-chunk terms contract over V -> use h^T/dh^T ([V,K]).
    h_T = S_hist.transpose(-1, -2)                            # [.,V,K]
    dh_T = dh.transpose(-1, -2)                               # [.,V,K]
    # dq = (do@h)*exp(g)*scale + ds@k
    dq_inter = torch.matmul(do, h_T) * decay_exp[..., None] * scale           # do@h^T [C,V]@[V,K]
    dq_intra = torch.matmul(ds, k_f32)                        # ds@k
    dq_p = dq_inter + dq_intra
    # dk_p = (v_new@dh)*exp(g_last-g) + ds^T@q
    dk_inter = torch.matmul(v_new, dh_T)                     # v_new@dh^T [C,V]@[V,K]->[C,K]
    gate_k = (g_last[..., None] - gg).exp()                    # exp(g_last-g_i) [b,h,N,C]
    dk_inter = dk_inter * gate_k[..., None]
    dk_intra = torch.matmul(ds.transpose(-1, -2), q_c)        # ds^T@q
    dk_p = dk_inter + dk_intra
    # dw = -(dv2@h)
    dw = -torch.matmul(dv2, h_T)                             # -dv2@h^T [C,V]@[V,K]->[C,K]
    # dg_last scalar per chunk = exp(g_last)*sum(h*dh) + sum(dk_inter*k)
    dg_last = g_last.exp() * (S_hist * dh).sum(dim=(-2, -1)) + (dk_inter * k_f32).sum(dim=(-2, -1))
    # dg_p per token = sum_K(dq_p*q) - sum_K(dk_p*k); dg_last folded onto last chunk row (C-1)
    dg_p = (dq_p * q_c).sum(-1) - (dk_p * k_f32).sum(-1)      # [b,h,N,C]
    dg_p = dg_p.clone()
    dg_p[..., -1] = dg_p[..., -1] + dg_last
    st.update(dq_p=dq_p, dk_p=dk_p, dw=dw, dg_p=dg_p)
    return (dq_p, dk_p, dw, dg_p)

def _m9(st):
    """M9 wy_adjoint (b4, b4b_dk). Closed-form WY-inverse adjoint."""
    C = st['C']
    attn, k_f32, v_b, beta_act = st['attn'], st['k_f32'], st['v_b'], st['beta_act']
    decay, decay_exp, kbeta_de = st['decay'], st['decay_exp'], st['kbeta_de']
    v_c, k_beta = st['v_c'], st['k_beta']
    dw, dv2, dk_p = st['dw'], st['dv2'], st['dk_p']
    gg = decay
    AT = attn.transpose(-1, -2)                                # A^T (FLA loads A transposed)
    # K-loop1
    dA = torch.matmul(dw, kbeta_de.transpose(-1, -2))         # dw@kbg^T (kbg=k*beta*exp(g)=kbeta_de)
    dkbg = torch.matmul(AT, dw)                               # A^T@dw
    dk_term1 = dkbg * (decay_exp * beta_act)[..., None]        # dkbg*exp(g)*beta
    db1 = (dkbg * k_f32 * decay_exp[..., None]).sum(-1)        # sum_K(dkbg*k*exp(g))
    dg2_loop1 = (dkbg * kbeta_de).sum(-1)                      # sum_K(dkbg*kbg)
    # V-loop
    dA = dA + torch.matmul(dv2, v_b.transpose(-1, -2))        # du@vb^T (vb=v_b)
    dvb = torch.matmul(AT, dv2)                               # A^T@du
    dv = dvb * beta_act[..., None]
    db2 = (dvb * v_c).sum(-1)                                  # sum_V(dvb*v)
    # closed-form inverse-VJP: dA = -tril_strict(exp(g_i-g_j) * (A^T tril_strict(dA) A^T))
    strict_low = torch.tril(torch.ones(C, C, dtype=torch.bool), diagonal=-1)  # i>j
    dA = dA * strict_low                                       # tril_strict(dA)
    dA = torch.matmul(dA, AT)                                  # dA@A^T
    dA = torch.matmul(AT, dA)                                  # A^T@(dA@A^T)
    gdiff = gg.unsqueeze(-1) - gg.unsqueeze(-2)               # g_i - g_j
    slf = strict_low.to(torch.float32)
    decay_fac = (gdiff * slf).exp() * slf                     # mask-before-exp: upper->0 (no inf)
    dA = dA * decay_fac                                        # strict-lower decayed
    dA = -dA                                                   # negated (already strict-lower via decay_fac)
    # K-loop2
    dkb = torch.matmul(dA, k_f32)                             # dA@k
    db3 = (dkb * k_f32).sum(-1)                                # sum_K(dkb*k)
    dkt = torch.matmul(dA.transpose(-1, -2), k_beta)          # dA^T@kb (kb=k_beta)
    dk2 = dkb * beta_act[..., None] + dkt + dk_term1
    # dg2
    A_kk = torch.matmul(k_f32, k_f32.transpose(-1, -2))       # k@k^T
    A_kk = A_kk * beta_act[..., None]                         # beta_i*(k_i.k_j)
    AdA = dA * A_kk
    dg2 = dg2_loop1 + AdA.sum(-1) - AdA.sum(-2)               # rowsum - colsum
    dbeta_wy = db1 + db2 + db3
    dk = dk_p + dk2                                            # b4b_dk cross-stage accumulate
    st.update(dk=dk, dv=dv, dbeta_wy=dbeta_wy, dg2=dg2)
    return (dk, dv, dbeta_wy, dg2)

def _m10(st):
    """M10 finalize_store (b5, b4b_dg, b6, b7). Final 5 public outputs (fp32 oracle)."""
    b, h, N, C, K, V, T = st['b'], st['h'], st['N'], st['C'], st['K'], st['V'], st['T']
    dq_p, dk, dv = st['dq_p'], st['dk'], st['dv']
    dg_p, dg2, dbeta_wy, sig = st['dg_p'], st['dg2'], st['dbeta_wy'], st['sig']
    dg_sum = dg_p + dg2                                        # [b,h,N,C]
    # b5: dg = reverse cumsum(dg_sum, axis=C) = triu(ones, incl diag) @ dg_sum
    triu_incl = torch.triu(torch.ones(C, C, dtype=torch.float32), diagonal=0)  # [i,j]=1 if j>=i
    dg_full = torch.matmul(triu_incl, dg_sum.unsqueeze(-1)).squeeze(-1)         # [b,h,N,C]
    # b6: dbeta = dbeta_wy * sigma*(1-sigma)  (chain through sigmoid logit)
    dbeta = dbeta_wy * sig * (1.0 - sig)
    # b7: slice [:T], transpose back to [B,T,H,*]

    def _unpad_t(x):
        d = x.shape[-1]
        x = x.reshape(b, h, N * C, d)[:, :, :T]
        return x.transpose(1, 2).contiguous()

    def _unpad_s(x):
        x = x.reshape(b, h, N * C)[:, :, :T]
        return x.transpose(1, 2).contiguous()

    dq = _unpad_t(dq_p)
    dk_o = _unpad_t(dk)
    dv_o = _unpad_t(dv)
    dg = _unpad_s(dg_full)
    dbeta_o = _unpad_s(dbeta)
    return (dq, dk_o, dv_o, dg, dbeta_o)

def gated_delta_rule_bwd_golden_stage12345678910(grad_o, q, k, v, g, beta):
    """Modules 1..10 accumulative golden. Returns Module 10's outputs (fp32)."""
    st = _prep(grad_o, q, k, v, g, beta)
    r1 = _m1(st)
    r2 = _m2(st)
    r3 = _m3(st)
    r4 = _m4(st)
    r5 = _m5(st)
    r6 = _m6(st)
    r7 = _m7(st)
    r8 = _m8(st)
    r9 = _m9(st)
    r10 = _m10(st)
    return r10

def _dims(B, T, H, K, V):
    N = (T + 63) // 64
    return dict(B=B, T=T, H=H, K=K, V=V, N=N, C=64,
                BHNCK=(B, H, N, 64, K), BHNCV=(B, H, N, 64, V), BHNC=(B, H, N, 64),
                BHNCC=(B, H, N, 64, 64), BHNKC=(B, H, N, K, 64), BHNKV=(B, H, N, K, V),
                BTHK=(B, T, H, K), BTHV=(B, T, H, V), BTH=(B, T, H))

def _mk_case(B, T, H, K, V, seed, g_lo=-4.0, g_hi=0.0):
    torch.manual_seed(seed)
    grad_o = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(g_lo, g_hi)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return grad_o, q, k, v, g, beta


_SHAPES = [(1, 128, 4, 64, 64), (1, 100, 4, 64, 64), (1, 192, 2, 128, 128), (1, 200, 3, 128, 96)]
_SEEDS = [42, 123, 456]

# ── Independent oracle: autograd VJP through the per-token recurrence (verbatim
#    from gated_delta_rule_bwd_golden_cpu.py) — the FP32 ground truth. ──

def _recurrent(q, k, v, beta, g, scale):
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                        for x in (q, k, v, beta, g)]
    B, H, T, K = k.shape
    Vv = v.shape[-1]
    o = torch.zeros(B, H, T, Vv, dtype=torch.float32)
    hh = torch.zeros(B, H, K, Vv, dtype=torch.float32)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]; b_k = k[:, :, i]; b_v = v[:, :, i].clone()
        hh = hh.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (hh.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        hh = hh.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = (b_q[..., None] * hh).sum(-2)
    return o.transpose(1, 2).contiguous()


def _bwd_oracle(grad_o, q, k, v, g, beta):
    qf = q.to(torch.float32).detach().requires_grad_(True)
    kf = k.to(torch.float32).detach().requires_grad_(True)
    vf = v.to(torch.float32).detach().requires_grad_(True)
    gf = g.to(torch.float32).detach().requires_grad_(True)
    bf = beta.to(torch.float32).detach().requires_grad_(True)
    do = grad_o.to(torch.float32)
    scale = 1.0 / (qf.shape[-1] ** 0.5)
    o = _recurrent(qf, kf, vf, torch.sigmoid(bf), gf, scale)
    dq, dk, dv, dg, db = torch.autograd.grad(o, [qf, kf, vf, gf, bf], grad_outputs=do)
    return (dq, dk, dv, dg, db)

def _validate():
    print("=" * 64)
    print("gated_delta_rule_bwd_golden_stage12345678910 (Module 10, FINAL) 验证 — vs autograd VJP oracle (golden_cpu math)")
    print("=" * 64)
    out_names = ("dq", "dk", "dv", "dg", "dbeta")
    all_pass = True
    global_worst = 0.0
    for (B, T, H, K, V) in _SHAPES:
        for seed in _SEEDS:
            grad_o, q, k, v, g, beta = _mk_case(B, T, H, K, V, seed)
            outs = gated_delta_rule_bwd_golden_stage12345678910(grad_o, q, k, v, g, beta)
            ref = _bwd_oracle(grad_o, q, k, v, g, beta)
            ok = True
            exp = {"dq": (B, T, H, K), "dk": (B, T, H, K), "dv": (B, T, H, V),
                   "dg": (B, T, H), "dbeta": (B, T, H)}
            line = []
            for nm, a, b_ in zip(out_names, outs, ref):
                if tuple(a.shape) != exp[nm]:
                    print(f"  [FAIL] {nm} shape {tuple(a.shape)} != {exp[nm]}"); ok = False
                if a.dtype != torch.float32:
                    print(f"  [FAIL] {nm} dtype {a.dtype} != fp32"); ok = False
                if not torch.isfinite(a).all():
                    print(f"  [FAIL] {nm} not finite"); ok = False
                d = (a - b_).abs()
                max_abs = d.max().item()
                peak = b_.abs().max().clamp_min(1e-6).item()
                rel_peak = max_abs / peak
                rel_l2 = (d.pow(2).sum().sqrt() / b_.pow(2).sum().sqrt().clamp_min(1e-12)).item()
                global_worst = max(global_worst, rel_l2)
                # identical math (chunk-parallel vs per-token) -> fp32 round-off; peak-scaled
                if not (rel_peak < 2e-3 or max_abs < 1e-3 or rel_l2 < 5e-4):
                    print(f"  [FAIL] {nm} max_abs={max_abs:.3e} rel_peak={rel_peak:.3e} "
                          f"rel_l2={rel_l2:.3e} peak={peak:.2e}"); ok = False
                line.append(f"{nm}(abs={max_abs:.2e},relL2={rel_l2:.2e},pk={peak:.1e})")
            all_pass = all_pass and ok
            print(f"  [{'PASS' if ok else 'FAIL'}] B{B}T{T}H{H}K{K}V{V} seed{seed}: " + " ".join(line))
    print("-" * 64)
    print(f"stage(final) vs golden_cpu(autograd VJP): global worst rel_l2 = {global_worst:.3e}")
    print("所有验证通过" if all_pass else "存在验证失败项")
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        raise SystemExit(1)
