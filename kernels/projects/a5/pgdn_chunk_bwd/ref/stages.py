"""Independent stage boundaries; matrix contractions differ from the VF loop."""
import torch

from .reference import CHUNK, INPUTS, group_sum, norm_vjp, prepare


@torch.no_grad()
def reference_stages(inputs, *, observe=None):
    q, k, v, ga, g, ba, beta = (inputs[n] for n in INPUTS)
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    ratio = HV // H
    saved = prepare(q, k, v, ga, g, ba, beta)
    result = dict(q_norm=saved['qnorm'][0], k_read=saved['knorm'][0], k_write=saved['write'],
                  A_history=saved['A'], q_raw_norm=saved['qnorm'][1].squeeze(-1),
                  k_raw_norm=saved['knorm'][1].squeeze(-1), final_A_state=saved['A'][:, -1].clone())
    read = result['k_read'].repeat_interleave(ratio, 2)
    write = result['k_write'].repeat_interleave(ratio, 2)
    query = result['q_norm'].repeat_interleave(ratio, 2)
    state = q.new_zeros(B, HV, K, V)
    checkpoints = q.new_empty(B, T // CHUNK, HV, K, V)
    tape = q.new_empty(B, HV, CHUNK, K, V)

    def step(previous, t):
        decayed = g[:, t].exp()[..., None, None] * previous
        residual = v[:, t] - torch.matmul(read[:, t].unsqueeze(-2), decayed).squeeze(-2)
        update = beta[:, t, :, None] * residual
        after = decayed + write[:, t, :, :, None] * update[..., None, :]
        return decayed, residual, update, after

    for t in range(T):
        if t % CHUNK == 0:
            checkpoints[:, t // CHUNK] = state
        if t < CHUNK:
            tape[:, :, t] = state
        state = step(state, t)[-1]
    result.update(checkpoints=checkpoints, final_state=state, tape=tape)
    if observe is not None:
        for name, value in result.items():
            observe(name, value)
    do, dht, dA = (inputs.get(n) for n in ('do', 'dht', 'dA_T'))
    dout = torch.zeros_like(v) if do is None else do
    G = torch.zeros_like(state) if dht is None else dht.clone()
    for name in ('dq_norm_parts', 'dk_read_parts', 'dk_write_parts'):
        result[name] = q.new_empty(B, T, HV, K)
    result.update(dv=torch.empty_like(v), dg=torch.empty_like(g), dbeta=torch.empty_like(beta))
    for chunk in reversed(range(T // CHUNK)):
        previous = checkpoints[:, chunk]
        history = []
        for t in range(chunk * CHUNK, (chunk + 1) * CHUNK):
            values = step(previous, t)
            previous = values[-1]
            history.append(values)
        for t in reversed(range(chunk * CHUNK, (chunk + 1) * CHUNK)):
            D, residual, update, state = history[t - chunk * CHUNK]
            dot = dout[:, t]
            result['dq_norm_parts'][:, t] = K**-.5 * torch.matmul(state, dot.unsqueeze(-1)).squeeze(-1)
            G = G + (K**-.5 * query[:, t, :, :, None]) * dot[..., None, :]
            result['dk_write_parts'][:, t] = torch.matmul(G, update.unsqueeze(-1)).squeeze(-1)
            dz = torch.matmul(write[:, t].unsqueeze(-2), G).squeeze(-2)
            dr = beta[:, t, :, None] * dz
            result['dk_read_parts'][:, t] = -torch.matmul(D, dr.unsqueeze(-1)).squeeze(-1)
            result['dv'][:, t] = dr
            result['dbeta'][:, t] = (dz * residual).sum(-1)
            GD = G - read[:, t, :, :, None] * dr[..., None, :]
            result['dg'][:, t] = (GD * D).sum((-1, -2))
            if observe is not None:
                for name, value in (('D', D), ('residual', residual), ('update', update),
                                    ('main_adjoint_with_output', G), ('dz', dz), ('GD', GD)):
                    observe(name, value)
            G = g[:, t].exp()[..., None, None] * GD
    result.update(dq=torch.empty_like(q), dk=torch.empty_like(k),
                  dg_atk=torch.empty_like(ga), dbeta_atk=torch.empty_like(ba))
    Z = k.new_zeros(B, H, K) if dA is None else dA.clone()
    for t in reversed(range(T)):
        qgrad = group_sum(result['dq_norm_parts'][:, t], H)
        rgrad = group_sum(result['dk_read_parts'][:, t], H)
        wgrad = group_sum(result['dk_write_parts'][:, t], H)
        u = result['k_read'][:, t]
        A = result['A_history'][:, t]
        previous_A = torch.zeros_like(A) if t == 0 else result['A_history'][:, t-1]
        r = (A + 1e-6).log() + .2
        M = (-saved['logx'] * r / (1 + r.abs())).exp()
        Z = Z + (-saved['logx'] * M / (1 + r.abs()).square()) * (wgrad * u) / (A + 1e-6)
        kgrad = rgrad + wgrad * M + 2 * ba[:, t, :, None] * u * Z
        result['dg_atk'][:, t] = (Z * (ga[:, t].exp()[..., None] * previous_A)).sum(-1)
        result['dbeta_atk'][:, t] = (Z * u.square()).sum(-1)
        if observe is not None:
            for name, value in (('squash_r', r), ('M', M), ('ATK_adjoint', Z),
                                ('dq_before_normalization', qgrad), ('dk_before_normalization', kgrad)):
                observe(name, value)
        Z = Z * ga[:, t].exp()[..., None]
        result['dq'][:, t] = norm_vjp(q[:, t], qgrad, tuple(x[:, t] for x in saved['qnorm']))[0]
        result['dk'][:, t] = norm_vjp(k[:, t], kgrad, tuple(x[:, t] for x in saved['knorm']))[0]
    if observe is not None:
        for name, value in result.items():
            observe(name, value)
        observe('q_norm_jacobian_bound', saved['qnorm'][2].reciprocal())
        observe('k_norm_jacobian_bound', saved['knorm'][2].reciprocal())
    return result
