"""Independent PGDN primal and analytical VJP, with bounded replay storage.

No FLA import, autograd differentiation, device kernel or sibling-unit import.
FP32 is B. FP64 supplies mathematical checks and the frozen classifier's local
terms; classification follows the literal FP32 normalization branch.
"""
import torch

INPUTS = ('q', 'k', 'v', 'g_atk', 'g', 'beta_atk', 'beta')
NAMES = ('dq', 'dk', 'dv', 'dg_atk', 'dg', 'dbeta_atk', 'dbeta')
EPS_NORM = 1e-12
CHUNK = 64


def group_sum(value, heads):
    """Ascending consecutive value heads, one addition per contributor."""
    B, HV = value.shape[:2]
    ratio = HV // heads
    parts = value.reshape(B, heads, ratio, *value.shape[2:])
    result = torch.zeros_like(parts[:, :, 0])
    for index in range(ratio):
        result = result + parts[:, :, index]
    return result


def normalization(raw, *, fp32_branch=False):
    norm = raw.norm(dim=-1, keepdim=True)
    if fp32_branch:
        # Select the literal FP32 branch before lifting arithmetic to FP64.
        branch = raw.float().norm(dim=-1, keepdim=True) >= EPS_NORM
        epsilon = torch.tensor(EPS_NORM, dtype=torch.float32).double().item()
    else:
        branch = norm >= EPS_NORM
        epsilon = EPS_NORM
    denominator = torch.where(branch, norm, torch.full_like(norm, epsilon))
    return raw / denominator, norm, denominator, branch


def norm_vjp(raw, grad, normalized):
    y, norm, denominator, branch = normalized
    radial = y * (y * grad).sum(-1, keepdim=True)
    radial = torch.where(branch, radial, torch.zeros_like(radial))
    result = (grad - radial) / denominator
    local_scale = (grad.abs() + radial.abs()) / denominator
    # An axis-aligned input has an algebraically zero radial derivative in
    # the selected norm branch, regardless of the incoming cotangent.
    structural = branch & ((raw != 0).sum(-1, keepdim=True) == 1) & (raw != 0)
    return result, local_scale, structural


def prepare(q, k, v, g_atk, g, beta_atk, beta, *, fp32_branch=False):
    qnorm = normalization(q, fp32_branch=fp32_branch)
    knorm = normalization(k, fp32_branch=fp32_branch)
    B, T, H, K = k.shape
    state = k.new_zeros(B, H, K)
    history, writes = [], []
    logx = k.new_tensor(1.5).log()
    for t in range(T):
        state = state * g_atk[:, t].exp()[..., None] + beta_atk[:, t, :, None] * knorm[0][:, t].square()
        r = (state + 1e-6).log() + .2
        multiplier = (-logx * r / (1 + r.abs())).exp()
        history.append(state)
        writes.append(knorm[0][:, t] * multiplier)
    return dict(qnorm=qnorm, knorm=knorm, A=torch.stack(history, 1), write=torch.stack(writes, 1), logx=logx)


def main_step(state, t, read, write, v, g, beta):
    decayed = g[:, t].exp()[..., None, None] * state
    residual = v[:, t] - (read[:, t, :, :, None] * decayed).sum(-2)
    update = beta[:, t, :, None] * residual
    after = decayed + write[:, t, :, :, None] * update[..., None, :]
    return decayed, residual, update, after


def forward(q, k, v, g_atk, g, beta_atk, beta):
    saved = prepare(q, k, v, g_atk, g, beta_atk, beta)
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    ratio = HV // H
    read = saved['knorm'][0].repeat_interleave(ratio, 2)
    write = saved['write'].repeat_interleave(ratio, 2)
    query = saved['qnorm'][0].repeat_interleave(ratio, 2) * K**-.5
    state = q.new_zeros(B, HV, K, V)
    outputs = []
    for t in range(T):
        state = main_step(state, t, read, write, v, g, beta)[-1]
        outputs.append((query[:, t, :, :, None] * state).sum(-2))
    return torch.stack(outputs, 1), state, saved['A'][:, -1]


@torch.no_grad()
def analytical(q, k, v, g_atk, g, beta_atk, beta, do=None, dht=None, dA_T=None,
               *, fp32_branch=False, auxiliary=False):
    """Seven gradients; auxiliary scales are computed without any candidate."""
    xs = (q, k, v, g_atk, g, beta_atk, beta)
    saved = prepare(*xs, fp32_branch=fp32_branch)
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    ratio = HV // H
    read = saved['knorm'][0].repeat_interleave(ratio, 2)
    write = saved['write'].repeat_interleave(ratio, 2)
    query = saved['qnorm'][0].repeat_interleave(ratio, 2)
    state = q.new_zeros(B, HV, K, V)
    boundaries = []
    for t in range(T):
        if t % CHUNK == 0:
            boundaries.append(state)
        state = main_step(state, t, read, write, v, g, beta)[-1]
    G = torch.zeros_like(state) if dht is None else dht.clone()
    Z = k.new_zeros(B, H, K) if dA_T is None else dA_T.clone()
    cotangent = torch.zeros_like(v) if do is None else do
    grads = {n: torch.empty_like(x) for n, x in zip(NAMES, xs)}
    scales = {n: torch.empty_like(x) for n, x in zip(NAMES, xs)} if auxiliary else None
    zeros = {n: torch.zeros_like(x, dtype=torch.uint8) for n, x in zip(NAMES, xs)} if auxiliary else None
    for chunk in reversed(range((T + CHUNK - 1) // CHUNK)):
        begin, end = chunk * CHUNK, min((chunk + 1) * CHUNK, T)
        state = boundaries[chunk]
        history = []
        for t in range(begin, end):
            values = main_step(state, t, read, write, v, g, beta)
            state = values[-1]
            history.append(values)
        for t in reversed(range(begin, end)):
            D, residual, update, state = history[t - begin]
            dout = cotangent[:, t]
            dq_normal = group_sum(K**-.5 * (state * dout[..., None, :]).sum(-1), H)
            G = G + (K**-.5 * query[:, t, :, :, None]) * dout[..., None, :]
            dw = group_sum((G * update[..., None, :]).sum(-1), H)
            dz_terms = G * write[:, t, :, :, None]
            dz = dz_terms.sum(-2)
            dr = beta[:, t, :, None] * dz
            dk_read = -group_sum((D * dr[..., None, :]).sum(-1), H)
            grads['dv'][:, t] = dr
            beta_terms = dz * residual
            grads['dbeta'][:, t] = beta_terms.sum(-1)
            GD = G - read[:, t, :, :, None] * dr[..., None, :]
            gate_terms = GD * D
            grads['dg'][:, t] = gate_terms.sum((-1, -2))
            G = g[:, t].exp()[..., None, None] * GD

            A = saved['A'][:, t]
            previous_A = torch.zeros_like(A) if t == 0 else saved['A'][:, t - 1]
            u = saved['knorm'][0][:, t]
            r = (A + 1e-6).log() + .2
            M = (-saved['logx'] * r / (1 + r.abs())).exp()
            Z = Z + (-saved['logx'] * M / (1 + r.abs()).square()) * (dw * u) / (A + 1e-6)
            dk_normal = dk_read + dw * M + 2 * beta_atk[:, t, :, None] * u * Z
            atk_gate_terms = Z * (g_atk[:, t].exp()[..., None] * previous_A)
            atk_beta_terms = Z * u.square()
            grads['dg_atk'][:, t] = atk_gate_terms.sum(-1)
            grads['dbeta_atk'][:, t] = atk_beta_terms.sum(-1)
            Z = Z * g_atk[:, t].exp()[..., None]
            qresult = norm_vjp(q[:, t], dq_normal, tuple(x[:, t] for x in saved['qnorm']))
            kresult = norm_vjp(k[:, t], dk_normal, tuple(x[:, t] for x in saved['knorm']))
            grads['dq'][:, t], grads['dk'][:, t] = qresult[0], kresult[0]
            if auxiliary:
                scales['dq'][:, t], scales['dk'][:, t] = qresult[1], kresult[1]
                scales['dv'][:, t] = (beta[:, t, :, None, None] * dz_terms).abs().sum(-2)
                scales['dbeta'][:, t] = beta_terms.abs().sum(-1)
                scales['dg'][:, t] = gate_terms.abs().sum((-1, -2))
                scales['dg_atk'][:, t] = atk_gate_terms.abs().sum(-1)
                scales['dbeta_atk'][:, t] = atk_beta_terms.abs().sum(-1)
                zeros['dq'][:, t] = qresult[2].to(torch.uint8)
                zeros['dk'][:, t] = kresult[2].to(torch.uint8)
    if not auxiliary:
        return grads
    # Reason codes: 1 normalization radial nullspace; 2 zero initial state;
    # 3 no main cotangent; 4 no output cotangent. Based solely on inputs/ABI.
    zeros['dg'][:, 0] = 2
    zeros['dg_atk'][:, 0] = 2
    if do is None:
        zeros['dq'].fill_(4)
    if do is None and dht is None:
        for name in ('dq', 'dv', 'dg', 'dbeta'):
            zeros[name].fill_(3)
    raw = {n: g.clone() for n, g in grads.items()}
    # Structural zeros are algebraic truths; retain their raw FP64 residuals.
    grads = {n: torch.where(zeros[n] != 0, torch.zeros_like(g), g) for n, g in grads.items()}
    return dict(gradients=grads, raw_gradients=raw, scales=scales, zero_reasons=zeros)
