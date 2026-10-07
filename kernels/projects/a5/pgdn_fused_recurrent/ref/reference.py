"""Independent H-owned ATK and grouped main recurrence; no oracle imports."""
from __future__ import annotations
import hashlib
import torch

SCALE = 128**-.5
S_MAX = 16
NAMES = ('q', 'k', 'v', 'g_atk', 'g', 'beta_atk', 'beta', 'initial_state', 'initial_A_state')
OUTPUTS = ('o', 'final_state', 'final_A_state')


def reference(inputs, *, fp64=False):
    dtype = torch.float64 if fp64 else torch.float32
    q, k, v, ga, g, ba, beta = (inputs[n].to(dtype) for n in NAMES[:7])
    b, steps, h, kd = q.shape
    hv, vd = v.shape[2:]
    state0, a0 = (inputs.get(n) for n in NAMES[-2:])
    state = torch.zeros(b, hv, kd, vd, dtype=dtype) if state0 is None else state0.to(dtype).clone()
    atk = torch.zeros(b, h, kd, dtype=dtype) if a0 is None else a0.to(dtype).clone()
    index = torch.tensor([head // (hv // h) for head in range(hv)])
    result = []
    logx = torch.tensor(1.5, dtype=dtype).log()
    for t in range(steps):
        rawq, rawk = q[:, t], k[:, t]
        query = rawq / torch.linalg.vector_norm(rawq, dim=-1, keepdim=True).clamp_min(1e-12)
        key = rawk / torch.linalg.vector_norm(rawk, dim=-1, keepdim=True).clamp_min(1e-12)
        atk = atk * ga[:, t].exp().unsqueeze(-1) + key.square() * ba[:, t].unsqueeze(-1)
        centered = (atk + 1e-6).log() + .2
        preconditioner = (-logx * (centered / (1 + centered.abs()))).exp()
        read = key.index_select(1, index).reshape(b*hv, 1, kd)
        write = (key * preconditioner).index_select(1, index).reshape(b*hv, kd, 1)
        query = (query * kd**-.5).index_select(1, index).reshape(b*hv, 1, kd)
        decayed = state.reshape(b*hv, kd, vd) * g[:, t].exp().reshape(b*hv, 1, 1)
        innovation = (v[:, t].reshape(b*hv, 1, vd) - torch.bmm(read, decayed)) * beta[:, t].reshape(b*hv, 1, 1)
        state = (decayed + torch.bmm(write, innovation)).reshape(b, hv, kd, vd)
        result.append(torch.bmm(query, state.reshape(b*hv, kd, vd)).reshape(b, hv, vd))
    return dict(o=torch.stack(result, dim=1), final_state=state, final_A_state=atk)


def metric(actual, expected):
    a, e = actual.double(), expected.double()
    diff = a-e
    norm, error = float(e.norm()), float(diff.norm())
    return dict(relative_l2=error/norm if norm else (0. if error == 0 else float('inf')),
                max_abs=float(diff.abs().max()), finite=bool(torch.isfinite(a).all()),
                reference_norm=norm, error_norm=error)


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def cases():
    templates = [dict(id='full', B=1, S=16, H=16, HV=32, state='random', atk='random')]
    for ratio in (1, 2, 4, 8):
        for steps in (1, 2, 3, 7, 15, 16):
            templates.append(dict(id=f'r{ratio}_s{steps}', B=1, S=steps, H=3, HV=3*ratio, state='random', atk='random'))
    for state in ('none', 'zero', 'random'):
        for atk in ('none', 'zero', 'random'):
            templates.append(dict(id=f'initial_{state}_{atk}', B=2, S=3, H=2, HV=8, state=state, atk=atk))
    for tag, update in [
        ('many_heads', dict(B=3,H=28,HV=224,S=3)),
        ('g_zero', dict(g=0.)), ('g_large', dict(g=-30.)), ('g_underflow', dict(g=-1000.)),
        ('ga_zero', dict(g_atk=0.)), ('ga_large', dict(g_atk=-30.)), ('ga_underflow', dict(g_atk=-1000.)),
        ('beta_zero', dict(beta=0.)), ('beta_one', dict(beta=1.)),
        ('ba_zero', dict(beta_atk=0.)), ('ba_one', dict(beta_atk=1.)),
        ('zero_q', dict(zero_q=True)), ('zero_k', dict(zero_k=True)), ('all_zero', dict(all_zero=True)),
        ('norm_below_clamp', dict(qk_scale=1e-14)), ('norm_above_clamp', dict(qk_scale=1e-11)),
        ('state_large', dict(state_scale=16.)), ('state_small', dict(state_scale=1e-6)),
        ('atk_large', dict(atk_scale=1e30)), ('atk_small', dict(atk_scale=1e-30)),
        ('atk_tiny_decay_large_initial', dict(S=1,atk_scale=1e30,g_atk=-100.,beta_atk=0.)),
        ('signed_spikes', dict(spikes=True)),
    ]:
        row = dict(id=tag,B=2,S=16,H=2,HV=8,state='random',atk='random')
        row.update(update); templates.append(row)
    for ga in (-87.3,-87.34,-88.,-90.,-95.,-103.,-104.):
        templates.append(dict(id=f'atk_decay_{abs(ga):g}',B=2,S=1,H=2,HV=8,state='random',atk='random',atk_scale=1e30,g_atk=ga,beta_atk=0.))
    return [dict(row,id=row['id']+'_'+dtype,dtype=dtype,seed=950000+i)
            for i,row in enumerate(templates) for dtype in ('float32','bfloat16')]


def make_inputs(case):
    p = case.get('parameters',case)
    rng = torch.Generator().manual_seed(case['seed'])
    b,s,h,hv = (p[n] for n in ('B','S','H','HV'))
    q = torch.randn(b,s,h,128,generator=rng) * p.get('qk_scale',1.)
    k = torch.randn(b,s,h,128,generator=rng) * p.get('qk_scale',1.)
    v = torch.rand(b,s,hv,128,generator=rng)*2-1
    g = -torch.rand(b,s,hv,generator=rng)*.25
    ga = -torch.rand(b,s,h,generator=rng)*.25
    beta = torch.rand(b,s,hv,generator=rng)
    ba = torch.rand(b,s,h,generator=rng)
    initial = (torch.rand(b,hv,128,128,generator=rng)*2-1)*p.get('state_scale',.25)
    atk = torch.rand(b,h,128,generator=rng)*p.get('atk_scale',.5)
    if p.get('zero_q'):q.zero_()
    if p.get('zero_k'):k.zero_()
    if p.get('spikes'):
        q.zero_();k.zero_();q[...,7]=-.75;k[...,71]=.5
        initial[...,71,7]=3.;v[...,7]=-2.;atk[...,71]=7.
    for name,x in [('g',g),('g_atk',ga),('beta',beta),('beta_atk',ba)]:
        if name in p:x.fill_(p[name])
    if p.get('all_zero'):
        for x in (q,k,v,g,ga,beta,ba,initial,atk):x.zero_()
    if p.get('state')=='zero':initial.zero_()
    if p.get('atk')=='zero':atk.zero_()
    dtype = getattr(torch,p.get('dtype','float32'))
    return dict(q=q.to(dtype),k=k.to(dtype),v=v.to(dtype),g_atk=ga,g=g,beta_atk=ba,beta=beta,
                initial_state=None if p.get('state')=='none' else initial,
                initial_A_state=None if p.get('atk')=='none' else atk)


def validate_inputs(inputs, case=None):
    if set(inputs)!=set(NAMES):raise ValueError('incorrect PGDN input names')
    q,v = inputs['q'],inputs['v']
    if q.ndim!=4 or q.shape[-1]!=128 or min(q.shape[:3])<1 or q.shape[1]>S_MAX:
        raise ValueError('positive B/H and S=1..16, K=128 required')
    b,s,h,_ = q.shape
    if v.ndim!=4 or v.shape[:2]!=(b,s) or v.shape[-1]!=128 or v.shape[2]<1 or v.shape[2]%h:
        raise ValueError('invalid grouped v shape')
    for n,x in inputs.items():
        if n in NAMES[-2:] and x is None:continue
        shape = ((b,v.shape[2],128,128) if n=='initial_state' else (b,h,128) if n=='initial_A_state' else
                 q.shape[:3] if n in ('g_atk','beta_atk') else v.shape[:3] if n in ('g','beta') else v.shape if n=='v' else q.shape)
        dtype = q.dtype if n in NAMES[:3] else torch.float32
        if x.shape!=shape or x.dtype!=dtype or q.dtype not in (torch.float32,torch.bfloat16):
            raise ValueError(f'{n}: invalid shape/dtype')
        if x.device.type!='cpu' or not x.is_contiguous() or not bool(torch.isfinite(x).all()):
            raise ValueError(f'{n}: finite contiguous CPU reference tensor required')
    for n in ('g','g_atk'):
        if bool((inputs[n]>0).any()):raise ValueError(f'{n} must be <=0')
    for n in ('beta','beta_atk'):
        if bool(((inputs[n]<0)|(inputs[n]>1)).any()):raise ValueError(f'{n} must be in[0,1]')
    if inputs['initial_A_state'] is not None and bool((inputs['initial_A_state']<0).any()):
        raise ValueError('initial_A_state must be nonnegative')


def validate_reference(inputs, outputs, case=None):
    b,s,hv,_ = inputs['v'].shape
    h = inputs['q'].shape[2]
    if set(outputs)!=set(OUTPUTS):raise ValueError('missing or extra output')
    for n,shape in [('o',(b,s,hv,128)),('final_state',(b,hv,128,128)),('final_A_state',(b,h,128))]:
        x = outputs[n]
        if x.shape!=shape or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):
            raise ValueError(f'invalid reference {n}')
        if float(x.double().norm())>0:
            for wrong in (torch.zeros_like(x),-x,x*1.25):
                assert metric(wrong,x)['relative_l2']>1e-4
        bad=x.clone();bad.reshape(-1)[0]=float('nan')
        assert not metric(bad,x)['finite']
