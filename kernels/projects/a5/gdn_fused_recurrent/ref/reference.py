"""Independent grouped GDN recurrence; no oracle or device imports.

Public arithmetic is FP32. The optional FP64 mode is mathematical calibration
only. There is no implicit normalization or input mutation.
"""
from __future__ import annotations
import hashlib
import torch

SCALE = 128**-0.5
S_MAX = 16
NAMES = ('q', 'k', 'v', 'g', 'beta', 'initial_state')
OUTPUTS = ('o', 'final_state')


def reference(inputs, *, fp64=False):
    dtype = torch.float64 if fp64 else torch.float32
    q, k, v, g, beta = (inputs[n].to(dtype) for n in NAMES[:5])
    b, steps, h, kd = q.shape
    hv, vd = v.shape[2:]
    state0 = inputs.get('initial_state')
    state = (torch.zeros(b, hv, kd, vd, dtype=dtype) if state0 is None
             else state0.to(dtype).clone())
    # Enumerating each value head is independent of the production index code.
    index = torch.tensor([head // (hv // h) for head in range(hv)])
    result = []
    for t in range(steps):
        key = k[:, t].index_select(1, index).reshape(b*hv, 1, kd)
        query = (q[:, t].index_select(1, index)*kd**-0.5).reshape(b*hv, 1, kd)
        decay = g[:, t].exp().reshape(b*hv, 1, 1)
        decayed = state.reshape(b*hv, kd, vd)*decay
        residual = v[:, t].reshape(b*hv, 1, vd)-torch.bmm(key, decayed)
        update = residual*beta[:, t].reshape(b*hv, 1, 1)
        state = (decayed+torch.bmm(key.transpose(1, 2), update)).reshape(b, hv, kd, vd)
        result.append(torch.bmm(query, state.reshape(b*hv, kd, vd)).reshape(b, hv, vd))
    return dict(o=torch.stack(result, dim=1), final_state=state)


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
    # Freeze the complete grid before kernel authoring. The first two cases are
    # the full target workload in both storage dtypes, not smoke shapes.
    rows = []
    templates = [dict(id='full', B=1, S=16, H=16, HV=32, state='random')]
    for ratio in (1, 2, 4, 8):
        for steps in (1, 2, 3, 7, 15, 16):
            templates.append(dict(id=f'r{ratio}_s{steps}', B=1, S=steps, H=3,
                                  HV=3*ratio, state='random'))
    for tag, update in [
        ('zero_initial', dict(state='none')),
        ('explicit_zero', dict(state='zero')),
        ('batch2_s3', dict(B=2, S=3)),
        ('batch2_s16', dict(B=2, S=16)),
        ('many_heads', dict(B=3, H=3, HV=24, S=3)),
        ('g_zero', dict(g=0.)), ('g_small', dict(g=-.001)),
        ('g_large', dict(g=-30.)), ('g_underflow', dict(g=-1000.)),
        ('beta_zero', dict(beta=0.)), ('beta_one', dict(beta=1.)),
        ('zero_q', dict(zero_q=True)), ('zero_k', dict(zero_k=True)),
        ('all_zero', dict(all_zero=True)),
        ('state_large', dict(state_scale=16.)),
        ('state_small', dict(state_scale=1e-6)),
        ('key_norm_one', dict(key_norm=1.)),
        ('key_norm_two', dict(key_norm=2.)),
        ('signed_spikes', dict(spikes=True)),
    ]:
        row=dict(id=tag, B=2, S=16, H=2, HV=8, state='random')
        row.update(update); templates.append(row)
    for i, row in enumerate(templates):
        for dtype in ('float32', 'bfloat16'):
            rows.append(dict(row, id=row['id']+'_'+dtype, dtype=dtype, seed=940000+i))
    return rows


def make_inputs(case):
    p=case.get('parameters', case)
    rng=torch.Generator().manual_seed(case['seed'])
    b,s,h,hv=(p[n] for n in ('B','S','H','HV'))
    q=torch.rand(b,s,h,128,generator=rng)*2-1
    k=torch.randn(b,s,h,128,generator=rng)
    # Input generator only: this does not change the operator's raw-key ABI.
    k=k/k.norm(dim=-1,keepdim=True)*p.get('key_norm',.75)
    v=torch.rand(b,s,hv,128,generator=rng)*2-1
    g=-torch.rand(b,s,hv,generator=rng)*.25
    beta=torch.rand(b,s,hv,generator=rng)
    initial=(torch.rand(b,hv,128,128,generator=rng)*2-1)*p.get('state_scale',.25)
    if p.get('zero_q'):q.zero_()
    if p.get('zero_k'):k.zero_()
    if p.get('spikes'):
        q.zero_();k.zero_();q[...,7]=-.75;k[...,71]=.5
        initial[...,71,7]=3.;v[...,7]=-2.
    if 'g' in p:g.fill_(p['g'])
    if 'beta' in p:beta.fill_(p['beta'])
    if p.get('all_zero'):
        for x in (q,k,v,g,beta,initial):x.zero_()
    if p.get('state')=='zero':initial.zero_()
    dtype=getattr(torch,p.get('dtype','float32'))
    return dict(q=q.to(dtype), k=k.to(dtype), v=v.to(dtype), g=g, beta=beta,
                initial_state=None if p.get('state')=='none' else initial)


def validate_inputs(inputs, case=None):
    if set(inputs)!=set(NAMES):raise ValueError('expected q/k/v/g/beta/initial_state')
    q,v=inputs['q'],inputs['v']
    if q.ndim!=4 or q.shape[-1]!=128 or min(q.shape[:3])<1 or q.shape[1]>S_MAX:
        raise ValueError('positive B/H and S=1..16, K=128 required')
    b,s,h,_=q.shape
    if v.ndim!=4 or v.shape[:2]!=(b,s) or v.shape[-1]!=128 or v.shape[2]<1 or v.shape[2]%h:
        raise ValueError('v shape or contiguous head group is invalid')
    for n,x in inputs.items():
        if n=='initial_state' and x is None:continue
        shape=((b,v.shape[2],128,128) if n=='initial_state' else
               v.shape[:3] if n in ('g','beta') else v.shape if n=='v' else q.shape)
        dtype=torch.float32 if n in ('g','beta','initial_state') else q.dtype
        if x.shape!=shape or x.dtype!=dtype or q.dtype not in (torch.float32,torch.bfloat16):
            raise ValueError(f'{n}: invalid shape/dtype')
        if x.device.type!='cpu' or not x.is_contiguous() or not bool(torch.isfinite(x).all()):
            raise ValueError(f'{n}: finite contiguous CPU reference tensor required')
    if bool((inputs['g']>0).any()) or bool(((inputs['beta']<0)|(inputs['beta']>1)).any()):
        raise ValueError('g<=0 and beta in[0,1] required')


def validate_reference(inputs, outputs, case=None):
    b,s,hv,_=inputs['v'].shape
    if set(outputs)!=set(OUTPUTS):raise ValueError('missing or extra output')
    for n,shape in [('o',(b,s,hv,128)),('final_state',(b,hv,128,128))]:
        x=outputs[n]
        if x.shape!=shape or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):
            raise ValueError(f'invalid reference {n}')
        if float(x.norm())>0:
            for wrong in (torch.zeros_like(x),-x,x*1.25):
                assert metric(wrong,x)['relative_l2']>1e-4
        bad=x.clone();bad.reshape(-1)[0]=float('nan')
        assert not metric(bad,x)['finite']
