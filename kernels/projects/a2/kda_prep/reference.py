"""Independent CPU references and frozen precision comparisons for A2 prep."""
from pathlib import Path
import json
import math

import torch

DTYPES = {'bf16': torch.bfloat16, 'f32': torch.float32}
ROOT = Path(__file__).resolve().parent


def make_inputs(case):
    p = dict(case['parameters'])
    types = p['types'].split('_')
    gen = torch.Generator().manual_seed(case['seed'])
    shape = tuple(p['shape'])
    x = torch.randn(shape, generator=gen).to(DTYPES[types[0]])
    if p['kind'] == 'gate':
        x = (x.float()*2-3).to(x.dtype)
    values = {'source': x}
    if p['kind'] == 'gate':
        hv = shape[-2]
        values['alog'] = torch.linspace(-.3, .3, hv).to(DTYPES[types[1]])
        values['bias'] = torch.linspace(-1.2, -.8, hv*128).to(DTYPES[types[2]])
    if p.get('backward', False):
        dtype = torch.bfloat16 if p['kind'] == 'norm' else torch.float32
        values['sensitivity'] = torch.randn(shape, generator=gen).to(dtype)
        if p['kind'] == 'beta':
            values['probability'] = x.float().sigmoid()
    return {'values': values, 'parameters': p}


def evaluate(inputs, precision=torch.float32):
    p, v = inputs['parameters'], inputs['values']
    kind, backward = p['kind'], p.get('backward', False)
    x = v['source'].to(precision)
    gy = v['sensitivity'].to(precision) if backward else None
    empty = torch.empty(0, dtype=precision)
    da = db = empty
    if kind == 'norm':
        denominator = (x.square().sum(-1, keepdim=True)+1e-6).sqrt()
        result = (gy/denominator - x*(x*gy).sum(-1, keepdim=True)/denominator.pow(3)
                  if backward else x/denominator)
        if backward:
            # The raw derivative retains the original FP32 forward overflow rule.
            overflow = torch.isinf(v['source'].float().square().sum(-1, keepdim=True)+1e-6)
            result = torch.where(overflow, torch.zeros_like(result), result)
    elif kind == 'gate':
        hv = x.shape[-2]
        decay = -v['alog'].to(precision).exp().view(1, 1, hv, 1)
        u = x + v['bias'].to(precision).reshape(1, 1, hv, 128)
        softplus = torch.nn.functional.softplus(u, threshold=20.)
        result = decay*softplus
        if backward:
            derivative = torch.where(u > 20., torch.ones_like(u), u.sigmoid())
            result = gy*decay*derivative
            da = (gy*decay*softplus).sum((0, 1, 3))
            db = result.sum((0, 1)).reshape(-1)
    else:
        result = x.sigmoid()
        if backward:
            exponential = (-x.abs()).exp()
            result = gy*exponential/(1+exponential).square()
            saturated = (v['probability'] == 0) | (v['probability'] == 1)
            result = torch.where(saturated, torch.zeros_like(result), result)
    return {'destination': result.reshape(1, -1), 'da': da, 'db': db}


def output_dtypes(inputs):
    p, v = inputs['parameters'], inputs['values']
    destination = (v['source'].dtype if p.get('backward', False) else
                   DTYPES[p['types'].split('_')[1]] if p['kind'] == 'norm' else torch.float32)
    gate_backward = p['kind'] == 'gate' and p.get('backward', False)
    return {'destination': destination,
            'da': v['alog'].dtype if gate_backward else torch.float32,
            'db': v['bias'].dtype if gate_backward else torch.float32}


def reference(inputs):
    types = output_dtypes(inputs)
    return {name: value.to(types[name]) for name, value in evaluate(inputs).items()}


def _metrics(actual, expected):
    a, e = actual.double(), expected.double()
    nonzero = e != 0
    difference = a-e
    norm = float(e.norm())
    relative_l2 = float(difference.norm())/norm if norm else (0. if bool((a == 0).all()) else math.inf)
    return {'finite': bool(torch.isfinite(a).all() and torch.isfinite(e).all()),
            'relative_l2': relative_l2,
            'max_relative_nonzero': float((difference[nonzero]/e[nonzero]).abs().max()) if bool(nonzero.any()) else 0.,
            'zero_mismatch': int((a[~nonzero] != 0).sum())}


def _ulp_bf16(actual, expected):
    def ordered(value):
        bits = value.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768-bits, bits)
    return int((ordered(actual)-ordered(expected)).abs().max())


def compare_precision(inputs, actual):
    """Keep the original per-dtype budgets in addition to protocol comparisons."""
    p = inputs['parameters']
    backward = p.get('backward', False)
    limits = json.loads((ROOT/('backward_budgets.json' if backward else 'budgets.json')).read_text())
    high, cpu = evaluate(inputs, torch.float64), evaluate(inputs, torch.float32)
    result = {}
    names = {'destination': 'dx' if p['kind'] == 'norm' else 'dg' if p['kind'] == 'gate' else 'dbeta',
             'da': 'dA_log', 'db': 'ddt_bias'}
    for name, value in actual.items():
        if value.numel() == 0:
            continue
        key = p['kind']+':'+p['types']
        if backward:
            key += ':'+names[name]
            budget = limits['groups'][key]
            l2, element = budget['relative_l2_limit'], budget['elementwise_relative_limit']
            ulp = budget['ulp_limit_each_reference']
        else:
            budget = limits['limits'][key]
            l2, element = budget['relative_l2'], budget['max_relative_nonzero']
            ulp = 1 if value.dtype == torch.bfloat16 else None
        metrics = _metrics(value, high[name])
        row = {'analytic_fp64': metrics, 'cpu_fp32': _metrics(value, cpu[name]), 'budget': budget}
        row['passed'] = (metrics['finite'] and metrics['relative_l2'] <= l2
                         and metrics['max_relative_nonzero'] <= element and metrics['zero_mismatch'] == 0)
        if ulp is not None:
            row['ulp_to_fp64'] = _ulp_bf16(value, high[name])
            row['ulp_to_fp32'] = _ulp_bf16(value, cpu[name])
            row['passed'] = row['passed'] and max(row['ulp_to_fp64'], row['ulp_to_fp32']) <= ulp
        result[name] = row
    return result
