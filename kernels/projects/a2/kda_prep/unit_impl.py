"""Standalone preparation execution and mandatory frozen-budget checks."""
import functools
import importlib.util
import json
from pathlib import Path

import torch

import reference as precision

make_inputs = precision.make_inputs
ROOT = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def _runtime():
    spec = importlib.util.spec_from_file_location('_a2_prep_unit_runtime', ROOT/'runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(inputs):
    return {name: value for name, value in precision.reference(inputs).items() if value.numel()}


def kernel_for(case):
    p = case['parameters']
    entries = _runtime().backward_kernels() if p.get('backward', False) else _runtime().kernels()
    return entries[p['kind']+'_'+p['types']]


def execute(inputs, options):
    from _unit_runner import launch_kernel

    if options['device'] != 'a2' or options['backend'] != 'cce' or options['block_dim'] not in (1, 2):
        raise ValueError('A2 preparation declares a2/CCE and block dimensions 1,2')
    p, v = inputs['parameters'], inputs['values']
    backward, kind = p.get('backward', False), p['kind']
    source, n = v['source'], v['source'].numel()
    dtypes = precision.output_dtypes(inputs)
    destination = torch.full((1, n), float('nan'), dtype=dtypes['destination'])
    entries = _runtime().backward_kernels() if backward else _runtime().kernels()
    entry = entries[kind+'_'+p['types']]
    args = (source.reshape(1, -1),)
    if backward:
        if kind == 'beta':
            args += (v['probability'].reshape(1, -1),)
        args += (v['sensitivity'].reshape(1, -1),)
    if kind == 'gate':
        hv, bt = source.shape[-2], source.shape[0]*source.shape[1]
        args += (v['alog'].reshape(1, -1), v['bias'].reshape(1, -1))
        if backward:
            count = hv*((bt+31)//32)*512
            partial = torch.full((1, count), float('nan'))
            actual, partial = launch_kernel(entry, args+(destination, partial, n, hv, hv*128, bt, count), options)
            da = torch.full((1, hv), float('nan'), dtype=dtypes['da'])
            db = torch.full((1, hv*128), float('nan'), dtype=dtypes['db'])
            reduce_entry = entries['gate_reduce_'+'_'.join(p['types'].split('_')[1:])]
            da, db = launch_kernel(reduce_entry, (partial, v['alog'].reshape(1, -1), da, db,
                                                  count, hv, hv*128, (bt+31)//32), options)
            result = {'destination': actual, 'da': da.reshape(-1), 'db': db.reshape(-1)}
        else:
            actual = launch_kernel(entry, args+(destination, n, hv, hv*128, bt), options)
            result = {'destination': actual}
    else:
        actual = launch_kernel(entry, args+(destination, n), options)
        result = {'destination': actual}
    checks = precision.compare_precision(inputs, result)
    if not all(row['passed'] for row in checks.values()):
        raise AssertionError(json.dumps({'frozen_precision_checks': checks}))
    return result


def reference_stages(inputs):
    p, v = inputs['parameters'], inputs['values']
    x = v['source'].double()
    hv, bt = x.shape[-2], x.shape[0]*x.shape[1]
    u = x+v['bias'].double().reshape(1, 1, hv, 128)
    gy = v['sensitivity'].double()
    soft = torch.nn.functional.softplus(u, threshold=20.)
    derivative = torch.where(u > 20., torch.ones_like(u), u.sigmoid())
    values_all = (gy*soft).reshape(bt, hv, 128)
    derivatives_all = (gy*derivative).reshape(bt, hv, 128)
    pieces = []
    for head in range(hv):
        for begin in range(0, bt, 32):
            values = values_all[begin:begin+32, head].sum(0)
            derivatives = derivatives_all[begin:begin+32, head].sum(0)
            pieces.append(torch.stack((values, derivatives)))
    return {'partial_sums': torch.stack(pieces).float()}


def execute_stages(inputs, options):
    from _unit_runner import launch_kernel

    p, v = inputs['parameters'], inputs['values']
    source = v['source']
    hv, bt, n = source.shape[-2], source.shape[0]*source.shape[1], source.numel()
    count = hv*((bt+31)//32)*512
    entry = _runtime().backward_kernels()['gate_'+p['types']]
    destination = torch.full((1, n), float('nan'), dtype=source.dtype)
    partial = torch.full((1, count), float('nan'))
    _, partial = launch_kernel(entry, (source.reshape(1, -1), v['sensitivity'].reshape(1, -1),
                                      v['alog'].reshape(1, -1), v['bias'].reshape(1, -1),
                                      destination, partial, n, hv, hv*128, bt, count), options)
    # Test observation only: compare the value represented by each high/low pair.
    pieces = partial.reshape(-1, 4, 128)
    return {'partial_sums': torch.stack((pieces[:, 0]+pieces[:, 1], pieces[:, 2]+pieces[:, 3]), dim=1)}
