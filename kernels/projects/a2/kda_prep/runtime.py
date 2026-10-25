"""Typed preparation ABI: metadata, output allocation, and native launches."""
from __future__ import annotations

import functools
import importlib.util
import itertools
from pathlib import Path

import torch

_DTYPES = {torch.bfloat16: 'bf16', torch.float32: 'f32'}
_BLOCK_DIMS = {'chunk': (1, 2), 'decode': (1, 2)}


@functools.lru_cache(maxsize=2)
def kernels(namespace='chunk'):
    if namespace not in _BLOCK_DIMS:
        raise ValueError('KDA prep namespace must be chunk or decode')
    from ascriptor.a2 import bf16, f32

    path = Path(__file__).parent / 'kernels' / 'forward.py'
    spec = importlib.util.spec_from_file_location('_afla_kda_prep_a2_forward', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    types = {'bf16': bf16, 'f32': f32}
    result = {}
    for kind, factory, count in (('norm', module.make_norm, 2),
                                 ('gate', module.make_gate, 3),
                                 ('beta', module.make_beta, 1)):
        for labels in itertools.product(types, repeat=count):
            key = kind + '_' + '_'.join(labels)
            name = f'kda_prep_a2_{namespace}_{key}_kernel'
            result[key] = factory(name, *(types[label] for label in labels))
    return result


@functools.lru_cache(maxsize=None)
def prepare(device='a2', block_dim=1, namespace='chunk'):
    """Register every dtype vendor before the process executes a custom op."""
    from ascend_fla.runtime.compile import compile_kernel

    if device != 'a2' or namespace not in _BLOCK_DIMS or block_dim not in _BLOCK_DIMS[namespace]:
        raise ValueError('KDA prep requires a2 and a supported chunk/decode block_dim')
    return {name: compile_kernel(fn, device=device, block_dim=block_dim)
            for name, fn in kernels(namespace).items()}


def _check_source(name, source):
    if not isinstance(source, torch.Tensor) or source.dtype not in _DTYPES:
        raise ValueError(f'{name} raw input must be BF16 or FP32; FP16/FP64 are unsupported')
    if source.device.type != 'npu' or not source.is_contiguous():
        raise ValueError(f'{name} raw input must be a contiguous NPU tensor')
    if source.numel() <= 0 or source.numel() >= 2**31:
        raise ValueError(f'{name} raw input size must be positive and fit signed int32')


def norm(source, dtype, *, device='a2', block_dim=1, namespace='chunk'):
    _check_source('q/k', source)
    if dtype not in _DTYPES or source.shape[-1] != 128:
        raise ValueError('KDA norm requires K=128 and BF16/FP32 output')
    destination = torch.empty(source.shape, dtype=dtype, device=source.device)
    key = f'norm_{_DTYPES[source.dtype]}_{_DTYPES[dtype]}'
    prepare(device, block_dim, namespace)[key](
        {'source': source.view(1, -1)}, {'N': source.numel()},
        {'destination': destination.view(1, -1)})
    return destination


def gate(source, alog, bias, *, device='a2', block_dim=1, namespace='chunk'):
    for name, value in (('g', source), ('A_log', alog), ('dt_bias', bias)):
        _check_source(name, value)
    if source.dim() != 4 or source.shape[-1] != 128:
        raise ValueError('KDA gate requires [B,T,HV,128]')
    hv = source.shape[-2]
    if alog.shape != (hv,) or bias.shape != (hv * 128,):
        raise ValueError('KDA gate requires A_log[HV] and dt_bias[HV*128]')
    if alog.device != source.device or bias.device != source.device:
        raise ValueError('KDA gate parameters must share the input device')
    destination = torch.empty(source.shape, dtype=torch.float32, device=source.device)
    key = f'gate_{_DTYPES[source.dtype]}_{_DTYPES[alog.dtype]}_{_DTYPES[bias.dtype]}'
    prepare(device, block_dim, namespace)[key](
        {'source': source.view(1, -1), 'alog': alog.view(1, -1), 'bias': bias.view(1, -1)},
        dict(N=source.numel(), HV=hv, Channels=hv*128, BT=source.shape[0]*source.shape[1]),
        {'destination': destination.view(1, -1)})
    return destination


def beta(source, *, device='a2', block_dim=1, namespace='chunk'):
    _check_source('beta', source)
    destination = torch.empty(source.shape, dtype=torch.float32, device=source.device)
    prepare(device, block_dim, namespace)[f'beta_{_DTYPES[source.dtype]}'](
        {'source': source.view(1, -1)}, {'N': source.numel()},
        {'destination': destination.view(1, -1)})
    return destination


@functools.lru_cache(maxsize=1)
def backward_kernels():
    from ascriptor.a2 import bf16, f32

    path = Path(__file__).parent / 'kernels' / 'backward.py'
    spec = importlib.util.spec_from_file_location('_afla_kda_prep_a2_backward', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    types = {'bf16': bf16, 'f32': f32}
    result = {}
    for kind, factory, count in (
        ('norm', module.make_norm_backward, 1),
        ('gate', module.make_gate_backward, 3),
        ('gate_reduce', module.make_gate_reduce, 2),
        ('beta', module.make_beta_backward, 1),
    ):
        for labels in itertools.product(types, repeat=count):
            key = kind + '_' + '_'.join(labels)
            result[key] = factory(f'kda_prep_a2_backward_{key}_kernel', *(types[label] for label in labels))
    return result


@functools.lru_cache(maxsize=None)
def prepare_backward(device='a2', block_dim=1):
    """All raw-gradient vendors must be ready before the first custom launch."""
    from ascend_fla.runtime.compile import compile_kernel

    if device != 'a2' or block_dim not in _BLOCK_DIMS['chunk']:
        raise ValueError('KDA prep backward requires a2 and chunk block_dim 1,2')
    return {name: compile_kernel(fn, device=device, block_dim=block_dim)
            for name, fn in backward_kernels().items()}


def _check_sensitivity(source, sensitivity, dtype):
    _check_source('sensitivity', sensitivity)
    if sensitivity.dtype != dtype or sensitivity.shape != source.shape or sensitivity.device != source.device:
        raise ValueError('KDA prep sensitivity must match source shape/device and prepared dtype')


def norm_backward(source, sensitivity, *, device='a2', block_dim=1):
    _check_source('q/k', source)
    _check_sensitivity(source, sensitivity, torch.bfloat16)
    if source.shape[-1] != 128:
        raise ValueError('KDA norm backward requires K=128')
    destination = torch.empty(source.shape, dtype=source.dtype, device=source.device)
    prepare_backward(device, block_dim)[f'norm_{_DTYPES[source.dtype]}'](
        {'source': source.view(1, -1), 'sensitivity': sensitivity.view(1, -1)},
        {'N': source.numel()}, {'destination': destination.view(1, -1)})
    return destination


def gate_backward(source, alog, bias, sensitivity, *, device='a2', block_dim=1):
    for name, value in (('g', source), ('A_log', alog), ('dt_bias', bias)):
        _check_source(name, value)
    _check_sensitivity(source, sensitivity, torch.float32)
    if source.dim() != 4 or source.shape[-1] != 128:
        raise ValueError('KDA gate backward requires [B,T,HV,128]')
    hv = source.shape[-2]
    if alog.shape != (hv,) or bias.shape != (hv * 128,):
        raise ValueError('KDA gate backward requires A_log[HV] and dt_bias[HV*128]')
    if alog.device != source.device or bias.device != source.device:
        raise ValueError('KDA gate backward parameters must share the input device')
    bt = source.shape[0] * source.shape[1]
    chunks = (bt + 31) // 32
    p = hv * chunks * 512
    if p >= 2**31:
        raise ValueError('KDA gate backward workspace must fit signed int32')
    destination = torch.empty(source.shape, dtype=source.dtype, device=source.device)
    partials = torch.empty((1, p), dtype=torch.float32, device=source.device)
    da = torch.empty(alog.shape, dtype=alog.dtype, device=source.device)
    db = torch.empty(bias.shape, dtype=bias.dtype, device=source.device)
    vendors = prepare_backward(device, block_dim)
    key = 'gate_' + '_'.join(_DTYPES[t.dtype] for t in (source, alog, bias))
    vendors[key](
        {'source': source.view(1, -1), 'sensitivity': sensitivity.view(1, -1),
         'alog': alog.view(1, -1), 'bias': bias.view(1, -1)},
        dict(N=source.numel(), HV=hv, Channels=hv * 128, BT=bt, P=p),
        {'destination': destination.view(1, -1), 'partials': partials})
    reduce_key = f'gate_reduce_{_DTYPES[alog.dtype]}_{_DTYPES[bias.dtype]}'
    vendors[reduce_key](
        {'partials': partials, 'alog': alog.view(1, -1)},
        dict(P=p, HV=hv, Channels=hv * 128, Chunks=chunks),
        {'da': da.view(1, -1), 'db': db.view(1, -1)})
    return destination, da, db


def beta_backward(source, probability, sensitivity, *, device='a2', block_dim=1):
    _check_source('source', source)
    _check_source('probability', probability)
    _check_sensitivity(source, probability, torch.float32)
    _check_sensitivity(source, sensitivity, torch.float32)
    destination = torch.empty(source.shape, dtype=source.dtype, device=source.device)
    prepare_backward(device, block_dim)[f'beta_{_DTYPES[source.dtype]}'](
        {'source': source.view(1, -1), 'probability': probability.view(1, -1), 'sensitivity': sensitivity.view(1, -1)},
        {'N': probability.numel()}, {'destination': destination.view(1, -1)})
    return destination
