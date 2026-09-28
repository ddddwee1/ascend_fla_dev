"""Standalone FP32 PGDN backward, including the shared ATK adjoint."""
from __future__ import annotations

import functools
import importlib
import importlib.util
import math
from pathlib import Path
import sys

import torch

SCALE = 128**-.5
NAMES = ('dq', 'dk', 'dv', 'dg_atk', 'dg', 'dbeta_atk', 'dbeta')


@functools.lru_cache(maxsize=1)
def _pipeline():
    directory = Path(__file__).resolve().parents[2] / 'kernels/projects/a5/pgdn_chunk_bwd/kernels'
    name = '_afla_pk05_kernels'
    spec = importlib.util.spec_from_file_location(name, directory / '__init__.py', submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name + '.pipeline')


def _options(device, block_dim):
    if device != 'a5' or type(block_dim) is not int or block_dim not in (1, 2):
        raise ValueError('PGDN backward requires a5 and integer block_dim in (1,2)')


@functools.lru_cache(maxsize=2)
def _compiled(block_dim):
    from ..runtime.compile import compile_kernel
    return tuple(compile_kernel(entry, device='a5', block_dim=block_dim, backend='cce')
                 for entry in _pipeline().entries())


def prepare(*, device='a5', block_dim=2):
    """Compile all four stages before the first CANN operator resolution."""
    _options(device, block_dim)
    _compiled(block_dim)


def _constant(name, value, expected):
    if type(value) not in (float, int) or not math.isfinite(value) or value != expected:
        raise ValueError(f'{name} must be {expected}; got {value}')


def _validate(q, k, v, g_atk, g, beta_atk, beta, do=None, dht=None, dA_T=None, *,
              scale=None, initial_state=None, initial_A_state=None,
              use_qk_l2norm_in_kernel=True, x=1.5, eps=1e-6, log_atk_scale=None,
              head_first=False, cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None,
              transpose_state_layout=False, device='a5', block_dim=2, launcher='inprocess'):
    _options(device, block_dim)
    if launcher not in ('inprocess', 'aclnn', 'board'):
        raise ValueError(f'unsupported launcher {launcher}')
    if head_first or transpose_state_layout:
        raise ValueError('requires token-major input and K-major state')
    if any(value is not None for value in (cu_seqlens, cu_seqlens_cpu, cp_context)):
        raise ValueError('varlen and context parallelism are unsupported')
    if use_qk_l2norm_in_kernel is not True:
        raise ValueError('use_qk_l2norm_in_kernel must be True')
    _constant('scale', SCALE if scale is None else scale, SCALE)
    _constant('x', x, 1.5)
    _constant('eps', eps, 1e-6)
    _constant('log_atk_scale', -.2 if log_atk_scale is None else log_atk_scale, -.2)
    tensors = dict(q=q, k=k, v=v, g_atk=g_atk, g=g, beta_atk=beta_atk, beta=beta)
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise ValueError('all differentiated inputs must be tensors')
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] < 1 or q.shape[1] % 64 or q.shape[1] > 4096 or q.shape[2] < 1 or q.shape[3] != 128:
        raise ValueError(f'requires B=1, positive H, T multiple of64 up to4096, K=128; got {tuple(q.shape)}')
    B, T, H, _ = q.shape
    if v.ndim != 4 or v.shape[:2] != (B, T) or v.shape[-1] != 128:
        raise ValueError(f'v requires [B,T,HV,128] with B/T={(B,T)}; got {tuple(v.shape)}')
    HV = v.shape[2]
    if HV < 1 or HV % H:
        raise ValueError(f'HV must be a positive multiple of H; got H={H},HV={HV}')
    for name, value, shape in (('initial_state', initial_state, (B, HV, 128, 128)),
                                ('initial_A_state', initial_A_state, (B, H, 128))):
        if value is not None:
            raise ValueError(f'{name} only accepts None for zero state; state layout would be {shape}')
    if do is None and dht is None and dA_T is None:
        raise ValueError('at least one of do/dht/dA_T is required')
    for name, value in (('do', do), ('dht', dht), ('dA_T', dA_T)):
        if value is not None:
            tensors[name] = value
    expected = dict(q=q.shape, k=q.shape, v=v.shape, g_atk=(B,T,H), beta_atk=(B,T,H),
                    g=(B,T,HV), beta=(B,T,HV), do=v.shape,
                    dht=(B,HV,128,128), dA_T=(B,H,128))
    # Complete dtype/shape/device/contiguity checks before any tensor predicate.
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name]:
            raise ValueError(f'{name} requires shape {tuple(expected[name])}')
        if value.dtype != torch.float32:
            raise ValueError(f'{name} requires float32; BF16 backward is outside PK-05')
    where = 'npu' if launcher == 'inprocess' else 'cpu'
    for name, value in tensors.items():
        if value.device != q.device or value.device.type != where:
            raise ValueError(f'{launcher} requires all tensors on one {where} device')
        if not value.is_contiguous():
            raise ValueError(f'{name} must be contiguous')
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must be finite')
    for name, value in (('g', g), ('g_atk', g_atk)):
        if bool((value > 0).any()):
            raise ValueError(f'{name} must be nonpositive')
        if not bool(torch.isfinite(value.reshape(B,T//64,64,-1).sum(2)).all()):
            raise ValueError(f'{name} chunk sum must remain finite in FP32')
    for name, value in (('beta', beta), ('beta_atk', beta_atk)):
        if bool(((value < 0) | (value > 1)).any()):
            raise ValueError(f'{name} must be in [0,1]')


@torch.no_grad()
def chunk_pgdn_bwd(q, k, v, g_atk, g, beta_atk, beta, do=None, dht=None, dA_T=None, *,
                   scale=None, initial_state=None, initial_A_state=None,
                   use_qk_l2norm_in_kernel=True, x=1.5, eps=1e-6, log_atk_scale=None,
                   head_first=False, cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None,
                   transpose_state_layout=False, device='a5', block_dim=2, launcher='inprocess',
                   board=None, out_dir=None, timeout=600):
    """Return FP32 (dq,dk,dv,dg_atk,dg,dbeta_atk,dbeta).

    Differentiate <do,o>+<dht,final_state>+<dA_T,final_A_state>. Consecutive
    value heads share read keys and one ATK state; their contributions sum
    before its reverse recurrence. Only B1/zero initial states are supported.
    No autograd wiring or higher-order derivative support. Near-zero q/k rows
    remain supported; numerical qualification discloses cancellation residuals
    separately from the ordinary-element relative budget. Value checks execute
    on NPU too and belong to public-call timing. No reference fallback.
    """
    _validate(q,k,v,g_atk,g,beta_atk,beta,do,dht,dA_T,scale=scale,
              initial_state=initial_state,initial_A_state=initial_A_state,
              use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,x=x,eps=eps,log_atk_scale=log_atk_scale,
              head_first=head_first,cu_seqlens=cu_seqlens,cu_seqlens_cpu=cu_seqlens_cpu,
              cp_context=cp_context,transpose_state_layout=transpose_state_layout,
              device=device,block_dim=block_dim,launcher=launcher)
    inputs = dict(q=q,k=k,v=v,g_atk=g_atk,g=g,beta_atk=beta_atk,beta=beta,do=do,dht=dht,dA_T=dA_T)
    pipeline = _pipeline()
    if launcher == 'inprocess':
        prepare(device=device, block_dim=block_dim)
        compiled = dict(zip((entry.name for entry in pipeline.entries()), _compiled(block_dim)))

        def launch(entry, sources, outputs, scalars):
            operation = compiled[entry.name]
            operation(sources, {n: scalars[n] for n in operation.scalar_names}, outputs)
            return outputs
    else:
        from ascriptor.runtime import OpExec

        def launch(entry, sources, outputs, scalars):
            root = None if out_dir is None else Path(out_dir) / entry.name
            operation = OpExec(entry, launcher=launcher, device=device, backend='cce', block_dim=block_dim,
                               board=board, out_dir=root, timeout=timeout)
            result = operation(*(tuple(sources.values()) + tuple(outputs.values()) + tuple(scalars.values())))
            return dict(zip(outputs, (result,) if len(outputs) == 1 else result))
    result = pipeline.run(inputs, launch)
    return tuple(result[name] for name in NAMES)
