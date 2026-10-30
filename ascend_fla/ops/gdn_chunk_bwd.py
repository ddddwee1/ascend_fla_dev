"""Standalone grouped GDN backward; forward/autograd dispatch is unchanged."""
from __future__ import annotations

import functools
import importlib.util
import math
from pathlib import Path
import sys

import torch

SCALE = 128**-.5


@functools.lru_cache(maxsize=1)
def _pipeline():
    path = Path(__file__).resolve().parents[2] / 'kernels/projects/a5/gdn_chunk_bwd/kernels'
    name = '_afla_gdn_chunk_bwd_kernels'
    spec = importlib.util.spec_from_file_location(name, path / '__init__.py', submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from importlib import import_module
    return import_module(name + '.pipeline')


@functools.lru_cache(maxsize=1)
def _bf16_pipeline():
    path = Path(__file__).resolve().parents[2] / 'kernels/projects/a5/gdn_chunk_bwd_bf16/kernels'
    name = '_afla_gdn_chunk_bwd_bf16_kernels'
    spec = importlib.util.spec_from_file_location(name, path / '__init__.py', submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from importlib import import_module
    return import_module(name + '.pipeline')


# Qualified native inprocess grid. All three launches partition independent GM
# slices by GetVecIdx()/GetVecNum() (== 2*block_dim) with no cross-core reduction,
# so widening the grid changes only how many vector cores share the same per-item
# arithmetic; returned gradients are bitwise equal across the grid. 28 is the
# ceiling: Ascend950PR has 28 cube / 56 vector cores and a block_dim above the
# physical count deadlocks on the hardware barrier.
BLOCK_DIMS = (1, 2, 4, 8, 16, 28)


@functools.lru_cache(maxsize=len(BLOCK_DIMS))
def _compiled(block_dim):
    from ..runtime.compile import compile_kernel
    return tuple(compile_kernel(e, device='a5', block_dim=block_dim, backend='cce')
                 for e in _pipeline().entries())


@functools.lru_cache(maxsize=len(BLOCK_DIMS))
def _bf16_compiled(block_dim):
    from ..runtime.compile import compile_kernel
    return tuple(compile_kernel(e, device='a5', block_dim=block_dim, backend='cce')
                 for e in _bf16_pipeline().entries())


def _options(device, block_dim):
    if device != 'a5' or type(block_dim) is not int or block_dim not in BLOCK_DIMS:
        raise ValueError(f'GDN backward requires a5 and integer block_dim in {BLOCK_DIMS}')


def prepare(*, device='a5', block_dim=2):
    """Compile every backward stage before the first CANN operator resolution."""
    _options(device, block_dim)
    _compiled(block_dim)
    _bf16_compiled(block_dim)


def _validate(q, k, v, g, beta, do, dht, *, initial_state=None, scale=SCALE,
              head_first=False, transpose_state_layout=False, cu_seqlens=None,
              cp_context=None, use_qk_l2norm_in_kernel=False, device='a5',
              block_dim=2, launcher='inprocess'):
    _options(device, block_dim)
    if launcher not in ('inprocess', 'aclnn', 'board'):
        raise ValueError(f'unsupported launcher {launcher}')
    if head_first or transpose_state_layout or cu_seqlens is not None or cp_context is not None:
        raise ValueError('require token-major, K-major state, fixed-length inputs and no CP')
    if use_qk_l2norm_in_kernel:
        raise ValueError('GDN backward does not normalize q/k')
    if initial_state is not None:
        raise ValueError('initial_state must be None; no dh0 is produced')
    if type(scale) not in (int, float) or not math.isclose(scale, SCALE, rel_tol=0., abs_tol=1e-12):
        raise ValueError('scale must be 128**-0.5')
    tensors = dict(q=q, k=k, v=v, g=g, beta=beta)
    if not all(isinstance(x, torch.Tensor) for x in tensors.values()):
        raise ValueError('q/k/v/g/beta must be tensors')
    if q.ndim != 4 or min(q.shape[:3]) < 1 or q.shape[-1] != 128 or q.shape[1] % 64 or q.shape[1] > 4096:
        raise ValueError('require positive B/H, T multiple of 64 up to 4096, and K=128')
    if v.ndim != 4 or v.shape[:2] != q.shape[:2] or v.shape[-1] != 128:
        raise ValueError('v requires matching B/T and V=128')
    B, T, H, _ = q.shape
    HV = v.shape[2]
    if HV < 1 or HV % H:
        raise ValueError('HV must be a positive multiple of H')
    if q.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('q/k/v/do require matching float32 or bfloat16 dtype')
    if do is None and dht is None:
        raise ValueError('at least one of do/dht is required')
    if do is not None:
        tensors['do'] = do
    if dht is not None:
        tensors['dht'] = dht
    expected = dict(q=q.shape, k=q.shape, v=v.shape, g=v.shape[:3], beta=v.shape[:3],
                    do=v.shape, dht=(B,HV,128,128))
    where = 'npu' if launcher == 'inprocess' else 'cpu'
    for name, x in tensors.items():
        if not isinstance(x, torch.Tensor) or x.shape != expected[name]:
            raise ValueError(f'{name} requires shape {tuple(expected[name])}')
        dtype = torch.float32 if name in ('g', 'beta', 'dht') else q.dtype
        if x.dtype != dtype:
            raise ValueError(f'{name} requires {dtype}; q/k/v/do must have matching dtype')
    # Validate metadata for every input before any diagnostic tensor operation.
    for name, x in tensors.items():
        if x.device != q.device or x.device.type != where:
            raise ValueError(f'{launcher} requires all tensors on one {where} device')
        if not x.is_contiguous():
            raise ValueError(f'{name} must be contiguous')
    if where == 'cpu':
        for name, x in tensors.items():
            if not bool(torch.isfinite(x).all()):
                raise ValueError(f'{name} must be finite')
    if where == 'cpu' and (bool((g > 0).any()) or bool(((beta < 0) | (beta > 1)).any())):
        raise ValueError('g<=0 and beta in [0,1] required')


@torch.no_grad()
def chunk_gdn_bwd(q, k, v, g, beta, do=None, dht=None, *, initial_state=None,
                  scale=SCALE, head_first=False, transpose_state_layout=False,
                  cu_seqlens=None, cp_context=None, use_qk_l2norm_in_kernel=False,
                  device='a5', block_dim=2, launcher='inprocess', board=None,
                  out_dir=None, timeout=600):
    """Return (dq, dk, dv, dg, dbeta) for <do,o> + <dht,final_state>.

    q/k gradients sum consecutive value-head contributions in FP32. q/k/v/do
    share FP32 or BF16 dtype; dq/dk/dv use that dtype, directly written by the
    kernel. Gates, dht, dg/dbeta and internal math remain FP32.
    At least one cotangent is required; an absent cotangent contributes zero.
    Finite inputs/cotangents, g<=0, beta in [0,1] are NPU caller preconditions,
    checked for explicit CPU launchers. Zero initial state only, no dh0.
    No q/k normalization, autograd wiring, or higher-order derivative support.
    """
    _validate(q,k,v,g,beta,do,dht,initial_state=initial_state,scale=scale,
              head_first=head_first,transpose_state_layout=transpose_state_layout,
              cu_seqlens=cu_seqlens,cp_context=cp_context,
              use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
              device=device,block_dim=block_dim,launcher=launcher)
    if q.dtype == torch.bfloat16:
        pipeline = _bf16_pipeline()
        inputs = dict(q=q,k=k,v=v,g=g,beta=beta,do=do,dht=dht)
    else:
        pipeline = _pipeline()
        inputs = dict(q=q,k=k,v=v,g=g,beta=beta,
                      do=torch.zeros_like(v) if do is None else do,
                      dht=torch.zeros(q.shape[0],v.shape[2],128,128,dtype=torch.float32,device=q.device)
                      if dht is None else dht)
    if launcher == 'inprocess':
        prepare(device=device, block_dim=block_dim)
        artifacts = _bf16_compiled(block_dim) if q.dtype == torch.bfloat16 else _compiled(block_dim)
        compiled = dict(zip((e.name for e in pipeline.entries()), artifacts))
        def launch(entry, sources, outputs, scalars):
            op = compiled[entry.name]
            op(sources,{n:scalars[n] for n in op.scalar_names},outputs)
            return outputs
    else:
        from ascriptor.runtime import OpExec
        def launch(entry, sources, outputs, scalars):
            root = None if out_dir is None else Path(out_dir)/entry.name
            op = OpExec(entry,launcher=launcher,device=device,backend='cce',block_dim=block_dim,
                        board=board,out_dir=root,timeout=timeout)
            result = op(*(tuple(sources.values())+tuple(outputs.values())+tuple(scalars.values())))
            return dict(zip(outputs,(result,) if len(outputs)==1 else result))
    result = pipeline.run(inputs,launch)
    return tuple(result[name] for name in ('dq', 'dk', 'dv', 'dg', 'dbeta'))
