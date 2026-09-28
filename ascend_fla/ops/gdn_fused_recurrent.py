"""Short-step GDN with raw keys, native input dtypes and fresh FP32 state."""
from __future__ import annotations
import functools
import importlib.util
import math
from pathlib import Path
import sys
import torch

SCALE = 128**-0.5
S_MAX = 16
# Candidate set. Qualification and public release are pending native evidence.
BLOCK_DIMS = (1, 2, 4, 8, 16, 28)


@functools.lru_cache(maxsize=1)
def _pipeline():
    root=Path(__file__).resolve().parents[2]/'kernels/projects/a5/gdn_fused_recurrent/kernels'
    name='_afla_gda04_kernels'
    spec=importlib.util.spec_from_file_location(name,root/'__init__.py',submodule_search_locations=[str(root)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    from importlib import import_module
    return import_module(name+'.pipeline')


def _options(device,block_dim):
    if device!='a5':raise ValueError('GDN recurrent requires device=a5')
    if type(block_dim) is not int or block_dim not in BLOCK_DIMS:
        raise ValueError(f'block_dim must be one of {BLOCK_DIMS}')


@functools.lru_cache(maxsize=len(BLOCK_DIMS))
def _compiled(block_dim):
    from ..runtime.compile import compile_kernel
    return tuple(compile_kernel(entry,device='a5',block_dim=block_dim,backend='cce')
                 for entry in _pipeline().all_entries())


def prepare(*,device='a5',block_dim=1):
    """Register both native dtypes before the first aclnn execution.

    For prefill/continuation, also call gdn_chunk_fwd.prepare before either
    operator executes. Each process must use one build per operator name.
    """
    _options(device,block_dim);_compiled(block_dim)


def _validate(q,k,v,g,beta,initial_state,output_final_state,scale,head_first,
              device,block_dim,launcher,unsupported):
    _options(device,block_dim)
    if unsupported:raise ValueError('unsupported options: '+', '.join(sorted(unsupported)))
    if head_first is not False:raise ValueError('head_first is unsupported; require token-major')
    if type(output_final_state) is not bool:raise ValueError('output_final_state must be bool')
    if type(scale) not in (int,float) or not math.isclose(scale,SCALE,rel_tol=0.,abs_tol=1e-12):
        raise ValueError('scale must be 128**-0.5')
    if launcher not in ('inprocess','aclnn','board'):raise ValueError('unsupported launcher')
    if not all(isinstance(x,torch.Tensor) for x in (q,k,v,g,beta)):
        raise ValueError('q/k/v/g/beta must be tensors')
    if initial_state is not None and not isinstance(initial_state,torch.Tensor):
        raise ValueError('initial_state must be a tensor or None')
    if q.ndim!=4 or q.shape[-1]!=128 or min(q.shape[:3])<1 or q.shape[1]>S_MAX:
        raise ValueError('q requires positive B/H, S=1..16, K=128')
    b,s,h,_=q.shape
    if v.ndim!=4 or v.shape[:2]!=(b,s) or v.shape[-1]!=128 or v.shape[2]<1 or v.shape[2]%h:
        raise ValueError('v requires [B,S,HV,128], positive HV multiple of H')
    hv=v.shape[2]
    if b*hv*128*128>2**31-1:raise ValueError('state exceeds signed32 indexing domain')
    expected_device='npu' if launcher=='inprocess' else 'cpu'
    values=dict(q=q,k=k,v=v,g=g,beta=beta)
    if initial_state is not None:values['initial_state']=initial_state
    for name,x in values.items():
        shape=((b,hv,128,128) if name=='initial_state' else
               (b,s,hv) if name in ('g','beta') else v.shape if name=='v' else q.shape)
        dtype=torch.float32 if name in ('g','beta','initial_state') else q.dtype
        if x.shape!=shape:raise ValueError(f'{name} shape must be {tuple(shape)}')
        if x.dtype!=dtype or q.dtype not in (torch.float32,torch.bfloat16):
            raise ValueError(f'{name}: q/k/v require matching float32/bfloat16; gates/state float32')
        if x.device!=q.device or x.device.type!=expected_device:
            raise ValueError(f'{launcher} requires tensors on one {expected_device} device')
        if not x.is_contiguous():raise ValueError(f'{name} must be contiguous')
        if x.numel()>2**31-1:raise ValueError(f'{name} exceeds signed32 indexing domain')
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError('GDN recurrent is inference-only; backward is unavailable')
        # Native execution does not read or transform tensor values on the host.
        if expected_device=='cpu' and not bool(torch.isfinite(x).all()):
            raise ValueError(f'{name} must be finite')
    if expected_device=='cpu' and (bool((g>0).any()) or bool(((beta<0)|(beta>1)).any())):
        raise ValueError('g<=0 and beta in[0,1] required')


def fused_recurrent_gdn(q,k,v,g,beta,*,initial_state=None,output_final_state=False,
                        scale=SCALE,head_first=False,device='a5',block_dim=1,
                        launcher='inprocess',board=None,out_dir=None,timeout=600,
                        **unsupported):
    """Return native-dtype output and optional fresh FP32 [B,HV,K,V] state.

    Consecutive HV/H value heads share one raw q/k head. No normalization,
    in-place state update, host cast, head expansion, or reference fallback.
    Native callers supply finite operands, g<=0 and beta in[0,1], avoiding
    overflowing recurrence intermediates; CPU diagnostic launchers check values.
    Numerical qualification is limited to the documented tested input domain.
    """
    _validate(q,k,v,g,beta,initial_state,output_final_state,scale,head_first,
              device,block_dim,launcher,unsupported)
    inputs=dict(q=q,k=k,v=v,g=g,beta=beta,initial_state=initial_state)
    if launcher=='inprocess':
        compiled=dict(zip((entry.name for entry in _pipeline().all_entries()),_compiled(block_dim)))
        def launch(entry,sources,outputs,scalars):
            op=compiled[entry.name]
            op(sources,{name:scalars[name] for name in op.scalar_names},outputs)
            return outputs
    else:
        from ascriptor.runtime import OpExec
        def launch(entry,sources,outputs,scalars):
            root=None if out_dir is None else Path(out_dir)/entry.name
            op=OpExec(entry,launcher=launcher,device=device,backend='cce',block_dim=block_dim,
                      board=board,out_dir=root,timeout=timeout)
            result=op(*(tuple(sources.values())+tuple(outputs.values())+tuple(scalars.values())))
            return dict(zip(outputs,(result,) if len(outputs)==1 else result))
    result=_pipeline().run(inputs,launch)
    return result['o'],result['final_state'] if output_final_state else None
