"""Short-step PGDN with native dtypes and two independent fresh FP32 states."""
from __future__ import annotations
import functools
import importlib.util
from pathlib import Path
import sys
import torch

SCALE = 128**-0.5
S_MAX = 16
# Candidate native grid; qualification is recorded in the research ledger.
BLOCK_DIMS = (1, 2, 4, 8, 16, 28)


@functools.lru_cache(maxsize=1)
def _pipeline():
    root=Path(__file__).resolve().parents[2]/'kernels/projects/a5/pgdn_fused_recurrent/kernels'
    name='_afla_pk06_kernels'
    spec=importlib.util.spec_from_file_location(name,root/'__init__.py',submodule_search_locations=[str(root)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    from importlib import import_module
    return import_module(name+'.pipeline')


def _options(device,block_dim):
    if device!='a5':raise ValueError('PGDN recurrent requires device=a5')
    if type(block_dim) is not int or block_dim not in BLOCK_DIMS:
        raise ValueError(f'block_dim must be one of {BLOCK_DIMS}')


@functools.lru_cache(maxsize=len(BLOCK_DIMS))
def _compiled(block_dim):
    from ..runtime.compile import compile_kernel
    return tuple(compile_kernel(entry,device='a5',block_dim=block_dim,backend='cce')
                 for entry in _pipeline().all_entries())


def prepare(*,device='a5',block_dim=1,chunk=False,chunk_block_dim=2):
    """Register two decode entries, optionally all twelve chunk entries as well.

    Call with chunk=True before either operator executes for prefill/decode.
    Every process must use one build per operator name.
    """
    _options(device,block_dim)
    if type(chunk) is not bool:raise ValueError('chunk must be bool')
    if type(chunk_block_dim) is not int or chunk_block_dim not in (1,2):
        raise ValueError('chunk_block_dim must be 1 or 2')
    _compiled(block_dim)
    if chunk:
        from .pgdn_chunk_fwd import prepare as chunk_prepare
        chunk_prepare(device=device,block_dim=chunk_block_dim)


def _validate(q,k,v,g_atk,g,beta_atk,beta,initial_state,initial_A_state,output_final_state,scale,x,eps,log_atk_scale,use_qk_l2norm_in_kernel,head_first,
              device,block_dim,launcher,unsupported):
    _options(device,block_dim)
    if unsupported:raise ValueError('unsupported options: '+', '.join(sorted(unsupported)))
    if head_first is not False:raise ValueError('head_first is unsupported; require token-major')
    if type(output_final_state) is not bool:raise ValueError('output_final_state must be bool')
    if type(scale) not in (int,float) or scale!=SCALE:
        raise ValueError('scale must be 128**-0.5')
    for name,value,expected in [('x',x,1.5),('eps',eps,1e-6),('log_atk_scale',-.2 if log_atk_scale is None else log_atk_scale,-.2)]:
        if type(value) not in (int,float) or value!=expected:raise ValueError(f'{name} must be {expected}')
    if use_qk_l2norm_in_kernel is not True:raise ValueError('use_qk_l2norm_in_kernel must be True')
    if launcher not in ('inprocess','aclnn','board'):raise ValueError('unsupported launcher')
    if not all(isinstance(x,torch.Tensor) for x in (q,k,v,g_atk,g,beta_atk,beta)):
        raise ValueError('q/k/v/g_atk/g/beta_atk/beta must be tensors')
    if initial_state is not None and not isinstance(initial_state,torch.Tensor):
        raise ValueError('initial_state must be a tensor or None')
    if initial_A_state is not None and not isinstance(initial_A_state,torch.Tensor):
        raise ValueError('initial_A_state must be a tensor or None')
    if q.ndim!=4 or q.shape[-1]!=128 or min(q.shape[:3])<1 or q.shape[1]>S_MAX:
        raise ValueError('q requires positive B/H, S=1..16, K=128')
    b,s,h,_=q.shape
    if v.ndim!=4 or v.shape[:2]!=(b,s) or v.shape[-1]!=128 or v.shape[2]<1 or v.shape[2]%h:
        raise ValueError('v requires [B,S,HV,128], positive HV multiple of H')
    hv=v.shape[2]
    if b*hv*128*128>2**31-1:raise ValueError('state exceeds signed32 indexing domain')
    expected_device='npu' if launcher=='inprocess' else 'cpu'
    values=dict(q=q,k=k,v=v,g_atk=g_atk,g=g,beta_atk=beta_atk,beta=beta)
    if initial_state is not None:values['initial_state']=initial_state
    if initial_A_state is not None:values['initial_A_state']=initial_A_state
    for name,x in values.items():
        shape=((b,hv,128,128) if name=='initial_state' else (b,h,128) if name=='initial_A_state' else
               (b,s,h) if name in ('g_atk','beta_atk') else
               (b,s,hv) if name in ('g','beta') else v.shape if name=='v' else q.shape)
        dtype=q.dtype if name in ('q','k','v') else torch.float32
        if x.shape!=shape:raise ValueError(f'{name} shape must be {tuple(shape)}')
        if x.dtype!=dtype or q.dtype not in (torch.float32,torch.bfloat16):
            raise ValueError(f'{name}: q/k/v require matching float32/bfloat16; gates/state float32')
        if x.device!=q.device or x.device.type!=expected_device:
            raise ValueError(f'{launcher} requires tensors on one {expected_device} device')
        if not x.is_contiguous():raise ValueError(f'{name} must be contiguous')
        if x.numel()>2**31-1:raise ValueError(f'{name} exceeds signed32 indexing domain')
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError('PGDN recurrent is inference-only; backward is unavailable')
        # Native execution does not read or transform tensor values on the host.
        if expected_device=='cpu' and not bool(torch.isfinite(x).all()):
            raise ValueError(f'{name} must be finite')
    if expected_device=='cpu':
        for name,value in [('g',g),('g_atk',g_atk)]:
            if bool((value>0).any()):raise ValueError(f'{name} must be <=0')
        for name,value in [('beta',beta),('beta_atk',beta_atk)]:
            if bool(((value<0)|(value>1)).any()):raise ValueError(f'{name} must be in[0,1]')
        if initial_A_state is not None and bool((initial_A_state<0).any()):
            raise ValueError('initial_A_state must be nonnegative')


def fused_recurrent_pgdn(q,k,v,g_atk,g,beta_atk,beta,*,initial_state=None,initial_A_state=None,output_final_state=False,
                        scale=SCALE,x=1.5,eps=1e-6,log_atk_scale=None,use_qk_l2norm_in_kernel=True,
                        head_first=False,device='a5',block_dim=1,
                        launcher='inprocess',board=None,out_dir=None,timeout=600,
                        **unsupported):
    """Return (o, main_state, ATK_state), with both states None unless requested.

    Native BF16/FP32 q/k normalize in FP32 with norm clamp1e-12, then q scales
    by128**-.5. ATK uses eps1e-6, center-.2 and x1.5. Consecutive HV/H value
    heads share one key head; ATK remains[B,H,128], main state[B,HV,128,128].
    Each initial state independently defaults to zero; returned states are fresh.
    Native callers supply finite operands, g/g_atk<=0, beta/beta_atk in[0,1],
    nonnegative A and nonoverflowing intermediates. CPU diagnostics check values.
    Numerical qualification is limited to the documented measured domain.
    """
    _validate(q,k,v,g_atk,g,beta_atk,beta,initial_state,initial_A_state,output_final_state,scale,x,eps,log_atk_scale,use_qk_l2norm_in_kernel,head_first,
              device,block_dim,launcher,unsupported)
    inputs=dict(q=q,k=k,v=v,g_atk=g_atk,g=g,beta_atk=beta_atk,beta=beta,initial_state=initial_state,initial_A_state=initial_A_state)
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
    return (result['o'],result['final_state'] if output_final_state else None,
            result['final_A_state'] if output_final_state else None)
