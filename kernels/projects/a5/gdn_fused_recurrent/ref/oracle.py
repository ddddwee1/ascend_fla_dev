"""Literal CPU A approved by PM issue94 comment5870217647.

The pinned fused-recurrent implementation is Triton, not a CPU backend. This
adapter loads the literal naive implementation; it never executes Triton.
FP64 mode changes only its two explicit FP32 cast nodes for calibration.
"""
import ast
import functools
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import torch

PIN='e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2'
SHA256='d1cf17992349fd3e94af999b22e3d3a81be4a2d1881ce5b70a3457257166e0cb'


@functools.lru_cache(None)
def load(fp64=False):
    path=Path(os.environ['FLA_GDN_NAIVE'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('FLA_GDN_NAIVE source hash mismatch')
    spec=importlib.util.spec_from_file_location('_gda04_literal_naive',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fn=module.naive_recurrent_gated_delta_rule
    if not fp64:return fn
    tree=ast.parse(inspect.getsource(fn));changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='torch' and node.attr=='float32':
            changed.append((node.lineno,node.col_offset));node.attr='float64'
    if len(changed)!=2:raise ValueError(f'unexpected FP32 materializations: {changed}')
    namespace=dict(module.__dict__)
    exec(compile(tree,'<GDA04-two-dtype-nodes-FP64-calibration>','exec'),namespace)
    return namespace[fn.__name__]


def oracle(inputs, *, fp64=False):
    h=inputs['q'].shape[2];hv=inputs['v'].shape[2]
    q,k=(inputs[n].repeat_interleave(hv//h,dim=2) for n in ('q','k'))
    o,state=load(fp64)(q,k,inputs['v'],inputs['beta'],inputs['g'],
                       initial_state=inputs.get('initial_state'),output_final_state=True)
    return dict(o=o,final_state=state)
