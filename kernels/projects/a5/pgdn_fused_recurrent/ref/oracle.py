"""Hash-checked literal pinned PGDN naive A; no CUDA/Triton execution.

The explicitly labeled FP64 lift changes four dtype nodes. The tensor-center
branch is included in the inventory although fixed scalar -.2 never enters it.
"""
import ast
import functools
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path

PIN = 'e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2'
SHA256 = '3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8'
DTYPE_LINES = (95,116,118,120)


@functools.lru_cache(None)
def load(fp64=False):
    path = Path(os.environ['FLA_PGDN_NAIVE'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('FLA_PGDN_NAIVE source hash mismatch')
    spec = importlib.util.spec_from_file_location('_pk06_literal_naive',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fn = module.naive_recurrent_precond_gated_delta_rule
    if not fp64:return fn
    source,start = inspect.getsourcelines(fn)
    tree = ast.parse(''.join(source)); changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='torch' and node.attr=='float32':
            changed.append(node.lineno+start-1);node.attr='float64'
    if tuple(sorted(changed))!=DTYPE_LINES:raise ValueError(f'unexpected dtype nodes: {changed}')
    namespace = dict(module.__dict__)
    exec(compile(tree,'<PK06-four-dtype-nodes-FP64-lift>','exec'),namespace)
    return namespace[fn.__name__]


def oracle(inputs, *, fp64=False):
    o,state,atk = load(fp64)(**inputs,output_final_state=True)
    return dict(o=o,final_state=state,final_A_state=atk)
