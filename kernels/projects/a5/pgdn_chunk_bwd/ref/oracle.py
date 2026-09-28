"""Literal, hash-checked FLA A; FP64 lift is qualification-only."""
import ast
import functools
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path

import torch

PIN = 'e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2'
SHA256 = '3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8'
NAMES = ('dq', 'dk', 'dv', 'dg_atk', 'dg', 'dbeta_atk', 'dbeta')


@functools.lru_cache(None)
def load(fp64=False):
    path = Path(os.environ['FLA_PGDN_NAIVE'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('FLA_PGDN_NAIVE does not match the accepted source')
    spec = importlib.util.spec_from_file_location('_pk05_literal_naive', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module.naive_recurrent_precond_gated_delta_rule
    if not fp64:
        return fn
    tree = ast.parse(inspect.getsource(fn))
    changed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'torch' and node.attr == 'float32':
            changed.append((node.lineno, node.col_offset))
            node.attr = 'float64'
    if len(changed) != 4:
        raise ValueError(f'Unexpected FP32 materialization inventory: {changed}')
    namespace = dict(module.__dict__)
    exec(compile(tree, '<PK05-four-dtype-nodes-FP64-qualification>', 'exec'), namespace)
    return namespace[fn.__name__]


def outputs(*xs, fp64=False):
    return load(fp64)(*xs, output_final_state=True)


def autograd(*xs, do=None, dht=None, dA_T=None):
    if any(x.device.type != 'cpu' or x.dtype != torch.float32 for x in xs):
        raise ValueError('Literal A requires CPU FP32 inputs')
    leaves = [x.detach().requires_grad_() for x in xs]
    values = outputs(*leaves)
    cotangents = tuple(torch.zeros_like(y) if d is None else d for y, d in zip(values, (do, dht, dA_T)))
    return dict(zip(NAMES, torch.autograd.grad(values, leaves, cotangents)))
