"""Existing public GDN/PGDN APIs and independent, unchanged CPU references.

Reference preparation is outside every measured interval. No implementation
kernel, input ABI, or numerical budget is replaced by this benchmark.
"""
from __future__ import annotations

import hashlib
import importlib
from functools import partial
from pathlib import Path

import torch


FLA_PIN = 'e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2'
ORACLE_SHA256 = {
    'FLA_GDN_NAIVE': 'd1cf17992349fd3e94af999b22e3d3a81be4a2d1881ce5b70a3457257166e0cb',
    'FLA_PGDN_NAIVE': '3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8',
}


FAMILIES = {
    'gdn_fwd': ('gdn_chunk_fwd', 'chunk_gdn', 'gdn_chunk_fwd_bf16'),
    'pgdn_fwd': ('pgdn_chunk_fwd', 'chunk_pgdn', 'pgdn_chunk_fwd_bf16'),
    'gdn_bwd': ('gdn_chunk_bwd', 'chunk_gdn_bwd', 'gdn_chunk_bwd_bf16'),
    'pgdn_bwd': ('pgdn_chunk_bwd', 'chunk_pgdn_bwd', 'pgdn_chunk_bwd'),
    'gdn_decode': ('gdn_fused_recurrent', 'fused_recurrent_gdn', 'gdn_fused_recurrent'),
    'pgdn_decode': ('pgdn_fused_recurrent', 'fused_recurrent_pgdn', 'pgdn_fused_recurrent'),
}


def digest(value):
    if value is None:
        return None
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def module(family):
    return importlib.import_module('ascend_fla.ops.' + FAMILIES[family][0])


def refmodule(family, name):
    return importlib.import_module('kernels.projects.a5.' + FAMILIES[family][2] + '.ref.' + name)


def block_dims(family):
    return (1, 2, 4, 8, 16, 28) if family.endswith('decode') else (1, 2)


def cases(family):
    """Full DONE workload first, then the two model shapes; seeds exclude bd."""
    dtypes = ('float32',) if family == 'pgdn_bwd' else ('bfloat16', 'float32')
    result = []
    if family.endswith('decode'):
        for steps in (16, 1):
            for dtype in dtypes:
                result.append(dict(id=f's{steps}_{dtype}', B=1, S=steps, H=16, HV=32,
                                   dtype=dtype, seed=950000 if family == 'pgdn_decode' else 940000,
                                   state='random', atk='random', measure=True,
                                   source='GDA-04/PK-06 DONE decode shape'))
    else:
        for length, heads, value_heads, measure, source in (
            (4096, 8, 8, False, 'existing DONE full workload'),
            (4096, 16, 32, True, 'models.json:test_shapes.long_context'),
            (1024, 16, 32, True, 'models.json:test_shapes.qwen3_next_layer'),
        ):
            for dtype in dtypes:
                result.append(dict(id=f't{length}_h{heads}_hv{value_heads}_{dtype}',
                                   B=1, T=length, H=heads, HV=value_heads, dtype=dtype,
                                   seed=173000 + length, mode='both', mask=7, kind='random',
                                   measure=measure, source=source))
    return result


def kernel_pairs(family, block_dim):
    """All registered entries, including companion dtype entries, for one bd."""
    api = module(family)
    api.prepare(block_dim=block_dim)
    if family == 'gdn_fwd':
        entries = api._native_pipeline().all_entries()
        compiled = api._compiled(block_dim)
    elif family == 'gdn_bwd':
        entries = (*api._pipeline().entries(), *api._bf16_pipeline().entries())
        compiled = (*api._compiled(block_dim), *api._bf16_compiled(block_dim))
    elif family == 'pgdn_fwd':
        entries = (*api._pipeline().entries(), *api._bf16_pipeline().entries())
        compiled = api._compiled(block_dim)
    else:
        pipeline = api._pipeline()
        entries = pipeline.all_entries() if family.endswith('decode') else pipeline.entries()
        compiled = api._compiled(block_dim)
    assert len(entries) == len(compiled)
    return list(zip(entries, compiled))


def selected_entries(family, dtype):
    api = module(family)
    if family == 'gdn_fwd':
        return api._native_pipeline().entries(dtype)
    if family in ('gdn_bwd', 'pgdn_fwd'):
        pipeline = api._bf16_pipeline() if dtype == torch.bfloat16 else api._pipeline()
        return pipeline.entries()
    if family == 'pgdn_bwd':
        return api._pipeline().entries()
    return (api._pipeline().all_entries()[0 if dtype == torch.float32 else 1],)


class Workload:
    def __init__(self, family, case):
        self.family, self.case = family, case
        self.api = module(family)
        self.reference = refmodule(family, 'reference')
        self.oracle = refmodule(family, 'oracle')
        self.dtype = getattr(torch, case['dtype'])
        self.auxiliary = self.classes = None
        wrapped = dict(id=case['id'], seed=case['seed'], dtype=case['dtype'], parameters=case)
        if family == 'gdn_bwd':
            self.bf = refmodule(family, 'bf16')
            self.cpu = self.bf.make_inputs(wrapped, self.dtype)
            self.names = self.bf.NAMES
            self.refs = dict(A=self.oracle.autograd(**self.cpu), B=self.bf.reference(self.cpu))
        elif family == 'pgdn_bwd':
            calibration = refmodule(family, 'calibrate')
            self.classification = refmodule(family, 'classification')
            xs, ds = calibration.inputs(case)
            self.cpu = {**dict(zip(self.reference.INPUTS, xs)), **ds}
            self.names = self.reference.NAMES
            self.refs = dict(A=self.oracle.autograd(*xs, **ds), B=self.reference.analytical(*xs, **ds))
            self.auxiliary = self.reference.analytical(
                *(x.double() for x in xs), **{n: x.double() for n, x in ds.items()},
                fp32_branch=True, auxiliary=True)
            self.classes = self.classification.classify(self.auxiliary)
        else:
            self.cpu = self.reference.make_inputs(case if family.endswith('decode') else wrapped)
            self.names = ('o', 'final_state', 'final_A_state') if family.startswith('pgdn') else ('o', 'final_state')
            oracle = self.oracle.oracle if family.endswith('decode') else self.oracle.reference
            with torch.no_grad():
                self.refs = dict(A=oracle(self.cpu), B=self.reference.reference(self.cpu))
        assert all(x is None or x.device.type == 'cpu' for x in self.cpu.values())
        self.input_hashes = {n: digest(x) for n, x in self.cpu.items()}

    def upload(self):
        # The chunk API accepts None for zero initial state. The independent
        # block-solve reference represents that same state explicitly on CPU.
        self.device_inputs = {n: None if x is None else x.to('npu')
                              for n, x in self.cpu.items()
                              if not (self.family == 'pgdn_fwd' and n == 'initial_state')}
        assert {n: digest(x) for n, x in self.device_inputs.items()} == {
            n: self.input_hashes[n] for n in self.device_inputs}

    def call(self, block_dim, values=None):
        return self.bind(block_dim, values)()

    def bind(self, block_dim, values=None):
        values = self.device_inputs if values is None else values
        kwargs = dict(block_dim=block_dim)
        if not self.family.endswith('bwd'):
            kwargs['output_final_state'] = True
        return partial(getattr(self.api, FAMILIES[self.family][1]), **values, **kwargs)

    def compare(self, actual, output_dir, label):
        if set(actual) != set(self.names):
            raise ValueError('Missing or extra output')
        for n, value in actual.items():
            narrow = n in ('dq', 'dk', 'dv') if self.family == 'gdn_bwd' else n == 'o'
            dtype = self.dtype if narrow else torch.float32
            if value.dtype != dtype or value.shape != self.refs['B'][n].shape:
                raise ValueError(f'Unexpected public metadata: {n}')
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Nonfinite public output: {n}')
        if self.family == 'pgdn_bwd':
            result = self.classification.compare(actual, self.refs, self.classes)
            disclosure = self.classification.write_disclosures(
                Path(output_dir) / (label + '-disclosures.jsonl'), actual, self.auxiliary, self.classes)
            assert result['ordinary_budget_satisfied'], result
            return dict(comparison=result, disclosure=disclosure)
        result = {}
        for source, expected in self.refs.items():
            if self.family == 'gdn_bwd':
                result[source] = self.bf.comparison(actual, expected, self.dtype)
                assert self.bf.acceptable(result[source]), result[source]
                continue
            numbers = {}
            for name, target in expected.items():
                metrics = self.reference.metric(actual[name], target)
                if self.family.endswith('decode'):
                    limit = None if name == 'o' and self.dtype == torch.bfloat16 else 1e-4
                elif self.family == 'gdn_fwd':
                    limit = self.reference.budget(target, self.dtype)
                else:
                    limit = self.reference.budget(target, self.dtype, name)
                passed = metrics['finite'] and (limit is None or metrics['relative_l2'] <= limit)
                numbers[name] = dict(**metrics, budget=limit, passed=passed)
                assert passed, (name, numbers[name])
            result[source] = numbers
        return result

    def check(self, block_dim, output_dir, label):
        returned = self.call(block_dim)
        torch.npu.synchronize()
        assert len(returned) == len(self.names)
        for value in returned:
            assert value.device == self.device_inputs['q'].device
            assert all(source is None or value.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
                       for source in self.device_inputs.values()), 'Output aliases an input'
        actual = {n: x.cpu() for n, x in zip(self.names, returned)}
        try:
            comparison = self.compare(actual, output_dir, label)
            current = {n: digest(x) for n, x in self.device_inputs.items()}
            assert current == {n: self.input_hashes[n] for n in self.device_inputs}, 'Input mutation'
            result = dict(comparison=comparison, output_sha256={n: digest(x) for n, x in actual.items()},
                          input_sha256=self.input_hashes, inputs_unchanged=True)
            if self.family.endswith('decode') and self.dtype == torch.bfloat16:
                # Test-only companion widening occurs before upload and outside
                # timing. The measured invocation still receives BF16 storage.
                widened = {n: x.float() if n in ('q', 'k', 'v') else x for n, x in self.cpu.items()}
                device = {n: None if x is None else x.to('npu') for n, x in widened.items()}
                companion = self.call(block_dim, device)
                torch.npu.synchronize()
                companion = {n: x.cpu() for n, x in zip(self.names, companion)}
                numbers = {source: {n: self.reference.metric(companion[n], target[n]) for n in self.names}
                           for source, target in self.refs.items()}
                assert all(m['finite'] and m['relative_l2'] <= 1e-4 for group in numbers.values() for m in group.values())
                exact = {n: digest(actual[n]) == digest(companion[n].bfloat16() if n == 'o' else companion[n])
                         for n in self.names}
                assert all(exact.values()), exact
                result['same_input_fp32'] = dict(comparison=numbers, storage_byte_equal=exact)
            return result
        except Exception:
            torch.save(dict(inputs=self.cpu, actual=actual, references=self.refs),
                       Path(output_dir) / (label + '-failure.private.pt'))
            raise
