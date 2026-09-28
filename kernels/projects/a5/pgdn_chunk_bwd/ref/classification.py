"""PK-05 frozen-budget classifier and lossless disclosure records.

Classification accepts only independent FP64 analytical results, never a
candidate. PM issue93 comments5842101164 and5842100963 authorize this rule.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from .reference import NAMES

FACTOR = 64 * 2**-23
BUDGET = 1e-4
REASONS = {0: 'abs(reference64)<=64*2^-23*local_scale',
           1: 'normalization_radial_nullspace', 2: 'zero_initial_state',
           3: 'no_main_cotangent', 4: 'no_output_cotangent'}


def classify(auxiliary):
    result = {}
    for name in NAMES:
        value, scale = auxiliary['gradients'][name], auxiliary['scales'][name]
        reason = auxiliary['zero_reasons'][name]
        if value.dtype != torch.float64 or scale.dtype != torch.float64:
            raise ValueError('Classifier requires independent FP64 quantities')
        if value.shape != scale.shape or value.shape != reason.shape:
            raise ValueError('Malformed classifier shapes')
        if not bool(torch.isfinite(value).all() and torch.isfinite(scale).all() and (scale >= 0).all()):
            raise ValueError('Nonfinite/negative reference classifier quantities')
        if not bool(((reason >= 0) & (reason <= 4)).all()):
            raise ValueError('Unknown algebraic reason code')
        if not bool((value[reason != 0] == 0).all()):
            raise ValueError('Algebraic zero reference was not canonicalized')
        threshold = FACTOR * scale
        disclosure = (reason != 0) | (value.abs() <= threshold)
        result[name] = dict(disclosure=disclosure, ordinary=~disclosure,
                            threshold=threshold, reason=reason)
    return result


def ratio(numerator, denominator):
    if denominator == 0:
        return 0. if numerator == 0 else 'infinity'
    value = numerator / denominator
    return value if math.isfinite(value) else str(value)


def numbers(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError('Comparison shape mismatch')
    actual, expected = actual.double(), expected.double()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return dict(finite=False, relative_l2='nonfinite', max_abs='nonfinite', reference_norm='nonfinite')
    error = actual - expected
    norm = expected.norm().item()
    return dict(finite=True, relative_l2=ratio(error.norm().item(), norm),
                max_abs=error.abs().max().item() if error.numel() else 0., reference_norm=norm)


def compare(actual, references, classes):
    if set(actual) != set(NAMES):
        raise ValueError('Seven named gradient outputs required')
    gradients = {}
    total_disclosure = 0
    satisfied = True
    for name in NAMES:
        tensor = actual[name]
        ordinary = classes[name]['ordinary']
        if tensor.dtype != torch.float32 or tensor.shape != ordinary.shape:
            raise ValueError(f'{name}: expected FP32 and exact declared shape')
        count = int(classes[name]['disclosure'].sum())
        total_disclosure += count
        row = dict(ordinary_count=int(ordinary.sum()), disclosure_count=count,
                   finite=bool(torch.isfinite(tensor).all()), comparisons={})
        good = row['finite']
        for label, reference in references.items():
            metrics = numbers(tensor[ordinary], reference[name][ordinary])
            rel = metrics['relative_l2']
            accepted = metrics['finite'] and isinstance(rel, (int, float)) and rel <= BUDGET
            good = good and accepted
            row['comparisons'][label] = dict(ordinary=metrics, ordinary_budget_satisfied=accepted,
                                             raw_whole_gradient=numbers(tensor, reference[name]))
        row['ordinary_budget_satisfied'] = good
        satisfied = satisfied and good
        gradients[name] = row
    return dict(gradients=gradients, ordinary_budget_satisfied=satisfied,
                disclosure_count=total_disclosure,
                numerical_status='disclosure_required_nonpass' if total_disclosure else ('passed' if satisfied else 'failed'))


def write_disclosures(path, actual, auxiliary, classes):
    """Complete indexed JSONL, run-length encoded only for identical records.

    Each start/count range denotes every flat index in row-major order; shape
    gives its full coordinate. This is lossless, not a sample or a summary.
    Distinct signed-zero payloads are retained through JSON text equality.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total, runs = 0, 0
    with path.open('w') as stream:
        stream.write(json.dumps(dict(schema='pk05-disclosure-jsonl/1', ordering='row-major',
                                     encoding='start/count ranges of exactly identical values',
                                     multiplier=FACTOR, budget=BUDGET)) + '\n')
        for name in NAMES:
            mask = classes[name]['disclosure'].flatten()
            value = actual[name].detach().cpu().double().flatten()
            truth = auxiliary['gradients'][name].flatten()
            raw_truth = auxiliary['raw_gradients'][name].flatten()
            scale = auxiliary['scales'][name].flatten()
            threshold = classes[name]['threshold'].flatten()
            reason = classes[name]['reason'].flatten()
            norm = auxiliary['gradients'][name].norm().item()
            stream.write(json.dumps(dict(gradient=name, shape=list(actual[name].shape),
                                         full_reference64_norm=norm, disclosure_count=int(mask.sum()))) + '\n')
            indices = mask.nonzero().flatten()
            if indices.numel() == 0:
                continue
            # Find maximal exact runs in tensor operations. This avoids a
            # Python record per structural zero in full dA_T-only workloads.
            starts = torch.ones(indices.numel(), dtype=torch.bool)
            starts[1:] = indices[1:] != indices[:-1] + 1
            for column in (value, truth, raw_truth, scale, threshold):
                bits = column[indices].contiguous().view(torch.int64)
                starts[1:] |= bits[1:] != bits[:-1]
            codes = reason[indices]
            starts[1:] |= codes[1:] != codes[:-1]
            boundaries = starts.nonzero().flatten().tolist() + [indices.numel()]
            for begin, end in zip(boundaries, boundaries[1:]):
                index = int(indices[begin])
                got, ref = value[index].item(), truth[index].item()
                error = abs(got - ref)
                contribution = ratio(error, norm) if norm else ('undefined' if error == 0 else 'infinity')
                payload = dict(candidate=got, reference64=ref, raw_reference64=raw_truth[index].item(),
                               local_scale=scale[index].item(), threshold=threshold[index].item(),
                               absolute_output=abs(got), absolute_error=error,
                               global_norm_contribution=contribution, reason=REASONS[int(reason[index])])
                encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False)
                stream.write('{"start":%d,"count":%d,"values":%s}\n' % (index, end - begin, encoded))
                runs += 1
                total += end - begin
    return dict(file=path.name, elements=total, runs=runs,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
