"""Shared independent native evidence checks and byte-comparison controls."""
import hashlib
import json
from pathlib import Path

import torch

from .calibrate import cases
from .classification import numbers
from .reference import INPUTS, NAMES

def digest(tensor):
    host = tensor.detach().cpu().contiguous()
    return hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()


def internal_comparison(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('Stage schema mismatch')
    result = {}
    for name in expected:
        if actual[name].shape != expected[name].shape or actual[name].dtype != expected[name].dtype:
            raise ValueError('Stage shape/dtype mismatch: ' + name)
        if name in NAMES:
            continue  # Final gradients require the independently frozen class.
        result[name] = numbers(actual[name], expected[name])
        relative = result[name]['relative_l2']
        result[name]['budget_satisfied'] = (result[name]['finite'] and
                                            isinstance(relative, (int, float)) and relative <= 1e-4)
    return result


def compare_runs(left, right):
    case_set = left.get('case_set', 'original_frozen_256')
    if case_set != right.get('case_set', 'original_frozen_256'):
        raise ValueError('Mismatched case sets')
    if case_set == 'original_frozen_256':
        expected_cases = cases()
    elif case_set == 'supplemental_dense_clamp':
        from .clamp_cases import cases as clamp_cases
        expected_cases = clamp_cases()
    else:
        raise ValueError('Unknown case set')
    if {left['block_dim'], right['block_dim']} != {1, 2}:
        raise ValueError('Require separate block_dim1 and block_dim2 reports')
    if not all(r['all_required_checks_complete'] for r in (left, right)):
        raise ValueError('Incomplete native qualification')
    a = {r['case']['id']: r for r in left['cases']}
    b = {r['case']['id']: r for r in right['cases']}
    if len(a) != len(left['cases']) or len(b) != len(right['cases']) or set(a) != set(b):
        raise ValueError('Duplicate or mismatched case coverage')
    if set(a) != {c['id'] for c in expected_cases}:
        raise ValueError('The complete frozen or supplemental grid is required')
    schema = json.loads((Path(__file__).resolve().parents[1] / 'contract.json').read_text())
    for report in (a, b):
        for row in report.values():
            if (set(row['stage_sha256']) != set(schema['stages']) or
                    set(row['public_sha256']) != set(NAMES) or set(row['public_schema']) != set(NAMES) or
                    set(row['cpu_input_sha256']) != set(INPUTS) | {'do', 'dht', 'dA_T'}):
                raise ValueError('Incomplete tensor hash/schema coverage')
    rows = []
    for key, x in a.items():
        y = b[key]
        checks = {field: x[field] == y[field] for field in
                  ('case', 'cpu_input_sha256', 'stage_sha256', 'public_sha256', 'public_schema')}
        rows.append(dict(case=key, checks=checks, byte_identical=all(checks.values())))
    return dict(case_set=case_set, cases=rows, all_byte_identical=all(r['byte_identical'] for r in rows),
                numerical_status='numerical_disclosure_complete_nonpass',
                scope='Returned stage and public tensor hashes, equal runtime-generated inputs; no numerical PASS claim')
