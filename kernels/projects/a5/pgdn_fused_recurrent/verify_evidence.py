"""Recompute published evidence invariants; this does not execute a kernel."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'evidence'


def load(name):
    return json.loads((ROOT / name).read_text())


def metric(row, budget=True):
    assert row['finite']
    for field in ('reference_norm', 'error_norm', 'max_abs', 'relative_l2'):
        assert math.isfinite(row[field]) and row[field] >= 0, field
    norm, error = row['reference_norm'], row['error_norm']
    expected = error / norm if norm else 0.0
    assert norm or error == 0.0, 'Zero reference requires zero residual'
    assert math.isclose(row['relative_l2'], expected, rel_tol=1e-12, abs_tol=1e-30)
    if budget:
        assert expected <= 1e-4, row


def comparison(row):
    for oracle in ('A', 'B'):
        assert set(row['comparison']['metrics'][oracle]) == {'o', 'final_state', 'final_A_state'}
        for name, values in row['comparison']['metrics'][oracle].items():
            metric(values, name != 'o' or row['case']['dtype'] == 'float32')


def chain(calls):
    for previous, current in zip(calls, calls[1:]):
        assert previous['stop'] == current['start']
        for initial, final in (('initial_state', 'final_state'), ('initial_A_state', 'final_A_state')):
            assert previous['returned_sha256'][final] == current['initial_sha256'][initial]


def verify():
    index = load('index.json')
    for stage in index['stages']:
        if stage['historical']:
            continue
        receipt = load(f"native/{stage['label']}/receipt.json")
        assert receipt['checks_complete'] and receipt['returncode'] == 0
        assert receipt['artifact_hashes_before'] and receipt['artifact_hashes_after']
        assert receipt['shared_lock_held'] and all(receipt['healthy'].values())
        context = receipt['context_observations']
        assert context['before_users'] == context['after_users'] == context['verified_max_foreign'] == 0
        manifest = load(receipt['source_manifest'])
        repo = Path(__file__).resolve().parents[4]
        for relative, expected in manifest['repo'].items():
            assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == expected, relative
    grids = [load(f'native/grid-bd{bd}-final/results.json') for bd in (1, 2, 4, 8, 16, 28)]
    baseline = grids[0]['records']
    maximum = {key: 0.0 for key in ('o', 'final_state', 'final_A_state')}
    for grid in grids:
        assert len(grid['records']) == grid['cases'] == 126
        for original, row in zip(baseline, grid['records']):
            assert original['case'] == row['case']
            assert original['input_sha256'] == row['input_sha256']
            assert original['output_sha256'] == row['output_sha256']
            comparison(row)
            if row['case']['dtype'] == 'bfloat16':
                assert all(row['bf16_storage'].values()) and len(row['bf16_storage']) == 3
            for oracle in row['comparison']['metrics'].values():
                for name, values in oracle.items():
                    if name != 'o' or row['case']['dtype'] == 'float32':
                        maximum[name] = max(maximum[name], values['relative_l2'])
    integration = load('native/integration-bd2-final/results.json')
    assert len(integration['integration_cases']) == 14
    for row in integration['integration_cases']:
        assert {c['width'] for c in row['continuations']} == {1, 3, 7, 16}
        for continuation in row['continuations']:
            chain(continuation['calls'])
            for call in continuation['calls']:
                for values in call['vs_CPU_B_states'].values():
                    metric(values)
            for source in ('vs_whole_chunk', 'vs_CPU_B'):
                values = continuation[source]
                for name in ('final_state', 'final_A_state'):
                    metric(values[name])
                assert len(values['tokens']) == 64
                for token in values['tokens']:
                    metric(token, row['case']['dtype'] == 'float32')
    assert len(integration['split_cases']) == 150
    for row in integration['split_cases']:
        chain(row['calls'])
        assert row['passed']
        # The split trace records every output independently, including both states.
        assert all(row['trace']['byte_identical'].values())
        for values in row['trace']['tokens'] + [row['trace']['final_state'], row['trace']['final_A_state']]:
            metric(values)
            assert values['error_norm'] == 0.0
    timing = load('native/timing-bd4-final/results.json')
    sample_count = 0
    for row in timing['records']:
        for result in row['correctness'].values():
            comparison(dict(case=row['case'], comparison=result))
        assert len(row['rounds']) == 3
        for round_ in row['rounds']:
            for leg in ('baseline_before', 'candidate', 'baseline_after'):
                values = round_[leg]
                assert len(values['samples_us']) == 50
                assert all(math.isfinite(x) and x > 0 for x in values['samples_us'])
                assert statistics.median(values['samples_us']) == values['median_us']
                sample_count += len(values['samples_us'])
    assert sample_count == 1800
    audit = load('native/host-audit-bd4-final/results.json')
    assert len(audit['records']) == 8
    for row in audit['records']:
        comparison(row)
        assert not row['unexpected'] and row['output_flag_checked']
        for key in ('operations', 'output_final_state_false_operations'):
            assert all(v['operator'] == 'aten.empty.memory_format' for v in row[key])
    for label, count in [('loop-bounds-bd4-final', 32), ('precision-boundaries-bd4-final', 4)]:
        data = load(f'native/{label}/results.json')
        assert len(data['records']) == count
        for row in data['records']:
            comparison(row)
    failure = load('native/atk-underflow-bd1-v1/results.json')
    assert failure['comparison']['metrics']['A']['final_A_state']['relative_l2'] == 1.0
    repaired = next(row for row in baseline if row['case']['id'] == failure['case']['id'])
    assert repaired['input_sha256'] == failure['input_sha256']
    assert repaired['comparison']['metrics']['A']['final_A_state']['error_norm'] == 0
    return dict(kind='Published-record audit, not new device execution', grid_cases=756,
                bf16_companions=378, max_budgeted_relative_l2=maximum,
                prefill_cases=14, continuations=56, splits=150, timing_samples=sample_count,
                host_audit_cases=8, loop_bound_cases=32, extra_precision_cases=4,
                original_failure_retained=True, passed=True)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
