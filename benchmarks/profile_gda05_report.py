"""Audit private run directories and render the GDA-05 measurement report.

The manifest is ignored external configuration: a list of objects with
``directory`` (collected run root), ``execution`` (private launch receipt),
``compile_receipt`` (separate build receipt), and ``device_group`` (anonymous
label). Machine fields never enter the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

from profile_gda05_workloads import FAMILIES, FLA_PIN, ORACLE_SHA256, block_dims, cases


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    assert isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def audit_isolation(directory, execution, *, completion_marker='PROFILE_COMPLETE'):
    """Require empty endpoints and qualified, positive in-run driver samples."""
    assert not (directory / 'isolation-stop.json').exists(), 'Isolation monitor rejected run'
    snapshots = [read(directory / 'host-contexts-before.json')]
    snapshots += [json.loads(line) for line in (directory / 'host-contexts-during.jsonl').read_text().splitlines()]
    snapshots += [read(directory / 'host-contexts-after.json')]
    assert len(snapshots) >= 3 and not snapshots[0]['users'] and not snapshots[-1]['users']
    owned_svm = owned_trs = retired_contexts = 0
    history = {}
    for row in snapshots:
        assert row['device'] == execution['device'] and not row['foreign']
        assert all(user['owned'] for user in row['users'])
        current_history = {}
        for user in row['users']:
            command = user['command']
            pid = user['pid']
            registries = sorted(x['registry'] for name in ('svm', 'trs')
                                for x in row[name] if x['pid'] == pid)
            if user.get('ownership_basis') == 'previous_live_context':
                # Independently reconstruct the preceding live observation.
                # A disappearing process alone is never evidence of ownership.
                previous = history[pid]
                assert user['process_exited'] and command in ('PROCESS_EXIT_RACE', '')
                assert user['previous_live'] == previous
                assert registries and set(registries) <= set(previous['registries'])
                assert 0 <= row['time'] - previous['time'] <= 10
                command = previous['command']
                current_history[pid] = previous
                retired_contexts += 1
            else:
                assert 'ownership_basis' not in user
                current_history[pid] = dict(time=row['time'], command=command,
                                           namespace_pid=user['namespace_pid'], registries=registries)
            assert '/benchmarks/profile_gda05_run.py' in command and '--prepare-only' not in command
            assert '--family ' + execution['family'] in command
            assert '--block-dim ' + str(execution['block_dim']) in command
            assert execution['output'] + '/run' in command
        history = current_history
        owned_svm += bool(row['svm'])
        owned_trs += bool(row['trs'])
    assert owned_svm and owned_trs, 'Missing positive control in driver context registries'
    gaps = [right['time'] - left['time'] for left, right in zip(snapshots, snapshots[1:])]
    assert all(0 <= gap <= 10 for gap in gaps), 'Isolation monitor has an excessive sampling gap'
    for name in ('health-before.log', 'health-after.log'):
        health = (directory / name).read_text()
        assert re.search(r'\|\s*health\s*\|\s*Healthy\s*\|', health)
        assert 'Ascend 950PR' in health and int(re.search(r'AI Core Count\s*\|\s*(\d+)', health)[1]) >= 28
    log = (directory / 'native.log').read_text()
    assert completion_marker in log and not re.search(r'Traceback|\[ERROR\]', log)
    warnings = [line for line in log.splitlines() if re.search(r'Warning|\[WARNING\]', line)]
    notice = ('The current CANN and Soc versions require processing for 32 padding size, '
              'with memory allocation.')
    # Located in AddPadSize/round_size at the package-reported owner revision.
    # This notice announces allocator capacity padding, not a kernel failure.
    # Keep it counted and disclosed; every other warning requires investigation.
    assert all('NPUCachingAllocator.cpp:201]' in line and
               line.endswith('Warning: ' + notice + ' (function operator())') for line in warnings)
    return dict(empty_endpoints=True, foreign_contexts=0, snapshots=len(snapshots),
                positive_svm_snapshots=owned_svm, positive_trs_snapshots=owned_trs,
                max_sample_gap_s=max(gaps), health_before='Healthy', health_after='Healthy',
                retired_owned_context_observations=retired_contexts,
                allocator_padding_notice_count=len(warnings))


def audit_check(check, family):
    assert check['inputs_unchanged']
    for value in (*check['input_sha256'].values(), *check['output_sha256'].values()):
        assert value is None or re.fullmatch('[0-9a-f]{64}', value)
    comparison = check['comparison']
    if family == 'pgdn_bwd':
        disclosure = comparison['disclosure']
        comparison = comparison['comparison']
        assert disclosure['elements'] == comparison['disclosure_count']
        assert comparison['disclosure_count'] == sum(
            gradient['disclosure_count'] for gradient in comparison['gradients'].values())
        assert comparison['ordinary_budget_satisfied']
        for gradient in comparison['gradients'].values():
            assert gradient['finite'] and gradient['ordinary_budget_satisfied']
            for reference in gradient['comparisons'].values():
                assert reference['ordinary_budget_satisfied']
                assert reference['ordinary']['finite'] and reference['ordinary']['relative_l2'] <= 1e-4
        assert comparison['numerical_status'] == ('disclosure_required_nonpass'
               if comparison['disclosure_count'] else 'passed')
    else:
        assert set(comparison) == {'A', 'B'}
        for reference in comparison.values():
            for metric in reference.values():
                assert metric['finite'] and metric['passed']
                finite(metric['relative_l2'])
                assert metric['budget'] is None or metric['relative_l2'] <= metric['budget']
    if 'same_input_fp32' in check:
        companion = check['same_input_fp32']
        assert all(companion['storage_byte_equal'].values())
        assert all(m['finite'] and m['relative_l2'] <= 1e-4
                   for group in companion['comparison'].values() for m in group.values())


def audit_disclosure(directory, check):
    """Verify complete, ordered index ranges against the numerical classification."""
    from kernels.projects.a5.pgdn_chunk_bwd.ref.classification import BUDGET, FACTOR, REASONS
    from kernels.projects.a5.pgdn_chunk_bwd.ref.reference import NAMES

    metadata = check['comparison']['disclosure']
    gradients = check['comparison']['comparison']['gradients']
    file = (directory / metadata['file']).resolve()
    assert file.is_relative_to(directory.resolve())
    assert sha(file) == metadata['sha256'] and file.stat().st_size == metadata['bytes']
    lines = file.read_text().splitlines()
    header = json.loads(lines[0])
    assert header == dict(schema='pk05-disclosure-jsonl/1', ordering='row-major',
                          encoding='start/count ranges of exactly identical values',
                          multiplier=FACTOR, budget=BUDGET)
    counts, runs, name, end, limit = {}, 0, None, 0, 0
    for line in lines[1:]:
        record = json.loads(line)
        if 'gradient' in record:
            name = record['gradient']
            assert name in NAMES and name not in counts
            counts[name], end = 0, 0
            assert all(type(size) is int and size > 0 for size in record['shape'])
            limit = math.prod(record['shape'])
            assert limit == gradients[name]['ordinary_count'] + gradients[name]['disclosure_count']
            assert record['disclosure_count'] == gradients[name]['disclosure_count']
            finite(record['full_reference64_norm'])
        else:
            assert name is not None and set(record) == {'start', 'count', 'values'}
            start, count = record['start'], record['count']
            assert type(start) is int and type(count) is int
            assert start >= end and count > 0 and start + count <= limit
            values = record['values']
            assert values['reason'] in REASONS.values()
            for field in ('candidate', 'reference64', 'raw_reference64'):
                assert math.isfinite(values[field])
            for field in ('local_scale', 'threshold', 'absolute_output', 'absolute_error'):
                finite(values[field])
            assert values['absolute_output'] == abs(values['candidate'])
            assert values['absolute_error'] == abs(values['candidate'] - values['reference64'])
            counts[name] += count
            end = start + count
            runs += 1
    assert list(counts) == list(NAMES)
    assert all(counts[name] == gradients[name]['disclosure_count'] for name in NAMES)
    assert sum(counts.values()) == metadata['elements'] and runs == metadata['runs']
    return lines


def audit_measurement(report):
    identity = report['identity']
    assert identity['fla_pin'] == FLA_PIN and identity['oracle_sha256'] == ORACLE_SHA256
    family, bd = identity['family'], identity['block_dim']
    assert family in FAMILIES and bd in block_dims(family)
    assert report['schema'] == 'gda05-profile/1' and report['completed']
    assert [report[k] for k in ('warmup', 'repeat', 'rounds', 'event_repeat', 'profiler_repeat')] == [10, 20, 3, 20, 3]
    assert report['profiler_schedule'] == dict(wait=0, warmup=1, active=1, repeat=1)
    assert report['profiler_config'] == dict(level='Level1', aic_metrics='AiCoreNone')
    assert report['profiler_processes'] == 'one fresh process per capture'
    assert [row['case'] for row in report['cases']] == cases(family), 'Incomplete or reordered workload matrix'
    for row in report['cases']:
        for stage in ('before', 'after'):
            audit_check(row[stage], family)
            if family.endswith('decode') and row['case']['dtype'] == 'bfloat16':
                assert 'same_input_fp32' in row[stage], 'Missing same-input FP32 companion'
        for key in ('input_sha256', 'output_sha256'):
            assert row['before'][key] == row['after'][key], 'Input/output drift within run'
        if not row['case']['measure']:
            assert not row['segments'] and not row.get('events') and not row.get('profiles')
            continue
        assert [(s['round'], s['name']) for s in row['segments']] == [
            (r, n) for r in (1, 2, 3) for n in ('baseline_before', 'baseline_middle', 'baseline_after')]
        for segment in row['segments']:
            assert len(segment['samples']) == 20
            for sample in segment['samples']:
                assert set(sample) == {'host_call_ms', 'sync_wait_ms', 'wall_ms', 'calling_thread_cpu_ms'}
                for value in sample.values():
                    finite(value)
                assert abs(sample['wall_ms'] - sample['host_call_ms'] - sample['sync_wait_ms']) <= 1e-9
        assert len(row['events']) == 20 and len(row['profiles']) == 3
        for event in row['events']:
            assert set(event) == {'event_interval_ms', 'instrumented_wall_ms'}
            for value in event.values():
                finite(value)
                assert value > 0
        for profile in row['profiles']:
            assert profile['kernels'] and profile['native_kernel_counts']
            assert all(count == 1 for count in profile['native_kernel_counts'].values()), 'Expected one call per native entry'
            observed = {entry: sum(k.get('native_entry') == entry for k in profile['kernels'])
                        for entry in profile['native_kernel_counts']}
            assert observed == profile['native_kernel_counts'], 'Native attribution count differs from kernel rows'
            assert all(int(k['block_num']) == bd for k in profile['kernels'] if k.get('native_entry')), 'Native trace block count differs from requested bd'
            assert 0 < profile['interval_union_us'] <= profile['duration_sum_us'] + 1e-6
            assert abs(sum(k['duration_us'] for k in profile['kernels']) - profile['duration_sum_us']) <= 1e-6
            for kernel in profile['kernels']:
                finite(kernel['start_us'])
                finite(kernel['duration_us'])
            assert set(profile['host_markers_us']) == {'gda05_public_call', 'gda05_completion_wait'}
    cache = report['cache']
    assert cache['cold_paths_rejected'] and cache['bindings'] == {} and cache['key_cardinality'] == 5
    assert [s['entries'] for s in cache['scales']] == [1, 64, 4096]
    measurements = [cache['empty_call'], *cache['public'].values()]
    measurements += [s[key] for s in cache['scales'] for key in ('signature', 'process')]
    for value in measurements:
        assert value['iterations'] == 10000 and value['batches'] == len(value['ns_per_call']) == 20
        for sample in value['ns_per_call']:
            finite(sample)
    for value in cache['public'].values():
        assert value['hit_delta'] == 200000 and value['miss_delta'] == 0


def audit_captures(directory, report):
    """Reparse each private capture and verify its artifact, bytes and full control."""
    import torch
    from profile_gda05_run import parse_profile
    from profile_gda05_workloads import selected_entries

    root = (directory / 'run').resolve()
    identity = report['identity']
    assert sys.version.split()[0] == identity['python'], \
        'Reparse captures with the recorded Python version (float sum changed in Python 3.12)'
    family, bd = identity['family'], identity['block_dim']
    op_names = {entry['name']: entry['op_name'] for entry in identity['entries']}
    for row in report['cases']:
        for profile in row.get('profiles', []):
            receipt = (root / profile['capture_receipt']).resolve()
            assert receipt.is_relative_to(root) and sha(receipt) == profile['capture_receipt_sha256']
            capture = read(receipt)
            assert capture['completed'] and capture['case'] == row['case']
            assert capture['identity'] == identity
            assert capture['profiler_schedule'] == report['profiler_schedule']
            assert capture['profiler_config'] == report['profiler_config']
            for stage in ('before', 'after'):
                audit_check(capture[stage], family)
                if family == 'pgdn_bwd':
                    audit_disclosure(receipt.parent / row['case']['id'], capture[stage])
                for key in ('input_sha256', 'output_sha256'):
                    assert capture[stage][key] == row['before'][key]
            control = next(c for c in cases(family) if c['dtype'] == row['case']['dtype'])
            assert capture['control'] == control
            if control != row['case']:
                audit_check(capture['control_check'], family)
                if family == 'pgdn_bwd':
                    audit_disclosure(receipt.parent / control['id'], capture['control_check'])
                original = next(r for r in report['cases'] if r['case'] == control)
                for key in ('input_sha256', 'output_sha256'):
                    assert capture['control_check'][key] == original['before'][key]
            expected = {entry.name: op_names[entry.name]
                        for entry in selected_entries(family, getattr(torch, row['case']['dtype']))}
            parsed = parse_profile(receipt.parent / row['case']['id'] / 'profile-private', expected, bd)
            assert parsed == capture['profile']
            assert parsed == {key: value for key, value in profile.items()
                              if key not in ('capture_receipt', 'capture_receipt_sha256')}


def load_manifest(path):
    entries, results, bindings = read(path), {}, {}
    for entry in entries:
        directory = Path(entry['directory'])
        report = read(directory / 'run/measurement.json')
        execution = read(entry['execution'])
        prepared = read(entry['compile_receipt'])
        audit_measurement(report)
        audit_captures(directory, report)
        for field in ('source_commit', 'family', 'block_dim', 'python', 'torch', 'torch_git',
                      'ascriptor', 'torch_npu', 'artifact_sha256', 'entries'):
            assert report['identity'][field] == prepared[field], ('Compile receipt mismatch', field)
        key = report['identity']['family'], report['identity']['block_dim']
        assert key == (execution['family'], execution['block_dim']) and key not in results
        group = entry['device_group']
        assert re.fullmatch(r'group-[a-z]+', group), 'Use an anonymous device group label'
        # A caller must not relabel different physical devices as the same group.
        binding = (execution.get('board_id', 'primary'), execution['device'])
        assert bindings.setdefault(group, binding) == binding
        audit = audit_isolation(directory, execution)
        disclosures = {}
        if key[0] == 'pgdn_bwd':
            for row in report['cases']:
                for stage in ('before', 'after'):
                    disclosures[row['case']['id'] + '/' + stage] = audit_disclosure(
                        directory / 'run' / row['case']['id'], row[stage])
        results[key] = dict(report=report, isolation=audit, device_group=group,
                            measurement_sha256=sha(directory / 'run/measurement.json'),
                            compile_receipt_sha256=sha(entry['compile_receipt']), disclosures=disclosures)
    expected = {(f, bd) for f in FAMILIES for bd in block_dims(f)}
    assert set(results) == expected, ('Incomplete family/block_dim matrix', expected - set(results))
    for family in FAMILIES:
        baseline = results[family, 1]
        for bd in block_dims(family):
            current = results[family, bd]
            assert current['device_group'] == baseline['device_group'], 'Family changed physical device'
            base_identity, identity = baseline['report']['identity'], current['report']['identity']
            assert identity['loaded_source_sha256'] == base_identity['loaded_source_sha256'], 'Sources changed across bd'
            if bd != 1:
                base_signatures = {e['name']: e['signature'] for e in base_identity['entries']}
                assert all(e['signature'] != base_signatures[e['name']] for e in identity['entries']), 'Build reused across bd'
            for row, reference in zip(current['report']['cases'], baseline['report']['cases']):
                for field in ('input_sha256', 'output_sha256'):
                    assert row['before'][field] == reference['before'][field], ('Cross-bd bytes differ', family, bd, field)
    identities = [r['report']['identity'] for r in results.values()]
    for key in ('source_commit', 'python', 'torch', 'torch_git', 'ascriptor', 'torch_npu'):
        assert len({identity[key] for identity in identities}) == 1, ('Runtime identity differs', key)
    return results


def summary(results):
    rows = []
    for (family, bd), result in results.items():
        for row in result['report']['cases']:
            if not row['case']['measure']:
                continue
            samples = [sample for segment in row['segments'] for sample in segment['samples']]
            values = [statistics.median(v[key] for v in samples)
                      for key in ('wall_ms', 'host_call_ms', 'sync_wait_ms', 'calling_thread_cpu_ms')]
            values += [statistics.median(x['event_interval_ms'] for x in row['events'])]
            values += [statistics.median(p[key] for p in row['profiles']) / 1000
                       for key in ('duration_sum_us', 'interval_union_us')]
            rows.append('| ' + ' | '.join([family, result['device_group'], str(bd), row['case']['id'],
                                          *(f'{value:.6f}' for value in values)]) + ' |')
    assert len(rows) == 76
    return '\n'.join(rows)


def stage_table(report):
    """Expose native stages without assigning built-in work to inferred stages."""
    lines = ['| case | native entry / 其余设备算子 | capture 1 ms | capture 2 ms | capture 3 ms | 占 kernel 总和 %（中位数） |',
             '|---|---|---:|---:|---:|---:|']
    for row in report['cases']:
        if not row['case']['measure']:
            continue
        profiles = row['profiles']
        stages = list(profiles[0]['native_kernel_counts']) + [None]
        for stage in stages:
            durations = [sum(k['duration_us'] for k in p['kernels']
                             if k.get('native_entry') == stage) for p in profiles]
            shares = [100 * duration / p['duration_sum_us']
                      for duration, p in zip(durations, profiles)]
            lines.append('| ' + ' | '.join([
                row['case']['id'], stage or '其余设备算子合计',
                *(f'{value / 1000:.6f}' for value in durations),
                f'{statistics.median(shares):.6f}']) + ' |')
    return '\n'.join(lines)


def cache_table(report):
    cache = report['cache']
    rows = [('空调用循环', '—', cache['empty_call'])]
    rows += [('公共 ' + name, '固定 bd', values) for name, values in cache['public'].items()]
    rows += [(name, str(scale['entries']), scale[name])
             for scale in cache['scales'] for name in ('signature', 'process')]
    lines = ['| 热路径 | 缓存条目数 | ns/call（20 批中位数） |', '|---|---:|---:|']
    lines += [f'| {name} | {size} | {statistics.median(values["ns_per_call"]):.4f} |'
              for name, size, values in rows]
    return '\n'.join(lines)


def render(results, introduction):
    lines = [introduction.rstrip(), '',
             '| 实现 | 设备组 | bd | case | 总 wall ms | 调用返回 ms | 等待 ms | 调用线程 CPU ms | event ms | kernel 总和 ms | kernel 并集 ms |',
             '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|', summary(results), '',
             '以下保留每个 case 的全部原始样本、数值比较、输入/输出哈希与编译身份。未删除离群值。',
             'PGDN 反向的 `disclosure_required_nonpass` 保持非 PASS，并附完整逐元素披露。', '']
    for (family, bd), result in results.items():
        lines += [f'<details><summary>{family}, bd={bd}: 阶段耗时与热缓存</summary>', '',
                  '阶段列为各次 profiler 的设备 duration；占比按各次 kernel 总和计算后取中位数。',
                  '中位数占比之和不保证恰为 100%；不含 host 间隙，不代表总 wall 占比。',
                  '其余设备算子按实际记录合计，不将其推定归入某个 native 阶段。', '',
                  stage_table(result['report']), '', cache_table(result['report']), '', '</details>', '']
        lines += [f'<details><summary>{family}, bd={bd}: 原始测量与审计</summary>', '', '```json',
                  json.dumps({k: v for k, v in result.items() if k != 'disclosures'},
                             ensure_ascii=False, separators=(',', ':'), allow_nan=False), '```', '', '</details>', '']
        for case, disclosure in result['disclosures'].items():
            lines += [f'<details><summary>{family}, bd={bd}, {case}: 完整非 PASS 披露</summary>',
                      '', '```jsonl', *disclosure, '```', '', '</details>', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--introduction', required=True, type=Path, help='Reviewed report prose, without machine fields')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    results = load_manifest(args.manifest)
    args.output.write_text(render(results, args.introduction.read_text()))
    print('REPORT_COMPLETE', len(results), 'isolated runs; 76 timed cases; 90 numeric case checks')


if __name__ == '__main__':
    main()
