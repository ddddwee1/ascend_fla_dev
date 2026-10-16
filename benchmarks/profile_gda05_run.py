"""Measure one existing GDN/PGDN family and one block_dim per process.

The caller supplies device isolation, the shared lock, fresh health/occupancy
checks, pinned sources and a private build/output directory. No device is
selected from a machine-specific default. Raw profiler files stay private.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from profile_gda05_cache import cache_profile
from profile_gda05_workloads import FAMILIES, FLA_PIN, ORACLE_SHA256, Workload, block_dims, cases, kernel_pairs, module, selected_entries


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def artifacts(pairs):
    values = {}
    for entry, compiled in pairs:
        files = [compiled.out_dir / '.source_hash', *compiled.vendor_dir.rglob('*')]
        values[entry.name] = {
            str(p.relative_to(compiled.out_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}
        assert values[entry.name], 'Empty compiled artifact inventory'
    return values


def oracle_identity():
    values = {name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
              for name in ORACLE_SHA256}
    assert values == ORACLE_SHA256, 'CPU oracle files differ from the selected FLA pin'
    return values


def identity(args, pairs):
    repo = Path(__file__).resolve().parents[1]
    loaded = {}
    for imported in tuple(sys.modules.values()):
        filename = getattr(imported, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_file() and path.is_relative_to(repo):
            loaded[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    import ascriptor
    return dict(source_commit=args.source_commit, family=args.family, block_dim=args.block_dim,
                python=sys.version.split()[0], torch=torch.__version__, torch_git=torch.version.git_version,
                ascriptor=ascriptor.__version__, torch_npu=importlib.metadata.version('torch-npu'),
                fla_pin=FLA_PIN, oracle_sha256=oracle_identity(),
                loaded_source_sha256=loaded, artifact_sha256=artifacts(pairs),
                entries=[dict(name=e.name, op_name=c.op.op_name, signature=c.signature) for e, c in pairs])


def plain_sample(call):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    cpu_start = time.thread_time_ns()
    returned = call()
    cpu_stop = time.thread_time_ns()
    submitted = time.perf_counter_ns()
    torch.npu.synchronize()
    stopped = time.perf_counter_ns()
    # Keep outputs alive until completion; release outside the measured interval.
    del returned
    return dict(host_call_ms=(submitted-start)/1e6, sync_wait_ms=(stopped-submitted)/1e6,
                wall_ms=(stopped-start)/1e6, calling_thread_cpu_ms=(cpu_stop-cpu_start)/1e6)


def event_sample(call):
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    torch.npu.synchronize()
    start.record()
    wall_start = time.perf_counter_ns()
    returned = call()
    end.record()
    end.synchronize()
    wall_stop = time.perf_counter_ns()
    value = float(start.elapsed_time(end))
    assert math.isfinite(value) and value > 0, 'Invalid device event interval'
    del returned
    return dict(event_interval_ms=value, instrumented_wall_ms=(wall_stop-wall_start)/1e6)


def union_us(intervals):
    total, end = 0., -math.inf
    for begin, finish in sorted(intervals):
        assert math.isfinite(begin) and math.isfinite(finish) and finish >= begin
        total += max(0., finish-max(begin, end))
        end = max(end, finish)
    return total


def parse_profile(directory, expected_ops, block_dim=None):
    files = list(Path(directory).rglob('kernel_details.csv'))
    traces = list(Path(directory).rglob('trace_view.json'))
    assert len(files) == len(traces) == 1, 'Expected exactly one complete profiler export'
    kernels = []
    with files[0].open(newline='') as stream:
        for row in csv.DictReader(stream):
            duration, begin = float(row['Duration(us)']), Decimal(row['Start Time(us)'])
            assert math.isfinite(duration) and duration > 0 and begin.is_finite()
            kernels.append(dict(name=row['Name'], type=row['Type'], core=row['Accelerator Core'],
                                duration_us=duration, start_us=begin, block_num=row['Block Num']))
    assert kernels, 'No device kernels captured; wall-time subtraction is not a substitute'
    normalize = lambda name: ''.join(c.lower() for c in name if c.isalnum())
    matched = {entry: 0 for entry in expected_ops}
    for row in kernels:
        names = [normalize(row[key]) for key in ('name', 'type')]
        matches = [entry for entry, op in expected_ops.items()
                   if any(value.startswith(normalize(op)) for value in names)]
        assert len(matches) <= 1, 'Ambiguous native kernel attribution'
        row['native_entry'] = matches[0] if matches else None
        if matches:
            matched[matches[0]] += 1
    assert all(count == 1 for count in matched.values()), ('Expected one call per native entry', matched)
    if block_dim is not None:
        assert all(int(row['block_num']) == block_dim for row in kernels if row['native_entry']), \
            'Native trace block count differs from requested bd'
    # Raw trace can include machine identifiers. Only explicit marker durations
    # and whitelisted kernel fields are returned to the shareable report.
    trace = json.loads(traces[0].read_text())
    events = trace if isinstance(trace, list) else trace['traceEvents']
    markers = {name: [float(e['dur']) for e in events if e.get('ph') == 'X' and e.get('name') == name]
               for name in ('gda05_public_call', 'gda05_completion_wait')}
    assert all(len(values) == 1 for values in markers.values()), 'Missing or duplicate host markers'
    assert all(math.isfinite(v[0]) and v[0] >= 0 for v in markers.values()), 'Invalid host marker duration'
    origin = min(row['start_us'] for row in kernels)
    for row in kernels:
        row['start_us'] = float(row['start_us'] - origin)
    intervals = [(row['start_us'], row['start_us']+row['duration_us']) for row in kernels]
    return dict(kernels=kernels, native_kernel_counts=matched, duration_sum_us=sum(row['duration_us'] for row in kernels),
                interval_union_us=union_us(intervals), host_markers_us={k:v[0] for k,v in markers.items()},
                raw_trace_sha256=hashlib.sha256(traces[0].read_bytes()).hexdigest())


def profile_sample(call, output, expected_ops, block_dim):
    import torch_npu
    from torch.profiler import record_function
    activities = (torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU)
    assert set(activities) <= torch_npu.profiler.supported_activities()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=activities,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(output), worker_name='gda05', analyse_flag=True, async_mode=False),
        record_shapes=False, profile_memory=False, with_stack=False,
        # Level1 collects additional CANN task metadata; counters stay disabled.
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone),
    ) as profiler:
        warmup = call()
        torch.npu.synchronize()
        del warmup
        profiler.step()
        with record_function('gda05_public_call'):
            returned = call()
        with record_function('gda05_completion_wait'):
            torch.npu.synchronize()
        profiler.step()
    del returned
    return parse_profile(output, expected_ops, block_dim)


def isolated_profile(args, pairs, row, previous_rows, directory, index):
    """Capture in a new process, then require the parent's exact artifact/data."""
    output = directory / f'capture-{index}'
    command = [sys.executable, str(Path(__file__).resolve()), '--family', args.family,
               '--block-dim', str(args.block_dim), '--source-commit', args.source_commit,
               '--prepared-receipt', str(args.prepared_receipt), '--output', str(output),
               '--capture-case', row['case']['id']]
    if args.stop_file is not None:
        command += ['--stop-file', str(args.stop_file)]
    torch.npu.synchronize()
    subprocess.run(command, check=True)  # Child diagnostics remain in the native log.
    receipt = output / 'capture.json'
    capture = json.loads(receipt.read_text())
    assert capture['completed'] and capture['case'] == row['case']
    current = identity(args, pairs)
    assert capture['identity'] == current, 'Capture process used a different source/runtime/artifact'
    for stage in ('before', 'after'):
        for key in ('input_sha256', 'output_sha256'):
            assert capture[stage][key] == row['before'][key], 'Capture process input/output differs'
    control = next(case for case in cases(args.family) if case['dtype'] == row['case']['dtype'])
    assert capture['control'] == control
    if control != row['case']:
        parent_control = next(r for r in previous_rows if r['case'] == control)
        for key in ('input_sha256', 'output_sha256'):
            assert capture['control_check'][key] == parent_control['before'][key]
    return dict(**capture['profile'], capture_receipt=str(receipt.relative_to(args.output)),
                capture_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())


def run(args):
    def isolation_gate():
        if args.stop_file is not None and args.stop_file.exists():
            raise RuntimeError('External isolation monitor rejected this run; retain raw evidence')

    isolation_gate()
    oracle_identity()
    assert args.block_dim in block_dims(args.family)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if not args.prepare_only:
        import torch_npu  # noqa: F401 - register the actual backend
    pairs = kernel_pairs(args.family, args.block_dim)
    prepared = identity(args, pairs)
    if args.prepare_only:
        write_json(args.output/'compile-receipt.json', prepared)
        print('COMPILE_COMPLETE', args.family, args.block_dim, len(pairs), flush=True)
        return
    assert args.prepared_receipt is not None, 'A separately verified build receipt is required'
    receipt = json.loads(args.prepared_receipt.read_text())
    for field in ('source_commit', 'family', 'block_dim', 'python', 'torch', 'torch_git', 'ascriptor',
                  'torch_npu', 'artifact_sha256', 'entries'):
        assert prepared[field] == receipt[field], ('Prepared artifact mismatch', field)
    assert torch.npu.device_count() == 1, 'Expose exactly one externally assigned device'
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    # Keep the selected runtime context visible while CPU references are built.
    # This allocation performs no kernel computation and is outside all samples.
    context_anchor = torch.empty(1, dtype=torch.float32, device='npu')
    torch.npu.synchronize()
    if args.capture_case is not None:
        matrix = cases(args.family)
        target = next(case for case in matrix if case['id'] == args.capture_case)
        control = next(case for case in matrix if case['dtype'] == target['dtype'])
        capture = dict(identity=prepared, case=target, control=control, completed=False,
                       profiler_schedule=dict(wait=0, warmup=1, active=1, repeat=1),
                       profiler_config=dict(level='Level1', aic_metrics='AiCoreNone'))
        for case in ([control, target] if target != control else [target]):
            isolation_gate()
            work = Workload(args.family, case)
            directory = args.output / case['id']
            directory.mkdir()
            work.upload()
            checked = work.check(args.block_dim, directory, 'before')
            print('CAPTURE_OUTPUT_CHECK', case['id'], flush=True)
            if case == target:
                capture['before'] = checked
                op_names = {entry.name: compiled.op.op_name for entry, compiled in pairs}
                expected_ops = {entry.name: op_names[entry.name]
                                for entry in selected_entries(args.family, work.dtype)}
                isolation_gate()
                capture['profile'] = profile_sample(
                    work.bind(args.block_dim), directory/'profile-private', expected_ops, args.block_dim)
                capture['after'] = work.check(args.block_dim, directory, 'after')
                assert capture['before']['output_sha256'] == capture['after']['output_sha256']
            else:
                capture['control_check'] = checked
            del work
        isolation_gate()
        capture['identity'] = identity(args, pairs)
        assert capture['identity']['artifact_sha256'] == prepared['artifact_sha256']
        capture['completed'] = True
        write_json(args.output/'capture.json', capture)
        print('CAPTURE_COMPLETE', args.family, args.block_dim, target['id'], flush=True)
        del context_anchor
        return
    report = dict(schema='gda05-profile/1', identity=prepared, warmup=10, repeat=20, rounds=3,
                  event_repeat=20, profiler_repeat=3, cases=[], completed=False,
                  profiler_schedule=dict(wait=0, warmup=1, active=1, repeat=1),
                  profiler_config=dict(level='Level1', aic_metrics='AiCoreNone'),
                  profiler_processes='one fresh process per capture',
                  measurement='Same unchanged baseline in every segment; no optimization candidate')
    write_json(args.output/'measurement.json', report)
    selected = cases(args.family)
    for case in selected:
        isolation_gate()
        print('REFERENCE_START', case['id'], flush=True)
        work = Workload(args.family, case)
        directory = args.output/case['id']
        directory.mkdir()
        isolation_gate()
        work.upload()
        row = dict(case=case, before=work.check(args.block_dim, directory, 'before'), segments=[])
        report['cases'].append(row)
        write_json(args.output/'measurement.json', report)
        print('FULL_OUTPUT_CHECK', case['id'], flush=True)
        call = work.bind(args.block_dim)
        if case['measure']:
            # Qualify collection on the complete checked workload before the
            # longer uninstrumented series. Each later segment warms separately.
            row['profiles'] = []
            for index in range(report['profiler_repeat']):
                isolation_gate()
                row['profiles'].append(isolated_profile(
                    args, pairs, row, report['cases'], directory, index))
                write_json(args.output/'measurement.json', report)
            for iteration in range(report['rounds']):
                for segment in ('baseline_before', 'baseline_middle', 'baseline_after'):
                    isolation_gate()
                    for _ in range(report['warmup']):
                        plain_sample(call)
                    samples = [plain_sample(call) for _ in range(report['repeat'])]
                    row['segments'].append(dict(round=iteration+1, name=segment, samples=samples))
                    write_json(args.output/'measurement.json', report)
                    print('SAMPLES', case['id'], iteration+1, segment,
                          statistics.median(x['wall_ms'] for x in samples), flush=True)
            isolation_gate()
            row['events'] = [event_sample(call) for _ in range(report['event_repeat'])]
        row['after'] = work.check(args.block_dim, directory, 'after')
        assert row['after']['output_sha256'] == row['before']['output_sha256'], 'Output drift'
        write_json(args.output/'measurement.json', report)
        del work
    isolation_gate()
    report['cache'] = cache_profile(module(args.family), pairs, args.block_dim)
    final_identity = identity(args, pairs)
    assert final_identity['artifact_sha256'] == prepared['artifact_sha256'], 'Artifact changed during execution'
    report['identity'] = final_identity
    isolation_gate()
    report['completed'] = True
    write_json(args.output/'measurement.json', report)
    print('PROFILE_COMPLETE', args.family, args.block_dim, len(selected), flush=True)
    del context_anchor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True, choices=FAMILIES)
    parser.add_argument('--block-dim', required=True, type=int)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared-receipt', type=Path)
    parser.add_argument('--stop-file', type=Path, help='Private external isolation monitor stop marker')
    parser.add_argument('--capture-case', help='Diagnostic: full control check, then one capture in a fresh process')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
