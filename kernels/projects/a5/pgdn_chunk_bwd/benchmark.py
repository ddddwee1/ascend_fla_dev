"""Actual NPU output qualification with separately retained numerical disclosures.

The caller owns device isolation, fresh health/occupancy checks and the shared
lock. CPU oracle threads never receive NPU tensors. Timing uses a separate tool.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import torch

from ascend_fla.ops.pgdn_chunk_bwd import _compiled, _pipeline, chunk_pgdn_bwd, prepare
from ref.calibrate import cases, inputs
from ref.classification import classify, compare, write_disclosures
from ref.oracle import autograd
from ref.reference import INPUTS, NAMES, analytical
from ref.stages import reference_stages
from ref.ranges import tensor_range
from ref.verification import digest, internal_comparison


def references(cpu):
    assert torch.get_num_threads() == 1
    assert all(value is None or value.device.type == 'cpu' for value in cpu.values())
    xs = tuple(cpu[n] for n in INPUTS)
    ds = {n: cpu[n] for n in ('do', 'dht', 'dA_T')}
    a = autograd(*xs, **ds)
    b = analytical(*xs, **ds)
    auxiliary = analytical(*(x.double() for x in xs),
                           **{n: None if x is None else x.double() for n, x in ds.items()},
                           fp32_branch=True, auxiliary=True)
    return dict(A=a, B=b, auxiliary=auxiliary, classes=classify(auxiliary), stages=reference_stages(cpu))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block-dim', type=int, choices=(1, 2), required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--oracle-workers', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--supplemental-clamp', action='store_true')
    args = parser.parse_args()
    selected = cases()
    make_inputs = inputs
    if args.supplemental_clamp:
        from ref.clamp_cases import cases as clamp_cases, inputs as clamp_inputs
        selected, make_inputs = clamp_cases(), clamp_inputs
    if args.case and args.case != ['all']:
        selected = [c for c in selected if c['id'] in args.case]
        if {c['id'] for c in selected} != set(args.case):
            raise ValueError('Unknown case')
    selected.sort(key=lambda c: 0 if c['id'] in ('full_r1_m7', 'dense_full_both_m7') else (1 if c['T'] <= 192 else 2))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    import torch_npu
    assert torch.npu.device_count() == 1, 'external configuration must expose one assigned device'
    torch.npu.set_device(0)
    print('BUILD_START', flush=True)
    prepare(block_dim=args.block_dim)
    print('BUILD_DONE', flush=True)
    pipeline = _pipeline()
    compiled = dict(zip((entry.name for entry in pipeline.entries()), _compiled(args.block_dim)))

    def launch(entry, sources, outputs, scalars):
        for value in outputs.values():
            value.fill_(float('nan'))
        operation = compiled[entry.name]
        operation(sources, {n: scalars[n] for n in operation.scalar_names}, outputs)
        return outputs

    report = dict(stage='native_inprocess_classified_qualification', block_dim=args.block_dim,
                  case_set='supplemental_dense_clamp' if args.supplemental_clamp else 'original_frozen_256',
                  torch=torch.__version__, torch_npu=torch_npu.__version__, oracle_workers=args.oracle_workers,
                  cases=[], numerical_status='pending', all_required_checks_complete=False)

    def finish(job):
        case, cpu, gpu, host, public, immutable, future = job
        expected = future.result()
        directory = args.output / case['id']
        directory.mkdir()
        # Save actual returned tensors privately before any acceptance assertion.
        torch.save(dict(composition=host, public=public), directory / 'returned.private.pt')
        B, _, H, _ = cpu['q'].shape
        HV = cpu['v'].shape[2]
        known = {**cpu, **expected['stages'],
                 'dout': torch.zeros_like(cpu['v']) if cpu['do'] is None else cpu['do'],
                 'dht': torch.zeros(B,HV,128,128) if cpu['dht'] is None else cpu['dht'],
                 'dat': torch.zeros(B,H,128) if cpu['dA_T'] is None else cpu['dA_T']}

        def independent(entry, sources, outputs, scalars):
            device_sources = {name: known[name].npu() for name in sources}
            values = launch(entry, device_sources, outputs, scalars)
            torch.npu.synchronize()  # Isolated leaf's temporary inputs retire here.
            return values

        leaf_device = pipeline.run(gpu, independent, retain_stages=True)
        torch.npu.synchronize()
        leaves = {n: value.cpu() for n, value in leaf_device.items()}
        del leaf_device
        internal = {label: internal_comparison(values, expected['stages'])
                    for label, values in (('composition', host), ('independent_leaves', leaves))}
        branches = {}
        for label, values in (('composition', host), ('independent_leaves', leaves)):
            branches[label] = {n: torch.equal(values[n] >= 1e-12, cpu[source].norm(dim=-1) >= 1e-12)
                               for n, source in (('q_raw_norm', 'q'), ('k_raw_norm', 'k'))}
        refs = {n: expected[n] for n in ('A', 'B')}
        comparisons, disclosure = {}, {}
        for label, values in (('composition', host), ('independent_leaves', leaves), ('public', public)):
            actual = {n: values[n] for n in NAMES}
            comparisons[label] = compare(actual, refs, expected['classes'])
            # Nonfinite values are failures, not valid disclosure payloads.
            if all(bool(torch.isfinite(t).all()) for t in actual.values()):
                disclosure[label] = write_disclosures(directory / f'{label}-disclosures.jsonl', actual,
                                                      expected['auxiliary'], expected['classes'])
        schema = {n: dict(shape=list(public[n].shape), dtype=str(public[n].dtype)) for n in NAMES}
        identical = all(torch.equal(public[n].view(torch.uint8), host[n].view(torch.uint8)) for n in NAMES)
        good = (immutable and identical and
                all(row['budget_satisfied'] for values in internal.values() for row in values.values()) and
                all(ok for values in branches.values() for ok in values.values()) and
                all(row['ordinary_budget_satisfied'] for row in comparisons.values()) and len(disclosure) == 3)
        row = dict(case=case, observed_device_ranges={n: tensor_range(value) for n, value in host.items()},
                   inputs_unchanged=immutable, public_composition_byte_identical=identical,
                   public_schema=schema, internal=internal, normalization_branch_equal=branches,
                   comparisons=comparisons, disclosure=disclosure,
                   cpu_input_sha256={n: None if value is None else digest(value) for n, value in cpu.items()},
                   stage_sha256={n: digest(value) for n, value in host.items()},
                   public_sha256={n: digest(value) for n, value in public.items()},
                   private_tensor_sha256=hashlib.sha256((directory / 'returned.private.pt').read_bytes()).hexdigest(),
                   required_checks_complete=good,
                   numerical_status=('numerical_disclosure_complete_nonpass' if len(disclosure) == 3
                                     else 'failed_incomplete_disclosure'))
        (directory / 'result.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        report['cases'].append(row)
        (args.output / 'qualification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('NATIVE_CASE', case['id'], 'required_checks_complete', good,
              'disclosures', comparisons['public']['disclosure_count'], flush=True)
        if not good:
            raise AssertionError(f'Native qualification failed: {case["id"]}; raw reports retained')

    pending = deque()
    with ThreadPoolExecutor(max_workers=args.oracle_workers) as executor:
        for case in selected:
            xs, ds = make_inputs(case)
            cpu = {**dict(zip(INPUTS, xs)), **ds}
            future = executor.submit(references, cpu)
            gpu = {n: None if value is None else value.npu() for n, value in cpu.items()}
            before = {n: digest(value) for n, value in gpu.items() if value is not None}
            assert before == {n: digest(value) for n, value in cpu.items() if value is not None}
            print('EXECUTE', case['id'], json.dumps(case), flush=True)
            start = time.monotonic()
            actual = pipeline.run(gpu, launch, retain_stages=True)
            torch.npu.synchronize()
            host = {n: value.cpu() for n, value in actual.items()}
            del actual
            print('RETURNED', case['id'], time.monotonic() - start, flush=True)
            returned = chunk_pgdn_bwd(**gpu, block_dim=args.block_dim)
            torch.npu.synchronize()
            public = {n: value.cpu() for n, value in zip(NAMES, returned)}
            immutable = all(digest(value) == before[n] for n, value in gpu.items() if value is not None)
            pending.append((case, cpu, gpu, host, public, immutable, future))
            if len(pending) >= args.oracle_workers:
                finish(pending.popleft())
        while pending:
            finish(pending.popleft())
    report['all_required_checks_complete'] = all(c['required_checks_complete'] for c in report['cases'])
    report['numerical_status'] = 'numerical_disclosure_complete_nonpass'
    (args.output / 'qualification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('NATIVE_DISCLOSURE_QUALIFICATION_COMPLETE', len(report['cases']), flush=True)


if __name__ == '__main__':
    main()
