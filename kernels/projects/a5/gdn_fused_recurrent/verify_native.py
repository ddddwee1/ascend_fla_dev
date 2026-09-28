"""Actual public NPU calls with runtime inputs, dual CPU references and hashes."""
import argparse
import json
from pathlib import Path

import torch

from ascend_fla.ops.gdn_fused_recurrent import BLOCK_DIMS, fused_recurrent_gdn, prepare
from kernels.projects.a5.gdn_fused_recurrent.ref.integration import validate_returned
from kernels.projects.a5.gdn_fused_recurrent.ref.native_baseline import torch_npu_recurrent
from kernels.projects.a5.gdn_fused_recurrent.ref.oracle import oracle
from kernels.projects.a5.gdn_fused_recurrent.ref.reference import cases, digest, make_inputs, metric, reference, validate_inputs


def public(data, block_dim):
    o, state = fused_recurrent_gdn(**data, output_final_state=True, block_dim=block_dim)
    return dict(o=o, final_state=state)


def measured_outputs(call, device_inputs, cpu_inputs):
    before = {n: None if x is None else digest(x) for n, x in device_inputs.items()}
    result = call(device_inputs)
    torch.npu.synchronize()
    assert all(x.device == device_inputs['q'].device for x in result.values())
    for output in result.values():
        assert all(source is None or output.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
                   for source in device_inputs.values()), 'Output storage aliases an input'
    actual = {n: x.cpu() for n, x in result.items()}
    assert before == {n: None if x is None else digest(x) for n, x in device_inputs.items()}, 'Input mutation'
    validate_returned(cpu_inputs, actual, output_dtype=cpu_inputs['q'].dtype)
    return actual


def compare(actual, refs):
    metrics = {label: {name: metric(actual[name], target[name]) for name in actual}
               for label, target in refs.items()}
    # BF16 output is a storage/quality report; FP32 state remains budgeted.
    passed = all(row['finite'] and (name == 'o' and actual[name].dtype == torch.bfloat16
                                   or row['relative_l2'] <= 1e-4)
                 for group in metrics.values() for name, row in group.items())
    return dict(passed=passed, metrics=metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-dim', type=int, choices=BLOCK_DIMS, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    selected = cases()
    if args.case:
        wanted = set(args.case)
        assert wanted <= {x['id'] for x in selected}, 'Unknown requested case'
        selected = [x for x in selected if x['id'] in wanted]
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'summary.json').exists(), 'Fresh output directory required'
    import torch_npu  # noqa: F401: registers the actual NPU backend
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    prepare(block_dim=args.block_dim)
    records = []
    for case in selected:
        data = make_inputs(case)
        validate_inputs(data)
        refs = dict(A=oracle(data), B=reference(data))
        device_inputs = {n: None if x is None else x.to('npu') for n, x in data.items()}
        actual = measured_outputs(lambda x: public(x, args.block_dim), device_inputs, data)
        report = dict(case=case, block_dim=args.block_dim,
                      input_sha256={n: None if x is None else digest(x) for n, x in data.items()},
                      output_sha256={n: digest(x) for n, x in actual.items()},
                      comparison=compare(actual, refs))
        if case['dtype'] == 'bfloat16':
            # Test-only exact widening creates a same-input FP32 native run.
            widened = {n: x.float() if n in ('q', 'k', 'v') else x for n, x in data.items()}
            widened_device = {n: None if x is None else x.to('npu') for n, x in widened.items()}
            companion = measured_outputs(lambda x: public(x, args.block_dim), widened_device, widened)
            report['same_input_fp32_comparison'] = compare(companion, refs)
            report['bf16_storage'] = dict(
                output_equals_fp32_rne=digest(actual['o']) == digest(companion['o'].bfloat16()),
                state_equals_fp32=digest(actual['final_state']) == digest(companion['final_state']))
        if args.baseline:
            baseline = measured_outputs(torch_npu_recurrent, device_inputs, data)
            report['torch_npu_composition'] = compare(baseline, refs)
        records.append(report)
        (args.output/(case['id']+'.json')).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        passed = report['comparison']['passed']
        if 'bf16_storage' in report:
            passed = passed and all(report['bf16_storage'].values()) and report['same_input_fp32_comparison']['passed']
        if args.baseline:
            passed = passed and report['torch_npu_composition']['passed']
        print(json.dumps(dict(case=case['id'], passed=passed,
                              metrics=report['comparison']['metrics'],
                              bf16_storage=report.get('bf16_storage'))), flush=True)
        if not passed:
            torch.save(dict(inputs=data, actual=actual, references=refs), args.output/(case['id']+'-failure.pt'))
            raise AssertionError('Native case failed: '+case['id'])
    summary = dict(stage='native_inprocess', block_dim=args.block_dim, cases=len(records),
                   passed=True, baseline_executed=args.baseline, records=records)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
