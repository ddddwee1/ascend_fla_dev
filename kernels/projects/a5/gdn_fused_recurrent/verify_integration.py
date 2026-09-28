"""Native chunk-to-decode and decode-to-decode state continuation.

State is passed directly between actual NPU calls. CPU copies only generate
references and inspect returned values; there is no reference execution path.
"""
import argparse
import json
from pathlib import Path
import torch

from ascend_fla.ops.gdn_chunk_fwd import chunk_gdn, prepare as prepare_chunk
from ascend_fla.ops.gdn_fused_recurrent import BLOCK_DIMS, fused_recurrent_gdn, prepare
from kernels.projects.a5.gdn_fused_recurrent.ref.integration import (
    integration_cases, partitions, slice_inputs, trace_metrics, validate_returned)
from kernels.projects.a5.gdn_fused_recurrent.ref.reference import cases, digest, make_inputs, reference


def to_device(data):
    return {n: None if x is None else x.to('npu') for n, x in data.items()}


def checked_call(call, data, cpu_data):
    before = {n: None if x is None else digest(x) for n, x in data.items()}
    o, state = call(data)
    torch.npu.synchronize()
    result = dict(o=o, final_state=state)
    for x in result.values():
        assert x.device == data['q'].device
        assert all(y is None or x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr()
                   for y in data.values()), 'Result aliases an input'
    assert before == {n: None if x is None else digest(x) for n, x in data.items()}, 'Input mutation'
    cpu_result = {n: x.cpu() for n, x in result.items()}
    validate_returned(cpu_data, cpu_result, output_dtype=cpu_data['q'].dtype)
    return result, cpu_result


def chain(data, initial_device, initial_cpu, width, call):
    state, state_cpu = initial_device, initial_cpu
    outputs, calls = [], []
    for start, stop in partitions(data['q'].shape[1], width):
        cpu_data = slice_inputs(data, start, stop, state_cpu)
        # The test prepares each short input on CPU; state remains on the NPU.
        device_data = to_device(dict(cpu_data, initial_state=None))
        device_data['initial_state'] = state
        result, cpu_result = checked_call(call, device_data, cpu_data)
        calls.append(dict(start=start, stop=stop,
                          initial_state_sha256=None if state_cpu is None else digest(state_cpu),
                          returned_state_sha256=digest(cpu_result['final_state'])))
        state, state_cpu = result['final_state'], cpu_result['final_state']
        outputs.append(cpu_result['o'])
    return dict(o=torch.cat(outputs, dim=1), final_state=state_cpu), calls


def passed_trace(trace, dtype):
    return (trace['final_state']['finite'] and trace['final_state']['relative_l2'] <= 1e-4
            and all(x['finite'] and (dtype == 'bfloat16' or x['relative_l2'] <= 1e-4)
                    for x in trace['tokens']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--block-dim', type=int, choices=BLOCK_DIMS, required=True)
    p.add_argument('--chunk-block-dim', type=int, choices=(1, 2), default=2)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch_npu  # noqa: F401
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    # Both operator families must be registered before the first aclnn call.
    prepare(block_dim=a.block_dim)
    prepare_chunk(block_dim=a.chunk_block_dim)
    a.output.mkdir(parents=True, exist_ok=True)
    assert not (a.output/'summary.json').exists()
    decode = lambda x: fused_recurrent_gdn(**x, output_final_state=True, block_dim=a.block_dim)
    chunk = lambda x: chunk_gdn(**x, output_final_state=True, block_dim=a.chunk_block_dim)
    records = []
    for case in integration_cases():
        data = make_inputs(case)
        prefix = case['prefix']
        _, whole = checked_call(chunk, to_device(data), data)
        prefix_data = slice_inputs(data, 0, prefix, None)
        pre_device, pre_cpu = checked_call(chunk, to_device(prefix_data), prefix_data)
        suffix = slice_inputs(data, prefix, case['S'], pre_cpu['final_state'])
        expected = dict(o=whole['o'][:, prefix:].contiguous(), final_state=whole['final_state'])
        reference_b = reference(data)
        expected_b = dict(o=reference_b['o'][:, prefix:].contiguous(), final_state=reference_b['final_state'])
        rows = []
        for width in (1, 3, 7, 16):
            actual, calls = chain(suffix, pre_device['final_state'], pre_cpu['final_state'], width, decode)
            chunk_metrics = trace_metrics(actual, expected, token_offset=prefix)
            cpu_metrics = trace_metrics(actual, expected_b, token_offset=prefix)
            rows.append(dict(width=width, calls=calls, vs_whole_chunk=chunk_metrics,
                             vs_CPU_B=cpu_metrics,
                             passed=passed_trace(chunk_metrics, case['dtype'])
                                    and passed_trace(cpu_metrics, case['dtype'])))
        record = dict(case=case, continuations=rows, passed=all(x['passed'] for x in rows))
        (a.output/(case['id']+'.json')).write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(case=case['id'], passed=record['passed'])), flush=True)
        assert record['passed'], case['id']
        records.append(record)
    splits = []
    for case in cases():
        if case['S'] != 16:
            continue
        data = make_inputs(case)
        device_data = to_device(data)
        _, whole = checked_call(decode, device_data, data)
        for width in (1, 3, 7):
            actual, calls = chain(data, device_data['initial_state'], data['initial_state'], width, decode)
            trace = trace_metrics(actual, whole)
            row = dict(case=case, width=width, calls=calls, trace=trace,
                       passed=all(trace['byte_identical'].values()))
            splits.append(row)
            assert row['passed'], (case['id'], width, trace)
    report = dict(stage='native_state_continuation', block_dim=a.block_dim,
                  chunk_block_dim=a.chunk_block_dim, integration_cases=records,
                  split_cases=splits, passed=True)
    (a.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(stage=report['stage'], passed=True, prefill_cases=len(records),
                         split_comparisons=len(splits))), flush=True)


if __name__ == '__main__':
    main()
