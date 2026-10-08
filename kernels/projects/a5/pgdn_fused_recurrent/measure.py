"""Three same-card B/C/B rounds of synchronized public-call wall latency."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch

from ascend_fla.ops.pgdn_fused_recurrent import BLOCK_DIMS, prepare
from kernels.projects.a5.pgdn_fused_recurrent.ref.native_baseline import torch_npu_recurrent
from kernels.projects.a5.pgdn_fused_recurrent.ref.oracle import oracle
from kernels.projects.a5.pgdn_fused_recurrent.ref.reference import digest, make_inputs, reference
from kernels.projects.a5.pgdn_fused_recurrent.verify_native import public, measured_outputs, compare


def sample(call, data):
    for _ in range(10):
        result = call(data)
        torch.npu.synchronize()
    samples = []
    for _ in range(50):
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        result = call(data)
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns()-start)/1000.)
    # Retain and observe the actual last outputs, not an empty timing closure.
    return dict(samples_us=samples, median_us=statistics.median(samples),
                minimum_us=min(samples), maximum_us=max(samples),
                last_output_sha256={n: digest(x) for n, x in result.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--block-dim', type=int, choices=BLOCK_DIMS, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch_npu  # noqa: F401
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    prepare(block_dim=a.block_dim)
    a.output.mkdir(parents=True, exist_ok=True)
    assert not (a.output/'summary.json').exists()
    candidate = lambda x: public(x, a.block_dim)
    records = []
    for steps in (1, 16):
        for dtype in ('float32', 'bfloat16'):
            case = dict(id=f'S{steps}_{dtype}', B=1, S=steps, H=16, HV=32,
                        state='random', atk='random', dtype=dtype, seed=952000+steps)
            data = make_inputs(case)
            refs = dict(A=oracle(data), B=reference(data))
            device_data = {n: None if x is None else x.to('npu') for n, x in data.items()}
            initial_hashes = {n: None if x is None else digest(x) for n, x in device_data.items()}
            correctness = {}
            for name, call in [('candidate', candidate), ('baseline', torch_npu_recurrent)]:
                actual = measured_outputs(call, device_data, data)
                correctness[name] = compare(actual, refs)
                assert correctness[name]['passed'], (case['id'], name, correctness[name])
            rounds = []
            with torch.no_grad():
                for number in range(3):
                    row = dict(round=number+1,
                        baseline_before=sample(torch_npu_recurrent, device_data),
                        candidate=sample(candidate, device_data),
                        baseline_after=sample(torch_npu_recurrent, device_data))
                    rounds.append(row)
                    print(json.dumps(dict(case=case['id'], round=number+1,
                        medians_us={n: row[n]['median_us'] for n in ('baseline_before','candidate','baseline_after')})), flush=True)
            assert initial_hashes == {n: None if x is None else digest(x) for n, x in device_data.items()}
            report = dict(case=case, correctness=correctness, rounds=rounds,
                          input_sha256=initial_hashes)
            (a.output/(case['id']+'.json')).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            records.append(report)
    report = dict(stage='same_card_public_call_timing', block_dim=a.block_dim,
        metric='synchronized wall time in microseconds, not device-only kernel time',
        candidate='fused_recurrent_pgdn public inprocess call, allocations included',
        baseline='Actual torch_npu recurrence: FP32 naive normalization, ATK recurrence, scalar gates, contiguous groups, fresh FP32 main/ATK states, native output dtype; widening/group copies/allocations included',
        excluded='Input generation, CPU references, H2D transfer and compilation',
        rounds=3, warmup_per_leg=10, samples_per_leg=50, records=records,
        cuda_triton_executed=False, performance_threshold=None)
    (a.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('SAME_CARD_MEASUREMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
