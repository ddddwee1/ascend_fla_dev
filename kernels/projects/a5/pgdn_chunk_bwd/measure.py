"""Same-device saved-forward-state adjoint cost / complete backward sandwich.

The baseline reuses this candidate's normalization/ATK tapes and main boundary
checkpoints. It is not another full backward or a CUDA/Triton comparison.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from ascend_fla.ops.pgdn_chunk_bwd import _compiled, _pipeline, _validate, chunk_pgdn_bwd, prepare
from benchmark import digest, references
from ref.calibrate import inputs
from ref.classification import compare, write_disclosures
from ref.reference import INPUTS, NAMES


def sample(call):
    for _ in range(10):
        call()
    torch.npu.synchronize()
    raw = []
    for _ in range(50):
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        returned = call()
        torch.npu.synchronize()
        raw.append((time.perf_counter_ns() - start) * 1e-6)
        del returned
    return dict(samples_ms=raw, mean_ms=statistics.mean(raw), median_ms=statistics.median(raw))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block-dim', type=int, choices=(2,), default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    import torch_npu
    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    prepare(block_dim=args.block_dim)
    pipeline = _pipeline()
    operations = dict(zip((entry.name for entry in pipeline.entries()), _compiled(args.block_dim)))

    def launch(entry, sources, outputs, scalars):
        op = operations[entry.name]
        op(sources, {n: scalars[n] for n in op.scalar_names}, outputs)
        return outputs

    report = dict(stage='native_same_card_sandwich', block_dim=args.block_dim, warmup=10, repeat=50,
                  rounds=3, synchronization='before and after every timed call',
                  baseline='Same task-owned adjoint with saved normalization/ATK tapes and main chunk checkpoints; excludes all stage1/2 device computation, includes public validation and all pipeline allocation (including unused stage1/2 outputs).',
                  candidate='Complete public FP32 backward: validation, allocation, normalization/ATK, checkpoint regeneration and all four launches.',
                  scope='Cost comparison only; no separate full-backward backend, CUDA/Triton or weight verification.',
                  cases=[], numerical_status='pending', required_checks_complete=False)
    for T in (1024, 4096):
        case = dict(id=f'timing_t{T}', B=1, T=T, H=8, HV=8, kind='random', mask=7, seed=99240)
        xs, ds = inputs(case)
        cpu = {**dict(zip(INPUTS, xs)), **ds}
        # Finish CPU oracle work before starting any timed interval.
        expected = references(cpu)
        public = {n: value.npu() for n, value in cpu.items()}
        known = pipeline.run(public, launch, retain_stages=True)
        torch.npu.synchronize()
        cached = {n: known[n] for n in ('q_norm', 'k_read', 'k_write', 'A_history',
                  'q_raw_norm', 'k_raw_norm', 'final_A_state', 'checkpoints', 'final_state')}
        del known

        def cached_launch(entry, sources, outputs, scalars):
            if entry.name in ('pgdn_bwd_atk', 'pgdn_bwd_checkpoints'):
                return {n: cached[n] for n in outputs}
            return launch(entry, sources, outputs, scalars)

        @torch.no_grad()
        def baseline():
            _validate(**public, block_dim=args.block_dim)
            values = pipeline.run(public, cached_launch)
            return tuple(values[n] for n in NAMES)

        def candidate():
            return chunk_pgdn_bwd(**public, block_dim=args.block_dim)

        def check(label):
            a, b = baseline(), candidate()
            torch.npu.synchronize()
            aa = {n: value.cpu() for n, value in zip(NAMES, a)}
            bb = {n: value.cpu() for n, value in zip(NAMES, b)}
            exact = all(torch.equal(aa[n].view(torch.uint8), bb[n].view(torch.uint8)) for n in NAMES)
            comparisons = {key: compare(value, {n: expected[n] for n in ('A', 'B')}, expected['classes'])
                           for key, value in (('baseline', aa), ('candidate', bb))}
            disclosure = {}
            for key, value in (('baseline', aa), ('candidate', bb)):
                if all(bool(torch.isfinite(t).all()) for t in value.values()):
                    disclosure[key] = write_disclosures(args.output / f't{T}-{label}-{key}.jsonl', value,
                                                        expected['auxiliary'], expected['classes'])
            good = exact and len(disclosure) == 2 and all(r['ordinary_budget_satisfied'] for r in comparisons.values())
            result = dict(byte_identical=exact, comparisons=comparisons, disclosure=disclosure,
                          public_sha256={n: digest(value) for n, value in bb.items()}, required_checks_complete=good)
            (args.output / f't{T}-{label}-check.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            if not good:
                raise AssertionError('Timing numerical control failed; evidence retained')
            return result

        row = dict(case=case, cpu_input_sha256={n: digest(value) for n, value in cpu.items()}, rounds=[])
        report['cases'].append(row)
        for index in range(3):
            before = check(f'round{index+1}-before')
            triplet = dict(index=index+1, before=before, baseline_before=sample(baseline),
                           candidate=sample(candidate), baseline_after=sample(baseline))
            triplet['after'] = check(f'round{index+1}-after')
            row['rounds'].append(triplet)
            (args.output / 'measurement.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('TIMING_ROUND', T, index+1,
                  {n: triplet[n]['median_ms'] for n in ('baseline_before', 'candidate', 'baseline_after')}, flush=True)
    report['numerical_status'] = 'numerical_disclosure_complete_nonpass'
    report['required_checks_complete'] = True
    (args.output / 'measurement.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('MEASUREMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
