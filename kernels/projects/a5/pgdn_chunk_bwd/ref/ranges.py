"""Observed FP32 stage-reference ranges; distinct from actual device ranges."""
import argparse
import json
from pathlib import Path

import torch

from .calibrate import cases, inputs
from .reference import INPUTS
from .stages import reference_stages


def tensor_range(value):
    finite = torch.isfinite(value)
    normal = value[finite]
    return dict(min=normal.min().item() if normal.numel() else None,
                max=normal.max().item() if normal.numel() else None,
                max_abs=normal.abs().max().item() if normal.numel() else None,
                nonfinite=int((~finite).sum()), dtype=str(value.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    selected = [c for c in cases() if c['id'] in args.case]
    if {c['id'] for c in selected} != set(args.case):
        raise ValueError('Unknown range case')
    report = dict(stage='CPU_FP32_independent_stage_reference_ranges', cases=[],
                  scope='Observed values, not a bound for all finite inputs and not device measurement')
    for case in selected:
        xs, ds = inputs(case)
        cpu = {**dict(zip(INPUTS, xs)), **ds}
        ranges = {}

        def observe(name, value):
            row = tensor_range(value)
            if name in ranges:
                previous = ranges[name]
                row['min'] = min(row['min'], previous['min'])
                row['max'] = max(row['max'], previous['max'])
                row['max_abs'] = max(row['max_abs'], previous['max_abs'])
                row['nonfinite'] += previous['nonfinite']
            ranges[name] = row

        for name, value in cpu.items():
            if value is not None:
                observe('input_' + name, value)
        reference_stages(cpu, observe=observe)
        report['cases'].append(dict(case=case, ranges=ranges))
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('RANGES_COMPLETE', case['id'], flush=True)


if __name__ == '__main__':
    main()
