"""Generate the frozen-domain grid and retain classified A/B calibration."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import torch

from .classification import classify, compare, write_disclosures
from .oracle import autograd
from .reference import analytical


def cases():
    result = []

    def add(label, T, H, HV, kind='random', masks=range(1, 8)):
        for mask in masks:
            result.append(dict(id=f'{label}_m{mask}', B=1, T=T, H=H, HV=HV,
                               kind=kind, mask=mask, seed=95000 + T + 17 * H + HV))

    # Original full workload first; source and hardware runners use this order.
    for ratio in (1, 2, 4, 8):
        add(f'full_r{ratio}', 4096, 8 // ratio, 8, masks=(7, 1, 2, 3, 4, 5, 6))
    for T in (64, 128, 192):
        for H, HV in ((1, 1), (1, 2), (1, 4), (1, 8), (3, 3), (3, 6)):
            add(f'c{T//64}_h{H}_hv{HV}', T, H, HV)
    for kind in ('zero', 'axis_below', 'axis_at', 'axis_above', 'dense_below',
                 'dense_above', 'beta0', 'beta1', 'atk_beta0', 'atk_beta1',
                 'gates0', 'gates_underflow'):
        add(kind, 64, 1, 2, kind)
    for radius in ('1e-6', '.03', '.05', '.1', '.3', '.5', '1', '2'):
        add(f'radial_{radius}', 64, 1, 2, 'radial:' + radius, masks=(1, 4))
    add('full_gates0', 4096, 8, 8, 'gates0', masks=(7,))
    add('full_weak', 4096, 8, 8, 'weak', masks=(7,))
    assert len({c['id'] for c in result}) == len(result)
    return result


def inputs(case):
    B, T, H, HV = (case[n] for n in ('B', 'T', 'H', 'HV'))
    rng = torch.Generator().manual_seed(case['seed'])
    q = torch.randn(B, T, H, 128, generator=rng) * .05
    k = torch.randn(q.shape, generator=rng) * .05
    v = torch.randn(B, T, HV, 128, generator=rng) * .05
    ga = -torch.rand(B, T, H, generator=rng) * .03
    g = -torch.rand(B, T, HV, generator=rng)
    ba = torch.rand(B, T, H, generator=rng)
    beta = torch.rand(B, T, HV, generator=rng)
    kind = case['kind']
    if kind == 'zero':
        q.zero_()
        k.zero_()
    if kind.startswith('axis_') or kind.startswith('radial:'):
        radius = {'axis_below': 5e-13, 'axis_at': 1e-12, 'axis_above': 2e-12}.get(kind)
        if radius is None:
            radius = float(kind.split(':')[1])
        q.zero_()
        k.zero_()
        q[..., 0] = radius
        k[..., 0] = radius
    if kind.startswith('dense_'):
        radius = 5e-13 if kind == 'dense_below' else 2e-12
        q = q / q.norm(dim=-1, keepdim=True) * radius
        k = k / k.norm(dim=-1, keepdim=True) * radius
    if kind in ('beta0', 'beta1'):
        beta.fill_(0. if kind == 'beta0' else 1.)
    if kind in ('atk_beta0', 'atk_beta1'):
        ba.fill_(0. if kind == 'atk_beta0' else 1.)
    if kind in ('gates0', 'gates_underflow'):
        ga.fill_(0. if kind == 'gates0' else -120.)
        g.fill_(0. if kind == 'gates0' else -120.)
    if kind == 'weak':
        g.mul_(.03)
    cotangents = (torch.randn(v.shape, generator=rng) * .1,
                  torch.randn(B, HV, 128, 128, generator=rng) * .1,
                  torch.randn(B, H, 128, generator=rng) * .1)
    selected = tuple(value if case['mask'] & (1 << index) else None for index, value in enumerate(cotangents))
    return (q, k, v, ga, g, ba, beta), dict(zip(('do', 'dht', 'dA_T'), selected))


def digest(value):
    return None if value is None else hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def one(case, output):
    torch.set_num_threads(1)
    start = time.monotonic()
    xs, cotangents = inputs(case)
    print('CALIBRATION_BEGIN', case['id'], flush=True)
    a = autograd(*xs, **cotangents)
    print('CALIBRATION_A', case['id'], flush=True)
    b = analytical(*xs, **cotangents)
    lifted = tuple(x.double() for x in xs)
    lifted_cotangents = {n: None if x is None else x.double() for n, x in cotangents.items()}
    auxiliary = analytical(*lifted, **lifted_cotangents, fp32_branch=True, auxiliary=True)
    classes = classify(auxiliary)
    ab = compare(a, {'B': b}, classes)
    ba = compare(b, {'A': a}, classes)
    directory = Path(output) / case['id']
    directory.mkdir(parents=True, exist_ok=True)
    disclosure = {label: write_disclosures(directory / f'{label}-disclosures.jsonl', values, auxiliary, classes)
                  for label, values in (('A', a), ('B', b))}
    result = dict(case=case, torch=torch.__version__, elapsed_s=time.monotonic() - start,
                  input_sha256=[digest(x) for x in xs], cotangent_sha256={n: digest(x) for n, x in cotangents.items()},
                  A_against_B=ab, B_against_A=ba, disclosure=disclosure,
                  ordinary_budget_satisfied=ab['ordinary_budget_satisfied'] and ba['ordinary_budget_satisfied'],
                  numerical_status='numerical_disclosure_complete_nonpass' if ab['disclosure_count'] else ab['numerical_status'])
    (directory / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('CALIBRATION_RESULT', case['id'], result['ordinary_budget_satisfied'], ab['disclosure_count'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', action='append')
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    selected = cases()
    if args.case:
        selected = [c for c in selected if c['id'] in args.case]
        if {c['id'] for c in selected} != set(args.case):
            raise ValueError('Unknown case')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(stage='CPU_FP32_A_B_classified_calibration', budget=1e-4,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                  selected_cases=selected, workers=args.workers, results=[])

    def record(result):
        report['results'].append(result)
        (args.output / 'calibration.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    if args.workers == 1:
        for case in selected:
            record(one(case, args.output))
    else:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            jobs = [executor.submit(one, case, args.output) for case in selected]
            for future in as_completed(jobs):
                record(future.result())
    report['ordinary_budget_satisfied'] = all(r['ordinary_budget_satisfied'] for r in report['results'])
    report['disclosure_complete'] = len(report['results']) == len(selected)
    report['numerical_status'] = 'numerical_disclosure_complete_nonpass'
    (args.output / 'calibration.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not report['ordinary_budget_satisfied']:
        raise SystemExit('Ordinary subset failed; preserve evidence and investigate before kernel authoring')


if __name__ == '__main__':
    main()
