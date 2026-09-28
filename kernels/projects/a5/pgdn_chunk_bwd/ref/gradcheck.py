"""Independent A/B FP64 mathematical qualification, away from the clamp kink."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .oracle import outputs
from .reference import NAMES, analytical, forward


def run():
    torch.set_num_threads(1)
    report = dict(stage='CPU_FP64_mathematical_gradcheck', cases=[],
                  exact_clamp='not checked by central differences across the kink',
                  criteria=dict(eps=1e-6, atol=1e-8, rtol=1e-5))
    for ratio in (1, 2):
        for kind in ('normal', 'below', 'above'):
            rng = torch.Generator().manual_seed(93900 + ratio)
            q = torch.randn(1, 3, 1, 3, generator=rng, dtype=torch.float64) * .3
            k = torch.randn(q.shape, generator=rng, dtype=torch.float64) * .3
            factor = 1.
            if kind != 'normal':
                radius = .5 if kind == 'below' else 2.
                q = q / q.norm(dim=-1, keepdim=True) * radius
                k = k / k.norm(dim=-1, keepdim=True) * radius
                factor = 1e-12
            v = torch.randn(1, 3, ratio, 2, generator=rng, dtype=torch.float64) * .05
            ga = -torch.rand(1, 3, 1, generator=rng, dtype=torch.float64) * .03
            g = -torch.rand(1, 3, ratio, generator=rng, dtype=torch.float64)
            ba = torch.rand(1, 3, 1, generator=rng, dtype=torch.float64)
            beta = torch.rand(1, 3, ratio, generator=rng, dtype=torch.float64)
            values = tuple(x.requires_grad_() for x in (q, k, v, ga, g, ba, beta))
            cotangents = (torch.randn(v.shape, generator=rng, dtype=torch.float64) * .1,
                          torch.randn(1, ratio, 3, 2, generator=rng, dtype=torch.float64) * .1,
                          torch.randn(1, 1, 3, generator=rng, dtype=torch.float64) * .1)

            def raw(xs):
                return (xs[0] * factor, xs[1] * factor, *xs[2:])

            def loss(ys):
                return sum((y * d).sum() for y, d in zip(ys, cotangents))

            def A(*xs):
                return loss(outputs(*raw(xs), fp64=True))

            class B(torch.autograd.Function):
                @staticmethod
                def forward(ctx, *xs):
                    ctx.save_for_backward(*xs)
                    return loss(forward(*raw(xs)))

                @staticmethod
                def backward(ctx, upstream):
                    result = analytical(*raw(ctx.saved_tensors), *cotangents)
                    return tuple(upstream * result[n] * (factor if n in ('dq', 'dk') else 1.) for n in NAMES)

            row = dict(ratio=ratio, kind=kind, B=1, T=3, H=1, HV=ratio, K=3, V=2,
                       qk_parameterization_factor=factor, forward_loss_abs=(A(*values) - B.apply(*values)).abs().item())
            for name, function in (('A_lifted', A), ('B_analytical', B.apply)):
                row[name] = torch.autograd.gradcheck(function, values, **report['criteria'])
            report['cases'].append(row)
            print('GRADCHECK', json.dumps(row), flush=True)
    report['passed'] = all(r['A_lifted'] and r['B_analytical'] for r in report['cases'])
    report['sources'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in Path(__file__).parent.glob('*.py')}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
