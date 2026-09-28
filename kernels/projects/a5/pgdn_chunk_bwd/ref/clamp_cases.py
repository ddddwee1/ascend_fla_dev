"""Supplemental dense clamp inputs; original frozen cases remain unchanged."""
import torch

from .calibrate import inputs as original_inputs


def cases():
    result = []
    for T, H, HV, seed in ((64, 1, 2, 195082), (64, 1, 2, 884099),
                           (128, 3, 6, 195082)):
        for mode in ('q', 'k', 'both'):
            for mask in range(1, 8):
                result.append(dict(id=f'dense_t{T}_h{H}_hv{HV}_s{seed}_{mode}_m{mask}',
                                   B=1, T=T, H=H, HV=HV, seed=seed, kind='random',
                                   mask=mask, clamp_mode=mode, radius='at'))
    for mask in range(1, 8):
        result.append(dict(id=f'dense_r8_both_m{mask}', B=1, T=64, H=1, HV=8,
                           seed=49301, kind='random', mask=mask,
                           clamp_mode='both', radius='at'))
    for radius in ('below_ulp', 'above_ulp'):
        result.append(dict(id=f'dense_{radius}_both_m7', B=1, T=64, H=1, HV=2,
                           seed=884099, kind='random', mask=7,
                           clamp_mode='both', radius=radius))
    result.append(dict(id='dense_full_both_m7', B=1, T=4096, H=1, HV=8,
                       seed=195082, kind='random', mask=7,
                       clamp_mode='both', radius='at'))
    return result


def inputs(case):
    xs, ds = original_inputs(case)
    xs = list(xs)
    epsilon = torch.tensor(1e-12, dtype=torch.float32)
    radius = case['radius']
    if radius == 'below_ulp':
        epsilon = torch.nextafter(epsilon, torch.zeros_like(epsilon))
    elif radius == 'above_ulp':
        epsilon = torch.nextafter(epsilon, torch.full_like(epsilon, float('inf')))
    elif radius != 'at':
        raise ValueError('Unknown supplemental radius')
    mode = case['clamp_mode']
    if mode not in ('q', 'k', 'both'):
        raise ValueError('Unknown supplemental clamp mode')
    for index in ((0, 1) if mode == 'both' else (0,) if mode == 'q' else (1,)):
        raw = xs[index]
        xs[index] = (raw / raw.norm(dim=-1, keepdim=True) * epsilon).contiguous()
    return tuple(xs), ds
