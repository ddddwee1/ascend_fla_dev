"""Shared ordered launch graph for the standalone harness and public operator."""
from __future__ import annotations

# Ordered inputs/outputs form the complete workspace ABI; no hidden host math.
GRAPH = (
    ('prepare', ('q', 'k', 'v', 'g', 'g_atk', 'beta_atk', 'beta', 'initial_A_state', 'log_atk_scale'), ('qn', 'kn', 'gc', 'bk', 'wv', 'final_A_state')),
    ('scores', ('qn', 'kn', 'gc', 'bk'), ('lower', 'score')),
    ('wy', ('lower', 'gc', 'bk', 'wv'), ('u', 'wy')),
    ('scan', ('kn', 'gc', 'u', 'wy', 'initial_state'), ('states', 'delta', 'final_state')),
    ('output', ('qn', 'gc', 'score', 'states', 'delta'), ('o',)),
)


def entries():
    from .output import pkda_chunk_output
    from .prepare import pkda_chunk_prepare
    from .stages import STAGES
    return (pkda_chunk_prepare, *STAGES, pkda_chunk_output)


def run(inputs, launch, *, retain_stages=True):
    """Allocate fresh FP32 outputs and invoke each stage via launch(entry,...).

    launch accepts the entry, input/output dictionaries and runtime scalars,
    and returns its output dictionary. Both CPU-tensor harnesses and the
    in-process NPU bridge use this graph; neither computes reference math.
    """
    import torch
    batch, time, heads, _ = inputs['q'].shape
    chunks = (time + 63) // 64
    scalars = dict(B=batch, T=time, H=heads, N=chunks)
    scale = float(inputs.get('scale', 128**-0.5))
    base = (batch, chunks, heads)
    shapes = {name: (*base, 64, 128) for name in ('qn', 'kn', 'gc', 'bk', 'wv', 'u', 'wy', 'delta')}
    shapes.update(lower=(*base, 64, 64), score=(*base, 64, 64),
                  states=(*base, 128, 128), o=(batch, time, heads, 128),
                  final_state=(batch, heads, 128, 128), final_A_state=(batch, heads, 128))
    values = dict(inputs)
    values['log_atk_scale'] = inputs['log_atk_scale'].view(1, heads)
    checkpoints = {}
    for index, (entry, (_, names, outputs)) in enumerate(zip(entries(), GRAPH)):
        fresh = {name: torch.empty(shapes[name], dtype=torch.float32,
                                  device=inputs['q'].device) for name in outputs}
        if inputs['q'].device.type == 'cpu':
            for tensor in fresh.values():
                tensor.fill_(float('nan'))
        got = launch(entry, {name: values[name] for name in names}, fresh, dict(scalars, scale=scale) if index == 0 else scalars)
        if set(got) != set(outputs):
            raise RuntimeError(f'{entry.name}: incomplete stage outputs')
        values.update(got)
        if retain_stages:
            checkpoints.update(got)
        else:
            # Retire GM buffers after their last consuming launch. Keep the
            # three public outputs even when no later stage reads them.
            live = {'o', 'final_state', 'final_A_state'}
            for _, future_inputs, _ in GRAPH[index + 1:]:
                live.update(future_inputs)
            for name in tuple(values):
                if name not in live:
                    del values[name]
    return checkpoints if retain_stages else {name: values[name] for name in ('o', 'final_state', 'final_A_state')}
