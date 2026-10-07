"""One launch; production host performs allocation and metadata work only."""


def all_entries():
    from .step import pgdn_recurrent_fp32, pgdn_recurrent_bf16
    return pgdn_recurrent_fp32, pgdn_recurrent_bf16


def run(inputs, launch):
    import torch
    q, v = inputs['q'], inputs['v']
    b, steps, h, _ = q.shape
    hv = v.shape[2]
    sources = dict(inputs)
    has_initial = sources['initial_state'] is not None
    has_a = sources['initial_A_state'] is not None
    state_shape, atk_shape = (b, hv, 128, 128), (b, h, 128)
    # Dummy input addresses are required by the ABI; HAS flags prevent reads.
    if not has_initial:sources['initial_state'] = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    if not has_a:sources['initial_A_state'] = torch.empty(atk_shape, dtype=torch.float32, device=q.device)
    outputs = dict(o=torch.empty(v.shape, dtype=v.dtype, device=v.device),
                   final_state=torch.empty(state_shape, dtype=torch.float32, device=q.device),
                   final_A_state=torch.empty(atk_shape, dtype=torch.float32, device=q.device))
    if q.device.type == 'cpu':
        for value in outputs.values():value.fill_(float('nan'))
    scalars = dict(B=b, S=steps, H=h, HV=hv, HAS_INITIAL=int(has_initial), HAS_A=int(has_a))
    entry = all_entries()[0 if q.dtype == torch.float32 else 1]
    result = launch(entry, sources, outputs, scalars)
    if set(result) != set(outputs):raise RuntimeError('PGDN recurrent launch returned incomplete outputs')
    return result
