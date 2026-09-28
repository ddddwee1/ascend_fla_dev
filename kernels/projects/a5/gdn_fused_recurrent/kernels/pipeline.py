"""One launch; allocation and metadata only on the production host path."""


def all_entries():
    from .step import gdn_recurrent_fp32, gdn_recurrent_bf16
    return gdn_recurrent_fp32, gdn_recurrent_bf16


def run(inputs, launch):
    import torch
    q, v = inputs['q'], inputs['v']
    b, steps, h, _ = q.shape
    hv = v.shape[2]
    shape = (b, hv, 128, 128)
    sources = dict(inputs)
    has_initial = sources['initial_state'] is not None
    if not has_initial:
        # The GM argument exists in the ABI but the kernel must never read it.
        sources['initial_state'] = torch.empty(shape, dtype=torch.float32, device=q.device)
    outputs = dict(o=torch.empty(v.shape, dtype=v.dtype, device=v.device),
                   final_state=torch.empty(shape, dtype=torch.float32, device=q.device))
    if q.device.type == 'cpu':
        for value in outputs.values():
            value.fill_(float('nan'))
    scalars = dict(B=b, S=steps, H=h, HV=hv, HAS_INITIAL=int(has_initial))
    entry = all_entries()[0 if q.dtype == torch.float32 else 1]
    result = launch(entry, sources, outputs, scalars)
    if set(result) != set(outputs):
        raise RuntimeError('GDN recurrent launch returned incomplete outputs')
    return result
