"""Actual torch_npu GDN composition for a same-card measurement baseline.

This is test/measurement code, not a public implementation or fallback. Input
transfers are outside the measured call; widening, grouped-head materialization,
state/output allocation and every recurrence operation are inside it.
"""
import torch


@torch.no_grad()
def torch_npu_recurrent(inputs):
    q, k, v, g, beta = (inputs[n] for n in ('q', 'k', 'v', 'g', 'beta'))
    state0 = inputs.get('initial_state')
    tensors = [q, k, v, g, beta] + ([] if state0 is None else [state0])
    if any(x.device.type != 'npu' or x.device != q.device for x in tensors):
        raise ValueError('baseline requires inputs on one NPU; no CPU fallback')
    b, steps, h, kd = q.shape
    hv, vd = v.shape[2:]
    if kd != 128 or vd != 128 or not 1 <= steps <= 16 or hv < 1 or hv % h:
        raise ValueError('baseline requires the frozen GDN decode shape domain')
    if k.shape != q.shape or v.shape[:2] != (b, steps):
        raise ValueError('baseline q/k/v shapes differ')
    if q.dtype not in (torch.float32, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('baseline q/k/v require matching FP32 or BF16')
    if any(x.shape != (b, steps, hv) or x.dtype != torch.float32 for x in (g, beta)):
        raise ValueError('baseline requires scalar FP32 g/beta')
    if state0 is not None and (state0.shape != (b, hv, kd, vd) or state0.dtype != torch.float32):
        raise ValueError('baseline initial_state requires K-major FP32 storage')
    if any(not x.is_contiguous() for x in tensors):
        raise ValueError('baseline inputs must be contiguous')

    # All math and copies below are real NPU operations, included in timing.
    ratio = hv // h
    query = q.float().repeat_interleave(ratio, dim=2) * kd**-0.5
    key = k.float().repeat_interleave(ratio, dim=2)
    value = v.float()
    state = (torch.zeros((b, hv, kd, vd), dtype=torch.float32, device=q.device)
             if state0 is None else state0.clone())
    output = []
    for t in range(steps):
        decayed = state * g[:, t].exp()[..., None, None]
        read = (key[:, t, :, :, None] * decayed).sum(dim=-2)
        delta = (value[:, t] - read) * beta[:, t, :, None]
        state = decayed + key[:, t, :, :, None] * delta[..., None, :]
        output.append((query[:, t, :, :, None] * state).sum(dim=-2))
    return dict(o=torch.stack(output, dim=1).to(v.dtype), final_state=state)
