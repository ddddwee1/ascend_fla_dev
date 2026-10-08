"""Common validation of actual returned tensors, including independent ATK."""
import torch
from .reference import OUTPUTS


def validate_returned(inputs, result, *, output_dtype):
    """Reject broadcasting, omitted outputs, storage aliases and bad metadata."""
    if set(result) != set(OUTPUTS):
        raise ValueError("exactly o, final_state and final_A_state must be returned")
    b, s, hv, _ = inputs["v"].shape
    shapes = {"o": (b, s, hv, 128), "final_state": (b, hv, 128, 128), "final_A_state": (b, inputs["q"].shape[2], 128)}
    for name, value in result.items():
        dtype = output_dtype if name == "o" else torch.float32
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shapes[name]
                or value.dtype != dtype or value.device.type != "cpu"
                or not value.is_contiguous() or not bool(torch.isfinite(value).all())):
            raise ValueError(f"{name}: invalid returned shape/dtype/storage/value")
        for source in inputs.values():
            if source is not None and value.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
                raise ValueError(f"{name}: returned storage aliases an input")


def integration_cases():
    templates=[dict(id='full',B=1,H=16,HV=32,prefix=128)]
    templates += [dict(id=f'r{ratio}',B=1,H=3,HV=3*ratio,prefix=64) for ratio in (1,2,4,8)]
    templates += [dict(id='batch2',B=2,H=2,HV=8,prefix=128),dict(id='length4096',B=1,H=1,HV=8,prefix=4032)]
    return [dict(row,id=row['id']+'_'+dtype,dtype=dtype,S=row['prefix']+64,state='none',atk='none',seed=951000+i)
            for i,row in enumerate(templates) for dtype in ('float32','bfloat16')]


def slice_inputs(inputs,start,stop,state,atk):
    from .reference import NAMES
    if not 0<=start<stop<=inputs['q'].shape[1]:raise ValueError('invalid token slice')
    return dict({n:inputs[n][:,start:stop].contiguous() for n in NAMES[:7]},initial_state=state,initial_A_state=atk)


def trace_metrics(actual,expected,offset=0):
    from .reference import metric,digest
    assert actual['o'].shape==expected['o'].shape
    return dict(tokens=[dict(token=offset+t,**metric(actual['o'][:,t],expected['o'][:,t])) for t in range(actual['o'].shape[1])],
                final_state=metric(actual['final_state'],expected['final_state']),
                final_A_state=metric(actual['final_A_state'],expected['final_A_state']),
                byte_identical={n:digest(actual[n])==digest(expected[n]) for n in OUTPUTS})


def passed_trace(trace,dtype):
    return all(trace[n]['finite'] and trace[n]['relative_l2']<=1e-4 for n in ('final_state','final_A_state')) and all(
        m['finite'] and (dtype=='bfloat16' or m['relative_l2']<=1e-4) for m in trace['tokens'])
