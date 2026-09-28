"""Standalone unit hooks, with no parent-project or sibling-unit dependency."""
import torch

from ref.calibrate import inputs as generated_inputs
from ref.reference import INPUTS, NAMES, analytical
from ref.stages import reference_stages


def make_inputs(case):
    parameters = {**case['parameters'], 'seed': case['seed']}
    xs, ds = generated_inputs(parameters)
    values = dict(zip(INPUTS, xs))
    B, _, H, _ = values['q'].shape
    HV = values['v'].shape[2]
    shapes = dict(do=values['v'].shape, dht=(B,HV,128,128), dA_T=(B,H,128))
    # Canonical runner inputs are explicit CPU tensors. Public-ABI/native
    # qualification additionally exercises actual None for absent cotangents.
    values.update({n: torch.zeros(shapes[n], dtype=torch.float32) if value is None else value
                   for n, value in ds.items()})
    return values


def validate_inputs(inputs, case=None):
    if set(inputs) != set(INPUTS) | {'do', 'dht', 'dA_T'}:
        raise ValueError('Expected seven inputs and three explicit cotangent tensors')
    q, v = inputs['q'], inputs['v']
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] < 1 or q.shape[1] % 64 or q.shape[1] > 4096 or q.shape[2] < 1 or q.shape[3] != 128:
        raise ValueError('Require B1/T%64=0/T<=4096/positiveH/K128')
    B, T, H, _ = q.shape
    if v.ndim != 4 or v.shape[:2] != (B,T) or v.shape[-1] != 128 or v.shape[2] < 1 or v.shape[2] % H:
        raise ValueError('Invalid grouped value shape')
    HV = v.shape[2]
    shapes = dict(q=q.shape,k=q.shape,v=v.shape,g_atk=(B,T,H),beta_atk=(B,T,H),
                  g=(B,T,HV),beta=(B,T,HV),do=v.shape,dht=(B,HV,128,128),dA_T=(B,H,128))
    for name, tensor in inputs.items():
        if tensor.shape != shapes[name] or tensor.dtype != torch.float32 or tensor.device.type != 'cpu' or not tensor.is_contiguous():
            raise ValueError('Invalid input metadata: ' + name)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Nonfinite input: ' + name)
    for name in ('g', 'g_atk'):
        value = inputs[name]
        if bool((value > 0).any()) or not bool(torch.isfinite(value.reshape(B,T//64,64,-1).sum(2)).all()):
            raise ValueError('Invalid gate domain: ' + name)
    for name in ('beta', 'beta_atk'):
        if bool(((inputs[name] < 0) | (inputs[name] > 1)).any()):
            raise ValueError('Invalid beta domain: ' + name)
    if case and case.get('block_dim', 1) not in (1, 2):
        raise ValueError('Require block_dim1/2')


def reference(inputs):
    validate_inputs(inputs)
    return analytical(*(inputs[n] for n in INPUTS), **{n: inputs[n] for n in ('do', 'dht', 'dA_T')})


def validate_reference(inputs, outputs, case=None):
    if set(outputs) != set(NAMES):
        raise ValueError('Missing/extra reference output')
    for name, source in zip(NAMES, INPUTS):
        value = outputs[name]
        if value.shape != inputs[source].shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid reference output: ' + name)
    # These are schema/finite checks, not the PM's A/B classification. Complete
    # classified qualification separately preserves its nonPASS disclosures.


def _launch(inputs, options):
    from _unit_runner import launch_kernel
    validate_inputs(inputs)
    if options['device'] != 'a5' or options['backend'] != 'cce' or options['block_dim'] not in (1, 2):
        raise ValueError('Require A5/CCE/block_dim1-or2')

    def launch(entry, sources, outputs, scalars):
        result = launch_kernel(entry, tuple(sources.values()) + tuple(outputs.values()) + tuple(scalars.values()), options)
        return dict(zip(outputs, (result,) if len(outputs) == 1 else result))
    return launch


def execute_stages(inputs, options):
    from kernels.pipeline import run
    launch = _launch(inputs, options)
    known = {**inputs, **reference_stages(inputs), 'dout': inputs['do'], 'dat': inputs['dA_T']}

    def independent(entry, sources, outputs, scalars):
        return launch(entry, {name: known[name] for name in sources}, outputs, scalars)
    return run(inputs, independent, retain_stages=True)


def execute(inputs, options):
    from kernels.pipeline import run
    return run(inputs, _launch(inputs, options))
