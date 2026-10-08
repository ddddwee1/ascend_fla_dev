"""Canonical FP32 dual-state unit; BF16 storage is verified natively."""
from ref.oracle import oracle
from ref.reference import make_inputs, metric, reference as independent_reference
from ref.reference import validate_inputs, validate_reference as validate_b


def reference(inputs):
    validate_inputs(inputs)
    return oracle(inputs)


def validate_reference(inputs, outputs, case=None):
    validate_b(inputs, outputs, case)
    other = independent_reference(inputs)
    for name, value in outputs.items():
        row = metric(value, other[name])
        if not row['finite'] or row['relative_l2'] > 1e-5:
            raise AssertionError(f'A/B calibration failed for {name}: {row}')


def execute(inputs, options):
    from _unit_runner import launch_kernel
    from kernels.pipeline import run
    validate_inputs(inputs)
    if options['device'] != 'a5' or options['backend'] != 'cce':
        raise ValueError('Only A5/CCE is declared')
    if options['block_dim'] not in (1, 2, 4, 8, 16, 28):
        raise ValueError('block_dim outside candidate grid')
    def launch(entry, sources, outputs, scalars):
        actual = launch_kernel(entry, tuple(sources.values())+tuple(outputs.values())
                               +tuple(scalars.values()), options)
        return dict(zip(outputs, actual))
    actual = run(inputs, launch)
    comparisons = {}
    for label, expected in [('A', oracle(inputs)), ('B', independent_reference(inputs))]:
        comparisons[label] = {name: metric(actual[name], expected[name]) for name in actual}
    options.setdefault('_execution_evidence', []).append(dict(dual_oracle=comparisons))
    if any(not row['finite'] or row['relative_l2'] > 1e-4
           for group in comparisons.values() for row in group.values()):
        raise AssertionError(comparisons)
    return actual
