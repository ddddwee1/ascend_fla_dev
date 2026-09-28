"""PK-05 reference classification, disclosure and public-domain regression."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/pgdn_chunk_bwd'


def refs():
    name = '_pk05_test_ref'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'ref/__init__.py', submodule_search_locations=[str(ROOT / 'ref')])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return tuple(importlib.import_module(name + '.' + item) for item in ('reference', 'classification', 'calibrate'))


def valid():
    return dict(q=torch.zeros(1,64,2,128), k=torch.zeros(1,64,2,128),
                v=torch.zeros(1,64,4,128), g_atk=torch.zeros(1,64,2), g=torch.zeros(1,64,4),
                beta_atk=torch.zeros(1,64,2), beta=torch.zeros(1,64,4),
                do=torch.zeros(1,64,4,128), dht=torch.zeros(1,4,128,128), dA_T=torch.zeros(1,2,128))


@pytest.mark.parametrize('mask', range(1, 8))
@pytest.mark.parametrize('block_dim', (1, 2))
def test_all_cotangent_subsets_and_allowed_blocks(mask, block_dim):
    from ascend_fla.ops import pgdn_chunk_bwd as op
    values = valid()
    for index, name in enumerate(('do', 'dht', 'dA_T')):
        if not mask & (1 << index):
            values[name] = None
    op._validate(**values, block_dim=block_dim, launcher='board')


@pytest.mark.parametrize('name', ('q','k','v','g_atk','g','beta_atk','beta','do','dht','dA_T'))
@pytest.mark.parametrize('launcher', ('inprocess', 'aclnn', 'board'))
def test_bf16_rejection_precedes_tensor_math_and_kernel_import(name, launcher, monkeypatch):
    from ascend_fla.ops import pgdn_chunk_bwd as op
    from torch.utils._python_dispatch import TorchDispatchMode
    values = valid()
    values[name] = values[name].bfloat16()
    monkeypatch.setattr(op, '_pipeline', lambda: pytest.fail('unsupported dtype reached dispatch'))

    class NoTensorOperation(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            pytest.fail(f'dtype rejection must precede tensor math: {function}')

    with NoTensorOperation(), pytest.raises(ValueError, match='requires float32'):
        op.chunk_pgdn_bwd(**values, launcher=launcher)


@pytest.mark.parametrize('options', [
    dict(initial_state=torch.zeros(1,4,128,128)), dict(initial_A_state=torch.zeros(1,2,128)),
    dict(head_first=True), dict(transpose_state_layout=True), dict(cu_seqlens=torch.tensor([0,64])),
    dict(cu_seqlens_cpu=torch.tensor([0,64])), dict(cp_context=object()),
    dict(use_qk_l2norm_in_kernel=False), dict(use_qk_l2norm_in_kernel=1),
    dict(scale=1.), dict(scale=float('nan')), dict(x=2.), dict(eps=1e-5),
    dict(log_atk_scale=torch.tensor(-.2)), dict(log_atk_scale=0.),
    dict(device='a2'), dict(block_dim=3), dict(block_dim=True), dict(launcher='sim')])
def test_options_rejected_before_kernel_import(options, monkeypatch):
    from ascend_fla.ops import pgdn_chunk_bwd as op
    monkeypatch.setattr(op, '_pipeline', lambda: pytest.fail('unsupported options reached dispatch'))
    with pytest.raises(ValueError):
        op.chunk_pgdn_bwd(**valid(), **dict({'launcher': 'board'}, **options))


@pytest.mark.parametrize('name,shape', [
    ('q',(2,64,2,128)), ('q',(0,64,2,128)), ('q',(1,0,2,128)),
    ('q',(1,63,2,128)), ('q',(1,4160,2,128)), ('q',(1,64,0,128)),
    ('q',(1,64,2,64)), ('q',(64,2,128)), ('k',(1,128,2,128)),
    ('v',(1,64,3,128)), ('v',(1,64,0,128)), ('v',(1,64,4,64)),
    ('g_atk',(1,64,4)), ('beta_atk',(1,64,4)), ('g',(1,64,2)), ('beta',(1,64,2)),
    ('do',(1,64,2,128)), ('dht',(1,2,128,128)), ('dA_T',(1,4,128))])
def test_shape_boundaries_rejected(name, shape):
    from ascend_fla.ops import pgdn_chunk_bwd as op
    values = valid()
    values[name] = torch.zeros(shape)
    with pytest.raises(ValueError):
        op._validate(**values, launcher='board')


@pytest.mark.parametrize('name,value,message', [
    ('q',float('nan'),'finite'), ('k',float('inf'),'finite'), ('v',-float('inf'),'finite'),
    ('do',float('nan'),'finite'), ('dht',float('inf'),'finite'), ('dA_T',float('nan'),'finite'),
    ('g',.1,'nonpositive'), ('g_atk',.1,'nonpositive'),
    ('g',-1e37,'chunk sum'), ('g_atk',-1e37,'chunk sum'),
    ('beta',-.1,'in \\[0,1\\]'), ('beta',1.1,'in \\[0,1\\]'),
    ('beta_atk',-.1,'in \\[0,1\\]'), ('beta_atk',1.1,'in \\[0,1\\]')])
def test_value_policy(name, value, message):
    from ascend_fla.ops import pgdn_chunk_bwd as op
    values = valid()
    values[name].fill_(value)
    with pytest.raises(ValueError, match=message):
        op._validate(**values, launcher='board')


@pytest.mark.parametrize('radius', (0., 5e-13, 1e-12, 2e-12, .03, 2.))
def test_normalization_zero_and_near_zero_rows_are_accepted(radius):
    from ascend_fla.ops import pgdn_chunk_bwd as op
    values = valid()
    values['q'][..., 0] = radius
    values['k'][..., 0] = radius
    op._validate(**values, launcher='board')


def test_missing_cotangents_noncontiguous_and_wrong_device():
    from ascend_fla.ops import pgdn_chunk_bwd as op
    values = valid()
    with pytest.raises(ValueError, match='at least one'):
        op._validate(**{**values, 'do': None, 'dht': None, 'dA_T': None}, launcher='board')
    with pytest.raises(ValueError, match='one npu'):
        op._validate(**values)
    values['k'] = values['k'].transpose(1,2).contiguous().transpose(1,2)
    with pytest.raises(ValueError, match='contiguous'):
        op._validate(**values, launcher='board')


def test_literal_fp32_clamp_branch_is_preserved_when_lifted():
    ref, _, _ = refs()
    raw = torch.tensor([[[1e-12, 0., 0.]]], dtype=torch.float32)
    assert raw.double().norm() < 1e-12
    normalized, _, _, active = ref.normalization(raw.double(), fp32_branch=True)
    assert bool(active.all())
    assert normalized[0, 0, 0] == 1.
    assert not bool(ref.normalization(raw.double())[3].any())
    leaf = raw.clone().requires_grad_()
    grad, = torch.autograd.grad(torch.nn.functional.normalize(leaf, dim=-1).sum(), (leaf,))
    assert grad[0, 0, 0] == 0.


def test_classifier_candidate_independence_and_bad_output_controls():
    ref, classification, cal = refs()
    case = next(c for c in cal.cases() if c['id'] == 'c1_h1_hv2_m7')
    xs, ds = cal.inputs(case)
    b = ref.analytical(*xs, **ds)
    auxiliary = ref.analytical(*(x.double() for x in xs), **{n: x.double() for n, x in ds.items()},
                               fp32_branch=True, auxiliary=True)
    classes = classification.classify(auxiliary)
    original = {n: c['ordinary'].clone() for n, c in classes.items()}
    assert classification.compare(b, {'B': b}, classes)['ordinary_budget_satisfied']
    for multiplier in (0., -1., 1.01):
        actual = {n: x * multiplier for n, x in b.items()}
        assert not classification.compare(actual, {'B': b}, classes)['ordinary_budget_satisfied']
    actual = dict(b)
    actual['dk'] = torch.full_like(actual['dk'], float('nan'))
    assert not classification.compare(actual, {'B': b}, classes)['ordinary_budget_satisfied']
    for name in ref.NAMES:
        assert torch.equal(classification.classify(auxiliary)[name]['ordinary'], original[name])
    with pytest.raises(ValueError, match='Seven named'):
        classification.compare({n: x for n, x in b.items() if n != 'dk'}, {'B': b}, classes)
    with pytest.raises(ValueError, match='expected FP32'):
        classification.compare({**b, 'dk': b['dk'].double()}, {'B': b}, classes)


def test_disclosure_roundtrip_preserves_positions_values_and_zero_denominator(tmp_path):
    ref, classification, _ = refs()
    auxiliary = dict(gradients={}, raw_gradients={}, scales={}, zero_reasons={})
    actual = {}
    for name in ref.NAMES:
        auxiliary['gradients'][name] = torch.zeros(2, 3, dtype=torch.float64)
        auxiliary['raw_gradients'][name] = torch.zeros(2, 3, dtype=torch.float64)
        auxiliary['scales'][name] = torch.zeros(2, 3, dtype=torch.float64)
        auxiliary['zero_reasons'][name] = torch.full((2, 3), 2, dtype=torch.uint8)
        actual[name] = torch.tensor([[0., -0., .125], [.125, 0., 0.]])
    classes = classification.classify(auxiliary)
    path = tmp_path / 'complete.jsonl'
    receipt = classification.write_disclosures(path, actual, auxiliary, classes)
    assert receipt['elements'] == 42
    reconstructed, positions = {}, {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if 'gradient' in row:
            name = row['gradient']
            assert row['full_reference64_norm'] == 0.
            reconstructed[name] = torch.empty(6)
            positions[name] = set()
        if 'start' in row:
            value = row['values']
            for index in range(row['start'], row['start'] + row['count']):
                assert index not in positions[name]
                positions[name].add(index)
                reconstructed[name][index] = value['candidate']
                assert value['global_norm_contribution'] == ('undefined' if value['candidate'] == 0. else 'infinity')
                assert value['absolute_error'] == abs(value['candidate'])
                assert value['reference64'] == value['local_scale'] == value['threshold'] == 0.
    for name in ref.NAMES:
        assert positions[name] == set(range(6))
        assert torch.equal(reconstructed[name].view(torch.uint8), actual[name].flatten().view(torch.uint8))


def test_classifier_inclusive_threshold_and_no_global_scale_substitution():
    ref, classification, _ = refs()
    threshold = 64 * 2**-23
    above = torch.nextafter(torch.tensor(threshold, dtype=torch.float64), torch.tensor(float('inf'), dtype=torch.float64))
    values = torch.tensor([threshold, above.item(), 1e-10, 1e6], dtype=torch.float64)
    scale = torch.tensor([1., 1., 1e-10, 1e6], dtype=torch.float64)
    auxiliary = dict(gradients={n: values.clone() for n in ref.NAMES},
                     scales={n: scale.clone() for n in ref.NAMES},
                     zero_reasons={n: torch.zeros(4, dtype=torch.uint8) for n in ref.NAMES})
    classes = classification.classify(auxiliary)
    for name in ref.NAMES:
        assert classes[name]['disclosure'].tolist() == [True, False, False, False]


@pytest.mark.parametrize('case_id', ('c1_h1_hv2_m7', 'c2_h3_hv6_m7', 'c3_h1_hv4_m4',
                                    'axis_at_m7', 'axis_below_m1'))
def test_independent_stage_composition_matches_analytical_reference(case_id):
    ref, classification, cal = refs()
    stages = importlib.import_module(ref.__package__ + '.stages')
    case = next(c for c in cal.cases() if c['id'] == case_id)
    xs, ds = cal.inputs(case)
    actual = stages.reference_stages({**dict(zip(ref.INPUTS, xs)), **ds})
    b = ref.analytical(*xs, **ds)
    auxiliary = ref.analytical(*(x.double() for x in xs),
                               **{n: None if x is None else x.double() for n, x in ds.items()},
                               fp32_branch=True, auxiliary=True)
    classes = classification.classify(auxiliary)
    assert classification.compare({n: actual[n] for n in ref.NAMES}, {'B': b}, classes)['ordinary_budget_satisfied']
    assert actual['checkpoints'].shape == (1, case['T']//64, case['HV'], 128, 128)
    assert actual['tape'].shape == (1, case['HV'], 64, 128, 128)
    assert torch.count_nonzero(actual['checkpoints'][:, 0]) == 0
    assert torch.count_nonzero(actual['tape'][:, :, 0]) == 0
    assert torch.equal(actual['q_raw_norm'] >= 1e-12, xs[0].norm(dim=-1) >= 1e-12)
    assert torch.equal(actual['k_raw_norm'] >= 1e-12, xs[1].norm(dim=-1) >= 1e-12)


def test_atk_final_cotangent_is_seeded_once_per_key_head():
    ref, classification, cal = refs()
    case = dict(B=1, T=64, H=2, HV=2, seed=95233, kind='random', mask=4)
    xs, ds = cal.inputs(case)
    baseline = ref.analytical(*xs, **ds)
    for ratio in (2, 4, 8):
        expanded = list(xs)
        for index in (2, 4, 6):
            expanded[index] = xs[index].repeat_interleave(ratio, 2)
        actual = ref.analytical(*expanded, **ds)
        for name in ('dk', 'dg_atk', 'dbeta_atk'):
            assert torch.equal(actual[name].view(torch.uint8), baseline[name].view(torch.uint8))
        for name in ('dq', 'dv', 'dg', 'dbeta'):
            assert torch.count_nonzero(actual[name]) == 0
        wrong = ref.analytical(*expanded, **{**ds, 'dA_T': ds['dA_T'] * ratio})
        for name in ('dk', 'dg_atk', 'dbeta_atk'):
            assert classification.numbers(wrong[name], baseline[name])['relative_l2'] > 1e-4


def test_native_stage_checker_rejects_corruption_and_nonfinite():
    refs()
    helper = importlib.import_module('_pk05_test_ref.verification')
    expected = dict(checkpoints=torch.tensor([1., 2.]), k_write=torch.tensor([3., 4.]))
    assert all(row['budget_satisfied'] for row in helper.internal_comparison(expected, expected).values())
    for wrong in (torch.zeros(2), torch.tensor([float('nan'), 2.]), torch.tensor([1., -2.])):
        actual = {**expected, 'checkpoints': wrong}
        assert not helper.internal_comparison(actual, expected)['checkpoints']['budget_satisfied']
    with pytest.raises(ValueError, match='schema'):
        helper.internal_comparison({'k_write': expected['k_write']}, expected)
    with pytest.raises(ValueError, match='shape/dtype'):
        helper.internal_comparison({**expected, 'checkpoints': expected['checkpoints'].double()}, expected)


@pytest.mark.parametrize('case_set', ('original_frozen_256', 'supplemental_dense_clamp'))
def test_bd_comparison_requires_complete_matching_inputs_and_stage_bytes(case_set):
    import copy
    _, _, grid = refs()
    if case_set == 'supplemental_dense_clamp':
        grid = importlib.import_module('_pk05_test_ref.clamp_cases')
    helper = importlib.import_module('_pk05_test_ref.verification')
    schema = json.loads((ROOT / 'contract.json').read_text())
    rows = [dict(case=case, cpu_input_sha256={n: 'input' for n in schema['inputs']},
                 stage_sha256={n: 'stage' for n in schema['stages']},
                 public_sha256={n: 'output' for n in helper.NAMES},
                 public_schema={n: 'schema' for n in helper.NAMES}) for case in grid.cases()]
    left = dict(block_dim=1, case_set=case_set, cases=rows, all_required_checks_complete=True)
    right = dict(block_dim=2, case_set=case_set, cases=copy.deepcopy(rows), all_required_checks_complete=True)
    assert helper.compare_runs(left, right)['all_byte_identical']
    for field in ('cpu_input_sha256', 'stage_sha256', 'public_sha256'):
        changed = copy.deepcopy(right)
        changed['cases'][-1][field][next(iter(changed['cases'][-1][field]))] = 'different'
        assert not helper.compare_runs(left, changed)['all_byte_identical']
    incomplete = copy.deepcopy(right)
    incomplete['cases'].pop()
    with pytest.raises(ValueError, match='coverage'):
        helper.compare_runs(left, incomplete)
    left_partial = {**left, 'cases': left['cases'][:1]}
    right_partial = {**right, 'cases': right['cases'][:1]}
    with pytest.raises(ValueError, match='complete frozen'):
        helper.compare_runs(left_partial, right_partial)
    with pytest.raises(ValueError, match='case sets'):
        helper.compare_runs(left, {**right, 'case_set': 'different'})


@pytest.mark.parametrize('mode', ('q', 'k', 'both'))
def test_dense_clamp_still_has_ordinary_gradients_with_agreeing_references(mode):
    ref, classification, _ = refs()
    grid = importlib.import_module('_pk05_test_ref.clamp_cases')
    oracle = importlib.import_module('_pk05_test_ref.oracle')
    case = next(c for c in grid.cases() if c['T'] == 64 and c['mask'] == 7 and c['clamp_mode'] == mode)
    xs, ds = grid.inputs(case)
    a = oracle.autograd(*xs, **ds)
    b = ref.analytical(*xs, **ds)
    auxiliary = ref.analytical(*(x.double() for x in xs),
                               **{n: x.double() for n, x in ds.items()},
                               fp32_branch=True, auxiliary=True)
    classes = classification.classify(auxiliary)
    result = classification.compare(b, {'A': a}, classes)
    assert result['ordinary_budget_satisfied']
    assert all(result['gradients'][n]['ordinary_count'] > 0 for n in ('dq', 'dk'))
