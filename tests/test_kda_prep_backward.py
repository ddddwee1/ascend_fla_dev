"""Training graph/ABI checks with explicit test-only CPU vendor substitutes."""
import itertools
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ascend_fla.ops.kda import autograd, chunk, chunk_bwd


@pytest.mark.parametrize('kind', ['product', 'bounded_product', 'constant'])
def test_compensated_arithmetic_preserves_residual_and_primary_infinity(kind, tmp_path):
    pytest.importorskip('ascriptor')
    root = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/kda_prep'
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    probe = load('bf09_arithmetic_test', root / 'backward_arithmetic.py')
    runner = load('bf09_arithmetic_runner', root / '_unit_runner.py')
    a, b, expected_high, expected_low = probe.inputs_and_reference(kind)
    high, low = runner.launch_kernel(probe.make_probe(kind),
        (a, b, torch.full_like(a, float('nan')), torch.full_like(a, float('nan'))),
        dict(device='a5', backend='cce', block_dim=1, launcher='sim',
             out_dir=tmp_path, timeout=45., board=None))
    probe.check_outputs(high, low, expected_high, expected_low)
    # A rounded product alone loses the discriminating cancellation residual.
    assert torch.count_nonzero(expected_low)
    with pytest.raises(AssertionError, match='residual'):
        probe.check_outputs(high, torch.zeros_like(low), expected_high, expected_low)


def test_endpoint_classification_rejects_hidden_ordinary_or_zero_errors(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/kda_prep'
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    support = load('endpoint_support', 'backward_native_support.py')
    precision = load('endpoint_precision', 'ref/calibrate.py')
    budget = json.loads((root / 'backward_budgets.json').read_text())['groups']['norm:f32:dx']
    high = torch.tensor([1., 2.], dtype=torch.float64)
    mask = torch.tensor([False, True])
    def compare(actual, old, label):
        return support.gradient_record('norm:f32:dx', torch.tensor(actual), high,
            torch.tensor(old), precision, budget, tmp_path / label,
            endpoint_zero_mask=mask, endpoint_authority='D-PM-55 synthetic verifier control')
    correct = compare([1., 0.], [1., 0.], 'authorized_zero')
    assert correct['passed'] and correct['ordinary_elements'] == 1
    assert correct['to_fp64']['relative_l2'] > .8  # Still disclosed, never erased.
    assert not compare([1.001, 0.], [1., 0.], 'bad_ordinary')['passed']
    assert not compare([1., 1.e-10], [1., 0.], 'nonzero_candidate')['passed']
    assert not compare([1., 0.], [1., 1.e-10], 'nonzero_cpu')['passed']


def test_bf16_dual_ulp_owner_rule_preserves_both_required_lines(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/kda_prep'
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    support = load('dual_ulp_support', 'backward_native_support.py')
    precision = load('dual_ulp_precision', 'ref/calibrate.py')
    budget = json.loads((root / 'backward_budgets.json').read_text())['groups']['beta:bf16:dbeta']
    def value(bits):
        return torch.tensor([bits-65536 if bits >= 32768 else bits], dtype=torch.int16).view(torch.bfloat16)
    def compare(candidate, previous, label):
        return support.gradient_record('beta:bf16:dbeta', value(candidate), value(46111).double(),
            value(previous), precision, budget, tmp_path / label, condition=torch.ones(1))
    improved = compare(46111, 46120, 'old_nine_ulp')
    assert improved['passed'] and improved['old_host_ulp_policy']['disclosed_elements'] == 1
    detail = json.loads((tmp_path / 'old_nine_ulp' / '00000.json').read_text())['locations'][0]
    assert detail['old_to_fp64_ulp'] == 9 and detail['old_line_disclosure']
    assert not compare(46113, 46120, 'candidate_two_ulp')['passed']
    assert not compare(46110, 46112, 'both_within_one_but_two_apart')['passed']
    assert compare(46111, 46112, 'both_required_pass')['passed']


def test_native_flush_classification_is_not_cpu_accuracy_pass(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/kda_prep'
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    support = load('flush_support', 'backward_native_support.py')
    precision = load('flush_precision', 'ref/calibrate.py')
    budget = json.loads((root / 'backward_budgets.json').read_text())['groups']['beta:f32:dbeta']
    def compare(golden, actual, label):
        cpu = torch.tensor([1., golden], dtype=torch.float32)
        return support.gradient_record('beta:f32:dbeta', torch.tensor([1., actual]), cpu.double(), cpu,
            precision, budget, tmp_path / label, native_flush_mask=torch.tensor([False, True]), cpu_fp32=cpu)
    valid = compare(1e-39, 0., 'subnormal_zero')
    assert valid['classification_complete'] and valid['ordinary_criteria_passed']
    assert not valid['passed'] and not valid['native_flush']['cpu_correctness_pass']
    assert not compare(1e-30, 0., 'normal_hidden_as_flush')['classification_complete']
    assert not compare(1e-39, 1e-39, 'nonzero_hidden_as_flush')['classification_complete']


@pytest.fixture
def preparation_abi(monkeypatch):
    events = []

    class PrepABI:
        @staticmethod
        def _check_source(name, value):
            assert value.device.type == 'cpu' and value.is_contiguous()
            assert value.dtype in (torch.bfloat16, torch.float32)

        @staticmethod
        def prepare_backward(*args):
            events.append('raw_backward_ready')

        @staticmethod
        def norm(value, dtype, **options):
            assert events[:3] == ['forward_ready', 'backward_ready', 'raw_backward_ready']
            events.append('norm')
            z = value.float()
            return (z / (z.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(dtype)

        @staticmethod
        def gate(value, alog, bias, **options):
            assert events[:3] == ['forward_ready', 'backward_ready', 'raw_backward_ready']
            events.append('gate')
            return -alog.float().exp().view(-1, 1) * torch.nn.functional.softplus(
                value.float() + bias.float().view(value.shape[-2:]))

        @staticmethod
        def beta(value, **options):
            assert events[:3] == ['forward_ready', 'backward_ready', 'raw_backward_ready']
            events.append('beta')
            return value.float().sigmoid()

        @staticmethod
        def norm_backward(value, sensitivity, **options):
            events.append('norm_backward')
            assert sensitivity.dtype == torch.bfloat16 and sensitivity.is_contiguous()
            with torch.enable_grad():
                leaf = value.detach().clone().requires_grad_()
                return torch.autograd.grad(PrepABI.norm(leaf, torch.bfloat16), leaf, sensitivity)[0]

        @staticmethod
        def gate_backward(value, alog, bias, sensitivity, **options):
            events.append('gate_backward')
            assert sensitivity.dtype == torch.float32 and sensitivity.is_contiguous()
            with torch.enable_grad():
                leaves = [t.detach().clone().requires_grad_() for t in (value, alog, bias)]
                return torch.autograd.grad(PrepABI.gate(*leaves), leaves, sensitivity)

        @staticmethod
        def beta_backward(source, probability, sensitivity, **options):
            events.append('beta_backward')
            assert sensitivity.dtype == torch.float32 and sensitivity.is_contiguous()
            return (sensitivity * probability * (1 - probability)).to(source.dtype)

    monkeypatch.setattr(autograd, '_prep_runtime', lambda: PrepABI)
    monkeypatch.setattr(chunk, '_compiled_chain', lambda *a: events.append('forward_ready'))
    monkeypatch.setattr(chunk_bwd, '_compiled_chain', lambda *a: events.append('backward_ready'))
    monkeypatch.setattr(autograd, '_layout_runtime', lambda: SimpleNamespace(
        cast=lambda x, dtype, **kw: x.to(dtype).contiguous()))
    monkeypatch.setattr(chunk, '_prepare_inputs', lambda *a, **kw: pytest.fail('old host preparation called'))
    monkeypatch.setattr(autograd, '_prepare_inputs', lambda *a, **kw: pytest.fail('old host training called'))
    return events


def inputs(dtype):
    generator = torch.Generator().manual_seed(8008)
    q = torch.randn((1, 64, 2, 128), generator=generator).to(dtype)
    k = torch.randn(q.shape, generator=generator).to(dtype)
    g = torch.randn((1, 64, 4, 128), generator=generator).to(dtype)
    beta = torch.randn((1, 64, 4), generator=generator).to(dtype)
    a = torch.linspace(-2., -1., 4).to(dtype)
    bias = torch.linspace(-.2, .2, 512).to(dtype)
    return q, k, g, beta, a, bias


def independent(values, flags):
    q, k, g, beta, a, bias = values
    if flags[0]:
        x, y = q.float(), k.float()
        q = (x / (torch.sum(x.square(), -1, keepdim=True) + 1e-6).sqrt()).bfloat16()
        k = (y / (torch.sum(y.square(), -1, keepdim=True) + 1e-6).sqrt()).bfloat16()
    if flags[1]:
        g = -torch.exp(a.float()).view(-1, 1) * torch.nn.functional.softplus(
            g.float() + bias.float().view(g.shape[-2:]))
    if flags[2]:
        beta = torch.sigmoid(beta.float())
    return q, k, g, beta


@pytest.mark.parametrize('flags', list(itertools.product((False, True), repeat=3)))
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('selected', [(0, 1, 2, 3, 4, 5), (0, 2, 5), (4,), (5,)])
def test_flag_and_partial_gradient_graph(preparation_abi, flags, dtype, selected):
    values = tuple(t.requires_grad_(i in selected) for i, t in enumerate(inputs(dtype)))
    before = [t.detach().clone() for t in values]
    reference_values = tuple(t.detach().clone().requires_grad_(i in selected) for i, t in enumerate(values))
    options = dict(use_qk_l2norm_in_kernel=flags[0], use_gate_in_kernel=flags[1],
                   use_beta_sigmoid_in_kernel=flags[2])
    result = autograd._prepare_training_inputs(*values[:4], A_log=values[4], dt_bias=values[5], **options)
    reference = independent(reference_values, flags)
    generator = torch.Generator().manual_seed(8108)
    sensitivities = [torch.randn(t.shape, generator=generator).to(t.dtype) for t in result]
    actual_outputs = [t for t in result if t.requires_grad]
    expected_outputs = [t for t in reference if t.requires_grad]
    if actual_outputs:
        gs = [g for t, g in zip(result, sensitivities) if t.requires_grad]
        actual = torch.autograd.grad(actual_outputs, [values[i] for i in selected], gs, allow_unused=True)
        expected = torch.autograd.grad(expected_outputs, [reference_values[i] for i in selected], gs, allow_unused=True)
        for index, got, want in zip(selected, actual, expected):
            if want is None:
                assert got is None
            else:
                assert got.dtype == values[index].dtype
                assert torch.isfinite(got).all() and torch.count_nonzero(got)
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    else:
        assert not expected_outputs  # Disabled parameter flags create no edge.
    for i, (got, want) in enumerate(zip(result, reference)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        if not flags[(0, 0, 1, 2)[i]]:
            assert got is values[i]
    for got, old in zip(values, before):
        torch.testing.assert_close(got, old, rtol=0, atol=0)
    if not any(flags):
        assert preparation_abi == []


@pytest.mark.parametrize('types', list(itertools.product((torch.bfloat16, torch.float32), repeat=3)))
@pytest.mark.parametrize('selected', [(2,), (4,), (5,), (2, 4, 5)])
def test_independent_gate_parameter_dtypes(preparation_abi, types, selected):
    values = list(inputs(torch.float32))
    for i, dtype in zip((2, 4, 5), types):
        values[i] = values[i].to(dtype).requires_grad_(i in selected)
    result = autograd._prepare_training_inputs(*values[:4], A_log=values[4], dt_bias=values[5], use_gate_in_kernel=True)[2]
    assert result.dtype == torch.float32 and result.requires_grad
    sensitivity = torch.randn(result.shape, generator=torch.Generator().manual_seed(8118))
    grads = torch.autograd.grad(result, [values[i] for i in selected], sensitivity)
    for index, grad in zip(selected, grads):
        assert grad.dtype == values[index].dtype
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad)
    assert preparation_abi.count('gate_backward') == 1


def _prep_runtime_module():
    root = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/kda_prep'
    spec = importlib.util.spec_from_file_location('bf09p1_prep_runtime', root / 'runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('namespace', ['chunk', 'decode'])
def test_prep_block_dim_ceiling_tracks_the_platform_capability(namespace):
    """``kda_prep`` keeps its OWN copy of the block_dim ceiling.

    BF-09-P1 measured (A5 real hardware, card 2 serial, real shape
    B1/T4096/HV32) that the prep backward chain scales essentially linearly
    with ``block_dim`` all the way to 28 — 56 of 56 physical vector cores —
    and that every ``block_dim`` in 1..28 returns BITWISE IDENTICAL gradients.
    The chunk ceiling of 4 is therefore not a limit of these kernels; it is
    inherited from the upstream ``kda_fwd``/``kda_bwd`` contract that governs
    *different* kernels (``gaps.json`` ``block-dim-ceiling``), and the decode
    namespace of this very unit already runs the same factories at 28.

    Nothing locked this duplicated literal to ``platform.py`` before, so a
    future task could lift the ops-layer ceiling and leave the prep unit
    silently pinned (or vice versa). Pin the two together so whoever lifts one
    is pointed straight at the other.
    """
    from ascend_fla import platform

    runtime = _prep_runtime_module()
    assert tuple(runtime._BLOCK_DIMS[namespace]) == platform.supported_block_dim('a5', namespace), (
        f"kda_prep/runtime.py _BLOCK_DIMS[{namespace!r}] = "
        f"{tuple(runtime._BLOCK_DIMS[namespace])} but platform.supported_block_dim"
        f"('a5', {namespace!r}) = {platform.supported_block_dim('a5', namespace)}; "
        "lift both together or neither (see docs/research/kda_prep_backward_p1.md)")


def test_chunk_compile_includes_future_training_vendors(monkeypatch):
    events = []
    runtime = SimpleNamespace(prepare=lambda *a: events.append('prep_forward'),
                              prepare_backward=lambda *a: events.append('prep_backward'))
    monkeypatch.setattr(chunk, '_prep_runtime', lambda: runtime)
    monkeypatch.setattr(chunk, '_layout_runtime', lambda: SimpleNamespace(prepare=lambda *a: events.append('layout')))
    monkeypatch.setattr(chunk, 'kda_fwd_kernels', lambda *a: {})
    chunk._compiled_chain.cache_clear()
    try:
        chunk._compiled_chain('a5', 1, 'stable')
        assert events == ['layout', 'prep_forward', 'prep_backward']
    finally:
        chunk._compiled_chain.cache_clear()
