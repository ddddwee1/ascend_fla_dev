"""Pinned recurrent A versus independent block-solve B, and public ABI gates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from ascend_fla.ops import pgdn_chunk_fwd as public

ROOT = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/pgdn_chunk_fwd'
PIN = 'e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2'
NAIVE_SHA256 = '3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8'
NAMES = ('q', 'k', 'v', 'g_atk', 'g', 'beta_atk', 'beta')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load('pgdn_test_reference', ROOT / 'ref/reference.py')
CASES = [c for c in json.loads((ROOT / 'contract.json').read_text())['cases'] if c['block_dim'] == 1]


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope='module')
def oracle():
    path = os.environ.get('FLA_PGDN_NAIVE')
    if not path:
        pytest.skip(f'set FLA_PGDN_NAIVE to precond_gated_delta_rule/naive.py at {PIN}')
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == NAIVE_SHA256, 'FLA PGDN naive pin mismatch'
    return load('fla_pgdn_test_naive', path).naive_recurrent_precond_gated_delta_rule


def data(time=64, h=2, hv=4, **params):
    return ref.make_inputs({'seed': 690319, 'parameters': dict(B=1, T=time, H=h, HV=hv, **params)})


def compare(got, expected, budget=1e-5):
    assert got.shape == expected.shape
    assert bool(torch.isfinite(got).all())
    delta = got.float() - expected.float()
    assert float(delta.norm() / expected.float().norm().clamp_min(1e-30)) <= budget
    torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_independent_block_solve_against_pinned_recurrence(case, oracle):
    inputs = ref.make_inputs(case)
    before = {n: x.clone() for n, x in inputs.items()}
    got = ref.reference(inputs)
    expected = oracle(*(inputs[n] for n in NAMES), output_final_state=True)
    for name, value in zip(ref.OUTPUTS, expected):
        compare(got[name], value)
    for name in inputs:
        assert torch.equal(inputs[name], before[name]), name


@pytest.mark.parametrize('ratio', [1, 2, 4, 8])
def test_grouped_heads_equal_independent_single_head_calls(ratio):
    inputs = data(192, h=2, hv=2*ratio)
    got = ref.reference(inputs)
    for value_head in range(2*ratio):
        key_head = value_head // ratio
        one = {name: x[:, :, key_head:key_head+1].contiguous() for name, x in inputs.items()
               if name in ('q', 'k', 'g_atk', 'beta_atk')}
        one.update({name: inputs[name][:, :, value_head:value_head+1].contiguous() for name in ('v', 'g', 'beta')})
        one['initial_state'] = torch.zeros(1, 1, 128, 128)
        expected = ref.reference(one)
        compare(got['o'][:, :, value_head], expected['o'][:, :, 0])
        compare(got['final_state'][:, value_head], expected['final_state'][:, 0])
        compare(got['final_A_state'][:, key_head], expected['final_A_state'][:, 0])


def test_using_preconditioned_key_for_read_correction_is_detectably_wrong():
    inputs = data(192, h=1, hv=1, gate_scale=0.03)
    stages = ref.reference_stages(inputs)
    q, read, write = (stages[n][:, :, 0] for n in ('q_norm', 'k_read', 'k_write'))
    state = torch.zeros(1, 128, 128)
    wrong = []
    for t in range(192):
        state = state * inputs['g'][:, t, 0].exp()[:, None, None]
        delta = inputs['beta'][:, t, 0, None] * (inputs['v'][:, t, 0] - torch.einsum('bk,bkv->bv', write[:, t], state))
        state = state + write[:, t, :, None] * delta[:, None, :]
        wrong.append(torch.einsum('bk,bkv->bv', q[:, t] * 128**-0.5, state))
    wrong = torch.stack(wrong, dim=1).unsqueeze(2)
    assert float((wrong-stages['o']).norm()/stages['o'].norm()) > 0.01
    assert not torch.equal(read, write)


# --- PK-08 traversal invariants -------------------------------------------
# `output_vf` runs the full constant j trip count with no `j <= i` guard, and
# `scores_vf` walks the causal triangle as BLK x BLK blocks. Both rewrites are
# bitwise-neutral only while the invariants below hold, and both are the silent
# kind of defect if an invariant is ever broken: the output would stay finite
# and correctly shaped. These tests pin them, AGENTS.md section 6.5 step 2.
BLK_UNITS = ('pgdn_chunk_fwd', 'pgdn_chunk_fwd_bf16')


def stage_block_edge(unit):
    """BLK as written in the unit's kernel source, read without importing ascriptor."""
    import ast
    source = (ROOT.parent / unit / 'kernels/stages.py').read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'BLK':
            return int(ast.literal_eval(node.value))
    raise AssertionError(f'{unit}: kernels/stages.py declares no BLK')


@pytest.mark.parametrize('unit', BLK_UNITS)
def test_scores_block_edge_divides_the_chunk(unit):
    assert 64 % stage_block_edge(unit) == 0


@pytest.mark.parametrize('unit', BLK_UNITS)
def test_blocked_traversal_covers_the_causal_triangle_exactly(unit):
    """Replay `scores_vf`'s loop nest in Python: same coverage, fewer iterations."""
    blk = stage_block_edge(unit)
    score_pairs, lower_pairs, executed, guarded = [], [], 0, 0
    for ibase in range(0, 64, blk):
        for ii in range(blk):
            i = ibase + ii
            for jbase in range(0, ibase, blk):          # sub-diagonal: unguarded
                for jj in range(blk):
                    executed += 1
                    j = jbase + jj
                    score_pairs.append((i, j))
                    lower_pairs.append((i, j))          # j < i holds by construction
            for jj in range(blk):                       # diagonal block: guarded
                executed += 1
                guarded += 1
                if jj <= ii:
                    score_pairs.append((i, ibase + jj))
                    if jj < ii:
                        lower_pairs.append((i, ibase + jj))
    assert sorted(score_pairs) == sorted((i, j) for i in range(64) for j in range(64) if j <= i)
    assert sorted(lower_pairs) == sorted((i, j) for i in range(64) for j in range(64) if j < i)
    assert len(score_pairs) == len(set(score_pairs))    # every entry written once
    assert len(lower_pairs) == len(set(lower_pairs))
    for pairs in (score_pairs, lower_pairs):            # per row, j still increases
        for row in range(64):
            columns = [j for i, j in pairs if i == row]
            assert columns == sorted(columns)
    # The whole point of the rewrite: the flat guarded form executed 64*64.
    assert executed == 64 * blk * (64 // blk - 1) // 2 + 64 * blk
    assert executed < 64 * 64
    assert guarded == 64 * blk


@pytest.mark.parametrize('time,hv', [(64, 1), (64, 2), (128, 2), (192, 4)])
def test_acausal_scores_are_exact_zero_so_output_needs_no_guard(time, hv):
    """`output_vf` sums all 64 columns; the acausal ones must contribute nothing."""
    stages = ref.reference_stages(data(time, h=1, hv=hv, gate_scale=1.0))
    acausal = ~torch.ones(64, 64, dtype=torch.bool).tril()
    score, lower, delta = stages['score'], stages['lower'], stages['delta']
    assert torch.equal(score[..., acausal], torch.zeros_like(score[..., acausal]))
    assert torch.equal(lower[..., ~torch.ones(64, 64, dtype=torch.bool).tril(-1)],
                       torch.zeros_like(lower[..., ~torch.ones(64, 64, dtype=torch.bool).tril(-1)]))
    assert bool(torch.isfinite(delta).all()), 'a non-finite delta would turn 0*delta into NaN'
    # Replay the unguarded accumulation against the guarded one, bit for bit.
    flat_score = score.reshape(-1, 64, 64)
    flat_delta = delta.reshape(-1, 64, 128)
    for tile in range(flat_score.shape[0]):
        for row in range(64):
            guarded = torch.zeros(128)
            unguarded = torch.zeros(128)
            for col in range(64):
                term = flat_delta[tile, col] * flat_score[tile, row, col]
                unguarded = unguarded + term
                if col <= row:
                    guarded = guarded + term
            assert torch.equal(guarded, unguarded)


@pytest.mark.parametrize('norm', [0., 1e-6, 1e-4, 1e-2])
def test_normalization_fork_quantified(norm):
    inputs = data(h=1, hv=1, norm=norm)
    got = ref.reference_stages(inputs)['k_read']
    naive = torch.nn.functional.normalize(inputs['k'], dim=-1)
    chunk_formula = inputs['k'] / (inputs['k'].square().sum(-1, keepdim=True)+1e-6).sqrt()
    torch.testing.assert_close(got, naive, atol=0, rtol=0)
    expected_delta = 0. if norm == 0 else 1.-norm/(norm*norm+1e-6)**.5
    assert float((naive-chunk_formula).abs().max()) == pytest.approx(expected_delta, abs=1e-7)


def call(inputs, **options):
    return public.chunk_pgdn(*(inputs[n] for n in NAMES), launcher='board', **options)


@pytest.fixture
def no_launch(monkeypatch):
    def fail():
        raise AssertionError('unsupported input reached the kernel pipeline')
    monkeypatch.setattr(public, '_pipeline', fail)


@pytest.mark.parametrize('option,value,match', [
    ('initial_state', torch.zeros(1,4,128,128), 'initial_state'),
    ('initial_state', torch.zeros(1,2,128,128), 'state layout'),
    ('initial_A_state', torch.zeros(1,2,128), 'initial_A_state'),
    ('initial_A_state', torch.zeros(1,4,128), 'state layout'),
    ('use_qk_l2norm_in_kernel', False, 'must be True'),
    ('head_first', True, 'head_first'), ('transpose_state_layout', True, 'transpose_state_layout'),
    ('cu_seqlens', torch.tensor([0,64]), 'varlen'),
    ('cu_seqlens_cpu', torch.tensor([0,64]), 'varlen'), ('cp_context', object(), 'parallelism'),
    ('scale', 1., 'scale'), ('x', 2., 'x'), ('eps', 1e-5, 'eps'),
    ('log_atk_scale', 0., 'log_atk_scale'), ('log_atk_scale', torch.tensor(-.2), 'log_atk_scale'),
    ('x', float('nan'), 'x'), ('eps', float('inf'), 'eps'),
    ('block_dim', 0, 'block_dim'), ('block_dim', 3, 'block_dim'), ('block_dim', True, 'block_dim'),
    ('device', 'a2', 'a5'),
])
def test_options_reject_before_launch(option, value, match, no_launch):
    with pytest.raises(ValueError, match=match):
        call(data(), **{option: value})


@pytest.mark.parametrize('name', ['q','k','v','g','beta','g_atk','beta_atk'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_rejected(name, value, no_launch):
    inputs = data()
    inputs[name].flatten()[0] = value
    with pytest.raises(ValueError, match=f'{name} must be finite'):
        call(inputs)


@pytest.mark.parametrize('name', ['g','g_atk','beta','beta_atk'])
def test_gate_dtype_rejected(name, no_launch):
    inputs = data()
    inputs[name] = inputs[name].bfloat16()
    with pytest.raises(ValueError, match=f'{name} requires float32'):
        call(inputs)


@pytest.mark.parametrize('name,value', [('g',1.),('g_atk',1.),('beta',-.1),('beta',1.1),('beta_atk',-.1),('beta_atk',1.1)])
def test_gate_interval_rejected(name, value, no_launch):
    inputs = data()
    inputs[name].fill_(value)
    with pytest.raises(ValueError, match=name):
        call(inputs)


@pytest.mark.parametrize('name', ['g','g_atk'])
def test_finite_gates_with_overflowing_chunk_prefix_rejected(name, no_launch):
    inputs = data()
    inputs[name].fill_(-torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match='chunk prefix'):
        call(inputs)


@pytest.mark.parametrize('mutation,match', [
    ('tail','multiple'), ('empty','positive'), ('too_long','4096'), ('kdim','K=128'), ('vdim','v requires'),
    ('groups','positive multiple'), ('strided','contiguous'), ('mismatched_qkv','matching'),
    ('half','matching'), ('k_shape','k requires'), ('g_shape','g requires'),
    ('atk_hv','g_atk requires'), ('atk_state_axis','beta_atk requires'),
])
def test_shapes_and_layout_rejected(mutation, match, no_launch):
    inputs = data()
    if mutation in ('tail','empty','too_long'):
        length = {'tail':63, 'empty':0, 'too_long':4160}[mutation]
        for name in NAMES:
            shape = list(inputs[name].shape); shape[1] = length
            inputs[name] = torch.zeros(shape)
    elif mutation == 'kdim': inputs['q'] = inputs['q'][..., :64].contiguous()
    elif mutation == 'vdim': inputs['v'] = inputs['v'][..., :64].contiguous()
    elif mutation == 'groups': inputs['v'] = inputs['v'][:, :, :3].contiguous()
    elif mutation == 'strided': inputs['q'] = torch.zeros(1,64,2,256)[..., ::2]
    elif mutation == 'mismatched_qkv': inputs['k'] = inputs['k'].bfloat16()
    elif mutation == 'half':
        for name in ('q','k','v'): inputs[name] = inputs[name].half()
    elif mutation == 'k_shape': inputs['k'] = inputs['k'][:, :, :1].contiguous()
    elif mutation == 'g_shape': inputs['g'] = inputs['g'][:, :, :2].contiguous()
    elif mutation == 'atk_hv': inputs['g_atk'] = torch.zeros(1,64,4)
    elif mutation == 'atk_state_axis': inputs['beta_atk'] = torch.zeros(1,64,4)
    with pytest.raises(ValueError, match=match):
        call(inputs)


def test_backward_rejected(no_launch):
    inputs = data(); inputs['q'].requires_grad_(True)
    with pytest.raises(RuntimeError, match='inference-only'):
        call(inputs)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('final', [False, True])
def test_public_dtype_state_visibility_and_graph_with_reference_launch(monkeypatch, dtype, final):
    """CPU callback exercises marshaling only, never claims kernel execution."""
    import ascriptor.runtime
    inputs = data(128)
    for name in ('q','k','v'): inputs[name] = inputs[name].to(dtype)
    widened = {n: x.float() for n, x in inputs.items()}
    expected = ref.reference_stages(widened)
    seen = []
    pipeline = public._bf16_pipeline() if dtype == torch.bfloat16 else public._pipeline()
    graph = pipeline.GRAPH
    entry_names = [entry.name for entry in pipeline.entries()]
    class ReferenceLaunch:
        def __init__(self, entry, **kwargs):
            self.index = entry_names.index(entry.name)
        def __call__(self, *args):
            _, names, outputs = graph[self.index]
            for name, value in zip(names, args):
                compare(value, inputs[name] if name in inputs else expected[name])
            seen.append(self.index)
            buffers = args[len(names):len(names)+len(outputs)]
            results = tuple(expected[name].to(buffer.dtype).clone() for name, buffer in zip(outputs, buffers))
            return results[0] if len(results) == 1 else results
    monkeypatch.setattr(ascriptor.runtime, 'OpExec', ReferenceLaunch)
    o, state, a = call(inputs, output_final_state=final)
    assert seen == list(range(6))
    torch.testing.assert_close(o, expected['o'].to(dtype), atol=0, rtol=0)
    if final:
        assert state.shape == (1,4,128,128) and a.shape == (1,2,128)
        assert state.dtype == a.dtype == torch.float32
        compare(state, expected['final_state']); compare(a, expected['final_A_state'])
    else:
        assert state is None and a is None
