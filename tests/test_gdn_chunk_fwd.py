"""Independent GDN math, public rejection gates and launch graph contracts."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from ascend_fla.ops.gdn_chunk_fwd import chunk_gdn

ROOT = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/gdn_chunk_fwd'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load('gdn_test_reference', ROOT / 'ref/reference.py')
stages = load('gdn_test_stages', ROOT / 'ref/stages.py')


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def data(time=64, heads=3, gate=1., normalized=False):
    return ref.make_inputs({'seed':73+time,'parameters':{'B':1,'T':time,'H':heads,'gate_scale':gate,'normalized_keys':normalized}})


def compare(got, expected, budget=1e-5):
    assert torch.isfinite(got).all()
    delta=got.float()-expected.float()
    norm=expected.float().norm()
    assert float(delta.norm()/norm.clamp_min(1e-30)) <= budget
    torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('time,heads,gate',[(64,1,0.),(64,3,.03),(128,3,1.),(192,3,30.),(320,3,.03),(1024,16,.03),(4096,16,1.)])
def test_stage_equations_match_independent_block_solve(time, heads, gate):
    inp=data(time,heads,gate)
    before={n:x.clone() for n,x in inp.items()}
    expected=ref.reference(inp)
    got=stages.reference_stages(inp)
    for name in expected:
        compare(got[name],expected[name])
    for name in inp:
        assert torch.equal(inp[name],before[name])


@pytest.mark.parametrize('gate',[0.,.03,30.])
def test_fla_oracle(gate):
    path=os.environ.get('FLA_GDN_NAIVE')
    if not path:
        pytest.skip('set FLA_GDN_NAIVE to the pinned gated_delta_rule/naive.py')
    fla=load('fla_gdn_test_naive',Path(path))
    inp=data(192,3,gate,normalized=True)
    got=ref.reference(inp)
    o,s=fla.naive_recurrent_gated_delta_rule(*(inp[n] for n in ('q','k','v','beta','g')),output_final_state=True)
    compare(got['o'],o)
    compare(got['final_state'],s)


@pytest.mark.parametrize('beta',[0.,1.])
def test_gate_endpoints(beta):
    inp=data(128)
    inp['beta'].fill_(beta)
    got=ref.reference(inp)
    expected=stages.reference_stages(inp)
    for n in got:
        compare(got[n],expected[n])
        if beta==0:
            assert not got[n].any()


def call(inp, **kwargs):
    return chunk_gdn(*(inp[n] for n in ('q','k','v','g','beta')),launcher='board',**kwargs)


@pytest.mark.parametrize('option,value,match',[
    ('initial_state',torch.ones(1,3,128,128),'initial_state'),
    ('head_first',True,'head_first'),('scale',1.,'scale'),
    ('block_dim',3,'block_dim'),('block_dim',29,'block_dim'),
    ('block_dim',True,'block_dim'),('device','a2','a5')])
def test_options_reject_before_launch(option,value,match):
    with pytest.raises(ValueError,match=match):
        call(data(),**{option:value})


@pytest.mark.parametrize('mutation,match',[
    ('gqa','positive multiple'),('tail','multiple of 64'),('strided','contiguous'),
    ('dtype','matching'),('positive_gate','g<=0'),('beta','beta'),('nan','finite')])
def test_invalid_inputs(mutation,match):
    inp=data()
    if mutation=='gqa': inp['v']=torch.zeros(1,64,5,128)
    if mutation=='tail':
        for n in ('q','k','v','g','beta'): inp[n]=inp[n][:,:63].contiguous()
    if mutation=='strided': inp['q']=torch.zeros(1,64,3,256)[...,::2]
    if mutation=='dtype': inp['k']=inp['k'].half()
    if mutation=='positive_gate': inp['g'].fill_(1.)
    if mutation=='beta': inp['beta'].fill_(2.)
    if mutation=='nan': inp['v'].fill_(float('nan'))
    with pytest.raises(ValueError,match=match): call(inp)


def test_no_backward():
    inp=data(); inp['q'].requires_grad_(True)
    with pytest.raises(RuntimeError,match='inference-only'): call(inp)


def test_bulk_prepare_strides_and_repeated_buffer_ownership(tmp_path):
    """Regression for skipped-head GM row gaps; reduced pipesim, not hardware."""
    pytest.importorskip('ascriptor.runtime')
    from ascend_fla.ops.gdn_chunk_fwd import _pipeline
    runner=load('gdn_test_unit_runner',ROOT/'_unit_runner.py')
    inp=data(64,3,.03)
    expected=stages.reference_stages(inp)
    names=('qn','kn','gc','bk','wv')
    outputs=[torch.full_like(expected[n],float('nan')) for n in names]
    options=dict(device='a5',backend='cce',block_dim=1,launcher='pipesim',out_dir=tmp_path,timeout=60)
    args=tuple(inp[n] for n in ('q','k','v','g','beta'))+tuple(outputs)+(1,64,3,1)
    got=runner.launch_kernel(_pipeline().entries()[0],args,options)
    for n,x in zip(names,got): compare(x,expected[n])
    evidence=options['_execution_evidence'][0]
    assert not evidence['hazards'] and not evidence['deadlock'] and not evidence['event_balance']


@pytest.mark.parametrize('ratio',[1,2,3,4,8])
@pytest.mark.parametrize('chunks',[1,2,3])
def test_grouped_mapping_and_independent_oracles(ratio,chunks):
    from ascend_fla.ops.gdn_chunk_fwd import _pipeline, _validate, SCALE
    inp=ref.make_inputs({'seed':8200+ratio*10+chunks,'parameters':dict(B=2,T=chunks*64,H=3,HV=3*ratio,gate_scale=1.)})
    _validate(*(inp[n] for n in ('q','k','v','g','beta')),None,SCALE,False,'a5',1,'board')
    before={n:x.clone() for n,x in inp.items()}
    expanded=_pipeline().expand_inputs(inp)
    if ratio==1:
        assert expanded is inp
    for n in ('q','k'):
        for j in range(3*ratio):
            assert torch.equal(expanded[n][:,:,j],inp[n][:,:,j//ratio])
        if ratio>1: assert expanded[n].data_ptr()!=inp[n].data_ptr()
    expected=ref.grouped_recurrent(inp)
    for got in (ref.reference(inp),stages.reference_stages(inp)):
        for n in expected:compare(got[n],expected[n])
    path=os.environ.get('FLA_GDN_NAIVE')
    if path:
        fla=load('fla_gdn_grouped_test',Path(path))
        o,s=fla.naive_recurrent_gated_delta_rule(*(expanded[n] for n in ('q','k','v','beta','g')),output_final_state=True)
        compare(o,expected['o']);compare(s,expected['final_state'])
    for n in inp: assert torch.equal(inp[n],before[n])
    # A cyclic mapping has the right shapes but must fail semantic checks.
    if ratio>1:
        wrong=dict(inp,q=inp['q'].roll(1,2),k=inp['k'].roll(1,2))
        assert not torch.allclose(ref.reference(wrong)['o'],expected['o'],atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('hv',[0,1,2,4,5])
def test_invalid_group_ratio_rejected_before_pipeline(monkeypatch,hv):
    import ascend_fla.ops.gdn_chunk_fwd as api
    inp=data();inp['v']=torch.zeros(1,64,hv,128)
    def forbidden():raise AssertionError('pipeline accessed before validation')
    monkeypatch.setattr(api,'_pipeline',forbidden)
    with pytest.raises(ValueError,match=f'H=3, HV={hv}'):
        call(inp)


@pytest.mark.parametrize('name',['k','g','beta','initial_state'])
def test_grouped_reference_rejects_wrong_head_axis(name):
    inp=ref.make_inputs({'seed':82,'parameters':dict(B=1,T=64,H=3,HV=6)})
    inp[name]=torch.zeros(1,3,128,128) if name=='initial_state' else (torch.zeros(1,64,6,128) if name=='k' else torch.zeros(1,64,3))
    with pytest.raises(ValueError,match=name):ref.validate_inputs(inp)


@pytest.mark.parametrize('name',['k','g','beta'])
def test_grouped_public_shape_gates(name):
    inp=ref.make_inputs({'seed':82,'parameters':dict(B=1,T=64,H=3,HV=6)})
    inp[name]=torch.zeros(1,64,6,128) if name=='k' else torch.zeros(1,64,3)
    with pytest.raises(ValueError,match=name+' requires shape'):call(inp)


def test_bf16_group_replication_is_exact():
    from ascend_fla.ops.gdn_chunk_fwd import _pipeline
    inp=ref.make_inputs({'seed':82,'parameters':dict(B=1,T=64,H=3,HV=24)})
    inp={n:x.bfloat16().float() if n in ('q','k','v') else x for n,x in inp.items()}
    expanded=_pipeline().expand_inputs(inp)
    for n in ('q','k'):
        assert expanded[n].is_contiguous()
        assert torch.equal(expanded[n].view(1,64,3,8,128),inp[n].unsqueeze(3).expand(1,64,3,8,128))
