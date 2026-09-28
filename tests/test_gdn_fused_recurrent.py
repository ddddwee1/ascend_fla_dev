"""ABI and independent-reference checks for short-step GDN."""
import importlib.util
from pathlib import Path
import pytest
import torch
from ascend_fla.ops import gdn_fused_recurrent as api
from kernels.projects.a5.gdn_fused_recurrent.ref import integration
from kernels.projects.a5.gdn_fused_recurrent.ref.native_baseline import torch_npu_recurrent

ROOT=Path(__file__).resolve().parents[1]/'kernels/projects/a5/gdn_fused_recurrent'
_spec=importlib.util.spec_from_file_location('_gda04_reference_tests',ROOT/'ref/reference.py')
ref=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(ref)


def inputs(dtype='float32',state='random'):
    return ref.make_inputs(dict(B=1,S=3,H=2,HV=4,dtype=dtype,state=state,seed=9401))


def public(data,**options):
    return api.fused_recurrent_gdn(**data,launcher='aclnn',**options)


def metadata(data,**options):
    defaults=dict(output_final_state=True,scale=api.SCALE,head_first=False,
                  device='a5',block_dim=1,launcher='aclnn',unsupported={})
    defaults.update(options);api._validate(**data,**defaults)


@pytest.mark.parametrize('dtype',['float32','bfloat16'])
@pytest.mark.parametrize('state',['none','zero','random'])
def test_valid_metadata_does_not_modify_inputs(dtype,state):
    data=inputs(dtype,state);before={n:ref.digest(x) for n,x in data.items() if x is not None}
    metadata(data)
    assert before=={n:ref.digest(x) for n,x in data.items() if x is not None}


@pytest.mark.parametrize('options',[
    {'block_dim':0},{'block_dim':3},{'block_dim':29},{'block_dim':True},
    {'device':'a2'},{'device':'a3'},{'head_first':True},{'scale':1.},
    {'scale':float('nan')},{'scale':True},{'output_final_state':1},
    {'use_qk_l2norm_in_kernel':True},{'state_v_first':True},{'cu_seqlens':None},
    {'use_beta_sigmoid_in_kernel':False},{'unknown':1},
])
def test_reject_options_before_loading_kernels(monkeypatch,options):
    monkeypatch.setattr(api,'_pipeline',lambda:pytest.fail('invalid call reached kernel loading'))
    with pytest.raises(ValueError):public(inputs(),**options)


@pytest.mark.parametrize('field',['q','k','v','g','beta','initial_state'])
@pytest.mark.parametrize('kind',['dtype','shape','noncontiguous','nan','infinity','requires_grad'])
def test_reject_bad_tensor_before_loading_kernels(monkeypatch,field,kind):
    data=inputs();x=data[field]
    if kind=='dtype':data[field]=x.to(torch.float16)
    elif kind=='shape':data[field]=x[..., :-1].contiguous()
    elif kind=='noncontiguous':data[field]=torch.stack([x,x],dim=-1)[...,0]
    elif kind in ('nan','infinity'):
        data[field]=x.clone();data[field].reshape(-1)[0]=float('nan' if kind=='nan' else 'inf')
    else:data[field]=x.clone().requires_grad_(True)
    monkeypatch.setattr(api,'_pipeline',lambda:pytest.fail('invalid tensor reached kernel loading'))
    with pytest.raises((ValueError,RuntimeError)):public(data)


@pytest.mark.parametrize('s',[0,17])
def test_reject_sequence_bounds(s):
    data=inputs()
    for n in ('q','k','v','g','beta'):
        x=data[n];data[n]=torch.empty((x.shape[0],s,*x.shape[2:]),dtype=x.dtype)
    with pytest.raises(ValueError,match='S=1..16'):public(data)


def test_reject_zero_heads_and_nondivisible_groups():
    data=inputs();data['q']=data['q'][:,:,:0].contiguous()
    with pytest.raises(ValueError):public(data)
    data=inputs();data['v']=data['v'][:,:,:3].contiguous()
    with pytest.raises(ValueError,match='multiple of H'):public(data)


@pytest.mark.parametrize('name,value',[('g',.01),('beta',-.01),('beta',1.01)])
def test_reject_cpu_value_domain(name,value):
    data=inputs();data[name].fill_(value)
    with pytest.raises(ValueError,match='g<=0'):public(data)


def test_reject_non_tensor_and_wrong_launcher():
    data=inputs();data['initial_state']=False
    with pytest.raises(ValueError,match='tensor or None'):public(data)
    data=inputs();data['q']=None
    with pytest.raises(ValueError,match='must be tensors'):public(data)
    data=inputs()
    with pytest.raises(ValueError,match='launcher'):
        api.fused_recurrent_gdn(**data,launcher='sim')
    with pytest.raises(ValueError,match='one npu'):
        api.fused_recurrent_gdn(**data)


def test_reference_beta_zero_has_exact_state_and_query_read():
    data=inputs();data['g'].zero_();data['beta'].zero_()
    data['q'].zero_();data['q'][...,7]=1.
    result=ref.reference(data)
    torch.testing.assert_close(result['final_state'],data['initial_state'],rtol=0,atol=0)
    expected=(data['initial_state'][:,:,7,:]*api.SCALE)[:,None].expand_as(result['o'])
    torch.testing.assert_close(result['o'],expected,rtol=0,atol=0)
    assert result['final_state'].data_ptr()!=data['initial_state'].data_ptr()


def test_reference_none_and_explicit_zero_states_agree():
    data=inputs(state='none');a=ref.reference(data)
    data['initial_state']=torch.zeros(1,4,128,128);b=ref.reference(data)
    for n in ref.OUTPUTS:torch.testing.assert_close(a[n],b[n],rtol=0,atol=0)


def test_reference_grouping_with_distinct_one_hot_heads():
    data=inputs();data['q'].zero_();data['q'][:,:,0,0]=1.;data['q'][:,:,1,1]=1.
    data['k'].zero_();data['g'].zero_();data['initial_state'].zero_()
    data['initial_state'][:,:,0,:]=3.;data['initial_state'][:,:,1,:]=-7.
    result=ref.reference(data)
    expected=torch.tensor([3.,3.,-7.,-7.])*api.SCALE
    torch.testing.assert_close(result['o'][0,:,:,0],expected.expand(3,4),rtol=0,atol=0)


def test_comparator_zero_reference_and_nonfinite_rules():
    zero=torch.zeros(2,3);wrong=zero.clone();wrong[0,0]=1e-30
    assert ref.metric(zero,zero)['relative_l2']==0.
    assert ref.metric(wrong,zero)['relative_l2']==float('inf')
    wrong[0,0]=float('nan');assert not ref.metric(wrong,zero)['finite']


@pytest.mark.parametrize('dtype',['float32','bfloat16'])
@pytest.mark.parametrize('width',[1,3,7])
def test_state_chaining_matches_one_call_and_covers_short_tail(dtype,width):
    data=ref.make_inputs(dict(B=2,S=16,H=2,HV=8,dtype=dtype,state='random',seed=9416))
    whole=ref.reference(data)
    split,calls=integration.chained_decode(data,ref.reference,width,output_dtype=torch.float32)
    trace=integration.trace_metrics(split,whole)
    assert all(trace['byte_identical'].values())
    assert [t for row in calls for t in range(row['start'],row['stop'])]==list(range(16))
    assert all(0<row['stop']-row['start']<=width for row in calls)
    for previous,current in zip(calls,calls[1:]):
        assert previous['returned_state_sha256']==current['initial_state_sha256']


@pytest.mark.parametrize('fault',['missing','shape','dtype','nan','alias','mutation'])
def test_integration_checks_reject_faulty_return_or_input_mutation(fault):
    data=inputs()
    def faulty_call(values):
        result=ref.reference(values)
        if fault=='missing':del result['final_state']
        elif fault=='shape':result['o']=result['o'][:,:1].contiguous()
        elif fault=='dtype':result['final_state']=result['final_state'].bfloat16()
        elif fault=='nan':result['o'].reshape(-1)[0]=float('nan')
        elif fault=='alias':result['final_state']=values['initial_state'].view_as(values['initial_state'])
        else:values['initial_state'].add_(1.)
        return result
    with pytest.raises(ValueError):
        integration.run_checked(data,faulty_call,output_dtype=torch.float32)


def test_dropped_state_is_detected_at_first_token():
    data=inputs();data['q'].zero_();data['q'][...,7]=1.
    data['k'].zero_();data['beta'].zero_();data['g'].zero_()
    data['initial_state'].zero_();data['initial_state'][:,:,7,:]=2.
    expected=ref.reference(data)
    dropped=ref.reference(dict(data,initial_state=None))
    trace=integration.trace_metrics(dropped,expected,token_offset=128)
    assert trace['tokens'][0]['token']==128
    assert trace['tokens'][0]['relative_l2']==1.
    assert not integration.trace_passed(trace)


def test_trace_rejects_broadcastable_wrong_shape():
    expected=ref.reference(inputs())
    wrong=dict(expected,o=expected['o'][:,:1])
    with pytest.raises(ValueError,match='shapes differ'):
        integration.trace_metrics(wrong,expected)


def test_native_baseline_rejects_cpu_fallback():
    with pytest.raises(ValueError,match='no CPU fallback'):
        torch_npu_recurrent(inputs())
