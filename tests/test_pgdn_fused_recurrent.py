"""Frozen PGDN ABI, dual-state semantics and independent-reference controls."""
import pytest
import torch
from ascend_fla.ops import pgdn_fused_recurrent as api
from kernels.projects.a5.pgdn_fused_recurrent.ref import reference as ref
from kernels.projects.a5.pgdn_fused_recurrent.ref.integration import validate_returned
from kernels.projects.a5.pgdn_fused_recurrent.ref.native_baseline import torch_npu_recurrent


def inputs(dtype='float32',state='random',atk='random',steps=3):
    return ref.make_inputs(dict(B=1,S=steps,H=2,HV=4,dtype=dtype,state=state,atk=atk,seed=9501))


def public(data,**options):
    return api.fused_recurrent_pgdn(**data,launcher='aclnn',**options)


def metadata(data,**options):
    defaults=dict(output_final_state=True,scale=api.SCALE,x=1.5,eps=1e-6,log_atk_scale=None,
                  use_qk_l2norm_in_kernel=True,head_first=False,device='a5',block_dim=1,
                  launcher='aclnn',unsupported={})
    defaults.update(options);api._validate(**data,**defaults)


@pytest.mark.parametrize('dtype',['float32','bfloat16'])
@pytest.mark.parametrize('state',['none','zero','random'])
@pytest.mark.parametrize('atk',['none','zero','random'])
def test_independent_initials_validate_without_mutation(dtype,state,atk):
    data=inputs(dtype,state,atk);before={n:ref.digest(x) for n,x in data.items() if x is not None}
    metadata(data)
    assert before=={n:ref.digest(x) for n,x in data.items() if x is not None}


@pytest.mark.parametrize('options',[
    {'block_dim':0},{'block_dim':3},{'block_dim':29},{'block_dim':True},
    {'device':'a2'},{'device':'a3'},{'head_first':True},{'head_first':0},
    {'scale':1.},{'scale':api.SCALE+1e-13},{'scale':None},{'scale':float('nan')},{'scale':True},{'output_final_state':1},
    {'x':1.4},{'x':True},{'eps':1e-12},{'eps':float('nan')},{'log_atk_scale':0.},
    {'log_atk_scale':torch.tensor(-.2)},{'use_qk_l2norm_in_kernel':False},
    {'use_qk_l2norm_in_kernel':1},{'state_v_first':True},{'cu_seqlens':None},
    {'use_beta_sigmoid_in_kernel':False},{'unknown':1},
])
def test_reject_options_before_kernel_loading(monkeypatch,options):
    monkeypatch.setattr(api,'_pipeline',lambda:pytest.fail('invalid call loaded kernels'))
    with pytest.raises(ValueError):public(inputs(),**options)


@pytest.mark.parametrize('field',ref.NAMES)
@pytest.mark.parametrize('kind',['dtype','shape','noncontiguous','nan','infinity','requires_grad','not_tensor'])
def test_reject_bad_tensor_before_kernel_loading(monkeypatch,field,kind):
    data=inputs();x=data[field]
    if kind=='dtype':data[field]=x.to(torch.float16)
    elif kind=='shape':data[field]=x[..., :-1].contiguous()
    elif kind=='noncontiguous':data[field]=torch.stack([x,x],dim=-1)[...,0]
    elif kind in ('nan','infinity'):
        data[field]=x.clone();data[field].reshape(-1)[0]=float('nan' if kind=='nan' else 'inf')
    elif kind=='not_tensor':data[field]=False
    else:data[field]=x.clone().requires_grad_(True)
    monkeypatch.setattr(api,'_pipeline',lambda:pytest.fail('invalid tensor loaded kernels'))
    with pytest.raises((ValueError,RuntimeError)):public(data)


@pytest.mark.parametrize('steps',[0,17])
def test_sequence_bounds(steps):
    data=inputs()
    for n in ref.NAMES[:7]:
        x=data[n];data[n]=torch.empty((x.shape[0],steps,*x.shape[2:]),dtype=x.dtype)
    with pytest.raises(ValueError,match='S=1..16'):public(data)


def test_shapes_index_overflow_and_device():
    data=inputs();data['q']=data['q'][:,:,:0].contiguous()
    with pytest.raises(ValueError):public(data)
    data=inputs();data['v']=data['v'][:,:,:3].contiguous()
    with pytest.raises(ValueError,match='multiple of H'):public(data)
    data=inputs();data={n:None if x is None else torch.empty_like(x,device='meta') for n,x in data.items()}
    with pytest.raises(ValueError,match='one cpu device'):public(data)
    data=inputs();data['q']=torch.empty(2048,1,1,128,device='meta');data['v']=torch.empty(2048,1,64,128,device='meta')
    with pytest.raises(ValueError,match='signed32'):public(data)
    with pytest.raises(ValueError,match='launcher'):api.fused_recurrent_pgdn(**inputs(),launcher='sim')
    with pytest.raises(ValueError,match='one npu device'):api.fused_recurrent_pgdn(**inputs())


@pytest.mark.parametrize('name,value',[('g',.01),('g_atk',.01),('beta',-.01),('beta',1.01),('beta_atk',-.01),('beta_atk',1.01),('initial_A_state',-.01)])
def test_cpu_value_domain(name,value):
    data=inputs();data[name].fill_(value)
    with pytest.raises(ValueError,match=name):public(data)


@pytest.mark.parametrize('dtype',['float32','bfloat16'])
@pytest.mark.parametrize('width',[1,2,7])
def test_dual_state_chaining_matches_one_call(dtype,width):
    data=inputs(dtype,steps=15);full=ref.reference(data);state=data['initial_state'];atk=data['initial_A_state'];outs=[]
    for start in range(0,15,width):
        sliced={n:data[n][:,start:start+width].contiguous() for n in ref.NAMES[:7]}
        sliced.update(initial_state=state,initial_A_state=atk)
        result=ref.reference(sliced);state=result['final_state'];atk=result['final_A_state'];outs.append(result['o'])
    for name,value in dict(o=torch.cat(outs,dim=1),final_state=state,final_A_state=atk).items():
        assert ref.digest(value)==ref.digest(full[name])


@pytest.mark.parametrize('dropped',['initial_state','initial_A_state'])
def test_each_missing_initial_state_is_detected_at_first_token(dropped):
    data=inputs(steps=1);good=ref.reference(data);bad=ref.reference(dict(data,**{dropped:None}))
    target='final_A_state' if dropped=='initial_A_state' else 'final_state'
    assert ref.metric(bad[target],good[target])['relative_l2']>1e-4
    assert ref.metric(bad['o'],good['o'])['relative_l2']>1e-4


def test_zero_initials_are_independent_and_equivalent_to_none():
    for state,atk in [('none','random'),('random','none'),('none','none')]:
        data=inputs(state=state,atk=atk);other=dict(data)
        if state=='none':other['initial_state']=torch.zeros(1,4,128,128)
        if atk=='none':other['initial_A_state']=torch.zeros(1,2,128)
        a,b=ref.reference(data),ref.reference(other)
        assert all(ref.digest(a[n])==ref.digest(b[n]) for n in a)


@pytest.mark.parametrize('name',ref.OUTPUTS)
def test_returned_output_controls(name):
    data=inputs();good=ref.reference(data);validate_returned(data,good,output_dtype=torch.float32)
    for bad in [dict(good,**{name:good[name][...,0:1]}),dict(good,**{name:torch.full_like(good[name],float('nan'))})]:
        with pytest.raises(ValueError):validate_returned(data,bad,output_dtype=torch.float32)
    assert ref.metric(torch.ones(1),torch.zeros(1))['relative_l2']==float('inf')
    assert ref.metric(torch.zeros(1),torch.zeros(1))['relative_l2']==0


def test_native_baseline_rejects_cpu_fallback():
    with pytest.raises(ValueError,match='NPU'):torch_npu_recurrent(inputs())


def test_prepare_registers_both_families_without_execution(monkeypatch):
    from ascend_fla.ops import pgdn_chunk_fwd
    calls=[]
    monkeypatch.setattr(api,'_compiled',lambda bd:calls.append(('decode',bd)))
    monkeypatch.setattr(pgdn_chunk_fwd,'prepare',lambda **kw:calls.append(('chunk',kw)))
    api.prepare(block_dim=4,chunk=True,chunk_block_dim=2)
    assert calls==[('decode',4),('chunk',dict(device='a5',block_dim=2))]
    for options in ({'chunk':1},{'chunk_block_dim':3}):
        with pytest.raises(ValueError):api.prepare(**options)


def test_wrong_read_key_and_additive_normalization_are_observable():
    data=inputs(steps=1);correct=ref.reference(data)
    q,k,v=(data[n].float()[:,0] for n in ('q','k','v'))
    k=k/k.norm(dim=-1,keepdim=True);q=q/q.norm(dim=-1,keepdim=True)*api.SCALE
    atk=data['initial_A_state']*data['g_atk'][:,0].exp()[...,None]+data['beta_atk'][:,0,:,None]*k.square()
    r=(atk+1e-6).log()+.2
    write=k*(-torch.tensor(1.5).log()*(r/(1+r.abs()))).exp()
    read=k.repeat_interleave(2,dim=1);write=write.repeat_interleave(2,dim=1)
    decayed=data['initial_state']*data['g'][:,0].exp()[...,None,None]
    wrong_delta=(v-(decayed*write[...,None]).sum(-2))*data['beta'][:,0,:,None]
    wrong_state=decayed+write[...,None]*wrong_delta[...,None,:]
    assert ref.metric(wrong_state,correct['final_state'])['relative_l2']>1e-4
    tiny=inputs(steps=1)
    tiny['q'].mul_(1e-14);tiny['k'].mul_(1e-14)
    naive=ref.reference(tiny)
    # A deliberate additive-epsilon fork; this control changes both norms.
    bad=dict(tiny)
    for name in ('q','k'):
        raw=tiny[name];normalized=raw/(raw.square().sum(-1,keepdim=True)+1e-6).sqrt()
        # Feeding a vector below the clamp encodes the alternate normalized key
        # after the independent reference's compulsory normalization.
        bad[name]=normalized*1e-12
    alternate=ref.reference(bad)
    assert ref.metric(alternate['o'],naive['o'])['relative_l2']>1e-4
