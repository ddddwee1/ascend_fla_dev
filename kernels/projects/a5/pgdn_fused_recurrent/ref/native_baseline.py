"""Real torch_npu PGDN composition, used only as an actual measured baseline.

Includes NPU widening, naive normalization, grouping, allocations and arithmetic.
Transfers and input generation are outside timing. No CPU fallback exists.
"""
import torch


@torch.no_grad()
def torch_npu_recurrent(inputs):
    if any(x is not None and (x.device.type!='npu' or x.device!=inputs['q'].device) for x in inputs.values()):
        raise ValueError('baseline requires one NPU device')
    q,k,v = (inputs[n].float() for n in ('q','k','v'))
    b,steps,h,kd = q.shape
    hv,vd = v.shape[2:]
    query = torch.nn.functional.normalize(q,p=2,dim=-1)*kd**-.5
    key = torch.nn.functional.normalize(k,p=2,dim=-1)
    query = query.repeat_interleave(hv//h,dim=2)
    state = (torch.zeros(b,hv,kd,vd,dtype=torch.float32,device=q.device) if inputs['initial_state'] is None else inputs['initial_state'].clone())
    atk = (torch.zeros(b,h,kd,dtype=torch.float32,device=q.device) if inputs['initial_A_state'] is None else inputs['initial_A_state'].clone())
    logx = torch.tensor(1.5,dtype=torch.float32,device=q.device).log()
    outputs=[]
    for t in range(steps):
        atk = atk*inputs['g_atk'][:,t].exp()[...,None]+key[:,t].square()*inputs['beta_atk'][:,t,:,None]
        r = (atk+1e-6).log()+.2
        write = (key[:,t]*(-logx*(r/(1+r.abs()))).exp()).repeat_interleave(hv//h,dim=1)
        read = key[:,t].repeat_interleave(hv//h,dim=1)
        decayed = state*inputs['g'][:,t].exp()[...,None,None]
        delta = (v[:,t]-(decayed*read[...,None]).sum(-2))*inputs['beta'][:,t,:,None]
        state = decayed+write[...,None]*delta[...,None,:]
        outputs.append((state*query[:,t,:,:,None]).sum(-2))
    return dict(o=torch.stack(outputs,dim=1).to(inputs['v'].dtype),final_state=state,final_A_state=atk)
