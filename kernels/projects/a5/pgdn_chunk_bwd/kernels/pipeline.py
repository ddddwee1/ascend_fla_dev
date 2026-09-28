"""Four ordered device launches; host work is allocation and pointer binding."""
GRAPH = (
    (('q','k','g_atk','beta_atk'),
     ('q_norm','k_read','k_write','A_history','q_raw_norm','k_raw_norm','final_A_state'), ('B','T','H','N')),
    (('k_read','k_write','v','g','beta'), ('checkpoints','final_state'), ('B','T','H','HV','N')),
    (('q_norm','k_read','k_write','v','g','beta','dout','dht','checkpoints'),
     ('tape','dq_norm_parts','dk_read_parts','dk_write_parts','dv','dg','dbeta'), ('B','T','H','HV','N','has_do','has_dht')),
    (('q_norm','k_read','A_history','q_raw_norm','k_raw_norm','dq_norm_parts','dk_read_parts','dk_write_parts','g_atk','beta_atk','dat'),
     ('dq','dk','dg_atk','dbeta_atk'), ('B','T','H','HV','has_dat')),
)
PUBLIC = ('dq','dk','dv','dg_atk','dg','dbeta_atk','dbeta')


def entries():
    from .atk import pgdn_bwd_atk
    from .main import pgdn_bwd_checkpoints, pgdn_bwd_reverse
    from .atk_reverse import pgdn_bwd_atk_reverse
    return pgdn_bwd_atk, pgdn_bwd_checkpoints, pgdn_bwd_reverse, pgdn_bwd_atk_reverse


def run(inputs, launch, *, retain_stages=False):
    import torch
    B,T,H,_ = inputs['q'].shape
    HV = inputs['v'].shape[2]
    N = T//64
    shapes = {n:(B,T,H,128) for n in ('q_norm','k_read','k_write','A_history','dq','dk')}
    shapes.update({n:(B,T,H) for n in ('q_raw_norm','k_raw_norm','dg_atk','dbeta_atk')})
    shapes.update({n:(B,T,HV,128) for n in ('dq_norm_parts','dk_read_parts','dk_write_parts','dv')})
    shapes.update(dg=(B,T,HV),dbeta=(B,T,HV),final_A_state=(B,H,128),
                  checkpoints=(B,N,HV,128,128),final_state=(B,HV,128,128),tape=(B,HV,64,128,128))
    values = dict(inputs)
    all_scalars = dict(B=B,T=T,H=H,HV=HV,N=N,
                       has_do=int(inputs.get('do') is not None),
                       has_dht=int(inputs.get('dht') is not None),has_dat=int(inputs.get('dA_T') is not None))

    def empty(shape):
        return torch.empty(shape,dtype=torch.float32,device=inputs['q'].device)

    # Flags prevent reads of absent cotangents; the device initializes its UB.
    values['dout'] = empty((B,T,HV,128)) if inputs.get('do') is None else inputs['do']
    values['dht'] = empty((B,HV,128,128)) if inputs.get('dht') is None else inputs['dht']
    values['dat'] = empty((B,H,128)) if inputs.get('dA_T') is None else inputs['dA_T']
    retained = {}
    for index,(entry,(names,outputs,scalar_names)) in enumerate(zip(entries(),GRAPH)):
        fresh = {n:empty(shapes[n]) for n in outputs}
        if inputs['q'].device.type == 'cpu':
            for tensor in fresh.values():
                tensor.fill_(float('nan'))
        result = launch(entry,{n:values[n] for n in names},fresh,{n:all_scalars[n] for n in scalar_names})
        if set(result) != set(outputs):
            raise RuntimeError(f'{entry.name}: incomplete stage outputs')
        values.update(result)
        if retain_stages:
            retained.update(result)
        else:
            live = set(PUBLIC)
            for future_inputs,_,_ in GRAPH[index+1:]:
                live.update(future_inputs)
            for name in tuple(values):
                if name not in live:
                    del values[name]
    return retained if retain_stages else {n:values[n] for n in PUBLIC}
