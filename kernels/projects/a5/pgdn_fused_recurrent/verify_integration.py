"""Actual chunk->decode with BOTH device states passed directly, plus splits.

Width1 exposes both states after every token; longer calls expose both states
at their call boundaries. Whole chunk exposes terminal states only, so intermediate
states are compared to independent CPU B and terminal states to both.
"""
import argparse,json
from pathlib import Path
import torch
from ascend_fla.ops.pgdn_chunk_fwd import chunk_pgdn
from ascend_fla.ops.pgdn_fused_recurrent import BLOCK_DIMS,fused_recurrent_pgdn,prepare
from .ref.integration import integration_cases,slice_inputs,trace_metrics,passed_trace,validate_returned
from .ref.reference import cases,digest,make_inputs,reference,metric


def to_device(data):
    return {n:None if x is None else x.to('npu') for n,x in data.items()}


def checked_call(call,data,cpu_data):
    before={n:None if x is None else digest(x) for n,x in data.items()}
    o,state,atk=call(data);torch.npu.synchronize()
    result=dict(o=o,final_state=state,final_A_state=atk)
    for x in result.values():
        assert x.device==data['q'].device
        assert all(y is None or x.untyped_storage().data_ptr()!=y.untyped_storage().data_ptr() for y in data.values())
    assert before=={n:None if x is None else digest(x) for n,x in data.items()},'input mutation'
    cpu={n:x.cpu() for n,x in result.items()}
    validate_returned(cpu_data,cpu,output_dtype=cpu_data['q'].dtype)
    return result,cpu


def chain(data,initial_device,initial_cpu,width,call,state_targets=None):
    state,atk=initial_device;sc,ac=initial_cpu
    outputs=[];calls=[]
    for start in range(0,data['q'].shape[1],width):
        stop=min(start+width,data['q'].shape[1])
        cpu=slice_inputs(data,start,stop,sc,ac)
        dev=to_device(dict(cpu,initial_state=None,initial_A_state=None))
        dev.update(initial_state=state,initial_A_state=atk)
        result,actual=checked_call(call,dev,cpu)
        row=dict(start=start,stop=stop,
                 initial_sha256={n:None if cpu[n] is None else digest(cpu[n]) for n in ('initial_state','initial_A_state')},
                 returned_sha256={n:digest(actual[n]) for n in ('final_state','final_A_state')})
        if state_targets is not None:
            row['vs_CPU_B_states']={n:metric(actual[n],state_targets[stop-1][n]) for n in ('final_state','final_A_state')}
            assert all(m['finite'] and m['relative_l2']<=1e-4 for m in row['vs_CPU_B_states'].values()),row
        calls.append(row);outputs.append(actual['o'])
        state,atk=result['final_state'],result['final_A_state'];sc,ac=actual['final_state'],actual['final_A_state']
    return dict(o=torch.cat(outputs,dim=1),final_state=sc,final_A_state=ac),calls


def main():
    p=argparse.ArgumentParser();p.add_argument('--block-dim',type=int,choices=BLOCK_DIMS,required=True)
    p.add_argument('--chunk-block-dim',type=int,choices=(1,2),default=2);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch_npu  # noqa: F401
    torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False
    prepare(block_dim=a.block_dim,chunk=True,chunk_block_dim=a.chunk_block_dim)
    a.output.mkdir(parents=True,exist_ok=True);assert not (a.output/'summary.json').exists()
    decode=lambda x:fused_recurrent_pgdn(**x,output_final_state=True,block_dim=a.block_dim)
    chunk=lambda x:chunk_pgdn(**x,output_final_state=True,block_dim=a.chunk_block_dim)
    records=[]
    for case in integration_cases():
        data=make_inputs(case);prefix=case['prefix']
        _,whole=checked_call(chunk,to_device(data),data)
        prefix_data=slice_inputs(data,0,prefix,None,None)
        pre_device,pre_cpu=checked_call(chunk,to_device(prefix_data),prefix_data)
        suffix=slice_inputs(data,prefix,case['S'],pre_cpu['final_state'],pre_cpu['final_A_state'])
        expected=dict(whole,o=whole['o'][:,prefix:].contiguous())
        b=reference(prefix_data);state_targets=[];bo=[]
        for t in range(prefix,case['S']):
            b=reference(slice_inputs(data,t,t+1,b['final_state'],b['final_A_state']))
            state_targets.append({n:b[n] for n in ('final_state','final_A_state')});bo.append(b['o'])
        expected_b=dict(b,o=torch.cat(bo,dim=1));rows=[]
        for width in (1,3,7,16):
            actual,calls=chain(suffix,(pre_device['final_state'],pre_device['final_A_state']),
                               (pre_cpu['final_state'],pre_cpu['final_A_state']),width,decode,state_targets)
            cm=trace_metrics(actual,expected,prefix);bm=trace_metrics(actual,expected_b,prefix)
            rows.append(dict(width=width,calls=calls,vs_whole_chunk=cm,vs_CPU_B=bm,
                             passed=passed_trace(cm,case['dtype']) and passed_trace(bm,case['dtype'])))
        record=dict(case=case,continuations=rows,passed=all(r['passed'] for r in rows))
        (a.output/(case['id']+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(case=case['id'],passed=record['passed'])),flush=True)
        assert record['passed'],case['id'];records.append(record)
    splits=[]
    for case in cases():
        if case['S']!=16:continue
        data=make_inputs(case);dev=to_device(data);_,whole=checked_call(decode,dev,data)
        for width in (1,3,7):
            actual,calls=chain(data,(dev['initial_state'],dev['initial_A_state']),
                              (data['initial_state'],data['initial_A_state']),width,decode)
            trace=trace_metrics(actual,whole);row=dict(case=case,width=width,calls=calls,trace=trace,passed=all(trace['byte_identical'].values()))
            splits.append(row);assert row['passed'],(case['id'],width,trace)
    report=dict(stage='native_dual_state_continuation',block_dim=a.block_dim,chunk_block_dim=a.chunk_block_dim,
                integration_cases=records,split_cases=splits,passed=True)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,prefill_cases=len(records),split_comparisons=len(splits))),flush=True)


if __name__=='__main__':main()
