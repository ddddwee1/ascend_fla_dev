"""Reproduce unchanged-budget fixture failures and localize first mismatched seams."""
import argparse,datetime,hashlib,json
from pathlib import Path
import torch
from benchmarks.probe_bwd_span import A2BackwardNative,a2_gradient_metrics
from benchmarks.verify_decode import _a2_identity,_a2_digest
p=argparse.ArgumentParser();p.add_argument('--bd',type=int,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
def emit(row):
    line=json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**row},allow_nan=False)
    with (a.out/'events.jsonl').open('a') as f:f.write(line+'\n')
    print(line,flush=True)
identity=_a2_identity();identity['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(a.out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
native=A2BackwardNative(a.bd,emit,gc_dtype='fp32');rules=native.contract['comparison'];rows=[]
original_launch=native.launch
capture_context={}
capture_names={'bwd.scan_fused','bwd.inverse_epilogue','bwd.finalize_pre','bwd.finalize_post','bwd.inverse_mm'}
def capture_launch(kernel,args,options):
    name=native.names[id(kernel)]
    take=name in capture_names and capture_context.get('repeat')==0
    before=tuple(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for v in args) if take else None
    result=original_launch(kernel,args,options)
    if take:
        output=result if isinstance(result,tuple) else (result,)
        torch.save({'kernel':name,'kernel_symbol':kernel.name,'block_dim':a.bd,'args':before,
                    'actual_outputs':tuple(v.detach().cpu().clone() for v in output)},
                   a.out/(capture_context['case']+'-'+name.replace('.','-')+'-leaf.pt'))
    return result
native.launch=capture_launch
cases=[c for c in native.contract['cases'] if c['id'] in ('gentle_decay_bd1','grid_c1_hv2_bd1')]
for case in cases:
    inputs=native.backward.make_inputs(case);expected=native.backward.reference_stages(inputs)
    golden=native.backward.reference(inputs)
    for repeat in range(2):
        capture_context.update(case=case['id'],repeat=repeat)
        start=len(native.trace);actual,stages=native.backward_call(inputs)
        actual={n:v.cpu() for n,v in actual.items()};stages={n:v.cpu() for n,v in stages.items()}
        metrics={n:a2_gradient_metrics(v,expected[n],{**rules['default'],**rules.get('stage_outputs',{}).get(n,{})}) for n,v in stages.items()}
        failures={}
        for n,m in metrics.items():
            if not m['passed']:
                got=stages[n].float();ref=expected[n].float();rule={**rules['default'],**rules.get('stage_outputs',{}).get(n,{})}
                diff=got!=ref;loc=diff.nonzero();examples=[]
                for idx in loc[:24]:
                    ix=tuple(idx.tolist());examples.append({'index':list(ix),'actual':got[ix].item(),'expected':ref[ix].item()})
                failures[n]={'metrics':m,'different_elements':int(diff.sum()),'outside_allclose':int((~torch.isclose(got,ref,rtol=rule.get('rtol',0),atol=rule.get('atol',0))).sum()),'examples':examples}
        # Independent FP32 CPU dot vs built-in NPU BF16/FP32 dot from actual native inputs.
        left=stages['finalize_pre.M_qk'];right=stages['finalize_pre.kg'];shape=stages['finalize_pair.qk_left'].shape
        cpu=(left.float()@right.float()).bfloat16().reshape(shape)
        npu_bf16=(left.to('npu')@right.to('npu')).cpu().reshape(shape)
        npu_fp32=(left.float().to('npu')@right.float().to('npu')).cpu().bfloat16().reshape(shape)
        pair={n:{'digest':_a2_digest(v),'bitwise_native':torch.equal(v,stages['finalize_pair.qk_left']),**a2_gradient_metrics(v,stages['finalize_pair.qk_left'],{'max_relative_l2':1e-5})} for n,v in [('cpu_fp32_accum',cpu),('torch_npu_bf16',npu_bf16),('torch_npu_fp32_accum',npu_fp32)]}
        row={'stage':'fixture_failure_localization','case':case,'block_dim':a.bd,'repeat':repeat,'failures':failures,
             'gradients':{n:a2_gradient_metrics(v,golden[n],{**rules['default'],**rules.get('outputs',{}).get(n,{})}) for n,v in actual.items()},
             'pair_same_native_input_replay':pair,'stage_digests':{n:_a2_digest(v) for n,v in stages.items()},'launches':native.trace[start:]}
        torch.save({'inputs':inputs,'native':actual,'native_stages':stages,'expected_stages':expected},a.out/f"{case['id']}-repeat{repeat}.pt")
        rows.append(row);emit(row)
(a.out/'receipt.json').write_text(json.dumps({'stage':'fixture_failure_diagnosis','rows':rows,'thresholds_unchanged':True},indent=2)+'\n')
