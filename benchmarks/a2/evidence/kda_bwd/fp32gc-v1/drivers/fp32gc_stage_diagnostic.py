"""Full native FP32-gc chain plus unchanged CPU FP32 checkpoint diagnostics."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import torch
from benchmarks.probe_bwd_span import A2BackwardNative, a2_gradient_metrics, a2_test_helpers
from benchmarks.verify_decode import _a2_identity, _a2_digest

p=argparse.ArgumentParser()
p.add_argument('--bd',type=int,choices=(1,2),required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--spans',type=float,nargs='+',default=[24.,28.,32.,64.,96.,106.5,128.])
p.add_argument('--seeds',type=int,nargs='+',default=[0])
p.add_argument('--lengths',type=int,nargs='+',default=[128])
p.add_argument('--gates',nargs='+',choices=('uniform','fla_initialization'),default=['uniform'])
p.add_argument('--actual-suite',action='store_true')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
def emit(row):
    line=json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**row},allow_nan=False)
    with (a.out/'events.jsonl').open('a') as f:f.write(line+'\n')
    print(line,flush=True)
identity=_a2_identity()
repo=Path(__import__('benchmarks.probe_bwd_span',fromlist=['x']).__file__).resolve().parents[1]
paths=[repo/'benchmarks/probe_bwd_span.py']+[repo/'tests'/f'test_kda_{n}_npu.py' for n in ('bwd','bwd_deep','caches')]
paths += [x for x in (repo/'kernels/projects/a2/kda_bwd_stable_fp32gc').rglob('*') if x.is_file() and (x.suffix=='.py' or x.name=='contract.json') and '__pycache__' not in x.parts]
identity.update(backward_unit='kda_bwd_stable_fp32gc',gc_dtype='fp32',diagnostic_sources={str(x.relative_to(repo)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},diagnostic_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(a.out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
emit({'stage':'identity_verified','soc':identity['soc'],'cann':identity['cann'],'opp_packages':identity['opp_packages']})
native=A2BackwardNative(a.bd,emit,gc_dtype='fp32')
fixtures,oracle,caches=a2_test_helpers();oracle.test_a2_segmented_oracle_matches_full_autograd()
rules=native.contract['comparison'];rows=[]
cases=fixtures.a2_qualification_cases() if a.actual_suite else fixtures.a2_span_cases(a.lengths,a.seeds,a.spans,a.gates)
for case in cases:
    emit({'stage':'case_start','case':case});inputs,distribution=fixtures.a2_qualification_inputs(case);start=len(native.trace)
    x,raw,assembly=caches.a2_actual_forward_caches(native,inputs)
    bad={n:int((~v.isfinite()).sum()) for n,v in x['saved'].items()}
    if any(bad.values()):
        row={'stage':'pre_backward_nonfinite','case':case,'nonfinite':bad,'passed':False,'native_backward_executed':False}
        rows.append(row);emit(row);continue
    actual,stages=native.backward_call(x)
    actual={n:v.detach().cpu() for n,v in actual.items()};stages={n:v.detach().cpu() for n,v in stages.items()}
    emit({'stage':'full_native_chain_complete','case':case['id'],'launch_count':len(native.trace)-start})
    _,golden=oracle.a2_fp32_end_to_end(inputs)
    _,npu_golden=oracle.a2_fp32_end_to_end(inputs,device='npu')
    npu_reference_metrics={n:a2_gradient_metrics(npu_golden[n],golden[n],{'max_relative_l2':.05}) for n in golden}
    del npu_golden
    replay=native.backward.reference_stages(x)
    metrics={n:a2_gradient_metrics(actual[n],golden[n],{'max_relative_l2':.05}) for n in actual}
    checkpoints={n:a2_gradient_metrics(stages[n],replay[n],{**rules['default'],**rules.get('stage_outputs',{}).get(n,{})}) for n in stages}
    slices={}
    for n,v in actual.items():
        detail=[]
        if n=='dh0':
            for b in range(v.shape[0]):
                for h in range(v.shape[1]):detail.append({'batch':b,'head':h,**a2_gradient_metrics(v[b,h],golden[n][b,h],{'max_relative_l2':.05})})
        else:
            for b in range(v.shape[0]):
                for c in range(v.shape[1]//64):
                    for h in range(v.shape[2]):detail.append({'batch':b,'chunk':c,'head':h,**a2_gradient_metrics(v[b,c*64:(c+1)*64,h],golden[n][b,c*64:(c+1)*64,h],{'max_relative_l2':.05})})
        slices[n]=detail
    row={'stage':'fp32gc_stage_diagnostic','case':case,'block_dim':a.bd,'distribution':distribution,'assembly':assembly,
         'native_vs_full_cpu_fp32':metrics,'native_33_stage_checks':checkpoints,'torch_npu_fp32_vs_cpu':npu_reference_metrics,
         'per_head_chunk_diagnostic':slices,'slice_budget_is_diagnostic_marker_only':True,
         'native_output_digests':{n:_a2_digest(v) for n,v in actual.items()},
         'native_stage_digests':{n:_a2_digest(v) for n,v in stages.items()},
         'launches':native.trace[start:],'end_to_end_passed':all(m['passed'] for m in metrics.values()),
         'original_checkpoint_budgets_passed':all(m['passed'] for m in checkpoints.values())}
    torch.save({'inputs':inputs,'saved':x['saved'],'raw_forward':raw,'native_outputs':actual,'native_stages':stages,'golden':golden},a.out/(case['id']+'.pt'))
    rows.append(row);emit(row)
receipt={'stage':'full_native_with_checkpoint_diagnostics','gc_dtype':'fp32','rows':rows,
         'end_to_end_passed':all(r.get('end_to_end_passed',False) for r in rows),
         'original_checkpoint_budgets_passed':all(r.get('original_checkpoint_budgets_passed',False) for r in rows)}
(a.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
raise SystemExit(0 if receipt['end_to_end_passed'] and receipt['original_checkpoint_budgets_passed'] else 1)
