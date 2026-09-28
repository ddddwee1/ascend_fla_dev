"""Locate cache precision errors. CPU counterfactuals never count as NPU acceptance."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from benchmarks.probe_bwd_span import A2BackwardNative, a2_gradient_metrics, a2_test_helpers
from benchmarks.verify_decode import _a2_identity, _a2_digest

p = argparse.ArgumentParser()
p.add_argument('--bd', type=int, choices=(1, 2), required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--spans', type=float, nargs='+', default=[24., 28., 32.])
p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=False)

def emit(row):
    line = json.dumps({'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **row},
                      allow_nan=False)
    with (args.out/'events.jsonl').open('a') as stream:
        stream.write(line+'\n')
    print(line, flush=True)

identity = _a2_identity()
repo = Path(__import__('benchmarks.probe_bwd_span', fromlist=['x']).__file__).resolve().parents[1]
paths = [repo/'benchmarks/probe_bwd_span.py']
paths += [repo/'tests'/f'test_kda_{suffix}_npu.py' for suffix in ('bwd', 'bwd_deep', 'caches')]
paths += [x for x in (repo/'kernels/projects/a2/kda_bwd_stable').rglob('*')
          if x.is_file() and (x.suffix=='.py' or x.name=='contract.json')
          and 'evidence' not in x.parts and '__pycache__' not in x.parts]
identity['diagnostic_sources'] = {str(x.relative_to(repo)):hashlib.sha256(x.read_bytes()).hexdigest()
                                for x in paths}
identity['diagnostic_runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(args.out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
emit({'stage':'identity_verified','soc':identity['soc'],'cann':identity['cann'],
      'opp_packages':identity['opp_packages']})
native = A2BackwardNative(args.bd, emit)
fixtures, oracle, caches = a2_test_helpers()
oracle.test_a2_segmented_oracle_matches_full_autograd()
rules = native.contract['comparison']
output_stages = {'dq':'finalize_reduce.dq','dk':'finalize_reduce.dk','dv':'inverse_epilogue.dv',
                 'dg':'finalize_post.dg','dbeta':'finalize_post.dbeta','dh0':'scan.dh0'}
rows=[]
for case in fixtures.a2_span_cases([128], args.seeds, args.spans, ['uniform']):
    emit({'stage':'case_start','case':case})
    inputs, distribution = fixtures.a2_qualification_inputs(case)
    trace_start=len(native.trace)
    x, raw, assembly = caches.a2_actual_forward_caches(native, inputs)
    actual, stages = native.backward_call(x)
    actual={n:v.detach().cpu() for n,v in actual.items()}
    stages={n:v.detach().cpu() for n,v in stages.items()}
    _, golden = oracle.a2_fp32_end_to_end(inputs)
    replay = native.backward.reference_stages(x)
    stage_metrics = {n:a2_gradient_metrics(stages[n], replay[n],
                     {**rules['default'],**rules.get('stage_outputs',{}).get(n,{})})
                     for n in stages}
    original_metrics={n:a2_gradient_metrics(actual[n], golden[n],{'max_relative_l2':.05})
                      for n in output_stages}
    replay_metrics={n:a2_gradient_metrics(replay[stage], golden[n],{'max_relative_l2':.05})
                    for n,stage in output_stages.items()}
    native_replay={n:a2_gradient_metrics(actual[n],replay[stage],{'max_relative_l2':.05})
                   for n,stage in output_stages.items()}
    b,t,h,d=inputs['q'].shape;hv=inputs['v'].shape[2]
    exact_gc=(raw['g_cumsum']*(1.0/math.log(2))).permute(0,2,3,1,4).contiguous().view(b,t,hv,d)
    changed={**x,'saved':{**x['saved'],'g_cumsum':exact_gc}}
    # Deliberately call only the pure CPU mathematics, not unit validation/launch.
    counterfactual = native.backward_ref.reference_stages(changed)
    counterfactual_metrics={n:a2_gradient_metrics(counterfactual[stage],golden[n],{'max_relative_l2':.05})
                            for n,stage in output_stages.items()}
    row={'stage':'precision_diagnostic','case':case,'distribution':distribution,'assembly':assembly,
         'native_vs_full_cpu_fp32':original_metrics,'native_vs_same_cache_cpu_replay':native_replay,
         'same_cache_cpu_replay_vs_full_fp32':replay_metrics,'native_33_stage_checks':stage_metrics,
         'cpu_only_fp32_gc_counterfactual_vs_full_fp32':counterfactual_metrics,
         'cpu_counterfactual_is_hardware_evidence':False,
         'counterfactual_change':'only g_cumsum: actual forward FP32 natural-log cache * 1/ln2; no BF16 narrowing',
         'native_bf16_gc_digest':_a2_digest(x['saved']['g_cumsum']),
         'cpu_counterfactual_fp32_gc_digest':_a2_digest(exact_gc),
         'native_output_digests':{n:_a2_digest(v) for n,v in actual.items()},
         'launches':native.trace[trace_start:]}
    # Preserve actual tensors locally to support further located, read-only analysis.
    torch.save({'inputs':inputs,'saved':x['saved'],'raw_forward':raw,'native_outputs':actual,
                'native_stages':stages,'golden':golden},args.out/(case['id']+'.pt'))
    emit(row);rows.append(row)
(args.out/'receipt.json').write_text(json.dumps({'stage':'diagnosis_only','rows':rows,
    'native_acceptance_passed':all(m['passed'] for r in rows for m in r['native_vs_full_cpu_fp32'].values()),
    'no_repaired_kernel_executed':True},indent=2,allow_nan=False)+'\n')
