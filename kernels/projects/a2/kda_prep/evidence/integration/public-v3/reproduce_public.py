"""Replay A2 public full-shape forward/cached-forward/backward and bd1 layer inference.

Select the accepted native environment, device lock, and isolated compile cache
through external ignored machine configuration before invoking this script.
This process precompiles all 65 vendors; it does not qualify the full layer.
"""
from pathlib import Path
import importlib.util,json,hashlib,os,subprocess,sys,time
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--block-dim',type=int,choices=(1,2),required=True)
args=parser.parse_args()
repo=args.repo.resolve();out=args.output.resolve();bd=args.block_dim
out.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(repo),str(repo/'benchmarks')]
import torch,torch_npu,ascriptor
from verify_decode import _a2_identity
from ascend_fla.runtime.compile import compile_kernel

torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False
identity=_a2_identity();assert identity['python']=='3.11.15' and identity['torch_npu']=='2.10.0' and identity['ascriptor']=='0.1.0'
lib=Path(os.environ['ASCRIPTOR_WORKSPACE'])/'library';assert Path(ascriptor.__file__).resolve()==(lib/'ascriptor/__init__.py').resolve()
assert subprocess.check_output(['git','-C',str(lib),'rev-parse','HEAD'],text=True).strip()=='90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5'
assert not subprocess.check_output(['git','-C',str(lib),'status','--porcelain'],text=True).strip()
from ascend_fla.ops.kda import chunk,chunk_bwd
import importlib
fused=importlib.import_module('ascend_fla.ops.kda.fused_recurrent')
from unittest.mock import patch
from ascend_fla import platform
from ascend_fla.ops.kda import prepare,chunk_kda_fwd,chunk_kda_fwd_with_caches,chunk_kda_bwd
from ascend_fla.runtime import compile as compiler
from verify_real_shapes import make_inputs,ref_fwd,_ref_grads_checkpointed
spec=importlib.util.spec_from_file_location('a215_cpu_cache_reference',repo/'kernels/projects/a2/kda_bwd_stable_fp32gc/reference.py');cache_ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache_ref)
original_compile=compiler.compile_kernel;compiled={};launched=False

def track_compile(entry,**kwargs):
 name=entry.name
 assert not launched or name in compiled,'No new vendor after first custom launch'
 obj=original_compile(entry,**kwargs)
 compiled[name]={'name':name,'signature':obj.signature,'device':kwargs.get('device'),'block_dim':kwargs.get('block_dim')}
 (out/'compile.json').write_text(json.dumps(list(compiled.values()),indent=2)+'\n')
 print(json.dumps({'stage':'registered','name':name,'unique':len(compiled)}),flush=True)
 return obj

def digest(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def metrics(a,b):
 a=a.float();b=b.float();return {'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-30)),'max_abs':float((a-b).abs().max())}
identity.update(block_dim=bd,scope='Fresh public forward/cached-forward and all nine backward kernels; layer autograd separately pending')
identity['source_sha256']={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for sub in ('ascend_fla/ops/kda','kernels/projects/a2/kda_layout','kernels/projects/a2/kda_prep') for p in (repo/sub).rglob('*.py')}
(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
rows=[]
with patch.object(compiler,'compile_kernel',track_compile),patch.dict(platform.CAPABILITIES['a2'],{'qualified':True}):
 prepare('a2',bd,backward=True,decode=True)
 assert len(compiled)==65,(len(compiled),list(compiled))
 assert len([n for n in compiled if n.startswith('kda_prep_a2_backward_')])==16
 print(json.dumps({'stage':'all_public_vendors_ready','count':len(compiled),'prep_backward':16,'core_backward':9}),flush=True)
 for label,b,h,hv,c,span in [('long_context',1,16,32,64,46.),('kimi_linear_layer',1,32,32,16,46.),('qwen3_next_layer',1,16,32,16,46.)]:
  print(json.dumps({'stage':'cpu_reference_start','case':label}),flush=True)
  x=make_inputs(b,h,hv,c,span,seed=2026,want_grads=True)
  expected_o,expected_state=ref_fwd(x); expected_grad=_ref_grads_checkpointed(x,c)
  expected_cache=cache_ref.build_saved_forward(*(x[k] for k in ('q','k','v','g','beta','h0')),64)
  dev={k:v.to('npu') for k,v in x.items()};options=dict(initial_state=dev['h0'],device='a2',block_dim=bd,layout_device='npu')
  print(json.dumps({'stage':'public_run_start','case':label}),flush=True)
  launched=True
  plain_o,plain_state=chunk_kda_fwd(*(dev[k] for k in ('q','k','v','g','beta')),output_final_state=True,**options)
  o,state,caches=chunk_kda_fwd_with_caches(*(dev[k] for k in ('q','k','v','g','beta')),**options)
  beta=chunk._layout_runtime('a2').cast(dev['beta'],torch.bfloat16,device='a2',block_dim=bd)
  gradients=chunk_kda_bwd(*(dev[k] for k in ('q','k','v')),beta,dev['do'],dev['dht'],caches,device='a2',block_dim=bd,layout_device='npu')
  torch.npu.synchronize();actual={'o':o.cpu(),'final_state':state.cpu(),**{k:v.cpu() for k,v in gradients.items()}}
  expected={'o':expected_o,'final_state':expected_state,**expected_grad}
  budgets={'o':.05,'final_state':.05,'dq':.05,'dk':.15,'dv':.05,'dg':.25,'dbeta':.05,'dh0':.05}
  comparisons={k:metrics(actual[k],expected[k]) for k in expected}
  cached={k:v.cpu() for k,v in caches.items()};assert set(cached)==set(chunk.BWD_CACHE_NAMES)
  cache_comparisons={k:metrics(cached[k],expected_cache[k]) for k in cached}
  assert cached['g_cumsum'].dtype==torch.float32 and all(v.dtype==torch.bfloat16 for k,v in cached.items() if k!='g_cumsum')
  row={'case':label,'shape':[b,c*64,h,hv,128],'span':span,'block_dim':bd,'cpu_fp32':comparisons,'budgets':budgets,'caches_cpu_fp32':cache_comparisons,'cache_budget':.02,
       'plain_cached_o_bitwise':digest(plain_o.cpu())==digest(actual['o']),'plain_cached_state_bitwise':digest(plain_state.cpu())==digest(actual['final_state']),
       'input_sha256':{k:digest(v) for k,v in x.items()},'output_sha256':{k:digest(v) for k,v in actual.items()},'cache_sha256':{k:digest(v) for k,v in cached.items()}}
  row['per_head_chunk_output']=[{'batch':bi,'head':head,'chunk':ci,**metrics(actual['o'][bi,ci*64:(ci+1)*64,head],expected_o[bi,ci*64:(ci+1)*64,head])} for bi in range(b) for head in range(hv) for ci in range(c)]
  row['per_head_state']=[{'batch':bi,'head':head,**metrics(actual['final_state'][bi,head],expected_state[bi,head])} for bi in range(b) for head in range(hv)]
  row['passed']=all(x['finite'] and x['relative_l2']<.05 for x in row['per_head_chunk_output']+row['per_head_state']) and all(v['finite'] and v['relative_l2']<budgets[k] for k,v in comparisons.items()) and all(v['finite'] and v['relative_l2']<.02 for v in cache_comparisons.values()) and row['plain_cached_o_bitwise'] and row['plain_cached_state_bitwise']
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)

 # The original per-token layer inference test is independent of autograd helper edits.
 # It fixes block_dim=1 internally, so execute it only in that process.
 if bd==1:
  test_path=repo/'tests/test_kda_layer_npu.py'
  spec=importlib.util.spec_from_file_location('a215_original_layer_tests',test_path)
  tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
  observed=[];measure=tests._rel_l2
  def capture_error(actual,expected):
   value=measure(actual,expected)
   observed.append({'relative_l2':value,'actual_sha256':digest(actual.detach().cpu()),'expected_sha256':digest(expected.detach().cpu())})
   return value
  print(json.dumps({'stage':'original_layer_prefill_decode_start'}),flush=True)
  with patch.object(tests,'_rel_l2',capture_error):
   tests.test_prefill_then_decode_matches_one_shot_reference()
  assert len(observed)==6
  receipt={'passed':True,'test':'test_prefill_then_decode_matches_one_shot_reference','source_sha256':hashlib.sha256(test_path.read_bytes()).hexdigest(),'block_dim':1,'prefill_tokens':128,'prefill':observed[0],'decode':[{'token':128+i,**row} for i,row in enumerate(observed[1:])],'budget':tests._budget('output'),'scope':'Original layer inference test only; complete layer training tests remain pending'}
  (out/'layer-prefill-decode.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print(json.dumps(receipt),flush=True)

assert platform.CAPABILITIES['a2']['qualified'] is False
print(json.dumps({'stage':'complete','cases':len(rows),'failed':[r['case'] for r in rows if not r['passed']],'qualified_after':False}),flush=True)
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
