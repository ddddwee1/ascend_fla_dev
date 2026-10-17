"""Replay three synchronized forward timing sandwiches after complete precompile.

Select the accepted environment, device lock, and isolated cache through external
ignored machine configuration. This does not measure training or full-layer time.
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

from ascend_fla.reference.kda import kda_chunk_vectorized
from benchmarks.bench_kda_torch_npu import SHAPES,make_inputs as performance_inputs
import statistics

def timed(fn):
 for _ in range(3):fn()
 torch.npu.synchronize();start=time.perf_counter()
 for _ in range(10):fn()
 torch.npu.synchronize();return (time.perf_counter()-start)*1000/10

rows=[]
with patch.object(compiler,'compile_kernel',track_compile),patch.dict(platform.CAPABILITIES['a2'],{'qualified':True}),torch.no_grad():
 prepare('a2',bd,backward=True,decode=True)
 assert len(compiled)==65
 identity['scope']='Three-round synchronized public forward/cached-forward versus Torch NPU; no training timing'
 (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
 print(json.dumps({'stage':'all_public_vendors_ready','count':65}),flush=True)
 for name in ('long_context','kimi_linear_layer','qwen3_next_layer'):
  x=performance_inputs(**SHAPES[name],dtype=torch.bfloat16,device='npu')
  options=dict(device='a2',block_dim=bd,layout_device='npu');rounds=[];launched=True
  for index in range(3):
   before=timed(lambda:kda_chunk_vectorized(**x,output_final_state=True))
   fwd=timed(lambda:chunk_kda_fwd(**x,output_final_state=True,check_gate_range=False,**options))
   checked=timed(lambda:chunk_kda_fwd(**x,output_final_state=True,**options))
   cached=timed(lambda:chunk_kda_fwd_with_caches(**x,**options))
   after=timed(lambda:kda_chunk_vectorized(**x,output_final_state=True))
   rounds.append({'round':index+1,'baseline_before_ms':before,'forward_ms':fwd,'checked_forward_ms':checked,'cached_forward_ms':cached,'baseline_after_ms':after,'forward_speedup':((before+after)/2)/fwd})
   print(json.dumps({'stage':'round_complete','shape':name,**rounds[-1]}),flush=True)
   (out/'partial-rounds.json').write_text(json.dumps({'shape':name,'rounds':rounds},indent=2)+'\n')
  row={'shape':name,'dimensions':SHAPES[name],'dtype':'torch.bfloat16','block_dim':bd,'warmup':3,'iterations':10,'synchronized':True,'backward_precompiled':True,'rounds':rounds,
       'median_forward_ms':statistics.median(x['forward_ms'] for x in rounds),'median_cached_forward_ms':statistics.median(x['cached_forward_ms'] for x in rounds),'median_forward_speedup':statistics.median(x['forward_speedup'] for x in rounds),
       'scope':'Operator forward/cached-forward only; not complete layer or training performance'}
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'stage':'complete','shapes':len(rows),'rounds_each':3,'qualified_after':platform.CAPABILITIES['a2']['qualified']}),flush=True)
