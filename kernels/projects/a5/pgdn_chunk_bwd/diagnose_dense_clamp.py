"""Supplemental native diagnostic of dense FP32 norm clamp branch selection."""
import argparse,json,sys,hashlib
from pathlib import Path
unit=Path(__file__).resolve().parent
sys.path.insert(0,str(unit))
import torch
from ascend_fla.ops.pgdn_chunk_bwd import prepare,_compiled,_pipeline,chunk_pgdn_bwd
from benchmark import references
from ref.calibrate import inputs
from ref.reference import INPUTS,NAMES
from ref.classification import compare,write_disclosures
from ref.verification import digest
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--block-dim',type=int,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
import torch_npu
torch.npu.set_device(0);prepare(block_dim=a.block_dim);pipeline=_pipeline();compiled=dict(zip((e.name for e in pipeline.entries()),_compiled(a.block_dim)))
def launch(entry,sources,outputs,scalars):
 for x in outputs.values():x.fill_(float('nan'))
 op=compiled[entry.name];op(sources,{n:scalars[n] for n in op.scalar_names},outputs);return outputs
report=dict(stage='supplemental_dense_clamp_diagnostic',production_changed=False,budget_changed=False,cases=[])
for mode in ('q','k','both'):
 case=dict(id='dense_clamp_'+mode,B=1,T=64,H=1,HV=2,kind='random',mask=7,seed=195082)
 xs,ds=inputs(case);cpu={**dict(zip(INPUTS,xs)),**ds};eps=torch.tensor(1e-12,dtype=torch.float32)
 for name in (('q','k') if mode=='both' else (mode,)):
  x=cpu[name];cpu[name]=(x/x.norm(dim=-1,keepdim=True)*eps).contiguous()
 expected=references(cpu);gpu={n:None if x is None else x.npu() for n,x in cpu.items()}
 actual=pipeline.run(gpu,launch,retain_stages=True);torch.npu.synchronize();host={n:x.cpu() for n,x in actual.items()}
 public=chunk_pgdn_bwd(**gpu,block_dim=a.block_dim);torch.npu.synchronize();public={n:x.cpu() for n,x in zip(NAMES,public)}
 comparison=compare(public,{n:expected[n] for n in ('A','B')},expected['classes'])
 ab=compare(expected['B'],{'A':expected['A']},expected['classes'])
 directory=a.output/case['id'];directory.mkdir();torch.save(dict(cpu=cpu,returned=host,public=public,A=expected['A'],B=expected['B']),directory/'diagnostic.private.pt')
 disclosure=write_disclosures(directory/'public-disclosures.jsonl',public,expected['auxiliary'],expected['classes'])
 branches={}
 for n in ('q','k'):
  reference=cpu[n].norm(dim=-1);got=host[n+'_raw_norm'];mismatch=(reference>=eps)!=(got>=eps)
  branches[n]=dict(mismatches=int(mismatch.sum()),positions=mismatch.nonzero().tolist(),cpu_norms=reference.flatten().tolist(),native_norms=got.flatten().tolist(),eps_float32=eps.item(),cpu_at_clamp=int((reference==eps).sum()))
 row=dict(case=case,branches=branches,input_sha256={n:None if x is None else digest(x) for n,x in cpu.items()},comparison=comparison,B_against_A=ab,disclosure=disclosure,public_composition_byte_identical=all(torch.equal(public[n].view(torch.uint8),host[n].view(torch.uint8)) for n in NAMES))
 report['cases'].append(row);(directory/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');(a.output/'diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print('DENSE_CLAMP_DIAGNOSTIC',mode,'branch_mismatches',{n:x['mismatches'] for n,x in branches.items()},'ordinary_budget_satisfied',comparison['ordinary_budget_satisfied'],'B_against_A',ab['ordinary_budget_satisfied'],flush=True)
print('DIAGNOSTIC_EXECUTION_COMPLETE',flush=True)
