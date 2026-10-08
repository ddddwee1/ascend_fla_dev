"""Bounded source-based models on arguments captured from a fresh native full chain."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import torch,ascriptor
from benchmarks.verify_decode import _a2_unit,_a2_digest
from benchmarks.probe_bwd_span import a2_gradient_metrics
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--model',choices=('sim','pipesim'),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
ws=Path(os.environ['ASCRIPTOR_WORKSPACE']);repo=Path(__file__).resolve().parent/'repo';library=Path(ascriptor.__file__).resolve().parents[1]
assert library==(ws/'library').resolve()
pin=json.loads((repo/'docs/matrix/ops.json').read_text())['ascriptor_pin']['current_compatibility_pin']
for name in ('library','kernels'):
    target=pin[name]['commit'] if isinstance(pin[name],dict) else pin[name]
    assert subprocess.check_output(['git','-C',str(ws/name),'rev-parse','HEAD'],text=True).strip()==target
for name in subprocess.check_output(['git','-C',str(library),'ls-files','ascriptor'],text=True).splitlines():
    assert (library/name).read_bytes()==subprocess.check_output(['git','-C',str(library),'show','HEAD:'+name])
cap=torch.load(a.capture,map_location='cpu',weights_only=False);assert cap['block_dim']==1
unit,_=_a2_unit('kda_bwd_stable_fp32gc');name=cap['kernel'].split('.')[1];kernel=unit._kernels()[name];assert kernel.name==cap['kernel_symbol']
args=tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in cap['args']);started=time.monotonic();evidence={}
if a.model=='sim':
    from ascriptor.backends.sim.launch import run_kernel
    result=run_kernel(kernel,*args,block_dim=1,timeout=60,seed_outputs=True,processes=False)
    result=result if isinstance(result,tuple) else (result,)
else:
    from ascriptor.backends.sim.pipesim import simulate
    from ascriptor.passes import PIPELINE,PassManager
    from ascriptor.passes.autosync import check_balance
    lowered=PassManager(PIPELINE).run(kernel.ir());balance=check_balance(lowered);assert not balance,balance
    simulation=simulate(lowered,args,block_dim=1,timeout=60,seed_outputs=True,check_gm=True,processes=False)
    simulation.write_trace(a.out/'trace.private.json')
    evidence={'event_balance':balance,'hazards':simulation.hazards,'report':simulation.report,'model_cycles':simulation.cycles}
    result=tuple(simulation.outputs)
metrics=[a2_gradient_metrics(g,e,{'max_relative_l2':.05}) for g,e in zip(result,cap['actual_outputs'])]
case_id=a.capture.name.split('-bwd-')[0]
contract=json.loads((unit.HERE/'contract.json').read_text())
case=next(x for x in contract['cases'] if x['id']==case_id)
fresh_inputs=unit.make_inputs(case)
expected=unit.reference_stages(fresh_inputs)
stage_names={
 'scan_fused':['scan.dAqk','scan.dh','scan.dv','scan.dh0'],
 'inverse_mm':['inverse_mm.d_qg','inverse_mm.d_kg','inverse_mm.d_w','inverse_mm.d_v_beta','inverse_mm.d_k_beta_g'],
 'inverse_epilogue':['inverse_epilogue.dq_hv','inverse_epilogue.dk_hv','inverse_epilogue.dv','inverse_epilogue.dbeta','inverse_epilogue.dg_core','inverse_epilogue.k_exp'],
 'finalize_pre':['finalize_pre.q_scaled','finalize_pre.k_scaled','finalize_pre.kg','finalize_pre.M_qk','finalize_pre.M_base','finalize_pre.M_beta'],
 'finalize_post':['finalize_post.dq_hv','finalize_post.dk_hv','finalize_post.dbeta','finalize_post.dg']}
rules=contract['comparison']
independent={n:a2_gradient_metrics(g.reshape(expected[n].shape),expected[n],{**rules['default'],**rules.get('stage_outputs',{}).get(n,{})}) for n,g in zip(stage_names[name],result)}
assert len(result)==len(cap['actual_outputs'])
receipt={'model':a.model,'kernel':cap['kernel'],'block_dim':1,'capture_sha256':hashlib.sha256(a.capture.read_bytes()).hexdigest(),
         'source_sha256':hashlib.sha256((unit.HERE/'kernels'/f'{name}.py').read_bytes()).hexdigest(),'ascriptor':ascriptor.__version__,
         'comparison_scope':'model actual outputs versus captured native same-input output; CPU golden checks are in native full-chain evidence',
         'metrics':metrics,'fresh_cpu_fp32_full_fixture_stage_comparison':independent,'model_output_digests':[_a2_digest(x) for x in result],'elapsed_host_s':time.monotonic()-started,
         'hardware_acceptance':False,**evidence}
(a.out/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str,allow_nan=False)+'\n');print(json.dumps(receipt,default=str),flush=True)
if evidence.get('hazards') or evidence.get('report',{}).get('deadlock') or not all(m['passed'] for m in metrics) or not all(m['passed'] for m in independent.values()):raise SystemExit(1)
