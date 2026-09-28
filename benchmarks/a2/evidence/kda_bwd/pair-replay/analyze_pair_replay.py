"""Read-only CPU analysis of retained native checkpoint tensors."""
from pathlib import Path
import json
import torch
from benchmarks.verify_decode import _a2_unit
from benchmarks.probe_bwd_span import a2_gradient_metrics

root=Path(__file__).resolve().parent
unit,ref=_a2_unit('kda_bwd_stable')
rules=json.loads((unit.HERE/'contract.json').read_text())['comparison']['stage_outputs']
rows=[]
for span in (24,28,32):
    source=root/'native/precision-v1-bd1/receipts'/f'range_uniform_t128_seed0_span{span}.pt'
    data=torch.load(source,map_location='cpu',weights_only=True)
    saved={**data['inputs'],'saved':data['saved']}
    expected=ref.reference_stages(saved)
    actual=data['native_stages']
    pairs={'qk_left':('M_qk','kg',False),'qk_right':('M_qk','q_scaled',True),
           's_base':('M_base','kg',False),'t_beta':('M_beta','k_scaled',True)}
    differences={n:int((v!=expected[n]).sum()) for n,v in actual.items() if n.startswith('finalize_pre.')}
    checks={}
    for name,(left,right,transpose) in pairs.items():
        key='finalize_pair.'+name
        lhs=actual['finalize_pre.'+left].float()
        if transpose:lhs=lhs.transpose(-1,-2)
        rhs=actual['finalize_pre.'+right].float()
        fp32=(lhs@rhs).bfloat16().reshape_as(actual[key])
        fp64=(lhs.double()@rhs.double()).bfloat16().reshape_as(actual[key])
        checks[key]={'native_vs_actual_input_fp32_matmul':a2_gradient_metrics(actual[key],fp32,rules[key]),
                     'native_vs_actual_input_fp64_matmul_diagnostic':a2_gradient_metrics(actual[key],fp64,rules[key]),
                     'fp32_mismatched_elements':int((actual[key]!=fp32).sum()),
                     'fp64_mismatched_elements':int((actual[key]!=fp64).sum())}
    row={'span':span,'stage':'CPU diagnosis of saved native tensors','pre_stage_element_mismatches':differences,
         'pair_checks':checks,'fp64_is_not_acceptance_golden':True}
    rows.append(row);print(json.dumps(row),flush=True)
(root/'pair-replay-analysis.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
