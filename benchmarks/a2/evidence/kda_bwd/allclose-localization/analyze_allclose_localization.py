"""Locate original allclose failures in retained native seed checkpoints on CPU."""
import hashlib,json,math
from pathlib import Path
import torch
from benchmarks.verify_decode import _a2_unit
root=Path(__file__).resolve().parent;unit,ref=_a2_unit('kda_bwd_stable')
rules=json.loads((unit.HERE/'contract.json').read_text())['comparison'];rows=[]
for span in (106.5,128):
    path=root/f'native/precision-seeds-v1-bd1/receipts/range_uniform_t128_seed1_span{span}.pt'
    data=torch.load(path,map_location='cpu',weights_only=True)
    expected=ref.reference_stages({**data['inputs'],'saved':data['saved']})
    for stage in ('finalize_pre.k_scaled','finalize_pair.s_base'):
        actual=data['native_stages'][stage];golden=expected[stage]
        rule={**rules['default'],**rules.get('stage_outputs',{}).get(stage,{})}
        close=torch.isclose(actual.float(),golden.float(),rtol=rule['rtol'],atol=rule['atol'])
        bad=(~close).nonzero();points=[]
        for idx in bad[:64]:
            pos=tuple(idx.tolist());a=actual[pos].float().item();g=golden[pos].float().item()
            point={'index':list(pos),'native':a,'cpu_fp32_seam':g,'absolute_difference':abs(a-g),
                   'allclose_allowance':rule['atol']+rule['rtol']*abs(g),
                   'bf16_bit_distance':abs(int(actual[pos].view(torch.int16))-int(golden[pos].view(torch.int16)))}
            if stage=='finalize_pre.k_scaled':
                gate=ref._pack(data['saved']['g_cumsum']).float()
                k=ref._pack(data['inputs']['k']).float()
                shift=gate-gate[...,-1:,:]*.5
                # Diagnostic precision only; neither expression replaces the original golden.
                exact=(k.double()*torch.exp(shift.double()*math.log(2)))
                point.update(shift_fp32=shift[pos].item(),key_fp32=k[pos].item(),
                    cpu_fp64_unrounded_diagnostic=exact[pos].item(),
                    cpu_fp64_to_bf16_diagnostic=exact[pos].bfloat16().float().item(),
                    bf16_midpoint_between_observed_values=(a+g)/2)
            points.append(point)
        row={'span':span,'stage':stage,'allclose_failures':len(bad),'total_elements':actual.numel(),
             'points_truncated':len(bad)>len(points),'points':points,'rule_unchanged':rule,
             'source_tensor_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'fp64_is_not_acceptance_golden':True}
        rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
(root/'allclose-localization.json').write_text(json.dumps({'rows':rows},indent=2,allow_nan=False)+'\n')
