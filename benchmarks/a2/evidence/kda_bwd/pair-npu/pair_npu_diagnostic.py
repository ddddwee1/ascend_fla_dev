"""Torch NPU matmul diagnostic of retained native pair inputs; no custom launches."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch_npu
from benchmarks.verify_decode import _a2_identity,_a2_unit
from benchmarks.probe_bwd_span import a2_gradient_metrics
p=argparse.ArgumentParser();p.add_argument('--bd',type=int,required=True);p.add_argument('--out',type=Path,required=True)
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False
identity=_a2_identity();identity['diagnostic_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(args.out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
root=Path(__file__).resolve().parent
unit,_=_a2_unit('kda_bwd_stable')
rules=json.loads((unit.HERE/'contract.json').read_text())['comparison']['stage_outputs']
rows=[]
for span in (24,28,32):
    source=root/'native/precision-v1-bd1/receipts'/f'range_uniform_t128_seed0_span{span}.pt'
    data=torch.load(source,map_location='cpu',weights_only=True);actual=data['native_stages']
    pairs={'qk_left':('M_qk','kg',False),'qk_right':('M_qk','q_scaled',True),
           's_base':('M_base','kg',False),'t_beta':('M_beta','k_scaled',True)}
    checks={}
    for name,(left,right,transpose) in pairs.items():
        key='finalize_pair.'+name;lhs=actual['finalize_pre.'+left];rhs=actual['finalize_pre.'+right]
        if transpose:lhs=lhs.transpose(-1,-2)
        lhs=lhs.contiguous();rhs=rhs.contiguous()
        expected=(lhs.float()@rhs.float()).bfloat16().reshape_as(actual[key])
        outputs={}
        for dtype in (torch.float32,torch.bfloat16):
            product=lhs.to(dtype).to('npu')@rhs.to(dtype).to('npu')
            torch.npu.synchronize()
            cpu=product.detach().cpu().bfloat16().reshape_as(actual[key])
            outputs[str(dtype)]={'vs_cpu_fp32_golden':a2_gradient_metrics(cpu,expected,rules[key]),
                'vs_retained_custom_native':a2_gradient_metrics(cpu,actual[key],rules[key]),
                'identical_to_retained_native_bytes':torch.equal(cpu.view(torch.uint8),actual[key].view(torch.uint8))}
        checks[key]=outputs
    row={'span':span,'source_tensor_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'stage':'fresh Torch NPU pair diagnostic with retained native inputs','source_native_block_dim':args.bd,
         'custom_launches':0,'checks':checks}
    rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
(args.out/'receipt.json').write_text(json.dumps({'stage':'Torch NPU diagnostic, not repaired-kernel acceptance',
    'custom_launches':0,'torch_npu_allow_hf32':torch.npu.matmul.allow_hf32,'rows':rows},indent=2,allow_nan=False)+'\n')
