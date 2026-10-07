"""Located exponential/materialization comparison; CPU FP32 golden remains unchanged."""
import argparse,hashlib,json,math
from pathlib import Path
import torch
from benchmarks.verify_decode import _a2_identity,_a2_digest
p=argparse.ArgumentParser();p.add_argument('--bd',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parent;identity=_a2_identity();identity['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(a.out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
paths=[root/'native/fp32gc-fixture-diag-v1-bd1/receipts/grid_c1_hv2_bd1-repeat0.pt']
paths+=sorted((root/'native/fp32gc-seeds-v1-bd1/receipts').glob('*.pt'))
paths+=sorted((root/'native/fp32gc-stages-seed0-v1-bd1/receipts').glob('*.pt'))
rows=[]
for path in paths:
    data=torch.load(path,map_location='cpu',weights_only=False);x=data['inputs'];saved=data.get('saved',x.get('saved'));stages=data['native_stages'];b,t,h,d=x['q'].shape;hv=x['v'].shape[2];c=t//64
    def pack(v):return v.reshape(b,c,64,hv,128).permute(0,3,1,2,4).contiguous().float()
    g=pack(saved['g_cumsum']);mid=g[...,-1:,:]*.5
    for name,base,delta in [('q_scaled','q',g-mid),('k_scaled','k',g-mid),('kg','k',mid-g)]:
        v=pack(x[base].repeat_interleave(hv//h,dim=2));argument=delta*math.log(2.)
        cpu_exp=argument.exp();cpu_product=v*cpu_exp;cpu_bf16=cpu_product.bfloat16()
        npu_exp=argument.to('npu').exp().cpu();npu_product=(v.to('npu')*npu_exp.to('npu')).cpu();npu_bf16=npu_product.bfloat16()
        native=stages['finalize_pre.'+name];indices=(native!=cpu_bf16).nonzero();items=[]
        for idx in indices[:32]:
            ix=tuple(idx.tolist());items.append({'index':list(ix),'operand_fp32':float(v[ix]),'exponent_fp32':float(argument[ix]),'cpu_exp_fp32':float(cpu_exp[ix]),'npu_exp_fp32':float(npu_exp[ix]),'cpu_product_fp32':float(cpu_product[ix]),'npu_product_fp32':float(npu_product[ix]),'cpu_bf16':float(cpu_bf16[ix]),'npu_bf16':float(npu_bf16[ix]),'native_bf16':float(native[ix])})
        row={'case':path.stem,'stage':'finalize_pre.'+name,'native_vs_cpu_different':len(indices),'npu_builtin_vs_native_bitwise':torch.equal(npu_bf16,native),'npu_builtin_vs_native_different':int((npu_bf16!=native).sum()),'examples':items,'native_digest':_a2_digest(native)}
        rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
receipt={'stage':'torch_npu_exp_localization','custom_launches':0,'golden':'CPU FP32, not replaced by NPU','rows':rows}
(a.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
