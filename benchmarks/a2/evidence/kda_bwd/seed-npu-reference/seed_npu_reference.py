"""Fresh Torch NPU FP32 end-to-end references for saved full native seed cases."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch_npu
from benchmarks.verify_decode import _a2_identity
from benchmarks.probe_bwd_span import a2_test_helpers,a2_gradient_metrics
p=argparse.ArgumentParser();p.add_argument('--bd',type=int,required=True);p.add_argument('--out',type=Path,required=True)
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False
identity=_a2_identity();identity['diagnostic_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(args.out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
root=Path(__file__).resolve().parent
source=root/'native/precision-seeds-v1-bd1/receipts'
assert (source/'receipt.json').exists(), 'Full native chain must finish first'
_,oracle,_=a2_test_helpers();rows=[]
for path in sorted(source.glob('*.pt')):
    data=torch.load(path,map_location='cpu',weights_only=True)
    _,reference=oracle.a2_fp32_end_to_end(data['inputs'],device='npu')
    torch.npu.synchronize()
    metrics={n:a2_gradient_metrics(reference[n],data['golden'][n],{'max_relative_l2':.05}) for n in data['golden']}
    native={n:a2_gradient_metrics(data['native_outputs'][n],reference[n],{'max_relative_l2':.05}) for n in data['golden']}
    row={'case':path.stem,'source_tensor_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'stage':'fresh Torch NPU FP32 full autograd reference; no custom launches',
         'reference_vs_cpu_fp32':metrics,'retained_native_vs_torch_npu_fp32':native}
    rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
assert len(rows)==6
passed=all(v['passed'] for r in rows for v in r['reference_vs_cpu_fp32'].values())
(args.out/'receipt.json').write_text(json.dumps({'stage':'reference validation only','custom_launches':0,
    'reference_agreement_passed':passed,'rows':rows},indent=2,allow_nan=False)+'\n')
raise SystemExit(0 if passed else 1)
