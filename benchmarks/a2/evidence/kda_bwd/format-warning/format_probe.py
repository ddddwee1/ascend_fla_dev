"""Investigate a Torch NPU allocation-format warning without custom kernels."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from benchmarks.verify_decode import _a2_identity

p=argparse.ArgumentParser()
p.add_argument('--bd',type=int,required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
identity=_a2_identity()
(args.out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
rows=[]
for shape in ((1,64,32,128),(1,128,32,128)):
    host=torch.randn(shape,generator=torch.Generator().manual_seed(213))
    device=host.to('npu')
    result=torch.zeros_like(device)
    torch.npu.synchronize()
    row={'shape':shape,'input_format':torch_npu.get_npu_format(device),
         'result_format':torch_npu.get_npu_format(result),
         'dtype_preserved':result.dtype==host.dtype,
         'shape_preserved':result.shape==host.shape,
         'input_unchanged':torch.equal(device.cpu(),host),
         'all_zeros_bitwise':torch.equal(result.cpu().view(torch.uint8),torch.zeros_like(host).view(torch.uint8)),
         'custom_launches':0}
    assert all(row[n] for n in ('dtype_preserved','shape_preserved','input_unchanged','all_zeros_bitwise'))
    rows.append(row);print(json.dumps(row),flush=True)
(args.out/'receipt.json').write_text(json.dumps({'stage':'Torch NPU allocation diagnostic',
    'custom_launches':0,'rows':rows,'torch_npu_reported_git_version':torch_npu.version.git_version},indent=2)+'\n')
