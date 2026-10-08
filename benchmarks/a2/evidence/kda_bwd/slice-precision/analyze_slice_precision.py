"""CPU-only localization of saved native gradient error; no new acceptance budget."""
from pathlib import Path
import hashlib,json,math
import torch
from benchmarks.verify_decode import _a2_unit
from benchmarks.probe_bwd_span import a2_gradient_metrics
root=Path(__file__).resolve().parent
_,ref=_a2_unit('kda_bwd_stable')
outputs={'dq':'finalize_reduce.dq','dk':'finalize_reduce.dk','dv':'inverse_epilogue.dv',
         'dg':'finalize_post.dg','dbeta':'finalize_post.dbeta','dh0':'scan.dh0'}
rows=[]
for span in (24,28,32,64,96,128):
    version=1 if span<64 else 2
    path=root/f'native/precision-v{version}-bd1/receipts/range_uniform_t128_seed0_span{span}.pt'
    data=torch.load(path,map_location='cpu',weights_only=True)
    inputs=data['inputs']; b,t,h,d=inputs['q'].shape;hv=inputs['v'].shape[2]
    gc=(data['raw_forward']['g_cumsum']*(1/math.log(2))).permute(0,2,3,1,4).contiguous().view(b,t,hv,d)
    replay=ref.reference_stages({**inputs,'saved':{**data['saved'],'g_cumsum':gc}})
    result={}
    for name,stage in outputs.items():
        golden=data['golden'][name]
        groups={}
        for kind,value in {'native':data['native_outputs'][name],
                           'cpu_fp32_gc_diagnostic':replay[stage].reshape_as(golden)}.items():
            slices=[]
            axis=1 if name=='dh0' else 2
            for head in range(golden.shape[axis]):
                selector=[slice(None)]*golden.ndim;selector[axis]=head
                for chunk in ([None] if name=='dh0' else range(t//64)):
                    current=selector.copy()
                    if chunk is not None:current[1]=slice(chunk*64,(chunk+1)*64)
                    metric=a2_gradient_metrics(value[tuple(current)],golden[tuple(current)],{'max_relative_l2':.05})
                    # .05 is a localization marker, not a newly imposed per-slice acceptance contract.
                    slices.append({'head':head,'chunk':chunk,**metric})
            worst=max(slices,key=lambda x:x['relative_l2'])
            groups[kind]={'slices':slices,'worst_slice':worst,
                          'slices_over_point05':sum(not x['passed'] for x in slices)}
        result[name]=groups
    row={'span':span,'input_tensor_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'hardware_execution':'previously recorded full precision chain, no new launch',
         'per_slice_point05_is_diagnostic_marker_only':True,'gradients':result}
    rows.append(row)
    print(json.dumps({'span':span,'summary':{n:{k:{'worst':v['worst_slice'],
           'over_point05':v['slices_over_point05']} for k,v in g.items()} for n,g in result.items()}}),flush=True)
(root/'slice-precision-analysis.json').write_text(json.dumps({'rows':rows},indent=2,allow_nan=False)+'\n')
