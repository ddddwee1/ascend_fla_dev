"""Emit/compile or run the bounded ATK first-divergence probe."""
import argparse,json
from pathlib import Path
import torch
from ascend_fla.runtime.compile import compile_kernel
from .kernels.decay_probe import pk06_decay_probe
from .ref.reference import metric


def main():
    p=argparse.ArgumentParser();p.add_argument('--block-dim',type=int,choices=(1,),default=1);p.add_argument('--compile',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    op=compile_kernel(pk06_decay_probe,device='a5',block_dim=1,backend='cce')
    if a.compile:return
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    gates=[-80.,-87.,-87.3,-87.34,-88.,-90.,-95.,-100.,-103.,-104.,-1000.]
    g=torch.tensor((gates*6)[:64],dtype=torch.float32).reshape(1,64)
    state=torch.full((1,64),1e30,dtype=torch.float32)
    gd,ad=g.to('npu'),state.to('npu');yd=torch.empty((4,64),dtype=torch.float32,device='npu')
    op(dict(g=gd,a=ad),{},dict(y=yd));torch.npu.synchronize();y=yd.cpu()
    expected=g.exp()*state;rows=[]
    for i,gate in enumerate(gates):
        rows.append(dict(g=gate,literal_exp=float(g.exp()[0,i]),actual_exp=float(y[0,i]),literal_product=float(expected[0,i]),
                         actual_product=float(y[1,i]),scaled_count=float(y[2,i]),scaled_product=float(y[3,i]),
                         scaled_relative_l2=metric(y[3,i:i+1],expected[0,i:i+1])['relative_l2']))
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'summary.json').write_text(json.dumps(dict(stage='native_decay_probe',rows=rows),indent=2)+'\n')
    torch.save(dict(g=g,state=state,actual=y,expected=expected),a.output/'returned.pt')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
