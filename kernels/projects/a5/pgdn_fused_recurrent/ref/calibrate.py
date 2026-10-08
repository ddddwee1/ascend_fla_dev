"""Before-kernel A/B calibration with the pinned literal PGDN naive A."""
import argparse,json,platform
from pathlib import Path
import torch
from .reference import cases,make_inputs,reference,metric,digest,validate_inputs,validate_reference
from .oracle import oracle,PIN,SHA256,DTYPE_LINES


def calibrate():
    torch.set_num_threads(1);records=[]
    for case in cases():
        inputs=make_inputs(case);validate_inputs(inputs)
        before={n:digest(x) for n,x in inputs.items() if x is not None}
        a=oracle(inputs);b=reference(inputs);validate_reference(inputs,b)
        assert before=={n:digest(x) for n,x in inputs.items() if x is not None}
        assert a['final_state'].data_ptr()!=getattr(inputs['initial_state'],'data_ptr',lambda:0)()
        assert a['final_A_state'].data_ptr()!=getattr(inputs['initial_A_state'],'data_ptr',lambda:0)()
        ms={n:metric(b[n],a[n]) for n in a}
        records.append(dict(case=case,input_sha256=before,A_vs_B=ms,
                            float32_reference_dtype=all(x.dtype==torch.float32 for x in a.values())))
    fp64=[]
    rng=torch.Generator().manual_seed(948864)
    for b,s,h,hv,k,v in [(1,1,1,1,3,5),(2,3,2,4,7,5),(1,4,2,8,5,3)]:
        inp={n:torch.randn(b,s,hv if n=='v' else h,v if n=='v' else k,generator=rng,dtype=torch.float64)*.1 for n in ('q','k','v')}
        inp.update(g_atk=-torch.rand(b,s,h,generator=rng,dtype=torch.float64),beta_atk=torch.rand(b,s,h,generator=rng,dtype=torch.float64),initial_A_state=torch.rand(b,h,k,generator=rng,dtype=torch.float64),g=-torch.rand(b,s,hv,generator=rng,dtype=torch.float64),beta=torch.rand(b,s,hv,generator=rng,dtype=torch.float64),initial_state=torch.randn(b,hv,k,v,generator=rng,dtype=torch.float64)*.1)
        a=oracle(inp,fp64=True);ref=reference(inp,fp64=True)
        fp64.append(dict(shape=[b,s,h,hv,k,v],metrics={n:metric(ref[n],a[n]) for n in a}))
    max32=max(m['relative_l2'] for r in records for m in r['A_vs_B'].values())
    max64=max(m['relative_l2'] for r in fp64 for m in r['metrics'].values())
    return dict(kind='CPU calibration, not device acceptance',oracle_identity='literal PGDN naive A', fp64_lift_source_lines=DTYPE_LINES,
                fla_pin=PIN,oracle_sha256=SHA256,python=platform.python_version(),torch=torch.__version__,
                cases=records,fp64_lift_cases=fp64,max_fp32_A_B_relative_l2=max32,max_fp64_relative_l2=max64,
                calibrated=max32<=1e-5 and max64<=1e-12)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=calibrate();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','fp64_lift_cases')}))
    if not result['calibrated']:raise SystemExit(1)
