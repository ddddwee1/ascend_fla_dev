#!/usr/bin/env python3
# coding: utf-8
"""Time fla's naive_chunk_gated_delta_rule on torch_npu as the GDR performance baseline.

This is the second of the two oracles: a composed native-op implementation on NPU,
which doubles as the perf baseline. It is a NAIVE reference (its triangular inverse is
a 63-iteration Python loop), so beating it is necessary, not sufficient.

Usage: ASCEND_RT_VISIBLE_DEVICES=<d> python fla_npu_baseline.py <warmup> <repeats> B,T,H,K,V ...
"""
import os, sys, time, statistics, torch, torch_npu
from fla.ops.gated_delta_rule.naive import (naive_chunk_gated_delta_rule,
                                            naive_recurrent_gated_delta_rule)
torch.manual_seed(0)
torch.npu.set_device(0)
dev = "npu:0"

def sync(): torch.npu.synchronize()

def mk(B,T,H,K,V):
    q=torch.randn(B,T,H,K); k=torch.randn(B,T,H,K); v=torch.randn(B,T,H,V)
    beta=torch.rand(B,T,H); g=-torch.rand(B,T,H)*4.0
    return q,k,v,beta,g

# ---- correctness: naive_chunk on NPU vs naive_recurrent on CPU (oracle #2 check)
B,T,H,K,V = 1,192,2,128,128
q,k,v,beta,g = mk(B,T,H,K,V); scale=1/(K**0.5)
o_cpu,_ = naive_recurrent_gated_delta_rule(q,k,v,beta,g,scale=scale)
o_npu = naive_chunk_gated_delta_rule(q.to(dev),k.to(dev),v.to(dev),g.to(dev),beta.to(dev),scale=scale)
o_npu = (o_npu[0] if isinstance(o_npu,(tuple,list)) else o_npu).cpu().float()
d=(o_npu-o_cpu).abs(); rel=(d.pow(2).sum().sqrt()/o_cpu.pow(2).sum().sqrt()).item()
print("CORRECTNESS naive_chunk@NPU vs naive_recurrent@CPU  B%dT%dH%d: max_abs=%.3e rel_l2=%.3e peak=%.3e"
      %(B,T,H,d.max().item(),rel,o_cpu.abs().max().item()))

# ---- timing
WARM,REP = int(sys.argv[1]), int(sys.argv[2])
for (B,T,H,K,V) in [tuple(int(x) for x in s.split(",")) for s in sys.argv[3:]]:
    q,k,v,beta,g = mk(B,T,H,K,V); scale=1/(K**0.5)
    qd,kd,vd,bd,gd = q.to(dev),k.to(dev),v.to(dev),beta.to(dev),g.to(dev)
    sync()
    try:
        for _ in range(WARM): naive_chunk_gated_delta_rule(qd,kd,vd,gd,bd,scale=scale)
        sync()
        ts=[]
        for _ in range(REP):
            t0=time.perf_counter(); naive_chunk_gated_delta_rule(qd,kd,vd,gd,bd,scale=scale); sync()
            ts.append((time.perf_counter()-t0)*1e6)
        print("TIMING B=%d T=%d H=%d K=V=%d  median=%.1f us  samples=%s"
              %(B,T,H,K,statistics.median(ts),", ".join("%.0f"%x for x in ts)))
    except Exception as e:
        print("TIMING B=%d T=%d H=%d K=V=%d  FAILED: %s: %s"%(B,T,H,K,type(e).__name__,str(e)[:200]))
