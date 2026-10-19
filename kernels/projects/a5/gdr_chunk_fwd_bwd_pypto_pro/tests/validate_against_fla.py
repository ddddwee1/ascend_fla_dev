#!/usr/bin/env python3
# coding: utf-8
"""Validate the gated_delta_rule goldens against fla's naive reference (fp32, CPU).

The goldens were transcribed from our own forward oracle, so without this check every
test would agree with a misreading of GDR's semantics. fla's naive.py is the semantic
authority. Run on CPU; needs `fla` installed (0.5.2 verified). Reports numbers, not OK.
"""
import importlib.util, os, sys, torch

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fg", os.path.join(HERE, "gated_delta_rule_fwd_golden.py"))
fg = importlib.util.module_from_spec(spec); spec.loader.exec_module(fg)
from fla.ops.gated_delta_rule.naive import (naive_recurrent_gated_delta_rule,
                                            naive_chunk_gated_delta_rule)

def stats(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    rel_l2 = (d.pow(2).sum().sqrt() / b.pow(2).sum().sqrt().clamp_min(1e-30)).item()
    return d.max().item(), rel_l2, b.abs().max().item()

print("%-26s %-18s %12s %12s %12s" % ("case", "comparison", "max_abs", "rel_l2", "ref_peak"))
for (B,T,H,K,V) in [(1,128,2,64,64),(1,192,2,128,128),(1,256,4,128,128),(2,128,3,128,96)]:
    q = torch.randn(B,T,H,K); k = torch.randn(B,T,H,K); v = torch.randn(B,T,H,V)
    beta = torch.rand(B,T,H)                 # PRE-ACTIVATED, same tensor to both
    g = -torch.rand(B,T,H)*4.0               # log-space decay, g<0
    scale = 1.0/(K**0.5)
    tag = "B%dT%dH%dK%dV%d" % (B,T,H,K,V)

    o_ours, h_ours = fg._recurrent(q,k,v,beta,g,scale)
    o_fla,  h_fla  = naive_recurrent_gated_delta_rule(q,k,v,beta,g,scale=scale,output_final_state=True)
    m,r,p = stats(o_ours,o_fla); print("%-26s %-18s %12.3e %12.3e %12.4e"%(tag,"recurrent o",m,r,p))
    m,r,p = stats(h_ours,h_fla); print("%-26s %-18s %12.3e %12.3e %12.4e"%(tag,"recurrent state",m,r,p))

    oc_ours = fg._chunk(q,k,v,g,beta,scale)
    oc_ours = oc_ours[0] if isinstance(oc_ours,(tuple,list)) else oc_ours
    oc_fla = naive_chunk_gated_delta_rule(q,k,v,g,beta,scale=scale)
    oc_fla = oc_fla[0] if isinstance(oc_fla,(tuple,list)) else oc_fla
    m,r,p = stats(oc_ours,oc_fla); print("%-26s %-18s %12.3e %12.3e %12.4e"%(tag,"chunk o",m,r,p))
    m,r,p = stats(oc_ours,o_fla);  print("%-26s %-18s %12.3e %12.3e %12.4e"%(tag,"ourchunk vs flarec",m,r,p))
    print()
