"""Focused diagnostic after a complete native workload; not a production entry."""
from ascriptor.a5 import *


@vf()
def inspect_decay(g: Tensor, a: Tensor, y: Tensor):
    gate = Reg(DT.float)
    state = Reg(DT.float)
    decay = Reg(DT.float)
    product = Reg(DT.float)
    count = Reg(DT.int32)
    scaled = Reg(DT.float)
    cfg = CastConfig(round_mode=RoundMode.TO_EVEN)
    gate <<= g[0:1, 0:64]
    state <<= a[0:1, 0:64]
    decay <<= gate.exp()
    y[0:1, 0:64] <<= decay
    product <<= state * decay
    y[1:2, 0:64] <<= product
    scaled <<= gate.vmins(-87.0)
    scaled <<= scaled + 103.27892990343185
    scaled <<= scaled.exp()
    count <<= scaled.astype(DT.int32, cfg)
    scaled <<= count.astype(DT.float, cfg)
    y[2:3, 0:64] <<= scaled
    scaled <<= scaled * 2.5849394142282115e-26
    state <<= state * 5.421010862427522e-20
    product <<= state * scaled
    y[3:4, 0:64] <<= product
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel(mode='vec', block_dim=1)
def pk06_decay_probe(g: GM[f32, (1,64)], a: GM[f32,(1,64)], y: GM[f32,(4,64)]):
    gu=Tensor(DT.float,[1,64],Position.UB)
    au=Tensor(DT.float,[1,64],Position.UB)
    yu=Tensor(DT.float,[4,64],Position.UB)
    with auto_sync():
        gu <<= g
        au <<= a
        inspect_decay(gu,au,yu)
        y <<= yu
    return y
