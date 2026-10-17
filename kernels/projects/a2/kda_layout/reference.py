"""Independent CPU references for strided layout and exact dtype boundaries."""
import math

import torch

DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}


def make_inputs(case):
    p = dict(case["parameters"])
    gen = torch.Generator().manual_seed(case["seed"])
    shape, strides = p["shape"], p["strides"]
    span = 1 + sum((d - 1) * s for d, s in zip(shape, strides))
    x = torch.randn(span, generator=gen).to(DTYPES[p["source"]])
    return {"source": x, "parameters": p}


def reference(inputs):
    x, p = inputs["source"], inputs["parameters"]
    dtype = DTYPES[p["destination"]]
    if p.get("zero", False):
        y = torch.zeros(p["shape"], dtype=dtype)
    else:
        y = x.as_strided(p["shape"], p["strides"], x.storage_offset())
        if p.get("multiply", False):
            y = y.float() * float(p.get("factor", 1.0 / math.log(2.0)))
        y = y.to(dtype).contiguous()
    return {"destination": y.reshape(1, -1)}
