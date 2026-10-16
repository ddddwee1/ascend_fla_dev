"""Standalone A2 layout leaf with an independent CPU reference."""
import functools
import importlib.util
import math
from pathlib import Path

import torch

from reference import DTYPES, make_inputs, reference


@functools.lru_cache(maxsize=1)
def _runtime():
    path = Path(__file__).with_name("runtime.py")
    spec = importlib.util.spec_from_file_location("_a2_layout_unit_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch_description(inputs):
    x, p = inputs["source"], inputs["parameters"]
    shape, strides = p["shape"], p["strides"]
    key = ("zero_" + p["destination"] if p.get("zero", False)
           else p["source"] + "_" + p["destination"])
    scalars = dict(N=math.prod(shape))
    if not p.get("zero", False):
        scalars = dict(Storage=x.numel(), N=math.prod(shape),
                       **{f"D{i}": shape[i] for i in range(1, 5)},
                       **{f"S{i}": strides[i] for i in range(5)},
                       multiply=int(p.get("multiply", False)),
                       factor=float(p.get("factor", 1.0 / math.log(2.0))))
    return key, scalars


def kernel_for(case):
    p = case["parameters"]
    key = ("zero_" + p["destination"] if p.get("zero", False)
           else p["source"] + "_" + p["destination"])
    return _runtime().kernels()[key]


def execute(inputs, options):
    from _unit_runner import launch_kernel

    if options["device"] != "a2" or options["backend"] != "cce" or options["block_dim"] not in (1, 2):
        raise ValueError("A2 layout candidate requires a2/CCE with bd1 or bd2")
    key, scalars = launch_description(inputs)
    out = torch.full((1, scalars["N"]), float("nan"),
                     dtype=DTYPES[inputs["parameters"]["destination"]])
    args = (() if key.startswith("zero_") else (inputs["source"].reshape(1, -1),))
    actual = launch_kernel(_runtime().kernels()[key], args + (out, *scalars.values()), options)
    return {"destination": actual}


def validate_reference(inputs, expected, case):
    actual = expected["destination"]
    assert actual.device.type == "cpu"
    assert actual.dtype == DTYPES[case["parameters"]["destination"]]
    assert actual.shape == (1, math.prod(case["parameters"]["shape"]))
