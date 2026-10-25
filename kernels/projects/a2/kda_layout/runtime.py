"""Metadata, allocation and native launches for the A2 KDA layout unit."""
from __future__ import annotations

import functools
import importlib.util
import math
from pathlib import Path

import torch

_DTYPES = {torch.bfloat16: 'bf16', torch.float32: 'f32'}


@functools.lru_cache(maxsize=1)
def kernels():
    path = Path(__file__).parent / 'kernels' / 'move.py'
    spec = importlib.util.spec_from_file_location('_afla_kda_layout_a2_move', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.entries()


@functools.lru_cache(maxsize=None)
def prepare(device='a2', block_dim=1):
    """Compile all six entries before any custom operation resolves vendors."""
    from ascend_fla.runtime.compile import compile_kernel

    if device != 'a2' or block_dim not in (1, 2):
        raise ValueError('KDA layout requires a2 and block_dim in (1,2); other dimensions are unqualified')
    return {name: compile_kernel(fn, device=device, block_dim=block_dim)
            for name, fn in kernels().items()}



def move(source, shape, strides, *, dtype=None, device='a2', block_dim=1,
         multiply=False, factor=1.0):
    """Gather a five-dimensional logical view into a new contiguous tensor.

    ``strides`` are source element strides, relative to source.data_ptr(). The
    last logical dimension is a multiple of 64. Storage views never copy data.
    """
    dtype = source.dtype if dtype is None else dtype
    shape, strides = tuple(shape), tuple(strides)
    if source.dtype not in _DTYPES or dtype not in _DTYPES:
        raise ValueError('KDA layout supports BF16 and FP32 only')
    if len(shape) != 5 or len(strides) != 5 or min(shape) <= 0 or min(strides) < 0:
        raise ValueError('KDA layout requires five positive dimensions and nonnegative strides')
    if shape[-1] % 64:
        raise ValueError('KDA layout last dimension must be divisible by 64')
    if multiply and source.dtype != torch.float32:
        raise ValueError('the gate multiplier requires FP32 source')
    count = math.prod(shape)
    span = 1 + sum((d - 1) * s for d, s in zip(shape, strides))
    if max(count, span, *shape, *strides) >= 2**31:
        raise ValueError('KDA layout address dimensions must fit signed int32')
    available = source.untyped_storage().nbytes() // source.element_size() - source.storage_offset()
    if span > available:
        raise ValueError('KDA layout source view exceeds its storage')
    # ACL receives contiguous metadata for the owned storage span; the kernel
    # receives the original logical strides separately and performs the gather.
    if source.is_contiguous():
        storage = source.view(1, source.numel())
        if span > source.numel():
            raise ValueError('KDA layout source mapping exceeds its tensor')
    else:
        storage = source.as_strided((1, span), (span, 1), source.storage_offset())
    destination = torch.empty(shape, dtype=dtype, device=source.device)
    scalars = dict(Storage=storage.numel(), N=count,
                   **{f'D{i}': shape[i] for i in range(1, 5)},
                   **{f'S{i}': strides[i] for i in range(5)},
                   multiply=int(multiply), factor=float(factor))
    key = f'{_DTYPES[source.dtype]}_{_DTYPES[dtype]}'
    prepare(device, block_dim)[key](
        {'source': storage}, scalars, {'destination': destination.view(1, count)})
    return destination


def cast(source, dtype, *, device='a2', block_dim=1):
    """Preserve logical order while converting dtype/packing a strided tensor."""
    if source.dtype == dtype and source.is_contiguous():
        return source
    shape, strides = tuple(source.shape), tuple(source.stride())
    if source.is_contiguous():
        shape, strides = (1, 1, 1, 1, source.numel()), (0, 0, 0, 0, 1)
    else:
        if len(shape) > 5:
            raise ValueError('KDA layout accepts at most five dimensions')
        shape = (1,) * (5 - len(shape)) + shape
        strides = (0,) * (5 - len(strides)) + strides
    result = move(source, shape, strides, dtype=dtype, device=device, block_dim=block_dim)
    return result.view(source.shape)


def zeros(shape, dtype, target, *, device='a2', block_dim=1):
    """Allocate uninitialized storage, then fill every element on the device."""
    shape = tuple(shape)
    count = math.prod(shape)
    if dtype not in _DTYPES or min(shape) <= 0 or count % 64 or count >= 2**31:
        raise ValueError('KDA zero fill requires positive BF16/FP32 shape with numel divisible by64')
    destination = torch.empty(shape, dtype=dtype, device=target)
    prepare(device, block_dim)[f'zero_{_DTYPES[dtype]}'](
        {}, {'N': count}, {'destination': destination.view(1, count)})
    return destination
