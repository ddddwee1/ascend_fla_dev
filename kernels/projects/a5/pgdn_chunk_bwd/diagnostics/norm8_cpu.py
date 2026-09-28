"""CPU-only diagnostic; no changes to production or frozen references."""
import ctypes
import hashlib
import json
import math
import torch

torch.set_num_threads(1)
libm = ctypes.CDLL('libm.so.6')
sqrtf = libm.sqrtf
sqrtf.argtypes = [ctypes.c_float]
sqrtf.restype = ctypes.c_float
generator = torch.Generator().manual_seed(195082)
x = torch.randn((2048, 128), generator=generator)
eps = torch.tensor(1e-12, dtype=torch.float32)


def digest(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def compare(value, literal):
    different = value.view(torch.int32) != literal.view(torch.int32)
    indexes = different.nonzero().flatten()[:8].tolist()
    return dict(bit_mismatch=int(different.sum()),
                branch_mismatch=int(((value >= eps) != (literal >= eps)).sum()),
                first_differences=[dict(row=i, actual=float(value[i]), literal=float(literal[i])) for i in indexes])


rows = []
for scale, factor in [('ordinary', None), ('dense_at_clamp', 1.),
                      ('dense_below', .99999), ('dense_above', 1.00001)]:
    raw = x if factor is None else (x / x.norm(dim=-1, keepdim=True) * eps * factor).contiguous()
    literal = raw.norm(dim=-1)
    acc = torch.zeros(raw.shape[0], 8)
    for i in range(0, 128, 8):
        value = raw[:, i:i+8]
        acc = acc + value.square()
    total = acc[:, 0]
    for i in range(1, 8):
        total = total + acc[:, i]
    vector = total.sqrt()
    scalar = torch.tensor([sqrtf(value) for value in total.tolist()], dtype=torch.float32)
    double_rounded = torch.tensor([math.sqrt(value) for value in total.tolist()], dtype=torch.float32)
    rows.append(dict(scale=scale, rows=len(raw),
                     hashes={name: digest(value) for name, value in dict(raw=raw, eight_lane_sum=total,
                             literal=literal, vector_sqrt=vector, scalar_sqrtf=scalar,
                             double_sqrt_rounded=double_rounded).items()},
                     vector_sqrt=compare(vector, literal), scalar_sqrtf=compare(scalar, literal),
                     double_sqrt_rounded=compare(double_rounded, literal)))
print(json.dumps(dict(torch=torch.__version__, git=torch.version.git_version,
                     capability=torch.backends.cpu.get_cpu_capability(), cpu_only=True,
                     mkl_available=torch.backends.mkl.is_available(), cases=rows)))
