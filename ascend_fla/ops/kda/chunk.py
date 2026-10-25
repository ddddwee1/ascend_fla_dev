"""KDA 分块前向 —— 把 ascriptor ``a5.kda_fwd`` 的五个 kernel 接到 runtime 桥上。

五段与 fla 的对应关系::

    gate      g_raw 的 chunk 内 cumsum -> eg          ~ ops/utils/cumsum
    scores    Aqk 与严格下三角 strict                  ~ ops/common/chunk_o + chunk_scaled_dot_kkt
    inverse   strict 的 64x64 严格下三角求逆 -> Akk     ~ ops/utils/solve_tril
    wy        WY 表示 -> w, u, qg, kg                  ~ kda/wy_fast
    recurrent chunk 递推 -> o, final_state              ~ ops/common/chunk_delta_h

公开 ABI 与 fla ``fla.ops.kda.chunk_kda`` 一致（token-major），内部由自编译布局 kernel 转到
kernel 的 BHCLD 布局 —— 与 ascriptor 单元 ``composition.chunked()`` 的做法相同。

硬约束（不满足直接报错，不静默降级，见 AGENTS.md §7）：
``L=64``、``K=V=128``、``T % 64 == 0``、``HV % H == 0``、q/k/v 为 bf16、g/beta/state 为 fp32。
"""
from __future__ import annotations

import functools
import importlib.util
import math
import os
import pathlib
import sys
from typing import Any

import torch

from ascend_fla.platform import capability, require_qualified, resolve_soc

__all__ = ["chunk_kda_fwd", "kda_fwd_kernels"]

L_PER_CHUNK = 64
HEAD_DIM = 128
VALUE_DIM = 128


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """FLA's additive squared epsilon, evaluated in FP32 before ABI rounding."""
    x = x.float()
    return x / torch.sqrt(x.square().sum(dim=-1, keepdim=True) + 1e-6)


# BF16 round-to-nearest bound, calibrated over 1,048,704 K=128 rows.
# See test_kda_domain_checks.py: observed maximum norm 1.0027123859343965.
_QK_NORM_TOL = 2 ** -8


def _validate_raw_inputs(q, k, g, beta, *, A_log=None, dt_bias=None,
                    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                    use_beta_sigmoid_in_kernel=False, qk_dtype=torch.bfloat16):
    """Shared metadata checks, completed before preparation can launch."""
    def raw_tensor(name, x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError(f"{name} must be a floating tensor for raw-input preparation")
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{name} raw input must be BF16 or FP32; FP16/FP64 are unsupported")
        if not x.is_contiguous():
            raise ValueError(f"{name} must be contiguous before raw-input preparation")

    if use_gate_in_kernel:
        if A_log is None or dt_bias is None:
            missing = ", ".join(n for n, x in (("A_log", A_log), ("dt_bias", dt_bias)) if x is None)
            raise ValueError(f"use_gate_in_kernel=True requires {missing}")
        for name, x in (("g", g), ("A_log", A_log), ("dt_bias", dt_bias)):
            raw_tensor(name, x)
        if g.dim() != 4:
            raise ValueError("g must have shape [B,T,HV,K]")
        hv, kd = g.shape[-2:]
        if A_log.shape != (hv,):
            raise ValueError(f"A_log must have shape [HV]=[{hv}], got {tuple(A_log.shape)}")
        if dt_bias.shape != (hv * kd,):
            raise ValueError(f"dt_bias must have shape [HV*K]=[{hv * kd}], got {tuple(dt_bias.shape)}")
        if A_log.device != g.device or dt_bias.device != g.device:
            raise ValueError("A_log and dt_bias must be on the same device as g")
    if use_qk_l2norm_in_kernel:
        for name, x in (("q", q), ("k", k)):
            raw_tensor(name, x)
        if q.dim() != 4 or k.shape != q.shape:
            raise ValueError("q and k must have the same shape [B,T,H,K]")
        if qk_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("normalized q/k output must be BF16 or FP32")
    if use_beta_sigmoid_in_kernel:
        raw_tensor("beta", beta)


def _prepare_inputs(q, k, g, beta, *, A_log=None, dt_bias=None,
                    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                    use_beta_sigmoid_in_kernel=False, qk_dtype=torch.bfloat16):
    """Preserved differentiable training graph pending BF-08.

    Production training uses native preparation autograd; this helper remains
    available for predecessor comparisons. Inference and decode use
    ``_prepare_kernel_inputs``. Disabled routes retain
    their original objects. This training arithmetic remains an audited exception.
    """
    _validate_raw_inputs(q, k, g, beta, A_log=A_log, dt_bias=dt_bias,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        use_gate_in_kernel=use_gate_in_kernel,
                        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
                        qk_dtype=qk_dtype)
    if use_gate_in_kernel:
        hv, kd = g.shape[-2:]
        g = -A_log.float().exp().view(hv, 1) * torch.nn.functional.softplus(
            g.float() + dt_bias.float().view(hv, kd))
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q).to(qk_dtype), _l2norm(k).to(qk_dtype)
    if use_beta_sigmoid_in_kernel:
        beta = beta.float().sigmoid()
    return q, k, g, beta


@functools.lru_cache(maxsize=None)
def _prep_runtime(device="a5"):
    path = pathlib.Path(__file__).resolve().parents[3] / f"kernels/projects/{device}/kda_prep/runtime.py"
    spec = importlib.util.spec_from_file_location(f"_afla_kda_prep_{device}_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_kernel_inputs(q, k, g, beta, *, A_log=None, dt_bias=None,
                           use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                           use_beta_sigmoid_in_kernel=False, qk_dtype=torch.bfloat16,
                           device=None, block_dim=1, namespace="chunk", impl="stable"):
    """Validate, precompile dependencies, then prepare enabled inputs on NPU."""
    _validate_raw_inputs(q, k, g, beta, A_log=A_log, dt_bias=dt_bias,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        use_gate_in_kernel=use_gate_in_kernel,
                        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
                        qk_dtype=qk_dtype)
    if not (use_qk_l2norm_in_kernel or use_gate_in_kernel or use_beta_sigmoid_in_kernel):
        return q, k, g, beta
    device = resolve_soc(device)
    runtime = (_prep_runtime(device) if device == "a2" else _prep_runtime())
    sources = []
    if use_qk_l2norm_in_kernel:
        if q.shape[-1] != HEAD_DIM or min(q.shape) <= 0:
            raise ValueError("raw q/k require positive [B,T,H,128]")
        sources += [("q", q), ("k", k)]
    if use_gate_in_kernel:
        if g.shape[-1] != HEAD_DIM or min(g.shape) <= 0:
            raise ValueError("raw g requires positive [B,T,HV,128]")
        sources += [("g", g), ("A_log", A_log), ("dt_bias", dt_bias)]
    if use_beta_sigmoid_in_kernel:
        sources += [("beta", beta)]
    for name, source in sources:
        runtime._check_source(name, source)
    if namespace == "chunk":
        from .chunk_bwd import _compiled_chain as bwd_chain
        _compiled_chain(device, block_dim, impl)
        bwd_chain(device, block_dim, impl)
    elif namespace == "decode":
        from .fused_recurrent import _compiled_pair
        _compiled_pair(device, block_dim)
    else:
        raise ValueError("KDA prep namespace must be chunk or decode")
    options = dict(device=device, block_dim=block_dim, namespace=namespace)
    if use_gate_in_kernel:
        g = runtime.gate(g, A_log, dt_bias, **options)
    if use_qk_l2norm_in_kernel:
        q, k = runtime.norm(q, qk_dtype, **options), runtime.norm(k, qk_dtype, **options)
    if use_beta_sigmoid_in_kernel:
        beta = runtime.beta(beta, **options)
    return q, k, g, beta


def _check_input_domain(q, k, g, beta, *, use_qk_l2norm_in_kernel=False,
                        use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False):
    """Chunk-only heuristic: small raw vectors cannot be distinguished."""
    # Rank errors belong to the existing ABI guard in both execution paths.
    # Numerical heuristics on malformed tensors would obscure that diagnostic;
    # returning here never permits them through the lower-level shape checks.
    if q.dim() != 4 or k.dim() != 4 or g.dim() != 4 or beta.dim() != 3:
        return

    def reject(name, flag, requirement):
        raise ValueError(
            f"{name} must satisfy {requirement}; this looks like a raw, unactivated input. "
            f"Pass {flag}=True for raw inputs, or check_domain=False to opt out of "
            "this heuristic (ABI and gate-span checks still apply).")

    with torch.no_grad():
        if not use_gate_in_kernel and not bool((torch.isfinite(g) & (g <= 0)).all()):
            reject("g", "use_gate_in_kernel", "finite g <= 0")
        if not use_beta_sigmoid_in_kernel and not bool(
                (torch.isfinite(beta) & (beta > 0) & (beta < 1)).all()):
            reject("beta", "use_beta_sigmoid_in_kernel", "finite beta in (0,1)")
        if not use_qk_l2norm_in_kernel:
            for name, x in (("q", q), ("k", k)):
                norm = x.float().norm(dim=-1)
                if not bool((torch.isfinite(norm) & (norm <= 1 + _QK_NORM_TOL)).all()):
                    reject(name, "use_qk_l2norm_in_kernel", f"L2 norm <= {1 + _QK_NORM_TOL}")


# ascriptor kda_fwd contract.json 的 shapes.block_dim 声明；只有这几个值被 cases 覆盖过。
# 契约的 core_ownership 说明分区方式：gate 按向量核切 B*HV*C，scores/WY/inverse 按 cube
# 组切，融合尾部按 B*HV 头对切（两个 V=64 tile 必须留在同一组）。
#: a5 chunk 路径的 block_dim，从能力表读（a5 的别名）。真值住在 ``platform.CAPABILITIES``。
SUPPORTED_BLOCK_DIM = capability("a5")["supported_block_dim"]["chunk"]

#: Stable uses the local gate/scores/WY kernels for gate-span stability and the
#: repaired recurrent kernel for continuous Aqk slot rotation across heads.
#: Only inverse is shared. Upstream preserves the original five kernels, with
#: an odd-C repeated-head guard for its known Aqk handoff defect.
IMPLS = ("stable", "upstream")

_UPSTREAM_KERNELS = {
    "gate": ("gate", "kda_sub1_gate_kernel"),
    "scores": ("intra", "kda_sub2_score_kernel"),
    "inverse": ("triangular_inverse", "tril_inverse64_v2_strict_bf16_kernel"),
    "wy": ("wy", "kda_sub3_wy_kernel"),
    "recurrent": ("recurrent", "kda_sub45_fused_kernel"),
}
_STABLE_KERNELS = {
    "gate": ("gate", "kda_sub1_gate_stable_kernel"),
    "scores": ("intra", "kda_sub2_score_stable_kernel"),
    "wy": ("wy", "kda_sub3_wy_stable_kernel"),
    "recurrent": ("recurrent", "kda_sub45_aqk_repaired_kernel"),
}


def _kernels_root() -> pathlib.Path:
    """ascriptor kernels 仓里 ``projects/a5/kda_fwd`` 的位置。

    依次尝试 ``$ASCRIPTOR_KDA_FWD``、``$ASCRIPTOR_WORKSPACE/kernels/...``、同级
    ``../ascriptor/kernels/...``。我们只读不改（见 AGENTS.md §3）。
    """
    direct = os.environ.get("ASCRIPTOR_KDA_FWD")
    if direct:
        return pathlib.Path(direct)
    ws = os.environ.get("ASCRIPTOR_WORKSPACE")
    candidates = []
    if ws:
        candidates.append(pathlib.Path(ws) / "kernels/projects/a5/kda_fwd")
    repo = pathlib.Path(__file__).resolve().parents[3]
    candidates += [
        repo.parent / "ascriptor/kernels/projects/a5/kda_fwd",
        repo / "src/kernels/projects/a5/kda_fwd",
    ]
    for c in candidates:
        if (c / "kernels").is_dir():
            return c
    raise FileNotFoundError(
        "找不到 ascriptor 的 kda_fwd 单元；设 $ASCRIPTOR_KDA_FWD 指向它。已试："
        + ", ".join(str(c) for c in candidates)
    )


def _stable_kernels_root(device="a5") -> pathlib.Path:
    """本仓自有单元 ``kernels/projects/a5/kda_fwd_stable`` 的位置。"""
    repo = pathlib.Path(__file__).resolve().parents[3]
    root = repo / f"kernels/projects/{device}/kda_fwd_stable"
    if not (root / "kernels").is_dir():
        raise FileNotFoundError(f"找不到本仓的 kda_fwd_stable 单元（试了 {root}）")
    return root


def _load_kernel(tag: str, pkg_dir: pathlib.Path, module_name: str, fn_name: str) -> Any:
    """按文件加载单个 kernel 定义。

    逐模块按文件加载，**不** import 单元的 ``kernels`` 包 —— 那里的 ``composition.py``
    依赖单元本地的 ``_unit_runner``，会把 harness 拖进来。
    """
    path = pkg_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_afla_kda_{tag}_{module_name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


@functools.lru_cache(maxsize=None)
def kda_fwd_kernels(impl: str = "stable", device: str = "a5") -> dict[str, Any]:
    """载入某一套实现的五个 kernel 定义。

    Stable selects local gate/scores/WY/recurrent; inverse stays upstream.
    """
    if impl not in IMPLS:
        raise ValueError(f"impl 只能是 {IMPLS}，收到 {impl!r}")
    if device == "a2":
        if impl != "stable":
            raise ValueError("A2 KDA declares only impl='stable'")
        directory = _stable_kernels_root(device) / "kernels"
        entries = {
            "gate": ("gate", "kda_sub1_gate_a2_kernel"),
            "scores": ("intra", "kda_sub2_score_a2_kernel"),
            "inverse": ("triangular_inverse", "tril_inverse64_a2_kernel"),
            "wy": ("wy", "kda_sub3_wy_a2_kernel"),
            "recurrent": ("recurrent", "kda_sub45_a2_kernel"),
        }
        return {stage: _load_kernel("a2_st", directory, module, name)
                for stage, (module, name) in entries.items()}
    up_root = _kernels_root()
    if str(up_root) not in sys.path:
        sys.path.insert(0, str(up_root))

    out: dict[str, Any] = {}
    for stage, (module_name, fn_name) in _UPSTREAM_KERNELS.items():
        out[stage] = _load_kernel("up", up_root / "kernels", module_name, fn_name)
    if impl == "stable":
        st_dir = _stable_kernels_root() / "kernels"
        for stage, (module_name, fn_name) in _STABLE_KERNELS.items():
            out[stage] = _load_kernel("st", st_dir, module_name, fn_name)
    return out


@functools.lru_cache(maxsize=None)
def _compiled_chain(device: str, block_dim: int, impl: str = "stable") -> dict[str, Any]:
    """(device, block_dim) → 已编译的 5 个 kernel。

    缓存到这一层是因为热路径不该每次前向都去查 5 次编译缓存；``compile_kernel``
    自己也有进程内缓存，但查它要算签名（见 runtime/compile.py 的 ``_sig_memo``）。
    """
    from ...runtime.compile import compile_kernel

    (_layout_runtime(device) if device == "a2" else _layout_runtime()).prepare(device, block_dim)
    (_prep_runtime(device) if device == "a2" else _prep_runtime()).prepare(device, block_dim, "chunk")
    # A later training call in the same process cannot register new vendors
    # after an earlier inference launch has initialized CANN's operator lookup.
    (_prep_runtime(device) if device == "a2" else _prep_runtime()).prepare_backward(device, block_dim)
    return {name: compile_kernel(fn, device=device, block_dim=block_dim)
            for name, fn in (kda_fwd_kernels(impl, device) if device == "a2" else kda_fwd_kernels(impl)).items()}


def _check(q, k, v, g, beta, initial_state, block_dim, impl="stable", *, device="a5") -> tuple[int, int, int, int]:
    """门控。返回 ``(B, H, HV, C)``。任何不满足都报错，绝不静默降级。"""
    if impl not in IMPLS:
        raise ValueError(f"impl must be one of {IMPLS}, got {impl!r}")
    if device == "a2":
        if impl != "stable":
            raise ValueError("A2 KDA declares only impl='stable'")
        if block_dim not in capability(device)["supported_block_dim"]["chunk"]:
            raise ValueError("A2 chunk block_dim must be one of (1, 2)")
    if block_dim not in SUPPORTED_BLOCK_DIM:
        raise ValueError(
            f"block_dim 只支持 {SUPPORTED_BLOCK_DIM}（ascriptor kda_fwd 契约声明的范围），"
            f"收到 {block_dim}；更大的值未经契约 case 覆盖，且超过物理核数会在硬件 "
            f"barrier 死锁，见 docs/matrix/gaps.json 的 block-dim-ceiling"
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"q/k/v 应为 4 维 [B,T,H,D]，收到 {q.shape} / {k.shape} / {v.shape}")
    b, t, h, kd = q.shape
    hv, vd = v.shape[2], v.shape[3]
    if kd != HEAD_DIM or vd != VALUE_DIM:
        raise ValueError(
            f"ascriptor a5.kda_fwd 定尺要求 K=V={HEAD_DIM}，收到 K={kd} V={vd}；"
            f"见 docs/matrix/gaps.json 的 fixed-kv-128"
        )
    if t % L_PER_CHUNK:
        raise ValueError(
            f"T 必须是 {L_PER_CHUNK} 的整数倍（无 tail 路径），收到 T={t}；"
            f"最近的合法值是 {t // L_PER_CHUNK * L_PER_CHUNK} 或 {(t // L_PER_CHUNK + 1) * L_PER_CHUNK}；"
            f"见 gaps.json 的 no-tail-path"
        )
    if hv % h:
        raise ValueError(f"HV({hv}) 必须是 H({h}) 的整数倍")
    if k.shape != q.shape:
        raise ValueError(f"k 的形状应与 q 相同，收到 {k.shape} vs {q.shape}")
    if g.shape != (b, t, hv, kd):
        raise ValueError(f"g 应为 [B,T,HV,K]={(b, t, hv, kd)}，收到 {tuple(g.shape)}")
    if beta.shape != (b, t, hv):
        raise ValueError(f"beta 应为 [B,T,HV]={(b, t, hv)}，收到 {tuple(beta.shape)}")
    for name, x, want in (("q", q, torch.bfloat16), ("k", k, torch.bfloat16), ("v", v, torch.bfloat16),
                          ("g", g, torch.float32), ("beta", beta, torch.float32)):
        if x.dtype != want:
            raise ValueError(f"{name} 的 dtype 应为 {want}（kda_fwd ABI），收到 {x.dtype}")
    if initial_state is not None:
        if initial_state.shape != (b, hv, HEAD_DIM, VALUE_DIM):
            raise ValueError(
                f"initial_state 应为 [B,HV,K,V]={(b, hv, HEAD_DIM, VALUE_DIM)}，"
                f"收到 {tuple(initial_state.shape)}"
            )
        if initial_state.dtype != torch.float32:
            raise ValueError(f"initial_state 的 dtype 应为 float32，收到 {initial_state.dtype}")
    for name, x in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta),
                    *((("initial_state", initial_state),) if initial_state is not None else ())):
        if x.device.type != "npu":
            raise ValueError(f"{name} 应在 NPU 上，收到 device={x.device}")
        # 必须连续：kernel 按连续 GM 布局读，而且在缺内置算子包的机器上我们既不能在
        # device 上做 contiguous()（要 d2d copy），也不能对跨步视图直接 D2H
        # （要 NPU 侧的 Slice，实测报 errno 561000 —— 见 chunk_bwd.py 里 g_last 那段）。
        # 所以这里报错而不是悄悄修正 —— 悄悄 .contiguous() 在那种机器上根本做不到。
        if not x.is_contiguous():
            raise ValueError(
                f"{name} 必须是连续张量，收到 stride={tuple(x.stride())} "
                f"shape={tuple(x.shape)}；先自己 .contiguous() 再传进来"
            )
    c = t // L_PER_CHUNK
    _check_recurrent_heads(b, hv, c, block_dim, impl)
    return b, h, hv, c


def _check_recurrent_heads(b: int, hv: int, c: int, block_dim: int, impl: str) -> None:
    """Reject the original recurrent's unsafe odd-C repeated-head ownership.

    A5K-01 confirmed C=1 and C=3 corruption on silicon. The original two-credit
    Aqk channel restarts slot rotation at each head: odd C reuses the last slot
    before its final reader retires. Even C and at most one head per core avoid
    that boundary. Stable now selects the qualified continuous-rotation repair.
    See docs/research/kda_aqk_handoff_repair.md for the source and measured scope.
    """
    if impl not in IMPLS:
        raise ValueError(f"impl must be one of {IMPLS}, got {impl!r}")
    if impl == "stable" or c % 2 == 0 or b * hv <= block_dim:
        return
    raise ValueError(
        f"impl='upstream' with odd C={c} and B*HV={b * hv} > block_dim={block_dim} "
        "can silently corrupt o (静默错误) in kda_sub45_fused_kernel. "
        "Use impl='stable' for the validated Aqk handoff repair, or use an even "
        "chunk count or B*HV <= block_dim. See docs/research/kda_aqk_handoff_repair.md."
    )


@functools.lru_cache(maxsize=None)
def _layout_runtime(device="a5"):
    """Load the owned layout unit once; warm dispatch is an O(1) cache lookup."""
    path = pathlib.Path(__file__).resolve().parents[3] / f"kernels/projects/{device}/kda_layout/runtime.py"
    spec = importlib.util.spec_from_file_location(f"_afla_kda_layout_{device}_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _to_bhcld(x: torch.Tensor, heads: int, *, on_cpu: bool = False,
              device=None, block_dim=1) -> torch.Tensor:
    """Token-major → head/chunk-major, entirely in the owned layout kernel."""
    if on_cpu:
        raise ValueError("CPU layout conversion is prohibited by D-PM-37")
    device = resolve_soc(device)
    b, t = x.shape[:2]
    c = t // L_PER_CHUNK
    if x.dim() == 3:
        shape = (b, heads, c, 1, L_PER_CHUNK)
        strides = (t*heads, 1, L_PER_CHUNK*heads, 0, heads)
        result = (_layout_runtime(device) if device == "a2" else _layout_runtime()).move(x, shape, strides, device=device, block_dim=block_dim)
        return result.view(b, heads, c, L_PER_CHUNK)
    if x.dim() != 4:
        raise ValueError(f"_to_bhcld requires rank3/4, got {tuple(x.shape)}")
    d = x.shape[-1]
    return (_layout_runtime(device) if device == "a2" else _layout_runtime()).move(
        x, (b, heads, c, L_PER_CHUNK, d),
        (t*heads*d, d, L_PER_CHUNK*heads*d, heads*d, 1),
        device=device, block_dim=block_dim)


def _from_bhcld(x: torch.Tensor, *, on_cpu: bool = False, dtype=None,
                device=None, block_dim=1, multiply=False, factor=1.0) -> torch.Tensor:
    """Head/chunk-major → token-major, optionally narrowing in the same launch."""
    if on_cpu:
        raise ValueError("CPU layout conversion is prohibited by D-PM-37")
    device = resolve_soc(device)
    b, hv, c, l, d = x.shape
    result = (_layout_runtime(device) if device == "a2" else _layout_runtime()).move(
        x, (b, c, l, hv, d), (hv*c*l*d, l*d, d, c*l*d, 1),
        dtype=dtype, device=device, block_dim=block_dim, multiply=multiply, factor=factor)
    return result.view(b, c*l, hv, d)


#: chunk 内门控跨度的上限，按 **实现 × 跑哪条链** 两维。
#:
#: **为什么要分前向/反向。** 两条链的约束不是同一回事，用一个数会说谎：
#:
#: * 前向的限制是**有限性** —— 越线就是 NaN。``upstream`` 的 gate 只写 ``eg = exp(gc)``，
#:   它在 ``-ln(FLT_MIN_NORMAL) ≈ 87.3`` 处下溢到 0，下游 ``k/eg`` 变 ``0×inf``；
#:   ``stable`` 按逐通道中点对称分解，实测跨度到 155.97 时 ``o`` 的相对 L2 仍稳定在
#:   2.85e-03~3.19e-03（**完全不退化**）。
#: * 反向的限制是**精度** —— 它先于有限性到来。``stable`` 的反向到 169.76 都还是有限值，
#:   但对 fp32 递推参考的相对 L2 随跨度单调上升，``dq`` 在 130 处越过契约预算 0.05
#:   （实测 46→2.89e-02、94→4.55e-02、105→4.86e-02、110→4.99e-02、130→6.17e-02、
#:   169→6.88e-02，且 169 处 ``dg`` 崩到 6.49e-01）。让它"有限但超预算"地跑过去就是
#:   静默降级（AGENTS.md §7），所以反向的闸设在实测仍有余量的 105。
#:
#: **反向的 105 是怎么选的**：下界是 fla 初始化能产生的跨度上界 ——
#: ``exp(A_log) ≤ 16``、``dt ≤ 0.1``、63 步 → ``16 × 0.1 × 63 ≈ 100.8``（实测 HV=8 时
#: 8 个 seed 给 54.2~100.6，顶到上界）；上界是契约预算还成立的最深实测点（110 处 ``dq``
#: 只剩 0.2% 余量，所以不取它）。105 同时满足两头，余量 2.8%。
#:
#: 于是：纯推理（``chunk_kda_fwd``）可以用到 155，训练（``chunk_kda_fwd_with_caches`` /
#: ``chunk_kda``）到 105。
#: 数字全部是**实测点**，不是推算：见两个单元 contract.json 的 ``domain.gate_span``。
#: a5 的门控跨度上限，从能力表读（a5 的别名）。真值住在 ``platform.CAPABILITIES``：
#: upstream 两条链都受 ~87/88.7 那条硬线约束（前向下溢、反向上溢），取 80 留余量。
MAX_GATE_SPAN = capability("a5")["max_gate_span"]

#: 两条链的名字。``_check_gate_range`` 的 ``path`` 只接受这两个。
GATE_PATHS = ("forward", "backward")


def _gate_span(g: torch.Tensor, c: int, *, on_cpu: bool) -> float:
    """g 在 chunk 内累计后的最大跨度（``max(cumsum) - min(cumsum)``）。

    必须按 **chunk 内**算 —— kernel 的 cumsum 每 64 个 token 重置，跨 chunk 的累计不参与
    那个 ``exp(m−g)``。``on_cpu=True`` 时绕主机算（缺内置算子的机器上 cumsum 不可用）。
    """
    src = g.cpu() if on_cpu else g
    b, t, hv, kd = src.shape
    cum = src.float().view(b, c, L_PER_CHUNK, hv, kd).cumsum(dim=2)
    return (cum.amax(dim=2) - cum.amin(dim=2)).max().item()


def _check_gate_range(g: torch.Tensor, c: int, *, on_cpu: bool, impl: str,
                      path: str = "backward", device: str = "a5") -> None:
    """门控跨度超限就报错，绝不让 kernel 静默吐 NaN（AGENTS.md §7）。

    ⚠️ **这条限制比 contract 声明的输入域宽得多。** contract 的 ``input_generation`` 是
    ``g_raw ∈ [-0.03, 0]``（64 token 跨度 ≤1.92），而 fla 自己的 KDA 初始化
    （``A_log = log(U(1,16))``、``dt`` 最大 0.1）给出的跨度约 **94** —— 宽约 50 倍。
    ``upstream`` 的上限 80 撑不住它（会吐 NaN），``stable`` 的 160 可以。两个上限的由来
    与实测见 ``docs/matrix/gaps.json`` 的 ``gate-range-beyond-declared``。
    """
    if path not in GATE_PATHS:
        raise ValueError(f"path 只能是 {GATE_PATHS}，收到 {path!r}")
    limit = (capability(device)["max_gate_span"][impl][path]
             if device == "a2" else MAX_GATE_SPAN[impl][path])
    span = _gate_span(g, c, on_cpu=on_cpu)
    if span <= limit:
        return
    if device == "a2":
        raise ValueError(f"A2 {impl} {path} gate span {span:.1f} exceeds {limit}; "
                         "see CAPABILITIES[a2] and the qualified A2 unit contract")
    if impl == "upstream":
        why = "kernel 会在 fp32 下溢/上溢并输出 NaN"
        hint = (f"换 impl=\"stable\"（本仓的数值稳定实现，{path} 上限 "
                f"{MAX_GATE_SPAN['stable'][path]}）")
    elif path == "forward":
        why = "kernel 会在 fp32 下溢/上溢并输出 NaN"
        hint = "减小 g 的量级：KDA 层里即减小 exp(A_log) 或 dt"
    else:
        # 反向的闸守的是精度而不是有限性 —— 说清楚，否则用户会以为越线就是 NaN
        why = ("反向梯度仍是**有限值**（实测到 169.8 都有限），但精度会超出契约预算："
               "dq 在跨度 130 处达 6.17e-02（预算 0.05），dg 在 169 处达 6.49e-01（预算 0.25）")
        hint = ("减小 g 的量级（KDA 层里即减小 exp(A_log) 或 dt）；"
                f"只做推理不需要反向时用 chunk_kda_fwd，它的上限是 "
                f"{MAX_GATE_SPAN[impl]['forward']}")
    raise ValueError(
        f"chunk 内门控跨度 {span:.1f} 超过 impl={impl!r} 在 {path} 链上的上限 {limit}。"
        f"{why}。g 是 log 空间的 per-token 衰减增量；{hint}。"
        f"确知可接受时传 check_gate_range=False 跳过本检查。"
        f"详见 docs/matrix/gaps.json 的 gate-range-beyond-declared 与 gate-span-still-bounded"
    )


def _resolve_layout(layout_device: str) -> bool:
    """Deprecated auto/npu aliases both select kernel-side layout conversion."""
    if layout_device == "cpu":
        raise ValueError("layout_device='cpu' is prohibited by D-PM-37; use auto/npu for kernel-side conversion")
    if layout_device not in ("auto", "npu"):
        raise ValueError(f"layout_device must be auto/npu, got {layout_device!r}")
    return False


def _launch_a2(compiled, inputs, outputs, scalars):
    """Bind the selected A2 HostSpec using tensor metadata only.

    A2 entries declare token-major GM dimensions such as BT and HVK in
    addition to their explicit scalar arguments. No tensor is copied or cast.
    """
    ins = dict(zip(compiled.input_names, inputs, strict=True))
    outs = dict(zip(compiled.output_names, outputs, strict=True))
    bindings = dict(scalars)
    for entry, tensor in zip(compiled.spec.inputs + compiled.spec.outputs,
                             (*inputs, *outputs), strict=True):
        if len(entry["dims"]) != tensor.ndim:
            raise ValueError(f"A2 {entry['name']} rank differs from its kernel ABI")
        for symbol, extent in zip(entry["dims"], tensor.shape, strict=True):
            if isinstance(symbol, str):
                previous = bindings.setdefault(symbol, int(extent))
                if previous != extent:
                    raise ValueError(f"A2 ABI dimension {symbol}: {previous} != {extent}")
            elif symbol != extent:
                raise ValueError(f"A2 ABI dimension {symbol} != {extent}")
    compiled(ins, {name: bindings[name] for name in compiled.scalar_names}, outs)


def _run_chain_a2(q, k, v, g, beta, scale, initial_state, *, block_dim,
                  b, h, hv, c, impl):
    """Launch the existing five A2 entries on their token-major input ABI."""
    compiled = _compiled_chain("a2", block_dim, impl)
    t = c * L_PER_CHUNK
    bhc = (b, hv, c, L_PER_CHUNK, HEAD_DIM)
    sq = (b, hv, c, L_PER_CHUNK, L_PER_CHUNK)

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=q.device)

    q2, k2 = q.view(b*t, h*HEAD_DIM), k.view(b*t, h*HEAD_DIM)
    v2, g2, beta2 = v.view(b*t, hv*VALUE_DIM), g.view(b*t, hv*HEAD_DIM), beta.view(b*t, hv)
    state0 = initial_state
    if state0 is None:
        state0 = _layout_runtime("a2").zeros((b, hv, HEAD_DIM, VALUE_DIM),
                                            torch.float32, q.device, block_dim=block_dim)
    gc, eg = empty(bhc, torch.float32), empty(bhc, torch.float32)
    common = dict(B=b, H=h, HV=hv, C=c)
    _launch_a2(compiled["gate"], (g2,), (gc, eg), common)
    aqk, strict = empty(sq), empty(sq, torch.float32)
    _launch_a2(compiled["scores"], (q2, k2, gc, beta2), (aqk, strict),
               {**common, "scale": scale})
    akk = empty(sq)
    _launch_a2(compiled["inverse"], (strict,), (akk,), common)
    w, u, qg, kg = (empty(bhc) for _ in range(4))
    _launch_a2(compiled["wy"], (q2, k2, v2, beta2, akk, gc), (w, u, qg, kg), common)
    o = empty((b*t, hv*VALUE_DIM))
    final_state = empty((b, hv, HEAD_DIM, VALUE_DIM), torch.float32)
    _launch_a2(compiled["recurrent"], (q2, aqk, kg, w, u, eg, state0),
               (o, final_state), {**common, "scale": scale})
    return {"eg": eg, "g_cumsum": gc, "Aqk": aqk, "strict": strict, "Akk": akk,
            "w": w, "u": u, "qg": qg, "kg": kg, "o": o.view(b, t, hv, VALUE_DIM),
            "final_state": final_state, "initial_state": state0}


def _run_chain(q, k, v, g, beta, scale, initial_state, *, device, block_dim,
               on_cpu, b, h, hv, c, impl="stable") -> dict[str, torch.Tensor]:
    """跑完五个前向 kernel，返回**全部** BHCLD 中间量。

    ``chunk_kda_fwd`` 只要其中的 ``o`` 与 ``final_state``；``chunk_kda_fwd_with_caches``
    还要 ``eg`` / ``Aqk`` / ``Akk`` / ``w`` / ``u`` / ``qg`` / ``kg`` 去拼 kda_bwd 需要的
    九个前向检查点。抽成一处是为了两条路径**共用同一次 kernel 调用**，不会因为实现
    漂移而给出不同的中间量。
    """
    if device == "a2":
        return _run_chain_a2(q, k, v, g, beta, scale, initial_state,
                             block_dim=block_dim, b=b, h=h, hv=hv, c=c, impl=impl)
    dev = q.device
    compiled = _compiled_chain(device, block_dim, impl)

    def empty(shape, dtype):
        return torch.empty(*shape, dtype=dtype, device=dev)

    qc, kc = _to_bhcld(q, h, device=device, block_dim=block_dim), _to_bhcld(k, h, device=device, block_dim=block_dim)
    vc, gc = _to_bhcld(v, hv, device=device, block_dim=block_dim), _to_bhcld(g, hv, device=device, block_dim=block_dim)
    bc = _to_bhcld(beta, hv, device=device, block_dim=block_dim)
    state0 = initial_state if initial_state is not None else (_layout_runtime(device) if device == "a2" else _layout_runtime()).zeros(
        (b, hv, HEAD_DIM, VALUE_DIM), torch.float32, dev,
        device=device, block_dim=block_dim)

    bhc = (b, hv, c, L_PER_CHUNK, HEAD_DIM)
    sq = (b, hv, c, L_PER_CHUNK, L_PER_CHUNK)

    # 1) gate：chunk 内 cumsum。stable 版同时写出 log 空间的 g_cumsum ——
    # 下游 scores/wy 要靠它把除法变成减法（见 kernels/.../kda_fwd_stable/gate.py）
    eg = empty(bhc, torch.float32)
    gate_scalars = {"B": b, "HV": hv, "C": c,
                    "length_per_chunk": L_PER_CHUNK, "head_dim": HEAD_DIM}
    if impl == "stable":
        g_cumsum = empty(bhc, torch.float32)
        compiled["gate"]({"g_raw": gc}, gate_scalars,
                         {"g_cumsum": g_cumsum, "eg": eg})
    else:
        g_cumsum = None
        compiled["gate"]({"g_raw": gc}, gate_scalars, {"eg": eg})
    gate_in = {"g_cumsum": g_cumsum} if impl == "stable" else {"eg": eg}

    # 2) scores：Aqk 与严格下三角
    aqk, strict = empty(sq, torch.bfloat16), empty(sq, torch.float32)
    compiled["scores"](
        {"q": qc, "k": kc, "beta": bc, **gate_in},
        {"B": b, "H": h, "HV": hv, "C": c, "length_per_chunk": L_PER_CHUNK,
         "head_dim": HEAD_DIM, "scale": scale},
        {"Aqk": aqk, "strict": strict},
    )

    # 3) inverse：严格下三角求逆（kernel 的 H 位传 hv）
    akk = empty(sq, torch.bfloat16)
    compiled["inverse"]({"a": strict}, {"B": b, "H": hv, "C": c}, {"inv": akk})

    # 4) wy：WY 表示
    w, u, qg, kg = (empty(bhc, torch.bfloat16) for _ in range(4))
    compiled["wy"](
        {"q": qc, "k": kc, "v": vc, "beta": bc, "Akk": akk, **gate_in},
        {"B": b, "H": h, "HV": hv, "C": c, "length_per_chunk": L_PER_CHUNK,
         "head_dim": HEAD_DIM, "value_dim": VALUE_DIM},
        {"w": w, "u": u, "qg": qg, "kg": kg},
    )

    # 5) recurrent：chunk 递推
    o_c = empty(bhc, torch.bfloat16)
    final_state = empty((b, hv, HEAD_DIM, VALUE_DIM), torch.float32)
    compiled["recurrent"](
        {"q": qc, "Aqk": aqk, "kg": kg, "w": w, "u": u, "eg": eg, "initial_state": state0},
        {"B": b, "H": h, "HV": hv, "C": c, "length_per_chunk": L_PER_CHUNK,
         "head_dim": HEAD_DIM, "value_dim": VALUE_DIM, "scale": scale},
        {"o": o_c, "final_state": final_state},
    )

    return {"eg": eg, "g_cumsum": g_cumsum, "Aqk": aqk, "strict": strict, "Akk": akk,
            "w": w, "u": u, "qg": qg, "kg": kg, "o": o_c, "final_state": final_state,
            "initial_state": state0}


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    *,
    device: str | None = None,
    block_dim: int = 1,
    layout_device: str = "auto",
    check_gate_range: bool = True,
    impl: str = "stable",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """KDA 分块前向。

    Args:
        q, k: ``[B, T, H, 128]`` bfloat16。
        v: ``[B, T, HV, 128]`` bfloat16，``HV % H == 0``。
        g: ``[B, T, HV, 128]`` float32，log 空间的 per-dimension 衰减**增量**
            （kernel 内部做 chunk 内 cumsum）。
        beta: ``[B, T, HV]`` float32。
        scale: q 的缩放，默认 ``128 ** -0.5``。kernel 有 f32 标量入口，不需要 host 预乘。
        initial_state: ``[B, HV, 128, 128]`` float32，可选。
        output_final_state: 是否返回末态。
        device: ascriptor 设备名。
        block_dim: 启动核组数，只接受 ``SUPPORTED_BLOCK_DIM``。kernel 用
            ``GetVecIdx()/GetVecNum()`` 自行切分，而 ``GetVecNum() == 2 * block_dim``，
            所以这个值直接决定并行度 —— ``block_dim=1`` 只用到 2 个向量核。
            契约只覆盖到 4；Ascend950PR 物理上有 28 cube / 56 vec，但超过物理核数
            会在硬件 barrier 死锁。
        layout_device: deprecated ``"auto"`` / ``"npu"`` aliases; both select
            the owned device layout kernel. ``"cpu"`` raises under D-PM-37.
        check_gate_range: 是否校验 chunk 内门控跨度不超过
            :data:`MAX_GATE_SPAN` ``[impl]["forward"]``。默认开 —— 超限时 kernel 会静默吐
            NaN，那比报错糟得多。代价是对 ``g`` 做一次 cumsum + 两次规约。
            **这里用的是前向那条闸**（``stable`` 下 155）；要跑反向请走
            :func:`chunk_kda_fwd_with_caches` 或 :func:`~ascend_fla.ops.kda.chunk_kda`，
            它们用更严的反向闸（105）。
        impl: ``"stable"`` (default) selects the local stable gate/scores/WY
            and repaired recurrent; forward gate span <=155.
            Stable supports odd-C repeated-head ownership; upstream rejects it.
            ``"upstream"`` uses the original five kernels with gate span <=80.
            Both implement the same recurrence on their supported domains.

    Returns:
        ``(o, final_state)``，``o`` 为 ``[B, T, HV, 128]`` bfloat16；
        ``final_state`` 为 ``[B, HV, 128, 128]`` float32 或 ``None``。
    """
    device = resolve_soc(device)
    require_qualified(device)
    b, h, hv, c = _check(q, k, v, g, beta, initial_state, block_dim, impl, device=device)
    on_cpu = _resolve_layout(layout_device)
    if check_gate_range:
        _check_gate_range(g, c, on_cpu=on_cpu, impl=impl, path="forward",
                          **({"device": device} if device == "a2" else {}))
    scale = HEAD_DIM ** -0.5 if scale is None else float(scale)
    chain = _run_chain(q, k, v, g, beta, scale, initial_state, device=device,
                       block_dim=block_dim, on_cpu=on_cpu, b=b, h=h, hv=hv, c=c, impl=impl)
    o = (chain["o"] if device == "a2" else
         _from_bhcld(chain["o"], device=device, block_dim=block_dim))
    return o, (chain["final_state"] if output_final_state else None)


#: ``kda_bwd`` 声明的九个前向检查点。顺序无关，但**名字和形状必须完全一致** ——
#: 它的 ``validate_inputs`` 会逐个核对，多一个少一个都报错。
BWD_CACHE_NAMES = ("g_cumsum", "Aqk", "Akk", "w", "u", "qg", "kg", "v_new", "h")


def _scan_states(w, u, kg, eg, state0, *, b, hv, c, on_cpu=False):
    """逐 chunk 递推出 ``h``（chunk 起始状态）与 ``v_new``。

    ``kda_sub45_fused_kernel`` 内部算的就是这两个量，但它只写出 ``o`` 与
    ``final_state`` —— 见 `docs/matrix/gaps.json` 的 ``fwd-caches-not-emitted``。
    这里在 host 侧用 torch 复算一遍，**正确但慢**（C 次迭代 × 2 次 bmm，全是
    torch_npu 的小算子），只为先把反向链的正确性立住。

    ``on_cpu=True`` 时整段绕到 CPU 上算（D2H → 算 → H2D）。**这不是性能选项，是可用性
    选项**：这段用的是 Cast / bmm / stack 等内置算子，在内置算子包不覆盖当前 SoC 的机器上
    全部不可用（见 AGENTS.md §5 的可用面表）。也就是说 ``fwd-caches-not-emitted`` 除了慢，
    还让**训练路径**依赖内置算子包，而纯前向路径不依赖 —— 这一条曾让反向在只有 910 算子包
    的机器上以 ``561103`` / ``Cast ADD_TO_LAUNCHER_LIST_AICORE failed`` 失败。
    把这三项挪进 kernel 之后这个选项就可以删掉。

    递推（与 kda_bwd 的 ref/forward.py 逐行对应，fp32 累加、存储时降到 bf16）::

        h[c]      = state
        v_new[c]  = u[c] - w[c] @ state
        state     = state * exp2(g_last[c])[:, None] + kg[c]^T @ v_new[c]

    其中 ``exp2(g_last[c])`` 就是 ``eg`` 在该 chunk 末行的值（``eg == 2**g_cumsum``）。

    Returns:
        ``(h, v_new)``：``h`` 为 ``[B,C,HV,128,128]``、``v_new`` 为 BHCLD 的
        ``[B,HV,C,64,128]``，均 bfloat16。
    """
    dev = state0.device
    # 先 D2H 再算（不是先算再 D2H）—— 缺算子包的机器上 NPU 侧连 .float() 都不可用
    host = (lambda x: x.cpu()) if on_cpu else (lambda x: x)
    w, u, kg, eg = host(w), host(u), host(kg), host(eg)

    state = host(state0).float()
    h_chunks, v_new_chunks = [], []
    for ci in range(c):
        h_chunks.append(state)
        wc = w[:, :, ci].float()                       # [B,HV,64,128]
        vn = u[:, :, ci].float() - wc @ state          # [B,HV,64,128]
        v_new_chunks.append(vn)
        g_last = eg[:, :, ci, L_PER_CHUNK - 1, :]      # [B,HV,128] = exp2(g_cumsum 末行)
        state = state * g_last[..., None] + kg[:, :, ci].float().transpose(-1, -2) @ vn
    h = torch.stack(h_chunks, dim=1).bfloat16()        # [B,C,HV,128,128]
    v_new = torch.stack(v_new_chunks, dim=2).bfloat16()  # [B,HV,C,64,128]
    return (h.to(dev), v_new.to(dev)) if on_cpu else (h, v_new)


def chunk_kda_fwd_with_caches(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    *,
    device: str | None = None,
    block_dim: int = 1,
    layout_device: str = "auto",
    check_gate_range: bool = True,
    impl: str = "stable",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """前向，并额外产出 ``kda_bwd`` 需要的九个检查点。

    与 :func:`chunk_kda_fwd` **共用同一次 kernel 调用**（都走 ``_run_chain``），所以
    ``o`` 与 ``final_state`` 逐位相同。

    九个检查点里六个直接来自前向 kernel（``Aqk`` / ``Akk`` / ``w`` / ``u`` / ``qg`` /
    ``kg``），另外三个要补：

    * ``g_cumsum = log2(eg)`` —— gate kernel 只写出 ``eg = 2**g_cumsum``，不写 cumsum 本身。
    * ``h`` / ``v_new`` —— 融合 recurrent kernel 内部有，但不写出（见
      :func:`_scan_states`）。

    **这三项都是 host 侧补的，是当前反向链的性能瓶颈**，缺口记在
    ``docs/matrix/gaps.json`` 的 ``fwd-caches-not-emitted``。

    本函数**会顺便把反向链编译好**（首次调用时多花一次编译时间）。这不是顺手而为：
    CANN 只在首次算子解析时读 ``ASCEND_CUSTOM_OPP_PATH``，反向的 vendor 树若在前向
    执行之后才注册就解析不到（``rc=161001``）。见 ``runtime/binding.py`` 的
    ``register_custom_opp_path``。

    Returns:
        ``(o, final_state, caches)``。``caches`` 的键正是 :data:`BWD_CACHE_NAMES`，
        全部 bfloat16、token-major（``h`` 为 ``[B,C,HV,128,128]``），可直接喂 ``kda_bwd``。
    """
    device = resolve_soc(device)
    require_qualified(device)
    b, h_q, hv, c = _check(q, k, v, g, beta, initial_state, block_dim, impl, device=device)
    on_cpu = _resolve_layout(layout_device)
    # ⚠️ 门控必须在编译**之前**查。两个理由：
    #   ① 注定要被拒的调用不该先付一次完整编译的代价（十几个 kernel，分钟级）；
    #   ② 更要紧的是**错误信息会被换掉**：编译会注册新的 vendor 树，而 CANN 只在首次算子
    #      解析时读 ASCEND_CUSTOM_OPP_PATH，所以同进程里换 impl 时 binding 会先抛
    #      "不能再注册新的 vendor 树"，把本该报的"门控跨度超限"盖掉 —— 实测踩过，
    #      排查时一路看的是 opp 路径而不是真正的原因。
    if check_gate_range:
        # 这是训练入口（产检查点就是为了跑反向），所以用反向那条更严的闸
        _check_gate_range(g, c, on_cpu=on_cpu, impl=impl, path="backward",
                          **({"device": device} if device == "a2" else {}))
    # 反向链要在**首次 aclnn 调用之前**注册完 vendor 树，否则它的算子解析不到
    # （见 runtime/binding.py 的 register_custom_opp_path）。要缓存的唯一理由就是
    # 接着跑反向，所以在这里一并编译好。编译有两级缓存，重复调用不花钱。
    from .chunk_bwd import _compiled_chain as _bwd_chain

    _bwd_chain(device, block_dim, impl)
    scale = HEAD_DIM ** -0.5 if scale is None else float(scale)
    chain = _run_chain(q, k, v, g, beta, scale, initial_state, device=device,
                       block_dim=block_dim, on_cpu=on_cpu, b=b, h=h_q, hv=hv, c=c, impl=impl)

    h_states, v_new = _scan_states(chain["w"], chain["u"], chain["kg"], chain["eg"],
                                   chain["initial_state"], b=b, hv=hv, c=c, on_cpu=on_cpu)
    def tok(x: torch.Tensor, *, multiply=False) -> torch.Tensor:
        return _from_bhcld(x, dtype=torch.bfloat16, device=device, block_dim=block_dim,
                           multiply=multiply, factor=1.0 / math.log(2.0))

    # D-PM-42: stable's existing single FP32 multiply is fused with layout/cast,
    # preserving its rounded constant and materialization before BF16 RNE.
    # Upstream log2 remains a registered legacy arithmetic exception.
    if chain["g_cumsum"] is not None:
        if device == "a2":
            g_cum_log2 = _from_bhcld(chain["g_cumsum"], dtype=torch.float32,
                                    device=device, block_dim=block_dim, multiply=True,
                                    factor=1.0 / math.log(2.0))
        else:
            g_cum_log2 = tok(chain["g_cumsum"], multiply=True)
    else:
        g_cum_log2 = tok(chain["eg"].log2())
    caches = {
        "g_cumsum": g_cum_log2,
        "Aqk": tok(chain["Aqk"]),
        "Akk": tok(chain["Akk"]),
        "w": tok(chain["w"]),
        "u": tok(chain["u"]),
        "qg": tok(chain["qg"]),
        "kg": tok(chain["kg"]),
        "v_new": tok(v_new),
        "h": h_states,
    }
    missing = set(BWD_CACHE_NAMES) ^ set(caches)
    if missing:
        raise RuntimeError(f"检查点名字与 kda_bwd 的声明不符，差异 {sorted(missing)}")
    o = (chain["o"] if device == "a2" else
         _from_bhcld(chain["o"], device=device, block_dim=block_dim))
    return o, chain["final_state"], caches
