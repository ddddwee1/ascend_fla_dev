"""KDA 分块反向 —— 经 runtime 桥串起 ascriptor 的九个 kda_bwd kernel。

三段：

* **scan**（1 个 kernel）：沿 chunk 反向扫，出 ``dAqk`` / ``dh`` / ``dv`` / ``dh0``。
* **inverse**（4 个）：穿过 WY 表示与严格下三角求逆，出 ``dq`` / ``dk`` / ``dv`` /
  ``dbeta`` / ``dg`` 的主项和 ``dAkk``。
* **finalize**（4 个）：补上 chunk 内成对项，把 HV 维按分组规约回 H 维。

反向需要九个**前向检查点**，由 :func:`ascend_fla.ops.kda.chunk.chunk_kda_fwd_with_caches`
产出。检查点的正确性是这整条链的地基 —— 错了会表现成"某个梯度偏大"这种极难定位的
样子，所以 `tests/test_kda_caches_npu.py` 逐个对过参考实现。

接线时踩到的五处**与直觉不符**的地方（都以 ascriptor 的 ``kernels/stages.py`` 与
``kernels/composition.py`` 为准，不要凭 kernel 签名推断）：

1. ``inverse_mm`` 的第三个输出在 host 侧**取负**后才叫 ``dw``（``stages.py`` 里是
   ``-result[2]``）。漏掉负号不会报错，只会让 ``dk`` / ``dbeta`` 系统性偏。
2. ``dbeta`` 在 inverse 段是 **fp32**、finalize 段输出才是 bf16。两处 dtype 不同。
3. ``finalize_post`` 的 launch 参数顺序与 ``stages.py`` 里那个 python 函数的签名**不同**
   （``g_cumsum`` 在最前）。我们按 kernel 的 HostSpec 名字传，不依赖位置。
4. ``g`` 与 ``initial_state`` 虽然在 ``kda_bwd`` 的 contract ``inputs`` 里，**kernel 链
   却完全不用它们** —— 它们只用于造检查点。所以本函数不收这两个参数。
5. aclnn 的 HostSpec 把 GM 形状里出现的**全部符号维**列成标量，不只是 kernel 签名里
   显式声明的那些。九个反向 kernel 都因此多一个 ``T``（前向那五个恰好把符号都显式声明
   了，所以没踩到）。标量名一律以 ``CompiledKernel.scalar_names`` 为准。

同一 chunk/token 的张量在两种布局间来回：token-major ``(B,T,HV,D)`` 与 BHCLD
``(B,HV,C,64,D)``。这里全部用 ``view`` 完成（``C*64 == T`` 且内存连续），不产生拷贝。
"""
from __future__ import annotations

import functools
import importlib.util
import pathlib
from typing import Any

import torch

from ascend_fla.platform import capability, require_qualified, resolve_soc

from .chunk import (
    BWD_CACHE_NAMES,
    IMPLS,
    HEAD_DIM,
    L_PER_CHUNK,
    VALUE_DIM,
    SUPPORTED_BLOCK_DIM,
    _kernels_root,
    _resolve_layout,
    _layout_runtime,
    _load_kernel,
    _launch_a2,
)

__all__ = ["chunk_kda_bwd", "kda_bwd_kernels"]

#: 文件名 → kernel 函数名。按 ascriptor 的 kda_bwd 单元，顺序即执行顺序。
_KERNEL_MODULES = {
    "scan_fused": "scan_fused_kernel",
    "inverse_mm": "inverse_mm_kernel",
    "inverse_epilogue": "inverse_epilogue_kernel",
    "inverse_dainv": "inverse_dainv_kernel",
    "inverse_dakk_fused": "inverse_dakk_fused_kernel",
    "finalize_pre": "finalize_pre_kernel",
    "finalize_pair": "finalize_pair_kernel",
    "finalize_post": "finalize_post_kernel",
    "finalize_reduce": "finalize_reduce_kernel",
}

# The two finalize kernels stabilize gate exponentials. The inverse_mm derivative
# preserves arithmetic while fitting the accepted 32-ID local mutex budget.
_STABLE_BWD_KERNELS = {
    "inverse_mm": "inverse_mm_bounded_kernel",
    "finalize_pre": "finalize_pre_stable_kernel",
    "finalize_post": "finalize_post_stable_kernel",
}


def _bwd_kernels_root() -> pathlib.Path:
    """``kernels/projects/a5/kda_bwd`` 的位置。与 fwd 同级，只读引用（AGENTS.md §3）。"""
    import os

    direct = os.environ.get("ASCRIPTOR_KDA_BWD")
    if direct:
        return pathlib.Path(direct)
    root = _kernels_root().parent / "kda_bwd"
    if not (root / "kernels").is_dir():
        raise FileNotFoundError(
            f"找不到 ascriptor 的 kda_bwd 单元（试了 {root}）；设 $ASCRIPTOR_KDA_BWD 指向它"
        )
    return root


def _stable_bwd_root(device="a5") -> pathlib.Path:
    """本仓自有单元 ``kernels/projects/a5/kda_bwd_stable`` 的位置。"""
    unit = "kda_bwd_stable_fp32gc" if device == "a2" else "kda_bwd_stable"
    root = pathlib.Path(__file__).resolve().parents[3] / f"kernels/projects/{device}/{unit}"
    if not (root / "kernels").is_dir():
        raise FileNotFoundError(f"找不到本仓的 kda_bwd_stable 单元（试了 {root}）")
    return root


@functools.lru_cache(maxsize=None)
def kda_bwd_kernels(impl: str = "stable", device: str = "a5") -> dict[str, Any]:
    """按文件逐个加载九个 kernel。

    不 import 单元的 ``kernels`` 包 —— 它的 ``stages.py`` 依赖 ``_unit_runner``
    （harness 专用），而九个 kernel 文件本身只依赖 ``ascriptor.a5``。

    Args:
        impl: ``"stable"`` selects local ``finalize_pre`` / ``finalize_post`` and
            resource-bounded ``inverse_mm``; ``"upstream"`` keeps owner sources.
            **必须与前向的 impl 一致** —— 两边对可用门控跨度的上限不同，混用会让
            门控检查挡不住实际会失效的那一侧。
    """
    if impl not in IMPLS:
        raise ValueError(f"impl 只能是 {IMPLS}，收到 {impl!r}")
    if device == "a2":
        if impl != "stable":
            raise ValueError("A2 KDA backward declares only impl='stable'")
        directory = _stable_bwd_root(device) / "kernels"
        entries = (('scan_fused', 'scan_fused_a2_fp32gc_kernel'), ('inverse_mm', 'inverse_mm_a2_kernel'), ('inverse_epilogue', 'inverse_epilogue_a2_fp32gc_kernel'), ('inverse_dainv', 'inverse_dainv_a2_kernel'), ('inverse_dakk_fused', 'inverse_dakk_fused_a2_kernel'), ('finalize_pre', 'finalize_pre_a2_fp32gc_kernel'), ('finalize_pair', 'finalize_pair_a2_kernel'), ('finalize_post', 'finalize_post_a2_fp32gc_kernel'), ('finalize_reduce', 'finalize_reduce_a2_kernel'))
        return {stage: _load_kernel("a2_bwd", directory, stage, name)
                for stage, name in entries}
    up_dir = _bwd_kernels_root() / "kernels"
    st_dir = _stable_bwd_root() / "kernels" if impl == "stable" else None
    plan = []
    for stem, fn_name in _KERNEL_MODULES.items():
        if st_dir is not None and stem in _STABLE_BWD_KERNELS:
            plan.append((st_dir, stem, _STABLE_BWD_KERNELS[stem], "st"))
        else:
            plan.append((up_dir, stem, fn_name, "up"))

    out = {}
    for root_dir, stem, fn_name, tag in plan:
        path = root_dir / f"{stem}.py"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 kernel 文件 {path}")
        spec = importlib.util.spec_from_file_location(f"_afla_kda_bwd_{tag}_{stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out[stem] = getattr(module, fn_name)
    return out


@functools.lru_cache(maxsize=None)
def _compiled_chain(device: str, block_dim: int, impl: str = "stable") -> dict[str, Any]:
    """(device, block_dim, impl) → 已编译的九个 kernel。理由同前向的 ``_compiled_chain``。"""
    from ...runtime.compile import compile_kernel

    (_layout_runtime(device) if device == "a2" else _layout_runtime()).prepare(device, block_dim)
    return {name: compile_kernel(fn, device=device, block_dim=block_dim)
            for name, fn in (kda_bwd_kernels(impl, device) if device == "a2" else kda_bwd_kernels(impl)).items()}


def _check(q, k, v, beta, do, dht, caches, block_dim, *, device="a5") -> tuple[int, int, int, int]:
    """门控。返回 ``(B, H, HV, C)``。任何不满足都报错，绝不静默降级。"""
    if device == "a2" and block_dim not in capability(device)["supported_block_dim"]["chunk"]:
        raise ValueError("A2 backward block_dim must be one of (1, 2)")
    if block_dim not in SUPPORTED_BLOCK_DIM:
        raise ValueError(
            f"block_dim 只支持 {SUPPORTED_BLOCK_DIM}（ascriptor kda_bwd 契约声明的范围），"
            f"收到 {block_dim}；见 docs/matrix/gaps.json 的 block-dim-ceiling"
        )
    if q.dim() != 4:
        raise ValueError(f"q 应为 4 维 [B,T,H,128]，收到 {tuple(q.shape)}")
    b, t, h, kd = q.shape
    hv = v.shape[2]
    if kd != HEAD_DIM or v.shape[3] != VALUE_DIM:
        raise ValueError(f"kda_bwd 定尺要求 K=V=128，收到 K={kd} V={v.shape[3]}")
    if t % L_PER_CHUNK:
        raise ValueError(f"T 必须是 {L_PER_CHUNK} 的整数倍，收到 T={t}")
    if hv % h:
        raise ValueError(f"HV({hv}) 必须是 H({h}) 的整数倍")
    c = t // L_PER_CHUNK

    want = {
        "q": (b, t, h, HEAD_DIM), "k": (b, t, h, HEAD_DIM), "v": (b, t, hv, VALUE_DIM),
        "beta": (b, t, hv), "do": (b, t, hv, VALUE_DIM), "dht": (b, hv, HEAD_DIM, VALUE_DIM),
    }
    for name, tensor in (("q", q), ("k", k), ("v", v), ("beta", beta), ("do", do), ("dht", dht)):
        if tuple(tensor.shape) != want[name]:
            raise ValueError(f"{name} 应为 {want[name]}，收到 {tuple(tensor.shape)}")
        # kda_bwd 的 ABI 全 bf16 —— 与 kda_fwd 的 fp32 不一致，代价已量化，
        # 见 docs/matrix/gaps.json 的 kda-fwd-bwd-dtype-mismatch
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} 的 dtype 应为 bfloat16（kda_bwd ABI），收到 {tensor.dtype}")
        if tensor.device.type != "npu":
            raise ValueError(f"{name} 应在 NPU 上，收到 device={tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} 必须是连续张量，收到 stride={tuple(tensor.stride())}；"
                f"理由同 chunk.py 的 _check"
            )

    if set(caches) != set(BWD_CACHE_NAMES):
        raise ValueError(
            f"caches 必须恰好是 {BWD_CACHE_NAMES}；"
            f"多了 {sorted(set(caches) - set(BWD_CACHE_NAMES))}，"
            f"少了 {sorted(set(BWD_CACHE_NAMES) - set(caches))}"
        )
    cache_shapes = {
        "g_cumsum": (b, t, hv, HEAD_DIM), "w": (b, t, hv, HEAD_DIM),
        "u": (b, t, hv, VALUE_DIM), "qg": (b, t, hv, HEAD_DIM),
        "kg": (b, t, hv, HEAD_DIM), "v_new": (b, t, hv, VALUE_DIM),
        "Aqk": (b, t, hv, L_PER_CHUNK), "Akk": (b, t, hv, L_PER_CHUNK),
        "h": (b, c, hv, HEAD_DIM, VALUE_DIM),
    }
    for name, shape in cache_shapes.items():
        got = caches[name]
        if tuple(got.shape) != shape:
            raise ValueError(f"caches[{name!r}] 应为 {shape}，收到 {tuple(got.shape)}")
        expected_dtype = torch.float32 if device == "a2" and name == "g_cumsum" else torch.bfloat16
        if got.dtype != expected_dtype:
            raise ValueError(f"caches[{name!r}] 的 dtype 应为 {expected_dtype}，收到 {got.dtype}")
        if not got.is_contiguous():
            raise ValueError(f"caches[{name!r}] 必须是连续张量，收到 stride={tuple(got.stride())}")
    return b, h, hv, c


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    caches: dict[str, torch.Tensor],
    *,
    device: str | None = None,
    block_dim: int = 1,
    impl: str = "stable",
    layout_device: str = "auto",
) -> dict[str, torch.Tensor]:
    """KDA 分块反向。

    Args:
        q, k: ``[B, T, H, 128]`` bfloat16。
        v: ``[B, T, HV, 128]`` bfloat16，``HV % H == 0``。
        beta: ``[B, T, HV]`` bfloat16。
        do: ``[B, T, HV, 128]`` bfloat16，对 ``o`` 的上游梯度。
        dht: ``[B, HV, 128, 128]`` bfloat16，对 ``final_state`` 的上游梯度。
        caches: :data:`~ascend_fla.ops.kda.chunk.BWD_CACHE_NAMES` 九项，
            由 :func:`~ascend_fla.ops.kda.chunk.chunk_kda_fwd_with_caches` 产出。
        device: ascriptor 设备名。
        block_dim: 启动核组数，只接受 ``SUPPORTED_BLOCK_DIM``。
        impl: ``"stable"``（默认）或 ``"upstream"``，见 :func:`kda_bwd_kernels`。
            **要与造 caches 时用的前向 impl 一致。**
        layout_device: deprecated ``"auto"`` / ``"npu"`` aliases for kernel-side
            checkpoint gathering; ``"cpu"`` raises under D-PM-37.

    Returns:
        ``{"dq", "dk", "dv", "dbeta", "dg", "dh0"}``，全部 bfloat16。
        ``dq``/``dk`` 为 ``[B,T,H,128]``（已按分组规约回 H 维），``dv``/``dg`` 为
        ``[B,T,HV,128]``，``dbeta`` 为 ``[B,T,HV]``，``dh0`` 为 ``[B,HV,128,128]``。
    """
    device = resolve_soc(device)
    require_qualified(device)
    b, h, hv, c = _check(q, k, v, beta, do, dht, caches, block_dim, device=device)
    _resolve_layout(layout_device)
    if device == "a2":
        return _run_chain_a2(q, k, v, beta, do, dht, caches, block_dim=block_dim, impl=impl)
    compiled = _compiled_chain(device, block_dim, impl)
    dev = q.device
    t = c * L_PER_CHUNK

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(*shape, dtype=dtype, device=dev)

    tok_d = (b, t, hv, HEAD_DIM)        # token-major，最后一维 128
    tok_l = (b, t, hv, L_PER_CHUNK)     # token-major，最后一维 64
    bhcld = (b, hv, c, L_PER_CHUNK, HEAD_DIM)
    bhcll = (b, hv, c, L_PER_CHUNK, L_PER_CHUNK)
    state = (b, hv, HEAD_DIM, VALUE_DIM)

    # Gather each chunk's last gate row on-device. The source slice is metadata
    # only; explicit strides retain the original token-major checkpoint ABI.
    gate_source = caches["g_cumsum"].view(-1)[(L_PER_CHUNK-1)*hv*HEAD_DIM:]
    g_last = _layout_runtime().move(
        gate_source, (b, c, hv, 1, HEAD_DIM),
        (t*hv*HEAD_DIM, L_PER_CHUNK*hv*HEAD_DIM, HEAD_DIM, 0, 1),
        device=device, block_dim=block_dim).view(b, c, hv, HEAD_DIM)
    d_aqk, dh, dv_scan, dh0 = (empty(tok_l), empty((b, c, hv, HEAD_DIM, VALUE_DIM)),
                               empty(tok_d), empty(state))
    compiled["scan_fused"](
        {"kg": caches["kg"], "qg": caches["qg"], "w": caches["w"], "g_last": g_last,
         "grad_out": do, "Aqk": caches["Aqk"], "v_new": caches["v_new"],
         "dht": dht.view(b, hv, HEAD_DIM // 2, 2 * VALUE_DIM)},
        {"B": b, "HV": hv, "C": c, "T": t},
        {"dAqk": d_aqk, "dh": dh, "dv": dv_scan, "dh0": dh0},
    )

    # ---- inverse：穿过 WY 与求逆 ----
    d_qg, d_kg, d_vh, d_v_beta, d_k_beta_g = (empty(bhcld) for _ in range(5))
    compiled["inverse_mm"](
        {"do_bf16": do, "vnew_bf16": caches["v_new"], "dv_bf16": dv_scan,
         "h_bf16": caches["h"], "dh_bf16": dh, "Akk_bf16": caches["Akk"]},
        {"B": b, "HV": hv, "C": c, "T": t},
        {"d_qg": d_qg, "d_kg": d_kg, "d_vh": d_vh,
         "d_v_beta": d_v_beta, "d_k_beta_g": d_k_beta_g},
    )
    # D-PM-42 registered legacy arithmetic: no layout/cast accompanies this
    # negation, so it remains separately audited until the kernel batch.
    dw = -d_vh

    dq_hv, dk_hv = empty(tok_d), empty(tok_d)
    dv_out, dg_core, k_exp = empty(tok_d), empty(tok_d), empty(bhcld)
    dbeta_f32 = empty((b, t, hv), torch.float32)   # inverse 段的 dbeta 是 fp32
    compiled["inverse_epilogue"](
        {"d_qg": d_qg, "d_kg": d_kg, "d_v_beta": d_v_beta, "d_k_beta_g": d_k_beta_g,
         "q": q, "k": k, "v": v, "g_cumsum": caches["g_cumsum"], "beta": beta,
         "h": caches["h"], "dh": dh},
        {"B": b, "HV": hv, "C": c, "Hq": h, "G": hv // h, "T": t},
        {"dq_hv": dq_hv, "dk_hv": dk_hv, "dv": dv_out, "dbeta": dbeta_f32,
         "dg_core": dg_core, "k_exp": k_exp},
    )

    d_tri = empty(bhcll)
    compiled["inverse_dainv"](
        {"dv": dv_scan, "v": v, "d_w": dw, "k_exp": k_exp, "beta": beta},
        {"B": b, "HV": hv, "C": c, "T": t},
        {"dtri": d_tri},
    )

    d_akk = empty(tok_l)
    compiled["inverse_dakk_fused"](
        {"Akk_bf16": caches["Akk"], "Dtri_bf16": d_tri.view(b, hv, t, L_PER_CHUNK)},
        {"B": b, "HV": hv, "T": t},
        {"dAkk": d_akk},
    )

    # ---- finalize：chunk 内成对项 + 按分组规约 ----
    q_scaled, k_scaled, kg_f = (empty(bhcld) for _ in range(3))
    m_qk, m_base, m_beta = (empty(bhcll) for _ in range(3))
    compiled["finalize_pre"](
        {"q": q, "k": k, "g_cumsum": caches["g_cumsum"], "beta": beta,
         "dAqk": d_aqk, "dAkk": d_akk},
        {"B": b, "HV": hv, "C": c, "H": h, "G": hv // h, "T": t},
        {"q_scaled": q_scaled, "k_scaled": k_scaled, "kg": kg_f,
         "m_qk": m_qk, "m_base": m_base, "m_beta": m_beta},
    )

    # finalize_pair 按 (B,HV,T,·) 看同一块内存
    def flat(x):
        return x.view(b, hv, t, x.shape[-1])

    dq_pair, dk_pair, s_base, t_beta = (empty((b, hv, t, HEAD_DIM)) for _ in range(4))
    compiled["finalize_pair"](
        {"Mqk_bf16": flat(m_qk), "Mbase_bf16": flat(m_base), "Mbeta_bf16": flat(m_beta),
         "q_bf16": flat(q_scaled), "k_bf16": flat(k_scaled), "kg_bf16": flat(kg_f)},
        {"B": b, "HV": hv, "T": t},
        {"dq_pair": dq_pair, "dk_pair": dk_pair, "s_base": s_base, "t_beta": t_beta},
    )

    # finalize_post 又要回 (B,HV,C,64,·)
    def chunked(x):
        return x.view(b, hv, c, L_PER_CHUNK, x.shape[-1])

    dq_post = empty(bhcld, torch.float32)
    dk_post = empty(bhcld, torch.float32)
    dbeta_out, dg_out = empty((b, t, hv)), empty(tok_d)
    compiled["finalize_post"](
        {"g_cumsum": caches["g_cumsum"], "qk_left": chunked(dq_pair),
         "qk_right": chunked(dk_pair), "s_base": chunked(s_base), "t_beta": chunked(t_beta),
         "q": q, "k": k, "beta": beta, "dq_hv_in": dq_hv, "dk_hv_in": dk_hv,
         "dbeta_in": dbeta_f32, "dg_core_in": dg_core},
        {"B": b, "HV": hv, "C": c, "H": h, "G": hv // h, "T": t},
        {"dq_hv_out": dq_post, "dk_hv_out": dk_post,
         "dbeta_out": dbeta_out, "dg_out": dg_out},
    )

    dq, dk = empty((b, t, h, HEAD_DIM)), empty((b, t, h, HEAD_DIM))
    compiled["finalize_reduce"](
        {"dq_hv": dq_post, "dk_hv": dk_post},
        {"B": b, "HV": hv, "H": h, "C": c, "G": hv // h, "T": t},
        {"dq_out": dq, "dk_out": dk},
    )

    return {"dq": dq, "dk": dk, "dv": dv_out, "dbeta": dbeta_out, "dg": dg_out, "dh0": dh0}


def _run_chain_a2(q, k, v, beta, do, dht, caches, *, block_dim, impl):
    """Execute the FP32-gc unit's nine-stage ABI without host transformations."""
    kern = _compiled_chain("a2", block_dim, impl)
    inputs = dict(q=q, k=k, v=v, beta=beta, do=do, dht=dht, saved=caches)
    L, D = L_PER_CHUNK, HEAD_DIM
    options = None

    def empty(*shape, dtype=torch.bfloat16):
        # Public execution allocates outputs; the standalone harness seeds NaNs.
        return torch.empty(shape, dtype=dtype, device=q.device)

    def launch_kernel(entry, args, unused_options):
        ni, no = len(entry.input_names), len(entry.output_names)
        outputs = args[ni:ni+no]
        explicit = args[ni+no:]
        scalars = dict(zip(entry.scalar_names[:len(explicit)], explicit, strict=True))
        _launch_a2(entry, args[:ni], outputs, scalars)
        return outputs[0] if no == 1 else outputs
    saved = inputs['saved']
    b, t, h, _ = inputs['q'].shape
    hv, c, bt = (inputs['v'].shape[2], t // L, b * t)

    def tok(x, width):
        return x.view(bt, width)
    q2, k2 = (tok(inputs['q'], h * D), tok(inputs['k'], h * D))
    v2, do2, beta2 = (tok(inputs['v'], hv * D), tok(inputs['do'], hv * D), tok(inputs['beta'], hv))
    gc2 = tok(saved['g_cumsum'], hv * D)
    aqk2, akk2 = (tok(saved['Aqk'], hv * L), tok(saved['Akk'], hv * L))
    kg2, qg2, w2, vnew2 = (tok(saved[n], hv * D) for n in ('kg', 'qg', 'w', 'v_new'))
    d_aqk, d_h = (empty(bt, hv * L), empty(b, c, hv, D, D))
    d_v_scan, d_h0 = (empty(bt, hv * D), empty(b, hv, D, D))
    d_aqk, d_h, d_v_scan, d_h0 = launch_kernel(kern['scan_fused'], (kg2, qg2, w2, gc2, do2, aqk2, vnew2, inputs['dht'], d_aqk, d_h, d_v_scan, d_h0, b, hv, c), options)
    mm_out = tuple((empty(b, hv, c, L, D) for _ in range(5)))
    d_qg, d_kg, d_w, d_v_beta, d_k_beta_g = launch_kernel(kern['inverse_mm'], (do2, vnew2, d_v_scan, saved['h'], d_h, akk2, *mm_out, b, hv, c), options)
    ep_out = (empty(bt, hv * D), empty(bt, hv * D), empty(bt, hv * D), empty(bt, hv, dtype=torch.float32), empty(bt, hv * D), empty(b, hv, c, L, D))
    dq_hv, dk_hv, dv_out, dbeta, dg_core, k_exp = launch_kernel(kern['inverse_epilogue'], (d_qg, d_kg, d_v_beta, d_k_beta_g, q2, k2, v2, gc2, beta2, saved['h'], d_h, *ep_out, b, h, hv, c), options)
    dtri = launch_kernel(kern['inverse_dainv'], (d_v_scan, v2, d_w, k_exp, beta2, empty(b, hv, c, L, L), b, hv, c), options)
    d_akk = launch_kernel(kern['inverse_dakk_fused'], (akk2, dtri, empty(bt, hv * L), b, hv, c), options)
    pre_out = (empty(b, hv, c, L, D), empty(b, hv, c, L, D), empty(b, hv, c, L, D), empty(b, hv, c, L, L), empty(b, hv, c, L, L), empty(b, hv, c, L, L))
    q_scaled, k_scaled, kg_f, m_qk, m_base, m_beta = launch_kernel(kern['finalize_pre'], (q2, k2, gc2, beta2, d_aqk, d_akk, *pre_out, b, h, hv, c), options)

    def flat(x):
        return x.view(b, hv, t, x.shape[-1])
    pair_out = tuple((empty(b, hv, t, D) for _ in range(4)))
    qk_left, qk_right, s_base, t_beta = launch_kernel(kern['finalize_pair'], (flat(m_qk), flat(m_base), flat(m_beta), flat(q_scaled), flat(k_scaled), flat(kg_f), *pair_out, b, hv, t), options)

    def chunked(x):
        return x.view(b, hv, c, L, x.shape[-1])
    post_out = (empty(b, hv, c, L, D, dtype=torch.float32), empty(b, hv, c, L, D, dtype=torch.float32), empty(bt, hv), empty(bt, hv * D))
    dq_post, dk_post, dbeta_out, dg_out = launch_kernel(kern['finalize_post'], (gc2, chunked(qk_left), chunked(qk_right), chunked(s_base), chunked(t_beta), q2, k2, beta2, dq_hv, dk_hv, dbeta, dg_core, *post_out, b, h, hv, c), options)
    dq, dk = launch_kernel(kern['finalize_reduce'], (dq_post, dk_post, empty(bt, h * D), empty(bt, h * D), b, hv, h, c), options)
    outputs = {'dq': dq.view(b, t, h, D), 'dk': dk.view(b, t, h, D), 'dv': dv_out.view(b, t, hv, D), 'dbeta': dbeta_out.view(b, t, hv), 'dg': dg_out.view(b, t, hv, D), 'dh0': d_h0}
    return outputs
