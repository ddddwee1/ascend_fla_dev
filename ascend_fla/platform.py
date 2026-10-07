"""SoC 显式化：解析、能力表、入口检查（AGENTS.md §9）。

SoC 顺序改成 A2 → A3 → A5 之后，``device="a5"`` 不能再当默认值 —— **在 A2 上用 A5 的门控
上限就是静默错误**（AGENTS.md §7）。本模块把"跑在哪个 SoC 上、这个 SoC 上实测过什么"
变成显式的、按 SoC 查表的能力声明。

解析顺序（``resolve_soc``）：显式实参 → ``ASCEND_FLA_SOC`` 环境变量 → torch_npu 设备名探测；
三条都失败就报错并给出设法，**绝不悄悄默认成某个 SoC**。能力表 :data:`CAPABILITIES` 按 SoC 查，
a5 从 ``ops/kda`` 现有常量原样搬来（只搬家不改值），a2/a3 标 ``qualified=False`` 与空限值，
调用即报"该 SoC 未验收"。

解析结果进程内备忘（``_RESOLVED``），因为它会落在每次前向的热路径上，必须 O(1)
（AGENTS.md §6 铁律三：缓存键不能比缓存贵）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: 解析用环境变量。主机侧无 NPU 时必须显式设。
SOC_ENV = "ASCEND_FLA_SOC"

#: a5 的实测能力，是这些数字的**唯一真值源**：``ops/kda/{chunk,fused_recurrent}.py`` 的
#: ``SUPPORTED_BLOCK_DIM`` / ``MAX_GATE_SPAN`` 现在从这里读（a5 的别名），A2-02 只搬家不改值。
#: ``supported_block_dim`` 分 chunk / decode 两条路；``max_gate_span`` 是二维的 ``[impl][direction]``。
_A5_CAPABILITY: dict[str, Any] = {
    "qualified": True,
    "supported_block_dim": {"chunk": (1, 2, 3, 4), "decode": (1, 2, 4, 8, 16, 28)},
    "max_gate_span": {
        "upstream": {"forward": 80.0, "backward": 80.0},
        "stable": {"forward": 155.0, "backward": 105.0},
    },
    "unit_root": "kernels/projects/a5",
    "evidence": (
        "A5/950 实测；ops/kda/{chunk,fused_recurrent}.py 常量 + 两对 contract.json 的 "
        "domain.gate_span；docs/matrix/gaps.json"
    ),
}


def _unqualified(soc: str, why: str) -> dict[str, Any]:
    """未验收 SoC 的占位能力：空限值 + 证据指针，调用即报错。"""
    return {
        "qualified": False,
        "supported_block_dim": {},
        "max_gate_span": {},
        "unit_root": f"kernels/projects/{soc}",
        "evidence": why,
    }


#: 按 SoC 查的能力表。a2/a3 未验收 —— a2 的验收门是 A2-11，a3 的是 A3 波次。
CAPABILITIES: dict[str, dict[str, Any]] = {
    "a5": _A5_CAPABILITY,
    "a2": {
        "qualified": False,
        "supported_block_dim": {"chunk": (1, 2), "decode": (1, 2)},
        "max_gate_span": {"stable": {
            "forward": 158.0,
            # A2-13: kda_bwd_stable_fp32gc real-cache end-to-end evidence only;
            # benchmarks/a2/evidence/kda_bwd/fp32gc-v1/README.md records limitations.
            "backward": 128.0,
        }},
        "unit_root": "kernels/projects/a2",
        "evidence": (
            "A2-12：910B3/CANN 9.0.0 的 stable 前向单元实测，qkv BF16、状态 FP32；"
            "公共 dispatch 尚未接线，保留未验收总开关；"
            "benchmarks/a2/evidence/kda_fwd/README.md 与 RESULTS.md"
        ),
    },
    "a3": _unqualified("a3", "A3 尚未开工；能力待 A3 波次"),
}

#: 已知 SoC 集合。解析出来的 SoC 必须落在这里。
KNOWN_SOCS = tuple(CAPABILITIES)

_RESOLVED: str | None = None  # 进程内备忘；仅缓存自动解析的结果，显式实参不入缓存


def _probe_from_device_name() -> str | None:
    """读 torch_npu 的设备名做映射：Ascend950* → a5、910B* → a2、910_93 / 910C → a3。

    以真机读数为准的完整映射在 A2-10 / A3 补齐；这里只认目前确定的三条。无 torch_npu、
    无可见 NPU、或名字不认识都返回 None（交给上层报错），绝不猜。
    """
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
    except Exception:
        return None
    try:
        if not torch.npu.is_available():
            return None
        name = torch.npu.get_device_name(0) or ""
    except Exception:
        return None
    lowered = name.lower()
    if "950" in lowered:
        return "a5"
    if "910b" in lowered:
        return "a2"
    if "910_93" in lowered or "910c" in lowered:
        return "a3"
    return None


def resolve_soc(explicit: str | None = None) -> str:
    """解析当前 SoC。

    顺序：``explicit`` 实参 → :data:`SOC_ENV` 环境变量 → 设备名探测。三条都空就报错并给出
    设法（AGENTS.md §7：报清楚哪条约束没满足、怎么修）。自动解析的结果进程内备忘；显式实参
    不缓存，方便同进程内针对不同 SoC 做入口检查。
    """
    global _RESOLVED
    if explicit is not None:
        soc = explicit
    elif _RESOLVED is not None:
        return _RESOLVED
    else:
        soc = os.environ.get(SOC_ENV) or _probe_from_device_name()
    if not soc:
        raise RuntimeError(
            f"无法解析 SoC：既没有设 {SOC_ENV}，也没有从 torch_npu 探到已知设备名。"
            f"主机侧无 NPU 时请显式设 {SOC_ENV}=a2|a3|a5，或在有 torch_npu 的真机上运行。"
        )
    if soc not in CAPABILITIES:
        raise ValueError(f"未知 SoC {soc!r}，只认 {KNOWN_SOCS}")
    if explicit is None:
        _RESOLVED = soc
    return soc


def require_qualified(soc: str) -> None:
    """SoC 未验收就报错。**必须排在有副作用的编译之前调**（plan.md 第二期教训 4：
    校验要排在副作用之前，否则副作用的报错会顶替掉真正的原因）。"""
    cap = CAPABILITIES[soc]
    if not cap["qualified"]:
        raise RuntimeError(
            f"该 SoC 未验收：{soc} —— {cap['evidence']}。"
            f"当前只有 a5 通过真机验收；a2 的验收门是 A2-11。"
        )


def capability(soc: str) -> dict[str, Any]:
    """取某 SoC 的能力字典（原始引用，别原地改）。未知 SoC 报错。"""
    if soc not in CAPABILITIES:
        raise ValueError(f"未知 SoC {soc!r}，只认 {KNOWN_SOCS}")
    return CAPABILITIES[soc]


def unit_root(soc: str) -> Path:
    """kernel 单元根 ``kernels/projects/<soc>``（绝对路径）。"""
    return _REPO_ROOT / CAPABILITIES[soc]["unit_root"]


def supported_block_dim(soc: str, path: str) -> tuple[int, ...]:
    """某 SoC / 某路径（``chunk`` 或 ``decode``）实测支持的 block_dim。未验收先报错。"""
    require_qualified(soc)
    table = CAPABILITIES[soc]["supported_block_dim"]
    if path not in table:
        raise ValueError(f"{soc} 没有 {path!r} 路径的 block_dim 表，只有 {tuple(table)}")
    return table[path]


def max_gate_span(soc: str) -> dict[str, dict[str, float]]:
    """某 SoC 的门控跨度上限 ``[impl][direction]``。未验收先报错。"""
    require_qualified(soc)
    return CAPABILITIES[soc]["max_gate_span"]


def _reset_cache() -> None:
    """清进程内备忘。仅供测试在切 ``ASCEND_FLA_SOC`` 之间用。"""
    global _RESOLVED
    _RESOLVED = None
