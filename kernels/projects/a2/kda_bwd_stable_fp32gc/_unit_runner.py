"""Self-contained unit protocol ascriptor.kernel-unit/1 (library RFC-0012).

This file is the canonical helper. ``tools/export_runner.py`` copies its exact bytes
into a unit as ``_unit_runner.py`` and records their SHA-256 in ``runner-source.json``.
Imports of Torch, NumPy, the DSL and runtime are lazy; CPU references need no CANN.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import operator
import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "ascriptor.kernel-unit/1"
LAUNCHER_STAGES = {"sim": "sim", "pipesim": "pipesim", "cannsim": "cannsim", "board": "board", "aclnn": "board", "pypto": "board"}
DTYPES = {"f32": "float32", "fp32": "float32", "float": "float32", "f16": "float16", "fp16": "float16", "half": "float16", "bf16": "bfloat16", "i32": "int32", "i64": "int64"}


class ContractError(ValueError):
    """A malformed unit, unsupported option or unsuccessful verification."""


def _names(value: Any, label: str) -> set[str]:
    if isinstance(value, dict):
        names = list(value)
    elif isinstance(value, list):
        names = [item if isinstance(item, str) else item.get("id", item.get("name")) for item in value]
    else:
        raise ContractError(f"{label} must be an object or array of named entries")
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ContractError(f"{label} needs nonempty unique names")
    return set(names)


def comparison_rule(contract: dict, name: str, stage: bool = False) -> dict:
    comparison = contract["comparison"]
    rule = dict(comparison.get("default", {}))
    rule.update(comparison.get("stage_outputs" if stage else "outputs", {}).get(name, {}))
    if rule.get("mode") not in {"bitwise", "exact", "allclose", "one_of"}:
        raise ContractError(f"comparison for {name} requires mode bitwise, exact, allclose or one_of")
    if not isinstance(rule.get("reason"), str) or not rule["reason"].strip():
        raise ContractError(f"comparison for {name} requires a reason")
    if rule["mode"] == "one_of":
        count = rule.get("max_candidates")
        if stage or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ContractError(f"comparison for {name}: one_of requires final outputs and positive max_candidates")
        if {"rtol", "atol", "max_relative_l2", "equal_nan", "reference_dtype"} & rule.keys():
            raise ContractError(f"comparison for {name}: one_of cannot combine with numeric tolerance fields")
    if rule["mode"] == "allclose":
        for key in ("rtol", "atol"):
            value = rule.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ContractError(f"comparison for {name} requires finite nonnegative {key}")
    if "equal_nan" in rule and not isinstance(rule["equal_nan"], bool):
        raise ContractError(f"comparison for {name}: equal_nan must be boolean")
    if "max_relative_l2" in rule:
        value = rule["max_relative_l2"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ContractError(f"comparison for {name}: max_relative_l2 must be finite and nonnegative")
    if "reference_dtype" in rule:
        declared = rule["reference_dtype"]
        if rule["mode"] != "allclose" or declared not in ("float32", "float64"):
            raise ContractError(f"comparison for {name}: reference_dtype requires allclose and float32 or float64")
        specs = contract.get("stages" if stage else "outputs", {})
        spec = specs.get(name, {}) if isinstance(specs, dict) else {}
        actual = spec.get("dtype") if isinstance(spec, dict) else None
        if not isinstance(actual, str) or DTYPES.get(actual, actual) not in {"float16", "bfloat16", "float32", "float64"}:
            raise ContractError(f"comparison for {name}: reference_dtype requires an explicit floating actual dtype")
    return rule


def validate_contract(contract: dict) -> dict:
    """Validate the portable semantic subset in addition to repository JSON Schema."""
    required = {"schema", "id", "kind", "description", "inputs", "outputs", "domain", "dependencies", "comparison", "cases", "support", "provenance"}
    missing = required - contract.keys()
    if missing:
        raise ContractError(f"contract missing fields: {', '.join(sorted(missing))}")
    if contract["schema"] != SCHEMA or contract["kind"] not in {"kernel", "project", "api-example", "attention"}:
        raise ContractError("unsupported contract schema or kind")
    if not contract["id"] or not contract["description"] or not contract["domain"] or not contract["provenance"]:
        raise ContractError("id, description, domain and provenance must be nonempty")
    _names(contract["inputs"], "inputs")
    outputs = _names(contract["outputs"], "outputs")
    if not isinstance(contract["dependencies"], dict) or {"python", "ascriptor", "host"} - contract["dependencies"].keys():
        raise ContractError("dependencies require python, ascriptor and host")
    if not isinstance(contract["comparison"], dict):
        raise ContractError("comparison must be an object")
    for name in outputs:
        comparison_rule(contract, name)
    extras = set(contract["comparison"].get("outputs", {})) - outputs
    if extras:
        raise ContractError(f"comparison names undeclared outputs: {sorted(extras)}")
    if contract["kind"] == "project":
        if {"stages", "workspace", "saved_state"} - contract.keys():
            raise ContractError("project requires stages, workspace and saved_state")
        for name in _names(contract["stages"], "stages"):
            comparison_rule(contract, name, stage=True)
    cases = contract["cases"]
    if not isinstance(cases, list) or not cases:
        raise ContractError("cases must be a nonempty array")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or {"id", "seed", "parameters"} - case.keys():
            raise ContractError("each case requires id, seed and parameters")
        name = case["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in seen:
            raise ContractError("case IDs must be unique, nonempty path-safe names")
        seen.add(name)
        if isinstance(case["seed"], bool) or not isinstance(case["seed"], int) or not isinstance(case["parameters"], dict):
            raise ContractError(f"case {name} requires integer seed and object parameters")
        if "block_dim" in case:
            validate_block_dim(contract, case["block_dim"])
    if not isinstance(contract["support"], list) or not contract["support"]:
        raise ContractError("support must be a nonempty array")
    for row in contract["support"]:
        if not isinstance(row, dict) or {"device", "backend", "stage", "status"} - row.keys():
            raise ContractError("each support row requires device, backend, stage and status")
        if row["backend"] not in {"cce", "pypto_pro", "pto_isa"} or row["stage"] not in {"emit", "compile", "reference", *LAUNCHER_STAGES.values()} or row["status"] not in {"untested", "passed", "failed", "gap"}:
            raise ContractError("invalid support backend, stage or status")
        if row["status"] == "gap" and not row.get("reason"):
            raise ContractError("a support gap requires a located reason")
        if "cases" in row:
            scope = row["cases"]
            if (not isinstance(scope, list) or not scope or any(not isinstance(case, str) for case in scope)
                    or len(set(scope)) != len(scope) or set(scope) - seen):
                raise ContractError("support cases must be a nonempty array of unique declared case IDs")
    return contract


def validate_block_dim(contract: dict, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError("block_dim must be a positive integer")
    allowed = contract["domain"].get("block_dim")
    if isinstance(allowed, list) and value not in allowed:
        raise ContractError(f"block_dim {value} is outside the declared domain {allowed}")
    if isinstance(allowed, dict) and (value < allowed.get("min", 1) or value > allowed.get("max", value)):
        raise ContractError(f"block_dim {value} is outside the declared domain")
    return value


def _array(value: Any) -> tuple[Any, str, bytes]:
    import numpy as np

    if hasattr(value, "detach"):
        import torch

        tensor = value.detach().cpu().contiguous()
        dtype = str(tensor.dtype).removeprefix("torch.")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        try:
            array = tensor.numpy()
        except TypeError:
            array = tensor.to(torch.complex128 if tensor.is_complex() else torch.float64).numpy()
    else:
        array = np.asarray(value)
        dtype, raw = str(array.dtype), array.tobytes()
    if array.dtype.kind not in "biufc":
        raise ContractError(f"outputs must be numeric values, got {dtype}")
    return array, dtype, raw


def fingerprint(value: Any) -> str:
    """Detect input mutation/reuse without persisting expected tensor data."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for name in sorted(item):
                digest.update(str(name).encode())
                visit(item[name])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}".encode())
            for child in item:
                visit(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(repr(item).encode())
        else:
            array, dtype, raw = _array(item)
            digest.update(f"{dtype}:{array.shape}".encode())
            digest.update(raw)

    visit(value)
    return digest.hexdigest()


def _storage_ids(value: Any) -> set[tuple[str, int]]:
    """Identify live tensor allocations so a generator cannot reuse reference buffers."""
    if isinstance(value, dict):
        return set().union(*(_storage_ids(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(_storage_ids(item) for item in value))
    if hasattr(value, "untyped_storage"):
        return {("torch", value.untyped_storage().data_ptr())}
    if hasattr(value, "__array_interface__"):
        while getattr(value, "base", None) is not None:
            value = value.base
        if hasattr(value, "__array_interface__"):
            return {("numpy", value.__array_interface__["data"][0])}
    return set()


def _dimension(expression: Any, parameters: dict) -> int:
    if isinstance(expression, int):
        return expression
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in parameters and isinstance(parameters[node.id], int):
            return parameters[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ContractError(f"unsupported output shape expression {expression!r}")

    return evaluate(ast.parse(str(expression), mode="eval").body)


def _check_named(values: Any, names: set[str], label: str) -> dict:
    if not isinstance(values, dict) or not values or set(values) != names:
        actual = sorted(values) if isinstance(values, dict) else type(values).__name__
        raise ContractError(f"{label} names must be {sorted(names)}, got {actual}")
    return values


def describe_outputs(values: dict, contract: dict, case: dict, stage: bool = False) -> dict:
    import numpy as np

    names = _names(contract["stages"] if stage else contract["outputs"], "stages" if stage else "outputs")
    _check_named(values, names, "stage reference" if stage else "reference")
    result = {}
    for name, value in values.items():
        array, dtype, raw = _array(value)
        if array.size == 0:
            raise ContractError(f"reference output {name} is empty")
        rule = comparison_rule(contract, name, stage)
        if not np.isfinite(array).all() and not rule.get("equal_nan", False):
            raise ContractError(f"reference output {name} contains nonfinite values")
        specs = contract.get("stages" if stage else "outputs", {})
        spec = specs.get(name, {}) if isinstance(specs, dict) else {}
        if isinstance(spec, dict):
            declared = rule.get("reference_dtype", spec.get("dtype"))
            if declared and DTYPES.get(declared, declared) != dtype:
                raise ContractError(f"reference {name} dtype {dtype} disagrees with contract {declared}")
            if "shape" in spec:
                shape = tuple(_dimension(dim, case["parameters"]) for dim in spec["shape"])
                if array.shape != shape:
                    raise ContractError(f"reference {name} shape {array.shape} disagrees with contract {shape}")
        result[name] = {"shape": list(array.shape), "dtype": dtype, "sha256": hashlib.sha256(raw).hexdigest(), "comparison": rule}
    return result


def validate_candidates(expected: dict, contract: dict, candidates: dict | None) -> tuple[dict, dict]:
    """Validate independent per-position choices before observing actual outputs."""
    import numpy as np

    names = {name for name in expected if comparison_rule(contract, name)["mode"] == "one_of"}
    if candidates is None:
        candidates = {}
    if not isinstance(candidates, dict) or set(candidates) != names:
        raise ContractError(f"reference_candidates names must be {sorted(names)}")
    arrays, descriptions = {}, {}
    for name in sorted(names):
        values = candidates[name]
        rule = comparison_rule(contract, name)
        if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= rule["max_candidates"]:
            raise ContractError(f"reference_candidates {name}: invalid candidate count")
        reference, dtype, _ = _array(expected[name])
        choices, hashes = [], []
        for value in values:
            array, candidate_dtype, raw = _array(value)
            if array.shape != reference.shape or candidate_dtype != dtype or not np.isfinite(array).all():
                raise ContractError(f"reference_candidates {name}: invalid shape, dtype or nonfinite value")
            choices.append(array)
            hashes.append(hashlib.sha256(raw).hexdigest())
        stacked = np.stack(choices)
        if not np.all(np.any(stacked == reference, axis=0)):
            raise ContractError(f"reference_candidates {name}: nominal reference is not included")
        arrays[name] = stacked
        descriptions[name] = {"count": len(values), "sha256": hashes,
                              "ambiguous_elements": int(np.count_nonzero(np.any(stacked != stacked[0], axis=0)))}
    return arrays, descriptions


def compare_outputs(actual: dict, expected: dict, contract: dict, stage: bool = False,
                    *, candidates: dict | None = None) -> dict:
    import numpy as np

    _check_named(actual, set(expected), "execution outputs")
    choices, descriptions = ({}, {}) if stage else validate_candidates(expected, contract, candidates)
    result = {}
    for name, reference in expected.items():
        a, adtype, abytes = _array(actual[name])
        e, edtype, ebytes = _array(reference)
        rule = comparison_rule(contract, name, stage)
        if "reference_dtype" in rule:
            specs = contract["stages" if stage else "outputs"]
            declared_actual = specs[name]["dtype"]
            declared_actual = DTYPES.get(declared_actual, declared_actual)
            if adtype != declared_actual or edtype != rule["reference_dtype"]:
                raise ContractError(f"{name}: actual dtype {adtype} and reference dtype {edtype} disagree with "
                                    f"declared {declared_actual} and {rule['reference_dtype']}")
        if a.shape != e.shape or ("reference_dtype" not in rule and adtype != edtype):
            raise ContractError(f"{name}: actual {adtype}{a.shape} != reference {edtype}{e.shape}")
        if rule["mode"] == "bitwise":
            ok = abytes == ebytes
        elif rule["mode"] == "exact":
            ok = bool(np.array_equal(a, e, equal_nan=rule.get("equal_nan", False)))
        elif rule["mode"] == "one_of":
            ok = bool(np.all(np.any(choices[name] == a, axis=0)))
        else:
            ok = bool(np.allclose(a, e, rtol=rule["rtol"], atol=rule["atol"], equal_nan=rule.get("equal_nan", False)))
        with np.errstate(invalid="ignore", over="ignore"):
            delta = np.abs(a.astype(np.complex128 if np.iscomplexobj(a) else np.float64) - e)
            finite = delta[np.isfinite(delta)]
            max_abs = float(finite.max()) if finite.size else None
        result[name] = {"passed": ok, "comparison": rule, "max_abs_error": max_abs}
        if rule["mode"] == "one_of":
            result[name].update(descriptions[name], non_nominal_elements=int(np.count_nonzero(a != e)))
        if "reference_dtype" in rule:
            result[name].update(actual_dtype=adtype, reference_dtype=edtype)
        if "max_relative_l2" in rule:
            expected_norm = float(np.linalg.norm(e.astype(np.complex128 if np.iscomplexobj(e) else np.float64).reshape(-1)))
            residual_norm = float(np.linalg.norm(delta.reshape(-1)))
            ratio = residual_norm / expected_norm if expected_norm else (0.0 if residual_norm == 0 else math.inf)
            ok = ok and math.isfinite(ratio) and ratio <= rule["max_relative_l2"]
            result[name].update(passed=ok, relative_l2_error=ratio if math.isfinite(ratio) else None)
        if not ok:
            raise ContractError(f"{name}: {rule['mode']} comparison failed; max finite absolute error {max_abs}")
    return result


def _load_unit(root: Path, contract: dict) -> Any:
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("unit", root / "unit.py")
    if spec is None or spec.loader is None:
        raise ContractError("unit.py could not be loaded")
    unit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = unit
    spec.loader.exec_module(unit)
    required = ["make_inputs", "reference", "execute"]
    if contract["kind"] == "project":
        required += ["reference_stages", "execute_stages"]
    for name in required:
        if not callable(getattr(unit, name, None)):
            raise ContractError(f"unit.py requires callable {name}")
    if callable(getattr(unit, "reference_stages", None)) != callable(getattr(unit, "execute_stages", None)):
        raise ContractError("stage reference and execution hooks must be supplied together")
    return unit


def _model_processes(options: dict) -> bool:
    mode = options.get("sim_processes", "threads")
    if mode not in ("threads", "fork"):
        raise ContractError("sim_processes must be threads or fork")
    if mode == "fork":
        import multiprocessing

        if options["launcher"] not in ("sim", "pipesim"):
            raise ContractError("--sim-processes fork applies only to sim or pipesim")
        if "fork" not in multiprocessing.get_all_start_methods():
            raise ContractError("this host does not support forked model core groups")
    return mode == "fork"


def launch_kernel(kernel: Any, args: tuple, options: dict) -> Any:
    """Launch actual kernel code; preserve initialized outputs and separate model evidence.

    Unit execute hooks may use this helper. It adds concise per-kernel records to
    options['_execution_evidence']; callers still return only their named outputs.
    """
    launcher = options["launcher"]
    processes = _model_processes(options)
    evidence = {"kernel": kernel.name, "device": options["device"], "requested_backend": options["backend"], "stage": LAUNCHER_STAGES[launcher], "launcher": launcher, "block_dim": options["block_dim"]}
    if launcher == "sim":
        from ascriptor.backends.sim.launch import run_kernel

        result = run_kernel(kernel, *args, block_dim=options["block_dim"], timeout=options["timeout"], seed_outputs=True, processes=processes)
        evidence.update(model="functional Surface IR", backend_executed=False, sim_processes="fork" if processes else "threads")
    elif launcher == "pipesim":
        from ascriptor.backends.sim.pipesim import simulate
        from ascriptor.passes import PIPELINE, PassManager
        from ascriptor.passes.autosync import check_balance

        lowered = PassManager(PIPELINE).run(kernel.ir())
        balance = check_balance(lowered)
        if balance:
            raise ContractError(f"{kernel.name}: event balance failed: {balance}")
        simulation = simulate(lowered, args, block_dim=options["block_dim"], timeout=options["timeout"], seed_outputs=True, check_gm=True, processes=processes)
        if simulation.hazards or simulation.report.get("deadlock"):
            raise ContractError(f"{kernel.name}: pipe simulation hazards/deadlock: {simulation.report}")
        result = simulation.outputs[0] if len(simulation.outputs) == 1 else simulation.outputs
        evidence.update(model="lowered events and hazards", backend_executed=False, event_balance=balance, hazards=simulation.hazards, deadlock=simulation.report.get("deadlock"), model_cycles=simulation.cycles, sim_processes="fork" if processes else "threads")
    else:
        from ascriptor.runtime import OpExec

        executor = OpExec(kernel, launcher=launcher, backend=options["backend"], device=options["device"], board=options["board"], out_dir=Path(options["out_dir"]) / kernel.name, block_dim=options["block_dim"], timeout=options["timeout"], seed_outputs=True)
        result = executor(*args)
        evidence.update(backend_executed=True)
    options.setdefault("_execution_evidence", []).append(evidence)
    return result


def _versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("ascriptor", "torch", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(root: Path, argv: list[str] | None = None) -> dict:
    contract = validate_contract(json.loads((root / "contract.json").read_text(encoding="utf-8")))
    first = contract["support"][0]
    parser = argparse.ArgumentParser(description=contract["description"])
    parser.add_argument("command", choices=("reference", "check", "profile"))
    parser.add_argument("--case", default="all")
    parser.add_argument("--device", default=first["device"])
    parser.add_argument("--backend", choices=("cce", "pypto_pro", "pto_isa"), default=first["backend"])
    parser.add_argument("--launcher", choices=tuple(LAUNCHER_STAGES), default="sim")
    parser.add_argument("--board")
    parser.add_argument("--block-dim", type=int)
    parser.add_argument("--sim-processes", choices=("threads", "fork"), default="threads")
    parser.add_argument("--probe-gaps", action="store_true", help="Attempt declared support gaps again with the selected backend")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=root / "tmp" / "run")
    parsed = parser.parse_args(argv)
    if not math.isfinite(parsed.timeout) or parsed.timeout <= 0:
        raise ContractError("timeout must be finite and positive")
    if parsed.command != "reference":
        _model_processes({"sim_processes": parsed.sim_processes, "launcher": parsed.launcher})
        if parsed.launcher == "pypto" and parsed.backend != "pypto_pro":
            raise ContractError("launcher pypto requires backend pypto_pro")
        if parsed.backend == "pypto_pro" and parsed.launcher not in ("pypto", "sim", "pipesim"):
            raise ContractError("backend pypto_pro requires launcher pypto for runtime execution")
    stage = "reference" if parsed.command == "reference" else LAUNCHER_STAGES[parsed.launcher]
    rows = [row for row in contract["support"] if (row["device"], row["backend"], row["stage"]) == (parsed.device, parsed.backend, stage)]
    if not rows:
        raise ContractError(f"undeclared combination: {parsed.device}/{parsed.backend}/{stage}")
    cases = [case for case in contract["cases"] if parsed.case in ("all", case["id"])]
    if not cases:
        raise ContractError(f"unknown case {parsed.case!r}; no cases selected")
    scoped = {case["id"]: [row for row in rows if "cases" not in row or case["id"] in row["cases"]]
              for case in cases}
    if any(not matching for matching in scoped.values()):
        missing = [name for name, matching in scoped.items() if not matching]
        raise ContractError(f"undeclared cases for {parsed.device}/{parsed.backend}/{stage}: {missing}")
    gap_cases = [name for name, matching in scoped.items() if all(row["status"] == "gap" for row in matching)]
    # A recorded failure remains reproducible. Runs never edit canonical support rows.
    report = {"schema": "ascriptor.unit-result/1", "unit": contract["id"], "command": parsed.command, "stage": stage, "device": parsed.device, "backend": parsed.backend, "launcher": parsed.launcher if parsed.command != "reference" else None, "versions": _versions(), "contract_sha256": hashlib.sha256((root / "contract.json").read_bytes()).hexdigest(), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "started_at": datetime.now(timezone.utc).isoformat(), "cases": [], "passed": False}
    try:
        if not parsed.probe_gaps and gap_cases:
            rows = [row for row in rows if any(row in scoped[name] for name in gap_cases)]
            reason = "; ".join(row["reason"] for row in rows)
            proven = all(row.get("owner") == "upstream" and row.get("gap_id") and row.get("evidence") for row in rows)
            report["backend_gap"] = {"type": "DeclaredSupportGap", "owner": "upstream" if proven else "unmapped",
                                     "reason": reason, "gap_ids": [row["gap_id"] for row in rows if row.get("gap_id")],
                                     "evidence": [row["evidence"] for row in rows if row.get("evidence")],
                                     "cases": gap_cases,
                                     "location": "contract.json/support", "opcode": rows[0].get("opcode")}
            raise ContractError("located support gap: " + reason)
        unit = _load_unit(root, contract)
        for declared_case in cases:
            block_dim = validate_block_dim(contract, parsed.block_dim if parsed.block_dim is not None else declared_case.get("block_dim", 1))
            case = {**copy.deepcopy(declared_case), "block_dim": block_dim}
            options = {"device": parsed.device, "backend": parsed.backend, "launcher": parsed.launcher, "board": parsed.board, "out_dir": str((parsed.output / case["id"]).resolve()), "block_dim": block_dim, "timeout": parsed.timeout, "sim_processes": parsed.sim_processes}
            inputs = unit.make_inputs(case)
            if not isinstance(inputs, dict) or not inputs:
                raise ContractError("make_inputs must return a nonempty dictionary")
            if callable(getattr(unit, "validate_inputs", None)):
                unit.validate_inputs(inputs, case)
            before = fingerprint(inputs)
            expected = unit.reference(inputs)
            if fingerprint(inputs) != before:
                raise ContractError("reference mutated its inputs")
            if callable(getattr(unit, "validate_reference", None)):
                unit.validate_reference(inputs, expected, case)
            if fingerprint(inputs) != before:
                raise ContractError("validate_reference mutated its inputs")
            entry = {"id": case["id"], "seed": case["seed"], "parameters": case["parameters"], "block_dim": block_dim, "reference": describe_outputs(expected, contract, case), "passed": False}
            report["cases"].append(entry)
            candidates = None
            if any(comparison_rule(contract, name)["mode"] == "one_of" for name in expected):
                hook = getattr(unit, "reference_candidates", None)
                if not callable(hook):
                    raise ContractError("one_of requires callable reference_candidates")
                candidates = hook(inputs)
                if fingerprint(inputs) != before:
                    raise ContractError("reference_candidates mutated its inputs")
                _, entry["reference_candidates"] = validate_candidates(expected, contract, candidates)
            stage_expected = None
            if callable(getattr(unit, "reference_stages", None)):
                stage_expected = unit.reference_stages(inputs)
                if fingerprint(inputs) != before:
                    raise ContractError("reference_stages mutated its inputs")
                entry["stage_reference"] = describe_outputs(stage_expected, contract, case, stage=True)
            if parsed.command != "reference":
                def fresh(case=case, expected_fingerprint=before, reference_inputs=inputs) -> dict:
                    generated = unit.make_inputs(case)
                    if fingerprint(generated) != expected_fingerprint:
                        raise ContractError("make_inputs is not deterministic for the case seed")
                    if _storage_ids(generated) & _storage_ids(reference_inputs):
                        raise ContractError("make_inputs reused reference input storage; executions require fresh inputs")
                    return generated

                if stage_expected is not None:
                    actual_stages = unit.execute_stages(fresh(), options)
                    entry["stage_comparison"] = compare_outputs(actual_stages, stage_expected, contract, stage=True)
                actual = unit.execute(fresh(), options)
                entry["comparison"] = compare_outputs(actual, expected, contract, candidates=candidates)
                if parsed.command == "profile":
                    profile = contract.get("profile", {})
                    warmup, repeat = profile.get("warmup"), profile.get("repeat")
                    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0 or isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
                        raise ContractError("profile requires declared nonnegative warmup and positive repeat")
                    for _ in range(warmup):
                        unit.execute(fresh(), options)
                    seconds = []
                    for _ in range(repeat):
                        generated = fresh()
                        start = time.perf_counter()
                        unit.execute(generated, options)
                        seconds.append(time.perf_counter() - start)
                    entry["profile"] = {"warmup": warmup, "repeat": repeat, "host_wall_seconds": seconds, "median_host_wall_seconds": statistics.median(seconds), "scope": "whole execute hook including launch overhead; not device kernel latency", "timing_kind": "simulation" if parsed.launcher in ("sim", "pipesim", "cannsim") else "runtime_host_wall"}
                entry["execution_evidence"] = options.get("_execution_evidence", [])
            entry["passed"] = True
            _write_json(parsed.output / case["id"] / "result.json", entry)
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        # A failed backend translation is not a failed numerical comparison,
        # nor does a missing mapping prove an upstream limitation.
        if type(error).__name__ in ("CceGap", "PtoIsaGap", "PyptoGap"):
            op = getattr(error, "op", None)
            report["backend_gap"] = {
                "type": type(error).__name__, "owner": getattr(error, "owner", "") or "unmapped",
                "opcode": getattr(op, "opcode", None), "reason": getattr(error, "why", str(error)),
                "location": str(getattr(op, "loc", "") or ""),
            }
        raise
    finally:
        report["imports"] = {name: str(getattr(module, "__file__", "")) for name, module in sorted(sys.modules.items()) if name == "ascriptor" or name in {"ascriptor.a2", "ascriptor.a3", "ascriptor.a5", "unit", "ref", "kernels"}}
        report["dsl_imported"] = any(name.startswith("ascriptor.frontend") for name in sys.modules)
        report["kernel_imported"] = "kernels" in sys.modules or any(name.startswith("kernels.") for name in sys.modules)
        report["simulator_imported"] = any(name.startswith("ascriptor.backends.sim") for name in sys.modules)
        if parsed.command == "reference" and (report["dsl_imported"] or report["kernel_imported"] or report["simulator_imported"]):
            report["passed"] = False
            report["error"] = "reference imported DSL, kernel or simulator modules; reference unit imports must be lazy"
        _write_json(parsed.output / "summary.json", report)
    if not report["passed"]:
        raise ContractError(report["error"])
    return report


def main(root: str | Path | None = None, argv: list[str] | None = None) -> int:
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    try:
        report = run(root, argv)
    except Exception as error:  # noqa: BLE001 - the CLI serializes any failure and returns a nonzero status.
        print(json.dumps({"passed": False, "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ("unit", "command", "stage", "versions", "passed")} | {"cases": len(report["cases"])}))
    return 0
