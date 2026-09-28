"""State-continuation workload and checks, independent of the device kernel.

The CLI exercises the checks with CPU B and the existing CPU chunk block-solve
reference. It does not run chunk_gdn, a decode kernel, or the pending CPU A.
Native acceptance must reuse these checks on actual returned device results.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from .reference import NAMES, OUTPUTS, S_MAX, cases, digest, make_inputs, metric, reference


def integration_cases():
    templates = [dict(id="full", B=1, H=16, HV=32, prefix=128)]
    templates.extend(dict(id=f"r{ratio}", B=1, H=3, HV=3*ratio, prefix=64)
                     for ratio in (1, 2, 4, 8))
    templates.extend([
        dict(id="batch2", B=2, H=2, HV=8, prefix=128),
        dict(id="length4096", B=1, H=1, HV=8, prefix=4032),
    ])
    return [dict(row, id=row["id"]+"_"+dtype, dtype=dtype,
                 S=row["prefix"]+64, state="none", seed=941000+i)
            for i, row in enumerate(templates)
            for dtype in ("float32", "bfloat16")]


def partitions(total, width):
    if type(total) is not int or total < 1:
        raise ValueError("positive integer step count required")
    if type(width) is not int or not 1 <= width <= S_MAX:
        raise ValueError("decode call width must be 1..16")
    return [(start, min(start+width, total)) for start in range(0, total, width)]


def slice_inputs(inputs, start, stop, state):
    if not 0 <= start < stop <= inputs["q"].shape[1]:
        raise ValueError("nonempty token slice within inputs required")
    result = {name: inputs[name][:, start:stop].contiguous() for name in NAMES[:5]}
    result["initial_state"] = state
    return result


def _fingerprint(inputs):
    return {name: None if value is None else digest(value)
            for name, value in inputs.items()}


def validate_returned(inputs, result, *, output_dtype):
    """Reject broadcasting, omitted outputs, storage aliases and bad metadata."""
    if set(result) != set(OUTPUTS):
        raise ValueError("exactly o and final_state must be returned")
    b, s, hv, _ = inputs["v"].shape
    shapes = {"o": (b, s, hv, 128), "final_state": (b, hv, 128, 128)}
    for name, value in result.items():
        dtype = output_dtype if name == "o" else torch.float32
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shapes[name]
                or value.dtype != dtype or value.device.type != "cpu"
                or not value.is_contiguous() or not bool(torch.isfinite(value).all())):
            raise ValueError(f"{name}: invalid returned shape/dtype/storage/value")
        for source in inputs.values():
            if source is not None and value.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
                raise ValueError(f"{name}: returned storage aliases an input")


def run_checked(inputs, call, *, output_dtype):
    before = _fingerprint(inputs)
    result = call(inputs)
    if before != _fingerprint(inputs):
        raise ValueError("call modified its inputs or initial_state")
    validate_returned(inputs, result, output_dtype=output_dtype)
    return result


def chained_decode(inputs, call, width, *, output_dtype):
    state = inputs.get("initial_state")
    outputs = []
    calls = []
    for start, stop in partitions(inputs["q"].shape[1], width):
        data = slice_inputs(inputs, start, stop, state)
        result = run_checked(data, call, output_dtype=output_dtype)
        calls.append(dict(start=start, stop=stop, initial_state_sha256=(
            None if state is None else digest(state)),
            returned_state_sha256=digest(result["final_state"])))
        outputs.append(result["o"])
        state = result["final_state"]
    return dict(o=torch.cat(outputs, dim=1), final_state=state), calls


def trace_metrics(actual, expected, *, token_offset=0):
    if set(actual) != set(OUTPUTS) or set(expected) != set(OUTPUTS):
        raise ValueError("missing or extra output in trace")
    for name in OUTPUTS:
        if actual[name].shape != expected[name].shape:
            raise ValueError(f"{name}: trace shapes differ")
        if not bool(torch.isfinite(expected[name]).all()):
            raise ValueError(f"{name}: nonfinite reference")
    return dict(tokens=[dict(token=token_offset+t,
                             **metric(actual["o"][:, t], expected["o"][:, t]))
                        for t in range(expected["o"].shape[1])],
                final_state=metric(actual["final_state"], expected["final_state"]),
                byte_identical={name: digest(actual[name]) == digest(expected[name])
                                for name in OUTPUTS})


def trace_passed(trace, budget=1e-4):
    return all(row["finite"] and row["relative_l2"] <= budget
               for row in [*trace["tokens"], trace["final_state"]])


def calibrate_integration():
    # Read-only use of a different algorithm. Native acceptance must replace
    # this callback with the actual public chunk kernel, never a CPU fallback.
    from kernels.projects.a5.gdn_chunk_fwd.ref.reference import block_solve

    def chunk_reference(data):
        if data["initial_state"] is not None:
            raise ValueError("chunk prefill starts from None only")
        o, state = block_solve(*(data[n] for n in ("q", "k", "v", "beta", "g")))
        return dict(o=o, final_state=state)

    torch.set_num_threads(1)
    records = []
    for case in integration_cases():
        data = make_inputs(case)
        prefix = case["prefix"]
        whole = run_checked(data, chunk_reference, output_dtype=torch.float32)
        recurrent = run_checked(data, reference, output_dtype=torch.float32)
        prefill = run_checked(slice_inputs(data, 0, prefix, None), chunk_reference,
                              output_dtype=torch.float32)
        suffix = slice_inputs(data, prefix, case["S"], prefill["final_state"])
        expected = dict(o=whole["o"][:, prefix:].contiguous(), final_state=whole["final_state"])
        expected_b = dict(o=recurrent["o"][:, prefix:].contiguous(),
                          final_state=recurrent["final_state"])
        continuations = []
        for width in (1, 3, 7, 16):
            result, calls = chained_decode(suffix, reference, width, output_dtype=torch.float32)
            continuations.append(dict(width=width, calls=calls,
                vs_whole_chunk_reference=trace_metrics(result, expected, token_offset=prefix),
                vs_whole_recurrent_B=trace_metrics(result, expected_b, token_offset=prefix)))
        # Deliberately lose the prefill state. Requiring the very first token
        # to expose this guards against a sequence-average hiding the fault.
        dropped = dict(suffix, initial_state=None)
        wrong, _ = chained_decode(dropped, reference, 16, output_dtype=torch.float32)
        negative = trace_metrics(wrong, expected, token_offset=prefix)
        records.append(dict(case=case, input_sha256=_fingerprint(data),
            whole_chunk_vs_recurrent_B=trace_metrics(whole, recurrent),
            continuations=continuations, dropped_state_negative=negative,
            dropped_state_detected_at_first_token=negative["tokens"][0]["relative_l2"] > 1e-4))
        print(json.dumps(dict(case=case["id"], phase="CPU integration calibration",
                              first_token_dropped_state_l2=negative["tokens"][0]["relative_l2"])), flush=True)

    splits = []
    for case in cases():
        if case["S"] != S_MAX:
            continue
        data = make_inputs(case)
        whole = run_checked(data, reference, output_dtype=torch.float32)
        for width in (1, 3, 7):
            result, calls = chained_decode(data, reference, width, output_dtype=torch.float32)
            splits.append(dict(case=case, width=width, calls=calls,
                               vs_one_call=trace_metrics(result, whole)))

    checks = [r["whole_chunk_vs_recurrent_B"] for r in records]
    checks += [c[key] for r in records for c in r["continuations"]
               for key in ("vs_whole_chunk_reference", "vs_whole_recurrent_B")]
    return dict(kind="CPU harness calibration only; no device or CPU A acceptance",
        python=platform.python_version(), torch=torch.__version__,
        integration_cases=records, split_cases=splits,
        max_relative_l2=max(m["relative_l2"] for trace in checks
                            for m in [*trace["tokens"], trace["final_state"]]),
        calibrated=all(trace_passed(trace) for trace in checks)
            and all(r["dropped_state_detected_at_first_token"] for r in records)
            and all(all(r["vs_one_call"]["byte_identical"].values()) for r in splits))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = calibrate_integration()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("integration_cases", "split_cases")}))
    if not report["calibrated"]:
        raise SystemExit(1)
