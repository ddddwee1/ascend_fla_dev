"""Observe ATen host work around actual NPU public calls after preparation."""
import argparse
import json
from pathlib import Path
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ascend_fla.ops.gdn_fused_recurrent import BLOCK_DIMS, fused_recurrent_gdn, prepare
from kernels.projects.a5.gdn_fused_recurrent.ref.oracle import oracle
from kernels.projects.a5.gdn_fused_recurrent.ref.reference import digest, make_inputs, reference
from kernels.projects.a5.gdn_fused_recurrent.verify_native import compare


class Observe(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        repo = Path(__file__).resolve().parents[4]
        sites = [dict(file=str(Path(f.filename).relative_to(repo)), line=f.lineno, function=f.name)
                 for f in traceback.extract_stack()[:-1] if Path(f.filename).is_relative_to(repo)]
        self.operations.append(dict(operator=str(func), shape=list(args[0]),
                                    dtype=str(kwargs.get('dtype')), call_sites=sites))
        return func(*args, **(kwargs or {}))


def check_operations(operations, expected_allocations):
    assert all(x['operator']=='aten.empty.memory_format' for x in operations), operations
    workspace = [x for x in operations if any(s['file']=='ascend_fla/runtime/binding.py'
                 and s['function']=='_workspace' for s in x['call_sites'])]
    assert len(workspace) <= 1 and all(x['dtype']=='torch.uint8' for x in workspace)
    assert len(operations)-len(workspace)==expected_allocations, operations
    return len(workspace)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--block-dim', type=int, choices=BLOCK_DIMS, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch_npu  # noqa: F401
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    prepare(block_dim=a.block_dim)
    a.output.mkdir(parents=True, exist_ok=True)
    assert not (a.output/'summary.json').exists()
    records = []
    for dtype in ('float32', 'bfloat16'):
        for initial in ('none', 'random'):
            case = dict(id=f'{dtype}_{initial}', B=1, S=16, H=16, HV=32,
                        dtype=dtype, state=initial, seed=943100)
            cpu = make_inputs(case)
            data = {n: None if v is None else v.to('npu') for n,v in cpu.items()}
            before = {n: None if v is None else digest(v) for n,v in data.items()}
            with Observe() as observation:
                o, state = fused_recurrent_gdn(**data, block_dim=a.block_dim, output_final_state=True)
            torch.npu.synchronize()
            actual = dict(o=o.cpu(), final_state=state.cpu())
            comparison = compare(actual, dict(A=oracle(cpu), B=reference(cpu)))
            expected_allocations = 3 if initial == 'none' else 2
            workspace_allocations = check_operations(observation.operations, expected_allocations)
            assert comparison['passed']
            assert before == {n: None if v is None else digest(v) for n,v in data.items()}
            assert all(state.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
                       for v in data.values() if v is not None)
            with Observe() as hidden_observation:
                hidden_o, hidden_state = fused_recurrent_gdn(**data, block_dim=a.block_dim, output_final_state=False)
            torch.npu.synchronize()
            assert hidden_state is None
            assert digest(hidden_o) == digest(o)
            hidden_workspace = check_operations(hidden_observation.operations, expected_allocations)
            assert hidden_workspace == 0, 'Binding must reuse its per-device workspace'
            record = dict(case=case, operations=observation.operations,
                          output_final_state_false_operations=hidden_observation.operations,
                          expected_allocations=expected_allocations, unexpected=[],
                          additional_binding_workspace_allocations=workspace_allocations,
                          output_flag_checked=True, comparison=comparison, passed=True)
            records.append(record)
            print(json.dumps(record), flush=True)
    report = dict(stage='native_public_host_work_audit', block_dim=a.block_dim,
        observed='ATen operations within prepared public inprocess calls; CCE execution is outside ATen.',
        excluded='Preparation, input generation, H2D, CPU oracle execution and result copies are outside observation.',
        records=records, passed=True)
    (a.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
