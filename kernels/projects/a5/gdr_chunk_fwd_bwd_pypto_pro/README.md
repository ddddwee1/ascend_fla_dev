# Gated delta rule, chunked forward and backward, A5

Training forward and backward for the gated delta rule — the
`fla.ops.gated_delta_rule` family, with one **scalar** forget gate per token and
head. This is not GDN-2, whose gate is per channel.

The operator is delivered as **nine kernels**: eight that compose into the
training forward and backward, and one that is the standalone inference forward.
Each kernel is a single launch of a single fused cube/vector program; the eight
are sequenced in Python with nothing between them but GM tensors.

```
gdr_fwd_train_wy  ->  gdr_fwd_train_scan  ->  gdr_bwd_dv_local  ->  gdr_bwd_reverse_scan
                  ->  gdr_bwd_dqkw  ->  gdr_bwd_wy_matmuls  ->  gdr_bwd_wy_reduce
                  ->  gdr_bwd_finalize  ->  (dq, dk, dv, dg, dbeta)
```

Read [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for the mathematics and why the
work is split this way, and [`docs/KERNELS.md`](docs/KERNELS.md) for the
per-kernel contract, grid and on-chip budget.

## This is not an ascriptor unit

AGENTS.md §9 describes `kernels/projects/<soc>/` as holding this repository's own
*ascriptor* units, which follow the unit protocol: `unit.py` exporting
`make_inputs` / `reference` / `execute`, plus `contract.json` and `run.py`. This
directory does not follow that protocol, because it is not ascriptor code:

- it is written in **PyPTO-Pro** (`pypto` 0.2.1), a different kernel DSL, whose
  programs are Python functions decorated with `@pl.jit` and `@pl.vector_function`;
- it was developed in another repository, under that repository's own workflow,
  and that remains its authoritative home;
- there is therefore no `unit.py` / `contract.json` / `run.py`, and
  `tools/unit_runner.py` cannot execute it.

Nothing in this repository imports it, so it can be moved without consequence if
`kernels/projects/a5/` turns out to be the wrong home for a non-ascriptor unit.

## Layout

| path | contents |
|---|---|
| `kernels/` | the nine kernel implementations, each with its host entry point, plus `gated_delta_rule_train.py`, the composed training step |
| `tests/` | per-kernel acceptance tests, the end-to-end composition test, the shared precision engine, and a float32 CPU and NPU golden per kernel |
| `docs/ALGORITHM.md` | the mathematics, the chunked reformulation, and the decomposition |
| `docs/KERNELS.md` | per-kernel reference: contract, grid, on-chip budget |
| `docs/VALIDATION.md` | what correctness is judged against, with the numbers |
| `docs/PERFORMANCE.md` | measured latency, how it was measured, and the arithmetic floor |

## Supported domain

| axis | supported | enforced by |
|---|---|---|
| batch `B` | 1 | `gdr_bwd_dqkw` pins the batch axis of every tensor to 1 |
| sequence `T` | any; exercised to 16384 | internally padded to a multiple of 64 |
| heads `H` | up to 32; exercised to 16 | `gdr_fwd_train_wy` asserts `H <= 32` |
| `K`, `V` | multiples of 64 up to 128 | `gdr_fwd_train_wy` asserts it |
| dtypes | `q`, `k`, `v`, `beta` bfloat16; `g` float32; all arithmetic float32 | kernel signatures |
| chunk `C` | 64, fixed | compile-time constant of the family |

`g` is a log-space decay and must be non-positive; `beta` is passed as sigmoid
logits and activated inside the kernel. The gradients `dq`, `dk`, `dv` come back
in bfloat16, `dg` and `dbeta` in float32.

Outside that domain the operator raises rather than degrading: an out-of-range
`K` or `H` fails its assertion in the host entry, and `B > 1` fails tensor
validation before any kernel launches.

## Running it

Needs an Ascend 950PR with CANN and `torch_npu`. The tests take the device from
`TILE_FWK_DEVICE_ID`, or `ASCEND_RT_VISIBLE_DEVICES` as the environment usually
sets it.

```sh
# end to end: all eight operators chained, against the autograd oracle
python tests/test_composition.py

# the composed and pipelined training step, against the shipped chain
python tests/test_train_entry.py

# the same chain, reporting per-kernel device time
python tests/test_composition.py --timing --warmup 3 --repeats 1

# one operator at a time
python tests/test_gdr_fwd_train_wy.py
python tests/test_gdr_bwd_dqkw.py aligned      # a subset, by case-name prefix

# the standalone inference forward
python tests/test_gated_delta_rule_fwd.py

# semantics against fla's reference implementation, on CPU
python tests/validate_against_fla.py

# the same reference on torch_npu, as a performance baseline
python tests/fla_npu_baseline.py
```

A test imports its kernel from `../kernels` and its goldens from its own
directory, so the directory is self-contained: copy it anywhere and the commands
above still work.

To run training steps, call
`gated_delta_rule_train.gated_delta_rule_fwd_bwd_pipelined(batches)` rather than
chaining the eight wrappers by hand. It takes one independent step per batch --
the microbatches of an accumulation window -- and issues them back to back on
one device, with a single synchronisation for the whole sequence and the two
independent branches of each step on separate streams. That is **1.95 times cheaper per step** than
the shipped chain (828.5 µs against 1617.5 µs at the reference shape) and
bit-identical to it. Queue the whole accumulation window in one call: four steps
a call gives only 1.77 times, because every call pays for filling and draining
the pipeline. All of
the concurrency is streams within one device; nothing here spans devices, and a
batch whose tensors are not all on the same device is rejected.

`gated_delta_rule_fwd_bwd` runs a single step and synchronises, for when one
isolated step is what you want; it costs about 9% more per step.
`docs/PERFORMANCE.md` has the measurements.

## Where it stands

Correctness is judged in float32 against two independent references, and the
chunk path is **bit-identical** to `fla`'s `naive_chunk_gated_delta_rule`. The
composed chain passes all five shapes, including the production shape, with all
thirty intermediate tensors clean against the per-stage goldens.
[`docs/VALIDATION.md`](docs/VALIDATION.md) has the numbers, and the three gaps
that are not closed.

At B=1, T=2048, H=8, K=V=128 the eight kernels take **912 µs** of device time;
the forward alone is **4.87 ms** at T=16384, H=16, which is 2.41 times its
float32 arithmetic floor on this part.
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) has the per-kernel breakdown, the
method, and what the numbers do and do not license.
