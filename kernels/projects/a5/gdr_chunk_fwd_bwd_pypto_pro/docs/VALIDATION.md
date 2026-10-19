# Validation

Every number in this document was produced by the code in this directory, on an
Ascend 950PR (28 cube and 56 vector cores, DAV_3510, CANN 9.1.0-beta.1).

Correctness is judged in float32. A bfloat16 element-wise comparison has no
discriminating power, so the bfloat16 public outputs are compared as float32
against a float32 reference, and the dtype contract is checked separately.

## What it is judged against

**The semantic authority is `fla`.** `fla.ops.gated_delta_rule.naive` is pure
torch and is what defines the operator; our goldens were written against it and
are checked against it rather than against themselves. `tests/validate_against_fla.py`
runs that comparison on CPU in float32, passing identical pre-activated `beta`
and log-space `g` to both, over four shapes — `(1,128,2,64,64)`,
`(1,192,2,128,128)`, `(1,256,4,128,128)` and `(2,128,3,128,96)`:

| comparison | max abs | rel L2 |
|---|--:|--:|
| our chunk path vs `naive_chunk_gated_delta_rule` | **0.000e+00** | **0.000e+00** |
| our recurrent final state vs `naive_recurrent_gated_delta_rule` | **0.000e+00** | **0.000e+00** |
| our recurrent output `o` vs `fla` | 2.1e-4 – 4.2e-4 | 2.0e-7 – 4.6e-7 |
| our chunk path vs `fla` recurrent, across paths | 1.7e-4 – 3.1e-3 | 2.1e-6 – 3.7e-6 |

Bit-identical on the chunk path and on the recurrent state. The only deviation
is the output projection, where we write a broadcast product and a sum and `fla`
writes an `einsum`: a float32 reduction order, not a semantic difference, which
the bit-exact state confirms. The chunk and recurrent paths agreeing to ~3e-6 is
the cross-path check, each being the other's best oracle.

The backward golden is `torch.autograd.grad` of that forward oracle, so it
inherits the validation. `fla` has no separate backward reference; it also
obtains gradients by autograd.

**The second reference runs on the device.** `fla`'s `naive_chunk` is pure
torch, so it runs under `torch_npu` and gives an independent implementation on
the same part. At `(1,192,2,128,128)` it agrees with `naive_recurrent` on CPU to
`max_abs 5.615e-03`, `rel L2 1.903e-06` against a peak of 2.783e+03 — the same
decade as our own cross-path figure, so the two references corroborate each
other rather than merely coexisting. `tests/fla_npu_baseline.py` runs it.

## The goldens in this directory

Per kernel there is a float32 CPU golden and an NPU golden. The CPU golden is
the acceptance reference; the NPU golden exists to separate a numerical
difference from a layout or dtype difference.

There are also cumulative staging goldens — `gated_delta_rule_bwd_golden_stage*`
— which compute the same chain in torch, one operator boundary at a time. They
are what makes a failure localisable: a divergence is attributed to the first
boundary that breaks, not to "the chain is wrong".

## The acceptance criterion

`tests/precision_compare.py` is the single judge. A tensor passes if either

* `torch.allclose(actual, reference, rtol=1e-2, atol=1e-2)`, or
* the peak-scaled criterion: element-wise
  `|delta| <= atol_floor + rtol * |reference|` with at least 99% of elements
  matched, and global relative L2 residual `<= 2e-2`. For the bfloat16-stored
  outputs `rtol = 5e-2` and `atol_floor = 5e-2 * peak`; for the float32 outputs
  `1e-2` and `1e-2 * peak`.

The second form is there because a fixed absolute tolerance is meaningless
against a tensor whose peak is 1e6: one float32 unit in the last place at that
magnitude already exceeds 1e-2. Thresholds come from that file, not from the
individual tests, and every comparison prints its numbers rather than a verdict.

## The composition gate

Eight correct operators can still be wired together wrongly, so component tests
are necessary and not sufficient. `tests/test_composition.py` chains all eight,
materialising every boundary in GM, and checks the six public outputs against
the autograd oracle plus all thirty intermediate tensors against the staging
goldens.

Five shapes pass, including the production shape, with 185 comparisons clean:

| case | B | T | H | K | V |
|---|--:|--:|--:|--:|--:|
| `aligned_b1_t128_h4_k64_v64` | 1 | 128 | 4 | 64 | 64 |
| `ttail_b1_t100_h4_k64_v64` | 1 | 100 | 4 | 64 | 64 |
| `t3c_b1_t192_h2_k128_v128` | 1 | 192 | 2 | 128 | 128 |
| `tail_vlt_b1_t200_h3_k128_v96` | 1 | 200 | 3 | 128 | 96 |
| `production_b1_t2048_h8_k128_v128` | 1 | 2048 | 8 | 128 | 128 |

The production case takes the same code path as the other four: no relaxed
tolerance, no separate branch.

At the production shape, the six public outputs:

| output | matched | rel L2 | max abs | peak |
|---|--:|--:|--:|--:|
| `o` | 1.000000 | 4.568e-03 | 1.979e+03 | 4.628e+05 |
| `dq` | 1.000000 | 1.698e-03 | 9.883e+02 | 3.483e+05 |
| `dk` | 1.000000 | 1.644e-03 | 2.285e+03 | 1.079e+06 |
| `dv` | 1.000000 | 1.512e-03 | 9.460e+02 | 3.943e+05 |
| `dg` | 1.000000 | 1.227e-06 | 2.250e+00 | 1.489e+06 |
| `dbeta` | 1.000000 | 1.168e-06 | 9.141e-01 | 7.736e+05 |

`o`, `dq`, `dk` and `dv` sit three orders of magnitude worse than `dg` and
`dbeta` because they are stored in bfloat16, whose relative resolution is about
4e-3 — that is the dominant term in those four rows, not the arithmetic. `o` and
`dg` pass on the peak-scaled criterion rather than plain `allclose`, for the
reason given above.

The thirty boundaries are clean at 1e-8 to 7e-6 relative L2 in float32, with
five of them exactly zero (`q_s`, `k_f32`, `g_f32`, `q_t` and their derived
rows), which is what one expects of a reindex and a widen.

## The composed entry point

`tests/test_train_entry.py` gates `kernels/gated_delta_rule_train.py`, and its
criterion is **exact equality**, not a tolerance. The entry point issues the same
eight launches as the composition gate chains by hand; only the host-side
ordering, the stream assignment and the core budgets differ, and none of those
touches the arithmetic. So anything other than bit-identical output is a defect,
and the test asserts `torch.equal` on all six public results against the chain
as the wrappers ship it.

It covers three shapes (`B=1` with `T=128 H=4 K=V=64`, `T=192 H=2 K=V=128`, and
the production `T=2048 H=8 K=V=128`) across six configurations each: the
sequential order, the concurrent order at the computed budgets, and four
`(scan, reverse)` budget pairs that under-commit, exactly commit and
over-commit the 28 cube cores.

It then gates the pipelined entry point, which is the shipped path. Four steps
queued on two alternating streams with no per-step synchronisation must each
equal the single-step reference; and
`gated_delta_rule_fwd_bwd_pipelined` is run over **five steps with distinct
per-step inputs** at one, two and three streams in flight, each step compared
against its own single-step result. Distinct inputs matter here: with the same
batch repeated, a stream hazard or an ordering mistake would return the right
answer anyway. The degenerate sequences are covered too -- one step, and none --
as is the rejection of a batch whose tensors do not all live on one device,
which is skipped with a logged note when only one device is visible.

Thirty-eight equality checks in total.

## What is not validated

Three gaps, stated rather than implied:

* **Correctness has never been checked at T=16384, H=16**, the shape whose
  latency `PERFORMANCE.md` quotes. The largest validated shape is T=2048, H=8.
  The CPU oracle at the large shape is expensive, not impossible; it has not
  been run.
* **There is no prefill-to-decode consistency test.** The chunked forward and a
  token-by-token recurrence with state hand-off should agree, and that is the
  natural check on a state-carrying operator. The inference forward here has no
  decode counterpart to compare against.
* **The batch axis is untested beyond B=1**, because `gdr_bwd_dqkw` pins it.
  Several components do support B>1 individually — `gdr_fwd_train_scan` passes a
  B=3 case, and `dv_local`, `wy_reduce` and `finalize` take a dynamic batch —
  but the pipeline is gated by the one that does not.

## Reproducing

```sh
python tests/test_composition.py        # the gate above
python tests/validate_against_fla.py    # the semantic authority, on CPU
python tests/fla_npu_baseline.py        # the second reference, on device
python tests/test_gdr_bwd_dqkw.py       # or any single operator
```
