# Performance

Measured on an Ascend 950PR, 28 cube and 56 vector cores, DAV_3510, CANN
9.1.0-beta.1. Every number below was produced by the code in this directory.

## How it was measured

One `msprof` pass per figure, single metric group `PipeUtilization`, over a
process that runs three untimed warm-up iterations and then five timed ones. The
warm-up matters: the first launch of a kernel includes its JIT compilation. The
figures are medians of the warm occurrences of each `Op Name`, taken from
`op_summary`'s `Task Duration`, which is device time.

Device time and host wall time are reported separately and never averaged
together. Nothing here is a simulator estimate.

## The composed training pipeline

B=1, T=2048, H=8, K=V=128 — the production shape. `aic_mac` is the fraction of
the kernel's window in which the cube's multiply-accumulate pipe is busy;
`cube %` is `cube_utilization`, which is core occupancy, not efficiency.

| # | kernel | µs | share | blocks | `aic_mac` | cube % |
|--:|---|--:|--:|--:|--:|--:|
| 1 | `gdr_fwd_train_wy` | 132.29 | 15% | 28 | 0.502 | 89.1 |
| 2 | `gdr_fwd_train_scan` | 182.71 | 20% | 24 | 0.796 | 84.1 |
| 3 | `gdr_bwd_dv_local` | 37.35 | 4% | 28 | 0.705 | 87.9 |
| 4 | `gdr_bwd_reverse_scan` | 216.99 | 24% | 16 | 0.570 | 56.7 |
| 5 | `gdr_bwd_dqkw` | 161.02 | 18% | 28 | 0.716 | 89.0 |
| 6 | `gdr_bwd_wy_matmuls` | 111.99 | 12% | 28 | 0.812 | 88.0 |
| 7 | `gdr_bwd_wy_reduce` | 41.64 | 5% | 56 | — | pure vector |
| 8 | `gdr_bwd_finalize` | 27.99 | 3% | 56 | — | pure vector |
| | **total** | **911.98** | | | | |

The load is spread: the top five kernels are each 12–24% of the total, so
halving any single one buys at most an eighth of the pipeline.

`gdr_bwd_reverse_scan` launches 16 of the 28 cube blocks, and that is a property
of this shape rather than of the kernel. Its independent work is
`B * H * ceil(V/64)` items — 16 at `H=8` — and every refinement of the item space
doubles a power-of-two axis, so the reachable counts are 8, 16, 32, … and none
lands in (16, 28]. At `H=16` there are 32 items and it launches all 28 blocks.

The same run's host wall time is **1728 µs**, because the timing mode brackets
each of the eight launches with its own `torch.npu.synchronize()`. The gap is
per-launch dispatch, not device idle. Benchmark the path end to end and the
dispatch is paid once per launch; do not use wall clock to judge a kernel.

## How it scales

One axis at a time from the reference shape, same method:

| T | N | total µs | µs per chunk |
|--:|--:|--:|--:|
| 256 | 4 | 203.7 | 50.9 |
| 512 | 8 | 311.8 | 39.0 |
| 1024 | 16 | 533.3 | 33.3 |
| 2048 | 32 | 1052.9 | 32.9 |
| 4096 | 64 | 2079.6 | 32.5 |

Per-chunk cost flattens from T=1024; below that, launch and barrier overhead does
not amortise. T=256 costs 56% more per chunk than T=4096. There is no
super-linear term anywhere through T=4096.

At **B=1, T=16384, H=16, K=V=128** — sixteen times the work of the reference —
the pipeline takes **14.24 ms**, against the 14.59 ms that scaling the reference
linearly would predict. Scaling is slightly *sub-linear*: per chunk and head the
cost falls from 3.563 to 3.477 µs. Six of the eight kernels clear 93% occupancy
at that shape, against two of eight at the reference shape, because larger shapes
fill the cores better. The reference shape is therefore the pessimistic one; do
not extrapolate from it linearly and assume the result is conservative.

Correctness has not been checked at that shape — see `VALIDATION.md`.

## Forward only

Training forward is kernels 1 and 2. Measured on its own at B=1, T=16384, H=16,
K=V=128: `gdr_fwd_train_wy` 2047.4 µs plus `gdr_fwd_train_scan` 2825.9 µs =
**4873.3 µs** of device time, with a host wall of 5391 µs.

The forward costs the same alone as it does inside the full pipeline (0.09%
apart), so the two halves do not interfere through L2. Forward is 34.2% of
training; backward is 1.92 times forward.

## Against the machine

The chunked forward's arithmetic requirement is countable. Per `(batch, head,
chunk)` at C=64, K=V=128, summing the nine matmuls of the chunk — the two
triangular Gram products, the inverse, the two WY applications, the two output
terms and the state update — gives 5,810,858 multiply-accumulates. At
`B*H*N = 4096` items that is **23.80 GMAC** for the whole forward.

The machine's peaks were measured with a probe of back-to-back
`[64,64] x [64,64]` MMADs: 623 ns per MMAD in float32 and 42 ns in bfloat16,
which is 11.78 and 174.76 TMAC/s across 28 cores. So this shape's floors are
**2020 µs** in float32 and 136 µs in bfloat16.

| | |
|---|--:|
| measured forward | 4873 µs |
| float32 arithmetic floor | 2020 µs |
| **measured / floor** | **2.41x** |
| MACs actually issued | 38.40 GMAC |
| **issued / required** | **1.61x** |
| fraction of the window with the MAC pipe busy | 67% |

The 2.41x decomposes into three independent factors: 1.61x redundant MAC issue
(tile padding, the two triangular products that compute a full square and
discard half, and the transposed duplicates the backward needs), 33% of the
window not MAC-busy, and the dtype itself. The dtype term is the largest by an
order of magnitude: even with perfect scheduling and no redundant work, float32
caps this shape at 2020 µs where bfloat16 caps it at 136 µs.

## Against the framework

`fla`'s `naive_chunk_gated_delta_rule` is pure torch, so it runs under
`torch_npu` on the same part. Synchronised wall time, same session,
float32 inputs, `scale = 1/sqrt(K)`:

| shape | `fla` naive_chunk | ours, wall | speedup | ours, device |
|---|--:|--:|--:|--:|
| B=1, T=2048, H=8, K=V=128 | 9161.3 µs | 695.1 µs | **13.2x** | 316.6 µs |
| B=1, T=16384, H=16, K=V=128 | 32361.7 µs | 5391 µs | **6.0x** | 4873.3 µs |

`fla`'s samples at the large shape were 31422 / 32362 / 33488 µs; ours are
medians of 21 repeats at the production shape and 5 at the large one.

## Composing the step: one synchronisation and two streams

The table above charges each kernel its own `torch.npu.synchronize()`, which is
what the component wrappers do when called on their own. A training step chains
eight of them, so that is eight host-device round trips, and the closing note of
this document observes that at these shapes dispatch is a bigger lever than
anything inside the kernels. `kernels/gated_delta_rule_train.py` is the
composition that collects it.

Three things change, none of them inside a kernel. The chain synchronises once
instead of eight times. The two independent branches -- the forward scan, and
`dv_local -> reverse_scan` -- are issued on separate streams and rejoined before
`dqkw`. And the two co-running launches are sized so that their block counts sum
to the part instead of each claiming all 28 cube cores.

Per step, medians of paired samples interleaved within one run:

| | B=1 T=2048 H=8 | B=1 T=16384 H=16 |
|---|--:|--:|
| the eight wrappers chained as they ship | 1638 µs | 14996 µs |
| one synchronisation, sequential | 1066 | 14428 |
| + two streams, co-sized | 1028 | 12559 |
| + steps queued on alternating streams | **860** | **12311** |
| against the shipped chain | **1.90x** | **1.22x** |

The last row is what `gated_delta_rule_fwd_bwd_pipelined` does, and it is the
figure to quote for the operator: it is the per-step cost of a training loop.
**It depends strongly on how many steps are queued per call**, which the table
below makes precise; the 860 µs row above was four steps a call, and sixteen is
better.

Earlier attempts to re-measure it read as high as 928 µs, and that was an
artefact of how it was measured rather than the code. Two mistakes, both worth
recording because they are easy to repeat:

* **The pipeline needs about ten iterations to reach steady state** and reads
  ~50% high before it does — 2408 µs on the first iteration, 1364 on the fifth,
  settled by the tenth. Three warm-up iterations, which is enough for a single
  synchronised launch, is not enough here.
* **Interleaving configurations stops any of them settling.** Alternating three
  arms within one loop gave every arm 914–922 µs, worse than the *worst* arm of
  a cleaner run. Each configuration now gets its own block with its own
  warm-up.

The rest was chased down rather than assumed:

* The **device side is stable.** The eight kernels re-profiled at 909.14 µs
  against the 911.98 µs in the table above, 0.3% apart, every kernel within
  2 µs of its published figure. Nothing regressed on device.
* The **860 µs row is reproducible once measured properly**, and is beaten:
  that configuration queued four steps a call, and queueing sixteen reaches
  828.5 µs. See the table below.
* Ruled out by measurement, each with a control: arm ordering (medians flat
  across positions to 5 µs), the `record_stream` calls the entry point adds
  (removing them is not faster), the runtime's repeated per-launch log records
  (worth 1.3 µs here, though they do cost 95 µs of *host* dispatch -- that
  dispatch overlaps device execution rather than serialising with it), distinct
  versus repeated input batches (5 µs), the number of streams in flight, and
  device contention (the device was idle and healthy throughout).

### How many steps to queue, and the headline

Every call fills the pipeline and drains it: the first step has nothing ahead of
it to overlap with, the last step's tail has nothing following. Those two edges
are paid once per call however long the sequence, so a short sequence pays them
over few steps. One session, each arm in its own block with 12 warm-up
iterations discarded and 24 measured:

| | µs per step | best | p90 | vs the chain |
|---|--:|--:|--:|--:|
| the eight wrappers chained as they ship | 1617.5 | 1602.1 | 1637.9 | 1.00x |
| one synchronised step | 1027.0 | 1015.2 | 1193.1 | 1.58x |
| pipelined, 4 steps queued | 912.7 | 883.2 | 922.4 | 1.77x |
| pipelined, 8 steps queued | 864.7 | 858.7 | 868.9 | 1.87x |
| **pipelined, 16 steps queued** | **828.5** | **824.7** | **831.4** | **1.95x** |

**828.5 µs per step, 1.95x the shipped chain**, and the longest sequence is also
the steadiest: p90 831.4 against a 828.5 median, where four steps a call spans
912.7 to 922.4. Queue the whole accumulation window in one call.

That figure is well under the 909.14 µs of device time the eight kernels need
on their own, which is only possible because successive steps' device work
overlaps. The cube is busy for nearly all of a step — the two scans are co-sized
to fill the part exactly — so what the next step overlaps with is chiefly the
66.4 µs of pure-vector tail in `wy_reduce` and `finalize`, where the cube is
idle. Being below the device sum is the thing worth checking in any re-measure.

Steps in flight is not the lever: at sixteen queued, two / three / four in
flight give 833.2 / 832.5 / 836.1 µs. Two is the default because it holds the
least memory.

### Why a step's branch streams must not be shared between steps

Enabling both concurrency axes at once used to be slower in the median and much
worse in the tail -- p90 1202 µs against a 922 µs median, with a 377 µs spread
run to run. The cause was not oversubscription. It was that the pair of branch
streams a step uses for its two independent halves was cached **per device**, so
every step in flight borrowed the same pair, and the waits crossed: the pair
ends up waiting on both steps' streams while both of those wait on the pair.
Neither step's branches can begin until the other's queue drains, and whether
that bites depends on where the device had reached when each launch was queued.
That produces a long tail rather than a slower median, which is exactly what was
seen.

One pair per in-flight step removes the sharing. Over 25 rotated iterations
through the shipped function:

| | median | p10 | p90 | worst | spread |
|---|--:|--:|--:|--:|--:|
| one branch pair for all steps | 922.0 µs | 905.7 | 1202.1 | 1281.3 | 377.0 |
| **one pair per in-flight step** | **914.1** | **904.6** | **971.7** | **986.2** | **89.6** |
| no within-step concurrency | 921.4 | 910.3 | 1025.2 | 1077.4 | 185.8 |

The best median and a spread four times tighter than sharing a pair, two times
tighter than not splitting the step at all -- so the two axes do compose, and
`concurrent=True` is the default. The earlier reading that they were substitutes,
and that disabling the within-step split was the better trade, was an artefact of
the shared pair.

`tests/test_train_entry.py` pins the structure rather than the timing: with N
steps in flight it asserts that N distinct slots were allocated and that no
stream object is shared between them. A correctness test cannot see this defect,
since sharing the pair gives the right answer and only disturbs when work runs.

Every row is bit-identical to every other, and to the sequential chain: the
stream assignment and the core budgets change only how independent item spaces
fold onto cores, never the arithmetic. `tests/test_train_entry.py` asserts that
as exact equality over three shapes, four budget pairs including ones that
over-commit the part, and four steps queued without a per-step synchronisation.

Two of these rows deserve a note.

**Co-sizing is what makes the streams pay, not the streams themselves.** Each
tiling selector maximises its own core fill assuming it owns the part, and two
launches that both claim 28 blocks do not overlap -- they serialise. Giving the
forward scan the cores the backward branch leaves free is what converts the
concurrency into time, and the optimum is sharp: at the large shape the two
counts sum to exactly the part, and a pair that over-commits it to 34 costs 26%
more than the pair that fits.

**The last row is below the device sum.** 860 µs is less than the 911.98 µs of
the eight kernels' individually measured device times, so queued steps overlap
device work and not merely host dispatch. This is the figure a training loop
sees; the 1028 µs row is the latency of one isolated synchronised step. Which
one to quote depends on the question being asked.

## What these numbers do and do not license

Four ways they can be overread:

1. **The framework reference is a *naive* reference.** Its triangular inverse is
   a 63-iteration Python loop doing in-place slice assignment, so much of its
   cost is eager-mode launch overhead rather than arithmetic. Beating it is
   necessary and nowhere near sufficient; a competently written torch_npu
   composition with a batched inverse would land far below it. The arithmetic
   floor is the stronger comparison precisely because it does not depend on how
   good somebody else's reference is.
2. **The ratio shrinks with size**, 13.2x to 6.0x, as the reference's fixed
   overhead amortises. Quote the shape with the number, never the number alone.
3. **The reference is forward-only and emits no training intermediates.** Our
   forward emits seventeen tensors, several of which exist only to feed the
   backward, so on that axis it is being charged for strictly more work.
4. **Our own host overhead is large at small shapes**: 695.1 µs wall against
   316.6 µs device is 54% dispatch at the production shape, falling to 10% at
   the large shape. Wall-to-wall is therefore the conservative ratio to publish,
   and at small shapes dispatch is a bigger lever than anything inside the
   kernels.

And one about occupancy: `cube_utilization` measures how many cores are busy,
not how well. Multiplying it by `aic_mac` gives the fraction of the machine's
multiply-accumulate capability actually in use, which on this pipeline is 33–82%
depending on the kernel. The two figures answer different questions and neither
alone is "utilisation".

**Defensible from the above**: the pipeline runs at 912 µs of device time at the
production shape, and a pipelined step costs 828.5 µs there with the
accumulation window queued in one call, 1.95 times less than chaining the
wrappers as they ship; the forward is 2.41 times its float32 arithmetic floor on this part and
issues 1.61 times the algorithmic minimum of multiply-accumulates; it is 6.0 to
13.2 times faster than the framework's own reference implementation at the
shapes stated. **Not defensible**: that it is the fastest implementation of this
operator on this part. That would need a tuned competitor to compare against,
and no such kernel exists for this operator family here.
