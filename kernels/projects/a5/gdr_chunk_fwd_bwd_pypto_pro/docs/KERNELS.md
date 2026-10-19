# Per-kernel reference

Every kernel is one launch of one `@pl.jit` program. The host entry point
validates shapes, allocates outputs, computes its scalars in Python and
launches; it performs no arithmetic on device tensors.

`C = 64` is the chunk size and a frozen constant of the family;
`N = ceil(T / C)`. Internal tensors are chunk-major `[B, H, N, C, *]`, public
ones token-major `[B, T, H, *]`.

## Summary

| operator | cores | work item | launches | UB used |
|---|---|---|--:|--:|
| `gdr_fwd_train_wy` | cube + vector | (batch, head, chunk) | 1 | 127,744 B |
| `gdr_fwd_train_scan` | cube + vector | (batch, head, V-tile group) | 1 | 115,968 B |
| `gdr_bwd_dv_local` | cube + vector | (batch, head, chunk) | 1 | 111,104 B |
| `gdr_bwd_reverse_scan` | cube + vector | (batch, head, V sub-block) | 1 | 140,032 B |
| `gdr_bwd_dqkw` | cube + vector | (batch, head, chunk) | 1 | 238,944 B |
| `gdr_bwd_wy_matmuls` | cube + vector | (batch, head, chunk) | 1 | 217,088 B |
| `gdr_bwd_wy_reduce` | vector | (batch, head, chunk) | 1 | 247,808 B |
| `gdr_bwd_finalize` | vector | (batch, chunk, head group) | 1 | 133,120 B |
| `gated_delta_rule_fwd` | cube + vector | two phases, see below | 1 | 237,568 B |

The budget is 253,952 B of unified buffer, 524,288 B of L1, 65,536 B each of
L0A and L0B, 262,144 B of L0C, 32 mutex ids and 16 cross-core event ids per
kernel.

`kernels/gated_delta_rule_train.py` is not a kernel and so has no row above: it
is the host-side composition of operators 1 to 8 into one training step, issuing
the two independent branches on separate streams and sizing their grids so that
the block counts sum to the part. Operators 2 and 4 take a `cube_core_budget`
for it.

## Glossary of the intermediate tensors

| name | shape | meaning |
|---|---|---|
| `q_s` | `[B,H,N,C,K]` | `q * 1/sqrt(K)` |
| `k_f32` | `[B,H,N,C,K]` | `k` widened to float32 |
| `v_b` | `[B,H,N,C,V]` | `v * beta_act` |
| `beta_act`, `sig` | `[B,H,N,C]` | `sigmoid(beta)`; `sig` is the same value, kept under the name the epilogue uses |
| `g_f32`, `decay` | `[B,H,N,C]` | per-token log gate; its cumulative sum within the chunk |
| `decay_exp` | `[B,H,N,C]` | `exp(decay)` |
| `dfac` | `[B,H,N,C,K]` | `exp(decay_{C-1} - decay)`, broadcast along K |
| `attn` | `[B,H,N,C,C]` | the WY transform, unit lower triangular inverse |
| `a_mat` | `[B,H,N,C,C]` | within-chunk attention, gated and causally masked |
| `u`, `w` | `[B,H,N,C,V]`, `[B,H,N,C,K]` | `attn @ v_b`, `attn @ (k_beta * decay_exp)` |
| `q_de` | `[B,H,N,C,K]` | `q_s * decay_exp` |
| `q_t`, `kbeta_t`, `w_t` | `[B,H,N,K,C]` | pre-transposed `q_s`, `k_beta * decay_exp`, `w` |
| `s_hist` | `[B,H,N,K,V]` | the state entering each chunk; `s_hist[:, :, 0] = 0` |
| `v_new` | `[B,H,N,C,V]` | the value rows after the delta correction |
| `dh` | `[B,H,N,K,V]` | the state gradient entering each chunk; `dh[:, :, N-1] = 0` |
| `dv_local`, `dv2` | `[B,H,N,C,V]` | `dv` within a chunk; `dv` after the reverse scan |
| `dq_p`, `dk_p`, `dw`, `dg_p` | `[B,H,N,C,K]`, `[B,H,N,C]` | the chunk-parallel backward's outputs |
| `dkbg`, `dkb`, `dkt`, `dvb`, `d_a` | chunk-local | the WY adjoint's matmul results |
| `dbeta_wy`, `dg2` | `[B,H,N,C]` | the WY adjoint's contributions to `dbeta` and `dg` |

---

## 1. `gdr_fwd_train_wy`

Within-chunk WY transform, and every residual the backward pass needs.

Four phases in one kernel: preprocess on the vector core, the two score
matmuls on the cube, the WY build on the vector core, and the application of
`attn` back on the cube. Three event pairs carry the hand-offs; they are flat
and non-overlapping, because nesting them deadlocks a single-buffered pipeline.

* **in** `q, k [B,T,H,K]` bf16, `v [B,T,H,V]` bf16, `g [B,T,H]` f32,
  `beta [B,T,H]` bf16
* **out** 17 float32 chunk-major tensors: `q_s, k_f32, v_b, g_f32, beta_act,
  sig, q_t, decay, decay_exp, dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t`
* **grid** `B*H*N` items, contiguous per-core runs, `block_dim = min(cores, items)`
* **budget** UB 127,744 B; L1 top 0x2d000; L0A and L0B at half their capacity
* **notes** The WY build is banded across the two vector sub-blocks: one takes
  `decay_exp`, the decay mask and `attn`, which holds the sequential forward
  substitution; the other takes `a_mat, q_de, kbeta_de, kbeta_t, dfac`, which
  moves four times the data. The transpose writes down a column at an odd row
  pitch, because a stride that is an even multiple of the 32-byte bank width
  puts every lane of a register in one bank.

## 2. `gdr_fwd_train_scan`

The forward chunk recurrence; emits the output and the two state trails.

    s_hist_i = S
    v_new_i  = u_i - w_i @ S
    o_i      = q_de_i @ S + a_mat_i @ v_new_i
    S        = S * decay_exp_i[C-1] + (k_f32_i * dfac_i)^T @ v_new_i

* **in** `u, w, q_de, a_mat, k_f32, dfac, decay_exp`, all float32 chunk-major
* **out** `o [B, C*N, H, V]` bf16 (the caller slices the first `T` rows),
  `s_hist`, `v_new`
* **grid** `B * H * ceil(V/VT) / streams` items; the chunk axis is serial, the
  `V` columns are independent. Tile width and stream count are chosen on the
  host from the core count, which `cube_core_budget` caps for a caller running
  this launch concurrently with another
* **budget** UB 115,968 B; L1 311,296 B; 27 of 32 mutex ids; 7 of 16 events
* **notes** `s_hist` must be snapshotted before the state update. Matmul and
  fixpipe always run on the full physical tile: a partially valid N axis leaves
  stale rows in the destination of a split-M drain, so the narrow `V` tail
  lives in the loads, the stores and the vector column masks instead. `k^T`
  reaches L0A through a transposing load into an `ZN` L1 tile, once per chunk.

## 3. `gdr_bwd_dv_local`

The within-chunk contribution to `dv`; the only backward operator with no
state dependency.

    scores[i, j] = (k_i . q_j) * exp(decay_j - decay_i)   for j >= i, else 0
    dv_local     = scores @ grad_o_chunk

* **in** `grad_o [B,T,H,V]` bf16, `k_f32`, `q_t`, `g_f32`
* **out** `dv_local [B,H,N,C,V]` f32
* **grid** `B*H*N` items, contiguous per-core runs
* **budget** UB 111,104 B; L1 176 KB; 22 of 32 mutex ids
* **notes** The mask is applied before the exponential, so the discarded half
  never produces an infinity to cancel. `scale` is already folded into `q_t`.
  The masked scores and the widened `grad_o` reach L1 through NZ staging in UB,
  eight columns at a time, because the vector core cannot write an L1 operand
  directly.

## 4. `gdr_bwd_reverse_scan`

The backward chunk recurrence, the directional dual of operator 2.

    dh_i      = dS
    dv2_i     = dv_local_i + (k_f32_i @ dS) * exp(g_last_i - decay_i)
    dS        = dS * decay_exp_i[C-1] + q_t_i @ (decay_exp_i . grad_o_i)
                - w_t_i @ dv2_i

* **in** `q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local`
* **out** `dh [B,H,N,K,V]`, `dv2 [B,H,N,C,V]`, both f32
* **grid** `B * H * ceil(V/64)` items; chunks are visited in reverse within an
  item. The kernel strides over the item space, so the block count is a free
  host-side choice that affects balance only, and `cube_core_budget` sets it
* **budget** UB 140,032 B; L1 163,840 B; L0A exactly at its 64 KB capacity
* **notes** `dS` lives in UB as a single persistent buffer and reaches the cube
  through a GM bridge. All vector work runs on sub-block 0, guarded, because the
  state is single-buffered.

## 5. `gdr_bwd_dqkw`

The chunk-parallel backward: eleven matmuls per chunk against the two state
trails. See the module docstring for the equations.

* **in** `grad_o`, `v_new`, `k_f32`, `q_s`, `s_hist`, `dh`, `dv2`, `decay`
* **out** `dq_p`, `dk_p`, `dw [B,H,N,C,K]`, `dg_p [B,H,N,C]`, all f32
* **grid** `B*H*N` items, contiguous per-core runs
* **budget** UB 238,944 B; 32 of 32 mutex ids; L0A and L0B at capacity
* **notes** `scores^T` is never computed: masking commutes with the transpose
  here, so the masked scores are written once and read back transposed.
  `v_new` cannot be replaced by the primary `v` — substituting it makes `dq`
  wrong by O(1), not by a tolerance. Two accumulations leave one partial per
  lane and are reduced after their producing loop, because a read-modify-write
  on one tile address inside a vector loop is not ordered across iterations.
  **This kernel pins the batch axis to 1 and is what limits the pipeline to
  B = 1.**

## 6. `gdr_bwd_wy_matmuls`

The matmul half of the WY adjoint: eight matmuls in three cube phases,
separated by two vector phases.

    dA   = dw @ kbeta_t + dv2 @ v_b^T
    dkbg = attn^T @ dw                  dvb = attn^T @ dv2
    dA  <- A^T @ (strict_lower(dA) @ A^T)
    d_a  = -(dA * exp(min(decay_row - decay_col, 0)))     strict lower
    dkb  = d_a @ k_f32                  dkt = d_a^T @ k_beta

* **in** `attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2`; no public tensor
* **out** `dkbg, dkb, dkt [B,H,N,C,K]`, `dvb [B,H,N,C,V]`, `d_a [B,H,N,C,C]`
* **grid** `B*H*N` items, contiguous per-core runs
* **budget** UB 217,088 B; L1 278,528 B; L0C 114,688 B; L0A and L0B at half
  capacity, which is what leaves room for the NZ staging a transposed operand
  needs
* **notes** `attn^T` is loaded once per item and serves as the left operand of
  three matmuls and the right operand of a fourth. The bridge between the two
  phase-B matmuls goes L0C to UB to L1, because there is no accumulator-to-L1
  path on this part. The masked `dA` returns to the cube through GM, which costs
  no extra write because it is a required output anyway.

## 7. `gdr_bwd_wy_reduce`

The reduction half of the WY adjoint. Pure vector: every contraction the chain
rule asks for over the triangular factor collapses into a row or column sum.

* **in** `dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p`, and the
  primary `v [B,T,H,V]` bf16
* **out** `dk [B,H,N,C,K]`, `dv [B,H,N,C,V]`, `dbeta_wy`, `dg2 [B,H,N,C]`
* **grid** `B*H*N` items; pure vector, so the vector core count bounds it
* **budget** UB 247,808 B, the tightest in the set
* **notes** The module docstring carries the ledger of where `beta_act`,
  `decay_exp` and the scale are applied, each exactly once on each path; this is
  the easiest place in the pipeline to double-apply a factor. A free axis wider
  than 64 is handled as two halves, each its own tile, and the upper half
  carries zero valid lanes when `K` or `V` is 64.

## 8. `gdr_bwd_finalize`

The epilogue: assembles `dg` and `dbeta`, and republishes the three large
gradients in the public layout.

    dg[t]    = sum over t' >= t of (dg_p[t'] + dg2[t'])     suffix scan in-chunk
    dbeta[t] = dbeta_wy[t] * sig[t] * (1 - sig[t])

* **in** `grad_o` (for its geometry only), `dq_p`, `dk`, `dv`, `dg_p`, `dg2`,
  `dbeta_wy`, `sig`
* **out** `dq, dk [B,T,H,K]` bf16, `dv [B,T,H,V]` bf16, `dg, dbeta [B,T,H]` f32
* **grid** `B * N * head-groups` items; the group width is chosen on the host to
  minimise `waves * heads_per_group`
* **budget** UB 133,120 B; 20 of 32 mutex ids
* **notes** The suffix scan is exact in six Hillis-Steele steps over the 64
  register lanes, with no cube matmul. `dg` and `dbeta` are one scalar per token
  and head, and `pl.store` always writes a tile's free axis as physically
  contiguous bytes, so the token axis must be the tile's partition axis: the
  results are staged head-major, transposed once per group, and stored once.

## 9. `gated_delta_rule_fwd`

The inference forward: the same mathematics as operators 1 and 2 with the
residuals dropped, in one launch. Two phases separated by one grid-wide
barrier — the chunk-parallel preparation, and the state scan over the flat
`(batch-head, V-tile)` item space.

* **in** `q, k, v, beta` bf16 and `g` f32, token-major
* **out** `o [B,T,H,V]` bf16
* **budget** UB 237,568 B; the state is carried through a GM workspace rather
  than kept in L1, which is what lets `K` and `V` reach 256 here
* **notes** Its module docstring records three build behaviours it is shaped
  around, each found through a wrong numeric result rather than a compile error:
  a matmul with `N` below 64 returns wrong results, so `V` is padded; an L0A or
  L0C tile declared at 128 rows but used with 64 valid rows returns wrong
  results; and a transpose-loaded tile is usable as a right operand but not as a
  left one, so `k^T` is materialised by a vector transpose.
