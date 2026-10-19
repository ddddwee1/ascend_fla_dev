# The gated delta rule, and how this implementation is organised

## The operator

The gated delta rule is a linear-attention recurrence with a state matrix
`S` of shape `[K, V]`, a scalar forget gate and a delta correction. For one
`(batch, head)` and tokens `t = 0 .. T-1`, with `g` a log-space decay, `beta` a
weight in `(0, 1)` and `scale = 1/sqrt(K)`:

    S <- S * exp(g_t)
    v_t <- beta_t * (v_t  -  S^T k_t)          the delta correction
    S <- S + k_t v_t^T
    o_t = (scale * q_t)^T S

Two properties matter for an implementation. The gate is one scalar per token
and head, so it commutes with the matrix structure and can be folded into
per-row factors — unlike GDN-2, whose per-channel gate cannot. And the
correction makes `v_t` depend on the state, so the recurrence does not unroll
into a single matmul the way plain linear attention does.

## The chunked reformulation

The sequence is cut into chunks of `C = 64` tokens. Within a chunk the whole
delta recurrence collapses into one triangular solve, which is what makes the
operator worth putting on a matrix unit at all.

Write `decay` for the cumulative sum of `g` inside the chunk, and per chunk:

    beta_act  = sigmoid(beta)                   q_s   = q * scale
    k_beta    = k * beta_act                    v_b   = v * beta_act
    decay_exp = exp(decay)                      dfac  = exp(decay_{C-1} - decay)
    L[i, j]   = exp(decay_i - decay_j)          for j <= i, else 0

    attn      = (I - strict_lower((k_beta @ k^T) * L))^-1
    a_mat     = lower_incl((q_s @ k^T) * L)
    u         = attn @ v_b
    w         = attn @ (k_beta * decay_exp)
    q_de      = q_s * decay_exp

`attn` is the WY transform of the chunk. The matrix being inverted is unit lower
triangular, so the inverse is exact after one pass of forward substitution over
the `C` rows — no iteration, no conditioning concern.

What remains is a recurrence over chunks only, `N = ceil(T / C)` steps instead
of `T`:

    s_hist_i = S                                the state entering chunk i
    v_new_i  = u_i - w_i @ S
    o_i      = q_de_i @ S + a_mat_i @ v_new_i
    S        = S * decay_exp_i[C-1] + (k_i * dfac_i)^T @ v_new_i

Everything above the chunk recurrence is independent per chunk, so the work
splits into a wide parallel part and a narrow serial part. That split is the
organising principle of this implementation.

## The backward pass

Differentiating the chunked form gives a mirror image: one serial scan that
carries `dS` backwards over the chunks, and a set of chunk-parallel operators
around it.

    dh_i  = dS                                  the state gradient entering chunk i
    dS   <- dS * decay_exp_i[C-1]
            + q_t_i @ (decay_exp_i . grad_o_i)
            - w_t_i @ dv2_i

Once `s_hist` and `dh` exist, every remaining gradient is chunk-local. The
chain splits into three groups:

* `dv` in two parts: the within-chunk contribution, which needs only
  `grad_o`, `q` and `k`; and the cross-chunk contribution, which the reverse
  scan adds as it goes.
* `dq`, `dk`, `dw` and most of `dg`, which are eleven matmuls per chunk against
  the two state trails.
* the adjoint of the WY transform. Differentiating a triangular inverse puts
  `attn^T` on both sides of the incoming gradient, with a mask and a gate
  between the two products. Its matmul half produces four large chunk-local
  tensors; its reduction half contracts them along `K` and `V`, and needs no
  matmul at all, because every contraction the chain rule asks for collapses
  into a row or column sum.

The gate gradient `dg` is assembled last. It arrives in two pieces, one from the
chunk-parallel backward and one from the WY adjoint, and is closed by a suffix
scan within the chunk — the transpose dual of the forward's cumulative sum —
plus one per-chunk scalar folded onto the chunk's last token.

## Why nine kernels

A single fused program would be ideal and does not fit. Two limits decide it.

**On-chip memory.** The forward's five phases, fused into one kernel, project to
359,424 B of unified buffer against a 253,952 B budget. UB is allocated by a
monotone bump pointer with no reuse across phases that are both resident, so two
resident phases add rather than overlap. Split, each kernel holds one resident
phase.

**Task grids.** The chunk-parallel phases want `B * H * N` independent items;
the two scans cannot split the chunk axis at all and want `B * H`, or
`B * H * V-tiles` once the independent `V` columns are exposed. One kernel has
one grid, so a fused kernel would have to run the parallel phases on the scan's
grid and leave most cores idle.

The split is therefore along the phase boundaries, and the boundaries are
materialised in GM. Nothing else crosses them: a host entry point validates
shapes, allocates outputs with `torch.empty`, computes scalars in Python and
launches once. No arithmetic on device tensors happens on the host.

## The eight training operators

| # | operator | role | reads | writes |
|--:|---|---|---|---|
| 1 | `gdr_fwd_train_wy` | within-chunk WY transform and every residual the backward needs | `q, k, v, g, beta` | 17 chunk-major tensors |
| 2 | `gdr_fwd_train_scan` | the forward chunk recurrence | `u, w, q_de, a_mat, k_f32, dfac, decay_exp` | `o, s_hist, v_new` |
| 3 | `gdr_bwd_dv_local` | within-chunk part of `dv` | `grad_o, k_f32, q_t, g_f32` | `dv_local` |
| 4 | `gdr_bwd_reverse_scan` | the backward chunk recurrence | `q_t, w_t, k_f32, grad_o, decay_exp, g_f32, dv_local` | `dh, dv2` |
| 5 | `gdr_bwd_dqkw` | chunk-parallel `dq`, `dk`, `dw`, and part of `dg` | `grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, decay` | `dq_p, dk_p, dw, dg_p` |
| 6 | `gdr_bwd_wy_matmuls` | matmul half of the WY adjoint | `attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2` | `dkbg, dvb, dkb, dkt, d_a` |
| 7 | `gdr_bwd_wy_reduce` | reduction half of the WY adjoint | the four above, `k_f32, beta_act, decay_exp, dk_p, v` | `dk, dv, dbeta_wy, dg2` |
| 8 | `gdr_bwd_finalize` | assemble and publish the five gradients | `dq_p, dk, dv, dg_p, dg2, dbeta_wy, sig` | `dq, dk, dv, dg, dbeta` |

The ninth kernel, `gated_delta_rule_fwd`, is the inference forward: the same
mathematics with the residuals dropped, fused into one launch with a single
grid-wide barrier between its chunk-parallel and scan phases. It exists because
an inference path should not pay for seventeen tensors it will never read.

## Layouts and conventions

Public tensors are token-major, `[B, T, H, D]`, which is what `fla` and the
calling layer use. Everything internal is chunk-major, `[B, H, N, C, D]`, which
is what the matrix unit wants: a chunk is then a contiguous tile.

The conversion is absorbed into the loads. A 2-D tile reading a public tensor
passes `order=[1, 3]`, which walks the token and channel axes at a fixed head
instead of the default trailing two; without it a load silently strides by
`t * H + h`, which is invisible at `H = 1`.

Sequence length is padded to `N * C` inside the operator, with `g = 0` and
`beta_act = 0` on the padding rows, so every product they enter is exactly zero
rather than approximately so. The trim back to the true `T` happens once, in the
backward epilogue, together with the transpose to the public layout.

Three tensors are published pre-transposed — `q_t`, `kbeta_t` and `w_t` — so
that the operators downstream can load a left matmul operand directly. A
transposed operand destined for L0A has to come from an L1 tile declared `ZN`
and filled by a descending-order load; materialising the transpose once in the
forward is cheaper than arranging that at every use.

All arithmetic is float32. The only narrowings are the two public outputs that
the contract defines as bfloat16: `o` in the forward, and `dq`, `dk`, `dv` in
the backward epilogue. Both happen on the store path, where the accumulator to
GM conversion is free.
