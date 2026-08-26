# Pad-Safe SVD: The Final Algorithm

Direct specification of the algorithm, without the exploratory detours
(those live in `padded_svd.pdf`, Methods A-D, if context is ever needed).

## Problem

Matrices of varying shapes are zero-padded to a common static shape
`(N, M)` for GPU/JAX batching. Pad rows and columns may sit at **arbitrary
(interior) positions**, specified by boolean masks `row_pad (N,)` and
`col_pad (M,)`. A black-box SVD of the padded matrix is wrong when the data
is rank-deficient: the null-space columns of `U` are an arbitrary basis of a
degenerate subspace that *contains the pad coordinates*, so they generically
land on pad rows. Restricted to data rows, `U` is then non-orthonormal and
possibly rank-deficient. The same happens to `V` with pad columns.

## Contract

```
pad -> svd -> unpad   ==   svd of the unpadded n x m matrix A
```

with `unpad` slicing by the masks in their original locations. Concretely,
the algorithm returns `U (N x M)`, `S (M,)`, `V (M x M)` such that:

* `A_pad == U @ diag(S) @ V.T` (a genuine economy SVD of the padded matrix);
* the **first `m = M - sum(col_pad)` triplets** are a valid economy SVD of
  the unpadded `A`: `k` positive singular values plus `m - k` data-null
  triplets, with `U[:, :m]` **bitwise zero** on pad rows and `V[:, :m]`
  **bitwise zero** on pad-column coordinates -- bitwise `== 0.0`, not just
  small;
* the remaining `M - m` don't-care triplets have `sigma = 0`; `V`'s slots
  there are exact pad coordinate vectors `e_j`, `U`'s are an orthonormal
  left-null completion (free to touch pads);
* **no rank tolerance is used anywhere**; every count comes from the masks
  or from bitwise {0,1} indicators.

Feasibility: `n >= m` (the unpadded problem's own condition) and `N >= M`
(padded shape tall; transpose first otherwise). `n < M` is fully supported.

## Algorithm

Fixed setup, once per shape `M`: a Haar-orthonormal sketch
`Omega = qr(randn(M, M)).Q` with sign fix (kappa = 1), from a fixed seed.

```
1. LEFT BASIS
   Y  = A_pad @ Omega                    # pad rows bitwise zero, any Omega
   pr = argsort(row_pad)                 # data rows first (runtime data)
   Q  = qr(Y[pr]).Q[argsort(pr)]         # economy QR, then un-permute
   t  = (|Q| * row_pad[:, None]).max(0) > 0.5    # surplus indicator {0,1}

2. AUGMENTED SVD (static M x 2M)
   c  = 4 * ||A_pad||_F   (+1 if zero)   # 4x margin -- see caveat below
   W1, Sa, _ = svd([B, c * diag(t)], full_matrices=False)   # B = Q.T @ A_pad
                                          # right factor NEVER used

3. SORT + ASSEMBLE U, S
   order = argsort(where(Sa > c/2, -1, Sa)) reversed    # kept desc, pins last
   W1    = W1[:, order];  zero rows t of the kept (non-pinned) columns
   U     = Q @ W1
   S     = where(arange(M) < m, Sa[order], 0)

4. REBUILD V FROM U
   W  = A_pad.T @ U                      # == V @ Sigma exactly; bitwise
   pc = argsort(col_pad)                 #   zero on pad-column rows
   Vq, R = qr(W[pc]);  V = Vq[argsort(pc)] * sign(diag(R))   (sign 0 -> 1)
```

Reference implementations: `pad_safe_svd.py` (NumPy, with a dynamic-shape
variant kept for readability) and `pad_safe_svd_jax.py` (the static-shape
form above, verified inside a jitted `lax.scan`). **The JAX file is the one
to port**; the two agree on the contract.

## Why each step works (one paragraph each)

**Step 1 -- matvec principle + permutation rule.** `A_pad @ x` has bitwise
zero pad rows for any `x`, so `Y` does. Householder QR preserves those zeros
*provided no pivot lands on a pad row*: the reflector at step `i` is
`v = x -/+ ||x|| e_i`, zero at zero rows of `x` **except at the pivot
position**. Permuting pads to the bottom makes the first `min(n, M)` pivots
data rows. If `n < M`, degradation is graceful: from step `n+1` the trailing
submatrix is entirely on pad rows, hence bitwise zero, hence `H = I` -- the
surplus columns of `Q` come out as *exact* pad coordinate vectors, flagged
bitwise by `t`.

**Step 2 -- pin and discard.** `col(A_pad) ⊆ span(Q)` exactly (square
invertible sketch), so `A_pad = Q @ B` and the SVD of `B` carries `Sigma`
and `V`. The `c * diag(t)` block pins each surplus coordinate at
`sigma = c` as an *exact* singular triplet (row `i` of `B` is bitwise zero
when `t_i = 1`), and makes the left null space of the augmented matrix
exactly the data-left-null -- so every kept left vector, including
`sigma = 0` ones, avoids the surplus coordinates. The zero columns where
`t = 0` pollute only the augmented SVD's right factor, which is discarded.

**Step 3 -- counting, not thresholding.** Pins sit at `c` with an
`||A||`-sized gap; `sigma > c/2` is a {pin, data} classifier with margin,
not a rank tolerance. Sorted descending, the kept triplets' first `m` slots
are the economy SVD of `A` (`k` positive + exact-arithmetic zeros); `m`
comes from `col_pad`. Zeroing the kept columns' `t`-coordinates removes
`O(eps)` dirt that is exactly zero in exact arithmetic; the subsequent
`Q @ W1` then combines only data-supported columns of `Q`, giving bitwise
clean `U[:, :m]`.

**Step 4 -- W = V Sigma is already a QR.** Every column of `A_pad.T @ U` is
bitwise supported on data columns (pad columns of `A_pad` kill those rows),
and the matrix equals `V @ Sigma` with `Sigma` decreasing --
orthonormal-times-diagonal, which *is* its own QR factorization up to
column signs (recovered from `sign(diag(R))`). So the leading columns are
the true right vectors -- exactly, even through repeated-sigma clusters --
the middle columns are data-supported null completions (reflections of
eps-level data-column content, pivots on data columns by the same
permutation rule), and the trailing columns are exact pad coordinate
vectors (`H = I` on the bitwise-zero trailing block). The whole final `V`,
in order, tolerance-free.

## Shapes and cost

Everything is `N x M`, `M x 2M`, or `M x M`; no `N x N` object exists.
Cost `O(N M^2 + M^3)`, independent of pad counts. All mask-dependent
operations (`argsort`, gathers, `diag` indicators) are **traced runtime
data with static shapes**: one jit compile covers every mask pattern
(verified: `lax.scan` traces the body once over heterogeneous masks and
the jit cache stays at size 1).

## Invariants to test after any port (see tests/)

1. `U[row_pad][:, :m] == 0.0` and `V[col_pad][:, :m] == 0.0` -- **bitwise**.
2. `U.T @ U == I` and `V.T @ V == I` to ~1e-14 (float64) over all M columns.
3. `S[:m]` matches `numpy.linalg.svd(A)` of the unpadded block.
4. `U * S @ V.T` reconstructs `A_pad` to ~eps * ||A||.
5. Jit cache size stays 1 across calls with different mask patterns.
6. Include in the sweep: interior pads (some inside the first M rows),
   n < M, rank 0, A = 0, all columns padded, repeated singular values,
   a tiny genuine singular value (~1e-8), numerically-zero data rows.

## Caveats

* **The 4x margin is load-bearing.** `c = 2 * ||A||` puts the classifier
  threshold exactly at `sigma_max`; a computed value one ulp high silently
  deletes the largest triplet (rank-1 matrices with representable norms
  trigger it deterministically). Do not "simplify" the constant.
* **Householder QR dependence.** The bitwise claims rest on Householder
  semantics (`0 - 0*x = 0`, `H = I` on exact-zero trailing blocks), true
  for LAPACK/cuSOLVER and JAX's `qr` lowering on all backends. If the QR is
  ever swapped (Gram-Schmidt, custom TPU kernel), re-run invariant 1 before
  trusting anything. The big-matrix **SVD is never called** -- only the
  small augmented core's -- so SVD backend differences (GPU Jacobi, TPU
  QDWH) don't matter.
* **Precision.** The bitwise-zero claims hold in float32/bfloat16 too
  (zeros are exact in every format); only the eps-level orthonormality and
  sigma accuracy scale with precision.
* **`Omega` handling.** Draw once per `M` from a fixed seed, outside jit,
  and close over it. Do not re-draw per call (wasteful, and per-instance
  randomness buys nothing).
* **Batching.** `vmap` works as well as `scan` (everything is static);
  choose by memory/latency preference.
* **Degenerate padded shapes.** `N < M` (wide padded shape): transpose,
  run, swap U/V. `n < m`: infeasible for any method -- it is the unpadded
  problem's own infeasibility; the implementation asserts it.
