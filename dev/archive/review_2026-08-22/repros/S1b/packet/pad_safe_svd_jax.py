"""
Method D, static-shape JAX version, with a compiled-scan sanity check.

Same contract as pad_safe_svd.py:  pad -> svd -> unpad == svd of unpadded A.
Given A_pad (N x M) and boolean masks row_pad (N,), col_pad (M,):
  * first m = M - sum(col_pad) triplets are the economy SVD of the unpadded
    A, with U[:, :m] bitwise zero on pad rows and V[:, :m] bitwise zero on
    pad-column coordinates -- in their ORIGINAL (possibly interior)
    locations: all internal permutations are inverted before returning;
  * remaining M - m triplets carry sigma = 0 (don't-care but still valid);
  * requires n >= m only; n < M supported; no rank tolerance anywhere.

Static-shape design (this file's point):
  * mask patterns are RUNTIME DATA: argsort / gather / scatter are traced
    ops with fixed shapes -- changing the pattern changes values, not the
    compiled graph, so there are no per-pattern recompiles;
  * the only dynamic-width object in the NumPy version was the
    augmentation; here it is the static M x 2M matrix [B, c*diag(t)]
    (zero columns where t = 0 are harmless: they only pollute the
    augmented SVD's right factor, WHICH IS NEVER USED);
  * V is rebuilt from U via W = A_pad.T @ U:  W = V @ Sigma exactly, so
    its permuted QR *is* V (up to column signs, fixed from diag(R)) --
    leading columns are the true right vectors even through sigma
    clusters, trailing columns are data-supported null completions and
    exact pad-coordinate vectors, in order, with no tolerance.  This
    removes the second sketch and the completion QR of the NumPy version.

Run this file to execute the scan test: one jit-compiled lax.scan over a
stack of matrices with DIFFERENT masks per element, verified element-wise
against the contract, plus a retrace check.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def make_sketch(M, seed=0):
    """Fixed Haar-orthonormal sketch; draw once, close over it (a constant)."""
    Q, R = np.linalg.qr(np.random.default_rng(seed).standard_normal((M, M)))
    return jnp.asarray(Q * np.sign(np.diag(R)))


def pad_safe_svd_jax(A_pad, row_pad, col_pad, Omega):
    """All shapes static; masks are traced runtime data."""
    N, M = A_pad.shape
    m = M - jnp.sum(col_pad)                       # traced scalar

    # ---- left basis: sketch, permute pads to bottom, QR, un-permute ----
    Y = A_pad @ Omega                              # pad rows bitwise zero
    pr = jnp.argsort(row_pad)                      # data rows first (traced)
    Qp, _ = jnp.linalg.qr(Y[pr, :])                # economy, N x M
    Q = Qp[jnp.argsort(pr), :]                     # back to original locations
    # bitwise split: data-supported columns | exact pad coordinate vectors
    t = (jnp.abs(Q) * row_pad[:, None]).max(axis=0) > 0.5   # surplus indicator

    # ---- one SVD of the static augmented projection --------------------
    B = Q.T @ A_pad                                # M x M
    nA = jnp.linalg.norm(A_pad)
    c = 4.0 * nA + (nA == 0.0)                     # 4x SAFETY MARGIN: filter at c/2 = 2||A||_F
                                                   # keeps computed sigma_max (< ||A||_F (1+eps))
                                                   # strictly below it; 2x would sit exactly AT
                                                   # sigma_max for rank-1 matrices
    C_aug = jnp.concatenate([B, c * jnp.diag(t.astype(A_pad.dtype))], axis=1)
    # left null of [B, c*diag(t)] == data-left-null exactly (the diag block
    # kills surplus components), so every kept left vector is clean; the
    # zero columns where t = 0 pollute only the right factor, never used.
    W1, Sa, _ = jnp.linalg.svd(C_aug, full_matrices=False)   # W1: M x M

    pinned = Sa > c / 2.0
    order = jnp.argsort(jnp.where(pinned, -1.0, Sa))[::-1]   # kept desc, pins last
    W1 = W1[:, order]
    So, pino = Sa[order], pinned[order]
    W1 = W1 * (1.0 - jnp.outer(t, ~pino))          # zero exact-arithmetic
                                                   # surplus dirt of kept cols
    U = Q @ W1                                     # first m cols bitwise clean
    S = jnp.where(jnp.arange(M) < m, So, 0.0)      # don't-care sigmas -> 0

    # ---- right factor rebuilt from U:  W = A^T U = V Sigma exactly -----
    Wv = A_pad.T @ U                               # bitwise zero on pad-col
    pc = jnp.argsort(col_pad)                      # coords, for EVERY column
    Vq, Rv = jnp.linalg.qr(Wv[pc, :])              # permuted QR *is* V Sigma
    V = Vq[jnp.argsort(pc), :]
    d = jnp.sign(jnp.diagonal(Rv))
    V = V * jnp.where(d == 0.0, 1.0, d)            # fix column signs
    return U, S, V


# ======================================================================
# Sanity check: jit-compiled scan over a stack with heterogeneous masks
# ======================================================================
if __name__ == "__main__":
    N, M, BATCH = 10, 5, 8
    Omega = make_sketch(M)

    @jax.jit
    def f(A_stack, row_stack, col_stack):
        def body(carry, x):
            A, r, cm = x
            return carry, pad_safe_svd_jax(A, r, cm, Omega)
        _, out = jax.lax.scan(body, None, (A_stack, row_stack, col_stack))
        return out                                  # stacked (U, S, V)

    # -- build a batch: same padded shape, DIFFERENT masks per element,
    #    spanning n >= M, n < M, varying rank, zero data rows, p_c = 0
    rng = np.random.default_rng(7)
    A_stack = np.zeros((BATCH, N, M))
    row_stack = np.zeros((BATCH, N), dtype=bool)
    col_stack = np.zeros((BATCH, M), dtype=bool)
    for b in range(BATCH):
        col_pad = rng.random(M) < rng.uniform(0, 0.6)
        m = M - col_pad.sum()
        n = int(rng.integers(max(m, 1), N + 1))     # n < M in some elements
        row_pad = np.zeros(N, bool)
        row_pad[rng.permutation(N)[: N - n]] = True # interior pad locations
        k = int(rng.integers(0, m + 1)) if m else 0
        if m and k:
            G = rng.standard_normal((n, k)) @ rng.standard_normal((k, m))
            if n > 1 and rng.random() < 0.5:
                G[rng.integers(n), :] = 0.0         # numerically-zero data row
            A = np.zeros((N, M)); A[np.ix_(~row_pad, ~col_pad)] = G
            A_stack[b] = A
        row_stack[b], col_stack[b] = row_pad, col_pad

    U, S, V = map(np.asarray, f(jnp.asarray(A_stack),
                                jnp.asarray(row_stack),
                                jnp.asarray(col_stack)))

    # -- element-wise verification of the contract, in ORIGINAL locations
    for b in range(BATCH):
        rp, cp = row_stack[b], col_stack[b]
        m = M - cp.sum(); n = N - rp.sum()
        A = A_stack[b][np.ix_(~rp, ~cp)]
        sig = np.linalg.svd(A, compute_uv=False)[:m] if m else np.zeros(0)
        assert np.all(U[b][rp][:, :m] == 0.0), b     # bitwise, original rows
        assert np.all(V[b][cp][:, :m] == 0.0), b     # bitwise, original cols
        assert np.linalg.norm(U[b].T @ U[b] - np.eye(M)) < 1e-12, b
        assert np.linalg.norm(V[b].T @ V[b] - np.eye(M)) < 1e-12, b
        assert np.allclose(S[b][:m], sig, atol=1e-10), b
        err = np.linalg.norm(U[b] * S[b] @ V[b].T - A_stack[b])
        assert err < 1e-10 * max(1.0, np.linalg.norm(A_stack[b])), b
        print(f"  elem {b}: n={n:2d} m={m}  "
              f"{'n<M ' if n < M else 'n>=M'}  sigma[:m]="
              f"{np.array2string(S[b][:m], precision=2)}  ok")

    # -- retrace check: new batch, new mask patterns, same shapes
    A2 = jnp.asarray(rng.standard_normal((BATCH, N, M)) * 0)
    r2 = jnp.asarray(np.roll(row_stack, 3, axis=1))
    c2 = jnp.asarray(np.roll(col_stack, 1, axis=1))
    f(A2, r2, c2)
    print(f"\njit cache size after 2 calls with different mask patterns: "
          f"{f._cache_size()}  (1 == no recompiles; scan traced the body once)")
    print("all scan-batch assertions passed")
