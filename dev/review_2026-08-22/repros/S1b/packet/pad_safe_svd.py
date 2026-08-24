"""
Method D: pad-safe SVD of a zero-padded matrix (minimal working example).

Given A_pad (N x M) whose pad rows / pad columns are bitwise zero, specified
by boolean masks row_pad (N,) and col_pad (M,), compute an economy SVD
    A_pad = U @ diag(S) @ V.T
realizing the contract   pad -> svd -> unpad  ==  svd of the unpadded A:

  * the FIRST m = M - sum(col_pad) triplets are exactly the economy SVD of
    the unpadded n x m matrix A (k positive values + (m-k) data-null),
    with U[:, :m] BITWISE zero on pad rows and V[:, :m] BITWISE zero on
    pad-column coordinates;
  * the remaining M - m "don't-care" triplets carry sigma = 0 and are still
    genuine triplets of A_pad (V slots = exact pad coordinate vectors e_j;
    U slots = an orthonormal left-null completion, free to touch pads);
  * feasibility requires only n >= m -- the same condition the unpadded
    economy SVD needs.  n < M is fully supported.
  * no rank tolerance is used anywhere.

Pipeline (see the accompanying PDF, Method D):
  1. sketch the column space:      Y  = A_pad @ Omega1        (pad rows stay 0)
  2. permute pads to the bottom, economy QR  ->  Q.  Bitwise split: the
     first min(n, M) columns are data-supported; if n < M, the surplus
     M - n columns are EXACT pad coordinate vectors (graceful degradation).
  3. project:                      B  = Q.T @ A_pad   (pad cols + surplus
                                                       rows bitwise zero)
  4. sketch the row space:         Y2 = B.T @ Omega2
  5. same permutation discipline   ->  Q2  (columns split bitwise:
     data-supported | exact pad-column coordinate vectors)
  6. core:                         C  = B @ Q2
  7. TWO-SIDED Method-A augmentation at the core: pin surplus row
     coordinates (indicator t, from Q) and pad column coordinates
     (indicator s, from Q2) at sigma = c; SVD; discard the pinned cluster.
     Counting theorem: exactly m triplets survive.
  8. zero the (exact-arithmetic-zero) pinned coordinates of the kept core
     factors -> bitwise-clean U, V; complete the don't-care slots.

Written in NumPy so it runs anywhere; LAPACK semantics (Householder QR,
exact-zero preservation) are identical to what jax.numpy uses on CPU.
JAX adaptation notes are at the bottom of this file.
"""

import numpy as np


def _haar(M, rng):
    """Fixed Haar-orthonormal sketch (kappa = 1, avoids Gaussian sigma_min tail)."""
    Q, R = np.linalg.qr(rng.standard_normal((M, M)))
    return Q * np.sign(np.diag(R))          # sign fix -> Haar measure


def _inv_perm(p):
    return np.argsort(p)


def pad_safe_svd(A_pad, row_pad, col_pad, seed=0):
    A_pad = np.asarray(A_pad, dtype=float)
    row_pad = np.asarray(row_pad, dtype=bool)
    col_pad = np.asarray(col_pad, dtype=bool)
    N, M = A_pad.shape
    n = N - int(row_pad.sum())              # data rows
    m = M - int(col_pad.sum())              # data columns
    assert N >= M, "padded shape must be tall (transpose first otherwise)"
    assert n >= m, (
        f"need n >= m (got n={n}, m={m}): this is the feasibility of the "
        f"unpadded economy SVD itself -- no method crosses it")

    rng = np.random.default_rng(seed)
    Omega1, Omega2 = _haar(M, rng), _haar(M, rng)

    # ---- left side: sketch, permute, QR --------------------------------
    Y = A_pad @ Omega1                       # pad rows bitwise zero, any Omega1
    pr = np.argsort(row_pad, kind="stable")  # data rows first, pads trailing
    Qp, _ = np.linalg.qr(Y[pr])              # economy, N x M
    Q = Qp[_inv_perm(pr)]
    # Bitwise split of Q's columns: data-supported | exact e_pad (surplus).
    t = (np.abs(Q[row_pad, :]).max(axis=0) > 0.5) if row_pad.any() \
        else np.zeros(M, dtype=bool)         # surplus indicator, exact {0,1}
    p_l = int(t.sum())
    assert p_l == max(0, M - n)              # structural invariant of step 2

    B = Q.T @ A_pad                          # M x M; pad columns bitwise zero,
                                             # surplus rows bitwise zero

    # ---- right side: same discipline on B.T ----------------------------
    Y2 = B.T @ Omega2                        # pad-coordinate rows bitwise zero
    pc = np.argsort(col_pad, kind="stable")
    Q2p, _ = np.linalg.qr(Y2[pc])            # M x M
    Q2 = Q2p[_inv_perm(pc)]
    s = (np.abs(Q2[col_pad, :]).max(axis=0) > 0.5) if col_pad.any() \
        else np.zeros(M, dtype=bool)         # pad-column indicator, exact {0,1}
    p_c = int(s.sum())
    assert p_c == int(col_pad.sum())         # structural invariant of step 5

    C = B @ Q2                               # core: rows marked by t and
                                             # columns marked by s bitwise zero

    # ---- core: TWO-SIDED Method-A augmentation -------------------------
    normA = np.linalg.norm(A_pad, 2)
    c = 4.0 * normA if normA > 0 else 1.0    # 4x margin: filter c/2 = 2||A|| sits
                                             # strictly above sigma_max; 2x would put
                                             # it exactly AT sigma_max (rank-1 case)
    El = np.eye(M)[:, t]                     # M x p_l  (surplus row coords)
    Er = np.eye(M)[:, s]                     # M x p_c  (pad column coords)
    C_aug = np.block([[C,          c * El],
                      [c * Er.T,   np.zeros((p_c, p_l))]])
    # shape (M+p_c) x (M+p_l);  n >= m  <=>  p_l <= p_c  => columns are the
    # short side, so the economy SVD returns M+p_l triplets; p_l+p_c are
    # pinned at sigma = c; exactly (M+p_l)-(p_l+p_c) = M-p_c = m survive.

    Ua, Sa, Vta = np.linalg.svd(C_aug, full_matrices=False)
    pin = p_l + p_c
    kept = Sa.shape[0] - pin
    assert kept == m                         # the counting theorem

    U_core = Ua[:M, pin:].copy()             # M x m, drop augmented coords
    U_core[t, :] = 0.0                       # exact-arithmetic zeros: remove
    V_core = Vta[pin:, :M].T.copy()          # O(eps) dirt so the products
    V_core[s, :] = 0.0                       # below are BITWISE clean
    S_dat = Sa[pin:]

    # kept factors now combine only data-supported columns of Q / Q2:
    U_dat = Q @ U_core                       # N x m, bitwise zero on pad rows
    V_dat = Q2 @ V_core                      # M x m, bitwise zero on pad coords

    # ---- don't-care completion (sigma = 0; still true triplets of A_pad)
    Qc, _ = np.linalg.qr(U_core, mode="complete")     # M x M, in core coords
    U_free = Q @ Qc[:, m:]                   # N x (M-m); may touch pads: allowed
    V_free = np.eye(M)[:, col_pad]           # exact pad coordinate vectors

    U = np.hstack([U_dat, U_free])
    S = np.concatenate([S_dat, np.zeros(M - m)])
    V = np.hstack([V_dat, V_free])
    return U, S, V


# ======================================================================
# Demo 1: n >= M  (all M columns of U pad-free, as before)
# Demo 2: n <  M  (the new regime: first m columns clean, rest don't-care)
# ======================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def build(N, M, row_pad, col_pad, k, zero_data_row=True):
        n, m = N - row_pad.sum(), M - col_pad.sum()
        G = rng.standard_normal((n, k)) @ rng.standard_normal((k, m))
        if zero_data_row and n > m:
            G[n - 1, :] = 0.0                # a data row that happens to be zero
        A_pad = np.zeros((N, M))
        A_pad[np.ix_(~row_pad, ~col_pad)] = G
        return A_pad

    def report(tag, A_pad, row_pad, col_pad, U, S, V=None, m=None):
        m = U.shape[1] if m is None else m
        pad_mass = np.linalg.norm(U[row_pad, :m], axis=0)
        Ud = U[~np.asarray(row_pad)][:, :m]
        print(f"--- {tag}")
        print("  sigma[:m]                :", np.array2string(S[:m], precision=3))
        print("  pad mass, first m U cols :", np.array2string(pad_mass, precision=3))
        print("  U[:, :m] pads bitwise 0  :", bool(np.all(U[row_pad, :m] == 0.0)))
        print("  ||Ud^T Ud - I||          : %.2e"
              % np.linalg.norm(Ud.T @ Ud - np.eye(m)))
        if V is not None:
            M = U.shape[1]
            print("  ||U^T U - I|| (all M)    : %.2e"
                  % np.linalg.norm(U.T @ U - np.eye(M)))
            print("  ||V^T V - I||            : %.2e"
                  % np.linalg.norm(V.T @ V - np.eye(M)))
            print("  ||U S V^T - A_pad||      : %.2e"
                  % np.linalg.norm(U * S @ V.T - A_pad))

    # ---------------- Demo 1: n >= M ----------------
    row_pad = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=bool)   # N=9, n=7
    col_pad = np.array([0, 0, 1, 0], dtype=bool)                  # M=4, m=3
    A_pad = build(9, 4, row_pad, col_pad, k=2)

    Un, Sn, _ = np.linalg.svd(A_pad, full_matrices=False)
    report("naive SVD            (n>=M)", A_pad, row_pad, col_pad, Un, Sn)
    U, S, V = pad_safe_svd(A_pad, row_pad, col_pad)
    report("Method D             (n>=M)", A_pad, row_pad, col_pad, U, S, V, m=3)

    # ---------------- Demo 2: n < M -----------------
    row_pad = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=bool)   # N=9, n=4
    col_pad = np.array([0, 1, 0, 0, 1], dtype=bool)               # M=5, m=3
    A_pad = build(9, 5, row_pad, col_pad, k=2, zero_data_row=False)

    Un, Sn, _ = np.linalg.svd(A_pad, full_matrices=False)
    report("naive SVD            (n< M)", A_pad, row_pad, col_pad, Un, Sn)
    U, S, V = pad_safe_svd(A_pad, row_pad, col_pad)
    report("Method D             (n< M)", A_pad, row_pad, col_pad, U, S, V, m=3)

    # unpad equivalence: first m triplets == economy SVD of the unpadded A
    A = A_pad[np.ix_(~row_pad, ~col_pad)]
    sig_true = np.linalg.svd(A, compute_uv=False)
    m = 3
    assert np.allclose(S[:m], sig_true, atol=1e-12)
    Aud = U[~row_pad][:, :m] * S[:m] @ V[~col_pad][:, :m].T
    assert np.linalg.norm(Aud - A) < 1e-12 * max(1, np.linalg.norm(A))
    assert np.all(U[row_pad][:, :m] == 0.0)          # bitwise, not eps
    assert np.all(V[col_pad][:, :m] == 0.0)          # bitwise, not eps
    print("\nunpad-equivalence and structural assertions passed")

# ----------------------------------------------------------------------
# JAX notes (batched / jit / vmap version):
#   * jnp.linalg.qr / jnp.linalg.svd have the same semantics on CPU; on
#     GPU/TPU the SVD backend differs, but Method D only calls the SVD on
#     the small augmented core, and QR lowering is Householder-based
#     across backends (the exactness claims live in the QR + matvec steps).
#   * Omega1/Omega2: draw once from a fixed PRNGKey outside the jitted fn.
#   * argsort(mask) / take_along_axis replace fancy indexing; masks are
#     runtime data of static length -> no recompiles.
#   * dynamic p_l, p_c: keep static-width augmentation blocks
#     c*jnp.diag(t) and c*jnp.diag(s) (M columns / M rows each, with zero
#     rows/cols where the indicator is 0), giving a static (2M) x (2M)
#     augmented core; the extra exact-zero triplets join the discarded /
#     don't-care pool.  Filter by (Sa > c/2) as a mask, order kept
#     triplets with argsort on (pinned_mask, -Sa), and slice statically
#     at M.  kept-count = m is then implicit in the sigma ordering.
#   * mode="complete" QR: jnp.linalg.qr(..., mode="complete").
# ----------------------------------------------------------------------
