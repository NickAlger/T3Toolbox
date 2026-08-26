"""Q2 check: the SYMMETRIC contract needs no runtime transpose.

Claim: Method D's `n >= m` restriction is only a contract statement. With one change --
zero the pinned sigmas BY FLAG (not by position m) -- the same pipeline delivers:

    first q = min(n, m) triplets  ==  economy SVD of the unpadded A,
    U[:, :q] bitwise zero on pad rows,  V[:, :q] bitwise zero on pad cols,
    remaining triplets sigma = 0 (don't-care), full U/V orthonormal, exact reconstruction

for ANY real (n, m) -- tall, wide, or mixed within a batch -- with no branch on traced data.
The only transpose is the STATIC one (padded N < M), a Python `if` on concrete shapes.
"""
import numpy as np

def _haar(M, rng):
    Q, R = np.linalg.qr(rng.standard_normal((M, M)))
    return Q * np.sign(np.diag(R))

def _tall(A_pad, row_pad, col_pad, Omega, c_factor=4.0, use_spectral=False):
    N, M = A_pad.shape
    n = N - int(row_pad.sum()); m = M - int(col_pad.sum())
    Y = A_pad @ Omega
    pr = np.argsort(row_pad, kind='stable')
    Qp, _ = np.linalg.qr(Y[pr]); Q = Qp[np.argsort(pr)]
    t = (np.abs(Q) * row_pad[:, None]).max(axis=0) > 0.5
    B = Q.T @ A_pad
    nA = np.linalg.norm(A_pad, 2) if use_spectral else np.linalg.norm(A_pad)
    c = c_factor * nA + (nA == 0.0)
    C_aug = np.concatenate([B, c * np.diag(t.astype(float))], axis=1)
    W1, Sa, _ = np.linalg.svd(C_aug, full_matrices=False)
    pinned = Sa > c / 2.0
    order = np.argsort(np.where(pinned, -1.0, Sa), kind='stable')[::-1]
    W1 = W1[:, order]; So, pino = Sa[order], pinned[order]
    W1 = W1 * (1.0 - np.outer(t, ~pino))
    U = Q @ W1
    q = min(n, m)
    S = np.where(pino, 0.0, So)                    # <-- the one change: zero pins BY FLAG
    S = np.where(np.arange(M) < q, S, 0.0)         #     then zero the sigma~0 tail exactly
    Wv = A_pad.T @ U
    pc = np.argsort(col_pad, kind='stable')
    Vq, Rv = np.linalg.qr(Wv[pc]); V = Vq[np.argsort(pc)]
    d = np.sign(np.diagonal(Rv)); V = V * np.where(d == 0.0, 1.0, d)
    return U, S, V

def pad_safe_svd_sym(A_pad, row_pad, col_pad, seed=0, **kw):
    """Symmetric contract; static transpose only (Python `if` on concrete shapes)."""
    N, M = A_pad.shape
    rng = np.random.default_rng(seed)
    if N < M:                                      # STATIC branch -- concrete shapes, jit-fine
        U, S, V = _tall(A_pad.T, col_pad, row_pad, _haar(N, rng), **kw)
        return V, S, U
    return _tall(A_pad, row_pad, col_pad, _haar(M, rng), **kw)

def check_sym(A_pad, row_pad, col_pad, U, S, V, tol=1e-11):
    N, M = A_pad.shape; K = min(N, M)
    n = int((~row_pad).sum()); m = int((~col_pad).sum()); q = min(n, m)
    A = A_pad[np.ix_(~row_pad, ~col_pad)]
    sA = max(1.0, np.linalg.norm(A_pad))
    assert np.all(U[row_pad][:, :q] == 0.0), "U pad rows not bitwise zero"
    assert np.all(V[col_pad][:, :q] == 0.0), "V pad coords not bitwise zero"
    assert np.linalg.norm(U.T @ U - np.eye(K)) < tol, "U not orthonormal"
    assert np.linalg.norm(V.T @ V - np.eye(K)) < tol, "V not orthonormal"
    assert np.linalg.norm(U * S @ V.T - A_pad) < 100 * tol * sA, "reconstruction"
    if q:
        sig = np.linalg.svd(A, compute_uv=False)[:q]
        assert np.allclose(S[:q], sig, atol=100 * tol * sA), "sigma mismatch"
    assert np.all(S[q:] == 0.0), "tail sigmas not zeroed"
    # real block of U orthonormal at full q (no lost directions)
    if q:
        Ud = U[~row_pad][:, :q]
        assert np.linalg.norm(Ud.T @ Ud - np.eye(q)) < tol, "real block lost rank"

rng = np.random.default_rng(1)
counts = {'tall': 0, 'wide(n<m)': 0, 'N<M static': 0}
for trial in range(600):
    N = int(rng.integers(2, 18)); M = int(rng.integers(1, 18))
    row_pad = rng.random(N) < rng.uniform(0, 0.6)
    col_pad = rng.random(M) < rng.uniform(0, 0.6)
    n = int((~row_pad).sum()); m = int((~col_pad).sum())
    k = int(rng.integers(0, min(n, m) + 1))
    A_pad = np.zeros((N, M))
    if k:
        G = rng.standard_normal((n, k)) @ rng.standard_normal((k, m))
        if n > 1 and rng.random() < 0.5:
            G[rng.integers(n), :] = 0.0            # numerically-zero data row
        if m > 1 and rng.random() < 0.3:
            G[:, rng.integers(m)] = 0.0            # numerically-zero data column
        A_pad[np.ix_(~row_pad, ~col_pad)] = G
    U, S, V = pad_safe_svd_sym(A_pad, row_pad, col_pad, seed=trial)
    check_sym(A_pad, row_pad, col_pad, U, S, V)
    counts['N<M static' if N < M else ('wide(n<m)' if n < m else 'tall')] += 1
print("symmetric-contract stress: 600/600 passed;", counts)
