"""Tucker site: M is N x n (rows = mode index, suffix-padded beyond N_i; columns = Tucker rank slots, prefix-
masked beyond n_i). One SVD of [M | eps*C], C = the first n coordinate vectors masked by the shape mask; keep
the first n left singular vectors U; remainder = U^T M. Check: U orthonormal, real-supported, range(M) ⊆ range(U)
(tensor exact), and the sigma>0 vectors unperturbed."""
import numpy as np
rng = np.random.default_rng(0)
worst = dict(orth=0, pad=0, recon=0, kept=0); n_cases = 0
for trial in range(400):
    N = int(rng.integers(4, 30)); n = int(rng.integers(1, min(6, N + 1)))   # n <= N always (n = max n_i <= max N_i = N)
    Ni = int(rng.integers(1, N + 1))                    # real mode size (suffix padding beyond it)
    ni = int(rng.integers(1, min(n, Ni) + 1))           # real Tucker rank (prefix mask)
    r = int(rng.integers(0, ni + 1))                    # numerical rank of the real block (r < ni = deficient)
    M = np.zeros((N, n)); M[:Ni, :r] = rng.standard_normal((Ni, r)) @ rng.standard_normal((r, r))
    M[:Ni, :ni] = M[:Ni, :ni] @ np.linalg.qr(rng.standard_normal((ni, ni)))[0]    # mix the real columns (rank r)
    real_rows = np.arange(N) < Ni
    C = np.eye(N, n) * real_rows[:, None]               # first n coordinate vectors, masked to the real rows
    smax = max(np.linalg.norm(M, 2), 1e-300); eps = 1e-10 * smax if r else 1.0
    U, s, _ = np.linalg.svd(np.concatenate([M, eps * C], axis=1), full_matrices=False)
    U = U[:, :n]
    U0, s0, _ = np.linalg.svd(M, full_matrices=False)
    # (a) orthonormal, (b) real-supported in every column that the mask calls real (k < ni) -- and beyond, (c) range
    worst['orth'] = max(worst['orth'], np.abs(U.T @ U - np.eye(n)).max())
    worst['pad'] = max(worst['pad'], np.abs(U[~real_rows, :ni]).max(initial=0.0))
    worst['recon'] = max(worst['recon'], np.linalg.norm(U @ (U.T @ M) - M) / smax)
    if r: worst['kept'] = max(worst['kept'], 1 - np.abs(np.sum(U[:, :r] * U0[:, :r], axis=0)).min())
    n_cases += 1
print('%d random Tucker-site cases (N up to 30, n up to 5, N_i in [1, N], real rank n_i <= min(n, N_i), numerical rank r <= n_i, incl. r = 0):' % n_cases)
print('   max |U^T U - I|            = %.1e' % worst['orth'])
print('   max |U| on PADDED rows     = %.1e   (the n_i REAL columns are real-supported; slots k >= n_i are masked anyway)' % worst['pad'])
print('   max |U U^T M - M| / |M|    = %.1e   (range(M) inside range(U): the tensor is exact)' % worst['recon'])
print('   max 1 - |<U_k, U0_k>|, k<r = %.1e   (the sigma > 0 vectors unperturbed for sigma >> eps)' % worst['kept'])
