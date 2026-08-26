"""
Test suite for the pad-safe SVD. Run:  pytest tests/ -q

Covers the invariants listed in ALGORITHM.md for both implementations:
bitwise pad-avoidance, orthonormality, unpad-equivalence, reconstruction,
edge cases (rank 0, A = 0, all columns padded, n = m, n < M), adversarial
spectra (repeated and tiny singular values), and single-compile behavior
of the JAX version under heterogeneous mask patterns.
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pad_safe_svd import pad_safe_svd  # noqa: E402

jax = pytest.importorskip("jax", reason="JAX tests skipped (not installed)")
import jax.numpy as jnp  # noqa: E402
jax.config.update("jax_enable_x64", True)
from pad_safe_svd_jax import pad_safe_svd_jax, make_sketch  # noqa: E402


# ---------------------------------------------------------------- helpers
def random_case(rng, N, M, force_n_lt_M=False, spectrum=None):
    """Random masks (interior pads), rank, data; honors only n >= m."""
    col_pad = rng.random(M) < rng.uniform(0, 0.7)
    m = int(M - col_pad.sum())
    lo, hi = max(m, 1), N + 1
    if force_n_lt_M and m < M and M > 1:
        hi = max(lo + 1, M)                      # n in [max(m,1), M)
    n = int(rng.integers(lo, hi))
    row_pad = np.zeros(N, bool)
    row_pad[rng.permutation(N)[: N - n]] = True
    k = int(rng.integers(0, m + 1)) if m else 0
    A = np.zeros((N, M))
    if m and k:
        if spectrum is None:
            G = rng.standard_normal((n, k)) @ rng.standard_normal((k, m))
            if n > 1 and rng.random() < 0.5:
                G[rng.integers(n), :] = 0.0      # numerically-zero data row
        else:
            Uo, _ = np.linalg.qr(rng.standard_normal((n, n)))
            Vo, _ = np.linalg.qr(rng.standard_normal((m, m)))
            sv = np.sort(rng.choice(spectrum, k))[::-1]
            G = Uo[:, :k] * sv @ Vo[:, :k].T
        A[np.ix_(~row_pad, ~col_pad)] = G
    return A, row_pad, col_pad


def check_contract(A_pad, row_pad, col_pad, U, S, V, tol=1e-11):
    N, M = A_pad.shape
    m = int(M - col_pad.sum())
    A = A_pad[np.ix_(~row_pad, ~col_pad)]
    sA = max(1.0, np.linalg.norm(A_pad))
    assert np.all(U[row_pad][:, :m] == 0.0), "U pad rows not bitwise zero"
    assert np.all(V[col_pad][:, :m] == 0.0), "V pad coords not bitwise zero"
    assert np.linalg.norm(U.T @ U - np.eye(M)) < tol, "U not orthonormal"
    assert np.linalg.norm(V.T @ V - np.eye(M)) < tol, "V not orthogonal"
    assert np.linalg.norm(U * S @ V.T - A_pad) < 100 * tol * sA, "reconstruction"
    if min(A.shape) > 0:
        sig = np.linalg.svd(A, compute_uv=False)[:m]
        assert np.allclose(S[:m], sig, atol=100 * tol * sA), "sigma mismatch"
    assert np.all(S[m:] == 0.0), "don't-care sigmas not zeroed"


# ----------------------------------------------------------------- numpy
def test_numpy_stress():
    rng = np.random.default_rng(1)
    for trial in range(300):
        N = int(rng.integers(3, 22)); M = int(rng.integers(1, N + 1))
        A, rp, cp = random_case(rng, N, M)
        U, S, V = pad_safe_svd(A, rp, cp, seed=trial)
        check_contract(A, rp, cp, U, S, V)


def test_numpy_n_lt_M():
    rng = np.random.default_rng(2)
    hit = 0
    for trial in range(150):
        N = int(rng.integers(4, 16)); M = int(rng.integers(2, N + 1))
        A, rp, cp = random_case(rng, N, M, force_n_lt_M=True)
        hit += (N - rp.sum()) < M
        U, S, V = pad_safe_svd(A, rp, cp, seed=trial)
        check_contract(A, rp, cp, U, S, V)
    assert hit > 30, "sweep failed to exercise n < M"


def test_numpy_edges():
    # A = 0 with pads; all columns padded; no pads at all; n == m
    for A, rp, cp in [
        (np.zeros((5, 3)), np.array([0, 1, 0, 1, 0], bool), np.array([0, 1, 0], bool)),
        (np.zeros((4, 2)), np.zeros(4, bool), np.ones(2, bool)),
        (np.arange(12.0).reshape(4, 3), np.zeros(4, bool), np.zeros(3, bool)),
    ]:
        U, S, V = pad_safe_svd(A, rp, cp)
        check_contract(A, rp, cp, U, S, V)
    rng = np.random.default_rng(3)                      # n == m edge
    A, rp, cp = np.zeros((6, 4)), np.array([1, 0, 0, 1, 0, 1], bool), np.array([0, 1, 0, 0], bool)
    A[np.ix_(~rp, ~cp)] = rng.standard_normal((3, 3))
    U, S, V = pad_safe_svd(A, rp, cp)
    check_contract(A, rp, cp, U, S, V)


def test_numpy_rank1_margin_regression():
    """c/2 must sit strictly above sigma_max: rank-1 with clean norm."""
    A = np.zeros((5, 3))
    A[[0, 1], 0] = [3.0, 4.0]; A[[0, 1], 1] = [6.0, 8.0]   # sigma = 5*sqrt(5)? no: rank1
    rp = np.array([0, 0, 1, 0, 1], bool); cp = np.array([0, 0, 1], bool)
    U, S, V = pad_safe_svd(A, rp, cp)
    check_contract(A, rp, cp, U, S, V)
    assert S[0] > 1.0, "largest triplet was misclassified as a pin"


# ------------------------------------------------------------------- jax
def test_jax_stress_with_adversarial_spectra():
    rng = np.random.default_rng(3)
    for N, M in [(6, 3), (8, 5), (12, 4)]:
        Om = make_sketch(M)
        g = jax.jit(lambda A, r, c: pad_safe_svd_jax(A, r, c, Om))
        for trial in range(80):
            A, rp, cp = random_case(rng, N, M,
                                    spectrum=np.array([3.0, 3.0, 1.0, 1e-8]))
            U, S, V = map(np.asarray, g(jnp.asarray(A), jnp.asarray(rp),
                                        jnp.asarray(cp)))
            check_contract(A, rp, cp, U, S, V)
        assert g._cache_size() == 1, "mask patterns triggered a recompile"


def test_jax_scan_single_compile():
    N, M, BATCH = 10, 5, 6
    Om = make_sketch(M)

    @jax.jit
    def f(A_stack, r_stack, c_stack):
        def body(carry, x):
            return carry, pad_safe_svd_jax(*x, Om)
        return jax.lax.scan(body, None, (A_stack, r_stack, c_stack))[1]

    rng = np.random.default_rng(7)
    for call in range(2):                      # second call: new mask patterns
        As, rs, cs = [], [], []
        for _ in range(BATCH):
            A, rp, cp = random_case(rng, N, M)
            As.append(A); rs.append(rp); cs.append(cp)
        U, S, V = map(np.asarray, f(jnp.asarray(np.stack(As)),
                                    jnp.asarray(np.stack(rs)),
                                    jnp.asarray(np.stack(cs))))
        for b in range(BATCH):
            check_contract(As[b], rs[b], cs[b], U[b], S[b], V[b])
    assert f._cache_size() == 1, "scan recompiled on new mask patterns"


def test_jax_matches_numpy_sigmas():
    rng = np.random.default_rng(11)
    N, M = 9, 4
    Om = make_sketch(M)
    for trial in range(40):
        A, rp, cp = random_case(rng, N, M)
        m = int(M - cp.sum())
        _, S1, _ = pad_safe_svd(A, rp, cp, seed=trial)
        _, S2, _ = pad_safe_svd_jax(jnp.asarray(A), jnp.asarray(rp),
                                    jnp.asarray(cp), Om)
        assert np.allclose(S1[:m], np.asarray(S2)[:m], atol=1e-10)
