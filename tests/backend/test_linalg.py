# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Tests for ``backend/linalg.py`` -- currently the ``pad_safe_svd`` invariant suite.

(The ``truncated_svd`` / directional-SVD family is covered by its module doctests and by every
downstream orthogonalization/SVD test; what needs its own suite is ``pad_safe_svd``, whose
guarantees -- BITWISE pad avoidance, tolerance-free counting, the symmetric ``min(n, m)``
contract -- are exactly the properties nothing downstream would pin precisely.)

The invariants follow the pad-safe SVD design packet (``dev/review_2026-08-22/repros/S1b/packet/``):

  1. ``U[pad_rows][:, :q] == 0.0`` and ``Vt[:q, :][:, pad_cols] == 0.0`` -- **bitwise**, not small;
  2. all ``K`` columns of ``U`` / rows of ``Vt`` orthonormal to ~1e-14 (float64);
  3. ``ss[:q]`` matches ``numpy.linalg.svd`` of the unpadded real block; ``ss[q:] == 0.0`` exactly;
  4. ``U @ diag(ss) @ Vt`` reconstructs the padded matrix;
  5. the real block of ``U[:, :q]`` is orthonormal at full ``q`` -- NO lost directions (the S1b
     failure mode this function exists to fix);
  6. the ``c = 4||A||_F`` margin keeps the largest triplet on generic rank-1 data (``c = 2 sigma_max``
     deletes it in ~38%% of cases -- ``dev/review_2026-08-22/repros/S1b/s1b_c_study.py``).

numpy-only (house testing strategy); jit dispatch + the compile-once-across-mask-patterns property
are covered in ``tests/test_dispatch.py``.
"""
import unittest

import numpy as np

import t3toolbox.backend.linalg as linalg


def random_case(rng, N, M, pad_density=0.6):
    """Random interior masks (no n >= m constraint), random rank, optional zero data rows/cols."""
    row_pad = rng.random(N) < rng.uniform(0, pad_density)
    col_pad = rng.random(M) < rng.uniform(0, pad_density)
    n, m = int((~row_pad).sum()), int((~col_pad).sum())
    k = int(rng.integers(0, min(n, m) + 1))
    A = np.zeros((N, M))
    if k:
        G = rng.standard_normal((n, k)) @ rng.standard_normal((k, m))
        if n > 1 and rng.random() < 0.5:
            G[rng.integers(n), :] = 0.0     # a data row that is numerically zero (z vs p: allowed)
        if m > 1 and rng.random() < 0.3:
            G[:, rng.integers(m)] = 0.0
        A[np.ix_(~row_pad, ~col_pad)] = G
    return A, ~row_pad, ~col_pad            # library polarity: True = real


class TestPadSafeSvd(unittest.TestCase):

    def check_contract(self, A, row_mask, col_mask, U, ss, Vt, tol=1e-11):
        """The symmetric contract: first q = min(n, m) triplets are the SVD of the real block."""
        N, M = A.shape[-2:]
        K = min(N, M)
        n = int(row_mask.sum()); m = int(col_mask.sum()); q = min(n, m)
        V = Vt.T
        real = A[np.ix_(row_mask, col_mask)]
        sA = max(1.0, np.linalg.norm(A))
        self.assertTrue(np.all(U[~row_mask][:, :q] == 0.0), "U pad rows not bitwise zero")
        self.assertTrue(np.all(V[~col_mask][:, :q] == 0.0), "Vt pad coords not bitwise zero")
        self.assertLess(np.linalg.norm(U.T @ U - np.eye(K)), tol, "U not orthonormal")
        self.assertLess(np.linalg.norm(V.T @ V - np.eye(K)), tol, "Vt not orthonormal")
        self.assertLess(np.linalg.norm(U * ss @ Vt - A), 100 * tol * sA, "reconstruction failed")
        if q:
            sig = np.linalg.svd(real, compute_uv=False)[:q]
            self.assertTrue(np.allclose(ss[:q], sig, atol=100 * tol * sA), "sigma mismatch")
            Ur = U[row_mask][:, :q]
            self.assertLess(np.linalg.norm(Ur.T @ Ur - np.eye(q)), tol,
                            "real block of U lost rank (the S1b failure)")
        self.assertTrue(np.all(ss[q:] == 0.0), "don't-care sigmas not exactly zero")

    def test_stress_random(self):
        """300 random cases: tall / wide / statically-wide, interior pads, zero data rows/cols."""
        rng = np.random.default_rng(1)
        counts = {'tall': 0, 'wide(n<m)': 0, 'static wide (N<M)': 0}
        for trial in range(300):
            N = int(rng.integers(2, 18)); M = int(rng.integers(1, 18))
            A, rm, cm = random_case(rng, N, M)
            with self.subTest(trial=trial, N=N, M=M, n=int(rm.sum()), m=int(cm.sum())):
                U, ss, Vt = linalg.pad_safe_svd(A, rm, cm)
                self.check_contract(A, rm, cm, U, ss, Vt)
            counts['static wide (N<M)' if N < M else
                   ('wide(n<m)' if rm.sum() < cm.sum() else 'tall')] += 1
        for regime, hits in counts.items():                    # the sweep must exercise all three
            self.assertGreater(hits, 15, f"sweep failed to exercise the {regime} regime")

    def test_batched_matches_per_element(self):
        """Multi-axis leading batch with a DIFFERENT mask pattern per element == per-element calls."""
        rng = np.random.default_rng(2)
        S1, S2, N, M = 2, 3, 9, 4
        A = np.zeros((S1, S2, N, M))
        rms = np.zeros((S1, S2, N), bool)
        cms = np.zeros((S1, S2, M), bool)
        for i in range(S1):
            for j in range(S2):
                A[i, j], rms[i, j], cms[i, j] = random_case(rng, N, M, pad_density=0.5)
        Ub, sb, Vtb = linalg.pad_safe_svd(A, rms, cms)
        self.assertEqual(Ub.shape, (S1, S2, N, M))
        for i in range(S1):
            for j in range(S2):
                with self.subTest(elem=(i, j)):
                    self.check_contract(A[i, j], rms[i, j], cms[i, j], Ub[i, j], sb[i, j], Vtb[i, j])
                    _, s1, _ = linalg.pad_safe_svd(A[i, j], rms[i, j], cms[i, j])
                    self.assertTrue(np.allclose(sb[i, j], s1, atol=1e-12),
                                    "batched sigmas differ from the per-element call")

    def test_unbatched_masks_broadcast_over_batched_A(self):
        """One shared mask pattern, stacked matrices (the common sweep-site configuration)."""
        rng = np.random.default_rng(3)
        A0, rm, cm = random_case(rng, 8, 5, pad_density=0.5)
        A1, _, _ = random_case(rng, 8, 5, pad_density=0.0)
        A1p = np.zeros((8, 5)); A1p[np.ix_(rm, cm)] = A1[np.ix_(rm, cm)]
        A = np.stack([A0, A1p])
        U, ss, Vt = linalg.pad_safe_svd(A, rm, cm)
        for b in range(2):
            with self.subTest(elem=b):
                self.check_contract(A[b], rm, cm, U[b], ss[b], Vt[b])

    def test_edge_cases(self):
        cases = {
            'A = 0 with pads':      (np.zeros((5, 3)), np.array([1, 0, 1, 0, 1], bool), np.array([1, 0, 1], bool)),
            'all columns padded':   (np.zeros((4, 2)), np.ones(4, bool), np.zeros(2, bool)),
            'no pads at all':       (np.arange(12.0).reshape(4, 3), np.ones(4, bool), np.ones(3, bool)),
            'single column':        (np.array([[1.0], [0.0], [2.0]]), np.array([1, 0, 1], bool), np.ones(1, bool)),
            'square N == M':        (np.diag([3.0, 0.0, 0.0]), np.array([1, 1, 0], bool), np.array([1, 1, 0], bool)),
        }
        rng = np.random.default_rng(4)
        A, rm, cm = np.zeros((6, 4)), np.array([0, 1, 1, 0, 1, 0], bool), np.array([1, 0, 1, 1], bool)
        A[np.ix_(rm, cm)] = rng.standard_normal((3, 3))
        cases['n == m'] = (A, rm, cm)
        for name, (A, rm, cm) in cases.items():
            with self.subTest(case=name):
                U, ss, Vt = linalg.pad_safe_svd(A, rm, cm)
                self.check_contract(A, rm, cm, U, ss, Vt)

    def test_rank1_margin_regression(self):
        """Generic rank-1 data: the top triplet must survive classification (the c-margin killer).

        With ``c = 2 * sigma_max`` the {pin, data} threshold sits exactly at ``sigma_max`` and
        one-ulp rounding classifies the LARGEST triplet as a pin, silently zeroing it -- measured
        ~38%% of generic rank-1 cases. ``c = 4 * ||A||_F`` must get 100%%."""
        rng = np.random.default_rng(5)
        rm = ~np.array([0, 0, 1, 0, 0, 1, 0], bool)
        cm = ~np.array([0, 0, 1], bool)
        for trial in range(100):
            u = rng.standard_normal(5); v = rng.standard_normal(2)
            A = np.zeros((7, 3)); A[np.ix_(rm, cm)] = np.outer(u, v)
            sig = np.linalg.norm(u) * np.linalg.norm(v)
            with self.subTest(trial=trial):
                _, ss, _ = linalg.pad_safe_svd(A, rm, cm)
                self.assertTrue(np.isclose(ss[0], sig), "largest triplet was misclassified as a pin")

    def test_adversarial_spectra(self):
        """Repeated sigmas and a tiny genuine sigma (1e-8): kept values exact, pads still bitwise 0."""
        rng = np.random.default_rng(6)
        for trial in range(60):
            N, M = 10, 5
            row_pad = rng.random(N) < 0.4
            col_pad = rng.random(M) < 0.4
            n, m = int((~row_pad).sum()), int((~col_pad).sum())
            k = int(rng.integers(0, min(n, m) + 1))
            A = np.zeros((N, M))
            if k:
                sv = np.sort(rng.choice([3.0, 3.0, 1.0, 1e-8], k))[::-1]
                Uo, _ = np.linalg.qr(rng.standard_normal((n, n)))
                Vo, _ = np.linalg.qr(rng.standard_normal((m, m)))
                A[np.ix_(~row_pad, ~col_pad)] = Uo[:, :k] * sv @ Vo[:, :k].T
            with self.subTest(trial=trial, k=k):
                U, ss, Vt = linalg.pad_safe_svd(A, ~row_pad, ~col_pad)
                self.check_contract(A, ~row_pad, ~col_pad, U, ss, Vt)

    def test_warm_start_completion(self):
        """The S1b raison d'etre: a zero-padded warm start (exact zero rows AND columns inside the
        real block) keeps a full orthonormal real block -- the completion is real-supported."""
        rng = np.random.default_rng(7)
        N, M = 9, 4
        rm = ~np.array([0, 0, 1, 0, 0, 0, 1, 0, 0], bool)      # n = 7
        cm = np.ones(M, bool)                                  # m = 4, all structural
        A = np.zeros((N, M))
        G = np.zeros((7, 4))
        G[:5, :2] = rng.standard_normal((5, 2))                # real rank 2 of mask rank 4:
        A[np.ix_(rm, cm)] = G                                  #   2 exact-zero columns (the padding
        U, ss, Vt = linalg.pad_safe_svd(A, rm, cm)             #   of a rank-continuation restart)
        self.check_contract(A, rm, cm, U, ss, Vt)
        Ur = U[rm]                                             # 7 x 4 real block, orthonormal at 4:
        self.assertLess(np.linalg.norm(Ur.T @ Ur - np.eye(4)), 1e-12)
        self.assertLess(np.max(np.abs(ss[2:4])), 1e-13)        # deficient sigmas ~0 (computed)
        self.assertTrue(np.all(ss[4:] == 0.0))                 # beyond q: exact by construction

    def test_deterministic(self):
        """Fixed sketch: the same input gives the same output, bitwise, across calls."""
        rng = np.random.default_rng(8)
        A, rm, cm = random_case(rng, 8, 5)
        U1, s1, Vt1 = linalg.pad_safe_svd(A, rm, cm)
        U2, s2, Vt2 = linalg.pad_safe_svd(A.copy(), rm.copy(), cm.copy())
        self.assertTrue(np.array_equal(U1, U2) and np.array_equal(s1, s2) and np.array_equal(Vt1, Vt2))


if __name__ == '__main__':
    unittest.main()
