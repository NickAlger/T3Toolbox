# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Tests for the backend rank-policy helpers (``t3toolbox.backend.ranks``).

Focus: the Section 5.4.1 rank-continuation update ``compute_continuation_ranks`` and the
``edge_condition_numbers`` it is built on. Expected new ranks are worked by hand against
``compute_minimal_ranks`` (the paper's "useless-rank removal"). numpy-only: these are pure host
arithmetic on ranks (structure), never traced under jit."""
import unittest

import numpy as np

import t3toolbox.backend.ranks as ranks


class TestEdgeConditionNumbers(unittest.TestCase):
    def test_basic_ratio_and_conventions(self):
        # descending singular values per edge; boundary bonds are length-1 -> 1.0
        tucker_sv = [np.array([4.0, 2.0, 1.0]),      # kappa = 4 / 1 = 4
                     np.array([1.0])]                # single value -> 1
        tt_sv     = [np.array([1.0]),                # boundary bond -> 1
                     np.array([10.0, 0.1]),          # kappa = 100
                     np.array([5.0, 5.0]),           # kappa = 1
                     np.array([1.0])]                # boundary bond -> 1
        kappa_tucker, kappa_tt = ranks.edge_condition_numbers(tucker_sv, tt_sv)
        self.assertEqual(kappa_tucker, (4.0, 1.0))
        self.assertEqual(kappa_tt, (1.0, 100.0, 1.0, 1.0))

    def test_degenerate_edge_conventions(self):
        # all-zero edge -> 1.0 (trivial); rank-deficient edge (sigma_1>0, sigma_k~0) -> +inf
        zero_edge = np.array([0.0, 0.0])
        deficient = np.array([1.0, 0.0])
        kappa_tucker, _ = ranks.edge_condition_numbers([zero_edge, deficient], [np.array([1.0])])
        self.assertEqual(kappa_tucker[0], 1.0)
        self.assertTrue(np.isinf(kappa_tucker[1]))


class TestComputeContinuationRanks(unittest.TestCase):
    def test_selective_growth_by_conditioning(self):
        # mode 1 is ill-conditioned (kappa_max) so it is NOT grown; every other edge is far below
        # kappa_max/tau and IS grown. Large shape -> no useless-rank removal interferes.
        shape = (20, 20, 20)
        tucker_sv = [np.array([1.0, 0.9]),       # n0=2, kappa~1.11  -> grow
                     np.array([1.0, 0.001]),     # n1=2, kappa=1000  -> kappa_max, no grow
                     np.array([1.0, 0.5])]       # n2=2, kappa=2     -> grow
        tt_sv     = [np.array([1.0]),            # r0=1 boundary
                     np.array([1.0, 0.8]),       # r1=2, kappa=1.25  -> grow
                     np.array([1.0, 0.4]),       # r2=2, kappa=2.5   -> grow
                     np.array([1.0])]            # r3=1 boundary
        new_tucker, new_tt = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0)
        self.assertEqual(new_tucker, (3, 2, 3))      # boundary bonds untouched, ill edge frozen
        self.assertEqual(new_tt, (1, 3, 3, 1))

    def test_n_chunk(self):
        shape = (20, 20, 20)
        tucker_sv = [np.array([1.0, 0.9]), np.array([1.0, 0.001]), np.array([1.0, 0.5])]
        tt_sv     = [np.array([1.0]), np.array([1.0, 0.8]), np.array([1.0, 0.4]), np.array([1.0])]
        new_tucker, new_tt = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0, n_chunk=2)
        self.assertEqual(new_tucker, (4, 2, 4))
        self.assertEqual(new_tt, (1, 4, 4, 1))

    def test_uniform_bump_fallback(self):
        # all edges equally conditioned -> nothing passes kappa < kappa_max/tau -> bump every rank,
        # then re-clean. From all-rank-1 this is how continuation gets off the ground.
        shape = (20, 20, 20)
        tucker_sv = [np.array([2.0]), np.array([2.0]), np.array([2.0])]      # all n=1, kappa=1
        tt_sv     = [np.array([1.0]), np.array([1.5]), np.array([1.5]), np.array([1.0])]  # all r=1, kappa=1
        new_tucker, new_tt = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0)
        self.assertEqual(new_tucker, (2, 2, 2))
        self.assertEqual(new_tt, (1, 2, 2, 1))

    def test_fallback_from_zero_tensor(self):
        # the zero tensor's edges are all-zero -> kappa=1 everywhere -> uniform-bump fallback (no nan)
        shape = (20, 20, 20)
        z = [np.array([0.0]), np.array([0.0]), np.array([0.0])]
        zt = [np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])]
        new_tucker, new_tt = ranks.compute_continuation_ranks(shape, z, zt, tau=10.0)
        self.assertEqual(new_tucker, (2, 2, 2))
        self.assertEqual(new_tt, (1, 2, 2, 1))

    def test_useless_rank_removal_clamps_proposed_growth(self):
        # mode 0 is ill-conditioned (frozen at n0=2); bond 1 is well-conditioned and proposed to grow
        # to 3, but useless-rank removal clamps it back to 2 (a bond cannot exceed r0*n0 = 1*2). Other
        # well-conditioned edges (mode1, mode2, bond2) still grow -> new != old, no fallback.
        shape = (20, 20, 20)
        tucker_sv = [np.array([1.0, 1e-4]),      # n0=2, kappa=1e4 -> kappa_max, frozen
                     np.array([1.0, 0.99]),      # n1=2 -> grow
                     np.array([1.0, 0.99])]      # n2=2 -> grow
        tt_sv     = [np.array([1.0]),            # r0=1 boundary
                     np.array([1.0, 0.99]),      # r1=2 -> proposed 3, clamped back to 2
                     np.array([1.0, 0.99]),      # r2=2 -> grow to 3
                     np.array([1.0])]            # r3=1 boundary
        new_tucker, new_tt = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0)
        self.assertEqual(new_tucker, (2, 3, 3))
        self.assertEqual(new_tt, (1, 2, 3, 1))
        # the result is, by construction (compute_minimal_ranks), structurally minimal
        cleaned = ranks.compute_minimal_ranks(shape, new_tucker, new_tt)
        self.assertEqual((tuple(cleaned[0]), tuple(cleaned[1])), (new_tucker, new_tt))

    def test_already_maximal_returns_unchanged(self):
        # a structure already at the shape's caps cannot grow: both the rule and the fallback clamp
        # back, so the function returns it unchanged (the caller's termination test then stops).
        shape = (2, 2, 2)
        tucker_sv = [np.array([1.0, 0.9]), np.array([1.0, 0.9]), np.array([1.0, 0.9])]  # n=2 = cap
        tt_sv     = [np.array([1.0]), np.array([1.0, 0.9]), np.array([1.0, 0.9]), np.array([1.0])]
        new_tucker, new_tt = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0)
        self.assertEqual(new_tucker, (2, 2, 2))
        self.assertEqual(new_tt, (1, 2, 2, 1))

    def test_per_edge_guard_freezes_extremely_ill_conditioned_edge(self):
        # mode 0 is catastrophic (kappa_max=1e20 -> threshold 1e19), so the relative rule alone would
        # OK growing mode 1 (kappa 1e15 < 1e19). The absolute guard (1e12) freezes mode 1 anyway. The
        # genuinely well-conditioned mode 2 / bond 2 still grow.
        shape = (20, 20, 20)
        tucker_sv = [np.array([1.0, 1e-20]),     # n0=2, kappa=1e20  -> frozen (kappa_max)
                     np.array([1.0, 1e-15]),     # n1=2, kappa=1e15  -> below threshold but >= guard -> frozen
                     np.array([1.0, 0.5])]       # n2=2, kappa=2     -> grow
        tt_sv     = [np.array([1.0]),
                     np.array([1.0, 1e-15]),     # r1=2, kappa=1e15  -> frozen by guard
                     np.array([1.0, 0.5]),       # r2=2, kappa=2     -> grow
                     np.array([1.0])]
        new_tucker, new_tt = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0)
        self.assertEqual(new_tucker, (2, 2, 3))      # mode 1 held at 2 by the guard
        self.assertEqual(new_tt, (1, 2, 3, 1))
        # raising the guard above 1e15 lets the relative rule grow mode 1 / bond 1 again
        nt2, ntt2 = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0, kappa_guard=1e16)
        self.assertEqual(nt2, (2, 3, 3))
        self.assertEqual(ntt2, (1, 2, 3, 1))

    def test_global_guard_stops_when_all_edges_too_ill_conditioned(self):
        # every internal edge is extremely ill conditioned (kappa 1e15 >= guard 1e12): the relative rule
        # grows nothing and the fallback bumps nothing -> ranks unchanged -> caller stops. With a guard
        # above 1e15 the (comparably conditioned) edges instead trigger the uniform-bump fallback.
        shape = (20, 20, 20)
        tucker_sv = [np.array([1.0, 1e-15]), np.array([1.0, 1e-15]), np.array([1.0, 1e-15])]
        tt_sv     = [np.array([1.0]), np.array([1.0, 1e-15]), np.array([1.0, 1e-15]), np.array([1.0])]
        stopped = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0)
        self.assertEqual(stopped, ((2, 2, 2), (1, 2, 2, 1)))           # unchanged -> stop
        grown = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0, kappa_guard=1e16)
        self.assertEqual(grown, ((3, 3, 3), (1, 3, 3, 1)))            # guard lifted -> fallback bumps all

    def _sv(self, kappa, k):
        # k descending singular values with condition number sigma_1/sigma_k = kappa (k>=1 -> 1.0)
        return np.linspace(1.0, 1.0 / kappa, k)

    def test_max_grow_one_grows_single_best_conditioned_edge(self):
        # current ranks tucker (3,4,4,3), tt (1,3,4,3,1): the single-edge-growable interior edges are
        # n1, n2, r2 (the outer ones are structurally capped). mode 0 is the worst edge (frozen).
        shape = (20, 20, 20, 20)
        tucker_sv = [self._sv(1000., 3), self._sv(2.0, 4), self._sv(3.0, 4), self._sv(5.0, 3)]
        tt_sv     = [self._sv(1., 1), self._sv(4.0, 3), self._sv(2.5, 4), self._sv(5.0, 3), self._sv(1., 1)]
        full = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0)
        one  = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0, max_grow=1)
        # max_grow=1 grows exactly the best-conditioned growable edge: n1 (kappa 2.0)
        self.assertEqual(one, ((3, 5, 4, 3), (1, 3, 4, 3, 1)))
        # the uncapped rule grows several edges -> strictly more than the one-at-a-time result
        self.assertNotEqual(full, one)
        self.assertGreater(sum(full[0]) + sum(full[1]), sum(one[0]) + sum(one[1]))

    def test_max_grow_skips_structurally_capped_edge(self):
        # bond 1 has the smallest condition number but is structurally capped (r1 <= r0*n0 = 3); the
        # greedy must skip it and grow the next-best growable edge, n1 (kappa 2.0).
        shape = (20, 20, 20, 20)
        tucker_sv = [self._sv(1000., 3), self._sv(2.0, 4), self._sv(3.0, 4), self._sv(5.0, 3)]
        tt_sv     = [self._sv(1., 1), self._sv(1.5, 3), self._sv(2.5, 4), self._sv(5.0, 3), self._sv(1., 1)]
        one = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0, max_grow=1)
        self.assertEqual(one, ((3, 5, 4, 3), (1, 3, 4, 3, 1)))   # grew n1, not the capped bond 1

    def test_max_grow_two_grows_two_edges(self):
        shape = (20, 20, 20, 20)
        tucker_sv = [self._sv(1000., 3), self._sv(2.0, 4), self._sv(3.0, 4), self._sv(5.0, 3)]
        tt_sv     = [self._sv(1., 1), self._sv(4.0, 3), self._sv(2.5, 4), self._sv(5.0, 3), self._sv(1., 1)]
        two = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0, max_grow=2)
        self.assertEqual(two, ((3, 5, 4, 3), (1, 3, 5, 3, 1)))   # grew n1 (kappa 2.0) and r2 (kappa 2.5)

    def test_max_grow_fallback_uniform_escapes_degenerate_start(self):
        # from all-ones no SINGLE edge can grow, so the (uncapped) uniform-bump fallback fires even with
        # max_grow=1 -- otherwise one-at-a-time continuation could never get off the ground.
        shape = (20, 20, 20)
        z  = [self._sv(1., 1), self._sv(1., 1), self._sv(1., 1)]
        zt = [self._sv(1., 1), self._sv(1., 1), self._sv(1., 1), self._sv(1., 1)]
        got = ranks.compute_continuation_ranks(shape, z, zt, tau=1.0, max_grow=1)
        self.assertEqual(got, ((2, 2, 2), (1, 2, 2, 1)))

    def test_all_rank_deficient_stops(self):
        # every edge rank-deficient (kappa = +inf): nothing is below the guard -> ranks unchanged -> stop
        # (the guard prevents the slice-1 behavior of uniformly inflating an already-degenerate tensor).
        shape = (20, 20, 20)
        tucker_sv = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        tt_sv     = [np.array([1.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0])]
        stopped = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0)
        self.assertEqual(stopped, ((2, 2, 2), (1, 2, 2, 1)))


if __name__ == "__main__":
    unittest.main()
