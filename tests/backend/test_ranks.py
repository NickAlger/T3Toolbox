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


class TestSharedMinimalRanks(unittest.TestCase):
    """Shared (tied-Tucker-factor) minimal ranks: the group ceiling
    ``n_g <= min(N_g, sum_{i in g} min(N_g, rL_i*rR_i))``. Hand-worked cases here; every structure is
    ALSO cross-validated against the dense edge-cut ranks of a random tied T3 in
    tests/test_sharing.py (this file stays pure host arithmetic on ranks)."""

    def test_group_ceiling_keeps_rank_the_per_mode_reduction_clips(self):
        # group {0,1}: per-mode ceiling at mode 0 is rL*rR = 1*2 = 2 < 4, but the ceilings ADD across
        # the group: min(6, min(6,2) + min(6,4)) = 6 >= 4, so the shared rank 4 survives. The unshared
        # reduction clips mode 0 to 2 (and thereby UNTIES the group); mode 2's singleton cap 3 -> 2.
        self.assertEqual(ranks.compute_minimal_ranks((6, 6, 4), (4, 4, 3), (1, 2, 2, 1)),
                         ((2, 4, 2), (1, 2, 2, 1)))
        self.assertEqual(ranks.compute_minimal_ranks((6, 6, 4), (4, 4, 3), (1, 2, 2, 1),
                                                     sharing=(0, 0, 1)),
                         ((4, 4, 2), (1, 2, 2, 1)))

    def test_group_ceiling_sum_binds(self):
        # here the SUM binds before the mode size: min(9, min(9,2) + min(9,4)) = 6 < 9, so n_g: 7 -> 6
        self.assertEqual(ranks.compute_minimal_ranks((9, 9, 4), (7, 7, 2), (1, 2, 2, 1),
                                                     sharing=(0, 0, 1)),
                         ((6, 6, 2), (1, 2, 2, 1)))

    def test_all_modes_one_group(self):
        # ceiling min(7, min(7,2)+min(7,4)+min(7,2)) = 7: nothing clips even though every per-mode
        # ceiling (2, 4, 2) is far below 7
        self.assertEqual(ranks.compute_minimal_ranks((7, 7, 7), (7, 7, 7), (1, 2, 2, 1),
                                                     sharing=(0, 0, 0)),
                         ((7, 7, 7), (1, 2, 2, 1)))

    def test_non_adjacent_group_with_tt_reduction(self):
        # TT proposals 6 reduce (r1 <= n0*r0 = 2, r2 <= n2*r3 = 2); the non-adjacent group {0,2}
        # keeps n_g = 2 (ceiling min(5, 2+2) = 4 >= 2)
        self.assertEqual(ranks.compute_minimal_ranks((5, 4, 5), (2, 3, 2), (1, 6, 6, 1),
                                                     sharing=(0, 1, 0)),
                         ((2, 3, 2), (1, 2, 2, 1)))

    def test_mode_size_cap_applies_group_wide(self):
        # proposal n_g = 5 = N_g at the group, 4 > 3 = rL*rR at the trailing singleton
        self.assertEqual(ranks.compute_minimal_ranks((5, 5, 4), (5, 5, 4), (1, 3, 3, 1),
                                                     sharing=(0, 0, 1)),
                         ((5, 5, 3), (1, 3, 3, 1)))

    def test_none_and_all_singletons_match_unshared_exactly(self):
        rng = np.random.default_rng(0)
        for trial in range(50):
            d = int(rng.integers(2, 6))
            shape = tuple(int(v) for v in rng.integers(2, 7, size=d))
            tucker = tuple(int(v) for v in rng.integers(1, 9, size=d))
            tt = (1,) + tuple(int(v) for v in rng.integers(1, 9, size=d - 1)) + (1,)
            with self.subTest(shape=shape, tucker=tucker, tt=tt):
                unshared = ranks.compute_minimal_ranks(shape, tucker, tt)
                singletons = ranks.compute_minimal_ranks(shape, tucker, tt,
                                                         sharing=tuple(range(d)))
                self.assertEqual(unshared, singletons)

    def test_single_pass_is_idempotent(self):
        # the single-pass theorem is sensitive to the phase ordering -- assert empirically that a
        # second pass changes nothing, over random proposals x random partitions
        rng = np.random.default_rng(0)
        for trial in range(200):
            d = int(rng.integers(2, 6))
            labels = tuple(int(v) for v in rng.integers(0, min(d, 3), size=d))
            label_size = {lab: int(rng.integers(2, 7)) for lab in set(labels)}
            shape = tuple(label_size[lab] for lab in labels)
            tucker = [0] * d
            for lab in set(labels):
                ng = int(rng.integers(1, 9))
                for ii in range(d):
                    if labels[ii] == lab:
                        tucker[ii] = ng
            tucker = tuple(tucker)
            tt = (1,) + tuple(int(v) for v in rng.integers(1, 9, size=d - 1)) + (1,)
            with self.subTest(shape=shape, tucker=tucker, tt=tt, labels=labels):
                n1, r1 = ranks.compute_minimal_ranks(shape, tucker, tt, sharing=labels)
                n2, r2 = ranks.compute_minimal_ranks(shape, n1, r1, sharing=labels)
                self.assertEqual((n1, r1), (n2, r2))

    def test_unequal_ranks_within_group_raise(self):
        # isolated rejection: equal mode sizes (validate_sharing passes), valid lengths -- ONLY the
        # within-group rank equality fails
        with self.assertRaises(ValueError):
            ranks.compute_minimal_ranks((6, 6, 4), (4, 3, 2), (1, 2, 2, 1), sharing=(0, 0, 1))

    def test_stacked_array_mode_matches_per_element(self):
        # array input with a stack axis: per-element results equal the sequence-mode calls
        shape = (6, 6, 4)
        tucker = np.array([[4, 3], [4, 3], [3, 2]])          # (d,) + stack (2,)
        tt = np.array([[1, 1], [2, 2], [2, 2], [1, 1]])      # (d+1,) + stack (2,)
        n_arr, r_arr = ranks.compute_minimal_ranks(shape, tucker, tt, sharing=(0, 0, 1))
        self.assertEqual(n_arr.shape, (3, 2))
        self.assertEqual(r_arr.shape, (4, 2))
        for kk in range(2):
            n_seq, r_seq = ranks.compute_minimal_ranks(
                shape, tuple(int(v) for v in tucker[:, kk]), tuple(int(v) for v in tt[:, kk]),
                sharing=(0, 0, 1))
            self.assertEqual(tuple(int(v) for v in n_arr[:, kk]), n_seq)
            self.assertEqual(tuple(int(v) for v in r_arr[:, kk]), r_seq)


class TestSharedContinuationRanks(unittest.TestCase):
    """Shared rank continuation: a sharing group's Tucker edges are ONE edge (one kappa_g in the pool,
    one growth decision group-wide, one max_grow candidate), and useless-rank removal is the shared
    one. Group modes must carry the IDENTICAL spectrum (the grouped t3svd assigns one s_g array per
    group)."""

    def _sv(self, kappa, k):
        # k descending singular values with condition number sigma_1/sigma_k = kappa (k>=1 -> 1.0)
        return np.linspace(1.0, 1.0 / kappa, k)

    def test_group_grows_group_wide(self):
        # the group (kappa_g=2) and mode 2 / bond 2 are well below the worst edge (bond 1, kappa=1000)
        # -> the group grows on BOTH its modes; the ill bond is frozen
        shape = (20, 20, 20)
        s_g = self._sv(2.0, 2)
        tucker_sv = [s_g, s_g, self._sv(3.0, 2)]
        tt_sv     = [self._sv(1., 1), self._sv(1000., 2), self._sv(2.5, 2), self._sv(1., 1)]
        got = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0, sharing=(0, 0, 1))
        self.assertEqual(got, ((3, 3, 3), (1, 2, 3, 1)))

    def test_group_is_kappa_max_and_freezes(self):
        # the group's kappa_g=1000 is the worst edge -> the group holds on BOTH modes while the others
        # grow. Bond 1's proposed growth is then clipped by the shared removal (r1 <= r0*n0 = 2, with
        # the frozen group holding n0 = 2).
        shape = (20, 20, 20)
        s_g = self._sv(1000., 2)
        tucker_sv = [s_g, s_g, self._sv(2.0, 2)]
        tt_sv     = [self._sv(1., 1), self._sv(4.0, 2), self._sv(2.5, 2), self._sv(1., 1)]
        got = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0, sharing=(0, 0, 1))
        self.assertEqual(got, ((2, 2, 3), (1, 2, 3, 1)))

    def test_kappa_guard_applies_to_kappa_g(self):
        # bond 1 is catastrophic (kappa_max=1e20 -> threshold 1e19); the group (kappa_g=1e15) passes
        # the relative rule but the absolute guard (1e12) freezes it; lifting the guard grows it
        shape = (20, 20, 20)
        s_g = self._sv(1e15, 2)
        tucker_sv = [s_g, s_g, self._sv(2.0, 2)]
        tt_sv     = [self._sv(1., 1), self._sv(1e20, 2), self._sv(2.5, 2), self._sv(1., 1)]
        held = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0, sharing=(0, 0, 1))
        self.assertEqual(held, ((2, 2, 3), (1, 2, 3, 1)))
        grown = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0,
                                                 kappa_guard=1e16, sharing=(0, 0, 1))
        self.assertEqual(grown, ((3, 3, 3), (1, 2, 3, 1)))

    def test_max_grow_counts_group_as_one_candidate(self):
        # the group is the single best-conditioned edge (kappa_g=1.5): max_grow=1 grows BOTH group
        # modes and nothing else -- the group is one candidate, not two
        shape = (20, 20, 20)
        s_g = self._sv(1.5, 2)
        tucker_sv = [s_g, s_g, self._sv(5.0, 2)]
        tt_sv     = [self._sv(1., 1), self._sv(4.0, 2), self._sv(6.0, 2), self._sv(1., 1)]
        got = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=1.0, max_grow=1,
                                               sharing=(0, 0, 1))
        self.assertEqual(got, ((3, 3, 2), (1, 2, 2, 1)))

    def test_max_grow_skips_structurally_capped_group(self):
        # the (non-adjacent) group {0,2} has the smallest kappa but sits at its mode-size cap
        # (n_g = 3 = N_g): the greedy must skip it -- trialing the group-wide increment and having the
        # SHARED removal clip it back -- and grow the next-best edge, the middle singleton mode
        shape = (3, 20, 3)
        s_g = self._sv(1.5, 3)
        tucker_sv = [s_g, self._sv(2.0, 4), s_g]
        tt_sv     = [self._sv(1., 1), self._sv(3.0, 3), self._sv(4.0, 3), self._sv(1., 1)]
        got = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=1.0, max_grow=1,
                                               sharing=(0, 1, 0))
        self.assertEqual(got, ((3, 5, 3), (1, 3, 3, 1)))

    def test_uniform_bump_fallback_bumps_group_once(self):
        # all edges comparably conditioned -> nothing passes the relative rule -> the fallback bumps
        # every edge by n_chunk; the group modes bump TOGETHER (once each, staying tied)
        shape = (20, 20, 20)
        s_g = self._sv(1.1, 2)
        tucker_sv = [s_g, s_g, self._sv(1.2, 2)]
        tt_sv     = [self._sv(1., 1), self._sv(1.3, 2), self._sv(1.25, 2), self._sv(1., 1)]
        got = ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, tau=10.0, sharing=(0, 0, 1))
        self.assertEqual(got, ((3, 3, 3), (1, 3, 3, 1)))

    def test_unequal_group_spectra_raise(self):
        # isolated rejection: same lengths and a valid partition -- ONLY the within-group spectrum
        # identity fails (a shared group has ONE spectrum s_g)
        shape = (20, 20, 20)
        tucker_sv = [self._sv(2.0, 2), self._sv(2.5, 2), self._sv(3.0, 2)]
        tt_sv     = [self._sv(1., 1), self._sv(4.0, 2), self._sv(2.5, 2), self._sv(1., 1)]
        with self.assertRaises(ValueError):
            ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, sharing=(0, 0, 1))

    def test_all_singletons_match_unshared_exactly(self):
        shape = (20, 20, 20)
        tucker_sv = [self._sv(2.0, 2), self._sv(1000., 2), self._sv(3.0, 2)]
        tt_sv     = [self._sv(1., 1), self._sv(4.0, 2), self._sv(2.5, 2), self._sv(1., 1)]
        for kwargs in [dict(tau=10.0), dict(tau=1.5, n_chunk=2), dict(tau=1.0, max_grow=1)]:
            with self.subTest(**kwargs):
                self.assertEqual(
                    ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv, **kwargs),
                    ranks.compute_continuation_ranks(shape, tucker_sv, tt_sv,
                                                     sharing=(0, 1, 2), **kwargs))


class TestSharedManifoldDim(unittest.TestCase):
    """Shared manifold dimension: shared minimal reduction first, TT term unchanged, ONE Stiefel term
    per group. Hand-worked here; validated against dense tied-tangent ranks in tests/test_sharing.py."""

    def test_hand_worked_all_modes_one_group(self):
        # (5,5,5), n=(3,3,3), r=(1,3,3,1), already minimal (shared and unshared):
        # TT term = 1*3*3 + 3*3*3 + 3*3*1 - (3^2 + 3^2) = 45 - 18 = 27
        # Stiefel: unshared 3 * [3*(5-3)] = 18 -> 45; shared ONE term 3*(5-3) = 6 -> 33
        self.assertEqual(ranks.compute_manifold_dim((5, 5, 5), (3, 3, 3), (1, 3, 3, 1)), 45)
        self.assertEqual(ranks.compute_manifold_dim((5, 5, 5), (3, 3, 3), (1, 3, 3, 1),
                                                    sharing=(0, 0, 0)), 33)

    def test_group_ceiling_case_needs_the_shared_reduction(self):
        # n=(4,4,2) is shared-minimal but NOT unshared-minimal (mode 0 clips to 2): the unshared
        # formula applied to this shared structure miscounts (36), the shared one is exact (32 --
        # validated against the dense tied-tangent rank in test_sharing.py):
        # TT term = 1*4*2 + 2*4*2 + 2*2*1 - (2^2 + 2^2) = 28 - 8 = 20
        # Stiefel = 4*(6-4) [group] + 2*(4-2) [singleton] = 12 -> 32
        self.assertEqual(ranks.compute_manifold_dim((6, 6, 4), (4, 4, 2), (1, 2, 2, 1),
                                                    sharing=(0, 0, 1)), 32)
        self.assertEqual(ranks.compute_manifold_dim((6, 6, 4), (4, 4, 2), (1, 2, 2, 1)), 36)

    def test_sharing_none_and_singletons_agree(self):
        s = ((15, 16, 13), (9, 10, 8), (2, 7, 6, 3))
        self.assertEqual(ranks.compute_manifold_dim(*s), 578)                       # the known value
        self.assertEqual(ranks.compute_manifold_dim(*s, sharing=(0, 1, 2)), 578)


class TestSharedFrameHasMinimalRanks(unittest.TestCase):
    def test_shared_vs_unshared_verdicts(self):
        # (4,4,2)/(1,2,2,1) on (6,6,4): shared-minimal for the group {0,1} but NOT unshared-minimal
        args = ((6, 6, 4), (4, 4, 2), (4, 4, 2), (1, 2, 2, 1), (1, 2, 2, 1))
        self.assertFalse(ranks.frame_has_minimal_ranks(*args))
        self.assertTrue(ranks.frame_has_minimal_ranks(*args, sharing=(0, 0, 1)))

    def test_untied_ranks_are_not_shared_minimal(self):
        # unshared-minimal ranks whose group entries differ: fine unshared, False (not an error) shared
        args = ((6, 6, 4), (2, 4, 2), (2, 4, 2), (1, 2, 2, 1), (1, 2, 2, 1))
        self.assertTrue(ranks.frame_has_minimal_ranks(*args))
        self.assertFalse(ranks.frame_has_minimal_ranks(*args, sharing=(0, 0, 1)))


if __name__ == "__main__":
    unittest.main()
