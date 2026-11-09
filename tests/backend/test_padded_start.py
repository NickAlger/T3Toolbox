# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Tests for the padded start's working-rank rule and, later, its optimizer seam (the aligned-padded-frame plan's
T19 and T14).

So far T19: ``ranks.oversampled_ranks``, the one home of the working-rank rule ("``+p`` on every Tucker rank and
interior bond, structurally minimal, never below the input"), on integer sequences (Python-int tuples out) and on
per-stack-element ``(d,)+C`` / ``(d+1,)+C`` rank arrays (each element its own sequence call). T14 -- the geometry
members, ``Problem.local_model(frame=)``, ``padded_start``, the ``newton_cg`` keywords and ``g0_reference='cone'``
-- joins with the optimizer seam.

numpy-only: the rank function is host integer arithmetic on structure, never traced.
"""
import itertools
import unittest

import numpy as np

import t3toolbox.backend.ranks as ranks


def random_minimal_structure(rng, d, sharing=None):
    """A random structurally minimal (shape, tucker_ranks, tt_ranks), shared-minimal under ``sharing``."""
    shape = tuple(int(N) for N in rng.integers(1, 8, size=d))
    if sharing is not None:                                   # a group shares one mode size
        first = {}
        shape = tuple(first.setdefault(g, N) for g, N in zip(sharing, shape))
    tucker = [int(n) for n in rng.integers(1, 7, size=d)]
    if sharing is not None:
        first = {}
        tucker = [first.setdefault(g, n) for g, n in zip(sharing, tucker)]
    tt = (1,) + tuple(int(r) for r in rng.integers(1, 7, size=d - 1)) + (1,)
    return (shape,) + ranks.compute_minimal_ranks(shape, tuple(tucker), tt, sharing=sharing)


class TestOversampledRanks(unittest.TestCase):

    def test_p0_returns_a_minimal_input_unchanged(self):
        rng = np.random.default_rng(0)
        for trial in range(200):
            d = int(rng.integers(1, 6))
            shape, tucker, tt = random_minimal_structure(rng, d)
            with self.subTest(trial=trial, shape=shape, tucker=tucker, tt=tt):
                self.assertEqual(ranks.oversampled_ranks(shape, tucker, tt, 0), (tucker, tt))

    def test_minimal_input_gives_the_reduced_plus_p_ranks(self):
        """For a minimal input the maximum is a no-op (the reduction is monotone): the output is
        ``compute_minimal_ranks`` of the ``+p`` ranks, is itself minimal, and is never below the input."""
        rng = np.random.default_rng(1)
        for trial, p in itertools.product(range(100), (1, 2, 5)):
            d = int(rng.integers(1, 6))
            shape, tucker, tt = random_minimal_structure(rng, d)
            with self.subTest(trial=trial, p=p, shape=shape, tucker=tucker, tt=tt):
                out_tucker, out_tt = ranks.oversampled_ranks(shape, tucker, tt, p)
                plus = (tuple(n + p for n in tucker), (1,) + tuple(r + p for r in tt[1:-1]) + (1,))
                self.assertEqual((out_tucker, out_tt), ranks.compute_minimal_ranks(shape, *plus))
                self.assertEqual(ranks.compute_minimal_ranks(shape, out_tucker, out_tt), (out_tucker, out_tt))
                self.assertTrue(all(a >= b for a, b in zip(out_tucker + out_tt, tucker + tt)))
                self.assertTrue(all(type(r) is int for r in out_tucker + out_tt))

    def test_hand_worked_values(self):
        cases = [
            # (shape, tucker, tt, p) -> expected; the doctest's cases plus the saturated / d = 1 corners
            (((3, 10, 4), (2, 3, 2), (1, 2, 2, 1), 1), ((3, 4, 3), (1, 3, 3, 1))),
            (((3, 10, 4), (2, 3, 2), (1, 2, 2, 1), 5), ((3, 8, 4), (1, 3, 4, 1))),
            (((3, 3), (3, 3), (1, 3, 1), 4), ((3, 3), (1, 3, 1))),           # saturated: nothing to add
            (((5,), (1,), (1, 1), 3), ((1,), (1, 1))),                       # d = 1: the Tucker rank is capped
            (((5,), (2,), (1, 1), 3), ((2,), (1, 1))),                       #   by the bonds' product 1; a
        ]                                                                    #   non-minimal 2 is kept
        for (shape, tucker, tt, p), expected in cases:
            with self.subTest(shape=shape, tucker=tucker, tt=tt, p=p):
                self.assertEqual(ranks.oversampled_ranks(shape, tucker, tt, p), expected)

    def test_never_below_a_non_minimal_input(self):
        """A non-minimal input keeps its slack (elementwise, every store), so the working structure never
        undercuts the target -- including slack the +p reduction would remove."""
        rng = np.random.default_rng(2)
        for trial, p in itertools.product(range(100), (0, 1, 3)):
            d = int(rng.integers(1, 6))
            shape = tuple(int(N) for N in rng.integers(1, 6, size=d))
            tucker = tuple(int(n) for n in rng.integers(1, 9, size=d))
            tt = (1,) + tuple(int(r) for r in rng.integers(1, 9, size=d - 1)) + (1,)
            with self.subTest(trial=trial, p=p, shape=shape, tucker=tucker, tt=tt):
                out_tucker, out_tt = ranks.oversampled_ranks(shape, tucker, tt, p)
                self.assertTrue(all(a >= b for a, b in zip(out_tucker + out_tt, tucker + tt)))
                lo_tucker, lo_tt = ranks.compute_minimal_ranks(shape, tucker, tt)
                self.assertTrue(all(a >= b for a, b in zip(out_tucker + out_tt, lo_tucker + lo_tt)))
        self.assertEqual(ranks.oversampled_ranks((3, 10, 4), (5, 3, 2), (1, 2, 2, 1), 1),
                         ((5, 4, 3), (1, 3, 3, 1)))                          # Tucker 5 > N0 = 3 survives

    def test_sharing_a_group_keeps_one_rank(self):
        rng = np.random.default_rng(3)
        for trial, p in itertools.product(range(100), (0, 1, 4)):
            d = int(rng.integers(2, 6))
            sharing = tuple(int(g) for g in rng.integers(0, max(1, d - 1), size=d))
            shape, tucker, tt = random_minimal_structure(rng, d, sharing)
            with self.subTest(trial=trial, p=p, sharing=sharing, shape=shape, tucker=tucker, tt=tt):
                out_tucker, out_tt = ranks.oversampled_ranks(shape, tucker, tt, p, sharing=sharing)
                for g in set(sharing):
                    self.assertEqual(len({out_tucker[i] for i in range(d) if sharing[i] == g}), 1)
                self.assertEqual(ranks.compute_minimal_ranks(shape, out_tucker, out_tt, sharing=sharing),
                                 (out_tucker, out_tt))                       # shared-minimal
                self.assertTrue(all(a >= b for a, b in zip(out_tucker + out_tt, tucker + tt)))
                if p == 0:
                    self.assertEqual((out_tucker, out_tt), (tucker, tt))
        # the shared notion, not the unshared one: the tied rank may exceed a member's bond product
        self.assertEqual(ranks.oversampled_ranks((8, 3, 8), (2, 1, 2), (1, 1, 1, 1), 1, sharing=('a', 'b', 'a')),
                         ((3, 2, 3), (1, 2, 2, 1)))

    def test_structural_errors(self):
        args = ((3, 10, 4), (2, 3, 2), (1, 2, 2, 1))
        for p in (-1, 1.0, True, None):
            with self.subTest(p=p), self.assertRaisesRegex(ValueError, 'non-negative int'):
                ranks.oversampled_ranks(*args, p)
        with self.assertRaisesRegex(ValueError, 'len\\(tucker_ranks\\)'):
            ranks.oversampled_ranks((3, 10, 4), (2, 3), (1, 2, 2, 1), 1)
        with self.assertRaisesRegex(ValueError, 'len\\(tt_ranks\\)'):
            ranks.oversampled_ranks((3, 10, 4), (2, 3, 2), (1, 2, 1), 1)
        with self.assertRaises(TypeError):                                   # integer tuples in
            ranks.oversampled_ranks((3, 10, 4), (2.0, 3, 2), (1, 2, 2, 1), 1)
        with self.assertRaisesRegex(ValueError, 'equal within a sharing group'):
            ranks.oversampled_ranks((3, 10, 3), (2, 3, 1), (1, 2, 2, 1), 1, sharing=(0, 1, 0))

    def test_arrays_are_per_element_calls(self):
        """T19: (d,)+C / (d+1,)+C integer arrays give HOST integer arrays of the same shapes, each stack element
        equal to the sequence call on that element; with and without sharing."""
        rng = np.random.default_rng(4)
        for trial, C, p in itertools.product(range(20), [(), (3,), (2, 2)], (0, 1, 3)):
            d = int(rng.integers(1, 6))
            sharing = None if trial % 2 else tuple(int(g) for g in rng.integers(0, max(1, d - 1), size=d))
            shape = tuple(int(N) for N in rng.integers(1, 8, size=d))
            tucker = rng.integers(1, 9, size=(d,) + C)
            tt = rng.integers(1, 9, size=(d + 1,) + C)
            tt[0], tt[-1] = 1, 1
            if sharing is not None:                               # a group shares one mode size and one rank
                first = {}
                for i, g in enumerate(sharing):
                    j = first.setdefault(g, i)
                    tucker[i] = tucker[j]
                shape = tuple(shape[first[g]] for g in sharing)
            with self.subTest(trial=trial, C=C, p=p, sharing=sharing, shape=shape):
                out_tucker, out_tt = ranks.oversampled_ranks(shape, tucker, tt, p, sharing=sharing)
                for out, inp in ((out_tucker, tucker), (out_tt, tt)):
                    self.assertIsInstance(out, np.ndarray)
                    self.assertTrue(np.issubdtype(out.dtype, np.integer))
                    self.assertEqual(out.shape, inp.shape)
                for idx in np.ndindex(*C):
                    at = (slice(None),) + idx
                    want = ranks.oversampled_ranks(shape, tuple(int(n) for n in tucker[at]),
                                                   tuple(int(r) for r in tt[at]), p, sharing=sharing)
                    self.assertEqual((tuple(out_tucker[at].tolist()), tuple(out_tt[at].tolist())), want)

    def test_array_structural_errors(self):
        shape = (3, 10, 4)
        tucker = np.array([[2, 2], [3, 5], [2, 1]])
        tt = np.array([[1, 1], [2, 2], [2, 1], [1, 1]])
        ranks.oversampled_ranks(shape, tucker, tt, 1)                       # the good input
        with self.assertRaisesRegex(ValueError, 'share one stack'):
            ranks.oversampled_ranks(shape, tucker, tt[:, :1], 1)
        with self.assertRaisesRegex(ValueError, 'share one stack'):        # a sequence beside a stacked array
            ranks.oversampled_ranks(shape, (2, 3, 2), tt, 1)
        with self.assertRaises(TypeError):                                  # integer arrays in
            ranks.oversampled_ranks(shape, tucker.astype(float), tt, 1)
        with self.assertRaisesRegex(ValueError, 'len\\(tucker_ranks\\)'):
            ranks.oversampled_ranks(shape, tucker[:2], tt, 1)
        with self.assertRaisesRegex(ValueError, 'len\\(tt_ranks\\)'):
            ranks.oversampled_ranks(shape, tucker, tt[:3], 1)
        with self.assertRaisesRegex(ValueError, 'len\\(tt_ranks\\)'):       # d = 1: checked before the boundary
            ranks.oversampled_ranks((7,), np.array([[2, 3]]), np.array([[1, 1]]), 1)   # splice could lengthen it
        with self.assertRaisesRegex(ValueError, 'non-negative int'):
            ranks.oversampled_ranks(shape, tucker, tt, -1)


if __name__ == '__main__':
    unittest.main()
