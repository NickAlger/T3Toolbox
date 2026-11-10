# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Tests for ``backend.uniform_padded_frames`` and ``ufv_masking.ufv_frame_widths`` (the aligned-padded-frame plan's
T8 and T17; T9-T13 join with the step and the loop).

- **T8, the start.** ``ufv_padded_frame`` against the ragged ``fv_padded_frame`` through the scatter of
  ``_uniform_padded_frame_testing.py``: the old frame at four different own widths, gappy masks, finite garbage in
  every junk entry; the plan's cases (d = 2, 3, 5, uneven increments, non-minimal ranks, a bond growing alone,
  outside and inside completions in one core, the zero tensor) plus sharing, d = 1, the zero tensor padded with
  garbage, the zero tensor at d = 1 (its down store has real old rows but no old column), and an old frame that
  is itself in block layout (gappy masks in both blocks). The real parts read back equal the ragged start to 1e-12;
  every junk entry of the output is exactly 0; the output masks equal the independently derived ``[the old mask as
  given, padded | prefix(the counts)]``, the counts from the ragged stores; tied groups share their supercore slice
  bitwise; garbage changes nothing (bitwise, equal layouts); a stack with ranks varying per element equals its
  elements. ``ufv_empty_frame`` is the scatter of ``fv_empty_frame``. Structural errors, each isolated. jit vs numpy
  is the T16 row in ``tests/test_dispatch.py`` (x64 is on there).
- **T17.** ``ufv_frame_widths(N, n, r)`` equals the widths of ``ut3_orthogonal_representations`` over a sweep.
- The scatter helper round-trips in both of its modes.

numpy-only (house testing strategy).
"""
import os
import sys
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.padded_frames as padded_frames
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.sharing as sharing_module
import t3toolbox.backend.ufv_conversions as ufc
import t3toolbox.backend.ufv_masking as ufv_masking
import t3toolbox.backend.uniform_padded_frames as uniform_padded_frames

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _uniform_padded_frame_testing as testing


CASES = [  # label, seed, shape, old ranks (None: the zero tensor), target ranks, sharing
    ('d=5', 0, (5, 6, 5, 6, 5), ((2, 3, 3, 3, 2), (1, 2, 3, 3, 2, 1)), ((3, 5, 5, 5, 3), (1, 3, 5, 5, 3, 1)), None),
    ('d=2', 1, (7, 8), ((2, 2), (1, 2, 1)), ((4, 4), (1, 4, 1)), None),
    ('d=3', 2, (6, 7, 6), ((2, 3, 2), (1, 2, 2, 1)), ((4, 6, 4), (1, 4, 4, 1)), None),
    ('uneven increments', 3, (6, 6, 6, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 4, 5, 3), (1, 3, 5, 3, 1)), None),
    ('non-minimal ranks', 4, (6, 6, 6, 6), ((3, 3, 3, 3), (1, 2, 3, 2, 1)), ((5, 5, 5, 5), (1, 3, 5, 3, 1)), None),
    ('a bond grows alone', 5, (6, 6, 6, 6), ((3, 3, 3, 3), (1, 2, 3, 2, 1)), ((3, 3, 3, 3), (1, 2, 4, 2, 1)), None),
    ('outside and inside', 6, (6, 6, 6, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 3, 2, 2), (1, 2, 5, 2, 1)), None),
    ('the zero tensor', 7, (6, 7, 6, 7), None, ((3, 4, 4, 3), (1, 3, 4, 3, 1)), None),
    ('sharing', 8, (6, 6, 6, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((4, 4, 4, 4), (1, 4, 5, 4, 1)), (0, 0, 1, 1)),
    ('d=1', 9, (7,), ((2,), (1, 1)), ((5,), (1, 1)), None),
    ('the zero tensor, d=1', 10, (7,), None, ((3,), (1, 1)), None),   # real old rows (the boundary bonds), no old column
]


def old_case(seed, shape, old_ranks, sharing=None, garbage=True, padded_zero=False):
    """(the ragged old frame, the uniform old frame scattered at four different own widths with gappy masks and
    garbage). The zero tensor: the empty frames of both layers, or (``padded_zero``) the ragged empty frame scattered
    at widths (1, 2, 3, 2) with garbage."""
    rng = np.random.default_rng(100 + seed)
    np.random.seed(seed)
    if old_ranks is None:
        old = padded_frames.fv_empty_frame(shape)
        if padded_zero:
            return old, testing.scatter_frame(rng, old, (1, 2, 3, 2, max(shape) + 1), garbage=garbage)[0]
        return old, uniform_padded_frames.ufv_empty_frame(shape)
    x = t3.TuckerTensorTrain.randn(shape, *old_ranks)
    if sharing is not None:
        x = x.share(sharing, max_tucker_ranks=old_ranks[0], max_tt_ranks=old_ranks[1])
    old = fvc.t3_orthogonal_representations(x.data)[0]
    nU, nD, rL, rR = testing.stores_of(old)
    groups = () if sharing is None else sharing_module.validate_sharing(sharing, shape)
    widths = (max(nU) + 1, max(nD) + 2, max(rL) + 2, max(rR) + 1, max(shape))
    return old, testing.scatter_frame(rng, old, widths, groups=groups, garbage=garbage)[0]


def max_difference(Fa, Fb):
    """The largest entry difference between two ragged frames of equal shapes (shapes asserted)."""
    worst = 0.0
    for a, b in zip(Fa, Fb):
        for x, y in zip(a, b):
            assert x.shape == y.shape, (x.shape, y.shape)
            worst = max(worst, float(np.max(np.abs(x - y), initial=0.0)))
    return worst


class TestEmptyFrame(unittest.TestCase):

    def test_the_table(self):
        """U (d,)+C+(0, N), O / L / R (d,)+C+(1, 0, 1); up and down masks empty; left and right masks real at bonds
        0 and d only; N = max(shape) by default, or given; numpy, default float."""
        for shape, C, N in [((5, 6, 7), (), None), ((4,), (2,), 6), ((3, 4, 5, 6), (2, 3), None)]:
            with self.subTest(shape=shape, C=C, N=N):
                d, Nw = len(shape), N or max(shape)
                U, O, L, R, sh, (mU, mD, mL, mR) = uniform_padded_frames.ufv_empty_frame(shape, N, C)
                self.assertEqual((U.shape, O.shape, L.shape, R.shape, sh),
                                 ((d,) + C + (0, Nw), (d,) + C + (1, 0, 1), (d,) + C + (1, 0, 1), (d,) + C + (1, 0, 1),
                                  shape))
                self.assertEqual((mU.shape, mD.shape, mL.shape, mR.shape),
                                 ((d,) + C + (0,), (d,) + C + (0,), (d + 1,) + C + (1,), (d + 1,) + C + (1,)))
                bonds = np.zeros((d + 1,) + C + (1,), bool)
                bonds[0], bonds[d] = True, True
                self.assertTrue(np.array_equal(mL, bonds) and np.array_equal(mR, bonds))
                self.assertTrue(all(isinstance(a, np.ndarray) and a.dtype == float for a in (U, O, L, R)))

    def test_it_is_the_scatter_of_the_ragged_empty_frame(self):
        """Prefix masks, no garbage, at widths (0, 0, 1, 1): equal arrays and masks."""
        for shape in ((5, 6, 7), (4,)):
            with self.subTest(shape=shape):
                got = uniform_padded_frames.ufv_empty_frame(shape)
                want = testing.scatter_frame(np.random.default_rng(0), padded_frames.fv_empty_frame(shape),
                                             (0, 0, 1, 1, max(shape)), gappy=False, garbage=False)[0]
                for a, b in zip(got[:4], want[:4]):
                    self.assertEqual(a.shape, b.shape)
                    self.assertTrue(np.array_equal(a, b))
                self.assertEqual(got[4], want[4])
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(got[5], want[5])))


class TestStartAgainstRagged(unittest.TestCase):
    """T8: the uniform start reproduces the ragged one, with exact masks and junk."""

    def check(self, label, seed, shape, old_ranks, target, sharing, garbage=True, padded_zero=False):
        old, uold = old_case(seed, shape, old_ranks, sharing, garbage, padded_zero)
        return self.check_old(old, uold, shape, target, sharing)

    def check_old(self, old, uold, shape, target, sharing=None):
        F, split = uniform_padded_frames.ufv_padded_frame(uold, *target, sharing=sharing)
        want = padded_frames.fv_padded_frame(old, *target, sharing=sharing)
        self.assertLessEqual(max_difference(testing.ragged(F), want), 1e-12)
        self.assertTrue(testing.junk_is_zero(F))
        # the masks, derived independently: the old masks as given, padded to the split, then prefix(counts)
        Na, Rb = max(uold[0].shape[-2], uold[1].shape[-2]), max(uold[2].shape[-1], uold[3].shape[-1])
        self.assertEqual(split, (Na, Rb))
        counts = [np.array(t) - np.array(o) for t, o in
                  zip(ranks.compute_orthogonal_representation_ranks(shape, *target), testing.stores_of(old))]
        Ka, Kb = max(counts[0].max(), counts[1].max()), max(counts[2].max(), counts[3].max())
        for m_out, m_old, k, A, K in zip(F[5], uold[5], counts, (Na, Na, Rb, Rb), (Ka, Ka, Kb, Kb)):
            expected = np.zeros(m_old.shape[:-1] + (A + K,), bool)
            expected[..., :m_old.shape[-1]] = m_old
            expected[..., A:] = np.arange(K) < k[..., None]
            self.assertTrue(np.array_equal(m_out, expected))
            self.assertIsInstance(m_out, np.ndarray)
        if sharing is not None:                                          # tied: one supercore slice per group
            for group in sharing_module.validate_sharing(sharing, shape):
                self.assertTrue(all(np.array_equal(F[0][i], F[0][group[0]]) for i in group))
        return F

    def test_cases(self):
        for case in CASES:
            with self.subTest(case=case[0]):
                self.check(*case)

    def test_the_zero_tensor_padded_with_garbage(self):
        """The empty frame scattered at widths (1, 2, 3, 2): A-blocks of junk only, the split (2, 3)."""
        label, seed, shape, _, target, _ = CASES[7]
        F = self.check(label, seed, shape, None, target, None, padded_zero=True)
        self.assertEqual((F[0].shape[-2] - 2, F[2].shape[-1] - 3), (4, 4))   # Ka = max(kU, kD), Kb = max(kL, kR)

    def test_an_old_frame_in_block_layout(self):
        """The old frame is itself a padded frame in block layout (gappy masks in both of its blocks, garbage): the
        next increase from it."""
        shape, rng = (6, 6, 6, 6), np.random.default_rng(12)
        np.random.seed(12)
        first = fvc.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn(shape, (2, 2, 2, 2), (1, 2, 2, 2, 1)).data)[0]
        middle = padded_frames.fv_padded_frame(first, (3, 3, 3, 3), (1, 3, 3, 3, 1))
        s0, s1 = testing.stores_of(first), testing.stores_of(middle)
        widths = (max(s0[0] + s0[1]) + 1, max(np.subtract(s1[0], s0[0]).max(), np.subtract(s1[1], s0[1]).max()) + 1,
                  max(s0[2] + s0[3]) + 2, max(np.subtract(s1[2], s0[2]).max(), np.subtract(s1[3], s0[3]).max()) + 1,
                  7)
        umiddle = testing.scatter_frame(rng, middle, widths, old_stores=s0)[0]
        self.check_old(middle, umiddle, shape, ((4, 5, 5, 4), (1, 4, 5, 4, 1)))

    def test_garbage_changes_nothing(self):
        """The same old frame with zeros instead of garbage in its junk entries: the bitwise same output."""
        for label, seed, shape, old_ranks, target, sharing in CASES[:4]:
            with self.subTest(case=label):
                clean = uniform_padded_frames.ufv_padded_frame(old_case(seed, shape, old_ranks, garbage=False)[1],
                                                               *target)[0]
                dirty = uniform_padded_frames.ufv_padded_frame(old_case(seed, shape, old_ranks)[1], *target)[0]
                for a, b in zip(clean[:4], dirty[:4]):
                    self.assertTrue(np.array_equal(a, b))

    def test_a_stack_with_ranks_varying_per_element(self):
        """C = (2,): two old points of different ranks at one padded layout, per-element target ranks; each element
        reads back as its own ragged start."""
        shape, rng = (6, 6, 6, 6), np.random.default_rng(11)
        rank_pairs = [(((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 4, 5, 3), (1, 3, 5, 3, 1))),
                      (((3, 2, 2, 3), (1, 3, 2, 3, 1)), ((3, 3, 3, 3), (1, 3, 4, 3, 1)))]
        olds, uolds = [], []
        for e, (old_ranks, _) in enumerate(rank_pairs):
            np.random.seed(20 + e)
            old = fvc.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(shape, *old_ranks).data)[0]
            olds.append(old)
            uolds.append(testing.scatter_frame(rng, old, (5, 5, 5, 5, 6))[0])
        tucker = np.stack([np.array(t[0]) for _, t in rank_pairs], axis=1)
        tt = np.stack([np.array(t[1]) for _, t in rank_pairs], axis=1)
        F, _ = uniform_padded_frames.ufv_padded_frame(testing.stack_frames(uolds), tucker, tt)
        for e, (_, target) in enumerate(rank_pairs):
            F_e = testing.element(F, e)
            self.assertLessEqual(max_difference(testing.ragged(F_e), padded_frames.fv_padded_frame(olds[e], *target)),
                                 1e-12)
            self.assertTrue(testing.junk_is_zero(F_e))


class TestStartStructuralErrors(unittest.TestCase):
    """Each rejection isolated: the input is good in every respect but one, and the message names the check."""

    def setUp(self):
        self.shape = (6, 6, 6, 6)
        self.old, self.uold = old_case(3, self.shape, ((2, 2, 2, 2), (1, 2, 2, 2, 1)))
        self.target = ((3, 4, 5, 3), (1, 3, 5, 3, 1))

    def test_the_good_input_passes(self):
        uniform_padded_frames.ufv_padded_frame(self.uold, *self.target)

    def test_old_masks_not_an_orthogonal_representation(self):
        masks = list(self.uold[5])
        mD = masks[1].copy()
        mD[1, np.flatnonzero(~mD[1])[0]] = True                         # one extra real down slot at mode 1
        masks[1] = mD
        with self.assertRaisesRegex(ValueError, 'the down masks count'):
            uniform_padded_frames.ufv_padded_frame(self.uold[:5] + (tuple(masks),), *self.target)

    def test_target_below_the_old_frame(self):
        with self.assertRaisesRegex(ValueError, 'below the old frame'):
            uniform_padded_frames.ufv_padded_frame(self.uold, (2, 1, 2, 2), (1, 2, 2, 2, 1))

    def test_target_ranks(self):
        with self.assertRaisesRegex(ValueError, 'boundary TT ranks'):
            uniform_padded_frames.ufv_padded_frame(self.uold, self.target[0], (2, 3, 5, 3, 1))
        with self.assertRaisesRegex(ValueError, 'tucker_ranks has shape'):
            uniform_padded_frames.ufv_padded_frame(self.uold, (3, 4, 5), self.target[1])
        with self.assertRaises(TypeError):
            uniform_padded_frames.ufv_padded_frame(self.uold, (3.0, 4, 5, 3), self.target[1])

    def test_unequal_old_up_masks_in_a_group(self):
        """Sharing labels on an untied scatter (the members' up slots at different positions)."""
        with self.assertRaisesRegex(ValueError, 'unequal old up masks'):
            uniform_padded_frames.ufv_padded_frame(self.uold, (4, 4, 4, 4), (1, 4, 5, 4, 1), sharing=(0, 0, 1, 1))


class TestScatterHelper(unittest.TestCase):
    """The helper the uniform tests stand on: both modes read back as the ragged frame, the masks in their blocks."""

    def test_round_trips(self):
        shape, rng = (5, 6, 5), np.random.default_rng(3)
        np.random.seed(3)
        old = fvc.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)).data)[0]
        new = padded_frames.fv_padded_frame(old, (4, 4, 3), (1, 3, 4, 1))
        s0 = testing.stores_of(old)
        cases = (('own widths', old, (4, 5, 4, 3, 6), None), ('block layout', new, (3, 3, 3, 3, 6), s0))
        for label, F, widths, old_stores in cases:
            with self.subTest(mode=label):
                U, _ = testing.scatter_frame(rng, F, widths, old_stores=old_stores)
                self.assertEqual(max_difference(testing.ragged(U), F), 0.0)
                if old_stores is not None:                               # each slot's old count sits in its A-block
                    for m, olds, A in zip(U[5], s0, (3, 3, 3, 3)):
                        self.assertEqual(m[..., :A].sum(axis=-1).tolist(), list(olds))


class TestFrameWidths(unittest.TestCase):
    """T17: the widths rule against ``ut3_orthogonal_representations``."""

    def test_sweep(self):
        rng = np.random.default_rng(0)
        for trial in range(40):
            d = int(rng.integers(1, 5))
            shape = tuple(int(n) for n in rng.integers(1, 7, size=d))
            tucker = tuple(int(n) for n in rng.integers(1, 7, size=d))
            tt = (1,) + tuple(int(r) for r in rng.integers(1, 7, size=d - 1)) + (1,)
            x = t3.TuckerTensorTrain.randn(shape, tucker, tt)
            for extra_n, extra_r, extra_N in ((0, 0, 0), (2, 1, 1), (0, 3, 0)):         # n > N; r*r < n
                N, n, r = max(shape) + extra_N, max(tucker) + extra_n, max(tt) + extra_r
                with self.subTest(trial=trial, N=N, n=n, r=r):
                    frame = ufc.ut3_orthogonal_representations(
                        ut3.UniformTuckerTensorTrain.from_t3(x, N=N, n=n, r=r).data)[0]
                    widths = (frame[0].shape[-2], frame[1].shape[-2], frame[2].shape[-1], frame[3].shape[-1])
                    self.assertEqual(ufv_masking.ufv_frame_widths(N, n, r), widths)


if __name__ == '__main__':
    unittest.main()
