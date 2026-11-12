# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Tests for ``backend.uniform_padded_frames`` and ``ufv_masking.ufv_frame_widths`` (the aligned-padded-frame plan's
T8-T12 and T17; the jitted loop's T12 and T13 rows are in ``tests/test_dispatch.py``, where x64 is on).

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
- **T9, the step.** ``utv_align_padded_frame`` against the ragged ``tv_align_padded_frame``: the ragged start and its
  tangent are scattered in BLOCK LAYOUT (gappy masks in both blocks, junk slots in both, finite garbage in every junk
  entry, ``N`` padded), on T1's list (d = 1, 2, 3, 5, uneven, non-minimal, a tangent part 10x, truncating, the empty
  old frame, sharing groups) and the transported stores (the two colours and four other subsets). Frames by the
  largest principal-angle sine per directed edge and the movement, to 1e-9; the output masks exactly the
  independently derived ``[the input A-mask | prefix(the ragged output counts)]`` at the counts' maxima; the junk
  exactly 0; garbage changes nothing (bitwise, equal layouts). Structural errors, each isolated.
- **T10, stacks.** C = (3,) with different old, frame and output ranks per element, and C = (2, 2): each element of
  the stacked step equals its own call (frames by the same measure, movements, to 1e-13) and the ragged step; the
  start likewise, at C = (3,) and (2, 2).
- **T11, the compaction.** Masks exactly the prefix masks of the counts at the level widths; the real entries moved
  bitwise (the read-back equal); junk exactly 0; narrower and wider widths; a stack; ``utv_retract`` of the zero
  tangent at the compacted frame returns X_old; ``UniformManifoldGeometryOps.from_frame`` of the compacted start has
  ``from_point``'s variation masks at the zero-padded point; a count above its width raises.
- **T12, the loop** (eager; eager = jit is in the dispatch file). Against ``fv_aligned_padded_frame`` with the
  library projections of one full-rank hint on both layers, the old frame scattered with gappy masks and garbage:
  the defaults, tol 1e-10, the zero point, non-minimal and uneven, p = 0 at d = 2, simultaneous, sharing: the same
  pass counts, movements to 1e-12 (NaN after the last pass), spans to 1e-9, the level's prefix masks, junk 0. The
  no-op level (the old frame compacted, all-NaN movements), zero passes (the start compacted, no projection), the
  schedule recorded call by call and the pair stop rule on prescribed movements (as the ragged loop's), a stack
  C = (2,) with per-element ranks equal to its elements, structural errors.
- **The chain test (D-41).** start -> start -> start without a fit, both layers, at a coordinate-structured point and
  a generic one, with a step that grows a single bond: the convention at every level (1e-12, the uniform layer through
  the scatter) and every seed's pinned-QR pivot above 1e-3.
- **T17.** ``ufv_frame_widths(N, n, r)`` equals the widths of ``ut3_orthogonal_representations`` over a sweep.
- The scatter helper round-trips in both of its modes.

numpy-only (house testing strategy).
"""
import os
import sys
import unittest
import unittest.mock

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
import _padded_frame_oracles as oracles
import _uniform_padded_frame_testing as testing
import test_padded_frame as ragged_tests                 # T1's cases and setup (as a module: no re-collection)

import t3toolbox.backend.tv_operations as tvo
import t3toolbox.backend.utv_operations as utvo
import t3toolbox.backend.geometry as geometry
import t3toolbox.backend.ut3_conversions as ut3_conversions


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
    nD, nU, rL, rR = testing.stores_of(old)
    groups = () if sharing is None else sharing_module.validate_sharing(sharing, shape)
    widths = (max(nD) + 1, max(nU) + 2, max(rL) + 2, max(rR) + 1, max(shape))
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
                D, U, L, R, sh, (mD, mU, mL, mR) = uniform_padded_frames.ufv_empty_frame(shape, N, C)
                self.assertEqual((D.shape, U.shape, L.shape, R.shape, sh),
                                 ((d,) + C + (0, Nw), (d,) + C + (1, 0, 1), (d,) + C + (1, 0, 1), (d,) + C + (1, 0, 1),
                                  shape))
                self.assertEqual((mD.shape, mU.shape, mL.shape, mR.shape),
                                 ((d,) + C + (0,), (d,) + C + (0,), (d + 1,) + C + (1,), (d + 1,) + C + (1,)))
                bonds = np.zeros((d + 1,) + C + (1,), bool)
                bonds[0], bonds[d] = True, True
                self.assertTrue(np.array_equal(mL, bonds) and np.array_equal(mR, bonds))
                self.assertTrue(all(isinstance(a, np.ndarray) and a.dtype == float for a in (D, U, L, R)))

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
        self.assertEqual((F[0].shape[-2] - 2, F[2].shape[-1] - 3), (4, 4))   # Ka = max(kD, kU), Kb = max(kL, kR)

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


def coordinate_point(shape, n, r):
    """A point with coordinate-structured cores (identity-prefix Tucker factors, 0/1 TT cores; n <= r): its frame's
    columns are unit vectors -- the structure on which the earlier convention's seeds repeated a span."""
    d = len(shape)
    tucker = tuple(np.eye(n, N) for N in shape)
    bonds = (1,) + (r,) * (d - 1) + (1,)
    tt = []
    for i in range(d):
        G = np.zeros((bonds[i], n, bonds[i + 1]))
        for j in range(n):
            G[min(j, bonds[i] - 1), j, min(j, bonds[i + 1] - 1)] = 1.0
        tt.append(G)
    return t3.TuckerTensorTrain(tucker, tuple(tt))


class TestStartChains(unittest.TestCase):
    """D-41's chain test: start -> start -> start without a fit, at a coordinate-structured point and at a generic
    point, one step growing a single bond (no new rows in the other unfoldings). At every level both layers give the
    convention (the uniform one through the scatter, gappy masks and garbage) to 1e-12, and every seed is fresh: its
    pinned-QR pivot (its norm off the old block and the earlier seeds) is above 1e-3."""

    LEVELS = (((3, 3, 3, 3), (1, 3, 3, 3, 1)), ((3, 3, 3, 3), (1, 3, 4, 3, 1)), ((4, 4, 4, 4), (1, 4, 4, 4, 1)))

    def test_chains(self):
        shape = (5, 5, 5, 5)
        np.random.seed(3)
        points = {'coordinate-structured': coordinate_point(shape, 2, 2),
                  'generic': t3.TuckerTensorTrain.randn(shape, (2, 2, 2, 2), (1, 2, 2, 2, 1))}
        rng = np.random.default_rng(8)
        for name, x_old in points.items():
            with self.subTest(point=name):
                frame = fvc.t3_orthogonal_representations(x_old.data)[0]
                s0 = testing.stores_of(frame)
                uframe = testing.scatter_frame(rng, frame, (max(s0[0]) + 1, max(s0[1]) + 2, max(s0[2]) + 1,
                                                            max(s0[3]) + 2, 6))[0]
                for level, target in enumerate(self.LEVELS):
                    pivots = []
                    want = ragged_tests.convention_reference(frame, *target, pivots=pivots)
                    frame = padded_frames.fv_padded_frame(frame, *target)
                    uframe, _ = uniform_padded_frames.ufv_padded_frame(uframe, *target)
                    self.assertLessEqual(ragged_tests.worst_difference(frame, want), 1e-12, 'level %d' % level)
                    self.assertLessEqual(max_difference(testing.ragged(uframe), frame), 1e-12, 'level %d' % level)
                    self.assertTrue(testing.junk_is_zero(uframe))
                    self.assertGreater(min(pivots), 1e-3, 'level %d' % level)


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
        mU = masks[1].copy()
        mU[1, np.flatnonzero(~mU[1])[0]] = True                         # one extra real down slot at mode 1
        masks[1] = mU
        with self.assertRaisesRegex(ValueError, 'the up masks count'):
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

    def test_unequal_old_down_masks_in_a_group(self):
        """Sharing labels on an untied scatter (the members' up slots at different positions)."""
        with self.assertRaisesRegex(ValueError, 'unequal old down masks'):
            uniform_padded_frames.ufv_padded_frame(self.uold, (4, 4, 4, 4), (1, 4, 5, 4, 1), sharing=(0, 0, 1, 1))


def step_case(shape, old_ranks, work_ranks, sharing=None, zero_old=False, tangent=0.0, seed=0, garbage=True,
              stack_shape=()):
    """(the ragged start, its tangent, the old stores, the block-layout frame and variations, the split): T1's setup,
    scattered in block layout with one junk slot in the A-blocks, junk in the P-blocks, N padded by one."""
    old, old_ranks, _, start, S = ragged_tests.step_case(shape, old_ranks, work_ranks, sharing=sharing,
                                                         zero_old=zero_old, tangent=tangent, stack_shape=stack_shape)
    variations = tvo.tv_project_dense_onto_tangent_space(start, S)
    s0 = ranks.compute_orthogonal_representation_ranks(shape, *old_ranks)
    k = [max(np.subtract(b, a)) for a, b in zip(s0, testing.stores_of(start))]
    split = (max(s0[0] + s0[1]) + 1, max(s0[2] + s0[3]) + 1)
    widths = (split[0], max(k[0], k[1]) + 1, split[1], max(k[2], k[3]) + 2, max(shape) + 1)
    groups = () if sharing is None else sharing_module.validate_sharing(sharing, shape)
    rng = np.random.default_rng(seed)
    frame, positions = testing.scatter_frame(rng, start, widths, old_stores=s0, groups=groups, garbage=garbage)
    return start, variations, old_ranks, s0, frame, testing.scatter_variations(rng, variations, frame, positions,
                                                                               garbage=garbage), split


def frame_distance(Fa, Fb):
    """The largest principal-angle sine per directed edge between two ragged frames (the plan's measure)."""
    return max(oracles.distances(oracles.dense_frames(Fa), oracles.dense_frames(Fb)).values())


def expected_masks(frame, split, s0, s1):
    """``[the input A-mask | prefix(s1 - s0)]`` at the counts' maxima, derived from the ragged stores."""
    k1 = [np.subtract(b, a) for a, b in zip(s0, s1)]
    Ka1, Kb1 = max(k1[0].max(), k1[1].max()), max(k1[2].max(), k1[3].max())
    out = []
    for m, k, A, K in zip(frame[5], k1, (split[0], split[0], split[1], split[1]), (Ka1, Ka1, Kb1, Kb1)):
        out.append(np.concatenate([m[..., :A], np.arange(K) < np.asarray(k)[..., None]], axis=-1))
    return out


class TestStepAgainstRagged(unittest.TestCase):
    """T9: the uniform step reproduces the ragged one through the block-layout scatter."""

    def check(self, shape, old_ranks, work_ranks, out_ranks, options, update=('D', 'U', 'L', 'R')):
        sharing = options.get('sharing')
        start, variations, old_ranks, s0, frame, uvariations, split = step_case(
            shape, old_ranks, work_ranks, sharing=sharing, zero_old=options.get('zero_old', False),
            tangent=options.get('tangent', 0.0))
        F, movement = padded_frames.tv_align_padded_frame(start, variations, *old_ranks, *(out_ranks or (None, None)),
                                                          sharing=sharing, update=update)
        F1, movement1 = uniform_padded_frames.utv_align_padded_frame(frame, uvariations, split,
                                                                    *(out_ranks or (None, None)), sharing=sharing,
                                                                    update=update)
        self.assertLessEqual(frame_distance(testing.ragged(F1), F), 1e-9)
        self.assertEqual(np.shape(movement1), ())
        self.assertLessEqual(abs(float(movement1) - float(movement)), 1e-9)
        self.assertTrue(testing.junk_is_zero(F1))
        for got, want in zip(F1[5], expected_masks(frame, split, s0, testing.stores_of(F))):
            self.assertIsInstance(got, np.ndarray)
            self.assertTrue(np.array_equal(got, want))
        if sharing is not None:
            for group in sharing_module.validate_sharing(sharing, shape):
                self.assertTrue(all(np.array_equal(F1[0][i], F1[0][group[0]]) for i in group))
        return F1

    def test_cases(self):
        for name, args in ragged_tests.TestStepAgainstDense.CASES.items():
            with self.subTest(case=name):
                self.check(*args)

    def test_transported_stores(self):
        """Each store family transported, and the two colours (('D', 'L'), ('U', 'R'))."""
        args = ((5, 6, 5, 6), ((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((3, 5, 5, 3), (1, 3, 5, 3, 1)), None, {})
        for update in (('D', 'L'), ('U', 'R'), ('U', 'L', 'R'), ('D', 'U', 'L'), ('D',), ('U',)):
            with self.subTest(update=update):
                self.check(*args, update=update)

    def test_carriers_keep_the_old_block(self):
        """left[d-1] and right[0] come out as their masked A-blocks, zero-padded: content in their completion slots
        (real slots, so not masked) is dropped, as on the ragged layer."""
        shape, old_ranks, work_ranks, _, _ = ragged_tests.TestStepAgainstDense.CASES['d=3']
        start, variations, old_ranks, s0, frame, uvariations, (Na, Rb) = step_case(shape, old_ranks, work_ranks)
        rng = np.random.default_rng(9)
        D, U, L, R, sh, (mD, mU, mL, mR) = frame
        L, R = L.copy(), R.copy()
        L[-1] += rng.standard_normal(L[-1].shape) * (np.arange(L.shape[-2]) >= Na)[None, :, None]   # completion slots
        R[0] += rng.standard_normal(R[0].shape) * (np.arange(R.shape[-2]) >= Na)[None, :, None]
        F1, _ = uniform_padded_frames.utv_align_padded_frame((D, U, L, R, sh, (mD, mU, mL, mR)), uvariations, (Na, Rb))
        mask = lambda core, a, n, b: core * a[:, None, None] * n[None, :, None] * b[None, None, :]
        want_L = mask(L[-1], mL[-2], mD[-1], mL[-1])[:Rb, :Na, :Rb]
        want_R = mask(R[0], mR[0], mD[0], mR[1])[:Rb, :Na, :Rb]
        for got, want in ((F1[2][-1], want_L), (F1[3][0], want_R)):
            self.assertTrue(np.array_equal(got[:Rb, :Na, :Rb], want))
            got = got.copy()
            got[:Rb, :Na, :Rb] = 0.0
            self.assertTrue(np.all(got == 0.0))

    def test_garbage_changes_nothing(self):
        """Zeros instead of garbage in every junk entry of the frame and the variations: the bitwise same output."""
        for name in ('d=3', 'truncating', 'groups', 'empty'):
            shape, old_ranks, work_ranks, out_ranks, options = ragged_tests.TestStepAgainstDense.CASES[name]
            with self.subTest(case=name):
                runs = []
                for garbage in (False, True):
                    _, _, _, _, frame, uvariations, split = step_case(
                        shape, old_ranks, work_ranks, sharing=options.get('sharing'),
                        zero_old=options.get('zero_old', False), garbage=garbage)
                    runs.append(uniform_padded_frames.utv_align_padded_frame(
                        frame, uvariations, split, *(out_ranks or (None, None)), sharing=options.get('sharing')))
                (Fc, mc), (Fd, md) = runs
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(Fc[:4], Fd[:4])))
                self.assertTrue(np.array_equal(mc, md))


class TestStacks(unittest.TestCase):
    """T10: a stack is its elements, for the step and for the start."""

    ELEMENTS = [   # shape (6, 6, 6, 6): old ranks, frame (working) ranks, output ranks per element
        (((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 4, 3, 1))),
        (((3, 2, 2, 3), (1, 3, 2, 3, 1)), ((4, 4, 3, 4), (1, 4, 4, 4, 1)), ((4, 3, 3, 4), (1, 4, 3, 4, 1))),
        (((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((3, 4, 4, 3), (1, 3, 5, 3, 1)), ((3, 4, 4, 3), (1, 3, 5, 3, 1))),
    ]

    def stacked_case(self, elements, C):
        """The elements scattered at one block layout (split (4, 4), completion widths (3, 4), N = 7), stacked to C."""
        shape, rng = (6, 6, 6, 6), np.random.default_rng(5)
        ragged, frames, variations = [], [], []
        for e, (old_ranks, work_ranks, out_ranks) in enumerate(elements):
            old, _, _, start, S = ragged_tests.step_case(shape, old_ranks, work_ranks, seed=10 + e)
            tangent = tvo.tv_project_dense_onto_tangent_space(start, S)
            s0 = ranks.compute_orthogonal_representation_ranks(shape, *old_ranks)
            frame, positions = testing.scatter_frame(rng, start, (4, 3, 4, 4, 7), old_stores=s0)
            ragged.append((start, tangent, old_ranks, out_ranks))
            frames.append(frame)
            variations.append(testing.scatter_variations(rng, tangent, frame, positions))
        stack = lambda arrays: np.stack(arrays, axis=1).reshape(arrays[0].shape[:1] + C + arrays[0].shape[1:])
        frame = tuple(stack([f[k] for f in frames]) for k in range(4)) + (shape, tuple(
            stack([f[5][k] for f in frames]) for k in range(4)))
        uvariations = (stack([v[0] for v in variations]), stack([v[1] for v in variations]), shape, ())
        out = tuple(stack([np.array(r[3][j]) for r in ragged]) for j in range(2))
        return ragged, frames, variations, frame, uvariations, out

    def test_the_step(self):
        for C, elements in (((3,), self.ELEMENTS), ((2, 2), self.ELEMENTS[:2] * 2)):
            with self.subTest(C=C):
                ragged, frames, variations, frame, uvariations, out = self.stacked_case(elements, C)
                F1, movement = uniform_padded_frames.utv_align_padded_frame(frame, uvariations, (4, 4), *out)
                self.assertEqual(movement.shape, C)
                self.assertTrue(testing.junk_is_zero(F1))
                for e, idx in enumerate(np.ndindex(*C)):
                    start, tangent, old_ranks, out_ranks = ragged[e]
                    F_e, mv_e = uniform_padded_frames.utv_align_padded_frame(frames[e], variations[e], (4, 4),
                                                                            *out_ranks)
                    got = testing.ragged(testing.element(F1, idx))
                    self.assertLessEqual(frame_distance(got, testing.ragged(F_e)), 1e-13)   # elements of different ranks run at different padded widths; CI's BLAS gave 1.2e-14
                    self.assertLessEqual(abs(float(movement[idx]) - float(mv_e)), 1e-13)
                    F_r, mv_r = padded_frames.tv_align_padded_frame(start, tangent, *old_ranks, *out_ranks)
                    self.assertLessEqual(frame_distance(got, F_r), 1e-9)
                    self.assertLessEqual(abs(float(movement[idx]) - float(mv_r)), 1e-9)

    def test_the_start(self):
        """ufv_padded_frame on the stacked old frames with per-element targets: each element its own start."""
        shape, rng = (6, 6, 6, 6), np.random.default_rng(6)
        for C, elements in (((3,), self.ELEMENTS), ((2, 2), self.ELEMENTS[:2] * 2)):
            with self.subTest(C=C):
                olds, uolds = [], []
                for e, (old_ranks, _, _) in enumerate(elements):
                    np.random.seed(30 + e)
                    old = fvc.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(shape, *old_ranks).data)[0]
                    olds.append(old)
                    uolds.append(testing.scatter_frame(rng, old, (5, 5, 5, 5, 6))[0])
                stack = lambda arrays: np.stack(arrays, axis=1).reshape(arrays[0].shape[:1] + C + arrays[0].shape[1:])
                uold = tuple(stack([f[k] for f in uolds]) for k in range(4)) + (shape, tuple(
                    stack([f[5][k] for f in uolds]) for k in range(4)))
                target = tuple(stack([np.array(t[1][j]) for t in elements]) for j in range(2))
                F, _ = uniform_padded_frames.ufv_padded_frame(uold, *target)
                for e, idx in enumerate(np.ndindex(*C)):
                    F_e = testing.element(F, idx)
                    want = padded_frames.fv_padded_frame(olds[e], *elements[e][1])
                    self.assertLessEqual(max_difference(testing.ragged(F_e), want), 1e-14)
                    self.assertTrue(testing.junk_is_zero(F_e))


class TestStepStructuralErrors(unittest.TestCase):
    """Each rejection isolated: the input is good in every respect but one, and the message names the check."""

    def setUp(self):
        self.start, _, _, self.s0, self.frame, self.var, self.split = step_case(
            (5, 6, 5, 6), ((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((3, 5, 5, 3), (1, 3, 5, 3, 1)))
        self.out = ((3, 4, 4, 3), (1, 3, 4, 3, 1))

    def call(self, frame=None, var=None, split=None, out=None, **kw):
        return uniform_padded_frames.utv_align_padded_frame(frame or self.frame, var or self.var, split or self.split,
                                                            *(out or self.out), **kw)

    def test_the_good_input_passes(self):
        self.call()
        self.call(update=('D', 'L'), out=(None, None))

    def test_update(self):
        """The ragged step's validation (``padded_frames._update_stores``): bad letters name the letters; every str
        and a collection holding the 2026.4.0 letter 'O' add the 2026.4.0 sentence."""
        for update in ((), ('D', 'D'), ('D', 'L', 'X')):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'update must be'):
                self.call(update=update)
        for update in ('DL', 'ULRO', ('U', 'O')):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'update must be .*2026.4.0 form'):
                self.call(update=update)

    def test_update_reaches_the_kernel_canonical(self):
        """Any order or collection of the letters reaches the kernel, whose jit static it is, in ``_STORE_LETTERS``
        order, one static per set; the frames agree bitwise."""
        kernel, seen = uniform_padded_frames._utv_align_kernel, []

        def capture(dynamic, skeleton, split, update):
            seen.append(update)
            return kernel(dynamic, skeleton, split, update)
        with unittest.mock.patch.object(uniform_padded_frames, '_utv_align_kernel', side_effect=capture):
            runs = [self.call(update=u, out=(None, None)) for u in (('D', 'L'), ('L', 'D'), ['L', 'D'], {'D', 'L'})]
        self.assertEqual(seen, [('D', 'L')] * 4)
        (F0, movement0), others = runs[0], runs[1:]
        for F, movement in others:
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(F[:4] + (movement,), F0[:4] + (movement0,))))

    def test_shapes_and_split(self):
        with self.assertRaisesRegex(ValueError, 'tt variations has shape'):
            self.call(var=(self.var[0], self.var[1][..., :-1]) + self.var[2:])
        with self.assertRaisesRegex(ValueError, 'split'):
            self.call(split=(self.split[0], 0))

    def test_masks_not_an_orthogonal_representation(self):
        masks = list(self.frame[5])
        mU = masks[1].copy()
        mU[1, np.flatnonzero(~mU[1])[0]] = True                         # one extra real down slot at mode 1
        masks[1] = mU
        with self.assertRaisesRegex(ValueError, 'do not count an orthogonal representation'):
            self.call(frame=self.frame[:5] + (tuple(masks),))

    def test_output_ranks(self):
        with self.assertRaisesRegex(ValueError, 'give both'):
            uniform_padded_frames.utv_align_padded_frame(self.frame, self.var, self.split, self.out[0])
        with self.assertRaisesRegex(ValueError, 'output boundary TT ranks'):
            self.call(out=(self.out[0], (2, 3, 4, 3, 1)))
        with self.assertRaisesRegex(ValueError, 'the old block must fit the output and the output the frame'):
            self.call(out=((3, 6, 5, 3), (1, 3, 5, 3, 1)))              # above the frame
        with self.assertRaisesRegex(ValueError, 'the old block must fit'):
            self.call(out=((2, 2, 3, 2), (1, 2, 3, 2, 1)))              # below the old block

    def test_truncation_without_update(self):
        with self.assertRaisesRegex(ValueError, 'truncated .* but not updated'):
            self.call(update=('U', 'R'))

    def test_unequal_group(self):
        """Valid sharing labels (modes 0 and 2, both of size 5) on an untied scatter."""
        with self.assertRaisesRegex(ValueError, 'unequal Tucker stores or old down masks'):
            self.call(sharing=(0, 1, 0, 2), out=(None, None))


def full_rank_hint(shape, seed):
    """A generic T3 at the shape's minimal full ranks: the tensor both layers' loops align with."""
    tucker, tt = ranks.compute_minimal_ranks(shape, tuple(shape), (1,) + (10 ** 6,) * (len(shape) - 1) + (1,))
    np.random.seed(seed)
    return t3.TuckerTensorTrain.randn(shape, tucker, tt)


def level_of(x_old, shape, target):
    """The level: the zero-padded point at the target ranks (uniform), and the widths of its frame."""
    point = t3.TuckerTensorTrain.zeros(shape, *target) if x_old is None else x_old.resize(shape, *target)
    upoint = ut3.UniformTuckerTensorTrain.from_t3(point)
    frame = ufc.ut3_orthogonal_representations(upoint.data)[0]
    return upoint, (frame[0].shape[-2], frame[1].shape[-2], frame[2].shape[-1], frame[3].shape[-1])


def loop_case(shape, old_ranks, target, sharing=None, seed=0, garbage=True):
    """(x_old, the ragged old frame, the uniform old frame scattered at four own widths, the ragged hint, its uniform
    twin at the frame's N, the level widths)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    if old_ranks is None:
        x_old, old, uold = None, padded_frames.fv_empty_frame(shape), uniform_padded_frames.ufv_empty_frame(shape)
    else:
        x_old = t3.TuckerTensorTrain.randn(shape, *old_ranks)
        if sharing is not None:
            x_old = x_old.share(sharing, max_tucker_ranks=old_ranks[0], max_tt_ranks=old_ranks[1])
        old = fvc.t3_orthogonal_representations(x_old.data)[0]
        s0 = testing.stores_of(old)
        groups = () if sharing is None else sharing_module.validate_sharing(sharing, shape)
        uold = testing.scatter_frame(rng, old, (max(s0[0]) + 1, max(s0[1]) + 2, max(s0[2]) + 2, max(s0[3]) + 1,
                                                max(shape)), groups=groups, garbage=garbage)[0]
    hint = full_rank_hint(shape, seed + 1)
    if sharing is not None:
        hint = hint.share(sharing, max_tucker_ranks=hint.tucker_ranks, max_tt_ranks=hint.tt_ranks)
    uhint = ut3.UniformTuckerTensorTrain.from_t3(hint, N=uold[0].shape[-1])
    return x_old, old, uold, hint, uhint, level_of(x_old, shape, target)[1]


def passes_of(movements):
    """The number of passes a NaN-padded movement record holds."""
    m = np.asarray(movements).reshape(len(movements), -1)
    return int(np.sum(~np.isnan(m[:, 0])))


class TestCompaction(unittest.TestCase):
    """T11."""

    def padded(self, C=(), seed=0):
        """A block-layout start (gappy in both blocks, garbage) of a d = 4 point, and its level."""
        shape, old_ranks, target = (5, 6, 5, 6), ((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((3, 5, 5, 3), (1, 3, 5, 3, 1))
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        x_old = t3.TuckerTensorTrain.randn(shape, *old_ranks)
        old = fvc.t3_orthogonal_representations(x_old.data)[0]
        F = padded_frames.fv_padded_frame(old, *target)
        frame = testing.scatter_frame(rng, F, (4, 3, 4, 4, 7), old_stores=testing.stores_of(old))[0]
        return x_old, F, frame, shape, target

    def test_masks_entries_and_junk(self):
        _, F, frame, shape, target = self.padded()
        stores = ranks.compute_orthogonal_representation_ranks(shape, *target)
        for widths in ((5, 5, 5, 5), (7, 6, 9, 8), (6, 5, 5, 6)):                # level, wider, mixed
            with self.subTest(widths=widths):
                Fc = uniform_padded_frames.ufv_compact_frame(frame, widths)
                self.assertEqual((Fc[0].shape[-2], Fc[1].shape[-2], Fc[2].shape[-1], Fc[3].shape[-1]), widths)
                for m, count, W in zip(Fc[5], stores, widths):
                    self.assertTrue(np.array_equal(m, np.arange(W) < np.asarray(count)[:, None]))
                for a, b in zip(testing.ragged(Fc), testing.ragged(frame)):     # the real entries, moved bitwise
                    self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
                self.assertTrue(testing.junk_is_zero(Fc))

    def test_a_stack(self):
        frames = [self.padded(seed=s)[2] for s in (1, 2)]
        stacked = testing.stack_frames(frames)
        Fc = uniform_padded_frames.ufv_compact_frame(stacked, (5, 5, 5, 5))
        for e in range(2):
            self.assertTrue(all(np.array_equal(x, y) for a, b in zip(
                testing.ragged(testing.element(Fc, e)), testing.ragged(uniform_padded_frames.ufv_compact_frame(
                    frames[e], (5, 5, 5, 5)))) for x, y in zip(a, b)))

    def test_the_level_geometry(self):
        """At the level: the zero tangent retracts to X_old, and the compacted frame's variation masks are the
        zero-padded point's (from_frame vs from_point)."""
        x_old, _, frame, shape, target = self.padded()
        upoint, widths = level_of(x_old, shape, target)
        Fc = uniform_padded_frames.ufv_compact_frame(frame, widths)
        d = len(shape)
        zero = (np.zeros((d, widths[1], Fc[0].shape[-1])), np.zeros((d, widths[2], widths[0], widths[3])), Fc[4],
                ufv_masking.ufv_variation_masks(Fc[5]))
        X = ut3_conversions.ut3_to_dense(utvo.utv_retract(Fc, zero))
        X_old = x_old.to_dense()
        self.assertLessEqual(np.linalg.norm(X - X_old), 1e-12 * np.linalg.norm(X_old))
        from_frame = geometry.UniformManifoldGeometryOps.from_frame(Fc, upoint.masks.data)
        from_point = geometry.UniformManifoldGeometryOps.from_point(upoint.data)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(from_frame.var_masks, from_point.var_masks)))

    def test_a_count_above_its_width(self):
        _, _, frame, _, _ = self.padded()
        with self.assertRaisesRegex(ValueError, 'exceeds its level width'):
            uniform_padded_frames.ufv_compact_frame(frame, (5, 5, 4, 5))


class TestLoopAgainstRagged(unittest.TestCase):
    """T12 (eager): the uniform loop runs the ragged loop's passes."""

    CASES = {   # shape, old ranks (None: the zero tensor), target ranks, sharing, loop options
        'defaults':       ((5, 6, 5, 6), ((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((3, 5, 5, 3), (1, 3, 5, 3, 1)), None, {}),
        'tol 1e-10':      ((5, 6, 5, 6), ((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((3, 5, 5, 3), (1, 3, 5, 3, 1)), None,
                           {'tol': 1e-10, 'max_passes': 13}),
        'the zero point': ((4, 5, 4, 5), None, ((3, 4, 4, 3), (1, 3, 4, 3, 1)), None, {}),
        'non-minimal, uneven': ((6, 6, 6, 6), ((3, 3, 3, 3), (1, 2, 3, 2, 1)), ((3, 4, 5, 3), (1, 3, 5, 3, 1)), None,
                                {}),
        'p=0, d=2':       ((7, 8), ((2, 2), (1, 2, 1)), ((4, 4), (1, 4, 1)), None, {'oversampling': 0}),
        'simultaneous':   ((5, 6, 5), ((2, 3, 2), (1, 2, 2, 1)), ((3, 5, 3), (1, 3, 3, 1)), None,
                           {'alternate': False, 'max_passes': 8}),
        'sharing':        ((5, 5, 6), ((2, 2, 2), (1, 2, 2, 1)), ((3, 3, 4), (1, 3, 4, 1)), (0, 0, 1), {}),
    }

    def test_cases(self):
        for name, (shape, old_ranks, target, sharing, kw) in self.CASES.items():
            with self.subTest(case=name):
                _, old, uold, hint, uhint, widths = loop_case(shape, old_ranks, target, sharing)
                F, movements = padded_frames.fv_aligned_padded_frame(old, *target, hint.data,
                                                                     tvo.tv_project_t3_onto_tangent_space,
                                                                     sharing=sharing, **kw)
                Fu, mu = uniform_padded_frames.ufv_aligned_padded_frame(
                    uold, *target, uhint.data, utvo.utv_project_ut3_onto_tangent_space, widths, sharing=sharing, **kw)
                max_passes = kw.get('max_passes', padded_frames.ALIGN_MAX_PASSES)
                self.assertEqual(np.shape(mu), (max_passes,))
                self.assertEqual(passes_of(mu), len(movements))
                self.assertTrue(np.all(np.isnan(np.asarray(mu)[len(movements):])))
                self.assertLessEqual(max(abs(float(a) - float(b)) for a, b in zip(mu, movements)), 1e-12)
                self.assertLessEqual(frame_distance(testing.ragged(Fu), F), 1e-9)
                stores = ranks.compute_orthogonal_representation_ranks(shape, *target)
                for m, count, W in zip(Fu[5], stores, widths):
                    self.assertTrue(np.array_equal(m, np.arange(W) < np.asarray(count)[:, None]))
                self.assertTrue(testing.junk_is_zero(Fu))

    def test_nothing_to_complete(self):
        """Target stores = old stores: no pass, no projection; the old frame compacted; all-NaN movements."""
        _, _, uold, _, _, _ = loop_case((5, 6, 5, 6), ((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((2, 3, 3, 2), (1, 2, 3, 2, 1)))
        widths = (4, 4, 4, 4)
        with unittest.mock.patch.object(uniform_padded_frames, 'utv_align_padded_frame') as step:
            F, movements = uniform_padded_frames.ufv_aligned_padded_frame(
                uold, (2, 3, 3, 2), (1, 2, 3, 2, 1), None, ragged_tests.raising_project, widths)
        step.assert_not_called()
        self.assertEqual(movements.shape, (padded_frames.ALIGN_MAX_PASSES,))
        self.assertTrue(np.all(np.isnan(movements)))
        want = uniform_padded_frames.ufv_compact_frame(uold, widths)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(F[:4], want[:4])))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(F[5], want[5])))

    def test_zero_passes(self):
        """max_passes = 0 (any oversampling): the start at the target compacted, bitwise; (0,) movements; no
        projection."""
        shape, target = (5, 6, 5, 6), ((3, 5, 5, 3), (1, 3, 5, 3, 1))
        _, _, uold, _, _, widths = loop_case(shape, ((2, 3, 3, 2), (1, 2, 3, 2, 1)), target)
        want = uniform_padded_frames.ufv_compact_frame(
            uniform_padded_frames.ufv_padded_frame(uold, *target, seed=3)[0], widths)
        for p in (0, 5):
            with self.subTest(oversampling=p):
                F, movements = uniform_padded_frames.ufv_aligned_padded_frame(
                    uold, *target, None, ragged_tests.raising_project, widths, oversampling=p, max_passes=0, seed=3)
                self.assertEqual(movements.shape, (0,))
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(F[:4], want[:4])))

    def test_a_stack(self):
        """C = (2,) with per-element old and target ranks, tol 0 (every pass runs): each element is its own loop. The
        stack computes every element at the stack's padded widths (the maxima), its own call at its own, so six passes
        agree to roundoff, not bitwise (measured 1.0e-14): frames to 1e-13, movements to 1e-14."""
        shape = (5, 6, 5, 6)
        pairs = [(((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((3, 5, 5, 3), (1, 3, 5, 3, 1))),
                 (((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 4, 4, 3), (1, 3, 4, 3, 1)))]
        rng = np.random.default_rng(4)
        np.random.seed(4)
        olds = [testing.scatter_frame(rng, fvc.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn(shape, *o).data)[0], (4, 4, 4, 4, 6))[0] for o, _ in pairs]
        uhint = ut3.UniformTuckerTensorTrain.from_t3(full_rank_hint(shape, 8), N=6)
        widths = (5, 5, 5, 5)
        target = tuple(np.stack([np.array(t[j]) for _, t in pairs], axis=1) for j in range(2))
        stacked_hint = tuple(np.stack([a, a], axis=1) for a in uhint.data[:2]) + uhint.data[2:3] + (tuple(
            np.stack([m, m], axis=1) for m in uhint.data[3]),)
        project = utvo.utv_project_ut3_onto_tangent_space
        F, movements = uniform_padded_frames.ufv_aligned_padded_frame(testing.stack_frames(olds), *target,
                                                                      stacked_hint, project, widths, tol=0.0,
                                                                      max_passes=6)
        self.assertEqual(movements.shape, (6, 2))
        for e, (_, t) in enumerate(pairs):
            F_e, m_e = uniform_padded_frames.ufv_aligned_padded_frame(olds[e], *t, uhint.data, project, widths,
                                                                      tol=0.0, max_passes=6)
            self.assertLessEqual(frame_distance(testing.ragged(testing.element(F, e)), testing.ragged(F_e)), 1e-13)
            self.assertLessEqual(float(np.max(np.abs(movements[:, e] - m_e))), 1e-14)
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(testing.element(F, e)[5], F_e[5])))


class TestLoopSchedule(unittest.TestCase):
    """T12: the schedule recorded call by call (the real step, wrapped), and the pair stop rule on prescribed movements
    -- the ragged loop's (``test_padded_frame.TestLoopSchedule``), on the uniform loop."""

    def setUp(self):
        self.shape, self.target = (4, 5, 4, 3), ((3, 3, 3, 2), (1, 3, 3, 2, 1))
        _, _, self.old, _, self.hint, self.widths = loop_case(self.shape, ((2, 2, 2, 2), (1, 2, 2, 2, 1)), self.target)
        self.project = utvo.utv_project_ut3_onto_tangent_space

    def run_recorded(self, **kw):
        """[(update, output Tucker ranks or None)] per pass."""
        with unittest.mock.patch.object(uniform_padded_frames, 'utv_align_padded_frame',
                                        wraps=uniform_padded_frames.utv_align_padded_frame) as step:
            uniform_padded_frames.ufv_aligned_padded_frame(self.old, *self.target, self.hint.data, self.project,
                                                           self.widths, **kw)
        return [(c.kwargs['update'], None if c.args[3] is None else tuple(np.asarray(c.args[3]).tolist()))
                for c in step.call_args_list]

    def test_the_schedule(self):
        t = self.target[0]
        self.assertEqual(self.run_recorded(max_passes=6, tol=0.0),
                         [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t),
                          (('D', 'L'), None), (('U', 'R'), None), (('D', 'L'), None)])
        self.assertEqual(self.run_recorded(max_passes=4, tol=0.0, alternate=False),
                         [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t),
                          (('D', 'U', 'L', 'R'), None)])
        self.assertEqual(self.run_recorded(oversampling=0, max_passes=2, tol=0.0),
                         [(('D', 'U', 'L', 'R'), t), (('D', 'L'), None)])
        self.assertEqual(self.run_recorded(oversampled_passes=5, max_passes=3, tol=0.0),
                         [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t)])

    def stop_after(self, prescribed, **kw):
        """The passes the loop runs when the step returns the prescribed movements in turn (the real frames); each is
        recorded in its slot, NaN after the last pass."""
        moves = iter(prescribed)
        real = uniform_padded_frames.utv_align_padded_frame
        seen = []

        def fake(frame, variations, split, *args, **kws):
            F, _ = real(frame, variations, split, *args, **kws)
            seen.append(float(next(moves)))
            return F, np.asarray(seen[-1])
        with unittest.mock.patch.object(uniform_padded_frames, 'utv_align_padded_frame', side_effect=fake):
            _, movements = uniform_padded_frames.ufv_aligned_padded_frame(self.old, *self.target, self.hint.data,
                                                                          self.project, self.widths, **kw)
        np.testing.assert_array_equal(movements, seen + [np.nan] * (len(movements) - len(seen)))
        return len(seen)

    def test_the_stop_rule(self):
        lo, hi, nan = 0.01, 0.9, np.nan
        self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi]), 5)                  # the first pair
        self.assertEqual(self.stop_after([hi, hi, lo, lo, hi, lo, lo, hi]), 7)          # not the truncating pass
        self.assertEqual(self.stop_after([hi, hi, hi, hi, lo, lo, hi, hi, hi, hi]), 10)  # not across pairs
        self.assertEqual(self.stop_after([hi] * 10), 10)                                # 3 pairs and a final ('D', 'L')
        self.assertEqual(self.stop_after([hi] * 9, max_passes=9), 9)                    # an even cap
        self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final ('D', 'L')
        self.assertEqual(self.stop_after([hi, hi, hi, lo, nan, lo, lo, hi]), 7)         # a NaN never stops
        self.assertEqual(self.stop_after([hi, lo, lo, hi], alternate=False), 3)        # simultaneous
        self.assertEqual(self.stop_after([hi, hi, hi, nan, lo, hi], alternate=False), 5)

    def test_structural_errors(self):
        call = lambda **kw: uniform_padded_frames.ufv_aligned_padded_frame(
            self.old, *kw.pop('target', self.target), self.hint.data, self.project, kw.pop('widths', self.widths), **kw)
        for kw, message in ((dict(max_passes=-1), 'max_passes'), (dict(oversampling=True), 'oversampling'),
                            (dict(target=((2, 1, 2, 2), (1, 2, 2, 2, 1))), 'below the old frame'),
                            (dict(target=(self.target[0], (2, 3, 3, 2, 1))), 'boundary TT ranks'),
                            (dict(widths=(2, 3, 3, 3)), 'exceeds its level width')):
            with self.subTest(**{k: repr(v) for k, v in kw.items()}), self.assertRaisesRegex(ValueError, message):
                call(**kw)


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
                D, _ = testing.scatter_frame(rng, F, widths, old_stores=old_stores)
                self.assertEqual(max_difference(testing.ragged(D), F), 0.0)
                if old_stores is not None:                               # each slot's old count sits in its A-block
                    for m, olds, A in zip(D[5], s0, (3, 3, 3, 3)):
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
