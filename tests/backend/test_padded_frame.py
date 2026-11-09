# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Tests for ``backend.padded_frames``: the zero tensor's empty frame and the seeded padded frame (the start).

T5 of the aligned-padded-frame plan (``dev/aligned_padded_frame_impl_plan_2026-11-09.md``). The construction is the
one it replaced (the template-signature ``fv_operations.fv_padded_frame`` of commit 92640502, hint-free path from
2562264a), under the integer signature; that was checked once, bitwise on 30 cases, by
``dev/aligned_padded_frame_slices/N1_check_s2_equivalence.py``. What is pinned here, platform-independently:

- **The zero tensor.** At the empty frame the start is ``fv_zero_point_frame``'s frame, bitwise: the same QR of the
  same seeds in the same process.
- **The Tucker completion at a nonzero old factor** is the keyed stream ``RandomState([seed, min(group)])``'s first
  rows completed against the old rows, against an independent numpy QR (unshared and shared, two seeds).
- **The convention's properties.** A valid frame (orthonormal, both base-point reconstructions ``X_old``, the old
  block present bitwise with zeros beside it, the carriers zero on every new index); the outside seeds placed
  exactly in row-major order; in-block columns from the block's coordinate seeds in index order; the Tucker
  completion nested; a stack ``C`` equal to per-element calls; sign-gauge covariance; one Tucker array per sharing
  group; the seed moving the Tucker completions only.
- **The empty frame** is the orthogonal representation of the rank-0 Tucker tensor train.
- **Structural errors**, each isolated by its message.

numpy-only (house testing strategy); jax dispatch is in ``tests/test_dispatch.py``.
"""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.padded_frames as padded_frames
import t3toolbox.backend.sharing as sharing_module
import t3toolbox.backend.t3_conversions as t3c

STORES = ('up', 'down', 'left', 'right')


def frame_of(x):
    return fvc.t3_orthogonal_representations(x.data)[0]


def padded_case(shape, old_ranks, new_ranks, stack_shape=(), sharing=None, seed=0):
    """(x_old, its frame, the frame of its zero-padded resize): x_old random at old_ranks (shared first under
    ``sharing``); the third is the library's own frame at the target, an independent source of the shapes."""
    np.random.seed(seed)
    x_old = t3.TuckerTensorTrain.randn(shape, *old_ranks, stack_shape=stack_shape)
    if sharing is not None:
        x_old = x_old.share(sharing)
    x_new = x_old.resize(shape, *new_ranks, sharing=sharing)
    return x_old, frame_of(x_old), frame_of(x_new)


def take(frame, idx):
    """The stack element ``idx`` of a stacked frame."""
    return tuple(tuple(c[idx] for c in cores) for cores in frame)


def projectors(frame):
    """Gauge-invariant summary of a frame: per mode the Tucker projector U^T U, per bond the dense left and
    right interface projectors, per mode the dense down-environment projector (O_i between the interfaces)."""
    up, down, left, right = frame
    d = len(up)
    phi = [np.ones((1, 1))]                             # Phi_k: (prod N_<k, rL_k)
    for k in range(d - 1):
        phi.append(np.einsum('xa,ni,anb->xib', phi[k], up[k], left[k]).reshape(-1, left[k].shape[2]))
    psi = [None] * d + [np.ones((1, 1))]                # Psi_k: (prod N_>=k, rR_k)
    for k in range(d - 1, 0, -1):
        psi[k] = np.einsum('anb,ni,yb->aiy', right[k], up[k], psi[k + 1]).reshape(right[k].shape[0], -1).T
    out = [u.T @ u for u in up]
    out += [p @ p.T for p in phi[1:]] + [p @ p.T for p in psi[1:d]]
    for i in range(d):
        E = np.einsum('xa,amb,yb->xym', phi[i], down[i], psi[i + 1]).reshape(-1, down[i].shape[1])
        out.append(E @ E.T)
    return out


def inside_count(old, new, axis):
    """The convention's in-block column count of a TT / down core (its output axis ``axis``): the new columns
    take the outside coordinates of the (row-major) unfolding first, and only the rest come from a completion
    inside the old block -- the only part a factorization computes."""
    q1, q2, c = np.moveaxis(old, axis, -1).shape[-3:]
    p1, p2, p3 = np.moveaxis(new, axis, -1).shape[-3:]
    outside = p1 * p2 - (q1 * q2 if c else 0)
    return max(0, (p3 - c) - outside)


class PaddedFrameAsserts(unittest.TestCase):

    def assert_valid(self, padded, old, expected_shapes, X_old):
        """A valid padded frame of X_old (unstacked): the target's shapes, orthonormal, both reconstructions
        X_old, the old block bitwise with zeros beside it, the carriers zero on every new index."""
        for got, want in zip(padded, expected_shapes):
            self.assertEqual([g.shape for g in got], [w.shape for w in want])
        self.assertLessEqual(float(fvo.fv_frame_orthogonality_residual(padded)), 1e-14)
        up, down, left, right = padded
        nX = max(np.linalg.norm(X_old), 1.0)
        for tt in (left, right):
            self.assertLessEqual(np.linalg.norm(t3c.t3_to_dense((up, tt)) - X_old), 1e-13 * nX)
        U0, O0, L0, R0 = old
        for i in range(len(up)):                         # the old block, embedded as a copy (structural)
            self.assertTrue(np.array_equal(up[i][:U0[i].shape[0]], U0[i]))
            self.assertTrue(np.array_equal(down[i][tuple(slice(0, n) for n in O0[i].shape)], O0[i]))
            self.assertTrue(np.array_equal(left[i][tuple(slice(0, n) for n in L0[i].shape)], L0[i]))
            self.assertTrue(np.array_equal(right[i][tuple(slice(0, n) for n in R0[i].shape)], R0[i]))
            a, n, b = L0[i].shape                        # zero elsewhere in the old columns / rows (structural)
            self.assertTrue(np.all(left[i][:, :, :b][a:] == 0.0) and np.all(left[i][:a, n:, :b] == 0.0))
            a, n, b = R0[i].shape
            self.assertTrue(np.all(right[i][:a, :, b:] == 0.0) and np.all(right[i][:a, n:, :b] == 0.0))
            a, m, b = O0[i].shape
            self.assertTrue(np.all(down[i][a:, :m] == 0.0) and np.all(down[i][:a, :m, b:] == 0.0))
        d = len(up)                                      # the carriers: zero on every new index
        a, n, b = L0[d - 1].shape
        self.assertTrue(np.all(left[d - 1][a:] == 0.0) and np.all(left[d - 1][:, n:] == 0.0))
        a, n, b = R0[0].shape
        self.assertTrue(np.all(right[0][:, :, b:] == 0.0) and np.all(right[0][:, n:] == 0.0))


class TestZeroTensor(unittest.TestCase):
    """At the empty frame the start is ``fv_zero_point_frame``'s frame (``'cone'`` and ``'zero'`` share it)."""

    CASES = [
        ((7,), (3,), (1, 1), None),
        ((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), None),
        ((6, 6, 7), (3, 3, 2), (1, 3, 2, 1), (0, 0, 1)),
        ((4, 5, 6, 3), (2, 3, 3, 2), (1, 2, 3, 2, 1), None),
        ((5, 5, 5, 4), (3, 3, 3, 2), (1, 3, 3, 2, 1), (0, 1, 0, 2)),
    ]

    def test_equals_the_zero_point_frame(self):
        """Bitwise on all four stores: the TT cores are placed 0/1 values in both, and the Tucker factor is the
        same sign-fixed QR of the same seeds, in the same process."""
        for shape, tucker, tt, sharing in self.CASES:
            template = frame_of(t3.TuckerTensorTrain.zeros(shape, tucker, tt))
            groups = () if sharing is None else sharing_module.canonical_groups(sharing, shape)
            for seed in (0, 5):
                with self.subTest(shape=shape, tucker=tucker, tt=tt, sharing=sharing, seed=seed):
                    padded = padded_frames.fv_padded_frame(padded_frames.fv_empty_frame(shape), tucker, tt,
                                                           sharing=sharing, seed=seed)
                    zero = fvo.fv_zero_point_frame(template, groups=groups, seed=seed)
                    for got, want in zip(padded, zero):
                        for g, w in zip(got, want):
                            self.assertTrue(np.array_equal(g, w))
                    for group in groups:                   # one array per group
                        self.assertTrue(all(padded[0][i] is padded[0][group[0]] for i in group))

    def test_stacked_is_the_zero_point_frame_per_element(self):
        """A stack C: every element is the unstacked zero-point frame (fv_zero_point_frame itself refuses
        stacks), bitwise -- the batched QR factors each element's identical seeds as the unstacked one does."""
        shape, tucker, tt = (5, 6, 7), (2, 3, 2), (1, 2, 2, 1)
        padded = padded_frames.fv_padded_frame(padded_frames.fv_empty_frame(shape, (2, 3)), tucker, tt)
        zero = fvo.fv_zero_point_frame(frame_of(t3.TuckerTensorTrain.zeros(shape, tucker, tt)))
        for idx in np.ndindex(2, 3):
            for got, want in zip(take(padded, idx), zero):
                for g, w in zip(got, want):
                    self.assertTrue(np.array_equal(g, w))


class TestEmptyFrame(unittest.TestCase):

    def test_is_the_rank_zero_orthogonal_representation(self):
        """Its core shapes are those of the orthogonal representation of the rank-0 T3 (Tucker ranks 0, TT ranks
        (1, 0, ..., 0, 1)); every core is empty; numpy, default float."""
        for shape in [(7,), (4, 5), (5, 6, 7), (3, 4, 5, 6)]:
            d = len(shape)
            for C in [(), (2,), (2, 3)]:
                with self.subTest(shape=shape, C=C):
                    rank0 = t3.TuckerTensorTrain.zeros(shape, (0,) * d, (1,) + (0,) * (d - 1) + (1,), stack_shape=C)
                    want = frame_of(rank0)
                    empty = padded_frames.fv_empty_frame(shape, C)
                    self.assertEqual([[c.shape for c in cores] for cores in empty],
                                     [[c.shape for c in cores] for cores in want])
                    for core in (c for cores in empty for c in cores):
                        self.assertEqual((core.size, core.dtype), (0, np.dtype(float)))
                        self.assertIsInstance(core, np.ndarray)

    def test_no_increase_from_the_empty_frame(self):
        """The start at the zero tensor's own ranks has nothing to complete: the empty frame again."""
        empty = padded_frames.fv_empty_frame((5, 6, 7), (2,))
        padded = padded_frames.fv_padded_frame(empty, (0, 0, 0), (1, 0, 0, 1))
        self.assertEqual([[c.shape for c in cores] for cores in padded],
                         [[c.shape for c in cores] for cores in empty])


class TestValidAtResizedPoints(PaddedFrameAsserts):
    """Valid frames at Tucker-only, TT-only and joint increases, in-block cases, d = 1 and 4, non-minimal ranks."""

    CASES = {
        'tucker only':        ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 2), (1, 2, 2, 1))),
        'tt only':            ((5, 6, 7), ((3, 3, 3), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 3, 1))),
        'joint':              ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))),
        'joint d=4':          ((4, 5, 6, 3), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 3, 4, 3), (1, 3, 4, 3, 1))),
        'd=1':                ((7,), ((2,), (1, 1)), ((5,), (1, 1))),
        'no increase':        ((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 3, 2), (1, 2, 2, 1))),
        'in-block TT':        ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 2), (1, 2, 3, 2, 1))),
        'in-block bond 1':    ((5, 6, 7), ((3, 3, 3), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 2, 1))),
        'in-block down':      ((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 4, 2), (1, 2, 2, 1))),
        'non-minimal target': ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((6, 4, 3), (1, 3, 5, 1))),
    }

    def test_valid(self):
        for name, (shape, old_ranks, new_ranks) in self.CASES.items():
            with self.subTest(case=name):
                x_old, old, expected = padded_case(shape, old_ranks, new_ranks)
                padded = padded_frames.fv_padded_frame(old, *new_ranks)
                self.assert_valid(padded, old, expected, x_old.to_dense())

    def test_from_the_empty_frame(self):
        """The zero tensor's start is valid too: an orthonormal frame at the target, both chains reconstructing 0."""
        shape, tucker, tt = (4, 5, 6, 3), (2, 3, 3, 2), (1, 2, 3, 2, 1)
        empty = padded_frames.fv_empty_frame(shape)
        padded = padded_frames.fv_padded_frame(empty, tucker, tt)
        expected = frame_of(t3.TuckerTensorTrain.zeros(shape, tucker, tt))
        self.assert_valid(padded, empty, expected, np.zeros(shape))

    def test_no_increase_returns_the_old_frame(self):
        _, old, _ = padded_case((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 3, 2), (1, 2, 2, 1)))
        for got, want in zip(padded_frames.fv_padded_frame(old, (2, 3, 2), (1, 2, 2, 1)), old):
            self.assertTrue(all(np.array_equal(g, w) for g, w in zip(got, want)))

    def test_sharing(self):
        """A tied old frame: one factor per group (the same array), still a valid frame of X_old."""
        new_ranks = ((5, 5, 3), (1, 4, 3, 1))
        x_old, old, expected = padded_case((6, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), new_ranks, sharing=(0, 0, 1))
        self.assertEqual([u.shape[0] for u in old[0][:2]], [4, 4])        # share() ties at the group rank 4
        self.assertTrue(np.array_equal(old[0][0], old[0][1]))           # TIED: the caller's precondition
        padded = padded_frames.fv_padded_frame(old, *new_ranks, sharing=('a', 'a', 'b'))
        self.assertIs(padded[0][0], padded[0][1])
        self.assert_valid(padded, old, expected, x_old.to_dense())


class TestConvention(PaddedFrameAsserts):
    """The hint-free convention: where each new column comes from."""

    def test_outside_seeds_are_placed_exactly(self):
        """New columns take the coordinates outside the old block first, as exact unit vectors in the
        unfolding's row-major order: left (a, n), right (n, b), down (a, b)."""
        shape, old_ranks, new_ranks = (4, 5, 6, 3), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 3, 4, 3), (1, 3, 4, 3, 1))
        _, old, _ = padded_case(shape, old_ranks, new_ranks)
        padded = padded_frames.fv_padded_frame(old, *new_ranks)
        checked = 0
        for store, axis, modes in (('left', -1, range(3)), ('right', -3, range(1, 4)), ('down', -2, range(4))):
            s = STORES.index(store)
            for i in modes:
                with self.subTest(store=store, mode=i):
                    A0 = np.moveaxis(old[s][i], axis, -1)
                    A = np.moveaxis(padded[s][i], axis, -1)
                    q1, q2, c = A0.shape
                    outside = [(x, y) for x in range(A.shape[0]) for y in range(A.shape[1])
                               if not (x < q1 and y < q2)]
                    for j, (x, y) in enumerate(outside[:A.shape[2] - c]):
                        unit = np.zeros(A.shape[:2])
                        unit[x, y] = 1.0
                        self.assertTrue(np.array_equal(A[:, :, c + j], unit))
                        checked += 1
        self.assertGreater(checked, 10)

    def in_block_columns(self, old_C_xyc, k_in):
        """The expected in-block columns: the old columns completed from the block's coordinate seeds."""
        q1, q2, c = old_C_xyc.shape
        return linalg.pinned_completion(old_C_xyc.reshape(q1 * q2, c), np.eye(q1 * q2)[:, :k_in], k_in)

    def test_in_block_columns_come_from_coordinate_seeds(self):
        """A bond that grows while its neighbours do not (and a down core at saturated bonds) completes inside
        the old block, from the block's coordinate directions in index order -- one and two in-block columns."""
        cases = {   # (shape, old ranks, new ranks, store, mode, output axis, old core, new core, in-block count)
            'left, d=4':    ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 2), (1, 2, 3, 2, 1)),
                             'left', 1, -1, (2, 2, 2), (2, 2, 3), 1),
            'left, bond 1': ((5, 6, 7), ((3, 3, 3), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 2, 1)),
                             'left', 0, -1, (1, 3, 2), (1, 3, 3), 1),
            'down':         ((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 4, 2), (1, 2, 2, 1)),
                             'down', 1, -2, (2, 3, 2), (2, 4, 2), 1),
            'left, two':    ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 2), (1, 2, 4, 2, 1)),
                             'left', 1, -1, (2, 2, 2), (2, 2, 4), 2),
            'right, two':   ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 2), (1, 2, 4, 2, 1)),
                             'right', 2, -3, (2, 2, 2), (4, 2, 2), 2),
        }
        for name, (shape, old_ranks, new_ranks, store, i, axis, old_shape, new_shape, k_in) in cases.items():
            with self.subTest(case=name):
                x_old, old, expected = padded_case(shape, old_ranks, new_ranks)
                padded = padded_frames.fv_padded_frame(old, *new_ranks)
                self.assert_valid(padded, old, expected, x_old.to_dense())
                s = STORES.index(store)
                self.assertEqual((old[s][i].shape, padded[s][i].shape), (old_shape, new_shape))
                A0 = np.moveaxis(old[s][i], axis, -1)
                A = np.moveaxis(padded[s][i], axis, -1)
                q1, q2, c = A0.shape
                self.assertEqual((inside_count(old[s][i], padded[s][i], axis), A.shape[-1] - c), (k_in, k_in))
                got = A[:, :, c:].reshape(q1 * q2, k_in)
                self.assertLessEqual(np.max(np.abs(got - self.in_block_columns(A0, k_in))), 1e-14)

    def test_tucker_completion_is_the_keyed_stream_completed(self):
        """At a nonzero old factor, each group's new Tucker rows are the first rows of RandomState([seed, min(g)])
        made orthonormal against the old rows: against numpy's sign-fixed QR of [U0^T | seeds^T] directly (not
        through the library's completion), unshared and shared, two seeds."""
        cases = {
            'unshared': ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1)), None),
            'shared':   ((6, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((5, 5, 3), (1, 4, 3, 1)), (0, 0, 1)),
        }
        for name, (shape, old_ranks, new_ranks, sharing) in cases.items():
            _, old, _ = padded_case(shape, old_ranks, new_ranks, sharing=sharing)
            groups = (tuple((i,) for i in range(len(shape))) if sharing is None
                      else sharing_module.validate_sharing(sharing, shape))
            for seed in (0, 3):
                with self.subTest(case=name, seed=seed):
                    up = padded_frames.fv_padded_frame(old, *new_ranks, sharing=sharing, seed=seed)[0]
                    completed = 0
                    for group in groups:
                        i = min(group)
                        U0, n, N = old[0][i], up[i].shape[0], up[i].shape[1]
                        s = U0.shape[0]
                        G = np.random.RandomState([seed, i]).standard_normal((n - s, N))
                        Q, R = np.linalg.qr(np.concatenate([U0.T, G.T], axis=1))
                        ref = (Q[:, s:] * np.where(np.diag(R)[s:] >= 0, 1.0, -1.0)).T
                        self.assertTrue(np.array_equal(up[i][:s], U0))
                        self.assertLessEqual(np.max(np.abs(up[i][s:] - ref), initial=0.0), 1e-14)
                        self.assertTrue(all(up[j] is up[i] for j in group))
                        completed += n - s
                    self.assertGreater(completed, 0)

    def test_tucker_completion_is_nested(self):
        """One more Tucker slot at one mode: the leading rows agree to 1e-14; the other modes are untouched."""
        shape, old_ranks = (5, 6, 7), ((2, 2, 2), (1, 2, 2, 1))
        _, old, _ = padded_case(shape, old_ranks, ((3, 3, 3), (1, 3, 3, 1)))
        up3 = padded_frames.fv_padded_frame(old, (3, 3, 3), (1, 3, 3, 1))[0]
        up4 = padded_frames.fv_padded_frame(old, (3, 4, 3), (1, 3, 3, 1))[0]
        self.assertLessEqual(np.max(np.abs(up4[1][:3] - up3[1])), 1e-14)
        self.assertTrue(np.array_equal(up4[0], up3[0]) and np.array_equal(up4[2], up3[2]))

    def test_the_seed_moves_the_tucker_completions_only(self):
        _, old, _ = padded_case((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1)))
        a = padded_frames.fv_padded_frame(old, (3, 4, 3), (1, 3, 3, 1), seed=0)
        b = padded_frames.fv_padded_frame(old, (3, 4, 3), (1, 3, 3, 1), seed=7)
        for got, want in zip(a[1:], b[1:]):
            self.assertTrue(all(np.array_equal(g, w) for g, w in zip(got, want)))
        for U_a, U_b, U0 in zip(a[0], b[0], old[0]):
            s = U0.shape[0]
            self.assertTrue(np.array_equal(U_a[:s], U_b[:s]))
            self.assertGreater(np.max(np.abs(U_a[s:] - U_b[s:])), 1e-3)

    def test_stack_equals_per_element(self):
        """A C = (2,) and (2, 3) stack equals per-element calls to 1e-14 (an in-block case included)."""
        cases = [((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))),
                 ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 2), (1, 2, 3, 2, 1)))]
        for C in [(2,), (2, 3)]:
            for shape, old_ranks, new_ranks in cases:
                with self.subTest(C=C, shape=shape):
                    x_old, old, expected = padded_case(shape, old_ranks, new_ranks, stack_shape=C)
                    padded = padded_frames.fv_padded_frame(old, *new_ranks)
                    for idx in np.ndindex(*C):
                        single = padded_frames.fv_padded_frame(take(old, idx), *new_ranks)
                        for got, want in zip(take(padded, idx), single):
                            for g, w in zip(got, want):
                                self.assertLessEqual(np.max(np.abs(g - w)), 1e-14)
                        self.assert_valid(single, take(old, idx), take(expected, idx), x_old.to_dense()[idx])

    @staticmethod
    def sign_gauge(frame, rng):
        """A random sign gauge: Tucker-row signs (shared by U, L, R), left / right bond signs (shared by the
        chain and the down cores), down-row signs; boundary bonds keep +1 so the base point is unchanged."""
        up, down, left, right = frame
        d = len(up)
        sg = lambda n: rng.choice([-1.0, 1.0], size=n)
        t = [sg(u.shape[0]) for u in up]
        bl = [np.ones(1)] + [sg(left[k].shape[0]) for k in range(1, d)] + [np.ones(1)]
        br = [np.ones(1)] + [sg(right[k].shape[0]) for k in range(1, d)] + [np.ones(1)]
        m = [sg(o.shape[1]) for o in down]
        up2 = tuple(t[i][:, None] * up[i] for i in range(d))
        left2 = tuple(np.einsum('a,n,anb,b->anb', bl[i], t[i], left[i], bl[i + 1]) for i in range(d))
        right2 = tuple(np.einsum('a,n,anb,b->anb', br[i], t[i], right[i], br[i + 1]) for i in range(d))
        down2 = tuple(np.einsum('a,m,amb,b->amb', bl[i], m[i], down[i], br[i + 1]) for i in range(d))
        return up2, down2, left2, right2

    def test_sign_gauge_covariance(self):
        """Two sign-gauge-equivalent old frames give equal assembled projectors: the coordinate seeds make the
        construction covariant under the layers' sign gauge."""
        rng = np.random.default_rng(0)
        cases = {
            'joint':         ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))),
            'in-block TT':   ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 2), (1, 2, 3, 2, 1))),
            'in-block down': ((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 4, 2), (1, 2, 2, 1))),
        }
        for name, (shape, old_ranks, new_ranks) in cases.items():
            _, old, _ = padded_case(shape, old_ranks, new_ranks)
            for trial in range(3):
                with self.subTest(case=name, trial=trial):
                    gauged = self.sign_gauge(old, rng)
                    self.assertLessEqual(float(fvo.fv_frame_orthogonality_residual(gauged)), 1e-14)
                    a = projectors(padded_frames.fv_padded_frame(old, *new_ranks))
                    b = projectors(padded_frames.fv_padded_frame(gauged, *new_ranks))
                    self.assertLessEqual(max(np.linalg.norm(p - q) for p, q in zip(a, b)), 1e-12)


class TestStructuralErrors(unittest.TestCase):
    """Each rejection isolated: the input is good in every respect but one, and the message names the check."""

    def setUp(self):
        _, self.old, _ = padded_case((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1)))
        self.target = ((3, 4, 3), (1, 3, 3, 1))

    def replaced(self, store, i, core):
        frame = [list(cores) for cores in self.old]
        frame[STORES.index(store)][i] = core
        return tuple(tuple(cores) for cores in frame)

    def test_the_good_input_passes(self):
        padded_frames.fv_padded_frame(self.old, *self.target)

    def test_old_frame_layout(self):
        """The old frame must have an orthogonal representation's layout, read off its up and left widths."""
        up, down, left, right = self.old
        cases = {
            'a down width':       (self.replaced('down', 1, np.zeros((2, 3, 2))), 'old_frame down\\[1\\]'),
            'a right width':      (self.replaced('right', 1, np.zeros((3, 2, 2))), 'old_frame right\\[1\\]'),
            'inconsistent left':  (self.replaced('left', 0, np.zeros((1, 2, 3))), 'old_frame left\\[0\\]'),
            'left boundary bond': (self.replaced('left', 0, np.zeros((2, 2, 2))), 'old_frame left\\[0\\]'),
            'a stack on one core': (self.replaced('down', 2, np.zeros((1,) + down[2].shape)), 'old_frame down\\[2\\]'),
            'store lengths':      ((up, down, left[:2], right), 'one core per mode'),
        }
        for name, (frame, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                padded_frames.fv_padded_frame(frame, *self.target)

    def test_target_below_the_old_frame(self):
        for target in [((2, 1, 2), (1, 2, 2, 1)), ((2, 2, 2), (1, 1, 2, 1))]:
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'below the old frame'):
                padded_frames.fv_padded_frame(self.old, *target)

    def test_target_boundary_ranks(self):
        for tt in [(2, 3, 3, 1), (1, 3, 3, 2)]:
            with self.subTest(tt=tt), self.assertRaisesRegex(ValueError, 'boundary TT ranks'):
                padded_frames.fv_padded_frame(self.old, self.target[0], tt)

    def test_target_lengths(self):
        with self.assertRaisesRegex(ValueError, 'len\\(tucker_ranks\\)'):
            padded_frames.fv_padded_frame(self.old, (3, 4), (1, 3, 3, 1))
        with self.assertRaisesRegex(ValueError, 'len\\(tt_ranks\\)'):
            padded_frames.fv_padded_frame(self.old, (3, 4, 3), (1, 3, 1))

    def test_sharing(self):
        """Unequal Tucker stores within a group, in the old frame alone and in the target alone; bad labels."""
        _, tied, _ = padded_case((6, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((4, 4, 3), (1, 4, 3, 1)), sharing=(0, 0, 1))
        _, untied, _ = padded_case((6, 6, 7), ((3, 2, 2), (1, 3, 2, 1)), ((4, 4, 3), (1, 4, 3, 1)))
        self.assertEqual([u.shape[0] for u in untied[0]], [3, 2, 2])
        padded_frames.fv_padded_frame(tied, (4, 4, 3), (1, 4, 3, 1), sharing=(0, 0, 1))      # the good input
        cases = {
            'old frame':    (untied, ((4, 4, 3), (1, 4, 3, 1)), (0, 0, 1), 'in the old frame'),
            'target':       (tied, ((5, 4, 3), (1, 4, 3, 1)), (0, 0, 1), 'in the target'),
            'label count':  (tied, ((4, 4, 3), (1, 4, 3, 1)), (0, 0), 'one group label per mode'),
            'mode sizes':   (tied, ((4, 4, 3), (1, 4, 3, 1)), (0, 1, 0), 'equal mode sizes'),
        }
        for name, (old, target, sharing, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                padded_frames.fv_padded_frame(old, *target, sharing=sharing)


if __name__ == '__main__':
    unittest.main()
