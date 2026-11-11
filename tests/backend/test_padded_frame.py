# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Tests for ``backend.padded_frames``: the empty frame, the seeded padded frame (the start), and one alignment step.

The plan is ``dev/aligned_padded_frame_impl_plan_2026-11-09.md``; T1-T3 (the step) and T4, T6 (the loop) follow T5.

T5, the start, on the D-41 convention (Gaussian seeds in every store, slot-indexed). What is pinned here,
platform-independently:

- **The zero tensor.** At the empty frame the start is the convention's (no longer ``fv_zero_point_frame``'s frame);
  a stack equals per-element calls, bitwise.
- **The convention.** Every store's completion is the sign-fixed pinned QR of its keyed stream's slot-indexed rows,
  against an independent numpy reference (unshared and shared, two seeds, a single-bond increase, d = 1); another
  seed moves every completion and leaves the old blocks bitwise; the Tucker completion is nested; a stack ``C``
  equals per-element calls; one Tucker array per sharing group.
- **Validity at resized points.** Orthonormal, both base-point reconstructions ``X_old``, the old block present
  bitwise with zeros beside it, the carriers zero on every new index.
- **The empty frame** is the orthogonal representation of the rank-0 Tucker tensor train.
- **Structural errors**, each isolated by its message.

T1-T3, the step ``tv_align_padded_frame``, against the dense per-edge reference of ``_padded_frame_oracles.py``
(no code shared with the implementation):

- **T1.** The new frames (largest principal-angle sine per directed edge) and the movement against the dense update,
  for d = 1, 2, 3, 5, uneven increments, non-minimal ranks, a tangent part 10x the normal part, truncating steps, the
  empty old frame, sharing groups, a stack C, and transported stores (``update`` subsets). Every output is a valid
  padded frame: orthonormal, the old block bitwise, the carriers and both base-point reconstructions X_old.
- **T2.** Ungauged variations give the same frames; the hint's scale (1e-14 ... 1e14) changes nothing; a float32 step
  is within a small multiple of float32's unit roundoff of the float64 one.
- **T3.** Degenerate hints (zero, rank one, an information-free increase) give valid frames; the structural
  rejections, each isolated by its message.

T4 and T6, the loop ``fv_aligned_padded_frame``:

- **T6.** Pass by pass against the dense loop on the same schedule (``_padded_frame_oracles.dense_loop``): two-colour
  with p = 0 and p = 5, 40 passes; the simultaneous schedule; the empty old frame; sharing; and with the default
  ``tol``, the same pass count. The schedule itself, recorded call by call: no oversampled phase when working =
  target; the cap counting the truncating pass; ``max_passes`` = 0, 1 and 2; target stores = old stores; the PAIR
  stop rule on prescribed movements (the stop tested only after a ``('UL', 'RO')`` pair, never on the truncating
  pass nor across two pairs; an odd remainder running one final 'UL' only when not stopped; the stack maximum; the
  cap). Movements are a tuple of ``C`` arrays. ``use_jit=True`` without jax warns once and runs eagerly on numpy
  (the optimizers' jax-absent policy). Eager = jit and T13 (a), the compile count, need jax with x64 and live in
  ``tests/test_dispatch.py`` (``TestAlignedLoopJit``).
- **T4.** Two-colour is transport of the other stores, against the dense loop (in T6's two-colour rows); the cycle
  regression on the 243-number case of ``_padded_frame_cycle_case.py``: the simultaneous loop still moves by more than
  0.05 after 200 passes, the alternating loop's movement falls below 1e-8.

numpy-only (house testing strategy); jax dispatch is in ``tests/test_dispatch.py``.
"""
import os
import sys
import unittest
import unittest.mock

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.padded_frames as padded_frames
import t3toolbox.backend.sharing as sharing_module
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.tv_operations as tvo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _padded_frame_cycle_case as cycle_case
import _padded_frame_oracles as oracles

STORES = ('down', 'up', 'left', 'right')


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


def stream_rows(store, key, seed, w, k, p):
    """Rows w .. w+k-1 of the keyed stream RandomState([seed, store tag, key]) (tags: up 0, down 1, left 2, right 3)."""
    tag = {'down': 0, 'up': 1, 'left': 2, 'right': 3}[store]
    return np.random.RandomState([seed, tag, key]).standard_normal((w + k, p))[w:]


def seeded_qr(P, S):
    """The pinned completion, independently: a sign-fixed numpy QR of [P | S], its last S.shape[1] columns; and the
    seeds' Gram-Schmidt pivots |R_jj| (their norms off the earlier columns)."""
    Q, R = np.linalg.qr(np.concatenate([P, S], axis=1))
    signs = np.where(np.diag(R) >= 0, 1.0, -1.0)
    c = P.shape[1]
    return (Q * signs)[:, c:], np.abs(np.diag(R))[c:]


def convention_reference(old, tucker, tt, sharing=None, seed=0, pivots=None):
    """The hint-free start rebuilt from the convention's definition (unstacked): each store's new slots are the seeded
    pinned QR of [the old block embedded | the stream's rows w .. w+k-1]; carriers the old content. ``pivots``, a
    list, collects every seed pivot."""
    down0, up0, left0, right0 = old
    d = len(down0)
    shape = tuple(D.shape[-1] for D in down0)
    nD, nU, rL, rR = ranks.compute_orthogonal_representation_ranks(shape, tucker, tt)
    first = list(range(d))
    if sharing is not None:
        for group in sharing_module.validate_sharing(sharing, shape):
            for i in group:
                first[i] = group[0]

    def core(old_core, widths, axis, store, key):
        A = np.moveaxis(old_core, axis, -1)
        rows = [w for j, w in enumerate(widths) if j != 3 + axis]
        c, k = A.shape[-1], widths[3 + axis] - A.shape[-1]
        P = np.zeros(tuple(rows) + (c,))
        P[:A.shape[0], :A.shape[1]] = A
        P = P.reshape(rows[0] * rows[1], c)
        new, piv = seeded_qr(P, stream_rows(store, key, seed, c, k, P.shape[0]).T)
        if pivots is not None:
            pivots.extend(piv)
        F = np.concatenate([P, new], axis=1).reshape(tuple(rows) + (c + k,))
        return np.moveaxis(F, -1, axis)

    down = []
    for i in range(d):
        f = first[i]
        D0 = down0[f]
        new, piv = seeded_qr(D0.T, stream_rows('down', f, seed, D0.shape[0], nD[f] - D0.shape[0], shape[f]).T)
        if pivots is not None and f == i:
            pivots.extend(piv)
        down.append(np.concatenate([D0, new.T], axis=0))
    pad = lambda A, w: np.pad(A, [(0, t - a) for t, a in zip(w, A.shape)])
    up = [core(up0[i], (rL[i], nU[i], rR[i + 1]), -2, 'up', i) for i in range(d)]
    left = [core(left0[i], (rL[i], nD[i], rL[i + 1]), -1, 'left', i) for i in range(d - 1)]
    left.append(pad(left0[d - 1], (rL[d - 1], nD[d - 1], rL[d])))
    right = [pad(right0[0], (rR[0], nD[0], rR[1]))] + [core(right0[i], (rR[i], nD[i], rR[i + 1]), -3, 'right', i)
                                                       for i in range(1, d)]
    return tuple(down), tuple(up), tuple(left), tuple(right)


def worst_difference(Fa, Fb):
    """The largest entry difference between two ragged frames (shapes asserted)."""
    worst = 0.0
    for a, b in zip(Fa, Fb):
        for x, y in zip(a, b):
            assert np.shape(x) == np.shape(y), (np.shape(x), np.shape(y))
            worst = max(worst, float(np.max(np.abs(np.asarray(x) - y), initial=0.0)))
    return worst


def take(frame, idx):
    """The stack element ``idx`` of a stacked frame."""
    return tuple(tuple(c[idx] for c in cores) for cores in frame)


def projectors(frame):
    """Gauge-invariant summary of a frame: per mode the Tucker projector U^T U, per bond the dense left and
    right interface projectors, per mode the dense down-environment projector (O_i between the interfaces)."""
    down, up, left, right = frame
    d = len(down)
    phi = [np.ones((1, 1))]                             # Phi_k: (prod N_<k, rL_k)
    for k in range(d - 1):
        phi.append(np.einsum('xa,ni,anb->xib', phi[k], down[k], left[k]).reshape(-1, left[k].shape[2]))
    psi = [None] * d + [np.ones((1, 1))]                # Psi_k: (prod N_>=k, rR_k)
    for k in range(d - 1, 0, -1):
        psi[k] = np.einsum('anb,ni,yb->aiy', right[k], down[k], psi[k + 1]).reshape(right[k].shape[0], -1).T
    out = [u.T @ u for u in down]
    out += [p @ p.T for p in phi[1:]] + [p @ p.T for p in psi[1:d]]
    for i in range(d):
        E = np.einsum('xa,amb,yb->xym', phi[i], up[i], psi[i + 1]).reshape(-1, up[i].shape[1])
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
        down, up, left, right = padded
        nX = max(np.linalg.norm(X_old), 1.0)
        for tt in (left, right):
            self.assertLessEqual(np.linalg.norm(t3c.t3_to_dense((down, tt)) - X_old), 1e-13 * nX)
        D0, U0, L0, R0 = old
        for i in range(len(down)):                         # the old block, embedded as a copy (structural)
            self.assertTrue(np.array_equal(down[i][:D0[i].shape[0]], D0[i]))
            self.assertTrue(np.array_equal(up[i][tuple(slice(0, n) for n in U0[i].shape)], U0[i]))
            self.assertTrue(np.array_equal(left[i][tuple(slice(0, n) for n in L0[i].shape)], L0[i]))
            self.assertTrue(np.array_equal(right[i][tuple(slice(0, n) for n in R0[i].shape)], R0[i]))
            a, n, b = L0[i].shape                        # zero elsewhere in the old columns / rows (structural)
            self.assertTrue(np.all(left[i][:, :, :b][a:] == 0.0) and np.all(left[i][:a, n:, :b] == 0.0))
            a, n, b = R0[i].shape
            self.assertTrue(np.all(right[i][:a, :, b:] == 0.0) and np.all(right[i][:a, n:, :b] == 0.0))
            a, m, b = U0[i].shape
            self.assertTrue(np.all(up[i][a:, :m] == 0.0) and np.all(up[i][:a, :m, b:] == 0.0))
        d = len(down)                                      # the carriers: zero on every new index
        a, n, b = L0[d - 1].shape
        self.assertTrue(np.all(left[d - 1][a:] == 0.0) and np.all(left[d - 1][:, n:] == 0.0))
        a, n, b = R0[0].shape
        self.assertTrue(np.all(right[0][:, :, b:] == 0.0) and np.all(right[0][:, n:] == 0.0))


class TestZeroTensor(unittest.TestCase):
    """At the empty frame the start is the convention's (D-41: Gaussian seeds in every store, so no longer
    ``fv_zero_point_frame``'s frame, whose TT cores are coordinate tensors); stacked = per element, bitwise."""

    CASES = [
        ((7,), (3,), (1, 1), None),
        ((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), None),
        ((6, 6, 7), (3, 3, 2), (1, 3, 2, 1), (0, 0, 1)),
        ((4, 5, 6, 3), (2, 3, 3, 2), (1, 2, 3, 2, 1), None),
        ((5, 5, 5, 4), (3, 3, 3, 2), (1, 3, 3, 2, 1), (0, 1, 0, 2)),
    ]

    def test_is_the_convention(self):
        for shape, tucker, tt, sharing in self.CASES:
            groups = () if sharing is None else sharing_module.canonical_groups(sharing, shape)
            for seed in (0, 5):
                with self.subTest(shape=shape, tucker=tucker, tt=tt, sharing=sharing, seed=seed):
                    empty = padded_frames.fv_empty_frame(shape)
                    padded = padded_frames.fv_padded_frame(empty, tucker, tt, sharing=sharing, seed=seed)
                    self.assertLessEqual(worst_difference(padded, convention_reference(empty, tucker, tt, sharing,
                                                                                       seed)), 1e-12)
                    for group in groups:                   # one array per group
                        self.assertTrue(all(padded[0][i] is padded[0][group[0]] for i in group))

    def test_stacked_is_per_element(self):
        shape, tucker, tt = (5, 6, 7), (2, 3, 2), (1, 2, 2, 1)
        padded = padded_frames.fv_padded_frame(padded_frames.fv_empty_frame(shape, (2, 3)), tucker, tt)
        single = padded_frames.fv_padded_frame(padded_frames.fv_empty_frame(shape), tucker, tt)
        for idx in np.ndindex(2, 3):
            for got, want in zip(take(padded, idx), single):
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
    """The hint-free convention (D-41): every store's new slots are the pinned completion of its keyed Gaussian stream's
    slot-indexed rows -- slot s seeded by row s of ``RandomState([seed, store tag, key])`` -- reproduced here
    independently (the stream from ``RandomState``, the completion by a sign-fixed numpy QR)."""

    def test_each_store_is_the_seeded_pinned_qr(self):
        """Every completion against the reference, to 1e-12: unshared and shared, two seeds, an increase that grows a
        single bond (no new rows in the other unfoldings), and d = 1."""
        cases = [((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1)), None),
                 ((4, 5, 6, 3), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 3, 4, 3), (1, 3, 4, 3, 1)), None),
                 ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 2), (1, 2, 3, 2, 1)), None),
                 ((6, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((4, 4, 4), (1, 4, 4, 1)), (0, 0, 1)),
                 ((7,), ((2,), (1, 1)), ((4,), (1, 1)), None)]
        for shape, old_ranks, new_ranks, sharing in cases:
            for seed in (0, 3):
                with self.subTest(shape=shape, new=new_ranks, sharing=sharing, seed=seed):
                    _, old, _ = padded_case(shape, old_ranks, new_ranks, sharing=sharing)
                    got = padded_frames.fv_padded_frame(old, *new_ranks, sharing=sharing, seed=seed)
                    want = convention_reference(old, *new_ranks, sharing, seed)
                    self.assertLessEqual(worst_difference(got, want), 1e-12)

    def test_the_seed_moves_every_completion(self):
        """Another seed: the old blocks bitwise, every completed store different."""
        shape, old_ranks, new_ranks = (4, 5, 6, 3), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 3, 4, 3), (1, 3, 4, 3, 1))
        _, old, _ = padded_case(shape, old_ranks, new_ranks)
        a = padded_frames.fv_padded_frame(old, *new_ranks, seed=0)
        b = padded_frames.fv_padded_frame(old, *new_ranks, seed=1)
        for k in range(4):
            self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a[k], b[k])), STORES[k])

    def test_tucker_completion_is_nested(self):
        """One more Tucker slot at one mode: the leading rows agree to 1e-14; the other modes are untouched."""
        shape, old_ranks = (5, 6, 7), ((2, 2, 2), (1, 2, 2, 1))
        _, old, _ = padded_case(shape, old_ranks, ((3, 3, 3), (1, 3, 3, 1)))
        down3 = padded_frames.fv_padded_frame(old, (3, 3, 3), (1, 3, 3, 1))[0]
        down4 = padded_frames.fv_padded_frame(old, (3, 4, 3), (1, 3, 3, 1))[0]
        self.assertLessEqual(np.max(np.abs(down4[1][:3] - down3[1])), 1e-14)
        self.assertTrue(np.array_equal(down4[0], down3[0]) and np.array_equal(down4[2], down3[2]))

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
        down, up, left, right = self.old
        cases = {
            'a down width':       (self.replaced('up', 1, np.zeros((2, 3, 2))), 'old_frame up\\[1\\]'),
            'a right width':      (self.replaced('right', 1, np.zeros((3, 2, 2))), 'old_frame right\\[1\\]'),
            'inconsistent left':  (self.replaced('left', 0, np.zeros((1, 2, 3))), 'old_frame left\\[0\\]'),
            'left boundary bond': (self.replaced('left', 0, np.zeros((2, 2, 2))), 'old_frame left\\[0\\]'),
            'a stack on one core': (self.replaced('up', 2, np.zeros((1,) + up[2].shape)), 'old_frame up\\[2\\]'),
            'store lengths':      ((down, up, left[:2], right), 'one core per mode'),
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



# ------------------------------------------------------------------------------------------- T1-T3: the step
EMPTY_RANKS = lambda d: ((0,) * d, (1,) + (0,) * (d - 1) + (1,))


def step_case(shape, old_ranks, work_ranks, sharing=None, stack_shape=(), seed=0, zero_old=False, tangent=0.0):
    """(old frame, its ranks, X_old dense, the start at the working ranks, the hint S): x_old random at old_ranks
    (shared, with its ranks kept, under ``sharing``), or the zero tensor; S a generic dense tensor, plus ``tangent``
    times a unit tensor in the old tangent space."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    d = len(shape)
    if zero_old:
        old_ranks = EMPTY_RANKS(d)
        old = padded_frames.fv_empty_frame(shape, stack_shape)
        X_old = np.zeros(stack_shape + shape)
    else:
        x_old = t3.TuckerTensorTrain.randn(shape, *old_ranks, stack_shape=stack_shape)
        if sharing is not None:
            x_old = x_old.share(sharing, max_tucker_ranks=old_ranks[0], max_tt_ranks=old_ranks[1])
            old_ranks = (x_old.tucker_ranks, x_old.tt_ranks)
        old, X_old = frame_of(x_old), x_old.to_dense()
    start = padded_frames.fv_padded_frame(old, *work_ranks, sharing=sharing)
    S = rng.standard_normal(stack_shape + shape)
    if tangent:
        T = t3c.t3_to_dense(tvo.tv_to_t3(old, tvo.tv_project_dense_onto_tangent_space(old, rng.standard_normal(shape))))
        S = S / np.linalg.norm(S) + tangent * T / np.linalg.norm(T)
    return old, old_ranks, X_old, start, S


def element(tree, idx):
    """Stack element ``idx`` of a tuple of tuples of stacked cores."""
    return tuple(tuple(c[idx] for c in cores) for cores in tree)


def worst(Fa, Fb):
    """The largest principal-angle sine between two frames' dense directed-edge bases."""
    return max(oracles.distances(oracles.dense_frames(Fa), oracles.dense_frames(Fb)).values())


def stores_of(frame):
    down, up, left, right = frame
    return (tuple(u.shape[-2] for u in down), tuple(o.shape[-2] for o in up),
            tuple(g.shape[-3] for g in left) + (left[-1].shape[-1],),
            tuple(g.shape[-3] for g in right) + (right[-1].shape[-1],))


class StepAsserts(unittest.TestCase):

    def assert_padded(self, F, old, X_old, stores, groups=()):
        """A valid padded frame (unstacked): the stores, orthonormal to 1e-12, the old block bitwise in the leading
        slots, both base-point reconstructions X_old to 1e-12, one Tucker array per group."""
        self.assertEqual(stores_of(F), tuple(tuple(int(n) for n in st) for st in stores))
        self.assertLessEqual(float(fvo.fv_frame_orthogonality_residual(F)), 1e-12)
        for new, o in zip(F, old):
            for i in range(len(new)):
                self.assertTrue(np.array_equal(new[i][tuple(slice(0, n) for n in o[i].shape)], o[i]))
        down, up, left, right = F
        nX = max(np.linalg.norm(X_old), 1.0)
        for tt in (left, right):
            self.assertLessEqual(np.linalg.norm(t3c.t3_to_dense((down, tt)) - X_old), 1e-12 * nX)
        for group in groups:
            self.assertTrue(all(down[i] is down[group[0]] for i in group))

    def assert_matches_dense(self, start, variations, old, F, movement, stores, update='ULRO', groups=(), tol=1e-9):
        """Frames and movement against the dense per-edge pass from the same start (unstacked)."""
        Z = t3c.t3_to_dense(tvo.tv_to_t3(start, variations))
        ref, ref_movement = oracles.dense_pass(oracles.dense_frames(old), oracles.dense_frames(start), Z, stores,
                                               update, groups)
        self.assertLessEqual(max(oracles.distances(oracles.dense_frames(F), ref).values()), tol)
        self.assertLessEqual(abs(float(movement) - ref_movement), tol)


class TestStepAgainstDense(StepAsserts):
    """T1: one step against the dense per-edge update, and valid padded frames."""

    CASES = {   # shape, old ranks, working ranks (the start), output ranks (None = the start's), options
        'd=1':           ((7,), None, ((1,), (1, 1)), None, {'zero_old': True}),   # see the d = 1 note below
        'd=2':           ((7, 8), ((2, 2), (1, 2, 1)), ((4, 4), (1, 4, 1)), None, {}),
        'd=3':           ((6, 7, 6), ((2, 3, 2), (1, 2, 2, 1)), ((4, 6, 4), (1, 4, 4, 1)), None, {}),
        'd=5':           ((4, 5, 4, 5, 4), ((2, 3, 3, 3, 2), (1, 2, 3, 3, 2, 1)),
                          ((3, 4, 4, 4, 3), (1, 3, 4, 4, 3, 1)), None, {}),
        'uneven':        ((6, 6, 6, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 4, 5, 3), (1, 3, 5, 3, 1)), None, {}),
        'non-minimal':   ((6, 6, 6, 6), ((3, 3, 3, 3), (1, 2, 3, 2, 1)), ((5, 5, 5, 5), (1, 3, 5, 3, 1)), None, {}),
        'tangent 10x':   ((6, 7, 6), ((2, 3, 2), (1, 2, 2, 1)), ((4, 6, 4), (1, 4, 4, 1)), None, {'tangent': 10.0}),
        'truncating':    ((5, 6, 5, 6), ((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((4, 6, 6, 4), (1, 4, 6, 4, 1)),
                          ((3, 5, 5, 3), (1, 3, 5, 3, 1)), {}),
        'empty':         ((5, 6, 5, 6), None, ((3, 4, 4, 3), (1, 3, 4, 3, 1)), None, {'zero_old': True}),
        'empty, truncating': ((5, 6, 5, 6), None, ((4, 6, 5, 4), (1, 4, 6, 4, 1)), ((3, 4, 4, 3), (1, 3, 4, 3, 1)),
                              {'zero_old': True}),
        'groups':        ((6, 6, 6, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((4, 4, 4, 4), (1, 4, 5, 4, 1)), None,
                          {'sharing': (0, 0, 1, 1)}),
        'groups, truncating': ((6, 6, 6, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((5, 5, 5, 5), (1, 5, 6, 5, 1)),
                               ((4, 4, 4, 4), (1, 4, 5, 4, 1)), {'sharing': (0, 0, 0, 0)}),
        'groups, spread': ((5, 6, 5, 6, 5), ((2, 3, 2, 3, 2), (1, 2, 3, 3, 2, 1)),
                           ((3, 4, 3, 4, 3), (1, 3, 4, 4, 3, 1)), None, {'sharing': ('a', 'b', 'a', 'c', 'a')}),
    }

    def check(self, shape, old_ranks, work_ranks, out_ranks, options, update='ULRO'):
        sharing = options.get('sharing')
        old, old_ranks, X_old, start, S = step_case(shape, old_ranks, work_ranks, sharing=sharing,
                                                    zero_old=options.get('zero_old', False),
                                                    tangent=options.get('tangent', 0.0))
        variations = tvo.tv_project_dense_onto_tangent_space(start, S)
        F, movement = padded_frames.tv_align_padded_frame(start, variations, *old_ranks,
                                                          *(out_ranks or (None, None)), sharing=sharing, update=update)
        stores = ranks.compute_orthogonal_representation_ranks(shape, *(out_ranks or work_ranks))
        groups = () if sharing is None else sharing_module.canonical_groups(sharing, shape)
        self.assertEqual(np.shape(movement), ())
        self.assert_padded(F, old, X_old, stores, groups)
        self.assert_matches_dense(start, variations, old, F, movement, stores, update, groups)

    # The d = 1 note. At d = 1 the down store is one wide (nU = min(nD, rL rR) = 1), so a nonzero old point's down
    # frame spans the whole far side and the up edge's content vanishes: its completion carries no information
    # (T3's information-free case). The informative d = 1 step starts at the zero tensor, at Tucker rank 1.

    def test_cases(self):
        for name, args in self.CASES.items():
            with self.subTest(case=name):
                self.check(*args)

    def test_transported_stores(self):
        """A store not in ``update`` keeps its completion, re-expressed in the new coordinates (the dense
        transport), while the others align -- each family transported, and the two-colour halves."""
        args = ((5, 6, 5, 6), ((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((3, 5, 5, 3), (1, 3, 5, 3, 1)), None, {})
        for update in ('UL', 'RO', 'LRO', 'ULO', 'U', 'O'):
            with self.subTest(update=update):
                self.check(*args, update=update)

    def test_the_tangent_projection_is_the_dense_one(self):
        """The setup's tangent vector is the dense orthogonal projection onto the padded frame's tangent space."""
        old, old_ranks, X_old, start, S = step_case((5, 6, 5), ((2, 3, 2), (1, 2, 2, 1)), ((3, 5, 3), (1, 3, 3, 1)))
        Z = t3c.t3_to_dense(tvo.tv_to_t3(start, tvo.tv_project_dense_onto_tangent_space(start, S)))
        self.assertLessEqual(np.linalg.norm(Z - oracles.dense_project(oracles.dense_frames(start), S)),
                             1e-12 * np.linalg.norm(Z))

    def test_stack(self):
        """A stack C = (2,) and (2, 3): each element matches the dense update and is valid; the movement has
        shape C and equals the per-element calls'."""
        shape, old_ranks, work_ranks = (5, 6, 5), ((2, 3, 2), (1, 2, 2, 1)), ((3, 5, 3), (1, 3, 3, 1))
        out_ranks = ((3, 4, 3), (1, 3, 3, 1))
        for C in [(2,), (2, 3)]:
            with self.subTest(C=C):
                old, _, X_old, start, S = step_case(shape, old_ranks, work_ranks, stack_shape=C)
                variations = tvo.tv_project_dense_onto_tangent_space(start, S)
                F, movement = padded_frames.tv_align_padded_frame(start, variations, *old_ranks, *out_ranks)
                self.assertEqual(movement.shape, C)
                stores = ranks.compute_orthogonal_representation_ranks(shape, *out_ranks)
                for idx in np.ndindex(*C):
                    F_e, mv_e = padded_frames.tv_align_padded_frame(element(start, idx), element(variations, idx),
                                                                    *old_ranks, *out_ranks)
                    self.assertLessEqual(worst(element(F, idx), F_e), 1e-12)
                    self.assertLessEqual(abs(float(movement[idx]) - float(mv_e)), 1e-12)
                    self.assert_padded(element(F, idx), element(old, idx), X_old[idx], stores)
                    self.assert_matches_dense(element(start, idx), element(variations, idx), element(old, idx),
                                              element(F, idx), movement[idx], stores)


class TestStepInvariances(StepAsserts):
    """T2: the gauge of the variations, the scale of the hint, and float32."""

    def setUp(self):
        self.old, self.old_ranks, self.X_old, self.start, S = step_case(
            (5, 6, 5, 6), ((2, 3, 3, 2), (1, 2, 3, 2, 1)), ((3, 5, 5, 3), (1, 3, 5, 3, 1)), seed=3)
        self.variations = tvo.tv_project_dense_onto_tangent_space(self.start, S)
        self.F, self.movement = padded_frames.tv_align_padded_frame(self.start, self.variations, *self.old_ranks)

    def test_ungauged_variations(self):
        """The same tangent vector with O(1) gauge terms moved across every bond and every leaf-core edge."""
        rng = np.random.default_rng(11)
        D, U, L, R = self.start
        V, H = [np.array(v) for v in self.variations[0]], [np.array(h) for h in self.variations[1]]
        d = len(D)
        for i in range(d - 1):                                           # across bond i+1
            Cm = rng.standard_normal((L[i].shape[2], R[i + 1].shape[0]))
            H[i] = H[i] + np.einsum('anc,cb->anb', L[i], Cm)
            H[i + 1] = H[i + 1] - np.einsum('ac,cnb->anb', Cm, R[i + 1])
        for i in range(d):                                               # between leaf i and core i
            Mm = rng.standard_normal((U[i].shape[1], D[i].shape[0]))
            V[i] = V[i] + Mm @ D[i]
            H[i] = H[i] - np.einsum('amb,mn->anb', U[i], Mm)
        ungauged = (tuple(V), tuple(H))
        Z, Zu = (t3c.t3_to_dense(tvo.tv_to_t3(self.start, v)) for v in (self.variations, ungauged))
        self.assertLessEqual(np.linalg.norm(Zu - Z), 1e-12 * np.linalg.norm(Z))        # the same tangent vector
        self.assertGreater(min(np.linalg.norm(a - b) for a, b in zip(V, self.variations[0])), 1e-1)
        F, movement = padded_frames.tv_align_padded_frame(self.start, ungauged, *self.old_ranks)
        self.assertLessEqual(worst(F, self.F), 1e-9)
        self.assertLessEqual(abs(float(movement) - float(self.movement)), 1e-9)

    def test_the_hint_scale_changes_nothing(self):
        for c in (1e-14, 1e-7, 1e7, 1e14):
            with self.subTest(scale=c):
                scaled = tuple(tuple(c * a for a in cores) for cores in self.variations)
                F, movement = padded_frames.tv_align_padded_frame(self.start, scaled, *self.old_ranks)
                self.assertLessEqual(worst(F, self.F), 1e-9)
                self.assertLessEqual(abs(float(movement) - float(self.movement)), 1e-9)

    def test_float32(self):
        """One step from an informed frame, hint N + s T (N normal, T in the old tangent space, unit norms), in float32
        against float64: within 30 u max(1, s), u = float32's unit roundoff; float32 out."""
        rng = np.random.default_rng(5)
        shape = (5, 6, 5, 6)
        F0 = self.start
        for _ in range(4):                                               # a few passes: the frame carries information
            F0, _ = padded_frames.tv_align_padded_frame(
                F0, tvo.tv_project_dense_onto_tangent_space(F0, rng.standard_normal(shape)), *self.old_ranks)
        T_old = lambda Y: t3c.t3_to_dense(tvo.tv_to_t3(self.old, tvo.tv_project_dense_onto_tangent_space(self.old, Y)))
        S = rng.standard_normal(shape)
        N = S - T_old(S)
        N /= np.linalg.norm(N)
        T = T_old(rng.standard_normal(shape))
        T /= np.linalg.norm(T)
        cast = lambda tree, dt: tuple(tuple(np.asarray(a, dtype=dt) for a in cores) for cores in tree)
        u = np.finfo(np.float32).eps / 2
        for s in (0.0, 1e2, 1e4):
            with self.subTest(s=s):
                g = tvo.tv_project_dense_onto_tangent_space(F0, N + s * T)
                F64, _ = padded_frames.tv_align_padded_frame(F0, g, *self.old_ranks)
                F32, mv32 = padded_frames.tv_align_padded_frame(cast(F0, np.float32), cast(g, np.float32),
                                                                *self.old_ranks)
                self.assertTrue(all(a.dtype == np.float32 for cores in F32 for a in cores))
                self.assertEqual(mv32.dtype, np.float32)
                self.assertLessEqual(worst(cast(F32, np.float64), F64), 30 * u * max(1.0, s))


class TestStepDegenerateAndStructural(StepAsserts):
    """T3: degenerate hints give valid frames; structural rejections, each isolated."""

    def test_degenerate_hints(self):
        """A zero hint, a rank-one hint, and an information-free Tucker increase (rows beyond r_i r_(i+1) have no
        down slots, so the tangent space does not depend on them): valid frames, no rank decision anywhere."""
        cases = {   # shape, old ranks, working ranks, output ranks, hint
            'zero':            ((6, 6, 6, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((4, 4, 4, 4), (1, 4, 5, 4, 1)), None,
                                'zero'),
            'zero, truncating': ((6, 6, 6, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((5, 5, 5, 5), (1, 5, 6, 5, 1)),
                                 ((4, 4, 4, 4), (1, 4, 5, 4, 1)), 'zero'),
            'rank one':        ((6, 6, 6, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((4, 4, 4, 4), (1, 4, 5, 4, 1)), None,
                                'rank one'),
            'information-free': ((7, 7, 7), ((2, 4, 2), (1, 2, 2, 1)), ((3, 6, 3), (1, 2, 2, 1)), None, 'generic'),
            'information-free, d=1': ((7,), ((2,), (1, 1)), ((5,), (1, 1)), None, 'generic'),
        }
        for name, (shape, old_ranks, work_ranks, out_ranks, hint) in cases.items():
            with self.subTest(case=name):
                old, old_ranks, X_old, start, S = step_case(shape, old_ranks, work_ranks, seed=9)
                if hint == 'rank one':
                    rng = np.random.default_rng(9)
                    S = rng.standard_normal(shape[0])
                    for n in shape[1:]:
                        S = np.multiply.outer(S, rng.standard_normal(n))
                variations = tvo.tv_project_dense_onto_tangent_space(start, S)
                if hint == 'zero':
                    variations = tuple(tuple(np.zeros_like(a) for a in cores) for cores in variations)
                F, movement = padded_frames.tv_align_padded_frame(start, variations, *old_ranks,
                                                                  *(out_ranks or (None, None)))
                stores = ranks.compute_orthogonal_representation_ranks(shape, *(out_ranks or work_ranks))
                self.assert_padded(F, old, X_old, stores)
                self.assertTrue(np.isfinite(float(movement)))

    def setUp(self):
        self.shape = (5, 6, 5, 6)
        self.old_ranks = ((2, 3, 3, 2), (1, 2, 3, 2, 1))
        self.work = ((4, 6, 5, 4), (1, 4, 6, 4, 1))
        self.target = ((3, 5, 5, 3), (1, 3, 5, 3, 1))
        self.old, _, _, self.start, S = step_case(self.shape, self.old_ranks, self.work)
        self.variations = tvo.tv_project_dense_onto_tangent_space(self.start, S)

    def step(self, frame=None, variations=None, old_ranks=None, out=(None, None), **kw):
        return padded_frames.tv_align_padded_frame(self.start if frame is None else frame,
                                                   self.variations if variations is None else variations,
                                                   *(old_ranks or self.old_ranks), *out, **kw)

    def test_the_good_inputs_pass(self):
        self.step()
        self.step(out=self.target)
        self.step(update='UL')

    def test_transport_with_truncation(self):
        """Every family is truncated working -> target; leaving one family out of ``update`` names that family."""
        self.assertTrue(all(a > b for a, b in zip(
            [max(s) for s in ranks.compute_orthogonal_representation_ranks(self.shape, *self.work)],
            [max(s) for s in ranks.compute_orthogonal_representation_ranks(self.shape, *self.target)])))
        for missing, store in (('U', 'down'), ('O', 'up'), ('L', 'left'), ('R', 'right')):
            update = ''.join(c for c in 'ULRO' if c != missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, store + ' store .* not updated'):
                self.step(out=self.target, update=update)

    def test_bad_update(self):
        for update in ('', 'UU', 'X', 'ULROX', 'ulro'):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'without repeats'):
                self.step(update=update)

    def test_store_order(self):
        """The old block must fit the output, and the output the frame."""
        with self.assertRaisesRegex(ValueError, 'old block must fit'):
            self.step(old_ranks=((4, 3, 3, 2), (1, 2, 3, 2, 1)), out=self.target)         # old up 4 > output 3
        with self.assertRaisesRegex(ValueError, 'old block must fit'):
            self.step(out=((4, 6, 5, 4), (1, 4, 6, 5, 1)))                    # left store 3: output 5 above the frame's 4

    def test_ranks(self):
        with self.assertRaisesRegex(ValueError, 'both output rank sequences'):
            self.step(out=(self.target[0], None))
        with self.assertRaisesRegex(ValueError, 'old boundary TT ranks'):
            self.step(old_ranks=(self.old_ranks[0], (2, 2, 3, 2, 1)))
        with self.assertRaisesRegex(ValueError, 'output boundary TT ranks'):
            self.step(out=(self.target[0], (1, 3, 5, 3, 2)))
        with self.assertRaisesRegex(ValueError, 'len\\(tucker_ranks\\)'):
            self.step(old_ranks=((2, 3, 3), (1, 2, 3, 2, 1)))

    def test_frame_and_variation_shapes(self):
        down, up, left, right = self.start
        bad_frame = (down, up[:1] + (np.zeros((2, 3, 2)),) + up[2:], left, right)
        with self.assertRaisesRegex(ValueError, 'frame up\\[1\\]'):
            self.step(frame=bad_frame)
        V, H = self.variations
        with self.assertRaisesRegex(ValueError, 'tucker variations'):
            self.step(variations=((np.zeros((1, 5)),) + V[1:], H))
        with self.assertRaisesRegex(ValueError, 'tt variations'):
            self.step(variations=(V, H[:3]))

    def test_unequal_group_stores(self):
        """Sharing labels on stores unequal within a group: the old frame's, the output's, the frame's own."""
        shape = (6, 6, 6, 6)
        old, old_ranks, _, start, S = step_case(shape, ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
                                                ((4, 4, 4, 4), (1, 4, 5, 4, 1)), sharing=(0, 0, 1, 1))
        variations = tvo.tv_project_dense_onto_tangent_space(start, S)
        step = lambda **kw: padded_frames.tv_align_padded_frame(start, variations, **kw)
        good = dict(old_tucker_ranks=old_ranks[0], old_tt_ranks=old_ranks[1], sharing=(0, 0, 1, 1))
        step(**good)
        with self.assertRaisesRegex(ValueError, 'unequal Tucker stores .* \\(old\\)'):
            step(**dict(good, old_tucker_ranks=(2, 1, 2, 2)))
        with self.assertRaisesRegex(ValueError, 'unequal Tucker stores .* \\(output\\)'):
            step(**dict(good, tucker_ranks=(4, 3, 4, 4), tt_ranks=(1, 4, 5, 4, 1)))
        uneven = padded_frames.fv_padded_frame(old, (4, 4, 5, 4), (1, 4, 5, 4, 1))      # Tucker stores 5, 4 in (2, 3)
        uneven_variations = tvo.tv_project_dense_onto_tangent_space(uneven, S)
        with self.assertRaisesRegex(ValueError, 'unequal Tucker stores .* \\(frame\\)'):
            padded_frames.tv_align_padded_frame(uneven, uneven_variations, *old_ranks, (4, 4, 4, 4), (1, 4, 5, 4, 1),
                                                sharing=(0, 0, 1, 1))


# ------------------------------------------------------------------------------------------- T4, T6: the loop
def recording_project(seen):
    """The dense projection, recording every frame it is called at (the loop's frame before each pass)."""
    def project(frame, S):
        seen.append(frame)
        return tvo.tv_project_dense_onto_tangent_space(frame, S)
    return project


def raising_project(frame, hint):
    raise AssertionError('project must not be called')


class TestLoopAgainstDense(StepAsserts):
    """T6 (and T4's two-colour = transport): the loop pass by pass against the dense loop on the same schedule."""

    CASES = {   # shape, old ranks (None: the empty frame), target ranks, sharing, oversampling, passes, alternate
        'two-colour, p=0':      ((4, 5, 4), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1)), None, 0, 40, True),
        'two-colour, p=5':      ((4, 5, 4), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1)), None, 5, 40, True),
        'simultaneous, p=5':    ((4, 5, 4), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1)), None, 5, 12, False),
        'two-colour, d=4':      ((4, 5, 4, 3), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 3, 3, 2), (1, 3, 3, 2, 1)), None,
                                 5, 6, True),
        'empty old frame':      ((4, 5, 4), None, ((2, 3, 2), (1, 2, 2, 1)), None, 5, 20, True),
        'sharing':              ((4, 4, 5), ((2, 2, 2), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 3, 1)), (0, 0, 1), 0, 20,
                                 True),
    }

    def test_against_the_dense_loop(self):
        for name, (shape, old_ranks, target, sharing, p, passes, alternate) in self.CASES.items():
            with self.subTest(case=name):
                old, old_ranks, X_old, _, S = step_case(shape, old_ranks or EMPTY_RANKS(len(shape)), target,
                                                        sharing=sharing, zero_old=old_ranks is None, seed=7)
                seen = []
                F, movements = padded_frames.fv_aligned_padded_frame(
                    old, *target, S, recording_project(seen), sharing=sharing, oversampling=p, max_passes=passes,
                    tol=0.0, alternate=alternate)
                self.assertIsInstance(movements, tuple)
                self.assertEqual(len(movements), passes)                          # tol 0: no early stop
                working = ranks.oversampled_ranks(shape, *target, p, sharing)
                self.assertEqual(stores_of(seen[0]), ranks.compute_orthogonal_representation_ranks(shape, *working))
                groups = () if sharing is None else sharing_module.canonical_groups(sharing, shape)
                frames, moves = oracles.dense_loop(
                    oracles.dense_frames(old), oracles.dense_frames(seen[0]), S,
                    ranks.compute_orthogonal_representation_ranks(shape, *working),
                    ranks.compute_orthogonal_representation_ranks(shape, *target),
                    padded_frames.ALIGN_OVERSAMPLED_PASSES, passes, 0.0, alternate, groups)
                for k, (Fk, Dk) in enumerate(zip(seen[1:] + [F], frames)):
                    self.assertLessEqual(max(oracles.distances(oracles.dense_frames(Fk), Dk).values()), 1e-9,
                                         'pass %d' % k)
                    self.assertLessEqual(abs(float(movements[k]) - moves[k]), 1e-9, 'pass %d' % k)
                stores = ranks.compute_orthogonal_representation_ranks(shape, *target)
                self.assert_padded(F, old, X_old, stores, groups)

    def test_the_default_tol_stops_both_at_the_same_pass(self):
        """With the default tol and a cap of 40 the loop and the dense loop stop after the same pass (both rules written
        from the plan's text, independently), early, with matching movements."""
        for name in ('two-colour, p=5', 'simultaneous, p=5', 'sharing'):
            shape, old_ranks, target, sharing, p, _, alternate = self.CASES[name]
            passes = 40
            with self.subTest(case=name):
                old, _, _, _, S = step_case(shape, old_ranks, target, sharing=sharing, seed=7)
                seen = []
                _, movements = padded_frames.fv_aligned_padded_frame(
                    old, *target, S, recording_project(seen), sharing=sharing, oversampling=p, max_passes=passes,
                    alternate=alternate)
                working = ranks.oversampled_ranks(shape, *target, p, sharing)
                groups = () if sharing is None else sharing_module.canonical_groups(sharing, shape)
                _, moves = oracles.dense_loop(
                    oracles.dense_frames(old), oracles.dense_frames(seen[0]), S,
                    ranks.compute_orthogonal_representation_ranks(shape, *working),
                    ranks.compute_orthogonal_representation_ranks(shape, *target),
                    padded_frames.ALIGN_OVERSAMPLED_PASSES, passes, padded_frames.ALIGN_TOL, alternate, groups)
                self.assertEqual(len(movements), len(moves))
                self.assertLess(len(movements), passes)                           # it did stop early
                self.assertLessEqual(max(abs(float(a) - b) for a, b in zip(movements, moves)), 1e-9)

    def test_stacked(self):
        """A stack C = (2,): movements are a tuple of C arrays, and each element is the per-element loop."""
        shape, old_ranks, target = (4, 5, 4), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))
        old, _, X_old, _, S = step_case(shape, old_ranks, target, stack_shape=(2,))
        project = lambda frame, Y: tvo.tv_project_dense_onto_tangent_space(frame, Y)
        F, movements = padded_frames.fv_aligned_padded_frame(old, *target, S, project, max_passes=6, tol=0.0)
        self.assertEqual(len(movements), 6)
        self.assertTrue(all(isinstance(m, np.ndarray) and m.shape == (2,) for m in movements))
        for e in range(2):
            F_e, movements_e = padded_frames.fv_aligned_padded_frame(element(old, e), *target, S[e], project,
                                                                     max_passes=6, tol=0.0)
            self.assertLessEqual(worst(element(F, e), F_e), 1e-12)
            self.assertLessEqual(max(abs(float(a[e]) - float(b)) for a, b in zip(movements, movements_e)), 1e-12)


class TestLoopSchedule(unittest.TestCase):
    """T6: the schedule, recorded call by call (the real step, wrapped), and the stop rule on prescribed movements."""

    def setUp(self):
        self.shape = (4, 5, 4, 3)
        self.old, _, _, _, self.S = step_case(self.shape, ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 3, 3, 2), (1, 3, 3, 2, 1)))
        self.target = ((3, 3, 3, 2), (1, 3, 3, 2, 1))
        self.project = lambda frame, Y: tvo.tv_project_dense_onto_tangent_space(frame, Y)

    def run_recorded(self, **kw):
        """(frame, movements, [(update, output Tucker ranks or None)] per pass)."""
        with unittest.mock.patch.object(padded_frames, 'tv_align_padded_frame',
                                        wraps=padded_frames.tv_align_padded_frame) as step:
            F, movements = padded_frames.fv_aligned_padded_frame(self.old, *self.target, self.S, self.project, **kw)
        calls = [(c.args[7], c.args[4]) for c in step.call_args_list]    # positional: (update, output Tucker ranks)
        return F, movements, calls

    def test_working_equals_target_has_no_oversampled_phase(self):
        F, movements, calls = self.run_recorded(oversampling=0, max_passes=4, tol=0.0)
        t = self.target[0]
        self.assertEqual(calls, [('ULRO', t), ('UL', t), ('RO', t), ('UL', t)])
        self.assertEqual(len(movements), 4)

    def test_the_default_shape_of_a_run(self):
        """Two oversampled passes, the truncating pass, then 'UL' and 'RO' in turn; the simultaneous schedule
        updates every store on every pass."""
        t = self.target[0]
        _, _, calls = self.run_recorded(max_passes=6, tol=0.0)
        self.assertEqual(calls, [('ULRO', None), ('ULRO', None), ('ULRO', t), ('UL', t), ('RO', t), ('UL', t)])
        _, _, calls = self.run_recorded(max_passes=4, tol=0.0, alternate=False)
        self.assertEqual(calls, [('ULRO', None), ('ULRO', None), ('ULRO', t), ('ULRO', t)])

    def test_two_passes_at_working_equal_to_target(self):
        """max_passes = 2 with no oversampled phase: the 'ULRO' pass, then one slot, no pair: the final 'UL'."""
        _, movements, calls = self.run_recorded(oversampling=0, max_passes=2, tol=0.0)
        self.assertEqual(calls, [('ULRO', self.target[0]), ('UL', self.target[0])])
        self.assertEqual(len(movements), 2)

    def test_the_cap_counts_the_truncating_pass(self):
        F, movements, calls = self.run_recorded(oversampled_passes=5, max_passes=3, tol=0.0)
        self.assertEqual(calls, [('ULRO', None), ('ULRO', None), ('ULRO', self.target[0])])
        self.assertEqual(stores_of(F), ranks.compute_orthogonal_representation_ranks(self.shape, *self.target))

    def test_one_pass(self):
        """max_passes = 1: one truncating pass from the working start, or one 'ULRO' pass at the target."""
        for p in (5, 0):
            with self.subTest(oversampling=p):
                F, movements, calls = self.run_recorded(oversampling=p, max_passes=1, tol=0.0)
                self.assertEqual(calls, [('ULRO', self.target[0])])
                self.assertEqual(len(movements), 1)
                self.assertEqual(stores_of(F), ranks.compute_orthogonal_representation_ranks(self.shape, *self.target))

    def test_zero_passes_is_the_start_at_the_target(self):
        """No projection, no step, empty movements; the frame is fv_padded_frame at the target, bitwise."""
        start = padded_frames.fv_padded_frame(self.old, *self.target, seed=3)
        for p in (0, 5):
            with self.subTest(oversampling=p):
                with unittest.mock.patch.object(padded_frames, 'tv_align_padded_frame') as step:
                    F, movements = padded_frames.fv_aligned_padded_frame(
                        self.old, *self.target, None, raising_project, oversampling=p, max_passes=0, seed=3)
                step.assert_not_called()
                self.assertEqual(movements, ())
                for got, want in zip(F, start):
                    self.assertTrue(all(np.array_equal(g, w) for g, w in zip(got, want)))

    def test_the_working_start_uses_the_seed(self):
        """The first frame the loop projects at is fv_padded_frame at the working ranks with the loop's seed, bitwise;
        another seed moves it."""
        working = ranks.oversampled_ranks(self.shape, *self.target, padded_frames.ALIGN_OVERSAMPLING)
        for seed in (3, 0):
            with self.subTest(seed=seed):
                seen = []
                padded_frames.fv_aligned_padded_frame(self.old, *self.target, self.S, recording_project(seen),
                                                      max_passes=1, seed=seed)
                want = padded_frames.fv_padded_frame(self.old, *working, seed=seed)
                for got, w in zip(seen[0], want):
                    self.assertTrue(all(np.array_equal(g, x) for g, x in zip(got, w)))
        self.assertFalse(np.array_equal(padded_frames.fv_padded_frame(self.old, *working, seed=3)[0][0],
                                        padded_frames.fv_padded_frame(self.old, *working, seed=0)[0][0]))

    def test_nothing_to_complete(self):
        """Target stores equal to the old stores: no pass, empty movements, the old frame (values, bitwise)."""
        with unittest.mock.patch.object(padded_frames, 'tv_align_padded_frame') as step:
            F, movements = padded_frames.fv_aligned_padded_frame(
                self.old, (2, 2, 2, 2), (1, 2, 2, 2, 1), None, raising_project)
        step.assert_not_called()
        self.assertEqual(movements, ())
        for got, want in zip(F, self.old):
            self.assertTrue(all(np.array_equal(g, w) for g, w in zip(got, want)))

    def stop_after(self, prescribed, **kw):
        """The number of passes the loop runs when the step returns the prescribed movements in turn."""
        moves = iter(prescribed)
        fake = lambda frame, variations, *args, **kws: (frame, np.asarray(next(moves)))
        with unittest.mock.patch.object(padded_frames, 'tv_align_padded_frame', side_effect=fake):
            _, movements = padded_frames.fv_aligned_padded_frame(self.old, *self.target, None, lambda F, h: None,
                                                                 **kw)
        return len(movements)

    def test_the_stop_rule(self):
        """The PAIR rule, with the default two oversampled passes (pass 2 truncates), cap 10 and tol 0.05: the passes
        after the truncating one come in ('UL', 'RO') pairs (3-4, 5-6, 7-8) and the stop is tested after each pair,
        both movements below tol; never on the truncating pass, never across two pairs; the odd slot left by the cap
        (pass 9) runs one final 'UL' only when not stopped. Simultaneous: the first pass from the truncating one on
        below tol. Oversampled passes never stop. The stack maximum decides; the cap stops."""
        lo, hi = 0.01, 0.9
        self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi]), 5)                  # the first pair
        self.assertEqual(self.stop_after([hi, hi, lo, lo, hi, lo, lo, hi]), 7)          # not the truncating pass
        self.assertEqual(self.stop_after([hi, hi, hi, hi, lo, lo, hi, hi, hi, hi]), 10)  # not across pairs (4-5)
        self.assertEqual(self.stop_after([lo, lo, lo, lo, lo, hi]), 5)                  # not during oversampling
        self.assertEqual(self.stop_after([hi, hi, hi, [lo, hi], lo, lo, lo, hi]), 7)    # the max over the stack
        self.assertEqual(self.stop_after([hi] * 9 + [lo]), 10)                          # the cap: 3 pairs, a final 'UL'
        self.assertEqual(self.stop_after([hi] * 9, max_passes=9), 9)                    # an even cap: no final 'UL'
        self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final 'UL'
        self.assertEqual(self.stop_after([hi] * 6, max_passes=6), 6)                    # not stopped: the final 'UL'
        self.assertEqual(self.stop_after([hi] * 3, max_passes=3), 3)
        self.assertEqual(self.stop_after([hi, lo, lo, hi], alternate=False), 3)        # simultaneous
        self.assertEqual(self.stop_after([lo, lo, hi, lo, hi], alternate=False), 4)
        self.assertEqual(self.stop_after([hi, hi, [lo, hi], lo, hi], alternate=False), 4)
        nan = np.nan                                                                    # a NaN never stops,
        self.assertEqual(self.stop_after([hi, hi, nan, lo, hi], alternate=False), 4)    #   in either place
        self.assertEqual(self.stop_after([hi, hi, hi, nan, lo, hi], alternate=False), 5)
        self.assertEqual(self.stop_after([hi, hi, hi, nan, lo, lo, lo, hi]), 7)         #   of a pair
        self.assertEqual(self.stop_after([hi, hi, hi, lo, nan, lo, lo, hi]), 7)
        self.assertEqual(self.stop_after([hi, hi, hi, [lo, nan], lo, lo, lo, hi]), 7)   #   or of a stack

    def test_the_stopping_pass_types(self):
        """Recorded: a stop after the first pair leaves (oversampled x2, truncating, 'UL', 'RO') and no more."""
        moves = iter([0.9, 0.9, 0.9, 0.01, 0.01, 0.9])
        updates = []

        def fake(frame, variations, *args):
            updates.append(args[5])
            return frame, np.asarray(next(moves))
        with unittest.mock.patch.object(padded_frames, 'tv_align_padded_frame', side_effect=fake):
            padded_frames.fv_aligned_padded_frame(self.old, *self.target, None, lambda F, h: None)
        self.assertEqual(updates, ['ULRO', 'ULRO', 'ULRO', 'UL', 'RO'])

    def test_use_jit_without_jax(self):
        """jax absent (simulated, as the optimizers' test does): use_jit=True warns once and runs eagerly on numpy,
        the result the eager run's; use_jit=False does not warn."""
        import warnings
        import t3toolbox.backend.common as common
        kw = dict(max_passes=4, tol=0.0)
        F, movements = padded_frames.fv_aligned_padded_frame(self.old, *self.target, self.S, self.project, **kw)
        saved = common.jax_available
        common.jax_available = False
        try:
            for use_jit in (True, False):
                with self.subTest(use_jit=use_jit):
                    common._JAX_WARNED.discard('use_jit=True')
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter('always')
                        F_n, movements_n = padded_frames.fv_aligned_padded_frame(
                            self.old, *self.target, self.S, self.project, use_jit=use_jit, **kw)
                    self.assertEqual(any('use_jit=True' in str(m.message) for m in w), use_jit)
                    self.assertTrue(all(isinstance(c, np.ndarray) for cores in F_n for c in cores))
                    self.assertTrue(all(np.array_equal(a, b) for x, y in zip(F_n, F) for a, b in zip(x, y)))
                    self.assertEqual([float(m) for m in movements_n], [float(m) for m in movements])
        finally:
            common.jax_available = saved
            common._JAX_WARNED.discard('use_jit=True')

    def test_structural_errors(self):
        for kw, message in ((dict(max_passes=-1), 'max_passes'), (dict(max_passes=True), 'max_passes'),
                            (dict(oversampled_passes=-1), 'oversampled_passes'), (dict(oversampling=-1), 'non-negative')):
            with self.subTest(**{k: repr(v) for k, v in kw.items()}), self.assertRaisesRegex(ValueError, message):
                padded_frames.fv_aligned_padded_frame(self.old, *self.target, self.S, self.project, **kw)
        with self.assertRaisesRegex(ValueError, 'below the old frame'):
            padded_frames.fv_aligned_padded_frame(self.old, (2, 1, 2, 2), (1, 2, 2, 2, 1), self.S, self.project)
        with self.assertRaisesRegex(ValueError, 'oversampling must be a non-negative int'):   # checked even when
            padded_frames.fv_aligned_padded_frame(self.old, *self.target, self.S, self.project,  # zero passes do not
                                                  oversampling=-1, max_passes=0)                # read it
        with self.assertRaisesRegex(ValueError, 'target boundary TT ranks'):
            padded_frames.fv_aligned_padded_frame(self.old, self.target[0], (2, 3, 3, 2, 1), self.S, self.project)


class TestLoopCycle(unittest.TestCase):
    """T4: the cycle regression on the 243-number case (``_padded_frame_cycle_case.py``): with p = 0 the simultaneous
    loop still moves by more than 0.05 after 200 passes (a period-2 cycle); the two-colour loop converges."""

    def setUp(self):
        self.old = fvc.t3_orthogonal_representations((cycle_case.OLD_TUCKER_CORES, cycle_case.OLD_TT_CORES))[0]
        self.project = tvo.tv_project_dense_onto_tangent_space

    def run_loop(self, **kw):
        return padded_frames.fv_aligned_padded_frame(self.old, cycle_case.TUCKER_RANKS, cycle_case.TT_RANKS,
                                                     cycle_case.S, self.project, oversampling=0,
                                                     seed=cycle_case.SEED, **kw)

    def test_the_simultaneous_loop_cycles(self):
        _, movements = self.run_loop(max_passes=200, tol=0.0, alternate=False)
        self.assertEqual(len(movements), 200)
        self.assertGreater(float(movements[-1]), 0.05)
        self.assertGreater(float(movements[-2]), 0.05)

    def test_the_two_colour_loop_converges(self):
        _, movements = self.run_loop(max_passes=200, tol=1e-8, alternate=True)
        self.assertLess(len(movements), 200)
        self.assertLess(max(float(m) for m in movements[-2:]), 1e-8)

if __name__ == '__main__':
    unittest.main()
