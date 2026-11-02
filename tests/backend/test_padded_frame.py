# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Tests for ``fv_operations.fv_padded_frame``, the frame of ``X_old`` at a zero-padded rank increase.

T2 of the padded-frame plan, the hint-free construction:

(a) ``old_frame=None`` reproduces ``fv_zero_point_frame`` (the coordinate cores and zero carriers bitwise, the
    Haar factors to 1e-14); (b) a valid frame at ``resize``d points; (c) the in-block cases, whose columns come
    from coordinate seeds; (d) the Tucker completion is nested; (e) a stack ``C`` equals per-element calls;
    (f) sign-gauge covariance: gauge-equivalent old frames give the same subspaces (assembled projectors).

T3, the hinted construction (the pinned T3-SVD of a hint T3): (a) valid frames for gradient hints (embedded by
``tv_to_t3`` at the library frame of the padded point, at a padded frame) and a zero gradient; (b) the hint's
internal coordinates (orthonormal overlaps for gradient hints, contractions otherwise); (c) the matrix fixed point;
(d) the truncation (a working-frame hint, the target template); (e) sharing; (f) a stack ``C``; (g) structural
errors; (h) the dense per-step oracle -- every new direction of every step against its dense definition; (i)
``d = 1``; (j) an arbitrary external hint. Direction claims use hints with at least ``k`` directions per slot (a
rank-deficient hint's trailing directions are roundoff-dependent); degenerate hints are asserted valid only.

"Valid frame" means: orthogonality residual <= 1e-14; both base-point reconstructions within 1e-13 relative of
``X_old``; the old block present bitwise as a copy (a structural fact); the template's shapes. Computed arrays
are compared with stated tolerances, never ``np.allclose`` defaults; frame claims across gauges use projectors.

numpy-only (house testing strategy); jax dispatch is in ``tests/test_dispatch.py``.
"""
import math
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.tv_operations as tvo


def frame_of(x):
    return fvc.t3_orthogonal_representations(x.data)[0]


def padded_case(shape, old_ranks, new_ranks, stack_shape=(), sharing=None, seed=0):
    """(x_old, old frame, template): x_old random at old_ranks, the template the frame of its resize."""
    np.random.seed(seed)
    x_old = t3.TuckerTensorTrain.randn(shape, *old_ranks, stack_shape=stack_shape)
    if sharing is not None:
        x_old = x_old.share(sharing)
    x_new = x_old.resize(shape, *new_ranks, sharing=sharing)
    return x_old, frame_of(x_old), frame_of(x_new)


def take(frame, idx):
    """The stack element ``idx`` of a stacked frame."""
    return tuple(tuple(c[idx] for c in cores) for cores in frame)


def stores(frame):
    up, down, left, right = frame
    return {'nU': [u.shape[-2] for u in up], 'nD': [o.shape[-2] for o in down],
            'rL': [g.shape[-3] for g in left] + [left[-1].shape[-1]],
            'rR': [g.shape[-3] for g in right] + [right[-1].shape[-1]]}


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


class PaddedFrameAsserts(unittest.TestCase):

    def assert_valid(self, padded, old, template, X_old):
        """Valid padded frame of X_old (unstacked)."""
        for got, want in zip(padded, template):
            self.assertEqual([g.shape for g in got], [w.shape for w in want])
        self.assertLessEqual(float(fvo.fv_frame_orthogonality_residual(padded)), 1e-14)
        up, down, left, right = padded
        nX = np.linalg.norm(X_old)
        for tt in (left, right):
            self.assertLessEqual(np.linalg.norm(t3c.t3_to_dense((up, tt)) - X_old), 1e-13 * nX)
        U0, O0, L0, R0 = old
        for i in range(len(up)):                         # the old block, embedded as a copy (structural)
            self.assertTrue(np.array_equal(up[i][:U0[i].shape[0]], U0[i]))
            self.assertTrue(np.array_equal(down[i][tuple(slice(0, n) for n in O0[i].shape)], O0[i]))
            self.assertTrue(np.array_equal(left[i][tuple(slice(0, n) for n in L0[i].shape)], L0[i]))
            self.assertTrue(np.array_equal(right[i][tuple(slice(0, n) for n in R0[i].shape)], R0[i]))
            # zero elsewhere in the old columns / rows (structural)
            a, n, b = L0[i].shape
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


class TestZeroPointIdentity(PaddedFrameAsserts):
    """T2(a): ``old_frame=None`` is ``fv_zero_point_frame`` -- coordinate cores bitwise, Haar to 1e-14."""

    CASES = [
        ((7,), (3,), (1, 1), ()),
        ((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), ()),
        ((6, 6, 7), (3, 3, 2), (1, 3, 2, 1), ((0, 1), (2,))),
        ((4, 5, 6, 3), (2, 3, 3, 2), (1, 2, 3, 2, 1), ()),
        ((5, 5, 5, 4), (3, 3, 3, 2), (1, 3, 3, 2, 1), ((0, 2), (1,), (3,))),
    ]

    def test_identity(self):
        for shape, tucker, tt, groups in self.CASES:
            with self.subTest(shape=shape, tucker=tucker, tt=tt, groups=groups):
                template = frame_of(t3.TuckerTensorTrain.zeros(shape, tucker, tt))
                zero = fvo.fv_zero_point_frame(template, groups=groups)
                for seed in (0, 5):
                    padded = fvo.fv_padded_frame(template, None, None, groups=groups, seed=seed)
                    zero = fvo.fv_zero_point_frame(template, groups=groups, seed=seed)
                    for got, want in zip(padded[1:], zero[1:]):     # coordinate cores, zero carriers: bitwise
                        for g, w in zip(got, want):
                            self.assertTrue(np.array_equal(g, w))
                    for g, w in zip(padded[0], zero[0]):             # the Haar factors: computed
                        self.assertLessEqual(np.max(np.abs(g - w)), 1e-14)
                    for group in groups:                             # one array per group
                        self.assertTrue(all(padded[0][i] is padded[0][group[0]] for i in group))

    def test_stacked_template_broadcasts(self):
        """A stacked template gives the zero-point frame broadcast over C (fv_zero_point_frame itself refuses
        stacks)."""
        template = frame_of(t3.TuckerTensorTrain.zeros((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2, 3)))
        padded = fvo.fv_padded_frame(template)
        zero = fvo.fv_zero_point_frame(take(template, (0, 0)))
        for idx in np.ndindex(2, 3):
            for got, want in zip(take(padded, idx), zero):
                for g, w in zip(got, want):
                    self.assertLessEqual(np.max(np.abs(g - w)), 1e-14)


class TestValidAtResizedPoints(PaddedFrameAsserts):
    """T2(b): valid frames at Tucker-only, TT-only and joint increases (d = 1, 3, 4)."""

    CASES = {
        'tucker only':    ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 2), (1, 2, 2, 1))),
        'tt only':        ((5, 6, 7), ((3, 3, 3), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 3, 1))),
        'joint':          ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))),
        'joint d=4':      ((4, 5, 6, 3), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 3, 4, 3), (1, 3, 4, 3, 1))),
        'd=1':            ((7,), ((2,), (1, 1)), ((5,), (1, 1))),
        'no increase':    ((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 3, 2), (1, 2, 2, 1))),
    }

    def test_valid(self):
        for name, (shape, old_ranks, new_ranks) in self.CASES.items():
            with self.subTest(case=name):
                x_old, old, template = padded_case(shape, old_ranks, new_ranks)
                padded = fvo.fv_padded_frame(template, old)
                self.assert_valid(padded, old, template, x_old.to_dense())

    def test_no_increase_returns_the_old_frame(self):
        x_old, old, template = padded_case((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 3, 2), (1, 2, 2, 1)))
        for got, want in zip(fvo.fv_padded_frame(template, old), old):
            self.assertTrue(all(np.array_equal(g, w) for g, w in zip(got, want)))

    def test_sharing(self):
        """A tied old frame: one factor per group (the same array), still a valid frame of X_old."""
        groups = ((0, 1), (2,))
        x_old, old, template = padded_case((6, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((5, 5, 3), (1, 4, 3, 1)),
                                           sharing=(0, 0, 1))
        self.assertEqual(stores(old)['nU'][:2], [4, 4])                  # share() ties at the group rank 4
        self.assertTrue(np.array_equal(old[0][0], old[0][1]))           # TIED: the caller's precondition
        padded = fvo.fv_padded_frame(template, old, groups=groups)
        self.assertIs(padded[0][0], padded[0][1])
        self.assert_valid(padded, old, template, x_old.to_dense())


class TestInBlock(PaddedFrameAsserts):
    """T2(c): a bond that grows while its neighbours do not completes inside the old block, from coordinate
    seeds in index order."""

    def in_block_columns(self, old_C_xyc, count_outside, count):
        """The expected in-block columns: the old columns completed from the block's coordinate seeds."""
        q1, q2, c = old_C_xyc.shape
        k_in = count - count_outside
        return linalg.pinned_completion(old_C_xyc.reshape(q1 * q2, c), np.eye(q1 * q2)[:, :k_in], k_in)

    def test_tt_only_increase_inside_the_block(self):
        # d = 4: bond 2 grows (2 -> 3) while rL_1 and nU_1 stay -> L_1 has no outside coordinate
        x_old, old, template = padded_case((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
                                           ((2, 2, 2, 2), (1, 2, 3, 2, 1)))
        padded = fvo.fv_padded_frame(template, old)
        self.assert_valid(padded, old, template, x_old.to_dense())
        L0, L = old[2][1], padded[2][1]                                  # (2, 2, 2) -> (2, 2, 3)
        self.assertEqual((L0.shape, L.shape), ((2, 2, 2), (2, 2, 3)))
        expected = self.in_block_columns(L0, 0, 1)
        self.assertLessEqual(np.max(np.abs(L[:, :, 2:].reshape(4, 1) - expected)), 1e-14)
        again = fvo.fv_padded_frame(template, old)
        for got, want in zip(again, padded):
            self.assertTrue(all(np.max(np.abs(g - w)) <= 1e-15 for g, w in zip(got, want)))

    def test_tt_only_increase_at_bond_1(self):
        # a non-minimal old point (Tucker 3 > bond 2 at mode 0): bond 1 grows inside the (1, 3) block
        x_old, old, template = padded_case((5, 6, 7), ((3, 3, 3), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 2, 1)))
        padded = fvo.fv_padded_frame(template, old)
        self.assert_valid(padded, old, template, x_old.to_dense())
        L0, L = old[2][0], padded[2][0]
        self.assertEqual((L0.shape, L.shape), ((1, 3, 2), (1, 3, 3)))
        self.assertLessEqual(np.max(np.abs(L[:, :, 2:].reshape(3, 1) - self.in_block_columns(L0, 0, 1))), 1e-14)

    def test_tucker_only_increase_at_saturated_bonds(self):
        # the middle mode's Tucker rank grows (3 -> 4) at bonds (2, 2): the down core gains a row inside
        # the (2, 2) bond block; L_1 / R_1 gain no columns / rows (their bonds are fixed)
        x_old, old, template = padded_case((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 4, 2), (1, 2, 2, 1)))
        padded = fvo.fv_padded_frame(template, old)
        self.assert_valid(padded, old, template, x_old.to_dense())
        O0, O = old[1][1], padded[1][1]
        self.assertEqual((O0.shape, O.shape), ((2, 3, 2), (2, 4, 2)))
        expected = self.in_block_columns(np.moveaxis(O0, 1, 2), 0, 1)
        self.assertLessEqual(np.max(np.abs(O[:, 3, :].reshape(4, 1) - expected)), 1e-14)

    def test_outside_first_then_in_block_counts(self):
        """The structural counts: new columns take the outside coordinates (exact unit vectors, in row-major
        order) before any in-block one."""
        x_old, old, template = padded_case((4, 5, 6, 3), ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
                                           ((3, 3, 4, 3), (1, 3, 4, 3, 1)))
        padded = fvo.fv_padded_frame(template, old)
        L0, L = old[2][1], padded[2][1]                                  # (2, 2, 2) -> (3, 3, 4)
        a, n, b = L0.shape
        outside = [(x, y) for x in range(L.shape[0]) for y in range(L.shape[1]) if not (x < a and y < n)]
        for j, (x, y) in enumerate(outside[:L.shape[2] - b]):
            unit = np.zeros(L.shape[:2]); unit[x, y] = 1.0
            self.assertTrue(np.array_equal(L[:, :, b + j], unit))


class TestNestedAndStacked(PaddedFrameAsserts):

    def test_tucker_completion_is_nested(self):
        """T2(d): kU -> kU + 1 at one mode: the leading rows agree to 1e-14; other modes are untouched."""
        shape, old_ranks = (5, 6, 7), ((2, 2, 2), (1, 2, 2, 1))
        _, old, template3 = padded_case(shape, old_ranks, ((3, 3, 3), (1, 3, 3, 1)))
        _, old_b, template4 = padded_case(shape, old_ranks, ((3, 4, 3), (1, 3, 3, 1)))
        up3 = fvo.fv_padded_frame(template3, old)[0]
        up4 = fvo.fv_padded_frame(template4, old_b)[0]
        self.assertLessEqual(np.max(np.abs(up4[1][:3] - up3[1])), 1e-14)
        self.assertTrue(np.array_equal(up4[0], up3[0]) and np.array_equal(up4[2], up3[2]))

    def test_stack_equals_per_element(self):
        """T2(e): a C = (2,) and (2, 3) stack equals per-element calls to 1e-14 (an in-block case included)."""
        cases = [((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))),
                 ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 2), (1, 2, 3, 2, 1)))]
        for C in [(2,), (2, 3)]:
            for shape, old_ranks, new_ranks in cases:
                with self.subTest(C=C, shape=shape):
                    x_old, old, template = padded_case(shape, old_ranks, new_ranks, stack_shape=C)
                    padded = fvo.fv_padded_frame(template, old)
                    for idx in np.ndindex(*C):
                        single = fvo.fv_padded_frame(take(template, idx), take(old, idx))
                        for got, want in zip(take(padded, idx), single):
                            for g, w in zip(got, want):
                                self.assertLessEqual(np.max(np.abs(g - w)), 1e-14)
                        self.assert_valid(single, take(old, idx), take(template, idx), x_old.to_dense()[idx])


class TestSignGaugeCovariance(unittest.TestCase):
    """T2(f): two sign-gauge-equivalent old frames give hint-free frames with equal assembled projectors."""

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

    def test_covariance(self):
        rng = np.random.default_rng(0)
        cases = {
            'joint':             ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))),
            'in-block TT':       ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 2), (1, 2, 3, 2, 1))),
            'in-block down':     ((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 4, 2), (1, 2, 2, 1))),
        }
        for name, (shape, old_ranks, new_ranks) in cases.items():
            x_old, old, template = padded_case(shape, old_ranks, new_ranks)
            for trial in range(3):
                with self.subTest(case=name, trial=trial):
                    gauged = self.sign_gauge(old, rng)
                    self.assertLessEqual(float(fvo.fv_frame_orthogonality_residual(gauged)), 1e-14)
                    a = projectors(fvo.fv_padded_frame(template, old))
                    b = projectors(fvo.fv_padded_frame(template, gauged))
                    self.assertLessEqual(max(np.linalg.norm(p - q) for p, q in zip(a, b)), 1e-12)


class TestStructuralErrors(unittest.TestCase):

    def setUp(self):
        self.x_old, self.old, self.template = padded_case((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)),
                                                          ((3, 4, 3), (1, 3, 3, 1)))

    def test_errors(self):
        up, down, left, right = self.template
        bigger = frame_of(t3.TuckerTensorTrain.zeros((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)))
        other_N = frame_of(t3.TuckerTensorTrain.randn((5, 6, 8), (2, 2, 2), (1, 2, 2, 1)))
        stacked = frame_of(t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1), stack_shape=(2,)))
        d2 = frame_of(t3.TuckerTensorTrain.randn((5, 6), (2, 2), (1, 2, 1)))
        cases = {
            'old above the template': (bigger, self.template, (), 'exceeds the template'),
            'mode sizes':             (self.template, other_N, (), 'mode sizes'),
            'stack':                  (self.template, stacked, (), 'stack'),
            'd':                      (self.template, d2, (), 'modes'),
            'inconsistent template':  ((up, down, (left[0][:, :2],) + left[1:], right), None, (), 'left\\[0\\]'),
            'inadmissible template':  (((np.zeros((3, 7)),), (np.zeros((1, 3, 1)),), (np.zeros((1, 3, 1)),),
                                        (np.zeros((1, 3, 1)),)), None, (), 'admit no'),   # nD = 3 > rL rR = 1
            'group shapes':           (self.template, self.old, ((0, 1),), 'sharing group'),
            'group index':            (self.template, self.old, ((0, 3),), 'outside'),
        }
        for name, (template, old, groups, msg) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, msg):
                fvo.fv_padded_frame(template, old, groups=groups)

    def test_template_values_are_not_read(self):
        """Only the template's shapes and dtype are read (the fv_zero_point_frame convention)."""
        noisy = tuple(tuple(np.random.randn(*c.shape) for c in cores) for cores in self.template)
        for got, want in zip(fvo.fv_padded_frame(noisy, self.old), fvo.fv_padded_frame(self.template, self.old)):
            self.assertTrue(all(np.array_equal(g, w) for g, w in zip(got, want)))


# ---------------------------------------------------------------------------------------------------- T3: hinted
def interfaces(frame):
    """Dense left interfaces Phi[k] (prod N_<k, rL_k) and right interfaces Psi[k] (prod N_>=k, rR_k), k = 0..d."""
    up, down, left, right = frame
    d = len(up)
    phi = [np.ones((1, left[0].shape[0]))]
    for k in range(d):
        phi.append(np.einsum('xa,ni,anb->xib', phi[k], up[k], left[k])
                   .reshape(phi[k].shape[0] * up[k].shape[1], left[k].shape[2]))
    psi = [None] * d + [np.ones((1, right[-1].shape[2]))]
    for k in range(d - 1, -1, -1):
        psi[k] = np.einsum('anb,ni,yb->aiy', right[k], up[k], psi[k + 1]) \
            .reshape(right[k].shape[0], up[k].shape[1] * psi[k + 1].shape[0]).T
    return phi, psi


def embed(a, shape):
    out = np.zeros(shape)
    out[tuple(slice(0, n) for n in a.shape)] = a
    return out


def leading(P, M, count, scale):
    """The leading left singular directions of (I - P P^T) M that are DETERMINED: the first r <= count, each with
    singular value above 1e-8 * scale and separated from the next by 1e-6 * scale. Returns (basis, r)."""
    A = M - P @ (P.T @ M)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    r = min(count, int(np.sum(s > 1e-8 * scale)))
    while 0 < r < s.size and s[r - 1] - s[r] < 1e-6 * scale:
        r -= 1
    return U[:, :r], r


def subspace_distance(A, B):
    return np.linalg.norm(A @ A.T - B @ B.T)


def oracle_distances(frame, old, Z, groups=()):
    """T3(h), the dense per-step oracle (unstacked): for every step, (projector distance between the frame's first
    r new directions and the oracle's leading r, r, the directions the content can supply: min(slot count, the
    oracle matrix's columns)). Z is the dense hint.

    Tucker i (per group): (I - P_{U°}) [Z_(i) (I - E^T E) for the members], E[m] = sum_ab O°[a, m, b] Phi°_i[a] (x)
    Psi°_{i+1}[b]; L_k: (I - P P^T) M (I - Psi°_{k+1} Psi°_{k+1}^T), M = (Phi_k (x) U_k)^T Z_(..k | k+1..), P the
    embedded L°_k; R_k: the mirror image; O_i: Z's coordinates in Phi_i (x) U_i (x) Psi_{i+1} on the new Tucker
    slots, in the (a, b) bond space, pinned off the embedded O°_i."""
    up, down, left, right = frame
    U0, O0, L0, R0 = old
    d, N, scale = len(up), Z.shape, np.linalg.norm(Z)
    phi, psi = interfaces(frame)
    phio, psio = interfaces(old)
    out = {}
    key = list(range(d))
    for group in groups:
        for i in group:
            key[i] = min(group)
    for kg in sorted(set(key)):
        blocks = []
        for i in [j for j in range(d) if key[j] == kg]:
            Zi = np.moveaxis(Z, i, 0).reshape(N[i], -1)
            E = np.einsum('amb,xa,yb->mxy', O0[i], phio[i], psio[i + 1]).reshape(O0[i].shape[1], Zi.shape[1])
            blocks.append(Zi - (Zi @ E.T) @ E)
        s, count = U0[kg].shape[0], up[kg].shape[0] - U0[kg].shape[0]
        M = np.concatenate(blocks, axis=1)
        Q, r = leading(U0[kg].T, M, count, scale)
        out['U%d' % kg] = (subspace_distance(up[kg][s:s + r].T, Q), r, min(count, M.shape[1]))
    for k in range(d - 1):
        Zk = Z.reshape(math.prod(N[:k]), N[k], -1)
        a, n, b = left[k].shape
        M = np.einsum('xa,ni,xiy->any', phi[k], up[k], Zk).reshape(a * n, Zk.shape[2])
        M = M - (M @ psio[k + 1]) @ psio[k + 1].T
        sL1 = L0[k].shape[2]
        P = embed(L0[k], (a, n, sL1)).reshape(a * n, sL1)
        Q, r = leading(P, M, b - sL1, scale)
        out['L%d' % k] = (subspace_distance(left[k].reshape(a * n, b)[:, sL1:sL1 + r], Q), r,
                          min(b - sL1, M.shape[1]))
    for k in range(1, d):
        Zk = Z.reshape(math.prod(N[:k]), N[k], -1)
        a, n, b = right[k].shape
        M = np.einsum('ni,yb,xiy->nbx', up[k], psi[k + 1], Zk).reshape(n * b, Zk.shape[0])
        M = M - (M @ phio[k]) @ phio[k].T
        sR = R0[k].shape[0]
        P = embed(R0[k], (sR, n, b)).reshape(sR, n * b).T
        Q, r = leading(P, M, a - sR, scale)
        out['R%d' % k] = (subspace_distance(right[k].reshape(a, n * b)[sR:sR + r].T, Q), r, min(a - sR, M.shape[1]))
    for i in range(d):
        Zi = Z.reshape(math.prod(N[:i]), N[i], -1)
        core = np.einsum('xa,ni,yb,xiy->anb', phi[i], up[i], psi[i + 1], Zi)
        s = U0[i].shape[0]
        a, m, b = down[i].shape
        A = np.moveaxis(core[:, s:, :], 1, 2).reshape(a * b, core.shape[1] - s)
        sD = O0[i].shape[1]
        P = np.moveaxis(embed(O0[i], (a, sD, b)), 1, 2).reshape(a * b, sD)
        Q, r = leading(P, A, m - sD, scale)
        new = np.moveaxis(down[i], 1, 2).reshape(a * b, m)[:, sD:sD + r]
        out['O%d' % i] = (subspace_distance(new, Q), r, min(m - sD, A.shape[1]))
    return out


def gradient_hint(frame, G):
    """The driver's hint shape: the projection of a dense G onto the frame's tangent space, embedded as a T3."""
    return tvo.tv_to_t3(frame, tvo.tv_project_dense_onto_tangent_space(frame, G))


def captured_norm(frame, Y):
    """||P_{T_F} Y|| for an orthonormal frame: the norm of the gauged coordinates."""
    tk, tt = tvo.tv_project_dense_onto_tangent_space(frame, Y)
    return math.sqrt(sum(float(np.sum(v * v)) for v in tk + tt))


def rich_hint(shape, new_ranks, stack_shape=()):
    """A random T3 above the target ranks (Tucker +1, interior bonds +1, capped at the mode sizes / 4): at least
    as many directions as every slot of the target."""
    tucker = tuple(min(N, n + 1) for N, n in zip(shape, new_ranks[0]))
    tt = tuple(r if j in (0, len(shape)) else min(r + 1, 4) for j, r in enumerate(new_ranks[1]))
    return t3.TuckerTensorTrain.randn(shape, tucker, tt, stack_shape=stack_shape)


HINTED_CASES = {
    'joint d=3':     ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))),
    'joint d=4':     ((4, 5, 6, 3), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 3, 4, 3), (1, 3, 4, 3, 1))),
    'in-block d=4':  ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 3), (1, 2, 3, 2, 1))),
    'tucker only':   ((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((3, 4, 2), (1, 2, 2, 1))),
}


class TestHintedValid(PaddedFrameAsserts):
    """T3(a), (d), (i): valid padded frames for gradient hints, a zero gradient, the truncation, and d = 1."""

    def test_gradient_hints(self):
        rng = np.random.default_rng(0)
        for name, (shape, old_ranks, new_ranks) in HINTED_CASES.items():
            x_old, old, template = padded_case(shape, old_ranks, new_ranks)
            X_old = x_old.to_dense()
            hint_free = fvo.fv_padded_frame(template, old)
            G = rng.standard_normal(shape)
            hints = {'at the library frame': gradient_hint(template, G),
                     'at a padded frame':    gradient_hint(hint_free, G),
                     'zero gradient':        gradient_hint(template, np.zeros(shape))}
            for kind, hint in hints.items():
                with self.subTest(case=name, hint=kind):
                    self.assert_valid(fvo.fv_padded_frame(template, old, hint), old, template, X_old)

    def test_truncation(self):
        """A hint embedded at the WORKING (oversampled) frame, the TARGET template: a valid target frame."""
        shape, old_ranks = (6, 7, 8), ((2, 2, 2), (1, 2, 2, 1))
        x_old, old, target = padded_case(shape, old_ranks, ((3, 3, 3), (1, 3, 3, 1)))
        _, _, working = padded_case(shape, old_ranks, ((5, 5, 5), (1, 5, 5, 1)))
        F_w = fvo.fv_padded_frame(working, old)
        hint = gradient_hint(F_w, np.random.default_rng(1).standard_normal(shape))
        self.assert_valid(fvo.fv_padded_frame(target, old, hint), old, target, x_old.to_dense())

    def test_d1(self):
        x_old, old, template = padded_case((7,), ((2,), (1, 1)), ((4,), (1, 1)))
        for hint in (gradient_hint(template, np.random.default_rng(2).standard_normal(7)),
                     t3.TuckerTensorTrain.randn((7,), (3,), (1, 1)).data):
            padded = fvo.fv_padded_frame(template, old, hint)
            self.assert_valid(padded, old, template, x_old.to_dense())

    def test_external_hints_are_valid(self):
        """T3(j), validity: raw external hints -- Tucker ranks below the target's, and a T3 sum (boundary bonds 2,
        squashed by the constructor) -- at s > 0 and at s = 0."""
        for name, (shape, old_ranks, new_ranks) in HINTED_CASES.items():
            x_old, old, template = padded_case(shape, old_ranks, new_ranks)
            low = t3.TuckerTensorTrain.randn(shape, tuple(max(1, n - 1) for n in new_ranks[0]), new_ranks[1])
            total = low + t3.TuckerTensorTrain.randn(shape, *new_ranks)
            self.assertEqual((total.data[1][0].shape[0], total.data[1][-1].shape[-1]), (2, 2))
            for kind, hint in (('low rank', low.data), ('T3 sum', total.data)):
                with self.subTest(case=name, hint=kind):
                    self.assert_valid(fvo.fv_padded_frame(template, old, hint), old, template, x_old.to_dense())
                    at_zero = fvo.fv_padded_frame(template, None, hint)
                    self.assertLessEqual(float(fvo.fv_frame_orthogonality_residual(at_zero)), 1e-14)
                    self.assertEqual(float(np.abs(t3c.t3_to_dense((at_zero[0], at_zero[2]))).max()), 0.0)

    def test_rank_deficient_and_zero_hints_are_valid(self):
        """Fewer hint directions than slots (rank-1, zero): completed, still valid (directions not asserted)."""
        for name, (shape, old_ranks, new_ranks) in HINTED_CASES.items():
            x_old, old, template = padded_case(shape, old_ranks, new_ranks)
            rank1 = t3.TuckerTensorTrain.randn(shape, (1,) * len(shape), (1,) * (len(shape) + 1))
            zero = t3.TuckerTensorTrain.zeros(shape, new_ranks[0], new_ranks[1])
            for kind, hint in (('rank 1', rank1.data), ('zero', zero.data), ('zero + X_old', (zero + x_old).data)):
                with self.subTest(case=name, hint=kind):
                    self.assert_valid(fvo.fv_padded_frame(template, old, hint), old, template, x_old.to_dense())


class TestHintCoordinates(unittest.TestCase):
    """T3(b): for gradient hints the old frame's overlaps with the hint's chains have orthonormal rows (the
    exactness condition); for an arbitrary T3 hint they are contractions (norm <= 1)."""

    def test_overlaps(self):
        rng = np.random.default_rng(3)
        for name, (shape, old_ranks, new_ranks) in HINTED_CASES.items():
            x_old, old, template = padded_case(shape, old_ranks, new_ranks)
            hint_free = fvo.fv_padded_frame(template, old)
            G = rng.standard_normal(shape)
            other = t3.TuckerTensorTrain.randn(shape, tuple(max(1, n - 1) for n in new_ranks[0]), new_ranks[1]).data
            for kind, hint, exact in (('library frame', gradient_hint(template, G), True),
                                      ('padded frame', gradient_hint(hint_free, G), True),
                                      ('arbitrary T3', other, False)):
                with self.subTest(case=name, hint=kind):
                    _, _, _, _, A, OmL, OmR, _ = fvo._hint_coordinates(hint, old[0], old[2], old[3])
                    for X in A + OmL[:-1] + OmR[1:]:        # the interfaces (the boundary zippers are scalars)
                        if exact:
                            self.assertLessEqual(np.linalg.norm(X @ X.T - np.eye(X.shape[0])), 1e-13)
                        else:
                            self.assertLessEqual(np.linalg.norm(X, 2) if X.size else 0.0, 1 + 1e-13)


class TestDenseOracle(unittest.TestCase):
    """T3(h): every step's new directions against their dense definition (projector distance < 1e-9), on hints with
    at least k directions per slot -- unshared and shared, d = 3 and 4, C = () and (2,)."""

    def check(self, frame, old, Z, groups=(), require_full=True):
        res = oracle_distances(frame, old, Z, groups)
        for step, (dist, r, count) in res.items():
            with self.subTest(step=step):
                if require_full:
                    self.assertEqual(r, count, 'the oracle determines %d of %d directions of %s' % (r, count, step))
                self.assertLess(dist, 1e-9)
        return res

    def test_gradient_hints(self):
        """Gradient hints contain X_old's interfaces (exact); degenerate slots are compared where determined."""
        rng = np.random.default_rng(4)
        for name, (shape, old_ranks, new_ranks) in HINTED_CASES.items():
            x_old, old, template = padded_case(shape, old_ranks, new_ranks)
            hint = gradient_hint(template, rng.standard_normal(shape))
            with self.subTest(case=name):
                res = self.check(fvo.fv_padded_frame(template, old, hint), old, t3c.t3_to_dense(hint),
                                 require_full=False)
                self.assertGreater(sum(r for _, r, _ in res.values()), 0)

    def test_external_hint_plus_x_old(self):
        """An arbitrary T3 H made exact by adding X_old (T3(j)): the oracle with Z = H, every direction."""
        for name, (shape, old_ranks, new_ranks) in HINTED_CASES.items():
            x_old, old, template = padded_case(shape, old_ranks, new_ranks, seed=5)
            H = rich_hint(shape, new_ranks)
            with self.subTest(case=name):
                self.check(fvo.fv_padded_frame(template, old, (H + x_old).data), old, H.to_dense())

    def test_zero_tensor(self):
        """s = 0 (old_frame=None): the pinned T3-SVD of H itself, every direction."""
        for name, (shape, _, new_ranks) in HINTED_CASES.items():
            template = frame_of(t3.TuckerTensorTrain.zeros(shape, *new_ranks))
            H = rich_hint(shape, new_ranks)
            empty = fvo._empty_old_frame(shape, (), np.float64, np)
            with self.subTest(case=name):
                self.check(fvo.fv_padded_frame(template, None, H.data), empty, H.to_dense())

    def test_shared(self):
        """T3(e): one U per group (the same array), and the group's new rows span the top eigenvectors of the
        members' summed projected Grams."""
        groups = ((0, 1), (2,))
        x_old, old, template = padded_case((6, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((5, 5, 3), (1, 4, 3, 1)),
                                           sharing=(0, 0, 1))
        H = rich_hint((6, 6, 7), ((5, 5, 3), (1, 4, 3, 1)))
        padded = fvo.fv_padded_frame(template, old, (H + x_old).data, groups=groups)
        self.assertIs(padded[0][0], padded[0][1])
        PaddedFrameAsserts.assert_valid(self, padded, old, template, x_old.to_dense())
        self.check(padded, old, H.to_dense(), groups)

    def test_stacked(self):
        """C = (2,): each element of the stacked call passes the oracle (and equals the per-element call, T3(f))."""
        for name in ('joint d=3', 'joint d=4'):
            shape, old_ranks, new_ranks = HINTED_CASES[name]
            x_old, old, template = padded_case(shape, old_ranks, new_ranks, stack_shape=(2,))
            H = rich_hint(shape, new_ranks, stack_shape=(2,))
            padded = fvo.fv_padded_frame(template, old, (H + x_old).data)
            for e in range(2):
                with self.subTest(case=name, element=e):
                    single = fvo.fv_padded_frame(take(template, e), take(old, e), take2((H + x_old).data, e))
                    for got, want in zip(take(padded, e), single):
                        for g, w in zip(got, want):
                            self.assertLessEqual(np.max(np.abs(g - w)), 1e-12)
                    self.check(single, take(old, e), H.to_dense()[e])


def take2(t3_data, idx):
    """The stack element ``idx`` of a stacked T3 .data pair."""
    return tuple(tuple(c[idx] for c in cores) for cores in t3_data)


class TestMatrixCase(unittest.TestCase):
    """d = 2 with Tucker ranks = TT rank: the frame tangent space is the matrix one (T3(c), T3(j))."""

    shape = (6, 7)

    def setUp(self):
        self.x_old, self.old, self.template = padded_case(self.shape, ((1, 1), (1, 1, 1)), ((3, 3), (1, 3, 1)), seed=6)
        self.u, self.v = self.old[0][0], self.old[0][1]                  # X_old's row / column spaces (1 x N)

    def normal_part(self, Y):
        Pu, Pv = np.eye(6) - self.u.T @ self.u, np.eye(7) - self.v.T @ self.v
        return Pu @ Y @ Pv

    def test_external_hint_top_singular_subspaces(self):
        """With H + X_old the new slots are the top-2 singular subspaces of P_u^perp H P_v^perp (< 1e-10)."""
        H = t3.TuckerTensorTrain.randn(self.shape, (4, 5), (1, 4, 1))
        up = fvo.fv_padded_frame(self.template, self.old, (H + self.x_old).data)[0]
        Ul, _, Vt = np.linalg.svd(self.normal_part(H.to_dense()))
        self.assertLess(subspace_distance(up[0][1:].T, Ul[:, :2]), 1e-10)
        self.assertLess(subspace_distance(up[1][1:].T, Vt[:2].T), 1e-10)

    def test_raw_external_hint_captures_at_least_the_library_frame(self):
        """With the raw H (the caveat case) the frame is valid and captures at least the library frame's share."""
        H = t3.TuckerTensorTrain.randn(self.shape, (4, 5), (1, 4, 1))
        padded = fvo.fv_padded_frame(self.template, self.old, H.data)
        PaddedFrameAsserts.assert_valid(self, padded, self.old, self.template, self.x_old.to_dense())
        Hd = H.to_dense()
        self.assertGreaterEqual(captured_norm(padded, Hd), captured_norm(self.template, Hd))

    def test_fixed_point(self):
        """T3(c): new slots aligned with the top singular subspaces of N, hint P_T G at that frame: the
        constructor returns the same subspaces (< 1e-10)."""
        G = np.random.default_rng(7).standard_normal(self.shape)
        Ul, _, Vt = np.linalg.svd(self.normal_part(G))
        up, down, left, right = fvo.fv_padded_frame(self.template, self.old)
        aligned = ((np.concatenate([self.u, Ul[:, :2].T]), np.concatenate([self.v, Vt[:2]])), down, left, right)
        self.assertLessEqual(float(fvo.fv_frame_orthogonality_residual(aligned)), 1e-14)
        up2 = fvo.fv_padded_frame(self.template, self.old, gradient_hint(aligned, G))[0]
        self.assertLess(subspace_distance(up2[0][1:].T, Ul[:, :2]), 1e-10)
        self.assertLess(subspace_distance(up2[1][1:].T, Vt[:2].T), 1e-10)


class TestHintStructuralErrors(unittest.TestCase):
    """T3(g): each a ValueError naming the store."""

    def test_errors(self):
        x_old, old, template = padded_case((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1)))
        Ht, Hg = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1)).data
        stacked = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), stack_shape=(3,)).data
        other_N = t3.TuckerTensorTrain.randn((5, 6, 8), (2, 3, 2), (1, 2, 2, 1)).data
        cases = {
            'mode sizes':        (other_N, 'mode size'),
            'bond mismatch':     ((Ht, (Hg[0], Hg[1][:, :, :1], Hg[2])), 'disagree on bond'),
            'tucker mismatch':   ((Ht, (Hg[0], Hg[1][:, :2], Hg[2])), 'Tucker rank'),
            'another stack':     (stacked, "caller's"),
            'too few cores':     ((Ht[:2], Hg[:2]), 'd = 3'),
            'not a pair':        ((Ht,), 'pair'),
        }
        for name, (hint, msg) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, msg):
                fvo.fv_padded_frame(template, old, hint)

if __name__ == '__main__':
    unittest.main()
