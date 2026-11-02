# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Tests for ``fv_operations.fv_padded_frame``, the frame of ``X_old`` at a zero-padded rank increase.

So far T2 of the padded-frame plan, the hint-free construction:

(a) ``old_frame=None`` reproduces ``fv_zero_point_frame`` (the coordinate cores and zero carriers bitwise, the
    Haar factors to 1e-14); (b) a valid frame at ``resize``d points; (c) the in-block cases, whose columns come
    from coordinate seeds; (d) the Tucker completion is nested; (e) a stack ``C`` equals per-element calls;
    (f) sign-gauge covariance: gauge-equivalent old frames give the same subspaces (assembled projectors).

"Valid frame" means: orthogonality residual <= 1e-14; both base-point reconstructions within 1e-13 relative of
``X_old``; the old block present bitwise as a copy (a structural fact); the template's shapes. Computed arrays
are compared with stated tolerances, never ``np.allclose`` defaults; frame claims across gauges use projectors.

numpy-only (house testing strategy); jax dispatch is in ``tests/test_dispatch.py``.
"""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.t3_conversions as t3c


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

    def test_hint_is_not_implemented_yet(self):
        with self.assertRaises(NotImplementedError):
            fvo.fv_padded_frame(self.template, self.old, self.x_old.data)

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


if __name__ == '__main__':
    unittest.main()
