# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The exact Riemannian Hessian's building blocks, against dense ground truth and the oracles of
``_curvature_oracles.py`` (numpy-only; jax dispatch is in ``tests/test_dispatch.py``).

* **T1 factors** -- ``fv_center_factors`` / ``fv_lift_factors``: the centers are the construction's, the
  bond and Tucker factors are ``<L_i, H_i>`` and ``V_i U_i^T``, their spectra are the dense unfoldings',
  and ``sum sigma(S_0)^2 = ||X||^2`` (so ``eps = curvature_damping * ||X||``).
* **T2 lift identity** -- ``tv_to_dense(t3_corewise_frame(U, L), L v) == tv_to_dense(frame, v)`` at
  ``eps = 0``, gauged and ungauged, on minimal, Tucker-slack and TT-slack frames.
* **T3 lift adjoint** -- ``<L v, y> == <v, L^T y>`` per ``K + C`` element, at several dampings.
* **T4 damping** -- ``eps -> 0`` continuity; finite at a zero-padded restart; zero at the zero point
  except the last TT slot; the filters themselves.

Stacks: none, ``C``, ``K``, ``K + C`` and multi-axis ``K = (2, 2)``, ``C = (2, 3)``. Per-test seeded.
"""
import os
import sys
import unittest
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.corewise as cw
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.sharing as sharing
import t3toolbox.backend.tv_operations as tv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _curvature_oracles as oracles

_MINIMAL = ((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
_TUCKER_SLACK = ((6, 6, 6), (4, 4, 4), (1, 2, 2, 1))        # frame down ranks (2, 4, 2) < up (4, 4, 4)
_TT_SLACK = ((6, 6, 6), (4, 4, 4), (1, 9, 9, 1))            # frame left ranks != right ranks
_D1 = ((5,), (3,), (1, 1))
_D2 = ((5, 6), (3, 3), (1, 3, 1))
_D4 = ((4, 5, 4, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1))
_RANK1 = ((5, 6, 7), (1, 1, 1), (1, 1, 1, 1))
_MAXIMAL = ((2, 3, 2), (2, 3, 2), (1, 2, 2, 1))             # every slot at maximal rank

_STRUCTURES = {'minimal': _MINIMAL, 'tucker_slack': _TUCKER_SLACK, 'tt_slack': _TT_SLACK, 'd1': _D1,
               'd2': _D2, 'd4': _D4, 'rank1': _RANK1, 'maximal': _MAXIMAL}
_STACKS = [((), ()), ((), (2,)), ((3,), ()), ((3,), (2,)), ((2, 2), (2, 3))]   # (K, C)


def _elem(tree, idx):
    """Slice every array leaf of a nested tuple at the leading index ``idx``."""
    if isinstance(tree, (tuple, list)):
        return tuple(_elem(t, idx) for t in tree)
    return tree[idx]


def _relerr_per_element(a, b, n_stack):
    """max over the leading ``n_stack`` axes of ||a - b|| / ||b|| (trailing axes reduced)."""
    a, b = np.asarray(a), np.asarray(b)
    axes = tuple(range(n_stack, b.ndim))
    num = np.sqrt(np.sum((a - b) ** 2, axis=axes))
    den = np.maximum(np.sqrt(np.sum(b ** 2, axis=axes)), 1e-300)
    return float(np.max(num / den))


def _point_and_frame(struct, C):
    x = t3.TuckerTensorTrain.randn(*struct, stack_shape=C)
    frame, variations = fvc.t3_orthogonal_representations(x.data)
    return x, frame, variations


def _random_variations(frame, KC):
    return fvo.fv_variations_randn(fvo.fv_variation_shapes(frame), KC)


def _raw_like_frame_cores(frame, KC, rng):
    """A random raw core variation shaped like theta = (U, L), stack K + C."""
    U, _, L, _ = frame
    return (tuple(rng.standard_normal(KC + u.shape[-2:]) for u in U),
            tuple(rng.standard_normal(KC + g.shape[-3:]) for g in L))


class TestFrameFactors(unittest.TestCase):
    """T1: the center factors and the lift factors' spectra / damping scale."""

    def setUp(self):
        np.random.seed(1)

    def test_factors_match_the_construction_and_the_dense_spectra(self):
        for name in ('minimal', 'tucker_slack', 'tt_slack', 'd1', 'd2', 'd4', 'rank1', 'maximal'):
            struct = _STRUCTURES[name]
            for C in [(), (2,), (2, 3)]:
                with self.subTest(structure=name, C=C):
                    x, frame, variations = _point_and_frame(struct, C)
                    U, O, L, R = frame
                    V, H = variations
                    d = len(U)
                    centers, STs, Ts = fvo.fv_center_factors(frame)
                    self.assertEqual((len(centers), len(STs), len(Ts)), (d, d, d - 1))
                    for i in range(d):
                        self.assertEqual(STs[i].shape, C + (O[i].shape[-2], U[i].shape[-2]))
                        scale = max(np.linalg.norm(H[i]), 1.0)
                        self.assertLess(np.linalg.norm(centers[i] - H[i]), 1e-12 * scale)
                        VUt = np.einsum('...xn,...un->...xu', V[i], U[i])          # S_i^T = V_i U_i^T
                        self.assertLess(np.linalg.norm(STs[i] - VUt), 1e-12 * scale)
                    for i in range(d - 1):
                        self.assertEqual(Ts[i].shape, C + (L[i].shape[-1], R[i + 1].shape[-3]))
                        LH = np.einsum('...abi,...abj->...ij', L[i], H[i])          # T_i = <L_i, H_i>
                        self.assertLess(np.linalg.norm(Ts[i] - LH), 1e-12 * max(np.linalg.norm(LH), 1.0))

                    cd = 1e-3
                    lf = fvo.fv_lift_factors(frame, cd)
                    X = np.asarray(x.to_dense())
                    for c in np.ndindex(*C):
                        Xc = X[c]
                        nrm2 = float(np.sum(Xc ** 2))
                        self.assertAlmostEqual(float(np.sum(lf.tucker_svals[0][c] ** 2)) / nrm2, 1.0, delta=1e-12)
                        self.assertAlmostEqual(float(lf.eps[c]), cd * np.sqrt(nrm2), delta=1e-12 * np.sqrt(nrm2))
                        for i in range(d):
                            unf = np.moveaxis(Xc, i, 0).reshape(Xc.shape[i], -1)
                            s_dense = np.linalg.svd(unf, compute_uv=False)
                            s = lf.tucker_svals[i][c]
                            self.assertTrue(np.allclose(s, s_dense[:len(s)], atol=1e-12 * s_dense[0]))
                        for i in range(d - 1):
                            unf = Xc.reshape(int(np.prod(Xc.shape[:i + 1])), -1)
                            s_dense = np.linalg.svd(unf, compute_uv=False)
                            s = lf.bond_svals[i][c]
                            self.assertTrue(np.allclose(s, s_dense[:len(s)], atol=1e-12 * s_dense[0]))

    def test_shapes_and_the_pinvs_at_zero_damping(self):
        for name in ('minimal', 'tucker_slack', 'tt_slack', 'd1'):
            with self.subTest(structure=name):
                _, frame, _ = _point_and_frame(_STRUCTURES[name], (2,))
                _, STs, Ts = fvo.fv_center_factors(frame)
                lf = fvo.fv_lift_factors(frame, 0.0)
                self.assertEqual(lf.eps.shape, (2,))
                self.assertTrue(np.all(lf.eps == 0.0))
                for S_T, P in zip(STs, lf.tucker_pinvs):
                    self.assertEqual(P.shape, S_T.shape[:-2] + S_T.shape[:-3:-1])
                    self.assertTrue(np.allclose(P, np.linalg.pinv(S_T)))
                for T, P in zip(Ts, lf.bond_pinvs):
                    self.assertEqual(P.shape, T.shape[:-2] + T.shape[:-3:-1])
                    self.assertTrue(np.allclose(P, np.linalg.pinv(T)))

    def test_negative_damping_raises(self):
        _, frame, _ = _point_and_frame(_MINIMAL, ())
        with self.assertRaises(ValueError):
            fvo.fv_lift_factors(frame, -1e-6)


class TestLiftIdentity(unittest.TestCase):
    """T2: the lift reproduces the frame's tangent through the core parametrization, exactly at eps = 0."""

    def setUp(self):
        np.random.seed(2)

    def test_lift_identity(self):
        for name in ('minimal', 'tucker_slack', 'tt_slack', 'd1', 'd2', 'd4', 'rank1', 'maximal'):
            for K, C in _STACKS:
                with self.subTest(structure=name, K=K, C=C):
                    _, frame, _ = _point_and_frame(_STRUCTURES[name], C)
                    corewise_frame = fvc.t3_corewise_frame((frame[0], frame[2]))
                    lf = fvo.fv_lift_factors(frame, 0.0)
                    kappa = max(float(np.max(s[..., 0] / s[..., -1])) for s in lf.tucker_svals + lf.bond_svals)
                    tol = 1e-13 * max(1.0, kappa)              # the lift applies 1/sigma: roundoff * kappa
                    v_raw = _random_variations(frame, K + C)
                    v = tv.tv_orthogonal_gauge_projection(frame, v_raw)
                    for which, vv in (('gauged', v), ('ungauged', v_raw)):
                        theta_hat = tv.tv_corewise_lift(frame, vv, lf)
                        for i, (Ud, U) in enumerate(zip(theta_hat[0], frame[0])):
                            self.assertEqual(Ud.shape, K + U.shape)
                        for Gd, L in zip(theta_hat[1], frame[2]):
                            self.assertEqual(Gd.shape, K + L.shape)
                        lhs = tv.tv_to_dense(corewise_frame, theta_hat)
                        rhs = tv.tv_to_dense(frame, vv)
                        self.assertLess(_relerr_per_element(lhs, rhs, len(K + C)), tol, which)

    def test_the_last_tt_slot_passes_through(self):
        _, frame, _ = _point_and_frame(_MINIMAL, (2,))
        v = _random_variations(frame, (3, 2))
        theta_hat = tv.tv_corewise_lift(frame, v, fvo.fv_lift_factors(frame, 1e-1))
        self.assertIs(theta_hat[1][-1], v[1][-1])

    def test_mode_count_mismatch_raises(self):
        _, frame, _ = _point_and_frame(_MINIMAL, ())
        _, frame4, _ = _point_and_frame(_D4, ())
        v = _random_variations(frame, ())
        with self.assertRaises(ValueError):
            tv.tv_corewise_lift(frame, v, fvo.fv_lift_factors(frame4, 0.0))
        with self.assertRaises(ValueError):
            tv.tv_corewise_lift_transpose(frame4, v, fvo.fv_lift_factors(frame4, 0.0))


class TestLiftAdjoint(unittest.TestCase):
    """T3: L^T is the exact coordinate adjoint of L, per K + C element, for every damping."""

    def setUp(self):
        np.random.seed(3)

    def test_adjoint_identity(self):
        rng = np.random.default_rng(3)
        padded = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)).t3svd()[0].resize(
            (5, 6, 7), (3, 3, 2), (1, 3, 2, 1))
        cases = [(n, _STRUCTURES[n]) for n in ('minimal', 'tucker_slack', 'tt_slack', 'd1', 'd4')]
        for name, struct in cases + [('padded', None)]:
            for K, C in _STACKS:
                if name == 'padded' and C != ():
                    continue
                for cd in (0.0, 1e-3, 1e-1, 10.0):
                    with self.subTest(structure=name, K=K, C=C, curvature_damping=cd):
                        if name == 'padded':
                            frame, _ = fvc.t3_orthogonal_representations(padded.data)
                        else:
                            _, frame, _ = _point_and_frame(struct, C)
                        lf = fvo.fv_lift_factors(frame, cd)
                        v = _random_variations(frame, K + C)
                        y = _raw_like_frame_cores(frame, K + C, rng)
                        Lv = tv.tv_corewise_lift(frame, v, lf)
                        LTy = tv.tv_corewise_lift_transpose(frame, y, lf)
                        for Vt, V in zip(LTy[0] + LTy[1], v[0] + v[1]):
                            self.assertEqual(Vt.shape, V.shape)
                        n = len(K + C)
                        lhs = np.asarray(cw.corewise_stack_dot(Lv, y, n))
                        rhs = np.asarray(cw.corewise_stack_dot(v, LTy, n))
                        scale = (np.asarray(cw.corewise_stack_norm(Lv, n))
                                 * np.asarray(cw.corewise_stack_norm(y, n)))
                        self.assertTrue(np.all(np.abs(lhs - rhs) <= 1e-12 * np.maximum(scale, 1e-300)))


class TestDamping(unittest.TestCase):
    """T4: the damped lift's continuity, the restart and zero-point regimes, and the filters."""

    def setUp(self):
        np.random.seed(4)

    def test_continuity_as_damping_vanishes(self):
        for K, C in _STACKS:
            with self.subTest(K=K, C=C):
                x, frame, _ = _point_and_frame(_MINIMAL, C)
                v = _random_variations(frame, K + C)
                lf0 = fvo.fv_lift_factors(frame, 0.0)
                base = tv.tv_corewise_lift(frame, v, lf0)
                sig_min = min(float(np.min(s)) for s in lf0.tucker_svals + lf0.bond_svals)
                nrm_X = float(np.max(np.sqrt(np.sum(np.asarray(x.to_dense()) ** 2,
                                                    axis=tuple(range(len(C), len(C) + 3))))))
                errs = []
                for cd in (1e-2, 1e-4, 1e-6):
                    lifted = tv.tv_corewise_lift(frame, v, fvo.fv_lift_factors(frame, cd))
                    num = float(cw.corewise_norm(cw.corewise_sub(lifted, base)))
                    err = num / float(cw.corewise_norm(base))
                    bound = 2.0 * (cd * nrm_X / sig_min) ** 2 + 1e-14   # |f - 1/s| <= (eps/s)^2 / s
                    self.assertLessEqual(err, bound)
                    errs.append(err)
                self.assertEqual(errs, sorted(errs, reverse=True))

    def test_finite_at_a_zero_padded_restart(self):
        y = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)).t3svd()[0]
        yp = y.resize((5, 6, 7), (3, 3, 2), (1, 3, 2, 1))
        frame, _ = fvc.t3_orthogonal_representations(yp.data)
        rng = np.random.default_rng(4)
        for K in [(), (3,)]:
            v = _random_variations(frame, K)
            ys = _raw_like_frame_cores(frame, K, rng)
            lifts = {}
            for cd in (0.0, 1e-12, 1e-6, 1e-2):
                with self.subTest(K=K, curvature_damping=cd):
                    lf = fvo.fv_lift_factors(frame, cd)
                    leaves = (lf.tucker_pinvs + lf.bond_pinvs + tv.tv_corewise_lift(frame, v, lf)[0]
                              + tv.tv_corewise_lift(frame, v, lf)[1]
                              + tv.tv_corewise_lift_transpose(frame, ys, lf)[0]
                              + tv.tv_corewise_lift_transpose(frame, ys, lf)[1])
                    self.assertTrue(all(np.all(np.isfinite(a)) for a in leaves))
                    # the restart has genuinely rank-deficient factors, and the pinvs drop exactly those
                    smin = min(float(np.min(s)) for s in lf.tucker_svals + lf.bond_svals)
                    self.assertLess(smin, 1e-14 * np.linalg.norm(yp.to_dense()))
                    for P, s in zip(lf.tucker_pinvs + lf.bond_pinvs, lf.tucker_svals + lf.bond_svals):
                        kept = int(np.sum(s > 1e-10 * s[0]))
                        self.assertEqual(np.linalg.matrix_rank(P, tol=1e-10 * np.linalg.norm(P, 2)), kept)
                    lifts[cd] = tv.tv_corewise_lift(frame, v, lf)
            # the clip, not the damping, handles the zero singular values: eps -> 0 is continuous here too
            err = float(cw.corewise_norm(cw.corewise_sub(lifts[1e-12], lifts[0.0])))
            self.assertLess(err, 1e-12 * float(cw.corewise_norm(lifts[0.0])))

    def test_zero_point_lifts_to_zero_except_the_last_tt_slot(self):
        for struct in (_MINIMAL, _D4, _D1):
            for cd in (0.0, 1e-6):
                with self.subTest(shape=struct[0], curvature_damping=cd):
                    z = t3.TuckerTensorTrain.zeros(*struct)
                    frame, _ = fvc.t3_orthogonal_representations(z.data)
                    lf = fvo.fv_lift_factors(frame, cd)
                    self.assertEqual(float(lf.eps), 0.0)
                    v = _random_variations(frame, (2,))
                    Ud, Gd = tv.tv_corewise_lift(frame, v, lf)
                    self.assertTrue(all(np.all(a == 0.0) for a in Ud))
                    self.assertTrue(all(np.all(a == 0.0) for a in Gd[:-1]))
                    self.assertTrue(np.array_equal(Gd[-1], v[1][-1]))

    def test_filtered_reciprocal(self):
        s = np.array([[4.0, 2.0, 1e-20, 0.0], [1.0, 1e-3, 1e-17, 0.0], [0.0, 0.0, 0.0, 0.0]])  # C = (3,)
        exact = linalg.filtered_reciprocal(s, None)
        self.assertTrue(np.array_equal(exact[0], [0.25, 0.5, 0.0, 0.0]))
        self.assertTrue(np.array_equal(exact[1], [1.0, 1e3, 0.0, 0.0]))
        self.assertTrue(np.array_equal(exact[2], np.zeros(4)))                    # the zero spectrum
        eps = np.array([0.0, 0.5, 2.0])
        damped = linalg.filtered_reciprocal(s, eps)
        keep = s > 4 * np.finfo(float).eps * s[:, :1]
        ref = np.where(keep, s / np.where(keep, s * s + eps[:, None] ** 2, 1.0), 0.0)
        self.assertTrue(np.array_equal(damped, ref))
        self.assertTrue(np.all(damped <= 1.0 / (2.0 * np.maximum(eps[:, None], 1e-300)) + 1e-15))
        self.assertTrue(np.allclose(linalg.filtered_reciprocal(s, np.zeros(3)), exact, rtol=1e-15, atol=0))
        # rcond: a looser clip drops more
        self.assertTrue(np.array_equal(linalg.filtered_reciprocal(s, None, rcond=1e-2)[1], [1.0, 0.0, 0.0, 0.0]))

    def test_filtered_reciprocal_reproduces_the_tied_solve_filter_bitwise(self):
        """eps=None is the shared geometry's clipped group solve, bit for bit (given its rcond)."""
        np.random.seed(0)
        cases = [((6, 6, 6), (3, 3, 3), (1, 2, 2, 1), (0, 0, 0)),
                 ((6, 6, 7), (4, 4, 2), (1, 3, 2, 1), (0, 0, 1))]
        for shape, tr, ttr, labels in cases:
            for padded in (False, True):
                with self.subTest(shape=shape, padded=padded):
                    groups = sharing.validate_sharing(labels, shape)
                    x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
                    tk, tt = x.data
                    tk = list(tk)
                    for g in groups:
                        for i in g[1:]:
                            tk[i] = tk[g[0]]
                    x = t3.TuckerTensorTrain(tuple(tk), tt)
                    if padded:                                  # a restart: zero group-spectrum levels
                        x = x.t3svd(sharing=labels)[0].resize(
                            shape, tuple(t + 1 for t in tr), tuple(1 if k in (0, len(shape)) else t + 1
                                                                   for k, t in enumerate(ttr)),
                            sharing=labels)
                    frame, _ = fvc.t3_orthogonal_representations(x.data)
                    sfd = sharing.fv_shared_frame_data(frame, groups)
                    vs = fvo.fv_variations_randn(fvo.fv_variation_shapes(frame), (2,))
                    for gi, group in enumerate(sharing.nontrivial_groups(groups)):
                        U_M, s_g, Vt_M = sfd.svd_U[gi], sfd.svd_s[gi], sfd.svd_Vt[gi]
                        rcond = np.finfo(s_g.dtype).eps * max(sfd.row_splits[gi][-1], Vt_M.shape[-1])
                        s_inv = linalg.filtered_reciprocal(s_g, None, rcond)
                        Vstack = np.concatenate([vs[0][ii] for ii in group], axis=-2)
                        t = np.einsum('...xw,...xn->...wn', U_M, Vstack)
                        t = np.einsum('...w,...wn->...wn', s_inv, t)
                        mine = np.einsum('...wy,...wn->...yn', Vt_M, t)
                        theirs = sharing._tied_solve(sfd, gi, vs[0], None, np)
                        self.assertTrue(np.array_equal(mine, theirs))

    def test_damped_pseudo_inverse(self):
        rng = np.random.default_rng(5)
        M = rng.standard_normal((2, 3, 5, 4))                                     # C = (2, 3)
        P, s = linalg.damped_pseudo_inverse(M, None)
        self.assertEqual((P.shape, s.shape), ((2, 3, 4, 5), (2, 3, 4)))
        self.assertTrue(np.allclose(P, np.linalg.pinv(M)))
        eps = rng.uniform(0.1, 2.0, size=(2, 3))
        P_eps, _ = linalg.damped_pseudo_inverse(M, eps)
        for c in np.ndindex(2, 3):
            Mc = M[c]
            ref = np.linalg.solve(Mc.T @ Mc + eps[c] ** 2 * np.eye(4), Mc.T)     # Tikhonov
            self.assertTrue(np.allclose(P_eps[c], ref))
            self.assertLessEqual(np.linalg.norm(P_eps[c], 2), 1.0 / (2.0 * eps[c]) + 1e-12)
        # wide matrices, and pad transparency: zero rows/columns of M give exact-zero columns/rows of M^+
        A = np.zeros((6, 7))
        A[:3, :4] = rng.standard_normal((3, 4))
        for e in (None, np.array(0.0), np.array(0.3)):
            Ap, sA = linalg.damped_pseudo_inverse(A, e)
            self.assertEqual(Ap.shape, (7, 6))
            self.assertLess(float(np.max(np.abs(Ap[4:, :]))), 1e-15)
            self.assertLess(float(np.max(np.abs(Ap[:, 3:]))), 1e-15)
            self.assertTrue(np.all(np.isfinite(Ap)))
        Zp, sZ = linalg.damped_pseudo_inverse(np.zeros((3, 2)), None)               # the zero matrix
        self.assertTrue(np.array_equal(Zp, np.zeros((2, 3))) and np.array_equal(sZ, np.zeros(2)))


if __name__ == '__main__':
    unittest.main()
