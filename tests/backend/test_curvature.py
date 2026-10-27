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
* **T5 two-holes (T3-T3)** -- ``t3_inner_product_corewise_hessian`` against the block-sum oracle, the dense
  multilinear Hessian and a finite difference of the one-hole; the one-hole against the dense gradient;
  the tangent zippers against finite differences.
* **T5 two-holes (probe derivatives)** -- ``t3_probe_corewise_derivatives_hessian[_from_sweep]`` at orders
  0-3 against the block-sum oracle (``d`` = 4, 3, 1, slack frames, raw unsquashed cores) and a finite
  difference of the one-hole, and **the shipped (recurrence) kernels against the dense-``trs`` reference twins**
  of ``_curvature_oracles.py`` (orders 0-4), whole and piece by piece, to 1e-14 relative (observed <= 4.4e-16;
  not bitwise -- the summation order differs); the model's sweep
  is reused bitwise; stacked == per element (multi-axis ``W``, ``K``, ``C``); uniform supercores == ragged,
  chunked or not.
* **T6 symmetry** -- ``<a, K b> = <b, K a>`` per ``C`` element (a ``K``-stack Gram matrix is symmetric), for
  the T3-T3 and the probe-derivative two-holes.
* **T11 weights** -- the probe-derivative two-hole at ``z~ = omega^2 r`` (per-order, per-mode and full
  ``omega[mode, order]``) is the curvature of ``1/2 ||omega (.) r||^2``: a finite difference of its gradient
  minus the Gauss-Newton term.
* **T20 flat slots, backend half** -- at maximal rank and at ``d = 1`` the normal space is trivial, so the
  T3-T3 term at ``Z = P_X R`` cancels the dense ``K_R`` on a lifted tangent; and at a generic point the
  normal curvature ``K_{R - P_X R}`` annihilates vertical (gauge) directions (the Lagrangian null space).

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
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.sharing as sharing
import t3toolbox.backend.t3_constructors as t3k
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.t3_linalg as t3la
import t3toolbox.backend.tt_operations as tto
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



# ---------------------------------------------------------------------------------------------------
# S2: the T3-T3 two-hole
# ---------------------------------------------------------------------------------------------------
# (theta structure, z structure): d = 1..4, rank-1 bonds, and an UNSQUASHED leading/trailing bond
_T3T3_CASES = {
    'd1':         (((5,), (3,), (1, 1)),                        ((5,), (2,), (1, 1))),
    'd2':         (((5, 6), (3, 3), (1, 3, 1)),                 ((5, 6), (2, 4), (1, 2, 1))),
    'd3':         ((_MINIMAL),                                  ((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))),
    'd4':         ((_D4),                                       ((4, 5, 4, 5), (3, 2, 2, 3), (1, 3, 2, 3, 1))),
    'rank1':      ((_RANK1),                                    ((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))),
    'unsquashed': (((5, 6, 7), (2, 3, 2), (2, 2, 2, 3)),        ((5, 6, 7), (3, 2, 3), (2, 3, 2, 2))),
}


def _directions(theta, K, rng):
    """Random raw directions at theta: the tangent stack K prepended to theta's own core shapes (C included)."""
    U, G = theta
    return (tuple(rng.standard_normal(K + u.shape) for u in U), tuple(rng.standard_normal(K + g.shape) for g in G))


def _norm_flat(a):
    return float(np.sqrt(sum(np.sum(np.asarray(c) ** 2) for c in a[0] + a[1])))


def _relerr_tree(a, b):
    num = float(np.sqrt(sum(np.sum((np.asarray(p) - np.asarray(q)) ** 2) for p, q in zip(a[0] + a[1], b[0] + b[1]))))
    return num / max(_norm_flat(b), 1e-300)


class TestTangentZipper(unittest.TestCase):
    """The tangent zippers: the product rule of the zipper, both directions, both representations."""

    def setUp(self):
        np.random.seed(5)

    def test_against_finite_differences_and_the_polymorphic_path(self):
        rng = np.random.default_rng(5)
        shapes_A, shapes_B = [(1, 3, 2), (2, 4, 3), (3, 2, 2), (2, 3, 1)], [(1, 3, 3), (3, 4, 2), (2, 2, 2), (2, 3, 1)]
        for K, C in [((), ()), ((2,), ()), ((2,), (3,)), ((2, 2), (2, 3))]:
            with self.subTest(K=K, C=C):
                A = tuple(rng.standard_normal(C + s) for s in shapes_A)
                B = tuple(rng.standard_normal(C + s) for s in shapes_B)
                Ad = tuple(rng.standard_normal(K + C + s) for s in shapes_A)
                Bd = tuple(rng.standard_normal(K + C + s) for s in shapes_B)
                for name, zipper, tangent in (
                        ('left_to_right', tto.tt_zipper_left_to_right, tto.tt_zipper_tangent_left_to_right),
                        ('right_to_left', tto.tt_zipper_right_to_left, tto.tt_zipper_tangent_right_to_left)):
                    Zd = tangent(A, B, Ad, Bd)
                    self.assertEqual(len(Zd), 5)
                    Zd2 = tangent(A, B, Ad, Bd, zippers=zipper(A, B))          # precomputed zippers
                    self.assertTrue(all(np.array_equal(p, q) for p, q in zip(Zd, Zd2)))
                    edge = 0 if name == 'left_to_right' else -1
                    self.assertTrue(np.all(Zd[edge] == 0.0))                   # the boundary does not move
                    for kk in np.ndindex(*K):
                        h = 1e-4
                        at = lambda t: zipper([a + t * ad[kk] for a, ad in zip(A, Ad)],
                                              [b + t * bd[kk] for b, bd in zip(B, Bd)])
                        for z, p, m in zip(Zd, at(h), at(-h)):
                            fd = (p - m) / (2 * h)
                            self.assertLess(np.linalg.norm(z[kk] - fd), 1e-6 * max(np.linalg.norm(fd), 1.0), name)
        # uniform supercores (equal core shapes) give the ragged answer
        A = tuple(rng.standard_normal((2, 3, 2)) for _ in range(3))
        B = tuple(rng.standard_normal((2, 3, 2)) for _ in range(3))
        Ad = tuple(rng.standard_normal((4, 2, 3, 2)) for _ in range(3))
        Bd = tuple(rng.standard_normal((4, 2, 3, 2)) for _ in range(3))
        for tangent in (tto.tt_zipper_tangent_left_to_right, tto.tt_zipper_tangent_right_to_left):
            ragged = tangent(A, B, Ad, Bd)
            uniform = tangent(np.stack(A), np.stack(B), np.stack(Ad), np.stack(Bd))
            self.assertTrue(all(np.allclose(p, q, rtol=1e-14, atol=1e-14) for p, q in zip(ragged, uniform)))


class TestInnerProductTwoHole(unittest.TestCase):
    """T5 (T3-T3 part): the one-hole and the two-hole of theta -> <z, Phi(theta)>."""

    def setUp(self):
        np.random.seed(6)

    def test_one_hole_against_the_dense_gradient(self):
        for name, (th_struct, z_struct) in _T3T3_CASES.items():
            with self.subTest(case=name):
                theta = t3k.t3_corewise_randn(*th_struct)
                z = t3k.t3_corewise_randn(*z_struct)
                got = t3la.t3_inner_product_corewise_transpose(z, theta)
                ref = oracles.dense_inner_product_gradient(np.asarray(t3c.t3_to_dense(z)), theta)
                self.assertLess(_relerr_tree(got, ref), 1e-13)

    def test_two_hole_against_the_block_sum_oracle(self):
        rng = np.random.default_rng(6)
        for name, (th_struct, z_struct) in _T3T3_CASES.items():
            for K, C in _STACKS:
                with self.subTest(case=name, K=K, C=C):
                    theta = t3k.t3_corewise_randn(*th_struct, stack_shape=C)
                    z = t3k.t3_corewise_randn(*z_struct, stack_shape=C)
                    dirs = _directions(theta, K, rng)
                    got = t3la.t3_inner_product_corewise_hessian(z, theta, dirs)
                    for blk, ref_c in zip(got[0] + got[1], dirs[0] + dirs[1]):
                        self.assertEqual(blk.shape, ref_c.shape)
                    for kk in np.ndindex(*K):
                        for cc in np.ndindex(*C):
                            th_c, z_c = _elem(theta, cc), _elem(z, cc)
                            dir_kc = _elem(dirs, kk + cc)
                            ref = oracles.blocksum(
                                t3la.t3_inner_product_corewise_transpose(z_c, oracles.doubled(th_c, dir_kc)), th_c)
                            self.assertLess(_relerr_tree(_elem(got, kk + cc), ref), 1e-13)

    def test_two_hole_against_the_dense_hessian_and_finite_differences(self):
        rng = np.random.default_rng(7)
        for name, (th_struct, z_struct) in _T3T3_CASES.items():
            with self.subTest(case=name):
                theta = t3k.t3_corewise_randn(*th_struct)
                z = t3k.t3_corewise_randn(*z_struct)
                dirs = _directions(theta, (), rng)
                got = t3la.t3_inner_product_corewise_hessian(z, theta, dirs)
                ref = oracles.dense_inner_product_hessian(np.asarray(t3c.t3_to_dense(z)), theta, dirs)
                self.assertLess(_relerr_tree(got, ref), 1e-12)
                # the derivative of the one-hole: a polynomial along the line, central difference O(h^2)
                h = 1e-4
                shift = lambda t: cw.corewise_add(theta, cw.corewise_scale(dirs, t))
                fd = cw.corewise_scale(cw.corewise_sub(t3la.t3_inner_product_corewise_transpose(z, shift(h)),
                                                       t3la.t3_inner_product_corewise_transpose(z, shift(-h))),
                                       0.5 / h)
                self.assertLess(_relerr_tree(got, fd), 1e-6)

    def test_from_sweep_is_the_one_shot_and_is_reusable(self):
        rng = np.random.default_rng(8)
        theta = t3k.t3_corewise_randn(*_D4, stack_shape=(2,))
        z = t3k.t3_corewise_randn((4, 5, 4, 5), (3, 2, 2, 3), (1, 3, 2, 3, 1), stack_shape=(2,))
        sweep = t3la.t3_precompute_inner_product_sweep(z, theta)
        BU, G2, ZL, ZR, env = sweep
        self.assertEqual(tuple(len(p) for p in sweep), (4, 4, 4, 4, 4))
        self.assertEqual(env[1].shape, (2,) + theta[1][1].shape[-3:-2] + z[0][1].shape[-2:-1] + theta[1][1].shape[-1:])
        # the sweep IS the gradient's intermediates: dG_i = env_i x_2 BU_i
        grad = t3la.t3_inner_product_corewise_transpose(z, theta)
        for i in range(4):
            self.assertTrue(np.allclose(np.einsum('...azb,...zu->...aub', env[i], BU[i]), grad[1][i]))
        for K in [(), (3,)]:
            dirs = _directions(theta, K, rng)
            one = t3la.t3_inner_product_corewise_hessian(z, theta, dirs)
            two = t3la.t3_inner_product_corewise_hessian_from_sweep(z, theta, dirs, sweep)
            self.assertTrue(all(np.array_equal(p, q) for p, q in zip(one[0] + one[1], two[0] + two[1])))

    def test_symmetry(self):
        """T6: the K-stack Gram matrix <dir_k, K dir_l> is symmetric, per C element."""
        rng = np.random.default_rng(9)
        for name, (th_struct, z_struct) in _T3T3_CASES.items():
            for C in [(), (2,), (2, 3)]:
                with self.subTest(case=name, C=C):
                    theta = t3k.t3_corewise_randn(*th_struct, stack_shape=C)
                    z = t3k.t3_corewise_randn(*z_struct, stack_shape=C)
                    n_dir = 4
                    dirs = _directions(theta, (n_dir,), rng)
                    Kd = t3la.t3_inner_product_corewise_hessian(z, theta, dirs)
                    c = 'abcdef'[:len(C)]
                    gram = sum(np.einsum('k%sx,l%sx->kl%s' % (c, c, c),                # (k, l) + C
                                         a.reshape((n_dir,) + C + (-1,)), b.reshape((n_dir,) + C + (-1,)))
                               for a, b in zip(dirs[0] + dirs[1], Kd[0] + Kd[1]))
                    asym = np.abs(gram - np.swapaxes(gram, 0, 1)).max()
                    self.assertLess(asym, 1e-12 * np.abs(gram).max())


# ---------------------------------------------------------------------------------------------------
# The probe-derivative two-hole
# ---------------------------------------------------------------------------------------------------
_PD_STRUCTURES = {'d4': _D4, 'd3': _MINIMAL, 'd1': _D1}
_PD_ORDERS = (0, 1, 2, 3)
_PD_STACKS = [((2,), ()), ((2,), (2,)), ((2, 3), ()), ((2, 3), (2,))]     # (K, C)


def _pd_problem(struct, order, K, C, W, rng, raw=False):
    """theta (a frame's base point (U, L), or -- raw -- unconstrained corewise cores), the samples, the
    residual jets z~ (no K) and the directions (K + C)."""
    if raw:
        theta = t3k.t3_corewise_randn(*struct, stack_shape=C)
        frame = None
    else:
        _, frame, _ = _point_and_frame(struct, C)
        theta = (tuple(frame[0]), tuple(frame[2]))
    shape = struct[0]
    ww = [rng.standard_normal(W + (N,)) for N in shape]
    pp = [rng.standard_normal(W + (N,)) for N in shape]
    zt = [rng.standard_normal((order + 1,) + W + C + (N,)) for N in shape]
    return frame, theta, ww, pp, zt, _directions(theta, K, rng)


def _pd_blocksum(zt, ww, pp, theta, dirs, order, kk, cc):
    """The block-sum oracle (F5) for one K + C element: the one-hole at the doubled-rank Y, read back."""
    nW = ww[0].ndim - 1
    th_c, dir_kc = _elem(theta, cc), _elem(dirs, kk + cc)
    zt_c = [z[(slice(None),) * (1 + nW) + cc] for z in zt]
    gY = sd.t3_probe_corewise_derivatives_transpose(zt_c, ww, pp, oracles.doubled(th_c, dir_kc), order,
                                                    sum_over_probes=True, chunk_size=None)
    return oracles.blocksum(gY, th_c)


def _tree_relerr(a, b):
    """||a - b|| / ||b|| over flat sequences of arrays (0 when both vanish)."""
    num = float(np.sqrt(sum(np.sum((np.asarray(p) - np.asarray(q)) ** 2) for p, q in zip(a, b))))
    den = float(np.sqrt(sum(np.sum(np.asarray(q) ** 2) for q in b)))
    return 0.0 if num == 0.0 else num / max(den, 1e-300)


class TestProbeDerivativesTwoHole(unittest.TestCase):
    """T5 (probe derivatives): the shipped two-hole against the block-sum oracle, a finite difference of the
    one-hole, and the dense-trs reference twins; sweep reuse, stacking, and the polymorphic uniform path."""

    def setUp(self):
        np.random.seed(13)

    def test_against_the_block_sum_oracle(self):
        rng = np.random.default_rng(13)
        cases = [(name, struct, False) for name, struct in _PD_STRUCTURES.items()]
        cases += [('tucker_slack', _TUCKER_SLACK, False), ('tt_slack', _TT_SLACK, False),
                  ('raw_unsquashed', ((4, 5, 3), (2, 3, 2), (2, 2, 2, 3)), True),
                  ('raw_d1_unsquashed', ((5,), (3,), (2, 3)), True)]
        for name, struct, raw in cases:
            for order in _PD_ORDERS:
                for K, C in _PD_STACKS:
                    if name not in _PD_STRUCTURES and (K, C) != ((2,), (2,)):
                        continue                                    # the extra structures at one stack
                    with self.subTest(structure=name, order=order, K=K, C=C):
                        _, theta, ww, pp, zt, dirs = _pd_problem(struct, order, K, C, (5,), rng, raw)
                        got = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, dirs, order)
                        for blk, ref_c in zip(got[0] + got[1], dirs[0] + dirs[1]):
                            self.assertEqual(blk.shape, ref_c.shape)
                        for kk in np.ndindex(*K):
                            for cc in np.ndindex(*C):
                                ref = _pd_blocksum(zt, ww, pp, theta, dirs, order, kk, cc)
                                self.assertGreater(_norm_flat(ref), 1e-3)
                                self.assertLess(_relerr_tree(_elem(got, kk + cc), ref), 1e-13)

    def test_against_a_finite_difference_of_the_one_hole(self):
        rng = np.random.default_rng(14)
        for name, struct in _PD_STRUCTURES.items():
            for order in (0, 2):
                with self.subTest(structure=name, order=order):
                    _, theta, ww, pp, zt, dirs = _pd_problem(struct, order, (), (), (6,), rng)
                    got = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, dirs, order)
                    one = lambda cores: sd.t3_probe_corewise_derivatives_transpose(
                        zt, ww, pp, cores, order, sum_over_probes=True, chunk_size=None)
                    h = 2e-5                        # a polynomial in h: central difference O(h^2)
                    shift = lambda t: cw.corewise_add(theta, cw.corewise_scale(dirs, t))
                    fd = cw.corewise_scale(cw.corewise_sub(one(shift(h)), one(shift(-h))), 0.5 / h)
                    self.assertLess(_relerr_tree(got, fd), 1e-6)

    def test_production_equals_the_trs_twins(self):
        """The shipped kernels (banded recurrences / order scans with a carried shifted jet; never the binomial
        tensor) == the dense-trs twins, whole and piece by piece (each piece fed the twin's own inputs), at orders
        0..4, a single and a multi-axis sample stack W, d in {1, 3, 4}. Observed <= 4.4e-16; the bar is 1e-14
        relative -- NOT bitwise: the recurrences sum in a different order from the dense einsums, and CI runs two
        numpy generations (other einsum / BLAS orders), so a ~2x margin would be flaky. It loses no
        sensitivity: a dropped or rescaled product-rule term moves the result by O(1e-2..1). The dotted dxi~ is
        formed only on its two read rows (the twin computes all ``order+1``), so it is compared on those; the
        TT assembly is checked against both the nine-term twin and the factored dense form S3 shipped."""
        rng = np.random.default_rng(15)
        worst = 0.0
        for name, struct in _PD_STRUCTURES.items():
            for order in range(5):
                for (K, C), W in [(kc, (5,)) for kc in _PD_STACKS] + [(((2,), (2,)), (2, 3)), (((2, 3), ()), (2, 3))]:
                    with self.subTest(structure=name, order=order, K=K, C=C, W=W):
                        _, theta, ww, pp, zt, dirs = _pd_problem(struct, order, K, C, W, rng)
                        st = oracles.probe_derivatives_two_hole_states_trs(zt, ww, pp, theta, dirs, order)
                        G, Gd = theta[1], dirs[1]
                        s2 = min(2, order + 1)
                        dxi_got = sd.compute_dxi_tilde_tangent_jets(
                            G, Gd, st['mu'], st['mud'], st['nu'], st['nud'], st['sigma_t'], st['sigma_td'],
                            st['tau_t'], st['tau_td'])
                        self.assertEqual([np.shape(g)[0] for g in dxi_got], [2] * len(dxi_got))   # the two-row contract
                        if order == 0:
                            self.assertTrue(all(np.all(g[1] == 0) for g in dxi_got))         # row 1 is zero at order 0
                        tt_args = (st['sigma_t'], st['sigma_td'], st['tau_t'], st['tau_td'], st['deta_t'],
                                   st['deta_td'], st['xi'], st['xid'], st['mu'], st['mud'], st['nu'], st['nud'])
                        tt_got = sd.assemble_tt_tangent_variation_jets(*tt_args, len(W))
                        pieces = {
                            'whole': (sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, dirs, order),
                                      (st['dU'], st['dG'])),
                            'deta_tilde_dot': (sd.compute_deta_tilde_tangent_jets(theta[0], dirs[0], zt),
                                               st['deta_td']),
                            'tau_tilde_dot': (sd.compute_tau_tilde_tangent_jets(
                                G, Gd, st['xi'], st['xid'], st['deta_t'], st['deta_td'], st['mu'], st['mud'],
                                st['tau_t']), st['tau_td']),
                            'sigma_tilde_dot': (sd.compute_sigma_tilde_tangent_jets(
                                G, Gd, st['xi'], st['xid'], st['deta_t'], st['deta_td'], st['nu'], st['nud'],
                                st['sigma_t']), st['sigma_td']),
                            'dxi_tilde_dot': (tuple(g[:s2] for g in dxi_got), tuple(r[:2] for r in st['dxi_td'])),
                            'tucker_assembly': (sd.assemble_tucker_tangent_variation_jets(
                                zt, tuple(r[:2] for r in st['dxi_td']), ww, pp, st['etad'], len(W)), st['dU']),
                            'tt_assembly': (tt_got, st['dG']),
                            'tt_assembly_vs_factored': (tt_got, oracles.tt_tangent_assembly_factored_trs(
                                *tt_args, st['trs'], st['n_frame'])),
                        }
                        self.assertGreater(_norm_flat((st['dU'], st['dG'])), 1e-3)
                        for piece, (got, ref) in pieces.items():
                            if piece == 'whole':
                                got, ref = got[0] + got[1], ref[0] + ref[1]
                            self.assertEqual([np.shape(g) for g in got], [np.shape(r) for r in ref], piece)
                            err = _tree_relerr(got, ref)
                            worst = max(worst, err)
                            self.assertLess(err, 1e-14, piece)
        self.assertLess(worst, 1e-14)

    def test_from_sweep_reuses_the_model_sweep(self):
        """The manifold model's probe sweep at the frame (U, O, L, R) carries the corewise xi / mu of its base
        point (U, L): from_sweep with it -- or with the lean corewise (xi, mu) pair -- is the one-shot, bitwise;
        and one sweep serves every direction."""
        rng = np.random.default_rng(16)
        for C in [(), (2,)]:
            for order in (0, 2):
                frame, theta, ww, pp, zt, _ = _pd_problem(_D4, order, (), C, (5,), rng)
                model_sweep = sd.tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order)
                lean_sweep = sd.tv_precompute_apply_frame_sweep_jets(fvc.t3_corewise_frame(theta), ww, pp, order)
                for K in [(), (3,)]:
                    with self.subTest(C=C, order=order, K=K):
                        dirs = _directions(theta, K, rng)
                        one = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, dirs, order)
                        for sweep in (model_sweep, lean_sweep):
                            two = sd.t3_probe_corewise_derivatives_hessian_from_sweep(
                                zt, ww, pp, theta, dirs, order, sweep)
                            self.assertTrue(all(np.array_equal(p, q) for p, q in zip(one[0] + one[1], two[0] + two[1])))

    def test_stacked_equals_per_element(self):
        """Multi-axis W, K and C (and an empty W -- one probe): each stack element is the unstacked result."""
        rng = np.random.default_rng(17)
        for W, K, C in [((2, 3), (2, 2), (2, 3)), ((), (2,), (2,)), ((4,), (), ())]:
            for order in (0, 2):
                with self.subTest(W=W, K=K, C=C, order=order):
                    _, theta, ww, pp, zt, dirs = _pd_problem(_MINIMAL, order, K, C, W, rng)
                    got = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, dirs, order)
                    for cc in np.ndindex(*C):
                        zt_c = [z[(slice(None),) * (1 + len(W)) + cc] for z in zt]
                        for kk in np.ndindex(*K):
                            ref = sd.t3_probe_corewise_derivatives_hessian(
                                zt_c, ww, pp, _elem(theta, cc), _elem(dirs, kk + cc), order)
                            self.assertLess(_relerr_tree(_elem(got, kk + cc), ref), 1e-13)

    def test_uniform_supercores_and_chunking(self):
        """The kernel is polymorphic: packed uniform supercores (clean padding) give the ragged real block, with
        the assemblies dense or W-chunked; on the ragged path chunk_size is ignored (bitwise)."""
        import t3toolbox.backend.ut3_conversions as u3c
        import t3toolbox.backend.ut3_operations as u3o
        rng = np.random.default_rng(18)
        for name, struct in _PD_STRUCTURES.items():
            for K, C in [((2,), ()), ((2, 3), (2,))]:
                with self.subTest(structure=name, K=K, C=C):
                    shape, tr, ttr = struct
                    d, order, W = len(shape), 2, 7
                    _, theta, ww, pp, zt, dirs = _pd_problem(struct, order, K, C, (W,), rng)
                    rag = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, dirs, order, chunk_size=None)
                    rag3 = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, dirs, order, chunk_size=3)
                    self.assertTrue(all(np.array_equal(p, q) for p, q in zip(rag[0] + rag[1], rag3[0] + rag3[1])))
                    N, n, r = max(shape), max(tr) + 1, max(ttr) + 1      # forced-larger padding everywhere
                    th_u = u3c.t3_to_ut3(theta, N=N, n=n, r=r, squash_tails=False)[:2]
                    dir_u = u3c.t3_to_ut3(dirs, N=N, n=n, r=r, squash_tails=False)[:2]
                    packed = [u3o.pack_vectors(v, N) for v in (zt, ww, pp)]
                    for cs in (None, 3, 2):                                # dense; 2 full chunks + rem; 3 + rem
                        oU, oG = sd.t3_probe_corewise_derivatives_hessian(*packed, th_u, dir_u, order, chunk_size=cs)
                        self.assertEqual(oU.shape, (d,) + K + C + (n, N))
                        self.assertEqual(oG.shape, (d,) + K + C + (r, n, r))
                        real = ([oU[i][..., :tr[i], :shape[i]] for i in range(d)],
                                [oG[i][..., :ttr[i], :tr[i], :ttr[i + 1]] for i in range(d)])
                        self.assertLess(_relerr_tree(real, rag), 1e-13, cs)

    def test_symmetry(self):
        """T6: the K-stack Gram matrix <dir_k, K dir_l> is symmetric, per C element."""
        rng = np.random.default_rng(19)
        for name, struct in _PD_STRUCTURES.items():
            for order in _PD_ORDERS:
                for C in [(), (2,)]:
                    with self.subTest(structure=name, order=order, C=C):
                        n_dir = 4
                        _, theta, ww, pp, zt, dirs = _pd_problem(struct, order, (n_dir,), C, (5,), rng)
                        Kd = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, dirs, order)
                        c = 'abcdef'[:len(C)]
                        gram = sum(np.einsum('k%sx,l%sx->kl%s' % (c, c, c),
                                             a.reshape((n_dir,) + C + (-1,)), b.reshape((n_dir,) + C + (-1,)))
                                   for a, b in zip(dirs[0] + dirs[1], Kd[0] + Kd[1]))
                        asym = np.abs(gram - np.swapaxes(gram, 0, 1)).max()
                        self.assertGreater(np.abs(gram).max(), 1e-3)
                        self.assertLess(asym, 1e-12 * np.abs(gram).max())

    def test_residual_weights(self):
        """T11: with the kind's omega^2 applied to the residual, the two-hole is the curvature part of the Hessian
        of f(theta) = 1/2 sum_{i,t} omega[i,t]^2 ||z_i^(t)(theta) - y_i^(t)||^2: a central difference of
        grad f = J^T (omega^2 r) along theta_hat, minus the Gauss-Newton term J^T (omega^2 J theta_hat)."""
        rng = np.random.default_rng(20)
        struct, order, W = _MINIMAL, 2, (6,)
        d = len(struct[0])
        weights = {'per_order': [1.0, 0.5, 0.25],
                   'per_mode': [[1.0], [0.3], [2.0]],
                   'matrix': rng.uniform(0.2, 2.0, (d, order + 1))}
        _, theta, ww, pp, _, dirs = _pd_problem(struct, order, (), (), W, rng)
        y = [rng.standard_normal((order + 1,) + W + (N,)) for N in struct[0]]
        z_of = lambda cores: sd.t3_probe_derivatives(ww, pp, cores, order)
        JT = lambda r, cores: sd.t3_probe_corewise_derivatives_transpose(r, ww, pp, cores, order,
                                                                       sum_over_probes=True, chunk_size=None)
        for label, w in weights.items():
            with self.subTest(weight=label):
                w2 = lambda r: bfit.ProbeDerivativesKind(order=order, weight=w)._apply_weight(list(r), 2)
                grad = lambda cores: JT(w2([z - yy for z, yy in zip(z_of(cores), y)]), cores)
                h = 2e-5                            # a polynomial in h: central difference O(h^2)
                shift = lambda t: cw.corewise_add(theta, cw.corewise_scale(dirs, t))
                hess = cw.corewise_scale(cw.corewise_sub(grad(shift(h)), grad(shift(-h))), 0.5 / h)
                gn = JT(w2(z_of(oracles.doubled(theta, dirs))), theta)          # J theta_hat = z(Y)
                curv_fd = cw.corewise_sub(hess, gn)
                r = [z - yy for z, yy in zip(z_of(theta), y)]
                got = sd.t3_probe_corewise_derivatives_hessian(w2(r), ww, pp, theta, dirs, order)
                self.assertLess(_relerr_tree(got, curv_fd), 1e-6)
                unweighted = sd.t3_probe_corewise_derivatives_hessian(r, ww, pp, theta, dirs, order)
                self.assertGreater(_relerr_tree(unweighted, curv_fd), 1e-2)       # the weight is not a no-op


class TestFlatSlotsAndVerticalDirections(unittest.TestCase):
    """T20 (backend half) and the Lagrangian null space (F7), through the lift and the T3-T3 term against
    the dense multilinear Hessian of an ambient R (standing in for A^T r)."""

    def setUp(self):
        np.random.seed(10)

    @staticmethod
    def _setup(struct, rng):
        x = t3.TuckerTensorTrain.randn(*struct)
        frame, _ = fvc.t3_orthogonal_representations(x.data)
        theta = (tuple(frame[0]), tuple(frame[2]))                 # the frame's own raw cores (U, L)
        R = rng.standard_normal(struct[0])                         # an ambient tensor (A^T r)
        g = tv.tv_project_dense_onto_tangent_space(frame, R)       # P_X R, gauged coordinates
        Zg = tv.tv_to_t3(frame, g)                                 # ... as a doubled-rank T3
        return frame, theta, R, Zg

    def test_normal_curvature_vanishes_on_flat_structures(self):
        rng = np.random.default_rng(10)
        flat = {'maximal': _MAXIMAL, 'maximal_d4': ((2, 2, 2, 2), (2, 2, 2, 2), (1, 2, 4, 2, 1)),
                'd1': _D1, 'd1_rank1': ((7,), (1,), (1, 1))}
        for name, struct in flat.items():
            with self.subTest(structure=name):
                frame, theta, R, Zg = self._setup(struct, rng)
                self.assertLess(np.linalg.norm(tv.tv_to_dense(frame, tv.tv_project_dense_onto_tangent_space(frame, R)) - R),
                                1e-12 * np.linalg.norm(R))          # flat: the tangent space is everything
                lf = fvo.fv_lift_factors(frame, 0.0)
                p = tv.tv_orthogonal_gauge_projection(frame, _random_variations(frame, ()))
                theta_hat = tv.tv_corewise_lift(frame, p, lf)
                K_R = oracles.dense_inner_product_hessian(R, theta, theta_hat)
                K_g = t3la.t3_inner_product_corewise_hessian(Zg, theta, theta_hat)
                K_N = cw.corewise_sub(K_R, K_g)
                self.assertLess(_norm_flat(K_N), 1e-12 * _norm_flat(K_R))
                LT_KN = tv.tv_corewise_lift_transpose(frame, K_N, lf)
                LT_KR = tv.tv_corewise_lift_transpose(frame, K_R, lf)
                self.assertLess(_norm_flat(LT_KN), 1e-12 * _norm_flat(LT_KR))

    def test_normal_curvature_annihilates_vertical_directions(self):
        rng = np.random.default_rng(11)
        for struct in (_MINIMAL, _D4, _D2):
            with self.subTest(shape=struct[0]):
                frame, theta, R, Zg = self._setup(struct, rng)
                U, O, L, Rr = frame
                d = len(U)
                # (a) a frame-coordinate vertical p (E p = 0), lifted; (b) a raw gauge direction
                VV = [np.zeros(o.shape[-2:-1] + u.shape[-1:]) for o, u in zip(O, U)]
                HH = [np.zeros(h) for h in fvo.fv_variation_shapes(frame)[1]]
                for i in range(d):
                    X = rng.standard_normal((O[i].shape[-2], U[i].shape[-2]))
                    VV[i] += X @ U[i]
                    HH[i] -= np.einsum('aib,ij->ajb', O[i], X)
                for i in range(d - 1):
                    Y = rng.standard_normal((L[i].shape[-1], Rr[i + 1].shape[-3]))
                    HH[i] += np.einsum('iaj,jk->iak', L[i], Y)
                    HH[i + 1] -= np.einsum('jk,kbl->jbl', Y, Rr[i + 1])
                p_vert = (tuple(VV), tuple(HH))
                self.assertLess(np.linalg.norm(tv.tv_to_dense(frame, p_vert)), 1e-12 * _norm_flat(p_vert))
                lifted = tv.tv_corewise_lift(frame, p_vert, fvo.fv_lift_factors(frame, 0.0))
                for which, theta_hat in (('lifted', lifted), ('raw gauge', oracles.vertical_direction(theta, rng))):
                    corewise_frame = fvc.t3_corewise_frame(theta)
                    self.assertLess(np.linalg.norm(tv.tv_to_dense(corewise_frame, theta_hat)),
                                    1e-12 * _norm_flat(theta_hat), which)       # DPhi[theta_hat] = 0
                    K_R = oracles.dense_inner_product_hessian(R, theta, theta_hat)
                    K_g = t3la.t3_inner_product_corewise_hessian(Zg, theta, theta_hat)
                    self.assertGreater(_norm_flat(K_R), 1e-3)                    # not trivially zero
                    self.assertLess(_norm_flat(cw.corewise_sub(K_R, K_g)), 1e-11 * _norm_flat(K_R), which)

    def test_dense_lagrangian_hessian_has_the_vertical_null_space(self):
        """The helper's full Lagrangian Hessian (validating the oracle itself): at G = P_X A^T r the
        vertical directions are null; at G = 0 (the plain Hessian of f o Phi) they are not."""
        rng = np.random.default_rng(12)
        struct = ((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        x = t3.TuckerTensorTrain.randn(*struct)
        frame, _ = fvc.t3_orthogonal_representations(x.data)
        theta = (tuple(frame[0]), tuple(frame[2]))
        A = rng.standard_normal((40, 60))
        y = rng.standard_normal(40)
        r = A @ np.asarray(t3c.t3_to_dense(theta)).ravel() - y
        Atr = (A.T @ r).reshape(struct[0])
        G = tv.tv_to_dense(frame, tv.tv_project_dense_onto_tangent_space(frame, Atr))
        theta_vert = oracles.vertical_direction(theta, rng)
        lag = oracles.dense_lagrangian_hessian(A, y, theta, theta_vert, G)
        plain = oracles.dense_lagrangian_hessian(A, y, theta, theta_vert, np.zeros(struct[0]))
        self.assertGreater(_norm_flat(plain), 1e-3)
        self.assertLess(_norm_flat(lag), 1e-11 * _norm_flat(plain))
        # and its curvature part is the T3-T3 term at Z = P_X A^T r beside the dense one at A^T r
        Zg = tv.tv_to_t3(frame, tv.tv_project_dense_onto_tangent_space(frame, Atr))
        a = _directions(theta, (), rng)
        J = oracles.dense_core_jacobian(theta)
        gn = J.T @ (A.T @ (A @ (J @ oracles.flatten(a))))
        curv = oracles.flatten(cw.corewise_sub(oracles.dense_inner_product_hessian(Atr, theta, a),
                                               t3la.t3_inner_product_corewise_hessian(Zg, theta, a)))
        full = oracles.flatten(oracles.dense_lagrangian_hessian(A, y, theta, a, G))
        self.assertLess(np.linalg.norm(gn + curv - full), 1e-11 * np.linalg.norm(full))


if __name__ == '__main__':
    unittest.main()
