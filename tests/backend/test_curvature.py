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
* **T5 two-holes (plain probe / apply / entries, S10 / S11)** -- ``t3_{probe,apply,entries}_corewise_hessian
  [_from_sweep]`` against the block-sum oracle (``d`` = 4, 3, 1, slack frames, raw unsquashed cores, multi-axis
  ``K``) and a finite difference of the one-hole; each probe piece against a finite difference of the GN kernel it
  differentiates; the factored TT assemblies against their unfactored product rule (there is no order axis, so no
  binomial structure and no ``_trs`` twin); sweep reuse bitwise; stacked == per element; uniform supercores.
* **T5 two-holes (apply / entries derivatives, S12)** -- ``sd.t3_{apply,entries}_corewise_derivatives_hessian
  [_from_sweep]`` at orders 0-3 against the block-sum oracle and an exact (Richardson) finite difference of the
  one-hole; the shipped kernels against their dense-``trs`` twins, whole and piece by piece, orders 0-4, to 1e-14;
  order 0 is the plain two-hole; the containments (entries == apply at the one-hot vectors, bitwise; apply ==
  probe derivatives at the paired residual jets); sweep reuse bitwise; stacked == per element; uniform supercores.
* **T6 symmetry** -- ``<a, K b> = <b, K a>`` per ``C`` element (a ``K``-stack Gram matrix is symmetric), for
  the T3-T3, the probe-derivative, the plain kinds' and the apply / entries derivative two-holes.
* **T11 weights** -- the probe-derivative two-hole at ``z~ = omega^2 r`` (per-order, per-mode and full
  ``omega[mode, order]``) is the curvature of ``1/2 ||omega (.) r||^2``: a finite difference of its gradient
  minus the Gauss-Newton term; for plain probe the per-mode weight row (apply / entries carry no weight); for the
  apply / entries derivatives the per-order row (a per-mode one is their structural error).
* **T20 flat slots, backend half** -- at maximal rank and at ``d = 1`` the normal space is trivial, so the
  T3-T3 term at ``Z = P_X R`` cancels the dense ``K_R`` on a lifted tangent; and at a generic point the
  normal curvature ``K_{R - P_X R}`` annihilates vertical (gauge) directions (the Lagrangian null space).

The exact local model, ``Problem.local_model(hessian='exact')``:

* **T7** -- ``<p, H p>`` is ``d^2/dt^2 f(tv_retract(t p))`` (an exact oracle: the T3-SVD retraction is second
  order), resolving the curvature term itself; probe derivatives at orders 0-2 with every weight shape, and an
  apply kind whose two-hole is the block-sum oracle. **T8** -- ``K_N L p_vert = 0``, ``hvp(p) = hvp(Pi p)``,
  ``Pi hvp = hvp``. **T9** -- ``Pi L^T H_lag L Pi`` with the dense Lagrangian Hessian. **T10** -- regularized.
  The full-hvp symmetry (T6), ``K + C`` stacks, the derived mode, the bit-identical gradient, and the
  ``curvature_damping`` rules.
* **T13** -- corewise: the plain Hessian of ``f o Phi`` (block-sum; a finite difference of the gradient).
* **T20, the model half** -- the manifold ``d = 1`` model is Gauss-Newton; at maximal rank and ``d = 1`` (the
  shortcut bypassed) ``L^T K_N L Pi p`` vanishes; corewise ``d = 1`` runs exact.
* **T23** -- ``corewise_hessian`` is the required sixth operation: a five-operation kind works for Gauss-Newton
  and is rejected for exact (the marker and the identity forms agree); every built-in kind implements it, every
  uniform kind on its own class (the S4-S12 window stub is gone), and every uniform kind builds exact.
* **The plain kinds through the model (S10 / S11)** -- T7 (the retraction's second derivative), T8 (``K_N L
  p_vert = 0`` through each kind), T13 (corewise), the full-hvp symmetry and stacks, and each kind's
  ``corewise_hessian`` (``omega^2 r``, the model's sweep reused).
* **The apply / entries derivative kinds through the model (S12)** -- the same, at orders 0-2 with per-order
  weights, plus one ragged SHARED subcase per kind (the hvp is the derivative of the projected shared gradient).

Exact Newton end to end, ``newton_cg(hessian='exact')`` with its trust region (S5):

* **T16** -- the local rate: q-quadratic at ``cg_forcing_power=1.0``, superlinear of order ~1.5 at the default
  0.5, Gauss-Newton linear; ``rho`` ~ 1 on the asymptotic steps.
* **T21** -- a zero-padded continuation restart converges on the gradient test with at most two rejections.
* **T25** -- from Tucker-slack and TT-slack points (no minimal reduction): the slack frame, and the same
  trajectory as from the minimal-rank tensor.

The shared manifold (S7):

* **T12** -- the tied lift (one raw factor direction per sharing group) represents a tied tangent, its transpose is
  the exact adjoint; the exact model against the second derivative along the shared retraction (measured O(h^2),
  F14), the derivative of the projected shared gradient (exact for any retraction) and the dense Lagrangian of
  ``Phi o T`` through an independent dense tied lift; gauge laws, symmetry, stacks; ``sigma_min`` reads ``s_g``; a
  zero-padded shared restart gates the new ``s_g`` level. Every model check also catches the per-mode lift on
  structures where a group's rank exceeds a member's ceiling.
* **T22** -- exact newton_cg from a zero-padded shared restart converges on the gradient test.

Stacks: none, ``C``, ``K``, ``K + C`` and multi-axis ``K = (2, 2)``, ``C = (2, 3)``. Per-test seeded.
"""
import dataclasses as dc
import os
import sys
import unittest
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.corewise as cw
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.sharing as sharing
import t3toolbox.backend.t3_constructors as t3k
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.t3_linalg as t3la
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv
from t3toolbox.backend import apply as bapply
from t3toolbox.backend import entries as bentries
import t3toolbox.backend.probing as probing

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
        """eps=None is the shared geometry's clipped group solve, bit for bit (given its rcond): the refactored
        solve (``_tied_filtered_solve`` at ``_tied_filter(eps=None)``, S7) equals the pre-S7 ``_tied_solve`` --
        frozen here verbatim: its own double-``where`` filter and its three einsums -- through both public
        callers (the tied ambient directions, and the tied projection's redistribution)."""
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
                    udots = sharing.fv_tied_ambient_directions(vs, sfd)
                    tied, _ = sharing.fv_share_tucker_variations(vs, sfd)
                    for gi, group in enumerate(sharing.nontrivial_groups(groups)):
                        U_M, s_g, Vt_M = sfd.svd_U[gi], sfd.svd_s[gi], sfd.svd_Vt[gi]
                        splits = sfd.row_splits[gi]
                        # the pre-S7 _tied_solve(sfd, gi, V, rcond=None, np), verbatim
                        eff_rcond = float(np.finfo(s_g.dtype).eps) * max(splits[-1], Vt_M.shape[-1])
                        keep = s_g > eff_rcond * s_g[..., :1]
                        s_inv = np.where(keep, 1.0 / np.where(keep, s_g, 1.0), 0.0)
                        self.assertTrue(np.array_equal(linalg.filtered_reciprocal(s_g, None, eff_rcond), s_inv))
                        self.assertTrue(np.array_equal(sharing._tied_filter(sfd, gi, None, None), s_inv))
                        Vstack = np.concatenate([vs[0][ii] for ii in group], axis=-2)
                        t = np.einsum('...xw,...xn->...wn', U_M, Vstack)
                        t = np.einsum('...w,...wn->...wn', s_inv, t)
                        frozen = np.einsum('...wy,...wn->...yn', Vt_M, t)
                        self.assertTrue(np.array_equal(udots[gi], frozen))
                        r = np.einsum('...w,...wn->...wn', s_g, np.einsum('...wy,...yn->...wn', Vt_M, frozen))
                        R = np.einsum('...xw,...wn->...xn', U_M, r)                   # the redistribution
                        for jj, ii in enumerate(group):
                            self.assertTrue(np.array_equal(tied[ii], R[..., splits[jj]:splits[jj + 1], :]))

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


# ---------------------------------------------------------------------------------------------------
# The plain kinds' two-holes (S10 / S11): probe, apply, entries
# ---------------------------------------------------------------------------------------------------
_PLAIN_KINDS = ('probe', 'apply', 'entries')
_PLAIN_STACKS = [((), ()), ((2,), ()), ((2,), (2,)), ((2, 3), ()), ((2, 3), (2,))]     # (K, C)


def _plain_ops(kind):
    """(two-hole, two-hole from a sweep, the one-hole -- the W-summed corewise transpose, the sweep builder at a
    frame) for a plain kind; the residual is the kind's own layout (probe: a list of d arrays W+C+(Ni,); apply /
    entries: one array W+C)."""
    if kind == 'probe':
        return (probing.t3_probe_corewise_hessian, probing.t3_probe_corewise_hessian_from_sweep,
                lambda z, s, th: probing.t3_probe_corewise_transpose(z, s, th, sum_over_probes=True),
                probing.tv_precompute_probe_frame_sweep)
    if kind == 'apply':
        return (bapply.t3_apply_corewise_hessian, bapply.t3_apply_corewise_hessian_from_sweep,
                lambda z, s, th: bapply.t3_apply_corewise_transpose(z, s, th, sum_over_probes=True),
                bapply.tv_precompute_apply_frame_sweep)
    return (bentries.t3_entries_corewise_hessian, bentries.t3_entries_corewise_hessian_from_sweep,
            lambda z, s, th: bentries.t3_entries_corewise_transpose(z, s, th, sum_over_probes=True),
            bentries.tv_precompute_entries_frame_sweep)


def _plain_problem(kind, struct, K, C, W, rng, raw=False):
    """theta (a frame's base point (U, L), or -- raw -- unconstrained corewise cores), the sample (ww, or an index
    grid for entries), the residual (no K) and the directions (K + C), as :py:func:`_pd_problem`."""
    if raw:
        theta = t3k.t3_corewise_randn(*struct, stack_shape=C)
        frame = None
    else:
        _, frame, _ = _point_and_frame(struct, C)
        theta = (tuple(frame[0]), tuple(frame[2]))
    shape = struct[0]
    if kind == 'entries':
        sample = np.stack([rng.integers(0, N, size=W) for N in shape]) if W else np.stack(
            [np.asarray(rng.integers(0, N)) for N in shape])
    else:
        sample = [rng.standard_normal(W + (N,)) for N in shape]
    z = ([rng.standard_normal(W + C + (N,)) for N in shape] if kind == 'probe' else rng.standard_normal(W + C))
    return frame, theta, sample, z, _directions(theta, K, rng)


def _plain_residual_at(kind, z, nW, cc):
    """The residual of one C element ``cc``."""
    sl = (slice(None),) * nW + cc
    return [zi[sl] for zi in z] if kind == 'probe' else z[sl]


def _plain_blocksum(kind, z, sample, theta, dirs, kk, cc):
    """The block-sum oracle (F5) for one K + C element: the one-hole at the doubled-rank Y, read back."""
    nW = (sample[0].ndim - 1) if kind != 'entries' else sample.ndim - 1
    th_c, dir_kc = _elem(theta, cc), _elem(dirs, kk + cc)
    one = _plain_ops(kind)[2]
    return oracles.blocksum(one(_plain_residual_at(kind, z, nW, cc), sample, oracles.doubled(th_c, dir_kc)), th_c)


class TestPlainKindsTwoHole(unittest.TestCase):
    """T5 / T6 / T11 for the plain kinds (S10 / S11): ``probing.t3_probe_corewise_hessian``,
    ``apply.t3_apply_corewise_hessian``, ``entries.t3_entries_corewise_hessian`` (and their ``_from_sweep`` forms)
    against the block-sum oracle (``d`` = 1, 3, 4, slack frames, raw unsquashed cores, ``K`` and ``C`` stacks
    including multi-axis ``K``), a finite difference of the one-hole, symmetry, stacked == per element, the
    polymorphic uniform path; the probe pieces each against a finite difference of the GN kernel it differentiates,
    and the factored TT assemblies against their unfactored product rule. (T5's "production == the ``_trs`` twins"
    clause is vacuous here: there is no order axis, so no binomial structure and no twin.)"""

    def setUp(self):
        np.random.seed(51)

    def test_against_the_block_sum_oracle(self):
        rng = np.random.default_rng(51)
        cases = [(name, struct, False) for name, struct in _PD_STRUCTURES.items()]
        cases += [('tucker_slack', _TUCKER_SLACK, False), ('tt_slack', _TT_SLACK, False),
                  ('raw_unsquashed', ((4, 5, 3), (2, 3, 2), (2, 2, 2, 3)), True),
                  ('raw_d1_unsquashed', ((5,), (3,), (2, 3)), True)]
        for kind in _PLAIN_KINDS:
            hess = _plain_ops(kind)[0]
            for name, struct, raw in cases:
                for K, C in _PLAIN_STACKS:
                    if name not in _PD_STRUCTURES and (K, C) != ((2,), (2,)):
                        continue                                    # the extra structures at one stack
                    with self.subTest(kind=kind, structure=name, K=K, C=C):
                        _, theta, sample, z, dirs = _plain_problem(kind, struct, K, C, (5,), rng, raw)
                        got = hess(z, sample, theta, dirs)
                        for blk, ref_c in zip(got[0] + got[1], dirs[0] + dirs[1]):
                            self.assertEqual(blk.shape, ref_c.shape)
                        for kk in np.ndindex(*K):
                            for cc in np.ndindex(*C):
                                ref = _plain_blocksum(kind, z, sample, theta, dirs, kk, cc)
                                self.assertGreater(_norm_flat(ref), 1e-3)
                                self.assertLess(_relerr_tree(_elem(got, kk + cc), ref), 1e-13)

    def test_against_a_finite_difference_of_the_one_hole(self):
        """The one-hole along ``theta + s theta_hat`` is a polynomial in ``s`` of degree ``<= 2d - 1`` (``Phi`` is
        multilinear in the ``2d`` cores), so central differences at ``d`` step sizes, Richardson-eliminated through
        ``h^(2d-2)``, are its derivative EXACTLY up to rounding: the two-hole matches to <= 1e-12 (observed <= 3.9e-15),
        not to a finite difference's truncation."""
        rng = np.random.default_rng(52)
        for kind in _PLAIN_KINDS:
            hess, _, one, _ = _plain_ops(kind)
            for name, struct in _PD_STRUCTURES.items():
                with self.subTest(kind=kind, structure=name):
                    _, theta, sample, z, dirs = _plain_problem(kind, struct, (), (), (6,), rng)
                    got = hess(z, sample, theta, dirs)
                    d = len(struct[0])
                    hs = [0.1 / 2 ** k for k in range(d)]
                    weights = np.linalg.solve(np.array([[h ** (2 * j) for j in range(d)] for h in hs]).T, np.eye(d)[0])
                    g = lambda t: one(z, sample, cw.corewise_add(theta, cw.corewise_scale(dirs, t)))
                    fd = None
                    for w, h in zip(weights, hs):
                        term = cw.corewise_scale(cw.corewise_sub(g(h), g(-h)), w * 0.5 / h)
                        fd = term if fd is None else cw.corewise_add(fd, term)
                    self.assertLess(_relerr_tree(got, fd), 1e-12)

    def test_probe_pieces_against_finite_differences_of_the_gn_kernels(self):
        """Each dotted probe piece is the directional derivative of the GN kernel it is named after (fed the
        corewise frame's undotted state, which is polynomial in the cores: a central difference is O(h^2)); the
        factored TT assembly equals its nine-term product rule to rounding."""
        rng = np.random.default_rng(53)
        for name, struct in _PD_STRUCTURES.items():
            for K, C in [((), ()), ((2,), (2,))]:
                with self.subTest(structure=name, K=K, C=C):
                    _, theta, ww, zt, dirs = _plain_problem('probe', struct, K, C, (5,), rng)
                    U, G = theta
                    Ud, Gd = dirs

                    def state(cores):                                # the corewise undotted state of theta
                        Uc, Gc = cores
                        xi = probing.compute_xi(Uc, ww)
                        mu, nu = probing.compute_mu(Gc, xi), probing.compute_nu(Gc, xi)
                        dt = probing.compute_deta_tilde(Uc, zt)
                        tt = probing.compute_tau_tilde(dt, Gc, xi, mu)
                        st = probing.compute_sigma_tilde(dt, Gc, xi, nu)
                        return dict(xi=xi, mu=mu, nu=nu, eta=probing.compute_eta(Gc, mu, nu), dt=dt, tt=tt, st=st,
                                    dxi=probing.compute_dxi_tilde(st, tt, Gc, mu, nu))
                    s0 = state(theta)
                    for kk in (np.ndindex(*K) if K else [()]):
                        dk = _elem(dirs, kk)
                        h = 1e-5
                        sp, sm = (state(cw.corewise_add(theta, cw.corewise_scale(dk, t))) for t in (h, -h))
                        fd = {key: tuple((p - m) / (2 * h) for p, m in zip(sp[key], sm[key])) for key in s0}
                        xid = probing.compute_dxi(dk[0], ww)
                        mud = probing.compute_sigma(dk[1], G, G, s0['xi'], xid, s0['mu'])
                        nud = probing.compute_tau(dk[1], G, G, s0['xi'], xid, s0['nu'])
                        dtd = probing.compute_deta_tilde_tangent(U, dk[0], zt)
                        ttd = probing.compute_tau_tilde_tangent(G, dk[1], s0['xi'], xid, s0['dt'], dtd, s0['mu'], mud,
                                                                s0['tt'])
                        std = probing.compute_sigma_tilde_tangent(G, dk[1], s0['xi'], xid, s0['dt'], dtd, s0['nu'],
                                                                  nud, s0['st'])
                        dxd = probing.compute_dxi_tilde_tangent(G, dk[1], s0['mu'], mud, s0['nu'], nud, s0['st'], std,
                                                                s0['tt'], ttd)
                        etad = probing.compute_deta(dk[1], G, G, s0['mu'], s0['nu'], mud, nud)
                        for label, got, ref in (('mu_dot', mud, fd['mu']), ('nu_dot', nud, fd['nu']),
                                                ('eta_dot', etad, fd['eta']), ('deta_tilde_dot', dtd, fd['dt']),
                                                ('tau_tilde_dot', ttd, fd['tt']), ('sigma_tilde_dot', std, fd['st']),
                                                ('dxi_tilde_dot', dxd, fd['dxi'])):
                            num = sum(float(np.sum((np.asarray(a) - np.asarray(b)) ** 2)) for a, b in zip(got, ref))
                            den = sum(float(np.sum(np.asarray(b) ** 2)) for b in ref)
                            if den == 0.0:                         # d = 1: the tildes and dxi~ vanish identically
                                self.assertEqual(num, 0.0, label)
                                continue
                            self.assertLess((num / den) ** 0.5, 1e-8, label)
                    # the factored TT assembly == the nine-term product rule (the full K stack)
                    xid = probing.compute_dxi(Ud, ww)
                    mud = probing.compute_sigma(Gd, G, G, s0['xi'], xid, s0['mu'])
                    nud = probing.compute_tau(Gd, G, G, s0['xi'], xid, s0['nu'])
                    dtd = probing.compute_deta_tilde_tangent(U, Ud, zt)
                    ttd = probing.compute_tau_tilde_tangent(G, Gd, s0['xi'], xid, s0['dt'], dtd, s0['mu'], mud, s0['tt'])
                    std = probing.compute_sigma_tilde_tangent(G, Gd, s0['xi'], xid, s0['dt'], dtd, s0['nu'], nud,
                                                              s0['st'])
                    args = (s0['st'], std, s0['tt'], ttd, s0['dt'], dtd, s0['xi'], xid, s0['mu'], mud, s0['nu'], nud, 1)
                    got = probing.assemble_tt_tangent_variations(*args)
                    ref = oracles.tt_tangent_assembly_nine_term(*args)
                    self.assertLess(_tree_relerr(got, ref), 1e-14)

    def test_apply_tt_assembly_is_the_three_term_product_rule(self):
        """The apply / entries two-hole's factored TT assembly (two finals) equals the unfactored three-term
        product rule of the transpose's ``mu (x) xi (x) sigma_hat``, read off the same intermediates."""
        rng = np.random.default_rng(54)
        for K, C in [((), ()), ((2,), (2,))]:
            with self.subTest(K=K, C=C):
                _, theta, ww, c, dirs = _plain_problem('apply', _D4, K, C, (5,), rng)
                U, G = theta
                xi = probing.compute_xi(U, ww)
                mu = probing.compute_mu(G, xi)
                xid = probing.compute_dxi(dirs[0], ww)
                mud = probing.compute_sigma(dirs[1], G, G, xi, xid, mu)
                sh = probing.compute_sigma_hat(G, xi, c)
                shd = probing.compute_tau(dirs[1], G, G, xi, xid, sh)
                got = bapply.t3_apply_corewise_hessian(c, ww, theta, dirs)[1]
                ref = oracles.apply_tt_tangent_assembly_three_term(mu, mud, xi, xid, sh, shd, 1)
                self.assertLess(_tree_relerr(got, ref), 1e-14)

    def test_from_sweep_reuses_the_model_sweep(self):
        """The manifold model's sweep at the frame (U, O, L, R) carries the corewise xi / mu of its base point
        (U, L): from_sweep with it -- or with the sweep at the corewise frame -- is the one-shot, bitwise; one sweep
        serves every direction."""
        rng = np.random.default_rng(55)
        for kind in _PLAIN_KINDS:
            hess, from_sweep, _, precompute = _plain_ops(kind)
            for C in [(), (2,)]:
                frame, theta, sample, z, _ = _plain_problem(kind, _D4, (), C, (5,), rng)
                for sweep_name, sweep in (('model', precompute(frame, sample)),
                                          ('corewise', precompute(fvc.t3_corewise_frame(theta), sample))):
                    for K in [(), (3,)]:
                        with self.subTest(kind=kind, C=C, sweep=sweep_name, K=K):
                            dirs = _directions(theta, K, rng)
                            one = hess(z, sample, theta, dirs)
                            two = from_sweep(z, sample, theta, dirs, sweep)
                            self.assertTrue(all(np.array_equal(p, q) for p, q in zip(one[0] + one[1], two[0] + two[1])))

    def test_stacked_equals_per_element(self):
        """Multi-axis W, K and C (and an empty W -- one sample): each stack element is the unstacked result."""
        rng = np.random.default_rng(56)
        for kind in _PLAIN_KINDS:
            hess = _plain_ops(kind)[0]
            for W, K, C in [((2, 3), (2, 2), (2, 3)), ((), (2,), (2,)), ((4,), (), ())]:
                with self.subTest(kind=kind, W=W, K=K, C=C):
                    _, theta, sample, z, dirs = _plain_problem(kind, _MINIMAL, K, C, W, rng)
                    got = hess(z, sample, theta, dirs)
                    for cc in np.ndindex(*C):
                        z_c = _plain_residual_at(kind, z, len(W), cc)
                        for kk in np.ndindex(*K):
                            ref = hess(z_c, sample, _elem(theta, cc), _elem(dirs, kk + cc))
                            self.assertLess(_relerr_tree(_elem(got, kk + cc), ref), 1e-13)

    def test_uniform_supercores(self):
        """The kernels are polymorphic: packed uniform supercores (clean padding) give the ragged real block."""
        import t3toolbox.backend.ut3_conversions as u3c
        import t3toolbox.backend.ut3_operations as u3o
        rng = np.random.default_rng(57)
        for kind in _PLAIN_KINDS:
            hess = _plain_ops(kind)[0]
            for name, struct in _PD_STRUCTURES.items():
                for K, C in [((2,), ()), ((2, 3), (2,))]:
                    with self.subTest(kind=kind, structure=name, K=K, C=C):
                        shape, tr, ttr = struct
                        d = len(shape)
                        _, theta, sample, z, dirs = _plain_problem(kind, struct, K, C, (7,), rng)
                        rag = hess(z, sample, theta, dirs)
                        N, n, r = max(shape), max(tr) + 1, max(ttr) + 1      # forced-larger padding everywhere
                        th_u = u3c.t3_to_ut3(theta, N=N, n=n, r=r, squash_tails=False)[:2]
                        dir_u = u3c.t3_to_ut3(dirs, N=N, n=n, r=r, squash_tails=False)[:2]
                        s_u = sample if kind == 'entries' else u3o.pack_vectors(sample, N)
                        z_u = u3o.pack_vectors(z, N) if kind == 'probe' else z
                        oU, oG = hess(z_u, s_u, th_u, dir_u)
                        self.assertEqual(oU.shape, (d,) + K + C + (n, N))
                        self.assertEqual(oG.shape, (d,) + K + C + (r, n, r))
                        real = ([oU[i][..., :tr[i], :shape[i]] for i in range(d)],
                                [oG[i][..., :ttr[i], :tr[i], :ttr[i + 1]] for i in range(d)])
                        self.assertLess(_relerr_tree(real, rag), 1e-13)

    def test_symmetry(self):
        """T6: the K-stack Gram matrix <dir_k, K dir_l> is symmetric, per C element."""
        rng = np.random.default_rng(58)
        for kind in _PLAIN_KINDS:
            hess = _plain_ops(kind)[0]
            for name, struct in _PD_STRUCTURES.items():
                for C in [(), (2,)]:
                    with self.subTest(kind=kind, structure=name, C=C):
                        n_dir = 4
                        _, theta, sample, z, dirs = _plain_problem(kind, struct, (n_dir,), C, (5,), rng)
                        Kd = hess(z, sample, theta, dirs)
                        c = 'abcdef'[:len(C)]
                        gram = sum(np.einsum('k%sx,l%sx->kl%s' % (c, c, c),
                                             a.reshape((n_dir,) + C + (-1,)), b.reshape((n_dir,) + C + (-1,)))
                                   for a, b in zip(dirs[0] + dirs[1], Kd[0] + Kd[1]))
                        asym = np.abs(gram - np.swapaxes(gram, 0, 1)).max()
                        self.assertGreater(np.abs(gram).max(), 1e-3)
                        self.assertLess(asym, 1e-12 * np.abs(gram).max())

    def test_residual_weights(self):
        """T11: the two-hole at the kind's ``omega^2 r`` is the curvature part of the Hessian of ``f(theta) = 1/2
        ||omega (.) (S(Phi(theta)) - y)||^2`` -- a central difference of ``grad f = J^T (omega^2 r)`` along theta_hat,
        minus the Gauss-Newton term ``J^T (omega^2 J theta_hat)``. Probe takes a per-MODE weight row (its only
        weight); apply / entries carry no weight at all (no mode or order axis), so for them this is the
        unweighted statement."""
        rng = np.random.default_rng(59)
        struct, W = _MINIMAL, (6,)
        cases = [('probe', 'per_mode', [0.4, 1.0, 2.5]), ('probe', 'none', None), ('apply', 'none', None),
                 ('entries', 'none', None)]
        for kind, label, w in cases:
            with self.subTest(kind=kind, weight=label):
                hess, _, one, _ = _plain_ops(kind)
                _, theta, sample, _, dirs = _plain_problem(kind, struct, (), (), W, rng)
                bkind = {'probe': bfit.ProbeKind(weight=w), 'apply': bfit.APPLY, 'entries': bfit.ENTRIES}[kind]
                fwd = lambda cores: bkind.point_forward(cores, sample)
                y = fwd(_unit_point(struct))
                w2 = lambda r: bkind._apply_weight(list(r) if kind == 'probe' else r, 2)
                sub = (lambda a, b: [p - q for p, q in zip(a, b)]) if kind == 'probe' else (lambda a, b: a - b)
                grad = lambda cores: one(w2(sub(fwd(cores), y)), sample, cores)
                h = 2e-5                            # a polynomial in h: central difference O(h^2)
                shift = lambda t: cw.corewise_add(theta, cw.corewise_scale(dirs, t))
                fd_hess = cw.corewise_scale(cw.corewise_sub(grad(shift(h)), grad(shift(-h))), 0.5 / h)
                gn = one(w2(fwd(oracles.doubled(theta, dirs))), sample, theta)          # J theta_hat = S(Y)
                curv_fd = cw.corewise_sub(fd_hess, gn)
                r = sub(fwd(theta), y)
                got = hess(w2(r), sample, theta, dirs)
                self.assertLess(_relerr_tree(got, curv_fd), 1e-6)
                if w is not None:
                    unweighted = hess(r, sample, theta, dirs)
                    self.assertGreater(_relerr_tree(unweighted, curv_fd), 1e-2)       # the weight is not a no-op


# ---------------------------------------------------------------------------------------------------
# The apply / entries derivative two-holes (S12)
# ---------------------------------------------------------------------------------------------------
_DERIV_KINDS = ('apply_derivatives', 'entries_derivatives')


def _deriv_ops(kind, order):
    """(two-hole, two-hole from a sweep, the one-hole -- the W-summed corewise derivative transpose, the sweep
    builder at a frame) for a derivative kind at ``order``, in the plain kinds' call shape: ``sample`` is ``(ww, pp)``
    (apply) or ``(index, pp)`` (entries), the residual a jet ``(order+1,)+W+C``."""
    if kind == 'apply_derivatives':
        return (lambda c, s, th, dd: sd.t3_apply_corewise_derivatives_hessian(c, s[0], s[1], th, dd, order),
                lambda c, s, th, dd, sw: sd.t3_apply_corewise_derivatives_hessian_from_sweep(
                    c, s[0], s[1], th, dd, order, sw),
                lambda c, s, th: sd.t3_apply_corewise_derivatives_transpose(c, s[0], s[1], th, order,
                                                                            sum_over_probes=True),
                lambda f, s: sd.tv_precompute_apply_frame_sweep_jets(f, s[0], s[1], order))
    return (lambda c, s, th, dd: sd.t3_entries_corewise_derivatives_hessian(c, s[0], s[1], th, dd, order),
            lambda c, s, th, dd, sw: sd.t3_entries_corewise_derivatives_hessian_from_sweep(
                c, s[0], s[1], th, dd, order, sw),
            lambda c, s, th: sd.t3_entries_corewise_derivatives_transpose(c, s[0], s[1], th, order,
                                                                          sum_over_probes=True),
            lambda f, s: sd.tv_precompute_entries_frame_sweep_jets(f, s[0], s[1], order))


def _deriv_sample(kind, shape, W, rng):
    """The paired sample: ``(ww, pp)`` or ``(index, pp)``."""
    if kind == 'entries_derivatives':
        first = np.stack([np.asarray(rng.integers(0, N, size=W)) for N in shape])    # (d,)+W
    else:
        first = [rng.standard_normal(W + (N,)) for N in shape]
    return first, [rng.standard_normal(W + (N,)) for N in shape]


def _deriv_problem(kind, struct, order, K, C, W, rng, raw=False):
    """theta (a frame's base point (U, L), or -- raw -- unconstrained corewise cores), the paired sample, the residual
    jet (no K) and the directions (K + C), as :py:func:`_pd_problem`."""
    if raw:
        theta = t3k.t3_corewise_randn(*struct, stack_shape=C)
        frame = None
    else:
        _, frame, _ = _point_and_frame(struct, C)
        theta = (tuple(frame[0]), tuple(frame[2]))
    sample = _deriv_sample(kind, struct[0], W, rng)
    c = rng.standard_normal((order + 1,) + W + C)
    return frame, theta, sample, c, _directions(theta, K, rng)


def _deriv_residual_at(c, nW, cc):
    """The residual jet of one C element ``cc``."""
    return c[(slice(None),) * (1 + nW) + cc]


class TestDerivativeKindsTwoHole(unittest.TestCase):
    """T5 / T6 / T11 for the apply / entries derivative two-holes (S12): ``sd.t3_{apply,entries}_corewise_derivatives_
    hessian[_from_sweep]`` at orders 0-3 against the block-sum oracle (``d`` = 4, 3, 1, slack frames, raw unsquashed
    cores, ``K = (2, 3)`` and ``C`` stacks) and an exact (Richardson) finite difference of the one-hole; **the shipped
    (recurrence) kernels against the dense-``trs`` reference twins** of ``_curvature_oracles.py``, whole and piece by
    piece, at orders 0-4 to 1e-14; order 0 is the plain two-hole; the containments ``entries_derivatives ==
    apply_derivatives`` at the one-hot vectors and ``apply_derivatives == probe_derivatives`` at the residual jets
    ``z~_i^(t) = alpha_i (c^(t) w_i + (t+1) c^(t+1) p_i)``; sweep reuse bitwise; stacked == per element; uniform
    supercores; symmetry; the per-order weight (``omega^2 r``)."""

    def setUp(self):
        np.random.seed(71)

    def test_against_the_block_sum_oracle(self):
        rng = np.random.default_rng(71)
        cases = [(name, struct, False) for name, struct in _PD_STRUCTURES.items()]
        cases += [('tucker_slack', _TUCKER_SLACK, False), ('tt_slack', _TT_SLACK, False),
                  ('raw_unsquashed', ((4, 5, 3), (2, 3, 2), (2, 2, 2, 3)), True),
                  ('raw_d1_unsquashed', ((5,), (3,), (2, 3)), True)]
        for kind in _DERIV_KINDS:
            for name, struct, raw in cases:
                for order in _PD_ORDERS:
                    hess, _, one, _ = _deriv_ops(kind, order)
                    for K, C in _PD_STACKS:
                        if name not in _PD_STRUCTURES and (K, C) != ((2,), (2,)):
                            continue                                    # the extra structures at one stack
                        with self.subTest(kind=kind, structure=name, order=order, K=K, C=C):
                            _, theta, sample, c, dirs = _deriv_problem(kind, struct, order, K, C, (5,), rng, raw)
                            got = hess(c, sample, theta, dirs)
                            for blk, ref_c in zip(got[0] + got[1], dirs[0] + dirs[1]):
                                self.assertEqual(blk.shape, ref_c.shape)
                            for kk in np.ndindex(*K):
                                for cc in np.ndindex(*C):
                                    th_c = _elem(theta, cc)
                                    ref = oracles.blocksum(one(_deriv_residual_at(c, 1, cc), sample,
                                                               oracles.doubled(th_c, _elem(dirs, kk + cc))), th_c)
                                    self.assertGreater(_norm_flat(ref), 1e-3)
                                    self.assertLess(_relerr_tree(_elem(got, kk + cc), ref), 1e-13)

    def test_against_a_finite_difference_of_the_one_hole(self):
        """The one-hole along ``theta + s theta_hat`` is a polynomial in ``s`` of degree ``<= 2d - 1`` (the jets are
        linear in ``Phi``, which is multilinear in the ``2d`` cores), so the Richardson-eliminated central difference
        is its derivative exactly up to rounding (as for the plain kinds): the two-hole matches to <= 1e-12."""
        rng = np.random.default_rng(72)
        for kind in _DERIV_KINDS:
            for name, struct in _PD_STRUCTURES.items():
                for order in (0, 2):
                    hess, _, one, _ = _deriv_ops(kind, order)
                    with self.subTest(kind=kind, structure=name, order=order):
                        _, theta, sample, c, dirs = _deriv_problem(kind, struct, order, (), (), (6,), rng)
                        got = hess(c, sample, theta, dirs)
                        d = len(struct[0])
                        hs = [0.1 / 2 ** k for k in range(d)]
                        weights = np.linalg.solve(np.array([[h ** (2 * j) for j in range(d)] for h in hs]).T,
                                                  np.eye(d)[0])
                        g = lambda t: one(c, sample, cw.corewise_add(theta, cw.corewise_scale(dirs, t)))
                        fd = None
                        for w, h in zip(weights, hs):
                            term = cw.corewise_scale(cw.corewise_sub(g(h), g(-h)), w * 0.5 / h)
                            fd = term if fd is None else cw.corewise_add(fd, term)
                        self.assertLess(_relerr_tree(got, fd), 1e-12)

    def test_production_equals_the_trs_twins(self):
        """The shipped kernels (banded recurrences / order scans; never the binomial tensor) == the dense-trs twins,
        whole and piece by piece (each piece fed the twin's own inputs), at orders 0..4, a single and a multi-axis
        sample stack W, d in {1, 3, 4}, to 1e-14 relative (not bitwise: the summation order differs; as for the
        probe-derivative twins). The dotted dxi^ is formed only on its two read rows (the twin computes all
        ``order+1``), so it is compared on those; the TT assembly (factored: two finals) against the unfactored
        three-term twin."""
        rng = np.random.default_rng(73)
        worst = 0.0
        for kind in _DERIV_KINDS:
            entries = kind == 'entries_derivatives'
            for name, struct in _PD_STRUCTURES.items():
                for order in range(5):
                    for (K, C), W in [(kc, (5,)) for kc in _PD_STACKS] + [(((2,), (2,)), (2, 3)),
                                                                          (((2, 3), ()), (2, 3))]:
                        with self.subTest(kind=kind, structure=name, order=order, K=K, C=C, W=W):
                            _, theta, sample, c, dirs = _deriv_problem(kind, struct, order, K, C, W, rng)
                            st = oracles.apply_derivatives_two_hole_states_trs(c, sample[0], sample[1], theta, dirs,
                                                                              order, entries=entries)
                            G, Gd = theta[1], dirs[1]
                            s2 = min(2, order + 1)
                            dxi_core = (sd._dxi_hat_tangent_jets_step_ragged((G[i], Gd[i], st['mu'][i], st['mud'][i],
                                                                              st['sigma_hat'][i],
                                                                              st['sigma_hat_d'][i]))[0]
                                        for i in range(len(G)))
                            dxi_got = tuple(dxi_core)
                            self.assertEqual([np.shape(g)[0] for g in dxi_got], [2] * len(dxi_got))  # two rows
                            if order == 0:
                                self.assertTrue(all(np.all(g[1] == 0) for g in dxi_got))      # row 1 is zero
                            asm = tuple(zip(*(sd._apply_tangent_assemble_step(
                                (st['xi'][i], st['xid'][i], st['mu'][i], st['mud'][i], st['sigma_hat'][i],
                                 st['sigma_hat_d'][i], st['dxi_hat_d'][i][:2], st['w'][i])) for i in range(len(G)))))
                            pieces = {
                                'whole': (_deriv_ops(kind, order)[0](c, sample, theta, dirs), (st['dU'], st['dG'])),
                                'sigma_hat': (sd.compute_sigma_hat_jets(G, st['xi'], c), st['sigma_hat']),
                                'sigma_hat_dot': (sd.compute_sigma_hat_tangent_jets(G, Gd, st['xi'], st['xid'],
                                                                                    st['sigma_hat']),
                                                  st['sigma_hat_d']),
                                'dxi_hat_dot': (tuple(g[:s2] for g in dxi_got),
                                                tuple(r[:s2] for r in st['dxi_hat_d'])),
                                'tucker_assembly': (asm[0], st['dU']),
                                'tt_assembly': (asm[1], st['dG']),
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

    def test_order_zero_is_the_plain_two_hole(self):
        """At ``order = 0`` the derivative two-holes ARE the plain apply / entries two-holes (to rounding: the jet
        kernels take their order-0 branches, the plain ones plain probing's kernels -- observed <= 3.2e-16)."""
        rng = np.random.default_rng(74)
        plain = {'apply_derivatives': lambda c, s, th, dd: bapply.t3_apply_corewise_hessian(c, s[0], th, dd),
                 'entries_derivatives': lambda c, s, th, dd: bentries.t3_entries_corewise_hessian(c, s[0], th, dd)}
        for kind in _DERIV_KINDS:
            for name, struct in _PD_STRUCTURES.items():
                for K, C in _PD_STACKS:
                    with self.subTest(kind=kind, structure=name, K=K, C=C):
                        _, theta, sample, c, dirs = _deriv_problem(kind, struct, 0, K, C, (5,), rng)
                        got = _deriv_ops(kind, 0)[0](c, sample, theta, dirs)
                        ref = plain[kind](c[0], sample, theta, dirs)
                        self.assertLess(_tree_relerr(got[0] + got[1], ref[0] + ref[1]), 1e-14)

    def test_containment(self):
        """The sampling containments, at the two-hole level (the functions ``theta -> <c, S(Phi(theta))>`` coincide, so
        their Hessians do): (a) ``entries_derivatives(c, index, P) == apply_derivatives(c, e_index, P)`` -- bitwise
        (a fiber slice IS the contraction with a one-hot, exactly); (b) ``apply_derivatives(c, w, P) ==
        probe_derivatives(z~)`` at ``z~_i^(t) = alpha_i (c^(t) w_i + (t+1) c^(t+1) p_i)`` (``c^(order+1) = 0``) for any
        ``alpha`` summing to one -- the apply jet is ``<w_i + s p_i, Y_i(s)>`` for every mode ``i``, whose Taylor
        coefficients pair ``c`` with the probe jets ``Y_i`` exactly so."""
        rng = np.random.default_rng(75)
        for name, struct in _PD_STRUCTURES.items():
            shape = struct[0]
            d = len(shape)
            for order in _PD_ORDERS:
                for K, C in _PD_STACKS:
                    with self.subTest(structure=name, order=order, K=K, C=C):
                        _, theta, (index, pp), c, dirs = _deriv_problem('entries_derivatives', struct, order, K, C,
                                                                        (6,), rng)
                        onehot = [np.eye(N)[index[i]] for i, N in enumerate(shape)]
                        ent = sd.t3_entries_corewise_derivatives_hessian(c, index, pp, theta, dirs, order)
                        app = sd.t3_apply_corewise_derivatives_hessian(c, onehot, pp, theta, dirs, order)
                        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(ent[0] + ent[1], app[0] + app[1])))
                        ww = [rng.standard_normal((6, N)) for N in shape]
                        alpha = rng.uniform(0.2, 1.0, d)
                        alpha = alpha / alpha.sum()
                        c_up = np.concatenate([c[1:], np.zeros_like(c[:1])], axis=0)        # c^(t+1), zero at order
                        t1 = np.arange(1, order + 2).reshape((order + 1,) + (1,) * (c.ndim - 1))
                        bc = lambda v: v.reshape((1, 6) + (1,) * len(C) + v.shape[-1:])     # W + (N,) -> jet layout
                        zt = [a * (c[..., None] * bc(w) + (t1 * c_up)[..., None] * bc(p))
                              for a, w, p in zip(alpha, ww, pp)]
                        app = sd.t3_apply_corewise_derivatives_hessian(c, ww, pp, theta, dirs, order)
                        prb = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, dirs, order)
                        self.assertLess(_tree_relerr(app[0] + app[1], prb[0] + prb[1]), 1e-13)

    def test_from_sweep_reuses_the_model_sweep(self):
        """The manifold model's lean jet sweep at the frame (U, O, L, R) carries the corewise xi / mu of its base point
        (U, L): from_sweep with it -- or with the sweep at the corewise frame -- is the one-shot, bitwise; one sweep
        serves every direction."""
        rng = np.random.default_rng(76)
        for kind in _DERIV_KINDS:
            for order in (0, 2):
                hess, from_sweep, _, precompute = _deriv_ops(kind, order)
                for C in [(), (2,)]:
                    frame, theta, sample, c, _ = _deriv_problem(kind, _D4, order, (), C, (5,), rng)
                    for sweep_name, sweep in (('model', precompute(frame, sample)),
                                              ('corewise', precompute(fvc.t3_corewise_frame(theta), sample))):
                        for K in [(), (3,)]:
                            with self.subTest(kind=kind, order=order, C=C, sweep=sweep_name, K=K):
                                dirs = _directions(theta, K, rng)
                                one = hess(c, sample, theta, dirs)
                                two = from_sweep(c, sample, theta, dirs, sweep)
                                self.assertTrue(all(np.array_equal(p, q) for p, q in zip(one[0] + one[1],
                                                                                         two[0] + two[1])))

    def test_stacked_equals_per_element(self):
        """Multi-axis W, K and C (and an empty W -- one sample): each stack element is the unstacked result."""
        rng = np.random.default_rng(77)
        for kind in _DERIV_KINDS:
            for order in (0, 2):
                hess = _deriv_ops(kind, order)[0]
                for W, K, C in [((2, 3), (2, 2), (2, 3)), ((), (2,), (2,)), ((4,), (), ())]:
                    with self.subTest(kind=kind, order=order, W=W, K=K, C=C):
                        _, theta, sample, c, dirs = _deriv_problem(kind, _MINIMAL, order, K, C, W, rng)
                        got = hess(c, sample, theta, dirs)
                        for cc in np.ndindex(*C):
                            c_c = _deriv_residual_at(c, len(W), cc)
                            for kk in np.ndindex(*K):
                                ref = hess(c_c, sample, _elem(theta, cc), _elem(dirs, kk + cc))
                                self.assertLess(_relerr_tree(_elem(got, kk + cc), ref), 1e-13)

    def test_uniform_supercores(self):
        """The kernels are polymorphic: packed uniform supercores (clean padding) give the ragged real block."""
        import t3toolbox.backend.ut3_conversions as u3c
        import t3toolbox.backend.ut3_operations as u3o
        rng = np.random.default_rng(78)
        for kind in _DERIV_KINDS:
            hess = _deriv_ops(kind, 2)[0]
            for name, struct in _PD_STRUCTURES.items():
                for K, C in [((2,), ()), ((2, 3), (2,))]:
                    with self.subTest(kind=kind, structure=name, K=K, C=C):
                        shape, tr, ttr = struct
                        d = len(shape)
                        _, theta, sample, c, dirs = _deriv_problem(kind, struct, 2, K, C, (7,), rng)
                        rag = hess(c, sample, theta, dirs)
                        N, n, r = max(shape), max(tr) + 1, max(ttr) + 1      # forced-larger padding everywhere
                        th_u = u3c.t3_to_ut3(theta, N=N, n=n, r=r, squash_tails=False)[:2]
                        dir_u = u3c.t3_to_ut3(dirs, N=N, n=n, r=r, squash_tails=False)[:2]
                        first = sample[0] if kind == 'entries_derivatives' else u3o.pack_vectors(sample[0], N)
                        oU, oG = hess(c, (first, u3o.pack_vectors(sample[1], N)), th_u, dir_u)
                        self.assertEqual(oU.shape, (d,) + K + C + (n, N))
                        self.assertEqual(oG.shape, (d,) + K + C + (r, n, r))
                        real = ([oU[i][..., :tr[i], :shape[i]] for i in range(d)],
                                [oG[i][..., :ttr[i], :tr[i], :ttr[i + 1]] for i in range(d)])
                        self.assertLess(_relerr_tree(real, rag), 1e-13)

    def test_symmetry(self):
        """T6: the K-stack Gram matrix <dir_k, K dir_l> is symmetric, per C element."""
        rng = np.random.default_rng(79)
        for kind in _DERIV_KINDS:
            for name, struct in _PD_STRUCTURES.items():
                for order in _PD_ORDERS:
                    hess = _deriv_ops(kind, order)[0]
                    for C in [(), (2,)]:
                        with self.subTest(kind=kind, structure=name, order=order, C=C):
                            n_dir = 4
                            _, theta, sample, c, dirs = _deriv_problem(kind, struct, order, (n_dir,), C, (5,), rng)
                            Kd = hess(c, sample, theta, dirs)
                            cl = 'abcdef'[:len(C)]
                            gram = sum(np.einsum('k%sx,l%sx->kl%s' % (cl, cl, cl),
                                                 a.reshape((n_dir,) + C + (-1,)), b.reshape((n_dir,) + C + (-1,)))
                                       for a, b in zip(dirs[0] + dirs[1], Kd[0] + Kd[1]))
                            asym = np.abs(gram - np.swapaxes(gram, 0, 1)).max()
                            self.assertGreater(np.abs(gram).max(), 1e-3)
                            self.assertLess(asym, 1e-12 * np.abs(gram).max())

    def test_residual_weights(self):
        """T11: with the kind's ``omega^2`` (a per-ORDER row: apply / entries have no mode axis) applied to the
        residual, the two-hole is the curvature part of the Hessian of ``f(theta) = 1/2 sum_t omega_t^2
        ||S^(t)(Phi(theta)) - y^(t)||^2`` -- a central difference of ``grad f = J^T (omega^2 r)`` along theta_hat, minus
        the Gauss-Newton term ``J^T (omega^2 J theta_hat)``; and the kind operation weights it so (a per-mode weight is
        the kinds' structural error)."""
        rng = np.random.default_rng(80)
        struct, order, W = _MINIMAL, 2, (6,)
        for kind in _DERIV_KINDS:
            hess, _, one, _ = _deriv_ops(kind, order)
            cls = bfit.ApplyDerivativesKind if kind == 'apply_derivatives' else bfit.EntriesDerivativesKind
            for label, w in (('per_order', [1.0, 0.5, 0.25]), ('none', None)):
                with self.subTest(kind=kind, weight=label):
                    _, theta, sample, _, dirs = _deriv_problem(kind, struct, order, (), (), W, rng)
                    bkind = cls(order=order, weight=w)
                    fwd = lambda cores: bkind.point_forward(cores, sample)
                    y = fwd(_unit_point(struct))
                    w2 = lambda r: bkind._apply_weight(r, 2)
                    grad = lambda cores: one(w2(fwd(cores) - y), sample, cores)
                    h = 2e-5                            # a polynomial in h: central difference O(h^2)
                    shift = lambda t: cw.corewise_add(theta, cw.corewise_scale(dirs, t))
                    fd_hess = cw.corewise_scale(cw.corewise_sub(grad(shift(h)), grad(shift(-h))), 0.5 / h)
                    gn = one(w2(fwd(oracles.doubled(theta, dirs))), sample, theta)        # J theta_hat = S(Y)
                    curv_fd = cw.corewise_sub(fd_hess, gn)
                    r = fwd(theta) - y
                    got = hess(w2(r), sample, theta, dirs)
                    self.assertLess(_relerr_tree(got, curv_fd), 1e-6)
                    frame = fvc.t3_corewise_frame(theta)
                    op = bkind.corewise_hessian(dirs, r, sample, frame, bkind.precompute(frame, sample), theta)
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(op[0] + op[1], got[0] + got[1])))
                    if w is not None:
                        self.assertGreater(_relerr_tree(hess(r, sample, theta, dirs), curv_fd), 1e-2)   # not a no-op
            with self.subTest(kind=kind, weight='per_mode'):
                with self.assertRaisesRegex(ValueError, 'no mode axis'):
                    bk = cls(order=order, weight=[[1.0, 0.5, 0.25], [0.3, 0.2, 0.1], [2.0, 1.0, 0.5]])
                    _, theta, sample, c, dirs = _deriv_problem(kind, struct, order, (), (), W, rng)
                    frame = fvc.t3_corewise_frame(theta)
                    bk.corewise_hessian(dirs, c, sample, frame, bk.precompute(frame, sample), theta)


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


# ---------------------------------------------------------------------------------------------------
# S4: the exact local model -- Problem.local_model(hessian='exact')
# ---------------------------------------------------------------------------------------------------
_MANIFOLD = bgeo.ManifoldGeometryOps()
_COREWISE = bgeo.CorewiseGeometryOps()


def _unit_point(struct, C=()):
    """A random T3 scaled to ``||X||_HS = 1`` per stack element (so finite differences are well scaled)."""
    x = t3.TuckerTensorTrain.randn(*struct, stack_shape=C)
    X = np.asarray(x.to_dense())
    nrm = np.asarray(np.sqrt(np.sum(X ** 2, axis=tuple(range(len(C), X.ndim)))))
    tk, tt = x.data
    return tuple(tk), tuple(tt[:-1]) + (tt[-1] / nrm[..., None, None, None],)


def _pd_fit(struct, order, W, rng, C=(), weight=None, geom=_MANIFOLD, regularizer=None, chunk_size=None):
    """A probe-derivative least-squares problem whose data come from ANOTHER unit point (plus noise), so the
    residual -- hence the curvature term -- is O(1), and the point ``x`` to linearize at."""
    x, other = _unit_point(struct, C), _unit_point(struct, C)
    ww = [rng.standard_normal(W + (N,)) for N in struct[0]]
    pp = [rng.standard_normal(W + (N,)) for N in struct[0]]
    data = [z + 0.1 * rng.standard_normal(z.shape) for z in sd.t3_probe_derivatives(ww, pp, other, order)]
    kind = bfit.ProbeDerivativesKind(order=order, weight=weight, chunk_size=chunk_size)
    return bopt.least_squares_problem(geom, kind, (ww, pp), data, regularizer=regularizer), x


def _unit_tangent(geom, frame, K=(), C=()):
    """A random tangent at ``frame`` (projected), scaled to unit coordinate norm when unstacked."""
    v = geom.project(frame, fvo.fv_variations_randn(fvo.fv_variation_shapes(frame), K + C))
    if K + C:
        return v
    return cw.corewise_scale(v, 1.0 / float(geom.inner(v, v)) ** 0.5)


def _second_derivative(f, h):
    """``f''(0)`` by the central second difference with one Richardson step (error O(h^4))."""
    f0 = f(0.0)
    d2 = lambda s: (f(s) - 2.0 * f0 + f(-s)) / (s * s)
    return (4.0 * d2(0.5 * h) - d2(h)) / 3.0


def _fd_step(lm):
    """A step for a finite difference along the retraction: the quadratic model is valid on a radius ~ the
    smallest lift singular value (the curvature carries 1/sigma), so h scales with it (unit X, unit p)."""
    data = lm.curvature.geometry_data
    sigma = 1.0 if data is None else min(float(np.min(data.sigma_min)), 1.0)
    return 0.05 * sigma


def _flat_inner(a, b):
    return float(sum(np.sum(np.asarray(p) * np.asarray(q)) for p, q in zip(a[0] + a[1], b[0] + b[1])))


@dc.dataclass(frozen=True, eq=False)
class _BlockSumApplyKind(bfit.ApplyKind):
    """APPLY with an ORACLE corewise_hessian (the block-sum, F5: the one-hole at the doubled-rank ``Y``, read
    back) -- so the exact local model's machinery can be checked on a second kind, independently of any
    shipped two-hole kernel."""

    def corewise_hessian(self, core_directions, r, ww, frame, sweep, base_point):
        Y = oracles.doubled(base_point, core_directions)
        return oracles.blocksum(bapply.t3_apply_corewise_transpose(r, ww, Y, sum_over_probes=True), base_point)


@dc.dataclass(frozen=True, eq=False)
class _FiveOperationApplyKind(bfit.ScalarOutputKind):
    """A user-written kind with the five operations a Gauss-Newton fit needs and NO corewise_hessian."""
    name = 'five_operation_apply'

    def precompute(self, frame, ww):
        return bapply.tv_precompute_apply_frame_sweep(frame, ww)

    def forward(self, v, ww, frame, sweep):
        return bapply.tv_apply_jacobian_from_sweep(v, ww, frame, sweep)

    def transpose(self, r, ww, frame, sweep):
        return bapply.tv_apply_transpose_from_sweep(r, ww, frame, sweep, sum_over_probes=True)

    def point_forward(self, x_cores, ww):
        return bapply.t3_apply(x_cores, ww)

    def take(self, sample, data, idx):
        return bfit._take_apply(sample, data, idx)

    def w_axes(self, sample):
        return sample[0].ndim - 1

    def n_measurements(self, sample):
        return int(np.prod(sample[0].shape[:-1]))


class _GaussNewtonOnlyGeometry:
    """A duck-typed geometry with the nine protocol members and none of the four exact-mode ones."""

    def __init__(self):
        self._g = _MANIFOLD

    def __getattr__(self, name):
        if name in bopt._EXACT_GEOMETRY_MEMBERS:
            raise AttributeError(name)
        return getattr(self._g, name)


class TestExactLocalModel(unittest.TestCase):
    """T7 (retraction FD), T8 (gauge laws), T9 (dense Lagrangian), T10 (regularized), the full-hvp symmetry and
    stacking, and the mode / damping contracts of ``Problem.local_model(hessian='exact')``."""

    def setUp(self):
        np.random.seed(31)

    def test_quadratic_form_is_the_second_derivative_along_the_retraction(self):
        """T7: <p, H p> == d^2/dt^2 f(tv_retract(t p)) at 0 -- an EXACT oracle, since the T3-SVD retraction is
        second order -- to the finite difference's resolution, which must also separate it from Gauss-Newton
        by a wide margin (so the curvature term is genuinely checked). Probe derivatives at orders 0-2 with the
        three weight shapes, both dampings, minimal and Tucker- / TT-slack frames; and an apply kind whose
        two-hole is the block-sum oracle."""
        rng = np.random.default_rng(31)
        d = 3
        cases = {'d3_per_order': (_MINIMAL, 2, [1.0, 0.5, 0.25]),
                 'd3_matrix': (_MINIMAL, 1, rng.uniform(0.3, 2.0, (d, 2))),
                 'd3_per_mode': (_MINIMAL, 1, [[1.0], [0.4], [2.0]]), 'd4': (_D4, 1, None), 'd2': (_D2, 0, None),
                 'rank1': (_RANK1, 2, None),
                 'tucker_slack': (_TUCKER_SLACK, 1, [1.0, 0.5]),          # the lift a one-sided pinv (F3)
                 'tt_slack': (_TT_SLACK, 1, None)}
        for name, (struct, order, weight) in cases.items():
            for cd in (0.0, None):
                with self.subTest(case=name, curvature_damping=cd):
                    prob, x = _pd_fit(struct, order, (12,), rng, weight=weight)
                    self._check_retraction_fd(prob, x, cd)
        # the apply kind (block-sum oracle two-hole): the model's machinery is kind-generic
        for cd in (0.0, None):
            with self.subTest(case='apply_blocksum', curvature_damping=cd):
                x, other = _unit_point(_MINIMAL), _unit_point(_MINIMAL)
                ww = [rng.standard_normal((40, N)) for N in _MINIMAL[0]]
                data = bapply.t3_apply(other, ww) + 0.1 * rng.standard_normal(40)
                prob = bopt.least_squares_problem(_MANIFOLD, _BlockSumApplyKind(), ww, data)
                self._check_retraction_fd(prob, x, cd)

    def _check_retraction_fd(self, prob, x, cd):
        lm = prob.local_model(x, hessian='exact', curvature_damping=cd)
        v = _unit_tangent(prob.geom, lm.frame)
        q = float(prob.geom.inner(v, lm.hvp(v)))
        q_gn = float(lm.gn_quadratic(v))
        self.assertLess(abs(float(lm.quadratic(v)) - q), 1e-12 * abs(q))
        fd = _second_derivative(lambda t: float(prob.objective(tv.tv_retract(lm.frame, cw.corewise_scale(v, t)))),
                                _fd_step(lm))
        scale = max(abs(q), abs(q_gn))
        self.assertLess(abs(fd - q), 1e-5 * scale)
        self.assertLess(abs(fd - q), 1e-3 * abs(q - q_gn))          # resolves the curvature term itself

    def test_gauge_laws(self):
        """T8: the normal curvature annihilates a lifted vertical direction (K_N L p_vert = 0, the Lagrangian null
        space), and the exact Hessian lives on gauged coordinates: hvp(p) == hvp(Pi p) and Pi hvp == hvp. (A
        vertical p_vert itself is NOT in the kernel of hvp under the orthogonal Pi: Pi p_vert != 0.)"""
        rng = np.random.default_rng(32)
        for name, struct in (('d3', _MINIMAL), ('d4', _D4), ('d2', _D2), ('tucker_slack', _TUCKER_SLACK),
                             ('tt_slack', _TT_SLACK)):
            with self.subTest(structure=name):
                prob, x = _pd_fit(struct, 2, (10,), rng, weight=[1.0, 0.5, 0.25])
                lm = prob.local_model(x, hessian='exact', curvature_damping=0.0)
                geom, frame, data = prob.geom, lm.frame, lm.curvature.geometry_data
                U, O, L, R = frame
                VV = [np.zeros(o.shape[-2:-1] + u.shape[-1:]) for o, u in zip(O, U)]
                HH = [np.zeros(h) for h in fvo.fv_variation_shapes(frame)[1]]
                for i in range(len(U)):
                    X = rng.standard_normal((O[i].shape[-2], U[i].shape[-2]))
                    VV[i] += X @ U[i]
                    HH[i] -= np.einsum('aib,ij->ajb', O[i], X)
                for i in range(len(U) - 1):
                    Y = rng.standard_normal((L[i].shape[-1], R[i + 1].shape[-3]))
                    HH[i] += np.einsum('iaj,jk->iak', L[i], Y)
                    HH[i + 1] -= np.einsum('jk,kbl->jbl', Y, R[i + 1])
                p_vert = (tuple(VV), tuple(HH))
                self.assertLess(np.linalg.norm(tv.tv_to_dense(frame, p_vert)), 1e-12 * _norm_flat(p_vert))
                theta_hat = geom.lift(frame, p_vert, data)
                K_r = lm.kind.corewise_hessian(theta_hat, lm.residual, lm.sample, frame, lm.sweep,
                                               geom.base_point(frame))
                K_N = cw.corewise_sub(K_r, geom.tangential_curvature(frame, theta_hat, data))
                self.assertGreater(_norm_flat(K_r), 1e-3)
                self.assertLess(_norm_flat(K_N), 1e-11 * _norm_flat(K_r))
                self.assertGreater(_norm_flat(geom.project(frame, p_vert)), 0.1 * _norm_flat(p_vert))   # F8

                p = fvo.fv_variations_randn(fvo.fv_variation_shapes(frame))              # UNGAUGED
                Hp = lm.hvp(p)
                self.assertLess(_relerr_tree(lm.hvp(geom.project(frame, p)), Hp), 1e-12)
                self.assertLess(_relerr_tree(geom.project(frame, Hp), Hp), 1e-12)
                pg = geom.project(frame, p)
                self.assertLess(_relerr_tree(lm.curvature_hvp(pg), lm.curvature_hvp(p)), 1e-12)
                # the quadratic forms project too: an ungauged p reads its gauged part, <Pi p, curvature_hvp(Pi p)>
                cq, cq_g = float(lm.curvature_quadratic(p)), float(lm.curvature_quadratic(pg))
                self.assertLess(abs(cq - cq_g), 1e-12 * abs(cq_g))
                self.assertLess(abs(cq_g - float(geom.inner(pg, lm.curvature_hvp(pg)))), 1e-12 * abs(cq_g))
                q, q_g = float(lm.quadratic(p)), float(lm.quadratic(pg))
                self.assertLess(abs(q - q_g), 1e-12 * abs(q_g))
                self.assertGreater(_relerr_tree(p, pg), 1e-2)                  # p is genuinely ungauged

    def test_against_the_dense_lagrangian_hessian(self):
        """T9: on gauged coordinates the exact Hessian is ``Pi L^T H_lag L Pi`` with ``H_lag`` the (exact, dense)
        core-coordinate Hessian of the Lagrangian ``f(Phi(theta)) - <grad f, Phi(theta)>`` at the base point, for
        the dense measurement matrix of the probe-derivative operator (columns = the operator on unit tensors)."""
        rng = np.random.default_rng(33)
        for name, struct, order, weight in (('d3', ((4, 5, 3), (2, 2, 2), (1, 2, 2, 1)), 2, [1.0, 0.5, 0.25]),
                                            ('d2', ((4, 5), (2, 3), (1, 2, 1)), 1, [[1.0, 0.7], [0.3, 1.5]])):
            with self.subTest(structure=name):
                shape = struct[0]
                prob, x = _pd_fit(struct, order, (4,), rng, weight=weight)
                kind, (ww, pp) = prob.kind, prob.sample
                lm = prob.local_model(x, hessian='exact', curvature_damping=0.0)
                frame = lm.frame
                omega = kind.weight                                               # canonical omega[mode, order]

                def measure(cores):                                               # omega (.) S(X), flattened
                    out = sd.t3_probe_derivatives(ww, pp, cores, order)
                    return np.concatenate([(z * omega[i if omega.shape[0] > 1 else 0].reshape(-1, 1, 1)).ravel()
                                           for i, z in enumerate(out)])
                cols = []
                for idx in np.ndindex(*shape):
                    unit = (tuple(np.eye(N)[j][None, :] for N, j in zip(shape, idx)),
                            tuple(np.ones((1, 1, 1)) for _ in shape))
                    cols.append(measure(unit))
                A = np.stack(cols, axis=1)
                y = A @ np.asarray(t3c.t3_to_dense(x)).ravel() - np.concatenate(
                    [(r * omega[i if omega.shape[0] > 1 else 0].reshape(-1, 1, 1)).ravel()
                     for i, r in enumerate(lm.residual)])                         # so that A X - y = omega (.) r
                theta = (tuple(frame[0]), tuple(frame[2]))
                G = tv.tv_to_dense(frame, lm.curvature.misfit_gradient)          # grad f, dense
                lf = fvo.fv_lift_factors(frame, 0.0)
                for _ in range(2):
                    p = fvo.fv_variations_randn(fvo.fv_variation_shapes(frame))
                    pg = tv.tv_orthogonal_gauge_projection(frame, p)
                    H_theta = oracles.dense_lagrangian_hessian(A, y, theta, tv.tv_corewise_lift(frame, pg, lf), G)
                    ref = tv.tv_orthogonal_gauge_projection(frame, tv.tv_corewise_lift_transpose(frame, H_theta, lf))
                    self.assertLess(_relerr_tree(lm.hvp(p), ref), 1e-10)

    def test_regularized(self):
        """T10: with ``IdentityRegularizer(lambda)`` (H_R = lambda Pi, already exact on the manifold) the exact model
        is the second derivative of ``f + rho`` along the retraction; the curvature uses the MISFIT gradient only."""
        rng = np.random.default_rng(34)
        for lam in (0.5, 20.0):
            with self.subTest(strength=lam):
                prob, x = _pd_fit(_MINIMAL, 1, (12,), rng, regularizer=bopt.IdentityRegularizer(lam))
                self._check_retraction_fd(prob, x, None)
                plain = dc.replace(prob, regularizer=None)
                lm, lm0 = prob.local_model(x, hessian='exact'), plain.local_model(x, hessian='exact')
                g_R = prob.regularizer.gradient(prob.geom, lm.frame)
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(sum(lm.curvature.misfit_gradient, ()),
                                                                         sum(lm0.curvature.misfit_gradient, ()))))
                self.assertLess(_relerr_tree(lm.gradient, cw.corewise_add(lm0.gradient, g_R)), 1e-15)
                v = _unit_tangent(prob.geom, lm.frame)
                self.assertLess(_relerr_tree(lm.hvp(v), cw.corewise_add(lm0.hvp(v), cw.corewise_scale(v, lam))), 1e-13)

    def test_symmetry(self):
        """T6 (the full hvp): the K-stack Gram matrix <p_k, H p_l> is symmetric, on the manifold and corewise."""
        rng = np.random.default_rng(35)
        for geom_name, geom in (('manifold', _MANIFOLD), ('corewise', _COREWISE)):
            for name, struct in (('d3', _MINIMAL), ('d4', _D4), ('tt_slack', _TT_SLACK),
                                 ('tucker_slack', _TUCKER_SLACK)):
                with self.subTest(geometry=geom_name, structure=name):
                    prob, x = _pd_fit(struct, 2, (8,), rng, geom=geom)
                    lm = prob.local_model(x, hessian='exact')
                    n_dir = 4
                    p = _unit_tangent(geom, lm.frame, K=(n_dir,))
                    Hp = lm.hvp(p)
                    gram = sum(a.reshape(n_dir, -1) @ b.reshape(n_dir, -1).T
                               for a, b in zip(p[0] + p[1], Hp[0] + Hp[1]))
                    self.assertLess(np.abs(gram - gram.T).max(), 1e-12 * np.abs(gram).max())
                    q = np.asarray(lm.quadratic(p))
                    self.assertEqual(q.shape, (n_dir,))
                    self.assertTrue(np.allclose(q, np.diag(gram), rtol=1e-12, atol=0))

    def test_stacked_equals_per_element(self):
        """K + C stacks: each element of a stacked exact model's hvp / quadratic / curvature_quadratic is the
        unstacked model's (a C-stacked point with its own data per element; a K-stack of tangents)."""
        rng = np.random.default_rng(36)
        for K, C in [((3,), ()), ((), (2,)), ((2,), (2,))]:
            with self.subTest(K=K, C=C):
                prob, x = _pd_fit(_MINIMAL, 1, (6,), rng, C=C)
                lm = prob.local_model(x, hessian='exact')
                p = _unit_tangent(prob.geom, lm.frame, K=K, C=C)
                Hp, q, qc = lm.hvp(p), np.asarray(lm.quadratic(p)), np.asarray(lm.curvature_quadratic(p))
                self.assertEqual(q.shape, K + C)
                self.assertEqual(qc.shape, K + C)
                for cc in np.ndindex(*C):
                    nW = 1
                    data_c = [z[(slice(None),) * (1 + nW) + cc] for z in prob.data]
                    lm_c = dc.replace(prob, data=data_c).local_model(_elem(x, cc), hessian='exact')
                    for kk in np.ndindex(*K):
                        p_kc = _elem(p, kk + cc)
                        self.assertLess(_relerr_tree(_elem(Hp, kk + cc), lm_c.hvp(p_kc)), 1e-12)
                        self.assertAlmostEqual(float(q[kk + cc]), float(lm_c.quadratic(p_kc)),
                                               delta=1e-11 * abs(q[kk + cc]))
                        self.assertAlmostEqual(float(qc[kk + cc]), float(lm_c.curvature_quadratic(p_kc)),
                                               delta=1e-11 * abs(qc[kk + cc]))

    def test_mode_is_derived_and_the_gradient_is_the_gauss_newton_one(self):
        """``is_exact`` is ``curvature is not None``; the exact gradient is the Gauss-Newton gradient bit for bit
        (it is the model's one misfit transpose, stored); GN-only methods mean the same in both modes; the
        curvature-only methods raise on a GN model; the modes are case-insensitive strings."""
        from unittest import mock
        rng = np.random.default_rng(37)
        prob, x = _pd_fit(_MINIMAL, 2, (8,), rng, weight=[1.0, 0.5, 0.25], regularizer=bopt.IdentityRegularizer(0.3))
        gn, ex = prob.local_model(x), prob.local_model(x, hessian='EXACT')
        self.assertFalse(gn.is_exact)
        self.assertIsNone(gn.curvature)
        self.assertTrue(ex.is_exact)
        self.assertIsInstance(ex.curvature, bopt.CurvatureState)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(sum(gn.gradient, ()), sum(ex.gradient, ()))))
        p = _unit_tangent(prob.geom, gn.frame)
        self.assertEqual(float(gn.gn_quadratic(p)), float(ex.gn_quadratic(p)))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(gn.jacobian(p), ex.jacobian(p))))
        self.assertEqual(float(gn.quadratic(p)), float(gn.gn_quadratic(p)))     # GN: quadratic is gn_quadratic
        self.assertAlmostEqual(float(ex.quadratic(p)), float(ex.gn_quadratic(p)) + float(ex.curvature_quadratic(p)),
                               delta=1e-12 * abs(float(ex.quadratic(p))))
        self.assertLess(_relerr_tree(ex.hvp(p), cw.corewise_add(gn.hvp(p), ex.curvature_hvp(p))), 1e-12)
        self.assertNotAlmostEqual(float(ex.quadratic(p)), float(gn.quadratic(p)), places=3)
        for method in ('curvature_hvp', 'curvature_quadratic'):
            with self.assertRaisesRegex(ValueError, 'needs an exact-mode model'):
                getattr(gn, method)(p)
        # the ONE misfit transpose: built once with the exact model, then read (GN transposes on demand)
        with mock.patch.object(bfit.ProbeDerivativesKind, 'transpose', autospec=True,
                               side_effect=bfit.ProbeDerivativesKind.transpose) as spy:
            ex2 = prob.local_model(x, hessian='exact')
            self.assertEqual(spy.call_count, 1)
            ex2.gradient, ex2.gradient
            self.assertEqual(spy.call_count, 1)
            gn2 = prob.local_model(x)
            self.assertEqual(spy.call_count, 1)
            gn2.gradient
            self.assertEqual(spy.call_count, 2)
        for bad in (True, None, 'newton', 'Gauss-Newton', ''):
            with self.subTest(hessian=bad):
                with self.assertRaisesRegex(ValueError, "hessian must be 'gauss_newton' or 'exact'"):
                    prob.local_model(x, hessian=bad)
        self.assertFalse(prob.local_model(x, hessian='Gauss_Newton').is_exact)

    def test_sigma_min_is_the_smallest_lift_singular_value(self):
        """``sigma_min`` (shape ``C``) is the minimum over the Tucker spectra ``sigma(S_i^T)`` AND the bond spectra
        ``sigma(T_i)`` of the frame's center factors. At d = 3 the end bonds' spectra ARE modes 0 / 2's Tucker
        spectra, so the bond-carried cases are d = 4 with the middle bond (no Tucker unfolding) weakened."""
        rng = np.random.default_rng(40)
        bond_wins, tucker_wins = 0, 0
        for name, struct, C, weak_bond in (
                ('d3_stacked', _MINIMAL, (2,), False), ('d4', _D4, (), False),
                ('tucker_slack', _TUCKER_SLACK, (), False),
                ('tt_slack', _TT_SLACK, (), False), ('tt_slack_stacked', _TT_SLACK, (3,), False),
                ('d4_weak_bond', _D4, (), True), ('d4_weak_bond_stacked', _D4, (2,), True)):
            with self.subTest(structure=name, C=C):
                prob, x = _pd_fit(struct, 0, (5,), rng, C=C)
                if weak_bond:                                         # damp one channel of the middle bond
                    G1 = x[1][1].copy()
                    G1[..., -1] *= 1e-2
                    x = (x[0], (x[1][0], G1) + x[1][2:])
                lm = prob.local_model(x, hessian='exact')
                _, STs, Ts = fvo.fv_center_factors(lm.frame)
                s_tucker = np.min([np.linalg.svd(S, compute_uv=False).min(axis=-1) for S in STs], axis=0)
                s_bond = np.min([np.linalg.svd(T, compute_uv=False).min(axis=-1) for T in Ts], axis=0)
                sigma_min = np.asarray(lm.curvature.geometry_data.sigma_min)
                self.assertEqual(sigma_min.shape, C)
                self.assertTrue(np.allclose(sigma_min, np.minimum(s_tucker, s_bond), rtol=1e-12, atol=0))
                bond_wins += int(np.sum(s_bond < s_tucker))
                tucker_wins += int(np.sum(s_tucker < s_bond))
        self.assertGreater(bond_wins, 0)                       # so a sigma_min that skips the bonds is caught
        self.assertGreater(tucker_wins, 0)

    def test_curvature_damping_reaches_the_lift_and_is_rejected_where_it_does_nothing(self):
        """The damping is a solver option handed to the geometry unread: a value reaches ``fv_lift_factors`` (the
        model's factors are a direct call's, bitwise), ``None`` is the geometry's DEFAULT_CURVATURE_DAMPING; a
        value with ``hessian='gauss_newton'`` or on a corewise geometry raises, as does a negative one."""
        rng = np.random.default_rng(38)
        prob, x = _pd_fit(_MINIMAL, 1, (8,), rng)
        frame = prob.local_model(x).frame
        for cd, ref in ((0.0, 0.0), (1e-3, 1e-3), (None, bgeo.DEFAULT_CURVATURE_DAMPING)):
            with self.subTest(curvature_damping=cd):
                lf = prob.local_model(x, hessian='exact', curvature_damping=cd).curvature.geometry_data.lift_factors
                direct = fvo.fv_lift_factors(frame, ref)
                for a, b in zip(lf.tucker_pinvs + lf.bond_pinvs + (lf.eps,),
                                direct.tucker_pinvs + direct.bond_pinvs + (direct.eps,)):
                    self.assertTrue(np.array_equal(a, b))
        self.assertEqual(bgeo.DEFAULT_CURVATURE_DAMPING, 1e-6)
        with self.assertRaisesRegex(ValueError, "curvature_damping is only meaningful with hessian='exact'"):
            prob.local_model(x, curvature_damping=1e-6)
        with self.assertRaisesRegex(ValueError, 'curvature_damping must be >= 0'):
            prob.local_model(x, hessian='exact', curvature_damping=-1e-6)
        cprob, cx = _pd_fit(_MINIMAL, 1, (8,), rng, geom=_COREWISE)
        self.assertEqual(cprob.local_model(cx, hessian='exact').curvature.geometry_data, ())
        for cd in (0.0, 1e-6):
            with self.assertRaisesRegex(ValueError, 'no effect on a corewise geometry'):
                cprob.local_model(cx, hessian='exact', curvature_damping=cd)
        # shared (tied) exact mode (S7): the damping reaches the TIED lift factors, built on the model's companion
        sgeom = _MANIFOLD.with_sharing((0, 0, 1), (5, 5, 7))
        sprob, sx = _pd_fit(((5, 5, 7), (2, 2, 2), (1, 2, 2, 1)), 1, (8,), rng, geom=sgeom)
        sx = ((sx[0][0], sx[0][0], sx[0][2]), sx[1])
        self.assertFalse(sprob.local_model(sx).is_exact)
        for cd, ref in ((0.0, 0.0), (1e-3, 1e-3), (None, bgeo.DEFAULT_CURVATURE_DAMPING)):
            with self.subTest(shared_curvature_damping=cd):
                slm = sprob.local_model(sx, hessian='exact', curvature_damping=cd)
                lf = slm.curvature.geometry_data.lift_factors
                self.assertIs(lf.shared_data, slm.geom_aux)                      # the model's companion, reused
                direct = sharing.fv_tied_lift_factors(slm.frame, ref, slm.geom_aux)
                for a, b in zip(lf.tucker_pinvs + lf.bond_pinvs + lf.tied_filters + (lf.eps,),
                                direct.tucker_pinvs + direct.bond_pinvs + direct.tied_filters + (direct.eps,)):
                    self.assertTrue(np.array_equal(a, b))
        with self.assertRaisesRegex(ValueError, 'curvature_damping must be >= 0'):
            sprob.local_model(sx, hessian='exact', curvature_damping=-1e-6)

    def test_the_kind_operation_weights_the_residual_and_passes_the_chunk_size(self):
        """``ProbeDerivativesKind.corewise_hessian`` is the shipped two-hole at ``omega^2 r`` with the model's sweep
        and the kind's own ``chunk_size`` (bitwise), so hvp's curvature block is exactly that kernel's."""
        from unittest import mock
        rng = np.random.default_rng(39)
        w = rng.uniform(0.3, 2.0, (3, 3))
        prob, x = _pd_fit(_MINIMAL, 2, (8,), rng, weight=w, chunk_size=7)
        lm = prob.local_model(x, hessian='exact')
        th = _raw_like_frame_cores(lm.frame, (2,), rng)
        got = lm.kind.corewise_hessian(th, lm.residual, lm.sample, lm.frame, lm.sweep, _MANIFOLD.base_point(lm.frame))
        ref = sd.t3_probe_corewise_derivatives_hessian(lm.kind._apply_weight(list(lm.residual), 2), *lm.sample,
                                                       _MANIFOLD.base_point(lm.frame), th, 2)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(got[0] + got[1], ref[0] + ref[1])))
        with mock.patch.object(sd, 't3_probe_corewise_derivatives_hessian_from_sweep',
                               wraps=sd.t3_probe_corewise_derivatives_hessian_from_sweep) as spy:
            lm.hvp(_unit_tangent(prob.geom, lm.frame))
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.kwargs['chunk_size'], 7)
        self.assertIs(spy.call_args.args[6], lm.sweep)                    # the model's sweep, reused


def _plain_fit(kind, struct, W, rng, C=(), weight=None, geom=_MANIFOLD, noise=0.1):
    """A plain-kind least-squares problem (probe / apply / entries) whose data come from ANOTHER unit point plus
    noise -- so the residual, hence the curvature term, is O(1) -- and the point ``x`` to linearize at."""
    x, other = _unit_point(struct, C), _unit_point(struct, C)
    shape = struct[0]
    if kind == 'entries':
        sample = np.stack([rng.integers(0, N, size=W) for N in shape])
    else:
        sample = [rng.standard_normal(W + (N,)) for N in shape]
    bkind = {'probe': bfit.ProbeKind(weight=weight), 'apply': bfit.APPLY, 'entries': bfit.ENTRIES}[kind]
    clean = bkind.point_forward(other, sample)
    data = ([z + noise * rng.standard_normal(z.shape) for z in clean] if kind == 'probe'
            else clean + noise * rng.standard_normal(clean.shape))
    return bopt.least_squares_problem(geom, bkind, sample, data), x


def _plain_fit_away_from_singular(test, kind, struct, W, seed, weight=None, cd=None):
    """:py:func:`_plain_fit` drawn until the exact model's smallest lift singular value exceeds 1e-3 (deterministic:
    the first passing seed from ``seed``) -- the retraction's finite difference resolves the curvature term only away
    from a near-singular lift (E1; ~4 % of random draws fall below, so a bare assert would fail on some reseed)."""
    for s in range(seed, seed + 20):
        np.random.seed(s)
        prob, x = _plain_fit(kind, struct, W, np.random.default_rng(s), weight=weight)
        lm = prob.local_model(x, hessian='exact', curvature_damping=cd)
        if float(np.min(lm.curvature.geometry_data.sigma_min)) > 1e-3:
            return prob, x
    test.fail('no draw with sigma_min > 1e-3 in 20 seeds from %d' % seed)


class TestPlainKindsExactLocalModel(unittest.TestCase):
    """``Problem.local_model(hessian='exact')`` for the plain kinds, through their shipped ``corewise_hessian``: T7
    (``<p, H p>`` is the second derivative along the T3-SVD retraction, resolving the curvature term itself), T8
    (``K_N L p_vert = 0``: the kind's two-hole and the T3-T3 term agree on a lifted vertical direction), T13 (on the
    corewise geometry the plain Hessian of ``f o Phi``: GN + the block-sum, and the derivative of the gradient), the
    full-hvp symmetry, stacks, and the kind operation (``omega^2 r``, the model's sweep reused, bitwise)."""

    def setUp(self):
        np.random.seed(61)

    def _check_retraction_fd(self, prob, x, cd):
        TestExactLocalModel._check_retraction_fd(self, prob, x, cd)

    def test_quadratic_form_is_the_second_derivative_along_the_retraction(self):
        """T7 per kind. The finite difference resolves the curvature term only where the quadratic model's validity
        radius (~ the smallest lift singular value, E1) leaves room for a step well above rounding, so each case
        draws its own point until it is not near-singular (``sigma_min > 1e-3``; :py:func:`_plain_fit_away_from_singular`):
        a random Tucker-slack point can have ``sigma_min ~ 2e-4``, where the second difference is rounding-dominated
        (observed 3.7e-5 of the scale on numpy 1.26) -- a property of the FD, not of the Hessian."""
        cases = {'d3': (_MINIMAL, None), 'd4': (_D4, None), 'd2': (_D2, None), 'tucker_slack': (_TUCKER_SLACK, None),
                 'tt_slack': (_TT_SLACK, None), 'd3_per_mode': (_MINIMAL, [1.0, 0.4, 2.0])}
        for k, kind in enumerate(_PLAIN_KINDS):
            for j, (name, (struct, weight)) in enumerate(cases.items()):
                if weight is not None and kind != 'probe':
                    continue                                     # only probe is weightable (per mode)
                for cd in (0.0, None):
                    with self.subTest(kind=kind, case=name, curvature_damping=cd):
                        prob, x = _plain_fit_away_from_singular(self, kind, struct, (40,),
                                                                610 + 20 * k + 2 * j + (cd is None), weight=weight,
                                                                cd=cd)
                        self._check_retraction_fd(prob, x, cd)

    def test_normal_curvature_annihilates_a_lifted_vertical_direction(self):
        """T8 through each kind: ``K_N L p_vert = 0`` -- the kind's two-hole at ``omega^2 r`` minus the geometry's
        tangential term vanishes on a lifted vertical (gauge) direction, while the two-hole alone does not."""
        rng = np.random.default_rng(62)
        for kind in _PLAIN_KINDS:
            for name, struct in (('d3', _MINIMAL), ('d4', _D4), ('tt_slack', _TT_SLACK)):
                with self.subTest(kind=kind, structure=name):
                    w = list(np.linspace(0.5, 2.0, len(struct[0]))) if kind == 'probe' else None   # per mode
                    prob, x = _plain_fit(kind, struct, (20,), rng, weight=w)
                    lm = prob.local_model(x, hessian='exact', curvature_damping=0.0)
                    geom, frame, data = prob.geom, lm.frame, lm.curvature.geometry_data
                    U, O, L, R = frame
                    VV = [np.zeros(o.shape[-2:-1] + u.shape[-1:]) for o, u in zip(O, U)]
                    HH = [np.zeros(h) for h in fvo.fv_variation_shapes(frame)[1]]
                    for i in range(len(U)):
                        X = rng.standard_normal((O[i].shape[-2], U[i].shape[-2]))
                        VV[i] += X @ U[i]
                        HH[i] -= np.einsum('aib,ij->ajb', O[i], X)
                    for i in range(len(U) - 1):
                        Y = rng.standard_normal((L[i].shape[-1], R[i + 1].shape[-3]))
                        HH[i] += np.einsum('iaj,jk->iak', L[i], Y)
                        HH[i + 1] -= np.einsum('jk,kbl->jbl', Y, R[i + 1])
                    p_vert = (tuple(VV), tuple(HH))
                    theta_hat = geom.lift(frame, p_vert, data)
                    K_r = lm.kind.corewise_hessian(theta_hat, lm.residual, lm.sample, frame, lm.sweep,
                                                   geom.base_point(frame))
                    K_N = cw.corewise_sub(K_r, geom.tangential_curvature(frame, theta_hat, data))
                    self.assertGreater(_norm_flat(K_r), 1e-3)
                    self.assertLess(_norm_flat(K_N), 1e-11 * _norm_flat(K_r))

    def test_corewise_is_the_plain_hessian(self):
        """T13: GN + the block-sum ``K_{A^T omega^2 r}``, and the (Richardson) derivative of the corewise gradient."""
        rng = np.random.default_rng(63)
        for kind in _PLAIN_KINDS:
            for name, struct in (('d3', _MINIMAL), ('d4', _D4), ('d1', _D1), ('tt_slack', _TT_SLACK)):
                with self.subTest(kind=kind, structure=name):
                    w = list(np.linspace(0.5, 2.0, len(struct[0]))) if kind == 'probe' else None   # per mode
                    prob, x = _plain_fit(kind, struct, (12,), rng, geom=_COREWISE, weight=w)
                    lm, gn = prob.local_model(x, hessian='exact'), prob.local_model(x)
                    self.assertEqual(lm.curvature.geometry_data, ())
                    p = (tuple(rng.standard_normal(u.shape) for u in x[0]),
                         tuple(rng.standard_normal(g.shape) for g in x[1]))
                    Hp = lm.hvp(p)
                    w2r = (lm.kind._apply_weight(list(lm.residual), 2) if kind == 'probe' else lm.residual)
                    K = oracles.blocksum(_plain_ops(kind)[2](w2r, prob.sample, oracles.doubled(x, p)), x)
                    self.assertLess(_relerr_tree(Hp, cw.corewise_add(gn.hvp(p), K)), 1e-12)
                    grad = lambda t: prob.local_model(cw.corewise_add(x, cw.corewise_scale(p, t))).gradient
                    d1 = lambda h: cw.corewise_scale(cw.corewise_sub(grad(h), grad(-h)), 0.5 / h)
                    h = 1e-3
                    fd = cw.corewise_sub(cw.corewise_scale(d1(0.5 * h), 4.0 / 3.0), cw.corewise_scale(d1(h), 1.0 / 3.0))
                    self.assertLess(_relerr_tree(Hp, fd), 1e-8)
                    self.assertGreater(_relerr_tree(gn.hvp(p), fd), 1e-3)            # the curvature term matters

    def test_symmetry_and_stacks(self):
        """T6 on the full hvp (manifold and corewise), and K + C stacks per element."""
        rng = np.random.default_rng(64)
        for kind in _PLAIN_KINDS:
            for geom_name, geom in (('manifold', _MANIFOLD), ('corewise', _COREWISE)):
                with self.subTest(kind=kind, geometry=geom_name):
                    prob, x = _plain_fit(kind, _D4, (16,), rng, geom=geom)
                    lm = prob.local_model(x, hessian='exact')
                    n_dir = 4
                    p = _unit_tangent(geom, lm.frame, K=(n_dir,))
                    Hp = lm.hvp(p)
                    gram = sum(a.reshape(n_dir, -1) @ b.reshape(n_dir, -1).T for a, b in zip(p[0] + p[1], Hp[0] + Hp[1]))
                    self.assertLess(np.abs(gram - gram.T).max(), 1e-12 * np.abs(gram).max())
                    self.assertTrue(np.allclose(np.asarray(lm.quadratic(p)), np.diag(gram), rtol=1e-12, atol=0))
            with self.subTest(kind=kind, stacks='K+C'):
                K, C = (2,), (2,)
                prob, x = _plain_fit(kind, _MINIMAL, (8,), rng, C=C)
                lm = prob.local_model(x, hessian='exact')
                p = _unit_tangent(prob.geom, lm.frame, K=K, C=C)
                Hp, q = lm.hvp(p), np.asarray(lm.quadratic(p))
                self.assertEqual(q.shape, K + C)
                for cc in np.ndindex(*C):
                    data_c = _plain_residual_at(kind, prob.data, 1, cc)
                    lm_c = dc.replace(prob, data=data_c).local_model(_elem(x, cc), hessian='exact')
                    for kk in np.ndindex(*K):
                        p_kc = _elem(p, kk + cc)
                        self.assertLess(_relerr_tree(_elem(Hp, kk + cc), lm_c.hvp(p_kc)), 1e-12)
                        self.assertAlmostEqual(float(q[kk + cc]), float(lm_c.quadratic(p_kc)),
                                               delta=1e-11 * abs(q[kk + cc]))

    def test_the_kind_operation_weights_the_residual_and_reuses_the_model_sweep(self):
        """Each plain kind's ``corewise_hessian`` is its shipped two-hole at ``omega^2 r`` (probe; apply / entries are
        unweighted) on the model's own sweep, bitwise, and the model's hvp calls it once with that sweep."""
        from unittest import mock
        rng = np.random.default_rng(65)
        modules = {'probe': (probing, 't3_probe_corewise_hessian_from_sweep'),
                   'apply': (bapply, 't3_apply_corewise_hessian_from_sweep'),
                   'entries': (bentries, 't3_entries_corewise_hessian_from_sweep')}
        for kind in _PLAIN_KINDS:
            with self.subTest(kind=kind):
                prob, x = _plain_fit(kind, _MINIMAL, (8,), rng, weight=[0.3, 1.0, 2.0] if kind == 'probe' else None)
                lm = prob.local_model(x, hessian='exact')
                th = _raw_like_frame_cores(lm.frame, (2,), rng)
                base = _MANIFOLD.base_point(lm.frame)
                got = lm.kind.corewise_hessian(th, lm.residual, lm.sample, lm.frame, lm.sweep, base)
                w2r = lm.kind._apply_weight(list(lm.residual), 2) if kind == 'probe' else lm.residual
                ref = _plain_ops(kind)[0](w2r, lm.sample, base, th)
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(got[0] + got[1], ref[0] + ref[1])))
                module, fname = modules[kind]
                with mock.patch.object(module, fname, wraps=getattr(module, fname)) as spy:
                    lm.hvp(_unit_tangent(prob.geom, lm.frame))
                self.assertEqual(spy.call_count, 1)
                self.assertIs(spy.call_args.args[4], lm.sweep)                    # the model's sweep, reused


def _deriv_kind(kind, order, weight=None):
    return (bfit.ApplyDerivativesKind if kind == 'apply_derivatives' else bfit.EntriesDerivativesKind)(
        order=order, weight=weight)


def _deriv_fit(kind, struct, order, W, rng, C=(), weight=None, geom=_MANIFOLD, noise=0.1, labels=None):
    """A derivative-kind least-squares problem (apply / entries derivatives) whose data come from ANOTHER unit point
    plus noise -- so the residual, hence the curvature term, is O(1) -- and the point ``x`` to linearize at; with
    ``labels``, both points tied (for a shared geometry)."""
    point = _unit_point if labels is None else (lambda s, cc: _shared_unit_point(s, labels, cc))
    x, other = point(struct, C), point(struct, C)
    sample = _deriv_sample(kind, struct[0], W, rng)
    bkind = _deriv_kind(kind, order, weight)
    clean = bkind.point_forward(other, sample)
    return bopt.least_squares_problem(geom, bkind, sample, clean + noise * rng.standard_normal(clean.shape)), x


def _deriv_fit_away_from_singular(test, kind, struct, order, W, seed, weight=None, cd=None):
    """:py:func:`_deriv_fit` drawn until the exact model's smallest lift singular value exceeds 1e-3 (deterministic:
    the first passing seed from ``seed``) -- the retraction's finite difference resolves the curvature term only away
    from a near-singular lift (E1; the S10 / S11 review's recommendation over a bare assert)."""
    for s in range(seed, seed + 20):
        np.random.seed(s)
        prob, x = _deriv_fit(kind, struct, order, W, np.random.default_rng(s), weight=weight)
        lm = prob.local_model(x, hessian='exact', curvature_damping=cd)
        if float(np.min(lm.curvature.geometry_data.sigma_min)) > 1e-3:
            return prob, x
    test.fail('no draw with sigma_min > 1e-3 in 20 seeds from %d' % seed)


class TestDerivativeKindsExactLocalModel(unittest.TestCase):
    """``Problem.local_model(hessian='exact')`` for the apply / entries derivative kinds (S12), through their shipped
    ``corewise_hessian``: T7 (``<p, H p>`` is the second derivative along the T3-SVD retraction, resolving the curvature
    term itself; orders 0-2, a per-order weight), T8 (``K_N L p_vert = 0`` through each kind), T13 (corewise: GN + the
    block-sum, and the derivative of the gradient), the full-hvp symmetry, stacks, the kind operation (``omega^2 r``,
    the model's sweep reused, bitwise), and one SHARED subcase per kind (the hvp is the derivative of the projected
    shared gradient; the output tied; the per-mode lift misses)."""

    def setUp(self):
        np.random.seed(81)

    def test_quadratic_form_is_the_second_derivative_along_the_retraction(self):
        cases = {'d3_order2': (_MINIMAL, 2, [1.0, 0.5, 0.25]), 'd4_order1': (_D4, 1, None), 'd2_order0': (_D2, 0, None),
                 'tucker_slack': (_TUCKER_SLACK, 1, [1.0, 0.5]), 'tt_slack': (_TT_SLACK, 1, None)}
        for k, kind in enumerate(_DERIV_KINDS):
            for j, (name, (struct, order, weight)) in enumerate(cases.items()):
                for cd in (0.0, None):
                    with self.subTest(kind=kind, case=name, curvature_damping=cd):
                        prob, x = _deriv_fit_away_from_singular(self, kind, struct, order, (40,),
                                                                810 + 100 * k + 10 * j + 5 * (cd is None),
                                                                weight=weight, cd=cd)
                        TestExactLocalModel._check_retraction_fd(self, prob, x, cd)

    def test_normal_curvature_annihilates_a_lifted_vertical_direction(self):
        """T8 through each kind: the kind's two-hole at ``omega^2 r`` minus the geometry's tangential term vanishes on a
        lifted vertical (gauge) direction, while the two-hole alone does not."""
        rng = np.random.default_rng(82)
        for kind in _DERIV_KINDS:
            for name, struct in (('d3', _MINIMAL), ('d4', _D4), ('tt_slack', _TT_SLACK)):
                with self.subTest(kind=kind, structure=name):
                    prob, x = _deriv_fit(kind, struct, 2, (20,), rng, weight=[1.0, 0.5, 0.25])
                    lm = prob.local_model(x, hessian='exact', curvature_damping=0.0)
                    geom, frame, data = prob.geom, lm.frame, lm.curvature.geometry_data
                    U, O, L, R = frame
                    VV = [np.zeros(o.shape[-2:-1] + u.shape[-1:]) for o, u in zip(O, U)]
                    HH = [np.zeros(h) for h in fvo.fv_variation_shapes(frame)[1]]
                    for i in range(len(U)):
                        X = rng.standard_normal((O[i].shape[-2], U[i].shape[-2]))
                        VV[i] += X @ U[i]
                        HH[i] -= np.einsum('aib,ij->ajb', O[i], X)
                    for i in range(len(U) - 1):
                        Y = rng.standard_normal((L[i].shape[-1], R[i + 1].shape[-3]))
                        HH[i] += np.einsum('iaj,jk->iak', L[i], Y)
                        HH[i + 1] -= np.einsum('jk,kbl->jbl', Y, R[i + 1])
                    theta_hat = geom.lift(frame, (tuple(VV), tuple(HH)), data)
                    K_r = lm.kind.corewise_hessian(theta_hat, lm.residual, lm.sample, frame, lm.sweep,
                                                   geom.base_point(frame))
                    K_N = cw.corewise_sub(K_r, geom.tangential_curvature(frame, theta_hat, data))
                    self.assertGreater(_norm_flat(K_r), 1e-3)
                    self.assertLess(_norm_flat(K_N), 1e-11 * _norm_flat(K_r))

    def test_corewise_is_the_plain_hessian(self):
        """T13: GN + the block-sum ``K_{A^T omega^2 r}``, and the (Richardson) derivative of the corewise gradient."""
        rng = np.random.default_rng(83)
        for kind in _DERIV_KINDS:
            for name, struct, order in (('d3', _MINIMAL, 2), ('d4', _D4, 1), ('d1', _D1, 1), ('tt_slack', _TT_SLACK, 0)):
                with self.subTest(kind=kind, structure=name, order=order):
                    w = [1.0] + [0.5] * order
                    prob, x = _deriv_fit(kind, struct, order, (12,), rng, geom=_COREWISE, weight=w)
                    lm, gn = prob.local_model(x, hessian='exact'), prob.local_model(x)
                    self.assertEqual(lm.curvature.geometry_data, ())
                    p = (tuple(rng.standard_normal(u.shape) for u in x[0]),
                         tuple(rng.standard_normal(g.shape) for g in x[1]))
                    Hp = lm.hvp(p)
                    one = _deriv_ops(kind, order)[2]
                    K = oracles.blocksum(one(lm.kind._apply_weight(lm.residual, 2), prob.sample,
                                             oracles.doubled(x, p)), x)
                    self.assertLess(_relerr_tree(Hp, cw.corewise_add(gn.hvp(p), K)), 1e-12)
                    grad = lambda t: prob.local_model(cw.corewise_add(x, cw.corewise_scale(p, t))).gradient
                    d1 = lambda h: cw.corewise_scale(cw.corewise_sub(grad(h), grad(-h)), 0.5 / h)
                    h = 1e-3
                    fd = cw.corewise_sub(cw.corewise_scale(d1(0.5 * h), 4.0 / 3.0), cw.corewise_scale(d1(h), 1.0 / 3.0))
                    self.assertLess(_relerr_tree(Hp, fd), 1e-8)
                    self.assertGreater(_relerr_tree(gn.hvp(p), fd), 1e-3)            # the curvature term matters

    def test_symmetry_and_stacks(self):
        """T6 on the full hvp (manifold and corewise), and K + C stacks per element."""
        rng = np.random.default_rng(84)
        for kind in _DERIV_KINDS:
            for geom_name, geom in (('manifold', _MANIFOLD), ('corewise', _COREWISE)):
                with self.subTest(kind=kind, geometry=geom_name):
                    prob, x = _deriv_fit(kind, _D4, 2, (16,), rng, geom=geom, weight=[1.0, 0.5, 0.25])
                    lm = prob.local_model(x, hessian='exact')
                    n_dir = 4
                    p = _unit_tangent(geom, lm.frame, K=(n_dir,))
                    Hp = lm.hvp(p)
                    gram = sum(a.reshape(n_dir, -1) @ b.reshape(n_dir, -1).T for a, b in zip(p[0] + p[1], Hp[0] + Hp[1]))
                    self.assertLess(np.abs(gram - gram.T).max(), 1e-12 * np.abs(gram).max())
                    self.assertTrue(np.allclose(np.asarray(lm.quadratic(p)), np.diag(gram), rtol=1e-12, atol=0))
            with self.subTest(kind=kind, stacks='K+C'):
                K, C = (2,), (2,)
                prob, x = _deriv_fit(kind, _MINIMAL, 1, (8,), rng, C=C, weight=[1.0, 0.5])
                lm = prob.local_model(x, hessian='exact')
                p = _unit_tangent(prob.geom, lm.frame, K=K, C=C)
                Hp, q = lm.hvp(p), np.asarray(lm.quadratic(p))
                self.assertEqual(q.shape, K + C)
                for cc in np.ndindex(*C):
                    lm_c = dc.replace(prob, data=_deriv_residual_at(prob.data, 1, cc)).local_model(
                        _elem(x, cc), hessian='exact')
                    for kk in np.ndindex(*K):
                        p_kc = _elem(p, kk + cc)
                        self.assertLess(_relerr_tree(_elem(Hp, kk + cc), lm_c.hvp(p_kc)), 1e-12)
                        self.assertAlmostEqual(float(q[kk + cc]), float(lm_c.quadratic(p_kc)),
                                               delta=1e-11 * abs(q[kk + cc]))

    def test_the_kind_operation_weights_the_residual_and_reuses_the_model_sweep(self):
        """Each derivative kind's ``corewise_hessian`` is its shipped two-hole at ``omega^2 r`` (a per-order row) on
        the model's own lean jet sweep, bitwise, and the model's hvp calls it once with that sweep."""
        from unittest import mock
        rng = np.random.default_rng(85)
        fname = {'apply_derivatives': 't3_apply_corewise_derivatives_hessian_from_sweep',
                 'entries_derivatives': 't3_entries_corewise_derivatives_hessian_from_sweep'}
        for kind in _DERIV_KINDS:
            with self.subTest(kind=kind):
                prob, x = _deriv_fit(kind, _MINIMAL, 2, (8,), rng, weight=[1.0, 0.3, 2.0])
                lm = prob.local_model(x, hessian='exact')
                th = _raw_like_frame_cores(lm.frame, (2,), rng)
                base = _MANIFOLD.base_point(lm.frame)
                got = lm.kind.corewise_hessian(th, lm.residual, lm.sample, lm.frame, lm.sweep, base)
                ref = _deriv_ops(kind, 2)[0](lm.kind._apply_weight(lm.residual, 2), lm.sample, base, th)
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(got[0] + got[1], ref[0] + ref[1])))
                unweighted = _deriv_ops(kind, 2)[0](lm.residual, lm.sample, base, th)
                self.assertGreater(_relerr_tree(unweighted, got), 1e-2)                 # the weight is not a no-op
                with mock.patch.object(sd, fname[kind], wraps=getattr(sd, fname[kind])) as spy:
                    lm.hvp(_unit_tangent(prob.geom, lm.frame))
                self.assertEqual(spy.call_count, 1)
                self.assertIs(spy.call_args.args[6], lm.sweep)                    # the model's sweep, reused

    def test_shared_subcase(self):
        """One SHARED subcase per kind (ragged; the group-ceiling structure, where the tied lift differs from the
        per-mode one): ``H v == d/dt P_x grad_sh f(R_sh(t v))`` at 0 (exact for any retraction), the output tied,
        Gauss-Newton and the per-mode-lift twin off."""
        rng = np.random.default_rng(86)
        struct, labels = _SH_CEILING
        geom = _MANIFOLD.with_sharing(labels, struct[0])
        for kind in _DERIV_KINDS:
            with self.subTest(kind=kind):
                prob, x = _deriv_fit(kind, struct, 2, (12,), rng, weight=[1.0, 0.5, 0.25], geom=geom, labels=labels)
                lm = prob.local_model(x, hessian='exact', curvature_damping=0.0)
                frame, aux = lm.frame, lm.geom_aux
                v = _unit_tangent(geom, frame)

                def grad_at(t):                    # P_x grad_sh f(R(t v)): the ambient gradient, projected at x
                    lm_y = prob.local_model(lm.retract(cw.corewise_scale(v, t)))
                    return tv.tv_project_t3_onto_tangent_space(frame, tv.tv_to_t3(lm_y.frame, lm_y.gradient),
                                                                shared_data=aux)

                h = 1e-4
                fd = cw.corewise_scale(cw.corewise_sub(grad_at(h), grad_at(-h)), 0.5 / h)
                Hv = lm.hvp(v)
                err = _relerr_tree(Hv, fd)
                self.assertLess(err, 1e-5)
                self.assertGreater(_relerr_tree(prob.local_model(x).hvp(v), fd), 1e2 * err)
                self.assertGreater(_relerr_tree(_per_mode_twin(lm).hvp(v), fd), 1e2 * err)
                self.assertLess(float(np.max(sharing.fv_tied_variations_residual(Hv, aux))), 1e-12)


class TestCorewiseExact(unittest.TestCase):
    """T13: on the corewise geometry the exact model is the plain Hessian of ``theta -> f(Phi(theta))``,
    ``J^T omega^2 J + K_{A^T omega^2 r}`` -- no lift, no tangential term."""

    def setUp(self):
        np.random.seed(41)

    def test_against_the_block_sum_and_a_finite_difference_of_the_gradient(self):
        rng = np.random.default_rng(41)
        for name, struct, order in (('d3', _MINIMAL, 2), ('d4', _D4, 1), ('d1', _D1, 1), ('tt_slack', _TT_SLACK, 0)):
            with self.subTest(structure=name):
                prob, x = _pd_fit(struct, order, (8,), rng, geom=_COREWISE, weight=None)
                lm, gn = prob.local_model(x, hessian='exact'), prob.local_model(x)
                self.assertEqual(lm.curvature.geometry_data, ())
                p = (tuple(rng.standard_normal(u.shape) for u in x[0]),
                     tuple(rng.standard_normal(g.shape) for g in x[1]))
                Hp = lm.hvp(p)
                # (a) GN + the block-sum K_{A^T r}
                zt = lm.kind._apply_weight(list(lm.residual), 2)
                K = oracles.blocksum(sd.t3_probe_corewise_derivatives_transpose(
                    zt, *prob.sample, oracles.doubled(x, p), order, sum_over_probes=True, chunk_size=None), x)
                self.assertLess(_relerr_tree(Hp, cw.corewise_add(gn.hvp(p), K)), 1e-12)
                # (b) the derivative of the corewise gradient along p (a polynomial: Richardson on a central difference)
                grad = lambda t: prob.local_model(cw.corewise_add(x, cw.corewise_scale(p, t))).gradient
                d1 = lambda h: cw.corewise_scale(cw.corewise_sub(grad(h), grad(-h)), 0.5 / h)
                h = 1e-3
                fd = cw.corewise_sub(cw.corewise_scale(d1(0.5 * h), 4.0 / 3.0), cw.corewise_scale(d1(h), 1.0 / 3.0))
                self.assertLess(_relerr_tree(Hp, fd), 1e-8)
                self.assertGreater(_relerr_tree(gn.hvp(p), fd), 1e-3)            # the curvature term is not negligible


    def test_shared_corewise(self):
        """Shared corewise (tied factors, Pi the per-group mean): the generic exact body is the Hessian of the tied
        objective, ``Pi H Pi`` on the copies -- the unshared exact model's hvp, projected -- its output is tied, and
        it is the derivative of the shared gradient along a tied direction (the additive retraction keeps tying)."""
        rng = np.random.default_rng(49)
        struct, labels = ((5, 5, 4), (2, 2, 2), (1, 2, 2, 1)), (0, 0, 1)
        geom = _COREWISE.with_sharing(labels, struct[0])
        for order, weight in ((1, [1.0, 0.5]), (2, None)):
            with self.subTest(order=order):
                prob, x = _pd_fit(struct, order, (8,), rng, geom=geom, weight=weight)
                x = ((x[0][0], x[0][0], x[0][2]), x[1])                          # a tied point
                lm = prob.local_model(x, hessian='exact')
                lm_untied = dc.replace(prob, geom=_COREWISE).local_model(x, hessian='exact')
                p = geom.project(lm.frame, (tuple(rng.standard_normal(u.shape) for u in x[0]),
                                            tuple(rng.standard_normal(g.shape) for g in x[1])))
                Hp = lm.hvp(p)
                self.assertTrue(np.array_equal(Hp[0][0], Hp[0][1]))              # the output is tied
                self.assertLess(_relerr_tree(Hp, geom.project(lm.frame, lm_untied.hvp(p))), 1e-13)
                grad = lambda t: prob.local_model(geom.retract(lm.frame, cw.corewise_scale(p, t))).gradient
                d1 = lambda h: cw.corewise_scale(cw.corewise_sub(grad(h), grad(-h)), 0.5 / h)
                h = 1e-3
                fd = cw.corewise_sub(cw.corewise_scale(d1(0.5 * h), 4.0 / 3.0), cw.corewise_scale(d1(h), 1.0 / 3.0))
                self.assertLess(_relerr_tree(Hp, fd), 1e-8)
                self.assertGreater(_relerr_tree(prob.local_model(x).hvp(p), fd), 1e-3)


class TestFlatSlotsModel(unittest.TestCase):
    """T20 (the model half): where the representable set is flat the normal curvature vanishes with no special
    case in the formula; the manifold's d = 1 shortcut skips the work and leaves the Gauss-Newton hvp."""

    def setUp(self):
        np.random.seed(42)

    def test_manifold_d1_is_gauss_newton(self):
        rng = np.random.default_rng(42)
        prob, x = _pd_fit(_D1, 2, (8,), rng, weight=[1.0, 0.5, 0.25])
        ex, gn = prob.local_model(x, hessian='exact'), prob.local_model(x)
        self.assertTrue(ex.is_exact)
        self.assertIsNone(ex.curvature.geometry_data)
        p = _unit_tangent(prob.geom, ex.frame, K=(3,))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(sum(ex.hvp(p), ()), sum(gn.hvp(p), ()))))
        self.assertTrue(np.array_equal(np.asarray(ex.quadratic(p)), np.asarray(gn.gn_quadratic(p))))
        self.assertTrue(all(np.all(a == 0.0) for a in sum(ex.curvature_hvp(p), ())))
        cq = np.asarray(ex.curvature_quadratic(p))
        self.assertEqual(cq.shape, (3,))
        self.assertTrue(np.all(cq == 0.0))

    def test_normal_curvature_vanishes_without_the_shortcut(self):
        """Maximal rank (through the model) and d = 1 (the shortcut bypassed: the curvature data built by the
        geometry's own body): ``||L^T K_N L Pi p|| <= 1e-12 ||L^T K_{A^T r} L Pi p||``."""
        rng = np.random.default_rng(43)
        flat = {'maximal': _MAXIMAL, 'maximal_d4': ((2, 2, 2, 2), (2, 2, 2, 2), (1, 2, 4, 2, 1)),
                'd1': _D1, 'd1_rank1': ((7,), (1,), (1, 1))}
        for name, struct in flat.items():
            for cd in (0.0, None):
                with self.subTest(structure=name, curvature_damping=cd):
                    prob, x = _pd_fit(struct, 1, (10,), rng)
                    lm = prob.local_model(x, hessian='exact', curvature_damping=cd)
                    geom, frame = prob.geom, lm.frame
                    data = lm.curvature.geometry_data
                    if data is None:                                             # d = 1: bypass the shortcut
                        data = bgeo._manifold_curvature_data(frame, lm.curvature.misfit_gradient,
                                                             bgeo.DEFAULT_CURVATURE_DAMPING if cd is None else cd)
                    p = _unit_tangent(geom, frame)
                    theta_hat = geom.lift(frame, p, data)
                    K_r = lm.kind.corewise_hessian(theta_hat, lm.residual, lm.sample, frame, lm.sweep,
                                                   geom.base_point(frame))
                    K_N = cw.corewise_sub(K_r, geom.tangential_curvature(frame, theta_hat, data))
                    LT_KN, LT_Kr = geom.lift_transpose(frame, K_N, data), geom.lift_transpose(frame, K_r, data)
                    self.assertGreater(_norm_flat(LT_Kr), 1e-3)
                    self.assertLess(_norm_flat(LT_KN), 1e-12 * _norm_flat(LT_Kr))
                    if lm.curvature.geometry_data is not None:                   # the model agrees
                        self.assertLess(_norm_flat(lm.curvature_hvp(p)), 1e-12 * _norm_flat(LT_Kr))

    def test_corewise_d1_runs_exact(self):
        rng = np.random.default_rng(44)
        prob, x = _pd_fit(_D1, 1, (8,), rng, geom=_COREWISE)
        lm = prob.local_model(x, hessian='exact')
        self.assertEqual(lm.curvature.geometry_data, ())
        p = (tuple(rng.standard_normal(u.shape) for u in x[0]), tuple(rng.standard_normal(g.shape) for g in x[1]))
        self.assertGreater(_relerr_tree(lm.hvp(p), prob.local_model(x).hvp(p)), 1e-3)


class TestRequiredSixthOperation(unittest.TestCase):
    """T23: ``corewise_hessian`` is a required sixth ``SamplingKind`` operation, checked structurally (by the
    stdlib abstractmethod marker) in ``Problem.local_model(hessian='exact')`` before any work; Gauss-Newton never
    consults it. Every built-in kind implements it, and every uniform kind on its own class (no window stub remains)."""

    def setUp(self):
        np.random.seed(45)

    def test_a_five_operation_kind_works_for_gauss_newton_and_is_rejected_for_exact(self):
        rng = np.random.default_rng(45)
        kind = _FiveOperationApplyKind()
        self.assertIs(type(kind).corewise_hessian, bfit.SamplingKind.corewise_hessian)       # the identity form
        self.assertTrue(getattr(type(kind).corewise_hessian, '__isabstractmethod__', False))  # the marker form
        x, other = _unit_point(_MINIMAL), _unit_point(_MINIMAL)
        ww = [rng.standard_normal((30, N)) for N in _MINIMAL[0]]
        data = bapply.t3_apply(other, ww)
        prob = bopt.least_squares_problem(_MANIFOLD, kind, ww, data)
        ref = bopt.least_squares_problem(_MANIFOLD, bfit.APPLY, ww, data)
        p = _unit_tangent(_MANIFOLD, prob.local_model(x).frame)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(sum(prob.local_model(x).hvp(p), ()),
                                                                 sum(ref.local_model(x).hvp(p), ()))))
        _, stats = bopt.newton_cg(prob, x, max_newton=3)                                    # GN runs
        self.assertEqual(stats['newton_steps'], 3)
        with self.assertRaisesRegex(ValueError, "_FiveOperationApplyKind does not implement corewise_hessian"):
            prob.local_model(x, hessian='exact')
        with self.assertRaisesRegex(NotImplementedError, 'which hessian=.exact. needs'):
            kind.corewise_hessian(None, None, None, None, None, None)

    def test_the_marker_and_the_identity_forms_agree_on_every_ragged_kind(self):
        kinds = [bfit.APPLY, bfit.ENTRIES, bfit.PROBE, bfit.ProbeKind(weight=[1.0, 2.0, 0.5]),
                 bfit.ApplyDerivativesKind(order=1), bfit.EntriesDerivativesKind(order=1),
                 bfit.ProbeDerivativesKind(order=1), _BlockSumApplyKind(), _FiveOperationApplyKind()]
        for kind in kinds:
            with self.subTest(kind=type(kind).__name__):
                identity = type(kind).corewise_hessian is bfit.SamplingKind.corewise_hessian
                marker = getattr(type(kind).corewise_hessian, '__isabstractmethod__', False)
                self.assertEqual(identity, marker)
        for implemented in (bfit.ProbeDerivativesKind, bfit.ProbeKind, bfit.ApplyKind, bfit.EntriesKind,
                            bfit.ApplyDerivativesKind, bfit.EntriesDerivativesKind):        # S4, S10, S11, S12: all six
            self.assertFalse(getattr(implemented.corewise_hessian, '__isabstractmethod__', False))
            self.assertIsNot(implemented.corewise_hessian, bfit.SamplingKind.corewise_hessian)
            self.assertIn('corewise_hessian', vars(implemented))                       # its own, not inherited
        # and instantiation is unaffected (a plain dataclass, not an ABC)
        self.assertIsInstance(_FiveOperationApplyKind(), bfit.SamplingKind)

    def test_no_uniform_stub_remains(self):
        """After S12 every uniform kind implements ``corewise_hessian`` ON ITS OWN CLASS: the S4-S12 window stub and its
        slice table are gone, the uniform-layer mixin (which precedes the ragged kind in the MRO) defines no
        ``corewise_hessian``, and none is marked abstract -- so each uniform kind's method is its own masked override,
        never the ragged (mask-free) one. (Mutation check, verified by hand: deleting a uniform override makes that kind
        inherit the ragged method -- the own-class and not-the-ragged assertions below fail.) Every uniform kind builds
        an exact local model."""
        import t3toolbox.backend.uniform_fitting as uf
        import t3toolbox.uniform_tucker_tensor_train as ut3
        self.assertFalse(hasattr(uf, '_UNIFORM_HESSIAN_SLICE'))
        self.assertFalse(hasattr(uf, '_UniformKind'))
        self.assertNotIn('corewise_hessian', vars(uf._UniformLayerMixin))
        for cls, ragged in ((uf.UniformApplyKind, bfit.ApplyKind), (uf.UniformEntriesKind, bfit.EntriesKind),
                            (uf.UniformProbeKind, bfit.ProbeKind),
                            (uf.UniformApplyDerivativesKind, bfit.ApplyDerivativesKind),
                            (uf.UniformEntriesDerivativesKind, bfit.EntriesDerivativesKind),
                            (uf.UniformProbeDerivativesKind, bfit.ProbeDerivativesKind)):
            with self.subTest(kind=cls.__name__):
                mro = cls.__mro__
                self.assertLess(mro.index(uf._UniformLayerMixin), mro.index(ragged))
                self.assertIn('corewise_hessian', vars(cls))                         # its own masked override
                self.assertIsNot(cls.corewise_hessian, ragged.corewise_hessian)      # never the ragged (mask-free) one
                self.assertFalse(getattr(cls.corewise_hessian, '__isabstractmethod__', False))
        rng = np.random.default_rng(46)
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain(*_unit_point(_MINIMAL)))
        ww = [rng.standard_normal((10, N)) for N in _MINIMAL[0]]
        pp = [rng.standard_normal((10, N)) for N in _MINIMAL[0]]
        index = np.stack([rng.integers(0, N, size=10) for N in _MINIMAL[0]])
        sc = (ux0.data[0], ux0.data[1])
        other = _unit_point(_MINIMAL)
        for name, sample, order in (('probe', ww, None), ('apply', ww, None), ('entries', index, None),
                                    ('apply_derivatives', (ww, pp), 1), ('entries_derivatives', (index, pp), 1),
                                    ('probe_derivatives', (ww, pp), 1)):
            with self.subTest(builds=name):
                rk = ({'probe': bfit.PROBE, 'apply': bfit.APPLY, 'entries': bfit.ENTRIES}[name] if order is None else
                      {'apply_derivatives': bfit.ApplyDerivativesKind, 'entries_derivatives': bfit.EntriesDerivativesKind,
                       'probe_derivatives': bfit.ProbeDerivativesKind}[name](order=order))
                prob = uf.uniform_least_squares_problem('manifold', name, ux0, sample, rk.point_forward(other, sample),
                                                        order=order)
                self.assertTrue(prob.local_model(sc, hessian='exact').is_exact)

    def test_a_geometry_without_the_exact_members_works_for_gauss_newton_and_is_rejected_for_exact(self):
        rng = np.random.default_rng(47)
        prob, x = _pd_fit(_MINIMAL, 1, (8,), rng)
        duck = dc.replace(prob, geom=_GaussNewtonOnlyGeometry())
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(sum(duck.local_model(x).gradient, ()),
                                                                 sum(prob.local_model(x).gradient, ()))))
        with self.assertRaisesRegex(ValueError, 'does not implement curvature_precompute, lift, lift_transpose, '
                                                'tangential_curvature'):
            duck.local_model(x, hessian='exact')


# ---------------------------------------------------------------------------------------------------
# S5: exact Newton end to end -- newton_cg(hessian='exact') with its trust region (T16, T21, T25)
# ---------------------------------------------------------------------------------------------------
_EPS64 = float(np.finfo(np.float64).eps)


def _data_norm(problem):
    return float(problem.kind.sumsq(problem.data, problem.kind.w_axes(problem.sample))) ** 0.5


def _newton_steps(problem, stats):
    """Per accepted step of a trust-region run: (e_k, e_{k+1}, ρ, |pred|/res), with e = ‖g‖/‖g₀‖ and
    res = ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|) the objective's resolution at the step's point."""
    y = _data_norm(problem)
    h = stats['history']
    out = []
    for row, nxt in zip(h, h[1:]):
        if row['tr_accepted']:
            pred = row['slope'] + 0.5 * row['pHp']
            res = _EPS64 * (y * (2.0 * row['misfit']) ** 0.5 + abs(row['objective']))
            out.append((row['gnorm'] / row['g0norm'], nxt['gnorm'] / nxt['g0norm'], row['rho'], abs(pred) / res))
    return out


def _t16_problem(seed):
    """T16's problem (F22, re-calibrated on probe derivatives in S5): a rank-2 fit of a rank-4 target on
    (6,6,6) -- misspecified, so the residual and with it the curvature term are O(1) and Gauss-Newton is
    linear -- probe derivatives of order 1, |W| = 200, 1 % noise; the start is 6 Gauss-Newton steps from a
    random point (more would leave only one or two steps between the start and the objective's resolution)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    shape = (6, 6, 6)
    target = t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1))
    ww = [rng.standard_normal((200, N)) for N in shape]
    pp = [rng.standard_normal((200, N)) for N in shape]
    kind = bfit.ProbeDerivativesKind(order=1)
    data = [z + 0.01 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in kind.point_forward(target.data, (ww, pp))]
    problem = bopt.least_squares_problem(_MANIFOLD, kind, (ww, pp), data)
    x, _ = bopt.newton_cg(problem, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data,
                          max_newton=6, gtol_rel=0.0)
    return problem, x


class TestExactNewtonRate(unittest.TestCase):
    """T16 (F22, F24, F29): the local rate of newton_cg(hessian='exact') on a misspecified fit, against
    Gauss-Newton's linear rate. e_k = ‖g_k‖/‖g_0‖ (the warm-start reference); the asymptotic steps are those
    with e_k < 1e-1 AND |pred| > 1e3·res (the n15 criterion: it excludes the steps that are dominated by the
    objective's floating-point resolution, where no rate is measurable). Constants run-and-paste (S5):

    * power 1.0: e_{k+1}/e_k² = 0.234, 0.768 (seed 1); 0.538, 1.02, 0.813 (seed 2) -- C = 2;
    * power 0.5: e_{k+1}/e_k^1.5 = 0.605, 0.754, 0.706 (seed 1); 0.320, 0.407, 0.692 (seed 2) -- C = 2,
      while e_{k+1}/e_k² reaches 18.2 / 5.95 (superlinear, not q-quadratic);
    * |1 - ρ| <= 1.6e-3 on every such step; both stop 'objective_floor' within 7 rows;
    * Gauss-Newton + line search: ratios e_{k+1}/e_k in [0.45, 0.68] (linear) in e in [1e-10, 1e-1]."""

    def _run(self, seed, power):
        problem, x = _t16_problem(seed)
        _, stats = bopt.newton_cg(problem, x, max_newton=30, gtol_rel=1e-12, g0_reference='warm_start',
                                  hessian='exact', cg_forcing_power=power)
        self.assertIn(stats['stop_reason'], ('gradient', 'objective_floor'))
        window = [s for s in _newton_steps(problem, stats) if s[0] < 1e-1 and s[3] > 1e3]
        for e_k, e_next, rho, _ in window:
            self.assertLess(abs(1.0 - rho), 0.1)
        return window

    def test_power_one_is_q_quadratic(self):
        """Power 1.0 must be q-QUADRATIC: its job is to catch any bug that loses the exact Hessian's asymptotic rate
        (I-34; a wrong curvature term still converges, but only linearly)."""
        for seed in (1, 2):
            with self.subTest(seed=seed):
                window = self._run(seed, 1.0)
                self.assertGreaterEqual(len(window), 2)
                for e_k, e_next, _, _ in window:
                    self.assertLessEqual(e_next, 2.0 * e_k ** 2)

    def test_power_half_is_superlinear_of_order_one_and_a_half(self):
        """Power 0.5 -- the production default -- must be SUPERLINEAR: its job is to stress the method end to end on
        the default the users run (I-34), not to pin the asymptotic order."""
        for seed in (1, 2):
            with self.subTest(seed=seed):
                window = self._run(seed, 0.5)
                self.assertGreaterEqual(len(window), 2)
                for e_k, e_next, _, _ in window:
                    self.assertLessEqual(e_next, 2.0 * e_k ** 1.5)
                self.assertTrue(any(e_next > 2.0 * e_k ** 2 for e_k, e_next, _, _ in window))   # not q-quadratic

    def test_gauss_newton_control_is_linear(self):
        for seed in (1, 2):
            with self.subTest(seed=seed):
                problem, x = _t16_problem(seed)
                _, stats = bopt.newton_cg(problem, x, max_newton=25, gtol_rel=1e-12, g0_reference='warm_start')
                e = [row['gnorm'] / row['g0norm'] for row in stats['history']]
                ratios = [b / a for a, b in zip(e, e[1:]) if 1e-10 < a < 1e-1]
                self.assertGreaterEqual(len(ratios), 10)
                for r in ratios:
                    self.assertTrue(0.3 < r < 0.9, ratios)


class TestExactNewtonRestart(unittest.TestCase):
    """T21 (E2 / F19 / F26, ragged): exact newton_cg from a zero-padded continuation restart -- the frame
    is rank-deficient there (σ_min = 0 on the padded slots, the damped lift keeps it finite) -- converges on the
    gradient test with at most two rejections. c24's template on probe derivatives (order 2, ω = [1, .5, .25],
    5 % noise, target rank 4, level 1 at rank 2 fitted to 1e-10, restart at rank 3); |W| = 60 here.
    Observed (S5, defaults: zero reference, power 0.5): 8 outer steps, 0 rejected, against 28 for Gauss-Newton +
    Armijo; at c24's |W| = 120: 9 / 0 (zero reference), 10 outer and 28 CG iterations (warm start; F19/F26:
    10 / 28)."""

    def test_zero_padded_restart_converges(self):
        rng = np.random.default_rng(7)
        np.random.seed(7)
        shape, W = (6, 6, 6), 60
        target = t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1))
        ww = [rng.standard_normal((W, N)) for N in shape]
        pp = [rng.standard_normal((W, N)) for N in shape]
        kind = bfit.probe_derivatives_kind(2, np.array([1.0, 0.5, 0.25]))
        data = [z + 0.05 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
                for z in kind.point_forward(target.data, (ww, pp))]
        problem = bopt.least_squares_problem(_MANIFOLD, kind, (ww, pp), data)
        x1, _ = bopt.newton_cg(problem, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data,
                               max_newton=80, gtol_rel=1e-10)
        restart = t3.TuckerTensorTrain(*x1).resize(shape, (3, 3, 3), (1, 3, 3, 1)).data
        _, stats = bopt.newton_cg(problem, restart, max_newton=40, hessian='exact')
        h = stats['history']
        self.assertEqual(stats['stop_reason'], 'gradient')
        self.assertLessEqual(sum(row['tr_accepted'] is False for row in h), 2)
        self.assertEqual(h[0]['lift_sigma_min'], 0.0)                       # the padded slots: a singular lift
        self.assertTrue(all(np.isfinite(row['pKp']) for row in h if not row['converged']))
        self.assertLessEqual(stats['newton_steps'], 12)


def _slack_problem(W, order, x_ranks):
    """c42's set-up on probe derivatives: a rank-3 target on (6,6,6), 2 % noise, and a start point of the
    (non-minimal) ranks ``x_ranks``."""
    rng = np.random.default_rng(11)
    np.random.seed(11)
    shape = (6, 6, 6)
    target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
    ww = [rng.standard_normal((W, N)) for N in shape]
    pp = [rng.standard_normal((W, N)) for N in shape]
    kind = bfit.ProbeDerivativesKind(order=order)
    data = [z + 0.02 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in kind.point_forward(target.data, (ww, pp))]
    return bopt.least_squares_problem(_MANIFOLD, kind, (ww, pp), data), t3.TuckerTensorTrain.randn(shape, *x_ranks)


class TestExactNewtonFromNonMinimalPoints(unittest.TestCase):
    """T25 (I-23, F28 / F38, ragged): exact newton_cg runs from a NON-minimal point with no minimal reduction --
    the first local model's frame is the slack one ``t3_orthogonal_representations`` gives -- and converges on the
    gradient test in the same number of outer (and accepted) steps as the same run from the tensor reduced to its
    minimal ranks, along the same trajectory (the objective row by row to 1e-8; observed <= 3.6e-10). (The final
    ‖g‖/‖g₀‖ sits at round-off, 1e-12, so it is compared only through the stop.) Observed (S5): Tucker slack 14
    outer / 1 rejected, both starts, and curvature_damping 0 the same; TT slack 22 / 6.
    (Gauss-Newton's behaviour from a TT-slack start is a property of the problem, not of the frame: E11.)"""

    def _check(self, problem, x0, up, down, left, right, outer, curvature_dampings=(None,)):
        frame = problem.local_model(x0.data, hessian='exact').frame
        reference = fvc.t3_orthogonal_representations(x0.data)[0]
        self.assertEqual([a.shape for a in sum(map(tuple, frame), ())], [a.shape for a in sum(map(tuple, reference), ())])
        U, O, L, R = frame
        self.assertEqual(([u.shape[0] for u in U], [o.shape[1] for o in O]), (up, down))
        self.assertEqual(([L[0].shape[0]] + [l.shape[2] for l in L], [R[0].shape[0]] + [r.shape[2] for r in R]),
                         (left, right))
        y, _, _ = x0.t3svd()
        minimal = y.rank_adjustment_sweep('right_to_left')
        runs = {}
        for name, start in (('slack', x0.data), ('minimal', minimal.data)):
            for cd in curvature_dampings:
                _, stats = bopt.newton_cg(problem, start, max_newton=80, hessian='exact', curvature_damping=cd)
                self.assertEqual(stats['stop_reason'], 'gradient', (name, cd))
                runs[name, cd] = stats
        reference = runs['slack', None]
        self.assertEqual(reference['newton_steps'], outer)
        for key, stats in runs.items():
            with self.subTest(run=key):
                self.assertEqual((stats['newton_steps'], stats['newton_accepted']),
                                 (reference['newton_steps'], reference['newton_accepted']))
                worst = max(abs(a - b) / abs(b) for a, b in zip(stats['losses'], reference['losses']))
                self.assertLess(worst, 1e-8)                                # the same trajectory (observed <= 3.6e-10)

    def test_tucker_slack(self):
        problem, x0 = _slack_problem(100, 1, ((4, 4, 4), (1, 2, 2, 1)))
        self._check(problem, x0, up=[4, 4, 4], down=[2, 4, 2], left=[1, 2, 2, 1], right=[1, 2, 2, 1], outer=14,
                    curvature_dampings=(None, 0.0))

    def test_tt_slack(self):
        problem, x0 = _slack_problem(200, 0, ((4, 4, 4), (1, 9, 9, 1)))
        self._check(problem, x0, up=[4, 4, 4], down=[4, 4, 4], left=[1, 4, 9, 1], right=[1, 4, 4, 1], outer=22)


# ---------------------------------------------------------------------------------------------------
# S7: the exact Hessian on the SHARED manifold -- the tied lift (T12, T22)
# ---------------------------------------------------------------------------------------------------
# (structure, sharing labels). On 'minimal' every group mode's S_i^T is square, so the tied lift equals the
# per-mode one; on the others a group's rank exceeds some member's own ceiling (its down rank), S_i^T is wide
# there, and the per-mode lift -- a minimum-norm solve per mode -- is not TIED: no direction of the shared
# parametrization Phi(T theta_sh), and the wrong curvature (these structures discriminate the two).
_SH_MINIMAL = (((5, 5, 6), (2, 2, 2), (1, 2, 2, 1)), (0, 0, 1))
_SH_CEILING = (((6, 6, 4), (4, 4, 2), (1, 2, 2, 1)), (0, 0, 1))                # mode 0: down rank 2 < 4
_SH_TWO_GROUPS = (((5, 6, 5, 6), (3, 2, 3, 2), (1, 2, 3, 2, 1)), (0, 1, 0, 1))  # mode 0: down rank 2 < 3
_SH_ONE_GROUP = (((5, 5, 5), (3, 3, 3), (1, 2, 2, 1)), (0, 0, 0))               # modes 0, 2: down rank 2 < 3
_SH_STRUCTURES = {'minimal': _SH_MINIMAL, 'ceiling': _SH_CEILING, 'two_groups': _SH_TWO_GROUPS,
                  'one_group': _SH_ONE_GROUP}


def _shared_unit_point(struct, labels, C=()):
    """A random TIED T3 (one factor array per sharing group) scaled to ``||X||_HS = 1`` per stack element."""
    x = t3.TuckerTensorTrain.randn(*struct, stack_shape=C)
    tk, tt = sharing.t3_tie_tucker_factors(x.data, labels)
    X = np.asarray(t3c.t3_to_dense((tk, tt)))
    nrm = np.asarray(np.sqrt(np.sum(X ** 2, axis=tuple(range(len(C), X.ndim)))))
    return tuple(tk), tuple(tt[:-1]) + (tt[-1] / nrm[..., None, None, None],)


def _shared_fit(struct, labels, order, W, rng, C=(), weight=None, regularizer=None):
    """:py:func:`_pd_fit` on the shared manifold: data from another tied unit point (plus noise), a tied ``x``."""
    geom = _MANIFOLD.with_sharing(labels, struct[0])
    x, other = _shared_unit_point(struct, labels, C), _shared_unit_point(struct, labels, C)
    ww = [rng.standard_normal(W + (N,)) for N in struct[0]]
    pp = [rng.standard_normal(W + (N,)) for N in struct[0]]
    data = [z + 0.1 * rng.standard_normal(z.shape) for z in sd.t3_probe_derivatives(ww, pp, other, order)]
    kind = bfit.ProbeDerivativesKind(order=order, weight=weight)
    return bopt.least_squares_problem(geom, kind, (ww, pp), data, regularizer=regularizer), x


def _per_mode_twin(lm):
    """The same exact model with the UNTIED (per-mode) lift factors: what the shared Hessian would be without the
    tied lift -- the mutation every T12 check must catch on the discriminating structures."""
    data = lm.curvature.geometry_data
    untied = dc.replace(data, lift_factors=fvo.fv_lift_factors(lm.frame, 0.0))
    return dc.replace(lm, curvature=bopt.CurvatureState(lm.curvature.misfit_gradient, untied))


def _groups_of(labels, shape):
    return sharing.nontrivial_groups(sharing.validate_sharing(labels, shape))


class TestSharedExactLift(unittest.TestCase):
    """T12, the lift: the TIED lift (``fv_tied_lift_factors`` + ``tv_corewise_lift``) represents a tied tangent
    through the core parametrization with ONE factor direction per group, and its transpose is the exact
    adjoint at every damping -- (0, 0, 1), a two-group partition and an all-mode group; K, C and K + C stacks."""

    def setUp(self):
        np.random.seed(51)

    def test_tied_lift_identity_and_tie(self):
        """``DPhi[L v] == E v`` at eps = 0 (to roundoff), the group's raw factor directions one array; the per-mode
        lift represents ``v`` too but is untied on the discriminating structures (and equals the tied one where
        every group mode's ``S_i^T`` is square)."""
        for name, (struct, labels) in _SH_STRUCTURES.items():
            groups = _groups_of(labels, struct[0])
            for K, C in (((), ()), ((3,), ()), ((), (2,)), ((2,), (2,))):
                with self.subTest(structure=name, K=K, C=C):
                    x = _shared_unit_point(struct, labels, C)
                    frame, _ = fvc.t3_orthogonal_representations(x)
                    sfd = sharing.fv_shared_frame_data(frame, sharing.validate_sharing(labels, struct[0]))
                    v = tv.tv_orthogonal_gauge_projection(frame, _random_variations(frame, K + C), shared_data=sfd)
                    th = tv.tv_corewise_lift(frame, v, sharing.fv_tied_lift_factors(frame, 0.0, sfd))
                    corewise_frame = fvc.t3_corewise_frame((frame[0], frame[2]))
                    n = len(K + C)
                    self.assertLess(_relerr_per_element(tv.tv_to_dense(corewise_frame, th),
                                                        tv.tv_to_dense(frame, v), n), 1e-12)
                    for group in groups:
                        for ii in group[1:]:
                            self.assertIs(th[0][ii], th[0][group[0]])
                    pm = tv.tv_corewise_lift(frame, v, fvo.fv_lift_factors(frame, 0.0))
                    self.assertLess(_relerr_per_element(tv.tv_to_dense(corewise_frame, pm),
                                                        tv.tv_to_dense(frame, v), n), 1e-12)
                    untied = max(_relerr_per_element(pm[0][ii], pm[0][g[0]], n) for g in groups for ii in g[1:])
                    if name == 'minimal':
                        self.assertLess(_relerr_tree(pm, th), 1e-12)
                    else:
                        self.assertGreater(untied, 1e-2)

    def test_tied_lift_adjoint(self):
        """``<L v, y> == <v, L^T y>`` per K + C element for ANY coordinates and raw directions, at dampings 0 to
        10 -- the group's copies summed (``T^T``), then the tied solve's adjoint."""
        rng = np.random.default_rng(52)
        for name in ('ceiling', 'two_groups', 'one_group'):
            struct, labels = _SH_STRUCTURES[name]
            for K, C in (((), ()), ((3,), ()), ((2,), (2,))):
                for cd in (0.0, 1e-2, 0.3, 10.0):
                    with self.subTest(structure=name, K=K, C=C, curvature_damping=cd):
                        frame, _ = fvc.t3_orthogonal_representations(_shared_unit_point(struct, labels, C))
                        sfd = sharing.fv_shared_frame_data(frame, sharing.validate_sharing(labels, struct[0]))
                        lf = sharing.fv_tied_lift_factors(frame, cd, sfd)
                        v = _random_variations(frame, K + C)
                        y = _raw_like_frame_cores(frame, K + C, rng)
                        Lv, LTy = tv.tv_corewise_lift(frame, v, lf), tv.tv_corewise_lift_transpose(frame, y, lf)
                        for Vt, V in zip(LTy[0] + LTy[1], v[0] + v[1]):
                            self.assertEqual(Vt.shape, V.shape)
                        n = len(K + C)
                        lhs = np.asarray(cw.corewise_stack_dot(Lv, y, n))
                        rhs = np.asarray(cw.corewise_stack_dot(v, LTy, n))
                        scale = np.asarray(cw.corewise_stack_norm(Lv, n)) * np.asarray(cw.corewise_stack_norm(y, n))
                        self.assertTrue(np.all(np.abs(lhs - rhs) <= 1e-12 * np.maximum(scale, 1e-300)))

    def test_the_tied_filter_at_no_damping_is_the_projections(self):
        """The damped tied filter is ``filtered_reciprocal`` of the group spectrum at the per-mode factors' own
        absolute ``eps``; at ``curvature_damping = 0`` it is the tied projection's clipped filter, and the lift
        of a tied tangent recovers exactly the ambient direction the retraction's tied embedding uses."""
        struct, labels = _SH_CEILING
        frame, _ = fvc.t3_orthogonal_representations(_shared_unit_point(struct, labels))
        sfd = sharing.fv_shared_frame_data(frame, sharing.validate_sharing(labels, struct[0]))
        for cd in (0.0, 1e-3):
            lf = sharing.fv_tied_lift_factors(frame, cd, sfd)
            plain = fvo.fv_lift_factors(frame, cd)
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(
                lf.tucker_pinvs + lf.bond_pinvs + (lf.eps,), plain.tucker_pinvs + plain.bond_pinvs + (plain.eps,))))
            self.assertIs(lf.shared_data, sfd)
            self.assertEqual(len(lf.tied_filters), 1)
            s_g = sfd.svd_s[0]
            self.assertTrue(np.array_equal(lf.tied_filters[0], linalg.filtered_reciprocal(
                s_g, lf.eps, np.finfo(float).eps * max(sfd.row_splits[0][-1], sfd.svd_Vt[0].shape[-1]))))
        lf0 = sharing.fv_tied_lift_factors(frame, 0.0, sfd)
        self.assertTrue(np.allclose(lf0.tied_filters[0], sharing._tied_filter(sfd, 0, None, None), rtol=1e-15, atol=0))
        v = tv.tv_orthogonal_gauge_projection(frame, _random_variations(frame, ()), shared_data=sfd)
        self.assertLess(_relerr_tree(tv.tv_corewise_lift(frame, v, lf0)[0][0],
                                     sharing.fv_tied_ambient_directions(v, sfd)[0]), 1e-13)
        with self.assertRaisesRegex(ValueError, 'tied filters'):
            tv.tv_corewise_lift(frame, v, dc.replace(lf0, tied_filters=()))


class TestSharedExactLocalModel(unittest.TestCase):
    """T12, the exact model on the shared manifold: ``Pi_sh [J^T J + L^T K_N L] Pi_sh`` with ``L`` the tied lift and
    ``N = A^T omega^2 r - grad_sh f`` (the TIED misfit gradient). Oracles: the second derivative along the shared
    retraction (measured second order, F14 -- O(h^2), not proven), the derivative of the projected shared gradient
    along it (exact for any retraction), and the dense Lagrangian Hessian of ``Phi o T`` through an independent
    dense tied lift. Every check also catches the per-mode lift on the discriminating structures."""

    def setUp(self):
        np.random.seed(53)

    def test_quadratic_form_is_the_second_derivative_along_the_shared_retraction(self):
        """``<p, H p> == d^2/dt^2 f(R_sh(t p))`` to the finite difference's resolution, which also separates it from
        Gauss-Newton and from the per-mode-lift twin; orders 0-2 with the weight shapes, both dampings."""
        rng = np.random.default_rng(53)
        cases = {'minimal': (_SH_MINIMAL, 2, [1.0, 0.5, 0.25]), 'ceiling': (_SH_CEILING, 1, None),
                 'ceiling_matrix': (_SH_CEILING, 2, rng.uniform(0.3, 2.0, (3, 3))),
                 'two_groups': (_SH_TWO_GROUPS, 1, [[1.0], [0.4], [2.0], [0.7]]), 'one_group': (_SH_ONE_GROUP, 0, None)}
        for name, ((struct, labels), order, weight) in cases.items():
            for cd in (0.0, None):
                with self.subTest(case=name, curvature_damping=cd):
                    prob, x = _shared_fit(struct, labels, order, (12,), rng, weight=weight)
                    lm = prob.local_model(x, hessian='exact', curvature_damping=cd)
                    v = _unit_tangent(prob.geom, lm.frame)
                    q, q_gn = float(lm.quadratic(v)), float(lm.gn_quadratic(v))
                    self.assertLess(abs(float(prob.geom.inner(v, lm.hvp(v))) - q), 1e-12 * abs(q))
                    fd = _second_derivative(lambda t: float(prob.objective(lm.retract(cw.corewise_scale(v, t)))),
                                            _fd_step(lm))
                    self.assertLess(abs(fd - q), 1e-5 * max(abs(q), abs(q_gn)))
                    self.assertLess(abs(fd - q), 1e-3 * abs(q - q_gn))
                    if name != 'minimal' and cd == 0.0:
                        q_pm = float(_per_mode_twin(lm).quadratic(v))
                        self.assertGreater(abs(fd - q_pm), 1e3 * abs(fd - q))

    def test_the_shared_retraction_is_second_order_here(self):
        """F14 (``c15_shared_exact``'s set-up, seed 5; measured, not proven): the plain central second difference
        along the shared retraction converges to the exact model's ``<v, H v>`` at O(h^2) -- the error falls ~4x per
        halving (observed ratios 4.0, 4.1 over h = 2e-2, 1e-2, 5e-3, from 1.1e-4) -- while Gauss-Newton is off by
        0.16 (observed here: GN 1058.03248707, exact 1058.19431605; the values depend on LAPACK's frame signs, so
        only the convergence is asserted)."""
        np.random.seed(5)
        rng = np.random.default_rng(5)
        shape, labels = (5, 5, 6), (0, 0, 1)
        x = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
        x = x.share(labels, max_tucker_ranks=(2, 2, 2), max_tt_ranks=(1, 2, 2, 1))
        x = x.rank_adjustment_sweep('right_to_left', sharing=labels)
        geom = _MANIFOLD.with_sharing(labels, shape)
        frame = geom.frame(x.data)
        aux = geom.precompute(frame)
        ww = [rng.standard_normal((60, N)) for N in shape]
        pp = [rng.standard_normal((60, N)) for N in shape]
        kind = bfit.probe_derivatives_kind(2, None)
        data = [z + 0.3 * rng.standard_normal(z.shape) for z in kind.point_forward(x.data, (ww, pp))]
        prob = bopt.least_squares_problem(geom, kind, (ww, pp), data)
        U, O, L, _ = frame
        raw = (tuple(rng.standard_normal(o.shape[-2:-1] + u.shape[-1:]) for o, u in zip(O, U)),
               tuple(rng.standard_normal(l.shape) for l in L))
        v = geom.project(frame, raw, aux)
        v = cw.corewise_scale(v, 1.0 / np.linalg.norm(tv.tv_to_dense(frame, v)))
        lm = prob.local_model(x.data, hessian='exact', curvature_damping=0.0)
        q, q_gn = float(lm.quadratic(v)), float(lm.gn_quadratic(v))
        f0 = float(prob.objective(x.data))
        errors = []
        for h in (2e-2, 1e-2, 5e-3):
            fp = float(prob.objective(lm.retract(cw.corewise_scale(v, h))))
            fm = float(prob.objective(lm.retract(cw.corewise_scale(v, -h))))
            errors.append(abs((fp - 2.0 * f0 + fm) / h ** 2 - q))
        for e_h, e_half in zip(errors, errors[1:]):
            self.assertTrue(3.0 < e_h / e_half < 5.0, errors)
        self.assertLess(errors[-1], 1e-3 * abs(q - q_gn))

    def test_hvp_is_the_derivative_of_the_projected_shared_gradient(self):
        """``H v == d/dt P_x grad_sh f(R_sh(t v))`` at 0 -- the Riemannian Hessian of an embedded submanifold is the
        projected derivative of the gradient along ANY curve with velocity ``v``, so this oracle needs no second-order
        retraction. Central difference at h = 1e-4 (observed <= 4e-7 relative); GN and the per-mode twin miss it."""
        rng = np.random.default_rng(54)
        for name, (struct, labels) in _SH_STRUCTURES.items():
            with self.subTest(structure=name):
                prob, x = _shared_fit(struct, labels, 2, (12,), rng, weight=[1.0, 0.5, 0.25])
                lm = prob.local_model(x, hessian='exact', curvature_damping=0.0)
                geom, frame, aux = prob.geom, lm.frame, lm.geom_aux
                v = _unit_tangent(geom, frame)

                def grad_at(t):                    # P_x grad_sh f(R(t v)): the ambient gradient, projected at x
                    y = lm.retract(cw.corewise_scale(v, t))
                    lm_y = prob.local_model(y)
                    return tv.tv_project_t3_onto_tangent_space(frame, tv.tv_to_t3(lm_y.frame, lm_y.gradient),
                                                                shared_data=aux)

                h = 1e-4
                fd = cw.corewise_scale(cw.corewise_sub(grad_at(h), grad_at(-h)), 0.5 / h)
                err = _relerr_tree(lm.hvp(v), fd)
                self.assertLess(err, 1e-5)
                self.assertGreater(_relerr_tree(prob.local_model(x).hvp(v), fd), 1e2 * err)
                if name != 'minimal':
                    self.assertGreater(_relerr_tree(_per_mode_twin(lm).hvp(v), fd), 1e2 * err)

    def test_against_the_dense_lagrangian_hessian(self):
        """``Pi_sh L^T H_lag L Pi_sh`` with ``H_lag`` the exact dense core-coordinate Hessian of the Lagrangian
        ``f(Phi(theta)) - <grad_sh f, Phi(theta)>`` and ``L`` the INDEPENDENT dense tied lift (pinv of the stacked
        ``S_i^T``) -- the Hessian of ``Phi o T`` with none of the production tied-lift code -- on (0, 0, 1) and a
        two-group partition, both past a member's ceiling (and below the mode size: a group at rank ``N`` has no
        Tucker variation, so the tied and per-mode lifts would coincide)."""
        rng = np.random.default_rng(55)
        for name, struct, labels, order, weight in (
                ('ceiling', ((4, 4, 3), (3, 3, 2), (1, 2, 2, 1)), (0, 0, 1), 2, [1.0, 0.5, 0.25]),
                ('two_groups', ((4, 4, 4, 4), (3, 2, 3, 2), (1, 2, 2, 2, 1)), (0, 1, 0, 1), 1, [[1.0, 0.7]] * 4)):
            with self.subTest(structure=name):
                shape = struct[0]
                prob, x = _shared_fit(struct, labels, order, (4,), rng, weight=weight)
                kind, (ww, pp) = prob.kind, prob.sample
                lm = prob.local_model(x, hessian='exact', curvature_damping=0.0)
                frame, aux = lm.frame, lm.geom_aux
                self.assertLess(min(o.shape[-2] for o in frame[1]), max(u.shape[-2] for u in frame[0]))
                omega = kind.weight

                def measure(cores):                                               # omega (.) S(X), flattened
                    out = sd.t3_probe_derivatives(ww, pp, cores, order)
                    return np.concatenate([(z * omega[i if omega.shape[0] > 1 else 0].reshape(-1, 1, 1)).ravel()
                                           for i, z in enumerate(out)])
                cols = []
                for idx in np.ndindex(*shape):
                    unit = (tuple(np.eye(N)[j][None, :] for N, j in zip(shape, idx)),
                            tuple(np.ones((1, 1, 1)) for _ in shape))
                    cols.append(measure(unit))
                A = np.stack(cols, axis=1)
                y = A @ np.asarray(t3c.t3_to_dense(x)).ravel() - np.concatenate(
                    [(r * omega[i if omega.shape[0] > 1 else 0].reshape(-1, 1, 1)).ravel()
                     for i, r in enumerate(lm.residual)])
                theta = (tuple(frame[0]), tuple(frame[2]))
                G = tv.tv_to_dense(frame, lm.curvature.misfit_gradient)          # grad_sh f, dense
                groups = sharing.validate_sharing(labels, shape)
                for _ in range(2):
                    p = fvo.fv_variations_randn(fvo.fv_variation_shapes(frame))
                    pg = prob.geom.project(frame, p, aux)
                    H_theta = oracles.dense_lagrangian_hessian(A, y, theta, oracles.tied_lift(frame, groups, pg), G)
                    ref = prob.geom.project(frame, oracles.tied_lift_transpose(frame, groups, H_theta), aux)
                    err = _relerr_tree(lm.hvp(p), ref)
                    self.assertLess(err, 1e-10)
                    self.assertGreater(_relerr_tree(_per_mode_twin(lm).hvp(p), ref), 1e4 * err)

    def test_gauge_laws_symmetry_and_stacks(self):
        """``hvp(p) == hvp(Pi_sh p)``, ``Pi_sh hvp == hvp`` and the output TIED; the K-stack Gram matrix symmetric;
        a C-stacked shared model's hvp / quadratic per element (K + C)."""
        rng = np.random.default_rng(56)
        for name in ('ceiling', 'two_groups', 'one_group'):
            struct, labels = _SH_STRUCTURES[name]
            with self.subTest(structure=name):
                prob, x = _shared_fit(struct, labels, 2, (8,), rng)
                lm = prob.local_model(x, hessian='exact')
                geom, frame, aux = prob.geom, lm.frame, lm.geom_aux
                p = fvo.fv_variations_randn(fvo.fv_variation_shapes(frame))           # ungauged, untied
                Hp = lm.hvp(p)
                self.assertLess(_relerr_tree(lm.hvp(geom.project(frame, p, aux)), Hp), 1e-12)
                self.assertLess(_relerr_tree(geom.project(frame, Hp, aux), Hp), 1e-12)
                self.assertLess(float(sharing.fv_tied_variations_residual(Hp, aux)), 1e-12)
                self.assertLess(float(sharing.fv_tied_variations_residual(lm.curvature_hvp(p), aux)), 1e-12)
                n_dir = 4
                P = _unit_tangent(geom, frame, K=(n_dir,))
                HP = lm.hvp(P)
                gram = sum(a.reshape(n_dir, -1) @ b.reshape(n_dir, -1).T for a, b in zip(P[0] + P[1], HP[0] + HP[1]))
                self.assertLess(np.abs(gram - gram.T).max(), 1e-12 * np.abs(gram).max())
                self.assertTrue(np.allclose(np.asarray(lm.quadratic(P)), np.diag(gram), rtol=1e-12, atol=0))
        K, C = (2,), (2,)
        struct, labels = _SH_CEILING
        prob, x = _shared_fit(struct, labels, 1, (6,), rng, C=C)
        lm = prob.local_model(x, hessian='exact')
        p = _unit_tangent(prob.geom, lm.frame, K=K, C=C)
        Hp, q = lm.hvp(p), np.asarray(lm.quadratic(p))
        self.assertEqual(q.shape, K + C)
        for cc in np.ndindex(*C):
            data_c = [z[(slice(None),) * 2 + cc] for z in prob.data]
            lm_c = dc.replace(prob, data=data_c).local_model(_elem(x, cc), hessian='exact')
            for kk in np.ndindex(*K):
                p_kc = _elem(p, kk + cc)
                self.assertLess(_relerr_tree(_elem(Hp, kk + cc), lm_c.hvp(p_kc)), 1e-12)
                self.assertAlmostEqual(float(q[kk + cc]), float(lm_c.quadratic(p_kc)), delta=1e-11 * abs(q[kk + cc]))

    def test_sigma_min_reads_the_group_spectrum(self):
        """``sigma_min`` (shape C) is the smallest singular value the TIED lift inverts: the bond spectra, the Tucker
        spectra of the untied modes and each group's spectrum ``s_g`` -- not a group mode's own ``S_i^T`` spectrum,
        which can be tiny on a regular shared point (here mode 1 barely uses the group's third basis row, which
        mode 2 uses fully: its own spectrum reaches ~1e-2 while every lift singular value is O(1))."""
        labels, shape = (0, 1, 1, 2), (4, 5, 5, 4)
        geom = _MANIFOLD.with_sharing(labels, shape)
        for seed, C in ((0, ()), (1, ()), (2, (2,))):
            with self.subTest(seed=seed, C=C):
                np.random.seed(seed)
                x = t3.TuckerTensorTrain.randn(shape, (2, 3, 3, 2), (1, 2, 3, 2, 1), stack_shape=C)
                tk, tt = sharing.t3_tie_tucker_factors(x.data, labels)
                G1 = np.array(tt[1])
                G1[..., 2, :] *= 1e-3
                x = (tk, (tt[0], G1) + tuple(tt[2:]))
                frame = geom.frame(x)
                aux = geom.precompute(frame)
                g = fvo.fv_variations_zeros(fvo.fv_variation_shapes(frame), C, False)
                data = geom.curvature_precompute(frame, g, aux, 0.0)
                lf = data.lift_factors
                spectra = list(lf.bond_svals) + [lf.tucker_svals[0], lf.tucker_svals[3]] + list(aux.svd_s)
                ref = np.min([np.min(s, axis=-1) for s in spectra], axis=0)
                self.assertEqual(np.shape(data.sigma_min), C)
                self.assertTrue(np.array_equal(np.asarray(data.sigma_min), ref))
                own = np.min([np.min(s, axis=-1) for s in lf.tucker_svals], axis=0)
                self.assertTrue(np.all(own < 1e-1 * ref))                    # a per-mode reading would be caught

    def test_curvature_precompute_without_the_companion_builds_it(self):
        """``curvature_precompute(frame, g, aux=None, cd)`` on a shared geometry builds the companion itself
        (``self.precompute(frame)``) and returns the same data, leaf for leaf and bitwise, as with the companion
        passed in -- the fallback a caller without a stored ``aux`` takes."""
        labels, shape = (0, 1, 1, 2), (4, 5, 5, 4)
        geom = _MANIFOLD.with_sharing(labels, shape)

        def leaves(t):
            if dc.is_dataclass(t) and not isinstance(t, type):
                return [a for f in dc.fields(t) for a in leaves(getattr(t, f.name))]
            if isinstance(t, (tuple, list)):
                return [a for u in t for a in leaves(u)]
            return [t] if isinstance(t, np.ndarray) else []

        for C in ((), (2,)):
            with self.subTest(C=C):
                np.random.seed(7)
                x = t3.TuckerTensorTrain.randn(shape, (2, 3, 3, 2), (1, 2, 3, 2, 1), stack_shape=C)
                frame = geom.frame(sharing.t3_tie_tucker_factors(x.data, labels))
                aux = geom.precompute(frame)
                g = geom.project(frame, fvo.fv_variations_randn(fvo.fv_variation_shapes(frame), C), aux)
                for cd in (0.0, 1e-3):
                    a = leaves(geom.curvature_precompute(frame, g, aux, cd))
                    b = leaves(geom.curvature_precompute(frame, g, None, cd))
                    self.assertGreater(len(a), 0)
                    self.assertEqual(len(a), len(b))
                    self.assertTrue(all(np.array_equal(u, v) for u, v in zip(a, b)))

    def test_padded_shared_restart(self):
        """A zero-padded shared restart (``resize(..., sharing=)``) sits on the lower shared-rank stratum: the new
        ``s_g`` level is zero and its tied filter exactly 0 at every damping, so the gated tied channel gets no
        curvature -- the lift's group direction has no component in ``M_g``'s null space and the transpose ignores
        it -- while the TT variations (the escape) and the untied mode lift as unshared, bit for bit;
        ``sigma_min`` reports the zero level; the exact model stays finite and symmetric there."""
        rng = np.random.default_rng(57)
        np.random.seed(57)
        shape, labels = (6, 6, 6), (0, 0, 1)
        geom = _MANIFOLD.with_sharing(labels, shape)
        level1 = t3.TuckerTensorTrain(*_shared_unit_point((shape, (2, 2, 2), (1, 2, 2, 1)), labels))
        restart = level1.t3svd(sharing=labels)[0].resize(shape, (3, 3, 3), (1, 3, 3, 1), sharing=labels)
        self.assertTrue(bool(np.all(restart.has_shared_tucker_factors(labels))))
        frame = geom.frame(restart.data)
        aux = geom.precompute(frame)
        _, STs, _ = fvo.fv_center_factors(frame)
        _, s_M, Vt_M = np.linalg.svd(np.concatenate([STs[0], STs[1]], axis=0))
        null = Vt_M[s_M <= 1e-12 * s_M[0]]                                 # M_g's null space: the gated levels
        self.assertEqual(null.shape[0], 1)
        for cd in (0.0, 1e-6):
            with self.subTest(curvature_damping=cd):
                lf = sharing.fv_tied_lift_factors(frame, cd, aux)
                f = lf.tied_filters[0]
                self.assertEqual(int(np.sum(f == 0.0)), 1)                     # the new level: exactly 0
                self.assertTrue(np.all(f[:-1] > 0.0))
                v = _random_variations(frame, (3,))
                th = tv.tv_corewise_lift(frame, v, lf)
                self.assertLess(np.abs(np.einsum('ju,kun->kjn', null, th[0][0])).max(),
                                1e-14 * np.abs(th[0][0]).max())
                plain = tv.tv_corewise_lift(frame, v, fvo.fv_lift_factors(frame, cd))
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(th[1] + (th[0][2],), plain[1] + (plain[0][2],))))
                y = _raw_like_frame_cores(frame, (3,), rng)
                y_gated = ((y[0][0] + np.einsum('ju,kjn->kun', null, rng.standard_normal((3, 1, 6))),) + y[0][1:], y[1])
                a, b = tv.tv_corewise_lift_transpose(frame, y, lf), tv.tv_corewise_lift_transpose(frame, y_gated, lf)
                self.assertLess(_relerr_tree(b, a), 1e-14)
        ww = [rng.standard_normal((10, N)) for N in shape]
        pp = [rng.standard_normal((10, N)) for N in shape]
        data = [z + 0.1 for z in sd.t3_probe_derivatives(ww, pp, _shared_unit_point((shape, (3, 3, 3), (1, 3, 3, 1)),
                                                                                    labels), 1)]
        prob = bopt.least_squares_problem(geom, bfit.ProbeDerivativesKind(order=1), (ww, pp), data)
        lm = prob.local_model(restart.data, hessian='exact')
        cdata = lm.curvature.geometry_data
        self.assertEqual(float(cdata.sigma_min), float(np.min(aux.svd_s[0])))
        self.assertLess(float(cdata.sigma_min), 1e-12)
        P = _unit_tangent(geom, lm.frame, K=(3,))
        HP = lm.hvp(P)
        self.assertTrue(all(np.all(np.isfinite(a)) for a in HP[0] + HP[1]))
        gram = sum(a.reshape(3, -1) @ b.reshape(3, -1).T for a, b in zip(P[0] + P[1], HP[0] + HP[1]))
        self.assertLess(np.abs(gram - gram.T).max(), 1e-12 * np.abs(gram).max())


class TestSharedExactNewtonRestart(unittest.TestCase):
    """T22 (F19 on the shared manifold, ``shared_manifold((0, 0, 1))``): exact newton_cg from a zero-padded SHARED
    continuation restart converges on the gradient test with at most two rejections -- the tied lift's zero ``s_g``
    level is clipped, the TT variations carry the escape. c24's template, tied: probe derivatives of order 2,
    ω = [1, .5, .25], 5 % noise, a tied target of rank 4, level 1 at shared rank 2 (30 Gauss-Newton steps), the
    restart at rank 3; |W| = 60. Observed (defaults: zero reference, trust region, power 0.5): 10 outer steps,
    0 rejected, 41 CG iterations (F19's unshared problem: 8-10 outer, 23-28 CG), against Gauss-Newton + Armijo,
    which has not converged after 25 steps (40: still 7e-6)."""

    def test_zero_padded_shared_restart_converges(self):
        problem, restart = _shared_restart_problem(60)
        _, stats = bopt.newton_cg(problem, restart.data, max_newton=40, hessian='exact')
        h = stats['history']
        self.assertEqual(stats['stop_reason'], 'gradient')
        self.assertLessEqual(sum(row['tr_accepted'] is False for row in h), 2)
        self.assertEqual(h[0]['lift_sigma_min'], 0.0)                 # the zero s_g level: a singular tied lift
        self.assertTrue(all(np.isfinite(row['pKp']) for row in h if not row['converged']))
        self.assertLessEqual(stats['newton_steps'], 14)
        _, gn = bopt.newton_cg(problem, restart.data, max_newton=25)
        self.assertEqual(gn['stop_reason'], 'max_newton')


def _shared_restart_problem(W):
    """T22's problem: c24 on ``shared_manifold((0, 0, 1))`` -- a tied rank-4 target, a shared rank-2 fit (30
    Gauss-Newton steps) zero-padded to rank 3 with ``resize(..., sharing=)``. Returns the backend problem and the
    tied restart."""
    rng = np.random.default_rng(7)
    np.random.seed(7)
    shape, labels = (6, 6, 6), (0, 0, 1)
    target = t3.TuckerTensorTrain(*sharing.t3_tie_tucker_factors(
        t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1)).data, labels))
    ww = [rng.standard_normal((W, N)) for N in shape]
    pp = [rng.standard_normal((W, N)) for N in shape]
    kind = bfit.probe_derivatives_kind(2, np.array([1.0, 0.5, 0.25]))
    data = [z + 0.05 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in kind.point_forward(target.data, (ww, pp))]
    problem = bopt.least_squares_problem(_MANIFOLD.with_sharing(labels, shape), kind, (ww, pp), data)
    x0 = sharing.t3_tie_tucker_factors(t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data, labels)
    x1, _ = bopt.newton_cg(problem, x0, max_newton=30, gtol_rel=1e-10)
    return problem, t3.TuckerTensorTrain(*x1).resize(shape, (3, 3, 3), (1, 3, 3, 1), sharing=labels)


if __name__ == '__main__':
    unittest.main()
