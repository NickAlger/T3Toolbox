"""Tests for the backend-first optimizers (dev/archive/optimizers_plan.md, G3), on the graded-messages seam (P16):
a fit is a geometry, an oracle, its batch and a regularizer (`tests/_fits.py` builds them from plain specs).

Correctness gold standard: the backend LocalModel's values against DENSE ground truth (the dense tensor, its
Jacobian along a direction, finite differences of the dense objective), never the model against itself; and its
consistency (`<p, hvp(p)> = gn_quadratic(p)`, the objective equals `point_objective`). `TestOracleSeam`: the head of
the optimizers, `held` never crossing a frame or a batch (MC-SGD, Adam and a minibatched Newton against references
built afresh per draw), the minibatch rule (the regularizer unscaled), the structural efficiency of a model query
given `held`, the model's `f` against `point_objective` at four kinds of frame, the floor's `eps` read off `f`.
Plus: the optimizers descend; T14 (Steihaug-Toint CG, `TestSteihaugCG`) and T15 (newton_cg's trust-region loop
and its floors, `TestTrustRegionNewton`). numpy-only (jit dispatch is covered separately in test_dispatch).

The Newton timing plan's T1-T5 on numpy (`TestNewtonTiming`): timing off is today's run; the records tile each row's
window and the run; timing changes no number; the record's shape; every phase carries its own work and waits on its
own outputs; the compile listener's events. Its `check_timing` is shared with the padded start's and the jax rows
(`test_padded_start.py`, `tests/test_dispatch.py`).

The Gauss-Newton-to-exact switch plan's T1-T4 and T6-T8 on numpy (`TestSwitchNewton`): switch_tau=0 is the Gauss-Newton
run and switch_k=0 the exact run, bit for bit; the junction; what counts and what resets the count; where it fires; the
options; the stats and row contract; the tracking phase of a timed run. T5 and T9 jit: `tests/test_dispatch.py`."""
import collections
import contextlib
import dataclasses as dc
import typing as typ
import math
import os
import re
import time
import types
import unittest
import unittest.mock

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.optimizers as opt
import t3toolbox.corewise as cw
import t3toolbox.optimizers as topt
import t3toolbox.shared_geometry as sgm
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.uniform_tucker_tensor_train as ut3
from t3toolbox.backend.common import is_jax_ndarray
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
from t3toolbox.backend.jets import affine_jet
from t3toolbox.backend.sampling_variables import frame_form
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # tests/
import _derivative_data as dd       # derivative data from the merged methods (P13b)
import _fits as F                   # a backend fit (geometry, oracle, batch, regularizer) from plain specs (P16)

SHAPE, TUCKER, TT = (8, 8, 8), (3, 3, 3), (1, 3, 3, 1)


def dense_apply(A, ww):
    res = np.einsum("i...,si->s...", A, ww[0])
    for m in range(1, len(ww)):
        res = np.einsum("sj...,sj->s...", res, ww[m])
    return res


def dense_probe(A, ww):
    d = len(ww); out = []
    for free in range(d):
        ops = [A, list(range(d))]
        for j in range(d):
            if j != free:
                ops += [ww[j], [d, j]]
        ops += [[d, free]]
        out.append(np.einsum(*ops))
    return out


def unit_vecs(M, shape, rng):
    ww = [rng.standard_normal((M, N)) for N in shape]
    return [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]


class _ToyModel:
    """The whole interface CG needs of a local model: `hvp` and `geom.inner`. A toy diagonal operator on a
    `(tucker=[vec], tt=[])` tangent tree exercises every CG branch deterministically."""
    def __init__(self, diagonal):
        self.diagonal = np.asarray(diagonal, dtype=float)
        self.geom = types.SimpleNamespace(inner=cw.corewise_dot)

    def hvp(self, t):
        return ([self.diagonal * t[0][0]], [])


def _toy_rhs(n=3):
    return ([np.ones(n)], [])


def _plain_cg_reference(lm, rhs, tol, maxiter):
    """FROZEN copy of the pre-trust-region (2026.2.0) eager CG -- the arithmetic `_cg_solve` with
    `radius = inf` must reproduce bit for bit (I-5). Do not modernize."""
    inner = lm.geom.inner
    rs0 = inner(rhs, rhs)
    tol2 = tol * tol
    p, r, d, rs, i, ok = cw.corewise_zeros_like(rhs), rhs, rhs, rs0, 0, rs0 > tol2
    while bool((rs > tol2) & (i < maxiter) & ok):
        Hd = lm.hvp(d)
        dHd = inner(d, Hd)
        pos = dHd > 0.0
        alpha = np.where(pos, rs / np.where(pos, dHd, 1.0), 0.0)
        p = cw.corewise_add(p, cw.corewise_scale(d, alpha))
        r = cw.corewise_sub(r, cw.corewise_scale(Hd, alpha))
        rs_new = inner(r, r)
        beta = np.where(pos, rs_new / rs, 0.0)
        d = cw.corewise_add(r, cw.corewise_scale(d, beta))
        rs, i, ok = rs_new, i + 1, pos
    return p, i, rs, ok


def _tree_bytes(t):
    return [np.asarray(a).tobytes() for a in list(t[0]) + list(t[1])]


def _toy_quadratic(lm, rhs, p):
    """The CG model value q(p) = ½⟨p, Hp⟩ − ⟨rhs, p⟩ (Steihaug decreases it monotonically)."""
    return 0.5 * float(cw.corewise_dot(p, lm.hvp(p))) - float(cw.corewise_dot(rhs, p))


class TestSteihaugCG(unittest.TestCase):
    """T14: Steihaug-Toint CG (`_cg_solve(..., radius)`) on the toy diagonal operator and a real local model."""

    def test_indefinite_operator_stops_on_the_boundary(self):
        """`dᵀHd ≤ 0` with a finite radius steps to the boundary (‖p‖ = Δ) instead of truncating with a zero
        step; `ok` reports the nonpositive curvature, `boundary` the stop."""
        rhs = _toy_rhs()
        for diag, radius in (([-1.0, -2.0, -3.0], 0.5), ([2.0, -1.0, 3.0], 10.0), ([1.0, 1.0, -1e-3], 1e3)):
            with self.subTest(diagonal=diag, radius=radius):
                lm = _ToyModel(diag)
                p, i, rs, ok, boundary = opt._cg_solve(lm, rhs, 1e-12, 50, False, radius)
                self.assertTrue(bool(boundary))
                self.assertFalse(bool(ok))                                       # stopped ON nonpositive curvature
                self.assertAlmostEqual(float(cw.corewise_norm(p)), radius, delta=1e-13 * radius)
                self.assertLess(_toy_quadratic(lm, rhs, p), 0.0)                 # a model decrease
                self.assertGreater(float(cw.corewise_dot(rhs, p)), 0.0)          # a descent direction for −rhs
        # first-iteration case: p is exactly the boundary point along the steepest-descent direction
        p, i, rs, ok, boundary = opt._cg_solve(_ToyModel([-1.0, -2.0, -3.0]), rhs, 1e-12, 50, False, 0.5)
        self.assertEqual(int(i), 1)
        self.assertTrue(np.allclose(p[0][0], 0.5 / np.sqrt(3.0) * np.ones(3), rtol=1e-14, atol=0.0))

    def test_positive_definite_operator_with_a_small_radius_stops_on_the_boundary(self):
        """A PD operator whose Newton step is longer than Δ: the CG iterate that would leave the region is
        clipped to ‖p‖ = Δ (`ok` stays True); with Δ above the Newton step's length CG converges inside."""
        rhs = _toy_rhs()
        lm = _ToyModel([2.0, 3.0, 5.0])
        newton = 1.0 / lm.diagonal                                               # ‖newton‖ ≈ 0.62
        for radius in (0.05, 0.3, 0.6):
            with self.subTest(radius=radius):
                p, i, rs, ok, boundary = opt._cg_solve(lm, rhs, 1e-12, 50, False, radius)
                self.assertTrue(bool(boundary) and bool(ok))
                self.assertAlmostEqual(float(cw.corewise_norm(p)), radius, delta=1e-13 * radius)
                self.assertLess(_toy_quadratic(lm, rhs, p), 0.0)
        p, i, rs, ok, boundary = opt._cg_solve(lm, rhs, 1e-12, 50, False, 0.7)
        self.assertFalse(bool(boundary))
        self.assertTrue(np.allclose(p[0][0], newton, rtol=1e-12, atol=0.0))
        self.assertLessEqual(float(rs) ** 0.5, 1e-12)

    def test_infinite_radius_is_the_plain_cg_bit_for_bit(self):
        """I-5: `radius = inf` (the default, and the line-search mode) is the pre-trust-region CG, BIT FOR BIT
        on eager numpy -- the solution, the iteration count, the residual and the curvature flag -- on PD and
        indefinite toys and on a real Gauss-Newton local model (where the Steihaug arithmetic also runs)."""
        rng = np.random.default_rng(0)
        np.random.seed(0)
        ww = unit_vecs(60, SHAPE, rng)
        A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        data = dense_probe(A.to_dense(), ww)
        cases = [('pd_toy', _ToyModel([2.0, 3.0, 5.0, 7.0, 0.5]), _toy_rhs(5), 1e-9),
                 ('indefinite_toy', _ToyModel([2.0, -1.0, 3.0]), _toy_rhs(), 1e-12)]
        for gname, geom in (('manifold', bgeo.ManifoldGeometryOps()), ('corewise', bgeo.CorewiseGeometryOps())):
            pr = F.fit(geom, 'probe', ww, data)
            lm = pr.local_model(t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data)
            g = lm.gradient
            cases.append((gname, lm, cw.corewise_scale(g, -1.0), 1e-3 * float(geom.inner(g, g)) ** 0.5))
        for name, lm, rhs, tol in cases:
            with self.subTest(case=name):
                p0, i0, rs0, ok0 = _plain_cg_reference(lm, rhs, tol, 40)
                for p, i, rs, ok, boundary in (opt._cg_solve(lm, rhs, tol, 40, False),
                                               opt._cg_solve(lm, rhs, tol, 40, False, math.inf)):
                    self.assertEqual(_tree_bytes(p), _tree_bytes(p0))
                    self.assertEqual((int(i), np.float64(rs).tobytes(), bool(ok)),
                                     (int(i0), np.float64(rs0).tobytes(), bool(ok0)))
                    self.assertFalse(bool(boundary))

    def test_cg_truncated_excludes_boundary_stops(self):
        """`cg_truncated = (not cg_converged) and (not cg_ok) and (not cg_boundary)`: a Steihaug stop on
        nonpositive curvature (ok False) is a BOUNDARY stop, not a truncation. On the indefinite exact corewise
        Hessian with a trust region the boundary stops occur; every row's CG outcome is exactly one of
        converged / boundary / truncated / the cap, and the boundary rows are never `cg_truncated`."""
        rng = np.random.default_rng(1)
        np.random.seed(1)
        ww, pp = unit_vecs(30, SHAPE, rng), unit_vecs(30, SHAPE, rng)
        target = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        line = affine_jet(ww, pp)
        pr = F.fit(bgeo.CorewiseGeometryOps(), 'probe', line, t3.TuckerTensorTrain(*target).probe(line, (1,)), order=(1,))
        x0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        _, stats = pr.newton_cg(x0, max_newton=8, hessian='exact')
        rows = [r for r in stats['history'] if not r['converged']]
        for r in rows:
            cap = not (r['cg_converged'] or r['cg_boundary'] or r['cg_truncated'])
            self.assertEqual(sum([r['cg_converged'], r['cg_boundary'], r['cg_truncated'], cap]), 1, r)
            if r['cg_boundary']:
                self.assertFalse(r['cg_truncated'])
        _, i, _, ok, boundary = opt._cg_solve(_ToyModel([-1.0, -2.0, -3.0]), _toy_rhs(), 1e-12, 50, False, 0.5)
        self.assertTrue(bool(boundary) and not bool(ok))          # not ok AND boundary: the case the redefinition is for
        self.assertTrue(any(r['cg_boundary'] for r in rows), 'expected a trust-region boundary stop in this run')


class TestBackendOptimizers(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed for determinism
        self.rng = np.random.default_rng(0)
        self.A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).to_dense()
        self.X = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)

    # (geometry name, backend GeometryOps, frontend geometry singleton)
    GEOMS = [('corewise', bgeo.CorewiseGeometryOps(), t3m.COREWISE), ('manifold', bgeo.ManifoldGeometryOps(), t3m.MANIFOLD)]

    def _fit_and_dense(self, geom_b, kind_name, M=60):
        """A fit of `kind_name` data of self.A at M samples, and the dense sampler z(T) of a dense T, of the type of z."""
        rng, A = self.rng, self.A
        if kind_name == 'apply':
            sample = unit_vecs(M, SHAPE, rng); zfun = lambda T: dense_apply(T, sample)
        elif kind_name == 'probe':
            sample = unit_vecs(M, SHAPE, rng); zfun = lambda T: dense_probe(T, sample)
        else:  # entries
            flat = rng.choice(int(np.prod(SHAPE)), size=M, replace=False)
            sample = np.array(np.unravel_index(flat, SHAPE)); zfun = lambda T: T[tuple(sample)]
        return F.fit(geom_b, kind_name, sample, zfun(A)), zfun

    def test_model_matches_dense_ground_truth(self):
        """The backend LocalModel at self.X, against dense ground truth, for every (geometry, sampling kind) pair:
        the objective is 1/2 |z(X) - y|^2; the gradient's slope along a tangent p is <z(dX[p]), r>, with dX[p] the
        tangent's dense direction (on the manifold the tangent's dense tensor at the frame, on corewise the
        derivative of the cores' tensor, exactly by a polynomial difference quotient); gn_quadratic(p) and
        <p, hvp(p)> are |z(dX[p])|^2; jacobian(p) is z(dX[p]) and residual is r."""
        from t3toolbox.backend import tv_operations as tvo
        flat = lambda z: np.concatenate([np.ravel(a) for a in z]) if isinstance(z, (list, tuple)) else np.ravel(z)
        for gname, geom_b, geom_f in self.GEOMS:
            for kind in ('apply', 'entries', 'probe'):
                with self.subTest(geometry=gname, kind=kind):
                    fit, zfun = self._fit_and_dense(geom_b, kind)
                    lm = fit.local_model(self.X.data)
                    r = flat(zfun(self.X.to_dense())) - flat(zfun(self.A))
                    self.assertLess(abs(float(lm.objective) - 0.5 * float(r @ r)) / (0.5 * float(r @ r)), 1e-11)
                    np.random.seed(5)
                    frame_f = geom_f.frame(self.X)
                    p = geom_b.project(lm.frame, geom_f.randn(frame_f).variations.data)
                    if gname == 'manifold':                            # the tangent's dense tensor at the frame
                        dX = tvo.tv_to_dense(lm.frame, p, include_shift=False)
                    else:                                              # d/dt T(cores + t p) at 0: T is of degree 2d = 6
                        T = lambda t: t3.TuckerTensorTrain(*cw.corewise_add(self.X.data, cw.corewise_scale(p, t))
                                                           ).to_dense()                     # in t, so its odd part is
                        dX = (45 * (T(1) - T(-1)) - 9 * (T(2) - T(-2)) + (T(3) - T(-3))) / 60.0   # t, t^3, t^5: exact
                    Jp = flat(zfun(dX))                                # the samplers are linear in T
                    self.assertLess(abs(float(geom_b.inner(lm.gradient, p)) - float(Jp @ r)) / abs(float(Jp @ r)), 1e-9)
                    self.assertLess(abs(float(lm.gn_quadratic(p)) - float(Jp @ Jp)) / float(Jp @ Jp), 1e-9)
                    Hp = lm.hvp(p)
                    self.assertLess(abs(float(geom_b.inner(p, Hp)) - float(Jp @ Jp)) / float(Jp @ Jp), 1e-9)
                    PHp = geom_b.project(lm.frame, Hp)                         # hvp lies in the gauged space
                    self.assertLess(float(cw.corewise_norm(cw.corewise_sub(PHp, Hp))) / float(cw.corewise_norm(Hp)),
                                    1e-12)
                    self.assertLess(np.max(np.abs(flat(lm.jacobian(p)) - Jp)) / np.max(np.abs(Jp)), 1e-9)
                    self.assertLess(np.max(np.abs(flat(lm.residual) - r)) / np.max(np.abs(r)), 1e-12)

    def test_gradient_descent_descends(self):
        """Cauchy + Armijo gradient_descent decreases the loss monotonically on BOTH geometries (the
        Armijo line search is what keeps it robust on the additive corewise chart)."""
        def rms(arrs):
            ss = sum(float(np.sum(np.asarray(a) ** 2)) for a in arrs)
            return float(np.sqrt(ss / sum(np.asarray(a).size for a in arrs)))
        for gname, geom_b, geom_f in self.GEOMS:
            with self.subTest(geometry=gname):
                problem, _ = self._fit_and_dense(geom_b, 'probe')
                # rescale the start so the initial probes match the data magnitude (well-conditioned start)
                sc = (rms(problem.batch.y) / rms(self.X.probe(problem.batch.sample))) ** (1.0 / (len(TUCKER) + len(TT) - 1))
                x0 = (tuple(sc * C for C in self.X.data[0]), tuple(sc * C for C in self.X.data[1]))
                _, stats = problem.gradient_descent(x0, n_iter=60)
                L = stats['losses']
                self.assertTrue(all(L[i + 1] <= L[i] + 1e-9 * L[0] for i in range(len(L) - 1)), "not monotone")
                self.assertLess(L[-1], 0.5 * L[0], "did not make substantial progress")

    def test_stochastic_optimizers_recover(self):
        """mc_sgd (manifold) and adam (corewise) reduce the true error substantially from a scaled start."""
        rng = np.random.default_rng(2)
        ww = unit_vecs(200, SHAPE, rng); data = dense_probe(self.A, ww)
        def rms(arrs):
            ss = sum(float(np.sum(np.asarray(a) ** 2)) for a in arrs)
            return float(np.sqrt(ss / sum(np.asarray(a).size for a in arrs)))
        sc = (rms(data) / rms(self.X.probe(ww))) ** (1.0 / (len(TUCKER) + len(TT) - 1))
        x0 = (tuple(sc * C for C in self.X.data[0]), tuple(sc * C for C in self.X.data[1]))
        A_norm = float(np.linalg.norm(self.A))
        def true_err(cores):
            return float(np.linalg.norm(t3.TuckerTensorTrain(*cores).to_dense() - self.A)) / A_norm
        e0 = true_err(x0)

        with self.subTest(optimizer='mc_sgd', geometry='manifold'):
            pm = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data)
            xm, _ = pm.mc_sgd(x0, np.random.default_rng(3), batch_size=40, max_iter=500)
            self.assertLess(true_err(xm), 0.3 * e0)
        with self.subTest(optimizer='adam', geometry='corewise'):
            pc = F.fit(bgeo.CorewiseGeometryOps(), 'probe', ww, data)
            xc, _ = pc.adam(x0, np.random.default_rng(3), batch_size=40, lr=2e-2, max_iter=600)
            self.assertLess(true_err(xc), 0.3 * e0)

    def test_derivative_data(self):
        """Derivative data along lines (apply / entries / probe, a per-order weight ω) feed the SAME generic
        optimizers through the oracle: the apply or probe kind at the jet shape (n,) on affine_jet(ww, pp) (entries:
        apply at affine_jet(index_to_one_hot(index), pp)), the data in Taylor coefficients and the weight through
        prepare_derivative_data. Check: (1) the objective is the old derivative kinds' 1/2 sum_t ω_t² |d_t(X) - y_t|²
        on the derivatives themselves; (2) the corewise gradient matches a finite difference of it; (3) the default
        flat draw flattens a multi-axis W and builds a minibatch local model; (4) mc_sgd recovers from a zero start."""
        rng = np.random.default_rng(0)
        order, NP, NX = 2, 4, 3
        omega = np.array([1.0, 0.5, 0.3])
        Xtrue = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        X0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        ww = [rng.standard_normal((NP, NX, N)) for N in SHAPE]
        pp = [rng.standard_normal((NP, NX, N)) for N in SHAPE]
        index = np.stack([rng.integers(0, N, size=(NP, NX)) for N in SHAPE], axis=0)
        for name, sample in (('apply', ww), ('entries', index), ('probe', ww)):
            with self.subTest(kind=name):
                weight = np.tile(omega, (len(SHAPE), 1)) if name == 'probe' else omega
                data = dd.forward(Xtrue, name, sample, pp, order)
                prob = F.derivative_fit(bgeo.CorewiseGeometryOps(), name, order, sample, pp, data, weight=weight,
                                        shape=SHAPE)
                d0 = dd.forward(X0, name, sample, pp, order)                  # the derivatives at X0
                w = weight.reshape(weight.shape + (1,) * (np.ndim(data[0] if name == 'probe' else data) - 1))
                res = [w[i] * (a - b) for i, (a, b) in enumerate(zip(d0, data))] if name == 'probe' else \
                      [w * (np.asarray(d0) - np.asarray(data))]
                misfit = 0.5 * sum(float(np.sum(a ** 2)) for a in res)
                self.assertLess(abs(float(prob.objective(X0.data)) - misfit) / misfit, 1e-12)
                gB, gG = prob.local_model(X0.data).gradient
                dB_dir = [rng.standard_normal(b.shape) for b in X0.tucker_cores]
                dG = [rng.standard_normal(g.shape) for g in X0.tt_cores]
                inner = (sum(float(np.sum(np.asarray(gB[i]) * dB_dir[i])) for i in range(len(dB_dir)))
                         + sum(float(np.sum(np.asarray(gG[i]) * dG[i])) for i in range(len(dG))))
                eps = 1e-6
                plus = (tuple(b + eps * db for b, db in zip(X0.tucker_cores, dB_dir)),
                        tuple(g + eps * dg for g, dg in zip(X0.tt_cores, dG)))
                minus = (tuple(b - eps * db for b, db in zip(X0.tucker_cores, dB_dir)),
                         tuple(g - eps * dg for g, dg in zip(X0.tt_cores, dG)))
                fd = (float(prob.objective(plus)) - float(prob.objective(minus))) / (2 * eps)
                self.assertLess(abs(inner - fd) / max(abs(fd), 1e-30), 1e-5)
                # default flat draw: a multi-axis W=(NP,NX) flattens to NP*NX; a minibatch local model builds
                self.assertEqual(prob.oracle.n_measurements(prob.batch), NP * NX)
                drawn = opt.flat_draw(prob.oracle, prob.batch, 5)(rng)
                self.assertEqual(drawn.minibatch_scale, NP * NX / 5)
                self.assertTrue(np.isfinite(float(opt.local_model(prob.geom, prob.oracle, drawn, X0.data).objective)))
        # recovery: mc_sgd (manifold) from a zero start, apply-derivative data + the flat default draw
        prob_m = F.derivative_fit(bgeo.ManifoldGeometryOps(), 'apply', order, ww, pp,
                                  dd.forward(Xtrue, 'apply', ww, pp, order), weight=omega)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        _, stats = prob_m.mc_sgd(x0, np.random.default_rng(1), 6, max_iter=400)
        self.assertLess(stats['losses'][-1], 0.1 * stats['losses'][0])

    def test_newton_cg_recovers_to_high_accuracy(self):
        """Manifold Newton-CG (2nd-order) recovers an exact low-rank target to high accuracy from a zero
        start (the orthonormal frame completion makes the zero tensor a valid start). Eager numpy (float64)
        and use_jit=True (which auto-converts the numpy inputs to jax -> float32 -> jits) both recover; the
        jit path returns a jax-backed result."""
        rng = np.random.default_rng(4)
        ww = unit_vecs(300, SHAPE, rng); data = dense_probe(self.A, ww)
        problem = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        A_norm = float(np.linalg.norm(self.A))
        for use_jit in (False, True):
            with self.subTest(use_jit=use_jit):
                x, stats = problem.newton_cg(x0, max_newton=30, use_jit=use_jit)
                true_e = float(np.linalg.norm(np.asarray(t3.TuckerTensorTrain(*x).to_dense()) - self.A)) / A_norm
                self.assertLess(true_e, 1e-4)
                self.assertEqual(is_jax_ndarray(x[0][0]), use_jit)   # use_jit auto-converts numpy -> jax-backed

    def test_cg_solve_reports_state(self):
        """D1: `_cg_solve` returns `(p, iters, resid², ok, boundary)` -- converges to the exact solution on a
        PD operator (`ok` True, `resid ≤ tol`), and truncates on a nonpositive-curvature direction (`ok`
        False) when there is no trust region. A toy diagonal operator wrapped as a `(tucker, tt)` tree
        exercises the branch deterministically."""
        rhs = _toy_rhs()
        # PD: CG converges to D^-1 rhs, ok stays True, residual under tol
        p, i, rs, ok, boundary = opt._cg_solve(_ToyModel([2.0, 3.0, 5.0]), rhs, 1e-10, 50, False)
        self.assertTrue(bool(ok))
        self.assertFalse(bool(boundary))
        self.assertLessEqual(float(rs) ** 0.5, 1e-10)
        self.assertTrue(np.allclose(p[0][0], 1.0 / np.array([2.0, 3.0, 5.0])))
        # indefinite (negative-definite here): dᵀHd < 0 on the first direction -> immediate truncation
        p2, i2, rs2, ok2, boundary2 = opt._cg_solve(_ToyModel([-1.0, -2.0, -3.0]), rhs, 1e-10, 50, False)
        self.assertFalse(bool(ok2))                                 # truncated on nonpositive curvature
        self.assertFalse(bool(boundary2))                           # ... with no trust region: not a boundary stop
        self.assertGreater(float(rs2) ** 0.5, 1e-10)                # did NOT reach the tolerance

    def test_newton_cg_diagnostics(self):
        """D1: newton_cg returns a per-iteration `history`, fires `callback(NewtonInfo)` each iteration
        (carrying the LocalModel + point for per-block errors), and reports CG / line-search / ρ. The
        manifold Hessian is PD, so CG converges (not truncated); the final line is the converged one."""
        rng = np.random.default_rng(4)
        ww = unit_vecs(200, SHAPE, rng); data = dense_probe(self.A, ww)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        pm = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data)
        seen = []
        x, stats = pm.newton_cg(x0, max_newton=15, callback=seen.append)

        self.assertIn('losses', stats)
        self.assertNotIn('newton', stats)                                      # removed in 2026.3.0, not aliased
        for key in ('newton_steps', 'newton_accepted', 'stop_reason'):
            self.assertIn(key, stats)
        self.assertEqual(stats['stop_reason'], 'gradient')
        self.assertEqual(len(stats['history']), len(seen))                     # one record per callback
        self.assertGreater(len(seen), 1)
        for info in seen:
            self.assertIsInstance(info, opt.NewtonInfo)
            self.assertIsNotNone(info.lm)                                      # the model, for per-block errors
            self.assertIsNotNone(info.x_cores)
        self.assertTrue(seen[-1].converged)                                    # last line is the converged one
        self.assertIsNone(seen[-1].cg_iters)                                   # ... with no step info

        stepped = [i for i in seen if not i.converged]
        self.assertTrue(stepped)
        for info in stepped:
            self.assertLessEqual(info.cg_iters, 200)
            self.assertTrue(info.cg_converged and not info.cg_truncated)       # PD manifold H -> CG hits tol
            self.assertTrue(0.0 < info.alpha <= 1.0)
            self.assertTrue(np.isfinite(info.rho))
            self.assertGreaterEqual(info.wall_time, 0.0)
            self.assertLess(info.delta_f, 1e-9)                                # objective decreased (≤ 0)

        row = stats['history'][0]                                              # history rows are scalar-only
        self.assertNotIn('lm', row); self.assertNotIn('x_cores', row)
        self.assertEqual(set(row), set(opt._NEWTON_SCALAR_FIELDS))

    def test_newton_cg_g0norm_and_forcing_overrides(self):
        """The reference ‖g0‖ can be overridden per stopping test, with the chained fallback: `g0norm_newton`
        sets the Newton reference AND is inherited by CG unless `g0norm_cg` also given; `g0norm_cg` alone
        touches only CG. `cg_forcing_power` sets the exponent in η = min(0.5, (‖g‖/‖g0‖)**power). Checked by
        capturing NewtonInfo and reconstructing the reference each test used (uncapped regime, BIG >> g0)."""
        rng = np.random.default_rng(4)
        ww = unit_vecs(200, SHAPE, rng); data = dense_probe(self.A, ww)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        pm = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data)

        def run(**kw):
            seen = []; kw.setdefault('max_newton', 8)
            pm.newton_cg(x0, callback=seen.append, **kw)
            return seen

        # --- baseline (warm start): the reference is the computed initial gradient norm (== the first ‖g‖) ---
        base = run(g0_reference='warm_start')
        g0 = base[0].gnorm
        self.assertGreater(g0, 0.0)
        for info in base:
            self.assertAlmostEqual(info.g0norm, g0, places=12)              # reported reference == computed norm
            self.assertEqual((info.g0_reference, info.cg_reference), ('warm_start', 'warm_start'))
        BIG = 1.0e3 * g0                                                    # a much larger (warm-start-scale) reference

        # --- the default on the manifold: the zero reference, at the canonical zero-point frame ---
        zero = run()
        geom = pm.geom
        F0 = geom.zero_point_frame(pm.local_model(x0).frame)
        from t3toolbox.backend import tv_operations as tvo                  # dense: the ambient gradient at 0 is
        A0 = -sum(np.einsum(','.join('w' + 'abc'[j] for j in range(3)) + '->' + 'abc',   # -sum_i sum_w y_i (x) ww_(j!=i)
                            *[data[i] if j == i else ww[j] for j in range(3)]) for i in range(3))
        gz = tvo.tv_project_dense_onto_tangent_space(F0, A0)
        z0 = float(geom.inner(gz, gz)) ** 0.5                               # ‖Π Aᵀ(0 − data)‖ at F0, dense
        for info in zero:
            self.assertAlmostEqual(info.g0norm, z0, delta=1e-12 * z0)
            self.assertEqual((info.g0_reference, info.cg_reference), ('zero', 'zero'))
        self.assertAlmostEqual(zero[0].forcing_eta, min(0.5, (zero[0].gnorm / z0) ** 0.5), places=12)

        # --- g0norm_newton only: BOTH tests use BIG (CG inherits the Newton reference) ---
        n_only = run(g0norm_newton=BIG)
        for info in n_only:
            self.assertAlmostEqual(info.g0norm, BIG, places=6)             # Newton reference is BIG
            self.assertEqual((info.g0_reference, info.cg_reference), ('override', 'override'))
        s = n_only[0]
        self.assertFalse(s.converged)
        self.assertLess(s.forcing_eta, 0.5)                                # uncapped (gnorm/BIG tiny)
        self.assertAlmostEqual(s.forcing_eta, (s.gnorm / BIG) ** 0.5, places=10)   # CG inherited BIG

        # --- g0norm_cg only: CG uses BIG, the Newton reference is UNCHANGED ---
        c_only = run(g0norm_cg=BIG, g0_reference='warm_start')
        for info in c_only:
            self.assertAlmostEqual(info.g0norm, g0, places=12)             # Newton reference still the computed norm
            self.assertEqual((info.g0_reference, info.cg_reference), ('warm_start', 'override'))
        s = c_only[0]                                                       # iter 0: ‖g‖ == g0 (same start)
        self.assertAlmostEqual(s.forcing_eta, (g0 / BIG) ** 0.5, places=10)

        # --- both supplied: each test uses its own reference ---
        b = run(g0norm_newton=BIG, g0norm_cg=2.0 * BIG)[0]
        self.assertAlmostEqual(b.g0norm, BIG, places=6)                    # Newton reference
        self.assertAlmostEqual(b.forcing_eta, (b.gnorm / (2.0 * BIG)) ** 0.5, places=10)   # CG reference

        # --- cg_forcing_power: near the solution ‖g‖/‖g0‖ < 1, so a larger power tightens CG (smaller η) ---
        half = run(g0norm_cg=BIG, cg_forcing_power=0.5)[0]
        one  = run(g0norm_cg=BIG, cg_forcing_power=1.0)[0]
        self.assertAlmostEqual(half.forcing_eta, (g0 / BIG) ** 0.5, places=10)
        self.assertAlmostEqual(one.forcing_eta,  (g0 / BIG) ** 1.0, places=10)
        self.assertLess(one.forcing_eta, half.forcing_eta)                 # power 1.0 => tighter CG than 0.5

        # --- non-positive overrides are structural errors (0.0 used to crash in the forcing term) ---
        for kw in (dict(g0norm_newton=0.0), dict(g0norm_cg=-1.0), dict(g0norm_newton=float('nan'))):
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                with self.assertRaises(ValueError):
                    run(**kw)

    def test_newton_cg_stats_contract(self):
        """T29: `stats` is {losses, newton_steps, newton_accepted, stop_reason, history} with no `newton`;
        every row carries step_control='line_search' and the reference kinds; `stop_reason` sits on the last
        row only, with the same value as `stats['stop_reason']`; line-search rows have int `ls_steps`; and
        `losses[k] == history[k]['objective']`. The three stop reasons are each driven once."""
        rng = np.random.default_rng(4)
        ww = unit_vecs(200, SHAPE, rng); data = dense_probe(self.A, ww)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        pm = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data)

        def check_common(stats):
            self.assertEqual(set(stats), {'losses', 'newton_steps', 'newton_accepted', 'stop_reason', 'history'})
            h = stats['history']
            self.assertEqual(stats['losses'], [row['objective'] for row in h])
            self.assertEqual(stats['newton_steps'], sum(row['cg_iters'] is not None for row in h))
            for row in h:
                self.assertEqual(row['step_control'], 'line_search')
                self.assertEqual((row['g0_reference'], row['cg_reference']), ('zero', 'zero'))
                if not row['converged']:
                    self.assertIsInstance(row['ls_steps'], int)
            self.assertEqual(h[-1]['stop_reason'], stats['stop_reason'])
            self.assertTrue(all(row['stop_reason'] is None for row in h[:-1]))

        cases = {
            'gradient':    dict(max_newton=15),                                  # converges
            'max_newton':  dict(max_newton=2),                                   # the cap
            'line_search': dict(max_newton=5, c_armijo=10.0),                    # Armijo cannot be met
        }
        for reason, kw in cases.items():
            with self.subTest(stop_reason=reason):
                x, stats = pm.newton_cg(x0, **kw)
                check_common(stats)
                self.assertEqual(stats['stop_reason'], reason)
                h = stats['history']
                if reason == 'gradient':
                    self.assertTrue(h[-1]['converged'])
                    self.assertEqual(stats['newton_steps'], len(h) - 1)
                    self.assertEqual(stats['newton_accepted'], stats['newton_steps'])
                elif reason == 'max_newton':
                    self.assertEqual(len(h), 2)
                    self.assertEqual((stats['newton_steps'], stats['newton_accepted']), (2, 2))
                else:                                                        # rejected, x unchanged
                    self.assertEqual((stats['newton_steps'], stats['newton_accepted']), (1, 0))
                    self.assertTrue(h[-1]['ls_failed'])
                    self.assertTrue(all(a is b for a, b in zip(x[0] + x[1], x0[0] + x0[1])))
        # 'accept' takes the exhausted trial step: it counts as accepted, and the run ends on the cap
        _, stats = pm.newton_cg(x0, max_newton=2, c_armijo=10.0, on_line_search_failure='accept')
        check_common(stats)
        self.assertEqual((stats['stop_reason'], stats['newton_steps'], stats['newton_accepted']),
                         ('max_newton', 2, 2))
        self.assertTrue(all(row['ls_failed'] for row in stats['history']))

    def test_identity_regularizer_contributions(self):
        """IdentityRegularizer folds ρ=½λ‖X‖² into the local model on both geometries: `value` = ½λ‖X‖²
        (manifold HS ridge / corewise weight-decay), the TOTAL gradient FD-matches the objective, and
        gn_quadratic gains exactly λ‖Πp‖²."""
        rng = np.random.default_rng(1)
        ww = unit_vecs(80, SHAPE, rng); data = dense_probe(self.A, ww)
        X0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        lam = 0.37
        for gname, geom, _ in self.GEOMS:
            with self.subTest(geometry=gname):
                reg = opt.IdentityRegularizer(lam)
                prob  = F.fit(geom, 'probe', ww, data, regularizer=reg)
                prob0 = F.fit(geom, 'probe', ww, data)
                lm, lm0 = prob.local_model(X0.data), prob0.local_model(X0.data)
                frame = lm.frame
                base = geom.base_point(frame)                                # the frame's point: (D, L) or (B, G)
                # value: manifold = ½λ‖X‖²_HS ; corewise = ½λ Σ‖core‖²
                if gname == 'manifold':
                    rho_true = 0.5 * lam * float(np.linalg.norm(t3.TuckerTensorTrain(*base).to_dense())) ** 2
                else:
                    rho_true = 0.5 * lam * sum(float(np.sum(c * c)) for c in tuple(base[0]) + tuple(base[1]))
                self.assertAlmostEqual(float(reg.value(geom, base)), rho_true, places=5)
                # the TOTAL (data + reg) gradient FD-matches the objective along a projected direction
                p = geom.project(frame, t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data)
                slope = float(cw.corewise_dot(lm.gradient, geom.project(frame, p)))
                eps = 1e-6
                fd = (float(prob.objective(geom.retract(frame, cw.corewise_scale(p, eps))))
                      - float(prob.objective(geom.retract(frame, cw.corewise_scale(p, -eps))))) / (2 * eps)
                self.assertLess(abs(slope - fd) / abs(fd), 1e-5)
                # reg contribution to gn_quadratic is exactly λ‖Πp‖²
                Pp = geom.project(frame, p)
                self.assertAlmostEqual(float(lm.gn_quadratic(p)) - float(lm0.gn_quadratic(p)),
                                       lam * float(cw.corewise_dot(Pp, Pp)), places=5)

    def test_manifold_point_tangent_is_vX(self):
        """ManifoldGeometryOps.point_tangent(frame) = the attachment point as a gauged tangent v_X: dense(v_X) = X,
        and ‖X‖_HS = ‖L_last‖ = point_norm_sq**½, across structures (design §4)."""
        from t3toolbox.backend import tv_operations as tvo
        for shp, tk, tt in [((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)), ((4, 4, 4, 4), (2, 2, 2, 2), (1, 2, 3, 2, 1))]:
            with self.subTest(shape=shp):
                X = t3.TuckerTensorTrain.randn(shp, tk, tt); Xd = X.to_dense()
                frame = bgeo.ManifoldGeometryOps().frame(X.data)[0]
                vX = bgeo.ManifoldGeometryOps().point_tangent(frame)
                self.assertTrue(np.allclose(tvo.tv_to_dense(frame, vX, include_shift=False), Xd))   # dense(v_X)=X
                self.assertAlmostEqual(float(bgeo.ManifoldGeometryOps().point_norm_sq((frame[0], frame[2]))) ** 0.5,
                                       float(np.linalg.norm(Xd)), places=5)                          # ‖X‖=‖L_last‖

    def test_newton_cg_line_search_exhaustion(self):
        """R7-7: exhausting the Armijo search stops the run by default (rejecting the step) and is
        flagged ls_failed; alpha records the LAST TRIAL actually evaluated (2^-39, not one halving
        past it); 'accept' opts into taking the step and continuing (the escape mode)."""
        rng = np.random.default_rng(7)
        ww = unit_vecs(200, SHAPE, rng); data = dense_probe(self.A, ww)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        pm = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data)

        # default 'stop': the run terminates AT the first exhaustion, with the honest alpha
        x, stats = pm.newton_cg(x0, max_newton=60, gtol_rel=1e-30)
        hist = stats['history']
        failed = [row for row in hist if row.get('ls_failed')]
        self.assertEqual(len(failed), 1, 'expected exactly one failed row (the stop)')
        self.assertTrue(hist[-1]['ls_failed'])                     # ... and it is the last row
        self.assertLess(len(hist), 61)                             # terminated early
        self.assertEqual(hist[-1]['ls_steps'], 40)
        self.assertEqual(hist[-1]['alpha'], 2.0 ** -39)            # the last TRIAL's step (was 2^-40)
        for row in hist[:-1]:
            if row.get('ls_steps') is not None and not row.get('ls_failed'):
                self.assertEqual(row['alpha'], 2.0 ** -row['ls_steps'])   # alpha = 2^-ls_steps invariant

        # 'accept': the failed step is taken and iteration continues past it
        n_stop = len(hist)
        x2, stats2 = pm.newton_cg(x0, max_newton=n_stop + 3, gtol_rel=1e-30,
                                   on_line_search_failure='accept')
        hist2 = stats2['history']
        self.assertGreater(len(hist2), n_stop)                     # kept going past the exhaustion
        self.assertTrue(any(row.get('ls_failed') for row in hist2[:-1]))   # a failed row that is NOT last

        with self.assertRaises(ValueError):
            pm.newton_cg(x0, on_line_search_failure='explode')

    def test_gradient_descent_line_search_exhaustion(self):
        """R7-7 (gradient_descent): from a floored point the default stops with the flag set;
        'accept' runs on."""
        rng = np.random.default_rng(8)
        ww = unit_vecs(200, SHAPE, rng); data = dense_probe(self.A, ww)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        pm = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data)
        x_floor, _ = pm.newton_cg(x0, max_newton=60, gtol_rel=1e-30)   # drive to the floor

        x, stats = pm.gradient_descent(x_floor, n_iter=10, gtol_rel=1e-30)
        self.assertTrue(stats['line_search_failed'])
        self.assertLess(stats['n_iter'], 10)                       # stopped at the exhaustion

        x2, stats2 = pm.gradient_descent(x_floor, n_iter=5, gtol_rel=1e-30,
                                          on_line_search_failure='accept')
        self.assertEqual(stats2['n_iter'], 5)                      # pushed through
        with self.assertRaises(ValueError):
            pm.gradient_descent(x_floor, on_line_search_failure='explode')

    def test_regularized_newton_cg_shrinks(self):
        """Regularized manifold Newton-CG: λ=0 recovers the exact tensor; a larger λ (ridge) shrinks ‖x‖
        monotonically toward 0 while still converging. Corewise weight-decay likewise biases + converges."""
        rng = np.random.default_rng(3)
        ww = unit_vecs(300, SHAPE, rng); data = dense_probe(self.A, ww)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        A_norm = float(np.linalg.norm(self.A))
        nrm = lambda c: float(np.linalg.norm(t3.TuckerTensorTrain(*c).to_dense()))
        err = lambda c: float(np.linalg.norm(t3.TuckerTensorTrain(*c).to_dense() - self.A)) / A_norm
        res = {}
        for lam in (0.0, 1e-3, 1e-1):
            reg = opt.IdentityRegularizer(lam) if lam > 0 else None
            prob = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data, regularizer=reg)
            x, _ = prob.newton_cg(x0, max_newton=30)
            res[lam] = (err(x), nrm(x))
        self.assertLess(res[0.0][0], 1e-6)                              # λ=0 recovers exactly
        self.assertGreater(res[0.0][1], res[1e-3][1])                   # ridge shrinks ‖x‖ ...
        self.assertGreater(res[1e-3][1], res[1e-1][1])                  # ... monotonically toward 0

    def test_history_misfit_reg_split(self):
        """stats['history'] carries the objective split: each row's misfit + regularization == objective for
        a regularized run (reg ≥ 0); regularization is None (misfit == objective) with no regularizer."""
        rng = np.random.default_rng(4)
        ww = unit_vecs(300, SHAPE, rng); data = dense_probe(self.A, ww)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        prob = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data,
                                         regularizer=opt.IdentityRegularizer(1e-2))
        _, st = prob.newton_cg(x0, max_newton=8)
        for row in st['history']:                                       # regularized: obj = misfit + reg, reg ≥ 0
            self.assertIsNotNone(row['regularization'])
            self.assertGreaterEqual(row['regularization'], 0.0)
            self.assertAlmostEqual(row['misfit'] + row['regularization'], row['objective'],
                                   delta=1e-9 * abs(row['objective']) + 1e-12)
        prob0 = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data)
        _, st0 = prob0.newton_cg(x0, max_newton=8)
        for row in st0['history']:                                      # unregularized: reg None, misfit == obj
            self.assertIsNone(row['regularization'])
            self.assertEqual(row['misfit'], row['objective'])

    def test_uniform_point_norm_sq_garbage_robust(self):
        """S3/§7: the uniform manifold point_norm_sq equals ‖X‖²_HS and is robust to garbage in the input
        padding (1e6 into the masked-out region must not change the reduction -- the phantom-rank tripwire)."""
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.backend.uniform_fitting as uf
        import t3toolbox.backend.ut3_masking as ut3mask
        Xr = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        Xn2 = float(np.linalg.norm(Xr.to_dense())) ** 2
        Xum = uf.uniform_minimal(ut3.UniformTuckerTensorTrain.from_t3(Xr))
        tk, tt, shape, masks = Xum.data
        gops = bgeo.UniformManifoldGeometryOps.from_point(Xum.data)
        self.assertAlmostEqual(float(gops.point_norm_sq((tk, tt))) / Xn2, 1.0, places=5)     # == ‖X‖²_HS
        onem = ut3mask.ut3_apply_masks((np.ones_like(tk), np.ones_like(tt), shape, masks))   # 1 in-mask, 0 padding
        g = (np.asarray(tk) + 1e6 * (1.0 - np.asarray(onem[0])),
             np.asarray(tt) + 1e6 * (1.0 - np.asarray(onem[1])))
        self.assertAlmostEqual(float(gops.point_norm_sq(g)) / Xn2, 1.0, places=5)            # garbage-robust

    def test_stochastic_regularizer_is_not_scaled(self):
        """The minibatch rule (O-21; TARGET_DECISIONS 2026-11-21, item 2): a drawn batch's misfit is scaled by n/b
        (an unbiased estimate of the full misfit) and the regularizer is NOT scaled, so the step objective is
        (n/b) misfit_B + rho. Checked on the model a stochastic step builds: its objective, gradient and gn_quadratic
        against the drawn batch's unscaled model (minibatch_scale dropped) and the regularizer's own terms. The
        full-batch check keeps the full objective. mc_sgd + reg shrinks ‖x‖."""
        rng = np.random.default_rng(4)
        n, b = 200, 40
        ww = unit_vecs(n, SHAPE, rng); data = dense_probe(self.A, ww)
        reg = opt.IdentityRegularizer(0.5)
        prob = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data, regularizer=reg)
        geom, x = prob.geom, t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        drawn = opt.flat_draw(prob.oracle, prob.batch, b)(np.random.default_rng(1))
        self.assertEqual(drawn.minibatch_scale, n / b)
        plain = dc.replace(drawn, minibatch_scale=None)                          # the same samples, unscaled
        lm, lm_plain = (opt.local_model(geom, prob.oracle, B, x, regularizer=reg) for B in (drawn, plain))
        lm_free = opt.local_model(geom, prob.oracle, plain, x)                    # no regularizer
        s = n / b
        rho = float(reg.value(geom, geom.base_point(lm.frame)))
        self.assertGreater(rho, 0.0)
        self.assertAlmostEqual(float(lm.objective), s * float(lm_free.objective) + rho, delta=1e-10 * float(lm.objective))
        g, g_free = lm.gradient, lm_free.gradient
        g_reg = reg.gradient(geom, lm.frame)
        want = cw.corewise_add(cw.corewise_scale(g_free, s), g_reg)
        self.assertLess(float(cw.corewise_norm(cw.corewise_sub(g, want))) / float(cw.corewise_norm(want)), 1e-12)
        q_want = s * float(lm_free.gn_quadratic(g)) + float(reg.quadratic(geom, lm.frame, g))
        self.assertAlmostEqual(float(lm.gn_quadratic(g)), q_want, delta=1e-10 * q_want)
        self.assertAlmostEqual(float(lm_plain.objective), float(lm_free.objective) + rho,       # the unscaled control
                               delta=1e-10 * float(lm_plain.objective))
        self.assertAlmostEqual(float(prob.objective(x)), float(opt.local_model(geom, prob.oracle, prob.batch, x,
                                                                               regularizer=reg).objective),
                               delta=1e-10 * float(prob.objective(x)))           # the full-batch objective

        # mc_sgd (manifold) with reg shrinks ‖x‖ toward 0 vs the unregularized fit
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        nrm = lambda c: float(np.linalg.norm(t3.TuckerTensorTrain(*c).to_dense()))
        xu, _ = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data).mc_sgd(x0, np.random.default_rng(7), 50,
                                                                           max_iter=500)
        xr, _ = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data, regularizer=opt.IdentityRegularizer(0.3)
                      ).mc_sgd(x0, np.random.default_rng(7), 50, max_iter=500)
        self.assertLess(nrm(xr), nrm(xu))

    def test_jit_paths_recover(self):
        """With jax inputs + use_jit=True, newton_cg (jit CG), mc_sgd, and adam jit-compile their kernels
        (a stray np.* on a tracer would raise) and recover -- the jit dispatch check for the optimizers."""
        import jax
        import jax.numpy as jnp
        rng = np.random.default_rng(6)
        ww = unit_vecs(200, SHAPE, rng); data = dense_probe(self.A, ww)
        ww_j = [jnp.asarray(w) for w in ww]; data_j = [jnp.asarray(d) for d in data]
        A_norm = float(np.linalg.norm(self.A))
        def true_err(cores):
            return float(np.linalg.norm(np.asarray(t3.TuckerTensorTrain(*cores).to_dense()) - self.A)) / A_norm
        def rms(a): return float(np.sqrt(np.mean(np.concatenate([np.asarray(x).ravel() for x in a]) ** 2)))
        sc = (rms(data) / rms(self.X.probe(ww))) ** (1.0 / (len(TUCKER) + len(TT) - 1))
        x0_j = jax.tree_util.tree_map(jnp.asarray, (tuple(sc * C for C in self.X.data[0]),
                                                    tuple(sc * C for C in self.X.data[1])))
        e0 = true_err(x0_j)
        pm = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww_j, data_j)
        pc = F.fit(bgeo.CorewiseGeometryOps(), 'probe', ww_j, data_j)

        with self.subTest(optimizer='newton_cg'):
            x0z = jax.tree_util.tree_map(jnp.asarray, t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data)
            xn, _ = pm.newton_cg(x0z, max_newton=20, use_jit=True)
            self.assertIsInstance(xn[0][0], jnp.ndarray)
            self.assertLess(true_err(xn), 1e-3)
        with self.subTest(optimizer='newton_cg', regularized=True):     # the reg Hessian λ·Π runs inside the jit CG
            pm_reg = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww_j, data_j,
                                               regularizer=opt.IdentityRegularizer(1e-2))
            xr, _ = pm_reg.newton_cg(x0z, max_newton=15, use_jit=True)
            self.assertIsInstance(xr[0][0], jnp.ndarray)
        with self.subTest(optimizer='mc_sgd'):
            xm, _ = pm.mc_sgd(x0_j, np.random.default_rng(7), batch_size=40, max_iter=300, use_jit=True)
            self.assertIsInstance(xm[0][0], jnp.ndarray)
            self.assertLess(true_err(xm), 0.4 * e0)
        with self.subTest(optimizer='adam'):
            xa, _ = pc.adam(x0_j, np.random.default_rng(7), batch_size=40, lr=2e-2, max_iter=400, use_jit=True)
            self.assertIsInstance(xa[0][0], jnp.ndarray)
            self.assertLess(true_err(xa), 0.4 * e0)
        with self.subTest(optimizer='mc_sgd', kind='apply_derivatives'):
            # the derivative kind (leading order axis, paired (ww,pp) sample, ω weight) composes with jit
            order, NW = 2, 80
            wwa = [jnp.asarray(np.asarray(w)[:NW]) for w in ww]            # (NW, Ni) -- a subset of the W stack
            ppa = [jnp.asarray(rng.standard_normal((NW, N))) for N in SHAPE]
            da = jnp.ones((order + 1, NW))
            prob_d = F.derivative_fit(bgeo.ManifoldGeometryOps(), 'apply', order, wwa, ppa, da,
                                      weight=np.array([1.0, .5, .3]))
            xd, _ = prob_d.mc_sgd(x0z, np.random.default_rng(7), batch_size=20, max_iter=120, use_jit=True)
            self.assertIsInstance(xd[0][0], jnp.ndarray)        # jit-compiled (a stray np.* on a tracer raises)

    def test_use_jit_without_jax_warns_and_runs_eager(self):
        """use_jit=True with jax unavailable runs eager on numpy with a RuntimeWarning -- the library-wide
        jax-absent policy (review 2026-08-22, C10; it used to raise, and before 2026.1.0 it silently dropped
        the request). The guard fires at entry, before any kernel is built (common.jax_available patched
        off to simulate a jax-less install; the once-per-site memo is reset so the warning is observable)."""
        import warnings
        import t3toolbox.backend.common as common
        rng = np.random.default_rng(4)
        ww = unit_vecs(40, SHAPE, rng); data = dense_probe(self.A, ww)
        pm = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data)
        pc = F.fit(bgeo.CorewiseGeometryOps(), 'probe', ww, data)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        saved = common.jax_available
        common.jax_available = False
        try:
            for run in (lambda: pm.newton_cg(x0, max_newton=1, use_jit=True),
                        lambda: pm.mc_sgd(x0, rng, 20, max_iter=1, use_jit=True),
                        lambda: pc.adam(x0, rng, 20, max_iter=1, use_jit=True)):
                common._JAX_WARNED.discard('use_jit=True')
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    xj, _ = run()
                self.assertFalse(is_jax_ndarray(xj[0][0]))           # eager numpy result
                self.assertTrue(any('use_jit=True' in str(m.message) for m in w), [str(m.message) for m in w])
            # use_jit=False is unaffected -- eager on numpy, no warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                xf, _ = pm.newton_cg(x0, max_newton=1, use_jit=False)
            self.assertFalse(is_jax_ndarray(xf[0][0]))
            self.assertFalse(any('jax' in str(m.message) for m in w))
        finally:
            common.jax_available = saved
            common._JAX_WARNED.discard('use_jit=True')



# ---------------------------------------------------------------------------------------------------
# T15: the trust-region loop of newton_cg (step_control='trust_region', the exact Hessian's default)
# ---------------------------------------------------------------------------------------------------
_EPS64 = float(np.finfo(np.float64).eps)
_PD_SHAPE = (6, 6, 6)


def _pd_problem(seed, target_ranks=((3, 3, 3), (1, 3, 3, 1)), W=60, order=1, noise=0.01):
    """A fit of probe-derivative data (order ``order`` along lines; the probe kind at the jet shape ``(order,)``, which
    has the exact Hessian) on the manifold: data from a target T3 plus relative noise ``noise`` (0 = noise-free)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    target = t3.TuckerTensorTrain.randn(_PD_SHAPE, *target_ranks)
    ww = [rng.standard_normal((W, n)) for n in _PD_SHAPE]
    pp = [rng.standard_normal((W, n)) for n in _PD_SHAPE]
    data = [z + noise * np.linalg.norm(z) / math.sqrt(z.size) * rng.standard_normal(z.shape)
            for z in dd.forward(target, 'probe', ww, pp, order)]
    return F.derivative_fit(bgeo.ManifoldGeometryOps(), 'probe', order, ww, pp, data)


def _rank2_start(seed=5):
    np.random.seed(seed)
    return t3.TuckerTensorTrain.randn(_PD_SHAPE, (2, 2, 2), (1, 2, 2, 1)).data


def _pred(row):
    """The trust-region model change of a history row, gᵀp + ½pᵀHp (α = 1)."""
    return row['slope'] + 0.5 * row['pHp']


def _resolution(problem, row):
    """ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|) at a row's point: the objective's floating-point resolution (float64), by hand from the
    batch (the Taylor data and weight: ω⊙y is the same on the derivatives)."""
    y, w = problem.batch.y, problem.batch.weight
    wy = math.sqrt(sum(float(np.sum((w[i].reshape((-1,) + (1,) * (np.ndim(a) - 1)) * a) ** 2)) for i, a in enumerate(y)))
    return _EPS64 * (wy * math.sqrt(2.0 * row['misfit']) + abs(row['objective']))


def _point_key(frame):
    """A fingerprint of a corewise frame's cores (the point itself): the sum of squares of its core entries, bitwise
    reproducible for the same arrays, different for any other point the tests meet."""
    def leaves(t):
        return [t] if hasattr(t, 'shape') else [a for s in t for a in leaves(s)]
    return float(sum(np.sum(np.asarray(a, dtype=float) ** 2) for a in leaves(frame[:2])))


@dc.dataclass(frozen=True, eq=False)
class _OffsetOracle(LeastSquaresOracle):
    """A least-squares oracle whose ``f`` at every POINT other than the run's start (a corewise frame whose cores are
    not the start's, ``start_key``: where the optimizers score a trial step) is raised by a constant. While the run
    stays at its start, every trial then looks like an ascent, so every one is rejected while the model stays accurate
    -- the only way to drive the radius to its floor with |pred| far above the resolution. The mechanism does not depend
    on which route reads the current point's value: that point IS the start, unshifted on either route (D9). (A user
    oracle: the fitting layer's extension point.)"""
    offset: float = 0.0
    start_key: typ.Optional[float] = None            # None: every point is shifted

    def query(self, frame, batch, want, **kwargs):
        values, held = super().query(frame, batch, want, **kwargs)
        if frame_form(frame)[1] == 'corewise' and (self.start_key is None or _point_key(frame) != self.start_key):
            values = tuple(v + self.offset if name == 'f' else v for name, v in zip(want, values))
        return values, held


def _offset(fit, offset, x0=None):
    """``fit`` with its oracle's objective raised by ``offset`` away from the start ``x0`` (:class:`_OffsetOracle`)."""
    key = None if x0 is None else _point_key(fit.geom.point_data(x0))
    return fit.with_(oracle=_OffsetOracle(fit.oracle.kind, fit.oracle.order, offset, key))


def _restart_problem(W=60, order=1):
    """E2's degenerate restart (c21 case A, on probe derivatives): noise-free data of a rank-2 target, fitted
    at rank 2 to 1e-10, then zero-padded to rank 3 -- the start is already at the objective's floor, a few
    steps from its resolution."""
    rng = np.random.default_rng(7)
    np.random.seed(7)
    target = t3.TuckerTensorTrain.randn(_PD_SHAPE, (2, 2, 2), (1, 2, 2, 1))
    ww = [rng.standard_normal((W, n)) for n in _PD_SHAPE]
    pp = [rng.standard_normal((W, n)) for n in _PD_SHAPE]
    problem = F.derivative_fit(bgeo.ManifoldGeometryOps(), 'probe', order, ww, pp,
                               dd.forward(target, 'probe', ww, pp, order))
    x1, _ = problem.newton_cg(t3.TuckerTensorTrain.randn(_PD_SHAPE, (2, 2, 2), (1, 2, 2, 1)).data,
                              max_newton=80, gtol_rel=1e-10)
    return problem, t3.TuckerTensorTrain(*x1).resize(_PD_SHAPE, (3, 3, 3), (1, 3, 3, 1)).data


class TestTrustRegionNewton(unittest.TestCase):
    """T15: the trust-region loop -- rejection keeps x and the model, the radius rules (incl. Δ_max), the
    descent guard, the bookkeeping, and the two floors (a)-(e)."""

    def check_bookkeeping(self, stats, exact=True):
        """`losses[k] == history[k]['objective']`; newton_steps = the CG solves; newton_accepted = the accepted
        trials; stop_reason on the last row only; trust-region rows carry tr_* and no line-search fields."""
        h = stats['history']
        self.assertEqual(set(stats), {'losses', 'newton_steps', 'newton_accepted', 'stop_reason', 'history'})
        self.assertEqual(stats['losses'], [row['objective'] for row in h])
        stepped = [row for row in h if not row['converged']]
        self.assertEqual(stats['newton_steps'], len(stepped))
        self.assertEqual(stats['newton_accepted'], sum(row['tr_accepted'] is True for row in stepped))
        self.assertEqual(h[-1]['stop_reason'], stats['stop_reason'])
        self.assertTrue(all(row['stop_reason'] is None for row in h[:-1]))
        for row, nxt in zip(h, h[1:]):                       # D9: the carried value (accepted: to delta_f's rounding)
            if row['tr_accepted']:
                self.assertAlmostEqual(nxt['objective'], row['objective'] + row['delta_f'],
                                       delta=1e-15 * max(abs(nxt['objective']), abs(row['objective'])))
            else:
                self.assertEqual(nxt['objective'], row['objective'])
        for row in h:
            self.assertEqual((row['step_control'], row['hessian']), ('trust_region', 'exact' if exact else 'gauss_newton'))
        for row in stepped:
            self.assertIsNone(row['ls_steps']); self.assertIsNone(row['ls_failed'])
            self.assertEqual(row['alpha'], 1.0)
            self.assertIsInstance(row['tr_accepted'], bool)
            self.assertGreater(row['tr_radius'], 0.0)
            self.assertEqual(row['pKp'] is not None, exact)
            self.assertEqual(row['tr_accepted'], math.isfinite(row['rho']) and row['rho'] > 0.1)
            if exact:                                                               # the manifold d = 3 lift diagnostics
                self.assertGreaterEqual(row['lift_sigma_min'], 0.0)                 # 0 at a zero-padded restart
                self.assertGreater(row['lift_eps'], 0.0)

    def test_rejection_keeps_x_and_reuses_the_model(self):
        """A rejected trial emits its own row (tr_accepted=False), keeps x and the local model -- the next row
        starts from the SAME point object with the SAME model object, and counts toward max_newton; an accepted
        trial moves x and rebuilds the model."""
        problem = _pd_problem(0)
        seen = []
        x, stats = problem.newton_cg(_rank2_start(), max_newton=10, hessian='exact', tr_radius0=1e3,
                                 callback=seen.append)
        self.check_bookkeeping(stats)
        self.assertEqual(len(seen), 10)                                              # rejections count toward the cap
        self.assertEqual(stats['stop_reason'], 'max_newton')
        rejected = [k for k in range(len(seen) - 1) if seen[k].tr_accepted is False]
        accepted = [k for k in range(len(seen) - 1) if seen[k].tr_accepted is True]
        self.assertTrue(rejected and accepted)
        for k in rejected:
            self.assertIs(seen[k + 1].x_cores, seen[k].x_cores)
            self.assertIs(seen[k + 1].lm, seen[k].lm)
            self.assertEqual((seen[k + 1].objective, seen[k + 1].gnorm), (seen[k].objective, seen[k].gnorm))
        for k in accepted:
            self.assertIsNot(seen[k + 1].x_cores, seen[k].x_cores)
            self.assertIsNot(seen[k + 1].lm, seen[k].lm)
        self.assertEqual(len({id(info.lm) for info in seen}), 1 + len(accepted))     # one model per accepted point

    def test_radius_rules(self):
        """Δ ← ¼‖p‖ when ρ < 0.25 (or not finite); Δ ← min(2Δ, Δ_max) when ρ > 0.75 on a boundary step;
        unchanged otherwise. Δ_max = 1e4·Δ0 caps a long run of expansions from a tiny Δ0; the default Δ0 is the
        Gauss-Newton Cauchy length ‖g‖³/⟨g, H_GN g⟩."""
        problem = _pd_problem(0)
        x0 = _rank2_start()
        lm = problem.local_model(x0)
        g = lm.gradient
        gnorm = float(problem.geom.inner(g, g)) ** 0.5
        cauchy = gnorm ** 3 / float(lm.gn_quadratic(g))
        cases = {'default': dict(), 'large': dict(tr_radius0=1e3), 'tiny': dict(tr_radius0=1e-6 * cauchy)}
        for name, kw in cases.items():
            with self.subTest(radius0=name):
                seen = []
                _, stats = problem.newton_cg(x0, max_newton=18 if name == 'tiny' else 10, hessian='exact',
                                         callback=seen.append, **kw)
                self.check_bookkeeping(stats)
                r0 = kw.get('tr_radius0', cauchy)
                self.assertAlmostEqual(seen[0].tr_radius, r0, delta=1e-12 * r0)
                rules = set()
                for a, b in zip(seen, seen[1:]):
                    p_norm = a.step_rel * float(cw.corewise_norm(a.x_cores))           # ‖p‖ = step_rel·‖x‖ (α = 1)
                    if (not math.isfinite(a.rho)) or a.rho < 0.25:
                        expected, rule = 0.25 * p_norm, 'shrink'
                    elif a.rho > 0.75 and a.cg_boundary:
                        expected, rule = min(2.0 * a.tr_radius, 1e4 * r0), 'expand'
                    else:
                        expected, rule = a.tr_radius, 'keep'
                    rules.add(rule)
                    self.assertAlmostEqual(b.tr_radius, expected, delta=1e-12 * expected)
                    self.assertLessEqual(b.tr_radius, 1e4 * r0)
                if name == 'large':
                    self.assertEqual(rules, {'shrink', 'expand', 'keep'})
                if name == 'tiny':                                                  # expansions, then the cap
                    self.assertEqual(stats['newton_accepted'], len(seen))
                    self.assertEqual(seen[-1].tr_radius, 1e4 * r0)
                    self.assertTrue(all(b.tr_radius == 2.0 * a.tr_radius for a, b in zip(seen[:13], seen[1:14])))

    def test_descent_guard(self):
        """If CG returns a non-descent direction, the trust-region step is −g scaled into the region:
        p = −g·min(1, Δ/‖g‖) (and the line-search step is −g, as before). CG is replaced by an ascent stub."""
        problem = _pd_problem(0)
        x0 = _rank2_start()
        g = problem.local_model(x0).gradient
        gnorm = float(problem.geom.inner(g, g)) ** 0.5
        x_norm = float(cw.corewise_norm(x0))

        def ascent(lm, rhs, tol, maxiter, use_jit, radius=math.inf):         # p = +g: slope ≥ 0 -> the guard fires
            return cw.corewise_scale(rhs, -1.0), 1, 0.0, True, False

        for step_control, r0 in (('trust_region', 0.1 * gnorm), ('trust_region', 10.0 * gnorm), ('line_search', None)):
            with self.subTest(step_control=step_control, radius0=r0):
                with unittest.mock.patch.object(opt, '_cg_solve', ascent):
                    _, stats = problem.newton_cg(x0, max_newton=1, step_control=step_control, tr_radius0=r0)
                row = stats['history'][0]
                scale = min(1.0, r0 / gnorm) if r0 is not None else 1.0
                self.assertAlmostEqual(row['slope'], -scale * gnorm ** 2, delta=1e-12 * gnorm ** 2)
                step = row['step_rel'] * x_norm / row['alpha']                       # ‖p‖ (the LS α is 2^-ls_steps)
                self.assertAlmostEqual(step, scale * gnorm, delta=1e-10 * scale * gnorm)
        # A guard step shortened to the region (scale < 1) is limited by Δ, so it counts as a boundary step for
        # the expansion rule (CG's own cg_boundary flag stays False): ρ > 0.75 doubles Δ on the next row.
        real, calls = opt._cg_solve, []

        def ascent_once(lm, rhs, tol, maxiter, use_jit, radius=math.inf):    # the stub on the first solve only
            calls.append(radius)
            return (ascent if len(calls) == 1 else real)(lm, rhs, tol, maxiter, use_jit, radius)

        with unittest.mock.patch.object(opt, '_cg_solve', ascent_once):
            _, stats = problem.newton_cg(x0, max_newton=2, step_control='trust_region', tr_radius0=1e-5 * gnorm)
        first, second = stats['history'][:2]
        self.assertTrue(first['tr_accepted'] and first['rho'] > 0.75 and not first['cg_boundary'], first)
        self.assertEqual(second['tr_radius'], 2.0 * first['tr_radius'])

    def test_shrink_is_a_quarter_of_the_step_not_of_the_radius(self):
        """Δ ← ¼‖p‖, not ¼Δ: after a rejected INTERIOR step (CG converged well inside a large region) the next
        radius is a quarter of that step's length, far below a quarter of the old radius."""
        base = _pd_problem(0)
        problem = _offset(base, 10.0 * float(base.objective(_rank2_start())), _rank2_start())   # every trial rejected
        _, stats = problem.newton_cg(_rank2_start(), max_newton=2, step_control='trust_region', tr_radius0=1e3)
        first, second = stats['history']
        p_norm = first['step_rel'] * float(cw.corewise_norm(_rank2_start()))
        self.assertFalse(first['tr_accepted'] or first['cg_boundary'])
        self.assertLess(p_norm, 0.5 * first['tr_radius'])                                   # an interior step
        self.assertAlmostEqual(second['tr_radius'], 0.25 * p_norm, delta=1e-12 * p_norm)

    def test_floor_a_degenerate_restart(self):
        """(a) E2's degenerate restart: accepted steps first, then the first rejection's predicted change is below
        the objective's resolution -> 'objective_floor' (converged=False: not the gradient test, not a failure)."""
        problem, x = _restart_problem()
        _, stats = problem.newton_cg(x, max_newton=40, hessian='exact', g0_reference='warm_start')
        self.check_bookkeeping(stats)
        h = stats['history']
        self.assertEqual(stats['stop_reason'], 'objective_floor')
        self.assertGreaterEqual(stats['newton_accepted'], 1)
        self.assertFalse(h[-1]['converged'])
        self.assertIs(h[-1]['tr_accepted'], False)
        self.assertTrue(all(row['tr_accepted'] for row in h[:-1]))                  # the floor fired at the FIRST rejection
        self.assertLessEqual(abs(_pred(h[-1])), 10.0 * _resolution(problem, h[-1]))

    def test_floor_b_near_stationary_start(self):
        """(b) A misspecified fit already at its floor: the first trial is rejected and its predicted change is
        below the resolution -> 'objective_floor' at once."""
        problem = _pd_problem(1, noise=0.01)
        x, _ = problem.newton_cg(_rank2_start(), max_newton=40, gtol_rel=0.0)    # GN to the floor
        _, stats = problem.newton_cg(x, max_newton=10, hessian='exact', g0_reference='warm_start', gtol_rel=1e-14)
        self.check_bookkeeping(stats)
        h = stats['history']
        self.assertEqual(stats['stop_reason'], 'objective_floor')
        self.assertIs(h[-1]['tr_accepted'], False)
        self.assertEqual(stats['newton_accepted'], len(h) - 1)
        self.assertLessEqual(abs(_pred(h[-1])), 10.0 * _resolution(problem, h[-1]))

    def test_floor_c_accepted_steps_below_the_resolution_never_stop(self):
        """(c) The floor is checked on REJECTIONS only. A forced-tiny Δ0 (set so the first predicted change is
        ~5x the resolution) gives accepted steps whose |pred| is within the objective_floor bound -- a rule
        checked on every step would stop at row 0; this one keeps going and expands Δ."""
        problem = _pd_problem(0)
        x0 = _rank2_start()
        lm = problem.local_model(x0)
        g = lm.gradient
        gnorm = float(problem.geom.inner(g, g)) ** 0.5
        row0 = dict(misfit=float(lm.misfit), objective=float(lm.objective))
        _, stats = problem.newton_cg(x0, max_newton=4, hessian='exact',
                                 tr_radius0=5.0 * _resolution(problem, row0) / gnorm)
        self.check_bookkeeping(stats)
        h = stats['history']
        self.assertEqual((stats['stop_reason'], len(h), stats['newton_accepted']), ('max_newton', 4, 4))
        self.assertLessEqual(abs(_pred(h[0])), 10.0 * _resolution(problem, h[0]))     # would have been a floor stop
        self.assertTrue(all(b['tr_radius'] == 2.0 * a['tr_radius'] for a, b in zip(h, h[1:])))

    def test_floor_d_trust_region_floor(self):
        """(d) 'trust_region_floor' fires only after rejections drive Δ below 1e-12·Δ0 -- here every trial is
        rejected (every point but the start carries an offset, _OffsetOracle) while |pred| stays far above
        the resolution, so the objective floor never applies; with GN + trust region too."""
        base = _pd_problem(0)
        f0 = float(base.objective(_rank2_start()))
        problem = _offset(base, 10.0 * f0, _rank2_start())
        for hessian in ('exact', 'gauss_newton'):
            with self.subTest(hessian=hessian):
                _, stats = problem.newton_cg(_rank2_start(), max_newton=60, hessian=hessian,
                                         step_control='trust_region')
                self.check_bookkeeping(stats, exact=hessian == 'exact')
                h = stats['history']
                self.assertEqual(stats['stop_reason'], 'trust_region_floor')
                self.assertEqual(stats['newton_accepted'], 0)
                r0 = h[0]['tr_radius']
                for row, nxt in zip(h, h[1:]):
                    self.assertGreaterEqual(nxt['tr_radius'], 1e-12 * r0)                 # not yet below the floor
                    self.assertLessEqual(nxt['tr_radius'], 0.25 * row['tr_radius'] * (1 + 1e-12))
                p_last = h[-1]['step_rel'] * float(cw.corewise_norm(_rank2_start()))
                self.assertLess(0.25 * p_last, 1e-12 * r0)                                 # ... and now it is
                for row in h:
                    self.assertGreater(abs(_pred(row)), 10.0 * _resolution(problem, row))

    def test_floor_e_resolution_eps(self):
        """(e) The resolution's ε is the residual DTYPE's machine epsilon (host-side, no array read), the
        largest over the floating leaves of a mixed tree; float64's when there is none. (The float32 end-to-end
        run, use_jit=True with x64 off, is in tests/test_dispatch.py: it is a jit compile.)"""
        f32, f64 = np.zeros(3, np.float32), np.zeros(3, np.float64)
        e32, e64 = float(np.finfo(np.float32).eps), float(np.finfo(np.float64).eps)
        self.assertEqual(opt._resolution_eps(f64), e64)
        self.assertEqual(opt._resolution_eps(f32), e32)
        self.assertEqual(opt._resolution_eps([f64, (f64, f64)]), e64)
        self.assertEqual(opt._resolution_eps([f64, (f32, f64)]), e32)                  # mixed: the coarser one
        self.assertEqual(opt._resolution_eps((np.zeros(2, np.int64), f64)), e64)       # integer leaves ignored
        self.assertEqual(opt._resolution_eps(()), e64)

    def test_option_validation(self):
        """The new keywords' structural errors (n16: an option with no effect raises)."""
        problem = _pd_problem(0)
        x0 = _rank2_start()
        bad = [dict(hessian='newton'), dict(hessian=True), dict(step_control='dogleg'),
               dict(curvature_damping=1e-3),                                       # GN: no lift to damp
               dict(tr_radius0=1.0),                                               # GN default: line search
               dict(hessian='exact', step_control='line_search', tr_radius0=1.0),
               dict(hessian='exact', tr_radius0=0.0), dict(hessian='exact', tr_radius0=float('inf')),
               dict(hessian='exact', on_line_search_failure='accept')]             # no line search to accept
        for kw in bad:
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                with self.assertRaises(ValueError):
                    problem.newton_cg(x0, max_newton=1, **kw)
        for kw, mode in ((dict(), 'line_search'), (dict(hessian='EXACT'), 'trust_region'),
                         (dict(hessian='exact', step_control='Line_Search'), 'line_search'),
                         (dict(step_control='trust_region'), 'trust_region')):
            with self.subTest(**kw):
                _, stats = problem.newton_cg(x0, max_newton=2, **kw)
                self.assertEqual(stats['history'][0]['step_control'], mode)

    def test_exact_line_search_and_gauss_newton_trust_region(self):
        """Both step controls combine with both Hessians: exact + line search records the curvature (pKp, the
        lift diagnostics) on Armijo rows; Gauss-Newton + trust region has no exact fields."""
        problem = _pd_problem(0)
        x0 = _rank2_start()
        _, ls = problem.newton_cg(x0, max_newton=4, hessian='exact', step_control='line_search')
        for row in ls['history']:
            self.assertEqual((row['hessian'], row['step_control']), ('exact', 'line_search'))
            if not row['converged']:
                self.assertIsInstance(row['ls_steps'], int)
                self.assertIsNone(row['tr_radius']); self.assertIsNone(row['tr_accepted'])
                self.assertIsInstance(row['pKp'], float)
                self.assertFalse(row['cg_boundary'])                                # no trust region, no boundary
        _, tr = problem.newton_cg(x0, max_newton=4, step_control='trust_region')
        self.check_bookkeeping(tr, exact=False)
        for row in tr['history']:
            self.assertIsNone(row['pKp']); self.assertIsNone(row['lift_sigma_min']); self.assertIsNone(row['lift_eps'])


class TestProblemContracts(unittest.TestCase):
    """Contracts of the objective at a point and of the model's build (the 2026-08-22 review's S6 and S7, on the oracle
    seam): another batch is another fit, scored on its own data; centers come with the frame they belong to; and the
    regularized objective is exact at a RAW, non-left-orthogonal point (S7)."""

    def _problem(self, regularizer=None):
        np.random.seed(3)
        x_true = t3.TuckerTensorTrain.randn((4, 5, 6), (3, 3, 3), (1, 2, 2, 1))
        ww = tuple(np.random.randn(20, n) for n in (4, 5, 6))
        b = x_true.apply(ww)
        return F.fit(bgeo.ManifoldGeometryOps(), 'apply', ww, b, regularizer=regularizer), ww, b

    def test_another_batch_is_scored_on_its_own_data(self):
        """S6's point on the seam: there is no (sample, data) pair to mismatch -- a batch carries both -- and the
        objective at a point and the model on another batch score that batch's data, not the training data."""
        problem, ww, b = self._problem()
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
        other = LeastSquaresBatch(ww, b + 5.0)
        truth = 0.5 * float(np.sum((x.apply(ww) - (b + 5.0)) ** 2))
        self.assertAlmostEqual(float(problem.objective(x.data, batch=other)), truth, places=8)
        self.assertAlmostEqual(float(opt.local_model(problem.geom, problem.oracle, other, x.data).objective), truth,
                               places=8)
        self.assertNotAlmostEqual(float(problem.objective(x.data)), truth, places=3)

    def test_centers_need_their_frame(self):
        """D3: centers= without frame= raises (structural); with the frame they belong to, the model is the
        geom.frame model."""
        problem, ww, b = self._problem()
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
        frame, centers = problem.geom.frame(x.data)
        with self.assertRaises(ValueError):
            problem.local_model(x.data, centers=centers)
        self.assertEqual(float(problem.local_model(x.data, frame=frame, centers=centers).objective),
                         float(problem.local_model(x.data).objective))

    def test_regularized_objective_exact_at_a_raw_point(self):
        # point_norm_sq read ‖last TT core‖², exact only for a left-orthogonal point; a randn point gave
        # reg term 3 where the truth was 1400. Now exact for any point (ragged == uniform == dense).
        import t3toolbox.backend.regularization as reg
        import t3toolbox.uniform_tucker_tensor_train as ut3
        lam = 0.7
        problem, ww, b = self._problem(reg.IdentityRegularizer(lam))
        for ttr in [(1, 2, 2, 1), (2, 2, 2, 3)]:                     # incl. unsquashed tails
            with self.subTest(tt=ttr):
                x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), ttr)
                truth = 0.5 * float(np.sum((x.apply(ww) - b) ** 2)) + 0.5 * lam * float(x.norm()) ** 2
                self.assertLess(abs(float(problem.objective(x.data)) - truth) / truth, 1e-10)
                geom = bgeo.ManifoldGeometryOps()
                self.assertLess(abs(float(geom.point_norm_sq(x.data)) - float(x.norm()) ** 2) / truth, 1e-10)
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                ugeom = bgeo.UniformManifoldGeometryOps.from_point(ux.data)
                self.assertLess(abs(float(ugeom.point_norm_sq(ux.data[:2])) - float(x.norm()) ** 2) / truth, 1e-10)


class TestOracleSeam(unittest.TestCase):
    """The graded-messages seam (P16 part 2): what the optimizers ask of the oracle, and what they must never do with
    ``held`` (valid for one frame and one batch: TARGET_DECISIONS, the notes carried to P16, item 1)."""

    def setUp(self):
        np.random.seed(0)
        self.rng = np.random.default_rng(0)
        self.A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)

    # ---- helpers -------------------------------------------------------------------------------------------------
    @staticmethod
    def _direct_gradient(fit, x, batch):
        """The model's gradient, gn_quadratic of it and objective at x on batch, by DIRECT oracle queries with no
        held and no LocalModel (the reference the optimizer loops are compared with)."""
        geom, oracle, reg = fit.geom, fit.oracle, fit.regularizer
        frame, centers = geom.frame(x)
        Vc = None if centers is None else centers[0]
        aux = geom.precompute(frame)
        (f, g), _ = oracle.query(frame, batch, ('f', 'g'), Vc=Vc)
        g = geom.project(frame, g, aux=aux)
        if reg is not None:
            g = cw.corewise_add(g, reg.gradient(geom, frame, aux=aux))
            f = f + reg.value(geom, geom.base_point(frame))
        (q,), _ = oracle.query(frame, batch, ('Jp_norm_sq',), Vc=Vc, variations=geom.project(frame, g, aux=aux))
        if reg is not None:
            q = q + reg.quadratic(geom, frame, g, aux=aux)
        return frame, aux, g, q, f

    def _probe_fit(self, geometry='manifold', n=60, regularizer=None):
        ww = unit_vecs(n, SHAPE, self.rng)
        geom = bgeo.ManifoldGeometryOps() if geometry == 'manifold' else bgeo.CorewiseGeometryOps()
        return F.fit(geom, 'probe', ww, dense_probe(self.A.to_dense(), ww), regularizer=regularizer)

    @staticmethod
    def _rules_run(call):
        """The engine rules ``call()`` runs (the engine's ``_run`` recorded)."""
        import t3toolbox.backend.sampling_variables as sv
        ran, real = [], sv._run

        def recording(rule, *args, **kwargs):
            ran.append(rule.name)
            return real(rule, *args, **kwargs)
        with unittest.mock.patch.object(sv, '_run', recording):
            call()
        return ran

    # ---- the head ------------------------------------------------------------------------------------------------
    def test_the_heads(self):
        """The four optimizers' heads (plan 2.4, N30): (geom, oracle, batch, x0, ...), regularizer after the positional
        arguments; local_model and point_objective likewise; padded_start (D2) and flat_draw."""
        import inspect
        heads = {'newton_cg': ['geom', 'oracle', 'batch', 'x0', 'regularizer'],
                 'gradient_descent': ['geom', 'oracle', 'batch', 'x0', 'regularizer'],
                 'mc_sgd': ['geom', 'oracle', 'batch', 'x0', 'rng', 'batch_size', 'regularizer', 'draw'],
                 'adam': ['geom', 'oracle', 'batch', 'x0', 'rng', 'batch_size', 'regularizer', 'draw'],
                 'local_model': ['geom', 'oracle', 'batch', 'x_cores', 'regularizer', 'hessian', 'curvature_damping',
                                 'frame', 'centers'],
                 'point_objective': ['geom', 'oracle', 'batch', 'x_cores', 'regularizer'],
                 'padded_start': ['geom', 'oracle', 'batch', 'x0', 'x_old'],
                 'flat_draw': ['oracle', 'batch', 'batch_size']}
        for name, head in heads.items():
            with self.subTest(function=name):
                self.assertEqual(list(inspect.signature(getattr(opt, name)).parameters)[:len(head)], head)
        for gone in ('Problem', 'least_squares_problem', '_minibatch_step_problem', '_STOCK_LOCAL_MODEL'):
            self.assertFalse(hasattr(opt, gone), gone)
        import t3toolbox.backend.regularization as breg
        import t3toolbox.backend.uniform_fitting as uf
        self.assertFalse(hasattr(breg, '_ScaledRegularizer'))
        self.assertFalse(hasattr(uf, 'uniform_least_squares_problem'))

    def test_want_is_a_literal_at_every_query(self):
        """Every oracle query of backend/optimizers.py names its want as a literal tuple of strings (the compile key is
        static and the call site readable; plan 7.7)."""
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(opt))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'query']
        self.assertGreaterEqual(len(calls), 10)
        for call in calls:
            with self.subTest(line=call.lineno):
                want = call.args[2] if len(call.args) > 2 else next(k.value for k in call.keywords if k.arg == 'want')
                self.assertIsInstance(want, ast.Tuple)
                self.assertTrue(want.elts and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                                                  for e in want.elts))

    # ---- held never crosses a draw, a trial, a build ---------------------------------------------------------------
    def test_mc_sgd_builds_each_draw_afresh(self):
        """MC-SGD's iterates equal a reference loop built from direct oracle queries on each drawn batch, with no held
        (the same rng calls as the flat draw): a step that carried a held (r of the previous draw, same shapes) across
        draws would give another iterate. Regularized, so the unscaled-regularizer rule is in the reference too."""
        reg = opt.IdentityRegularizer(0.1)
        fit = self._probe_fit(n=80, regularizer=reg)
        x0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        x, stats = fit.mc_sgd(x0, np.random.default_rng(9), 20, max_iter=6, check_every=3)
        rng, n = np.random.default_rng(9), fit.oracle.n_measurements(fit.batch)
        xr = x0
        for _ in range(6):
            batch = fit.oracle.take(fit.batch, rng.choice(n, size=20, replace=False))
            frame, aux, g, q, _ = self._direct_gradient(fit, xr, batch)
            alpha = float(fit.geom.inner(g, g)) / max(float(q), 1e-30)
            xr = fit.geom.retract(frame, cw.corewise_scale(g, -alpha), aux=aux)
        self.assertLess(np.linalg.norm(t3.TuckerTensorTrain(*x).to_dense() - t3.TuckerTensorTrain(*xr).to_dense())
                        / np.linalg.norm(t3.TuckerTensorTrain(*xr).to_dense()), 1e-12)
        self.assertEqual(stats['n_iter'], 6)

    def test_adam_builds_each_draw_afresh(self):
        """Adam's iterates equal a reference loop built from direct oracle queries on each drawn batch (no held)."""
        fit = self._probe_fit('corewise', n=80)
        x0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        x, _ = fit.adam(x0, np.random.default_rng(4), 16, max_iter=5, lr=1e-2, cosine=False)
        rng, n = np.random.default_rng(4), fit.oracle.n_measurements(fit.batch)
        xr, m, v = x0, cw.corewise_zeros_like(x0), cw.corewise_zeros_like(x0)
        for k in range(5):
            batch = fit.oracle.take(fit.batch, rng.choice(n, size=16, replace=False))
            _, _, g, _, _ = self._direct_gradient(fit, xr, batch)
            m = cw.corewise_map(lambda mi, gi: 0.9 * mi + 0.1 * gi, m, g)
            v = cw.corewise_map(lambda vi, gi: 0.999 * vi + 0.001 * gi * gi, v, g)
            b1, b2 = 1.0 - 0.9 ** (k + 1), 1.0 - 0.999 ** (k + 1)
            xr = cw.corewise_sub(xr, cw.corewise_map(lambda mi, vi: 1e-2 * (mi / b1) / (np.sqrt(vi / b2) + 1e-8), m, v))
        self.assertLess(float(cw.corewise_norm(cw.corewise_sub(x, xr))) / float(cw.corewise_norm(xr)), 1e-12)

    def test_minibatched_newton_builds_each_point_afresh(self):
        """Newton on drawn batches (newton_cg called per draw: nothing crosses calls) and, within a call, a model built
        afresh at each accepted point: every row's objective is the DENSE objective of that call's batch at the row's
        point, and an accepted row's objective plus its delta_f (the trial, scored by point_objective with nothing
        held) is the next row's. A build reusing the last model's held, or a trial scored with it, breaks both."""
        fit = self._probe_fit(n=120)
        rng, n = np.random.default_rng(2), fit.oracle.n_measurements(fit.batch)
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        for draw in range(3):
            batch = fit.oracle.take(fit.batch, rng.choice(n, size=40, replace=False))
            seen = []
            x, stats = opt.newton_cg(fit.geom, fit.oracle, batch, x, max_newton=3, callback=seen.append,
                                     g0_reference='warm_start')
            ww = batch.sample
            for k, info in enumerate(seen):
                with self.subTest(draw=draw, row=k):
                    T = t3.TuckerTensorTrain(*info.x_cores).to_dense()
                    dense = 0.5 * (n / 40) * sum(float(np.sum((z - y) ** 2)) for z, y in zip(dense_probe(T, ww), batch.y))
                    self.assertAlmostEqual(info.objective, dense, delta=1e-10 * dense)
                    if k + 1 < len(seen) and not info.converged and not info.ls_failed:
                        self.assertAlmostEqual(info.objective + info.delta_f, seen[k + 1].objective,
                                               delta=1e-10 * info.objective)

    def test_a_trial_is_scored_with_nothing_held(self):
        """Policy a: every trial's objective is point_objective's (its terms) at the trial's raw cores, the dense
        objective there (trust region and line search), not the model's held residual. So is the start's value, the
        acceptance's first reference (D9): the first scored point is x0 itself."""
        fit = self._probe_fit()
        x0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        trials = []
        real = opt._point_objective_parts

        def recording(geom, oracle, batch, x_cores, regularizer=None):
            f, rho = real(geom, oracle, batch, x_cores, regularizer)
            trials.append((x_cores, float(f)))
            return f, rho
        for step_control in ('trust_region', 'line_search'):
            with self.subTest(step_control=step_control):
                trials.clear()
                with unittest.mock.patch.object(opt, '_point_objective_parts', recording):
                    fit.newton_cg(x0, max_newton=3, step_control=step_control)
                self.assertGreater(len(trials), 1)
                self.assertIs(trials[0][0], x0)
                for x_trial, f in trials:
                    T = t3.TuckerTensorTrain(*x_trial).to_dense()
                    dense = 0.5 * sum(float(np.sum((z - y) ** 2)) for z, y in zip(dense_probe(T, fit.batch.sample),
                                                                                    fit.batch.y))
                    self.assertAlmostEqual(f, dense, delta=1e-10 * dense)

    def test_the_acceptance_compares_like_with_like(self):
        """D9: the step acceptance compares point_objective with point_objective. Row 0's objective is the start's
        point_objective (bitwise); every row's misfit and regularization are point_objective's own terms at that row's
        point (bitwise: the trial's value carried, never a frame-route one); misfit + regularization = objective; on an
        accepted row objective + delta_f is the next row's objective (to the rounding of delta_f = f_new - f), on a
        rejected or failed one the next row keeps it (bitwise). Trust region (a rejection scripted: row 1's trial raised),
        line search (a backtrack, scripted the same way), the 'accept' after an exhausted search, regularized, ragged
        and uniform, both geometries. gradient_descent likewise: losses[k] is point_objective at its k-th
        point."""
        import t3toolbox.backend.regularization as reg
        for layer in ('ragged', 'uniform'):
            for geometry in ('manifold', 'corewise'):
                fit = self._probe_fit(geometry, regularizer=reg.IdentityRegularizer(1e-3))
                x0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
                if layer == 'uniform':
                    fit, xs = F.uniform(fit, ut3.UniformTuckerTensorTrain.from_t3(x0, N=10, n=3, r=3))
                else:
                    xs = x0.data
                runs = {'trust region': dict(step_control='trust_region'),
                        'line search': dict(step_control='line_search'),
                        'accept': dict(step_control='line_search', c_armijo=10.0, on_line_search_failure='accept')}
                for name, kw in runs.items():
                    with self.subTest(layer=layer, geometry=geometry, run=name):
                        seen = []
                        f0 = float(fit.objective(xs))                       # the 2nd trial (row 1's first on a
                        with _objective_stub_at(2, lambda f: f + 2.0 * f0 + 1.0):   # trust region): an ascent, rejected
                            _, stats = fit.newton_cg(xs, max_newton=5, callback=seen.append, **kw)
                        h = stats['history']
                        self.assertEqual(h[0]['objective'], float(fit.objective(xs)))
                        rejected = 0
                        for row, nxt, info in zip(h, h[1:], seen):
                            moved = row['tr_accepted'] if row['step_control'] == 'trust_region' else (
                                not row['ls_failed'] or name == 'accept')
                            rejected += not moved
                            if moved:
                                self.assertAlmostEqual(nxt['objective'], row['objective'] + row['delta_f'],
                                                       delta=1e-15 * max(abs(nxt['objective']), abs(row['objective'])))
                            else:
                                self.assertEqual(nxt['objective'], row['objective'])
                        for row, info in zip(h, seen):
                            f, rho = opt._point_objective_parts(fit.geom, fit.oracle, fit.batch, info.x_cores,
                                                                fit.regularizer)
                            self.assertEqual((row['misfit'], row['regularization']), (float(f), float(rho)))
                            self.assertEqual(row['objective'], row['misfit'] + row['regularization'])
                        if name == 'trust region':
                            self.assertGreaterEqual(rejected, 1)
                        if name == 'accept':
                            self.assertTrue(any(row['ls_failed'] for row in h[:-1]))
                with self.subTest(layer=layer, geometry=geometry, run='gradient_descent'):
                    _, stats = fit.gradient_descent(xs, n_iter=4, on_line_search_failure='accept')
                    for k in range(4):
                        xk, _ = fit.gradient_descent(xs, n_iter=k, on_line_search_failure='accept')
                        self.assertEqual(stats['losses'][k], float(fit.objective(xk)))

    def test_a_query_given_the_models_held_runs_no_plain_sweep(self):
        """The review's structural test: at a frame with centers (every frame the library builds), a Gauss-Newton
        model's gradient, gn_quadratic and hvp, given its held, run no plain sweep (xi, mu, nu, eta, z, r and the
        frame's nu_L, eta_D): the build computed them and held them. Probe, apply and entries, ragged and uniform, on a
        manifold frame and a corewise point. Without centers the probe kind's hvp recomputes its right sweep (the
        documented cost of a start frame with none)."""
        import t3toolbox.backend.uniform_fitting as uf
        plain = {'xi', 'mu', 'nu', 'eta', 'z', 'r', 'nu_L', 'eta_D'}
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        ww = unit_vecs(20, SHAPE, self.rng)
        index = np.stack([self.rng.integers(0, N, 20) for N in SHAPE])
        Ad = self.A.to_dense()
        data = {'probe': (ww, dense_probe(Ad, ww)), 'apply': (ww, dense_apply(Ad, ww)), 'entries': (index, Ad[tuple(index)])}
        for layer in ('ragged', 'uniform'):
            for geometry in ('manifold', 'corewise'):
                for kind, (sample, y) in data.items():
                    with self.subTest(layer=layer, geometry=geometry, kind=kind):
                        geom = bgeo.ManifoldGeometryOps() if geometry == 'manifold' else bgeo.CorewiseGeometryOps()
                        fit, xc = F.fit(geom, kind, sample, y), x.data
                        if layer == 'uniform':
                            fit, xc = F.uniform(fit, ut3.UniformTuckerTensorTrain.from_t3(x))
                        lm = fit.local_model(xc)
                        p = lm.gradient
                        for name, call in (('gradient', lambda: lm.gradient), ('gn_quadratic', lambda: lm.gn_quadratic(p)),
                                           ('hvp', lambda: lm.hvp(p))):
                            ran = set(self._rules_run(call))
                            self.assertTrue(ran, name)
                            self.assertEqual(ran & plain, set(), name)
        frame, _ = bgeo.ManifoldGeometryOps().frame(x.data)
        lm = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, data['probe'][1]).local_model(x.data, frame=frame)
        self.assertEqual(lm.held.keys(), ('xi', 'mu', 'r'))
        self.assertEqual(set(self._rules_run(lambda: lm.hvp(lm.gradient))) & plain, {'nu', 'eta'})

    def test_the_models_f_is_point_objective_at_four_frames(self):
        """The review's test: the model's f (read through the frame, its centers when it has them) equals
        point_objective(x) (read through the point), at four kinds of frame: a corewise point; geom.frame(x) with its
        centers; a given frame of x without centers; and a padded start's aligned frame with the zero-padded old
        centers (D3). Each against the dense objective too."""
        fit = self._probe_fit()
        dense = lambda xc: 0.5 * sum(float(np.sum((z - y) ** 2)) for z, y in
                                     zip(dense_probe(t3.TuckerTensorTrain(*xc).to_dense(), fit.batch.sample), fit.batch.y))
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        x_old = t3.TuckerTensorTrain.randn(SHAPE, (2, 2, 2), (1, 2, 2, 1))
        x_pad = x_old.resize(SHAPE, TUCKER, TT).data
        frame, centers, _ = fit.padded_start(x_pad, x_old.data, max_passes=2)
        cases = [('corewise point', fit.with_(geom=bgeo.CorewiseGeometryOps()), x.data, {}),
                 ('frame(x) with centers', fit, x.data, {}),
                 ('a given frame, no centers', fit, x.data, dict(frame=bgeo.ManifoldGeometryOps().frame(x.data)[0])),
                 ('padded start, its centers', fit, x_pad, dict(frame=frame, centers=centers))]
        for name, f_, xc, kw in cases:
            with self.subTest(case=name):
                lm = f_.local_model(xc, **kw)
                self.assertEqual(lm.centers is None, name in ('corewise point', 'a given frame, no centers'))
                self.assertAlmostEqual(float(lm.f), float(f_.objective(xc)), delta=1e-11 * float(lm.f))
                self.assertAlmostEqual(float(lm.f), dense(xc), delta=1e-11 * float(lm.f))

    def test_point_data_reads_garbage_padding_as_nothing(self):
        """D1: the uniform geometries' point_data is the point's .data with the geometry's own shape and POINT masks, so
        point_objective at a bare supercore pair with garbage in its padding is the ragged objective (the dense one),
        on both uniform geometries and every kind; ragged point_data is the cores themselves."""
        import t3toolbox.backend.ut3_masking as ut3mask
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=10, n=4, r=4)                    # forced padding on every axis
        tk, tt, shape, masks = ux.data
        ones = ut3mask.ut3_apply_masks((np.ones_like(tk), np.ones_like(tt), shape, masks))
        dirty = (tk + 1e3 * (1.0 - ones[0]), tt + 1e3 * (1.0 - ones[1]))
        ww = unit_vecs(20, SHAPE, self.rng)
        index = np.stack([self.rng.integers(0, N, 20) for N in SHAPE])
        Ad, X = self.A.to_dense(), x.to_dense()
        cases = {'probe': (ww, dense_probe(Ad, ww), dense_probe(X, ww)),
                 'apply': (ww, dense_apply(Ad, ww), dense_apply(X, ww)),
                 'entries': (index, Ad[tuple(index)], X[tuple(index)])}
        for geometry, cls in (('manifold', bgeo.UniformManifoldGeometryOps), ('corewise', bgeo.UniformCorewiseGeometryOps)):
            geom = cls.from_point(ux.data)
            point = geom.point_data(dirty)
            self.assertTrue(point[0] is dirty[0] and point[1] is dirty[1] and tuple(point[2]) == tuple(shape))
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(point[3], geom.masks)))
            for kind, (sample, y, z) in cases.items():
                with self.subTest(geometry=geometry, kind=kind):
                    fit, _ = F.uniform(F.fit(bgeo.ManifoldGeometryOps(), kind, sample, y), ux)
                    fit = fit.with_(geom=geom)
                    truth = 0.5 * sum(float(np.sum((a - b) ** 2)) for a, b in zip(z, y)) if kind == 'probe' \
                        else 0.5 * float(np.sum((z - y) ** 2))
                    self.assertAlmostEqual(float(fit.objective(dirty)), truth, delta=1e-11 * truth)
        self.assertIs(bgeo.ManifoldGeometryOps().point_data(x.data), x.data)
        self.assertIs(bgeo.CorewiseGeometryOps().point_data(x.data), x.data)
        self.assertIs(bgeo.CorewiseGeometryOps().frame(x.data)[0][1][0], x.data[1][0])

    # ---- the floor -------------------------------------------------------------------------------------------------
    def test_the_floor_reads_eps_off_f(self):
        """D8: the trust region's floor is the oracle's f_resolution plus eps |rho| (the 2026.5.0 floor
        eps (|wy| |wr| + |misfit + rho|), by hand), eps read off the dtype of the model's f: float32 data give a float32
        f and eps; an oracle without f_resolution gets eps (|f| + |rho|)."""
        reg = opt.IdentityRegularizer(0.2)
        for dtype in (np.float64, np.float32):
            with self.subTest(dtype=dtype.__name__):
                ww = [w.astype(dtype) for w in unit_vecs(40, SHAPE, self.rng)]
                y = [a.astype(dtype) for a in dense_probe(self.A.to_dense(), ww)]
                fit = F.fit(bgeo.ManifoldGeometryOps(), 'probe', ww, y, regularizer=reg)
                x = tuple(tuple(c.astype(dtype) for c in cs) for cs in t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data)
                lm = fit.local_model(x)
                eps = float(np.finfo(dtype).eps)
                self.assertEqual(np.dtype(lm.f.dtype), np.dtype(dtype))
                self.assertEqual(opt._resolution_eps(lm.f), eps)
                rho = float(lm.regularization)
                r = np.concatenate([np.ravel(np.asarray(a, dtype=np.float64)) for a in lm.residual])
                wy = math.sqrt(sum(float(np.sum(np.asarray(a, dtype=np.float64) ** 2)) for a in y))
                by_hand = eps * (wy * math.sqrt(float(r @ r)) + abs(float(lm.f) + rho))
                self.assertAlmostEqual(opt._objective_resolution(lm, rho), by_hand, delta=1e-5 * by_hand)
                bare = fit.with_(oracle=_NoResolutionOracle('probe', ()))
                lm_bare = bare.local_model(x)
                self.assertAlmostEqual(opt._objective_resolution(lm_bare, rho), eps * (abs(float(lm_bare.f)) + rho),
                                       delta=1e-6 * eps * (abs(float(lm_bare.f)) + rho))
                self.assertAlmostEqual(opt._objective_resolution(lm_bare, None), eps * abs(float(lm_bare.f)),
                                       delta=1e-6 * eps * abs(float(lm_bare.f)))

    # ---- what an oracle must supply --------------------------------------------------------------------------------
    def test_optional_outputs_are_checked_where_needed(self):
        """Structural, before any work: mc_sgd and adam without a draw need an oracle with take and n_measurements; the
        exact mode needs Kp; jacobian needs Jp and residual needs r. With a draw, an oracle without take runs."""
        fit = self._probe_fit()
        x0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        bare = fit.with_(oracle=_BareOracle('probe', ()))
        for name, call in (('mc_sgd', lambda: bare.mc_sgd(x0, np.random.default_rng(0), 10, max_iter=1)),
                           ('adam', lambda: bare.adam(x0, np.random.default_rng(0), 10, max_iter=1))):
            with self.subTest(optimizer=name):
                with unittest.mock.patch.object(opt, '_prepare_jit_inputs', wraps=opt._prepare_jit_inputs) as work:
                    with self.assertRaisesRegex(ValueError, "_BareOracle has no take"):
                        call()
                work.assert_not_called()
        draw = lambda rng: fit.oracle.take(fit.batch, rng.choice(60, size=10, replace=False))
        x, _ = bare.mc_sgd(x0, np.random.default_rng(0), 10, draw=draw, max_iter=2)
        self.assertEqual(len(x[0]), 3)
        lm = bare.local_model(x0)
        for who, call, name in (('jacobian', lambda: lm.jacobian(lm.gradient), 'Jp'), ('residual', lambda: lm.residual, 'r')):
            with self.subTest(member=who):
                with self.assertRaisesRegex(ValueError, "_BareOracle does not supply %r" % name):
                    call()
        with self.assertRaisesRegex(ValueError, "does not supply 'Kp'"):
            bare.local_model(x0, hessian='exact')


@dc.dataclass(frozen=True, eq=False)
class _NoResolutionOracle(LeastSquaresOracle):
    """An oracle without f_resolution: the floor falls back to eps |f|."""
    optional_outputs = ('Kp', 'Jp', 'r', 'r_block_sumsq', 'y_block_sumsq')


@dc.dataclass(frozen=True, eq=False)
class _BareOracle(LeastSquaresOracle):
    """An oracle with the four required outputs only, and no draw (take, n_measurements hidden)."""
    optional_outputs = ()
    take = None
    n_measurements = None


class TestStackedPointIsRejected(unittest.TestCase):
    """Stacked points: the optimizers reject them, and so does any regularized objective.

    Two separate limitations, both raising NotImplementedError rather than failing obscurely.

    (1) Optimizing a stacked point is not supported at all -- the objective is an array of shape ``C``
    and every optimizer reduces it with a Python ``float()``. Before the guard this surfaced as a
    ``TypeError`` from inside the loop, and for ``adam`` only when its loss logging first fired at
    iteration 50. Building the local model on a stacked point is fine and still works.

    (2) A regularized objective would silently mis-weight a stacked point.

    The data misfit keeps the frame stack ``C`` (one value per element) while every regularizer scalar
    collapses it, so ``objective = misfit + rho`` would add the whole-stack regularization total to each
    element -- inflating the effective lambda by about ``|C|``, and unevenly. The regularizer *gradient*
    is per-element correct, which is what made the inconsistency easy to miss."""

    def setUp(self):
        np.random.seed(0)
        self.rng = np.random.default_rng(0)
        self.A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT, stack_shape=(3,))
        self.ww = unit_vecs(20, SHAPE, self.rng)
        self.data = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).apply(self.ww)   # y is W: it broadcasts over C
        self.reg = opt.IdentityRegularizer(0.5)

    def test_every_optimizer_rejects_a_stacked_point(self):
        """All four, unregularized -- the limitation is the scalar reduction, not the regularizer."""
        problem = F.fit(bgeo.ManifoldGeometryOps(), 'apply', self.ww, self.data)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT, stack_shape=(3,)).data
        rng = np.random.default_rng(0)
        for name, call in (('gradient_descent', lambda: problem.gradient_descent(x0, n_iter=2)),
                           ('newton_cg', lambda: problem.newton_cg(x0, max_newton=2)),
                           ('mc_sgd', lambda: problem.mc_sgd(x0, rng, 8, max_iter=2)),
                           ('adam', lambda: problem.adam(x0, rng, 8, max_iter=2))):
            with self.subTest(optimizer=name):
                with self.assertRaises(NotImplementedError):
                    call()

    def test_a_stacked_local_model_still_works(self):
        """The supported stacked path: build the model, drive your own loop. Its objective keeps C, and each
        element's is the element's own model's (a size-1 axis or a sum over C would not give that)."""
        element_problem = F.fit(bgeo.ManifoldGeometryOps(), 'apply', self.ww, self.data)
        problem = F.shared_over(element_problem, (3,))                   # the stacked model's y: a size-1 C axis
        lm = problem.local_model(self.A.data)
        self.assertEqual(np.shape(lm.objective), (3,))
        self.assertEqual(np.shape(problem.objective(self.A.data)), (3,))
        for k, element in enumerate(self.A.unstack()):
            with self.subTest(element=k):
                self.assertAlmostEqual(float(lm.objective[k]),
                                       float(element_problem.local_model(element.data).objective),
                                       delta=1e-12 * float(lm.objective[k]))
        g = lm.gradient
        self.assertEqual(np.shape(problem.geom.inner(g, g)), (3,))
        self.assertEqual(np.shape(lm.gn_quadratic(g)), (3,))

    def test_backend_problem_raises_on_a_stacked_point(self):
        problem = F.fit(bgeo.ManifoldGeometryOps(), 'apply', self.ww, self.data, regularizer=self.reg)
        for label, call in (('local_model', lambda: problem.local_model(self.A.data)),
                            ('objective', lambda: problem.objective(self.A.data)),
                            ('newton_cg', lambda: problem.newton_cg(self.A.data, max_newton=1))):
            with self.subTest(entry=label):
                with self.assertRaises(NotImplementedError):
                    call()

    def test_unstacked_regularized_is_unaffected(self):
        B = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        data = B.apply(self.ww)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)
        problem = F.fit(bgeo.ManifoldGeometryOps(), 'apply', self.ww, data, regularizer=self.reg)
        _x, stats = problem.newton_cg(x0.data, max_newton=4)
        self.assertLess(stats['losses'][-1], stats['losses'][0])



class TestWeightCombinations(unittest.TestCase):
    """Combinations the suite never made, chosen because they share the SHAPE of a bug that shipped.

    The stacked-plus-regularized defect survived into 2026.1.0 because no test combined `stack_shape`
    with a `regularizer` — a missing COMBINATION, not a missing file. These two are the neighbours of
    that gap: a residual weight had never been combined with a stack, nor with a regularizer."""

    def setUp(self):
        np.random.seed(0)
        self.rng = np.random.default_rng(0)

    def test_weighted_stacked_model_matches_the_per_element_models(self):
        """A weighted model on a stacked point must equal the per-element weighted models, elementwise: the weight
        is one value per mode, placed on the data, and a frame stack C must not shift where it applies. The data come
        from one other tensor (shared by the stack: size-1 C axes), so each element's objective is far from zero."""
        import t3toolbox.backend.stacking as stacking
        K = 3
        ww = unit_vecs(20, SHAPE, self.rng)
        omega = np.linspace(0.5, 1.5, len(SHAPE))                     # per-mode ω for the probe kind: (d,)
        elements = [t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT) for _ in range(K)]
        stacked = t3.TuckerTensorTrain(*stacking.stack(tuple(e.data for e in elements), axes=(0,)))
        data = dense_probe(t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).to_dense(), ww)

        geom = bgeo.ManifoldGeometryOps()
        stacked_obj = np.asarray(F.shared_over(F.fit(geom, 'probe', ww, data, weight=omega), (K,))
                                 .local_model(stacked.data).objective)
        self.assertEqual(stacked_obj.shape, (K,))
        self.assertGreater(float(np.min(stacked_obj)), 1.0, 'objective must be far from zero to compare')
        for i, element in enumerate(elements):
            with self.subTest(element=i):
                X = element.to_dense()
                truth = 0.5 * sum(float(np.sum((omega[m] * (z - y)) ** 2))
                                  for m, (z, y) in enumerate(zip(dense_probe(X, ww), data)))
                self.assertAlmostEqual(float(stacked_obj[i]) / truth, 1.0, places=10)

    def test_weighted_and_regularized_objective_splits_exactly(self):
        """A residual weight and a regularizer together: `objective == misfit + regularization` exactly,
        and the regularizer's quadratic term is λ‖Πp‖² regardless of the weight."""
        order, lam = 2, 0.25
        ww = unit_vecs(16, SHAPE, self.rng)
        pp = [self.rng.standard_normal((16, n)) for n in SHAPE]
        omega = np.outer(np.linspace(0.5, 1.5, len(SHAPE)), np.linspace(0.4, 1.2, order + 1))
        A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        data = [np.asarray(z) for z in dd.forward(A, 'probe', ww, pp, order)]
        reg = opt.IdentityRegularizer(lam)

        for name, geom in (('manifold', bgeo.ManifoldGeometryOps()), ('corewise', bgeo.CorewiseGeometryOps())):
            with self.subTest(geometry=name):
                problem = F.derivative_fit(geom, 'probe', order, ww, pp, data, regularizer=reg, weight=omega)
                lm = problem.local_model(t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data)   # far from A: f > 0
                self.assertGreater(float(lm.misfit), 1.0)
                self.assertAlmostEqual(float(lm.objective),
                                       float(lm.misfit) + float(lm.regularization), places=10)
                np.random.seed(2)
                p = cw.corewise_map(lambda z: np.random.randn(*z.shape), lm.gradient)
                projected = geom.project(lm.frame, p)
                self.assertAlmostEqual(
                    float(reg.quadratic(geom, lm.frame, p)),
                    lam * float(geom.inner(projected, projected)), places=10)


# ----------------------------------------------------------------------------------------------------------------------
# The Newton timing plan's T1-T5 (numpy)
# ----------------------------------------------------------------------------------------------------------------------
_HISTORY_FIELDS_2026_4 = (                       # _NEWTON_SCALAR_FIELDS before the timing feature (T1)
    'iteration', 'objective', 'gnorm', 'g0norm', 'converged', 'misfit', 'regularization', 'forcing_eta', 'cg_tol',
    'cg_iters', 'cg_maxiter', 'cg_resid', 'cg_converged', 'cg_truncated', 'ls_steps', 'ls_failed', 'alpha', 'slope',
    'pHp', 'delta_f', 'rho', 'step_rel', 'wall_time', 'g0_reference', 'cg_reference', 'step_control', 'stop_reason',
    'hessian', 'tr_radius', 'tr_accepted', 'cg_boundary', 'pKp', 'lift_sigma_min', 'lift_eps')
_PHASES = tuple(f.name for f in dc.fields(opt.PhaseSeconds))
_MODEL_PHASES = ('frame', 'oracle', 'companion', 'curvature', 'gradient')
_STEP_PHASES = ('cg', 'forms', 'trial_retraction', 'trial_objective')
_ONE_OFFS = ('setup', 'setup_compile', 'alignment', 'alignment_compile', 'alignment_passes', 'reference_passes')


def canonical(tree):
    """A tree with every float as its bytes and every array as (dtype, shape, bytes): equality is bitwise (NaN
    included), container types and dict key order kept."""
    if isinstance(tree, dict):
        return ('dict', tuple((k, canonical(v)) for k, v in tree.items()))
    if isinstance(tree, (tuple, list)):
        return (type(tree).__name__, tuple(canonical(t) for t in tree))
    if isinstance(tree, float):
        return ('float', np.float64(tree).tobytes())
    if hasattr(tree, 'dtype'):
        a = np.asarray(tree)
        return ('array', str(a.dtype), a.shape, a.tobytes())
    return tree


def untimed(run):
    """(x, stats) of a run, minus what timing adds or the clock decides: stats['timing'] and the history's wall_time."""
    x, stats = run
    stats = {k: v for k, v in stats.items() if k != 'timing'}
    stats['history'] = [{k: v for k, v in row.items() if k != 'wall_time'} for row in stats['history']]
    return canonical((x, stats))


def check_timing(test, stats, seen, compiled=False, switch=False):
    """T2 + T4 on one timed run: ``stats`` its stats, ``seen`` the NewtonInfo rows its callback received. The rows
    tile their windows (and the run); each phase is None exactly where it did not run; the one-offs sit on row 0;
    stats['timing'] is the rows' sums and records. ``compiled``: a compile listener ran (else every compile field is
    None). ``switch``: a hessian='switch' run (its builds are exact-mode; the tracked rows lap the tracking
    product). (The build is always the library's: a Problem whose local_model a subclass overrode, timed whole, went
    with Problem in P16; the extension point is the oracle, whose query is the build's 'oracle' lap.)"""
    T = stats['timing']
    test.assertEqual(list(T), ['total', 'setup', 'alignment', 'alignment_passes', 'reference_passes', 'last_callback',
                               'phases', 'compile', 'cg_per_iteration', 'rows'])
    test.assertEqual(len(seen), len(stats['history']))
    trust_region = bool(seen) and seen[0].step_control == 'trust_region'
    for k, info in enumerate(seen):
        record, seconds = info.timing, info.timing.seconds
        ran = {name: getattr(seconds, name) is not None for name in _PHASES}
        with test.subTest(row=k):
            test.assertIsInstance(record, opt.NewtonTiming)
            if info.wall_time is not None:
                test.assertEqual(record.window, info.wall_time)                     # the same two readings
            test.assertAlmostEqual(sum(getattr(seconds, n) for n in _PHASES if ran[n]), record.window, delta=1e-9)
            test.assertGreaterEqual(seconds.remainder, -1e-6)
            test.assertTrue(all(getattr(seconds, n) >= 0.0 for n in _PHASES if ran[n]))
            reused = k > 0 and seen[k - 1].tr_accepted is False                    # a rejected trial's model
            test.assertEqual(ran['previous_callback'], k > 0)
            test.assertEqual(ran['reference'], k == 0)
            test.assertEqual(ran['tracking'], switch and info.switch_pKp is not None)   # the switch's tracked rows only
            test.assertTrue(ran['remainder'])
            test.assertEqual(ran['frame'], not reused)
            test.assertEqual(ran['gradient'], not reused)
            for name in ('oracle', 'companion'):
                test.assertEqual(ran[name], not reused)
            test.assertEqual(ran['curvature'], not reused and (info.hessian == 'exact' or switch))
            for name in _STEP_PHASES:
                test.assertEqual(ran[name], not info.converged)
            if info.converged:
                test.assertIsNone(info.wall_time)
                test.assertIsInstance(record.window, float)
                test.assertIsNone(record.trials)
                test.assertIsNone(record.accepted_retraction)
            else:
                test.assertEqual(record.trials, 1 if trust_region else (40 if info.ls_failed else info.ls_steps + 1))
                if info.tr_accepted is False or info.stop_reason == 'line_search':   # a rejected step
                    test.assertIsNone(record.accepted_retraction)
                else:
                    test.assertGreater(record.accepted_retraction, 0.0)
                    if record.trials > 1:                                            # one trial's share of several
                        test.assertLess(record.accepted_retraction, seconds.trial_retraction)
                    else:
                        test.assertEqual(record.accepted_retraction, seconds.trial_retraction)
            if compiled:
                test.assertIsInstance(record.compile_seconds, opt.PhaseSeconds)
                for name in _PHASES:
                    test.assertEqual(getattr(record.compile_seconds, name) is not None, ran[name])
                test.assertGreaterEqual(record.compile_seconds.remainder, -1e-9)
            else:
                test.assertIsNone(record.compile_seconds)
            for name in _ONE_OFFS:
                if k > 0:
                    test.assertIsNone(getattr(record, name), name)
            if k == 0:
                test.assertGreaterEqual(record.setup, 0.0)
                test.assertEqual(record.setup_compile is not None, compiled)
                test.assertEqual(record.alignment is not None, 'start_movements' in stats and stats['start_frame'] == 'padded')
                test.assertEqual(record.alignment_compile is not None, compiled and record.alignment is not None)
                for name in _ONE_OFFS:
                    test.assertEqual(T.get(name, getattr(record, name)), getattr(record, name))
    test.assertEqual(T['rows'], [dc.asdict(info.timing) for info in seen])
    test.assertEqual(T['phases'], {n: sum(getattr(i.timing.seconds, n) or 0.0 for i in seen) for n in _PHASES})
    test.assertGreaterEqual(T['last_callback'], 0.0)
    test.assertAlmostEqual(T['setup'] + (T['alignment'] or 0.0) + sum(i.timing.window for i in seen) + T['last_callback'],
                           T['total'], delta=1e-6)
    iterations = sum(info.cg_iters or 0 for info in seen)
    if compiled:
        test.assertEqual(set(T['compile']), set(_PHASES) | {'setup', 'alignment', 'last_callback'})
        cg = sum(i.timing.seconds.cg - i.timing.compile_seconds.cg for i in seen if i.timing.seconds.cg is not None)
    else:
        test.assertIsNone(T['compile'])
        cg = sum(i.timing.seconds.cg for i in seen if i.timing.seconds.cg is not None)
    if iterations:
        test.assertAlmostEqual(T['cg_per_iteration'], cg / iterations, delta=1e-12)
    else:
        test.assertIsNone(T['cg_per_iteration'])


class TestNewtonTiming(unittest.TestCase):
    """The Newton timing plan's T1-T5 on numpy."""

    CASES = {                                       # (problem kwargs, newton_cg kwargs)
        'line search':             (dict(), dict(max_newton=5)),
        'trust region, rejection': (dict(), dict(max_newton=8, hessian='exact', tr_radius0=1e3)),
        'exact, line search':      (dict(), dict(max_newton=3, hessian='exact', step_control='line_search')),
        'exhausted, accept':       (dict(), dict(max_newton=2, c_armijo=10.0, on_line_search_failure='accept')),
        'exhausted, stop':         (dict(), dict(max_newton=3, c_armijo=10.0)),
        'converged':               (dict(), dict(max_newton=20, gtol_rel=1e-3)),
        'converged at row 0':      (dict(), dict(max_newton=3, gtol_rel=10.0)),
        'warm start, overrides':   (dict(), dict(max_newton=3, g0_reference='warm_start', g0norm_cg=2.0)),
    }

    @staticmethod
    def newton(problem, x0, **kw):
        seen = []
        out = problem.newton_cg(x0, callback=seen.append, **kw)
        return out, seen

    def test_off_is_today(self):
        """T1: no timing record on any row, no stats key, and the history fields are 2026.4.0's."""
        self.assertEqual(opt._NEWTON_SCALAR_FIELDS, _HISTORY_FIELDS_2026_4)
        problem, x0 = _pd_problem(0), _rank2_start()
        for name, (_, kw) in self.CASES.items():
            for timing in ({}, {'timing': False}):
                with self.subTest(case=name, timing=timing):
                    (_, stats), seen = self.newton(problem, x0, **kw, **timing)
                    self.assertTrue(all(info.timing is None for info in seen))
                    self.assertNotIn('timing', stats)
                    self.assertTrue(all(tuple(row) == _HISTORY_FIELDS_2026_4 for row in stats['history']))

    def test_the_records(self):
        """T2 + T4 + T5's numpy row on every case: the identities, each phase None exactly where it did not run, the
        one-offs on row 0, stats['timing'] the rows' sums and records, no compile column on numpy. The cases cover a
        rejected trust-region trial (its next row reuses the model), the exact Hessian, an exhausted line search
        (40 trials) under both policies, and a converged last row."""
        problem, x0 = _pd_problem(0), _rank2_start()
        covered = set()
        for name, (_, kw) in self.CASES.items():
            with self.subTest(case=name):
                (_, stats), seen = self.newton(problem, x0, timing=True, **kw)
                check_timing(self, stats, seen)
                covered |= {'rejection' for i in seen if i.tr_accepted is False}
                covered |= {'converged' for i in seen if i.converged}
                covered |= {'40 trials' for i in seen if i.timing.trials == 40}
                covered |= {'exact' for i in seen if i.timing.seconds.curvature is not None}
        self.assertEqual(covered, {'rejection', 'converged', '40 trials', 'exact'})

    def test_no_row(self):
        """max_newton=0: no row, and the run is setup + last_callback."""
        (_, stats), seen = self.newton(_pd_problem(0), _rank2_start(), max_newton=0, timing=True)
        self.assertEqual(seen, [])
        check_timing(self, stats, seen)
        self.assertEqual(stats['timing']['rows'], [])
        self.assertEqual(stats['timing']['phases'], dict.fromkeys(_PHASES, 0.0))

    def test_timing_changes_no_number(self):
        """T3: x and stats bitwise on vs off (stats['timing'] and wall_time aside), the stock oracle and a user
        oracle (whose query the build's 'oracle' lap times)."""
        problem, x0 = _pd_problem(0), _rank2_start()
        for name, (_, kw) in self.CASES.items():
            with self.subTest(case=name):
                self.assertEqual(untimed(problem.newton_cg(x0, timing=True, **kw)),
                                 untimed(problem.newton_cg(x0, **kw)))
        user = _offset(problem, 0.0)                                       # a user oracle, here a no-op one
        off = user.newton_cg(x0, max_newton=4)
        (on, seen) = self.newton(user, x0, max_newton=4, timing=True)
        self.assertEqual(untimed(on), untimed(off))
        self.assertEqual(untimed(off), untimed(problem.newton_cg(x0, max_newton=4)))
        check_timing(self, on[1], seen)

    def test_each_phase_is_its_own_work(self):
        """Every phase carries exactly its own work: the run's calls of the parts each phase names, and the clock's
        marks, laps and closes, are recorded in order, and each call is attributed to the lap that ends after it (or
        to the remainder, when a mark or a close comes first). Exact + trust region (a rejected trial, the zero
        reference), Gauss-Newton + line search. Every value goes through the oracle's one operation, so its query is
        traced as one part, with where each query happens: the build (``oracle``), the gradient, the reference, CG,
        the forms, the trial and the start's scored value (``point_objective``'s terms; the start's in the reference
        lap, D9) and, on a rejection, the floor's ``f_resolution`` (the remainder).
        (Two library-wide helpers are traced too, so the remainder's own work is pinned: ``‖x‖`` and ``-g`` before
        CG.)"""
        geom_cls = bgeo.ManifoldGeometryOps
        traced = {'geom.frame': (geom_cls, 'frame'), 'oracle.query': (LeastSquaresOracle, 'query'),
                  'geom.precompute': (geom_cls, 'precompute'), 'point_objective': (opt, '_point_objective_parts'),
                  'geom.curvature_precompute': (geom_cls, 'curvature_precompute'),
                  '_objective_resolution': (opt, '_objective_resolution'),
                  'geom.retract': (geom_cls, 'retract'), '_lift_diagnostics': (opt, '_lift_diagnostics'),
                  '_resolve_g0_reference': (opt, '_resolve_g0_reference'), '_cg_solve': (opt, '_cg_solve'),
                  '_initial_radius': (opt, '_initial_radius'),
                  'cw.corewise_scale': (cw, 'corewise_scale'), 'cw.corewise_norm': (cw, 'corewise_norm')}
        def attribute(events):
            phases, pending = {}, []
            for event in events + [('close',)]:
                if event[0] == 'call':
                    pending.append(event[1])
                    continue
                phase = event[1] if event[0] == 'lap' else 'remainder'
                for name in pending:
                    phases.setdefault(name, set()).add(phase)
                pending = []
            return phases

        def traced_run(problem, **kw):
            events = []

            def wrap(name, fn):
                def wrapper(*args, **kwargs):
                    events.append(('call', name))
                    return fn(*args, **kwargs)
                return wrapper

            def wrap_clock(method, fn):
                def wrapper(self, *args):
                    events.append((method,) + tuple(a for a in args if isinstance(a, str)))
                    return fn(self, *args)
                return wrapper
            with contextlib.ExitStack() as stack:
                for name, (owner, attr) in traced.items():
                    stack.enter_context(unittest.mock.patch.object(owner, attr, wrap(name, getattr(owner, attr))))
                for method in ('mark', 'lap', 'close'):
                    stack.enter_context(unittest.mock.patch.object(
                        opt._PhaseClock, method, wrap_clock(method, getattr(opt._PhaseClock, method))))
                seen = []
                problem.newton_cg(_rank2_start(), timing=True,
                              callback=lambda info: (events.append(('call', 'callback')), seen.append(info)), **kw)
            return attribute(events), seen

        problem = _pd_problem(0)
        common = {'geom.frame': {'frame'},
                  'oracle.query': {'oracle', 'gradient', 'reference', 'cg', 'forms', 'trial_objective'},
                  'geom.precompute': {'companion', 'reference'}, 'point_objective': {'reference', 'trial_objective'},
                  'geom.retract': {'trial_retraction'},
                  '_lift_diagnostics': {'gradient'}, '_resolve_g0_reference': {'reference'}, '_cg_solve': {'cg'},
                  'callback': {'previous_callback', 'remainder'},
                  'cw.corewise_norm': {'remainder'},                                  # ‖x‖
                  'cw.corewise_scale': {'remainder', 'cg', 'trial_retraction'}}       # -g; CG; α p
        with self.subTest(case='exact, trust region'):
            phases, seen = traced_run(problem, max_newton=4, hessian='exact')        # the default radius: forms
            self.assertTrue(any(i.tr_accepted is False for i in seen) and any(i.tr_accepted for i in seen))
            self.assertEqual(phases, dict(common, **{'geom.curvature_precompute': {'curvature'},
                                                     'oracle.query': {'oracle', 'reference', 'cg', 'forms',
                                                                      'trial_objective', 'remainder'},
                                                     '_objective_resolution': {'remainder'},
                                                     '_initial_radius': {'forms'},
                                                     'cw.corewise_scale': {'remainder', 'cg'}}))  # p, α = 1
        with self.subTest(case='Gauss-Newton, line search'):
            phases, _ = traced_run(problem, max_newton=3)
            self.assertEqual(phases, common)

    WAIT = 0.03                                     # seconds: every mocked wait sleeps this long

    def waited_run(self, problem, x0, **kw):
        """A timed run whose every wait (``common.tree_block_until_ready``, mocked) sleeps ``WAIT`` and is recorded with
        the lap it ends. Returns (stats, seen, stretches): ``stretches[0]`` the waits before the loop (the setup's,
        phase None; the alignment's), ``stretches[k + 1]`` row k's, each a list of (phase, tree)."""
        stretches, phase = [[]], [None]
        lap, close = opt._PhaseClock.lap, opt._PhaseClock.close

        def recording_lap(clock, name, outputs=None):
            phase[0] = name
            return lap(clock, name, outputs)

        def recording_close(clock, t):
            stretches.append([])
            phase[0] = None
            return close(clock, t)

        def sleeping_wait(tree):
            stretches[-1].append((phase[0], tree))
            time.sleep(self.WAIT)
        seen = []
        with unittest.mock.patch.object(opt, 'tree_block_until_ready', side_effect=sleeping_wait), \
                unittest.mock.patch.object(opt._PhaseClock, 'lap', recording_lap), \
                unittest.mock.patch.object(opt._PhaseClock, 'close', recording_close):
            _, stats = problem.newton_cg(x0, timing=True, callback=seen.append, **kw)
        self.assertEqual(stretches[len(seen) + 1:], [[], []])               # nothing waits after the last row
        return stats, seen, stretches

    def check_waits_inside_laps(self, seen, stretches):
        """Each wait is inside its own lap: every phase lasts at least WAIT per wait it ended on its row, and no row's
        remainder holds a wait. (A lap that reads the clock before it waits leaks its WAIT out of its phase.)"""
        pre = stretches[0]
        self.assertEqual(pre[0][0], None)                                   # the setup's wait comes first
        self.assertGreaterEqual(seen[0].timing.setup, self.WAIT)
        aligning = sum(name == 'alignment' for name, _ in pre)
        if aligning:
            self.assertGreaterEqual(seen[0].timing.alignment, aligning * self.WAIT)
            self.assertLess(seen[0].timing.setup, 2 * self.WAIT)            # the alignment's wait is not in setup
        for k, (info, waits) in enumerate(zip(seen, stretches[1:])):
            with self.subTest(row=k):
                for name, n in collections.Counter(name for name, _ in waits).items():
                    self.assertGreaterEqual(getattr(info.timing.seconds, name), n * self.WAIT, name)
                self.assertLess(info.timing.seconds.remainder, self.WAIT)

    def test_each_phase_waits_on_its_outputs(self):
        """With timing, every lap first waits on its own outputs (common.tree_block_until_ready), inside the lap: the
        start point and batch once at entry; by identity, the model's frame with its centers, the build query's f and
        held, the companion, the misfit gradient and the curvature data; CG's five outputs, the step and each trial's
        point; the alignment's frame, centers and movements; the laps ending in a float wait on nothing. Exact + trust
        region (two rows, the first a rejected trial, so the second reuses its model), Gauss-Newton + line search and a
        padded start. Without timing nothing is waited on."""
        problem, x0 = _pd_problem(0), _rank2_start()
        with unittest.mock.patch.object(opt, 'tree_block_until_ready') as sync:
            problem.newton_cg(x0, max_newton=2, hessian='exact')
        sync.assert_not_called()
        with self.subTest(case='exact, trust region'):
            _, seen, stretches = self.waited_run(problem, x0, max_newton=2, hessian='exact')
            self.check_waits_inside_laps(seen, stretches)
            self.assertIs(seen[0].tr_accepted, False)                       # so row 1 reuses row 0's model
            lm = seen[0].lm
            (setup_phase, setup), = stretches[0]
            self.assertTrue(setup[0] is x0 and setup[1] is problem.batch)
            row0, row1 = ([tree for _, tree in waits] for waits in stretches[1:3])
            self.assertEqual([name for name, _ in stretches[1]],
                             ['frame', 'oracle', 'companion', 'gradient', 'curvature', 'gradient',
                              'reference', 'forms', 'cg', 'forms', 'trial_retraction', 'trial_objective'])
            self.assertTrue(row0[0][0] is lm.frame and row0[0][1] is lm.centers)
            self.assertTrue(row0[1][0] is lm.f and row0[1][1] is lm.held)
            for got, want in zip(row0[2:5], (lm.geom_aux, lm.curvature.misfit_gradient, lm.curvature)):
                self.assertIs(got, want)
            gradient, reference, radius, cg, forms, retraction, objective = row0[5:]
            self.assertEqual((gradient, reference, radius, objective), (None, None, None, None))
            self.assertEqual((len(cg), len(forms), len(retraction), len(retraction[0])), (5, 2, 2, len(x0[0])))
            self.assertEqual([name for name, _ in stretches[2]],
                             ['previous_callback', 'cg', 'forms', 'trial_retraction', 'trial_objective'])
            callback, cg, forms, retraction, objective = row1
            self.assertEqual((callback, objective), (None, None))
            self.assertEqual((len(cg), len(forms), len(retraction)), (5, 2, 2))
        with self.subTest(case='Gauss-Newton, line search'):
            _, seen, stretches = self.waited_run(problem, x0, max_newton=2)
            self.check_waits_inside_laps(seen, stretches)
            trials = [tree for name, tree in stretches[1] if name == 'trial_retraction']
            self.assertEqual(len(trials), seen[0].timing.trials)
            self.assertTrue(all(len(t) == 2 and len(t[0]) == len(x0[0]) for t in trials))
            self.assertIs(seen[0].ls_failed, False)
            self.assertIs(trials[-1], seen[1].x_cores)                      # the accepted trial is the next point
        with self.subTest(case='a padded start'):
            x_old = _rank2_start()
            padded = t3.TuckerTensorTrain(*x_old).resize(_PD_SHAPE, (3, 3, 3), (1, 3, 3, 1)).data
            stats, seen, stretches = self.waited_run(problem, padded, max_newton=1, padded_from=x_old,
                                                     padded_max_passes=2)
            self.check_waits_inside_laps(seen, stretches)
            self.assertEqual([name for name, _ in stretches[0]], [None, 'alignment'])
            frame, centers, movements = stretches[0][1][1]
            self.assertIs(frame, seen[0].lm.frame)                          # the start frame the model is built at
            self.assertIs(centers, seen[0].lm.centers)                      # ... with the padded centers (D3)
            self.assertEqual(len(movements), len(stats['start_movements']))

    def test_the_compile_listener_counts_three_events(self):
        """T5's listener: exactly jax's three compile events are summed (any other event, e.g. the persistent cache's
        or a pjit-level one, is ignored), and extra event keywords are accepted."""
        listener = opt._CompileSeconds()
        for event in opt._CompileSeconds.EVENTS:
            listener(event, 0.25)
        listener('/jax/core/compile/backend_compile_duration', 0.5, fun_name='f')
        for other in ('/jax/compilation_cache/cache_retrieval_time_sec', '/jax/core/compile/jaxpr_trace',
                      '/jax/pjit/compile_duration', 'backend_compile_duration'):
            listener(other, 100.0)
        self.assertEqual(listener.total, 1.25)
        self.assertEqual(set(opt._CompileSeconds.EVENTS), {'/jax/core/compile/jaxpr_trace_duration',
                                                           '/jax/core/compile/jaxpr_to_mlir_module_duration',
                                                           '/jax/core/compile/backend_compile_duration'})
        self.assertIsNone(opt._CompileSeconds.register(_rank2_start()))     # a numpy run registers nothing

    @unittest.skipUnless(os.environ.get('T3TOOLBOX_SLOW_TESTS'),
                         'the remainder bound: set T3TOOLBOX_SLOW_TESTS=1 (required at release)')
    def test_the_remainder_is_small(self):
        """T2's slow row: on a problem whose rows take well above Python's per-row overhead, the unattributed
        remainder is at most a tenth of the windows."""
        problem = _pd_problem(0, W=600, order=2)
        (_, stats), seen = self.newton(problem, _rank2_start(), max_newton=6, hessian='exact', timing=True)
        check_timing(self, stats, seen)
        windows = sum(info.timing.window for info in seen)
        self.assertGreater(windows / len(seen), 0.05)
        self.assertLessEqual(stats['timing']['phases']['remainder'], 0.1 * windows)


# ----------------------------------------------------------------------------------------------------------------------
# The Gauss-Newton-to-exact switch, newton_cg(hessian='switch'): the switch plan's T1-T4, T6-T8 on numpy (T5 and T9,
# which jit, are in tests/test_dispatch.py)
# ----------------------------------------------------------------------------------------------------------------------
_SWITCH_STATS = ('hessian', 'switch_track', 'switched_at')     # the run keys a switch run appends, in order


def shared_part(run):
    """(x, stats) of a run minus what the switch adds (its run keys, its rows' switch fields) and the rows' wall_time,
    as a canonical tree: equal iff bitwise equal ("row-equal", the rows' part)."""
    x, stats = run
    stats = {k: v for k, v in stats.items() if k not in _SWITCH_STATS}
    stats['history'] = [{k: v for k, v in row.items() if k not in opt._SWITCH_FIELDS + ('wall_time',)}
                        for row in stats['history']]
    return canonical((x, stats))


def _doc_problem():
    """``newton_cg``'s docstring problem: probes along lines (the jet shape (1,)) on (5, 5, 5), a rank-3 target with 1 %
    noise, its rank-2 fit warm-started by five Gauss-Newton steps. Returns (problem, x0)."""
    np.random.seed(0)
    shape, rng = (5, 5, 5), np.random.default_rng(0)
    ww, pp = ([rng.standard_normal((40, n)) for n in shape] for _ in range(2))
    target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
    line = affine_jet(ww, pp)
    y = [z + 0.01 * rng.standard_normal(z.shape) for z in target.probe(line, (1,))]
    problem = F.fit(bgeo.ManifoldGeometryOps(), 'probe', line, y, order=(1,))
    x0, _ = problem.newton_cg(t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data, max_newton=5)
    return problem, x0


def _switch_case(layer, geometry, seed=11, regularizer=None):
    """(problem, x0): a probe-derivative fit (order 1, 5 % noise) of a rank-3 target at rank 2, from a random start.
    ``layer``: 'ragged' | 'uniform'; ``geometry``: 'manifold' | 'corewise' | 'shared' (the shared manifold, groups
    (0, 1, 1, 0)). The uniform start is reduced to minimal ranks (the shared notion when shared)."""
    import t3toolbox.backend.uniform_fitting as uf
    sharing = (0, 1, 1, 0) if geometry == 'shared' else None
    shape = (4, 5, 5, 4) if sharing else (5, 5, 5)
    d = len(shape)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    truth = t3.TuckerTensorTrain.randn(shape, (3,) * d, (1,) + (3,) * (d - 1) + (1,))
    x0 = t3.TuckerTensorTrain.randn(shape, (2,) * d, (1,) + (2,) * (d - 1) + (1,))
    if sharing:
        truth = truth.share(sharing, max_tucker_ranks=(3,) * d, max_tt_ranks=(1,) + (3,) * (d - 1) + (1,))
        x0 = x0.share(sharing, max_tucker_ranks=(2,) * d, max_tt_ranks=(1,) + (2,) * (d - 1) + (1,))
    ww, pp = ([rng.standard_normal((30, n)) for n in shape] for _ in range(2))
    data = [z + 0.05 * np.linalg.norm(z) / math.sqrt(z.size) * rng.standard_normal(z.shape)
            for z in dd.forward(truth, 'probe', ww, pp, 1)]
    geom = bgeo.CorewiseGeometryOps() if geometry == 'corewise' else bgeo.ManifoldGeometryOps()
    if sharing:
        geom = geom.with_sharing(sharing, shape)
    fit = F.derivative_fit(geom, 'probe', 1, ww, pp, data, regularizer=regularizer)
    if layer == 'ragged':
        return fit, x0.data
    return F.uniform(fit, uf.uniform_minimal(ut3.UniformTuckerTensorTrain.from_t3(x0), sharing), sharing)


def _cg_stub_at(n, change):
    """Patch ``_cg_solve`` so that its ``n``-th call (1-based) returns ``change(p, rhs)`` in place of CG's step."""
    real, calls = opt._cg_solve, []

    def stub(lm, rhs, tol, maxiter, use_jit, radius=math.inf):
        calls.append(1)
        out = real(lm, rhs, tol, maxiter, use_jit, radius)
        return ((change(out[0], rhs),) + tuple(out[1:])) if len(calls) == n else out
    return unittest.mock.patch.object(opt, '_cg_solve', stub)


_ASCENT = lambda p, rhs: cw.corewise_scale(rhs, -1.0)          # p = +g: the slope is >= 0, so the -g fallback fires
_OVERSHOOT = lambda p, rhs: cw.corewise_scale(p, 4.0)          # four Newton steps: Armijo backtracks


def _objective_stub_at(n, change):
    """Patch the scored objective (``point_objective``'s terms, what ``newton_cg`` compares) so that its ``n``-th TRIAL
    (1-based; the start's one value is not a trial) has its misfit ``change(f)`` in place of ``f``. Under the trust
    region ``newton_cg`` scores one trial per row, so the ``n``-th trial is row ``n - 1``'s. A changed value is the
    trial's on either side of the comparison: carried if the trial is accepted, dropped if it is rejected (D9)."""
    real, calls = opt._point_objective_parts, []

    def stub(*args, **kwargs):
        calls.append(1)
        f, rho = real(*args, **kwargs)
        return (change(f), rho) if len(calls) == n + 1 else (f, rho)
    return unittest.mock.patch.object(opt, '_point_objective_parts', stub)


def _untied_radius(problem, x0):
    """Half the default ``tr_radius0`` (the Gauss-Newton Cauchy length ``‖g‖³ / ⟨g, H_GN g⟩``), for a trust-region run
    whose rows a test depends on. Under Gauss-Newton the default is a TIE: CG's first iterate is the Cauchy point, whose
    length IS the default radius, so whether row 0 stops on the boundary (and so whether the radius doubles for row 1)
    is decided by rounding -- it differs between machines (BLAS kernels) and between eager and jit, and the whole
    trajectory with it. At half the default that iterate is twice the radius: a boundary stop by a factor of 2."""
    lm = problem.local_model(x0)
    g = lm.gradient
    return 0.5 * opt._initial_radius(lm, g, float(problem.geom.inner(g, g)) ** 0.5)


@dc.dataclass(frozen=True, eq=False)
class _NoKpOracle(LeastSquaresOracle):
    """An oracle that does not supply ``Kp`` (it is not in its ``optional_outputs``): it works with Gauss-Newton only."""
    optional_outputs = ('Jp', 'r', 'r_block_sumsq', 'y_block_sumsq', 'f_resolution')


class TestSwitchNewton(unittest.TestCase):
    """The switch plan's T1-T4 and T6-T8 on numpy: the threshold-disabled and forced-at-0 runs ARE the pure runs; the
    junction; what counts and what resets; where it fires; the options; the stats and row contract (and the timing
    record's tracking phase)."""

    MATRIX = [(layer, geometry) for layer in ('ragged', 'uniform') for geometry in ('manifold', 'corewise', 'shared')]

    def matrix(self):
        """(name, problem, x0) over T1/T2's matrix, plus a regularized ragged manifold fit."""
        for layer, geometry in self.MATRIX:
            yield '%s %s' % (layer, geometry), *_switch_case(layer, geometry)
        yield 'ragged manifold regularized', *_switch_case('ragged', 'manifold', regularizer=opt.IdentityRegularizer(0.3))

    def test_threshold_disabled_is_gauss_newton(self):
        """T1: ``switch_tau=0`` tracks and never switches, so the run IS the Gauss-Newton run with the same step control
        and forcing power, bit for bit (x; every stats key and row field but the switch's and wall_time): CG, the step
        and the rows see the Gauss-Newton view (today's operator, gauge and regularizer included) and the gradient is
        the exact build's one transpose, Gauss-Newton's bit for bit."""
        for name, problem, x0 in self.matrix():
            for step_control in ('trust_region', 'line_search'):
                with self.subTest(case=name, step_control=step_control):
                    kw = dict(max_newton=8, step_control=step_control, cg_forcing_power=0.7)
                    gn = problem.newton_cg(x0, **kw)
                    sw = problem.newton_cg(x0, hessian='switch', switch_tau=0.0, **kw)
                    self.assertEqual(shared_part(sw), shared_part(gn))
                    stats, rows = sw[1], sw[1]['history']
                    self.assertEqual(list(stats), list(gn[1]) + list(_SWITCH_STATS))
                    self.assertEqual([stats[k] for k in _SWITCH_STATS], ['switch', 'countable', None])
                    self.assertTrue(all(row['hessian'] == 'gauss_newton' for row in rows))
                    self.assertTrue(all(row['switch_fired'] is False and row['switch_count'] == 0 for row in rows))
                    self.assertTrue(any(row['switch_e_n'] is not None for row in rows))      # the tracking ran

    def test_one_transpose_per_point(self):
        """A Gauss-Newton-phase point costs ONE misfit transpose and ONE geometry companion, as in either pure mode: the
        gradient is read from the exact build, not recomputed by the view, and the view shares the build's companion --
        so a switch_tau=0 run makes exactly the Gauss-Newton run's transposes (the oracle queries wanting ``g`` or
        ``JtJp``: the gradient's and CG's) and companions (the shared manifold's grouped frame data), on both step
        controls. (The switch run's other queries are its tracking: ``Kp`` and the rounding gate's ``f_resolution``.)"""
        for geometry in ('manifold', 'shared'):
            problem, x0 = _switch_case('ragged', geometry)
            for step_control in ('trust_region', 'line_search'):
                with self.subTest(geometry=geometry, step_control=step_control):
                    counts = []
                    for kw in (dict(), dict(hessian='switch', switch_tau=0.0)):
                        wants = []
                        real_query = LeastSquaresOracle.query

                        def query(oracle, frame, batch, want, **more):
                            wants.append(tuple(want))
                            return real_query(oracle, frame, batch, want, **more)
                        with unittest.mock.patch.object(LeastSquaresOracle, 'query', query), \
                                unittest.mock.patch.object(bgeo.ManifoldGeometryOps, 'precompute', autospec=True,
                                                           side_effect=bgeo.ManifoldGeometryOps.precompute) as comp:
                            problem.newton_cg(x0, max_newton=6, step_control=step_control, **kw)
                        counts.append(dict(transposes=sum(('g' in w) + ('JtJp' in w) for w in wants),
                                           companion=comp.call_count))
                    self.assertEqual(counts[1], counts[0])
                    self.assertGreater(counts[0]['companion'], 0)
                    self.assertGreater(counts[0]['transposes'], 0)

    def test_defaults(self):
        """The documented defaults (switch_tau 0.25, switch_k 2, switch_cg_forcing_power 1.0, switch_track 'countable')
        are the constants, and the defaulted run is the run with them spelled out, bitwise."""
        self.assertEqual((opt._SWITCH_TAU, opt._SWITCH_K, opt._SWITCH_CG_FORCING_POWER, opt._SWITCH_TRACKS),
                         (0.25, 2, 1.0, ('countable', 'all')))
        problem, x0 = _pd_problem(0), _rank2_start()
        default = problem.newton_cg(x0, max_newton=12, hessian='switch')
        spelled = problem.newton_cg(x0, max_newton=12, hessian='switch', switch_tau=0.25, switch_k=2,
                                switch_cg_forcing_power=1.0, switch_track='countable')
        self.assertIsNotNone(default[1]['switched_at'])
        self.assertEqual(untimed(default), untimed(spelled))

    def test_forced_at_zero_is_exact(self):
        """T2: ``switch_k=0`` is exact from iteration 0, with ``switch_cg_forcing_power`` the forcing power, so the run IS
        the exact run with ``cg_forcing_power`` set to it, bit for bit; nothing is tracked."""
        for name, problem, x0 in self.matrix():
            for step_control in ('trust_region', 'line_search'):
                with self.subTest(case=name, step_control=step_control):
                    kw = dict(max_newton=8, step_control=step_control)
                    ex = problem.newton_cg(x0, hessian='exact', cg_forcing_power=0.8, **kw)
                    sw = problem.newton_cg(x0, hessian='switch', switch_k=0, switch_cg_forcing_power=0.8, **kw)
                    self.assertEqual(shared_part(sw), shared_part(ex))
                    stats, rows = sw[1], sw[1]['history']
                    self.assertEqual([stats[k] for k in _SWITCH_STATS], ['switch', None, 0])
                    for row in rows:
                        self.assertEqual((row['hessian'], row['switch_fired']), ('exact', False))
                        self.assertEqual([row[k] for k in opt._SWITCH_FIELDS[:4]], [None] * 4)

    def test_the_junction(self):
        """T3: a run that switches at row j is the Gauss-Newton trust-region run up to j (its radius carried into row j)
        and, from j on, the exact run from x_j with tr_radius0 = row j's radius, the switch run's references and the
        exact phase's forcing power, bitwise; on a problem where Δ_max never binds and no trust-region floor fires in
        either run (asserted: Δ0, hence Δ_max and the floor, differ between the two)."""
        problem, x0 = _pd_problem(0), _rank2_start()
        seen = []
        sw = problem.newton_cg(x0, max_newton=30, hessian='switch', switch_cg_forcing_power=0.9,
                           callback=seen.append)
        rows, j = sw[1]['history'], sw[1]['switched_at']
        self.assertGreaterEqual(j, 3)                                   # a Gauss-Newton phase with a reset and a count
        self.assertTrue(any(row['switch_count'] == 0 for row in rows[:j - 2]))
        drop = lambda row, *more: {k: v for k, v in row.items() if k not in opt._SWITCH_FIELDS + ('wall_time',) + more}
        gn = problem.newton_cg(x0, max_newton=j + 1, step_control='trust_region')[1]['history']
        self.assertEqual(canonical([drop(r, 'stop_reason') for r in rows[:j]]),
                         canonical([drop(r, 'stop_reason') for r in gn[:j]]))
        self.assertEqual(rows[j]['tr_radius'], gn[j]['tr_radius'])     # the radius carries across the switch
        ex = problem.newton_cg(seen[j].x_cores, max_newton=30 - j, hessian='exact', cg_forcing_power=0.9,
                           tr_radius0=rows[j]['tr_radius'], g0norm_newton=rows[j]['g0norm'],
                           g0norm_cg=rows[j]['g0norm'])
        numeric = lambda row: drop(row, 'iteration', 'g0_reference', 'cg_reference')
        self.assertEqual(canonical([numeric(r) for r in rows[j:]]), canonical([numeric(r) for r in ex[1]['history']]))
        self.assertEqual(canonical(sw[0]), canonical(ex[0]))
        self.assertTrue(all(row['hessian'] == 'exact' for row in rows[j:]))
        self.assertNotEqual(rows[0]['tr_radius'], rows[j]['tr_radius'])                     # the two Δ0 differ
        for history, radius0 in ((rows, rows[0]['tr_radius']), (ex[1]['history'], rows[j]['tr_radius'])):
            self.assertLess(max(r['tr_radius'] for r in history if r['tr_radius'] is not None), 0.5e4 * radius0)
        self.assertEqual((sw[1]['stop_reason'], ex[1]['stop_reason']), ('gradient', 'gradient'))

    @staticmethod
    def forced_hits():
        """``_switch_sample`` patched so that every tracked row is a hit (its errors as computed): the count then moves
        on every countable row and resets on every other, whatever the errors."""
        real = opt._switch_sample

        def always(delta_f, predicted, curvature, floor, tau):
            return real(delta_f, predicted, curvature, floor, tau)[:2] + (True, True)
        return unittest.mock.patch.object(opt, '_switch_sample', always)

    def reset_runs(self, **kw):
        """T4's runs, each a {name: (stats, the row its condition is on)}: a rejected trust-region trial (row 3's trial
        objective raised by f(x0), so that trial is an ascent from any accepted point: three accepted steps, then a
        rejection), an Armijo backtrack (CG's second step quadrupled), the -g fallback under each step control (CG's
        second step an ascent), a forced 'accept' (Armijo cannot be met) and a countable step that CG stopped on the
        trust-region boundary. Each event is driven, not found on a trajectory: the trust-region runs start at
        ``_untied_radius`` (half the default Δ0), so rows 0 and 1 stop on the boundary by a factor, not by rounding."""
        base, x0 = _pd_problem(0), _rank2_start()
        f0, radius = float(base.objective(x0)), _untied_radius(base, x0)
        run = lambda **more: base.newton_cg(x0, hessian='switch', **more, **kw)[1]
        with _objective_stub_at(4, lambda f: f + f0):                 # f(trial) + f(x0) - f(x) > 0 once f(x) < f(x0)
            rejected = run(max_newton=6, tr_radius0=radius)
        with _cg_stub_at(2, _OVERSHOOT):
            backtrack = run(max_newton=4, step_control='line_search')
        with _cg_stub_at(2, _ASCENT):
            fallback_tr = run(max_newton=4, tr_radius0=radius)
        with _cg_stub_at(2, _ASCENT):
            fallback_ls = run(max_newton=4, step_control='line_search')
        runs = {'rejected trial': (rejected, 3), 'backtrack': (backtrack, 1),
                'fallback, trust region': (fallback_tr, 1), 'fallback, line search': (fallback_ls, 1),
                'forced accept': (run(max_newton=2, c_armijo=10.0, step_control='line_search',
                                      on_line_search_failure='accept'), 0),
                'boundary': (run(max_newton=3, tr_radius0=radius), 1)}
        h = {name: stats['history'] for name, (stats, _) in runs.items()}
        self.assertTrue(h['rejected trial'][2]['tr_accepted'] and h['rejected trial'][3]['tr_accepted'] is False)
        self.assertTrue(h['backtrack'][1]['ls_steps'] > 0 and not h['backtrack'][1]['ls_failed'])
        self.assertTrue(h['fallback, trust region'][1]['tr_accepted'])
        self.assertTrue(h['forced accept'][0]['ls_failed'])
        self.assertTrue(h['boundary'][1]['cg_boundary'] and h['boundary'][1]['tr_accepted'])
        return runs

    def test_what_counts_and_what_resets(self):
        """T4, the counting: with every tracked row a hit, the count after a row is the count before plus one on a
        countable row (a full model step: an accepted trust-region trial, an Armijo step at α = 1, not the -g fallback)
        and 0 on every other -- a rejected trial, a backtrack, the fallback, a forced 'accept' -- while a countable step
        that CG stopped on the trust-region boundary counts. The same under switch_track='all', which tracks every row
        but counts only the countable ones."""
        for track in ('countable', 'all'):
            with self.forced_hits():
                runs = self.reset_runs(switch_k=50, switch_track=track)
            for name, (stats, at) in runs.items():
                with self.subTest(track=track, case=name):
                    rows = stats['history']
                    count = 0
                    for row in rows:
                        if row['converged']:
                            self.assertEqual(row['switch_count'], count)              # carried
                            continue
                        trust = row['step_control'] == 'trust_region'
                        countable = (row['tr_accepted'] if trust else row['ls_steps'] == 0 and not row['ls_failed'])
                        countable = countable and not (name.startswith('fallback') and row['iteration'] == at)
                        count = count + 1 if countable else 0
                        self.assertEqual(row['switch_count'], count, row['iteration'])
                        self.assertEqual(row['switch_e_n'] is not None, countable or track == 'all')
                    self.assertEqual(rows[at]['switch_count'], 2 if name == 'boundary' else 0)
                    self.assertIsNone(stats['switched_at'])
        for step_control, radius in (('trust_region', dict(tr_radius0=1e3)), ('line_search', dict())):
            with self.subTest(case='capped CG', step_control=step_control):         # at its cap: counted like any step
                with self.forced_hits():                                       # (the large radius: no boundary stop)
                    _, stats = _pd_problem(0).newton_cg(_rank2_start(), max_newton=6, hessian='switch',
                                             switch_k=50, step_control=step_control, cg_maxiter=2, g0norm_cg=1e12,
                                             **radius)
                rows = stats['history']
                for row in rows:
                    self.assertEqual(row['cg_iters'], 2)
                    self.assertFalse(row['cg_converged'] or row['cg_boundary'] or row['cg_truncated'])
                self.assertEqual([row['switch_count'] for row in rows], list(range(1, 7)))

    def test_a_miss_resets(self):
        """T4, consecutiveness: with switch_k=3 and the nine tracked rows' samples scripted hit, hit, miss, hit, hit,
        uninformative, hit, hit, hit (every row a countable trust-region step), the count is 1, 2, 0, 1, 2, 0, 1, 2, 3 and
        the switch fires on the ninth row: a countable miss and a countable uninformative sample each reset a count of 2
        to 0 (not merely lower it), and only switch_k consecutive hits switch."""
        hit, miss, uninformative = (True, True), (True, False), (False, False)
        script = iter([hit, hit, miss, hit, hit, uninformative, hit, hit, hit])
        real = opt._switch_sample

        def scripted(delta_f, predicted, curvature, floor, tau):
            return real(delta_f, predicted, curvature, floor, tau)[:2] + next(script)
        with unittest.mock.patch.object(opt, '_switch_sample', scripted):
            _, stats = _pd_problem(0).newton_cg(_rank2_start(), max_newton=11, hessian='switch', switch_k=3)
        rows = stats['history']
        self.assertTrue(all(row['tr_accepted'] and row['switch_e_n'] is not None for row in rows[:9]))
        self.assertEqual([row['switch_count'] for row in rows[:9]], [1, 2, 0, 1, 2, 0, 1, 2, 3])
        self.assertEqual([row['switch_fired'] for row in rows], [k == 8 for k in range(len(rows))])
        self.assertEqual(stats['switched_at'], 9)

    def test_reset_rows(self):
        """T4 without mocks: each reset row has count 0 and switch_e_gn set, and by default no product (switch_pKp and
        switch_e_n None). Under switch_track='all' the same rows carry the product with the count still 0, and the run
        equals the default run: x bitwise, the rows row-equal, the counts, the firing and switched_at equal, e_GN on
        every row and the product's fields on the countable rows equal."""
        default, every = self.reset_runs(), self.reset_runs(switch_track='all')
        for name, (stats, at) in default.items():
            with self.subTest(case=name):
                rows, all_rows = stats['history'], every[name][0]['history']
                if name != 'boundary':
                    row = rows[at]
                    self.assertEqual(row['switch_count'], 0)
                    self.assertIsInstance(row['switch_e_gn'], float)
                    self.assertTrue(row['switch_pKp'] is None and row['switch_e_n'] is None)
                    self.assertIsInstance(all_rows[at]['switch_pKp'], float)
                    self.assertIsInstance(all_rows[at]['switch_e_n'], float)
                    self.assertEqual(all_rows[at]['switch_count'], 0)
                self.assertEqual(shared_part((None, stats)), shared_part((None, every[name][0])))
                for row in all_rows:                                  # the two errors, from the row's own fields
                    if row['converged']:
                        continue
                    a = row['alpha']
                    pred = (row['slope'] + 0.5 * row['pHp'] if row['step_control'] == 'trust_region'
                            else a * row['slope'] + 0.5 * a * a * row['pHp'])
                    self.assertEqual(row['switch_e_gn'], abs(row['delta_f'] - pred))
                    self.assertEqual(row['switch_e_n'], abs(row['delta_f'] - (pred + 0.5 * a * a * row['switch_pKp'])))
                self.assertEqual(stats['switched_at'], every[name][0]['switched_at'])
                for a, b in zip(rows, all_rows):
                    self.assertEqual([a[k] for k in ('switch_count', 'switch_fired', 'switch_e_gn')],
                                     [b[k] for k in ('switch_count', 'switch_fired', 'switch_e_gn')])
                    if a['switch_e_n'] is not None:                                     # a countable row
                        self.assertEqual((a['switch_pKp'], a['switch_e_n']), (b['switch_pKp'], b['switch_e_n']))
        runs = []                                                       # x bitwise, on a run that also switches
        for track in ('countable', 'all'):
            with _cg_stub_at(2, _ASCENT):
                runs.append(_pd_problem(0).newton_cg(_rank2_start(), max_newton=12, hessian='switch',
                                          switch_track=track))
        self.assertIsNotNone(runs[0][1]['switched_at'])
        self.assertEqual(runs[0][1]['switched_at'], runs[1][1]['switched_at'])
        self.assertEqual(shared_part(runs[0]), shared_part(runs[1]))

    def test_switch_sample(self):
        """T4: _switch_sample over a hit, a miss (strict: e_n = τ·e_gn misses), the rounding gate (both errors at or
        below the floor: uninformative; e_n at it with e_gn above: a hit; e_gn at it with e_n above: an informative
        miss) and non-finite inputs (uninformative)."""
        S = opt._switch_sample
        # Δf = -1, m_GN = -0.75, m_N = -0.9375: e_gn = 0.25, e_n = 0.0625 (exact binary fractions)
        self.assertEqual(S(-1.0, -0.75, -0.1875, 0.0, 0.5), (0.25, 0.0625, True, True))         # a hit
        self.assertEqual(S(-1.0, -0.75, -0.1875, 0.0, 0.25), (0.25, 0.0625, True, False))       # e_n = τ·e_gn: a miss
        self.assertEqual(S(-1.0, -0.75, 0.0, 0.0, 0.5), (0.25, 0.25, True, False))              # no curvature: a miss
        self.assertEqual(S(-1.0, -0.75, -0.1875, 0.25, 0.5), (0.25, 0.0625, False, False))      # both at the floor
        self.assertEqual(S(-1.0, -0.75, -0.1875, 0.1, 0.5), (0.25, 0.0625, True, True))         # e_n at it: a hit
        self.assertEqual(S(-1.0, -0.9375, 0.1875, 0.1, 0.5), (0.0625, 0.25, True, False))       # e_gn at it: a miss
        self.assertEqual(S(0.0, 0.0, 0.0, 0.0, 0.5), (0.0, 0.0, False, False))                  # 0 is never above
        for args in ((math.nan, -0.75, -0.1875), (-1.0, math.nan, -0.1875), (-1.0, -0.75, math.inf),
                     (-math.inf, -0.75, -0.1875)):
            with self.subTest(args=args):
                self.assertEqual(S(*args, 0.0, 0.5)[2:], (False, False))

    def test_it_fires_where_it_should(self):
        """T6, the docstring problem (a noisy rank-3 target, the rank-2 fit warm-started): it fires, after switch_k
        countable rows with e_N < 0.25·e_GN, its rows Gauss-Newton before and exact from switched_at, in no more Newton
        steps (rejected trials included) than Gauss-Newton with the trust region."""
        problem, x0 = _doc_problem()
        kw = dict(max_newton=40, g0_reference='warm_start')
        _, gn = problem.newton_cg(x0, step_control='trust_region', **kw)
        _, sw = problem.newton_cg(x0, hessian='switch', **kw)
        rows, j = sw['history'], sw['switched_at']
        self.assertIsNotNone(j)
        for row in rows[j - 2:j]:
            self.assertTrue(row['tr_accepted'])
            self.assertLess(row['switch_e_n'], 0.25 * row['switch_e_gn'])
        self.assertEqual([row['switch_count'] for row in rows[j - 2:j]], [1, 2])
        self.assertEqual([row['switch_fired'] for row in rows], [k == j - 1 for k in range(len(rows))])
        self.assertEqual([row['hessian'] for row in rows], ['gauss_newton'] * j + ['exact'] * (len(rows) - j))
        self.assertLessEqual(sw['newton_steps'], gn['newton_steps'])

    def test_it_never_fires_on_a_flat_set(self):
        """T6: on the manifold at d = 1 the curvature term vanishes (the geometry has no curvature data there), so
        e_N = e_GN on every tracked row: never a hit, never a switch."""
        rng = np.random.default_rng(3)
        np.random.seed(3)
        ww, pp = [rng.standard_normal((20, 7))], [rng.standard_normal((20, 7))]
        target = t3.TuckerTensorTrain.randn((7,), (3,), (1, 1))
        data = [z + 0.05 * rng.standard_normal(z.shape) for z in dd.forward(target, 'probe', ww, pp, 1)]
        problem = F.derivative_fit(bgeo.ManifoldGeometryOps(), 'probe', 1, ww, pp, data)
        x0 = t3.TuckerTensorTrain.randn((7,), (2,), (1, 1)).data
        self.assertIsNone(problem.local_model(x0, hessian='exact').curvature.geometry_data)
        for track in ('countable', 'all'):
            with self.subTest(track=track):
                _, stats = problem.newton_cg(x0, max_newton=10, hessian='switch', switch_track=track)
                tracked = [row for row in stats['history'] if row['switch_e_n'] is not None]
                self.assertTrue(tracked)
                self.assertTrue(all(row['switch_e_n'] == row['switch_e_gn'] for row in tracked))
                self.assertIsNone(stats['switched_at'])

    def test_options(self):
        """T7 (backend): each rule raises before any work, its message checked: a switch_* option without
        hessian='switch'; each option's type and range; an option the run would not read (switch_tau or switch_track
        with switch_k=0, switch_k or switch_cg_forcing_power with switch_tau=0, so (0, 0) either way); the mode's
        name; an oracle without Kp (named for 'switch'). step_control=None resolves to the trust region,
        'ALL' to 'all'; the line search, curvature_damping (it reaches every exact build) and numpy scalars are
        accepted; a model has one Hessian, so local_model rejects 'switch'."""
        problem, x0 = _pd_problem(0), _rank2_start()
        bad = [(dict(switch_tau=0.5), "newton_cg: switch_tau set without hessian='switch'"),
               (dict(hessian='exact', switch_track='all', switch_k=1),
                "newton_cg: switch_k, switch_track set without hessian='switch'"),
               (dict(switch_cg_forcing_power=1.0), "switch_cg_forcing_power set without hessian='switch'"),
               (dict(hessian='switch', switch_tau=1.0), 'switch_tau must be a number in [0, 1); got 1.0'),
               (dict(hessian='switch', switch_tau=-0.25), 'switch_tau must be a number in [0, 1)'),
               (dict(hessian='switch', switch_tau=math.nan), 'switch_tau must be a number in [0, 1)'),
               (dict(hessian='switch', switch_tau=False), 'switch_tau must be a number in [0, 1)'),
               (dict(hessian='switch', switch_tau='0.25'), 'switch_tau must be a number in [0, 1)'),
               (dict(hessian='switch', switch_k=-1), 'switch_k must be an int >= 0 (not a bool); got -1'),
               (dict(hessian='switch', switch_k=True), 'switch_k must be an int >= 0'),
               (dict(hessian='switch', switch_k=2.0), 'switch_k must be an int >= 0'),
               (dict(hessian='switch', switch_track='every'),
                "switch_track must be None, 'countable' or 'all' (case-insensitive); got 'every'"),
               (dict(hessian='switch', switch_track=1), "switch_track must be None, 'countable' or 'all'"),
               (dict(hessian='switch', switch_k=0, switch_tau=0.5), 'switch_tau set with switch_k=0'),
               (dict(hessian='switch', switch_k=0, switch_track='all'), 'switch_track set with switch_k=0'),
               (dict(hessian='switch', switch_k=0, switch_tau=0.0), 'switch_tau set with switch_k=0'),
               (dict(hessian='switch', switch_tau=0.0, switch_k=3), 'switch_k set with switch_tau=0'),
               (dict(hessian='switch', switch_tau=0.0, switch_cg_forcing_power=1.0),
                'switch_cg_forcing_power set with switch_tau=0'),
               (dict(hessian='swich'), "newton_cg: hessian must be 'gauss_newton', 'exact' or 'switch' "
                                       "(case-insensitive); got 'swich'"),
               (dict(hessian='switch', on_line_search_failure='accept'), "on_line_search_failure='accept' has no"),
               (dict(curvature_damping=1e-3), "newton_cg: curvature_damping is only meaningful with hessian='exact' or "
                                              "'switch'"),
               (dict(hessian='switch', step_control='line_search', tr_radius0=1.0), 'tr_radius0 is only meaningful')]
        no_two_hole = problem.with_(oracle=_NoKpOracle(problem.oracle.kind, problem.oracle.order))
        cases = [(problem, kw, message) for kw, message in bad] + [
            (no_two_hole, dict(hessian='switch'), "_NoKpOracle does not supply 'Kp' (it is not in its optional_outputs), "
                                                  "which hessian='switch' (it still works with hessian='gauss_newton') "
                                                  "needs"),
            (no_two_hole, dict(hessian='exact'), "which hessian='exact' (it still works with hessian='gauss_newton')")]
        for prob, kw, message in cases:
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                with unittest.mock.patch.object(opt, '_prepare_jit_inputs', wraps=opt._prepare_jit_inputs) as work:
                    with self.assertRaisesRegex(ValueError, re.escape(message)):
                        prob.newton_cg(x0, max_newton=1, **kw)
                work.assert_not_called()
        no_two_hole.newton_cg(x0, max_newton=1)                                        # Gauss-Newton runs
        _, stats = problem.newton_cg(x0, max_newton=2, hessian='SWITCH', switch_track='ALL')
        self.assertEqual((stats['history'][0]['step_control'], stats['switch_track']), ('trust_region', 'all'))
        _, stats = problem.newton_cg(x0, max_newton=2, hessian='switch', step_control='line_search',
                                 switch_k=np.int64(3), switch_tau=np.float32(0.5))
        self.assertEqual(stats['history'][0]['step_control'], 'line_search')
        with unittest.mock.patch.object(bgeo.ManifoldGeometryOps, 'curvature_precompute', autospec=True,
                                        side_effect=bgeo.ManifoldGeometryOps.curvature_precompute) as build:
            problem.newton_cg(x0, max_newton=9, hessian='switch', curvature_damping=1e-3)
        self.assertGreater(build.call_count, 7)                                             # both phases' builds
        self.assertTrue(all(call.args[4] == 1e-3 for call in build.call_args_list))
        with self.assertRaisesRegex(ValueError, "local_model: hessian must be 'gauss_newton' or 'exact'"):
            problem.local_model(x0, hessian='switch')
        with self.assertRaisesRegex(ValueError, "_NoKpOracle does not supply 'Kp'"):
            no_two_hole.local_model(x0, hessian='exact')                                      # the model checks too

    def test_stats_and_rows(self):
        """T8: switch mode appends the three stats keys; every row has the five switch fields after the others;
        switch_fired is True on exactly the firing row, and switched_at is the first exact row (len(history) when it
        fired on the last row); Gauss-Newton-phase rows carry the Gauss-Newton view as lm (no pKp, no lift
        diagnostics), exact rows the exact model; the converged row's fields (the carried count in the Gauss-Newton
        phase); a rejected Gauss-Newton-phase trial keeps x and hands the same view to the next row."""
        problem, x0 = _pd_problem(0), _rank2_start()
        seen = []
        _, stats = problem.newton_cg(x0, max_newton=30, hessian='switch', callback=seen.append)
        rows, j = stats['history'], stats['switched_at']
        self.assertEqual(list(stats), ['losses', 'newton_steps', 'newton_accepted', 'stop_reason', 'history']
                         + list(_SWITCH_STATS))
        self.assertTrue(all(tuple(row) == opt._NEWTON_SCALAR_FIELDS + opt._SWITCH_FIELDS for row in rows))
        self.assertEqual([k for k, row in enumerate(rows) if row['switch_fired']], [j - 1])
        for k, (row, info) in enumerate(zip(rows, seen)):
            with self.subTest(row=k):
                gn_phase = k < j
                self.assertEqual(row['hessian'], 'gauss_newton' if gn_phase else 'exact')
                self.assertEqual(info.lm.is_exact, not gn_phase)
                if gn_phase:
                    self.assertIsNone(row['pKp']); self.assertIsNone(row['lift_sigma_min'])
                    self.assertIsInstance(row['switch_count'], int); self.assertIsInstance(row['switch_e_gn'], float)
                elif not row['converged']:
                    self.assertIsInstance(row['pKp'], float); self.assertGreater(row['lift_eps'], 0.0)
                if not gn_phase:
                    self.assertEqual([row[f] for f in opt._SWITCH_FIELDS], [None] * 4 + [False])
        self.assertTrue(rows[-1]['converged'])
        _, last = problem.newton_cg(x0, max_newton=j, hessian='switch')                 # fires on its last row
        self.assertEqual(last['switched_at'], len(last['history']))
        self.assertIs(last['history'][-1]['switch_fired'], True)
        _, late = problem.newton_cg(x0, max_newton=60, hessian='switch', switch_k=50, gtol_rel=1e-3)
        self.assertIsNone(late['switched_at'])
        self.assertTrue(late['history'][-1]['converged'])
        self.assertGreater(late['history'][-1]['switch_count'], 0)                           # the count carried
        self.assertEqual(late['history'][-1]['switch_count'], late['history'][-2]['switch_count'])
        self.assertEqual([late['history'][-1][f] for f in ('switch_pKp', 'switch_e_gn', 'switch_e_n', 'switch_fired')],
                         [None, None, None, False])
        base = problem
        seen = []
        with _objective_stub_at(3, lambda f: f + float(base.objective(x0))):  # row 2's trial: an ascent, rejected
            base.newton_cg(x0, max_newton=6, hessian='switch', callback=seen.append)
        rejected = [k for k in range(len(seen) - 1) if seen[k].tr_accepted is False]
        self.assertTrue(rejected)
        for k in rejected:
            self.assertIs(seen[k + 1].x_cores, seen[k].x_cores)
            self.assertIs(seen[k + 1].lm, seen[k].lm)
            self.assertFalse(seen[k].lm.is_exact)

    def test_timing_records_the_tracking(self):
        """With timing=True a switch run's rows tile their windows (check_timing), the Gauss-Newton phase's builds are
        exact-mode (curvature lapped), the tracking product is lapped in tracking on exactly the tracked rows, under
        either switch_track; and timing changes no number."""
        problem, x0 = _pd_problem(0), _rank2_start()
        for track in ('countable', 'all'):
            with self.subTest(track=track):
                seen = []
                timed = problem.newton_cg(x0, max_newton=10, hessian='switch', switch_track=track, timing=True,
                                      callback=seen.append)
                check_timing(self, timed[1], seen, switch=True)
                self.assertTrue(any(info.hessian == 'exact' for info in seen))
                self.assertTrue(any(info.timing.seconds.tracking is not None for info in seen))
                self.assertGreater(timed[1]['timing']['phases']['tracking'], 0.0)
                self.assertEqual(untimed(timed), untimed(problem.newton_cg(x0, max_newton=10, hessian='switch',
                                                                       switch_track=track)))


if __name__ == "__main__":
    unittest.main()
