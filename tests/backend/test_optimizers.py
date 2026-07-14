"""Tests for the backend-first optimizers (dev/archive/optimizers_plan.md, G3).

Correctness gold standard: the check-free backend oracle (`backend.optimizers`) must reproduce the
frontend `fitting.GaussNewtonModel` exactly (it is the same math through the same backend functions).
Plus: the optimizers descend. numpy-only (jit dispatch is covered separately in test_dispatch)."""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.fitting as fitting
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.optimizers as opt
import t3toolbox.backend.sampling_derivatives as pd
import t3toolbox.corewise as cw
from t3toolbox.backend.common import is_jax_ndarray

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


class TestBackendOptimizers(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed for determinism
        self.rng = np.random.default_rng(0)
        self.A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).to_dense()
        self.X = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)

    # (geometry name, backend GeometryOps, frontend geometry singleton)
    GEOMS = [('corewise', opt.COREWISE_OPS, t3m.COREWISE), ('manifold', opt.MANIFOLD_OPS, t3m.MANIFOLD)]
    _FMODEL = {'apply': fitting.apply_model, 'entries': fitting.entries_model, 'probe': fitting.probe_model}
    _BKIND = {'apply': bfit.APPLY, 'entries': bfit.ENTRIES, 'probe': bfit.PROBE}

    def _problem_and_frontend(self, geom_b, geom_f, kind_name, M=60):
        """Build a `Problem` (backend geometry) and the matching frontend `GaussNewtonModel` (at self.X)."""
        rng, A, X = self.rng, self.A, self.X
        if kind_name == 'apply':
            sample = unit_vecs(M, SHAPE, rng); data = dense_apply(A, sample); r = np.asarray(X.apply(sample)) - data
        elif kind_name == 'probe':
            sample = unit_vecs(M, SHAPE, rng); data = dense_probe(A, sample)
            r = [np.asarray(p) - d for p, d in zip(X.probe(sample), data)]
        else:  # entries
            flat = rng.choice(int(np.prod(SHAPE)), size=M, replace=False)
            sample = np.array(np.unravel_index(flat, SHAPE)); data = A[tuple(sample)]; r = np.asarray(X.entries(sample)) - data
        problem = opt.least_squares_problem(geom_b, self._BKIND[kind_name], sample, data)
        fmodel = self._FMODEL[kind_name](geom_f, X, sample, r)
        return problem, fmodel

    def test_oracle_matches_frontend(self):
        """The backend LocalModel reproduces GaussNewtonModel's gradient / objective / gn_quadratic / hvp,
        for every (geometry, sampling-kind) pair -- it is the same math through the same backend functions."""
        for gname, geom_b, geom_f in self.GEOMS:
            for kind in ('apply', 'entries', 'probe'):
                with self.subTest(geometry=gname, kind=kind):
                    problem, fmodel = self._problem_and_frontend(geom_b, geom_f, kind)
                    lm = problem.local_model(self.X.data)

                    def relerr_tree(a, b):
                        return float(cw.corewise_norm(cw.corewise_sub(a, b)) / cw.corewise_norm(b))

                    self.assertLess(relerr_tree(lm.gradient, fmodel.gradient.variations.data), 1e-11)
                    self.assertLess(abs(float(lm.objective) - float(fmodel.objective_value))
                                    / abs(float(fmodel.objective_value)), 1e-11)

                    pt = geom_f.randn(geom_f.frame(self.X)); p = pt.variations.data
                    self.assertLess(abs(float(lm.gn_quadratic(p)) - float(fmodel.gn_quadratic(pt)))
                                    / abs(float(fmodel.gn_quadratic(pt))), 1e-11)
                    self.assertLess(relerr_tree(lm.hvp(p), fmodel.gn_hessian(pt).variations.data), 1e-11)

    def test_gradient_descent_descends(self):
        """Cauchy + Armijo gradient_descent decreases the loss monotonically on BOTH geometries (the
        Armijo line search is what keeps it robust on the additive corewise chart)."""
        def rms(arrs):
            ss = sum(float(np.sum(np.asarray(a) ** 2)) for a in arrs)
            return float(np.sqrt(ss / sum(np.asarray(a).size for a in arrs)))
        for gname, geom_b, geom_f in self.GEOMS:
            with self.subTest(geometry=gname):
                problem, _ = self._problem_and_frontend(geom_b, geom_f, 'probe')
                # rescale the start so the initial probes match the data magnitude (well-conditioned start)
                sc = (rms(problem.data) / rms(self.X.probe(problem.sample))) ** (1.0 / (len(TUCKER) + len(TT) - 1))
                x0 = (tuple(sc * C for C in self.X.data[0]), tuple(sc * C for C in self.X.data[1]))
                _, stats = opt.gradient_descent(problem, x0, n_iter=60)
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
            pm = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data)
            xm, _ = opt.mc_sgd(pm, x0, np.random.default_rng(3), batch=40, max_iter=500)
            self.assertLess(true_err(xm), 0.3 * e0)
        with self.subTest(optimizer='adam', geometry='corewise'):
            pc = opt.least_squares_problem(opt.COREWISE_OPS, bfit.PROBE, ww, data)
            xc, _ = opt.adam(pc, x0, np.random.default_rng(3), batch=40, lr=2e-2, max_iter=600)
            self.assertLess(true_err(xc), 0.3 * e0)

    def test_derivative_kinds(self):
        """The derivative SamplingKinds (apply/entries/probe, per-order weight ω) feed the SAME generic
        Problem/optimizers. Check: (1) the corewise gradient matches a finite difference of the ω-weighted
        objective; (2) the default flat draw flattens a multi-axis W and builds a minibatch local model;
        (3) mc_sgd recovers from a zero start."""
        rng = np.random.default_rng(0)
        order, NP, NX = 2, 4, 3
        omega = np.array([1.0, 0.5, 0.3])
        Xtrue = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        X0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        ww = [rng.standard_normal((NP, NX, N)) for N in SHAPE]
        pp = [rng.standard_normal((NP, NX, N)) for N in SHAPE]
        index = np.stack([rng.integers(0, N, size=(NP, NX)) for N in SHAPE], axis=0)
        cases = {
            'apply':   (bfit.apply_derivatives_kind(order, omega), (ww, pp),
                        np.asarray(pd.t3_apply_derivatives(ww, pp, Xtrue.data, order))),
            'entries': (bfit.entries_derivatives_kind(order, omega), (index, pp),
                        np.asarray(pd.t3_entries_derivatives(index, pp, Xtrue.data, order))),
            'probe':   (bfit.probe_derivatives_kind(order, omega), (ww, pp),
                        [np.asarray(z) for z in pd.t3_probe_derivatives(ww, pp, Xtrue.data, order)]),
        }
        for name, (kind, sample, data) in cases.items():
            with self.subTest(kind=name):
                prob = opt.least_squares_problem(opt.COREWISE_OPS, kind, sample, data)
                gU, gG = prob.local_model(X0.data).gradient
                dU = [rng.standard_normal(u.shape) for u in X0.tucker_cores]
                dG = [rng.standard_normal(g.shape) for g in X0.tt_cores]
                inner = (sum(float(np.sum(np.asarray(gU[i]) * dU[i])) for i in range(len(dU)))
                         + sum(float(np.sum(np.asarray(gG[i]) * dG[i])) for i in range(len(dG))))
                eps = 1e-6
                plus = (tuple(u + eps * du for u, du in zip(X0.tucker_cores, dU)),
                        tuple(g + eps * dg for g, dg in zip(X0.tt_cores, dG)))
                minus = (tuple(u - eps * du for u, du in zip(X0.tucker_cores, dU)),
                         tuple(g - eps * dg for g, dg in zip(X0.tt_cores, dG)))
                fd = (float(prob.objective(plus)) - float(prob.objective(minus))) / (2 * eps)
                self.assertLess(abs(inner - fd) / max(abs(fd), 1e-30), 1e-5)
                # default flat draw: a multi-axis W=(NP,NX) flattens to NP*NX; minibatch local model builds
                self.assertEqual(kind.n_measurements(sample), NP * NX)
                sB, dB = opt.flat_draw(prob, batch=5)(rng)
                self.assertTrue(np.isfinite(float(prob.local_model(X0.data, sB, dB).objective)))
        # recovery: mc_sgd (manifold) from a zero start, with the apply-derivatives kind + flat default
        prob_m = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.apply_derivatives_kind(order, omega), (ww, pp),
                                           np.asarray(pd.t3_apply_derivatives(ww, pp, Xtrue.data, order)))
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        _, stats = opt.mc_sgd(prob_m, x0, np.random.default_rng(1), batch=6, max_iter=400)
        self.assertLess(stats['losses'][-1], 0.1 * stats['losses'][0])

    def test_newton_cg_recovers_to_high_accuracy(self):
        """Manifold Newton-CG (2nd-order) recovers an exact low-rank target to high accuracy from a zero
        start (the orthonormal frame completion makes the zero tensor a valid start). Eager numpy (float64)
        and use_jit=True (which auto-converts the numpy inputs to jax -> float32 -> jits) both recover; the
        jit path returns a jax-backed result."""
        rng = np.random.default_rng(4)
        ww = unit_vecs(300, SHAPE, rng); data = dense_probe(self.A, ww)
        problem = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        A_norm = float(np.linalg.norm(self.A))
        for use_jit in (False, True):
            with self.subTest(use_jit=use_jit):
                x, stats = opt.newton_cg(problem, x0, max_newton=30, use_jit=use_jit)
                true_e = float(np.linalg.norm(np.asarray(t3.TuckerTensorTrain(*x).to_dense()) - self.A)) / A_norm
                self.assertLess(true_e, 1e-4)
                self.assertEqual(is_jax_ndarray(x[0][0]), use_jit)   # use_jit auto-converts numpy -> jax-backed

    def test_cg_solve_reports_state(self):
        """D1: `_cg_solve` returns `(p, iters, resid², ok)` -- converges to the exact solution on a PD
        operator (`ok` True, `resid ≤ tol`), and truncates on a nonpositive-curvature direction (`ok`
        False). A toy diagonal operator wrapped as a `(tucker, tt)` tree exercises the branch deterministically."""
        rhs = ([np.array([1.0, 1.0, 1.0])], [])                     # a (tucker=[vec], tt=[]) tangent tree
        make_hvp = lambda D: (lambda t: ([D * t[0][0]], []))        # diagonal H
        inner = cw.corewise_dot
        # PD: CG converges to D^-1 rhs, ok stays True, residual under tol
        p, i, rs, ok = opt._cg_solve(make_hvp(np.array([2.0, 3.0, 5.0])), rhs,
                                     tol=1e-10, maxiter=50, use_jit=False, inner=inner)
        self.assertTrue(bool(ok))
        self.assertLessEqual(float(rs) ** 0.5, 1e-10)
        self.assertTrue(np.allclose(p[0][0], 1.0 / np.array([2.0, 3.0, 5.0])))
        # indefinite (negative-definite here): dᵀHd < 0 on the first direction -> immediate truncation
        p2, i2, rs2, ok2 = opt._cg_solve(make_hvp(np.array([-1.0, -2.0, -3.0])), rhs,
                                         tol=1e-10, maxiter=50, use_jit=False, inner=inner)
        self.assertFalse(bool(ok2))                                 # truncated on nonpositive curvature
        self.assertGreater(float(rs2) ** 0.5, 1e-10)               # did NOT reach the tolerance

    def test_newton_cg_diagnostics(self):
        """D1: newton_cg returns a per-iteration `history`, fires `callback(NewtonInfo)` each iteration
        (carrying the LocalModel + point for per-block errors), and reports CG / line-search / ρ. The
        manifold Hessian is PD, so CG converges (not truncated); the final line is the converged one."""
        rng = np.random.default_rng(4)
        ww = unit_vecs(200, SHAPE, rng); data = dense_probe(self.A, ww)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        pm = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data)
        seen = []
        x, stats = opt.newton_cg(pm, x0, max_newton=15, callback=seen.append)

        self.assertIn('losses', stats); self.assertIn('newton', stats)         # backward compat
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
        pm = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data)

        def run(**kw):
            seen = []; kw.setdefault('max_newton', 8)
            opt.newton_cg(pm, x0, callback=seen.append, **kw)
            return seen

        # --- baseline: the reference is the computed initial gradient norm (== the first iterate's ‖g‖) ---
        base = run()
        g0 = base[0].gnorm
        self.assertGreater(g0, 0.0)
        for info in base:
            self.assertAlmostEqual(info.g0norm, g0, places=12)              # reported reference == computed norm
        BIG = 1.0e3 * g0                                                    # a much larger (warm-start-scale) reference

        # --- g0norm_newton only: BOTH tests use BIG (CG inherits the Newton reference) ---
        n_only = run(g0norm_newton=BIG)
        for info in n_only:
            self.assertAlmostEqual(info.g0norm, BIG, places=6)             # Newton reference is BIG
        s = n_only[0]
        self.assertFalse(s.converged)
        self.assertLess(s.forcing_eta, 0.5)                                # uncapped (gnorm/BIG tiny)
        self.assertAlmostEqual(s.forcing_eta, (s.gnorm / BIG) ** 0.5, places=10)   # CG inherited BIG

        # --- g0norm_cg only: CG uses BIG, the Newton reference is UNCHANGED ---
        c_only = run(g0norm_cg=BIG)
        for info in c_only:
            self.assertAlmostEqual(info.g0norm, g0, places=12)             # Newton reference still the computed norm
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
                prob  = opt.least_squares_problem(geom, bfit.PROBE, ww, data, regularizer=reg)
                prob0 = opt.least_squares_problem(geom, bfit.PROBE, ww, data)
                lm, lm0 = prob.local_model(X0.data), prob0.local_model(X0.data)
                frame = lm.frame
                base = (frame[0], frame[2])                                  # the frame's tensor X = (U, P)
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
        """MANIFOLD_OPS.point_tangent(frame) = the attachment point as a gauged tangent v_X: dense(v_X) = X,
        and ‖X‖_HS = ‖P_last‖ = point_norm_sq**½, across structures (design §4)."""
        from t3toolbox.backend import tv_operations as tvo
        for shp, tk, tt in [((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)), ((4, 4, 4, 4), (2, 2, 2, 2), (1, 2, 3, 2, 1))]:
            with self.subTest(shape=shp):
                X = t3.TuckerTensorTrain.randn(shp, tk, tt); Xd = X.to_dense()
                frame = opt.MANIFOLD_OPS.frame(X.data)
                vX = opt.MANIFOLD_OPS.point_tangent(frame)
                self.assertTrue(np.allclose(tvo.tv_to_dense(frame, vX, include_shift=False), Xd))   # dense(v_X)=X
                self.assertAlmostEqual(float(opt.MANIFOLD_OPS.point_norm_sq((frame[0], frame[2]))) ** 0.5,
                                       float(np.linalg.norm(Xd)), places=5)                          # ‖X‖=‖P_last‖

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
            prob = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data, regularizer=reg)
            x, _ = opt.newton_cg(prob, x0, max_newton=30)
            res[lam] = (err(x), nrm(x))
        self.assertLess(res[0.0][0], 1e-6)                              # λ=0 recovers exactly
        self.assertGreater(res[0.0][1], res[1e-3][1])                   # ridge shrinks ‖x‖ ...
        self.assertGreater(res[1e-3][1], res[1e-1][1])                  # ... monotonically toward 0

    def test_stochastic_regularizer_scaling(self):
        """The stochastic optimizers scale the (deterministic) regularizer by batch/n per step (so λ matches
        the full-batch optimizers) while the full-batch stop keeps the full-strength reg; mc_sgd + reg
        shrinks ‖x‖."""
        rng = np.random.default_rng(4)
        n = 200
        ww = unit_vecs(n, SHAPE, rng); data = dense_probe(self.A, ww)
        reg = opt.IdentityRegularizer(0.5)
        prob = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data, regularizer=reg)
        frame = opt.MANIFOLD_OPS.frame(t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data)
        tnorm = lambda t: float(cw.corewise_dot(t, t)) ** 0.5
        g_full = tnorm(reg.gradient(opt.MANIFOLD_OPS, frame))
        for b in (40, n):                                              # step-problem reg-grad == (b/n)·full
            sp = opt._minibatch_step_problem(prob, b)
            self.assertAlmostEqual(tnorm(sp.regularizer.gradient(opt.MANIFOLD_OPS, frame)),
                                   (min(b, n) / n) * g_full, places=6)
        self.assertIsNot(opt._minibatch_step_problem(prob, 40), prob)      # a distinct (scaled) problem
        p_unreg = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data)
        self.assertIs(opt._minibatch_step_problem(p_unreg, 40), p_unreg)   # unregularized -> no-op (unchanged)

        # mc_sgd (manifold) with reg shrinks ‖x‖ toward 0 vs the unregularized fit
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        nrm = lambda c: float(np.linalg.norm(t3.TuckerTensorTrain(*c).to_dense()))
        xu, _ = opt.mc_sgd(opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data),
                           x0, np.random.default_rng(7), batch=50, max_iter=500)
        xr, _ = opt.mc_sgd(opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data,
                                                     regularizer=opt.IdentityRegularizer(0.3)),
                           x0, np.random.default_rng(7), batch=50, max_iter=500)
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
        pm = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww_j, data_j)
        pc = opt.least_squares_problem(opt.COREWISE_OPS, bfit.PROBE, ww_j, data_j)

        with self.subTest(optimizer='newton_cg'):
            x0z = jax.tree_util.tree_map(jnp.asarray, t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data)
            xn, _ = opt.newton_cg(pm, x0z, max_newton=20, use_jit=True)
            self.assertIsInstance(xn[0][0], jnp.ndarray)
            self.assertLess(true_err(xn), 1e-3)
        with self.subTest(optimizer='newton_cg', regularized=True):     # the reg Hessian λ·Π runs inside the jit CG
            pm_reg = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww_j, data_j,
                                               regularizer=opt.IdentityRegularizer(1e-2))
            xr, _ = opt.newton_cg(pm_reg, x0z, max_newton=15, use_jit=True)
            self.assertIsInstance(xr[0][0], jnp.ndarray)
        with self.subTest(optimizer='mc_sgd'):
            xm, _ = opt.mc_sgd(pm, x0_j, np.random.default_rng(7), batch=40, max_iter=300, use_jit=True)
            self.assertIsInstance(xm[0][0], jnp.ndarray)
            self.assertLess(true_err(xm), 0.4 * e0)
        with self.subTest(optimizer='adam'):
            xa, _ = opt.adam(pc, x0_j, np.random.default_rng(7), batch=40, lr=2e-2, max_iter=400, use_jit=True)
            self.assertIsInstance(xa[0][0], jnp.ndarray)
            self.assertLess(true_err(xa), 0.4 * e0)
        with self.subTest(optimizer='mc_sgd', kind='apply_derivatives'):
            # the derivative kind (leading order axis, paired (ww,pp) sample, ω weight) composes with jit
            order, NW = 2, 80
            wwa = [jnp.asarray(np.asarray(w)[:NW]) for w in ww]            # (NW, Ni) -- a subset of the W stack
            ppa = [jnp.asarray(rng.standard_normal((NW, N))) for N in SHAPE]
            da = jnp.asarray(pd.t3_apply_derivatives(wwa, ppa, t3.TuckerTensorTrain(*x0z).data, order)) * 0.0 + 1.0
            prob_d = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.apply_derivatives_kind(order, [1.0, .5, .3]),
                                               (wwa, ppa), da)
            xd, _ = opt.mc_sgd(prob_d, x0z, np.random.default_rng(7), batch=20, max_iter=120, use_jit=True)
            self.assertIsInstance(xd[0][0], jnp.ndarray)        # jit-compiled (a stray np.* on a tracer raises)

    def test_use_jit_requires_jax(self):
        """use_jit=True with jax unavailable raises (not the old silent eager fallback) for all three
        optimizers. No compilation: the guard fires at entry before any kernel is built (jax_available
        patched off to simulate a jax-less install)."""
        rng = np.random.default_rng(4)
        ww = unit_vecs(40, SHAPE, rng); data = dense_probe(self.A, ww)
        pm = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data)
        pc = opt.least_squares_problem(opt.COREWISE_OPS, bfit.PROBE, ww, data)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        saved = opt.jax_available
        opt.jax_available = False
        try:
            with self.assertRaises(ValueError):
                opt.newton_cg(pm, x0, max_newton=1, use_jit=True)
            with self.assertRaises(ValueError):
                opt.mc_sgd(pm, x0, rng, batch=20, max_iter=1, use_jit=True)
            with self.assertRaises(ValueError):
                opt.adam(pc, x0, rng, batch=20, max_iter=1, use_jit=True)
            # use_jit=False is unaffected -- still runs eager on numpy
            xf, _ = opt.newton_cg(pm, x0, max_newton=1, use_jit=False)
            self.assertFalse(is_jax_ndarray(xf[0][0]))
        finally:
            opt.jax_available = saved


if __name__ == "__main__":
    unittest.main()
