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
        start (the orthonormal frame completion makes the zero tensor a valid start). Eager + use_jit."""
        rng = np.random.default_rng(4)
        ww = unit_vecs(300, SHAPE, rng); data = dense_probe(self.A, ww)
        problem = opt.least_squares_problem(opt.MANIFOLD_OPS, bfit.PROBE, ww, data)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        A_norm = float(np.linalg.norm(self.A))
        for use_jit in (False, True):                       # eager and (silent-fallback / jax) jit paths agree
            with self.subTest(use_jit=use_jit):
                x, stats = opt.newton_cg(problem, x0, max_newton=30, use_jit=use_jit)
                true_e = float(np.linalg.norm(t3.TuckerTensorTrain(*x).to_dense() - self.A)) / A_norm
                self.assertLess(true_e, 1e-4)

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


if __name__ == "__main__":
    unittest.main()
