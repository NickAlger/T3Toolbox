"""Tests for the backend-first optimizers (docs/optimizers_plan.md, G3).

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
        self.rng = np.random.default_rng(0)
        self.A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).to_dense()
        self.X = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)

    # (geometry name, backend GeometryOps, frontend geometry singleton)
    GEOMS = [('corewise', opt.COREWISE, t3m.COREWISE), ('manifold', opt.MANIFOLD, t3m.MANIFOLD)]
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

                    pt = geom_f.randn(geom_f.base(self.X)); p = pt.variations.data
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
            pm = opt.least_squares_problem(opt.MANIFOLD, bfit.PROBE, ww, data)
            xm, _ = opt.mc_sgd(pm, x0, np.random.default_rng(3), batch=40, max_iter=500)
            self.assertLess(true_err(xm), 0.3 * e0)
        with self.subTest(optimizer='adam', geometry='corewise'):
            pc = opt.least_squares_problem(opt.COREWISE, bfit.PROBE, ww, data)
            xc, _ = opt.adam(pc, x0, np.random.default_rng(3), batch=40, lr=2e-2, max_iter=600)
            self.assertLess(true_err(xc), 0.3 * e0)

    def test_newton_cg_recovers_to_high_accuracy(self):
        """Manifold Newton-CG (2nd-order) recovers an exact low-rank target to high accuracy from a zero
        start (the orthonormal frame completion makes the zero tensor a valid start). Eager + use_jit."""
        rng = np.random.default_rng(4)
        ww = unit_vecs(300, SHAPE, rng); data = dense_probe(self.A, ww)
        problem = opt.least_squares_problem(opt.MANIFOLD, bfit.PROBE, ww, data)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        A_norm = float(np.linalg.norm(self.A))
        for use_jit in (False, True):                       # eager and (silent-fallback / jax) jit paths agree
            with self.subTest(use_jit=use_jit):
                x, stats = opt.newton_cg(problem, x0, max_newton=30, use_jit=use_jit)
                true_e = float(np.linalg.norm(t3.TuckerTensorTrain(*x).to_dense() - self.A)) / A_norm
                self.assertLess(true_e, 1e-4)


if __name__ == "__main__":
    unittest.main()
