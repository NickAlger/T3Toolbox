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

    def _problem_and_frontend(self, kind_name, M=60):
        """Build a corewise `Problem` and the matching frontend `GaussNewtonModel` (residual at self.X)."""
        rng, A, X = self.rng, self.A, self.X
        if kind_name == 'probe':
            ww = unit_vecs(M, SHAPE, rng); data = dense_probe(A, ww)
            problem = opt.least_squares_problem(opt.COREWISE, bfit.PROBE, ww, data)
            r = [np.asarray(p) - d for p, d in zip(X.probe(ww), data)]
            fmodel = fitting.probe_model(t3m.COREWISE, X, ww, r)
        elif kind_name == 'apply':
            ww = unit_vecs(M, SHAPE, rng); data = dense_apply(A, ww)
            problem = opt.least_squares_problem(opt.COREWISE, bfit.APPLY, ww, data)
            r = np.asarray(X.apply(ww)) - data
            fmodel = fitting.apply_model(t3m.COREWISE, X, ww, r)
        else:  # entries
            flat = rng.choice(int(np.prod(SHAPE)), size=M, replace=False)
            index = np.array(np.unravel_index(flat, SHAPE)); data = A[tuple(index)]
            problem = opt.least_squares_problem(opt.COREWISE, bfit.ENTRIES, index, data)
            r = np.asarray(X.entries(index)) - data
            fmodel = fitting.entries_model(t3m.COREWISE, X, index, r)
        return problem, fmodel

    def test_oracle_matches_frontend(self):
        """The backend LocalModel reproduces GaussNewtonModel's gradient / objective / gn_quadratic / hvp."""
        for kind in ('apply', 'entries', 'probe'):
            with self.subTest(kind=kind):
                problem, fmodel = self._problem_and_frontend(kind)
                lm = problem.local_model(self.X.data)

                def relerr_tree(a, b):
                    return float(cw.corewise_norm(cw.corewise_sub(a, b)) / cw.corewise_norm(b))

                self.assertLess(relerr_tree(lm.gradient, fmodel.gradient.variations.data), 1e-12)
                self.assertLess(abs(float(lm.objective) - float(fmodel.objective_value))
                                / abs(float(fmodel.objective_value)), 1e-12)

                pt = t3m.COREWISE.randn(t3m.COREWISE.base(self.X)); p = pt.variations.data
                self.assertLess(abs(float(lm.gn_quadratic(p)) - float(fmodel.gn_quadratic(pt)))
                                / abs(float(fmodel.gn_quadratic(pt))), 1e-12)
                self.assertLess(relerr_tree(lm.hvp(p), fmodel.gn_hessian(pt).variations.data), 1e-12)

    def test_gradient_descent_descends(self):
        """Cauchy + Armijo gradient_descent decreases the loss monotonically (robust on the corewise chart)."""
        problem, _ = self._problem_and_frontend('probe')
        # rescale the start so the initial probes match the data magnitude (well-conditioned start)
        def rms(arrs):
            ss = sum(float(np.sum(np.asarray(a) ** 2)) for a in arrs)
            return float(np.sqrt(ss / sum(np.asarray(a).size for a in arrs)))
        sc = (rms(problem.data) / rms(self.X.probe(problem.sample))) ** (1.0 / (len(TUCKER) + len(TT) - 1))
        x0 = (tuple(sc * C for C in self.X.data[0]), tuple(sc * C for C in self.X.data[1]))

        _, stats = opt.gradient_descent(problem, x0, n_iter=60)
        L = stats['losses']
        self.assertTrue(all(L[i + 1] <= L[i] + 1e-9 * L[0] for i in range(len(L) - 1)), "not monotone")
        self.assertLess(L[-1], 0.5 * L[0], "did not make substantial progress")


if __name__ == "__main__":
    unittest.main()
