"""Frontend optimizer adapter (t3toolbox/optimizers.py) -- end-to-end: it returns a TuckerTensorTrain,
the loss descends, and the fit recovers a low-rank target. The heavy correctness check (backend oracle ==
GaussNewtonModel) lives in tests/backend/test_optimizers.py."""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt

SHAPE, TUCKER, TT = (8, 8, 8), (3, 3, 3), (1, 3, 3, 1)


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


class TestFrontendOptimizers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.A_t3 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        self.A = self.A_t3.to_dense()
        self.A_norm = float(np.linalg.norm(self.A))
        M = 200
        ww = [rng.standard_normal((M, N)) for N in SHAPE]
        self.ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
        self.data = dense_probe(self.A, self.ww)
        # a scaled random start (well-conditioned)
        X = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        def rms(a): return float(np.sqrt(np.mean(np.concatenate([np.asarray(x).ravel() for x in a]) ** 2)))
        sc = (rms(self.data) / rms(X.probe(self.ww))) ** (1.0 / (len(TUCKER) + len(TT) - 1))
        self.X0 = t3.TuckerTensorTrain(tuple(sc * C for C in X.data[0]), tuple(sc * C for C in X.data[1]))

    def _true_err(self, X):
        return float(np.linalg.norm(X.to_dense() - self.A)) / self.A_norm

    def test_gradient_descent_recovers(self):
        """gradient_descent through the adapter returns a TuckerTensorTrain whose loss descends and whose
        true error drops -- on both geometries."""
        for geometry in (t3m.MANIFOLD, t3m.COREWISE):
            with self.subTest(geometry=type(geometry).__name__):
                x_opt, stats = topt.gradient_descent(geometry, 'probe', self.ww, self.data, self.X0, n_iter=120)
                self.assertIsInstance(x_opt, t3.TuckerTensorTrain)
                L = stats['losses']
                self.assertLess(L[-1], 0.1 * L[0], "loss did not descend substantially")
                self.assertLess(self._true_err(x_opt), 0.5 * self._true_err(self.X0), "true error did not drop")

    def test_bad_kind_errors(self):
        with self.assertRaises(ValueError):
            topt.gradient_descent(t3m.MANIFOLD, 'nope', self.ww, self.data, self.X0, n_iter=1)


if __name__ == "__main__":
    unittest.main()
