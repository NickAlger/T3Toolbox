"""Frontend optimizer adapter (t3toolbox/optimizers.py) -- end-to-end: it returns a TuckerTensorTrain,
the loss descends, and the fit recovers a low-rank target. The heavy correctness check (backend oracle ==
GaussNewtonModel) lives in tests/backend/test_optimizers.py."""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
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
        np.random.seed(1)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed for determinism
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

    def test_mc_sgd_and_adam_adapters(self):
        """The stochastic adapters (mc_sgd on manifold, adam on corewise) run end-to-end and recover."""
        e0 = self._true_err(self.X0)
        x_m, _ = topt.mc_sgd(t3m.MANIFOLD, 'probe', self.ww, self.data, self.X0,
                             np.random.default_rng(5), batch=40, max_iter=400)
        self.assertIsInstance(x_m, t3.TuckerTensorTrain)
        self.assertLess(self._true_err(x_m), 0.4 * e0)
        x_a, _ = topt.adam(t3m.COREWISE, 'probe', self.ww, self.data, self.X0,
                           np.random.default_rng(5), batch=40, lr=2e-2, max_iter=500)
        self.assertIsInstance(x_a, t3.TuckerTensorTrain)
        self.assertLess(self._true_err(x_a), 0.4 * e0)

    def test_newton_cg_adapter(self):
        """Newton-CG through the adapter recovers the (exact low-rank, noiseless) target to high accuracy."""
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)   # manifold zero-start is valid
        x, _ = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, x0, max_newton=30)
        self.assertIsInstance(x, t3.TuckerTensorTrain)
        self.assertLess(self._true_err(x), 1e-3)

    def test_bad_kind_errors(self):
        with self.assertRaises(ValueError):
            topt.gradient_descent(t3m.MANIFOLD, 'nope', self.ww, self.data, self.X0, n_iter=1)

    def test_verbose_display_and_diagnostics(self):
        """D4: verbose=True prints a per-iteration block (captured) and returns stats['diagnostics'] with
        the per-mode error matrices + validation column; a custom callback= overrides verbose."""
        import io, contextlib
        rng = np.random.default_rng(9)
        wwv = [w / np.linalg.norm(w, axis=1, keepdims=True)
               for w in (rng.standard_normal((40, N)) for N in SHAPE)]
        datav = dense_probe(self.A, wwv)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            x, stats = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, t3.TuckerTensorTrain(*x0),
                                      verbose=True, val_sample=wwv, val_data=datav, max_newton=6)
        out = buf.getvalue()
        self.assertIn('rel err', out); self.assertIn('cols=mode', out)      # the plain-probe table printed
        self.assertIn('diagnostics', stats)
        self.assertEqual(len(stats['diagnostics']), len(stats['history']))
        self.assertEqual(np.asarray(stats['diagnostics'][0]['train_err']).shape, (len(SHAPE), 1))
        self.assertIn('val_err', stats['diagnostics'][0])

        # a custom callback overrides verbose and receives NewtonInfo
        seen = []
        topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, t3.TuckerTensorTrain(*x0),
                       callback=seen.append, verbose=True, max_newton=3)
        self.assertTrue(seen and hasattr(seen[0], 'gnorm'))


    def test_per_mode_weight_recovers(self):
        """A per-mode-weighted plain-probe Newton-CG through the adapter still recovers the exact target
        (the noiseless minimizer is weight-independent) -- exercises the topt weight plumbing."""
        d = len(SHAPE)
        omega = np.linspace(0.5, 2.0, d)                      # per-mode (d,)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)
        x, _ = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, x0, weight=omega, max_newton=30)
        self.assertIsInstance(x, t3.TuckerTensorTrain)
        self.assertLess(self._true_err(x), 1e-3)

    def test_weight_contracts_rejected(self):
        """Plain apply/entries take no residual weight (no mode/order axis); a per-mode weight to plain
        probe must be 1-D (a 2-D (d,1) is rejected)."""
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)
        with self.assertRaises(ValueError):    # plain apply: no axis to weight (raised early in _setup)
            topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, np.ones(len(self.data[0])), x0,
                           weight=[1.0], max_newton=1)
        with self.assertRaises(ValueError):    # plain probe: a 2-D (d,1) is rejected (no order axis)
            topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, x0,
                           weight=np.ones((len(SHAPE), 1)), max_newton=1)


class TestFrontendUniformOptimizers(unittest.TestCase):
    """U7a: the same four frontend optimizers accept a UniformTuckerTensorTrain x0 + the uniform geometry
    singletons, inferring the representation from x0's type. The uniform run returns a
    UniformTuckerTensorTrain, recovers the target, and (deterministic gradient_descent) matches the ragged
    run to floating-point tolerance -- the equivalence bar. The frontend calls uniform_minimal(x0)
    transparently, so a non-minimal x0 is accepted (unlike the backend, which rejects it)."""
    def setUp(self):
        # reuse the ragged fixture's problem, then build the uniform twin of the start point.
        r = TestFrontendOptimizers()
        r.setUp()
        self.A, self.A_norm, self.ww, self.data, self.X0 = r.A, r.A_norm, r.ww, r.data, r.X0
        self.uX0 = ut3.UniformTuckerTensorTrain.from_t3(self.X0)

    def _true_err(self, X):  # X is a UniformTuckerTensorTrain
        return float(np.linalg.norm(X.to_dense() - self.A)) / self.A_norm

    def test_gradient_descent_matches_ragged(self):
        """Deterministic gradient_descent on the uniform layer returns a UniformTuckerTensorTrain and its
        loss trajectory matches the ragged run to tolerance (the equivalence contract; ~1e-14 rel here --
        packed-vs-ragged summation order, never bit-exactness)."""
        xr, sr = topt.gradient_descent(t3m.MANIFOLD, 'probe', self.ww, self.data, self.X0, n_iter=8)
        xu, su = topt.gradient_descent(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, self.uX0, n_iter=8)
        self.assertIsInstance(xu, ut3.UniformTuckerTensorTrain)
        self.assertTrue(np.allclose(sr['losses'], su['losses']))
        self.assertTrue(np.allclose(xu.to_dense(), xr.to_dense(), atol=1e-8))

    def test_newton_cg_recovers(self):
        """Newton-CG on the uniform manifold recovers the exact low-rank target to high accuracy from a
        zero start (returned in kind)."""
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT))
        xu, _ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, ux0, max_newton=30)
        self.assertIsInstance(xu, ut3.UniformTuckerTensorTrain)
        self.assertLess(self._true_err(xu), 1e-3)

    def test_per_mode_weight_recovers(self):
        """A per-mode-weighted uniform plain-probe Newton-CG recovers the exact target (the uniform twin of
        the ragged weight plumbing)."""
        omega = np.linspace(0.5, 2.0, len(SHAPE))
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT))
        xu, _ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, ux0, weight=omega, max_newton=30)
        self.assertIsInstance(xu, ut3.UniformTuckerTensorTrain)
        self.assertLess(self._true_err(xu), 1e-3)

    def test_verbose_display_matches_ragged(self):
        """D6: verbose Newton-CG works on the uniform layer (packed block_sumsq, auto-packed validation)
        and its diagnostic error matrices match the ragged run (the equivalence contract)."""
        import io, contextlib
        rng = np.random.default_rng(9)
        wwv = [w / np.linalg.norm(w, axis=1, keepdims=True)
               for w in (rng.standard_normal((40, N)) for N in SHAPE)]
        datav = dense_probe(self.A, wwv)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            xu, su = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, self.uX0,
                                    verbose=True, val_sample=wwv, val_data=datav, max_newton=6)
            _, sr = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, self.X0,
                                   verbose=True, val_sample=wwv, val_data=datav, max_newton=6)
        self.assertIsInstance(xu, ut3.UniformTuckerTensorTrain)
        self.assertIn('cols=mode', buf.getvalue())                     # the plain-probe table printed
        self.assertEqual(np.asarray(su['diagnostics'][0]['train_err']).shape, (len(SHAPE), 1))
        # uniform diagnostics match the ragged run's, block for block (same fit, packed == ragged)
        for du, dr in zip(su['diagnostics'], sr['diagnostics']):
            self.assertTrue(np.allclose(du['train_err'], dr['train_err'], atol=1e-7))
            self.assertTrue(np.allclose(du['val_err'], dr['val_err'], atol=1e-7))

    def test_mc_sgd_and_adam_adapters(self):
        """The stochastic adapters (mc_sgd on the uniform manifold, adam on uniform corewise) run
        end-to-end on packed minibatches and recover."""
        e0 = self._true_err(self.uX0)
        x_m, _ = topt.mc_sgd(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, self.uX0,
                             np.random.default_rng(5), batch=40, max_iter=400)
        self.assertIsInstance(x_m, ut3.UniformTuckerTensorTrain)
        self.assertLess(self._true_err(x_m), 0.4 * e0)
        x_a, _ = topt.adam(ut3m.UNIFORM_COREWISE, 'probe', self.ww, self.data, self.uX0,
                           np.random.default_rng(5), batch=40, lr=2e-2, max_iter=500)
        self.assertIsInstance(x_a, ut3.UniformTuckerTensorTrain)
        self.assertLess(self._true_err(x_a), 0.4 * e0)

    def test_representation_geometry_must_match(self):
        """A uniform x0 with a ragged geometry (or a ragged x0 with a uniform geometry) is a structural
        error -- the geometry must match x0's representation."""
        with self.assertRaises(ValueError):   # uniform x0 + ragged geometry
            topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, self.uX0, max_newton=1)
        with self.assertRaises(ValueError):   # ragged x0 + uniform geometry
            topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, self.X0, max_newton=1)

    def test_nonminimal_x0_handled_transparently(self):
        """The frontend reduces a non-minimal x0 to minimal ranks (uniform_minimal) transparently -- unlike
        the backend uniform_least_squares_problem, which rejects it. So a frontend user never meets the
        minimal-rank requirement."""
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(
            t3.TuckerTensorTrain.randn(SHAPE, (2, 2, 2), (1, 3, 3, 1)))   # TT bond 3 unrealizable for 2x2x2
        self.assertFalse(bool(np.all(ux0.has_minimal_ranks)))            # genuinely non-minimal
        xu, stats = topt.gradient_descent(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, ux0, n_iter=5)
        self.assertIsInstance(xu, ut3.UniformTuckerTensorTrain)
        self.assertTrue(bool(np.all(xu.has_minimal_ranks)))             # ran from the reduced (minimal) frame
        self.assertLess(stats['losses'][-1], stats['losses'][0])


if __name__ == "__main__":
    unittest.main()
