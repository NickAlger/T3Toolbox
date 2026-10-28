"""Frontend optimizer adapter (t3toolbox/optimizers.py) -- end-to-end: it returns a TuckerTensorTrain,
the loss descends, and the fit recovers a low-rank target. The heavy correctness check (backend oracle ==
fitting.LocalModel) lives in tests/backend/test_optimizers.py."""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.optimizers as topt
from t3toolbox.backend.common import jax_available

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

    def test_newton_cg_g0norm_keywords_forward(self):
        """The warm-start reference overrides (g0norm_newton / g0norm_cg / cg_forcing_power) forward through
        the adapter's explicit keywords to the backend loop -- verified via the reported reference + η in
        NewtonInfo."""
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)
        seen = []
        topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, x0,
                       callback=seen.append, g0norm_newton=7.0, g0norm_cg=3.0,
                       cg_forcing_power=1.0, max_newton=3)
        self.assertTrue(seen and not seen[0].converged)
        s = seen[0]
        self.assertAlmostEqual(s.g0norm, 7.0, places=9)                             # Newton reference forwarded
        self.assertAlmostEqual(s.forcing_eta, min(0.5, (s.gnorm / 3.0) ** 1.0), places=9)   # CG ref + power forwarded

    def test_newton_cg_use_jit_returns_jax(self):
        """use_jit=True auto-converts the numpy inputs to jax through the adapter and returns a jax-backed
        TuckerTensorTrain (the 'opting into jit = opting into jax world' contract)."""
        from t3toolbox.backend.common import is_jax_ndarray
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)
        x, _ = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, x0, use_jit=True, max_newton=3)
        self.assertIsInstance(x, t3.TuckerTensorTrain)
        self.assertTrue(is_jax_ndarray(x.data[0][0]))         # jax-backed result

    def test_stochastic_regularizer(self):
        """mc_sgd (manifold) and adam (corewise) accept a regularizer (scaled batch/n internally) and shrink
        the solution vs the unregularized fit."""
        nrm = lambda X: float(np.linalg.norm(X.to_dense()))
        x0z = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)
        xu, _ = topt.mc_sgd(t3m.MANIFOLD, 'probe', self.ww, self.data, x0z,
                            np.random.default_rng(5), batch=40, max_iter=400)
        xr, _ = topt.mc_sgd(t3m.MANIFOLD, 'probe', self.ww, self.data, x0z,
                            np.random.default_rng(5), batch=40, max_iter=400,
                            regularizer=topt.IdentityRegularizer(0.3))
        self.assertIsInstance(xr, t3.TuckerTensorTrain)
        self.assertLess(nrm(xr), nrm(xu))

        au, _ = topt.adam(t3m.COREWISE, 'probe', self.ww, self.data, self.X0,
                          np.random.default_rng(5), batch=40, lr=2e-2, max_iter=500)
        ar, _ = topt.adam(t3m.COREWISE, 'probe', self.ww, self.data, self.X0,
                          np.random.default_rng(5), batch=40, lr=2e-2, max_iter=500,
                          regularizer=topt.IdentityRegularizer(1.0))
        self.assertIsInstance(ar, t3.TuckerTensorTrain)
        self.assertLess(nrm(ar), nrm(au))

    def test_newton_cg_regularizer(self):
        """Identity regularization through the adapter: λ=0 recovers, λ>0 shrinks ‖x‖ (ridge); a residual
        weight ω composes with the regularizer (ω-independence). (Uniform reg is covered in
        TestFrontendUniformOptimizers.)"""
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)
        x_u, _ = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, x0, max_newton=30)
        x_r, _ = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, x0,
                                regularizer=topt.IdentityRegularizer(1e-1), max_newton=30)
        self.assertIsInstance(x_r, t3.TuckerTensorTrain)
        self.assertLess(self._true_err(x_u), 1e-3)                                   # λ=0 recovers
        nrm = lambda X: float(np.linalg.norm(X.to_dense()))
        self.assertLess(nrm(x_r), nrm(x_u))                                          # ridge shrinks ‖x‖

        # ω-independence: a per-mode residual weight composes with the regularizer (still shrinks, still runs)
        omega = np.linspace(0.5, 2.0, len(SHAPE))
        x_wr, _ = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, x0,
                                 weight=omega, regularizer=topt.IdentityRegularizer(1e-1), max_newton=30)
        self.assertIsInstance(x_wr, t3.TuckerTensorTrain)
        self.assertLess(nrm(x_wr), nrm(x_u))

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
        with self.assertRaises(ValueError):    # plain apply: no axis to weight (raised early, in least_squares_problem)
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

    def test_regularizer_matches_ragged_and_garbage_robust(self):
        """S3: identity regularization on the uniform layer (1) matches the ragged reg fit exactly (the
        equivalence contract), (2) shrinks the solution, and (3) is robust to garbage in the input padding
        (the §7 tripwire) -- 1e6 written into the masked-out padding must not change the fit."""
        import t3toolbox.backend.ut3_masking as ut3mask
        lam = 0.1
        xr0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)
        xu0 = ut3.UniformTuckerTensorTrain.from_t3(xr0)
        R = topt.IdentityRegularizer(lam)
        xr, _ = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, self.data, xr0, regularizer=R, max_newton=30)
        xu, _ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, xu0, regularizer=R, max_newton=30)
        self.assertIsInstance(xu, ut3.UniformTuckerTensorTrain)
        self.assertTrue(np.allclose(np.asarray(xu.to_dense()), xr.to_dense(), atol=1e-6))   # uniform == ragged
        xu_u, _ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, xu0, max_newton=30)
        self.assertLess(float(np.linalg.norm(np.asarray(xu.to_dense()))),
                        float(np.linalg.norm(np.asarray(xu_u.to_dense()))))                 # ridge shrinks ‖x‖

        # garbage-robustness: 1e6 in the masked-out padding of a (nonzero) start must not change the fit
        clean, _ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, self.uX0, regularizer=R, max_newton=20)
        tk, tt, shape, masks_tuple = self.uX0.data
        onem = ut3mask.ut3_apply_masks((np.ones_like(tk), np.ones_like(tt), shape, masks_tuple))
        uX0g = ut3.UniformTuckerTensorTrain(np.asarray(tk) + 1e6 * (1.0 - np.asarray(onem[0])),
                                            np.asarray(tt) + 1e6 * (1.0 - np.asarray(onem[1])), shape, self.uX0.masks)
        dirty, _ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe', self.ww, self.data, uX0g, regularizer=R, max_newton=20)
        self.assertTrue(np.allclose(np.asarray(dirty.to_dense()), np.asarray(clean.to_dense()), atol=1e-6))

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



class TestReview20260822Frontend(unittest.TestCase):
    """Frontend-adapter findings from the 2026-08-22 review (C6, C9, C12)."""

    def setUp(self):
        np.random.seed(7)
        self.x_true = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        self.ww = tuple(np.random.randn(12, n) for n in self.x_true.shape)
        self.b = self.x_true.apply(self.ww)

    def test_adam_on_manifold_with_non_minimal_x0_runs_and_warns(self):
        # adam allocated its moments at x0's core shapes; the first retraction dropped the redundant rank
        # (n_1 = 3 > r_1 r_2 = 2 here) and step 2 crashed. x0 is now reduced to minimal ranks on entry,
        # and adam-on-manifold warns (gauge-dependent moments).
        x0 = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 1, 1))
        self.assertFalse(x0.has_minimal_ranks)
        with self.assertWarns(RuntimeWarning):
            x_opt, stats = topt.adam(t3m.MANIFOLD, 'apply', self.ww, self.b, x0, np.random.default_rng(0),
                                     batch=6, max_iter=3, lr=1e-3)
        self.assertTrue(x_opt.has_minimal_ranks)
        # the other manifold optimizers also get the reduced start, and a MINIMAL start is untouched
        x_min = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        x_opt, _ = topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, self.b, x_min, max_newton=1)
        self.assertEqual(x_opt.ranks, x_min.ranks)

    def test_validation_args_go_together(self):
        with self.assertRaises(ValueError):
            topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x_true, max_newton=1,
                           verbose=True, val_data=self.b)
        with self.assertRaises(ValueError):
            topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x_true, max_newton=1,
                           verbose=True, val_sample=self.ww)

    def test_kind_string_is_case_insensitive(self):
        a, _ = topt.newton_cg(t3m.MANIFOLD, 'Apply', self.ww, self.b, self.x_true, max_newton=1)
        b, _ = topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x_true, max_newton=1)
        self.assertLess(float(np.linalg.norm(a.to_dense() - b.to_dense())), 1e-12)


class TestLeastSquaresProblemAndKeywords(unittest.TestCase):
    """2026.3.0 (S0e): the public ``optimizers.least_squares_problem`` (which replaced the private
    ``_setup``), its named result, the ``GaussNewtonModel`` -> ``LocalModel`` rename (not aliased), and
    ``newton_cg``'s explicit keywords (no ``**kwargs`` pass-through)."""

    # The keywords T3Polynomial passes to the frontend newton_cg (plan §K) -- all must be explicit.
    T3POLYNOMIAL_KEYS = ('order', 'weight', 'max_newton', 'verbose', 'val_sample', 'val_data',
                         'g0norm_newton', 'gtol_rel', 'use_jit', 'g0_reference', 'chunk_size', 'callback',
                         'hessian')

    def setUp(self):
        np.random.seed(7)
        self.x_true = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        self.ww = tuple(np.random.randn(12, n) for n in self.x_true.shape)
        self.b = self.x_true.apply(self.ww)
        self.x0 = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))

    def test_named_result(self):
        import t3toolbox.backend.optimizers as bopt
        setup = topt.least_squares_problem(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x0)
        self.assertIsInstance(setup, topt.LeastSquaresSetup)
        self.assertIsInstance(setup.problem, bopt.Problem)
        with self.assertRaises(TypeError):             # a named result, not a tuple: no unpacking, no [0]
            problem, x0_cores, rewrap = setup
        with self.assertRaises(TypeError):
            setup[0]
        x = setup.rewrap(setup.x0_cores)               # a minimal ragged start: the same cores back
        self.assertIsInstance(x, t3.TuckerTensorTrain)
        self.assertTrue(x.corewise_equal(self.x0))

    def test_x0_cores_minimal_on_manifold_only(self):
        """On a manifold geometry a non-minimal ragged x0 comes back at minimal ranks (the same tensor); on
        corewise it is untouched."""
        x0 = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 1, 1))
        self.assertFalse(x0.has_minimal_ranks)
        xm = topt.least_squares_problem(t3m.MANIFOLD, 'apply', self.ww, self.b, x0)
        xm = xm.rewrap(xm.x0_cores)
        self.assertTrue(xm.has_minimal_ranks)
        self.assertLess(float(np.linalg.norm(xm.to_dense() - x0.to_dense())),
                        1e-12 * float(np.linalg.norm(x0.to_dense())))                # the same tensor
        xc = topt.least_squares_problem(t3m.COREWISE, 'apply', self.ww, self.b, x0)
        self.assertTrue(xc.rewrap(xc.x0_cores).corewise_equal(x0))

    def test_gradient_matches_frontend_local_model(self):
        """The constructor's problem linearizes to the same gradient as the frontend fitting.LocalModel."""
        import t3toolbox.fitting as fitting
        setup = topt.least_squares_problem(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x0)
        g_tucker, g_tt = setup.problem.local_model(setup.x0_cores).gradient
        model = fitting.apply_model(t3m.MANIFOLD, self.x0, self.ww, self.x0.apply(self.ww) - self.b)
        self.assertIsInstance(model, fitting.LocalModel)
        f_tucker, f_tt = model.gradient.variations.data
        for a, b in zip(tuple(g_tucker) + tuple(g_tt), tuple(f_tucker) + tuple(f_tt)):
            self.assertTrue(np.allclose(a, b, rtol=1e-12, atol=1e-14))

    def test_the_four_optimizers_build_through_it(self):
        from unittest import mock
        with mock.patch.object(topt, 'least_squares_problem', wraps=topt.least_squares_problem) as spy:
            topt.gradient_descent(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x0, n_iter=1)
            topt.mc_sgd(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x0, np.random.default_rng(0), batch=6,
                        max_iter=1)
            topt.adam(t3m.COREWISE, 'apply', self.ww, self.b, self.x0, np.random.default_rng(0), batch=6,
                      max_iter=1)
            topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x0, max_newton=1)
        self.assertEqual(spy.call_count, 4)

    def test_removed_and_public_names(self):
        import t3toolbox
        import t3toolbox.fitting as fitting
        self.assertFalse(hasattr(topt, '_setup'))                       # removed, not aliased
        self.assertFalse(hasattr(fitting, 'GaussNewtonModel'))          # renamed, not aliased
        self.assertFalse(hasattr(t3toolbox, 'GaussNewtonModel'))
        self.assertIs(t3toolbox.LocalModel, fitting.LocalModel)         # the frontend class at the root
        self.assertIn('LocalModel', t3toolbox.__all__)
        self.assertIn('least_squares_problem', topt.__all__)
        self.assertIn('LeastSquaresSetup', topt.__all__)
        self.assertFalse(hasattr(t3toolbox, 'least_squares_problem'))   # module-level only, not at the root
        self.assertNotIn('ragged_backend_geometry', fitting.__all__)

    def test_newton_cg_misspelt_option_raises_before_setup(self):
        """A misspelt option is a TypeError at the frontend call: the problem is never built (it used to
        be forwarded and rejected by the backend loop, after the setup work)."""
        from unittest import mock
        with mock.patch.object(topt, 'least_squares_problem') as spy:
            with self.assertRaises(TypeError):
                topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x0, max_newtn=3)
        spy.assert_not_called()

    def test_newton_cg_keywords_mirror_backend(self):
        """Every backend newton_cg option is an explicit frontend keyword with the backend's default, and
        every keyword T3Polynomial passes is among them."""
        import inspect
        import t3toolbox.backend.optimizers as bopt
        front = inspect.signature(topt.newton_cg).parameters
        back = inspect.signature(bopt.newton_cg).parameters
        self.assertFalse(any(p.kind is inspect.Parameter.VAR_KEYWORD for p in front.values()))
        for name, p in back.items():
            if name in ('problem', 'x0'):
                continue
            with self.subTest(name=name):
                self.assertIn(name, front)
                self.assertEqual(front[name].default, p.default)
        for name in self.T3POLYNOMIAL_KEYS:
            self.assertIn(name, front)

    def test_newton_cg_forwards_each_keyword(self):
        """Each solver keyword reaches the backend loop with the value passed (none dropped or swapped)."""
        from unittest import mock
        import t3toolbox.backend.optimizers as bopt
        opts = dict(max_newton=7, gtol_rel=3e-5, cg_maxiter=11, c_armijo=2e-3, on_line_search_failure='accept',
                    g0_reference='warm_start', g0norm_newton=5.0, g0norm_cg=4.0, cg_forcing_power=0.75,
                    use_jit=False, hessian='exact', curvature_damping=1e-4, step_control='trust_region',
                    tr_radius0=0.5)
        cb = lambda info: None
        with mock.patch.object(bopt, 'newton_cg', side_effect=lambda problem, x0, **kw: (x0, {})) as spy:
            topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x0, callback=cb, **opts)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.kwargs, dict(opts, callback=cb))
        with mock.patch.object(topt, 'least_squares_problem', wraps=topt.least_squares_problem) as lsp, \
                mock.patch.object(bopt, 'newton_cg', side_effect=lambda problem, x0, **kw: (x0, {})):
            topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x0, hessian='Exact')
            topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, self.b, self.x0)
        self.assertEqual([c.kwargs['hessian'] for c in lsp.call_args_list],
                         ['Exact', 'gauss_newton'])                      # for the problem's 'auto' chunk


def _histories_equal(h1, h2, skip=('wall_time',)):
    """Two newton_cg histories equal field by field (NaN == NaN), ignoring the wall clock."""
    import math
    if len(h1) != len(h2):
        return False
    for a, b in zip(h1, h2):
        if a.keys() != b.keys():
            return False
        for k in a:
            if k in skip:
                continue
            u, v = a[k], b[k]
            if isinstance(u, float) and isinstance(v, float) and math.isnan(u) and math.isnan(v):
                continue
            if u != v:
                return False
    return True


def _restart_problem(W):
    """T21's set-up (tests/backend/test_curvature.py, c24's template): probe derivatives of order 2 with
    omega = [1, .5, .25], 5 % noise, a rank-4 target; level 1 fitted at rank 2 to 1e-10, then zero-padded to rank 3."""
    import t3toolbox.backend.optimizers as bopt
    import t3toolbox.backend.fitting as bfit
    import t3toolbox.backend.geometry as bgeo
    rng = np.random.default_rng(7)
    np.random.seed(7)
    shape = (6, 6, 6)
    target = t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1))
    ww = [rng.standard_normal((W, N)) for N in shape]
    pp = [rng.standard_normal((W, N)) for N in shape]
    weight = np.array([1.0, 0.5, 0.25])
    kind = bfit.probe_derivatives_kind(2, weight)
    data = [z + 0.05 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in kind.point_forward(target.data, (ww, pp))]
    problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), kind, (ww, pp), data)
    x1, _ = bopt.newton_cg(problem, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data,
                           max_newton=80, gtol_rel=1e-10)
    return (ww, pp), data, weight, t3.TuckerTensorTrain(*x1).resize(shape, (3, 3, 3), (1, 3, 3, 1))


def _shared_restart_problem(W):
    """T22's set-up (tests/backend/test_curvature.py): ``_restart_problem`` on ``shared_manifold((0, 0, 1))`` -- a
    tied rank-4 target, a shared rank-2 fit (30 Gauss-Newton steps), zero-padded to rank 3 with ``resize(...,
    sharing=)``."""
    import t3toolbox.backend.optimizers as bopt
    import t3toolbox.backend.fitting as bfit
    import t3toolbox.backend.geometry as bgeo
    import t3toolbox.backend.sharing as bsharing
    rng = np.random.default_rng(7)
    np.random.seed(7)
    shape, labels = (6, 6, 6), (0, 0, 1)
    target = t3.TuckerTensorTrain(*bsharing.t3_tie_tucker_factors(
        t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1)).data, labels))
    ww = [rng.standard_normal((W, N)) for N in shape]
    pp = [rng.standard_normal((W, N)) for N in shape]
    weight = np.array([1.0, 0.5, 0.25])
    kind = bfit.probe_derivatives_kind(2, weight)
    data = [z + 0.05 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in kind.point_forward(target.data, (ww, pp))]
    problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps().with_sharing(labels, shape), kind, (ww, pp), data)
    x0 = bsharing.t3_tie_tucker_factors(t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data, labels)
    x1, _ = bopt.newton_cg(problem, x0, max_newton=30, gtol_rel=1e-10)
    return (ww, pp), data, weight, t3.TuckerTensorTrain(*x1).resize(shape, (3, 3, 3), (1, 3, 3, 1), sharing=labels)


class TestFrontendExactNewton(unittest.TestCase):
    """S6: exact Newton through the frontend ``optimizers.newton_cg`` -- the keywords reach the backend loop, the
    run is bit-identical to the backend run on ``least_squares_problem``'s problem (every history field but the
    wall clock, every loss, the final cores), T21 reproduces through the frontend, ``chunk_size='auto'`` is sized
    for the exact mode, and the layers / geometries not built yet raise their structural errors."""

    def test_frontend_is_the_backend_run_bitwise(self):
        """The frontend newton_cg is the backend newton_cg on ``least_squares_problem(..., hessian=)``'s problem
        and x0, bit for bit, for the exact Hessian with each step control, a damping, a radius and a forcing
        power, and for Gauss-Newton; and the uniform Gauss-Newton run likewise."""
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        sample, data, weight, restart = _restart_problem(40)
        runs = [dict(hessian='exact'),
                dict(hessian='exact', curvature_damping=0.0, cg_forcing_power=1.0, tr_radius0=0.3),
                dict(hessian='exact', step_control='line_search', g0_reference='warm_start'),
                dict(step_control='trust_region'),
                {}]
        for kw in runs:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                xf, sf = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', sample, data, restart, order=2,
                                        weight=weight, max_newton=12, **kw)
                setup = topt.least_squares_problem(t3m.MANIFOLD, 'probe_derivatives', sample, data, restart,
                                                   order=2, weight=weight, hessian=kw.get('hessian', 'gauss_newton'))
                xb, sb = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=12, **kw)
                self.assertEqual(sf['losses'], sb['losses'])
                self.assertTrue(_histories_equal(sf['history'], sb['history']))
                self.assertEqual((sf['newton_steps'], sf['newton_accepted'], sf['stop_reason']),
                                 (sb['newton_steps'], sb['newton_accepted'], sb['stop_reason']))
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[0] + xf.data[1], xb[0] + xb[1])))
                self.assertEqual(sf['history'][0]['hessian'], kw.get('hessian', 'gauss_newton'))
        ux = ut3.UniformTuckerTensorTrain.from_t3(restart)
        xf, sf = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', sample, data, ux, order=2,
                                weight=weight, max_newton=6, chunk_size=None)
        setup = topt.least_squares_problem(ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', sample, data, ux, order=2,
                                           weight=weight, chunk_size=None)
        xb, sb = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=6)
        self.assertEqual(sf['losses'], sb['losses'])
        self.assertTrue(_histories_equal(sf['history'], sb['history']))
        self.assertTrue(np.array_equal(xf.data[0], xb[0]) and np.array_equal(xf.data[1], xb[1]))

    def test_zero_padded_restart_through_the_frontend(self):
        """T21 through the frontend: from the zero-padded rank-3 restart the exact Hessian (defaults: the zero
        reference, the trust region, power 0.5) converges on the gradient test with no rejection, in a fraction
        of Gauss-Newton + Armijo's steps -- observed 8 outer steps against 28, the S5 backend numbers (the run
        is the backend's bit for bit: ``test_frontend_is_the_backend_run_bitwise``); ``verbose=True`` prints the
        trust-region rows and keeps the diagnostics."""
        import contextlib
        import io
        sample, data, weight, restart = _restart_problem(60)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, ex = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', sample, data, restart, order=2, weight=weight,
                                   max_newton=40, hessian='exact', verbose=True)
        _, gn = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', sample, data, restart, order=2, weight=weight,
                               max_newton=40)
        h = ex['history']
        self.assertEqual(ex['stop_reason'], 'gradient')
        self.assertEqual(sum(row['tr_accepted'] is False for row in h), 0)
        self.assertLessEqual(ex['newton_steps'], 10)                  # observed 8 (S5's backend T21: 8, 0 rejected)
        self.assertGreater(gn['newton_steps'], 2 * ex['newton_steps'])   # observed 28
        self.assertEqual(gn['stop_reason'], 'gradient')
        self.assertEqual(h[0]['lift_sigma_min'], 0.0)                 # the padded slots: a singular lift
        self.assertTrue(all(row['step_control'] == 'trust_region' for row in h))
        self.assertIn('TR Δ', buf.getvalue())
        self.assertEqual(len(ex['diagnostics']), len(h))

    @unittest.skipUnless(jax_available, "chunk_size='auto' resolves through the jax-measured estimator")
    def test_chunk_size_auto_is_sized_for_the_exact_mode(self):
        """S4 review m6: ``least_squares_problem(hessian=)`` hands the solver's mode to the ``'auto'`` resolution
        (the estimator then sizes the chunk for the exact two-hole too), and ``newton_cg`` passes its own; the
        resolved chunk is the kind's field. A bad mode raises before any work."""
        from unittest import mock
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        import t3toolbox.backend.sampling_derivatives as sd
        sample, data, weight, restart = _restart_problem(30)
        ux = ut3.UniformTuckerTensorTrain.from_t3(restart)
        with mock.patch.object(sd, 'estimate_chunk_size', side_effect=lambda *a, **k: 7) as spy:
            setup = topt.least_squares_problem(ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', sample, data, ux,
                                               order=2, weight=weight, hessian='EXACT')
            self.assertEqual(setup.problem.kind.chunk_size, 7)
            topt.least_squares_problem(ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', sample, data, ux, order=2)
            _, st = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', sample, data, ux, order=2,
                                   weight=weight, max_newton=1, hessian='exact')   # sized for the exact two-hole
            self.assertEqual(st['history'][0]['hessian'], 'exact')
            topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', sample, data, ux, order=2, weight=weight,
                           max_newton=1)
            topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', sample, data, restart, order=2, weight=weight,
                           max_newton=1, hessian='exact')                        # ragged: no estimate
        self.assertEqual([c.kwargs['hessian'] for c in spy.call_args_list],
                         ['exact', 'gauss_newton', 'exact', 'gauss_newton'])
        with mock.patch.object(topt, '_resolve_chunk_size') as resolve, \
                mock.patch.object(topt, '_ragged_minimal') as reduce_:
            for bad in ('newton', True, None):
                with self.subTest(hessian=bad):
                    with self.assertRaisesRegex(ValueError, "least_squares_problem: hessian must be"):
                        topt.least_squares_problem(t3m.MANIFOLD, 'probe_derivatives', sample, data, restart,
                                                   order=2, hessian=bad)
        resolve.assert_not_called()
        reduce_.assert_not_called()

    def test_options_with_no_effect_and_unbuilt_geometries_raise(self):
        """The backend's structural rules, through the frontend: ``curvature_damping`` with Gauss-Newton or on a
        corewise geometry, ``tr_radius0`` with the line search, an unknown ``step_control``; the exact Hessian
        runs on both shared geometries (the shared manifold's tied lift: S7) and on their uniform twins (S9)."""
        import t3toolbox.shared_geometry as sg
        import t3toolbox.backend.sharing as bsharing
        sample, data, weight, restart = _restart_problem(30)
        call = lambda geom, x, **kw: topt.newton_cg(geom, 'probe_derivatives', sample, data, x, order=2,
                                                    weight=weight, max_newton=3, **kw)
        with self.assertRaisesRegex(ValueError, "curvature_damping is only meaningful with hessian='exact'"):
            call(t3m.MANIFOLD, restart, curvature_damping=1e-6)
        with self.assertRaisesRegex(ValueError, 'no effect on a corewise geometry'):
            call(t3m.COREWISE, restart, hessian='exact', curvature_damping=1e-6)
        with self.assertRaisesRegex(ValueError, "tr_radius0 is only meaningful with step_control='trust_region'"):
            call(t3m.MANIFOLD, restart, tr_radius0=1.0)
        with self.assertRaisesRegex(ValueError, 'step_control'):
            call(t3m.MANIFOLD, restart, hessian='exact', step_control='dogleg')
        from unittest import mock
        with mock.patch.object(topt, 'least_squares_problem') as setup:
            with self.assertRaisesRegex(ValueError, "newton_cg: hessian must be 'gauss_newton' or 'exact'"):
                call(t3m.MANIFOLD, restart, hessian='newton')
        setup.assert_not_called()
        tied = t3.TuckerTensorTrain(*bsharing.t3_tie_tucker_factors(restart.data, (0, 0, 1)))
        utied = ut3.UniformTuckerTensorTrain.from_t3(tied)
        for geom, x in ((sg.shared_manifold((0, 0, 1)), tied), (sg.shared_corewise((0, 0, 1)), tied),
                        (sg.shared(ut3m.UNIFORM_MANIFOLD, (0, 0, 1)), utied),
                        (sg.shared(ut3m.UNIFORM_COREWISE, (0, 0, 1)), utied)):
            with self.subTest(geometry=repr(geom)):
                _, stats = call(geom, x, hessian='exact', chunk_size=None)      # (uniform: skip the 'auto' estimate)
                self.assertEqual(stats['history'][0]['hessian'], 'exact')
        with self.assertRaisesRegex(ValueError, 'curvature_damping must be >= 0'):
            call(sg.shared_manifold((0, 0, 1)), tied, hessian='exact', curvature_damping=-1.0)
        with self.assertRaisesRegex(ValueError, 'curvature_damping must be >= 0'):
            call(sg.shared(ut3m.UNIFORM_MANIFOLD, (0, 0, 1)), utied, hessian='exact', curvature_damping=-1.0,
                 chunk_size=None)
        with self.assertRaisesRegex(ValueError, 'no effect on a corewise geometry'):
            call(sg.shared(ut3m.UNIFORM_COREWISE, (0, 0, 1)), utied, hessian='exact', curvature_damping=1e-6,
                 chunk_size=None)

    def test_zero_padded_shared_restart_through_the_frontend(self):
        """T22 through the frontend (S7): ``newton_cg(shared_manifold((0, 0, 1)), ..., hessian='exact')`` from a
        zero-padded SHARED restart converges on the gradient test with no rejection, its first row reporting the
        zero group-spectrum level (``lift_sigma_min == 0``), every iterate tied; the run is the backend run on
        ``least_squares_problem``'s problem, bit for bit. |W| = 40 here (the backend T22 uses 60); observed 8
        outer steps, 0 rejected, 31 CG iterations."""
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.shared_geometry as sg
        sample, data, weight, restart = _shared_restart_problem(40)
        geom = sg.shared_manifold((0, 0, 1))
        xf, sf = topt.newton_cg(geom, 'probe_derivatives', sample, data, restart, order=2, weight=weight,
                                max_newton=40, hessian='exact')
        h = sf['history']
        self.assertEqual(sf['stop_reason'], 'gradient')
        self.assertEqual(sum(row['tr_accepted'] is False for row in h), 0)
        self.assertLessEqual(sf['newton_steps'], 12)
        self.assertEqual(h[0]['lift_sigma_min'], 0.0)
        self.assertTrue(bool(np.all(xf.has_shared_tucker_factors((0, 0, 1)))))
        setup = topt.least_squares_problem(geom, 'probe_derivatives', sample, data, restart, order=2, weight=weight,
                                           hessian='exact')
        xb, sb = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=40, hessian='exact')
        self.assertEqual(sf['losses'], sb['losses'])
        self.assertTrue(_histories_equal(sf['history'], sb['history']))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[0] + xf.data[1], xb[0] + xb[1])))

    def test_zero_padded_shared_restart_through_the_uniform_frontend(self):
        """T22 through the frontend on the SHARED UNIFORM layer (S9): ``newton_cg(shared(UNIFORM_MANIFOLD, (0, 0, 1)),
        ..., hessian='exact')`` from the zero-padded shared restart (as a uniform point) converges on the gradient test
        with no more than two rejections, its first row reporting the zero group-spectrum level (the real level, not a
        padded one), every iterate tied; the run is the backend uniform run on ``least_squares_problem``'s problem, bit
        for bit. Its trajectory is not the ragged run's -- a padded restart is rank-deficient, where the two layers'
        frames legitimately differ (``docs/uniform_equivalence_contract.md``) -- but it reaches the same minimizer.
        |W| = 40 (the ragged frontend twin above); observed 11 outer steps, 0 rejected, 44 CG iterations (ragged
        8 / 0 / 31), the same final objective."""
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.shared_geometry as sg
        sample, data, weight, restart = _shared_restart_problem(40)
        urestart = ut3.UniformTuckerTensorTrain.from_t3(restart)
        geom = sg.shared(ut3m.UNIFORM_MANIFOLD, (0, 0, 1))
        xf, sf = topt.newton_cg(geom, 'probe_derivatives', sample, data, urestart, order=2, weight=weight,
                                max_newton=40, hessian='exact', chunk_size=None)
        h = sf['history']
        self.assertEqual(sf['stop_reason'], 'gradient')
        self.assertLessEqual(sum(row['tr_accepted'] is False for row in h), 2)
        self.assertLessEqual(sf['newton_steps'], 14)
        self.assertLess(h[0]['lift_sigma_min'], 1e-12)
        self.assertTrue(bool(np.all(xf.has_shared_tucker_factors((0, 0, 1)))))
        setup = topt.least_squares_problem(geom, 'probe_derivatives', sample, data, urestart, order=2, weight=weight,
                                           hessian='exact', chunk_size=None)
        xb, sb = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=40, hessian='exact')
        self.assertEqual(sf['losses'], sb['losses'])
        self.assertTrue(_histories_equal(sf['history'], sb['history']))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[:2], xb)))
        _, sr = topt.newton_cg(sg.shared_manifold((0, 0, 1)), 'probe_derivatives', sample, data, restart, order=2,
                               weight=weight, max_newton=40, hessian='exact', chunk_size=None)
        self.assertLess(abs(sf['losses'][-1] - sr['losses'][-1]) / sr['losses'][-1], 1e-9)   # the same minimizer


def _plain_fit_problem(kind, W=60, seed=13):
    """A misspecified plain-kind fit (S10 / S11): rank-2 fits of a rank-3 target on (6, 6, 6), 5 % noise, from a
    random start; the sample is ww (probe / apply) or an index grid (entries)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    shape = (6, 6, 6)
    target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
    if kind == 'entries':
        sample = np.stack([rng.integers(0, N, size=3 * W) for N in shape])
        clean = np.asarray(target.entries(sample))
    else:
        sample = [rng.standard_normal((W, N)) for N in shape]
        clean = np.asarray(target.apply(sample)) if kind == 'apply' else target.probe(sample)
    noisy = lambda z: z + 0.05 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
    data = [noisy(np.asarray(z)) for z in clean] if kind == 'probe' else noisy(clean)
    return sample, data, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))


class TestFrontendExactNewtonPlainKinds(unittest.TestCase):
    """S10 / S11 through the frontend: ``optimizers.newton_cg(geometry, 'probe' | 'apply' | 'entries', ...,
    hessian='exact')`` converges on the gradient test in fewer outer steps than Gauss-Newton on a misspecified fit,
    and is the backend run on ``least_squares_problem``'s problem bit for bit -- ragged and uniform (where
    ``chunk_size='auto'`` resolves to None for these kinds, in both modes: their transposes do not chunk)."""

    def test_probe(self):
        self._converges_and_is_the_backend_run('probe')

    def test_apply(self):
        self._converges_and_is_the_backend_run('apply')

    def test_entries(self):
        self._converges_and_is_the_backend_run('entries')

    def _converges_and_is_the_backend_run(self, kind):
        from unittest import mock
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.backend.sampling_derivatives as sd
        sample, data, x0 = _plain_fit_problem(kind)
        weight = [0.5, 1.0, 2.0] if kind == 'probe' else None
        with self.subTest(kind=kind, layer='ragged'):
            xf, ex = topt.newton_cg(t3m.MANIFOLD, kind, sample, data, x0, weight=weight, max_newton=60,
                                    hessian='exact')
            _, gn = topt.newton_cg(t3m.MANIFOLD, kind, sample, data, x0, weight=weight, max_newton=40)
            self.assertEqual(ex['stop_reason'], 'gradient')
            self.assertLess(ex['newton_steps'], gn['newton_steps'])     # observed probe 13 / 24, apply 24 / >= 40,
            #                                                           # entries 16 / 36 (exact / Gauss-Newton)
            setup = topt.least_squares_problem(t3m.MANIFOLD, kind, sample, data, x0, weight=weight,
                                               hessian='exact')
            xb, sb = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=60, hessian='exact')
            self.assertEqual(ex['losses'], sb['losses'])
            self.assertTrue(_histories_equal(ex['history'], sb['history']))
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[0] + xf.data[1], xb[0] + xb[1])))
        with self.subTest(kind=kind, layer='uniform'):
            ux = ut3.UniformTuckerTensorTrain.from_t3(x0)
            with mock.patch.object(sd, 'estimate_chunk_size') as spy:
                xu, uf_ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, kind, sample, data, ux, weight=weight,
                                         max_newton=8, hessian='exact', chunk_size='auto')
                setup = topt.least_squares_problem(ut3m.UNIFORM_MANIFOLD, kind, sample, data, ux, weight=weight,
                                                   hessian='exact', chunk_size='auto')
            spy.assert_not_called()
            xb, sb = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=8, hessian='exact')
            self.assertEqual(uf_['losses'], sb['losses'])
            self.assertTrue(_histories_equal(uf_['history'], sb['history']))
            self.assertTrue(np.array_equal(xu.data[0], xb[0]) and np.array_equal(xu.data[1], xb[1]))
            self.assertEqual(uf_['history'][0]['hessian'], 'exact')


def _derivative_fit_problem(kind, W=60, seed=13, order=2):
    """A misspecified apply / entries derivative fit (S12): rank-2 fits of a rank-3 target on (6, 6, 6), orders
    0..order, 5 % noise, from a random start; the sample is (ww, pp) or (index, pp)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    shape = (6, 6, 6)
    target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
    pp = [rng.standard_normal((W, N)) for N in shape]
    if kind == 'entries_derivatives':
        first = np.stack([rng.integers(0, N, size=W) for N in shape])
        clean = np.asarray(target.entries_derivatives(first, pp, order))
    else:
        first = [rng.standard_normal((W, N)) for N in shape]
        clean = np.asarray(target.apply_derivatives(first, pp, order))
    data = clean + 0.05 * np.linalg.norm(clean) / clean.size ** 0.5 * rng.standard_normal(clean.shape)
    return (first, pp), data, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))


class TestFrontendExactNewtonDerivativeKinds(unittest.TestCase):
    """S12 through the frontend: ``optimizers.newton_cg(geometry, 'apply_derivatives' | 'entries_derivatives', ...,
    hessian='exact')`` (order 2, a per-order weight) converges on the gradient test in fewer outer steps than
    Gauss-Newton on a misspecified fit, to the same minimizer, and is the backend run on ``least_squares_problem``'s
    problem bit for bit -- ragged and uniform (where ``chunk_size='auto'`` resolves to None for these kinds, in both
    modes, without the estimator: their transposes and two-holes do not chunk)."""

    def test_apply_derivatives(self):
        self._converges_and_is_the_backend_run('apply_derivatives')

    def test_entries_derivatives(self):
        self._converges_and_is_the_backend_run('entries_derivatives')

    def _converges_and_is_the_backend_run(self, kind):
        from unittest import mock
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.backend.sampling_derivatives as sd
        sample, data, x0 = _derivative_fit_problem(kind)
        weight = [1.0, 0.5, 0.25]
        with self.subTest(kind=kind, layer='ragged'):
            xf, ex = topt.newton_cg(t3m.MANIFOLD, kind, sample, data, x0, order=2, weight=weight, max_newton=60,
                                    hessian='exact')
            _, gn = topt.newton_cg(t3m.MANIFOLD, kind, sample, data, x0, order=2, weight=weight, max_newton=60)
            self.assertEqual(ex['stop_reason'], 'gradient')
            self.assertEqual(gn['stop_reason'], 'gradient')
            self.assertLess(ex['newton_steps'], gn['newton_steps'])     # observed apply 18 / 34, entries 24 / 39
            self.assertLess(abs(ex['losses'][-1] - gn['losses'][-1]), 1e-9 * gn['losses'][-1])   # the same minimum
            setup = topt.least_squares_problem(t3m.MANIFOLD, kind, sample, data, x0, order=2, weight=weight,
                                               hessian='exact')
            xb, sb = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=60, hessian='exact')
            self.assertEqual(ex['losses'], sb['losses'])
            self.assertTrue(_histories_equal(ex['history'], sb['history']))
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[0] + xf.data[1], xb[0] + xb[1])))
        with self.subTest(kind=kind, layer='uniform'):
            ux = ut3.UniformTuckerTensorTrain.from_t3(x0)
            with mock.patch.object(sd, 'estimate_chunk_size') as spy:
                xu, uf_ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, kind, sample, data, ux, order=2, weight=weight,
                                         max_newton=8, hessian='exact', chunk_size='auto')
                setup = topt.least_squares_problem(ut3m.UNIFORM_MANIFOLD, kind, sample, data, ux, order=2,
                                                   weight=weight, hessian='exact', chunk_size='auto')
            spy.assert_not_called()
            xb, sb = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=8, hessian='exact')
            self.assertEqual(uf_['losses'], sb['losses'])
            self.assertTrue(_histories_equal(uf_['history'], sb['history']))
            self.assertTrue(np.array_equal(xu.data[0], xb[0]) and np.array_equal(xu.data[1], xb[1]))
            self.assertEqual(uf_['history'][0]['hessian'], 'exact')


if __name__ == "__main__":
    unittest.main()
