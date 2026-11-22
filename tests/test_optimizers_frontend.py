"""Frontend optimizer adapter (t3toolbox/optimizers.py) -- end-to-end: it returns a TuckerTensorTrain,
the loss descends, and the fit recovers a low-rank target. The heavy correctness check (backend oracle ==
fitting.LocalModel) lives in tests/backend/test_optimizers.py."""
import typing as typ
import unittest
import warnings

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.optimizers as topt
from t3toolbox.backend.common import jax_available
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _derivative_data as dd
import _fits as F       # the backend fits and the 2026.5.0 vocabulary in the new form (P16)

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
        from t3toolbox.fitting import LeastSquaresOracle, LeastSquaresBatch
        oracle = LeastSquaresOracle('probe', order=())
        batch, val_batch = LeastSquaresBatch(self.ww, self.data), LeastSquaresBatch(wwv, datav)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            x, stats = topt.newton_cg(t3m.MANIFOLD, oracle, batch, t3.TuckerTensorTrain(*x0),
                                      verbose=True, val_batch=val_batch, max_newton=6)
        out = buf.getvalue()
        self.assertIn('rel err', out); self.assertIn('cols=mode', out)      # the plain-probe table printed
        self.assertIn('diagnostics', stats)
        self.assertEqual(len(stats['diagnostics']), len(stats['history']))
        self.assertEqual(np.asarray(stats['diagnostics'][0]['train_err']).shape, (len(SHAPE),))   # the weight's shape
        self.assertIn('val_err', stats['diagnostics'][0])

        # a custom callback overrides verbose and receives NewtonInfo
        seen = []
        topt.newton_cg(t3m.MANIFOLD, oracle, batch, t3.TuckerTensorTrain(*x0),
                       callback=seen.append, verbose=True, max_newton=3)
        self.assertTrue(seen and hasattr(seen[0], 'gnorm'))

    def test_verbose_error_table_and_the_runs_batch(self):
        """P16: ``verbose=True, error_table=False`` prints each row's header line alone, and ``stats['diagnostics']``
        equals the run with the table (arrays bitwise); the records at ``order=(1, 1)`` keep the weight's shape and the
        table has one column per multi-index, row-major; the display is handed the very batch the run uses, so each
        row's numerator reads the model's held residual (its query on the run's batch carries the model's held), ragged
        and under ``use_jit`` (the batch converted once, before the display)."""
        import io, contextlib, re
        from unittest import mock
        from t3toolbox.fitting import LeastSquaresOracle, LeastSquaresBatch, affine_jet
        rng = np.random.default_rng(3)
        jets = [np.stack([np.stack([w, p]), np.stack([q, np.zeros_like(w)])])
                for w, p, q in zip(*([rng.standard_normal((30, N)) for N in SHAPE] for _ in range(3)))]
        oracle = LeastSquaresOracle('apply', order=(1, 1))
        batch = LeastSquaresBatch(jets, self.A_t3.apply(jets, (1, 1)))
        x0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        clock = re.compile(r'\| \d+\.\d\ds\b')
        texts, stats = [], []
        for flag in (True, False):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats.append(topt.newton_cg(t3m.MANIFOLD, oracle, batch, x0, verbose=True, error_table=flag,
                                            max_newton=3)[1])
            texts.append(clock.sub('| T', buf.getvalue()))
        on, off = stats
        self.assertEqual(texts[1].count('rel err'), 0)
        self.assertEqual(texts[1].split('\n'), [line for line in texts[0].split('\n') if line.startswith('iter ')]
                         + [''])
        self.assertIn('ord(0,0)  ord(0,1)  ord(1,0)  ord(1,1)', texts[0])
        for a, b in zip(on['diagnostics'], off['diagnostics']):
            self.assertEqual(np.shape(a['train_err']), (2, 2))
            self.assertTrue(np.array_equal(a['train_err'], b['train_err']))
        real = LeastSquaresOracle.query
        for use_jit in ((False, True) if jax_available else (False,)):
            calls = []

            def recording(self_, frame, b, want, **kw):
                calls.append((b, tuple(want), kw.get('held')))
                return real(self_, frame, b, want, **kw)
            with mock.patch.object(LeastSquaresOracle, 'query', recording), contextlib.redirect_stdout(io.StringIO()):
                _, st = topt.newton_cg(t3m.MANIFOLD, oracle, batch, x0, verbose=True, max_newton=2, use_jit=use_jit)
            numerators = [c for c in calls if c[1] == ('r_block_sumsq',)]
            self.assertEqual(len(numerators), len(st['history']))
            self.assertTrue(all(c[2] is not None for c in numerators), use_jit)   # the fast path: the model's held


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
        with self.assertRaises(ValueError):    # plain apply: no axis to weight (raised early, in the translation of the old form)
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
        from t3toolbox.fitting import LeastSquaresOracle, LeastSquaresBatch
        oracle = LeastSquaresOracle('probe', order=())
        batch, val_batch = LeastSquaresBatch(self.ww, self.data), LeastSquaresBatch(wwv, datav)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            xu, su = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, oracle, batch, self.uX0,
                                    verbose=True, val_batch=val_batch, max_newton=6)
            _, sr = topt.newton_cg(t3m.MANIFOLD, oracle, batch, self.X0,
                                   verbose=True, val_batch=val_batch, max_newton=6)
        self.assertIsInstance(xu, ut3.UniformTuckerTensorTrain)
        self.assertIn('cols=mode', buf.getvalue())                     # the plain-probe table printed
        self.assertEqual(np.asarray(su['diagnostics'][0]['train_err']).shape, (len(SHAPE),))
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
        the backend uniform geometry's ``from_point``, which rejects it (D5). So a frontend user never meets the
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


class TestFrontendHeadsAndKeywords(unittest.TestCase):
    """P16 part 4: the four optimizers take ``(geometry, oracle, batch, x0, ...)`` (plan 2.5); ``least_squares_problem``,
    ``LeastSquaresSetup`` and the frontend ``chunk_size`` are gone; ``newton_cg``'s keywords are explicit (no
    ``**kwargs``), with ``error_table`` beside ``verbose`` and ``val_batch``; structural problems raise before the
    setup work (``optimizers._setup``)."""

    # The keywords T3Polynomial passes to the frontend newton_cg (plan §K), in the new form -- all must be explicit.
    T3POLYNOMIAL_KEYS = ('max_newton', 'verbose', 'val_batch', 'g0norm_newton', 'gtol_rel', 'use_jit', 'g0_reference',
                         'callback', 'hessian', 'error_table', 'regularizer')

    def setUp(self):
        from t3toolbox.fitting import LeastSquaresOracle, LeastSquaresBatch
        np.random.seed(7)
        self.x_true = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        self.ww = tuple(np.random.randn(12, n) for n in self.x_true.shape)
        self.b = self.x_true.apply(self.ww)
        self.x0 = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        self.oracle = LeastSquaresOracle('apply', order=())
        self.batch = LeastSquaresBatch(self.ww, self.b)

    def test_the_heads(self):
        import inspect
        heads = {'gradient_descent': ['geometry', 'oracle', 'batch', 'x0', 'regularizer', 'g0_reference'],
                 'mc_sgd': ['geometry', 'oracle', 'batch', 'x0', 'rng', 'batch_size', 'regularizer', 'draw', 'use_jit'],
                 'adam': ['geometry', 'oracle', 'batch', 'x0', 'rng', 'batch_size', 'regularizer', 'draw', 'use_jit'],
                 'newton_cg': ['geometry', 'oracle', 'batch', 'x0', 'regularizer', 'verbose', 'error_table',
                               'val_batch', 'callback', 'use_jit']}
        for name, head in heads.items():
            with self.subTest(name=name):
                params = list(inspect.signature(getattr(topt, name)).parameters)
                self.assertEqual(params[:len(head)], head)
                self.assertFalse({'kind', 'sample', 'data', 'order', 'weight', 'chunk_size', 'val_sample',
                                  'val_data'} & set(params))

    def test_x0_minimal_on_manifold_only(self):
        """On a manifold geometry a non-minimal ragged x0 is reduced to minimal ranks (the same tensor) before the
        backend call; on corewise it is untouched."""
        x0 = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 1, 1))
        self.assertFalse(x0.has_minimal_ranks)
        sm = topt._setup(t3m.MANIFOLD, self.oracle, self.batch, x0)
        xm = sm.rewrap(sm.x0_cores)
        self.assertTrue(xm.has_minimal_ranks)
        self.assertLess(float(np.linalg.norm(xm.to_dense() - x0.to_dense())),
                        1e-12 * float(np.linalg.norm(x0.to_dense())))                # the same tensor
        sc = topt._setup(t3m.COREWISE, self.oracle, self.batch, x0)
        self.assertTrue(sc.rewrap(sc.x0_cores).corewise_equal(x0))

    def test_gradient_matches_frontend_local_model(self):
        """The backend model at the setup's cores has the frontend local model's gradient (and the dense one)."""
        import t3toolbox.fitting as fitting
        import t3toolbox.backend.optimizers as bopt
        s = topt._setup(t3m.MANIFOLD, self.oracle, self.batch, self.x0)
        g_tucker, g_tt = bopt.local_model(s.geom, self.oracle, s.batch, s.x0_cores).gradient
        model = fitting.local_model(t3m.MANIFOLD, self.oracle, self.batch, self.x0)
        self.assertIsInstance(model, fitting.LocalModel)
        f_tucker, f_tt = model.gradient.variations.data
        for a, b in zip(tuple(g_tucker) + tuple(g_tt), tuple(f_tucker) + tuple(f_tt)):
            self.assertTrue(np.allclose(a, b, rtol=1e-12, atol=1e-14))

    def test_the_four_optimizers_set_up_through_it(self):
        from unittest import mock
        with mock.patch.object(topt, '_setup', wraps=topt._setup) as spy:
            topt.gradient_descent(t3m.MANIFOLD, self.oracle, self.batch, self.x0, n_iter=1)
            topt.mc_sgd(t3m.MANIFOLD, self.oracle, self.batch, self.x0, np.random.default_rng(0), 6, max_iter=1)
            topt.adam(t3m.COREWISE, self.oracle, self.batch, self.x0, np.random.default_rng(0), 6, max_iter=1)
            topt.newton_cg(t3m.MANIFOLD, self.oracle, self.batch, self.x0, max_newton=1)
        self.assertEqual(spy.call_count, 4)

    def test_removed_and_public_names(self):
        import t3toolbox
        import t3toolbox.fitting as fitting
        for name in ('least_squares_problem', 'LeastSquaresSetup', '_resolve_chunk_size', '_check_kind'):
            self.assertFalse(hasattr(topt, name), name)                  # removed, not aliased
        for name in ('apply_model', 'entries_model', 'probe_model', 'apply_derivatives_model',
                     'entries_derivatives_model', 'probe_derivatives_model', '_check_ragged_sample'):
            self.assertFalse(hasattr(fitting, name), name)
            self.assertFalse(hasattr(t3toolbox, name), name)
        self.assertFalse(hasattr(fitting, 'GaussNewtonModel'))          # renamed, not aliased
        self.assertIs(t3toolbox.LocalModel, fitting.LocalModel)         # the frontend class at the root
        for name in ('LocalModel', 'local_model', 'LeastSquaresOracle', 'LeastSquaresBatch'):   # D7
            self.assertIn(name, t3toolbox.__all__)
        self.assertIs(t3toolbox.local_model, fitting.local_model)
        self.assertIs(t3toolbox.LeastSquaresOracle, fitting.LeastSquaresOracle)
        self.assertIs(t3toolbox.LeastSquaresBatch, fitting.LeastSquaresBatch)
        self.assertNotIn('ragged_backend_geometry', fitting.__all__)

    def test_newton_cg_misspelt_option_raises_before_setup(self):
        """A misspelt option is a TypeError at the frontend call: the setup never runs."""
        from unittest import mock
        with mock.patch.object(topt, '_setup') as spy:
            with self.assertRaises(TypeError):
                topt.newton_cg(t3m.MANIFOLD, self.oracle, self.batch, self.x0, max_newtn=3)
        spy.assert_not_called()

    def test_a_bad_batch_raises_before_any_work(self):
        """The batch is checked by the oracle (structural) before x0 is reduced or anything is built; the backend
        optimizer takes the same bad batch without a check (it fails later, inside the engine, or not at all)."""
        from unittest import mock
        from t3toolbox.fitting import LeastSquaresBatch
        import t3toolbox.backend.optimizers as bopt
        bad = LeastSquaresBatch(self.ww[:2], self.b)                    # two vectors for a 3-mode tensor
        calls = (('newton_cg', lambda: topt.newton_cg(t3m.MANIFOLD, self.oracle, bad, self.x0, max_newton=1)),
                 ('gradient_descent', lambda: topt.gradient_descent(t3m.MANIFOLD, self.oracle, bad, self.x0, n_iter=1)))
        for name, call in calls:
            with mock.patch.object(topt, '_ragged_minimal') as reduce, mock.patch.object(bopt, name) as run:
                with self.assertRaises(ValueError):
                    call()
            reduce.assert_not_called()
            run.assert_not_called()
        with mock.patch.object(topt, '_setup') as setup:                # a bad val_batch: before the setup too
            with self.assertRaises(ValueError):
                topt.newton_cg(t3m.MANIFOLD, self.oracle, self.batch, self.x0, val_batch=bad, verbose=True,
                               max_newton=1)
        setup.assert_not_called()
        with mock.patch.object(self.oracle.__class__, 'check_batch') as check:
            try:
                bopt.newton_cg(bgeo_manifold(), self.oracle, bad, self.x0.data, max_newton=1)
            except Exception:
                pass
        check.assert_not_called()                                       # the backend checks nothing

    def test_newton_cg_keywords_mirror_backend(self):
        """Every backend newton_cg option is an explicit frontend keyword with the backend's default, and every
        keyword T3Polynomial passes is among them."""
        import inspect
        import t3toolbox.backend.optimizers as bopt
        front = inspect.signature(topt.newton_cg).parameters
        back = inspect.signature(bopt.newton_cg).parameters
        self.assertFalse(any(p.kind is inspect.Parameter.VAR_KEYWORD for p in front.values()))
        for name, p in back.items():
            if name in ('geom', 'oracle', 'batch', 'x0'):
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
        with mock.patch.object(bopt, 'newton_cg', side_effect=lambda geom, oracle, batch, x0, reg, **kw: (x0, {})) as spy:
            topt.newton_cg(t3m.MANIFOLD, self.oracle, self.batch, self.x0, callback=cb, **opts)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.kwargs, dict(opts, callback=cb))

    def test_newton_cg_forwards_timing_only_when_set(self):
        """T9: ``timing`` is keyword-only, right after ``cone_seed`` (as on the backend); it reaches the
        backend as ``timing=True`` only when set; a numpy bool is a bool."""
        import inspect
        from unittest import mock
        import t3toolbox.backend.optimizers as bopt
        front = inspect.signature(topt.newton_cg).parameters
        names = list(front)
        self.assertEqual(names[names.index('timing') - 1], 'cone_seed')
        self.assertIs(front['timing'].kind, inspect.Parameter.KEYWORD_ONLY)
        for value, forwarded in ((True, True), (np.True_, True), (False, False), (np.False_, False), (None, None)):
            kwargs = {} if value is None else dict(timing=value)
            with self.subTest(timing=value if value is None else bool(value),     # builtin labels: xdist (execnet)
                              numpy_bool=isinstance(value, np.bool_)):          # cannot ship numpy 1.x's np.bool_
                with mock.patch.object(bopt, 'newton_cg',
                                       side_effect=lambda geom, oracle, batch, x0, reg, **kw: (x0, {})) as spy:
                    topt.newton_cg(t3m.MANIFOLD, self.oracle, self.batch, self.x0, **kwargs)
                if forwarded:
                    self.assertIs(spy.call_args.kwargs['timing'], True)
                else:
                    self.assertNotIn('timing', spy.call_args.kwargs)

    def test_newton_cg_timing_must_be_a_bool(self):
        """T9: a ``timing`` that is not a bool is a ``TypeError`` at the call, before the setup."""
        from unittest import mock
        for value in ('yes', 1, 0, None, 1.0):
            with self.subTest(timing=value):
                with mock.patch.object(topt, '_setup') as spy:
                    with self.assertRaisesRegex(TypeError, 'timing must be a bool'):
                        topt.newton_cg(t3m.MANIFOLD, self.oracle, self.batch, self.x0, timing=value)
                spy.assert_not_called()

    def test_newton_cg_timing_runs(self):
        """T9: a timed frontend run, ragged and uniform, returns ``stats['timing']`` with one record per history row
        and the same numbers as the untimed run; with ``verbose=True`` every row's block carries its ``time`` line and
        the run ends with the closing table (printed once)."""
        import contextlib
        import io
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(self.x0)
        for geometry, x0 in ((t3m.MANIFOLD, self.x0), (ut3m.UNIFORM_MANIFOLD, ux0)):
            with self.subTest(layer=type(x0).__name__):
                _, plain = topt.newton_cg(geometry, self.oracle, self.batch, x0, max_newton=4)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    _, timed = topt.newton_cg(geometry, self.oracle, self.batch, x0, max_newton=4, timing=True,
                                              verbose=True)
                self.assertNotIn('timing', plain)
                self.assertEqual(len(timed['timing']['rows']), len(timed['history']))
                self.assertEqual(timed['losses'], plain['losses'])                # timing changes no number
                self.assertTrue(_histories_equal(timed['history'], plain['history']))
                self.assertEqual(len(timed['diagnostics']), len(timed['history']))   # the records are as before
                text = out.getvalue()
                self.assertEqual(text.count('\n  time  '), len(timed['history']))
                self.assertEqual(text.count('  timing  %d rows' % len(timed['history'])), 1)
                self.assertLess(text.rfind('iter '), text.find('  timing  '))     # the table follows the last row


def bgeo_manifold():
    import t3toolbox.backend.geometry as bgeo
    return bgeo.ManifoldGeometryOps()


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


class _Twin(typ.NamedTuple):
    geom:      typ.Any
    x0_cores:  typ.Any
    batch:     typ.Any


def _twin(geometry, oracle, batch, x0):
    """The backend twin of a frontend call, built independently of ``optimizers._setup``: the backend geometry
    (shared: ``with_sharing``), ``x0`` reduced to minimal ranks (on a manifold geometry; always on the uniform
    layer), the batch packed for a uniform ``x0``."""
    import t3toolbox.backend.geometry as bgeo
    import t3toolbox.backend.uniform_fitting as ufit
    import t3toolbox.shared_geometry as sg
    sharing = geometry.sharing if isinstance(geometry, sg.SharedGeometry) else None
    base = geometry.base if isinstance(geometry, sg.SharedGeometry) else geometry
    if isinstance(x0, ut3.UniformTuckerTensorTrain):
        x = ufit.uniform_minimal(x0, sharing=sharing)
        cls = bgeo.UniformManifoldGeometryOps if base is ut3m.UNIFORM_MANIFOLD else bgeo.UniformCorewiseGeometryOps
        return _Twin(cls.from_point(x.data, sharing), (x.data[0], x.data[1]),
                     ufit.pack_batch(oracle.kind, oracle.order, batch, x.N))
    manifold = base is t3m.MANIFOLD
    geom = bgeo.ManifoldGeometryOps() if manifold else bgeo.CorewiseGeometryOps()
    if sharing is not None:
        geom = geom.with_sharing(sharing, x0.shape)
    x = topt._ragged_minimal(x0, sharing) if manifold else x0
    return _Twin(geom, x.data, batch)


def _restart_problem(W):
    """T21's set-up (tests/backend/test_curvature.py, c24's template): probe derivatives of order 2 with
    omega = [1, .5, .25], 5 % noise, a rank-4 target; level 1 fitted at rank 2 to 1e-10, then zero-padded to rank 3.
    Returns (oracle, batch, restart)."""
    import t3toolbox.backend.optimizers as bopt
    import t3toolbox.backend.geometry as bgeo
    rng = np.random.default_rng(7)
    np.random.seed(7)
    shape = (6, 6, 6)
    target = t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1))
    ww = [rng.standard_normal((W, N)) for N in shape]
    pp = [rng.standard_normal((W, N)) for N in shape]
    weight = np.array([1.0, 0.5, 0.25])
    data = [z + 0.05 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in dd.forward(target, 'probe', ww, pp, 2)]
    oracle, batch = F.old_vocabulary('probe_derivatives', (ww, pp), data, shape, order=2, weight=weight)
    x1, _ = bopt.newton_cg(bgeo.ManifoldGeometryOps(), oracle, batch,
                           t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data,
                           max_newton=80, gtol_rel=1e-10)
    return oracle, batch, t3.TuckerTensorTrain(*x1).resize(shape, (3, 3, 3), (1, 3, 3, 1))


def _shared_restart_problem(W):
    """T22's set-up (tests/backend/test_curvature.py): ``_restart_problem`` on ``shared_manifold((0, 0, 1))`` -- a
    tied rank-4 target, a shared rank-2 fit (30 Gauss-Newton steps), zero-padded to rank 3 with ``resize(...,
    sharing=)``. Returns (oracle, batch, restart)."""
    import t3toolbox.backend.optimizers as bopt
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
    data = [z + 0.05 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in dd.forward(target, 'probe', ww, pp, 2)]
    oracle, batch = F.old_vocabulary('probe_derivatives', (ww, pp), data, shape, order=2, weight=weight)
    x0 = bsharing.t3_tie_tucker_factors(t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data, labels)
    x1, _ = bopt.newton_cg(bgeo.ManifoldGeometryOps().with_sharing(labels, shape), oracle, batch, x0, max_newton=30,
                           gtol_rel=1e-10)
    return oracle, batch, t3.TuckerTensorTrain(*x1).resize(shape, (3, 3, 3), (1, 3, 3, 1), sharing=labels)


class TestFrontendExactNewton(unittest.TestCase):
    """S6: exact Newton through the frontend ``optimizers.newton_cg`` -- the keywords reach the backend loop, the
    run is bit-identical to the backend run on its backend twin (``_twin``) (every history field but the
    wall clock, every loss, the final cores), T21 reproduces through the frontend, and the layers / geometries not built yet raise their structural errors."""

    def test_frontend_is_the_backend_run_bitwise(self):
        """The frontend newton_cg is the backend newton_cg on its backend twin's geometry, batch
        and x0, bit for bit, for the exact Hessian with each step control, a damping, a radius and a forcing
        power, and for Gauss-Newton; and the uniform Gauss-Newton run likewise."""
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        oracle, batch, restart = _restart_problem(40)
        runs = [dict(hessian='exact'),
                dict(hessian='exact', curvature_damping=0.0, cg_forcing_power=1.0, tr_radius0=0.3),
                dict(hessian='exact', step_control='line_search', g0_reference='warm_start'),
                dict(step_control='trust_region'),
                {}]
        for kw in runs:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                xf, sf = topt.newton_cg(t3m.MANIFOLD, oracle, batch, restart,
                                        max_newton=12, **kw)
                setup = _twin(t3m.MANIFOLD, oracle, batch, restart)
                xb, sb = bopt.newton_cg(setup.geom, oracle, setup.batch, setup.x0_cores, max_newton=12, **kw)
                self.assertEqual(sf['losses'], sb['losses'])
                self.assertTrue(_histories_equal(sf['history'], sb['history']))
                self.assertEqual((sf['newton_steps'], sf['newton_accepted'], sf['stop_reason']),
                                 (sb['newton_steps'], sb['newton_accepted'], sb['stop_reason']))
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[0] + xf.data[1], xb[0] + xb[1])))
                self.assertEqual(sf['history'][0]['hessian'], kw.get('hessian', 'gauss_newton'))
        ux = ut3.UniformTuckerTensorTrain.from_t3(restart)
        xf, sf = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, oracle, batch, ux,
                                max_newton=6)
        setup = _twin(ut3m.UNIFORM_MANIFOLD, oracle, batch, ux)
        xb, sb = bopt.newton_cg(setup.geom, oracle, setup.batch, setup.x0_cores, max_newton=6)
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
        oracle, batch, restart = _restart_problem(60)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, ex = topt.newton_cg(t3m.MANIFOLD, oracle, batch, restart, max_newton=40, hessian='exact', verbose=True)
        _, gn = topt.newton_cg(t3m.MANIFOLD, oracle, batch, restart, max_newton=40)
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

    def test_options_with_no_effect_and_unbuilt_geometries_raise(self):
        """The backend's structural rules, through the frontend: ``curvature_damping`` with Gauss-Newton or on a
        corewise geometry, ``tr_radius0`` with the line search, an unknown ``step_control``; the exact Hessian
        runs on both shared geometries (the shared manifold's tied lift: S7) and on their uniform twins (S9)."""
        import t3toolbox.shared_geometry as sg
        import t3toolbox.backend.sharing as bsharing
        oracle, batch, restart = _restart_problem(30)
        call = lambda geom, x, **kw: topt.newton_cg(geom, oracle, batch, x,
                                                     max_newton=3, **kw)
        with self.assertRaisesRegex(ValueError, "curvature_damping is only meaningful with hessian='exact'"):
            call(t3m.MANIFOLD, restart, curvature_damping=1e-6)
        with self.assertRaisesRegex(ValueError, 'no effect on a corewise geometry'):
            call(t3m.COREWISE, restart, hessian='exact', curvature_damping=1e-6)
        with self.assertRaisesRegex(ValueError, "tr_radius0 is only meaningful with step_control='trust_region'"):
            call(t3m.MANIFOLD, restart, tr_radius0=1.0)
        with self.assertRaisesRegex(ValueError, 'step_control'):
            call(t3m.MANIFOLD, restart, hessian='exact', step_control='dogleg')
        from unittest import mock
        with mock.patch.object(topt, '_setup') as setup:
            with self.assertRaisesRegex(ValueError, "newton_cg: hessian must be 'gauss_newton', 'exact' or 'switch'"):
                call(t3m.MANIFOLD, restart, hessian='newton')
        setup.assert_not_called()
        tied = t3.TuckerTensorTrain(*bsharing.t3_tie_tucker_factors(restart.data, (0, 0, 1)))
        utied = ut3.UniformTuckerTensorTrain.from_t3(tied)
        for geom, x in ((sg.shared_manifold((0, 0, 1)), tied), (sg.shared_corewise((0, 0, 1)), tied),
                        (sg.shared(ut3m.UNIFORM_MANIFOLD, (0, 0, 1)), utied),
                        (sg.shared(ut3m.UNIFORM_COREWISE, (0, 0, 1)), utied)):
            with self.subTest(geometry=repr(geom)):
                _, stats = call(geom, x, hessian='exact')      # (uniform: skip the 'auto' estimate)
                self.assertEqual(stats['history'][0]['hessian'], 'exact')
        with self.assertRaisesRegex(ValueError, 'curvature_damping must be >= 0'):
            call(sg.shared_manifold((0, 0, 1)), tied, hessian='exact', curvature_damping=-1.0)
        with self.assertRaisesRegex(ValueError, 'curvature_damping must be >= 0'):
            call(sg.shared(ut3m.UNIFORM_MANIFOLD, (0, 0, 1)), utied, hessian='exact', curvature_damping=-1.0)
        with self.assertRaisesRegex(ValueError, 'no effect on a corewise geometry'):
            call(sg.shared(ut3m.UNIFORM_COREWISE, (0, 0, 1)), utied, hessian='exact', curvature_damping=1e-6)

    def test_zero_padded_shared_restart_through_the_frontend(self):
        """T22 through the frontend (S7): ``newton_cg(shared_manifold((0, 0, 1)), ..., hessian='exact')`` from a
        zero-padded SHARED restart converges on the gradient test with no rejection, its first row reporting the
        zero group-spectrum level (``lift_sigma_min == 0``), every iterate tied; the run is the backend run on
        its backend twin (``_twin``), bit for bit. |W| = 40 here (the backend T22 uses 60); observed 8
        outer steps, 0 rejected, 31 CG iterations."""
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.shared_geometry as sg
        oracle, batch, restart = _shared_restart_problem(40)
        geom = sg.shared_manifold((0, 0, 1))
        xf, sf = topt.newton_cg(geom, oracle, batch, restart, max_newton=40, hessian='exact')
        h = sf['history']
        self.assertEqual(sf['stop_reason'], 'gradient')
        self.assertEqual(sum(row['tr_accepted'] is False for row in h), 0)
        self.assertLessEqual(sf['newton_steps'], 12)
        self.assertEqual(h[0]['lift_sigma_min'], 0.0)
        self.assertTrue(bool(np.all(xf.has_shared_tucker_factors((0, 0, 1)))))
        setup = _twin(geom, oracle, batch, restart)
        xb, sb = bopt.newton_cg(setup.geom, oracle, setup.batch, setup.x0_cores, max_newton=40, hessian='exact')
        self.assertEqual(sf['losses'], sb['losses'])
        self.assertTrue(_histories_equal(sf['history'], sb['history']))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[0] + xf.data[1], xb[0] + xb[1])))

    def test_zero_padded_shared_restart_through_the_uniform_frontend(self):
        """T22 through the frontend on the SHARED UNIFORM layer (S9): ``newton_cg(shared(UNIFORM_MANIFOLD, (0, 0, 1)),
        ..., hessian='exact')`` from the zero-padded shared restart (as a uniform point) converges on the gradient test
        with no more than two rejections, its first row reporting the zero group-spectrum level (the real level, not a
        padded one), every iterate tied; the run is the backend uniform run on its backend twin (``_twin``), bit
        for bit. Its trajectory is not the ragged run's -- a padded restart is rank-deficient, where the two layers'
        frames legitimately differ (``docs/uniform_equivalence_contract.md``) -- but it reaches the same minimizer.
        |W| = 40 (the ragged frontend twin above); observed 11 outer steps, 0 rejected, 44 CG iterations (ragged
        8 / 0 / 31), the same final objective."""
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.shared_geometry as sg
        oracle, batch, restart = _shared_restart_problem(40)
        urestart = ut3.UniformTuckerTensorTrain.from_t3(restart)
        geom = sg.shared(ut3m.UNIFORM_MANIFOLD, (0, 0, 1))
        xf, sf = topt.newton_cg(geom, oracle, batch, urestart, max_newton=40, hessian='exact')
        h = sf['history']
        self.assertEqual(sf['stop_reason'], 'gradient')
        self.assertLessEqual(sum(row['tr_accepted'] is False for row in h), 2)
        self.assertLessEqual(sf['newton_steps'], 14)
        self.assertLess(h[0]['lift_sigma_min'], 1e-12)
        self.assertTrue(bool(np.all(xf.has_shared_tucker_factors((0, 0, 1)))))
        setup = _twin(geom, oracle, batch, urestart)
        xb, sb = bopt.newton_cg(setup.geom, oracle, setup.batch, setup.x0_cores, max_newton=40, hessian='exact')
        self.assertEqual(sf['losses'], sb['losses'])
        self.assertTrue(_histories_equal(sf['history'], sb['history']))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[:2], xb)))
        _, sr = topt.newton_cg(sg.shared_manifold((0, 0, 1)), oracle, batch, restart,
                               max_newton=40, hessian='exact')
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
    and is the backend run on its backend twin (``_twin``) bit for bit -- ragged and uniform."""

    def test_probe(self):
        self._converges_and_is_the_backend_run('probe')

    def test_apply(self):
        self._converges_and_is_the_backend_run('apply')

    def test_entries(self):
        self._converges_and_is_the_backend_run('entries')

    def _converges_and_is_the_backend_run(self, kind):
        import t3toolbox.backend.optimizers as bopt
        sample, data, x0 = _plain_fit_problem(kind)
        weight = [0.5, 1.0, 2.0] if kind == 'probe' else None
        oracle, batch = F.old_vocabulary(kind, sample, data, x0.shape, weight=weight)
        with self.subTest(kind=kind, layer='ragged'):
            xf, ex = topt.newton_cg(t3m.MANIFOLD, oracle, batch, x0, max_newton=60,
                                    hessian='exact')
            _, gn = topt.newton_cg(t3m.MANIFOLD, oracle, batch, x0, max_newton=40)
            self.assertEqual(ex['stop_reason'], 'gradient')
            self.assertLess(ex['newton_steps'], gn['newton_steps'])     # observed probe 13 / 24, apply 24 / >= 40,
            #                                                           # entries 16 / 36 (exact / Gauss-Newton)
            setup = _twin(t3m.MANIFOLD, oracle, batch, x0)
            xb, sb = bopt.newton_cg(setup.geom, oracle, setup.batch, setup.x0_cores, max_newton=60, hessian='exact')
            self.assertEqual(ex['losses'], sb['losses'])
            self.assertTrue(_histories_equal(ex['history'], sb['history']))
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[0] + xf.data[1], xb[0] + xb[1])))
        with self.subTest(kind=kind, layer='uniform'):
            ux = ut3.UniformTuckerTensorTrain.from_t3(x0)
            xu, uf_ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, oracle, batch, ux, max_newton=8, hessian='exact')
            setup = _twin(ut3m.UNIFORM_MANIFOLD, oracle, batch, ux)
            xb, sb = bopt.newton_cg(setup.geom, oracle, setup.batch, setup.x0_cores, max_newton=8, hessian='exact')
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
        clean = np.asarray(dd.forward(target, 'entries', first, pp, order))
    else:
        first = [rng.standard_normal((W, N)) for N in shape]
        clean = np.asarray(dd.forward(target, 'apply', first, pp, order))
    data = clean + 0.05 * np.linalg.norm(clean) / clean.size ** 0.5 * rng.standard_normal(clean.shape)
    return (first, pp), data, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))


class TestFrontendExactNewtonDerivativeKinds(unittest.TestCase):
    """S12 through the frontend: ``optimizers.newton_cg(geometry, 'apply_derivatives' | 'entries_derivatives', ...,
    hessian='exact')`` (order 2, a per-order weight) converges on the gradient test in fewer outer steps than
    Gauss-Newton on a misspecified fit, to the same minimizer, and is the backend run on its backend twin
    (``_twin``) bit for bit -- ragged and uniform."""

    def test_apply_derivatives(self):
        self._converges_and_is_the_backend_run('apply_derivatives')

    def test_entries_derivatives(self):
        self._converges_and_is_the_backend_run('entries_derivatives')

    def _converges_and_is_the_backend_run(self, kind):
        import t3toolbox.backend.optimizers as bopt
        sample, data, x0 = _derivative_fit_problem(kind)
        weight = [1.0, 0.5, 0.25]
        oracle, batch = F.old_vocabulary(kind, sample, data, x0.shape, order=2, weight=weight)
        with self.subTest(kind=kind, layer='ragged'):
            xf, ex = topt.newton_cg(t3m.MANIFOLD, oracle, batch, x0, max_newton=60,
                                    hessian='exact')
            _, gn = topt.newton_cg(t3m.MANIFOLD, oracle, batch, x0, max_newton=60)
            self.assertEqual(ex['stop_reason'], 'gradient')
            self.assertEqual(gn['stop_reason'], 'gradient')
            self.assertLess(ex['newton_steps'], gn['newton_steps'])     # observed apply 18 / 34, entries 24 / 39
            self.assertLess(abs(ex['losses'][-1] - gn['losses'][-1]), 1e-9 * gn['losses'][-1])   # the same minimum
            setup = _twin(t3m.MANIFOLD, oracle, batch, x0)
            xb, sb = bopt.newton_cg(setup.geom, oracle, setup.batch, setup.x0_cores, max_newton=60, hessian='exact')
            self.assertEqual(ex['losses'], sb['losses'])
            self.assertTrue(_histories_equal(ex['history'], sb['history']))
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[0] + xf.data[1], xb[0] + xb[1])))
        with self.subTest(kind=kind, layer='uniform'):
            ux = ut3.UniformTuckerTensorTrain.from_t3(x0)
            xu, uf_ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, oracle, batch, ux, max_newton=8, hessian='exact')
            setup = _twin(ut3m.UNIFORM_MANIFOLD, oracle, batch, ux)
            xb, sb = bopt.newton_cg(setup.geom, oracle, setup.batch, setup.x0_cores, max_newton=8, hessian='exact')
            self.assertEqual(uf_['losses'], sb['losses'])
            self.assertTrue(_histories_equal(uf_['history'], sb['history']))
            self.assertTrue(np.array_equal(xu.data[0], xb[0]) and np.array_equal(xu.data[1], xb[1]))
            self.assertEqual(uf_['history'][0]['hessian'], 'exact')


def _switch_problem():
    """``newton_cg``'s docstring problem through the frontend: probe derivatives of order 1 on (5, 5, 5), a rank-3
    target with 1 % noise, the rank-2 fit warm-started by five Gauss-Newton steps (the switch fires there: the
    backend's T6)."""
    np.random.seed(0)
    shape, rng = (5, 5, 5), np.random.default_rng(0)
    ww, pp = ([rng.standard_normal((40, n)) for n in shape] for _ in range(2))
    target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
    data = [z + 0.01 * rng.standard_normal(z.shape) for z in dd.forward(target, 'probe', ww, pp, 1)]
    oracle, batch = F.old_vocabulary('probe_derivatives', (ww, pp), data, shape, order=1)
    x0, _ = topt.newton_cg(t3m.MANIFOLD, oracle, batch, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)),
                           max_newton=5)
    return oracle, batch, x0


class TestFrontendSwitchNewton(unittest.TestCase):
    """The switch plan's T7 (frontend) and T11: ``optimizers.newton_cg(hessian='switch')`` -- the four ``switch_*``
    keywords after ``timing`` as on the backend, each rule raising at the call before the problem is built, each
    keyword forwarded as given and only when set, and a
    frontend switch run that is the backend run bit for bit, ragged and uniform, with the display's ``sw`` segment."""

    NAMES = ('switch_tau', 'switch_k', 'switch_cg_forcing_power', 'switch_track')

    def setUp(self):
        np.random.seed(7)
        x_true = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        self.ww = tuple(np.random.randn(12, n) for n in x_true.shape)
        self.b = x_true.apply(self.ww)
        self.x0 = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        self.oracle, self.batch = F.old_vocabulary('apply', self.ww, self.b, x_true.shape)

    def test_the_keywords(self):
        """The four keywords close the signature after ``timing``, keyword-only, default ``None``, in the backend's
        order."""
        import inspect
        import t3toolbox.backend.optimizers as bopt
        front = inspect.signature(topt.newton_cg).parameters
        back = inspect.signature(bopt.newton_cg).parameters
        self.assertEqual(list(front)[-5:], ['timing', *self.NAMES])
        self.assertEqual(list(back)[-5:], ['timing', *self.NAMES])
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assertIs(front[name].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIsNone(front[name].default)

    def test_options_raise_before_setup(self):
        """T7 (frontend): each rule raises with the backend's message before ``optimizers._setup`` is called -- a
        ``switch_*`` option without ``hessian='switch'`` (the mode read case-insensitively), each option's type and
        range, an option the run would not read, and an unknown mode, whose message now lists ``'switch'``."""
        import re
        from unittest import mock
        bad = [(dict(switch_tau=0.5), "newton_cg: switch_tau set without hessian='switch'"),
               (dict(hessian='Exact', switch_track='all', switch_k=1),
                "newton_cg: switch_k, switch_track set without hessian='switch'"),
               (dict(switch_cg_forcing_power=1.0), "newton_cg: switch_cg_forcing_power set without hessian='switch'"),
               (dict(hessian='switch', switch_tau=1.0), 'newton_cg: switch_tau must be a number in [0, 1); got 1.0'),
               (dict(hessian='Switch', switch_tau=-0.25), 'newton_cg: switch_tau must be a number in [0, 1)'),
               (dict(hessian='switch', switch_tau=float('nan')), 'newton_cg: switch_tau must be a number in [0, 1)'),
               (dict(hessian='switch', switch_tau=False), 'newton_cg: switch_tau must be a number in [0, 1)'),
               (dict(hessian='switch', switch_k=-1), 'newton_cg: switch_k must be an int >= 0 (not a bool); got -1'),
               (dict(hessian='switch', switch_k=True), 'newton_cg: switch_k must be an int >= 0 (not a bool)'),
               (dict(hessian='switch', switch_k=2.0), 'newton_cg: switch_k must be an int >= 0 (not a bool)'),
               (dict(hessian='switch', switch_track='every'),
                "newton_cg: switch_track must be None, 'countable' or 'all' (case-insensitive); got 'every'"),
               (dict(hessian='switch', switch_track=1), "newton_cg: switch_track must be None, 'countable' or 'all'"),
               (dict(hessian='switch', switch_k=0, switch_tau=0.5),
                'newton_cg: switch_tau set with switch_k=0, which tracks nothing (the run is exact from iteration 0)'),
               (dict(hessian='switch', switch_k=0, switch_track='all'), 'newton_cg: switch_track set with switch_k=0'),
               (dict(hessian='switch', switch_tau=0.0, switch_k=3),
                'newton_cg: switch_k set with switch_tau=0, which never switches (the run tracks only)'),
               (dict(hessian='switch', switch_tau=0.0, switch_cg_forcing_power=1.0),
                'newton_cg: switch_cg_forcing_power set with switch_tau=0'),
               (dict(hessian='swich'),
                "newton_cg: hessian must be 'gauss_newton', 'exact' or 'switch' (case-insensitive); got 'swich'")]
        for kw, message in bad:
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                with mock.patch.object(topt, '_setup') as setup:
                    with self.assertRaisesRegex(ValueError, re.escape(message)):
                        topt.newton_cg(t3m.MANIFOLD, self.oracle, self.batch, self.x0, **kw)
                setup.assert_not_called()

    def test_forwarded_as_given_and_only_when_set(self):
        """T11: with ``hessian='Switch'`` and the four at non-default values, each reaches the backend as given (none
        dropped, swapped or converted), beside the mode and the solver options; each alone is forwarded alone, a zero
        included; unset, none is forwarded, in every mode."""
        from unittest import mock
        import t3toolbox.backend.optimizers as bopt
        stub = lambda geom, oracle, batch, x0, reg, **kw: (x0, {})
        opts = dict(hessian='Switch', switch_tau=0.125, switch_k=3, switch_cg_forcing_power=0.875, switch_track='All',
                    step_control='line_search', cg_forcing_power=0.75)
        with mock.patch.object(bopt, 'newton_cg', side_effect=stub) as spy:
            topt.newton_cg(t3m.MANIFOLD, self.oracle, self.batch, self.x0, **opts)
        self.assertEqual({k: spy.call_args.kwargs[k] for k in opts}, opts)
        for name, value in (('switch_tau', np.float64(0.0)), ('switch_k', np.int64(0)),
                            ('switch_cg_forcing_power', np.float32(0.875)), ('switch_track', 'all')):
            with self.subTest(alone=name):
                with mock.patch.object(bopt, 'newton_cg', side_effect=stub) as spy:
                    topt.newton_cg(t3m.MANIFOLD, self.oracle, self.batch, self.x0, hessian='switch', **{name: value})
                forwarded = {k: v for k, v in spy.call_args.kwargs.items() if k.startswith('switch_')}
                self.assertEqual(forwarded, {name: value})
                self.assertIs(type(forwarded[name]), type(value))
        for mode in ('switch', 'exact', 'gauss_newton'):
            with self.subTest(unset=mode):
                with mock.patch.object(bopt, 'newton_cg', side_effect=stub) as spy:
                    topt.newton_cg(t3m.MANIFOLD, self.oracle, self.batch, self.x0, hessian=mode)
                self.assertFalse([k for k in spy.call_args.kwargs if k.startswith('switch_')])
                self.assertEqual(spy.call_args.kwargs['hessian'], mode)

    def test_a_switch_run_is_the_backend_run(self):
        """T11: a frontend switch run, ragged (the defaults) and uniform (``switch_track='ALL'``), is the backend run on
        its backend twin (``_twin``) bit for bit -- the losses, every history field (the switch fields
        included), the cores -- with the switch's ``stats`` keys (``step_control`` resolved to the trust region,
        ``switch_track`` lower-cased); the switch fires, and ``verbose=True`` prints the ``sw`` segment on each
        Gauss-Newton-phase row, the mark on one."""
        import contextlib
        import io
        import t3toolbox.backend.optimizers as bopt
        oracle, batch, x0 = _switch_problem()
        for layer, geometry, x, kw, track in (
                ('ragged', t3m.MANIFOLD, x0, {}, 'countable'),
                ('uniform', ut3m.UNIFORM_MANIFOLD, ut3.UniformTuckerTensorTrain.from_t3(x0), dict(switch_track='ALL'),
                 'all')):
            with self.subTest(layer=layer):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    xf, sf = topt.newton_cg(geometry, oracle, batch, x, max_newton=12,
                                            hessian='switch', verbose=True, **kw)
                setup = _twin(geometry, oracle, batch, x)
                xb, sb = bopt.newton_cg(setup.geom, oracle, setup.batch, setup.x0_cores, max_newton=12,
                                        hessian='switch', **kw)
                self.assertEqual(sf['losses'], sb['losses'])
                self.assertTrue(_histories_equal(sf['history'], sb['history']))
                self.assertIn('switch_fired', sf['history'][0])
                xf_arrays = xf.data[0] + xf.data[1] if layer == 'ragged' else xf.data[:2]
                xb_arrays = xb[0] + xb[1] if layer == 'ragged' else xb
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf_arrays, xb_arrays)))
                self.assertEqual(list(sf), list(sb) + ['diagnostics'])
                self.assertEqual((sf['hessian'], sf['switch_track']), ('switch', track))
                self.assertEqual(sf['history'][0]['step_control'], 'trust_region')
                j = sf['switched_at']
                self.assertIsNotNone(j)
                self.assertEqual([row['hessian'] for row in sf['history'][j - 1:j + 1]], ['gauss_newton', 'exact'])
                text = out.getvalue()
                self.assertEqual(text.count(' | sw '), j)                  # the j Gauss-Newton-phase step rows
                self.assertEqual(text.count(' ⇒ exact'), 1)
                fired = [line for line in text.split('\n') if line.startswith('iter %2d |' % (j - 1))]
                self.assertTrue(fired[0].endswith(' ⇒ exact'), fired)                # on the row that switched


class TestReview4aChecks(unittest.TestCase):
    """P16 part 4a, review round 1: the frontend's checks of a given frame (``local_model(frame=)`` and ``newton_cg``'s
    ``start_frame``, one helper), the one batch-preparation helper, the old plain-probe weight of length 1, the
    shared ``retract`` reusing the build's companion, and the test shim's one-``y`` rule."""

    def setUp(self):
        from t3toolbox.fitting import LeastSquaresOracle, LeastSquaresBatch
        np.random.seed(11)
        self.x = t3.TuckerTensorTrain.randn((4, 5, 4), (2, 2, 2), (1, 2, 2, 1))
        self.ww = [np.random.randn(20, n) for n in self.x.shape]
        self.oracle = LeastSquaresOracle('apply', order=())
        self.batch = LeastSquaresBatch(self.ww, np.random.randn(20))

    def test_a_float32_frame_the_library_built_passes_and_a_perturbed_one_raises(self):
        """Orthonormality is checked at ``safety.effective_rtol`` (as consistency is), so the float32 frame the
        library builds for a jax point passes; one with its down cores scaled by 1 % raises."""
        if not jax_available:
            self.skipTest('jax not installed')
        import t3toolbox.fitting as fitting
        import t3toolbox.frame_variations_format as bvf
        import jax.numpy as jnp
        f32 = lambda cores: tuple(jnp.asarray(c, dtype=jnp.float32) for c in cores)
        xj = t3.TuckerTensorTrain(f32(self.x.data[0]), f32(self.x.data[1]))   # float32 whatever jax's x64 flag
        frame = t3m.MANIFOLD.frame(xj)
        self.assertEqual(str(frame.data[0][0].dtype), 'float32')
        self.assertGreater(float(np.max(np.asarray(frame.orthogonality_residual))), 1e-9)   # fails a fixed 1e-9
        model = fitting.local_model(t3m.MANIFOLD, self.oracle, self.batch, xj, frame=frame)
        self.assertIs(model.backend.frame[0], frame.data[0])                                 # the caller's frame
        D, U, L, R = frame.data
        scaled = bvf.T3Frame(tuple(d * 1.01 for d in D), U, L, R)
        with self.assertRaisesRegex(ValueError, 'not orthonormal'):
            fitting.local_model(t3m.MANIFOLD, self.oracle, self.batch, xj, frame=scaled)

    def test_frame_on_a_corewise_geometry_raises_structurally(self):
        """``frame=`` is for a manifold geometry: on a corewise one the frame is the point. Structural: it raises in
        unsafe mode too, ragged, uniform and shared."""
        import t3toolbox.fitting as fitting
        import t3toolbox.safety as safety
        import t3toolbox.shared_geometry as sg
        ux = ut3.UniformTuckerTensorTrain.from_t3(self.x)
        for geometry, x in ((t3m.COREWISE, self.x), (ut3m.UNIFORM_COREWISE, ux),
                            (sg.shared_corewise((0, 1, 0)), self.x.share((0, 1, 0)))):
            with self.subTest(geometry=type(geometry).__name__, layer=type(x).__name__):
                frame = geometry.frame(x)
                for unsafe in (False, True):
                    with self.assertRaisesRegex(ValueError, 'frame= needs a manifold geometry'):
                        if unsafe:
                            with safety.unsafe():
                                fitting.local_model(geometry, self.oracle, self.batch, x, frame=frame)
                        else:
                            fitting.local_model(geometry, self.oracle, self.batch, x, frame=frame)

    def test_a_shared_geometrys_given_frame_must_be_tied(self):
        """On a shared geometry a given frame's down cores must be tied (safe mode; ``local_model(frame=)`` and
        ``newton_cg(start_frame=)``, ragged and uniform); the frame of the tied point passes; ``safety.unsafe()``
        skips the check."""
        import t3toolbox.fitting as fitting
        import t3toolbox.safety as safety
        import t3toolbox.shared_geometry as sg
        sharing = (0, 1, 0)
        tied = self.x.share(sharing)
        ux, utied = (ut3.UniformTuckerTensorTrain.from_t3(a) for a in (self.x, tied))
        for base, untied, good in ((t3m.MANIFOLD, self.x, tied), (ut3m.UNIFORM_MANIFOLD, ux, utied)):
            geometry = sg.shared(base, sharing)
            calls = (('local_model', lambda x, f: fitting.local_model(geometry, self.oracle, self.batch, x, frame=f)),
                     ('newton_cg', lambda x, f: topt.newton_cg(geometry, self.oracle, self.batch, x, start_frame=f,
                                                               max_newton=1)))
            for name, call in calls:
                with self.subTest(layer=type(untied).__name__, call=name):
                    self.assertFalse(bool(np.all(untied.has_shared_tucker_factors(sharing))))
                    with self.assertRaisesRegex(ValueError, 'not tied'):
                        call(untied, base.frame(untied))
                    with safety.unsafe():
                        call(untied, base.frame(untied))
                    call(good, base.frame(good))

    def test_the_batch_is_checked_unless_packed_for_a_uniform_point(self):
        """``prepare_batch`` checks every batch of a ragged point (an array sample included) and a ragged batch of a
        uniform one; only a batch already packed for a uniform point skips the (ragged-form) checker."""
        from unittest import mock
        import t3toolbox.fitting as fitting
        import t3toolbox.backend.uniform_fitting as ufit
        from t3toolbox.fitting import LeastSquaresBatch, LeastSquaresOracle
        ux = ut3.UniformTuckerTensorTrain.from_t3(self.x)
        ww4 = [np.random.randn(20, 4) for _ in range(3)]
        x4 = t3.TuckerTensorTrain.randn((4, 4, 4), (2, 2, 2), (1, 2, 2, 1))
        array_sample = LeastSquaresBatch(np.stack(ww4), np.zeros(20))     # an ndarray sample, for a ragged point
        packed = ufit.pack_batch('apply', (), self.batch, ux.N)
        for x, batch, checked in ((self.x, self.batch, True), (x4, array_sample, True), (ux, self.batch, True),
                                  (ux, packed, False)):
            with self.subTest(layer=type(x).__name__, sample=type(batch.sample).__name__):
                with mock.patch.object(LeastSquaresOracle, 'check_batch') as check:
                    out = fitting.prepare_batch(self.oracle, batch, x)
                self.assertEqual(check.called, checked)
                if isinstance(x, ut3.UniformTuckerTensorTrain):
                    self.assertFalse(isinstance(out.sample, (tuple, list)))   # packed
        with mock.patch.object(LeastSquaresOracle, 'check_batch') as check:
            fitting.prepare_batch(self.oracle, self.batch, ux, check=False)      # a draw's batch: packed only
        check.assert_not_called()

    def test_the_old_plain_probe_weight_of_length_one_broadcasts(self):
        """2026.5.0 read a plain-probe weight of length 1 as one weight for every mode; the translation does too."""
        yp = self.x.probe(self.ww)
        x0 = t3.TuckerTensorTrain.randn(self.x.shape, (2, 2, 2), (1, 2, 2, 1))
        _, one = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, yp, x0, weight=[2.0], max_newton=3)
        _, full = topt.newton_cg(t3m.MANIFOLD, 'probe', self.ww, yp, x0, weight=[2.0, 2.0, 2.0], max_newton=3)
        self.assertEqual(one['losses'], full['losses'])
        oracle, batch = topt._old_fit('probe', self.ww, yp, self.x.shape, None, [2.0], 'test')
        self.assertTrue(np.array_equal(batch.weight, np.full(3, 2.0)))

    def test_shared_retract_reuses_the_builds_companion(self):
        """The frontend model's ``retract`` on a shared manifold geometry is the geometry's retraction with the
        companion the build computed (``backend.geom_aux``), bit for bit the one recomputed from the frame."""
        import t3toolbox.fitting as fitting
        import t3toolbox.shared_geometry as sg
        from unittest import mock
        sharing = (0, 1, 0)
        tied = self.x.share(sharing)
        for base, x in ((t3m.MANIFOLD, tied), (ut3m.UNIFORM_MANIFOLD, ut3.UniformTuckerTensorTrain.from_t3(tied))):
            with self.subTest(layer=type(x).__name__):
                geometry = sg.shared(base, sharing)
                model = fitting.local_model(geometry, self.oracle, self.batch, x)
                p = model.gradient
                with mock.patch.object(sg.SharedGeometry, 'precompute', wraps=geometry.precompute) as spy:
                    moved = model.retract(p)
                spy.assert_not_called()
                again = geometry.retract(p, shared_data=geometry.precompute(model.frame))
                self.assertTrue(all(np.array_equal(a, b) for a, b in
                                    zip(cw_leaves(moved.data[:2]), cw_leaves(again.data[:2]))))

    def test_the_old_model_shim_needs_one_y_over_the_stack(self):
        """``_fits.old_model`` on a stacked point: a residual ``z(x_c) - y`` with one ``y`` builds the model (its
        residual is the stated one); a residual with a different ``y`` per element raises."""
        x = t3.TuckerTensorTrain.randn((4, 5, 4), (2, 2, 2), (1, 2, 2, 1), stack_shape=(2,))
        y = np.random.randn(20)
        z = np.asarray(x.apply(self.ww))                                  # W + C = (20, 2)
        model = F.old_model('apply', t3m.MANIFOLD, x, self.ww, z - y[:, None])
        self.assertTrue(np.allclose(np.asarray(model.residual), z - y[:, None], rtol=1e-10, atol=1e-10))
        with self.assertRaisesRegex(ValueError, 'one y for every element'):
            F.old_model('apply', t3m.MANIFOLD, x, self.ww, np.random.randn(20, 2))


def cw_leaves(tree):
    """The arrays of a nested tuple of arrays, in order."""
    if isinstance(tree, (tuple, list)):
        return [leaf for item in tree for leaf in cw_leaves(item)]
    return [tree]



# --------------------------------------------------------------------------------------------------
# The translation of the 2026.5.0 call form (P16; deleted with the translation in P17)
# --------------------------------------------------------------------------------------------------
# The fits of the reviewer's frontend side-by-side (P16 part 4a review), built from dense coefficients with no library
# code, so that the numbers below -- PINNED from runs of the pre-P16 library (git 0acf7868) -- are those of the
# 2026.5.0 frontend on the very same arrays. The translation must reproduce them.
_TR_SHAPE = (5, 6, 5)
_TR_W = 40


def _tr_target(seed):
    rng = np.random.default_rng(seed)
    nD, r = (2, 3, 2), (1, 2, 2, 1)
    B = [rng.standard_normal((nD[i], _TR_SHAPE[i])) for i in range(3)]
    G = [rng.standard_normal((r[i], nD[i], r[i + 1])) for i in range(3)]
    return B, G


def _tr_dense(B, G):
    core = np.einsum('aib,bjc,ckd->aijkd', *G)[0, ..., 0]
    return np.einsum('ijk,ix,jy,kz->xyz', core, *B)


def _tr_coeffs_apply(Td, ww, pp, n):
    '''Taylor coefficients in s of <T, (w1 + s p1) x (w2 + s p2) x (w3 + s p3)>, t = 0..n: (n+1, W).'''
    import itertools
    out = np.zeros((n + 1, ww[0].shape[0]))
    for t in range(min(n, 3) + 1):
        for S in itertools.combinations(range(3), t):
            out[t] += np.einsum('abc,wa,wb,wc->w', Td, *[pp[j] if j in S else ww[j] for j in range(3)])
    return out


def _tr_coeffs_probe(Td, ww, pp, n):
    import itertools
    subs = {0: 'abc,wb,wc->wa', 1: 'abc,wa,wc->wb', 2: 'abc,wa,wb->wc'}
    outs = []
    for i in range(3):
        others = [j for j in range(3) if j != i]
        out = np.zeros((n + 1, ww[0].shape[0], _TR_SHAPE[i]))
        for t in range(min(n, 2) + 1):
            for S in itertools.combinations(others, t):
                out[t] += np.einsum(subs[i], Td, *[pp[j] if j in S else ww[j] for j in others])
        outs.append(out)
    return outs


def _tr_problem(kind, n, seed=0):
    '''(sample, data) of an old-form fit: noisy data of a fixed target, derivative data as derivatives (t! times the
    Taylor coefficients).'''
    import math
    rng = np.random.default_rng(100 + seed)
    Td = _tr_dense(*_tr_target(seed))
    ww = [rng.standard_normal((_TR_W, N)) / np.sqrt(N) for N in _TR_SHAPE]
    pp = [rng.standard_normal((_TR_W, N)) / np.sqrt(N) for N in _TR_SHAPE]
    index = np.stack([rng.integers(0, N, _TR_W) for N in _TR_SHAPE])
    noise = lambda a: a + 0.01 * rng.standard_normal(np.shape(a))
    fact = np.array([math.factorial(t) for t in range(n + 1)])
    one_hot = [np.eye(N)[index[i]] for i, N in enumerate(_TR_SHAPE)]
    if kind == 'apply':
        return ww, noise(_tr_coeffs_apply(Td, ww, pp, 0)[0])
    if kind == 'probe':
        return ww, [noise(z[0]) for z in _tr_coeffs_probe(Td, ww, pp, 0)]
    if kind == 'apply_derivatives':
        return (ww, pp), noise(_tr_coeffs_apply(Td, ww, pp, n)) * fact[:, None]
    if kind == 'entries_derivatives':
        return (index, pp), noise(_tr_coeffs_apply(Td, one_hot, pp, n)) * fact[:, None]
    return (ww, pp), [noise(z) * fact[:, None, None] for z in _tr_coeffs_probe(Td, ww, pp, n)]


def _tr_x0(uniform=False):
    B, G = _tr_target(7)
    x = t3.TuckerTensorTrain(tuple(0.3 * b for b in B), tuple(G))
    return ut3.UniformTuckerTensorTrain.from_t3(x) if uniform else x


def _tr_draw(sample, data, kind, b):
    '''An old custom draw ``draw(rng) -> (sample_B, data_B)``, the rng calls of the pinned runs.'''
    deriv = kind.endswith('_derivatives')

    def draw(rng):
        idx = rng.choice(_TR_W, size=b, replace=False)
        if kind.startswith('entries'):
            s = (sample[0][:, idx], [p[idx] for p in sample[1]]) if deriv else sample[:, idx]
        else:
            s = ([w[idx] for w in sample[0]], [p[idx] for p in sample[1]]) if deriv else [w[idx] for w in sample]
        if kind.startswith('probe'):
            y = [a[:, idx] for a in data] if deriv else [a[idx] for a in data]
        else:
            y = data[:, idx] if deriv else data[idx]
        return s, y
    return draw


class TestTheTranslation(unittest.TestCase):
    """The old call form ``(geometry, kind, sample, data, x0, order=, weight=, ...)`` through the translation gives the
    numbers of the pre-P16 library (2026.5.0) on the same arrays: the last loss and the norm of the result, pinned from
    runs of git 0acf7868 (the reviewer's ``fe_sbs.py``; the new tree agreed to 1e-14 there, adam to 9e-8 through its
    ``eps``). Covered: every old weight form (probe per mode on noisy data, a weight of length 1, extra rows; a bare
    vector on a derivative kind at ``d == n + 1``, read per order; the full matrix, a column, a row), the derivative
    kinds at order 2 (the ``t!`` conversion), ``entries_derivatives`` (the apply kind at the one-hot jet), order 0 as
    ``(0,)``, positional ``order`` and ``weight``, ``val_sample`` / ``val_data`` (the verbose records), a custom draw
    with a regularizer (the minibatch scale) and the positional and keyword ``batch``, ``chunk_size`` dropped, and the
    uniform layer."""

    REG = 0.05
    PINNED = {      # (the last loss (None: not recorded), the norm of the result's dense tensor), 2026.5.0
        'A_probe_permode': (101.81680882219945, 24.482585294446604),
        'B_probe_len1': (43.93344565502416, 25.400570486387807),
        'B2_probe_extra_rows': (101.81680882219945, 24.482585294446604),
        'C_applyd2_bare3': (3.3419818610303746, 24.010695259182732),
        'D_probed2_bare3': (79.22746946664677, 25.37224588298701),
        'E_probed1_matrix': (10.1208247472897, 25.124930941667223),
        'G_probed1_col': (3.6156203731947434, 25.397063355379526),
        'G2_probed1_row': (42.02938073833289, 25.45188266628661),
        'H_entriesd1': (15.629787689430692, 23.433122909494486),
        'I_probed0': (101.8168088221995, 24.482585294446626),
        'I2_applyd0': (0.8951524913891846, 21.462725076750125),
        'P_applyd1_positional_exact': (1291.4888004039535, 0.17385824518500023),
        'J_val_verbose': (374.76055081428274, 24.82345446133301),
        'K_mcsgd_probed1_draw_reg': (None, 25.190918541457627),
        'L_mcsgd_apply_flat_kwbatch_reg': (None, 12.810876230378927),
        'M_adam_cw_entriesd_draw_reg': (None, 0.06933936763393679),
        'N_gd_apply_chunk': (12.532266293638418, 11.876981899733108),
        'O_uniform_probed1_matrix': (10.120824747290046, 25.124930941667206),
        'O2_uniform_mcsgd_draw_reg': (None, 24.335923870026893),
    }
    J_TRAIN = [[0.1852343569440412, 0.17971357316154093], [0.19511138094346941, 0.2001638232247505],
               [0.1750628675113875, 0.13378721155165466]]
    J_VAL = [[2.7044717546854415, 2.549557249192397], [1.8541105485577671, 2.4509470324235196],
             [2.39460151239039, 3.2433890554602254]]

    def _check(self, name, result, rtol=1e-8):
        x, stats = result
        loss, norm = self.PINNED[name]
        if loss is not None:
            self.assertLess(abs(stats['losses'][-1] - loss), rtol * abs(loss), name)
        got = float(np.linalg.norm(np.asarray(x.to_dense())))
        self.assertLess(abs(got - norm), rtol * norm, name)

    def test_newton_cg_weight_forms_and_kinds(self):
        nc = dict(max_newton=6)
        cases = [('A_probe_permode', 'probe', None, [1., 2., .5]), ('B_probe_len1', 'probe', None, [2.]),
                 ('B2_probe_extra_rows', 'probe', None, [1., 2., .5, 9.]),
                 ('C_applyd2_bare3', 'apply_derivatives', 2, [1., .5, .25]),
                 ('D_probed2_bare3', 'probe_derivatives', 2, [1., .5, .25]),
                 ('E_probed1_matrix', 'probe_derivatives', 1, [[1., .5], [2., .3], [.7, 1.1]]),
                 ('G_probed1_col', 'probe_derivatives', 1, [[1.], [2.], [.7]]),
                 ('G2_probed1_row', 'probe_derivatives', 1, [[1., 3.]]),
                 ('H_entriesd1', 'entries_derivatives', 1, [1., 2.]),
                 ('I_probed0', 'probe_derivatives', 0, [[1.], [2.], [.5]]),
                 ('I2_applyd0', 'apply_derivatives', 0, None)]
        for name, kind, order, weight in cases:
            with self.subTest(case=name):
                sample, data = _tr_problem(kind, order or 0)
                kw = {} if order is None else dict(order=order)
                self._check(name, topt.newton_cg(t3m.MANIFOLD, kind, sample, data, _tr_x0(), weight=weight, **kw,
                                                 **nc))
        sample, data = _tr_problem('apply_derivatives', 1)                  # positional order and weight; exact
        self._check('P_applyd1_positional_exact', topt.newton_cg(t3m.MANIFOLD, 'apply_derivatives', sample, data,
                                                                 _tr_x0(), 1, [1., 2.], hessian='exact', max_newton=5))
        sample, data = _tr_problem('probe_derivatives', 1)                  # the uniform layer
        self._check('O_uniform_probed1_matrix', topt.newton_cg(
            ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', sample, data, _tr_x0(uniform=True), order=1,
            weight=[[1., .5], [2., .3], [.7, 1.1]], max_newton=6))

    def test_validation_records_and_both_or_neither(self):
        import contextlib
        import io
        sample, data = _tr_problem('probe_derivatives', 1, seed=0)
        vsample, vdata = _tr_problem('probe_derivatives', 1, seed=1)
        with contextlib.redirect_stdout(io.StringIO()):
            x, stats = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', sample, data, _tr_x0(), order=1,
                                      weight=[1., 2.], verbose=True, val_sample=vsample, val_data=vdata, max_newton=4)
        self._check('J_val_verbose', (x, stats))
        last = stats['diagnostics'][-1]
        self.assertTrue(np.allclose(last['train_err'], self.J_TRAIN, rtol=1e-8, atol=0))
        self.assertTrue(np.allclose(last['val_err'], self.J_VAL, rtol=1e-8, atol=0))
        for kw in (dict(val_sample=vsample), dict(val_data=vdata)):
            with self.subTest(given=list(kw)[0]):
                with self.assertRaisesRegex(ValueError, 'pass val_sample and val_data together, or neither'):
                    topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', sample, data, _tr_x0(), order=1, max_newton=2,
                                   **kw)

    def test_stochastic_draws_the_scale_and_the_batch_argument(self):
        reg = topt.IdentityRegularizer(self.REG)
        sample, data = _tr_problem('probe_derivatives', 1)
        self._check('K_mcsgd_probed1_draw_reg', topt.mc_sgd(           # a custom draw, a regularizer, positional batch
            t3m.MANIFOLD, 'probe_derivatives', sample, data, _tr_x0(), np.random.default_rng(3), 15, order=1,
            weight=[1., 2.], draw=_tr_draw(sample, data, 'probe_derivatives', 15), regularizer=reg, max_iter=8))
        self._check('O2_uniform_mcsgd_draw_reg', topt.mc_sgd(           # the same on the uniform layer
            ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', sample, data, _tr_x0(uniform=True), np.random.default_rng(3),
            15, order=1, draw=_tr_draw(sample, data, 'probe_derivatives', 15), regularizer=reg, max_iter=6))
        sa, ya = _tr_problem('apply', 0)
        self._check('L_mcsgd_apply_flat_kwbatch_reg', topt.mc_sgd(     # the flat draw, batch= by keyword
            t3m.MANIFOLD, 'apply', sa, ya, _tr_x0(), np.random.default_rng(4), batch=20, regularizer=reg, max_iter=8))
        se, ye = _tr_problem('entries_derivatives', 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._check('M_adam_cw_entriesd_draw_reg', topt.adam(      # Adam: the scale differs only through eps
                t3m.COREWISE, 'entries_derivatives', se, ye, _tr_x0(), np.random.default_rng(5), 12, order=1,
                draw=_tr_draw(se, ye, 'entries_derivatives', 12), regularizer=reg, max_iter=8, lr=1e-2), rtol=1e-6)

    def test_chunk_size_is_accepted_and_dropped(self):
        sa, ya = _tr_problem('apply', 0)
        self._check('N_gd_apply_chunk', topt.gradient_descent(t3m.MANIFOLD, 'apply', sa, ya, _tr_x0(), chunk_size=None,
                                                              n_iter=5))
        _, a = topt.newton_cg(t3m.MANIFOLD, 'apply', sa, ya, _tr_x0(), chunk_size=7, max_newton=2)
        _, b = topt.newton_cg(t3m.MANIFOLD, 'apply', sa, ya, _tr_x0(), max_newton=2)
        self.assertEqual(a['losses'], b['losses'])

    def test_each_old_form_is_the_new_form(self):
        """The translation's oracle and batch (``_old_fit``) against the new form built by hand: the jet shape, the
        lines, the Taylor data and the weight times ``t!`` exactly, the old weight forms broadcast to the exact shape,
        and the old entries derivatives as the apply kind at the one-hot jet."""
        from t3toolbox.fitting import affine_jet, index_to_one_hot
        import math
        shape, d = _TR_SHAPE, 3
        fact2 = np.array([math.factorial(t) for t in range(3)], dtype=float)
        s, y = _tr_problem('probe_derivatives', 2)
        for weight, full in (([1., .5, .25], np.tile([1., .5, .25], (d, 1))),            # bare: per order
                             ([[1.], [2.], [.7]], np.tile([[1.], [2.], [.7]], (1, 3))),     # column: per mode
                             ([[1., 3., 2.]], np.tile([1., 3., 2.], (d, 1))),               # row
                             (np.arange(12.).reshape(4, 3) + 1, np.arange(9.).reshape(3, 3) + 1)):   # extra rows
            with self.subTest(weight=np.shape(weight)):
                oracle, batch = topt._old_fit('probe_derivatives', s, y, shape, 2, weight, 'test')
                self.assertEqual((oracle.kind, oracle.order), ('probe', (2,)))
                self.assertTrue(np.array_equal(batch.weight, full * fact2))
                for got, want in zip(batch.y, y):
                    self.assertTrue(np.allclose(got, want / fact2[:, None, None], rtol=1e-15, atol=0))
                for got, want in zip(batch.sample, affine_jet(*s)):
                    self.assertTrue(np.array_equal(got, want))
        s, y = _tr_problem('entries_derivatives', 1)
        oracle, batch = topt._old_fit('ENTRIES_DERIVATIVES', s, y, shape, 1, [1., 2.], 'test')
        self.assertEqual((oracle.kind, oracle.order), ('apply', (1,)))
        for got, want in zip(batch.sample, affine_jet(index_to_one_hot(s[0], shape), s[1])):
            self.assertTrue(np.array_equal(got, want))
        self.assertTrue(np.array_equal(batch.weight, [1., 2.]))
        s, y = _tr_problem('probe', 0)
        oracle, batch = topt._old_fit('probe', s, y, shape, None, [1., 2., .5, 9.], 'test')   # extra rows dropped
        self.assertEqual((oracle.kind, oracle.order), ('probe', ()))
        self.assertTrue(np.array_equal(batch.weight, [1., 2., .5]))
        with self.assertRaisesRegex(ValueError, "takes no residual weight"):
            topt._old_fit('apply', *_tr_problem('apply', 0), shape, None, [1.], 'test')
        with self.assertRaisesRegex(ValueError, 'mode dimension'):
            topt._old_fit('probe', s, y, shape, None, [1., 2.], 'test')                      # fewer rows than modes
        with self.assertRaisesRegex(ValueError, 'requires order='):
            topt._old_fit('probe_derivatives', *_tr_problem('probe_derivatives', 1), shape, None, None, 'test')
        # a custom draw's batch carries 2026.5.0's nominal factor n / min(batch, n), batch above n included
        sample, data = _tr_problem('probe_derivatives', 1)
        for b, scale in ((15, _TR_W / 15), (_TR_W, 1.0), (60, 1.0)):
            with self.subTest(batch=b):
                args, kw = topt._translate('mc_sgd', (t3m.MANIFOLD, 'probe_derivatives', sample, data, _tr_x0(),
                                                      np.random.default_rng(0), b),
                                           dict(order=1, draw=_tr_draw(sample, data, 'probe_derivatives', min(b, _TR_W))))
                self.assertEqual(args[5], b)                                     # batch -> batch_size
                drawn = kw['draw'](np.random.default_rng(1))
                self.assertEqual(drawn.minibatch_scale, scale)



class TestFrontendOnlyAndUserOracles(unittest.TestCase):
    """P16 (the answers of 20:15): a fit from derivative data needs nothing from ``t3toolbox.backend``; and the
    frontend optimizers drive an oracle that is not the least-squares one: one without a draw (``take``) raises in
    ``mc_sgd`` and ``adam`` before any work and runs with a draw, and one without ``f_resolution`` gets the trust
    region's fallback floor ``eps |f|`` (the backend's ``test_the_floor_reads_eps_off_f`` pins its value)."""

    def test_a_frontend_only_fit_from_derivative_data(self):
        """Derivatives of a rank-(2, 2, 2) tensor along lines, up to order 2, are fitted from the zero tensor with the
        frontend alone (the oracle, the batch, ``affine_jet`` and ``prepare_derivative_data`` from
        ``t3toolbox.fitting``; ``t3toolbox.optimizers.newton_cg``); the source of this test names no backend
        module."""
        import inspect
        import math
        import t3toolbox.fitting as fitting
        import t3toolbox.optimizers as optimizers
        self.assertNotIn('backend', inspect.getsource(TestFrontendOnlyAndUserOracles._fit_body))
        x, target = self._fit_body(fitting, optimizers, math)
        err = float(np.linalg.norm(x.to_dense() - target.to_dense()) / np.linalg.norm(target.to_dense()))
        self.assertLess(err, 1e-8)

    @staticmethod
    def _fit_body(fitting, optimizers, math):
        rng = np.random.default_rng(0)
        np.random.seed(0)
        shape, order = (4, 5, 6), 2
        target = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
        ww, pp = ([rng.standard_normal((50, N)) / math.sqrt(N) for N in shape] for _ in range(2))
        line = fitting.affine_jet(ww, pp)                                    # the lines ww + s pp
        derivatives = fitting.taylor_to_derivatives('probe', (order,), target.probe(line, (order,)))   # the data
        weight = np.tile([1.0, 0.5, 0.25], (len(shape), 1))                  # on the derivatives, per (mode, order)
        y, w, _ = fitting.prepare_derivative_data('probe', (order,), derivatives, weight)
        oracle = fitting.LeastSquaresOracle('probe', order=(order,))
        x0 = t3.TuckerTensorTrain.zeros(shape, (2, 2, 2), (1, 2, 2, 1))
        x, _ = optimizers.newton_cg(t3m.MANIFOLD, oracle, fitting.LeastSquaresBatch(line, y, w), x0, max_newton=30)
        return x, target

    def _apply_fit(self):
        from t3toolbox.fitting import LeastSquaresBatch
        np.random.seed(3)
        x_true = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        ww = tuple(np.random.randn(30, n) for n in x_true.shape)
        x0 = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        return x_true, ww, LeastSquaresBatch(ww, x_true.apply(ww)), x0

    def test_a_user_oracle_without_a_draw(self):
        from unittest import mock
        import t3toolbox.backend.optimizers as bopt
        _, _, batch, x0 = self._apply_fit()
        bare = _NoDrawOracle('apply', ())
        for name, call in (('mc_sgd', lambda: topt.mc_sgd(t3m.MANIFOLD, bare, batch, x0, np.random.default_rng(0), 8,
                                                          max_iter=1)),
                           ('adam', lambda: topt.adam(t3m.COREWISE, bare, batch, x0, np.random.default_rng(0), 8,
                                                      max_iter=1))):
            with self.subTest(optimizer=name):
                with mock.patch.object(bopt, 'local_model', wraps=bopt.local_model) as work:
                    with self.assertRaisesRegex(ValueError, '_NoDrawOracle has no take'):
                        call()
                work.assert_not_called()
        from t3toolbox.fitting import LeastSquaresOracle
        full = LeastSquaresOracle('apply', ())
        draw = lambda rng: full.take(batch, rng.choice(30, size=8, replace=False))
        x, _ = topt.mc_sgd(t3m.MANIFOLD, bare, batch, x0, np.random.default_rng(0), 8, draw=draw, max_iter=2)
        self.assertIsInstance(x, t3.TuckerTensorTrain)

    def test_the_floor_without_f_resolution(self):
        """A trust-region run (exact, a large first radius: trials are rejected, so the floor is read) on an oracle without
        ``f_resolution`` never asks for it, reads the floor ``eps (|f| + |rho|)`` with ``eps`` of ``f``'s dtype, and is
        the backend run on the same oracle (the frontend forwards the oracle)."""
        from unittest import mock
        import t3toolbox.backend.optimizers as bopt
        _, _, batch, x0 = self._apply_fit()
        oracle = _NoResolutionOracle('apply', ())
        asked = []
        real = _NoResolutionOracle.query

        def spy(self_, frame, batch_, want, *args, **kwargs):
            asked.extend(want)
            return real(self_, frame, batch_, want, *args, **kwargs)

        calls = []
        real_floor = bopt._objective_resolution

        def floor(lm, rho):
            value = real_floor(lm, rho)
            calls.append((float(lm.f), float(rho or 0.0), np.dtype(lm.f.dtype), float(value)))
            return value

        with mock.patch.object(_NoResolutionOracle, 'query', spy), mock.patch.object(bopt, '_objective_resolution',
                                                                                      floor):
            x, stats = topt.newton_cg(t3m.MANIFOLD, oracle, batch, x0, hessian='exact', tr_radius0=1e3, max_newton=12)
        self.assertNotIn('f_resolution', asked)
        self.assertGreater(len(calls), 0)                         # the floor was read (a rejection)
        for f, rho, dtype, value in calls:                        # eps (|f| + |rho|), eps of f's dtype
            eps = float(np.finfo(dtype).eps)
            self.assertAlmostEqual(value, eps * (abs(f) + abs(rho)), delta=1e-9 * eps * (abs(f) + abs(rho)))
        twin = _twin(t3m.MANIFOLD, oracle, batch, x0)
        xb, sb = bopt.newton_cg(twin.geom, oracle, twin.batch, twin.x0_cores, hessian='exact', tr_radius0=1e3,
                                max_newton=12)
        self.assertEqual(stats['losses'], sb['losses'])
        self.assertEqual(stats['stop_reason'], sb['stop_reason'])


def _ls_subclass(name, **fields):
    import dataclasses as dc_
    from t3toolbox.fitting import LeastSquaresOracle
    return dc_.dataclass(frozen=True, eq=False)(type(name, (LeastSquaresOracle,), fields))


_NoDrawOracle = _ls_subclass('_NoDrawOracle', optional_outputs=(), take=None, n_measurements=None)
_NoResolutionOracle = _ls_subclass('_NoResolutionOracle',
                                   optional_outputs=('Kp', 'Jp', 'r', 'r_block_sumsq', 'y_block_sumsq'))


if __name__ == "__main__":
    unittest.main()
