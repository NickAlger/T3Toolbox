"""Tests for the uniform-layer fitting seams (backend/uniform_fitting.py) -- optimizers-on-uniform U2.

Correctness gold standard: the backend uniform ``GeometryOps`` factories (raw supercore pairs, masks
closed over) must reproduce the already-verified frontend ``UNIFORM_MANIFOLD`` / ``UNIFORM_COREWISE``
geometry ``.data`` path exactly (same math through the same ``utv_operations`` primitives). The
factory captures the loop-invariant masks at ``x0``'s fixed rank; a second test evaluates the ops at a
DIFFERENT same-rank point to confirm the masks are correctly reused (the property the optimizer loop
relies on). numpy-only (jit dispatch is covered in test_dispatch)."""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.manifold as t3m
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.apply as bapply
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.utv_operations as utv_operations
import t3toolbox.backend.ut3_operations as uops
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # tests/
import _derivative_data as dd       # derivative data from the merged methods (P13b)
import _fits as F                   # a backend fit (geometry, oracle, batch, regularizer) from plain specs (P16)

_STRUCT = ((10, 11, 12), (2, 4, 2), (1, 2, 2, 1))   # (shape, tucker, tt); MINIMAL ranks (tucker capped by
                                                    # the tt bonds), still rank-varied -> real padding

# name -> (frontend geometry singleton, backend GeometryOps factory)
_GEOMS = {
    'manifold': (ut3m.UNIFORM_MANIFOLD, bgeo.UniformManifoldGeometryOps.from_point),
    'corewise': (ut3m.UNIFORM_COREWISE, bgeo.UniformCorewiseGeometryOps.from_point),
}


def _uniform_x(seed):
    np.random.seed(seed)
    return ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*_STRUCT))


def _sc_close(a, b):   # two bare supercore pairs
    return all(np.allclose(np.asarray(ai), np.asarray(bi)) for ai, bi in zip(a, b))


def _frame_close(front_data, back_data):   # UT3Frame.data vs raw frame .data: supercores + shape + masks
    return (all(np.allclose(np.asarray(front_data[i]), np.asarray(back_data[i])) for i in range(4))
            and tuple(front_data[4]) == tuple(back_data[4])
            and all(np.array_equal(fm, bm) for fm, bm in zip(front_data[5], back_data[5])))


class TestUniformGeometryOps(unittest.TestCase):
    def _compare_ops(self, geom_front, ops, x):
        """Backend ops (on bare supercore pairs) vs the frontend geometry (on typed objects), at point x."""
        x_sc = (x.data[0], x.data[1])
        frame_front, frame_back = geom_front.frame(x), ops.frame(x_sc)[0]
        if len(frame_back) == 4:                    # a corewise geometry: its frame is the point (P16); the frontend's
            fd = frame_front.data                   #   is still (B, G, G, G, shape, (tk, tk, tt, tt) masks) until part 4
            self.assertTrue(_sc_close((fd[0], fd[2]), frame_back[:2]) and np.array_equal(fd[2], fd[1])
                            and tuple(fd[4]) == tuple(frame_back[2])
                            and np.array_equal(fd[5][0], frame_back[3][0])
                            and np.array_equal(fd[5][2], frame_back[3][1]), 'frame')
        else:
            self.assertTrue(_frame_close(frame_front.data, frame_back), 'frame')

        v1 = ubv.UT3Variations.randn_like(frame_front)     # ungauged variations at the frame
        v2 = ubv.UT3Variations.randn_like(frame_front)

        proj_front = geom_front.project(ut3m.UT3Tangent(frame_front, v1)).variations.supercores
        self.assertTrue(_sc_close(proj_front, ops.project(frame_back, v1.supercores)), 'project')

        retr_front = geom_front.retract(ut3m.UT3Tangent(frame_front, v1))   # UniformTuckerTensorTrain
        self.assertTrue(_sc_close((retr_front.data[0], retr_front.data[1]),
                                  ops.retract(frame_back, v1.supercores)), 'retract')

        # GeometryOps.inner is the check-free COORDINATE dot == UT3Tangent.corewise_inner (not HS)
        inner_front = float(ut3m.UT3Tangent(frame_front, v1).corewise_inner(ut3m.UT3Tangent(frame_front, v2)))
        self.assertTrue(np.isclose(inner_front, float(ops.inner(v1.supercores, v2.supercores))), 'inner')

    def test_ops_match_frontend(self):
        for name, (geom_front, factory) in _GEOMS.items():
            with self.subTest(geometry=name):
                x = _uniform_x(0)
                self._compare_ops(geom_front, factory(x.data), x)

    def test_masks_loop_invariant_across_points(self):
        # The factory captures the fixed-rank masks at x0; the ops must still match the frontend at a
        # DIFFERENT same-rank point (the frame supercores change every optimizer step; the masks do not).
        for name, (geom_front, factory) in _GEOMS.items():
            with self.subTest(geometry=name):
                x0, x = _uniform_x(0), _uniform_x(1)
                self._compare_ops(geom_front, factory(x0.data), x)


class TestUniformSamplingMirror(unittest.TestCase):
    """U3.5: the user-facing uniform sampling ops MIRROR their vector input's packedness -- ragged in ->
    ragged (len=d tuple) out, packed in -> packed (d,)+... out -- and the two agree via pack/unpack. The
    frontend methods inherit this by delegation."""
    def setUp(self):
        np.random.seed(0)
        self.x = _uniform_x(0)
        self.N = self.x.data[0].shape[-1]

    def _ww(self, W=6):
        return [np.random.randn(W, n) for n in _STRUCT[0]]

    def test_probe_mirrors_and_agrees(self):
        ww_r = self._ww()
        ww_p = uops.pack_vectors(ww_r, self.N)
        out_r = self.x.probe(ww_r)
        out_p = self.x.probe(ww_p)
        self.assertIsInstance(out_r, tuple)             # ragged in -> ragged out
        self.assertNotIsInstance(out_p, (list, tuple))  # packed in -> packed out (one array)
        unpacked = uops.unpack_vectors(out_p, _STRUCT[0])
        self.assertTrue(all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(unpacked, out_r)))

    def test_apply_accepts_both(self):
        ww_r = self._ww()
        ww_p = uops.pack_vectors(ww_r, self.N)
        self.assertTrue(np.allclose(np.asarray(self.x.apply(ww_r)), np.asarray(self.x.apply(ww_p))))

    def test_probe_derivatives_mirrors_and_agrees(self):
        ww_r, pp_r = self._ww(), self._ww()
        ww_p, pp_p = uops.pack_vectors(ww_r, self.N), uops.pack_vectors(pp_r, self.N)
        out_r = dd.forward(self.x, 'probe', ww_r, pp_r, 3)
        out_p = dd.forward(self.x, 'probe', ww_p, pp_p, 3)
        self.assertIsInstance(out_r, tuple)
        self.assertNotIsInstance(out_p, (list, tuple))
        unpacked = uops.unpack_vectors(out_p, _STRUCT[0])
        self.assertTrue(all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(unpacked, out_r)))


class TestUniformProblem(unittest.TestCase):
    """U4: the uniform fit (the batch packed once, pack_batch; the geometry from_point) reproduces the ragged fit's
    LocalModel -- objective, gradient (via <g, p>), and gn_quadratic(p) -- on the equivalent (up to
    gauge) frame, for every sampling kind (plain, and derivative data along lines through the oracle at a jet shape)
    and both geometries. This certifies the batch packing and the LocalModel wiring on the uniform layout."""
    _GEOMS = [('manifold', bgeo.ManifoldGeometryOps(), t3m.MANIFOLD), ('corewise', bgeo.CorewiseGeometryOps(), t3m.COREWISE)]

    def setUp(self):
        import t3toolbox.corewise as cw
        self.cw = cw
        np.random.seed(0)
        self.x0 = t3.TuckerTensorTrain.randn(*_STRUCT)
        self.ux0 = ut3.UniformTuckerTensorTrain.from_t3(self.x0)
        self.W = 15
        # per-geometry trial tangent at x0's frame (ragged .data + its uniform supercores)
        self.trial = {}
        for name, _bg, fg in self._GEOMS:
            p = fg.randn(fg.frame(self.x0))
            if name == 'manifold':
                # The uniform manifold frame is gauge-EQUIVALENT to ragged, not bit-identical (the
                # pad-safe sweep, review S1b), so the same coordinate array means a DIFFERENT tangent
                # at the uniform frame. Transport p through the ambient embedding instead: project
                # p's doubled-rank T3 onto the tangent space at the frame the uniform problem builds
                # (exact -- p already lies in that tangent space, which is gauge-independent).
                uframe = ubv.UT3Frame.from_ut3(self.ux0)
                proj = utv_operations.utv_project_ut3_onto_tangent_space(
                    uframe.data, ut3.UniformTuckerTensorTrain.from_t3(p.to_t3()).data)
                p_u = (proj[0], proj[1])
            else:
                # corewise: the "frame" is the point's own cores -- identical across representations
                p_u = ubv.UT3Variations.from_t3variations(p.variations).supercores
            self.trial[name] = (p.variations.data, p_u)

    def _check(self, geom, prob_r):
        p_r, p_u = self.trial[geom]
        prob_u, x_u = F.uniform(prob_r, self.ux0)
        lm_r = prob_r.local_model(self.x0.data)
        lm_u = prob_u.local_model(x_u)
        self.assertTrue(np.isclose(float(lm_r.objective), float(lm_u.objective)), 'objective')
        self.assertTrue(np.isclose(float(self.cw.corewise_dot(lm_r.gradient, p_r)),
                                   float(prob_u.geom.inner(lm_u.gradient, p_u))), '<g, p>')
        self.assertTrue(np.isclose(float(lm_r.gn_quadratic(p_r)), float(lm_u.gn_quadratic(p_u))),
                        'gn_quadratic')

    def _sample_data(self, kind_name):
        SH = _STRUCT[0]
        ww = [np.random.randn(self.W, n) for n in SH]
        pp = [np.random.randn(self.W, n) for n in SH]
        idx = np.stack([np.random.randint(0, n, size=self.W) for n in SH], axis=0)
        O = 3
        return {
            'apply':   (ww, np.random.randn(self.W)),
            'entries': (idx, np.random.randn(self.W)),
            'probe':   (ww, [np.random.randn(self.W, n) for n in SH]),
            'apply_derivatives':   ((ww, pp), np.random.randn(O + 1, self.W)),
            'entries_derivatives': ((idx, pp), np.random.randn(O + 1, self.W)),
            'probe_derivatives':   ((ww, pp), [np.random.randn(O + 1, self.W, n) for n in SH]),
        }[kind_name]

    def test_plain_kinds_match_ragged(self):
        for geom, bgeom, _fg in self._GEOMS:
            for kind_name in ('apply', 'entries', 'probe'):
                with self.subTest(geometry=geom, kind=kind_name):
                    sample, data = self._sample_data(kind_name)
                    self._check(geom, F.fit(bgeom, kind_name, sample, data))

    def test_derivative_kinds_match_ragged(self):
        O, WEIGHT = 3, np.array([1.0, 0.5, 0.3, 0.2])
        for geom, bgeom, _fg in self._GEOMS:
            for kind_name in ('apply', 'entries', 'probe'):
                with self.subTest(geometry=geom, kind=kind_name):
                    (sample, pp), data = self._sample_data(kind_name + '_derivatives')
                    weight = np.tile(WEIGHT, (3, 1)) if kind_name == 'probe' else WEIGHT
                    self._check(geom, F.derivative_fit(bgeom, kind_name, O, sample, pp, data, weight=weight,
                                                       shape=_STRUCT[0]))


class TestUniformOptimizers(unittest.TestCase):
    """U5: the four backend optimizers run on the uniform Problem (fully packed) and reproduce / match the
    ragged run. Deliberately small + short (this suite runs a lot): correctness needs a *few* iterations,
    not deep convergence. gradient_descent is deterministic -> its trajectory matches ragged (to tolerance
    -- never rely on bit-exactness: it happens to hold on this machine, but different hardware / BLAS /
    summation orders can drift ~1e-15); newton_cg at a few steps still tracks ragged (its inner CG only
    diverges once such differences accumulate over many steps); the stochastic mc_sgd/adam descend and
    (mc_sgd) track ragged -- also exercising the packed-aware minibatch `take`. All numerical assertions
    are tolerance tests (np.allclose / np.isclose)."""
    def setUp(self):
        np.random.seed(0)
        SH, TK, TT = (4, 5, 6), (2, 2, 2), (1, 2, 2, 1)     # small + easy -> fast
        W = 25
        x_true = t3.TuckerTensorTrain.randn(SH, TK, TT)
        self.ww = [np.random.randn(W, n) for n in SH]
        self.data = bapply.t3_apply(x_true.data, self.ww)
        self.x0 = t3.TuckerTensorTrain.randn(SH, TK, TT)
        self.ux0 = ut3.UniformTuckerTensorTrain.from_t3(self.x0)
        self._x0u = (self.ux0.data[0], self.ux0.data[1])

    def _problems(self, geom):
        bg = {'manifold': bgeo.ManifoldGeometryOps(), 'corewise': bgeo.CorewiseGeometryOps()}[geom]
        pr = F.fit(bg, 'apply', self.ww, self.data)
        return pr, F.uniform(pr, self.ux0)[0]

    def test_gradient_descent_matches_ragged(self):
        pr, pu = self._problems('manifold')
        _, sr = pr.gradient_descent(self.x0.data, n_iter=8)
        _, su = pu.gradient_descent(self._x0u, n_iter=8)
        self.assertTrue(np.allclose(sr['losses'], su['losses']))   # deterministic -> matches ragged (to tol)

    def test_newton_cg_matches_ragged(self):
        pr, pu = self._problems('manifold')
        _, sr = pr.newton_cg(self.x0.data, max_newton=4)
        _, su = pu.newton_cg(self._x0u, max_newton=4)
        self.assertLess(su['losses'][-1], su['losses'][0])                          # descends
        self.assertTrue(np.isclose(su['losses'][-1], sr['losses'][-1], rtol=1e-2))  # tracks ragged

    def test_mc_sgd_matches_ragged_and_descends(self):
        # same rng + same flat draw + the packed take gives the same minibatch as ragged, so the descent
        # tracks the ragged run.
        pr, pu = self._problems('manifold')
        _, sr = pr.mc_sgd(self.x0.data, np.random.default_rng(0), 10, max_iter=100)
        _, su = pu.mc_sgd(self._x0u, np.random.default_rng(0), 10, max_iter=100)
        self.assertLess(su['losses'][-1], 0.7 * su['losses'][0])                    # descends
        self.assertTrue(np.isclose(su['losses'][-1], sr['losses'][-1], rtol=1e-3))  # tracks ragged

    def test_adam_descends(self):
        _, pu = self._problems('corewise')
        _, su = pu.adam(self._x0u, np.random.default_rng(0), 10, lr=5e-2, max_iter=200)
        self.assertLess(su['losses'][-1], 0.5 * su['losses'][0])                    # descends


class TestUniformMinimalRank(unittest.TestCase):
    """U5.6: uniform fitting requires a minimal-rank frame (from a non-minimal frame the retraction truncates
    to the realizable rank and desyncs from the fixed masks -> a mid-loop crash). uniform_minimal reduces
    to it (same tensor; a no-op if already minimal); UniformManifoldGeometryOps.from_point rejects a non-minimal
    x0 with a clear error up front (D5: where uniform_least_squares_problem's guard was; with sharing, the shared
    notion); the optimizer runs cleanly from the reduced frame. A local model at a non-minimal point (never
    retracted) builds its geometry from_frame, and its gradient, gn_quadratic and hvp work there."""
    def setUp(self):
        np.random.seed(0)
        self.ww = [np.random.randn(30, n) for n in (6, 6, 6)]
        self.data = np.random.randn(30)

    def _x0(self, TK, TT):
        return ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((6, 6, 6), TK, TT))

    def test_nonminimal_x0_rejected(self):
        x0 = self._x0((2, 2, 2), (1, 3, 3, 1))          # TT bond rank 3 unrealizable for a 2x2x2 core
        self.assertFalse(bool(np.all(x0.has_minimal_ranks)))
        with self.assertRaisesRegex(ValueError, 'requires a minimal-rank point'):
            bgeo.UniformManifoldGeometryOps.from_point(x0.data)
        bgeo.UniformCorewiseGeometryOps.from_point(x0.data)          # additive: no retraction to desync

    def test_shared_minimality_is_the_shared_notion(self):
        """With sharing, minimal means the shared notion (the group ceiling admits a group rank the per-mode bound
        clips): Tucker (2, 2, 1), TT (1, 1, 1, 1) on (6, 6, 6) with modes 0 and 1 shared is shared-minimal and not
        minimal per mode, so from_point accepts it with the sharing and raises without it."""
        import t3toolbox.backend.ranks as ranks
        shape, sharing, tk, tt = (6, 6, 6), (0, 0, 1), (2, 2, 1), (1, 1, 1, 1)
        self.assertEqual([int(n) for n in ranks.compute_minimal_ranks(shape, tk, tt, sharing=sharing)[0]], list(tk))
        self.assertNotEqual([int(n) for n in ranks.compute_minimal_ranks(shape, tk, tt)[0]], list(tk))
        x0 = self._x0(tk, tt)
        bgeo.UniformManifoldGeometryOps.from_point(x0.data, sharing)
        with self.assertRaisesRegex(ValueError, 'requires a minimal-rank point'):
            bgeo.UniformManifoldGeometryOps.from_point(x0.data)

    def test_a_local_model_at_a_nonminimal_point(self):
        """D5's other half: from_frame (the frame in hand) builds the geometry of a non-minimal point, and the model's
        gradient, gn_quadratic and hvp there agree with the ragged model's (<g, p>, <p, Hp> at the same tangent)."""
        import t3toolbox.backend.ufv_conversions as ufc
        import t3toolbox.corewise as cw
        x = t3.TuckerTensorTrain.randn((6, 6, 6), (2, 2, 2), (1, 3, 3, 1))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        self.assertFalse(bool(np.all(ux.has_minimal_ranks)))
        frame, centers = ufc.ut3_orthogonal_representations(ux.data)
        geom = bgeo.UniformManifoldGeometryOps.from_frame(frame, ux.data[3])
        rfit = F.fit(bgeo.ManifoldGeometryOps(), 'apply', self.ww, self.data)
        ufit = F.uniform(F.fit(bgeo.CorewiseGeometryOps(), 'apply', self.ww, self.data), ux)[0].with_(geom=geom)
        lm = ufit.local_model((ux.data[0], ux.data[1]), frame=frame, centers=(centers[0], centers[1]))
        lm_r = rfit.local_model(x.data)
        g = lm.gradient
        self.assertTrue(np.isclose(float(geom.inner(g, g)), float(rfit.geom.inner(lm_r.gradient, lm_r.gradient))))
        self.assertTrue(np.isclose(float(lm.gn_quadratic(g)), float(geom.inner(g, lm.hvp(g)))))
        self.assertTrue(np.isclose(float(lm.objective), float(lm_r.objective)))

    def test_uniform_minimal_reduces_same_tensor(self):
        x0 = self._x0((2, 2, 2), (1, 3, 3, 1))
        x0m = uf.uniform_minimal(x0)
        self.assertTrue(bool(np.all(x0m.has_minimal_ranks)))
        self.assertTrue(np.allclose(x0m.to_dense(), x0.to_dense()))   # lossless: the same tensor

    def test_uniform_minimal_noop_on_minimal(self):
        x0 = self._x0((2, 2, 2), (1, 2, 2, 1))           # already minimal
        self.assertIs(uf.uniform_minimal(x0), x0)        # returned unchanged (no re-gauge)

    def test_optimizer_runs_after_minimal(self):
        x0 = uf.uniform_minimal(self._x0((2, 2, 2), (1, 3, 3, 1)))
        prob, xc = F.uniform(F.fit(bgeo.ManifoldGeometryOps(), 'apply', self.ww, self.data), x0)
        _, s = prob.gradient_descent(xc, n_iter=10)
        self.assertLess(s['losses'][-1], s['losses'][0])



class TestPackSampleAndData(unittest.TestCase):
    """pack_sample(kind, order, sample, N) and pack_data(kind, order, y, N), the forms of P16 (in place of the old kind
    names): the input jets packed with their jet axes brought to the largest size by zero coefficients and the modes
    padded to N with zeros, the probe data packed likewise; the index and the scalar data unchanged; pack_batch is
    the two. Checked against the ragged arrays slot by slot (a jet slot beyond a mode's own is zero: data, O-45)."""

    def test_the_forms(self):
        rng = np.random.default_rng(0)
        shape, N, W = (4, 5, 3), 7, 6
        ww = [rng.standard_normal((F_in, W, n)) for F_in, n in zip((2, 1, 2), shape)]     # mixed jet sizes, order (2,)
        y = [rng.standard_normal((3, W, n)) for n in shape]                         # F = (3,)
        packed = uf.pack_sample('probe', (2,), ww, N)
        self.assertEqual(packed.shape, (3, 2, W, N))
        for i, (w, n) in enumerate(zip(ww, shape)):
            self.assertTrue(np.array_equal(packed[i, :w.shape[0], :, :n], w))
            self.assertFalse(np.any(packed[i, w.shape[0]:]))                         # the missing jet slot: zero
            self.assertFalse(np.any(packed[i, :, :, n:]))                            # the mode padding: zero
        self.assertIs(uf.pack_sample('apply', (2,), packed, N), packed)               # a packed sample is kept
        py = uf.pack_data('probe', (2,), y, N)
        self.assertEqual(py.shape, (3, 3, W, N))
        self.assertTrue(all(np.array_equal(py[i, ..., :n], a) for i, (a, n) in enumerate(zip(y, shape))))
        index = np.stack([rng.integers(0, n, W) for n in shape])
        self.assertIs(uf.pack_sample('entries', (), index, N), index)
        scalar = rng.standard_normal((3, W))
        self.assertIs(uf.pack_data('apply', (2,), scalar, N), scalar)
        batch = uf.pack_batch('probe', (2,), bfit.LeastSquaresBatch(ww, y, np.ones((3, 3))), N)
        self.assertTrue(np.array_equal(batch.sample, packed) and np.array_equal(batch.y, py))


class TestSlackPaddedStart(unittest.TestCase):
    """Review 2026-08-22 (C5): a minimal-rank x0 stored with slack padding (from_t3(x, n=, r=) above the
    real ranks) must run through every uniform optimizer; the retraction used to shrink the padded dims."""

    def test_optimizers_run_from_a_slack_padded_x0(self):
        np.random.seed(9)
        x_true = t3.TuckerTensorTrain.randn((4, 5, 6), (3, 3, 2), (1, 2, 2, 1))
        ww = tuple(np.random.randn(10, n) for n in x_true.shape)
        b = x_true.apply(ww)
        x0 = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))   # minimal (the frontend reduced the
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0, n=5, r=4)              # old (3, 3, 2) first); slack: real
        self.assertEqual((ux0.n, ux0.r), (5, 4))                              # max (3, 2)
        self.assertTrue(bool(np.all(ux0.has_minimal_ranks)))
        frame = ut3m.UNIFORM_MANIFOLD.frame(ux0)
        y = ut3m.UNIFORM_MANIFOLD.retract(ut3m.UNIFORM_MANIFOLD.randn(frame) * 1e-3)
        self.assertEqual((y.n, y.r), (5, 4))                                   # the frame's padded dims
        self.assertTrue(bool(np.all(np.asarray(y.tucker_ranks) == np.asarray(ux0.tucker_ranks))))
        rng = np.random.default_rng(0)
        prob, xc = F.uniform(F.fit(bgeo.ManifoldGeometryOps(), 'apply', ww, b), ux0)
        prob.newton_cg(xc, max_newton=2)
        prob.mc_sgd(xc, rng, 5, max_iter=3)
        prob.gradient_descent(xc, n_iter=2)

if __name__ == '__main__':
    unittest.main()
