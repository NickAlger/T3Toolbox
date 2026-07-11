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
import t3toolbox.backend.apply as bapply
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.utv_operations as utv_operations
import t3toolbox.backend.ut3_operations as uops

_STRUCT = ((10, 11, 12), (2, 4, 2), (1, 2, 2, 1))   # (shape, tucker, tt); MINIMAL ranks (tucker capped by
                                                    # the tt bonds), still rank-varied -> real padding

# name -> (frontend geometry singleton, backend GeometryOps factory)
_GEOMS = {
    'manifold': (ut3m.UNIFORM_MANIFOLD, uf.uniform_manifold_ops),
    'corewise': (ut3m.UNIFORM_COREWISE, uf.uniform_corewise_ops),
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
        frame_front, frame_back = geom_front.frame(x), ops.frame(x_sc)
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


class TestUniformSamplingKind(unittest.TestCase):
    """U3: the uniform SamplingKind builders (apply/entries/probe) reproduce the ragged SamplingKind on the
    equivalent frame (the uniform-equivalence contract), satisfy the adjoint identity <r, Jv> = <Jᵀr, v>,
    and ignore garbage in the masked-out variation padding."""
    # (name, ragged kind, sample-is-integer-index)
    _KINDS = [('apply', bfit.APPLY, False), ('probe', bfit.PROBE, False), ('entries', bfit.ENTRIES, True)]

    def setUp(self):
        np.random.seed(0)
        self.x = _uniform_x(0)
        self.frame = ut3m.UNIFORM_MANIFOLD.frame(self.x)
        self.var = ubv.UT3Variations.randn_like(self.frame)
        self.frame_r = self.frame.to_t3frame()               # equivalent ragged frame
        self.var_r = self.var.to_t3variations()            # equivalent ragged variation
        self.vmask = uf._var_masks_from_frame(self.frame.data)

    def _sample(self, is_index, W=15):
        shape = _STRUCT[0]
        if is_index:
            return np.stack([np.random.randint(0, n, size=W) for n in shape], axis=0)   # (d,)+W
        return [np.random.randn(W, n) for n in shape]                                    # len=d, W+(Ni,)

    def test_forward_matches_ragged(self):
        for name, kind_r, is_index in self._KINDS:
            with self.subTest(kind=name):
                kind_u = uf.uniform_sampling_kind(name, self.x.data)
                sample = self._sample(is_index)
                sw_u = kind_u.precompute(self.frame.data, sample)
                sw_r = kind_r.precompute(self.frame_r.data, sample)
                fu = kind_u.forward(self.var.supercores, sample, self.frame.data, sw_u)
                fr = kind_r.forward(self.var_r.data, sample, self.frame_r.data, sw_r)
                if name == 'probe':
                    fu = uops.unpack_vectors(fu, _STRUCT[0])   # the split-seam forward is PACKED; unpack to compare
                    self.assertTrue(all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(fu, fr)))
                else:
                    self.assertTrue(np.allclose(np.asarray(fu), np.asarray(fr)))

    def test_point_forward_matches_ragged(self):
        for name, kind_r, is_index in self._KINDS:
            with self.subTest(kind=name):
                kind_u = uf.uniform_sampling_kind(name, self.x.data)
                sample = self._sample(is_index)
                x_r = self.frame_r.to_t3()                   # the ragged point (== x)
                su = kind_u.point_forward((self.x.data[0], self.x.data[1]), sample)
                sr = kind_r.point_forward(x_r.data, sample)
                if name == 'probe':
                    self.assertTrue(all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(su, sr)))
                else:
                    self.assertTrue(np.allclose(np.asarray(su), np.asarray(sr)))

    def test_adjoint_identity(self):
        for name, kind_r, is_index in self._KINDS:
            with self.subTest(kind=name):
                kind_u = uf.uniform_sampling_kind(name, self.x.data)
                sample = self._sample(is_index)
                sw = kind_u.precompute(self.frame.data, sample)
                fwd = kind_u.forward(self.var.supercores, sample, self.frame.data, sw)
                # probe forward is PACKED (d,)+W+C+(N,); a packed random residual exercises the packed path.
                r = np.random.randn(*np.asarray(fwd).shape)
                lhs = float(np.sum(r * np.asarray(fwd)))
                jt = kind_u.transpose(r, sample, self.frame.data, sw)     # bare (dU, dG)
                rhs = float(utv_operations.utv_corewise_inner(
                    (jt[0], jt[1], _STRUCT[0], self.vmask), self.var.data, 0))
                self.assertTrue(np.isclose(lhs, rhs), f"{name}: {lhs} != {rhs}")

    def test_forward_garbage_robust(self):
        # garbage in the masked-out variation padding must not change the forward (mask-once contracts it away)
        for name, kind_r, is_index in self._KINDS:
            with self.subTest(kind=name):
                kind_u = uf.uniform_sampling_kind(name, self.x.data)
                sample = self._sample(is_index)
                sw = kind_u.precompute(self.frame.data, sample)
                clean = kind_u.forward(self.var.supercores, sample, self.frame.data, sw)
                V = self.var
                tkv, ttv = V.supercores
                m_tkv, m_ttv = ubv.UT3Variations(np.ones_like(tkv), np.ones_like(ttv),
                                                 V.shape, V.masks).apply_masks().supercores
                ck_tkv, ck_ttv = V.apply_masks().supercores
                garb = (ck_tkv + 1e6 * (1.0 - m_tkv), ck_ttv + 1e6 * (1.0 - m_ttv))
                dirty = kind_u.forward(garb, sample, self.frame.data, sw)
                self.assertTrue(np.allclose(np.asarray(clean), np.asarray(dirty)))   # scalar / packed array


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
        self.assertIsInstance(out_r, tuple)            # ragged in -> ragged out
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
        out_r = self.x.probe_derivatives(ww_r, pp_r, 3)
        out_p = self.x.probe_derivatives(ww_p, pp_p, 3)
        self.assertIsInstance(out_r, tuple)
        self.assertNotIsInstance(out_p, (list, tuple))
        unpacked = uops.unpack_vectors(out_p, _STRUCT[0])
        self.assertTrue(all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(unpacked, out_r)))


class TestUniformDerivativeSamplingKind(unittest.TestCase):
    """U3': the uniform DERIVATIVE (jet) SamplingKind builders reproduce the ragged derivative kind on the
    equivalent frame, satisfy the omega-weighted adjoint identity <w^2 r, Jv> = <J^T w^2 r, v>, and ignore
    garbage in the masked-out variation padding."""
    _ORDER = 3
    _WEIGHT = [1.0, 0.5, 0.3, 0.2]
    # (name, ragged kind factory, sample-first-is-integer-index)
    _KINDS = [('apply_derivatives', bfit.apply_derivatives_kind, False),
              ('probe_derivatives', bfit.probe_derivatives_kind, False),
              ('entries_derivatives', bfit.entries_derivatives_kind, True)]

    def setUp(self):
        np.random.seed(0)
        self.x = _uniform_x(0)
        self.frame = ut3m.UNIFORM_MANIFOLD.frame(self.x)
        self.var = ubv.UT3Variations.randn_like(self.frame)
        self.frame_r = self.frame.to_t3frame()
        self.var_r = self.var.to_t3variations()
        self.vmask = uf._var_masks_from_frame(self.frame.data)
        self.aw = bfit._make_order_weight(self._WEIGHT, self._ORDER)                    # order-leading (apply/entries)
        self.aw_packed = bfit._make_order_weight(self._WEIGHT, self._ORDER, order_axis=1)  # packed probe: order at axis 1

    def _sample(self, is_index, W=15):
        shape = _STRUCT[0]
        pp = [np.random.randn(W, n) for n in shape]
        if is_index:
            return (np.stack([np.random.randint(0, n, size=W) for n in shape], axis=0), pp)
        return ([np.random.randn(W, n) for n in shape], pp)

    def _kinds(self, name, ragged_factory):
        return (uf.uniform_derivatives_kind(name, self.x.data, self._ORDER, self._WEIGHT),
                ragged_factory(self._ORDER, self._WEIGHT))

    def test_forward_matches_ragged(self):
        for name, factory, is_index in self._KINDS:
            with self.subTest(kind=name):
                kind_u, kind_r = self._kinds(name, factory)
                sample = self._sample(is_index)
                fu = kind_u.forward(self.var.supercores, sample, self.frame.data,
                                    kind_u.precompute(self.frame.data, sample))
                fr = kind_r.forward(self.var_r.data, sample, self.frame_r.data,
                                    kind_r.precompute(self.frame_r.data, sample))
                if name == 'probe_derivatives':
                    fu = uops.unpack_vectors(fu, _STRUCT[0])   # the split-seam forward is PACKED; unpack to compare
                    self.assertTrue(all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(fu, fr)))
                else:
                    self.assertTrue(np.allclose(np.asarray(fu), np.asarray(fr)))

    def test_point_forward_matches_ragged(self):
        for name, factory, is_index in self._KINDS:
            with self.subTest(kind=name):
                kind_u, kind_r = self._kinds(name, factory)
                sample = self._sample(is_index)
                x_r = self.frame_r.to_t3()
                pu = kind_u.point_forward((self.x.data[0], self.x.data[1]), sample)
                pr = kind_r.point_forward(x_r.data, sample)
                if name == 'probe_derivatives':
                    self.assertTrue(all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(pu, pr)))
                else:
                    self.assertTrue(np.allclose(np.asarray(pu), np.asarray(pr)))

    def test_adjoint_identity_weighted(self):
        for name, factory, is_index in self._KINDS:
            with self.subTest(kind=name):
                kind_u, _ = self._kinds(name, factory)
                sample = self._sample(is_index)
                sw = kind_u.precompute(self.frame.data, sample)
                fwd = kind_u.forward(self.var.supercores, sample, self.frame.data, sw)
                # transpose applies the omega**2 weight aw(r, 2) internally -> the identity is
                # <aw(r,2), Jv> = <J^T aw(r,2), v>. probe_derivatives forward/residual are PACKED with the
                # order axis at 1 (after d), so its omega uses order_axis=1 (aw_packed).
                if name == 'probe_derivatives':
                    r = np.random.randn(*np.asarray(fwd).shape)   # packed (d,)+(order+1,)+W+C+(N,)
                    lhs = float(np.sum(np.asarray(self.aw_packed(r, 2)) * np.asarray(fwd)))
                else:
                    r = np.random.randn(*np.asarray(fwd).shape)
                    lhs = float(np.sum(np.asarray(self.aw(r, 2)) * np.asarray(fwd)))
                jt = kind_u.transpose(r, sample, self.frame.data, sw)
                rhs = float(utv_operations.utv_corewise_inner((jt[0], jt[1], _STRUCT[0], self.vmask), self.var.data, 0))
                self.assertTrue(np.isclose(lhs, rhs), f"{name}: {lhs} != {rhs}")

    def test_forward_garbage_robust(self):
        for name, factory, is_index in self._KINDS:
            with self.subTest(kind=name):
                kind_u, _ = self._kinds(name, factory)
                sample = self._sample(is_index)
                sw = kind_u.precompute(self.frame.data, sample)
                clean = kind_u.forward(self.var.supercores, sample, self.frame.data, sw)
                V = self.var
                tkv, ttv = V.supercores
                m_tkv, m_ttv = ubv.UT3Variations(np.ones_like(tkv), np.ones_like(ttv),
                                                 V.shape, V.masks).apply_masks().supercores
                ck_tkv, ck_ttv = V.apply_masks().supercores
                garb = (ck_tkv + 1e6 * (1.0 - m_tkv), ck_ttv + 1e6 * (1.0 - m_ttv))
                dirty = kind_u.forward(garb, sample, self.frame.data, sw)
                self.assertTrue(np.allclose(np.asarray(clean), np.asarray(dirty)))   # scalar / packed array


class TestUniformProblem(unittest.TestCase):
    """U4: the uniform least-squares Problem (fully packed) reproduces the ragged backend Problem's
    LocalModel -- objective, gradient (via <g, p>), and gn_quadratic(p) -- on the equivalent frame, for
    every sampling kind and both geometries. This certifies the Problem factory's sample/data packing and
    the reused LocalModel wiring."""
    _GEOMS = [('manifold', bopt.MANIFOLD_OPS, t3m.MANIFOLD), ('corewise', bopt.COREWISE_OPS, t3m.COREWISE)]

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
            self.trial[name] = (p.variations.data,
                                ubv.UT3Variations.from_t3variations(p.variations).supercores)

    def _check(self, geom, bgeom, kind_name, ragged_kind, sample, data, order=None, weight=None):
        p_r, p_u = self.trial[geom]
        prob_r = bopt.least_squares_problem(bgeom, ragged_kind, sample, data)
        prob_u = uf.uniform_least_squares_problem(geom, kind_name, self.ux0, sample, data, order, weight)
        lm_r = prob_r.local_model(self.x0.data)
        lm_u = prob_u.local_model((self.ux0.data[0], self.ux0.data[1]))
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
        kinds = {'apply': bfit.APPLY, 'entries': bfit.ENTRIES, 'probe': bfit.PROBE}
        for geom, bgeom, _fg in self._GEOMS:
            for kind_name, ragged_kind in kinds.items():
                with self.subTest(geometry=geom, kind=kind_name):
                    sample, data = self._sample_data(kind_name)
                    self._check(geom, bgeom, kind_name, ragged_kind, sample, data)

    def test_derivative_kinds_match_ragged(self):
        O, WEIGHT = 3, [1.0, 0.5, 0.3, 0.2]
        kinds = {'apply_derivatives': bfit.apply_derivatives_kind,
                 'entries_derivatives': bfit.entries_derivatives_kind,
                 'probe_derivatives': bfit.probe_derivatives_kind}
        for geom, bgeom, _fg in self._GEOMS:
            for kind_name, factory in kinds.items():
                with self.subTest(geometry=geom, kind=kind_name):
                    sample, data = self._sample_data(kind_name)
                    self._check(geom, bgeom, kind_name, factory(O, WEIGHT), sample, data, order=O, weight=WEIGHT)


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
        bg = {'manifold': bopt.MANIFOLD_OPS, 'corewise': bopt.COREWISE_OPS}[geom]
        return (bopt.least_squares_problem(bg, bfit.APPLY, self.ww, self.data),
                uf.uniform_least_squares_problem(geom, 'apply', self.ux0, self.ww, self.data))

    def test_gradient_descent_matches_ragged(self):
        pr, pu = self._problems('manifold')
        _, sr = bopt.gradient_descent(pr, self.x0.data, n_iter=8)
        _, su = bopt.gradient_descent(pu, self._x0u, n_iter=8)
        self.assertTrue(np.allclose(sr['losses'], su['losses']))   # deterministic -> matches ragged (to tol)

    def test_newton_cg_matches_ragged(self):
        pr, pu = self._problems('manifold')
        _, sr = bopt.newton_cg(pr, self.x0.data, max_newton=4)
        _, su = bopt.newton_cg(pu, self._x0u, max_newton=4)
        self.assertLess(su['losses'][-1], su['losses'][0])                          # descends
        self.assertTrue(np.isclose(su['losses'][-1], sr['losses'][-1], rtol=1e-2))  # tracks ragged

    def test_mc_sgd_matches_ragged_and_descends(self):
        # same rng + same flat draw + the packed take gives the same minibatch as ragged, so the descent
        # tracks the ragged run.
        pr, pu = self._problems('manifold')
        _, sr = bopt.mc_sgd(pr, self.x0.data, np.random.default_rng(0), batch=10, max_iter=100)
        _, su = bopt.mc_sgd(pu, self._x0u, np.random.default_rng(0), batch=10, max_iter=100)
        self.assertLess(su['losses'][-1], 0.7 * su['losses'][0])                    # descends
        self.assertTrue(np.isclose(su['losses'][-1], sr['losses'][-1], rtol=1e-3))  # tracks ragged

    def test_adam_descends(self):
        _, pu = self._problems('corewise')
        _, su = bopt.adam(pu, self._x0u, np.random.default_rng(0), batch=10, lr=5e-2, max_iter=200)
        self.assertLess(su['losses'][-1], 0.5 * su['losses'][0])                    # descends


class TestUniformMinimalRank(unittest.TestCase):
    """U5.6: uniform fitting requires a minimal-rank frame (from a non-minimal frame the retraction truncates
    to the realizable rank and desyncs from the fixed masks -> a mid-loop crash). uniform_minimal reduces
    to it (same tensor; a no-op if already minimal); uniform_least_squares_problem rejects a non-minimal x0
    with a clear error up front; the optimizer runs cleanly from the reduced frame."""
    def setUp(self):
        np.random.seed(0)
        self.ww = [np.random.randn(30, n) for n in (6, 6, 6)]
        self.data = np.random.randn(30)

    def _x0(self, TK, TT):
        return ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((6, 6, 6), TK, TT))

    def test_nonminimal_x0_rejected(self):
        x0 = self._x0((2, 2, 2), (1, 3, 3, 1))          # TT bond rank 3 unrealizable for a 2x2x2 core
        self.assertFalse(bool(np.all(x0.has_minimal_ranks)))
        with self.assertRaises(ValueError):
            uf.uniform_least_squares_problem('manifold', 'apply', x0, self.ww, self.data)

    def test_uniform_minimal_reduces_same_tensor(self):
        x0 = self._x0((2, 2, 2), (1, 3, 3, 1))
        x0m = uf.uniform_minimal(x0)
        self.assertTrue(bool(np.all(x0m.has_minimal_ranks)))
        self.assertTrue(np.allclose(x0m.to_dense(), x0.to_dense()))   # lossless: the same tensor

    def test_uniform_minimal_noop_on_minimal(self):
        x0 = self._x0((2, 2, 2), (1, 2, 2, 1))          # already minimal
        self.assertIs(uf.uniform_minimal(x0), x0)        # returned unchanged (no re-gauge)

    def test_optimizer_runs_after_minimal(self):
        x0 = uf.uniform_minimal(self._x0((2, 2, 2), (1, 3, 3, 1)))
        prob = uf.uniform_least_squares_problem('manifold', 'apply', x0, self.ww, self.data)
        _, s = bopt.gradient_descent(prob, (x0.data[0], x0.data[1]), n_iter=10)
        self.assertLess(s['losses'][-1], s['losses'][0])


if __name__ == '__main__':
    unittest.main()
