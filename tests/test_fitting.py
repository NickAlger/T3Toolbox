'''Tests for the geometry-generic Gauss-Newton fitting model (``fitting.py`` + ``backend/fitting.py``).

One ``GaussNewtonModel`` is parameterized over the **sampling kind** (apply / entries / probe) AND the
**geometry** (manifold / corewise). The headline oracle is **exact dense ground truth**: because the
sampling forward is linear in the ambient tensor, the least-squares objective is exactly quadratic, so the
Gauss-Newton model is the exact restriction of the objective to the affine tangent space ``r +
dense(𝒥 Π p)`` -- and ``dense(𝒥 Π p)`` is just ``geometry.project(p).to_dense()`` for *both* geometries
(manifold: a gauged tangent; corewise: the sum-of-core-swaps at ``(U,G,G,G)``). We also check the
two-form consistency, the razor self-containment (raw ``p`` == projected ``Π p``), the matched pair
(manifold gauges, corewise does not), GN symmetry, the ``J`` / ``Jᵀ`` adjoint, and -- as an independent
cross-check -- agreement with the established ``T3Tangent`` / ``TuckerTensorTrain`` transpose operators.
See ``dev/archive/geometry_refactor_plan.md``.
'''

import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.fitting as fb
import t3toolbox.fitting as fitting
import t3toolbox.corewise as cw

SHAPE = (7, 8, 9)
TUCKER_RANKS = (3, 4, 2)
TT_RANKS = (1, 2, 3, 1)
N_SAMPLES = 20
KINDS = ('apply', 'entries', 'probe')
GEOMS = {'manifold': t3m.MANIFOLD, 'corewise': t3m.COREWISE}
_FACTORY = {'apply': fitting.apply_model, 'entries': fitting.entries_model, 'probe': fitting.probe_model}
# tol: manifold matches to ~1e-13, corewise to ~1e-9 (large raw-core magnitudes) -> a relative band.
RTOL, ATOL = 1e-9, 1e-9


def apply_dense(T, ww, n_c):
    '''Sample-space all-modes apply of a dense (possibly C-stacked) tensor: ``T(w_1,...,w_d)`` per sample.
    ``T`` is ``C + (N_1,...,N_d)``, each ``ww[i]`` is ``(W,) + (N_i,)``; returns ``W + C``.'''
    d = len(ww)
    c_axes, mode_axes, w_axis = list(range(n_c)), [n_c + i for i in range(d)], n_c + d
    ops = [T, c_axes + mode_axes]
    for i in range(d):
        ops += [ww[i], [w_axis, mode_axes[i]]]
    ops += [[w_axis] + c_axes]
    return np.einsum(*ops)


def entries_dense(T, index, n_c):
    '''Entries of a dense tensor at integer ``index`` (shape ``(d,)+W``): ``T[index]``, returned ``W + C``.'''
    d = index.shape[0]
    n_w = index.ndim - 1
    sel = (slice(None),) * n_c + tuple(index[i] for i in range(d))
    return np.moveaxis(T[sel], tuple(range(n_c, n_c + n_w)), tuple(range(n_w)))


def dense_probe(T, ww, n_c):
    '''Probes of a dense tensor: leave mode ``i`` free, contract the rest. Returns ``d`` arrays ``W+C+(Ni,)``.'''
    d = len(ww)
    c_axes, mode_axes, w_axis = list(range(n_c)), [n_c + i for i in range(d)], n_c + d
    out = []
    for free in range(d):
        ops = [T, c_axes + mode_axes]
        for j in range(d):
            if j != free:
                ops += [ww[j], [w_axis, mode_axes[j]]]
        ops += [[w_axis] + c_axes + [mode_axes[free]]]
        out.append(np.einsum(*ops))
    return out


def _setup(kind, geom_name, C):
    '''Build a model of one (kind, geometry, frame-stack C), plus the dense forward oracle and the
    kind-aware sample-space reducers (samp_add / samp_dot / rand_like). One dict per call.'''
    np.random.seed(0)
    x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS, stack_shape=C)
    geometry = GEOMS[geom_name]
    m, n_c, n_w = N_SAMPLES, len(C), 1
    if kind == 'apply':
        sample = [np.random.randn(m, N) for N in SHAPE]
        dense_fwd = lambda T: apply_dense(T, sample, n_c)
        r = np.random.randn(*((m,) + C))
        samp_dot = lambda a, b: np.sum(a * b, axis=tuple(range(n_w)))
        samp_add = lambda a, b: a + b
        rand_like = lambda v: np.random.randn(*v.shape)
    elif kind == 'entries':
        sample = np.stack([np.random.randint(0, N, size=m) for N in SHAPE])   # (d,)+W
        dense_fwd = lambda T: entries_dense(T, sample, n_c)
        r = np.random.randn(*((m,) + C))
        samp_dot = lambda a, b: np.sum(a * b, axis=tuple(range(n_w)))
        samp_add = lambda a, b: a + b
        rand_like = lambda v: np.random.randn(*v.shape)
    else:  # probe -- vector-valued (one free mode each)
        sample = [np.random.randn(m, N) for N in SHAPE]
        dense_fwd = lambda T: dense_probe(T, sample, n_c)
        r = [np.random.randn(*((m,) + C + (N,))) for N in SHAPE]
        samp_dot = lambda a, b: sum(np.sum(ai * bi, axis=tuple(range(n_w)) + (ai.ndim - 1,))
                                    for ai, bi in zip(a, b))
        samp_add = lambda a, b: [ai + bi for ai, bi in zip(a, b)]
        rand_like = lambda v: [np.random.randn(*vi.shape) for vi in v]
    model = _FACTORY[kind](geometry, x, sample, r)
    return dict(x=x, geometry=geometry, geom_name=geom_name, model=model, frame=model.frame,
                sample=sample, r=r, c=0.5 * samp_dot(r, r), n_c=n_c, n_w=n_w, dense_fwd=dense_fwd,
                samp_dot=samp_dot, samp_add=samp_add, rand_like=rand_like)


def _raw_step(s):
    '''A raw (any-gauge) trial tangent at the model's frame -- ungauged on the manifold (tests Π),
    a raw core perturbation on the corewise frame.'''
    return t3m.COREWISE.randn(s['frame'])


class TestGaussNewtonModel(unittest.TestCase):
    '''The generic GN model, parameterized over the sampling kind, the geometry, and the frame stack C.'''

    def test_dense_truth(self):
        '''HEADLINE: model.evaluate(p) == ½‖r + 𝒥(Πp)‖² from the dense projected tangent (both geometries).'''
        for kind in KINDS:
            for geom_name in GEOMS:
                for C in [(), (2,)]:
                    with self.subTest(kind=kind, geom=geom_name, C=C):
                        s = _setup(kind, geom_name, C)
                        p = _raw_step(s)
                        Pp_dense = s['geometry'].project(p).to_dense()   # dense(𝒥 Π p); corewise: sum-of-swaps
                        res = s['samp_add'](s['r'], s['dense_fwd'](Pp_dense))
                        oracle = 0.5 * s['samp_dot'](res, res)
                        self.assertTrue(np.allclose(s['model'].evaluate(p), oracle, rtol=RTOL, atol=ATOL))

    def test_two_form_consistency(self):
        '''m(p) == c + ⟨g, p⟩ + ½⟨p, H p⟩ (the gn_hessian-based quadratic term), both geometries.'''
        for kind in KINDS:
            for geom_name in GEOMS:
                for C in [(), (2,)]:
                    with self.subTest(kind=kind, geom=geom_name, C=C):
                        s = _setup(kind, geom_name, C)
                        model, p = s['model'], _raw_step(s)
                        two_form = model.objective_value + model.gradient.corewise_inner(p) + 0.5 * p.corewise_inner(model.gn_hessian(p))
                        self.assertTrue(np.allclose(model.evaluate(p), two_form, rtol=RTOL, atol=ATOL))

    def test_razor_self_containment(self):
        '''The model applies Π itself: a raw p gives the same result as the projected Πp (both geometries).'''
        for kind in KINDS:
            for geom_name in GEOMS:
                for C in [(), (2,)]:
                    with self.subTest(kind=kind, geom=geom_name, C=C):
                        s = _setup(kind, geom_name, C)
                        model, p = s['model'], _raw_step(s)
                        Pp = s['geometry'].project(p)
                        self.assertTrue(np.allclose(model.evaluate(p), model.evaluate(Pp), rtol=RTOL, atol=ATOL))
                        self.assertTrue(model.gn_hessian(p).allclose(model.gn_hessian(Pp), rtol=RTOL, atol=ATOL).all())

    def test_matched_pair(self):
        '''The structural matched pair: MANIFOLD gauges (g, H gauged); COREWISE does NOT (g == bare 𝒥ᵀr).'''
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    sm = _setup(kind, 'manifold', C)
                    self.assertTrue(sm['model'].gradient.is_gauged().all())
                    self.assertTrue(sm['model'].gn_hessian(_raw_step(sm)).is_gauged().all())

                    sc = _setup(kind, 'corewise', C)
                    bare = sc['model'].kind.transpose(sc['r'], sc['sample'], sc['frame'].data, sc['model'].sweep)
                    gd = sc['model'].gradient.variations.data
                    for a, b in zip(gd[0] + gd[1], bare[0] + bare[1]):   # corewise gradient == bare 𝒥ᵀr, no Π
                        self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))

    def test_gn_hessian_symmetric(self):
        '''The GN normal operator is symmetric: ⟨q, H p⟩ == ⟨p, H q⟩ (both geometries).'''
        for kind in KINDS:
            for geom_name in GEOMS:
                with self.subTest(kind=kind, geom=geom_name):
                    s = _setup(kind, geom_name, ())
                    model, p, q = s['model'], _raw_step(s), _raw_step(s)
                    lhs = float(q.corewise_inner(model.gn_hessian(p)))
                    rhs = float(p.corewise_inner(model.gn_hessian(q)))
                    self.assertTrue(np.allclose(lhs, rhs, rtol=RTOL, atol=ATOL))

    def test_jacobian_gradient_adjoint(self):
        '''Adjoint ⟨z, 𝒥 Π p⟩_samples == ⟨Π 𝒥ᵀ z, p⟩_corewise (J = 𝒥∘Π, gradient = Π∘𝒥ᵀ), both geometries.'''
        for kind in KINDS:
            for geom_name in GEOMS:
                for C in [(), (2,)]:
                    with self.subTest(kind=kind, geom=geom_name, C=C):
                        s = _setup(kind, geom_name, C)
                        model, geometry, p = s['model'], s['geometry'], _raw_step(s)
                        Jp = model.jacobian(p)                                    # J p = 𝒥(Π p)
                        z = s['rand_like'](Jp)
                        gz_raw = model.kind.transpose(z, s['sample'], s['frame'].data, model.sweep)
                        gz = geometry.project(t3m.T3Tangent(s['frame'], bvf.T3Variations(*gz_raw)))  # Π 𝒥ᵀ z
                        lhs = s['samp_dot'](z, Jp)
                        rhs = gz.corewise_inner(p)
                        self.assertTrue(np.allclose(lhs, rhs, rtol=RTOL, atol=ATOL))

    def test_jacobian_and_gn_quadratic(self):
        '''jacobian(p) == the dense forward 𝒥(Πp); gn_quadratic(p) == pᵀHp == ‖Jp‖² (one forward, both geoms).'''
        for kind in KINDS:
            for geom_name in GEOMS:
                for C in [(), (2,)]:
                    with self.subTest(kind=kind, geom=geom_name, C=C):
                        s = _setup(kind, geom_name, C)
                        model, p = s['model'], _raw_step(s)
                        # gn_quadratic == pᵀ H p (the cheap Cauchy / line-search denominator)
                        self.assertTrue(np.allclose(model.gn_quadratic(p),
                                                    p.corewise_inner(model.gn_hessian(p)), rtol=RTOL, atol=ATOL))
                        # jacobian == the dense forward of the projected tangent (a sequence for probe)
                        Jp = model.jacobian(p)
                        Jp_oracle = s['dense_fwd'](s['geometry'].project(p).to_dense())
                        seq = lambda v: list(v) if isinstance(v, (list, tuple)) else [v]
                        for a, b in zip(seq(Jp), seq(Jp_oracle)):
                            self.assertTrue(np.allclose(a, b, rtol=RTOL, atol=ATOL))

    def test_same_frame_guard(self):
        '''A trial tangent at a different frame is a structural error (identity, not value), both geometries.'''
        for kind in KINDS:
            for geom_name in GEOMS:
                with self.subTest(kind=kind, geom=geom_name):
                    s = _setup(kind, geom_name, ())
                    other_x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS)
                    p_other = s['geometry'].randn(s['geometry'].frame(other_x))
                    with self.assertRaises(ValueError):
                        s['model'].gn_hessian(p_other)
                    with self.assertRaises(ValueError):
                        s['model'].evaluate(p_other)

    def test_caching(self):
        '''The frame sweep / gradient / objective are cached -- the reuse mechanism, computed once.'''
        model = _setup('apply', 'manifold', ())['model']
        self.assertIs(model.sweep, model.sweep)
        self.assertIs(model.gradient, model.gradient)
        self.assertIs(model.objective_value, model.objective_value)

    def test_matches_established_manifold(self):
        '''Manifold gradient/Hessian == the established bare T3Tangent transpose + MANIFOLD.project (apply).'''
        s = _setup('apply', 'manifold', ())
        frame, ww, r, model = s['frame'], s['sample'], s['r'], s['model']
        ref_g = t3m.MANIFOLD.project(t3m.T3Tangent.apply_transpose(r, ww, frame, sum_over_probes=True))
        gd = model.gradient.variations.data
        for a, b in zip(gd[0] + gd[1], ref_g.variations.data[0] + ref_g.variations.data[1]):
            self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))
        V = t3m.MANIFOLD.randn(frame)
        ref_HV = t3m.MANIFOLD.project(t3m.T3Tangent.apply_transpose(
            V.apply(ww), ww, frame, sum_over_probes=True))
        Hv = model.gn_hessian(V)
        for a, b in zip(Hv.variations.data[0] + Hv.variations.data[1],
                        ref_HV.variations.data[0] + ref_HV.variations.data[1]):
            self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))

    def test_matches_established_corewise(self):
        '''Corewise gradient == the established (jax.grad-verified) TuckerTensorTrain corewise transpose
        (all kinds) -- confirming the §6.3 substitution per kind and that NO projection sneaks in.'''
        ref_method = {'apply': 'apply_corewise_transpose',
                      'entries': 'entries_corewise_transpose',
                      'probe': 'probe_corewise_transpose'}
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    s = _setup(kind, 'corewise', C)
                    g_ref = getattr(s['x'], ref_method[kind])(s['r'], s['sample'], sum_over_probes=True)
                    gd = s['model'].gradient.variations.data
                    for a, b in zip(gd[0] + gd[1], g_ref[0] + g_ref[1]):
                        self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))

    def test_derivative_models(self):
        '''The derivative GN models (apply/entries/probe, per-order weight ω) wrap the derivative kind and
        reproduce the backend LocalModel's objective / gradient / gn_quadratic / gn_hessian -- oracle ==
        frontend, both geometries. (RAW residual r = S(x); backend data = 0, so the residuals match.)'''
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.backend.fitting as bfit
        import t3toolbox.backend.sampling_derivatives as pd
        rng = np.random.default_rng(0)
        shape, order, NW = (5, 6, 7), 2, 15
        omega = np.array([1.0, 0.5, 0.3])
        X = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
        ww = [rng.standard_normal((NW, N)) for N in shape]
        pp = [rng.standard_normal((NW, N)) for N in shape]
        index = np.stack([rng.integers(0, N, size=NW) for N in shape], axis=0)
        cases = [
            ('apply', fitting.apply_derivatives_model, (X, ww, pp, order),
             bfit.apply_derivatives_kind(order, omega), (ww, pp),
             np.asarray(pd.t3_apply_derivatives(ww, pp, X.data, order))),
            ('entries', fitting.entries_derivatives_model, (X, index, pp, order),
             bfit.entries_derivatives_kind(order, omega), (index, pp),
             np.asarray(pd.t3_entries_derivatives(index, pp, X.data, order))),
            ('probe', fitting.probe_derivatives_model, (X, ww, pp, order),
             bfit.probe_derivatives_kind(order, omega), (ww, pp),
             [np.asarray(z) for z in pd.t3_probe_derivatives(ww, pp, X.data, order)]),
        ]
        relerr = lambda a, b: float(cw.corewise_norm(cw.corewise_sub(a, b)) / cw.corewise_norm(b))
        for geom_f, geom_b in [(t3m.MANIFOLD, bopt.MANIFOLD_OPS), (t3m.COREWISE, bopt.COREWISE_OPS)]:
            for name, factory, fargs, bkind, sample, Sx in cases:
                with self.subTest(geom=geom_f, kind=name):
                    r = [np.asarray(z) for z in Sx] if isinstance(Sx, list) else Sx
                    data = [np.zeros_like(z) for z in r] if isinstance(r, list) else np.zeros_like(r)
                    fmodel = factory(geom_f, *fargs, r, weight=omega)
                    lm = bopt.least_squares_problem(geom_b, bkind, sample, data).local_model(X.data)
                    self.assertTrue(np.allclose(float(fmodel.objective_value), float(lm.objective)))
                    self.assertLess(relerr(fmodel.gradient.variations.data, lm.gradient), 1e-10)
                    pt = geom_f.randn(fmodel.frame); p = pt.variations.data
                    self.assertTrue(np.allclose(float(fmodel.gn_quadratic(pt)), float(lm.gn_quadratic(p))))
                    self.assertLess(relerr(fmodel.gn_hessian(pt).variations.data, lm.hvp(p)), 1e-10)


class TestResidualWeighting(unittest.TestCase):
    '''The ``ω[mode, order]`` residual weight matrix (``½‖ω⊙r‖²``). Mode weighting is **probe-only**;
    apply/entries take an ORDER-only weight (a per-mode weight is a structural error). Oracles place
    ``ω[i, t]`` by **explicit numpy indexing** (NOT via ``_make_weight``, so the axis-placement check is
    non-circular): the objective is a hand-summed ``½ Σ (ω⊙r)²``, and the weighted gradient ``𝒥ᵀ(ω²r)``
    equals the *unweighted* gradient fed the explicitly-scaled residual ``ω²⊙r``. Also: the ``(mode,order)``
    broadcasting (row = order, column = mode, matrix = both), the backward-compatible bare-vector rule, and
    the ``pᵀHp == ‖ω⊙Jp‖²`` / hand-rolled ``𝒥ᵀ(ω²𝒥p)`` Hessian consistency under the matrix weight.'''

    def _setup(self, order=2, C=(), NW=15):
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS, stack_shape=C)
        ww = [np.random.randn(NW, N) for N in SHAPE]
        pp = [np.random.randn(NW, N) for N in SHAPE]
        d = len(SHAPE)
        # RAW residual jets: probe -> list of d, (order+1)+W+C+(Ni,); apply/entries -> (order+1)+W+C
        r_probe = [np.random.randn(*((order + 1, NW) + C + (N,))) for N in SHAPE]
        r_scalar = np.random.randn(*((order + 1, NW) + C))
        return dict(x=x, ww=ww, pp=pp, d=d, order=order, C=C, r_probe=r_probe, r_scalar=r_scalar)

    def _probe_obj_oracle(self, r, W, order):
        '''½ Σ_{i,t} ‖W[i,t] r_i[t]‖² with W[i,t] placed by explicit indexing (r a list of d arrays).'''
        tot = 0.0
        for i, ri in enumerate(r):
            wi = np.asarray(W)[i, :].reshape((order + 1,) + (1,) * (ri.ndim - 1))
            tot += np.sum((wi * ri) ** 2)
        return 0.5 * tot

    def _probe_scale(self, r, W, order, power):
        '''ω**power ⊙ r, W[i,t] placed by explicit indexing (the non-circular scaled residual).'''
        out = []
        for i, ri in enumerate(r):
            wi = np.asarray(W)[i, :].reshape((order + 1,) + (1,) * (ri.ndim - 1))
            out.append((wi ** power) * ri)
        return out

    def test_probe_derivatives_full_matrix(self):
        '''probe_derivatives with a full ω[mode, order] matrix: hand-summed objective + the ω²r-scaled
        gradient identity, both geometries. Non-circular (ω placed by explicit indexing).'''
        s = self._setup()
        d, order = s['d'], s['order']
        rng = np.random.default_rng(1)
        W = rng.uniform(0.2, 2.0, size=(d, order + 1))           # a genuine full matrix
        for geom in (t3m.MANIFOLD, t3m.COREWISE):
            with self.subTest(geom=geom):
                w = fitting.probe_derivatives_model(geom, s['x'], s['ww'], s['pp'], order, s['r_probe'], weight=W)
                self.assertTrue(np.allclose(float(w.objective_value),
                                            self._probe_obj_oracle(s['r_probe'], W, order)))
                # weighted gradient 𝒥ᵀ(ω²r) == unweighted gradient on the explicitly-scaled ω²⊙r
                u2 = fitting.probe_derivatives_model(geom, s['x'], s['ww'], s['pp'], order,
                                                     self._probe_scale(s['r_probe'], W, order, 2), weight=None)
                self.assertTrue(w.gradient.allclose(u2.gradient, rtol=1e-9, atol=1e-11).all())
                # Hessian consistency: pᵀHp == ‖ω⊙Jp‖², and H p == 𝒥ᵀ(ω²⊙𝒥Πp) (hand-rolled, ω explicit)
                p = t3m.COREWISE.randn(w.frame)
                Jp = w.jacobian(p)                                # raw 𝒥Πp (a list of d arrays)
                q_oracle = sum(np.sum((np.asarray(W)[i, :].reshape((order + 1,) + (1,) * (Jp[i].ndim - 1)) * Jp[i]) ** 2)
                               for i in range(d))
                self.assertTrue(np.allclose(float(w.gn_quadratic(p)), q_oracle))
                self.assertTrue(np.allclose(float(w.gn_quadratic(p)), float(p.corewise_inner(w.gn_hessian(p)))))

    def test_probe_derivatives_row_col_broadcast(self):
        '''Row (order+1,) = per-order (broadcast over modes); column (d,1) = per-mode (broadcast over
        orders); both are the matching full matrix by np.broadcast_to. Bare (order+1,) == its (1,order+1)
        row (the backward-compatible rule).'''
        s = self._setup()
        d, order, geom = s['d'], s['order'], t3m.MANIFOLD
        rng = np.random.default_rng(2)
        row = rng.uniform(0.3, 1.5, size=order + 1)              # per-order
        col = rng.uniform(0.3, 1.5, size=(d, 1))                 # per-mode
        for w_in, full in [(row, np.broadcast_to(row, (d, order + 1))),
                           (col, np.broadcast_to(col, (d, order + 1)))]:
            with self.subTest(shape=np.shape(w_in)):
                a = fitting.probe_derivatives_model(geom, s['x'], s['ww'], s['pp'], order, s['r_probe'], weight=w_in)
                b = fitting.probe_derivatives_model(geom, s['x'], s['ww'], s['pp'], order, s['r_probe'], weight=full)
                self.assertTrue(np.allclose(float(a.objective_value), float(b.objective_value)))
                self.assertTrue(a.gradient.allclose(b.gradient, rtol=1e-10, atol=1e-12).all())
        # bare 1-D order vector == explicit (1, order+1) row (backward compat)
        bare = fitting.probe_derivatives_model(geom, s['x'], s['ww'], s['pp'], order, s['r_probe'], weight=row)
        rowm = fitting.probe_derivatives_model(geom, s['x'], s['ww'], s['pp'], order, s['r_probe'],
                                               weight=row.reshape(1, -1))
        self.assertTrue(np.allclose(float(bare.objective_value), float(rowm.objective_value)))

    def test_apply_entries_order_only(self):
        '''apply/entries derivatives take an ORDER-only weight (hand-summed ½Σ_t (ω_t r_t)²); a genuine
        per-mode weight (mode dim > 1) is a structural error (no mode axis).'''
        s = self._setup()
        order, geom = s['order'], t3m.MANIFOLD
        rng = np.random.default_rng(3)
        omega = rng.uniform(0.3, 1.5, size=order + 1)
        index = np.stack([np.random.randint(0, N, size=s['r_scalar'].shape[1]) for N in SHAPE], axis=0)
        for name, model in [('apply', fitting.apply_derivatives_model(geom, s['x'], s['ww'], s['pp'], order, s['r_scalar'], weight=omega)),
                            ('entries', fitting.entries_derivatives_model(geom, s['x'], index, s['pp'], order, s['r_scalar'], weight=omega))]:
            with self.subTest(kind=name):
                oracle = 0.5 * np.sum((omega.reshape((order + 1,) + (1,) * (s['r_scalar'].ndim - 1)) * s['r_scalar']) ** 2)
                self.assertTrue(np.allclose(float(model.objective_value), oracle))
        # a per-mode weight (d, order+1) with d>1 rows -> structural error (backend raises when applied)
        bad = np.ones((s['d'], order + 1))
        with self.assertRaises(ValueError):
            fitting.apply_derivatives_model(geom, s['x'], s['ww'], s['pp'], order, s['r_scalar'], weight=bad).objective_value

    def test_plain_probe_per_mode(self):
        '''Plain probe (order 0) per-mode weight (d,): hand-summed ½Σ_i ω_i²‖r_i‖² + the ω²r-scaled gradient
        identity. Backend probe_kind is lenient on (d,1); the frontend 1-D contract is S2's concern.'''
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS)
        ww = [np.random.randn(N_SAMPLES, N) for N in SHAPE]
        r = [np.random.randn(N_SAMPLES, N) for N in SHAPE]       # plain probe residual: list of d, W+(Ni,)
        d = len(SHAPE)
        omega = np.array([0.5, 2.0, 1.3])[:d]
        for geom in (t3m.MANIFOLD, t3m.COREWISE):
            with self.subTest(geom=geom):
                frame = geom.frame(x)
                kind = fb.probe_kind(omega)
                w = fitting.GaussNewtonModel(geom, frame, kind, ww, r, kind.precompute(frame.data, ww))
                obj_oracle = 0.5 * sum((omega[i] ** 2) * np.sum(r[i] ** 2) for i in range(d))
                self.assertTrue(np.allclose(float(w.objective_value), obj_oracle))
                r2 = [(omega[i] ** 2) * r[i] for i in range(d)]  # explicit ω²⊙r
                u = fitting.GaussNewtonModel(geom, frame, fb.PROBE, ww, r2, fb.PROBE.precompute(frame.data, ww))
                self.assertTrue(w.gradient.allclose(u.gradient, rtol=1e-9, atol=1e-11).all())
                p = t3m.COREWISE.randn(frame)
                self.assertTrue(np.allclose(float(w.gn_quadratic(p)), float(p.corewise_inner(w.gn_hessian(p)))))
        # backend probe_kind is lenient: (d,) and (d,1) give the same weighted kind
        k1 = fb.probe_kind(omega)
        k2 = fb.probe_kind(omega.reshape(d, 1))
        fr = t3m.MANIFOLD.frame(x); sw = k1.precompute(fr.data, ww)
        m1 = fitting.GaussNewtonModel(t3m.MANIFOLD, fr, k1, ww, r, sw)
        m2 = fitting.GaussNewtonModel(t3m.MANIFOLD, fr, k2, ww, r, sw)
        self.assertTrue(np.allclose(float(m1.objective_value), float(m2.objective_value)))


class TestUniformGaussNewtonModel(unittest.TestCase):
    '''U7b: the uniform roll-your-own surface. ``fitting.apply_model`` &c. dispatch a
    ``UniformTuckerTensorTrain`` x to a ``UniformGaussNewtonModel`` (UT3Tangent-valued gradient / Hessian).
    Oracle: the tested backend ``LocalModel`` (== ragged) -- objective / gradient / gn_quadratic /
    gn_hessian agree to ~machine precision (both run the SAME packed kind + gauge projection). We feed the
    backend model's exact packed residual to the frontend model so the residuals match by construction.'''

    def _relerr(self, a, b):
        return float(cw.corewise_norm(cw.corewise_sub(a, b)) / cw.corewise_norm(b))

    def _cases(self):
        import t3toolbox.uniform_tucker_tensor_train as ut3
        rng = np.random.default_rng(0)
        shape, order, NW = (5, 6, 7), 2, 15
        d = len(shape)
        omega = [1.0, 0.5, 0.3]                                         # per-order (order+1,)
        omega_mode = rng.uniform(0.3, 2.0, size=d)                      # per-mode (d,), for plain probe
        wmat = rng.uniform(0.3, 2.0, size=(d, order + 1))              # full ω[mode,order], for probe-derivs
        x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)))
        ww = [rng.standard_normal((NW, N)) for N in shape]
        pp = [rng.standard_normal((NW, N)) for N in shape]
        index = np.stack([rng.integers(0, N, size=NW) for N in shape], axis=0)
        scal = rng.standard_normal(NW)                                 # apply/entries observed data
        vecs = [rng.standard_normal((NW, N)) for N in shape]           # probe observed data (ragged)
        jet_s = rng.standard_normal((order + 1, NW))                   # apply/entries-deriv observed
        jet_v = [rng.standard_normal((order + 1, NW, N)) for N in shape]  # probe-deriv observed
        # (kind_name, frontend factory, factory sample-args, backend sample, backend data, order, weight)
        return x, order, omega, [
            ('apply',   fitting.apply_model,   (ww,),        ww,           scal,  None,  None),
            ('entries', fitting.entries_model, (index,),     index,        scal,  None,  None),
            ('probe',   fitting.probe_model,   (ww,),        ww,           vecs,  None,  None),
            ('probe',   fitting.probe_model,   (ww,),        ww,           vecs,  None,  omega_mode),   # per-mode
            ('apply_derivatives',   fitting.apply_derivatives_model,   (ww, pp, order),    (ww, pp),    jet_s, order, omega),
            ('entries_derivatives', fitting.entries_derivatives_model, (index, pp, order), (index, pp), jet_s, order, omega),
            ('probe_derivatives',   fitting.probe_derivatives_model,   (ww, pp, order),    (ww, pp),    jet_v, order, omega),
            ('probe_derivatives',   fitting.probe_derivatives_model,   (ww, pp, order),    (ww, pp),    jet_v, order, wmat),  # matrix
        ]

    def test_matches_backend_local_model(self):
        import t3toolbox.uniform_manifold as ut3m
        import t3toolbox.backend.uniform_fitting as uf
        x, order, omega, cases = self._cases()
        geoms = [('manifold', ut3m.UNIFORM_MANIFOLD), ('corewise', ut3m.UNIFORM_COREWISE)]
        for gname, geom in geoms:
            for kind, factory, fargs, bsample, bdata, o, w in cases:
                with self.subTest(geom=gname, kind=kind, wshape=None if w is None else np.shape(w)):
                    prob = uf.uniform_least_squares_problem(gname, kind, x, bsample, bdata, o, w)
                    lm = prob.local_model((x.data[0], x.data[1]))
                    kw = {'weight': w} if w is not None else {}
                    fmodel = factory(geom, x, *fargs, lm.residual, **kw)   # lm.residual is packed -> mirror no-op
                    self.assertIsInstance(fmodel, fitting.UniformGaussNewtonModel)
                    self.assertTrue(np.allclose(float(fmodel.objective_value), float(lm.objective)))
                    self.assertLess(self._relerr(fmodel.gradient.variations.supercores, lm.gradient), 1e-10)
                    pt = geom.randn(fmodel.frame)
                    p = pt.variations.supercores
                    self.assertTrue(np.allclose(float(fmodel.gn_quadratic(pt)), float(lm.gn_quadratic(p)), rtol=1e-9))
                    self.assertLess(self._relerr(fmodel.gn_hessian(pt).variations.supercores, lm.hvp(p)), 1e-10)

    def test_gauged_gradient_and_self_consistency(self):
        '''On the manifold the gradient is gauged and ``pᵀHp == ‖J p‖²`` (the GN quadratic form matches the
        Hessian action); the corewise gradient is NOT gauged.'''
        import t3toolbox.uniform_manifold as ut3m
        import t3toolbox.backend.uniform_fitting as uf
        x, order, omega, cases = self._cases()
        for kind, factory, fargs, bsample, bdata, o, w in cases:
            for gname, geom, want_gauged in [('manifold', ut3m.UNIFORM_MANIFOLD, True),
                                             ('corewise', ut3m.UNIFORM_COREWISE, False)]:
                with self.subTest(kind=kind, geom=gname, wshape=None if w is None else np.shape(w)):
                    prob = uf.uniform_least_squares_problem(gname, kind, x, bsample, bdata, o, w)
                    lm = prob.local_model((x.data[0], x.data[1]))
                    kw = {'weight': w} if w is not None else {}
                    m = factory(geom, x, *fargs, lm.residual, **kw)
                    self.assertEqual(bool(m.gradient.is_gauged().all()), want_gauged)
                    p = geom.randn(m.frame)
                    self.assertTrue(np.allclose(float(m.gn_quadratic(p)),
                                                float(p.corewise_inner(m.gn_hessian(p))), rtol=1e-9))

    def test_representation_geometry_must_match(self):
        '''A uniform x with a ragged geometry (or a ragged x with a uniform geometry) is a structural error.'''
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        rng = np.random.default_rng(0)
        shape = (5, 6, 7)
        ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)))
        rx = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
        ww = [rng.standard_normal((10, N)) for N in shape]
        r = rng.standard_normal(10)
        with self.assertRaises(ValueError):   # uniform x + ragged geometry
            fitting.apply_model(t3m.MANIFOLD, ux, ww, r)
        with self.assertRaises(ValueError):   # ragged x + uniform geometry
            fitting.apply_model(ut3m.UNIFORM_MANIFOLD, rx, ww, r)

    def test_same_frame_guard(self):
        '''A trial tangent at a different frame is rejected (the numerical same-frame guard).'''
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        rng = np.random.default_rng(0)
        shape = (5, 6, 7)
        x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)))
        other = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)))
        ww = [rng.standard_normal((10, N)) for N in shape]
        m = fitting.apply_model(ut3m.UNIFORM_MANIFOLD, x, ww, rng.standard_normal(10))
        p_other = ut3m.UNIFORM_MANIFOLD.randn(ut3m.UNIFORM_MANIFOLD.frame(other))
        with self.assertRaises(ValueError):
            m.gn_hessian(p_other)


if __name__ == '__main__':
    unittest.main()
