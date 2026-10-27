'''Tests for the geometry-generic Gauss-Newton fitting model (``fitting.py`` + ``backend/fitting.py``).

One ``fitting.LocalModel`` is parameterized over the **sampling kind** (apply / entries / probe) AND the
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
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.fitting as fb
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.sharing as bsharing
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


class TestLocalModel(unittest.TestCase):
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

    def test_regularizer(self):
        '''An IdentityRegularizer folds ρ=½λ‖X‖² into the frontend LocalModel on all kinds/geometries:
        the two-form identity `m(p)=c+⟨g,p⟩+½⟨p,Hp⟩` and `gn_quadratic==pᵀHp` still hold (reg folded
        consistently across all five methods) and the objective gains ρ.'''
        import t3toolbox.backend.optimizers as bopt
        lam = 0.4
        BOPS = {'manifold': bgeo.ManifoldGeometryOps(), 'corewise': bgeo.CorewiseGeometryOps()}
        for kind in KINDS:
            for geom_name in GEOMS:
                with self.subTest(kind=kind, geom=geom_name):
                    s = _setup(kind, geom_name, ())
                    R = bopt.IdentityRegularizer(lam)
                    model = _FACTORY[kind](s['geometry'], s['x'], s['sample'], s['r'], regularizer=R)
                    p = _raw_step(s)
                    two_form = (model.objective_value + model.gradient.corewise_inner(p)
                                + 0.5 * p.corewise_inner(model.gn_hessian(p)))
                    self.assertTrue(np.allclose(model.evaluate(p), two_form, rtol=RTOL, atol=ATOL))
                    self.assertTrue(np.allclose(model.gn_quadratic(p),
                                                p.corewise_inner(model.gn_hessian(p)), rtol=RTOL, atol=ATOL))
                    rho = float(R.value(BOPS[geom_name], (model.frame.data[0], model.frame.data[2])))
                    self.assertTrue(np.allclose(model.objective_value, s['c'] + rho, rtol=RTOL, atol=ATOL))

    def test_uniform_regularizer(self):
        '''S3b: the uniform-layer model supports the same regularizer (roll-your-own uniform reg): the
        two-form / `gn_quadratic==pᵀHp` identities hold with reg, and the uniform objective matches the
        ragged model's exactly (the equivalence contract). Uses minimal ranks -- the model factory (unlike
        the optimizer) does not call uniform_minimal.'''
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        np.random.seed(0)
        R = bopt.IdentityRegularizer(0.4)
        Xr = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))   # minimal / realizable ranks
        Xu = ut3.UniformTuckerTensorTrain.from_t3(Xr)
        ww = [np.random.randn(20, N) for N in (6, 7, 8)]
        r = [np.random.randn(20, N) for N in (6, 7, 8)]
        for ug, rg in [(ut3m.UNIFORM_MANIFOLD, t3m.MANIFOLD), (ut3m.UNIFORM_COREWISE, t3m.COREWISE)]:
            with self.subTest(geom=type(ug).__name__):
                um = fitting.probe_model(ug, Xu, ww, r, regularizer=R)
                rm = fitting.probe_model(rg, Xr, ww, r, regularizer=R)
                p = ug.randn(um.frame)
                two_form = (um.objective_value + um.gradient.corewise_inner(p)
                            + 0.5 * p.corewise_inner(um.gn_hessian(p)))
                self.assertTrue(np.allclose(um.evaluate(p), two_form, rtol=RTOL, atol=1e-7))
                self.assertTrue(np.allclose(um.gn_quadratic(p),
                                            p.corewise_inner(um.gn_hessian(p)), rtol=RTOL, atol=1e-7))
                self.assertTrue(np.allclose(um.objective_value, rm.objective_value, rtol=RTOL, atol=1e-7))

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
        for geom_f, geom_b in [(t3m.MANIFOLD, bgeo.ManifoldGeometryOps()), (t3m.COREWISE, bgeo.CorewiseGeometryOps())]:
            for name, factory, fargs, bkind, sample, Sx in cases:
                with self.subTest(geom=type(geom_f).__name__, kind=name):   # name, not the object -- xdist reports must serialize (execnet)
                    r = [np.asarray(z) for z in Sx] if isinstance(Sx, list) else Sx
                    data = [np.zeros_like(z) for z in r] if isinstance(r, list) else np.zeros_like(r)
                    fmodel = factory(geom_f, *fargs, r, weight=omega)
                    lm = bopt.least_squares_problem(geom_b, bkind, sample, data).local_model(X.data)
                    self.assertTrue(np.allclose(float(fmodel.objective_value), float(lm.objective)))
                    self.assertLess(relerr(fmodel.gradient.variations.data, lm.gradient), 1e-10)
                    pt = geom_f.randn(fmodel.frame); p = pt.variations.data
                    self.assertTrue(np.allclose(float(fmodel.gn_quadratic(pt)), float(lm.gn_quadratic(p))))
                    self.assertLess(relerr(fmodel.gn_hessian(pt).variations.data, lm.hvp(p)), 1e-10)


class TestBlockSumsq(unittest.TestCase):
    '''``SamplingKind.block_sumsq`` (D2): the per-`(mode, order)` sum-of-squares that keeps the mode/order
    axes (for the Newton-CG diagnostic error table). Oracle: each block equals a hand ``np.sum(r_ij**2)``
    (non-circular); the matrix shape mirrors the kind's ``ω`` shape; and ``block_sumsq(r).sum()`` equals
    the UNWEIGHTED ``‖r‖²`` (ω never enters, even on a weighted kind).'''

    def test_block_sumsq_all_kinds(self):
        rng = np.random.default_rng(0)
        d, order, NW = 3, 2, 7
        Ns = [4, 5, 6]
        r_probe = [rng.standard_normal((NW, N)) for N in Ns]                    # plain probe: W+(Ni,)
        r_pd = [rng.standard_normal((order + 1, NW, N)) for N in Ns]            # probe-deriv: (order+1)+W+(Ni,)
        r_scal = rng.standard_normal((NW,))                                     # plain apply/entries: W
        r_ad = rng.standard_normal((order + 1, NW))                            # apply/entries-deriv: (order+1)+W
        # (kind, residual, expected (n_mode, n_order), oracle block[i,t])
        cases = [
            (fb.PROBE, r_probe, (d, 1), lambda i, t: np.sum(r_probe[i] ** 2)),
            (fb.probe_derivatives_kind(order), r_pd, (d, order + 1), lambda i, t: np.sum(r_pd[i][t] ** 2)),
            (fb.APPLY, r_scal, (1, 1), lambda i, t: np.sum(r_scal ** 2)),
            (fb.apply_derivatives_kind(order), r_ad, (1, order + 1), lambda i, t: np.sum(r_ad[t] ** 2)),
        ]
        for kind, r, shape, oracle in cases:
            with self.subTest(kind=kind.name):
                bs = kind.block_sumsq(r, 1)
                self.assertEqual(bs.shape, shape)                               # mirrors the ω shape
                for i in range(shape[0]):
                    for t in range(shape[1]):
                        self.assertTrue(np.allclose(bs[i, t], oracle(i, t)))    # per-block dense oracle
                total = sum(np.sum(z ** 2) for z in r) if isinstance(r, list) else np.sum(r ** 2)
                self.assertTrue(np.allclose(float(bs.sum()), float(total)))     # sum == unweighted ‖r‖²

    def test_block_sumsq_ignores_weight(self):
        '''A weighted probe_derivatives kind's block_sumsq is still the RAW ‖r_ij‖² (the table is the honest
        data-norm error, not the weighted residual the objective sees).'''
        rng = np.random.default_rng(1)
        d, order, NW = 3, 2, 6
        Ns = [4, 5, 6]
        r = [rng.standard_normal((order + 1, NW, N)) for N in Ns]
        W = rng.uniform(0.2, 3.0, size=(d, order + 1))
        weighted = fb.probe_derivatives_kind(order, W)
        bs = weighted.block_sumsq(r, 1)
        self.assertTrue(np.allclose(bs[1, 2], np.sum(r[1][2] ** 2)))            # raw, ω[1,2] not applied


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
            with self.subTest(geom=type(geom).__name__):   # name, not the object -- xdist reports must serialize (execnet)
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
            with self.subTest(geom=type(geom).__name__):   # name, not the object -- xdist reports must serialize (execnet)
                frame = geom.frame(x)
                kind = fb.probe_kind(omega)
                w = fitting.LocalModel(geom, frame, kind, ww, r, kind.precompute(frame.data, ww))
                obj_oracle = 0.5 * sum((omega[i] ** 2) * np.sum(r[i] ** 2) for i in range(d))
                self.assertTrue(np.allclose(float(w.objective_value), obj_oracle))
                r2 = [(omega[i] ** 2) * r[i] for i in range(d)]  # explicit ω²⊙r
                u = fitting.LocalModel(geom, frame, fb.PROBE, ww, r2, fb.PROBE.precompute(frame.data, ww))
                self.assertTrue(w.gradient.allclose(u.gradient, rtol=1e-9, atol=1e-11).all())
                p = t3m.COREWISE.randn(frame)
                self.assertTrue(np.allclose(float(w.gn_quadratic(p)), float(p.corewise_inner(w.gn_hessian(p)))))
        # backend probe_kind is lenient: (d,) and (d,1) give the same weighted kind
        k1 = fb.probe_kind(omega)
        k2 = fb.probe_kind(omega.reshape(d, 1))
        fr = t3m.MANIFOLD.frame(x); sw = k1.precompute(fr.data, ww)
        m1 = fitting.LocalModel(t3m.MANIFOLD, fr, k1, ww, r, sw)
        m2 = fitting.LocalModel(t3m.MANIFOLD, fr, k2, ww, r, sw)
        self.assertTrue(np.allclose(float(m1.objective_value), float(m2.objective_value)))


class TestPerModeWeightRowCount(unittest.TestCase):
    """Review 2026-08-22 (S8): a per-mode weight with MORE rows than modes is truncated (intended -- one
    weight rides through a continuation scheme that adds modes), on both layers; FEWER rows is a structural
    error at construction on the frontend and at the first reduction on the backend (not an IndexError)."""

    def setUp(self):
        np.random.seed(5)
        self.x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        self.ww = tuple(np.random.randn(7, n) for n in (4, 5, 6))
        self.r = [np.random.randn(7, n) for n in (4, 5, 6)]                    # a probe residual, W=(7,)
        self.w5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_over_long_is_truncated(self):
        import t3toolbox.backend.fitting as bfit
        m_long = fitting.probe_model(t3m.MANIFOLD, self.x, self.ww, self.r, weight=self.w5)
        m_d = fitting.probe_model(t3m.MANIFOLD, self.x, self.ww, self.r, weight=self.w5[:3])
        self.assertAlmostEqual(float(m_long.objective_value), float(m_d.objective_value), places=10)
        k_long = bfit.probe_kind(weight=self.w5)
        k_d = bfit.probe_kind(weight=self.w5[:3])
        self.assertAlmostEqual(float(k_long.sumsq(self.r, 1)), float(k_d.sumsq(self.r, 1)), places=10)

    def test_short_is_rejected(self):
        import t3toolbox.backend.fitting as bfit
        with self.assertRaises(ValueError):
            fitting.probe_model(t3m.MANIFOLD, self.x, self.ww, self.r, weight=self.w5[:2])
        with self.assertRaises(ValueError):
            bfit.probe_kind(weight=self.w5[:2]).sumsq(self.r, 1)


class TestUniformLayerLocalModel(unittest.TestCase):
    '''U7b: the uniform roll-your-own surface. ``fitting.apply_model`` &c. dispatch a
    ``UniformTuckerTensorTrain`` x to a ``fitting.LocalModel`` on a UT3Frame (UT3Tangent-valued gradient /
    Hessian) -- one model class serves both representations.
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
                    self.assertIsInstance(fmodel, fitting.LocalModel)
                    self.assertIsInstance(fmodel.frame, ubv.UT3Frame)   # the uniform layer, one model class
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



class TestFactoryValidation(unittest.TestCase):
    """Review 2026-08-22 (C13): the factories validate sample / residual against the point up front."""

    def test_mismatches_are_named_value_errors(self):
        np.random.seed(6)
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
        ww = tuple(np.random.randn(7, n) for n in x.shape)
        r = x.apply(ww)
        fitting.apply_model(t3m.MANIFOLD, x, ww, r)                                   # the good case
        with self.assertRaises(ValueError):
            fitting.apply_model(t3m.MANIFOLD, x, ww, r[:5])                            # residual shape
        with self.assertRaises(ValueError):
            fitting.apply_model(t3m.MANIFOLD, x, ww[:2], r)                            # len(ww)
        bad = (ww[0], np.random.randn(7, 9), ww[2])
        with self.assertRaises(ValueError):
            fitting.apply_model(t3m.MANIFOLD, x, bad, r)                               # N_i
        pr = x.probe(ww)
        with self.assertRaises(ValueError):
            fitting.probe_model(t3m.MANIFOLD, x, ww, pr[:2])                           # probe residual len
        idx = np.random.randint(0, 4, size=(3, 7)) % np.array(x.shape)[:, None]
        with self.assertRaises(ValueError):
            fitting.entries_model(t3m.MANIFOLD, x, idx[:2], x.entries(idx))            # index rows
        pp = tuple(np.random.randn(7, n) for n in x.shape)
        jets = x.apply_derivatives(ww, pp, 2)
        fitting.apply_derivatives_model(t3m.MANIFOLD, x, ww, pp, 2, jets)
        with self.assertRaises(ValueError):
            fitting.apply_derivatives_model(t3m.MANIFOLD, x, ww, pp, 2, jets[:2])     # order axis


# ---------------------------------------------------------------------------------------------------
# 2026.3.0 (S6a / S6): the uniform backend geometry from the frame, and the frontend exact mode (T18)
# ---------------------------------------------------------------------------------------------------
_EX_MINIMAL = ((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
_EX_TUCKER_SLACK = ((6, 6, 6), (4, 4, 4), (1, 2, 2, 1))     # frame down ranks (2, 4, 2) < up (4, 4, 4)
_EX_TT_SLACK = ((6, 6, 6), (4, 4, 4), (1, 9, 9, 1))         # frame left ranks != right ranks
_EX_D4 = ((4, 5, 4, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1))
_EX_SHAREABLE = ((5, 5, 7), (2, 2, 2), (1, 2, 2, 1))        # modes 0 and 1 can share a factor


def _ex_unit_point(struct):
    '''A random T3 with ``||X||_HS = 1`` (so finite differences are well scaled).'''
    x = t3.TuckerTensorTrain.randn(*struct)
    tk, tt = x.data
    return t3.TuckerTensorTrain(tuple(tk), tuple(tt[:-1]) + (tt[-1] / float(np.linalg.norm(x.to_dense())),))


def _ex_problem(struct, order, W, rng, weight=None, sharing=None):
    '''A probe-derivative fit whose data come from ANOTHER unit point plus 10 % noise (an O(1) residual, so the
    curvature term is O(1)): the point x, the sample, the data, and the canonical weight.'''
    x, other = _ex_unit_point(struct), _ex_unit_point(struct)
    if sharing is not None:                                   # a tied point (the shared geometries' set)
        x = t3.TuckerTensorTrain(*bsharing.t3_tie_tucker_factors(x.data, sharing))
    ww = [rng.standard_normal((W, N)) for N in struct[0]]
    pp = [rng.standard_normal((W, N)) for N in struct[0]]
    data = [z + 0.1 * rng.standard_normal(z.shape) for z in other.probe_derivatives(ww, pp, order)]
    return x, (ww, pp), data


def _ex_residual(x, sample, data, order):
    return [z - y for z, y in zip(x.probe_derivatives(*sample, order), data)]


def _flat(tree):
    '''Every array of a (nested tuple of) core tuple(s), in order.'''
    if isinstance(tree, (tuple, list)):
        return [a for t in tree for a in _flat(t)]
    return [np.asarray(tree)]


def _bitwise(a, b):
    fa, fb_ = _flat(a), _flat(b)
    return len(fa) == len(fb_) and all(u.shape == v.shape and np.array_equal(u, v) for u, v in zip(fa, fb_))


def _relerr_trees(a, b):
    fa, fb_ = _flat(a), _flat(b)
    num = np.sqrt(sum(float(np.sum((u - v) ** 2)) for u, v in zip(fa, fb_)))
    return num / np.sqrt(sum(float(np.sum(v ** 2)) for v in fb_))


class TestUniformBackendGeometryFromFrame(unittest.TestCase):
    '''S6a (X13): the frontend model's uniform backend geometry is built from the frame it already holds
    (``UniformManifoldGeometryOps.from_frame``), not re-orthogonalized from the base point -- the SAME geometry
    as ``from_point`` (value equality over every field, the variation masks included), so the Gauss-Newton
    numbers are unchanged; on minimal, Tucker-slack and TT-slack points, a C-stacked point and a shared one.'''

    def setUp(self):
        np.random.seed(61)

    def test_from_frame_is_from_point_without_the_orthogonalization(self):
        from unittest import mock
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        import t3toolbox.shared_geometry as sg
        import t3toolbox.backend.ufv_conversions as ufc
        rng = np.random.default_rng(61)
        cases = [('minimal', _EX_MINIMAL, (), None), ('tucker_slack', _EX_TUCKER_SLACK, (), None),
                 ('tt_slack', _EX_TT_SLACK, (), None), ('stacked', _EX_MINIMAL, (2,), None),
                 ('shared', _EX_SHAREABLE, (), (0, 0, 1))]
        for name, struct, C, sharing in cases:
            with self.subTest(case=name):
                x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*struct, stack_shape=C))
                geom = ut3m.UNIFORM_MANIFOLD if sharing is None else sg.shared(ut3m.UNIFORM_MANIFOLD, sharing)
                ww = [rng.standard_normal((6, N)) for N in struct[0]]
                r = [rng.standard_normal((6,) + C + (N,)) for N in struct[0]]
                m = fitting.probe_model(geom, x, ww, r)
                with mock.patch.object(ufc, 'ut3_orthogonal_representations',
                                       wraps=ufc.ut3_orthogonal_representations) as spy:
                    built = m._bgeom
                self.assertEqual(spy.call_count, 0)                           # no re-orthogonalization
                self.assertIsInstance(built, bgeo.UniformManifoldGeometryOps)
                x0_data = (m.frame.data[0], m.frame.data[2], m.kind.shape, m.kind.masks)
                reference = bgeo.UniformManifoldGeometryOps.from_point(x0_data, sharing)
                self.assertEqual(built, reference)                             # value equality: every field
                self.assertEqual(hash(built), hash(reference))
                for a, b in zip(built.var_masks + built.masks, reference.var_masks + reference.masks):
                    self.assertEqual(a.shape, b.shape)
                    self.assertTrue(np.array_equal(a, b))
                self.assertEqual(built.groups, reference.groups)
                direct = bgeo.UniformManifoldGeometryOps.from_frame(m.frame.data, m.kind.masks, sharing)
                self.assertEqual(direct, reference)

    def test_gauss_newton_regularized_numbers_unchanged(self):
        '''The geometry feeds the regularizer: the uniform regularized model's objective, gradient and
        quadratic are those computed with the ``from_point`` geometry, bit for bit.'''
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        rng = np.random.default_rng(62)
        R = bopt.IdentityRegularizer(0.4)
        x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*_EX_MINIMAL))
        ww = [rng.standard_normal((9, N)) for N in _EX_MINIMAL[0]]
        r = [rng.standard_normal((9, N)) for N in _EX_MINIMAL[0]]
        m = fitting.probe_model(ut3m.UNIFORM_MANIFOLD, x, ww, r, regularizer=R)
        ref = bgeo.UniformManifoldGeometryOps.from_point((m.frame.data[0], m.frame.data[2], m.kind.shape,
                                                          m.kind.masks))
        p = ut3m.UNIFORM_MANIFOLD.randn(m.frame)
        c = 0.5 * m.kind.sumsq(m.residual, m._n_w) + R.value(ref, (m.frame.data[0], m.frame.data[2]))
        self.assertEqual(float(m.objective_value), float(c))
        q = m.kind.sumsq(m.jacobian(p), m._n_w) + R.quadratic(ref, m.frame.data, p.variations.supercores)
        self.assertEqual(float(m.gn_quadratic(p)), float(q))
        g_reg = R.gradient(ref, m.frame.data)
        g = m.gradient.variations.supercores
        g_mis = m._wrap(m.kind.transpose(m.residual, m.sample, m.frame.data, m.sweep)).variations.supercores
        self.assertTrue(_bitwise(g, tuple(a + b for a, b in zip(g_mis, g_reg))))


class TestExactLocalModel(unittest.TestCase):
    '''T18: the frontend exact mode. ``fitting.LocalModel`` built with ``hessian='exact'`` equals the backend
    ``Problem.local_model(hessian='exact')`` on the same data; ``is_exact`` is derived (no ``hessian`` field);
    the exact gradient is the Gauss-Newton one bit for bit; ``hessian`` / ``quadratic`` / ``evaluate`` are
    mode-dependent and the ``curvature_*`` actions exist only in exact mode; ``retract`` is the geometry's
    (a shared model's with its companion) and the backend model's; ``curvature_damping`` reaches the lift and
    raises where it does nothing; the structural errors of the kinds and layers not built yet; and the exact
    Hessian against a finite difference of the Riemannian gradient along the retraction (the T3Polynomial
    ``x07`` cross-check).'''

    def setUp(self):
        np.random.seed(71)

    @staticmethod
    def _backend_twin(geom_ops, x, sample, data, order, weight, regularizer=None, hessian='gauss_newton',
                      curvature_damping=None):
        import t3toolbox.backend.optimizers as bopt
        wm = fitting._canonical_weight(weight, 'probe_derivatives', len(sample[0]), order)
        kind = fb.probe_derivatives_kind(order, wm)
        prob = bopt.least_squares_problem(geom_ops, kind, sample, data, regularizer=regularizer)
        return prob.local_model(x.data, hessian=hessian, curvature_damping=curvature_damping)

    def test_mode_is_derived(self):
        '''``is_exact`` is ``curvature is not None`` -- False for a Gauss-Newton model, True for
        ``hessian='exact'`` (case-insensitive); there is no ``hessian`` field or attribute other than the
        method; an unknown mode raises.'''
        import dataclasses
        import t3toolbox.backend.optimizers as bopt
        rng = np.random.default_rng(71)
        x, sample, data = _ex_problem(_EX_MINIMAL, 1, 8, rng)
        r = _ex_residual(x, sample, data, 1)
        gn = fitting.probe_derivatives_model(t3m.MANIFOLD, x, *sample, 1, r)
        ex = fitting.probe_derivatives_model(t3m.MANIFOLD, x, *sample, 1, r, hessian='EXACT')
        self.assertFalse(gn.is_exact)
        self.assertIsNone(gn.curvature)
        self.assertTrue(ex.is_exact)
        self.assertIsInstance(ex.curvature, bopt.CurvatureState)
        names = [f.name for f in dataclasses.fields(fitting.LocalModel)]
        self.assertIn('curvature', names)
        self.assertNotIn('hessian', names)
        self.assertNotIn('is_exact', names)
        self.assertNotIn('hessian', vars(ex))                                  # no instance attribute
        self.assertTrue(callable(ex.hessian) and callable(gn.hessian))        # the method, in both modes
        self.assertIsInstance(vars(fitting.LocalModel)['is_exact'], property)  # read-only, derived
        with self.assertRaises(AttributeError):
            ex.is_exact = False
        self.assertFalse(fitting.probe_derivatives_model(t3m.MANIFOLD, x, *sample, 1, r,
                                                         hessian='Gauss_Newton').is_exact)
        for bad in (True, None, 'newton', 'Gauss-Newton', ''):
            with self.subTest(hessian=bad):
                with self.assertRaisesRegex(ValueError, "hessian must be 'gauss_newton' or 'exact'"):
                    fitting.probe_derivatives_model(t3m.MANIFOLD, x, *sample, 1, r, hessian=bad)
        p = t3m.MANIFOLD.randn(gn.frame)
        for method in ('curvature_hessian', 'curvature_quadratic'):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, 'needs an exact-mode model'):
                    getattr(gn, method)(p)

    def test_matches_the_backend_exact_local_model(self):
        '''Every action of the exact frontend model equals the backend exact model's on the same data: the
        gradient and the curvature actions bit for bit (the same code on the same arrays), ``hessian`` /
        ``quadratic`` to rounding (the frontend sums its two gauged halves, the backend projects once). Ragged
        manifold (minimal, Tucker slack, TT slack, d = 4, a full ω matrix, a regularizer) and corewise, a
        tangent stack K included.'''
        import t3toolbox.backend.optimizers as bopt
        rng = np.random.default_rng(72)
        man, cwg = bgeo.ManifoldGeometryOps(), bgeo.CorewiseGeometryOps()
        cases = [('minimal', _EX_MINIMAL, 1, None, None, t3m.MANIFOLD, man),
                 ('tucker_slack', _EX_TUCKER_SLACK, 1, None, None, t3m.MANIFOLD, man),
                 ('tt_slack', _EX_TT_SLACK, 0, None, None, t3m.MANIFOLD, man),
                 ('d4_omega', _EX_D4, 2, 'matrix', None, t3m.MANIFOLD, man),
                 ('regularized', _EX_MINIMAL, 2, 'row', bopt.IdentityRegularizer(0.3), t3m.MANIFOLD, man),
                 ('corewise', _EX_MINIMAL, 1, 'row', None, t3m.COREWISE, cwg)]
        for name, struct, order, wkind, reg, geom, geom_ops in cases:
            with self.subTest(case=name):
                x, sample, data = _ex_problem(struct, order, 7, rng)
                d = len(struct[0])
                w = {None: None, 'row': rng.uniform(0.3, 2.0, order + 1),
                     'matrix': rng.uniform(0.3, 2.0, (d, order + 1))}[wkind]
                blm = self._backend_twin(geom_ops, x, sample, data, order, w, reg, hessian='exact')
                m = fitting.probe_derivatives_model(geom, x, *sample, order, blm.residual, weight=w,
                                                    regularizer=reg, hessian='exact')
                self.assertTrue(_bitwise(m.frame.data, blm.frame))
                self.assertEqual(float(m.objective_value), float(blm.objective))
                self.assertTrue(_bitwise(m.gradient.variations.data, blm.gradient))
                self.assertTrue(_bitwise(m.curvature.misfit_gradient, blm.curvature.misfit_gradient))
                for K in ((), (2,)):
                    p = geom.randn(m.frame, stack_shape=K)
                    raw = p.variations.data
                    self.assertTrue(_bitwise(m.curvature_hessian(p).variations.data, blm.curvature_hvp(raw)))
                    self.assertTrue(np.array_equal(np.asarray(m.curvature_quadratic(p)),
                                                   np.asarray(blm.curvature_quadratic(raw))))
                    self.assertTrue(np.array_equal(np.asarray(m.quadratic(p)), np.asarray(blm.quadratic(raw))))
                    self.assertLess(_relerr_trees(m.hessian(p).variations.data, blm.hvp(raw)), 1e-12)
                    self.assertTrue(np.array_equal(np.asarray(m.gn_quadratic(p)), np.asarray(blm.gn_quadratic(raw))))

    def test_uniform_matches_the_backend_exact_local_model(self):
        '''S8 (the DEFERRED S6 gate): the exact frontend model on the UNIFORM layer is the backend exact model of
        ``uniform_least_squares_problem`` on the same supercores -- its ``_blm`` runs on the 6-tuple frame and
        ``_tangent_at`` wraps bare supercores: ``gradient``, the misfit gradient, ``curvature_hessian``,
        ``curvature_quadratic`` and ``quadratic`` bit for bit, ``hessian`` to rounding (two gauged halves vs one
        projection), on both uniform geometries, a tangent stack K included; and the uniform m27 -- the exact
        gradient IS the Gauss-Newton gradient, bit for bit, masks included.'''
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        import t3toolbox.backend.uniform_fitting as ufit
        rng = np.random.default_rng(74)
        for struct, order in ((_EX_MINIMAL, 1), (_EX_D4, 2)):
            x, sample, data = _ex_problem(struct, order, 7, rng)
            ux = ut3.UniformTuckerTensorTrain.from_t3(x)
            w = rng.uniform(0.3, 2.0, (len(struct[0]), order + 1))
            for geom, name in ((ut3m.UNIFORM_MANIFOLD, 'manifold'), (ut3m.UNIFORM_COREWISE, 'corewise')):
                with self.subTest(struct=struct, geometry=name):
                    prob = ufit.uniform_least_squares_problem(name, 'probe_derivatives', ux, sample, data, order=order,
                                                              weight=w)
                    blm = prob.local_model((ux.data[0], ux.data[1]), hessian='exact')
                    m = fitting.probe_derivatives_model(geom, ux, *sample, order, blm.residual, weight=w,
                                                        hessian='exact')
                    gn = fitting.probe_derivatives_model(geom, ux, *sample, order, blm.residual, weight=w)
                    self.assertTrue(m.is_exact)
                    self.assertTrue(_bitwise(m.frame.data[:4], blm.frame[:4]))
                    self.assertEqual(float(m.objective_value), float(blm.objective))
                    self.assertTrue(_bitwise(m.gradient.variations.supercores, blm.gradient))
                    self.assertTrue(_bitwise(m.curvature.misfit_gradient, blm.curvature.misfit_gradient))
                    self.assertTrue(_bitwise(m.gradient.variations.supercores, gn.gradient.variations.supercores))
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(m.gradient.variations.masks.data,
                                                                             gn.gradient.variations.masks.data)))
                    for K in ((), (2,)):
                        p = geom.randn(m.frame, stack_shape=K)
                        raw = p.variations.supercores
                        self.assertTrue(_bitwise(m.curvature_hessian(p).variations.supercores, blm.curvature_hvp(raw)))
                        self.assertTrue(np.array_equal(np.asarray(m.curvature_quadratic(p)),
                                                       np.asarray(blm.curvature_quadratic(raw))))
                        self.assertTrue(np.array_equal(np.asarray(m.quadratic(p)), np.asarray(blm.quadratic(raw))))
                        self.assertLess(_relerr_trees(m.hessian(p).variations.supercores, blm.hvp(raw)), 1e-12)
                        self.assertEqual(np.shape(m.quadratic(p)), K)

    def test_shared_uniform_matches_the_backend_exact_local_model(self):
        '''S9: the exact frontend model on a SHARED uniform geometry (``shared(UNIFORM_MANIFOLD / UNIFORM_COREWISE,
        ...)``) is the backend exact model of ``uniform_least_squares_problem(..., sharing=)`` on the same supercores
        -- ``gradient`` (the tied one), the misfit gradient, ``curvature_hessian``, ``curvature_quadratic`` and
        ``quadratic`` bit for bit, ``hessian`` to rounding, a tangent stack K included -- and on the manifold its
        backend view runs on the model's ONE uniform companion (``_blm.geom_aux`` is ``geometry_aux``, the tied lift
        factors carry it, no curvature action rebuilds it). On the group-ceiling structure the tied lift is
        load-bearing: the per-mode lift would differ.'''
        import dataclasses
        from unittest import mock
        import t3toolbox.shared_geometry as sg
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.backend.ufv_operations as ufo
        import t3toolbox.backend.uniform_fitting as ufit
        rng = np.random.default_rng(82)
        cases = [('ceiling', ((6, 6, 4), (4, 4, 2), (1, 2, 2, 1)), (0, 0, 1), 2, 'row'),
                 ('all_modes', ((5, 5, 5), (3, 3, 3), (1, 2, 2, 1)), (0, 0, 0), 1, None)]
        for name, struct, labels, order, wkind in cases:
            x, sample, data = _ex_problem(struct, order, 7, rng, sharing=labels)
            ux = ut3.UniformTuckerTensorTrain.from_t3(x)
            w = None if wkind is None else rng.uniform(0.3, 2.0, order + 1)
            for base, gname in ((ut3m.UNIFORM_MANIFOLD, 'manifold'), (ut3m.UNIFORM_COREWISE, 'corewise')):
                with self.subTest(case=name, geometry=gname):
                    geom = sg.shared(base, labels)
                    prob = ufit.uniform_least_squares_problem(gname, 'probe_derivatives', ux, sample, data,
                                                              order=order, weight=w, sharing=labels)
                    blm = prob.local_model((ux.data[0], ux.data[1]), hessian='exact')
                    m = fitting.probe_derivatives_model(geom, ux, *sample, order, blm.residual, weight=w,
                                                        hessian='exact')
                    self.assertTrue(m.is_exact)
                    self.assertTrue(_bitwise(m.frame.data[:4], blm.frame[:4]))
                    self.assertTrue(_bitwise(m.gradient.variations.supercores, blm.gradient))
                    self.assertTrue(_bitwise(m.curvature.misfit_gradient, blm.curvature.misfit_gradient))
                    if gname == 'manifold':
                        self.assertIs(m._blm.geom_aux, m.geometry_aux)
                        self.assertIs(m.curvature.geometry_data.lift_factors.shared_data, m.geometry_aux)
                    for K in ((), (2,)):
                        p = geom.randn(m.frame, stack_shape=K)
                        raw = p.variations.supercores
                        with mock.patch.object(bsharing, 'ufv_shared_frame_data',
                                               wraps=bsharing.ufv_shared_frame_data) as spy:
                            Hc, qc, q = m.curvature_hessian(p), m.curvature_quadratic(p), m.quadratic(p)
                            H = m.hessian(p)
                        self.assertEqual(spy.call_count, 0)                            # the stored companion
                        self.assertTrue(_bitwise(Hc.variations.supercores, blm.curvature_hvp(raw)))
                        self.assertTrue(np.array_equal(np.asarray(qc), np.asarray(blm.curvature_quadratic(raw))))
                        self.assertTrue(np.array_equal(np.asarray(q), np.asarray(blm.quadratic(raw))))
                        self.assertLess(_relerr_trees(H.variations.supercores, blm.hvp(raw)), 1e-12)
                        self.assertEqual(np.shape(q), K)
                    if name == 'ceiling' and gname == 'manifold':                     # the tied lift matters
                        gd = blm.curvature.geometry_data
                        untied = dataclasses.replace(blm, curvature=bopt.CurvatureState(
                            blm.curvature.misfit_gradient,
                            dataclasses.replace(gd, lift_factors=ufo.ufv_lift_factors(blm.frame, 1e-6))))
                        p = geom.randn(m.frame)
                        self.assertGreater(_relerr_trees(m.curvature_hessian(p).variations.supercores,
                                                         untied.curvature_hvp(p.variations.supercores)), 1e-3)

    def test_gradient_is_the_gauss_newton_gradient_bitwise(self):
        '''m27: the exact model's gradient is the Gauss-Newton model's, bit for bit (with and without a
        regularizer), because it IS the same misfit transpose -- done once, at construction, and read.'''
        from unittest import mock
        import t3toolbox.backend.optimizers as bopt
        rng = np.random.default_rng(73)
        x, sample, data = _ex_problem(_EX_MINIMAL, 2, 8, rng)
        r = _ex_residual(x, sample, data, 2)
        for reg in (None, bopt.IdentityRegularizer(0.3)):
            for geom in (t3m.MANIFOLD, t3m.COREWISE):
                with self.subTest(regularizer=reg is not None, geometry=type(geom).__name__):
                    gn = fitting.probe_derivatives_model(geom, x, *sample, 2, r, weight=[1.0, 0.5, 0.25],
                                                         regularizer=reg)
                    with mock.patch.object(fb.ProbeDerivativesKind, 'transpose', autospec=True,
                                           side_effect=fb.ProbeDerivativesKind.transpose) as spy:
                        ex = fitting.probe_derivatives_model(geom, x, *sample, 2, r, weight=[1.0, 0.5, 0.25],
                                                             regularizer=reg, hessian='exact')
                        self.assertEqual(spy.call_count, 1)                    # the one misfit transpose
                        g_ex = ex.gradient
                        self.assertEqual(spy.call_count, 1)                    # read, not recomputed
                    self.assertTrue(_bitwise(g_ex.variations.data, gn.gradient.variations.data))

    def test_hessian_quadratic_and_evaluate_are_mode_dependent(self):
        '''Gauss-Newton: ``hessian`` / ``quadratic`` ARE ``gn_hessian`` / ``gn_quadratic`` and ``evaluate`` is
        unchanged. Exact: ``hessian = gn_hessian + curvature_hessian``, ``quadratic = gn_quadratic +
        curvature_quadratic = ⟨p, hessian(p)⟩``, ``evaluate(p) = c + ⟨g, p⟩ + ½ quadratic(p)``; the Gauss-Newton
        members mean the same in both modes.'''
        import t3toolbox.backend.optimizers as bopt
        rng = np.random.default_rng(74)
        x, sample, data = _ex_problem(_EX_MINIMAL, 1, 8, rng)
        r = _ex_residual(x, sample, data, 1)
        R = bopt.IdentityRegularizer(0.2)
        gn = fitting.probe_derivatives_model(t3m.MANIFOLD, x, *sample, 1, r, regularizer=R)
        ex = fitting.probe_derivatives_model(t3m.MANIFOLD, x, *sample, 1, r, regularizer=R, hessian='exact')
        p = t3m.MANIFOLD.randn(gn.frame)
        self.assertTrue(_bitwise(gn.hessian(p).variations.data, gn.gn_hessian(p).variations.data))
        self.assertEqual(float(gn.quadratic(p)), float(gn.gn_quadratic(p)))
        self.assertEqual(float(gn.evaluate(p)), float(gn.objective_value + gn.gradient.corewise_inner(p)
                                                      + 0.5 * gn.kind.sumsq(gn.jacobian(p), gn._n_w)
                                                      + 0.5 * R.quadratic(gn._bgeom, gn.frame.data,
                                                                          p.variations.data)))
        for member in ('gn_hessian', 'jacobian'):
            self.assertTrue(_bitwise(getattr(ex, member)(p) if member == 'jacobian'
                                     else getattr(ex, member)(p).variations.data,
                                     getattr(gn, member)(p) if member == 'jacobian'
                                     else getattr(gn, member)(p).variations.data))
        self.assertEqual(float(ex.gn_quadratic(p)), float(gn.gn_quadratic(p)))
        self.assertEqual(float(ex.objective_value), float(gn.objective_value))
        Hp = ex.hessian(p)
        self.assertTrue(_bitwise(Hp.variations.data, (ex.gn_hessian(p) + ex.curvature_hessian(p)).variations.data))
        q = float(ex.quadratic(p))
        self.assertEqual(q, float(ex.gn_quadratic(p) + ex.curvature_quadratic(p)))
        self.assertAlmostEqual(q, float(p.corewise_inner(Hp)), delta=1e-12 * abs(q))
        self.assertGreater(abs(q - float(gn.quadratic(p))), 1e-3 * abs(q))     # the curvature term is there
        self.assertAlmostEqual(float(ex.evaluate(p)),
                               float(ex.objective_value + ex.gradient.corewise_inner(p)) + 0.5 * q,
                               delta=1e-12 * abs(float(ex.evaluate(p))))

    def test_hessian_against_a_finite_difference_of_the_riemannian_gradient(self):
        '''The ``x07`` cross-check, numpy: ``hessian(p)`` is ``d/dt P_x grad f(R_x(t p))`` at ``t = 0`` -- the
        Riemannian gradient at the retracted point, as an ambient T3 projected onto the tangent space at ``x``,
        differenced centrally along the retraction (exact for any retraction: the Riemannian Hessian of an
        embedded submanifold is ``P_x D(grad f)(x)[p]``). The Gauss-Newton Hessian misses it.'''
        import t3toolbox.backend.optimizers as bopt
        rng = np.random.default_rng(75)
        for name, struct, order, reg in (('minimal', _EX_MINIMAL, 1, None), ('d4', _EX_D4, 2, None),
                                         ('regularized', _EX_MINIMAL, 1, bopt.IdentityRegularizer(0.5))):
            with self.subTest(case=name):
                x, sample, data = _ex_problem(struct, order, 8, rng)

                def model_at(y, hessian='gauss_newton'):
                    return fitting.probe_derivatives_model(t3m.MANIFOLD, y, *sample, order,
                                                           _ex_residual(y, sample, data, order),
                                                           regularizer=reg, hessian=hessian)

                m = model_at(x, 'exact')
                p = t3m.MANIFOLD.randn(m.frame)
                p = p * (1.0 / float(t3m.MANIFOLD.norm(p)))
                h = 0.05 * min(float(np.min(m.curvature.geometry_data.sigma_min)), 1.0) * 1e-2

                def grad_at(t):                     # P_x grad f(R_x(t p)), as gauged coordinates at x's frame
                    g = model_at(m.retract(p * t)).gradient
                    return t3m.MANIFOLD.project_ambient(m.frame, g.to_t3())

                fd = (grad_at(h) - grad_at(-h)) * (1.0 / (2.0 * h))
                exact_err = _relerr_trees(m.hessian(p).variations.data, fd.variations.data)
                gn_err = _relerr_trees(m.gn_hessian(p).variations.data, fd.variations.data)
                self.assertLess(exact_err, 1e-5)
                self.assertGreater(gn_err, 1e3 * exact_err)

    def test_retract_is_the_geometrys_and_the_backend_models(self):
        '''m38: ``retract(p)`` is ``geometry.retract(p)`` at this frame -- a shared manifold model's passes its
        companion (no rebuild) -- and equals the backend ``LocalModel.retract`` bit for bit; the same in both
        modes; ragged and uniform; a trial step at another frame is rejected.'''
        from unittest import mock
        import t3toolbox.shared_geometry as sg
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        import t3toolbox.backend.optimizers as bopt
        import t3toolbox.backend.uniform_fitting as uf
        rng = np.random.default_rng(76)
        shape = _EX_SHAREABLE[0]
        cases = [('manifold', t3m.MANIFOLD, bgeo.ManifoldGeometryOps(), None),
                 ('corewise', t3m.COREWISE, bgeo.CorewiseGeometryOps(), None),
                 ('shared_manifold', sg.shared_manifold((0, 0, 1)),
                  bgeo.ManifoldGeometryOps().with_sharing((0, 0, 1), shape), (0, 0, 1)),
                 ('shared_corewise', sg.shared_corewise((0, 0, 1)),
                  bgeo.CorewiseGeometryOps().with_sharing((0, 0, 1), shape), (0, 0, 1))]
        for name, geom, geom_ops, sharing in cases:
            with self.subTest(case=name):
                x, sample, data = _ex_problem(_EX_SHAREABLE, 1, 7, rng, sharing=(0, 0, 1))
                blm = self._backend_twin(geom_ops, x, sample, data, 1, None)
                m = fitting.probe_derivatives_model(geom, x, *sample, 1, blm.residual)
                p = geom.randn(m.frame) * 0.1
                with mock.patch.object(sg.SharedGeometry, 'shared_frame_data',
                                       wraps=sg.SharedGeometry.shared_frame_data, autospec=True) as spy:
                    y = m.retract(p)
                self.assertEqual(spy.call_count, 0)                                   # the stored companion
                self.assertIsInstance(y, t3.TuckerTensorTrain)
                self.assertTrue(_bitwise(y.data, geom.retract(p).data))
                self.assertTrue(_bitwise(y.data, blm.retract(p.variations.data)))
                ex = fitting.probe_derivatives_model(geom, x, *sample, 1, blm.residual, hessian='exact')
                with mock.patch.object(sg.SharedGeometry, 'shared_frame_data',
                                       wraps=sg.SharedGeometry.shared_frame_data, autospec=True) as spy:
                    y_ex = ex.retract(p)
                self.assertEqual(spy.call_count, 0)
                self.assertTrue(_bitwise(y_ex.data, y.data))
                self.assertTrue(_bitwise(y_ex.data, ex._blm.retract(p.variations.data)))
        other = t3m.MANIFOLD.randn(t3m.MANIFOLD.frame(t3.TuckerTensorTrain.randn(*_EX_SHAREABLE)))
        with self.assertRaises(ValueError):
            m.retract(other)
        # uniform: the uniform geometry's retract, and the backend uniform model's
        x, sample, data = _ex_problem(_EX_MINIMAL, 1, 7, rng)
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        ulm = uf.uniform_least_squares_problem('manifold', 'probe_derivatives', ux, sample, data, 1
                                               ).local_model((ux.data[0], ux.data[1]))
        um = fitting.probe_derivatives_model(ut3m.UNIFORM_MANIFOLD, ux, *sample, 1, ulm.residual)
        up = ut3m.UNIFORM_MANIFOLD.randn(um.frame) * 0.1
        uy = um.retract(up)
        self.assertIsInstance(uy, ut3.UniformTuckerTensorTrain)
        self.assertTrue(_bitwise(uy.data[:2], ut3m.UNIFORM_MANIFOLD.retract(up).data[:2]))
        self.assertTrue(_bitwise(uy.data[:2], ulm.retract(up.variations.supercores)))

    def test_curvature_damping_reaches_the_lift_and_is_rejected_where_it_does_nothing(self):
        '''The damping is handed to the backend geometry unread: the exact model's lift factors are a direct
        ``fv_lift_factors(frame, value)`` call's, bit for bit (``None`` -> ``DEFAULT_CURVATURE_DAMPING``); a value
        with ``hessian='gauss_newton'`` raises with the backend's message, as does one on a corewise geometry
        (shared included) and a negative one.'''
        import t3toolbox.shared_geometry as sg
        import t3toolbox.backend.fv_operations as fvo
        rng = np.random.default_rng(77)
        x, sample, data = _ex_problem(_EX_MINIMAL, 1, 8, rng)
        r = _ex_residual(x, sample, data, 1)
        for cd, ref in ((0.0, 0.0), (1e-3, 1e-3), (None, bgeo.DEFAULT_CURVATURE_DAMPING)):
            with self.subTest(curvature_damping=cd):
                m = fitting.probe_derivatives_model(t3m.MANIFOLD, x, *sample, 1, r, hessian='exact',
                                                    curvature_damping=cd)
                lf, direct = m.curvature.geometry_data.lift_factors, fvo.fv_lift_factors(m.frame.data, ref)
                self.assertTrue(_bitwise(lf.tucker_pinvs + lf.bond_pinvs + (lf.eps,),
                                         direct.tucker_pinvs + direct.bond_pinvs + (direct.eps,)))
        for geom in (t3m.MANIFOLD, t3m.COREWISE):
            with self.subTest(gauss_newton=type(geom).__name__):
                with self.assertRaisesRegex(ValueError, "probe_derivatives_model: curvature_damping is only "
                                                        "meaningful with hessian='exact'"):
                    fitting.probe_derivatives_model(geom, x, *sample, 1, r, curvature_damping=1e-6)
        with self.assertRaisesRegex(ValueError, 'curvature_damping must be >= 0'):
            fitting.probe_derivatives_model(t3m.MANIFOLD, x, *sample, 1, r, hessian='exact', curvature_damping=-1.0)
        xs, ssample, sdata = _ex_problem(_EX_SHAREABLE, 1, 8, rng, sharing=(0, 0, 1))
        rs = _ex_residual(xs, ssample, sdata, 1)
        for name, geom, xx, ss, rr in (('corewise', t3m.COREWISE, x, sample, r),
                                       ('shared_corewise', sg.shared_corewise((0, 0, 1)), xs, ssample, rs)):
            with self.subTest(exact=name):
                self.assertTrue(fitting.probe_derivatives_model(geom, xx, *ss, 1, rr, hessian='exact').is_exact)
                for cd in (0.0, 1e-6):
                    with self.assertRaisesRegex(ValueError, 'no effect on a corewise geometry'):
                        fitting.probe_derivatives_model(geom, xx, *ss, 1, rr, hessian='exact', curvature_damping=cd)

    def test_shared_corewise_matches_the_backend(self):
        '''Shared COREWISE exact mode is live (the tied Hessian): the frontend model equals the backend one.'''
        import t3toolbox.shared_geometry as sg
        rng = np.random.default_rng(78)
        x, sample, data = _ex_problem(_EX_SHAREABLE, 1, 8, rng, sharing=(0, 0, 1))
        blm = self._backend_twin(bgeo.CorewiseGeometryOps().with_sharing((0, 0, 1), _EX_SHAREABLE[0]),
                                 x, sample, data, 1, None, hessian='exact')
        m = fitting.probe_derivatives_model(sg.shared_corewise((0, 0, 1)), x, *sample, 1, blm.residual,
                                            hessian='exact')
        p = sg.shared_corewise((0, 0, 1)).randn(m.frame)
        self.assertTrue(_bitwise(m.gradient.variations.data, blm.gradient))
        self.assertLess(_relerr_trees(m.hessian(p).variations.data, blm.hvp(p.variations.data)), 1e-12)
        self.assertTrue(np.array_equal(np.asarray(m.quadratic(p)), np.asarray(blm.quadratic(p.variations.data))))

    def test_structural_errors_where_the_exact_hessian_is_not_built_yet(self):
        '''The kinds without ``corewise_hessian`` raise the backend's structural ``ValueError`` at construction
        (naming the kind). Gauss-Newton is unaffected. (The shared manifold's exact mode -- the tied lift -- is
        built: S7; the unshared uniform layer's -- the masked uniform two-hole -- S8; the SHARED uniform
        geometries' -- the uniform tied lift -- S9.)'''
        import t3toolbox.shared_geometry as sg
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        rng = np.random.default_rng(79)
        x = t3.TuckerTensorTrain.randn(*_EX_MINIMAL)
        ww = [rng.standard_normal((6, N)) for N in x.shape]
        pp = [rng.standard_normal((6, N)) for N in x.shape]
        idx = np.stack([rng.integers(0, N, size=6) for N in x.shape])
        builds = {'apply': (fitting.apply_model, (ww, x.apply(ww)), 'ApplyKind'),
                  'entries': (fitting.entries_model, (idx, x.entries(idx)), 'EntriesKind'),
                  'probe': (fitting.probe_model, (ww, x.probe(ww)), 'ProbeKind'),
                  'apply_derivatives': (fitting.apply_derivatives_model,
                                        (ww, pp, 1, x.apply_derivatives(ww, pp, 1)), 'ApplyDerivativesKind'),
                  'entries_derivatives': (fitting.entries_derivatives_model,
                                          (idx, pp, 1, x.entries_derivatives(idx, pp, 1)), 'EntriesDerivativesKind')}
        for name, (factory, args, kind_name) in builds.items():
            with self.subTest(kind=name):
                self.assertFalse(factory(t3m.MANIFOLD, x, *args).is_exact)
                with self.assertRaisesRegex(ValueError, kind_name + ' does not implement corewise_hessian'):
                    factory(t3m.MANIFOLD, x, *args, hessian='exact')
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        r = x.probe_derivatives(ww, pp, 1)
        for geom in (ut3m.UNIFORM_MANIFOLD, ut3m.UNIFORM_COREWISE):
            with self.subTest(uniform=type(geom).__name__):
                self.assertFalse(fitting.probe_derivatives_model(geom, ux, ww, pp, 1, r).is_exact)
                self.assertTrue(fitting.probe_derivatives_model(geom, ux, ww, pp, 1, r, hessian='exact').is_exact)
        with self.assertRaisesRegex(ValueError, 'UniformApplyDerivativesKind does not implement corewise_hessian'):
            fitting.apply_derivatives_model(ut3m.UNIFORM_MANIFOLD, ux, ww, pp, 1, x.apply_derivatives(ww, pp, 1),
                                            hessian='exact')
        xt = t3.TuckerTensorTrain(*bsharing.t3_tie_tucker_factors(
            t3.TuckerTensorTrain.randn(*_EX_SHAREABLE).data, (0, 0, 1)))
        uxt = ut3.UniformTuckerTensorTrain.from_t3(xt)
        wws, pps = [rng.standard_normal((6, N)) for N in xt.shape], [rng.standard_normal((6, N)) for N in xt.shape]
        rt = xt.probe_derivatives(wws, pps, 1)
        for base in (ut3m.UNIFORM_MANIFOLD, ut3m.UNIFORM_COREWISE):
            with self.subTest(shared_uniform=type(base).__name__):
                ugeom = sg.shared(base, (0, 0, 1))
                self.assertFalse(fitting.probe_derivatives_model(ugeom, uxt, wws, pps, 1, rt).is_exact)
                self.assertTrue(fitting.probe_derivatives_model(ugeom, uxt, wws, pps, 1, rt, hessian='exact').is_exact)
        xs, ssample, sdata = _ex_problem(_EX_SHAREABLE, 1, 8, rng, sharing=(0, 0, 1))
        rs = _ex_residual(xs, ssample, sdata, 1)
        geom = sg.shared_manifold((0, 0, 1))
        self.assertFalse(fitting.probe_derivatives_model(geom, xs, *ssample, 1, rs).is_exact)
        self.assertTrue(fitting.probe_derivatives_model(geom, xs, *ssample, 1, rs, hessian='exact').is_exact)

    def test_shared_manifold_matches_the_backend(self):
        '''S7 (the DEFERRED S6 gate): the exact frontend model on a shared MANIFOLD geometry is the backend exact
        model on the same data -- ``gradient``, ``curvature_hessian``, ``curvature_quadratic``, ``quadratic`` and
        ``evaluate`` bit for bit, ``hessian`` to rounding (two gauged halves vs one projection), ``retract`` the
        backend's -- and its backend view runs on the model's ONE companion (``_blm.geom_aux`` is
        ``geometry_aux``): no curvature action rebuilds it. On the group-ceiling structure (mode 0's own ``S_0^T``
        wide) the tied lift is load-bearing, so a model whose curvature used the per-mode lift would differ.'''
        import dataclasses
        from unittest import mock
        import t3toolbox.shared_geometry as sg
        rng = np.random.default_rng(81)
        cases = [('shareable', _EX_SHAREABLE, (0, 0, 1), 1, None),
                 ('ceiling', ((6, 6, 4), (4, 4, 2), (1, 2, 2, 1)), (0, 0, 1), 2, 'row'),
                 ('all_modes', ((5, 5, 5), (3, 3, 3), (1, 2, 2, 1)), (0, 0, 0), 1, None)]
        for name, struct, labels, order, wkind in cases:
            with self.subTest(case=name):
                x, sample, data = _ex_problem(struct, order, 7, rng, sharing=labels)
                w = None if wkind is None else rng.uniform(0.3, 2.0, order + 1)
                geom = sg.shared_manifold(labels)
                blm = self._backend_twin(bgeo.ManifoldGeometryOps().with_sharing(labels, struct[0]),
                                         x, sample, data, order, w, hessian='exact')
                m = fitting.probe_derivatives_model(geom, x, *sample, order, blm.residual, weight=w, hessian='exact')
                self.assertTrue(m.is_exact)
                self.assertIs(m._blm.geom_aux, m.geometry_aux)
                self.assertIs(m.curvature.geometry_data.lift_factors.shared_data, m.geometry_aux)
                self.assertTrue(_bitwise(m.frame.data, blm.frame))
                self.assertTrue(_bitwise(m.gradient.variations.data, blm.gradient))
                self.assertTrue(_bitwise(m.curvature.misfit_gradient, blm.curvature.misfit_gradient))
                for K in ((), (2,)):
                    p = geom.randn(m.frame, stack_shape=K)
                    raw = p.variations.data
                    with mock.patch.object(bsharing, 'fv_shared_frame_data',
                                           wraps=bsharing.fv_shared_frame_data) as spy:
                        Hc, qc, q = m.curvature_hessian(p), m.curvature_quadratic(p), m.quadratic(p)
                        H = m.hessian(p)
                        ev = m.evaluate(p) if K == () else None                    # evaluate is unstacked in p
                    self.assertEqual(spy.call_count, 0)                                # the stored companion
                    self.assertTrue(_bitwise(Hc.variations.data, blm.curvature_hvp(raw)))
                    self.assertTrue(np.array_equal(np.asarray(qc), np.asarray(blm.curvature_quadratic(raw))))
                    self.assertTrue(np.array_equal(np.asarray(q), np.asarray(blm.quadratic(raw))))
                    self.assertLess(_relerr_trees(H.variations.data, blm.hvp(raw)), 1e-12)
                    if ev is not None:
                        self.assertAlmostEqual(float(ev), float(m.objective_value + m.gradient.corewise_inner(p))
                                               + 0.5 * float(q), delta=1e-12 * abs(float(ev)))
                p = geom.randn(m.frame) * 0.1
                self.assertTrue(_bitwise(m.retract(p).data, blm.retract(p.variations.data)))
                if name == 'ceiling':                                                   # the tied lift matters
                    import t3toolbox.backend.fv_operations as fvo
                    import t3toolbox.backend.optimizers as bopt
                    gd = blm.curvature.geometry_data
                    untied = dataclasses.replace(blm, curvature=bopt.CurvatureState(blm.curvature.misfit_gradient,
                        dataclasses.replace(gd, lift_factors=fvo.fv_lift_factors(blm.frame, 0.0))))
                    p = geom.randn(m.frame)
                    self.assertGreater(_relerr_trees(m.curvature_hessian(p).variations.data,
                                                     untied.curvature_hvp(p.variations.data)), 1e-3)

    def test_d1_manifold_is_gauss_newton(self):
        '''At d = 1 the manifold's curvature term vanishes (every slot is at maximal rank): the exact model is
        exact, with no curvature data, and its hessian / quadratic are the Gauss-Newton ones.'''
        rng = np.random.default_rng(80)
        x, sample, data = _ex_problem(((5,), (3,), (1, 1)), 1, 8, rng)
        m = fitting.probe_derivatives_model(t3m.MANIFOLD, x, *sample, 1, _ex_residual(x, sample, data, 1),
                                            hessian='exact')
        self.assertTrue(m.is_exact)
        self.assertIsNone(m.curvature.geometry_data)
        p = t3m.MANIFOLD.randn(m.frame)
        self.assertTrue(_bitwise(m.hessian(p).variations.data, m.gn_hessian(p).variations.data))
        self.assertEqual(float(m.quadratic(p)), float(m.gn_quadratic(p)))
        self.assertEqual(float(m.curvature_quadratic(p)), 0.0)
        self.assertTrue(all(not np.any(a) for a in _flat(m.curvature_hessian(p).variations.data)))



class TestFiniteDifferenceHessianCheckInSafeMode(unittest.TestCase):
    """S7b (I-38, the S0b report's section 5 finding): ``MANIFOLD.norm(H_fd - hessian(p))`` -- the
    finite-difference verification of the exact Hessian along the retraction -- used to raise in safe mode,
    because the near-cancelling difference was gauge-checked against its own tiny norm. Measured against its
    operands' scale it passes, at every step size."""

    def setUp(self):
        np.random.seed(0)

    def test_fd_minus_exact_hessian_passes_safe_mode(self):
        import t3toolbox.safety as safety
        import t3toolbox.backend.tv_operations as tvo
        M = t3m.MANIFOLD
        np.random.seed(0)
        shape, ranks = (4, 5, 6), ((2, 2, 2), (1, 2, 2, 1))
        ww, pp = [np.random.randn(20, N) for N in shape], [np.random.randn(20, N) for N in shape]
        y = t3.TuckerTensorTrain.randn(shape, *ranks).probe_derivatives(ww, pp, 1)

        def model_at(x, hessian='gauss_newton'):
            r = [z - yi for z, yi in zip(x.probe_derivatives(ww, pp, 1), y)]
            return fitting.probe_derivatives_model(M, x, ww, pp, 1, r, hessian=hessian)

        mx = model_at(t3.TuckerTensorTrain.randn(shape, *ranks), 'exact')
        self.assertIsNotNone(safety.current_safety())                            # safe mode (the default)
        for name, p in (('steepest', mx.gradient * -1.0), ('random', M.randn(mx.frame))):
            p = p * (1.0 / float(M.norm(p)))
            Hp = mx.hessian(p)
            for h in (1e-4, 1e-6):
                with self.subTest(direction=name, h=h):
                    gp = M.project_ambient(mx.frame, model_at(M.retract(p * h)).gradient.to_t3())
                    gm = M.project_ambient(mx.frame, model_at(M.retract(p * (-h))).gradient.to_t3())
                    D = (gp - gm) * (1.0 / (2 * h)) - Hp
                    self.assertLess(float(D.corewise_norm() / Hp.corewise_norm()), 1e-5)     # FD agrees
                    # discriminating: self-relative, the difference looks ungauged (the pre-S7b raise)
                    self.assertGreater(float(tvo.tv_gauge_residual(mx.frame.data, D.variations.data)), 1e-9)
                    self.assertLess(float(D.gauge_residual), 1e-13)
                    self.assertTrue(np.isclose(float(M.norm(D)), float(D.corewise_norm())))   # no raise


if __name__ == '__main__':
    unittest.main()
