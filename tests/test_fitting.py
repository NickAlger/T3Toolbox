'''Tests for the frontend local model, ``fitting.local_model`` / ``fitting.LocalModel`` (P16: over the oracle).

One ``fitting.LocalModel`` serves every **oracle** (here the least-squares one, of the kinds apply / entries / probe,
plain or along lines) and every **geometry** (manifold / corewise, ragged / uniform, shared). The headline oracle is
**exact dense ground truth**: the sampling forward is linear in the ambient tensor, so the least-squares objective is
exactly quadratic and the Gauss-Newton model is the exact restriction of the objective to the affine tangent space
``r + dense(𝒥 Π p)``, where ``dense(𝒥 Π p)`` is ``geometry.project(p).to_dense()`` for *both* geometries (manifold: a
gauged tangent; corewise: the sum-of-core-swaps at ``(B,G,G,G)``). Derivative data are checked against the independent
dense oracle ``tests/oracle_sweeps/oracle.py`` (exact polynomial interpolation, no library jet code). We also check the
two-form consistency, the razor self-containment (raw ``p`` == projected ``Π p``), the matched pair (manifold gauges,
corewise does not), GN symmetry, the ``J`` / ``Jᵀ`` adjoint against the ``T3Tangent`` transposes, the residual weights,
and that the frontend model is its backend twin (``bopt.local_model`` on the same oracle and batch, built here).
'''

import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.shared_geometry as sg
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.sharing as bsharing
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.uniform_fitting as ufit
import t3toolbox.fitting as fitting
import t3toolbox.corewise as cw
from t3toolbox.fitting import LeastSquaresOracle, LeastSquaresBatch
from t3toolbox.backend.jets import blocks_to_weight
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'oracle_sweeps'))
import _derivative_data as dd       # derivative data from the merged methods (P13b)
import _fits as F                   # the 2026.5.0 vocabulary (kind names, derivative data) in the new form (P16)
import oracle as dense_oracle       # independent dense oracles for the six old kinds (no library jet code)

SHAPE = (7, 8, 9)
TUCKER_RANKS = (3, 4, 2)
TT_RANKS = (1, 2, 3, 1)
N_SAMPLES = 20
KINDS = ('apply', 'entries', 'probe')
GEOMS = {'manifold': t3m.MANIFOLD, 'corewise': t3m.COREWISE}
# tol: manifold matches to ~1e-13, corewise to ~1e-9 (large raw-core magnitudes) -> a relative band.
RTOL, ATOL = 1e-9, 1e-9


def _backend_geometry(geometry, x):
    '''The backend geometry of a frontend one at ``x``, built here, independently of ``fitting.backend_geometry``
    (uniform: ``from_point``, so ``x`` must be minimal).'''
    sharing = geometry.sharing if isinstance(geometry, sg.SharedGeometry) else None
    base = geometry.base if isinstance(geometry, sg.SharedGeometry) else geometry
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        cls = bgeo.UniformManifoldGeometryOps if base is ut3m.UNIFORM_MANIFOLD else bgeo.UniformCorewiseGeometryOps
        return cls.from_point(x.data, sharing)
    geom = bgeo.ManifoldGeometryOps() if base is t3m.MANIFOLD else bgeo.CorewiseGeometryOps()
    return geom if sharing is None else geom.with_sharing(sharing, x.shape)


def _backend_model(geometry, oracle, batch, x, **kw):
    '''The backend twin of ``fitting.local_model(geometry, oracle, batch, x, **kw)``, built here from the backend pieces
    (the batch packed for a uniform ``x``).'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        batch = ufit.pack_batch(oracle.kind, oracle.order, batch, x.N)
    return bopt.local_model(_backend_geometry(geometry, x), oracle, batch, (x.data[0], x.data[1]), **kw)


def _model(geometry, x, kind, sample, data, order=None, weight=None, **kw):
    '''The frontend model of a fit stated in the 2026.5.0 vocabulary (``F.old_vocabulary``: derivative data along lines,
    a derivative-convention weight).'''
    oracle, batch = F.old_vocabulary(kind, sample, data, tuple(x.shape), order, weight)
    return fitting.local_model(geometry, oracle, batch, x, **kw)


def _pair(geometry, x, kind, sample, data, order=None, weight=None, **kw):
    '''(the frontend model, its backend twin) on the same oracle and batch.'''
    oracle, batch = F.old_vocabulary(kind, sample, data, tuple(x.shape), order, weight)
    return (fitting.local_model(geometry, oracle, batch, x, **kw),
            _backend_model(geometry, oracle, batch, x, **kw))


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


_TRANSPOSE = {'apply': t3m.T3Tangent.apply_transpose, 'entries': t3m.T3Tangent.entries_transpose,
              'probe': t3m.T3Tangent.probe_transpose}


def _setup(kind, geom_name, C):
    '''Build a model of one (kind, geometry, frame-stack C), plus the dense forward oracle and the kind-aware
    sample-space reducers (samp_add / samp_dot / rand_like). The data ``y`` are one dataset shared by the stack
    (size-1 ``C`` axes, of the type of ``z``); each element's residual is ``r_c = z(x_c) - y``, from the dense
    forward. One dict per call.'''
    np.random.seed(0)
    x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS, stack_shape=C)
    geometry = GEOMS[geom_name]
    m, n_c, n_w = N_SAMPLES, len(C), 1
    if kind == 'apply':
        sample = [np.random.randn(m, N) for N in SHAPE]
        dense_fwd = lambda T: apply_dense(T, sample, n_c)
    elif kind == 'entries':
        sample = np.stack([np.random.randint(0, N, size=m) for N in SHAPE])   # (d,)+W
        dense_fwd = lambda T: entries_dense(T, sample, n_c)
    else:  # probe -- vector-valued (one free mode each)
        sample = [np.random.randn(m, N) for N in SHAPE]
        dense_fwd = lambda T: dense_probe(T, sample, n_c)
    z = dense_fwd(x.to_dense())
    if kind == 'probe':
        y = [np.random.randn(m, N).reshape((m,) + (1,) * n_c + (N,)) for N in SHAPE]   # shared by C: size-1 axes
        r = [zi - yi for zi, yi in zip(z, y)]
        samp_dot = lambda a, b: sum(np.sum(ai * bi, axis=tuple(range(n_w)) + (ai.ndim - 1,))
                                    for ai, bi in zip(a, b))
        samp_add = lambda a, b: [ai + bi for ai, bi in zip(a, b)]
        rand_like = lambda v: [np.random.randn(*vi.shape) for vi in v]
    else:
        y = np.random.randn(m).reshape((m,) + (1,) * n_c)                          # shared by C: size-1 axes
        r = z - y
        samp_dot = lambda a, b: np.sum(a * b, axis=tuple(range(n_w)))
        samp_add = lambda a, b: a + b
        rand_like = lambda v: np.random.randn(*v.shape)
    oracle, batch = LeastSquaresOracle(kind, ()), LeastSquaresBatch(sample, y)
    model = fitting.local_model(geometry, oracle, batch, x)
    return dict(x=x, geometry=geometry, geom_name=geom_name, model=model, frame=model.frame, oracle=oracle,
                batch=batch, kind=kind, sample=sample, y=y, r=r, c=0.5 * samp_dot(r, r), n_c=n_c, n_w=n_w,
                dense_fwd=dense_fwd, samp_dot=samp_dot, samp_add=samp_add, rand_like=rand_like)


def _raw_step(s):
    '''A raw (any-gauge) trial tangent at the model's frame -- ungauged on the manifold (tests Π),
    a raw core perturbation on the corewise frame.'''
    return t3m.COREWISE.randn(s['frame'])


class TestLocalModel(unittest.TestCase):
    '''The generic GN model, parameterized over the sampling kind, the geometry, and the frame stack C.'''

    def test_dense_truth(self):
        '''HEADLINE: model.evaluate(p) == ½‖r + 𝒥(Πp)‖² from the dense projected tangent (both geometries), and the
        objective is ½‖r‖² with r from the dense forward, per element of a frame stack C (one y for all).'''
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
                        self.assertTrue(np.allclose(s['model'].objective_value, s['c'], rtol=RTOL, atol=ATOL))
                        self.assertEqual(np.shape(s['model'].objective_value), C)

    def test_two_form_consistency(self):
        '''m(p) == c + ⟨g, p⟩ + ½⟨p, H p⟩ (the gn_hessian-based quadratic term), both geometries.'''
        for kind in KINDS:
            for geom_name in GEOMS:
                for C in [(), (2,)]:
                    with self.subTest(kind=kind, geom=geom_name, C=C):
                        s = _setup(kind, geom_name, C)
                        model, p = s['model'], _raw_step(s)
                        two_form = (model.objective_value + model.gradient.corewise_inner(p)
                                    + 0.5 * p.corewise_inner(model.gn_hessian(p)))
                        self.assertTrue(np.allclose(model.evaluate(p), two_form, rtol=RTOL, atol=ATOL))

    def test_regularizer(self):
        '''An IdentityRegularizer folds ρ=½λ‖X‖² into the frontend LocalModel on all kinds/geometries:
        the two-form identity `m(p)=c+⟨g,p⟩+½⟨p,Hp⟩` and `gn_quadratic==pᵀHp` still hold (reg folded
        consistently across all five methods) and the objective gains ρ.'''
        lam = 0.4
        BOPS = {'manifold': bgeo.ManifoldGeometryOps(), 'corewise': bgeo.CorewiseGeometryOps()}
        for kind in KINDS:
            for geom_name in GEOMS:
                with self.subTest(kind=kind, geom=geom_name):
                    s = _setup(kind, geom_name, ())
                    R = bopt.IdentityRegularizer(lam)
                    model = fitting.local_model(s['geometry'], s['oracle'], s['batch'], s['x'], regularizer=R)
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
        ragged model's exactly (the equivalence contract).'''
        np.random.seed(0)
        R = bopt.IdentityRegularizer(0.4)
        Xr = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))   # minimal / realizable ranks
        Xu = ut3.UniformTuckerTensorTrain.from_t3(Xr)
        ww = [np.random.randn(20, N) for N in (6, 7, 8)]
        y = [np.random.randn(20, N) for N in (6, 7, 8)]
        oracle, batch = LeastSquaresOracle('probe', ()), LeastSquaresBatch(ww, y)
        for ug, rg in [(ut3m.UNIFORM_MANIFOLD, t3m.MANIFOLD), (ut3m.UNIFORM_COREWISE, t3m.COREWISE)]:
            with self.subTest(geom=type(ug).__name__):
                um = fitting.local_model(ug, oracle, batch, Xu, regularizer=R)
                rm = fitting.local_model(rg, oracle, batch, Xr, regularizer=R)
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
                        self.assertTrue(model.gn_hessian(p).allclose(model.gn_hessian(Pp), rtol=RTOL,
                                                                     atol=ATOL).all())

    def test_matched_pair(self):
        '''The structural matched pair: MANIFOLD gauges (g, H gauged); COREWISE does NOT (g == the bare 𝒥ᵀr, the
        ``T3Tangent`` transpose at the corewise frame ``(B, G, G, G)``, with r from the dense forward).'''
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    sm = _setup(kind, 'manifold', C)
                    self.assertTrue(sm['model'].gradient.is_gauged().all())
                    self.assertTrue(sm['model'].gn_hessian(_raw_step(sm)).is_gauged().all())

                    sc = _setup(kind, 'corewise', C)
                    bare = _TRANSPOSE[kind](sc['frame'], sc['sample'], residual=sc['r']).variations.data
                    gd = sc['model'].gradient.variations.data
                    for a, b in zip(gd[0] + gd[1], bare[0] + bare[1]):   # corewise gradient == bare 𝒥ᵀr, no Π
                        self.assertTrue(np.allclose(a, b, rtol=1e-10, atol=1e-10))

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
        '''Adjoint ⟨z, 𝒥 Π p⟩_samples == ⟨Π 𝒥ᵀ z, p⟩_corewise (J = 𝒥∘Π; 𝒥ᵀ the ``T3Tangent`` transpose at the model's
        frame, then the geometry's Π), both geometries, a frame stack C included.'''
        for kind in KINDS:
            for geom_name in GEOMS:
                for C in [(), (2,)]:
                    with self.subTest(kind=kind, geom=geom_name, C=C):
                        s = _setup(kind, geom_name, C)
                        model, geometry, p = s['model'], s['geometry'], _raw_step(s)
                        Jp = model.jacobian(p)                                    # J p = 𝒥(Π p)
                        z = s['rand_like'](Jp)
                        gz = geometry.project(_TRANSPOSE[kind](s['frame'], s['sample'], residual=z))   # Π 𝒥ᵀ z
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

    def test_residual(self):
        '''``residual`` is ``z(x) - y`` against the dense forward, per element of a frame stack C.'''
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    s = _setup(kind, 'manifold', C)
                    seq = lambda v: list(v) if isinstance(v, (list, tuple)) else [v]
                    for a, b in zip(seq(s['model'].residual), seq(s['r'])):
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
        '''The frame, the gradient and the objective are cached (computed once); the build's query is held.'''
        model = _setup('apply', 'manifold', ())['model']
        self.assertIs(model.frame, model.frame)
        self.assertIs(model.gradient, model.gradient)
        self.assertIs(model.objective_value, model.objective_value)
        self.assertIs(model.held, model.backend.held)

    def test_matches_established_manifold(self):
        '''Manifold gradient/Hessian == the established bare T3Tangent transpose + MANIFOLD.project (apply).'''
        s = _setup('apply', 'manifold', ())
        frame, ww, r, model = s['frame'], s['sample'], s['r'], s['model']
        ref_g = t3m.MANIFOLD.project(t3m.T3Tangent.apply_transpose(frame, ww, residual=r))
        gd = model.gradient.variations.data
        for a, b in zip(gd[0] + gd[1], ref_g.variations.data[0] + ref_g.variations.data[1]):
            self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-11))
        V = t3m.MANIFOLD.randn(frame)
        ref_HV = t3m.MANIFOLD.project(t3m.T3Tangent.apply_transpose(frame, ww, residual=V.apply(ww)))
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
                    g_ref = getattr(s['x'], ref_method[kind])(s['sample'], residual=s['r'])
                    gd = s['model'].gradient.variations.data
                    for a, b in zip(gd[0] + gd[1], g_ref[0] + g_ref[1]):
                        self.assertTrue(np.allclose(a, b, rtol=1e-10, atol=1e-10))

    def test_derivative_models(self):
        '''Fits of derivative data along lines (apply / entries / probe at order 2, a per-order weight ω in the
        derivatives' convention, through ``prepare_derivative_data``): the objective is the dense oracle's
        ``½ Σ_t ω_t² ‖D_t(x) - y_t‖²`` (no library jet code), and ``evaluate(t p)`` is the dense objective at
        ``x + t dense(Π p)`` for ``t = 1, 2`` (the gradient and the Gauss-Newton term against ground truth), both
        geometries; and the frontend model is its backend twin.'''
        rng = np.random.default_rng(0)
        shape, order, NW = (5, 6, 7), 2, 15
        omega = np.array([1.0, 0.5, 0.3])
        X = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
        other = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)).to_dense()
        ww = [rng.standard_normal((NW, N)) for N in shape]
        pp = [rng.standard_normal((NW, N)) for N in shape]
        index = np.stack([rng.integers(0, N, size=NW) for N in shape], axis=0)
        cases = [('apply_derivatives', (ww, pp)), ('entries_derivatives', (index, pp)),
                 ('probe_derivatives', (ww, pp))]
        Xd = X.to_dense()
        for geom in (t3m.MANIFOLD, t3m.COREWISE):
            for kind, sample in cases:
                with self.subTest(geom=type(geom).__name__, kind=kind):
                    clean = dense_oracle.S(kind, other, sample, order)
                    data = ([z + 0.1 * rng.standard_normal(z.shape) for z in clean] if kind.startswith('probe')
                            else clean + 0.1 * rng.standard_normal(clean.shape))
                    fmodel, lm = _pair(geom, X, kind, sample, data, order, omega)
                    truth = dense_oracle.misfit(kind, Xd, sample, data, omega, order)
                    self.assertTrue(np.allclose(float(fmodel.objective_value), truth, rtol=1e-10))
                    pt = geom.randn(fmodel.frame)
                    Pp = geom.project(pt).to_dense()
                    for t in (1.0, 2.0):
                        self.assertTrue(np.allclose(float(fmodel.evaluate(pt * t)),
                                                    dense_oracle.misfit(kind, Xd + t * Pp, sample, data, omega, order),
                                                    rtol=1e-9))
                    self.assertTrue(_bitwise(fmodel.gradient.variations.data, lm.gradient))
                    p = pt.variations.data
                    self.assertEqual(float(fmodel.gn_quadratic(pt)), float(lm.gn_quadratic(p)))
                    self.assertTrue(_bitwise(fmodel.gn_hessian(pt).variations.data, lm.hvp(p)))


class TestResidualWeighting(unittest.TestCase):
    '''The residual weight (``½‖ω⊙r‖²``), in the new form: exactly the shape ``(d,)+F`` (probe), ``F`` (apply), ``()``
    (entries); the derivatives' weight through ``prepare_derivative_data``. Oracles place ``ω[i, t]`` by **explicit
    numpy indexing**: the objective is the dense oracle's (``tests/oracle_sweeps/oracle.py``), and the weighted
    gradient ``𝒥ᵀ(ω²r)`` equals the *unweighted* gradient of data whose residual is the explicitly-scaled ``ω²⊙r``.
    (The 2026.5.0 broadcasting forms -- a bare row, a column, extra rows -- are the translation's, tested in
    ``tests/test_optimizers_frontend.py``.)'''

    def _setup(self, order=2, NW=15):
        rng = np.random.default_rng(0)
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS)
        other = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS).to_dense()
        ww = [rng.standard_normal((NW, N)) for N in SHAPE]
        pp = [rng.standard_normal((NW, N)) for N in SHAPE]
        index = np.stack([rng.integers(0, N, size=NW) for N in SHAPE], axis=0)
        data = {kind: dense_oracle.S(kind, other, sample, order) for kind, sample in
                (('probe_derivatives', (ww, pp)), ('apply_derivatives', (ww, pp)),
                 ('entries_derivatives', (index, pp)))}
        return dict(x=x, xd=x.to_dense(), ww=ww, pp=pp, index=index, d=len(SHAPE), order=order, data=data)

    def test_probe_derivatives_full_matrix(self):
        '''probe derivatives with a full ω[mode, order] matrix: the dense objective, and the ω²r-scaled gradient
        identity, both geometries; the Hessian consistency ``pᵀHp == ‖ω⊙Jp‖²`` with ``Jp`` in the derivatives'
        convention (slot ``t`` times ``t!``).'''
        import math
        s = self._setup()
        d, order, kind, sample = s['d'], s['order'], 'probe_derivatives', (s['ww'], s['pp'])
        y = s['data'][kind]
        rng = np.random.default_rng(1)
        W = rng.uniform(0.2, 2.0, size=(d, order + 1))           # a genuine full matrix
        Dx = dense_oracle.S(kind, s['xd'], sample, order)         # the derivatives at x, dense
        r = [a - b for a, b in zip(Dx, y)]
        y2 = [Dx[i] - W[i].reshape((order + 1, 1, 1)) ** 2 * r[i] for i in range(d)]   # residual ω²⊙r, unweighted
        fact = np.array([math.factorial(t) for t in range(order + 1)], dtype=float).reshape((order + 1, 1, 1))
        for geom in (t3m.MANIFOLD, t3m.COREWISE):
            with self.subTest(geom=type(geom).__name__):   # name, not the object -- xdist reports must serialize (execnet)
                w = _model(geom, s['x'], kind, sample, y, order, W)
                self.assertTrue(np.allclose(float(w.objective_value),
                                            dense_oracle.misfit(kind, s['xd'], sample, y, W, order)))
                u2 = _model(geom, s['x'], kind, sample, y2, order)
                self.assertTrue(w.gradient.allclose(u2.gradient, rtol=1e-9, atol=1e-11).all())
                p = t3m.COREWISE.randn(w.frame)
                Jp = w.jacobian(p)                                # Taylor coefficients 𝒥Πp (a list of d arrays)
                q_oracle = sum(np.sum((W[i].reshape((order + 1, 1, 1)) * fact * Jp[i]) ** 2) for i in range(d))
                self.assertTrue(np.allclose(float(w.gn_quadratic(p)), q_oracle))
                self.assertTrue(np.allclose(float(w.gn_quadratic(p)), float(p.corewise_inner(w.gn_hessian(p)))))

    def test_apply_entries_order_only(self):
        '''apply / entries derivatives take an ORDER-only weight (the dense ½Σ_t (ω_t r_t)²); a per-mode weight
        ``(d, order+1)`` is a structural error (no mode axis), raised by the batch check before any work.'''
        s = self._setup()
        order, geom = s['order'], t3m.MANIFOLD
        rng = np.random.default_rng(3)
        omega = rng.uniform(0.3, 1.5, size=order + 1)
        for kind, sample in (('apply_derivatives', (s['ww'], s['pp'])),
                             ('entries_derivatives', (s['index'], s['pp']))):
            with self.subTest(kind=kind):
                model = _model(geom, s['x'], kind, sample, s['data'][kind], order, omega)
                self.assertTrue(np.allclose(float(model.objective_value),
                                            dense_oracle.misfit(kind, s['xd'], sample, s['data'][kind], omega,
                                                                order)))
                with self.assertRaises(ValueError):
                    _model(geom, s['x'], kind, sample, s['data'][kind], order, np.ones((s['d'], order + 1)))

    def test_plain_probe_per_mode(self):
        '''Plain probe per-mode weight (d,): the dense ½Σ_i ω_i²‖r_i‖² + the ω²r-scaled gradient identity, both
        geometries.'''
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS)
        ww = [np.random.randn(N_SAMPLES, N) for N in SHAPE]
        y = [np.random.randn(N_SAMPLES, N) for N in SHAPE]
        d = len(SHAPE)
        omega = np.array([0.5, 2.0, 1.3])
        z = dense_probe(x.to_dense(), ww, 0)
        r = [a - b for a, b in zip(z, y)]
        y2 = [z[i] - omega[i] ** 2 * r[i] for i in range(d)]      # residual ω²⊙r, unweighted
        oracle = LeastSquaresOracle('probe', ())
        for geom in (t3m.MANIFOLD, t3m.COREWISE):
            with self.subTest(geom=type(geom).__name__):   # name, not the object -- xdist reports must serialize (execnet)
                w = fitting.local_model(geom, oracle, LeastSquaresBatch(ww, y, blocks_to_weight('probe', (), omega, y)),
                                        x)
                obj_oracle = 0.5 * sum((omega[i] ** 2) * np.sum(r[i] ** 2) for i in range(d))
                self.assertTrue(np.allclose(float(w.objective_value), obj_oracle))
                u = fitting.local_model(geom, oracle, LeastSquaresBatch(ww, y2), x)
                self.assertTrue(w.gradient.allclose(u.gradient, rtol=1e-9, atol=1e-11).all())
                p = t3m.COREWISE.randn(w.frame)
                self.assertTrue(np.allclose(float(w.gn_quadratic(p)), float(p.corewise_inner(w.gn_hessian(p)))))


class TestPerModeWeightRowCount(unittest.TestCase):
    """Review 2026-08-22 (S8), in the new form: a probe weight has exactly ``d`` rows, so fewer rows and more rows
    are structural errors of the batch, raised before any work. (2026.5.0's truncation of extra rows is the
    translation's: ``tests/test_optimizers_frontend.py``.)"""

    def setUp(self):
        np.random.seed(5)
        self.x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        self.ww = tuple(np.random.randn(7, n) for n in (4, 5, 6))
        self.y = [np.random.randn(7, n) for n in (4, 5, 6)]                    # probe data, W=(7,)
        self.w5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_over_long_is_rejected(self):
        oracle = LeastSquaresOracle('probe', ())
        with self.assertRaisesRegex(ValueError, 'weight'):
            fitting.local_model(t3m.MANIFOLD, oracle, LeastSquaresBatch(self.ww, self.y, self.w5), self.x)
        fitting.local_model(t3m.MANIFOLD, oracle,
                            LeastSquaresBatch(self.ww, self.y, blocks_to_weight('probe', (), self.w5[:3], self.y)), self.x)

    def test_short_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'weight'):
            fitting.local_model(t3m.MANIFOLD, LeastSquaresOracle('probe', ()),
                                LeastSquaresBatch(self.ww, self.y, self.w5[:2]), self.x)


class TestUniformLayerLocalModel(unittest.TestCase):
    '''U7b: the uniform roll-your-own surface. ``fitting.local_model`` at a ``UniformTuckerTensorTrain`` x gives a
    ``fitting.LocalModel`` on a UT3Frame (UT3Tangent-valued gradient / Hessian) -- one model class serves both
    representations. Oracle: the backend ``LocalModel`` built here on the same oracle and packed batch (its
    correctness is the backend's tests'), and the ragged model at the same point (the equivalence contract).'''

    def _relerr(self, a, b):
        return float(cw.corewise_norm(cw.corewise_sub(a, b)) / cw.corewise_norm(b))

    def _cases(self):
        rng = np.random.default_rng(0)
        shape, order, NW = (5, 6, 7), 2, 15
        d = len(shape)
        omega = np.array([1.0, 0.5, 0.3])                              # per-order (order+1,)
        omega_mode = rng.uniform(0.3, 2.0, size=d)                      # per-mode (d,), for plain probe
        wmat = rng.uniform(0.3, 2.0, size=(d, order + 1))               # full ω[mode,order], for probe-derivs
        x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)))
        ww = [rng.standard_normal((NW, N)) for N in shape]
        pp = [rng.standard_normal((NW, N)) for N in shape]
        index = np.stack([rng.integers(0, N, size=NW) for N in shape], axis=0)
        scal = rng.standard_normal(NW)                                 # apply/entries observed data
        vecs = [rng.standard_normal((NW, N)) for N in shape]           # probe observed data (ragged)
        jet_s = rng.standard_normal((order + 1, NW))                   # apply/entries-deriv observed
        jet_v = [rng.standard_normal((order + 1, NW, N)) for N in shape]  # probe-deriv observed
        # (kind, sample, data, order, weight), in the 2026.5.0 vocabulary
        return x, [
            ('apply', ww, scal, None, None),
            ('entries', index, scal, None, None),
            ('probe', ww, vecs, None, None),
            ('probe', ww, vecs, None, omega_mode),                      # per-mode
            ('apply_derivatives', (ww, pp), jet_s, order, omega),
            ('entries_derivatives', (index, pp), jet_s, order, omega),
            ('probe_derivatives', (ww, pp), jet_v, order, omega),
            ('probe_derivatives', (ww, pp), jet_v, order, wmat),         # matrix
        ]

    def test_matches_backend_local_model(self):
        x, cases = self._cases()
        geoms = [('manifold', ut3m.UNIFORM_MANIFOLD, t3m.MANIFOLD), ('corewise', ut3m.UNIFORM_COREWISE, t3m.COREWISE)]
        for gname, geom, rgeom in geoms:
            for kind, sample, data, o, w in cases:
                with self.subTest(geom=gname, kind=kind, wshape=None if w is None else np.shape(w)):
                    fmodel, lm = _pair(geom, x, kind, sample, data, o, w)
                    self.assertIsInstance(fmodel, fitting.LocalModel)
                    self.assertIsInstance(fmodel.frame, ubv.UT3Frame)   # the uniform layer, one model class
                    self.assertEqual(float(fmodel.objective_value), float(lm.objective))
                    self.assertTrue(_bitwise(fmodel.gradient.variations.supercores, lm.gradient))
                    pt = geom.randn(fmodel.frame)
                    p = pt.variations.supercores
                    self.assertEqual(float(fmodel.gn_quadratic(pt)), float(lm.gn_quadratic(p)))
                    self.assertTrue(_bitwise(fmodel.gn_hessian(pt).variations.supercores, lm.hvp(p)))
                    rmodel = _model(rgeom, x.to_t3(), kind, sample, data, o, w)        # the ragged twin
                    self.assertTrue(np.allclose(float(fmodel.objective_value), float(rmodel.objective_value),
                                                rtol=1e-10))

    def test_gauged_gradient_and_self_consistency(self):
        '''On the manifold the gradient is gauged and ``pᵀHp == ‖J p‖²`` (the GN quadratic form matches the
        Hessian action); the corewise gradient is NOT gauged.'''
        x, cases = self._cases()
        for kind, sample, data, o, w in cases:
            for gname, geom, want_gauged in [('manifold', ut3m.UNIFORM_MANIFOLD, True),
                                             ('corewise', ut3m.UNIFORM_COREWISE, False)]:
                with self.subTest(kind=kind, geom=gname, wshape=None if w is None else np.shape(w)):
                    m = _model(geom, x, kind, sample, data, o, w)
                    self.assertEqual(bool(m.gradient.is_gauged().all()), want_gauged)
                    p = geom.randn(m.frame)
                    self.assertTrue(np.allclose(float(m.gn_quadratic(p)),
                                                float(p.corewise_inner(m.gn_hessian(p))), rtol=1e-9))

    def test_representation_geometry_must_match(self):
        '''A uniform x with a ragged geometry (or a ragged x with a uniform geometry) is a structural error.'''
        rng = np.random.default_rng(0)
        shape = (5, 6, 7)
        ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)))
        rx = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
        oracle = LeastSquaresOracle('apply', ())
        batch = LeastSquaresBatch([rng.standard_normal((10, N)) for N in shape], rng.standard_normal(10))
        with self.assertRaises(ValueError):   # uniform x + ragged geometry
            fitting.local_model(t3m.MANIFOLD, oracle, batch, ux)
        with self.assertRaises(ValueError):   # ragged x + uniform geometry
            fitting.local_model(ut3m.UNIFORM_MANIFOLD, oracle, batch, rx)

    def test_same_frame_guard(self):
        '''A trial tangent at a different frame is rejected (the numerical same-frame guard).'''
        rng = np.random.default_rng(0)
        shape = (5, 6, 7)
        x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)))
        other = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)))
        batch = LeastSquaresBatch([rng.standard_normal((10, N)) for N in shape], rng.standard_normal(10))
        m = fitting.local_model(ut3m.UNIFORM_MANIFOLD, LeastSquaresOracle('apply', ()), batch, x)
        p_other = ut3m.UNIFORM_MANIFOLD.randn(ut3m.UNIFORM_MANIFOLD.frame(other))
        with self.assertRaises(ValueError):
            m.gn_hessian(p_other)


class TestFactoryValidation(unittest.TestCase):
    """Review 2026-08-22 (C13), in the new form: ``local_model`` checks the batch against the point up front
    (``oracle.check_batch``), each mismatch a ``ValueError``."""

    def test_mismatches_are_named_value_errors(self):
        np.random.seed(6)
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
        ww = tuple(np.random.randn(7, n) for n in x.shape)
        model = lambda kind, sample, y, order=(): fitting.local_model(
            t3m.MANIFOLD, LeastSquaresOracle(kind, order), LeastSquaresBatch(sample, y), x)
        y = x.apply(ww)
        model('apply', ww, y)                                                          # the good case
        with self.assertRaises(ValueError):
            model('apply', ww, y[:5])                                                  # data shape
        with self.assertRaises(ValueError):
            model('apply', ww[:2], y)                                                  # len(ww)
        with self.assertRaises(ValueError):
            model('apply', (ww[0], np.random.randn(7, 9), ww[2]), y)                   # N_i
        with self.assertRaises(ValueError):
            model('probe', ww, x.probe(ww)[:2])                                        # probe data len
        idx = np.random.randint(0, 4, size=(3, 7)) % np.array(x.shape)[:, None]
        with self.assertRaises(ValueError):
            model('entries', idx[:2], x.entries(idx))                                  # index rows
        pp = tuple(np.random.randn(7, n) for n in x.shape)
        line = fitting.affine_jet(ww, pp)
        jets = x.apply(line, (2,))
        model('apply', line, jets, (2,))
        with self.assertRaises(ValueError):
            model('apply', line, jets[:2], (2,))                                       # the jet axis


# ---------------------------------------------------------------------------------------------------
# 2026.3.0 (S6a / S6): the uniform backend geometry from the frame, and the frontend exact mode (T18)
# ---------------------------------------------------------------------------------------------------
_EX_MINIMAL = ((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
_EX_TUCKER_SLACK = ((6, 6, 6), (4, 4, 4), (1, 2, 2, 1))     # frame up ranks (2, 4, 2) < down (4, 4, 4)
_EX_TT_SLACK = ((6, 6, 6), (4, 4, 4), (1, 9, 9, 1))         # frame left ranks != right ranks
_EX_D4 = ((4, 5, 4, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1))
_EX_SHAREABLE = ((5, 5, 7), (2, 2, 2), (1, 2, 2, 1))        # modes 0 and 1 can share a factor


def _ex_unit_point(struct):
    '''A random T3 with ``||X||_HS = 1`` (so finite differences are well scaled).'''
    x = t3.TuckerTensorTrain.randn(*struct)
    tk, tt = x.data
    return t3.TuckerTensorTrain(tuple(tk), tuple(tt[:-1]) + (tt[-1] / float(np.linalg.norm(x.to_dense())),))


def _ex_problem(struct, order, W, rng, sharing=None):
    '''A probe-derivative fit whose data come from ANOTHER unit point plus 10 % noise (an O(1) residual, so the
    curvature term is O(1)): the point x, the sample ``(ww, pp)``, the derivative data.'''
    x, other = _ex_unit_point(struct), _ex_unit_point(struct)
    if sharing is not None:                                   # a tied point (the shared geometries' set)
        x = t3.TuckerTensorTrain(*bsharing.t3_tie_tucker_factors(x.data, sharing))
    ww = [rng.standard_normal((W, N)) for N in struct[0]]
    pp = [rng.standard_normal((W, N)) for N in struct[0]]
    data = [z + 0.1 * rng.standard_normal(z.shape) for z in dd.forward(other, 'probe', ww, pp, order)]
    return x, (ww, pp), data


import contextlib as _contextlib


@_contextlib.contextmanager
def _no_spy():
    yield None


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
    '''S6a (X13) and D5: the frontend uniform manifold model builds its backend geometry from the frame it computes
    (``UniformManifoldGeometryOps.from_frame``), one orthogonalization in all: the SAME geometry as ``from_point``
    (value equality over every field, the variation masks included) on a minimal point, a C-stacked one and a shared
    one; and a NON-minimal point (Tucker slack, TT slack), which ``from_point`` refuses, still gets a working model
    whose numbers are the ragged model's at the same point.'''

    def setUp(self):
        np.random.seed(61)

    def test_from_frame_is_from_point_without_the_orthogonalization(self):
        from unittest import mock
        import t3toolbox.backend.ufv_conversions as ufc
        rng = np.random.default_rng(61)
        cases = [('minimal', _EX_MINIMAL, (), None), ('stacked', _EX_MINIMAL, (2,), None),
                 ('shared', _EX_SHAREABLE, (), (0, 0, 1)),
                 ('tucker_slack', _EX_TUCKER_SLACK, (), None), ('tt_slack', _EX_TT_SLACK, (), None)]
        for name, struct, C, sharing in cases:
            with self.subTest(case=name):
                rx = t3.TuckerTensorTrain.randn(*struct, stack_shape=C)
                if sharing is not None:
                    rx = t3.TuckerTensorTrain(*bsharing.t3_tie_tucker_factors(rx.data, sharing))
                x = ut3.UniformTuckerTensorTrain.from_t3(rx)
                geom = ut3m.UNIFORM_MANIFOLD if sharing is None else sg.shared(ut3m.UNIFORM_MANIFOLD, sharing)
                ww = [rng.standard_normal((6, N)) for N in struct[0]]
                y = [rng.standard_normal((6,) + (1,) * len(C) + (N,)) for N in struct[0]]   # shared by C: size 1
                oracle, batch = LeastSquaresOracle('probe', ()), LeastSquaresBatch(ww, y)
                with mock.patch.object(ufc, 'ut3_orthogonal_representations',
                                       wraps=ufc.ut3_orthogonal_representations) as spy:
                    m = fitting.local_model(geom, oracle, batch, x)
                self.assertEqual(spy.call_count, 1)                           # the frame's, and no other
                built = m.backend.geom
                self.assertIsInstance(built, bgeo.UniformManifoldGeometryOps)
                direct = bgeo.UniformManifoldGeometryOps.from_frame(m.backend.frame, x.masks.data, sharing)
                self.assertEqual(built, direct)
                if name in ('tucker_slack', 'tt_slack'):                       # D5: non-minimal works
                    self.assertFalse(rx.has_minimal_ranks)
                    with self.assertRaises(ValueError):
                        bgeo.UniformManifoldGeometryOps.from_point(x.data, sharing)
                    rm = fitting.local_model(t3m.MANIFOLD, oracle, batch, rx)
                    p = ut3m.UNIFORM_MANIFOLD.randn(m.frame)
                    self.assertTrue(np.allclose(float(m.objective_value), float(rm.objective_value), rtol=1e-10))
                    self.assertTrue(np.allclose(float(m.gradient.corewise_norm()), float(rm.gradient.corewise_norm()),
                                                rtol=1e-9))
                    self.assertTrue(np.allclose(float(m.gn_quadratic(p)), float(p.corewise_inner(m.gn_hessian(p))),
                                                rtol=1e-9))
                    continue
                reference = bgeo.UniformManifoldGeometryOps.from_point(x.data, sharing)
                self.assertEqual(built, reference)                             # value equality: every field
                self.assertEqual(hash(built), hash(reference))
                for a, b in zip(built.var_masks + built.masks, reference.var_masks + reference.masks):
                    self.assertEqual(a.shape, b.shape)
                    self.assertTrue(np.array_equal(a, b))
                self.assertEqual(built.groups, reference.groups)

    def test_gauss_newton_regularized_numbers_unchanged(self):
        '''The geometry feeds the regularizer: the uniform regularized model's objective, gradient and
        quadratic are those of the backend model built with the ``from_point`` geometry, bit for bit.'''
        rng = np.random.default_rng(62)
        R = bopt.IdentityRegularizer(0.4)
        x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*_EX_MINIMAL))
        ww = [rng.standard_normal((9, N)) for N in _EX_MINIMAL[0]]
        y = [rng.standard_normal((9, N)) for N in _EX_MINIMAL[0]]
        m, ref = _pair(ut3m.UNIFORM_MANIFOLD, x, 'probe', ww, y, regularizer=R)
        p = ut3m.UNIFORM_MANIFOLD.randn(m.frame)
        self.assertEqual(float(m.objective_value), float(ref.objective))
        self.assertEqual(float(m.gn_quadratic(p)), float(ref.gn_quadratic(p.variations.supercores)))
        self.assertTrue(_bitwise(m.gradient.variations.supercores, ref.gradient))
        self.assertGreater(float(ref.regularization), 0.0)


class TestExactLocalModel(unittest.TestCase):
    '''T18: the frontend exact mode. ``fitting.local_model(..., hessian='exact')`` equals its backend twin
    (``bopt.local_model(..., hessian='exact')`` on the same oracle and batch, built here); ``is_exact`` is derived
    (no ``hessian`` field); the exact gradient is the Gauss-Newton one bit for bit; ``hessian`` / ``quadratic`` /
    ``evaluate`` are mode-dependent and the ``curvature_*`` actions exist only in exact mode; ``retract`` is the
    geometry's (a shared model's with its companion) and the backend model's; ``curvature_damping`` reaches the lift
    and raises where it does nothing; and the exact Hessian against a finite difference of the Riemannian gradient
    along the retraction (the T3Polynomial ``x07`` cross-check).'''

    def setUp(self):
        np.random.seed(71)

    def test_mode_is_derived(self):
        '''``is_exact`` is the backend model's (``curvature is not None``) -- False for a Gauss-Newton model, True for
        ``hessian='exact'`` (case-insensitive); the frontend model's fields are the geometry and the backend model
        only; an unknown mode raises.'''
        import dataclasses
        rng = np.random.default_rng(71)
        x, sample, data = _ex_problem(_EX_MINIMAL, 1, 8, rng)
        gn = _model(t3m.MANIFOLD, x, 'probe_derivatives', sample, data, 1)
        ex = _model(t3m.MANIFOLD, x, 'probe_derivatives', sample, data, 1, hessian='EXACT')
        self.assertFalse(gn.is_exact)
        self.assertIsNone(gn.backend.curvature)
        self.assertTrue(ex.is_exact)
        self.assertIsInstance(ex.backend.curvature, bopt.CurvatureState)
        names = [f.name for f in dataclasses.fields(fitting.LocalModel)]
        self.assertEqual(names, ['geometry', 'backend'])
        self.assertNotIn('hessian', vars(ex))                                  # no instance attribute
        self.assertTrue(callable(ex.hessian) and callable(gn.hessian))         # the method, in both modes
        self.assertIsInstance(vars(fitting.LocalModel)['is_exact'], property)  # read-only, derived
        with self.assertRaises(AttributeError):
            ex.is_exact = False
        self.assertFalse(_model(t3m.MANIFOLD, x, 'probe_derivatives', sample, data, 1,
                                hessian='Gauss_Newton').is_exact)
        for bad in (True, None, 'newton', 'Gauss-Newton', ''):
            with self.subTest(hessian=repr(bad)):
                with self.assertRaisesRegex(ValueError, "hessian must be 'gauss_newton' or 'exact'"):
                    _model(t3m.MANIFOLD, x, 'probe_derivatives', sample, data, 1, hessian=bad)
        p = t3m.MANIFOLD.randn(gn.frame)
        for method in ('curvature_hessian', 'curvature_quadratic'):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, 'needs an exact-mode model'):
                    getattr(gn, method)(p)

    def _check_actions(self, m, blm, geom, coords, Ks=((), (2,)), companion=None):
        '''Every action of the frontend exact model against its backend twin's, bit for bit, a tangent stack K
        included. ``companion``: the name of the ``backend.sharing`` function that builds a shared companion, which the
        model's actions must not call (they run on the stored one).'''
        from unittest import mock
        self.assertTrue(_bitwise(m.backend.frame, blm.frame))
        self.assertEqual(float(m.objective_value), float(blm.objective))
        self.assertTrue(_bitwise(coords(m.gradient), blm.gradient))
        self.assertTrue(_bitwise(m.backend.curvature.misfit_gradient, blm.curvature.misfit_gradient))
        for K in Ks:
            p = geom.randn(m.frame, stack_shape=K)
            raw = coords(p)
            spy = mock.patch.object(bsharing, companion, wraps=getattr(bsharing, companion)) if companion else None
            with (spy if spy is not None else _no_spy()) as called:
                Hc, qc, q, H, gq = (m.curvature_hessian(p), m.curvature_quadratic(p), m.quadratic(p), m.hessian(p),
                                    m.gn_quadratic(p))
            if spy is not None:
                self.assertEqual(called.call_count, 0)                             # the stored companion
            self.assertTrue(_bitwise(coords(Hc), blm.curvature_hvp(raw)))
            self.assertTrue(np.array_equal(np.asarray(qc), np.asarray(blm.curvature_quadratic(raw))))
            self.assertTrue(np.array_equal(np.asarray(q), np.asarray(blm.quadratic(raw))))
            self.assertTrue(_bitwise(coords(H), blm.hvp(raw)))
            self.assertTrue(np.array_equal(np.asarray(gq), np.asarray(blm.gn_quadratic(raw))))
            self.assertEqual(np.shape(q), K)
            self.assertGreater(float(np.max(np.abs(np.asarray(qc)))), 0.0)

    def test_matches_the_backend_exact_local_model(self):
        '''Every action of the exact frontend model equals its backend twin's, bit for bit. Ragged manifold (minimal,
        Tucker slack, TT slack, d = 4, a full ω matrix, a regularizer) and corewise, a tangent stack K included.'''
        rng = np.random.default_rng(72)
        cases = [('minimal', _EX_MINIMAL, 1, None, None, t3m.MANIFOLD),
                 ('tucker_slack', _EX_TUCKER_SLACK, 1, None, None, t3m.MANIFOLD),
                 ('tt_slack', _EX_TT_SLACK, 0, None, None, t3m.MANIFOLD),
                 ('d4_omega', _EX_D4, 2, 'matrix', None, t3m.MANIFOLD),
                 ('regularized', _EX_MINIMAL, 2, 'row', bopt.IdentityRegularizer(0.3), t3m.MANIFOLD),
                 ('corewise', _EX_MINIMAL, 1, 'row', None, t3m.COREWISE)]
        for name, struct, order, wkind, reg, geom in cases:
            with self.subTest(case=name):
                x, sample, data = _ex_problem(struct, order, 7, rng)
                d = len(struct[0])
                w = {None: None, 'row': rng.uniform(0.3, 2.0, order + 1),
                     'matrix': rng.uniform(0.3, 2.0, (d, order + 1))}[wkind]
                m, blm = _pair(geom, x, 'probe_derivatives', sample, data, order, w, regularizer=reg,
                               hessian='exact')
                self._check_actions(m, blm, geom, lambda t: t.variations.data)

    def test_uniform_matches_the_backend_exact_local_model(self):
        '''S8: the exact frontend model on the UNIFORM layer is its backend twin on the same supercores -- every action
        bit for bit, on both uniform geometries, a tangent stack K included; and the uniform m27 -- the exact gradient
        IS the Gauss-Newton gradient, bit for bit, masks included.'''
        rng = np.random.default_rng(74)
        for struct, order in ((_EX_MINIMAL, 1), (_EX_D4, 2)):
            x, sample, data = _ex_problem(struct, order, 7, rng)
            ux = ut3.UniformTuckerTensorTrain.from_t3(x)
            w = rng.uniform(0.3, 2.0, (len(struct[0]), order + 1))
            for geom, name in ((ut3m.UNIFORM_MANIFOLD, 'manifold'), (ut3m.UNIFORM_COREWISE, 'corewise')):
                with self.subTest(struct=struct, geometry=name):
                    m, blm = _pair(geom, ux, 'probe_derivatives', sample, data, order, w, hessian='exact')
                    gn = _model(geom, ux, 'probe_derivatives', sample, data, order, w)
                    self.assertTrue(m.is_exact)
                    self._check_actions(m, blm, geom, lambda t: t.variations.supercores)
                    self.assertTrue(_bitwise(m.gradient.variations.supercores, gn.gradient.variations.supercores))
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(m.gradient.variations.masks.data,
                                                                             gn.gradient.variations.masks.data)))

    def test_shared_uniform_matches_the_backend_exact_local_model(self):
        '''S9: the exact frontend model on a SHARED uniform geometry is its backend twin -- every action bit for bit, a
        tangent stack K included -- and on the manifold the curvature runs on the model's ONE uniform companion (the
        tied lift factors carry ``backend.geom_aux``; no curvature action rebuilds it). On the group-ceiling structure
        the tied lift is load-bearing: the per-mode lift would differ.'''
        import dataclasses
        from unittest import mock
        import t3toolbox.backend.ufv_operations as ufo
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
                    m, blm = _pair(geom, ux, 'probe_derivatives', sample, data, order, w, hessian='exact')
                    self.assertTrue(m.is_exact)
                    if gname == 'manifold':
                        self.assertIs(m.backend.curvature.geometry_data.lift_factors.shared_data, m.backend.geom_aux)
                    self._check_actions(m, blm, geom, lambda t: t.variations.supercores,
                                        companion='ufv_shared_frame_data')
                    if name == 'ceiling' and gname == 'manifold':                     # the tied lift matters
                        gd = blm.curvature.geometry_data
                        untied = dataclasses.replace(blm, curvature=bopt.CurvatureState(
                            blm.curvature.misfit_gradient,
                            dataclasses.replace(gd, lift_factors=ufo.ufv_lift_factors(blm.frame, 1e-6))))
                        p = geom.randn(m.frame)
                        self.assertGreater(_relerr_trees(m.curvature_hessian(p).variations.supercores,
                                                         untied.curvature_hvp(p.variations.supercores)), 1e-3)

    def test_gradient_is_the_gauss_newton_gradient_bitwise(self):
        '''m27: the exact model's gradient is the Gauss-Newton model's, bit for bit (with and without a regularizer),
        because it IS the same misfit transpose -- queried once, at the build (``('f', 'g')``), and read.'''
        from unittest import mock
        rng = np.random.default_rng(73)
        x, sample, data = _ex_problem(_EX_MINIMAL, 2, 8, rng)
        for reg in (None, bopt.IdentityRegularizer(0.3)):
            for geom in (t3m.MANIFOLD, t3m.COREWISE):
                with self.subTest(regularizer=reg is not None, geometry=type(geom).__name__):
                    gn = _model(geom, x, 'probe_derivatives', sample, data, 2, [1.0, 0.5, 0.25], regularizer=reg)
                    with mock.patch.object(LeastSquaresOracle, 'query', autospec=True,
                                           side_effect=LeastSquaresOracle.query) as spy:
                        ex = _model(geom, x, 'probe_derivatives', sample, data, 2, [1.0, 0.5, 0.25],
                                    regularizer=reg, hessian='exact')
                        self.assertEqual(spy.call_count, 1)                    # the one build query
                        self.assertIn('g', spy.call_args.args[3])               # it wants g
                        g_ex = ex.gradient
                        self.assertEqual(spy.call_count, 1)                    # read, not recomputed
                    self.assertTrue(_bitwise(g_ex.variations.data, gn.gradient.variations.data))

    def test_hessian_quadratic_and_evaluate_are_mode_dependent(self):
        '''Gauss-Newton: ``hessian`` / ``quadratic`` ARE ``gn_hessian`` / ``gn_quadratic`` and ``evaluate`` is
        ``c + ⟨g, p⟩ + ½(‖𝒥Πp‖² + ⟨p, H_R p⟩)``. Exact: ``hessian = gn_hessian + curvature_hessian``, ``quadratic =
        gn_quadratic + curvature_quadratic = ⟨p, hessian(p)⟩``, ``evaluate(p) = c + ⟨g, p⟩ + ½ quadratic(p)``; the
        Gauss-Newton members mean the same in both modes.'''
        rng = np.random.default_rng(74)
        x, sample, data = _ex_problem(_EX_MINIMAL, 1, 8, rng)
        R = bopt.IdentityRegularizer(0.2)
        gn = _model(t3m.MANIFOLD, x, 'probe_derivatives', sample, data, 1, regularizer=R)
        ex = _model(t3m.MANIFOLD, x, 'probe_derivatives', sample, data, 1, regularizer=R, hessian='exact')
        p = t3m.MANIFOLD.randn(gn.frame)
        self.assertTrue(_bitwise(gn.hessian(p).variations.data, gn.gn_hessian(p).variations.data))
        self.assertEqual(float(gn.quadratic(p)), float(gn.gn_quadratic(p)))
        Jp_sq = sum(float(np.sum(np.asarray(z) ** 2)) for z in gn.jacobian(p))   # unweighted: ‖𝒥Πp‖²
        reg_q = float(R.quadratic(gn.backend.geom, gn.backend.frame, p.variations.data))
        self.assertAlmostEqual(float(gn.evaluate(p)), float(gn.objective_value + gn.gradient.corewise_inner(p))
                               + 0.5 * (Jp_sq + reg_q), delta=1e-12 * abs(float(gn.evaluate(p))))
        for member in ('gn_hessian', 'jacobian'):
            self.assertTrue(_bitwise(getattr(ex, member)(p) if member == 'jacobian'
                                     else getattr(ex, member)(p).variations.data,
                                     getattr(gn, member)(p) if member == 'jacobian'
                                     else getattr(gn, member)(p).variations.data))
        self.assertEqual(float(ex.gn_quadratic(p)), float(gn.gn_quadratic(p)))
        self.assertEqual(float(ex.objective_value), float(gn.objective_value))
        Hp = ex.hessian(p)
        self.assertLess(_relerr_trees(Hp.variations.data, (ex.gn_hessian(p) + ex.curvature_hessian(p)).variations.data),
                        1e-13)
        q = float(ex.quadratic(p))
        self.assertAlmostEqual(q, float(ex.gn_quadratic(p) + ex.curvature_quadratic(p)), delta=1e-13 * abs(q))
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
        rng = np.random.default_rng(75)
        for name, struct, order, reg in (('minimal', _EX_MINIMAL, 1, None), ('d4', _EX_D4, 2, None),
                                         ('regularized', _EX_MINIMAL, 1, bopt.IdentityRegularizer(0.5))):
            with self.subTest(case=name):
                x, sample, data = _ex_problem(struct, order, 8, rng)

                def model_at(y, hessian='gauss_newton'):
                    return _model(t3m.MANIFOLD, y, 'probe_derivatives', sample, data, order, regularizer=reg,
                                  hessian=hessian)

                m = model_at(x, 'exact')
                p = t3m.MANIFOLD.randn(m.frame)
                p = p * (1.0 / float(t3m.MANIFOLD.norm(p)))
                h = 0.05 * min(float(np.min(m.backend.curvature.geometry_data.sigma_min)), 1.0) * 1e-2

                def grad_at(t):                     # P_x grad f(R_x(t p)), as gauged coordinates at x's frame
                    g = model_at(m.retract(p * t)).gradient
                    return t3m.MANIFOLD.project_ambient(m.frame, g.to_t3())

                fd = (grad_at(h) - grad_at(-h)) * (1.0 / (2.0 * h))
                exact_err = _relerr_trees(m.hessian(p).variations.data, fd.variations.data)
                gn_err = _relerr_trees(m.gn_hessian(p).variations.data, fd.variations.data)
                self.assertLess(exact_err, 1e-5)
                self.assertGreater(gn_err, 1e3 * exact_err)

    def test_retract_is_the_geometrys_and_the_backend_models(self):
        '''m38: ``retract(p)`` is ``geometry.retract(p)`` at this frame -- a shared manifold model's with its
        companion (no rebuild) -- and equals the backend ``LocalModel.retract`` bit for bit; the same in both modes;
        ragged and uniform; a trial step at another frame is rejected.'''
        from unittest import mock
        rng = np.random.default_rng(76)
        cases = [('manifold', t3m.MANIFOLD), ('corewise', t3m.COREWISE),
                 ('shared_manifold', sg.shared_manifold((0, 0, 1))),
                 ('shared_corewise', sg.shared_corewise((0, 0, 1)))]
        for name, geom in cases:
            with self.subTest(case=name):
                x, sample, data = _ex_problem(_EX_SHAREABLE, 1, 7, rng, sharing=(0, 0, 1))
                m, blm = _pair(geom, x, 'probe_derivatives', sample, data, 1)
                p = geom.randn(m.frame) * 0.1
                with mock.patch.object(sg.SharedGeometry, 'shared_frame_data',
                                       wraps=sg.SharedGeometry.shared_frame_data, autospec=True) as spy:
                    y = m.retract(p)
                self.assertEqual(spy.call_count, 0)                                   # the stored companion
                self.assertIsInstance(y, t3.TuckerTensorTrain)
                self.assertTrue(_bitwise(y.data, geom.retract(p).data))
                self.assertTrue(_bitwise(y.data, blm.retract(p.variations.data)))
                ex = _model(geom, x, 'probe_derivatives', sample, data, 1, hessian='exact')
                with mock.patch.object(sg.SharedGeometry, 'shared_frame_data',
                                       wraps=sg.SharedGeometry.shared_frame_data, autospec=True) as spy:
                    y_ex = ex.retract(p)
                self.assertEqual(spy.call_count, 0)
                self.assertTrue(_bitwise(y_ex.data, y.data))
                self.assertTrue(_bitwise(y_ex.data, ex.backend.retract(p.variations.data)))
        other = t3m.MANIFOLD.randn(t3m.MANIFOLD.frame(t3.TuckerTensorTrain.randn(*_EX_SHAREABLE)))
        with self.assertRaises(ValueError):
            m.retract(other)
        # uniform: the uniform geometry's retract, and the backend uniform model's
        x, sample, data = _ex_problem(_EX_MINIMAL, 1, 7, rng)
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        um, ulm = _pair(ut3m.UNIFORM_MANIFOLD, ux, 'probe_derivatives', sample, data, 1)
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
        import t3toolbox.backend.fv_operations as fvo
        rng = np.random.default_rng(77)
        x, sample, data = _ex_problem(_EX_MINIMAL, 1, 8, rng)
        for cd, ref in ((0.0, 0.0), (1e-3, 1e-3), (None, bgeo.DEFAULT_CURVATURE_DAMPING)):
            with self.subTest(curvature_damping=cd):
                m = _model(t3m.MANIFOLD, x, 'probe_derivatives', sample, data, 1, hessian='exact',
                           curvature_damping=cd)
                lf = m.backend.curvature.geometry_data.lift_factors
                direct = fvo.fv_lift_factors(m.backend.frame, ref)
                self.assertTrue(_bitwise(lf.tucker_pinvs + lf.bond_pinvs + (lf.eps,),
                                         direct.tucker_pinvs + direct.bond_pinvs + (direct.eps,)))
        for geom in (t3m.MANIFOLD, t3m.COREWISE):
            with self.subTest(gauss_newton=type(geom).__name__):
                with self.assertRaisesRegex(ValueError, "local_model: curvature_damping is only "
                                                        "meaningful with hessian='exact'"):
                    _model(geom, x, 'probe_derivatives', sample, data, 1, curvature_damping=1e-6)
        with self.assertRaisesRegex(ValueError, 'curvature_damping must be >= 0'):
            _model(t3m.MANIFOLD, x, 'probe_derivatives', sample, data, 1, hessian='exact', curvature_damping=-1.0)
        xs, ssample, sdata = _ex_problem(_EX_SHAREABLE, 1, 8, rng, sharing=(0, 0, 1))
        for name, geom, xx, ss, dd_ in (('corewise', t3m.COREWISE, x, sample, data),
                                        ('shared_corewise', sg.shared_corewise((0, 0, 1)), xs, ssample, sdata)):
            with self.subTest(exact=name):
                self.assertTrue(_model(geom, xx, 'probe_derivatives', ss, dd_, 1, hessian='exact').is_exact)
                for cd in (0.0, 1e-6):
                    with self.assertRaisesRegex(ValueError, 'no effect on a corewise geometry'):
                        _model(geom, xx, 'probe_derivatives', ss, dd_, 1, hessian='exact', curvature_damping=cd)

    def test_shared_corewise_matches_the_backend(self):
        '''Shared COREWISE exact mode is live (the tied Hessian): the frontend model is its backend twin.'''
        rng = np.random.default_rng(78)
        x, sample, data = _ex_problem(_EX_SHAREABLE, 1, 8, rng, sharing=(0, 0, 1))
        geom = sg.shared_corewise((0, 0, 1))
        m, blm = _pair(geom, x, 'probe_derivatives', sample, data, 1, hessian='exact')
        self._check_actions(m, blm, geom, lambda t: t.variations.data)

    def test_plain_kinds_match_the_backend_exact_local_model(self):
        '''S10 / S11: the exact frontend model of the plain probe / apply / entries fits is its backend twin -- every
        action bit for bit, a tangent stack K included -- on the ragged manifold and corewise geometries, the uniform
        twins and a shared uniform manifold. Probe with a per-mode weight.'''
        rng = np.random.default_rng(84)
        for kind in ('apply', 'entries', 'probe'):
            for struct, labels in ((_EX_MINIMAL, None), (_EX_SHAREABLE, (0, 0, 1))):
                x, other = _ex_unit_point(struct), _ex_unit_point(struct)
                if labels is not None:
                    x = t3.TuckerTensorTrain(*bsharing.t3_tie_tucker_factors(x.data, labels))
                d = len(struct[0])
                if kind == 'entries':
                    sample = np.stack([rng.integers(0, N, size=9) for N in struct[0]])
                    data = other.entries(sample) + 0.1 * rng.standard_normal(9)
                else:
                    sample = [rng.standard_normal((9, N)) for N in struct[0]]
                    data = (other.apply(sample) + 0.1 * rng.standard_normal(9) if kind == 'apply'
                            else [z + 0.1 * rng.standard_normal(z.shape) for z in other.probe(sample)])
                w = np.linspace(0.5, 2.0, d) if kind == 'probe' else None
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                if labels is None:
                    setups = [('ragged manifold', t3m.MANIFOLD, x), ('ragged corewise', t3m.COREWISE, x),
                              ('uniform manifold', ut3m.UNIFORM_MANIFOLD, ux),
                              ('uniform corewise', ut3m.UNIFORM_COREWISE, ux)]
                else:
                    setups = [('shared uniform manifold', sg.shared(ut3m.UNIFORM_MANIFOLD, labels), ux)]
                for name, geom, xx in setups:
                    with self.subTest(kind=kind, setup=name):
                        coords = ((lambda t: t.variations.supercores) if xx is ux else (lambda t: t.variations.data))
                        m, blm = _pair(geom, xx, kind, sample, data, None, w, hessian='exact')
                        self.assertTrue(m.is_exact)
                        self._check_actions(m, blm, geom, coords)

    def test_derivative_kinds_match_the_backend_exact_local_model(self):
        '''S12: the exact frontend model of an apply / entries derivative fit (the entries one is the apply kind at
        the one-hot jet) is its backend twin -- every action bit for bit, a tangent stack K included -- on the ragged
        manifold and corewise geometries, the uniform twins and a shared uniform manifold, with a per-order weight.'''
        rng = np.random.default_rng(86)
        order, w = 2, np.array([1.0, 0.5, 0.25])
        for kind in ('apply_derivatives', 'entries_derivatives'):
            for struct, labels in ((_EX_MINIMAL, None), (_EX_SHAREABLE, (0, 0, 1))):
                x, other = _ex_unit_point(struct), _ex_unit_point(struct)
                if labels is not None:
                    x = t3.TuckerTensorTrain(*bsharing.t3_tie_tucker_factors(x.data, labels))
                pp = [rng.standard_normal((9, N)) for N in struct[0]]
                if kind == 'entries_derivatives':
                    first = np.stack([rng.integers(0, N, size=9) for N in struct[0]])
                    clean = dd.forward(other, 'entries', first, pp, order)
                else:
                    first = [rng.standard_normal((9, N)) for N in struct[0]]
                    clean = dd.forward(other, 'apply', first, pp, order)
                data = clean + 0.1 * rng.standard_normal(clean.shape)
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                if labels is None:
                    setups = [('ragged manifold', t3m.MANIFOLD, x), ('ragged corewise', t3m.COREWISE, x),
                              ('uniform manifold', ut3m.UNIFORM_MANIFOLD, ux),
                              ('uniform corewise', ut3m.UNIFORM_COREWISE, ux)]
                else:
                    setups = [('shared uniform manifold', sg.shared(ut3m.UNIFORM_MANIFOLD, labels), ux)]
                for name, geom, xx in setups:
                    with self.subTest(kind=kind, setup=name):
                        coords = ((lambda t: t.variations.supercores) if xx is ux else (lambda t: t.variations.data))
                        m, blm = _pair(geom, xx, kind, (first, pp), data, order, w, hessian='exact')
                        self.assertTrue(m.is_exact)
                        self._check_actions(m, blm, geom, coords)

    def test_every_kind_builds_exact(self):
        '''Every sampling kind of the least-squares oracle supplies ``Kp``: each of the six fits of 2026.5.0 builds an
        exact model (``is_exact``) on the ragged manifold and on both uniform geometries, and the Gauss-Newton model is
        unaffected; and the shared manifold geometries, ragged and uniform, build exact too (S7 / S9).'''
        rng = np.random.default_rng(79)
        x = t3.TuckerTensorTrain.randn(*_EX_MINIMAL)
        ww = [rng.standard_normal((6, N)) for N in x.shape]
        pp = [rng.standard_normal((6, N)) for N in x.shape]
        idx = np.stack([rng.integers(0, N, size=6) for N in x.shape])
        builds = {'apply': (ww, x.apply(ww), None), 'entries': (idx, x.entries(idx), None),
                  'probe': (ww, x.probe(ww), None),
                  'apply_derivatives': ((ww, pp), dd.forward(x, 'apply', ww, pp, 1), 1),
                  'entries_derivatives': ((idx, pp), dd.forward(x, 'entries', idx, pp, 1), 1),
                  'probe_derivatives': ((ww, pp), dd.forward(x, 'probe', ww, pp, 1), 1)}
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        for name, (sample, data, order) in builds.items():
            for geom, xx in ((t3m.MANIFOLD, x), (ut3m.UNIFORM_MANIFOLD, ux), (ut3m.UNIFORM_COREWISE, ux)):
                with self.subTest(kind=name, geometry=type(geom).__name__):
                    self.assertFalse(_model(geom, xx, name, sample, data, order).is_exact)
                    self.assertTrue(_model(geom, xx, name, sample, data, order, hessian='exact').is_exact)
        xt = t3.TuckerTensorTrain(*bsharing.t3_tie_tucker_factors(
            t3.TuckerTensorTrain.randn(*_EX_SHAREABLE).data, (0, 0, 1)))
        uxt = ut3.UniformTuckerTensorTrain.from_t3(xt)
        wws, pps = [rng.standard_normal((6, N)) for N in xt.shape], [rng.standard_normal((6, N)) for N in xt.shape]
        dt = dd.forward(xt, 'probe', wws, pps, 1)
        for base in (ut3m.UNIFORM_MANIFOLD, ut3m.UNIFORM_COREWISE):
            with self.subTest(shared_uniform=type(base).__name__):
                ugeom = sg.shared(base, (0, 0, 1))
                self.assertFalse(_model(ugeom, uxt, 'probe_derivatives', (wws, pps), dt, 1).is_exact)
                self.assertTrue(_model(ugeom, uxt, 'probe_derivatives', (wws, pps), dt, 1, hessian='exact').is_exact)
        xs, ssample, sdata = _ex_problem(_EX_SHAREABLE, 1, 8, rng, sharing=(0, 0, 1))
        geom = sg.shared_manifold((0, 0, 1))
        self.assertFalse(_model(geom, xs, 'probe_derivatives', ssample, sdata, 1).is_exact)
        self.assertTrue(_model(geom, xs, 'probe_derivatives', ssample, sdata, 1, hessian='exact').is_exact)

    def test_shared_manifold_matches_the_backend(self):
        '''S7: the exact frontend model on a shared MANIFOLD geometry is its backend twin -- every action bit for bit,
        ``evaluate`` its two-form, ``retract`` the backend's -- and the curvature runs on the model's ONE companion
        (the tied lift factors carry ``backend.geom_aux``): no curvature action rebuilds it. On the group-ceiling
        structure (mode 0's own ``S_0^T`` wide) the tied lift is load-bearing, so a model whose curvature used the
        per-mode lift would differ.'''
        import dataclasses
        from unittest import mock
        import t3toolbox.backend.fv_operations as fvo
        rng = np.random.default_rng(81)
        cases = [('shareable', _EX_SHAREABLE, (0, 0, 1), 1, None),
                 ('ceiling', ((6, 6, 4), (4, 4, 2), (1, 2, 2, 1)), (0, 0, 1), 2, 'row'),
                 ('all_modes', ((5, 5, 5), (3, 3, 3), (1, 2, 2, 1)), (0, 0, 0), 1, None)]
        for name, struct, labels, order, wkind in cases:
            with self.subTest(case=name):
                x, sample, data = _ex_problem(struct, order, 7, rng, sharing=labels)
                w = None if wkind is None else rng.uniform(0.3, 2.0, order + 1)
                geom = sg.shared_manifold(labels)
                m, blm = _pair(geom, x, 'probe_derivatives', sample, data, order, w, hessian='exact')
                self.assertTrue(m.is_exact)
                self.assertIs(m.backend.curvature.geometry_data.lift_factors.shared_data, m.backend.geom_aux)
                self._check_actions(m, blm, geom, lambda t: t.variations.data, companion='fv_shared_frame_data')
                p = geom.randn(m.frame)
                with mock.patch.object(bsharing, 'fv_shared_frame_data', wraps=bsharing.fv_shared_frame_data) as spy:
                    ev, q = m.evaluate(p), m.quadratic(p)
                self.assertEqual(spy.call_count, 0)                                # the stored companion
                self.assertAlmostEqual(float(ev), float(m.objective_value + m.gradient.corewise_inner(p))
                                       + 0.5 * float(q), delta=1e-12 * abs(float(ev)))
                p = geom.randn(m.frame) * 0.1
                self.assertTrue(_bitwise(m.retract(p).data, blm.retract(p.variations.data)))
                if name == 'ceiling':                                                   # the tied lift matters
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
        m = _model(t3m.MANIFOLD, x, 'probe_derivatives', sample, data, 1, hessian='exact')
        self.assertTrue(m.is_exact)
        self.assertIsNone(m.backend.curvature.geometry_data)
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
        y = dd.forward(t3.TuckerTensorTrain.randn(shape, *ranks), 'probe', ww, pp, 1)

        def model_at(x, hessian='gauss_newton'):
            return _model(M, x, 'probe_derivatives', (ww, pp), y, 1, hessian=hessian)

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
