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
See ``docs/geometry_refactor_plan.md``.
'''

import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.basis_variations_format as bvf
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


def probe_dense(T, ww, n_c):
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
    '''Build a model of one (kind, geometry, base-stack C), plus the dense forward oracle and the
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
        dense_fwd = lambda T: probe_dense(T, sample, n_c)
        r = [np.random.randn(*((m,) + C + (N,))) for N in SHAPE]
        samp_dot = lambda a, b: sum(np.sum(ai * bi, axis=tuple(range(n_w)) + (ai.ndim - 1,))
                                    for ai, bi in zip(a, b))
        samp_add = lambda a, b: [ai + bi for ai, bi in zip(a, b)]
        rand_like = lambda v: [np.random.randn(*vi.shape) for vi in v]
    model = _FACTORY[kind](geometry, x, sample, r)
    return dict(x=x, geometry=geometry, geom_name=geom_name, model=model, base=model.base,
                sample=sample, r=r, c=0.5 * samp_dot(r, r), n_c=n_c, n_w=n_w, dense_fwd=dense_fwd,
                samp_dot=samp_dot, samp_add=samp_add, rand_like=rand_like)


def _raw_step(s):
    '''A raw (any-gauge) trial tangent at the model's base -- ungauged on the manifold (tests Π),
    a raw core perturbation on the corewise frame.'''
    return t3m.COREWISE.randn(s['base'])


class TestGaussNewtonModel(unittest.TestCase):
    '''The generic GN model, parameterized over the sampling kind, the geometry, and the base stack C.'''

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
                        self.assertTrue(model.gn_hessian(p).allclose(model.gn_hessian(Pp), rtol=RTOL, atol=ATOL))

    def test_matched_pair(self):
        '''The structural matched pair: MANIFOLD gauges (g, H gauged); COREWISE does NOT (g == bare 𝒥ᵀr).'''
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    sm = _setup(kind, 'manifold', C)
                    self.assertTrue(sm['model'].gradient.is_gauged())
                    self.assertTrue(sm['model'].gn_hessian(_raw_step(sm)).is_gauged())

                    sc = _setup(kind, 'corewise', C)
                    bare = sc['model'].kind.transpose(sc['r'], sc['sample'], sc['base'].data, sc['model']._base_sweep)
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
                        gz_raw = model.kind.transpose(z, s['sample'], s['base'].data, model._base_sweep)
                        gz = geometry.project(t3m.T3Tangent(s['base'], bvf.T3Variations(*gz_raw)))  # Π 𝒥ᵀ z
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

    def test_same_base_guard(self):
        '''A trial tangent at a different base is a structural error (identity, not value), both geometries.'''
        for kind in KINDS:
            for geom_name in GEOMS:
                with self.subTest(kind=kind, geom=geom_name):
                    s = _setup(kind, geom_name, ())
                    other_x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS)
                    p_other = s['geometry'].randn(s['geometry'].base(other_x))
                    with self.assertRaises(ValueError):
                        s['model'].gn_hessian(p_other)
                    with self.assertRaises(ValueError):
                        s['model'].evaluate(p_other)

    def test_caching(self):
        '''The base sweep / gradient / objective are cached -- the reuse mechanism, computed once.'''
        model = _setup('apply', 'manifold', ())['model']
        self.assertIs(model._base_sweep, model._base_sweep)
        self.assertIs(model.gradient, model.gradient)
        self.assertIs(model.objective_value, model.objective_value)

    def test_matches_established_manifold(self):
        '''Manifold gradient/Hessian == the established bare T3Tangent transpose + MANIFOLD.project (apply).'''
        s = _setup('apply', 'manifold', ())
        base, ww, r, model = s['base'], s['sample'], s['r'], s['model']
        ref_g = t3m.MANIFOLD.project(t3m.T3Tangent.apply_transpose(r, ww, base, sum_over_probes=True))
        gd = model.gradient.variations.data
        for a, b in zip(gd[0] + gd[1], ref_g.variations.data[0] + ref_g.variations.data[1]):
            self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))
        V = t3m.MANIFOLD.randn(base)
        ref_HV = t3m.MANIFOLD.project(t3m.T3Tangent.apply_transpose(
            V.apply(ww), ww, base, sum_over_probes=True))
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


if __name__ == '__main__':
    unittest.main()
