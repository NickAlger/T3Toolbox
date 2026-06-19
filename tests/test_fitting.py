'''Tests for the Gauss-Newton fitting operators (``backend/fitting.py`` + ``fitting.py``; apply/entries/probe).

The headline oracle is **exact dense ground truth**: because the sampling forward is linear in the
ambient tensor, the least-squares objective is exactly quadratic, so the Gauss-Newton model is the exact
restriction of the objective to the affine tangent space ``x + dense(Π p)``. We also check the gauge
correctness (outputs gauged, operator symmetric), the razor self-containment (raw ``p`` == gauged ``Π p``),
and the ``J`` / ``Jᵀ`` adjoint identity -- parameterized over the sampling kind. See ``docs/fitting_plan.md`` §9.
'''

import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.probing as probing
import t3toolbox.backend.tangent_operations as tangent_operations
import t3toolbox.backend.fitting as fb
import t3toolbox.fitting as fitting
import t3toolbox.corewise as cw

SHAPE = (7, 8, 9)
TUCKER_RANKS = (3, 4, 2)
TT_RANKS = (1, 2, 3, 1)
N_SAMPLES = 20
KINDS = ('apply', 'entries', 'probe')


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


def gauged(base, variation):
    return t3m.T3Tangent(base, bvf.T3Variations(*variation)).is_gauged()


def _kind_setup(kind, C):
    '''Build a base + a kind-specific sample + residual, and bind the backend ops, the dense forward
    oracle, and kind-aware sample-space reducers (samp_add / samp_dot / rand_like). One dict per call.'''
    np.random.seed(0)
    x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS, stack_shape=C)
    base, _ = bvf.t3_orthogonal_representations(x)
    m, n_c, n_w = N_SAMPLES, len(C), 1
    if kind == 'apply':
        sample = [np.random.randn(m, N) for N in SHAPE]
        sweep = probing.precompute_base_sweep(base.data, sample)
        ops = (fb.apply_jacobian, fb.apply_gradient, fb.apply_gn_hessian, fb.apply_model_value)
        dense_fwd = lambda T: apply_dense(T, sample, n_c)
        r = np.random.randn(*((m,) + C))
        samp_dot = lambda a, b: np.sum(a * b, axis=tuple(range(n_w)))
        samp_add = lambda a, b: a + b
        rand_like = lambda v: np.random.randn(*v.shape)
    elif kind == 'entries':
        sample = np.stack([np.random.randint(0, N, size=m) for N in SHAPE])   # (d,)+W
        sweep = probing.precompute_entries_base_sweep(base.data, sample)
        ops = (fb.entries_jacobian, fb.entries_gradient, fb.entries_gn_hessian, fb.entries_model_value)
        dense_fwd = lambda T: entries_dense(T, sample, n_c)
        r = np.random.randn(*((m,) + C))
        samp_dot = lambda a, b: np.sum(a * b, axis=tuple(range(n_w)))
        samp_add = lambda a, b: a + b
        rand_like = lambda v: np.random.randn(*v.shape)
    else:  # probe -- vector-valued (one free mode each)
        sample = [np.random.randn(m, N) for N in SHAPE]
        sweep = probing.precompute_base_sweep(base.data, sample)
        ops = (fb.probe_jacobian, fb.probe_gradient, fb.probe_gn_hessian, fb.probe_model_value)
        dense_fwd = lambda T: probe_dense(T, sample, n_c)
        r = [np.random.randn(*((m,) + C + (N,))) for N in SHAPE]
        samp_dot = lambda a, b: sum(np.sum(ai * bi, axis=tuple(range(n_w)) + (ai.ndim - 1,))
                                    for ai, bi in zip(a, b))
        samp_add = lambda a, b: [ai + bi for ai, bi in zip(a, b)]
        rand_like = lambda v: [np.random.randn(*vi.shape) for vi in v]
    jac, grad, gnh, mval = ops
    c = 0.5 * samp_dot(r, r)
    g = grad(r, sample, base.data, sweep)
    p = t3m.T3Tangent.randn(base, apply_gauge_projection=False).variations.data   # UN-gauged -> tests Π
    Pp = tangent_operations.orthogonal_gauge_projection(base.data, p)
    return dict(base=base, sample=sample, sweep=sweep, r=r, c=c, g=g, p=p, Pp=Pp, n_c=n_c, n_w=n_w,
                jac=jac, grad=grad, gnh=gnh, mval=mval, dense_fwd=dense_fwd,
                samp_dot=samp_dot, samp_add=samp_add, rand_like=rand_like)


class TestGaussNewtonBackend(unittest.TestCase):
    '''The backend operators, parameterized over the sampling kind and the base stack C.'''

    def test_dense_truth_model_value(self):
        '''HEADLINE: m(p) == ½‖r + 𝒥(Πp)‖² computed from the dense gauge-projected tangent.'''
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    s = _kind_setup(kind, C)
                    Pp_dense = t3m.T3Tangent(s['base'], bvf.T3Variations(*s['Pp'])).to_dense()
                    res = s['samp_add'](s['r'], s['dense_fwd'](Pp_dense))
                    oracle = 0.5 * s['samp_dot'](res, res)
                    mval = s['mval'](s['p'], s['sample'], s['base'].data, s['sweep'], s['g'], s['c'])
                    self.assertTrue(np.allclose(mval, oracle, rtol=0, atol=1e-9), f'{kind} C={C}')

    def test_two_form_consistency(self):
        '''m(p) == c + ⟨g, Πp⟩ + ½⟨Πp, H Πp⟩ (the gn_hessian-based quadratic term).'''
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    s = _kind_setup(kind, C)
                    Hp = s['gnh'](s['p'], s['sample'], s['base'].data, s['sweep'])
                    two_form = (s['c'] + cw.corewise_stack_dot(s['g'], s['Pp'], s['n_c'])
                                + 0.5 * cw.corewise_stack_dot(Hp, s['Pp'], s['n_c']))
                    mval = s['mval'](s['p'], s['sample'], s['base'].data, s['sweep'], s['g'], s['c'])
                    self.assertTrue(np.allclose(two_form, mval, rtol=0, atol=1e-9))

    def test_outputs_gauged_and_symmetric(self):
        '''gradient and gn_hessian outputs are gauged; the GN operator is symmetric.'''
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    s = _kind_setup(kind, C)
                    Hp = s['gnh'](s['p'], s['sample'], s['base'].data, s['sweep'])
                    self.assertTrue(gauged(s['base'], s['g']))
                    self.assertTrue(gauged(s['base'], Hp))
                    q = t3m.T3Tangent.randn(s['base'], apply_gauge_projection=False).variations.data
                    Hq = s['gnh'](q, s['sample'], s['base'].data, s['sweep'])
                    lhs = cw.corewise_stack_dot(q, Hp, s['n_c'])
                    rhs = cw.corewise_stack_dot(s['p'], Hq, s['n_c'])
                    self.assertTrue(np.allclose(lhs, rhs, rtol=0, atol=1e-9))

    def test_razor_self_containment(self):
        '''The functions apply Π themselves: a raw p gives the same result as the gauge-projected Πp.'''
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    s = _kind_setup(kind, C)
                    args = (s['sample'], s['base'].data, s['sweep'])
                    m_raw = s['mval'](s['p'], *args, s['g'], s['c'])
                    m_gau = s['mval'](s['Pp'], *args, s['g'], s['c'])
                    self.assertTrue(np.allclose(m_raw, m_gau, rtol=0, atol=1e-9))
                    H_raw = s['gnh'](s['p'], *args)
                    H_gau = s['gnh'](s['Pp'], *args)
                    for a, b in zip(H_raw[0] + H_raw[1], H_gau[0] + H_gau[1]):
                        self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-9))

    def test_jacobian_gradient_adjoint(self):
        '''Adjoint identity ⟨z, J p⟩_samples == ⟨Π𝒥ᵀz, p⟩_corewise (J = 𝒥∘Π, gradient = Π∘𝒥ᵀ).'''
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    s = _kind_setup(kind, C)
                    Jp = s['jac'](s['p'], s['sample'], s['base'].data, s['sweep'])
                    z = s['rand_like'](Jp)
                    gz = s['grad'](z, s['sample'], s['base'].data, s['sweep'])
                    lhs = s['samp_dot'](z, Jp)
                    rhs = cw.corewise_stack_dot(gz, s['p'], s['n_c'])
                    self.assertTrue(np.allclose(lhs, rhs, rtol=0, atol=1e-9))


_MODEL_CLS = {'apply': fitting.ApplyGaussNewtonModel,
              'entries': fitting.EntriesGaussNewtonModel,
              'probe': fitting.ProbeGaussNewtonModel}


class TestGaussNewtonModelFrontend(unittest.TestCase):
    '''The frontend dataclasses: dense-truth through the model, delegation, the same-base guard, caching.'''

    def test_dense_truth_through_model(self):
        '''End-to-end: model.evaluate(p) == ½‖r + 𝒥(Πp)‖² (dense oracle), via the frontend, all kinds.'''
        for kind in KINDS:
            for C in [(), (2,)]:
                with self.subTest(kind=kind, C=C):
                    s = _kind_setup(kind, C)
                    model = _MODEL_CLS[kind](s['base'], s['sample'], s['r'])
                    p = t3m.T3Tangent.randn(s['base'], apply_gauge_projection=False)
                    Pp = tangent_operations.orthogonal_gauge_projection(s['base'].data, p.variations.data)
                    Pp_dense = t3m.T3Tangent(s['base'], bvf.T3Variations(*Pp)).to_dense()
                    res = s['samp_add'](s['r'], s['dense_fwd'](Pp_dense))
                    oracle = 0.5 * s['samp_dot'](res, res)
                    self.assertTrue(np.allclose(model.evaluate(p), oracle, rtol=0, atol=1e-9))

    def test_delegates_to_backend(self):
        '''The model's properties/methods equal the backend functions it wraps (all kinds).'''
        for kind in KINDS:
            with self.subTest(kind=kind):
                s = _kind_setup(kind, ())
                model = _MODEL_CLS[kind](s['base'], s['sample'], s['r'])
                self.assertAlmostEqual(float(model.objective_value), float(s['c']), places=10)
                gd = model.gradient.variations.data
                for a, b in zip(gd[0] + gd[1], s['g'][0] + s['g'][1]):
                    self.assertTrue(np.allclose(a, b))
                p = t3m.T3Tangent.randn(s['base'], apply_gauge_projection=False)
                h_back = s['gnh'](p.variations.data, s['sample'], s['base'].data, s['sweep'])
                hd = model.gn_hessian(p).variations.data
                for a, b in zip(hd[0] + hd[1], h_back[0] + h_back[1]):
                    self.assertTrue(np.allclose(a, b))

    def test_same_base_guard(self):
        '''A trial tangent at a different base is a structural error (identity, not value), all kinds.'''
        for kind in KINDS:
            with self.subTest(kind=kind):
                s = _kind_setup(kind, ())
                model = _MODEL_CLS[kind](s['base'], s['sample'], s['r'])
                other, _ = bvf.t3_orthogonal_representations(
                    t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS))
                p_other = t3m.T3Tangent.randn(other)
                with self.assertRaises(ValueError):
                    model.gn_hessian(p_other)
                with self.assertRaises(ValueError):
                    model.evaluate(p_other)

    def test_base_sweep_cached(self):
        '''The base sweep / gradient / objective are cached -- the reuse mechanism, computed once.'''
        s = _kind_setup('apply', ())
        model = _MODEL_CLS['apply'](s['base'], s['sample'], s['r'])
        self.assertIs(model._base_sweep, model._base_sweep)
        self.assertIs(model.gradient, model.gradient)
        self.assertIs(model.objective_value, model.objective_value)

    def test_matches_reference_operators(self):
        '''The apply model reproduces the closure-based reference operators (the example) on the gauged
        subspace: gradient bit-for-bit, GN Hessian on a gauged input.'''
        s = _kind_setup('apply', ())
        base, ww, r = s['base'], s['sample'], s['r']
        model = _MODEL_CLS['apply'](base, ww, r)
        ref_g = t3m.T3Tangent.apply_transpose(r, ww, base, sum_over_probes=True).orthogonal_gauge_projection()
        gd = model.gradient.variations.data
        for a, b in zip(gd[0] + gd[1], ref_g.variations.data[0] + ref_g.variations.data[1]):
            self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))
        V = t3m.T3Tangent.randn(base, apply_gauge_projection=True)
        ref_HV = t3m.T3Tangent.apply_transpose(
            V.apply(ww), ww, base, sum_over_probes=True).orthogonal_gauge_projection()
        Hv = model.gn_hessian(V)
        for a, b in zip(Hv.variations.data[0] + Hv.variations.data[1],
                        ref_HV.variations.data[0] + ref_HV.variations.data[1]):
            self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))


def corewise_dense_lin(x, dcores):
    '''The corewise linearization as a dense tensor: sum over single-core replacements
    ``Σ_core dense(x with that core -> its perturbation)`` -- this is exactly ``dense(J_corewise·dcores)``.'''
    dtucker, dtt = dcores
    tucker, tt = x.tucker_cores, x.tt_cores
    total = None
    for i in range(len(tucker)):
        Ti = t3.TuckerTensorTrain(tucker[:i] + (dtucker[i],) + tucker[i + 1:], tt).to_dense()
        total = Ti if total is None else total + Ti
    for i in range(len(tt)):
        Ti = t3.TuckerTensorTrain(tucker, tt[:i] + (dtt[i],) + tt[i + 1:]).to_dense()
        total = total + Ti
    return total


class TestCorewiseApply(unittest.TestCase):
    '''The corewise (free-core, NO Π) apply operators -- the matched-pair partner of the tangent ones.'''

    def _setup(self, C):
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER_RANKS, TT_RANKS, stack_shape=C)
        ww = [np.random.randn(N_SAMPLES, N) for N in SHAPE]
        r = np.random.randn(*((N_SAMPLES,) + C))
        sweep = fb.precompute_corewise_base_sweep(x.data, ww)
        p = (tuple(np.random.randn(*u.shape) for u in x.tucker_cores),     # a raw core-perturbation step
             tuple(np.random.randn(*cc.shape) for cc in x.tt_cores))
        n_c, n_w = len(C), 1
        c = 0.5 * np.sum(r ** 2, axis=tuple(range(n_w)))
        g = fb.apply_corewise_gradient(r, ww, sweep)
        return dict(x=x, ww=ww, r=r, sweep=sweep, p=p, n_c=n_c, n_w=n_w, c=c, g=g)

    def test_dense_truth(self):
        '''m(p) == ½‖r + apply_dense(Σ_core dense(core→δcore))‖² -- NO gauge projection (the no-Π oracle).'''
        for C in [(), (2,)]:
            with self.subTest(C=C):
                s = self._setup(C)
                Jp_dense = apply_dense(corewise_dense_lin(s['x'], s['p']), s['ww'], s['n_c'])
                oracle = 0.5 * np.sum((s['r'] + Jp_dense) ** 2, axis=tuple(range(s['n_w'])))
                mval = fb.apply_corewise_model_value(s['p'], s['ww'], s['x'].data, s['sweep'], s['g'], s['c'])
                self.assertTrue(np.allclose(mval, oracle, rtol=1e-9, atol=1e-9))  # large raw-core magnitudes

    def test_matches_established_corewise_transpose(self):
        '''gradient and gn_hessian match the established (jax.grad-verified) TuckerTensorTrain corewise
        transpose -- confirming the §6.3 substitution and that NO projection sneaks in.'''
        for C in [(), (2,)]:
            with self.subTest(C=C):
                s = self._setup(C)
                x, ww = s['x'], s['ww']
                g_ref = x.apply_corewise_transpose(s['r'], ww, sum_over_probes=True)
                for a, b in zip(s['g'][0] + s['g'][1], g_ref[0] + g_ref[1]):
                    self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))
                Jp = fb.apply_corewise_jacobian(s['p'], ww, x.data, s['sweep'])
                Hp_ref = x.apply_corewise_transpose(Jp, ww, sum_over_probes=True)
                Hp = fb.apply_corewise_gn_hessian(s['p'], ww, x.data, s['sweep'])
                for a, b in zip(Hp[0] + Hp[1], Hp_ref[0] + Hp_ref[1]):
                    self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))

    def test_two_form_and_adjoint(self):
        '''m == c + ⟨g,p⟩ + ½⟨p,Hp⟩ (corewise dots), and the J/Jᵀ adjoint identity.'''
        for C in [(), (2,)]:
            with self.subTest(C=C):
                s = self._setup(C)
                Hp = fb.apply_corewise_gn_hessian(s['p'], s['ww'], s['x'].data, s['sweep'])
                two_form = (s['c'] + cw.corewise_stack_dot(s['g'], s['p'], s['n_c'])
                            + 0.5 * cw.corewise_stack_dot(s['p'], Hp, s['n_c']))
                mval = fb.apply_corewise_model_value(s['p'], s['ww'], s['x'].data, s['sweep'], s['g'], s['c'])
                self.assertTrue(np.allclose(two_form, mval, rtol=1e-9, atol=1e-9))
                Jp = fb.apply_corewise_jacobian(s['p'], s['ww'], s['x'].data, s['sweep'])
                z = np.random.randn(*Jp.shape)
                gz = fb.apply_corewise_gradient(z, s['ww'], s['sweep'])
                lhs = np.sum(z * Jp, axis=tuple(range(s['n_w'])))
                rhs = cw.corewise_stack_dot(gz, s['p'], s['n_c'])
                self.assertTrue(np.allclose(lhs, rhs, rtol=1e-9, atol=1e-9))

    def test_frontend(self):
        '''CorewiseApplyGaussNewtonModel delegates to the backend (gradient / gn_hessian / evaluate).'''
        s = self._setup(())
        model = fitting.CorewiseApplyGaussNewtonModel(s['x'], s['ww'], s['r'])
        self.assertAlmostEqual(float(model.objective_value), float(s['c']), places=10)
        for a, b in zip(model.gradient[0] + model.gradient[1], s['g'][0] + s['g'][1]):
            self.assertTrue(np.allclose(a, b))
        Hp = fb.apply_corewise_gn_hessian(s['p'], s['ww'], s['x'].data, s['sweep'])
        for a, b in zip(model.gn_hessian(s['p'])[0] + model.gn_hessian(s['p'])[1], Hp[0] + Hp[1]):
            self.assertTrue(np.allclose(a, b))
        mval = fb.apply_corewise_model_value(s['p'], s['ww'], s['x'].data, s['sweep'], s['g'], s['c'])
        self.assertTrue(np.allclose(model.evaluate(s['p']), mval))


if __name__ == '__main__':
    unittest.main()
