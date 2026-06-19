'''Tests for the Gauss-Newton fitting operators (``backend/fitting.py``, apply, tangent).

The headline oracle is **exact dense ground truth**: because the sampling forward is linear in the
ambient tensor, the least-squares objective is exactly quadratic, so the Gauss-Newton model is the exact
restriction of the objective to the affine tangent space ``x + dense(Π p)``. We also check the gauge
correctness (outputs gauged, operator symmetric), the razor self-containment (raw ``p`` == gauged ``Π p``),
and the ``J`` / ``Jᵀ`` adjoint identity. See ``docs/fitting_plan.md`` §9.
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


def apply_dense(T, ww, n_c):
    '''Sample-space all-modes apply of a dense (possibly C-stacked) tensor: ``T(w_1,...,w_d)`` per
    sample. ``T`` has shape ``C + (N_1,...,N_d)``, each ``ww[i]`` has shape ``(W,) + (N_i,)`` (one W
    axis); returns shape ``W + C`` (base-inner). The ground-truth ``𝒥`` for the dense oracle.'''
    d = len(ww)
    c_axes = list(range(n_c))
    mode_axes = [n_c + i for i in range(d)]
    w_axis = n_c + d
    ops = [T, c_axes + mode_axes]
    for i in range(d):
        ops += [ww[i], [w_axis, mode_axes[i]]]
    ops += [[w_axis] + c_axes]                       # output W + C
    return np.einsum(*ops)


def gauged(base, variation):
    return t3m.T3Tangent(base, bvf.T3Variations(*variation)).is_gauged()


class TestApplyGaussNewtonModel(unittest.TestCase):
    SHAPE = (7, 8, 9)
    TUCKER_RANKS = (3, 4, 2)
    TT_RANKS = (1, 2, 3, 1)
    N_SAMPLES = 20

    def _setup(self, C):
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn(self.SHAPE, self.TUCKER_RANKS, self.TT_RANKS, stack_shape=C)
        base, _ = bvf.t3_orthogonal_representations(x)
        m = self.N_SAMPLES
        ww = [np.random.randn(m, N) for N in self.SHAPE]
        r = np.random.randn(*((m,) + C))                    # residual, W=(m,), then C
        sweep = probing.precompute_apply_base_sweep(base.data, ww)
        n_c, n_w = len(C), 1
        c = 0.5 * np.sum(r ** 2, axis=tuple(range(n_w)))    # objective value, shape C
        g = fb.compute_gradient(r, ww, base.data, sweep)      # gauged gradient
        # an UN-gauged random trial tangent -> exercises Π (sensitivity) and the razor
        p = t3m.T3Tangent.randn(base, apply_gauge_projection=False).variations.data
        Pp = tangent_operations.orthogonal_gauge_projection(base.data, p)
        return base, ww, r, sweep, c, g, p, Pp, n_c, n_w

    def test_dense_truth_model_value(self):
        '''HEADLINE: m(p) == ½‖r + 𝒥(Πp)‖² computed from the dense gauge-projected tangent.'''
        for C in [(), (2,)]:
            with self.subTest(C=C):
                base, ww, r, sweep, c, g, p, Pp, n_c, n_w = self._setup(C)
                Pp_dense = t3m.T3Tangent(base, bvf.T3Variations(*Pp)).to_dense()
                oracle = 0.5 * np.sum((r + apply_dense(Pp_dense, ww, n_c)) ** 2, axis=tuple(range(n_w)))
                mval = fb.quadratic_model_value(p, ww, base.data, sweep, g, c)
                self.assertTrue(np.allclose(mval, oracle, rtol=0, atol=1e-9),
                                f'C={C}: {mval} vs {oracle}')

    def test_two_form_consistency(self):
        '''m(p) == c + ⟨g, Πp⟩ + ½⟨Πp, H Πp⟩ (the gn_hessian-based quadratic term).'''
        for C in [(), (2,)]:
            with self.subTest(C=C):
                base, ww, r, sweep, c, g, p, Pp, n_c, n_w = self._setup(C)
                Hp = fb.apply_gn_hessian(p, ww, base.data, sweep)
                two_form = (c + cw.corewise_stack_dot(g, Pp, n_c)
                            + 0.5 * cw.corewise_stack_dot(Hp, Pp, n_c))
                mval = fb.quadratic_model_value(p, ww, base.data, sweep, g, c)
                self.assertTrue(np.allclose(two_form, mval, rtol=0, atol=1e-9))

    def test_outputs_gauged_and_symmetric(self):
        '''gradient and gn_hessian outputs are gauged; the GN operator is symmetric.'''
        for C in [(), (2,)]:
            with self.subTest(C=C):
                base, ww, r, sweep, c, g, p, Pp, n_c, n_w = self._setup(C)
                Hp = fb.apply_gn_hessian(p, ww, base.data, sweep)
                self.assertTrue(gauged(base, g))
                self.assertTrue(gauged(base, Hp))
                q = t3m.T3Tangent.randn(base, apply_gauge_projection=False).variations.data
                Hq = fb.apply_gn_hessian(q, ww, base.data, sweep)
                lhs = cw.corewise_stack_dot(q, Hp, n_c)         # ⟨q, Hp⟩
                rhs = cw.corewise_stack_dot(p, Hq, n_c)         # ⟨p, Hq⟩
                self.assertTrue(np.allclose(lhs, rhs, rtol=0, atol=1e-9))

    def test_razor_self_containment(self):
        '''The functions apply Π themselves: a raw p gives the same result as the gauge-projected Πp.'''
        for C in [(), (2,)]:
            with self.subTest(C=C):
                base, ww, r, sweep, c, g, p, Pp, n_c, n_w = self._setup(C)
                m_raw = fb.quadratic_model_value(p,  ww, base.data, sweep, g, c)
                m_gau = fb.quadratic_model_value(Pp, ww, base.data, sweep, g, c)
                self.assertTrue(np.allclose(m_raw, m_gau, rtol=0, atol=1e-9))
                H_raw = fb.apply_gn_hessian(p,  ww, base.data, sweep)
                H_gau = fb.apply_gn_hessian(Pp, ww, base.data, sweep)
                for a, b in zip(H_raw[0] + H_raw[1], H_gau[0] + H_gau[1]):
                    self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-9))

    def test_jacobian_gradient_adjoint(self):
        '''Adjoint identity ⟨z, J p⟩_samples == ⟨Π𝒥ᵀz, p⟩_corewise (J = 𝒥∘Π, gradient = Π∘𝒥ᵀ).'''
        for C in [(), (2,)]:
            with self.subTest(C=C):
                base, ww, r, sweep, c, g, p, Pp, n_c, n_w = self._setup(C)
                Jp = fb.apply_jacobian(p, ww, base.data, sweep)         # 𝒥 Π p, shape W+C
                z = np.random.randn(*Jp.shape)
                gz = fb.compute_gradient(z, ww, base.data, sweep)         # Π 𝒥ᵀ z
                lhs = np.sum(z * Jp, axis=tuple(range(n_w)))            # ⟨z, Jp⟩_samples, keep C
                rhs = cw.corewise_stack_dot(gz, p, n_c)                 # ⟨gz, p⟩_corewise
                self.assertTrue(np.allclose(lhs, rhs, rtol=0, atol=1e-9))


class TestGaussNewtonModelFrontend(unittest.TestCase):
    '''The GaussNewtonModel dataclass: delegation, the same-base guard, and the cached base-sweep reuse.'''

    def _model(self, C):
        np.random.seed(1)
        x = t3.TuckerTensorTrain.randn((7, 8, 9), (3, 4, 2), (1, 2, 3, 1), stack_shape=C)
        base, _ = bvf.t3_orthogonal_representations(x)
        ww = [np.random.randn(20, N) for N in (7, 8, 9)]
        r = np.random.randn(*((20,) + C))
        return fitting.GaussNewtonModel(base, ww, r), base, ww, r

    def test_dense_truth_through_model(self):
        '''End-to-end: model.evaluate(p) == ½‖r + 𝒥(Πp)‖² (dense oracle), via the frontend.'''
        for C in [(), (2,)]:
            with self.subTest(C=C):
                model, base, ww, r = self._model(C)
                p = t3m.T3Tangent.randn(base, apply_gauge_projection=False)   # un-gauged -> tests Π
                Pp = tangent_operations.orthogonal_gauge_projection(base.data, p.variations.data)
                Pp_dense = t3m.T3Tangent(base, bvf.T3Variations(*Pp)).to_dense()
                oracle = 0.5 * np.sum((r + apply_dense(Pp_dense, ww, len(C))) ** 2, axis=0)
                self.assertTrue(np.allclose(model.evaluate(p), oracle, rtol=0, atol=1e-9))

    def test_delegates_to_backend(self):
        '''The model's properties/methods equal the backend functions it wraps.'''
        model, base, ww, r = self._model(())
        sweep = probing.precompute_apply_base_sweep(base.data, ww)
        self.assertAlmostEqual(float(model.objective_value), 0.5 * float(np.sum(r ** 2)), places=10)
        g_back = fb.compute_gradient(r, ww, base.data, sweep)
        for a, b in zip(model.gradient.variations.data[0] + model.gradient.variations.data[1],
                        g_back[0] + g_back[1]):
            self.assertTrue(np.allclose(a, b))
        p = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
        h_back = fb.apply_gn_hessian(p.variations.data, ww, base.data, sweep)
        for a, b in zip(model.gn_hessian(p).variations.data[0] + model.gn_hessian(p).variations.data[1],
                        h_back[0] + h_back[1]):
            self.assertTrue(np.allclose(a, b))

    def test_same_base_guard(self):
        '''A trial tangent at a different base is a structural error (identity, not value).'''
        model, base, ww, r = self._model(())
        other, _ = bvf.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn((7, 8, 9), (3, 4, 2), (1, 2, 3, 1)))
        p_other = t3m.T3Tangent.randn(other)
        with self.assertRaises(ValueError):
            model.gn_hessian(p_other)
        with self.assertRaises(ValueError):
            model.evaluate(p_other)

    def test_matches_reference_operators(self):
        '''The model reproduces the closure-based reference operators (the manifold apply_transpose used
        by examples/fit_hilbert_tensor_newton_cg.py) on the gauged subspace CG lives in: the gradient
        bit-for-bit, and the GN Hessian on a gauged input. (On an UN-gauged input the model differs --
        it is the proper symmetric Π𝒥ᵀ𝒥Π; the reference projects only the output.)'''
        model, base, ww, r = self._model(())
        ref_g = t3m.T3Tangent.apply_transpose(r, ww, base, sum_over_probes=True).orthogonal_gauge_projection()
        for a, b in zip(model.gradient.variations.data[0] + model.gradient.variations.data[1],
                        ref_g.variations.data[0] + ref_g.variations.data[1]):
            self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))
        V = t3m.T3Tangent.randn(base, apply_gauge_projection=True)        # gauged -> operators must agree
        ref_HV = t3m.T3Tangent.apply_transpose(
            V.apply(ww), ww, base, sum_over_probes=True).orthogonal_gauge_projection()
        Hv = model.gn_hessian(V)
        for a, b in zip(Hv.variations.data[0] + Hv.variations.data[1],
                        ref_HV.variations.data[0] + ref_HV.variations.data[1]):
            self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12))

    def test_base_sweep_cached(self):
        '''The base sweep (and gradient/objective) are cached -- the reuse mechanism, computed once.'''
        model, base, ww, r = self._model(())
        self.assertIs(model._base_sweep, model._base_sweep)     # same object -> cached, not recomputed
        self.assertIs(model.gradient, model.gradient)
        self.assertIs(model.objective_value, model.objective_value)


if __name__ == '__main__':
    unittest.main()
