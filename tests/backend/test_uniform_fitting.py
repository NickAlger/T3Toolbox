"""Tests for the uniform-layer fitting seams (backend/uniform_fitting.py) -- optimizers-on-uniform U2.

Correctness gold standard: the backend uniform ``GeometryOps`` factories (raw supercore pairs, masks
closed over) must reproduce the already-verified frontend ``UNIFORM_MANIFOLD`` / ``UNIFORM_COREWISE``
geometry ``.data`` path exactly (same math through the same ``ubv_tangent_operations`` primitives). The
factory captures the loop-invariant masks at ``x0``'s fixed rank; a second test evaluates the ops at a
DIFFERENT same-rank point to confirm the masks are correctly reused (the property the optimizer loop
relies on). numpy-only (jit dispatch is covered in test_dispatch)."""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_basis_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.ubv_tangent_operations as ubto

_STRUCT = ((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))   # (shape, tucker_ranks, tt_ranks)

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


def _basis_close(front_data, back_data):   # UT3Basis.data vs raw frame .data: supercores + shape + masks
    return (all(np.allclose(np.asarray(front_data[i]), np.asarray(back_data[i])) for i in range(4))
            and tuple(front_data[4]) == tuple(back_data[4])
            and all(np.array_equal(fm, bm) for fm, bm in zip(front_data[5], back_data[5])))


class TestUniformGeometryOps(unittest.TestCase):
    def _compare_ops(self, geom_front, ops, x):
        """Backend ops (on bare supercore pairs) vs the frontend geometry (on typed objects), at point x."""
        x_sc = (x.data[0], x.data[1])
        base_front, base_back = geom_front.base(x), ops.base(x_sc)
        self.assertTrue(_basis_close(base_front.data, base_back), 'base')

        v1 = ubv.UT3Variations.randn_like(base_front)     # ungauged variations at the base
        v2 = ubv.UT3Variations.randn_like(base_front)

        proj_front = geom_front.project(ut3m.UT3Tangent(base_front, v1)).variations.supercores
        self.assertTrue(_sc_close(proj_front, ops.project(base_back, v1.supercores)), 'project')

        retr_front = geom_front.retract(ut3m.UT3Tangent(base_front, v1))   # UniformTuckerTensorTrain
        self.assertTrue(_sc_close((retr_front.data[0], retr_front.data[1]),
                                  ops.retract(base_back, v1.supercores)), 'retract')

        # GeometryOps.inner is the check-free COORDINATE dot == UT3Tangent.corewise_inner (not HS)
        inner_front = float(ut3m.UT3Tangent(base_front, v1).corewise_inner(ut3m.UT3Tangent(base_front, v2)))
        self.assertTrue(np.isclose(inner_front, float(ops.inner(v1.supercores, v2.supercores))), 'inner')

    def test_ops_match_frontend(self):
        for name, (geom_front, factory) in _GEOMS.items():
            with self.subTest(geometry=name):
                x = _uniform_x(0)
                self._compare_ops(geom_front, factory(x.data), x)

    def test_masks_loop_invariant_across_points(self):
        # The factory captures the fixed-rank masks at x0; the ops must still match the frontend at a
        # DIFFERENT same-rank point (the base supercores change every optimizer step; the masks do not).
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
        self.base = ut3m.UNIFORM_MANIFOLD.base(self.x)
        self.var = ubv.UT3Variations.randn_like(self.base)
        self.base_r = self.base.to_t3basis()               # equivalent ragged frame
        self.var_r = self.var.to_t3variations()            # equivalent ragged variation
        self.vmask = uf._var_masks_from_base(self.base.data)

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
                sw_u = kind_u.precompute(self.base.data, sample)
                sw_r = kind_r.precompute(self.base_r.data, sample)
                fu = kind_u.forward(self.var.supercores, sample, self.base.data, sw_u)
                fr = kind_r.forward(self.var_r.data, sample, self.base_r.data, sw_r)
                if name == 'probe':
                    self.assertTrue(all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(fu, fr)))
                else:
                    self.assertTrue(np.allclose(np.asarray(fu), np.asarray(fr)))

    def test_point_forward_matches_ragged(self):
        for name, kind_r, is_index in self._KINDS:
            with self.subTest(kind=name):
                kind_u = uf.uniform_sampling_kind(name, self.x.data)
                sample = self._sample(is_index)
                x_r = self.base_r.to_t3()                   # the ragged point (== x)
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
                sw = kind_u.precompute(self.base.data, sample)
                fwd = kind_u.forward(self.var.supercores, sample, self.base.data, sw)
                if name == 'probe':
                    r = [np.random.randn(*np.asarray(z).shape) for z in fwd]
                    lhs = sum(float(np.sum(ri * np.asarray(zi))) for ri, zi in zip(r, fwd))
                else:
                    r = np.random.randn(*np.asarray(fwd).shape)
                    lhs = float(np.sum(r * np.asarray(fwd)))
                jt = kind_u.transpose(r, sample, self.base.data, sw)     # bare (dU, dG)
                rhs = float(ubto.ubv_corewise_inner(
                    (jt[0], jt[1], _STRUCT[0], self.vmask), self.var.data, 0))
                self.assertTrue(np.isclose(lhs, rhs), f"{name}: {lhs} != {rhs}")

    def test_forward_garbage_robust(self):
        # garbage in the masked-out variation padding must not change the forward (mask-once contracts it away)
        for name, kind_r, is_index in self._KINDS:
            with self.subTest(kind=name):
                kind_u = uf.uniform_sampling_kind(name, self.x.data)
                sample = self._sample(is_index)
                sw = kind_u.precompute(self.base.data, sample)
                clean = kind_u.forward(self.var.supercores, sample, self.base.data, sw)
                V = self.var
                tkv, ttv = V.supercores
                m_tkv, m_ttv = ubv.UT3Variations(np.ones_like(tkv), np.ones_like(ttv),
                                                 V.shape, V.masks).apply_masks().supercores
                ck_tkv, ck_ttv = V.apply_masks().supercores
                garb = (ck_tkv + 1e6 * (1.0 - m_tkv), ck_ttv + 1e6 * (1.0 - m_ttv))
                dirty = kind_u.forward(garb, sample, self.base.data, sw)
                if name == 'probe':
                    self.assertTrue(all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(clean, dirty)))
                else:
                    self.assertTrue(np.allclose(np.asarray(clean), np.asarray(dirty)))


if __name__ == '__main__':
    unittest.main()
