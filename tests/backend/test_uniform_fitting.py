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
import t3toolbox.backend.uniform_fitting as uf

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


if __name__ == '__main__':
    unittest.main()
