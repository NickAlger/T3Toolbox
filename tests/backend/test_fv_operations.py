# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Direct tests for the ``fv_operations`` backend surface (Phase D of the 2026-08-22 review, finding
R4-15; promoted from ``repros/R4/r4_04_fv_ops.py``): the variations constructors
(``fv_variations_{zeros,randn,unit,from_vector}``), ``fv_variation_shapes``, and ``fv_frame_reverse``."""
import unittest
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.tv_operations as tv

_CASES = [((5, 6, 7), (3, 2, 4), (1, 2, 3, 1)),              # asymmetric
          ((6, 6, 6), (4, 4, 4), (1, 2, 2, 1))]              # over-ranked: nU != nD slack


def _tree_relerr(a, b):
    return float(cw.corewise_norm(cw.corewise_sub(a, b))) / max(float(cw.corewise_norm(b)), 1e-300)


class TestVariationsConstructors(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)

    def test_shapes_zeros_randn_unit_from_vector(self):
        K = (2,)
        for struct in _CASES:
            for C in [(), (2, 3)]:
                with self.subTest(struct=struct, C=C):
                    x = t3.TuckerTensorTrain.randn(*struct, stack_shape=C)
                    frame, _ = bvf.t3_orthogonal_representations(x)
                    vs = frame.variation_shapes
                    self.assertEqual(fvo.fv_variation_shapes(frame.data), vs)
                    z = fvo.fv_variations_zeros(vs, K + C)
                    r = fvo.fv_variations_randn(vs, K + C)
                    for fam, shapes in ((0, vs[0]), (1, vs[1])):
                        self.assertTrue(all(c.shape == K + C + s for c, s in zip(z[fam], shapes)))
                        self.assertTrue(all(c.shape == K + C + s for c, s in zip(r[fam], shapes)))
                    self.assertEqual(float(cw.corewise_norm(z)), 0.0)
                    # unit: a single 1, broadcast over the K+C stack
                    i = len(struct[0]) - 1
                    idx = tuple(s - 1 for s in vs[1][i])
                    u = fvo.fv_variations_unit(vs, (True, i, idx), K + C)
                    self.assertEqual(float(np.sum(u[1][i])), float(np.prod(K + C)))
                    self.assertTrue(np.isclose(float(cw.corewise_norm(u)) ** 2, np.prod(K + C)))
                    self.assertTrue(bool(np.all(u[1][i][(Ellipsis,) + idx] == 1.0)))
                    # from_vector inverts t3_to_vector on a K+C stack, exactly
                    back = fvo.fv_variations_from_vector(t3c.t3_to_vector(r), vs, K + C)
                    self.assertEqual(_tree_relerr(r, back), 0.0)
                    # the frontend twin round-trip
                    tng = t3m.COREWISE.randn(frame, stack_shape=K)
                    tng2 = t3m.T3Tangent.from_vector(tng.to_vector(), frame, tangent_stack_shape=K)
                    self.assertEqual(_tree_relerr(tng.variations.data, tng2.variations.data), 0.0)


class TestFrameReverse(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)

    def test_reverse_properties(self):
        for struct in _CASES:
            for C in [(), (2, 3)]:
                with self.subTest(struct=struct, C=C):
                    shape = struct[0]
                    d = len(shape)
                    x = t3.TuckerTensorTrain.randn(*struct, stack_shape=C)
                    frame, variations = bvf.t3_orthogonal_representations(x)
                    rf = bvf.T3Frame(*fvo.fv_frame_reverse(frame.data))
                    self.assertTrue(bool(rf.is_orthogonal().all()))
                    self.assertTrue(bool(rf.is_consistent().all()))
                    perm = tuple(range(len(C))) + tuple(len(C) + d - 1 - m for m in range(d))
                    self.assertLess(float(np.linalg.norm(np.transpose(frame.to_dense(), perm)
                                                         - rf.to_dense())),
                                    1e-12 * np.linalg.norm(frame.to_dense()))
                    self.assertEqual(_tree_relerr(frame.data, fvo.fv_frame_reverse(rf.data)), 0.0)
                    # the frontend twin: T3Tangent.reverse commutes with to_dense
                    tr_ = t3m.T3Tangent(frame, variations).reverse()
                    ref = np.transpose(tv.tv_to_dense(frame.data, variations.data), perm)
                    self.assertLess(float(np.linalg.norm(ref - tr_.to_dense())),
                                    1e-12 * max(np.linalg.norm(ref), 1e-300))
