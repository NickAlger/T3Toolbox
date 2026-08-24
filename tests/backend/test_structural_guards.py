# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Regression tests for the 2026-08-22 review's obscure-error cluster (E items, backend half):
malformed inputs must fail with a STRUCTURAL error naming the problem, not a deep numpy/einsum
error, a silent truncation, or an infinite recursion. One test per finding; the frontend half of
the cluster lives beside its classes (``test_uniform_tucker_tensor_train`` / ``test_weighted`` /
``test_manifold``)."""
import unittest

import numpy as np

import t3toolbox.backend.common as common
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.t3_linalg as t3_linalg
import t3toolbox.tucker_tensor_train as t3


class TestTreeHelperGuards(unittest.TestCase):
    def test_str_leaves_do_not_recurse_forever(self):
        """R2-8: a str IS a Sequence of strs -- the tree recursions must treat it as a leaf."""
        self.assertFalse(common.tree_contains_jax('abc'))
        self.assertFalse(common.tree_contains_jax((np.ones(2), 'abc')))
        self.assertEqual(stacking.tree_depth('abc'), 0)
        self.assertEqual(stacking.get_first_leaf('abc'), 'abc')
        out = common.tree_to_jax((np.ones(2),))          # smoke: no recursion through array leaves
        self.assertEqual(len(out), 1)

    def test_tree_zip_rejects_structure_mismatch(self):
        """R2-12: a silent zip truncates to the shorter branch; now a structural ValueError."""
        ok = stacking.tree_zip((1, (2, 3)), ('a', ('b', 'c')))
        self.assertEqual(ok, ((1, 'a'), ((2, 'b'), (3, 'c'))))
        with self.assertRaises(ValueError):
            stacking.tree_zip((np.ones(2), np.ones(2)), (np.ones(2),))       # length mismatch
        with self.assertRaises(ValueError):
            stacking.tree_zip((np.ones(2), (np.ones(2),)), (np.ones(2), np.ones(2)))  # depth mismatch


class TestLinalgCornerGuards(unittest.TestCase):
    def test_truncated_svd_min_gt_max_raises(self):
        """R3-6: min_rank used to silently override max_rank."""
        A = np.random.default_rng(0).standard_normal((8, 6))
        with self.assertRaises(ValueError):
            linalg.truncated_svd(A, min_rank=5, max_rank=2)
        U, ss, Vt = linalg.truncated_svd(A, min_rank=2, max_rank=2)          # consistent pair still fine
        self.assertEqual(ss.shape[-1], 2)

    def test_inner_product_accepts_list_tuple_mixes(self):
        """R2-13: the family concat used to TypeError on a list/tuple mix."""
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn((4, 5), (2, 2), (1, 2, 1))
        got = t3_linalg.t3_inner_product((list(x.data[0]), tuple(x.data[1])), x.data)
        ref = t3_linalg.t3_inner_product(x.data, x.data)
        self.assertLess(abs(float(got) - float(ref)), 1e-12 * (abs(float(ref)) + 1))

    def test_sum_stack_accepts_ndarray_axis(self):
        """R2-13: t3_sum_stack(axis=np.array(...)) used to fall into the scalar branch."""
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn((4, 5), (2, 2), (1, 2, 1), stack_shape=(2, 3))
        a = t3_linalg.t3_sum_stack(x.data, axis=np.array([0]))
        b = t3_linalg.t3_sum_stack(x.data, axis=0)
        for ca, cb in zip(a[0] + a[1], b[0] + b[1]):
            self.assertTrue(np.allclose(np.asarray(ca), np.asarray(cb)))


if __name__ == '__main__':
    unittest.main()
