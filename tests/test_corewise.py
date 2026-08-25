# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Direct tests for the small ``corewise`` helpers the 2026-08-22 review found untested (Phase D,
finding R4-15): ``corewise_err``, ``corewise_logical_not``, ``corewise_stack_scale``,
``corewise_stack_sum``."""
import unittest
import numpy as np

import t3toolbox.corewise as cw


def _tree():
    np.random.seed(3)
    return ((np.random.randn(2, 3, 4, 5), np.random.randn(2, 3, 3)),
            (np.random.randn(2, 3, 2, 4, 2),))


class TestCorewiseHelpers(unittest.TestCase):
    def test_corewise_err(self):
        X = _tree()
        self.assertEqual(float(cw.corewise_err(X, X)), 0.0)
        Y = cw.corewise_scale(X, 2.0)
        self.assertTrue(np.isclose(float(cw.corewise_err(X, Y)), float(cw.corewise_norm(X))))

    def test_corewise_logical_not(self):
        X = ((np.array([True, False]),), (np.array([[False, True], [True, True]]),))
        out = cw.corewise_logical_not(X)
        self.assertTrue(np.array_equal(out[0][0], np.array([False, True])))
        self.assertTrue(np.array_equal(out[1][0], np.array([[True, False], [False, False]])))

    def test_corewise_stack_scale(self):
        X = _tree()                                            # every leaf: stack (2, 3) leading
        s = np.random.randn(2, 3)
        out = cw.corewise_stack_scale(X, s)
        for fam_out, fam_in in zip(out, X):
            for a, b in zip(fam_out, fam_in):
                ref = b * s.reshape(s.shape + (1,) * (b.ndim - 2))
                self.assertTrue(np.allclose(a, ref))
        # a 0-d factor scales uniformly
        out2 = cw.corewise_stack_scale(X, np.asarray(2.0))
        self.assertTrue(np.allclose(out2[0][0], 2.0 * X[0][0]))

    def test_corewise_stack_sum(self):
        X = _tree()
        out_all = cw.corewise_stack_sum(X, None, 2)            # axis=None: all stack axes summed
        self.assertTrue(np.allclose(out_all[0][0], X[0][0].sum(axis=(0, 1))))
        out0 = cw.corewise_stack_sum(X, 0, 2)
        self.assertTrue(np.allclose(out0[1][0], X[1][0].sum(axis=0)))
        out_neg = cw.corewise_stack_sum(X, -1, 2)              # negative wraps relative to n_stack
        self.assertTrue(np.allclose(out_neg[0][1], X[0][1].sum(axis=1)))
