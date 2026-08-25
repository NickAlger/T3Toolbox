# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Direct tests for the orthogonalization backend's untested public surface (Phase D of the 2026-08-22
review, finding R2-9; promoted from ``repros/R2/r2_04_orthogonalization.py``): ``t3_right_orthogonalize``,
the relative-to-core orthogonalizations, the five single-core SVD steps, and
``t3_up_orthogonalize_tt_cores`` -- each must preserve the represented tensor and attain its
orthogonality contract."""
import unittest
import numpy as np

import t3toolbox.tucker_tensor_train as t3
from t3toolbox.backend import t3_orthogonalization as O, t3_conversions as C

STRUCTS = [((4,), (3,), (1, 1)),
           ((4, 5), (3, 2), (1, 3, 1)),
           ((4, 5, 6, 3), (3, 2, 4, 2), (1, 2, 3, 2, 1)),
           ((3, 5, 6), (4, 2, 3), (2, 2, 3, 2))]          # unsquashed boundary tails
STACKS = [(), (2,), (2, 3)]


def _dense(d):
    return C.t3_to_dense(d)


def _relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def _tucker_res(B):
    return float(np.abs(np.einsum('...io,...jo->...ij', B, B) - np.eye(B.shape[-2])).max())


def _left_res(G):
    return float(np.abs(np.einsum('...aib,...aic->...bc', G, G) - np.eye(G.shape[-1])).max())


def _right_res(G):
    return float(np.abs(np.einsum('...aib,...cib->...ac', G, G) - np.eye(G.shape[-3])).max())


def _up_res(G):
    return float(np.abs(np.einsum('...aib,...ajb->...ij', G, G) - np.eye(G.shape[-2])).max())


class TestRightOrthogonalize(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_preserves_tensor_and_right_orthogonal(self):
        for shape, tr, ttr in STRUCTS:
            for ss in STACKS:
                with self.subTest(shape=shape, stack=ss):
                    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss).data
                    y = O.t3_right_orthogonalize(x)
                    self.assertLess(_relerr(_dense(y), _dense(x)), 1e-12)
                    self.assertLess(float(np.max(O.t3_orthogonality_residual(y, 'right'))), 1e-12)


class TestRelativeToCoreOrthogonalizations(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_relative_to_tucker_core(self):
        # everything orthogonal EXCEPT tucker core ii; tensor preserved; tt core ii up-orthonormal
        for shape, tr, ttr in STRUCTS[1:3]:
            for ss in [(), (2,)]:
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss).data
                X = _dense(x)
                for ii in range(len(shape)):
                    with self.subTest(shape=shape, stack=ss, ii=ii):
                        tk, tt = O.t3_orthogonalize_relative_to_tucker_core(x, ii)
                        self.assertLess(_relerr(_dense((tk, tt)), X), 1e-12)
                        res = max([_tucker_res(B) for j, B in enumerate(tk) if j != ii]
                                  + [_left_res(G) for G in tt[:ii]]
                                  + [_right_res(G) for G in tt[ii + 1:]] + [_up_res(tt[ii])])
                        self.assertLess(res, 1e-12)

    def test_relative_to_tt_core(self):
        for shape, tr, ttr in STRUCTS[1:3]:
            for ss in [(), (2,)]:
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss).data
                X = _dense(x)
                for ii in range(len(shape)):
                    with self.subTest(shape=shape, stack=ss, ii=ii):
                        tk, tt = O.t3_orthogonalize_relative_to_tt_core(x, ii)
                        self.assertLess(_relerr(_dense((tk, tt)), X), 1e-12)
                        res = max([_tucker_res(B) for B in tk]
                                  + [_left_res(G) for G in tt[:ii]]
                                  + [_right_res(G) for G in tt[ii + 1:]])
                        self.assertLess(res, 1e-12)


class TestSingleCoreSvdSteps(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_each_step_preserves_tensor(self):
        steps = [('down_svd_tucker', O.t3_down_svd_tucker_core), ('left_svd_tt', O.t3_left_svd_tt_core),
                 ('right_svd_tt', O.t3_right_svd_tt_core), ('down_svd_tt', O.t3_down_svd_tt_core),
                 ('up_svd_tt', O.t3_up_svd_tt_core)]
        for shape, tr, ttr in STRUCTS[1:3]:
            for ss in [(), (2,)]:
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss).data
                X = _dense(x)
                for ii in range(len(shape)):
                    for name, f in steps:
                        with self.subTest(shape=shape, stack=ss, ii=ii, step=name):
                            nx, _sv = f(x, ii)
                            self.assertLess(_relerr(_dense(nx), X), 1e-12)


class TestUpOrthogonalizeTtCores(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_variation_outer_split(self):
        # (variations, outer) represent the same tensor; the outer cores are up-orthonormal
        for shape, tr, ttr in STRUCTS:
            for ss in STACKS:
                with self.subTest(shape=shape, stack=ss):
                    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss).data
                    V, Oc = O.t3_up_orthogonalize_tt_cores(x)
                    self.assertLess(_relerr(_dense((V, Oc)), _dense(x)), 1e-12)
                    self.assertLess(max(_up_res(G) for G in Oc), 1e-12)
