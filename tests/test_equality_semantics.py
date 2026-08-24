# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The explicit-equality design (review R1-13, extended; ruled 2026-08-24).

Part 1 (this file's check classes): the two NAMED equality checks every runtime array-carrying
class offers -- ``allclose`` (mathematical / tolerance-aware, per stack element, norm-based with the
symmetric ``atol + rtol*max`` reference; for trains the represented TENSORS via the stable
norm-of-difference route; for frames the BASE POINT, gauge-invariant -- the pre-existing
``allclose`` family's semantics, now uniform across all twelve classes) and ``corewise_equal``
(bitwise representation).
Part 2 (TestExplicitEqOperator): ``==`` is intentionally undefined -- identity-True, else a
directive TypeError -- and the classes are unhashable; hash/eq belongs to the jit-cache-key (aux)
objects only."""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_frame_variations_format as ubvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.corewise as cw
import t3toolbox.safety as safety


def _fixtures():
    """One instance of each of the twelve classes (+ a same-value rebuild where meaningful)."""
    np.random.seed(0)
    x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1))
    fr, var = bvf.t3_orthogonal_representations(x)
    v = t3m.MANIFOLD.randn(fr)
    W = t3.T3Weights.from_t3svd(x)
    fw = bvf.T3FrameWeights.from_t3weights(W)
    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    uf, uvar = ubvf.ut3_orthogonal_representations(ux)
    uv = ut3m.UNIFORM_MANIFOLD.randn(uf)
    uW = ut3.UT3Weights.from_ut3svd(ux)
    ufw = ubvf.UT3FrameWeights.from_ut3weights(uW)
    return x, fr, var, v, W, fw, ux, uf, uvar, uv, uW, ufw


class TestNamedEqualityChecks(unittest.TestCase):

    def test_train_tensor_equality_is_gauge_and_padding_invariant(self):
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1))
        z = x.t3svd()[0]                                        # same tensor, different gauge
        self.assertTrue(bool(x.allclose(z)))
        self.assertFalse(x.corewise_equal(z))                   # ... but a different representation
        self.assertFalse(bool(x.allclose(x * (1.0 + 1e-3))))
        self.assertTrue(bool(x.allclose(x * (1.0 + 1e-3), rtol=1e-2)))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        uz = ut3.UniformTuckerTensorTrain.from_t3(x, n=4, r=4)  # same tensor, different padding
        self.assertTrue(bool(ux.allclose(uz)))
        self.assertFalse(ux.corewise_equal(uz))
        with self.assertRaises(ValueError):                     # different ambient space: raises
            x.allclose(t3.TuckerTensorTrain.randn((4, 4), (2, 2), (1, 2, 1)))

    def test_train_checks_are_per_stack_element(self):
        np.random.seed(0)
        x3 = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1), stack_shape=(3,))
        got = x3.allclose(x3 * 1.0)
        self.assertEqual(np.shape(got), (3,))
        self.assertTrue(bool(np.all(got)))

    def test_frame_equality_is_the_same_tangent_space_question(self):
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1))
        fr, _ = bvf.t3_orthogonal_representations(x)
        fr2, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain(*x.data))   # rebuilt
        fr_other, _ = bvf.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1)))
        self.assertTrue(bool(fr.allclose(fr2).all()))  # the jit-round-trip / rebuild case
        self.assertFalse(bool(fr.allclose(fr_other).all()))
        # the three frame questions: same BASE POINT (gauge-invariant) vs same FRAME vs bitwise
        frg = bvf.t3svd_orthogonal_representations(x)[0]        # same base point, t3svd gauge
        self.assertTrue(bool(fr.allclose(frg).all()))           # allclose: same base point
        self.assertFalse(safety.frames_equal(fr.data, frg.data))  # frames_equal: different FRAME
        self.assertFalse(fr.corewise_equal(frg))                # corewise: different representation
        with self.assertRaises(ValueError):
            fr.allclose(bvf.t3_orthogonal_representations(
                t3.TuckerTensorTrain.randn((4, 4), (2, 2), (1, 2, 1)))[0])
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        uf, _ = ubvf.ut3_orthogonal_representations(ux)
        uf2, _ = ubvf.ut3_orthogonal_representations(
            ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain(*x.data)))
        self.assertTrue(bool(uf.allclose(uf2).all()))
        self.assertTrue(uf.corewise_equal(uf2))                 # deterministic sweep: bitwise too

    def test_tangent_weights_variations_checks(self):
        x, fr, var, v, W, fw, ux, uf, uvar, uv, uW, ufw = _fixtures()
        for obj in (var, v, W, fw, uvar, uv, uW, ufw):
            with self.subTest(cls=type(obj).__name__):
                self.assertTrue(bool(np.all(obj.allclose(obj))))
                self.assertTrue(obj.corewise_equal(obj))
        self.assertFalse(bool(np.all(v.allclose(v * 1.001))))
        self.assertTrue(bool(np.all(v.allclose(v * (1.0 + 1e-12), rtol=1e-9))))
        vk = t3m.MANIFOLD.randn(fr, stack_shape=(2,))            # per-element over the tangent stack K
        self.assertEqual(np.shape(vk.allclose(vk * 1.0)), (2,))
        W2 = t3.T3Weights(tuple(w * 1.001 for w in W.tucker_weights), W.tt_weights)
        self.assertFalse(bool(W.allclose(W2)))
        self.assertFalse(W.corewise_equal(W2))

    def test_corewise_equal_helper(self):
        a = (np.ones(3), (np.zeros((2, 2)),))
        b = (np.ones(3), (np.zeros((2, 2)),))
        self.assertTrue(cw.corewise_equal(a, b))
        self.assertFalse(cw.corewise_equal(a, (np.ones(3),)))            # structure mismatch: False
        self.assertFalse(cw.corewise_equal(a, (np.ones(4), (np.zeros((2, 2)),))))


if __name__ == '__main__':
    unittest.main()
