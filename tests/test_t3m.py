# Tests for elementwise multiplication of Tucker tensor trains (TuckerTensorTrain.t3m).
# Oracle = the dense elementwise product. Reusable harness; methods added per phase.
import unittest
import numpy as np

import t3toolbox.tucker_tensor_train as t3

# (shape, tucker_ranks, tt_ranks)
STRUCTURES = [
    ((10,),           (3,),         (1, 1)),
    ((8, 9),          (3, 4),       (1, 2, 1)),
    ((7, 8, 9),       (3, 4, 3),    (1, 2, 2, 1)),
    ((6, 7, 8, 9),    (2, 3, 3, 2), (1, 2, 3, 2, 1)),
]
STACK_SHAPES = [(), (2,)]


def _pair(structure, stack_shape):
    np.random.seed(0)
    shape, tr, ttr = structure
    A = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=stack_shape)
    B = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=stack_shape)
    return A, B


def _oracle(A, B):
    return np.asarray(A.to_dense()) * np.asarray(B.to_dense())


def _relerr(approx, exact):
    return np.linalg.norm(np.asarray(approx) - exact) / np.linalg.norm(exact)


class TestT3M(unittest.TestCase):
    # ---- reusable per-method checks (each phase wires up its method) ----
    def check_exact(self, method):
        # No truncation == the dense elementwise product, across structures and stack shapes.
        for structure in STRUCTURES:
            for C in STACK_SHAPES:
                with self.subTest(method=method, structure=structure, stack=C):
                    A, B = _pair(structure, C)
                    P = A.t3m(B, method=method)
                    self.assertEqual(A.shape, P.shape)
                    self.assertEqual(C, P.stack_shape)
                    self.assertLess(_relerr(P.to_dense(), _oracle(A, B)), 1e-12)

    def check_truncated(self, method):
        # Capped ranks, and truncation error no worse than ~2x the dense-t3svd reference.
        structure = ((7, 8, 9), (3, 4, 3), (1, 2, 2, 1))
        A, B = _pair(structure, ())
        oracle = _oracle(A, B)
        cap_t, cap_r = 2, 2
        P = A.t3m(B, method=method, max_tucker_ranks=cap_t, max_tt_ranks=cap_r)
        self.assertTrue(all(n <= cap_t for n in P.tucker_ranks), P.tucker_ranks)
        self.assertTrue(all(r <= cap_r for r in P.tt_ranks), P.tt_ranks)
        ref = t3.TuckerTensorTrain.t3svd_dense(oracle, max_tucker_ranks=cap_t, max_tt_ranks=cap_r)[0]
        self.assertLess(_relerr(P.to_dense(), oracle),
                        2.0 * _relerr(ref.to_dense(), oracle) + 1e-12)

    def check_sweep_exact(self, method):
        # Generous max-ranks (no real truncation) exercise the sweep path -> must reproduce the product.
        for structure in STRUCTURES:
            for C in STACK_SHAPES:
                with self.subTest(method=method, structure=structure, stack=C):
                    A, B = _pair(structure, C)
                    P = A.t3m(B, method=method, max_tucker_ranks=10000, max_tt_ranks=10000)
                    self.assertEqual(A.shape, P.shape)
                    self.assertEqual(C, P.stack_shape)
                    self.assertLess(_relerr(P.to_dense(), _oracle(A, B)), 1e-10)

    # ---- method (a) ----
    def test_form_then_round(self):
        self.check_exact('form_then_round')
        self.check_truncated('form_then_round')

    # ---- method (b) ----
    def test_inplace_fused(self):
        self.check_exact('inplace_fused')        # no-truncation short-circuit
        self.check_sweep_exact('inplace_fused')  # generous-rank sweep path (exercises the fused sweep)
        self.check_truncated('inplace_fused')

    def test_mul_routes_through_t3m(self):
        # `*` is the exact form_then_round path and works on stacked T3s.
        for C in STACK_SHAPES:
            A, B = _pair(((7, 8, 9), (3, 4, 3), (1, 2, 2, 1)), C)
            self.assertLess(_relerr((A * B).to_dense(), _oracle(A, B)), 1e-12)

    # ---- validation ----
    def test_validation(self):
        A, B = _pair(((7, 8, 9), (3, 4, 3), (1, 2, 2, 1)), ())
        with self.assertRaises(ValueError):  # shape mismatch
            A.t3m(t3.TuckerTensorTrain.randn((7, 8, 10), (3, 4, 3), (1, 2, 2, 1)))
        with self.assertRaises(ValueError):  # stack mismatch
            A.t3m(t3.TuckerTensorTrain.randn((7, 8, 9), (3, 4, 3), (1, 2, 2, 1), stack_shape=(2,)))
        with self.assertRaises(ValueError):  # unknown method
            A.t3m(B, method='bogus')
        with self.assertRaises(NotImplementedError):  # not-yet-implemented method
            A.t3m(B, method='swap')
        As, Bs = _pair(((7, 8, 9), (3, 4, 3), (1, 2, 2, 1)), (2,))
        with self.assertRaises(ValueError):  # rtol/atol + stacking
            As.t3m(Bs, method='form_then_round', rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
