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

    # ---- method (c) ----
    def test_swap(self):
        self.check_exact('swap')         # no-truncation short-circuit
        self.check_sweep_exact('swap')   # generous-rank sweep path (exercises swaps + contracts)
        self.check_truncated('swap')

    def test_swap_oversample(self):
        # Oversampling improves quality toward optimal and preserves the exact path.
        structure = ((7, 8, 9), (3, 4, 3), (1, 2, 2, 1))
        A, B = _pair(structure, ())
        oracle = _oracle(A, B)
        kw = dict(method='swap', max_tucker_ranks=2, max_tt_ranks=2)
        err1 = _relerr(A.t3m(B, oversample=1, **kw).to_dense(), oracle)
        err3 = _relerr(A.t3m(B, oversample=3, **kw).to_dense(), oracle)
        self.assertLess(err3, err1)                                    # oversampling helps
        ref = t3.TuckerTensorTrain.t3svd_dense(oracle, max_tucker_ranks=2, max_tt_ranks=2)[0]
        self.assertLess(err3, 1.1 * _relerr(ref.to_dense(), oracle) + 1e-12)  # ~optimal
        # oversample preserves the exact (generous-rank) path
        P = A.t3m(B, method='swap', max_tucker_ranks=10000, max_tt_ranks=10000, oversample=2)
        self.assertLess(_relerr(P.to_dense(), oracle), 1e-10)

    def test_swap_per_position_tt(self):
        # A non-uniform max_tt_ranks sequence is honored bond-by-bond (via the t3svd cleanup).
        structure = ((6, 7, 8, 9), (2, 3, 3, 2), (1, 2, 3, 2, 1))
        A, B = _pair(structure, ())
        oracle = _oracle(A, B)
        seq = (1, 2, 3, 2, 1)
        P = A.t3m(B, method='swap', max_tt_ranks=seq, oversample=2)
        self.assertTrue(all(r <= s for r, s in zip(P.tt_ranks, seq)), (P.tt_ranks, seq))
        ref = t3.TuckerTensorTrain.t3svd_dense(oracle, max_tt_ranks=seq)[0]
        self.assertLess(_relerr(P.to_dense(), oracle), 1.5 * _relerr(ref.to_dense(), oracle) + 1e-12)

    def test_swap_stacked_oversample(self):
        # Stacked + max-rank + oversample: the t3svd cleanup must be stacking-safe; quality ~ method (a).
        for structure in STRUCTURES:
            with self.subTest(structure=structure):
                A, B = _pair(structure, (2,))
                oracle = _oracle(A, B)
                Pa = A.t3m(B, method='form_then_round', max_tucker_ranks=2, max_tt_ranks=2)
                Pc = A.t3m(B, method='swap', max_tucker_ranks=2, max_tt_ranks=2, oversample=2)
                self.assertEqual((2,), Pc.stack_shape)
                self.assertTrue(all(n <= 2 for n in Pc.tucker_ranks), Pc.tucker_ranks)
                self.assertTrue(all(r <= 2 for r in Pc.tt_ranks), Pc.tt_ranks)
                self.assertLess(_relerr(Pc.to_dense(), oracle),
                                2.0 * _relerr(Pa.to_dense(), oracle) + 1e-12)

    def test_swap_joint_quality(self):
        # Decision 1: for the same rtol, swap (oversample=2) keeps ranks <= form_then_round (+ slack).
        for structure in (((7, 8, 9), (3, 4, 3), (1, 2, 2, 1)),
                          ((6, 7, 8, 9), (2, 3, 3, 2), (1, 2, 3, 2, 1))):
            with self.subTest(structure=structure):
                A, B = _pair(structure, ())
                a = A.t3m(B, method='form_then_round', rtol=1e-2)
                c = A.t3m(B, method='swap', rtol=1e-2, oversample=2)
                a_sz = sum(a.tucker_ranks) + sum(a.tt_ranks)
                c_sz = sum(c.tucker_ranks) + sum(c.tt_ranks)
                self.assertLessEqual(c_sz, a_sz + 2,
                                     (a.tucker_ranks, a.tt_ranks, c.tucker_ranks, c.tt_ranks))

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
        with self.assertRaises(ValueError):  # oversample < 1
            A.t3m(B, method='swap', max_tucker_ranks=2, oversample=0.5)
        with self.assertRaises(ValueError):  # oversample with a non-swap method
            A.t3m(B, method='inplace_fused', max_tucker_ranks=2, oversample=2)
        As, Bs = _pair(((7, 8, 9), (3, 4, 3), (1, 2, 2, 1)), (2,))
        with self.assertRaises(ValueError):  # rtol/atol + stacking
            As.t3m(Bs, method='form_then_round', rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
