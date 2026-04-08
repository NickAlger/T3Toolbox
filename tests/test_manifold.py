import numpy as np
import unittest
import t3tools.tucker_tensor_train as t3
import t3tools.manifold as t3m

np.random.seed(0)
numpy_tol = 1e-9
jax_tol = 1e-5
norm = np.linalg.norm
randn = np.random.randn

class TestTuckerTensorTrain(unittest.TestCase):
    def test_hole_shapes(self):
        basis_cores = (np.ones((10, 14)), np.ones((11, 15)), np.ones((12, 16)))
        left_tt_cores = (np.ones((5, 10, 2)), np.ones((2, 11, 3)), np.ones((3, 12, 2)))
        right_tt_cores = (np.ones((3, 10, 4)), np.ones((4, 11, 5)), np.ones((5, 12, 4)))
        outer_tt_cores = (np.ones((5, 9, 4)), np.ones((2, 8, 5)), np.ones((3, 7, 4)))
        base = (basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
        (var_basis_shapes, var_tt_shapes) = t3m.hole_shapes(base)
        self.assertEqual(var_basis_shapes, ((9, 14), (8, 15), (7, 16)))
        self.assertEqual(var_tt_shapes, ((5, 10, 4), (2, 11, 5), (3, 12, 4)))

    def test_ith_bv_to_t3(self):
        (U0, U1, U2) = (randn(10, 14), randn(11, 15), randn(12, 16))
        (L0, L1, L2) = (randn(5, 10, 2), randn(2, 11, 3), randn(3, 12, 2))
        (R0, R1, R2) = (randn(3, 10, 4), randn(4, 11, 5), randn(5, 12, 4))
        (O0, O1, O2) = (randn(5, 9, 4), randn(2, 8, 5), randn(3, 7, 4))
        base = ((U0, U1, U2), (L0, L1, L2), (R0, R1, R2), (O0, O1, O2))
        (V0, V1, V2) = (randn(9, 14), randn(8, 15), randn(7, 16))
        (H0, H1, H2) = (randn(1, 10, 4), randn(2, 11, 5), randn(3, 12, 1))
        variation = ((V0, V1, V2), (H0, H1, H2))

        # TT replacements

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(0, True, base, variation)
        self.assertEqual(((U0, U1, U2), (H0, R1, R2)), ((B0, B1, B2), (G0, G1, G2)))

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(1, True, base, variation)
        self.assertEqual(((U0, U1, U2), (L0, H1, R2)), ((B0, B1, B2), (G0, G1, G2)))

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(2, True, base, variation)
        self.assertEqual(((U0, U1, U2), (L0, L1, H2)), ((B0, B1, B2), (G0, G1, G2)))

        # Basis replacements

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(0, False, base, variation)
        self.assertEqual(((V0, U1, U2), (O0, R1, R2)), ((B0, B1, B2), (G0, G1, G2)))

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(1, False, base, variation)
        self.assertEqual(((U0, V1, U2), (L0, O1, R2)), ((B0, B1, B2), (G0, G1, G2)))

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(2, False, base, variation)
        self.assertEqual(((U0, U1, V2), (L0, L1, O2)), ((B0, B1, B2), (G0, G1, G2)))


    def test_orthogonal_representations(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)))
        base, variation = t3m.orthogonal_representations(x)  # Compute orthogonal representations
        basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
        basis_vars, tt_vars = variation
        (U0, U1, U2) = basis_cores
        (L0, L1, L2) = left_tt_cores
        (R0, R1, R2) = right_tt_cores
        (O0, O1, O2) = outer_tt_cores
        (V0, V1, V2) = basis_vars
        (H0, H1, H2) = tt_vars

        # TT replacement

        x2 = ((U0, U1, U2), (H0, R1, R2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        x2 = ((U0, U1, U2), (L0, H1, R2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        x2 = ((U0, U1, U2), (L0, L1, H2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        # basis replacement

        x2 = ((V0, U1, U2), (O0, R1, R2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        x2 = ((U0, V1, U2), (L0, O1, R2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        x2 = ((U0, U1, V2), (L0, L1, O2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        # Basis orthogonality

        for U in [U0, U1, U2]:
            self.assertLessEqual(
                norm(U @ U.T - np.eye(U.shape[0])),
                numpy_tol * norm(np.eye(U.shape[0]))
            )

        # left orthogonality

        for L in [L0, L1]: # Last core need not be left orthogonal
            self.assertLessEqual(
                norm(np.einsum('iaj,iak->jk', L, L) - np.eye(L.shape[2])),
                numpy_tol * norm(np.eye(L.shape[2]))
            )

        # right orthogonality

        for R in [R1, R2]: # First core need not be right orthogonal
            self.assertLessEqual(
                norm(np.einsum('iaj,kaj->ik', R, R) - np.eye(R.shape[0])),
                numpy_tol * norm(np.eye(R.shape[0]))
            )

        # outer orthogonality

        for O in [O0, O1, O2]:
            self.assertLessEqual(
                norm(np.einsum('iaj,ibj->ab', O, O) - np.eye(O.shape[1])),
                numpy_tol * norm(np.eye(O.shape[1]))
            )



if __name__ == '__main__':
    unittest.main()
