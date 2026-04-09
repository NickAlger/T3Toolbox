# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3tools.tucker_tensor_train as t3
import t3tools.t3svd as t3svd


np.random.seed(0)
numpy_tol = 1e-9
jax_tol = 1e-5
norm = np.linalg.norm
randn = np.random.randn

class TestT3SVD(unittest.TestCase):
    def test_t3_svd1(self):
        x = t3.t3_corewise_randn(((12, 11, 10), (14, 5, 13), (1, 17, 14, 1)))  # r0=rd=1
        x2, ss_basis, ss_tt = t3svd.t3_svd(x)  # Compute T3-SVD
        x_dense = t3.t3_to_dense(x)
        x2_dense = t3.t3_to_dense(x2)
        self.assertLessEqual(
            norm(x_dense - x2_dense), numpy_tol * norm(x2_dense)
        )
        self.assertTrue(t3.are_t3_ranks_minimal(x2))

    def test_t3_svd2(self):
        x = t3.t3_corewise_randn(((12, 11, 10), (14, 5, 13), (2, 17, 14, 3)))
        x2, ss_basis, ss_tt = t3svd.t3_svd(x)  # Compute T3-SVD
        x_dense = t3.t3_to_dense(x)
        x2_dense = t3.t3_to_dense(x2)
        self.assertLessEqual(
            norm(x_dense - x2_dense), numpy_tol * norm(x2_dense)
        )
        self.assertTrue(t3.are_t3_ranks_minimal(x2))

    def test_t3_svd3(self):
        x = t3.t3_corewise_randn(((12, 11, 10), (14, 5, 13), (2, 17, 14, 3)))
        x2, ss_basis, ss_tt = t3svd.t3_svd(x)  # Compute T3-SVD
        self.assertLessEqual(
            norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x))
        )

        (N0, N1, N2), (n0, n1, n2), (r0, r1, r2, r3) = t3.get_structure(x2)

        x2_dense = t3.t3_to_dense(x2, contract_ones=False)

        ss_tt0 = np.linalg.svd(x2_dense.reshape((r0, N0 * N1 * N2 * r3)))[1]
        ss_tt1 = np.linalg.svd(x2_dense.reshape((r0 * N0, N1 * N2 * r3)))[1]
        ss_tt2 = np.linalg.svd(x2_dense.reshape((r0 * N0 * N1, N2 * r3)))[1]
        ss_tt3 = np.linalg.svd(x2_dense.reshape((r0 * N0 * N1 * N2, r3)))[1]

        ss_basis0 = np.linalg.svd(x2_dense.swapaxes(0, 1).reshape((N0, -1)))[1]
        ss_basis1 = np.linalg.svd(x2_dense.swapaxes(0, 2).reshape((N1, -1)))[1]
        ss_basis2 = np.linalg.svd(x2_dense.swapaxes(0, 3).reshape((N2, -1)))[1]

        ss_tt0_a, ss_tt0_b = ss_tt0[:r0], ss_tt0[r0:]
        ss_tt1_a, ss_tt1_b = ss_tt1[:r1], ss_tt1[r1:]
        ss_tt2_a, ss_tt2_b = ss_tt2[:r2], ss_tt2[r2:]
        ss_tt3_a, ss_tt3_b = ss_tt3[:r3], ss_tt3[r3:]

        ss_basis0_a, ss_basis0_b = ss_basis0[:n0], ss_basis0[n0:]
        ss_basis1_a, ss_basis1_b = ss_basis1[:n1], ss_basis1[n1:]
        ss_basis2_a, ss_basis2_b = ss_basis2[:n2], ss_basis2[n2:]

        self.assertLessEqual(norm(ss_tt0_a - ss_tt[0]), numpy_tol * norm(ss_tt0_a))
        self.assertLessEqual(norm(ss_tt1_a - ss_tt[1]), numpy_tol * norm(ss_tt1_a))
        self.assertLessEqual(norm(ss_tt2_a - ss_tt[2]), numpy_tol * norm(ss_tt2_a))
        self.assertLessEqual(norm(ss_tt3_a - ss_tt[3]), numpy_tol * norm(ss_tt3_a))

        self.assertLessEqual(norm(ss_basis0_a - ss_basis[0]), numpy_tol * norm(ss_basis0_a))
        self.assertLessEqual(norm(ss_basis1_a - ss_basis[1]), numpy_tol * norm(ss_basis1_a))
        self.assertLessEqual(norm(ss_basis2_a - ss_basis[2]), numpy_tol * norm(ss_basis2_a))

        self.assertLess(norm(ss_tt0_b), numpy_tol * norm(ss_tt0))
        self.assertLess(norm(ss_tt1_b), numpy_tol * norm(ss_tt1))
        self.assertLess(norm(ss_tt2_b), numpy_tol * norm(ss_tt2))
        self.assertLess(norm(ss_tt3_b), numpy_tol * norm(ss_tt3))

        self.assertLess(norm(ss_basis0_b), numpy_tol * norm(ss_basis0))
        self.assertLess(norm(ss_basis1_b), numpy_tol * norm(ss_basis1))
        self.assertLess(norm(ss_basis2_b), numpy_tol * norm(ss_basis2))

    def test_t3_svd_dense(self):
        N0, N1, N2 = 10, 11, 9
        x_dense = np.random.randn(N0, N1, N2)
        x2, ss_basis, ss_tt = t3svd.t3_svd_dense(x_dense)
        x2_dense = t3.t3_to_dense(x2)
        self.assertLessEqual(norm(x_dense - x2_dense), numpy_tol * norm(x_dense))

        ss_tt0 = np.linalg.svd(x_dense.reshape((1, N0 * N1 * N2 * 1)))[1]
        ss_tt1 = np.linalg.svd(x_dense.reshape((1 * N0, N1 * N2 * 1)))[1]
        ss_tt2 = np.linalg.svd(x_dense.reshape((1 * N0 * N1, N2 * 1)))[1]
        ss_tt3 = np.linalg.svd(x_dense.reshape((1 * N0 * N1 * N2, 1)))[1]

        ss_basis0 = np.linalg.svd(x_dense.swapaxes(0, 0).reshape((N0, -1)))[1]
        ss_basis1 = np.linalg.svd(x_dense.swapaxes(0, 1).reshape((N1, -1)))[1]
        ss_basis2 = np.linalg.svd(x_dense.swapaxes(0, 2).reshape((N2, -1)))[1]

        _, (n0, n1, n2), (r0, r1, r2, r3) = t3.get_structure(x2)

        ss_tt0_a, ss_tt0_b = ss_tt0[:r0], ss_tt0[r0:]
        ss_tt1_a, ss_tt1_b = ss_tt1[:r1], ss_tt1[r1:]
        ss_tt2_a, ss_tt2_b = ss_tt2[:r2], ss_tt2[r2:]
        ss_tt3_a, ss_tt3_b = ss_tt3[:r3], ss_tt3[r3:]

        ss_basis0_a, ss_basis0_b = ss_basis0[:n0], ss_basis0[n0:]
        ss_basis1_a, ss_basis1_b = ss_basis1[:n1], ss_basis1[n1:]
        ss_basis2_a, ss_basis2_b = ss_basis2[:n2], ss_basis2[n2:]

        self.assertLessEqual(norm(ss_tt0_a - ss_tt[0]), numpy_tol * norm(ss_tt0_a))
        self.assertLessEqual(norm(ss_tt1_a - ss_tt[1]), numpy_tol * norm(ss_tt1_a))
        self.assertLessEqual(norm(ss_tt2_a - ss_tt[2]), numpy_tol * norm(ss_tt2_a))
        self.assertLessEqual(norm(ss_tt3_a - ss_tt[3]), numpy_tol * norm(ss_tt3_a))

        self.assertLessEqual(norm(ss_basis0_a - ss_basis[0]), numpy_tol * norm(ss_basis0_a))
        self.assertLessEqual(norm(ss_basis1_a - ss_basis[1]), numpy_tol * norm(ss_basis1_a))
        self.assertLessEqual(norm(ss_basis2_a - ss_basis[2]), numpy_tol * norm(ss_basis2_a))

        self.assertLess(norm(ss_tt0_b), numpy_tol * norm(ss_tt0))
        self.assertLess(norm(ss_tt1_b), numpy_tol * norm(ss_tt1))
        self.assertLess(norm(ss_tt2_b), numpy_tol * norm(ss_tt2))
        self.assertLess(norm(ss_tt3_b), numpy_tol * norm(ss_tt3))

        self.assertLess(norm(ss_basis0_b), numpy_tol * norm(ss_basis0))
        self.assertLess(norm(ss_basis1_b), numpy_tol * norm(ss_basis1))
        self.assertLess(norm(ss_basis2_b), numpy_tol * norm(ss_basis2))


if __name__ == '__main__':
    unittest.main()

