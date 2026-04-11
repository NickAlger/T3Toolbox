# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.t3svd as t3svd

try:
    import t3toolbox.jax.t3svd as t3svd_jax
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    t3svd_jax = t3svd

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestT3SVD(unittest.TestCase):
    def check_relerr(self, xtrue, x):
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def test_t3_svd1(self):
        structures = [
            ((12, 11, 10), (14, 5, 13), (1, 17, 14, 1)), # tail ranks 1 (r0=rd=1)
            ((12, 11, 10), (14, 5, 13), (2, 17, 14, 3)), # non-1 tail ranks

        ]

        for STRUCTURE in structures:
            for T3SVD in [t3svd, t3svd_jax]:
                with self.subTest(T3SVD=T3SVD, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)

                    x2, ss_tucker, ss_tt = T3SVD.t3_svd(x)  # Compute T3-SVD

                    x_dense = t3.t3_to_dense(x)
                    x2_dense = t3.t3_to_dense(x2)
                    self.check_relerr(x_dense, x2_dense)
                    self.assertTrue(t3.are_t3_ranks_minimal(x2))

    def test_t3_svd2(self):
        structures = [
            ((12, 11, 10), (14, 5, 13), (1, 17, 14, 1)), # Mathematically, only supposed to work for tail ranks 1
        ]

        for STRUCTURE in structures:
            for T3SVD in [t3svd, t3svd_jax]:
                with self.subTest(T3SVD=T3SVD, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)

                    x2, ss_tucker, ss_tt = T3SVD.t3_svd(x)  # Compute T3-SVD

                    self.assertLessEqual(
                        norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), tol * norm(t3.t3_to_dense(x))
                    )

                    (N0, N1, N2), (n0, n1, n2), (r0, r1, r2, r3) = t3.get_structure(x2)

                    x2_dense = t3.t3_to_dense(x2, contract_ones=False)

                    ss_tt0 = np.linalg.svd(x2_dense.reshape((r0, N0 * N1 * N2 * r3)))[1]
                    ss_tt1 = np.linalg.svd(x2_dense.reshape((r0 * N0, N1 * N2 * r3)))[1]
                    ss_tt2 = np.linalg.svd(x2_dense.reshape((r0 * N0 * N1, N2 * r3)))[1]
                    ss_tt3 = np.linalg.svd(x2_dense.reshape((r0 * N0 * N1 * N2, r3)))[1]

                    ss_tucker0 = np.linalg.svd(x2_dense.swapaxes(0, 1).reshape((N0, -1)))[1]
                    ss_tucker1 = np.linalg.svd(x2_dense.swapaxes(0, 2).reshape((N1, -1)))[1]
                    ss_tucker2 = np.linalg.svd(x2_dense.swapaxes(0, 3).reshape((N2, -1)))[1]

                    ss_tt0_a, ss_tt0_b = ss_tt0[:r0], ss_tt0[r0:]
                    ss_tt1_a, ss_tt1_b = ss_tt1[:r1], ss_tt1[r1:]
                    ss_tt2_a, ss_tt2_b = ss_tt2[:r2], ss_tt2[r2:]
                    ss_tt3_a, ss_tt3_b = ss_tt3[:r3], ss_tt3[r3:]

                    ss_tucker0_a, ss_tucker0_b = ss_tucker0[:n0], ss_tucker0[n0:]
                    ss_tucker1_a, ss_tucker1_b = ss_tucker1[:n1], ss_tucker1[n1:]
                    ss_tucker2_a, ss_tucker2_b = ss_tucker2[:n2], ss_tucker2[n2:]

                    self.check_relerr(ss_tt0_a, ss_tt[0])
                    self.check_relerr(ss_tt1_a, ss_tt[1])
                    self.check_relerr(ss_tt2_a, ss_tt[2])
                    self.check_relerr(ss_tt3_a, ss_tt[3])

                    self.check_relerr(ss_tucker0_a, ss_tucker[0])
                    self.check_relerr(ss_tucker1_a, ss_tucker[1])
                    self.check_relerr(ss_tucker2_a, ss_tucker[2])

                    self.assertLess(norm(ss_tt0_b), tol * norm(ss_tt0))
                    self.assertLess(norm(ss_tt1_b), tol * norm(ss_tt1))
                    self.assertLess(norm(ss_tt2_b), tol * norm(ss_tt2))
                    self.assertLess(norm(ss_tt3_b), tol * norm(ss_tt3))

                    self.assertLess(norm(ss_tucker0_b), tol * norm(ss_tucker0))
                    self.assertLess(norm(ss_tucker1_b), tol * norm(ss_tucker1))
                    self.assertLess(norm(ss_tucker2_b), tol * norm(ss_tucker2))

    def test_t3_svd_dense(self):
        shapes = [
            (10,11,12),
        ]

        for SHAPE in shapes:
            for T3SVD in [t3svd, t3svd_jax]:
                with self.subTest(T3SVD=T3SVD, SHAPE=SHAPE):
                    N0, N1, N2 = SHAPE
                    x_dense = np.random.randn(N0, N1, N2)

                    x2, ss_tucker, ss_tt = T3SVD.t3_svd_dense(x_dense)

                    x2_dense = t3.t3_to_dense(x2)
                    self.assertLessEqual(norm(x_dense - x2_dense), tol * norm(x_dense))

                    ss_tt0 = np.linalg.svd(x_dense.reshape((1, N0 * N1 * N2 * 1)))[1]
                    ss_tt1 = np.linalg.svd(x_dense.reshape((1 * N0, N1 * N2 * 1)))[1]
                    ss_tt2 = np.linalg.svd(x_dense.reshape((1 * N0 * N1, N2 * 1)))[1]
                    ss_tt3 = np.linalg.svd(x_dense.reshape((1 * N0 * N1 * N2, 1)))[1]

                    ss_tucker0 = np.linalg.svd(x_dense.swapaxes(0, 0).reshape((N0, -1)))[1]
                    ss_tucker1 = np.linalg.svd(x_dense.swapaxes(0, 1).reshape((N1, -1)))[1]
                    ss_tucker2 = np.linalg.svd(x_dense.swapaxes(0, 2).reshape((N2, -1)))[1]

                    _, (n0, n1, n2), (r0, r1, r2, r3) = t3.get_structure(x2)

                    ss_tt0_a, ss_tt0_b = ss_tt0[:r0], ss_tt0[r0:]
                    ss_tt1_a, ss_tt1_b = ss_tt1[:r1], ss_tt1[r1:]
                    ss_tt2_a, ss_tt2_b = ss_tt2[:r2], ss_tt2[r2:]
                    ss_tt3_a, ss_tt3_b = ss_tt3[:r3], ss_tt3[r3:]

                    ss_tucker0_a, ss_tucker0_b = ss_tucker0[:n0], ss_tucker0[n0:]
                    ss_tucker1_a, ss_tucker1_b = ss_tucker1[:n1], ss_tucker1[n1:]
                    ss_tucker2_a, ss_tucker2_b = ss_tucker2[:n2], ss_tucker2[n2:]

                    self.check_relerr(ss_tt0_a, ss_tt[0])
                    self.check_relerr(ss_tt1_a, ss_tt[1])
                    self.check_relerr(ss_tt2_a, ss_tt[2])
                    self.check_relerr(ss_tt3_a, ss_tt[3])

                    self.check_relerr(ss_tucker0_a, ss_tucker[0])
                    self.check_relerr(ss_tucker1_a, ss_tucker[1])
                    self.check_relerr(ss_tucker2_a, ss_tucker[2])

                    self.assertLess(norm(ss_tt0_b), tol * norm(ss_tt0))
                    self.assertLess(norm(ss_tt1_b), tol * norm(ss_tt1))
                    self.assertLess(norm(ss_tt2_b), tol * norm(ss_tt2))
                    self.assertLess(norm(ss_tt3_b), tol * norm(ss_tt3))

                    self.assertLess(norm(ss_tucker0_b), tol * norm(ss_tucker0))
                    self.assertLess(norm(ss_tucker1_b), tol * norm(ss_tucker1))
                    self.assertLess(norm(ss_tucker2_b), tol * norm(ss_tucker2))


if __name__ == '__main__':
    unittest.main()

