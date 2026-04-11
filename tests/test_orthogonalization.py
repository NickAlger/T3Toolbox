# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3toolbox.orthogonalization as orth
import t3toolbox.tucker_tensor_train as t3

try:
    import t3toolbox.jax.orthogonalization as orth_jax
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    orth_jax = orth

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class Orthogonalization(unittest.TestCase):
    def check_relerr(self, xtrue, x):
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def test_up_svd_ith_tucker_core(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 6)),
        ]

        for STRUCTURE in structures:
            for ORTH in [orth, orth_jax]:
                with self.subTest(ORTH=ORTH, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    dense_x = t3.t3_to_dense(x)
                    for ind in range(len(STRUCTURE[0])):

                        x2, ss = ORTH.up_svd_ith_tucker_core(ind, x)

                        dense_x2 = t3.t3_to_dense(x2)
                        self.check_relerr(dense_x, dense_x2)

                        tucker_cores2, tt_cores2 = x2
                        B = tucker_cores2[ind]
                        G = tt_cores2[ind]
                        rank = len(ss)
                        self.assertEqual(B.shape[0], rank)
                        self.assertEqual(G.shape[1], rank)

                        I = np.eye(rank)
                        self.check_relerr(I, B @ B.T)

    def test_left_svd_ith_tt_core(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 6)),
        ]

        for STRUCTURE in structures:
            for ORTH in [orth, orth_jax]:
                with self.subTest(ORTH=ORTH, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    dense_x = t3.t3_to_dense(x)
                    for ind in range(len(STRUCTURE[0])-1):

                        x2, ss = ORTH.left_svd_ith_tt_core(ind, x)

                        dense_x2 = t3.t3_to_dense(x2)
                        self.check_relerr(dense_x, dense_x2)

                        tucker_cores2, tt_cores2 = x2
                        G = tt_cores2[ind]
                        G_next = tt_cores2[ind+1]
                        rank = len(ss)
                        self.assertEqual(G.shape[2], rank)
                        self.assertEqual(G_next.shape[0], rank)

                        I = np.eye(rank)
                        self.check_relerr(I, np.einsum('iaj,iak->jk', G, G))

    def test_right_svd_ith_tt_core(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 6)),
        ]

        for STRUCTURE in structures:
            for ORTH in [orth, orth_jax]:
                with self.subTest(ORTH=ORTH, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    dense_x = t3.t3_to_dense(x)
                    for ind in range(len(STRUCTURE)-1,0,-1):

                        x2, ss = ORTH.right_svd_ith_tt_core(ind, x)

                        dense_x2 = t3.t3_to_dense(x2)
                        self.check_relerr(dense_x, dense_x2)

                        tucker_cores2, tt_cores2 = x2
                        G_prev = tt_cores2[ind-1]
                        G = tt_cores2[ind]
                        rank = len(ss)
                        self.assertEqual(G.shape[0], rank)
                        self.assertEqual(G_prev.shape[2], rank)

                        I = np.eye(rank)
                        self.check_relerr(I, np.einsum('iaj,kaj->ik', G, G))

    def test_up_svd_ith_tt_core(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 6)),
        ]

        for STRUCTURE in structures:
            for ORTH in [orth, orth_jax]:
                with self.subTest(ORTH=ORTH, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    dense_x = t3.t3_to_dense(x)
                    for ind in range(len(STRUCTURE[0])):

                        x2, ss = ORTH.up_svd_ith_tt_core(ind, x)

                        dense_x2 = t3.t3_to_dense(x2)
                        self.check_relerr(dense_x, dense_x2)

                        tucker_cores2, tt_cores2 = x2
                        B = tucker_cores2[ind]
                        G = tt_cores2[ind]
                        rank = len(ss)
                        self.assertEqual(G.shape[1], rank)
                        self.assertEqual(B.shape[0], rank)

                        # No orthogonality on this one

    def test_down_svd_ith_tt_core(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 6)),
        ]

        for STRUCTURE in structures:
            for ORTH in [orth, orth_jax]:
                with self.subTest(ORTH=ORTH, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    dense_x = t3.t3_to_dense(x)
                    for ind in range(len(STRUCTURE[0])):

                        x2, ss = ORTH.down_svd_ith_tt_core(ind, x)

                        dense_x2 = t3.t3_to_dense(x2)
                        self.check_relerr(dense_x, dense_x2)

                        tucker_cores2, tt_cores2 = x2
                        B = tucker_cores2[ind]
                        G = tt_cores2[ind]
                        rank = len(ss)
                        self.assertEqual(G.shape[1], rank)
                        self.assertEqual(B.shape[0], rank)

                        I = np.eye(rank)
                        self.check_relerr(I, np.einsum('iaj,ibj->ab', G, G))

    def test_orthogonalize_relative_to_ith_tucker_core(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 6)),
        ]

        for STRUCTURE in structures:
            for ORTH in [orth, orth_jax]:
                with self.subTest(ORTH=ORTH, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    dense_x = t3.t3_to_dense(x)

                    # ind=0
                    x2 = ORTH.orthogonalize_relative_to_ith_tucker_core(0, x)

                    dense_x2 = t3.t3_to_dense(x2)
                    self.check_relerr(dense_x, dense_x2)

                    ((B0, B1, B2), (G0, G1, G2)) = x2
                    X = np.einsum('yj,zk,axb,byc,czd->axjkd', B1, B2, G0, G1, G2)
                    I = np.eye(B0.shape[0])
                    self.check_relerr(I, np.einsum('axjkd,ayjkd->xy', X, X))

                    # ind=1
                    x2 = ORTH.orthogonalize_relative_to_ith_tucker_core(1, x)

                    dense_x2 = t3.t3_to_dense(x2)
                    self.check_relerr(dense_x, dense_x2)

                    ((B0, B1, B2), (G0, G1, G2)) = x2
                    X = np.einsum('xi,zk,axb,byc,czd->aiykd', B0, B2, G0, G1, G2)
                    I = np.eye(B1.shape[0])
                    self.check_relerr(I, np.einsum('aiykd,aiwkd->yw', X, X))

                    # ind=2
                    x2 = ORTH.orthogonalize_relative_to_ith_tucker_core(2, x)

                    dense_x2 = t3.t3_to_dense(x2)
                    self.check_relerr(dense_x, dense_x2)

                    ((B0, B1, B2), (G0, G1, G2)) = x2
                    X = np.einsum('xi,yj,axb,byc,czd->aijzd', B0, B1, G0, G1, G2)
                    I = np.eye(B2.shape[0])
                    self.check_relerr(I,np.einsum('aijzd,aijwd->zw', X, X))


    def test_orthogonalize_relative_to_ith_tt_core(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 6)),
        ]

        for STRUCTURE in structures:
            for ORTH in [orth, orth_jax]:
                with self.subTest(ORTH=ORTH, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    dense_x = t3.t3_to_dense(x)

                    # ind=0
                    x2 = ORTH.orthogonalize_relative_to_ith_tt_core(0, x)

                    dense_x2 = t3.t3_to_dense(x2)
                    self.check_relerr(dense_x, dense_x2)

                    ((B0, B1, B2), (G0, G1, G2)) = x2
                    X = np.einsum('yj,zk,byc,czd->bjkd', B1, B2, G1, G2)
                    I = np.eye(G0.shape[2])
                    self.check_relerr(I, np.einsum('bjkd,cjkd->bc', X, X))

                    I = np.eye(G0.shape[1])
                    self.check_relerr(I, np.einsum('ai,bi->ab', B0, B0))

                    # ind=1
                    x2 = ORTH.orthogonalize_relative_to_ith_tt_core(1, x)

                    dense_x2 = t3.t3_to_dense(x2)
                    self.check_relerr(dense_x, dense_x2)

                    ((B0, B1, B2), (G0, G1, G2)) = x2
                    X = np.einsum('xi,axb->aib', B0, G0)
                    I = np.eye(G1.shape[0])
                    self.check_relerr(I, np.einsum('aib,aic->bc', X, X))

                    I = np.eye(G1.shape[1])
                    self.check_relerr(I, np.einsum('ai,bi->ab', B1, B1))

                    X = np.einsum('zk,czd->ckd', B2, G2)
                    I = np.eye(G1.shape[2])
                    self.check_relerr(I, np.einsum('ckd,bkd->cb', X, X))

                    # ind=2
                    x2 = ORTH.orthogonalize_relative_to_ith_tt_core(2, x)

                    dense_x2 = t3.t3_to_dense(x2)
                    self.check_relerr(dense_x, dense_x2)

                    ((B0, B1, B2), (G0, G1, G2)) = x2
                    X = np.einsum('xi,yj,axb,byc->aijc', B0, B1, G0, G1)
                    I = np.eye(G2.shape[0])
                    self.check_relerr(I, np.einsum('aijc,aijd->cd', X, X))

                    I = np.eye(G2.shape[1])
                    self.check_relerr(I, np.einsum('ai,bi->ab', B2, B2))

    def test_orthogonal_representations(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)),
        ]

        for STRUCTURE in structures:
            for ORTH in [orth, orth_jax]:
                with self.subTest(ORTH=ORTH, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)

                    base, variation = ORTH.orthogonal_representations(x)  # Compute orthogonal representations

                    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
                    tucker_vars, tt_vars = variation
                    (U0, U1, U2) = tucker_cores
                    (L0, L1, L2) = left_tt_cores
                    (R0, R1, R2) = right_tt_cores
                    (O0, O1, O2) = outer_tt_cores
                    (V0, V1, V2) = tucker_vars
                    (H0, H1, H2) = tt_vars

                    # TT replacement

                    x2 = ((U0, U1, U2), (H0, R1, R2))
                    self.check_relerr(t3.t3_to_dense(x), t3.t3_to_dense(x2))

                    x2 = ((U0, U1, U2), (L0, H1, R2))
                    self.check_relerr(t3.t3_to_dense(x), t3.t3_to_dense(x2))

                    x2 = ((U0, U1, U2), (L0, L1, H2))
                    self.check_relerr(t3.t3_to_dense(x), t3.t3_to_dense(x2))

                    # tucker replacement

                    x2 = ((V0, U1, U2), (O0, R1, R2))
                    self.check_relerr(t3.t3_to_dense(x), t3.t3_to_dense(x2))

                    x2 = ((U0, V1, U2), (L0, O1, R2))
                    self.check_relerr(t3.t3_to_dense(x), t3.t3_to_dense(x2))

                    x2 = ((U0, U1, V2), (L0, L1, O2))
                    self.check_relerr(t3.t3_to_dense(x), t3.t3_to_dense(x2))

                    # Basis orthogonality
                    for U in [U0, U1, U2]:
                        self.check_relerr(np.eye(U.shape[0]), U @ U.T)

                    # left orthogonality
                    for L in [L0, L1]: # Last core need not be left orthogonal
                        self.check_relerr(np.eye(L.shape[2]), np.einsum('iaj,iak->jk', L, L))

                    # right orthogonality
                    for R in [R1, R2]: # First core need not be right orthogonal
                        self.check_relerr(np.eye(R.shape[0]), np.einsum('iaj,kaj->ik', R, R))

                    # outer orthogonality
                    for O in [O0, O1, O2]:
                        self.check_relerr(np.eye(O.shape[1]), np.einsum('iaj,ibj->ab', O, O))


if __name__ == '__main__':
    unittest.main()

