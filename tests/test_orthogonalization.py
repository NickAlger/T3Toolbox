# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3tools.linalg
import t3tools.orthogonalization as orth
import t3tools.tucker_tensor_train as t3


np.random.seed(0)
numpy_tol = 1e-9
jax_tol = 1e-5
norm = np.linalg.norm
randn = np.random.randn

class Orthogonalization(unittest.TestCase):
    def test_up_svd_ith_basis_core(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (4, 3, 2, 6)
        x = t3.t3_corewise_randn((shape, tucker_ranks, tt_ranks))
        dense_x = t3.t3_to_dense(x)
        for ind in range(len(shape)):
            x2, ss = orth.up_svd_ith_basis_core(ind, x)
            dense_x2 = t3.t3_to_dense(x2)
            self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

            basis_cores2, tt_cores2 = x2
            B = basis_cores2[ind]
            G = tt_cores2[ind]
            rank = len(ss)
            self.assertEqual(B.shape[0], rank)
            self.assertEqual(G.shape[1], rank)

            I = np.eye(rank)
            self.assertLessEqual(norm(B @ B.T - I), numpy_tol * norm(I))

    def test_left_svd_ith_tt_core(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (4, 3, 2, 6)
        x = t3.t3_corewise_randn((shape, tucker_ranks, tt_ranks))
        dense_x = t3.t3_to_dense(x)
        for ind in range(len(shape)-1):
            x2, ss = orth.left_svd_ith_tt_core(ind, x)
            dense_x2 = t3.t3_to_dense(x2)
            self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

            basis_cores2, tt_cores2 = x2
            G = tt_cores2[ind]
            G_next = tt_cores2[ind+1]
            rank = len(ss)
            self.assertEqual(G.shape[2], rank)
            self.assertEqual(G_next.shape[0], rank)

            I = np.eye(rank)
            self.assertLessEqual(
                norm(np.einsum('iaj,iak->jk', G, G) - I),
                numpy_tol * norm(I)
            )

    def test_right_svd_ith_tt_core(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (4, 3, 2, 6)
        x = t3.t3_corewise_randn((shape, tucker_ranks, tt_ranks))
        dense_x = t3.t3_to_dense(x)
        for ind in range(len(shape)-1,0,-1):
            x2, ss = orth.right_svd_ith_tt_core(ind, x)
            dense_x2 = t3.t3_to_dense(x2)
            self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

            basis_cores2, tt_cores2 = x2
            G_prev = tt_cores2[ind-1]
            G = tt_cores2[ind]
            rank = len(ss)
            self.assertEqual(G.shape[0], rank)
            self.assertEqual(G_prev.shape[2], rank)

            I = np.eye(rank)
            self.assertLessEqual(
                norm(np.einsum('iaj,kaj->ik', G, G) - I),
                numpy_tol * norm(I)
            )

    def test_up_svd_ith_tt_core(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (4, 3, 2, 6)
        x = t3.t3_corewise_randn((shape, tucker_ranks, tt_ranks))
        dense_x = t3.t3_to_dense(x)
        for ind in range(len(shape) - 1, 0, -1):
            x2, ss = orth.up_svd_ith_tt_core(ind, x)
            dense_x2 = t3.t3_to_dense(x2)
            self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

            basis_cores2, tt_cores2 = x2
            B = basis_cores2[ind]
            G = tt_cores2[ind]
            rank = len(ss)
            self.assertEqual(G.shape[1], rank)
            self.assertEqual(B.shape[0], rank)

            # No orthogonality on this one

    def test_down_svd_ith_tt_core(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (4, 3, 2, 6)
        x = t3.t3_corewise_randn((shape, tucker_ranks, tt_ranks))
        dense_x = t3.t3_to_dense(x)
        for ind in range(len(shape) - 1, 0, -1):
            x2, ss = orth.down_svd_ith_tt_core(ind, x)
            dense_x2 = t3.t3_to_dense(x2)
            self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

            basis_cores2, tt_cores2 = x2
            B = basis_cores2[ind]
            G = tt_cores2[ind]
            rank = len(ss)
            self.assertEqual(G.shape[1], rank)
            self.assertEqual(B.shape[0], rank)

            I = np.eye(rank)
            self.assertLessEqual(
                norm(np.einsum('iaj,ibj->ab', G, G) - I),
                numpy_tol * norm(I)
            )

    def test_orthogonalize_relative_to_ith_basis_core(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (4, 3, 2, 6)
        x = t3.t3_corewise_randn((shape, tucker_ranks, tt_ranks))
        dense_x = t3.t3_to_dense(x)

        # ind=0
        x2 = orth.orthogonalize_relative_to_ith_basis_core(0, x)
        dense_x2 = t3.t3_to_dense(x2)
        self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

        ((B0, B1, B2), (G0, G1, G2)) = x2
        X = np.einsum('yj,zk,axb,byc,czd->axjkd', B1, B2, G0, G1, G2)
        I = np.eye(B0.shape[0])
        self.assertLessEqual(
            norm(np.einsum('axjkd,ayjkd->xy', X, X) - I),
            norm(numpy_tol * I)
        )

        # ind=1
        x2 = orth.orthogonalize_relative_to_ith_basis_core(1, x)
        dense_x2 = t3.t3_to_dense(x2)
        self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

        ((B0, B1, B2), (G0, G1, G2)) = x2
        X = np.einsum('xi,zk,axb,byc,czd->aiykd', B0, B2, G0, G1, G2)
        I = np.eye(B1.shape[0])
        self.assertLessEqual(
            norm(np.einsum('aiykd,aiwkd->yw', X, X) - I),
            norm(numpy_tol * I)
        )

        # ind=2
        x2 = orth.orthogonalize_relative_to_ith_basis_core(2, x)
        dense_x2 = t3.t3_to_dense(x2)
        self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

        ((B0, B1, B2), (G0, G1, G2)) = x2
        X = np.einsum('xi,yj,axb,byc,czd->aijzd', B0, B1, G0, G1, G2)
        I = np.eye(B2.shape[0])
        self.assertLessEqual(
            norm(np.einsum('aijzd,aijwd->zw', X, X) - I),
            norm(numpy_tol * I)
        )

    def test_orthogonalize_relative_to_ith_tt_core(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (4, 3, 2, 6)
        x = t3.t3_corewise_randn((shape, tucker_ranks, tt_ranks))
        dense_x = t3.t3_to_dense(x)

        # ind=0
        x2 = orth.orthogonalize_relative_to_ith_tt_core(0, x)
        dense_x2 = t3.t3_to_dense(x2)
        self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

        ((B0, B1, B2), (G0, G1, G2)) = x2
        X = np.einsum('yj,zk,byc,czd->bjkd', B1, B2, G1, G2)
        I = np.eye(G0.shape[2])
        self.assertLessEqual(
            norm(np.einsum('bjkd,cjkd->bc', X, X) - I),
            norm(numpy_tol * I)
        )

        I = np.eye(G0.shape[1])
        self.assertLessEqual(
            norm(np.einsum('ai,bi->ab', B0, B0) - I),
            norm(numpy_tol * I)
        )

        # ind=1
        x2 = orth.orthogonalize_relative_to_ith_tt_core(1, x)
        dense_x2 = t3.t3_to_dense(x2)
        self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

        ((B0, B1, B2), (G0, G1, G2)) = x2
        X = np.einsum('xi,axb->aib', B0, G0)
        I = np.eye(G1.shape[0])
        self.assertLessEqual(
            norm(np.einsum('aib,aic->bc', X, X) - I),
            norm(numpy_tol * I)
        )

        I = np.eye(G1.shape[1])
        self.assertLessEqual(
            norm(np.einsum('ai,bi->ab', B1, B1) - I),
            norm(numpy_tol * I)
        )

        X = np.einsum('zk,czd->ckd', B2, G2)
        I = np.eye(G1.shape[2])
        self.assertLessEqual(
            norm(np.einsum('ckd,bkd->cb', X, X) - I),
            norm(numpy_tol * I)
        )

        # ind=2
        x2 = orth.orthogonalize_relative_to_ith_tt_core(2, x)
        dense_x2 = t3.t3_to_dense(x2)
        self.assertLessEqual(norm(dense_x - dense_x2), numpy_tol * norm(dense_x))

        ((B0, B1, B2), (G0, G1, G2)) = x2
        X = np.einsum('xi,yj,axb,byc->aijc', B0, B1, G0, G1)
        I = np.eye(G2.shape[0])
        self.assertLessEqual(
            norm(np.einsum('aijc,aijd->cd', X, X) - I),
            norm(numpy_tol * I)
        )

        I = np.eye(G2.shape[1])
        self.assertLessEqual(
            norm(np.einsum('ai,bi->ab', B2, B2) - I),
            norm(numpy_tol * I)
        )

    def test_orthogonal_representations(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)))
        base, variation = orth.orthogonal_representations(x)  # Compute orthogonal representations
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

