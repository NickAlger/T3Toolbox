# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest
import t3tools.tucker_tensor_train as t3

np.random.seed(0)
numpy_tol = 1e-9
jax_tol = 1e-5
norm = np.linalg.norm
randn = np.random.randn

class TestTuckerTensorTrain(unittest.TestCase):
    def test_structure1(self):
        basis_cores = (np.ones((4, 14)), np.ones((5, 15)), np.ones((6, 16)))
        tt_cores = (np.ones((2, 4, 3)), np.ones((3, 5, 7)), np.ones((7, 6, 5)))
        x = (basis_cores, tt_cores)
        shape, tucker_ranks, tt_ranks = t3.structure(x)
        self.assertEqual((14, 15, 16), shape)
        self.assertEqual((4, 5, 6), tucker_ranks)
        self.assertEqual((2,3,7,5), tt_ranks)

    def test_t3_to_dense1(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 7, 5)))  # make TuckerTensorTrain
        x_dense = t3.t3_to_dense(x)  # Convert TuckerTensorTrain to dense tensor
        ((B0, B1, B2), (G0, G1, G2)) = x
        x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->ijk', B0, B1, B2, G0, G1, G2)
        self.assertLessEqual(norm(x_dense - x_dense2), numpy_tol * norm(x_dense))

    def test_t3_to_dense2(self):
        # leading and trailing ones not contracted
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 4, 5)))  # make TuckerTensorTrain
        x_dense = t3.t3_to_dense(x, contract_ones=False)  # Convert TuckerTensorTrain to dense tensor
        self.assertEqual((2, 14, 15, 16, 5), x_dense.shape)
        ((B0, B1, B2), (G0, G1, G2)) = x
        x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->aijkd', B0, B1, B2, G0, G1, G2)
        self.assertLessEqual(norm(x_dense - x_dense2), numpy_tol * norm(x_dense))

    def test_t3_reverse1(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))  # Make TuckerTensorTrain
        self.assertEqual(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)), t3.structure(x))
        reversed_x = t3.t3_reverse(x)
        self.assertEqual(((16, 15, 14), (6, 5, 4), (1, 2, 3, 2)), t3.structure(reversed_x))
        x_dense = t3.t3_to_dense(x)
        reversed_x_dense = t3.t3_to_dense(reversed_x)
        x_dense2 = reversed_x_dense.transpose([2, 1, 0])
        self.assertLessEqual(norm(x_dense - x_dense2), numpy_tol * norm(x_dense))

    def test_t3_zeros1(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (2, 3, 2, 1)
        structure = (shape, tucker_ranks, tt_ranks)
        z = t3.t3_zeros(structure)
        self.assertEqual(structure, t3.structure(z))
        dense_z = t3.t3_to_dense(z)
        self.assertLessEqual(norm(dense_z), numpy_tol)

    def test_t3_corewise_randn1(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (1, 3, 2, 1)
        structure = (shape, tucker_ranks, tt_ranks)
        x = t3.t3_corewise_randn(structure)  # TuckerTensorTrain with random cores
        self.assertEqual(structure, t3.structure(x))

    def test_t3_save_and_t3_load1(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)))
        fname = 't3_test_file1945145272' # hopefully no one else has used this filename
        t3.t3_save(fname, x)  # Save to file 't3_file.npz'
        x2 = t3.t3_load(fname)  # Load from file
        basis_cores, tt_cores = x
        basis_cores2, tt_cores2 = x2
        for B, B2 in zip(basis_cores, basis_cores2):
            self.assertLessEqual(norm(B - B2), numpy_tol * norm(B))
        for G, G2 in zip(tt_cores, tt_cores2):
            self.assertLessEqual(norm(G - G2), numpy_tol * norm(G))

    def test_t3_apply1(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)))
        vecs = [randn(14), randn(15), randn(16)]
        result = t3.t3_apply(x, vecs)  # <-- contract x with vecs in all indices
        result2 = np.einsum('ijk,i,j,k', t3.t3_to_dense(x), vecs[0], vecs[1], vecs[2])
        self.assertLessEqual(np.abs(result - result2), numpy_tol * np.abs(result2))

    def test_t3_apply2(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)))
        vecs = [randn(3, 14), randn(3, 15), randn(3, 16)]
        result = t3.t3_apply(x, vecs)
        result2 = np.einsum('ijk,ni,nj,nk->n', t3.t3_to_dense(x), vecs[0], vecs[1], vecs[2])
        self.assertLessEqual(norm(result - result2), numpy_tol * norm(result2))

    def test_t3_entry1(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)))
        index = [9, 4, 7]  # get entry (9,4,7)
        result = t3.t3_entry(x, index)
        result2 = t3.t3_to_dense(x)[9, 4, 7]
        self.assertLessEqual(np.abs(result - result2), numpy_tol * np.abs(result2))

    def test_t3_entry2(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (6, 2, 4, 3)))
        index = [[9, 8], [4, 10], [7, 13]]  # get entries (9,4,7) and (8,10,13)
        entries = t3.t3_entry(x, index)
        x_dense = t3.t3_to_dense(x)
        entries2 = np.array([x_dense[9, 4, 7], x_dense[8, 10, 13]])
        self.assertLessEqual(norm(entries - entries2), numpy_tol * norm(entries2))

    def test_compute_minimal_ranks1(self):
        mr = t3.compute_minimal_ranks(((10, 11, 12, 13), (14, 15, 16, 17), (98, 99, 100, 101, 102)))
        mr_true = ((10, 11, 12, 13), (1, 10, 100, 13, 1))
        self.assertEqual(mr, mr_true)

    def test_are_ranks_minimal1(self):
        x = t3.t3_corewise_randn(((13, 14, 15, 16), (4, 5, 6, 7), (1, 4, 9, 7, 1)))
        self.assertTrue(t3.are_t3_ranks_minimal(x))

    def test_are_ranks_minimal2(self):
        x = t3.t3_corewise_randn(((13, 14, 15, 16), (4, 5, 6, 7), (1, 99, 9, 7, 1)))
        self.assertFalse(t3.are_t3_ranks_minimal(x))

    def test_are_ranks_minimal3(self):
        x = t3.t3_corewise_randn(((13, 14, 15, 16), (4, 5, 6, 7), (1, 4, 9, 7, 2)))
        self.assertFalse(t3.are_t3_ranks_minimal(x))

    def test_are_ranks_minimal4(self):
        x = t3.t3_corewise_randn(((13, 14, 15, 16), (4, 17, 6, 7), (1, 4, 9, 7, 1)))
        self.assertFalse(t3.are_t3_ranks_minimal(x))

    def test_pad_t3(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 6, 5), (3, 3, 2, 4)))
        new_structure = ((17, 18, 17), (8, 8, 8), (5, 5, 6, 7))
        padded_x = t3.pad_t3(x, new_structure)
        self.assertEqual(new_structure, t3.structure(padded_x))

    def test_left_svd_3tensor(self):
        G_i_a_j = np.random.randn(5, 7, 6)
        U_i_a_x, ss_x, Vt_x_j = t3.left_svd_3tensor(G_i_a_j)
        G_i_a_j2 = np.einsum('iax,x,xj->iaj', U_i_a_x, ss_x, Vt_x_j)
        self.assertLessEqual(norm(G_i_a_j - G_i_a_j2), numpy_tol * norm(G_i_a_j))

        rank = len(ss_x)
        self.assertEqual(6, rank)

        self.assertLessEqual(
            norm(np.einsum('iax,iay->xy', U_i_a_x, U_i_a_x) - np.eye(rank)),
            numpy_tol * norm(np.eye(rank))
        )
        self.assertLessEqual(
            norm(np.einsum('xj,yj->xy', Vt_x_j, Vt_x_j) - np.eye(rank)),
            numpy_tol * norm(np.eye(rank))
        )

    def test_right_svd_3tensor(self):
        G_i_a_j = np.random.randn(5, 7, 6)
        U_i_x, ss_x, Vt_x_a_j = t3.right_svd_3tensor(G_i_a_j)
        G_i_a_j2 = np.einsum('ix,x,xaj->iaj', U_i_x, ss_x, Vt_x_a_j)
        self.assertLessEqual(norm(G_i_a_j - G_i_a_j2), numpy_tol * norm(G_i_a_j))

        rank = len(ss_x)
        self.assertEqual(5, rank)

        self.assertLessEqual(
            norm(np.einsum('ix,iy->xy', U_i_x, U_i_x) - np.eye(rank)),
            numpy_tol * norm(np.eye(rank))
        )
        self.assertLessEqual(
            norm(np.einsum('xaj,yaj->xy', Vt_x_a_j, Vt_x_a_j) - np.eye(rank)),
            numpy_tol * norm(np.eye(rank))
        )

    def test_outer_svd_3tensor(self):
        G_i_a_j = np.random.randn(5, 7, 6)
        U_i_x_j, ss_x, Vt_x_a = t3.outer_svd_3tensor(G_i_a_j)
        G_i_a_j2 = np.einsum('ixj,x,xa->iaj', U_i_x_j, ss_x, Vt_x_a)
        self.assertLessEqual(norm(G_i_a_j - G_i_a_j2), numpy_tol * norm(G_i_a_j))

        rank = len(ss_x)
        self.assertEqual(7, rank)

        self.assertLessEqual(
            norm(np.einsum('ixj,iyj->xy', U_i_x_j, U_i_x_j) - np.eye(rank)),
            numpy_tol * norm(np.eye(rank))
        )
        self.assertLessEqual(
            norm(np.einsum('xa,ya->xy', Vt_x_a, Vt_x_a) - np.eye(rank)),
            numpy_tol * norm(np.eye(rank))
        )

    def test_up_svd_ith_basis_core(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (4, 3, 2, 6)
        x = t3.t3_corewise_randn((shape, tucker_ranks, tt_ranks))
        dense_x = t3.t3_to_dense(x)
        for ind in range(len(shape)):
            x2, ss = t3.up_svd_ith_basis_core(ind, x)
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
            x2, ss = t3.left_svd_ith_tt_core(ind, x)
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
            x2, ss = t3.right_svd_ith_tt_core(ind, x)
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
            x2, ss = t3.up_svd_ith_tt_core(ind, x)
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
            x2, ss = t3.down_svd_ith_tt_core(ind, x)
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
        x2 = t3.orthogonalize_relative_to_ith_basis_core(0, x)
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
        x2 = t3.orthogonalize_relative_to_ith_basis_core(1, x)
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
        x2 = t3.orthogonalize_relative_to_ith_basis_core(2, x)
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
        x2 = t3.orthogonalize_relative_to_ith_tt_core(0, x)
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
        x2 = t3.orthogonalize_relative_to_ith_tt_core(1, x)
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
        x2 = t3.orthogonalize_relative_to_ith_tt_core(2, x)
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

    def test_t3_add(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        y = t3.t3_corewise_randn(((14, 15, 16), (3, 7, 2), (1, 5, 6, 5)))
        z = t3.t3_add(x, y)
        self.assertEqual(((14, 15, 16), (7, 12, 8), (3, 8, 8, 6)), t3.structure(z))
        dense_x_plus_y = t3.t3_to_dense(z)
        dense_x_plus_y2 = t3.t3_to_dense(x) + t3.t3_to_dense(y)
        self.assertLessEqual(
            norm(dense_x_plus_y - dense_x_plus_y2),
            numpy_tol * norm(dense_x_plus_y2)
        )

    def test_t3_scale(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        s = 5.2
        sx = t3.t3_scale(x, s)
        dense_x = t3.t3_to_dense(x)
        dense_sx = t3.t3_to_dense(sx)
        self.assertLessEqual(
            norm(s * dense_x - dense_sx),
            numpy_tol * norm(s * dense_x)
        )

    def test_t3_neg(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        neg_x = t3.t3_neg(x)
        dense_x = t3.t3_to_dense(x)
        dense_neg_x = t3.t3_to_dense(neg_x)
        self.assertLessEqual(
            norm(-dense_x - dense_neg_x),
            numpy_tol * norm(-dense_x)
        )

    def test_t3_sub(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        y = t3.t3_corewise_randn(((14, 15, 16), (3, 7, 2), (1, 5, 6, 5)))
        z = t3.t3_sub(x, y)
        self.assertEqual(((14, 15, 16), (7, 12, 8), (3, 8, 8, 6)), t3.structure(z))
        dense_x_minus_y = t3.t3_to_dense(z)
        dense_x_minus_y2 = t3.t3_to_dense(x) - t3.t3_to_dense(y)
        self.assertLessEqual(
            norm(dense_x_minus_y - dense_x_minus_y2),
            numpy_tol * norm(dense_x_minus_y2)
        )

    def test_t3_dot_t3(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        y = t3.t3_corewise_randn(((14, 15, 16), (3, 7, 2), (1, 5, 6, 5)))
        x_dot_y = t3.t3_dot_t3(x, y)
        x_dot_y2 = np.sum(t3.t3_to_dense(x) * t3.t3_to_dense(y))
        self.assertLessEqual(
            np.abs(x_dot_y - x_dot_y2),
            numpy_tol * np.abs(x_dot_y2)
        )

    def test_t3_norm(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        norm_x = t3.t3_norm(x)
        norm_x2 = np.linalg.norm(t3.t3_to_dense(x))
        self.assertLessEqual(np.abs(norm_x - norm_x2), numpy_tol * np.abs(norm_x))

    def test_t3_svd1(self):
        x = t3.t3_corewise_randn(((12, 11, 10), (14, 5, 13), (1, 17, 14, 1))) #r0=rd=1
        x2, ss_basis, ss_tt = t3.t3_svd(x)  # Compute T3-SVD
        x_dense = t3.t3_to_dense(x)
        x2_dense = t3.t3_to_dense(x2)
        self.assertLessEqual(
            norm(x_dense - x2_dense), numpy_tol * norm(x2_dense)
        )
        self.assertTrue(t3.are_t3_ranks_minimal(x2))

    def test_t3_svd2(self):
        x = t3.t3_corewise_randn(((12, 11, 10), (14, 5, 13), (2, 17, 14, 3)))
        x2, ss_basis, ss_tt = t3.t3_svd(x)  # Compute T3-SVD
        x_dense = t3.t3_to_dense(x)
        x2_dense = t3.t3_to_dense(x2)
        self.assertLessEqual(
            norm(x_dense - x2_dense), numpy_tol * norm(x2_dense)
        )
        self.assertTrue(t3.are_t3_ranks_minimal(x2))

    def test_t3_svd3(self):
        x = t3.t3_corewise_randn(((12, 11, 10), (14, 5, 13), (2, 17, 14, 3)))
        x2, ss_basis, ss_tt = t3.t3_svd(x)  # Compute T3-SVD
        self.assertLessEqual(
            norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x))
        )

        (N0, N1, N2), (n0, n1, n2), (r0, r1, r2, r3) = t3.structure(x2)

        x2_dense = t3.t3_to_dense(x2, contract_ones=False)

        ss_tt0 = np.linalg.svd(x2_dense.reshape((r0, N0*N1*N2*r3)))[1]
        ss_tt1 = np.linalg.svd(x2_dense.reshape((r0*N0, N1*N2*r3)))[1]
        ss_tt2 = np.linalg.svd(x2_dense.reshape((r0*N0*N1, N2*r3)))[1]
        ss_tt3 = np.linalg.svd(x2_dense.reshape((r0*N0*N1*N2, r3)))[1]

        ss_basis0 = np.linalg.svd(x2_dense.swapaxes(0,1).reshape((N0,-1)))[1]
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
        x2, ss_basis, ss_tt = t3.t3_svd_dense(x_dense)
        x2_dense = t3.t3_to_dense(x2)
        self.assertLessEqual(norm(x_dense - x2_dense), numpy_tol * norm(x_dense))

        ss_tt0 = np.linalg.svd(x_dense.reshape((1, N0*N1*N2*1)))[1]
        ss_tt1 = np.linalg.svd(x_dense.reshape((1*N0, N1*N2*1)))[1]
        ss_tt2 = np.linalg.svd(x_dense.reshape((1*N0*N1, N2*1)))[1]
        ss_tt3 = np.linalg.svd(x_dense.reshape((1*N0*N1*N2, 1)))[1]

        ss_basis0 = np.linalg.svd(x_dense.swapaxes(0, 0).reshape((N0,-1)))[1]
        ss_basis1 = np.linalg.svd(x_dense.swapaxes(0, 1).reshape((N1, -1)))[1]
        ss_basis2 = np.linalg.svd(x_dense.swapaxes(0, 2).reshape((N2, -1)))[1]

        _, (n0, n1, n2), (r0, r1, r2, r3) = t3.structure(x2)

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