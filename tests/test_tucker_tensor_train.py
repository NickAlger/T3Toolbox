# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3tools.tucker_tensor_train as t3


np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestTuckerTensorTrain(unittest.TestCase):
    def test_structure1(self):
        basis_cores = (np.ones((4, 14)), np.ones((5, 15)), np.ones((6, 16)))
        tt_cores = (np.ones((2, 4, 3)), np.ones((3, 5, 7)), np.ones((7, 6, 5)))
        x = (basis_cores, tt_cores)
        shape, tucker_ranks, tt_ranks = t3.get_structure(x)
        self.assertEqual((14, 15, 16), shape)
        self.assertEqual((4, 5, 6), tucker_ranks)
        self.assertEqual((2,3,7,5), tt_ranks)

    def test_squash_tails(self):
        x = t3.t3_corewise_randn(((11, 12, 13), (6, 7, 8), (9, 3, 4, 8)))
        x2 = t3.squash_tails(x)
        self.assertEqual(((11, 12, 13), (6, 7, 8), (1, 3, 4, 1)), t3.get_structure(x2))
        x_dense = t3.t3_to_dense(x)
        x2_dense = t3.t3_to_dense(x2)
        self.assertLessEqual(norm(x_dense - x2_dense), tol * norm(x_dense))

    def test_t3_to_dense1(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 7, 5)))  # make TuckerTensorTrain
        for contract_ones, true_shape in zip([True, False], [(14, 15, 16), (2, 14, 15, 16, 5)]):
            with self.subTest(contract_ones=contract_ones, true_shape=true_shape):
                x_dense = t3.t3_to_dense(x, contract_ones=contract_ones)  # Convert TuckerTensorTrain to dense tensor
                ((B0, B1, B2), (G0, G1, G2)) = x
                if contract_ones:
                    x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->ijk', B0, B1, B2, G0, G1, G2)
                else:
                    x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->aijkd', B0, B1, B2, G0, G1, G2)
                self.assertEqual(true_shape, x_dense.shape)
                self.assertLessEqual(norm(x_dense - x_dense2), tol * norm(x_dense))

    def test_t3_to_dense2(self):
        # leading and trailing ones not contracted
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 4, 5)))  # make TuckerTensorTrain
        x_dense = t3.t3_to_dense(x, contract_ones=False)  # Convert TuckerTensorTrain to dense tensor
        ((B0, B1, B2), (G0, G1, G2)) = x
        x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->aijkd', B0, B1, B2, G0, G1, G2)
        self.assertEqual((2, 14, 15, 16, 5), x_dense.shape)
        self.assertLessEqual(norm(x_dense - x_dense2), tol * norm(x_dense))

    def test_t3_reverse1(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))  # Make TuckerTensorTrain
        self.assertEqual(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)), t3.get_structure(x))
        reversed_x = t3.reverse_t3(x)
        self.assertEqual(((16, 15, 14), (6, 5, 4), (1, 2, 3, 2)), t3.get_structure(reversed_x))
        x_dense = t3.t3_to_dense(x)
        reversed_x_dense = t3.t3_to_dense(reversed_x)
        x_dense2 = reversed_x_dense.transpose([2, 1, 0])
        self.assertLessEqual(norm(x_dense - x_dense2), tol * norm(x_dense))

    def test_t3_zeros1(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (2, 3, 2, 1)
        structure = (shape, tucker_ranks, tt_ranks)
        z = t3.t3_zeros(structure)
        self.assertEqual(structure, t3.get_structure(z))
        dense_z = t3.t3_to_dense(z)
        self.assertLessEqual(norm(dense_z), tol)

    def test_t3_corewise_randn1(self):
        shape = (14, 15, 16)
        tucker_ranks = (4, 5, 6)
        tt_ranks = (1, 3, 2, 1)
        structure = (shape, tucker_ranks, tt_ranks)
        x = t3.t3_corewise_randn(structure)  # TuckerTensorTrain with random cores
        self.assertEqual(structure, t3.get_structure(x))

    def test_t3_save_and_t3_load1(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)))
        fname = 't3_test_file1945145272' # hopefully no one else has used this filename
        t3.t3_save(fname, x)  # Save to file 't3_file.npz'
        x2 = t3.t3_load(fname)  # Load from file
        basis_cores, tt_cores = x
        basis_cores2, tt_cores2 = x2
        for B, B2 in zip(basis_cores, basis_cores2):
            self.assertLessEqual(norm(B - B2), tol * norm(B))
        for G, G2 in zip(tt_cores, tt_cores2):
            self.assertLessEqual(norm(G - G2), tol * norm(G))

    def test_t3_apply1(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)))
        vecs = [randn(14), randn(15), randn(16)]
        result = t3.t3_apply(x, vecs)  # <-- contract x with vecs in all indices
        result2 = np.einsum('ijk,i,j,k', t3.t3_to_dense(x), vecs[0], vecs[1], vecs[2])
        self.assertLessEqual(np.abs(result - result2), tol * np.abs(result2))

    def test_t3_apply2(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)))
        vecs = [randn(3, 14), randn(3, 15), randn(3, 16)]
        result = t3.t3_apply(x, vecs)
        result2 = np.einsum('ijk,ni,nj,nk->n', t3.t3_to_dense(x), vecs[0], vecs[1], vecs[2])
        self.assertLessEqual(norm(result - result2), tol * norm(result2))

    def test_t3_entry1(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)))
        index = [9, 4, 7]  # get entry (9,4,7)
        result = t3.t3_entry(x, index)
        result2 = t3.t3_to_dense(x)[9, 4, 7]
        self.assertLessEqual(np.abs(result - result2), tol * np.abs(result2))

    def test_t3_entry2(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (6, 2, 4, 3)))
        index = [[9, 8], [4, 10], [7, 13]]  # get entries (9,4,7) and (8,10,13)
        entries = t3.t3_entry(x, index)
        x_dense = t3.t3_to_dense(x)
        entries2 = np.array([x_dense[9, 4, 7], x_dense[8, 10, 13]])
        self.assertLessEqual(norm(entries - entries2), tol * norm(entries2))

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
        self.assertEqual(new_structure, t3.get_structure(padded_x))

    def test_t3_add(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        y = t3.t3_corewise_randn(((14, 15, 16), (3, 7, 2), (1, 5, 6, 5)))
        z = t3.t3_add(x, y)
        self.assertEqual(((14, 15, 16), (7, 12, 8), (3, 8, 8, 6)), t3.get_structure(z))
        dense_x_plus_y = t3.t3_to_dense(z)
        dense_x_plus_y2 = t3.t3_to_dense(x) + t3.t3_to_dense(y)
        self.assertLessEqual(
            norm(dense_x_plus_y - dense_x_plus_y2),
            tol * norm(dense_x_plus_y2)
        )

    def test_t3_scale(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        s = 5.2
        sx = t3.t3_scale(x, s)
        dense_x = t3.t3_to_dense(x)
        dense_sx = t3.t3_to_dense(sx)
        self.assertLessEqual(
            norm(s * dense_x - dense_sx),
            tol * norm(s * dense_x)
        )

    def test_t3_neg(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        neg_x = t3.t3_neg(x)
        dense_x = t3.t3_to_dense(x)
        dense_neg_x = t3.t3_to_dense(neg_x)
        self.assertLessEqual(
            norm(-dense_x - dense_neg_x),
            tol * norm(-dense_x)
        )

    def test_t3_sub(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        y = t3.t3_corewise_randn(((14, 15, 16), (3, 7, 2), (1, 5, 6, 5)))
        z = t3.t3_sub(x, y)
        self.assertEqual(((14, 15, 16), (7, 12, 8), (3, 8, 8, 6)), t3.get_structure(z))
        dense_x_minus_y = t3.t3_to_dense(z)
        dense_x_minus_y2 = t3.t3_to_dense(x) - t3.t3_to_dense(y)
        self.assertLessEqual(
            norm(dense_x_minus_y - dense_x_minus_y2),
            tol * norm(dense_x_minus_y2)
        )

    def test_t3_dot_t3(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        y = t3.t3_corewise_randn(((14, 15, 16), (3, 7, 2), (1, 5, 6, 5)))
        x_dot_y = t3.t3_dot_t3(x, y)
        x_dot_y2 = np.sum(t3.t3_to_dense(x) * t3.t3_to_dense(y))
        self.assertLessEqual(
            np.abs(x_dot_y - x_dot_y2),
            tol * np.abs(x_dot_y2)
        )

    def test_t3_norm(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)))
        norm_x = t3.t3_norm(x)
        norm_x2 = np.linalg.norm(t3.t3_to_dense(x))
        self.assertLessEqual(np.abs(norm_x - norm_x2), tol * np.abs(norm_x))


if __name__ == '__main__':
    unittest.main()