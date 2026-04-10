# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest
import os

import t3tools.tucker_tensor_train as t3

try:
    import t3tools.jax.t3 as t3_jax
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    t3_jax = t3

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestTuckerTensorTrain(unittest.TestCase):
    def check_relerr(self, xtrue, x):
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def test_get_structure(self):
        for T3 in [t3, t3_jax]:
            with self.subTest(T3=T3):
                basis_cores = (np.ones((4, 14)), np.ones((5, 15)), np.ones((6, 16)))
                tt_cores = (np.ones((2, 4, 3)), np.ones((3, 5, 7)), np.ones((7, 6, 5)))
                x = (basis_cores, tt_cores)

                shape, tucker_ranks, tt_ranks = T3.get_structure(x)

                self.assertEqual((14, 15, 16), shape)
                self.assertEqual((4, 5, 6), tucker_ranks)
                self.assertEqual((2,3,7,5), tt_ranks)

    def test_squash_tails(self):
        for T3 in [t3, t3_jax]:
            with self.subTest(T3=T3):
                x = t3.t3_corewise_randn(((11, 12, 13), (6, 7, 8), (9, 3, 4, 8)))

                x2 = T3.squash_tails(x)

                self.assertEqual(((11, 12, 13), (6, 7, 8), (1, 3, 4, 1)), t3.get_structure(x2))
                x_dense = t3.t3_to_dense(x)
                x2_dense = t3.t3_to_dense(x2)
                self.check_relerr(x_dense, x2_dense)

    def test_t3_to_dense1(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (2, 3, 7, 5)),
        ]

        for STRUCTURE in structures:
            for T3 in [t3, t3_jax]:
                with self.subTest(T3=T3, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)  # make TuckerTensorTrain

                    SHAPE, TUCKER_RANKS, TT_RANKS = STRUCTURE
                    EXTENDED_SHAPE = (TT_RANKS[0],) + SHAPE + (TT_RANKS[-1],)
                    for contract_ones, true_shape in zip([True, False], [SHAPE, EXTENDED_SHAPE]):
                        with self.subTest(contract_ones=contract_ones, true_shape=true_shape):

                            x_dense = T3.t3_to_dense(x, contract_ones=contract_ones)  # Convert TuckerTensorTrain to dense tensor

                            ((B0, B1, B2), (G0, G1, G2)) = x
                            if contract_ones:
                                x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->ijk', B0, B1, B2, G0, G1, G2)
                            else:
                                x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->aijkd', B0, B1, B2, G0, G1, G2)
                            self.assertEqual(true_shape, x_dense.shape)
                            self.check_relerr(x_dense, x_dense2)


    def test_t3_reverse(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)),
        ]

        for STRUCTURE in structures:
            for T3 in [t3, t3_jax]:
                with self.subTest(T3=T3, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)  # Make TuckerTensorTrain

                    reversed_x = T3.reverse_t3(x)

                    self.assertEqual(
                        (STRUCTURE[0][::-1], STRUCTURE[1][::-1], STRUCTURE[2][::-1]),
                        t3.get_structure(reversed_x)
                    )
                    x_dense = t3.t3_to_dense(x)
                    reversed_x_dense = t3.t3_to_dense(reversed_x)
                    x_dense2 = reversed_x_dense.transpose(list(range(len(STRUCTURE[0])))[::-1])
                    self.check_relerr(x_dense, x_dense2)

    def test_t3_zeros(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)),
        ]

        for STRUCTURE in structures:
            for T3 in [t3, t3_jax]:
                with self.subTest(T3=T3, STRUCTURE=STRUCTURE):
                    z = T3.t3_zeros(STRUCTURE)

                    self.assertEqual(STRUCTURE, t3.get_structure(z))
                    dense_z = t3.t3_to_dense(z)
                    self.assertLessEqual(norm(dense_z), tol)

    def test_t3_corewise_randn(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)),
        ]

        for STRUCTURE in structures:
            for T3 in [t3, t3_jax]:
                with self.subTest(T3=T3, STRUCTURE=STRUCTURE):
                    x = T3.t3_corewise_randn(STRUCTURE)  # TuckerTensorTrain with random cores
                    self.assertEqual(STRUCTURE, t3.get_structure(x))

    def test_t3_save_and_t3_load(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)),
        ]

        for STRUCTURE in structures:
            for T3 in [t3, t3_jax]:
                with self.subTest(T3=T3, STRUCTURE=STRUCTURE):
                    x = T3.t3_corewise_randn(STRUCTURE)

                    fname0 = 't3_saveload_test_file'
                    fname = fname0 + '.npz'
                    if os.path.exists(fname):
                        success = False
                        for ii in range(39781): # hopefully these file names are not all already existing! How unlikely
                            fname = fname0 + str(ii) + '.npz'
                            if not os.path.exists(fname):
                                success = True
                                break
                        if not success:
                            raise RuntimeError('No available filenames to save to.')

                    T3.t3_save(fname, x)  # Save to file
                    x2 = T3.t3_load(fname)  # Load from file

                    os.remove(fname)

                    basis_cores, tt_cores = x
                    basis_cores2, tt_cores2 = x2

                    for B, B2 in zip(basis_cores, basis_cores2):
                        self.check_relerr(B, B2)

                    for G, G2 in zip(tt_cores, tt_cores2):
                        self.check_relerr(G, G2)

    def test_t3_apply1(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)),
        ]

        for STRUCTURE in structures:
            for T3 in [t3, t3_jax]:
                with self.subTest(T3=T3, STRUCTURE=STRUCTURE):
                    x = T3.t3_corewise_randn(STRUCTURE)

                    SHAPE = STRUCTURE[0]
                    vecs = [np.random.randn(SHAPE[0]),
                            np.random.randn(SHAPE[1]),
                            np.random.randn(SHAPE[2])]

                    result = T3.t3_apply(x, vecs)  # <-- contract x with vecs in all indices

                    result2 = np.einsum('ijk,i,j,k', t3.t3_to_dense(x), vecs[0], vecs[1], vecs[2])
                    self.check_relerr(result2, result)

    def test_t3_apply2(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)),
        ]
        NUM_PROBES = 3

        for STRUCTURE in structures:
            for T3 in [t3, t3_jax]:
                with self.subTest(T3=T3, STRUCTURE=STRUCTURE):
                    x = T3.t3_corewise_randn(STRUCTURE)

                    SHAPE = STRUCTURE[0]
                    vecs = [np.random.randn(NUM_PROBES, SHAPE[0]),
                            np.random.randn(NUM_PROBES, SHAPE[1]),
                            np.random.randn(NUM_PROBES, SHAPE[2])]

                    result = T3.t3_apply(x, vecs)

                    result2 = np.einsum('ijk,ni,nj,nk->n', t3.t3_to_dense(x), vecs[0], vecs[1], vecs[2])
                    self.check_relerr(result2, result)

    def test_t3_entry1(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)),
        ]
        indices = [
            (9, 4, 7),
        ]

        for INDEX in indices:
            for STRUCTURE in structures:
                for T3 in [t3, t3_jax]:
                    with self.subTest(T3=T3, STRUCTURE=STRUCTURE, INDEX=INDEX):
                        x = T3.t3_corewise_randn(STRUCTURE)

                        result = T3.t3_entry(x, INDEX)

                        result2 = t3.t3_to_dense(x)[INDEX]
                        self.check_relerr(result2, result)

    def test_t3_entry2(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (6, 2, 4, 3)),
        ]
        index_sets = [
            ((9, 8), (4, 10), (7, 13)),
        ]

        for INDEX_SET in index_sets:
            for STRUCTURE in structures:
                for T3 in [t3, t3_jax]:
                    with self.subTest(T3=T3, STRUCTURE=STRUCTURE, INDEX_SET=INDEX_SET):
                        x = t3.t3_corewise_randn(STRUCTURE)

                        entries = T3.t3_entry(x, INDEX_SET)

                        x_dense = t3.t3_to_dense(x)
                        entries2 = []
                        for ii in range(len(INDEX_SET[0])):
                            INDEX = []
                            for INDEX_COMPONENT in INDEX_SET:
                                INDEX.append(INDEX_COMPONENT[ii])
                            entries2.append(x_dense[tuple(INDEX)])
                        entries2 = np.array(entries2)
                        self.check_relerr(entries2, entries)

    def test_compute_minimal_ranks(self):
        for T3 in [t3, t3_jax]:
            with self.subTest(T3=T3):

                mr = T3.compute_minimal_ranks(((10, 11, 12, 13), (14, 15, 16, 17), (98, 99, 100, 101, 102)))

                mr_true = ((10, 11, 12, 13), (1, 10, 100, 13, 1))
                self.assertEqual(mr, mr_true)

    def test_are_ranks_minimal1(self):
        structures = [
            ((13, 14, 15, 16), (4, 5, 6, 7), (1, 4, 9, 7, 1)),
            ((13, 14, 15, 16), (4, 5, 6, 7), (1, 99, 9, 7, 1)),
            ((13, 14, 15, 16), (4, 5, 6, 7), (1, 4, 9, 7, 2)),
            ((13, 14, 15, 16), (4, 17, 6, 7), (1, 4, 9, 7, 1))
        ]
        results = [
            True,
            False,
            False,
            False,
        ]

        for STRUCTURE, RESULT in zip(structures, results):
            for T3 in [t3, t3_jax]:
                with self.subTest(T3=T3):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    self.assertEqual(RESULT, T3.are_t3_ranks_minimal(x))

    def test_pad_t3(self):
        structures1 = [
            ((14, 15, 16), (4, 6, 5), (3, 3, 2, 4)),
        ]
        structures2 = [
            ((17, 18, 17), (8, 8, 8), (5, 5, 6, 7)),
        ]

        for STRUCTURE1, STRUCTURE2 in zip(structures1, structures2):
            for T3 in [t3, t3_jax]:
                with self.subTest(T3=T3, STRUCTURE1=STRUCTURE1, STRUCTURE2=STRUCTURE2):
                    x = t3.t3_corewise_randn(STRUCTURE1)
                    padded_x = T3.change_structure(x, STRUCTURE2)
                    self.assertEqual(STRUCTURE2, t3.get_structure(padded_x))

    def test_t3_add_sub(self):
        structures_x = [
            ((14, 15, 16), (4, 6, 5), (3, 3, 2, 4)),
        ]
        structures_y = [
            ((14, 15, 16), (3, 7, 2), (1, 5, 6, 5)),
        ]

        for SQUASH in [True, False]:
            for STRUCTURE_X, STRUCTURE_Y in zip(structures_x, structures_y):
                for T3 in [t3, t3_jax]:
                    for OP in [T3.t3_add, T3.t3_sub]:
                        with self.subTest(SQUASH=SQUASH, T3=T3, STRUCTURE_X=STRUCTURE_X, STRUCTURE_Y=STRUCTURE_Y, OP=OP):
                            x = t3.t3_corewise_randn(STRUCTURE_X)
                            y = t3.t3_corewise_randn(STRUCTURE_Y)

                            z = OP(x, y, squash=SQUASH)

                            Z_SHAPE         = STRUCTURE_X[0]
                            Z_TUCKER_RANKS  = tuple([SX + SY for SX, SY in zip(STRUCTURE_X[1], STRUCTURE_Y[1])])
                            Z_TT_RANKS      = tuple([SX + SY for SX, SY in zip(STRUCTURE_X[2], STRUCTURE_Y[2])])

                            if SQUASH:
                                Z_TT_RANKS = (1,) + Z_TT_RANKS[1:-1] + (1,)

                            STRUCTURE_Z = (Z_SHAPE, Z_TUCKER_RANKS, Z_TT_RANKS)

                            self.assertEqual(t3.get_structure(z), STRUCTURE_Z)
                            dense_z = t3.t3_to_dense(z)
                            if OP is T3.t3_add:
                                dense_z2 = t3.t3_to_dense(x) + t3.t3_to_dense(y)
                            else:
                                dense_z2 = t3.t3_to_dense(x) - t3.t3_to_dense(y)
                            self.check_relerr(dense_z2, dense_z)

    def test_t3_scale_neg(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)),
        ]
        SCALE_FACTOR = 5.3

        for STRUCTURE in structures:
            for T3 in [t3, t3_jax]:
                for OP in [T3.t3_scale, T3.t3_neg]:
                    with self.subTest(T3=T3, STRUCTURE=STRUCTURE, OP=OP):
                        x = t3.t3_corewise_randn(STRUCTURE)
                        dense_x = t3.t3_to_dense(x)
                        if OP is T3.t3_scale:
                            y = T3.t3_scale(x, SCALE_FACTOR) # <--
                            Y_true = SCALE_FACTOR * dense_x
                        else:
                            y = T3.t3_neg(x) # <--
                            Y_true = -dense_x

                        Y = t3.t3_to_dense(y)
                        self.check_relerr(Y_true, Y)

    def test_t3_dot_t3(self):
        structures_x = [
            ((14, 15, 16), (4, 6, 5), (3, 3, 2, 4)),
        ]
        structures_y = [
            ((14, 15, 16), (3, 7, 2), (1, 5, 6, 5)),
        ]

        for STRUCTURE_X, STRUCTURE_Y in zip(structures_x, structures_y):
            for T3 in [t3, t3_jax]:
                    with self.subTest(T3=T3, STRUCTURE_X=STRUCTURE_X, STRUCTURE_Y=STRUCTURE_Y):
                        x = t3.t3_corewise_randn(STRUCTURE_X)
                        y = t3.t3_corewise_randn(STRUCTURE_Y)
                        x_dot_y = t3.t3_inner_product_t3(x, y)
                        x_dot_y2 = np.sum(t3.t3_to_dense(x) * t3.t3_to_dense(y))
                        self.check_relerr(x_dot_y2, x_dot_y)

    def test_t3_norm(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)),
        ]

        for STRUCTURE in structures:
            for T3 in [t3, t3_jax]:
                with self.subTest(T3=T3, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    norm_x = t3.t3_norm(x)
                    norm_x2 = np.linalg.norm(t3.t3_to_dense(x))
                    self.check_relerr(norm_x2, norm_x)


if __name__ == '__main__':
    unittest.main()