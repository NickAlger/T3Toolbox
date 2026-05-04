# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest
import os
import itertools

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.corewise as cw
from t3toolbox.backend.common import *
from t3toolbox.backend.tucker_tensor_train.t3_operations import squash_tt_tails

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jnp = np

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn


def structure_to_cores(STRUCTURE):
    shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE

    tucker_cores = tuple(
        np.random.randn(*(stack_shape + (n, N)))
        for n, N in zip(tucker_ranks, shape)
    )
    tt_cores = tuple(
        np.random.randn(*(stack_shape + (rL, n, rR)))
        for rL, n, rR in zip(tt_ranks[:-1], tucker_ranks, tt_ranks[1:])
    )
    return tucker_cores, tt_cores


class TestTuckerTensorTrain(unittest.TestCase):
    def check_relerr(self, xtrue, x):
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def test_t3_validate(self):
        tucker_cores = [np.ones((2,3, 4,14)), np.ones((2,3, 5,15)), np.ones((2,3, 6,16))]
        tt_cores = [np.ones((2,3, 5,4,3)), np.ones((2,3, 3,5,2)), np.ones((2,3, 2,6,3))]
        t3.TuckerTensorTrain(tucker_cores, tt_cores)  # Good. Don't raise error

        with self.assertRaises(ValueError):
            tucker_cores = [np.ones((2,3, 4,14)), np.ones((2,3, 5,15))]
            tt_cores = [np.ones((2,3, 5,4,3)), np.ones((2,3, 3, 5,2)), np.ones((2,3, 2,6,3))]
            t3.TuckerTensorTrain(tucker_cores, tt_cores) # Different number of Tucker and TT cores

        with self.assertRaises(ValueError):
            tucker_cores = ()
            tt_cores = ()
            t3.TuckerTensorTrain(tucker_cores, tt_cores) # Empty TuckerTensorTrain not supported

        with self.assertRaises(ValueError):
            tucker_cores = [np.ones((2,3, 4,14)), np.ones((2,3, 5,15)), np.ones((2,3, 6,16))]
            tt_cores = [np.ones((2,3, 5,4,3)), np.ones((2,3, 3,5,2))]
            t3.TuckerTensorTrain(tucker_cores, tt_cores)  # Too few TT-cores

        with self.assertRaises(ValueError):
            tucker_cores = [np.ones((2,3, 14)), np.ones((2,3, 5,15)), np.ones((2,3, 6,16))]
            tt_cores = [np.ones((2,3, 5,4,3)), np.ones((2,3, 3,5,2)), np.ones((2,3, 2,6,3))]
            x =t3.TuckerTensorTrain(tucker_cores, tt_cores)  # Tucker core is not a matrix

        with self.assertRaises(ValueError):
            tucker_cores = [np.ones((2,3, 4,14)), np.ones((2,3, 5,15)), np.ones((2,3, 6,16))]
            tt_cores = [np.ones((2,3, 5,4,3)), np.ones((2,3, 3,5,2,1)), np.ones((2,3, 2,6,3))]
            t3.TuckerTensorTrain(tucker_cores, tt_cores)  # TT-cores is not a 3-tensor

        with self.assertRaises(ValueError):
            tucker_cores = [np.ones((2,3, 4,14)), np.ones((2,3, 5,15)), np.ones((2,3, 6,16))]
            tt_cores = [np.ones((2,3, 5,4,6)), np.ones((2,3, 3,5,2)), np.ones((2,3, 2,6,3))]
            t3.TuckerTensorTrain(tucker_cores, tt_cores)  # TT-ranks inconsistent with each other

        with self.assertRaises(ValueError):
            tucker_cores = [np.ones((2,3, 6,14)), np.ones((2,3, 5,15)), np.ones((2,3, 6,16))]
            tt_cores = [np.ones((2,3, 5,4,3)), np.ones((2,3, 3,5,2)), np.ones((2,3, 2,6,3))]
            t3.TuckerTensorTrain(tucker_cores, tt_cores)  # TT and Tucker cores have inconsistent Tucker ranks

        with self.assertRaises(ValueError):
            tucker_cores = [np.ones((2,1, 4,14)), np.ones((2,3, 5,15)), np.ones((2,3, 6,16))]
            tt_cores = [np.ones((2,3, 5,4,3)), np.ones((2,3, 3,5,2)), np.ones((2,3, 2,6,3))]
            t3.TuckerTensorTrain(tucker_cores, tt_cores)  # Inconsistent stack shapes

    def test_structural_properties(self):
        #   (shape,             tucker_ranks,   tt_ranks,           stack_shape)
        all_structures = [
            ((14, 15, 16),      (4, 5, 6),      (4, 5, 3, 2),       (2, 3)),
            ((14, 15, 16),      (4, 5, 6),      (1 ,2, 3, 1),       (2, 3)),
            ((14, 15, 16),      (4, 25, 6),     (4, 5, 3, 2),       (2, 3)),
            ((),                (),             (4,),               (2, 3)), # empty edge of size 4
            ((14,),             (4,),           (4, 5),             (2, 3)),
            ((14, 15),          (4, 5),         (4, 5, 3),          (2, 3)),
            ((14, 15, 16, 17),  (4, 5, 6, 7),   (4, 5, 3, 2, 1),    (2, 3)),
            ((14, 15, 16),      (4, 5, 6),      (4, 5, 3, 2),       ()),
        ]
        for STRUCTURE in all_structures:
            with self.subTest(STRUCTURE=STRUCTURE):
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                tucker_cores, tt_cores = structure_to_cores(STRUCTURE)

                print([x.shape for x in tucker_cores])
                print([x.shape for x in tt_cores])

                x = t3.TuckerTensorTrain(tucker_cores, tt_cores)  # random TuckerTensorTrain

                self.assertEqual((tucker_cores, tt_cores), x.data)
                self.assertEqual(len(shape),    x.d)
                self.assertEqual(len(shape)==0, x.is_empty)
                self.assertEqual(stack_shape,   x.stack_shape)
                self.assertEqual(shape,         x.shape)
                self.assertEqual(tucker_ranks,  x.tucker_ranks)
                self.assertEqual(tt_ranks,      x.tt_ranks)
                self.assertEqual(STRUCTURE,     x.structure)
                self.assertEqual(
                    (
                        tuple((n, N) for n, N in zip(tucker_ranks, shape)),
                        tuple((rL, n, rR) for rL, n, rR in zip(tt_ranks[:-1], tucker_ranks, tt_ranks[1:])),
                    ),
                    x.core_shapes,
                )
                self.assertEqual(
                    sum(x.size for x in tucker_cores) + sum(x.size for x in tt_cores),
                    x.size,
                )

    def test_minimal_ranks(self):
        structures = [
            ((14, 15, 16),      (4, 6, 5),      (1, 4, 5, 1),       (2, 3)), # minimal
            ((14, 15, 16),      (5, 6, 5),      (1, 4, 5, 1),       (2, 3)), # tt rank too small vs tucker rank
            ((14, 15, 16),      (4, 6, 5),      (1, 40, 5, 1),      (2, 3)), # tt rank too big
            ((14, 15, 16),      (4, 60, 5),     (1, 4, 5, 1),       (2, 3)), # tucker rank too big
            ((14, 15, 16),      (4, 6, 5),      (2, 4, 5, 1),       (2, 3)), # not squashed
            ((14, 15, 16),      (4, 6, 5),      (1, 4, 5, 1),       ()), # minimal, no stacking.
        ]
        minimal_structures = [
            ((14, 15, 16),      (4, 6, 5),      (1, 4, 5, 1),       (2, 3)), # do nothing
            ((14, 15, 16),      (4, 6, 5),      (1, 4, 5, 1),       (2, 3)), # decrease tucker rank
            ((14, 15, 16),      (4, 6, 5),      (1, 4, 5, 1),       (2, 3)), # decrease tt-rank
            ((14, 15, 16),      (4, 15, 5),     (1, 4, 5, 1),       (2, 3)), # decrease tucker rank
            ((14, 15, 16),      (4, 6, 5),      (1, 4, 5, 1),       (2, 3)), # squash
            ((14, 15, 16),      (4, 6, 5),      (1, 4, 5, 1),       ()), # do nothing
        ]

        for STRUCTURE, MIN_STRUCTURE in zip(structures, minimal_structures):
            with self.subTest(STRUCTURE=STRUCTURE):
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                tucker_cores, tt_cores = structure_to_cores(STRUCTURE)
                x = t3.TuckerTensorTrain(tucker_cores, tt_cores)  # random TuckerTensorTrain

                is_minimal = True
                for n, N in zip(tucker_ranks, shape):
                    is_minimal = is_minimal and n <= N

                for rL, n, rR in zip(tt_ranks[:-1], tucker_ranks, tt_ranks[1:]):
                    is_minimal = is_minimal and rL <= n * rR
                    is_minimal = is_minimal and n <= rL * rR
                    is_minimal = is_minimal and rR <= rL * n

                is_minimal = is_minimal and tt_ranks[0] == 1
                is_minimal = is_minimal and tt_ranks[-1] == 1

                self.assertEqual(is_minimal,            x.has_minimal_ranks)
                self.assertEqual(MIN_STRUCTURE[1:3],    x.minimal_ranks)

    def test_to_dense(self):
        structures = [
            ((8, 9, 7), (3, 4, 5), (2, 3, 7, 5), (2, 3)),
            ((8, 9, 7), (3, 4, 5), (2, 3, 7, 5), ()), # no stacking
            ((8, 9, 7), (3, 4, 5), (1, 3, 7, 1), (2,3)), # no tails to squash
        ]

        for STRUCTURE in structures:
            for SQUASH_TAILS in [True, False]:
                for USE_JAX in [True, False]:
                    with self.subTest(STRUCTURE=STRUCTURE, SQUASH_TAILS=SQUASH_TAILS, USE_JAX=USE_JAX):
                        shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                        tucker_cores, tt_cores = structure_to_cores(STRUCTURE)
                        x = t3.TuckerTensorTrain(tucker_cores, tt_cores)  # random TuckerTensorTrain

                        x_dense = x.to_dense(squash_tails=SQUASH_TAILS, use_jax=USE_JAX)

                        ((B0, B1, B2), (G0, G1, G2)) = tucker_cores, tt_cores
                        ss = 'LMNOP'[:len(stack_shape)]
                        if SQUASH_TAILS:
                            x_dense2 = np.einsum(
                                ss+'xi,' + ss+'yj,' + ss+'zk,' + ss+'axb,' + ss+'byc,' + ss+'czd' +
                                '->' +
                                ss+'ijk',
                                B0, B1, B2, G0, G1, G2,
                            )
                        else:
                            x_dense2 = np.einsum(
                                ss+'xi,' + ss+'yj,' + ss+'zk,' + ss+'axb,' + ss+'byc,' + ss+'czd' +
                                '->' +
                                ss+'aijkd',
                                B0, B1, B2, G0, G1, G2,
                            )

                        self.assertEqual(x_dense.shape, x_dense2.shape)
                        self.check_relerr(x_dense,      x_dense2)

    def test_segment(self):
        tk = (randn(4,14), randn(5,15), randn(6,16), randn(7,17), randn(8,18), randn(9,19))
        tt = (randn(2,4,3), randn(3,5,2), randn(2,6,2), randn(2,7,3), (randn(3,8,4)), (randn(4,9,1)))
        x = t3.TuckerTensorTrain(tk[:3], tt[:3])
        y = t3.TuckerTensorTrain(tk[3:4], tt[3:4])
        z = t3.TuckerTensorTrain(tk[4:], tt[4:])

        xyz = t3.TuckerTensorTrain(tk, tt)

        x2 = xyz.segment(0,3)
        self.assertLessEqual(cw.corewise_relerr(x.data, x2.data), tol * cw.corewise_norm(x.data))

        x3 = xyz.segment(None,3)
        self.assertLessEqual(cw.corewise_relerr(x.data, x3.data), tol * cw.corewise_norm(x.data))

        #

        y2 = xyz.segment(3, 4)
        self.assertLessEqual(cw.corewise_relerr(y.data, y2.data), tol * cw.corewise_norm(y.data))

        y3 = xyz.segment(3, -2)
        self.assertLessEqual(cw.corewise_relerr(y.data, y3.data), tol * cw.corewise_norm(y.data))

        y4 = xyz.segment(-3, 4)
        self.assertLessEqual(cw.corewise_relerr(y.data, y4.data), tol * cw.corewise_norm(y.data))

        y5 = xyz.segment(-3, -2)
        self.assertLessEqual(cw.corewise_relerr(y.data, y5.data), tol * cw.corewise_norm(y.data))

        #

        z2 = xyz.segment(4, 6)
        self.assertLessEqual(cw.corewise_relerr(z.data, z2.data), tol * cw.corewise_norm(z.data))

        z3 = xyz.segment(4, None)
        self.assertLessEqual(cw.corewise_relerr(z.data, z3.data), tol * cw.corewise_norm(z.data))


    def test_concatenate(self):
        tk = (randn(4,14), randn(5,15), randn(6,16), randn(7,17), randn(8,18), randn(9,19))
        tt = (randn(2,4,3), randn(3,5,2), randn(2,6,2), randn(2,7,3), (randn(3,8,4)), (randn(4,9,1)))
        x = t3.TuckerTensorTrain(tk[:3], tt[:3])
        y = t3.TuckerTensorTrain(tk[3:4], tt[3:4])
        z = t3.TuckerTensorTrain(tk[4:], tt[4:])

        x2 = t3.TuckerTensorTrain.concatenate([x])
        self.assertLessEqual(cw.corewise_relerr(x.data, x2.data), tol * cw.corewise_norm(x.data))

        xy = t3.TuckerTensorTrain(tk[:4], tt[:4])
        xy2 = t3.TuckerTensorTrain.concatenate([x, y])
        self.assertLessEqual(cw.corewise_relerr(xy.data, xy2.data), tol * cw.corewise_norm(xy.data))

        xyz = t3.TuckerTensorTrain(tk, tt)
        xyz2 = t3.TuckerTensorTrain.concatenate([x, y, z])
        self.assertLessEqual(cw.corewise_relerr(xyz.data, xyz2.data), tol * cw.corewise_norm(xyz.data))

    def test_squash_tails(self):
        structures = [
            ((8, 9, 7), (3, 4, 5), (2, 3, 7, 5), (2, 3)),
            ((8, 9, 7), (3, 4, 5), (2, 3, 7, 5), ()), # no stacking
            ((8, 9, 7), (3, 4, 5), (1, 3, 7, 1), (2,3)), # no tails to squash
        ]

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                with self.subTest(STRUCTURE=STRUCTURE, USE_JAX=USE_JAX):
                    shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                    tucker_cores, tt_cores = structure_to_cores(STRUCTURE)
                    x = t3.TuckerTensorTrain(tucker_cores, tt_cores)  # random TuckerTensorTrain

                    x2 = x.squash_tails(use_jax=USE_JAX)

                    squashed_tt_ranks = (1,) + tt_ranks[1:-1] + (1,)
                    squashed_structure = (shape, tucker_ranks, squashed_tt_ranks, stack_shape)

                    self.assertEqual(squashed_structure, x2.structure)
                    self.check_relerr(x.to_dense(), x2.to_dense())

    def test_reverse(self):
        all_structures = [
            ((14,),             (4,),           (4, 5),             (2, 3)),
            ((14, 15),          (4, 5),         (4, 5, 3),          (2, 3)),
            ((14, 15, 16, 17),  (4, 5, 6, 7),   (4, 5, 3, 2, 1),    (2, 3)),
            ((14, 15, 16),      (4, 5, 6),      (4, 5, 3, 2),       ()),
        ]

        for STRUCTURE in all_structures:
            with self.subTest(STRUCTURE=STRUCTURE):
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                tucker_cores, tt_cores = structure_to_cores(STRUCTURE)
                x = t3.TuckerTensorTrain(tucker_cores, tt_cores)  # random TuckerTensorTrain

                reversed_x = x.reverse()

                reversed_structure = (shape[::-1], tucker_ranks[::-1], tt_ranks[::-1], stack_shape)
                self.assertEqual(reversed_structure, reversed_x.structure)

                x_dense = x.to_dense()
                reversed_x_dense = reversed_x.to_dense()

                nss = len(stack_shape)
                transpose_inds = tuple(range(nss)) + tuple(range(nss, nss+len(shape)))[::-1]

                x_dense2 = reversed_x_dense.transpose(transpose_inds)
                self.check_relerr(x_dense, x_dense2)

    def test_resize_cores(self):
        structures = [
            ((14,),             (4,),           (4, 5),             (2, 3)),
            ((14, 15),          (4, 5),         (4, 5, 4),          (2, 3)),
            ((14, 15, 16, 17),  (4, 5, 6, 7),   (4, 5, 4, 3, 2),    (2, 3)),
            ((14, 15, 16),      (4, 5, 6),      (4, 5, 4, 3),       ()),
        ]

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                tucker_cores, tt_cores = structure_to_cores(STRUCTURE)
                x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
                dense_x = x.to_dense()

                with self.subTest(STRUCTURE=STRUCTURE, USE_JAX=USE_JAX, OP='DO_NOTHING'):
                    x2 = x.resize_cores(shape, tucker_ranks, tt_ranks, use_jax=USE_JAX)
                    self.check_relerr(dense_x, x2.to_dense())

                with self.subTest(STRUCTURE=STRUCTURE, USE_JAX=USE_JAX, OP='INCREASE_SHAPE'):
                    new_shape = tuple(s + 3 for s in shape)
                    x2 = x.resize_cores(new_shape, tucker_ranks, tt_ranks, use_jax=USE_JAX)
                    self.assertEqual(new_shape, x2.shape)
                    self.assertEqual(tucker_ranks, x2.tucker_ranks)
                    self.assertEqual(tt_ranks, x2.tt_ranks)
                    self.assertEqual(stack_shape, x2.stack_shape)

                    dense_x2 = x2.to_dense()

                    pad = [(0,0) for _ in range(len(stack_shape))]
                    pad = pad + [(0, ns - s) for ns, s in zip(new_shape, shape)]
                    padded_dense_x = np.pad(dense_x, pad)
                    self.check_relerr(padded_dense_x, dense_x2)

                with self.subTest(STRUCTURE=STRUCTURE, USE_JAX=USE_JAX, OP='INCREASE_TUCKER_RANKS'):
                    new_tucker_ranks = tuple(r + 3 for r in tucker_ranks)
                    x2 = x.resize_cores(shape, new_tucker_ranks, tt_ranks, use_jax=USE_JAX)
                    self.assertEqual(shape, x2.shape)
                    self.assertEqual(new_tucker_ranks, x2.tucker_ranks)
                    self.assertEqual(tt_ranks, x2.tt_ranks)
                    self.assertEqual(stack_shape, x2.stack_shape)

                    dense_x2 = x2.to_dense()
                    self.check_relerr(dense_x, dense_x2)

                with self.subTest(STRUCTURE=STRUCTURE, USE_JAX=USE_JAX, OP='INCREASE_TT_RANKS'):
                    new_tt_ranks = tuple(n + 3 for n in tt_ranks)
                    x2 = x.resize_cores(shape, tucker_ranks, new_tt_ranks, use_jax=USE_JAX)
                    self.assertEqual(shape, x2.shape)
                    self.assertEqual(tucker_ranks, x2.tucker_ranks)
                    self.assertEqual(new_tt_ranks, x2.tt_ranks)
                    self.assertEqual(stack_shape, x2.stack_shape)

                    dense_x2 = x2.to_dense()
                    self.check_relerr(dense_x, dense_x2)

                with self.subTest(STRUCTURE=STRUCTURE, USE_JAX=USE_JAX, OP='TRUNCATE_SHAPE'):
                    new_shape = tuple(s - 1 for s in shape)
                    x2 = x.resize_cores(new_shape, tucker_ranks, tt_ranks, use_jax=USE_JAX)
                    self.assertEqual(new_shape, x2.shape)
                    self.assertEqual(tucker_ranks, x2.tucker_ranks)
                    self.assertEqual(tt_ranks, x2.tt_ranks)
                    self.assertEqual(stack_shape, x2.stack_shape)

                    for B, B2, N in zip(x.tucker_cores, x2.tucker_cores, new_shape):
                        B = np.moveaxis(np.moveaxis(B, -1,0)[:N], 0, -1)
                        self.check_relerr(B, B2)

                with self.subTest(STRUCTURE=STRUCTURE, USE_JAX=USE_JAX, OP='TRUNCATE_TUCKER_RANKS'):
                    new_tucker_ranks = tuple(n - 1 for n in tucker_ranks)
                    x2 = x.resize_cores(shape, new_tucker_ranks, tt_ranks, use_jax=USE_JAX)
                    self.assertEqual(shape, x2.shape)
                    self.assertEqual(new_tucker_ranks, x2.tucker_ranks)
                    self.assertEqual(tt_ranks, x2.tt_ranks)
                    self.assertEqual(stack_shape, x2.stack_shape)

                    for B, B2, n in zip(x.tucker_cores, x2.tucker_cores, new_tucker_ranks):
                        B = np.moveaxis(np.moveaxis(B, -2,0)[:n], 0, -2)
                        self.check_relerr(B, B2)

                    for G, G2, n in zip(x.tt_cores, x2.tt_cores, new_tucker_ranks):
                        G = np.moveaxis(np.moveaxis(G, -2,0)[:n], 0, -2)
                        self.check_relerr(G, G2)

                with self.subTest(STRUCTURE=STRUCTURE, USE_JAX=USE_JAX, OP='TRUNCATE_TT_RANKS'):
                    new_tt_ranks = tuple(r - 1 for r in tt_ranks)
                    x2 = x.resize_cores(shape, tucker_ranks, new_tt_ranks, use_jax=USE_JAX)
                    self.assertEqual(shape, x2.shape)
                    self.assertEqual(tucker_ranks, x2.tucker_ranks)
                    self.assertEqual(new_tt_ranks, x2.tt_ranks)
                    self.assertEqual(stack_shape, x2.stack_shape)

                    for G, G2, rL, rR in zip(x.tt_cores, x2.tt_cores, new_tt_ranks[:-1], new_tt_ranks[1:]):
                        G = np.moveaxis(np.moveaxis(G, (-3,-1), (0,1))[:rL,:rR], (0,1), (-3,-1))
                        self.check_relerr(G, G2)

        with self.subTest(OP='GENERIC_RESIZE'):
            shape = (14, 15, 16, 17)
            tucker_ranks = (4, 5, 6, 7)
            tt_ranks = (4, 5, 4, 3, 2)
            stack_shape = (2, 3)
            delta_shape = (2, -3, 0, 1)
            delta_tucker_ranks = (1,0,-4,-1)
            delta_tt_ranks = (3, -3, 3, -3, 0)
            new_shape = tuple(s+ds for s, ds in zip(shape, delta_shape))
            new_tucker_ranks = tuple(n + dn for n, dn in zip(tucker_ranks, delta_tucker_ranks))
            new_tt_ranks = tuple(r + dr for r, dr in zip(tt_ranks, delta_tt_ranks))

            tucker_cores, tt_cores = structure_to_cores((shape, tucker_ranks, tt_ranks, stack_shape))
            x = t3.TuckerTensorTrain(tucker_cores, tt_cores)

            x2 = x.resize_cores(new_shape, new_tucker_ranks, new_tt_ranks)
            self.assertEqual(new_shape, x2.shape)
            self.assertEqual(new_tucker_ranks, x2.tucker_ranks)
            self.assertEqual(new_tt_ranks, x2.tt_ranks)
            self.assertEqual(stack_shape, x2.stack_shape)

            for B, B2, N, n, N2, n2 in zip(
                    x.tucker_cores, x2.tucker_cores,
                    shape, tucker_ranks,
                    new_shape, new_tucker_ranks,
            ):
                N_small = min(N, N2)
                n_small = min(n, n2)
                self.check_relerr(B[:,:,:n_small,:N_small], B2[:,:,:n_small,:N_small])
                self.assertLessEqual(np.linalg.norm(B2[:, :, n_small:, :]), tol)
                self.assertLessEqual(np.linalg.norm(B2[:, :, :, N_small:]), tol)

            for G, G2, rL, n, rR, rL2, n2, rR2 in zip(
                    x.tt_cores, x2.tt_cores,
                    tt_ranks[:-1], tucker_ranks, tt_ranks[1:],
                    new_tt_ranks[:-1], new_tucker_ranks, new_tt_ranks[1:],
            ):
                rL_small = min(rL, rL2)
                n_small = min(n, n2)
                rR_small = min(rR, rR2)
                self.check_relerr(G[:,:, :rL_small,:n_small,:rR_small], G2[:,:, :rL_small,:n_small,:rR_small])
                self.assertLessEqual(np.linalg.norm(G2[:,:, rL_small:,:,:]), tol)
                self.assertLessEqual(np.linalg.norm(G2[:,:, :,n_small:,:]), tol)
                self.assertLessEqual(np.linalg.norm(G2[:,:, :,:,rR_small:]), tol)

    def test_unstack(self):
        base_structures = [
            ((14,),             (4,),           (4, 5)),
            ((14, 15),          (4, 5),         (4, 5, 4)),
            ((14, 15, 16, 17),  (4, 5, 6, 7),   (4, 5, 4, 3, 2)),
        ]
        stack_shapes = [(), (1,), (2,), (1,1), (1,3), (2,3), (2,1)]

        for BASE_STRUCTURE in base_structures:
            for STACK_SHAPE in stack_shapes:
                with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    shape, tucker_ranks, tt_ranks = BASE_STRUCTURE
                    structure = BASE_STRUCTURE + (STACK_SHAPE,)
                    tucker_cores, tt_cores = structure_to_cores(structure)
                    x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
                    dense_x = x.to_dense()

                    xx = x.unstack()

                    if len(STACK_SHAPE) == 0:
                        self.assertTrue(isinstance(xx, t3.TuckerTensorTrain))
                        self.assertEqual(shape, xx.shape)
                        self.assertEqual(tucker_ranks, xx.tucker_ranks)
                        self.assertEqual(tt_ranks, xx.tt_ranks)
                        self.assertEqual((), x.stack_shape)
                        self.check_relerr(dense_x, xx.to_dense())

                    elif len(STACK_SHAPE) == 1:
                        self.assertEqual(STACK_SHAPE[0], len(xx))
                        for ii in range(STACK_SHAPE[0]):
                            self.assertTrue(isinstance(xx[ii], t3.TuckerTensorTrain))
                            self.assertEqual(shape, xx[ii].shape)
                            self.assertEqual(tucker_ranks, xx[ii].tucker_ranks)
                            self.assertEqual(tt_ranks, xx[ii].tt_ranks)
                            self.assertEqual((), xx[ii].stack_shape)
                            self.check_relerr(dense_x[ii], xx[ii].to_dense())

                    elif len(STACK_SHAPE) == 2:
                        self.assertEqual(STACK_SHAPE[0], len(xx))
                        for ii in range(STACK_SHAPE[0]):
                            self.assertEqual(STACK_SHAPE[1], len(xx[ii]))
                            for jj in range(STACK_SHAPE[1]):
                                self.assertTrue(isinstance(xx[ii][jj], t3.TuckerTensorTrain))
                                self.assertEqual(shape, xx[ii][jj].shape)
                                self.assertEqual(tucker_ranks, xx[ii][jj].tucker_ranks)
                                self.assertEqual(tt_ranks, xx[ii][jj].tt_ranks)
                                self.assertEqual((), xx[ii][jj].stack_shape)
                                self.check_relerr(dense_x[ii,jj], xx[ii][jj].to_dense())

    def test_stack(self):
        base_structures = [
            ((14,),             (4,),           (4, 5)),
            ((14, 15),          (4, 5),         (4, 5, 4)),
            ((14, 15, 16, 17),  (4, 5, 6, 7),   (4, 5, 4, 3, 2)),
        ]
        stack_shapes = [(), (1,), (2,), (1,1), (1,3), (2,3), (2,1)]

        for BASE_STRUCTURE in base_structures:
            for STACK_SHAPE in stack_shapes:
                with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    shape, tucker_ranks, tt_ranks = BASE_STRUCTURE
                    structure = BASE_STRUCTURE + ((),)

                    if len(STACK_SHAPE) == 0:
                        tucker_cores, tt_cores = structure_to_cores(structure)
                        xx = t3.TuckerTensorTrain(tucker_cores, tt_cores)
                        xx_dense = xx.to_dense()

                    if len(STACK_SHAPE) == 1:
                        xx = []
                        xx_dense = []
                        for ii in range(STACK_SHAPE[0]):
                            tucker_cores, tt_cores = structure_to_cores(structure)
                            xi = t3.TuckerTensorTrain(tucker_cores, tt_cores)
                            xx.append(xi)
                            xx_dense.append(xi.to_dense())

                    if len(STACK_SHAPE) == 2:
                        xx = []
                        xx_dense = []
                        for ii in range(STACK_SHAPE[0]):
                            xxi = []
                            xxi_dense = []
                            for jj in range(STACK_SHAPE[1]):
                                tucker_cores, tt_cores = structure_to_cores(structure)
                                xi = t3.TuckerTensorTrain(tucker_cores, tt_cores)
                                xxi.append(xi)
                                xxi_dense.append(xi.to_dense())
                            xx.append(xxi)
                            xx_dense.append(xxi_dense)

                    x = t3.TuckerTensorTrain.stack(xx)
                    self.assertEqual(shape, x.shape)
                    self.assertEqual(tucker_ranks, x.tucker_ranks)
                    self.assertEqual(tt_ranks, x.tt_ranks)
                    self.assertEqual(STACK_SHAPE, x.stack_shape)
                    self.check_relerr(np.array(xx_dense), x.to_dense())

    def test_zeros(self):
        structures = [
            ((14,), (4,), (4, 5), (2, 3)),
            ((14, 15), (4, 5), (4, 5, 4), (2, 3)),
            ((14, 15, 16, 17), (4, 5, 6, 7), (4, 5, 4, 3, 2), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), ()),
        ]

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                for TUCKER_RANKS in [tucker_ranks, None]:
                    for TT_RANKS in [tt_ranks, None]:
                        with self.subTest(STRUCTURE=STRUCTURE, USE_JAX=USE_JAX, TUCKER_RANKS=TUCKER_RANKS, TT_RANKS=TT_RANKS):
                            if TUCKER_RANKS is None and TT_RANKS is None:
                                x = t3.TuckerTensorTrain.zeros(
                                    shape, stack_shape=stack_shape, use_jax=USE_JAX,
                                )
                                self.assertEqual((1,)*len(shape), x.tucker_ranks)
                                self.assertEqual((1,)*(len(shape)+1), x.tt_ranks)
                            elif TUCKER_RANKS is None:
                                x = t3.TuckerTensorTrain.zeros(
                                    shape, tt_ranks=tt_ranks, stack_shape=stack_shape,
                                    use_jax=USE_JAX,
                                )
                                self.assertEqual((1,)*len(shape), x.tucker_ranks)
                                self.assertEqual(tt_ranks, x.tt_ranks)
                            elif TT_RANKS is None:
                                x = t3.TuckerTensorTrain.zeros(
                                    shape, tucker_ranks=tucker_ranks, stack_shape=stack_shape,
                                    use_jax=USE_JAX,
                                )
                                self.assertEqual(tucker_ranks, x.tucker_ranks)
                                self.assertEqual((1,)*(len(shape)+1), x.tt_ranks)
                            else:
                                x = t3.TuckerTensorTrain.zeros(
                                    shape, tucker_ranks=tucker_ranks, tt_ranks=tt_ranks, stack_shape=stack_shape,
                                    use_jax=USE_JAX,
                                )
                                self.assertEqual(tucker_ranks, x.tucker_ranks)
                                self.assertEqual(tt_ranks, x.tt_ranks)

                            self.assertEqual(shape, x.shape)
                            self.assertEqual(stack_shape, x.stack_shape)
                            self.assertLessEqual(np.linalg.norm(x.to_dense()), tol)

    def test_ones(self):
        shapes = [
            (14,),
            (14, 15),
            (14, 15, 16),
            (14, 15, 16, 17),
        ]
        stack_shapes = [(), (1,), (2,), (1,1), (1,3), (2,3), (2,1)]

        for SHAPE in shapes:
            for STACK_SHAPE in stack_shapes:
                for USE_JAX in [True, False]:
                    with self.subTest(SHAPE=SHAPE, STACK_SHAPE=STACK_SHAPE):
                        x = t3.TuckerTensorTrain.ones(SHAPE, stack_shape=STACK_SHAPE, use_jax=USE_JAX)

                        self.assertEqual(SHAPE, x.shape)
                        self.assertEqual((1,)*len(SHAPE), x.tucker_ranks)
                        self.assertEqual((1,)*(len(SHAPE)+1), x.tt_ranks)
                        self.assertEqual(STACK_SHAPE, x.stack_shape)
                        self.check_relerr(np.ones(STACK_SHAPE+SHAPE), x.to_dense())

    def test_corewise_randn(self):
        structures = [
            ((14,), (4,), (4, 5), (2, 3)),
            ((14, 15), (4, 5), (4, 5, 4), (2, 3)),
            ((14, 15, 16, 17), (4, 5, 6, 7), (4, 5, 4, 3, 2), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), ()),
        ]

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                with self.subTest(STRUCTURE=STRUCTURE, USE_JAX=USE_JAX):
                    shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                    x = t3.TuckerTensorTrain.corewise_randn(
                        shape, tucker_ranks, tt_ranks, stack_shape=stack_shape, use_jax=USE_JAX,
                    )

                    self.assertEqual(shape, x.shape)
                    self.assertEqual(tucker_ranks, x.tucker_ranks)
                    self.assertEqual(tt_ranks, x.tt_ranks)
                    self.assertEqual(stack_shape, x.stack_shape)

                    # Unclear how to check that the entries are indeed random...

    def test_to_vector_and_from_vector(self):
        structures = [
            ((14,), (4,), (4, 5), (2, 3)),
            ((14, 15), (4, 5), (4, 5, 4), (2, 3)),
            ((14, 15, 16, 17), (4, 5, 6, 7), (4, 5, 4, 3, 2), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), ()),
        ]

        for STRUCTURE in structures:
            with self.subTest(STRUCTURE=STRUCTURE):
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                x = t3.TuckerTensorTrain.corewise_randn(
                    shape, tucker_ranks, tt_ranks, stack_shape=stack_shape,
                )

                x_flat = x.to_vector()
                self.assertEqual(1, len(x_flat.shape))

                x2 = t3.TuckerTensorTrain.from_vector(x_flat, x.shape, x.tucker_ranks, x.tt_ranks, stack_shape=x.stack_shape)

                self.assertLessEqual(
                    cw.corewise_norm(cw.corewise_sub(x.data, x2.data)),
                    tol * cw.corewise_norm(x.data)
                )

    def test_t3_save_and_t3_load(self):
        structures = [
            ((14,), (4,), (4, 5), (2, 3)),
            ((14, 15), (4, 5), (4, 5, 4), (2, 3)),
            ((14, 15, 16, 17), (4, 5, 6, 7), (4, 5, 4, 3, 2), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), ()),
        ]

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                with self.subTest(USE_JAX=USE_JAX, STRUCTURE=STRUCTURE):
                    x = t3.TuckerTensorTrain.corewise_randn(*STRUCTURE, use_jax=USE_JAX)

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

                    x.save(fname)  # Save to file
                    x2 = t3.TuckerTensorTrain.load(fname, use_jax=USE_JAX)  # Load from file

                    os.remove(fname)

                    tucker_cores, tt_cores = x.data
                    tucker_cores2, tt_cores2 = x2.data

                    for B, B2 in zip(tucker_cores, tucker_cores2):
                        self.check_relerr(B, B2)

                    for G, G2 in zip(tt_cores, tt_cores2):
                        self.check_relerr(G, G2)

    def test_dunder_mul(self):
        structures = [
            ((14,), (4,), (4, 5), (2, 3)),
            ((14, 15), (4, 5), (4, 5, 4), (2, 3)),
            ((14, 15, 16, 17), (4, 5, 6, 7), (4, 5, 4, 3, 2), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), ()),
        ]

        other_ranks = [
            ((3,), (2, 6)),
            ((4, 2), (4, 1, 3)),
            ((1, 2, 3, 4), (1, 3, 2, 1, 2)),
            ((5, 5, 5), (2, 2, 2, 2)),
        ]

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                x = t3.TuckerTensorTrain.corewise_randn(*STRUCTURE, use_jax=USE_JAX)
                with self.subTest(USE_JAX=USE_JAX, STRUCTURE=STRUCTURE, OP='T3_TIMES_SCALAR'):
                    s = 3.2

                    sx = x * s

                    self.assertIsInstance(sx, t3.TuckerTensorTrain)
                    self.check_relerr(s*x.to_dense(), sx.to_dense())

                with self.subTest(USE_JAX=USE_JAX, STRUCTURE=STRUCTURE, OP='T3_TIMES_ARRAYSCALAR'):
                    s = np.array(3.2)

                    sx = x * s

                    self.assertIsInstance(sx, t3.TuckerTensorTrain)
                    self.check_relerr(s * x.to_dense(), sx.to_dense())

                with self.subTest(USE_JAX=USE_JAX, STRUCTURE=STRUCTURE, OP='T3_TIMES_DENSE'):
                    y = np.random.randn(*(x.stack_shape + x.shape))

                    xy = x * y

                    self.assertTrue(is_ndarray(xy))
                    self.check_relerr(x.to_dense()*y, xy)

        for STRUCTURE, OTHER_RANKS in zip(structures, other_ranks):
            for USE_JAX in [True, False]:
                x = t3.TuckerTensorTrain.corewise_randn(*STRUCTURE, use_jax=USE_JAX)
                for SQUASH in [True, False]:
                    with self.subTest(
                            USE_JAX=USE_JAX, STRUCTURE=STRUCTURE, OTHER_RANKS=OTHER_RANKS,
                            SQUASH=SQUASH, OP='T3_TIMES_T3',
                    ):
                        y_structure = STRUCTURE[:1] + OTHER_RANKS + STRUCTURE[3:]
                        y = t3.TuckerTensorTrain.corewise_randn(*y_structure)

                        xy = x * y

                        self.assertIsInstance(xy, t3.TuckerTensorTrain)
                        self.check_relerr(x.to_dense()*y.to_dense(), xy.to_dense())

                        prod_tucker_ranks   = tuple(nx*ny for nx, ny in zip(STRUCTURE[1], OTHER_RANKS[0]))
                        prod_tt_ranks       = tuple(rx*ry for rx, ry in zip(STRUCTURE[2], OTHER_RANKS[1]))
                        self.assertEqual(prod_tucker_ranks, xy.tucker_ranks)
                        self.assertEqual(prod_tt_ranks, xy.tt_ranks)


#####

    def test_t3_apply1(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)),
        ]

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                with self.subTest(USE_JAX=USE_JAX, STRUCTURE=STRUCTURE):
                    x = t3.TuckerTensorTrain.corewise_randn(STRUCTURE)

                    SHAPE = STRUCTURE[0]
                    vecs = [np.random.randn(SHAPE[0]),
                            np.random.randn(SHAPE[1]),
                            np.random.randn(SHAPE[2])]

                    result = t3.t3_apply(x, vecs, use_jax=USE_JAX)  # <-- contract x with vecs in all indices

                    result2 = np.einsum('ijk,i,j,k', t3.t3_to_dense(x), vecs[0], vecs[1], vecs[2])
                    self.check_relerr(result2, result)

    def test_t3_apply2(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (4, 3, 2, 7)),
        ]
        NUM_PROBES = 3

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                with self.subTest(USE_JAX=USE_JAX, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)

                    SHAPE = STRUCTURE[0]
                    vecs = [np.random.randn(NUM_PROBES, SHAPE[0]),
                            np.random.randn(NUM_PROBES, SHAPE[1]),
                            np.random.randn(NUM_PROBES, SHAPE[2])]

                    result = t3.t3_apply(x, vecs, use_jax=USE_JAX)

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
                for USE_JAX in [True, False]:
                    with self.subTest(USE_JAX=USE_JAX, STRUCTURE=STRUCTURE, INDEX=INDEX):
                        x = t3.t3_corewise_randn(STRUCTURE)

                        result = t3.t3_entry(x, INDEX, use_jax=USE_JAX)

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
                for USE_JAX in [True, False]:
                    with self.subTest(USE_JAX=USE_JAX, STRUCTURE=STRUCTURE, INDEX_SET=INDEX_SET):
                        x = t3.t3_corewise_randn(STRUCTURE)

                        entries = t3.t3_entry(x, INDEX_SET, use_jax=USE_JAX)

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
        mr = t3.compute_minimal_ranks(((10, 11, 12, 13), (14, 15, 16, 17), (98, 99, 100, 101, 102)))

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
            with self.subTest(STRUCTURE=STRUCTURE, RESULT=RESULT):
                x = t3.t3_corewise_randn(STRUCTURE)
                self.assertEqual(RESULT, t3.are_t3_ranks_minimal(x))

    def test_pad_t3(self):
        structures1 = [
            ((14, 15, 16), (4, 6, 5), (3, 3, 2, 4)),
        ]
        structures2 = [
            ((17, 18, 17), (8, 8, 8), (5, 5, 6, 7)),
        ]

        for STRUCTURE1, STRUCTURE2 in zip(structures1, structures2):
            for USE_JAX in [True, False]:
                with self.subTest(USE_JAX=USE_JAX, STRUCTURE1=STRUCTURE1, STRUCTURE2=STRUCTURE2):
                    x = t3.t3_corewise_randn(STRUCTURE1)
                    padded_x = t3.change_structure(x, STRUCTURE2, use_jax=USE_JAX)
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
                for USE_JAX in [True, False]:
                    for OP in [t3.t3_add, t3.t3_sub]:
                        with self.subTest(SQUASH=SQUASH, USE_JAX=USE_JAX, STRUCTURE_X=STRUCTURE_X, STRUCTURE_Y=STRUCTURE_Y, OP=OP):
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
                            if OP is t3.t3_add:
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
            for OP in [t3.t3_scale, t3.t3_neg]:
                with self.subTest(STRUCTURE=STRUCTURE, OP=OP):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    dense_x = t3.t3_to_dense(x)
                    if OP is t3.t3_scale:
                        y = t3.t3_scale(x, SCALE_FACTOR) # <--
                        Y_true = SCALE_FACTOR * dense_x
                    else:
                        y = t3.t3_neg(x) # <--
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
            for USE_JAX in [True, False]:
                with self.subTest(USE_JAX=USE_JAX, STRUCTURE_X=STRUCTURE_X, STRUCTURE_Y=STRUCTURE_Y):
                    x = t3.t3_corewise_randn(STRUCTURE_X)
                    y = t3.t3_corewise_randn(STRUCTURE_Y)
                    x_dot_y = t3.t3_inner_product_t3(x, y, use_jax=USE_JAX)
                    x_dot_y2 = np.sum(t3.t3_to_dense(x) * t3.t3_to_dense(y))
                    self.check_relerr(x_dot_y2, x_dot_y)

    def test_t3_norm(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (2, 3, 2, 1)),
        ]

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                with self.subTest(USE_JAX=USE_JAX, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    norm_x = t3.t3_norm(x, use_jax=USE_JAX)
                    norm_x2 = np.linalg.norm(t3.t3_to_dense(x))
                    self.check_relerr(norm_x2, norm_x)


if __name__ == '__main__':
    unittest.main()