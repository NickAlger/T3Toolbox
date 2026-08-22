# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
import numpy as np
import unittest
import os
import itertools
import math

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.corewise as cw
import t3toolbox.backend.common as common
import t3toolbox.backend.sampling_derivatives as sampling_derivatives

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


def _structure_to_cores(STRUCTURE):
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


def _td(z):
    if isinstance(z, t3.TuckerTensorTrain):
        return z.to_dense()
    return z

def _random_preconditioned_t3(shape, tucker_ranks, tt_ranks, stack_shape=()):
    x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks, stack_shape)
    cc_s = tuple(1.0 / (1.0 + np.arange(s))**2 for s in shape)
    cc_tk = tuple(np.ones(n) for n in tucker_ranks)
    cc_tt = tuple(1.0 / (1.0 + np.arange(r))**2 for r in tt_ranks)
    tucker_cores2 = tuple(
        np.einsum('...io,o->...io', B / np.linalg.norm(B), c)
        for B, c in zip(x.tucker_cores, cc_s)
    )
    tt_cores2 = tuple(
        np.einsum('...aib,a,i,b->...aib', G / np.linalg.norm(G), cl, cm, cr)
        for G, cl, cm, cr in zip(x.tt_cores, cc_tt[:-1], cc_tk, cc_tt[1:])
    )
    x = t3.TuckerTensorTrain(tucker_cores2, tt_cores2)  # random preconditioned T3
    return x


class TestTuckerTensorTrain(unittest.TestCase):
    def check_relerr(self, xtrue, x):
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def test_repr(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        self.assertEqual(
            "TuckerTensorTrain(shape=(5, 6, 4), tucker_ranks=(2, 3, 2), tt_ranks=(1, 2, 2, 1))", repr(x))
        xs = t3.TuckerTensorTrain.randn((5, 6, 4), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
        self.assertIn("stack_shape=(2,)", repr(xs))
        self.assertNotIn("array", repr(x))

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
            ((14,),             (4,),           (4, 5),             (2, 3)),
            ((14, 15),          (4, 5),         (4, 5, 3),          (2, 3)),
            ((14, 15, 16, 17),  (4, 5, 6, 7),   (4, 5, 3, 2, 1),    (2, 3)),
            ((14, 15, 16),      (4, 5, 6),      (4, 5, 3, 2),       ()),
        ]
        for STRUCTURE in all_structures:
            with self.subTest(STRUCTURE=STRUCTURE):
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                tucker_cores, tt_cores = _structure_to_cores(STRUCTURE)

                print([x.shape for x in tucker_cores])
                print([x.shape for x in tt_cores])

                x = t3.TuckerTensorTrain(tucker_cores, tt_cores)  # random TuckerTensorTrain

                self.assertEqual((tucker_cores, tt_cores), x.data)
                self.assertEqual(len(shape),    x.d)
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
                self.assertEqual(math.prod(shape), x.size)
                self.assertEqual(
                    sum(x.size for x in tucker_cores) + sum(x.size for x in tt_cores),
                    x.data_size,
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
                tucker_cores, tt_cores = _structure_to_cores(STRUCTURE)
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

    def test_has_numerically_minimal_ranks(self):
        np.random.seed(0)
        # a full-rank random tensor is both structurally and numerically minimal
        x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        self.assertTrue(x.has_minimal_ranks)
        self.assertTrue(x.has_numerically_minimal_ranks())
        # padding a Tucker rank makes it structurally non-minimal -> the structural short-circuit -> False
        xbig = x.resize((6, 7, 5), (3, 2, 2), (1, 2, 2, 1))
        self.assertFalse(xbig.has_minimal_ranks)
        self.assertFalse(xbig.has_numerically_minimal_ranks())
        # a t3svd'd (no-truncation) tensor is numerically minimal; its ranks survive a re-svd
        x2 = xbig.t3svd()[0]
        self.assertTrue(x2.has_minimal_ranks)
        self.assertTrue(x2.has_numerically_minimal_ranks())

    def test_continuation_ranks(self):
        import t3toolbox.backend.ranks as branks
        np.random.seed(0)
        # the frontend method = compute_continuation_ranks on this T3's own t3svd singular values
        x = t3.TuckerTensorTrain.randn((8, 9, 7), (2, 2, 2), (1, 2, 2, 1))
        new_tucker, new_tt = x.continuation_ranks()
        _, ss_tucker, ss_tt = x.t3svd()
        self.assertEqual((new_tucker, new_tt),
                         branks.compute_continuation_ranks(x.shape, ss_tucker, ss_tt))
        # the proposed ranks are structurally valid: a zero-padded warm start preserves the tensor
        x0 = x.resize(x.shape, new_tucker, new_tt)
        self.assertEqual((x0.tucker_ranks, x0.tt_ranks), (new_tucker, new_tt))
        self.assertTrue(np.allclose(x0.to_dense(), x.to_dense()))
        # params (tau, n_chunk, kappa_guard, max_grow) thread through verbatim to the backend
        self.assertEqual(x.continuation_ranks(tau=1.5, n_chunk=2, max_grow=1),
                         branks.compute_continuation_ranks(x.shape, ss_tucker, ss_tt,
                                                           tau=1.5, n_chunk=2, max_grow=1))
        # defined for a single T3 only: a stacked T3 is a structural error
        xs = t3.TuckerTensorTrain.randn((8, 9, 7), (2, 2, 2), (1, 2, 2, 1), stack_shape=(3,))
        with self.assertRaises(ValueError):
            xs.continuation_ranks()

    def test_to_dense(self):
        structures = [
            ((8, 9, 7), (3, 4, 5), (2, 3, 7, 5), (2, 3)),
            ((8, 9, 7), (3, 4, 5), (2, 3, 7, 5), ()), # no stacking
            ((8, 9, 7), (3, 4, 5), (1, 3, 7, 1), (2,3)), # no tails to squash
        ]

        for STRUCTURE in structures:
            for SQUASH_TAILS in [True, False]:
                with self.subTest(STRUCTURE=STRUCTURE, SQUASH_TAILS=SQUASH_TAILS):
                    shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                    tucker_cores, tt_cores = _structure_to_cores(STRUCTURE)
                    x = t3.TuckerTensorTrain(tucker_cores, tt_cores)  # random TuckerTensorTrain

                    x_dense = x.to_dense(squash_tails=SQUASH_TAILS)
                    if common.jax_available:
                        self.assertEqual(False, common.is_jax_ndarray(x_dense))

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

    def test_squash(self):
        structures = [
            ((8,),      (3,),       (2, 5),         ()),
            ((8,),      (3,),       (2, 5),         (2, 3)),
            ((8, 9, 7), (3, 4, 5),  (2, 3, 7, 5),   (2, 3)),
            ((8, 9, 7), (3, 4, 5),  (2, 3, 7, 5),   ()), # no stacking
            ((8, 9, 7), (3, 4, 5),  (1, 3, 7, 1),   (2, 3)), # no tails to squash
        ]

        for STRUCTURE in structures:
            with self.subTest(STRUCTURE=STRUCTURE):
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                tucker_cores, tt_cores = _structure_to_cores(STRUCTURE)
                x = t3.TuckerTensorTrain(tucker_cores, tt_cores)  # random TuckerTensorTrain

                x2 = x.squash_tails()

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
                tucker_cores, tt_cores = _structure_to_cores(STRUCTURE)
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

    def test_resize(self):
        structures = [
            ((14,),             (4,),           (4, 5),             (2, 3)),
            ((14, 15),          (4, 5),         (4, 5, 4),          (2, 3)),
            ((14, 15, 16),      (4, 5, 6),      (4, 5, 4, 3),       (2, 3)),
            ((14, 15, 16, 17),  (4, 5, 6, 7),   (4, 5, 4, 3, 2),    (2, 3)),
            ((14, 15, 16),      (4, 5, 6),      (4, 5, 4, 3),       ()),
        ]

        for STRUCTURE in structures:
            shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
            x = t3.TuckerTensorTrain.randn(*STRUCTURE, use_jax=False)
            dense_x = x.to_dense()

            with self.subTest(STRUCTURE=STRUCTURE, OP='DO_NOTHING'):

                x2 = x.resize(shape, tucker_ranks, tt_ranks)

                self.check_relerr(dense_x, x2.to_dense())

            with self.subTest(STRUCTURE=STRUCTURE, OP='INCREASE_SHAPE'):
                new_shape = tuple(s + 3 for s in shape)

                x2 = x.resize(new_shape, tucker_ranks, tt_ranks)

                self.assertEqual(new_shape, x2.shape)
                self.assertEqual(tucker_ranks, x2.tucker_ranks)
                self.assertEqual(tt_ranks, x2.tt_ranks)
                self.assertEqual(stack_shape, x2.stack_shape)

                dense_x2 = x2.to_dense()
                pad = [(0,0) for _ in range(len(stack_shape))]
                pad = pad + [(0, ns - s) for ns, s in zip(new_shape, shape)]
                padded_dense_x = np.pad(dense_x, pad)
                self.check_relerr(padded_dense_x, dense_x2)

            with self.subTest(STRUCTURE=STRUCTURE, OP='INCREASE_TUCKER_RANKS'):
                new_tucker_ranks = tuple(r + 3 for r in tucker_ranks)

                x2 = x.resize(shape, new_tucker_ranks, tt_ranks)

                self.assertEqual(shape, x2.shape)
                self.assertEqual(new_tucker_ranks, x2.tucker_ranks)
                self.assertEqual(tt_ranks, x2.tt_ranks)
                self.assertEqual(stack_shape, x2.stack_shape)

                dense_x2 = x2.to_dense()
                self.check_relerr(dense_x, dense_x2)

            with self.subTest(STRUCTURE=STRUCTURE, OP='INCREASE_TT_RANKS'):
                new_tt_ranks = tuple(n + 3 for n in tt_ranks)

                x2 = x.resize(shape, tucker_ranks, new_tt_ranks)

                self.assertEqual(shape, x2.shape)
                self.assertEqual(tucker_ranks, x2.tucker_ranks)
                self.assertEqual(new_tt_ranks, x2.tt_ranks)
                self.assertEqual(stack_shape, x2.stack_shape)

                dense_x2 = x2.to_dense()
                self.check_relerr(dense_x, dense_x2)

            with self.subTest(STRUCTURE=STRUCTURE, OP='TRUNCATE_SHAPE'):
                new_shape = tuple(s - 1 for s in shape)

                x2 = x.resize(new_shape, tucker_ranks, tt_ranks)

                self.assertEqual(new_shape, x2.shape)
                self.assertEqual(tucker_ranks, x2.tucker_ranks)
                self.assertEqual(tt_ranks, x2.tt_ranks)
                self.assertEqual(stack_shape, x2.stack_shape)

                for B, B2, N in zip(x.tucker_cores, x2.tucker_cores, new_shape):
                    B = np.moveaxis(np.moveaxis(B, -1,0)[:N], 0, -1)
                    self.check_relerr(B, B2)

            with self.subTest(STRUCTURE=STRUCTURE, OP='TRUNCATE_TUCKER_RANKS'):
                new_tucker_ranks = tuple(n - 1 for n in tucker_ranks)

                x2 = x.resize(shape, new_tucker_ranks, tt_ranks)

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

            with self.subTest(STRUCTURE=STRUCTURE, OP='TRUNCATE_TT_RANKS'):
                new_tt_ranks = tuple(r - 1 for r in tt_ranks)

                x2 = x.resize(shape, tucker_ranks, new_tt_ranks)

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

            tucker_cores, tt_cores = _structure_to_cores((shape, tucker_ranks, tt_ranks, stack_shape))
            x = t3.TuckerTensorTrain(tucker_cores, tt_cores)

            x2 = x.resize(new_shape, new_tucker_ranks, new_tt_ranks)

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

    def test_to_jax(self):
        structures = [
            ((14,),             (4,),           (4, 5),             (2, 3)),
            ((14, 15),          (4, 5),         (4, 5, 4),          (2, 3)),
            ((14, 15, 16, 17),  (4, 5, 6, 7),   (4, 5, 4, 3, 2),    (2, 3)),
            ((14, 15, 16),      (4, 5, 6),      (4, 5, 4, 3),       ()),
        ]

        for STRUCTURE in structures:
            with self.subTest(STRUCTURE=STRUCTURE):
                x = t3.TuckerTensorTrain.randn(*STRUCTURE, use_jax=False)
                x_jax = x.to_jax()

                self.assertLessEqual(
                    cw.corewise_norm(cw.corewise_sub(x.data, x_jax.data)),
                    tol * cw.corewise_norm(x.data)
                )

                if common.jax_available:
                    for B in x_jax.tucker_cores:
                        self.assertTrue(common.is_jax_ndarray(B))
                    for G in x_jax.tt_cores:
                        self.assertTrue(common.is_jax_ndarray(G))

    def test_to_numpy(self):
        structures = [
            ((14,), (4,), (4, 5), (2, 3)),
            ((14, 15), (4, 5), (4, 5, 4), (2, 3)),
            ((14, 15, 16, 17), (4, 5, 6, 7), (4, 5, 4, 3, 2), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), ()),
        ]

        for STRUCTURE in structures:
            with self.subTest(STRUCTURE=STRUCTURE):
                x = t3.TuckerTensorTrain.randn(*STRUCTURE, use_jax=True)
                x_numpy = x.to_numpy()

                self.assertLessEqual(
                    cw.corewise_norm(cw.corewise_sub(x.data, x_numpy.data)),
                    tol * cw.corewise_norm(x.data)
                )

                for B in x_numpy.tucker_cores:
                    self.assertTrue(common.is_numpy_ndarray(B))
                for G in x_numpy.tt_cores:
                    self.assertTrue(common.is_numpy_ndarray(G))

    def test_contains_jax(self):
        structure = (14, 15, 16), (4, 5, 6), (4, 5, 4, 3), (2,3)
        tucker_cores0, tt_cores0 = _structure_to_cores(structure)

        all_tf_combos = [
            [True, True, True],
            [True, True, False],
            [True, False, True],
            [True, False, False],
            [False, True, True],
            [False, True, False],
            [False, False, True],
            [False, False, False],
        ]
        for TUCKER_JAX_INDS in all_tf_combos:
            for TT_JAX_INDS in all_tf_combos:
                with self.subTest(TUCKER_JAX_INDS=TUCKER_JAX_INDS, TT_JAX_INDS=TT_JAX_INDS):
                    tucker_cores = [B.copy() for B in tucker_cores0]
                    for ii in range(len(tucker_cores)):
                        if TUCKER_JAX_INDS[ii]:
                            tucker_cores[ii] = jnp.array(tucker_cores[ii])
                        else:
                            tucker_cores[ii] = np.array(tucker_cores[ii])

                    tt_cores = [G.copy() for G in tt_cores0]
                    for ii in range(len(tt_cores)):
                        if TT_JAX_INDS[ii]:
                            tt_cores[ii] = jnp.array(tt_cores[ii])
                        else:
                            tt_cores[ii] = np.array(tt_cores[ii])

                    x = t3.TuckerTensorTrain(tuple(tucker_cores), tuple(tt_cores))

                    if common.jax_available:
                        true_contains_jax = any(TUCKER_JAX_INDS) or any(TT_JAX_INDS)
                        self.assertEqual(true_contains_jax, x.contains_jax)
                    else:
                        self.assertEqual(False, x.contains_jax)

    def test_copy(self):
        structures = [
            ((14,), (4,), (4, 5), (2, 3)),
            ((14, 15), (4, 5), (4, 5, 4), (2, 3)),
            ((14, 15, 16, 17), (4, 5, 6, 7), (4, 5, 4, 3, 2), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), ()),
        ]

        for STRUCTURE in structures:
            with self.subTest(STRUCTURE=STRUCTURE):
                x = t3.TuckerTensorTrain.randn(*STRUCTURE)
                x2 = x.copy()

                self.assertLessEqual(
                    cw.corewise_norm(cw.corewise_sub(x.data, x2.data)),
                    tol * cw.corewise_norm(x.data)
                )

    def test_unstack(self):
        frame_structures = [
            ((14,),             (4,),           (4, 5)),
            ((14, 15),          (4, 5),         (4, 5, 4)),
            ((14, 15, 16, 17),  (4, 5, 6, 7),   (4, 5, 4, 3, 2)),
        ]
        stack_shapes = [(), (1,), (2,), (1,1), (1,3), (2,3), (2,1)]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                with self.subTest(FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    shape, tucker_ranks, tt_ranks = FRAME_STRUCTURE
                    structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                    tucker_cores, tt_cores = _structure_to_cores(structure)
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
        frame_structures = [
            ((14,),             (4,),           (4, 5)),
            ((14, 15),          (4, 5),         (4, 5, 4)),
            ((14, 15, 16, 17),  (4, 5, 6, 7),   (4, 5, 4, 3, 2)),
        ]
        stack_shapes = [(), (1,), (2,), (1,1), (1,3), (2,3), (2,1)]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                with self.subTest(FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    shape, tucker_ranks, tt_ranks = FRAME_STRUCTURE
                    structure = FRAME_STRUCTURE + ((),)

                    if len(STACK_SHAPE) == 0:
                        tucker_cores, tt_cores = _structure_to_cores(structure)
                        xx = t3.TuckerTensorTrain(tucker_cores, tt_cores)
                        xx_dense = xx.to_dense()

                    if len(STACK_SHAPE) == 1:
                        xx = []
                        xx_dense = []
                        for ii in range(STACK_SHAPE[0]):
                            tucker_cores, tt_cores = _structure_to_cores(structure)
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
                                tucker_cores, tt_cores = _structure_to_cores(structure)
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
            shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
            for TUCKER_RANKS in [tucker_ranks, None]:
                for TT_RANKS in [tt_ranks, None]:
                    with self.subTest(STRUCTURE=STRUCTURE, TUCKER_RANKS=TUCKER_RANKS, TT_RANKS=TT_RANKS):
                        if TUCKER_RANKS is None and TT_RANKS is None:
                            x = t3.TuckerTensorTrain.zeros(
                                shape, stack_shape=stack_shape, use_jax=False,
                            )
                            self.assertEqual((1,)*len(shape), x.tucker_ranks)
                            self.assertEqual((1,)*(len(shape)+1), x.tt_ranks)
                        elif TUCKER_RANKS is None:
                            x = t3.TuckerTensorTrain.zeros(
                                shape, tt_ranks=tt_ranks, stack_shape=stack_shape,
                                use_jax=False,
                            )
                            self.assertEqual((1,)*len(shape), x.tucker_ranks)
                            self.assertEqual(tt_ranks, x.tt_ranks)
                        elif TT_RANKS is None:
                            x = t3.TuckerTensorTrain.zeros(
                                shape, tucker_ranks=tucker_ranks, stack_shape=stack_shape,
                                use_jax=False,
                            )
                            self.assertEqual(tucker_ranks, x.tucker_ranks)
                            self.assertEqual((1,)*(len(shape)+1), x.tt_ranks)
                        else:
                            x = t3.TuckerTensorTrain.zeros(
                                shape, tucker_ranks=tucker_ranks, tt_ranks=tt_ranks, stack_shape=stack_shape,
                                use_jax=False,
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
                with self.subTest(SHAPE=SHAPE, STACK_SHAPE=STACK_SHAPE):
                    x = t3.TuckerTensorTrain.ones(SHAPE, stack_shape=STACK_SHAPE, use_jax=False)

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
            with self.subTest(STRUCTURE=STRUCTURE):
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                x = t3.TuckerTensorTrain.randn(
                    shape, tucker_ranks, tt_ranks, stack_shape=stack_shape, use_jax=False,
                )

                self.assertEqual(shape, x.shape)
                self.assertEqual(tucker_ranks, x.tucker_ranks)
                self.assertEqual(tt_ranks, x.tt_ranks)
                self.assertEqual(stack_shape, x.stack_shape)

                # Unclear how to check that the entries are indeed random...

    def test_from_canonical(self):
        shapes = [
            (14,),
            (14, 15),
            (14, 15, 16),
            (14, 15, 16, 17),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]
        ranks = [1,3,6] # canonical rank

        for SHAPE in shapes:
            for STACK_SHAPE in stack_shapes:
                for RANK in ranks:
                    with self.subTest(
                            SHAPE=SHAPE, STACK_SHAPE=STACK_SHAPE,
                            RANK=RANK
                    ):
                        FF = [np.random.randn(*(STACK_SHAPE+(RANK, N))) for N in SHAPE]

                        x = t3.TuckerTensorTrain.from_canonical(FF)
                        x_dense = x.to_dense()

                        if len(SHAPE) == 1:
                            x_dense2 = np.einsum('...ri->...i', FF[0])
                        elif len(SHAPE) == 2:
                            x_dense2 = np.einsum('...ri,...rj->...ij', FF[0], FF[1])
                        elif len(SHAPE) == 3:
                            x_dense2 = np.einsum('...ri,...rj,...rk->...ijk', FF[0], FF[1], FF[2])
                        elif len(SHAPE) == 4:
                            x_dense2 = np.einsum('...ri,...rj,...rk,...rl->...ijkl', FF[0], FF[1], FF[2], FF[3])
                        else:
                            raise ValueError

                        self.check_relerr(x_dense2, x_dense)

                        self.assertEqual((RANK,)*len(SHAPE), x.tucker_ranks)
                        self.assertEqual((RANK,)*(len(SHAPE)+1), x.tt_ranks)

    def test_from_tensor_train(self):
        tt_structures = [
            ((14,),             (4, 5)),
            ((14, 15),          (4, 5, 4)),
            ((14, 15, 16),      (4, 5, 4, 3)),
            ((14, 15, 16, 17),  (4, 5, 4, 3, 2)),
        ]
        stack_shapes = [
            (),
            (2,3),
        ]

        for TT_STRUCTURE in tt_structures:
            for STACK_SHAPE in stack_shapes:
                with self.subTest(
                        TT_STRUCTURE=TT_STRUCTURE, STACK_SHAPE=STACK_SHAPE
                ):
                    shape, tt_ranks = TT_STRUCTURE
                    tt_cores = tuple(
                        np.random.randn(*(STACK_SHAPE + (rL, n, rR)))
                        for rL, n, rR in zip(tt_ranks[:-1], shape, tt_ranks[1:])
                    )

                    x = t3.TuckerTensorTrain.from_tensor_train(tt_cores)

                    self.assertEqual(tt_ranks, x.tt_ranks)
                    self.assertEqual(shape, x.tucker_ranks)
                    self.assertEqual(shape, x.shape)

                    x_dense = x.to_dense()
                    if len(shape) == 1:
                        x_dense2 = np.einsum('...aib->...i', *tt_cores)
                    elif len(shape) == 2:
                        x_dense2 = np.einsum('...aib,...bjc->...ij', *tt_cores)
                    elif len(shape) == 3:
                        x_dense2 = np.einsum('...aib,...bjc,...ckd->...ijk', *tt_cores)
                    elif len(shape) == 4:
                        x_dense2 = np.einsum('...aib,...bjc,...ckd,...dle->...ijkl', *tt_cores)
                    else:
                        raise ValueError

                    self.check_relerr(x_dense2, x_dense)

    def test_to_tensor_train(self):
        structures = [
            ((14,), (4,), (4, 5), (2, 3)),
            ((14, 15), (4, 5), (4, 5, 4), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), (2,3)),
            ((14, 15, 16, 17), (4, 5, 6, 7), (4, 5, 4, 3, 2), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), ()),
        ]

        for STRUCTURE in structures:
            with self.subTest(STRUCTURE=STRUCTURE):
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                x = t3.TuckerTensorTrain.randn(
                    shape, tucker_ranks, tt_ranks, stack_shape=stack_shape, use_jax=False,
                )
                big_tt_cores = x.to_tensor_train()

                if len(shape) == 1:
                    x_dense = np.einsum('...aib->...i', *big_tt_cores)
                elif len(shape) == 2:
                    x_dense = np.einsum('...aib,...bjc->...ij', *big_tt_cores)
                elif len(shape) == 3:
                    x_dense = np.einsum('...aib,...bjc,...ckd->...ijk', *big_tt_cores)
                elif len(shape) == 4:
                    x_dense = np.einsum('...aib,...bjc,...ckd,...dle->...ijkl', *big_tt_cores)
                else:
                    raise ValueError

                x_dense2 = x.to_dense()
                self.check_relerr(x_dense2, x_dense)

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
                x = t3.TuckerTensorTrain.randn(
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
            with self.subTest(STRUCTURE=STRUCTURE):
                x = t3.TuckerTensorTrain.randn(*STRUCTURE, use_jax=False)

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
                x2 = t3.TuckerTensorTrain.load(fname, use_jax=False)  # Load from file

                os.remove(fname)

                tucker_cores, tt_cores = x.data
                tucker_cores2, tt_cores2 = x2.data

                for B, B2 in zip(tucker_cores, tucker_cores2):
                    self.check_relerr(B, B2)

                for G, G2 in zip(tt_cores, tt_cores2):
                    self.check_relerr(G, G2)

    def test_dunder_neg(self):
        structures = [
            ((14,), (4,), (4, 5), (2, 3)),
            ((14, 15), (4, 5), (4, 5, 4), (2, 3)),
            ((14, 15, 16, 17), (4, 5, 6, 7), (4, 5, 4, 3, 2), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), ()),
        ]

        for STRUCTURE in structures:
            with self.subTest(STRUCTURE=STRUCTURE):
                x = t3.TuckerTensorTrain.randn(*STRUCTURE, use_jax=False)

                neg_x = -x

                self.assertIsInstance(neg_x, t3.TuckerTensorTrain)
                self.check_relerr(-x.to_dense(), neg_x.to_dense())

    def test_dunder_add_sub_mul(self):
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

        for STRUCTURE, OTHER_RANKS in zip(structures, other_ranks):
            for OP in ['PLUS', 'MINUS', 'MUL']:
                x = t3.TuckerTensorTrain.randn(*STRUCTURE, use_jax=False)
                for OTHER_TYPE in [
                    'SCALAR', 'NUMPY_SCALAR', 'JAX_SCALAR',
                    'NUMPY_DENSE', 'JAX_DENSE',
                    'NUMPY_T3', 'JAX_T3',
                ]:
                    with self.subTest(
                            STRUCTURE=STRUCTURE, OTHER_RANKS=OTHER_RANKS,
                            OP=OP, OTHER_TYPE=OTHER_TYPE):
                        if OTHER_TYPE == 'SCALAR':
                            y = 3.2

                        elif OTHER_TYPE == 'NUMPY_SCALAR':
                            y = np.array(3.2)

                        elif OTHER_TYPE == 'JAX_SCALAR':
                            y = jnp.array(3.2)

                        elif OTHER_TYPE == 'NUMPY_DENSE':
                            y = np.random.randn(*(x.stack_shape + x.shape))

                        elif OTHER_TYPE == 'JAX_DENSE':
                            y = jnp.array(np.random.randn(*(x.stack_shape + x.shape)))

                        elif OTHER_TYPE == 'NUMPY_T3':
                            y_structure = STRUCTURE[:1] + OTHER_RANKS + STRUCTURE[3:]
                            y = t3.TuckerTensorTrain.randn(*y_structure, use_jax=False)

                        elif OTHER_TYPE == 'JAX_T3':
                            y_structure = STRUCTURE[:1] + OTHER_RANKS + STRUCTURE[3:]
                            y = t3.TuckerTensorTrain.randn(*y_structure, use_jax=True)

                        else:
                            print('OTHER_TYPE=', OTHER_TYPE)
                            raise ValueError


                        if OP == 'PLUS':
                            x_op_y = x + y
                            self.check_relerr(_td(x) + _td(y), _td(x_op_y))

                        elif OP == 'MINUS':
                            x_op_y = x - y
                            self.check_relerr(_td(x) - _td(y), _td(x_op_y))

                        elif OP == 'MUL':
                            x_op_y = x * y
                            self.check_relerr(_td(x) * _td(y), _td(x_op_y))

                        else:
                            print('OP=', OP)
                            raise ValueError


                        if OTHER_TYPE == 'NUMPY_T3' or OTHER_TYPE == 'JAX_T3':
                            if OP == 'PLUS' or OP == 'MINUS':
                                sum_tucker_ranks = tuple(nx + ny for nx, ny in zip(STRUCTURE[1], OTHER_RANKS[0]))
                                sum_tt_ranks = tuple(rx + ry for rx, ry in zip(STRUCTURE[2], OTHER_RANKS[1]))
                                self.assertEqual(sum_tucker_ranks, x_op_y.tucker_ranks)
                                self.assertEqual(sum_tt_ranks, x_op_y.tt_ranks)

                            elif OP == 'MUL':
                                prod_tucker_ranks = tuple(nx * ny for nx, ny in zip(STRUCTURE[1], OTHER_RANKS[0]))
                                prod_tt_ranks = tuple(rx * ry for rx, ry in zip(STRUCTURE[2], OTHER_RANKS[1]))
                                self.assertEqual(prod_tucker_ranks, x_op_y.tucker_ranks)
                                self.assertEqual(prod_tt_ranks, x_op_y.tt_ranks)

                            else:
                                raise ValueError

    def test_inner(self):
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

        for STRUCTURE, OTHER_RANKS in zip(structures, other_ranks):
            for USE_ORTHOGONALIZATION in [True, False]:
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                x = t3.TuckerTensorTrain.randn(*STRUCTURE, use_jax=False)
                for OTHER_TYPE in [
                    'NUMPY_DENSE', 'JAX_DENSE',
                    'NUMPY_T3', 'JAX_T3',
                ]:
                    with self.subTest(
                            USE_ORTHOGONALIZATION=USE_ORTHOGONALIZATION,
                            STRUCTURE=STRUCTURE, OTHER_RANKS=OTHER_RANKS, OTHER_TYPE=OTHER_TYPE,
                    ):
                        if OTHER_TYPE == 'NUMPY_DENSE':
                            y = np.random.randn(*(x.stack_shape + x.shape))

                        elif OTHER_TYPE == 'JAX_DENSE':
                            y = jnp.array(np.random.randn(*(x.stack_shape + x.shape)))

                        elif OTHER_TYPE == 'NUMPY_T3':
                            y_structure = STRUCTURE[:1] + OTHER_RANKS + STRUCTURE[3:]
                            y = t3.TuckerTensorTrain.randn(*y_structure, use_jax=False)

                        elif OTHER_TYPE == 'JAX_T3':
                            y_structure = STRUCTURE[:1] + OTHER_RANKS + STRUCTURE[3:]
                            y = t3.TuckerTensorTrain.randn(*y_structure, use_jax=True)

                        else:
                            print('OTHER_TYPE=', OTHER_TYPE)
                            raise ValueError

                        sum_axes = tuple(range(len(stack_shape), len(stack_shape + shape)))
                        x_dot_y_true = np.sum(_td(x) * _td(y), axis=sum_axes)

                        x_dot_y = x.inner(
                            y, use_orthogonalization=USE_ORTHOGONALIZATION
                        )
                        self.check_relerr(x_dot_y_true, x_dot_y)

    def test_norm(self):
        structures = [
            ((14,), (4,), (4, 5), (2, 3)),
            ((14, 15), (4, 5), (4, 5, 4), (2, 3)),
            ((14, 15, 16, 17), (4, 5, 6, 7), (4, 5, 4, 3, 2), (2, 3)),
            ((14, 15, 16), (4, 5, 6), (4, 5, 4, 3), ()),
        ]

        for STRUCTURE in structures:
            for USE_ORTHOGONALIZATION in [True, False]:
                shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
                x = t3.TuckerTensorTrain.randn(*STRUCTURE, use_jax=False)
                with self.subTest(
                        USE_ORTHOGONALIZATION=USE_ORTHOGONALIZATION,
                        STRUCTURE=STRUCTURE,
                ):
                    sum_axes = tuple(range(len(stack_shape), len(stack_shape + shape)))
                    x_dense = x.to_dense()
                    norm_x_true = np.sqrt(np.sum(x_dense**2, axis=sum_axes))

                    norm_x = x.norm(use_orthogonalization=USE_ORTHOGONALIZATION)

                    self.check_relerr(norm_x_true, norm_x)

    def test_sum(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                shape, tucker_ranks, tt_ranks, stack_shape = structure
                x = t3.TuckerTensorTrain.randn(*structure)
                with self.subTest(
                        FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                        AXES=None,
                ):
                    S = x.sum()
                    dense_x = x.to_dense()
                    non_stack_axes = tuple(ii + len(STACK_SHAPE) for ii in range(len(shape)))
                    S2 = dense_x.sum(axis=non_stack_axes)
                    self.check_relerr(S2, S)

                for ax in range(len(shape)):
                    with self.subTest(
                            FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                            AXES=ax,
                    ):
                        S = x.sum(axis=ax)
                        S_dense = S.to_dense() if isinstance(S, t3.TuckerTensorTrain) else S

                        dense_x = x.to_dense()
                        shifted_axis = ax + len(x.stack_shape)
                        S2_dense = dense_x.sum(axis=shifted_axis)
                        self.check_relerr(S2_dense, S_dense)

                all_axes = tuple(range(len(shape)))
                for num_ax in range(len(all_axes)+1):
                    for axes in itertools.combinations(all_axes, num_ax):
                        with self.subTest(
                                FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                                AXES=axes,
                        ):
                            S = x.sum(axis=axes)
                            S_dense = S.to_dense() if isinstance(S, t3.TuckerTensorTrain) else S

                            dense_x = x.to_dense()
                            shifted_axes = tuple(ii + len(x.stack_shape) for ii in axes)
                            S2_dense = dense_x.sum(axis=shifted_axes)
                            self.check_relerr(S2_dense, S_dense)

    def test_sum_stack(self):
        frame_structures = [
            ((8,),              (3,),           (1, 2)),
            ((5, 6),            (2, 3),         (1, 2, 1)),
            ((4, 5, 6),         (2, 3, 2),      (1, 2, 2, 1)),
            ((4, 5, 6),         (2, 2, 2),      (2, 2, 2, 2)),  # nontrivial leading/trailing TT ranks
        ]
        stack_shapes = [
            (2,),
            (3,),
            (2, 3),
            (2, 1, 2),
        ]

        for FRAME_STRUCTURE in frame_structures:
            shape, tucker_ranks, tt_ranks = FRAME_STRUCTURE
            for STACK_SHAPE in stack_shapes:
                x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks, STACK_SHAPE)
                m = len(STACK_SHAPE)

                all_axes = tuple(range(m))
                axis_options = [None, 0]
                for num_ax in range(1, m + 1):
                    axis_options += list(itertools.combinations(all_axes, num_ax))

                for AXIS in axis_options:
                    with self.subTest(
                            FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                            AXIS=AXIS,
                    ):
                        y = x.sum_stack(axis=AXIS)

                        if AXIS is None:
                            summed = all_axes
                        elif isinstance(AXIS, int):
                            summed = (AXIS,)
                        else:
                            summed = tuple(AXIS)
                        kept = tuple(STACK_SHAPE[i] for i in range(m) if i not in summed)
                        S = int(np.prod([STACK_SHAPE[i] for i in summed]))

                        dense_x = _td(x)
                        y_dense_true = dense_x.sum(axis=summed)
                        self.check_relerr(y_dense_true, _td(y))

                        self.assertEqual(kept, y.stack_shape)
                        self.assertEqual(tuple(S * n for n in tucker_ranks), y.tucker_ranks)
                        expected_tt_ranks = (1,) + tuple(S * r for r in tt_ranks[1:-1]) + (1,)
                        self.assertEqual(expected_tt_ranks, y.tt_ranks)

    def test_sum_stack_corewise(self):
        frame_structures = [
            ((8,),              (3,),           (1, 2)),
            ((5, 6),            (2, 3),         (1, 2, 1)),
            ((4, 5, 6),         (2, 3, 2),      (1, 2, 2, 1)),
        ]
        stack_shapes = [
            (2,),
            (2, 3),
            (2, 1, 2),
        ]

        for FRAME_STRUCTURE in frame_structures:
            shape, tucker_ranks, tt_ranks = FRAME_STRUCTURE
            for STACK_SHAPE in stack_shapes:
                x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks, STACK_SHAPE)
                m = len(STACK_SHAPE)

                all_axes = tuple(range(m))
                axis_options = [None, 0]
                for num_ax in range(1, m + 1):
                    axis_options += list(itertools.combinations(all_axes, num_ax))

                for AXIS in axis_options:
                    with self.subTest(
                            FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                            AXIS=AXIS,
                    ):
                        y = x.sum_stack_corewise(axis=AXIS)

                        if AXIS is None:
                            summed = all_axes
                        elif isinstance(AXIS, int):
                            summed = (AXIS,)
                        else:
                            summed = tuple(AXIS)
                        kept = tuple(STACK_SHAPE[i] for i in range(m) if i not in summed)

                        self.assertEqual(kept, y.stack_shape)
                        self.assertEqual((tucker_ranks, tt_ranks), y.ranks)  # ranks unchanged

                        tucker_cores2 = tuple(B.sum(axis=summed) for B in x.tucker_cores)
                        tt_cores2 = tuple(G.sum(axis=summed) for G in x.tt_cores)
                        err = cw.corewise_norm(cw.corewise_sub((tucker_cores2, tt_cores2), y.data))
                        self.assertLessEqual(float(err), tol)

    ####

    def test_down_svd_tucker_core(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]


        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                shape, tucker_ranks, tt_ranks, stack_shape = structure
                for MIN_RANK, MAX_RANK in zip(
                    [None, 2,    None, 2],
                    [None, None, 2,    3],
                ):
                    for X_TYPE in ['RANDN', 'ONES']:
                        if X_TYPE == 'RANDN':
                            x = t3.TuckerTensorTrain.randn(*structure, use_jax=False)
                        else:
                            x = t3.TuckerTensorTrain.ones(
                                shape, stack_shape=STACK_SHAPE, use_jax=False,
                            )
                            x = x.resize(shape, tucker_ranks, tt_ranks)

                        for CORE_IND in range(len(shape)):
                            with self.subTest(
                                    FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                                    MIN_RANK=MIN_RANK, MAX_RANK=MAX_RANK,
                                    CORE_IND=CORE_IND,
                            ):
                                x2, ss = x.down_svd_tucker_core(CORE_IND, MIN_RANK, MAX_RANK)
                                r = ss.shape[-1]
                                self.assertEqual(r, x2.tucker_ranks[CORE_IND])

                                if MAX_RANK is not None:
                                    self.assertLessEqual(r, MAX_RANK)
                                else:
                                    self.check_relerr(x2.to_dense(), x.to_dense())

                                if MIN_RANK is not None:
                                    self.assertGreaterEqual(r, MIN_RANK)

                                B = x.tucker_cores[CORE_IND]
                                _, ss2, _ = np.linalg.svd(B, full_matrices=False)
                                self.check_relerr(ss2[..., :r], ss)

                                B2 = x2.tucker_cores[CORE_IND]
                                self.check_relerr(
                                    np.eye(B2.shape[-2]),
                                    np.einsum('...io,...jo->...ij', B2, B2)
                                )

    def test_down_svd_tucker_core_tols(self):
        structures = [
            ((10,),             (7,),           (6, 7)),
            ((10, 11),          (7, 8),         (6, 7, 8)),
            ((10, 11, 12),      (7, 8, 9),      (6, 7, 8, 7)),
            ((10, 11, 12, 13),  (7, 8, 9, 8),   (6, 7, 8, 7, 6)),
        ]

        for STRUCTURE in structures:
            shape, tucker_ranks, tt_ranks = STRUCTURE
            x = _random_preconditioned_t3(shape, tucker_ranks, tt_ranks)

            for RTOL in [5e-1, 5e-2, 5e-3, 5e-4]:
                for ATOL in [5e-1, 5e-2, 5e-3, 5e-4]:
                    for MIN_RANK in [1,2,3,4,5,6,7]:
                        for MAX_RANK in [1,2,3,4,5,6,7]:
                            for CORE_IND in range(len(shape)):
                                with self.subTest(
                                        STRUCTURE=STRUCTURE,
                                        RTOL=RTOL, ATOL=ATOL,
                                        MIN_RANK=MIN_RANK, MAX_RANK=MAX_RANK,
                                        CORE_IND=CORE_IND,
                                ):
                                    x2, ss = x.down_svd_tucker_core(
                                        CORE_IND, min_rank=MIN_RANK, max_rank=MAX_RANK, rtol=RTOL, atol=ATOL,
                                    )
                                    r = ss.shape[-1]
                                    self.assertEqual(r, x2.tucker_ranks[CORE_IND])

                                    B = x.tucker_cores[CORE_IND]
                                    _, ss_big, _ = np.linalg.svd(B, full_matrices=False)
                                    fronorm = np.sqrt(np.sum(ss_big**2))
                                    tail_fronorms = np.sqrt(np.cumsum(ss_big[::-1]**2))[::-1]
                                    r0 = np.sum(tail_fronorms >= np.maximum(fronorm * RTOL, ATOL))
                                    K = len(ss_big)

                                    # print('r=', r, ', K=', K, ', MIN_RANK=', MIN_RANK, ', MAX_RANK=', MAX_RANK)

                                    r_true = np.maximum(np.minimum(K, MIN_RANK), np.minimum(r0, MAX_RANK))
                                    self.assertEqual(r_true, r)
                                    self.check_relerr(ss_big[:r], ss)

                                    B2 = x2.tucker_cores[CORE_IND]
                                    self.check_relerr(
                                        np.eye(B2.shape[-2]),
                                        np.einsum('...io,...jo->...ij', B2, B2)
                                    )

    def test_left_svd_tt_core(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                shape, tucker_ranks, tt_ranks, stack_shape = structure
                for MIN_RANK, MAX_RANK in zip(
                    [None, 2,    None, 2],
                    [None, None, 2,    3],
                ):
                    for X_TYPE in ['RANDN', 'ONES']:
                        if X_TYPE == 'RANDN':
                            x = t3.TuckerTensorTrain.randn(*structure, use_jax=False)
                        else:
                            x = t3.TuckerTensorTrain.ones(
                                shape, stack_shape=STACK_SHAPE, use_jax=False,
                            )
                            x = x.resize(shape, tucker_ranks, tt_ranks)

                        for CORE_IND in range(len(shape)):
                            with self.subTest(
                                    FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                                    MIN_RANK=MIN_RANK, MAX_RANK=MAX_RANK,
                                    CORE_IND=CORE_IND,
                            ):
                                x2, ss = x.left_svd_tt_core(CORE_IND, MIN_RANK, MAX_RANK)
                                r = ss.shape[-1]
                                self.assertEqual(r, x2.tt_ranks[CORE_IND+1])

                                if MAX_RANK is not None:
                                    self.assertLessEqual(r, MAX_RANK)
                                else:
                                    self.check_relerr(x2.to_dense(), x.to_dense())

                                if MIN_RANK is not None:
                                    self.assertGreaterEqual(r, MIN_RANK)

                                G = x.tt_cores[CORE_IND]
                                A = G.reshape(stack_shape+(G.shape[-3]*G.shape[-2], G.shape[-1]))
                                _, ss2, _ = np.linalg.svd(A, full_matrices=False)
                                self.check_relerr(ss2[..., :r], ss)

                                if CORE_IND < len(shape) - 1:
                                    G2 = x2.tt_cores[CORE_IND]
                                    self.check_relerr(
                                        np.eye(G2.shape[-1]),
                                        np.einsum('...iaj,...iak ->...jk', G2, G2)
                                    )

    def test_left_svd_tucker_core_tols(self):
        structures = [
            ((10,),             (7,),           (6, 7)),
            ((10, 11),          (7, 8),         (6, 7, 8)),
            ((10, 11, 12),      (7, 8, 9),      (6, 7, 8, 7)),
            ((10, 11, 12, 13),  (7, 8, 9, 8),   (6, 7, 8, 7, 6)),
        ]

        for STRUCTURE in structures:
            shape, tucker_ranks, tt_ranks = STRUCTURE
            x = _random_preconditioned_t3(shape, tucker_ranks, tt_ranks)

            for RTOL in [5e-1, 5e-2, 5e-3, 5e-4]:
                for ATOL in [5e-1, 5e-2, 5e-3, 5e-4]:
                    for MIN_RANK in [1,2,3,4,5,6,7]:
                        for MAX_RANK in [1,2,3,4,5,6,7]:
                            for CORE_IND in range(len(shape)):
                                with self.subTest(
                                        STRUCTURE=STRUCTURE,
                                        RTOL=RTOL, ATOL=ATOL,
                                        MIN_RANK=MIN_RANK, MAX_RANK=MAX_RANK,
                                        CORE_IND=CORE_IND,
                                ):
                                    x2, ss = x.left_svd_tt_core(
                                        CORE_IND, min_rank=MIN_RANK, max_rank=MAX_RANK, rtol=RTOL, atol=ATOL,
                                    )
                                    r = ss.shape[-1]
                                    self.assertEqual(r, x2.tt_ranks[CORE_IND+1])

                                    G = x.tt_cores[CORE_IND]
                                    _, ss_big, _ = np.linalg.svd(
                                        G.reshape((G.shape[0]*G.shape[1], G.shape[2])),
                                        full_matrices=False
                                    )
                                    fronorm = np.sqrt(np.sum(ss_big ** 2))
                                    tail_fronorms = np.sqrt(np.cumsum(ss_big[::-1] ** 2))[::-1]
                                    r0 = np.sum(tail_fronorms >= np.maximum(fronorm * RTOL, ATOL))
                                    K = len(ss_big)

                                    # print('r=', r, ', K=', K, ', MIN_RANK=', MIN_RANK, ', MAX_RANK=', MAX_RANK)

                                    r_true = np.maximum(np.minimum(K, MIN_RANK), np.minimum(r0, MAX_RANK))
                                    self.assertEqual(r_true, r)
                                    self.check_relerr(ss_big[:r], ss)

                                    if CORE_IND < len(shape) - 1:
                                        G2 = x2.tt_cores[CORE_IND]
                                        self.check_relerr(
                                            np.eye(G2.shape[-1]),
                                            np.einsum('...iaj,...iak ->...jk', G2, G2)
                                        )

    def test_right_svd_tt_core(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                shape, tucker_ranks, tt_ranks, stack_shape = structure
                for MIN_RANK, MAX_RANK in zip(
                    [None, 2,    None, 2],
                    [None, None, 2,    3],
                ):
                    for X_TYPE in ['RANDN', 'ONES']:
                        if X_TYPE == 'RANDN':
                            x = t3.TuckerTensorTrain.randn(*structure, use_jax=False)
                        else:
                            x = t3.TuckerTensorTrain.ones(
                                shape, stack_shape=STACK_SHAPE, use_jax=False,
                            )
                            x = x.resize(shape, tucker_ranks, tt_ranks)

                        for CORE_IND in range(len(shape)):
                            with self.subTest(
                                    FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                                    MIN_RANK=MIN_RANK, MAX_RANK=MAX_RANK,
                                    CORE_IND=CORE_IND,
                            ):
                                x2, ss = x.right_svd_tt_core(CORE_IND, MIN_RANK, MAX_RANK)
                                r = ss.shape[-1]
                                self.assertEqual(r, x2.tt_ranks[CORE_IND])

                                if MAX_RANK is not None:
                                    self.assertLessEqual(r, MAX_RANK)
                                else:
                                    self.check_relerr(x2.to_dense(), x.to_dense())

                                if MIN_RANK is not None:
                                    self.assertGreaterEqual(r, MIN_RANK)

                                G = x.tt_cores[CORE_IND]
                                A = G.reshape(stack_shape+(G.shape[-3], G.shape[-2]*G.shape[-1]))
                                _, ss2, _ = np.linalg.svd(A, full_matrices=False)
                                self.check_relerr(ss2[..., :r], ss)

                                if CORE_IND > 1:
                                    G2 = x2.tt_cores[CORE_IND]
                                    self.check_relerr(
                                        np.eye(G2.shape[-3]),
                                        np.einsum('...iaj,...kaj->...ik', G2, G2)
                                    )

    def test_right_svd_tucker_core_tols(self):
        structures = [
            ((10,),             (7,),           (6, 7)),
            ((10, 11),          (7, 8),         (6, 7, 8)),
            ((10, 11, 12),      (7, 8, 9),      (6, 7, 8, 7)),
            ((10, 11, 12, 13),  (7, 8, 9, 8),   (6, 7, 8, 7, 6)),
        ]

        for STRUCTURE in structures:
            shape, tucker_ranks, tt_ranks = STRUCTURE
            x = _random_preconditioned_t3(shape, tucker_ranks, tt_ranks)

            for RTOL in [5e-1, 5e-2, 5e-3, 5e-4]:
                for ATOL in [5e-1, 5e-2, 5e-3, 5e-4]:
                    for MIN_RANK in [1,2,3,4,5,6,7]:
                        for MAX_RANK in [1,2,3,4,5,6,7]:
                            for CORE_IND in range(len(shape)):
                                with self.subTest(
                                        STRUCTURE=STRUCTURE,
                                        RTOL=RTOL, ATOL=ATOL,
                                        MIN_RANK=MIN_RANK, MAX_RANK=MAX_RANK,
                                        CORE_IND=CORE_IND,
                                ):
                                    x2, ss = x.right_svd_tt_core(
                                        CORE_IND, min_rank=MIN_RANK, max_rank=MAX_RANK, rtol=RTOL, atol=ATOL,
                                    )
                                    r = ss.shape[-1]
                                    self.assertEqual(r, x2.tt_ranks[CORE_IND])

                                    G = x.tt_cores[CORE_IND]
                                    _, ss_big, _ = np.linalg.svd(
                                        G.reshape((G.shape[0], G.shape[1]*G.shape[2])),
                                        full_matrices=False
                                    )
                                    fronorm = np.sqrt(np.sum(ss_big ** 2))
                                    tail_fronorms = np.sqrt(np.cumsum(ss_big[::-1] ** 2))[::-1]
                                    r0 = np.sum(tail_fronorms >= np.maximum(fronorm * RTOL, ATOL))
                                    K = len(ss_big)

                                    # print('r=', r, ', K=', K, ', MIN_RANK=', MIN_RANK, ', MAX_RANK=', MAX_RANK)

                                    r_true = np.maximum(np.minimum(K, MIN_RANK), np.minimum(r0, MAX_RANK))
                                    self.assertEqual(r_true, r)
                                    self.check_relerr(ss_big[:r], ss)

                                    if CORE_IND > 1:
                                        G2 = x2.tt_cores[CORE_IND]
                                        self.check_relerr(
                                            np.eye(G2.shape[-3]),
                                            np.einsum('...iaj,...kaj->...ik', G2, G2)
                                        )

    def test_up_svd_tt_core(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                shape, tucker_ranks, tt_ranks, stack_shape = structure
                for MIN_RANK, MAX_RANK in zip(
                    [None, 2,    None, 2],
                    [None, None, 2,    3],
                ):
                    for X_TYPE in ['RANDN', 'ONES']:
                        if X_TYPE == 'RANDN':
                            x = t3.TuckerTensorTrain.randn(*structure, use_jax=False)
                        else:
                            x = t3.TuckerTensorTrain.ones(
                                shape, stack_shape=STACK_SHAPE, use_jax=False,
                            )
                            x = x.resize(shape, tucker_ranks, tt_ranks)

                        for CORE_IND in range(len(shape)):
                            with self.subTest(
                                    FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                                    MIN_RANK=MIN_RANK, MAX_RANK=MAX_RANK,
                                    CORE_IND=CORE_IND,
                            ):
                                x2, ss = x.up_svd_tt_core(CORE_IND, MIN_RANK, MAX_RANK)
                                r = ss.shape[-1]
                                self.assertEqual(r, x2.tucker_ranks[CORE_IND])

                                if MAX_RANK is not None:
                                    self.assertLessEqual(r, MAX_RANK)
                                else:
                                    self.check_relerr(x2.to_dense(), x.to_dense())

                                if MIN_RANK is not None:
                                    self.assertGreaterEqual(r, MIN_RANK)

                                G = x.tt_cores[CORE_IND]
                                A = G.swapaxes(-1, -2)
                                A = A.reshape(stack_shape+(A.shape[-3]*A.shape[-2], A.shape[-1]))
                                _, ss2, _ = np.linalg.svd(A, full_matrices=False)
                                self.check_relerr(ss2[..., :r], ss)

                                G2 = x2.tt_cores[CORE_IND]
                                self.check_relerr(
                                    np.eye(G2.shape[-2]),
                                    np.einsum('...aib,...ajb->...ij', G2, G2)
                                )

    def test_up_svd_tt_core_tols(self):
        structures = [
            ((10,),             (7,),           (6, 7)),
            ((10, 11),          (7, 8),         (6, 7, 8)),
            ((10, 11, 12),      (7, 8, 9),      (6, 7, 8, 7)),
            ((10, 11, 12, 13),  (7, 8, 9, 8),   (6, 7, 8, 7, 6)),
        ]

        for STRUCTURE in structures:
            shape, tucker_ranks, tt_ranks = STRUCTURE
            x = _random_preconditioned_t3(shape, tucker_ranks, tt_ranks)

            for RTOL in [5e-1, 5e-2, 5e-3, 5e-4]:
                for ATOL in [5e-1, 5e-2, 5e-3, 5e-4]:
                    for MIN_RANK in [1,2,3,4,5,6,7]:
                        for MAX_RANK in [1,2,3,4,5,6,7]:
                            for CORE_IND in range(len(shape)):
                                with self.subTest(
                                        STRUCTURE=STRUCTURE,
                                        RTOL=RTOL, ATOL=ATOL,
                                        MIN_RANK=MIN_RANK, MAX_RANK=MAX_RANK,
                                        CORE_IND=CORE_IND,
                                ):
                                    x2, ss = x.up_svd_tt_core(
                                        CORE_IND, min_rank=MIN_RANK, max_rank=MAX_RANK, rtol=RTOL, atol=ATOL,
                                    )
                                    r = ss.shape[-1]
                                    self.assertEqual(r, x2.tucker_ranks[CORE_IND])

                                    G = x.tt_cores[CORE_IND]
                                    A = G.swapaxes(-2, -1)
                                    _, ss_big, _ = np.linalg.svd(
                                        A.reshape((A.shape[-3]*A.shape[-2], A.shape[-1])),
                                        full_matrices=False
                                    )
                                    fronorm = np.sqrt(np.sum(ss_big ** 2))
                                    tail_fronorms = np.sqrt(np.cumsum(ss_big[::-1] ** 2))[::-1]
                                    r0 = np.sum(tail_fronorms >= np.maximum(fronorm * RTOL, ATOL))
                                    K = len(ss_big)

                                    # print('r=', r, ', K=', K, ', MIN_RANK=', MIN_RANK, ', MAX_RANK=', MAX_RANK)

                                    r_true = np.maximum(np.minimum(K, MIN_RANK), np.minimum(r0, MAX_RANK))
                                    self.assertEqual(r_true, r)
                                    self.check_relerr(ss_big[:r], ss)

                                    G2 = x2.tt_cores[CORE_IND]
                                    self.check_relerr(
                                        np.eye(G2.shape[-2]),
                                        np.einsum('...aib,...ajb->...ij', G2, G2)
                                    )

    def test_orthogonalize_relative_to_tucker_core(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                shape, tucker_ranks, tt_ranks, stack_shape = structure
                x = t3.TuckerTensorTrain.randn(*structure, use_jax=False)
                for CORE_IND in range(len(shape)):
                    with self.subTest(
                            FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                            CORE_IND=CORE_IND,
                    ):
                        dense_x = x.to_dense()

                        x2 = x.orthogonalize_relative_to_tucker_core(CORE_IND)

                        dense_x2 = x2.to_dense()
                        self.check_relerr(dense_x, dense_x2)

                        for ii, B in enumerate(x2.tucker_cores):
                            if ii != CORE_IND:
                                self.check_relerr(
                                    np.eye(B.shape[-2]),
                                    np.einsum('...io,...jo->...ij', B, B)
                                )

                        for G in x2.tt_cores[:CORE_IND]:
                            self.check_relerr(
                                np.eye(G.shape[-1]),
                                np.einsum('...aib,...aic->...bc', G, G)
                            )

                        Gm = x2.tt_cores[CORE_IND]
                        self.check_relerr(
                            np.eye(Gm.shape[-2]),
                            np.einsum('...aib,...ajb->...ij', Gm, Gm)
                        )

                        for G in x2.tt_cores[CORE_IND+1:]:
                            self.check_relerr(
                                np.eye(G.shape[-3]),
                                np.einsum('...aib,...cib->...ac', G, G)
                            )

    def test_orthogonalize_relative_to_tt_core(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                shape, tucker_ranks, tt_ranks, stack_shape = structure
                x = t3.TuckerTensorTrain.randn(*structure, use_jax=False)
                for CORE_IND in range(len(shape)):
                    with self.subTest(
                            FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                            CORE_IND=CORE_IND,
                    ):
                        dense_x = x.to_dense()

                        x2 = x.orthogonalize_relative_to_tt_core(CORE_IND)

                        dense_x2 = x2.to_dense()
                        self.check_relerr(dense_x, dense_x2)

                        for B in x2.tucker_cores:
                            self.check_relerr(
                                np.eye(B.shape[-2]),
                                np.einsum('...io,...jo->...ij', B, B)
                            )

                        for G in x2.tt_cores[:CORE_IND]:
                            self.check_relerr(
                                np.eye(G.shape[-1]),
                                np.einsum('...aib,...aic->...bc', G, G)
                            )

                        for G in x2.tt_cores[CORE_IND+1:]:
                            self.check_relerr(
                                np.eye(G.shape[-3]),
                                np.einsum('...aib,...cib->...ac', G, G)
                            )


    def test_down_orthogonalize_tucker_cores(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                x = t3.TuckerTensorTrain.randn(*structure, use_jax=False)
                with self.subTest(
                        FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                ):
                    dense_x = x.to_dense()

                    x2 = x.down_orthogonalize_tucker_cores()

                    dense_x2 = x2.to_dense()
                    self.check_relerr(dense_x, dense_x2)

                    for B in x2.tucker_cores:
                        self.check_relerr(
                            np.eye(B.shape[-2]),
                            np.einsum('...io,...jo->...ij', B, B)
                        )

    def test_up_orthogonalize_tt_cores(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                x = t3.TuckerTensorTrain.randn(*structure, use_jax=False)
                with self.subTest(
                        FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                ):
                    dense_x = x.to_dense()

                    x2 = x.up_orthogonalize_tt_cores()

                    dense_x2 = x2.to_dense()
                    self.check_relerr(dense_x, dense_x2)

                    for G in x2.tt_cores:
                        self.check_relerr(
                            np.eye(G.shape[-2]),
                            np.einsum('...aib,...ajb->...ij', G, G)
                        )

    def test_left_orthogonalize_tt_cores(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                x = t3.TuckerTensorTrain.randn(*structure, use_jax=False)
                with self.subTest(
                        FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                ):
                    dense_x = x.to_dense()

                    x2 = x.left_orthogonalize_tt_cores()

                    dense_x2 = x2.to_dense()
                    self.check_relerr(dense_x, dense_x2)

                    for G in x2.tt_cores[:-1]:
                        self.check_relerr(
                            np.eye(G.shape[-1]),
                            np.einsum('...aib,...aic->...bc', G, G)
                        )

    def test_right_orthogonalize_tt_cores(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [
            (),
            (2,3)
        ]

        for FRAME_STRUCTURE in frame_structures:
            for STACK_SHAPE in stack_shapes:
                structure = FRAME_STRUCTURE + (STACK_SHAPE,)
                x = t3.TuckerTensorTrain.randn(*structure, use_jax=False)
                with self.subTest(
                        FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                ):
                    dense_x = x.to_dense()

                    x2 = x.right_orthogonalize_tt_cores()

                    dense_x2 = x2.to_dense()
                    self.check_relerr(dense_x, dense_x2)

                    for G in x2.tt_cores[1:]:
                        self.check_relerr(
                            np.eye(G.shape[-3]),
                            np.einsum('...aib,...cib->...ac', G, G)
                        )

    def test_entries(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
        ]
        stack_shapes = [
            (),
            (2, 3),
        ]
        index_stack_shapes = [
            (),
            (5,),
            (2,3),
        ]

        for FRAME_STRUCTURE in frame_structures:
            shape, tucker_ranks, tt_ranks = FRAME_STRUCTURE
            for STACK_SHAPE in stack_shapes:
                for INDEX_STACK_SHAPE in index_stack_shapes:
                    with self.subTest(
                            FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                            INDEX_STACK_SHAPE=INDEX_STACK_SHAPE
                    ):
                        x = t3.TuckerTensorTrain.randn(*(FRAME_STRUCTURE + (STACK_SHAPE,)))

                        index = np.array([np.random.choice(N, size=INDEX_STACK_SHAPE) for N in shape])

                        entries = x.entries(index)
                        self.assertEqual(INDEX_STACK_SHAPE + STACK_SHAPE, entries.shape)  # F + G

                        def _get_entries_dense(a, ind, ss, iss):
                            if len(ss) == 0 and len(iss) == 0:
                                return a[tuple(ind)]
                            elif len(ss) == 0:
                                return np.array([
                                    _get_entries_dense(a, ind[:,ii], ss, iss[1:])
                                    for ii in range(iss[0])
                                ])
                            else:
                                return np.array([
                                    _get_entries_dense(a[ii], ind, ss[1:], iss)
                                    for ii in range(ss[0])
                                ])

                        x_dense = x.to_dense()
                        entries2 = _get_entries_dense(x_dense, index, STACK_SHAPE, INDEX_STACK_SHAPE)
                        # reference is STACK + INDEX; reorder to INDEX + STACK to match the F+G output
                        nS, nI = len(STACK_SHAPE), len(INDEX_STACK_SHAPE)
                        entries2 = np.moveaxis(entries2, tuple(range(nS)), tuple(range(nI, nI + nS)))

                        self.check_relerr(entries2, entries)


    def test_apply(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
        ]
        stack_shapes = [
            (),
            (2, 3),
        ]
        vecs_stack_shapes = [
            (),
            (5,),
            (2,3),
        ]

        for FRAME_STRUCTURE in frame_structures:
            shape, tucker_ranks, tt_ranks = FRAME_STRUCTURE
            for STACK_SHAPE in stack_shapes:
                for VECS_STACK_SHAPE in vecs_stack_shapes:
                    with self.subTest(
                            FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                    ):
                        x = t3.TuckerTensorTrain.randn(*(FRAME_STRUCTURE + (STACK_SHAPE,)))

                        vecs = [np.random.randn(*(VECS_STACK_SHAPE + (N,))) for N in shape]

                        result = x.apply(vecs)
                        self.assertEqual(VECS_STACK_SHAPE + STACK_SHAPE, result.shape)  # F + G

                        def _apply_dense(a, vecs, ss, vss):
                            if len(ss) == 0 and len(vss) == 0:
                                if len(a.shape) == 1:
                                    return np.einsum('i,i', a, *vecs)
                                elif len(a.shape) == 2:
                                    return np.einsum('ij,i,j', a, *vecs)
                                elif len(a.shape) == 3:
                                    return np.einsum('ijk,i,j,k', a, *vecs)
                                elif len(a.shape) == 4:
                                    return np.einsum('ijkl,i,j,k,l', a, *vecs)
                                else:
                                    raise ValueError
                            elif len(ss) == 0:
                                subvecs = [
                                    [vecs[jj][ii] for jj in range(len(vecs))]
                                    for ii in range(len(vecs[0]))
                                ]
                                return np.array([
                                    _apply_dense(a, subvecs[ii], ss, vss[1:])
                                    for ii in range(vss[0])
                                ])
                            else:
                                return np.array([
                                    _apply_dense(a[ii], vecs, ss[1:], vss)
                                    for ii in range(ss[0])
                                ])

                        x_dense = x.to_dense()
                        result2 = _apply_dense(x_dense, vecs, STACK_SHAPE, VECS_STACK_SHAPE)
                        # reference is STACK + VECS; reorder to VECS + STACK to match the F+G output
                        nS, nV = len(STACK_SHAPE), len(VECS_STACK_SHAPE)
                        result2 = np.moveaxis(result2, tuple(range(nS)), tuple(range(nV, nV + nS)))

                        self.check_relerr(result2, result)

    def test_apply_ambient_transpose(self):
        # ambient adjoint of apply: primary (sum=False) keeps probe stack W; sum=True contracts it (J^T r).
        # Returns CP factors; from_canonical realizes them as a TuckerTensorTrain.
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [(), (2, 3)]            # C (frame/core stack, carried by the residual)
        probe_stack_shapes = [(), (5,), (2, 3)]  # W (apply-vector stack)

        for BASE in frame_structures:
            shape, tucker_ranks, tt_ranks = BASE
            d = len(shape)
            for C in stack_shapes:
                for W in probe_stack_shapes:
                    with self.subTest(BASE=BASE, C=C, W=W):
                        nW, nN = len(W), d
                        x = t3.TuckerTensorTrain.randn(*(BASE + (C,)))
                        ww = [np.random.randn(*(W + (N,))) for N in shape]
                        c = np.asarray(np.random.randn(*(W + C)))
                        fwd = np.asarray(x.apply(ww))                          # W + C
                        xd = x.to_dense()                                     # C + N

                        def ndot(ATd, lead):  # contract (lead + C + N) with xd (C + N) over N, x bcast over lead
                            xb = xd.reshape((1,) * lead + xd.shape)
                            return np.sum(ATd * xb, axis=tuple(range(ATd.ndim - nN, ATd.ndim)))

                        # primary: W is a passthrough stack (CP rank R=1); per-probe identity
                        # <AT(c)_W, x> == c*apply, after realizing the CP factors via from_canonical.
                        ATf_factors = t3.TuckerTensorTrain.apply_ambient_transpose(c, ww)
                        self.assertEqual(W + C + (1, shape[0]), ATf_factors[0].shape)   # CP rank 1, W stacked
                        ATf = t3.TuckerTensorTrain.from_canonical(ATf_factors)
                        self.assertEqual(W + C, ATf.stack_shape)
                        self.check_relerr(c * fwd, ndot(ATf.to_dense(), nW))

                        # summed: W becomes the CP rank |W| (the J^T r back-projection), O(|W|N) -- the
                        # |W|^2 copy-tensor cost lives only in from_canonical, not in the returned factors.
                        ATt_factors = t3.TuckerTensorTrain.apply_ambient_transpose(c, ww, sum_over_probes=True)
                        self.assertEqual(C + (int(np.prod(W, dtype=int)), shape[0]), ATt_factors[0].shape)
                        ATt = t3.TuckerTensorTrain.from_canonical(ATt_factors)
                        self.assertEqual(C, ATt.stack_shape)
                        lhs = np.sum(ATt.to_dense().reshape(C + (-1,)) * xd.reshape(C + (-1,)), axis=-1)
                        self.check_relerr(np.sum(c * fwd, axis=tuple(range(nW))), lhs)

                        # consistency: sum=True == sum over W of sum=False
                        self.check_relerr(ATt.to_dense(), ATf.to_dense().sum(axis=tuple(range(nW))))

    def test_entries_ambient_transpose(self):
        # ambient adjoint of entries: scatter c at index. Primary keeps W; sum=True scatter-adds
        # collisions. Returns CP factors (one-hots); from_canonical realizes them as a T3.
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        stack_shapes = [(), (2, 3)]
        index_stack_shapes = [(), (5,), (2, 3)]

        for BASE in frame_structures:
            shape, tucker_ranks, tt_ranks = BASE
            d = len(shape)
            for C in stack_shapes:
                for W in index_stack_shapes:
                    with self.subTest(BASE=BASE, C=C, W=W):
                        nW, nN = len(W), d
                        x = t3.TuckerTensorTrain.randn(*(BASE + (C,)))
                        idx = np.array([np.random.randint(0, N, size=W) for N in shape])  # (d,) + W
                        c = np.asarray(np.random.randn(*(W + C)))
                        fwd = np.asarray(x.entries(idx))                      # W + C
                        xd = x.to_dense()                                     # C + N

                        def ndot(ETd, lead):
                            xb = xd.reshape((1,) * lead + xd.shape)
                            return np.sum(ETd * xb, axis=tuple(range(ETd.ndim - nN, ETd.ndim)))

                        ETf_factors = t3.TuckerTensorTrain.entries_ambient_transpose(c, idx, shape)
                        self.assertEqual(W + C + (1, shape[0]), ETf_factors[0].shape)   # CP rank 1, W stacked
                        ETf = t3.TuckerTensorTrain.from_canonical(ETf_factors)
                        self.assertEqual(W + C, ETf.stack_shape)
                        self.check_relerr(c * fwd, ndot(ETf.to_dense(), nW))

                        ETt_factors = t3.TuckerTensorTrain.entries_ambient_transpose(c, idx, shape, sum_over_probes=True)
                        self.assertEqual(C + (int(np.prod(W, dtype=int)), shape[0]), ETt_factors[0].shape)
                        ETt = t3.TuckerTensorTrain.from_canonical(ETt_factors)
                        self.assertEqual(C, ETt.stack_shape)
                        lhs = np.sum(ETt.to_dense().reshape(C + (-1,)) * xd.reshape(C + (-1,)), axis=-1)
                        self.check_relerr(np.sum(c * fwd, axis=tuple(range(nW))), lhs)

                        self.check_relerr(ETt.to_dense(), ETf.to_dense().sum(axis=tuple(range(nW))))

    def test_apply_corewise_transpose(self):
        # corewise (non-manifold, Sec 6.3) transpose: gradient of apply w.r.t. the cores. Oracle: the
        # adjoint identity vs the EXACT forward corewise Jacobian (sum of single-core replacements).
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        def replace(x, kind, i, new):                      # x with one core replaced
            tk, tt = [list(cs) for cs in x.data]
            (tk if kind == 'U' else tt)[i] = new
            return t3.TuckerTensorTrain(tuple(tk), tuple(tt))
        def core_dot(gA, gB, nC):                          # sum_cores <a,b>, keep leading C stack
            return sum(np.sum(a * b, axis=tuple(range(nC, a.ndim)))
                       for a, b in zip(gA[0] + gA[1], gB[0] + gB[1]))
        for BASE in frame_structures:
            shape, _, _ = BASE
            d = len(shape)
            for C in [(), (2, 3)]:                          # frame stack
                for W in [(), (5,)]:                        # probe stack
                    with self.subTest(BASE=BASE, C=C, W=W):
                        nW, nC = len(W), len(C)
                        x = t3.TuckerTensorTrain.randn(*(BASE + (C,)))
                        tk, tt = [list(cs) for cs in x.data]
                        ww = [np.random.randn(*(W + (N,))) for N in shape]
                        c = np.asarray(np.random.randn(*(W + C)))
                        dU = [np.random.randn(*u.shape) for u in tk]
                        dG = [np.random.randn(*g.shape) for g in tt]
                        # exact forward corewise Jacobian J(dcores) = sum over single-core replacements
                        Jd = sum(np.asarray(replace(x, 'U', i, dU[i]).apply(ww)) for i in range(d)) \
                           + sum(np.asarray(replace(x, 'G', i, dG[i]).apply(ww)) for i in range(d))  # W+C

                        # summed (J^T r): gradients shaped exactly like the cores (no |W| anywhere)
                        gU, gG = x.apply_corewise_transpose(c, ww, sum_over_probes=True)
                        self.assertEqual([u.shape for u in tk], [g.shape for g in gU])
                        self.assertEqual([g.shape for g in tt], [g.shape for g in gG])
                        lhs = core_dot(([np.asarray(g) for g in gU], [np.asarray(g) for g in gG]), (dU, dG), nC)
                        rhs = np.sum(c * Jd, axis=tuple(range(nW)))               # sum over W, keep C
                        self.check_relerr(rhs, lhs)

                        # unsummed: W is a leading stack on each gradient; sum=True == sum_W of sum=False
                        gUk, gGk = x.apply_corewise_transpose(c, ww)
                        self.assertEqual(W + tk[0].shape, gUk[0].shape)
                        for s, k in zip(gU, gUk):
                            self.check_relerr(np.asarray(s), np.asarray(k).sum(axis=tuple(range(nW))))
                        for s, k in zip(gG, gGk):
                            self.check_relerr(np.asarray(s), np.asarray(k).sum(axis=tuple(range(nW))))

    def test_entries_corewise_transpose(self):
        # entries counterpart of test_apply_corewise_transpose (Sec 6.3 substitution, one-hot apply vecs).
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        def replace(x, kind, i, new):
            tk, tt = [list(cs) for cs in x.data]
            (tk if kind == 'U' else tt)[i] = new
            return t3.TuckerTensorTrain(tuple(tk), tuple(tt))
        def core_dot(gA, gB, nC):
            return sum(np.sum(a * b, axis=tuple(range(nC, a.ndim)))
                       for a, b in zip(gA[0] + gA[1], gB[0] + gB[1]))
        for BASE in frame_structures:
            shape, _, _ = BASE
            d = len(shape)
            for C in [(), (2, 3)]:
                for W in [(), (5,)]:
                    with self.subTest(BASE=BASE, C=C, W=W):
                        nW, nC = len(W), len(C)
                        x = t3.TuckerTensorTrain.randn(*(BASE + (C,)))
                        tk, tt = [list(cs) for cs in x.data]
                        idx = np.array([np.random.randint(0, N, size=W) for N in shape])   # (d,)+W
                        c = np.asarray(np.random.randn(*(W + C)))
                        dU = [np.random.randn(*u.shape) for u in tk]
                        dG = [np.random.randn(*g.shape) for g in tt]
                        Jd = sum(np.asarray(replace(x, 'U', i, dU[i]).entries(idx)) for i in range(d)) \
                           + sum(np.asarray(replace(x, 'G', i, dG[i]).entries(idx)) for i in range(d))

                        gU, gG = x.entries_corewise_transpose(c, idx, sum_over_probes=True)
                        self.assertEqual([u.shape for u in tk], [g.shape for g in gU])
                        self.assertEqual([g.shape for g in tt], [g.shape for g in gG])
                        lhs = core_dot(([np.asarray(g) for g in gU], [np.asarray(g) for g in gG]), (dU, dG), nC)
                        rhs = np.sum(c * Jd, axis=tuple(range(nW)))
                        self.check_relerr(rhs, lhs)

                        gUk, gGk = x.entries_corewise_transpose(c, idx)
                        self.assertEqual(W + tk[0].shape, gUk[0].shape)
                        for s, k in zip(gU, gUk):
                            self.check_relerr(np.asarray(s), np.asarray(k).sum(axis=tuple(range(nW))))
                        for s, k in zip(gG, gGk):
                            self.check_relerr(np.asarray(s), np.asarray(k).sum(axis=tuple(range(nW))))

    def test_probe_ambient_transpose(self):
        # ambient probe transpose: literal adjoint of probe (linear in X) -> a rank-d CP back-projection.
        # adjoint identity <probe^T(z), x> == sum_i <z_i, probe(x)_i>; returns CP factors.
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        for BASE in frame_structures:
            shape, _, _ = BASE
            d = len(shape)
            for C in [(), (2, 3)]:
                for W in [(), (5,)]:
                    with self.subTest(BASE=BASE, C=C, W=W):
                        nW, nC, nN = len(W), len(C), d
                        x = t3.TuckerTensorTrain.randn(*(BASE + (C,)))
                        ww = [np.random.randn(*(W + (N,))) for N in shape]
                        zt = [np.random.randn(*(W + C + (N,))) for N in shape]
                        probes = [np.asarray(p) for p in x.probe(ww)]       # d vecs, each W+C+(Ni,)
                        xd = x.to_dense()                                   # C + N
                        # <z, probe(x)> per (W,C): contract each mode over Ni, sum over modes
                        zprobe = sum(np.sum(zt[i] * probes[i], axis=-1) for i in range(d))   # W + C
                        def ndot(Td, lead):  # <Td, xd> over N, x broadcast over the leading `lead` axes
                            xb = xd.reshape((1,) * lead + xd.shape)
                            return np.sum(Td * xb, axis=tuple(range(Td.ndim - nN, Td.ndim)))

                        # primary: W passthrough; CP rank d
                        f = t3.TuckerTensorTrain.probe_ambient_transpose(zt, ww)
                        self.assertEqual(W + C + (d, shape[0]), f[0].shape)
                        T = t3.TuckerTensorTrain.from_canonical(f)
                        self.assertEqual(W + C, T.stack_shape)
                        self.check_relerr(zprobe, ndot(T.to_dense(), nW))

                        # summed: W folds into the CP rank (d|W|); == sum over W of the primary
                        fs = t3.TuckerTensorTrain.probe_ambient_transpose(zt, ww, sum_over_probes=True)
                        self.assertEqual(C + (d * int(np.prod(W, dtype=int)), shape[0]), fs[0].shape)
                        Ts = t3.TuckerTensorTrain.from_canonical(fs)
                        self.assertEqual(C, Ts.stack_shape)
                        lhs = np.sum(Ts.to_dense().reshape(C + (-1,)) * xd.reshape(C + (-1,)), axis=-1)
                        self.check_relerr(np.sum(zprobe, axis=tuple(range(nW))), lhs)
                        self.check_relerr(Ts.to_dense(), T.to_dense().sum(axis=tuple(range(nW))))

    def test_probe_corewise_transpose(self):
        # corewise (Sec 6.3) probe transpose: gradient of probe w.r.t. the cores. Oracle: adjoint
        # identity vs the EXACT forward corewise Jacobian (per mode, sum of single-core replacements).
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
        ]
        def replace(x, kind, i, new):
            tk, tt = [list(cs) for cs in x.data]
            (tk if kind == 'U' else tt)[i] = new
            return t3.TuckerTensorTrain(tuple(tk), tuple(tt))
        def core_dot(gA, gB, nC):
            return sum(np.sum(a * b, axis=tuple(range(nC, a.ndim)))
                       for a, b in zip(gA[0] + gA[1], gB[0] + gB[1]))
        for BASE in frame_structures:
            shape, _, _ = BASE
            d = len(shape)
            for C in [(), (2, 3)]:
                for W in [(), (5,)]:
                    with self.subTest(BASE=BASE, C=C, W=W):
                        nW, nC = len(W), len(C)
                        x = t3.TuckerTensorTrain.randn(*(BASE + (C,)))
                        tk, tt = [list(cs) for cs in x.data]
                        ww = [np.random.randn(*(W + (N,))) for N in shape]
                        zt = [np.random.randn(*(W + C + (N,))) for N in shape]
                        dU = [np.random.randn(*u.shape) for u in tk]
                        dG = [np.random.randn(*g.shape) for g in tt]
                        # exact forward corewise Jacobian per mode: Jm[m] = sum_i probe(replace_i)_m
                        Jm = [np.zeros(W + C + (N,)) for N in shape]
                        for i in range(d):
                            for kind, dd in (('U', dU), ('G', dG)):
                                pr = replace(x, kind, i, dd[i]).probe(ww)
                                for m in range(d):
                                    Jm[m] = Jm[m] + np.asarray(pr[m])
                        # <z, J(dcores)> per (W,C), then sum over W (keep C)
                        zJ = sum(np.sum(zt[m] * Jm[m], axis=-1) for m in range(d))   # W + C
                        rhs = np.sum(zJ, axis=tuple(range(nW)))                      # C

                        gU, gG = x.probe_corewise_transpose(zt, ww, sum_over_probes=True)
                        self.assertEqual([u.shape for u in tk], [g.shape for g in gU])
                        self.assertEqual([g.shape for g in tt], [g.shape for g in gG])
                        lhs = core_dot(([np.asarray(g) for g in gU], [np.asarray(g) for g in gG]), (dU, dG), nC)
                        self.check_relerr(rhs, lhs)

                        # unsummed: W leads each gradient; sum=True == sum_W of sum=False
                        gUk, gGk = x.probe_corewise_transpose(zt, ww)
                        self.assertEqual(W + tk[0].shape, gUk[0].shape)
                        for s, k in zip(gU, gUk):
                            self.check_relerr(np.asarray(s), np.asarray(k).sum(axis=tuple(range(nW))))
                        for s, k in zip(gG, gGk):
                            self.check_relerr(np.asarray(s), np.asarray(k).sum(axis=tuple(range(nW))))

    def test_probe(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
        ]
        stack_shapes = [
            (),
            (2, 3),
        ]
        vecs_stack_shapes = [
            (),
            (5,),
            (2,3),
        ]

        for FRAME_STRUCTURE in frame_structures:
            shape, tucker_ranks, tt_ranks = FRAME_STRUCTURE
            for STACK_SHAPE in stack_shapes:
                for VECS_STACK_SHAPE in vecs_stack_shapes:
                    with self.subTest(
                            FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                            VECS_STACK_SHAPE=VECS_STACK_SHAPE
                    ):
                        x = t3.TuckerTensorTrain.randn(*(FRAME_STRUCTURE + (STACK_SHAPE,)))

                        vecs = [np.random.randn(*(VECS_STACK_SHAPE + (N,))) for N in shape]

                        result = x.probe(vecs)

                        stack_inds      = list(itertools.product(*[tuple(range(s)) for s in STACK_SHAPE]))
                        vecs_stack_inds = list(itertools.product(*[tuple(range(s)) for s in VECS_STACK_SHAPE]))

                        x_dense = x.to_dense()
                        for ind in stack_inds:
                            X = x_dense[ind]
                            for vind in vecs_stack_inds:
                                # probes are stacked F + G (vec stack outer, T3 stack inner)
                                zz = [z[vind+ind] for z in result]
                                vv = [v[vind] for v in vecs]
                                if len(shape) == 1:
                                    zz_true = [
                                        np.einsum('i->i', X)
                                    ]
                                elif len(shape) == 2:
                                    zz_true = [
                                        np.einsum('ij,j->i', X, vv[1]),
                                        np.einsum('ij,i->j', X, vv[0]),
                                    ]
                                elif len(shape) == 3:
                                    zz_true = [
                                        np.einsum('ijk,j,k->i', X, vv[1], vv[2]),
                                        np.einsum('ijk,i,k->j', X, vv[0], vv[2]),
                                        np.einsum('ijk,i,j->k', X, vv[0], vv[1]),
                                    ]
                                elif len(shape) == 4:
                                    zz_true = [
                                        np.einsum('ijkl,j,k,l->i', X, vv[1], vv[2], vv[3]),
                                        np.einsum('ijkl,i,k,l->j', X, vv[0], vv[2], vv[3]),
                                        np.einsum('ijkl,i,j,l->k', X, vv[0], vv[1], vv[3]),
                                        np.einsum('ijkl,i,j,k->l', X, vv[0], vv[1], vv[2]),
                                    ]
                                else:
                                    raise ValueError('shape=' + str(shape))

                                for z, zt in zip(zz, zz_true):
                                    self.check_relerr(zt, z)

    def test_t3svd(self):
        frame_structures = [
            ((8,),          (7,),       (6, 7)),
            ((8, 9, 10),    (7, 8, 9),  (6, 7, 8, 7)),
        ]
        stack_shapes = [
            (),
            (2,3),
        ]

        for FRAME_STRUCTURE in frame_structures:
            shape, tucker_ranks, tt_ranks = FRAME_STRUCTURE

            tucker_ranks_limits = [
                None,
                tuple(1 for _ in range(len(tucker_ranks))),
                (2, 3, 4, 3)[:len(tucker_ranks)],
            ]
            tt_ranks_limits = [
                None,
                tuple(1 for _ in range(len(tt_ranks))),
                (2, 3, 4, 3, 2)[:len(tt_ranks)],
            ]

            for STACK_SHAPE in stack_shapes:
                x = _random_preconditioned_t3(shape, tucker_ranks, tt_ranks, STACK_SHAPE)

                xs = x.squash_tails().to_dense()

                num_stack_dims = len(STACK_SHAPE)

                all_unfolding_ss = []
                for ii in range(len(shape)+1):
                    N = math.prod(shape[:ii])
                    M = math.prod(shape[ii:])
                    xi = xs.reshape(STACK_SHAPE + (N, M))
                    _, ss, _ = np.linalg.svd(xi, full_matrices=False)
                    all_unfolding_ss.append(ss)

                all_matricization_ss = []
                for ii in range(len(shape)):
                    N = shape[ii]
                    M = math.prod(shape[:ii]+shape[ii+1:])
                    xi = np.swapaxes(xs, num_stack_dims, num_stack_dims+ii).reshape(STACK_SHAPE + (N, M))
                    _, ss, _ = np.linalg.svd(xi, full_matrices=False)
                    all_matricization_ss.append(ss)

                if STACK_SHAPE == ():
                    all_tols = [None, 5e-1, 5e-2, 5e-3, 5e-4]
                else:
                    all_tols = [None]

                for RTOL in all_tols:
                    for ATOL in all_tols:
                        for MAX_TUCKER_RANKS in tucker_ranks_limits:
                            for MAX_TT_RANKS in tt_ranks_limits:
                                with self.subTest(
                                        FRAME_STRUCTURE=FRAME_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                                        RTOL=RTOL, ATOL=ATOL,
                                        MAX_TUCKER_RANKS=MAX_TUCKER_RANKS,
                                        MAX_TT_RANKS=MAX_TT_RANKS,
                                ):
                                    x2, ss_tk, ss_tt = x.t3svd(
                                        max_tt_ranks=MAX_TT_RANKS,
                                        max_tucker_ranks=MAX_TUCKER_RANKS,
                                        rtol=RTOL,
                                        atol=ATOL,
                                    )

                                    if (
                                            RTOL is None and ATOL is None and
                                            MAX_TUCKER_RANKS is None and MAX_TT_RANKS is None
                                    ):
                                        self.check_relerr(xs, x2.to_dense())

                                        for sss, sss2 in zip(ss_tk, all_matricization_ss):
                                            self.check_relerr(sss2[..., :sss.shape[-1]], sss)

                                        for sss, sss2 in zip(ss_tt, all_unfolding_ss):
                                            self.check_relerr(sss2[..., :sss.shape[-1]], sss)

                                    if MAX_TUCKER_RANKS is not None:
                                        for n2, n_max in zip(x2.tucker_ranks, MAX_TUCKER_RANKS):
                                            self.assertLessEqual(n2, n_max)

                                    if MAX_TT_RANKS is not None:
                                        for r2, r_max in zip(x2.tt_ranks, MAX_TT_RANKS):
                                            self.assertLessEqual(r2, r_max)

                                    rt = RTOL if RTOL is not None else 0.0
                                    at = ATOL if ATOL is not None else 0.0
                                    # t3svd truncates SEQUENTIALLY, so check rank/error against the ranks it
                                    # ACTUALLY chose (x2.*_ranks), evaluated on the ORIGINAL unfoldings'/
                                    # matricizations' singular values (Oseledets 2011 Thm 2.2, generalized to
                                    # Tucker tensor trains). The error bound uses the chosen ranks; the optimal
                                    # unfolding delta-rank is a provable UPPER bound on the chosen rank (the
                                    # running norm only decreases during the sweep, so rt*||x2|| lower-bounds the
                                    # per-step threshold) -- which guards against passing via huge ranks.
                                    tt_caps = MAX_TT_RANKS if MAX_TT_RANKS is not None else (None,) * len(all_unfolding_ss)
                                    tk_caps = MAX_TUCKER_RANKS if MAX_TUCKER_RANKS is not None else (None,) * len(all_matricization_ss)

                                    stack_inds = list(itertools.product(*[tuple(range(s)) for s in STACK_SHAPE]))

                                    x2_dense = x2.to_dense()
                                    for ind in stack_inds:
                                        rank_thresh = max(rt * np.linalg.norm(np.asarray(x2_dense[ind])), at)

                                        def _esq_and_rank_check(all_ss, actual_ranks, caps):
                                            Esq = []
                                            for sss, r_act, cap in zip(all_ss, actual_ranks, caps):
                                                ss = np.asarray(sss[ind])
                                                Esq.append(float(np.sum(ss[int(r_act):] ** 2)))
                                                tails = np.sqrt(np.cumsum(ss[::-1] ** 2))[::-1]
                                                ub = max(1, int(np.sum(tails >= rank_thresh)))
                                                if cap is not None:
                                                    ub = min(ub, cap)
                                                self.assertLessEqual(int(r_act), ub)
                                            return Esq

                                        unfolding_Esq = _esq_and_rank_check(all_unfolding_ss, x2.tt_ranks, tt_caps)
                                        matricization_Esq = _esq_and_rank_check(all_matricization_ss, x2.tucker_ranks, tk_caps)

                                        error_upper_bound = np.sqrt(np.sum(unfolding_Esq) + np.sum(matricization_Esq))
                                        self.assertLessEqual(
                                            np.linalg.norm(np.asarray(xs[ind]) - np.asarray(x2_dense[ind])),
                                            error_upper_bound + tol * np.linalg.norm(np.asarray(xs[ind]))
                                        )

    def test_t3svd_tols(self):
        structures = [
            ((8,),              (7,),           (6, 7),             ()),
            ((8, 9),            (7, 8),         (6, 7, 8),          ()),
            ((8, 9, 10),        (7, 8, 9),      (6, 7, 8, 7),       ()),
            ((8, 9, 10, 11),    (7, 8, 9, 10),  (6, 7, 8, 7, 5),    ()),
        ]

        for STRUCTURE in structures:
            shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
            resized_tucker_ranks = (max(tucker_ranks)+5,) * len(shape)
            resized_tt_ranks = (max(tt_ranks)+6,) * (len(shape) + 1)

            min_tucker_ranks, min_tt_ranks = t3.TuckerTensorTrain.get_minimal_ranks(shape, tucker_ranks, tt_ranks)

            x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks, stack_shape)

            x = x.resize(x.shape, resized_tucker_ranks, resized_tt_ranks)
            for RTOL, ATOL in [(None, 1e-7), (1e-7, None), (1e-7, 1e-7)]:
                with self.subTest(
                        STRUCTURE=STRUCTURE, RTOL=RTOL, ATOL=ATOL
                ):
                    x2, ss_tk, ss_tt = x.t3svd(rtol=RTOL, atol=ATOL)

                    self.assertEqual(shape, x2.shape)
                    self.assertEqual(min_tucker_ranks, x2.tucker_ranks)
                    self.assertEqual(min_tt_ranks, x2.tt_ranks)

    def test_t3svd_scalar_max_ranks(self):
        # A scalar max-rank caps every position; result is identical to the broadcast list.
        import t3toolbox.backend.ranks as ranks
        self.assertEqual((None, None, None), ranks.normalize_max_ranks(None, 3))
        self.assertEqual((2, 2, 2, 2), ranks.normalize_max_ranks(2, 4))
        self.assertEqual((1, 2, 2, 1), ranks.normalize_max_ranks((1, 2, 2, 1), 4))
        with self.assertRaises(ValueError):
            ranks.normalize_max_ranks((1, 2), 4)

        shape, tr, ttr = (9, 8, 10, 7), (5, 5, 4, 4), (1, 4, 5, 3, 1)
        d = len(shape)
        x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
        a = x.t3svd(max_tucker_ranks=3, max_tt_ranks=2)[0]
        b = x.t3svd(max_tucker_ranks=(3,) * d, max_tt_ranks=(2,) * (d + 1))[0]
        self.assertEqual(a.ranks, b.ranks)
        self.check_relerr(b.to_dense(), a.to_dense())

        T = np.asarray(x.to_dense())
        ad = t3.TuckerTensorTrain.t3svd_dense(T, max_tucker_ranks=3, max_tt_ranks=2)[0]
        bd = t3.TuckerTensorTrain.t3svd_dense(T, max_tucker_ranks=(3,) * d, max_tt_ranks=(2,) * (d + 1))[0]
        self.assertEqual(ad.ranks, bd.ranks)
        self.check_relerr(bd.to_dense(), ad.to_dense())

    def test_t3svd_is_left_orthogonal_not_necessarily_minimal(self):
        # t3svd is the basic algorithm: always left-orthogonal, NOT guaranteed minimal under truncation.
        structures = [
            ((5, 6, 7),       (4, 5, 6),    (1, 3, 2, 1)),
            ((8, 7, 6, 5),    (4, 5, 4, 3), (1, 4, 3, 2, 1)),
            ((9, 2, 9, 4),    (8, 2, 8, 3), (1, 5, 5, 2, 1)),
            ((6, 6, 6),       (10, 10, 10), (1, 12, 12, 1)),  # structurally degenerate (inflated)
        ]
        caps = [None, 1, 2, 3]
        for shape, tr, ttr in structures:
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
            for MAX_TK, MAX_TT in itertools.product(caps, caps):
                with self.subTest(shape=shape, max_tucker=MAX_TK, max_tt=MAX_TT):
                    x2, ss_tk, ss_tt = x.t3svd(max_tucker_ranks=MAX_TK, max_tt_ranks=MAX_TT)
                    self.assertTrue(x2.is_left_orthogonal().all())                       # always left-orthogonal
                    self.assertEqual(tuple(s.shape[-1] for s in ss_tk), x2.tucker_ranks)
                    self.assertEqual(tuple(s.shape[-1] for s in ss_tt), x2.tt_ranks)

    def test_rank_adjustment_sweep(self):
        # rank_adjustment_sweep drops redundant ranks losslessly: a single sweep in the direction
        # matching the input's orthogonality reaches minimal. A t3svd output is left-orthogonal, so
        # 'right_to_left' minimizes it (-> right-orthogonal); composing both gives minimal left-orthogonal.
        structures = [
            ((5, 6, 7),       (4, 5, 6),    (1, 3, 2, 1)),
            ((8, 7, 6, 5),    (4, 5, 4, 3), (1, 4, 3, 2, 1)),
            ((9, 2, 9, 4),    (8, 2, 8, 3), (1, 5, 5, 2, 1)),
            ((10, 10, 10),    (9, 9, 9),    (1, 9, 9, 1)),
        ]
        cap_patterns = [(None, 2), (3, 2), (2, 2), (2, None), ([9, 1, 9], [1, 9, 2, 1])]  # incl bond-orphan
        for shape, tr, ttr in structures:
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
            for MAX_TK, MAX_TT in cap_patterns:
                if isinstance(MAX_TK, list) and len(MAX_TK) != len(shape):
                    continue
                with self.subTest(shape=shape, max_tucker=MAX_TK, max_tt=MAX_TT):
                    x2, _, _ = x.t3svd(max_tucker_ranks=MAX_TK, max_tt_ranks=MAX_TT)  # left-orth, maybe non-min
                    rl = x2.rank_adjustment_sweep('right_to_left')    # left-orth input -> R->L minimizes
                    self.assertTrue(rl.has_minimal_ranks)
                    self.assertTrue(rl.is_right_orthogonal().all())
                    self.check_relerr(x2.to_dense(), rl.to_dense())   # lossless
                    both = rl.rank_adjustment_sweep('left_to_right')  # right-orth input -> L->R -> minimal left-orth
                    self.assertTrue(both.has_minimal_ranks)
                    self.assertTrue(both.is_left_orthogonal().all())
                    self.check_relerr(x2.to_dense(), both.to_dense())
        with self.assertRaises(ValueError):
            x.rank_adjustment_sweep('sideways')

    def test_t3svd_lossless_compression_of_degenerate(self):
        # A generic (i.i.d.-filled) T3 with inflated/degenerate core shapes compresses to minimal ranks
        # with ZERO loss under no-truncation t3svd (every SVD keeps exactly the structural rank).
        rng = np.random.default_rng(0)
        for _ in range(40):
            d = int(rng.integers(2, 5))
            shape = tuple(int(v) for v in rng.integers(2, 8, d))
            tr = tuple(int(v) for v in rng.integers(1, 10, d))             # inflated Tucker ranks
            ttr = tuple([1] + [int(v) for v in rng.integers(1, 13, d - 1)] + [1])  # inflated bonds
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
            x2, _, _ = x.t3svd()
            with self.subTest(shape=shape, tucker=tr, tt=ttr):
                self.assertTrue(x2.has_minimal_ranks)  # no-truncation t3svd IS minimal
                self.assertTrue(x2.is_left_orthogonal().all())
                self.check_relerr(x.to_dense(), x2.to_dense())

    def test_is_left_right_orthogonal_checkers(self):
        import t3toolbox.backend.t3_orthogonalization as orthx
        import t3toolbox.backend.tt_orthogonalization as orth
        for shape, tr, ttr in [((6, 7, 8), (5, 6, 7), (1, 4, 3, 1)),
                               ((5, 6, 7, 4), (4, 5, 6, 3), (1, 3, 4, 2, 1))]:
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
            with self.subTest(shape=shape):
                # a random T3 is in neither orthogonal form
                self.assertFalse(x.is_left_orthogonal().all())
                self.assertFalse(x.is_right_orthogonal().all())
                # build the two forms via the backend orthogonalizers
                tk, tt = orthx.t3_down_orthogonalize_tucker_cores(x.data)
                xL = t3.TuckerTensorTrain(tk, orth.tt_left_orthogonalize(tt))
                xR = t3.TuckerTensorTrain(tk, orth.tt_right_orthogonalize(tt))
                self.assertTrue(xL.is_left_orthogonal().all())
                self.assertFalse(xL.is_right_orthogonal().all())
                self.assertTrue(xR.is_right_orthogonal().all())
                self.assertFalse(xR.is_left_orthogonal().all())
                # a t3svd result is left-orthogonal
                x2, _, _ = x.t3svd()
                self.assertTrue(x2.is_left_orthogonal().all())

    def test_t3svd_assume_orthogonal(self):
        # assume_orthogonal=True (input already right-orthogonal) skips the redundant orthogonalization
        # and gives the same result as the default.
        structures = [((6, 7, 8), (5, 6, 7), (1, 4, 3, 1)),
                      ((5, 6, 7, 4), (4, 5, 6, 3), (1, 3, 4, 2, 1))]
        for shape, tr, ttr in structures:
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
            xR = x.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores()  # right-orthogonal
            self.assertTrue(xR.is_right_orthogonal().all())
            for MAX_TT in [None, 2, 3]:
                with self.subTest(shape=shape, max_tt=MAX_TT):
                    a, _, _ = xR.t3svd(max_tt_ranks=MAX_TT, assume_orthogonal=True)
                    b, _, _ = xR.t3svd(max_tt_ranks=MAX_TT)
                    self.assertEqual(a.ranks, b.ranks)
                    self.check_relerr(b.to_dense(), a.to_dense())
                    self.assertTrue(a.is_left_orthogonal().all())

    def test_compute_minimal_ranks_matches_matricization(self):
        # compute_minimal_ranks must equal the GENERIC numerical rank of every tensor-network edge cut:
        # Tucker edges <-> mode-i matricizations, TT bonds <-> contiguous-split unfoldings. (The T3
        # network is a tree, so each single-edge cut is a clean bipartition with no hidden degeneracy.)
        import t3toolbox.backend.ranks as ranks

        def numerical_rank(M, rel=1e-9):
            s = np.linalg.svd(M, compute_uv=False)
            return int((s > rel * s[0]).sum()) if s.size and s[0] > 0 else 0

        structures = [
            ((5, 6, 7),    (4, 5, 6),    (1, 3, 2, 1)),
            ((6, 7, 8, 5), (6, 7, 8, 5), (1, 9, 9, 9, 1)),  # over-declared bonds
            ((10, 3, 10),  (8, 3, 8),    (1, 7, 2, 1)),
            ((4, 4, 4, 4), (4, 4, 4, 4), (1, 2, 8, 2, 1)),  # one fat interior bond
        ]
        for shape, tr, ttr in structures:
            d = len(shape)
            T = np.asarray(t3.TuckerTensorTrain.randn(shape, tr, ttr).to_dense())
            min_tk, min_tt = ranks.compute_minimal_ranks(shape, tr, ttr)
            min_tk = tuple(int(v) for v in min_tk)
            min_tt = tuple(int(v) for v in min_tt)
            tk_num = tuple(numerical_rank(np.moveaxis(T, i, 0).reshape(shape[i], -1)) for i in range(d))
            tt_num = (1,) + tuple(numerical_rank(T.reshape(int(np.prod(shape[:k])), -1))
                                  for k in range(1, d)) + (1,)
            with self.subTest(shape=shape):
                self.assertEqual(min_tk, tk_num)
                self.assertEqual(min_tt, tt_num)

    def test_compute_minimal_ranks_inequalities(self):
        # The minimal ranks satisfy the no-redundancy inequalities at every core:
        #   n <= N,  n <= rL*rR,  rL <= n*rR,  rR <= n*rL.
        import t3toolbox.backend.ranks as ranks
        rng = np.random.default_rng(3)
        for _ in range(1000):
            d = int(rng.integers(2, 6))
            shape = tuple(int(v) for v in rng.integers(2, 10, d))
            tr = tuple(int(v) for v in rng.integers(1, 12, d))
            ttr = tuple([1] + [int(v) for v in rng.integers(1, 15, d - 1)] + [1])
            n, r = ranks.compute_minimal_ranks(shape, tr, ttr)
            n = [int(v) for v in n]
            r = [int(v) for v in r]
            for i in range(d):
                rL, rR = r[i], r[i + 1]
                with self.subTest(shape=shape, tucker=tr, tt=ttr, core=i):
                    self.assertLessEqual(n[i], shape[i])
                    self.assertLessEqual(n[i], rL * rR)
                    self.assertLessEqual(rL, n[i] * rR)
                    self.assertLessEqual(rR, n[i] * rL)

    def test_t3svd_dense(self):
        shapes = [
            (8,),
            (8, 9),
            (8, 9, 10),
        ]
        stack_shapes = [
            (),
            (2,3),
        ]

        for SHAPE in shapes:
            tucker_ranks_limits = [
                None,
                tuple(1 for _ in range(len(SHAPE))),
                (2, 3, 4, 3)[:len(SHAPE)],
            ]
            tt_ranks_limits = [
                None,
                tuple(1 for _ in range(len(SHAPE)+1)),
                (2, 3, 4, 3, 2)[:len(SHAPE)+1],
            ]

            for STACK_SHAPE in stack_shapes:
                num_stack_dims = len(STACK_SHAPE)
                X = np.random.randn(*(STACK_SHAPE + SHAPE))
                for ii, N in enumerate(SHAPE):
                    c = 1.0 / (1.0 + np.arange(N))**2
                    ax = num_stack_dims + ii
                    X = np.einsum('...i,i->...i', X.swapaxes(ax, -1), c).swapaxes(-1, ax)

                all_unfolding_ss = []
                for ii in range(len(SHAPE)+1):
                    N = math.prod(SHAPE[:ii])
                    M = math.prod(SHAPE[ii:])
                    XI = X.reshape(STACK_SHAPE + (N, M))
                    _, ss, _ = np.linalg.svd(XI, full_matrices=False)
                    all_unfolding_ss.append(ss)

                all_matricization_ss = []
                for ii in range(len(SHAPE)):
                    N = SHAPE[ii]
                    M = math.prod(SHAPE[:ii]+SHAPE[ii+1:])
                    XI = np.swapaxes(X, num_stack_dims, num_stack_dims+ii).reshape(STACK_SHAPE + (N, M))
                    _, ss, _ = np.linalg.svd(XI, full_matrices=False)
                    all_matricization_ss.append(ss)

                if STACK_SHAPE == ():
                    all_tols = [None, 5e-1, 5e-2, 5e-3, 5e-4]
                else:
                    all_tols = [None]

                for RTOL in all_tols:
                    for ATOL in all_tols:
                        for MAX_TUCKER_RANKS in tucker_ranks_limits:
                            for MAX_TT_RANKS in tt_ranks_limits:
                                with self.subTest(
                                        SHAPE=SHAPE, STACK_SHAPE=STACK_SHAPE,
                                        RTOL=RTOL, ATOL=ATOL,
                                        MAX_TUCKER_RANKS=MAX_TUCKER_RANKS,
                                        MAX_TT_RANKS=MAX_TT_RANKS,
                                ):
                                    x2, ss_tk, ss_tt = t3.TuckerTensorTrain.t3svd_dense(
                                        X,
                                        stack_shape=STACK_SHAPE,
                                        max_tt_ranks=MAX_TT_RANKS,
                                        max_tucker_ranks=MAX_TUCKER_RANKS,
                                        rtol=RTOL,
                                        atol=ATOL,
                                    )

                                    if (
                                            RTOL is None and ATOL is None and
                                            MAX_TUCKER_RANKS is None and MAX_TT_RANKS is None
                                    ):
                                        self.check_relerr(X, x2.to_dense())

                                        for sss, sss2 in zip(ss_tk, all_matricization_ss):
                                            self.check_relerr(sss2[..., :sss.shape[-1]], sss)

                                        for sss, sss2 in zip(ss_tt, all_unfolding_ss):
                                            self.check_relerr(sss2[..., :sss.shape[-1]], sss)

                                    if MAX_TUCKER_RANKS is not None:
                                        for n2, n_max in zip(x2.tucker_ranks, MAX_TUCKER_RANKS):
                                            self.assertLessEqual(n2, n_max)

                                    if MAX_TT_RANKS is not None:
                                        for r2, r_max in zip(x2.tt_ranks, MAX_TT_RANKS):
                                            self.assertLessEqual(r2, r_max)

                                    # The dense T3-SVD (Algorithm 9) truncates SEQUENTIALLY on the reduced
                                    # tensor, so the rank/error must be checked against the ranks it
                                    # ACTUALLY chose (x2.tucker_ranks / x2.tt_ranks), evaluated using the
                                    # ORIGINAL unfoldings'/matricizations' singular values. This is the
                                    # Oseledets (2011, Theorem 2.2) quasi-optimality bound generalized to
                                    # Tucker tensor trains (matricization + unfolding tails in quadrature).
                                    rt = RTOL if RTOL is not None else 0.0
                                    at = ATOL if ATOL is not None else 0.0
                                    # Max-rank caps (None = no cap), used to bound the chosen ranks below.
                                    tt_caps = MAX_TT_RANKS if MAX_TT_RANKS is not None else (None,) * len(all_unfolding_ss)
                                    tk_caps = MAX_TUCKER_RANKS if MAX_TUCKER_RANKS is not None else (None,) * len(all_matricization_ss)

                                    stack_inds = list(itertools.product(*[tuple(range(s)) for s in STACK_SHAPE]))

                                    x2_dense = x2.to_dense()
                                    for ind in stack_inds:
                                        # ||x2|| <= every reduced norm during the sweep (the running norm only
                                        # decreases), so rtol*||x2|| is a lower bound on t3svd's per-step
                                        # threshold -- hence the delta-rank of the ORIGINAL unfolding at this
                                        # threshold is a provable UPPER bound on the rank t3svd can choose.
                                        rank_thresh = max(rt * np.linalg.norm(np.asarray(x2_dense[ind])), at)

                                        def _esq_and_rank_check(all_ss, actual_ranks, caps):
                                            Esq = []
                                            for sss, r_act, cap in zip(all_ss, actual_ranks, caps):
                                                ss = np.asarray(sss[ind])
                                                # error: original tail energy beyond the ACTUAL rank
                                                Esq.append(float(np.sum(ss[int(r_act):] ** 2)))
                                                # rank: actual rank must not exceed the delta-rank of the
                                                # original unfolding (floored at 1, the min core rank; capped
                                                # at the max rank). Guards against passing via huge ranks.
                                                tails = np.sqrt(np.cumsum(ss[::-1] ** 2))[::-1]
                                                ub = max(1, int(np.sum(tails >= rank_thresh)))
                                                if cap is not None:
                                                    ub = min(ub, cap)
                                                self.assertLessEqual(int(r_act), ub)
                                            return Esq

                                        unfolding_Esq = _esq_and_rank_check(all_unfolding_ss, x2.tt_ranks, tt_caps)
                                        matricization_Esq = _esq_and_rank_check(all_matricization_ss, x2.tucker_ranks, tk_caps)

                                        error_upper_bound = np.sqrt(np.sum(unfolding_Esq) + np.sum(matricization_Esq))
                                        self.assertLessEqual(
                                            np.linalg.norm(np.asarray(X[ind]) - np.asarray(x2_dense[ind])),
                                            error_upper_bound + tol * np.linalg.norm(np.asarray(X[ind]))
                                        )

    def test_t3svd_dense_tols(self):
        structures = [
            ((8,),              (7,),           (6, 7),             ()),
            ((8, 9),            (7, 8),         (6, 7, 8),          ()),
            ((8, 9, 10),        (7, 8, 9),      (6, 7, 8, 7),       ()),
            ((8, 9, 10, 11),    (7, 8, 9, 10),  (6, 7, 8, 7, 5),    ()),
        ]

        for STRUCTURE in structures:
            shape, tucker_ranks, tt_ranks, stack_shape = STRUCTURE
            resized_tucker_ranks = (max(tucker_ranks)+5,) * len(shape)
            resized_tt_ranks = (max(tt_ranks)+6,) * (len(shape) + 1)

            min_tucker_ranks, min_tt_ranks = t3.TuckerTensorTrain.get_minimal_ranks(shape, tucker_ranks, tt_ranks)

            x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks, stack_shape)

            x = x.resize(x.shape, resized_tucker_ranks, resized_tt_ranks)
            X = x.to_dense()

            for RTOL, ATOL in [(None, 1e-7), (1e-7, None), (1e-7, 1e-7)]:
                with self.subTest(
                        STRUCTURE=STRUCTURE, RTOL=RTOL, ATOL=ATOL
                ):
                    x2, ss_tk, ss_tt = t3.TuckerTensorTrain.t3svd_dense(X, rtol=RTOL, atol=ATOL)

                    self.assertEqual(shape, x2.shape)
                    self.assertEqual(min_tucker_ranks, x2.tucker_ranks)
                    self.assertEqual(min_tt_ranks, x2.tt_ranks)

    def test_get_minimal_ranks(self):
        mr = t3.TuckerTensorTrain.get_minimal_ranks((10, 11, 12, 13), (14, 15, 16, 17), (98, 99, 100, 101, 102))

        mr_true = ((10, 11, 12, 13), (1, 10, 100, 13, 1))
        self.assertEqual(mr, mr_true)

    def test_get_core_shapes(self):
        frame_structures = [
            ((8,),              (4,),           (4, 5)),
            ((8, 9),            (4, 5),         (4, 5, 4)),
            ((8, 9, 10),        (4, 5, 6),      (4, 5, 4, 3)),
            ((8, 9, 10, 11),    (4, 5, 6, 7),   (4, 5, 4, 3, 3)),
        ]
        stack_shapes = [
            (),
            (2, 3),
        ]

        for FRAME_STRUCTURE in frame_structures:
            shape, tucker_ranks, tt_ranks = FRAME_STRUCTURE

            for STACK_SHAPE in stack_shapes:
                x = t3.TuckerTensorTrain.zeros(shape, tucker_ranks, tt_ranks, STACK_SHAPE)

                tucker_shapes, tt_shapes = t3.TuckerTensorTrain.get_core_shapes(shape, tucker_ranks, tt_ranks, STACK_SHAPE)
                self.assertEqual(tuple(B.shape for B in x.tucker_cores), tucker_shapes)
                self.assertEqual(tuple(G.shape for G in x.tt_cores), tt_shapes)


    def test_derivative_methods(self):
        # TuckerTensorTrain.{probe,apply,entries}_derivatives + corewise transposes: order-0 == the
        # plain op; the corewise gradient matches a finite difference; the X/P consistency check raises.
        STRUCT = ((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        shapes = STRUCT[0]
        d = len(shapes)
        x = t3.TuckerTensorTrain.randn(*STRUCT)
        W = (2,)
        ww = [np.random.randn(*(W + (N,))) for N in shapes]
        pp = [np.random.randn(*(W + (N,))) for N in shapes]
        index = np.stack([np.random.randint(0, N, size=W) for N in shapes], axis=0)
        ORDER = 3

        # order 0 == the non-derivative op
        for zj, z0 in zip(x.probe_derivatives(ww, pp, ORDER), x.probe(ww)):
            self.check_relerr(np.asarray(z0), np.asarray(zj)[0])
        self.check_relerr(np.asarray(x.apply(ww)), np.asarray(x.apply_derivatives(ww, pp, ORDER))[0])
        self.check_relerr(np.asarray(x.entries(index)), np.asarray(x.entries_derivatives(index, pp, ORDER))[0])

        # corewise transpose: <g, dcores> matches a central finite difference of <r, forward(cores)>
        r = [np.random.randn(*np.asarray(z).shape) for z in x.probe_derivatives(ww, pp, ORDER)]
        gU, gG = x.probe_corewise_derivatives_transpose(r, ww, pp, ORDER, sum_over_probes=True)
        dU = [np.random.randn(*B.shape) for B in x.tucker_cores]
        dG = [np.random.randn(*G.shape) for G in x.tt_cores]
        inner = (sum(np.sum(np.asarray(gU[i]) * dU[i]) for i in range(d))
                 + sum(np.sum(np.asarray(gG[i]) * dG[i]) for i in range(d)))
        eps = 1e-6
        dot = lambda data: sum(np.sum(r[i] * np.asarray(sampling_derivatives.t3_probe_derivatives(ww, pp, data, ORDER)[i]))
                               for i in range(d))
        plus = ([B + eps * du for B, du in zip(x.tucker_cores, dU)], [G + eps * dg for G, dg in zip(x.tt_cores, dG)])
        minus = ([B - eps * du for B, du in zip(x.tucker_cores, dU)], [G - eps * dg for G, dg in zip(x.tt_cores, dG)])
        fd = (dot(plus) - dot(minus)) / (2 * eps)
        self.assertLessEqual(abs(inner - fd) / max(abs(fd), 1e-30), 1e-5)

        # X/P sample-stack consistency: hard error
        pp_bad = [np.random.randn(*((3,) + (N,))) for N in shapes]
        with self.assertRaises(ValueError):
            x.probe_derivatives(ww, pp_bad, ORDER)
        with self.assertRaises(ValueError):
            x.entries_derivatives(index, pp_bad, ORDER)

    def test_corewise_derivative_transpose_nontrivial_boundary(self):
        # Regression: the adjoint-state apply/entries (derivative) transpose seeds the reverse sigma_hat
        # sweep at the terminal bond, which the corewise (U,G,G,G) substitution makes != 1 (the forward
        # SUMS that bond), so the seed must broadcast c over rR_d. (Probe is unaffected -- no scalar seed.)
        np.random.seed(0)
        shapes = (6, 7, 8); ORDER = 2; W = (2,); d = len(shapes); eps = 1e-6
        x = t3.TuckerTensorTrain.randn(shapes, (3, 4, 3), (1, 3, 2, 3))   # rR_d = 3, not 1
        ww = [np.random.randn(*(W + (N,))) for N in shapes]
        pp = [np.random.randn(*(W + (N,))) for N in shapes]
        index = np.stack([np.random.randint(0, N, size=W) for N in shapes], axis=0)
        cases = [
            ('apply', lambda rr: x.apply_corewise_derivatives_transpose(rr, ww, pp, ORDER, sum_over_probes=True),
                      lambda data: sampling_derivatives.t3_apply_derivatives(ww, pp, data, ORDER)),
            ('entries', lambda rr: x.entries_corewise_derivatives_transpose(rr, index, pp, ORDER, sum_over_probes=True),
                        lambda data: sampling_derivatives.t3_entries_derivatives(index, pp, data, ORDER)),
        ]
        for name, transpose, forward in cases:
            with self.subTest(op=name):
                r = np.random.randn(*np.asarray(forward(x.data)).shape)
                gU, gG = transpose(r)
                dU = [np.random.randn(*B.shape) for B in x.tucker_cores]
                dG = [np.random.randn(*G.shape) for G in x.tt_cores]
                inner = (sum(np.sum(np.asarray(gU[i]) * dU[i]) for i in range(d))
                         + sum(np.sum(np.asarray(gG[i]) * dG[i]) for i in range(d)))
                dot = lambda data: float(np.sum(r * np.asarray(forward(data))))
                plus = ([B + eps * du for B, du in zip(x.tucker_cores, dU)], [G + eps * dg for G, dg in zip(x.tt_cores, dG)])
                minus = ([B - eps * du for B, du in zip(x.tucker_cores, dU)], [G - eps * dg for G, dg in zip(x.tt_cores, dG)])
                fd = (dot(plus) - dot(minus)) / (2 * eps)
                self.assertLessEqual(abs(inner - fd) / max(abs(fd), 1e-30), 1e-5)


class TestEntriesIndexSemantics(unittest.TestCase):
    """Review 2026-08-22 (S5): all four entries ops follow numpy index semantics -- the ambient transpose
    used to build its one-hots with ``arange(N) == idx``, which matched nothing for a negative index and
    silently returned zero factors (breaking its defining adjoint identity)."""

    def test_negative_indices_agree_across_all_entries_ops(self):
        np.random.seed(11)
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
        xd = np.asarray(x.to_dense())
        for index in ([-1, -2, -3], [[-1, 2], [0, -6], [3, -1]]):     # single and W-stacked indices
            with self.subTest(index=index):
                idx = np.array(index)
                e = np.asarray(x.entries(idx))
                ref = xd[tuple(idx)]
                self.assertLess(norm(e - ref) / max(1.0, norm(ref)), 1e-12)
                c = np.random.randn(*e.shape)
                # ambient transpose: <E^T c, X> == <c, E X>  (the adjoint identity it used to fail)
                T = t3.TuckerTensorTrain.from_canonical(x.entries_ambient_transpose(c, idx, x.shape))
                lhs = float(np.sum(np.asarray(T.to_dense()) * xd))
                self.assertLess(abs(lhs - float(np.sum(c * e))) / max(1.0, abs(lhs)), 1e-12)
                # corewise transpose: negative indices == the wrapped positive ones, core by core
                g_neg = x.entries_corewise_transpose(c, idx)
                g_pos = x.entries_corewise_transpose(c, idx % np.array(x.shape)[:, None] if idx.ndim > 1
                                                     else idx % np.array(x.shape))
                for a, b in zip(g_neg[0] + g_neg[1], g_pos[0] + g_pos[1]):
                    self.assertLess(norm(np.asarray(a) - np.asarray(b)), 1e-12)

    def test_structural_checks_are_errors_not_asserts(self):
        # Review 2026-08-22 (S13): structural problems hard-error in every mode -- including `python -O`,
        # which strips `assert` (a wrong-shaped `x * ndarray` used to broadcast silently there).
        import subprocess, sys
        x = t3.TuckerTensorTrain.randn((4, 5), (2, 2), (1, 2, 1))
        with self.assertRaises(ValueError):
            x * np.ones((1, 5))
        xs = t3.TuckerTensorTrain.randn((4, 5), (2, 2), (1, 2, 1), stack_shape=(2,))
        with self.assertRaises(ValueError):
            xs.sum_stack(axis=5)
        with self.assertRaises(ValueError):
            cw.corewise_add(x.tucker_cores, x.tucker_cores[:1])
        code = ('import numpy as np, t3toolbox.tucker_tensor_train as t3\n'
                'x = t3.TuckerTensorTrain.randn((4, 5), (2, 2), (1, 2, 1))\n'
                'try:\n    x * np.ones((1, 5))\nexcept ValueError:\n    print("RAISED")\n')
        out = subprocess.run([sys.executable, '-O', '-c', code], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': os.getcwd()})
        self.assertIn('RAISED', out.stdout, out.stderr[-500:])

    def test_python_float_residual(self):
        # Review 2026-08-22 (C8): the unstacked apply/entries residual is a bare float; the tangent and
        # corewise transposes used to die on `c[..., None]` (the ambient twin's doctest passes 1.7).
        import t3toolbox.manifold as t3m
        np.random.seed(12)
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
        ww = tuple(np.random.randn(n) for n in x.shape)
        idx = np.array([1, 2, 3])
        frame = t3m.MANIFOLD.frame(x)
        for c in (1.7, np.float64(1.7), np.asarray(1.7)):
            with self.subTest(type=type(c).__name__):
                ga = x.apply_corewise_transpose(c, ww)
                ge = x.entries_corewise_transpose(c, idx)
                ta = t3m.T3Tangent.apply_transpose(c, ww, frame)
                te = t3m.T3Tangent.entries_transpose(c, idx, frame)
                ref = x.apply_corewise_transpose(np.asarray(1.7), ww)
                for a, b in zip(ga[0] + ga[1], ref[0] + ref[1]):
                    self.assertLess(norm(np.asarray(a) - np.asarray(b)), 1e-12)
                self.assertEqual(len(ge), 2)
                self.assertEqual(ta.frame, frame)
                self.assertEqual(te.frame, frame)


if __name__ == '__main__':
    unittest.main()

