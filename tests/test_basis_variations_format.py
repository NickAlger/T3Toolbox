# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
import numpy as np
import unittest

import t3toolbox.basis_variations_format as bvf
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.corewise as cw

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


def _random_basis_variations(structure, use_jax=False):
    """Build a consistent (T3Basis, T3Variations) pair from a rank spec.

    structure = (shape, up_ranks, down_ranks, left_ranks, right_ranks, stack_shape)
        up_ranks, down_ranks:  len=d
        left_ranks, right_ranks: len=d+1
    Core shapes (with leading stack_shape):
        U_i = (nU_i, N_i)           L_i = (rL_i, nU_i, rL_(i+1))
        R_i = (rR_i, nU_i, rR_(i+1))  D_i = (rL_i, nD_i, rR_(i+1))
        V_i = (nD_i, N_i)           H_i = (rL_i, nU_i, rR_(i+1))
    """
    shape, up_ranks, down_ranks, left_ranks, right_ranks, stack_shape = structure
    rnd = (lambda *s: jnp.array(np.random.randn(*s))) if use_jax else (lambda *s: np.random.randn(*s))

    U = tuple(rnd(*(stack_shape + (nU, N)))       for nU, N       in zip(up_ranks, shape))
    L = tuple(rnd(*(stack_shape + (rL, nU, rLn))) for rL, nU, rLn in zip(left_ranks[:-1], up_ranks, left_ranks[1:]))
    R = tuple(rnd(*(stack_shape + (rR, nU, rRn))) for rR, nU, rRn in zip(right_ranks[:-1], up_ranks, right_ranks[1:]))
    D = tuple(rnd(*(stack_shape + (rL, nD, rRn))) for rL, nD, rRn in zip(left_ranks[:-1], down_ranks, right_ranks[1:]))
    V = tuple(rnd(*(stack_shape + (nD, N)))       for nD, N       in zip(down_ranks, shape))
    H = tuple(rnd(*(stack_shape + (rL, nU, rRn))) for rL, nU, rRn in zip(left_ranks[:-1], up_ranks, right_ranks[1:]))

    return bvf.T3Basis(U, D, L, R), bvf.T3Variations(V, H)


def _good_basis_cores():
    """A small, valid set of (up, down, left, right) cores for d=2, no stacking."""
    U = (randn(4, 14), randn(5, 15))            # (nU_i, N_i)
    L = (randn(1, 4, 2), randn(2, 5, 1))        # (rL_i, nU_i, rL_(i+1))
    R = (randn(1, 4, 2), randn(2, 5, 1))        # (rR_i, nU_i, rR_(i+1))
    D = (randn(1, 3, 2), randn(2, 4, 1))        # (rL_i, nD_i, rR_(i+1))
    return U, D, L, R


class TestBasisVariationsFormat(unittest.TestCase):
    base_structures = [
        #  (shape,             up_ranks,      down_ranks,    left_ranks,        right_ranks)
        ((14,),                (4,),          (3,),          (1, 1),            (1, 1)),
        ((14, 15),             (4, 5),        (3, 4),        (1, 2, 1),         (1, 2, 1)),
        ((14, 15, 16),         (4, 5, 6),     (3, 4, 5),     (1, 2, 3, 1),      (1, 3, 2, 1)),
        ((10, 11, 12, 13),     (2, 3, 4, 3),  (2, 2, 3, 2),  (1, 2, 3, 2, 1),   (1, 2, 2, 3, 1)),
    ]
    stack_shapes = [(), (2,), (2, 3)]

    def _equal_cores(self, AA, BB):
        self.assertEqual(len(AA), len(BB))
        for A, B in zip(AA, BB):
            self.assertTrue(np.array_equal(np.asarray(A), np.asarray(B)))

    @staticmethod
    def _expected_variation_shapes(structure):
        shape, up_ranks, down_ranks, left_ranks, right_ranks = structure[:5]
        tucker_variation_shapes = tuple(zip(down_ranks, shape))
        tt_variation_shapes = tuple(zip(left_ranks[:-1], up_ranks, right_ranks[1:]))
        return tucker_variation_shapes, tt_variation_shapes

    def test_t3basis_properties(self):
        for BASE_STRUCTURE in self.base_structures:
            shape, up_ranks, down_ranks, left_ranks, right_ranks = BASE_STRUCTURE
            for STACK_SHAPE in self.stack_shapes:
                structure = BASE_STRUCTURE + (STACK_SHAPE,)
                for USE_JAX in [False, True]:
                    with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                        base, _ = _random_basis_variations(structure, use_jax=USE_JAX)

                        self.assertEqual(len(shape), base.d)
                        self.assertEqual(shape, base.shape)
                        self.assertEqual(up_ranks, base.up_ranks)
                        self.assertEqual(down_ranks, base.down_ranks)
                        self.assertEqual(left_ranks, base.left_ranks)
                        self.assertEqual(right_ranks, base.right_ranks)
                        self.assertEqual(STACK_SHAPE, base.stack_shape)
                        self.assertEqual(structure, base.structure)
                        self.assertEqual(self._expected_variation_shapes(structure), base.variation_shapes)
                        self.assertEqual((base.up_tucker_cores, base.down_tt_cores,
                                          base.left_tt_cores, base.right_tt_cores), base.data)

    def test_t3variations_properties(self):
        for BASE_STRUCTURE in self.base_structures:
            shape, up_ranks, down_ranks, left_ranks, right_ranks = BASE_STRUCTURE
            for STACK_SHAPE in self.stack_shapes:
                structure = BASE_STRUCTURE + (STACK_SHAPE,)
                for USE_JAX in [False, True]:
                    with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                        _, variations = _random_basis_variations(structure, use_jax=USE_JAX)

                        self.assertEqual(len(shape), variations.d)
                        self.assertEqual(shape, variations.shape)
                        self.assertEqual(STACK_SHAPE, variations.stack_shape)
                        self.assertEqual(self._expected_variation_shapes(structure), variations.variation_shapes)
                        self.assertEqual((variations.tucker_variations, variations.tt_variations), variations.data)

    def test_t3basis_validate_raises(self):
        # Each corruption introduces exactly one inconsistency into an otherwise-valid set of cores.
        corruptions = [
            ("wrong number of cores",   lambda U, D, L, R: (U[:-1], D, L, R)),
            ("up core not a matrix",    lambda U, D, L, R: ((randn(4),) + U[1:], D, L, R)),
            ("left core not 3-tensor",  lambda U, D, L, R: (U, D, (randn(1, 4),) + L[1:], R)),
            ("tucker rank mismatch",    lambda U, D, L, R: ((randn(99, 14),) + U[1:], D, L, R)),
            ("down-left rank mismatch", lambda U, D, L, R: (U, (randn(99, 3, 2),) + D[1:], L, R)),
            ("down-right rank mismatch", lambda U, D, L, R: (U, (randn(1, 3, 99),) + D[1:], L, R)),
            ("left chain inconsistent", lambda U, D, L, R: (U, D, (L[0], randn(99, 5, 1)), R)),
            ("right chain inconsistent", lambda U, D, L, R: (U, D, L, (R[0], randn(99, 5, 1)))),
            ("inconsistent stack shape", lambda U, D, L, R: (U, D, (randn(2, 1, 4, 2),) + L[1:], R)),
        ]
        for label, corrupt in corruptions:
            with self.subTest(corruption=label):
                U, D, L, R = _good_basis_cores()
                with self.assertRaises(ValueError):
                    bvf.T3Basis(*corrupt(U, D, L, R))

    def test_t3basis_validate_accepts_good(self):
        U, D, L, R = _good_basis_cores()
        bvf.T3Basis(U, D, L, R)  # must not raise

    def test_t3variations_validate_raises(self):
        def good():
            V = (randn(3, 14), randn(4, 15))         # (nD_i, N_i)
            H = (randn(1, 4, 2), randn(2, 5, 1))     # (rL_i, nU_i, rR_(i+1))
            return V, H

        corruptions = [
            ("wrong number of cores",    lambda V, H: (V[:-1], H)),
            ("tucker variation not matrix", lambda V, H: ((randn(3),) + V[1:], H)),
            ("tt variation not 3-tensor", lambda V, H: (V, (randn(1, 4),) + H[1:])),
            ("inconsistent stack shape", lambda V, H: ((randn(2, 3, 14),) + V[1:], H)),
        ]
        for label, corrupt in corruptions:
            with self.subTest(corruption=label):
                V, H = good()
                with self.assertRaises(ValueError):
                    bvf.T3Variations(*corrupt(V, H))

    def test_check_bv_pair(self):
        structure = ((14, 15, 16), (4, 5, 6), (3, 4, 5), (1, 2, 3, 1), (1, 3, 2, 1), ())
        base, variations = _random_basis_variations(structure)
        bvf.check_bv_pair(base, variations)  # consistent: must not raise

        V, H = variations.tucker_variations, variations.tt_variations

        # Tucker variation does not fit a base hole
        badV = (randn(V[0].shape[-2] + 1, V[0].shape[-1]),) + V[1:]
        with self.assertRaises(ValueError):
            bvf.check_bv_pair(base, bvf.T3Variations(badV, H))

        # TT variation does not fit a base hole
        badH = (randn(H[0].shape[-3] + 1, H[0].shape[-2], H[0].shape[-1]),) + H[1:]
        with self.assertRaises(ValueError):
            bvf.check_bv_pair(base, bvf.T3Variations(V, badH))

        # stack shape mismatch
        stacked_structure = ((14, 15, 16), (4, 5, 6), (3, 4, 5), (1, 2, 3, 1), (1, 3, 2, 1), (2,))
        _, stacked_variations = _random_basis_variations(stacked_structure)
        with self.assertRaises(ValueError):
            bvf.check_bv_pair(base, stacked_variations)

    def test_stack_unstack(self):
        for BASE_STRUCTURE in self.base_structures:
            for STACK_SHAPE in [(2,), (2, 3)]:
                structure = BASE_STRUCTURE + (STACK_SHAPE,)
                for USE_JAX in [False, True]:
                    with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                        base, variations = _random_basis_variations(structure, use_jax=USE_JAX)

                        # round trips
                        base2 = bvf.T3Basis.stack(base.unstack())
                        self.assertLessEqual(float(cw.corewise_norm(cw.corewise_sub(base.data, base2.data))), tol)
                        variations2 = bvf.T3Variations.stack(variations.unstack())
                        self.assertLessEqual(float(cw.corewise_norm(cw.corewise_sub(variations.data, variations2.data))), tol)

                        # an unstacked leaf equals the manually sliced sub-basis
                        idx = tuple(0 for _ in STACK_SHAPE)
                        leaf = base.unstack()
                        for k in idx:
                            leaf = leaf[k]
                        sliced = bvf.T3Basis(
                            tuple(U[idx] for U in base.up_tucker_cores),
                            tuple(G[idx] for G in base.down_tt_cores),
                            tuple(G[idx] for G in base.left_tt_cores),
                            tuple(G[idx] for G in base.right_tt_cores),
                        )
                        self.assertLessEqual(float(cw.corewise_norm(cw.corewise_sub(sliced.data, leaf.data))), tol)

    def test_bv_to_t3(self):
        for BASE_STRUCTURE in self.base_structures:
            shape = BASE_STRUCTURE[0]
            d = len(shape)
            for STACK_SHAPE in self.stack_shapes:
                structure = BASE_STRUCTURE + (STACK_SHAPE,)
                for USE_JAX in [False, True]:
                    base, variations = _random_basis_variations(structure, use_jax=USE_JAX)
                    U, D, L, R = base.data
                    V, H = variations.data
                    for ii in range(d):
                        with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                                          USE_JAX=USE_JAX, ii=ii, kind="TT"):
                            x = bvf.bv_to_t3((True, ii), base, variations)
                            self._equal_cores(x.tucker_cores, U)
                            self._equal_cores(x.tt_cores, L[:ii] + (H[ii],) + R[ii + 1:])

                        with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                                          USE_JAX=USE_JAX, ii=ii, kind="Tucker"):
                            x = bvf.bv_to_t3((False, ii), base, variations)
                            self._equal_cores(x.tucker_cores, U[:ii] + (V[ii],) + U[ii + 1:])
                            self._equal_cores(x.tt_cores, L[:ii] + (D[ii],) + R[ii + 1:])


if __name__ == "__main__":
    unittest.main()
