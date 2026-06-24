# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Tests for the uniform basis-variations layer (UT3Basis), uniform-fix slice 3a.

Increment 1 covers the rebuilt UT3Basis data structure: the int-tuple `shape` + UT3BasisMasks holder +
pytree composition (mirroring the plain UT3 layer). Conversions / to_dense round-trips, unstack/stack, and
the method buildout land in later increments.
"""
import unittest
import numpy as np

import t3toolbox.uniform_basis_variations_format as ubv

try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _prefix_mask(ranks, pad):  # ranks: HOST int (...,) -> bool (..., pad)
    return np.arange(pad) < np.asarray(ranks)[..., None]


def _make_basis(d, N, nU, nD, rL, rR, shape, up_r, down_r, left_r, right_r, ss=()):
    """Build a structurally-valid UT3Basis with prefix masks (ranks may vary per stack element)."""
    up    = np.random.randn(*((d,) + ss + (nU, N)))
    down  = np.random.randn(*((d,) + ss + (rL, nD, rR)))
    left  = np.random.randn(*((d,) + ss + (rL, nU, rL)))
    right = np.random.randn(*((d,) + ss + (rR, nU, rR)))
    masks = ubv.UT3BasisMasks(
        _prefix_mask(up_r, nU), _prefix_mask(down_r, nD),
        _prefix_mask(left_r, rL), _prefix_mask(right_r, rR),
    )
    return ubv.UT3Basis(up, down, left, right, tuple(shape), masks)


# concrete padded structure shared across tests
_D, _N, _NU, _ND, _RL, _RR = 3, 6, 4, 5, 3, 2
_SHAPE = (4, 5, 6)
_UP_R, _DOWN_R = [2, 3, 4], [3, 4, 5]            # (d,)
_LEFT_R, _RIGHT_R = [1, 2, 3, 1], [1, 2, 2, 1]   # (d+1,)


class TestUT3Basis(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def _basis(self, ss=()):
        # broadcast the unstacked ranks onto the stack (same ranks every element, for simplicity)
        def b(r, length):
            a = np.broadcast_to(np.array(r).reshape((length,) + (1,) * len(ss)), (length,) + ss)
            return a.copy()
        return _make_basis(_D, _N, _NU, _ND, _RL, _RR, _SHAPE,
                           b(_UP_R, _D), b(_DOWN_R, _D), b(_LEFT_R, _D + 1), b(_RIGHT_R, _D + 1), ss=ss)

    # ---- construction + structure recovery ----
    def test_construct_and_structure(self):
        for ss in ((), (2,)):
            with self.subTest(stack=ss):
                B = self._basis(ss)
                self.assertEqual(B.shape, _SHAPE)
                self.assertEqual(B.stack_shape, ss)
                self.assertEqual(B.uniform_structure, (_D, _N, _NU, _ND, _RL, _RR, ss))
                # first stack element's ranks recover the chosen ranks
                idx = (slice(None),) + (0,) * len(ss)
                self.assertEqual(np.asarray(B.up_ranks)[idx].tolist(), _UP_R)
                self.assertEqual(np.asarray(B.down_ranks)[idx].tolist(), _DOWN_R)
                self.assertEqual(np.asarray(B.left_ranks)[idx].tolist(), _LEFT_R)
                self.assertEqual(np.asarray(B.right_ranks)[idx].tolist(), _RIGHT_R)

    def test_data_layout(self):
        B = self._basis()
        up_sc, down_sc, left_sc, right_sc, shape, masks = B.data
        self.assertEqual(shape, _SHAPE)              # .data[4] is the static int tuple
        self.assertEqual(len(masks), 4)              # .data[5] is the 4-mask tuple
        self.assertTrue(all(m.dtype == bool for m in masks))

    # ---- validate (structural hard errors) ----
    def test_validate_rejects_bad_supercore(self):
        B = self._basis()
        with self.assertRaises(ValueError):
            ubv.UT3Basis(B.up_tucker_supercore[..., :-1],  # wrong N on up core
                         B.down_tt_supercore, B.left_tt_supercore, B.right_tt_supercore,
                         B.shape, B.masks)

    def test_validate_rejects_bad_shape_tuple(self):
        B = self._basis()
        with self.assertRaises(ValueError):
            ubv.UT3Basis(*B.data[:4], _SHAPE[:-1], B.masks)            # wrong length
        with self.assertRaises(ValueError):
            ubv.UT3Basis(*B.data[:4], (_N + 1,) + _SHAPE[1:], B.masks)  # exceeds padded N

    # ---- masking semantics: real region preserved, padding zeroed ----
    def test_apply_masks_zeros_padding(self):
        # all-ones up core -> after masking, exactly the (up_mask AND shape_mask) region survives
        up = np.ones((_D, _NU, _N))
        down  = np.random.randn(_D, _RL, _ND, _RR)
        left  = np.random.randn(_D, _RL, _NU, _RL)
        right = np.random.randn(_D, _RR, _NU, _RR)
        masks = ubv.UT3BasisMasks(_prefix_mask(_UP_R, _NU), _prefix_mask(_DOWN_R, _ND),
                                  _prefix_mask(_LEFT_R, _RL), _prefix_mask(_RIGHT_R, _RR))
        B = ubv.UT3Basis(up, down, left, right, _SHAPE, masks)

        masked_up = B.apply_masks().up_tucker_supercore
        shape_mask = np.arange(_N) < np.array(_SHAPE)[:, None]            # (d, N)
        expected = (_prefix_mask(_UP_R, _NU)[:, :, None] & shape_mask[:, None, :]).astype(float)
        self.assertTrue(np.array_equal(masked_up, expected))

    def test_apply_masks_idempotent(self):
        B = self._basis((2,))
        once = B.apply_masks()
        twice = once.apply_masks()
        for a, b in zip(once.data[:4], twice.data[:4]):
            self.assertEqual(float(np.linalg.norm(a - b)), 0.0)

    # ---- value-based mask hashing (the jit-cache-stability contract) ----
    def test_masks_value_hash_eq(self):
        # a rebuilt-but-array-identical UT3BasisMasks must be == and hash-equal (so a re-orthogonalized
        # frame is the SAME jit cache key); a different rank structure must not be.
        def masks(up_r):
            return ubv.UT3BasisMasks(_prefix_mask(up_r, _NU), _prefix_mask(_DOWN_R, _ND),
                                     _prefix_mask(_LEFT_R, _RL), _prefix_mask(_RIGHT_R, _RR))
        a, b = masks(_UP_R), masks(_UP_R)
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, masks([1, 3, 4]))   # different up ranks -> not equal

    # ---- jax pytree composition (supercores = children; (shape, masks) = static aux) ----
    @unittest.skipUnless(HAS_JAX, 'jax not installed')
    def test_pytree_roundtrip(self):
        B = self._basis((2,))
        leaves, treedef = jax.tree_util.tree_flatten(B)
        self.assertEqual(len(leaves), 4)                                  # the 4 supercores are children
        B2 = jax.tree_util.tree_unflatten(treedef, leaves)
        for a, b in zip(B.data[:4], B2.data[:4]):
            self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
        self.assertEqual(B2.shape, B.shape)                              # value-hashed shape survives in aux
        self.assertIs(B2.masks, B.masks)                                 # identity-hashed holder carried in aux
        self.assertIsInstance(hash(treedef), int)                        # aux is hashable -> valid jit key

    @unittest.skipUnless(HAS_JAX, 'jax not installed')
    def test_jit_over_basis_keeps_masks_concrete(self):
        # a jitted op on a UT3Basis traces the supercores; masks/shape stay static host structure
        B = self._basis()

        @jax.jit
        def total(b):
            return sum(jax.numpy.sum(c) for c in b.data[:4])

        val = total(B)
        self.assertTrue(np.isfinite(float(val)))
        self.assertTrue(all(isinstance(m, np.ndarray) for m in B.masks.data))


if __name__ == '__main__':
    unittest.main()
