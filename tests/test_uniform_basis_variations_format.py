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


# ---- UT3Variations (increment 2): variation left/right masks are (d,), not (d+1,) ----
_V_LEFT_R, _V_RIGHT_R = [1, 2, 3], [1, 2, 2]   # (d,) variation TT ranks


def _make_variations(ss=()):
    def b(r, length):
        a = np.broadcast_to(np.array(r).reshape((length,) + (1,) * len(ss)), (length,) + ss)
        return a.copy()
    tkv = np.random.randn(*((_D,) + ss + (_ND, _N)))
    ttv = np.random.randn(*((_D,) + ss + (_RL, _NU, _RR)))
    masks = ubv.UT3VariationsMasks(
        _prefix_mask(b(_UP_R, _D), _NU), _prefix_mask(b(_DOWN_R, _D), _ND),
        _prefix_mask(b(_V_LEFT_R, _D), _RL), _prefix_mask(b(_V_RIGHT_R, _D), _RR),
    )
    return ubv.UT3Variations(tkv, ttv, _SHAPE, masks)


class TestUT3Variations(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_construct_and_structure(self):
        for ss in ((), (2,)):
            with self.subTest(stack=ss):
                V = _make_variations(ss)
                self.assertEqual(V.shape, _SHAPE)
                self.assertEqual(V.stack_shape, ss)
                self.assertEqual(V.uniform_structure, (_D, _N, _NU, _ND, _RL, _RR, ss))
                idx = (slice(None),) + (0,) * len(ss)
                self.assertEqual(np.asarray(V.up_ranks)[idx].tolist(), _UP_R)
                self.assertEqual(np.asarray(V.variation_left_ranks)[idx].tolist(), _V_LEFT_R)

    def test_data_layout(self):
        V = _make_variations()
        tkv, ttv, shape, masks = V.data
        self.assertEqual(shape, _SHAPE)
        self.assertEqual(len(masks), 4)
        self.assertTrue(all(m.dtype == bool for m in masks))

    def test_validate_rejects_bad(self):
        V = _make_variations()
        with self.assertRaises(ValueError):
            ubv.UT3Variations(V.tucker_variations[..., :-1], V.tt_variations, V.shape, V.masks)  # wrong N
        with self.assertRaises(ValueError):
            ubv.UT3Variations(*V.data[:2], _SHAPE[:-1], V.masks)                                  # wrong shape len

    def test_apply_masks_zeros_padding(self):
        # ones tucker_variations -> after masking, exactly (variations_down AND shape_mask) survives.
        # (this also pins the apply_variations_masks tucker-axis fix -- the old reshape was a bug.)
        tkv = np.ones((_D, _ND, _N))
        ttv = np.random.randn(_D, _RL, _NU, _RR)
        masks = ubv.UT3VariationsMasks(_prefix_mask(_UP_R, _NU), _prefix_mask(_DOWN_R, _ND),
                                       _prefix_mask(_V_LEFT_R, _RL), _prefix_mask(_V_RIGHT_R, _RR))
        V = ubv.UT3Variations(tkv, ttv, _SHAPE, masks)
        masked_tk = V.apply_masks().tucker_variations
        shape_mask = np.arange(_N) < np.array(_SHAPE)[:, None]                       # (d, N)
        expected = (_prefix_mask(_DOWN_R, _ND)[:, :, None] & shape_mask[:, None, :]).astype(float)
        self.assertEqual(masked_tk.shape, (_D, _ND, _N))                              # not the old broken shape
        self.assertTrue(np.array_equal(masked_tk, expected))

    def test_masks_value_hash_eq(self):
        def masks(up_r):
            return ubv.UT3VariationsMasks(_prefix_mask(up_r, _NU), _prefix_mask(_DOWN_R, _ND),
                                          _prefix_mask(_V_LEFT_R, _RL), _prefix_mask(_V_RIGHT_R, _RR))
        a, b = masks(_UP_R), masks(_UP_R)
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, masks([1, 3, 4]))

    @unittest.skipUnless(HAS_JAX, 'jax not installed')
    def test_pytree_roundtrip(self):
        V = _make_variations((2,))
        leaves, treedef = jax.tree_util.tree_flatten(V)
        self.assertEqual(len(leaves), 2)                                             # two variation supercores
        V2 = jax.tree_util.tree_unflatten(treedef, leaves)
        for a, b in zip(V.data[:2], V2.data[:2]):
            self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
        self.assertEqual(V2.shape, V.shape)
        self.assertIs(V2.masks, V.masks)


class TestCheckUbvPair(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def _pair(self, v_up_r=_UP_R):
        d, N, nU, nD, rL, rR = _D, _N, _NU, _ND, _RL, _RR
        up = _prefix_mask(_UP_R, nU); dn = _prefix_mask(_DOWN_R, nD)
        bl = _prefix_mask([1, 2, 3, 1], rL); br = _prefix_mask([1, 2, 2, 1], rR)      # (d+1,)
        B = ubv.UT3Basis(np.random.randn(d, nU, N), np.random.randn(d, rL, nD, rR),
                         np.random.randn(d, rL, nU, rL), np.random.randn(d, rR, nU, rR),
                         _SHAPE, ubv.UT3BasisMasks(up, dn, bl, br))
        V = ubv.UT3Variations(np.random.randn(d, nD, N), np.random.randn(d, rL, nU, rR), _SHAPE,
                              ubv.UT3VariationsMasks(_prefix_mask(v_up_r, nU), dn, bl[:-1], br[1:]))
        return B, V

    def test_consistent_passes(self):
        B, V = self._pair()
        ubv.check_ubv_pair(B, V)   # consistent -> no error

    def test_inconsistent_raises(self):
        B, Vbad = self._pair(v_up_r=[1, 3, 4])   # variation up ranks differ from the base's
        with self.assertRaises(ValueError):
            ubv.check_ubv_pair(B, Vbad)


class TestUt3OrthogonalRepresentations(unittest.TestCase):
    """The equivalence-contract anchor (increment 2b): orthogonalize a uniform T3, convert the frame back
    to ragged, and check it reconstructs the original tensor (and == the ragged orthogonal representation)."""
    def setUp(self):
        np.random.seed(0)

    def test_unstacked_roundtrip_reconstructs_x(self):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.basis_variations_format as bvf
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        base, variations = ubv.ut3_orthogonal_representations(ut3.t3_to_ut3(x))
        base.validate(); variations.validate(); ubv.check_ubv_pair(base, variations)
        rb = ubv.ut3basis_to_t3basis(base)                                  # uniform frame -> ragged T3Basis
        self.assertLess(float(np.linalg.norm(rb.to_dense() - x.to_dense())), 1e-10)
        rbase, _ = bvf.t3_orthogonal_representations(x)                     # uniform == ragged on real parts
        self.assertLess(float(np.linalg.norm(rb.to_dense() - rbase.to_dense())), 1e-10)

    def test_stacked_roundtrip_per_element(self):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
        base, _ = ubv.ut3_orthogonal_representations(ut3.t3_to_ut3(x))
        tree = ubv.ut3basis_to_t3basis(base)                               # nested tree of T3Basis
        xd = x.to_dense()
        for i in range(2):
            self.assertLess(float(np.linalg.norm(tree[i].to_dense() - xd[i])), 1e-10)


if __name__ == '__main__':
    unittest.main()
