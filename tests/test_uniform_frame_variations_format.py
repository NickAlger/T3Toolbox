# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Tests for the uniform frame-variations layer (UT3Frame), uniform-fix slice 3a.

Increment 1 covers the rebuilt UT3Frame data structure: the int-tuple `shape` + UT3FrameMasks holder +
pytree composition (mirroring the plain UT3 layer). Conversions / to_dense round-trips, unstack/stack, and
the method buildout land in later increments.
"""
import unittest
import numpy as np

import t3toolbox.uniform_frame_variations_format as ubv

try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _prefix_mask(ranks, pad):  # ranks: HOST int (...,) -> bool (..., pad)
    return np.arange(pad) < np.asarray(ranks)[..., None]


def _make_frame(d, N, nU, nD, rL, rR, shape, up_r, down_r, left_r, right_r, ss=()):
    """Build a structurally-valid UT3Frame with prefix masks (ranks may vary per stack element)."""
    up    = np.random.randn(*((d,) + ss + (nU, N)))
    down  = np.random.randn(*((d,) + ss + (rL, nD, rR)))
    left  = np.random.randn(*((d,) + ss + (rL, nU, rL)))
    right = np.random.randn(*((d,) + ss + (rR, nU, rR)))
    masks = ubv.UT3FrameMasks(
        _prefix_mask(up_r, nU), _prefix_mask(down_r, nD),
        _prefix_mask(left_r, rL), _prefix_mask(right_r, rR),
    )
    return ubv.UT3Frame(up, down, left, right, tuple(shape), masks)


# concrete padded structure shared across tests
_D, _N, _NU, _ND, _RL, _RR = 3, 6, 4, 5, 3, 2
_SHAPE = (4, 5, 6)
_UP_R, _DOWN_R = [2, 3, 4], [3, 4, 5]            # (d,)
_LEFT_R, _RIGHT_R = [1, 2, 3, 1], [1, 2, 2, 1]   # (d+1,)


class TestUT3Frame(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def _frame(self, ss=()):
        # broadcast the unstacked ranks onto the stack (same ranks every element, for simplicity)
        def b(r, length):
            a = np.broadcast_to(np.array(r).reshape((length,) + (1,) * len(ss)), (length,) + ss)
            return a.copy()
        return _make_frame(_D, _N, _NU, _ND, _RL, _RR, _SHAPE,
                           b(_UP_R, _D), b(_DOWN_R, _D), b(_LEFT_R, _D + 1), b(_RIGHT_R, _D + 1), ss=ss)

    # ---- construction + structure recovery ----
    def test_construct_and_structure(self):
        for ss in ((), (2,)):
            with self.subTest(stack=ss):
                B = self._frame(ss)
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
        B = self._frame()
        up_sc, down_sc, left_sc, right_sc, shape, masks = B.data
        self.assertEqual(shape, _SHAPE)              # .data[4] is the static int tuple
        self.assertEqual(len(masks), 4)              # .data[5] is the 4-mask tuple
        self.assertTrue(all(m.dtype == bool for m in masks))

    # ---- validate (structural hard errors) ----
    def test_validate_rejects_bad_supercore(self):
        B = self._frame()
        with self.assertRaises(ValueError):
            ubv.UT3Frame(B.up_tucker_supercore[..., :-1],  # wrong N on up core
                         B.down_tt_supercore, B.left_tt_supercore, B.right_tt_supercore,
                         B.shape, B.masks)

    def test_validate_rejects_bad_shape_tuple(self):
        B = self._frame()
        with self.assertRaises(ValueError):
            ubv.UT3Frame(*B.data[:4], _SHAPE[:-1], B.masks)            # wrong length
        with self.assertRaises(ValueError):
            ubv.UT3Frame(*B.data[:4], (_N + 1,) + _SHAPE[1:], B.masks)  # exceeds padded N

    # ---- masking semantics: real region preserved, padding zeroed ----
    def test_apply_masks_zeros_padding(self):
        # all-ones up core -> after masking, exactly the (up_mask AND shape_mask) region survives
        up = np.ones((_D, _NU, _N))
        down  = np.random.randn(_D, _RL, _ND, _RR)
        left  = np.random.randn(_D, _RL, _NU, _RL)
        right = np.random.randn(_D, _RR, _NU, _RR)
        masks = ubv.UT3FrameMasks(_prefix_mask(_UP_R, _NU), _prefix_mask(_DOWN_R, _ND),
                                  _prefix_mask(_LEFT_R, _RL), _prefix_mask(_RIGHT_R, _RR))
        B = ubv.UT3Frame(up, down, left, right, _SHAPE, masks)

        masked_up = B.apply_masks().up_tucker_supercore
        shape_mask = np.arange(_N) < np.array(_SHAPE)[:, None]            # (d, N)
        expected = (_prefix_mask(_UP_R, _NU)[:, :, None] & shape_mask[:, None, :]).astype(float)
        self.assertTrue(np.array_equal(masked_up, expected))

    def test_apply_masks_idempotent(self):
        B = self._frame((2,))
        once = B.apply_masks()
        twice = once.apply_masks()
        for a, b in zip(once.data[:4], twice.data[:4]):
            self.assertEqual(float(np.linalg.norm(a - b)), 0.0)

    # ---- value-based mask hashing (the jit-cache-stability contract) ----
    def test_masks_value_hash_eq(self):
        # a rebuilt-but-array-identical UT3FrameMasks must be == and hash-equal (so a re-orthogonalized
        # frame is the SAME jit cache key); a different rank structure must not be.
        def masks(up_r):
            return ubv.UT3FrameMasks(_prefix_mask(up_r, _NU), _prefix_mask(_DOWN_R, _ND),
                                     _prefix_mask(_LEFT_R, _RL), _prefix_mask(_RIGHT_R, _RR))
        a, b = masks(_UP_R), masks(_UP_R)
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, masks([1, 3, 4]))   # different up ranks -> not equal

    # ---- jax pytree composition (supercores = children; (shape, masks) = static aux) ----
    @unittest.skipUnless(HAS_JAX, 'jax not installed')
    def test_pytree_roundtrip(self):
        B = self._frame((2,))
        leaves, treedef = jax.tree_util.tree_flatten(B)
        self.assertEqual(len(leaves), 4)                                  # the 4 supercores are children
        B2 = jax.tree_util.tree_unflatten(treedef, leaves)
        for a, b in zip(B.data[:4], B2.data[:4]):
            self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
        self.assertEqual(B2.shape, B.shape)                              # value-hashed shape survives in aux
        self.assertIs(B2.masks, B.masks)                                 # identity-hashed holder carried in aux
        self.assertIsInstance(hash(treedef), int)                        # aux is hashable -> valid jit key

    @unittest.skipUnless(HAS_JAX, 'jax not installed')
    def test_jit_over_frame_keeps_masks_concrete(self):
        # a jitted op on a UT3Frame traces the supercores; masks/shape stay static host structure
        B = self._frame()

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
        B = ubv.UT3Frame(np.random.randn(d, nU, N), np.random.randn(d, rL, nD, rR),
                         np.random.randn(d, rL, nU, rL), np.random.randn(d, rR, nU, rR),
                         _SHAPE, ubv.UT3FrameMasks(up, dn, bl, br))
        V = ubv.UT3Variations(np.random.randn(d, nD, N), np.random.randn(d, rL, nU, rR), _SHAPE,
                              ubv.UT3VariationsMasks(_prefix_mask(v_up_r, nU), dn, bl[:-1], br[1:]))
        return B, V

    def test_consistent_passes(self):
        B, V = self._pair()
        ubv.check_ufv_pair(B, V)   # consistent -> no error

    def test_inconsistent_raises(self):
        B, Vbad = self._pair(v_up_r=[1, 3, 4])   # variation up ranks differ from the frame's
        with self.assertRaises(ValueError):
            ubv.check_ufv_pair(B, Vbad)

    def _K_variation(self, B, K, up_r=_UP_R):
        """A K-stacked variation (a bundle of K tangents at the single frame B): cores grow a leading K
        stack, masks are the frame's gauge-shifted masks broadcast constant along K."""
        d, N, nU, nD, rL, rR = _D, _N, _NU, _ND, _RL, _RR
        bm = B.masks
        bcast = lambda m: np.broadcast_to(m.reshape(m.shape[:1] + (1,) * len(K) + m.shape[1:]),
                                          m.shape[:1] + K + m.shape[1:])
        up = bcast(_prefix_mask(up_r, nU))           # allow an over-ridden up rank to test mismatch
        masks = ubv.UT3VariationsMasks(up, bcast(bm.down_mask),
                                       bcast(bm.frame_left_mask[:-1]), bcast(bm.frame_right_mask[1:]))
        return ubv.UT3Variations(np.random.randn(*((d,) + K + (nD, N))),
                                 np.random.randn(*((d,) + K + (rL, nU, rR))), _SHAPE, masks)

    def test_tangent_K_stack_passes(self):
        # the 3b-0 capability: a frame with C=() and a K-stacked variation (K != ()) is a consistent pair.
        B, _ = self._pair()
        for K in [(2,), (4,), (2, 3)]:
            with self.subTest(K=K):
                ubv.check_ufv_pair(B, self._K_variation(B, K))

    def test_tangent_K_plus_C_stack_passes(self):
        # frame carries a core (C) stack; variation carries K+C with C as the trailing suffix.
        d, N, nU, nD, rL, rR = _D, _N, _NU, _ND, _RL, _RR
        C, K = (3,), (2,)
        up = _prefix_mask(_UP_R, nU); dn = _prefix_mask(_DOWN_R, nD)
        bl = _prefix_mask([1, 2, 3, 1], rL); br = _prefix_mask([1, 2, 2, 1], rR)
        bcastC = lambda m: np.broadcast_to(m[:, None], m.shape[:1] + C + m.shape[1:])
        B = ubv.UT3Frame(np.random.randn(d, *C, nU, N), np.random.randn(d, *C, rL, nD, rR),
                         np.random.randn(d, *C, rL, nU, rL), np.random.randn(d, *C, rR, nU, rR),
                         _SHAPE, ubv.UT3FrameMasks(bcastC(up), bcastC(dn), bcastC(bl), bcastC(br)))
        # K+C variation masks: insert K after d, broadcast the frame's gauge-shifted (C-stacked) masks.
        bcastK = lambda m: np.broadcast_to(m[:, None], m.shape[:1] + K + m.shape[1:])
        masks = ubv.UT3VariationsMasks(bcastK(bcastC(up)), bcastK(bcastC(dn)),
                                       bcastK(bcastC(bl[:-1])), bcastK(bcastC(br[1:])))
        V = ubv.UT3Variations(np.random.randn(d, *K, *C, nD, N),
                              np.random.randn(d, *K, *C, rL, nU, rR), _SHAPE, masks)
        ubv.check_ufv_pair(B, V)

    def test_K_stack_mask_not_constant_raises(self):
        # a K-stacked variation whose mask is NOT constant along K (one slice has a different up rank).
        B, _ = self._pair()
        V = self._K_variation(B, (2,))
        bad_up = np.array(V.masks.variations_up_mask)
        bad_up[:, 1, :] = _prefix_mask([1, 3, 4], _NU)   # second K slice differs from the frame
        Vbad = ubv.UT3Variations(V.tucker_variations, V.tt_variations, _SHAPE,
                                 ubv.UT3VariationsMasks(bad_up, V.masks.variations_down_mask,
                                                        V.masks.variations_left_mask,
                                                        V.masks.variations_right_mask))
        with self.assertRaises(ValueError):
            ubv.check_ufv_pair(B, Vbad)

    def test_frame_stack_not_suffix_raises(self):
        # frame C=(3,) is NOT a trailing suffix of variation stack (5,) -> reject.
        d, N, nU, nD, rL, rR = _D, _N, _NU, _ND, _RL, _RR
        C = (3,)
        up = _prefix_mask(_UP_R, nU); dn = _prefix_mask(_DOWN_R, nD)
        bl = _prefix_mask([1, 2, 3, 1], rL); br = _prefix_mask([1, 2, 2, 1], rR)
        bcastC = lambda m, c: np.broadcast_to(m[:, None], m.shape[:1] + c + m.shape[1:])
        B = ubv.UT3Frame(np.random.randn(d, *C, nU, N), np.random.randn(d, *C, rL, nD, rR),
                         np.random.randn(d, *C, rL, nU, rL), np.random.randn(d, *C, rR, nU, rR),
                         _SHAPE, ubv.UT3FrameMasks(bcastC(up, C), bcastC(dn, C), bcastC(bl, C), bcastC(br, C)))
        bad = (5,)
        masks = ubv.UT3VariationsMasks(bcastC(up, bad), bcastC(dn, bad),
                                       bcastC(bl[:-1], bad), bcastC(br[1:], bad))
        V = ubv.UT3Variations(np.random.randn(d, *bad, nD, N),
                              np.random.randn(d, *bad, rL, nU, rR), _SHAPE, masks)
        with self.assertRaises(ValueError):
            ubv.check_ufv_pair(B, V)


class TestUt3OrthogonalRepresentations(unittest.TestCase):
    """The equivalence-contract anchor (increment 2b): orthogonalize a uniform T3, convert the frame back
    to ragged, and check it reconstructs the original tensor (and == the ragged orthogonal representation)."""
    def setUp(self):
        np.random.seed(0)

    def test_unstacked_roundtrip_reconstructs_x(self):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.frame_variations_format as bvf
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        frame, variations = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
        frame.validate(); variations.validate(); ubv.check_ufv_pair(frame, variations)
        rb = frame.to_t3frame()                                  # uniform frame -> ragged T3Frame
        self.assertLess(float(np.linalg.norm(rb.to_dense() - x.to_dense())), 1e-10)
        rframe, _ = bvf.t3_orthogonal_representations(x)                     # uniform == ragged on real parts
        self.assertLess(float(np.linalg.norm(rb.to_dense() - rframe.to_dense())), 1e-10)

    def test_stacked_roundtrip_per_element(self):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
        frame, _ = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
        tree = frame.to_t3frame()                               # nested tree of T3Frame
        xd = x.to_dense()
        for i in range(2):
            self.assertLess(float(np.linalg.norm(tree[i].to_dense() - xd[i])), 1e-10)

    def test_backend_path_on_raw_data(self):
        # the backend twin: raw plain-UT3 .data in -> raw (frame, variation) .data out, NO frontend objects.
        # the round-trip also proves the prefix masks are right (i.e. SVD put the real content upper-left).
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.backend.ufv_conversions as ubvc
        import t3toolbox.frame_variations_format as bvf
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        frame_data, variation_data = ubvc.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x).data)
        self.assertEqual(len(frame_data), 6)          # (up, down, left, right, shape, masks)
        self.assertEqual(len(variation_data), 4)      # (tucker_var, tt_var, shape, masks)
        self.assertEqual(frame_data[4], (4, 5, 6))    # shape carried through
        self.assertEqual(len(frame_data[5]), 4)       # four frame rank masks
        ragged_cores = ubvc.ut3frame_to_t3frame(frame_data)   # backend uniform->ragged, all on raw .data
        self.assertLess(float(np.linalg.norm(bvf.T3Frame(*ragged_cores).to_dense() - x.to_dense())), 1e-10)


class TestCrossLayerConverters(unittest.TestCase):
    """2c-A: the ragged<->uniform converters as methods (from_t3frame/to_t3frame,
    from_t3variations/to_t3variations), verified by round-trip + dense/corewise equivalence."""
    def setUp(self):
        np.random.seed(0)

    def _ragged_pair(self, ss=()):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.frame_variations_format as bvf
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=ss)
        frame, variations = bvf.t3_orthogonal_representations(x)
        return x, frame, variations

    def test_from_t3frame_roundtrip_unstacked(self):
        _, frame, _ = self._ragged_pair()
        UB = ubv.UT3Frame.from_t3frame(frame)              # ragged frame -> uniform
        UB.validate()
        B2 = UB.to_t3frame()                              # unstacked -> back to ragged T3Frame
        self.assertLess(float(np.linalg.norm(B2.to_dense() - frame.to_dense())), 1e-10)

    def test_from_t3frame_roundtrip_stacked(self):
        _, frame, _ = self._ragged_pair(ss=(2,))
        UB = ubv.UT3Frame.from_t3frame(frame)
        UB.validate()
        tree = UB.to_t3frame()                            # nested tree of T3Frame
        bd = frame.to_dense()
        for i in range(2):
            self.assertLess(float(np.linalg.norm(tree[i].to_dense() - bd[i])), 1e-10)

    def test_from_t3frame_extra_padding_still_roundtrips(self):
        # padding-invariance: forcing larger pad than the natural max must not change the represented point.
        _, frame, _ = self._ragged_pair()
        UB = ubv.UT3Frame.from_t3frame(frame, N=10, nU=8, nD=8, rL=6, rR=6)
        UB.validate()
        self.assertEqual((UB.N, UB.nU, UB.nD, UB.rL, UB.rR), (10, 8, 8, 6, 6))
        self.assertLess(float(np.linalg.norm(UB.to_t3frame().to_dense() - frame.to_dense())), 1e-10)

    def test_from_t3variations_roundtrip_unstacked(self):
        import t3toolbox.corewise as cw
        _, _, variations = self._ragged_pair()
        UV = ubv.UT3Variations.from_t3variations(variations)   # ragged variations -> uniform
        UV.validate()
        V2 = UV.to_t3variations()                              # unstacked -> back to ragged T3Variations
        self.assertTrue(np.allclose(cw.corewise_norm(cw.corewise_sub(V2.data, variations.data)), 0.0))

    def test_from_t3variations_roundtrip_stacked(self):
        import t3toolbox.corewise as cw
        _, _, variations = self._ragged_pair(ss=(2,))
        UV = ubv.UT3Variations.from_t3variations(variations)
        UV.validate()
        tree = UV.to_t3variations()                           # nested tree of T3Variations
        unstacked = variations.unstack()
        for i in range(2):
            self.assertTrue(np.allclose(
                cw.corewise_norm(cw.corewise_sub(tree[i].data, unstacked[i].data)), 0.0))


class TestBasePointAndDtype(unittest.TestCase):
    """2c-B: base-point conversions (to_ut3 / to_dense / from_ut3) + dtype/copy/repr utilities.
    Users build frames via ut3_orthogonal_representations, so that is the path under test."""
    def setUp(self):
        np.random.seed(0)

    def _frame(self, ss=()):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=ss)
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        frame, variations = ubv.ut3_orthogonal_representations(ux)
        return x, ux, frame, variations

    def test_to_dense_reconstructs_base_point(self):
        x, _, frame, _ = self._frame()
        self.assertLess(float(np.linalg.norm(frame.to_dense() - x.to_dense())), 1e-10)

    def test_to_ut3_matches_to_dense_and_x(self):
        import t3toolbox.uniform_tucker_tensor_train as ut3
        x, _, frame, _ = self._frame()
        ub = frame.to_ut3()
        self.assertIsInstance(ub, ut3.UniformTuckerTensorTrain)
        self.assertTrue(np.allclose(ub.to_dense(), frame.to_dense()))   # to_dense == to_ut3().to_dense()
        self.assertTrue(np.allclose(ub.to_dense(), x.to_dense()))      # and reconstructs x

    def test_to_dense_stacked(self):
        x, _, frame, _ = self._frame(ss=(2,))
        self.assertEqual(frame.stack_shape, (2,))
        self.assertTrue(np.allclose(frame.to_dense(), x.to_dense()))

    def test_from_ut3_reconstructs_x(self):
        x, ux, _, _ = self._frame()
        frame2 = ubv.UT3Frame.from_ut3(ux)
        self.assertIsInstance(frame2, ubv.UT3Frame)
        self.assertLess(float(np.linalg.norm(frame2.to_dense() - x.to_dense())), 1e-10)

    def test_frame_dtype_copy_repr(self):
        import t3toolbox.backend.common as common
        _, _, frame, _ = self._frame()
        cp = frame.copy()
        self.assertTrue(np.allclose(cp.to_dense(), frame.to_dense()))
        # deep copy: supercores are independent arrays (mirrors ragged T3Frame.copy)
        self.assertTrue(all(not np.shares_memory(a, b) for a, b in zip(cp.supercores, frame.supercores)))
        self.assertFalse(frame.contains_jax)
        self.assertIn('UT3Frame(shape=(4, 5, 6)', repr(frame))
        if HAS_JAX:
            jb = frame.to_jax()
            self.assertTrue(jb.contains_jax)
            self.assertTrue(all(common.is_jax_ndarray(sc) for sc in jb.supercores))    # data -> jax
            self.assertTrue(all(common.is_numpy_ndarray(m) for m in jb.masks.data))    # masks stay host
            self.assertFalse(jb.to_numpy().contains_jax)

    def test_variations_dtype_copy_repr(self):
        import t3toolbox.backend.common as common
        _, _, _, variations = self._frame()
        cp = variations.copy()
        self.assertTrue(np.allclose(cp.apply_masks().tucker_variations,
                                    variations.apply_masks().tucker_variations))
        # deep copy: supercores are independent arrays (mirrors ragged T3Variations.copy)
        self.assertTrue(all(not np.shares_memory(a, b) for a, b in zip(cp.supercores, variations.supercores)))
        self.assertFalse(variations.contains_jax)
        self.assertIn('UT3Variations(shape=(4, 5, 6)', repr(variations))
        if HAS_JAX:
            jv = variations.to_jax()
            self.assertTrue(jv.contains_jax)
            self.assertTrue(all(common.is_jax_ndarray(sc) for sc in jv.supercores))
            self.assertTrue(all(common.is_numpy_ndarray(m) for m in jv.masks.data))
            self.assertFalse(jv.to_numpy().contains_jax)


class TestStackUnstack(unittest.TestCase):
    """2c-C: UT3Frame / UT3Variations unstack <-> stack (the stack rides axes 1..len(stack_shape);
    shape is shared; the four masks unstack along the stack too)."""
    def setUp(self):
        np.random.seed(0)

    def _frame(self, ss):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=ss)
        frame, variations = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
        return x, frame, variations

    def _assert_exact_roundtrip(self, obj, restacked):
        for a, b in zip(restacked.supercores, obj.supercores):
            self.assertEqual(float(np.max(np.abs(a - b))), 0.0)        # supercores exact
        for a, b in zip(restacked.masks.data, obj.masks.data):
            self.assertTrue(np.array_equal(a, b))                      # masks exact

    def test_frame_stack_is_inverse_of_unstack(self):
        for ss in [(), (2,), (2, 3)]:
            with self.subTest(stack=ss):
                _, frame, _ = self._frame(ss)
                self._assert_exact_roundtrip(frame, ubv.UT3Frame.stack(frame.unstack()))

    def test_variations_stack_is_inverse_of_unstack(self):
        for ss in [(), (2,), (2, 3)]:
            with self.subTest(stack=ss):
                _, _, var = self._frame(ss)
                self._assert_exact_roundtrip(var, ubv.UT3Variations.stack(var.unstack()))

    def test_frame_unstack_leaves_reconstruct_per_element(self):
        x, frame, _ = self._frame((2,))
        tree = frame.unstack()                          # 1D stack -> tuple of UT3Frame
        self.assertEqual(len(tree), 2)
        xd = x.to_dense()
        for i in range(2):
            tree[i].validate()
            self.assertLess(float(np.linalg.norm(tree[i].to_dense() - xd[i])), 1e-10)

    def test_unstacked_unstack_returns_single_object(self):
        _, frame, _ = self._frame(())
        self.assertIsInstance(frame.unstack(), ubv.UT3Frame)   # no stack axes -> the object itself

    def test_stack_heterogeneous_ranks(self):
        # the point of the uniform layer: stack frames of DIFFERENT ranks into one batch (common padding),
        # a varying-rank slice of the determinantal variety. Per-element masks then differ along the stack.
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.frame_variations_format as bvf
        xs = [t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)),
              t3.TuckerTensorTrain.randn((4, 5, 6), (3, 3, 2), (1, 1, 2, 1))]
        pad = dict(N=6, nU=4, nD=4, rL=3, rR=3)        # common padded dims so they stack
        ubases = [ubv.UT3Frame.from_t3frame(bvf.t3_orthogonal_representations(x)[0], **pad) for x in xs]
        stacked = ubv.UT3Frame.stack(ubases)
        stacked.validate()
        self.assertEqual(stacked.stack_shape, (2,))
        self.assertFalse(np.array_equal(stacked.masks.up_mask[:, 0], stacked.masks.up_mask[:, 1]))  # ranks vary
        tree = stacked.unstack()
        for i, x in enumerate(xs):
            self.assertLess(float(np.linalg.norm(tree[i].to_dense() - x.to_dense())), 1e-10)


class TestVariationLinearAlgebra(unittest.TestCase):
    """2c-D: UT3Variations vector-space ops (corewise at a fixed mask) + constructors. Verified against
    the ragged T3Variations via the equivalence contract."""
    def setUp(self):
        np.random.seed(0)

    def _pair(self, ss=()):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=ss)
        return ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))

    def _cerr(self, a, b):
        import t3toolbox.corewise as cw
        return float(cw.corewise_norm(cw.corewise_sub(a, b)))

    def test_vector_space_ops_match_ragged(self):
        frame, v = self._pair()
        w = ubv.UT3Variations.randn_like(frame)          # second tangent at the SAME frame (same mask)
        rv, rw = v.to_t3variations(), w.to_t3variations()
        self.assertEqual(self._cerr((v + w).to_t3variations().data, (rv + rw).data), 0.0)
        self.assertEqual(self._cerr((v - w).to_t3variations().data, (rv - rw).data), 0.0)
        self.assertEqual(self._cerr((2.5 * v).to_t3variations().data, (2.5 * rv).data), 0.0)
        self.assertEqual(self._cerr((v * 2.5).to_t3variations().data, (2.5 * rv).data), 0.0)  # __rmul__
        self.assertEqual(self._cerr((-v).to_t3variations().data, (-rv).data), 0.0)

    def test_ops_preserve_mask(self):
        frame, v = self._pair()
        w = ubv.UT3Variations.randn_like(frame)
        for r in (v + w, v - w, 2.0 * v, -v):
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(r.masks.data, v.masks.data)))

    def test_sum_stack_matches_ragged(self):
        frame, v = self._pair(ss=(3,))
        s = v.sum_stack()
        s.validate()
        self.assertEqual(s.stack_shape, ())
        tree = v.to_t3variations()                       # 3 ragged T3Variations
        self.assertEqual(self._cerr(s.to_t3variations().data, (tree[0] + tree[1] + tree[2]).data), 0.0)

    def test_zeros_like_is_zero_tangent_at_frame(self):
        frame, _ = self._pair()
        z = ubv.UT3Variations.zeros_like(frame)
        z.validate()
        ubv.check_ufv_pair(frame, z)                       # carries the frame's gauge masks -> pairs
        self.assertEqual(float(np.max(np.abs(z.tucker_variations))), 0.0)
        self.assertTrue(np.array_equal(z.masks.variations_left_mask, frame.masks.frame_left_mask[:-1]))

    def test_randn_like_pairs_with_frame_stacked(self):
        frame, _ = self._pair(ss=(2,))
        r = ubv.UT3Variations.randn_like(frame)
        r.validate(); ubv.check_ufv_pair(frame, r)
        self.assertEqual(r.stack_shape, (2,))

    def test_zeros_default_all_true_masks(self):
        z = ubv.UT3Variations.zeros(((3, 4, 6), (3, 2, 5, 2)), (4, 5, 6))   # (d,nD,N),(d,rL,nU,rR)
        z.validate()
        self.assertTrue(all(m.all() for m in z.masks.data))                 # all-True (full rank)
        self.assertEqual(float(np.max(np.abs(z.tucker_variations))), 0.0)

    def test_unit(self):
        u = ubv.UT3Variations.unit(((3, 4, 6), (3, 2, 5, 2)), (4, 5, 6), (False, 1, (2, 3)))
        u.validate()
        self.assertEqual(u.tucker_variations[1, 2, 3], 1.0)
        self.assertEqual(float(np.sum(np.abs(u.tucker_variations))), 1.0)   # exactly one nonzero
        self.assertEqual(float(np.sum(np.abs(u.tt_variations))), 0.0)

    def test_allclose(self):
        frame, v = self._pair()
        self.assertTrue(v.allclose(v).all())
        self.assertTrue(v.allclose(v.copy()).all())
        self.assertFalse(v.allclose(ubv.UT3Variations.randn_like(frame)).all())

    def test_precondition_same_structure_different_masks(self):
        # the uniform-specific footgun: same padded shape, DIFFERENT masks (different ranks) -> different
        # tangent space. Ragged would catch a rank mismatch as a shape error; uniform padding hides it,
        # so add/sub must reject it explicitly.
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.frame_variations_format as bvf
        pad = dict(N=6, nU=4, nD=4, rL=3, rR=3)
        v1 = ubv.UT3Variations.from_t3variations(
            bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)))[1], **pad)
        v2 = ubv.UT3Variations.from_t3variations(
            bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn((4, 5, 6), (3, 3, 2), (1, 1, 2, 1)))[1], **pad)
        self.assertEqual(v1.uniform_structure, v2.uniform_structure)   # same padded structure...
        self.assertNotEqual(v1.masks, v2.masks)                        # ...but different rank masks
        with self.assertRaises(ValueError):
            v1 + v2


class TestReverseOrthogonalizeRandom(unittest.TestCase):
    """2c-E: UT3Frame/UT3Variations reverse + UT3Frame orthogonalize / random_orthogonal (direct uniform)."""
    def setUp(self):
        np.random.seed(0)

    def _pair(self, ss=()):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=ss)
        frame, var = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
        return x, frame, var

    def _cerr(self, a, b):
        import t3toolbox.corewise as cw
        return float(cw.corewise_norm(cw.corewise_sub(a, b)))

    def test_frame_reverse_commutes_with_to_t3frame(self):
        _, frame, _ = self._pair()
        frame.reverse().validate()
        # Nick's correctness lens: reverse commutes with conversion
        self.assertLess(float(np.linalg.norm(
            frame.reverse().to_t3frame().to_dense() - frame.to_t3frame().reverse().to_dense())), 1e-10)

    def test_frame_reverse_involution(self):
        for ss in [(), (2,)]:
            with self.subTest(stack=ss):
                _, frame, _ = self._pair(ss)
                self.assertLess(float(np.linalg.norm(
                    frame.reverse().reverse().to_dense() - frame.to_dense())), 1e-10)

    def test_variations_reverse_commutes_and_involution(self):
        _, _, var = self._pair()
        var.reverse().validate()
        self.assertEqual(self._cerr(var.reverse().to_t3variations().data,
                                    var.to_t3variations().reverse().data), 0.0)
        self.assertEqual(self._cerr(var.reverse().reverse().to_t3variations().data,
                                    var.to_t3variations().data), 0.0)

    def test_orthogonalize_reconstructs_base_point(self):
        x, frame, _ = self._pair()
        o = frame.orthogonalize()
        o.validate()
        self.assertLess(float(np.linalg.norm(o.to_dense() - x.to_dense())), 1e-10)

    def test_random_orthogonal(self):
        b = ubv.UT3Frame.random_orthogonal((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
        b.validate()
        self.assertEqual((b.shape, b.stack_shape), ((4, 5, 6), (2,)))

    def test_random_orthogonal_like_matches_structure(self):
        _, frame, _ = self._pair(ss=(2,))
        like = ubv.UT3Frame.random_orthogonal_like(frame)
        like.validate()
        self.assertEqual((like.shape, like.stack_shape), (frame.shape, frame.stack_shape))


class TestSaveLoad(unittest.TestCase):
    """2c-F: UT3Frame / UT3Variations save/load round-trip (3 families: supercores, masks, shape)."""
    def setUp(self):
        np.random.seed(0)

    def _pair(self):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
        return ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))

    def _roundtrip(self, obj, loader, name):
        import tempfile, os
        fname = os.path.join(tempfile.mkdtemp(), name)
        obj.save(fname)
        obj2 = loader(fname)
        obj2.validate()
        for a, b in zip(obj2.supercores, obj.supercores):
            self.assertEqual(float(np.max(np.abs(a - b))), 0.0)            # supercores exact
        for a, b in zip(obj2.masks.data, obj.masks.data):
            self.assertTrue(np.array_equal(a, b))                         # masks exact...
            self.assertTrue(isinstance(a, np.ndarray) and a.dtype == bool)  # ...and host bool
        self.assertEqual(obj2.shape, obj.shape)
        return fname

    def test_frame_save_load(self):
        frame, _ = self._pair()
        self._roundtrip(frame, ubv.UT3Frame.load, 'ut3frame.npz')

    def test_variations_save_load(self):
        _, var = self._pair()
        self._roundtrip(var, ubv.UT3Variations.load, 'ut3var.npz')

    @unittest.skipUnless(HAS_JAX, 'jax not installed')
    def test_load_use_jax_keeps_masks_host(self):
        import t3toolbox.backend.common as common
        frame, _ = self._pair()
        fname = self._roundtrip(frame, ubv.UT3Frame.load, 'ut3frame_jax.npz')
        jb = ubv.UT3Frame.load(fname, use_jax=True)
        self.assertTrue(all(common.is_jax_ndarray(s) for s in jb.supercores))   # supercores -> jax
        self.assertTrue(all(common.is_numpy_ndarray(m) for m in jb.masks.data)) # masks stay host


class TestUT3FrameCheckers(unittest.TestCase):
    """2c-G2: uniform-native per-element checkers (masked-Gram orthogonality, minimal-rank, consistency,
    allclose), verified against the ragged oracle (to_t3frame + per-element)."""
    def setUp(self):
        np.random.seed(0)

    def _frame(self, struct=((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)), ss=()):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        x = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss)
        return ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))[0]

    def _oracle(self, frame, fn):  # ragged per-element verdict via to_t3frame
        tree = frame.to_t3frame()
        if frame.stack_shape == ():
            return np.asarray(bool(fn(tree)))
        return np.array([bool(fn(tree[i])) for i in range(frame.stack_shape[0])])

    def test_is_orthogonal_matches_oracle(self):
        for ss in [(), (2,)]:
            with self.subTest(stack=ss):
                frame = self._frame(ss=ss)
                self.assertEqual(frame.is_orthogonal().shape, ss)        # scalar unstacked, array stacked
                self.assertTrue(np.array_equal(frame.is_orthogonal(),
                                               self._oracle(frame, lambda b: b.is_orthogonal())))
                self.assertTrue(frame.is_orthogonal().all())             # an orthogonal frame

    def test_is_orthogonal_mixed_stack(self):
        good = self._frame()
        bad = ubv.UT3Frame(good.up_tucker_supercore + 0.5, good.down_tt_supercore,
                           good.left_tt_supercore, good.right_tt_supercore, good.shape, good.masks)
        mixed = ubv.UT3Frame.stack([good, bad])
        self.assertTrue(np.array_equal(mixed.is_orthogonal(), np.array([True, False])))

    def test_minimal_rank_checkers_match_oracle(self):
        for struct in [((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)),       # minimal
                       ((10, 11, 12), (4, 5, 4), (1, 2, 3, 1))]:    # non-minimal (tucker rank 4 > 1*3)
            with self.subTest(struct=struct):
                frame = self._frame(struct, ss=(2,))
                self.assertTrue(np.array_equal(frame.has_minimal_ranks,
                                               self._oracle(frame, lambda b: b.has_minimal_ranks)))
                self.assertTrue(np.array_equal(frame.has_numerically_minimal_ranks(),
                                               self._oracle(frame, lambda b: b.has_numerically_minimal_ranks())))

    def test_is_consistent_matches_oracle(self):
        frame = self._frame(ss=(2,))
        self.assertTrue(np.array_equal(frame.is_consistent(), self._oracle(frame, lambda b: b.is_consistent())))
        self.assertTrue(frame.is_consistent().all())

    def test_allclose(self):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        frame = self._frame(ss=(2,))
        self.assertTrue(frame.allclose(frame).all())
        self.assertTrue(frame.allclose(frame.orthogonalize()).all())   # same point, possibly different gauge
        other = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(
            t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1), stack_shape=(2,))))[0]
        self.assertFalse(frame.allclose(other).all())                 # different base points


if __name__ == '__main__':
    unittest.main()
