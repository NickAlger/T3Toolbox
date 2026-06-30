# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Tests for the uniform tangent layer (UT3Tangent), uniform-fix slice 3b-1a.

The UT3Tangent skeleton: structural bundle + K/C inference, vector-space ops, raw coordinate
inner/norm/allclose/normalized, the delegating validity checkers, per-element tangent_space_dimension,
constructors, and reverse. Verified against the ragged manifold.T3Tangent (the equivalence contract:
uniform is a faster ragged, so the real/masked content matches per stack element) and structurally for
the K (tangent) stack the ragged layer cannot carry on a single object. No backend math yet (stack/
unstack tangent reshuffles + to_dense/to_ut3 land in later slices)."""
import unittest
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_basis_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.safety as safety

try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


_STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))   # (shape, tucker_ranks, tt_ranks)


def _uniform_base(x):
    """Orthogonal uniform frame + variations of a (possibly stacked) TuckerTensorTrain."""
    return ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))


def _K_variations(basis, K, seed=0):
    """A random K-stacked variation at ``basis``: stack K + C, the base's gauge masks broadcast along K."""
    np.random.seed(seed)
    gauge = ubv.UT3Variations._variation_masks_of(basis)
    masks = ut3m._broadcast_variation_masks_over_K(gauge, K)
    return ubv.UT3Variations.randn(basis.uniform_variation_shapes, basis.shape,
                                   stack_shape=tuple(K) + basis.stack_shape, masks=masks)


class TestStructureAndInference(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_unstacked_structure(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        v = ut3m.UT3Tangent(*_uniform_base(x))
        self.assertEqual(v.shape, (4, 5, 6))
        self.assertEqual(v.d, 3)
        self.assertEqual((v.stack_shape, v.base_stack_shape, v.tangent_stack_shape), ((), (), ()))

    def test_base_stack_inference(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=(2,))
        v = ut3m.UT3Tangent(*_uniform_base(x))
        self.assertEqual((v.stack_shape, v.base_stack_shape, v.tangent_stack_shape), ((2,), (2,), ()))

    def test_tangent_stack_inference(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, _ = _uniform_base(x)
        v = ut3m.UT3Tangent(B, _K_variations(B, (3,)))
        self.assertEqual((v.stack_shape, v.base_stack_shape, v.tangent_stack_shape), ((3,), (), (3,)))

    def test_tangent_plus_base_stack_inference(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=(2,))
        B, _ = _uniform_base(x)
        v = ut3m.UT3Tangent(B, _K_variations(B, (3,)))      # stack K+C = (3, 2)
        self.assertEqual((v.stack_shape, v.base_stack_shape, v.tangent_stack_shape), ((3, 2), (2,), (3,)))

    def test_repr_and_data(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, V = _uniform_base(x)
        v = ut3m.UT3Tangent(B, V)
        self.assertIn('UT3Tangent', repr(v))
        self.assertEqual(v.data, (B, V))

    def test_validate_rejects_incompatible_pair(self):
        # variations whose tangent stack is fine but masks mismatch the base -> check_ubv_pair fails.
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, _ = _uniform_base(x)
        bad = ubv.UT3Variations.randn(B.uniform_variation_shapes, B.shape)   # default all-True masks
        with self.assertRaises(ValueError):
            ut3m.UT3Tangent(B, bad)


class TestVectorSpace(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def _v(self, stack_shape=(), K=()):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=stack_shape)
        B, V = _uniform_base(x)
        if K:
            V = _K_variations(B, K)
        return ut3m.UT3Tangent(B, V)

    def test_add_sub_scale_neg(self):
        for stack_shape, K in [((), ()), ((2,), ()), ((), (3,)), ((2,), (3,))]:
            with self.subTest(stack_shape=stack_shape, K=K):
                v = self._v(stack_shape, K)
                self.assertTrue((2.0 * v - v).allclose(v).all())
                self.assertTrue((v + v).allclose(2.0 * v).all())
                self.assertTrue((-v).allclose(v * (-1.0)).all())
                self.assertTrue((v * 3.0).allclose(3.0 * v).all())   # __rmul__

    def test_add_requires_same_frame(self):
        x1 = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, V = _uniform_base(x1)
        v = ut3m.UT3Tangent(B, V)
        # a DIFFERENT frame with the SAME masks/shape (perturb the supercores, keep masks) -> frames differ.
        B2 = ubv.UT3Basis(B.up_tucker_supercore + 0.1, B.down_tt_supercore + 0.1,
                          B.left_tt_supercore + 0.1, B.right_tt_supercore + 0.1, B.shape, B.masks)
        v2 = ut3m.UT3Tangent(B2, V.copy())
        with self.assertRaises(ValueError):
            v + v2
        with safety.unsafe():                                 # numerical check skipped -> no raise
            _ = v + v2

    def test_add_requires_same_stack(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, V = _uniform_base(x)
        v = ut3m.UT3Tangent(B, V)
        vK = ut3m.UT3Tangent(B, _K_variations(B, (2,)))
        with self.assertRaises(ValueError):
            v + vK

    def test_add_requires_same_masks(self):
        # tangents at bases of DIFFERENT rank -> different masks (structural) -> raises (even unsafe).
        xa = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        xb = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 1, 1))
        va = ut3m.UT3Tangent(*_uniform_base(xa))
        vb = ut3m.UT3Tangent(*_uniform_base(xb))
        with self.assertRaises(ValueError):
            va + vb
        with self.assertRaises(ValueError), safety.unsafe():   # structural -> raises even unsafe
            va + vb


class TestCoordinateMetricVsRagged(unittest.TestCase):
    """The equivalence contract: uniform corewise inner/norm/dim == ragged on the real (masked) content."""
    def setUp(self):
        np.random.seed(0)

    def _pair(self, stack_shape=()):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=stack_shape)
        uv = ut3m.UT3Tangent(*_uniform_base(x))
        rb, rv = bvf.t3_orthogonal_representations(x)
        return uv, t3m.T3Tangent(rb, rv)

    def test_corewise_norm_matches_ragged(self):
        for stack_shape in [(), (2,)]:
            with self.subTest(stack_shape=stack_shape):
                uv, rv = self._pair(stack_shape)
                self.assertTrue(np.allclose(np.asarray(uv.corewise_norm()),
                                            np.asarray(rv.corewise_norm())))

    def test_corewise_inner_matches_ragged(self):
        uv, rv = self._pair()
        self.assertTrue(np.allclose(float(uv.corewise_inner(uv)), float(rv.corewise_inner(rv))))

    def test_normalized_unit_norm(self):
        for stack_shape in [(), (2,)]:
            with self.subTest(stack_shape=stack_shape):
                uv, _ = self._pair(stack_shape)
                self.assertTrue(np.allclose(np.asarray(uv.normalized().corewise_norm()), 1.0))

    def test_corewise_norm_ignores_garbage_padding(self):
        # the uniform layer is a faster ragged: garbage in the masked-out padding must not affect the norm.
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, V = _uniform_base(x)
        clean = float(ut3m.UT3Tangent(B, V).corewise_norm())
        # real-region indicator (1 in real slots, 0 in padding) = apply_masks on an all-ones variation.
        tkv, ttv = V.supercores
        ones = ubv.UT3Variations(np.ones_like(tkv), np.ones_like(ttv), V.shape, V.masks).apply_masks()
        m_tkv, m_ttv = ones.supercores
        ck_tkv, ck_ttv = V.apply_masks().supercores            # clean real content, padding already 0
        corrupt = ubv.UT3Variations(ck_tkv + 100.0 * (1.0 - m_tkv),  # garbage ONLY in the padding
                                    ck_ttv + 100.0 * (1.0 - m_ttv), V.shape, V.masks)
        self.assertTrue(np.allclose(float(ut3m.UT3Tangent(B, corrupt).corewise_norm()), clean))


class TestCheckersAndDimension(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_is_orthogonal_matches_basis(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=(2,))
        B, V = _uniform_base(x)
        v = ut3m.UT3Tangent(B, V)
        self.assertTrue(np.array_equal(np.asarray(v.is_orthogonal()), np.asarray(B.is_orthogonal())))
        self.assertTrue(v.is_orthogonal().all())

    def test_minimal_rank_checkers_delegate(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, V = _uniform_base(x)
        v = ut3m.UT3Tangent(B, V)
        self.assertEqual(bool(v.has_minimal_ranks), bool(B.has_minimal_ranks))
        self.assertEqual(bool(v.has_numerically_minimal_ranks()), bool(B.has_numerically_minimal_ranks()))

    def test_tangent_space_dimension_unstacked_int_matches_ragged(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        v = ut3m.UT3Tangent(*_uniform_base(x))
        rb, rv = bvf.t3_orthogonal_representations(x)
        d = v.tangent_space_dimension
        self.assertIsInstance(d, int)
        self.assertEqual(d, t3m.T3Tangent(rb, rv).tangent_space_dimension)

    def test_tangent_space_dimension_stacked_array(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=(2,))
        v = ut3m.UT3Tangent(*_uniform_base(x))
        d = v.tangent_space_dimension
        self.assertEqual(np.asarray(d).shape, (2,))
        # each base point has the same structure here, so the per-element dims agree with the ragged dim.
        rb, rv = bvf.t3_orthogonal_representations(x)
        # ragged stacked dim is a single int (shared ranks); compare every uniform element to it.
        self.assertTrue(np.all(np.asarray(d) == t3m.T3Tangent(rb, rv).tangent_space_dimension))

    def test_dimension_shared_with_K_is_base_stack_shaped(self):
        # K vectors share the base -> the tangent-space dimension is per base point (shape C), not K+C.
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=(2,))
        B, _ = _uniform_base(x)
        v = ut3m.UT3Tangent(B, _K_variations(B, (3,)))         # stack K+C=(3,2)
        self.assertEqual(np.asarray(v.tangent_space_dimension).shape, (2,))   # == base stack C


class TestReverseAndDtype(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_reverse_reverses_both(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, V = _uniform_base(x)
        v = ut3m.UT3Tangent(B, V)
        r = v.reverse()
        self.assertEqual(r.shape, (6, 5, 4))
        # reversing twice is identity (on the real content)
        self.assertTrue(r.reverse().allclose(v).all())

    def test_copy_is_independent(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        v = ut3m.UT3Tangent(*_uniform_base(x))
        c = v.copy()
        self.assertTrue(c.allclose(v).all())
        self.assertIsNot(c.basis.up_tucker_supercore, v.basis.up_tucker_supercore)

    @unittest.skipUnless(HAS_JAX, "jax not installed")
    def test_to_jax_to_numpy_roundtrip(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        v = ut3m.UT3Tangent(*_uniform_base(x))
        vj = v.to_jax()
        self.assertTrue(vj.contains_jax)
        vn = vj.to_numpy()
        self.assertFalse(vn.contains_jax)
        # jax defaults to float32, so the round-trip is float32-precise: the recovered frame differs from
        # the float64 original by ~1e-7, which (correctly) trips the strict same-frame guard. Skip it
        # (unsafe) and compare the values loosely -- the point here is the dtype round-trip, not the metric.
        with safety.unsafe():
            self.assertTrue(vn.allclose(v, rtol=1e-5).all())


class TestConstructors(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_zeros_is_additive_identity(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, V = _uniform_base(x)
        v = ut3m.UT3Tangent(B, V)
        z = ut3m.UT3Tangent.zeros(B)
        self.assertTrue((v + z).allclose(v).all())
        self.assertTrue(np.allclose(float(z.corewise_norm()), 0.0))

    def test_zeros_K_stack_masks_match_base(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=(2,))
        B, _ = _uniform_base(x)
        z = ut3m.UT3Tangent.zeros(B, stack_shape=(3,))
        self.assertEqual(z.stack_shape, (3, 2))
        ubv.check_ubv_pair(z.basis, z.variations)              # masks broadcast over K stay consistent
        self.assertTrue(np.allclose(np.asarray(z.corewise_norm()), 0.0))

    def test_zeros_like(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, _ = _uniform_base(x)
        v = ut3m.UT3Tangent(B, _K_variations(B, (3,)))
        z = ut3m.UT3Tangent.zeros_like(v)
        self.assertEqual(z.stack_shape, v.stack_shape)
        self.assertTrue(np.allclose(np.asarray(z.corewise_norm()), 0.0))

    def test_unit_is_nonzero_single_entry(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, _ = _uniform_base(x)
        u = ut3m.UT3Tangent.unit(B, (False, 0, (0, 0)))        # one tucker-variation entry
        ubv.check_ubv_pair(u.basis, u.variations)
        self.assertGreater(float(u.corewise_norm()), 0.0)


@unittest.skipUnless(HAS_JAX, "jax not installed")
class TestPytree(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_flatten_unflatten_roundtrip(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        v = ut3m.UT3Tangent(*_uniform_base(x)).to_jax()
        leaves, treedef = jax.tree_util.tree_flatten(v)
        v2 = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertTrue(v2.to_numpy().allclose(v.to_numpy()).all())

    def test_jit_identity_keeps_masks_concrete(self):
        # the base flows as a traced leaf; the masks ride as static aux -> a jitted identity round-trips.
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        v = ut3m.UT3Tangent(*_uniform_base(x)).to_jax()
        out = jax.jit(lambda t: t * 2.0)(v)
        self.assertTrue(out.to_numpy().allclose((v * 2.0).to_numpy()).all())

    def test_jit_corewise_norm(self):
        # the xnp reduction path under a trace: a stray np.* on a tracer would raise. Masks stay host.
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        v = ut3m.UT3Tangent(*_uniform_base(x)).to_jax()
        n = jax.jit(lambda t: t.corewise_norm())(v)
        self.assertGreater(float(n), 0.0)


if __name__ == '__main__':
    unittest.main()
