# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Tests for the uniform tangent layer (UT3Tangent), uniform-fix slice 3b-1a.

The UT3Tangent skeleton: structural bundle + K/C inference, vector-space ops, raw coordinate
inner/norm/allclose/normalized, the delegating validity checkers, per-element tangent_space_dimension,
constructors, and reverse. Verified against the ragged manifold.T3Tangent (the equivalence contract:
for tangents the uniform layer is a *faster* ragged -- same representational power, so the real/masked
content matches per stack element; the K and C stacks are both carried by ragged too). The K (tangent)
stack here is exercised on the uniform side; an explicit K-stack-vs-ragged equivalence check lands in
3b-1b alongside the tangent stack/unstack conversions. No backend math yet (the stack/unstack tangent
conversions + to_dense/to_ut3 land in later slices)."""
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


_HETERO = [((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)),     # two models of DIFFERENT base ranks (rank-sweep),
           ((4, 5, 6), (3, 3, 2), (1, 1, 2, 1))]     # padded to common dims so they stack on one C batch
_HETERO_PAD = dict(N=6, nU=4, nD=4, rL=3, rR=3)


def _hetero_tangents(seed=0):
    """A list of single-base UT3Tangents at DIFFERENT ranks (the varying-C rank-sweep case), plus the
    matching ragged T3Tangents for the equivalence checks."""
    np.random.seed(seed)
    us, rs = [], []
    for s in _HETERO:
        x = t3.TuckerTensorTrain.randn(*s)
        rb, rv = bvf.t3_orthogonal_representations(x)
        ub = ubv.UT3Basis.from_t3basis(rb, **_HETERO_PAD)
        uv = ubv.UT3Variations.from_t3variations(rv, **_HETERO_PAD)
        us.append(ut3m.UT3Tangent(ub, uv))
        rs.append(t3m.T3Tangent(rb, rv))
    return us, rs


# ------------------------------------------------------------------ 3b-4c hardening helpers
# Forced padding STRICTLY above the real max ranks of _STRUCT (nU=3, nD<=4, rL=rR=2; N=6) so EVERY core
# has a padded region (default from_t3 pads to max(ranks), leaving the max-rank core unpadded).
_PAD_T3   = dict(N=8, n=5, r=4)                                  # UniformTuckerTensorTrain.from_t3
_PAD_BV   = dict(N=8, nU=5, nD=5, rL=4, rR=4)                    # from_t3basis / from_t3variations


def _base(x, force_pad=False):
    """Orthogonal uniform frame + variations of x; force_pad pads every core (vs the tight default)."""
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, **_PAD_T3) if force_pad else ut3.UniformTuckerTensorTrain.from_t3(x)
    return ubv.ut3_orthogonal_representations(ux)


def _corrupt(obj, scale=1e3):
    """Add ``scale`` * garbage to ``obj``'s masked-out (padding) region; the real region is unchanged.

    The contract says padding is garbage-don't-care, so a correct (mask-once) op must be UNAFFECTED by
    this; an op that reads raw padding -- or whose output mask is too permissive -- will leak the garbage.
    ``obj``: UT3Variations / UT3Basis / UniformTuckerTensorTrain (uses its own -- correct -- mask)."""
    scs = obj.supercores
    if isinstance(obj, ubv.UT3Variations):
        ind = ubv.UT3Variations(*[np.ones_like(s) for s in scs], obj.shape, obj.masks).apply_masks().supercores
        new = [sc + scale * (1.0 - i) for sc, i in zip(scs, ind)]
        return ubv.UT3Variations(new[0], new[1], obj.shape, obj.masks)
    if isinstance(obj, ubv.UT3Basis):
        ind = ubv.UT3Basis(*[np.ones_like(s) for s in scs], obj.shape, obj.masks).apply_masks().supercores
        new = [sc + scale * (1.0 - i) for sc, i in zip(scs, ind)]
        return ubv.UT3Basis(new[0], new[1], new[2], new[3], obj.shape, obj.masks)
    if isinstance(obj, ut3.UniformTuckerTensorTrain):
        ind = ut3.UniformTuckerTensorTrain(*[np.ones_like(s) for s in scs], obj.shape, obj.masks).apply_masks().supercores
        new = [sc + scale * (1.0 - i) for sc, i in zip(scs, ind)]
        return ut3.UniformTuckerTensorTrain(new[0], new[1], obj.shape, obj.masks)
    raise TypeError(type(obj))


def _expected_doubled_masks(basis):
    """Independently build the doubled (tucker_edge_mask, tt_edge_mask) from the BASE ranks + the paper
    rule (eqs 50-53) -- a different derivation than tangent_to_ut3's concat, so the comparison is
    non-circular. Tucker = prefix-pair [up | down]; TT bond = [Q-block | P-block] with honest boundaries
    (Q0=0, Qi=right_rank_i; Pd=0, Pi=left_rank_i)."""
    nU, nD, rL, rR = basis.nU, basis.nD, basis.rL, basis.rR
    up_r, down_r = np.asarray(basis.up_ranks), np.asarray(basis.down_ranks)      # (d,)  +C
    left_r, right_r = np.asarray(basis.left_ranks), np.asarray(basis.right_ranks)  # (d+1,)+C
    tucker = np.concatenate([np.arange(nU) < up_r[..., None],
                             np.arange(nD) < down_r[..., None]], axis=-1)        # (d,)+C+(nU+nD,)
    q = right_r.copy(); q[0] = 0                                                 # Q honest at the left boundary
    p = left_r.copy();  p[-1] = 0                                                # P honest at the right boundary
    tt = np.concatenate([np.arange(rR) < q[..., None],
                         np.arange(rL) < p[..., None]], axis=-1)                 # (d+1,)+C+(rR+rL,)  [Q | P]
    return tucker, tt


def _bc_over_K(m, K):  # (lead,)+C+(size,) -> (lead,)+K+C+(size,): a base mask broadcast over the tangent stack
    return np.broadcast_to(m.reshape(m.shape[:1] + (1,) * len(K) + m.shape[1:]),
                           m.shape[:1] + tuple(K) + m.shape[1:])


def _prefix(ranks, size):  # int ranks -> boolean prefix mask of width `size` (the canonical form)
    return np.arange(size) < np.asarray(ranks)[..., None]


def _is_prefix(mask):  # True if every edge of `mask` is a boolean prefix (canonical form), i.e. no gaps
    return bool(np.array_equal(mask, _prefix(mask.sum(axis=-1), mask.shape[-1])))


# (stack_shape, K, force_pad): unstacked / C / K / K+C / forced-pad / multi-axis C / multi-axis K
_CONFIGS = [((), (), False), ((2,), (), False), ((), (3,), False), ((2,), (3,), False),
            ((), (), True), ((2,), (), True), ((2, 3), (), False), ((), (2, 3), False)]


def _make_tangent(stack_shape=(), K=(), force_pad=False, seed=0):
    np.random.seed(seed)
    B, V = _base(t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=stack_shape), force_pad=force_pad)
    return ut3m.UT3Tangent(B, _K_variations(B, K) if K else V)


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


class TestStackUnstack(unittest.TestCase):
    """3b-1b: the tangent stack/unstack conversions (tree <-> stacked tangent) + sum_tangents over K."""
    def setUp(self):
        np.random.seed(0)

    def _v(self, stack_shape=(), K=()):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=stack_shape)
        B, V = _uniform_base(x)
        return ut3m.UT3Tangent(B, _K_variations(B, K) if K else V)

    # ------- tangent (K) stack -------
    def test_unstack_stack_tangents_roundtrip(self):
        v = self._v(K=(3,))
        leaves = v.unstack_tangents()
        self.assertEqual(len(leaves), 3)
        self.assertTrue(all(t.tangent_stack_shape == () and t.base_stack_shape == () for t in leaves))
        self.assertTrue(all(t.basis is v.basis for t in leaves))            # one shared frame across K
        self.assertTrue(ut3m.UT3Tangent.stack_tangents(leaves).allclose(v).all())

    def test_unstack_stack_tangents_over_base_stack(self):
        v = self._v(stack_shape=(2,), K=(3,))                               # K+C = (3, 2)
        leaves = v.unstack_tangents()                                       # peel K=3, each leaf base_stack=(2,)
        self.assertEqual(len(leaves), 3)
        self.assertTrue(all(t.base_stack_shape == (2,) and t.tangent_stack_shape == () for t in leaves))
        self.assertTrue(ut3m.UT3Tangent.stack_tangents(leaves).allclose(v).all())

    def test_stack_tangents_requires_same_frame(self):
        v = self._v(K=())
        B2 = ubv.UT3Basis(v.basis.up_tucker_supercore + 0.1, v.basis.down_tt_supercore + 0.1,
                          v.basis.left_tt_supercore + 0.1, v.basis.right_tt_supercore + 0.1,
                          v.basis.shape, v.basis.masks)
        v2 = ut3m.UT3Tangent(B2, v.variations.copy())
        with self.assertRaises(ValueError):
            ut3m.UT3Tangent.stack_tangents((v, v2))
        with safety.unsafe():                                              # numerical check skipped
            self.assertEqual(ut3m.UT3Tangent.stack_tangents((v, v2)).tangent_stack_shape, (2,))

    # ------- base (C) stack -------
    def test_unstack_stack_basis_roundtrip(self):
        v = self._v(stack_shape=(2,), K=(3,))
        leaves = v.unstack_basis()                                          # peel C=2, each leaf tangent_stack=(3,)
        self.assertEqual(len(leaves), 2)
        self.assertTrue(all(t.base_stack_shape == () and t.tangent_stack_shape == (3,) for t in leaves))
        self.assertTrue(ut3m.UT3Tangent.stack_basis(leaves).allclose(v).all())

    def test_stack_basis_requires_matching_padded_dims_and_K(self):
        a = self._v(stack_shape=(), K=())
        # different tangent stack K -> reject
        b = self._v(stack_shape=(), K=(2,))
        with self.assertRaises(ValueError):
            ut3m.UT3Tangent.stack_basis((a, b))

    # ------- varying ranks across C (the rank-sweep use case) -------
    def test_stack_basis_varying_C_ranks(self):
        us, rs = _hetero_tangents()
        stacked = ut3m.UT3Tangent.stack_basis(us)                          # different base ranks in one C batch
        self.assertEqual(stacked.base_stack_shape, (2,))
        # per-element tangent-space dims differ and match the per-model ragged dims
        dims = np.asarray(stacked.tangent_space_dimension)
        self.assertEqual(dims.shape, (2,))
        self.assertEqual(list(dims), [rs[0].tangent_space_dimension, rs[1].tangent_space_dimension])
        self.assertNotEqual(dims[0], dims[1])                              # genuinely varying
        # unstack recovers each model
        back = stacked.unstack_basis()
        self.assertTrue(back[0].allclose(us[0]).all() and back[1].allclose(us[1]).all())

    def test_varying_C_corewise_norm_per_element_vs_ragged(self):
        us, rs = _hetero_tangents()
        stacked = ut3m.UT3Tangent.stack_basis(us)
        un = np.asarray(stacked.corewise_norm())
        self.assertTrue(np.allclose(un, [float(rs[0].corewise_norm()), float(rs[1].corewise_norm())]))

    # ------- K-stack equivalence vs the ragged T3Tangent (the deferred 3b-1a check) -------
    def test_K_stack_equivalence_vs_ragged(self):
        v = self._v(K=(3,))
        un = np.asarray(v.corewise_norm())                                 # per-K uniform norms
        for k, leaf in enumerate(v.unstack_tangents()):
            rt = t3m.T3Tangent(leaf.basis.to_t3basis(), leaf.variations.to_t3variations())
            self.assertTrue(np.allclose(float(un[k]), float(rt.corewise_norm())))

    # ------- sum over K -------
    def test_sum_tangents_all(self):
        v = self._v(K=(3,))
        leaves = v.unstack_tangents()
        summed = v.sum_tangents()
        self.assertEqual(summed.stack_shape, ())
        self.assertTrue(summed.allclose(leaves[0] + leaves[1] + leaves[2]).all())

    def test_sum_tangents_single_axis(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        B, _ = _uniform_base(x)
        v = ut3m.UT3Tangent(B, _K_variations(B, (2, 3)))                   # K = (2, 3)
        s = v.sum_tangents(axis=0)                                         # sum the outer K axis -> K=(3,)
        self.assertEqual(s.tangent_stack_shape, (3,))
        # cross-check against an explicit sum over that axis of the masked supercores
        tkv = np.asarray(v.variations.apply_masks().tucker_variations).sum(axis=1)
        self.assertTrue(np.allclose(np.asarray(s.variations.apply_masks().tucker_variations), tkv))

    @unittest.skipUnless(HAS_JAX, "jax not installed")
    def test_stack_basis_jax_supercores_host_masks(self):
        us, _ = _hetero_tangents()
        stacked = ut3m.UT3Tangent.stack_basis([t.to_jax() for t in us])
        import t3toolbox.backend.common as common
        self.assertTrue(stacked.contains_jax)
        self.assertTrue(all(common.is_jax_ndarray(sc) for sc in stacked.variations.supercores))
        self.assertTrue(all(common.is_numpy_ndarray(m) for m in stacked.variations.masks.data))


def _full_unstack(v):
    """Fully unstack a UT3Tangent into a FLAT list of single-element (unstacked) tangents, K-major then C.

    Handles multi-axis stacks: `unstack_tangents` / `unstack_basis` return a NESTED tree for a multi-axis
    stack, so flatten it (row-major, matching `reshape(d, -1)` order) before recursing."""
    if not v.stack_shape:
        return [v]
    sub = v.unstack_tangents() if v.tangent_stack_shape else v.unstack_basis()
    out = []
    for leaf in ut3m._flatten_tangents(sub):        # flatten the (possibly nested) tree, row-major
        out.extend(_full_unstack(leaf))
    return out


def _ragged_dense(leaf, shift):  # leaf: an unstacked UT3Tangent -> ragged dense ground truth
    return leaf.to_t3tangent().to_dense(include_shift=shift)


class TestDoubledRankToDense(unittest.TestCase):
    """3b-2: the doubled-rank keystone -- UT3Tangent.to_ut3 / to_dense verified per stack element against
    the ragged T3Tangent.to_dense (the equivalence contract), across structures / stacks / shift modes."""
    def setUp(self):
        np.random.seed(0)

    def _check(self, v, shift):
        dense = np.asarray(v.to_dense(include_shift=shift))                 # stack (K+C) + (N..)
        flat = _full_unstack(v)
        dflat = dense.reshape((-1,) + dense.shape[len(v.stack_shape):])
        for i, leaf in enumerate(flat):
            o = _ragged_dense(leaf, shift)
            self.assertLess(float(np.linalg.norm(dflat[i] - o)) / (float(np.linalg.norm(o)) + 1e-30), 1e-10)

    def test_structures_and_stacks(self):
        structs = [((4, 5, 6), (2, 3, 2), (1, 2, 2, 1)),                    # d=3
                   ((4, 5), (2, 3), (1, 2, 1)),                             # d=2
                   ((4, 5, 6, 7), (2, 3, 2, 3), (1, 2, 3, 2, 1))]          # d=4, rL != rR per bond
        for s in structs:
            B, V = _uniform_base(t3.TuckerTensorTrain.randn(*s))
            for shift in (False, True):
                with self.subTest(s=s, shift=shift):
                    self._check(ut3m.UT3Tangent(B, V), shift)

    def test_base_and_tangent_stacks(self):
        # all of _CONFIGS: C / K / K+C / forced-pad / multi-axis C / multi-axis K (D, E)
        for ss, K, fp in _CONFIGS:
            v = _make_tangent(ss, K, fp)
            for shift in (False, True):
                with self.subTest(ss=ss, K=K, fp=fp, shift=shift):
                    self._check(v, shift)

    def test_varying_C_ranks(self):
        us, _ = _hetero_tangents()
        v = ut3m.UT3Tangent.stack_basis(us)                                # different ranks across C
        for shift in (False, True):
            with self.subTest(shift=shift):
                self._check(v, shift)

    def test_to_ut3_is_doubled_uniform_t3(self):
        B, V = _uniform_base(t3.TuckerTensorTrain.randn(*_STRUCT))
        du = ut3m.UT3Tangent(B, V).to_ut3()
        self.assertIsInstance(du, ut3.UniformTuckerTensorTrain)
        self.assertEqual(du.n, B.nU + B.nD)                                # doubled Tucker padding
        self.assertEqual(du.r, B.rL + B.rR)                                # doubled TT bond padding
        # honest masks (eqs 50-53): doubled Tucker rank = up + down; TT bonds = left + right INTERIOR, but
        # the two boundary bonds stay rank 1 (the global "1" is in one block; the free block has rank 0).
        lr, rr = np.asarray(B.left_ranks), np.asarray(B.right_ranks)       # (d+1,)
        expected_tt = lr + rr
        expected_tt[0] = lr[0]; expected_tt[-1] = rr[-1]                   # boundaries: free block has no rank
        self.assertTrue(np.array_equal(np.asarray(du.tt_ranks), expected_tt))
        self.assertTrue(np.array_equal(np.asarray(du.tucker_ranks),
                                       np.asarray(B.up_ranks) + np.asarray(B.down_ranks)))

    def test_d1_not_implemented(self):
        import t3toolbox.backend.ubv_tangent_operations as ubvt
        B, V = _uniform_base(t3.TuckerTensorTrain.randn(*_STRUCT))
        # call the backend with a faked d=1 by slicing to one core is fragile; instead assert the guard
        # message is reachable through a genuine d=1 build path is out of scope -- check the guard directly.
        with self.assertRaises(NotImplementedError):
            # craft a minimal d=1 .data pair (square bonds so shapes are valid) to hit the guard
            d1_basis = (np.random.randn(1, 2, 5), np.random.randn(1, 1, 2, 1), np.random.randn(1, 1, 2, 1),
                        np.random.randn(1, 1, 2, 1), (5,),
                        (np.ones((1, 2), bool), np.ones((1, 1), bool), np.ones((2, 1), bool), np.ones((2, 1), bool)))
            d1_var = (np.random.randn(1, 1, 5), np.random.randn(1, 1, 2, 1), (5,),
                      (np.ones((1, 2), bool), np.ones((1, 1), bool), np.ones((1, 1), bool), np.ones((1, 1), bool)))
            ubvt.tangent_to_ut3(d1_basis, d1_var)

    @unittest.skipUnless(HAS_JAX, "jax not installed")
    def test_jit_to_dense(self):
        # the doubled build + dense under a trace: supercores flow through xnp, masks stay host constants.
        B, V = _uniform_base(t3.TuckerTensorTrain.randn(*_STRUCT))
        v = ut3m.UT3Tangent(B, V).to_jax()
        dense = jax.jit(lambda t: t.to_dense())(v)
        self.assertTrue(np.allclose(np.asarray(dense), np.asarray(v.to_numpy().to_dense()), atol=1e-5))


def _wrap_ut3(data):
    return ut3.UniformTuckerTensorTrain(data[0], data[1], data[2], ut3.UT3Masks(*data[3]))


class TestExactMasks(unittest.TestCase):
    """3b-4c (B): assert EXACT output masks (not just dense), so a too-permissive mask (phantom rank) is
    caught -- the bug class our dense-on-clean-padding tests are blind to. Expected masks are derived
    independently (base ranks + the paper rule), so the comparison is non-circular."""
    def setUp(self):
        np.random.seed(0)
        import t3toolbox.backend.ubv_tangent_operations as ubvt
        self.ubvt = ubvt

    def test_doubled_masks(self):
        for ss, K, fp in _CONFIGS:
            v = _make_tangent(ss, K, fp)
            du = v.to_ut3()
            exp_tk, exp_tt = _expected_doubled_masks(v.basis)
            with self.subTest(ss=ss, K=K, fp=fp):
                self.assertTrue(np.array_equal(du.masks.tucker_edge_mask, _bc_over_K(exp_tk, K)))
                self.assertTrue(np.array_equal(du.masks.tt_edge_mask, _bc_over_K(exp_tt, K)))

    def test_retract_masks_are_prefix(self):
        # the retracted ranks can drop BELOW the base ranks (base+tangent may have lower rank -- matches
        # ragged), so the ground truth is the ragged retract's ranks, checked per element in TestRetract.
        # Here we only assert the masks are canonical (prefix), which a garbage/non-prefix mask would fail.
        for ss, K, fp in _CONFIGS:
            v = _make_tangent(ss, K, fp)
            ru = _wrap_ut3(self.ubvt.retract(v.basis.data, v.variations.data))
            with self.subTest(ss=ss, K=K, fp=fp):
                self.assertTrue(_is_prefix(ru.masks.tucker_edge_mask))
                self.assertTrue(_is_prefix(ru.masks.tt_edge_mask))

    def test_gauge_preserves_masks(self):
        for which in ('orthogonal_gauge_projection', 'oblique_gauge_projection'):
            for ss, K, fp in _CONFIGS:
                v = _make_tangent(ss, K, fp)
                gd = getattr(self.ubvt, which)(v.basis.data, v.variations.data)
                with self.subTest(which=which, ss=ss, K=K, fp=fp):
                    self.assertEqual(_wrap_var(gd).masks, v.variations.masks)   # gauge must not change the masks

    def test_project_masks_are_gauge_masks(self):
        for ss, _K, fp in [c for c in _CONFIGS if not c[1]]:                    # project is C-only (no K)
            B, _ = _base(t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=ss), force_pad=fp)
            xu = (ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=ss), **_PAD_T3)
                  if fp else ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=ss)))
            pv = _wrap_var(self.ubvt.project_ut3_onto_tangent_space(B.data, xu.data))
            with self.subTest(ss=ss, fp=fp):
                self.assertEqual(pv.masks, ubv.UT3Variations._variation_masks_of(B))

    def test_stack_unstack_preserves_masks(self):
        # full unstack then restack: the masks must round-trip exactly (array-level, not just dense)
        v = _make_tangent((2,), (3,))
        for leaf in _full_unstack(v):
            back = ut3m.UT3Tangent.from_t3tangent(leaf.to_t3tangent())
            self.assertEqual(back.variations.masks, leaf.variations.masks)
        # varying-C stack_basis preserves each element's (different) masks
        us, _ = _hetero_tangents()
        stacked = ut3m.UT3Tangent.stack_basis(us)
        for i, leaf in enumerate(stacked.unstack_basis()):
            self.assertEqual(leaf.variations.masks, us[i].variations.masks)

    def test_doubled_masks_varying_C(self):
        us, _ = _hetero_tangents()
        v = ut3m.UT3Tangent.stack_basis(us)
        exp_tk, exp_tt = _expected_doubled_masks(v.basis)
        du = v.to_ut3()
        self.assertTrue(np.array_equal(du.masks.tucker_edge_mask, exp_tk))
        self.assertTrue(np.array_equal(du.masks.tt_edge_mask, exp_tt))


class TestGarbageInputRobustness(unittest.TestCase):
    """3b-4c (A): garbage in the masked-out padding is don't-care -- a correct (mask-once / tight-output-
    mask) op must give the IDENTICAL result on a garbage-padded input as on a clean one. Catches ops that
    read raw padding, and (for tangent_to_ut3, which builds from raw supercores) a too-permissive output
    mask that leaks the garbage."""
    def setUp(self):
        np.random.seed(0)
        import t3toolbox.backend.ubv_tangent_operations as ubvt
        self.ubvt = ubvt

    def _bv(self, ss, K, fp):
        v = _make_tangent(ss, K, fp)
        return v.basis, v.variations

    def test_tangent_to_ut3(self):
        for ss, K, fp in _CONFIGS:
            B, V = self._bv(ss, K, fp)
            Bc, Vc = _corrupt(B), _corrupt(V)
            for shift in (False, True):
                clean = np.asarray(ut3m.UT3Tangent(B, V).to_dense(include_shift=shift))
                dirty = np.asarray(ut3m.UT3Tangent(Bc, Vc).to_dense(include_shift=shift))
                with self.subTest(ss=ss, K=K, fp=fp, shift=shift):
                    self.assertTrue(np.allclose(clean, dirty, atol=1e-9))

    def test_retract(self):
        for ss, K, fp in _CONFIGS:
            B, V = self._bv(ss, K, fp)
            clean = np.asarray(_wrap_ut3(self.ubvt.retract(B.data, V.data)).to_dense())
            dirty = np.asarray(_wrap_ut3(self.ubvt.retract(_corrupt(B).data, _corrupt(V).data)).to_dense())
            with self.subTest(ss=ss, K=K, fp=fp):
                self.assertTrue(np.allclose(clean, dirty, atol=1e-9))

    def test_gauge(self):
        for which in ('orthogonal_gauge_projection', 'oblique_gauge_projection'):
            for ss, K, fp in _CONFIGS:
                B, V = self._bv(ss, K, fp)
                proj = getattr(self.ubvt, which)
                clean = np.asarray(ut3m.UT3Tangent(B, _wrap_var(proj(B.data, V.data))).to_dense())
                dirty = np.asarray(ut3m.UT3Tangent(B, _wrap_var(proj(_corrupt(B).data, _corrupt(V).data))).to_dense())
                with self.subTest(which=which, ss=ss, K=K, fp=fp):
                    self.assertTrue(np.allclose(clean, dirty, atol=1e-9))

    def test_gauge_residual(self):
        for ss, K, fp in _CONFIGS:
            B, V = self._bv(ss, K, fp)
            clean = np.asarray(self.ubvt.gauge_residual(B.data, V.data))
            dirty = np.asarray(self.ubvt.gauge_residual(_corrupt(B).data, _corrupt(V).data))
            with self.subTest(ss=ss, K=K, fp=fp):
                self.assertTrue(np.allclose(clean, dirty, atol=1e-9))

    def test_project(self):
        for ss, _K, fp in [c for c in _CONFIGS if not c[1]]:                    # project is C-only
            B, _ = _base(t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=ss), force_pad=fp)
            xt = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=ss)
            xu = ut3.UniformTuckerTensorTrain.from_t3(xt, **_PAD_T3) if fp else ut3.UniformTuckerTensorTrain.from_t3(xt)
            clean = np.asarray(ut3m.UT3Tangent(B, _wrap_var(self.ubvt.project_ut3_onto_tangent_space(B.data, xu.data))).to_dense())
            dirty = np.asarray(ut3m.UT3Tangent(B, _wrap_var(self.ubvt.project_ut3_onto_tangent_space(_corrupt(B).data, _corrupt(xu).data))).to_dense())
            with self.subTest(ss=ss, fp=fp):
                self.assertTrue(np.allclose(clean, dirty, atol=1e-9))


class TestRetract(unittest.TestCase):
    """3b-2: backend retract (shifted doubled-rank -> mask-truncated T3-SVD back to the base ranks),
    verified per stack element against the ragged tangent_operations.retract."""
    def setUp(self):
        np.random.seed(0)
        import t3toolbox.backend.ubv_tangent_operations as ubvt
        import t3toolbox.backend.tangent_operations as tops
        self.ubvt, self.tops = ubvt, tops

    def _check(self, v):
        ru = _wrap_ut3(self.ubvt.retract(v.basis.data, v.variations.data))
        # retracted padded dims = max actual rank across the stack, never exceeding the base padding
        self.assertLessEqual(ru.n, v.basis.nU)
        self.assertLessEqual(ru.r, v.basis.rL)
        self.assertTrue(_is_prefix(ru.masks.tucker_edge_mask) and _is_prefix(ru.masks.tt_edge_mask))  # canonical
        dense = np.asarray(ru.to_dense())
        flat = _full_unstack(v)
        dflat = dense.reshape((-1,) + dense.shape[len(v.stack_shape):])
        # per-element ranks, K-major-then-C (matches _full_unstack); ground truth = the ragged retract
        utk = np.asarray(ru.tucker_ranks).reshape(ru.d, -1)
        utt = np.asarray(ru.tt_ranks).reshape(ru.d + 1, -1)
        for i, leaf in enumerate(flat):
            rT = t3.TuckerTensorTrain(*self.tops.retract(leaf.basis.to_t3basis().data,
                                                         leaf.variations.to_t3variations().data))
            rd = rT.to_dense()
            self.assertLess(float(np.linalg.norm(dflat[i] - rd)) / (float(np.linalg.norm(rd)) + 1e-30), 1e-9)
            # STRICTNESS: the retracted ranks/masks must match the ragged ground truth EXACTLY -- this is
            # what catches a phantom (too-permissive) mask that the dense check alone is blind to.
            self.assertEqual(tuple(int(x) for x in utk[:, i]), tuple(rT.tucker_ranks))
            self.assertEqual(tuple(int(x) for x in utt[:, i]), tuple(rT.tt_ranks))

    def test_stacks(self):
        for ss, K, fp in _CONFIGS:                                          # +forced-pad, +multi-axis (D, E)
            with self.subTest(ss=ss, K=K, fp=fp):
                self._check(_make_tangent(ss, K, fp))

    def test_varying_C_ranks(self):
        us, _ = _hetero_tangents()
        self._check(ut3m.UT3Tangent.stack_basis(us))

    @unittest.skipUnless(HAS_JAX, "jax not installed")
    def test_jit_retract(self):
        # retract under a trace: supercores flow through xnp; masks/shape stay host constants (closed over).
        B, V = _uniform_base(t3.TuckerTensorTrain.randn(*_STRUCT))
        Bj, Vj = B.to_jax(), V.to_jax()
        bsh, bmk = Bj.data[4], Bj.data[5]
        vsh, vmk = Vj.data[2], Vj.data[3]

        def f(u, dn, lt, rt, tk, tt):
            rd = self.ubvt.retract((u, dn, lt, rt, bsh, bmk), (tk, tt, vsh, vmk))
            return _wrap_ut3(rd).to_dense()

        dense = np.asarray(jax.jit(f)(Bj.data[0], Bj.data[1], Bj.data[2], Bj.data[3], Vj.data[0], Vj.data[1]))
        ref = np.asarray(_wrap_ut3(self.ubvt.retract(B.data, V.data)).to_dense())   # float64 ground truth
        self.assertTrue(np.allclose(dense, ref, rtol=1e-3, atol=1e-3))              # jit is float32


def _wrap_var(vd):
    return ubv.UT3Variations(vd[0], vd[1], vd[2], ubv.UT3VariationsMasks(*vd[3]))


class TestGauge(unittest.TestCase):
    """3b-3: the gauge layer -- orthogonal / oblique gauge projection + gauge_residual / is_gauged,
    verified against the ragged tangent_operations (conditions (48)-(49), Appendix A.3)."""
    def setUp(self):
        np.random.seed(0)
        import t3toolbox.backend.ubv_tangent_operations as ubvt
        import t3toolbox.backend.tangent_operations as tops
        self.ubvt, self.tops = ubvt, tops

    def _gauged_tangent(self, v, which):  # apply a backend gauge projection -> a new UT3Tangent
        proj = getattr(self.ubvt, which)
        return ut3m.UT3Tangent(v.basis, _wrap_var(proj(v.basis.data, v.variations.data)))

    def _ragged(self, leaf):
        return leaf.to_t3tangent()

    def test_gauge_residual_matches_ragged(self):
        for ss in [(), (2,)]:
            B, V = _uniform_base(t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=ss))
            v = ut3m.UT3Tangent(B, V)
            ures = np.asarray(v.gauge_residual).reshape(-1)
            for i, leaf in enumerate(_full_unstack(v)):
                with self.subTest(ss=ss, i=i):
                    self.assertAlmostEqual(float(ures[i]), float(np.asarray(self._ragged(leaf).gauge_residual)), places=10)

    def test_is_gauged_false_then_true(self):
        B, V = _uniform_base(t3.TuckerTensorTrain.randn(*_STRUCT))
        v = ut3m.UT3Tangent(B, V)
        self.assertFalse(bool(v.is_gauged().all()))                         # random variations: ungauged
        for which in ('orthogonal_gauge_projection', 'oblique_gauge_projection'):
            with self.subTest(which=which):
                self.assertTrue(bool(self._gauged_tangent(v, which).is_gauged(atol=1e-9).all()))

    def test_orthogonal_gauge_matches_ragged_dense(self):
        # orthogonal projection CHANGES the tangent vector -- compare the changed dense vs ragged per element
        for ss, K, fp in _CONFIGS:                                          # incl. K (item C), forced-pad, multi-axis
            v = _make_tangent(ss, K, fp)
            ug = self._gauged_tangent(v, 'orthogonal_gauge_projection')
            dense = np.asarray(ug.to_dense()); dflat = dense.reshape((-1,) + dense.shape[len(v.stack_shape):])
            for i, leaf in enumerate(_full_unstack(v)):
                rt = self._ragged(leaf)
                rg = t3m.T3Tangent(rt.basis, bvf.T3Variations(*self.tops.orthogonal_gauge_projection(rt.basis.data, rt.variations.data)))
                with self.subTest(ss=ss, K=K, fp=fp, i=i):
                    self.assertLess(float(np.linalg.norm(dflat[i] - rg.to_dense())) / (float(np.linalg.norm(rg.to_dense())) + 1e-30), 1e-10)

    def test_oblique_gauge_preserves_tangent(self):
        # oblique projection PRESERVES the tangent vector (only the representation changes)
        for ss, K, fp in _CONFIGS:
            v = _make_tangent(ss, K, fp)
            ob = self._gauged_tangent(v, 'oblique_gauge_projection')
            with self.subTest(ss=ss, K=K, fp=fp):
                self.assertLess(float(np.linalg.norm(np.asarray(ob.to_dense()) - np.asarray(v.to_dense())))
                                / (float(np.linalg.norm(np.asarray(v.to_dense()))) + 1e-30), 1e-10)
                self.assertTrue(bool(ob.is_gauged().all()))

    def test_varying_C(self):
        us, rs = _hetero_tangents()
        v = ut3m.UT3Tangent.stack_basis(us)
        # per-element residual matches each ragged model
        ures = np.asarray(v.gauge_residual)
        self.assertTrue(np.allclose(ures, [float(np.asarray(r.gauge_residual)) for r in rs]))
        # both projections gauge a varying-C stack; oblique preserves it
        self.assertTrue(bool(self._gauged_tangent(v, 'orthogonal_gauge_projection').is_gauged().all()))
        ob = self._gauged_tangent(v, 'oblique_gauge_projection')
        self.assertTrue(bool(ob.is_gauged().all()))
        self.assertLess(float(np.linalg.norm(np.asarray(ob.to_dense()) - np.asarray(v.to_dense())))
                        / (float(np.linalg.norm(np.asarray(v.to_dense()))) + 1e-30), 1e-10)

    @unittest.skipUnless(HAS_JAX, "jax not installed")
    def test_jit_gauge(self):
        B, V = _uniform_base(t3.TuckerTensorTrain.randn(*_STRUCT))
        v = ut3m.UT3Tangent(B, V).to_jax()
        res = jax.jit(lambda t: t.gauge_residual)(v)                        # gauge_residual under a trace
        self.assertTrue(np.allclose(float(res), float(ut3m.UT3Tangent(B, V).gauge_residual), atol=1e-4))

    @unittest.skipUnless(HAS_JAX, "jax not installed")
    def test_jit_oblique_scan(self):
        # the oblique TT step is an xscan (compile body once, not d unrolled copies); confirm it jits and
        # still preserves the tangent under a trace. d=4 so the scan iterates a few times.
        B, V = _uniform_base(t3.TuckerTensorTrain.randn((4, 5, 6, 7), (2, 3, 2, 3), (1, 2, 3, 2, 1)))
        Bj, Vj = B.to_jax(), V.to_jax()
        bsh, bmk, vsh, vmk = Bj.data[4], Bj.data[5], Vj.data[2], Vj.data[3]

        def f(u, dn, lt, rt, tk, tt):
            bd = (u, dn, lt, rt, bsh, bmk)
            vd = self.ubvt.oblique_gauge_projection(bd, (tk, tt, vsh, vmk))
            return ut3m.UT3Tangent(ut3m._ut3basis_from_data(bd), _wrap_var(vd)).to_dense()

        dense = np.asarray(jax.jit(f)(Bj.data[0], Bj.data[1], Bj.data[2], Bj.data[3], Vj.data[0], Vj.data[1]))
        ref = np.asarray(ut3m.UT3Tangent(B, V).to_dense())                  # oblique preserves the tangent
        self.assertTrue(np.allclose(dense, ref, rtol=1e-3, atol=1e-3))


class TestProjectOntoTangent(unittest.TestCase):
    """3b-4: backend project_ut3_onto_tangent_space (orthogonal projection of a UT3 onto the tangent space
    at an orthogonal base), verified per element against ragged project_t3_onto_tangent_space."""
    def setUp(self):
        np.random.seed(0)
        import t3toolbox.backend.ubv_tangent_operations as ubvt
        import t3toolbox.backend.tangent_operations as tops
        self.ubvt, self.tops = ubvt, tops

    def _check_pair(self, p, x):  # p, x: ragged TuckerTensorTrains (same shape); project x onto tangent at p's frame
        B = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(p))[0]
        xu = ut3.UniformTuckerTensorTrain.from_t3(x)
        uv = ut3m.UT3Tangent(B, _wrap_var(self.ubvt.project_ut3_onto_tangent_space(B.data, xu.data)))
        rb = B.to_t3basis()
        rvar = self.tops.project_t3_onto_tangent_space(rb.data, x.data)
        rd = t3m.T3Tangent(rb, bvf.T3Variations(*rvar)).to_dense()
        self.assertLess(float(np.linalg.norm(np.asarray(uv.to_dense()) - np.asarray(rd))) / (float(np.linalg.norm(rd)) + 1e-30), 1e-10)
        self.assertTrue(bool(uv.is_gauged().all()))

    def test_unstacked_same_and_different_ranks(self):
        S = (4, 5, 6)
        self._check_pair(t3.TuckerTensorTrain.randn(S, (2, 3, 2), (1, 2, 2, 1)),
                         t3.TuckerTensorTrain.randn(S, (2, 3, 2), (1, 2, 2, 1)))     # same ranks
        self._check_pair(t3.TuckerTensorTrain.randn(S, (2, 3, 2), (1, 2, 2, 1)),
                         t3.TuckerTensorTrain.randn(S, (3, 2, 3), (1, 1, 2, 1)))     # x has different ranks
        self._check_pair(t3.TuckerTensorTrain.randn((4, 5, 6, 7), (2, 3, 2, 3), (1, 2, 3, 2, 1)),
                         t3.TuckerTensorTrain.randn((4, 5, 6, 7), (2, 2, 2, 2), (1, 1, 2, 1, 1)))  # d=4

    def test_C_stacked(self):
        S = (4, 5, 6)
        B = ubv.ut3_orthogonal_representations(
            ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(S, (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))))[0]
        xs = t3.TuckerTensorTrain.randn(S, (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
        xu = ut3.UniformTuckerTensorTrain.from_t3(xs)
        uv = ut3m.UT3Tangent(B, _wrap_var(self.ubvt.project_ut3_onto_tangent_space(B.data, xu.data)))
        ud = np.asarray(uv.to_dense())
        rb_tree, x_tree = B.to_t3basis(), xs.unstack()
        for i in range(2):
            rvar = self.tops.project_t3_onto_tangent_space(rb_tree[i].data, x_tree[i].data)
            rd = t3m.T3Tangent(rb_tree[i], bvf.T3Variations(*rvar)).to_dense()
            self.assertLess(float(np.linalg.norm(ud[i] - rd)) / (float(np.linalg.norm(rd)) + 1e-30), 1e-10)
        self.assertTrue(bool(uv.is_gauged().all()))

    def test_varying_C(self):
        # varying-rank base stack (the rank-sweep case) + a per-element x
        pad = dict(N=6, nU=4, nD=4, rL=3, rR=3)
        het = [((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)), ((4, 5, 6), (3, 3, 2), (1, 1, 2, 1))]
        bases, xs = [], []
        for s in het:
            p = t3.TuckerTensorTrain.randn(*s); x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 1, 1))
            bases.append(ubv.UT3Basis.from_t3basis(bvf.t3_orthogonal_representations(p)[0], **pad))
            xs.append((ut3.UniformTuckerTensorTrain.from_t3(x, N=6, n=4, r=3), x))
        B = ubv.UT3Basis.stack(bases)
        xu = ut3.UniformTuckerTensorTrain.stack([xe[0] for xe in xs])
        uv = ut3m.UT3Tangent(B, _wrap_var(self.ubvt.project_ut3_onto_tangent_space(B.data, xu.data)))
        ud = np.asarray(uv.to_dense())
        for i in range(2):
            rvar = self.tops.project_t3_onto_tangent_space(bases[i].to_t3basis().data, xs[i][1].data)
            rd = t3m.T3Tangent(bases[i].to_t3basis(), bvf.T3Variations(*rvar)).to_dense()
            self.assertLess(float(np.linalg.norm(ud[i] - rd)) / (float(np.linalg.norm(rd)) + 1e-30), 1e-10)
        self.assertTrue(bool(uv.is_gauged().all()))

    @unittest.skipUnless(HAS_JAX, "jax not installed")
    def test_jit_project(self):
        B = ubv.ut3_orthogonal_representations(
            ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*_STRUCT)))[0].to_jax()
        xu = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*_STRUCT)).to_jax()
        bsh, bmk, xsh, xmk = B.data[4], B.data[5], xu.data[2], xu.data[3]

        def f(u, dn, lt, rt, xtk, xtt):
            vd = self.ubvt.project_ut3_onto_tangent_space((u, dn, lt, rt, bsh, bmk), (xtk, xtt, xsh, xmk))
            return ut3m.UT3Tangent(ut3m._ut3basis_from_data((u, dn, lt, rt, bsh, bmk)), _wrap_var(vd)).to_dense()

        dense = np.asarray(jax.jit(f)(B.data[0], B.data[1], B.data[2], B.data[3], xu.data[0], xu.data[1]))
        ref = np.asarray(ut3m.UT3Tangent(B.to_numpy(), _wrap_var(self.ubvt.project_ut3_onto_tangent_space(
            B.to_numpy().data, xu.to_numpy().data))).to_dense())
        self.assertTrue(np.allclose(dense, ref, rtol=1e-3, atol=1e-3))


class TestCrossLayerConverters(unittest.TestCase):
    """3b-4: UT3Tangent <-> T3Tangent (to_t3tangent / from_t3tangent)."""
    def setUp(self):
        np.random.seed(0)

    def test_to_t3tangent_unstacked(self):
        B, V = _uniform_base(t3.TuckerTensorTrain.randn(*_STRUCT))
        v = ut3m.UT3Tangent(B, V)
        rt = v.to_t3tangent()
        self.assertIsInstance(rt, t3m.T3Tangent)
        for shift in (False, True):
            self.assertLess(float(np.linalg.norm(np.asarray(rt.to_dense(include_shift=shift))
                                                 - np.asarray(v.to_dense(include_shift=shift)))), 1e-10)

    def test_roundtrip_preserves_tangent(self):
        for ss, K in [((), ()), ((2,), ()), ((), (3,))]:
            B, V = _uniform_base(t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=ss))
            v = ut3m.UT3Tangent(B, _K_variations(B, K) if K else V)
            # for a stacked v, to_t3tangent is a tree -> round-trip per leaf
            for leaf in _full_unstack(v):
                back = ut3m.UT3Tangent.from_t3tangent(leaf.to_t3tangent())
                with self.subTest(ss=ss, K=K):
                    self.assertLess(float(np.linalg.norm(np.asarray(back.to_dense()) - np.asarray(leaf.to_dense()))), 1e-10)

    def test_to_t3tangent_stacked_tree(self):
        B, V = _uniform_base(t3.TuckerTensorTrain.randn(*_STRUCT))
        v = ut3m.UT3Tangent(B, _K_variations(B, (3,)))
        tree = v.to_t3tangent()
        self.assertEqual(len(tree), 3)
        vd = np.asarray(v.to_dense())
        for i in range(3):
            self.assertLess(float(np.linalg.norm(np.asarray(tree[i].to_dense()) - vd[i])), 1e-10)

    def test_from_t3tangent_K_stacked(self):
        x = t3.TuckerTensorTrain.randn(*_STRUCT)
        rb, _ = bvf.t3_orthogonal_representations(x)
        rt = t3m.T3Tangent(rb, bvf.T3Variations.randn(rb.variation_shapes, stack_shape=(3,) + rb.stack_shape))
        u = ut3m.UT3Tangent.from_t3tangent(rt)
        self.assertEqual(u.stack_shape, (3,))
        ud = np.asarray(u.to_dense())
        self.assertTrue(np.allclose(ud, np.asarray(rt.to_dense()), atol=1e-10))

    def test_varying_C_to_t3tangent_tree(self):
        us, rs = _hetero_tangents()
        v = ut3m.UT3Tangent.stack_basis(us)
        tree = v.to_t3tangent()                                             # different ranks per element
        self.assertEqual(len(tree), 2)
        for i in range(2):
            self.assertLess(float(np.linalg.norm(np.asarray(tree[i].to_dense()) - np.asarray(rs[i].to_dense()))), 1e-10)


if __name__ == '__main__':
    unittest.main()
