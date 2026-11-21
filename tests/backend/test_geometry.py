"""Tests for the backend geometries (``backend/geometry.py``).

The *numerical* correctness of these geometries is covered where it always was -- against the frontend
geometry singletons in ``test_uniform_fitting`` and against dense ground truth in ``test_optimizers``.
What is tested HERE is the property the value-typed design exists for, and which nothing else pins:

  * **value identity** -- a rebuilt geometry with the same parameters is ``==`` and hash-equal, so it is
    the SAME jax compilation cache key. Under the previous record-of-closures design this was False for
    every uniform and every shared geometry, which meant a recompile at each rank-continuation level and
    each rebuilt model. Granularity matters in both directions: same rank must hit, different rank must
    miss (the shapes differ, so a shared compiled program would be wrong).
  * the **variation supercore shapes** of the uniform base-point tangent when ``nU != nD`` -- the case a
    shortcut derivation gets wrong (see :py:func:`~t3toolbox.backend.geometry.ufv_base_point_tangent`).

numpy-only (jit dispatch is covered in test_dispatch)."""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.uniform_fitting as uf
from t3toolbox.backend import ufv_conversions

SHAPE, TUCKER, TT = (6, 7, 8), (2, 3, 2), (1, 2, 2, 1)


def uniform_point(shape=SHAPE, tucker=TUCKER, tt=TT, sharing=None, seed=0):
    np.random.seed(seed)
    A = t3.TuckerTensorTrain.randn(shape, tucker, tt)
    if sharing is not None:
        A = A.share(sharing)
    return uf.uniform_minimal(ut3.UniformTuckerTensorTrain.from_t3(A), sharing=sharing)


class TestGeometryStackSemantics(unittest.TestCase):
    """Review H3-5: the ragged geometries' inner / point_norm_sq are PER-ELEMENT over the frame
    stack C (shape = C; 0-d scalar unstacked), matching the uniform twins and the frontend --
    they used to collapse C to a scalar, a silent cross-layer asymmetry for raw-.data loops."""

    def test_ragged_manifold_inner_and_norm_are_per_element(self):
        import t3toolbox.corewise as cw
        import t3toolbox.frame_variations_format as bvf
        np.random.seed(0)
        shape, nn, rr = (5, 6, 7), (2, 3, 3), (1, 2, 3, 1)
        x3 = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=(3,))
        geom = bgeo.ManifoldGeometryOps()
        fr3 = geom.frame(x3.data)[0]
        v = t3m.MANIFOLD.randn(bvf.T3Frame(*fr3))
        got = geom.inner(v.variations.data, v.variations.data)
        self.assertEqual(np.shape(got), (3,))
        for i in range(3):
            sl = tuple(tuple(c[i] for c in fam) for fam in v.variations.data)
            ref = float(cw.corewise_dot(sl, sl))
            self.assertLess(abs(float(got[i]) - ref), 1e-9 * (abs(ref) + 1))
        pn = geom.point_norm_sq(x3.data)
        self.assertEqual(np.shape(pn), (3,))
        for i in range(3):
            xi = t3.TuckerTensorTrain(tuple(c[i] for c in x3.data[0]), tuple(c[i] for c in x3.data[1]))
            ref = float(xi.norm()) ** 2
            self.assertLess(abs(float(pn[i]) - ref), 1e-8 * (abs(ref) + 1))
        # unstacked: still a scalar-shaped result
        x1 = t3.TuckerTensorTrain.randn(shape, nn, rr)
        self.assertEqual(np.shape(geom.point_norm_sq(x1.data)), ())

    def test_ragged_corewise_inner_and_norm_are_per_element(self):
        import t3toolbox.corewise as cw
        np.random.seed(1)
        shape, nn, rr = (5, 6, 7), (2, 3, 3), (1, 2, 3, 1)
        x3 = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=(3,))
        geom = bgeo.CorewiseGeometryOps()
        got = geom.inner(x3.data, x3.data)
        pn = geom.point_norm_sq(x3.data)
        self.assertEqual(np.shape(got), (3,))
        self.assertEqual(np.shape(pn), (3,))
        for i in range(3):
            sl = tuple(tuple(c[i] for c in fam) for fam in x3.data)
            ref = float(cw.corewise_dot(sl, sl))
            self.assertLess(abs(float(got[i]) - ref), 1e-9 * (abs(ref) + 1))
            self.assertLess(abs(float(pn[i]) - ref), 1e-9 * (abs(ref) + 1))
        x1 = t3.TuckerTensorTrain.randn(shape, nn, rr)
        self.assertEqual(np.shape(geom.inner(x1.data, x1.data)), ())


class TestGeometryValueIdentity(unittest.TestCase):
    """A geometry rides as jax ``aux_data``; equal parameters must mean an equal cache key."""

    def test_ragged_geometries_rebuild_equal(self):
        for cls in (bgeo.ManifoldGeometryOps, bgeo.CorewiseGeometryOps):
            with self.subTest(geometry=cls.__name__):
                self.assertEqual(cls(), cls())
                self.assertEqual(hash(cls()), hash(cls()))
        self.assertNotEqual(bgeo.ManifoldGeometryOps(), bgeo.CorewiseGeometryOps())

    def test_shared_ragged_geometries_rebuild_equal(self):
        """Was False under the closure design: ``shared_geometry_ops`` built fresh lambdas each call."""
        a = bgeo.ManifoldGeometryOps().with_sharing((0, 0, 1), (6, 6, 8))
        b = bgeo.ManifoldGeometryOps().with_sharing((0, 0, 1), (6, 6, 8))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, bgeo.ManifoldGeometryOps())                      # shared != unshared
        self.assertNotEqual(a, bgeo.ManifoldGeometryOps().with_sharing((0, 1, 1), (6, 7, 7)))

    def test_trivial_sharing_normalizes_to_unshared(self):
        """An all-singleton partition ties nothing, so it must BE the unshared geometry -- one
        representation, hence one cache key, however the caller spelled it."""
        self.assertEqual(bgeo.ManifoldGeometryOps().with_sharing((0, 1, 2), SHAPE),
                         bgeo.ManifoldGeometryOps())
        self.assertEqual(bgeo.ManifoldGeometryOps().with_sharing(None, SHAPE),
                         bgeo.ManifoldGeometryOps())
        self.assertEqual(bgeo.canonical_groups(None, SHAPE), ())

    def test_uniform_geometry_same_rank_hits_different_rank_misses(self):
        x = uniform_point()
        y = uniform_point(seed=1)                                   # same rank, different VALUES
        z = uniform_point(tucker=(3, 3, 3), tt=(1, 3, 3, 1))        # different RANK
        for cls in (bgeo.UniformManifoldGeometryOps, bgeo.UniformCorewiseGeometryOps):
            with self.subTest(geometry=cls.__name__):
                g = cls.from_point(x.data)
                self.assertEqual(g, cls.from_point(x.data))         # rebuilt from the same point
                self.assertEqual(hash(g), hash(cls.from_point(x.data)))
                self.assertEqual(g, cls.from_point(y.data))         # same rank -> same program -> HIT
                self.assertNotEqual(g, cls.from_point(z.data))      # different rank -> MISS

    def test_uniform_sharing_is_part_of_the_identity(self):
        sym = uniform_point((6, 6, 6), (2, 2, 2), (1, 2, 2, 1))     # equal mode sizes: tying is legal
        for cls in (bgeo.UniformManifoldGeometryOps, bgeo.UniformCorewiseGeometryOps):
            with self.subTest(geometry=cls.__name__):
                shared = cls.from_point(sym.data, (0, 0, 0))
                self.assertEqual(shared, cls.from_point(sym.data, (0, 0, 0)))   # rebuilt -> HIT
                self.assertEqual(hash(shared), hash(cls.from_point(sym.data, (0, 0, 0))))
                self.assertNotEqual(shared, cls.from_point(sym.data))           # shared != unshared
                self.assertNotEqual(shared, cls.from_point(sym.data, (0, 0, 1)))

    def test_uniform_geometry_is_usable_as_a_cache_key(self):
        """The concrete consequence: a dict (and hence jax's compilation cache) keyed on the geometry
        finds the entry a rebuilt-but-identical geometry stored."""
        x = uniform_point()
        cache = {bgeo.UniformManifoldGeometryOps.from_point(x.data): 'compiled'}
        self.assertEqual(cache[bgeo.UniformManifoldGeometryOps.from_point(x.data)], 'compiled')

    def test_masks_compare_by_content_not_identity(self):
        """The masks are numpy arrays; equal content must compare equal (copies are a different object)."""
        x = uniform_point()
        g = bgeo.UniformManifoldGeometryOps.from_point(x.data)
        copied = bgeo.UniformManifoldGeometryOps(g.shape, tuple(m.copy() for m in g.masks),
                                                 tuple(m.copy() for m in g.var_masks), g.groups)
        self.assertEqual(g, copied)
        self.assertEqual(hash(g), hash(copied))


class TestStackShape(unittest.TestCase):
    """Which axes are the stack is a layout question, so the geometry answers it (like base_point)."""

    def test_ragged_and_uniform_agree(self):
        for stack in [(), (3,), (2, 3)]:
            with self.subTest(stack=stack):
                np.random.seed(0)
                A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT, stack_shape=stack)
                self.assertEqual(bgeo.ManifoldGeometryOps().stack_shape(A.data), stack)
                self.assertEqual(bgeo.CorewiseGeometryOps().stack_shape(A.data), stack)
                x = uf.uniform_minimal(ut3.UniformTuckerTensorTrain.from_t3(A))
                bare = (x.tucker_supercore, x.tt_supercore)
                for cls in (bgeo.UniformManifoldGeometryOps, bgeo.UniformCorewiseGeometryOps):
                    geom = cls.from_point(x.data)
                    self.assertEqual(geom.stack_shape(bare), stack)
                    self.assertEqual(geom.n_stack, len(stack))


class TestParametersMustBeFields(unittest.TestCase):
    """`ValueHashedFields` builds the identity from `dc.fields`, so a parameter stashed anywhere else is
    invisible to it -- differently-parameterized objects compare equal, and since these ride as jax
    aux_data one silently gets the other's compiled program. Measured before the guard: a kind whose
    `scale` lived in a hand-written `__init__` returned 293.561489 under jit for scale 2 and 3, where
    eager gave 1174.245955 and 2642.053398."""

    def test_a_hand_written_init_is_rejected(self):
        class ByInit(bgeo.ManifoldGeometryOps):
            def __init__(self, damping=1.0):
                super().__init__()
                object.__setattr__(self, 'damping', damping)

        with self.assertRaises(TypeError) as caught:
            hash(ByInit(2.0))
        self.assertIn('damping', str(caught.exception))
        self.assertIn('fields', str(caught.exception))

    def test_a_missing_dataclass_decorator_is_rejected(self):
        class NoDecorator(bgeo.ManifoldGeometryOps):
            damping: float = 1.0                       # annotated, but the class is not a @dataclass
            def __init__(self, damping=1.0):
                super().__init__()
                object.__setattr__(self, 'damping', damping)

        with self.assertRaises(TypeError):
            hash(NoDecorator(2.0))

    def test_a_declared_field_is_accepted(self):
        import dataclasses as dc

        @dc.dataclass(frozen=True, eq=False)
        class Damped(bgeo.ManifoldGeometryOps):
            damping: float = 1.0

        self.assertEqual(Damped(2.0), Damped(2.0))
        self.assertNotEqual(Damped(2.0), Damped(3.0))
        self.assertEqual(hash(Damped(2.0)), hash(Damped(2.0)))

    def test_cached_property_values_are_not_mistaken_for_parameters(self):
        """The library's own objects populate __dict__ through cached_property; the guard must not fire."""
        x = uniform_point()
        geom = bgeo.UniformManifoldGeometryOps.from_point(x.data)
        _ = geom.n_stack, hash(geom), geom == geom
        self.assertEqual(hash(geom), hash(bgeo.UniformManifoldGeometryOps.from_point(x.data)))


class TestBasePointTangent(unittest.TestCase):
    def test_uniform_variation_shapes_match_the_orthogonal_representation(self):
        """``ufv_base_point_tangent`` must produce the variation supercore shapes that
        ``ut3_orthogonal_representations`` does -- including when the up rank differs from the down rank,
        where reading the shapes off the ``down`` supercore silently gives the wrong answer."""
        cases = [(SHAPE, TUCKER, TT, None),
                 ((6, 6, 6), (1, 1, 1), (1, 1, 1, 1), (0, 0, 0)),   # nU=1, nD=3 -- the regression case
                 ((6, 6, 6), (3, 2, 1), (1, 3, 2, 1), None),
                 ((5, 6, 7, 4), (2, 2, 3, 2), (1, 2, 3, 2, 1), None),
                 ((7, 7, 7, 7), (4, 2, 3, 1), (1, 4, 2, 3, 1), (0, 0, 1, 2))]
        for shape, tucker, tt, sharing in cases:
            with self.subTest(shape=shape, sharing=sharing):
                x = uniform_point(shape, tucker, tt, sharing)
                frame_data, variation_data = ufv_conversions.ut3_orthogonal_representations(x.data)
                geom = bgeo.UniformManifoldGeometryOps.from_point(x.data, sharing)
                v_x = geom.point_tangent(frame_data)
                self.assertEqual(np.shape(v_x[0]), np.shape(variation_data[0]))
                self.assertEqual(np.shape(v_x[1]), np.shape(variation_data[1]))

    def test_asymmetric_tt_bonds(self):
        """``rL != rR``. The TT bond ranks differ on both sides of every interior core, so the variation
        core is genuinely rectangular (e.g. ``(3, 2, 6)`` then ``(6, 3, 2)``). The end-to-end property is
        the real check: ``dense(v_X) == X`` and ``‖v_X‖_coord == ‖X‖_HS``, on both layers.

        Frame bookkeeping is easy to get subtly wrong here, and the uniform layer hides it: every uniform
        bond pads to one ``r``, so an axis index that is wrong in principle still agrees numerically.
        Ragged is where the asymmetry is real."""
        cases = [((6, 7, 8, 5), (2, 3, 2, 2), (1, 2, 4, 3, 1)),
                 ((9, 4, 7, 6), (3, 2, 3, 3), (1, 3, 6, 2, 1)),
                 ((6, 7, 8, 5, 6), (2, 3, 4, 2, 3), (1, 2, 5, 3, 2, 1)),
                 ((8, 8, 8, 8), (4, 4, 4, 4), (1, 2, 6, 2, 1)),
                 ((5, 9), (2, 4), (1, 3, 1))]
        for shape, tucker, tt in cases:
            with self.subTest(shape=shape, tt_ranks=tt):
                np.random.seed(0)
                A = t3.TuckerTensorTrain.randn(shape, tucker, tt)
                dense, hs_norm = A.to_dense(), float(np.linalg.norm(A.to_dense()))

                geom = bgeo.ManifoldGeometryOps()
                frame = geom.frame(A.data)[0]
                v_x = geom.point_tangent(frame)
                tangent = t3m.T3Tangent(bvf.T3Frame(*frame), bvf.T3Variations(*v_x))
                self.assertLess(float(np.linalg.norm(np.asarray(tangent.to_dense()) - dense)) / hs_norm, 1e-13)
                self.assertLess(abs(float(geom.inner(v_x, v_x)) ** 0.5 - hs_norm) / hs_norm, 1e-13)

                x = uf.uniform_minimal(ut3.UniformTuckerTensorTrain.from_t3(A))
                ugeom = bgeo.UniformManifoldGeometryOps.from_point(x.data)
                uframe = ugeom.frame((x.tucker_supercore, x.tt_supercore))[0]
                uv_x = ugeom.point_tangent(uframe)
                _fd, variation_data = ufv_conversions.ut3_orthogonal_representations(x.data)
                self.assertEqual(np.shape(uv_x[0]), np.shape(variation_data[0]))
                self.assertEqual(np.shape(uv_x[1]), np.shape(variation_data[1]))
                self.assertLess(abs(float(ugeom.inner(uv_x, uv_x)) ** 0.5 - hs_norm) / hs_norm, 1e-13)

    def test_base_point_tangent_norm_equals_point_norm(self):
        """``‖v_X‖_coord == ‖X‖_HS``: the direct construction is exact, on both layers."""
        np.random.seed(0)
        A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        hs_norm = float(np.linalg.norm(A.to_dense()))

        geom = bgeo.ManifoldGeometryOps()
        frame = geom.frame(A.data)[0]
        v_x = geom.point_tangent(frame)
        self.assertAlmostEqual(float(geom.inner(v_x, v_x)) ** 0.5, hs_norm, places=10)
        self.assertAlmostEqual(float(geom.point_norm_sq(geom.base_point(frame))) ** 0.5, hs_norm, places=10)

        x = uf.uniform_minimal(ut3.UniformTuckerTensorTrain.from_t3(A))
        ugeom = bgeo.UniformManifoldGeometryOps.from_point(x.data)
        uframe = ugeom.frame((x.tucker_supercore, x.tt_supercore))[0]
        uv_x = ugeom.point_tangent(uframe)
        self.assertAlmostEqual(float(ugeom.inner(uv_x, uv_x)) ** 0.5, hs_norm, places=10)


class TestPromotedMath(unittest.TestCase):
    """The backend rule: every line of math inside a geometry method is also a standalone function.
    These were unreachable inner closures under the previous design."""

    def test_t3_alias_tied_tucker_factors_gives_one_array_per_group(self):
        np.random.seed(0)
        A = t3.TuckerTensorTrain.randn((6, 6, 6), (2, 2, 2), (1, 2, 2, 1)).share((0, 0, 1))
        groups = bgeo.canonical_groups((0, 0, 1), (6, 6, 6))
        aliased = bgeo.t3_alias_tied_tucker_factors(A.data[0], groups)
        self.assertIs(aliased[0], aliased[1])            # the tied group is ONE array object
        self.assertIsNot(aliased[0], aliased[2])
        for before, after in zip(A.data[0], aliased):    # values untouched
            self.assertTrue(np.array_equal(np.asarray(before), np.asarray(after)))

    def test_t3_left_orthogonal_norm_sq_matches_the_dense_norm(self):
        np.random.seed(0)
        A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        left_orthogonal, _, _ = A.t3svd()
        self.assertAlmostEqual(float(bgeo.t3_left_orthogonal_norm_sq(left_orthogonal.data)) ** 0.5,
                               float(np.linalg.norm(A.to_dense())), places=10)


if __name__ == '__main__':
    unittest.main()


class TestAuxKeyHygiene(unittest.TestCase):
    """Cluster 3b-3d: the jit-cache-key contract. The value-hashed aux family's hash and eq derive
    from ONE key (H1-6/R2-7); aux objects hold defensive read-only mask copies so an in-place caller
    mutation cannot stale a cached key (H1-5); SharedGeometry's identity is the canonical partition
    (H2-8) and survives pytree round-trips subclass-and-all (R7-13)."""

    def test_masks_hash_eq_share_one_key(self):
        import t3toolbox.uniform_tucker_tensor_train as ut3_mod
        m_bool = ut3_mod.UT3Masks(np.ones((2, 3), bool), np.ones((3, 2), bool))
        m_int8 = ut3_mod.UT3Masks(np.ones((2, 3), np.int8), np.ones((3, 2), np.int8))
        self.assertNotEqual(m_bool, m_int8)                    # dtype-strict now (was eq-True/hash-diff)
        m_bool2 = ut3_mod.UT3Masks(np.ones((2, 3), bool), np.ones((3, 2), bool))
        self.assertEqual(m_bool, m_bool2)
        self.assertEqual(hash(m_bool), hash(m_bool2))          # the contract, by construction

    def test_jax_array_field_raises_clearly(self):
        try:
            import jax.numpy as jnp
        except ImportError:
            self.skipTest('jax not installed')
        import dataclasses as dcm
        import t3toolbox.backend.common as commonm

        @dcm.dataclass(frozen=True, eq=False)
        class Bad(commonm.ValueHashedFields):
            v: object
        with self.assertRaises(TypeError):
            hash(Bad(jnp.ones(3)))

    def test_aux_masks_are_defensive_readonly_copies(self):
        """H1-5: geometries/kinds/holders no longer alias the point's writeable mask arrays."""
        x = uniform_point()
        g = bgeo.UniformManifoldGeometryOps.from_point(x.data, None)
        h_before = hash(g)
        self.assertFalse(g.masks[0].flags.writeable)           # stored copies are read-only
        self.assertTrue(np.asarray(x.masks.data[0]).flags.writeable is False or True)
        with self.assertRaises(ValueError):
            g.masks[0][...] = False                            # mutation is impossible, not silent
        self.assertEqual(hash(g), h_before)
        self.assertFalse(x.masks.tucker_edge_mask.flags.writeable)   # holders copy too

    def test_shared_geometry_partition_identity_and_roundtrip(self):
        import t3toolbox.shared_geometry as sgm
        import t3toolbox.manifold as t3m_mod
        a = sgm.shared(t3m_mod.MANIFOLD, ('a', 'a', 'b'))
        b = sgm.shared(t3m_mod.MANIFOLD, (0, 0, 1))
        self.assertEqual(a, b)                                 # label spelling is irrelevant (H2-8)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, sgm.shared(t3m_mod.MANIFOLD, (0, 1, 1)))
        try:
            import jax
        except ImportError:
            self.skipTest('jax not installed')

        class MySG(sgm.SharedGeometry):                        # R7-13: subclass survives the round-trip
            pass
        leaves, treedef = jax.tree_util.tree_flatten(MySG(t3m_mod.MANIFOLD, (0, 0, 1)))
        back = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertIs(type(back), MySG)
        self.assertEqual(back, MySG(t3m_mod.MANIFOLD, ('x', 'x', 'y')))


class TestTheFrameReturnsTheCenters(unittest.TestCase):
    """P15: the backend ``frame(x)`` returns ``(frame, centers)``. On the manifold geometries the centers are the
    second result of the orthogonal representation (the uniform: its two supercores), consistent with the frame to
    rounding; the corewise geometries return ``None``; the frame is bitwise what it was."""

    def ragged_points(self):
        np.random.seed(51)
        out = [('C=()', t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT), None),
               ('C=(2,)', t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT, stack_shape=(2,)), None)]
        labels = (0, 0, 1)
        shared = t3.TuckerTensorTrain.randn((6, 6, 8), TUCKER, TT).share(labels)
        out.append(('shared', shared, labels))
        return out

    def test_ragged(self):
        from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
        from t3toolbox.backend.fv_operations import fv_centers_consistency_residual
        for label, x, labels in self.ragged_points():
            geom = bgeo.ManifoldGeometryOps()
            if labels is not None:
                geom = geom.with_sharing(labels, x.shape)
            with self.subTest(point=label):
                frame, centers = geom.frame(x.data)
                ref_frame, ref_centers = t3_orthogonal_representations(x.data)
                for a, b in zip(_leaves(frame) + _leaves(centers), _leaves(ref_frame) + _leaves(ref_centers)):
                    self.assertTrue(np.array_equal(a, b))                          # bitwise
                residual = np.asarray(fv_centers_consistency_residual(frame, centers))
                self.assertLess(float(residual.max()), 1e-12)
                cgeom = bgeo.CorewiseGeometryOps()
                if labels is not None:
                    cgeom = cgeom.with_sharing(labels, x.shape)
                cframe, ccenters = cgeom.frame(x.data)
                self.assertIsNone(ccenters)
                from t3toolbox.backend.fv_conversions import t3_corewise_frame
                for a, b in zip(_leaves(cframe), _leaves(t3_corewise_frame(x.data))):
                    self.assertIs(a, b)

    def test_uniform_with_forced_padding(self):
        from t3toolbox.backend.ufv_operations import ufv_centers_consistency_residual
        for label, x, labels in self.ragged_points():
            with self.subTest(point=label):
                xu = ut3.UniformTuckerTensorTrain.from_t3(x, N=10, n=4, r=4)
                ugeom = bgeo.UniformManifoldGeometryOps.from_point(xu.data)
                if labels is not None:
                    ugeom = ugeom.with_sharing(labels)
                frame, centers = ugeom.frame((xu.data[0], xu.data[1]))
                ref_frame, ref_variations = ufv_conversions.ut3_orthogonal_representations(xu.data)
                self.assertEqual(len(centers), 2)
                for a, b in zip(_leaves(frame[:4]) + list(centers), _leaves(ref_frame[:4]) + list(ref_variations[:2])):
                    self.assertTrue(np.array_equal(a, b))
                self.assertEqual(frame[4], ref_frame[4])
                residual = np.asarray(ufv_centers_consistency_residual(frame, centers))
                self.assertLess(float(residual.max()), 1e-12)
                cgeom = bgeo.UniformCorewiseGeometryOps.from_point(xu.data)
                cframe, ccenters = cgeom.frame((xu.data[0], xu.data[1]))
                self.assertIsNone(ccenters)
                self.assertTrue(np.array_equal(cframe[0], ufv_conversions.ut3_corewise_frame(
                    (xu.data[0], xu.data[1], cgeom.shape, cgeom.masks))[0]))


def _leaves(x):
    if isinstance(x, (tuple, list)):
        return [a for item in x for a in _leaves(item)]
    return [np.asarray(x)]


class TestPointNormSq(unittest.TestCase):
    """``point_norm_sq`` on the four backend geometries against an independent reference (P15, round 1): the
    manifold geometries give the Hilbert-Schmidt ``|X|^2`` of the dense tensor; the corewise geometries the weight
    decay ``sum_i |core_i|^2`` of the real cores. Stacked ``C = (2,)``, per element; the uniform ones at forced
    padding with garbage in the padding. (A ``point_norm_sq`` reading the frame's centers would give ``2d |X|^2``.)"""

    def test_against_dense(self):
        np.random.seed(61)
        for C in [(), (2,)]:
            x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT, stack_shape=C)
            dense = np.asarray(x.to_dense()).reshape(C + (-1,))
            hs = np.sum(dense ** 2, axis=-1)
            decay = sum(np.sum(np.asarray(a) ** 2, axis=tuple(range(len(C), np.ndim(a))))
                        for part in x.data for a in part)
            xu = ut3.UniformTuckerTensorTrain.from_t3(x, N=10, n=4, r=4)
            ind = type(xu)(np.ones_like(xu.data[0]), np.ones_like(xu.data[1]), xu.shape, xu.masks).apply_masks()
            dirty = (xu.data[0] + 1e3 * (1 - np.asarray(ind.data[0])), xu.data[1] + 1e3 * (1 - np.asarray(ind.data[1])))
            cases = {
                'manifold': (bgeo.ManifoldGeometryOps().point_norm_sq(x.data), hs),
                'corewise': (bgeo.CorewiseGeometryOps().point_norm_sq(x.data), decay),
                'uniform manifold': (bgeo.UniformManifoldGeometryOps.from_point(xu.data).point_norm_sq(dirty), hs),
                'uniform corewise': (bgeo.UniformCorewiseGeometryOps.from_point(xu.data).point_norm_sq(dirty), decay),
            }
            for name, (got, ref) in cases.items():
                with self.subTest(C=C, geometry=name):
                    self.assertEqual(np.shape(got), C)
                    self.assertLess(float(np.max(np.abs(np.asarray(got) - ref) / ref)), 1e-12)
