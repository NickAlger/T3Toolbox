"""Tests for the backend geometries (``backend/geometry.py``).

The *numerical* correctness of these geometries is covered where it always was -- against the frontend
geometry singletons in ``test_uniform_fitting`` and against dense ground truth in ``test_optimizers``.
What is tested HERE is the property the value-typed design exists for, and which nothing else pins:

  * **value identity** -- a rebuilt geometry with the same parameters is ``==`` and hash-equal, so it is
    the SAME jax compilation cache key. Under the previous record-of-closures design this was False for
    every uniform and every shared geometry, which meant a recompile at each rank-continuation level and
    each rebuilt model. Granularity matters in both directions: same rank must hit, different rank must
    miss (the shapes differ, so a shared compiled program would be wrong).
  * the **variation supercore shapes** of the uniform base-point tangent when ``nD != nU`` -- the case a
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
        fr3 = geom.frame(x3.data)
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
        ``ut3_orthogonal_representations`` does -- including when the down rank differs from the up rank,
        where reading the shapes off the ``up`` supercore silently gives the wrong answer."""
        cases = [(SHAPE, TUCKER, TT, None),
                 ((6, 6, 6), (1, 1, 1), (1, 1, 1, 1), (0, 0, 0)),   # nD=1, nU=3 -- the regression case
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
                frame = geom.frame(A.data)
                v_x = geom.point_tangent(frame)
                tangent = t3m.T3Tangent(bvf.T3Frame(*frame), bvf.T3Variations(*v_x))
                self.assertLess(float(np.linalg.norm(np.asarray(tangent.to_dense()) - dense)) / hs_norm, 1e-13)
                self.assertLess(abs(float(geom.inner(v_x, v_x)) ** 0.5 - hs_norm) / hs_norm, 1e-13)

                x = uf.uniform_minimal(ut3.UniformTuckerTensorTrain.from_t3(A))
                ugeom = bgeo.UniformManifoldGeometryOps.from_point(x.data)
                uframe = ugeom.frame((x.tucker_supercore, x.tt_supercore))
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
        frame = geom.frame(A.data)
        v_x = geom.point_tangent(frame)
        self.assertAlmostEqual(float(geom.inner(v_x, v_x)) ** 0.5, hs_norm, places=10)
        self.assertAlmostEqual(float(geom.point_norm_sq(geom.base_point(frame))) ** 0.5, hs_norm, places=10)

        x = uf.uniform_minimal(ut3.UniformTuckerTensorTrain.from_t3(A))
        ugeom = bgeo.UniformManifoldGeometryOps.from_point(x.data)
        uframe = ugeom.frame((x.tucker_supercore, x.tt_supercore))
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
