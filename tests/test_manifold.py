# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
import t3toolbox.safety as safety
import t3toolbox.backend.probing as t3p
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm


def _random_variations(frame):
    """A random T3Variations fitting the holes of ``frame`` (ungauged)."""
    rnd = lambda *s: np.random.randn(*s)
    ss = frame.stack_shape
    tucker_hole_shapes, tt_hole_shapes = frame.variation_shapes
    V = tuple(rnd(*(ss + s)) for s in tucker_hole_shapes)
    H = tuple(rnd(*(ss + s)) for s in tt_hole_shapes)
    return bvf.T3Variations(V, H)


def _random_tangent(t3_structure, stack_shape=()):
    x = t3.TuckerTensorTrain.randn(*t3_structure, stack_shape=stack_shape)
    frame, _ = bvf.t3_orthogonal_representations(x)
    return t3m.T3Tangent(frame, _random_variations(frame))


def _slice_frame(frame, idx):
    """The unstacked T3Frame at stack index ``idx`` (idx=() returns the whole frame)."""
    s = lambda C: np.asarray(C)[idx]
    up, down, left, right = frame.data
    return bvf.T3Frame(tuple(map(s, up)), tuple(map(s, down)), tuple(map(s, left)), tuple(map(s, right)))


def _slice_t3(x, idx):
    """The unstacked TuckerTensorTrain at stack index ``idx``."""
    s = lambda C: np.asarray(C)[idx]
    tucker, tt = x.data
    return t3.TuckerTensorTrain(tuple(map(s, tucker)), tuple(map(s, tt)))


def _slice_tangent(frame, var, idx, n_frame):
    """The unstacked (frame, variation) tangent at full K+C index ``idx``.

    The base point is shared across the tangent stack K: the frame is sliced at the trailing C part
    of ``idx`` while the variation is sliced at the full ``idx``.
    """
    g_idx = idx[len(idx) - n_frame:] if n_frame > 0 else ()
    sV = lambda C: np.asarray(C)[idx]
    vslice = bvf.T3Variations(tuple(map(sV, var.tucker_variations)), tuple(map(sV, var.tt_variations)))
    return t3m.T3Tangent(_slice_frame(frame, g_idx), vslice)


def _tree_get(tree, idx):
    """Navigate an array-like tree by a multi-index (idx=() returns the depth-0 tree itself)."""
    for k in idx:
        tree = tree[k]
    return tree


def _dense_tangent_projector(frame):
    """Dense orthogonal projector onto the (gauged) tangent space at an unstacked ``frame``.

    Built from the dense embeddings of many gauged random tangents: ``A @ pinv(A)`` projects onto
    their column span, which is exactly the tangent space.
    """
    dim = t3m.manifold_dim((frame.shape, frame.up_ranks, frame.left_ranks))
    cols = [np.asarray(t3m.MANIFOLD.randn(frame).to_dense()).reshape(-1)
            for _ in range(3 * dim)]
    A = np.stack(cols, axis=1)
    # A is rank-deficient (the tangent parametrization is redundant): singular values are O(1) then a
    # clean gap to ~1e-16. pinv's DEFAULT rcond (~1e-15) sits next to those nulls and flakily keeps/
    # inverts one, contaminating the projector -- use an explicit cutoff well inside the gap.
    return A @ np.linalg.pinv(A, rcond=1e-8)


class TestManifold(unittest.TestCase):
    t3_structures = [
        #  (shape,            tucker_ranks,   tt_ranks)
        ((10,),               (3,),           (1, 1)),
        ((10, 11),            (3, 4),         (1, 2, 1)),
        ((10, 11, 12),        (3, 4, 3),      (1, 2, 2, 1)),
        ((9, 10, 11, 12),     (2, 3, 3, 2),   (1, 2, 3, 2, 1)),
    ]
    stack_shapes = [(), (2,), (2, 3)]

    def check_relerr(self, xtrue, x):
        xtrue, x = np.asarray(xtrue), np.asarray(x)
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def test_metadata_repr_validate(self):
        # T3Tangent slice-1: size/data_size, minimal_ranks, tangent_space_dimension, copy, repr, validate.
        STRUCT = ((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        v = None
        for C in [(), (2,)]:
            for K in [(), (3,)]:
                x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                frame, _ = bvf.t3_orthogonal_representations(x)
                v = t3m.COREWISE.randn(frame, stack_shape=K)
                self.assertEqual(int(np.prod(STRUCT[0])), v.size)                 # dense element count
                self.assertEqual(v.frame.data_size + v.variations.data_size, v.data_size)
                self.assertEqual(frame.minimal_ranks, v.minimal_ranks)            # delegates to frame
                self.assertEqual(t3m.manifold_dim((frame.shape, frame.up_ranks, frame.left_ranks)),
                                 v.tangent_space_dimension)
                cp = v.copy(); cp.variations.tucker_variations[0][...] = 9.0      # copy is independent
                self.assertFalse(np.allclose(np.asarray(v.variations.tucker_variations[0]), 9.0))
                self.assertIn("T3Tangent", repr(v)); self.assertNotIn("array", repr(v))
                v.validate()   # valid tangent; also runs in __post_init__
        # __post_init__ validate rejects an incompatible (frame, variations) pair
        b1, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(*STRUCT))
        _, var2 = bvf.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn((5, 6, 4), (3, 3, 3), (1, 2, 2, 1)))
        with self.assertRaises(Exception):
            t3m.T3Tangent(b1, var2)
        try:
            import jax  # noqa: F401
            self.assertTrue(v.to_jax().contains_jax)
            self.assertFalse(v.to_jax().to_numpy().contains_jax)
        except ImportError:
            pass

    def test_constructors(self):
        # MANIFOLD.random_orthogonal / unit / zeros_like / randn_like.
        STRUCT = ((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        v = t3m.MANIFOLD.random_orthogonal(*STRUCT, stack_shape=(2,), tangent_stack_shape=(3,))
        self.assertEqual(((2,), (3,)), (v.frame_stack_shape, v.tangent_stack_shape))
        self.assertTrue(v.is_orthogonal().all() and v.is_gauged().all())                   # gauged by default
        frame = bvf.T3Frame.random_orthogonal(*STRUCT)
        u = t3m.T3Tangent.unit(frame, (True, 1, (0, 1, 0)))
        self.assertEqual(1, sum(int(np.count_nonzero(np.asarray(c)))
                                for c in u.variations.tucker_variations + u.variations.tt_variations))
        w = t3m.COREWISE.randn(frame, stack_shape=(3,))
        zl = t3m.T3Tangent.zeros_like(w)
        self.assertEqual((3,), zl.tangent_stack_shape)
        self.assertEqual(0.0, float(np.max(np.abs(zl.corewise_norm()))))
        self.assertEqual((3,), t3m.MANIFOLD.randn_like(w).tangent_stack_shape)

    def test_to_from_vector(self):
        # T3Tangent.to_vector (variation DOF only) / from_vector round-trip.
        frame = bvf.T3Frame.random_orthogonal((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        v = t3m.COREWISE.randn(frame, stack_shape=(3,))
        flat = v.to_vector()
        self.assertEqual((v.variations.data_size,), flat.shape)   # variation DOF; frame excluded
        v2 = t3m.T3Tangent.from_vector(flat, frame, tangent_stack_shape=(3,))
        self.assertEqual(0.0, cw.corewise_relerr(v.variations.data, v2.variations.data))

    def test_save_load(self):
        import tempfile, os
        frame = bvf.T3Frame.random_orthogonal((5, 6, 4), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
        v = t3m.COREWISE.randn(frame, stack_shape=(3,))
        f = os.path.join(tempfile.mkdtemp(), 't.npz'); v.save(f)
        v2 = t3m.T3Tangent.load(f)
        self.assertEqual(0.0, cw.corewise_relerr(v.variations.data, v2.variations.data))
        self.assertEqual(0.0, cw.corewise_relerr(v.frame.data, v2.frame.data))

    def test_reverse(self):
        # T3Tangent.reverse commutes with to_dense (mode axes reversed); reverse is an involution.
        STRUCT = ((5, 6, 4), (2, 3, 2), (1, 2, 2, 1)); d = 3
        for C in [(), (2,)]:
            for K in [(), (3,)]:
                frame = bvf.T3Frame.random_orthogonal(*STRUCT, stack_shape=C)
                v = t3m.COREWISE.randn(frame, stack_shape=K)
                D = np.asarray(v.to_dense()); ns = D.ndim - d
                perm = tuple(range(ns)) + tuple(range(D.ndim - 1, ns - 1, -1))
                self.check_relerr(D.transpose(perm), np.asarray(v.reverse().to_dense()))
                self.check_relerr(D, np.asarray(v.reverse().reverse().to_dense()))

    def test_sum_tangents(self):
        # Summing over the tangent stack K commutes with to_dense (= the tensor sum, by linearity).
        frame = bvf.T3Frame.random_orthogonal((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        v = t3m.COREWISE.randn(frame, stack_shape=(3,))
        self.check_relerr(np.sum(np.asarray(v.to_dense()), axis=0), np.asarray(v.sum_tangents().to_dense()))
        self.assertEqual((), v.sum_tangents().tangent_stack_shape)

    def test_manifold_dim(self):
        self.assertEqual(578, t3m.manifold_dim(((15, 16, 13), (9, 10, 8), (2, 7, 6, 3))))
        self.assertEqual(29, t3m.manifold_dim(((5, 6, 3), (5, 3, 2), (2, 2, 4, 1))))

    def test_manifold_dim_via_svd(self):
        # The dimension of the tangent space = number of nonzero singular values of a sufficient
        # collection of dense tangent vectors. Uses a minimal-rank frame.
        shape, tucker_ranks, tt_ranks = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        s = (shape, tucker_ranks, tt_ranks)
        mdim = t3m.manifold_dim(s)
        x = t3.TuckerTensorTrain.randn(*s)
        self.assertTrue(x.has_minimal_ranks)
        frame, _ = bvf.t3_orthogonal_representations(x)
        dense_tangents = np.stack([
            t3m.T3Tangent(frame, _random_variations(frame)).to_dense().reshape(-1)
            for _ in range(2 * mdim)
        ])
        ss = np.linalg.svd(dense_tangents, compute_uv=False)
        rank = int(np.sum(ss > 1e-9 * ss[0]))
        self.assertEqual(mdim, rank)

    def test_to_dense_matches_explicit_sum(self):
        # Independent check of to_dense for d=3: the explicit 6-term (3 Tucker + 3 TT) sum.
        for STACK_SHAPE in [(), (2,)]:
            with self.subTest(STACK_SHAPE=STACK_SHAPE):
                x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                frame, var = bvf.t3_orthogonal_representations(x)
                v = t3m.T3Tangent(frame, var)

                U0, U1, U2 = (np.asarray(c) for c in frame.up_tucker_cores)
                D0, D1, D2 = (np.asarray(c) for c in frame.down_tt_cores)
                L0, L1, L2 = (np.asarray(c) for c in frame.left_tt_cores)
                R0, R1, R2 = (np.asarray(c) for c in frame.right_tt_cores)
                V0, V1, V2 = (np.asarray(c) for c in var.tucker_variations)
                H0, H1, H2 = (np.asarray(c) for c in var.tt_variations)

                f = lambda B0, B1, B2, G0, G1, G2: np.einsum(
                    '...ai,...bj,...ck,...xay,...ybz,...zcw->...ijk', B0, B1, B2, G0, G1, G2)
                manual = (f(U0, U1, U2, H0, R1, R2) + f(U0, U1, U2, L0, H1, R2) + f(U0, U1, U2, L0, L1, H2)
                          + f(V0, U1, U2, D0, R1, R2) + f(U0, V1, U2, L0, D1, R2) + f(U0, U1, V2, L0, L1, D2))
                self.check_relerr(manual, v.to_dense())

    def test_linalg(self):
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in self.stack_shapes:
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                    frame, _ = bvf.t3_orthogonal_representations(x)
                    v1 = t3m.T3Tangent(frame, _random_variations(frame))
                    v2 = t3m.T3Tangent(frame, _random_variations(frame))

                    self.check_relerr(v1.to_dense() + v2.to_dense(), (v1 + v2).to_dense())
                    self.check_relerr(v1.to_dense() - v2.to_dense(), (v1 - v2).to_dense())
                    self.check_relerr(2.5 * np.asarray(v1.to_dense()), (2.5 * v1).to_dense())
                    self.check_relerr(-np.asarray(v1.to_dense()), (-v1).to_dense())
                    self.assertLessEqual(norm(np.asarray(t3m.T3Tangent.zeros(frame).to_dense())), tol)

    def test_same_tangent_space_guard(self):
        # the same-frame guard is NUMERICAL (frames_equal), not object identity: value-equal frames PASS
        # (the jit-round-trip property), a genuinely different frame raises, and safety.unsafe() skips it.
        x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        frame_a, _ = bvf.t3_orthogonal_representations(x)
        frame_b = bvf.T3Frame(*frame_a.data)               # value-equal cores, DIFFERENT object (jit round-trip)
        va = t3m.T3Tangent(frame_a, _random_variations(frame_a))
        vb = t3m.T3Tangent(frame_b, _random_variations(frame_b))
        _ = va + t3m.T3Tangent(frame_a, _random_variations(frame_a))   # same object: OK
        _ = va + vb                                       # value-equal frame: OK (numerical guard accepts)
        _ = va.corewise_inner(vb)

        frame_c, _ = bvf.t3_orthogonal_representations(    # a genuinely different frame -> raises
            t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1)))
        vc = t3m.T3Tangent(frame_c, _random_variations(frame_c))
        for op in (lambda: va + vc, lambda: va - vc, lambda: va.corewise_inner(vc)):
            with self.assertRaises(ValueError):
                op()
        with safety.unsafe():                             # unsafe mode skips the numerical check
            _ = va + vc

    def test_manifold_orth_preconditions(self):
        # S5: the manifold projections/retraction enforce an ORTHOGONAL frame in safe mode (raise),
        # skip under safety.unsafe(), and pass on an orthogonal frame. CorewiseGeometry never checks.
        x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        frame_orth, _ = bvf.t3_orthogonal_representations(x)
        bad_cores = tuple(tuple(c + 0.3 * np.random.randn(*c.shape) for c in grp) for grp in frame_orth.data)
        frame_bad = bvf.T3Frame(*bad_cores)
        self.assertTrue(frame_orth.is_orthogonal().all())
        self.assertFalse(frame_bad.is_orthogonal().all())

        v_bad = t3m.T3Tangent(frame_bad, _random_variations(frame_bad))
        raisers = [
            lambda: t3m.MANIFOLD.project(v_bad),
            lambda: t3m.MANIFOLD.project_oblique(v_bad),
            lambda: t3m.MANIFOLD.retract(v_bad),
            lambda: t3m.MANIFOLD.project_ambient(frame_bad, x),
            lambda: t3m.MANIFOLD.transport(v_bad, frame_bad),
            lambda: t3m.MANIFOLD.randn(frame_bad),
        ]
        for op in raisers:                                   # safe mode (default): non-orthogonal -> raise
            with self.assertRaises(ValueError):
                op()
        with safety.unsafe():                                # unsafe mode skips every ORTH check
            for op in raisers:
                op()

        v_ok = t3m.MANIFOLD.randn(frame_orth)                 # orthogonal frame: all pass in safe mode
        t3m.MANIFOLD.project(v_ok)
        t3m.MANIFOLD.retract(v_ok)
        t3m.MANIFOLD.project_ambient(frame_orth, x)
        # CorewiseGeometry is gauge-free by design: no ORTH check even on the non-orthonormal corewise frame
        cframe = t3m.COREWISE.frame(x)
        self.assertFalse(cframe.is_orthogonal().all())
        t3m.COREWISE.project(t3m.COREWISE.randn(cframe))
        t3m.COREWISE.retract(t3m.COREWISE.randn(cframe))

    def test_orthogonality_residual_cached(self):
        # the ORTH check routes through a cached residual: a fixed frame is contracted once (inner-loop perf)
        x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        frame, _ = bvf.t3_orthogonal_representations(x)
        self.assertNotIn('orthogonality_residual', frame.__dict__)
        self.assertTrue(frame.is_orthogonal().all())
        self.assertIn('orthogonality_residual', frame.__dict__)        # cached after first check
        self.assertLess(float(frame.orthogonality_residual), 1e-9)

    def test_inner_norm(self):
        x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        frame, _ = bvf.t3_orthogonal_representations(x)
        v1 = t3m.T3Tangent(frame, _random_variations(frame))
        v2 = t3m.T3Tangent(frame, _random_variations(frame))

        # structural identities (corewise) hold regardless of gauge (unstacked -> scalar)
        self.assertAlmostEqual(float(v1.corewise_inner(v2)), float(cw.corewise_dot(v1.variations.data, v2.variations.data)))
        self.assertAlmostEqual(float(v1.corewise_norm()), float(np.sqrt(v1.corewise_inner(v1))))

    def test_is_orthogonal_and_is_gauged(self):
        x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        frame, var = bvf.t3_orthogonal_representations(x)

        # frame from t3_orthogonal_representations is orthogonal; its variations are not gauged
        self.assertTrue(t3m.T3Tangent(frame, var).is_orthogonal().all())
        self.assertFalse(t3m.T3Tangent(frame, var).is_gauged().all())

        # the zero tangent is trivially gauged (all variation cores are zero)
        self.assertTrue(t3m.T3Tangent.zeros(frame).is_gauged().all())

    # ``stack_shapes`` for the two-axis stacking tests: (frame_stack C, tangent_stack K) pairs.
    fv_stack_shapes = [((), (3,)), ((2,), (3,)), ((2,), ()), ((2,), (2, 2)), ((2, 3), (2,))]

    def _random_v_stacked(self, struct, frame_stack, V):
        x = t3.TuckerTensorTrain.randn(*struct, stack_shape=frame_stack)
        frame, _ = bvf.t3_orthogonal_representations(x)
        return t3m.COREWISE.randn(frame, stack_shape=V)

    def test_unstack_stack_tangents(self):
        # unstack_tangents peels the tangent stack K -> a K-shaped tree of tangents that SHARE the
        # frame (same T3Frame object, one tangent space). stack_tangents inverts it.
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for FRAME_STACK, V in self.fv_stack_shapes:
            with self.subTest(FRAME_STACK=FRAME_STACK, V=V):
                v = self._random_v_stacked(STRUCT, FRAME_STACK, V)
                dense = np.asarray(v.to_dense())  # K + C + (N...)
                tree = v.unstack_tangents()

                for vidx in np.ndindex(*V):
                    leaf = _tree_get(tree, vidx)
                    self.assertIs(leaf.frame, v.frame)  # shared frame object
                    self.assertEqual((), leaf.tangent_stack_shape)
                    self.assertEqual(FRAME_STACK, leaf.frame_stack_shape)
                    self.check_relerr(dense[vidx], leaf.to_dense())  # slice the leading V axes

                rt = t3m.T3Tangent.stack_tangents(tree)  # round-trip
                self.assertIs(rt.frame, v.frame)
                self.assertEqual(V, rt.tangent_stack_shape)
                self.check_relerr(dense, rt.to_dense())

    def test_unstack_stack_frame(self):
        # unstack_frame peels the frame stack C -> a C-shaped tree of single-base-point tangents (each
        # at a DIFFERENT base point, still carrying its K batch). stack_frame inverts it.
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for FRAME_STACK, V in self.fv_stack_shapes:
            with self.subTest(FRAME_STACK=FRAME_STACK, V=V):
                v = self._random_v_stacked(STRUCT, FRAME_STACK, V)
                dense = np.asarray(v.to_dense())  # K + C + (N...)
                nV = len(V)
                tree = v.unstack_frame()

                for gidx in np.ndindex(*FRAME_STACK):
                    leaf = _tree_get(tree, gidx)
                    self.assertEqual((), leaf.frame_stack_shape)
                    self.assertEqual(V, leaf.tangent_stack_shape)
                    ref = dense[(slice(None),) * nV + gidx]  # slice the interior C axes
                    self.check_relerr(ref, leaf.to_dense())

                rt = t3m.T3Tangent.stack_frame(tree)  # round-trip
                self.assertEqual(FRAME_STACK, rt.frame_stack_shape)
                self.assertEqual(V, rt.tangent_stack_shape)
                self.check_relerr(dense, rt.to_dense())

    def test_stack_tangents_guard(self):
        # stack_tangents requires the same frame (numerical check): value-equal frames stack fine,
        # a genuinely different frame raises.
        x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        frame_a, _ = bvf.t3_orthogonal_representations(x)
        frame_b = bvf.T3Frame(*frame_a.data)               # value-equal, different object
        ta, ta2 = t3m.COREWISE.randn(frame_a), t3m.COREWISE.randn(frame_a)
        tb = t3m.COREWISE.randn(frame_b)

        t3m.T3Tangent.stack_tangents([ta, ta2])          # same object: OK
        t3m.T3Tangent.stack_tangents([ta, tb])           # value-equal frame: OK (numerical guard accepts)

        frame_c, _ = bvf.t3_orthogonal_representations(   # a genuinely different frame -> raises
            t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1)))
        tc = t3m.COREWISE.randn(frame_c)
        with self.assertRaises(ValueError):
            t3m.T3Tangent.stack_tangents([ta, tc])

    def test_orthogonal_gauge_projection(self):
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                    frame, _ = bvf.t3_orthogonal_representations(x)
                    u = t3m.COREWISE.randn(frame)
                    ug = t3m.MANIFOLD.project(u)

                    self.assertTrue(ug.is_gauged().all())
                    # orthogonal projection: the removed component is perpendicular to the projection
                    residual_dot_proj = cw.corewise_dot(
                        cw.corewise_sub(u.variations.data, ug.variations.data), ug.variations.data)
                    scale = float(cw.corewise_dot(ug.variations.data, ug.variations.data))
                    self.assertLessEqual(abs(float(residual_dot_proj)), tol * max(1.0, scale))

    def test_oblique_gauge_projection(self):
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                    frame, _ = bvf.t3_orthogonal_representations(x)
                    u = t3m.COREWISE.randn(frame)
                    uo = t3m.MANIFOLD.project_oblique(u)

                    self.assertTrue(uo.is_gauged().all())
                    self.check_relerr(u.to_dense(), uo.to_dense())  # preserves the tangent vector

    def test_inner_norm_faithfulness(self):
        # orthogonal + minimal-rank frame + gauged variations => corewise inner/norm == Hilbert-Schmidt.
        # inner/norm vectorize over the stack, returning a STACK_SHAPE-shaped array (one value/slice).
        for STACK_SHAPE in [(), (2,), (2, 3)]:
            with self.subTest(STACK_SHAPE=STACK_SHAPE):
                x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                frame, _ = bvf.t3_orthogonal_representations(x)
                self.assertTrue(frame.is_orthogonal().all() and frame.has_minimal_ranks)

                u = t3m.MANIFOLD.randn(frame)  # gauged by default
                w = t3m.MANIFOLD.randn(frame)

                ud, wd = np.asarray(u.to_dense()), np.asarray(w.to_dense())
                tensor_axes = tuple(range(len(STACK_SHAPE), ud.ndim))  # the (N0..Nd) axes; keep stack
                hs_inner = np.sum(ud * wd, axis=tensor_axes)  # shape = STACK_SHAPE
                hs_norm = np.sqrt(np.sum(ud * ud, axis=tensor_axes))
                self.assertLessEqual(norm(np.asarray(u.corewise_inner(w)) - hs_inner), tol * max(1.0, norm(hs_inner)))
                self.assertLessEqual(norm(np.asarray(u.corewise_norm()) - hs_norm), tol * max(1.0, norm(hs_norm)))

    def test_geometry_inner_norm(self):
        '''The geometry-level metrics: MANIFOLD.inner/norm = Hilbert-Schmidt (safe mode checks the frame
        orthogonal + variations gauged); COREWISE.inner/norm = Euclidean (no orth/gauge check). Both equal
        the raw corewise op where their preconditions hold; the manifold one REJECTS a non-orthonormal
        frame or ungauged variations in safe mode, and skips the check under safety.unsafe().'''
        x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        frame, _ = bvf.t3_orthogonal_representations(x)
        g, h = t3m.MANIFOLD.randn(frame), t3m.MANIFOLD.randn(frame)          # gauged, orthonormal frame
        # HS == corewise for a gauged tangent at an orthonormal frame
        self.assertAlmostEqual(float(t3m.MANIFOLD.inner(g, h)), float(g.corewise_inner(h)))
        self.assertAlmostEqual(float(t3m.MANIFOLD.norm(g)), float(g.corewise_norm()))
        # Euclidean geometry: no orth/gauge requirement, == corewise even on a raw (ungauged) tangent
        raw = t3m.COREWISE.randn(frame)
        self.assertAlmostEqual(float(t3m.COREWISE.inner(raw, raw)), float(raw.corewise_inner(raw)))
        self.assertAlmostEqual(float(t3m.COREWISE.norm(raw)), float(raw.corewise_norm()))
        # safe mode: MANIFOLD.inner/norm reject ungauged variations ...
        with self.assertRaises(ValueError):
            t3m.MANIFOLD.inner(raw, raw)
        with self.assertRaises(ValueError):
            t3m.MANIFOLD.norm(raw)
        # ... and a non-orthonormal frame (the corewise (U,G,G,G) frame)
        cg = t3m.COREWISE.randn(t3m.COREWISE.frame(x))
        with self.assertRaises(ValueError):
            t3m.MANIFOLD.inner(cg, cg)
        # same-frame: tangents at different base points are rejected
        frame2, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1)))
        with self.assertRaises(ValueError):
            t3m.MANIFOLD.inner(g, t3m.MANIFOLD.randn(frame2))
        # unsafe mode: checks skipped -> MANIFOLD.inner falls back to the raw corewise dot
        with safety.unsafe():
            self.assertAlmostEqual(float(t3m.MANIFOLD.inner(raw, raw)), float(raw.corewise_inner(raw)))

    def test_inner_norm_tangent_stacked(self):
        # A T3Tangent may carry an extra OUTER tangent stack K (a batch of tangents sharing one frame);
        # inner/norm vectorize over the full K + C stack.
        for FRAME_STACK, V in [((), (3,)), ((2,), (3,)), ((2,), ())]:
            with self.subTest(FRAME_STACK=FRAME_STACK, V=V):
                x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=FRAME_STACK)
                frame, _ = bvf.t3_orthogonal_representations(x)
                u = t3m.COREWISE.randn(frame, stack_shape=V)
                w = t3m.COREWISE.randn(frame, stack_shape=V)

                self.assertEqual(V, u.tangent_stack_shape)
                self.assertEqual(FRAME_STACK, u.frame_stack_shape)
                self.assertEqual(V + FRAME_STACK, u.stack_shape)

                full = V + FRAME_STACK
                ip = np.asarray(u.corewise_inner(w))
                self.assertEqual(full, ip.shape)

                ref = np.zeros(full)  # per-slice corewise dot over the full stack
                for idx in np.ndindex(*full):
                    ud = (tuple(np.asarray(c)[idx] for c in u.variations.tucker_variations),
                          tuple(np.asarray(c)[idx] for c in u.variations.tt_variations))
                    wd = (tuple(np.asarray(c)[idx] for c in w.variations.tucker_variations),
                          tuple(np.asarray(c)[idx] for c in w.variations.tt_variations))
                    ref[idx] = float(cw.corewise_dot(ud, wd))
                self.assertLessEqual(norm(ip - ref), tol * max(1.0, norm(ref)))
                self.assertLessEqual(norm(np.asarray(u.corewise_norm()) - np.sqrt(np.abs(np.asarray(u.corewise_inner(u))))), tol)

    def test_tangent_probe(self):
        # forward J^(s): v.probe(ww) == dense_probe(ww, v.to_dense()); probes are stacked W + K + C + (N,)
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for FRAME_STACK in [(), (2,)]:
            for PROBE_STACK in [(), (2,)]:
                for TANGENT_STACK in [(), (2,)]:
                    with self.subTest(BASE=FRAME_STACK, PROBE=PROBE_STACK, TANGENT=TANGENT_STACK):
                        rnd = np.random.randn
                        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=FRAME_STACK)
                        frame, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.COREWISE.randn(frame, stack_shape=TANGENT_STACK)
                        ww = tuple(rnd(*(PROBE_STACK + (N,))) for N in STRUCT[0])

                        zz = v.probe(ww)  # numpy/jax inferred from inputs
                        zz2 = t3p.dense_probe(ww, v.to_dense())  # dense ground truth, also W + K + C
                        self.assertEqual(
                            PROBE_STACK + TANGENT_STACK + FRAME_STACK + (STRUCT[0][0],),
                            tuple(np.asarray(zz[0]).shape),
                        )
                        for a, b in zip(zz, zz2):
                            self.check_relerr(b, a)

                        # K-stack contract: probing the batch == stacking the per-tangent probes
                        # (each single tangent shares the frame), inserted at the V axis (after F).
                        if TANGENT_STACK != ():
                            per = [leaf.probe(ww) for leaf in v.unstack_tangents()]
                            for i in range(len(STRUCT[0])):
                                stacked = np.stack([np.asarray(p[i]) for p in per], axis=len(PROBE_STACK))
                                self.check_relerr(stacked, np.asarray(zz[i]))

    def test_tangent_probe_transpose(self):
        # adjoint identity <z, J v> = <J^T z, v>; J^T accepts K-stacked residuals (W + K + C).
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for FRAME_STACK in [(), (2,)]:
            for PROBE_STACK in [(), (2,)]:
                for TANGENT_STACK in [(), (2,)]:
                    with self.subTest(BASE=FRAME_STACK, PROBE=PROBE_STACK, TANGENT=TANGENT_STACK):
                        rnd = np.random.randn
                        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=FRAME_STACK)
                        frame, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.COREWISE.randn(frame, stack_shape=TANGENT_STACK)
                        ww = tuple(rnd(*(PROBE_STACK + (N,))) for N in STRUCT[0])
                        # residuals live in the forward probe space: W + K + C + (N,)
                        z = tuple(rnd(*(PROBE_STACK + TANGENT_STACK + FRAME_STACK + (N,))) for N in STRUCT[0])

                        Jv = v.probe(ww)  # numpy/jax inferred from inputs
                        JTz = t3m.T3Tangent.probe_transpose(z, ww, frame, sum_over_probes=True)
                        # full contraction of both sides (sums F, V, G, N) must agree
                        lhs = float(np.sum([np.sum(np.asarray(a) * np.asarray(b)) for a, b in zip(z, Jv)]))
                        rhs = float(np.sum(np.asarray(JTz.corewise_inner(v))))
                        self.assertLessEqual(abs(lhs - rhs), tol * max(1.0, abs(lhs)))

                        # sum=True keeps V (frame G), drops F; sum=False keeps F+V (frame G)
                        self.assertEqual(TANGENT_STACK, JTz.tangent_stack_shape)
                        self.assertEqual(FRAME_STACK, JTz.frame_stack_shape)
                        JTz_batch = t3m.T3Tangent.probe_transpose(z, ww, frame)  # sum_over_probes=False
                        self.assertEqual(PROBE_STACK + TANGENT_STACK, JTz_batch.tangent_stack_shape)
                        self.assertEqual(FRAME_STACK, JTz_batch.frame_stack_shape)

                        # sum=True == sum over the probe stack W of sum=False (validates sum=False)
                        f_axes = tuple(range(len(PROBE_STACK)))
                        for cs, cn in zip(JTz.variations.tucker_variations, JTz_batch.variations.tucker_variations):
                            self.check_relerr(np.asarray(cs), np.asarray(cn).sum(axis=f_axes))
                        for cs, cn in zip(JTz.variations.tt_variations, JTz_batch.variations.tt_variations):
                            self.check_relerr(np.asarray(cs), np.asarray(cn).sum(axis=f_axes))

    def test_tangent_apply(self):
        # apply(v, ww) contracts the dense tangent in ALL modes; result is stacked W + K + C.
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for FRAME_STACK in [(), (2,)]:
            for W in [(), (2,)]:           # apply-vector stack
                for K in [(), (3,)]:       # tangent stack
                    with self.subTest(BASE=FRAME_STACK, W=W, K=K):
                        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=FRAME_STACK)
                        frame, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.COREWISE.randn(frame, stack_shape=K)
                        ww = tuple(np.random.randn(*(W + (N,))) for N in STRUCT[0])

                        a = np.asarray(v.apply(ww))
                        self.assertEqual(W + K + FRAME_STACK, a.shape)
                        vd = np.asarray(v.to_dense())   # K + C + (N0,N1,N2)
                        if W:   # shared apply-vector stack w across all modes
                            ref = np.einsum('...ijk,wi,wj,wk->w...', vd, ww[0], ww[1], ww[2])
                        else:
                            ref = np.einsum('...ijk,i,j,k->...', vd, ww[0], ww[1], ww[2])
                        self.check_relerr(ref, a)

    def test_tangent_entries(self):
        # entries(v, idx) extracts entries of the dense tangent (no contraction); result W + K + C.
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for FRAME_STACK in [(), (2,)]:
            for W in [(), (2,)]:           # index stack
                for K in [(), (3,)]:
                    with self.subTest(BASE=FRAME_STACK, W=W, K=K):
                        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=FRAME_STACK)
                        frame, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.COREWISE.randn(frame, stack_shape=K)
                        idx = np.array(tuple(np.random.randint(0, N, size=W) for N in STRUCT[0]))  # (d,)+W

                        e = np.asarray(v.entries(idx))
                        self.assertEqual(W + K + FRAME_STACK, e.shape)
                        vd = np.asarray(v.to_dense())
                        if W:
                            ref = np.stack([vd[..., idx[0, w], idx[1, w], idx[2, w]] for w in range(W[0])], axis=0)
                        else:
                            ref = vd[..., int(idx[0]), int(idx[1]), int(idx[2])]
                        self.check_relerr(ref, e)

    def test_tangent_apply_transpose(self):
        # adjoint identity <apply^T c, v> == sum_W c*apply(v) (keep frame C); keep-W -> tangent stack W.
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for FRAME_STACK in [(), (2,)]:
            for W in [(), (2,)]:
                with self.subTest(BASE=FRAME_STACK, W=W):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=FRAME_STACK)
                    frame, _ = bvf.t3_orthogonal_representations(x)
                    v = t3m.COREWISE.randn(frame)
                    ww = tuple(np.random.randn(*(W + (N,))) for N in STRUCT[0])
                    c = np.asarray(np.random.randn(*(W + FRAME_STACK)))

                    ATc = t3m.T3Tangent.apply_transpose(c, ww, frame, sum_over_probes=True)
                    lhs = np.asarray(cw.corewise_stack_dot(ATc.variations.data, v.variations.data, len(FRAME_STACK)))
                    rhs = np.sum(c * np.asarray(v.apply(ww)), axis=tuple(range(len(W))))   # sum_W, keep C
                    self.check_relerr(rhs, lhs)
                    self.assertEqual((), ATc.tangent_stack_shape)               # summed over W
                    self.assertEqual(FRAME_STACK, ATc.frame_stack_shape)

                    # without summing, W becomes the tangent stack; sum=True == sum_W of sum=False
                    ATc_keep = t3m.T3Tangent.apply_transpose(c, ww, frame)
                    self.assertEqual(W, ATc_keep.tangent_stack_shape)
                    self.assertEqual(FRAME_STACK, ATc_keep.frame_stack_shape)
                    f_axes = tuple(range(len(W)))
                    for cs, cn in zip(ATc.variations.tucker_variations, ATc_keep.variations.tucker_variations):
                        self.check_relerr(np.asarray(cs), np.asarray(cn).sum(axis=f_axes))
                    for cs, cn in zip(ATc.variations.tt_variations, ATc_keep.variations.tt_variations):
                        self.check_relerr(np.asarray(cs), np.asarray(cn).sum(axis=f_axes))

    def test_tangent_entries_transpose(self):
        # adjoint identity <entries^T c, v> == sum_W c*entries(v, idx) (keep frame C).
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for FRAME_STACK in [(), (2,)]:
            for W in [(), (2,)]:
                with self.subTest(BASE=FRAME_STACK, W=W):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=FRAME_STACK)
                    frame, _ = bvf.t3_orthogonal_representations(x)
                    v = t3m.COREWISE.randn(frame)
                    idx = np.array(tuple(np.random.randint(0, N, size=W) for N in STRUCT[0]))  # (d,)+W
                    c = np.asarray(np.random.randn(*(W + FRAME_STACK)))

                    ETc = t3m.T3Tangent.entries_transpose(c, idx, frame, sum_over_probes=True)
                    lhs = np.asarray(cw.corewise_stack_dot(ETc.variations.data, v.variations.data, len(FRAME_STACK)))
                    rhs = np.sum(c * np.asarray(v.entries(idx)), axis=tuple(range(len(W))))
                    self.check_relerr(rhs, lhs)
                    self.assertEqual(W, t3m.T3Tangent.entries_transpose(c, idx, frame).tangent_stack_shape)

    def test_tangent_apply_transpose_kstack(self):
        # The adjoint-state apply/entries transpose is K-aware: the residual c may carry a tangent
        # stack K (W + K + C), the output space of a K-stacked forward. (The old scatter could NOT --
        # it had no adjoint sweep to propagate K.) Adjoint identity per k.
        import t3toolbox.backend.probing as pr
        STRUCT = ((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
        NW, K = 4, 3
        frame, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(*STRUCT))
        ww = tuple(np.random.randn(NW, N) for N in STRUCT[0])
        idx = np.array(tuple(np.random.randint(0, N, size=NW) for N in STRUCT[0]))
        vs = [t3m.MANIFOLD.randn(frame) for _ in range(K)]
        for op, fwd, tr in [('apply', lambda v: apply.tv_apply(ww, v.variations.data, frame.data),
                                      lambda c: apply.tv_apply_transpose(c, ww, frame.data, sum_over_probes=True)),
                            ('entries', lambda v: entries.tv_entries(idx, v.variations.data, frame.data),
                                        lambda c: entries.tv_entries_transpose(c, idx, frame.data, sum_over_probes=True))]:
            with self.subTest(op=op):
                JvK = np.stack([np.asarray(fwd(v)) for v in vs], axis=1)     # base-inner W + K
                c = np.random.randn(NW, K)
                dU, dG = tr(c)                                               # -> K + cores
                for k in range(K):
                    grad_k = (tuple(np.asarray(g)[k] for g in dU), tuple(np.asarray(g)[k] for g in dG))
                    lhs = float(np.sum(c[:, k] * JvK[:, k]))
                    rhs = float(cw.corewise_dot(grad_k, vs[k].variations.data))
                    self.check_relerr(np.array(rhs), np.array(lhs))

    def test_randn(self):
        frame, _ = bvf.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1)))
        self.assertTrue(t3m.MANIFOLD.randn(frame).is_gauged().all())                                   # gauged by default
        self.assertFalse(t3m.COREWISE.randn(frame).is_gauged().all())    # ungauged on request
        self.assertEqual(frame.stack_shape, t3m.MANIFOLD.randn(frame).stack_shape)                # construction validates fit

    def test_to_t3(self):
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                    frame, _ = bvf.t3_orthogonal_representations(x)
                    v = t3m.COREWISE.randn(frame)
                    base_point = t3.TuckerTensorTrain(frame.up_tucker_cores, frame.left_tt_cores).to_dense()

                    self.check_relerr(v.to_dense(), v.to_t3().to_dense())
                    self.check_relerr(np.asarray(base_point) + np.asarray(v.to_dense()),
                                      v.to_t3(include_shift=True).to_dense())

    def test_retract(self):
        # Dense correctness: retract(0) == base point; retract(v) == best rank-(frame) T3-SVD of (base point + v).
        for T3_STRUCTURE in [((10, 11), (3, 4), (1, 2, 1)), ((9, 10, 11, 12), (2, 3, 3, 2), (1, 2, 3, 2, 1))]:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                    frame, _ = bvf.t3_orthogonal_representations(x)
                    base_point = t3.TuckerTensorTrain(frame.up_tucker_cores, frame.left_tt_cores).to_dense()

                    self.check_relerr(base_point, t3m.MANIFOLD.retract(t3m.T3Tangent.zeros(frame)).to_dense())

                    v = t3m.COREWISE.randn(frame)
                    if STACK_SHAPE == ():  # compare against a from-dense T3-SVD of (base point + v)
                        shifted_dense = np.asarray(base_point) + np.asarray(v.to_dense())
                        ref, _, _ = t3.TuckerTensorTrain.t3svd_dense(
                            shifted_dense, max_tucker_ranks=tuple(frame.up_ranks), max_tt_ranks=tuple(frame.left_ranks))
                        self.check_relerr(ref.to_dense(), t3m.MANIFOLD.retract(v).to_dense())

        # Rank preservation holds on a minimal-rank frame.
        for STACK_SHAPE in [(), (2,)]:
            with self.subTest(rank_preservation=True, STACK_SHAPE=STACK_SHAPE):
                x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                frame, _ = bvf.t3_orthogonal_representations(x)
                self.assertTrue(frame.has_minimal_ranks)
                r = t3m.MANIFOLD.retract(t3m.COREWISE.randn(frame))
                self.assertEqual(tuple(frame.up_ranks), r.tucker_ranks)
                self.assertEqual(tuple(frame.left_ranks), r.tt_ranks)

    def test_project(self):
        # project(x, frame) is the orthogonal projection of x onto the tangent space T_P M.
        for STR_P, STR_X in [
            (((6, 7, 5), (2, 2, 2), (1, 2, 2, 1)), ((6, 7, 5), (3, 4, 3), (1, 2, 2, 1))),
            (((9, 10, 11, 12), (2, 3, 3, 2), (1, 2, 3, 2, 1)), ((9, 10, 11, 12), (3, 3, 4, 2), (1, 2, 2, 2, 1))),
        ]:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(STR_P=STR_P, STACK_SHAPE=STACK_SHAPE):
                    p = t3.TuckerTensorTrain.randn(*STR_P, stack_shape=STACK_SHAPE)
                    x = t3.TuckerTensorTrain.randn(*STR_X, stack_shape=STACK_SHAPE)
                    frame, _ = bvf.t3_orthogonal_representations(p)

                    proj = t3m.MANIFOLD.project_ambient(frame, x)
                    self.assertTrue(proj.is_gauged().all())

                    # idempotency: projecting a tangent vector (its unshifted embedding) recovers it
                    v = t3m.COREWISE.randn(frame)
                    proj_v = t3m.MANIFOLD.project_ambient(frame, v.to_t3())
                    self.check_relerr(v.to_dense(), proj_v.to_dense())

                    # orthogonality: the residual x - proj_x is perpendicular to the tangent space
                    residual = np.asarray(x.to_dense()) - np.asarray(proj.to_dense())
                    tensor_axes = tuple(range(len(STACK_SHAPE), residual.ndim))
                    for _ in range(3):
                        w_dense = np.asarray(
                            t3m.COREWISE.randn(frame).to_dense())
                        ip = norm(np.sum(residual * w_dense, axis=tensor_axes))
                        self.assertLessEqual(float(ip), tol * norm(residual) * norm(w_dense))

    def test_tangent_stacked_heavy_ops(self):
        # A K-stacked tangent is a batch of tangent vectors sharing one frame (one per (v, g) pair).
        # to_dense/to_t3/retract produce a K+C-stacked result whose every slice matches the
        # corresponding unstacked tangent (the shared base point replicated across the tangent stack K).
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))  # minimal-rank, so retract preserves ranks
        for FRAME_STACK, V in [((), (3,)), ((2,), (3,)), ((2,), ()), ((2,), (2, 2))]:
            with self.subTest(FRAME_STACK=FRAME_STACK, V=V):
                x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=FRAME_STACK)
                frame, _ = bvf.t3_orthogonal_representations(x)
                var = t3m.COREWISE.randn(frame, stack_shape=V).variations
                v = t3m.T3Tangent(frame, var)
                full = V + FRAME_STACK
                n_frame = len(FRAME_STACK)

                dense = np.asarray(v.to_dense())
                t3_dense = np.asarray(v.to_t3().to_dense())
                shifted = np.asarray(v.to_t3(include_shift=True).to_dense())
                retr = np.asarray(t3m.MANIFOLD.retract(v).to_dense())
                self.assertEqual(full + STRUCT[0], dense.shape)

                for idx in np.ndindex(*full):
                    s = _slice_tangent(frame, var, idx, n_frame)
                    self.check_relerr(s.to_dense(), dense[idx])
                    self.check_relerr(s.to_dense(), t3_dense[idx])  # to_t3 round-trips to to_dense
                    self.check_relerr(s.to_t3(include_shift=True).to_dense(), shifted[idx])
                    self.check_relerr(t3m.MANIFOLD.retract(s).to_dense(), retr[idx])

    def test_project_tangent_stacked(self):
        # project a BATCH of inputs x (stack K+C) onto a frame (stack G): the result is a K-stacked
        # tangent whose (v, g) slice equals projecting x[v, g] onto the shared base point frame[g].
        STR_P = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        STR_X = ((6, 7, 5), (3, 4, 3), (1, 2, 2, 1))
        for FRAME_STACK, V in [((), (3,)), ((2,), (3,))]:
            with self.subTest(FRAME_STACK=FRAME_STACK, V=V):
                p = t3.TuckerTensorTrain.randn(*STR_P, stack_shape=FRAME_STACK)
                x = t3.TuckerTensorTrain.randn(*STR_X, stack_shape=(V + FRAME_STACK))
                frame, _ = bvf.t3_orthogonal_representations(p)

                proj = t3m.MANIFOLD.project_ambient(frame, x)
                self.assertEqual(V, proj.tangent_stack_shape)
                self.assertEqual(FRAME_STACK, proj.frame_stack_shape)
                self.assertTrue(proj.is_gauged().all())

                proj_dense = np.asarray(proj.to_dense())  # K + C + (N...)
                full = V + FRAME_STACK
                n_frame = len(FRAME_STACK)
                for idx in np.ndindex(*full):
                    g_idx = idx[len(idx) - n_frame:] if n_frame > 0 else ()
                    ref = t3m.MANIFOLD.project_ambient(_slice_frame(frame, g_idx), _slice_t3(x, idx))
                    self.check_relerr(ref.to_dense(), proj_dense[idx])

    def test_normalized(self):
        # normalized() rescales each stacked tangent to unit norm (HS norm, since randn is gauged).
        for STACK_SHAPE in [(), (2,)]:
            for V in [(), (3,)]:
                with self.subTest(STACK_SHAPE=STACK_SHAPE, V=V):
                    x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                    frame, _ = bvf.t3_orthogonal_representations(x)
                    vn = t3m.MANIFOLD.randn(frame, stack_shape=V).normalized()
                    vn.validate()
                    self.assertLessEqual(norm(np.asarray(vn.corewise_norm()) - 1.0), tol)

    def test_allclose(self):
        # T3Tangent.allclose compares two tangents at the same base point.
        for STACK_SHAPE in [(), (2,)]:
            with self.subTest(STACK_SHAPE=STACK_SHAPE):
                x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                frame, _ = bvf.t3_orthogonal_representations(x)
                v = t3m.MANIFOLD.randn(frame)
                self.assertTrue(v.allclose(v).all())
                self.assertFalse(v.allclose(v * 2.0).all())
                self.assertTrue(v.allclose(v * (1.0 + 1e-12)).all())

    def test_project_dense_onto_tangent(self):
        # project_dense_onto_tangent == the dense orthogonal projector onto the tangent space.
        STR_P = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        frame, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(*STR_P))
        Pr = _dense_tangent_projector(frame)
        Z = np.random.randn(*STR_P[0])
        F = t3m.MANIFOLD.project_ambient(frame, Z)
        self.assertTrue(F.is_gauged().all())
        self.check_relerr((Pr @ Z.reshape(-1)).reshape(STR_P[0]), F.to_dense())

        # stacked C=(2,): valid gauged tangent with the right stack; matches the projector per slice
        frame2, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(*STR_P, stack_shape=(2,)))
        Z2 = np.random.randn(2, *STR_P[0])
        F2 = t3m.MANIFOLD.project_ambient(frame2, Z2)
        F2.validate()
        self.assertEqual((2,), F2.stack_shape)
        for i in range(2):
            Pri = _dense_tangent_projector(_slice_frame(frame2, (i,)))
            self.check_relerr((Pri @ Z2[i].reshape(-1)).reshape(STR_P[0]), np.asarray(F2.to_dense())[i])

        # both methods ('contraction' default, 't3svd') give the same projection.
        for method in ('contraction', 't3svd'):
            Fm = t3m.MANIFOLD.project_ambient(frame, Z, method=method)
            self.check_relerr((Pr @ Z.reshape(-1)).reshape(STR_P[0]), Fm.to_dense())
        with self.assertRaises(ValueError):
            t3m.MANIFOLD.project_ambient(frame, Z, method='bogus')

        # NON-minimal orthogonal frame: still matches (orthogonality is required, minimal rank is NOT).
        x_pad = t3.TuckerTensorTrain.randn(STR_P[0], (2, 2, 2), (1, 2, 2, 1)).resize(
            new_shape=STR_P[0], new_tucker_ranks=(3, 4, 2), new_tt_ranks=(1, 3, 3, 1))
        frame_nm = bvf.T3Frame.from_t3(x_pad)
        self.assertFalse(frame_nm.has_minimal_ranks)
        self.assertTrue(frame_nm.is_orthogonal().all())
        # reference: orthonormal projector onto the span of all dense unit tangents (any rank).
        cols = [np.asarray(t3m.T3Tangent.unit(frame_nm, (use_tt, i, tuple(idx))).to_dense()).reshape(-1)
                for use_tt, shapes in zip((False, True), frame_nm.variation_shapes)
                for i, shp in enumerate(shapes) for idx in np.ndindex(*shp)]
        A = np.stack(cols, axis=1)
        # rcond=1e-8: A is rank-deficient (redundant tangent directions); the default rcond ~1e-15 is
        # too close to its ~1e-16 null singular values and flakily contaminates A @ pinv(A) -- which
        # masquerades as a project_dense_onto_tangent bug. See _dense_tangent_projector.
        Pr_nm = A @ np.linalg.pinv(A, rcond=1e-8)
        self.check_relerr((Pr_nm @ Z.reshape(-1)).reshape(STR_P[0]),
                          t3m.MANIFOLD.project_ambient(frame_nm, Z).to_dense())

    def test_riemannian_gradient(self):
        # Riemannian gradient = tangent-space projection of the Euclidean gradient. Oracle: the explicit
        # dense projector Pr, applied to a dense Z and to a T3 g's dense form -- so the dense-input and
        # T3-input routes of project_ambient are both checked against ground truth (review O1-5: the
        # old body compared each expression with itself, vacuously).
        STR_P = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        frame, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(*STR_P))
        Pr = _dense_tangent_projector(frame)
        Z = np.random.randn(*STR_P[0])
        self.check_relerr((Pr @ Z.reshape(-1)).reshape(STR_P[0]),
                          t3m.MANIFOLD.project_ambient(frame, Z).to_dense())
        g = t3.TuckerTensorTrain.randn((6, 7, 5), (3, 4, 3), (1, 2, 2, 1))
        self.check_relerr((Pr @ g.to_dense().reshape(-1)).reshape(STR_P[0]),
                          t3m.MANIFOLD.project_ambient(frame, g).to_dense())

    def test_transport(self):
        # Projective transport == dense projection onto the new tangent space; result lives at new frame.
        STR = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        frame, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(*STR))
        new_frame, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(*STR))
        v = t3m.MANIFOLD.randn(frame)

        # transport to its own frame is the identity (v is already in T_frame M)
        self.check_relerr(v.to_dense(), t3m.MANIFOLD.transport(v, frame).to_dense())

        # transport to a different frame == dense projection onto T_new M
        Pr_new = _dense_tangent_projector(new_frame)
        vt = t3m.MANIFOLD.transport(v, new_frame)
        vt.validate()
        self.assertIs(new_frame, vt.frame)
        self.assertTrue(vt.is_gauged().all())
        self.check_relerr((Pr_new @ np.asarray(v.to_dense()).reshape(-1)).reshape(STR[0]), vt.to_dense())


    def test_derivative_methods(self):
        # T3Tangent.{probe,apply,entries}_derivatives + their transposes: order-0 == the plain op,
        # the transpose returns a T3Tangent at the same frame satisfying the adjoint identity, and the
        # X/P sample-stack consistency check is a hard error.
        STR = ((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        shapes = STR[0]
        frame, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(*STR))
        v = t3m.COREWISE.randn(frame)
        W = (2,)
        ww = [np.random.randn(*(W + (N,))) for N in shapes]
        pp = [np.random.randn(*(W + (N,))) for N in shapes]
        index = np.stack([np.random.randint(0, N, size=W) for N in shapes], axis=0)
        ORDER = 3

        # order 0 == the non-derivative op
        for zj, z0 in zip(v.probe_derivatives(ww, pp, ORDER), v.probe(ww)):
            self.check_relerr(np.asarray(z0), np.asarray(zj)[0])
        self.check_relerr(np.asarray(v.apply(ww)), np.asarray(v.apply_derivatives(ww, pp, ORDER))[0])
        self.check_relerr(np.asarray(v.entries(index)), np.asarray(v.entries_derivatives(index, pp, ORDER))[0])

        # tangent transpose: returns a T3Tangent at the same frame; adjoint identity <r, J v> = <J^T r, v>
        for kind in ['probe', 'apply', 'entries']:
            if kind == 'probe':
                Jv = v.probe_derivatives(ww, pp, ORDER)
                r = [np.random.randn(*np.asarray(z).shape) for z in Jv]
                JTr = t3m.T3Tangent.probe_derivatives_transpose(r, ww, pp, frame, ORDER, sum_over_probes=True)
                lhs = sum(float(np.sum(ri * np.asarray(zi))) for ri, zi in zip(r, Jv))
            elif kind == 'apply':
                Jv = v.apply_derivatives(ww, pp, ORDER)
                r = np.random.randn(*np.asarray(Jv).shape)
                JTr = t3m.T3Tangent.apply_derivatives_transpose(r, ww, pp, frame, ORDER, sum_over_probes=True)
                lhs = float(np.sum(r * np.asarray(Jv)))
            else:
                Jv = v.entries_derivatives(index, pp, ORDER)
                r = np.random.randn(*np.asarray(Jv).shape)
                JTr = t3m.T3Tangent.entries_derivatives_transpose(r, index, pp, frame, ORDER, sum_over_probes=True)
                lhs = float(np.sum(r * np.asarray(Jv)))
            self.assertIs(frame, JTr.frame)
            self.assertLessEqual(abs(lhs - float(JTr.corewise_inner(v))) / abs(lhs), 1e-9)

        # X/P sample-stack consistency: hard error
        pp_bad = [np.random.randn(*((3,) + (N,))) for N in shapes]   # W=(3,) != (2,)
        with self.assertRaises(ValueError):
            v.probe_derivatives(ww, pp_bad, ORDER)
        with self.assertRaises(ValueError):
            v.entries_derivatives(index, pp_bad, ORDER)




class TestStructuralTangentMismatch(unittest.TestCase):
    """Review 2026-08-22 (S12): tangents at frames of DIFFERENT rank structure are a structural error in
    every mode -- the numerical same-frame guard (skipped under unsafe()/jit) used to be the only check,
    and with broadcastable holes `a + b` silently returned a tangent of a's structure."""

    def test_different_structure_raises_even_in_unsafe_mode(self):
        np.random.seed(21)
        a = _random_tangent(((4, 5, 3), (2, 2, 2), (1, 2, 2, 1)))
        b = _random_tangent(((4, 5, 3), (1, 1, 1), (1, 1, 1, 1)))    # broadcastable holes (rank 1)
        for ctx in (safety.safe, safety.unsafe):
            with self.subTest(mode=ctx.__name__):
                with ctx():
                    with self.assertRaises(ValueError):
                        a + b
                    with self.assertRaises(ValueError):
                        a.corewise_inner(b)

    def test_fitting_model_rejects_wrong_structure_in_unsafe_mode(self):
        import t3toolbox.fitting as fitting
        np.random.seed(22)
        x = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        ww = tuple(np.random.randn(6, n) for n in x.shape)
        model = fitting.apply_model(t3m.MANIFOLD, x, ww, x.apply(ww) * 0.5)
        p = _random_tangent(((4, 5, 3), (1, 1, 1), (1, 1, 1, 1)))
        with safety.unsafe():
            with self.assertRaises(ValueError):
                model.gn_quadratic(p)



class TestD1Degenerate(unittest.TestCase):
    """Review 2026-08-22 (C1): a one-mode T3 is a vector; the T3-gradient projection and transport used
    to index an empty TT chain (IndexError) while every other d = 1 op worked."""

    def test_project_ambient_t3_and_transport_at_d1(self):
        np.random.seed(31)
        for C in [(), (3,)]:
            with self.subTest(stack=C):
                x = t3.TuckerTensorTrain.randn((7,), (3,), (1, 1), stack_shape=C)
                g = t3.TuckerTensorTrain.randn((7,), (2,), (1, 1), stack_shape=C)
                frame = t3m.MANIFOLD.frame(x)
                via_t3 = t3m.MANIFOLD.project_ambient(frame, g).to_dense()
                via_dense = t3m.MANIFOLD.project_ambient(frame, g.to_dense()).to_dense()
                self.assertLess(norm(np.asarray(via_t3) - np.asarray(via_dense)), 1e-10)
                via_svd = t3m.MANIFOLD.project_ambient(frame, g.to_dense(), method='t3svd').to_dense()
                self.assertLess(norm(np.asarray(via_svd) - np.asarray(via_dense)), 1e-10)
                v = t3m.MANIFOLD.randn(frame)
                w = t3m.MANIFOLD.transport(v, t3m.MANIFOLD.frame(g))
                self.assertEqual(w.frame.stack_shape, C)



class TestGaugeResidualIsRelative(unittest.TestCase):
    """Review 2026-08-22 (C7): the gauge residual is scale-free, so safe mode neither rejects a large
    gauged tangent nor accepts a tiny ungauged one."""

    def test_scale_invariance_and_tiny_ungauged(self):
        np.random.seed(41)
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        frame = t3m.MANIFOLD.frame(x)
        v = t3m.MANIFOLD.project(t3m.COREWISE.randn(frame))
        for s in (1e-8, 1.0, 1e8, 1e12):
            with self.subTest(scale=s):
                w = v * s
                self.assertTrue(bool(w.is_gauged()))
                t3m.MANIFOLD.norm(w)                                     # safe mode accepts it
                t3m.MANIFOLD.inner(w, w)
        u = t3m.COREWISE.randn(frame) * 1e-12                          # tiny but NOT gauged
        self.assertFalse(bool(u.is_gauged()))
        with self.assertRaises(ValueError):
            t3m.MANIFOLD.norm(u)

if __name__ == "__main__":
    unittest.main()


class TestCorewiseRetractFrameGuard(unittest.TestCase):
    """Review H3-8: COREWISE.retract on a MANIFOLD-frame tangent used to die in a broadcast error on
    a slack frame (nD != nU); now a structural ValueError naming the frame-kind mismatch. The
    corewise path itself is unchanged."""

    def test_manifold_frame_tangent_is_rejected_when_detectable(self):
        np.random.seed(0)
        xs = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 4), (1, 2, 3, 1))   # slack: nD != nU
        fr_s = bvf.t3_orthogonal_representations(xs)[0]
        v = t3m.MANIFOLD.randn(fr_s)
        with self.assertRaises(ValueError):
            t3m.COREWISE.retract(v)

    def test_stack_tangents_mixed_K_raises_structurally(self):
        """H3-6: stacking a K=() tangent with a K=(2,) tangent used to die inside numpy with an
        inhomogeneous-shape error; now a ValueError naming the mismatched K shapes."""
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1))
        fr = bvf.t3_orthogonal_representations(x)[0]
        v = t3m.MANIFOLD.randn(fr)
        vk = t3m.MANIFOLD.randn(fr, stack_shape=(2,))
        with self.assertRaises(ValueError):
            t3m.T3Tangent.stack_tangents([v, vk])
        s2 = t3m.T3Tangent.stack_tangents([v, v])              # matching K still stacks
        self.assertEqual(s2.tangent_stack_shape, (2,))

    def test_tangent_truediv(self):
        """R1-12: v / 2 == v * 0.5 (new); array divisors raise."""
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1))
        v = t3m.MANIFOLD.randn(bvf.t3_orthogonal_representations(x)[0])
        got = (v / 2.0).corewise_norm()
        ref = (v * 0.5).corewise_norm()
        self.assertLess(abs(float(got) - float(ref)), 1e-12 * (abs(float(ref)) + 1))
        with self.assertRaises(TypeError):
            v / np.ones(3)

    def test_corewise_tangent_still_retracts(self):
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1))
        p = t3m.COREWISE.randn(t3m.COREWISE.frame(x))
        y = t3m.COREWISE.retract(p)
        self.assertEqual(y.shape, x.shape)
