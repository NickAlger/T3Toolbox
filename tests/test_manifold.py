# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
import t3toolbox.backend.probing as t3p

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm


def _random_variations(base):
    """A random T3Variations fitting the holes of ``base`` (ungauged)."""
    rnd = lambda *s: np.random.randn(*s)
    ss = base.stack_shape
    tucker_hole_shapes, tt_hole_shapes = base.variation_shapes
    V = tuple(rnd(*(ss + s)) for s in tucker_hole_shapes)
    H = tuple(rnd(*(ss + s)) for s in tt_hole_shapes)
    return bvf.T3Variations(V, H)


def _random_tangent(t3_structure, stack_shape=()):
    x = t3.TuckerTensorTrain.randn(*t3_structure, stack_shape=stack_shape)
    base, _ = bvf.t3_orthogonal_representations(x)
    return t3m.T3Tangent(base, _random_variations(base))


def _slice_basis(base, idx):
    """The unstacked T3Basis at stack index ``idx`` (idx=() returns the whole base)."""
    s = lambda C: np.asarray(C)[idx]
    up, down, left, right = base.data
    return bvf.T3Basis(tuple(map(s, up)), tuple(map(s, down)), tuple(map(s, left)), tuple(map(s, right)))


def _slice_t3(x, idx):
    """The unstacked TuckerTensorTrain at stack index ``idx``."""
    s = lambda C: np.asarray(C)[idx]
    tucker, tt = x.data
    return t3.TuckerTensorTrain(tuple(map(s, tucker)), tuple(map(s, tt)))


def _slice_tangent(base, var, idx, n_base):
    """The unstacked (basis, variation) tangent at full K+C index ``idx``.

    The base point is shared across the tangent stack K: the basis is sliced at the trailing C part
    of ``idx`` while the variation is sliced at the full ``idx``.
    """
    g_idx = idx[len(idx) - n_base:] if n_base > 0 else ()
    sV = lambda C: np.asarray(C)[idx]
    vslice = bvf.T3Variations(tuple(map(sV, var.tucker_variations)), tuple(map(sV, var.tt_variations)))
    return t3m.T3Tangent(_slice_basis(base, g_idx), vslice)


def _tree_get(tree, idx):
    """Navigate an array-like tree by a multi-index (idx=() returns the depth-0 tree itself)."""
    for k in idx:
        tree = tree[k]
    return tree


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
                base, _ = bvf.t3_orthogonal_representations(x)
                v = t3m.T3Tangent.randn(base, stack_shape=K, apply_gauge_projection=False)
                self.assertEqual(int(np.prod(STRUCT[0])), v.size)                 # dense element count
                self.assertEqual(v.basis.data_size + v.variations.data_size, v.data_size)
                self.assertEqual(base.minimal_ranks, v.minimal_ranks)            # delegates to basis
                self.assertEqual(t3m.manifold_dim((base.shape, base.up_ranks, base.left_ranks)),
                                 v.tangent_space_dimension)
                cp = v.copy(); cp.variations.tucker_variations[0][...] = 9.0      # copy is independent
                self.assertFalse(np.allclose(np.asarray(v.variations.tucker_variations[0]), 9.0))
                self.assertIn("T3Tangent", repr(v)); self.assertNotIn("array", repr(v))
                v.validate()   # valid tangent; also runs in __post_init__
        # __post_init__ validate rejects an incompatible (basis, variations) pair
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
        # T3Tangent.random_orthogonal / unit / zeros_like / randn_like.
        STRUCT = ((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        v = t3m.T3Tangent.random_orthogonal(*STRUCT, stack_shape=(2,), tangent_stack_shape=(3,))
        self.assertEqual(((2,), (3,)), (v.base_stack_shape, v.tangent_stack_shape))
        self.assertTrue(v.is_orthogonal() and v.is_gauged())                   # gauged by default
        base = bvf.T3Basis.random_orthogonal(*STRUCT)
        u = t3m.T3Tangent.unit(base, (True, 1, (0, 1, 0)))
        self.assertEqual(1, sum(int(np.count_nonzero(np.asarray(c)))
                                for c in u.variations.tucker_variations + u.variations.tt_variations))
        w = t3m.T3Tangent.randn(base, stack_shape=(3,), apply_gauge_projection=False)
        zl = t3m.T3Tangent.zeros_like(w)
        self.assertEqual((3,), zl.tangent_stack_shape)
        self.assertEqual(0.0, float(np.max(np.abs(zl.norm()))))
        self.assertEqual((3,), t3m.T3Tangent.randn_like(w).tangent_stack_shape)

    def test_to_from_vector(self):
        # T3Tangent.to_vector (variation DOF only) / from_vector round-trip.
        base = bvf.T3Basis.random_orthogonal((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        v = t3m.T3Tangent.randn(base, stack_shape=(3,), apply_gauge_projection=False)
        flat = v.to_vector()
        self.assertEqual((v.variations.data_size,), flat.shape)   # variation DOF; basis excluded
        v2 = t3m.T3Tangent.from_vector(flat, base, tangent_stack_shape=(3,))
        self.assertEqual(0.0, cw.corewise_relerr(v.variations.data, v2.variations.data))

    def test_save_load(self):
        import tempfile, os
        base = bvf.T3Basis.random_orthogonal((5, 6, 4), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
        v = t3m.T3Tangent.randn(base, stack_shape=(3,), apply_gauge_projection=False)
        f = os.path.join(tempfile.mkdtemp(), 't.npz'); v.save(f)
        v2 = t3m.T3Tangent.load(f)
        self.assertEqual(0.0, cw.corewise_relerr(v.variations.data, v2.variations.data))
        self.assertEqual(0.0, cw.corewise_relerr(v.basis.data, v2.basis.data))

    def test_reverse(self):
        # T3Tangent.reverse commutes with to_dense (mode axes reversed); reverse is an involution.
        STRUCT = ((5, 6, 4), (2, 3, 2), (1, 2, 2, 1)); d = 3
        for C in [(), (2,)]:
            for K in [(), (3,)]:
                base = bvf.T3Basis.random_orthogonal(*STRUCT, stack_shape=C)
                v = t3m.T3Tangent.randn(base, stack_shape=K, apply_gauge_projection=False)
                D = np.asarray(v.to_dense()); ns = D.ndim - d
                perm = tuple(range(ns)) + tuple(range(D.ndim - 1, ns - 1, -1))
                self.check_relerr(D.transpose(perm), np.asarray(v.reverse().to_dense()))
                self.check_relerr(D, np.asarray(v.reverse().reverse().to_dense()))

    def test_sum_tangents(self):
        # Summing over the tangent stack K commutes with to_dense (= the tensor sum, by linearity).
        base = bvf.T3Basis.random_orthogonal((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        v = t3m.T3Tangent.randn(base, stack_shape=(3,), apply_gauge_projection=False)
        self.check_relerr(np.sum(np.asarray(v.to_dense()), axis=0), np.asarray(v.sum_tangents().to_dense()))
        self.assertEqual((), v.sum_tangents().tangent_stack_shape)

    def test_manifold_dim(self):
        self.assertEqual(578, t3m.manifold_dim(((15, 16, 13), (9, 10, 8), (2, 7, 6, 3))))
        self.assertEqual(29, t3m.manifold_dim(((5, 6, 3), (5, 3, 2), (2, 2, 4, 1))))

    def test_manifold_dim_via_svd(self):
        # The dimension of the tangent space = number of nonzero singular values of a sufficient
        # collection of dense tangent vectors. Uses a minimal-rank base.
        shape, tucker_ranks, tt_ranks = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        s = (shape, tucker_ranks, tt_ranks)
        mdim = t3m.manifold_dim(s)
        x = t3.TuckerTensorTrain.randn(*s)
        self.assertTrue(x.has_minimal_ranks)
        base, _ = bvf.t3_orthogonal_representations(x)
        dense_tangents = np.stack([
            t3m.T3Tangent(base, _random_variations(base)).to_dense().reshape(-1)
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
                base, var = bvf.t3_orthogonal_representations(x)
                v = t3m.T3Tangent(base, var)

                U0, U1, U2 = (np.asarray(c) for c in base.up_tucker_cores)
                D0, D1, D2 = (np.asarray(c) for c in base.down_tt_cores)
                L0, L1, L2 = (np.asarray(c) for c in base.left_tt_cores)
                R0, R1, R2 = (np.asarray(c) for c in base.right_tt_cores)
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
                    base, _ = bvf.t3_orthogonal_representations(x)
                    v1 = t3m.T3Tangent(base, _random_variations(base))
                    v2 = t3m.T3Tangent(base, _random_variations(base))

                    self.check_relerr(v1.to_dense() + v2.to_dense(), (v1 + v2).to_dense())
                    self.check_relerr(v1.to_dense() - v2.to_dense(), (v1 - v2).to_dense())
                    self.check_relerr(2.5 * np.asarray(v1.to_dense()), (2.5 * v1).to_dense())
                    self.check_relerr(-np.asarray(v1.to_dense()), (-v1).to_dense())
                    self.assertLessEqual(norm(np.asarray(t3m.T3Tangent.zeros(base).to_dense())), tol)

    def test_same_tangent_space_guard(self):
        x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        base_a, _ = bvf.t3_orthogonal_representations(x)
        base_b, _ = bvf.t3_orthogonal_representations(x)  # numerically equal cores, different object
        va = t3m.T3Tangent(base_a, _random_variations(base_a))
        vb = t3m.T3Tangent(base_b, _random_variations(base_b))
        va2 = t3m.T3Tangent(base_a, _random_variations(base_a))

        t3m.T3Tangent(base_a, va.variations) + va2  # same basis object: OK
        for op in (lambda: va + vb, lambda: va - vb, lambda: va.inner(vb)):
            with self.assertRaises(ValueError):
                op()

    def test_inner_norm(self):
        x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        base, _ = bvf.t3_orthogonal_representations(x)
        v1 = t3m.T3Tangent(base, _random_variations(base))
        v2 = t3m.T3Tangent(base, _random_variations(base))

        # structural identities (corewise) hold regardless of gauge (unstacked -> scalar)
        self.assertAlmostEqual(float(v1.inner(v2)), float(cw.corewise_dot(v1.variations.data, v2.variations.data)))
        self.assertAlmostEqual(float(v1.norm()), float(np.sqrt(v1.inner(v1))))

    def test_is_orthogonal_and_is_gauged(self):
        x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        base, var = bvf.t3_orthogonal_representations(x)

        # base from t3_orthogonal_representations is orthogonal; its variations are not gauged
        self.assertTrue(t3m.T3Tangent(base, var).is_orthogonal())
        self.assertFalse(t3m.T3Tangent(base, var).is_gauged())

        # the zero tangent is trivially gauged (all variation cores are zero)
        self.assertTrue(t3m.T3Tangent.zeros(base).is_gauged())

    # ``stack_shapes`` for the two-axis stacking tests: (base_stack C, tangent_stack K) pairs.
    bv_stack_shapes = [((), (3,)), ((2,), (3,)), ((2,), ()), ((2,), (2, 2)), ((2, 3), (2,))]

    def _random_v_stacked(self, struct, base_stack, V):
        x = t3.TuckerTensorTrain.randn(*struct, stack_shape=base_stack)
        base, _ = bvf.t3_orthogonal_representations(x)
        return t3m.T3Tangent.randn(base, stack_shape=V, apply_gauge_projection=False)

    def test_unstack_stack_tangents(self):
        # unstack_tangents peels the tangent stack K -> a K-shaped tree of tangents that SHARE the
        # base (same T3Basis object, one tangent space). stack_tangents inverts it.
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for BASE_STACK, V in self.bv_stack_shapes:
            with self.subTest(BASE_STACK=BASE_STACK, V=V):
                v = self._random_v_stacked(STRUCT, BASE_STACK, V)
                dense = np.asarray(v.to_dense())  # K + C + (N...)
                tree = v.unstack_tangents()

                for vidx in np.ndindex(*V):
                    leaf = _tree_get(tree, vidx)
                    self.assertIs(leaf.basis, v.basis)  # shared base object
                    self.assertEqual((), leaf.tangent_stack_shape)
                    self.assertEqual(BASE_STACK, leaf.base_stack_shape)
                    self.check_relerr(dense[vidx], leaf.to_dense())  # slice the leading V axes

                rt = t3m.T3Tangent.stack_tangents(tree)  # round-trip
                self.assertIs(rt.basis, v.basis)
                self.assertEqual(V, rt.tangent_stack_shape)
                self.check_relerr(dense, rt.to_dense())

    def test_unstack_stack_basis(self):
        # unstack_basis peels the base stack C -> a C-shaped tree of single-base-point tangents (each
        # at a DIFFERENT base point, still carrying its K batch). stack_basis inverts it.
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for BASE_STACK, V in self.bv_stack_shapes:
            with self.subTest(BASE_STACK=BASE_STACK, V=V):
                v = self._random_v_stacked(STRUCT, BASE_STACK, V)
                dense = np.asarray(v.to_dense())  # K + C + (N...)
                nV = len(V)
                tree = v.unstack_basis()

                for gidx in np.ndindex(*BASE_STACK):
                    leaf = _tree_get(tree, gidx)
                    self.assertEqual((), leaf.base_stack_shape)
                    self.assertEqual(V, leaf.tangent_stack_shape)
                    ref = dense[(slice(None),) * nV + gidx]  # slice the interior C axes
                    self.check_relerr(ref, leaf.to_dense())

                rt = t3m.T3Tangent.stack_basis(tree)  # round-trip
                self.assertEqual(BASE_STACK, rt.base_stack_shape)
                self.assertEqual(V, rt.tangent_stack_shape)
                self.check_relerr(dense, rt.to_dense())

    def test_stack_tangents_guard(self):
        # stack_tangents requires a shared T3Basis object (same tangent space); different base
        # objects (even with numerically equal cores) raise.
        x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        base_a, _ = bvf.t3_orthogonal_representations(x)
        base_b, _ = bvf.t3_orthogonal_representations(x)  # equal cores, different object
        ta = t3m.T3Tangent.randn(base_a, apply_gauge_projection=False)
        tb = t3m.T3Tangent.randn(base_b, apply_gauge_projection=False)
        ta2 = t3m.T3Tangent.randn(base_a, apply_gauge_projection=False)

        t3m.T3Tangent.stack_tangents([ta, ta2])  # same basis object: OK
        with self.assertRaises(ValueError):
            t3m.T3Tangent.stack_tangents([ta, tb])

    def test_orthogonal_gauge_projection(self):
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    u = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
                    ug = u.orthogonal_gauge_projection()

                    self.assertTrue(ug.is_gauged())
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
                    base, _ = bvf.t3_orthogonal_representations(x)
                    u = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
                    uo = u.oblique_gauge_projection()

                    self.assertTrue(uo.is_gauged())
                    self.check_relerr(u.to_dense(), uo.to_dense())  # preserves the tangent vector

    def test_inner_norm_faithfulness(self):
        # orthogonal + minimal-rank base + gauged variations => corewise inner/norm == Hilbert-Schmidt.
        # inner/norm vectorize over the stack, returning a STACK_SHAPE-shaped array (one value/slice).
        for STACK_SHAPE in [(), (2,), (2, 3)]:
            with self.subTest(STACK_SHAPE=STACK_SHAPE):
                x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                base, _ = bvf.t3_orthogonal_representations(x)
                self.assertTrue(base.is_orthogonal() and base.has_minimal_ranks)

                u = t3m.T3Tangent.randn(base)  # gauged by default
                w = t3m.T3Tangent.randn(base)

                ud, wd = np.asarray(u.to_dense()), np.asarray(w.to_dense())
                tensor_axes = tuple(range(len(STACK_SHAPE), ud.ndim))  # the (N0..Nd) axes; keep stack
                hs_inner = np.sum(ud * wd, axis=tensor_axes)  # shape = STACK_SHAPE
                hs_norm = np.sqrt(np.sum(ud * ud, axis=tensor_axes))
                self.assertLessEqual(norm(np.asarray(u.inner(w)) - hs_inner), tol * max(1.0, norm(hs_inner)))
                self.assertLessEqual(norm(np.asarray(u.norm()) - hs_norm), tol * max(1.0, norm(hs_norm)))

    def test_inner_norm_tangent_stacked(self):
        # A T3Tangent may carry an extra OUTER tangent stack K (a batch of tangents sharing one base);
        # inner/norm vectorize over the full K + C stack.
        for BASE_STACK, V in [((), (3,)), ((2,), (3,)), ((2,), ())]:
            with self.subTest(BASE_STACK=BASE_STACK, V=V):
                x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=BASE_STACK)
                base, _ = bvf.t3_orthogonal_representations(x)
                u = t3m.T3Tangent.randn(base, stack_shape=V, apply_gauge_projection=False)
                w = t3m.T3Tangent.randn(base, stack_shape=V, apply_gauge_projection=False)

                self.assertEqual(V, u.tangent_stack_shape)
                self.assertEqual(BASE_STACK, u.base_stack_shape)
                self.assertEqual(V + BASE_STACK, u.stack_shape)

                full = V + BASE_STACK
                ip = np.asarray(u.inner(w))
                self.assertEqual(full, ip.shape)

                ref = np.zeros(full)  # per-slice corewise dot over the full stack
                for idx in np.ndindex(*full):
                    ud = (tuple(np.asarray(c)[idx] for c in u.variations.tucker_variations),
                          tuple(np.asarray(c)[idx] for c in u.variations.tt_variations))
                    wd = (tuple(np.asarray(c)[idx] for c in w.variations.tucker_variations),
                          tuple(np.asarray(c)[idx] for c in w.variations.tt_variations))
                    ref[idx] = float(cw.corewise_dot(ud, wd))
                self.assertLessEqual(norm(ip - ref), tol * max(1.0, norm(ref)))
                self.assertLessEqual(norm(np.asarray(u.norm()) - np.sqrt(np.abs(np.asarray(u.inner(u))))), tol)

    def test_tangent_probe(self):
        # forward J^(s): v.probe(ww) == probe_dense(ww, v.to_dense()); probes are stacked W + K + C + (N,)
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for BASE_STACK in [(), (2,)]:
            for PROBE_STACK in [(), (2,)]:
                for TANGENT_STACK in [(), (2,)]:
                    with self.subTest(BASE=BASE_STACK, PROBE=PROBE_STACK, TANGENT=TANGENT_STACK):
                        rnd = np.random.randn
                        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=BASE_STACK)
                        base, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.T3Tangent.randn(base, stack_shape=TANGENT_STACK, apply_gauge_projection=False)
                        ww = tuple(rnd(*(PROBE_STACK + (N,))) for N in STRUCT[0])

                        zz = v.probe(ww)  # numpy/jax inferred from inputs
                        zz2 = t3p.probe_dense(ww, v.to_dense())  # dense ground truth, also W + K + C
                        self.assertEqual(
                            PROBE_STACK + TANGENT_STACK + BASE_STACK + (STRUCT[0][0],),
                            tuple(np.asarray(zz[0]).shape),
                        )
                        for a, b in zip(zz, zz2):
                            self.check_relerr(b, a)

                        # K-stack contract: probing the batch == stacking the per-tangent probes
                        # (each single tangent shares the base), inserted at the V axis (after F).
                        if TANGENT_STACK != ():
                            per = [leaf.probe(ww) for leaf in v.unstack_tangents()]
                            for i in range(len(STRUCT[0])):
                                stacked = np.stack([np.asarray(p[i]) for p in per], axis=len(PROBE_STACK))
                                self.check_relerr(stacked, np.asarray(zz[i]))

    def test_tangent_probe_transpose(self):
        # adjoint identity <z, J v> = <J^T z, v>; J^T accepts K-stacked residuals (W + K + C).
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for BASE_STACK in [(), (2,)]:
            for PROBE_STACK in [(), (2,)]:
                for TANGENT_STACK in [(), (2,)]:
                    with self.subTest(BASE=BASE_STACK, PROBE=PROBE_STACK, TANGENT=TANGENT_STACK):
                        rnd = np.random.randn
                        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=BASE_STACK)
                        base, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.T3Tangent.randn(base, stack_shape=TANGENT_STACK, apply_gauge_projection=False)
                        ww = tuple(rnd(*(PROBE_STACK + (N,))) for N in STRUCT[0])
                        # residuals live in the forward probe space: W + K + C + (N,)
                        z = tuple(rnd(*(PROBE_STACK + TANGENT_STACK + BASE_STACK + (N,))) for N in STRUCT[0])

                        Jv = v.probe(ww)  # numpy/jax inferred from inputs
                        JTz = t3m.T3Tangent.probe_transpose(z, ww, base, sum_over_probes=True)
                        # full contraction of both sides (sums F, V, G, N) must agree
                        lhs = float(np.sum([np.sum(np.asarray(a) * np.asarray(b)) for a, b in zip(z, Jv)]))
                        rhs = float(np.sum(np.asarray(JTz.inner(v))))
                        self.assertLessEqual(abs(lhs - rhs), tol * max(1.0, abs(lhs)))

                        # sum=True keeps V (base G), drops F; sum=False keeps F+V (base G)
                        self.assertEqual(TANGENT_STACK, JTz.tangent_stack_shape)
                        self.assertEqual(BASE_STACK, JTz.base_stack_shape)
                        JTz_batch = t3m.T3Tangent.probe_transpose(z, ww, base)  # sum_over_probes=False
                        self.assertEqual(PROBE_STACK + TANGENT_STACK, JTz_batch.tangent_stack_shape)
                        self.assertEqual(BASE_STACK, JTz_batch.base_stack_shape)

                        # sum=True == sum over the probe stack W of sum=False (validates sum=False)
                        f_axes = tuple(range(len(PROBE_STACK)))
                        for cs, cn in zip(JTz.variations.tucker_variations, JTz_batch.variations.tucker_variations):
                            self.check_relerr(np.asarray(cs), np.asarray(cn).sum(axis=f_axes))
                        for cs, cn in zip(JTz.variations.tt_variations, JTz_batch.variations.tt_variations):
                            self.check_relerr(np.asarray(cs), np.asarray(cn).sum(axis=f_axes))

    def test_tangent_apply(self):
        # apply(v, ww) contracts the dense tangent in ALL modes; result is stacked W + K + C.
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for BASE_STACK in [(), (2,)]:
            for W in [(), (2,)]:           # apply-vector stack
                for K in [(), (3,)]:       # tangent stack
                    with self.subTest(BASE=BASE_STACK, W=W, K=K):
                        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=BASE_STACK)
                        base, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.T3Tangent.randn(base, stack_shape=K, apply_gauge_projection=False)
                        ww = tuple(np.random.randn(*(W + (N,))) for N in STRUCT[0])

                        a = np.asarray(v.apply(ww))
                        self.assertEqual(W + K + BASE_STACK, a.shape)
                        vd = np.asarray(v.to_dense())   # K + C + (N0,N1,N2)
                        if W:   # shared apply-vector stack w across all modes
                            ref = np.einsum('...ijk,wi,wj,wk->w...', vd, ww[0], ww[1], ww[2])
                        else:
                            ref = np.einsum('...ijk,i,j,k->...', vd, ww[0], ww[1], ww[2])
                        self.check_relerr(ref, a)

    def test_tangent_entries(self):
        # entries(v, idx) extracts entries of the dense tangent (no contraction); result W + K + C.
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for BASE_STACK in [(), (2,)]:
            for W in [(), (2,)]:           # index stack
                for K in [(), (3,)]:
                    with self.subTest(BASE=BASE_STACK, W=W, K=K):
                        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=BASE_STACK)
                        base, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.T3Tangent.randn(base, stack_shape=K, apply_gauge_projection=False)
                        idx = np.array(tuple(np.random.randint(0, N, size=W) for N in STRUCT[0]))  # (d,)+W

                        e = np.asarray(v.entries(idx))
                        self.assertEqual(W + K + BASE_STACK, e.shape)
                        vd = np.asarray(v.to_dense())
                        if W:
                            ref = np.stack([vd[..., idx[0, w], idx[1, w], idx[2, w]] for w in range(W[0])], axis=0)
                        else:
                            ref = vd[..., int(idx[0]), int(idx[1]), int(idx[2])]
                        self.check_relerr(ref, e)

    def test_tangent_apply_transpose(self):
        # adjoint identity <apply^T c, v> == sum_W c*apply(v) (keep base C); keep-W -> tangent stack W.
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for BASE_STACK in [(), (2,)]:
            for W in [(), (2,)]:
                with self.subTest(BASE=BASE_STACK, W=W):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=BASE_STACK)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    v = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
                    ww = tuple(np.random.randn(*(W + (N,))) for N in STRUCT[0])
                    c = np.asarray(np.random.randn(*(W + BASE_STACK)))

                    ATc = t3m.T3Tangent.apply_transpose(c, ww, base, sum_over_probes=True)
                    lhs = np.asarray(cw.corewise_stack_dot(ATc.variations.data, v.variations.data, len(BASE_STACK)))
                    rhs = np.sum(c * np.asarray(v.apply(ww)), axis=tuple(range(len(W))))   # sum_W, keep C
                    self.check_relerr(rhs, lhs)
                    self.assertEqual((), ATc.tangent_stack_shape)               # summed over W
                    self.assertEqual(BASE_STACK, ATc.base_stack_shape)

                    # without summing, W becomes the tangent stack; sum=True == sum_W of sum=False
                    ATc_keep = t3m.T3Tangent.apply_transpose(c, ww, base)
                    self.assertEqual(W, ATc_keep.tangent_stack_shape)
                    self.assertEqual(BASE_STACK, ATc_keep.base_stack_shape)
                    f_axes = tuple(range(len(W)))
                    for cs, cn in zip(ATc.variations.tucker_variations, ATc_keep.variations.tucker_variations):
                        self.check_relerr(np.asarray(cs), np.asarray(cn).sum(axis=f_axes))
                    for cs, cn in zip(ATc.variations.tt_variations, ATc_keep.variations.tt_variations):
                        self.check_relerr(np.asarray(cs), np.asarray(cn).sum(axis=f_axes))

    def test_tangent_entries_transpose(self):
        # adjoint identity <entries^T c, v> == sum_W c*entries(v, idx) (keep base C).
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for BASE_STACK in [(), (2,)]:
            for W in [(), (2,)]:
                with self.subTest(BASE=BASE_STACK, W=W):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=BASE_STACK)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    v = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
                    idx = np.array(tuple(np.random.randint(0, N, size=W) for N in STRUCT[0]))  # (d,)+W
                    c = np.asarray(np.random.randn(*(W + BASE_STACK)))

                    ETc = t3m.T3Tangent.entries_transpose(c, idx, base, sum_over_probes=True)
                    lhs = np.asarray(cw.corewise_stack_dot(ETc.variations.data, v.variations.data, len(BASE_STACK)))
                    rhs = np.sum(c * np.asarray(v.entries(idx)), axis=tuple(range(len(W))))
                    self.check_relerr(rhs, lhs)
                    self.assertEqual(W, t3m.T3Tangent.entries_transpose(c, idx, base).tangent_stack_shape)

    def test_randn(self):
        base, _ = bvf.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1)))
        self.assertTrue(t3m.T3Tangent.randn(base).is_gauged())                                   # gauged by default
        self.assertFalse(t3m.T3Tangent.randn(base, apply_gauge_projection=False).is_gauged())    # ungauged on request
        self.assertEqual(base.stack_shape, t3m.T3Tangent.randn(base).stack_shape)                # construction validates fit

    def test_to_t3(self):
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    v = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
                    base_point = t3.TuckerTensorTrain(base.up_tucker_cores, base.left_tt_cores).to_dense()

                    self.check_relerr(v.to_dense(), v.to_t3().to_dense())
                    self.check_relerr(np.asarray(base_point) + np.asarray(v.to_dense()),
                                      v.to_t3(include_shift=True).to_dense())

    def test_retract(self):
        # Dense correctness: retract(0) == base point; retract(v) == best rank-(base) T3-SVD of (base point + v).
        for T3_STRUCTURE in [((10, 11), (3, 4), (1, 2, 1)), ((9, 10, 11, 12), (2, 3, 3, 2), (1, 2, 3, 2, 1))]:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    base_point = t3.TuckerTensorTrain(base.up_tucker_cores, base.left_tt_cores).to_dense()

                    self.check_relerr(base_point, t3m.T3Tangent.zeros(base).retract().to_dense())

                    v = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
                    if STACK_SHAPE == ():  # compare against a from-dense T3-SVD of (base point + v)
                        shifted_dense = np.asarray(base_point) + np.asarray(v.to_dense())
                        ref, _, _ = t3.TuckerTensorTrain.t3svd_dense(
                            shifted_dense, max_tucker_ranks=tuple(base.up_ranks), max_tt_ranks=tuple(base.left_ranks))
                        self.check_relerr(ref.to_dense(), v.retract().to_dense())

        # Rank preservation holds on a minimal-rank base.
        for STACK_SHAPE in [(), (2,)]:
            with self.subTest(rank_preservation=True, STACK_SHAPE=STACK_SHAPE):
                x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                base, _ = bvf.t3_orthogonal_representations(x)
                self.assertTrue(base.has_minimal_ranks)
                r = t3m.T3Tangent.randn(base, apply_gauge_projection=False).retract()
                self.assertEqual(tuple(base.up_ranks), r.tucker_ranks)
                self.assertEqual(tuple(base.left_ranks), r.tt_ranks)

    def test_project(self):
        # project(x, base) is the orthogonal projection of x onto the tangent space T_P M.
        for STR_P, STR_X in [
            (((6, 7, 5), (2, 2, 2), (1, 2, 2, 1)), ((6, 7, 5), (3, 4, 3), (1, 2, 2, 1))),
            (((9, 10, 11, 12), (2, 3, 3, 2), (1, 2, 3, 2, 1)), ((9, 10, 11, 12), (3, 3, 4, 2), (1, 2, 2, 2, 1))),
        ]:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(STR_P=STR_P, STACK_SHAPE=STACK_SHAPE):
                    p = t3.TuckerTensorTrain.randn(*STR_P, stack_shape=STACK_SHAPE)
                    x = t3.TuckerTensorTrain.randn(*STR_X, stack_shape=STACK_SHAPE)
                    base, _ = bvf.t3_orthogonal_representations(p)

                    proj = t3m.T3Tangent.project(x, base)
                    self.assertTrue(proj.is_gauged())

                    # idempotency: projecting a tangent vector (its unshifted embedding) recovers it
                    v = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
                    proj_v = t3m.T3Tangent.project(v.to_t3(), base)
                    self.check_relerr(v.to_dense(), proj_v.to_dense())

                    # orthogonality: the residual x - proj_x is perpendicular to the tangent space
                    residual = np.asarray(x.to_dense()) - np.asarray(proj.to_dense())
                    tensor_axes = tuple(range(len(STACK_SHAPE), residual.ndim))
                    for _ in range(3):
                        w_dense = np.asarray(
                            t3m.T3Tangent.randn(base, apply_gauge_projection=False).to_dense())
                        ip = norm(np.sum(residual * w_dense, axis=tensor_axes))
                        self.assertLessEqual(float(ip), tol * norm(residual) * norm(w_dense))

    def test_tangent_stacked_heavy_ops(self):
        # A K-stacked tangent is a batch of tangent vectors sharing one base (one per (v, g) pair).
        # to_dense/to_t3/retract produce a K+C-stacked result whose every slice matches the
        # corresponding unstacked tangent (the shared base point replicated across the tangent stack K).
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))  # minimal-rank, so retract preserves ranks
        for BASE_STACK, V in [((), (3,)), ((2,), (3,)), ((2,), ()), ((2,), (2, 2))]:
            with self.subTest(BASE_STACK=BASE_STACK, V=V):
                x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=BASE_STACK)
                base, _ = bvf.t3_orthogonal_representations(x)
                var = t3m.T3Tangent.randn(base, stack_shape=V, apply_gauge_projection=False).variations
                v = t3m.T3Tangent(base, var)
                full = V + BASE_STACK
                n_base = len(BASE_STACK)

                dense = np.asarray(v.to_dense())
                t3_dense = np.asarray(v.to_t3().to_dense())
                shifted = np.asarray(v.to_t3(include_shift=True).to_dense())
                retr = np.asarray(v.retract().to_dense())
                self.assertEqual(full + STRUCT[0], dense.shape)

                for idx in np.ndindex(*full):
                    s = _slice_tangent(base, var, idx, n_base)
                    self.check_relerr(s.to_dense(), dense[idx])
                    self.check_relerr(s.to_dense(), t3_dense[idx])  # to_t3 round-trips to to_dense
                    self.check_relerr(s.to_t3(include_shift=True).to_dense(), shifted[idx])
                    self.check_relerr(s.retract().to_dense(), retr[idx])

    def test_project_tangent_stacked(self):
        # project a BATCH of inputs x (stack K+C) onto a base (stack G): the result is a K-stacked
        # tangent whose (v, g) slice equals projecting x[v, g] onto the shared base point base[g].
        STR_P = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        STR_X = ((6, 7, 5), (3, 4, 3), (1, 2, 2, 1))
        for BASE_STACK, V in [((), (3,)), ((2,), (3,))]:
            with self.subTest(BASE_STACK=BASE_STACK, V=V):
                p = t3.TuckerTensorTrain.randn(*STR_P, stack_shape=BASE_STACK)
                x = t3.TuckerTensorTrain.randn(*STR_X, stack_shape=(V + BASE_STACK))
                base, _ = bvf.t3_orthogonal_representations(p)

                proj = t3m.T3Tangent.project(x, base)
                self.assertEqual(V, proj.tangent_stack_shape)
                self.assertEqual(BASE_STACK, proj.base_stack_shape)
                self.assertTrue(proj.is_gauged())

                proj_dense = np.asarray(proj.to_dense())  # K + C + (N...)
                full = V + BASE_STACK
                n_base = len(BASE_STACK)
                for idx in np.ndindex(*full):
                    g_idx = idx[len(idx) - n_base:] if n_base > 0 else ()
                    ref = t3m.T3Tangent.project(_slice_t3(x, idx), _slice_basis(base, g_idx))
                    self.check_relerr(ref.to_dense(), proj_dense[idx])

    def test_normalized(self):
        # normalized() rescales each stacked tangent to unit norm (HS norm, since randn is gauged).
        for STACK_SHAPE in [(), (2,)]:
            for V in [(), (3,)]:
                with self.subTest(STACK_SHAPE=STACK_SHAPE, V=V):
                    x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    vn = t3m.T3Tangent.randn(base, stack_shape=V).normalized()
                    vn.validate()
                    self.assertLessEqual(norm(np.asarray(vn.norm()) - 1.0), tol)

    def test_allclose(self):
        # T3Tangent.allclose compares two tangents at the same base point.
        for STACK_SHAPE in [(), (2,)]:
            with self.subTest(STACK_SHAPE=STACK_SHAPE):
                x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                base, _ = bvf.t3_orthogonal_representations(x)
                v = t3m.T3Tangent.randn(base)
                self.assertTrue(v.allclose(v))
                self.assertFalse(v.allclose(v * 2.0))
                self.assertTrue(v.allclose(v * (1.0 + 1e-12)))


if __name__ == "__main__":
    unittest.main()
