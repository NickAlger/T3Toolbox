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

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jnp = np

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm


def _random_variations(base, use_jax=False):
    """A random T3Variations fitting the holes of ``base`` (ungauged)."""
    rnd = (lambda *s: jnp.array(np.random.randn(*s))) if use_jax else (lambda *s: np.random.randn(*s))
    ss = base.stack_shape
    tucker_hole_shapes, tt_hole_shapes = base.variation_shapes
    V = tuple(rnd(*(ss + s)) for s in tucker_hole_shapes)
    H = tuple(rnd(*(ss + s)) for s in tt_hole_shapes)
    return bvf.T3Variations(V, H)


def _random_tangent(t3_structure, stack_shape=(), use_jax=False):
    x = t3.TuckerTensorTrain.randn(*t3_structure, stack_shape=stack_shape)
    if use_jax:
        x = x.to_jax()
    base, _ = bvf.t3_orthogonal_representations(x)
    return t3m.T3Tangent(base, _random_variations(base, use_jax=use_jax))


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
            for USE_JAX in [False, True]:
                with self.subTest(STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                    x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                    if USE_JAX:
                        x = x.to_jax()
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
                for USE_JAX in [False, True]:
                    with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                        x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                        if USE_JAX:
                            x = x.to_jax()
                        base, _ = bvf.t3_orthogonal_representations(x)
                        v1 = t3m.T3Tangent(base, _random_variations(base, use_jax=USE_JAX))
                        v2 = t3m.T3Tangent(base, _random_variations(base, use_jax=USE_JAX))

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

    def test_stack_unstack(self):
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in [(2,), (2, 3)]:
                for USE_JAX in [False, True]:
                    with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                        v = _random_tangent(T3_STRUCTURE, stack_shape=STACK_SHAPE, use_jax=USE_JAX)
                        v2 = t3m.T3Tangent.stack(v.unstack())
                        err = cw.corewise_norm(cw.corewise_sub(v.variations.data, v2.variations.data))
                        self.assertLessEqual(float(err), tol)
                        # a leaf of the unstacked tangent matches its dense slice
                        leaf = v.unstack()
                        idx = tuple(0 for _ in STACK_SHAPE)
                        for k in idx:
                            leaf = leaf[k]
                        self.check_relerr(np.asarray(v.to_dense())[idx], leaf.to_dense())

    def test_orthogonal_gauge_projection(self):
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in [(), (2,)]:
                for USE_JAX in [False, True]:
                    with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                        x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                        if USE_JAX:
                            x = x.to_jax()
                        base, _ = bvf.t3_orthogonal_representations(x)
                        u = t3m.T3Tangent.randn(base, apply_gauge_projection=False, use_jax=USE_JAX)
                        ug = u.orthogonal_gauge_projection(use_jax=USE_JAX)

                        self.assertTrue(ug.is_gauged())
                        # orthogonal projection: the removed component is perpendicular to the projection
                        residual_dot_proj = cw.corewise_dot(
                            cw.corewise_sub(u.variations.data, ug.variations.data), ug.variations.data, use_jax=USE_JAX)
                        scale = float(cw.corewise_dot(ug.variations.data, ug.variations.data, use_jax=USE_JAX))
                        self.assertLessEqual(abs(float(residual_dot_proj)), tol * max(1.0, scale))

    def test_oblique_gauge_projection(self):
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in [(), (2,)]:
                for USE_JAX in [False, True]:
                    with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                        x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                        if USE_JAX:
                            x = x.to_jax()
                        base, _ = bvf.t3_orthogonal_representations(x)
                        u = t3m.T3Tangent.randn(base, apply_gauge_projection=False, use_jax=USE_JAX)
                        uo = u.oblique_gauge_projection(use_jax=USE_JAX)

                        self.assertTrue(uo.is_gauged())
                        self.check_relerr(u.to_dense(), uo.to_dense())  # preserves the tangent vector

    def test_inner_norm_faithfulness(self):
        # orthogonal + minimal-rank base + gauged variations => corewise inner/norm == Hilbert-Schmidt.
        # inner/norm vectorize over the stack, returning a STACK_SHAPE-shaped array (one value/slice).
        for STACK_SHAPE in [(), (2,), (2, 3)]:
            for USE_JAX in [False, True]:
                with self.subTest(STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                    x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                    if USE_JAX:
                        x = x.to_jax()
                    base, _ = bvf.t3_orthogonal_representations(x)
                    self.assertTrue(base.is_orthogonal() and base.has_minimal_ranks)

                    u = t3m.T3Tangent.randn(base, use_jax=USE_JAX)  # gauged by default
                    w = t3m.T3Tangent.randn(base, use_jax=USE_JAX)

                    ud, wd = np.asarray(u.to_dense()), np.asarray(w.to_dense())
                    tensor_axes = tuple(range(len(STACK_SHAPE), ud.ndim))  # the (N0..Nd) axes; keep stack
                    hs_inner = np.sum(ud * wd, axis=tensor_axes)  # shape = STACK_SHAPE
                    hs_norm = np.sqrt(np.sum(ud * ud, axis=tensor_axes))
                    self.assertLessEqual(norm(np.asarray(u.inner(w)) - hs_inner), tol * max(1.0, norm(hs_inner)))
                    self.assertLessEqual(norm(np.asarray(u.norm()) - hs_norm), tol * max(1.0, norm(hs_norm)))

    def test_inner_norm_tangent_stacked(self):
        # A T3Tangent may carry an extra OUTER tangent stack V (a batch of tangents sharing one base);
        # inner/norm vectorize over the full V + G stack.
        for BASE_STACK, V in [((), (3,)), ((2,), (3,)), ((2,), ())]:
            for USE_JAX in [False, True]:
                with self.subTest(BASE_STACK=BASE_STACK, V=V, USE_JAX=USE_JAX):
                    x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=BASE_STACK)
                    if USE_JAX:
                        x = x.to_jax()
                    base, _ = bvf.t3_orthogonal_representations(x)
                    u = t3m.T3Tangent.randn(base, stack_shape=V, apply_gauge_projection=False, use_jax=USE_JAX)
                    w = t3m.T3Tangent.randn(base, stack_shape=V, apply_gauge_projection=False, use_jax=USE_JAX)

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
        # forward J^(s): v.probe(ww) == probe_dense(ww, v.to_dense()); probes are stacked F + G + (N,)
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for BASE_STACK in [(), (2,)]:
            for PROBE_STACK in [(), (2,)]:
                for USE_JAX in [False, True]:
                    with self.subTest(BASE_STACK=BASE_STACK, PROBE_STACK=PROBE_STACK, USE_JAX=USE_JAX):
                        rnd = (lambda *s: jnp.array(np.random.randn(*s))) if USE_JAX else np.random.randn
                        v = _random_tangent(STRUCT, stack_shape=BASE_STACK, use_jax=USE_JAX)
                        ww = tuple(rnd(*(PROBE_STACK + (N,))) for N in STRUCT[0])
                        zz = v.probe(ww)  # numpy/jax inferred from inputs (jax when USE_JAX)
                        zz2 = t3p.probe_dense(ww, v.to_dense(use_jax=USE_JAX))
                        self.assertEqual(PROBE_STACK + BASE_STACK + (STRUCT[0][0],), tuple(np.asarray(zz[0]).shape))
                        for a, b in zip(zz, zz2):
                            self.check_relerr(b, a)

    def test_tangent_probe_transpose(self):
        # adjoint identity <z, J v> = <J^T z, v> (sum over probes); non-sum gives a V-stacked tangent
        STRUCT = ((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        for BASE_STACK in [(), (2,)]:
            for PROBE_STACK in [(), (2,)]:
                for USE_JAX in [False, True]:
                    with self.subTest(BASE_STACK=BASE_STACK, PROBE_STACK=PROBE_STACK, USE_JAX=USE_JAX):
                        rnd = (lambda *s: jnp.array(np.random.randn(*s))) if USE_JAX else np.random.randn
                        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=BASE_STACK)
                        if USE_JAX:
                            x = x.to_jax()
                        base, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.T3Tangent.randn(base, use_jax=USE_JAX)
                        ww = tuple(rnd(*(PROBE_STACK + (N,))) for N in STRUCT[0])
                        z = tuple(rnd(*(PROBE_STACK + BASE_STACK + (N,))) for N in STRUCT[0])  # F + G + (N,)

                        Jv = v.probe(ww)  # numpy/jax inferred from inputs (jax when USE_JAX)
                        JTz = t3m.T3Tangent.probe_transpose(z, ww, base, sum_over_probes=True)
                        # <z, Jv> sums over F, G, N; <J^T z, v> = sum over G of JTz.inner(v) (which keeps G)
                        lhs = float(np.sum([np.sum(np.asarray(a) * np.asarray(b)) for a, b in zip(z, Jv)]))
                        rhs = float(np.sum(np.asarray(JTz.inner(v, use_jax=USE_JAX))))
                        self.assertLessEqual(abs(lhs - rhs), tol * max(1.0, abs(lhs)))

                        # without summing, the result is a tangent-stacked T3Tangent (V = probe stack)
                        JTz_batch = t3m.T3Tangent.probe_transpose(z, ww, base)
                        self.assertEqual(PROBE_STACK, JTz_batch.tangent_stack_shape)
                        self.assertEqual(BASE_STACK, JTz_batch.base_stack_shape)

    def test_randn(self):
        base, _ = bvf.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1)))
        self.assertTrue(t3m.T3Tangent.randn(base).is_gauged())                                   # gauged by default
        self.assertFalse(t3m.T3Tangent.randn(base, apply_gauge_projection=False).is_gauged())    # ungauged on request
        self.assertEqual(base.stack_shape, t3m.T3Tangent.randn(base).stack_shape)                # construction validates fit

    def test_to_t3(self):
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in [(), (2,)]:
                for USE_JAX in [False, True]:
                    with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                        x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                        if USE_JAX:
                            x = x.to_jax()
                        base, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.T3Tangent.randn(base, apply_gauge_projection=False, use_jax=USE_JAX)
                        base_point = t3.TuckerTensorTrain(base.up_tucker_cores, base.left_tt_cores).to_dense()

                        self.check_relerr(v.to_dense(), v.to_t3().to_dense())
                        self.check_relerr(np.asarray(base_point) + np.asarray(v.to_dense()),
                                          v.to_t3(include_shift=True).to_dense())

    def test_retract(self):
        # Dense correctness: retract(0) == base point; retract(v) == best rank-(base) T3-SVD of (base point + v).
        for T3_STRUCTURE in [((10, 11), (3, 4), (1, 2, 1)), ((9, 10, 11, 12), (2, 3, 3, 2), (1, 2, 3, 2, 1))]:
            for STACK_SHAPE in [(), (2,)]:
                for USE_JAX in [False, True]:
                    with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                        x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                        if USE_JAX:
                            x = x.to_jax()
                        base, _ = bvf.t3_orthogonal_representations(x)
                        base_point = t3.TuckerTensorTrain(base.up_tucker_cores, base.left_tt_cores).to_dense()

                        self.check_relerr(base_point, t3m.T3Tangent.zeros(base).retract(use_jax=USE_JAX).to_dense())

                        v = t3m.T3Tangent.randn(base, apply_gauge_projection=False, use_jax=USE_JAX)
                        if STACK_SHAPE == ():  # compare against a from-dense T3-SVD of (base point + v)
                            shifted_dense = np.asarray(base_point) + np.asarray(v.to_dense())
                            ref, _, _ = t3.TuckerTensorTrain.t3svd_dense(
                                shifted_dense, max_tucker_ranks=tuple(base.up_ranks), max_tt_ranks=tuple(base.left_ranks))
                            self.check_relerr(ref.to_dense(), v.retract(use_jax=USE_JAX).to_dense())

        # Rank preservation holds on a minimal-rank base.
        for STACK_SHAPE in [(), (2,)]:
            for USE_JAX in [False, True]:
                with self.subTest(rank_preservation=True, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                    x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
                    if USE_JAX:
                        x = x.to_jax()
                    base, _ = bvf.t3_orthogonal_representations(x)
                    self.assertTrue(base.has_minimal_ranks)
                    r = t3m.T3Tangent.randn(base, apply_gauge_projection=False, use_jax=USE_JAX).retract(use_jax=USE_JAX)
                    self.assertEqual(tuple(base.up_ranks), r.tucker_ranks)
                    self.assertEqual(tuple(base.left_ranks), r.tt_ranks)

    def test_project(self):
        # project(x, base) is the orthogonal projection of x onto the tangent space T_P M.
        for STR_P, STR_X in [
            (((6, 7, 5), (2, 2, 2), (1, 2, 2, 1)), ((6, 7, 5), (3, 4, 3), (1, 2, 2, 1))),
            (((9, 10, 11, 12), (2, 3, 3, 2), (1, 2, 3, 2, 1)), ((9, 10, 11, 12), (3, 3, 4, 2), (1, 2, 2, 2, 1))),
        ]:
            for STACK_SHAPE in [(), (2,)]:
                for USE_JAX in [False, True]:
                    with self.subTest(STR_P=STR_P, STACK_SHAPE=STACK_SHAPE, USE_JAX=USE_JAX):
                        p = t3.TuckerTensorTrain.randn(*STR_P, stack_shape=STACK_SHAPE)
                        x = t3.TuckerTensorTrain.randn(*STR_X, stack_shape=STACK_SHAPE)
                        if USE_JAX:
                            p, x = p.to_jax(), x.to_jax()
                        base, _ = bvf.t3_orthogonal_representations(p)

                        proj = t3m.T3Tangent.project(x, base, use_jax=USE_JAX)
                        self.assertTrue(proj.is_gauged())

                        # idempotency: projecting a tangent vector (its unshifted embedding) recovers it
                        v = t3m.T3Tangent.randn(base, apply_gauge_projection=False, use_jax=USE_JAX)
                        proj_v = t3m.T3Tangent.project(v.to_t3(use_jax=USE_JAX), base, use_jax=USE_JAX)
                        self.check_relerr(v.to_dense(), proj_v.to_dense())

                        # orthogonality: the residual x - proj_x is perpendicular to the tangent space
                        residual = np.asarray(x.to_dense()) - np.asarray(proj.to_dense())
                        tensor_axes = tuple(range(len(STACK_SHAPE), residual.ndim))
                        for _ in range(3):
                            w_dense = np.asarray(
                                t3m.T3Tangent.randn(base, apply_gauge_projection=False, use_jax=USE_JAX).to_dense())
                            ip = norm(np.sum(residual * w_dense, axis=tensor_axes))
                            self.assertLessEqual(float(ip), tol * norm(residual) * norm(w_dense))


if __name__ == "__main__":
    unittest.main()
