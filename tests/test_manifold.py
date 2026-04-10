# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3tools.base_variation_format as bvf
import t3tools.corewise
import t3tools.orthogonalization as orth
import t3tools.tucker_tensor_train as t3
import t3tools.manifold as t3m

try:
    import t3tools.jax.manifold as t3m_jax
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    t3m_jax = t3m

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestManifold(unittest.TestCase):
    def check_abserr(self, xtrue, x):
        self.assertLessEqual(norm(xtrue - x), tol)

    def check_relerr(self, xtrue, x):
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def check_relerr_corewise(self, xtrue, x):
        for XX, XX_jax in zip(xtrue, x):
            for X, X_jax in zip(XX, XX_jax):
                self.check_relerr(X, X_jax)

    def test_manifold_dim(self):
        shapes = [
            ((5, 6, 3), (5, 3, 2), (2, 2, 4, 1)),
        ]

        for SHAPE in shapes:
            for T3M in [t3m, t3m_jax]:
                with self.subTest(T3M=T3M, SHAPE=SHAPE):
                    mdim = T3M.manifold_dim(SHAPE)

                    p = t3.t3_corewise_randn(SHAPE)
                    base, _ = orth.orthogonal_representations(p)
                    basis_shapes, tt_shapes = bvf.hole_shapes(base)
                    num_basis_entries = np.sum([np.prod(shape) for shape in basis_shapes])
                    num_tt_entries = np.sum([np.prod(shape) for shape in tt_shapes])
                    num_core_entries = num_basis_entries + num_tt_entries
                    vv = [t3m.tangent_randn(base, apply_gauge_projection=False) for _ in range(num_core_entries)]
                    dense_vv = np.stack([t3m.tangent_to_dense(v, base) for v in vv])
                    _, ss, _ = np.linalg.svd(dense_vv.reshape((num_core_entries, -1)), full_matrices=False)
                    self.assertLessEqual(ss[mdim], tol * ss[0]) # first zero singular value
                    self.assertGreaterEqual(ss[mdim-1], 1e8 * ss[mdim]) # last nonzero singular value

    def test_tangent_to_dense(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (2, 3, 2, 2)),
        ]

        for STRUCTURE in structures:
            for T3M in [t3m, t3m_jax]:
                with self.subTest(T3M=T3M, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, _ = orth.orthogonal_representations(p)
                    variation = t3m.tangent_randn(base)

                    v_dense     = T3M.tangent_to_dense(variation, base)  # Convert tangent to dense

                    ((U0, U1, U2), (L0, L1, L2), (R0, R1, R2), (O0, O1, O2)) = base
                    ((V0, V1, V2), (H0, H1, H2)) = variation
                    s1 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0, U1, U2, H0, R1, R2)
                    s2 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0, U1, U2, L0, H1, R2)
                    s3 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0, U1, U2, L0, L1, H2)
                    s4 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', V0, U1, U2, O0, R1, R2)
                    s5 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0, V1, U2, L0, O1, R2)
                    s6 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0, U1, V2, L0, L1, O2)
                    v_dense2 = s1 + s2 + s3 + s4 + s5 + s6
                    self.check_relerr(v_dense2, v_dense)

                    p_plus_v_dense = t3m.tangent_to_dense(variation, base, include_shift=True)  # Convert shifted tangent, p+v, to dense
                    p_plus_v_dense2 = t3.t3_to_dense(p) + v_dense
                    self.check_relerr(p_plus_v_dense2, p_plus_v_dense)

    def test_tangent_to_t3(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (2, 3, 2, 2)),
        ]

        for STRUCTURE in structures:
            for T3M in [t3m, t3m_jax]:
                with self.subTest(T3M=T3M, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, _ = orth.orthogonal_representations(p)
                    variation = t3m.tangent_randn(base)

                    v_t3 = T3M.tangent_to_t3(variation, base)  # tangent vector only (attached at zero)

                    v_dense = t3.t3_to_dense(v_t3)
                    v_dense_jax = t3.t3_to_dense(v_t3)
                    self.check_relerr(v_dense, v_dense_jax)

                    v_dense2 = t3m.tangent_to_dense(variation, base)
                    self.check_relerr(v_dense2, v_dense)

                    p_plus_v_t3 = t3m.tangent_to_t3(variation, base, include_shift=True)  # shifted tangent vector (include attachment at base point)
                    p_plus_v_dense      = t3.t3_to_dense(p_plus_v_t3)

                    p_plus_v_dense2 = v_dense2 + t3.t3_to_dense(p)
                    self.check_relerr(p_plus_v_dense2, p_plus_v_dense)

    def test_tangent_zeros(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (7, 3, 2, 5)),
        ]

        for STRUCTURE in structures:
            for T3M in [t3m, t3m_jax]:
                with self.subTest(T3M=T3M, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, _ = orth.orthogonal_representations(p)

                    z = T3M.tangent_zeros(base)

                    self.assertLessEqual(norm(t3m.tangent_to_dense(z, base)), tol)

                    shapes = bvf.hole_shapes(base)
                    basis_z, tt_z = z
                    basis_shapes, tt_shapes = shapes

                    for B, s in zip(basis_z, basis_shapes):
                        self.assertEqual(B.shape, s)

                    for G, s in zip(tt_z, tt_shapes):
                        self.assertEqual(G.shape, s)


    def test_tangent_randn(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (7, 3, 2, 5)),
        ]

        for STRUCTURE in structures:
            for T3M in [t3m, t3m_jax]:
                with self.subTest(T3M=T3M, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, vars0 = orth.orthogonal_representations(p)

                    v = T3M.tangent_randn(base)  # Random tangent vector, gauged.

                    shapes = bvf.hole_shapes(base)
                    basis_v, tt_v = v
                    basis_shapes, tt_shapes = shapes

                    for B, s in zip(basis_v, basis_shapes):
                        self.assertEqual(B.shape, s)

                    for G, s in zip(tt_v, tt_shapes):
                        self.assertEqual(G.shape, s)

    def test_orthogonal_projection(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)),
        ]

        for STRUCTURE in structures:
            for T3M in [t3m, t3m_jax]:
                with self.subTest(T3M=T3M, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, _ = orth.orthogonal_representations(p)
                    variation = t3m.tangent_randn(base)

                    proj_variation = T3M.orthogonal_gauge_projection(variation, base)  # Make gauged via orthogonal projection

                    (U0, U1, U2), (L0, L1, L2), _, _ = base
                    ((V0, V1, V2), (H0, H1, H2)) = proj_variation

                    # Gauge conditions

                    for V, U in zip([V0, V1, V2], [U0, U1, U2]):
                        self.assertLessEqual(norm(V @ U.T), tol)

                    for H, L in zip([H0, H1], [L0, L1]): # no gauge condition on H2
                        self.assertLessEqual(norm(np.einsum('iaj,iak->jk', H, L)), tol)

                    # Check that projection was orthogonal

                    v_minus_p_dot_p = t3tools.corewise.corewise_dot(
                        t3tools.corewise.corewise_sub(variation, proj_variation),
                        proj_variation
                    )

                    self.assertLessEqual(
                        np.abs(v_minus_p_dot_p),
                        tol * t3tools.corewise.corewise_norm(variation)
                    )

    def test_oblique_gauge_projection(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)),
        ]

        for STRUCTURE in structures:
            for T3M in [t3m, t3m_jax]:
                with self.subTest(T3M=T3M, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, _ = orth.orthogonal_representations(p)
                    variation = t3m.tangent_randn(base)

                    proj_variation = T3M.oblique_gauge_projection(variation, base)  # Make gauged via oblique projection

                    v_dense = t3m.tangent_to_dense(variation, base)
                    proj_v_dense = t3m.tangent_to_dense(proj_variation, base)

                    # Check that vector represents same tangent after oblique projection
                    self.check_relerr(v_dense, proj_v_dense)

                    (U0, U1, U2), (L0, L1, L2), _, _ = base
                    ((V0, V1, V2), (H0, H1, H2)) = proj_variation

                    # Check gauge conditions

                    for U, V in zip([U0, U1, U2], [V0, V1, V2]):
                        self.assertLessEqual(norm(V @ U.T), tol)

                    for L, H in zip([L0, L1], [H0, H1]):
                        self.assertLessEqual(norm(np.einsum('iaj,iak->jk', H, L)), tol)

    def test_project_t3_into_tangent_space(self):
        structures_p = [
            ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)),
        ]
        structures_x = [
            ((14, 15, 16), (7, 4, 8), (2, 5, 4, 2)),
        ]

        for STRUCTURE_P, STRUCTURE_X in zip(structures_p, structures_x):
            for T3M in [t3m, t3m_jax]:
                with self.subTest(T3M=T3M, STRUCTURE_P=STRUCTURE_P, STRUCTURE_X=STRUCTURE_X):
                    p = t3.t3_corewise_randn(STRUCTURE_P)
                    base, _ = orth.orthogonal_representations(p)
                    x = t3.t3_corewise_randn(STRUCTURE_X)

                    proj_x = T3M.project_t3_onto_tangent_space(x, base)  # Project x onto tangent space

                    P = t3.t3_to_dense(p)
                    X = t3.t3_to_dense(x)
                    proj_X = t3m.tangent_to_dense(proj_x, base)
                    self.assertLessEqual(np.abs(np.sum((X - proj_X) * (proj_X - P))), tol * np.linalg.norm(X))


    def test_retract(self):
        structures = [
            ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)),
        ]

        for STRUCTURE in structures:
            for T3M in [t3m, t3m_jax]:
                with self.subTest(T3M=T3M, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, _ = orth.orthogonal_representations(p)
                    z = t3m.tangent_zeros(base)

                    ret_v = T3M.retract(z, base)

                    ret_V = t3.t3_to_dense(ret_v)
                    P = t3.t3_to_dense(p)
                    self.assertLessEqual(norm(ret_V - P), tol)

        # Need to figure out a good way to automatically check the condition that
        # the Jacobian of the retraction is the identity.



if __name__ == '__main__':
    unittest.main()
