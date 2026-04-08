# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest
import t3tools.tucker_tensor_train as t3
import t3tools.manifold as t3m
import t3tools.util as util

np.random.seed(0)
numpy_tol = 1e-9
jax_tol = 1e-5
norm = np.linalg.norm
randn = np.random.randn

class TestTuckerTensorTrain(unittest.TestCase):
    def test_hole_shapes(self):
        basis_cores = (np.ones((10, 14)), np.ones((11, 15)), np.ones((12, 16)))
        left_tt_cores = (np.ones((5, 10, 2)), np.ones((2, 11, 3)), np.ones((3, 12, 2)))
        right_tt_cores = (np.ones((3, 10, 4)), np.ones((4, 11, 5)), np.ones((5, 12, 4)))
        outer_tt_cores = (np.ones((5, 9, 4)), np.ones((2, 8, 5)), np.ones((3, 7, 4)))
        base = (basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
        (var_basis_shapes, var_tt_shapes) = t3m.hole_shapes(base)
        self.assertEqual(var_basis_shapes, ((9, 14), (8, 15), (7, 16)))
        self.assertEqual(var_tt_shapes, ((5, 10, 4), (2, 11, 5), (3, 12, 4)))

    def test_ith_bv_to_t3(self):
        (U0, U1, U2) = (randn(10, 14), randn(11, 15), randn(12, 16))
        (L0, L1, L2) = (randn(5, 10, 2), randn(2, 11, 3), randn(3, 12, 2))
        (R0, R1, R2) = (randn(3, 10, 4), randn(4, 11, 5), randn(5, 12, 4))
        (O0, O1, O2) = (randn(5, 9, 4), randn(2, 8, 5), randn(3, 7, 4))
        base = ((U0, U1, U2), (L0, L1, L2), (R0, R1, R2), (O0, O1, O2))
        (V0, V1, V2) = (randn(9, 14), randn(8, 15), randn(7, 16))
        (H0, H1, H2) = (randn(1, 10, 4), randn(2, 11, 5), randn(3, 12, 1))
        variation = ((V0, V1, V2), (H0, H1, H2))

        # TT replacements

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(0, True, base, variation)
        self.assertEqual(((U0, U1, U2), (H0, R1, R2)), ((B0, B1, B2), (G0, G1, G2)))

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(1, True, base, variation)
        self.assertEqual(((U0, U1, U2), (L0, H1, R2)), ((B0, B1, B2), (G0, G1, G2)))

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(2, True, base, variation)
        self.assertEqual(((U0, U1, U2), (L0, L1, H2)), ((B0, B1, B2), (G0, G1, G2)))

        # Basis replacements

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(0, False, base, variation)
        self.assertEqual(((V0, U1, U2), (O0, R1, R2)), ((B0, B1, B2), (G0, G1, G2)))

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(1, False, base, variation)
        self.assertEqual(((U0, V1, U2), (L0, O1, R2)), ((B0, B1, B2), (G0, G1, G2)))

        ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(2, False, base, variation)
        self.assertEqual(((U0, U1, V2), (L0, L1, O2)), ((B0, B1, B2), (G0, G1, G2)))


    def test_orthogonal_representations(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)))
        base, variation = t3m.orthogonal_representations(x)  # Compute orthogonal representations
        basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
        basis_vars, tt_vars = variation
        (U0, U1, U2) = basis_cores
        (L0, L1, L2) = left_tt_cores
        (R0, R1, R2) = right_tt_cores
        (O0, O1, O2) = outer_tt_cores
        (V0, V1, V2) = basis_vars
        (H0, H1, H2) = tt_vars

        # TT replacement

        x2 = ((U0, U1, U2), (H0, R1, R2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        x2 = ((U0, U1, U2), (L0, H1, R2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        x2 = ((U0, U1, U2), (L0, L1, H2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        # basis replacement

        x2 = ((V0, U1, U2), (O0, R1, R2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        x2 = ((U0, V1, U2), (L0, O1, R2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        x2 = ((U0, U1, V2), (L0, L1, O2))
        self.assertLessEqual(norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)), numpy_tol * norm(t3.t3_to_dense(x)))

        # Basis orthogonality

        for U in [U0, U1, U2]:
            self.assertLessEqual(
                norm(U @ U.T - np.eye(U.shape[0])),
                numpy_tol * norm(np.eye(U.shape[0]))
            )

        # left orthogonality

        for L in [L0, L1]: # Last core need not be left orthogonal
            self.assertLessEqual(
                norm(np.einsum('iaj,iak->jk', L, L) - np.eye(L.shape[2])),
                numpy_tol * norm(np.eye(L.shape[2]))
            )

        # right orthogonality

        for R in [R1, R2]: # First core need not be right orthogonal
            self.assertLessEqual(
                norm(np.einsum('iaj,kaj->ik', R, R) - np.eye(R.shape[0])),
                numpy_tol * norm(np.eye(R.shape[0]))
            )

        # outer orthogonality

        for O in [O0, O1, O2]:
            self.assertLessEqual(
                norm(np.einsum('iaj,ibj->ab', O, O) - np.eye(O.shape[1])),
                numpy_tol * norm(np.eye(O.shape[1]))
            )

    def test_manifold_dim(self):
        s = ((5, 6, 3), (5, 3, 2), (2, 2, 4, 1))
        mdim = t3m.manifold_dim(s)
        p = t3.t3_corewise_randn(s)
        base, _ = t3m.orthogonal_representations(p)
        basis_shapes, tt_shapes = t3m.hole_shapes(base)
        num_basis_entries = np.sum([np.prod(shape) for shape in basis_shapes])
        num_tt_entries = np.sum([np.prod(shape) for shape in tt_shapes])
        num_core_entries = num_basis_entries + num_tt_entries
        vv = [t3m.tangent_randn(base, apply_gauge_projection=False) for _ in range(num_core_entries)]
        dense_vv = np.stack([t3m.tangent_to_dense(v, base) for v in vv])
        _, ss, _ = np.linalg.svd(dense_vv.reshape((num_core_entries, -1)), full_matrices=False)
        self.assertLessEqual(ss[mdim], numpy_tol * ss[0]) # first zero singular value
        self.assertGreaterEqual(ss[mdim-1], 1e8 * ss[mdim]) # last nonzero singular value

    def test_tangent_to_dense(self):
        p = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 2)))
        base, _ = t3m.orthogonal_representations(p)
        variation = t3m.tangent_randn(base)
        v_dense = t3m.tangent_to_dense(variation, base)  # Convert tangent to dense
        ((U0, U1, U2), (L0, L1, L2), (R0, R1, R2), (O0, O1, O2)) = base
        ((V0, V1, V2), (H0, H1, H2)) = variation
        s1 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0, U1, U2, H0, R1, R2)
        s2 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0, U1, U2, L0, H1, R2)
        s3 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0, U1, U2, L0, L1, H2)
        s4 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', V0, U1, U2, O0, R1, R2)
        s5 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0, V1, U2, L0, O1, R2)
        s6 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0, U1, V2, L0, L1, O2)
        v_dense2 = s1 + s2 + s3 + s4 + s5 + s6
        self.assertLessEqual(norm(v_dense - v_dense2), numpy_tol * norm(v_dense2))

        p_plus_v_dense = t3m.tangent_to_dense(variation, base, include_shift=True)  # Convert shifted tangent, p+v, to dense
        p_plus_v_dense2 = t3.t3_to_dense(p) + v_dense
        self.assertLessEqual(norm(p_plus_v_dense - p_plus_v_dense2), numpy_tol * norm(p_plus_v_dense2))

    def test_tangent_to_t3(self):
        p = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (2, 3, 2, 2)))
        base, _ = t3m.orthogonal_representations(p)
        variation = t3m.tangent_randn(base)
        v_t3 = t3m.tangent_to_t3(variation, base)  # tangent vector only (attached at zero)
        v_dense = t3.t3_to_dense(v_t3)
        v_dense2 = t3m.tangent_to_dense(variation, base)
        self.assertLessEqual(norm(v_dense - v_dense2), numpy_tol * norm(v_dense))

        p_plus_v_t3 = t3m.tangent_to_t3(variation, base, include_shift=True)  # shifted tangent vector (include attachment at base point)
        p_plus_v_dense = t3.t3_to_dense(p_plus_v_t3)
        p_plus_v_dense2 = v_dense2 + t3.t3_to_dense(p)
        self.assertLessEqual(norm(p_plus_v_dense - p_plus_v_dense2), numpy_tol * norm(p_plus_v_dense2))

    def test_tangent_zeros(self):
        p = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (7, 3, 2, 5)))
        base, _ = t3m.orthogonal_representations(p)
        z = t3m.tangent_zeros(base)
        self.assertLessEqual(norm(t3m.tangent_to_dense(z, base)), numpy_tol)

        shapes = t3m.hole_shapes(base)
        basis_z, tt_z = z
        basis_shapes, tt_shapes = shapes

        for B, s in zip(basis_z, basis_shapes):
            self.assertEqual(B.shape, s)

        for G, s in zip(tt_z, tt_shapes):
            self.assertEqual(G.shape, s)

    def test_tangent_randn(self):
        p = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (7, 3, 2, 5)))
        base, vars0 = t3m.orthogonal_representations(p)
        v = t3m.tangent_randn(base)  # Random tangent vector, gauged.

        shapes = t3m.hole_shapes(base)
        basis_v, tt_v = v
        basis_shapes, tt_shapes = shapes

        for B, s in zip(basis_v, basis_shapes):
            self.assertEqual(B.shape, s)

        for G, s in zip(tt_v, tt_shapes):
            self.assertEqual(G.shape, s)

    def test_orthogonal_projection(self):
        p = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)))
        base, _ = t3m.orthogonal_representations(p)
        variation = t3m.tangent_randn(base)
        proj_variation = t3m.orthogonal_gauge_projection(variation, base)  # Make gauged via orthogonal projection
        (U0, U1, U2), (L0, L1, L2), _, _ = base
        ((V0, V1, V2), (H0, H1, H2)) = proj_variation

        # Gauge conditions

        for V, U in zip([V0, V1, V2], [U0, U1, U2]):
            self.assertLessEqual(norm(V @ U.T), numpy_tol)

        for H, L in zip([H0, H1], [L0, L1]): # no gauge condition on H2
            self.assertLessEqual(norm(np.einsum('iaj,iak->jk', H, L)), numpy_tol)

        # Check that projection was orthogonal

        v_minus_p_dot_p = util.corewise_dot(
            util.corewise_sub(variation, proj_variation),
            proj_variation
        )

        self.assertLessEqual(
            np.abs(v_minus_p_dot_p),
            numpy_tol * util.corewise_norm(variation)
        )

    def test_oblique_gauge_projection(self):
        p = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)))
        base, _ = t3m.orthogonal_representations(p)
        variation = t3m.tangent_randn(base)
        proj_variation = t3m.oblique_gauge_projection(variation, base)  # Make gauged via oblique projection
        v_dense = t3m.tangent_to_dense(variation, base)
        proj_v_dense = t3m.tangent_to_dense(proj_variation, base)

        # Check that vector represents same tangent after oblique projection
        self.assertLessEqual(norm(v_dense - proj_v_dense), numpy_tol * norm(v_dense))

        (U0, U1, U2), (L0, L1, L2), _, _ = base
        ((V0, V1, V2), (H0, H1, H2)) = proj_variation

        # Check gauge conditions

        for U, V in zip([U0, U1, U2], [V0, V1, V2]):
            self.assertLessEqual(norm(V @ U.T), numpy_tol)

        for L, H in zip([L0, L1], [H0, H1]):
            self.assertLessEqual(norm(np.einsum('iaj,iak->jk', H1, L1)), numpy_tol)

    def test_project_t3_into_tangent_space(self):
        p = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)))
        base, _ = t3m.orthogonal_representations(p)
        x = t3.t3_corewise_randn(((14, 15, 16), (7, 4, 8), (2, 5, 4, 2)))
        proj_x = t3m.project_t3_onto_tangent_space(x, base)  # Project x onto tangent space
        P = t3.t3_to_dense(p)
        X = t3.t3_to_dense(x)
        proj_X = t3m.tangent_to_dense(proj_x, base)
        self.assertLessEqual(np.abs(np.sum((X - proj_X) * (proj_X - P))), numpy_tol * np.linalg.norm(X))


    def test_retract(self):
        p = t3.t3_corewise_randn(((14, 15, 16), (4, 5, 6), (1, 3, 2, 1)))
        base, _ = t3m.orthogonal_representations(p)
        z = t3m.tangent_zeros(base)
        ret_v = t3m.retract(z, base)
        ret_V = t3.t3_to_dense(ret_v)
        P = t3.t3_to_dense(p)
        self.assertLessEqual(norm(ret_V - P), numpy_tol)

        # Need to figure out a good way to automatically check the condition that
        # the Jacobian of the retraction is the identity.



if __name__ == '__main__':
    unittest.main()
