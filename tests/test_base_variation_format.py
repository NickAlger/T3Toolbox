# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3toolbox.base_variation_format as bvf
try:
    import t3toolbox.jax.base_variation_format as bvf_jax
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    bvf_jax = bvf


np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestBaseVariationFormat(unittest.TestCase):
    def test_hole_shapes(self):
        for BVF in [bvf, bvf_jax]:
            with self.subTest(BVF=BVF):
                tucker_cores = (np.ones((10, 14)), np.ones((11, 15)), np.ones((12, 16)))
                left_tt_cores = (np.ones((5, 10, 2)), np.ones((2, 11, 3)), np.ones((3,12,4)))
                right_tt_cores = (np.ones((1,10,4)), np.ones((4, 11, 5)), np.ones((5, 12, 4)))
                outer_tt_cores = (np.ones((5, 9, 4)), np.ones((2, 8, 5)), np.ones((3, 7, 4)))
                base = (tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)

                shapes = BVF.get_base_hole_shapes(base)

                var_tucker_shapes, var_tt_shapes = shapes

                self.assertEqual(var_tucker_shapes, ((9, 14), (8, 15), (7, 16)))
                self.assertEqual(var_tt_shapes, ((5, 10, 4), (2, 11, 5), (3, 12, 4)))

                shapes_jax = bvf_jax.get_base_hole_shapes(base)
                self.assertEqual(shapes, shapes_jax)


    def test_ith_bv_to_t3(self):
        for BVF in [bvf, bvf_jax]:
            with self.subTest(BVF=BVF):
                (U0, U1, U2) = (randn(10, 14), randn(11, 15), randn(12, 16))
                (L0, L1, L2) = (randn(5, 10, 2), randn(2, 11, 3), randn(3, 12, 2))
                (R0, R1, R2) = (randn(3,10,4), randn(4, 11, 5), randn(5, 12, 4))
                (O0, O1, O2) = (randn(5, 9, 4), randn(2, 8, 5), randn(3, 7, 4))
                base = ((U0, U1, U2), (L0, L1, L2), (R0, R1, R2), (O0, O1, O2))
                (V0, V1, V2) = (randn(9, 14), randn(8, 15), randn(7, 16))
                (H0, H1, H2) = (randn(1, 10, 4), randn(2, 11, 5), randn(3, 12, 1))
                variation = ((V0, V1, V2), (H0, H1, H2))

                # TT replacements

                ((B0, B1, B2), (G0, G1, G2)) = BVF.ith_bv_to_t3(0, True, base, variation)
                self.assertEqual(((U0, U1, U2), (H0, R1, R2)), ((B0, B1, B2), (G0, G1, G2)))

                ((B0, B1, B2), (G0, G1, G2)) = BVF.ith_bv_to_t3(1, True, base, variation)
                self.assertEqual(((U0, U1, U2), (L0, H1, R2)), ((B0, B1, B2), (G0, G1, G2)))

                ((B0, B1, B2), (G0, G1, G2)) = BVF.ith_bv_to_t3(2, True, base, variation)
                self.assertEqual(((U0, U1, U2), (L0, L1, H2)), ((B0, B1, B2), (G0, G1, G2)))

                # Basis replacements

                ((B0, B1, B2), (G0, G1, G2)) = BVF.ith_bv_to_t3(0, False, base, variation)
                self.assertEqual(((V0, U1, U2), (O0, R1, R2)), ((B0, B1, B2), (G0, G1, G2)))

                ((B0, B1, B2), (G0, G1, G2)) = BVF.ith_bv_to_t3(1, False, base, variation)
                self.assertEqual(((U0, V1, U2), (L0, O1, R2)), ((B0, B1, B2), (G0, G1, G2)))

                ((B0, B1, B2), (G0, G1, G2)) = BVF.ith_bv_to_t3(2, False, base, variation)
                self.assertEqual(((U0, U1, V2), (L0, L1, O2)), ((B0, B1, B2), (G0, G1, G2)))



if __name__ == '__main__':
    unittest.main()

