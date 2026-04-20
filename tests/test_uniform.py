# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.OLD_uniform as ut3

try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestUniform(unittest.TestCase):
    def check_relerr(self, xtrue, x):
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def test_padded_and_original_structure(self):
        s = ((14,15,16), (4,6,5), (2,3,2,1))
        x = t3.t3_zeros(s)
        cores, masks = ut3.t3_to_ut3(x, squash_tails=False)
        self.assertEqual((3, 16, 6, 3), ut3.get_uniform_structure(cores))
        self.assertEqual(s, ut3.get_original_structure(masks))

    def test_unpack_edge_vectors1(self):
        for USE_JAX in [True, False]:
            with self.subTest(USE_JAX=USE_JAX):
                E = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
                submask = [[True, False, True, True], [False, True, False, False]]

                ee = ut3.unpack(E, submask, use_jax=USE_JAX)

                self.assertEqual([1, 3, 4], list(ee[0]))
                self.assertEqual([6], list(ee[1]))

    def test_unpack_edge_vectors2(self):
        for USE_JAX in [True, False]:
            with self.subTest(USE_JAX=USE_JAX):
                E = np.random.randn(6, 5, 4, 3, 2)
                submask = [[False, False], [False, True], [True, True]]

                ee = ut3.unpack(E, submask, use_jax=USE_JAX)

                self.assertEqual([(6, 5, 4, 0), (6, 5, 4, 1), (6, 5, 4, 2)], [e.shape for e in ee])

    def test_t3_to_ut3_to_t3(self):
        structures = [
            ((14, 15, 16), (4, 6, 5), (3, 3, 2, 4))
        ]

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                with self.subTest(USE_JAX=USE_JAX):
                    x = t3.t3_corewise_randn(STRUCTURE)

                    cores, masks = ut3.t3_to_ut3(x, use_jax=USE_JAX)  # Convert t3 -> ut3
                    x2 = ut3.ut3_to_t3(cores, masks, use_jax=USE_JAX)  # Convert ut3 -> t3

                    dense_x = t3.t3_to_dense(x)
                    dense_x2 = t3.t3_to_dense(x2)
                    self.check_relerr(dense_x, dense_x2)

    def test_ut3_to_dense(self):
        structures = [
            ((14, 15, 16), (4, 6, 5), (3, 3, 2, 4))
        ]

        for STRUCTURE in structures:
            for USE_JAX in [True, False]:
                with self.subTest(USE_JAX=USE_JAX):
                    x = t3.t3_corewise_randn(STRUCTURE)
                    cores, masks = ut3.t3_to_ut3(x)  # Convert t3 -> ut3
                    dense_x = t3.t3_to_dense(x, use_jax=USE_JAX)
                    dense_x2 = ut3.ut3_to_dense(cores, masks)
                    self.check_relerr(dense_x, dense_x2)


if __name__ == '__main__':
    unittest.main()
