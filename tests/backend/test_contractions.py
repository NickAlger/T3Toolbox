# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3toolbox.backend.contractions as contractions

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jnp = np

np.random.seed(0)
tol = 1e-9

numpy_randn = np.random.randn
jax_randn = lambda *args: jnp.array(np.random.randn(*args))

class TestContractions(unittest.TestCase):
    def check_relerr(self, xtrue, x):
        self.assertLessEqual(
            np.linalg.norm(xtrue - x),
            tol * np.linalg.norm(xtrue)
        )

    def test_Na_Maib_Ni_to_NMb(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over N and M:
                xyz_a = RANDN(2,3,4, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                xyz_i = RANDN(2,3,4, 11)
                result = contractions.Na_Maib_Ni_to_NMb(xyz_a, uv_aib, xyz_i)
                result_true = np.einsum('xyza,uvaib,xyzi->xyzuvb', xyz_a, uv_aib, xyz_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over N only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                xyz_i = RANDN(2,3,4, 11)
                result = contractions.Na_Maib_Ni_to_NMb(xyz_a, aib, xyz_i)
                result_true = np.einsum('xyza,aib,xyzi->xyzb', xyz_a, aib, xyz_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over M only:
                a = RANDN(10)
                uv_aib = RANDN(5,6, 10,11,12)
                i = RANDN(11)
                result = contractions.Na_Maib_Ni_to_NMb(a, uv_aib, i)
                result_true = np.einsum('a,uvaib,i->uvb', a, uv_aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                i = RANDN(11)
                result = contractions.Na_Maib_Ni_to_NMb(a, aib, i)
                result_true = np.einsum('a,aib,i->b', a, aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)