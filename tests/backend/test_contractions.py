# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
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

# Distinct einsum letters per grouped block (avoiding the single-index letters a, i, b, o).
GROUP_POOL = {'W': 'fgh', 'K': 'vwx', 'C': 'mnp'}
SINGLE_SIZE = {'a': 3, 'i': 4, 'b': 5, 'o': 6, 'j': 7}

# (W, K, C) stack shapes exercised by the three-group contractions: all present, K/W multi-axis,
# each block empty in turn (K empty is the degenerate two-group reduction), and all empty.
THREE_GROUP_COMBOS = [
    ((2,),   (3,),   (2,)),
    ((2,),   (3, 2), (2,)),
    ((2, 2), (3,),   (2,)),
    ((),     (3,),   (2,)),
    ((2,),   (),     (2,)),
    ((2,),   (3,),   ()),
    ((),     (),     ()),
]


class TestContractions(unittest.TestCase):
    def check_relerr(self, xtrue, x):
        self.assertLessEqual(
            np.linalg.norm(xtrue - x),
            tol * np.linalg.norm(xtrue)
        )

    def test_Wa_Caib_Wi_to_WCb(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                xyz_a = RANDN(2,3,4, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                xyz_i = RANDN(2,3,4, 11)
                result = contractions.Wa_Caib_Wi_to_WCb(xyz_a, uv_aib, xyz_i)
                result_true = np.einsum('xyza,uvaib,xyzi->xyzuvb', xyz_a, uv_aib, xyz_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over W only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                xyz_i = RANDN(2,3,4, 11)
                result = contractions.Wa_Caib_Wi_to_WCb(xyz_a, aib, xyz_i)
                result_true = np.einsum('xyza,aib,xyzi->xyzb', xyz_a, aib, xyz_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over C only:
                a = RANDN(10)
                uv_aib = RANDN(5,6, 10,11,12)
                i = RANDN(11)
                result = contractions.Wa_Caib_Wi_to_WCb(a, uv_aib, i)
                result_true = np.einsum('a,uvaib,i->uvb', a, uv_aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                i = RANDN(11)
                result = contractions.Wa_Caib_Wi_to_WCb(a, aib, i)
                result_true = np.einsum('a,aib,i->b', a, aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_CWa_Caib_Wo_Cio_to_CWb(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                uv_xyz_a = RANDN(5,6, 2,3,4, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                xyz_o = RANDN(2,3,4, 13)
                uv_io = RANDN(5,6, 11,13)
                result = contractions.CWa_Caib_Wo_Cio_to_CWb(uv_xyz_a, uv_aib, xyz_o, uv_io)
                result_true = np.einsum('uvxyza,uvaib,xyzo,uvio->uvxyzb', uv_xyz_a, uv_aib, xyz_o, uv_io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over W only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                xyz_o = RANDN(2,3,4, 13)
                io = RANDN(11,13)
                result = contractions.CWa_Caib_Wo_Cio_to_CWb(xyz_a, aib, xyz_o, io)
                result_true = np.einsum('xyza,aib,xyzo,io->xyzb', xyz_a, aib, xyz_o, io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over C only:
                uv_a = RANDN(5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                o = RANDN(13)
                uv_io = RANDN(5,6, 11,13)
                result = contractions.CWa_Caib_Wo_Cio_to_CWb(uv_a, uv_aib, o, uv_io)
                result_true = np.einsum('uva,uvaib,o,uvio->uvb', uv_a, uv_aib, o, uv_io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                o = RANDN(13)
                io = RANDN(11,13)
                result = contractions.CWa_Caib_Wo_Cio_to_CWb(a, aib, o, io)
                result_true = np.einsum('a,aib,o,io->b', a, aib, o, io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_WCa_Caib_Wo_Cio_to_WCb(self):
        # WC twin of the apply contraction (base-inner: W outer, C inner).
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                xyz_uv_a = RANDN(2,3,4, 5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                xyz_o = RANDN(2,3,4, 13)
                uv_io = RANDN(5,6, 11,13)
                result = contractions.WCa_Caib_Wo_Cio_to_WCb(xyz_uv_a, uv_aib, xyz_o, uv_io)
                result_true = np.einsum('xyzuva,uvaib,xyzo,uvio->xyzuvb', xyz_uv_a, uv_aib, xyz_o, uv_io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over W only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                xyz_o = RANDN(2,3,4, 13)
                io = RANDN(11,13)
                result = contractions.WCa_Caib_Wo_Cio_to_WCb(xyz_a, aib, xyz_o, io)
                result_true = np.einsum('xyza,aib,xyzo,io->xyzb', xyz_a, aib, xyz_o, io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over C only:
                uv_a = RANDN(5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                o = RANDN(13)
                uv_io = RANDN(5,6, 11,13)
                result = contractions.WCa_Caib_Wo_Cio_to_WCb(uv_a, uv_aib, o, uv_io)
                result_true = np.einsum('uva,uvaib,o,uvio->uvb', uv_a, uv_aib, o, uv_io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                o = RANDN(13)
                io = RANDN(11,13)
                result = contractions.WCa_Caib_Wo_Cio_to_WCb(a, aib, o, io)
                result_true = np.einsum('a,aib,o,io->b', a, aib, o, io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_CWa_Caib_CiW_to_CWb(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                uv_xyz_a = RANDN(5,6, 2,3,4, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                uv_i_xyz = RANDN(5,6, 11, 2,3,4)
                result = contractions.CWa_Caib_CiW_to_CWb(uv_xyz_a, uv_aib, uv_i_xyz)
                result_true = np.einsum('uvxyza,uvaib,uvixyz->uvxyzb', uv_xyz_a, uv_aib, uv_i_xyz)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over W only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                i_xyz = RANDN(11, 2,3,4)
                result = contractions.CWa_Caib_CiW_to_CWb(xyz_a, aib, i_xyz)
                result_true = np.einsum('xyza,aib,ixyz->xyzb', xyz_a, aib, i_xyz)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over C only:
                uv_a = RANDN(5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                uv_i = RANDN(5,6, 11)
                result = contractions.CWa_Caib_CiW_to_CWb(uv_a, uv_aib, uv_i)
                result_true = np.einsum('uva,uvaib,uvi->uvb', uv_a, uv_aib, uv_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                i = RANDN(11)
                result = contractions.CWa_Caib_CiW_to_CWb(a, aib, i)
                result_true = np.einsum('a,aib,i->b', a, aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_WCa_Caib_WCi_to_WCb(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                xyz_uv_a = RANDN(2,3,4, 5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                xyz_uv_i = RANDN(2,3,4, 5,6, 11)
                result = contractions.WCa_Caib_WCi_to_WCb(xyz_uv_a, uv_aib, xyz_uv_i)
                result_true = np.einsum('xyzuva,uvaib,xyzuvi->xyzuvb', xyz_uv_a, uv_aib, xyz_uv_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over W only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                xyz_i = RANDN(2,3,4, 11)
                result = contractions.WCa_Caib_WCi_to_WCb(xyz_a, aib, xyz_i)
                result_true = np.einsum('xyza,aib,xyzi->xyzb', xyz_a, aib, xyz_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over C only:
                uv_a = RANDN(5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                uv_i = RANDN(5,6, 11)
                result = contractions.WCa_Caib_WCi_to_WCb(uv_a, uv_aib, uv_i)
                result_true = np.einsum('uva,uvaib,uvi->uvb', uv_a, uv_aib, uv_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                i = RANDN(11)
                result = contractions.WCa_Caib_WCi_to_WCb(a, aib, i)
                result_true = np.einsum('a,aib,i->b', a, aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_Cio_Wo_to_WCi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                Cio = RANDN(5,6, 10,13)
                Wo = RANDN(2,3,4, 13)
                result = contractions.Cio_Wo_to_WCi(Cio, Wo)
                result_true = np.einsum('uvio,xyzo->xyzuvi', Cio, Wo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over W only:
                Cio = RANDN(10,13)
                Wo = RANDN(2,3,4, 13)
                result = contractions.Cio_Wo_to_WCi(Cio, Wo)
                result_true = np.einsum('io,xyzo->xyzi', Cio, Wo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over C only:
                Cio = RANDN(5,6, 10,13)
                Wo = RANDN(13)
                result = contractions.Cio_Wo_to_WCi(Cio, Wo)
                result_true = np.einsum('uvio,o->uvi', Cio, Wo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Wo vectorization:
                Cio = RANDN(10,13)
                Wo = RANDN(13)
                result = contractions.Cio_Wo_to_WCi(Cio, Wo)
                result_true = np.einsum('io,o->i', Cio, Wo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_dCio_dWo_to_dWCi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                dCio = RANDN(8, 5,6, 10,13)
                dWo = RANDN(8, 2,3,4, 13)
                result = contractions.dCio_dWo_to_dWCi(dCio, dWo)
                result2 = np.einsum('duvio,dxyzo->dxyzuvi', dCio, dWo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over W only:
                dCio = RANDN(8, 10,13)
                dWo = RANDN(8, 2,3,4, 13)
                result = contractions.dCio_dWo_to_dWCi(dCio, dWo)
                result2 = np.einsum('dio,dxyzo->dxyzi', dCio, dWo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over C only:
                dCio = RANDN(8, 5,6, 10,13)
                dWo = RANDN(8, 13)
                result = contractions.dCio_dWo_to_dWCi(dCio, dWo)
                result2 = np.einsum('duvio,do->duvi', dCio, dWo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                dCio = RANDN(8, 10,13)
                dWo = RANDN(8, 13)
                result = contractions.dCio_dWo_to_dWCi(dCio, dWo)
                result2 = np.einsum('dio,do->di', dCio, dWo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_WCa_Caib_WCb_to_WCi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                WCa = RANDN(4,5,6, 2,3, 10)
                Caib = RANDN(2,3, 10,11,12)
                WCb = RANDN(4,5,6, 2,3, 12)
                result = contractions.WCa_Caib_WCb_to_WCi(WCa, Caib, WCb)
                result2 = np.einsum('xyzuva,uvaib,xyzuvb->xyzuvi', WCa, Caib, WCb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over W only:
                CWa = RANDN(4,5,6, 10)
                Caib = RANDN(10,11,12)
                CWb = RANDN(4,5,6, 12)
                result = contractions.WCa_Caib_WCb_to_WCi(CWa, Caib, CWb)
                result2 = np.einsum('xyza,aib,xyzb->xyzi', CWa, Caib, CWb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over C only:
                CWa = RANDN(2,3, 10)
                Caib = RANDN(2,3, 10,11,12)
                CWb = RANDN(2,3, 12)
                result = contractions.WCa_Caib_WCb_to_WCi(CWa, Caib, CWb)
                result2 = np.einsum('uva,uvaib,uvb->uvi', CWa, Caib, CWb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                CWa = RANDN(10)
                Caib = RANDN(10,11,12)
                CWb = RANDN(12)
                result = contractions.WCa_Caib_WCb_to_WCi(CWa, Caib, CWb)
                result2 = np.einsum('a,aib,b->i', CWa, Caib, CWb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_dWCa_dCaib_dWCb_to_dWCi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                dWCa = RANDN(8, 4,5,6, 2,3, 10)
                dCaib = RANDN(8, 2,3, 10,11,12)
                dWCb = RANDN(8, 4,5,6, 2,3, 12)
                result = contractions.dWCa_dCaib_dWCb_to_dWCi(dWCa, dCaib, dWCb)
                result2 = np.einsum('dxyzuva,duvaib,dxyzuvb->dxyzuvi', dWCa, dCaib, dWCb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over W only:
                dCWa = RANDN(8, 4,5,6, 10)
                dCaib = RANDN(8, 10,11,12)
                dCWb = RANDN(8, 4,5,6, 12)
                result = contractions.dWCa_dCaib_dWCb_to_dWCi(dCWa, dCaib, dCWb)
                result2 = np.einsum('dxyza,daib,dxyzb->dxyzi', dCWa, dCaib, dCWb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over C only:
                dCWa = RANDN(8, 2,3, 10)
                dCaib = RANDN(8, 2,3, 10,11,12)
                dCWb = RANDN(8, 2,3, 12)
                result = contractions.dWCa_dCaib_dWCb_to_dWCi(dCWa, dCaib, dCWb)
                result2 = np.einsum('duva,duvaib,duvb->duvi', dCWa, dCaib, dCWb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                dCWa = RANDN(8, 10)
                dCaib = RANDN(8, 10,11,12)
                dCWb = RANDN(8, 12)
                result = contractions.dWCa_dCaib_dWCb_to_dWCi(dCWa, dCaib, dCWb)
                result2 = np.einsum('da,daib,db->di', dCWa, dCaib, dCWb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_WCi_Cio_to_WCo(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                WCi = RANDN(2,3,4, 5,6, 10)
                Cio = RANDN(5,6, 10,13)
                result = contractions.WCi_Cio_to_WCo(WCi, Cio)
                result2 = np.einsum('xyzuvi,uvio->xyzuvo', WCi, Cio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over W only:
                CWi = RANDN(2,3,4, 10)
                Cio = RANDN(10,13)
                result = contractions.WCi_Cio_to_WCo(CWi, Cio)
                result2 = np.einsum('xyzi,io->xyzo', CWi, Cio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over C only:
                CWi = RANDN(5,6, 10)
                Cio = RANDN(5,6, 10,13)
                result = contractions.WCi_Cio_to_WCo(CWi, Cio)
                result2 = np.einsum('uvi,uvio->uvo', CWi, Cio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                CWi = RANDN(10)
                Cio = RANDN(10,13)
                result = contractions.WCi_Cio_to_WCo(CWi, Cio)
                result2 = np.einsum('i,io->o', CWi, Cio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_dWCi_dCio_to_dWCo(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over W and C:
                dWCi = RANDN(8, 2,3,4, 5,6, 10)
                dCio = RANDN(8, 5,6, 10,13)
                result = contractions.dWCi_dCio_to_dWCo(dWCi, dCio)
                result2 = np.einsum('dxyzuvi,duvio->dxyzuvo', dWCi, dCio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over W only:
                dCWi = RANDN(8, 2,3,4, 10)
                dCio = RANDN(8, 10,13)
                result = contractions.dWCi_dCio_to_dWCo(dCWi, dCio)
                result2 = np.einsum('dxyzi,dio->dxyzo', dCWi, dCio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over C only:
                dCWi = RANDN(8, 5,6, 10)
                dCio = RANDN(8, 5,6, 10,13)
                result = contractions.dWCi_dCio_to_dWCo(dCWi, dCio)
                result2 = np.einsum('duvi,duvio->duvo', dCWi, dCio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                dCWi = RANDN(8, 10)
                dCio = RANDN(8, 10,13)
                result = contractions.dWCi_dCio_to_dWCo(dCWi, dCio)
                result2 = np.einsum('di,dio->do', dCWi, dCio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def _check_3group(self, func, op_specs, out_spec, needs_n_base=False, needs_n_probe=False):
        """Check a three-group (W, K, C) contraction against an explicit np.einsum reference.

        op_specs/out_spec are (groups, singles) pairs, e.g. ('WKC', 'a') or ('KC', 'aib'); each
        grouped block is mapped to one of the THREE_GROUP_COMBOS stack shapes. needs_n_base passes
        len(C) as the trailing argument (variation-core-only forward contractions); needs_n_probe
        passes len(W) (transpose-assemble contractions). A sum-over-W contraction is expressed by an
        out_spec whose groups omit ``W`` -- np.einsum then sums it.
        """
        for RANDN in [numpy_randn, jax_randn]:
            for W, K, C in THREE_GROUP_COMBOS:
                with self.subTest(RANDN=RANDN, W=W, K=K, C=C):
                    stacks = {'W': W, 'K': K, 'C': C}
                    glet = {grp: GROUP_POOL[grp][:len(stacks[grp])] for grp in 'WKC'}

                    def sub(groups, singles):
                        return ''.join(glet[grp] for grp in groups) + singles

                    def shp(groups, singles):
                        s = ()
                        for grp in groups:
                            s = s + tuple(stacks[grp])
                        return s + tuple(SINGLE_SIZE[c] for c in singles)

                    operands = [RANDN(*shp(grp, si)) for grp, si in op_specs]
                    in_subs = ','.join(sub(grp, si) for grp, si in op_specs)
                    out_sub = sub(*out_spec)
                    ref = np.einsum(in_subs + '->' + out_sub, *operands)

                    if needs_n_base:
                        result = func(*operands, len(C))
                    elif needs_n_probe:
                        result = func(*operands, len(W))
                    else:
                        result = func(*operands)
                    result = np.asarray(result)
                    self.assertEqual(ref.shape, result.shape)
                    self.check_relerr(ref, result)

    def test_WKCa_Caib_WCi_to_WKCb(self):
        self._check_3group(
            contractions.WKCa_Caib_WCi_to_WKCb,
            [('WKC', 'a'), ('C', 'aib'), ('WC', 'i')], ('WKC', 'b'),
        )

    def test_WCa_Caib_WKCi_to_WKCb(self):
        self._check_3group(
            contractions.WCa_Caib_WKCi_to_WKCb,
            [('WC', 'a'), ('C', 'aib'), ('WKC', 'i')], ('WKC', 'b'),
        )

    def test_WKCa_Caib_WCb_to_WKCi(self):
        self._check_3group(
            contractions.WKCa_Caib_WCb_to_WKCi,
            [('WKC', 'a'), ('C', 'aib'), ('WC', 'b')], ('WKC', 'i'),
        )

    def test_WCa_Caib_WKCb_to_WKCi(self):
        self._check_3group(
            contractions.WCa_Caib_WKCb_to_WKCi,
            [('WC', 'a'), ('C', 'aib'), ('WKC', 'b')], ('WKC', 'i'),
        )

    def test_WCa_KCaib_WCi_to_WKCb(self):
        self._check_3group(
            contractions.WCa_KCaib_WCi_to_WKCb,
            [('WC', 'a'), ('KC', 'aib'), ('WC', 'i')], ('WKC', 'b'), needs_n_base=True,
        )

    def test_WCa_KCaib_WCb_to_WKCi(self):
        self._check_3group(
            contractions.WCa_KCaib_WCb_to_WKCi,
            [('WC', 'a'), ('KC', 'aib'), ('WC', 'b')], ('WKC', 'i'), needs_n_base=True,
        )

    def test_WCi_KCio_to_WKCo(self):
        self._check_3group(
            contractions.WCi_KCio_to_WKCo,
            [('WC', 'i'), ('KC', 'io')], ('WKC', 'o'), needs_n_base=True,
        )

    def test_WKCi_Cio_to_WKCo(self):
        self._check_3group(
            contractions.WKCi_Cio_to_WKCo,
            [('WKC', 'i'), ('C', 'io')], ('WKC', 'o'),
        )

    # ---- transpose-assemble outer products (keep-W and sum-W forms) ----

    def test_WKCo_WCa_to_WKCao(self):
        self._check_3group(
            contractions.WKCo_WCa_to_WKCao,
            [('WKC', 'o'), ('WC', 'a')], ('WKC', 'ao'), needs_n_probe=True,
        )

    def test_WKCo_WCa_to_KCao(self):
        self._check_3group(
            contractions.WKCo_WCa_to_KCao,
            [('WKC', 'o'), ('WC', 'a')], ('KC', 'ao'), needs_n_probe=True,
        )

    def test_Wo_WKCa_to_WKCao(self):
        self._check_3group(
            contractions.Wo_WKCa_to_WKCao,
            [('W', 'o'), ('WKC', 'a')], ('WKC', 'ao'),
        )

    def test_Wo_WKCa_to_KCao(self):
        self._check_3group(
            contractions.Wo_WKCa_to_KCao,
            [('W', 'o'), ('WKC', 'a')], ('KC', 'ao'),
        )

    def test_WCi_WCa_WKCj_to_WKCiaj(self):
        self._check_3group(
            contractions.WCi_WCa_WKCj_to_WKCiaj,
            [('WC', 'i'), ('WC', 'a'), ('WKC', 'j')], ('WKC', 'iaj'), needs_n_probe=True,
        )

    def test_WCi_WCa_WKCj_to_KCiaj(self):
        self._check_3group(
            contractions.WCi_WCa_WKCj_to_KCiaj,
            [('WC', 'i'), ('WC', 'a'), ('WKC', 'j')], ('KC', 'iaj'), needs_n_probe=True,
        )

    def test_WKCi_WCa_WCj_to_WKCiaj(self):
        self._check_3group(
            contractions.WKCi_WCa_WCj_to_WKCiaj,
            [('WKC', 'i'), ('WC', 'a'), ('WC', 'j')], ('WKC', 'iaj'), needs_n_probe=True,
        )

    def test_WKCi_WCa_WCj_to_KCiaj(self):
        self._check_3group(
            contractions.WKCi_WCa_WCj_to_KCiaj,
            [('WKC', 'i'), ('WC', 'a'), ('WC', 'j')], ('KC', 'iaj'), needs_n_probe=True,
        )

    def test_WCi_WKCa_WCj_to_WKCiaj(self):
        self._check_3group(
            contractions.WCi_WKCa_WCj_to_WKCiaj,
            [('WC', 'i'), ('WKC', 'a'), ('WC', 'j')], ('WKC', 'iaj'), needs_n_probe=True,
        )

    def test_WCi_WKCa_WCj_to_KCiaj(self):
        self._check_3group(
            contractions.WCi_WKCa_WCj_to_KCiaj,
            [('WC', 'i'), ('WKC', 'a'), ('WC', 'j')], ('KC', 'iaj'), needs_n_probe=True,
        )

    def test_WCa_WCi_WKCb_to_WKCaib(self):  # adjoint-state apply/entries dG assemble (keep W)
        self._check_3group(
            contractions.WCa_WCi_WKCb_to_WKCaib,
            [('WC', 'a'), ('WC', 'i'), ('WKC', 'b')], ('WKC', 'aib'), needs_n_probe=True,
        )

    def test_WCa_WCi_WKCb_to_KCaib(self):  # adjoint-state apply/entries dG assemble (sum W = J^T r)
        self._check_3group(
            contractions.WCa_WCi_WKCb_to_KCaib,
            [('WC', 'a'), ('WC', 'i'), ('WKC', 'b')], ('KC', 'aib'), needs_n_probe=True,
        )

    # ---- d-prefixed uniform WKC contractions (3b-6a): the core index d vectorized over a leading batch ----

    def _check_3group_d(self, func, op_specs, out_spec, needs_n_base=False, needs_n_probe=False):
        """d-prefixed twin of :py:meth:`_check_3group`: prepend a leading core-index axis ``d`` to every
        operand and the output, and verify against the d-vectorized ``np.einsum`` reference. Since a
        d-batched einsum equals the ragged contraction applied per d-slice, this IS the "per d-index ==
        ragged WKC" oracle (the ragged twins are the same op_specs, checked above)."""
        D = 3
        for RANDN in [numpy_randn, jax_randn]:
            for W, K, C in THREE_GROUP_COMBOS:
                with self.subTest(RANDN=RANDN, W=W, K=K, C=C):
                    stacks = {'W': W, 'K': K, 'C': C}
                    glet = {grp: GROUP_POOL[grp][:len(stacks[grp])] for grp in 'WKC'}

                    def sub(groups, singles):
                        return 'd' + ''.join(glet[grp] for grp in groups) + singles

                    def shp(groups, singles):
                        s = (D,)
                        for grp in groups:
                            s = s + tuple(stacks[grp])
                        return s + tuple(SINGLE_SIZE[c] for c in singles)

                    operands = [RANDN(*shp(grp, si)) for grp, si in op_specs]
                    in_subs = ','.join(sub(grp, si) for grp, si in op_specs)
                    out_sub = sub(*out_spec)
                    ref = np.einsum(in_subs + '->' + out_sub, *operands)

                    if needs_n_base:
                        result = func(*operands, len(C))
                    elif needs_n_probe:
                        result = func(*operands, len(W))
                    else:
                        result = func(*operands)
                    result = np.asarray(result)
                    self.assertEqual(ref.shape, result.shape)
                    self.check_relerr(ref, result)

    def test_dWKCa_dCaib_dWCb_to_dWKCi(self):
        self._check_3group_d(contractions.dWKCa_dCaib_dWCb_to_dWKCi,
                             [('WKC', 'a'), ('C', 'aib'), ('WC', 'b')], ('WKC', 'i'))

    def test_dWCa_dCaib_dWKCb_to_dWKCi(self):
        self._check_3group_d(contractions.dWCa_dCaib_dWKCb_to_dWKCi,
                             [('WC', 'a'), ('C', 'aib'), ('WKC', 'b')], ('WKC', 'i'))

    def test_dWCa_dKCaib_dWCb_to_dWKCi(self):
        self._check_3group_d(contractions.dWCa_dKCaib_dWCb_to_dWKCi,
                             [('WC', 'a'), ('KC', 'aib'), ('WC', 'b')], ('WKC', 'i'), needs_n_base=True)

    def test_dWKCi_dCio_to_dWKCo(self):
        self._check_3group_d(contractions.dWKCi_dCio_to_dWKCo,
                             [('WKC', 'i'), ('C', 'io')], ('WKC', 'o'))

    def test_dWCi_dKCio_to_dWKCo(self):
        self._check_3group_d(contractions.dWCi_dKCio_to_dWKCo,
                             [('WC', 'i'), ('KC', 'io')], ('WKC', 'o'), needs_n_base=True)

    def test_dWCo_dCio_to_dWCi(self):  # shared-C (compute_deta_tildes); a two-group, K unused
        self._check_3group_d(contractions.dWCo_dCio_to_dWCi,
                             [('WC', 'o'), ('C', 'io')], ('WC', 'i'))

    def test_dWKCo_dWCa_to_dWKCao(self):
        self._check_3group_d(contractions.dWKCo_dWCa_to_dWKCao,
                             [('WKC', 'o'), ('WC', 'a')], ('WKC', 'ao'), needs_n_probe=True)

    def test_dWKCo_dWCa_to_dKCao(self):
        self._check_3group_d(contractions.dWKCo_dWCa_to_dKCao,
                             [('WKC', 'o'), ('WC', 'a')], ('KC', 'ao'), needs_n_probe=True)

    def test_dWo_dWKCa_to_dWKCao(self):
        self._check_3group_d(contractions.dWo_dWKCa_to_dWKCao,
                             [('W', 'o'), ('WKC', 'a')], ('WKC', 'ao'))

    def test_dWo_dWKCa_to_dKCao(self):
        self._check_3group_d(contractions.dWo_dWKCa_to_dKCao,
                             [('W', 'o'), ('WKC', 'a')], ('KC', 'ao'))

    def test_dWCi_dWCa_dWKCj_to_dWKCiaj(self):
        self._check_3group_d(contractions.dWCi_dWCa_dWKCj_to_dWKCiaj,
                             [('WC', 'i'), ('WC', 'a'), ('WKC', 'j')], ('WKC', 'iaj'), needs_n_probe=True)

    def test_dWCi_dWCa_dWKCj_to_dKCiaj(self):
        self._check_3group_d(contractions.dWCi_dWCa_dWKCj_to_dKCiaj,
                             [('WC', 'i'), ('WC', 'a'), ('WKC', 'j')], ('KC', 'iaj'), needs_n_probe=True)

    def test_dWKCi_dWCa_dWCj_to_dWKCiaj(self):
        self._check_3group_d(contractions.dWKCi_dWCa_dWCj_to_dWKCiaj,
                             [('WKC', 'i'), ('WC', 'a'), ('WC', 'j')], ('WKC', 'iaj'), needs_n_probe=True)

    def test_dWKCi_dWCa_dWCj_to_dKCiaj(self):
        self._check_3group_d(contractions.dWKCi_dWCa_dWCj_to_dKCiaj,
                             [('WKC', 'i'), ('WC', 'a'), ('WC', 'j')], ('KC', 'iaj'), needs_n_probe=True)

    def test_dWCi_dWKCa_dWCj_to_dWKCiaj(self):
        self._check_3group_d(contractions.dWCi_dWKCa_dWCj_to_dWKCiaj,
                             [('WC', 'i'), ('WKC', 'a'), ('WC', 'j')], ('WKC', 'iaj'), needs_n_probe=True)

    def test_dWCi_dWKCa_dWCj_to_dKCiaj(self):
        self._check_3group_d(contractions.dWCi_dWKCa_dWCj_to_dKCiaj,
                             [('WC', 'i'), ('WKC', 'a'), ('WC', 'j')], ('KC', 'iaj'), needs_n_probe=True)

    def test_dWCa_dWCi_dWKCb_to_dWKCaib(self):
        self._check_3group_d(contractions.dWCa_dWCi_dWKCb_to_dWKCaib,
                             [('WC', 'a'), ('WC', 'i'), ('WKC', 'b')], ('WKC', 'aib'), needs_n_probe=True)

    def test_dWCa_dWCi_dWKCb_to_dKCaib(self):
        self._check_3group_d(contractions.dWCa_dWCi_dWKCb_to_dKCaib,
                             [('WC', 'a'), ('WC', 'i'), ('WKC', 'b')], ('KC', 'aib'), needs_n_probe=True)

    # ---- order-threaded three-group contractions (K-stacked derivative probing) ----

    def _check_jet3(self, func, op_specs, out_spec, trs_sub='trs', needs_n_base=False, needs_n_probe=False):
        """Check an order-threaded 3-group (W,K,C) contraction against an explicit np.einsum reference.

        op_specs/out_spec are (order_letters, groups, singles), where order_letters is a subset of
        ``trsu`` (each a single derivative-order axis, size ORD). ``trs_sub`` is the binomial tensor's
        subscript (e.g. ``'trs'`` forward, ``'tus'`` / ``'tru'`` for the adjoint hooks) prepended as the
        first operand; pass ``''`` for the no-trs lift/order-diagonal contractions (the order axis rides
        passively or is a plain order-diagonal sum). An ``out_spec`` whose order_letters is empty sums
        the order axis (the order-less gradient assembly). needs_n_base passes len(C) (variation-core
        terms); needs_n_probe passes len(W) (assembly terms with no pure W/C operand).
        """
        from t3toolbox.backend.probe_derivatives import binomial_combine_tensor
        ORD = 3                                       # order axis length (order=2); exercises the convolution
        OSIZE = {'t': ORD, 'r': ORD, 's': ORD, 'u': ORD}
        for RANDN in [numpy_randn, jax_randn]:
            for W, K, C in THREE_GROUP_COMBOS:
                with self.subTest(RANDN=RANDN, W=W, K=K, C=C):
                    stacks = {'W': W, 'K': K, 'C': C}
                    glet = {grp: GROUP_POOL[grp][:len(stacks[grp])] for grp in 'WKC'}

                    def sub(ords, groups, singles):
                        return ords + ''.join(glet[grp] for grp in groups) + singles

                    def shp(ords, groups, singles):
                        s = tuple(OSIZE[o] for o in ords)
                        for grp in groups:
                            s = s + tuple(stacks[grp])
                        return s + tuple(SINGLE_SIZE[c] for c in singles)

                    operands = [RANDN(*shp(*spec)) for spec in op_specs]
                    in_subs = [sub(*spec) for spec in op_specs]
                    args = list(operands)
                    if trs_sub:
                        trs_t = binomial_combine_tensor(ORD - 1)
                        operands = [trs_t] + operands
                        in_subs = [trs_sub] + in_subs
                        args = [trs_t] + args
                    ref = np.einsum(','.join(in_subs) + '->' + sub(*out_spec), *operands)
                    extra = (len(C),) if needs_n_base else (len(W),) if needs_n_probe else ()
                    result = np.asarray(func(*args, *extra))
                    self.assertEqual(ref.shape, result.shape)
                    self.check_relerr(ref, result)

    def test_trs_rWKCa_Caib_sWCi_to_tWKCb(self):
        self._check_jet3(contractions.trs_rWKCa_Caib_sWCi_to_tWKCb,
                         [('r', 'WKC', 'a'), ('', 'C', 'aib'), ('s', 'WC', 'i')], ('t', 'WKC', 'b'))

    def test_trs_rWCa_Caib_sWKCi_to_tWKCb(self):
        self._check_jet3(contractions.trs_rWCa_Caib_sWKCi_to_tWKCb,
                         [('r', 'WC', 'a'), ('', 'C', 'aib'), ('s', 'WKC', 'i')], ('t', 'WKC', 'b'))

    def test_trs_rWCa_KCaib_sWCi_to_tWKCb(self):
        self._check_jet3(contractions.trs_rWCa_KCaib_sWCi_to_tWKCb,
                         [('r', 'WC', 'a'), ('', 'KC', 'aib'), ('s', 'WC', 'i')], ('t', 'WKC', 'b'),
                         needs_n_base=True)

    def test_trs_rWKCa_Caib_sWCb_to_tWKCi(self):
        self._check_jet3(contractions.trs_rWKCa_Caib_sWCb_to_tWKCi,
                         [('r', 'WKC', 'a'), ('', 'C', 'aib'), ('s', 'WC', 'b')], ('t', 'WKC', 'i'))

    def test_trs_rWCa_Caib_sWKCb_to_tWKCi(self):
        self._check_jet3(contractions.trs_rWCa_Caib_sWKCb_to_tWKCi,
                         [('r', 'WC', 'a'), ('', 'C', 'aib'), ('s', 'WKC', 'b')], ('t', 'WKC', 'i'))

    def test_trs_rWCa_KCaib_sWCb_to_tWKCi(self):
        self._check_jet3(contractions.trs_rWCa_KCaib_sWCb_to_tWKCi,
                         [('r', 'WC', 'a'), ('', 'KC', 'aib'), ('s', 'WC', 'b')], ('t', 'WKC', 'i'),
                         needs_n_base=True)

    def test_tWKCi_Cio_to_tWKCo(self):
        self._check_jet3(contractions.tWKCi_Cio_to_tWKCo,
                         [('t', 'WKC', 'i'), ('', 'C', 'io')], ('t', 'WKC', 'o'), trs_sub="")

    def test_tWCi_KCio_to_tWKCo(self):
        self._check_jet3(contractions.tWCi_KCio_to_tWKCo,
                         [('t', 'WC', 'i'), ('', 'KC', 'io')], ('t', 'WKC', 'o'), trs_sub="",
                         needs_n_base=True)

    # ---- order-threaded 3-group ADJOINT contractions (K-stacked derivative-probe transpose) ----

    def test_trs_tWKCa_Caib_uWCi_to_sWKCb(self):
        self._check_jet3(contractions.trs_tWKCa_Caib_uWCi_to_sWKCb,
                         [('t', 'WKC', 'a'), ('', 'C', 'aib'), ('u', 'WC', 'i')], ('s', 'WKC', 'b'),
                         trs_sub='tus')

    def test_trs_rWCa_Caib_tWKCi_to_sWKCb(self):
        self._check_jet3(contractions.trs_rWCa_Caib_tWKCi_to_sWKCb,
                         [('r', 'WC', 'a'), ('', 'C', 'aib'), ('t', 'WKC', 'i')], ('s', 'WKC', 'b'),
                         trs_sub='trs')

    def test_trs_tWKCa_Caib_sWCb_to_uWKCi(self):
        self._check_jet3(contractions.trs_tWKCa_Caib_sWCb_to_uWKCi,
                         [('t', 'WKC', 'a'), ('', 'C', 'aib'), ('s', 'WC', 'b')], ('u', 'WKC', 'i'),
                         trs_sub='tus')

    def test_trs_rWCa_Caib_tWKCb_to_uWKCi(self):
        self._check_jet3(contractions.trs_rWCa_Caib_tWKCb_to_uWKCi,
                         [('r', 'WC', 'a'), ('', 'C', 'aib'), ('t', 'WKC', 'b')], ('u', 'WKC', 'i'),
                         trs_sub='tru')

    def test_tWKCo_Cio_to_tWKCi(self):
        self._check_jet3(contractions.tWKCo_Cio_to_tWKCi,
                         [('t', 'WKC', 'o'), ('', 'C', 'io')], ('t', 'WKC', 'i'), trs_sub="")

    # dG gradient assembly (order summed -> out has no order letter)
    def test_trs_rWCa_uWCi_tWKCb_to_WKCaib(self):
        self._check_jet3(contractions.trs_rWCa_uWCi_tWKCb_to_WKCaib,
                         [('r', 'WC', 'a'), ('u', 'WC', 'i'), ('t', 'WKC', 'b')], ('', 'WKC', 'aib'),
                         trs_sub='tru', needs_n_probe=True)

    def test_trs_rWCa_uWCi_tWKCb_to_KCaib(self):
        self._check_jet3(contractions.trs_rWCa_uWCi_tWKCb_to_KCaib,
                         [('r', 'WC', 'a'), ('u', 'WC', 'i'), ('t', 'WKC', 'b')], ('', 'KC', 'aib'),
                         trs_sub='tru', needs_n_probe=True)

    def test_trs_tWKCa_uWCi_sWCb_to_WKCaib(self):
        self._check_jet3(contractions.trs_tWKCa_uWCi_sWCb_to_WKCaib,
                         [('t', 'WKC', 'a'), ('u', 'WC', 'i'), ('s', 'WC', 'b')], ('', 'WKC', 'aib'),
                         trs_sub='tus', needs_n_probe=True)

    def test_trs_tWKCa_uWCi_sWCb_to_KCaib(self):
        self._check_jet3(contractions.trs_tWKCa_uWCi_sWCb_to_KCaib,
                         [('t', 'WKC', 'a'), ('u', 'WC', 'i'), ('s', 'WC', 'b')], ('', 'KC', 'aib'),
                         trs_sub='tus', needs_n_probe=True)

    def test_trs_rWCa_tWKCi_sWCb_to_WKCaib(self):
        self._check_jet3(contractions.trs_rWCa_tWKCi_sWCb_to_WKCaib,
                         [('r', 'WC', 'a'), ('t', 'WKC', 'i'), ('s', 'WC', 'b')], ('', 'WKC', 'aib'),
                         trs_sub='trs', needs_n_probe=True)

    def test_trs_rWCa_tWKCi_sWCb_to_KCaib(self):
        self._check_jet3(contractions.trs_rWCa_tWKCi_sWCb_to_KCaib,
                         [('r', 'WC', 'a'), ('t', 'WKC', 'i'), ('s', 'WC', 'b')], ('', 'KC', 'aib'),
                         trs_sub='trs', needs_n_probe=True)

    # dU gradient assembly (order-diagonal sum -> no trs operand)
    def test_tWCa_tWKCo_to_WKCao(self):
        self._check_jet3(contractions.tWCa_tWKCo_to_WKCao,
                         [('t', 'WC', 'a'), ('t', 'WKC', 'o')], ('', 'WKC', 'ao'),
                         trs_sub="", needs_n_probe=True)

    def test_tWCa_tWKCo_to_KCao(self):
        self._check_jet3(contractions.tWCa_tWKCo_to_KCao,
                         [('t', 'WC', 'a'), ('t', 'WKC', 'o')], ('', 'KC', 'ao'),
                         trs_sub="", needs_n_probe=True)

    def test_uWKCa_uWo_to_WKCao(self):
        self._check_jet3(contractions.uWKCa_uWo_to_WKCao,
                         [('u', 'WKC', 'a'), ('u', 'W', 'o')], ('', 'WKC', 'ao'), trs_sub="")

    def test_uWKCa_uWo_to_KCao(self):
        self._check_jet3(contractions.uWKCa_uWo_to_KCao,
                         [('u', 'WKC', 'a'), ('u', 'W', 'o')], ('', 'KC', 'ao'), trs_sub="")

    # ---- d-prefixed uniform JET contractions (3b-6'a) ----

    def _check_jet3_d(self, func, op_specs, out_spec, trs_sub='trs', needs_n_base=False, needs_n_probe=False):
        """d-prefixed twin of :py:meth:`_check_jet3`: prepend a leading core-index axis ``d`` to every
        jet/core operand and the output (``trs`` stays shared, no ``d``), and verify against the
        d-vectorized ``np.einsum`` reference. Uniform jet layout is ``d + order + W + K + C + (...)``, so the
        order letters sit AFTER ``d``. Since a d-batched einsum equals the ragged jet applied per d-slice,
        this IS the "per d-index == ragged trs_*" oracle (the ragged twins share op_specs, checked above)."""
        from t3toolbox.backend.probe_derivatives import binomial_combine_tensor
        D = 3
        ORD = 3                                       # order axis length (order=2); exercises the convolution
        OSIZE = {'t': ORD, 'r': ORD, 's': ORD, 'u': ORD}
        for RANDN in [numpy_randn, jax_randn]:
            for W, K, C in THREE_GROUP_COMBOS:
                with self.subTest(RANDN=RANDN, W=W, K=K, C=C):
                    stacks = {'W': W, 'K': K, 'C': C}
                    glet = {grp: GROUP_POOL[grp][:len(stacks[grp])] for grp in 'WKC'}

                    def sub(ords, groups, singles):
                        return 'd' + ords + ''.join(glet[grp] for grp in groups) + singles

                    def shp(ords, groups, singles):
                        s = (D,) + tuple(OSIZE[o] for o in ords)
                        for grp in groups:
                            s = s + tuple(stacks[grp])
                        return s + tuple(SINGLE_SIZE[c] for c in singles)

                    operands = [RANDN(*shp(*spec)) for spec in op_specs]
                    in_subs = [sub(*spec) for spec in op_specs]
                    args = list(operands)
                    if trs_sub:
                        trs_t = binomial_combine_tensor(ORD - 1)
                        operands = [trs_t] + operands
                        in_subs = [trs_sub] + in_subs
                        args = [trs_t] + args
                    ref = np.einsum(','.join(in_subs) + '->' + sub(*out_spec), *operands)
                    extra = (len(C),) if needs_n_base else (len(W),) if needs_n_probe else ()
                    result = np.asarray(func(*args, *extra))
                    self.assertEqual(ref.shape, result.shape)
                    self.check_relerr(ref, result)

    # forward combine + lift
    def test_trs_drWCa_dCaib_dsWCb_to_dtWCi(self):
        self._check_jet3_d(contractions.trs_drWCa_dCaib_dsWCb_to_dtWCi,
                           [('r', 'WC', 'a'), ('', 'C', 'aib'), ('s', 'WC', 'b')], ('t', 'WC', 'i'))

    def test_dtWCi_dCio_to_dtWCo(self):
        self._check_jet3_d(contractions.dtWCi_dCio_to_dtWCo,
                           [('t', 'WC', 'i'), ('', 'C', 'io')], ('t', 'WC', 'o'), trs_sub="")

    def test_dtWKCi_dCio_to_dtWKCo(self):
        self._check_jet3_d(contractions.dtWKCi_dCio_to_dtWKCo,
                           [('t', 'WKC', 'i'), ('', 'C', 'io')], ('t', 'WKC', 'o'), trs_sub="")

    def test_dtWCi_dKCio_to_dtWKCo(self):
        self._check_jet3_d(contractions.dtWCi_dKCio_to_dtWKCo,
                           [('t', 'WC', 'i'), ('', 'KC', 'io')], ('t', 'WKC', 'o'), trs_sub="", needs_n_base=True)

    def test_trs_drWKCa_dCaib_dsWCb_to_dtWKCi(self):
        self._check_jet3_d(contractions.trs_drWKCa_dCaib_dsWCb_to_dtWKCi,
                           [('r', 'WKC', 'a'), ('', 'C', 'aib'), ('s', 'WC', 'b')], ('t', 'WKC', 'i'))

    def test_trs_drWCa_dKCaib_dsWCb_to_dtWKCi(self):
        self._check_jet3_d(contractions.trs_drWCa_dKCaib_dsWCb_to_dtWKCi,
                           [('r', 'WC', 'a'), ('', 'KC', 'aib'), ('s', 'WC', 'b')], ('t', 'WKC', 'i'),
                           needs_n_base=True)

    def test_trs_drWCa_dCaib_dsWKCb_to_dtWKCi(self):
        self._check_jet3_d(contractions.trs_drWCa_dCaib_dsWKCb_to_dtWKCi,
                           [('r', 'WC', 'a'), ('', 'C', 'aib'), ('s', 'WKC', 'b')], ('t', 'WKC', 'i'))

    # transpose sweeps + assembly
    def test_dtWKCo_dCio_to_dtWKCi(self):
        self._check_jet3_d(contractions.dtWKCo_dCio_to_dtWKCi,
                           [('t', 'WKC', 'o'), ('', 'C', 'io')], ('t', 'WKC', 'i'), trs_sub="")

    def test_trs_dtWKCa_dCaib_dsWCb_to_duWKCi(self):
        self._check_jet3_d(contractions.trs_dtWKCa_dCaib_dsWCb_to_duWKCi,
                           [('t', 'WKC', 'a'), ('', 'C', 'aib'), ('s', 'WC', 'b')], ('u', 'WKC', 'i'),
                           trs_sub='tus')

    def test_trs_drWCa_dCaib_dtWKCb_to_duWKCi(self):
        self._check_jet3_d(contractions.trs_drWCa_dCaib_dtWKCb_to_duWKCi,
                           [('r', 'WC', 'a'), ('', 'C', 'aib'), ('t', 'WKC', 'b')], ('u', 'WKC', 'i'),
                           trs_sub='tru')

    # dG gradient assembly (order summed -> out has no order letter)
    def test_trs_drWCa_duWCi_dtWKCb_to_dWKCaib(self):
        self._check_jet3_d(contractions.trs_drWCa_duWCi_dtWKCb_to_dWKCaib,
                           [('r', 'WC', 'a'), ('u', 'WC', 'i'), ('t', 'WKC', 'b')], ('', 'WKC', 'aib'),
                           trs_sub='tru', needs_n_probe=True)

    def test_trs_drWCa_duWCi_dtWKCb_to_dKCaib(self):
        self._check_jet3_d(contractions.trs_drWCa_duWCi_dtWKCb_to_dKCaib,
                           [('r', 'WC', 'a'), ('u', 'WC', 'i'), ('t', 'WKC', 'b')], ('', 'KC', 'aib'),
                           trs_sub='tru', needs_n_probe=True)

    def test_trs_dtWKCa_duWCi_dsWCb_to_dWKCaib(self):
        self._check_jet3_d(contractions.trs_dtWKCa_duWCi_dsWCb_to_dWKCaib,
                           [('t', 'WKC', 'a'), ('u', 'WC', 'i'), ('s', 'WC', 'b')], ('', 'WKC', 'aib'),
                           trs_sub='tus', needs_n_probe=True)

    def test_trs_dtWKCa_duWCi_dsWCb_to_dKCaib(self):
        self._check_jet3_d(contractions.trs_dtWKCa_duWCi_dsWCb_to_dKCaib,
                           [('t', 'WKC', 'a'), ('u', 'WC', 'i'), ('s', 'WC', 'b')], ('', 'KC', 'aib'),
                           trs_sub='tus', needs_n_probe=True)

    def test_trs_drWCa_dtWKCi_dsWCb_to_dWKCaib(self):
        self._check_jet3_d(contractions.trs_drWCa_dtWKCi_dsWCb_to_dWKCaib,
                           [('r', 'WC', 'a'), ('t', 'WKC', 'i'), ('s', 'WC', 'b')], ('', 'WKC', 'aib'),
                           trs_sub='trs', needs_n_probe=True)

    def test_trs_drWCa_dtWKCi_dsWCb_to_dKCaib(self):
        self._check_jet3_d(contractions.trs_drWCa_dtWKCi_dsWCb_to_dKCaib,
                           [('r', 'WC', 'a'), ('t', 'WKC', 'i'), ('s', 'WC', 'b')], ('', 'KC', 'aib'),
                           trs_sub='trs', needs_n_probe=True)

    # dU gradient assembly (order-diagonal sum -> no trs operand)
    def test_dtWCa_dtWKCo_to_dWKCao(self):
        self._check_jet3_d(contractions.dtWCa_dtWKCo_to_dWKCao,
                           [('t', 'WC', 'a'), ('t', 'WKC', 'o')], ('', 'WKC', 'ao'),
                           trs_sub="", needs_n_probe=True)

    def test_dtWCa_dtWKCo_to_dKCao(self):
        self._check_jet3_d(contractions.dtWCa_dtWKCo_to_dKCao,
                           [('t', 'WC', 'a'), ('t', 'WKC', 'o')], ('', 'KC', 'ao'),
                           trs_sub="", needs_n_probe=True)

    def test_duWKCa_duWo_to_dWKCao(self):
        self._check_jet3_d(contractions.duWKCa_duWo_to_dWKCao,
                           [('u', 'WKC', 'a'), ('u', 'W', 'o')], ('', 'WKC', 'ao'), trs_sub="")

    def test_duWKCa_duWo_to_dKCao(self):
        self._check_jet3_d(contractions.duWKCa_duWo_to_dKCao,
                           [('u', 'WKC', 'a'), ('u', 'W', 'o')], ('', 'KC', 'ao'), trs_sub="")

    def test_jet_d_order0_anchor(self):
        """The 3b-6' anchor (plan lesson #3): at order 0 each d-prefixed jet contraction reduces to the
        plain dWKC contraction verified in 3b-6a. A free cross-check on top of the np.einsum oracle above:
        it catches order-axis bookkeeping slips (a stray order slot in the wrong place). Slice all order
        axes to index 0 and compare against the plain d-prefixed twin on the order-0 operand slices."""
        from t3toolbox.backend.probe_derivatives import binomial_combine_tensor
        D, W, K, C = 3, (2,), (4,), (5,)
        a, i, b, o = 6, 7, 8, 9
        trs = binomial_combine_tensor(2)              # (t,r,s) = (3,3,3)
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # forward combine term1: trs_drWKCa_dCaib_dsWCb_to_dtWKCi  ==0==  dWKCa_dCaib_dWCb_to_dWKCi
                drWKCa = RANDN(D, 3, *W, *K, *C, a); dCaib = RANDN(D, *C, a, i, b); dsWCb = RANDN(D, 3, *W, *C, b)
                jet = np.asarray(contractions.trs_drWKCa_dCaib_dsWCb_to_dtWKCi(trs, drWKCa, dCaib, dsWCb))
                plain = np.asarray(contractions.dWKCa_dCaib_dWCb_to_dWKCi(
                    np.asarray(drWKCa)[:, 0], dCaib, np.asarray(dsWCb)[:, 0]))
                self.check_relerr(plain, jet[:, 0])

                # dG assembly (sum W): trs_drWCa_duWCi_dtWKCb_to_dKCaib  ==0==  dWCa_dWCi_dWKCb_to_dKCaib.
                # The assembly sums all orders (r+u=t), so zero orders >0 -> only the r=u=t=0 term survives.
                drWCa = np.array(RANDN(D, 3, *W, *C, a)); duWCi = np.array(RANDN(D, 3, *W, *C, i))
                dtWKCb = np.array(RANDN(D, 3, *W, *K, *C, b))
                drWCa[:, 1:] = 0.0; duWCi[:, 1:] = 0.0; dtWKCb[:, 1:] = 0.0
                jetG = np.asarray(contractions.trs_drWCa_duWCi_dtWKCb_to_dKCaib(trs, drWCa, duWCi, dtWKCb, len(W)))
                plainG = np.asarray(contractions.dWCa_dWCi_dWKCb_to_dKCaib(
                    drWCa[:, 0], duWCi[:, 0], dtWKCb[:, 0], len(W)))
                self.check_relerr(plainG, jetG)


