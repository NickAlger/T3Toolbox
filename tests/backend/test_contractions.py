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

    def test_Fa_Gaib_Fi_to_FGb(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                xyz_a = RANDN(2,3,4, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                xyz_i = RANDN(2,3,4, 11)
                result = contractions.Fa_Gaib_Fi_to_FGb(xyz_a, uv_aib, xyz_i)
                result_true = np.einsum('xyza,uvaib,xyzi->xyzuvb', xyz_a, uv_aib, xyz_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over F only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                xyz_i = RANDN(2,3,4, 11)
                result = contractions.Fa_Gaib_Fi_to_FGb(xyz_a, aib, xyz_i)
                result_true = np.einsum('xyza,aib,xyzi->xyzb', xyz_a, aib, xyz_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over G only:
                a = RANDN(10)
                uv_aib = RANDN(5,6, 10,11,12)
                i = RANDN(11)
                result = contractions.Fa_Gaib_Fi_to_FGb(a, uv_aib, i)
                result_true = np.einsum('a,uvaib,i->uvb', a, uv_aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                i = RANDN(11)
                result = contractions.Fa_Gaib_Fi_to_FGb(a, aib, i)
                result_true = np.einsum('a,aib,i->b', a, aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_GFa_Gaib_Fo_Gio_to_GFb(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                uv_xyz_a = RANDN(5,6, 2,3,4, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                xyz_o = RANDN(2,3,4, 13)
                uv_io = RANDN(5,6, 11,13)
                result = contractions.GFa_Gaib_Fo_Gio_to_GFb(uv_xyz_a, uv_aib, xyz_o, uv_io)
                result_true = np.einsum('uvxyza,uvaib,xyzo,uvio->uvxyzb', uv_xyz_a, uv_aib, xyz_o, uv_io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over F only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                xyz_o = RANDN(2,3,4, 13)
                io = RANDN(11,13)
                result = contractions.GFa_Gaib_Fo_Gio_to_GFb(xyz_a, aib, xyz_o, io)
                result_true = np.einsum('xyza,aib,xyzo,io->xyzb', xyz_a, aib, xyz_o, io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over G only:
                uv_a = RANDN(5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                o = RANDN(13)
                uv_io = RANDN(5,6, 11,13)
                result = contractions.GFa_Gaib_Fo_Gio_to_GFb(uv_a, uv_aib, o, uv_io)
                result_true = np.einsum('uva,uvaib,o,uvio->uvb', uv_a, uv_aib, o, uv_io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                o = RANDN(13)
                io = RANDN(11,13)
                result = contractions.GFa_Gaib_Fo_Gio_to_GFb(a, aib, o, io)
                result_true = np.einsum('a,aib,o,io->b', a, aib, o, io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_GFa_Gaib_GiF_to_GFb(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                uv_xyz_a = RANDN(5,6, 2,3,4, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                uv_i_xyz = RANDN(5,6, 11, 2,3,4)
                result = contractions.GFa_Gaib_GiF_to_GFb(uv_xyz_a, uv_aib, uv_i_xyz)
                result_true = np.einsum('uvxyza,uvaib,uvixyz->uvxyzb', uv_xyz_a, uv_aib, uv_i_xyz)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over F only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                i_xyz = RANDN(11, 2,3,4)
                result = contractions.GFa_Gaib_GiF_to_GFb(xyz_a, aib, i_xyz)
                result_true = np.einsum('xyza,aib,ixyz->xyzb', xyz_a, aib, i_xyz)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over G only:
                uv_a = RANDN(5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                uv_i = RANDN(5,6, 11)
                result = contractions.GFa_Gaib_GiF_to_GFb(uv_a, uv_aib, uv_i)
                result_true = np.einsum('uva,uvaib,uvi->uvb', uv_a, uv_aib, uv_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                i = RANDN(11)
                result = contractions.GFa_Gaib_GiF_to_GFb(a, aib, i)
                result_true = np.einsum('a,aib,i->b', a, aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_GFa_Gaib_GFi_to_GFb(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                uv_xyz_a = RANDN(5,6,  2,3,4, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                uv_xyz_i = RANDN(5,6, 2,3,4, 11)
                result = contractions.GFa_Gaib_GFi_to_GFb(uv_xyz_a, uv_aib, uv_xyz_i)
                result_true = np.einsum('uvxyza,uvaib,uvxyzi->uvxyzb', uv_xyz_a, uv_aib, uv_xyz_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over F only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                xyz_i = RANDN(2,3,4, 11)
                result = contractions.GFa_Gaib_GFi_to_GFb(xyz_a, aib, xyz_i)
                result_true = np.einsum('xyza,aib,xyzi->xyzb', xyz_a, aib, xyz_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over G only:
                uv_a = RANDN(5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                uv_i = RANDN(5,6, 11)
                result = contractions.GFa_Gaib_GFi_to_GFb(uv_a, uv_aib, uv_i)
                result_true = np.einsum('uva,uvaib,uvi->uvb', uv_a, uv_aib, uv_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                i = RANDN(11)
                result = contractions.GFa_Gaib_GFi_to_GFb(a, aib, i)
                result_true = np.einsum('a,aib,i->b', a, aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_Gio_Fo_to_GFi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                Gio = RANDN(5,6, 10,13)
                Fo = RANDN(2,3,4, 13)
                result = contractions.Gio_Fo_to_GFi(Gio, Fo)
                result_true = np.einsum('uvio,xyzo->uvxyzi', Gio, Fo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over F only:
                Gio = RANDN(10,13)
                Fo = RANDN(2,3,4, 13)
                result = contractions.Gio_Fo_to_GFi(Gio, Fo)
                result_true = np.einsum('io,xyzo->xyzi', Gio, Fo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over G only:
                Gio = RANDN(5,6, 10,13)
                Fo = RANDN(13)
                result = contractions.Gio_Fo_to_GFi(Gio, Fo)
                result_true = np.einsum('uvio,o->uvi', Gio, Fo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Fo vectorization:
                Gio = RANDN(10,13)
                Fo = RANDN(13)
                result = contractions.Gio_Fo_to_GFi(Gio, Fo)
                result_true = np.einsum('io,o->i', Gio, Fo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_dGio_dFo_to_dGFi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                dGio = RANDN(8, 5,6, 10,13)
                dFo = RANDN(8, 2,3,4, 13)
                result = contractions.dGio_dFo_to_dGFi(dGio, dFo)
                result2 = np.einsum('duvio,dxyzo->duvxyzi', dGio, dFo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over F only:
                dGio = RANDN(8, 10,13)
                dFo = RANDN(8, 2,3,4, 13)
                result = contractions.dGio_dFo_to_dGFi(dGio, dFo)
                result2 = np.einsum('dio,dxyzo->dxyzi', dGio, dFo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over G only:
                dGio = RANDN(8, 5,6, 10,13)
                dFo = RANDN(8, 13)
                result = contractions.dGio_dFo_to_dGFi(dGio, dFo)
                result2 = np.einsum('duvio,do->duvi', dGio, dFo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                dGio = RANDN(8, 10,13)
                dFo = RANDN(8, 13)
                result = contractions.dGio_dFo_to_dGFi(dGio, dFo)
                result2 = np.einsum('dio,do->di', dGio, dFo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_GFa_Gaib_GFb_to_GFi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                GFa = RANDN(2,3, 4,5,6, 10)
                Gaib = RANDN(2,3, 10,11,12)
                GFb = RANDN(2,3, 4,5,6, 12)
                result = contractions.GFa_Gaib_GFb_to_GFi(GFa, Gaib, GFb)
                result2 = np.einsum('uvxyza,uvaib,uvxyzb->uvxyzi', GFa, Gaib, GFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over F only:
                GFa = RANDN(4,5,6, 10)
                Gaib = RANDN(10,11,12)
                GFb = RANDN(4,5,6, 12)
                result = contractions.GFa_Gaib_GFb_to_GFi(GFa, Gaib, GFb)
                result2 = np.einsum('xyza,aib,xyzb->xyzi', GFa, Gaib, GFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over G only:
                GFa = RANDN(2,3, 10)
                Gaib = RANDN(2,3, 10,11,12)
                GFb = RANDN(2,3, 12)
                result = contractions.GFa_Gaib_GFb_to_GFi(GFa, Gaib, GFb)
                result2 = np.einsum('uva,uvaib,uvb->uvi', GFa, Gaib, GFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                GFa = RANDN(10)
                Gaib = RANDN(10,11,12)
                GFb = RANDN(12)
                result = contractions.GFa_Gaib_GFb_to_GFi(GFa, Gaib, GFb)
                result2 = np.einsum('a,aib,b->i', GFa, Gaib, GFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_dGFa_dGaib_dGFb_to_dGFi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                dGFa = RANDN(8, 2,3, 4,5,6, 10)
                dGaib = RANDN(8, 2,3, 10,11,12)
                dGFb = RANDN(8, 2,3, 4,5,6, 12)
                result = contractions.dGFa_dGaib_dGFb_to_dGFi(dGFa, dGaib, dGFb)
                result2 = np.einsum('duvxyza,duvaib,duvxyzb->duvxyzi', dGFa, dGaib, dGFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over F only:
                dGFa = RANDN(8, 4,5,6, 10)
                dGaib = RANDN(8, 10,11,12)
                dGFb = RANDN(8, 4,5,6, 12)
                result = contractions.dGFa_dGaib_dGFb_to_dGFi(dGFa, dGaib, dGFb)
                result2 = np.einsum('dxyza,daib,dxyzb->dxyzi', dGFa, dGaib, dGFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over G only:
                dGFa = RANDN(8, 2,3, 10)
                dGaib = RANDN(8, 2,3, 10,11,12)
                dGFb = RANDN(8, 2,3, 12)
                result = contractions.dGFa_dGaib_dGFb_to_dGFi(dGFa, dGaib, dGFb)
                result2 = np.einsum('duva,duvaib,duvb->duvi', dGFa, dGaib, dGFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                dGFa = RANDN(8, 10)
                dGaib = RANDN(8, 10,11,12)
                dGFb = RANDN(8, 12)
                result = contractions.dGFa_dGaib_dGFb_to_dGFi(dGFa, dGaib, dGFb)
                result2 = np.einsum('da,daib,db->di', dGFa, dGaib, dGFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_GFi_Gio_to_GFo(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                GFi = RANDN(5,6, 2,3,4, 10)
                Gio = RANDN(5,6, 10,13)
                result = contractions.GFi_Gio_to_GFo(GFi, Gio)
                result2 = np.einsum('uvxyzi,uvio->uvxyzo', GFi, Gio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over F only:
                GFi = RANDN(2,3,4, 10)
                Gio = RANDN(10,13)
                result = contractions.GFi_Gio_to_GFo(GFi, Gio)
                result2 = np.einsum('xyzi,io->xyzo', GFi, Gio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over G only:
                GFi = RANDN(5,6, 10)
                Gio = RANDN(5,6, 10,13)
                result = contractions.GFi_Gio_to_GFo(GFi, Gio)
                result2 = np.einsum('uvi,uvio->uvo', GFi, Gio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                GFi = RANDN(10)
                Gio = RANDN(10,13)
                result = contractions.GFi_Gio_to_GFo(GFi, Gio)
                result2 = np.einsum('i,io->o', GFi, Gio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_dGFi_dGio_to_dGFo(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                dGFi = RANDN(8, 5,6, 2,3,4, 10)
                dGio = RANDN(8, 5,6, 10,13)
                result = contractions.dGFi_dGio_to_dGFo(dGFi, dGio)
                result2 = np.einsum('duvxyzi,duvio->duvxyzo', dGFi, dGio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over F only:
                dGFi = RANDN(8, 2,3,4, 10)
                dGio = RANDN(8, 10,13)
                result = contractions.dGFi_dGio_to_dGFo(dGFi, dGio)
                result2 = np.einsum('dxyzi,dio->dxyzo', dGFi, dGio)
                print(result.shape == result2.shape)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over G only:
                dGFi = RANDN(8, 5,6, 10)
                dGio = RANDN(8, 5,6, 10,13)
                result = contractions.dGFi_dGio_to_dGFo(dGFi, dGio)
                result2 = np.einsum('duvi,duvio->duvo', dGFi, dGio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                dGFi = RANDN(8, 10)
                dGio = RANDN(8, 10,13)
                result = contractions.dGFi_dGio_to_dGFo(dGFi, dGio)
                result2 = np.einsum('di,dio->do', dGFi, dGio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)


