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
GROUP_POOL = {'F': 'fgh', 'V': 'vwx', 'G': 'mnp'}
SINGLE_SIZE = {'a': 3, 'i': 4, 'b': 5, 'o': 6}

# (F, V, G) stack shapes exercised by the three-group contractions: all present, V/F multi-axis,
# each block empty in turn (V empty is the degenerate two-group reduction), and all empty.
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

    def test_FGa_Gaib_Fo_Gio_to_FGb(self):
        # FG twin of the apply contraction (base-inner: F outer, G inner).
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                xyz_uv_a = RANDN(2,3,4, 5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                xyz_o = RANDN(2,3,4, 13)
                uv_io = RANDN(5,6, 11,13)
                result = contractions.FGa_Gaib_Fo_Gio_to_FGb(xyz_uv_a, uv_aib, xyz_o, uv_io)
                result_true = np.einsum('xyzuva,uvaib,xyzo,uvio->xyzuvb', xyz_uv_a, uv_aib, xyz_o, uv_io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over F only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                xyz_o = RANDN(2,3,4, 13)
                io = RANDN(11,13)
                result = contractions.FGa_Gaib_Fo_Gio_to_FGb(xyz_a, aib, xyz_o, io)
                result_true = np.einsum('xyza,aib,xyzo,io->xyzb', xyz_a, aib, xyz_o, io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over G only:
                uv_a = RANDN(5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                o = RANDN(13)
                uv_io = RANDN(5,6, 11,13)
                result = contractions.FGa_Gaib_Fo_Gio_to_FGb(uv_a, uv_aib, o, uv_io)
                result_true = np.einsum('uva,uvaib,o,uvio->uvb', uv_a, uv_aib, o, uv_io)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                o = RANDN(13)
                io = RANDN(11,13)
                result = contractions.FGa_Gaib_Fo_Gio_to_FGb(a, aib, o, io)
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

    def test_FGa_Gaib_FGi_to_FGb(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                xyz_uv_a = RANDN(2,3,4, 5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                xyz_uv_i = RANDN(2,3,4, 5,6, 11)
                result = contractions.FGa_Gaib_FGi_to_FGb(xyz_uv_a, uv_aib, xyz_uv_i)
                result_true = np.einsum('xyzuva,uvaib,xyzuvi->xyzuvb', xyz_uv_a, uv_aib, xyz_uv_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over F only:
                xyz_a = RANDN(2,3,4, 10)
                aib = RANDN(10,11,12)
                xyz_i = RANDN(2,3,4, 11)
                result = contractions.FGa_Gaib_FGi_to_FGb(xyz_a, aib, xyz_i)
                result_true = np.einsum('xyza,aib,xyzi->xyzb', xyz_a, aib, xyz_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over G only:
                uv_a = RANDN(5,6, 10)
                uv_aib = RANDN(5,6, 10,11,12)
                uv_i = RANDN(5,6, 11)
                result = contractions.FGa_Gaib_FGi_to_FGb(uv_a, uv_aib, uv_i)
                result_true = np.einsum('uva,uvaib,uvi->uvb', uv_a, uv_aib, uv_i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # No vectorization:
                a = RANDN(10)
                aib = RANDN(10,11,12)
                i = RANDN(11)
                result = contractions.FGa_Gaib_FGi_to_FGb(a, aib, i)
                result_true = np.einsum('a,aib,i->b', a, aib, i)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_Gio_Fo_to_FGi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                Gio = RANDN(5,6, 10,13)
                Fo = RANDN(2,3,4, 13)
                result = contractions.Gio_Fo_to_FGi(Gio, Fo)
                result_true = np.einsum('uvio,xyzo->xyzuvi', Gio, Fo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over F only:
                Gio = RANDN(10,13)
                Fo = RANDN(2,3,4, 13)
                result = contractions.Gio_Fo_to_FGi(Gio, Fo)
                result_true = np.einsum('io,xyzo->xyzi', Gio, Fo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Vectorize over G only:
                Gio = RANDN(5,6, 10,13)
                Fo = RANDN(13)
                result = contractions.Gio_Fo_to_FGi(Gio, Fo)
                result_true = np.einsum('uvio,o->uvi', Gio, Fo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

                # Fo vectorization:
                Gio = RANDN(10,13)
                Fo = RANDN(13)
                result = contractions.Gio_Fo_to_FGi(Gio, Fo)
                result_true = np.einsum('io,o->i', Gio, Fo)
                self.assertEqual(result_true.shape, result.shape)
                self.check_relerr(result_true, result)

    def test_dGio_dFo_to_dFGi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                dGio = RANDN(8, 5,6, 10,13)
                dFo = RANDN(8, 2,3,4, 13)
                result = contractions.dGio_dFo_to_dFGi(dGio, dFo)
                result2 = np.einsum('duvio,dxyzo->dxyzuvi', dGio, dFo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over F only:
                dGio = RANDN(8, 10,13)
                dFo = RANDN(8, 2,3,4, 13)
                result = contractions.dGio_dFo_to_dFGi(dGio, dFo)
                result2 = np.einsum('dio,dxyzo->dxyzi', dGio, dFo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over G only:
                dGio = RANDN(8, 5,6, 10,13)
                dFo = RANDN(8, 13)
                result = contractions.dGio_dFo_to_dFGi(dGio, dFo)
                result2 = np.einsum('duvio,do->duvi', dGio, dFo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                dGio = RANDN(8, 10,13)
                dFo = RANDN(8, 13)
                result = contractions.dGio_dFo_to_dFGi(dGio, dFo)
                result2 = np.einsum('dio,do->di', dGio, dFo)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_FGa_Gaib_FGb_to_FGi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                FGa = RANDN(4,5,6, 2,3, 10)
                Gaib = RANDN(2,3, 10,11,12)
                FGb = RANDN(4,5,6, 2,3, 12)
                result = contractions.FGa_Gaib_FGb_to_FGi(FGa, Gaib, FGb)
                result2 = np.einsum('xyzuva,uvaib,xyzuvb->xyzuvi', FGa, Gaib, FGb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over F only:
                GFa = RANDN(4,5,6, 10)
                Gaib = RANDN(10,11,12)
                GFb = RANDN(4,5,6, 12)
                result = contractions.FGa_Gaib_FGb_to_FGi(GFa, Gaib, GFb)
                result2 = np.einsum('xyza,aib,xyzb->xyzi', GFa, Gaib, GFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over G only:
                GFa = RANDN(2,3, 10)
                Gaib = RANDN(2,3, 10,11,12)
                GFb = RANDN(2,3, 12)
                result = contractions.FGa_Gaib_FGb_to_FGi(GFa, Gaib, GFb)
                result2 = np.einsum('uva,uvaib,uvb->uvi', GFa, Gaib, GFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                GFa = RANDN(10)
                Gaib = RANDN(10,11,12)
                GFb = RANDN(12)
                result = contractions.FGa_Gaib_FGb_to_FGi(GFa, Gaib, GFb)
                result2 = np.einsum('a,aib,b->i', GFa, Gaib, GFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_dFGa_dGaib_dFGb_to_dFGi(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                dFGa = RANDN(8, 4,5,6, 2,3, 10)
                dGaib = RANDN(8, 2,3, 10,11,12)
                dFGb = RANDN(8, 4,5,6, 2,3, 12)
                result = contractions.dFGa_dGaib_dFGb_to_dFGi(dFGa, dGaib, dFGb)
                result2 = np.einsum('dxyzuva,duvaib,dxyzuvb->dxyzuvi', dFGa, dGaib, dFGb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over F only:
                dGFa = RANDN(8, 4,5,6, 10)
                dGaib = RANDN(8, 10,11,12)
                dGFb = RANDN(8, 4,5,6, 12)
                result = contractions.dFGa_dGaib_dFGb_to_dFGi(dGFa, dGaib, dGFb)
                result2 = np.einsum('dxyza,daib,dxyzb->dxyzi', dGFa, dGaib, dGFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over G only:
                dGFa = RANDN(8, 2,3, 10)
                dGaib = RANDN(8, 2,3, 10,11,12)
                dGFb = RANDN(8, 2,3, 12)
                result = contractions.dFGa_dGaib_dFGb_to_dFGi(dGFa, dGaib, dGFb)
                result2 = np.einsum('duva,duvaib,duvb->duvi', dGFa, dGaib, dGFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                dGFa = RANDN(8, 10)
                dGaib = RANDN(8, 10,11,12)
                dGFb = RANDN(8, 12)
                result = contractions.dFGa_dGaib_dFGb_to_dFGi(dGFa, dGaib, dGFb)
                result2 = np.einsum('da,daib,db->di', dGFa, dGaib, dGFb)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_FGi_Gio_to_FGo(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                FGi = RANDN(2,3,4, 5,6, 10)
                Gio = RANDN(5,6, 10,13)
                result = contractions.FGi_Gio_to_FGo(FGi, Gio)
                result2 = np.einsum('xyzuvi,uvio->xyzuvo', FGi, Gio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over F only:
                GFi = RANDN(2,3,4, 10)
                Gio = RANDN(10,13)
                result = contractions.FGi_Gio_to_FGo(GFi, Gio)
                result2 = np.einsum('xyzi,io->xyzo', GFi, Gio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over G only:
                GFi = RANDN(5,6, 10)
                Gio = RANDN(5,6, 10,13)
                result = contractions.FGi_Gio_to_FGo(GFi, Gio)
                result2 = np.einsum('uvi,uvio->uvo', GFi, Gio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                GFi = RANDN(10)
                Gio = RANDN(10,13)
                result = contractions.FGi_Gio_to_FGo(GFi, Gio)
                result2 = np.einsum('i,io->o', GFi, Gio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def test_dFGi_dGio_to_dFGo(self):
        for RANDN in [numpy_randn, jax_randn]:
            with self.subTest(RANDN=RANDN):
                # Vectorize over F and G:
                dFGi = RANDN(8, 2,3,4, 5,6, 10)
                dGio = RANDN(8, 5,6, 10,13)
                result = contractions.dFGi_dGio_to_dFGo(dFGi, dGio)
                result2 = np.einsum('dxyzuvi,duvio->dxyzuvo', dFGi, dGio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over F only:
                dGFi = RANDN(8, 2,3,4, 10)
                dGio = RANDN(8, 10,13)
                result = contractions.dFGi_dGio_to_dFGo(dGFi, dGio)
                result2 = np.einsum('dxyzi,dio->dxyzo', dGFi, dGio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # Vectorize over G only:
                dGFi = RANDN(8, 5,6, 10)
                dGio = RANDN(8, 5,6, 10,13)
                result = contractions.dFGi_dGio_to_dFGo(dGFi, dGio)
                result2 = np.einsum('duvi,duvio->duvo', dGFi, dGio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

                # No vectorization:
                dGFi = RANDN(8, 10)
                dGio = RANDN(8, 10,13)
                result = contractions.dFGi_dGio_to_dFGo(dGFi, dGio)
                result2 = np.einsum('di,dio->do', dGFi, dGio)
                self.assertEqual(result2.shape, result.shape)
                self.check_relerr(result2, result)

    def _check_3group(self, func, op_specs, out_spec, needs_n_base=False):
        """Check a three-group (F, V, G) contraction against an explicit np.einsum reference.

        op_specs/out_spec are (groups, singles) pairs, e.g. ('FVG', 'a') or ('VG', 'aib'); each
        grouped block is mapped to one of the THREE_GROUP_COMBOS stack shapes. needs_n_base passes
        len(G) as the trailing argument (for the variation-core-only contractions).
        """
        for RANDN in [numpy_randn, jax_randn]:
            for F, V, G in THREE_GROUP_COMBOS:
                with self.subTest(RANDN=RANDN, F=F, V=V, G=G):
                    stacks = {'F': F, 'V': V, 'G': G}
                    glet = {grp: GROUP_POOL[grp][:len(stacks[grp])] for grp in 'FVG'}

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
                        result = func(*operands, len(G))
                    else:
                        result = func(*operands)
                    result = np.asarray(result)
                    self.assertEqual(ref.shape, result.shape)
                    self.check_relerr(ref, result)

    def test_FVGa_Gaib_FGi_to_FVGb(self):
        self._check_3group(
            contractions.FVGa_Gaib_FGi_to_FVGb,
            [('FVG', 'a'), ('G', 'aib'), ('FG', 'i')], ('FVG', 'b'),
        )

    def test_FGa_Gaib_FVGi_to_FVGb(self):
        self._check_3group(
            contractions.FGa_Gaib_FVGi_to_FVGb,
            [('FG', 'a'), ('G', 'aib'), ('FVG', 'i')], ('FVG', 'b'),
        )

    def test_FVGa_Gaib_FGb_to_FVGi(self):
        self._check_3group(
            contractions.FVGa_Gaib_FGb_to_FVGi,
            [('FVG', 'a'), ('G', 'aib'), ('FG', 'b')], ('FVG', 'i'),
        )

    def test_FGa_Gaib_FVGb_to_FVGi(self):
        self._check_3group(
            contractions.FGa_Gaib_FVGb_to_FVGi,
            [('FG', 'a'), ('G', 'aib'), ('FVG', 'b')], ('FVG', 'i'),
        )

    def test_FGa_VGaib_FGi_to_FVGb(self):
        self._check_3group(
            contractions.FGa_VGaib_FGi_to_FVGb,
            [('FG', 'a'), ('VG', 'aib'), ('FG', 'i')], ('FVG', 'b'), needs_n_base=True,
        )

    def test_FGa_VGaib_FGb_to_FVGi(self):
        self._check_3group(
            contractions.FGa_VGaib_FGb_to_FVGi,
            [('FG', 'a'), ('VG', 'aib'), ('FG', 'b')], ('FVG', 'i'), needs_n_base=True,
        )

    def test_FGi_VGio_to_FVGo(self):
        self._check_3group(
            contractions.FGi_VGio_to_FVGo,
            [('FG', 'i'), ('VG', 'io')], ('FVG', 'o'), needs_n_base=True,
        )

    def test_FVGi_Gio_to_FVGo(self):
        self._check_3group(
            contractions.FVGi_Gio_to_FVGo,
            [('FVG', 'i'), ('G', 'io')], ('FVG', 'o'),
        )


