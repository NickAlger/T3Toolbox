# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3toolbox.corewise as cw
import t3toolbox.orthogonalization as orth
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.probing as t3p

try:
    import t3toolbox.jax.probing as t3p_jax
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    probing_jax = t3p

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestProbing(unittest.TestCase):
    def check_relerr(self, xtrue, x):
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def test_probe_dense1(self):
        shapes = [
            (10, 11, 12),
        ]

        for SHAPE in shapes:
            for T3P in [t3p, t3p_jax]:
                with self.subTest(T3P=T3P, SHAPE=SHAPE):
                    T = np.random.randn(*SHAPE)
                    u0 = np.random.randn(SHAPE[0])
                    u1 = np.random.randn(SHAPE[1])
                    u2 = np.random.randn(SHAPE[2])

                    yy = T3P.probe_dense(T, (u0, u1, u2))

                    y0 = np.einsum('ijk,j,k', T, u1, u2)
                    y1 = np.einsum('ijk,i,k', T, u0, u2)
                    y2 = np.einsum('ijk,i,j', T, u0, u1)
                    self.check_relerr(y0, yy[0])
                    self.check_relerr(y1, yy[1])
                    self.check_relerr(y2, yy[2])

    def test_probe_dense2(self):
        shapes = [
            (10, 11, 12),
        ]

        for SHAPE in shapes:
            for T3P in [t3p, t3p_jax]:
                with self.subTest(T3P=T3P, SHAPE=SHAPE):
                    T = np.random.randn(*SHAPE)
                    u0, v0 = np.random.randn(SHAPE[0]), np.random.randn(SHAPE[0])
                    u1, v1 = np.random.randn(SHAPE[1]), np.random.randn(SHAPE[1])
                    u2, v2 = np.random.randn(SHAPE[2]), np.random.randn(SHAPE[2])
                    uuu = [np.vstack([u0, v0]), np.vstack([u1, v1]), np.vstack([u2, v2])]

                    yyy = T3P.probe_dense(T, uuu)

                    yy_u = t3p.probe_dense(T, (u0, u1, u2))
                    yy_v = t3p.probe_dense(T, (v0, v1, v2))

                    self.check_relerr(yy_u[0], yyy[0][0, :])
                    self.check_relerr(yy_u[1], yyy[1][0, :])
                    self.check_relerr(yy_u[2], yyy[2][0, :])

                    self.check_relerr(yy_v[0], yyy[0][1, :])
                    self.check_relerr(yy_v[1], yyy[1][1, :])
                    self.check_relerr(yy_v[2], yyy[2][1, :])

    def test_probe_t3_1(self):
        structures = [
            ((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)),
        ]

        for STRUCTURE in structures:
            for T3P in [t3p, t3p_jax]:
                with self.subTest(T3P=T3P, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)

                    SHAPE = STRUCTURE[0]
                    ww = (np.random.randn(SHAPE[0]),
                          np.random.randn(SHAPE[1]),
                          np.random.randn(SHAPE[2]))

                    zz = T3P.probe_t3(x, ww)

                    x_dense = t3.t3_to_dense(x)
                    zz2 = t3p.probe_dense(x_dense, ww)

                    for z, z2 in zip(zz, zz2):
                        self.check_relerr(z2, z)

    def test_probe_t3_2(self):
        structures = [
            ((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)),
        ]
        NUM_PROBES = 2

        for STRUCTURE in structures:
            for T3P in [t3p, t3p_jax]:
                with self.subTest(T3P=T3P, STRUCTURE=STRUCTURE):
                    x = t3.t3_corewise_randn(STRUCTURE)

                    SHAPE = STRUCTURE[0]
                    www = (np.random.randn(NUM_PROBES, SHAPE[0]),
                           np.random.randn(NUM_PROBES, SHAPE[1]),
                           np.random.randn(NUM_PROBES, SHAPE[2]))

                    zzz = T3P.probe_t3(x, www)

                    zzz2 = t3p.probe_dense(t3.t3_to_dense(x), www)

                    for z, z2 in zip(zzz, zzz2):
                        self.check_relerr(z2, z)

    def test_probe_tangent(self):
        structures = [
            ((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)),
        ]

        for STRUCTURE in structures:
            for T3P in [t3p, t3p_jax]:
                with self.subTest(T3P=T3P, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, _ = orth.orthogonal_representations(p)
                    variation = t3m.tangent_randn(base)

                    SHAPE = STRUCTURE[0]
                    ww = (np.random.randn(SHAPE[0]),
                          np.random.randn(SHAPE[1]),
                          np.random.randn(SHAPE[2]))

                    zz = T3P.probe_tangent(variation, ww, base)

                    zz2 = t3p.probe_dense(t3m.tangent_to_dense(variation, base), ww)

                    for z, z2 in zip(zz, zz2):
                        self.check_relerr(z2, z)

    def test_probe_tangent2(self):
        structures = [
            ((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)),
        ]
        NUM_PROBES = 2

        for STRUCTURE in structures:
            for T3P in [t3p, t3p_jax]:
                with self.subTest(T3P=T3P, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, _ = orth.orthogonal_representations(p)
                    variation = t3m.tangent_randn(base)

                    SHAPE = STRUCTURE[0]
                    www = (np.random.randn(NUM_PROBES, SHAPE[0]),
                           np.random.randn(NUM_PROBES, SHAPE[1]),
                           np.random.randn(NUM_PROBES, SHAPE[2]))

                    zzz = T3P.probe_tangent(variation, www, base)  # Compute probes!

                    zzz2 = t3p.probe_dense(t3m.tangent_to_dense(variation, base), www)

                    for zz, zz2 in zip(zzz, zzz2):
                        self.check_relerr(zz2, zz)

    def test_probe_tangent_transpose1(self):
        structures = [
            ((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)),
        ]

        for STRUCTURE in structures:
            for T3P in [t3p, t3p_jax]:
                with self.subTest(T3P=T3P, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, _ = orth.orthogonal_representations(p)

                    SHAPE = STRUCTURE[0]
                    ww = (np.random.randn(SHAPE[0]),
                          np.random.randn(SHAPE[1]),
                          np.random.randn(SHAPE[2]))

                    v1 = t3m.tangent_randn(base)

                    zz1 = T3P.probe_tangent(v1, ww, base)

                    zz2 = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
                    v2 = t3p.probe_tangent_transpose(zz2, ww, base)
                    ipA = cw.corewise_dot(v1, v2)
                    ipB = cw.corewise_dot(zz1, zz2)
                    self.assertLessEqual(np.abs(ipA - ipB), tol * (np.abs(ipA) + np.abs(ipB)))

    def test_probe_tangent_transpose2(self):
        structures = [
            ((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)),
        ]
        NUM_PROBES = 2

        for STRUCTURE in structures:
            for T3P in [t3p, t3p_jax]:
                with self.subTest(T3P=T3P, STRUCTURE=STRUCTURE):
                    p = t3.t3_corewise_randn(STRUCTURE)
                    base, _ = orth.orthogonal_representations(p)

                    SHAPE = STRUCTURE[0]
                    www = (np.random.randn(NUM_PROBES, SHAPE[0]),
                           np.random.randn(NUM_PROBES, SHAPE[1]),
                           np.random.randn(NUM_PROBES, SHAPE[2]))

                    apply_J = lambda v: t3p.probe_tangent(v, www, base)
                    apply_Jt = lambda zz: T3P.probe_tangent_transpose(zz, www, base)
                    v = t3m.tangent_randn(base)

                    zzz = (np.random.randn(NUM_PROBES, SHAPE[0]),
                           np.random.randn(NUM_PROBES, SHAPE[1]),
                           np.random.randn(NUM_PROBES, SHAPE[2]))

                    ipA = cw.corewise_dot(zzz, apply_J(v))
                    ipB = cw.corewise_dot(apply_Jt(zzz), v)
                    self.assertLessEqual(np.abs(ipA - ipB), tol * (np.abs(ipA) + np.abs(ipB)))


if __name__ == '__main__':
    unittest.main()

