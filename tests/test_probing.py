# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3tools.corewise
import t3tools.orthogonalization
import t3tools.tucker_tensor_train as t3
import t3tools.manifold as t3m
import t3tools.probing as t3p
import t3tools.common as common

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestProbing(unittest.TestCase):
    def test_probe_dense1(self):
        T = np.random.randn(10, 11, 12)
        u0 = np.random.randn(10)
        u1 = np.random.randn(11)
        u2 = np.random.randn(12)
        yy = t3p.probe_dense(T, (u0, u1, u2))
        y0 = np.einsum('ijk,j,k', T, u1, u2)
        y1 = np.einsum('ijk,i,k', T, u0, u2)
        y2 = np.einsum('ijk,i,j', T, u0, u1)
        self.assertLessEqual(norm(yy[0] - y0), tol * norm(y0))
        self.assertLessEqual(norm(yy[1] - y1), tol * norm(y1))
        self.assertLessEqual(norm(yy[2] - y2), tol * norm(y2))

    def test_probe_dense2(self):
        T = np.random.randn(10, 11, 12)
        u0, v0 = np.random.randn(10), np.random.randn(10)
        u1, v1 = np.random.randn(11), np.random.randn(11)
        u2, v2 = np.random.randn(12), np.random.randn(12)
        uuu = [np.vstack([u0, v0]), np.vstack([u1, v1]), np.vstack([u2, v2])]
        yyy = t3p.probe_dense(T, uuu)
        yy_u = t3p.probe_dense(T, (u0, u1, u2))
        yy_v = t3p.probe_dense(T, (v0, v1, v2))

        self.assertLessEqual(norm(yy_u[0] - yyy[0][0, :]), tol * norm(yy_u[0]))
        self.assertLessEqual(norm(yy_u[1] - yyy[1][0, :]), tol * norm(yy_u[1]))
        self.assertLessEqual(norm(yy_u[2] - yyy[2][0, :]), tol * norm(yy_u[2]))

        self.assertLessEqual(norm(yy_v[0] - yyy[0][1, :]), tol * norm(yy_v[0]))
        self.assertLessEqual(norm(yy_v[1] - yyy[1][1, :]), tol * norm(yy_v[1]))
        self.assertLessEqual(norm(yy_v[2] - yyy[2][1, :]), tol * norm(yy_v[2]))

    def test_probe_t3_1(self):
        x = t3.t3_corewise_randn(((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)))
        ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        zz = t3p.probe_t3(x, ww)
        x_dense = t3.t3_to_dense(x)
        zz2 = t3p.probe_dense(x_dense, ww)

        for z, z2 in zip(zz, zz2):
            self.assertLessEqual(norm(z - z2), tol * norm(z2))

        def test_probe_t3_2(self):
            x = t3.t3_corewise_randn(((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)))
            www = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
            zzz = t3p.probe_t3(x, www)
            zzz2 = t3p.probe_dense(t3.t3_to_dense(x), www)

            for z, z2 in zip(zzz, zzz2):
                self.assertLessEqual(norm(z - z2), tol * norm(z2))

    def test_probe_tangent(self):
        p = t3.t3_corewise_randn(((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)))
        base, _ = t3tools.orthogonalization.orthogonal_representations(p)
        variation = t3m.tangent_randn(base)
        ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        zz = t3p.probe_tangent(variation, ww, base)
        zz2 = t3p.probe_dense(t3m.tangent_to_dense(variation, base), ww)

        for z, z2 in zip(zz, zz2):
            self.assertLessEqual(norm(z - z2), tol * norm(z2))

    def test_probe_tangent2(self):
        p = t3.t3_corewise_randn(((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)))
        base, _ = t3tools.orthogonalization.orthogonal_representations(p)
        variation = t3m.tangent_randn(base)
        www = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
        zzz = t3p.probe_tangent(variation, www, base)  # Compute probes!
        zzz2 = t3p.probe_dense(t3m.tangent_to_dense(variation, base), www)

        for zz, zz2 in zip(zzz, zzz2):
            self.assertLessEqual(norm(zz - zz2), tol * norm(zz2))

    def test_probe_tangent_transpose1(self):
        p = t3.t3_corewise_randn(((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)))
        base, _ = t3tools.orthogonalization.orthogonal_representations(p)
        ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        v1 = t3m.tangent_randn(base)
        zz1 = t3p.probe_tangent(v1, ww, base)
        zz2 = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        v2 = t3p.probe_tangent_transpose(zz2, ww, base)
        ipA = t3tools.corewise.corewise_dot(v1, v2)
        ipB = t3tools.corewise.corewise_dot(zz1, zz2)
        self.assertLessEqual(np.abs(ipA - ipB), tol * (np.abs(ipA) + np.abs(ipB)))

    def test_probe_tangent_transpose2(self):
        p = t3.t3_corewise_randn(((10, 11, 12), (5, 6, 4), (2, 3, 4, 2)))
        base, _ = t3tools.orthogonalization.orthogonal_representations(p)
        ww = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
        apply_J = lambda v: t3p.probe_tangent(v, ww, base)
        apply_Jt = lambda z: t3p.probe_tangent_transpose(z, ww, base)
        v = t3m.tangent_randn(base)
        z = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
        ipA = t3tools.corewise.corewise_dot(z, apply_J(v))
        ipB = t3tools.corewise.corewise_dot(apply_Jt(z), v)
        self.assertLessEqual(np.abs(ipA - ipB), tol * (np.abs(ipA) + np.abs(ipB)))


if __name__ == '__main__':
    unittest.main()

