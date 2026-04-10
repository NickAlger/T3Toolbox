# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3tools.linalg as linalg
try:
    import t3tools.jax.linalg as linalg_jax
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    linalg_jax = linalg

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestLinalgJax(unittest.TestCase):
    def check_relerr(self, xtrue, x):
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def test_truncated_svd(self):
        A = np.diag(np.random.randn(100))

        _, ss_big, _ = np.linalg.svd(A)
        loose_rtol = 0.5
        tight_rtol = 0.1
        loose_atol = ss_big[0] * 0.6
        tight_atol = ss_big[0] * 0.2

        for svd_atol, svd_rtol in zip(
                [None, None,        loose_atol, loose_atol, tight_atol],
                [None, loose_rtol,  None,       tight_rtol, loose_rtol]
        ):
            U, ss, Vt               = linalg.truncated_svd(A,       atol=svd_atol, rtol=svd_rtol)
            U_jax, ss_jax, Vt_jax   = linalg_jax.truncated_svd(A,   atol=svd_atol, rtol=svd_rtol)

            svd_atol = 0.0 if svd_atol is None else svd_atol
            svd_rtol = 0.0 if svd_rtol is None else svd_rtol

            U0, ss0, Vt0 = np.linalg.svd(A)
            rank = np.sum(ss_big >= np.maximum(svd_atol, svd_rtol * ss_big[0]))
            U_trunc = U[:,:rank]
            ss_trunc = ss0[:rank]
            Vt_trunc = Vt[:rank,:]

            self.check_relerr(U_trunc, U)
            self.check_relerr(ss_trunc, ss)
            self.check_relerr(Vt_trunc, Vt)

            self.check_relerr(U, U_jax)
            self.check_relerr(ss, ss_jax)
            self.check_relerr(Vt, Vt_jax)

    def test_left_svd_3tensor(self):
        G_i_a_j = np.random.randn(5, 7, 6)
        U_i_a_x,        ss_x,       Vt_x_j      = linalg.left_svd_3tensor(G_i_a_j)
        U_i_a_x_jax,    ss_x_jax,   Vt_x_j_jax  = linalg_jax.left_svd_3tensor(G_i_a_j)

        G_i_a_j2 = np.einsum('iax,x,xj->iaj', U_i_a_x, ss_x, Vt_x_j)
        self.assertLessEqual(norm(G_i_a_j - G_i_a_j2), tol * norm(G_i_a_j))

        rank = len(ss_x)
        true_rank = min(G_i_a_j.shape[0]*G_i_a_j.shape[1], G_i_a_j.shape[2])
        self.assertEqual(true_rank, rank)

        self.assertLessEqual(
            norm(np.einsum('iax,iay->xy', U_i_a_x, U_i_a_x) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )
        self.assertLessEqual(
            norm(np.einsum('xj,yj->xy', Vt_x_j, Vt_x_j) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )

        self.check_relerr(U_i_a_x, U_i_a_x_jax)
        self.check_relerr(ss_x, ss_x_jax)
        self.check_relerr(Vt_x_j, Vt_x_j_jax)

    def test_right_svd_3tensor(self):
        G_i_a_j = np.random.randn(5, 7, 6)
        U_i_x,      ss_x,       Vt_x_a_j        = linalg.right_svd_3tensor(G_i_a_j)
        U_i_x_jax,  ss_x_jax,   Vt_x_a_j_jax    = linalg_jax.right_svd_3tensor(G_i_a_j)

        G_i_a_j2 = np.einsum('ix,x,xaj->iaj', U_i_x, ss_x, Vt_x_a_j)
        self.assertLessEqual(norm(G_i_a_j - G_i_a_j2), tol * norm(G_i_a_j))

        rank = len(ss_x)
        true_rank = min(G_i_a_j.shape[0], G_i_a_j.shape[1]*G_i_a_j.shape[2])
        self.assertEqual(true_rank, rank)

        self.assertLessEqual(
            norm(np.einsum('ix,iy->xy', U_i_x, U_i_x) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )
        self.assertLessEqual(
            norm(np.einsum('xaj,yaj->xy', Vt_x_a_j, Vt_x_a_j) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )

        self.check_relerr(U_i_x, U_i_x_jax)
        self.check_relerr(ss_x, ss_x_jax)
        self.check_relerr(Vt_x_a_j, Vt_x_a_j_jax)

    def test_outer_svd_3tensor(self):
        G_i_a_j = np.random.randn(5, 7, 6)
        U_i_x_j,        ss_x,       Vt_x_a      = linalg.outer_svd_3tensor(G_i_a_j)
        U_i_x_j_jax,    ss_x_jax,   Vt_x_a_jax  = linalg_jax.outer_svd_3tensor(G_i_a_j)

        G_i_a_j2 = np.einsum('ixj,x,xa->iaj', U_i_x_j, ss_x, Vt_x_a)
        self.assertLessEqual(norm(G_i_a_j - G_i_a_j2), tol * norm(G_i_a_j))

        rank = len(ss_x)
        true_rank = min(G_i_a_j.shape[0]*G_i_a_j.shape[2], G_i_a_j.shape[1])
        self.assertEqual(true_rank, rank)

        self.assertLessEqual(
            norm(np.einsum('ixj,iyj->xy', U_i_x_j, U_i_x_j) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )
        self.assertLessEqual(
            norm(np.einsum('xa,ya->xy', Vt_x_a, Vt_x_a) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )

        self.check_relerr(U_i_x_j, U_i_x_j_jax)
        self.check_relerr(ss_x, ss_x_jax)
        self.check_relerr(Vt_x_a, Vt_x_a_jax)


if __name__ == '__main__':
    unittest.main()

