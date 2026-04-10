# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3tools.linalg

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestLinalg(unittest.TestCase):
    def test_left_svd_3tensor(self):
        G_i_a_j = np.random.randn(5, 7, 6)
        U_i_a_x, ss_x, Vt_x_j = t3tools.linalg.left_svd_3tensor(G_i_a_j)
        G_i_a_j2 = np.einsum('iax,x,xj->iaj', U_i_a_x, ss_x, Vt_x_j)
        self.assertLessEqual(norm(G_i_a_j - G_i_a_j2), tol * norm(G_i_a_j))

        rank = len(ss_x)
        self.assertEqual(6, rank)

        self.assertLessEqual(
            norm(np.einsum('iax,iay->xy', U_i_a_x, U_i_a_x) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )
        self.assertLessEqual(
            norm(np.einsum('xj,yj->xy', Vt_x_j, Vt_x_j) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )

    def test_right_svd_3tensor(self):
        G_i_a_j = np.random.randn(5, 7, 6)
        U_i_x, ss_x, Vt_x_a_j = t3tools.linalg.right_svd_3tensor(G_i_a_j)
        G_i_a_j2 = np.einsum('ix,x,xaj->iaj', U_i_x, ss_x, Vt_x_a_j)
        self.assertLessEqual(norm(G_i_a_j - G_i_a_j2), tol * norm(G_i_a_j))

        rank = len(ss_x)
        self.assertEqual(5, rank)

        self.assertLessEqual(
            norm(np.einsum('ix,iy->xy', U_i_x, U_i_x) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )
        self.assertLessEqual(
            norm(np.einsum('xaj,yaj->xy', Vt_x_a_j, Vt_x_a_j) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )

    def test_outer_svd_3tensor(self):
        G_i_a_j = np.random.randn(5, 7, 6)
        U_i_x_j, ss_x, Vt_x_a = t3tools.linalg.outer_svd_3tensor(G_i_a_j)
        G_i_a_j2 = np.einsum('ixj,x,xa->iaj', U_i_x_j, ss_x, Vt_x_a)
        self.assertLessEqual(norm(G_i_a_j - G_i_a_j2), tol * norm(G_i_a_j))

        rank = len(ss_x)
        self.assertEqual(7, rank)

        self.assertLessEqual(
            norm(np.einsum('ixj,iyj->xy', U_i_x_j, U_i_x_j) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )
        self.assertLessEqual(
            norm(np.einsum('xa,ya->xy', Vt_x_a, Vt_x_a) - np.eye(rank)),
            tol * norm(np.eye(rank))
        )



if __name__ == '__main__':
    unittest.main()

