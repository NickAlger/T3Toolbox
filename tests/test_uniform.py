# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import unittest

import t3tools.base_variation_format as bvf
import t3tools.orthogonalization as orth
import t3tools.tucker_tensor_train as t3
import t3tools.manifold as t3m
import t3tools.common as common
import t3tools.uniform as ut3


np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn

class TestUniform(unittest.TestCase):
    def test_padded_and_original_structure(self):
        s = ((14,15,16), (4,6,5), (2,3,2,1))
        x = t3.t3_corewise_randn(s)
        cores, masks = ut3.t3_to_ut3(x)
        self.assertEqual((3, 16, 6, 3), ut3.get_padded_structure(cores))
        self.assertEqual(s, ut3.get_original_structure(masks))

    def test_unpack_edge_vectors1(self):
        E = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        submask = [[True, False, True, True], [False, True, False, False]]
        ee = ut3.unpack_edge_tensors(E, submask)
        self.assertEqual([1, 3, 4], list(ee[0]))
        self.assertEqual([6], list(ee[1]))

    def test_unpack_edge_vectors2(self):
        E = np.random.randn(6, 5, 4, 3, 2)
        submask = [[False, False], [False, True], [True, True]]
        ee = ut3.unpack_edge_tensors(E, submask)
        self.assertEqual([(6, 5, 4, 0), (6, 5, 4, 1), (6, 5, 4, 2)], [e.shape for e in ee])

    def test_t3_to_ut3_to_t3(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 6, 5), (3, 3, 2, 4)))
        cores, masks = ut3.t3_to_ut3(x)  # Convert t3 -> ut3
        x2 = ut3.ut3_to_t3(cores, masks)  # Convert ut3 -> t3
        dense_x = t3.t3_to_dense(x)
        dense_x2 = t3.t3_to_dense(x2)
        self.assertLessEqual(norm(dense_x - dense_x2), tol * norm(dense_x))

    def test_ut3_to_dense(self):
        x = t3.t3_corewise_randn(((14, 15, 16), (4, 6, 5), (3, 3, 2, 4)))
        cores, masks = ut3.t3_to_ut3(x)  # Convert t3 -> ut3
        dense_x = t3.t3_to_dense(x)
        dense_x2 = ut3.ut3_to_dense(cores, masks)
        self.assertLessEqual(norm(dense_x - dense_x2), tol * norm(dense_x))




if __name__ == '__main__':
    unittest.main()
