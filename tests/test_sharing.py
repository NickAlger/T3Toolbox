# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.sharing as sharing

np.random.seed(0)


# (shape, tucker_ranks, tt_ranks, sharing) -- shapes equal within each sharing group.
SHARED_STRUCTURES = [
    ((6,),           (3,),         (1, 1),           (0,)),            # d=1, singleton only
    ((6, 6),         (3, 3),       (1, 2, 1),        (0, 0)),          # one group of two
    ((6, 6, 6, 5),   (3, 3, 3, 2), (1, 2, 3, 2, 1),  (0, 0, 0, 1)),    # a triple + a singleton
    ((5, 6, 5, 6),   (2, 3, 2, 3), (1, 2, 2, 2, 1),  ('a', 'b', 'a', 'b')),  # non-adjacent groups
]
STACK_SHAPES = [(), (2,), (2, 3)]


def _tied_data(structure, stack_shape):
    """Random T3 data with the Tucker factors tied (same array object) within each group."""
    shape, tucker_ranks, tt_ranks, sharing_spec = structure
    x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks, stack_shape=stack_shape)
    tucker_cores, tt_cores = [list(c) for c in x.data]
    groups = sharing.validate_sharing(sharing_spec, shape)
    for group in groups:
        for ii in group:
            tucker_cores[ii] = tucker_cores[group[0]]
    return (tuple(tucker_cores), tuple(tt_cores)), sharing_spec, groups


class TestValidateSharing(unittest.TestCase):
    def test_canonical_groups(self):
        # groups ordered by first mode, modes ascending, singletons included; labels any hashables
        self.assertEqual(sharing.validate_sharing((0, 1, 1, 2, 2, 2), (4, 5, 5, 6, 6, 6)),
                         ((0,), (1, 2), (3, 4, 5)))
        self.assertEqual(sharing.validate_sharing(('in', 'out', 'in'), (7, 5, 7)),
                         ((0, 2), (1,)))
        self.assertEqual(sharing.validate_sharing((None, None), (4, 4)), ((0, 1),))
        self.assertEqual(sharing.validate_sharing((3, 1, 2), (4, 5, 6)),   # all singletons
                         ((0,), (1,), (2,)))

    def test_structural_errors(self):
        with self.assertRaises(ValueError):
            sharing.validate_sharing((0, 0), (4, 5))            # unequal sizes within a group
        with self.assertRaises(ValueError):
            sharing.validate_sharing((0, 1), (4, 5, 6))         # wrong length
        with self.assertRaises(ValueError):
            sharing.validate_sharing(([0], [0]), (4, 4))        # unhashable labels


class TestSharingCheckers(unittest.TestCase):
    def test_exactly_tied_residual_is_zero(self):
        for STRUCTURE in SHARED_STRUCTURES:
            for STACK_SHAPE in STACK_SHAPES:
                with self.subTest(STRUCTURE=STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x_data, sharing_spec, _ = _tied_data(STRUCTURE, STACK_SHAPE)
                    r = np.asarray(sharing.t3_sharing_residual(x_data, sharing_spec))
                    self.assertEqual(r.shape, STACK_SHAPE)
                    self.assertTrue(np.all(r == 0.0))
                    self.assertTrue(np.all(sharing.t3_tucker_factors_shared(x_data, sharing_spec)))

    def test_perturbation_is_detected_relative(self):
        eps = 1e-6
        for STRUCTURE in SHARED_STRUCTURES[1:]:                  # structures with a real group
            for STACK_SHAPE in STACK_SHAPES:
                with self.subTest(STRUCTURE=STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    (tk, tt), sharing_spec, groups = _tied_data(STRUCTURE, STACK_SHAPE)
                    group = next(g for g in groups if len(g) > 1)
                    tk = list(tk)
                    B = np.asarray(tk[group[-1]]).copy()
                    B += eps * np.linalg.norm(B) / np.sqrt(B.size) * np.random.randn(*B.shape)
                    tk[group[-1]] = B
                    r = np.asarray(sharing.t3_sharing_residual((tuple(tk), tt), sharing_spec))
                    self.assertTrue(np.all(r > eps / 100) and np.all(r < eps * 100))
                    self.assertTrue(np.all(sharing.t3_tucker_factors_shared((tuple(tk), tt),
                                                                            sharing_spec, rtol=1e-3)))
                    self.assertFalse(np.any(sharing.t3_tucker_factors_shared((tuple(tk), tt),
                                                                             sharing_spec, rtol=1e-9)))

    def test_per_stack_element_verdicts(self):
        (tk, tt), sharing_spec, groups = _tied_data(SHARED_STRUCTURES[2], (2,))
        group = next(g for g in groups if len(g) > 1)
        tk = list(tk)
        B = np.asarray(tk[group[1]]).copy()
        B[1] += 1e-3 * np.random.randn(*B.shape[1:])             # perturb stack element 1 only
        tk[group[1]] = B
        r = np.asarray(sharing.t3_sharing_residual((tuple(tk), tt), sharing_spec))
        self.assertEqual(r.shape, (2,))
        self.assertEqual(float(r[0]), 0.0)
        self.assertGreater(float(r[1]), 1e-5)
        verdicts = np.asarray(sharing.t3_tucker_factors_shared((tuple(tk), tt), sharing_spec))
        self.assertTrue(bool(verdicts[0]) and not bool(verdicts[1]))

    def test_zero_reference_conventions(self):
        # tied zeros -> 0; zero reference with a nonzero other factor -> inf
        shape, n, r = (6, 6), (3, 3), (1, 2, 1)
        x = t3.TuckerTensorTrain.randn(shape, n, r)
        _, tt = x.data
        Z = np.zeros((3, 6))
        self.assertEqual(float(sharing.t3_sharing_residual(((Z, Z), tt), (0, 0))), 0.0)
        r_inf = sharing.t3_sharing_residual(((Z, np.random.randn(3, 6)), tt), (0, 0))
        self.assertTrue(np.isinf(float(r_inf)))

    def test_group_rank_mismatch_raises(self):
        # equal mode sizes but unequal Tucker ranks within a group: structural error
        tk = (np.random.randn(3, 6), np.random.randn(2, 6))
        tt = (np.random.randn(1, 3, 2), np.random.randn(2, 2, 1))
        with self.assertRaises(ValueError):
            sharing.t3_sharing_residual((tk, tt), (0, 0))
        with self.assertRaises(ValueError):
            sharing.t3_share_tucker_cores((tk, tt), (0, 0))


class TestShareTuckerCores(unittest.TestCase):
    def test_mean_and_identity_assignment(self):
        for STRUCTURE in SHARED_STRUCTURES:
            for STACK_SHAPE in STACK_SHAPES:
                with self.subTest(STRUCTURE=STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    shape, n, r, sharing_spec = STRUCTURE
                    x = t3.TuckerTensorTrain.randn(shape, n, r, stack_shape=STACK_SHAPE)
                    tk, tt = x.data
                    tk2, tt2 = sharing.t3_share_tucker_cores(x.data, sharing_spec)
                    groups = sharing.validate_sharing(sharing_spec, shape)
                    self.assertIs(tt2[0], tt[0])                 # tt cores untouched (same objects)
                    for group in groups:
                        for ii in group:
                            self.assertIs(tk2[ii], tk2[group[0]])   # ONE array per group
                        if len(group) > 1:
                            ref = sum(np.asarray(tk[ii]) for ii in group) / len(group)
                            self.assertTrue(np.allclose(np.asarray(tk2[group[0]]), ref))
                        else:
                            self.assertIs(tk2[group[0]], tk[group[0]])   # singleton: passthrough
                    res = sharing.t3_sharing_residual((tk2, tt2), sharing_spec)
                    self.assertTrue(np.all(np.asarray(res) == 0.0))

    def test_tied_input_is_unchanged(self):
        # mean of identical arrays is exact in floating point -> values unchanged, tensor unchanged
        x_data, sharing_spec, _ = _tied_data(SHARED_STRUCTURES[2], ())
        tk2, tt2 = sharing.t3_share_tucker_cores(x_data, sharing_spec)
        for A, B in zip(tk2, x_data[0]):
            self.assertTrue(np.array_equal(np.asarray(A), np.asarray(B)))
        x_dense = t3.TuckerTensorTrain(*x_data).to_dense()
        y_dense = t3.TuckerTensorTrain(tk2, tt2).to_dense()
        self.assertTrue(np.array_equal(np.asarray(x_dense), np.asarray(y_dense)))


if __name__ == '__main__':
    unittest.main()
