# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.sharing as sharing
import t3toolbox.backend.fv_conversions as fvc

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


class TestSharedFrameData(unittest.TestCase):
    """Permanent invariants of the shared-frame companion (the S_i machinery)."""

    @staticmethod
    def _svd_reconstruct(sfd, gi):
        # M_g = U diag(s) Vt, batched over the frame stack
        return np.einsum('...ux,...x,...xv->...uv',
                         np.asarray(sfd.svd_U[gi]), np.asarray(sfd.svd_s[gi]),
                         np.asarray(sfd.svd_Vt[gi]))

    def test_centers_match_construction_exactly(self):
        # the re-sweep recompute IS the construction's own computation: bit-identical centers
        for STRUCTURE in SHARED_STRUCTURES:
            for STACK_SHAPE in STACK_SHAPES:
                with self.subTest(STRUCTURE=STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x_data, sharing_spec, groups = _tied_data(STRUCTURE, STACK_SHAPE)
                    frame_d, variations_d = fvc.t3_orthogonal_representations(x_data)
                    sfd = sharing.fv_shared_frame_data(frame_d, groups)
                    HH = variations_d[1]                     # the construction's center cores
                    for gi, group in enumerate(sharing.nontrivial_groups(groups)):
                        for jj, ii in enumerate(group):
                            self.assertTrue(np.array_equal(np.asarray(sfd.centers[gi][jj]),
                                                           np.asarray(HH[ii])))

    def test_s_factor_identities(self):
        # S_i S_i^T == Gamma_i (from the centers) and S_i-absorbed O_i == H_i, per group mode;
        # singular values descending; row splits match the nD_i
        for STRUCTURE in SHARED_STRUCTURES[1:]:
            for STACK_SHAPE in STACK_SHAPES:
                with self.subTest(STRUCTURE=STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x_data, _, groups = _tied_data(STRUCTURE, STACK_SHAPE)
                    frame_d, _ = fvc.t3_orthogonal_representations(x_data)
                    OO = frame_d[1]
                    sfd = sharing.fv_shared_frame_data(frame_d, groups)
                    for gi, group in enumerate(sharing.nontrivial_groups(groups)):
                        M = self._svd_reconstruct(sfd, gi)
                        splits = sfd.row_splits[gi]
                        ss = np.asarray(sfd.svd_s[gi])
                        self.assertTrue(np.all(np.diff(ss, axis=-1) <= 1e-12 * ss[..., :1]))
                        for jj, ii in enumerate(group):
                            S_T = M[..., splits[jj]:splits[jj + 1], :]      # (C)+(nDi, nU)
                            S = np.swapaxes(S_T, -1, -2)
                            H = np.asarray(sfd.centers[gi][jj])
                            Gamma = np.einsum('...aub,...avb->...uv', H, H)
                            scale = np.linalg.norm(Gamma)
                            self.assertLess(np.linalg.norm(
                                np.einsum('...ux,...vx->...uv', S, S) - Gamma), 1e-10 * scale)
                            SO = np.einsum('...ux,...axb->...aub', S, np.asarray(OO[ii]))
                            self.assertLess(np.linalg.norm(SO - H), 1e-10 * np.linalg.norm(H))

    def test_group_spectrum_matches_dense(self):
        # svd_s == singular values of the dense concatenated matricizations (M7.1 / test 14b)
        STRUCTURE = SHARED_STRUCTURES[2]                     # (6,6,6,5), group (0,1,2)
        for STACK_SHAPE in [(), (2,)]:
            with self.subTest(STACK_SHAPE=STACK_SHAPE):
                x_data, _, groups = _tied_data(STRUCTURE, STACK_SHAPE)
                frame_d, _ = fvc.t3_orthogonal_representations(x_data)
                sfd = sharing.fv_shared_frame_data(frame_d, groups)
                group = sharing.nontrivial_groups(groups)[0]
                n_g = x_data[0][group[0]].shape[-2]
                dense = np.asarray(t3.TuckerTensorTrain(*x_data).to_dense())
                for idx in np.ndindex(*STACK_SHAPE) if STACK_SHAPE else [()]:
                    T = dense[idx]
                    mats = [np.moveaxis(T, ii, 0).reshape(T.shape[ii], -1) for ii in group]
                    s_dense = np.linalg.svd(np.concatenate(mats, axis=1), compute_uv=False)
                    s_g = np.asarray(sfd.svd_s[0])[idx]
                    self.assertLess(np.linalg.norm(s_g - s_dense[:n_g]), 1e-9 * s_dense[0])

    def test_padded_point_degeneracy(self):
        # zero-padded shared restart: S_i rows for the new directions are exactly zero and the
        # trailing group spectrum levels vanish (the restart analysis, handoff v3 section 4.8)
        shape, n, r, sharing_spec = SHARED_STRUCTURES[2]
        x_data, _, groups = _tied_data((shape, n, r, sharing_spec), ())
        x = t3.TuckerTensorTrain(*x_data)
        xp = x.resize(shape, (4, 4, 4, 2), r)                # group Tucker rank 3 -> 4
        self.assertTrue(np.allclose(np.asarray(xp.to_dense()), np.asarray(x.to_dense())))
        frame_d, _ = fvc.t3_orthogonal_representations(xp.data)
        OO = frame_d[1]
        sfd = sharing.fv_shared_frame_data(frame_d, groups)
        group = sharing.nontrivial_groups(groups)[0]
        for gi, ii in enumerate(group):
            H = np.asarray(sfd.centers[0][gi])
            S = np.einsum('...aub,...axb->...ux', H, np.asarray(OO[ii]))   # (nU'=4, nDi)
            self.assertEqual(float(np.linalg.norm(S[..., 3:, :])), 0.0)   # new row exactly zero
        ss = np.asarray(sfd.svd_s[0])
        self.assertLessEqual(float(ss[-1]), 1e-14 * float(ss[0]))         # degenerate group level

    def test_all_singleton_companion_is_empty(self):
        x = t3.TuckerTensorTrain.randn((4, 5), (2, 3), (1, 2, 1))
        frame_d, _ = fvc.t3_orthogonal_representations(x.data)
        groups = sharing.validate_sharing((0, 1), (4, 5))
        sfd = sharing.fv_shared_frame_data(frame_d, groups)
        self.assertEqual(sfd.centers, ())
        self.assertEqual(sfd.svd_s, ())
        self.assertEqual(sfd.groups, groups)


if __name__ == '__main__':
    unittest.main()
