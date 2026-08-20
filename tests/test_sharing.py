# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.shared_geometry as sg
import t3toolbox.fitting as fitting
import t3toolbox.optimizers as optimizers
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.sharing as sharing
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.t3_svd as bt3svd
import t3toolbox.backend.ut3_svd as but3svd
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.tv_operations as tvo
import t3toolbox.safety as safety

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
    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test

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
    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test

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

    def test_frontend_method_has_shared_tucker_factors(self):
        # the method form of t3_tucker_factors_shared (a property checker of the T3, hence a method
        # -- not a free function: it combines the T3 with a spec, not with another substantive object)
        x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 2, 2, 1))
        self.assertFalse(bool(x.has_shared_tucker_factors((0, 0, 1))))
        self.assertTrue(bool(x.share((0, 0, 1)).has_shared_tucker_factors((0, 0, 1))))
        # per-stack-element verdicts, matching the backend checker exactly
        x_data, spec, _ = _tied_data(SHARED_STRUCTURES[2], (2,))
        xs = t3.TuckerTensorTrain(*x_data)
        verdicts = np.asarray(xs.has_shared_tucker_factors(spec))
        self.assertEqual(verdicts.shape, (2,))
        self.assertTrue(np.all(verdicts))


class TestShareTuckerCores(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test

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


def _dense_group_svals(x_dense, group):
    """Singular values of the dense concatenated matricizations [X_(i1) | ... | X_(ik)]."""
    mats = [np.moveaxis(x_dense, ii, 0).reshape(x_dense.shape[ii], -1) for ii in group]
    return np.linalg.svd(np.concatenate(mats, axis=1), compute_uv=False)


class TestGroupedT3svd(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test

    def _assert_bit_identical(self, res_a, res_b):
        (xa, ska, sta), (xb, skb, stb) = res_a, res_b
        for fam_a, fam_b in zip(xa, xb):
            for A, B in zip(fam_a, fam_b):
                self.assertTrue(np.array_equal(np.asarray(A), np.asarray(B)))
        for SA, SB in zip(ska + sta, skb + stb):
            self.assertTrue(np.array_equal(np.asarray(SA), np.asarray(SB)))

    def test_none_and_all_singleton_dispatch_bit_identical(self):
        # test 7.2: sharing=None and all-singleton partitions run the literal unshared sweep
        for STACK_SHAPE in [(), (2,)]:
            with self.subTest(STACK_SHAPE=STACK_SHAPE):
                x = t3.TuckerTensorTrain.randn((6, 5, 4), (3, 2, 2), (1, 2, 2, 1),
                                               stack_shape=STACK_SHAPE)
                ref = bt3svd.t3svd(x.data, max_tucker_ranks=2)
                self._assert_bit_identical(bt3svd.t3svd(x.data, max_tucker_ranks=2, sharing=None), ref)
                self._assert_bit_identical(
                    bt3svd.t3svd(x.data, max_tucker_ranks=2, sharing=('a', 'b', 'c')), ref)

    def test_lossless_agreement_with_unshared(self):
        # test 7.4: on exactly-tied input with no truncation, shared and unshared represent the
        # same tensor and report the same TT ranks; the shared output is tied and left-orthogonal
        for STRUCTURE in SHARED_STRUCTURES[1:]:
            for STACK_SHAPE in STACK_SHAPES:
                with self.subTest(STRUCTURE=STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x_data, sharing_spec, groups = _tied_data(STRUCTURE, STACK_SHAPE)
                    x_dense = np.asarray(t3.TuckerTensorTrain(*x_data).to_dense())
                    y, sk, st = bt3svd.t3svd(x_data, sharing=sharing_spec)
                    y2, _, _ = bt3svd.t3svd(x_data)
                    yT = t3.TuckerTensorTrain(*y)
                    self.assertTrue(np.allclose(np.asarray(yT.to_dense()), x_dense))
                    self.assertEqual(yT.tt_ranks, t3.TuckerTensorTrain(*y2).tt_ranks)
                    self.assertTrue(np.all(np.asarray(yT.is_left_orthogonal())))
                    for group in sharing.nontrivial_groups(groups):
                        for ii in group[1:]:
                            self.assertIs(y[0][ii], y[0][group[0]])
                            self.assertTrue(np.array_equal(np.asarray(sk[ii]),
                                                           np.asarray(sk[group[0]])))

    def test_group_spectrum_and_subspace(self):
        # test 7.3: s_g equals the dense concatenated-matricization spectrum, and the shared
        # basis spans its top left singular subspace (exact rank -> projector match)
        x_data, sharing_spec, groups = _tied_data(SHARED_STRUCTURES[2], ())
        group = sharing.nontrivial_groups(groups)[0]
        x_dense = np.asarray(t3.TuckerTensorTrain(*x_data).to_dense())
        y, sk, _ = bt3svd.t3svd(x_data, sharing=sharing_spec)
        n_g = y[0][group[0]].shape[-2]
        s_dense = _dense_group_svals(x_dense, group)
        self.assertLess(np.linalg.norm(np.asarray(sk[group[0]]) - s_dense[:n_g]),
                        1e-9 * s_dense[0])
        U_rows = np.asarray(y[0][group[0]])                              # (n_g, N), orthonormal rows
        mats = [np.moveaxis(x_dense, ii, 0).reshape(6, -1) for ii in group]
        Yd = np.linalg.svd(np.concatenate(mats, axis=1))[0][:, :n_g]     # dense top left vectors
        P_ours, P_dense = U_rows.T @ U_rows, Yd @ Yd.T
        self.assertLess(np.linalg.norm(P_ours - P_dense), 1e-8)

    def test_truncation_error_bounds(self):
        # group-only truncation error is bounded by the tail of s_g (the sum of single-mode
        # projection errors; equality only for singleton groups) ...
        x_data, sharing_spec, _ = _tied_data(SHARED_STRUCTURES[2], ())
        x_dense = np.asarray(t3.TuckerTensorTrain(*x_data).to_dense())
        _, sk_full, _ = bt3svd.t3svd(x_data, sharing=sharing_spec)
        y, _, _ = bt3svd.t3svd(x_data, sharing=sharing_spec, max_tucker_ranks=(2, 2, 2, 2))
        err = np.linalg.norm(np.asarray(t3.TuckerTensorTrain(*y).to_dense()) - x_dense)
        tail = np.sqrt(sum(float(np.sum(np.asarray(sk_full[ii])[2:] ** 2)) for ii in (0, 3)))
        self.assertLessEqual(err, tail * (1 + 1e-9))
        # ... and the full result is quasi-optimal: err <= C(d) * ||noise|| on tied low-rank+noise
        shape, n, r, spec = SHARED_STRUCTURES[2]
        d = len(shape)
        lo_data, _, _ = _tied_data((shape, n, r, spec), ())
        lo = t3.TuckerTensorTrain(*lo_data)
        noise_data, _, _ = _tied_data((shape, (2, 2, 2, 2), (1, 2, 2, 2, 1), spec), ())
        noise = t3.TuckerTensorTrain(*noise_data)
        eps = 1e-4 * float(np.linalg.norm(lo.to_dense())) / float(np.linalg.norm(noise.to_dense()))
        noisy = lo + t3.TuckerTensorTrain(
            noise.data[0], (np.asarray(noise.data[1][0]) * eps,) + tuple(noise.data[1][1:]))
        y, _, _ = bt3svd.t3svd(noisy.data, sharing=spec, max_tucker_ranks=n, max_tt_ranks=r)
        err = np.linalg.norm(np.asarray(t3.TuckerTensorTrain(*y).to_dense())
                             - np.asarray(noisy.to_dense()))
        noise_norm = eps * float(np.linalg.norm(noise.to_dense()))
        C_d = np.sqrt(d) + np.sqrt(d) * np.sqrt(d - 1) + np.sqrt(d - 1)
        self.assertLessEqual(err, C_d * noise_norm)

    def test_exact_rank_recovery_and_upper_bound(self):
        # test 7.5a: inflate an exactly-shared exact-rank T3, recover the true ranks exactly
        shape, n, r, spec = SHARED_STRUCTURES[2]
        x_data, _, groups = _tied_data((shape, n, r, spec), ())
        x_dense = np.asarray(t3.TuckerTensorTrain(*x_data).to_dense())
        xz = t3.TuckerTensorTrain(*x_data) + t3.TuckerTensorTrain.zeros(
            shape, (2, 2, 2, 2), (1, 2, 2, 2, 1))
        ztk = list(xz.data[0])
        for group in sharing.nontrivial_groups(groups):
            for ii in group:
                ztk[ii] = ztk[group[0]]                     # concatenated tied factors are equal
        w, _, _ = bt3svd.t3svd((tuple(ztk), xz.data[1]), sharing=spec, rtol=1e-12)
        wT = t3.TuckerTensorTrain(*w)
        self.assertEqual(wT.tucker_ranks, n)
        self.assertEqual(wT.tt_ranks, r)
        self.assertIs(w[0][0], w[0][1])
        self.assertTrue(np.allclose(np.asarray(wT.to_dense()), x_dense))
        # test 7.5b: tolerance-based ranks obey the tail-energy bound on the ORIGINAL's spectra
        RT = 0.3
        y_data, spec_y, groups_y = _tied_data(SHARED_STRUCTURES[2], ())
        y_dense = np.asarray(t3.TuckerTensorTrain(*y_data).to_dense())
        yt, _, _ = bt3svd.t3svd(y_data, sharing=spec_y, rtol=RT)
        ytT = t3.TuckerTensorTrain(*yt)
        out_norm = float(np.linalg.norm(np.asarray(ytT.to_dense())))
        group = sharing.nontrivial_groups(groups_y)[0]
        k = len(group)
        thresh_g = RT * np.sqrt(k) * out_norm
        tails = lambda ss: np.sqrt(np.cumsum(np.asarray(ss)[::-1] ** 2))[::-1]
        ub_g = max(1, int(np.sum(tails(_dense_group_svals(y_dense, group)) >= thresh_g)))
        self.assertLessEqual(ytT.tucker_ranks[group[0]], ub_g)
        thresh = RT * out_norm                              # singleton Tucker + TT edges
        for ii in [jj for jj in range(4) if jj not in group]:
            s_i = np.linalg.svd(np.moveaxis(y_dense, ii, 0).reshape(y_dense.shape[ii], -1),
                                compute_uv=False)
            self.assertLessEqual(ytT.tucker_ranks[ii], max(1, int(np.sum(tails(s_i) >= thresh))))
        for ii in range(1, 4):
            s_u = np.linalg.svd(y_dense.reshape(int(np.prod(y_dense.shape[:ii])), -1),
                                compute_uv=False)
            self.assertLessEqual(ytT.tt_ranks[ii], max(1, int(np.sum(tails(s_u) >= thresh))))

    def test_stacked_caps_match_per_element(self):
        # caps-only grouped truncation on a stack: each element represents what the same call on
        # the unstacked element represents (tolerances on stacks raise, as unshared)
        x_data, spec, groups = _tied_data(SHARED_STRUCTURES[2], (2,))
        y, _, _ = bt3svd.t3svd(x_data, sharing=spec, max_tucker_ranks=(2, 2, 2, 2),
                               max_tt_ranks=(1, 2, 2, 2, 1))
        yT = t3.TuckerTensorTrain(*y)
        self.assertIs(y[0][0], y[0][1])
        for ee in range(2):
            xe = (tuple(np.asarray(B)[ee] for B in x_data[0]),
                  tuple(np.asarray(G)[ee] for G in x_data[1]))
            ye, _, _ = bt3svd.t3svd(xe, sharing=spec, max_tucker_ranks=(2, 2, 2, 2),
                                    max_tt_ranks=(1, 2, 2, 2, 1))
            self.assertTrue(np.allclose(
                np.asarray(yT.to_dense())[ee],
                np.asarray(t3.TuckerTensorTrain(*ye).to_dense()), atol=1e-9))
        with self.assertRaises(ValueError):
            bt3svd.t3svd(x_data, sharing=spec, rtol=1e-6)

    def test_adjustment_group_ceiling_and_dispatch(self):
        # the grouped lossless reduction respects the group ceiling (n_g may exceed a single
        # mode's local rL*rR ceiling), stays lossless and tied; all-singleton == unshared bitwise
        x_data, spec, groups = _tied_data(SHARED_STRUCTURES[1], ())      # (6,6), group (0,1)
        y, _, _ = bt3svd.t3svd(x_data, sharing=spec, max_tt_ranks=(1, 1, 1))
        z = bt3svd.t3_rank_adjustment_sweep(y, 'right_to_left', sharing=spec)
        zT = t3.TuckerTensorTrain(*z)
        self.assertEqual(zT.tucker_ranks, (2, 2))            # ceiling sum(1*1, 1*1) = 2 < n_g = 3
        self.assertIs(z[0][0], z[0][1])
        self.assertTrue(np.allclose(np.asarray(zT.to_dense()),
                                    np.asarray(t3.TuckerTensorTrain(*y).to_dense())))
        w = t3.TuckerTensorTrain.randn((6, 5), (3, 2), (1, 2, 1))
        ref = bt3svd.t3_rank_adjustment_sweep(w.data, 'right_to_left')
        alt = bt3svd.t3_rank_adjustment_sweep(w.data, 'right_to_left', sharing=('a', 'b'))
        for fam_a, fam_b in zip(alt, ref):
            for A, B in zip(fam_a, fam_b):
                self.assertTrue(np.array_equal(np.asarray(A), np.asarray(B)))

    def test_frontend_safe_mode_tied_precondition(self):
        # untied factors + sharing raise in safe mode; skipped under safety.unsafe()
        x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 2, 2, 1))
        with self.assertRaises(ValueError):
            x.t3svd(sharing=(0, 0, 1))
        with safety.unsafe():
            y, _, _ = x.t3svd(sharing=(0, 0, 1))            # insurance-ties and proceeds
        self.assertIs(y.data[0][0], y.data[0][1])
        x.t3svd(sharing=(0, 1, 2))                          # all-singleton: no tie required


def _untie_representation(x_data):
    """Rotate each mode's factor by a random orthogonal Q (core absorbs Q^T): same tensor,
    untied factors."""
    tk, tt = [list(c) for c in x_data]
    for ii in range(len(tk)):
        n = tk[ii].shape[-2]
        Q = np.linalg.qr(np.random.randn(n, n))[0]
        tk[ii] = Q @ np.asarray(tk[ii])
        tt[ii] = np.einsum('...aub,xu->...axb', np.asarray(tt[ii]), Q)
    return tuple(tk), tuple(tt)


class TestShareTuckerFactors(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test

    def test_exact_recovery_from_unshared_representation(self):
        # test 6: densified-shared recovery -- a shared tensor with an untied representation
        # comes back exactly, tied, at the true shared ranks
        for STRUCTURE in SHARED_STRUCTURES[1:3]:
            with self.subTest(STRUCTURE=STRUCTURE):
                shape, n, r, spec = STRUCTURE
                x_data, _, groups = _tied_data(STRUCTURE, ())
                x_dense = np.asarray(t3.TuckerTensorTrain(*x_data).to_dense())
                x_un = _untie_representation(x_data)
                self.assertGreater(float(np.asarray(
                    sharing.t3_sharing_residual(x_un, spec))), 1e-3)     # genuinely untied
                y, _, _ = bt3svd.t3_share_tucker_factors(x_un, spec, rtol=1e-12)
                yT = t3.TuckerTensorTrain(*y)
                self.assertTrue(np.allclose(np.asarray(yT.to_dense()), x_dense))
                self.assertEqual(yT.tucker_ranks, n)
                self.assertEqual(yT.tt_ranks, r)
                for group in sharing.nontrivial_groups(groups):
                    for ii in group[1:]:
                        self.assertIs(y[0][ii], y[0][group[0]])

    def test_agrees_with_grouped_t3svd_on_shared_input(self):
        # at equal truncation settings (rtol, or caps), share == grouped t3svd on tied input
        x_data, spec, _ = _tied_data(SHARED_STRUCTURES[2], ())
        for kwargs in [dict(rtol=1e-12), dict(max_tucker_ranks=(2, 2, 2, 2))]:
            with self.subTest(kwargs=kwargs):
                a, ska, _ = bt3svd.t3_share_tucker_factors(x_data, spec, **kwargs)
                b, skb, _ = bt3svd.t3svd(x_data, sharing=spec, **kwargs)
                self.assertTrue(np.allclose(
                    np.asarray(t3.TuckerTensorTrain(*a).to_dense()),
                    np.asarray(t3.TuckerTensorTrain(*b).to_dense())))
                self.assertEqual(t3.TuckerTensorTrain(*a).tucker_ranks,
                                 t3.TuckerTensorTrain(*b).tucker_ranks)
                self.assertTrue(np.allclose(np.asarray(ska[0]), np.asarray(skb[0])))

    def test_quasi_optimality_on_low_rank_plus_noise(self):
        # err <= C(d) * ||noise|| when a shared tensor at the target ranks exists
        shape, n, r, spec = SHARED_STRUCTURES[2]
        d = len(shape)
        x_data, _, _ = _tied_data((shape, n, r, spec), ())
        x_un = _untie_representation(x_data)
        noise = t3.TuckerTensorTrain.randn(shape, (2, 2, 2, 2), (1, 2, 2, 2, 1))
        eps = 1e-4 * float(np.linalg.norm(t3.TuckerTensorTrain(*x_data).to_dense())) \
            / float(np.linalg.norm(noise.to_dense()))
        noisy = t3.TuckerTensorTrain(*x_un) + t3.TuckerTensorTrain(
            noise.data[0], (np.asarray(noise.data[1][0]) * eps,) + tuple(noise.data[1][1:]))
        y, _, _ = bt3svd.t3_share_tucker_factors(noisy.data, spec,
                                                 max_tucker_ranks=n, max_tt_ranks=r)
        err = np.linalg.norm(np.asarray(t3.TuckerTensorTrain(*y).to_dense())
                             - np.asarray(noisy.to_dense()))
        C_d = np.sqrt(d) + np.sqrt(d) * np.sqrt(d - 1) + np.sqrt(d - 1)
        self.assertLessEqual(err, C_d * eps * float(np.linalg.norm(noise.to_dense())))

    def test_all_singleton_dispatches_to_plain_t3svd(self):
        x = t3.TuckerTensorTrain.randn((6, 5, 4), (3, 2, 2), (1, 2, 2, 1))
        a, _, _ = bt3svd.t3_share_tucker_factors(x.data, (0, 1, 2))
        b, _, _ = bt3svd.t3svd(x.data)
        for fam_a, fam_b in zip(a, b):
            for A, B in zip(fam_a, fam_b):
                self.assertTrue(np.array_equal(np.asarray(A), np.asarray(B)))

    def test_stacked_caps_match_per_element(self):
        shape, n, r, spec = SHARED_STRUCTURES[2]
        x_data, _, _ = _tied_data((shape, n, r, spec), (2,))
        x_un = _untie_representation(x_data)                 # same rotations across the stack
        y, _, _ = bt3svd.t3_share_tucker_factors(x_un, spec, max_tucker_ranks=(2, 2, 2, 2),
                                                 max_tt_ranks=(1, 2, 2, 2, 1))
        yT = t3.TuckerTensorTrain(*y)
        self.assertIs(y[0][0], y[0][1])
        for ee in range(2):
            xe = (tuple(np.asarray(B)[ee] for B in x_un[0]),
                  tuple(np.asarray(G)[ee] for G in x_un[1]))
            ye, _, _ = bt3svd.t3_share_tucker_factors(xe, spec, max_tucker_ranks=(2, 2, 2, 2),
                                                      max_tt_ranks=(1, 2, 2, 2, 1))
            self.assertTrue(np.allclose(
                np.asarray(yT.to_dense())[ee],
                np.asarray(t3.TuckerTensorTrain(*ye).to_dense()), atol=1e-9))

    def test_frontend_share_method(self):
        x_data, spec, _ = _tied_data(SHARED_STRUCTURES[1], ())
        x_un = t3.TuckerTensorTrain(*_untie_representation(x_data))
        y = x_un.share(spec, rtol=1e-12)
        self.assertIsInstance(y, t3.TuckerTensorTrain)
        self.assertIs(y.data[0][0], y.data[0][1])
        self.assertTrue(np.allclose(np.asarray(y.to_dense()), np.asarray(x_un.to_dense())))


class TestSharedFrameData(unittest.TestCase):
    """Permanent invariants of the shared-frame companion (the S_i machinery)."""
    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test


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


def _dense_of_variations(frame_d, variations_d):
    return np.asarray(tvo.tv_to_dense(frame_d, variations_d)).reshape(-1)


def _dense_tied_projector(frame_d, groups, variation_shapes):
    """Dense orthogonal projector onto the TIED gauged tangent subspace: basis = every TT unit
    direction + every singleton-Tucker unit direction + per-group tied directions S_i^T E for a
    spanning set of gauged ambient E (built independently of the post-pass implementation)."""
    UU, OO = frame_d[0], frame_d[1]
    # centers via a fresh right sweep of the frame's left chain (independent route)
    import t3toolbox.backend.tt_orthogonalization as tth
    _, HH = tth.tt_right_orthogonalize(frame_d[2], return_variation_cores=True)
    tkshapes, ttshapes = variation_shapes
    d = len(tkshapes)
    singleton_modes = [g[0] for g in groups if len(g) == 1]
    basis = []
    zero_vars = lambda: ([np.zeros(s) for s in tkshapes], [np.zeros(s) for s in ttshapes])
    for ii in range(d):
        for kk in range(int(np.prod(ttshapes[ii]))):
            tkv, ttv = zero_vars()
            ttv[ii].flat[kk] = 1.0
            basis.append(_dense_of_variations(frame_d, (tuple(tkv), tuple(ttv))))
    for ii in singleton_modes:
        for kk in range(int(np.prod(tkshapes[ii]))):
            tkv, ttv = zero_vars()
            tkv[ii].flat[kk] = 1.0
            basis.append(_dense_of_variations(frame_d, (tuple(tkv), tuple(ttv))))
    for group in sharing.nontrivial_groups(groups):
        SS = {ii: np.einsum('...aub,...axb->...ux', np.asarray(HH[ii]), np.asarray(OO[ii]))
              for ii in group}
        U_rows = np.asarray(UU[group[0]])
        n_g, N = U_rows.shape[-2], U_rows.shape[-1]
        for aa in range(n_g):
            for bb in range(N):
                E = np.zeros((n_g, N))
                E[aa, bb] = 1.0
                E -= (E @ U_rows.T) @ U_rows                 # gauge the ambient direction
                if np.linalg.norm(E) < 1e-12:
                    continue
                tkv, ttv = zero_vars()
                for ii in group:
                    tkv[ii] = SS[ii].T @ E
                basis.append(_dense_of_variations(frame_d, (tuple(tkv), tuple(ttv))))
    B = np.stack(basis, axis=1)
    return B @ np.linalg.pinv(B, rcond=1e-10)


class TestSharedPostPass(unittest.TestCase):
    """The tied post-pass (manifold Gram/SVD solve + corewise mean) and its threading."""
    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test


    def test_matches_dense_projection(self):
        # matrix test 7a: project a random dense tensor onto the tied tangent subspace through
        # the public projection entry point; compare against the dense orthogonal projector
        shape, n, r, spec = SHARED_STRUCTURES[2]
        x_data, _, groups = _tied_data((shape, n, r, spec), ())
        frame_d, _ = fvc.t3_orthogonal_representations(x_data)
        sfd = sharing.fv_shared_frame_data(frame_d, groups)
        tkshapes = tuple((O.shape[-2], U.shape[-1]) for O, U in zip(frame_d[1], frame_d[0]))
        ttshapes = tuple((L.shape[-3], U.shape[-2], R.shape[-1])
                         for L, U, R in zip(frame_d[2], frame_d[0], frame_d[3]))
        P = _dense_tied_projector(frame_d, groups, (tkshapes, ttshapes))
        Z = np.random.randn(*shape)
        var_tied = tvo.tv_project_dense_onto_tangent_space(frame_d, Z, shared_data=sfd)
        lhs = _dense_of_variations(frame_d, var_tied)
        rhs = P @ Z.reshape(-1)
        self.assertLess(np.linalg.norm(lhs - rhs), 1e-9 * np.linalg.norm(rhs))

    def test_idempotent_gauged_fixed_point_and_recovery(self):
        # idempotence; gauge preservation; exactly-tied inputs are fixed points; a constructed
        # tied tangent's ambient direction is recovered through the tied coordinates
        for STRUCTURE in SHARED_STRUCTURES[1:]:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(STRUCTURE=STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x_data, spec, groups = _tied_data(STRUCTURE, STACK_SHAPE)
                    frame_d, _ = fvc.t3_orthogonal_representations(x_data)
                    sfd = sharing.fv_shared_frame_data(frame_d, groups)
                    raw = ([np.random.randn(*(STACK_SHAPE + (O.shape[-2], U.shape[-1])))
                            for O, U in zip(frame_d[1], frame_d[0])],
                           [np.random.randn(*(STACK_SHAPE + H.shape[-3:])) for H in frame_d[2]])
                    v1 = tvo.tv_orthogonal_gauge_projection(frame_d, (tuple(raw[0]), tuple(raw[1])),
                                                            shared_data=sfd)
                    gauge_res = np.asarray(tvo.tv_gauge_residual(frame_d, v1))
                    self.assertTrue(np.all(gauge_res <= 1e-9))
                    v2 = sharing.fv_share_tucker_variations(v1, sfd)
                    for A, B in zip(v2[0], v1[0]):
                        self.assertTrue(np.allclose(np.asarray(A), np.asarray(B), atol=1e-10))

    def test_degenerate_point_min_norm(self):
        # at a zero-padded shared point, the tied projection puts exactly zero in the gated new
        # directions (the clipped pinv's min-norm solution) -- the restart analysis, verified P3
        shape, n, r, spec = SHARED_STRUCTURES[2]
        x_data, _, groups = _tied_data((shape, n, r, spec), ())
        xp = t3.TuckerTensorTrain(*x_data).resize(shape, (4, 4, 4, 2), r)
        frame_d, _ = fvc.t3_orthogonal_representations(xp.data)
        sfd = sharing.fv_shared_frame_data(frame_d, groups)
        Z = np.random.randn(*shape)
        var_tied = tvo.tv_project_dense_onto_tangent_space(frame_d, Z, shared_data=sfd)
        group = sharing.nontrivial_groups(groups)[0]
        OO = frame_d[1]
        import t3toolbox.backend.tt_orthogonalization as tth
        _, HH = tth.tt_right_orthogonalize(frame_d[2], return_variation_cores=True)
        M = np.concatenate([np.einsum('axb,aub->xu', np.asarray(OO[ii]), np.asarray(HH[ii]))
                            for ii in group], axis=0)        # stacked S^T, (sum nD, 4)
        Vstack = np.concatenate([np.asarray(var_tied[0][ii]) for ii in group], axis=0)
        Udot = np.linalg.lstsq(M, Vstack, rcond=1e-10)[0]    # (4, N): the tied ambient direction
        self.assertEqual(float(np.linalg.norm(Udot[3:, :])), 0.0)   # gated new direction: exactly 0
        self.assertLess(float(np.linalg.norm(M @ Udot - Vstack)),   # tied coords in range(M)
                        1e-9 * (1 + float(np.linalg.norm(Vstack))))

    def test_mean_post_pass_and_geometry_separation(self):
        # corewise mean: exact tie by identity, drift fixed point; and on generic (unequal-S)
        # gauged input the mean and the Gram/SVD solve produce DIFFERENT projections -- each is
        # the right one in its own geometry only
        x_data, spec, groups = _tied_data(SHARED_STRUCTURES[2], ())
        frame_d, _ = fvc.t3_orthogonal_representations(x_data)
        sfd = sharing.fv_shared_frame_data(frame_d, groups)
        raw = ([np.random.randn(O.shape[-2], U.shape[-1])
                for O, U in zip(frame_d[1], frame_d[0])],
               [np.random.randn(*H.shape[-3:]) for H in frame_d[2]])
        gauged = tvo.tv_orthogonal_gauge_projection(frame_d, (tuple(raw[0]), tuple(raw[1])))
        tied_gram = sharing.fv_share_tucker_variations(gauged, sfd)
        # mean: requires equal shapes within the group -- give the corewise-style variations
        # (raw core perturbations at the (U,G,G,G) frame have full nU rows)
        corewise_vars = ([np.random.randn(U.shape[-2], U.shape[-1]) for U in x_data[0]],
                         [np.random.randn(*G.shape[-3:]) for G in x_data[1]])
        tied_mean = sharing.fv_mean_tucker_variations(
            (tuple(corewise_vars[0]), tuple(corewise_vars[1])), groups)
        group = sharing.nontrivial_groups(groups)[0]
        self.assertIs(tied_mean[0][group[0]], tied_mean[0][group[1]])
        ref = sum(np.asarray(corewise_vars[0][ii]) for ii in group) / len(group)
        self.assertTrue(np.allclose(np.asarray(tied_mean[0][group[0]]), ref))
        tied_twice = sharing.fv_mean_tucker_variations(tied_mean, groups)
        self.assertTrue(np.array_equal(np.asarray(tied_twice[0][group[0]]),
                                       np.asarray(tied_mean[0][group[0]])))
        # geometry separation, strongest form: on THIS structure the gauged manifold
        # coordinates have different shapes across the group (nD_i = min(n, rL_i*rR_i) differ),
        # so the arithmetic mean is not even well-formed on them
        self.assertNotEqual(np.asarray(gauged[0][group[0]]).shape,
                            np.asarray(gauged[0][group[1]]).shape)
        # ...and where shapes DO match (a symmetric-bond structure), the mean of the gauged
        # coordinates differs from the Gram/SVD-tied coordinates by a large factor
        y_data, spec_y, groups_y = _tied_data(SHARED_STRUCTURES[1], ())     # nD = (2, 2)
        frame_y, _ = fvc.t3_orthogonal_representations(y_data)
        sfd_y = sharing.fv_shared_frame_data(frame_y, groups_y)
        raw_y = ([np.random.randn(O.shape[-2], U.shape[-1])
                  for O, U in zip(frame_y[1], frame_y[0])],
                 [np.random.randn(*H.shape[-3:]) for H in frame_y[2]])
        gauged_y = tvo.tv_orthogonal_gauge_projection(frame_y, (tuple(raw_y[0]), tuple(raw_y[1])))
        tied_y = sharing.fv_share_tucker_variations(gauged_y, sfd_y)
        group_y = sharing.nontrivial_groups(groups_y)[0]
        mean_of_gauged = sum(np.asarray(gauged_y[0][ii]) for ii in group_y) / len(group_y)
        rel_gap = (np.linalg.norm(np.asarray(tied_y[0][group_y[0]]) - mean_of_gauged)
                   / np.linalg.norm(mean_of_gauged))
        self.assertGreater(float(rel_gap), 0.05)

    def test_threading_composition(self):
        # tv_project_t3/dense with shared_data == (project without) then fv_share, exactly
        x_data, spec, groups = _tied_data(SHARED_STRUCTURES[1], ())
        frame_d, _ = fvc.t3_orthogonal_representations(x_data)
        sfd = sharing.fv_shared_frame_data(frame_d, groups)
        other = t3.TuckerTensorTrain.randn((6, 6), (2, 2), (1, 2, 1))
        a = tvo.tv_project_t3_onto_tangent_space(frame_d, other.data, shared_data=sfd)
        b = sharing.fv_share_tucker_variations(
            tvo.tv_project_t3_onto_tangent_space(frame_d, other.data), sfd)
        for fam_a, fam_b in zip(a, b):
            for A, B in zip(fam_a, fam_b):
                self.assertTrue(np.array_equal(np.asarray(A), np.asarray(B)))

    def test_k_over_c_broadcast(self):
        # matrix test 13 (post-pass slice): one frame (C=()) with a K-stack of variations --
        # the companion (C axes) broadcasts against the K+C variations; per-K-element agreement
        x_data, spec, groups = _tied_data(SHARED_STRUCTURES[2], ())
        frame_d, _ = fvc.t3_orthogonal_representations(x_data)
        sfd = sharing.fv_shared_frame_data(frame_d, groups)
        K = 3
        tkv = tuple(np.random.randn(K, O.shape[-2], U.shape[-1])
                    for O, U in zip(frame_d[1], frame_d[0]))
        ttv = tuple(np.random.randn(K, *H.shape[-3:]) for H in frame_d[2])
        stacked = sharing.fv_share_tucker_variations((tkv, ttv), sfd)
        for kk in range(K):
            single = sharing.fv_share_tucker_variations(
                (tuple(v[kk] for v in tkv), tuple(v[kk] for v in ttv)), sfd)
            for A, B in zip(stacked[0], single[0]):
                self.assertTrue(np.allclose(np.asarray(A)[kk], np.asarray(B), atol=1e-12))
        # and a C-stacked frame with K+C variations
        xc_data, _, groups_c = _tied_data(SHARED_STRUCTURES[2], (2,))
        frame_c, _ = fvc.t3_orthogonal_representations(xc_data)
        sfd_c = sharing.fv_shared_frame_data(frame_c, groups_c)
        tkv_c = tuple(np.random.randn(K, 2, O.shape[-2], U.shape[-1])
                      for O, U in zip(frame_c[1], frame_c[0]))
        ttv_c = tuple(np.random.randn(K, 2, *H.shape[-3:]) for H in frame_c[2])
        out = sharing.fv_share_tucker_variations((tkv_c, ttv_c), sfd_c)
        self.assertEqual(np.asarray(out[0][0]).shape[:2], (K, 2))


class TestSharedGeometry(unittest.TestCase):
    """The frontend wrapper: retraction axioms, gradient consistency, optimizer integration."""
    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test


    @staticmethod
    def _tied_point(structure):
        x_data, spec, groups = _tied_data(structure, ())
        return t3.TuckerTensorTrain(*x_data), spec, groups

    def test_wrapper_identity_and_hash(self):
        a, b = sg.shared_manifold((0, 0, 1)), sg.shared_manifold((0, 0, 1))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, sg.shared_manifold((0, 1, 1)))
        self.assertNotEqual(a, sg.shared_corewise((0, 0, 1)))
        with self.assertRaises(ValueError):
            sg.shared('not a geometry', (0, 0))

    def test_retraction_axioms(self):
        # matrix test 10: retract(0) == x; second-order agreement (manifold) / exactness
        # (corewise); output exactly tied (one array per group); both geometries
        x, spec, groups = self._tied_point(SHARED_STRUCTURES[2])
        x_dense = np.asarray(x.to_dense())
        group = sharing.nontrivial_groups(groups)[0]
        for geom in [sg.shared_manifold(spec), sg.shared_corewise(spec)]:
            with self.subTest(base=geom.base_name):
                frame = geom.frame(x)
                y0 = geom.retract(t3m.T3Tangent.zeros(frame))
                self.assertTrue(np.allclose(np.asarray(y0.to_dense()), x_dense))
                self.assertIs(y0.data[0][group[0]], y0.data[0][group[1]])
                v = geom.randn(frame)
                errs = []
                for tval in (1e-2, 1e-3):
                    yt = geom.retract(tval * v)
                    target = x_dense + tval * np.asarray(v.to_dense())
                    errs.append(float(np.linalg.norm(np.asarray(yt.to_dense()) - target)))
                    self.assertIs(yt.data[0][group[0]], yt.data[0][group[1]])
                # first-order retraction on BOTH geometries: the tensor is multilinear in the
                # cores, so even the additive corewise chart agrees with x + t*v only to O(t^2)
                self.assertLess(errs[1], errs[0] / 50.0)      # a decade of t -> ~100x
                if geom.base_name == 'corewise':
                    # ...but it is EXACT in parameter space: cores == x.cores + t * variations
                    tval = 1e-2
                    yt = geom.retract(tval * v)
                    for ii, (C, C0, dC) in enumerate(zip(yt.data[1], x.data[1],
                                                         v.variations.tt_variations)):
                        self.assertTrue(np.allclose(np.asarray(C),
                                                    np.asarray(C0) + tval * np.asarray(dC)))
                if geom.base_name == 'manifold':
                    self.assertEqual(y0.tucker_ranks, x.tucker_ranks)
                    self.assertEqual(y0.tt_ranks, x.tt_ranks)

    def test_gradient_consistency(self):
        # matrix test 8: finite differences of f(retract(x, t*xi)) at t=0 vs inner(grad, xi),
        # through GaussNewtonModel, both geometries
        x, spec, _ = self._tied_point(SHARED_STRUCTURES[1])
        ww = [np.random.randn(40, N) for N in x.shape]
        ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
        b = np.random.randn(40)
        for geom in [sg.shared_manifold(spec), sg.shared_corewise(spec)]:
            with self.subTest(base=geom.base_name):
                r = np.asarray(x.apply(ww)) - b
                model = fitting.apply_model(geom, x, ww, r)
                g = model.gradient
                xi = geom.randn(model.frame)
                h = 1e-6

                def f_of(t):
                    y = geom.retract(t * xi) if t != 0.0 else x
                    return 0.5 * float(np.sum((np.asarray(y.apply(ww)) - b) ** 2))

                fd = (f_of(h) - f_of(-h)) / (2 * h)
                ip = float(g.corewise_inner(xi))
                self.assertLess(abs(fd - ip), 1e-4 * max(1.0, abs(ip)))

    def test_fitting_model_gates_and_tied_pipeline(self):
        # the model factories accept the wrapper; the gradient and Hessian actions are tied
        # (fixed points of the post-pass); the companion rides the model once per frame
        x, spec, groups = self._tied_point(SHARED_STRUCTURES[2])
        ww = [np.random.randn(60, N) for N in x.shape]
        r = np.asarray(x.apply(ww)) - np.random.randn(60)
        geom = sg.shared_manifold(spec)
        model = fitting.apply_model(geom, x, ww, r)
        self.assertIsInstance(model.geometry_aux, sharing.T3SharedFrameData)
        g = model.gradient
        tied_again = sharing.fv_share_tucker_variations(g.variations.data, model.geometry_aux)
        for A, B in zip(tied_again[0], g.variations.tucker_variations):
            self.assertTrue(np.allclose(np.asarray(A), np.asarray(B), atol=1e-9))
        Hp = model.gn_hessian(geom.randn(model.frame))
        tied_H = sharing.fv_share_tucker_variations(Hp.variations.data, model.geometry_aux)
        for A, B in zip(tied_H[0], Hp.variations.tucker_variations):
            self.assertTrue(np.allclose(np.asarray(A), np.asarray(B), atol=1e-9))
        # regularizer path composes (the backend GeometryOps mapping accepts the wrapper)
        model_reg = fitting.apply_model(geom, x, ww, r,
                                        regularizer=optimizers.IdentityRegularizer(1e-3))
        self.assertGreater(float(model_reg.objective_value), float(model.objective_value))
        _ = model_reg.gradient

    def test_transport_and_project_ambient(self):
        x, spec, groups = self._tied_point(SHARED_STRUCTURES[1])
        geom = sg.shared_manifold(spec)
        frame = geom.frame(x)
        v = geom.randn(frame)
        y, spec2, _ = self._tied_point(SHARED_STRUCTURES[1])
        new_frame = geom.frame(y)
        w = geom.transport(v, new_frame)
        sfd_new = geom.shared_frame_data(new_frame)
        tied_w = sharing.fv_share_tucker_variations(w.variations.data, sfd_new)
        for A, B in zip(tied_w[0], w.variations.tucker_variations):
            self.assertTrue(np.allclose(np.asarray(A), np.asarray(B), atol=1e-9))
        w2 = geom.project_ambient(new_frame, v.to_t3())
        for A, B in zip(w2.variations.tucker_variations, w.variations.tucker_variations):
            self.assertTrue(np.allclose(np.asarray(A), np.asarray(B)))

    def test_safe_mode_preconditions(self):
        x_untied = t3.TuckerTensorTrain.randn((6, 6), (3, 3), (1, 2, 1))
        geom = sg.shared_manifold((0, 0))
        with self.assertRaises(ValueError):
            geom.frame(x_untied)
        x, spec, _ = self._tied_point(SHARED_STRUCTURES[1])
        frame = geom.frame(x)
        v_untied = t3m.MANIFOLD.randn(frame)                 # gauged but NOT tied
        with self.assertRaises(ValueError):
            geom.retract(v_untied)
        with safety.unsafe():
            y = geom.retract(v_untied)                       # tied-projects and proceeds
        self.assertIs(y.data[0][0], y.data[0][1])

    def test_end_to_end_recovery_iterates_stay_tied(self):
        # matrix test 11 (ragged): recover a tied target; every Newton iterate stays tied
        shape, n, r = (6, 6, 6), (2, 2, 2), (1, 2, 2, 1)
        A0 = t3.TuckerTensorTrain.randn(shape, n, r)
        tk, tt = A0.data
        A = t3.TuckerTensorTrain((tk[0],) * 3, tt)
        Ad = np.asarray(A.to_dense())
        ww = [np.random.randn(150, N) for N in shape]
        ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
        b = A.apply(ww)
        spec = (0, 0, 0)
        tied_per_iter = []

        def cb(info):
            tied_per_iter.append(bool(np.all(np.asarray(
                sharing.t3_tucker_factors_shared(info.x_cores, spec)))))

        x0 = t3.TuckerTensorTrain.zeros(shape, n, r)
        x_fit, stats = optimizers.newton_cg(sg.shared_manifold(spec), 'apply', ww, b, x0,
                                            max_newton=25, callback=cb)
        rel = float(np.linalg.norm(np.asarray(x_fit.to_dense()) - Ad) / np.linalg.norm(Ad))
        self.assertLess(rel, 1e-6)
        self.assertTrue(all(tied_per_iter) and len(tied_per_iter) > 3)
        self.assertIs(x_fit.data[0][0], x_fit.data[0][1])
        # corewise: adam from a small tied random start (zero is a critical point of the
        # multilinear parametrization -- a corewise fact, nothing to do with sharing)
        x0c0 = t3.TuckerTensorTrain.randn(shape, n, r)
        tkc, ttc = x0c0.data
        x0c = t3.TuckerTensorTrain((tkc[0],) * 3, tuple(0.3 * np.asarray(G) for G in ttc))
        xc, _ = optimizers.adam(sg.shared_corewise(spec), 'apply', ww, b, x0c,
                                np.random.default_rng(0), batch=64, lr=3e-2, max_iter=1500)
        relc = float(np.linalg.norm(np.asarray(xc.to_dense()) - Ad) / np.linalg.norm(Ad))
        self.assertLess(relc, 1e-4)
        self.assertIs(xc.data[0][0], xc.data[0][1])


class TestSharedContinuation(unittest.TestCase):
    """Matrix 14: shared rank continuation on real tensors -- the kappa_g spectrum properties
    (symmetric degeneration, the mediant bound), the tied zero-padded restart and its gated
    channels, the two-step escape, and the end-to-end continuation loop. The synthetic-spectra
    growth-rule tests live in tests/backend/test_ranks.py."""

    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test

    @staticmethod
    def _tied_target(shape, tucker_ranks, tt_ranks, groups):
        x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks)
        tk = list(x.tucker_cores)
        for group in groups:
            for ii in group[1:]:
                tk[ii] = tk[group[0]]
        return t3.TuckerTensorTrain(tuple(tk), x.tt_cores)

    def test_symmetric_degeneration(self):
        # 14c (cor:sym): on an exactly symmetric tensor the group Grams are all equal, so
        # s_g = sqrt(k) * sigma elementwise and kappa_g EQUALS the per-mode condition number
        i, j, k = np.ogrid[1:7, 1:7, 1:7]
        A = 1.0 / (i + j + k + 1)                                        # the Hilbert tensor: symmetric
        xs = t3.TuckerTensorTrain.t3svd_dense(A)[0].share((0, 0, 0))     # exact tied representation
        _, sk_g, _ = xs.t3svd(sharing=(0, 0, 0))
        _, sk_u, _ = xs.t3svd()
        for m in range(3):
            with self.subTest(mode=m):
                s_g, s_u = np.asarray(sk_g[m]), np.asarray(sk_u[m])
                self.assertEqual(s_g.size, s_u.size)
                self.assertLess(np.max(np.abs(s_g - np.sqrt(3.0) * s_u)), 1e-12 * float(s_g[0]))
        kappa_g = float(sk_g[0][0] / sk_g[0][-1])
        kappa_loc = float(sk_u[0][0] / sk_u[0][-1])
        self.assertLess(abs(kappa_g - kappa_loc), 1e-6 * kappa_loc)

    def test_mediant_bound_and_complementary_gain(self):
        # 14d (prop:mediant): kappa_g <= max per-mode kappa always; strictly (and arbitrarily) smaller
        # when the group spectra are complementary -- a direction is well-determined if SOME mode
        # informs it
        for trial in range(10):
            x = self._tied_target((6, 6, 5), (3, 3, 2), (1, 3, 3, 1), ((0, 1),))
            _, s_g, _ = x.t3svd(sharing=(0, 0, 1))
            _, s_u, _ = x.t3svd()
            kappa_g = float(s_g[0][0] / s_g[0][-1])
            kappa_loc_max = max(float(s_u[m][0] / s_u[m][-1]) for m in (0, 1))
            self.assertLessEqual(kappa_g, kappa_loc_max * (1 + 1e-9))
        # the constructed complementary case: Gamma_0 ~ diag(1, eps^2), Gamma_1 ~ diag(eps^2, 1)
        # -> per-mode kappas 1/eps but kappa_g = 1 exactly
        eps = 1e-4
        p, q = np.random.randn(5), np.random.randn(5)
        T = np.zeros((2, 2, 5))
        T[0, 1, :] = p / np.linalg.norm(p)
        T[1, 0, :] = eps * q / np.linalg.norm(q)
        xt = t3.TuckerTensorTrain.t3svd_dense(T)[0].share((0, 0, 1))
        _, sc_g, _ = xt.t3svd(sharing=(0, 0, 1))
        _, sc_u, _ = xt.t3svd()
        kappa_g = float(sc_g[0][0] / sc_g[0][-1])
        kappa_loc = min(float(sc_u[m][0] / sc_u[m][-1]) for m in (0, 1))
        self.assertLess(kappa_g, 2.0)
        self.assertGreater(kappa_loc, 1e3)

    def test_padded_restart_tied_dense_equal_and_gated(self):
        # 14f + P1: resize(sharing=) pads the group factor ONCE (one array per group), preserves the
        # tensor, and the fresh directions carry exactly-zero group-spectrum levels (the tied Tucker
        # channel is gated at the restart)
        spec = (0, 0, 1)
        x = self._tied_target((6, 6, 5), (2, 2, 2), (1, 2, 2, 1), ((0, 1),))
        xp = x.resize(x.shape, (3, 3, 2), (1, 3, 2, 1), sharing=spec)
        self.assertIs(xp.tucker_cores[0], xp.tucker_cores[1])
        self.assertTrue(np.allclose(np.asarray(xp.to_dense()), np.asarray(x.to_dense())))
        _, s_pad, _ = xp.t3svd(sharing=spec)
        self.assertEqual(np.asarray(s_pad[0]).size, 3)
        self.assertEqual(float(np.asarray(s_pad[0])[-1]), 0.0)          # exactly zero, not merely small
        # the companion sees the same gating: svd_s new level exactly zero
        geom = sg.shared_manifold(spec)
        sfd = geom.shared_frame_data(geom.frame(xp))
        self.assertEqual(float(np.asarray(sfd.svd_s[0])[-1]), 0.0)
        # untied input is rejected in safe mode (the padded factor would silently overwrite mode 1)
        x_untied = t3.TuckerTensorTrain.randn((6, 6, 5), (2, 2, 2), (1, 2, 2, 1))
        with self.assertRaises(ValueError):
            x_untied.resize(x_untied.shape, (3, 3, 2), (1, 3, 2, 1), sharing=spec)

    def test_restart_escape_activates_new_directions(self):
        # 14g (rem:restart): from a zero-padded shared start the tied Tucker channel is gated, but the
        # untied TT-variation channel activates the new group directions within two Newton steps --
        # s_g's fresh level goes from exactly 0 to O(1) mass
        shape, spec = (6, 6, 5), (0, 0, 1)
        A = self._tied_target(shape, (3, 3, 3), (1, 3, 3, 1), ((0, 1),))
        ww = [np.random.randn(150, N) for N in shape]
        ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
        b = A.apply(ww)
        x_lo, _, _ = A.t3svd(max_tucker_ranks=2, max_tt_ranks=2, sharing=spec)
        xp = x_lo.resize(shape, (3, 3, 3), (1, 3, 3, 1), sharing=spec)
        _, s_pad, _ = xp.t3svd(sharing=spec)
        self.assertEqual(float(np.asarray(s_pad[0])[-1]), 0.0)
        x2, _ = optimizers.newton_cg(sg.shared_manifold(spec), 'apply', ww, b, xp, max_newton=2)
        _, s_2, _ = x2.t3svd(sharing=spec)
        self.assertGreater(float(np.asarray(s_2[0])[-1]), 1e-3 * float(np.asarray(s_2[0])[0]))
        self.assertTrue(bool(np.all(np.asarray(x2.has_shared_tucker_factors(spec)))))

    def test_end_to_end_continuation_reaches_true_shared_ranks(self):
        # 14h: the full outer loop -- fit, grow (grouped), zero-padded tied restart -- recovers a tied
        # target at exactly its true shared ranks. g0norm_newton is pinned across levels per the
        # docs/rank_continuation.md warm-start guidance (a padded restart's initial gradient is small).
        shape, spec = (6, 6, 5), (0, 0, 1)
        A = self._tied_target(shape, (3, 3, 3), (1, 3, 3, 1), ((0, 1),))
        Ad = np.asarray(A.to_dense())
        ww = [np.random.randn(150, N) for N in shape]
        ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
        b = A.apply(ww)
        X = t3.TuckerTensorTrain.zeros(shape, (1, 1, 1), (1, 1, 1, 1))
        g0 = None
        rel = np.inf
        for level in range(6):
            kwargs = dict(max_newton=30)
            if g0 is not None:
                kwargs['g0norm_newton'] = g0
            X, stats = optimizers.newton_cg(sg.shared_manifold(spec), 'apply', ww, b, X, **kwargs)
            if g0 is None:
                g0 = stats['history'][0]['gnorm']
            self.assertTrue(bool(np.all(np.asarray(X.has_shared_tucker_factors(spec)))))
            rel = float(np.linalg.norm(np.asarray(X.to_dense()) - Ad) / np.linalg.norm(Ad))
            if rel < 1e-8:
                break
            new_n, new_r = X.continuation_ranks(sharing=spec)
            if (new_n, new_r) == (X.tucker_ranks, X.tt_ranks):
                break
            X = X.resize(shape, new_n, new_r, sharing=spec)
        self.assertLess(rel, 1e-8)
        self.assertEqual((X.tucker_ranks, X.tt_ranks), ((3, 3, 3), (1, 3, 3, 1)))
        self.assertIs(X.data[0][0], X.data[0][1])


class TestUniformSharedTangent(unittest.TestCase):
    """Slice 10: the uniform companion (``ufv_shared_frame_data``) + the tied post-passes and tied
    retraction (``utv_*`` ``shared_data=``). Verified GAUGE-INVARIANTLY: each layer builds its own
    frame (padded SVDs choose their own sign gauge, so raw coordinates are not comparable across
    layers), and the comparisons are on represented DENSE tangents/points plus the gauge-invariant
    group spectrum. Structures are shared-minimal (the fitting pipeline's guarantee via
    ``uniform_minimal`` -- at non-minimal ranks the two layers legitimately build different frames)."""

    # (shape, tucker_ranks, tt_ranks, sharing) -- all shared-minimal (asserted in setUp)
    _CASES = [
        ((6, 6, 5),    (3, 3, 2),    (1, 3, 2, 1),    (0, 0, 1)),
        ((5, 6, 5, 6), (2, 3, 2, 3), (1, 2, 3, 2, 1), ('a', 'b', 'a', 'b')),   # unequal nD in a group
        ((6, 6, 4),    (4, 4, 2),    (1, 2, 2, 1),    (0, 0, 1)),              # the group-ceiling case
        ((7, 7, 7),    (4, 4, 4),    (1, 3, 3, 1),    (0, 0, 0)),              # all modes one group
    ]

    def setUp(self):
        np.random.seed(0)
        import t3toolbox.backend.ranks as ranks
        for shape, n, r, spec in self._CASES:
            assert ranks.compute_minimal_ranks(shape, n, r, sharing=spec) == (n, r)

    @staticmethod
    def _frames_and_companions(shape, n, r, spec):
        import t3toolbox.backend.ufv_conversions as ufvc
        x = _tied_t3(shape, n, r, spec)
        groups = sharing.validate_sharing(spec, shape)
        u = ut3.UniformTuckerTensorTrain.from_t3(x)
        frame_r, _ = fvc.t3_orthogonal_representations(x.data)
        frame_u, _ = ufvc.ut3_orthogonal_representations(u.data)
        sfd_r = sharing.fv_shared_frame_data(frame_r, groups)
        sfd_u = sharing.ufv_shared_frame_data(frame_u, groups)
        return x, u, frame_r, frame_u, sfd_r, sfd_u, groups

    @staticmethod
    def _tvdense(frame_d, var_d):
        return np.asarray(t3.TuckerTensorTrain(*tvo.tv_to_t3(frame_d, var_d)).to_dense())

    def test_companion_matches_ragged_and_dense(self):
        # svd_s (gauge-invariant) == the ragged companion's == the dense concatenated-matricization
        # spectrum; the padded tail is exactly zero (the raw re-sweep's completion rows are
        # orthogonal to the centers' row space)
        for shape, n, r, spec in self._CASES:
            with self.subTest(shape=shape, sharing=spec):
                x, _, _, _, sfd_r, sfd_u, groups = self._frames_and_companions(shape, n, r, spec)
                x_dense = np.asarray(x.to_dense())
                for gi, group in enumerate(sharing.nontrivial_groups(groups)):
                    s_r, s_u = np.asarray(sfd_r.svd_s[gi]), np.asarray(sfd_u.svd_s[gi])
                    self.assertLess(float(np.linalg.norm(s_u[:s_r.size] - s_r)), 1e-9 * s_r[0])
                    if s_u.size > s_r.size:
                        self.assertLess(float(np.abs(s_u[s_r.size:]).max()), 1e-9 * s_r[0])
                    s_dense = _dense_group_svals(x_dense, group)
                    self.assertLess(float(np.linalg.norm(s_u[:s_r.size] - s_dense[:s_r.size])),
                                    1e-9 * s_dense[0])

    def test_tied_projection_matches_ragged(self):
        # project a fixed ambient T3 onto the tied tangent space in each layer; compare the
        # represented dense tangents. Also: the shared_data= threading through the gauge projection
        # equals the separate post-pass, and the post-pass is idempotent.
        import t3toolbox.backend.ufv_conversions as ufvc
        import t3toolbox.backend.utv_operations as utvo
        for shape, n, r, spec in self._CASES:
            with self.subTest(shape=shape, sharing=spec):
                x, u, frame_r, frame_u, sfd_r, sfd_u, groups = \
                    self._frames_and_companions(shape, n, r, spec)
                z = t3.TuckerTensorTrain.randn(shape, n, r)
                proj_r = tvo.tv_project_t3_onto_tangent_space(frame_r, z.data, shared_data=sfd_r)
                zu = ut3.UniformTuckerTensorTrain.from_t3(
                    z, n=u.tucker_supercore.shape[-2], r=u.tt_supercore.shape[-1])
                raw_u = utvo.utv_project_ut3_onto_tangent_space(frame_u, zu.data)
                proj_u = sharing.ufv_share_tucker_variations(raw_u, sfd_u)
                d_r = self._tvdense(frame_r, proj_r)
                d_u = self._tvdense(ufvc.ut3frame_to_t3frame(frame_u),
                                    ufvc.ut3variations_to_t3variations(proj_u))
                self.assertLess(float(np.linalg.norm(d_u - d_r)), 1e-9 * np.linalg.norm(d_r))
                # threading: gauge projection with shared_data == gauge then the post-pass
                gauged_then_tied = utvo.utv_orthogonal_gauge_projection(frame_u, raw_u,
                                                                        shared_data=sfd_u)
                separate = sharing.ufv_share_tucker_variations(
                    utvo.utv_orthogonal_gauge_projection(frame_u, raw_u), sfd_u)
                self.assertTrue(np.allclose(np.asarray(gauged_then_tied[0]),
                                            np.asarray(separate[0])))
                # idempotent
                twice = sharing.ufv_share_tucker_variations(proj_u, sfd_u)
                self.assertTrue(np.allclose(np.asarray(twice[0]), np.asarray(proj_u[0]),
                                            atol=1e-12 * float(np.linalg.norm(d_r))))

    def test_tied_retract_matches_ragged(self):
        import t3toolbox.backend.ufv_conversions as ufvc
        import t3toolbox.backend.utv_operations as utvo
        from t3toolbox.uniform_tucker_tensor_train import _from_data
        for shape, n, r, spec in self._CASES:
            with self.subTest(shape=shape, sharing=spec):
                x, u, frame_r, frame_u, sfd_r, sfd_u, groups = \
                    self._frames_and_companions(shape, n, r, spec)
                z = t3.TuckerTensorTrain.randn(shape, n, r)
                proj_r = tvo.tv_project_t3_onto_tangent_space(frame_r, z.data, shared_data=sfd_r)
                zu = ut3.UniformTuckerTensorTrain.from_t3(
                    z, n=u.tucker_supercore.shape[-2], r=u.tt_supercore.shape[-1])
                proj_u = sharing.ufv_share_tucker_variations(
                    utvo.utv_project_ut3_onto_tangent_space(frame_u, zu.data), sfd_u)
                y_r = t3.TuckerTensorTrain(*tvo.tv_retract(frame_r, proj_r, shared_data=sfd_r))
                y_u_data = utvo.utv_retract(frame_u, proj_u, shared_data=sfd_u)
                y_u = _from_data(y_u_data)
                self.assertLess(
                    float(np.linalg.norm(np.asarray(y_u.to_dense()) - np.asarray(y_r.to_dense()))),
                    1e-9 * float(np.linalg.norm(np.asarray(y_r.to_dense()))))
                self.assertEqual(tuple(int(v) for v in y_u.tucker_ranks), y_r.tucker_ranks)
                self.assertEqual(tuple(int(v) for v in y_u.tt_ranks), y_r.tt_ranks)
                self.assertEqual(float(sharing.ut3_sharing_residual(y_u_data, spec)), 0.0)

    def test_mean_post_pass_matches_ragged_slices(self):
        # the corewise twin: the uniform drift-form mean == the ragged mean on the masked slices
        shape, n, r, spec = self._CASES[0]
        _, u, _, frame_u, _, _, groups = self._frames_and_companions(shape, n, r, spec)
        import t3toolbox.backend.ufv_conversions as ufvc
        _, var_u = ufvc.ut3_orthogonal_representations(u.data)
        tkv = np.random.randn(*np.asarray(var_u[0]).shape)
        var_d = (tkv, var_u[1], var_u[2], var_u[3])
        out = sharing.ufv_mean_tucker_variations(var_d, groups)
        import t3toolbox.backend.ufv_masking as ufv_masking
        tkv_m, _ = ufv_masking.ufv_apply_variations_masks(var_d)
        ref, _ = sharing.fv_mean_tucker_variations((tkv_m, var_u[1]), groups)
        for ii in range(len(shape)):
            self.assertTrue(np.array_equal(np.asarray(out[0][ii]), np.asarray(ref[ii])))

    def test_garbage_robust_tangent_ops(self):
        # garbage in the VARIATION padding must not change the tied projection or retraction
        # (the frame is used as stored -- its padding is part of the construction, by design)
        import t3toolbox.backend.ufv_masking as ufv_masking
        import t3toolbox.backend.utv_operations as utvo
        shape, n, r, spec = self._CASES[1]                       # unequal nD: real padding to corrupt
        _, u, _, frame_u, _, sfd_u, groups = self._frames_and_companions(shape, n, r, spec)
        z = t3.TuckerTensorTrain.randn(shape, n, r)
        zu = ut3.UniformTuckerTensorTrain.from_t3(
            z, n=u.tucker_supercore.shape[-2], r=u.tt_supercore.shape[-1])
        proj = utvo.utv_project_ut3_onto_tangent_space(frame_u, zu.data)
        ind = ufv_masking.ufv_apply_variations_masks(
            (np.ones_like(np.asarray(proj[0])), np.ones_like(np.asarray(proj[1])),
             proj[2], proj[3]))
        dirty = (proj[0] + 1e3 * (1.0 - ind[0]), proj[1] + 1e3 * (1.0 - ind[1]),
                 proj[2], proj[3])
        clean_tied = sharing.ufv_share_tucker_variations(proj, sfd_u)
        dirty_tied = sharing.ufv_share_tucker_variations(dirty, sfd_u)
        self.assertTrue(np.allclose(np.asarray(clean_tied[0]), np.asarray(dirty_tied[0])))
        y_clean = utvo.utv_retract(frame_u, clean_tied, shared_data=sfd_u)
        y_dirty = utvo.utv_retract(frame_u, dirty_tied, shared_data=sfd_u)
        self.assertTrue(np.allclose(np.asarray(y_clean[0]), np.asarray(y_dirty[0])))
        self.assertTrue(np.allclose(np.asarray(y_clean[1]), np.asarray(y_dirty[1])))


class TestUniformSharedGeometry(unittest.TestCase):
    """Slice 11: the shared uniform geometry surface -- SharedGeometry over the uniform singletons,
    the geometry factories, and the fitting gates. Deterministic runs must MATCH the ragged shared
    runs (the equivalence contract on whole optimizer trajectories); recovery must stay tied."""

    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test

    @staticmethod
    def _problem(shape, spec):
        A = _tied_t3(shape, (3, 3, 3), (1, 3, 3, 1), spec)
        ww = [np.random.randn(150, N) for N in shape]
        ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
        return A, ww, A.apply(ww)

    def test_trajectories_match_ragged_and_recovery_stays_tied(self):
        import t3toolbox.uniform_manifold as um
        shape, spec = (6, 6, 5), (0, 0, 1)
        A, ww, b = self._problem(shape, spec)
        Ad = np.asarray(A.to_dense())
        # deterministic gradient descent from a tied NONZERO, SHARED-MINIMAL start (an exact-zero
        # start has an arbitrary orthogonal completion in its frame, and a non-minimal start is
        # transparently reduced on the uniform path only -- both legitimately diverge the layers)
        x00 = _tied_t3(shape, (3, 3, 2), (1, 3, 2, 1), spec)
        X0 = t3.TuckerTensorTrain(x00.tucker_cores, tuple(0.5 * np.asarray(G) for G in x00.tt_cores))
        uX0 = ut3.UniformTuckerTensorTrain.from_t3(X0)
        xr, s_r = optimizers.gradient_descent(sg.shared_manifold(spec), 'apply', ww, b, X0, n_iter=8)
        xu, s_u = optimizers.gradient_descent(sg.shared(um.UNIFORM_MANIFOLD, spec), 'apply', ww, b,
                                              uX0, n_iter=8)
        self.assertIsInstance(xu, ut3.UniformTuckerTensorTrain)
        self.assertTrue(np.allclose(s_r['losses'], s_u['losses']))
        self.assertTrue(np.allclose(np.asarray(xu.to_dense()), np.asarray(xr.to_dense()), atol=1e-8))
        self.assertTrue(np.all(np.asarray(xu.has_shared_tucker_factors(spec))))
        # corewise: the additive geometry, tied start
        x0c = t3.TuckerTensorTrain(x00.tucker_cores, tuple(0.3 * np.asarray(G) for G in x00.tt_cores))
        ux0c = ut3.UniformTuckerTensorTrain.from_t3(x0c)
        xr2, _ = optimizers.gradient_descent(sg.shared_corewise(spec), 'apply', ww, b, x0c, n_iter=8)
        xu2, _ = optimizers.gradient_descent(sg.shared(um.UNIFORM_COREWISE, spec), 'apply', ww, b,
                                             ux0c, n_iter=8)
        self.assertTrue(np.allclose(np.asarray(xu2.to_dense()), np.asarray(xr2.to_dense()), atol=1e-8))
        self.assertTrue(np.all(np.asarray(xu2.has_shared_tucker_factors(spec))))
        # newton_cg at the true shared ranks: recovers, every output tied
        uX03 = ut3.UniformTuckerTensorTrain.from_t3(
            t3.TuckerTensorTrain.zeros(shape, (3, 3, 3), (1, 3, 3, 1)))
        xu3, _ = optimizers.newton_cg(sg.shared(um.UNIFORM_MANIFOLD, spec), 'apply', ww, b, uX03,
                                      max_newton=25)
        rel = float(np.linalg.norm(np.asarray(xu3.to_dense()) - Ad) / np.linalg.norm(Ad))
        self.assertLess(rel, 1e-8)
        self.assertTrue(np.all(np.asarray(xu3.has_shared_tucker_factors(spec))))

    def test_model_invariants_match_ragged(self):
        # the frontend fitting-model path: shared uniform GaussNewton model == shared ragged model on
        # every gauge-invariant quantity; the companion rides as the model's geometry_aux leaf
        import t3toolbox.uniform_manifold as um
        from t3toolbox.backend.regularization import IdentityRegularizer
        shape, spec = (6, 6, 5), (0, 0, 1)
        x = _tied_t3(shape, (3, 3, 2), (1, 3, 2, 1), spec)
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        ww = [np.random.randn(15, N) for N in shape]
        r = np.asarray(x.apply(ww)) - np.random.randn(15)
        mr = fitting.apply_model(sg.shared_manifold(spec), x, ww, r)
        mu = fitting.apply_model(sg.shared(um.UNIFORM_MANIFOLD, spec), ux, ww, r)
        self.assertIsInstance(mu.geometry_aux, sharing.T3SharedFrameData)
        self.assertTrue(np.isclose(float(mr.objective_value), float(mu.objective_value)))
        self.assertTrue(np.isclose(float(mr.gradient.corewise_inner(mr.gradient)),
                                   float(mu.gradient.corewise_inner(mu.gradient))))
        self.assertTrue(np.isclose(float(mr.gn_quadratic(mr.gradient)),
                                   float(mu.gn_quadratic(mu.gradient))))
        self.assertTrue(np.isclose(float(mr.gradient.corewise_inner(mr.gn_hessian(mr.gradient))),
                                   float(mu.gradient.corewise_inner(mu.gn_hessian(mu.gradient)))))
        mu_reg = fitting.apply_model(sg.shared(um.UNIFORM_MANIFOLD, spec), ux, ww, r,
                                     regularizer=IdentityRegularizer(0.1))
        mr_reg = fitting.apply_model(sg.shared_manifold(spec), x, ww, r,
                                     regularizer=IdentityRegularizer(0.1))
        self.assertTrue(np.isclose(float(mu_reg.objective_value), float(mr_reg.objective_value)))

    def test_regularizer_uses_the_precomputed_companion(self):
        # the precompute audit fix (2026-08-20): the Regularizer protocol threads aux, so a
        # regularized shared matvec must NOT rebuild the SF-T3 companion per call -- previously the
        # one per-matvec leak (regularizer.hessian/quadratic called geom.project 2-arg)
        from unittest import mock
        from t3toolbox.backend.regularization import IdentityRegularizer
        spec = (0, 0, 1)
        x = _tied_t3((6, 6, 5), (3, 3, 2), (1, 3, 2, 1), spec)
        ww = [np.random.randn(15, N) for N in x.shape]
        r = np.asarray(x.apply(ww)) - np.random.randn(15)
        geom = sg.shared_manifold(spec)
        m = fitting.apply_model(geom, x, ww, r, regularizer=IdentityRegularizer(0.1))
        self.assertIsNotNone(m.geometry_aux)
        p = geom.randn(m.frame)
        _ = m.gradient                                     # realize the cached property first
        with mock.patch.object(sharing, 'fv_shared_frame_data',
                               wraps=sharing.fv_shared_frame_data) as spy:
            for _ in range(3):
                m.gn_hessian(p)
                m.gn_quadratic(p)
            self.assertEqual(spy.call_count, 0,
                             'a regularized shared matvec rebuilt the companion (aux not threaded)')

    def test_layer_mismatch_gates(self):
        # a SharedGeometry's base layer must match the point's layer, both directions
        import t3toolbox.uniform_manifold as um
        shape, spec = (6, 6, 5), (0, 0, 1)
        x = _tied_t3(shape, (2, 2, 2), (1, 2, 2, 1), spec)
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        ww = [np.random.randn(10, N) for N in shape]
        b = x.apply(ww)
        with self.assertRaises(ValueError):
            optimizers.gradient_descent(sg.shared(um.UNIFORM_MANIFOLD, spec), 'apply', ww, b, x,
                                        n_iter=1)
        with self.assertRaises(ValueError):
            optimizers.gradient_descent(sg.shared_manifold(spec), 'apply', ww, b, ux, n_iter=1)
        with self.assertRaises(ValueError):
            fitting.apply_model(sg.shared(um.UNIFORM_MANIFOLD, spec), x, ww, np.zeros(10))
        with self.assertRaises(ValueError):
            fitting.apply_model(sg.shared_manifold(spec), ux, ww, np.zeros(10))


class TestSharedMinimalRanksGroundTruth(unittest.TestCase):
    """Shared minimal ranks == generic dense edge-cut ranks of a TIED T3 (non-circular ground truth;
    mirrors the unshared test_compute_minimal_ranks_matches_matricization). The hand-worked expected
    tuples for the same structures live in tests/backend/test_ranks.py."""

    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test

    def test_matches_dense_edge_cut_ranks_of_tied_t3(self):
        # group Tucker rank = rank of the concatenated matricization [X_(i1)|...|X_(ik)];
        # TT ranks = the sequential-unfolding ranks. Structures cover: the group ceiling keeping a
        # rank the per-mode reduction clips, the sum-bound ceiling, non-adjacent groups, an all-modes
        # group, TT reductions interacting with the group, and the d=2 matrix case.
        cases = [
            ((6, 6, 4),    (4, 4, 3),    (1, 2, 2, 1),    (0, 0, 1)),
            ((9, 9, 4),    (7, 7, 2),    (1, 2, 2, 1),    (0, 0, 1)),
            ((5, 5, 4),    (5, 5, 4),    (1, 3, 3, 1),    (0, 0, 1)),
            ((7, 7, 7),    (7, 7, 7),    (1, 2, 2, 1),    (0, 0, 0)),
            ((5, 4, 5),    (2, 3, 2),    (1, 6, 6, 1),    (0, 1, 0)),
            ((3, 5, 3, 5), (3, 4, 3, 4), (1, 2, 3, 2, 1), ('a', 'b', 'a', 'b')),
            ((5, 5, 6, 6), (5, 5, 6, 6), (1, 2, 2, 2, 1), (0, 0, 1, 1)),
            ((6, 6),       (6, 6),       (1, 3, 1),       (0, 0)),
        ]
        for shape, tucker, tt, spec in cases:
            with self.subTest(shape=shape, tucker=tucker, tt=tt, sharing=spec):
                groups = sharing.validate_sharing(spec, shape)
                x_data, _, _ = _tied_data((shape, tucker, tt, spec), ())
                T = np.asarray(t3.TuckerTensorTrain(*x_data).to_dense())
                d = len(shape)

                dense_n = [0] * d
                for group in groups:
                    M = np.concatenate([np.moveaxis(T, jj, 0).reshape(shape[jj], -1)
                                        for jj in group], axis=1)
                    rank = int(np.linalg.matrix_rank(M, tol=1e-9 * np.linalg.norm(M, ord=2)))
                    for jj in group:
                        dense_n[jj] = rank
                dense_r = [1] * (d + 1)
                for ii in range(1, d):
                    M = T.reshape(int(np.prod(shape[:ii])), -1)
                    dense_r[ii] = int(np.linalg.matrix_rank(M, tol=1e-9 * np.linalg.norm(M, ord=2)))

                got = t3.TuckerTensorTrain.get_minimal_ranks(shape, tucker, tt, sharing=spec)
                self.assertEqual(got, (tuple(dense_n), tuple(dense_r)))


class TestSharedManifoldDimGroundTruth(unittest.TestCase):
    """manifold_dim(s, sharing=) == the rank of a dense basis of random tied tangents (the empirical
    backstop for the arbitrary-partition dimension formula -- the papers prove a single trailing
    block only). Hand-worked values for these structures live in tests/backend/test_ranks.py."""

    def setUp(self):
        np.random.seed(0)

    def test_dense_tied_tangent_rank_matches_formula(self):
        # all structures shared-minimal; the first has n_g = 4 > rL*rR = 2 at mode 0 (the group
        # ceiling case, where the UNSHARED reduction would clip the rank and the unshared formula
        # miscounts -- asserted disjoint below)
        cases = [
            ((6, 6, 4),    (4, 4, 2), (1, 2, 2, 1),    (0, 0, 1)),
            ((5, 5, 5),    (3, 3, 3), (1, 3, 3, 1),    (0, 0, 0)),
            ((4, 5, 4),    (3, 3, 3), (1, 3, 3, 1),    (0, 1, 0)),
            ((4, 4, 3, 3), (2, 2, 3, 3), (1, 2, 3, 2, 1), (0, 0, 1, 1)),
        ]
        for shape, tucker, tt, spec in cases:
            with self.subTest(shape=shape, tucker=tucker, tt=tt, sharing=spec):
                formula = t3m.manifold_dim((shape, tucker, tt), sharing=spec)
                x_data, _, _ = _tied_data((shape, tucker, tt, spec), ())
                geom = sg.shared_manifold(spec)
                frame = geom.frame(t3.TuckerTensorTrain(*x_data))
                self.assertEqual((tuple(frame.up_ranks), tuple(frame.left_ranks)), (tucker, tt))
                dense_vv = np.stack([np.asarray(geom.randn(frame).to_dense()).reshape(-1)
                                     for _ in range(formula + 25)])
                ss = np.linalg.svd(dense_vv, compute_uv=False)
                dense_dim = int(np.sum(ss > 1e-9 * ss[0]))
                self.assertEqual(dense_dim, formula)
                self.assertLess(formula, t3m.manifold_dim((shape, tucker, tt)))   # tying removes params


def _tied_t3(shape, tucker_ranks, tt_ranks, spec):
    """A random ragged T3 with the Tucker factors tied (same array) within each sharing group."""
    x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks)
    tk = list(x.tucker_cores)
    for group in sharing.validate_sharing(spec, shape):
        for ii in group[1:]:
            tk[ii] = tk[group[0]]
    return t3.TuckerTensorTrain(tuple(tk), x.tt_cores)


def _corrupt_ut3(u, scale=1e3):
    """Add ``scale`` * garbage to the masked-out (padding) region; the real region is unchanged
    (the testing_strategy garbage-robustness probe -- the tests/test_uniform_manifold.py pattern)."""
    ind = ut3.UniformTuckerTensorTrain(np.ones_like(np.asarray(u.tucker_supercore)),
                                       np.ones_like(np.asarray(u.tt_supercore)),
                                       u.shape, u.masks).apply_masks().supercores
    return ut3.UniformTuckerTensorTrain(u.tucker_supercore + scale * (1.0 - np.asarray(ind[0])),
                                        u.tt_supercore + scale * (1.0 - np.asarray(ind[1])),
                                        u.shape, u.masks)


class TestWeightsSharing(unittest.TestCase):
    """The weights x sharing resolution (2026-08-20): the combination already composes within the
    existing framework -- the ONLY addition is the non-enforcing compatibility checker (group-equal
    Tucker weights <=> absorb preserves tying; TT weights never touch the factors). Nothing gates."""

    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test

    @staticmethod
    def _tied(shape=(6, 6, 5), n=(3, 3, 2), r=(1, 3, 2, 1), spec=(0, 0, 1)):
        return _tied_t3(shape, n, r, spec)

    def test_equal_weights_preserve_tying_unequal_untie_without_error(self):
        spec = (0, 0, 1)
        xs = self._tied()
        w_eq = t3.T3Weights((np.array([1., 2., 3.]), np.array([1., 2., 3.]), np.array([1., 2.])),
                            (np.ones(1), np.array([1., 2., 3.]), np.array([2., 1.]), np.ones(1)))
        self.assertTrue(bool(w_eq.has_shared_tucker_weights(spec)))
        xa = t3.t3_absorb_weights(xs, w_eq)
        self.assertTrue(np.all(np.asarray(xa.has_shared_tucker_factors(spec))))
        # ... and the absorbed tied network flows through the grouped machinery
        ya, _, _ = xa.t3svd(sharing=spec)
        self.assertTrue(np.all(np.asarray(ya.has_shared_tucker_factors(spec))))
        # unequal group weights: checker False, absorb legal, result untied (the legitimate escape)
        w_neq = t3.T3Weights((np.array([1., 2., 3.]), np.array([3., 1., 2.]), np.array([1., 2.])),
                             w_eq.data[1])
        self.assertFalse(bool(w_neq.has_shared_tucker_weights(spec)))
        xb = t3.t3_absorb_weights(xs, w_neq)                             # no error
        self.assertFalse(np.all(np.asarray(xb.has_shared_tucker_factors(spec))))
        # TT-bond weights are unconstrained: any values keep the checker verdict
        w_tt = t3.T3Weights(w_eq.data[0],
                            (np.ones(1), np.array([9., 1., 2.]), np.array([5., 7.]), np.ones(1)))
        self.assertTrue(bool(w_tt.has_shared_tucker_weights(spec)))
        self.assertTrue(np.all(np.asarray(
            t3.t3_absorb_weights(xs, w_tt).has_shared_tucker_factors(spec))))

    def test_grouped_from_t3svd_is_compatible_and_algebra_is_closed(self):
        spec = (0, 0, 1)
        xs = self._tied()
        W = t3.T3Weights.from_t3svd(xs, sharing=spec)          # group spectra: group-equal by construction
        self.assertTrue(bool(W.has_shared_tucker_weights(spec)))
        self.assertTrue(np.array_equal(np.asarray(W.data[0][0]), np.asarray(W.data[0][1])))
        # per-mode svals of the same tied tensor are NOT group-equal (checker catches it)
        self.assertFalse(bool(t3.T3Weights.from_t3svd(xs).has_shared_tucker_weights(spec)))
        # the weight algebra preserves group-equality
        for out in (W.reciprocal(), W.sqrt(), W.concatenate(W), W.kronecker(W)):
            self.assertTrue(bool(out.has_shared_tucker_weights(spec)))

    def test_structural_and_stacked(self):
        spec = (0, 0, 1)
        # isolated structural rejection: unequal weight LENGTHS within a group (unequal ranks)
        w_bad = ((np.ones(2), np.ones(3), np.ones(2)),
                 (np.ones(1), np.ones(2), np.ones(2), np.ones(1)))
        with self.assertRaises(ValueError):
            sharing.t3_weights_sharing_residual(w_bad, spec)
        # per-stack-element verdicts
        tw = np.ones((2, 3))
        tw2 = tw.copy()
        tw2[1] += 1e-3                                          # perturb stack element 1 only
        W = t3.T3Weights((tw, tw2, np.ones((2, 2))),
                         (np.ones((2, 1)), np.ones((2, 3)), np.ones((2, 2)), np.ones((2, 1))))
        verdicts = np.asarray(W.has_shared_tucker_weights(spec))
        self.assertTrue(bool(verdicts[0]) and not bool(verdicts[1]))
        # all-singleton partitions are trivially compatible
        self.assertTrue(bool(W.has_shared_tucker_weights((0, 1, 2)).all()))

    def test_uniform_twin_masked_and_garbage_robust(self):
        spec = (0, 0, 1)
        xs = self._tied()
        uxs = ut3.UniformTuckerTensorTrain.from_t3(xs)
        W = ut3.UT3Weights.from_ut3svd(uxs, sharing=spec)
        self.assertTrue(bool(np.all(np.asarray(W.has_shared_tucker_weights(spec)))))
        self.assertFalse(bool(np.all(np.asarray(
            ut3.UT3Weights.from_ut3svd(uxs).has_shared_tucker_weights(spec)))))
        # garbage in the padded weight slots must not change the verdict
        tkm, _ = W.masks.data
        dirty = ut3.UT3Weights(np.asarray(W.tucker_weight_supercore) + 7.7 * (1.0 - tkm),
                               W.tt_weight_supercore, W.masks)
        self.assertTrue(bool(np.all(np.asarray(dirty.has_shared_tucker_weights(spec)))))


class TestUniformShared(unittest.TestCase):
    """Slice 9: the uniform mirror of the grouped truncation family, under the uniform equivalence
    contract (``to_uniform -> op -> to_ragged == op_ragged`` on real parts), with exact output-mask
    assertions (against the RAGGED output ranks -- non-circular) and garbage robustness, per
    docs/contributor/testing_strategy.md."""

    def setUp(self):
        np.random.seed(0)   # TuckerTensorTrain.randn draws from the GLOBAL rng -> seed per test

    def _check_matches_ragged(self, x, spec, u, cap_n=None, cap_r=None):
        """The contract for one input: dense + exact masks (== ragged ranks) + spectra + tied output."""
        yr_data, sk_r, st_r = bt3svd.t3svd(x.data, max_tucker_ranks=cap_n, max_tt_ranks=cap_r,
                                           sharing=spec)
        yr = t3.TuckerTensorTrain(*yr_data)
        yu, sk_u, st_u = u.t3svd(max_tucker_ranks=cap_n, max_tt_ranks=cap_r, sharing=spec)
        self.assertTrue(np.allclose(np.asarray(yu.to_dense()), np.asarray(yr.to_dense())))
        # exact output masks, derived from the ragged output's core shapes (non-circular)
        self.assertEqual(tuple(int(v) for v in yu.tucker_ranks), yr.tucker_ranks)
        self.assertEqual(tuple(int(v) for v in yu.tt_ranks), yr.tt_ranks)
        # ONE group rank mask at every group mode; output exactly tied on real content
        tkm = np.asarray(yu.masks.data[0])
        for group in sharing.nontrivial_groups(sharing.validate_sharing(spec, x.shape)):
            for ii in group[1:]:
                self.assertTrue(np.array_equal(tkm[group[0]], tkm[ii]))
        self.assertTrue(np.all(np.asarray(yu.has_shared_tucker_factors(spec))))
        # reported spectra match the ragged ones on the masked prefix
        for m in range(len(x.shape)):
            ref = np.asarray(sk_r[m])
            self.assertLess(float(np.linalg.norm(np.asarray(sk_u[m])[:ref.size] - ref)),
                            1e-9 * max(float(ref[0]), 1e-300))
        for m in range(len(x.shape) + 1):
            ref = np.asarray(st_r[m])
            self.assertLess(float(np.linalg.norm(np.asarray(st_u[m])[:ref.size] - ref)),
                            1e-9 * max(float(ref[0]), 1e-300))

    def test_grouped_ut3svd_matches_ragged(self):
        # lossless + capped, adjacent + non-adjacent + all-modes groups, natural + forced padding
        cases = [
            ((6, 6, 5),    (3, 3, 2),    (1, 3, 3, 1),    (0, 0, 1),          None,        None,   {}),
            ((6, 6, 5),    (3, 3, 2),    (1, 3, 3, 1),    (0, 0, 1),          (2, 2, 2),   2,      {}),
            ((5, 6, 5, 6), (2, 3, 2, 3), (1, 2, 3, 2, 1), ('a', 'b', 'a', 'b'), (2, 2, 2, 2), None, {}),
            ((7, 7, 7),    (4, 4, 4),    (1, 3, 3, 1),    (0, 0, 0),          3,           2,      {}),
            ((6, 6, 5),    (3, 3, 2),    (1, 3, 3, 1),    (0, 0, 1),          None,        None,
             dict(N=8, n=5, r=5)),                                     # forced-larger padding
        ]
        for shape, tucker, tt, spec, cap_n, cap_r, pad in cases:
            with self.subTest(shape=shape, sharing=spec, cap_n=cap_n, cap_r=cap_r, pad=pad):
                x = _tied_t3(shape, tucker, tt, spec)
                u = ut3.UniformTuckerTensorTrain.from_t3(x, **pad)
                self._check_matches_ragged(x, spec, u, cap_n=cap_n, cap_r=cap_r)

    def test_grouped_ut3svd_varying_ranks_across_stack(self):
        # a varying-rank C stack (two tied elements with different group ranks) + per-element caps
        spec = (0, 0, 1)
        x1 = _tied_t3((6, 6, 5), (3, 3, 2), (1, 3, 3, 1), spec)
        x2 = _tied_t3((6, 6, 5), (2, 2, 2), (1, 2, 2, 1), spec)
        u = ut3.UniformTuckerTensorTrain.stack([ut3.UniformTuckerTensorTrain.from_t3(x, n=3, r=3)
                                                for x in (x1, x2)])
        cap_n = np.array([[2, 2], [2, 2], [2, 2]])          # (d,) + stack: cap both elements to 2
        cap_r = np.array([[1, 1], [3, 2], [3, 2], [1, 1]])  # per-element bond caps
        yu, _, _ = u.t3svd(max_tucker_ranks=cap_n, max_tt_ranks=cap_r, sharing=spec)
        for kk, x in enumerate((x1, x2)):
            with self.subTest(element=kk):
                yr_data, _, _ = bt3svd.t3svd(x.data,
                                             max_tucker_ranks=tuple(int(v) for v in cap_n[:, kk]),
                                             max_tt_ranks=tuple(int(v) for v in cap_r[:, kk]),
                                             sharing=spec)
                yr = t3.TuckerTensorTrain(*yr_data)
                yu_k = yu.unstack()[kk].to_t3()
                self.assertTrue(np.allclose(np.asarray(yu_k.to_dense()), np.asarray(yr.to_dense())))
                self.assertEqual(yu_k.tucker_ranks, yr.tucker_ranks)
                self.assertEqual(yu_k.tt_ranks, yr.tt_ranks)
        self.assertTrue(np.all(np.asarray(yu.has_shared_tucker_factors(spec))))

    def test_grouped_ut3svd_garbage_robust(self):
        # mask-once: garbage in the padding must not change the result (bitwise on masked content)
        spec = (0, 0, 1)
        x = _tied_t3((6, 6, 5), (3, 3, 2), (1, 3, 3, 1), spec)
        u = ut3.UniformTuckerTensorTrain.from_t3(x, n=5, r=5)   # real padding to corrupt
        y_clean, sk_c, st_c = u.t3svd(max_tucker_ranks=2, sharing=spec)
        y_dirty, sk_d, st_d = _corrupt_ut3(u).t3svd(max_tucker_ranks=2, sharing=spec)
        for a, b in zip(y_clean.apply_masks().supercores, y_dirty.apply_masks().supercores):
            self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
        for a, b in zip(y_clean.masks.data, y_dirty.masks.data):
            self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(np.asarray(sk_c), np.asarray(sk_d)))
        self.assertTrue(np.array_equal(np.asarray(st_c), np.asarray(st_d)))

    def test_dispatch_anchor_none_and_singletons(self):
        # sharing=None and all-singleton partitions run the literal unshared sweep, bit-identically
        x = t3.TuckerTensorTrain.randn((6, 5, 4), (3, 3, 2), (1, 3, 2, 1))
        u = ut3.UniformTuckerTensorTrain.from_t3(x)
        ref, sk0, st0 = but3svd.ut3svd(u.data, max_tucker_ranks=2)
        for spec in (None, (0, 1, 2)):
            with self.subTest(sharing=spec):
                got, sk, st = but3svd.ut3svd(u.data, max_tucker_ranks=2, sharing=spec)
                for a, b in zip(ref[:2], got[:2]):
                    self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
                self.assertTrue(np.array_equal(np.asarray(sk0), np.asarray(sk)))
                self.assertTrue(np.array_equal(np.asarray(st0), np.asarray(st)))

    def test_shared_adjustment_and_uniform_minimal(self):
        # the untie hazard (group-ceiling structure): the per-mode reduction clips the group rank
        # (4,4,2)->(2,4,2), structurally untying it; the shared path keeps the rank, the ties, and
        # the tensor -- and matches the ragged shared adjustment exactly
        spec = (0, 0, 1)
        x = _tied_t3((6, 6, 4), (4, 4, 2), (1, 2, 2, 1), spec)     # shared-minimal, NOT unshared-minimal
        u = ut3.UniformTuckerTensorTrain.from_t3(x)
        um_plain = uf.uniform_minimal(u)
        self.assertEqual(tuple(int(v) for v in um_plain.tucker_ranks), (2, 4, 2))   # untied ranks
        um_shared = uf.uniform_minimal(u, sharing=spec)
        self.assertIs(um_shared, u)                                 # already shared-minimal: no-op
        # a padded (non-minimal) shared start reduces to the shared minimal ranks, tied, same tensor
        x_pad = x.resize(x.shape, (4, 4, 2), (1, 5, 5, 1), sharing=spec)
        u_pad = ut3.UniformTuckerTensorTrain.from_t3(x_pad)
        um2 = uf.uniform_minimal(u_pad, sharing=spec)
        self.assertEqual(tuple(int(v) for v in um2.tucker_ranks), (4, 4, 2))
        self.assertTrue(np.all(np.asarray(um2.has_shared_tucker_factors(spec))))
        self.assertTrue(np.allclose(np.asarray(um2.to_dense()), np.asarray(u_pad.to_dense())))
        # adjustment sweep == the ragged shared adjustment (dense + ranks), both directions
        y_l, _, _ = u_pad.t3svd(sharing=spec)
        for direction in ('right_to_left', 'left_to_right'):
            with self.subTest(direction=direction):
                src_u = y_l if direction == 'right_to_left' else um2
                src_r = src_u.to_t3()
                yr = src_r.rank_adjustment_sweep(direction, sharing=spec)
                yu = src_u.rank_adjustment_sweep(direction, sharing=spec)
                self.assertTrue(np.allclose(np.asarray(yu.to_dense()), np.asarray(yr.to_dense())))
                self.assertEqual(tuple(int(v) for v in yu.tucker_ranks), yr.tucker_ranks)
                self.assertEqual(tuple(int(v) for v in yu.tt_ranks), yr.tt_ranks)
                self.assertTrue(np.all(np.asarray(yu.has_shared_tucker_factors(spec))))

    def test_uniform_checker_per_element_and_safe_mode(self):
        spec = (0, 0, 1)
        x1 = _tied_t3((6, 6, 5), (3, 3, 2), (1, 3, 3, 1), spec)
        x2 = _tied_t3((6, 6, 5), (3, 3, 2), (1, 3, 3, 1), spec)
        u = ut3.UniformTuckerTensorTrain.stack([ut3.UniformTuckerTensorTrain.from_t3(x) for x in (x1, x2)])
        self.assertTrue(np.all(np.asarray(u.has_shared_tucker_factors(spec))))
        # perturb element 1's REAL factor content at one group mode -> per-element verdicts
        tk = np.asarray(u.tucker_supercore).copy()
        tk[1, 1, 0, 0] += 1e-3                                        # mode 1, element 1, real slot
        u2 = ut3.UniformTuckerTensorTrain(tk, u.tt_supercore, u.shape, u.masks)
        verdicts = np.asarray(u2.has_shared_tucker_factors(spec))
        self.assertTrue(bool(verdicts[0]) and not bool(verdicts[1]))
        # garbage-only perturbation leaves the verdicts True (padding is don't-care)
        u3 = _corrupt_ut3(u)
        self.assertTrue(np.all(np.asarray(u3.has_shared_tucker_factors(spec))))
        # safe mode rejects a grouped t3svd on untied factors; unsafe passes
        with self.assertRaises(ValueError):
            u2.t3svd(sharing=spec)
        with safety.unsafe():
            u2.t3svd(sharing=spec)


if __name__ == '__main__':
    unittest.main()
