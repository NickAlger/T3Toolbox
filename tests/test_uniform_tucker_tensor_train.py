# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3

np.random.seed(0)
norm = np.linalg.norm
TOL = 1e-10

# (shape, tucker_ranks, tt_ranks)
STRUCTURES = [
    ((5, 6, 7), (3, 4, 2), (1, 3, 2, 1)),
    ((4, 5, 6, 7), (2, 3, 4, 3), (1, 2, 4, 3, 1)),
    ((8, 9), (3, 4), (1, 5, 1)),
]
STACK_SHAPES = [(), (2,), (2, 3)]


def relerr(a, b):
    return norm(a - b) / max(1.0, norm(b))


class TestUniformTuckerTensorTrain(unittest.TestCase):
    def _cases(self):
        for shape, tr, ttr in STRUCTURES:
            for ss in STACK_SHAPES:
                yield shape, tr, ttr, ss

    # ---- the governing contract: uniform == ragged on the represented tensor ----
    def test_to_dense_matches_ragged(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.t3_to_ut3(x)
                self.assertLessEqual(relerr(ux.to_dense(), x.to_dense()), TOL)

    def test_roundtrip_t3_ut3_t3(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                back = ut3.ut3_to_t3(ut3.t3_to_ut3(x))
                if ss == ():
                    self.assertLessEqual(relerr(back.to_dense(), x.to_dense()), TOL)
                else:
                    stacked = t3.TuckerTensorTrain.stack(back)
                    self.assertLessEqual(relerr(stacked.to_dense(), x.to_dense()), TOL)

    # ---- structure ----
    def test_structure(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.t3_to_ut3(x)
                self.assertEqual(ux.shape, shape)
                self.assertEqual(ux.stack_shape, ss)
                self.assertEqual(ux.d, len(shape))
                self.assertEqual(ux.N, max(shape))
                self.assertEqual(ux.n, max(tr))
                self.assertEqual(ux.r, max(ttr))
                # ranks (uniform across the stack here) broadcast back to x's ranks
                self.assertEqual(tuple(int(v) for v in np.reshape(ux.tucker_ranks, (len(shape),) + ss)[(slice(None),) + (0,) * len(ss)]), tuple(tr))
                self.assertEqual(tuple(int(v) for v in np.reshape(ux.tt_ranks, (len(shape) + 1,) + ss)[(slice(None),) + (0,) * len(ss)]), tuple(ttr))

    # ---- masking / don't-care padding ----
    def test_apply_masks_preserves_tensor(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.t3_to_ut3(x)
                self.assertLessEqual(relerr(ux.apply_masks().to_dense(), x.to_dense()), TOL)

    def test_padding_is_dont_care(self):
        # garbage injected only into the padded ("garbage") region must not affect the dense tensor
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        ux = ut3.t3_to_ut3(x)
        tkm, ttm = ux.masks.data
        sm = np.arange(ux.N) < np.asarray(ux.shape)[:, None]   # (d, N) shape mask, reconstructed from ints
        tucker_real = np.einsum('d...n,dN->d...nN', tkm.astype(float), sm.astype(float))  # 1 on real, 0 on pad
        garbage = np.random.randn(*ux.tucker_supercore.shape) * (1.0 - tucker_real)
        ux_g = ut3.UniformTuckerTensorTrain(ux.tucker_supercore + garbage, ux.tt_supercore, ux.shape, ux.masks)
        self.assertLessEqual(relerr(ux_g.to_dense(), x.to_dense()), TOL)

    # ---- structural manipulations ----
    def test_reverse(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.t3_to_ut3(x)
                d = len(shape)
                mode_axes = tuple(range(len(ss), len(ss) + d))
                perm = tuple(range(len(ss))) + tuple(reversed(mode_axes))
                self.assertLessEqual(relerr(ux.reverse().to_dense(), np.transpose(x.to_dense(), perm)), TOL)

    def test_squash_tails_preserves_tensor(self):
        # use squash_tails=False on conversion so the boundary bonds are nontrivial, then squash
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (2, 3, 2, 2), stack_shape=(2,))
        ux = ut3.t3_to_ut3(x, squash_tails=False)
        squashed = ux.squash_tails()
        self.assertLessEqual(relerr(squashed.to_dense(), x.to_dense()), TOL)
        # boundary TT ranks are now 1
        self.assertTrue(np.all(squashed.tt_ranks[0] == 1))
        self.assertTrue(np.all(squashed.tt_ranks[-1] == 1))

    # ---- stacking ----
    def test_unstack_stack_roundtrip(self):
        for shape, tr, ttr, ss in self._cases():
            if ss == ():
                continue
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.t3_to_ut3(x)
                ux2 = ut3.UniformTuckerTensorTrain.stack(ux.unstack())
                self.assertLessEqual(relerr(ux2.to_dense(), x.to_dense()), TOL)

    def test_unstack_leaves_match_elements(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2, 3))
        ux = ut3.t3_to_ut3(x)
        tree = ux.unstack()
        xd = x.to_dense()
        for i in range(2):
            for j in range(3):
                self.assertLessEqual(relerr(tree[i][j].to_dense(), xd[i, j]), TOL)

    # ---- the variety: a stack whose elements have DIFFERENT ranks ----
    def test_varying_rank_stack(self):
        xa = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
        xb = t3.TuckerTensorTrain.randn((6, 7, 8), (4, 5, 3), (1, 4, 3, 1))
        N, n, r = 8, 5, 4
        ua = ut3.t3_to_ut3(xa, N=N, n=n, r=r)
        ub = ut3.t3_to_ut3(xb, N=N, n=n, r=r)
        ustacked = ut3.UniformTuckerTensorTrain.stack([ua, ub])
        self.assertEqual(ustacked.stack_shape, (2,))
        dense = ustacked.to_dense()
        self.assertLessEqual(relerr(dense[0], xa.to_dense()), TOL)
        self.assertLessEqual(relerr(dense[1], xb.to_dense()), TOL)
        # ranks genuinely differ across the stack (the variety)
        self.assertFalse(np.array_equal(ustacked.tucker_ranks[:, 0], ustacked.tucker_ranks[:, 1]))
        self.assertFalse(np.array_equal(ustacked.tt_ranks[:, 0], ustacked.tt_ranks[:, 1]))

    # ---- linear algebra (vs dense ground truth) ----
    def test_scale_neg(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.t3_to_ut3(x); xd = x.to_dense()
                self.assertLessEqual(relerr((ux * 2.5).to_dense(), 2.5 * xd), TOL)
                self.assertLessEqual(relerr((-ux).to_dense(), -xd), TOL)

    def test_add_sub_different_ranks(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                tr2 = tuple(v + 1 for v in tr)
                ttr2 = (1,) + tuple(v + 1 for v in ttr[1:-1]) + (1,)
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                y = t3.TuckerTensorTrain.randn(shape, tr2, ttr2, stack_shape=ss)
                ux, uy = ut3.t3_to_ut3(x), ut3.t3_to_ut3(y)
                xd, yd = x.to_dense(), y.to_dense()
                self.assertLessEqual(relerr((ux + uy).to_dense(), xd + yd), TOL)
                self.assertLessEqual(relerr((ux - uy).to_dense(), xd - yd), TOL)

    def test_inner(self):
        for shape, tr, ttr, ss in self._cases():
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
            y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
            ux, uy = ut3.t3_to_ut3(x), ut3.t3_to_ut3(y)
            xd, yd = x.to_dense(), y.to_dense()
            ax = tuple(range(len(ss), xd.ndim))
            expected = np.sum(xd * yd, axis=ax) if ss else np.sum(xd * yd)
            for uo in (True, False):
                with self.subTest(shape=shape, stack=ss, orth=uo):
                    self.assertLessEqual(relerr(ux.inner(uy, use_orthogonalization=uo), expected), TOL)

    def test_norm(self):
        for shape, tr, ttr, ss in self._cases():
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
            ux = ut3.t3_to_ut3(x); xd = x.to_dense()
            ax = tuple(range(len(ss), xd.ndim))
            expected = np.sqrt(np.sum(xd ** 2, axis=ax)) if ss else norm(xd)
            for uo in (True, False):
                with self.subTest(shape=shape, stack=ss, orth=uo):
                    self.assertLessEqual(relerr(ux.norm(use_orthogonalization=uo), expected), TOL)

    def test_sum_stack(self):
        for shape, tr, ttr in STRUCTURES:
            for ss in [(2,), (2, 3)]:
                with self.subTest(shape=shape, stack=ss):
                    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                    ux = ut3.t3_to_ut3(x)
                    expected = x.to_dense().sum(axis=tuple(range(len(ss))))
                    self.assertLessEqual(relerr(ux.sum_stack().to_dense(), expected), TOL)

    # ---- sampling / evaluation (vs keystone / dense) ----
    def test_entries(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.t3_to_ut3(x)
                idx = [min(2, N - 1) for N in shape]
                self.assertLessEqual(relerr(ux.entries(idx), x.entries(idx)), TOL)
                idxm = [np.array([0, N // 2, N - 1]) for N in shape]
                self.assertLessEqual(relerr(ux.entries(idxm), x.entries(idxm)), TOL)

    def test_apply_and_probe(self):
        # use the SAME vectors for uniform and ragged
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.t3_to_ut3(x)
                d = len(shape)
                for W in ((), (4,)):
                    vecs = [np.random.randn(*(W + (N,))) for N in shape]
                    self.assertLessEqual(relerr(ux.apply(vecs), x.apply(vecs)), TOL)
                    uzz, xzz = ux.probe(vecs), x.probe(vecs)
                    for i in range(d):
                        self.assertLessEqual(relerr(uzz[i], xzz[i]), TOL)

    def test_sum_full(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.t3_to_ut3(x); xd = x.to_dense()
                ax = tuple(range(len(ss), xd.ndim))
                expected = xd.sum(axis=ax) if ss else xd.sum()
                self.assertLessEqual(relerr(ux.sum(), expected), TOL)

    def test_sum_partial_raises(self):
        ux = ut3.t3_to_ut3(t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1)))
        with self.assertRaises(NotImplementedError):
            ux.sum(axis=0)

    # ---- orthogonalization (vs ragged; rank reduction is minimal-for-free) ----
    ORTH_METHODS = ['down_orthogonalize_tucker_cores', 'up_orthogonalize_tt_cores',
                    'left_orthogonalize_tt_cores', 'right_orthogonalize_tt_cores']

    def _assert_ranks_match(self, uxr, xr):
        self.assertEqual(tuple(int(v) for v in np.asarray(uxr.tucker_ranks).reshape(uxr.d)),
                         tuple(xr.tucker_ranks))
        self.assertEqual(tuple(int(v) for v in np.asarray(uxr.tt_ranks).reshape(uxr.d + 1)),
                         tuple(xr.tt_ranks))

    def test_orthogonalize_matches_ragged(self):
        for shape, tr, ttr, ss in self._cases():
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
            ux = ut3.t3_to_ut3(x)
            for m in self.ORTH_METHODS:
                with self.subTest(shape=shape, stack=ss, method=m):
                    xr = getattr(x, m)()
                    uxr = getattr(ux, m)()
                    self.assertLessEqual(relerr(uxr.to_dense(), x.to_dense()), TOL)   # preserves the tensor
                    self.assertLessEqual(relerr(uxr.to_dense(), xr.to_dense()), TOL)  # matches ragged
                    if ss == ():
                        self._assert_ranks_match(uxr, xr)

    def test_orthogonalize_non_minimal(self):
        # ranks reduce to the structural minimum the SVD produces -- must match ragged exactly
        cases = [
            ((5, 6), (8, 4), (1, 3, 1)),               # tucker rank 8 > shape 5
            ((6, 7, 8), (5, 5, 5), (1, 40, 40, 1)),    # inflated TT ranks
        ]
        for shape, tr, ttr in cases:
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
            ux = ut3.t3_to_ut3(x)
            for m in self.ORTH_METHODS:
                with self.subTest(shape=shape, method=m):
                    xr = getattr(x, m)()
                    uxr = getattr(ux, m)()
                    self.assertLessEqual(relerr(uxr.to_dense(), xr.to_dense()), TOL)
                    self._assert_ranks_match(uxr, xr)

    # ---- t3svd (vs t3svd / dense) ----
    def test_t3svd_no_truncation(self):
        # reduces to minimal ranks, same tensor; no-truncation ranks match ragged exactly
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.t3_to_ut3(x)
                ux2, _, _ = ux.t3svd()
                x2, _, _ = x.t3svd()
                self.assertLessEqual(relerr(ux2.to_dense(), x2.to_dense()), TOL)
                if ss == ():
                    self._assert_ranks_match(ux2, x2)

    def _first_elem_ranks(self, uxr, ss):
        d = uxr.d
        utk = np.asarray(uxr.tucker_ranks).reshape((d,) + ss)
        utt = np.asarray(uxr.tt_ranks).reshape((d + 1,) + ss)
        sel = (slice(None),) + (0,) * len(ss)
        return tuple(int(v) for v in utk[sel]), tuple(int(v) for v in utt[sel])

    def test_t3svd_truncation(self):
        # t3svd is the basic algorithm: left-orthogonal, raw-sweep ranks (NOT necessarily minimal). Tensor
        # AND ranks match ragged t3svd (both raw), including the asymmetric/divergent cap patterns.
        cap_patterns = [(2, 2), (None, 2), (3, 2), (2, None), (None, 3)]
        for shape, tr, ttr, ss in self._cases():
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
            ux = ut3.t3_to_ut3(x)
            for mtk, mtt in cap_patterns:
                with self.subTest(shape=shape, stack=ss, max_tucker=mtk, max_tt=mtt):
                    ux2, _, _ = ux.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt)
                    x2, _, _ = x.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt)  # ragged, same caps
                    self.assertLessEqual(relerr(ux2.to_dense(), x2.to_dense()), TOL)
                    self.assertTrue(ux2.is_left_orthogonal())   # always left-orthogonal
                    self.assertEqual(self._first_elem_ranks(ux2, ss), (x2.tucker_ranks, x2.tt_ranks))

    def test_rank_adjustment_sweep(self):
        # rank_adjustment_sweep minimizes a (left-orthogonal) t3svd output via 'right_to_left'; uniform
        # matches ragged in tensor, ranks, AND gauge. Composing both directions gives minimal left-orth.
        cap_patterns = [(None, 2), (3, 2), (2, 2), (2, None)]
        for shape, tr, ttr, ss in self._cases():
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
            ux = ut3.t3_to_ut3(x)
            for mtk, mtt in cap_patterns:
                with self.subTest(shape=shape, stack=ss, max_tucker=mtk, max_tt=mtt):
                    ux2, _, _ = ux.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt)  # left-orth, maybe non-min
                    x2, _, _ = x.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt)
                    url = ux2.rank_adjustment_sweep('right_to_left')   # left-orth input -> R->L minimizes
                    rrl = x2.rank_adjustment_sweep('right_to_left')
                    self.assertLessEqual(relerr(url.to_dense(), rrl.to_dense()), TOL)  # matches ragged
                    self.assertLessEqual(relerr(url.to_dense(), ux2.to_dense()), TOL)  # lossless
                    self.assertTrue(url.has_minimal_ranks)
                    self.assertTrue(url.is_right_orthogonal())
                    self.assertEqual(self._first_elem_ranks(url, ss), (rrl.tucker_ranks, rrl.tt_ranks))
                    both = url.rank_adjustment_sweep('left_to_right')  # -> minimal, left-orthogonal
                    self.assertTrue(both.has_minimal_ranks)
                    self.assertTrue(both.is_left_orthogonal())
                    self.assertLessEqual(relerr(both.to_dense(), ux2.to_dense()), TOL)
        with self.assertRaises(ValueError):
            ut3.t3_to_ut3(x).rank_adjustment_sweep('sideways')

    def test_is_left_right_orthogonal(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                ux = ut3.t3_to_ut3(t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss))
                self.assertFalse(ux.is_left_orthogonal())          # a random T3 is in neither form
                self.assertFalse(ux.is_right_orthogonal())
                uxL = ux.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores()
                uxR = ux.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores()
                self.assertTrue(uxL.is_left_orthogonal())
                self.assertFalse(uxL.is_right_orthogonal())
                self.assertTrue(uxR.is_right_orthogonal())
                self.assertFalse(uxR.is_left_orthogonal())
                self.assertTrue(ux.t3svd()[0].is_left_orthogonal())  # a t3svd result is left-orthogonal

    def test_t3svd_assume_orthogonal(self):
        # assume_orthogonal=True (input already right-orthogonal) skips the orthogonalization; same result.
        for shape, tr, ttr, ss in self._cases():
            ux = ut3.t3_to_ut3(t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss))
            uR = ux.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores()  # right-orthogonal
            self.assertTrue(uR.is_right_orthogonal())
            for mtk, mtt in [(None, 2), (3, 2), (2, 2)]:
                with self.subTest(shape=shape, stack=ss, max_tucker=mtk, max_tt=mtt):
                    a, _, _ = uR.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt, assume_orthogonal=True)
                    b, _, _ = uR.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt)
                    self.assertLessEqual(relerr(a.to_dense(), b.to_dense()), TOL)
                    self.assertTrue(a.is_left_orthogonal())

    def test_t3svd_non_minimal(self):
        for shape, tr, ttr in [((5, 6), (8, 4), (1, 3, 1)), ((6, 7, 8), (5, 5, 5), (1, 40, 40, 1))]:
            with self.subTest(shape=shape):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
                ux = ut3.t3_to_ut3(x)
                ux2, _, _ = ux.t3svd()
                x2, _, _ = x.t3svd()
                self.assertLessEqual(relerr(ux2.to_dense(), x2.to_dense()), TOL)
                self._assert_ranks_match(ux2, x2)

    def test_t3svd_per_stack_max_ranks(self):
        # the variety: different rank caps per stack element, in one call
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (4, 5, 4), (1, 4, 4, 1), stack_shape=(2,))
        ux = ut3.t3_to_ut3(x)
        max_tk = np.array([[2, 4], [2, 4], [2, 4]])  # (d=3, stack=2): elem0 cap 2, elem1 cap 4
        ux2, _, _ = ux.t3svd(max_tucker_ranks=max_tk)
        xx = x.unstack()
        e0, _, _ = xx[0].t3svd(max_tucker_ranks=2)
        e1, _, _ = xx[1].t3svd(max_tucker_ranks=4)
        d = ux2.to_dense()
        self.assertLessEqual(relerr(d[0], e0.to_dense()), TOL)
        self.assertLessEqual(relerr(d[1], e1.to_dense()), TOL)

    def test_t3svd_has_no_rtol_atol(self):
        # uniform t3svd truncates by max rank only -- rtol/atol are not parameters (data-dependent shapes)
        ux = ut3.t3_to_ut3(t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1)))
        with self.assertRaises(TypeError):
            ux.t3svd(rtol=1e-3)
        with self.assertRaises(TypeError):
            ux.t3svd(atol=1e-3)

    # ---- validate (structural hard errors) ----
    def test_validate_raises_on_bad_shape(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        ux = ut3.t3_to_ut3(x)
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore[..., :-1], ux.shape, ux.masks)

    def test_validate_raises_on_nonbool_mask(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        ux = ut3.t3_to_ut3(x)
        bad = ut3.UT3Masks(ux.masks.tucker_edge_mask.astype(float), ux.masks.tt_edge_mask)
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore, ux.shape, bad)

    def test_validate_raises_on_bad_shape_tuple(self):
        # shape must be a length-d tuple of mode dims within the padded N (the int-tuple invariant)
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        ux = ut3.t3_to_ut3(x)
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore, ux.shape[:-1], ux.masks)  # wrong length
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore,
                                         (ux.N + 1,) + ux.shape[1:], ux.masks)  # exceeds padded N

    # ---- constructors (zeros / ones / randn) ----
    def test_zeros(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                z = ut3.UniformTuckerTensorTrain.zeros(shape, tr, ttr, stack_shape=ss)
                self.assertEqual(z.shape, shape)
                self.assertEqual(z.stack_shape, ss)
                self.assertEqual(float(norm(z.to_dense())), 0.0)
                # ranks (uniform across the stack) match the request
                self.assertEqual(self._first_elem_ranks(z, ss), (tuple(tr), tuple(ttr)))

    def test_zeros_default_ranks(self):
        z = ut3.UniformTuckerTensorTrain.zeros((5, 6, 7))
        self.assertEqual(z.shape, (5, 6, 7))
        self.assertEqual(float(norm(z.to_dense())), 0.0)
        self.assertEqual(self._first_elem_ranks(z, ()), ((1, 1, 1), (1, 1, 1, 1)))

    def test_ones(self):
        for ss in STACK_SHAPES:
            with self.subTest(stack=ss):
                x = ut3.UniformTuckerTensorTrain.ones((5, 6, 7), stack_shape=ss)
                self.assertEqual(float(norm(x.to_dense() - np.ones(ss + (5, 6, 7)))), 0.0)
                self.assertEqual(self._first_elem_ranks(x, ss), ((1, 1, 1), (1, 1, 1, 1)))

    def test_randn_structure_and_matches_ragged_roundtrip(self):
        # randn cores are random N(0,1); the represented tensor equals the ragged round-trip's
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                ux = ut3.UniformTuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                self.assertEqual(ux.shape, shape)
                self.assertEqual(ux.stack_shape, ss)
                self.assertEqual(self._first_elem_ranks(ux, ss), (tuple(tr), tuple(ttr)))
                self.assertTrue(np.any(np.asarray(ux.tucker_supercore) != 0.0))
                # the padded ("garbage") regions are masked to zero -> ragged round-trip is faithful
                back = ut3.ut3_to_t3(ux)
                rebuilt = back if ss == () else t3.TuckerTensorTrain.stack(back)
                self.assertEqual(rebuilt.shape, shape)
                self.assertEqual(rebuilt.tucker_ranks, tuple(tr))
                self.assertEqual(rebuilt.tt_ranks, tuple(ttr))

    def test_randn_padding_is_masked(self):
        # garbage cannot hide in the padding: re-masking is a no-op (padding already zero)
        ux = ut3.UniformTuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        self.assertLessEqual(relerr(ux.apply_masks().to_dense(), ux.to_dense()), TOL)

    def test_randn_per_stack_element_ranks(self):
        # the variety: a full (d,)+stack rank array -> ranks vary per stack element, one padded shape
        tucker_ranks = np.array([[2, 4], [3, 5], [2, 3]])           # (d=3, stack=2)
        tt_ranks = np.array([[1, 1], [2, 4], [2, 3], [1, 1]])       # (d+1=4, stack=2)
        ux = ut3.UniformTuckerTensorTrain.randn((6, 7, 8), tucker_ranks, tt_ranks, stack_shape=(2,))
        self.assertEqual(ux.stack_shape, (2,))
        self.assertEqual(ux.n, 5)   # max padded Tucker rank
        self.assertEqual(ux.r, 4)   # max padded TT rank
        self.assertTrue(np.array_equal(np.asarray(ux.tucker_ranks), tucker_ranks))
        self.assertTrue(np.array_equal(np.asarray(ux.tt_ranks), tt_ranks))
        # ranks genuinely differ across the stack
        self.assertFalse(np.array_equal(ux.tucker_ranks[:, 0], ux.tucker_ranks[:, 1]))
        # each stack element's represented tensor has the right shape and is reconstructible
        tree = ux.unstack()
        for i in range(2):
            self.assertEqual(tree[i].shape, (6, 7, 8))

    def test_zeros_per_stack_element_ranks(self):
        tucker_ranks = np.array([[2, 4], [3, 5], [2, 3]])
        tt_ranks = np.array([[1, 1], [2, 4], [2, 3], [1, 1]])
        z = ut3.UniformTuckerTensorTrain.zeros((6, 7, 8), tucker_ranks, tt_ranks, stack_shape=(2,))
        self.assertEqual(float(norm(z.to_dense())), 0.0)
        self.assertTrue(np.array_equal(np.asarray(z.tucker_ranks), tucker_ranks))
        self.assertTrue(np.array_equal(np.asarray(z.tt_ranks), tt_ranks))

    def test_randn_scalar_rank_spec(self):
        # a scalar rank caps every mode the same
        ux = ut3.UniformTuckerTensorTrain.randn((5, 6, 7), 2, 2)
        self.assertEqual(self._first_elem_ranks(ux, ()), ((2, 2, 2), (2, 2, 2, 2)))

    def test_constructor_bad_rank_spec_raises(self):
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain.randn((5, 6, 7), (3, 4), (1, 3, 2, 1))   # wrong-length tucker

    # ---- conversions to/from other formats ----
    def test_from_canonical(self):
        rank, shape, ss = 3, (5, 6, 7), (2,)
        FF = [np.random.randn(*(ss + (rank, N))) for N in shape]
        ux = ut3.UniformTuckerTensorTrain.from_canonical(FF)
        expected = np.einsum('ari,arj,ark->aijk', FF[0], FF[1], FF[2])
        self.assertLessEqual(relerr(ux.to_dense(), expected), TOL)
        # Tucker ranks = canonical rank; boundary TT bonds squashed to 1 by t3_to_ut3's default squash
        self.assertEqual(self._first_elem_ranks(ux, ss), ((rank, rank, rank), (1, rank, rank, 1)))
        # matches the ragged from_canonical exactly
        xr = t3.TuckerTensorTrain.from_canonical(FF)
        self.assertLessEqual(relerr(ux.to_dense(), xr.to_dense()), TOL)
        self.assertTrue(all(m.dtype == bool for m in ux.masks.data))   # masks numpy bool

    def test_from_tensor_train(self):
        tt = [np.random.randn(1, 5, 4), np.random.randn(4, 6, 3), np.random.randn(3, 7, 1)]
        ux = ut3.UniformTuckerTensorTrain.from_tensor_train(tt)
        expected = np.einsum('aib,bjc,ckd->ijk', *tt)
        self.assertLessEqual(relerr(ux.to_dense(), expected), TOL)
        xr = t3.TuckerTensorTrain.from_tensor_train(tt)
        self.assertLessEqual(relerr(ux.to_dense(), xr.to_dense()), TOL)

    def test_to_tensor_train_unstacked(self):
        x = t3.TuckerTensorTrain.randn((6, 7, 8), (3, 4, 2), (1, 3, 2, 1))
        ux = ut3.t3_to_ut3(x)
        tt = ux.to_tensor_train()
        got = np.einsum('aib,bjc,ckd->ijk', *tt)
        self.assertLessEqual(relerr(got, x.to_dense()), TOL)
        # matches the ragged to_tensor_train
        ragged_tt = x.to_tensor_train()
        self.assertLessEqual(relerr(got, np.einsum('aib,bjc,ckd->ijk', *ragged_tt)), TOL)

    def test_to_tensor_train_stacked_tree(self):
        x = t3.TuckerTensorTrain.randn((6, 7, 8), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        ux = ut3.t3_to_ut3(x)
        tree = ux.to_tensor_train()
        self.assertEqual(len(tree), 2)
        xd = x.to_dense()
        for i in range(2):
            got = np.einsum('aib,bjc,ckd->ijk', *tree[i])
            self.assertLessEqual(relerr(got, xd[i]), TOL)

    # ---- save / load ----
    def test_save_load_roundtrip(self):
        import tempfile, os
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        ux = ut3.t3_to_ut3(x)
        fname = os.path.join(tempfile.mkdtemp(), 'ut3_test.npz')
        ux.save(fname)
        ux2 = ut3.UniformTuckerTensorTrain.load(fname)
        self.assertEqual(float(norm(ux2.to_dense() - ux.to_dense())), 0.0)
        # loaded masks are numpy (host) bool -- never jax (the slice-7 invariant)
        for m in ux2.masks.data:
            self.assertTrue(isinstance(m, np.ndarray))
            self.assertEqual(m.dtype, bool)
        # structure preserved
        self.assertTrue(np.array_equal(np.asarray(ux2.tucker_ranks), np.asarray(ux.tucker_ranks)))
        self.assertTrue(np.array_equal(np.asarray(ux2.tt_ranks), np.asarray(ux.tt_ranks)))

    def test_save_load_varying_rank_stack(self):
        import tempfile, os
        tucker_ranks = np.array([[2, 4], [3, 5], [2, 3]])
        tt_ranks = np.array([[1, 1], [2, 4], [2, 3], [1, 1]])
        ux = ut3.UniformTuckerTensorTrain.randn((6, 7, 8), tucker_ranks, tt_ranks, stack_shape=(2,))
        fname = os.path.join(tempfile.mkdtemp(), 'ut3_variety.npz')
        ux.save(fname)
        ux2 = ut3.UniformTuckerTensorTrain.load(fname)
        self.assertEqual(float(norm(ux2.to_dense() - ux.to_dense())), 0.0)
        # the per-element ranks survive the round-trip
        self.assertTrue(np.array_equal(np.asarray(ux2.tucker_ranks), tucker_ranks))
        self.assertTrue(np.array_equal(np.asarray(ux2.tt_ranks), tt_ranks))

    # ---- backend-only path: constructors/IO reproducible on raw .data ----
    def test_constructors_backend_only(self):
        # every constructor/IO frontend method is a thin wrapper over a ut3_constructors backend function
        import t3toolbox.backend.ut3_constructors as bc
        import t3toolbox.backend.ut3_conversions as conv
        z = bc.ut3_zeros((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        self.assertEqual(float(norm(conv.ut3_to_dense(z))), 0.0)
        self.assertEqual(z[2], (5, 6, 7))   # .data[2] is the static int-tuple shape
        # rank masks (.data[3]) are numpy bool even for a backend-built constructor
        self.assertTrue(all(isinstance(m, np.ndarray) and m.dtype == bool for m in z[3]))
        o = bc.ut3_ones((5, 6, 7))
        self.assertEqual(float(norm(conv.ut3_to_dense(o) - np.ones((5, 6, 7)))), 0.0)
        FF = [np.random.randn(3, N) for N in (5, 6, 7)]
        fc = bc.ut3_from_canonical(FF)
        self.assertLessEqual(relerr(conv.ut3_to_dense(fc),
                                    np.einsum('ri,rj,rk->ijk', *FF)), TOL)

    # ---- dtype / copy ----
    def test_to_numpy_and_copy(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        ux = ut3.t3_to_ut3(x)
        self.assertLessEqual(relerr(ux.copy().to_dense(), x.to_dense()), TOL)
        self.assertLessEqual(relerr(ux.to_numpy().to_dense(), x.to_dense()), TOL)

    # ---- backend-only path: every frontend op is reproducible on the raw .data tuple ----
    def test_backend_only_on_data(self):
        import t3toolbox.backend.ut3_conversions as conv
        import t3toolbox.backend.ut3_orthogonalization as bo
        import t3toolbox.backend.ut3_operations as bops
        import t3toolbox.backend.ut3_linalg as bl
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        y = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        ux, uy = ut3.t3_to_ut3(x), ut3.t3_to_ut3(y)

        # to_dense, orthogonalize, add+squash, inner -- all via backend functions on .data
        self.assertLessEqual(relerr(conv.ut3_to_dense(ux.data), ux.to_dense()), TOL)
        self.assertLessEqual(relerr(
            conv.ut3_to_dense(bo.down_orthogonalize_tucker_cores(ux.data)),
            ux.down_orthogonalize_tucker_cores().to_dense()), TOL)
        self.assertLessEqual(relerr(
            conv.ut3_to_dense(bops.ut3_squash_tails(bl.ut3_add(ux.data, uy.data))),
            (ux + uy).to_dense()), TOL)
        orth_data = lambda d: bo.left_orthogonalize_tt_cores(bo.down_orthogonalize_tucker_cores(d))
        self.assertLessEqual(relerr(
            bl.ut3_inner_product(orth_data(ux.data), orth_data(uy.data)),
            ux.inner(uy)), TOL)


if __name__ == '__main__':
    unittest.main()
