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
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                self.assertLessEqual(relerr(ux.to_dense(), x.to_dense()), TOL)

    def test_roundtrip_t3_ut3_t3(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                back = ut3.UniformTuckerTensorTrain.from_t3(x).to_t3()
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
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
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
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                self.assertLessEqual(relerr(ux.apply_masks().to_dense(), x.to_dense()), TOL)

    def test_padding_is_dont_care(self):
        # garbage injected only into the padded ("garbage") region must not affect the dense tensor
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
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
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                d = len(shape)
                mode_axes = tuple(range(len(ss), len(ss) + d))
                perm = tuple(range(len(ss))) + tuple(reversed(mode_axes))
                self.assertLessEqual(relerr(ux.reverse().to_dense(), np.transpose(x.to_dense(), perm)), TOL)

    def test_squash_tails_preserves_tensor(self):
        # use squash_tails=False on conversion so the boundary bonds are nontrivial, then squash
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (2, 3, 2, 2), stack_shape=(2,))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, squash_tails=False)
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
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                ux2 = ut3.UniformTuckerTensorTrain.stack(ux.unstack())
                self.assertLessEqual(relerr(ux2.to_dense(), x.to_dense()), TOL)

    def test_unstack_leaves_match_elements(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2, 3))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
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
        ua = ut3.UniformTuckerTensorTrain.from_t3(xa, N=N, n=n, r=r)
        ub = ut3.UniformTuckerTensorTrain.from_t3(xb, N=N, n=n, r=r)
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
                ux = ut3.UniformTuckerTensorTrain.from_t3(x); xd = x.to_dense()
                self.assertLessEqual(relerr((ux * 2.5).to_dense(), 2.5 * xd), TOL)
                self.assertLessEqual(relerr((-ux).to_dense(), -xd), TOL)

    def test_add_sub_different_ranks(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                tr2 = tuple(v + 1 for v in tr)
                ttr2 = (1,) + tuple(v + 1 for v in ttr[1:-1]) + (1,)
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                y = t3.TuckerTensorTrain.randn(shape, tr2, ttr2, stack_shape=ss)
                ux, uy = ut3.UniformTuckerTensorTrain.from_t3(x), ut3.UniformTuckerTensorTrain.from_t3(y)
                xd, yd = x.to_dense(), y.to_dense()
                self.assertLessEqual(relerr((ux + uy).to_dense(), xd + yd), TOL)
                self.assertLessEqual(relerr((ux - uy).to_dense(), xd - yd), TOL)

    def test_inner(self):
        for shape, tr, ttr, ss in self._cases():
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
            y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
            ux, uy = ut3.UniformTuckerTensorTrain.from_t3(x), ut3.UniformTuckerTensorTrain.from_t3(y)
            xd, yd = x.to_dense(), y.to_dense()
            ax = tuple(range(len(ss), xd.ndim))
            expected = np.sum(xd * yd, axis=ax) if ss else np.sum(xd * yd)
            for uo in (True, False):
                with self.subTest(shape=shape, stack=ss, orth=uo):
                    self.assertLessEqual(relerr(ux.inner(uy, use_orthogonalization=uo), expected), TOL)

    def test_norm(self):
        for shape, tr, ttr, ss in self._cases():
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
            ux = ut3.UniformTuckerTensorTrain.from_t3(x); xd = x.to_dense()
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
                    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                    expected = x.to_dense().sum(axis=tuple(range(len(ss))))
                    self.assertLessEqual(relerr(ux.sum_stack().to_dense(), expected), TOL)

    # ---- sampling / evaluation (vs keystone / dense) ----
    def test_entries(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                idx = [min(2, N - 1) for N in shape]
                self.assertLessEqual(relerr(ux.entries(idx), x.entries(idx)), TOL)
                idxm = [np.array([0, N // 2, N - 1]) for N in shape]
                self.assertLessEqual(relerr(ux.entries(idxm), x.entries(idxm)), TOL)

    def test_apply_and_probe(self):
        # use the SAME vectors for uniform and ragged
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
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
                ux = ut3.UniformTuckerTensorTrain.from_t3(x); xd = x.to_dense()
                ax = tuple(range(len(ss), xd.ndim))
                expected = xd.sum(axis=ax) if ss else xd.sum()
                self.assertLessEqual(relerr(ux.sum(), expected), TOL)

    def test_sum_partial_raises(self):
        ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1)))
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
            ux = ut3.UniformTuckerTensorTrain.from_t3(x)
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
            ux = ut3.UniformTuckerTensorTrain.from_t3(x)
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
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
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
            ux = ut3.UniformTuckerTensorTrain.from_t3(x)
            for mtk, mtt in cap_patterns:
                with self.subTest(shape=shape, stack=ss, max_tucker=mtk, max_tt=mtt):
                    ux2, _, _ = ux.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt)
                    x2, _, _ = x.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt)  # ragged, same caps
                    self.assertLessEqual(relerr(ux2.to_dense(), x2.to_dense()), TOL)
                    self.assertTrue(ux2.is_left_orthogonal().all())   # always left-orthogonal
                    self.assertEqual(self._first_elem_ranks(ux2, ss), (x2.tucker_ranks, x2.tt_ranks))

    def test_rank_adjustment_sweep(self):
        # rank_adjustment_sweep minimizes a (left-orthogonal) t3svd output via 'right_to_left'; uniform
        # matches ragged in tensor, ranks, AND gauge. Composing both directions gives minimal left-orth.
        cap_patterns = [(None, 2), (3, 2), (2, 2), (2, None)]
        for shape, tr, ttr, ss in self._cases():
            x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
            ux = ut3.UniformTuckerTensorTrain.from_t3(x)
            for mtk, mtt in cap_patterns:
                with self.subTest(shape=shape, stack=ss, max_tucker=mtk, max_tt=mtt):
                    ux2, _, _ = ux.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt)  # left-orth, maybe non-min
                    x2, _, _ = x.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt)
                    url = ux2.rank_adjustment_sweep('right_to_left')   # left-orth input -> R->L minimizes
                    rrl = x2.rank_adjustment_sweep('right_to_left')
                    self.assertLessEqual(relerr(url.to_dense(), rrl.to_dense()), TOL)  # matches ragged
                    self.assertLessEqual(relerr(url.to_dense(), ux2.to_dense()), TOL)  # lossless
                    self.assertTrue(url.has_minimal_ranks.all())
                    self.assertTrue(url.is_right_orthogonal().all())
                    self.assertEqual(self._first_elem_ranks(url, ss), (rrl.tucker_ranks, rrl.tt_ranks))
                    both = url.rank_adjustment_sweep('left_to_right')  # -> minimal, left-orthogonal
                    self.assertTrue(both.has_minimal_ranks.all())
                    self.assertTrue(both.is_left_orthogonal().all())
                    self.assertLessEqual(relerr(both.to_dense(), ux2.to_dense()), TOL)
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain.from_t3(x).rank_adjustment_sweep('sideways')

    def test_is_left_right_orthogonal(self):
        for shape, tr, ttr, ss in self._cases():
            with self.subTest(shape=shape, stack=ss):
                ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss))
                self.assertFalse(ux.is_left_orthogonal().all())          # a random T3 is in neither form
                self.assertFalse(ux.is_right_orthogonal().all())
                uxL = ux.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores()
                uxR = ux.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores()
                self.assertTrue(uxL.is_left_orthogonal().all())
                self.assertFalse(uxL.is_right_orthogonal().all())
                self.assertTrue(uxR.is_right_orthogonal().all())
                self.assertFalse(uxR.is_left_orthogonal().all())
                self.assertTrue(ux.t3svd()[0].is_left_orthogonal().all())  # a t3svd result is left-orthogonal

    def test_t3svd_assume_orthogonal(self):
        # assume_orthogonal=True (input already right-orthogonal) skips the orthogonalization; same result.
        for shape, tr, ttr, ss in self._cases():
            ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss))
            uR = ux.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores()  # right-orthogonal
            self.assertTrue(uR.is_right_orthogonal().all())
            for mtk, mtt in [(None, 2), (3, 2), (2, 2)]:
                with self.subTest(shape=shape, stack=ss, max_tucker=mtk, max_tt=mtt):
                    a, _, _ = uR.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt, assume_orthogonal=True)
                    b, _, _ = uR.t3svd(max_tucker_ranks=mtk, max_tt_ranks=mtt)
                    self.assertLessEqual(relerr(a.to_dense(), b.to_dense()), TOL)
                    self.assertTrue(a.is_left_orthogonal().all())

    def test_t3svd_non_minimal(self):
        for shape, tr, ttr in [((5, 6), (8, 4), (1, 3, 1)), ((6, 7, 8), (5, 5, 5), (1, 40, 40, 1))]:
            with self.subTest(shape=shape):
                x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                ux2, _, _ = ux.t3svd()
                x2, _, _ = x.t3svd()
                self.assertLessEqual(relerr(ux2.to_dense(), x2.to_dense()), TOL)
                self._assert_ranks_match(ux2, x2)

    def test_t3svd_per_stack_max_ranks(self):
        # the variety: different rank caps per stack element, in one call
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (4, 5, 4), (1, 4, 4, 1), stack_shape=(2,))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
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
        ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1)))
        with self.assertRaises(TypeError):
            ux.t3svd(rtol=1e-3)
        with self.assertRaises(TypeError):
            ux.t3svd(atol=1e-3)

    # ---- validate (structural hard errors) ----
    def test_validate_raises_on_bad_shape(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore[..., :-1], ux.shape, ux.masks)

    def test_validate_raises_on_nonbool_mask(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        bad = ut3.UT3Masks(ux.masks.tucker_edge_mask.astype(float), ux.masks.tt_edge_mask)
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore, ux.shape, bad)

    def test_validate_raises_on_bad_shape_tuple(self):
        # shape must be a length-d tuple of mode dims within the padded N (the int-tuple invariant)
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore, ux.shape[:-1], ux.masks)  # wrong length
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore,
                                         (ux.N + 1,) + ux.shape[1:], ux.masks)  # exceeds padded N

    def test_masks_value_hash_eq(self):
        # UT3Masks hashes/compares by mask CONTENT, so a rebuilt-but-identical holder is the same jit
        # cache key (no per-iteration recompile). A different rank structure is not equal.
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        a = ut3.UniformTuckerTensorTrain.from_t3(x).masks
        b = ut3.UniformTuckerTensorTrain.from_t3(x).masks                       # rebuilt -> distinct object, identical structure
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        c = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))).masks
        self.assertNotEqual(a, c)                         # different ranks -> not equal

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
                back = ux.to_t3()
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

    # (No from_canonical / from_tensor_train / to_tensor_train: those round-trip ragged CP/TT data
    # through TuckerTensorTrain -- ambiguous -- so they were removed. Convert explicitly via
    # UniformTuckerTensorTrain.from_t3 / .to_t3 instead, exercised by the round-trip tests elsewhere.)

    # ---- save / load ----
    def test_save_load_roundtrip(self):
        import tempfile, os
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
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

    # ---- dtype / copy ----
    def test_to_numpy_and_copy(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        cp = ux.copy()
        self.assertLessEqual(relerr(cp.to_dense(), x.to_dense()), TOL)
        # deep copy: the supercores are independent arrays (mirrors ragged T3*.copy)
        self.assertFalse(np.shares_memory(cp.tucker_supercore, ux.tucker_supercore))
        self.assertFalse(np.shares_memory(cp.tt_supercore, ux.tt_supercore))
        self.assertLessEqual(relerr(ux.to_numpy().to_dense(), x.to_dense()), TOL)

    # ---- backend-only path: every frontend op is reproducible on the raw .data tuple ----
    def test_backend_only_on_data(self):
        import t3toolbox.backend.ut3_conversions as conv
        import t3toolbox.backend.ut3_orthogonalization as bo
        import t3toolbox.backend.ut3_operations as bops
        import t3toolbox.backend.ut3_linalg as bl
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        y = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        ux, uy = ut3.UniformTuckerTensorTrain.from_t3(x), ut3.UniformTuckerTensorTrain.from_t3(y)

        # to_dense, orthogonalize, add+squash, inner -- all via backend functions on .data
        self.assertLessEqual(relerr(conv.ut3_to_dense(ux.data), ux.to_dense()), TOL)
        self.assertLessEqual(relerr(
            conv.ut3_to_dense(bo.ut3_down_orthogonalize_tucker_cores(ux.data)),
            ux.down_orthogonalize_tucker_cores().to_dense()), TOL)
        self.assertLessEqual(relerr(
            conv.ut3_to_dense(bops.ut3_squash_tails(bl.ut3_add(ux.data, uy.data))),
            (ux + uy).to_dense()), TOL)
        orth_data = lambda d: bo.ut3_left_orthogonalize_tt_cores(bo.ut3_down_orthogonalize_tucker_cores(d))
        self.assertLessEqual(relerr(
            bl.ut3_inner_product(orth_data(ux.data), orth_data(uy.data)),
            ux.inner(uy)), TOL)


def _corrupt(ux, scale=1e3):
    """Add ``scale`` * garbage to EVERY padded slot of ``ux`` (Tucker padding AND the TT padding, including
    the boundary-bond slots a corewise gradient step leaves nonzero); the real region is unchanged. A
    correct (mask-once) op must be UNAFFECTED (``docs/uniform_equivalence_contract.md``)."""
    scs = ux.supercores
    ind = type(ux)(*([np.ones_like(s) for s in scs] + [ux.shape, ux.masks])).apply_masks().supercores
    new = [sc + scale * (1.0 - i) for sc, i in zip(scs, ind)]
    return type(ux)(*(new + [ux.shape, ux.masks]))


class TestUniformGarbageAndDegenerate(unittest.TestCase):
    """The garbage-padded-input prong (``docs/contributor/testing_strategy.md``) for the structural
    arithmetic, and the ``d = 1`` degenerate case -- both regressions from the 2026-08-22 review."""

    def test_squash_and_arithmetic_are_garbage_robust(self):
        # ut3_squash_tails used to sum the UNMASKED boundary bonds, so garbage in a padded boundary-bond
        # slot changed the tensor -- through squash_tails, +, -, and sum_stack (all squash).
        for shape, tr, ttr in STRUCTURES + [((5, 6, 7), (3, 4, 2), (2, 3, 2, 2))]:   # incl. unsquashed tails
            for ss in STACK_SHAPES:
                with self.subTest(shape=shape, tt=ttr, stack=ss):
                    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                    y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
                    ux = _corrupt(ut3.UniformTuckerTensorTrain.from_t3(x, squash_tails=False))
                    uy = _corrupt(ut3.UniformTuckerTensorTrain.from_t3(y, squash_tails=False))
                    xd, yd = x.to_dense(), y.to_dense()
                    self.assertLessEqual(relerr(ux.squash_tails().to_dense(), xd), TOL)
                    self.assertLessEqual(relerr((ux + uy).to_dense(), xd + yd), TOL)
                    self.assertLessEqual(relerr((ux - uy).to_dense(), xd - yd), TOL)
                    # sum_stack sums over EVERY stack axis (the new leading one and the original ss)
                    summed = np.sum(xd + yd, axis=tuple(range(len(ss)))) if ss else xd + yd
                    self.assertLessEqual(
                        relerr(ut3.UniformTuckerTensorTrain.stack((ux, uy)).sum_stack().to_dense(), summed), TOL)

    def test_d1_degenerates_to_vector(self):
        # A one-mode T3 is a vector in a rank-n subspace; every uniform op must reduce to that case
        # (the uniform squash used to duplicate the single core, breaking everything downstream).
        import t3toolbox.uniform_frame_variations_format as ufvf
        for ttr, ss in [((1, 1), ()), ((2, 3), ()), ((1, 1), (2,)), ((2, 2), (2, 3))]:
            with self.subTest(tt=ttr, stack=ss):
                x = t3.TuckerTensorTrain.randn((7,), (3,), ttr, stack_shape=ss)
                y = t3.TuckerTensorTrain.randn((7,), (3,), ttr, stack_shape=ss)
                ux = ut3.UniformTuckerTensorTrain.from_t3(x, squash_tails=False)
                uy = ut3.UniformTuckerTensorTrain.from_t3(y, squash_tails=False)
                xd, yd = x.to_dense(), y.to_dense()
                self.assertLessEqual(relerr(ux.squash_tails().to_dense(), xd), TOL)
                self.assertLessEqual(relerr((ux + uy).to_dense(), xd + yd), TOL)
                self.assertLessEqual(relerr(ux.norm(), norm(xd.reshape(ss + (-1,)), axis=-1)), TOL)
                self.assertLessEqual(relerr(ux.inner(uy), np.sum(xd * yd, axis=-1)), TOL)
                self.assertLessEqual(relerr(ux.t3svd()[0].to_dense(), xd), TOL)
                self.assertLessEqual(relerr(ux.rank_adjustment_sweep('right_to_left').to_dense(), xd), TOL)
                self.assertLessEqual(relerr(ux.rank_adjustment_sweep('left_to_right').to_dense(), xd), TOL)
                self.assertTrue(np.all(ux.t3svd()[0].is_left_orthogonal()))
                frame = ufvf.UT3Frame.from_ut3(ux)
                self.assertTrue(np.all(frame.is_orthogonal()))
                self.assertLessEqual(relerr(frame.to_dense(), xd), TOL)



class TestReviewC13Uniform(unittest.TestCase):
    def test_padded_width_mismatch_is_a_structural_error(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        a = ut3.UniformTuckerTensorTrain.from_t3(x)
        b = ut3.UniformTuckerTensorTrain.from_t3(x, N=9)
        with self.assertRaises(ValueError) as cm:
            a + b
        self.assertIn('padded', str(cm.exception))
        with self.assertRaises(ValueError):
            a.inner(b)

if __name__ == '__main__':
    unittest.main()


class TestStructuralGuards(unittest.TestCase):
    """Review cluster-1 (frontend, uniform): malformed inputs fail with structural errors naming the
    problem, not deep numpy errors (H3-7, R8-6), and rank metadata stays host numpy (R8-9)."""

    def setUp(self):
        np.random.seed(0)
        self.x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
        self.ux = ut3.UniformTuckerTensorTrain.from_t3(self.x)

    def test_mul_is_scalar_only(self):
        """H3-7: UT3 * UT3 used to die inside numpy; now a TypeError naming the ragged route."""
        uy = ut3.UniformTuckerTensorTrain.from_t3(self.x)
        with self.assertRaises(TypeError):
            self.ux * uy
        with self.assertRaises(TypeError):
            self.ux * np.ones(3)
        y = self.ux * 2.0                                   # scalars still fine (incl. 0-d arrays)
        self.assertTrue(np.allclose(np.asarray(y.to_t3().to_dense()),
                                    2.0 * np.asarray(self.x.to_dense())))

    def test_stack_rejects_mismatched_pads(self):
        """R8-6: stacking leaves with different padded sizes used to die inside numpy."""
        uz = ut3.UniformTuckerTensorTrain.from_t3(self.x, n=4, r=4)
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain.stack([self.ux, uz])
        s2 = ut3.UniformTuckerTensorTrain.stack([self.ux, self.ux])    # matching layouts still fine
        self.assertEqual(s2.stack_shape, (2,))

    def test_truediv_is_scalar_only(self):
        """R1-12: x / 2 works (new on both layers); array divisors are named TypeErrors."""
        y = self.ux / 2.0
        self.assertTrue(np.allclose(np.asarray(y.to_t3().to_dense()),
                                    np.asarray(self.x.to_dense()) / 2.0))
        with self.assertRaises(TypeError):
            self.ux / np.ones(3)

    def test_minimal_ranks_are_host_numpy(self):
        """R8-9: rank metadata is host numpy even on a jax-backed train (the masks/aux rule)."""
        try:
            import jax  # noqa: F401
        except ImportError:
            self.skipTest('jax not installed')
        mn = self.ux.to_jax().minimal_ranks
        import t3toolbox.backend.common as common
        for fam in mn:
            self.assertTrue(common.is_numpy_ndarray(np.asarray(fam)))
            self.assertFalse(common.tree_contains_jax((fam,)))


# ---------------------------------------------------------------------- exact-mask / stack-matrix prongs
# The `(C, force_pad)` config matrix of `docs/contributor/testing_strategy.md` for this file, plus the
# varying-rank stack (the determinantal variety). Promoted from the 2026-08-22 review's
# `repros/H5/repro_exact_masks.py` and `repros/R8/three_prong_sweep.py` (Phase D).
_CONFIGS = [((), False), ((2,), False), ((), True), ((2,), True), ((2, 3), False), ((2, 3), True)]
_PAD = dict(N=9, n=5, r=5)         # forced-larger padding: EVERY core gets a padded region
_STRUCT_ASYM = ((5, 7, 6), (2, 3, 2), (1, 2, 2, 1))   # distinct mode sizes (asymmetric)


def _prefix(ranks, size):  # host bool mask: arange(size) < ranks[..., None]
    return np.arange(size) < np.asarray(ranks)[..., None]


def _is_prefix(mask):
    return bool(np.array_equal(mask, _prefix(mask.sum(axis=-1), mask.shape[-1])))


def _varying_stack():
    """A (2,)-stack whose elements have DIFFERENT ranks (clean padding); returns (stacked, [xa, xb])."""
    xa = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 3, 2), (1, 2, 2, 1))
    xb = t3.TuckerTensorTrain.randn((6, 7, 5), (4, 2, 3), (1, 3, 2, 1))
    ua = ut3.UniformTuckerTensorTrain.from_t3(xa, N=7, n=4, r=3)
    ub = ut3.UniformTuckerTensorTrain.from_t3(xb, N=7, n=4, r=3)
    return ut3.UniformTuckerTensorTrain.stack([ua, ub]), [xa, xb]


class TestExactOutputMasks(unittest.TestCase):
    """Prong 2 of ``docs/contributor/testing_strategy.md``: EXACT output masks, derived NON-circularly
    from the ragged twin's ranks (never from the object's own construction). Dense-vs-ragged alone is
    blind to a too-permissive (phantom-rank) mask, because clean padding makes the extra slots zero."""

    def setUp(self):
        np.random.seed(0)

    def _pair(self, ss, fp):
        shape, tr, ttr = _STRUCT_ASYM
        x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, **(_PAD if fp else {}))
        return x, ux

    def _assert_masks(self, u, tucker_ranks, tt_ranks):
        # expected: prefix masks of the (non-circular) expected ranks, broadcast over the stack
        exp_tk = np.broadcast_to(np.asarray(tucker_ranks).reshape((u.d,) + (1,) * len(u.stack_shape)),
                                 (u.d,) + u.stack_shape)
        exp_tt = np.broadcast_to(np.asarray(tt_ranks).reshape((u.d + 1,) + (1,) * len(u.stack_shape)),
                                 (u.d + 1,) + u.stack_shape)
        self.assertTrue(np.array_equal(u.masks.data[0], _prefix(exp_tk, u.n)))
        self.assertTrue(np.array_equal(u.masks.data[1], _prefix(exp_tt, u.r)))

    def test_t3svd_and_sweep_masks_match_ragged(self):
        shape, tr, ttr = _STRUCT_ASYM
        caps_tk = tuple(max(1, n - 1) for n in tr)
        caps_tt = (1,) + tuple(max(1, r - 1) for r in ttr[1:-1]) + (1,)
        for ss, fp in _CONFIGS:
            with self.subTest(ss=ss, fp=fp):
                x, ux = self._pair(ss, fp)
                xr = x.t3svd()[0]
                u = ux.t3svd()[0]
                self._assert_masks(u, xr.tucker_ranks, xr.tt_ranks)
                self.assertLessEqual(relerr(u.to_dense(), xr.to_dense()), TOL)
                xr2 = x.t3svd(max_tucker_ranks=caps_tk, max_tt_ranks=caps_tt)[0]
                u2 = ux.t3svd(max_tucker_ranks=caps_tk, max_tt_ranks=caps_tt)[0]
                self._assert_masks(u2, xr2.tucker_ranks, xr2.tt_ranks)
                self.assertLessEqual(relerr(u2.to_dense(), xr2.to_dense()), 1e-8)
                xr3 = xr2.rank_adjustment_sweep('right_to_left')
                u3 = u2.rank_adjustment_sweep('right_to_left')
                self._assert_masks(u3, xr3.tucker_ranks, xr3.tt_ranks)
                self.assertLessEqual(relerr(u3.to_dense(), xr3.to_dense()), TOL)

    def test_orthogonalization_masks_match_ragged(self):
        for ss, fp in _CONFIGS:
            x, ux = self._pair(ss, fp)
            for op in ('down_orthogonalize_tucker_cores', 'up_orthogonalize_tt_cores',
                       'left_orthogonalize_tt_cores', 'right_orthogonalize_tt_cores'):
                with self.subTest(ss=ss, fp=fp, op=op):
                    a, b = getattr(ux, op)(), getattr(x, op)()
                    self._assert_masks(a, b.tucker_ranks, b.tt_ranks)
                    self.assertLessEqual(relerr(a.to_dense(), b.to_dense()), TOL)
                    # no real content outside the mask (re-masking must not change the tensor)
                    self.assertLessEqual(relerr(a.apply_masks().to_dense(), a.to_dense()), TOL)

    def test_add_masks_exact_concatenation(self):
        # + concatenates the edge masks (interior), boundary bonds -> [True, False, ...] (rank 1); the
        # result may be GAPPY (working form) -- to_t3 must still be exact and t3svd re-canonicalizes.
        for ss, fp in _CONFIGS:
            with self.subTest(ss=ss, fp=fp):
                x, ux = self._pair(ss, fp)
                y, uy = self._pair(ss, fp)
                s, sr = ux + uy, x + y
                self.assertLessEqual(relerr(s.to_dense(), sr.to_dense()), TOL)
                tkm_e = np.concatenate([ux.masks.data[0], uy.masks.data[0]], axis=-1)
                ttm_e = np.concatenate([ux.masks.data[1], uy.masks.data[1]], axis=-1)
                b = np.zeros_like(ttm_e[0]); b[..., 0] = True
                ttm_e[0] = b; ttm_e[-1] = b
                self.assertTrue(np.array_equal(s.masks.data[0], tkm_e))
                self.assertTrue(np.array_equal(s.masks.data[1], ttm_e))
                self.assertLessEqual(relerr(s.to_t3().to_dense() if not ss else
                                            np.asarray([e.to_dense() for e in np.asarray(s.to_t3(), dtype=object).reshape(-1)]),
                                            sr.to_dense().reshape((-1,) + x.shape) if ss else sr.to_dense()), TOL)
                sc = s.t3svd()[0]
                self.assertTrue(all(_is_prefix(np.asarray(m)) for m in sc.masks.data))
                self.assertLessEqual(relerr(sc.to_dense(), sr.to_dense()), TOL)

    def test_sum_stack_masks_match_ragged(self):
        # sum_stack routes through + so its masks are in GAPPY working form (uniform_masks_vs_ranks.md);
        # the non-circular contract is: per-slot rank COUNTS == the ragged sum's ranks, and t3svd
        # re-canonicalizes to prefix masks with the ragged svd ranks.
        for ss, fp in [c for c in _CONFIGS if c[0]]:
            with self.subTest(ss=ss, fp=fp):
                x, ux = self._pair(ss, fp)
                u, r = ux.sum_stack(), x.sum_stack()
                self.assertLessEqual(relerr(u.to_dense(), r.to_dense()), TOL)
                self.assertEqual(tuple(u.masks.data[0].sum(-1).tolist()), tuple(r.tucker_ranks))
                self.assertEqual(tuple(u.masks.data[1].sum(-1).tolist()), tuple(r.tt_ranks))
                uc, rc = u.t3svd()[0], r.t3svd()[0]
                self.assertTrue(all(_is_prefix(np.asarray(m)) for m in uc.masks.data))
                self._assert_masks(uc, rc.tucker_ranks, rc.tt_ranks)
                self.assertLessEqual(relerr(uc.to_dense(), rc.to_dense()), TOL)

    def test_varying_rank_stack_per_element_masks(self):
        # per-element ranks over a varying-rank C stack == each element's own ragged ranks
        ust, (xa, xb) = _varying_stack()
        u = ust.t3svd()[0]
        for i, r in enumerate((xa.t3svd()[0], xb.t3svd()[0])):
            with self.subTest(elem=i, op='t3svd'):
                self.assertEqual(tuple(u.masks.data[0][:, i].sum(-1).tolist()), tuple(r.tucker_ranks))
                self.assertEqual(tuple(u.masks.data[1][:, i].sum(-1).tolist()), tuple(r.tt_ranks))
                self.assertLessEqual(relerr(u.to_dense()[i], r.to_dense()), TOL)
        u2 = u.rank_adjustment_sweep('right_to_left')
        for i, r in enumerate((xa.t3svd()[0].rank_adjustment_sweep('right_to_left'),
                               xb.t3svd()[0].rank_adjustment_sweep('right_to_left'))):
            with self.subTest(elem=i, op='rank_adjustment_sweep'):
                self.assertEqual(tuple(u2.masks.data[0][:, i].sum(-1).tolist()), tuple(r.tucker_ranks))
                self.assertEqual(tuple(u2.masks.data[1][:, i].sum(-1).tolist()), tuple(r.tt_ranks))
                self.assertLessEqual(relerr(u2.to_dense()[i], r.to_dense()), TOL)
        for op in ('down_orthogonalize_tucker_cores', 'up_orthogonalize_tt_cores',
                   'left_orthogonalize_tt_cores', 'right_orthogonalize_tt_cores'):
            a = getattr(ust, op)()
            for i, xx in enumerate((xa, xb)):
                b = getattr(xx, op)()
                with self.subTest(elem=i, op=op):
                    self.assertEqual(tuple(a.masks.data[0][:, i].sum(-1).tolist()), tuple(b.tucker_ranks))
                    self.assertEqual(tuple(a.masks.data[1][:, i].sum(-1).tolist()), tuple(b.tt_ranks))
                    self.assertLessEqual(relerr(a.to_dense()[i], b.to_dense()), TOL)
        # per-ELEMENT caps (arrays over the stack) truncate each element like its own ragged capped svd
        caps_tk = np.array([[2, 3], [2, 2], [2, 2]]); caps_tt = np.array([[1, 1], [2, 2], [2, 2], [1, 1]])
        u3 = ust.t3svd(max_tucker_ranks=caps_tk, max_tt_ranks=caps_tt)[0]
        for i, xx in enumerate((xa, xb)):
            r = xx.t3svd(max_tucker_ranks=tuple(caps_tk[:, i]), max_tt_ranks=tuple(caps_tt[:, i]))[0]
            with self.subTest(elem=i, op='t3svd(per-element caps)'):
                self.assertEqual(tuple(u3.masks.data[0][:, i].sum(-1).tolist()), tuple(r.tucker_ranks))
                self.assertEqual(tuple(u3.masks.data[1][:, i].sum(-1).tolist()), tuple(r.tt_ranks))
                self.assertLessEqual(relerr(u3.to_dense()[i], r.to_dense()), 1e-8)
        self.assertLessEqual(relerr(ust.sum_stack().to_dense(), xa.to_dense() + xb.to_dense()), TOL)
