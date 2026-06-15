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
        sm, tkm, ttm = ux.masks.data
        tucker_real = np.einsum('d...n,dN->d...nN', tkm.astype(float), sm.astype(float))  # 1 on real, 0 on pad
        garbage = np.random.randn(*ux.tucker_supercore.shape) * (1.0 - tucker_real)
        ux_g = ut3.UniformTuckerTensorTrain(ux.tucker_supercore + garbage, ux.tt_supercore, ux.masks)
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

    # ---- validate (structural hard errors) ----
    def test_validate_raises_on_bad_shape(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        ux = ut3.t3_to_ut3(x)
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore[..., :-1], ux.masks)

    def test_validate_raises_on_nonbool_mask(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        ux = ut3.t3_to_ut3(x)
        bad = ut3.UT3Masks(ux.masks.shape_mask.astype(float), ux.masks.tucker_edge_mask, ux.masks.tt_edge_mask)
        with self.assertRaises(ValueError):
            ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore, bad)

    # ---- dtype / copy ----
    def test_to_numpy_and_copy(self):
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        ux = ut3.t3_to_ut3(x)
        self.assertLessEqual(relerr(ux.copy().to_dense(), x.to_dense()), TOL)
        self.assertLessEqual(relerr(ux.to_numpy().to_dense(), x.to_dense()), TOL)


if __name__ == '__main__':
    unittest.main()
