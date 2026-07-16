# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Oracle tests for the fourteen contractions that used to FUSE two named blocks into one.

Six delegated to a twin that folded W+K into a single block; eight merged K+C into one einsum letter
``X``. Both forms are numerically exact (a pure reinterpretation), so no dense test could ever see
them -- they were found only when a downstream consumer tried to shard. Unfusing them (the passive
block now rides as ``'...'``, every axis intact -- see the SHARDING block in contractions.py) made all
fourteen GENUINE contractions rather than thin delegating wrappers, so each needs a real oracle rather
than a smoke test.

The oracle here is a plain ``np.einsum`` with the FULL EXPLICIT subscript -- every block spelled out,
one letter per axis, built per shape. That is derived independently of the implementation (which uses
a fixed subscript plus reshapes), so it checks the reshape bookkeeping rather than restating it.

The matrix covers EMPTY blocks (the "= 1 when empty" convention) and MULTI-AXIS blocks, which is where
the flattening bookkeeping actually bites. Numpy-only, per the house convention: the backend is
backend-agnostic, so jax computes the same numbers and its dispatch is covered by test_dispatch.py.
"""
import unittest

import numpy as np

import t3toolbox.backend.contractions as contractions

# (W_shape, K_shape, C_shape)
SHAPE_MATRIX = [
    ((), (), ()),                 # every block empty
    ((5,), (2,), (3,)),           # single-axis blocks
    ((), (2,), (3,)),             # empty W
    ((5,), (), (3,)),             # empty K
    ((5,), (2,), ()),             # empty C
    ((2, 3), (2,), (2, 3)),       # multi-axis W and C
    ((2, 3), (2, 2), (2, 3)),     # every block multi-axis
    ((4,), (1,), (1,)),           # singleton K and C (degenerate but legal)
]

D, T, I, O, A = 2, 3, 4, 5, 6

LETTERS = 'wxyzpqrsuv'    # per-axis letters for the oracle's W/K/C axes (disjoint from d,t,i,o,a)


def _sub(n_w, n_k, n_c):
    """Distinct per-axis oracle letters for the W, K and C blocks."""
    w = LETTERS[:n_w]
    k = LETTERS[n_w:n_w + n_k]
    c = LETTERS[n_w + n_k:n_w + n_k + n_c]
    return w, k, c


class TestUnfusedContractions(unittest.TestCase):
    """Each rewritten contraction vs a plain np.einsum with the full explicit subscript."""

    def _operands(self, seed, *shapes):
        r = np.random.default_rng(seed)
        return [r.standard_normal(s) for s in shapes]

    # ---------------------------------------------------------------- the 6 ex-delegation sites
    # These fused W+K into one block by handing the array to a WC-named twin. The oracle spells W
    # and K out as individual axes.

    def test_WKCi_Cio_to_WKCo(self):
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, W + K + C + (I,), C + (I, O))
                want = np.einsum('%s%s%si,%sio->%s%s%so' % (w, k, c, c, w, k, c), x, y)
                got = contractions.WKCi_Cio_to_WKCo(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_tWKCi_Cio_to_tWKCo(self):
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (T,) + W + K + C + (I,), C + (I, O))
                want = np.einsum('t%s%s%si,%sio->t%s%s%so' % (w, k, c, c, w, k, c), x, y)
                got = contractions.tWKCi_Cio_to_tWKCo(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_tWKCo_Cio_to_tWKCi(self):
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (T,) + W + K + C + (O,), C + (I, O))
                want = np.einsum('t%s%s%so,%sio->t%s%s%si' % (w, k, c, c, w, k, c), x, y)
                got = contractions.tWKCo_Cio_to_tWKCi(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_dWKCi_dCio_to_dWKCo(self):
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (D,) + W + K + C + (I,), (D,) + C + (I, O))
                want = np.einsum('d%s%s%si,d%sio->d%s%s%so' % (w, k, c, c, w, k, c), x, y)
                got = contractions.dWKCi_dCio_to_dWKCo(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_dtWKCi_dCio_to_dtWKCo(self):
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (D, T) + W + K + C + (I,), (D,) + C + (I, O))
                want = np.einsum('dt%s%s%si,d%sio->dt%s%s%so' % (w, k, c, c, w, k, c), x, y)
                got = contractions.dtWKCi_dCio_to_dtWKCo(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_dtWKCo_dCio_to_dtWKCi(self):
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (D, T) + W + K + C + (O,), (D,) + C + (I, O))
                want = np.einsum('dt%s%s%so,d%sio->dt%s%s%si' % (w, k, c, c, w, k, c), x, y)
                got = contractions.dtWKCo_dCio_to_dtWKCi(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    # ---------------------------------------------------------------- the 4 ex-internal sites
    # These merged K+C into one einsum letter X ('Wo,WXa->WXao'). The oracle keeps K and C apart.
    # The `_to_KCao` / `_to_dKCao` pair SUM over the probe stack W (W absent from the output).

    def test_Wo_WKCa_to_WKCao(self):
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, W + (O,), W + K + C + (A,))
                want = np.einsum('%so,%s%s%sa->%s%s%sao' % (w, w, k, c, w, k, c), x, y)
                got = contractions.Wo_WKCa_to_WKCao(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_Wo_WKCa_to_KCao(self):
        """Sums over the probe stack W -- W is absent from the output."""
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, W + (O,), W + K + C + (A,))
                want = np.einsum('%so,%s%s%sa->%s%sao' % (w, w, k, c, k, c), x, y)
                got = contractions.Wo_WKCa_to_KCao(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_dWo_dWKCa_to_dWKCao(self):
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (D,) + W + (O,), (D,) + W + K + C + (A,))
                want = np.einsum('d%so,d%s%s%sa->d%s%s%sao' % (w, w, k, c, w, k, c), x, y)
                got = contractions.dWo_dWKCa_to_dWKCao(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_dWo_dWKCa_to_dKCao(self):
        """Sums over the probe stack W -- W is absent from the output."""
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (D,) + W + (O,), (D,) + W + K + C + (A,))
                want = np.einsum('d%so,d%s%s%sa->d%s%sao' % (w, w, k, c, k, c), x, y)
                got = contractions.dWo_dWKCa_to_dKCao(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)


    # ---------------------------------------------------------------- the 4 that the inventory MISSED
    # uWKCa_uWo_to_* / duWKCa_duWo_to_* are backed by the _assemble_dU_dxi[_d] helpers, which merged K+C
    # into the letter X exactly like the four above. The unfusing plan's "complete, mechanically derived"
    # inventory listed the _assemble_* helpers as already clean -- true of _assemble_dU_eta and
    # _assemble_dG_jet3, false of these two. Found by grepping the file for the fused letter rather than
    # trusting the list. The order axis u is summed (order-diagonal), and the _to_KCao forms also sum W.

    def test_uWKCa_uWo_to_WKCao(self):
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (T,) + W + K + C + (A,), (T,) + W + (O,))
                want = np.einsum('u%s%s%sa,u%so->%s%s%sao' % (w, k, c, w, w, k, c), x, y)
                got = contractions.uWKCa_uWo_to_WKCao(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_uWKCa_uWo_to_KCao(self):
        """Sums over the order axis u AND the probe stack W."""
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (T,) + W + K + C + (A,), (T,) + W + (O,))
                want = np.einsum('u%s%s%sa,u%so->%s%sao' % (w, k, c, w, k, c), x, y)
                got = contractions.uWKCa_uWo_to_KCao(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_duWKCa_duWo_to_dWKCao(self):
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (D, T) + W + K + C + (A,), (D, T) + W + (O,))
                want = np.einsum('du%s%s%sa,du%so->d%s%s%sao' % (w, k, c, w, w, k, c), x, y)
                got = contractions.duWKCa_duWo_to_dWKCao(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)

    def test_duWKCa_duWo_to_dKCao(self):
        """Sums over the order axis u AND the probe stack W."""
        for n, (W, K, C) in enumerate(SHAPE_MATRIX):
            with self.subTest(W=W, K=K, C=C):
                w, k, c = _sub(len(W), len(K), len(C))
                x, y = self._operands(n, (D, T) + W + K + C + (A,), (D, T) + W + (O,))
                want = np.einsum('du%s%s%sa,du%so->d%s%sao' % (w, k, c, w, k, c), x, y)
                got = contractions.duWKCa_duWo_to_dKCao(x, y)
                self.assertEqual(want.shape, got.shape)
                np.testing.assert_allclose(want, got, rtol=1e-13, atol=1e-13)


class TestPassiveBlockSplitIsIrrelevant(unittest.TestCase):
    """The passive block's internal split is NOT pinned by the operands -- and must not matter.

    In WKCi_Cio_to_WKCo, Cio pins len(C) but nothing pins len(W) vs len(K): W=(2,3),K=() and
    W=(2,),K=(3,) and W=(),K=(2,3) all present the SAME operand shapes. That is exactly why these
    contractions once fused, and why the unfused form rides the passive block as '...' rather than
    inventing an n_probe parameter to recover a split it never needs. This test pins the property
    that makes that legitimate: the result depends only on the CONCATENATION W+K, so every reading
    of the same operands agrees.
    """

    def test_all_readings_of_one_operand_agree(self):
        r = np.random.default_rng(0)
        x = r.standard_normal((2, 3, 3, I))     # prefix (2,3) = W+K under any split; C=(3,)
        y = r.standard_normal((3,) + (I, O))
        out = contractions.WKCi_Cio_to_WKCo(x, y)
        self.assertEqual((2, 3, 3, O), out.shape)
        # the explicit oracle, written with the prefix axes spelled out individually
        want = np.einsum('wxci,cio->wxco', x, y)
        np.testing.assert_allclose(want, out, rtol=1e-13, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
