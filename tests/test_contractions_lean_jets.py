# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Oracle tests for the lean (recurrence/scan) probe-derivative jet contractions in contractions.py.

These are the sharding-aware twins of the dense ``trs_*`` jet contractions, extracted from the lean
compute_* jet functions in sampling_derivatives.py. Each is checked against a plain ``np.einsum`` with
the FULL EXPLICIT subscript (every block spelled out, one letter per axis) built per shape -- derived
independently of the implementation (fixed subscript + reshapes / '...'), so it checks the bookkeeping.
The matrix covers EMPTY and MULTI-AXIS blocks (the "= 1 when empty" convention), where the flattening
bites. Numpy-only, per the house convention (jax dispatch is test_dispatch.py).
"""
import unittest

import numpy as np

import t3toolbox.backend.contractions as contractions

# bond/mode single-axis sizes (disjoint letters from the block letters below)
A, I, B, S, T = 2, 4, 5, 2, 3
# per-axis letters for the W and C blocks (disjoint from the s,t,a,i,b singles)
WL, CL = 'wxyz', 'pqru'


def _blk(letters, shape):
    return ''.join(letters[:len(shape)])


class TestLeanJetContractions(unittest.TestCase):
    # (W_shape, C_shape): empty / single / multi-axis blocks
    WC = [((), ()), ((6,), (3,)), ((), (3,)), ((6,), ()), ((2, 3), (2, 3))]

    def _rand(self, shape):
        return np.random.default_rng(0).standard_normal(shape)

    def test_Caib_sWCi_to_sWCab(self):
        # C shared (letter), s + W passive ('...'); contract i, keep a,b
        for W, C in self.WC:
            with self.subTest(W=W, C=C):
                w, c = _blk(WL, W), _blk(CL, C)
                Caib = self._rand(C + (A, I, B))
                sWCi = self._rand((S,) + W + C + (I,))
                got = np.asarray(contractions.Caib_sWCi_to_sWCab(Caib, sWCi))
                ref = np.einsum(f'{c}aib,s{w}{c}i->s{w}{c}ab', Caib, sWCi)
                self.assertEqual(got.shape, (S,) + W + C + (A, B))
                self.assertTrue(np.allclose(got, ref))

    def test_tWCa_WCab_to_tWCb(self):
        # W + C ride as one '...' batch on both operands; contract a, keep order t
        for W, C in self.WC:
            with self.subTest(W=W, C=C):
                w, c = _blk(WL, W), _blk(CL, C)
                tWCa = self._rand((T,) + W + C + (A,))
                WCab = self._rand(W + C + (A, B))
                got = np.asarray(contractions.tWCa_WCab_to_tWCb(tWCa, WCab))
                ref = np.einsum(f't{w}{c}a,{w}{c}ab->t{w}{c}b', tWCa, WCab)
                self.assertEqual(got.shape, (T,) + W + C + (B,))
                self.assertTrue(np.allclose(got, ref))

    def test_stWCa_sWCab_to_tWCb(self):
        # fused: contract the jet-pair s AND the bond a; W + C ride as '...'
        for W, C in self.WC:
            with self.subTest(W=W, C=C):
                w, c = _blk(WL, W), _blk(CL, C)
                stWCa = self._rand((S, T) + W + C + (A,))
                sWCab = self._rand((S,) + W + C + (A, B))
                got = np.asarray(contractions.stWCa_sWCab_to_tWCb(stWCa, sWCab))
                ref = np.einsum(f'st{w}{c}a,s{w}{c}ab->t{w}{c}b', stWCa, sWCab)
                self.assertEqual(got.shape, (T,) + W + C + (B,))
                self.assertTrue(np.allclose(got, ref))


if __name__ == '__main__':
    unittest.main()
