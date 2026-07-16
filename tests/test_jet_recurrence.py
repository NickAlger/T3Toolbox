# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Equivalence of the EXPERIMENTAL banded-recurrence jet functions with the trs-based originals.

The banded forms (sampling_derivatives.compute_*_jets_banded, module-private) unroll the affine jet
convolution into a two-term recurrence -- no dense trs contraction. They are being workshopped one at
a time; each must give BIT-FOR-BIT-close results to the trs original it mirrors, over the same stack
shapes those originals are tested at (test_probe_derivatives.py): W in {(), (3,), (2,2)}, an optional
frame stack C, orders 0..4. Numpy-only, per the suite's convention (jax dispatch is test_dispatch).
"""
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.sampling_derivatives as pd


class TestMuJetsBanded(unittest.TestCase):
    STRUCTS = [
        ((4, 5),        (2, 3),       (1, 2, 1)),
        ((4, 5, 6),     (2, 3, 2),    (1, 2, 2, 1)),
        ((5, 4, 6, 5),  (2, 3, 2, 3), (1, 2, 3, 2, 1)),
    ]

    def _mu_inputs(self, STRUCT, W, C, ORDER, rng):
        """Reproduce t3_probe_derivatives' construction up to compute_mu_jets: a genuinely C-stacked
        T3, ambient probes projected onto the frame (mode dim = tucker rank), and the input jets."""
        shapes = STRUCT[0]
        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
        tucker_cores, tt_cores = x.data
        ww = [rng.standard_normal(W + (N,)) for N in shapes]
        pp = [rng.standard_normal(W + (N,)) for N in shapes]
        xis = pd.compute_xi(tucker_cores, ww)          # U_i w_i -> W + C + (nU_i,)
        dxis = pd.compute_xi(tucker_cores, pp)
        xi_jets = pd.build_input_jets(xis, dxis)
        trs = pd.binomial_combine_tensor(ORDER)
        return tt_cores, xi_jets, trs

    def test_banded_matches_trs_mu(self):
        rng = np.random.default_rng(0)
        for STRUCT in self.STRUCTS:
            for W in [(), (3,), (2, 2)]:
                for C in [(), (2,)]:
                    for ORDER in [0, 1, 2, 3, 4]:
                        with self.subTest(STRUCT=STRUCT[0], W=W, C=C, ORDER=ORDER):
                            tt_cores, xi_jets, trs = self._mu_inputs(STRUCT, W, C, ORDER, rng)

                            expected = pd.compute_mu_jets(tt_cores, xi_jets, trs)
                            actual = pd.compute_mu_jets_banded(tt_cores, xi_jets, trs)

                            self.assertEqual(len(actual), len(expected))
                            for i in range(len(expected)):
                                e, a = np.asarray(expected[i]), np.asarray(actual[i])
                                self.assertEqual(a.shape, e.shape)
                                denom = np.linalg.norm(e)
                                rel = np.linalg.norm(a - e) / denom if denom else np.linalg.norm(a)
                                self.assertLess(rel, 1e-12, 'core %d: rel err %.2e' % (i, rel))


if __name__ == '__main__':
    unittest.main()
