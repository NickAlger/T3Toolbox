# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Equivalence of the standard recurrence/scan/chunk jet functions with their dense `*_trs` twins.

The standard forms (sampling_derivatives.compute_mu_jets, compute_eta_jets, ... -- the ones wired into
the sampling-derivative call sites) unroll the binomial convolution into a two-term recurrence (affine
mu/nu/sigma/tau) or an order-scan (full eta/deta/tilde) -- no dense trs contraction. Each must give
results close to the dense `*_trs` reference it replaces, over the same stack shapes those references
are tested at (test_probe_derivatives.py): W in {(), (3,), (2,2)}, an optional frame stack C, orders
0..4. This keeps the (now call-site-orphaned) `*_trs` forms exercised: standard == trs here, and
standard == dense in test_probe_derivatives, so trs stays anchored to ground truth transitively.
Numpy-only, per the suite's convention (jax dispatch is test_dispatch).
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
        ((6, 6, 6),     (4, 4, 4),    (1, 2, 2, 1)),     # over-ranked Tucker: nU != nD frame slack
    ]                                                     # (review R6-8: break the nD == nU degeneracy)

    def _mu_inputs(self, STRUCT, W, C, ORDER, rng):
        """Reproduce t3_probe_derivatives' construction up to compute_mu_jets_trs: a genuinely C-stacked
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

    BANDED = {'fused': pd.compute_mu_jets}    # the standard mu (was compute_mu_jets_banded_fused)

    def test_banded_matches_trs_mu(self):
        rng = np.random.default_rng(0)
        for variant, fn in self.BANDED.items():
            for STRUCT in self.STRUCTS:
                for W in [(), (3,), (2, 2)]:
                    for C in [(), (2,)]:
                        for ORDER in [0, 1, 2, 3, 4]:
                            with self.subTest(variant=variant, STRUCT=STRUCT[0], W=W, C=C, ORDER=ORDER):
                                tt_cores, xi_jets, trs = self._mu_inputs(STRUCT, W, C, ORDER, rng)

                                expected = pd.compute_mu_jets_trs(tt_cores, xi_jets, trs)
                                actual = fn(tt_cores, xi_jets, trs)

                                self.assertEqual(len(actual), len(expected))
                                for i in range(len(expected)):
                                    e, a = np.asarray(expected[i]), np.asarray(actual[i])
                                    self.assertEqual(a.shape, e.shape)
                                    denom = np.linalg.norm(e)
                                    rel = np.linalg.norm(a - e) / denom if denom else np.linalg.norm(a)
                                    self.assertLess(rel, 1e-12, 'core %d: rel err %.2e' % (i, rel))

    def test_banded_matches_trs_nu(self):
        # nu = reverse(mu) wrapper, so equivalence follows from mu; check it directly anyway.
        rng = np.random.default_rng(1)
        for STRUCT in self.STRUCTS:
            for W in [(), (3,), (2, 2)]:
                for C in [(), (2,)]:
                    for ORDER in [0, 1, 2, 3, 4]:
                        with self.subTest(STRUCT=STRUCT[0], W=W, C=C, ORDER=ORDER):
                            tt_cores, xi_jets, trs = self._mu_inputs(STRUCT, W, C, ORDER, rng)
                            expected = pd.compute_nu_jets_trs(tt_cores, xi_jets, trs)
                            actual = pd.compute_nu_jets(tt_cores, xi_jets, trs)
                            self.assertEqual(len(actual), len(expected))
                            for i in range(len(expected)):
                                e, a = np.asarray(expected[i]), np.asarray(actual[i])
                                self.assertEqual(a.shape, e.shape)
                                denom = np.linalg.norm(e)
                                rel = np.linalg.norm(a - e) / denom if denom else np.linalg.norm(a)
                                self.assertLess(rel, 1e-12, 'core %d: rel err %.2e' % (i, rel))


class TestEtaJetsScanned(TestMuJetsBanded):
    """The memory-lean order-scan eta must equal the dense-trs compute_eta_jets_trs on real widths."""

    def test_scanned_matches_trs_eta(self):
        rng = np.random.default_rng(0)
        for STRUCT in self.STRUCTS:
            for W in [(), (3,), (2, 2)]:
                for C in [(), (2,)]:
                    for ORDER in [0, 1, 2, 3, 4]:
                        with self.subTest(STRUCT=STRUCT[0], W=W, C=C, ORDER=ORDER):
                            x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                            tucker_cores, tt_cores = x.data
                            shapes = STRUCT[0]
                            ww = [rng.standard_normal(W + (n,)) for n in shapes]
                            pp = [rng.standard_normal(W + (n,)) for n in shapes]
                            xi_jets = pd.build_input_jets(pd.compute_xi(tucker_cores, ww),
                                                          pd.compute_xi(tucker_cores, pp))
                            trs = pd.binomial_combine_tensor(ORDER)
                            mu = pd.compute_mu_jets_trs(tt_cores, xi_jets, trs)
                            nu = pd.compute_nu_jets_trs(tt_cores, xi_jets, trs)

                            expected = pd.compute_eta_jets_trs(tt_cores, mu, nu, trs)
                            actual = pd.compute_eta_jets(tt_cores, mu, nu, trs)

                            self.assertEqual(len(actual), len(expected))
                            for k in range(len(expected)):
                                e, a = np.asarray(expected[k]), np.asarray(actual[k])
                                self.assertEqual(a.shape, e.shape)
                                denom = np.linalg.norm(e)
                                rel = np.linalg.norm(a - e) / denom if denom else np.linalg.norm(a)
                                self.assertLess(rel, 1e-12, 'core %d: rel err %.2e' % (k, rel))


class TestForwardTangentJets(unittest.TestCase):
    """The forward-Jacobian tangent jets (sigma/tau banded, deta scanned) must equal the trs versions,
    over the K-stacked shapes of test_probe_derivatives.test_tangent_derivatives_K_stacked."""

    import t3toolbox.frame_variations_format as _bvf
    import t3toolbox.manifold as _t3m

    def _tangent_inputs(self, STRUCT, W, K, C, ORDER, rng):
        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
        frame, _ = self._bvf.t3_orthogonal_representations(x)
        v = self._t3m.COREWISE.randn(frame, stack_shape=K)
        up, down, left, right = frame.data
        dU, dG = v.variations.data
        shapes = STRUCT[0]
        ww = [rng.standard_normal(W + (n,)) for n in shapes]
        pp = [rng.standard_normal(W + (n,)) for n in shapes]
        xi = pd.build_input_jets(pd.compute_xi(up, ww), pd.compute_xi(up, pp))
        dxi = pd.build_input_jets(pd.compute_xi(dU, ww), pd.compute_xi(dU, pp))
        trs = pd.binomial_combine_tensor(ORDER)
        mu = pd.compute_mu_jets_trs(left, xi, trs)
        nu = pd.compute_nu_jets_trs(right, xi, trs)
        return dict(dG=dG, P=left, Q=right, O=down, xi=xi, dxi=dxi, mu=mu, nu=nu, trs=trs)

    def _close(self, got, ref, tag):
        self.assertEqual(len(got), len(ref))
        for k in range(len(ref)):
            a, b = np.asarray(got[k]), np.asarray(ref[k])
            self.assertEqual(a.shape, b.shape)
            denom = np.linalg.norm(b)
            rel = np.linalg.norm(a - b) / denom if denom else np.linalg.norm(a)
            self.assertLess(rel, 1e-12, '%s core %d: rel %.2e' % (tag, k, rel))

    def test_forward_tangent_jets_match_trs(self):
        rng = np.random.default_rng(0)
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        for W, K, C in [((), (), ()), ((), (3,), ()), ((2,), (3,), (2,)), ((2, 2), (2,), (2,))]:
            for ORDER in [0, 1, 2, 3]:
                with self.subTest(W=W, K=K, C=C, ORDER=ORDER):
                    d = self._tangent_inputs(STRUCT, W, K, C, ORDER, rng)
                    s_ref = pd.compute_sigma_jets_trs(d['dG'], d['Q'], d['O'], d['xi'], d['dxi'], d['mu'], d['trs'])
                    s_got = pd.compute_sigma_jets(d['dG'], d['Q'], d['O'], d['xi'], d['dxi'], d['mu'], d['trs'])
                    t_ref = pd.compute_tau_jets_trs(d['dG'], d['P'], d['O'], d['xi'], d['dxi'], d['nu'], d['trs'])
                    t_got = pd.compute_tau_jets(d['dG'], d['P'], d['O'], d['xi'], d['dxi'], d['nu'], d['trs'])
                    self._close(s_got, s_ref, 'sigma')
                    self._close(t_got, t_ref, 'tau')

                    dt_ref = pd.compute_deta_jets_trs(d['dG'], d['P'], d['Q'], d['mu'], d['nu'], s_ref, t_ref, d['trs'])
                    dt_got = pd.compute_deta_jets(d['dG'], d['P'], d['Q'], d['mu'], d['nu'], s_got, t_got, d['trs'])
                    self._close(dt_got, dt_ref, 'deta')


class TestTildeTangentJets(unittest.TestCase):
    """The transpose (tilde) edge-variable jets (sigma_tilde/tau_tilde scanned) must equal the trs
    versions -- a two-term reverse recurrence (prop) + a full reverse-convolution order-scan (src)."""

    import t3toolbox.frame_variations_format as _bvf
    import t3toolbox.manifold as _t3m

    def test_tilde_jets_match_trs(self):
        rng = np.random.default_rng(0)
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        shapes = STRUCT[0]
        for W, K, C in [((), (), ()), ((), (3,), ()), ((2,), (3,), (2,)), ((2, 2), (2,), (2,))]:
            for ORDER in [0, 1, 2, 3]:
                with self.subTest(W=W, K=K, C=C, ORDER=ORDER):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                    frame, _ = self._bvf.t3_orthogonal_representations(x)
                    v = self._t3m.COREWISE.randn(frame, stack_shape=K)
                    up, down, left, right = frame.data
                    ww = [rng.standard_normal(W + (n,)) for n in shapes]
                    pp = [rng.standard_normal(W + (n,)) for n in shapes]
                    xi = pd.build_input_jets(pd.compute_xi(up, ww), pd.compute_xi(up, pp))
                    trs = pd.binomial_combine_tensor(ORDER)
                    mu = pd.compute_mu_jets_trs(left, xi, trs)
                    nu = pd.compute_nu_jets_trs(right, xi, trs)
                    zt = [rng.standard_normal((ORDER + 1,) + W + K + C + (n,)) for n in shapes]
                    deta_t = pd.compute_deta_tilde_jets(up, zt)

                    for tag, ref, got in [
                        ('tau_tilde', pd.compute_tau_tilde_jets_trs(left, xi, deta_t, mu, trs),
                         pd.compute_tau_tilde_jets(left, xi, deta_t, mu, trs)),
                        ('sigma_tilde', pd.compute_sigma_tilde_jets_trs(right, xi, deta_t, nu, trs),
                         pd.compute_sigma_tilde_jets(right, xi, deta_t, nu, trs)),
                    ]:
                        self.assertEqual(len(got), len(ref))
                        for k in range(len(ref)):
                            e, a = np.asarray(ref[k]), np.asarray(got[k])
                            self.assertEqual(a.shape, e.shape)
                            denom = np.linalg.norm(e)
                            rel = np.linalg.norm(a - e) / denom if denom else np.linalg.norm(a)
                            self.assertLess(rel, 1e-12, '%s core %d: rel %.2e' % (tag, k, rel))


class TestAssemblyChunked(unittest.TestCase):
    """The W-chunked assembly must equal the dense assembly for any chunk_size (it is a reorganization
    of the same sum) -- both reducers: add (sum_over_probes) and concat (kept). Random uniform arrays
    suffice; the dense assembly's correctness is pinned by the transpose adjoint-identity tests."""

    def test_chunked_matches_dense(self):
        rng = np.random.default_rng(0)
        d = 5
        for K in [1, 3]:
            for order in [2, 3]:
                for W in [7, 10]:
                    r = 4
                    R = lambda *s: rng.standard_normal(s)
                    sig, tau, deta = R(d, order + 1, W, K, r), R(d, order + 1, W, K, r), R(d, order + 1, W, K, r)
                    xi, mu, nu = R(d, 2, W, r), R(d, order + 1, W, r), R(d, order + 1, W, r)
                    trs = pd.binomial_combine_tensor(order)
                    for sop in [True, False]:
                        ref = pd.assemble_tt_variation_jets_trs(sig, tau, deta, xi, mu, nu, trs, 1, sop)
                        for cs in [3, 4, W, W + 5]:      # non-divisor, divisor-ish, exact, oversize
                            with self.subTest(K=K, order=order, W=W, sum_over_probes=sop, chunk_size=cs):
                                got = pd.assemble_tt_variation_jets(
                                    sig, tau, deta, xi, mu, nu, trs, 1, sop, chunk_size=cs)
                                e, a = np.asarray(ref), np.asarray(got)
                                self.assertEqual(a.shape, e.shape)
                                self.assertLess(np.linalg.norm(a - e) / np.linalg.norm(e), 1e-12)

    def test_tucker_chunked_matches_dense(self):
        rng = np.random.default_rng(0)
        d = 5
        for K in [1, 3]:
            for order in [2, 3]:
                for W in [7, 10]:
                    nO, N = 4, 5
                    R = lambda *s: rng.standard_normal(s)
                    zt, dxt = R(d, order + 1, W, K, N), R(d, order + 1, W, K, nO)
                    ww, pp, eta = R(d, W, N), R(d, W, N), R(d, order + 1, W, nO)
                    for sop in [True, False]:
                        ref = pd.assemble_tucker_variation_jets_trs(zt, dxt, ww, pp, eta, 1, sop)
                        for cs in [3, 4, W, W + 5]:
                            with self.subTest(K=K, order=order, W=W, sum_over_probes=sop, chunk_size=cs):
                                got = pd.assemble_tucker_variation_jets(
                                    zt, dxt, ww, pp, eta, 1, sop, chunk_size=cs)
                                e, a = np.asarray(ref), np.asarray(got)
                                self.assertEqual(a.shape, e.shape)
                                self.assertLess(np.linalg.norm(a - e) / np.linalg.norm(e), 1e-12)


if __name__ == '__main__':
    unittest.main()
