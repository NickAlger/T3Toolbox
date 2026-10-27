# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Equivalence of the shipped jet kernels with their dense binomial-tensor twins.

Every jet product is a binomial convolution ``sum_{r+s=t} C(t,r) a^(r) b^(s)`` (or its adjoint
correlation); the derivation writes it as a contraction against ``trs[t,r,s] = C(t,r)[r+s=t]``. The
shipped kernels in ``sampling_derivatives`` never contract ``trs``: an affine input jet makes each product
a two-term banded recurrence (mu/nu/sigma/tau, the adjoint propagation, the two-row dxi~ correlation, the
TT assembly's sigma~/tau~ groups), and the full convolutions are order scans whose binomial weights come
from a shift recurrence on a carried jet (eta/deta, the adjoint deta~ source, the TT assembly's deta~
group). Each must equal its dense twin (``tests/_jet_trs_twins.py``, test-only -- the package never
imports it) to 1e-14 RELATIVE, not bitwise: the two forms sum in different orders. Swept over
``d in {1, 3, 4}``, orders 0..4, and the W / K / C stacks (empty, single- and multi-axis). The twins are
anchored to ground truth by ``test_probe_derivatives.py`` (shipped == dense multilinear oracle), so the
chain is closed. Numpy-only, per the suite's convention (jax dispatch is test_dispatch).
"""
import os
import sys
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.sampling_derivatives as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _jet_trs_twins as twins

TOL = 1e-14                                               # relative; the forms sum in different orders
ORDERS = (0, 1, 2, 3, 4)
STRUCTS = [
    ((5,),          (3,),         (1, 1)),               # d = 1
    ((4, 5, 6),     (2, 3, 2),    (1, 2, 2, 1)),         # d = 3
    ((5, 4, 6, 5),  (2, 3, 2, 3), (1, 2, 3, 2, 1)),      # d = 4
    ((6, 6, 6),     (4, 4, 4),    (1, 2, 2, 1)),         # over-ranked Tucker: nU != nD frame slack
]
STACKS = [((), (), ()), ((3,), (2,), ()), ((2, 2), (2,), (2,)), ((3,), (2, 2), (2, 2))]   # (W, K, C)


def _relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(b)
    return np.linalg.norm(a - b) / denom if denom else np.linalg.norm(a)


def _problem(STRUCT, W, K, C, order, rng):
    """A C-stacked frame, a K-stacked variation, paired samples, and every jet the kernels consume -- the
    frame jets built by the dense twins, so each kernel is fed exactly its twin's inputs."""
    shapes = STRUCT[0]
    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
    frame, _ = bvf.t3_orthogonal_representations(x)
    v = t3m.COREWISE.randn(frame, stack_shape=K)
    U, O, P, Q = frame.data
    dU, dG = v.variations.data
    ww = [rng.standard_normal(W + (n,)) for n in shapes]
    pp = [rng.standard_normal(W + (n,)) for n in shapes]
    trs = twins.binomial_combine_tensor(order)
    xi = pd.build_input_jets(pd.compute_xi(U, ww), pd.compute_xi(U, pp))
    dxi = pd.build_input_jets(pd.compute_xi(dU, ww), pd.compute_xi(dU, pp))
    mu = twins.compute_mu_jets_trs(P, xi, trs)
    nu = twins.compute_nu_jets_trs(Q, xi, trs)
    zt = [rng.standard_normal((order + 1,) + W + K + C + (n,)) for n in shapes]
    c = rng.standard_normal((order + 1,) + W + K + C)
    return dict(U=U, O=O, P=P, Q=Q, dU=dU, dG=dG, ww=ww, pp=pp, trs=trs, xi=xi, dxi=dxi, mu=mu, nu=nu,
                zt=zt, deta_t=pd.compute_deta_tilde_jets(U, zt), c=c, wj=pd.build_input_jets(ww, pp))


class _Sweep(unittest.TestCase):

    def _close(self, got, ref, tag):
        got, ref = tuple(got), tuple(ref)
        self.assertEqual(len(got), len(ref), tag)
        for k, (a, b) in enumerate(zip(got, ref)):
            self.assertEqual(np.shape(a), np.shape(b), '%s core %d' % (tag, k))
            err = _relerr(a, b)
            self.worst = max(self.worst, err)
            self.assertLess(err, TOL, '%s core %d: rel err %.2e' % (tag, k, err))

    def _sweep(self, check, seed):
        rng = np.random.default_rng(seed)
        self.worst = 0.0
        for STRUCT in STRUCTS:
            for W, K, C in STACKS:
                for order in ORDERS:
                    with self.subTest(d=len(STRUCT[0]), shapes=STRUCT[0], W=W, K=K, C=C, order=order):
                        check(_problem(STRUCT, W, K, C, order, rng), order, len(W))
        self.assertLess(self.worst, TOL)


class TestForwardJets(_Sweep):
    """The forward jets: the banded pushthroughs (mu / nu / sigma / tau, and the apply/entries terminal carries
    built on them) and the order-scan combines (eta / deta)."""

    def test_forward_jets_match_trs(self):
        def check(p, order, n_probe):
            trs = p['trs']
            self._close(pd.compute_mu_jets(p['P'], p['xi'], order), p['mu'], 'mu')
            self._close(pd.compute_nu_jets(p['Q'], p['xi'], order), p['nu'], 'nu')
            self._close(pd.compute_eta_jets(p['O'], p['mu'], p['nu']),
                        twins.compute_eta_jets_trs(p['O'], p['mu'], p['nu'], trs), 'eta')
            s_ref = twins.compute_sigma_jets_trs(p['dG'], p['Q'], p['O'], p['xi'], p['dxi'], p['mu'], trs)
            t_ref = twins.compute_tau_jets_trs(p['dG'], p['P'], p['O'], p['xi'], p['dxi'], p['nu'], trs)
            self._close(pd.compute_sigma_jets(p['dG'], p['Q'], p['O'], p['xi'], p['dxi'], p['mu']), s_ref, 'sigma')
            self._close(pd.compute_tau_jets(p['dG'], p['P'], p['O'], p['xi'], p['dxi'], p['nu']), t_ref, 'tau')
            self._close(pd.compute_deta_jets(p['dG'], p['P'], p['Q'], p['mu'], p['nu'], s_ref, t_ref),
                        twins.compute_deta_jets_trs(p['dG'], p['P'], p['Q'], p['mu'], p['nu'], s_ref, t_ref, trs),
                        'deta')
            self._close([pd._apply_derivatives_t3_from_xi_jets(p['xi'], p['P'], order)],
                        [twins.apply_derivatives_t3_trs(p['xi'], p['P'], trs)], 'apply (Euclidean)')
            self._close([pd._apply_derivatives_from_jets(p['xi'], p['dxi'], p['mu'], p['Q'], p['O'], p['dG'])],
                        [twins.apply_derivatives_tv_trs(p['xi'], p['dxi'], p['mu'], p['Q'], p['O'], p['dG'], trs)],
                        'apply (tangent)')
        self._sweep(check, 0)


class TestTransposeJets(_Sweep):
    """The transpose jets: the adjoint sweeps (a reverse banded recurrence + an order-scan correlation), the
    two-row dxi~ correlation, the TT assembly (banded sigma~/tau~ groups + a lag scan for deta~), and the
    apply/entries transpose (the seeded sigma^ sweep, the two-row dxi^, the banded assembly)."""

    def test_transpose_jets_match_trs(self):
        def check(p, order, n_probe):
            trs = p['trs']
            tt = twins.compute_tau_tilde_jets_trs(p['P'], p['xi'], p['deta_t'], p['mu'], trs)
            st = twins.compute_sigma_tilde_jets_trs(p['Q'], p['xi'], p['deta_t'], p['nu'], trs)
            self._close(pd.compute_tau_tilde_jets(p['P'], p['xi'], p['deta_t'], p['mu']), tt, 'tau_tilde')
            self._close(pd.compute_sigma_tilde_jets(p['Q'], p['xi'], p['deta_t'], p['nu']), st, 'sigma_tilde')
            # dxi~: only the two read rows are formed (the twin computes all order+1); row 1 is zero at order 0
            s2 = min(2, order + 1)
            dxi = pd.compute_dxi_tilde_jets(p['O'], p['mu'], p['nu'], st, tt)
            self.assertEqual([np.shape(a)[0] for a in dxi], [2] * len(dxi))
            if order == 0:
                self.assertTrue(all(np.all(a[1] == 0) for a in dxi))
            self._close([a[:s2] for a in dxi],
                        [a[:2] for a in twins.compute_dxi_tilde_jets_trs(p['O'], p['mu'], p['nu'], st, tt, trs)],
                        'dxi_tilde')
            for sop in (True, False):
                self._close(pd.assemble_tt_variation_jets(st, tt, p['deta_t'], p['xi'], p['mu'], p['nu'],
                                                          n_probe, sop),
                            twins.assemble_tt_variation_jets_trs(st, tt, p['deta_t'], p['xi'], p['mu'], p['nu'],
                                                                 trs, n_probe, sop),
                            'tt assembly (sum_over_probes=%s)' % sop)
            self._close(pd.compute_sigma_hat_jets(p['Q'], p['xi'], p['c']),
                        twins.compute_sigma_hat_jets_trs(p['Q'], p['xi'], p['c'], trs), 'sigma_hat')
            for sop in (True, False):
                got = pd._apply_derivatives_transpose_from_jets(p['c'], p['xi'], p['mu'], p['wj'], p['O'], p['Q'],
                                                                sop)
                ref = twins.apply_derivatives_transpose_trs(p['c'], p['xi'], p['mu'], p['wj'], p['O'], p['Q'],
                                                            trs, n_probe, sop)
                self._close(got[0], ref[0], 'apply transpose dU (sum_over_probes=%s)' % sop)
                self._close(got[1], ref[1], 'apply transpose dG (sum_over_probes=%s)' % sop)
        self._sweep(check, 1)


class TestUniformKernelsMatchRagged(unittest.TestCase):
    """The order-scan kernels map one per-core body over the cores on both layouts (``lax.map`` / ``lax.scan``
    on uniform+jax): a uniform supercore of equal-shape cores must give the ragged result, core by core. Random
    uniform-shape arrays suffice -- the kernels are mask-free and polymorphic."""

    def test_uniform_equals_ragged(self):
        rng = np.random.default_rng(2)
        d, r, n = 4, 3, 2
        R = lambda *s: rng.standard_normal(s)
        for W, K, C in STACKS:
            for order in ORDERS:
                with self.subTest(W=W, K=K, C=C, order=order):
                    op = order + 1
                    core = lambda *k: R(d, *k, *C, r, n, r)                     # (d,)+[K+]C+(r,n,r)
                    jet = lambda rows, *k: R(d, rows, *W, *k, *C, r)            # (d,order)+W+[K+]C+(r,)
                    O, P, Q, dG = core(), core(), core(), core(*K)
                    xi, mu, nu = R(d, 2, *W, *C, n), jet(op), jet(op)
                    sig, tau, st, tt = jet(op, *K), jet(op, *K), jet(op, *K), jet(op, *K)
                    deta_t = R(d, op, *W, *K, *C, n)
                    cases = {
                        'eta': lambda a: pd.compute_eta_jets(*a), 'eta_ops': (O, mu, nu),
                        'deta': lambda a: pd.compute_deta_jets(*a), 'deta_ops': (dG, P, Q, mu, nu, sig, tau),
                        'tau_tilde': lambda a: pd.compute_tau_tilde_jets(*a), 'tau_tilde_ops': (P, xi, deta_t, mu),
                        'dxi_tilde': lambda a: pd.compute_dxi_tilde_jets(*a), 'dxi_tilde_ops': (O, mu, nu, st, tt),
                    }
                    for name in ('eta', 'deta', 'tau_tilde', 'dxi_tilde'):
                        ops = cases[name + '_ops']
                        uni = np.asarray(cases[name](ops))
                        rag = np.stack(cases[name](tuple(tuple(o) for o in ops)))
                        self.assertEqual(uni.shape, rag.shape, name)
                        self.assertLess(_relerr(uni, rag), TOL, name)
                    for sop in (True, False):
                        ops = (st, tt, deta_t, xi, mu, nu)
                        uni = np.asarray(pd.assemble_tt_variation_jets(*ops, len(W), sop, chunk_size=None))
                        rag = np.stack(pd.assemble_tt_variation_jets(*(tuple(o) for o in ops), len(W), sop))
                        self.assertLess(_relerr(uni, rag), TOL, 'tt assembly')


class TestAssemblyChunked(unittest.TestCase):
    """The W-chunked assembly must equal the unchunked one for any chunk_size (it is a reorganization of the
    same sum) -- both reducers: add (sum_over_probes) and concat (kept). Random uniform arrays suffice; the TT
    reference is the dense twin itself, so this also pins the per-chunk recurrence kernel."""

    def test_chunked_matches_dense(self):
        rng = np.random.default_rng(0)
        d = 5
        for K in [1, 3]:
            for order in [0, 2, 3]:
                for W in [7, 10]:
                    r = 4
                    R = lambda *s: rng.standard_normal(s)
                    sig, tau, deta = R(d, order + 1, W, K, r), R(d, order + 1, W, K, r), R(d, order + 1, W, K, r)
                    xi, mu, nu = R(d, 2, W, r), R(d, order + 1, W, r), R(d, order + 1, W, r)
                    trs = twins.binomial_combine_tensor(order)
                    for sop in [True, False]:
                        ref = twins.assemble_tt_variation_jets_trs(sig, tau, deta, xi, mu, nu, trs, 1, sop)
                        for cs in [3, 4, W, W + 5]:      # non-divisor, divisor-ish, exact, oversize
                            with self.subTest(K=K, order=order, W=W, sum_over_probes=sop, chunk_size=cs):
                                got = pd.assemble_tt_variation_jets(sig, tau, deta, xi, mu, nu, 1, sop, chunk_size=cs)
                                e, a = np.asarray(ref), np.asarray(got)
                                self.assertEqual(a.shape, e.shape)
                                self.assertLess(_relerr(a, e), TOL)

    def test_tucker_chunked_matches_dense(self):
        rng = np.random.default_rng(0)
        d = 5
        for K in [1, 3]:
            for order in [0, 2, 3]:
                for W in [7, 10]:
                    nO, N = 4, 5
                    R = lambda *s: rng.standard_normal(s)
                    zt, dxt = R(d, order + 1, W, K, N), R(d, 2, W, K, nO)       # dxi~: its two read rows
                    ww, pp, eta = R(d, W, N), R(d, W, N), R(d, order + 1, W, nO)
                    for sop in [True, False]:
                        ref = pd.assemble_tucker_variation_jets(zt, dxt, ww, pp, eta, 1, sop, chunk_size=None)
                        for cs in [3, 4, W, W + 5]:
                            with self.subTest(K=K, order=order, W=W, sum_over_probes=sop, chunk_size=cs):
                                got = pd.assemble_tucker_variation_jets(zt, dxt, ww, pp, eta, 1, sop, chunk_size=cs)
                                e, a = np.asarray(ref), np.asarray(got)
                                self.assertEqual(a.shape, e.shape)
                                self.assertLess(_relerr(a, e), TOL)


if __name__ == '__main__':
    unittest.main()
