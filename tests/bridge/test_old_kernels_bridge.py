# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""TEMPORARY (deleted in P20 with ``sampling_derivatives``): the engine against the library's old sampling kernels.

The old kernels take ``(ww, pp)`` and an integer order, work in DERIVATIVES (slot ``t`` is the ``t``-th derivative
along ``ww + s pp``), and their transpose takes its residual FORWARD (not turned). The engine is called here at the
input jets ``affine_jet(ww, pp)`` and the jet shape ``(n,)``, one jet variable (the old code has no other). The
conversions, slot by slot (derived in P4c and checked by script for P6):

- ``z``, ``mu``, ``nu``, ``eta``: old = ``t!`` times new (``compute_mu_jets[i]`` is the left of bond ``i``,
  ``compute_nu_jets[i]`` the right of bond ``i + 1``);
- ``g``: the old transpose (``tv_probe_derivatives_transpose``, and ``ProbeDerivativesKind.transpose``) computes
  ``sum_t t! J_t^T c_t``; given ``c_t = weight_t^2 r_t / t!`` in FORWARD order it is the engine's ``g``, whose seed
  is the same residual weighted, squared and turned;
- the old adjoint variables, under that same forward ``c``: ``deta_tilde``, ``tau_tilde``, ``sigma_tilde`` at slot
  ``r`` are ``xi_tilde``, ``mu_tilde``, ``nu_tilde`` at slot ``n - r``, divided by ``r!``; ``dxi_tilde``'s two rows
  ``u = 0, 1`` are ``eta_tilde`` at slot ``n - u`` (``u! = 1``).
- with no jets (``n = 0``): ``t3_probe`` and ``tv_probe_transpose`` of ``probing``, given ``weight^2 r``.
- ``Kp`` (P8): ``ProbeDerivativesKind.corewise_hessian`` at the base point ``(D, L)`` along ``(D_dot, L_dot)``,
  given the forward ``weight_t^2 r_t / t!`` (with no weight of its own), is the new ``Kp``: the old Hessian is of
  ``sum_t t! <c_t, z_t>``, which with that ``c`` is the new ``sum_t weight_t^2 <r_t, z_t>``, the seed held.
- **the apply kind** (P9a): with no jets, ``apply.t3_apply`` is ``z``, ``tv_apply`` is ``Jp``,
  ``tv_apply_transpose`` (and ``ApplyKind.transpose``) given ``weight^2 r`` is ``g`` and given ``weight^2 Jp`` is
  ``JtJp``, and ``t3_apply_corewise_hessian`` given ``weight^2 r`` is ``Kp``; at ``n`` = 1 to 3,
  ``t3_apply_derivatives`` and ``ApplyDerivativesKind.point_forward`` are ``t!`` times ``z``, ``.forward`` is ``t!``
  times ``Jp``, and ``.transpose`` and ``.corewise_hessian`` given the forward ``weight_t^2 r_t / t!`` (and
  ``weight_t^2 Jp_t / t!``) are ``g``, ``Kp`` (and ``JtJp``), the conversions of the probe kind;
- ``Jp`` (P7): ``ProbeDerivativesKind.forward`` (and ``tv_probe_derivatives``) is ``t!`` times the new ``Jp``;
  ``.transpose`` given ``weight_t^2 Jp_t / t!``, forward, is the new ``JtJp`` (the conversion of ``g``); with no
  stack of directions and with ``K = (2,)``.

**Old kernels left out, and why:**

- ``compute_sigma_jets``, ``compute_tau_jets``, ``compute_deta_jets`` (the dots of the frame form): their variables
  are compared through what reads them, ``Jp`` and ``JtJp`` (the old kernels take derivative-normalized input jets
  of their own layout; the outputs are the documented interface);
- ``compute_deta_tilde_tangent_jets``, ``compute_tau_tilde_tangent_jets``, ``compute_sigma_tilde_tangent_jets``,
  ``compute_dxi_tilde_tangent_jets``, ``compute_sigma_hat_tangent_jets`` (the two-hole): compared through what reads
  them, ``Kp`` (the outputs are the documented interface);
- ``compute_sigma_hat_jets`` (the seed of apply and entries on bond ``d``) and the apply kernels' other private
  sweeps: compared through what reads them, the apply kind's outputs (below);
- ``dense_*_derivatives``: the dense functions are already compared with them in ``test_sampling_definitions.py``
  (``TestAgainstTheOldKernels`` and ``TestDenseFunctions``), through the same conversions; the bridge does not repeat
  it.

numpy only; random frames (no consistency is needed by any of the compared kernels) on three problems, ``n`` from 0
to 3, a nonuniform weight.
"""
import math
import os
import sys
import unittest

import numpy as np

import t3toolbox.backend.apply as old_apply
import t3toolbox.backend.fitting as old_fitting
import t3toolbox.backend.jets as jets
import t3toolbox.backend.probing as probing
import t3toolbox.backend.sampling_derivatives as sampling_derivatives
import t3toolbox.backend.sampling_variables as sv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_sampling_variables as tsv          # the engine's problems

TOL = 1e-12
NAMES = ('z', 'r', 'g', 'xi', 'mu', 'nu', 'eta', 'xi_tilde', 'mu_tilde', 'nu_tilde', 'eta_tilde')


def relative_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0)


class TestOldKernels(unittest.TestCase):

    def setUp(self):
        self.cases = []
        for name, make in tsv.PROBLEMS:
            problem = make()
            for n in (0, 1, 2, 3):
                rng = np.random.default_rng(10 + n)
                box = (n + 1,) if n else ()
                y = [rng.standard_normal(box + problem.W + (N,)) for N in problem.shape]
                weight = 0.5 + rng.random((problem.d,) + box)
                sample = list(jets.affine_jet(problem.ww, problem.pp)) if n else problem.ww
                got = dict(zip(NAMES, sv.sampling_variables('probe', problem.frame, sample, NAMES,
                                                             order=(n,) if n else (), y=y, weight=weight)))
                self.cases.append((name, problem, n, weight, got))

    @staticmethod
    def per_slot(x, factors):
        return np.asarray(factors).reshape((len(factors),) + (1,) * (np.ndim(x) - 1)) * x

    def forward_seed(self, problem, n, weight, r):
        factorials = np.array([math.factorial(t) for t in range(n + 1)], dtype=float)
        return [self.per_slot(r[i], weight[i] ** 2 / factorials) for i in range(problem.d)]

    def test_z_and_the_plain_variables(self):
        for name, problem, n, weight, got in self.cases:
            with self.subTest(problem=name, order=n):
                if n == 0:
                    for a, b in zip(got['z'], probing.t3_probe(problem.ww, (problem.D, problem.L))):
                        self.assertLessEqual(relative_error(a, b), TOL)
                    continue
                factorials = np.array([math.factorial(t) for t in range(n + 1)], dtype=float)
                old_z = sampling_derivatives.t3_probe_derivatives(problem.ww, problem.pp, (problem.D, problem.L), n)
                for a, b in zip(got['z'], old_z):
                    self.assertLessEqual(relative_error(self.per_slot(a, factorials), b), TOL)
                D, U, L, R = problem.frame
                old_mu = sampling_derivatives.compute_mu_jets(L, got['xi'], n)
                old_nu = sampling_derivatives.compute_nu_jets(R, got['xi'], n)
                old_eta = sampling_derivatives.compute_eta_jets(U, old_mu, old_nu)
                for i in range(problem.d):
                    self.assertLessEqual(relative_error(self.per_slot(got['mu'][i], factorials), old_mu[i]), TOL)
                    self.assertLessEqual(relative_error(self.per_slot(got['nu'][i + 1], factorials), old_nu[i]), TOL)
                    self.assertLessEqual(relative_error(self.per_slot(got['eta'][i], factorials), old_eta[i]), TOL)

    def test_g_through_the_forward_seed(self):
        for name, problem, n, weight, got in self.cases:
            with self.subTest(problem=name, order=n):
                new = list(got['g'][0]) + list(got['g'][1])
                if n == 0:
                    old = probing.tv_probe_transpose([weight[i] ** 2 * got['r'][i] for i in range(problem.d)],
                                                     problem.ww, problem.frame, sum_over_probes=True)
                    for a, b in zip(new, list(old[0]) + list(old[1])):
                        self.assertLessEqual(relative_error(a, b), TOL)
                    continue
                forward = self.forward_seed(problem, n, weight, got['r'])
                old = sampling_derivatives.tv_probe_derivatives_transpose(
                    forward, problem.ww, problem.pp, problem.frame, n, sum_over_probes=True)
                for a, b in zip(new, list(old[0]) + list(old[1])):
                    self.assertLessEqual(relative_error(a, b), TOL)
                kind = old_fitting.ProbeDerivativesKind(order=n, weight=None, chunk_size=None)
                sample = (problem.ww, problem.pp)
                old = kind.transpose(forward, sample, problem.frame, kind.precompute(problem.frame, sample))
                for a, b in zip(new, list(old[0]) + list(old[1])):
                    self.assertLessEqual(relative_error(a, b), TOL)

    def test_the_adjoint_variables(self):
        for name, problem, n, weight, got in self.cases:
            if n == 0:
                continue
            with self.subTest(problem=name, order=n):
                factorials = np.array([math.factorial(t) for t in range(n + 1)], dtype=float)
                D, U, L, R = problem.frame
                forward = self.forward_seed(problem, n, weight, got['r'])
                old_mu = sampling_derivatives.compute_mu_jets(L, got['xi'], n)
                old_nu = sampling_derivatives.compute_nu_jets(R, got['xi'], n)
                deta = sampling_derivatives.compute_deta_tilde_jets(D, forward)
                tau = sampling_derivatives.compute_tau_tilde_jets(L, got['xi'], deta, old_mu)
                sigma = sampling_derivatives.compute_sigma_tilde_jets(R, got['xi'], deta, old_nu)
                dxi = sampling_derivatives.compute_dxi_tilde_jets(U, old_mu, old_nu, sigma, tau)
                turned = lambda x: self.per_slot(x[::-1], 1.0 / factorials)       # slot r: new[n - r] / r!
                for i in range(problem.d):
                    self.assertLessEqual(relative_error(turned(got['xi_tilde'][i]), deta[i]), TOL)
                    self.assertLessEqual(relative_error(turned(got['mu_tilde'][i]), tau[i]), TOL)
                    self.assertLessEqual(relative_error(turned(got['nu_tilde'][i + 1]), sigma[i]), TOL)
                    self.assertLessEqual(relative_error(got['eta_tilde'][i][::-1][:2], dxi[i]), TOL)


class TestTheDirection(unittest.TestCase):
    """``Jp`` and ``JtJp`` against the old derivative kind, at ``n`` = 1 to 3, with no stack of directions and with
    ``K = (2,)``."""

    def test_forward_and_transpose(self):
        for name, make in tsv.PROBLEMS:
            problem = make()
            for n in (1, 2, 3):
                rng = np.random.default_rng(20 + n)
                weight = 0.5 + rng.random((problem.d, n + 1))
                factorials = np.array([math.factorial(t) for t in range(n + 1)], dtype=float)
                per_slot = TestOldKernels.per_slot
                for label, variations in (('K=()', (problem.V, problem.H)),
                                          ('K=(2,)', tsv.stacked_directions(problem, seed=n))):
                    Jp, JtJp = sv.sampling_variables('probe', problem.frame, list(jets.affine_jet(problem.ww, problem.pp)),
                                                     ('Jp', 'JtJp'), order=(n,), weight=weight, variations=variations)
                    kind = old_fitting.ProbeDerivativesKind(order=n, weight=None, chunk_size=None)
                    sample = (problem.ww, problem.pp)
                    sweep = kind.precompute(problem.frame, sample)
                    with self.subTest(problem=name, order=n, stack=label):
                        old = kind.forward(variations, sample, problem.frame, sweep)
                        for a, b in zip(Jp, old):
                            self.assertLessEqual(relative_error(per_slot(a, factorials), b), TOL)
                        forward = [per_slot(Jp[i], weight[i] ** 2 / factorials) for i in range(problem.d)]
                        old = kind.transpose(forward, sample, problem.frame, sweep)
                        for a, b in zip(list(JtJp[0]) + list(JtJp[1]), list(old[0]) + list(old[1])):
                            self.assertLessEqual(relative_error(a, b), TOL)


class TestTheCurvature(unittest.TestCase):

    def test_Kp_against_the_old_corewise_hessian(self):
        for name, make in tsv.PROBLEMS:
            problem = make()
            for n in (1, 2, 3):
                rng = np.random.default_rng(30 + n)
                weight = 0.5 + rng.random((problem.d, n + 1))
                y = [rng.standard_normal((n + 1,) + problem.W + (N,)) for N in problem.shape]
                factorials = np.array([math.factorial(t) for t in range(n + 1)], dtype=float)
                for label, core_directions in (('K=()', (problem.D_dot, problem.L_dot)),
                                               ('K=(2,)', tsv.stacked_core_directions(problem, seed=n))):
                    Kp, r = sv.sampling_variables('probe', problem.frame, list(jets.affine_jet(problem.ww, problem.pp)),
                                                  ('Kp', 'r'), order=(n,), y=y, weight=weight,
                                                  core_directions=core_directions)
                    kind = old_fitting.ProbeDerivativesKind(order=n, weight=None, chunk_size=None)
                    sample = (problem.ww, problem.pp)
                    forward = [TestOldKernels.per_slot(r[i], weight[i] ** 2 / factorials) for i in range(problem.d)]
                    old = kind.corewise_hessian(core_directions, forward, sample, problem.frame,
                                                kind.precompute(problem.frame, sample), (problem.D, problem.L))
                    with self.subTest(problem=name, order=n, stack=label):
                        for a, b in zip(list(Kp[0]) + list(Kp[1]), list(old[0]) + list(old[1])):
                            self.assertLessEqual(relative_error(a, b), TOL)



class TestTheApplyKind(unittest.TestCase):
    """The apply kind against the old apply kernels and kinds: ``z``, ``g``, ``Jp``, ``JtJp`` and ``Kp``, with no jets
    and at ``n`` = 1 to 3, with ``K = ()`` and ``K = (2,)``, a nonuniform weight, on three problems (one with end bonds
    that differ between ``L`` and ``R``)."""

    @staticmethod
    def blocks(pair):
        return list(pair[0]) + list(pair[1])

    def assert_blocks(self, new, old):
        for a, b in zip(self.blocks(new), self.blocks(old)):
            self.assertLessEqual(relative_error(a, b), TOL)

    def test_against_the_old_kernels(self):
        per_slot = TestOldKernels.per_slot
        for name, make in tsv.PROBLEMS:
            problem = make()
            for n in (0, 1, 2, 3):
                rng = np.random.default_rng(40 + n)
                box = (n + 1,) if n else ()
                y = rng.standard_normal(box + problem.W)
                weight = 0.5 + rng.random(box)
                sample = list(jets.affine_jet(problem.ww, problem.pp)) if n else problem.ww
                factorials = np.array([math.factorial(t) for t in range(n + 1)], dtype=float)
                for label, variations, core_directions in (
                        ('K=()', (problem.V, problem.H), (problem.D_dot, problem.L_dot)),
                        ('K=(2,)', tsv.stacked_directions(problem, seed=n),
                         tsv.stacked_core_directions(problem, seed=n))):
                    z, r, g, Jp, JtJp, Kp = sv.sampling_variables(
                        'apply', problem.frame, sample, ('z', 'r', 'g', 'Jp', 'JtJp', 'Kp'), order=box and (n,), y=y,
                        weight=weight, variations=variations, core_directions=core_directions)
                    with self.subTest(problem=name, order=n, stack=label):
                        if n == 0:
                            weight_2 = weight ** 2
                            self.assertLessEqual(relative_error(z, old_apply.t3_apply((problem.D, problem.L),
                                                                                      problem.ww)), TOL)
                            self.assertLessEqual(relative_error(Jp, old_apply.tv_apply(problem.ww, variations,
                                                                                       problem.frame)), TOL)
                            self.assert_blocks(g, old_apply.tv_apply_transpose(weight_2 * r, problem.ww, problem.frame,
                                                                               sum_over_probes=True))
                            self.assert_blocks(JtJp, old_apply.tv_apply_transpose(weight_2 * Jp, problem.ww,
                                                                                  problem.frame, sum_over_probes=True))
                            self.assert_blocks(Kp, old_apply.t3_apply_corewise_hessian(
                                weight_2 * r, problem.ww, (problem.D, problem.L), core_directions))
                            kind = old_fitting.ApplyKind()
                            self.assert_blocks(g, kind.transpose(weight_2 * r, problem.ww, problem.frame,
                                                                 kind.precompute(problem.frame, problem.ww)))
                            continue
                        old_sample = (problem.ww, problem.pp)
                        old_z = sampling_derivatives.t3_apply_derivatives(problem.ww, problem.pp,
                                                                          (problem.D, problem.L), n)
                        self.assertLessEqual(relative_error(per_slot(z, factorials), old_z), TOL)
                        kind = old_fitting.ApplyDerivativesKind(order=n, weight=None)
                        sweep = kind.precompute(problem.frame, old_sample)
                        old_z = kind.point_forward((problem.D, problem.L), old_sample)
                        self.assertLessEqual(relative_error(per_slot(z, factorials), old_z), TOL)
                        self.assertLessEqual(relative_error(per_slot(Jp, factorials),
                                                            kind.forward(variations, old_sample, problem.frame, sweep)),
                                             TOL)
                        forward = per_slot(r, weight ** 2 / factorials)
                        self.assert_blocks(g, kind.transpose(forward, old_sample, problem.frame, sweep))
                        self.assert_blocks(JtJp, kind.transpose(per_slot(Jp, weight ** 2 / factorials), old_sample,
                                                                problem.frame, sweep))
                        self.assert_blocks(Kp, kind.corewise_hessian(core_directions, forward, old_sample,
                                                                     problem.frame, sweep, (problem.D, problem.L)))


if __name__ == '__main__':
    unittest.main()
