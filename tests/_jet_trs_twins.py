# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Test-only reference twins: the sampling-derivative jet kernels written as DENSE binomial-tensor
contractions.

The binomial tensor ``trs[t,r,s] = C(t,r)`` if ``r + s == t`` else ``0`` (:py:func:`binomial_combine_tensor`)
is the product rule on Taylor jets: contracting it against two jets gives
``sum_{r+s=t} C(t,r) (.)^(r) (.)^(s)``. That is how the jet formulas are derived (``docs/symmetric_probe_
derivatives.tex``), and it is the most direct way to write each kernel -- one ``trs`` einsum per
term -- so these forms are the reference the shipped kernels are pinned against. The shipped kernels in
``t3toolbox.backend.sampling_derivatives`` never contract ``trs`` (a dense convolution tensor, wasteful of
flops and of peak memory): affine input jets become two-term banded recurrences, and the full convolutions
are order scans whose binomial weights come from a shift recurrence on a carried jet. Equality of the two
forms (``tests/test_jet_recurrence.py``, ``tests/backend/test_curvature.py``) is what keeps the shipped
kernels honest; ``tests/test_probe_derivatives.py`` anchors both to the dense multilinear oracles.

Everything here is ragged-only (tuples of per-core arrays) and stack-aware (``W + K + C``, frame-inner), and
lives only in the test suite -- the package never imports it. Signatures follow the kernels they twin, with
the ``trs`` operand the shipped kernels no longer take.
"""
import math

import numpy as np

import t3toolbox.backend.tt_operations as tt_operations
from t3toolbox.backend.contractions import contract


def binomial_combine_tensor(order):
    """``trs[t,r,s] = C(t,r)`` when ``r + s == t``, else ``0``; shape ``(order+1,) * 3``."""
    trs = np.zeros((order + 1, order + 1, order + 1))
    for t in range(order + 1):
        for r in range(t + 1):
            trs[t, r, t - r] = math.comb(t, r)
    return trs


def _s_size(trs):
    """The affine input jet's live orders: ``{0, 1}``, capped at the order."""
    return min(2, trs.shape[0])


# ------------------------------------------------------------------------------------------------------
# Forward jets
# ------------------------------------------------------------------------------------------------------

def _init_jet(order, stack_shape, r0):
    jet = np.zeros((order + 1,) + tuple(stack_shape) + (r0,))
    jet[0] = 1.0
    return jet


def compute_mu_jets_trs(tt_cores, xi_jets, trs):
    """Left pushthrough jets: ``mu_i^(t) = sum C(t,r) mu_{i-1}^(r) G_i xi_i^(s)`` (``s in {0,1}``), the jet
    ENTERING each core. len=d, elm_shape=(order+1,)+W+C+(rLi,)."""
    s2 = _s_size(trs)
    mu = _init_jet(trs.shape[0] - 1, xi_jets[0].shape[1:-1], tt_cores[0].shape[-3])
    out = []
    for G, xi in zip(tt_cores, xi_jets):
        out.append(mu)
        mu = contract('trs,rWCa,Caib,sWCi->tWCb', trs[:, :, :s2], mu, G, xi[:s2])
    return tuple(out)


def compute_nu_jets_trs(tt_cores, xi_jets, trs):
    """Right pushthrough jets: the left sweep of the reversed train, reversed."""
    return compute_mu_jets_trs(tt_operations.tt_reverse(tt_cores), xi_jets[::-1], trs)[::-1]


def compute_eta_jets_trs(tt_cores, mu_jets, nu_jets, trs):
    """Combine at each free mode: ``eta_i^(t) = sum_{r+s=t} C(t,r) mu^(r) G nu^(s)``."""
    return tuple(contract('trs,rWCa,Caib,sWCb->tWCi', trs, mu, G, nu)
                 for G, mu, nu in zip(tt_cores, mu_jets, nu_jets))


def _sigma_jet_step(sigma_jet, Q, O, dG, xi_jet, dxi_jet, mu_jet, trs):
    """One K-aware dense step of the sigma recursion: three affine pushthroughs (carried sigma through Q,
    the frame mu through the variation core dG and through the down frame O)."""
    s2 = _s_size(trs)
    trs_push = trs[:, :, :s2]
    n_frame = Q.ndim - 3
    return (contract('trs,rWKCa,Caib,sWCi->tWKCb', trs_push, sigma_jet, Q, xi_jet[:s2])
            + contract('trs,rWCa,KCaib,sWCi->tWKCb', trs_push, mu_jet, dG, xi_jet[:s2], len_C=n_frame)
            + contract('trs,rWCa,Caib,sWKCi->tWKCb', trs_push, mu_jet, O, dxi_jet[:s2]))


def _sigma_sweep_trs(var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets, trs):
    """The sigma sweep: ``(per-core entering jets, terminal carry)``."""
    sigma = np.zeros((trs.shape[0],) + dxi_jets[0].shape[1:-1] + (right_tt_cores[0].shape[-3],))
    out = []
    for Q, O, dG, xi, dxi, mu in zip(right_tt_cores, down_tt_cores, var_tt_cores, xi_jets, dxi_jets, mu_jets):
        out.append(sigma)
        sigma = _sigma_jet_step(sigma, Q, O, dG, xi, dxi, mu, trs)
    return tuple(out), sigma


def compute_sigma_jets_trs(var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets, trs):
    """Variation-leftward jets ``sigma_i = sigma_{i-1} Q_i(xi_i) + mu_{i-1} dG_i(xi_i) + mu_{i-1} O_i(dxi_i)``."""
    return _sigma_sweep_trs(var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets, trs)[0]


def compute_tau_jets_trs(var_tt_cores, left_tt_cores, down_tt_cores, xi_jets, dxi_jets, nu_jets, trs):
    """Variation-rightward jets: the sigma sweep of the reversed train, reversed."""
    rev = tt_operations.tt_reverse
    return compute_sigma_jets_trs(rev(var_tt_cores), rev(left_tt_cores), rev(down_tt_cores),
                                  xi_jets[::-1], dxi_jets[::-1], nu_jets[::-1], trs)[::-1]


def compute_deta_jets_trs(var_tt_cores, left_tt_cores, right_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets,
                          trs):
    """Tangent combine ``deta_i = sigma Q nu + mu dG nu + mu P tau``, three dense binomial combines."""
    out = []
    for P, Q, dG, mu, nu, sig, tau in zip(left_tt_cores, right_tt_cores, var_tt_cores, mu_jets, nu_jets,
                                          sigma_jets, tau_jets):
        n_frame = Q.ndim - 3
        out.append(contract('trs,rWKCa,Caib,sWCb->tWKCi', trs, sig, Q, nu)
                   + contract('trs,rWCa,KCaib,sWCb->tWKCi', trs, mu, dG, nu, len_C=n_frame)
                   + contract('trs,rWCa,Caib,sWKCb->tWKCi', trs, mu, P, tau))
    return tuple(out)


def apply_derivatives_t3_trs(xi_jets, tt_cores, trs):
    """Euclidean apply/entries derivatives from the input jets: the terminal carry of the dense mu sweep,
    bond summed. shape=(order+1,)+W+C."""
    s2 = _s_size(trs)
    mu = _init_jet(trs.shape[0] - 1, xi_jets[0].shape[1:-1], tt_cores[0].shape[-3])
    for G, xi in zip(tt_cores, xi_jets):
        mu = contract('trs,rWCa,Caib,sWCi->tWCb', trs[:, :, :s2], mu, G, xi[:s2])
    return np.sum(mu, axis=-1)


def apply_derivatives_tv_trs(xi_jets, dxi_jets, mu_jets, right_tt_cores, down_tt_cores, var_tt_cores, trs):
    """Riemannian apply/entries derivatives from the jets: the terminal carry of the dense sigma sweep,
    bond summed. shape=(order+1,)+W+K+C."""
    terminal = _sigma_sweep_trs(var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets, trs)[1]
    return np.sum(terminal, axis=-1)


# ------------------------------------------------------------------------------------------------------
# Transpose (adjoint) jets and the gradient assemblies
# ------------------------------------------------------------------------------------------------------

def _adj_sweep(P_cores, xi_jets, deta_tildes, edge_jets, trs):
    """The dense adjoint sweep: per core, the adjoint-hooked pushthrough (propagation, the input jet on
    ``r in {0,1}``) plus the deta~ source (a full binomial correlation), both wired as the transpose."""
    s2 = _s_size(trs)
    carry = np.zeros((trs.shape[0],) + deta_tildes[0].shape[1:-1] + (P_cores[0].shape[-3],))
    out = []
    for P, xi, deta_t, edge in zip(P_cores, xi_jets, deta_tildes, edge_jets):
        out.append(carry)
        carry = (contract('trs,tWKCa,Caib,rWCi->sWKCb', trs[:, :s2, :], carry, P, xi[:s2])
                 + contract('trs,rWCa,Caib,tWKCi->sWKCb', trs, edge, P, deta_t))
    return tuple(out)


def compute_tau_tilde_jets_trs(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs):
    """Adjoint-var-rightward jets tau~ (the dense adjoint sweep through P)."""
    return _adj_sweep(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs)


def compute_sigma_tilde_jets_trs(right_tt_cores, xi_jets, deta_tildes, nu_jets, trs):
    """Adjoint-var-leftward jets sigma~: the dense adjoint sweep of the reversed train, reversed."""
    return _adj_sweep(tt_operations.tt_reverse(right_tt_cores), xi_jets[::-1], deta_tildes[::-1],
                      nu_jets[::-1], trs)[::-1]


def compute_dxi_tilde_jets_trs(down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes, trs):
    """Adjoint-var-down jets dxi~, ALL ``order+1`` rows of the dense adjoint-hooked combines
    ``tau~ O nu + mu O sigma~`` (only the rows ``u in {0,1}`` are ever read: the shipped kernel returns just
    those two, so compare ``[:2]``)."""
    return tuple(contract('trs,tWKCa,Caib,sWCb->rWKCi', trs, tt, O, nu)
                 + contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu, O, st)
                 for O, mu, nu, st, tt in zip(down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes))


def assemble_tt_variation_jets_trs(sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets, trs, n_probe,
                                   sum_over_probes):
    """TT-core variation gradients: three order-less dense ``trs`` outer products
    ``mu (x) xi (x) sigma~ + tau~ (x) xi (x) nu + mu (x) deta~ (x) nu`` (W summed, or kept)."""
    s2 = _s_size(trs)
    out = 'KCaib' if sum_over_probes else 'WKCaib'
    out_tt = []
    for st, tt, dt, xi, mu, nu in zip(sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets):
        out_tt.append(contract('trs,rWCa,sWCi,tWKCb->' + out, trs[:, :, :s2], mu, xi[:s2], st, len_W=n_probe)
                      + contract('trs,tWKCa,rWCi,sWCb->' + out, trs[:, :s2, :], tt, xi[:s2], nu, len_W=n_probe)
                      + contract('trs,rWCa,tWKCi,sWCb->' + out, trs, mu, dt, nu, len_W=n_probe))
    return tuple(out_tt)


def compute_sigma_hat_jets_trs(right_tt_cores, xi_jets, c, trs):
    """The apply/entries transpose's seeded adjoint sweep via Q: a dense propagation-only sweep from the
    terminal bond (``c`` broadcast over it), right to left."""
    s2 = _s_size(trs)
    rev_Q, rev_xi = tt_operations.tt_reverse(right_tt_cores), xi_jets[::-1]
    carry = np.broadcast_to(c[..., None], tuple(c.shape) + (right_tt_cores[-1].shape[-1],))
    out = []
    for Q, xi in zip(rev_Q, rev_xi):
        out.append(carry)
        carry = contract('trs,tWKCa,Caib,rWCi->sWKCb', trs[:, :s2, :], carry, Q, xi[:s2])
    return tuple(out)[::-1]


def apply_derivatives_transpose_trs(c, xi_jets, mu_jets, w_jets, down_tt_cores, right_tt_cores, trs, n_probe,
                                    sum_over_probes):
    """The apply/entries derivative transpose from the jets: the seeded sigma^ sweep, the dense dxi^ combine
    ``mu O sigma^`` (all rows) and the single-term assembly ``dG = mu (x) xi (x) sigma^``, ``dU = dxi^ (x) w``.
    Returns ``(dU, dG)``."""
    s2 = _s_size(trs)
    sigma_hats = compute_sigma_hat_jets_trs(right_tt_cores, xi_jets, c, trs)
    gU, gG = ('KCao', 'KCaib') if sum_over_probes else ('WKCao', 'WKCaib')
    dU, dG = [], []
    for O, xi, mu, sh, wj in zip(down_tt_cores, xi_jets, mu_jets, sigma_hats, w_jets):
        dxh = contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu, O, sh)
        dG.append(contract('trs,rWCa,sWCi,tWKCb->' + gG, trs[:, :, :s2], mu, xi[:s2], sh, len_W=n_probe))
        dU.append(contract('uWKCa,uWo->' + gU, dxh[:s2], wj[:s2]))
    return tuple(dU), tuple(dG)
