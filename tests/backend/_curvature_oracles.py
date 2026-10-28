# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Test-only oracles for the exact Riemannian Hessian's building blocks (the lift, the two-hole sweeps).

Two oracles, each independent of the kernel it checks:

* **The block-sum oracle.** For a function ``F(theta) = <c, S(Phi(theta))>`` that is LINEAR in the
  tensor (a sampling transpose, or an inner product with a fixed ``Z``), the core-coordinate Hessian
  applied to ``theta_hat`` is the gradient of ``theta -> DF(theta)[theta_hat]``. Since
  ``DPhi(theta)[theta_hat] = Phi(Y)`` for the doubled-rank T3 ``Y = tv_to_t3(t3_corewise_frame(theta),
  theta_hat)`` and ``Y`` is linear in ``theta`` (each of its blocks is a core of ``theta``, of
  ``theta_hat``, or zero), the Hessian is the ONE-HOLE gradient at ``Y`` read back onto the blocks where
  ``theta``'s cores sit: :py:func:`blocksum`. It uses only the one-hole kernel and ``tv_to_t3``.
* **The dense multilinear Hessian.** ``theta -> <Z, Phi(theta)>`` is multilinear in the ``2d`` cores, so
  its Hessian has zero diagonal blocks and its off-diagonal blocks are exact: block ``a`` of ``H theta_hat``
  is ``sum_{b != a} grad_{theta_a} <Z, Phi(theta[b <- theta_hat_b])>``, and each gradient of a function
  LINEAR in ``theta_a`` is read off by unit vectors through ``t3_to_dense``. Exact (no finite differences)
  and shares nothing with the sweeps. :py:func:`dense_inner_product_hessian` (with the gradient,
  :py:func:`dense_inner_product_gradient`), and the full core-coordinate Lagrangian Hessian of
  ``f(Phi(theta)) - <G, Phi(theta)>`` for a dense least-squares ``f``, :py:func:`dense_lagrangian_hessian`.
* **The tied lift** of a shared point, from dense pseudo-inverses of the stacked ``S_i^T`` (no shared-companion
  SVD, no production lift): :py:func:`tied_lift` / :py:func:`tied_lift_transpose`. With the dense Lagrangian
  Hessian it gives the shared manifold's exact Hessian independently of the tied-lift code (the Hessian of
  ``Phi o T``, ``T`` copying the group factor).

A third family is not an oracle but a **reference twin**: the dense-``trs`` forms of the probe-derivative
two-hole (:py:func:`probe_derivatives_two_hole_trs` and its pieces). The shipped kernels in
``sampling_derivatives`` are the product rule of the Gauss-Newton kernel each one differentiates, in its form
(a banded recurrence where a factor is an input jet, an order scan with a carried shifted jet where the
convolution is full, a factored assembly) and never contract the binomial tensor; the twins are the same
mathematics written as one dense binomial contraction per product-rule term, built from the GN twins of
``tests/_jet_trs_twins.py`` wherever one exists. :py:func:`tt_tangent_assembly_factored_trs` is the dense
factored assembly the two-hole shipped with before the kernels became recurrences (S3b), kept as a second
reference. They live only in the test suite -- never imported by the package -- and the shipped kernels agree
with them to ~1e-15 (pinned at 1e-14 in ``test_curvature.py``), as ``tests/test_jet_recurrence.py`` pins the
GN jet kernels to theirs. The plain kinds' two-holes (no order axis, so no binomial structure and no twin) have
unfactored references for their factored TT assemblies instead: :py:func:`tt_tangent_assembly_nine_term` (probe)
and :py:func:`apply_tt_tangent_assembly_three_term` (apply / entries).

All functions take unstacked data unless stated; the tests loop over stack elements. Every random draw
is the caller's (a seeded ``np.random.Generator``).
"""
import os
import sys

import numpy as np

import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.probing as probing
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv
from t3toolbox.backend.contractions import contract

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tests/: the GN twins
import _jet_trs_twins as twins


def doubled(theta, theta_hat):
    """The doubled-rank T3 ``Y`` with ``Phi(Y) = DPhi(theta)[theta_hat]`` (stack: ``theta_hat``'s)."""
    return tv.tv_to_t3(fvc.t3_corewise_frame(theta), theta_hat)


def blocksum(gY, theta):
    """Read a gradient with respect to the doubled-rank ``Y = doubled(theta, .)`` back onto ``theta``'s
    cores: the transpose of the (linear-in-``theta``) embedding ``theta -> Y``. Stack-aware (leading axes
    of ``gY`` ride along); the block offsets come from ``theta``'s core shapes."""
    gYU, gYG = gY
    U, G = theta
    d = len(U)
    gU = tuple(gYU[i][..., :U[i].shape[-2], :] for i in range(d))      # U_i sits in the top Tucker rows
    if d == 1:                                                         # Y_0 = [Gdot_0 ; G_0] on the n axis
        n = U[0].shape[-2]
        return gU, (gYG[0][..., :, n:, :],)
    gG = []
    for i in range(d):
        r, n, rn = G[i].shape[-3], G[i].shape[-2], G[i].shape[-1]
        g = gYG[i]
        if i == 0:                                       # [[Gdot, G], [G, 0]]: top-right and bottom-left
            gG.append(g[..., :, :n, rn:] + g[..., :, n:, :rn])
        elif i == d - 1:                                 # [[R ; Gdot], [0 ; O]]: top-first and bottom-second
            gG.append(g[..., :r, :n, :] + g[..., r:, n:, :])
        else:                                            # R (top-left), L (top, second-second), O (bottom)
            gG.append(g[..., :r, :n, :rn] + g[..., r:, :n, rn:] + g[..., r:, n:, :rn])
    return gU, tuple(gG)


def _blocks(theta):
    """theta = (U, G) as a flat list of 2d core arrays (Tucker first)."""
    return list(theta[0]) + list(theta[1])


def _unblocks(blocks, d):
    return tuple(blocks[:d]), tuple(blocks[d:])


def _dense(blocks, d):
    return np.asarray(t3c.t3_to_dense(_unblocks(blocks, d)))


def dense_inner_product_gradient(Z_dense, theta):
    """``grad_theta <Z_dense, Phi(theta)>``, exactly, by linearity in each block (unstacked)."""
    d = len(theta[0])
    th = _blocks(theta)
    out = []
    for a in range(2 * d):
        ga = np.zeros(np.shape(th[a]))
        for idx in np.ndindex(*ga.shape):
            base = list(th)
            e = np.zeros(ga.shape)
            e[idx] = 1.0
            base[a] = e
            ga[idx] = np.sum(Z_dense * _dense(base, d))
        out.append(ga)
    return _unblocks(out, d)


def dense_inner_product_hessian(Z_dense, theta, theta_hat):
    """``H theta_hat`` for ``H`` the core-coordinate Hessian of ``theta -> <Z_dense, Phi(theta)>``, exactly,
    by multilinearity (unstacked ``theta``, ``theta_hat``; a dense ``Z_dense``)."""
    d = len(theta[0])
    th, dh = _blocks(theta), _blocks(theta_hat)
    out = []
    for a in range(2 * d):
        ga = np.zeros_like(np.asarray(th[a]), dtype=float)
        for b in range(2 * d):
            if b == a:
                continue
            base = list(th)
            base[b] = dh[b]
            for idx in np.ndindex(*ga.shape):            # <Z, Phi> is linear in block a: unit vectors
                e = np.zeros(ga.shape)
                e[idx] = 1.0
                base[a] = e
                ga[idx] += np.sum(Z_dense * _dense(base, d))
        out.append(ga)
    return _unblocks(out, d)


def dense_core_jacobian(theta):
    """The Jacobian of ``theta -> vec(Phi(theta))`` (exact, by multilinearity), shape ``(prod N, n_theta)``,
    with ``theta``'s blocks flattened Tucker-first in C order."""
    d = len(theta[0])
    th = _blocks(theta)
    cols = []
    for a in range(2 * d):
        for idx in np.ndindex(*np.shape(th[a])):
            base = list(th)
            e = np.zeros(np.shape(th[a]))
            e[idx] = 1.0
            base[a] = e
            cols.append(_dense(base, d).ravel())
    return np.stack(cols, axis=1)


def flatten(theta):
    return np.concatenate([np.ravel(np.asarray(c)) for c in _blocks(theta)])


def unflatten(vec, like):
    d = len(like[0])
    out, o = [], 0
    for c in _blocks(like):
        k = int(np.prod(np.shape(c)))
        out.append(np.asarray(vec[o:o + k]).reshape(np.shape(c)))
        o += k
    return _unblocks(out, d)


def dense_lagrangian_hessian(A, y, theta, theta_hat, G_dense):
    """``H theta_hat`` for ``H`` the core-coordinate Hessian of the Lagrangian
    ``theta -> f(Phi(theta)) - <G_dense, Phi(theta)>``, ``f(X) = 1/2 ||A vec(X) - y||^2`` (a dense
    measurement matrix ``A``): ``J^T A^T A J theta_hat + H_{A^T r - G} theta_hat``, exact. With ``G_dense``
    the Riemannian gradient ``P_X A^T r`` this is ``(DPhi)^T Hess f (DPhi)`` (the vertical directions are in
    its null space); with ``G_dense = 0`` it is the plain Hessian of ``f o Phi``."""
    X = _dense(_blocks(theta), len(theta[0]))
    r = A @ X.ravel() - y
    J = dense_core_jacobian(theta)
    gn = J.T @ (A.T @ (A @ (J @ flatten(theta_hat))))
    Atr = (A.T @ r).reshape(X.shape)
    curv = flatten(dense_inner_product_hessian(Atr - G_dense, theta, theta_hat))
    return unflatten(gn + curv, theta)


def _tied_lift_matrices(frame, groups):
    """The eps = 0 lift's matrices by plain ``numpy.linalg.pinv`` (unstacked): per nontrivial sharing group the
    pseudo-inverse of the stacked ``M_g = [S_i1^T; ...]`` with its row splits, per other mode ``(S_i^T)^+``, per
    bond ``T_i^+``. The center factors are ``fv_center_factors``'s (pinned to the dense spectra in T1)."""
    import t3toolbox.backend.fv_operations as fvo
    _, STs, Ts = fvo.fv_center_factors(frame)
    tied = [g for g in groups if len(g) > 1]
    group_pinvs = []
    for g in tied:
        M = np.concatenate([np.asarray(STs[i]) for i in g], axis=0)
        group_pinvs.append((g, np.linalg.pinv(M), np.cumsum([0] + [STs[i].shape[0] for i in g])))
    mode_pinvs = [np.linalg.pinv(np.asarray(S)) for S in STs]
    bond_pinvs = [np.linalg.pinv(np.asarray(T)) for T in Ts]
    return group_pinvs, mode_pinvs, bond_pinvs


def tied_lift(frame, groups, v):
    """An independent TIED lift at eps = 0 (unstacked coordinates ``v``): one factor direction per nontrivial group,
    ``Udot_g = pinv(M_g) [V_i1; ...]``, copied to the group's modes; the per-mode and bond lifts elsewhere -- the
    lift of the tied parametrization ``Phi(T theta_sh)``, built from dense pseudo-inverses (no SVD of the shared
    companion, no ``tv_corewise_lift``)."""
    group_pinvs, mode_pinvs, bond_pinvs = _tied_lift_matrices(frame, groups)
    V, H = v
    d = len(V)
    Ud = [mode_pinvs[i] @ np.asarray(V[i]) for i in range(d)]
    for g, Mp, _ in group_pinvs:
        Ug = Mp @ np.concatenate([np.asarray(V[i]) for i in g], axis=0)
        for i in g:
            Ud[i] = Ug
    Gd = [np.einsum('aib,bc->aic', np.asarray(H[i]), bond_pinvs[i]) for i in range(d - 1)] + [np.asarray(H[-1])]
    return tuple(Ud), tuple(Gd)


def tied_lift_transpose(frame, groups, y):
    """The adjoint of :py:func:`tied_lift` (unstacked raw ``y``): each group's copies summed, then ``pinv(M_g)^T``
    split by the row splits; the per-mode and bond transposes elsewhere."""
    group_pinvs, mode_pinvs, bond_pinvs = _tied_lift_matrices(frame, groups)
    Y, Z = y
    d = len(Y)
    V = [mode_pinvs[i].T @ np.asarray(Y[i]) for i in range(d)]
    for g, Mp, splits in group_pinvs:
        Vs = Mp.T @ sum(np.asarray(Y[i]) for i in g)
        for jj, i in enumerate(g):
            V[i] = Vs[splits[jj]:splits[jj + 1]]
    H = [np.einsum('aic,bc->aib', np.asarray(Z[i]), bond_pinvs[i]) for i in range(d - 1)] + [np.asarray(Z[-1])]
    return tuple(V), tuple(H)


def vertical_direction(theta, rng):
    """A random raw direction along the gauge orbit of ``theta = (U, G)``: ``DPhi(theta)[result] = 0``
    exactly (to first order). Tucker gauge ``U_i -> A U_i``, ``G_i -> G_i x_2 A^{-T}``; TT gauge
    ``G_i -> G_i B``, ``G_{i+1} -> B^{-1} G_{i+1}``. Unstacked."""
    U, G = theta
    d = len(U)
    dU = [np.zeros(np.shape(u)) for u in U]
    dG = [np.zeros(np.shape(g)) for g in G]
    for i in range(d):
        n = U[i].shape[-2]
        Ai = rng.standard_normal((n, n))
        dU[i] += Ai @ U[i]
        dG[i] -= np.einsum('aib,ij->ajb', G[i], Ai)
    for i in range(d - 1):
        r = G[i].shape[-1]
        B = rng.standard_normal((r, r))
        dG[i] += np.einsum('aib,bc->aic', G[i], B)
        dG[i + 1] -= np.einsum('bc,cid->bid', B, G[i + 1])
    return tuple(dU), tuple(dG)


# ------------------------------------------------------------------------------------------------------
# The probe-derivative two-hole: dense-trs reference twins (stack-aware: W + K + C, ragged)
# ------------------------------------------------------------------------------------------------------
# Notation: a trailing ``d`` is the derivative along theta_hat (it carries the tangent stack K); the undotted
# operands carry W + C only (the residual jets z~ have no K). theta = (U, G) is the corewise point, so its
# corewise frame is (U, G, G, G): every frame core of the GN transpose is G and its derivative is Gd.

def _w_c_counts(xi_jets, frame_core):
    """(n_probe, n_frame) from an undotted input jet (2,)+W+C+(n,) and a C-only frame core C+(a,i,b)."""
    n_frame = frame_core.ndim - 3
    return xi_jets[0].ndim - 2 - n_frame, n_frame


def adjoint_sweep_tangent_trs(P, Pd, xi, xid, deta, detd, edge, edged, tildes, trs):
    """The derivative of ``twins._adj_sweep`` (the dense adjoint sweep ``tilde' = prop(tilde, P, xi) +
    src(edge, P, deta)``) along the dotted operands: six dense-trs terms per core, K on the dotted operand
    of each (carry / core / xi ; edge / core / deta). ``tildes`` are the undotted sweep's outputs (the carry
    entering each core). Returns the dotted tildes, len=d, elm_shape=(order+1,)+W+K+C+(rL_i,)."""
    n_probe, n_frame = _w_c_counts(xi, P[0])
    kw = dict(len_W=n_probe, len_C=n_frame)
    s2 = min(2, trs.shape[0])
    trs_xi = trs[:, :s2, :]
    cd = np.zeros((trs.shape[0],) + xid[0].shape[1:-1] + (P[0].shape[-3],))
    out = []
    for k in range(len(P)):
        out.append(cd)
        cd = (contract('trs,tWKCa,Caib,rWCi->sWKCb', trs_xi, cd,        P[k],  xi[k][:s2],  **kw)
              + contract('trs,tWCa,KCaib,rWCi->sWKCb', trs_xi, tildes[k], Pd[k], xi[k][:s2],  **kw)
              + contract('trs,tWCa,Caib,rWKCi->sWKCb', trs_xi, tildes[k], P[k],  xid[k][:s2], **kw)
              + contract('trs,rWKCa,Caib,tWCi->sWKCb', trs, edged[k], P[k],  deta[k], **kw)
              + contract('trs,rWCa,KCaib,tWCi->sWKCb', trs, edge[k],  Pd[k], deta[k], **kw)
              + contract('trs,rWCa,Caib,tWKCi->sWKCb', trs, edge[k],  P[k],  detd[k], **kw))
    return tuple(out)


def tau_tilde_tangent_jets_trs(G, Gd, xi, xid, deta, detd, mu, mud, tau_tildes, trs):
    """Reference twin of ``sd.compute_tau_tilde_tangent_jets``."""
    return adjoint_sweep_tangent_trs(G, Gd, xi, xid, deta, detd, mu, mud, tau_tildes, trs)


def sigma_tilde_tangent_jets_trs(G, Gd, xi, xid, deta, detd, nu, nud, sigma_tildes, trs):
    """Reference twin of ``sd.compute_sigma_tilde_tangent_jets`` (the reversed sweep)."""
    rv = tto.tt_reverse
    return adjoint_sweep_tangent_trs(rv(G), rv(Gd), xi[::-1], xid[::-1], deta[::-1], detd[::-1],
                                     nu[::-1], nud[::-1], sigma_tildes[::-1], trs)[::-1]


def deta_tilde_tangent_jets_trs(Ud, ztildes, n_frame):
    """Reference twin of ``sd.compute_deta_tilde_tangent_jets``: ``Udot_i z~_i`` per core."""
    return tuple(contract('tWCo,KCio->tWKCi', z, u, len_C=n_frame) for z, u in zip(ztildes, Ud))


def dxi_tilde_tangent_jets_trs(G, Gd, mu, mud, nu, nud, sigma_tildes, sigma_tildes_d, tau_tildes, tau_tildes_d, trs):
    """Reference twin of ``sd.compute_dxi_tilde_tangent_jets``: the six-term product rule of the dense dxi~
    combine (``twins.compute_dxi_tilde_jets_trs``) at ``O = G``, ALL ``order+1`` rows (the shipped kernel forms
    only the two read rows ``u in {0, 1}``; compare ``[:2]``)."""
    n_probe, n_frame = _w_c_counts(mu, G[0])           # mu (order+1,)+W+C+(r,): the same W + C layout as xi
    kw = dict(len_W=n_probe, len_C=n_frame)
    out = []
    for k in range(len(G)):
        out.append(contract('trs,tWKCa,Caib,sWCb->rWKCi', trs, tau_tildes_d[k], G[k],  nu[k],  **kw)
                   + contract('trs,tWCa,KCaib,sWCb->rWKCi', trs, tau_tildes[k],  Gd[k], nu[k],  **kw)
                   + contract('trs,tWCa,Caib,sWKCb->rWKCi', trs, tau_tildes[k],  G[k],  nud[k], **kw)
                   + contract('trs,rWKCa,Caib,tWCb->sWKCi', trs, mud[k], G[k],  sigma_tildes[k],   **kw)
                   + contract('trs,rWCa,KCaib,tWCb->sWKCi', trs, mu[k],  Gd[k], sigma_tildes[k],   **kw)
                   + contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu[k],  G[k],  sigma_tildes_d[k], **kw))
    return tuple(out)


def tucker_tangent_assembly_trs(ztildes, dxi_tildes_d, ww, pp, eta_jets_d, n_frame):
    """Reference twin of ``sd.assemble_tucker_tangent_variation_jets`` (W summed):
    ``sum_t etadot^(t) (x) z~^(t) + sum_u dxi~dot^(u) (x) w^(u)``."""
    s2 = min(2, eta_jets_d[0].shape[0])
    kw = dict(len_W=ww[0].ndim - 1, len_C=n_frame)
    return tuple(contract('tWKCa,tWCo->KCao', ed, z, **kw)
                 + contract('uWKCa,uWo->KCao', dxd[:s2], np.stack([w, p])[:s2], **kw)
                 for z, dxd, w, p, ed in zip(ztildes, dxi_tildes_d, ww, pp, eta_jets_d))


def tt_tangent_assembly_trs(sigma_tildes, sigma_tildes_d, tau_tildes, tau_tildes_d, deta_tildes, deta_tildes_d,
                            xi, xid, mu, mud, nu, nud, trs, n_frame):
    """Reference twin of ``sd.assemble_tt_tangent_variation_jets`` (W summed): the nine-term product rule of
    the GN assembly's three dense-trs outer products, one term per (group, dotted factor) -- the unfactored
    form the shipped kernel factors into pair intermediates."""
    s2 = min(2, trs.shape[0])
    t_sig, t_tau = trs[:, :, :s2], trs[:, :s2, :]
    kw = dict(len_W=xi[0].ndim - 2 - n_frame, len_C=n_frame)
    out = []
    for k in range(len(xi)):
        x, xd = xi[k][:s2], xid[k][:s2]
        out.append(contract('trs,rWKCa,sWCi,tWCb->KCaib', t_sig, mud[k], x,  sigma_tildes[k],   **kw)
                   + contract('trs,rWCa,sWKCi,tWCb->KCaib', t_sig, mu[k],  xd, sigma_tildes[k],   **kw)
                   + contract('trs,rWCa,sWCi,tWKCb->KCaib', t_sig, mu[k],  x,  sigma_tildes_d[k], **kw)
                   + contract('trs,tWKCa,rWCi,sWCb->KCaib', t_tau, tau_tildes_d[k], x,  nu[k],  **kw)
                   + contract('trs,tWCa,rWKCi,sWCb->KCaib', t_tau, tau_tildes[k],   xd, nu[k],  **kw)
                   + contract('trs,tWCa,rWCi,sWKCb->KCaib', t_tau, tau_tildes[k],   x,  nud[k], **kw)
                   + contract('trs,rWKCa,tWCi,sWCb->KCaib', trs, mud[k], deta_tildes[k],   nu[k],  **kw)
                   + contract('trs,rWCa,tWKCi,sWCb->KCaib', trs, mu[k],  deta_tildes_d[k], nu[k],  **kw)
                   + contract('trs,rWCa,tWCi,sWKCb->KCaib', trs, mu[k],  deta_tildes[k],   nud[k], **kw))
    return tuple(out)


def tt_tangent_assembly_factored_trs(sigma_tildes, sigma_tildes_d, tau_tildes, tau_tildes_d, deta_tildes,
                                     deta_tildes_d, xi, xid, mu, mud, nu, nud, trs, n_frame):
    """The factored dense-trs TT tangent assembly (W summed) -- the per-core kernel the two-hole shipped with in
    S3, before S3b made every production kernel a recurrence: each group's undotted pair and its dotted twin
    formed once with the binomial weights folded into one factor (``M[t,s] = sum_r trs[t,r,s] mu^(r)``,
    ``N[r,s] = sum_t trs[t,r,s] tau~^(t)``), the tau~ and deta~ groups sharing their remaining factor ``nu``.
    A second reference beside the nine-term :py:func:`tt_tangent_assembly_trs`."""
    s2 = min(2, trs.shape[0])
    t_tau = trs[:, :s2, :]
    nW = xi[0].ndim - 2 - n_frame
    out = []
    for k in range(len(xi)):
        xs, xsd = xi[k][:s2], xid[k][:s2]
        M = contract('trs,rWCa->tsWCa', trs, mu[k])
        Md = contract('trs,rWKCa->tsWKCa', trs, mud[k], len_W=nW)
        N = contract('trs,tWCa->rsWCa', t_tau, tau_tildes[k])
        Nd = contract('trs,tWKCa->rsWKCa', t_tau, tau_tildes_d[k], len_W=nW)
        P = contract('tsWCa,sWCi->tWCai', M[:, :s2], xs)
        Pd = (contract('tsWKCa,sWCi->tWKCai', Md[:, :s2], xs, len_W=nW)
              + contract('tsWCa,sWKCi->tWKCai', M[:, :s2], xsd, len_W=nW))
        Q = contract('rsWCa,rWCi->sWCai', N, xs) + contract('tsWCa,tWCi->sWCai', M, deta_tildes[k])
        Qd = (contract('rsWKCa,rWCi->sWKCai', Nd, xs, len_W=nW)
              + contract('rsWCa,rWKCi->sWKCai', N, xsd, len_W=nW)
              + contract('tsWKCa,tWCi->sWKCai', Md, deta_tildes[k], len_W=nW)
              + contract('tsWCa,tWKCi->sWKCai', M, deta_tildes_d[k], len_W=nW))
        out.append(contract('tWKCai,tWCb->KCaib', Pd, sigma_tildes[k], len_W=nW)
                   + contract('tWCai,tWKCb->KCaib', P, sigma_tildes_d[k], len_W=nW)
                   + contract('sWKCai,sWCb->KCaib', Qd, nu[k], len_W=nW)
                   + contract('sWCai,sWKCb->KCaib', Q, nud[k], len_W=nW))
    return tuple(out)


def probe_derivatives_two_hole_states_trs(ztildes, ww, pp, core_pair, core_directions, order, frame_sweep=None):
    """Every intermediate of the dense-trs two-hole, as a dict (the per-piece twins' inputs and outputs).

    The undotted state is the corewise frame's (xi, mu reused from ``frame_sweep`` when given, else built
    from ``core_pair``; nu re-swept through G), with the undotted tildes; the dotted forward is the GN
    tangent forward at ``Q = O = P = G`` along ``(Ud, Gd)``. Every jet kernel here is a dense ``*_trs`` twin."""
    U, G = core_pair
    Ud, Gd = core_directions
    n_frame = U[0].ndim - 2
    trs = twins.binomial_combine_tensor(order)
    if frame_sweep is None:
        xi = sd.build_input_jets(probing.compute_xi(U, ww), probing.compute_xi(U, pp))
        mu = twins.compute_mu_jets_trs(G, xi, trs)
    else:
        xi, mu = frame_sweep[0], frame_sweep[1]
    st = dict(trs=trs, n_frame=n_frame, xi=xi, mu=mu)
    st['nu'] = twins.compute_nu_jets_trs(G, xi, trs)
    st['deta_t'] = sd.compute_deta_tilde_jets(U, ztildes)
    st['tau_t'] = twins.compute_tau_tilde_jets_trs(G, xi, st['deta_t'], mu, trs)
    st['sigma_t'] = twins.compute_sigma_tilde_jets_trs(G, xi, st['deta_t'], st['nu'], trs)
    st['xid'] = sd.build_input_jets(probing.compute_xi(Ud, ww), probing.compute_xi(Ud, pp))
    st['mud'] = twins.compute_sigma_jets_trs(Gd, G, G, xi, st['xid'], mu, trs)
    st['nud'] = twins.compute_tau_jets_trs(Gd, G, G, xi, st['xid'], st['nu'], trs)
    st['etad'] = twins.compute_deta_jets_trs(Gd, G, G, mu, st['nu'], st['mud'], st['nud'], trs)
    st['deta_td'] = deta_tilde_tangent_jets_trs(Ud, ztildes, n_frame)
    st['tau_td'] = tau_tilde_tangent_jets_trs(G, Gd, xi, st['xid'], st['deta_t'], st['deta_td'], mu, st['mud'],
                                              st['tau_t'], trs)
    st['sigma_td'] = sigma_tilde_tangent_jets_trs(G, Gd, xi, st['xid'], st['deta_t'], st['deta_td'], st['nu'],
                                                  st['nud'], st['sigma_t'], trs)
    st['dxi_td'] = dxi_tilde_tangent_jets_trs(G, Gd, mu, st['mud'], st['nu'], st['nud'], st['sigma_t'],
                                              st['sigma_td'], st['tau_t'], st['tau_td'], trs)
    st['dU'] = tucker_tangent_assembly_trs(ztildes, st['dxi_td'], ww, pp, st['etad'], n_frame)
    st['dG'] = tt_tangent_assembly_trs(st['sigma_t'], st['sigma_td'], st['tau_t'], st['tau_td'], st['deta_t'],
                                       st['deta_td'], xi, st['xid'], mu, st['mud'], st['nu'], st['nud'], trs,
                                       n_frame)
    return st


def tt_tangent_assembly_nine_term(sigma_tildes, sigma_tildes_d, tau_tildes, tau_tildes_d, deta_tildes, deta_tildes_d,
                                  xi, xid, mu, mud, nu, nud, n_probe):
    """Reference for ``probing.assemble_tt_tangent_variations`` (the plain probe two-hole's TT assembly, W summed):
    the nine-term product rule of ``probing.assemble_tt_variations``' three trilinear outer products, one direct
    three-operand contraction per (group, dotted factor) -- the unfactored form the shipped kernel groups into four
    finals. (No binomial structure: plain probing has no order axis, so this is a reference for the factoring, not a
    dense-``trs`` twin.)"""
    kw = dict(len_W=n_probe)
    trilinear = lambda a, b, c, k: contract(('WKCi,WCa,WCj->KCiaj', 'WCi,WKCa,WCj->KCiaj', 'WCi,WCa,WKCj->KCiaj')[k],
                                            a, b, c, **kw)
    out = []
    for k in range(len(xi)):
        out.append(trilinear(mud[k], xi[k], sigma_tildes[k], 0) + trilinear(mu[k], xid[k], sigma_tildes[k], 1)
                   + trilinear(mu[k], xi[k], sigma_tildes_d[k], 2)
                   + trilinear(tau_tildes_d[k], xi[k], nu[k], 0) + trilinear(tau_tildes[k], xid[k], nu[k], 1)
                   + trilinear(tau_tildes[k], xi[k], nud[k], 2)
                   + trilinear(mud[k], deta_tildes[k], nu[k], 0) + trilinear(mu[k], deta_tildes_d[k], nu[k], 1)
                   + trilinear(mu[k], deta_tildes[k], nud[k], 2))
    return tuple(out)


def apply_tt_tangent_assembly_three_term(mu, mud, xi, xid, sigma_hat, sigma_hat_d, n_probe):
    """Reference for the apply / entries two-hole's TT assembly (W summed): the three-term product rule of the
    transpose's single trilinear ``mu (x) xi (x) sigma_hat``, unfactored."""
    kw = dict(len_W=n_probe)
    return tuple(contract('WKCa,WCi,WCb->KCaib', md, x, s, **kw) + contract('WCa,WKCi,WCb->KCaib', m, xd, s, **kw)
                 + contract('WCa,WCi,WKCb->KCaib', m, x, sd, **kw)
                 for m, md, x, xd, s, sd in zip(mu, mud, xi, xid, sigma_hat, sigma_hat_d))


def probe_derivatives_two_hole_trs(ztildes, ww, pp, core_pair, core_directions, order, frame_sweep=None):
    """The dense-trs reference twin of ``sd.t3_probe_corewise_derivatives_hessian[_from_sweep]`` (ragged):
    ``K_z theta_hat = d/ds grad_theta <z~, S(Phi(theta + s theta_hat))>``, W summed, raw K + C."""
    st = probe_derivatives_two_hole_states_trs(ztildes, ww, pp, core_pair, core_directions, order, frame_sweep)
    return st['dU'], st['dG']
