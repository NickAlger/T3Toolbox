# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Test-only oracles for the exact Riemannian Hessian's building blocks (the lift, the two-hole sweeps).

Two oracles, each independent of the kernel it checks:

* **The block-sum oracle.** For a function ``F(theta) = <c, S(Phi(theta))>`` that is LINEAR in the
  tensor (a sampling transpose, or an inner product with a fixed ``Z``), the core-coordinate Hessian
  applied to ``theta_hat`` is the gradient of ``theta -> dF(theta)[theta_hat]``. Since
  ``dPhi(theta)[theta_hat] = Phi(Y)`` for the doubled-rank T3 ``Y = tv_to_t3(t3_corewise_frame(theta),
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

All functions take unstacked data unless stated; the tests loop over stack elements. Every random draw
is the caller's (a seeded ``np.random.Generator``).
"""
import numpy as np

import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.tv_operations as tv


def doubled(theta, theta_hat):
    """The doubled-rank T3 ``Y`` with ``Phi(Y) = dPhi(theta)[theta_hat]`` (stack: ``theta_hat``'s)."""
    return tv.tv_to_t3(fvc.t3_corewise_frame(theta), theta_hat)


def blocksum(gY, theta):
    """Read a gradient with respect to the doubled-rank ``Y = doubled(theta, .)`` back onto ``theta``'s
    cores: the transpose of the (linear-in-``theta``) embedding ``theta -> Y``. Stack-aware (leading axes
    of ``gY`` ride along); the block offsets come from ``theta``'s core shapes."""
    gYB, gYG = gY
    B, G = theta
    d = len(B)
    gB = tuple(gYB[i][..., :B[i].shape[-2], :] for i in range(d))      # B_i sits in the top Tucker rows
    if d == 1:                                                         # Y_0 = [Gdot_0 ; G_0] on the n axis
        n = B[0].shape[-2]
        return gB, (gYG[0][..., :, n:, :],)
    gG = []
    for i in range(d):
        r, n, rn = G[i].shape[-3], G[i].shape[-2], G[i].shape[-1]
        g = gYG[i]
        if i == 0:                                       # [[Gdot, G], [G, 0]]: top-right and bottom-left
            gG.append(g[..., :, :n, rn:] + g[..., :, n:, :rn])
        elif i == d - 1:                                 # [[R ; Gdot], [0 ; U]]: top-first and bottom-second
            gG.append(g[..., :r, :n, :] + g[..., r:, n:, :])
        else:                                            # R (top-left), L (top, second-second), U (bottom)
            gG.append(g[..., :r, :n, :rn] + g[..., r:, :n, rn:] + g[..., r:, n:, :rn])
    return gB, tuple(gG)


def _blocks(theta):
    """theta = (B, G) as a flat list of 2d core arrays (Tucker first)."""
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
    the Riemannian gradient ``P_X A^T r`` this is ``(dPhi)^T Hess f (dPhi)`` (the vertical directions are in
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
    ``Bdot_g = pinv(M_g) [V_i1; ...]``, copied to the group's modes; the per-mode and bond lifts elsewhere -- the
    lift of the tied parametrization ``Phi(T theta_sh)``, built from dense pseudo-inverses (no SVD of the shared
    companion, no ``tv_corewise_lift``)."""
    group_pinvs, mode_pinvs, bond_pinvs = _tied_lift_matrices(frame, groups)
    V, H = v
    d = len(V)
    Bd = [mode_pinvs[i] @ np.asarray(V[i]) for i in range(d)]
    for g, Mp, _ in group_pinvs:
        Bg = Mp @ np.concatenate([np.asarray(V[i]) for i in g], axis=0)
        for i in g:
            Bd[i] = Bg
    Gd = [np.einsum('aib,bc->aic', np.asarray(H[i]), bond_pinvs[i]) for i in range(d - 1)] + [np.asarray(H[-1])]
    return tuple(Bd), tuple(Gd)


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
    """A random raw direction along the gauge orbit of ``theta = (B, G)``: ``dPhi(theta)[result] = 0``
    exactly (to first order). Tucker gauge ``B_i -> A B_i``, ``G_i -> G_i x_2 A^{-T}``; TT gauge
    ``G_i -> G_i M``, ``G_{i+1} -> M^{-1} G_{i+1}``. Unstacked."""
    B, G = theta
    d = len(B)
    dB = [np.zeros(np.shape(b)) for b in B]
    dG = [np.zeros(np.shape(g)) for g in G]
    for i in range(d):
        n = B[i].shape[-2]
        Ai = rng.standard_normal((n, n))
        dB[i] += Ai @ B[i]
        dG[i] -= np.einsum('aib,ij->ajb', G[i], Ai)
    for i in range(d - 1):
        r = G[i].shape[-1]
        M = rng.standard_normal((r, r))
        dG[i] += np.einsum('aib,bc->aic', G[i], M)
        dG[i + 1] -= np.einsum('bc,cid->bid', M, G[i + 1])
    return tuple(dB), tuple(dG)

