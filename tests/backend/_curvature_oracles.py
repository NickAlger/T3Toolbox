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

All functions take unstacked data unless stated; the tests loop over stack elements. Every random draw
is the caller's (a seeded ``np.random.Generator``).
"""
import numpy as np

import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.tv_operations as tv


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
