# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Dense reference for the padded-frame alignment step (tests only; tiny sizes).

The step is written here directly on dense tensors, one DIRECTED edge at a time, with no code shared with
``backend.padded_frames``: per edge ``a -> b``, in message order,

    P+ = Q . (leading left singular vectors of  Q^T Z_(a->b) (I - V V^T)),

with ``Q`` an orthonormal basis of the candidate space built from the NEW incoming completions, minus the edge's old
frame, and ``V`` the old frame on the far side. On the core -> leaf edges the content is the hint with the old Tucker
frame removed (the unrooted rule). A store not in ``update`` is TRANSPORTED: its old completion projected into the new
candidate space, ``Q . (leading left singular vectors of Q^T P_old)``. The movement of an edge is
``||T - W W^T T||_F`` with ``T = Q^T P_old`` and ``W`` the new completion in ``Q``'s coordinates.

Ported from the research prototypes (``dense_oracle.py`` and ``cycle/cycle_schemes.select_pass`` in the
maintainer-local research repo), with the movement added.
"""
import math

import numpy as np


def interfaces(frame):
    """Dense left interfaces ``phi[k]`` (prod N_<k, rL_k) and right interfaces ``psi[k]`` (prod N_>=k, rR_k)."""
    down, up, left, right = frame
    d = len(down)
    phi = [np.ones((1, 1))]
    for k in range(d - 1):
        phi.append(np.einsum('xa,ni,anb->xib', phi[k], down[k], left[k])
                   .reshape(phi[k].shape[0] * down[k].shape[1], left[k].shape[2]))
    psi = [None] * d + [np.ones((1, 1))]
    for k in range(d - 1, 0, -1):
        psi[k] = (np.einsum('anb,ni,yb->aiy', right[k], down[k], psi[k + 1])
                  .reshape(right[k].shape[0], down[k].shape[1] * psi[k + 1].shape[0]).T)
    return phi, psi


def dense_frames(frame):
    """One orthonormal dense basis per DIRECTED edge: ``(D, phi, psi, U)`` = leaf -> core, core -> core rightward,
    core -> core leftward, core -> leaf."""
    down, up, left, right = (tuple(np.asarray(c, dtype=float) for c in cores) for cores in frame)
    phi, psi = interfaces((down, up, left, right))
    D = [u.T for u in down]
    U = [np.einsum('xa,amb,yb->xym', phi[i], up[i], psi[i + 1])
         .reshape(phi[i].shape[0] * psi[i + 1].shape[0], up[i].shape[1]) for i in range(len(down))]
    return D, phi, psi, U


def complement_in(B, A):
    """An orthonormal basis of span(B) minus span(A), for A nested in B (both orthonormal columns)."""
    c = B.T @ A
    Uc, sc, _ = np.linalg.svd(c, full_matrices=True)
    assert sc.size == 0 or sc.min() > 1 - 1e-8, 'the old frame is not nested in the candidate space'
    return B @ Uc[:, A.shape[1]:]


def top_left(Y, k):
    """The leading ``k`` left singular vectors of ``Y``."""
    Uy, _, _ = np.linalg.svd(Y, full_matrices=True)
    return Uy[:, :k]


def dense_pass(old_dense, cur_dense, Z, stores, update=('D', 'U', 'L', 'R'), groups=()):
    """One step on dense tensors: the new frames ``(D, phi, psi, U)`` at ``stores`` = (nD, nU, rL, rR), and the
    movement (max over edges). ``old_dense`` / ``cur_dense``: the dense frames of the old and the current (padded)
    frame; ``Z``: the dense tangent vector; ``groups``: canonical sharing groups (tuples of mode indices), singletons
    optional."""
    Do, phio, psio, Uo = old_dense
    Dc, phic, psic, Uc = cur_dense
    nD, nU, rL, rR = stores
    N = Z.shape
    d = len(N)
    Zi = [np.moveaxis(Z, i, 0).reshape(N[i], -1) for i in range(d)]
    moves = []

    def choose(Q, content, P_old, k, on):
        T = Q.T @ P_old
        W = top_left(content if on else T, k)
        moves.append(np.linalg.norm(T - W @ (W.T @ T)))
        return Q @ W

    D = [None] * d
    grouped = set(i for g in groups for i in g)
    for g in [tuple(g) for g in groups] + [(i,) for i in range(d) if i not in grouped]:      # leaf -> core
        i0 = g[0]
        s = Do[i0].shape[1]
        Q = complement_in(np.eye(N[i0]), Do[i0])
        blocks = []
        for i in g:
            Y = Q.T @ Zi[i]
            blocks.append(Y - (Y @ Uo[i]) @ Uo[i].T)
        W = choose(Q, np.concatenate(blocks, axis=1), Dc[i0][:, s:], nD[i0] - s, 'D' in update)
        for i in g:
            D[i] = np.concatenate([Do[i0], W], axis=1)
    phi = [np.ones((1, 1))]
    for k in range(d - 1):                                        # core k -> core k+1
        B = np.einsum('xa,in->xian', phi[k], D[k]).reshape(phi[k].shape[0] * N[k], -1)
        r0 = phio[k + 1].shape[1]
        Q = complement_in(B, phio[k + 1])
        Y = Q.T @ Z.reshape(B.shape[0], -1)
        Y = Y - (Y @ psio[k + 1]) @ psio[k + 1].T
        phi.append(np.concatenate([phio[k + 1], choose(Q, Y, phic[k + 1][:, r0:], rL[k + 1] - r0, 'L' in update)], 1))
    psi = [None] * d + [np.ones((1, 1))]
    for k in range(d - 1, 0, -1):                                 # core k -> core k-1
        B = np.einsum('in,yb->iynb', D[k], psi[k + 1]).reshape(N[k] * psi[k + 1].shape[0], -1)
        r0 = psio[k].shape[1]
        Q = complement_in(B, psio[k])
        Y = Q.T @ Z.reshape(-1, B.shape[0]).T
        Y = Y - (Y @ phio[k]) @ phio[k].T
        psi[k] = np.concatenate([psio[k], choose(Q, Y, psic[k][:, r0:], rR[k] - r0, 'R' in update)], 1)
    U = []
    for i in range(d):                                            # core -> leaf (unrooted)
        B = np.einsum('xa,yb->xyab', phi[i], psi[i + 1]).reshape(phi[i].shape[0] * psi[i + 1].shape[0], -1)
        r0 = Uo[i].shape[1]
        Q = complement_in(B, Uo[i])
        Zt = Zi[i].T
        content = Q.T @ (Zt - (Zt @ Do[i]) @ Do[i].T)
        U.append(np.concatenate([Uo[i], choose(Q, content, Uc[i][:, r0:], nU[i] - r0, 'U' in update)], 1))
    return (D, phi, psi, U), max(moves)


def dense_loop(old_dense, start_dense, S, working, target, oversampled_passes, max_passes, tol, alternate, groups=()):
    """The alignment loop's schedule (the plan's section 2.3) on the dense step: from the dense start ``startD`` at the
    ``working`` stores, ``n_over`` passes at the working stores, the truncating pass to the ``target`` stores, then
    (``alternate``) (('D', 'L'), ('U', 'R')) pairs while a whole pair fits under ``max_passes``, stopping after a
    pair whose two movements are below ``tol``, and one final ('D', 'L') pass in a remaining slot unless stopped; or
    (not ``alternate``) all-store passes, stopping after the first pass from the truncating one on whose movement is
    below ``tol``. Every pass is against ``dense_project(D, S)``. Returns the dense frames and the movement after
    every pass."""
    n_over = 0 if tuple(working) == tuple(target) else min(oversampled_passes, max_passes - 1)
    state = {'D': start_dense}
    frames, moves = [], []

    def run(stores, update):
        state['D'], movement = dense_pass(old_dense, state['D'], dense_project(state['D'], S), stores, update, groups)
        frames.append(state['D'])
        moves.append(movement)

    for it in range(n_over + 1):
        run(working if it < n_over else target, ('D', 'U', 'L', 'R'))
    if alternate:
        stop = False
        while not stop and len(moves) + 2 <= max_passes:
            run(target, ('D', 'L'))
            run(target, ('U', 'R'))
            stop = max(moves[-2:]) < tol
        if not stop and len(moves) < max_passes:
            run(target, ('D', 'L'))
    else:
        while not moves[-1] < tol and len(moves) < max_passes:
            run(target, ('D', 'U', 'L', 'R'))
    return frames, moves


def tangent_basis(dense, N):
    """An orthonormal basis of the two-sided tangent space at the dense frames ``dense``."""
    D, phi, psi, U = dense
    d = len(N)
    cols = []
    for i in range(d):
        nx, ny, m = math.prod(N[:i]), math.prod(N[i + 1:]), U[i].shape[1]
        cols.append(np.einsum('ij,xym->xiyjm', np.eye(N[i]), U[i].reshape(nx, ny, m)).reshape(math.prod(N), -1))
        cols.append(np.einsum('xa,in,yb->xiyanb', phi[i], D[i], psi[i + 1]).reshape(math.prod(N), -1))
    Q, s, _ = np.linalg.svd(np.concatenate(cols, axis=1), full_matrices=False)
    return Q[:, s > 1e-9 * s[0]]


def dense_project(dense, S):
    """The orthogonal projection of the dense tensor ``S`` onto the tangent space at ``dense``."""
    Q = tangent_basis(dense, S.shape)
    return (Q @ (Q.T @ S.ravel())).reshape(S.shape)


def distances(dense_a, dense_b):
    """The largest principal-angle sine between two sets of dense frames, per kind of directed edge
    ('D' leaf -> core, 'L' rightward, 'R' leftward, 'U' core -> leaf); 1.0 where the widths differ."""
    out = {}
    for name, a, b in zip(('D', 'L', 'R', 'U'), dense_a, dense_b):
        ds = []
        for x, y in zip(a, b):
            if x is None or y is None:
                continue
            ds.append(1.0 if x.shape != y.shape else (np.linalg.norm(x - y @ (y.T @ x), 2) if x.size else 0.0))
        out[name] = max(ds) if ds else 0.0
    return out
