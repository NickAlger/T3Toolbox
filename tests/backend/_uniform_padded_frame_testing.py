# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Test helpers for the uniform padded frames: scatter ragged frames into uniform layouts with GAPPY masks and FINITE
GARBAGE in every junk entry, read the real parts back, and check that a uniform frame's junk is exactly zero.

Ported from the research prototype ``uniform_testing.py`` (maintainer-local research repo). The garbage is written
into arrays of the clean layout's own shape and order (``testing_strategy.md``: bitwise invariance needs equal
layouts).
"""
import numpy as np

import t3toolbox.backend.ufv_conversions as ufc


def _pick(rng, width, count, gappy):
    """``count`` sorted slot positions in ``range(width)``: random when gappy, the prefix otherwise."""
    return np.sort(rng.choice(width, count, replace=False)) if gappy else np.arange(count)


def stores_of(frame):
    """A ragged frame's four stores (nU, nD, rL, rR), read off its core shapes."""
    up, down, left, right = frame
    return (tuple(u.shape[-2] for u in up), tuple(o.shape[-2] for o in down),
            tuple(g.shape[-3] for g in left) + (left[-1].shape[-1],),
            tuple(g.shape[-3] for g in right) + (right[-1].shape[-1],))


def scatter_frame(rng, F, widths, old_stores=None, groups=(), gappy=True, garbage=True):
    """Scatter an unstacked ragged frame ``F`` into uniform frame ``.data``.

    ``old_stores=None``: ``F`` is an old frame, scattered at its own widths ``widths = (nU, nD, rL, rR, N)`` (they may
    all differ). ``old_stores`` given: ``F`` is a padded frame whose leading slots hold those stores, scattered in BLOCK
    LAYOUT, ``widths = (Na, Ka, Rb, Kb, N)``: Tucker slots ``[old Na | completion Ka]``, bond slots ``[old Rb | Kb]``.
    Real slots sit at random sorted positions inside each block (``gappy``); junk entries hold random garbage. With
    ``groups``, a group's members share their up positions and supercore slice (a tied factor).
    Returns (frame ``.data``, positions (pU, pD, pL, pR), each a list of index arrays)."""
    U, O, L, R = F
    d = len(U)
    nU, nD, rL, rR = stores_of(F)
    Nsh = tuple(u.shape[-1] for u in U)
    if old_stores is None:
        wU, wD, wL, wR, N = widths
        pos = lambda counts, w: [_pick(rng, w, c, gappy) for c in counts]
        pU, pD, pL, pR = pos(nU, wU), pos(nD, wD), pos(rL, wL), pos(rR, wR)
    else:
        Na, Ka, Rb, Kb, N = widths
        nU0, nD0, rL0, rR0 = old_stores
        pos = lambda olds, counts, wa, wk: [np.concatenate([_pick(rng, wa, o, gappy), wa + _pick(rng, wk, c - o, gappy)])
                                            for o, c in zip(olds, counts)]
        pU, pD = pos(nU0, nU, Na, Ka), pos(nD0, nD, Na, Ka)
        pL, pR = pos(rL0, rL, Rb, Kb), pos(rR0, rR, Rb, Kb)
        wU = wD = Na + Ka
        wL = wR = Rb + Kb
    for g in groups:
        for i in g:
            pU[i] = pU[min(g)]
    junk = (lambda *s: rng.standard_normal(s)) if garbage else (lambda *s: np.zeros(s))
    Uu, Ou, Lu, Ru = junk(d, wU, N), junk(d, wL, wD, wR), junk(d, wL, wU, wL), junk(d, wR, wU, wR)
    mU, mD = np.zeros((d, wU), bool), np.zeros((d, wD), bool)
    mL, mR = np.zeros((d + 1, wL), bool), np.zeros((d + 1, wR), bool)
    for j in range(d + 1):
        mL[j, pL[j]] = True
        mR[j, pR[j]] = True
    for i in range(d):
        mU[i, pU[i]] = True
        mD[i, pD[i]] = True
        Uu[i][np.ix_(pU[i], np.arange(Nsh[i]))] = U[i]
        Ou[i][np.ix_(pL[i], pD[i], pR[i + 1])] = O[i]
        Lu[i][np.ix_(pL[i], pU[i], pL[i + 1])] = L[i]
        Ru[i][np.ix_(pR[i], pU[i], pR[i + 1])] = R[i]
    for g in groups:
        for i in g:
            Uu[i] = Uu[min(g)]
    return (Uu, Ou, Lu, Ru, Nsh, (mU, mD, mL, mR)), (pU, pD, pL, pR)


def scatter_variations(rng, variations, frame, positions, garbage=True):
    """Scatter unstacked ragged variations ``(V, H)`` into the uniform layout of ``frame`` (a scattered frame's ``.data``)
    at that scatter's ``positions``: V on the down slots and the shape prefix, H on (left, up, right); finite garbage in
    every junk entry (or zeros). Returns the variations ``.data`` (V, H, shape, the frame's variation masks)."""
    V, H = variations
    pU, pD, pL, pR = positions
    d = len(V)
    wU, N, wL, wR = frame[0].shape[-2], frame[0].shape[-1], frame[2].shape[-1], frame[3].shape[-1]
    junk = (lambda *s: rng.standard_normal(s)) if garbage else (lambda *s: np.zeros(s))
    Vu, Hu = junk(d, wU, N), junk(d, wL, wU, wR)
    for i in range(d):
        Vu[i][np.ix_(pD[i], np.arange(V[i].shape[-1]))] = V[i]
        Hu[i][np.ix_(pL[i], pU[i], pR[i + 1])] = H[i]
    mU, mD, mL, mR = frame[5]
    return Vu, Hu, frame[4], (mU, mD, mL[:-1], mR[1:])


def stack_frames(items):
    """Stack uniform frame ``.data`` elements (equal widths, any masks) along a new stack axis (axis 1)."""
    supercores = tuple(np.stack([np.asarray(it[k]) for it in items], axis=1) for k in range(4))
    masks = tuple(np.stack([it[5][k] for it in items], axis=1) for k in range(4))
    return supercores + (items[0][4], masks)


def element(frame, e):
    """Stack element ``e`` (an index tuple, or an int for one stack axis) of a stacked uniform frame ``.data``."""
    e = e if isinstance(e, tuple) else (e,)
    return tuple(np.asarray(a)[(slice(None),) + e] for a in frame[:4]) + (frame[4], tuple(m[(slice(None),) + e]
                                                                                         for m in frame[5]))


def ragged(frame):
    """The real parts of an unstacked uniform frame (any mask pattern), in mask order."""
    return ufc.ut3frame_to_t3frame(tuple(np.asarray(a) for a in frame[:4]) + tuple(frame[4:]))


def junk_is_zero(frame):
    """True when every entry of a uniform frame outside its real slots (its own masks and the shape prefix) is
    exactly 0.0."""
    U, O, L, R = [np.asarray(a) for a in frame[:4]]
    shape, (mU, mD, mL, mR) = frame[4], frame[5]
    d, N = U.shape[0], U.shape[-1]
    mN = (np.arange(N)[None, :] < np.asarray(shape)[:, None]).reshape((d,) + (1,) * (U.ndim - 3) + (N,))
    a3 = lambda m: m[..., :, None, None]
    b3 = lambda m: m[..., None, :, None]
    c3 = lambda m: m[..., None, None, :]
    real = ((U, mU[..., :, None] & mN[..., None, :]), (O, a3(mL[:-1]) & b3(mD) & c3(mR[1:])),
            (L, a3(mL[:-1]) & b3(mU) & c3(mL[1:])), (R, a3(mR[:-1]) & b3(mU) & c3(mR[1:])))
    return all(np.all(A[~np.broadcast_to(m, A.shape)] == 0.0) for A, m in real)
