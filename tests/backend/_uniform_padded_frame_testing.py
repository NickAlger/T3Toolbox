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
    """A ragged frame's four stores (nD, nU, rL, rR), read off its core shapes."""
    down, up, left, right = frame
    return (tuple(u.shape[-2] for u in down), tuple(o.shape[-2] for o in up),
            tuple(g.shape[-3] for g in left) + (left[-1].shape[-1],),
            tuple(g.shape[-3] for g in right) + (right[-1].shape[-1],))


def scatter_frame(rng, F, widths, old_stores=None, groups=(), gappy=True, garbage=True):
    """Scatter an unstacked ragged frame ``F`` into uniform frame ``.data``.

    ``old_stores=None``: ``F`` is an old frame, scattered at its own widths ``widths = (nD, nU, rL, rR, N)`` (they may
    all differ). ``old_stores`` given: ``F`` is a padded frame whose leading slots hold those stores, scattered in BLOCK
    LAYOUT, ``widths = (Na, Ka, Rb, Kb, N)``: Tucker slots ``[old Na | completion Ka]``, bond slots ``[old Rb | Kb]``.
    Real slots sit at random sorted positions inside each block (``gappy``); junk entries hold random garbage. With
    ``groups``, a group's members share their down positions and supercore slice (a tied factor).
    Returns (frame ``.data``, positions (pD, pU, pL, pR), each a list of index arrays)."""
    D, U, L, R = F
    d = len(D)
    nD, nU, rL, rR = stores_of(F)
    Nsh = tuple(Di.shape[-1] for Di in D)
    if old_stores is None:
        wD, wU, wL, wR, N = widths
        pos = lambda counts, w: [_pick(rng, w, c, gappy) for c in counts]
        pD, pU, pL, pR = pos(nD, wD), pos(nU, wU), pos(rL, wL), pos(rR, wR)
    else:
        Na, Ka, Rb, Kb, N = widths
        nD0, nU0, rL0, rR0 = old_stores
        pos = lambda olds, counts, wa, wk: [np.concatenate([_pick(rng, wa, o, gappy), wa + _pick(rng, wk, c - o, gappy)])
                                            for o, c in zip(olds, counts)]
        pD, pU = pos(nD0, nD, Na, Ka), pos(nU0, nU, Na, Ka)
        pL, pR = pos(rL0, rL, Rb, Kb), pos(rR0, rR, Rb, Kb)
        wD = wU = Na + Ka
        wL = wR = Rb + Kb
    for g in groups:
        for i in g:
            pD[i] = pD[min(g)]
    junk = (lambda *s: rng.standard_normal(s)) if garbage else (lambda *s: np.zeros(s))
    Du, Uu, Lu, Ru = junk(d, wD, N), junk(d, wL, wU, wR), junk(d, wL, wD, wL), junk(d, wR, wD, wR)
    mD, mU = np.zeros((d, wD), bool), np.zeros((d, wU), bool)
    mL, mR = np.zeros((d + 1, wL), bool), np.zeros((d + 1, wR), bool)
    for j in range(d + 1):
        mL[j, pL[j]] = True
        mR[j, pR[j]] = True
    for i in range(d):
        mD[i, pD[i]] = True
        mU[i, pU[i]] = True
        Du[i][np.ix_(pD[i], np.arange(Nsh[i]))] = D[i]
        Uu[i][np.ix_(pL[i], pU[i], pR[i + 1])] = U[i]
        Lu[i][np.ix_(pL[i], pD[i], pL[i + 1])] = L[i]
        Ru[i][np.ix_(pR[i], pD[i], pR[i + 1])] = R[i]
    for g in groups:
        for i in g:
            Du[i] = Du[min(g)]
    return (Du, Uu, Lu, Ru, Nsh, (mD, mU, mL, mR)), (pD, pU, pL, pR)


def scatter_variations(rng, variations, frame, positions, garbage=True):
    """Scatter unstacked ragged variations ``(V, H)`` into the uniform layout of ``frame`` (a scattered frame's ``.data``)
    at that scatter's ``positions``: V on the up slots and the shape prefix, H on (left, down, right); finite garbage in
    every junk entry (or zeros). Returns the variations ``.data`` (V, H, shape, the frame's variation masks)."""
    V, H = variations
    pD, pU, pL, pR = positions
    d = len(V)
    wD, N, wL, wR = frame[0].shape[-2], frame[0].shape[-1], frame[2].shape[-1], frame[3].shape[-1]
    junk = (lambda *s: rng.standard_normal(s)) if garbage else (lambda *s: np.zeros(s))
    Vu, Hu = junk(d, wD, N), junk(d, wL, wD, wR)
    for i in range(d):
        Vu[i][np.ix_(pU[i], np.arange(V[i].shape[-1]))] = V[i]
        Hu[i][np.ix_(pL[i], pD[i], pR[i + 1])] = H[i]
    mD, mU, mL, mR = frame[5]
    return Vu, Hu, frame[4], (mD, mU, mL[:-1], mR[1:])


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
    D, U, L, R = [np.asarray(a) for a in frame[:4]]
    shape, (mD, mU, mL, mR) = frame[4], frame[5]
    d, N = D.shape[0], D.shape[-1]
    mN = (np.arange(N)[None, :] < np.asarray(shape)[:, None]).reshape((d,) + (1,) * (D.ndim - 3) + (N,))
    a3 = lambda m: m[..., :, None, None]
    b3 = lambda m: m[..., None, :, None]
    c3 = lambda m: m[..., None, None, :]
    real = ((D, mD[..., :, None] & mN[..., None, :]), (U, a3(mL[:-1]) & b3(mU) & c3(mR[1:])),
            (L, a3(mL[:-1]) & b3(mD) & c3(mL[1:])), (R, a3(mR[:-1]) & b3(mD) & c3(mR[1:])))
    return all(np.all(A[~np.broadcast_to(m, A.shape)] == 0.0) for A, m in real)
