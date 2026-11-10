# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The padded frame at a zero-padded rank increase, on uniform frame data, in BLOCK LAYOUT.

Every rank slot of a padded frame is ``[A-block | P-block]``: the old frame's supercores as given (any mask pattern,
junk masked to zero once), then the completion, each padded to one width with its own mask. Tucker slots are
``[Na | Ka]`` and bond slots ``[Rb | Kb]``, so the only positional fact is the split ``(Na, Rb)``.

``ufv_empty_frame`` is the frame of the zero tensor (``Na = 0``, ``Rb = 1``). ``ufv_padded_frame`` is the seeded start,
the uniform twin of ``padded_frames.fv_padded_frame``: the same seeds (every store's slot-indexed rows of its keyed
Gaussian stream, generated on the host at the ragged unfolding's size and placed at the real rows in their ragged
row-major order), completed by ``linalg.pad_safe_pinned_completion``, one batched completion per store. ``utv_align_padded_frame`` is one alignment
step, the uniform twin of ``padded_frames.tv_align_padded_frame``: the maps over modes are batched contractions, the
sweeps scans with module-level bodies, and every edge's completion one ``linalg.pad_safe_svd``. The host prepares the
masks, selections and group tables; the arithmetic is one jittable kernel, run eagerly here.
``ufv_compact_frame`` gathers a frame junk-last at a level's widths. ``ufv_aligned_padded_frame`` is the loop, the
twin of ``padded_frames.fv_aligned_padded_frame`` (the same schedule and pair stop rule), compacted at exit; it runs
eagerly, or under ``use_jit`` as ONE ``jax.jit`` of the whole loop.
"""
import functools as ft
import typing as typ
import numpy as np

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.sharing as sharing_module
import t3toolbox.backend.ufv_masking as ufv_masking
from t3toolbox.backend.common import *
from t3toolbox.backend.fv_operations import ZERO_FRAME_SEED
from t3toolbox.backend.padded_frames import (ALIGN_OVERSAMPLING, ALIGN_OVERSAMPLED_PASSES, ALIGN_MAX_PASSES,
                                             ALIGN_TOL, ALIGN_ALTERNATE, _seed_rows, _unfold, _fold, _root,
                                             _new_coordinates, _edge_content, _left_message_step,
                                             _right_message_step, _jitted)

__all__ = [
    'ufv_empty_frame',
    'ufv_padded_frame',
    'utv_align_padded_frame',
    'ufv_compact_frame',
    'ufv_aligned_padded_frame',
]

UFrameData = typ.Tuple[NDArray, NDArray, NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray, NDArray, NDArray]]
#   up (d,)+C+(nU, N), down (d,)+C+(rL, nD, rR), left (d,)+C+(rL, nU, rL), right (d,)+C+(rR, nU, rR), shape (len=d),
#   masks HOST bool (up (d,)+C+(nU,), down (d,)+C+(nD,), left (d+1,)+C+(rL,), right (d+1,)+C+(rR,)); block layout: nU = nD, rL = rR
UVariationsData = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray, NDArray, NDArray]]
#   V (d,)+C+(nD, N), H (d,)+C+(rL, nU, rR), shape, variation masks; only V, H are read (masked with the frame's own masks)


def _rank_array(
        rank_spec:    typ.Union[typ.Sequence[int], NDArray],   # len=length, or HOST int (length,)+C
        length:       int,                                    # d or d+1
        stack_shape:  typ.Tuple[int, ...],                    # C
        name:         str,                                    # for the message
) -> NDArray:                                                 # HOST int, (length,)+C
    """Ranks per stack element: a sequence broadcast over the stack, or a full array checked."""
    arr = np.asarray(rank_spec)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError('%s must be integers; got dtype %s' % (name, arr.dtype))
    if arr.shape == (length,):
        return np.broadcast_to(arr.reshape((length,) + (1,) * len(stack_shape)), (length,) + stack_shape)
    if arr.shape == (length,) + stack_shape:
        return arr
    raise ValueError('%s has shape %s; expected (%d,) or (%d,)+C = %s' % (name, arr.shape, length, length,
                                                                        (length,) + stack_shape))


def _pad_trailing(
        A:       NDArray,             # (...,)+(a1, ..., ak)
        widths:  typ.Sequence[int],   # len=k, each >= a
) -> NDArray:                         # (...,)+widths: A in the leading slots, zeros elsewhere
    """Junk slots appended (zeros); nothing moves."""
    nl = A.ndim - len(widths)
    return linalg.pad_or_truncate(A, [(0, 0)] * nl + [(0, w - a) for w, a in zip(widths, A.shape[nl:])])


def _pad_mask(
        mask:   NDArray,   # HOST bool, (...,)+(w,)
        width:  int,       # >= w
) -> NDArray:              # HOST bool, (...,)+(width,): False appended
    """A mask with junk slots appended."""
    return np.concatenate([mask, np.zeros(mask.shape[:-1] + (width - mask.shape[-1],), dtype=bool)], axis=-1)


def _store_seeds(
        store:   str,       # 'down' | 'left' | 'right'
        keys:    NDArray,   # HOST int, (M,): each core's mode
        rows_a:  NDArray,   # HOST bool, (M,)+C+(wa,): the unfolding's first row slot, real slots
        rows_b:  NDArray,   # HOST bool, (M,)+C+(wb,): the second row slot (a-major rows)
        pins:    NDArray,   # HOST bool, (M,)+C+(w,): the old columns' mask (its sum: the old width)
        count:   NDArray,   # HOST int, (M,)+C: the new columns
        width:   int,       # the completion block's width, >= every count
        seed:    int,
) -> typ.Tuple[
    NDArray,   # seeds. HOST float, (M,)+C+(wa*wb, width): the ragged convention's seed rows at the real rows, junk zero
    NDArray,   # the real rows. HOST bool, (M,)+C+(wa*wb,)
]:
    """The hint-free convention's seeds of one store, per core and stack element, placed at the unfolding's real rows
    (in flat slot order: the ragged row-major order, since the masks keep the real slots in order)."""
    lead = rows_a.shape[:-1]
    wa, wb = rows_a.shape[-1], rows_b.shape[-1]
    real = (rows_a[..., :, None] & rows_b[..., None, :]).reshape(lead + (wa * wb,))
    old = pins.sum(axis=-1)
    seeds = np.zeros(lead + (wa * wb, width))
    for index in np.ndindex(*lead):                                     # host bookkeeping, per core and element
        k = int(count[index])
        if k:
            r = np.flatnonzero(real[index])
            seeds[index][r, :k] = _seed_rows(store, int(keys[index[0]]), seed, int(old[index]), k, r.size).T
    return seeds, real


def _store_completion(
        P:            NDArray,   # (M,)+C+(rows, w): the old cores unfolded at the new widths (zero off their real rows)
        pinned_mask:  NDArray,   # HOST bool, (M,)+C+(w,): the old columns' mask
        seeds:        NDArray,   # HOST float, (M,)+C+(rows, width): from _store_seeds
        real:         NDArray,   # HOST bool, (M,)+C+(rows,): the unfolding's real rows
        count:        NDArray,   # HOST int, (M,)+C: the new columns
) -> NDArray:                    # (M,)+C+(rows, width): the completion, real columns first, junk bitwise zero
    """One store's completion, batched over its cores: the seeds completed against the old columns (pad-safe)."""
    xnp, _, _ = get_backend(True, tree_contains_jax(P))
    width = seeds.shape[-1]
    if width == 0:
        return xnp.zeros(P.shape[:-1] + (0,), dtype=P.dtype)
    return linalg.pad_safe_pinned_completion(P, pinned_mask, xnp.asarray(seeds, dtype=P.dtype),
                                             prefix_mask(count, width), real)


def ufv_empty_frame(
        shape:        typ.Sequence[int],        # len=d, the real mode sizes
        N:            typ.Optional[int] = None, # the padded mode size; None = max(shape)
        stack_shape:  typ.Sequence[int] = (),   # C
        use_jax:      bool = False,             # array-free constructor: chooses the supercore type (masks are always host)
) -> typ.Tuple[                                 # the frame of the zero tensor (Na = 0, Rb = 1; default float):
    NDArray,               # up.    shape=(d,)+C+(0, N)
    NDArray,               # down.  shape=(d,)+C+(1, 0, 1)
    NDArray,               # left.  shape=(d,)+C+(1, 0, 1)
    NDArray,               # right. shape=(d,)+C+(1, 0, 1)
    typ.Tuple[int, ...],   # shape, len=d
    typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # masks, HOST bool: up/down (d,)+C+(0,); left/right (d+1,)+C+(1,), True at bonds 0 and d only
]:
    """The frame of the zero tensor in block layout: its orthogonal representation's prefix masks at widths (0, 0, 1, 1).

    Examples
    --------
    >>> import t3toolbox.backend.uniform_padded_frames as uniform_padded_frames
    >>> up, down, left, right, shape, masks = uniform_padded_frames.ufv_empty_frame((5, 6, 7), stack_shape=(2,))
    >>> print(up.shape, down.shape, shape)
    (3, 2, 0, 7) (3, 2, 1, 0, 1) (5, 6, 7)
    >>> print(masks[2][:, 0].ravel().tolist())         # left masks, stack element 0: real at bonds 0 and d only
    [True, False, False, True]
    """
    xnp, _, _ = get_backend(True, use_jax)
    d = len(shape)
    C = tuple(stack_shape)
    N = max(shape) if N is None else int(N)
    if N < max(shape):
        raise ValueError('N = %d is below the largest mode size %d' % (N, max(shape)))
    zeros = np.zeros((d,) + C, dtype=int)
    bonds = np.zeros((d + 1,) + C, dtype=int)
    bonds[0], bonds[d] = 1, 1
    stores = ranks.compute_orthogonal_representation_ranks(shape, zeros, bonds)
    masks = ufv_masking.ufv_make_frame_masks(*stores, 0, 0, 1, 1)
    cores = tuple(xnp.zeros((d,) + C + (1, 0, 1)) for _ in range(3))
    return (xnp.zeros((d,) + C + (0, N)),) + cores + (tuple(int(n) for n in shape), masks)


def ufv_padded_frame(
        old_frame_data:  UFrameData,                                  # the old frame at its own widths, any mask pattern, stack C
        tucker_ranks:    typ.Union[typ.Sequence[int], NDArray],       # len=d, or HOST int (d,)+C (per stack element)
        tt_ranks:        typ.Union[typ.Sequence[int], NDArray],       # len=d+1, or HOST int (d+1,)+C

        sharing:         typ.Optional[typ.Sequence] = None,           # len=d group labels; None = unshared
        seed:            int = ZERO_FRAME_SEED,                       # the keyed seed streams
) -> typ.Tuple[
    UFrameData,            # block layout: [A-block | P-block] per slot, old frame as given (junk zeroed), completions real-first
    typ.Tuple[int, int],   # split (Na, Rb): the static A-block widths
]:
    """The seeded padded frame in block layout: the old frame in the A-blocks, the completions by the hint-free convention."""
    U0, O0, L0, R0, shape, masks = old_frame_data
    require_concrete_masks(*masks)
    xnp, _, _ = get_backend(True, tree_contains_jax((U0, O0, L0, R0)))
    mU0, mD0, mL0, mR0 = (np.asarray(m, dtype=bool) for m in masks)
    d = len(shape)
    C = tuple(U0.shape[1:-2])
    N = int(U0.shape[-1])
    dtype = U0.dtype

    old = (mU0.sum(axis=-1), mD0.sum(axis=-1), mL0.sum(axis=-1), mR0.sum(axis=-1))   # per element, HOST int
    layout = ranks.compute_orthogonal_representation_ranks(shape, old[0], old[2])
    for name, have, want in zip(('up', 'down', 'left', 'right'), old, layout):
        if not np.array_equal(have, want):
            raise ValueError('old_frame_data: the %s masks count %s; an orthogonal representation with these up and '
                             'left counts has %s' % (name, have.tolist(), want.tolist()))
    tucker = _rank_array(tucker_ranks, d, C, 'tucker_ranks')
    tt = _rank_array(tt_ranks, d + 1, C, 'tt_ranks')
    if np.any(tt[0] != 1) or np.any(tt[-1] != 1):
        raise ValueError("the target boundary TT ranks are not 1; a frame's boundary bonds are 1")
    target = ranks.compute_orthogonal_representation_ranks(shape, tucker, tt)
    kU, kD, kL, kR = (np.asarray(t) - o for t, o in zip(target, old))   # the completion counts, HOST int
    for name, k in zip(('up', 'down', 'left', 'right'), (kU, kD, kL, kR)):
        if np.any(k < 0):
            raise ValueError('the target %s stores are below the old frame\'s: the target ranks must contain the old '
                             'structure' % name)

    Na, Rb = max(U0.shape[-2], O0.shape[-2]), max(L0.shape[-1], R0.shape[-1])
    Ka, Kb = int(max(kU.max(), kD.max())), int(max(kL.max(), kR.max()))
    nT, rT = Na + Ka, Rb + Kb
    U0, O0, L0, R0 = ufv_masking.ufv_apply_frame_masks(old_frame_data)              # masked once
    U0, O0 = _pad_trailing(U0, (Na, N)), _pad_trailing(O0, (Rb, Na, Rb))
    L0, R0 = _pad_trailing(L0, (Rb, Na, Rb)), _pad_trailing(R0, (Rb, Na, Rb))
    mU0, mD0, mL0, mR0 = _pad_mask(mU0, Na), _pad_mask(mD0, Na), _pad_mask(mL0, Rb), _pad_mask(mR0, Rb)
    mU = np.concatenate([mU0, prefix_mask(kU, Ka)], axis=-1)
    mD = np.concatenate([mD0, prefix_mask(kD, Ka)], axis=-1)
    mL = np.concatenate([mL0, prefix_mask(kL, Kb)], axis=-1)
    mR = np.concatenate([mR0, prefix_mask(kR, Kb)], axis=-1)

    # -- Tucker: one completion per sharing group, batched over the groups (a group's key is its first mode)
    groups = tuple((i,) for i in range(d)) if sharing is None else sharing_module.validate_sharing(sharing, shape)
    for group in groups:
        if any(not np.array_equal(mU0[i], mU0[group[0]]) or not np.array_equal(kU[i], kU[group[0]]) for i in group):
            raise ValueError('sharing group %r: unequal old up masks or Tucker counts; a shared factor needs one '
                             'layout per group' % (group,))
    first = np.array([group[0] for group in groups])
    group_of = np.array([next(g for g, group in enumerate(groups) if i in group) for i in range(d)])
    seeds = np.zeros((len(groups),) + C + (N, Ka))
    old_U = mU0[first].sum(axis=-1)                                     # (G,)+C: the old Tucker widths
    for g, f in enumerate(first):                                       # host bookkeeping: each group's keyed stream,
        for index in np.ndindex(*C):                                    #   per stack element
            k = int(kU[(f,) + index])
            if k:
                seeds[(g,) + index][:shape[f], :k] = _seed_rows('up', int(f), seed, int(old_U[(g,) + index]), k,
                                                                shape[f]).T
    mN = prefix_mask(np.asarray(shape), N)                              # (d, N)
    new_U = linalg.pad_safe_pinned_completion(
        U0[first].swapaxes(-2, -1), mU0[first], xnp.asarray(seeds, dtype=dtype),
        prefix_mask(kU[first], Ka), mN[first].reshape((len(groups),) + (1,) * len(C) + (N,)))
    U = xnp.concatenate([U0[first[group_of]], new_U[group_of].swapaxes(-2, -1)], axis=-2)   # tied within groups

    # -- left cores 0 .. d-2: new columns in the (a, n) unfolding; left[d-1] is the carrier
    P_L = _unfold(_pad_trailing(L0[:d - 1], (rT, nT, Rb)), -1)
    seeds_L, real_L = _store_seeds('left', np.arange(d - 1), mL[:d - 1], mU[:d - 1], mL0[1:d], kL[1:d], Kb, seed)
    new_L = _store_completion(P_L, mL0[1:d], seeds_L, real_L, kL[1:d])
    L = xnp.concatenate([xnp.concatenate([P_L, new_L], axis=-1).reshape((d - 1,) + C + (rT, nT, rT)),
                         _pad_trailing(L0[d - 1:], (rT, nT, rT))], axis=0)
    # -- right cores 1 .. d-1: new rows in the (n, b) unfolding; right[0] is the carrier
    P_R = _unfold(_pad_trailing(R0[1:], (Rb, nT, rT)), -3)
    seeds_R, real_R = _store_seeds('right', np.arange(1, d), mU[1:], mR[2:], mR0[1:d], kR[1:d], Kb, seed)
    new_R = _store_completion(P_R, mR0[1:d], seeds_R, real_R, kR[1:d])
    R = xnp.concatenate([_pad_trailing(R0[:1], (rT, nT, rT)),
                         xnp.moveaxis(xnp.concatenate([P_R, new_R], axis=-1).reshape((d - 1,) + C + (nT, rT, rT)),
                                      -1, -3)], axis=0)
    # -- down cores, all modes: new down slots in the (a, b) bond space
    P_O = _unfold(_pad_trailing(O0, (rT, Na, rT)), -2)
    seeds_O, real_O = _store_seeds('down', np.arange(d), mL[:-1], mR[1:], mD0, kD, Ka, seed)
    new_O = _store_completion(P_O, mD0, seeds_O, real_O, kD)
    O = xnp.moveaxis(xnp.concatenate([P_O, new_O], axis=-1).reshape((d,) + C + (rT, rT, nT)), -1, -2)
    return (U, O, L, R, tuple(int(n) for n in shape), (mU, mD, mL, mR)), (Na, Rb)


# --------------------------------------------------------------------------------------------------
# One alignment step in block layout (the uniform twin of padded_frames.tv_align_padded_frame). Every slot is
# [A-block | P-block] at the static split (Na, Rb); the P-parts are static slices. Map sites (all modes, or all sharing
# groups, at once) use the d-prefixed contraction twins; scan bodies the plain strings, and they derive the split from
# their operands' shapes (Kb = o.shape[-1], Rb = rT - Kb). Every edge's completion: pad_safe_svd of
# [Abar | Y_perp / (4 |Y_perp|_F)], its triplets after the pins (the A-block mask sum) picked by a host one-hot.
# --------------------------------------------------------------------------------------------------
def _new_coordinates_d(
        X:     NDArray,   # (d,)+C+(a, n, b): cores in their old slot coordinates
        o:     NDArray,   # (d,)+C+(k1, k): the coordinate maps of the edges indexing slot `axis`
        axis:  int,       # -3 | -1
) -> NDArray:             # (d,)+C+(a', n, b'): slot `axis` in the new coordinates
    """:py:func:`padded_frames._new_coordinates` over all modes at once (a bond slot)."""
    xnp, _, _ = get_backend(True, tree_contains_jax((X, o)))
    r = X.shape[axis] - o.shape[-1]
    if axis == -3:
        head, tail = X[..., :r, :, :], contractions.contract('dCxa,dCanb->dCxnb', o, X[..., r:, :, :])
    else:
        head, tail = X[..., :r], contractions.contract('dCxb,dCanb->dCanx', o, X[..., r:])
    return xnp.concatenate([head, tail], axis=axis)


def _pad_safe_basis(
        Abar:   NDArray,   # (...,)+(m, a): the pinned old block, orthonormal on its real columns, junk zero
        pins:   NDArray,   # bool, (...,)+(a,): its real columns
        Yperp:  NDArray,   # (...,)+(m, w): the content, orthogonal to Abar, zero on junk rows
        rows:   NDArray,   # bool, (...,)+(m,): the edge's real rows
) -> NDArray:              # (...,)+(m, min(m, a + w)): the pins, the content by singular value, then real completions
    """The ordered left singular vectors of ``[Abar | Yperp / (4 |Yperp|_F)]`` (a zero content stays zero)."""
    xnp, _, _ = get_backend(True, tree_contains_jax((Abar, Yperp)))
    lead = Yperp.shape[:-2]
    norm = xnp.sqrt(xnp.sum(Yperp * Yperp, axis=(-2, -1), keepdims=True))
    Yn = xnp.where(norm > 0, Yperp / xnp.where(norm > 0, 4.0 * norm, 1.0), 0.0)
    M = xnp.concatenate([xnp.broadcast_to(Abar, lead + Abar.shape[-2:]), Yn], axis=-1)
    col_mask = xnp.concatenate([xnp.broadcast_to(xnp.asarray(pins, dtype=bool), lead + pins.shape[-1:]),
                                xnp.ones(lead + Yn.shape[-1:], dtype=bool)], axis=-1)
    Us, _, _ = linalg.pad_safe_svd(M, xnp.asarray(rows, dtype=bool), col_mask)
    return Us


def _u_edge(
        Abar:   NDArray,   # C+(m, a): the pinned old block
        pins:   NDArray,   # bool, C+(a,): its real columns
        g_old:  NDArray,   # C+(m, k): the old completion in the new coordinates
        Y:      NDArray,   # C+(m, w): the content (g_old itself when the store is transported)
        rows:   NDArray,   # bool, C+(m,): the edge's real rows
        sel:    NDArray,   # bool, C+(a + k1, k1): the one-hot picking the triplets after the pins
) -> typ.Tuple[
    NDArray,   # P+. C+(m, k1): the new completion, junk rows and unselected columns exactly zero
    NDArray,   # o.  C+(k1, k): P+^T g_old
    NDArray,   # movement, shape=C: ||g_old - P+ o||_F
]:
    """One directed edge's new completion, coordinate map and movement (a scan body's edge)."""
    xnp, _, _ = get_backend(True, tree_contains_jax((Abar, g_old, Y, sel)))
    if sel.shape[-1] == 0:                                               # no completion slot
        P = xnp.zeros(Y.shape[:-1] + (0,), dtype=Y.dtype)
    else:
        Us = _pad_safe_basis(Abar, pins, Y - contractions.contract(
            'Cia,Cab->Cib', Abar, contractions.contract('Cia,Cib->Cab', Abar, Y)), rows)
        m = min(Us.shape[-1], sel.shape[-2])
        P = contractions.contract('Cik,Ckb->Cib', Us[..., :m], xnp.asarray(sel)[..., :m, :].astype(Us.dtype))
    o = contractions.contract('Cia,Cib->Cab', P, g_old)
    out = g_old - contractions.contract('Cia,Cab->Cib', P, o)
    return P, o, xnp.sqrt(xnp.sum(out * out, axis=(-2, -1)))


def _u_edge_d(
        Abar:   NDArray,   # (d,)+C+(m, a)
        pins:   NDArray,   # bool, (d,)+C+(a,)
        g_old:  NDArray,   # (d,)+C+(m, k)
        Y:      NDArray,   # (d,)+C+(m, w)
        rows:   NDArray,   # bool, (d,)+C+(m,)
        sel:    NDArray,   # bool, (d,)+C+(a + k1, k1)
) -> typ.Tuple[
    NDArray,   # P+. (d,)+C+(m, k1)
    NDArray,   # o.  (d,)+C+(k1, k)
    NDArray,   # movement, (d,)+C
]:
    """:py:func:`_u_edge` for all modes (or all sharing groups) at once."""
    xnp, _, _ = get_backend(True, tree_contains_jax((Abar, g_old, Y)))
    if sel.shape[-1] == 0:
        P = xnp.zeros(Y.shape[:-1] + (0,), dtype=Y.dtype)
    else:
        Us = _pad_safe_basis(Abar, pins, Y - contractions.contract(
            'dCia,dCab->dCib', Abar, contractions.contract('dCia,dCib->dCab', Abar, Y)), rows)
        m = min(Us.shape[-1], sel.shape[-2])
        P = contractions.contract('dCik,dCkb->dCib', Us[..., :m], xnp.asarray(sel[..., :m, :], dtype=Us.dtype))
    o = contractions.contract('dCia,dCib->dCab', P, g_old)
    out = g_old - contractions.contract('dCia,dCab->dCib', P, o)
    return P, o, xnp.sqrt(xnp.sum(out * out, axis=(-2, -1)))


def _u_content_d(
        g_old:  NDArray,   # (d,)+C+(m, k): the old completion in the new coordinates
        g_new:  NDArray,   # (d,)+C+(m, kf): the near terms on the far completion
        PhiP:   NDArray,   # (d,)+C+(k, kf): the far messages, completion part
        Xi:     NDArray,   # (d,)+C+(k, z): the far messages' Gram roots
) -> NDArray:              # (d,)+C+(m, kf + z): Y
    """An edge's normal content ``Y = [g_old Phi_P + g_new | g_old Xi]`` for all modes at once."""
    xnp, _, _ = get_backend(True, tree_contains_jax((g_old, g_new, PhiP, Xi)))
    return xnp.concatenate([contractions.contract('dCik,dCkb->dCib', g_old, PhiP) + g_new,
                            contractions.contract('dCik,dCkb->dCib', g_old, Xi)], axis=-1)


def _u_down_messages(
        VP:  NDArray,   # (d,)+C+(Ka, N): the variations' completion rows
        U:   NDArray,   # (d,)+C+(nT, N)
) -> typ.Tuple[
    NDArray,   # PhiO. (d,)+C+(Ka, nT)
    NDArray,   # XiO.  (d,)+C+(Ka, min(N, Ka))
]:
    """The messages of the edges core -> leaf, all modes at once."""
    PhiO = contractions.contract('dCkn,dCun->dCku', VP, U)
    return PhiO, _root(VP.swapaxes(-2, -1) - contractions.contract('dCun,dCku->dCnk', U, PhiO))


def _u_up_messages(
        H_P:  NDArray,   # (d,)+C+(rT, Ka, rT): the tangent cores at the Tucker completion slots
        bL:   NDArray,   # (d,)+C+(rT, Ka, Kb)
        bR:   NDArray,   # (d,)+C+(Kb, Ka, rT)
        O:    NDArray,   # (d,)+C+(rT, nT, rT)
        PhiL: NDArray,   # (d,)+C+(Kb, rT): edge c_i -> c_(i+1); zero at d-1
        XiL:  NDArray,   # (d,)+C+(Kb, Kb)
        PhiR: NDArray,   # (d,)+C+(Kb, rT): edge c_i -> c_(i-1); zero at 0
        XiR:  NDArray,   # (d,)+C+(Kb, Kb)
) -> typ.Tuple[
    NDArray,   # PhiU. (d,)+C+(Ka, nT)
    NDArray,   # XiU.  (d,)+C+(Ka, z)
]:
    """The messages of the edges leaf -> core, all modes at once."""
    xnp, _, _ = get_backend(True, tree_contains_jax((H_P, bL, bR, O, PhiL, XiL, PhiR, XiR)))
    Vb = (H_P + contractions.contract('dCaxk,dCkb->dCaxb', bL, PhiL)
          + contractions.contract('dCkxb,dCka->dCaxb', bR, PhiR))
    rows = [_unfold(contractions.contract('dCaxk,dCkz->dCaxz', bL, XiL), -2),
            _unfold(contractions.contract('dCkxb,dCkz->dCzxb', bR, XiR), -2)]
    PhiU = contractions.contract('dCaxb,dCaub->dCxu', Vb, O)
    W = _unfold(Vb - contractions.contract('dCxu,dCaub->dCaxb', PhiU, O), -2)
    return PhiU, _root(xnp.concatenate([W] + rows, axis=-2))


def _u_left_edge(
        carry:  typ.Tuple[NDArray, NDArray],   # (o_L, D_L) of the edge c_(i-1) -> c_i: C+(Kb1, Kb) ; C+(rT1, Kb)
        x:      typ.Tuple[NDArray, ...],       # (H_P, O_P, R_P, L_P, A, o_U, DU, PhiP, Xi, rows, pins, sel) at core i:
                                               #   C+(rT, nT, Kb) ; C+(rT, Ka, Kb) ; C+(Kb, nT, Kb) ; C+(rT, nT, Kb) ;
                                               #   C+(rT1, nT1, Rb) ; C+(Ka1, Ka) ; C+(nT1, Ka) ; C+(Kb, Kb) ; C+(Kb, Kb) ;
                                               #   bool C+(rT1 nT1,) ; bool C+(Rb,) ; bool C+(Rb + Kb1, Kb1)
        on:     bool,                          # False: the store is transported
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],               # (o_L, D_L) of the edge c_i -> c_(i+1)
    typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # (L+_i C+(rT1, nT1, rT1), o_L, D_L, movement C)
]:
    """One edge c_i -> c_(i+1) of the left sweep."""
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, x)))
    o_L, D_L = carry
    H_P, O_P, R_P, L_P, A, o_U, DU, PhiP, Xi, rows, pins, sel = x
    M = (_new_coordinates(_new_coordinates(H_P, o_L, -3), o_U, -2)
         + _new_coordinates(contractions.contract('Cakx,Cnk->Canx', O_P, DU), o_L, -3)
         + _new_coordinates(contractions.contract('Cknx,Cak->Canx', R_P, D_L), o_U, -2))
    g_new = _unfold(M, -1)
    g_old = _unfold(_new_coordinates(_new_coordinates(L_P, o_L, -3), o_U, -2), -1)
    Abar = _unfold(A, -1)
    Y = _edge_content(g_old, (g_new,), (PhiP,), (Xi,)) if on else g_old
    P, o, movement = _u_edge(Abar, pins, g_old, Y, rows, sel)
    F = xnp.concatenate([Abar, P], axis=-1)
    D = contractions.contract('Cia,Cib->Cab', F, g_new)
    return (o, D), (_fold(F, A.shape[-3:-1] + F.shape[-1:], -1), o, D, movement)


def _u_left_edge_step(
        carry:  typ.Tuple[NDArray, NDArray],   # as _u_left_edge
        x:      typ.Tuple[NDArray, ...],       # as _u_left_edge
) -> typ.Tuple[typ.Tuple[NDArray, NDArray], typ.Tuple[NDArray, NDArray, NDArray, NDArray]]:
    """One edge of the left sweep, the store updated (scan body)."""
    return _u_left_edge(carry, x, True)


def _u_left_edge_transport_step(
        carry:  typ.Tuple[NDArray, NDArray],   # as _u_left_edge
        x:      typ.Tuple[NDArray, ...],       # as _u_left_edge
) -> typ.Tuple[typ.Tuple[NDArray, NDArray], typ.Tuple[NDArray, NDArray, NDArray, NDArray]]:
    """One edge of the left sweep, the store transported (scan body)."""
    return _u_left_edge(carry, x, False)


def _u_right_edge(
        carry:  typ.Tuple[NDArray, NDArray],   # (o_R, D_R) of the edge c_(i+1) -> c_i: C+(Kb1, Kb) ; C+(rT1, Kb)
        x:      typ.Tuple[NDArray, ...],       # (H_P, O_P, L_P, R_P, A, o_U, DU, PhiP, Xi, rows, pins, sel) at core i:
                                               #   C+(Kb, nT, rT) ; C+(Kb, Ka, rT) ; C+(Kb, nT, Kb) ; C+(Kb, nT, rT) ;
                                               #   C+(Rb, nT1, rT1) ; C+(Ka1, Ka) ; C+(nT1, Ka) ; C+(Kb, Kb) ; C+(Kb, Kb) ;
                                               #   bool C+(nT1 rT1,) ; bool C+(Rb,) ; bool C+(Rb + Kb1, Kb1)
        on:     bool,                          # False: the store is transported
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],               # (o_R, D_R) of the edge c_i -> c_(i-1)
    typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # (R+_i C+(rT1, nT1, rT1), o_R, D_R, movement C)
]:
    """One edge c_i -> c_(i-1) of the right sweep (run on reversed inputs)."""
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, x)))
    o_R, D_R = carry
    H_P, O_P, L_P, R_P, A, o_U, DU, PhiP, Xi, rows, pins, sel = x
    M = (_new_coordinates(_new_coordinates(H_P, o_U, -2), o_R, -1)
         + _new_coordinates(contractions.contract('Cxkb,Cnk->Cxnb', O_P, DU), o_R, -1)
         + _new_coordinates(contractions.contract('Cxnk,Cbk->Cxnb', L_P, D_R), o_U, -2))
    g_new = _unfold(M, -3)
    g_old = _unfold(_new_coordinates(_new_coordinates(R_P, o_U, -2), o_R, -1), -3)
    Abar = _unfold(A, -3)
    Y = _edge_content(g_old, (g_new,), (PhiP,), (Xi,)) if on else g_old
    P, o, movement = _u_edge(Abar, pins, g_old, Y, rows, sel)
    F = xnp.concatenate([Abar, P], axis=-1)
    D = contractions.contract('Cia,Cib->Cab', F, g_new)
    return (o, D), (_fold(F, F.shape[-1:] + A.shape[-2:], -3), o, D, movement)


def _u_right_edge_step(
        carry:  typ.Tuple[NDArray, NDArray],   # as _u_right_edge
        x:      typ.Tuple[NDArray, ...],       # as _u_right_edge
) -> typ.Tuple[typ.Tuple[NDArray, NDArray], typ.Tuple[NDArray, NDArray, NDArray, NDArray]]:
    """One edge of the right sweep, the store updated (scan body)."""
    return _u_right_edge(carry, x, True)


def _u_right_edge_transport_step(
        carry:  typ.Tuple[NDArray, NDArray],   # as _u_right_edge
        x:      typ.Tuple[NDArray, ...],       # as _u_right_edge
) -> typ.Tuple[typ.Tuple[NDArray, NDArray], typ.Tuple[NDArray, NDArray, NDArray, NDArray]]:
    """One edge of the right sweep, the store transported (scan body)."""
    return _u_right_edge(carry, x, False)


def _sweep(
        xscan:   typ.Callable,                 # the backend's scan
        body:    typ.Callable,                 # a module-level scan body
        init:    typ.Tuple[NDArray, ...],      # the carry past the end
        xs:      typ.Tuple[NDArray, ...],      # the per-step inputs, (n,)+... each
        empty:   typ.Tuple[NDArray, ...],      # the outputs of a zero-length sweep (d = 1), (0,)+... each
        reverse: bool,                         # True: from the last step to the first
) -> typ.Tuple[NDArray, ...]:                  # the stacked outputs, in step order
    """A sweep along the mode axis (``xscan`` on reversed inputs for a reverse sweep, outputs reversed back)."""
    if xs[0].shape[0] == 0:
        return empty
    if reverse:
        return tuple(y[::-1] for y in xscan(body, init, tuple(x[::-1] for x in xs))[1])
    return tuple(xscan(body, init, xs)[1])


def _selection(
        pins:    NDArray,   # HOST int, (...,): the pin count (the A-block mask sum)
        k_mask:  NDArray,   # HOST bool, (...,)+(K1,): the output completion mask
        A:       int,       # the A-block width
) -> NDArray:               # HOST bool, (...,)+(A + K1, K1): column j picks triplet pins + j where k_mask[j]
    """The static one-hot selecting an edge's completion triplets (they follow the pins in pad_safe_svd's order)."""
    K1 = k_mask.shape[-1]
    return (np.arange(A + K1)[:, None] == np.asarray(pins)[..., None, None] + np.arange(K1)) & k_mask[..., None, :]


def _utv_align_kernel(
        dynamic:   typ.Tuple[NDArray, ...],   # partition_static's dynamic leaves: the supercores and the variations
        skeleton:  StaticSkeleton,            # the rest: shape, masks, row masks, pins, selections, group tables
        split:     typ.Tuple[int, int],       # (Na, Rb), static
        update:    str,                       # static: the stores updated; the rest transported
) -> typ.Tuple[
    NDArray,   # U+. (d,)+C+(nT1, N)
    NDArray,   # O+. (d,)+C+(rT1, nT1, rT1)
    NDArray,   # L+. (d,)+C+(rT1, nT1, rT1)
    NDArray,   # R+. (d,)+C+(rT1, nT1, rT1)
    NDArray,   # movement, shape=C
]:
    """The arithmetic of :py:func:`utv_align_padded_frame` (jittable: every static is in ``skeleton``, ``split``, ``update``)."""
    (U, O, L, R), (V, H), host = rebuild_static(dynamic, skeleton)
    shape, masks, (rowU, rowL, rowR, rowO), (pinU, pinL, pinR, pinO), (selU, selL, selR, selO), groups = host
    members, valid, first, group_of = (np.asarray(t) for t in groups)
    xnp, _, xscan = get_backend(True, tree_contains_jax((U, O, L, R, V, H)))
    Na, Rb = split
    d = U.shape[0]
    C = tuple(U.shape[1:-2])
    nT, N = U.shape[-2:]
    rT = L.shape[-1]
    Ka, Kb = nT - Na, rT - Rb
    Ka1, Kb1 = selU.shape[-1], selL.shape[-1]
    nT1, rT1 = Na + Ka1, Rb + Kb1
    dtype = xnp.result_type(U, V)
    zeros = lambda *s: xnp.zeros(s, dtype=dtype)

    U, O, L, R = ufv_masking.ufv_apply_frame_masks((U, O, L, R, shape, masks))            # masked once
    V, H = ufv_masking.ufv_apply_variations_masks((V, H, shape, ufv_masking.ufv_variation_masks(masks)))

    # -- step 1: far-side messages at the old frames
    PhiO, XiO = _u_down_messages(V[..., Na:, :], U)
    zero_msg = (zeros(*C, Kb, rT), zeros(*C, Kb, Kb))
    empty_msg = (zeros(0, *C, Kb, rT), zeros(0, *C, Kb, Kb))
    PhiL, XiL = _sweep(xscan, _left_message_step, zero_msg, (
        H[1:, ..., Rb:, :, :], O[1:, ..., Rb:, Na:, :], L[1:, ..., Rb:, :, Rb:], R[1:], PhiO[1:], XiO[1:]),
        empty_msg, True)
    PhiL, XiL = (xnp.concatenate([m, z[None]], axis=0) for m, z in zip((PhiL, XiL), zero_msg))   # c_i -> c_(i+1)
    PhiR, XiR = _sweep(xscan, _right_message_step, zero_msg, (
        H[:-1, ..., Rb:], O[:-1, ..., Na:, Rb:], R[:-1, ..., Rb:, :, Rb:], L[:-1], PhiO[:-1], XiO[:-1]),
        empty_msg, False)
    PhiR, XiR = (xnp.concatenate([z[None], m], axis=0) for m, z in zip((PhiR, XiR), zero_msg))   # c_i -> c_(i-1)
    PhiU, XiU = _u_up_messages(H[..., Na:, :], L[..., Na:, Rb:], R[..., Rb:, Na:, :], O, PhiL, XiL, PhiR, XiR)

    # -- step 2, up edges: one completion per sharing group, the members' contents side by side
    Abar = U[..., :Na, :].swapaxes(-2, -1)                                       # (d,)+C+(N, Na)
    g_old, g_new = U[..., Na:, :].swapaxes(-2, -1), V[..., Na:, :].swapaxes(-2, -1)
    Y = _u_content_d(g_old, g_new, PhiU[..., Na:], XiU) if 'U' in update else g_old
    G, m = members.shape
    Yg = Y[members] * xnp.asarray(valid, dtype=dtype).reshape((G, m) + (1,) * (len(C) + 2))
    Yg = xnp.moveaxis(Yg, 1, -2).reshape((G,) + C + (N, m * Y.shape[-1]))      # missing members zero
    P, _, _ = _u_edge_d(Abar[first], pinU, g_old[first], Yg, rowU, selU)       # (G,)+C+(N, Ka1)
    P = P[group_of]
    Abar = Abar[first[group_of]]                                                 # the group's old block, every member
    oU = contractions.contract('dCia,dCib->dCab', P, g_old)
    out = g_old - contractions.contract('dCia,dCab->dCib', P, oU)
    moves = [xnp.sqrt(xnp.sum(out * out, axis=(-2, -1)))]
    DU = contractions.contract('dCia,dCib->dCab', xnp.concatenate([Abar, P], axis=-1), g_new)   # (d,)+C+(nT1, Ka)
    U1 = xnp.concatenate([Abar.swapaxes(-2, -1), P.swapaxes(-2, -1)], axis=-2)                # tied within groups

    # -- the left sweep (i = 0 .. d-2) and the right sweep (i = d-1 .. 1)
    zero_edge = (zeros(*C, Kb1, Kb), zeros(*C, rT1, Kb))
    empty_edge = (zeros(0, *C, rT1, nT1, rT1), zeros(0, *C, Kb1, Kb), zeros(0, *C, rT1, Kb), zeros(0, *C))
    L1, oL, DL, mvL = _sweep(xscan, _u_left_edge_step if 'L' in update else _u_left_edge_transport_step, zero_edge, (
        H[:-1, ..., Rb:], O[:-1, ..., Na:, Rb:], R[:-1, ..., Rb:, :, Rb:], L[:-1, ..., Rb:],
        _pad_trailing(L[:-1, ..., :Rb, :Na, :Rb], (rT1, nT1, Rb)), oU[:-1], DU[:-1], PhiL[:-1, ..., Rb:], XiL[:-1],
        rowL, pinL, selL), empty_edge, False)
    R1, oR, DR, mvR = _sweep(xscan, _u_right_edge_step if 'R' in update else _u_right_edge_transport_step, zero_edge, (
        H[1:, ..., Rb:, :, :], O[1:, ..., Rb:, Na:, :], L[1:, ..., Rb:, :, Rb:], R[1:, ..., Rb:, :, :],
        _pad_trailing(R[1:, ..., :Rb, :Na, :Rb], (Rb, nT1, rT1)), oU[1:], DU[1:], PhiR[1:, ..., Rb:], XiR[1:],
        rowR, pinR, selR), empty_edge, True)
    moves += [mvL, mvR]

    # -- down edges, all modes at once, with the incoming coordinate maps (zero past the ends)
    oL, DL = (xnp.concatenate([z[None], a], axis=0) for a, z in zip((oL, DL), zero_edge))   # c_(i-1) -> c_i
    oR, DR = (xnp.concatenate([a, z[None]], axis=0) for a, z in zip((oR, DR), zero_edge))   # c_(i+1) -> c_i
    M = (_new_coordinates_d(_new_coordinates_d(H[..., Na:, :], oL, -3), oR, -1)
         + _new_coordinates_d(contractions.contract('dCaxk,dCbk->dCaxb', L[..., Na:, Rb:], DR), oL, -3)
         + _new_coordinates_d(contractions.contract('dCkxb,dCak->dCaxb', R[..., Rb:, Na:, :], DL), oR, -1))
    g_new = _unfold(M, -2)
    g_old = _unfold(_new_coordinates_d(_new_coordinates_d(O[..., Na:, :], oL, -3), oR, -1), -2)
    Abar = _unfold(_pad_trailing(O[..., :Rb, :Na, :Rb], (rT1, Na, rT1)), -2)
    Y = _u_content_d(g_old, g_new, PhiO[..., Na:], XiO) if 'O' in update else g_old
    P, _, mvO = _u_edge_d(Abar, pinO, g_old, Y, rowO, selO)
    O1 = _fold(xnp.concatenate([Abar, P], axis=-1), (rT1, nT1, rT1), -2)
    moves.append(mvO)

    L1 = xnp.concatenate([L1, _pad_trailing(L[d - 1:, ..., :Rb, :Na, :Rb], (rT1, nT1, rT1))], axis=0)   # carriers:
    R1 = xnp.concatenate([_pad_trailing(R[:1, ..., :Rb, :Na, :Rb], (rT1, nT1, rT1)), R1], axis=0)       # old only
    movement = xnp.max(xnp.concatenate([mv.reshape((-1,) + C) for mv in moves], axis=0), axis=0)
    return U1, O1, L1, R1, movement


def utv_align_padded_frame(
        frame_data:       UFrameData,                                        # block layout, any mask pattern in either block
        variations_data:  UVariationsData,                                   # a tangent vector at `frame_data`
        split:            typ.Tuple[int, int],                               # (Na, Rb), static

        tucker_ranks:     typ.Optional[typ.Union[typ.Sequence[int], NDArray]] = None,   # len=d or HOST int (d,)+C; None = the frame's own
        tt_ranks:         typ.Optional[typ.Union[typ.Sequence[int], NDArray]] = None,   # len=d+1 or HOST int (d+1,)+C; None = the frame's own

        sharing:          typ.Optional[typ.Sequence] = None,                 # len=d group labels; None = unshared
        update:           str = 'ULRO',                                      # static; as tv_align_padded_frame
) -> typ.Tuple[
    UFrameData,   # P-blocks at the output widths (the counts' maxima), real-first; A-blocks as given
    NDArray,      # movement, shape=C
]:
    """One alignment step in block layout: the uniform twin of ``padded_frames.tv_align_padded_frame``."""
    if not update or len(set(update)) != len(update) or not set(update) <= set('ULRO'):
        raise ValueError("update must be a non-empty set of the stores 'U', 'L', 'R', 'O' without repeats; got %r"
                         % (update,))
    U, O, L, R, shape, masks = frame_data
    V, H = variations_data[0], variations_data[1]
    require_concrete_masks(*masks)
    mU, mD, mL, mR = (np.asarray(m, dtype=bool) for m in masks)
    shape = tuple(int(n) for n in shape)
    d = len(shape)
    C = tuple(U.shape[1:-2])
    nT, N = int(U.shape[-2]), int(U.shape[-1])
    rT = int(L.shape[-1])
    Na, Rb = (int(w) for w in split)
    core = (d,) + C + (rT, nT, rT)
    for name, A, want in (('up', U, (d,) + C + (nT, N)), ('down', O, core), ('left', L, core), ('right', R, core),
                          ('tucker variations', V, (d,) + C + (nT, N)), ('tt variations', H, core),
                          ('up mask', mU, (d,) + C + (nT,)), ('down mask', mD, (d,) + C + (nT,)),
                          ('left mask', mL, (d + 1,) + C + (rT,)), ('right mask', mR, (d + 1,) + C + (rT,))):
        if tuple(A.shape) != want:
            raise ValueError('%s has shape %s; block layout (nU = nD = %d, rL = rR = %d) gives %s'
                             % (name, tuple(A.shape), nT, rT, want))
    if not (0 <= Na <= nT and 1 <= Rb <= rT):
        raise ValueError('split %r is outside the widths (nT, rT) = (%d, %d)' % ((Na, Rb), nT, rT))

    old = (mU[..., :Na].sum(-1), mD[..., :Na].sum(-1), mL[..., :Rb].sum(-1), mR[..., :Rb].sum(-1))   # HOST int
    own = (mU.sum(-1), mD.sum(-1), mL.sum(-1), mR.sum(-1))
    for which, counts in (('A-block', old), ('frame', own)):
        layout = ranks.compute_orthogonal_representation_ranks(shape, counts[0], counts[2])
        bounds = (counts[2][0], counts[2][-1], counts[3][0], counts[3][-1])
        if any(not np.array_equal(c, w) for c, w in zip(counts, layout)) or any(np.any(b != 1) for b in bounds):
            raise ValueError('the %s masks do not count an orthogonal representation (stores %r, which its up and left '
                             'counts give as %r, boundary bonds 1)' % (which, [c.tolist() for c in counts],
                                                                      [w.tolist() for w in layout]))
    if (tucker_ranks is None) != (tt_ranks is None):
        raise ValueError('give both output rank sequences (tucker_ranks, tt_ranks) or neither')
    out = own
    if tucker_ranks is not None:
        tt = _rank_array(tt_ranks, d + 1, C, 'tt_ranks')
        if np.any(tt[0] != 1) or np.any(tt[-1] != 1):
            raise ValueError("the output boundary TT ranks are not 1; a frame's boundary bonds are 1")
        out = ranks.compute_orthogonal_representation_ranks(shape, _rank_array(tucker_ranks, d, C, 'tucker_ranks'), tt)
    out = tuple(np.broadcast_to(np.asarray(s), np.shape(o)) for s, o in zip(out, old))
    for store, letter, o, s1, s in zip(('up', 'down', 'left', 'right'), 'UOLR', old, out, own):
        if np.any(o > s1) or np.any(s1 > s):
            raise ValueError('%s stores: old %s, output %s, frame %s; the old block must fit the output and the output '
                             'the frame' % (store, o.tolist(), s1.tolist(), s.tolist()))
        if letter not in update and np.any(s1 < s):
            raise ValueError('%s stores are truncated (%s -> %s) but not updated: a pass may not both transport and '
                             'truncate' % (store, s.tolist(), s1.tolist()))
    groups = tuple((i,) for i in range(d)) if sharing is None else sharing_module.validate_sharing(sharing, shape)
    for group in groups:
        f = group[0]
        if any(not (np.array_equal(old[0][i], old[0][f]) and np.array_equal(out[0][i], out[0][f])
                    and np.array_equal(own[0][i], own[0][f]) and np.array_equal(mU[i, ..., :Na], mU[f, ..., :Na]))
               for i in group):
            raise ValueError('sharing group %r: unequal Tucker stores or old up masks; a shared factor needs one layout '
                             'per group' % (group,))

    k1 = tuple(np.asarray(s1) - o for s1, o in zip(out, old))                  # the output completion counts
    Ka1, Kb1 = int(max(k1[0].max(), k1[1].max(), 0)), int(max(k1[2].max(), k1[3].max(), 0))
    nT1, rT1 = Na + Ka1, Rb + Kb1
    pU, pD, pL, pR = prefix_mask(k1[0], Ka1), prefix_mask(k1[1], Ka1), prefix_mask(k1[2], Kb1), prefix_mask(k1[3], Kb1)
    mU1, mD1 = np.concatenate([mU[..., :Na], pU], -1), np.concatenate([mD[..., :Na], pD], -1)
    mL1, mR1 = np.concatenate([mL[..., :Rb], pL], -1), np.concatenate([mR[..., :Rb], pR], -1)

    m = max(len(g) for g in groups)
    members = tuple(tuple(g) + (g[0],) * (m - len(g)) for g in groups)
    valid = np.array([[j < len(g) for j in range(m)] for g in groups], dtype=bool)
    first = tuple(g[0] for g in groups)
    group_of = tuple(next(k for k, g in enumerate(groups) if i in g) for i in range(d))
    lead = lambda a: a.reshape(a.shape[:1 + len(C)] + (-1,))
    rows = (np.broadcast_to(prefix_mask(np.asarray(shape), N)[list(first)].reshape((len(groups),) + (1,) * len(C) + (N,)),
                            (len(groups),) + C + (N,)),
            lead(mL1[:-1][..., :, None] & mU1[..., None, :])[:-1],
            lead(mU1[..., :, None] & mR1[1:][..., None, :])[1:],
            lead(mL1[:-1][..., :, None] & mR1[1:][..., None, :]))
    pins = (mU[list(first)][..., :Na], mL[1:-1][..., :Rb], mR[1:-1][..., :Rb], mD[..., :Na])
    sels = (_selection(pins[0].sum(-1), pU[list(first)], Na), _selection(pins[1].sum(-1), pL[1:-1], Rb),
            _selection(pins[2].sum(-1), pR[1:-1], Rb), _selection(pins[3].sum(-1), pD, Na))
    host = (shape, (mU, mD, mL, mR), rows, pins, sels, (members, valid, first, group_of))
    dynamic, skeleton = partition_static(((U, O, L, R), (V, H), host))
    U1, O1, L1, R1, movement = _utv_align_kernel(dynamic, skeleton, (Na, Rb), update)
    return (U1, O1, L1, R1, shape, (mU1, mD1, mL1, mR1)), movement


# --------------------------------------------------------------------------------------------------
# Compaction and the loop. The loop core runs the schedule of padded_frames.fv_aligned_padded_frame (the same pair
# stop rule) with Python loops when eager, and with lax.fori_loop / while_loop / cond when the whole core is jitted.
# Its loop bodies are module-level; the static structure of a phase (project, the hint, the masks, the split, the
# output ranks, sharing, the colour) is bound explicitly. That binding is traced once per compiled program: the lax
# loops run only inside the one jax.jit of the core, whose cache is the core's identity and static arguments.
# --------------------------------------------------------------------------------------------------
def ufv_compact_frame(
        frame_data:    UFrameData,                     # any layout and mask pattern
        level_widths:  typ.Tuple[int, int, int, int],  # (WU, WD, WL, WR), each >= every real count of its store
) -> UFrameData:                                       # junk-last, junk zeroed; masks = ufv_make_frame_masks of the counts at level_widths
    """The frame with every store's real slots first (in their order) at ``level_widths``: static gathers, prefix masks."""
    U, O, L, R, shape, masks = frame_data
    require_concrete_masks(*masks)
    xnp, _, _ = get_backend(True, tree_contains_jax((U, O, L, R)))
    masks = tuple(np.asarray(m, dtype=bool) for m in masks)
    WU, WD, WL, WR = (int(w) for w in level_widths)
    counts = tuple(m.sum(axis=-1) for m in masks)
    for name, count, W in zip(('up', 'down', 'left', 'right'), counts, (WU, WD, WL, WR)):
        if np.any(count > W):
            raise ValueError('a real %s count (%d) exceeds its level width %d' % (name, int(count.max()), W))
    wide = (max(U.shape[-2], WU), max(O.shape[-2], WD), max(L.shape[-1], WL), max(R.shape[-1], WR))
    U = _pad_trailing(U, (wide[0], U.shape[-1]))                              # junk appended where narrower
    O = _pad_trailing(O, (wide[2], wide[1], wide[3]))
    L, R = _pad_trailing(L, (wide[2], wide[0], wide[2])), _pad_trailing(R, (wide[3], wide[0], wide[3]))
    iU, iD, iL, iR = (np.argsort(~_pad_mask(m, w), axis=-1, kind='stable')[..., :W]          # real slots first
                      for m, w, W in zip(masks, wide, (WU, WD, WL, WR)))
    take = lambda A, index, axis: xnp.take_along_axis(A, xnp.asarray(index), axis=axis)
    a3, b3, c3 = (lambda i: i[..., :, None, None]), (lambda i: i[..., None, :, None]), (lambda i: i[..., None, None, :])
    U = take(U, iU[..., :, None], -2)
    O = take(take(take(O, a3(iL[:-1]), -3), b3(iD), -2), c3(iR[1:]), -1)
    L = take(take(take(L, a3(iL[:-1]), -3), b3(iU), -2), c3(iL[1:]), -1)
    R = take(take(take(R, a3(iR[:-1]), -3), b3(iU), -2), c3(iR[1:]), -1)
    masks = ufv_masking.ufv_make_frame_masks(*counts, WU, WD, WL, WR)
    return ufv_masking.ufv_apply_frame_masks((U, O, L, R, shape, masks)) + (tuple(shape), masks)


def _record(
        movements:  NDArray,   # (max_passes,)+C
        i:          typ.Any,   # the pass index (int, or traced)
        movement:   NDArray,   # C
) -> NDArray:                  # movements with slot i set
    """One pass's movement written into the fixed-length record (branch-free)."""
    xnp, _, _ = get_backend(True, tree_contains_jax((movements, movement)))
    hit = (xnp.arange(movements.shape[0]) == i).reshape((-1,) + (1,) * (movements.ndim - 1))
    return xnp.where(hit, movement, movements)


def _aligning_pass(
        statics:  typ.Tuple,                  # (project, hint, shape, masks, split, (tucker, tt) output ranks or (None, None), sharing, update)
        cores:    typ.Tuple[NDArray, ...],    # (U, O, L, R) in block layout at `masks`
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                  # the cores after the pass
    typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # their masks (host)
    NDArray,                                  # the pass's movement, shape=C
]:
    """One pass: project at the block-layout frame, then one step."""
    project, hint, shape, masks, split, out_ranks, sharing, update = statics
    frame = tuple(cores) + (shape, masks)
    F, movement = utv_align_padded_frame(frame, project(frame, hint), split, *out_ranks, sharing=sharing, update=update)
    return tuple(F[:4]), F[5], movement


def _oversampled_body(
        statics:  typ.Tuple,   # as _aligning_pass, at the working structure, 'ULRO'
        i:        typ.Any,     # the pass index
        state:    typ.Tuple,   # (cores, movements)
) -> typ.Tuple:
    """One oversampled pass (fori body)."""
    cores, movements = state
    cores, _, movement = _aligning_pass(statics, cores)
    return cores, _record(movements, i, movement)


def _pairs_cond(
        state:  typ.Tuple,   # (pass index, cores, movements, stop, tol, max_passes)
) -> NDArray:                # 0-d bool: run another ('UL', 'RO') pair
    """Not stopped, and a whole pair fits under the cap."""
    i, _, _, stop, _, cap = state
    return ~stop & (i + 2 <= cap)


def _pairs_body(
        statics:  typ.Tuple[typ.Tuple, typ.Tuple],   # the 'UL' and the 'RO' passes' statics, at the target structure
        state:    typ.Tuple,                         # (pass index, cores, movements, stop, tol, max_passes)
) -> typ.Tuple:
    """One ('UL', 'RO') pair; the stop: both movements below tol (the max over the stack; a NaN never stops)."""
    xnp, _, _ = get_backend(True, tree_contains_jax(state[1:3]))
    i, cores, movements, _, tol, cap = state
    cores, _, a = _aligning_pass(statics[0], cores)
    cores, _, b = _aligning_pass(statics[1], cores)
    movements = _record(_record(movements, i, a), i + 1, b)
    return i + 2, cores, movements, xnp.maximum(xnp.max(a), xnp.max(b)) < tol, tol, cap


def _final_pass(
        statics:  typ.Tuple,   # the 'UL' pass's statics
        state:    typ.Tuple,   # (pass index, cores, movements, stop, tol, max_passes)
) -> typ.Tuple:
    """The odd remainder's final 'UL' pass."""
    i, cores, movements, stop, tol, cap = state
    cores, _, movement = _aligning_pass(statics, cores)
    return i + 1, cores, _record(movements, i, movement), stop, tol, cap


def _no_pass(
        state:  typ.Tuple,
) -> typ.Tuple:
    """The state unchanged (the other branch of the final pass)."""
    return state


def _single_cond(
        state:  typ.Tuple,   # (pass index, cores, movements, stop, tol, max_passes)
) -> NDArray:                # 0-d bool
    """Not stopped, and under the cap."""
    i, _, _, stop, _, cap = state
    return ~stop & (i < cap)


def _single_body(
        statics:  typ.Tuple,   # the 'ULRO' pass's statics, at the target structure
        state:    typ.Tuple,
) -> typ.Tuple:
    """One simultaneous pass; the stop: its movement below tol."""
    xnp, _, _ = get_backend(True, tree_contains_jax(state[1:3]))
    i, cores, movements, _, tol, cap = state
    cores, _, movement = _aligning_pass(statics, cores)
    return i + 1, cores, _record(movements, i, movement), xnp.max(movement) < tol, tol, cap


def _fori(n, body, state, use_jit):
    """``n`` iterations of ``body(i, state)``: ``jax.lax.fori_loop`` inside the jit, else a Python loop."""
    if use_jit:
        import jax
        return jax.lax.fori_loop(0, n, body, state)
    for i in range(n):
        state = body(i, state)
    return state


def _cond(pred, true_fn, false_fn, state, use_jit):
    """``jax.lax.cond`` inside the jit, else a Python branch."""
    if use_jit:
        import jax
        return jax.lax.cond(pred, true_fn, false_fn, state)
    return true_fn(state) if bool(pred) else false_fn(state)


def _nested_tuple(
        a:  NDArray,   # HOST int array
) -> typ.Any:          # nested tuples of Python ints (hashable; np.asarray restores the array)
    """A host int array as a static (hashable) value."""
    return tuple(_nested_tuple(x) for x in a) if np.ndim(a) else int(a)


def _ufv_aligned_loop_kernel(
        dynamic:   typ.Tuple,   # partition_static's dynamic leaves of (old frame data, hint)
        tol:       typ.Any,     # the stop threshold (traced under jit)
        skeleton:  StaticSkeleton,   # the rest of (old frame data, hint)
        project:   typ.Callable,     # project(frame_data, hint) -> variations (module-level under jit)
        schedule:  typ.Tuple,   # (working ranks, target ranks, sharing, seed, n_over, max_passes, alternate, level_widths), hashable
        use_jit:   bool,        # True: inside jax.jit (lax loops); False: Python loops, eager
) -> typ.Tuple[
    NDArray, NDArray, NDArray, NDArray,   # the aligned frame's supercores, compacted to level_widths
    NDArray,                              # movements, (max_passes,)+C, NaN after the last pass
]:
    """The loop of :py:func:`ufv_aligned_padded_frame`: the start, the passes on the pair schedule, the compaction.

    Its loop bodies are bound with ``functools.partial``; those partials reach a lax primitive only under
    ``use_jit=True``, inside the one memoized jit whose cache is keyed on ``skeleton``, ``project``, ``schedule`` and
    ``use_jit``; eagerly the loops are Python loops. This binding is accepted in this instance only and sets no
    precedent for other loop bodies."""
    old_frame_data, hint = rebuild_static(dynamic, skeleton)
    working, target, sharing, seed, n_over, max_passes, alternate, level_widths = schedule
    working, target = tuple(tuple(np.asarray(r) for r in ranks_) for ranks_ in (working, target))
    frame, split = ufv_padded_frame(old_frame_data, *working, sharing=sharing, seed=seed)
    xnp, _, _ = get_backend(True, tree_contains_jax(frame[:4]))
    shape, masks = frame[4], frame[5]
    C = tuple(frame[0].shape[1:-2])
    movements = xnp.full((max_passes,) + C, np.nan, dtype=frame[0].dtype)

    statics = (project, hint, shape, masks, split, (None, None), sharing, 'ULRO')
    cores, movements = _fori(n_over, ft.partial(_oversampled_body, statics), (tuple(frame[:4]), movements), use_jit)
    cores, masks, movement = _aligning_pass(statics[:5] + (target,) + statics[6:], cores)   # the truncating pass
    movements = _record(movements, n_over, movement)
    at = lambda update: (project, hint, shape, masks, split, (None, None), sharing, update)
    if alternate:
        state = (xnp.asarray(n_over + 1), cores, movements, xnp.asarray(False), tol, max_passes)
        state = xwhile(_pairs_cond, ft.partial(_pairs_body, (at('UL'), at('RO'))), state, use_jit)
        state = _cond(_single_cond(state), ft.partial(_final_pass, at('UL')), _no_pass, state, use_jit)
    else:
        state = (xnp.asarray(n_over + 1), cores, movements, xnp.max(movement) < tol, tol, max_passes)
        state = xwhile(_single_cond, ft.partial(_single_body, at('ULRO')), state, use_jit)
    compact = ufv_compact_frame(tuple(state[1]) + (shape, masks), level_widths)
    return tuple(compact[:4]) + (state[2],)


def ufv_aligned_padded_frame(
        old_frame_data:      UFrameData,                              # the old frame, stack C (ufv_empty_frame for the zero tensor)
        tucker_ranks:        typ.Union[typ.Sequence[int], NDArray],   # len=d or HOST int (d,)+C, target
        tt_ranks:            typ.Union[typ.Sequence[int], NDArray],   # len=d+1 or HOST int (d+1,)+C, target
        hint:                typ.Any,                                 # opaque: handed to `project` unread
        project:             typ.Callable,                            # project(frame_data, hint) -> UVariationsData (or the bare (V, H) pair); traceable and module-level under use_jit
        level_widths:        typ.Tuple[int, int, int, int],           # (WU, WD, WL, WR): the compaction widths (the level's frame)

        sharing:             typ.Optional[typ.Sequence] = None,       # len=d group labels; None = unshared
        oversampling:        int   = ALIGN_OVERSAMPLING,              # as fv_aligned_padded_frame (constants from padded_frames)
        oversampled_passes:  int   = ALIGN_OVERSAMPLED_PASSES,
        max_passes:          int   = ALIGN_MAX_PASSES,                # static under use_jit (the movements' length)
        tol:                 float = ALIGN_TOL,                       # traced under use_jit
        alternate:           bool  = ALIGN_ALTERNATE,
        seed:                int   = ZERO_FRAME_SEED,
        use_jit:             bool  = False,                           # the WHOLE loop as one jax.jit; else a host loop, eager kernel
) -> typ.Tuple[
    UFrameData,   # the aligned frame compacted to level_widths, prefix masks
    NDArray,      # movements, shape=(max_passes,)+C, NaN after the last pass (both modes)
]:
    """The aligned padded frame in block layout, compacted to ``level_widths``: the uniform twin of ``fv_aligned_padded_frame``."""
    for name, value in (('max_passes', max_passes), ('oversampled_passes', oversampled_passes),
                        ('oversampling', oversampling)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError('%s must be a non-negative int, got %r' % (name, value))
    jitted = bool(use_jit) and jax_or_warn('use_jit=True')     # no jax: a warning, then eager on numpy
    if jitted:                                                  # asking for jit opts into jax: the old frame's
        old_frame_data = tuple(tree_to_jax(a) for a in old_frame_data[:4]) + tuple(old_frame_data[4:])   # supercores
    U0, _, _, _, shape, masks = old_frame_data
    require_concrete_masks(*masks)
    shape = tuple(int(n) for n in shape)
    d, C = len(shape), tuple(U0.shape[1:-2])
    level_widths = tuple(int(w) for w in level_widths)
    old = tuple(np.asarray(m, dtype=bool).sum(axis=-1) for m in masks)
    tucker = _rank_array(tucker_ranks, d, C, 'tucker_ranks')
    tt = _rank_array(tt_ranks, d + 1, C, 'tt_ranks')
    if np.any(tt[0] != 1) or np.any(tt[-1] != 1):
        raise ValueError("the target boundary TT ranks are not 1; a frame's boundary bonds are 1")
    target_stores = tuple(np.broadcast_to(s, o.shape) for s, o in zip(
        ranks.compute_orthogonal_representation_ranks(shape, tucker, tt), old))
    for name, s, o, W in zip(('up', 'down', 'left', 'right'), target_stores, old, level_widths):
        if np.any(s < o):
            raise ValueError("the target %s stores are below the old frame's: the target ranks must contain the old "
                             "structure" % name)
        if np.any(s > W):
            raise ValueError('a target %s store (%d) exceeds its level width %d' % (name, int(s.max()), W))
    dtype = U0.dtype
    if max_passes == 0:                                         # the hint-free start at the target, compacted
        start, _ = ufv_padded_frame(old_frame_data, tucker, tt, sharing=sharing, seed=seed)
        return ufv_compact_frame(start, level_widths), np.full((0,) + C, np.nan, dtype=dtype)
    if all(np.array_equal(s, o) for s, o in zip(target_stores, old)):   # nothing to complete
        return ufv_compact_frame(old_frame_data, level_widths), np.full((max_passes,) + C, np.nan, dtype=dtype)

    working = ranks.oversampled_ranks(shape, tucker, tt, oversampling, sharing)
    working_stores = ranks.compute_orthogonal_representation_ranks(shape, *working)
    same = all(np.array_equal(np.broadcast_to(w, t.shape), t) for w, t in zip(working_stores, target_stores))
    n_over = min(0 if same else oversampled_passes, max_passes - 1)   # the cap counts the truncating pass
    schedule = (tuple(_nested_tuple(np.asarray(r)) for r in working), (_nested_tuple(tucker), _nested_tuple(tt)),
                None if sharing is None else tuple(sharing), int(seed), int(n_over), int(max_passes), bool(alternate),
                level_widths)
    dynamic, skeleton = partition_static((old_frame_data, hint))
    if jitted:
        out = _jitted(_ufv_aligned_loop_kernel, (2, 3, 4, 5))(dynamic, tol, skeleton, project, schedule, True)
    else:
        out = _ufv_aligned_loop_kernel(dynamic, tol, skeleton, project, schedule, False)
    level_masks = ufv_masking.ufv_make_frame_masks(*target_stores, *level_widths)
    return tuple(out[:4]) + (shape, level_masks), out[4]
