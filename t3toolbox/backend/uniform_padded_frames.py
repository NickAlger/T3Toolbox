# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The padded frame at a zero-padded rank increase, on uniform frame data, in BLOCK LAYOUT.

Every rank slot of a padded frame is ``[A-block | P-block]``: the old frame's supercores as given (any mask pattern,
junk masked to zero once), then the completion, each padded to one width with its own mask. Tucker slots are
``[Na | Ka]`` and bond slots ``[Rb | Kb]``, so the only positional fact is the split ``(Na, Rb)``.

``ufv_empty_frame`` is the frame of the zero tensor (``Na = 0``, ``Rb = 1``). ``ufv_padded_frame`` is the seeded start,
the uniform twin of ``padded_frames.fv_padded_frame``: the same seeds (keyed Haar rows per sharing group; coordinate
directions at the real rows of each unfolding in their ragged row-major order, outside the old block first), completed
by ``linalg.pad_safe_pinned_completion``, one batched completion per store. ``utv_align_padded_frame`` is one alignment
step, the uniform twin of ``padded_frames.tv_align_padded_frame``: the maps over modes are batched contractions, the
sweeps scans with module-level bodies, and every edge's completion one ``linalg.pad_safe_svd``. The host prepares the
masks, selections and group tables; the arithmetic is one jittable kernel, run eagerly here.
"""
import typing as typ
import numpy as np

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.sharing as sharing_module
import t3toolbox.backend.ufv_masking as ufv_masking
from t3toolbox.backend.common import *
from t3toolbox.backend.fv_operations import ZERO_FRAME_SEED
from t3toolbox.backend.padded_frames import (_haar_seed_rows, _unfold, _fold, _root, _new_coordinates, _edge_content,
                                             _left_message_step, _right_message_step)

__all__ = [
    'ufv_empty_frame',
    'ufv_padded_frame',
    'utv_align_padded_frame',
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


def _unfold(
        X:     NDArray,   # (...,)+(a, n, b)
        axis:  int,       # -3 | -2 | -1: the slot that becomes the columns
) -> NDArray:             # (...,)+(rows, cols): the other two slots, row-major, as the rows
    """A core unfolded with one slot as the columns (core axes only)."""
    xnp, _, _ = get_backend(True, tree_contains_jax(X))
    Y = xnp.moveaxis(X, axis, -1)
    return Y.reshape(Y.shape[:-3] + (Y.shape[-3] * Y.shape[-2], Y.shape[-1]))


def _units(
        rows:      NDArray,   # HOST bool, (...,)+(n,): the candidate rows
        j:         NDArray,   # HOST int, (...,)+(w,): column c is the unit vector at the j[c]-th True row
        selected:  NDArray,   # HOST bool, (...,)+(w,): the columns to fill
) -> NDArray:                 # HOST float, (...,)+(n, w): zero outside the selected columns
    """Unit columns at the True rows, in row-major order."""
    n = rows.shape[-1]
    if n == 0:
        return np.zeros(rows.shape[:-1] + (0, j.shape[-1]))
    order = np.argsort(~rows, axis=-1, kind='stable')                      # the True rows first, in order
    at = np.take_along_axis(order, np.clip(j, 0, n - 1), axis=-1)
    return ((np.arange(n)[:, None] == at[..., None, :]) & selected[..., None, :]).astype(float)


def _coordinate_seeds(
        rows_a:     NDArray,   # HOST bool, (M,)+C+(wa,): the unfolding's first row slot, real slots
        split_a:    int,       # the A-block width of that slot
        rows_b:     NDArray,   # HOST bool, (M,)+C+(wb,): the second row slot (a-major rows)
        split_b:    int,
        has_old:    NDArray,   # HOST bool, (M,)+C: the old block has columns
        count:      NDArray,   # HOST int, (M,)+C: the completion's columns
        width:      int,       # the completion block's width, >= every count
) -> typ.Tuple[
    NDArray,   # outside units. HOST float, (M,)+C+(wa*wb, width): columns 0..t-1, the rest zero
    NDArray,   # inside seeds.  HOST float, (M,)+C+(split_a*split_b, width): columns t..count-1 on the old block's sub-grid
    NDArray,   # inside seed mask. HOST bool, (M,)+C+(width,): columns t..count-1
    NDArray,   # the sub-grid's real rows. HOST bool, (M,)+C+(split_a*split_b,)
]:
    """The coordinate seeds from the masks: the real rows outside the old block, then inside it, in flat slot order."""
    lead = rows_a.shape[:-1]                                               # flat slot order = the ragged row-major order:
    wa, wb, sa, sb = rows_a.shape[-1], rows_b.shape[-1], split_a, split_b  #   the masks keep the real slots in order
    real = (rows_a[..., :, None] & rows_b[..., None, :]).reshape(lead + (wa * wb,))
    block = ((np.arange(wa) < sa)[:, None] & (np.arange(wb) < sb)[None, :]).reshape(wa * wb)
    outside = real & ~(block & has_old[..., None])
    t = np.minimum(count, outside.sum(axis=-1))                            # the rest fit inside: the real rows bound
    j = np.broadcast_to(np.arange(width), lead + (width,))                 #   the target store
    real_in = (rows_a[..., :sa, None] & rows_b[..., None, :sb]).reshape(lead + (sa * sb,)) & has_old[..., None]
    inside_mask = (j >= t[..., None]) & (j < count[..., None])
    return (_units(outside, j, j < t[..., None]), _units(real_in, j - t[..., None], inside_mask), inside_mask, real_in)


def _store_completion(
        P:            NDArray,   # (M,)+C+(wa*wb, w): the old cores unfolded at the new widths (zero off the A-block rows)
        pinned_mask:  NDArray,   # HOST bool, (M,)+C+(w,): the old columns' mask
        rows_a:       NDArray,   # HOST bool, (M,)+C+(wa,): the new mask of the first row slot
        split_a:      int,       # its A-block width
        rows_b:       NDArray,   # HOST bool, (M,)+C+(wb,): the new mask of the second row slot
        split_b:      int,
        count:        NDArray,   # HOST int, (M,)+C: the completion counts
        width:        int,       # the completion block's width
) -> NDArray:                    # (M,)+C+(wa*wb, width): the completion, real columns first, junk bitwise zero
    """One store's completion, batched over its modes: the outside units placed exactly, then the in-block completion."""
    xnp, _, _ = get_backend(True, tree_contains_jax(P))
    lead = P.shape[:-2]
    wa, wb = rows_a.shape[-1], rows_b.shape[-1]
    units, seeds_in, mask_in, rows_in = _coordinate_seeds(rows_a, split_a, rows_b, split_b,
                                                          pinned_mask.sum(axis=-1) > 0, count, width)
    out = xnp.asarray(units, dtype=P.dtype)
    if split_a * split_b == 0 or not mask_in.any():
        return out
    P_in = P.reshape(lead + (wa, wb, P.shape[-1]))[..., :split_a, :split_b, :].reshape(
        lead + (split_a * split_b, P.shape[-1]))
    new_in = linalg.pad_safe_pinned_completion(P_in, pinned_mask, xnp.asarray(seeds_in, dtype=P.dtype), mask_in, rows_in)
    return out + _pad_trailing(new_in.reshape(lead + (split_a, split_b, width)), (wa, wb, width)).reshape(
        lead + (wa * wb, width))


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
        seed:            int = ZERO_FRAME_SEED,                       # the keyed Haar stream
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
    seeds = np.zeros((len(groups), N, Ka))
    for g, f in enumerate(first):                                       # host constants: each group's keyed stream
        seeds[g, :shape[f], :] = _haar_seed_rows(Ka, shape[f], int(f), seed).T
    mN = prefix_mask(np.asarray(shape), N)                              # (d, N)
    new_U = linalg.pad_safe_pinned_completion(
        U0[first].swapaxes(-2, -1), mU0[first], xnp.asarray(seeds.reshape((len(groups),) + (1,) * len(C) + (N, Ka)),
                                                              dtype=dtype),
        prefix_mask(kU[first], Ka), mN[first].reshape((len(groups),) + (1,) * len(C) + (N,)))
    U = xnp.concatenate([U0[first[group_of]], new_U[group_of].swapaxes(-2, -1)], axis=-2)   # tied within groups

    # -- left cores 0 .. d-2: new columns in the (a, n) unfolding; left[d-1] is the carrier
    P_L = _unfold(_pad_trailing(L0[:d - 1], (rT, nT, Rb)), -1)
    new_L = _store_completion(P_L, mL0[1:d], mL[:d - 1], Rb, mU[:d - 1], Na, kL[1:d], Kb)
    L = xnp.concatenate([xnp.concatenate([P_L, new_L], axis=-1).reshape((d - 1,) + C + (rT, nT, rT)),
                         _pad_trailing(L0[d - 1:], (rT, nT, rT))], axis=0)
    # -- right cores 1 .. d-1: new rows in the (n, b) unfolding; right[0] is the carrier
    P_R = _unfold(_pad_trailing(R0[1:], (Rb, nT, rT)), -3)
    new_R = _store_completion(P_R, mR0[1:d], mU[1:], Na, mR[2:], Rb, kR[1:d], Kb)
    R = xnp.concatenate([_pad_trailing(R0[:1], (rT, nT, rT)),
                         xnp.moveaxis(xnp.concatenate([P_R, new_R], axis=-1).reshape((d - 1,) + C + (nT, rT, rT)),
                                      -1, -3)], axis=0)
    # -- down cores, all modes: new down slots in the (a, b) bond space
    P_O = _unfold(_pad_trailing(O0, (rT, Na, rT)), -2)
    new_O = _store_completion(P_O, mD0, mL[:-1], Rb, mR[1:], Rb, kD, Ka)
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


def _u_left_edge_step(carry, x):
    """One edge of the left sweep, the store updated (scan body)."""
    return _u_left_edge(carry, x, True)


def _u_left_edge_transport_step(carry, x):
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


def _u_right_edge_step(carry, x):
    """One edge of the right sweep, the store updated (scan body)."""
    return _u_right_edge(carry, x, True)


def _u_right_edge_transport_step(carry, x):
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
