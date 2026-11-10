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
by ``linalg.pad_safe_pinned_completion``, one batched completion per store.
"""
import typing as typ
import numpy as np

import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.sharing as sharing_module
import t3toolbox.backend.ufv_masking as ufv_masking
from t3toolbox.backend.common import *
from t3toolbox.backend.fv_operations import ZERO_FRAME_SEED
from t3toolbox.backend.padded_frames import _haar_seed_rows

__all__ = [
    'ufv_empty_frame',
    'ufv_padded_frame',
]

UFrameData = typ.Tuple[NDArray, NDArray, NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray, NDArray, NDArray]]
#   up (d,)+C+(nU, N), down (d,)+C+(rL, nD, rR), left (d,)+C+(rL, nU, rL), right (d,)+C+(rR, nU, rR), shape (len=d),
#   masks HOST bool (up (d,)+C+(nU,), down (d,)+C+(nD,), left (d+1,)+C+(rL,), right (d+1,)+C+(rR,)); block layout: nU = nD, rL = rR


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
