# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The padded frame at a zero-padded rank increase, on ragged frame data.

``fv_empty_frame`` is the frame of the zero tensor. ``fv_padded_frame`` is the seeded start: an old frame in the
leading slots of every store, and the new slots completed by a data-independent convention. The ``ALIGN_*``
constants are the defaults of the alignment schedule, named once here.
"""
import typing as typ
import numpy as np

import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.sharing as sharing_module
from t3toolbox.backend.common import *
from t3toolbox.backend.fv_operations import ZERO_FRAME_SEED

__all__ = [
    'ALIGN_OVERSAMPLING',
    'ALIGN_OVERSAMPLED_PASSES',
    'ALIGN_MAX_PASSES',
    'ALIGN_TOL',
    'ALIGN_ALTERNATE',
    'fv_empty_frame',
    'fv_padded_frame',
]

FrameData = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray], typ.Sequence[NDArray], typ.Sequence[NDArray]]
#   (up, down, left, right), each len=d, elm_shapes C+(nUi, Ni), C+(rLi, nDi, rR(i+1)), C+(rLi, nUi, rL(i+1)), C+(rRi, nUi, rR(i+1))

ALIGN_OVERSAMPLING = 5          # the alignment loops' defaults, named once; the members and the frontend twins use them
ALIGN_OVERSAMPLED_PASSES = 2
ALIGN_MAX_PASSES = 10
ALIGN_TOL = 0.05
ALIGN_ALTERNATE = True

_STORE_NAMES = ('up', 'down', 'left', 'right')


def _frame_core_shapes(
        shape:        typ.Sequence[int],   # len=d, the mode sizes (N0, ..., N(d-1))
        up_ranks:     typ.Sequence[int],   # len=d, nU
        down_ranks:   typ.Sequence[int],   # len=d, nD
        left_ranks:   typ.Sequence[int],   # len=d+1, rL
        right_ranks:  typ.Sequence[int],   # len=d+1, rR
        stack_shape:  typ.Sequence[int],   # C
) -> typ.Tuple[
    typ.Tuple[typ.Tuple[int, ...], ...],   # up.    len=d, each C+(nUi, Ni)
    typ.Tuple[typ.Tuple[int, ...], ...],   # down.  len=d, each C+(rLi, nDi, rR(i+1))
    typ.Tuple[typ.Tuple[int, ...], ...],   # left.  len=d, each C+(rLi, nUi, rL(i+1))
    typ.Tuple[typ.Tuple[int, ...], ...],   # right. len=d, each C+(rRi, nUi, rR(i+1))
]:
    """The core shapes of a frame with these stores (the frame's ``t3_operations.t3_core_shapes``)."""
    vs = tuple(stack_shape)
    d = len(shape)
    up = tuple(vs + (up_ranks[i], shape[i]) for i in range(d))
    down = tuple(vs + (left_ranks[i], down_ranks[i], right_ranks[i + 1]) for i in range(d))
    left = tuple(vs + (left_ranks[i], up_ranks[i], left_ranks[i + 1]) for i in range(d))
    right = tuple(vs + (right_ranks[i], up_ranks[i], right_ranks[i + 1]) for i in range(d))
    return up, down, left, right


def _old_frame_stores(
        old_frame:  FrameData,   # stack C
) -> typ.Tuple[
    typ.Tuple[int, ...],   # nU. len=d
    typ.Tuple[int, ...],   # nD. len=d
    typ.Tuple[int, ...],   # rL. len=d+1
    typ.Tuple[int, ...],   # rR. len=d+1
]:
    """The old frame's stores, from its up and left widths; ``ValueError`` unless every core has their layout."""
    up, down, left, right = old_frame
    d = len(up)
    if not len(down) == len(left) == len(right) == d:
        raise ValueError('old_frame: the four stores need one core per mode; got lengths %r'
                         % (tuple(len(cores) for cores in old_frame),))
    shape = tuple(int(U.shape[-1]) for U in up)
    stack_shape = tuple(up[0].shape[:-2])
    up_widths = tuple(int(U.shape[-2]) for U in up)
    left_widths = tuple(int(G.shape[-3]) for G in left) + (int(left[-1].shape[-1]),)
    stores = ranks.compute_orthogonal_representation_ranks(shape, up_widths, left_widths)
    expected = _frame_core_shapes(shape, *stores, stack_shape)
    for name, cores, core_shapes in zip(_STORE_NAMES, old_frame, expected):
        for i, (core, core_shape) in enumerate(zip(cores, core_shapes)):
            if tuple(core.shape) != core_shape:
                raise ValueError(
                    'old_frame %s[%d] has shape %r; an orthogonal representation with up widths %r and left widths '
                    '%r (stack %r) has %r there' % (name, i, tuple(core.shape), up_widths, left_widths,
                                                    stack_shape, core_shape))
    return stores


def _haar_seed_rows(
        k:     int,   # the number of new rows
        N:     int,   # the mode size
        key:   int,   # the stream key: the sharing group's first mode
        seed:  int,
) -> NDArray:         # HOST float64, shape=(k, N): the stream's first k rows
    """The Tucker completion's seeds: the first ``k`` rows of the keyed stream ``RandomState([seed, key])``."""
    return np.random.RandomState([seed, key]).standard_normal((k, N))


def _coordinate_seed_order(
        p1:  int,   # the unfolding's row axes, (p1, p2), row-major
        p2:  int,
        q1:  int,   # the old block's extent on them, (q1, q2) <= (p1, p2)
        q2:  int,
) -> NDArray:       # HOST int, shape=(p1*p2 - q1*q2,): the row indices outside the old block, in row-major order
    """The coordinate seeds of a TT or down completion outside the old block, in the order they are used."""
    outside = np.ones((p1, p2), dtype=bool)
    outside[:q1, :q2] = False
    return np.flatnonzero(outside)


def _embed(
        A:      NDArray,             # C+(a1, ..., ak)
        shape:  typ.Sequence[int],   # len=k, (t1, ..., tk), each t >= a
) -> NDArray:                        # C+(t1, ..., tk): A in the leading slots, zeros elsewhere
    """An old core in the leading slots of a larger core."""
    nC = A.ndim - len(shape)
    return linalg.pad_or_truncate(A, [(0, 0)] * nC + [(0, t - a) for t, a in zip(shape, A.shape[nC:])])


def _seeded_completion(
        old_C_xyc:  NDArray,   # C+(q1, q2, c): an old core, its two unfolding row axes first, its c old columns last
        p1:         int,       # the padded row axes, (p1, p2) >= (q1, q2)
        p2:         int,
        count:      int,       # the number of new columns
) -> NDArray:                  # C+(p1, p2, count): orthonormal, and orthogonal to the old columns embedded in (p1, p2)
    """New columns of one unfolding: unit vectors outside the old block, then a completion inside it."""
    xnp, _, _ = get_backend(False, tree_contains_jax(old_C_xyc))
    C = tuple(old_C_xyc.shape[:-3])
    q1, q2, c = (int(n) for n in old_C_xyc.shape[-3:])
    outside = _coordinate_seed_order(p1, p2, q1, q2) if c else _coordinate_seed_order(p1, p2, 0, 0)
    t = min(count, outside.size)
    units_xyt = np.zeros((p1 * p2, t))
    units_xyt[outside[:t], np.arange(t)] = 1.0                       # placed exactly: disjoint from the old block
    units_xyt = units_xyt.reshape(p1, p2, t)
    parts = [xnp.broadcast_to(xnp.asarray(units_xyt, dtype=old_C_xyc.dtype), C + units_xyt.shape)]
    k_in = count - t
    if k_in:                                                         # outside exhausted: complete inside the block
        P_Cxc = old_C_xyc.reshape(C + (q1 * q2, c))
        E_xk = np.eye(q1 * q2)[:, :k_in]
        E_Cxk = xnp.broadcast_to(xnp.asarray(E_xk, dtype=old_C_xyc.dtype), C + E_xk.shape)
        new_Cxk = linalg.pinned_completion(P_Cxc, E_Cxk, k_in)
        parts.append(_embed(new_Cxk.reshape(C + (q1, q2, k_in)), (p1, p2, k_in)))
    return xnp.concatenate(parts, axis=-1)


def _padded_core(
        old:     NDArray,                     # C+(q1, q2, q3): an old TT or down core
        widths:  typ.Tuple[int, int, int],    # the padded core's widths, each >= old's
        axis:    int,                         # -3 | -2 | -1: the output axis; the other two, in order, index the rows
) -> NDArray:                                 # C+widths: old in the leading slots, then the completed new output slots
    """A TT or down core of the padded frame: the old core embedded, its new output slots the hint-free completion."""
    xnp, _, _ = get_backend(False, tree_contains_jax(old))
    row_widths = list(widths)
    out_width = row_widths.pop(axis)
    old_C_xyc = xnp.moveaxis(old, axis, -1)
    c = int(old_C_xyc.shape[-1])
    emb_C_xyc = _embed(old_C_xyc, tuple(row_widths) + (c,))
    new_C_xyk = _seeded_completion(old_C_xyc, row_widths[0], row_widths[1], out_width - c)
    return xnp.moveaxis(xnp.concatenate([emb_C_xyc, new_C_xyk], axis=-1), -1, axis)


def _padded_tucker_factor(
        U0:    NDArray,   # C+(s, N): an old Tucker factor
        n:     int,       # the padded Tucker width, n >= s
        key:   int,       # the stream key: the sharing group's first mode
        seed:  int,
) -> NDArray:             # C+(n, N): U0's rows, then n - s Haar rows completed against them
    """A Tucker factor of the padded frame: the old rows, then the keyed Haar seeds completed against them."""
    xnp, _, _ = get_backend(False, tree_contains_jax(U0))
    C = tuple(U0.shape[:-2])
    s, N = (int(m) for m in U0.shape[-2:])
    seeds_Nk = _haar_seed_rows(n - s, N, key, seed).T
    seeds_CNk = xnp.broadcast_to(xnp.asarray(seeds_Nk, dtype=U0.dtype), C + seeds_Nk.shape)
    new_CNk = linalg.pinned_completion(U0.swapaxes(-2, -1), seeds_CNk, n - s)
    return xnp.concatenate([U0, new_CNk.swapaxes(-2, -1)], axis=-2)


def fv_empty_frame(
        shape:        typ.Sequence[int],        # len=d, the mode sizes (N0, ..., N(d-1))
        stack_shape:  typ.Sequence[int] = (),   # C
        use_jax:      bool = False,             # array-free constructor: the flag chooses the output type
) -> typ.Tuple[                                 # the frame of the zero tensor (default float), b = (1, 0, ..., 0, 1):
    typ.Tuple[NDArray, ...],    # up.    len=d, elm_shape=C+(0, Ni)
    typ.Tuple[NDArray, ...],    # down.  len=d, elm_shape=C+(bi, 0, b(i+1))
    typ.Tuple[NDArray, ...],    # left.  len=d, elm_shape=C+(bi, 0, b(i+1))
    typ.Tuple[NDArray, ...],    # right. len=d, elm_shape=C+(bi, 0, b(i+1))
]:
    """The frame of the zero tensor: the orthogonal representation's layout at Tucker ranks 0 and TT ranks ``b``.

    Examples
    --------
    Every rank axis has width 0 except the boundary bonds:

    >>> import t3toolbox.backend.padded_frames as padded_frames
    >>> up, down, left, right = padded_frames.fv_empty_frame((5, 6, 7))
    >>> print([U.shape for U in up], [G.shape for G in left])
    [(0, 5), (0, 6), (0, 7)] [(1, 0, 0), (0, 0, 0), (0, 0, 1)]

    It is the orthogonal representation of the rank-0 Tucker tensor train, here with a stack ``C = (2,)``:

    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.fv_conversions as fvc
    >>> x = t3.TuckerTensorTrain.zeros((5, 6, 7), (0, 0, 0), (1, 0, 0, 1), stack_shape=(2,))
    >>> frame = fvc.t3_orthogonal_representations(x.data)[0]
    >>> empty = padded_frames.fv_empty_frame((5, 6, 7), stack_shape=(2,))
    >>> print(all(a.shape == b.shape for f, g in zip(frame, empty) for a, b in zip(f, g)))
    True
    """
    xnp, _, _ = get_backend(False, use_jax)
    d = len(shape)
    stores = ranks.compute_orthogonal_representation_ranks(shape, (0,) * d, (1,) + (0,) * (d - 1) + (1,))
    return tuple(tuple(xnp.zeros(s) for s in core_shapes)
                 for core_shapes in _frame_core_shapes(shape, *stores, stack_shape))


def fv_padded_frame(
        old_frame:     FrameData,                          # the old frame, stack C (fv_empty_frame for the zero tensor)
        tucker_ranks:  typ.Sequence[int],                  # len=d, target Tucker ranks (stores >= the old frame's)
        tt_ranks:      typ.Sequence[int],                  # len=d+1, target TT ranks (boundary 1)

        sharing:       typ.Optional[typ.Sequence] = None,  # len=d group labels; None = unshared
        seed:          int = ZERO_FRAME_SEED,              # the keyed Haar stream of the Tucker completions
) -> typ.Tuple[                 # (nU, nD, rL, rR) = compute_orthogonal_representation_ranks(shape, tucker_ranks, tt_ranks)
    typ.Tuple[NDArray, ...],    # up.    len=d, elm_shape=C+(nUi, Ni): the old rows bitwise, then the Haar completion
    typ.Tuple[NDArray, ...],    # down.  len=d, elm_shape=C+(rLi, nDi, rR(i+1)): the old block bitwise, then seeded
    typ.Tuple[NDArray, ...],    # left.  len=d, elm_shape=C+(rLi, nUi, rL(i+1)); left[d-1] = the old carrier, zero-padded
    typ.Tuple[NDArray, ...],    # right. len=d, elm_shape=C+(rRi, nUi, rR(i+1)); right[0] = the old carrier, zero-padded
]:
    """The seeded padded frame: the old frame in the leading slots, the new slots completed by the hint-free convention."""
    xnp, _, _ = get_backend(False, tree_contains_jax(old_frame))
    up0, down0, left0, right0 = (tuple(xnp.asarray(c) for c in cores) for cores in old_frame)
    old_stores = _old_frame_stores((up0, down0, left0, right0))
    d = len(up0)
    shape = tuple(int(U.shape[-1]) for U in up0)

    new_stores = ranks.compute_orthogonal_representation_ranks(shape, tuple(tucker_ranks), tuple(tt_ranks))
    if tt_ranks[0] != 1 or tt_ranks[-1] != 1:
        raise ValueError('the target boundary TT ranks are (%d, %d); a frame\'s boundary bonds are 1'
                         % (tt_ranks[0], tt_ranks[-1]))
    for name, old, new in zip(_STORE_NAMES, old_stores, new_stores):
        for i, (a, b) in enumerate(zip(old, new)):
            if b < a:
                raise ValueError('the target %s store %d is %d, below the old frame\'s %d: the target ranks must '
                                 'contain the old structure' % (name, i, b, a))

    groups = tuple((i,) for i in range(d)) if sharing is None else sharing_module.validate_sharing(sharing, shape)
    for group in groups:
        for which, stores in (('old frame', old_stores), ('target', new_stores)):
            widths = tuple(stores[0][i] for i in group)
            if len(set(widths)) != 1:
                raise ValueError('sharing group %r has unequal Tucker stores %r in the %s; a shared factor needs one '
                                 'width per group' % (group, widths, which))

    nU, nD, rL, rR = new_stores
    first = tuple(next(group[0] for group in groups if i in group) for i in range(d))   # each mode's stream key
    factors = tuple(_padded_tucker_factor(up0[i], nU[i], i, seed) if first[i] == i else None for i in range(d))
    up = tuple(factors[first[i]] for i in range(d))                                   # one array per group
    down = tuple(_padded_core(down0[i], (rL[i], nD[i], rR[i + 1]), -2) for i in range(d))
    left = (tuple(_padded_core(left0[i], (rL[i], nU[i], rL[i + 1]), -1) for i in range(d - 1))
            + (_embed(left0[d - 1], (rL[d - 1], nU[d - 1], rL[d])),))                 # the carrier: old content only
    right = ((_embed(right0[0], (rR[0], nU[0], rR[1])),)
             + tuple(_padded_core(right0[i], (rR[i], nU[i], rR[i + 1]), -3) for i in range(1, d)))
    return up, down, left, right
