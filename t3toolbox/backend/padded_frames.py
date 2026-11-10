# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The padded frame at a zero-padded rank increase, on ragged frame data.

``fv_empty_frame`` is the frame of the zero tensor. ``fv_padded_frame`` is the seeded start: an old frame in the
leading slots of every store, and the new slots completed by a data-independent convention.
``tv_align_padded_frame`` is one alignment step: every completion replaced by the leading new directions of a tangent
vector at the frame, the old blocks held fixed (unrooted message passing over the T3 tree's directed edges).
``fv_aligned_padded_frame`` is the loop: from the start at oversampled working ranks, steps against the projection of
one fixed tensor, a truncating pass to the target ranks, then two-colour passes in ``('UL', 'RO')`` pairs. The
``ALIGN_*`` constants are the loop's defaults, named once here. Everything runs eagerly unless the loop is asked for
jit (``use_jit``): then the old frame goes to jax and the loop calls a memoized jitted step.
"""
import functools as ft
import typing as typ
import numpy as np

import t3toolbox.backend.contractions as contractions
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
    'tv_align_padded_frame',
    'fv_aligned_padded_frame',
]

FrameData = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray], typ.Sequence[NDArray], typ.Sequence[NDArray]]
#   (up, down, left, right), each len=d, elm_shapes C+(nUi, Ni), C+(rLi, nDi, rR(i+1)), C+(rLi, nUi, rL(i+1)), C+(rRi, nUi, rR(i+1))
VariationsData = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]
#   (tucker_variations, tt_variations), each len=d, elm_shapes C+(nDi, Ni), C+(rLi, nUi, rR(i+1))

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


def _frame_stores(
        frame:  FrameData,   # stack C
        name:   str,         # the argument's name, for the messages
) -> typ.Tuple[
    typ.Tuple[int, ...],   # nU. len=d
    typ.Tuple[int, ...],   # nD. len=d
    typ.Tuple[int, ...],   # rL. len=d+1
    typ.Tuple[int, ...],   # rR. len=d+1
]:
    """A frame's stores, from its up and left widths; ``ValueError`` unless every core has their layout."""
    up, down, left, right = frame
    d = len(up)
    if not len(down) == len(left) == len(right) == d:
        raise ValueError('%s: the four stores need one core per mode; got lengths %r'
                         % (name, tuple(len(cores) for cores in frame)))
    shape = tuple(int(U.shape[-1]) for U in up)
    stack_shape = tuple(up[0].shape[:-2])
    up_widths = tuple(int(U.shape[-2]) for U in up)
    left_widths = tuple(int(G.shape[-3]) for G in left) + (int(left[-1].shape[-1]),)
    stores = ranks.compute_orthogonal_representation_ranks(shape, up_widths, left_widths)
    expected = _frame_core_shapes(shape, *stores, stack_shape)
    for store, cores, core_shapes in zip(_STORE_NAMES, frame, expected):
        for i, (core, core_shape) in enumerate(zip(cores, core_shapes)):
            if tuple(core.shape) != core_shape:
                raise ValueError(
                    '%s %s[%d] has shape %r; an orthogonal representation with up widths %r and left widths %r '
                    '(stack %r) has %r there' % (name, store, i, tuple(core.shape), up_widths, left_widths,
                                                 stack_shape, core_shape))
    return stores


def _rank_stores(
        shape:         typ.Sequence[int],   # len=d
        tucker_ranks:  typ.Sequence[int],   # len=d
        tt_ranks:      typ.Sequence[int],   # len=d+1, boundary 1
        which:         str,                 # 'target' | 'old' | 'output', for the message
) -> typ.Tuple[
    typ.Tuple[int, ...],   # nU. len=d
    typ.Tuple[int, ...],   # nD. len=d
    typ.Tuple[int, ...],   # rL. len=d+1
    typ.Tuple[int, ...],   # rR. len=d+1
]:
    """The stores of an orthogonal representation at these ranks; ``ValueError`` on boundary TT ranks other than 1."""
    stores = ranks.compute_orthogonal_representation_ranks(shape, tuple(tucker_ranks), tuple(tt_ranks))
    if tt_ranks[0] != 1 or tt_ranks[-1] != 1:
        raise ValueError('the %s boundary TT ranks are (%d, %d); a frame\'s boundary bonds are 1'
                         % (which, tt_ranks[0], tt_ranks[-1]))
    return stores


def _require_contains(
        old_stores:     typ.Tuple[typ.Tuple[int, ...], ...],   # (nU, nD, rL, rR) of the old frame
        target_stores:  typ.Tuple[typ.Tuple[int, ...], ...],   # (nU, nD, rL, rR) of the target
) -> None:
    """``ValueError`` unless every target store is at least the old frame's."""
    for name, old, new in zip(_STORE_NAMES, old_stores, target_stores):
        for i, (a, b) in enumerate(zip(old, new)):
            if b < a:
                raise ValueError('the target %s store %d is %d, below the old frame\'s %d: the target ranks must '
                                 'contain the old structure' % (name, i, b, a))


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
    """The seeded padded frame: the old frame in the leading slots, the new slots by the hint-free convention."""
    xnp, _, _ = get_backend(False, tree_contains_jax(old_frame))
    up0, down0, left0, right0 = (tuple(xnp.asarray(c) for c in cores) for cores in old_frame)
    old_stores = _frame_stores((up0, down0, left0, right0), 'old_frame')
    d = len(up0)
    shape = tuple(int(U.shape[-1]) for U in up0)

    new_stores = _rank_stores(shape, tucker_ranks, tt_ranks, 'target')
    _require_contains(old_stores, new_stores)

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


# --------------------------------------------------------------------------------------------------
# One alignment step. Every directed edge of the T3 tree (leaf -> core, core -> core both ways, core -> leaf) gets a
# new completion: the leading directions of the tangent vector's content at that edge, off the pinned old block.
# The content is assembled from messages passed inward from the far side of the edge (step 1, at the old
# completions) and from the new completions of the edges feeding it (step 2), with no root.
# --------------------------------------------------------------------------------------------------
def _unfold(
        X:     NDArray,   # C+(a, n, b)
        axis:  int,       # -3 | -2 | -1: the slot that becomes the columns
) -> NDArray:             # C+(rows, cols): the other two slots, row-major, as the rows
    """A core unfolded with one slot as the columns (core axes only)."""
    xnp, _, _ = get_backend(False, tree_contains_jax(X))
    Y = xnp.moveaxis(X, axis, -1)
    return Y.reshape(Y.shape[:-3] + (Y.shape[-3] * Y.shape[-2], Y.shape[-1]))


def _fold(
        F:            NDArray,             # C+(rows, cols): an unfolding
        core_widths:  typ.Sequence[int],   # len=3, the core's (a, n, b); the entry at `axis` becomes cols
        axis:         int,                 # -3 | -2 | -1: the slot the columns become
) -> NDArray:                              # C+(a, n, b) with cols at `axis`
    """The inverse of :py:func:`_unfold`."""
    xnp, _, _ = get_backend(False, tree_contains_jax(F))
    rows = tuple(int(n) for j, n in enumerate(core_widths) if j != 3 + axis)
    return xnp.moveaxis(F.reshape(F.shape[:-2] + rows + (F.shape[-1],)), -1, axis)


def _root(
        M:  NDArray,   # C+(p, q)
) -> NDArray:          # C+(q, min(p, q)): Xi with Xi Xi^T = M^T M
    """A square root of a Gram matrix, from the thin SVD (any factor will do; only Xi Xi^T is used)."""
    xnp, _, _ = get_backend(False, tree_contains_jax(M))
    _, ss, Vh = xnp.linalg.svd(M, full_matrices=False)
    return Vh.swapaxes(-2, -1) * ss[..., None, :]


def _new_coordinates(
        X:     NDArray,   # C+(a, n, b): a core in its old slot coordinates
        o:     NDArray,   # C+(k1, k): P+^T P of the frame indexing slot `axis`
        axis:  int,       # -3 | -2 | -1
) -> NDArray:             # C+(a', n', b'): slot `axis` in the new coordinates, width X's - k + k1
    """Slot ``axis`` re-expressed in the new coordinates: its old block kept, its completion mapped by ``o``."""
    xnp, _, _ = get_backend(False, tree_contains_jax((X, o)))
    r = X.shape[axis] - o.shape[-1]
    if axis == -3:
        head, tail = X[..., :r, :, :], contractions.contract('Cxa,Canb->Cxnb', o, X[..., r:, :, :])
    elif axis == -2:
        head, tail = X[..., :r, :], contractions.contract('Cxn,Canb->Caxb', o, X[..., r:, :])
    else:
        head, tail = X[..., :r], contractions.contract('Cxb,Canb->Canx', o, X[..., r:])
    return xnp.concatenate([head, tail], axis=axis)


def _edge_content(
        g_old:   NDArray,                   # C+(m, k): the old completion in the new coordinates
        g_news:  typ.Sequence[NDArray],     # len=members, elm_shape=C+(m, kf): near terms on the far completion
        PhiPs:   typ.Sequence[NDArray],     # len=members, elm_shape=C+(k, kf): the far messages, completion part
        Xis:     typ.Sequence[NDArray],     # len=members, elm_shape=C+(k, z): the far messages' Gram roots
) -> NDArray:                               # C+(m, sum of (kf + z)): Y, the members side by side
    """An edge's normal content ``Y = [g_old Phi_P + g_new | g_old Xi]``, one block pair per sharing-group member."""
    xnp, _, _ = get_backend(False, tree_contains_jax((g_old, g_news, PhiPs, Xis)))
    parts = []
    for g_new, PhiP, Xi in zip(g_news, PhiPs, Xis):
        parts.append(contractions.contract('Cik,Ckb->Cib', g_old, PhiP) + g_new)
        parts.append(contractions.contract('Cik,Ckb->Cib', g_old, Xi))
    return xnp.concatenate(parts, axis=-1)


def _edge(
        Abar:   NDArray,   # C+(m, r): the pinned old block, orthonormal columns
        g_old:  NDArray,   # C+(m, k): the old completion in the new coordinates
        Y:      NDArray,   # C+(m, q): the content (g_old itself when the store is transported)
        k1:     int,       # the new completion's width, r + k1 <= m
) -> typ.Tuple[
    NDArray,   # P+. C+(m, k1): the leading directions of (I - Abar Abar^T) Y, completed against Abar
    NDArray,   # o.  C+(k1, k): P+^T g_old
    NDArray,   # movement. shape=C: ||g_old - P+ o||_F
]:
    """One directed edge's new completion, its coordinate map, and its movement."""
    xnp, _, _ = get_backend(False, tree_contains_jax((Abar, g_old, Y)))
    P = linalg.pinned_leading_columns(Abar, Y, k1)
    o = contractions.contract('Cia,Cib->Cab', P, g_old)
    out = g_old - contractions.contract('Cia,Cab->Cib', P, o)
    return P, o, xnp.sqrt(xnp.sum(out * out, axis=(-2, -1)))


def _down_message(
        x: typ.Tuple[NDArray, NDArray],   # (VP, U): C+(kDi, Ni), the variation's completion rows ; C+(nUi, Ni)
) -> typ.Tuple[
    NDArray,   # PhiO. C+(kD, nU)
    NDArray,   # XiO.  C+(kD, z)
]:
    """The message of the edge core -> leaf (map body)."""
    VP, U = x
    PhiO = contractions.contract('Ckn,Cun->Cku', VP, U)
    return PhiO, _root(VP.swapaxes(-2, -1) - contractions.contract('Cun,Cku->Cnk', U, PhiO))


def _left_message_step(
        carry:  typ.Tuple[NDArray, NDArray],   # (PhiL, XiL), edge c_j -> c_(j+1): C+(kL(j+1), rR(j+1)) ; C+(kL(j+1), z)
        x:      typ.Tuple[NDArray, ...],       # (H_P, bO, bL, R, PhiO, XiO) at core j: C+(kLj, nUj, rR(j+1)) ;
                                               #   C+(kLj, kDj, rR(j+1)); C+(kLj, nUj, kL(j+1)); C+(rRj, nUj, rR(j+1)) ;
                                               #   C+(kDj, nUj) ; C+(kDj, z)
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],               # (PhiL, XiL) of the edge c_(j-1) -> c_j
    typ.Tuple[NDArray, NDArray],               # the same, collected
]:
    """One core of the leftward message sweep (scan body; run on reversed inputs)."""
    xnp, _, _ = get_backend(False, tree_contains_jax((carry, x)))
    PhiL, XiL = carry
    H_P, bO, bL, R, PhiO, XiO = x
    Vb = (H_P + contractions.contract('Cxkb,Ckn->Cxnb', bO, PhiO)
          + contractions.contract('Cxnk,Ckb->Cxnb', bL, PhiL))
    rows = [_unfold(contractions.contract('Cxkb,Ckz->Cxzb', bO, XiO), -3),
            _unfold(contractions.contract('Cxnk,Ckz->Cxnz', bL, XiL), -3)]
    PhiL_next = contractions.contract('Cxnb,Canb->Cxa', Vb, R)
    W = _unfold(Vb - contractions.contract('Cxa,Canb->Cxnb', PhiL_next, R), -3)
    out = (PhiL_next, _root(xnp.concatenate([W] + rows, axis=-2)))
    return out, out


def _right_message_step(
        carry:  typ.Tuple[NDArray, NDArray],   # (PhiR, XiR) of the edge c_j -> c_(j-1): C+(kRj, rLj) ; C+(kRj, z)
        x:      typ.Tuple[NDArray, ...],       # (H_P, bO, bR, L, PhiO, XiO) at core j: C+(rLj, nUj, kR(j+1)) ;
                                               #   C+(rLj, kDj, kR(j+1)); C+(kRj, nUj, kR(j+1)); C+(rLj, nUj, rL(j+1)) ;
                                               #   C+(kDj, nUj) ; C+(kDj, z)
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],               # (PhiR, XiR) of the edge c_(j+1) -> c_j
    typ.Tuple[NDArray, NDArray],               # the same, collected
]:
    """One core of the rightward message sweep (scan body)."""
    xnp, _, _ = get_backend(False, tree_contains_jax((carry, x)))
    PhiR, XiR = carry
    H_P, bO, bR, L, PhiO, XiO = x
    Vb = (H_P + contractions.contract('Cakx,Ckn->Canx', bO, PhiO)
          + contractions.contract('Cknx,Cka->Canx', bR, PhiR))
    rows = [_unfold(contractions.contract('Cakx,Ckz->Cazx', bO, XiO), -1),
            _unfold(contractions.contract('Cknx,Ckz->Cznx', bR, XiR), -1)]
    PhiR_next = contractions.contract('Canx,Canb->Cxb', Vb, L)
    W = _unfold(Vb - contractions.contract('Cxb,Canb->Canx', PhiR_next, L), -1)
    out = (PhiR_next, _root(xnp.concatenate([W] + rows, axis=-2)))
    return out, out


def _up_message(
        x: typ.Tuple[NDArray, ...],   # (H_P, bL, bR, O, PhiL, XiL, PhiR, XiR) at core i: C+(rLi, kUi, rR(i+1)) ;
                                      #   C+(rLi, kUi, kL(i+1)) ; C+(kRi, kUi, rR(i+1)) ; C+(rLi, nDi, rR(i+1)) ;
                                      #   C+(kL(i+1), rR(i+1)) ; C+(kL(i+1), z) ; C+(kRi, rLi) ; C+(kRi, z)
) -> typ.Tuple[
    NDArray,   # PhiU. C+(kU, nD)
    NDArray,   # XiU.  C+(kU, z)
]:
    """The message of the edge leaf -> core (map body)."""
    xnp, _, _ = get_backend(False, tree_contains_jax(x))
    H_P, bL, bR, O, PhiL, XiL, PhiR, XiR = x
    Vb = (H_P + contractions.contract('Caxk,Ckb->Caxb', bL, PhiL)
          + contractions.contract('Ckxb,Cka->Caxb', bR, PhiR))
    rows = [_unfold(contractions.contract('Caxk,Ckz->Caxz', bL, XiL), -2),
            _unfold(contractions.contract('Ckxb,Ckz->Czxb', bR, XiR), -2)]
    PhiU = contractions.contract('Caxb,Caub->Cxu', Vb, O)
    W = _unfold(Vb - contractions.contract('Cxu,Caub->Caxb', PhiU, O), -2)
    return PhiU, _root(xnp.concatenate([W] + rows, axis=-2))


def _up_edge(
        x: typ.Tuple,   # (A, P_old, g_news, PhiPs, XiUs, k1, on), one sharing group: C+(s, N) ; C+(k, N) ;
                        #   per member C+(N, kDi) ; C+(k, kDi) ; C+(k, z) ; int ; bool (False = transport)
) -> typ.Tuple[
    NDArray,                  # U+.  C+(s + k1, N): the old rows, then the new completion
    NDArray,                  # o_U. C+(k1, k)
    typ.Tuple[NDArray, ...],  # DU per member, elm_shape=C+(s + k1, kD)
    NDArray,                  # movement, shape=C
]:
    """The edges leaf -> core of one sharing group (map body): one completion from the members' contents."""
    xnp, _, _ = get_backend(False, tree_contains_jax(x[:5]))
    A, P_old, g_news, PhiPs, XiUs, k1, on = x
    g_old = P_old.swapaxes(-2, -1)
    Y = _edge_content(g_old, g_news, PhiPs, XiUs) if on else g_old
    P, o, movement = _edge(A.swapaxes(-2, -1), g_old, Y, k1)
    U_new = xnp.concatenate([A, P.swapaxes(-2, -1)], axis=-2)
    return U_new, o, tuple(contractions.contract('Cun,Cnk->Cuk', U_new, g_new) for g_new in g_news), movement


def _left_edge_step(
        carry:  typ.Tuple[NDArray, NDArray],   # (o_L, D_L) of the edge c_(i-1) -> c_i: C+(k1Li, kLi) ; C+(rL1i, kRi)
        x:      typ.Tuple,                     # (H_P, O_P, R_P, L_P, A, o_U, DU, PhiP, Xi, k1, on) at core i:
                                               #   C+(rLi, nUi, kR(i+1)); C+(rLi, kDi, kR(i+1)); C+(kRi, nUi, kR(i+1)) ;
                                               #   C+(rLi, nUi, kL(i+1)) ; C+(rL1i, nU1i, rL0(i+1)) ; C+(k1Ui, kUi) ;
                                               #   C+(nU1i, kDi) ; C+(kL(i+1), kR(i+1)) ; C+(kL(i+1), z) ; int ; bool
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],               # (o_L, D_L) of the edge c_i -> c_(i+1)
    typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # (L+_i, o_L, D_L, movement)
]:
    """One edge c_i -> c_(i+1) of the left sweep (scan body)."""
    xnp, _, _ = get_backend(False, tree_contains_jax((carry, x[:9])))
    o_L, D_L = carry
    H_P, O_P, R_P, L_P, A, o_U, DU, PhiP, Xi, k1, on = x
    M = (_new_coordinates(_new_coordinates(H_P, o_L, -3), o_U, -2)
         + _new_coordinates(contractions.contract('Cakx,Cnk->Canx', O_P, DU), o_L, -3)
         + _new_coordinates(contractions.contract('Cknx,Cak->Canx', R_P, D_L), o_U, -2))
    g_new = _unfold(M, -1)
    g_old = _unfold(_new_coordinates(_new_coordinates(L_P, o_L, -3), o_U, -2), -1)
    Abar = _unfold(A, -1)
    Y = _edge_content(g_old, (g_new,), (PhiP,), (Xi,)) if on else g_old
    P, o, movement = _edge(Abar, g_old, Y, k1)
    F = xnp.concatenate([Abar, P], axis=-1)
    D = contractions.contract('Cia,Cib->Cab', F, g_new)
    return (o, D), (_fold(F, A.shape[-3:], -1), o, D, movement)


def _right_edge_step(
        carry:  typ.Tuple[NDArray, NDArray],   # (o_R, D_R), edge c_(i+1) -> c_i: C+(k1R(i+1), kR(i+1)); C+(rR1(i+1), .)
        x:      typ.Tuple,                     # (H_P, O_P, L_P, R_P, A, o_U, DU, PhiP, Xi, k1, on) at core i:
                                               #   C+(kLi, nUi, rR(i+1)); C+(kLi, kDi, rR(i+1)); C+(kLi, nUi, kL(i+1)) ;
                                               #   C+(kRi, nUi, rR(i+1)) ; C+(rR0i, nU1i, rR1(i+1)) ; C+(k1Ui, kUi) ;
                                               #   C+(nU1i, kDi) ; C+(kRi, kLi) ; C+(kRi, z) ; int ; bool
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],               # (o_R, D_R) of the edge c_i -> c_(i-1)
    typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # (R+_i, o_R, D_R, movement)
]:
    """One edge c_i -> c_(i-1) of the right sweep (scan body; run on reversed inputs)."""
    xnp, _, _ = get_backend(False, tree_contains_jax((carry, x[:9])))
    o_R, D_R = carry
    H_P, O_P, L_P, R_P, A, o_U, DU, PhiP, Xi, k1, on = x
    M = (_new_coordinates(_new_coordinates(H_P, o_U, -2), o_R, -1)
         + _new_coordinates(contractions.contract('Cxkb,Cnk->Cxnb', O_P, DU), o_R, -1)
         + _new_coordinates(contractions.contract('Cxnk,Cbk->Cxnb', L_P, D_R), o_U, -2))
    g_new = _unfold(M, -3)
    g_old = _unfold(_new_coordinates(_new_coordinates(R_P, o_U, -2), o_R, -1), -3)
    Abar = _unfold(A, -3)
    Y = _edge_content(g_old, (g_new,), (PhiP,), (Xi,)) if on else g_old
    P, o, movement = _edge(Abar, g_old, Y, k1)
    F = xnp.concatenate([Abar, P], axis=-1)
    D = contractions.contract('Cia,Cib->Cab', F, g_new)
    return (o, D), (_fold(F, A.shape[-3:], -3), o, D, movement)


def _down_edge(
        x: typ.Tuple,   # (H_P, L_P, R_P, O_P, A, o_L, D_L, o_R, D_R, PhiP, Xi, k1, on) at core i:
                        #   C+(rLi, kUi, rR(i+1)); C+(rLi, kUi, kL(i+1)); C+(kRi, kUi, rR(i+1)); C+(rLi, kDi, rR(i+1)) ;
                        #   C+(rL1i, nD0i, rR1(i+1)) ; C+(k1Li, kLi) ; C+(rL1i, kRi) ; C+(k1R(i+1), kR(i+1)) ;
                        #   C+(rR1(i+1), kL(i+1)) ; C+(kDi, kUi) ; C+(kDi, z) ; int ; bool
) -> typ.Tuple[
    NDArray,   # O+_i. C+(rL1i, nD1i, rR1(i+1))
    NDArray,   # movement, shape=C
]:
    """The edge core -> leaf (map body); its incoming coordinate maps come from the two edge sweeps."""
    xnp, _, _ = get_backend(False, tree_contains_jax(x[:11]))
    H_P, L_P, R_P, O_P, A, o_L, D_L, o_R, D_R, PhiP, Xi, k1, on = x
    M = (_new_coordinates(_new_coordinates(H_P, o_L, -3), o_R, -1)
         + _new_coordinates(contractions.contract('Caxk,Cbk->Caxb', L_P, D_R), o_L, -3)
         + _new_coordinates(contractions.contract('Ckxb,Cak->Caxb', R_P, D_L), o_R, -1))
    g_new = _unfold(M, -2)
    g_old = _unfold(_new_coordinates(_new_coordinates(O_P, o_L, -3), o_R, -1), -2)
    Abar = _unfold(A, -2)
    Y = _edge_content(g_old, (g_new,), (PhiP,), (Xi,)) if on else g_old
    P, _, movement = _edge(Abar, g_old, Y, k1)
    return _fold(xnp.concatenate([Abar, P], axis=-1), A.shape[-3:], -2), movement


def tv_align_padded_frame(
        frame:             FrameData,                                # a padded frame: old blocks in the leading slots, stack C
        variations:        VariationsData,                           # a tangent vector at `frame` (gauged or not), stack C
        old_tucker_ranks:  typ.Sequence[int],                        # len=d, the old blocks' extents; the zero tensor: (0,)*d
        old_tt_ranks:      typ.Sequence[int],                        # len=d+1; the zero tensor: (1, 0, ..., 0, 1)

        tucker_ranks:      typ.Optional[typ.Sequence[int]] = None,   # len=d, output ranks; None = the frame's own
        tt_ranks:          typ.Optional[typ.Sequence[int]] = None,   # len=d+1, output ranks; None = the frame's own

        sharing:           typ.Optional[typ.Sequence] = None,        # len=d group labels; None = unshared
        update:            str = 'ULRO',                             # stores updated: 'U' up, 'L' left, 'R' right, 'O' down; the rest transported
) -> typ.Tuple[
    FrameData,   # the frame at the output ranks: old blocks bitwise, completions aligned
    NDArray,     # movement, shape=C: max over directed edges of ‖g_old − P⁺P⁺ᵀ g_old‖_F
]:
    """One alignment step: each completion replaced by the tangent's leading new directions, old blocks pinned."""
    if not update or len(set(update)) != len(update) or not set(update) <= set('ULRO'):
        raise ValueError("update must be a non-empty set of the stores 'U', 'L', 'R', 'O' without repeats; got %r"
                         % (update,))
    xnp, xmap, xscan = get_backend(False, tree_contains_jax((frame, variations)))
    U, O, L, R = (tuple(xnp.asarray(c) for c in cores) for cores in frame)
    V, H = (tuple(xnp.asarray(c) for c in cores) for cores in variations)
    nU, nD, rL, rR = _frame_stores((U, O, L, R), 'frame')
    d = len(U)
    shape = tuple(int(Ui.shape[-1]) for Ui in U)
    C = tuple(U[0].shape[:-2])
    want = (tuple(C + (nD[i], shape[i]) for i in range(d)), tuple(C + (rL[i], nU[i], rR[i + 1]) for i in range(d)))
    for name, cores, core_shapes in zip(('tucker', 'tt'), (V, H), want):
        if tuple(tuple(c.shape) for c in cores) != core_shapes:
            raise ValueError('the %s variations have shapes %r; the frame gives %r'
                             % (name, tuple(tuple(c.shape) for c in cores), core_shapes))

    nU0, nD0, rL0, rR0 = _rank_stores(shape, old_tucker_ranks, old_tt_ranks, 'old')
    if (tucker_ranks is None) != (tt_ranks is None):
        raise ValueError('give both output rank sequences (tucker_ranks, tt_ranks) or neither')
    out = (nU, nD, rL, rR) if tucker_ranks is None else _rank_stores(shape, tucker_ranks, tt_ranks, 'output')
    nU1, nD1, rL1, rR1 = out
    for store, letter, old, new, own in zip(_STORE_NAMES, 'UOLR', (nU0, nD0, rL0, rR0), out, (nU, nD, rL, rR)):
        for i, (a, b, c) in enumerate(zip(old, new, own)):
            if not a <= b <= c:
                raise ValueError('%s store %d: old %d, output %d, frame %d; the old block must fit the output and the '
                                 'output the frame' % (store, i, a, b, c))
            if b < c and letter not in update:
                raise ValueError('%s store %d is truncated (%d -> %d) but not updated: a pass may not both transport '
                                 'and truncate' % (store, i, c, b))

    groups = tuple((i,) for i in range(d)) if sharing is None else sharing_module.validate_sharing(sharing, shape)
    for group in groups:
        for which, stores in (('old', nU0), ('output', nU1), ('frame', nU)):
            if len({stores[i] for i in group}) != 1:
                raise ValueError('sharing group %r has unequal Tucker stores %r (%s); a shared factor needs one width '
                                 'per group' % (group, tuple(stores[i] for i in group), which))

    dtype = xnp.result_type(U[0], V[0])
    edge_end = (xnp.zeros(C + (0, 1), dtype=dtype), xnp.zeros(C + (0, 0), dtype=dtype))   # the messages past the ends
    coordinates_end = (xnp.zeros(C + (0, 0), dtype=dtype), xnp.zeros(C + (1, 0), dtype=dtype))

    # -- step 1: far-side messages at the old frames, from the far leaves inward
    PhiO, XiO = xmap(_down_message, (tuple(V[i][..., nD0[i]:, :] for i in range(d)), U))
    js = tuple(range(d - 1, 0, -1))
    _, ys = xscan(_left_message_step, edge_end, (
        tuple(H[j][..., rL0[j]:, :, :] for j in js), tuple(O[j][..., rL0[j]:, nD0[j]:, :] for j in js),
        tuple(L[j][..., rL0[j]:, :, rL0[j + 1]:] for j in js), tuple(R[j] for j in js),
        tuple(PhiO[j] for j in js), tuple(XiO[j] for j in js)))
    PhiL, XiL = (tuple(m[::-1]) + (end,) for m, end in zip(ys or ((), ()), edge_end))   # index i: c_i -> c_(i+1)
    js = tuple(range(d - 1))
    _, ys = xscan(_right_message_step, edge_end, (
        tuple(H[j][..., rR0[j + 1]:] for j in js), tuple(O[j][..., nD0[j]:, rR0[j + 1]:] for j in js),
        tuple(R[j][..., rR0[j]:, :, rR0[j + 1]:] for j in js), tuple(L[j] for j in js),
        tuple(PhiO[j] for j in js), tuple(XiO[j] for j in js)))
    PhiR, XiR = ((end,) + tuple(m) for m, end in zip(ys or ((), ()), edge_end))         # index i: c_i -> c_(i-1)
    PhiU, XiU = xmap(_up_message, (
        tuple(H[i][..., nU0[i]:, :] for i in range(d)), tuple(L[i][..., nU0[i]:, rL0[i + 1]:] for i in range(d)),
        tuple(R[i][..., rR0[i]:, nU0[i]:, :] for i in range(d)), O, PhiL, XiL, PhiR, XiR))

    # -- step 2: new completions, new frames: up, the left sweep, the right sweep, down
    U1, oU, DUs, moves_U = xmap(_up_edge, (
        tuple(U[g[0]][..., :nU0[g[0]], :] for g in groups), tuple(U[g[0]][..., nU0[g[0]]:, :] for g in groups),
        tuple(tuple(V[i][..., nD0[i]:, :].swapaxes(-2, -1) for i in g) for g in groups),
        tuple(tuple(PhiU[i][..., nD0[i]:] for i in g) for g in groups), tuple(tuple(XiU[i] for i in g) for g in groups),
        tuple(nU1[g[0]] - nU0[g[0]] for g in groups), ('U' in update,) * len(groups)))
    group_of = tuple(next((j, g.index(i)) for j, g in enumerate(groups) if i in g) for i in range(d))
    oU = tuple(oU[j] for j, _ in group_of)
    DU = tuple(DUs[j][m] for j, m in group_of)
    up = tuple(U1[j] for j, _ in group_of)                                          # one array per group

    js = tuple(range(d - 1))
    _, ys = xscan(_left_edge_step, coordinates_end, (
        tuple(H[i][..., rR0[i + 1]:] for i in js), tuple(O[i][..., nD0[i]:, rR0[i + 1]:] for i in js),
        tuple(R[i][..., rR0[i]:, :, rR0[i + 1]:] for i in js), tuple(L[i][..., rL0[i + 1]:] for i in js),
        tuple(_embed(L[i][..., :rL0[i], :nU0[i], :rL0[i + 1]], (rL1[i], nU1[i], rL0[i + 1])) for i in js),
        tuple(oU[i] for i in js), tuple(DU[i] for i in js), tuple(PhiL[i][..., rR0[i + 1]:] for i in js),
        tuple(XiL[i] for i in js), tuple(rL1[i + 1] - rL0[i + 1] for i in js), ('L' in update,) * len(js)))
    left, oL, DL, moves_L = ys or ((), (), (), ())
    oL, DL = ((end,) + tuple(m) for m, end in zip((oL, DL), coordinates_end))         # index i: c_(i-1) -> c_i

    js = tuple(range(d - 1, 0, -1))
    _, ys = xscan(_right_edge_step, coordinates_end, (
        tuple(H[i][..., rL0[i]:, :, :] for i in js), tuple(O[i][..., rL0[i]:, nD0[i]:, :] for i in js),
        tuple(L[i][..., rL0[i]:, :, rL0[i + 1]:] for i in js), tuple(R[i][..., rR0[i]:, :, :] for i in js),
        tuple(_embed(R[i][..., :rR0[i], :nU0[i], :rR0[i + 1]], (rR0[i], nU1[i], rR1[i + 1])) for i in js),
        tuple(oU[i] for i in js), tuple(DU[i] for i in js), tuple(PhiR[i][..., rL0[i]:] for i in js),
        tuple(XiR[i] for i in js), tuple(rR1[i] - rR0[i] for i in js), ('R' in update,) * len(js)))
    right, oR, DR, moves_R = (tuple(m[::-1]) for m in (ys or ((), (), (), ())))
    oR, DR = (tuple(m) + (end,) for m, end in zip((oR, DR), coordinates_end))         # index i: c_(i+1) -> c_i

    down, moves_O = xmap(_down_edge, (
        tuple(H[i][..., nU0[i]:, :] for i in range(d)), tuple(L[i][..., nU0[i]:, rL0[i + 1]:] for i in range(d)),
        tuple(R[i][..., rR0[i]:, nU0[i]:, :] for i in range(d)), tuple(O[i][..., nD0[i]:, :] for i in range(d)),
        tuple(_embed(O[i][..., :rL0[i], :nD0[i], :rR0[i + 1]], (rL1[i], nD0[i], rR1[i + 1])) for i in range(d)),
        oL, DL, oR, DR, tuple(PhiO[i][..., nU0[i]:] for i in range(d)), XiO,
        tuple(nD1[i] - nD0[i] for i in range(d)), ('O' in update,) * d))

    left = left + (_embed(L[d - 1][..., :rL0[d - 1], :nU0[d - 1], :], (rL1[d - 1], nU1[d - 1], rL1[d])),)   # carriers:
    right = (_embed(R[0][..., :nU0[0], :rR0[1]], (rR1[0], nU1[0], rR1[1])),) + right    # old content only
    movement = xnp.max(xnp.stack(tuple(moves_U) + tuple(moves_L) + tuple(moves_R) + tuple(moves_O)), axis=0)
    return (up, tuple(down), left, right), movement


# --------------------------------------------------------------------------------------------------
# The loop: F <- align(F, project(F, hint)), on a fixed schedule
# --------------------------------------------------------------------------------------------------
@ft.lru_cache(maxsize=None)
def _jitted(
        fn:              typ.Callable,          # module-level, so a stable object (jax caches compiled programs on the wrapper)
        static_argnums:  typ.Tuple[int, ...],
) -> typ.Callable:
    """``jax.jit(fn, static_argnums=...)``, memoized: one wrapper per function, so its programs persist across calls."""
    import jax
    return jax.jit(fn, static_argnums=static_argnums)


_STEP_STATICS = (2, 3, 4, 5, 6, 7)   # tv_align_padded_frame's old ranks, output ranks, sharing and update


def _aligning_pass(
        step:       typ.Callable,                             # tv_align_padded_frame, or its jitted wrapper
        frame:      FrameData,                                # stack C
        hint:       typ.Any,                                  # handed to `project` unread
        project:    typ.Callable,                             # project(frame, hint) -> VariationsData
        old_ranks:  typ.Tuple[typ.Tuple[int, ...], ...],      # (old Tucker ranks len=d, old TT ranks len=d+1)
        out_ranks:  typ.Tuple[typ.Optional[typ.Tuple[int, ...]], ...],   # (Tucker, TT) output ranks; (None, None) = the frame's own
        sharing:    typ.Optional[typ.Tuple],                  # len=d group labels (hashable: static under jit)
        update:     str,                                      # 'ULRO' | 'UL' | 'RO'
) -> typ.Tuple[
    FrameData,   # the frame after the pass
    NDArray,     # its movement, shape=C
]:
    """One pass: project at the frame, then one step; the arguments positional, as the jitted step's statics are."""
    return step(frame, project(frame, hint), *old_ranks, *out_ranks, sharing, update)


def _peak(
        movements:  typ.Sequence[NDArray],   # each shape=C
) -> float:                                   # the largest movement, over the passes and the stack
    """The stop test's quantity; a NaN propagates, so it never stops the loop."""
    return float(np.max([np.max(np.asarray(m)) for m in movements]))


def fv_aligned_padded_frame(
        old_frame:           FrameData,                          # the old frame, stack C (fv_empty_frame for the zero tensor)
        tucker_ranks:        typ.Sequence[int],                  # len=d, target
        tt_ranks:            typ.Sequence[int],                  # len=d+1, target (boundary 1)
        hint:                typ.Any,                            # opaque: handed to `project` unread
        project:             typ.Callable,                       # project(frame, hint) -> VariationsData: ONE fixed tensor projected at `frame`

        sharing:             typ.Optional[typ.Sequence] = None,  # len=d group labels; None = unshared
        oversampling:        int   = ALIGN_OVERSAMPLING,         # p: working ranks = ranks.oversampled_ranks(shape, target, p, sharing)
        oversampled_passes:  int   = ALIGN_OVERSAMPLED_PASSES,   # passes at the working ranks before the truncating pass
        max_passes:          int   = ALIGN_MAX_PASSES,           # passes in all, the truncating pass included; 0 = the hint-free start at the target
        tol:                 float = ALIGN_TOL,                  # early exit on the movement
        alternate:           bool  = ALIGN_ALTERNATE,            # two-colour passes after the truncating pass
        seed:                int   = ZERO_FRAME_SEED,            # the start's Haar stream
        use_jit:             bool  = False,                      # the old frame to jax, the host loop calling a memoized jitted step; else eager
) -> typ.Tuple[
    FrameData,                 # the aligned padded frame at the target ranks
    typ.Tuple[NDArray, ...],   # movements, one per pass, each shape=C
]:
    """The aligned padded frame: the seeded start, then alignment passes against ``project(frame, hint)``."""
    for name, value in (('max_passes', max_passes), ('oversampled_passes', oversampled_passes),
                        ('oversampling', oversampling)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError('%s must be a non-negative int, got %r' % (name, value))
    tucker_ranks, tt_ranks = tuple(int(n) for n in tucker_ranks), tuple(int(r) for r in tt_ranks)
    jitted = bool(use_jit) and jax_or_warn('use_jit=True')     # no jax: a warning, then eager on numpy
    if jitted:                                                  # asking for jit opts into jax: the old frame is
        old_frame = tree_to_jax(old_frame)                      #   converted (the optimizers' auto-convert); the
    old_stores = _frame_stores(old_frame, 'old_frame')          #   hint is opaque and handed on as is
    old_tucker_ranks, old_tt_ranks = old_stores[0], old_stores[2]       # an orthogonal representation's own widths
    shape = tuple(int(U.shape[-1]) for U in old_frame[0])
    target_stores = _rank_stores(shape, tucker_ranks, tt_ranks, 'target')
    _require_contains(old_stores, target_stores)
    if max_passes == 0 or target_stores == old_stores:          # zero passes: the hint-free start at the target;
        return fv_padded_frame(old_frame, tucker_ranks, tt_ranks, sharing, seed), ()   # or nothing to complete

    working = ranks.oversampled_ranks(shape, tucker_ranks, tt_ranks, oversampling, sharing)
    n_over = 0 if _rank_stores(shape, *working, 'working') == target_stores else oversampled_passes
    n_over = min(n_over, max_passes - 1)                        # the cap counts the truncating pass (pass n_over)
    step = _jitted(tv_align_padded_frame, _STEP_STATICS) if jitted else tv_align_padded_frame
    old = (old_tucker_ranks, old_tt_ranks)
    target = (tucker_ranks, tt_ranks)
    sharing = None if sharing is None else tuple(sharing)       # hashable: a static argument of the jitted step
    frame = fv_padded_frame(old_frame, *working, sharing, seed)
    movements = []
    for it in range(n_over + 1):                                # the oversampled passes, then the truncating pass
        out = (None, None) if it < n_over else target
        frame, movement = _aligning_pass(step, frame, hint, project, old, out, sharing, 'ULRO')
        movements.append(movement)
    if alternate:                                               # ('UL', 'RO') pairs while a whole pair fits; the stop
        stop = False                                            #   is tested after each pair (both below tol)
        while not stop and len(movements) + 2 <= max_passes:
            for update in ('UL', 'RO'):
                frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, update)
                movements.append(movement)
            stop = _peak(movements[-2:]) < tol
        if not stop and len(movements) < max_passes:           # an odd remainder: one final 'UL' pass
            frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, 'UL')
            movements.append(movement)
    else:                                                       # 'ULRO' passes, the stop tested after each (the
        stop = _peak(movements[-1:]) < tol                      #   truncating pass included)
        while not stop and len(movements) < max_passes:
            frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, 'ULRO')
            movements.append(movement)
            stop = _peak([movement]) < tol
    return frame, tuple(movements)
