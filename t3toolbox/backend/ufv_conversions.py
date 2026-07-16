# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform frame/variations conversions (raw supercore + mask data).

Data-level ``ut3_orthogonal_representations`` plus the cross-layer converters
(``t3frame_to_ut3frame``, ``ut3variations_to_t3variations``, ...) between the ragged and uniform
frame/variations representations.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.ufv_masking as ufv_masking
import t3toolbox.backend.ut3_masking as ut3_masking
from t3toolbox.backend.common import *

__all__ = [
    'ut3_orthogonal_representations',
    'ut3frame_to_t3frame',
    't3frame_to_ut3frame',
    'ut3variations_to_t3variations',
    't3variations_to_ut3variations',
    't3frameweights_to_ut3frameweights',
    'ut3frameweights_to_t3frameweights',
]


def _pad_stack(
        cores:       typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+source_tail
        target_tail: typ.Tuple[int, ...],    # the common (padded) trailing shape every core is padded up to
        use_jax:     bool,
) -> NDArray:                                # supercore, shape=(d,)+stack_shape+target_tail
    """Zero-pad each (ragged-tailed) core up to ``target_tail`` and stack onto a new leading ``d`` axis.

    The leading ``stack_shape`` (= ``C``) axes are kept; only the trailing core axes are padded. Padding is
    don't-care garbage that the rank masks mark unreal; zeros are the safe fill (``docs/uniform_*``)."""
    xnp, _, _ = get_backend(False, use_jax)
    n_stack = len(cores[0].shape) - len(target_tail)   # = len(C)
    padded = []
    for c in cores:
        pad = ((0, 0),) * n_stack + tuple((0, t - s) for t, s in zip(target_tail, c.shape[n_stack:]))
        padded.append(xnp.pad(c, pad))
    return xnp.stack(padded)


def _broadcast_ranks_over_stack(
        ranks_seq:   typ.Sequence[int],      # len=d or d+1, one rank per edge (shared across the stack)
        stack_shape: typ.Tuple[int, ...],    # C
) -> NDArray:                                # HOST int, (len,)+stack_shape
    """Broadcast a per-edge rank sequence (one ragged object -> ranks shared across its ``C`` stack) to the
    ``(edge,)+stack_shape`` array ``ufv_make_frame_masks`` expects. HOST numpy -- masks are static structure."""
    ones_stack = np.ones(stack_shape, dtype=int)
    return np.stack([r * ones_stack for r in ranks_seq])


def ut3_orthogonal_representations(
        data: typ.Tuple[
            NDArray,                          # tucker_supercore
            NDArray,                          # tt_supercore
            typ.Tuple[int, ...],              # shape
            typ.Tuple[NDArray, NDArray],      # (tucker_edge_mask, tt_edge_mask) -- the plain-UT3 rank masks
        ],
        already_left_orthogonal: bool = False,
        squash_tails:                  bool = True,
) -> typ.Tuple[
    typ.Tuple[                                # frame .data:
        NDArray, NDArray, NDArray, NDArray,   #   up_sc, down_sc, left_sc, right_sc
        typ.Tuple[int, ...],                  #   shape
        typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (up, down, frame_left, frame_right) masks
    ],
    typ.Tuple[                                # variations .data:
        NDArray, NDArray,                     #   tucker_var_sc, tt_var_sc
        typ.Tuple[int, ...],                  #   shape
        typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (variations up, down, left, right) masks
    ],
]:
    '''Orthogonal (frame, variations) representation of a uniform Tucker tensor train, on raw ``.data``.

    Backend twin of the frontend ``ut3_orthogonal_representations`` (which wraps this into the OO
    ``UT3Frame`` / ``UT3Variations``). Takes a plain ``UniformTuckerTensorTrain.data`` and returns the
    **frame** and **variation** ``.data`` tuples (supercores + ``shape`` + the rank masks).

    WHY THIS IS A BACKEND FUNCTION (and not something to open-code): the output frame masks are **prefix**
    masks built from the orthogonal-representation *ranks* (``ufv_make_frame_masks`` = ``arange < rank``) --
    they assert the real orthonormal content sits in the **upper-left** ``[0, rank)`` slots of each
    supercore. That is correct ONLY because the orthogonalization is **SVD-based**: the SVD sorts content
    by singular value into the leading slots, with zeros / orthonormal completion trailing. A QR-based
    orthogonalization would scatter the real content across non-prefix positions and these masks would be
    WRONG -- see ``docs/contributor/uniform_svd_prefix_orthogonalization.md``. Building the masks any other way (e.g.
    from raw supercore magnitudes) is the easy mistake this function exists to prevent.

    The frame masks come from the orthogonal-representation ranks; the variation masks reuse the up/down
    masks and the frame left/right masks shifted by one (a variation occupies one TT slot, not a boundary
    edge -- hence ``left[:-1]`` / ``right[1:]``).
    '''
    tk_sc, tt_sc, shape, (tkm, ttm) = data
    masked_tk, masked_tt = ut3_masking.ut3_apply_masks(data)   # zero the garbage before the SVD sweep

    # fv_conversions.t3_orthogonal_representations is polymorphic (accepts uniform supercores) and SVD-based.
    (uc, dc, lc, rc), (tkv, ttv) = fv_conversions.t3_orthogonal_representations(
        (masked_tk, masked_tt), already_left_orthogonal=already_left_orthogonal, squash_tails=squash_tails)

    up_ranks, down_ranks, left_ranks, right_ranks = ranks.compute_orthogonal_representation_ranks(
        shape, tkm.sum(axis=-1), ttm.sum(axis=-1))

    nU, nD, rL, rR = uc.shape[-2], dc.shape[-2], lc.shape[-1], rc.shape[-1]
    um, dm, lm, rm = ufv_masking.ufv_make_frame_masks(up_ranks, down_ranks, left_ranks, right_ranks, nU, nD, rL, rR)

    frame_data     = (uc, dc, lc, rc, shape, (um, dm, lm, rm))
    variation_data = (tkv, ttv, shape, (um, dm, lm[:-1], rm[1:]))
    return frame_data, variation_data


def ut3frame_to_t3frame(
        x: typ.Tuple[
            NDArray,                          # up_tucker_supercore
            NDArray,                          # down_tt_supercore
            NDArray,                          # left_tt_supercore
            NDArray,                          # right_tt_supercore
            typ.Tuple[int, ...],              # shape
            typ.Tuple[                        # (up_mask, down_mask, frame_left_mask, frame_right_mask)
                NDArray, NDArray, NDArray, NDArray,
            ],
        ],
) -> typ.Union[
    typ.Tuple[typ.Tuple[NDArray, ...], ...],  # (up_cores, down_cores, left_cores, right_cores), if unstacked
    typ.Tuple,                                # else a nested tree (shaped like stack_shape) of those
]:
    '''Convert a uniform UT3Frame ``.data`` to ragged ``T3Frame`` core-tuples (or a nested tree, if stacked).

    The physical mode dims are a contiguous prefix, so they slice ``[:Ni]`` (from the ``shape`` ints, no
    argwhere); only the *rank* masks scatter, so they are extracted with ``np.argwhere`` (HOST numpy --
    masks are host). The supercores may be jax; advanced-indexing them with the host int indices is fine.
    '''
    (up_supercore, down_supercore, left_supercore, right_supercore,
     shape, (up_mask, down_mask, frame_left_mask, frame_right_mask)) = x
    require_concrete_masks(up_mask, down_mask, frame_left_mask, frame_right_mask)  # host: argwhere is np
    stack_shape = up_supercore.shape[1:-2]
    d = up_supercore.shape[0]

    if not stack_shape:  # unstacked -> one ragged (up, down, left, right) core set
        up_cores, down_cores, left_cores, right_cores = [], [], [], []
        for ind in range(d):
            up_inds   = np.argwhere(up_mask[ind]).reshape(-1)
            down_inds = np.argwhere(down_mask[ind]).reshape(-1)
            left_a    = np.argwhere(frame_left_mask[ind]).reshape(-1)
            left_b    = np.argwhere(frame_left_mask[ind + 1]).reshape(-1)
            right_a   = np.argwhere(frame_right_mask[ind]).reshape(-1)
            right_b   = np.argwhere(frame_right_mask[ind + 1]).reshape(-1)
            Ni = shape[ind]

            up_cores.append(   up_supercore[ind][up_inds, :][:, :Ni])
            down_cores.append( down_supercore[ind][left_a, :, :][:, down_inds, :][:, :, right_b])
            left_cores.append( left_supercore[ind][left_a, :, :][:, up_inds, :][:, :, left_b])
            right_cores.append(right_supercore[ind][right_a, :, :][:, up_inds, :][:, :, right_b])

        return tuple(up_cores), tuple(down_cores), tuple(left_cores), tuple(right_cores)

    all_T3Bs = []
    for ii in range(up_supercore.shape[1]):
        xi = (
            up_supercore[:, ii], down_supercore[:, ii], left_supercore[:, ii], right_supercore[:, ii],
            shape,
            (up_mask[:, ii], down_mask[:, ii], frame_left_mask[:, ii], frame_right_mask[:, ii]),
        )
        all_T3Bs.append(ut3frame_to_t3frame(xi))
    return tuple(all_T3Bs)


def t3frame_to_ut3frame(
        frame_data: typ.Tuple[
            typ.Tuple[NDArray, ...],  # up_tucker_cores,   len=d, elm_shape=C+(nUi, Ni)
            typ.Tuple[NDArray, ...],  # down_tt_cores,     len=d, elm_shape=C+(rLi, nDi, rR(i+1))
            typ.Tuple[NDArray, ...],  # left_tt_cores,     len=d, elm_shape=C+(rLi, nUi, rL(i+1))
            typ.Tuple[NDArray, ...],  # right_tt_cores,    len=d, elm_shape=C+(rRi, nUi, rR(i+1))
        ],
        N:  typ.Optional[int] = None,   # padded mode dim   (default max(shape))
        nU: typ.Optional[int] = None,   # padded up rank    (default max(up_ranks))
        nD: typ.Optional[int] = None,   # padded down rank  (default max(down_ranks))
        rL: typ.Optional[int] = None,   # padded left rank  (default max(left_ranks))
        rR: typ.Optional[int] = None,   # padded right rank (default max(right_ranks))
) -> typ.Tuple[
    NDArray,                          # up_tucker_supercore,  (d,)+C+(nU, N)
    NDArray,                          # down_tt_supercore,    (d,)+C+(rL, nD, rR)
    NDArray,                          # left_tt_supercore,    (d,)+C+(rL, nU, rL)
    NDArray,                          # right_tt_supercore,   (d,)+C+(rR, nU, rR)
    typ.Tuple[int, ...],              # shape = (N0,...,N(d-1))
    typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (up, down, frame_left, frame_right) masks
]:
    '''Pack a ragged ``T3Frame`` core-tuple into uniform frame ``.data`` (supercores + shape + masks).

    Inverse of :py:func:`ut3frame_to_t3frame`. A *single* ragged frame has ranks shared across its ``C``
    stack, so the masks come out **uniform across the stack** (varying-rank uniform batches arise only by
    ``stack``-ing a heterogeneous tree). Pads each family to common dims (default: max over modes; pass
    ``N``/``nU``/``nD``/``rL``/``rR`` to force larger) and records the real extents as prefix masks
    (``ufv_make_frame_masks``; the real content lands in the upper-left, so the prefix masks are correct).
    '''
    up_cores, down_cores, left_cores, right_cores = frame_data
    d = len(up_cores)
    use_jax = tree_contains_jax(frame_data)
    stack_shape = up_cores[0].shape[:-2]

    shape       = tuple(int(U.shape[-1]) for U in up_cores)                                   # (N0,...,N(d-1))
    up_ranks    = tuple(int(U.shape[-2]) for U in up_cores)                                   # (d,)   nU per mode
    down_ranks  = tuple(int(G.shape[-2]) for G in down_cores)                                 # (d,)   nD per mode
    left_ranks  = tuple(int(G.shape[-3]) for G in left_cores) + (int(left_cores[-1].shape[-1]),)    # (d+1,) rL edges
    right_ranks = tuple(int(G.shape[-3]) for G in right_cores) + (int(right_cores[-1].shape[-1]),)  # (d+1,) rR edges

    N  = max(shape)       if N  is None else N
    nU = max(up_ranks)    if nU is None else nU
    nD = max(down_ranks)  if nD is None else nD
    rL = max(left_ranks)  if rL is None else rL
    rR = max(right_ranks) if rR is None else rR

    up_sc    = _pad_stack(up_cores,    (nU, N),     use_jax)
    down_sc  = _pad_stack(down_cores,  (rL, nD, rR), use_jax)
    left_sc  = _pad_stack(left_cores,  (rL, nU, rL), use_jax)
    right_sc = _pad_stack(right_cores, (rR, nU, rR), use_jax)

    masks = ufv_masking.ufv_make_frame_masks(
        _broadcast_ranks_over_stack(up_ranks,    stack_shape),
        _broadcast_ranks_over_stack(down_ranks,  stack_shape),
        _broadcast_ranks_over_stack(left_ranks,  stack_shape),
        _broadcast_ranks_over_stack(right_ranks, stack_shape),
        nU, nD, rL, rR,
    )
    return up_sc, down_sc, left_sc, right_sc, shape, masks


def ut3variations_to_t3variations(
        x: typ.Tuple[
            NDArray,                          # tucker_variations,  (d,)+C+(nD, N)
            NDArray,                          # tt_variations,      (d,)+C+(rL, nU, rR)
            typ.Tuple[int, ...],              # shape
            typ.Tuple[                        # (variations up, down, left, right) masks
                NDArray, NDArray, NDArray, NDArray,
            ],
        ],
) -> typ.Union[
    typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]],  # (tucker_variations, tt_variations), if unstacked
    typ.Tuple,                                                    # else a nested tree (shape stack_shape) of those
]:
    '''Convert uniform ``UT3Variations`` ``.data`` to ragged ``T3Variations`` core-tuples (or a tree, if stacked).

    Variations twin of :py:func:`ut3frame_to_t3frame`. The physical mode dim is a prefix (slices ``[:Ni]``
    from ``shape``); the rank masks scatter, so they extract with ``np.argwhere`` (HOST numpy). The
    variation tt-core ``H_i`` has shape ``(rLi, nUi, rR(i+1))`` -- left/up/right masks index its three axes.
    '''
    tucker_variations, tt_variations, shape, (vup, vdown, vleft, vright) = x
    require_concrete_masks(vup, vdown, vleft, vright)  # host: argwhere is np
    stack_shape = tucker_variations.shape[1:-2]
    d = tucker_variations.shape[0]

    if not stack_shape:  # unstacked -> one ragged (tucker_variations, tt_variations)
        tucker_cores, tt_cores = [], []
        for ind in range(d):
            up_inds    = np.argwhere(vup[ind]).reshape(-1)
            down_inds  = np.argwhere(vdown[ind]).reshape(-1)
            left_inds  = np.argwhere(vleft[ind]).reshape(-1)
            right_inds = np.argwhere(vright[ind]).reshape(-1)
            Ni = shape[ind]

            tucker_cores.append(tucker_variations[ind][down_inds, :][:, :Ni])
            tt_cores.append(   tt_variations[ind][left_inds, :, :][:, up_inds, :][:, :, right_inds])

        return tuple(tucker_cores), tuple(tt_cores)

    all_T3Vs = []
    for ii in range(tucker_variations.shape[1]):
        xi = (
            tucker_variations[:, ii], tt_variations[:, ii],
            shape,
            (vup[:, ii], vdown[:, ii], vleft[:, ii], vright[:, ii]),
        )
        all_T3Vs.append(ut3variations_to_t3variations(xi))
    return tuple(all_T3Vs)


def t3variations_to_ut3variations(
        variations_data: typ.Tuple[
            typ.Tuple[NDArray, ...],  # tucker_variations, len=d, elm_shape=C+(nDi, Ni)
            typ.Tuple[NDArray, ...],  # tt_variations,     len=d, elm_shape=C+(rLi, nUi, rR(i+1))
        ],
        N:  typ.Optional[int] = None,   # padded mode dim   (default max(shape))
        nU: typ.Optional[int] = None,   # padded up rank    (default max(up_ranks))
        nD: typ.Optional[int] = None,   # padded down rank  (default max(down_ranks))
        rL: typ.Optional[int] = None,   # padded left rank  (default max(left_ranks))
        rR: typ.Optional[int] = None,   # padded right rank (default max(right_ranks))
) -> typ.Tuple[
    NDArray,                          # tucker_variations supercore, (d,)+C+(nD, N)
    NDArray,                          # tt_variations supercore,     (d,)+C+(rL, nU, rR)
    typ.Tuple[int, ...],              # shape
    typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (variations up, down, left, right) masks, all (d,)-leading
]:
    '''Pack a ragged ``T3Variations`` core-tuple into uniform variation ``.data``.

    Inverse of :py:func:`ut3variations_to_t3variations`. The variation masks are all ``(d,)``-leading (a
    variation occupies one TT slot, not a boundary edge), so the left/right ranks are the per-slot bonds
    ``rLi`` / ``rR(i+1)`` -- NOT a ``(d+1,)`` edge sequence. ``ufv_make_frame_masks`` builds the prefix masks
    from whatever leading shape its rank args carry, so it serves here too.
    '''
    tucker_cores, tt_cores = variations_data
    use_jax = tree_contains_jax(variations_data)
    stack_shape = tucker_cores[0].shape[:-2]

    shape       = tuple(int(V.shape[-1]) for V in tucker_cores)   # (N0,...,N(d-1))
    up_ranks    = tuple(int(H.shape[-2]) for H in tt_cores)       # (d,) nUi
    down_ranks  = tuple(int(V.shape[-2]) for V in tucker_cores)   # (d,) nDi
    left_ranks  = tuple(int(H.shape[-3]) for H in tt_cores)       # (d,) rLi  (per-slot left bond)
    right_ranks = tuple(int(H.shape[-1]) for H in tt_cores)       # (d,) rR(i+1) (per-slot right bond)

    N  = max(shape)       if N  is None else N
    nU = max(up_ranks)    if nU is None else nU
    nD = max(down_ranks)  if nD is None else nD
    rL = max(left_ranks)  if rL is None else rL
    rR = max(right_ranks) if rR is None else rR

    tkv = _pad_stack(tucker_cores, (nD, N),     use_jax)
    ttv = _pad_stack(tt_cores,     (rL, nU, rR), use_jax)

    masks = ufv_masking.ufv_make_frame_masks(
        _broadcast_ranks_over_stack(up_ranks,    stack_shape),
        _broadcast_ranks_over_stack(down_ranks,  stack_shape),
        _broadcast_ranks_over_stack(left_ranks,  stack_shape),
        _broadcast_ranks_over_stack(right_ranks, stack_shape),
        nU, nD, rL, rR,
    )
    return tkv, ttv, shape, masks


def t3frameweights_to_ut3frameweights(
        weights: typ.Tuple[
            typ.Sequence[NDArray],  # up_weights,    len=d, elm_shape=C+(nUi,)
            typ.Sequence[NDArray],  # down_weights,  len=d, elm_shape=C+(nDi,)
            typ.Sequence[NDArray],  # left_weights,  len=d, elm_shape=C+(rLi,)
            typ.Sequence[NDArray],  # right_weights, len=d, elm_shape=C+(rRi,)
        ],
        nU: int = None,             # padded up rank    (default max); pass to match the tangent's pad
        nD: int = None,             # padded down rank  (default max)
        rL: int = None,             # padded left rank  (default max)
        rR: int = None,             # padded right rank (default max)
) -> typ.Tuple[
    NDArray,  # up_weight_supercore,    (d,)+C+(nU,)
    NDArray,  # down_weight_supercore,  (d,)+C+(nD,)
    NDArray,  # left_weight_supercore,  (d,)+C+(rL,)
    NDArray,  # right_weight_supercore, (d,)+C+(rR,)
    typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # the four edge masks, HOST bool, static
]:
    """Pack a ragged ``T3FrameWeights`` into uniform frame-weight supercores + masks (the ``.data`` layout).

    The frame-weight twin of :py:func:`t3frame_to_ut3frame`, and simpler: each family is one vector per
    edge, so only the last axis is padded, and there is no physical ``shape`` (weights live on internal
    edges only). The ``C`` stack is carried through untouched -- a frame weight is **frame-like**, so it
    never grows a ``K`` axis (``docs/contributor/weighted_internals.md`` §8.5).

    Pass ``nU``/``nD``/``rL``/``rR`` to match the padding of the tangent these weights will pair with
    (e.g. from ``frame.uniform_structure``); the defaults pad tightly to the weights' own max ranks.
    """
    use_jax = tree_contains_jax(weights)
    up_weights, down_weights, left_weights, right_weights = weights
    stack_shape = up_weights[0].shape[:-1]   # C

    sizes = []
    for fam, override in zip(weights, (nU, nD, rL, rR)):
        sizes.append(max(w.shape[-1] for w in fam) if override is None else override)

    supercores = tuple(_pad_stack(fam, (size,), use_jax) for fam, size in zip(weights, sizes))
    # prefix_mask, not the masking layer: weighting and masking each call the shared neutral primitive.
    masks = tuple(prefix_mask(_broadcast_ranks_over_stack([w.shape[-1] for w in fam], stack_shape), size)
                  for fam, size in zip(weights, sizes))
    return supercores + (masks,)


def ut3frameweights_to_t3frameweights(
        weights: typ.Tuple[
            NDArray, NDArray, NDArray, NDArray,             # up, down, left, right supercores, (d,)+C+(size,)
            typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # the four edge masks
        ],
) -> typ.Union[
    typ.Tuple[typ.Tuple[NDArray, ...], ...],  # (up, down, left, right) ragged families, if unstacked
    typ.Tuple,                                 # else a nested tree (shaped like C) of those
]:
    """Convert uniform frame-weight supercores + masks back to ragged ``T3FrameWeights`` families.

    The frame-weight twin of :py:func:`ut3frame_to_t3frame`. As there, a **stacked** weight returns a
    *tree* of ragged weights (a varying-rank stack has no single ragged representation), and the real slots
    are selected *through the masks* rather than by slicing a prefix, since a mask may be gappy after
    ``+``/``x`` (``docs/uniform_masks_vs_ranks.md``).
    """
    supercores, masks = weights[:4], weights[4]
    require_concrete_masks(*masks)  # host masks: the boolean index must be concrete

    stack_shape = supercores[0].shape[1:-1]   # C

    if not stack_shape:  # unstacked -> one ragged 4-tuple of families
        return tuple(tuple(w[m] for m, w in zip(list(mask), list(sc)))
                     for sc, mask in zip(supercores, masks))

    return tuple(
        ut3frameweights_to_t3frameweights(
            tuple(sc[:, ii] for sc in supercores) + (tuple(m[:, ii] for m in masks),))
        for ii in range(supercores[0].shape[1])
    )
