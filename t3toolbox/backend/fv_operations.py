# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Operations on raw (frame, variations) data: variation constructors, frame ops, residuals.

``fv_variations_{zeros,randn,unit,from_vector}`` build variation tuples for a given frame;
``fv_frame_reverse`` and the ``fv_frame_*_residual`` checkers operate on the frame alone. The
``fv_*_weights`` functions are the weighted-layer tangent metric (absorb into the variation cores).
"""
from __future__ import annotations

import math
import typing as typ
import numpy as np

import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.t3_operations as t3_operations
from t3toolbox.backend.common import *
import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.corewise as corewise

__all__ = [
    'fv_variation_shapes',
    'fv_variations_from_vector',
    'fv_variations_zeros',
    'fv_variations_randn',
    'fv_variations_unit',
    'fv_frame_reverse',
    'fv_frame_orthogonality_residual',
    'fv_frame_consistency_residual',
    'fv_absorb_weights',
    'fv_weights_consistent',
    'fv_concatenate_weights',
    'fv_kronecker_weights',
    't3weights_to_t3frameweights',
    'fv_weighted_norm',
    'fv_weighted_inner',
]


# Constructors and the vector round-trip for T3 *variations* (the frame-variation tangent format).
#
# These cannot reuse the TuckerTensorTrain backends (t3_from_vector / t3_zeros / t3_corewise_randn):
# those derive their core shapes from (shape, tucker_ranks, tt_ranks) and build the Tucker+TT core
# layout, whereas variations are given their shapes directly as two families
# (tucker_variation_shapes, tt_variation_shapes).

VariationShapes = typ.Tuple[
    typ.Sequence[typ.Sequence[int]],  # tucker_variation_shapes
    typ.Sequence[typ.Sequence[int]],  # tt_variation_shapes
]

Variations = typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_variations
    typ.Tuple[NDArray, ...],  # tt_variations
]


def fv_variations_from_vector(
        flat:               NDArray,            # shape=(size,)
        variation_shapes:   VariationShapes,
        stack_shape:        typ.Sequence[int] = (),
) -> Variations:
    '''Rebuild variation cores from a 1D vector (inverse of flattening the variation cores).

    Each core is reshaped to ``stack_shape + core_shape``, consuming the flat vector in order
    (tucker-variation cores first, then tt-variation cores).
    '''
    xnp, _, _ = get_backend(False, tree_contains_jax(flat))
    flat = xnp.asarray(flat)
    tucker_shapes, tt_shapes = variation_shapes
    full = ([tuple(stack_shape) + tuple(s) for s in tucker_shapes]
            + [tuple(stack_shape) + tuple(s) for s in tt_shapes])
    cores, o = [], 0
    for shp in full:
        n = math.prod(shp)
        cores.append(flat[o:o + n].reshape(shp))
        o += n
    nt = len(tucker_shapes)
    return tuple(cores[:nt]), tuple(cores[nt:])


def fv_variation_shapes(
        frame:  typ.Tuple,   # (U, O, P, Q) frame data; len=d each, elm_shape=stack_shape+(...)
) -> VariationShapes:        # (tucker_variation_shapes, tt_variation_shapes); WITHOUT the stack
    '''The variation shapes of a frame: ``(nD_i, N_i)`` per Tucker variation and
    ``(rL_i, nU_i, rR_i)`` per TT variation, read off the frame cores.

    The backend twin of the frontend ``T3Frame.variation_shapes`` -- and the ONE place the
    frame-to-variation axis convention is written down for the ragged layer. Frame bookkeeping is
    non-obvious (the TT variation takes its left bond from the core itself but its right bond from the
    core's own trailing axis, which is the NEXT bond -- the gauge shift ``left[:-1]`` / ``right[1:]``
    in mask form), so open-coding these indices is how they drift. Feed the result to
    :py:func:`fv_variations_zeros` / :py:func:`fv_variations_randn`.'''
    up, down, left, right = frame
    d = len(up)
    tucker_variation_shapes = tuple((down[i].shape[-2], up[i].shape[-1]) for i in range(d))
    tt_variation_shapes = tuple((left[i].shape[-3], up[i].shape[-2], right[i].shape[-1])
                                for i in range(d))
    return tucker_variation_shapes, tt_variation_shapes


def fv_variations_zeros(
        variation_shapes:   VariationShapes,
        stack_shape:        typ.Sequence[int] = (),
        use_jax:            bool = False,
) -> Variations:
    '''All-zero variation cores of the given variation shapes (and stack).'''
    xnp, _, _ = get_backend(False, use_jax)
    tucker_shapes, tt_shapes = variation_shapes
    tucker = tuple(xnp.zeros(tuple(stack_shape) + tuple(s)) for s in tucker_shapes)
    tt = tuple(xnp.zeros(tuple(stack_shape) + tuple(s)) for s in tt_shapes)
    return tucker, tt


def fv_variations_randn(
        variation_shapes:   VariationShapes,
        stack_shape:        typ.Sequence[int] = (),
        use_jax:            bool = False,
) -> Variations:
    '''I.i.d. standard-normal variation cores of the given variation shapes (and stack).'''
    tucker_shapes, tt_shapes = variation_shapes
    tucker = tuple(randn(*(tuple(stack_shape) + tuple(s)), use_jax=use_jax) for s in tucker_shapes)
    tt = tuple(randn(*(tuple(stack_shape) + tuple(s)), use_jax=use_jax) for s in tt_shapes)
    return tucker, tt


def fv_variations_unit(
        variation_shapes:   VariationShapes,
        index:              typ.Tuple[bool, int, typ.Sequence[int]],  # (use_tt_coordinate, i, within_index)
        stack_shape:        typ.Sequence[int] = (),
        use_jax:            bool = False,
) -> Variations:
    '''Canonical unit variation: all-zero cores except a single entry set to 1.

    ``index = (use_tt_coordinate, i, within_index)`` selects the family (tt if ``use_tt_coordinate``
    else tucker), the core position ``i``, and the within-core entry (broadcast over the stack).
    '''
    use_tt_coordinate, i, within_index = index
    tucker_shapes, tt_shapes = variation_shapes
    tucker = [np.zeros(tuple(stack_shape) + tuple(s)) for s in tucker_shapes]
    tt = [np.zeros(tuple(stack_shape) + tuple(s)) for s in tt_shapes]
    (tt if use_tt_coordinate else tucker)[i][(Ellipsis,) + tuple(within_index)] = 1.0
    if use_jax:
        tucker = [to_jax(c) for c in tucker]
        tt = [to_jax(c) for c in tt]
    return tuple(tucker), tuple(tt)


def fv_frame_reverse(
        frame: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # up_tucker_cores  (mode order reversed)
    typ.Tuple[NDArray, ...],  # down_tt_cores
    typ.Tuple[NDArray, ...],  # left_tt_cores    (= reversed old right cores)
    typ.Tuple[NDArray, ...],  # right_tt_cores   (= reversed old left cores)
]:
    '''Reverse the mode order of a T3 frame 4-tuple ``(up, down, left, right)``.

    The left and right TT families **swap roles**: reversing a left-orthogonal chain yields a
    right-orthogonal one, so the new left family is the reversed old *right* family and vice versa
    (the up-tucker family is reversed; the down family is reversed per :py:func:`tt_reverse`). The
    redundant left/right store makes this exact with no re-orthogonalization. Inverse of itself.
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    rev = tt_operations.tt_reverse
    return (tuple(U.copy() for U in up_tucker_cores[::-1]),
            rev(down_tt_cores),
            rev(right_tt_cores),   # old right -> new left
            rev(left_tt_cores))    # old left  -> new right


def fv_frame_orthogonality_residual(
        frame: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
) -> NDArray:  # shape = stack_shape (per stack element; scalar/0-d when unstacked)
    '''Max deviation from orthogonality of the four frame core families, **per stack element**.

    Checks each stacked block's gram against the identity:

    - up_tucker U_i (all i), outer/down D_i (all i),
    - left L_i (i=0..d-2), right R_i (i=1..d-1).

    The last left core and first right core are boundary remainders and are not checked. Returns the max
    absolute deviation reduced over the **non-stack** axes (shape ``stack_shape``); a caller thresholds it
    (``<= atol``) for a per-element boolean orthogonality test.
    '''
    UU, DD, LL, RR = frame
    d = len(UU)
    xnp, _, _ = get_backend(False, tree_contains_jax(frame))

    def _dev(gram, n):  # max over the two gram axes only -> keep stack (the leading '...')
        return xnp.max(xnp.abs(gram - xnp.eye(n)), axis=(-2, -1))

    devs = []
    for ii in range(d):
        U = UU[ii]
        D = DD[ii]
        devs.append(_dev(xnp.einsum('...io,...jo->...ij', U, U), U.shape[-2]))
        devs.append(_dev(xnp.einsum('...iaj,...ibj->...ab', D, D), D.shape[-2]))
    for ii in range(d - 1):
        L = LL[ii]
        devs.append(_dev(xnp.einsum('...iaj,...iak->...jk', L, L), L.shape[-1]))
    for ii in range(1, d):
        R = RR[ii]
        devs.append(_dev(xnp.einsum('...iaj,...kaj->...ik', R, R), R.shape[-3]))
    return xnp.max(xnp.stack(devs), axis=0)   # max over the checks, keep stack_shape


def fv_frame_consistency_residual(
        frame: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
) -> NDArray:  # shape = stack_shape (per stack element; scalar/0-d when unstacked)
    '''Relative Frobenius mismatch between the left- and right-canonical reconstructions of the base
    point (``up`` over ``left`` vs ``up`` over ``right``), **per stack element**.

    Returns ``||left - right|| / max(1, ||right||)`` over the dense **mode** axes (the norm is reduced over
    the non-stack axes, so the result has shape ``stack_shape``); a caller thresholds it (``<= rtol``) for a
    per-element boolean consistency test. EXPENSIVE -- densifies both reconstructions.
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    xnp, _, _ = get_backend(False, tree_contains_jax(frame))
    d = len(up_tucker_cores)
    left = t3_conversions.t3_to_dense((up_tucker_cores, left_tt_cores))
    right = t3_conversions.t3_to_dense((up_tucker_cores, right_tt_cores))
    mode_axes = tuple(range(left.ndim - d, left.ndim))   # the d physical-mode axes; stack axes lead
    num = xnp.sqrt(xnp.sum((left - right) ** 2, axis=mode_axes))   # Frobenius over modes -> stack_shape
    den = xnp.sqrt(xnp.sum(right ** 2, axis=mode_axes))
    return num / xnp.maximum(1.0, den)


def fv_absorb_weights(
        variations: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_variations V, tt_variations H)
        weights:    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray],
                              typ.Sequence[NDArray], typ.Sequence[NDArray]],   # (up, down, left, right), each len=d
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # weighted tucker_variations, elm_shape=stack+(nDi, Ni)
    typ.Tuple[NDArray, ...],  # weighted tt_variations,     elm_shape=stack+(rLi, nUi, rRi)
]:
    """Absorb the four-family metric weights into the VARIATION cores (the tangent metric on coordinates,
    Approach-1 / metric-on-variations): ``down`` -> V's ``nD`` leg; ``up``/``left``/``right`` -> H's
    ``nU``/``rL``/``rR`` legs. The frame is left orthonormal and untouched. ``corewise_stack_norm`` of the
    result is the weighted (Grasedyck-Kramer) tangent norm. All families are len=d (one per variation core).

    **The weight is frame-like (stack ``C``) while the variations carry ``K + C``, and the broadcast is
    free**: the single leading ``'...'`` right-aligns, so one metric per base point lifts over the ``K``
    tangent vectors at that point. That works *only because* ``C`` is innermost -- the library-wide
    frame-inner convention (``docs/batching_and_stacking.md``). Do not read the weight as carrying
    ``K+C``: it does not, and conflating the two was a real bug once
    (``docs/contributor/weighted_internals.md``)."""
    xnp, _, _ = get_backend(False, tree_contains_jax((variations, weights)))
    V_cores, H_cores = variations
    up, down, left, right = weights
    wV = tuple(xnp.einsum('...i,...io->...io', dn, V) for dn, V in zip(down, V_cores))
    wH = tuple(xnp.einsum('...aib,...a,...i,...b->...aib', H, lf, u, rt)
               for H, lf, u, rt in zip(H_cores, left, up, right))
    return wV, wH


def fv_weights_consistent(
        variations: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (V, H)
        weights:    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray],
                              typ.Sequence[NDArray], typ.Sequence[NDArray]],   # (up, down, left, right)
) -> bool:                                                                     # True iff shape-consistent
    """True iff the four weight families (each len=d) can be absorbed into ``variations`` (non-raising).

    Ranks: up<->H.nU (axis -2), down<->V.nD (axis -2), left<->H.rL (axis -3), right<->H.rR (axis -1).

    Stacks -- the **trailing rule**: a weight is **frame-like**, carrying the frame stack ``C``, while the
    variations carry ``K + C`` (a ``K``-batch of tangents at one frame shares the one metric). So the
    weight's stack must be the **trailing (inner) part** of the variation stack -- exactly the rule
    ``check_fv_pair`` applies to a (frame, variations) pair, with the weight playing the frame's role.
    Absorption then broadcasts ``C`` over ``K + C`` for free through the leading ``'...'`` (which works
    because ``C`` is innermost). ``K`` may be empty, the common case.

    Like the variations themselves, this predicate is **blind to the frame**: a weight whose stack is the
    whole variation stack also passes, reading as ``C_w = K + C`` (that many base points, one tangent each)
    -- a legitimate absorption. Whether the weight is the metric of *this* tangent's frame needs the frame,
    and is checked by ``frame_variations_format.check_fw_pair`` at the tangent level.
    """
    V_cores, H_cores = variations
    up, down, left, right = weights
    d = len(V_cores)
    if not (len(up) == len(down) == len(left) == len(right) == d):
        return False
    var_stack = V_cores[0].shape[:-2]     # K + C
    stack = up[0].shape[:-1]              # C -- the weight is frame-like (all four families share it)
    if var_stack[len(var_stack) - len(stack):] != stack:  # C must be the trailing part of K + C
        return False
    for i in range(d):
        if up[i].shape != stack + (H_cores[i].shape[-2],):    return False
        if down[i].shape != stack + (V_cores[i].shape[-2],):  return False
        if left[i].shape != stack + (H_cores[i].shape[-3],):  return False
        if right[i].shape != stack + (H_cores[i].shape[-1],): return False
    return True


def fv_concatenate_weights(weights_A, weights_B):  # each (up, down, left, right) -> concatenated, ranks add
    """Per-edge concatenation of two frame-weight 4-tuples (the '+' combine; ranks add). Last-axis."""
    xnp, _, _ = get_backend(False, tree_contains_jax((weights_A, weights_B)))
    return tuple(tuple(xnp.concatenate([a, b], axis=-1) for a, b in zip(fA, fB))
                 for fA, fB in zip(weights_A, weights_B))


def fv_kronecker_weights(weights_A, weights_B):  # each (up, down, left, right) -> Kronecker, ranks multiply
    """Per-edge Kronecker product of two frame-weight 4-tuples (the Hadamard combine; ranks multiply).
    Last-axis outer product broadcasting the shared stack (A-major); NOT np.kron."""
    xnp, _, _ = get_backend(False, tree_contains_jax((weights_A, weights_B)))

    def kv(a, b):
        ss = xnp.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        return (a[..., :, None] * b[..., None, :]).reshape(ss + (a.shape[-1] * b.shape[-1],))

    return tuple(tuple(kv(a, b) for a, b in zip(fA, fB)) for fA, fB in zip(weights_A, weights_B))


def t3weights_to_t3frameweights(
        t3_weights: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_weights, tt_weights)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # up_weights    = tucker_weights
    typ.Tuple[NDArray, ...],  # down_weights  = tucker_weights
    typ.Tuple[NDArray, ...],  # left_weights  = tt_weights[:-1]
    typ.Tuple[NDArray, ...],  # right_weights = tt_weights[1:]
]:
    """Build frame weights (a tangent metric) from base-point edge weights ``(tucker_weights, tt_weights)``.

    ``up = down = tucker_weights``; ``left = tt_weights[:-1]``, ``right = tt_weights[1:]``. The TT slicing
    encodes the convention that ``H_i``'s left bond is TT bond ``i`` and its right bond is bond ``i+1`` --
    non-obvious, hence a named function. The result is a valid ``T3FrameWeights``; it pairs with a
    **minimal-rank** tangent (where the down/complement rank ``nD`` equals the Tucker rank ``nU``, e.g. from
    ``t3svd``). A non-minimal tangent has ``nD < nU`` and would mismatch the down family at use."""
    tucker_weights, tt_weights = t3_weights
    up = tuple(tucker_weights)
    down = tuple(tucker_weights)
    left = tuple(tt_weights[:-1])
    right = tuple(tt_weights[1:])
    return up, down, left, right


def fv_weighted_norm(
        variations: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (V, H)
        weights:    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray],
                              typ.Sequence[NDArray], typ.Sequence[NDArray]],   # (up, down, left, right)
        n_stack:    int,                                                       # leading K+C stack axes kept
) -> NDArray:                                                                  # weighted norm, shape=stack
    """Weighted (Grasedyck-Kramer) coordinate norm of a tangent's variations: the corewise stack-norm of
    the weight-absorbed variations. The frame (orthonormal) is not needed. Backend twin of
    ``T3Tangent.weighted_norm``."""
    return corewise.corewise_stack_norm(fv_absorb_weights(variations, weights), n_stack)


def fv_weighted_inner(
        variations_A: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (V, H) of A
        variations_B: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (V, H) of B
        weights:      typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray],
                                typ.Sequence[NDArray], typ.Sequence[NDArray]],   # one metric (up, down, left, right)
        n_stack:      int,                                                       # leading K+C stack axes kept
) -> NDArray:                                                                    # weighted inner, shape=stack
    """Weighted coordinate inner product ``<absorb(W,A), absorb(W,B)>`` w.r.t. one metric ``weights`` --
    the corewise stack-dot of the two weight-absorbed variations. The caller checks same-frame. Backend
    twin of ``T3Tangent.weighted_inner``."""
    return corewise.corewise_stack_dot(fv_absorb_weights(variations_A, weights),
                                       fv_absorb_weights(variations_B, weights), n_stack)
