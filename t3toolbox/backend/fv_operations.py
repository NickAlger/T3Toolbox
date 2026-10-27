# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Operations on raw (frame, variations) data: variation constructors, frame ops, residuals.

``fv_variations_{zeros,randn,unit,from_vector}`` build variation tuples for a given frame;
``fv_frame_reverse`` and the ``fv_frame_*_residual`` checkers operate on the frame alone,
``fv_zero_point_frame`` builds the canonical frame at the zero tensor from a frame's structure, and
``fv_center_factors`` / ``fv_lift_factors`` derive the small per-mode factors that relate the frame
coordinates to raw core variations (the lift of the exact Riemannian Hessian). The
``fv_*_weights`` functions are the weighted-layer tangent metric (absorb into the variation cores).
"""
from __future__ import annotations

import dataclasses as dc
import math
import typing as typ
import numpy as np

import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.t3_operations as t3_operations
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.linalg as linalg
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
    'fv_zero_point_frame',
    'fv_center_factors',
    'LiftFactors',
    'fv_lift_factors',
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


# --------------------------------------------------------------------------------------------------
# The canonical zero-point frame (the default Newton reference; docs/fitting_and_optimization.md §5)
# --------------------------------------------------------------------------------------------------
ZERO_FRAME_SEED = 0   # the Haar draws' seed: a fixed CONVENTION, not a tunable


def _identity_prefix(
        rows:  int,
        cols:  int,
) -> NDArray:          # HOST float64, shape=(rows, cols): ones on the leading diagonal, zeros elsewhere
    """The ``rows x cols`` matrix with ones on its leading diagonal -- orthonormal rows when
    ``rows <= cols``, orthonormal columns when ``rows >= cols``."""
    out = np.zeros((rows, cols))
    k = min(rows, cols)
    out[np.arange(k), np.arange(k)] = 1.0
    return out


def _haar_rows(
        n:     int,    # the number of rows (the group's Tucker rank), n <= N
        N:     int,    # the mode size
        key:   int,    # the stream key: the group's first mode index
        seed:  int,
) -> NDArray:          # HOST float64, shape=(n, N): orthonormal rows, Haar-distributed, sign-fixed
    """``n`` Haar-distributed orthonormal rows in ``R^N``, from the keyed legacy stream
    ``RandomState([seed, key])`` -- rank-independent and nested by construction (see
    :py:func:`fv_zero_point_frame`)."""
    G = np.random.RandomState([seed, key]).standard_normal((n, N))   # row-major: row j is the same for every n > j
    Q, R = np.linalg.qr(G.T)                                         # reduced QR, Q: (N, n)
    return (Q * np.where(np.diag(R) >= 0, 1.0, -1.0)).T              # sign-fixed: unique given G


def fv_zero_point_frame(
        frame: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores. len=d, elm_shape=(nUi, Ni)
            typ.Sequence[NDArray],  # down_tt_cores.   len=d, elm_shape=(rLi, nDi, rR(i+1))
            typ.Sequence[NDArray],  # left_tt_cores.   len=d, elm_shape=(rLi, nUi, rL(i+1))
            typ.Sequence[NDArray],  # right_tt_cores.  len=d, elm_shape=(rRi, nUi, rR(i+1))
        ],
        groups:  typ.Tuple[typ.Tuple[int, ...], ...] = (),  # sharing partition (canonical_groups); () = unshared
        seed:    int = ZERO_FRAME_SEED,                      # the Haar stream seed: a fixed convention
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # up:    seeded Haar rows, one draw per group.     len=d, elm_shape=(nUi, Ni)
    typ.Tuple[NDArray, ...],  # down:  identity-prefix, orthonormal.             len=d, elm_shape=(rLi, nDi, rR(i+1))
    typ.Tuple[NDArray, ...],  # left:  identity-prefix, left-orthogonal; L[d-1] = 0
    typ.Tuple[NDArray, ...],  # right: identity-prefix, right-orthogonal; R[0] = 0
]:
    '''The canonical orthonormal frame **at the zero tensor**, with ``frame``'s structure (its core shapes).

    At ``X = 0`` every orthonormal completion is a valid frame, and the Riemannian gradient's norm there
    depends on which one is chosen. This function fixes the choice by a convention, so that the zero-tensor
    gradient norm -- the default Newton reference of ``backend.optimizers.newton_cg``
    (``g0_reference='zero'``) -- is a deterministic number. Only the frame's **shapes** are read; its values
    are not, so any frame with the right ranks gives the same result.

    The construction:

    - **Tucker factors** (``up``): per sharing group ``g`` (an unshared mode is its own group), with key
      ``k_g = min(g)``, draw ``G = RandomState([seed, k_g]).standard_normal((n_g, N_g))``, take the
      sign-fixed reduced QR of ``G.T``, and set ``U_g = Q.T``: Haar-distributed orthonormal rows, the same
      array for every mode of the group.
    - **TT cores** (``down`` / ``left`` / ``right``): identity prefixes -- exactly orthonormal, no draw --
      with ``left[d-1] = right[0] = 0``, so both base-point reconstructions are the zero tensor.

    Three properties follow, and are what the reference relies on:

    - **Rank independence.** Each group draws from its own keyed stream, so changing one mode's rank or
      size leaves every other mode's factor bit-identical.
    - **Nesting.** QR column ``j`` depends only on draw rows ``<= j``, so the factor at rank ``r`` is the
      leading ``r``-row block of the factor at rank ``r + 1``: across rank-continuation levels the
      reference frames are nested subspaces.
    - **Layer independence.** The uniform twin
      (:py:func:`~t3toolbox.backend.ufv_operations.ufv_zero_point_frame`) builds the same real blocks, so
      the reference is identical on the ragged and uniform layers.

    **Why Haar, and only for the Tucker factors.** The Tucker factor is the only part of the frame that
    meets the data. A coordinate-aligned factor (an identity prefix, or LAPACK's completion of a zero core)
    gives a reference of **exactly zero** whenever the data have no energy on its leading coordinates; a
    Haar factor is generic. The TT cores act on rank indices, which carry no data orientation. The seed is a
    convention, not a parameter worth tuning: another seed moves the reference by an O(1) factor, which is
    harmless for a relative stopping test. ``RandomState``'s legacy stream is frozen across numpy versions;
    the QR is reproducible to ~1e-15 across BLAS builds.

    The result has the template's dtype and backend (numpy or jax, inferred). The draws themselves are
    host numpy by design -- a convention, not data. A **stacked** template (stack ``C != ()``) raises
    ``NotImplementedError``: the zero-point frame exists for the optimizers' reference, and the optimizers
    take an unstacked point (``backend.optimizers._require_unstacked``).

    Examples
    --------
    The zero-point frame of a rank-``(2, 3, 2)`` structure is orthonormal, sits at the zero tensor, and
    copies the template's core shapes:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.fv_conversions as fvc
    >>> import t3toolbox.backend.fv_operations as fvo
    >>> import t3toolbox.backend.t3_conversions as t3c
    >>> def structure(tucker_ranks, tt_ranks):       # any frame with these ranks; only its shapes are read
    ...     x = t3.TuckerTensorTrain.zeros((5, 6, 7), tucker_ranks, tt_ranks)
    ...     return fvc.t3_orthogonal_representations(x.data)[0]
    >>> template = structure((2, 3, 2), (1, 2, 2, 1))
    >>> up, down, left, right = fvo.fv_zero_point_frame(template)
    >>> print([u.shape for u in up], [g.shape for g in left])
    [(2, 5), (3, 6), (2, 7)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]
    >>> bool(fvo.fv_frame_orthogonality_residual((up, down, left, right)) < 1e-14)
    True
    >>> print(float(np.abs(t3c.t3_to_dense((up, left))).max()))   # the base point is the zero tensor
    0.0

    Nested across ranks, and independent of the other modes' ranks:

    >>> up_more = fvo.fv_zero_point_frame(structure((3, 3, 2), (1, 3, 2, 1)))[0]
    >>> bool(np.allclose(up_more[0][:2], up[0]))       # mode 0 at rank 3 extends mode 0 at rank 2
    True
    >>> bool(np.array_equal(up_more[2], up[2]))        # mode 2 is untouched by mode 0's rank
    True

    With sharing, one factor per group:

    >>> template = fvc.t3_orthogonal_representations(
    ...     t3.TuckerTensorTrain.zeros((6, 6, 7), (3, 3, 2), (1, 3, 2, 1)).data)[0]
    >>> up_shared = fvo.fv_zero_point_frame(template, groups=((0, 1), (2,)))[0]
    >>> bool(np.array_equal(up_shared[0], up_shared[1]))
    True
    '''
    up, down, left, right = frame
    d = len(up)
    stack = tuple(up[0].shape[:-2])
    if stack:
        raise NotImplementedError(
            "fv_zero_point_frame: a STACKED template (stack C = %r) is not supported. The zero-point frame "
            "exists for the optimizers' Newton reference, and the optimizers take an unstacked point "
            "(backend.optimizers._require_unstacked); build it per stack element." % (stack,))

    key = list(range(d))                                   # the stream key of each mode: its group's first mode
    for group in groups:
        if len({tuple(up[i].shape) for i in group}) != 1:
            raise ValueError("fv_zero_point_frame: sharing group %r has unequal Tucker-factor shapes %r; a "
                             "shared factor needs one (rank, size) per group"
                             % (tuple(group), [tuple(up[i].shape) for i in group]))
        for i in group:
            key[i] = min(group)

    for i in range(d):                                     # structural: the shapes must admit an orthonormal frame
        (nU, N), (rL, nD, rR) = up[i].shape, down[i].shape
        admissible = (nU <= N and nD <= rL * rR
                      and (i == d - 1 or left[i].shape[2] <= left[i].shape[0] * left[i].shape[1])
                      and (i == 0 or right[i].shape[0] <= right[i].shape[1] * right[i].shape[2]))
        if not admissible:
            raise ValueError("fv_zero_point_frame: the template's core shapes at mode %d admit no orthonormal "
                             "frame (up %r, down %r, left %r, right %r)"
                             % (i, up[i].shape, down[i].shape, left[i].shape, right[i].shape))

    xnp, _, _ = get_backend(False, tree_contains_jax(frame))
    dtype = up[0].dtype

    drawn = {}                                             # one cast Haar factor per group, replicated
    for i in range(d):
        if key[i] not in drawn:
            drawn[key[i]] = xnp.asarray(_haar_rows(up[i].shape[0], up[i].shape[1], key[i], seed), dtype=dtype)
    up_out = tuple(drawn[key[i]] for i in range(d))

    down_out, left_out, right_out = [], [], []
    for i in range(d):
        rL, nD, rR = down[i].shape
        D = np.moveaxis(_identity_prefix(nD, rL * rR).reshape(nD, rL, rR), 0, 1)   # orthonormal over nD
        a, n, b = left[i].shape
        L = _identity_prefix(a * n, b).reshape(a, n, b) if i < d - 1 else np.zeros((a, n, b))
        a, n, b = right[i].shape
        R = _identity_prefix(a, n * b).reshape(a, n, b) if i > 0 else np.zeros((a, n, b))
        down_out.append(xnp.asarray(D, dtype=dtype))
        left_out.append(xnp.asarray(L, dtype=dtype))
        right_out.append(xnp.asarray(R, dtype=dtype))
    return up_out, tuple(down_out), tuple(left_out), tuple(right_out)


# --------------------------------------------------------------------------------------------------
# The frame's center factors, and the lift factors of the exact Riemannian Hessian
# --------------------------------------------------------------------------------------------------
def fv_center_factors(
        frame:  typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores. len=d, elm_shape=C+(nUi, Ni)
            typ.Sequence[NDArray],  # down_tt_cores.   len=d, elm_shape=C+(rLi, nDi, rR(i+1))
            typ.Sequence[NDArray],  # left_tt_cores.   len=d, elm_shape=C+(rLi, nUi, rL(i+1))
            typ.Sequence[NDArray],  # right_tt_cores.  len=d, elm_shape=C+(rRi, nUi, rR(i+1))
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],        # centers H_i = L_i T_i.  len=d,   elm_shape=C+(rLi, nUi, rR(i+1))
    typ.Tuple[NDArray, ...],        # tucker factors S_i^T.   len=d,   elm_shape=C+(nDi, nUi)
    typ.Tuple[NDArray, ...],        # bond factors T_i.       len=d-1, elm_shape=C+(rL(i+1), rR(i+1))
]:
    '''The small per-mode factors that an orthogonal frame's construction produces and discards.

    Recomputed from the STORED cores, by contraction only (no SVD):

    - the **bond factors** ``T_i`` (``i <= d-2``): the right-to-left zipper of the left chain against the
      right chain at bond ``i+1`` (:py:func:`~t3toolbox.backend.tt_operations.tt_zipper_right_to_left`).
      Because the base point is ``X = L_{<i} L_i T_i R_{>i}``, ``T_i = <L_i, H_i>`` and its singular values
      are the TT singular values of ``X`` at bond ``i+1``;
    - the **centers** ``H_i = L_i T_i`` (at the last mode the zipper is the boundary ones matrix -- ``[[1]]``
      on the usual squashed frame -- so ``H_{d-1} = L_{d-1}``): the base point's core when the orthogonality
      center sits at mode ``i`` -- the construction's own TT variations of the point;
    - the **Tucker factors** ``S_i^T = <O_i, H_i>`` (shape ``nDi x nUi``): the base point's Tucker variation
      is ``V_i = S_i^T U_i``, and the singular values of ``S_i`` are the mode-``i`` Tucker singular values of
      ``X``. In particular ``||X||^2 = sum(sigma(S_0)^2) = ||S_0^T||_F^2``.

    The zipper recompute is GEMM-only, exact to roundoff, and consistent with the stored ``O``/``R``
    whatever built the frame; nothing is re-SVD'd, so no sign or degenerate-block choice is made here
    (the measurements are in the sharing design record, ``docs/contributor/sharing_internals.md``). The
    identities above presume an **orthogonal** frame (as from ``t3_orthogonal_representations``); like
    every backend function this one is check-free.

    Polymorphic: ``frame`` may also be the four uniform supercores (``(d,)+C+(...)``), indexed per mode --
    the uniform shared-frame companion relies on this. Two consumers: the shared geometry's companion
    (:py:func:`~t3toolbox.backend.sharing.fv_shared_frame_data`, the centers and ``S_i^T``) and the
    exact Hessian's lift (:py:func:`fv_lift_factors`, ``S_i^T`` and ``T_i``).

    Examples
    --------
    The centers are the construction's own, and the factors carry the dense tensor's spectra:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.fv_conversions as fvc
    >>> import t3toolbox.backend.fv_operations as fvo
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = fvc.t3_orthogonal_representations(x.data)
    >>> centers, tucker_factors_T, bond_factors = fvo.fv_center_factors(frame)
    >>> print([S.shape for S in tucker_factors_T], [T.shape for T in bond_factors])
    [(2, 2), (3, 3), (2, 2)] [(2, 2), (2, 2)]
    >>> print(all(np.allclose(H, H0) for H, H0 in zip(centers, variations[1])))
    True
    >>> X = x.to_dense()
    >>> X1 = np.moveaxis(X, 1, 0).reshape(6, -1)                        # the mode-1 unfolding
    >>> s_tucker = np.linalg.svd(tucker_factors_T[1], compute_uv=False)
    >>> print(np.allclose(s_tucker, np.linalg.svd(X1, compute_uv=False)[:3]))
    True
    >>> s_bond = np.linalg.svd(bond_factors[1], compute_uv=False)       # bond 2: modes (0,1) | (2,)
    >>> print(np.allclose(s_bond, np.linalg.svd(X.reshape(30, 7), compute_uv=False)[:2]))
    True
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    d = len(up_tucker_cores)

    ZZ = tt_operations.tt_zipper_right_to_left(left_tt_cores, right_tt_cores)   # len=d+1, C+(rLi, rRi)
    centers = tuple(contractions.contract('Caib,Cbc->Caic', left_tt_cores[ii], ZZ[ii + 1])
                    for ii in range(d))
    tucker_factors_T = tuple(contractions.contract('Caxb,Caub->Cxu', down_tt_cores[ii], centers[ii])
                             for ii in range(d))
    bond_factors = tuple(ZZ[ii + 1] for ii in range(d - 1))
    return centers, tucker_factors_T, bond_factors


@dc.dataclass(frozen=True, eq=False)   # eq=False -> identity hash/eq: array fields (the SharedFrameData precedent)
class LiftFactors:
    '''The per-frame factors of the lift ``L`` from frame coordinates to raw core variations, built by
    :py:func:`fv_lift_factors` (never stored on a frame).

    Every array field carries the frame stack ``C`` leading; as a jax pytree all of them are LEAVES, so a
    new frame -- or a new damping -- changes leaf values only, never a compilation key. ``eps`` is the
    ABSOLUTE damping actually used (``C``-shaped), kept beside the factors so the lift and its transpose
    cannot disagree about it.

    **Shared (tied) points.** :py:func:`~t3toolbox.backend.sharing.fv_tied_lift_factors` adds the frame's
    SF-T3 companion (``shared_data``, whose sharing partition stays static structure under jax) and one
    damped filter of the group spectrum per nontrivial group (``tied_filters``); the lift then lifts each
    group's Tucker slots to ONE factor direction, solved against the companion's stacked-``S`` SVD, and the
    group modes' entries of ``tucker_pinvs`` / ``tucker_svals`` go unused. Unshared, they are ``None`` /
    ``()``.

    **Uniform frames.** :py:func:`~t3toolbox.backend.ufv_operations.ufv_lift_factors` fills the same fields with
    SUPERCORES (mode index ``d`` leading, then ``C``) where this one holds per-mode tuples, at the frame's padded
    sizes with exactly-zero padding; its ``bond_pinvs`` carries one more slot, the last TT slot's boundary
    pass-through, so it has ``d`` slots (``bond_svals`` keeps ``d - 1``).
    '''
    tucker_pinvs:  typ.Tuple[NDArray, ...]  # (S_i^T)^{+eps}. len=d,   elm_shape=C+(nUi, nDi)
    bond_pinvs:    typ.Tuple[NDArray, ...]  # T_i^{+eps}.     len=d-1, elm_shape=C+(rR(i+1), rL(i+1))
    tucker_svals:  typ.Tuple[NDArray, ...]  # sigma(S_i).     len=d,   elm_shape=C+(nDi,)
    bond_svals:    typ.Tuple[NDArray, ...]  # sigma(T_i).     len=d-1, elm_shape=C+(rR(i+1),)
    eps:           NDArray                  # absolute damping used, shape=C

    shared_data:   typ.Any = None                 # shared points: the frame's SF-T3 companion (sharing.SharedFrameData)
    tied_filters:  typ.Tuple[NDArray, ...] = ()   # per nontrivial group: f(s_g) at eps, elm_shape=C+(q_g,)


def fv_lift_factors(
        frame:              typ.Tuple[
            typ.Sequence[NDArray],                  # up_tucker_cores. len=d, elm_shape=C+(nUi, Ni)
            typ.Sequence[NDArray],                  # down_tt_cores.   len=d, elm_shape=C+(rLi, nDi, rR(i+1))
            typ.Sequence[NDArray],                  # left_tt_cores.   len=d, elm_shape=C+(rLi, nUi, rL(i+1))
            typ.Sequence[NDArray],                  # right_tt_cores.  len=d, elm_shape=C+(rRi, nUi, rR(i+1))
        ],
        curvature_damping:  typ.Union[float, NDArray],  # RELATIVE: eps = curvature_damping * ||X||_HS, >= 0
        rcond:              typ.Optional[float] = None, # relative clip (linalg.damped_pseudo_inverse)
) -> LiftFactors:
    '''The damped pseudo-inverses that lift frame coordinates to raw core variations.

    At an orthogonal frame of ``X = Phi(theta)``, with ``theta = (U, L)`` the frame's own raw cores (the
    left-canonical gauge of the base point), a gauged-or-not coordinate ``v = (V_i, H_i)`` and a raw
    variation ``theta_hat = (Udot_i, Gdot_i)`` represent the same tangent vector,
    ``DPhi(theta)[theta_hat] = E v``, when

    ``Udot_i = (S_i^T)^+ V_i``,   ``Gdot_i = H_i x_3 T_i^+`` (``i <= d-2``),   ``Gdot_{d-1} = H_{d-1}``,

    with ``S_i^T`` and ``T_i`` the center factors (:py:func:`fv_center_factors`). This function returns
    those pseudo-inverses, damped by the Tikhonov filter ``sigma / (sigma^2 + eps^2)``
    (:py:func:`~t3toolbox.backend.linalg.damped_pseudo_inverse`), with the absolute damping

    ``eps = curvature_damping * ||X||_HS``,   ``||X||_HS^2 = sum(sigma(S_0)^2)``,

    so ``curvature_damping`` is scale-free. The lift itself is
    :py:func:`~t3toolbox.backend.tv_operations.tv_corewise_lift`; at a shared point (tied Tucker factors) build
    the factors with :py:func:`~t3toolbox.backend.sharing.fv_tied_lift_factors` instead, which lifts each
    sharing group to one factor direction.

    - **Exactness.** At ``eps = 0`` the lift is exact on every *numerically full-rank* frame from
      ``t3_orthogonal_representations``, minimal or not: a Tucker-slack ``S_i^T`` is wide (``nDi < nUi``) and a
      TT-slack ``T_i`` is tall (``rR < rL``), and on both the pseudo-inverse is an exact one-sided inverse. At a
      rank-deficient point (a zero-padded restart) the clipped directions drop out: the lift then satisfies
      ``DPhi[L v] = E(Q v)`` with ``Q`` the projection off the clipped directions, not ``DPhi[L v] = E v``.
    - **Rank-deficient points.** The filter clips (to exactly 0) every singular value below
      ``rcond * sigma_max``, so a zero-padded rank-continuation restart -- whose padded singular values are
      roundoff -- gives finite factors at any ``eps``, ``eps = 0`` included. At the zero tensor everything is
      clipped: the lift is zero except the undamped last TT slot.
    - **Why it exists.** The lift is the only place an inverse singular value enters the exact Hessian
      (the Weingarten term of Absil, Mahony & Trumpf (2013), "An extrinsic look at the Riemannian
      Hessian"; the TT analogue of Psenka & Boumal (2020), arXiv:2011.13395, ``R_k^{-1}``). The damping
      bounds it by ``1/(2 eps)`` near a rank drop.

    Requires an orthogonal frame (the identities presume it; check-free, like all backend functions).
    Stack-aware: the factors and ``eps`` carry the frame stack ``C``; they broadcast over a tangent stack
    ``K`` in the lift. Dispatch is inferred.

    Examples
    --------
    The absolute damping is relative to the tensor's norm, and at ``curvature_damping = 0`` the factors
    are the exact pseudo-inverses:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.fv_conversions as fvc
    >>> import t3toolbox.backend.fv_operations as fvo
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, _ = fvc.t3_orthogonal_representations(x.data)
    >>> lf = fvo.fv_lift_factors(frame, 1e-6)
    >>> print(bool(np.isclose(lf.eps, 1e-6 * np.linalg.norm(x.to_dense()))))
    True
    >>> lf0 = fvo.fv_lift_factors(frame, 0.0)
    >>> _, tucker_factors_T, bond_factors = fvo.fv_center_factors(frame)
    >>> print(np.allclose(lf0.tucker_pinvs[1], np.linalg.pinv(tucker_factors_T[1])),
    ...       np.allclose(lf0.bond_pinvs[0], np.linalg.pinv(bond_factors[0])))
    True True
    '''
    if isinstance(curvature_damping, (int, float)) and curvature_damping < 0:
        raise ValueError('fv_lift_factors: curvature_damping must be >= 0 (it is relative to ||X||_HS); '
                         'got %r' % (curvature_damping,))
    _, tucker_factors_T, bond_factors = fv_center_factors(frame)

    xnp, _, _ = get_backend(False, tree_contains_jax((tucker_factors_T, curvature_damping)))
    S0T = tucker_factors_T[0]
    norm_X = xnp.sqrt(xnp.sum(S0T * S0T, axis=(-2, -1)))       # ||X||_HS = ||S_0^T||_F, shape=C
    eps = curvature_damping * norm_X

    tucker = tuple(linalg.damped_pseudo_inverse(S_T, eps, rcond) for S_T in tucker_factors_T)
    bond = tuple(linalg.damped_pseudo_inverse(T, eps, rcond) for T in bond_factors)
    return LiftFactors(tucker_pinvs=tuple(p for p, _ in tucker), bond_pinvs=tuple(p for p, _ in bond),
                       tucker_svals=tuple(s for _, s in tucker), bond_svals=tuple(s for _, s in bond),
                       eps=eps)


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


if jax_available:
    import jax

    # The lift factors are a jax pytree whose every array is a LEAF (the per-frame pseudo-inverses, their
    # spectra, the absolute damping eps and, at a shared point, the companion's arrays and the tied filters
    # flow as traced data beside the frame) with no aux of their own: the companion carries its sharing
    # partition as ITS aux, so the structure stays static. A new frame, or a new curvature_damping, changes
    # leaf values only -- never the jit cache key.
    jax.tree_util.register_pytree_node(
        LiftFactors,
        lambda f: ((f.tucker_pinvs, f.bond_pinvs, f.tucker_svals, f.bond_svals, f.eps, f.shared_data,
                    f.tied_filters), None),
        lambda _, ch: LiftFactors(tucker_pinvs=ch[0], bond_pinvs=ch[1], tucker_svals=ch[2],
                                  bond_svals=ch[3], eps=ch[4], shared_data=ch[5], tied_filters=ch[6]),
    )
