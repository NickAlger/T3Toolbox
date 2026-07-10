# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
from __future__ import annotations

import math
import typing as typ
import numpy as np

import t3toolbox.backend.t3_operations as t3_operations
from t3toolbox.backend.common import *

__all__ = [
    'absorb_weights_into_tangent_cores',
    'variations_from_vector',
    'zeros_variations',
    'randn_variations',
    'unit_variations',
    'reverse_frame',
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


def variations_from_vector(
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


def zeros_variations(
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


def randn_variations(
        variation_shapes:   VariationShapes,
        stack_shape:        typ.Sequence[int] = (),
        use_jax:            bool = False,
) -> Variations:
    '''I.i.d. standard-normal variation cores of the given variation shapes (and stack).'''
    tucker_shapes, tt_shapes = variation_shapes
    tucker = tuple(randn(*(tuple(stack_shape) + tuple(s)), use_jax=use_jax) for s in tucker_shapes)
    tt = tuple(randn(*(tuple(stack_shape) + tuple(s)), use_jax=use_jax) for s in tt_shapes)
    return tucker, tt


def unit_variations(
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


def reverse_frame(
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
    (the up-tucker family is reversed; the down family is reversed per :py:func:`reverse_tt`). The
    redundant left/right store makes this exact with no re-orthogonalization. Inverse of itself.
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    rev = t3_operations.reverse_tt
    return (tuple(U.copy() for U in up_tucker_cores[::-1]),
            rev(down_tt_cores),
            rev(right_tt_cores),   # old right -> new left
            rev(left_tt_cores))    # old left  -> new right


# NOTE (parked): kept here for safekeeping pending a redesign of weighted tensor networks.
# This is the pre-refactor implementation. It uses the OLD frame-core ordering
# (up, left, right, outer) and is NOT wired into the manifold / T3Tangent API. Do not rely
# on it until the weighting code structure is reworked.
def absorb_weights_into_tangent_cores(
        variation,      # (tucker_variations, tt_variations)
        frame,           # OLD order: (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
        edge_weights = (None, None, None, None),
        use_jax: bool = False,
):
    """Contract edge weights with neighboring cores in frame-variation representation.

    Tensor network diagrams illustrating groupings::

             ____     ________     ____
            /    \\   /        \\   /    \\
        1---wL--L0---wL--H1---wR--R2---wR--1
                |        |        |
              / wU     / wU     / wU
              | |      | |      | |
              | U0     | U1     | U2
              | |      | |      | |
              \\ w      \\ w      \\ w
                |        |        |

    and::

             ____     ________     ____
            /    \\   /        \\   /    \\
        1---wL--L0---wL--O1---wR--R2---wR--1
                |        |        |
              / wU     / wO     / wU
              | |      | |      | |
              | U0     | V1     | U2
              | |      | |      | |
              \\ w      \\ w      \\ w
                |        |        |

    """
    is_uniform = not isinstance(frame[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    (shape_weights,
     up_tucker_weights, outer_tucker_weights,
     left_tt_weights, right_tt_weights,
     ) = edge_weights

    (up_tucker_cores0, left_tt_cores0, right_tt_cores0, outer_tt_cores0) = frame
    (var_tucker_cores0, var_tt_cores0) = variation

    if is_uniform:
        up_tucker_cores = xnp.einsum(
            'di,dio,do->dio', up_tucker_weights, up_tucker_cores0, shape_weights
        )
        var_tucker_cores = xnp.einsum(
            'di,dio,do->dio', outer_tucker_weights, var_tucker_cores0, shape_weights
        )
        left_tt_cores = xnp.einsum(
            'di,diaj->diaj', left_tt_weights, left_tt_cores0
        )
        right_tt_cores = xnp.einsum(
            'diaj,dj->diaj', right_tt_cores0, right_tt_weights
        )
        outer_tt_cores = xnp.einsum(
            'di,diaj,dj->diaj', left_tt_weights, outer_tt_cores0, right_tt_weights
        )
        var_tt_cores = xnp.einsum(
            'di,diaj,dj->diaj', left_tt_weights, var_tt_cores0, right_tt_weights
        )

    else:
        (up_tucker_cores,) = xmap(
            lambda x: (xnp.einsum('i,io,o->io', x[0], x[1], x[2]),),
            (up_tucker_weights, up_tucker_cores0, shape_weights)
        )
        (var_tucker_cores,) = xmap(
            lambda x: (xnp.einsum('i,io,o->io', x[0], x[1], x[2]),),
            (outer_tucker_weights, var_tucker_cores0, shape_weights)
        )
        (left_tt_cores,) = xmap(
            lambda x: (xnp.einsum('i,iaj->iaj', x[0], x[1]),),
            (left_tt_weights, left_tt_cores0)
        )
        (right_tt_cores,) = xmap(
            lambda x: (xnp.einsum('iaj,j->iaj', x[0], x[1]),),
            (right_tt_cores0, right_tt_weights)
        )
        (outer_tt_cores,) = xmap(
            lambda x: (xnp.einsum('i,iaj,j->iaj', x[0], x[1], x[2]),),
            (left_tt_weights, outer_tt_cores0, right_tt_weights)
        )
        (var_tt_cores,) = xmap(
            lambda x: (xnp.einsum('i,iaj,j->iaj', x[0], x[1], x[2]),),
            (left_tt_weights, var_tt_cores0, right_tt_weights)
        )

    weighted_frame = (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    weighted_variation = (var_tucker_cores, var_tt_cores)
    return weighted_variation, weighted_frame
