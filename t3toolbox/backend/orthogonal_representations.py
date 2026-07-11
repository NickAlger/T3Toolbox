# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.t3_operations as ragged_operations
import t3toolbox.backend.ut3_operations as uniform_operations
import t3toolbox.backend.tt_orthogonalization as orth
import t3toolbox.backend.t3_orthogonalization as ragged_orth
import t3toolbox.backend.ut3_orthogonalization as uniform_orth
from t3toolbox.backend.common import *

__all__ = [
    'orthogonal_representations',
    'frame_orthogonality_residual',
    'frame_consistency_residual',
]


def orthogonal_representations(
        x: typ.Union[
            typ.Tuple[
                typ.Tuple[NDArray,...], # tucker_cores
                typ.Tuple[NDArray,...], # tt_cores
            ], # ragged
            typ.Tuple[
                NDArray, # tucker_supercore
                NDArray, # tt_supercore
            ], # uniform
        ],
        already_left_orthogonal: bool = False,
        squash_tails: bool = True,
) -> typ.Union[
    typ.Tuple[
        typ.Tuple[
            typ.Tuple[NDArray,...], # up_tucker_cores
            typ.Tuple[NDArray, ...],  # down_tucker_cores
            typ.Tuple[NDArray,...], # left_tt_cores
            typ.Tuple[NDArray,...], # right_tucker_cores
        ],
        typ.Tuple[
            typ.Tuple[NDArray,...], # tucker_variations
            typ.Tuple[NDArray,...], # tt_variations
        ],
    ], # ragged
    typ.Tuple[
        typ.Tuple[
            NDArray,  # up_tucker_supercore
            NDArray,  # down_tucker_supercore
            NDArray,  # left_tt_supercore
            NDArray,  # right_tucker_supercore
        ],
        typ.Tuple[
            NDArray,  # tucker_variations_supercore
            NDArray,  # tt_variations_supercore
        ],
    ],  # uniform
]:
    '''Construct frame-variation representations of TuckerTensorTrain with orthogonal frame.

    Sweeping orthogonalization (Algorithm 11) producing the representations (45)-(46), Appendix A.3,
    of Alger et al. (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141). NOTE: the
    left/right sweep order here differs from Algorithm 11 (left-then-right vs the paper's
    right-then-left); the resulting orthogonal representations are equivalent.
    '''
    is_uniform = is_ndarray(x[0])

    if is_uniform:
        # uniform path operates on bare (masked) supercores -- the (n,N)/(rL,n,rR) arrays, not .data.
        squash_tails = lambda tk, tt: (tk, tt_operations.tt_squash_tails(tt))
        up_orthogonalize_tucker_cores = lambda x: uniform_orth.down_orthogonalize_tucker_supercores(*x)
        down_orthogonalize_tt_cores = lambda x: uniform_orth.up_orthogonalize_tt_supercores(*x)
    else:
        squash_tails = lambda tk, tt: (tk, tt_operations.tt_squash_tails(tt))
        up_orthogonalize_tucker_cores = ragged_orth.t3_down_orthogonalize_tucker_cores
        down_orthogonalize_tt_cores = ragged_orth.t3_up_orthogonalize_tt_cores

    if squash_tails:
        x = squash_tails(*x)

    if not already_left_orthogonal:
        # Orthogonalize Tucker cores upward to get up_tt_cores U
        up_tucker_cores, tt_cores = up_orthogonalize_tucker_cores(x)

        # Sweep left-to-right, generating left orthogonal tt_cores L
        left_tt_cores = orth.tt_left_orthogonalize(tt_cores)
    else:
        up_tucker_cores, left_tt_cores = x

    # Sweep right-to-left, generating tt_variations H, and right orthogonal tt_cores R
    right_tt_cores, tt_variations = orth.tt_right_orthogonalize(
        left_tt_cores, return_variation_cores=True,
    )

    # Orthogonalize TT cores downward to get outer_tt_cores O and tucker_variations V
    tucker_variations, down_tt_cores = down_orthogonalize_tt_cores(
        (up_tucker_cores, tt_variations),
    )

    frame = (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
    variation = (tucker_variations, tt_variations)
    return frame, variation


def frame_orthogonality_residual(
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


def frame_consistency_residual(
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

