# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tucker_tensor_train.ragged_t3_operations as ragged_operations
import t3toolbox.backend.uniform_tucker_tensor_train.uniform_t3_operations as uniform_operations
import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.tucker_tensor_train.ragged_orthogonalization as ragged_orth
import t3toolbox.backend.uniform_tucker_tensor_train.uniform_orthogonalization as uniform_orth
from t3toolbox.backend.common import *

__all__ = [
    'bc_to_t3',
    'orthogonal_representations',
]

def bc_to_t3(
        ii: int, # index of coordinate
        use_tt_coord: bool, # If True, use TT coordinate. If False, use Tucker coordinate
        basis: typ.Union[
            typ.Tuple[
                typ.Tuple[NDArray, ...],  # up_tucker_cores
                typ.Tuple[NDArray, ...],  # down_tucker_cores
                typ.Tuple[NDArray, ...],  # left_tt_cores
                typ.Tuple[NDArray, ...],  # right_tucker_cores
            ], # ragged
            typ.Tuple[
                NDArray,  # up_tucker_supercore
                NDArray,  # down_tucker_supercore
                NDArray,  # left_tt_supercore
                NDArray,  # right_tucker_supercore
            ], # uniform
        ],
        coords: typ.Union[
            typ.Tuple[
                typ.Tuple[NDArray, ...],  # tucker_coordinates
                typ.Tuple[NDArray, ...],  # tt_coordinates
            ], # ragged
            typ.Tuple[
                NDArray,  # tucker_coords_supercore
                NDArray,  # tt_coords_supercore
            ], # uniform
        ],
        use_jax: bool = False,
) -> typ.Union[
    typ.Tuple[
        typ.Tuple[NDArray,...], # tucker_cores
        typ.Tuple[NDArray,...], # tt_cores
    ], # ragged
    typ.Tuple[
        NDArray, # tucker_supercore
        NDArray, # tt_supercore
    ], # uniform
]:
    '''Convert ith basis-coordinates representation to TuckerTensorTrain.
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_coords, tt_coords = coords

    is_uniform = is_ndarray(up_tucker_cores)
    xnp, _, _ = get_backend(True, use_jax)

    if use_tt_coord:
        x_tucker_cores = up_tucker_cores

        LL = left_tt_cores[:ii]
        H = tt_coords[ii]
        RR = right_tt_cores[ii+1:]
        if is_uniform:
            x_tt_cores = xnp.concatenate([LL, H.reshape((1,)+H.shape), RR])
        else:
            x_tt_cores = tuple(LL) + (H,) + tuple(RR)
    else:
        left_UU = up_tucker_cores[:ii]
        V = tucker_coords[ii]
        right_UU = up_tucker_cores[ii+1:]
        if is_uniform:
            x_tucker_cores = xnp.concatenate([left_UU, V.reshape((1,)+V.shape), right_UU])
        else:
            x_tucker_cores = tuple(left_UU) + (V,) + tuple(right_UU)

        LL = left_tt_cores[:ii]
        D = down_tt_cores[ii]
        RR = right_tt_cores[ii+1:]
        if is_uniform:
            x_tt_cores = xnp.concatenate([LL, D.reshape((1,)+D.shape), RR])
        else:
            x_tt_cores = tuple(LL) + (D,) + tuple(RR)

    return x_tucker_cores, x_tt_cores

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
        squash: bool = True,
        use_jax: bool = False,
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
    '''Construct base-variation representations of TuckerTensorTrain with orthogonal base.
    '''
    is_uniform = is_ndarray(x[0])

    if is_uniform:
        squash_tails = lambda tk, tt: (tk, uniform_operations.uniform_squash_tt_tails(tt))
        up_orthogonalize_tucker_cores = uniform_orth.up_orthogonalize_uniform_tucker_cores
        down_orthogonalize_tt_cores = uniform_orth.down_orthogonalize_uniform_tt_cores
    else:
        squash_tails = lambda tk, tt: (tk, ragged_operations.squash_tt_tails(tt))
        up_orthogonalize_tucker_cores = ragged_orth.up_orthogonalize_tucker_cores
        down_orthogonalize_tt_cores = ragged_orth.down_orthogonalize_tt_cores

    if squash:
        x = squash_tails(*x)

    if not already_left_orthogonal:
        # Orthogonalize Tucker cores upward to get up_tt_cores U
        up_tucker_cores, tt_cores = up_orthogonalize_tucker_cores(
            x, use_jax=use_jax,
        )

        # Sweep left-to-right, generating left orthogonal tt_cores L
        left_tt_cores = orth.left_orthogonalize_tt_cores(
            tt_cores, use_jax=use_jax,
        )
    else:
        up_tucker_cores, left_tt_cores = x

    # Sweep right-to-left, generating tt_variations H, and right orthogonal tt_cores R
    right_tt_cores, tt_variations = orth.right_orthogonalize_tt_cores(
        left_tt_cores, return_variation_cores=True, use_jax=use_jax,
    )

    # Orthogonalize TT cores downward to get outer_tt_cores O and tucker_variations V
    tucker_variations, down_tt_cores = down_orthogonalize_tt_cores(
        (up_tucker_cores, tt_variations), use_jax=use_jax,
    )

    base = (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
    variation = (tucker_variations, tt_variations)
    return base, variation

