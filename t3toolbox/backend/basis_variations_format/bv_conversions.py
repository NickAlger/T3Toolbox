# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [

]

from t3toolbox.backend.common import NDArray, is_ndarray, get_backend


def bv_to_t3(
        index: typ.Tuple[
            bool, # If True, use TT coordinate. If False, use Tucker coordinate
            int, # index of coordinate
        ],
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
        variations: typ.Union[
            typ.Tuple[
                typ.Tuple[NDArray, ...],  # tucker_variations
                typ.Tuple[NDArray, ...],  # tt_variations
            ], # ragged
            typ.Tuple[
                NDArray,  # tucker_variations_supercore
                NDArray,  # tt_variations_supercore
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
    '''Convert ith basis-variation representation to TuckerTensorTrain.
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations

    is_uniform = is_ndarray(up_tucker_cores)
    xnp, _, _ = get_backend(True, use_jax)

    use_tt_coord, ii = index

    if use_tt_coord:
        x_tucker_cores = up_tucker_cores

        LL = left_tt_cores[:ii]
        H = tt_variations[ii]
        RR = right_tt_cores[ii+1:]
        if is_uniform:
            x_tt_cores = xnp.concatenate([LL, H.reshape((1,)+H.shape), RR])
        else:
            x_tt_cores = tuple(LL) + (H,) + tuple(RR)
    else:
        left_UU = up_tucker_cores[:ii]
        V = tucker_variations[ii]
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
