# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tucker_tensor_train.t3_operations as ragged_operations
import t3toolbox.backend.uniform_tucker_tensor_train.ut3_operations as uniform_operations
import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.tucker_tensor_train.t3_orthogonalization as ragged_orth
import t3toolbox.backend.uniform_tucker_tensor_train.ut3_orthogonalization as uniform_orth
from t3toolbox.backend.common import *

__all__ = [
    'orthogonal_representations',
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

