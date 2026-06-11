# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
from __future__ import annotations

import typing as typ
import numpy as np

import t3toolbox.backend.bv_conversions as bv_conversions
import t3toolbox.backend.t3_operations as ragged_operations
from t3toolbox.backend.common import *

__all__ = [
    'tangent_to_dense',
]


def tangent_to_dense(
        basis:      typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        variations: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations
            typ.Sequence[NDArray],  # tt_variations
        ],
        include_shift:  bool = False,  # False: tangent vector v. True: base point + v.
        use_jax:        bool = False,
) -> NDArray:  # dense tangent vector. shape=stack_shape+(N0,...,N(d-1))
    """Form the dense tensor represented by a basis-variations tangent vector.

    The tangent vector is the sum of the 2d single-core-replacement terms (one per Tucker hole
    and one per TT hole). This is stack-aware: leading stack axes ride along through ``bv_to_t3``
    and ``to_dense``.
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations

    num_cores = len(tucker_variations)

    terms = [bv_conversions.bv_to_t3((False, ii), basis, variations, use_jax=use_jax) for ii in range(num_cores)]
    terms += [bv_conversions.bv_to_t3((True, ii), basis, variations, use_jax=use_jax) for ii in range(num_cores)]

    V = ragged_operations.to_dense(terms[0])
    for term in terms[1:]:
        V = V + ragged_operations.to_dense(term)

    if include_shift:
        P = ragged_operations.to_dense((up_tucker_cores, left_tt_cores))
        V = P + V

    return V
