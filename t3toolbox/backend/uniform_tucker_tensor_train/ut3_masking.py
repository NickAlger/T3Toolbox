# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tucker_tensor_train.t3_operations as t3_ops
from t3toolbox.backend.common import *

__all__ = [
    'make_uniform_masks',
    'apply_masks_to_cores',
]

from t3toolbox.backend.common import *


def make_uniform_masks(
        shape:          typ.Tuple[int,...],
        tucker_ranks:   typ.Tuple[int,...],
        tt_ranks:       typ.Tuple[int,...],
        stack_shape:    typ.Tuple[int,...],
        N: int,
        n: int,
        r: int,
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # shape_mask, dtype=bool, shape=(d,N)
    NDArray, # tucker_edge_masks, dtype=bool, shape=(d,)+stack_shape+(n,)
    NDArray, # tt_edge_masks, dtype=bool, shape=(d,)+stack_shape+(r,)
]:
    xnp, xmap, xscan = get_backend(False, use_jax)


    shape_masks = xnp.stack([
        xnp.concatenate([
            xnp.ones((Ni,), dtype=bool),
            xnp.zeros((N-Ni,), dtype=bool),
        ], axis=-1,
        )
        for Ni in shape
    ])

    tucker_masks = xnp.stack([
        xnp.concatenate([
            xnp.ones(stack_shape+(ni,), dtype=bool),
            xnp.zeros(stack_shape+(n-ni,), dtype=bool)
        ], axis=-1,
        )
        for ni in tucker_ranks
    ])

    tt_masks = xnp.stack([
        xnp.concatenate([
            xnp.ones(stack_shape+(ri,), dtype=bool),
            xnp.zeros(stack_shape+(r-ri,), dtype=bool)
        ], axis=-1,
        )
        for ri in tt_ranks
    ])

    return shape_masks, tucker_masks, tt_masks


def apply_masks_to_cores(
        x: typ.Tuple[
            NDArray,  # tucker_supercore
            NDArray,  # tt_supercore
            NDArray,  # shape_mask
            NDArray,  # tucker_edge_mask
            NDArray,  # tt_edge_mask
        ],
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # masked_tucker_supercore
    NDArray, # masked_tt_supercore
]:
    """Applies masking to supercores, replacing unmasked regions with zeros.
    """
    xnp,_,_ = get_backend(True, use_jax)

    tucker_supercore, tt_supercore, shape_mask, tucker_edge_mask, tt_edge_mask = x

    masked_tucker_supercore = xnp.einsum(
        'd...nN,d...n,dN->d...nN',
        tucker_supercore, tucker_edge_mask, shape_mask,
    )
    masked_tt_supercore = xnp.einsum(
        'd...lnr,d...l,d...n,d...r->d...lnr',
        tt_supercore, tt_edge_mask[:-1], tucker_edge_mask, tt_edge_mask[1:],
    )
    return masked_tucker_supercore, masked_tt_supercore
