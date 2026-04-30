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
        shape:          typ.Tuple[int,...], # len=d
        tucker_ranks:   NDArray, # dtype=int, shape=(d,)+stack_shape
        tt_ranks:       NDArray, # dtype=int, shape=(d+1,)+stack_shape
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

    def _func1(kk, K):
        if np.array(kk).shape == ():
            mask = xnp.concatenate([
                xnp.ones((kk,), dtype=bool),
                xnp.zeros((K - kk,), dtype=bool)
            ])
            return mask
        return [_func1(ki, K) for ki in list(kk)]

    tucker_masks    = [_func1(nni, n) for nni in list(tucker_ranks)]
    tt_masks        = [_func1(rri, r) for rri in list(tt_ranks)]

    tucker_masks = xnp.stack(tucker_masks)
    tt_masks = xnp.stack(tt_masks)

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
