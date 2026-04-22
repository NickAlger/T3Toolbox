# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tucker_tensor_train.t3_operations as t3_ops
from t3toolbox.backend.common import *

__all__ = [

]

from t3toolbox.backend.tucker_tensor_train import t3_operations as t3_ops
from t3toolbox.backend.uniform_tucker_tensor_train.ut3_masking import make_uniform_masks
from t3toolbox.backend.common import *


def t3_to_ut3(
        x: typ.Tuple[
            typ.Tuple[NDArray,...], # tt_cores
            typ.Tuple[NDArray,...], # tucker_cores
        ],
        d: int = None,
        N: int = None,
        n: int = None,
        r: int = None,
        squash_tails: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # tucker_supercore
    NDArray, # tt_supercore
    NDArray, # shape_mask
    NDArray, # tucker_edge_mask
    NDArray, # tt_edge_mask
]:
    """Convert TuckerTensorTrain to UniformTuckerTensorTrain.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    if squash_tails:
        x = (x[0], t3_ops.squash_tt_tails(x[1], use_jax=use_jax))

    tucker_cores, tt_cores = x

    shape = tuple([B.shape[-1] for B in tucker_cores])
    tucker_ranks = tuple([B.shape[-2] for B in tucker_cores])
    tt_ranks = tuple([G.shape[-3] for G in tt_cores]) + (tt_cores[-1].shape[-1],)
    stack_shape = tucker_cores[0].shape[:-2]

    d = len(shape) if d is None else d
    N = max(shape) if N is None else N
    n = max(tucker_ranks) if n is None else n
    r = max(tt_ranks) if r is None else r

    padded_shape = (N,)*d
    padded_tucker_ranks = (n,)*d
    padded_tt_ranks = (r,)*(d+1)

    padded_tucker_cores = t3_ops.change_tucker_core_shapes(
        tucker_cores, padded_shape, padded_tucker_ranks, use_jax=use_jax,
    )
    padded_tt_cores = t3_ops.change_tt_core_shapes(
        tt_cores, padded_tucker_ranks, padded_tt_ranks, use_jax=use_jax,
    )

    tucker_supercore = xnp.stack(padded_tucker_cores)
    tt_supercore = xnp.stack(padded_tt_cores)

    shape_masks, tucker_masks, tt_masks = make_uniform_masks(
        shape, tucker_ranks, tt_ranks, stack_shape, N, n, r,
    )

    return tucker_supercore, tt_supercore, shape_masks, tucker_masks, tt_masks


def ut3_to_t3(
        x: typ.Tuple[
            NDArray, # tucker_supercore
            NDArray, # tt_supercore
            NDArray, # shape_mask
            NDArray, # tucker_edge_mask
            NDArray, # tt_edge_mask
        ],
        use_jax: bool = False,
) -> typ.Union[
    typ.Tuple, #
]:
    '''Convert UniformTuckerTensorTrain to TuckerTensorTrain.

    If uniform T3 is stacked, either:
        - return an array-line nesting of tuples containing the T3s (stack_t3s=False),
        - or one stacked T3 (stack_t3s=True)

    Can only return a stacked T3 if the stacked UT3s all have the same structure.
    '''
    xnp, _, _ = get_backend(True, use_jax)

    #
    tucker_supercore, tt_supercore, shape_masks, tucker_masks, tt_masks = x
    stack_shape = tucker_supercore[0].shape[:-2]

    if not stack_shape: # not stacked
        shape_inds  = [xnp.argwhere(em).reshape(-1) for em in list(shape_masks)]
        tucker_inds = [xnp.argwhere(em).reshape(-1) for em in list(tucker_masks)]
        tt_inds     = [xnp.argwhere(em).reshape(-1) for em in list(tt_masks)]

        tucker_cores = tuple([
            B[ii,:][:,jj]
            for ii, jj, B
            in zip(tucker_inds, shape_inds, list(tucker_supercore))
        ])
        tt_cores = tuple([
            G[ii, :, :][:,aa,:][:, :, jj]
            for ii, aa, jj, G
            in zip(tt_inds[:-1], tucker_inds, tt_inds[1:], list(tt_supercore))
        ])
        return tucker_cores, tt_cores

    all_T3s = []
    for ii in range(tucker_supercore.shape[1]):
        xi = (
            tucker_supercore[:, ii],
            tt_supercore[:, ii],
            shape_masks,
            tucker_masks[:, ii],
            tt_masks[:, ii],
        )
        ith_t3 = ut3_to_t3(xi, use_jax=use_jax)
        all_T3s.append(ith_t3)

    all_T3s = tuple(all_T3s)
    return all_T3s
