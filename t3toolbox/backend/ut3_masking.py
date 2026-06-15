# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'make_uniform_masks',
    'apply_masks_to_cores',
]


def make_uniform_masks(
        shape:          typ.Sequence[int],  # (N0,...,N(d-1))
        tucker_ranks:   NDArray,            # dtype=int, shape=(d,)   + stack_shape
        tt_ranks:       NDArray,            # dtype=int, shape=(d+1,) + stack_shape
        N:              int,                # padded mode dimension, N >= max(Ni)
        n:              int,                # padded Tucker rank,     n >= max(tucker_ranks)
        r:              int,                # padded TT rank,         r >= max(tt_ranks)
        use_jax:        bool = False,       # pure constructor (no array inputs) -> the flag picks output type
) -> typ.Tuple[
    NDArray,  # shape_mask,       dtype=bool, shape=(d, N)                 (no stack: the ambient shape is shared)
    NDArray,  # tucker_edge_mask, dtype=bool, shape=(d,)   + stack_shape + (n,)
    NDArray,  # tt_edge_mask,     dtype=bool, shape=(d+1,) + stack_shape + (r,)
]:
    """Build the prefix edge masks for a uniform Tucker tensor train.

    Slot ``j`` of an edge is marked real iff ``j < rank`` -- the canonical prefix form
    (``docs/uniform_masks_vs_ranks.md``). ``shape_mask`` carries no stack (shape is fixed across the
    stack); the rank masks do, so each stack element may declare its own ranks (the variety -- see
    ``docs/uniform_ranks_and_varieties.md``).
    """
    xnp, _, _ = get_backend(False, use_jax)

    shape        = xnp.asarray(shape)        # (d,)
    tucker_ranks = xnp.asarray(tucker_ranks) # (d,)   + stack_shape
    tt_ranks     = xnp.asarray(tt_ranks)     # (d+1,) + stack_shape

    shape_mask       = xnp.arange(N) < shape[:, None]            # (d, N)
    tucker_edge_mask = xnp.arange(n) < tucker_ranks[..., None]   # (d,)   + stack_shape + (n,)
    tt_edge_mask     = xnp.arange(r) < tt_ranks[..., None]       # (d+1,) + stack_shape + (r,)

    return shape_mask, tucker_edge_mask, tt_edge_mask


def apply_masks_to_cores(
        x: typ.Tuple[
            NDArray,  # tucker_supercore, shape=(d,)+stack_shape+(n,N)
            NDArray,  # tt_supercore,     shape=(d,)+stack_shape+(r,n,r)
            typ.Tuple[
                NDArray,  # shape_mask
                NDArray,  # tucker_edge_mask
                NDArray,  # tt_edge_mask
            ],
        ],
) -> typ.Tuple[
    NDArray,  # masked_tucker_supercore, shape=(d,)+stack_shape+(n,N)
    NDArray,  # masked_tt_supercore,     shape=(d,)+stack_shape+(r,n,r)
]:
    """Zero the padded ("garbage") regions of the supercores by multiplying through the edge masks.
    """
    tucker_supercore, tt_supercore, (shape_mask, tucker_edge_mask, tt_edge_mask) = x
    use_jax = tree_contains_jax((tucker_supercore, tt_supercore))
    xnp, _, _ = get_backend(True, use_jax)

    masked_tucker_supercore = xnp.einsum(
        'd...nN,d...n,dN->d...nN',
        tucker_supercore, tucker_edge_mask, shape_mask,
    )
    masked_tt_supercore = xnp.einsum(
        'd...lnr,d...l,d...n,d...r->d...lnr',
        tt_supercore, tt_edge_mask[:-1], tucker_edge_mask, tt_edge_mask[1:],
    )
    return masked_tucker_supercore, masked_tt_supercore
