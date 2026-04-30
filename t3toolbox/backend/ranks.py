# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.linalg as linalg
from t3toolbox.backend.common import *

__all__ = [
    'compute_minimal_ranks',
]


def compute_minimal_ranks(
        shape: typ.Sequence[int], # (N0, ..., N(d-1))
        tucker_ranks: typ.Union[
            typ.Sequence[int], # (n0,...,n(d-1))
            NDArray, # dtype=int, shape=(d,) + stack_shape
        ],
        tt_ranks: typ.Union[
            typ.Sequence[int], # (r0,...,rd)
            NDArray, # dtype=int, shape=(d+1,) + stack_shape
        ],
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Union[
        typ.Tuple[int,...],  # (n0',...,n(d-1)')
        NDArray,  # dtype=int, shape=(d,) + stack_shape
    ], # new_tucker_ranks
    typ.Union[
        typ.Tuple[int,...],  # (r0',...,rd')
        NDArray,  # dtype=int, shape=(d+1,) + stack_shape
    ], # new_tt_ranks
]:
    '''Find minimal ranks for a generic Tucker tensor train with a given structure.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    is_sequence: bool = False
    if isinstance(tucker_ranks, typ.Sequence):
        is_sequence = True

    tucker_ranks = xnp.array(tucker_ranks)
    tt_ranks = xnp.array(tt_ranks)

    d = len(shape)
    assert(len(tucker_ranks) == d)
    assert(len(tt_ranks) == d+1)

    new_tucker_ranks   = list(tucker_ranks)
    new_tt_ranks       = list(tt_ranks)

    for ii in range(d):
        new_tucker_ranks[ii] = xnp.minimum(new_tucker_ranks[ii], shape[ii])

    new_tt_ranks[-1] = xnp.ones(tt_ranks.shape[1:], dtype=int)
    for ii in range(d-1, 0, -1):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        new_tt_ranks[ii] = np.minimum(rL, n*rR)

    new_tt_ranks[0] = xnp.ones(tt_ranks.shape[1:], dtype=int)
    for ii in range(d):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        n = np.minimum(n, rL*rR)
        rR = np.minimum(rR, rL*n)
        new_tucker_ranks[ii] = n
        new_tt_ranks[ii+1] = rR

    if is_sequence:
        new_tucker_ranks = tuple(int(n) for n in new_tucker_ranks)
        new_tt_ranks = tuple(int(r) for r in new_tt_ranks)
    else:
        new_tucker_ranks = xnp.array(new_tucker_ranks)
        new_tt_ranks = xnp.array(new_tt_ranks)

    return new_tucker_ranks, new_tt_ranks
