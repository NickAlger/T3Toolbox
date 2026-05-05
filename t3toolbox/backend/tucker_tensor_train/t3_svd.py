# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tucker_tensor_train.t3_operations as ragged_ops
import t3toolbox.backend.tucker_tensor_train.t3_orthogonalization as ragged_orth
import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.linalg as linalg
from t3toolbox.backend.common import *

__all__ = [
    't3svd',
]

def t3svd(
        x: typ.Tuple[
            typ.Tuple[NDArray,...], # tucker_cores
            typ.Tuple[NDArray,...], # tt_cores
        ],
        min_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        min_tucker_ranks:   typ.Sequence[int] = None,  # len=d
        max_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        max_tucker_ranks:   typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
        squash_tails_first: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[NDArray, ...],  # new_tucker_cores
        typ.Tuple[NDArray, ...],  # new_tt_cores
    ],
    typ.Tuple[NDArray,...], # Tucker singular values, len=d
    typ.Tuple[NDArray,...], # TT singular values, len=d+1
]:
    '''Compute (truncated) T3-SVD of TuckerTensorTrain.
    '''
    is_uniform = False
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if len(x[0][0].shape) > 2:
        raise RuntimeError(
            'T3-SVD cannot be applied to stacked TuckerTensorTrains.\n' +
            'Resulting ranks depend on numerical properties of the T3.\n' +
            'Unstack and apply T3-SVD to each T3 individually.'
        )

    num_cores = len(x[0])

    # make leading and trailing TT-ranks equal to 1
    if squash_tails_first:
        x = (x[0], ragged_ops.squash_tt_tails(x[1], use_jax=use_jax))

    # Orthogonalize Tucker matrices
    x = ragged_orth.up_orthogonalize_tucker_cores(x, use_jax=use_jax)

    # Right orthogonalize
    x = (x[0], orth.right_orthogonalize_tt_cores(x[1], use_jax=use_jax))

    G0 = x[1][0]
    _, ss_first, _ = linalg.right_svd(G0, use_jax=use_jax)

    # Sweep left to right computing SVDS
    all_ss_tucker = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        min_rank = min_tucker_ranks[ii] if min_tucker_ranks is not None else None
        max_rank = max_tucker_ranks[ii] if max_tucker_ranks is not None else None
        # SVD inbetween TT core and Tucker core
        x, ss_tucker = ragged_orth.up_svd_tt_core(
            x, ii, min_rank, max_rank, rtol, atol, use_jax=use_jax,
        )
        all_ss_tucker.append(ss_tucker)

        if ii < num_cores-1:
            min_rank = min_tt_ranks[ii+1] if min_tt_ranks is not None else None
            max_rank = max_tt_ranks[ii+1] if max_tt_ranks is not None else None
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = ragged_orth.left_svd_tt_core(
                x, ii, min_rank, max_rank, rtol, atol, use_jax=use_jax,
            )
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = linalg.left_svd(Gf, use_jax=use_jax)
        all_ss_tt.append(ss_tt)

    return x, tuple(all_ss_tucker), tuple(all_ss_tt)

