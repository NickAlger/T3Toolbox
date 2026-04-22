# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import typing as typ


from t3toolbox.backend.common import *

__all__ = [
]


def ut3basis_to_t3basis(
        x: typ.Tuple[
            NDArray, # up_tucker_supercore
            NDArray, # down_tucker_supercore
            NDArray, # left_tt_supercore
            NDArray, # right_tucker_supercore
            NDArray, # shape_mask
            NDArray, # up_mask
            NDArray, # down_mask
            NDArray, # basis_left_mask
            NDArray, # basis_right_mask
        ],
        use_jax: bool = False,
) -> typ.Union[
    typ.Tuple, #
]:
    '''Convert UniformT3Basis to T3Basis.

    If uniform T3Basis is stacked, either:
        - return an array-line nesting of tuples containing the T3Basis's (stack_output=False),
        - or one stacked T3Basis (stack_output=True)

    Can only return a stacked T3Basis if the stacked UT3Basis's all have the same structure.
    '''
    xnp, _, _ = get_backend(True, use_jax)

    #
    (up_supercore, down_supercore, left_supercore, right_supercore,
     shape_mask, up_mask, down_mask, basis_left_mask, basis_right_mask) = x
    stack_shape = up_supercore[0].shape[:-2]

    if not stack_shape: # not stacked
        shape_inds  = [xnp.argwhere(em).reshape(-1) for em in list(shape_mask)]
        up_inds = [xnp.argwhere(em).reshape(-1) for em in list(up_mask)]
        down_inds = [xnp.argwhere(em).reshape(-1) for em in list(down_mask)]
        left_inds = [xnp.argwhere(em).reshape(-1) for em in list(basis_left_mask)]
        right_inds = [xnp.argwhere(em).reshape(-1) for em in list(basis_right_mask)]

        up_cores = tuple([
            U[ii,:][:,jj]
            for ii, jj, U
            in zip(up_inds, shape_inds, list(up_supercore))
        ])
        down_cores = tuple([
            D[ii, :][:, aa, :][:, jj]
            for ii, aa, jj, D
            in zip(left_inds[:-1], down_inds, right_inds[1:], list(down_supercore))
        ])
        left_cores = tuple([
            L[ii, :, :][:,aa,:][:, :, jj]
            for ii, aa, jj, L
            in zip(left_inds[:-1], up_inds, left_inds[1:], list(left_supercore))
        ])
        right_cores = tuple([
            R[ii, :, :][:, aa, :][:, :, jj]
            for ii, aa, jj, R
            in zip(right_inds[:-1], up_inds, right_inds[1:], list(right_supercore))
        ])
        return up_cores, down_cores, left_cores, right_cores

    all_T3Bs = []
    for ii in range(up_supercore.shape[1]):
        xi = (
            up_supercore[:, ii],
            down_supercore[:, ii],
            left_supercore[:, ii],
            right_supercore[:, ii],
            shape_mask,
            up_mask[:, ii],
            down_mask[:, ii],
            basis_left_mask[:, ii],
            basis_right_mask[:, ii],
        )
        ith_t3b = ut3basis_to_t3basis(xi, use_jax=use_jax)
        all_T3Bs.append(ith_t3b)

    all_T3Bs = tuple(all_T3Bs)
    return all_T3Bs


