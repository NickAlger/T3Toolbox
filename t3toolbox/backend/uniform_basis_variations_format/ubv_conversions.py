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
    '''Convert UniformT3Basis to array-like tree of T3Basis.
    '''
    xnp, _, _ = get_backend(True, use_jax)

    #
    (up_supercore, down_supercore, left_supercore, right_supercore,
     shape_mask, up_mask, down_mask, basis_left_mask, basis_right_mask) = x
    stack_shape = up_supercore.shape[1:-2]
    d = up_supercore.shape[0]

    if not stack_shape: # not stacked
        up_cores = []
        down_cores = []
        left_cores = []
        right_cores = []
        for ind in range(d):
            shape_inds  = xnp.argwhere(shape_mask[ind])          .reshape(-1)
            up_inds     = xnp.argwhere(up_mask[ind])             .reshape(-1)
            down_inds   = xnp.argwhere(down_mask[ind])           .reshape(-1)
            left_inds_a   = xnp.argwhere(basis_left_mask[ind])   .reshape(-1)
            left_inds_b   = xnp.argwhere(basis_left_mask[ind+1]) .reshape(-1)
            right_inds_a  = xnp.argwhere(basis_right_mask[ind])  .reshape(-1)
            right_inds_b  = xnp.argwhere(basis_right_mask[ind+1]).reshape(-1)

            # print('shape_inds=', shape_inds)
            # print('up_inds=', up_inds)
            # print('down_inds=', down_inds)
            # print('left_inds=', left_inds)
            # print('right_inds=', right_inds)
            # print('up_supercore.shape=', up_supercore.shape)

            uc = up_supercore[ind][up_inds,:][:, shape_inds]
            up_cores.append(uc)

            dc = down_supercore[ind][left_inds_a,:,:][:,down_inds,:][:,:,right_inds_b]
            down_cores.append(dc)

            lc = left_supercore[ind][left_inds_a,:,:][:,up_inds,:][:,:,left_inds_b]
            left_cores.append(lc)

            rc = right_supercore[ind][right_inds_a,:,:][:,up_inds,:][:,:,right_inds_b]
            right_cores.append(rc)

        return tuple(up_cores), tuple(down_cores), tuple(left_cores), tuple(right_cores)

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


