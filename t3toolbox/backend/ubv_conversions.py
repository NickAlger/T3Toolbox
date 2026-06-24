# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'ut3basis_to_t3basis',
]


def ut3basis_to_t3basis(
        x: typ.Tuple[
            NDArray,                          # up_tucker_supercore
            NDArray,                          # down_tt_supercore
            NDArray,                          # left_tt_supercore
            NDArray,                          # right_tt_supercore
            typ.Tuple[int, ...],              # shape
            typ.Tuple[                        # (up_mask, down_mask, basis_left_mask, basis_right_mask)
                NDArray, NDArray, NDArray, NDArray,
            ],
        ],
) -> typ.Union[
    typ.Tuple[typ.Tuple[NDArray, ...], ...],  # (up_cores, down_cores, left_cores, right_cores), if unstacked
    typ.Tuple,                                # else a nested tree (shaped like stack_shape) of those
]:
    '''Convert a uniform UT3Basis ``.data`` to ragged ``T3Basis`` core-tuples (or a nested tree, if stacked).

    The physical mode dims are a contiguous prefix, so they slice ``[:Ni]`` (from the ``shape`` ints, no
    argwhere); only the *rank* masks scatter, so they are extracted with ``np.argwhere`` (HOST numpy --
    masks are host). The supercores may be jax; advanced-indexing them with the host int indices is fine.
    '''
    (up_supercore, down_supercore, left_supercore, right_supercore,
     shape, (up_mask, down_mask, basis_left_mask, basis_right_mask)) = x
    stack_shape = up_supercore.shape[1:-2]
    d = up_supercore.shape[0]

    if not stack_shape:  # unstacked -> one ragged (up, down, left, right) core set
        up_cores, down_cores, left_cores, right_cores = [], [], [], []
        for ind in range(d):
            up_inds   = np.argwhere(up_mask[ind]).reshape(-1)
            down_inds = np.argwhere(down_mask[ind]).reshape(-1)
            left_a    = np.argwhere(basis_left_mask[ind]).reshape(-1)
            left_b    = np.argwhere(basis_left_mask[ind + 1]).reshape(-1)
            right_a   = np.argwhere(basis_right_mask[ind]).reshape(-1)
            right_b   = np.argwhere(basis_right_mask[ind + 1]).reshape(-1)
            Ni = shape[ind]

            up_cores.append(   up_supercore[ind][up_inds, :][:, :Ni])
            down_cores.append( down_supercore[ind][left_a, :, :][:, down_inds, :][:, :, right_b])
            left_cores.append( left_supercore[ind][left_a, :, :][:, up_inds, :][:, :, left_b])
            right_cores.append(right_supercore[ind][right_a, :, :][:, up_inds, :][:, :, right_b])

        return tuple(up_cores), tuple(down_cores), tuple(left_cores), tuple(right_cores)

    all_T3Bs = []
    for ii in range(up_supercore.shape[1]):
        xi = (
            up_supercore[:, ii], down_supercore[:, ii], left_supercore[:, ii], right_supercore[:, ii],
            shape,
            (up_mask[:, ii], down_mask[:, ii], basis_left_mask[:, ii], basis_right_mask[:, ii]),
        )
        all_T3Bs.append(ut3basis_to_t3basis(xi))
    return tuple(all_T3Bs)
