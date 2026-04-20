# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'up_orthogonalize_uniform_tucker_cores',
    'down_orthogonalize_uniform_tt_cores',
]


def up_orthogonalize_uniform_tucker_cores(
        tucker_supercore: NDArray,
        tt_supercore: NDArray,
        use_jax: bool = False,
) -> typ.Tuple[NDArray, NDArray]: # (up_tucker_supercore, new_tt_supercore)
    """Orthogonalize Tucker cores upwards, pushing remainders onto TT cores above.
    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    #

    B_d_i_o, G_d_a_i_b = tucker_supercore, tt_supercore
    B_d_o_i = B_d_i_o.swapaxes(-2,-1)

    U_d_o_x, ss_d_x, WT_d_x_i = xnp.linalg.svd(B_d_o_i, full_matrices=False)
    R_d_x_i = xnp.einsum('...x,...xi->...xi', ss_d_x, WT_d_x_i)

    new_G_d_a_x_b = xnp.einsum('...aib,...xi->...axb', G_d_a_i_b, R_d_x_i)
    new_U_d_x_o = U_d_o_x.swapaxes(-1, -2)
    up_tucker_cores, new_tt_cores = new_U_d_x_o, new_G_d_a_x_b

    return (up_tucker_cores, new_tt_cores)


def down_orthogonalize_uniform_tt_cores(
        tucker_supercore: NDArray,
        tt_supercore: NDArray,
        use_jax: bool = False,
) -> typ.Tuple[NDArray, NDArray]: # (tucker_variation_supercore, outer_tt_supercore)
    """Outer orthogonalize TT cores, pushing remainders downward onto tucker cores below.
    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    #

    U_d_i_o, H_d_a_i_b = tucker_supercore, tt_supercore

    d = H_d_a_i_b.shape[0]
    stack_shape = H_d_a_i_b.shape[1:-3]
    rL, n, rR = H_d_a_i_b.shape[-3:]
    H_d_ab_i = H_d_a_i_b.swapaxes(-1,-2).reshape((d,) + stack_shape + (rL*rR,n))

    O_d_ab_x, ss_d_x, WT_d_x_i = xnp.linalg.svd(H_d_ab_i, full_matrices=False)
    n2 = ss_d_x.shape[-1]
    O_d_a_x_b = O_d_ab_x.reshape((d,) + stack_shape + (rL, rR, n2)).swapaxes(-1,-2)

    C_d_x_i = xnp.einsum('d...x,d...xi->d...xi', ss_d_x, WT_d_x_i)

    V_d_x_o = np.einsum('d...xi,d...io->d...xo', C_d_x_i, U_d_i_o)
    tucker_variations, outer_tt_cores = V_d_x_o, O_d_a_x_b

    return tucker_variations, outer_tt_cores


