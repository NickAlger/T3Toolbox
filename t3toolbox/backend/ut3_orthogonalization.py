# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'down_orthogonalize_tucker_cores',
    'up_orthogonalize_tt_cores',
]


def down_orthogonalize_tucker_cores(
        tucker_supercore: NDArray,  # shape=(d,)+stack_shape+(n,N)
        tt_supercore:     NDArray,  # shape=(d,)+stack_shape+(r,n,r)
) -> typ.Tuple[
    NDArray,  # up_tucker_supercore, shape=(d,)+stack_shape+(x,N), x=min(N,n); rows orthonormal over N
    NDArray,  # new_tt_supercore,    shape=(d,)+stack_shape+(r,x,r)
]:
    """Orthogonalize the Tucker cores (rows orthonormal over the mode index), pushing the remainder up
    into the TT cores. Core-local, so done as one batched SVD over the leading ``(d,)+stack`` axes
    (the uniform vectorization win vs. the ragged map-over-modes). Expects masked input.
    """
    use_jax = tree_contains_jax((tucker_supercore, tt_supercore))
    xnp, _, _ = get_backend(True, use_jax)

    B_d_i_o = tucker_supercore
    G_d_a_i_b = tt_supercore
    B_d_o_i = B_d_i_o.swapaxes(-2, -1)

    U_d_o_x, ss_d_x, WT_d_x_i = xnp.linalg.svd(B_d_o_i, full_matrices=False)
    R_d_x_i = xnp.einsum('...x,...xi->...xi', ss_d_x, WT_d_x_i)

    new_G_d_a_x_b = xnp.einsum('...aib,...xi->...axb', G_d_a_i_b, R_d_x_i)
    new_U_d_x_o = U_d_o_x.swapaxes(-1, -2)
    return new_U_d_x_o, new_G_d_a_x_b


def up_orthogonalize_tt_cores(
        tucker_supercore: NDArray,  # shape=(d,)+stack_shape+(n,N)
        tt_supercore:     NDArray,  # shape=(d,)+stack_shape+(r,n,r)
) -> typ.Tuple[
    NDArray,  # new_tucker_supercore, shape=(d,)+stack_shape+(x,N), x=min(r*r,n)
    NDArray,  # up_tt_supercore,      shape=(d,)+stack_shape+(r,x,r); n-index orthonormal over (r,r)
]:
    """Up-orthogonalize the TT cores (mode index orthonormal over the two bond indices), pushing the
    remainder down into the Tucker cores. Core-local -> one batched SVD over ``(d,)+stack``. Expects
    masked input.
    """
    use_jax = tree_contains_jax((tucker_supercore, tt_supercore))
    xnp, _, _ = get_backend(True, use_jax)

    U_d_i_o = tucker_supercore
    H_d_a_i_b = tt_supercore

    d = H_d_a_i_b.shape[0]
    stack_shape = H_d_a_i_b.shape[1:-3]
    rL, n, rR = H_d_a_i_b.shape[-3:]
    H_d_ab_i = H_d_a_i_b.swapaxes(-1, -2).reshape((d,) + stack_shape + (rL * rR, n))

    O_d_ab_x, ss_d_x, WT_d_x_i = xnp.linalg.svd(H_d_ab_i, full_matrices=False)
    x = ss_d_x.shape[-1]
    O_d_a_x_b = O_d_ab_x.reshape((d,) + stack_shape + (rL, rR, x)).swapaxes(-1, -2)

    C_d_x_i = xnp.einsum('...x,...xi->...xi', ss_d_x, WT_d_x_i)
    V_d_x_o = xnp.einsum('...xi,...io->...xo', C_d_x_i, U_d_i_o)
    return V_d_x_o, O_d_a_x_b
