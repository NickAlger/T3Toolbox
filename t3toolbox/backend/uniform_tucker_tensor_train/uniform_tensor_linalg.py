# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tucker_tensor_train.orthogonalization as orth
import t3toolbox.backend.uniform_tucker_tensor_train.uniform_orthogonalization as uniform_orth
import t3toolbox.backend.uniform_tucker_tensor_train.uniform_t3_operations as ut3_ops
from t3toolbox.backend.common import *

__all__ = [
    'ut3_add',
    'scale_last_slice',
    'ut3_inner_product',
    'ut3_norm',
]




def ut3_add(
        x: typ.Tuple[
            NDArray, # tucker_supercore
            NDArray, # tt_supercore
            NDArray, # shape_mask
            NDArray, # tucker_edge_mask
            NDArray, # tt_edge_mask
        ],
        y: typ.Tuple[
            NDArray, # tucker_supercore
            NDArray, # tt_supercore
            NDArray, # shape_mask
            NDArray, # tucker_edge_mask
            NDArray, # tt_edge_mask
        ],
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # tucker_supercore
    NDArray, # tt_supercore
    NDArray, # shape_mask
    NDArray, # tucker_edge_mask
    NDArray, # tt_edge_mask
]: # z = x + y
    """Add two UniformTuckerTensorTrains, x,y -> x+y.
    """
    xnp, _, _ = get_backend(True, use_jax)

    #
    x_tucker_supercore, x_tt_supercore, x_shape_masks, x_tucker_masks, x_tt_masks = x
    y_tucker_supercore, y_tt_supercore, y_shape_masks, y_tucker_masks, y_tt_masks = y

    z_shape_masks  = xnp.logical_or(x_shape_masks, y_shape_masks)
    z_tucker_masks = xnp.concatenate([x_tucker_masks,   y_tucker_masks],   axis=-1)
    z_tt_masks     = xnp.concatenate([x_tt_masks,       y_tt_masks],       axis=-1)

    z_tucker_supercore = xnp.concatenate([x_tucker_supercore, y_tucker_supercore], axis=-2)

    d   = x_tt_supercore.shape[0]

    r_x = x_tt_supercore.shape[-1]
    n_x = x_tt_supercore.shape[-2]

    r_y = y_tt_supercore.shape[-1]
    n_y = y_tt_supercore.shape[-2]

    stack_shape = x_tucker_supercore.shape[1:-2]

    r0, n0 = r_x, n_x
    r1, n1 = r_y, n_y
    G000 = x_tt_supercore
    G001 = xnp.zeros((d,) + stack_shape + (r0, n0, r1))
    G010 = xnp.zeros((d,) + stack_shape + (r0, n1, r0))
    G011 = xnp.zeros((d,) + stack_shape + (r0, n1, r1))
    G100 = xnp.zeros((d,) + stack_shape + (r1, n0, r0))
    G101 = xnp.zeros((d,) + stack_shape + (r1, n0, r1))
    G110 = xnp.zeros((d,) + stack_shape + (r1, n1, r0))
    G111 = y_tt_supercore
    z_tt_supercore = xnp.block([[[G000, G001], [G010, G011]], [[G100, G101], [G110, G111]]])

    return z_tucker_supercore, z_tt_supercore, z_shape_masks, z_tucker_masks, z_tt_masks


def scale_last_slice(
        x: NDArray,
        s, # scalar
        use_jax: bool = False,
) -> NDArray: # x[:-1] -> x[:-1], x[-1] -> s*x[-1]
    """Apply scaling to only the last slice of an array.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.tucker_tensor_train.uniform.uniform_tensor_linalg as utla
    >>> x = np.random.randn(3,4,5)
    >>> s = 3.2
    >>> x_s = utla.scale_last_slice(x, s)
    >>> print(np.linalg.norm(x[:-1] - x_s[:-1]))
    0.0
    >>> print(np.linalg.norm(s*x[-1] - x_s[-1]))
    0.0
    """
    xnp, _, _ = get_backend(True, use_jax)
    return xnp.concatenate([x[:-1], s*x[-1:]], axis=0)



def ut3_inner_product(
        x: typ.Tuple[
            NDArray, # x_tucker_supercore
            NDArray, # x_tt_supercore
            NDArray, # x_shape_mask
            NDArray, # x_tucker_edge_mask
            NDArray, # x_tt_edge_mask
        ],
        y: typ.Tuple[
            NDArray, # x_tucker_supercore
            NDArray, # x_tt_supercore
            NDArray, # x_shape_mask
            NDArray, # x_tucker_edge_mask
            NDArray, # x_tt_edge_mask
        ],
        use_orthogonalization: bool = True,
        use_jax: bool = False,
):
    """Compute the Hilbert-Schmidt inner product between two uniform Tucker tensor trains.
    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    x_tucker_supercore, x_tt_supercore = ut3_ops.apply_masks_to_cores(x, use_jax=use_jax)
    y_tucker_supercore, y_tt_supercore = ut3_ops.apply_masks_to_cores(y, use_jax=use_jax)

    x_tt_supercore = ut3_ops.uniform_squash_tt_tails(x_tt_supercore, use_jax=use_jax)
    y_tt_supercore = ut3_ops.uniform_squash_tt_tails(y_tt_supercore, use_jax=use_jax)

    if use_orthogonalization:
        x_tucker_supercore, x_tt_supercore = uniform_orth.up_orthogonalize_uniform_tucker_cores(
            x_tucker_supercore, x_tt_supercore, use_jax=use_jax,
        )
        x_tt_supercore = orth.left_orthogonalize_tt_cores(x_tt_supercore, use_jax=use_jax)

        y_tucker_supercore, y_tt_supercore = uniform_orth.up_orthogonalize_uniform_tucker_cores(
            y_tucker_supercore, y_tt_supercore, use_jax=use_jax,
        )
        y_tt_supercore = orth.left_orthogonalize_tt_cores(y_tt_supercore, use_jax=use_jax)

    x_tt = xnp.einsum('d...io,d...aib->d...aob', x_tucker_supercore, x_tt_supercore)
    y_tt = xnp.einsum('d...io,d...aib->d...aob', y_tucker_supercore, y_tt_supercore)

    def _push(M, Gx_Gy):
        Gx, Gy = Gx_Gy
        M2 = xnp.einsum('...ab,...aoc,...bod->...cd', M, Gx, Gy)
        return M2, (0,)

    stack_shape = x_tucker_supercore.shape[1:-2]
    r_x = x_tt_supercore.shape[-1]
    r_y = y_tt_supercore.shape[-1]

    M = xnp.ones(stack_shape + (r_x,r_y))
    Mf, _ = xscan(_push, M, (x_tt, y_tt))
    return xnp.einsum('...ab->...', Mf)


def ut3_norm(
        x: typ.Tuple[
            NDArray, # x_tucker_supercore
            NDArray, # x_tt_supercore
            NDArray, # x_shape_mask
            NDArray, # x_tucker_edge_mask
            NDArray, # x_tt_edge_mask
        ],
        use_orthogonalization: bool = True,
        use_jax: bool = False,
):
    """Compute the Hilbert-Schmidt norm of a uniform Tucker tensor train.
    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    x_tucker_supercore, x_tt_supercore = ut3_ops.apply_masks_to_cores(x, use_jax=use_jax)

    x_tt_supercore = ut3_ops.uniform_squash_tt_tails(x_tt_supercore, use_jax=use_jax)

    if use_orthogonalization:
        x_tucker_supercore, x_tt_supercore = uniform_orth.up_orthogonalize_uniform_tucker_cores(
            x_tucker_supercore, x_tt_supercore, use_jax=use_jax,
        )
        x_tt_supercore = orth.left_orthogonalize_tt_cores(x_tt_supercore, use_jax=use_jax)

        Gf = x_tt_supercore[-1]
        return xnp.einsum('...aib,...aib', Gf, Gf)

    else:
        return ut3_inner_product(x, x, use_orthogonalization=False, use_jax=use_jax)






