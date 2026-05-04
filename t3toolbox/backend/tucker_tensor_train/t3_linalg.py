# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.tucker_tensor_train.t3_operations import squash_tt_tails
import t3toolbox.backend.tucker_tensor_train.t3_orthogonalization as ragged_orth
import t3toolbox.backend.tucker_tensor_train.t3_operations as t3_ops
from t3toolbox.backend.common import *

__all__ = [
    't3_add',
    't3_scale',
    't3_inner_product_t3',
    't3_norm',
    't3_mult',
    't3_plus_scalar',
]


def t3_add(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
) -> typ.Tuple[typ.Tuple[NDArray], typ.Tuple[NDArray]]: # (x_plus_y_tucker_cores, x_plus_y_tt_cores)
    """Add Tucker tensor trains x and y, yielding a Tucker tensor train with summed ranks.
    """
    use_jax = (is_jax_ndarray(x) or is_jax_ndarray(y))
    xnp, xmap, _ = get_backend(False, use_jax)

    #
    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    assert(vsx == vsy)

    tucker_cores_z = [xnp.concatenate([Bx, By], axis=-2) for Bx, By in zip(tucker_cores_x, tucker_cores_y)]

    tt_cores_z = []

    for Gx, Gy in zip(tt_cores_x, tt_cores_y):
        G000 = Gx
        G001 = xnp.zeros(vsx + (Gx.shape[-3], Gx.shape[-2], Gy.shape[-1]))
        G010 = xnp.zeros(vsx + (Gx.shape[-3], Gy.shape[-2], Gx.shape[-1]))
        G011 = xnp.zeros(vsx + (Gx.shape[-3], Gy.shape[-2], Gy.shape[-1]))
        G100 = xnp.zeros(vsx + (Gy.shape[-3], Gx.shape[-2], Gx.shape[-1]))
        G101 = xnp.zeros(vsx + (Gy.shape[-3], Gx.shape[-2], Gy.shape[-1]))
        G110 = xnp.zeros(vsx + (Gy.shape[-3], Gy.shape[-2], Gx.shape[-1]))
        G111 = Gy
        Gz = xnp.block([[[G000, G001], [G010, G011]], [[G100, G101], [G110, G111]]])
        tt_cores_z.append(Gz)

    return tuple(tucker_cores_z), tuple(tt_cores_z)


def t3_scale(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        s,  # scalar
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]: # x*s
    """Multipy a Tucker tensor train by a scaling factor.
    """
    tucker_cores, tt_cores = x

    scaled_tucker_cores = [B.copy() for B in tucker_cores]
    scaled_tucker_cores[-1] = scaled_tucker_cores[-1] * s

    copied_tt_cores = [G.copy() for G in tt_cores]

    return tuple(scaled_tucker_cores), tuple(copied_tt_cores)


def t3_inner_product_t3(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        use_orthogonalization: bool = True, # for numerical stability
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt inner product of two Tucker tensor trains.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    x = (x[0], squash_tt_tails(x[1], use_jax=use_jax))
    y = (y[0], squash_tt_tails(y[1], use_jax=use_jax))

    if use_orthogonalization:
        x = ragged_orth.left_orthogonalize_t3(x, use_jax=use_jax)
        y = ragged_orth.left_orthogonalize_t3(y, use_jax=use_jax)

    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    assert(vsx == vsy)

    r0_x = tt_cores_x[0].shape[-3]
    r0_y = tt_cores_y[0].shape[-3]
    stack_shape = x[0][0].shape[:-2]

    M_sp = xnp.ones(stack_shape + (r0_x, r0_y))
    for Bx_ai, Gx_sat, By_bi, Gy_pbq in zip(tucker_cores_x, tt_cores_x, tucker_cores_y, tt_cores_y):
        tmp_ab = xnp.einsum('...ai,...bi->...ab', Bx_ai, By_bi)
        tmp_sbt = xnp.einsum('...sat,...ab->...sbt', Gx_sat, tmp_ab)
        tmp_pbt = xnp.einsum('...sp,...sbt->...pbt', M_sp, tmp_sbt)
        tmp_tq = xnp.einsum('...pbt,...pbq->...tq', tmp_pbt, Gy_pbq)
        M_sp = tmp_tq

    rd_x = tt_cores_x[-1].shape[-1]
    rd_y = tt_cores_y[-1].shape[-1]

    result = xnp.einsum('...tq,t,q', M_sp, np.ones(rd_x), np.ones(rd_y))
    return result


def t3_norm(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        use_orthogonalization: bool = True, # for numerical stability
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt norm of a Tucker tensor train.
    """
    xnp, _, _ = get_backend(False, use_jax)
    x = (x[0], squash_tt_tails(x[1], use_jax=use_jax))
    if use_orthogonalization:
        x = ragged_orth.left_orthogonalize_t3(x, use_jax=use_jax)
        Gf = x[1][-1].sum(axis=-1)
        norm_sq = (Gf*Gf).sum(axis=(-2,-1)) # Don't sum over stacked axes
    else:
        norm_sq = t3_inner_product_t3(x, x, use_jax=use_jax)

    return xnp.sqrt(xnp.abs(norm_sq))


def t3_mult(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
) -> typ.Tuple[typ.Tuple[NDArray], typ.Tuple[NDArray]]: # (x_times_y_tucker_cores, x_times_y_tt_cores)
    """Pointwise multiply Tucker tensor trains x and y, yielding a Tucker tensor train with multiplied ranks.

    This is the conventional "dumb" algorithm which does not do intermediate rank truncation.
    Ideally, we should also implement the newer "TTM" algorithm at some point.
    """
    use_jax = (is_jax_ndarray(x) or is_jax_ndarray(y))
    xnp, xmap, _ = get_backend(False, use_jax)

    #
    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    assert(vsx == vsy)

    tucker_cores_xy = []
    for Bx, By in zip(tucker_cores_x, tucker_cores_y):
        nx, Nx = Bx.shape[-2:]
        ny, Ny = By.shape[-2:]
        Bxy0 = xnp.einsum('...io,...jo->...ijo', Bx, By)
        Bxy = Bxy0.reshape(vsx + (nx*ny, Nx))
        tucker_cores_xy.append(Bxy)
    tucker_cores_xy = tuple(tucker_cores_xy)

    tt_cores_xy = []
    for Gx, Gy in zip(tt_cores_x, tt_cores_y):
        rLx, nx, rRx = Gx.shape[-3:]
        rLy, ny, rRy = Gy.shape[-3:]
        Gxy0 = xnp.einsum('...aib,...ujv->...auijbv', Gx, Gy)
        Gxy = Gxy0.reshape(vsx + (rLx*rLy, nx*ny, rRx*rRy))
        tt_cores_xy.append(Gxy)
    tt_cores_xy = tuple(tt_cores_xy)

    return tucker_cores_xy, tt_cores_xy


def t3_plus_scalar(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        s,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:
    use_jax = is_jax_ndarray(x)

    x_shape = tuple(B.shape[-1] for B in x[0])
    x_stack_shape = x[0][0].shape[:-2]

    y0 = t3_ops.t3_ones(x_shape, x_stack_shape, use_jax=use_jax)
    y = t3_scale(y0, s)
    xs = t3_add(x, y)
    return xs








