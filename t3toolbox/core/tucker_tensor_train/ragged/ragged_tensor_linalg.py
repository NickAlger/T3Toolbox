import numpy as np
import typing as typ

from t3toolbox.core.tucker_tensor_train.ragged.ragged_operations import squash_tt_tails
from t3toolbox.common import *

__all__ = [
    't3_add',
    't3_scale',
    't3_inner_product_t3',
]


def t3_add(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
        squash: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[typ.Tuple[NDArray], typ.Tuple[NDArray]]: # (x_plus_y_tucker_cores, x_plus_y_tt_cores)
    """Add Tucker tensor trains x and y, yielding a Tucker tensor train with summed ranks.
    """
    xnp, _, _ = get_backend(False, use_jax)

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

    if squash:
        z = (tuple(tucker_cores_z), squash_tt_tails(tt_cores_z))
    else:
        z = (tuple(tucker_cores_z), tuple(tt_cores_z))
    return z


def t3_scale(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        s,  # scalar
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]: # x*s
    """Multipy a Tucker tensor train by a scaling factor.
    """
    tucker_cores, tt_cores = x

    scaled_tucker_cores = [B.copy() for B in tucker_cores]
    scaled_tucker_cores[-1] = s * scaled_tucker_cores[-1]

    copied_tt_cores = [G.copy() for G in tt_cores]

    return tuple(scaled_tucker_cores), tuple(copied_tt_cores)


def t3_inner_product_t3(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt inner product of two Tucker tensor trains.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    tt_cores_x = squash_tt_tails(tt_cores_x)
    tt_cores_y = squash_tt_tails(tt_cores_y)

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    assert(vsx == vsy)

    r0_x = tt_cores_x[0].shape[-3]
    r0_y = tt_cores_y[0].shape[-3]

    M_sp = xnp.ones((r0_x, r0_y))
    for Bx_ai, Gx_sat, By_bi, Gy_pbq in zip(tucker_cores_x, tt_cores_x, tucker_cores_y, tt_cores_y):
        tmp_ab = xnp.einsum('...ai,...bi->...ab', Bx_ai, By_bi)
        tmp_sbt = xnp.einsum('...sat,...ab->...sbt', Gx_sat, tmp_ab)
        tmp_pbt = xnp.einsum('...sp,...sbt->...pbt', M_sp, tmp_sbt)
        tmp_tq = xnp.einsum('...pbt,...pbq->...tq', tmp_pbt, Gy_pbq)
        M_sp = tmp_tq

    rd_x = tt_cores_x[-1].shape[2]
    rd_y = tt_cores_y[-1].shape[2]

    result = xnp.einsum('...tq,t,q', M_sp, np.ones(rd_x), np.ones(rd_y))
    return result

