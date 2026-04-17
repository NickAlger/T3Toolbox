import numpy as np
import typing as typ

import t3toolbox.util_linalg as linalg
from t3toolbox.common import *

__all__ = [
    'reverse_utt',
    'absorb_edge_weights_into_ut3',
]


def reverse_utt(
        tt_cores: typ.Union[typ.Sequence[NDArray], NDArray]
) -> typ.Union[typ.Sequence[NDArray], NDArray]:
    """Reverse a uniform tensor train (no Tucker).
    """
    return tt_cores[::-1, :, :, :].swapaxes(1, 3)


def absorb_edge_weights_into_ut3(
        x0: typ.Tuple[NDArray, NDArray], # (tucker_supercore, tt_supercore)
        weights: typ.Tuple[
            NDArray,  # shape_weights, shape=(d,Ni)
            NDArray,  # tucker_weights, shape=(d,ni)
            NDArray,  # tt_weights, elm_shape=(d+1,ri)
        ],
        use_jax: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:
    """Contract each edge weight into a neighboring core.
    """
    is_uniform = not isinstance(x0[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    tucker_cores0, tt_cores0 = x0
    shape_weights, tucker_weights, tt_weights = weights

    tucker_cores = xnp.einsum('di,dio,do->dio', tucker_weights, tucker_cores0, shape_weights)
    first_tt_cores = xnp.einsum('di,diaj->diaj', tt_weights[:-2], tt_cores0[:-1])

    Gf = xnp.einsum('i,iaj,j->iaj', tt_weights[-2], tt_cores0[-1], tt_weights[-1])
    tt_cores = xnp.concatenate([first_tt_cores, Gf.reshape((1,) + Gf.shape)], axis=0)

    return tucker_cores, tt_cores