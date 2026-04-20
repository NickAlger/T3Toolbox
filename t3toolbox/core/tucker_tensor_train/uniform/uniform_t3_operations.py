import numpy as np
import typing as typ

from IPython.utils.tokenutil import token_at_cursor

import t3toolbox.util_linalg as linalg
from t3toolbox.common import *

__all__ = [
    'reverse_utt',
    # 'absorb_edge_weights_into_ut3',
    'uniform_squash_tt_tails',
]


def reverse_utt(
        tt_cores: typ.Union[typ.Sequence[NDArray], NDArray]
) -> typ.Union[typ.Sequence[NDArray], NDArray]:
    """Reverse a uniform tensor train (no Tucker).
    """
    return tt_cores[::-1, :, :, :].swapaxes(1, 3)


# def absorb_edge_weights_into_ut3(
#         x0: typ.Tuple[NDArray, NDArray], # (tucker_supercore, tt_supercore)
#         weights: typ.Tuple[
#             NDArray,  # shape_weights, shape=(d,Ni)
#             NDArray,  # tucker_weights, shape=(d,ni)
#             NDArray,  # tt_weights, elm_shape=(d+1,ri)
#         ],
#         use_jax: bool = False,
# ) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:
#     """Contract each edge weight into a neighboring core.
#     """
#     is_uniform = not isinstance(x0[0], typ.Sequence)
#     xnp, xmap, xscan = get_backend(is_uniform, use_jax)
#
#     #
#     tucker_cores0, tt_cores0 = x0
#     shape_weights, tucker_weights, tt_weights = weights
#
#     tucker_cores = xnp.einsum('di,dio,do->dio', tucker_weights, tucker_cores0, shape_weights)
#     first_tt_cores = xnp.einsum('di,diaj->diaj', tt_weights[:-2], tt_cores0[:-1])
#
#     Gf = xnp.einsum('i,iaj,j->iaj', tt_weights[-2], tt_cores0[-1], tt_weights[-1])
#     tt_cores = xnp.concatenate([first_tt_cores, Gf.reshape((1,) + Gf.shape)], axis=0)
#
#     return tucker_cores, tt_cores


def uniform_squash_tt_tails(
        tt_supercore: NDArray,
        use_jax: bool = False,
) -> NDArray: # new_tt_supercore
    """Squash tails of uniform tensor train.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.core.tucker_tensor_train.uniform.operations as uniform_operations
    >>> tt_supercore = np.random.randn(4, 2,3, 5,6,5)
    >>> new_tt_supercore = uniform_operations.uniform_squash_tt_tails(tt_supercore)
    >>> print(np.linalg.norm(np.sum(tt_supercore[0], axis=-3) - new_tt_supercore[0, :,:, 0,:,:]))
    0.0
    >>> print(np.linalg.norm(new_tt_supercore[0, :,:, 1:,:,:]))
    0.0
    >>> print(np.linalg.norm(np.sum(tt_supercore[-1], axis=-1) - new_tt_supercore[-1, :,:, :,:,0]))
    0.0
    >>> print(np.linalg.norm(new_tt_supercore[-1, :,:, :,:,1:]))
    0.0
    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    d = tt_supercore.shape[0]
    stack_shape = tt_supercore.shape[1:-3]
    n = tt_supercore.shape[-2]
    r = tt_supercore.shape[-1]

    G0 = tt_supercore[:1] # shape=(1,)+stack_shape+(r,n,r)
    new_G0_flat = xnp.sum(G0, axis=-3, keepdims=True) # shape=(1,)+stack_shape+(1,n,r)
    Z0_flat = xnp.zeros((1,)+stack_shape+(r-1,n,r))
    new_G0 = xnp.concatenate([new_G0_flat, Z0_flat], axis=-3) # shape=(1,)+stack_shape+(r,n,r)

    GG_mid = tt_supercore[1:-1] # shape=(d-2,)+stack_shape+(r,n,r)

    Gf = tt_supercore[-1:] # shape=(1,)+stack_shape+(r,n,r)
    new_Gf_flat = xnp.sum(Gf, axis=-1, keepdims=True) # shape=(1,)+stack_shape+(r,n,1)
    Zf_flat = xnp.zeros((1,)+stack_shape+(r,n,r-1))
    new_Gf = xnp.concatenate([new_Gf_flat, Zf_flat], axis=-1)  # shape=(1,)+stack_shape+(r,n,r)

    new_tt_supercore = xnp.concatenate([new_G0, GG_mid, new_Gf], axis=0)
    return new_tt_supercore