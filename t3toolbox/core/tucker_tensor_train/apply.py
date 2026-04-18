import numpy as np
import typing as typ

import t3toolbox.utils.contractions as contractions
from t3toolbox.common import *

__all__ = [
    't3_apply',
]

def t3_apply(
        x: typ.Union[
            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
            typ.Tuple[NDArray, NDArray], # (tucker_supercore, tt_supercore)
        ],
        vecs: typ.Union[
            typ.Sequence[NDArray],  # len=d, elm_shape=vsv+(Ni,), ragged
            NDArray, # shape=(d,) + vsv +(Ni,), uniform (NOT IMPLEMENTED YET)
        ],
        use_jax: bool = False,
) -> NDArray:
    '''Contract a Tucker tensor train with vectors in all indices.
    '''
    xnp, _, xscan = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x

    #

    vsx = tucker_cores[0].shape[:-2] # vectorization shape for T3s
    vsv = vecs[0].shape[:-1] # vectorization shape for vecs

    def _func(mu_XVa, v_B_G):
        v_Vo, B_Xpo, G_Xapb = v_B_G
        mu_XVb = contractions.MNa_Maib_No_Mio_to_MNb(
            mu_XVa, G_Xapb, v_Vo, B_Xpo, use_jax=use_jax,
        )
        return mu_XVb, (0,)

    mu_XVa = xnp.ones(vsx + vsv + (tt_cores[0].shape[-3],))
    v_B_G = (vecs, tucker_cores, tt_cores)
    mu_XVz, _ = xscan(_func, mu_XVa, v_B_G)

    result = xnp.sum(mu_XVz, axis=-1)
    return result

    #

    # vecs_dims = [len(v.shape) for v in vecs]
    #
    # vectorized = True
    # if vecs_dims[0] == 1:
    #     vectorized = False
    #     vecs = [v.reshape((1,-1)) for v in vecs]
    #
    # num_applies = vecs[0].shape[0]
    #
    # mu_na = xnp.ones((num_applies, tt_cores[0].shape[0]))
    # for V_ni, B_xi, G_axb in zip(vecs, tucker_cores, tt_cores):
    #     v_nx = xnp.einsum('ni,xi->nx', V_ni, B_xi)
    #     g_anb = xnp.einsum('axb,nx->anb', G_axb, v_nx)
    #     mu_nb = xnp.einsum('na,anb->nb', mu_na, g_anb)
    #     mu_na = mu_nb
    # result = xnp.einsum('na->n', mu_na)
    #
    # if not vectorized:
    #     result = result[0]
    #
    # return result
