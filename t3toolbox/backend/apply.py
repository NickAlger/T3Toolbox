# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.contractions as contractions
from t3toolbox.backend.common import *

__all__ = [
    'tucker_tensor_train_apply',
]

def tucker_tensor_train_apply(
        x: typ.Union[
            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
            typ.Tuple[NDArray, NDArray], # (tucker_supercore, tt_supercore)
        ],
        vecs: typ.Union[
            typ.Sequence[NDArray],  # len=d, elm_shape=vsv+(Ni,), ragged
            NDArray, # shape=(d,) + vsv +(Ni,), uniform (NOT IMPLEMENTED YET)
        ],
) -> NDArray:
    '''Contract a Tucker tensor train with vectors in all indices.
    '''
    use_jax = tree_contains_jax((x, vecs))
    xnp, _, xscan = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x

    #

    vsx = tucker_cores[0].shape[:-2] # vectorization shape for T3s
    vsv = vecs[0].shape[:-1] # vectorization shape for vecs

    def _func(mu_XVa, v_B_G):
        v_Vo, B_Xpo, G_Xapb = v_B_G
        mu_XVb = contractions.GFa_Gaib_Fo_Gio_to_GFb(
            mu_XVa, G_Xapb, v_Vo, B_Xpo, use_jax=use_jax,
        )
        return mu_XVb, (0,)

    mu_XVa = xnp.ones(vsx + vsv + (tt_cores[0].shape[-3],))
    v_B_G = (vecs, tucker_cores, tt_cores)
    mu_XVz, _ = xscan(_func, mu_XVa, v_B_G)

    result = xnp.sum(mu_XVz, axis=-1)
    return result

