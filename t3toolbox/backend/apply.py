# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
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

    vsx = tucker_cores[0].shape[:-2] # core stack G (T3s)
    vsv = vecs[0].shape[:-1]         # vec stack F (probes), base-inner: F outer, G inner

    def _func(mu_VXa, v_B_G):
        v_Vo, B_Xpo, G_Xapb = v_B_G
        mu_VXb = contractions.FGa_Gaib_Fo_Gio_to_FGb(
            mu_VXa, G_Xapb, v_Vo, B_Xpo,
        )
        return mu_VXb, (0,)

    mu_VXa = xnp.ones(vsv + vsx + (tt_cores[0].shape[-3],))   # F + G
    v_B_G = (vecs, tucker_cores, tt_cores)
    mu_VXz, _ = xscan(_func, mu_VXa, v_B_G)

    result = xnp.sum(mu_VXz, axis=-1)
    return result

