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
            typ.Sequence[NDArray],  # len=d, elm_shape=vsw+(Ni,), ragged
            NDArray, # shape=(d,) + vsw +(Ni,), uniform (NOT IMPLEMENTED YET)
        ],
) -> NDArray:
    '''Contract a Tucker tensor train with vectors in all indices.
    '''
    use_jax = tree_contains_jax((x, vecs))
    xnp, _, xscan = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x

    #

    vsc = tucker_cores[0].shape[:-2] # core/base stack C (the batch of T3s)
    vsw = vecs[0].shape[:-1]         # vec stack W (the probe-like vectors), base-inner: W outer, C inner

    def _func(mu_WCa, v_B_G):
        v_Wo, B_Cpo, G_Capb = v_B_G
        mu_WCb = contractions.WCa_Caib_Wo_Cio_to_WCb(
            mu_WCa, G_Capb, v_Wo, B_Cpo,
        )
        return mu_WCb, (0,)

    mu_WCa = xnp.ones(vsw + vsc + (tt_cores[0].shape[-3],))   # W + C
    v_B_G = (vecs, tucker_cores, tt_cores)
    mu_WCz, _ = xscan(_func, mu_WCa, v_B_G)

    result = xnp.sum(mu_WCz, axis=-1)
    return result

