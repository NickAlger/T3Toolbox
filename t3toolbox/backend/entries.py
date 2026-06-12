# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.contractions as contractions
from t3toolbox.backend.common import *

__all__ = [
    'tucker_tensor_train_entries',
]


def tucker_tensor_train_entries(
        x: typ.Union[
            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
            typ.Tuple[NDArray, NDArray], # (tucker_supercore, tt_supercore)
        ],
        index: NDArray, # dtype=int, shape=(d,)+vsw. (or convertible to int array of this shape)
) -> NDArray: # shape=vsw+vsc (W + C, base-inner)
    '''Compute entries of a Tucker tensor train.
    '''
    use_jax = tree_contains_jax((x, index))
    is_uniform = is_ndarray(x[0])
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    #
    index = xnp.array(index)

    tucker_cores, tt_cores = x
    vsc = x[0][0].shape[:-2]
    index = xnp.array(index)

    vsw = index.shape[1:]    # index stack W (base-inner: W outer, C inner)
    n_idx = len(vsw)

    def _func(mu_WCa, ind_B_G):
        ind, B_Cpo, G_Capb = ind_B_G
        xi_CpW = B_Cpo[..., ind]                                   # C + (p,) + W (index batch trails)
        xi_WCp = xnp.moveaxis(                                     # -> W + C + (p,) = WCi
            xi_CpW, tuple(range(-n_idx, 0)), tuple(range(n_idx)),
        )

        mu_WCb = contractions.WCa_Caib_WCi_to_WCb(
            mu_WCa, G_Capb, xi_WCp,
        )

        return mu_WCb, (0,)

    mu_WCa = xnp.ones(vsw + vsc + (tt_cores[0].shape[-3],))   # W + C
    ind_B_G = (index, tucker_cores, tt_cores)
    mu_WCz, _ = xscan(_func, mu_WCa, ind_B_G)

    result = xnp.sum(mu_WCz, axis=-1)
    return result

