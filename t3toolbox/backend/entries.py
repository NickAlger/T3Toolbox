# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.contractions as contractions
from t3toolbox.backend.common import *

__all__ = [
    't3_get_entries',
]


def t3_get_entries(
        x: typ.Union[
            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
            typ.Tuple[NDArray, NDArray], # (tucker_supercore, tt_supercore)
        ],
        index: NDArray, # dtype=int, shape=(d,)+vsi. (or convertible to int array of this shape)
        use_jax: bool = False,
) -> NDArray: # shape=vsx+vsi
    '''Compute entries of a Tucker tensor train.
    '''
    is_uniform = is_ndarray(x[0])
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    #
    tucker_cores, tt_cores = x
    vsx = x[0][0].shape[:-2]
    index = xnp.array(index)

    vsi = index.shape[1:]

    def _func(mu_XIa, ind_B_G):
        ind, B_Xpo, G_Xapb = ind_B_G
        xi_XpI = B_Xpo[..., ind]

        mu_XIb = contractions.MNa_Maib_MiN_to_MNb(
            mu_XIa, G_Xapb, xi_XpI, use_jax=use_jax,
        )

        return mu_XIb, (0,)

    mu_XIa = xnp.ones(vsx + vsi + (tt_cores[0].shape[-3],))
    ind_B_G = (index, tucker_cores, tt_cores)
    mu_XIz, _ = xscan(_func, mu_XIa, ind_B_G)

    result = xnp.sum(mu_XIz, axis=-1)
    return result

