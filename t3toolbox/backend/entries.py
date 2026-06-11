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
        index: NDArray, # dtype=int, shape=(d,)+vsi. (or convertible to int array of this shape)
) -> NDArray: # shape=vsi+vsx (F + G, base-inner)
    '''Compute entries of a Tucker tensor train.
    '''
    use_jax = tree_contains_jax((x, index))
    is_uniform = is_ndarray(x[0])
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    #
    index = xnp.array(index)

    tucker_cores, tt_cores = x
    vsx = x[0][0].shape[:-2]
    index = xnp.array(index)

    vsi = index.shape[1:]    # index stack F (base-inner: F outer, G inner)
    n_idx = len(vsi)

    def _func(mu_IXa, ind_B_G):
        ind, B_Xpo, G_Xapb = ind_B_G
        xi_XpI = B_Xpo[..., ind]                                   # G + (p,) + I (index batch trails)
        xi_IXp = xnp.moveaxis(                                     # -> I + G + (p,) = FGi
            xi_XpI, tuple(range(-n_idx, 0)), tuple(range(n_idx)),
        )

        mu_IXb = contractions.FGa_Gaib_FGi_to_FGb(
            mu_IXa, G_Xapb, xi_IXp,
        )

        return mu_IXb, (0,)

    mu_IXa = xnp.ones(vsi + vsx + (tt_cores[0].shape[-3],))   # F + G
    ind_B_G = (index, tucker_cores, tt_cores)
    mu_IXz, _ = xscan(_func, mu_IXa, ind_B_G)

    result = xnp.sum(mu_IXz, axis=-1)
    return result

