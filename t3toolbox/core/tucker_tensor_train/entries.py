import numpy as np
import typing as typ

import t3toolbox.util_linalg as linalg
from t3toolbox.common import *

__all__ = [
    't3_get_entries',
]


def t3_get_entries(
        x: typ.Union[
            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
            typ.Tuple[NDArray, NDArray], # (tucker_supercore, tt_supercore)
        ],
        index: NDArray, # dtype=int, shape=(d,)+vsi
        use_jax: bool = False,
) -> NDArray:
    '''Compute entries of a Tucker tensor train.
    '''
    is_uniform = is_ndarray(x[0])
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    #
    tucker_cores, tt_cores = x
    num_cores = len(x[0])
    vsx = x[0][0].shape[:-2]

    vsi = index.shape[1:]

    NX = np.prod(vsx, dtype=int) # yes, np. We want this computed statically
    NI = np.prod(vsi, dtype=int)

    def _func(mu_IXa, ind_B_G):
        ind, B_Xpo, G_Xapb = ind_B_G
        N = B_Xpo.shape[-1]
        rL = G_Xapb.shape[-3]
        n = G_Xapb.shape[-2]
        rR = G_Xapb.shape[-1]

        B_Xpo = B_Xpo.reshape((NX, n, N))
        G_Xapb = G_Xapb.reshape((NX, rL, n, rR))

        v_XpI = B_Xpo[:,:,ind]
        mu_IXb = xnp.einsum('...Xa,Xapb,Xp...->...Xb', mu_IXa, G_Xapb, v_XpI)
        return mu_IXb, (0,)

    mu_IXa = xnp.ones((NI, NX)+(tt_cores[0].shape[-3],))
    ind_B_G = (index, tucker_cores, tt_cores)
    mu_IXz, _ = xscan(_func, mu_IXa, ind_B_G)

    result = xnp.einsum('...Xa->...X', mu_IXz).reshape(vsi + vsx)
    return result

