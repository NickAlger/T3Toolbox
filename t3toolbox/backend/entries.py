# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.apply as apply
from t3toolbox.backend.common import *

__all__ = [
    'tucker_tensor_train_entries',
    'tucker_tensor_train_entries_transpose',
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


def tucker_tensor_train_entries_transpose(
        c:                  NDArray,                # residual, shape=W+C
        index:              NDArray,                # int, shape=(d,)+W
        shape:              typ.Sequence[int],      # ambient dims (N0,...,N(d-1)) -- to size the one-hots
        sum_over_probes:    bool = False,           # True: scatter-add colliding indices (Gauss-Newton J^T r)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores
    typ.Tuple[NDArray, ...],  # tt_cores
]:
    '''Transpose of :py:func:`tucker_tensor_train_entries`: scatter a residual ``c`` at ``index``.

    Identical to :py:func:`tucker_tensor_train_apply_transpose` with the apply vectors replaced by the
    unit vectors ``e_{index_k}`` -- so each single-entry adjoint is the one-hot rank-1 tensor
    ``c * e_{idx_0} (x) ... (x) e_{idx_{d-1}}``. ``sum_over_probes=True`` scatter-adds colliding
    indices (the ``J^T r`` for entry sampling). ``shape`` supplies the ambient dims, which (unlike the
    apply case, where ``ww`` carries them) the residual and index alone do not determine.
    '''
    use_jax = tree_contains_jax((c, index))
    xnp, _, _ = get_backend(False, use_jax)
    index = xnp.array(index)
    ww = tuple(xnp.eye(N)[index[i]] for i, N in enumerate(shape))   # one-hot, elm_shape = W + (Ni,)
    return apply.tucker_tensor_train_apply_transpose(c, ww, sum_over_probes=sum_over_probes)

