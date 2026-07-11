# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.t3_operations as ragged_operations
import t3toolbox.backend.ut3_operations as uniform_operations
from t3toolbox.backend.common import *

__all__ = [
    'tt_left_orthogonalize',
    'tt_right_orthogonalize',
]


def tt_left_orthogonalize(
        tt_cores: typ.Union[
            typ.Sequence[NDArray], # ragged. len=d, elm_shape=stack_shape+(ri,ni,r(i+1))
            NDArray, # uniform. shape=(d,)+stack_shape+(r,n,r)
        ],
        return_variation_cores: bool = False,
) -> typ.Union[
    typ.Tuple[NDArray,...], # left_tt_cores
    typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]], # left_tt_cores, var_tt_cores
]:
    """Left-orthogonalize a Tensor train (no Tucker).
    """
    is_uniform = is_ndarray(tt_cores)
    use_jax = any([is_jax_ndarray(G) for G in tt_cores])
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    def _left_func(H0, x):
        G1 = x[0]
        L0, H1, _ = linalg.left_svd_pair(H0, G1)
        return H1, (L0, H0)

    init = tt_cores[0]
    xs = (tt_cores[1:],)
    if len(xs[0]) > 0:  # >1 core to sweep; len() works for a ragged tuple and a uniform supercore alike
        Hf, (LL0, HH0) = xscan(_left_func, init, xs)
    else:
        Hf = init
        LL0 = ()
        HH0 = ()

    if is_uniform:
        left_tt_cores = xnp.concatenate([LL0, Hf.reshape((1,)+Hf.shape)])
        var_tt_cores  = xnp.concatenate([HH0, Hf.reshape((1,)+Hf.shape)])
    else:
        left_tt_cores = tuple(LL0) + (Hf,)
        var_tt_cores  = tuple(HH0) + (Hf,)

    if return_variation_cores:
        return left_tt_cores, var_tt_cores
    else:
        return left_tt_cores


def tt_right_orthogonalize(
        tt_cores: typ.Union[
            typ.Sequence[NDArray],  # ragged. len=d, elm_shape=stack_shape+(ri,ni,r(i+1))
            NDArray,  # uniform. shape=(d,)+stack_shape+(r,n,r)
        ],
        return_variation_cores: bool = False,
) -> typ.Union[
    typ.Tuple[NDArray,...], # right_tt_cores
    typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]], # right_tt_cores, var_tt_cores
]:
    reverse = tt_operations.tt_reverse   # polymorphic (ragged core tuple or uniform supercore)

    result = tt_left_orthogonalize(
        reverse(tt_cores), return_variation_cores=return_variation_cores,
    )
    if return_variation_cores:
        return reverse(result[0]), reverse(result[1])
    else:
        return reverse(result)


