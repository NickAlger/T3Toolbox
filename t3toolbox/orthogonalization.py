# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.tucker_tensor_train as t3
from t3toolbox.common import *

__all__ = [
    'left_orthogonalize_tt_cores',
    'right_orthogonalize_tt_cores',
]


def left_orthogonalize_tt_cores(
        tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(ri,ni,r(i+1))
        return_variation_cores: bool = False,
        use_jax: bool = False,
):
    """Left-orthogonalize a Tensor train (no Tucker).
    """
    is_uniform = not isinstance(tt_cores, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #

    def _left_func(Cxb, left_func_args):
        Gbjc = left_func_args[0]

        Hxjc = xnp.einsum('xb,bjc->xjc', Cxb, Gbjc)

        rL, n, rR = Hxjc.shape
        H_xj_c = Hxjc.reshape((rL*n, rR))
        L_xj_y, ssy, VTyc = xnp.linalg.svd(H_xj_c, full_matrices=False)
        rR2 = len(ssy)
        Lxjy = L_xj_y.reshape((rL,n,rR2))

        Cyc = ssy.reshape((-1, 1)) * VTyc

        return Cyc, (Lxjy, Hxjc)


    init = xnp.eye(tt_cores[0].shape[0])
    xs = (tt_cores[:-1],)

    Cf, (LL, HH) = xscan(_left_func, init, xs)

    # Dealing with the last backend as a special case
    Lf = xnp.einsum('xb,bjc->xjc', Cf, tt_cores[-1])
    if is_uniform:
        left_tt_cores = xnp.concatenate([LL, Lf.reshape((1,) + Lf.shape)], axis=0)
        var_tt_cores = xnp.concatenate([HH, Lf.reshape((1,) + Lf.shape)], axis=0)
    else:
        left_tt_cores = tuple(LL) + (Lf,)
        var_tt_cores = tuple(HH) + (Lf,)

    if return_variation_cores:
        return left_tt_cores, var_tt_cores
    else:
        return left_tt_cores


def right_orthogonalize_tt_cores(
        tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(ri,ni,r(i+1))
        return_variation_cores: bool = False,
        use_jax: bool = False,
):
    result = left_orthogonalize_tt_cores(
        t3.reverse_tt(tt_cores), return_variation_cores=return_variation_cores, use_jax=use_jax,
    )
    if return_variation_cores:
        return t3.reverse_tt(result[0]), t3.reverse_tt(result[1])
    else:
        return t3.reverse_tt(result)



