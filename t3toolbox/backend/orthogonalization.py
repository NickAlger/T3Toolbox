# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.tucker_tensor_train.t3_operations as ragged_operations
import t3toolbox.backend.uniform_tucker_tensor_train.ut3_operations as uniform_operations
from t3toolbox.backend.common import *

__all__ = [
    'left_orthogonalize_tt_cores',
    'right_orthogonalize_tt_cores',
]


def left_orthogonalize_tt_cores(
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
    if xs[0]:
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

    # stack_shape = tt_cores[0].shape[:-3]
    #
    # def _left_func(Cxb, left_func_args):
    #     Gbjc = left_func_args[0]
    #
    #     Hxjc = xnp.einsum('...xb,...bjc->...xjc', Cxb, Gbjc)
    #
    #     rL, n, rR = Hxjc.shape[-3:]
    #     H_xj_c = Hxjc.reshape(stack_shape + (rL * n, rR))
    #     L_xj_y, ssy, VTyc = xnp.linalg.svd(H_xj_c, full_matrices=False)
    #     rR2 = ssy.shape[-1]
    #     Lxjy = L_xj_y.reshape(stack_shape + (rL, n, rR2))
    #
    #     Cyc = ssy.reshape(stack_shape + (-1, 1)) * VTyc
    #
    #     return Cyc, (Lxjy, Hxjc)
    #
    # rL0 = tt_cores[0].shape[-3]
    #
    # init = xnp.broadcast_to(xnp.eye(rL0), stack_shape+(rL0,rL0))
    # xs = (tt_cores[:-1],)
    #
    # Cf, (LL, HH) = xscan(_left_func, init, xs)
    #
    # # Dealing with the last core as a special case
    # Lf = xnp.einsum('...xb,...bjc->...xjc', Cf, tt_cores[-1])
    # if is_uniform:
    #     left_tt_cores = xnp.concatenate([LL, Lf.reshape((1,)+Lf.shape)])
    #     var_tt_cores  = xnp.concatenate([HH, Lf.reshape((1,) + Lf.shape)])
    # else:
    #     left_tt_cores = tuple(LL) + (Lf,)
    #     var_tt_cores = tuple(HH) + (Lf,)
    #
    # if return_variation_cores:
    #     return left_tt_cores, var_tt_cores
    # else:
    #     return left_tt_cores


def right_orthogonalize_tt_cores(
        tt_cores: typ.Union[
            typ.Sequence[NDArray],  # ragged. len=d, elm_shape=stack_shape+(ri,ni,r(i+1))
            NDArray,  # uniform. shape=(d,)+stack_shape+(r,n,r)
        ],
        return_variation_cores: bool = False,
) -> typ.Union[
    typ.Tuple[NDArray,...], # right_tt_cores
    typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]], # right_tt_cores, var_tt_cores
]:
    if is_ndarray(tt_cores):
        reverse = uniform_operations.reverse_utt
    else:
        reverse = ragged_operations.reverse_tt

    result = left_orthogonalize_tt_cores(
        reverse(tt_cores), return_variation_cores=return_variation_cores,
    )
    if return_variation_cores:
        return reverse(result[0]), reverse(result[1])
    else:
        return reverse(result)


