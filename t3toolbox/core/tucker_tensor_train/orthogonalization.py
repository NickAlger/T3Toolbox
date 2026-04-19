import numpy as np
import typing as typ

import t3toolbox.core.tucker_tensor_train.ragged.ragged_t3_operations as ragged_operations
import t3toolbox.core.tucker_tensor_train.uniform.uniform_t3_operations as uniform_operations
from t3toolbox.common import *

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
        use_jax: bool = False,
) -> typ.Union[
    typ.Tuple[NDArray,...], # left_tt_cores
    typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]], # left_tt_cores, var_tt_cores
]:
    """Left-orthogonalize a Tensor train (no Tucker).
    """
    is_uniform = False
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    stack_shape = tt_cores[0].shape[:-3]

    def _left_func(Cxb, left_func_args):
        Gbjc = left_func_args[0]

        Hxjc = xnp.einsum('...xb,...bjc->...xjc', Cxb, Gbjc)

        rL, n, rR = Hxjc.shape
        H_xj_c = Hxjc.reshape((rL * n, rR))
        L_xj_y, ssy, VTyc = xnp.linalg.svd(H_xj_c, full_matrices=False)
        rR2 = ssy.shape[-1]
        Lxjy = L_xj_y.reshape((rL, n, rR2))

        Cyc = ssy.reshape((-1, 1)) * VTyc

        return Cyc, (Lxjy, Hxjc)

    init0 = xnp.eye(tt_cores[0].shape[0])
    init = xnp.tile(init0, stack_shape+(1,)*len(init0.shape))
    xs = (tt_cores[:-1],)

    Cf, (LL, HH) = xscan(_left_func, init, xs)

    # Dealing with the last core as a special case
    Lf = xnp.einsum('...xb,...bjc->...xjc', Cf, tt_cores[-1])
    left_tt_cores = tuple(LL) + (Lf,)
    var_tt_cores = tuple(HH) + (Lf,)

    if return_variation_cores:
        return left_tt_cores, var_tt_cores
    else:
        return left_tt_cores


def right_orthogonalize_tt_cores(
        tt_cores: typ.Union[
            typ.Sequence[NDArray],  # ragged. len=d, elm_shape=stack_shape+(ri,ni,r(i+1))
            NDArray,  # uniform. shape=(d,)+stack_shape+(r,n,r)
        ],
        return_variation_cores: bool = False,
        use_jax: bool = False,
) -> typ.Union[
    typ.Tuple[NDArray,...], # right_tt_cores
    typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]], # right_tt_cores, var_tt_cores
]:
    if is_ndarray(tt_cores):
        reverse = ragged_operations.reverse_tt
    else:
        reverse = uniform_operations.reverse_utt

    result = left_orthogonalize_tt_cores(
        reverse(tt_cores), return_variation_cores=return_variation_cores, use_jax=use_jax,
    )
    if return_variation_cores:
        return reverse(result[0]), reverse(result[1])
    else:
        return reverse(result)