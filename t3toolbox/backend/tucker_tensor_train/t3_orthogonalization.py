# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.orthogonalization as orth
from t3toolbox.backend.common import *

__all__ = [
    'left_orthogonalize_t3',
    'right_orthogonalize_t3',
    'up_orthogonalize_tt_cores',
    'down_orthogonalize_tucker_cores',
    'down_svd_tucker_core',
    'left_svd_tt_core',
    'right_svd_tt_core',
]


def left_orthogonalize_t3(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        use_jax: bool = False,
) -> typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]]: # (tucker_variations, outer_tt_cores)
    """Left orthogonalize T3.
    """
    up_tucker_cores, tt_cores = down_orthogonalize_tucker_cores(x)
    left_tt_cores = orth.left_orthogonalize_tt_cores(tt_cores)
    return (up_tucker_cores, left_tt_cores)


def right_orthogonalize_t3(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        use_jax: bool = False,
) -> typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]]: # (tucker_variations, outer_tt_cores)
    """Left orthogonalize T3.
    """
    up_tucker_cores, tt_cores = down_orthogonalize_tucker_cores(x)
    right_tt_cores = orth.right_orthogonalize_tt_cores(tt_cores)
    return (up_tucker_cores, right_tt_cores)


def up_orthogonalize_tt_cores(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
) -> typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]]: # (tucker_variations, outer_tt_cores)
    """Outer orthogonalize TT cores, pushing remainders downward onto tucker cores below.
    """
    use_jax = any([is_jax_ndarray(c) for c in tuple(x[0]) + tuple(x[1])])
    is_uniform = False
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    stack_shape = x[0][0].shape[:-2]

    def _func(Uio_Haib):
        Uio, Haib, = Uio_Haib

        rL, n, rR = Haib.shape[-3:]
        H_ab_i = Haib.swapaxes(-2, -1).reshape(stack_shape + (rL * rR, n))

        O_ab_x, ssx, WTxi = xnp.linalg.svd(H_ab_i, full_matrices=False)
        n2 = ssx.shape[-1]
        Oaxb = O_ab_x.reshape(stack_shape + (rL, rR, n2)).swapaxes(-2, -1)

        Cxi = ssx.reshape(stack_shape + (-1, 1)) * WTxi

        Vxo = np.einsum('...xi,...io->...xo', Cxi, Uio)
        return (Vxo, Oaxb)

    tucker_variations, outer_tt_cores = xmap(_func, x)
    return (tucker_variations, outer_tt_cores)


def down_orthogonalize_tucker_cores(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
) -> typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]]: # (up_tucker_cores, new_tt_cores)
    """Orthogonalize Tucker cores upwards, pushing remainders onto TT cores above.
    """
    use_jax = any([is_jax_ndarray(c) for c in tuple(x[0]) + tuple(x[1])])
    is_uniform = False
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    def _func(up_func_args):
        Bio, Gaib = up_func_args
        Boi = Bio.swapaxes(-1,-2)

        Uox, ssx, WTxi = xnp.linalg.svd(Boi, full_matrices=False)
        Rxi = xnp.einsum('...x,...xi->...xi', ssx, WTxi)

        new_Gaxb = xnp.einsum('...aib,...xi->...axb', Gaib, Rxi)
        new_Uxo = Uox.swapaxes(-1,-2)
        return (new_Uxo, new_Gaxb)

    up_tucker_cores, new_tt_cores = xmap(_func, x)
    return (up_tucker_cores, new_tt_cores)


def down_svd_tucker_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,  # which base backend to orthogonalize
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # new_x
    NDArray,  # ss_x. singular values
]:
    '''Compute SVD of ith tucker core and contract non-orthogonal factor up into the TT-core above.
    '''
    tucker_cores, tt_cores = x

    G_a_i_b = tt_cores[ii]
    U_i_o = tucker_cores[ii]

    new_G, new_B, ss_x = linalg.down_svd_pair(
        G_a_i_b, U_i_o, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
    )

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = new_G

    new_tucker_cores = list(tucker_cores)
    new_tucker_cores[ii] = new_B

    new_x = (tuple(new_tucker_cores), tuple(new_tt_cores))

    return new_x, ss_x


def left_svd_tt_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,  # which tt core to orthogonalize
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # new_x
    NDArray,  # singular values, shape=(r(i+1),)
]:
    '''Compute SVD of ith TT-core left unfolding and contract non-orthogonal factor into the TT-core to the right.
    '''
    use_jax = any([is_jax_ndarray(c) for c in tuple(x[0]) + tuple(x[1])])
    is_uniform = False
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    tucker_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii]
    new_tt_cores = list(tt_cores)

    if ii < len(x[0]) - 1:
        B0_b_j_c = tt_cores[ii + 1]

        A_a_i_x, B_x_j_c, ss_x = linalg.left_svd_pair(
            A0_a_i_b, B0_b_j_c, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )

        new_tt_cores[ii] = A_a_i_x
        new_tt_cores[ii + 1] = B_x_j_c
    else:
        U_i_a_x, ss_x, Vt_x_b = linalg.left_svd(
            A0_a_i_b, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )
        A_a_i_x = xnp.einsum('...iax,...x,...xb->...iax', U_i_a_x, ss_x, Vt_x_b) # sum over 'b' index
        new_tt_cores[ii] = A_a_i_x

    return (tuple(tucker_cores), tuple(new_tt_cores)), ss_x


def right_svd_tt_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,  # which tt core to orthogonalize
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # new_x
    NDArray,  # singular values, shape=(new_ri,)
]:
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor into the TT-core to the left.
    '''
    use_jax = any([is_jax_ndarray(c) for c in tuple(x[0]) + tuple(x[1])])
    is_uniform = False
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    tucker_cores, tt_cores = x

    B0_b_j_c = tt_cores[ii]
    new_tt_cores = list(tt_cores)

    if ii > 0:
        A0_a_i_b = tt_cores[ii - 1]

        A_a_i_x, B_x_j_c, ss_x = linalg.right_svd_pair(
            A0_a_i_b, B0_b_j_c, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )

        new_tt_cores[ii-1] = A_a_i_x
        new_tt_cores[ii] = B_x_j_c
    else:
        U_a_x, ss_x, Vt_x_j_c = linalg.right_svd(
            B0_b_j_c, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )
        B_x_j_c = xnp.einsum('...ax,...x,...xjc->...xjc', U_a_x, ss_x, Vt_x_j_c) # sum over 'a' index

        new_tt_cores[ii] = new_tt_cores[ii] = B_x_j_c

    return (tuple(tucker_cores), tuple(new_tt_cores)), ss_x


def down_svd_tt_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,  # which tt core to orthogonalize
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # new_x
    NDArray,  # singular values, shape=(new_ni,)
]:
    '''Compute SVD of ith TT-core outer unfolding and keep non-orthogonal factor with this core.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    #

    tucker_cores, tt_cores = x

    if len(tucker_cores[0].shape) > 2:
        raise RuntimeError(
            'Cannot use up_svd_ith_tt_core for stacked Tucker tensor train.\n' +
            'Different elements of the stack could end out having different shapes.\n' +
            'First unstack, then call up_svd_ith_tt_core for each unstacked Tucker tensor train.'
        )

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = tucker_cores[ii]

    U_a_x_b, ss_x, Vt_x_i = linalg.up_svd(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax=use_jax)

    G_a_x_b = xnp.einsum('...axb,...x->...axb', U_a_x_b, ss_x)
    Q_x_o = xnp.tensordot(Vt_x_i, Q0_i_o, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_tucker_cores = list(tucker_cores)
    new_tucker_cores[ii] = Q_x_o

    return (tuple(new_tucker_cores), tuple(new_tt_cores)), ss_x


def up_svd_tt_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,  # which tt core to orthogonalize
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # new_x
    NDArray,  # singular values, shape=(new_ni,)
]:
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the tucker core below.
    '''
    tucker_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = tucker_cores[ii]

    new_G, new_B, ss_x = linalg.up_svd_pair(
        G0_a_i_b, Q0_i_o, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
    )

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = new_G

    new_tucker_cores = list(tucker_cores)
    new_tucker_cores[ii] = new_B

    return (tuple(new_tucker_cores), tuple(new_tt_cores)), ss_x


def orthogonalize_relative_to_tucker_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith tucker core.
    '''
    # use_jax = any([is_jax_ndarray(c) for c in tuple(x[0]) + tuple(x[1])])
    # is_uniform = is_ndarray(x[0])
    # xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    num_cores = len(x[0])

    new_x = x
    for jj in range(ii):
        # new_x = up_svd_tt_core(new_x, jj)[0]
        new_x = down_svd_tucker_core(new_x, jj)[0]
        new_x = left_svd_tt_core(new_x, jj)[0]

    for jj in range(num_cores - 1, ii, -1):
        # new_x = up_svd_tt_core(new_x, jj)[0]
        new_x = down_svd_tucker_core(new_x, jj)[0]
        new_x = right_svd_tt_core(new_x, jj)[0]

    new_x = up_svd_tt_core(new_x, ii)[0]
    return new_x


def orthogonalize_relative_to_tt_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith TT-core.
    '''
    num_cores = len(x[0])

    new_x = x
    for jj in range(ii):
        # new_x = up_svd_tt_core(new_x, jj)[0]
        new_x = down_svd_tucker_core(new_x, jj)[0]
        new_x = left_svd_tt_core(new_x, jj)[0]

    for jj in range(num_cores - 1, ii, -1):
        # new_x = up_svd_tt_core(new_x, jj)[0]
        new_x = down_svd_tucker_core(new_x, jj)[0]
        new_x = right_svd_tt_core(new_x, jj)[0]

    new_x = down_svd_tucker_core(new_x, ii)[0]
    return new_x



