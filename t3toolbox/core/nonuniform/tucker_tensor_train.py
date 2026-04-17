import numpy as np
import typing as typ

import t3toolbox.util_linalg as linalg
from t3toolbox.common import *

__all__ = [
    'to_dense',
    'squash_tt_tails',
    'reverse_tt',
    'change_tucker_core_shapes',
    'change_tt_core_shapes',
    't3_add',
    't3_scale',
    't3_inner_product_t3',
    'left_orthogonalize_tt_cores',
    'right_orthogonalize_tt_cores',
    'outer_orthogonalize_tt_cores',
    'up_orthogonalize_tucker_cores',
    'compute_minimal_ranks',
    'up_svd_ith_tucker_core',
    'left_svd_ith_tt_core',
    'right_svd_ith_tt_core',
    't3_get_entries',
]


def to_dense(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tt_cores, tucker_cores)
        squash_tails: bool = True,
        use_jax: bool = False,
) -> NDArray:
    """Fully contract a Tucker tensor train to create a dense tensor.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x
    vs = tucker_cores[0].shape[:-2] # vectorization_shape

    big_tt_cores = [xnp.einsum('...iaj,...ab->...ibj', G, U) for G, U in zip(tt_cores, tucker_cores)]

    T = big_tt_cores[0]
    for G in big_tt_cores[1:]:
        ts = T.shape[len(vs):-1]
        cs = (T.shape[-1],)
        T_a_b_c_xyz_r = T.reshape(vs + (xnp.prod(ts, dtype=int),) + cs)

        ts2 = G.shape[-2:]
        G_a_b_c_r_lm = G.reshape(vs + cs + (xnp.prod(ts2, dtype=int),))
        T_a_b_c_xyzlm = T_a_b_c_xyz_r @ G_a_b_c_r_lm
        T = T_a_b_c_xyzlm.reshape(vs + ts + ts2)

    if squash_tails:
        mu_L = xnp.ones(big_tt_cores[0].shape[-3])
        mu_R = xnp.ones(big_tt_cores[-1].shape[-1])

        T = xnp.tensordot(T, mu_R, axes=1)
        T = xnp.tensordot(mu_L, T, axes=((0,), (len(vs),)))

    return T


def squash_tt_tails(
        tt_cores: typ.Sequence[NDArray],
        use_jax: bool = False,
) -> typ.Tuple[NDArray]:
    """Make leading and trailing TT ranks equal to 1 (r0=rd=1), without changing tensor being represented.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    G0 = tt_cores[0]
    G0 = xnp.einsum('az,...aib->...zib', xnp.ones((G0.shape[-3],1)), G0)

    Gf = tt_cores[-1]
    Gf = xnp.einsum('...aib,bz->...aiz', Gf, xnp.ones((Gf.shape[-1],1)))

    return (G0,) + tuple(tt_cores[1:-1]) + (Gf,)


def reverse_tt(
        tt_cores: typ.Sequence[NDArray],
) -> typ.Tuple[NDArray,...]:
    """Reverse a tensor train (no Tucker).
    """
    return tuple([G.swapaxes(-3, -1) for G in tt_cores[::-1]])


def change_tucker_core_shapes(
        tucker_cores: typ.Sequence[NDArray],
        new_shape: typ.Sequence[int], # len=d
        new_tucker_ranks: typ.Sequence[int], # len=d
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]:
    """Increase Tucker and/or TT ranks for TT cores using zero padding.
    """
    xnp, _, _ = get_backend(False, use_jax)

    old_shape = [B.shape[1] for B in tucker_cores]
    old_tucker_ranks = [B.shape[0] for B in tucker_cores]

    num_cores = len(tucker_cores)

    delta_shape         = [N_new - N_old for N_new, N_old in zip(new_shape, old_shape)]
    delta_tucker_ranks  = [n_new - n_old for n_new, n_old in zip(new_tucker_ranks, old_tucker_ranks)]

    new_tucker_cores = []
    for ii in range(num_cores):
        new_tucker_cores.append(xnp.pad(
            tucker_cores[ii],
            (
                (0,delta_tucker_ranks[ii]),
                (0,delta_shape[ii]),
            ),
        ))

    return tuple(new_tucker_cores)


def change_tt_core_shapes(
        tt_cores: typ.Sequence[NDArray],
        new_tucker_ranks: typ.Sequence[int], # len=d
        new_tt_ranks: typ.Sequence[int], # len=d+1
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]:
    """Increase Tucker and/or TT ranks for TT cores using zero padding.
    """
    xnp, _, _ = get_backend(False, use_jax)

    old_tucker_ranks = [G.shape[1] for G in tt_cores]
    old_tt_ranks = [G.shape[0] for G in tt_cores] + [tt_cores[-1].shape[2]]

    num_cores = len(tt_cores)

    delta_tucker_ranks  = [n_new - n_old for n_new, n_old in zip(new_tucker_ranks, old_tucker_ranks)]
    delta_tt_ranks      = [r_new - r_old for r_new, r_old in zip(new_tt_ranks, old_tt_ranks)]

    new_tt_cores = []
    for ii in range(num_cores):
        new_tt_cores.append(xnp.pad(
            tt_cores[ii],
            (
                (0,delta_tt_ranks[ii]),
                (0,delta_tucker_ranks[ii]),
                (0,delta_tt_ranks[ii+1]),
            ),
        ))

    return tuple(new_tt_cores)


###########################################################
##################    Linear algebra    ###################
###########################################################

def t3_add(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
        squash: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[typ.Tuple[NDArray], typ.Tuple[NDArray]]: # (x_plus_y_tucker_cores, x_plus_y_tt_cores)
    """Add Tucker tensor trains x and y, yielding a Tucker tensor train with summed ranks.
    """
    xnp, _, _ = get_backend(False, use_jax)

    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    assert(vsx == vsy)

    tucker_cores_z = [xnp.concatenate([Bx, By], axis=-2) for Bx, By in zip(tucker_cores_x, tucker_cores_y)]

    tt_cores_z = []

    for Gx, Gy in zip(tt_cores_x, tt_cores_y):
        G000 = Gx
        G001 = xnp.zeros(vsx + (Gx.shape[-3], Gx.shape[-2], Gy.shape[-1]))
        G010 = xnp.zeros(vsx + (Gx.shape[-3], Gy.shape[-2], Gx.shape[-1]))
        G011 = xnp.zeros(vsx + (Gx.shape[-3], Gy.shape[-2], Gy.shape[-1]))
        G100 = xnp.zeros(vsx + (Gy.shape[-3], Gx.shape[-2], Gx.shape[-1]))
        G101 = xnp.zeros(vsx + (Gy.shape[-3], Gx.shape[-2], Gy.shape[-1]))
        G110 = xnp.zeros(vsx + (Gy.shape[-3], Gy.shape[-2], Gx.shape[-1]))
        G111 = Gy
        Gz = xnp.block([[[G000, G001], [G010, G011]], [[G100, G101], [G110, G111]]])
        tt_cores_z.append(Gz)

    if squash:
        z = (tuple(tucker_cores_z), squash_tt_tails(tt_cores_z))
    else:
        z = (tuple(tucker_cores_z), tuple(tt_cores_z))
    return z


def t3_scale(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        s,  # scalar
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]: # x*s
    """Multipy a Tucker tensor train by a scaling factor.
    """
    tucker_cores, tt_cores = x

    scaled_tucker_cores = [B.copy() for B in tucker_cores]
    scaled_tucker_cores[-1] = s * scaled_tucker_cores[-1]

    copied_tt_cores = [G.copy() for G in tt_cores]

    return tuple(scaled_tucker_cores), tuple(copied_tt_cores)


def t3_inner_product_t3(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt inner product of two Tucker tensor trains.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    tt_cores_x = squash_tt_tails(tt_cores_x)
    tt_cores_y = squash_tt_tails(tt_cores_y)

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    assert(vsx == vsy)

    r0_x = tt_cores_x[0].shape[-3]
    r0_y = tt_cores_y[0].shape[-3]

    M_sp = xnp.ones((r0_x, r0_y))
    for Bx_ai, Gx_sat, By_bi, Gy_pbq in zip(tucker_cores_x, tt_cores_x, tucker_cores_y, tt_cores_y):
        tmp_ab = xnp.einsum('...ai,...bi->...ab', Bx_ai, By_bi)
        tmp_sbt = xnp.einsum('...sat,...ab->...sbt', Gx_sat, tmp_ab)
        tmp_pbt = xnp.einsum('...sp,...sbt->...pbt', M_sp, tmp_sbt)
        tmp_tq = xnp.einsum('...pbt,...pbq->...tq', tmp_pbt, Gy_pbq)
        M_sp = tmp_tq

    rd_x = tt_cores_x[-1].shape[2]
    rd_y = tt_cores_y[-1].shape[2]

    result = xnp.einsum('tq,t,q', M_sp, np.ones(rd_x), np.ones(rd_y))
    return result


####


def left_orthogonalize_tt_cores(
        tt_cores: typ.Sequence[NDArray],  # len=d, elm_shape=(ri,ni,r(i+1))
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
    def _left_func(Cxb, left_func_args):
        Gbjc = left_func_args[0]

        Hxjc = xnp.einsum('xb,bjc->xjc', Cxb, Gbjc)

        rL, n, rR = Hxjc.shape
        H_xj_c = Hxjc.reshape((rL * n, rR))
        L_xj_y, ssy, VTyc = xnp.linalg.svd(H_xj_c, full_matrices=False)
        rR2 = len(ssy)
        Lxjy = L_xj_y.reshape((rL, n, rR2))

        Cyc = ssy.reshape((-1, 1)) * VTyc

        return Cyc, (Lxjy, Hxjc)

    init = xnp.eye(tt_cores[0].shape[0])
    xs = (tt_cores[:-1],)

    Cf, (LL, HH) = xscan(_left_func, init, xs)

    # Dealing with the last core as a special case
    Lf = xnp.einsum('xb,bjc->xjc', Cf, tt_cores[-1])
    left_tt_cores = tuple(LL) + (Lf,)
    var_tt_cores = tuple(HH) + (Lf,)

    if return_variation_cores:
        return left_tt_cores, var_tt_cores
    else:
        return left_tt_cores


def right_orthogonalize_tt_cores(
        tt_cores: typ.Sequence[NDArray],  # len=d, elm_shape=(ri,ni,r(i+1))
        return_variation_cores: bool = False,
        use_jax: bool = False,
) -> typ.Union[
    typ.Tuple[NDArray,...], # right_tt_cores
    typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]], # right_tt_cores, var_tt_cores
]:
    result = left_orthogonalize_tt_cores(
        reverse_tt(tt_cores), return_variation_cores=return_variation_cores, use_jax=use_jax,
    )
    if return_variation_cores:
        return reverse_tt(result[0]), reverse_tt(result[1])
    else:
        return reverse_tt(result)


def outer_orthogonalize_tt_cores(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        use_jax: bool = False,
) -> typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]]: # (tucker_variations, outer_tt_cores)
    """Outer orthogonalize TT cores, pushing remainders downward onto tucker cores below.
    """
    is_uniform = False
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    def _down_func(Uio_Haib):
        Uio, Haib, = Uio_Haib

        rL, n, rR = Haib.shape
        H_ab_i = Haib.swapaxes(1, 2).reshape((rL * rR, n))

        O_ab_x, ssx, WTxi = xnp.linalg.svd(H_ab_i, full_matrices=False)
        n2 = len(ssx)
        Oaxb = O_ab_x.reshape((rL, rR, n2)).swapaxes(1, 2)

        Cxi = ssx.reshape((-1, 1)) * WTxi

        Vxo = np.einsum('xi,io->xo', Cxi, Uio)
        return (Vxo, Oaxb)

    tucker_variations, outer_tt_cores = xmap(_down_func, x)
    return (tucker_variations, outer_tt_cores)


def up_orthogonalize_tucker_cores(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        use_jax: bool = False,
) -> typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]]: # (up_tucker_cores, new_tt_cores)
    """Orthogonalize Tucker cores upwards, pushing remainders onto TT cores above.
    """
    is_uniform = False
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    def _up_func(up_func_args):
        Bio, Gaib = up_func_args
        Boi = Bio.T

        Uox, ssx, WTxi = xnp.linalg.svd(Boi, full_matrices=False)
        Rxi = xnp.einsum('x,xi->xi', ssx, WTxi)

        new_Gaxb = xnp.einsum('aib,xi->axb', Gaib, Rxi)
        new_Uxo = Uox.T
        return (new_Uxo, new_Gaxb)

    up_tucker_cores, new_tt_cores = xmap(_up_func, x)
    return (up_tucker_cores, new_tt_cores)


def up_svd_ith_tucker_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,  # which base core to orthogonalize
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # new_x
    NDArray,  # ss_x. singular values
]:
    '''Compute SVD of ith tucker core and contract non-orthogonal factor up into the TT-core above.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x

    G_a_i_b = tt_cores[ii]
    U_i_o = tucker_cores[ii]
    U_o_i = U_i_o.T

    U2_o_x, ss_x, Vt_x_i = linalg.truncated_svd(U_o_i, min_rank, max_rank, rtol, atol, use_jax=use_jax)
    R_x_i = xnp.einsum('x,xi->xi', ss_x, Vt_x_i)
    # U2_o_x, R_x_i = xnp.linalg.qr(U_o_i, mode='reduced')

    G2_a_x_b = xnp.einsum('aib,xi->axb', G_a_i_b, R_x_i)
    U2_x_o = U2_o_x.T

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G2_a_x_b

    new_tucker_cores = list(tucker_cores)
    new_tucker_cores[ii] = U2_x_o

    new_x = (tuple(new_tucker_cores), tuple(new_tt_cores))

    return new_x, ss_x


def left_svd_ith_tt_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,  # which tt core to orthogonalize
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # new_x
    NDArray,  # singular values, shape=(r(i+1),)
]:
    '''Compute SVD of ith TT-core left unfolding and contract non-orthogonal factor into the TT-core to the right.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii]
    B0_b_j_c = tt_cores[ii + 1]

    A_a_i_x, ss_x, Vt_x_b = linalg.left_svd_3tensor(A0_a_i_b, min_rank, max_rank, rtol, atol, use_jax=use_jax)
    B_x_j_c = xnp.tensordot(ss_x.reshape((-1, 1)) * Vt_x_b, B0_b_j_c, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = A_a_i_x
    new_tt_cores[ii + 1] = B_x_j_c

    return (tuple(tucker_cores), tuple(new_tt_cores)), ss_x


def right_svd_ith_tt_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,  # which tt core to orthogonalize
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # new_x
    NDArray,  # singular values, shape=(new_ri,)
]:
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor into the TT-core to the left.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii - 1]
    B0_b_j_c = tt_cores[ii]

    U_b_x, ss_x, B_x_j_c = linalg.right_svd_3tensor(B0_b_j_c, min_rank, max_rank, rtol, atol, use_jax=use_jax)
    A_a_i_x = xnp.tensordot(A0_a_i_b, U_b_x * ss_x.reshape((1, -1)), axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii - 1] = A_a_i_x
    new_tt_cores[ii] = B_x_j_c

    return (tuple(tucker_cores), tuple(new_tt_cores)), ss_x


def up_svd_ith_tt_core(
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

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = tucker_cores[ii]

    U_a_x_b, ss_x, Vt_x_i = linalg.outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax=use_jax)

    G_a_x_b = xnp.einsum('axb,x->axb', U_a_x_b, ss_x)
    Q_x_o = xnp.tensordot(Vt_x_i, Q0_i_o, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_tucker_cores = list(tucker_cores)
    new_tucker_cores[ii] = Q_x_o

    return (tuple(new_tucker_cores), tuple(new_tt_cores)), ss_x


def down_svd_ith_tt_core(
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
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the tucker core below.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = tucker_cores[ii]

    G_a_x_b, ss_x, Vt_x_i = linalg.outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax=use_jax)

    Q_x_o = (ss_x.reshape((-1, 1)) * Vt_x_i) @ Q0_i_o

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_tucker_cores = list(tucker_cores)
    new_tucker_cores[ii] = Q_x_o

    return (tuple(new_tucker_cores), tuple(new_tt_cores)), ss_x


def orthogonalize_relative_to_ith_tucker_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith tucker core.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    #
    num_cores = len(x[0])

    new_x = x
    for jj in range(ii):
        new_x = down_svd_ith_tt_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]
        new_x = up_svd_ith_tucker_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]
        new_x = left_svd_ith_tt_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]

    for jj in range(num_cores - 1, ii, -1):
        new_x = down_svd_ith_tt_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]
        new_x = up_svd_ith_tucker_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]
        new_x = right_svd_ith_tt_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]

    new_x = down_svd_ith_tt_core(new_x, ii, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]
    return new_x


def orthogonalize_relative_to_ith_tt_core(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        ii: int,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith TT-core.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    #
    num_cores = len(x[0])

    new_x = x
    for jj in range(ii):
        new_x = down_svd_ith_tt_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]
        new_x = up_svd_ith_tucker_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]
        new_x = left_svd_ith_tt_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]

    for jj in range(num_cores - 1, ii, -1):
        new_x = down_svd_ith_tt_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]
        new_x = up_svd_ith_tucker_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]
        new_x = right_svd_ith_tt_core(new_x, jj, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]

    new_x = up_svd_ith_tucker_core(new_x, ii, min_rank, max_rank, rtol, atol, use_jax=use_jax)[0]
    return new_x

#

def compute_minimal_ranks(
        shape:          typ.Sequence[int],
        tucker_ranks:   typ.Sequence[int],
        tt_ranks:       typ.Sequence[int],
) -> typ.Tuple[
    typ.Tuple[int,...], # new_tucker_ranks
    typ.Tuple[int,...], # new_tt_ranks
]:
    '''Find minimal ranks for a generic Tucker tensor train with a given structure.
    '''
    d = len(shape)
    assert(len(tucker_ranks) == d)
    assert(len(tt_ranks) == d+1)

    new_tucker_ranks   = list(tucker_ranks)
    new_tt_ranks       = list(tt_ranks)

    for ii in range(d):
        new_tucker_ranks[ii] = int(np.minimum(new_tucker_ranks[ii], shape[ii]))

    new_tt_ranks[-1] = 1
    for ii in range(d-1, 0, -1):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        new_tt_ranks[ii] = int(np.minimum(rL, n*rR))

    new_tt_ranks[0] = 1
    for ii in range(d):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        n = int(np.minimum(n, rL*rR))
        rR =int(np.minimum(rR, rL*n))
        new_tucker_ranks[ii] = n
        new_tt_ranks[ii+1] = rR

    return tuple(new_tucker_ranks), tuple(new_tt_ranks)


def t3_get_entries(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        index: NDArray, # dtype=int, shape=(d,)+vsi
        use_jax: bool = False,
) -> NDArray:
    '''Compute entries of a Tucker tensor train.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x
    num_cores = len(x[0])
    vsx = x[0][0].shape[:-2]

    vsi = index.shape[1:]

    NX = np.prod(vsx, dtype=int) # yes, np. We want this computed statically
    NI = np.prod(vsi, dtype=int)

    mu_IXa = xnp.ones((NI, NX)+(tt_cores[0].shape[-3],))
    for ind, B_Xpo, G_Xapb in zip(index, tucker_cores, tt_cores):
        N = B_Xpo.shape[-1]
        rL = G_Xapb.shape[-3]
        n = G_Xapb.shape[-2]
        rR = G_Xapb.shape[-1]

        B_Xpo = B_Xpo.reshape((NX, n, N))
        G_Xapb = G_Xapb.reshape((NX, rL, n, rR))

        v_XpI = B_Xpo[:,:,ind]
        mu_IXb = xnp.einsum('...Xa,Xapb,Xp...->...Xb', mu_IXa, G_Xapb, v_XpI)
        mu_IXa = mu_IXb

    result = xnp.einsum('...Xa->...X', mu_IXa).reshape(vsi + vsx)
    return result