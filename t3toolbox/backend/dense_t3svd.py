# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ
import math

import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.common as common
import t3toolbox.backend.ranks as ranks

__all__ = [
    'tucker_svd_dense',
    'ttsvd_dense',
    't3svd_dense',
]

def tucker_svd_dense(
        T: common.NDArray, # shape=(N1, N2, .., Nd)
        min_ranks:  typ.Sequence[int] = None, # len=d
        max_ranks:  typ.Sequence[int] = None,  # len=d
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[common.NDArray,...], # Tucker bases, ith_elm_shape=(ni, Ni)
        common.NDArray, # Tucker core, shape=(n1,n2,...,nd)
    ],
    typ.Tuple[common.NDArray,...], # singular values of matricizations
]:
    '''Compute Tucker decomposition and matricization singular values for dense tensor.

    Parameters
    ----------
    T: NDArray
        The dense tensor. shape=(N1, ..., Nd)
    min_ranks: typ.Sequence[int]
        Minimum Tucker ranks for truncation. len=d
    max_ranks: typ.Sequence[int]
        Maximum Tucker ranks for truncation. len=d
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[typ.Tuple[NDArray,...],NDArray]
        Tucker decomposition (tucker_bases, tucker_core). tucker_bases[ii].shape=(ni,Ni). tucker_core.shape=(n1,...,nd)
    typ.Tuple[NDArray,...]
        Singular values of matricizations

    See Also
    --------
    truncated_svd
    tt_svd_dense
    t3_svd_dense
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.common as common
    >>> import t3toolbox.t3svd as t3svd
    >>> T0 = np.random.randn(40, 50, 60)
    >>> c0 = 1.0 / np.arange(1, 41)**2
    >>> c1 = 1.0 / np.arange(1, 51)**2
    >>> c2 = 1.0 / np.arange(1, 61)**2
    >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2) # Preconditioned random tensor
    >>> (bases, core), ss = t3svd.tucker_svd_dense(T, rtol=1e-3) # Truncate Tucker SVD to reduce rank
    >>> print(core.shape)
    (9, 9, 9)
    >>> T2 = np.einsum('abc, ai,bj,ck->ijk', core, bases[0], bases[1], bases[2])
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
    0.002418671417862558
    '''
    bases = []
    singular_values_of_matricizations = []
    C = T
    for ii in range(len(T.shape)):
        C_swap = C.swapaxes(ii,0)
        old_shape_swap = C_swap.shape

        min_rank = None if min_ranks is None else min_ranks[ii]
        max_rank = None if max_ranks is None else max_ranks[ii]

        C_swap_mat = C_swap.reshape((old_shape_swap[0], -1))
        U, ss, Vt = linalg.truncated_svd(
            C_swap_mat, min_rank, max_rank, rtol, atol,
        )
        rM_new = len(ss)

        singular_values_of_matricizations.append(ss)
        bases.append(U.T)
        C_swap = (ss.reshape((-1,1)) * Vt).reshape((rM_new,) + old_shape_swap[1:])
        C = C_swap.swapaxes(0, ii)

    return (tuple(bases), C), tuple(singular_values_of_matricizations)


def ttsvd_dense(
        T: common.NDArray,
        min_ranks:  typ.Sequence[int] = None, # len=d+1
        max_ranks:  typ.Sequence[int] = None,  # len=d+1
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[common.NDArray,...], # tt_cores
    typ.Tuple[common.NDArray,...], # singular values of unfoldings
]:
    '''Compute tensor train (TT) decomposition and unfolding singular values for dense tensor.

    Parameters
    ----------
    T: NDArray
        The dense tensor. shape=(N1, ..., Nd)
    min_ranks: typ.Sequence[int]
        Minimum TT-ranks for truncation. len=d+1. e.g., (1,3,3,3,1)
    max_ranks: typ.Sequence[int]
        Maximum TT-ranks for truncation. len=d+1. e.g., (1,5,5,5,1)
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        TT cores. len=d. elm_shape=(ri, ni, r(i+1))
    typ.Tuple[NDArray,...]
        Singular values of unfoldings. len=d+1. elm_shape=(ri,)

    See Also
    --------
    truncated_svd
    tucker_svd_dense
    t3_svd_dense
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.t3svd as t3svd
    >>> T0 = np.random.randn(40, 50, 60)
    >>> c0 = 1.0 / np.arange(1, 41)**2
    >>> c1 = 1.0 / np.arange(1, 51)**2
    >>> c2 = 1.0 / np.arange(1, 61)**2
    >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2) # Preconditioned random tensor
    >>> cores, ss = t3svd.tt_svd_dense(T, rtol=1e-3) # Truncate TT-SVD to reduce rank
    >>> print([G.shape for G in cores])
    [(1, 40, 13), (13, 50, 13), (13, 60, 1)]
    >>> T2 = np.einsum('aib,bjc,ckd->ijk', cores[0], cores[1], cores[2])
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
    0.0023999063535883633
    '''
    use_jax = common.is_jax_ndarray(T)
    xnp, xmap, xscan = common.get_backend(False, use_jax)

    #
    nn = T.shape

    X = T.reshape((1,) + T.shape)
    singular_values_of_unfoldings = []
    cores = []
    for ii in range(len(nn)-1):
        rL = X.shape[0]

        min_rank = None if min_ranks is None else min_ranks[ii+1]
        max_rank = None if max_ranks is None else max_ranks[ii+1]

        U, ss, Vt = linalg.truncated_svd(
            X.reshape((rL * nn[ii], -1)), min_rank, max_rank, rtol, atol,
        )
        rR = len(ss)

        singular_values_of_unfoldings.append(ss)
        cores.append(U.reshape((rL, nn[ii], rR)))
        X = ss.reshape((-1,1)) * Vt
    cores.append(X.reshape(X.shape + (1,)))

    norm_T_vec = xnp.array([xnp.linalg.norm(T)])
    singular_values_of_unfoldings = [norm_T_vec,] + singular_values_of_unfoldings + [norm_T_vec,]

    return tuple(cores), tuple(singular_values_of_unfoldings)


def t3svd_dense(
        T: common.NDArray, # shape=stack_shape+(N0, .., N(d-1))
        stack_shape: typ.Sequence[int] = (),
        max_tucker_ranks:  typ.Sequence[int] = None,  # len=d
        max_tt_ranks:  typ.Sequence[int] = None,  # len=d+1
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[common.NDArray,...], # tucker_cores
        typ.Tuple[common.NDArray,...], # tt_cores
    ], # Approximation of T by Tucker tensor train
    typ.Tuple[common.NDArray,...], # Tucker singular values, len=d
    typ.Tuple[common.NDArray,...], # TT singular values, len=d+1
]:
    '''Compute TuckerTensorTrain and edge singular values for dense tensor.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.tucker_tensor_train.dense_t3svd as dt3svd
    >>> T = np.random.randn(2,3, 10,11,12)
    >>> (BB, GG), ss_tucker, ss_tt = dt3svd.t3svd_dense(T, stack_shape=(2,3))
    >>> GG_big = [np.einsum('...io,...aib->...aob', B, G) for B, G in zip(BB, GG)]
    >>> T2 = np.einsum('...aib,...bjc,...ckd->...ijk', *GG_big)
    >>> print(np.linalg.norm(T2 - T))
    3.4057168472825226e-13
    '''
    use_jax = common.is_jax_ndarray(T)
    xnp, xmap, xscan = common.get_backend(False, use_jax)

    #
    shape = T.shape[len(stack_shape):]

    # Accept scalar or per-position max ranks (None entry = no cap at that position).
    max_tucker_ranks    = ranks.normalize_max_ranks(max_tucker_ranks, len(shape))
    max_tt_ranks        = ranks.normalize_max_ranks(max_tt_ranks, len(shape)+1)

    ss_tt0 = xnp.linalg.norm(T.reshape(stack_shape+(-1,)), axis=-1).reshape(stack_shape + (1,))

    max_tt_ranks = list(max_tt_ranks)[1:]
    max_tucker_ranks = list(max_tucker_ranks)

    T = T.reshape(stack_shape + (1,) + shape)

    tucker_cores = []
    tt_cores = []
    ss_tucker = []
    ss_tt = [ss_tt0]
    while len(T.shape) > len(stack_shape)+1:
        rL = T.shape[len(stack_shape)]
        N = T.shape[len(stack_shape)+1]
        mm = T.shape[len(stack_shape)+2:]
        M = math.prod(mm)
        A = T.reshape(stack_shape + (rL, N, M)).swapaxes(-3, -2)
        A = A.reshape(stack_shape+(N, rL*M))

        U, ss, Vt = linalg.truncated_svd(A, max_rank=max_tucker_ranks[0], rtol=rtol, atol=atol)
        max_tucker_ranks = max_tucker_ranks[1:]
        n = ss.shape[-1]

        tucker_cores.append(U.swapaxes(-2,-1).copy())
        ss_tucker.append(ss)

        T = xnp.einsum(
            '...n,...nx->...nx', ss, Vt
        ).reshape(stack_shape + (n, rL, M)).swapaxes(-3, -2) # shape=stack_shape+(rL, n, M)

        A = T.reshape(stack_shape + (rL*n, M))
        U, ss, Vt = linalg.truncated_svd(A, max_rank=max_tt_ranks[0], rtol=rtol, atol=atol)
        max_tt_ranks = max_tt_ranks[1:]
        rR = ss.shape[-1]

        G = U.reshape(stack_shape + (rL, n, rR))
        tt_cores.append(G)
        ss_tt.append(ss)

        T = xnp.einsum('...r,...rx->...rx', ss, Vt).reshape(stack_shape + (rR,) + mm)

    Gf = tt_cores[-1]

    Gf = xnp.einsum('...aib,...b->...aib', Gf, T)
    tt_cores[-1] = Gf

    return (tuple(tucker_cores), tuple(tt_cores)), tuple(ss_tucker), tuple(ss_tt)


