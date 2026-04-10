# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3tools.linalg
import t3tools.tucker_tensor_train as t3
import t3tools.orthogonalization as orth
from t3tools.common import NDArray


__all__ = [
    't3_svd',
    'tucker_svd_dense',
    'tt_svd_dense',
    't3_svd_dense',
]


###############################################
#############    Dense T3-SVD    ##############
###############################################

def t3_svd(
        x: t3.TuckerTensorTrain,
        min_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        min_tucker_ranks:   typ.Sequence[int] = None,  # len=d
        max_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        max_tucker_ranks:   typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> typ.Tuple[
    t3.TuckerTensorTrain, # new_x
    typ.Tuple[NDArray,...], # basis singular values, len=d
    typ.Tuple[NDArray,...], # tt singular values, len=d+1
]:
    '''Compute (truncated) T3-SVD of TuckerTensorTrain.

    Parameters
    ----------
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_tucker_ranks: typ.Sequence[int]
        Minimum Tucker ranks for truncation.
    min_tt_ranks: typ.Sequence[int]
        Minimum TT-ranks for truncation.
    max_tucker_ranks: typ.Sequence[int]
        Maximum Tucker ranks for truncation.
    max_tt_ranks: typ.Sequence[int]
        Maximum TT-ranks for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    NDArray
        New TuckerTensorTrain representing the same tensor (or a truncated version), but with modified cores
    typ.Tuple[NDArray,...]
        Singular values associated with edges between basis cores and TT-cores
    typ.Tuple[NDArray,...]
        Singular values associated with edges between adjacent TT-cores

    See Also
    --------
    left_svd_3tensor
    right_svd_3tensor
    outer_svd_3tensor
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    truncated_svd

    Examples
    --------

    T3-SVD with no truncation:
    (ranks may decrease to minimal values, but no approximation error)

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.t3svd as t3svd
    >>> x = t3.t3_corewise_randn(((5,6,3), (4,4,3), (1,3,2,1)))
    >>> x2, ss_basis, ss_tt = t3svd.t3_svd(x) # Compute T3-SVD
    >>> x_dense = t3.t3_to_dense(x)
    >>> x2_dense = t3.t3_to_dense(x2)
    >>> print(np.linalg.norm(x_dense - x2_dense)) # Tensor unchanged
    7.556835759880194e-13
    >>> ss_tt1 = np.linalg.svd(x_dense.reshape((5, 6*3)))[1] # Singular values of unfolding 1
    >>> print(ss_tt1); print(ss_tt[1])
    [1.75326490e+02 3.41363029e+01 9.31164204e+00 1.33610061e-14 4.11601708e-15]
    [175.32648969  34.13630287   9.31164204]
    >>> ss_basis2 = np.linalg.svd(x_dense.transpose([2,0,1]).reshape((3,5*6)))[1] # Singular values of matricization 2
    >>> print(ss_basis2); print(ss_basis[2])
    [1.71350937e+02 5.12857505e+01 1.36927051e-14]
    [171.35093708  51.28575045]

    T3-SVD with truncation based on relative tolerance:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.t3svd as t3svd
    >>> B0 = np.random.randn(35,40) @ np.diag(1.0 / np.arange(1, 41)**2) # preconditioned indices
    >>> B1 = np.random.randn(45,50) @ np.diag(1.0 / np.arange(1, 51)**2)
    >>> B2 = np.random.randn(55,60) @ np.diag(1.0 / np.arange(1, 61)**2)
    >>> G0 = np.random.randn(1,35,30)
    >>> G1 = np.random.randn(30,45,40)
    >>> G2 = np.random.randn(40,55,1)
    >>> basis_cores_x = (B0, B1, B2)
    >>> tt_cores_x = (G0, G1, G2)
    >>> x = (basis_cores_x, tt_cores_x) # Tensor has spectral decay due to preconditioning
    >>> x2, ss_basis, ss_tt = t3svd.t3_svd(x, rtol=1e-2) # Truncate singular values to reduce rank
    >>> print(t3.get_structure(x))
    ((40, 50, 60), (35, 45, 55), (1, 30, 40, 1))
    >>> print(t3.get_structure(x2))
    ((40, 50, 60), (6, 6, 5), (1, 6, 5, 1))
    >>> x_dense = t3.t3_to_dense(x)
    >>> x2_dense = t3.t3_to_dense(x2)
    >>> print(np.linalg.norm(x_dense - x2_dense)/np.linalg.norm(x_dense)) # Should be near rtol=1e-2
    0.013078458673911168

    T3-SVD with truncation based on absolute tolerance:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.t3svd as t3svd
    >>> x = t3.t3_corewise_randn(((14,15,16), (10,11,12), (1,8,9,1)))
    >>> x2, ss_basis, ss_tt = t3svd.t3_svd(x, max_tucker_ranks=(3,3,3), max_tt_ranks=(1,2,2,1)) # Truncate based on ranks
    >>> print(t3.get_structure(x))
        ((14, 15, 16), (10, 11, 12), (1, 8, 9, 1))
    >>> print(t3.get_structure(x2))
        ((14, 15, 16), (3, 3, 2), (1, 2, 2, 1))

    Example where first and last ranks are not ones:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.t3svd as t3svd
    >>> x = t3.t3_corewise_randn(((5,6,3), (4,4,3), (2,3,2,2)))
    >>> x2, ss_basis, ss_tt = t3svd.t3_svd(x) # Compute T3-SVD
    >>> x_dense = t3.t3_to_dense(x, contract_ones=False)
    >>> x2_dense = t3.t3_to_dense(x2, contract_ones=False)
    >>> print(np.linalg.norm(x_dense - x2_dense)) # Tensor unchanged
    5.486408687260824e-13
    >>> ss_tt0 = np.linalg.svd(x_dense.reshape((2,5*6*3*2)))[1] # Singular values of leading unfolding
    >>> print(ss_tt0); print(ss_tt[0])
    [303.0474449   88.85034392]
    [303.0474449   88.85034392]
    >>> ss_tt3 = np.linalg.svd(x_dense.reshape((2*5*6*3,2)))[1] # Singular values of trailing unfolding
    >>> print(ss_tt3); print(ss_tt[3])
    [299.45433768 100.29574828]
    [299.45433768 100.29574828]
    '''
    basis_cores, tt_cores = x

    num_cores = len(tt_cores)

    # make leading and trailing TT-ranks equal to 1
    x = t3.squash_tails(x, xnp=xnp)

    # Orthogonalize basis matrices
    for ii in range(num_cores):
        x, _ = orth.up_svd_ith_basis_core(ii, x, xnp=xnp)

    # Right orthogonalize
    for ii in range(num_cores-1, 0, -1): # num_cores-1, num_cores-2, ..., 1
        x, _ = orth.right_svd_ith_tt_core(ii, x, xnp=xnp)

    G0 = x[1][0]
    _, ss_first, _ = t3tools.linalg.right_svd_3tensor(G0, xnp=xnp)

    # Sweep left to right computing SVDS
    all_ss_basis = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        min_rank = min_tucker_ranks[ii] if min_tucker_ranks is not None else None
        max_rank = max_tucker_ranks[ii] if max_tucker_ranks is not None else None
        # SVD inbetween tt core and basis core
        x, ss_basis = orth.up_svd_ith_tt_core(
            ii, x, min_rank, max_rank, rtol, atol, xnp=xnp,
        )
        all_ss_basis.append(ss_basis)

        if ii < num_cores-1:
            min_rank = min_tt_ranks[ii+1] if min_tt_ranks is not None else None
            max_rank = max_tt_ranks[ii+1] if max_tt_ranks is not None else None
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = orth.left_svd_ith_tt_core(
                ii, x, min_rank, max_rank, rtol, atol, xnp=xnp,
            )
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = t3tools.linalg.left_svd_3tensor(Gf, xnp=xnp)
        all_ss_tt.append(ss_tt)

    return x, tuple(all_ss_basis), tuple(all_ss_tt)


def tucker_svd_dense(
        T: NDArray, # shape=(N1, N2, .., Nd)
        min_ranks:  typ.Sequence[int] = None, # len=d
        max_ranks:  typ.Sequence[int] = None,  # len=d
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[NDArray,...], # Tucker bases, ith_elm_shape=(ni, Ni)
        NDArray, # Tucker core, shape=(n1,n2,...,nd)
    ],
    typ.Tuple[NDArray,...], # singular values of matricizations
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
    >>> import t3tools.common as common
    >>> import t3tools.t3svd as t3svd
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
        U, ss, Vt = t3tools.linalg.truncated_svd(
            C_swap_mat, min_rank, max_rank, rtol, atol, xnp=xnp,
        )
        rM_new = len(ss)

        singular_values_of_matricizations.append(ss)
        bases.append(U.T)
        C_swap = (ss.reshape((-1,1)) * Vt).reshape((rM_new,) + old_shape_swap[1:])
        C = C_swap.swapaxes(0, ii)

    return (tuple(bases), C), tuple(singular_values_of_matricizations)


def tt_svd_dense(
        T: NDArray,
        min_ranks:  typ.Sequence[int] = None, # len=d+1
        max_ranks:  typ.Sequence[int] = None,  # len=d+1
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> typ.Tuple[
    typ.Tuple[NDArray,...], # tt_cores
    typ.Tuple[NDArray,...], # singular values of unfoldings
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
    >>> import t3tools.t3svd as t3svd
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
    nn = T.shape

    X = T.reshape((1,) + T.shape)
    singular_values_of_unfoldings = []
    cores = []
    for ii in range(len(nn)-1):
        rL = X.shape[0]

        min_rank = None if min_ranks is None else min_ranks[ii+1]
        max_rank = None if max_ranks is None else max_ranks[ii+1]

        U, ss, Vt = t3tools.linalg.truncated_svd(
            X.reshape((rL * nn[ii], -1)), min_rank, max_rank, rtol, atol, xnp=xnp,
        )
        rR = len(ss)

        singular_values_of_unfoldings.append(ss)
        cores.append(U.reshape((rL, nn[ii], rR)))
        X = ss.reshape((-1,1)) * Vt
    cores.append(X.reshape(X.shape + (1,)))

    norm_T_vec = np.array([np.linalg.norm(T)])
    singular_values_of_unfoldings = [norm_T_vec,] + singular_values_of_unfoldings + [norm_T_vec,]

    return tuple(cores), tuple(singular_values_of_unfoldings)


def t3_svd_dense(
        T: NDArray, # shape=(N1, N2, .., Nd)
        min_tucker_ranks:  typ.Sequence[int] = None, # len=d
        max_tucker_ranks:  typ.Sequence[int] = None,  # len=d
        min_tt_ranks:  typ.Sequence[int] = None, # len=d+1
        max_tt_ranks:  typ.Sequence[int] = None,  # len=d+1
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> typ.Tuple[
    t3.TuckerTensorTrain, # Approximation of T by Tucker tensor train
    typ.Tuple[NDArray,...], # basis singular values, len=d
    typ.Tuple[NDArray,...], # tt singular values, len=d+1
]:
    '''Compute TuckerTensorTrain and edge singular values for dense tensor.

    Parameters
    ----------
    T: NDArray
        The dense tensor. shape=(N1, ..., Nd)
    min_tucker_ranks: typ.Sequence[int]
        Minimum Tucker ranks for truncation. len=d. e.g., (3,3,3)
    max_tucker_ranks: typ.Sequence[int]
        Maximum Tucker ranks for truncation. len=d. e.g., (5,5,5)
    min_tt_ranks: typ.Sequence[int]
        Minimum TT-ranks for truncation. len=d+1. e.g., (1,3,3,3,1)
    max_tt_ranks: typ.Sequence[int]
        Maximum TT-ranks for truncation. len=d+1. e.g., (1,5,5,5,1)
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    TuckerTensorTrain
        Tucker tensor train approxiamtion of T
    typ.Tuple[NDArray,...]
        Singular values of matricizations. len=d. elm_shape=(ni,)
    typ.Tuple[NDArray,...]
        Singular values of unfoldings. len=d+1. elm_shape=(ri,)

    See Also
    --------
    truncated_svd
    tucker_svd_dense
    tt_svd_dense
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.t3svd as t3svd
    >>> T0 = np.random.randn(40, 50, 60)
    >>> c0 = 1.0 / np.arange(1, 41)**2
    >>> c1 = 1.0 / np.arange(1, 51)**2
    >>> c2 = 1.0 / np.arange(1, 61)**2
    >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2) # Preconditioned random tensor
    >>> x, ss_tucker, ss_tt = t3svd.t3_svd_dense(T, rtol=1e-3) # Truncate T3-SVD to reduce rank
    >>> print(t3.get_structure(x))
    ((40, 50, 60), (12, 11, 12), (1, 12, 12, 1))
    >>> T2 = t3.t3_to_dense(x)
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
    0.0025147026955504846
    '''
    (basis_cores, tucker_core), ss_tucker = tucker_svd_dense(
        T, min_tucker_ranks, max_tucker_ranks, rtol, atol, xnp=xnp,
    )
    tt_cores, ss_tt = tt_svd_dense(
        tucker_core, min_tt_ranks, max_tt_ranks, rtol, atol, xnp=xnp,
    )
    return (basis_cores, tt_cores), ss_tucker, ss_tt
