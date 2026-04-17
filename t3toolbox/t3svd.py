# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ


import t3toolbox.tucker_tensor_train as t3
import t3toolbox.orthogonalization as orth
import t3toolbox.uniform as ut3
import t3toolbox.util_linalg as linalg
from t3toolbox.common import *

__all__ = [
    't3_svd',
    'tucker_svd_dense',
    'tt_svd_dense',
    't3_svd_dense',
    'uniform_t3_svd',
]


###############################################
################    T3-SVD    #################
###############################################

def t3_svd(
        x: t3.TuckerTensorTrain,
        min_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        min_tucker_ranks:   typ.Sequence[int] = None,  # len=d
        max_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        max_tucker_ranks:   typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
        squash_tails_first: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    t3.TuckerTensorTrain, # new_x
    typ.Tuple[NDArray,...], # Tucker singular values, len=d
    typ.Tuple[NDArray,...], # TT singular values, len=d+1
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
        Singular values associated with edges between Tucker cores and TT-cores
    typ.Tuple[NDArray,...]
        Singular values associated with edges between adjacent TT-cores

    See Also
    --------
    left_svd_3tensor
    right_svd_3tensor
    outer_svd_3tensor
    up_svd_ith_tucker_core
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
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.t3svd as t3svd
    >>> x = t3.t3_corewise_randn((5,6,3), (4,4,3), (1,3,2,1))
    >>> x2, ss_tucker, ss_tt = t3svd.t3_svd(x) # Compute T3-SVD
    >>> x_dense = x.to_dense()
    >>> x2_dense = x2.to_dense()
    >>> print(np.linalg.norm(x_dense - x2_dense)) # Tensor unchanged
    7.556835759880194e-13
    >>> ss_tt1 = np.linalg.svd(x_dense.reshape((5, 6*3)))[1] # Singular values of unfolding 1
    >>> print(ss_tt1); print(ss_tt[1])
    [1.75326490e+02 3.41363029e+01 9.31164204e+00 1.33610061e-14 4.11601708e-15]
    [175.32648969  34.13630287   9.31164204]
    >>> ss_tucker2 = np.linalg.svd(x_dense.transpose([2,0,1]).reshape((3,5*6)))[1] # Singular values of matricization 2
    >>> print(ss_tucker2); print(ss_tucker[2])
    [1.71350937e+02 5.12857505e+01 1.36927051e-14]
    [171.35093708  51.28575045]

    T3-SVD with truncation based on relative tolerance:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.t3svd as t3svd
    >>> B0 = np.random.randn(35,40) @ np.diag(1.0 / np.arange(1, 41)**2) # preconditioned indices
    >>> B1 = np.random.randn(45,50) @ np.diag(1.0 / np.arange(1, 51)**2)
    >>> B2 = np.random.randn(55,60) @ np.diag(1.0 / np.arange(1, 61)**2)
    >>> G0 = np.random.randn(1,35,30)
    >>> G1 = np.random.randn(30,45,40)
    >>> G2 = np.random.randn(40,55,1)
    >>> tucker_cores_x = (B0, B1, B2)
    >>> tt_cores_x = (G0, G1, G2)
    >>> x = t3.TuckerTensorTrain(tucker_cores_x, tt_cores_x) # Tensor has spectral decay due to preconditioning
    >>> x2, ss_tucker, ss_tt = t3svd.t3_svd(x, rtol=1e-2) # Truncate singular values to reduce rank
    >>> print(x.structure)
    ((40, 50, 60), (35, 45, 55), (1, 30, 40, 1))
    >>> print(x2.structure)
    ((40, 50, 60), (6, 6, 5), (1, 6, 5, 1))
    >>> x_dense = x.to_dense()
    >>> x2_dense = x2.to_dense()
    >>> print(np.linalg.norm(x_dense - x2_dense)/np.linalg.norm(x_dense)) # Should be near rtol=1e-2
    0.013078458673911168

    T3-SVD with truncation based on absolute tolerance:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.t3svd as t3svd
    >>> x = t3.t3_corewise_randn((14,15,16), (10,11,12), (1,8,9,1))
    >>> x2, ss_tucker, ss_tt = t3svd.t3_svd(x, max_tucker_ranks=(3,3,3), max_tt_ranks=(1,2,2,1)) # Truncate based on ranks
    >>> print(x.structure)
        ((14, 15, 16), (10, 11, 12), (1, 8, 9, 1))
    >>> print(x2.structure)
        ((14, 15, 16), (3, 3, 2), (1, 2, 2, 1))

    Example where first and last ranks are not ones:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.t3svd as t3svd
    >>> x = t3.t3_corewise_randn((5,6,3), (4,4,3), (2,3,2,2))
    >>> x2, ss_tucker, ss_tt = t3svd.t3_svd(x, squash_tails_first=False) # Compute T3-SVD
    >>> x_dense = x.to_dense(squash_tails=False)
    >>> x2_dense = x2.to_dense(squash_tails=False)
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
    is_uniform = False
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if x.vectorization_shape != ():
        raise NotImplementedError(
            'T3-SVD is not implemented for TuckerTensorTrains with vectorized cores. Must do one at a time.'
        )

    # make leading and trailing TT-ranks equal to 1
    if squash_tails_first:
        x = x.squash_tails(use_jax=use_jax) #t3.squash_tails(x, use_jax=use_jax) # This causes a problem for some reason

    x = x.data
    tucker_cores, tt_cores = x

    num_cores = len(tt_cores)

    # Orthogonalize Tucker matrices
    x = orth.up_orthogonalize_tucker_cores(x, use_jax=use_jax)

    # Right orthogonalize
    x = x[0], orth.right_orthogonalize_tt_cores(x[1], use_jax=use_jax)

    G0 = x[1][0]
    _, ss_first, _ = linalg.right_svd_3tensor(G0, use_jax=use_jax)

    # Sweep left to right computing SVDS
    all_ss_tucker = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        min_rank = min_tucker_ranks[ii] if min_tucker_ranks is not None else None
        max_rank = max_tucker_ranks[ii] if max_tucker_ranks is not None else None
        # SVD inbetween TT core and Tucker core
        x, ss_tucker = orth.up_svd_ith_tt_core(
            ii, x, min_rank, max_rank, rtol, atol, use_jax=use_jax,
        )
        all_ss_tucker.append(ss_tucker)

        if ii < num_cores-1:
            min_rank = min_tt_ranks[ii+1] if min_tt_ranks is not None else None
            max_rank = max_tt_ranks[ii+1] if max_tt_ranks is not None else None
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = orth.left_svd_ith_tt_core(
                ii, x, min_rank, max_rank, rtol, atol, use_jax=use_jax,
            )
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = linalg.left_svd_3tensor(Gf, use_jax=use_jax)
        all_ss_tt.append(ss_tt)

    return t3.TuckerTensorTrain(*x), tuple(all_ss_tucker), tuple(all_ss_tt)



###############################################
#############    Dense T3-SVD    ##############
###############################################

def tucker_svd_dense(
        T: NDArray, # shape=(N1, N2, .., Nd)
        min_ranks:  typ.Sequence[int] = None, # len=d
        max_ranks:  typ.Sequence[int] = None,  # len=d
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
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
    xnp, xmap, xscan = get_backend(True, use_jax)

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
            C_swap_mat, min_rank, max_rank, rtol, atol, use_jax=use_jax,
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
        use_jax: bool = False,
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
    xnp, xmap, xscan = get_backend(True, use_jax)

    nn = T.shape

    X = T.reshape((1,) + T.shape)
    singular_values_of_unfoldings = []
    cores = []
    for ii in range(len(nn)-1):
        rL = X.shape[0]

        min_rank = None if min_ranks is None else min_ranks[ii+1]
        max_rank = None if max_ranks is None else max_ranks[ii+1]

        U, ss, Vt = linalg.truncated_svd(
            X.reshape((rL * nn[ii], -1)), min_rank, max_rank, rtol, atol, use_jax=use_jax,
        )
        rR = len(ss)

        singular_values_of_unfoldings.append(ss)
        cores.append(U.reshape((rL, nn[ii], rR)))
        X = ss.reshape((-1,1)) * Vt
    cores.append(X.reshape(X.shape + (1,)))

    norm_T_vec = xnp.array([xnp.linalg.norm(T)])
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
        use_jax: bool = False,
) -> typ.Tuple[
    t3.TuckerTensorTrain, # Approximation of T by Tucker tensor train
    typ.Tuple[NDArray,...], # Tucker singular values, len=d
    typ.Tuple[NDArray,...], # TT singular values, len=d+1
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
    >>> import t3toolbox.t3svd as t3svd
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
    (tucker_cores, tucker_core), ss_tucker = tucker_svd_dense(
        T, min_tucker_ranks, max_tucker_ranks, rtol, atol, use_jax=use_jax,
    )
    tt_cores, ss_tt = tt_svd_dense(
        tucker_core, min_tt_ranks, max_tt_ranks, rtol, atol, use_jax=use_jax,
    )
    return (tucker_cores, tt_cores), ss_tucker, ss_tt


###################################################
##############    Uniform T3-SVD    ###############
###################################################

def uniform_t3_svd(
        cores: ut3.UniformTuckerTensorTrain,
        masks: ut3.UniformEdgeWeights,
        squash_tails_first: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    ut3.UniformTuckerTensorTrain,
    NDArray, # basis_singular_values, shape=(d, n)
    NDArray, # tt_singular_values, shape=(d+1, r)
]:
    """Compute T3-SVD of uniform Tucker tensor train.

    Only guaranteed to give correct results if ranks are minimal.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform as ut3
    >>> import t3toolbox.t3svd as t3svd
    >>> s0 = ((11,12,13), (6,7,5), (1,3,6,2))
    >>> s = (s0[0],) + t3.compute_minimal_t3_ranks(s0)
    >>> x = t3.t3_corewise_randn(s)
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> ux2, ss_basis_from_ut3, ss_tt_from_ut3 = t3svd.uniform_t3_svd(cores, masks) # Uniform T3-SVD
    >>> print(np.linalg.norm(ut3.ut3_to_dense(ux2, masks) - t3.t3_to_dense(x)))
    3.782447238250888e-12
    >>> _, ss_basis, ss_tt = t3svd.t3_svd(x) # Non-uniform T3-SVD
    >>> print(ss_tt[1])
    [980.86624688 624.1067954  159.88424271]
    >>> print(ss_tt_from_ut3[1])
    [980.86624688 624.1067954  159.88424271   0.           0.        ]
    >>> _, tucker_masks, tt_masks = masks
    >>> print(ut3.unpack(ss_tt_from_ut3, tt_masks)[1])
    [980.86624688 624.1067954  159.88424271]
    >>> ut3.unpack(ss_basis_from_ut3, tucker_masks)[0] - ss_basis[0]
    array([ 1.13686838e-12, -2.27373675e-13, -1.13686838e-13])

    Uniform example with degenerate (unnecessairily large) ranks. Also using jax to test it

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform as ut3
    >>> import t3toolbox.t3svd as t3svd
    >>> import jax
    >>> import t3toolbox.corewise as cw
    >>> jax.config.update("jax_enable_x64", True)
    >>> structure = ((3,4,3), (4,6,7), (3,5,1,2))
    >>> x = t3.t3_corewise_randn(structure)
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> inv_masks = cw.corewise_logical_not(masks)
    >>> junk = ut3.uniform_randn(ut3.get_uniform_structure(cores), masks=inv_masks)
    >>> cores = cw.corewise_add(cores, junk) # Add random junk outside the masks
    >>> ux2, ss_basis_from_ut3, ss_tt_from_ut3 = t3svd.uniform_t3_svd(cores, masks, squash_tails_first=False, use_jax=True)
    >>> print(np.linalg.norm(ut3.ut3_to_dense(ux2, masks) - t3.t3_to_dense(x))) # OK
    9.404253555983741e-13
    >>> _, ss_basis, ss_tt = t3svd.t3_svd(x) # Non-uniform T3-SVD
    >>> print(ss_tt[1])
    [913.44494453 127.532224    16.08102313]
    >>> print(ss_tt_from_ut3[1]) # Incorrect singular values:
    [417.45514528 401.58448034  72.5343983   22.41273808   0.        ]
    >>> ux4, ss_basis_from_ut4, ss_tt_from_ut4 = t3svd.uniform_t3_svd(cores, masks, use_jax=True)
    >>> print(ss_basis_from_ut4[1])

    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    #

    if squash_tails_first:
        cores = ut3.uniform_squash_tails(cores)

    basis_supercore, tt_supercore = cores
    shape_mask, basis_masks, tt_masks = masks

    d, n, N = basis_supercore.shape
    r = tt_supercore.shape[1]

    basis_supercore, tt_supercore = orth.up_orthogonalize_tucker_cores((basis_supercore, tt_supercore), use_jax=use_jax)
    tt_supercore = orth.right_orthogonalize_tt_cores(tt_supercore, use_jax=use_jax)

    # keep everything the same shape, for consistency with masks
    _, n2, _ = basis_supercore.shape
    basis_supercore = xnp.concatenate([basis_supercore, xnp.zeros((d, n-n2, N))], axis=1)
    tt_supercore    = xnp.concatenate([tt_supercore,    xnp.zeros((d, r, n-n2, r))], axis=2)

    _, ss_tt00, _ = xnp.linalg.svd(tt_supercore[0].reshape((r, n*r)), full_matrices=False)
    ss_tt0 = xnp.concatenate([ss_tt00, xnp.zeros(r-len(ss_tt00))], axis=0)

    ss_tt0 = ss_tt0 * tt_masks[0]

    def _step(
            carry: NDArray,
            x,
    ):
        Y = carry # shape=(r, r)
        B, G, basis_mask, tt_mask = x

        G = xnp.einsum('ij,jak->iak', Y, G) # shape=(r, n, r)
        # Note: B.shape=(n, N)

        M = G.swapaxes(1,2).reshape((r*r, n))
        U, ss_basis, Vt = xnp.linalg.svd(M, full_matrices=False)
        n2 = len(ss_basis)
        U           = xnp.concatenate([U,           xnp.zeros((r*r, n-n2))],    axis=1)
        ss_basis    = xnp.concatenate([ss_basis,    xnp.zeros((n-n2, ))],       axis=0)
        Vt          = xnp.concatenate([Vt,          xnp.zeros((n-n2, n))],      axis=0)

        U           = U         * basis_mask.reshape((1,-1))
        ss_basis    = ss_basis  * basis_mask
        Vt          = Vt        * basis_mask.reshape((-1,1))

        new_B = xnp.einsum('ij,jk->ik', Vt, B)

        M = xnp.einsum('ij,j->ij', U, ss_basis).reshape((r, r, n)).swapaxes(1,2).reshape((r*n, r))
        U, ss_tt, Vt = xnp.linalg.svd(M, full_matrices=False)

        U       = U     * tt_mask.reshape((1,-1))
        ss_tt   = ss_tt * tt_mask
        Vt      = Vt    * tt_mask.reshape((-1,1))

        new_G = U.reshape((r, n, r))

        Y_next = xnp.einsum('i,ij->ij', ss_tt, Vt)  # shape=(r, r)

        return Y_next, (new_B, new_G, ss_basis, ss_tt)

    Y0 = xnp.eye(r)
    Yf, (new_basis_cores, new_tt_cores, basis_singular_values, tt_singular_values0) = xscan(
        _step,
        Y0,
        (basis_supercore, tt_supercore, basis_masks, tt_masks[1:]),
    )

    # G_last = xnp.einsum('diaj,jk->diak', new_tt_cores[-1:], Yf)[:, :, :, :r]
    G_last = xnp.einsum('diaj,jk->diak', new_tt_cores[-1:], Yf)
    new_tt_cores = xnp.concatenate([new_tt_cores[:-1], G_last], axis=0)

    tt_singular_values = xnp.concatenate([ss_tt0.reshape((1, r)), tt_singular_values0], axis=0)
    return (new_basis_cores, new_tt_cores), basis_singular_values, tt_singular_values
