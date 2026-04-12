# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.linalg
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.base_variation_format as bvf
import t3toolbox.linalg as linalg
import t3toolbox.common as common

__all__ = [
    'up_svd_ith_tucker_core',
    'left_svd_ith_tt_core',
    'right_svd_ith_tt_core',
    'up_svd_ith_tt_core',
    'down_svd_ith_tt_core',
    'orthogonalize_relative_to_ith_tucker_core',
    'orthogonalize_relative_to_ith_tt_core',
    'orthogonal_representations',
]

NDArray = typ.TypeVar('NDArray') # Generic stand-in for np.ndarray, jnp.ndarray, or other array backend


##########################################
########    Orthogonalization    #########
##########################################

def up_svd_ith_tucker_core(
        ii: int, # which base core to orthogonalize
        x: t3.TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> typ.Tuple[
    t3.TuckerTensorTrain, # new_x
    NDArray, # ss_x. singular values
]:
    '''Compute SVD of ith tucker core and contract non-orthogonal factor up into the TT-core above.

    Parameters
    ----------
    ii: int
        index of tucker core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (r0,r1,...r(d-1),rd))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith tucker core orthogonal.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_tucker_cores[ii].shape = (new_ni, Ni)
        new_tucker_cores[ii] @ new_tucker_cores[ii].T = identity matrix
    ss_x: NDArray
        Singular values of prior ith tucker core. shape=(new_ni,).

    See Also
    --------
    truncated_svd
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> ind = 1
    >>> x2, ss = orth.up_svd_ith_tucker_core(ind, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    5.772851635866132e-13
    >>> tucker_cores2, tt_cores2 = x2
    >>> rank = len(ss)
    >>> B = tucker_cores2[ind]
    >>> print(np.linalg.norm(B @ B.T - np.eye(rank))) # Tucker core is orthogonal
    8.456498415401757e-16
    '''
    tucker_cores, tt_cores = x
    G_a_i_b = tt_cores[ii]
    U_i_o = tucker_cores[ii]
    U_o_i = U_i_o.T

    U2_o_x, ss_x, Vt_x_i = t3toolbox.linalg.truncated_svd(U_o_i, min_rank, max_rank, rtol, atol, xnp=xnp)
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
        ii: int, # which tt core to orthogonalize
        x: t3.TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> typ.Tuple[
    t3.TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(r(i+1),)
]:
    '''Compute SVD of ith TT-core left unfolding and contract non-orthogonal factor into the TT-core to the right.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith TT-core orthogonal.
        new_tt_cores[ii].shape = (ri, ni, new_r(i+1))
        new_tt_cores[ii+1].shape = (new_r(i+1), n(i+1), r(i+2))
        einsum('iaj,iak->jk', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
    ss_x: NDArray
        Singular values of prior ith TT-core left unfolding. shape=(new_r(i+1),).

    See Also
    --------
    truncated_svd
    left_svd_3tensor
    up_svd_ith_tucker_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> ind = 1
    >>> x2, ss = orth.left_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
        5.186463661974644e-13
    >>> tucker_cores2, tt_cores2 = x2
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', G, G) - np.eye(G.shape[2]))) # TT-core is left-orthogonal
        4.453244025338311e-16
    '''
    tucker_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii]
    B0_b_j_c = tt_cores[ii+1]

    A_a_i_x, ss_x, Vt_x_b = linalg.left_svd_3tensor(A0_a_i_b, min_rank, max_rank, rtol, atol, xnp=xnp)
    B_x_j_c = xnp.tensordot(ss_x.reshape((-1,1)) * Vt_x_b, B0_b_j_c, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = A_a_i_x
    new_tt_cores[ii+1] = B_x_j_c

    return (tuple(tucker_cores), tuple(new_tt_cores)), ss_x


def right_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: t3.TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> typ.Tuple[
    t3.TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(new_ri,)
]:
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor into the TT-core to the left.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith TT-core orthogonal.
        new_tt_cores[ii].shape = (new_ri, ni, r(i+1))
        new_tt_cores[ii-1].shape = (r(i-1), n(i-1), new_ri)
        einsum('iaj,kaj->ik', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
    ss_x: NDArray
        Singular values of prior ith TT-core right unfolding. shape=(new_ri,).

    See Also
    --------
    truncated_svd
    left_svd_3tensor
    up_svd_ith_tucker_core
    left_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> ind = 1
    >>> x2, ss = orth.right_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
        5.304678679078675e-13
    >>> tucker_cores2, tt_cores2 = x2
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(np.einsum('iaj,kaj->ik', G, G) - np.eye(G.shape[0]))) # TT-core is right orthogonal
        4.207841813173725e-16
    '''
    tucker_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii-1]
    B0_b_j_c = tt_cores[ii]

    U_b_x, ss_x, B_x_j_c = linalg.right_svd_3tensor(B0_b_j_c, min_rank, max_rank, rtol, atol, xnp=xnp)
    A_a_i_x = xnp.tensordot(A0_a_i_b, U_b_x * ss_x.reshape((1,-1)), axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii-1] = A_a_i_x
    new_tt_cores[ii] = B_x_j_c

    return (tuple(tucker_cores), tuple(new_tt_cores)), ss_x


def up_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: t3.TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> typ.Tuple[
    t3.TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(new_ni,)
]:
    '''Compute SVD of ith TT-core outer unfolding and keep non-orthogonal factor with this core.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_tucker_cores[ii].shape = (new_ni, Ni)
    ss_x: NDArray
        Singular values of prior ith TT-core outer unfolding. shape=(new_ri,).

    See Also
    --------
    truncated_svd
    outer_svd_3tensor
    up_svd_ith_tucker_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> x2, ss = orth.up_svd_ith_tt_core(1, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    1.002901486286745e-12
    '''
    tucker_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = tucker_cores[ii]

    U_a_x_b, ss_x, Vt_x_i = linalg.outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, xnp=xnp)

    G_a_x_b = xnp.einsum('axb,x->axb', U_a_x_b, ss_x)
    Q_x_o = xnp.tensordot(Vt_x_i, Q0_i_o, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_tucker_cores = list(tucker_cores)
    new_tucker_cores[ii] = Q_x_o

    return (tuple(new_tucker_cores), tuple(new_tt_cores)), ss_x


def down_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: t3.TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> typ.Tuple[
    t3.TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(new_ni,)
]:
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the tucker core below.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith TT-core outer orthogonal.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_tucker_cores[ii].shape = (new_ni, Ni)
        einsum('iaj,ibj->ab', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
    ss_x: NDArray
        Singular values of prior ith TT-core outer unfolding. shape=(new_ni,).

    See Also
    --------
    truncated_svd
    outer_svd_3tensor
    up_svd_ith_tucker_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> ind = 1
    >>> x2, ss = orth.down_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    4.367311712704942e-12
    >>> tucker_cores2, tt_cores2 = x2
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(np.einsum('iaj,ibj->ab', G, G) - np.eye(G.shape[1]))) # TT-core is outer orthogonal
    1.0643458053135608e-15
    '''
    tucker_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = tucker_cores[ii]

    G_a_x_b, ss_x, Vt_x_i = linalg.outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, xnp=xnp)

    Q_x_o = (ss_x.reshape((-1,1)) * Vt_x_i) @ Q0_i_o

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_tucker_cores = list(tucker_cores)
    new_tucker_cores[ii] = Q_x_o

    return (tuple(new_tucker_cores), tuple(new_tt_cores)), ss_x


def orthogonalize_relative_to_ith_tucker_core(
        ii: int,
        x: t3.TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> t3.TuckerTensorTrain:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith tucker core.

    Orthogonal is done relative to the ith tucker core:
        - ith tucker core is not orthogonalized
        - All other tucker cores are orthogonalized.
        - TT-cores to the left are left orthogonalized.
        - TT-core directly above is outer orthogonalized.
        - TT-cores to the right are right orthogonalized.

    Parameters
    ----------
    ii: int
        index of tucker core that is not orthogonalized
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith tucker core.

    See Also
    --------
    up_svd_ith_tucker_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> x2 = orth.orthogonalize_relative_to_ith_tucker_core(1, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    8.800032152216517e-13
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> X = np.einsum('xi,axb,byc,czd,zk->iyk', B0, G0, G1, G2, B2) # Contraction of everything except B1
    >>> print(np.linalg.norm(np.einsum('iyk,iwk->yw', X, X) - np.eye(B1.shape[0]))) # Complement of B1 is orthogonal
    1.7116160385376214e-15

    Example where first and last TT-ranks are not 1:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> x2 = orth.orthogonalize_relative_to_ith_tucker_core(0, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    5.152424496985265e-12
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> X = np.einsum('yj,zk,axb,byc,czd->axjkd', B1, B2, G0, G1, G2) # Contraction of everything except B0
    >>> print(np.linalg.norm(np.einsum('axjkd,ayjkd->xy', X, X) - np.eye(B0.shape[0]))) # Complement of B1 is orthogonal
    2.3594586449868743e-15
    '''
    shape, tucker_ranks, tt_ranks = t3.get_structure(x)

    new_x = x
    for jj in range(ii):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]
        new_x = up_svd_ith_tucker_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]
        new_x = left_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]

    for jj in range(len(shape)-1, ii, -1):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]
        new_x = up_svd_ith_tucker_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]
        new_x = right_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]

    new_x = down_svd_ith_tt_core(ii, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]
    return new_x


def orthogonalize_relative_to_ith_tt_core(
        ii: int,
        x: t3.TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        xnp = np,
) -> t3.TuckerTensorTrain:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith TT-core.

    Orthogonal is done relative to the ith TT-core:
        - All Tucker cores are orthogonalized.
        - TT-cores to the left are left orthogonalized.
        - ith TT-core is not orthogonalized.
        - TT-cores to the right are right orthogonalized.

    Parameters
    ----------
    ii: int
        index of TT-core that is not orthogonalized
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    See Also
    --------
    up_svd_ith_tucker_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith TT-core.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> x2 = orth.orthogonalize_relative_to_ith_tt_core(1, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    8.800032152216517e-13
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> XL = np.einsum('axb,xi -> aib', G0, B0) # Everything to the left of G1
    >>> print(np.linalg.norm(np.einsum('aib,aic->bc', XL, XL) - np.eye(G1.shape[0]))) # Left subtree is left orthogonal
    9.820411604510197e-16
    >>> print(np.linalg.norm(np.einsum('xi,yi->xy', B1, B1) - np.eye(G1.shape[1]))) # Core below G1 is up orthogonal
    2.1875310121178e-15
    >>> XR = np.einsum('axb,xi->aib', G2, B2) # Everything to the right of G1
    >>> print(np.linalg.norm(np.einsum('aib,cib->ac', XR, XR) - np.eye(G1.shape[2]))) # Right subtree is right orthogonal
    1.180550381921849e-15

    Example where first and last TT-ranks are not 1:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> x2 = orth.orthogonalize_relative_to_ith_tt_core(0, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    5.4708999671349535e-12
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> XR = np.einsum('yi,zj,byc,czd->bijd', B1, B2, G1, G2) # Everything to the right of G0
    >>> print(np.linalg.norm(np.einsum('bijd,cijd->bc', XR, XR) - np.eye(G0.shape[2]))) # Right subtree is right orthogonal
    8.816596607002667e-16
    '''
    shape, tucker_ranks, tt_ranks = t3.get_structure(x)

    new_x = x
    for jj in range(ii):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]
        new_x = up_svd_ith_tucker_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]
        new_x = left_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]

    for jj in range(len(shape)-1, ii, -1):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]
        new_x = up_svd_ith_tucker_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]
        new_x = right_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]

    new_x = up_svd_ith_tucker_core(ii, new_x, min_rank, max_rank, rtol, atol, xnp=xnp)[0]
    return new_x


def orthogonal_representations(
        x: t3.TuckerTensorTrain,
        already_left_orthogonal: bool = False,
        map = common.ragged_map,
        scan = common.ragged_scan,
        xnp = np,
) -> typ.Tuple[
    bvf.T3Base, # orthogonal base
    bvf.T3Variation, # variations
]:
    '''Construct base-variation representations of TuckerTensorTrain with orthogonal base.

    Input TuckerTensorTrain::

                  1 -- G0 -- G1 -- G2 -- G3 -- 1
        X    =         |     |     |     |
                       B0    B1    B2    B3
                       |     |     |     |

    Base-variation representation with non-orthogonal TT-core H1::

                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    U2    U3
                       |     |     |     |

    Base-variation representation with non-orthogonal tucker core V2::

                  1 -- L0 -- L1 -- O2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    V2    U3
                       |     |     |     |

    The input tensor train x is defined by:
        - x_tucker_cores     = (B0, B1, B2, B3)
        - x_tt_cores        = (G0, G1, G2, G3)
    The "base cores" are:
        - tucker_cores       = (U0,U1, U2, U3), up orthogonal
        - left_tt_cores     = (L0, L1, L2, L3), left orthogonal
        - right_tt_cores    = (R0, R1, R2, R3), right orthogonal
        - outer_tt_cores    = (O0, O1, O2, O3), down orthogonal
    The "variation cores" are:
        - tucker_variations  = (V0, V1, V2, V3)
        - tt_variations     = (H0, H1, H2, H3)

    Parameters
    ----------
    x: TuckerTensorTrain
        Input TuckerTensorTrain
        x = (x_tucker_cores, x_tt_cores)
        x_tucker_cores = (B0, ..., Bd)
        x_tt_cores = (G0, ..., Gd)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    T3Base
        Orthogonal base for base-variation representations of x.
    T3Variation
        Variation for base-variation representaions of x.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, variation = orth.orthogonal_representations(x) # Compute orthogonal representations
    >>> tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    >>> tucker_vars, tt_vars = variation
    >>> (U0,U1,U2) = tucker_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (O0,O1,O2) = outer_tt_cores
    >>> (V0,V1,V2) = tucker_vars
    >>> (H0,H1,H2) = tt_vars
    >>> x2 = ((U0,U1,U2), (L0,H1,R2)) # representation with TT-core variation in index 1
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Still represents origional tensor
    4.978421562425667e-12
    >>> x3 = ((U0,V1,U2), (L0,O1,R2)) # representation with tucker core variation in index 1
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x3))) # Still represents origional tensor
    5.4355175448533146e-12
    >>> print(np.linalg.norm(U1 @ U1.T - np.eye(U1.shape[0]))) # U: orthogonal
    1.1915111872574236e-15
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', L1, L1) - np.eye(L1.shape[2]))) # L: left orthogonal
    9.733823879665448e-16
    >>> print(np.linalg.norm(np.einsum('iaj,kaj->ik', R1, R1) - np.eye(R1.shape[0]))) # R: right orthogonal
    8.027553546330097e-16
    >>> print(np.linalg.norm(np.einsum('iaj,ibj->ab', O1, O1) - np.eye(O1.shape[1]))) # O: outer orthogonal
    1.3870474292323159e-15

    Example where r0 and rd are not 1:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> base, variation = orth.orthogonal_representations(x) # Compute orthogonal representations
    >>> tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    >>> tucker_vars, tt_vars = variation
    >>> (U0,U1,U2) = tucker_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (O0,O1,O2) = outer_tt_cores
    >>> (V0,V1,V2) = tucker_vars
    >>> (H0,H1,H2) = tt_vars
    >>> x2 = ((U0,U1,U2), (L0,H1,R2)) # representation with TT-core variation in index 1
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Still represents origional tensor
    2.5341562994067855e-12
    >>> x3 = ((V0,U1,U2), (O0,R1,R2)) # representation with tucker core variation in index 0
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x3))) # Still represents origional tensor
    2.9206090606788446e-12
    >>> print(np.linalg.norm(U0 @ U0.T - np.eye(U0.shape[0]))) # U: orthogonal
    1.675264510304594e-15
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', L0, L0) - np.eye(L0.shape[2]))) # L: left orthogonal
    9.046146325204653e-16
    >>> print(np.linalg.norm(np.einsum('iaj,kaj->ik', R2, R2) - np.eye(R2.shape[0]))) # R: right orthogonal
    1.1775693440128312e-16
    >>> print(np.linalg.norm(np.einsum('iaj,ibj->ab', O0, O0) - np.eye(O0.shape[1]))) # O: outer orthogonal
    1.2300840868850519e-15
    '''
    if not already_left_orthogonal:
        # Orthogonalize Tucker cores upward to get up_tt_cores U
        def _up_func(Bio_Gaib):
            Bio, Gaib = Bio_Gaib
            Boi = Bio.T

            Uox, ssx, WTxi = t3toolbox.linalg.truncated_svd(Boi, xnp=xnp)
            Rxi = xnp.einsum('x,xi->xi', ssx, WTxi)

            new_Gaxb = xnp.einsum('aib,xi->axb', Gaib, Rxi)
            new_Uxo = Uox.T
            return (new_Uxo, new_Gaxb)

        up_tucker_cores, tt_cores = map(_up_func, x)

        # Sweep left-to-right, generating left orthogonal tt_cores L
        def _left_func(Cxb, Gbjc_tuple):
            Gbjc = Gbjc_tuple[0]
            Hxjc = xnp.einsum('xb,bjc->xjc', Cxb, Gbjc)

            Lxjy, ssy, VTyc = linalg.left_svd_3tensor(Hxjc, xnp=xnp)
            Ryc = ssy.reshape((-1, 1)) * VTyc
            return Ryc, [Lxjy]

        C0 = xnp.eye(tt_cores[0].shape[0])
        Cf, (LL,) = scan(_left_func, C0, [tt_cores[:-1]])
        Lf = xnp.einsum('xb,bjc->xjc', Cf, tt_cores[-1])
        left_tt_cores = tuple(LL) + (Lf,)
    else:
        up_tucker_cores, left_tt_cores = x

    # Sweep right-to-left, generating tt_variations H, and right orthogonal tt_cores R
    def _right_func(Ccx, Gbjc_tuple):
        Gbjc = Gbjc_tuple[0]
        Hbjx = xnp.einsum('bjc,cx->bjx', Gbjc, Ccx)

        Wby, ssy, Ryjx = linalg.right_svd_3tensor(Hbjx, xnp=xnp)
        Cby = Wby * ssy.reshape((1, -1))
        return Cby, (Ryjx, Hbjx)

    Cf = xnp.eye(left_tt_cores[-1].shape[2])

    res = scan(_right_func, Cf, (left_tt_cores[1:][::-1],))
    C0, res1 = res
    (RR_rev, HH_rev) = res1

    R0 = xnp.einsum('bjc,cx->bjx', left_tt_cores[0], C0)
    right_tt_cores = (R0,) + tuple(RR_rev[::-1])
    tt_variations = (R0,) + tuple(HH_rev[::-1])

    # Orthogonalize TT cores downward to get outer_tt_cores O and tucker_variations V
    def _down_func(Uio_Haib):
        Uio, Haib,  = Uio_Haib
        Oaxb, ssx, WTxi = linalg.outer_svd_3tensor(Haib, xnp=xnp)
        Cxi = ssx.reshape((-1, 1)) * WTxi

        Vxo = np.einsum('xi,io->xo', Cxi, Uio)
        return (Vxo, Oaxb)

    tucker_variations, outer_tt_cores = map(_down_func, (up_tucker_cores, tt_variations))

    base = (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    variation = (tucker_variations, tt_variations)
    return base, variation

