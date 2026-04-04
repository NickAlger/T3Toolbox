# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ

try:
    import jax.numpy as jnp
except:
    print('jax import failed in tucker_tensor_train. Defaulting to numpy.')
    jnp = np

from t3tools.tucker_tensor_train import *

__all__ = [
    'T3Orthogonals',
    't3_svd',
    'truncated_svd',
    'left_svd_3tensor',
    'right_svd_3tensor',
    'outer_svd_3tensor',
    'up_svd_ith_basis_core',
    'left_svd_ith_tt_core',
    'right_svd_ith_tt_core',
    'up_svd_ith_tt_core',
    'down_svd_ith_tt_core',
    'orthogonalize_relative_to_ith_basis_core',
    'orthogonalize_relative_to_ith_tt_core',
    'tucker_svd_dense',
    'tt_svd_dense',
    't3_svd_dense',
]


T3Orthogonals = typ.Tuple[
    typ.Sequence[NDArray],  # orthogonal_basis_cores. B_xo B_yo = I_xy    B.shape = (n, N)
    typ.Sequence[NDArray],  # left_orthogonal_tt_cores. P_iax P_iay = I_xy, P.shape = (rL, n, rR)
    typ.Sequence[NDArray],  # right_orthogonal_tt_cores. Q_xaj Q_yaj = I_xy  Q.shape = (rL, n, rR)
    typ.Sequence[NDArray],  # outer_orthogonal_tt_cores. R_ixj R_iyj = I_xy  R.shape = (rL, n, rR)
]


###############################################
########    SVD of core unfoldings    #########
###############################################

def truncated_svd(
        A: NDArray, # shape=(N,M)
        min_rank: int = None,  # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None,  # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None,  # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None,  # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # U, shape=(N,k)
    NDArray, # ss, shape=(k,)
    NDArray, # Vt, shape=(k,M)
]:
    '''Compute (truncated) singular value decomposition of matrix.

    A = U @ diag(ss) @ Vt
    Equality may be approximate if truncation is used.

    Parameters
    ----------
    A: NDArray
        Matrix. shape=(N, M)
    min_rank: int
        Minimum rank for truncation. Should have 1 <= min_rank <= max_rank <= minimum(N, M).
    min_rank: int
        Maximum rank for truncation. Should have 1 <= min_rank <= max_rank <= minimum(N, M).
    rtol: float
        Relative tolerance for truncation. Remove singular values satisfying sigma < maximum(atol, rtol*sigma1).
    atol: float
        Absolute tolerance for truncation. Remove singular values satisfying sigma < maximum(atol, rtol*sigma1).
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    U: NDArray
        Left singular vectors. shape=(N, k).
        U.T @ U = identity matrix
    ss: NDArray
        Singular values. Non-negative. shape=(k,).
    Vt: NDArray
        Right singular vectors. shape=(k, M)
        Vt @ Vt.T = identity matrix

    See Also
    --------
    left_svd_3tensor
    right_svd_3tensor
    outer_svd_3tensor
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> A = np.random.randn(55,70)
    >>> U, ss, Vt = truncated_svd(A)
    >>> A2 = np.einsum('ix,x,xj->ij', U, ss, Vt)
    >>> print(np.linalg.norm(A - A2))
        1.0428742517412705e-13
    >>> rank = len(ss)
    >>> print(np.linalg.norm(U.T @ U - np.eye(rank)))
        1.1907994177245428e-14
    >>> print(np.linalg.norm(Vt @ Vt.T - np.eye(rank)))
        1.1027751835566194e-14

    >>> A = np.random.randn(55, 70) @ np.diag(1.0 / np.arange(1,71)**2) # Create matrix with spectral decay
    >>> U, ss, Vt = truncated_svd(A, rtol=1e-2)
    >>> A2 = np.einsum('ix,x,xj->ij', U, ss, Vt)
    >>> truncated_rank = len(ss)
    >>> print(truncated_rank)
        10
    >>> relerr_num = np.linalg.norm(A - A2, 2) # Check error in induced 2-norm
    >>> relerr_den = np.linalg.norm(A, 2)
    >>> print(relerr_num / relerr_den) # should be just less than rtol=1e-2
        0.008530627920514714

    >>> A = np.random.randn(55, 70) @ np.diag(1.0 / np.arange(1,71)**2) # Create matrix with spectral decay
    >>> U, ss, Vt = truncated_svd(A, atol=1e-2)
    >>> A2 = np.einsum('ix,x,xj->ij', U, ss, Vt)
    >>> truncated_rank = len(ss)
    >>> print(truncated_rank)
        24
    >>> err = np.linalg.norm(A - A2, 2)  # Check error in induced 2-norm
    >>> print(err) # should be just less than atol=1e-2
        0.00882416786402483
    '''
    xnp = jnp if use_jax else np

    rtol1 = 0.0 if rtol is None else rtol
    atol1 = 0.0 if atol is None else atol

    N, M = A.shape

    U0, ss0, Vt0 = xnp.linalg.svd(A, full_matrices=False)

    tol = xnp.maximum(ss0[0] * rtol1, atol1)

    min_possible_rank = 1
    max_possible_rank = xnp.minimum(N, M)

    min_rank = min_possible_rank if min_rank is None else xnp.maximum(min_rank, min_possible_rank)
    max_rank = max_possible_rank if max_rank is None else xnp.minimum(max_rank, max_possible_rank)

    num_significant_sigmas = xnp.sum(ss0 >= tol)
    nx = xnp.maximum(xnp.minimum(num_significant_sigmas, max_rank), min_rank)

    U = U0[:, :nx]
    ss = ss0[:nx]
    Vt = Vt0[:nx, :]

    return U, ss, Vt


def left_svd_3tensor(
        G0_i_a_j: NDArray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # U_i_a_x, shape=(ni, na, nx)
    NDArray, # ss_x,    shape=(nx,)
    NDArray, # Vt_x_j,  shape=(nx, nj)
]:
    '''Compute (truncated) singular value decomposition of 3-tensor left unfolding.

    First two indices of the tensor are grouped for the SVD.

    G0_i_a_j = einsum('iax,x,xj->ixj', U_i_a_x, ss_x, Vt_x_j).
    Equality may be approximate if truncation is used.

    Parameters
    ----------
    G0_i_a_j: NDArray
        3-tensor. shape=(ni, na, nj)
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    U_i_a_x: NDArray
        Left singular vectors, reshaped into 3-tensor. shape=(ni, na, nx).
        einsum('iax,iay->xy', U_i_a_x, U_i_a_x) = identity matrix
    ss_x: NDArray
        Singular values. Non-negative. shape=(nx,).
    Vt_x_j: NDArray
        Right singular vectors. shape=(nx, nj)
        einsum('xj,yj->xy', Vt_x_j, Vt_x_j) = identity matrix

    Raises
    ------
    RuntimeError
        Error raised if the provided indices in index are inconsistent with each other or the Tucker tensor train x.

    See Also
    --------
    truncated_svd
    right_svd_3tensor
    outer_svd_3tensor
    left_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_a_x, ss_x, Vt_x_j = left_svd_3tensor(G_i_a_j)
    >>> G_i_a_j2 = np.einsum('iax,x,xj->iaj', U_i_a_x, ss_x, Vt_x_j)
    >>> print(np.linalg.norm(G_i_a_j - G_i_a_j2))
        1.8290510387826402e-14
    >>> rank = len(ss_x)
    >>> print(np.linalg.norm(np.einsum('iax,iay->xy', U_i_a_x, U_i_a_x) - np.eye(rank)))
        1.6194412284045956e-15
    >>> print(np.linalg.norm(np.einsum('xj,yj->xy', Vt_x_j, Vt_x_j) - np.eye(rank)))
        1.4738004835812172e-15
    '''
    ni, na, nj = G0_i_a_j.shape
    G0_ia_j = G0_i_a_j.reshape((ni*na, nj))

    U_ia_x, ss_x, Vt_x_j = truncated_svd(G0_ia_j, min_rank, max_rank, rtol, atol, use_jax)

    nx = len(ss_x)
    U_i_a_x = U_ia_x.reshape((ni, na, nx))
    return U_i_a_x, ss_x, Vt_x_j


def right_svd_3tensor(
        G0_i_a_j: NDArray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # U_i_x,       shape=(ni, nx)
    NDArray, # ss_x,        shape=(nx,)
    NDArray, # Vt_x_a_j,    shape=(nx, na, nj)
]:
    '''Compute (truncated) singular value decomposition of 3-tensor right unfolding.

    Last two indices of the tensor are grouped for the SVD.

    G0_i_a_j = einsum('iax,x,xj->ixj', U_i_x, ss_x, Vt_x_a_j).
    Equality may be approximate if truncation is used.

    Parameters
    ----------
    G0_i_a_j: NDArray
        3-tensor. shape=(ni, na, nj)
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    U_i_x: NDArray
        Left singular vectors. shape=(ni, nx).
        einsum('ix,iy->xy', U_i_x, U_i_x) = identity matrix
    ss_x: NDArray
        Singular values. Non-negative. shape=(nx,).
    Vt_x_a_j: NDArray
        Right singular vectors, reshaped into 3-tensor. shape=(nx, na, nj)
        einsum('xaj,yaj->xy', Vt_x_a_j, Vt_x_a_j) = identity matrix

    Raises
    ------
    RuntimeError
        Error raised if the provided indices in index are inconsistent with each other or the Tucker tensor train x.

    See Also
    --------
    truncated_svd
    left_svd_3tensor
    outer_svd_3tensor
    right_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_x, ss_x, Vt_x_a_j = right_svd_3tensor(G_i_a_j)
    >>> G_i_a_j2 = np.einsum('ix,x,xaj->iaj', U_i_x, ss_x, Vt_x_a_j)
    >>> print(np.linalg.norm(G_i_a_j - G_i_a_j2))
        1.2503321403334437e-14
    >>> rank = len(ss_x)
    >>> print(np.linalg.norm(np.einsum('ix,iy->xy', U_i_x, U_i_x) - np.eye(rank)))
        1.6591938592301729e-15
    >>> print(np.linalg.norm(np.einsum('xaj,yaj->xy', Vt_x_a_j, Vt_x_a_j) - np.eye(rank)))
        1.9466202162000267e-15
    '''
    G0_j_a_i = G0_i_a_j.swapaxes(0, 2)
    Vt_j_a_x, ss_x, U_x_i = left_svd_3tensor(G0_j_a_i, min_rank, max_rank, rtol, atol, use_jax)
    Vt_x_a_j = Vt_j_a_x.swapaxes(0, 2)
    U_i_x = U_x_i.swapaxes(0,1)
    return U_i_x, ss_x, Vt_x_a_j


def outer_svd_3tensor(
        G0_i_a_j: NDArray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # U_i_x_j, shape=(ni, nx, nj),
    NDArray, # ss_x,    shape=(nx,)
    NDArray, # Vt_x_a,  shape=(nx, na)
]:
    '''Compute (truncated) singular value decomposition of 3-tensor outer unfolding.

    First and last indices of the tensor are grouped to form rows for the SVD.
    Middle index forms columns.

    G0_i_a_j = einsum('iax,x,xj->ixj', U_i_x_j, ss_x, Vt_x_a).
    Equality may be approximate if truncation is used.

    Parameters
    ----------
    G0_i_a_j: NDArray
        3-tensor. shape=(ni, na, nj)
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    U_i_x_j: NDArray
        Left singular vectors. shape=(ni, nx, nj).
        einsum('ixj,iyj->xy', U_i_x_j, U_i_x_j) = identity matrix
    ss_x: NDArray
        Singular values. Non-negative. shape=(nx,).
    Vt_x_a: NDArray
        Right singular vectors. shape=(nx, na)
        einsum('xa,ya->xy', Vt_x_a, Vt_x_a) = identity matrix

    Raises
    ------
    RuntimeError
        Error raised if the provided indices in index are inconsistent with each other or the Tucker tensor train x.

    See Also
    --------
    truncated_svd
    left_svd_3tensor
    right_svd_3tensor
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_x_j, ss_x, Vt_x_a = outer_svd_3tensor(G_i_a_j)
    >>> G_i_a_j2 = np.einsum('ixj,x,xa->iaj', U_i_x_j, ss_x, Vt_x_a)
    >>> print(np.linalg.norm(G_i_a_j - G_i_a_j2))
        1.4102138928233928e-14
    >>> rank = len(ss_x)
    >>> print(np.linalg.norm(np.einsum('ixj,iyj->xy', U_i_x_j, U_i_x_j) - np.eye(rank)))
        3.3426764835898436e-15
    >>> print(np.linalg.norm(np.einsum('xa,ya->xy', Vt_x_a, Vt_x_a) - np.eye(rank)))
        1.8969691003092744e-15
    '''
    G0_i_j_a = G0_i_a_j.swapaxes(1, 2)
    U_i_j_x, ss_x, Vt_x_a = left_svd_3tensor(G0_i_j_a, min_rank, max_rank, rtol, atol, use_jax)
    U_i_x_j = U_i_j_x.swapaxes(1, 2)
    return U_i_x_j, ss_x, Vt_x_a


##########################################
########    Orthogonalization    #########
##########################################

def up_svd_ith_basis_core(
        ii: int, # which base core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # ss_x. singular values
]:
    '''Compute SVD of ith basis core and contract non-orthogonal factor up into the TT-core above.

    Parameters
    ----------
    ii: int
        index of basis core to SVD
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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith basis core orthogonal.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_basis_cores[ii].shape = (new_ni, Ni)
        new_basis_cores[ii] @ new_basis_cores[ii].T = identity matrix
    ss_x: NDArray
        Singular values of prior ith basis core. shape=(new_ni,).

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
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> ind = 1
    >>> x2, ss = up_svd_ith_basis_core(ind, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        5.772851635866132e-13
    >>> basis_cores2, tt_cores2 = x2
    >>> rank = len(ss)
    >>> B = basis_cores2[ind]
    >>> print(np.linalg.norm(B @ B.T - np.eye(rank)))
        8.456498415401757e-16
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    G_a_i_b = tt_cores[ii]
    U_i_o = basis_cores[ii]
    U_o_i = U_i_o.T

    U2_o_x, ss_x, Vt_x_i = truncated_svd(U_o_i, min_rank, max_rank, rtol, atol, use_jax)
    R_x_i = xnp.einsum('x,xi->xi', ss_x, Vt_x_i)
    # U2_o_x, R_x_i = xnp.linalg.qr(U_o_i, mode='reduced')

    G2_a_x_b = xnp.einsum('aib,xi->axb', G_a_i_b, R_x_i)
    U2_x_o = U2_o_x.T

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G2_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = U2_x_o

    new_x = (tuple(new_basis_cores), tuple(new_tt_cores))

    return new_x, ss_x


def left_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

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
    up_svd_ith_basis_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> ind = 1
    >>> x2, ss = left_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        5.186463661974644e-13
    >>> basis_cores2, tt_cores2 = x2
    >>> rank = len(ss)
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(jnp.einsum('iaj,iak->jk', G, G) - np.eye(rank)))
        4.453244025338311e-16
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii]
    B0_b_j_c = tt_cores[ii+1]

    A_a_i_x, ss_x, Vt_x_b = left_svd_3tensor(A0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)
    B_x_j_c = xnp.tensordot(ss_x.reshape((-1,1)) * Vt_x_b, B0_b_j_c, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = A_a_i_x
    new_tt_cores[ii+1] = B_x_j_c

    return (tuple(basis_cores), tuple(new_tt_cores)), ss_x


def right_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

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
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> ind = 1
    >>> x2, ss = right_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        5.304678679078675e-13
    >>> basis_cores2, tt_cores2 = x2
    >>> rank = len(ss)
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(jnp.einsum('iaj,kaj->ik', G, G) - np.eye(rank)))
        4.207841813173725e-16
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii-1]
    B0_b_j_c = tt_cores[ii]

    U_b_x, ss_x, B_x_j_c = right_svd_3tensor(B0_b_j_c, min_rank, max_rank, rtol, atol, use_jax)
    A_a_i_x = xnp.tensordot(A0_a_i_b, U_b_x * ss_x.reshape((1,-1)), axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii-1] = A_a_i_x
    new_tt_cores[ii] = B_x_j_c

    return (tuple(basis_cores), tuple(new_tt_cores)), ss_x


def up_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_basis_cores[ii].shape = (new_ni, Ni)
    ss_x: NDArray
        Singular values of prior ith TT-core outer unfolding. shape=(new_ri,).

    See Also
    --------
    truncated_svd
    outer_svd_3tensor
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> x2, ss = up_svd_ith_tt_core(1, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        1.002901486286745e-12
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = basis_cores[ii]

    U_a_x_b, ss_x, Vt_x_i = outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)

    G_a_x_b = xnp.einsum('axb,x->axb', U_a_x_b, ss_x)
    Q_x_o = xnp.tensordot(Vt_x_i, Q0_i_o, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = Q_x_o

    return (tuple(new_basis_cores), tuple(new_tt_cores)), ss_x


def down_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(new_ni,)
]:
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the basis core below.

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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith TT-core outer orthogonal.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_basis_cores[ii].shape = (new_ni, Ni)
        einsum('iaj,ibj->ab', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
    ss_x: NDArray
        Singular values of prior ith TT-core outer unfolding. shape=(new_ni,).

    See Also
    --------
    truncated_svd
    outer_svd_3tensor
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> ind = 1
    >>> x2, ss = down_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        4.367311712704942e-12
    >>> basis_cores2, tt_cores2 = x2
    >>> rank = len(ss)
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(jnp.einsum('iaj,ibj->ab', G, G) - np.eye(rank)))
        1.0643458053135608e-15
    '''
    basis_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = basis_cores[ii]

    G_a_x_b, ss_x, Vt_x_i = outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)

    Q_x_o = (ss_x.reshape((-1,1)) * Vt_x_i) @ Q0_i_o

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = Q_x_o

    return (tuple(new_basis_cores), tuple(new_tt_cores)), ss_x


def orthogonalize_relative_to_ith_basis_core(
        ii: int,
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith basis core.

    Orthogonal is done relative to the ith basis core:
        - ith basis core is not orthogonalized
        - All other basis cores are orthogonalized.
        - TT-cores to the left are left orthogonalized.
        - TT-core directly above is outer orthogonalized.
        - TT-cores to the right are right orthogonalized.

    Parameters
    ----------
    ii: int
        index of basis core that is not orthogonalized
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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith basis core.

    See Also
    --------
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> x2 = orthogonalize_relative_to_ith_basis_core(1, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        8.800032152216517e-13
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> X = jnp.einsum('xi,axb,byc,czd,zk->iyk', B0, G0, G1, G2, B2) # Contraction of everything except B1
    >>> print(np.linalg.norm(jnp.einsum('iyk,iwk->yw', X, X) - np.eye(B1.shape[0])))
        1.7116160385376214e-15
    '''
    shape, tucker_ranks, tt_ranks = t3_structure(x)

    new_x = x
    for jj in range(ii):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = left_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    for jj in range(len(shape)-1, ii, -1):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = right_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    new_x = down_svd_ith_tt_core(ii, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
    return new_x


def orthogonalize_relative_to_ith_tt_core(
        ii: int,
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith TT-core.

    Orthogonal is done relative to the ith TT-core:
        - All basis cores are orthogonalized.
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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    See Also
    --------
    up_svd_ith_basis_core
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
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> x2 = orthogonalize_relative_to_ith_tt_core(1, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        8.800032152216517e-13
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> XL = np.einsum('axb,xi -> aib', G0, B0) # Everything to the left of G1
    >>> print(np.linalg.norm(np.einsum('aib,aic->bc', XL, XL) - np.eye(G1.shape[0])))
        9.820411604510197e-16
    >>> print(np.linalg.norm(np.einsum('xi,yi->xy', B1, B1) - np.eye(G1.shape[1]))) # Everything below G1
        2.1875310121178e-15
    >>> XR = np.einsum('axb,xi->aib', G2, B2) # Everything to the right of G1
    >>> print(np.linalg.norm(np.einsum('aib,cib->ac', XR, XR) - np.eye(G1.shape[2])))
        1.180550381921849e-15
    '''
    shape, tucker_ranks, tt_ranks = t3_structure(x)

    new_x = x
    for jj in range(ii):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = left_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    for jj in range(len(shape)-1, ii, -1):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = right_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    new_x = up_svd_ith_basis_core(ii, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
    return new_x


# def t3_orthogonal_representations(
#         x: TuckerTensorTrain,
#         use_jax: bool = False,
# ) -> typ.Tuple[
#     TuckerTensorTrain, # non-orthogonal cores
#     T3Orthogonals, # orthogonalizations of x
# ]:
#     xnp = jnp if use_jax else np
#
#     basis_cores, tt_cores = x
#
#     num_cores = len(tt_cores)
#
#     # Orthogonalize basis matrices
#     for ii in range(num_cores):
#         x = up_svd_ith_basis_core(ii, x, use_jax=use_jax)[0]
#
#     # Right orthogonalize
#     for ii in range(num_cores-1, 0, -1): # num_cores-1, num_cores-2, ..., 1
#         x = right_svd_ith_tt_core(ii, x, use_jax=use_jax)[0]
#
#     non_orthogonal_basis_cores = []
#     non_orthogonal_tt_cores = []
#
#     orthogonal_basis_cores = []
#     left_orthogonal_tt_cores = []
#     right_orthogonal_tt_cores = []
#     outer_orthogonal_tt_cores = []
#     # Sweep left to right
#     for ii in range(num_cores):
#         # SVD inbetween tt core and basis core
#         x, ss_basis = up_svd_ith_tt_core(
#             ii, x, use_jax=use_jax,
#         )
#         all_ss_basis.append(ss_basis)
#
#         if ii < num_cores-1:
#             # SVD inbetween ith tt core and (i+1)th tt core
#             x, ss_tt = left_svd_ith_tt_core(
#                 ii, x, use_jax=use_jax,
#             )
#         else:
#             Gf = x[1][-1]
#             _, ss_tt, _ = left_svd_3tensor(Gf, use_jax=use_jax)
#         all_ss_tt.append(ss_tt)
#
#     return x, tuple(all_ss_basis), tuple(all_ss_tt)






###############################
########    T3-SVD    #########
###############################

def t3_svd(
        x: TuckerTensorTrain,
        min_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        min_tucker_ranks:   typ.Sequence[int] = None,  # len=d
        max_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        max_tucker_ranks:   typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,

) -> typ.Tuple[
    TuckerTensorTrain, # new_x
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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

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
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,6), randn(5,7), randn(6,8))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> x2, ss_basis, ss_tt = t3_svd(x)
    >>> x_dense = t3_to_dense(x)
    >>> x2_dense = t3_to_dense(x2)
    >>> print(np.linalg.norm(x_dense - x2_dense))
        7.556835759880194e-13
    >>> x_dense = t3_to_dense(x)
    >>> ss_tt0 = np.linalg.svd(x_dense.reshape((1, 6*7*8)))[1]
    >>> print(ss_tt0); print(ss_tt[0])
        [405.41453572]
        [405.41453572]
    >>> ss_tt1 = np.linalg.svd(x_dense.reshape((6, 7*8)))[1]
    >>> print(ss_tt1); print(ss_tt[1])
        [3.26096778e+02 2.34056249e+02 5.69166861e+01 2.52531568e-14 1.72986412e-14 8.25218909e-15]
        [326.0967784  234.05624908  56.91668613]
    >>> ss_tt2 = np.linalg.svd(x_dense.reshape((6*7, 8)))[1]
    >>> print(ss_tt2); print(ss_tt[2])
        [3.92785730e+02 1.00400775e+02 3.88846558e-14 1.37176914e-14 5.89995607e-15 4.96667173e-15 3.77344519e-15 3.12383125e-15]
        [392.78573046 100.40077549]
    >>> ss_tt3 = np.linalg.svd(x_dense.reshape((6*7*8,1)))[1]
    >>> print(ss_tt3); print(ss_tt[3])
        [405.41453572]
        [405.41453572]
    >>> ss_basis0 = np.linalg.svd(x_dense.transpose([0,1,2]).reshape((6,7*8)))[1]
    >>> print(ss_basis0); print(ss_basis[0])
        [3.26096778e+02 2.34056249e+02 5.69166861e+01 2.52531568e-14 1.72986412e-14 8.25218909e-15]
        [326.0967784  234.05624908  56.91668613]
    >>> ss_basis1 = np.linalg.svd(x_dense.transpose([1,0,2]).reshape((7,6*8)))[1]
    >>> print(ss_basis1); print(ss_basis[1])
        [3.19638822e+02 2.37070453e+02 6.46637749e+01 4.21190167e+01 5.84413285e+00 2.08773511e-14 1.15080732e-14]
        [319.63882212 237.07045349  64.66377495  42.1190167    5.84413285]
    >>> ss_basis2 = np.linalg.svd(x_dense.transpose([2,0,1]).reshape((8,6*7)))[1]
    >>> print(ss_basis2); print(ss_basis[2])
        [3.92785730e+02 1.00400775e+02 5.17611232e-14 1.35674292e-14 7.94410140e-15 7.05351536e-15 5.31534353e-15 2.69976806e-15]
        [392.78573046 100.40077549]

    >>> B0 = randn(35,40) @ np.diag(1.0 / np.arange(1, 41)**2)
    >>> B1 = randn(45,50) @ np.diag(1.0 / np.arange(1, 51)**2)
    >>> B2 = randn(55,60) @ np.diag(1.0 / np.arange(1, 61)**2)
    >>> basis_cores_x = (B0, B1, B2)
    >>> tt_cores_x = (randn(1,35,30), randn(30,45,40), randn(40,55,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> x2, ss_basis, ss_tt = t3_svd(x, rtol=1e-2)
    >>> print(t3_structure(x))
        ((40, 50, 60), (35, 45, 55), (1, 30, 40, 1))
    >>> print(t3_structure(x2))
        ((40, 50, 60), (6, 6, 5), (1, 6, 5, 1))
    >>> x_dense = t3_to_dense(x)
    >>> x2_dense = t3_to_dense(x2)
    >>> relerr_num = np.linalg.norm(x_dense - x2_dense)
    >>> relerr_den = np.linalg.norm(x_dense)
    >>> print(relerr_num / relerr_den) # Should be around or slightly less than 3*rtol = 3e-3
        0.013078458673911168

    >>> basis_cores_x = (randn(10,14), randn(11,15), randn(12,16))
    >>> tt_cores_x = (randn(1,10,8), randn(8,11,9), randn(9,12,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> x2, ss_basis, ss_tt = t3_svd(x, max_tucker_ranks=(3,3,3), max_tt_ranks=(1,2,2,1))
    >>> print(t3_structure(x))
        ((14, 15, 16), (10, 11, 12), (1, 8, 9, 1))
    >>> print(t3_structure(x2))
        ((14, 15, 16), (3, 3, 2), (1, 2, 2, 1))
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    num_cores = len(tt_cores)

    # Orthogonalize basis matrices
    for ii in range(num_cores):
        x, _ = up_svd_ith_basis_core(ii, x, use_jax=use_jax)

    # Right orthogonalize
    for ii in range(num_cores-1, 0, -1): # num_cores-1, num_cores-2, ..., 1
        x, _ = right_svd_ith_tt_core(ii, x, use_jax=use_jax)

    G0 = x[1][0]
    _, ss_first, _ = right_svd_3tensor(G0, use_jax=use_jax)

    # Sweep left to right computing SVDS
    all_ss_basis = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        min_rank = min_tucker_ranks[ii] if min_tucker_ranks is not None else None
        max_rank = max_tucker_ranks[ii] if max_tucker_ranks is not None else None
        # SVD inbetween tt core and basis core
        x, ss_basis = up_svd_ith_tt_core(
            ii, x, min_rank, max_rank, rtol, atol, use_jax,
        )
        all_ss_basis.append(ss_basis)

        if ii < num_cores-1:
            min_rank = min_tt_ranks[ii+1] if min_tt_ranks is not None else None
            max_rank = max_tt_ranks[ii+1] if max_tt_ranks is not None else None
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = left_svd_ith_tt_core(
                ii, x, min_rank, max_rank, rtol, atol, use_jax,
            )
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = left_svd_3tensor(Gf, use_jax=use_jax)
        all_ss_tt.append(ss_tt)

    return x, tuple(all_ss_basis), tuple(all_ss_tt)


#

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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

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
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> T0 = np.random.randn(40, 50, 60)
    >>> c0 = 1.0 / np.arange(1, 41)**2
    >>> c1 = 1.0 / np.arange(1, 51)**2
    >>> c2 = 1.0 / np.arange(1, 61)**2
    >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2)
    >>> (bases, core), ss = tucker_svd_dense(T, rtol=1e-2)
    >>> print(core.shape)
        (3, 4, 3)
    >>> T2 = jnp.einsum('abc, ai,bj,ck->ijk', core, bases[0], bases[1], bases[2])
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-2
        0.016764601091053873
    >>> (bases, core), ss = tucker_svd_dense(T, rtol=1e-3)
    >>> print(core.shape)
        (9, 9, 9)
    >>> T2 = jnp.einsum('abc, ai,bj,ck->ijk', core, bases[0], bases[1], bases[2])
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
        0.002418671417862558
    >>> (bases, core), ss = tucker_svd_dense(T, rtol=1e-4)
    >>> print(core.shape)
        (20, 21, 21)
    >>> T2 = jnp.einsum('abc, ai,bj,ck->ijk', core, bases[0], bases[1], bases[2])
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-4
        0.0003418353245534034
    '''
    xnp = jnp if use_jax else np

    bases = []
    singular_values_of_matricizations = []
    C = T
    for ii in range(len(T.shape)):
        C_swap = C.swapaxes(ii,0)
        old_shape_swap = C_swap.shape

        min_rank = None if min_ranks is None else min_ranks[ii]
        max_rank = None if max_ranks is None else max_ranks[ii]

        C_swap_mat = C_swap.reshape((old_shape_swap[0], -1))
        U, ss, Vt = truncated_svd(C_swap_mat, min_rank, max_rank, rtol, atol, use_jax)
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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

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
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> T0 = np.random.randn(40, 50, 60)
    >>> c0 = 1.0 / np.arange(1, 41)**2
    >>> c1 = 1.0 / np.arange(1, 51)**2
    >>> c2 = 1.0 / np.arange(1, 61)**2
    >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2)
    >>> cores, ss = tt_svd_dense(T, rtol=1e-2)
    >>> print([G.shape for G in cores])
        [(1, 40, 6), (6, 50, 7), (7, 60, 1)]
    >>> T2 = jnp.einsum('aib,bjc,ckd->ijk', cores[0], cores[1], cores[2])
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-2
        0.013056156368977757
    >>> cores, ss = tt_svd_dense(T, rtol=1e-3)
    >>> print([G.shape for G in cores])
        [(1, 40, 13), (13, 50, 13), (13, 60, 1)]
    >>> T2 = jnp.einsum('aib,bjc,ckd->ijk', cores[0], cores[1], cores[2])
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
        0.0023999063535883633
    >>> cores, ss = tt_svd_dense(T, rtol=1e-4)
    >>> print([G.shape for G in cores])
        [(1, 40, 30), (30, 50, 30), (30, 60, 1)]
    >>> T2 = jnp.einsum('aib,bjc,ckd->ijk', cores[0], cores[1], cores[2])
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-4
        0.0002850622316036925
    '''
    nn = T.shape

    X = T.reshape((1,) + T.shape)
    singular_values_of_unfoldings = []
    cores = []
    for ii in range(len(nn)-1):
        rL = X.shape[0]

        min_rank = None if min_ranks is None else min_ranks[ii+1]
        max_rank = None if max_ranks is None else max_ranks[ii+1]

        U, ss, Vt = truncated_svd(X.reshape((rL * nn[ii], -1)), min_rank, max_rank, rtol, atol, use_jax)
        rR = len(ss)

        singular_values_of_unfoldings.append(ss)
        cores.append(U.reshape((rL, nn[ii], rR)))
        X = ss.reshape((-1,1)) * Vt
    cores.append(X.reshape(X.shape + (1,)))

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
    TuckerTensorTrain, # new_x
    typ.Tuple[NDArray,...], # basis singular values, len=k
    typ.Tuple[NDArray,...], # tt singular values, len=k-1
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
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

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
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> T0 = np.random.randn(40, 50, 60)
    >>> c0 = 1.0 / np.arange(1, 41)**2
    >>> c1 = 1.0 / np.arange(1, 51)**2
    >>> c2 = 1.0 / np.arange(1, 61)**2
    >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2)
    >>> x, ss_tucker, ss_tt = t3_svd_dense(T, rtol=1e-2)
    >>> print(t3_structure(x))
        ((40, 50, 60), (5, 5, 3), (1, 5, 3, 1))
    >>> T2 = t3_to_dense(x)
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-2
        0.018856713575257946
    >>> x, ss_tucker, ss_tt = t3_svd_dense(T, rtol=1e-3)
    >>> print(t3_structure(x))
        ((40, 50, 60), (12, 11, 12), (1, 12, 12, 1))
    >>> T2 = t3_to_dense(x)
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
        0.0025147026955504846
    >>> x, ss_tucker, ss_tt = t3_svd_dense(T, rtol=1e-4)
    >>> print(t3_structure(x))
        ((40, 50, 60), (26, 26, 27), (1, 26, 27, 1))
    >>> T2 = t3_to_dense(x)
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-4
        0.00034874423325066196
    '''
    (basis_cores, tucker_core), ss_tucker = tucker_svd_dense(T, min_tucker_ranks, max_tucker_ranks, rtol, atol, use_jax)
    tt_cores, ss_tt = tt_svd_dense(tucker_core, min_tt_ranks, max_tt_ranks, rtol, atol, use_jax)
    return (basis_cores, tt_cores), ss_tucker, ss_tt

