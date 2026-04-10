# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import typing as typ
import numpy as np

import t3tools
from t3tools.common import NDArray


__all__ = [
    'truncated_svd',
    'left_svd_3tensor',
    'right_svd_3tensor',
    'outer_svd_3tensor',
]

######################################
########    Truncated SVD    #########
######################################

def truncated_svd(
        A: NDArray, # shape=(N,M)
        min_rank: int = None,  # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None,  # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None,  # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None,  # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        xnp = np,
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
    xnp:
        Linear algebra backend. Default: np (numpy)

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

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.dense as dense
    >>> A = np.random.randn(55,70)
    >>> U, ss, Vt = dense.truncated_svd(A)
    >>> A2 = np.einsum('ix,x,xj->ij', U, ss, Vt)
    >>> print(np.linalg.norm(A - A2))
    1.0428742517412705e-13
    >>> rank = len(ss)
    >>> print(np.linalg.norm(U.T @ U - np.eye(rank)))
    1.1907994177245428e-14
    >>> print(np.linalg.norm(Vt @ Vt.T - np.eye(rank)))
    1.1027751835566194e-14

    >>> import numpy as np
    >>> import t3tools.dense as dense
    >>> A = np.random.randn(55, 70) @ np.diag(1.0 / np.arange(1,71)**2) # Create matrix with spectral decay
    >>> U, ss, Vt = dense.truncated_svd(A, rtol=1e-2) # Truncated SVD with relative tolerance 1e-2
    >>> A2 = np.einsum('ix,x,xj->ij', U, ss, Vt)
    >>> truncated_rank = len(ss)
    >>> print(truncated_rank)
    10
    >>> relerr_num = np.linalg.norm(A - A2, 2) # Check error in induced 2-norm
    >>> relerr_den = np.linalg.norm(A, 2)
    >>> print(relerr_num / relerr_den) # should be just less than rtol=1e-2
    0.008530627920514714
    >>> U, ss, Vt = dense.truncated_svd(A, atol=1e-2) # Truncated SVD with absolute tolerance 1e-2
    >>> A2 = np.einsum('ix,x,xj->ij', U, ss, Vt)
    >>> truncated_rank = len(ss)
    >>> print(truncated_rank)
    24
    >>> err = np.linalg.norm(A - A2, 2)  # Check error in induced 2-norm
    >>> print(err) # should be just less than atol=1e-2
    0.00882416786402483
    '''
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
        xnp = np,
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
    xnp:
        Linear algebra backend. Default: np (numpy)

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
    >>> import numpy as np
    >>> import t3tools.orthogonalization as orth
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_a_x, ss_x, Vt_x_j = orth.left_svd_3tensor(G_i_a_j)
    >>> G_i_a_j2 = np.einsum('iax,x,xj->iaj', U_i_a_x, ss_x, Vt_x_j)
    >>> print(np.linalg.norm(G_i_a_j - G_i_a_j2)) # SVD exact to numerical precision
    1.8290510387826402e-14
    >>> rank = len(ss_x)
    >>> print(np.linalg.norm(np.einsum('iax,iay->xy', U_i_a_x, U_i_a_x) - np.eye(rank))) # U is left-orthogonal
    1.6194412284045956e-15
    >>> print(np.linalg.norm(np.einsum('xj,yj->xy', Vt_x_j, Vt_x_j) - np.eye(rank))) # V is orthogonal
    1.4738004835812172e-15
    '''
    ni, na, nj = G0_i_a_j.shape
    G0_ia_j = G0_i_a_j.reshape((ni*na, nj))

    U_ia_x, ss_x, Vt_x_j = t3tools.linalg.truncated_svd(G0_ia_j, min_rank, max_rank, rtol, atol, xnp=xnp)

    nx = len(ss_x)
    U_i_a_x = U_ia_x.reshape((ni, na, nx))
    return U_i_a_x, ss_x, Vt_x_j


def right_svd_3tensor(
        G0_i_a_j: NDArray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        xnp = np,
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
    xnp:
        Linear algebra backend. Default: np (numpy)

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
    >>> import numpy as np
    >>> import t3tools.orthogonalization as orth
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_x, ss_x, Vt_x_a_j = orth.right_svd_3tensor(G_i_a_j)
    >>> G_i_a_j2 = np.einsum('ix,x,xaj->iaj', U_i_x, ss_x, Vt_x_a_j)
    >>> print(np.linalg.norm(G_i_a_j - G_i_a_j2)) # SVD exact to numerical precision
    1.2503321403334437e-14
    >>> rank = len(ss_x)
    >>> print(np.linalg.norm(np.einsum('ix,iy->xy', U_i_x, U_i_x) - np.eye(rank))) # U is orthogonal
    1.6591938592301729e-15
    >>> print(np.linalg.norm(np.einsum('xaj,yaj->xy', Vt_x_a_j, Vt_x_a_j) - np.eye(rank))) # Vt is right-orthogonal
    1.9466202162000267e-15
    '''
    G0_j_a_i = G0_i_a_j.swapaxes(0, 2)
    Vt_j_a_x, ss_x, U_x_i = left_svd_3tensor(G0_j_a_i, min_rank, max_rank, rtol, atol, xnp=xnp)
    Vt_x_a_j = Vt_j_a_x.swapaxes(0, 2)
    U_i_x = U_x_i.swapaxes(0,1)
    return U_i_x, ss_x, Vt_x_a_j


def outer_svd_3tensor(
        G0_i_a_j: NDArray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        xnp = np,
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
    xnp:
        Linear algebra backend. Default: np (numpy)

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
    >>> import numpy as np
    >>> import t3tools.orthogonalization as orth
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_x_j, ss_x, Vt_x_a = orth.outer_svd_3tensor(G_i_a_j)
    >>> G_i_a_j2 = np.einsum('ixj,x,xa->iaj', U_i_x_j, ss_x, Vt_x_a)
    >>> print(np.linalg.norm(G_i_a_j - G_i_a_j2)) # SVD exact to numerical precision
    1.4102138928233928e-14
    >>> rank = len(ss_x)
    >>> print(np.linalg.norm(np.einsum('ixj,iyj->xy', U_i_x_j, U_i_x_j) - np.eye(rank))) # U is outer-orthogonal
    3.3426764835898436e-15
    >>> print(np.linalg.norm(np.einsum('xa,ya->xy', Vt_x_a, Vt_x_a) - np.eye(rank))) # Vt is orthogonal
    1.8969691003092744e-15
    '''
    G0_i_j_a = G0_i_a_j.swapaxes(1, 2)
    U_i_j_x, ss_x, Vt_x_a = left_svd_3tensor(G0_i_j_a, min_rank, max_rank, rtol, atol, xnp=xnp)
    U_i_x_j = U_i_j_x.swapaxes(1, 2)
    return U_i_x_j, ss_x, Vt_x_a
