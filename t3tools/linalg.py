import typing as typ
import numpy as np

NDArray = np.ndarray
xnp = np

__all__ = [
    'truncated_svd',
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
