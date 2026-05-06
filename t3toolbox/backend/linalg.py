# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import typing as typ
import numpy as np

from t3toolbox.backend.common import *

__all__ = [
    'pad_or_truncate',
    'truncated_svd',
    'left_svd',
    'right_svd',
    'up_svd',
    'left_svd_pair',
    'right_svd_pair',
    'up_svd_pair',
    'down_svd_pair',
]


def pad_or_truncate(
        array,
        pad_width,
        mode='constant',
        use_jax: bool = False,
        **kwargs
):
    xnp, _, _ = get_backend(False, use_jax)

    ndim = array.ndim

    slices = []
    pad = []

    for ii in range(ndim):
        before, after = pad_width[ii]

        start = max(0, -before)
        end = array.shape[ii] - max(0, -after)
        slices.append(slice(start, max(start, end)))

        pad.append((max(0, before), max(0, after)))

    sliced_A = array[tuple(slices)]

    return xnp.pad(sliced_A, pad, mode=mode, **kwargs)


######################################
########    Truncated SVD    #########
######################################

def truncated_svd(
        A: NDArray, # shape=(...,N,M)
        min_rank: int = None,  # 1 <= min_rank <= max_rank <= minimum(N, M)
        max_rank: int = None,  # 1 <= min_rank <= max_rank <= minimum(N, M)
        rtol: float = None,  # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None,  # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
) -> typ.Tuple[
    NDArray, # U, shape=(...,N,r)
    NDArray, # ss, shape=(...,r)
    NDArray, # Vt, shape=(...,r,M)
]:
    '''Compute (truncated) singular value decomposition of matrix A.

    A = U @ diag(ss) @ Vt
    Equality may be approximate if truncation is used.

    Parameters
    ----------
    A: NDArray
        Matrix. shape=(..., N, M)
    min_rank: int
        Minimum rank for truncation. Should have 1 <= min_rank <= max_rank <= minimum(N, M).
    min_rank: int
        Maximum rank for truncation. Should have 1 <= min_rank <= max_rank <= minimum(N, M).
    rtol: float
        Relative tolerance for truncation. Remove singular values satisfying sigma < maximum(atol, rtol*sigma1).
        Cannot be used for stacked A (len(A.shape) > 2).
    atol: float
        Absolute tolerance for truncation. Remove singular values satisfying sigma < maximum(atol, rtol*sigma1).
        Cannot be used for stacked A (len(A.shape) > 2).

    Returns
    -------
    U: NDArray
        Left singular vectors. shape=(..., N, r).
        U.T @ U = identity matrix
    ss: NDArray
        Singular values. Non-negative. shape=(..., r).
    Vt: NDArray
        Right singular vectors. shape=(..., r, M)
        Vt @ Vt.T = identity matrix

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> A = np.random.randn(2,3,4, 55,70)
    >>> U, ss, Vt = linalg.truncated_svd(A)
    >>> A2 = np.einsum('...ix,...x,...xj->...ij', U, ss, Vt)
    >>> print(np.linalg.norm(A - A2))
    1.0428742517412705e-13
    >>> print(np.linalg.norm(np.einsum('...ij,...ik->...jk', U, U) - np.eye(U.shape[-1])))
    1.1907994177245428e-14
    >>> print(np.linalg.norm(np.einsum('...ij,...kj->...ik', Vt, Vt) - np.eye(Vt.shape[-2])))
    1.1027751835566194e-14
    >>> print(np.all(ss >= 0.0))
    True

    Using max_rank:

    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> A = np.random.randn(2,3,4, 55,70)
    >>> U, ss, Vt = linalg.truncated_svd(A, max_rank=5)
    >>> A2 = np.einsum('...ix,...x,...xj->...ij', U, ss, Vt)
    >>> truncated_rank = ss.shape[-1]
    >>> print(truncated_rank)
    5

    Using rtol:

    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> A = np.array([[1.0 / (ii + jj) for jj in range(1, 70)] for ii in range(1, 55)]) # Hilbert matrix
    >>> U, ss, Vt = linalg.truncated_svd(A, rtol=1e-2) # Truncated SVD with relative tolerance 1e-2
    >>> A2 = np.einsum('ix,x,xj->ij', U, ss, Vt)
    >>> truncated_rank = ss.shape[-1]
    >>> print(truncated_rank)
    3
    >>> print(np.linalg.norm(A - A2, 2) / np.linalg.norm(A, 2)) # should be just less than rtol=1e-2
    0.008954100371711833

    Using atol:

    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> A = np.array([[1.0 / (ii + jj) for jj in range(1, 70)] for ii in range(1, 55)]) # Hilbert matrix
    >>> U, ss, Vt = linalg.truncated_svd(A, atol=1e-2) # Truncated SVD with absolute tolerance 1e-2
    >>> A2 = np.einsum('ix,x,xj->ij', U, ss, Vt)
    >>> truncated_rank = ss.shape[-1]
    >>> print(truncated_rank)
    4
    >>> print(np.linalg.norm(A - A2, 2))  # Check error in induced 2-norm
    0.002528973452232782

    Using rtol and min_rank:

    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> A = np.array([[1.0 / (ii + jj) for jj in range(1, 70)] for ii in range(1, 55)]) # Hilbert matrix
    >>> U, ss, Vt = linalg.truncated_svd(A, rtol=1e-2, min_rank=10)
    >>> A2 = np.einsum('ix,x,xj->ij', U, ss, Vt)
    >>> truncated_rank = ss.shape[-1]
    >>> print(truncated_rank)
    10
    >>> print(np.linalg.norm(A - A2, 2) / np.linalg.norm(A, 2)) # should be just less than rtol=1e-2
    10
    '''
    use_jax = is_jax_ndarray(A)
    xnp, _, _ = get_backend(False, use_jax)

    #
    U0, ss0, Vt0 = xnp.linalg.svd(A, full_matrices=False)

    if rtol is None and atol is None:
        K = ss0.shape[-1]
    else:
        if len(A.shape) > 2:
            raise ValueError(
                'Cannot use truncated_svd with rtol or atol for stacked matrix A (len(A.shape) > 2).\n' +
                'Different elements of the stack could end out having different shapes.\n' +
                'First unstack, then call truncated_svd for each unstacked matrix.\n' +
                'A.shape = ' + str(A.shape)
            )
        rtol1 = 0.0 if rtol is None else rtol
        atol1 = 0.0 if atol is None else atol
        tol = xnp.maximum(ss0[0] * rtol1, atol1)
        K = int(xnp.sum(ss0 >= tol))

    max_rank = K if max_rank is None else min(K, max_rank)
    min_rank = 1 if min_rank is None else max(1, min_rank)
    r = max(min(K, max_rank), min_rank)

    U   = U0[..., :, :r]
    ss  = ss0[..., :r]
    Vt  = Vt0[..., :r, :]

    return U, ss, Vt


def left_svd(
        G0_i_a_j: NDArray, # shape=(..., ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
) -> typ.Tuple[
    NDArray, # U_i_a_x, shape=(..., ni, na, r)
    NDArray, # ss_x,    shape=(.., r)
    NDArray, # Vt_x_j,  shape=(..., r, nj)
]:
    '''Compute (truncated) singular value decomposition of 3-tensor left unfolding.

    First two indices of the tensor are grouped for the SVD.
    '''
    stack_shape = G0_i_a_j.shape[:-3]
    ni, na, nj = G0_i_a_j.shape[-3:]
    G0_ia_j = G0_i_a_j.reshape(stack_shape + (ni*na, nj))

    U_ia_x, ss_x, Vt_x_j = truncated_svd(G0_ia_j, min_rank, max_rank, rtol, atol)

    nx = ss_x.shape[-1]
    U_i_a_x = U_ia_x.reshape(stack_shape + (ni, na, nx))
    return U_i_a_x, ss_x, Vt_x_j


def right_svd(
        G0_i_a_j: NDArray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
) -> typ.Tuple[
    NDArray, # U_i_x,       shape=(ni, nx)
    NDArray, # ss_x,        shape=(nx,)
    NDArray, # Vt_x_a_j,    shape=(nx, na, nj)
]:
    '''Compute (truncated) singular value decomposition of 3-tensor right unfolding.

    Last two indices of the tensor are grouped for the SVD.
    '''
    G0_j_a_i = G0_i_a_j.swapaxes(-3, -1)
    Vt_j_a_x, ss_x, U_x_i = left_svd(G0_j_a_i, min_rank, max_rank, rtol, atol)
    Vt_x_a_j = Vt_j_a_x.swapaxes(-1, -3)
    U_i_x = U_x_i.swapaxes(-2,-1)
    return U_i_x, ss_x, Vt_x_a_j


def up_svd(
        G0_i_a_j: NDArray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
) -> typ.Tuple[
    NDArray, # U_i_x_j, shape=(ni, nx, nj),
    NDArray, # ss_x,    shape=(nx,)
    NDArray, # Vt_x_a,  shape=(nx, na)
]:
    '''Compute (truncated) singular value decomposition of 3-tensor up unfolding.

    First and last indices of the tensor are grouped to form rows for the SVD.
    Middle index forms columns.
    '''
    G0_i_j_a = G0_i_a_j.swapaxes(-2, -1)
    U_i_j_x, ss_x, Vt_x_a = left_svd(G0_i_j_a, min_rank, max_rank, rtol, atol)
    U_i_x_j = U_i_j_x.swapaxes(-1, -2)
    return U_i_x_j, ss_x, Vt_x_a


#

def left_svd_pair(
        G0_i_a_j: NDArray, # shape=(..., ni, na, nj)
        G1_j_b_k: NDArray, # shape=(..., nj, nb, nk)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
) -> typ.Tuple[
    NDArray, # new_G0, shape=(..., ni, na, r)
    NDArray, # new_G1, shape=(..., r, nb, nj)
    NDArray, # ss,     shape=(.., r)
]:
    '''Compute (truncated) singular value decomposition of G0, pushing non-orthogonal remainder onto G1.
    '''
    use_jax = is_jax_ndarray(G0_i_a_j) or is_jax_ndarray(G1_j_b_k)
    xnp, _, _ = get_backend(False, use_jax)

    #
    U_i_a_x, ss_x, Vt_x_j = left_svd(G0_i_a_j, min_rank, max_rank, rtol, atol)
    new_G0 = U_i_a_x
    new_G1 = xnp.einsum('...x,...xj,...jbk->...xbk', ss_x, Vt_x_j, G1_j_b_k)
    return new_G0, new_G1, ss_x


def right_svd_pair(
        G0_i_a_j: NDArray, # shape=(..., ni, na, nj)
        G1_j_b_k: NDArray, # shape=(..., nj, nb, nk)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
) -> typ.Tuple[
    NDArray, # new_G0, shape=(..., ni, na, r)
    NDArray, # new_G1, shape=(..., r, nb, nj)
    NDArray, # ss,     shape=(.., r)
]:
    '''Compute (truncated) singular value decomposition of G1, pushing non-orthogonal remainder onto G0.
    '''
    rev_new_G1, rev_new_G0, ss = left_svd_pair(
        G1_j_b_k.swapaxes(-1, -3), G0_i_a_j.swapaxes(-1, -3),
        max_rank=max_rank, min_rank=min_rank, rtol=rtol, atol=atol,
    )
    return rev_new_G0.swapaxes(-1,-3), rev_new_G1.swapaxes(-1,-3), ss


def up_svd_pair(
        G_i_a_j: NDArray, # shape=(..., ni, na, nj)
        B_a_o: NDArray, # shape=(..., na, N)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
) -> typ.Tuple[
    NDArray, # new_G, shape=(..., ni, nx, nj),
    NDArray, # new_B, shape=(..., nx, N)
    NDArray, # ss, shape=(..., nx)
]:
    '''Compute (truncated) singular value decomposition of G, pushing non-orthogonal remainder onto B.
    '''
    use_jax = is_jax_ndarray(G_i_a_j) or is_jax_ndarray(B_a_o)
    xnp, _, _ = get_backend(False, use_jax)

    #
    U_i_x_j, ss_x, Vt_x_a = up_svd(
        G_i_a_j, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
    )
    new_G = U_i_x_j
    new_B = xnp.einsum('...x,...xa,...ao->...xo', ss_x, Vt_x_a, B_a_o)
    return new_G, new_B, ss_x


def down_svd_pair(
        G_i_a_j: NDArray, # shape=(..., ni, na, nj)
        B_a_o: NDArray, # shape=(..., na, N)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
) -> typ.Tuple[
    NDArray, # new_G, shape=(..., ni, nx, nj),
    NDArray, # new_B, shape=(..., nx, N)
    NDArray, # ss, shape=(..., nx)
]:
    '''Compute (truncated) singular value decomposition of B, pushing non-orthogonal remainder onto G.
    '''
    use_jax = is_jax_ndarray(G_i_a_j) or is_jax_ndarray(B_a_o)
    xnp, _, _ = get_backend(False, use_jax)

    #
    U_a_x, ss_x, Vt_x_o = truncated_svd(
        B_a_o, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
    )

    new_B = Vt_x_o
    new_G = xnp.einsum('...iaj,...ax,...x->...ixj', G_i_a_j, U_a_x, ss_x)
    return new_G, new_B, ss_x




