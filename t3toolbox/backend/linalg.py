# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Dense linear-algebra primitives shared by the orthogonalization and SVD sweeps.

``truncated_svd``, the directional ``left/right/up_svd`` and ``*_svd_pair`` factorizations
(directions match the core-unfolding conventions), ``pad_safe_svd`` (the mask-aware SVD of a
zero-padded matrix -- the uniform layer's replacement for a black-box SVD), and
``pad_or_truncate``. Pure array-in, array-out helpers with no T3 semantics.
"""
import functools
import typing as typ
import numpy as np

from t3toolbox.backend.common import *

__all__ = [
    'pad_or_truncate',
    'truncated_svd',
    'pad_safe_svd',
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
        **kwargs
):
    '''Pad and/or truncate an array per axis, using signed ``(before, after)`` widths.

    ``pad_width`` has one ``(before, after)`` pair per axis (like :py:func:`numpy.pad`), but a
    **negative** width *removes* that many entries instead of adding them: positive pads (``mode``
    controls fill, default zeros), negative truncates. The two compose -- an axis can be truncated on
    one side and padded on the other.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> a = np.arange(6).reshape(2, 3)
    >>> linalg.pad_or_truncate(a, [(1, 0), (0, 2)]).tolist()   # +1 row before, +2 cols after (zeros)
    [[0, 0, 0, 0, 0], [0, 1, 2, 0, 0], [3, 4, 5, 0, 0]]
    >>> linalg.pad_or_truncate(a, [(0, -1), (-1, 0)]).tolist()  # drop last row, drop first col
    [[1, 2]]
    '''
    xnp, _, _ = get_backend(False, is_jax_ndarray(array))

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
        rtol: float = None,  # tail-Frobenius rule: keep the smallest r with ||sigma[r:]||_2 < max(atol, rtol*||sigma||_2)
        atol: float = None,  # (the TT-SVD truncation criterion; NOT a per-singular-value threshold)
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
        Relative tolerance for truncation: keep the smallest rank ``r`` whose discarded tail satisfies
        ``||sigma[r:]||_2 < max(atol, rtol * ||sigma||_2)`` (the tail-Frobenius / TT-SVD rule, so the
        truncation error of the whole matrix is bounded by the tolerance -- NOT a per-singular-value cut).
        Cannot be used for stacked A (len(A.shape) > 2).
    atol: float
        Absolute tolerance for truncation, the same tail-Frobenius rule with ``atol`` in place of
        ``rtol * ||sigma||_2``.
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
    Default (no truncation): a full thin SVD, applied over the leading stack axes. The factors
    reconstruct ``A`` and are orthonormal:

    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> np.random.seed(0)
    >>> A = np.random.randn(2, 3, 4, 55, 70)              # stack_shape=(2,3,4), matrices 55x70
    >>> U, ss, Vt = linalg.truncated_svd(A)
    >>> print(U.shape, ss.shape, Vt.shape)               # r = min(55, 70) = 55
    (2, 3, 4, 55, 55) (2, 3, 4, 55) (2, 3, 4, 55, 70)
    >>> A2 = np.einsum('...ix,...x,...xj->...ij', U, ss, Vt)
    >>> print(np.allclose(A, A2))                         # U @ diag(ss) @ Vt == A
    True
    >>> print(np.allclose(np.einsum('...ix,...iy->...xy', U, U), np.eye(U.shape[-1])))   # U^T U = I
    True
    >>> print(np.allclose(np.einsum('...xj,...yj->...xy', Vt, Vt), np.eye(Vt.shape[-2])))  # Vt Vt^T = I
    True
    >>> print(bool(np.all(ss >= 0.0)))                    # singular values non-negative
    True

    ``max_rank`` caps the kept rank ``r`` (changes the output shapes), and works on a stack:

    >>> np.random.seed(0)
    >>> A = np.random.randn(2, 3, 4, 55, 70)
    >>> U, ss, Vt = linalg.truncated_svd(A, max_rank=5)
    >>> print(U.shape, ss.shape, Vt.shape)               # r capped at 5
    (2, 3, 4, 55, 5) (2, 3, 4, 5) (2, 3, 4, 5, 70)

    ``rtol`` bounds the Frobenius norm of the discarded tail by ``rtol * ||sigma||_2`` -- an approximation, so we show
    the kept rank and assert the accuracy bound rather than equality. Feed a graded spectrum (a
    Hilbert-like matrix) so the tolerance actually truncates; ``rtol``/``atol`` require an unstacked
    ``A``:

    >>> A = np.array([[1.0 / (ii + jj) for jj in range(1, 70)] for ii in range(1, 55)])  # graded spectrum
    >>> U, ss, Vt = linalg.truncated_svd(A, rtol=1e-2)
    >>> print(ss.shape[-1])                               # rtol=1e-2 keeps 3 singular values
    3
    >>> A2 = np.einsum('ix,x,xj->ij', U, ss, Vt)
    >>> rel_err = np.linalg.norm(A - A2, 2) / np.linalg.norm(A, 2)
    >>> print(bool(rel_err < 1e-2))                       # relative 2-norm error below rtol
    True
    >>> ss_full = np.linalg.svd(A, compute_uv=False)      # accuracy bound (generalized Oseledets):
    >>> dropped = np.linalg.norm(ss_full[ss.shape[-1]:])  #   ||A - A2||_F <= sqrt(dropped energy)
    >>> print(bool(np.linalg.norm(A - A2) <= dropped + 1e-12))
    True

    ``min_rank`` is a floor that overrides the tolerance -- here it forces rank 10 even though
    ``rtol`` alone would keep only 3, driving the error far below the tolerance:

    >>> U, ss, Vt = linalg.truncated_svd(A, rtol=1e-2, min_rank=10)
    >>> print(ss.shape[-1])                               # floored at min_rank=10
    10

    Gotcha: ``rtol``/``atol`` on a stacked ``A`` raises -- the kept rank could differ per stack
    element, giving ragged shapes. Unstack first, then truncate each matrix:

    >>> np.random.seed(0)
    >>> A = np.random.randn(2, 3, 4, 55, 70)
    >>> linalg.truncated_svd(A, rtol=1e-2)               # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError
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
        total_fronorm = xnp.sqrt(xnp.sum(ss0**2))
        tail_fronorms = xnp.sqrt(xnp.cumsum(ss0[::-1]**2))[::-1]
        tol = xnp.maximum(total_fronorm * rtol1, atol1)
        K = int(xnp.sum(tail_fronorms >= tol))

    max_rank = K if max_rank is None else min(K, max_rank)
    min_rank = 1 if min_rank is None else max(1, min_rank)
    r = max(max_rank, min_rank)

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

    First two indices of the tensor are grouped for the SVD: ``G[i,a,j] = sum_x U[i,a,x] ss[x] Vt[x,j]``,
    with ``U`` orthonormal in its grouped ``(i,a)`` rows. Truncation args behave as in
    :py:func:`truncated_svd`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> np.random.seed(0)
    >>> G = np.random.randn(4, 5, 6)                      # (ni, na, nj)
    >>> U, ss, Vt = linalg.left_svd(G)
    >>> print(U.shape, ss.shape, Vt.shape)               # U keeps (i,a); Vt is the 2d right factor
    (4, 5, 6) (6,) (6, 6)
    >>> print(np.allclose(np.einsum('iax,x,xj->iaj', U, ss, Vt), G))   # reconstructs G
    True
    >>> Um = U.reshape(4 * 5, -1)
    >>> print(np.allclose(Um.T @ Um, np.eye(Um.shape[1])))   # left unfolding of U is orthonormal
    True
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

    Last two indices of the tensor are grouped for the SVD: ``G[i,a,j] = sum_x U[i,x] ss[x] Vt[x,a,j]``,
    with ``Vt`` orthonormal in its grouped ``(a,j)`` rows. Truncation args behave as in
    :py:func:`truncated_svd`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> np.random.seed(0)
    >>> G = np.random.randn(4, 5, 6)                      # (ni, na, nj)
    >>> U, ss, Vt = linalg.right_svd(G)
    >>> print(U.shape, ss.shape, Vt.shape)               # U is the 2d left factor; Vt keeps (a,j)
    (4, 4) (4,) (4, 5, 6)
    >>> print(np.allclose(np.einsum('ix,x,xaj->iaj', U, ss, Vt), G))   # reconstructs G
    True
    >>> Vm = Vt.reshape(Vt.shape[0], -1)
    >>> print(np.allclose(Vm @ Vm.T, np.eye(Vm.shape[0])))   # right unfolding of Vt is orthonormal
    True
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

    First and last indices of the tensor are grouped to form rows for the SVD; the middle index forms
    columns: ``G[i,a,j] = sum_x U[i,x,j] ss[x] Vt[x,a]``, with ``U`` orthonormal in its grouped
    ``(i,j)`` rows. Truncation args behave as in :py:func:`truncated_svd`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> np.random.seed(0)
    >>> G = np.random.randn(4, 5, 6)                      # (ni, na, nj)
    >>> U, ss, Vt = linalg.up_svd(G)
    >>> print(U.shape, ss.shape, Vt.shape)               # U keeps (i,j) on either side of x; Vt is 2d
    (4, 5, 6) (5,) (5, 5)
    >>> print(np.allclose(np.einsum('ixj,x,xa->iaj', U, ss, Vt), G))   # reconstructs G
    True
    >>> Um = U.transpose(0, 2, 1).reshape(4 * 6, -1)
    >>> print(np.allclose(Um.T @ Um, np.eye(Um.shape[1])))   # up unfolding of U is orthonormal
    True
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

    Orthogonalizes ``G0`` via its left unfolding (so ``new_G0`` is left-orthonormal) and absorbs the
    ``ss @ Vt`` remainder into the shared bond of ``G1``, leaving the contracted product over the
    shared index unchanged. Truncation args behave as in :py:func:`truncated_svd`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> np.random.seed(0)
    >>> G0 = np.random.randn(2, 3, 4)                     # (ni, na, nj)
    >>> G1 = np.random.randn(4, 5, 6)                     # (nj, nb, nk) -- shared bond nj=4
    >>> new_G0, new_G1, ss = linalg.left_svd_pair(G0, G1)
    >>> print(new_G0.shape, new_G1.shape, ss.shape)
    (2, 3, 4) (4, 5, 6) (4,)
    >>> before = np.einsum('iaj,jbk->iabk', G0, G1)
    >>> after  = np.einsum('iax,xbk->iabk', new_G0, new_G1)
    >>> print(np.allclose(before, after))                # product across the shared bond preserved
    True
    >>> Um = new_G0.reshape(2 * 3, -1)
    >>> print(np.allclose(Um.T @ Um, np.eye(Um.shape[1])))   # new_G0 is left-orthonormal
    True
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

    Mirror of :py:func:`left_svd_pair`: orthogonalizes ``G1`` via its right unfolding (so ``new_G1`` is
    right-orthonormal) and absorbs the remainder into the shared bond of ``G0``, leaving the contracted
    product over the shared index unchanged. Truncation args behave as in :py:func:`truncated_svd`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> np.random.seed(0)
    >>> G0 = np.random.randn(2, 3, 4)                     # (ni, na, nj)
    >>> G1 = np.random.randn(4, 5, 6)                     # (nj, nb, nk) -- shared bond nj=4
    >>> new_G0, new_G1, ss = linalg.right_svd_pair(G0, G1)
    >>> print(new_G0.shape, new_G1.shape, ss.shape)
    (2, 3, 4) (4, 5, 6) (4,)
    >>> before = np.einsum('iaj,jbk->iabk', G0, G1)
    >>> after  = np.einsum('iax,xbk->iabk', new_G0, new_G1)
    >>> print(np.allclose(before, after))                # product across the shared bond preserved
    True
    >>> Vm = new_G1.reshape(new_G1.shape[0], -1)
    >>> print(np.allclose(Vm @ Vm.T, np.eye(Vm.shape[0])))   # new_G1 is right-orthonormal
    True
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

    ``G`` (a 3-tensor core) and ``B`` (a matrix) share ``G``'s middle index with ``B``'s first index.
    Orthogonalizes ``G`` via its up unfolding (so ``new_G`` is up-orthonormal) and absorbs the
    ``ss @ Vt`` remainder into ``B``, leaving the product over the shared index unchanged. Truncation
    args behave as in :py:func:`truncated_svd`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> np.random.seed(0)
    >>> G = np.random.randn(2, 3, 4)                      # (ni, na, nj)
    >>> B = np.random.randn(3, 7)                         # (na, N) -- shared index na=3
    >>> new_G, new_B, ss = linalg.up_svd_pair(G, B)
    >>> print(new_G.shape, new_B.shape, ss.shape)
    (2, 3, 4) (3, 7) (3,)
    >>> before = np.einsum('iaj,ao->ijo', G, B)
    >>> after  = np.einsum('ixj,xo->ijo', new_G, new_B)
    >>> print(np.allclose(before, after))                # product across the shared index preserved
    True
    >>> Um = new_G.transpose(0, 2, 1).reshape(2 * 4, -1)
    >>> print(np.allclose(Um.T @ Um, np.eye(Um.shape[1])))   # new_G is up-orthonormal
    True
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

    Mirror of :py:func:`up_svd_pair`: orthogonalizes the matrix ``B`` (so ``new_B`` has orthonormal
    rows) and absorbs the ``U @ ss`` remainder into ``G``'s shared middle index, leaving the product
    over the shared index unchanged. Truncation args behave as in :py:func:`truncated_svd`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> np.random.seed(0)
    >>> G = np.random.randn(2, 3, 4)                      # (ni, na, nj)
    >>> B = np.random.randn(3, 7)                         # (na, N) -- shared index na=3
    >>> new_G, new_B, ss = linalg.down_svd_pair(G, B)
    >>> print(new_G.shape, new_B.shape, ss.shape)
    (2, 3, 4) (3, 7) (3,)
    >>> before = np.einsum('iaj,ao->ijo', G, B)
    >>> after  = np.einsum('ixj,xo->ijo', new_G, new_B)
    >>> print(np.allclose(before, after))                # product across the shared index preserved
    True
    >>> print(np.allclose(new_B @ new_B.T, np.eye(new_B.shape[0])))   # new_B has orthonormal rows
    True
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






######################################
########    Pad-safe SVD    ##########
######################################

@functools.lru_cache(maxsize=None)
def _haar_sketch(M: int) -> np.ndarray:
    '''Fixed Haar-orthonormal sketch, one per width ``M`` (host numpy, fixed seed -- a jit constant).

    The QR sign fix makes the distribution Haar; Haar (``kappa = 1``) rather than plain Gaussian
    avoids the mildly heavy-tailed ``sigma_min`` of a square Gaussian sketch. Drawn ONCE per width
    from a fixed seed and cached: per-call randomness buys nothing, and a fixed sketch keeps
    ``pad_safe_svd`` deterministic (same input -> same output, across calls and processes).
    '''
    Q, R = np.linalg.qr(np.random.default_rng(0).standard_normal((M, M)))
    d = np.sign(np.diag(R))
    return Q * np.where(d == 0.0, 1.0, d)


def pad_safe_svd(
        A:        NDArray, # shape=(...,N,M); padded rows/columns identically zero
        row_mask: NDArray, # bool, shape=broadcastable to (...,N); True = REAL row, False = padded
        col_mask: NDArray, # bool, shape=broadcastable to (...,M); True = REAL column, False = padded
) -> typ.Tuple[
    NDArray, # U,  shape=(...,N,K), K=minimum(N,M); first q=minimum(n,m) columns bitwise zero on padded rows
    NDArray, # ss, shape=(...,K);   ss[q:] == 0 exactly
    NDArray, # Vt, shape=(...,K,M); first q rows bitwise zero on padded columns
]:
    '''SVD of a zero-padded matrix whose null-space vectors are forced OFF the padding.

    A black-box SVD of a padded matrix is wrong when the real block is numerically rank-deficient:
    the sigma ~= 0 left singular vectors are an arbitrary basis of a degenerate subspace that
    contains the pad coordinates, so they generically land on padded rows -- and a downstream mask
    then erases them (a lost direction: sliced to its real rows, ``U`` is non-orthonormal and
    rank-deficient). ``pad_safe_svd`` takes the real/pad partition as data and realizes the contract

        ``pad -> svd -> unpad  ==  svd of the unpadded n x m real block``

    with ``n = row_mask.sum()``, ``m = col_mask.sum()`` and ``q = minimum(n, m)``:

    * ``A == U @ diag(ss) @ Vt`` -- a genuine economy SVD of the padded matrix;
    * the first ``q`` triplets are a valid economy SVD of the real block: its positive singular
      values, then real-supported null completions, with ``U[..., :q]`` **bitwise** zero on padded
      rows and ``Vt[..., :q, :]`` **bitwise** zero on padded columns (exact ``== 0.0``, not small);
    * the remaining ``K - q`` don't-care triplets carry ``ss == 0`` exactly;
    * **no rank tolerance is used anywhere** -- every count comes from the masks or from bitwise
      {0, 1} indicators, so exact and roundoff-level zeros need no distinction.

    Any real ``(n, m)`` is supported -- tall, wide, or mixed across a batch -- and pads may sit at
    arbitrary (interior) positions. Masks are runtime data of static length: under jax they may be
    traced (or host-numpy constants), and one jit compile covers every mask pattern. The only
    branch is on the static padded shape (``N < M`` transposes internally).

    Algorithm (Method D, "sketch-project", from the pad-safe SVD design packet; record:
    ``dev/review_2026-08-22/repros/S1b/packet/``). Load-bearing details -- do not "simplify":

    * pad rows are permuted to the TRAILING pivot positions before the QR (Householder reflectors
      then never place mass on a padded row; the surplus columns come out as exact pad coordinate
      vectors, flagged by the bitwise indicator ``t``);
    * the separation constant is ``c = 4 * ||A||_F`` (Frobenius, per batch element). The filter
      ``sigma > c/2`` then has ``||A||``-sized margins on both sides. ``c = 2 * sigma_max`` is
      fragile: the threshold sits exactly at ``sigma_max`` and one-ulp rounding deletes the largest
      triplet (~38%% of generic rank-1 matrices, measured);
    * the augmented SVD's right factor is discarded (zero columns where ``t = 0`` pollute only it);
    * ``Vt`` is rebuilt from ``A.T @ U == V @ diag(ss)`` -- exactly its own QR up to column signs.

    The bitwise-zero guarantees rest on Householder-QR semantics (true for LAPACK, cuSOLVER, and
    jax's ``qr`` lowering on all backends; the big matrix never sees an SVD -- only the small
    augmented core does). They hold in float32 as well; only orthonormality/sigma accuracy scales
    with precision.

    Cost ``O(N M^2 + M^3)``, independent of the pad counts; no ``(N, N)`` intermediate exists.

    The complete derivation -- the problem and contract, why each step works, the two-sided
    separation-constant measurements, and every alternative considered (augmentation/GSVD, post-hoc
    completion, masked noise, ...) -- is ``docs/pad_safe_svd.tex`` (+pdf).

    Examples
    --------
    The failure this exists for: a zero-padded warm start (interior pad rows, and a real column
    that is exactly zero -- the padding of a rank-continuation restart). A black-box SVD puts
    null-space mass on the padded rows, so the masked real block is no longer orthonormal:

    >>> import numpy as np
    >>> import t3toolbox.backend.linalg as linalg
    >>> np.random.seed(0)
    >>> row_mask = np.array([True, False, True, True, False, True])   # pads at rows 1, 4 (interior)
    >>> col_mask = np.array([True, True, True, False])                # 3 real columns of 4
    >>> A = np.zeros((6, 4))
    >>> A[np.ix_(row_mask, col_mask)] = np.hstack([np.random.randn(4, 2), np.zeros((4, 1))])
    >>> U0, ss0, _ = np.linalg.svd(A, full_matrices=False)            # black-box SVD:
    >>> print(bool(np.all(U0[~row_mask][:, :3] == 0.0)))              #   pad rows contaminated
    False
    >>> Ur0 = U0[row_mask][:, :3]
    >>> print(float(np.round(np.linalg.norm(Ur0.T @ Ur0 - np.eye(3)), 2)))   # real block skewed
    0.26

    ``pad_safe_svd`` takes the masks (``True`` = real, the library polarity) and returns clean
    factors -- bitwise-zero pads, the real block orthonormal at full mask rank, singular values
    exactly those of the unpadded block:

    >>> U, ss, Vt = linalg.pad_safe_svd(A, row_mask, col_mask)
    >>> print(bool(np.all(U[~row_mask][:, :3] == 0.0)), bool(np.all(Vt[:3, ~col_mask] == 0.0)))
    True True
    >>> Ur = U[row_mask][:, :3]
    >>> print(np.allclose(Ur.T @ Ur, np.eye(3)))                      # no lost directions
    True
    >>> print(np.allclose(ss[:3], np.linalg.svd(A[np.ix_(row_mask, col_mask)], compute_uv=False)))
    True
    >>> print(np.allclose(np.einsum('ix,x,xj->ij', U, ss, Vt), A), float(ss[3]))
    True 0.0

    A wide real block (``n < m``) needs no transpose by the caller -- the contract is symmetric
    in ``min(n, m)`` (here 2) -- and a statically wide matrix (``N < M``) transposes internally:

    >>> B = np.zeros((3, 5)); B[:2, :4] = np.random.randn(2, 4)
    >>> Uw, sw, Vtw = linalg.pad_safe_svd(B, np.array([1, 1, 0], bool), np.array([1, 1, 1, 1, 0], bool))
    >>> print(Uw.shape, sw.shape, Vtw.shape)                          # K = min(N, M) = 3 triplets
    (3, 3) (3,) (3, 5)
    >>> print(np.allclose(sw[:2], np.linalg.svd(B[:2, :4], compute_uv=False)), bool(np.all(sw[2:] == 0.0)))
    True True
    '''
    use_jax = tree_contains_jax((A, row_mask, col_mask))
    xnp, _, _ = get_backend(False, use_jax)

    A = xnp.asarray(A)
    N, M = A.shape[-2:]
    if N < M:  # static branch on concrete padded shapes: run tall, swap the factors back
        U_T, ss, Vt_T = pad_safe_svd(A.swapaxes(-2, -1), col_mask, row_mask)
        return Vt_T.swapaxes(-2, -1), ss, U_T.swapaxes(-2, -1)

    # broadcast the masks over the leading batch axes up front (take_along_axis needs full ndim)
    lead = np.broadcast_shapes(A.shape[:-2], np.shape(row_mask)[:-1], np.shape(col_mask)[:-1])
    A = xnp.broadcast_to(A, lead + (N, M))
    row_mask = xnp.broadcast_to(xnp.asarray(row_mask).astype(bool), lead + (N,))
    col_mask = xnp.broadcast_to(xnp.asarray(col_mask).astype(bool), lead + (M,))

    # -- left basis: sketch, permute pads to the trailing pivots, QR, un-permute ------------------
    Omega = xnp.asarray(_haar_sketch(M), dtype=A.dtype)
    Y_nk = xnp.einsum('...nm,mk->...nk', A, Omega)                  # padded rows bitwise zero for any Omega
    key_n = xnp.where(row_mask, 0, N) + xnp.arange(N)               # distinct ints: real rows first,
    pr = xnp.argsort(key_n, axis=-1)                                #   original order (sort-stable by key)
    inv_pr = xnp.argsort(pr, axis=-1)
    Qp_nx, _ = xnp.linalg.qr(xnp.take_along_axis(Y_nk, pr[..., :, None], axis=-2))
    Q_nx = xnp.take_along_axis(Qp_nx, inv_pr[..., :, None], axis=-2)
    # bitwise split of Q's columns: data-supported | exact pad coordinate vectors (the surplus)
    t_x = xnp.max(xnp.abs(Q_nx) * (~row_mask)[..., :, None], axis=-2) > 0.5

    # -- one SVD of the small augmented core (static (M, 2M)) -------------------------------------
    B_xm = xnp.einsum('...nx,...nm->...xm', Q_nx, A)                # surplus rows bitwise zero
    normA = xnp.sqrt(xnp.sum(A * A, axis=(-2, -1)))
    c = 4.0 * normA + (normA == 0.0)                                # 4x margin -- load-bearing, see above
    pin_block = c[..., None, None] * (t_x[..., :, None] * xnp.eye(M, dtype=A.dtype))
    W1_xk, Sa_k, _ = xnp.linalg.svd(xnp.concatenate([B_xm, pin_block], axis=-1), full_matrices=False)

    # -- classify {pin, data} with margin, sort kept descending + pins last, assemble U and ss ----
    pinned_k = Sa_k > c[..., None] / 2.0
    order = xnp.argsort(xnp.where(pinned_k, -1.0, Sa_k), axis=-1)[..., ::-1]
    W1_xk = xnp.take_along_axis(W1_xk, order[..., None, :], axis=-1)
    So_k = xnp.take_along_axis(Sa_k, order, axis=-1)
    pino_k = xnp.take_along_axis(pinned_k, order, axis=-1)
    W1_xk = xnp.where(t_x[..., :, None] & ~pino_k[..., None, :], 0.0, W1_xk)   # exact-arithmetic-zero dirt
    U_nk = xnp.einsum('...nx,...xk->...nk', Q_nx, W1_xk)            # first q columns bitwise clean
    q = xnp.minimum(xnp.sum(row_mask, axis=-1), xnp.sum(col_mask, axis=-1))
    ss_k = xnp.where(pino_k, 0.0, So_k)                             # zero pins BY FLAG (n < m support),
    ss_k = xnp.where(xnp.arange(M) < q[..., None], ss_k, 0.0)       #   then the sigma ~= 0 tail exactly

    # -- rebuild the right factor: A^T @ U == V @ diag(ss), already its own QR up to column signs --
    W_mk = xnp.einsum('...nm,...nk->...mk', A, U_nk)                # padded-column rows bitwise zero
    key_m = xnp.where(col_mask, 0, M) + xnp.arange(M)
    pc = xnp.argsort(key_m, axis=-1)
    inv_pc = xnp.argsort(pc, axis=-1)
    Vq_mk, R_kk = xnp.linalg.qr(xnp.take_along_axis(W_mk, pc[..., :, None], axis=-2))
    V_mk = xnp.take_along_axis(Vq_mk, inv_pc[..., :, None], axis=-2)
    d_k = xnp.sign(xnp.diagonal(R_kk, axis1=-2, axis2=-1))
    V_mk = V_mk * xnp.where(d_k == 0.0, 1.0, d_k)[..., None, :]
    return U_nk, ss_k, V_mk.swapaxes(-2, -1)
