# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ

__all__ = [
    'jnp',
    'NDArray',
    'numpy_scan',
    'jax_scan',
    #
    'truncated_svd',
    #
    'corewise_add',
    'corewise_scale',
    'corewise_neg',
    'corewise_sub',
    'corewise_dot',
    'corewise_norm',
]


def numpy_scan(f, init, xs, length=None):
    """Numpy version of jax.lax.scan.
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html

    Generated with help from AI.
    """
    if length is None:
        def get_length(tree):
            if isinstance(tree, (list, tuple)): return get_length(tree[0])
            return len(tree)

        length = get_length(xs)

    def get_slice(tree, i):
        if isinstance(tree, tuple):
            return tuple(get_slice(x, i) for x in tree)
        if isinstance(tree, list):
            return [get_slice(x, i) for x in tree]
        return tree[i]

    carry = init
    ys_list = []

    for i in range(length):
        current_xs = get_slice(xs, i)
        carry, y = f(carry, current_xs)
        ys_list.append(y)

    if isinstance(ys_list[0], (tuple, list)):
        return carry, tuple(np.stack([step[i] for step in ys_list]) for i in range(len(ys_list[0])))

    return carry, np.stack(ys_list)


try:
    import jax.numpy as jnp
    import jax
    jax_scan = jax.lax.scan
except ImportError:
    print('jax import failed. Defaulting to numpy.')
    jnp = np
    jax_scan = numpy_scan

NDArray = typ.Union[np.ndarray, jnp.ndarray]


###############################################
########    Corewise linear algebra    ########
###############################################

def corewise_add(X, Y):
    '''Add nested objects, X,Y -> X+Y.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> Y = (2*np.ones(3), (3, (), -np.ones(2)))
    >>> print(cw.corewise_add(X, Y))
    (array([3., 3., 3.]), (4, (), array([0., 0.])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        assert(isinstance(Y, list) or isinstance(Y, tuple))
        assert(len(X) == len(Y))
        return tuple([corewise_add(x, y) for x, y in zip(X, Y)])
    else:
        return X + Y


def corewise_sub(X, Y):
    '''Subtract nested objects, X,Y -> X-Y.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> Y = (2*np.ones(3), (3, (), -np.ones(2)))
    >>> print(cw.corewise_sub(X, Y))
    (array([-1., -1., -1.]), (-2, (), array([2., 2.])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        assert(isinstance(Y, list) or isinstance(Y, tuple))
        assert(len(X) == len(Y))
        return tuple([corewise_sub(x, y) for x, y in zip(X, Y)])
    else:
        return X - Y


def corewise_scale(X, s):
    '''Scale nested objects, X,s -> s*X.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> print(cw.corewise_scale(X, 1.5))
    (array([-1., -1., -1.]), (-2, (), array([2., 2.])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        return tuple([corewise_scale(x, s) for x in X])
    else:
        return s*X


def corewise_neg(X):
    '''Negate nested objects, X -> -X.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> print(cw.corewise_neg(X))
    (array([-1., -1., -1.]), (-1, (), array([-1., -1.])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        return tuple([corewise_neg(x) for x in X])
    else:
        return -X


def corewise_dot(X, Y, use_jax: bool = False):
    '''Dot product of nested objects, X,Y -> X.Y.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> Y = (2*np.ones(3), (3, (), -np.ones(2)))
    >>> print(cw.corewise_dot(X, Y))
    7.0
    '''
    xnp = jnp if use_jax else np
    if isinstance(X, list) or isinstance(X, tuple):
        assert(isinstance(Y, list) or isinstance(Y, tuple))
        assert(len(X) == len(Y))
        return xnp.sum(xnp.array([corewise_dot(x, y) for x, y in zip(X, Y)]))
    else:
        return xnp.sum(X * Y)


def corewise_norm(X, use_jax: bool = False):
    '''Norm of nested objects, X -> ||X||

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> print(cw.corewise_norm(X))
    2.449489742783178
    >>> print(np.sqrt(3 + 1 + 2))
    2.449489742783178
    '''
    xnp = jnp if use_jax else np
    return xnp.sqrt(corewise_dot(X, X, use_jax=use_jax))


######################################
########    Truncated SVD    #########
######################################

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


