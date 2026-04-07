# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ

try:
    import jax.numpy as jnp
except:
    print('jax import failed. Defaulting to numpy.')
    jnp = np

NDArray = typ.Union[np.ndarray, jnp.ndarray]

__all__ = [
    'numpy_scan',
    #
    'truncated_svd',
    'left_svd_3tensor',
    'right_svd_3tensor',
    'outer_svd_3tensor',
    #
    'probe_dense',
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
    """
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)


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
    >>> import numpy as np
    >>> import t3tools.dense as dense
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_a_x, ss_x, Vt_x_j = dense.left_svd_3tensor(G_i_a_j)
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
    >>> import numpy as np
    >>> import t3tools.dense as dense
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_x, ss_x, Vt_x_a_j = dense.right_svd_3tensor(G_i_a_j)
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
    >>> import numpy as np
    >>> import t3tools.dense as dense
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_x_j, ss_x, Vt_x_a = dense.outer_svd_3tensor(G_i_a_j)
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
    U_i_j_x, ss_x, Vt_x_a = left_svd_3tensor(G0_i_j_a, min_rank, max_rank, rtol, atol, use_jax)
    U_i_x_j = U_i_j_x.swapaxes(1, 2)
    return U_i_x_j, ss_x, Vt_x_a


###############################################
##########    Probe dense tensor    ###########
###############################################

def probe_dense(
        T:          NDArray,
        vectors:    typ.Sequence[NDArray],
        use_jax: bool = False,
) -> typ.Tuple[NDArray]:
    """Probe a dense tensor.

    Parameters
    ----------
    T: NDArray
        Tensor to be probed. shape=(N1,...,Nd)
    vectors: typ.Sequence[NDArray]
        Probing input vectors.
        len=d.
        elm_shape=(Ni,) or elm_shape=(num_probes, Ni)
    use_jax: bool
        Whether to use jax for numerical operations (default: False)

    Returns
    -------
    typ.Tuple[NDArray]
        Probes.
        len=d.
        elm_shape=(Ni,) or elm_shape=(num_probes, Ni)

    Examples
    --------

    Probe with one set of vectors:

    >>> import numpy as np
    >>> import t3tools.dense as dense
    >>> T = np.random.randn(10,11,12)
    >>> u0 = np.random.randn(10)
    >>> u1 = np.random.randn(11)
    >>> u2 = np.random.randn(12)
    >>> yy = dense.probe_dense(T, (u0,u1,u2))
    >>> y0 = np.einsum('ijk,j,k', T, u1, u2)
    >>> y1 = np.einsum('ijk,i,k', T, u0, u2)
    >>> y2 = np.einsum('ijk,i,j', T, u0, u1)
    >>> print(np.linalg.norm(yy[0] - y0))
    2.0928808318295785e-14
    >>> print(np.linalg.norm(yy[1] - y1))
    1.0841599276764049e-14
    >>> print(np.linalg.norm(yy[2] - y2))
    1.2970142174948615e-14

    Probe with two sets of vectors:

    >>> import numpy as np
    >>> import t3tools.dense as dense
    >>> T = np.random.randn(10,11,12)
    >>> u0, v0 = np.random.randn(10), np.random.randn(10)
    >>> u1, v1 = np.random.randn(11), np.random.randn(11)
    >>> u2, v2 = np.random.randn(12), np.random.randn(12)
    >>> uuu = [np.vstack([u0,v0]), np.vstack([u1,v1]), np.vstack([u2,v2])]
    >>> yyy = dense.probe_dense(T, uuu)
    >>> yy_u = dense.probe_dense(T, (u0,u1,u2))
    >>> yy_v = dense.probe_dense(T, (v0,v1,v2))
    >>> print(np.linalg.norm(yy_u[0] - yyy[0][0,:]))
    0.0
    >>> print(np.linalg.norm(yy_u[1] - yyy[1][0,:]))
    0.0
    >>> print(np.linalg.norm(yy_u[2] - yyy[2][0,:]))
    0.0
    >>> print(np.linalg.norm(yy_v[0] - yyy[0][1,:]))
    0.0
    >>> print(np.linalg.norm(yy_v[1] - yyy[1][1,:]))
    0.0
    >>> print(np.linalg.norm(yy_v[2] - yyy[2][1,:]))
    0.0
    """
    xnp = jnp if use_jax else np

    num_cores = len(T.shape)
    assert(len(vectors) == num_cores)
    if len(vectors[0].shape) == 1:
        vectorized=False
        for ii in range(num_cores):
            assert (len(vectors[ii].shape) == 1)
    elif len(vectors[0].shape) == 2:
        vectorized=True
        for ii in range(num_cores):
            assert (len(vectors[ii].shape) == 2)
    else:
        raise RuntimeError(
            'Wrong vectors[ii] shape in probe_dense. Should be vector or matrix.\n'
            + 'vectors[0].shape=' + str(vectors[0].shape)
        )

    vectors = list(vectors)
    if vectorized == False:
        for ii in range(num_cores):
            vectors[ii] = vectors[ii].reshape((1,-1))

    vector_lengths = tuple([x.shape[1] for x in vectors])
    assert(vector_lengths == T.shape)

    probes = []
    for ii in range(num_cores):
        Ai = T
        for jj in range(ii):
            if jj == 0:
                Ai = xnp.einsum('pi,i...->p...', vectors[jj], Ai)
            else:
                Ai = xnp.einsum('pi,pi...->p...', vectors[jj], Ai)

        for jj in range(num_cores-1, ii, -1):
            if ii==0 and jj==num_cores-1:
                Ai = xnp.einsum('pi,...i->p...', vectors[jj], Ai)
            else:
                Ai = xnp.einsum('pi,p...i->p...', vectors[jj], Ai)
        probes.append(Ai)

    if not vectorized:
        probes = [Ai.reshape(-1) for Ai in probes]

    return tuple(probes)