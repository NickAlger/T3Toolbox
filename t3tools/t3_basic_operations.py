import numpy as np
import jax
import jax.numpy as jnp
import typing as typ

from .tt_basic_operations import *

__all__ = [
    'TuckerTensorTrain',
    'T3Structure',
    't3_get_shape',
    't3_get_tucker_ranks',
    't3_get_tt_ranks',
    't3_get_structure',
    't3_svd',
    't3_to_dense',
    'left_svd_3tensor',
    'right_svd_3tensor',
    'outer_svd_3tensor',
    'orthogonalize_ith_basis_core',
    'left_svd_ith_tt_core',
    'right_svd_ith_tt_core',
    'inner_svd_ith_tt_core',
    'outer_svd_ith_tt_core',
    't3_check_correctness',
    't3_zeros',
    't3_remove_useless_rank',
    't3_pad_rank',
    'tucker_svd_dense',
    't3_svd_dense',
    't3_save',
    't3_load',
    #'t3_reverse',
]


######################################################
########    Tucker Tensor Train operations    ########
######################################################

Array = typ.Union[np.ndarray, jnp.ndarray]

TuckerTensorTrain = typ.Tuple[
    typ.Sequence[Array], # basis_cores, len=d, elm_shape=(ni, Ni)
    typ.Sequence[Array], # tt_cores, len=d, elm_shape=(ri, ni, r(i+1))
]

T3Structure = typ.Tuple[
    typ.Sequence[int], # shape, len=d
    typ.Sequence[int], # tucker_ranks, len=d
    typ.Sequence[int], # tt_ranks, len=d+1
]


def t3_get_shape(
        x: TuckerTensorTrain,
) -> typ.Tuple[int,...]: # shape of x, (N1,N2,...,Nd), len=d
    r"""Get the shape of the tensor represented by a Tucker tensor train.

    Parameters
    ----------
    x : TuckerTensorTrain
        Tucker tensor train representation of a tensor of shape (N1, N2, ..., Nd)

    Returns
    -------
    typ.Tuple[int,...]
        Shape of tensor represented by x. (N1, N2, ..., Nd). len=d

    See Also
    --------
    TuckerTensorTrain
    T3Structure
    t3_get_tucker_ranks
    t3_get_tt_ranks
    t3_get_structure

    Examples
    --------

    >>> basis_cores = (np.random.randn(4,14), np.random.randn(5,15), np.random.randn(6,16))
    >>> tt_cores = (np.random.randn(1,4,3), np.random.randn(3,5,2), np.random.randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> print(t3_get_shape(x))
    (14, 15, 16)
    >>> print(t3_to_dense(x).shape == t3_get_shape(x))
    True
    """
    basis_cores, tt_cores = x
    return tuple([B.shape[1] for B in basis_cores])


def t3_get_tucker_ranks(
        x: TuckerTensorTrain,
) -> typ.Tuple[int,...]: # tucker_ranks=(n1,n2,...,nd)
    basis_cores, tt_cores = x
    return tuple([B.shape[0] for B in basis_cores])


def t3_get_tt_ranks(
        x: TuckerTensorTrain,
) -> typ.Tuple[int,...]: # tucker_ranks=(r1,r2,...,n(d+1))
    basis_cores, tt_cores = x
    return tt_get_ranks(tt_cores)


def t3_get_structure(
        x: TuckerTensorTrain,
) -> T3Structure:
    return t3_get_shape(x), t3_get_tucker_ranks(x), t3_get_tt_ranks(x)


def t3_check_correctness(
        x: TuckerTensorTrain,
) -> None:
    basis_cores, tt_cores = x
    tt_check_correctness(tt_cores)

    assert(len(basis_cores) == len(tt_cores))

    for B in basis_cores:
        assert(len(B.shape) == 2)

    for n, B in zip(tt_get_shape(tt_cores), basis_cores):
        assert(B.shape[0] == n)


def left_svd_3tensor(
        G0_i_a_j: jnp.ndarray, # shape=(ni, na, nj)
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    jnp.ndarray, # U_i_a_x, shape=(ni, na, nx)
    jnp.ndarray, # ss_x,    shape=(nx,)
    jnp.ndarray, # Vt_x_j,  shape=(nx, nj)
]:
    rtol1 = 0.0 if rtol is None else rtol
    atol1 = 0.0 if atol is None else atol

    ni, na, nj = G0_i_a_j.shape
    G0_ia_j = G0_i_a_j.reshape((ni*na, nj))

    U_ia_y, ss_y, Vt_y_j = jnp.linalg.svd(G0_ia_j, full_matrices=False)

    tol = jnp.maximum(ss_y[0] * rtol1, atol1)

    if forced_rank is not None:
        nx = forced_rank
    elif rtol is None and atol is None:
        nx = len(ss_y)
    else:
        nx = jnp.sum(ss_y >= tol)
        if max_rank is not None:
            nx = jnp.minimum(nx, max_rank)

    U_ia_x = U_ia_y[:, :nx]
    ss_x = ss_y[:nx]
    Vt_x_j = Vt_y_j[:nx, :]

    nx = len(ss_x)
    U_i_a_x = U_ia_x.reshape((ni, na, nx))
    return U_i_a_x, ss_x, Vt_x_j


def right_svd_3tensor(
        G0_i_a_j: jnp.ndarray, # shape=(ni, na, nj)
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    jnp.ndarray, # U_i_x,       shape=(ni, nx)
    jnp.ndarray, # ss_x,        shape=(nx,)
    jnp.ndarray, # Vt_x_a_j,    shape=(nx, na, nj)
]:
    G0_j_a_i = G0_i_a_j.swapaxes(0, 2)
    Vt_j_a_x, ss_x, U_x_i = left_svd_3tensor(G0_j_a_i, max_rank, rtol, atol, forced_rank)
    Vt_x_a_j = Vt_j_a_x.swapaxes(0, 2)
    U_i_x = U_x_i.swapaxes(0,1)
    return U_i_x, ss_x, Vt_x_a_j


def outer_svd_3tensor(
        G0_i_a_j: jnp.ndarray, # shape=(ni, na, nj)
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    jnp.ndarray, # U_i_x_j, shape=(ni, nx, nj),
    jnp.ndarray, # ss_x,    shape=(nx,)
    jnp.ndarray, # Vt_x_a,  shape=(nx, na)
]:
    G0_i_j_a = G0_i_a_j.swapaxes(1, 2)
    U_i_j_x, ss_x, Vt_x_a = left_svd_3tensor(G0_i_j_a, max_rank, rtol, atol, forced_rank)
    U_i_x_j = U_i_j_x.swapaxes(1, 2)
    return U_i_x_j, ss_x, Vt_x_a


def orthogonalize_ith_basis_core(
        ii: int, # which base core to orthogonalize
        x: TuckerTensorTrain,
) -> TuckerTensorTrain:
    basis_cores, tt_cores = x
    G_a_i_b = tt_cores[ii]
    U_i_o = basis_cores[ii]
    U_o_i = U_i_o.T
    U2_o_x, R_x_i = jnp.linalg.qr(U_o_i, mode='reduced')
    G2_a_x_b = jnp.einsum('aib,xi->axb', G_a_i_b, R_x_i)
    U2_x_o = U2_o_x.T

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G2_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = U2_x_o

    return tuple(new_basis_cores), tuple(new_tt_cores)


def left_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    jnp.ndarray, # singular values, shape=(r(i+1),)
]:
    basis_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii]
    B0_b_j_c = tt_cores[ii+1]

    A_a_i_x, ss_x, Vt_x_b = left_svd_3tensor(A0_a_i_b, max_rank, rtol, atol, forced_rank)
    B_x_j_c = jnp.tensordot(ss_x.reshape((-1,1)) * Vt_x_b, B0_b_j_c, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = A_a_i_x
    new_tt_cores[ii+1] = B_x_j_c

    return (tuple(basis_cores), tuple(new_tt_cores)), ss_x


def right_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    jnp.ndarray, # singular values, shape=(ri,)
]:
    basis_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii-1]
    B0_b_j_c = tt_cores[ii]

    U_b_x, ss_x, B_x_j_c = right_svd_3tensor(B0_b_j_c, max_rank, rtol, atol, forced_rank)
    A_a_i_x = jnp.tensordot(A0_a_i_b, U_b_x * ss_x.reshape((1,-1)), axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii-1] = A_a_i_x
    new_tt_cores[ii] = B_x_j_c

    return (tuple(basis_cores), tuple(new_tt_cores)), ss_x


def inner_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    jnp.ndarray, # singular values, shape=(ri,)
]:
    basis_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = basis_cores[ii]

    U_a_x_b, ss_x, Vt_x_i = outer_svd_3tensor(G0_a_i_b, max_rank, rtol, atol, forced_rank)

    G_a_x_b = jnp.einsum('axb,x->axb', U_a_x_b, ss_x)
    Q_x_o = jnp.tensordot(Vt_x_i, Q0_i_o, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = Q_x_o

    return (tuple(new_basis_cores), tuple(new_tt_cores)), ss_x


def outer_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    jnp.ndarray, # singular values, shape=(ri,)
]:
    basis_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = basis_cores[ii]

    G_a_x_b, ss_x, Vt_x_i = outer_svd_3tensor(G0_a_i_b, max_rank, rtol, atol, forced_rank)

    Q_x_o = (ss_x.reshape((-1,1)) * Vt_x_i) @ Q0_i_o

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = Q_x_o

    return (tuple(new_basis_cores), tuple(new_tt_cores)), ss_x


def t3_svd(
        x: TuckerTensorTrain,
        max_tt_ranks:       typ.Sequence[int] = None, # len=k-1, i.e., Correct: (r1, ..., r_{k-1}), Incorrect: (1,r1, ..., r_{k-1},1)
        max_basis_ranks:    typ.Sequence[int] = None, # len=k,
        rtol: float = None,
        atol: float = None,
        forced_tt_ranks:    typ.Sequence[int] = None, # len=k-1, i.e., Correct: (r1, ..., r_{k-1}), Incorrect: (1,r1, ..., r_{k-1},1)
        forced_basis_ranks: typ.Sequence[int] = None, # len=k
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    typ.Tuple[jnp.ndarray,...], # basis singular values, len=k
    typ.Tuple[jnp.ndarray,...], # tt singular values, len=k+1
]:
    basis_cores, tt_cores = x

    num_cores = len(tt_cores)

    # Orthogonalize basis matrices
    for ii in range(num_cores):
        x = orthogonalize_ith_basis_core(ii, x)

    # Right orthogonalize
    for ii in range(num_cores-1, 0, -1): # num_cores-1, num_cores-2, ..., 1
        x, _ = right_svd_ith_tt_core(ii, x)

    G0 = x[1][0]
    _, ss_first, _ = right_svd_3tensor(G0)
    # _, ss_first, _ = right_svd_3tensor(tt_cores[0])

    # Sweep left to right computing SVDS
    all_ss_basis = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        max_r_basis = max_basis_ranks[ii] if max_basis_ranks is not None else None
        forced_r_basis = forced_basis_ranks[ii] if forced_basis_ranks is not None else None
        # SVD inbetween tt core and basis core
        x, ss_basis = inner_svd_ith_tt_core(
            ii, x, max_r_basis, rtol, atol, forced_rank=forced_r_basis,
        )
        all_ss_basis.append(ss_basis)

        if ii < num_cores-1:
            max_r_tt = max_tt_ranks[ii] if max_tt_ranks is not None else None
            forced_r_tt = forced_tt_ranks[ii] if forced_tt_ranks is not None else None
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = left_svd_ith_tt_core(
                ii, x, max_r_tt, rtol, atol, forced_rank=forced_r_tt,
            )
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = left_svd_3tensor(Gf)
            # _, ss_tt, _ = left_svd_3tensor(tt_cores[-1])
        all_ss_tt.append(ss_tt)

    return x, tuple(all_ss_basis), tuple(all_ss_tt)


def t3_to_dense(
        x: TuckerTensorTrain,
) -> jnp.ndarray:
    basis_cores, tt_cores = x
    big_tt_cores = [jnp.einsum('iaj,ab->ibj', G, U) for G, U in zip(tt_cores, basis_cores)]
    return tt_to_dense(big_tt_cores)


def t3_zeros(
        shape:          typ.Sequence[int],
        tucker_ranks:   typ.Sequence[int],
        tt_ranks:       typ.Sequence[int],
) -> TuckerTensorTrain:
    tt_cores = tt_zeros(tucker_ranks, tt_ranks)
    basis_cores = tuple([jnp.zeros((n, N)) for n, N  in zip(tucker_ranks, shape)])
    return basis_cores, tt_cores


def t3_remove_useless_rank(
        shape:          typ.Sequence[int], # len=d
        tucker_ranks:   typ.Sequence[int], # len=d
        tt_ranks:       typ.Sequence[int], # len=d+1
) -> typ.Tuple[
    typ.Tuple[int,...], # new_tucker_ranks
    typ.Tuple[int,...], # new_tt_ranks
]:
    d = len(shape)
    assert(len(tucker_ranks) == d)
    assert(len(tt_ranks) == d+1)

    new_tucker_ranks   = list(tucker_ranks)
    new_tt_ranks       = list(tt_ranks)

    for ii in range(d):
        new_tucker_ranks[ii] = int(np.minimum(new_tucker_ranks[ii], shape[ii]))

    for ii in range(d-1, 0, -1):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        new_tt_ranks[ii] = int(np.minimum(rL, n*rR))

    for ii in range(d):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        n = int(np.minimum(n, rL*rR))
        rR =int(np.minimum(rR, rL*n))
        new_tucker_ranks[ii] = n
        new_tt_ranks[ii+1] = rR

    return tuple(new_tucker_ranks), tuple(new_tt_ranks)


def t3_pad_rank(
        cores:              TuckerTensorTrain,
        new_tucker_ranks:   typ.Sequence[int],
        new_tt_ranks:       typ.Sequence[int],
) -> TuckerTensorTrain:
    t3_check_correctness(cores)
    shape = t3_get_shape(cores)
    num_cores = len(shape)
    old_tt_ranks        = t3_get_tt_ranks(cores)
    old_tucker_ranks    = t3_get_tucker_ranks(cores)
    assert(len(old_tucker_ranks) == len(new_tucker_ranks))
    assert(len(old_tt_ranks) == len(new_tt_ranks))

    delta_tucker_ranks  = [r_new - r_old for r_new, r_old in zip(new_tucker_ranks, old_tucker_ranks)]
    delta_tt_ranks      = [r_new - r_old for r_new, r_old in zip(new_tt_ranks, old_tt_ranks)]

    tucker_cores, tt_cores = cores

    new_tucker_cores = []
    for ii in range(num_cores):
        new_tucker_cores.append(jnp.pad(
            tucker_cores[ii],
            ((0,delta_tucker_ranks[ii]), (0,0)),
        ))

    new_tt_cores = []
    for ii in range(num_cores):
        new_tt_cores.append(jnp.pad(
            tt_cores[ii],
            (
                (0,delta_tt_ranks[ii]),
                (0,delta_tucker_ranks[ii]),
                (0,delta_tt_ranks[ii+1]),
            ),
        ))

    return tuple(new_tucker_cores), tuple(new_tt_cores)

#

def tucker_svd_dense(
        T: jnp.ndarray, # shape=(N1, N2, .., Nd)
        max_ranks: typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[jnp.ndarray, ...], # Tucker bases, ith_elm_shape=(ni, Ni)
        jnp.ndarray, # Tucker core, shape=(n1,n2,...,nd)
    ],
    typ.Tuple[jnp.ndarray], # singular values of unfoldings
]:
    rtol = 0.0 if rtol is None else rtol
    atol = 0.0 if atol is None else atol

    bases = []
    singular_values_of_unfoldings = []
    C = T
    for ii in range(len(T.shape)):
        C_swap = C.swapaxes(ii,0)
        old_shape_swap = C_swap.shape

        C_swap_mat = C_swap.reshape((old_shape_swap[0], -1))
        U, ss, Vt = jnp.linalg.svd(C_swap_mat, full_matrices=False)

        if max_ranks is not None:
            ss = ss[:max_ranks[ii]]

        tol = jnp.maximum(ss[0] * rtol, atol)

        rM_new = jnp.sum(ss >= tol)
        U = U[:, :rM_new]
        ss = ss[:rM_new]
        Vt = Vt[:rM_new, :]

        singular_values_of_unfoldings.append(ss)
        bases.append(U.T)
        C_swap = (ss.reshape((-1,1)) * Vt).reshape((rM_new,) + old_shape_swap[1:])
        C = C_swap.swapaxes(0, ii)

    return (tuple(bases), C), tuple(singular_values_of_unfoldings)


def t3_svd_dense(
        T: jnp.ndarray, # shape=(N1, N2, .., Nd)
        max_mid_tt_ranks:       typ.Sequence[int] = None, # len=k-1, i.e., Correct: (r1, ..., r_{k-1}), Incorrect: (1,r1, ..., r_{k-1},1)
        max_basis_ranks:        typ.Sequence[int] = None, # len=k,
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    typ.Tuple[jnp.ndarray,...], # basis singular values, len=k
    typ.Tuple[jnp.ndarray,...], # tt singular values, len=k-1
]:
    (basis_cores, tucker_core), ss_tucker = tucker_svd_dense(T, max_basis_ranks, rtol=rtol, atol=atol)
    tt_cores, ss_tt = tt_svd_dense(tucker_core, max_mid_tt_ranks, rtol=rtol, atol=atol)
    return (basis_cores, tt_cores), ss_tucker, ss_tt


def t3_save(
        file,
        cores: TuckerTensorTrain,
):
    t3_check_correctness(cores)
    basis_cores, tt_cores = cores
    cores_dict = {'basis_cores_'+str(ii): basis_cores[ii] for ii in range(len(basis_cores))}
    cores_dict.update({'tt_cores_'+str(ii): tt_cores[ii] for ii in range(len(tt_cores))})

    try:
        np.savez(file, **cores_dict)
    except RuntimeError:
        print('Failed to save TuckerTensorTrain to file')


def t3_load(
        file,
) -> TuckerTensorTrain:
    try:
        d = np.load(file)
    except RuntimeError:
        print('Failed to load TuckerTensorTrain from file')

    assert (len(d.files) % 2 == 0)
    num_cores = len(d.files) // 2
    basis_cores = [d['basis_cores_' + str(ii)] for ii in range(num_cores)]
    tt_cores = [d['tt_cores_' + str(ii)] for ii in range(num_cores)]
    return tuple(basis_cores), tuple(tt_cores)


