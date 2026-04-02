import numpy as np
import jax
import jax.numpy as jnp
import typing as typ


__all__ = [
    'TensorTrain',
    'tt_get_ranks',
    'tt_get_shape',
    'tt_reverse',
    'tt_check_correctness',
    'tt_to_dense',
    'left_orthogonalize_tt',
    'right_orthogonalize_tt',
    'tt_svd',
    'tt_zipper_left_to_right',
    'tt_zipper_right_to_left',
    'tt_inner_product',
    'tt_norm',
    'tt_add',
    'tt_scale',
    'tt_sub',
    'tt_zeros',
    'tt_pad_rank',
    'tt_pad_rank_and_shape',
    'tt_svd_dense',
    'tt_eval_entry',
    'pack_uniform_tensor_train',
    'unpack_uniform_tensor_train',
    'utt_reverse',
    'canonical_to_tensor_train',
    'tt_relative_fro_error',
    'tt_save',
    'tt_load',
    'load_array_sequence',
    'tt_unfolding_condition_numbers',
    'left_orthogonalize_one_core_in_tt',
]

#########################################################################
########################    Tensor train (TT)    ########################
#########################################################################

TensorTrain = typ.Tuple[jnp.ndarray, ...] # tt_cores, len=d, elm_shape=(ri, ni, r(i+1))


########    Basic operations    ########

def tt_get_ranks(cores: typ.Sequence[jnp.ndarray]) -> typ.Tuple[int, ...]:
    return tuple([int(cores[0].shape[0])] + [int(G.shape[2]) for G in cores])


def tt_get_shape(cores: typ.Sequence[jnp.ndarray]) -> typ.Tuple[int, ...]:
    return tuple([G.shape[1] for G in cores])


def tt_reverse(cores):
    return tuple([G.swapaxes(0, 2) for G in cores[::-1]])


def tt_check_correctness(cores: typ.Sequence[jnp.ndarray]):
    for G in cores:
        assert(len(G.shape) == 3)

    ranks = tt_get_ranks(cores)
    ranks2 = tt_get_ranks(tt_reverse(cores))[::-1]
    assert(ranks == ranks2)


def tt_to_dense(cores: typ.Sequence[jnp.ndarray]) -> jnp.ndarray:
    G = cores[0]
    rL, n, rR = G.shape
    T = G.reshape((n, rR))
    for G in cores[1:-1]:
        T = np.tensordot(T, G, axes=1)
    G = cores[-1]
    rL, n, rR = G.shape
    T = np.tensordot(T, G.reshape((rL, n)), axes=1)
    return T


def tt_eval_entry(
        cores: typ.Sequence[jnp.array],
        inds: typ.Sequence[int],
):
    assert(len(cores) == len(inds))
    mats = [G[:,ind,:] for G, ind in zip(cores, inds)]
    entry = mats[0]
    for M in mats[1:]:
        entry = entry @ M
    return float(entry)

########    Orthogonalization    ########

def _left_orthogonalize_pair_of_cores(
        A0: jnp.ndarray, # shape=(rL, n1, rM)
        B0: jnp.ndarray, # shape=(rM, n2, rR)
        orthogonalize_with_svd: bool = True,
) -> typ.Tuple[
    jnp.ndarray, # A. shape=(rL, n1, rM'), left orthogonal
    jnp.ndarray, # B. shape=(rM', n2, rR)
]:
    rL, n1, rM0 = A0.shape
    _, n2, rR = B0.shape

    if orthogonalize_with_svd:
        Q, ss, Vt = jnp.linalg.svd(A0.reshape((rL * n1, rM0)), full_matrices=False)
        R = ss.reshape((-1,1)) * Vt
    else:
        Q, R = jnp.linalg.qr(A0.reshape((rL * n1, rM0)), mode='reduced')

    rM = Q.shape[1]
    A = Q.reshape((rL, n1, rM))
    B = (R @ B0.reshape((rM0, n2 * rR))).reshape((rM, n2, rR))
    return A, B


def _right_orthogonalize_pair_of_cores(
        A0: jnp.ndarray, # shape=(rL, n1, rM)
        B0: jnp.ndarray, # shape=(rM, n2, rR)
        orthogonalize_with_svd: bool = True,
) -> typ.Tuple[
    jnp.ndarray, # A. shape=(rL, n1, rM')
    jnp.ndarray, # B. shape=(rM', n2, rR), right orthogonal
]:
    Bswap, Aswap = _left_orthogonalize_pair_of_cores(
        B0.swapaxes(0,2),
        A0.swapaxes(0,2),
        orthogonalize_with_svd=orthogonalize_with_svd,
    )
    return Aswap.swapaxes(0,2), Bswap.swapaxes(0,2)


def left_orthogonalize_one_core_in_tt(
        ind: int,
        cores: typ.Sequence[jnp.ndarray],
        orthogonalize_with_svd: bool = True,
) -> typ.Tuple[jnp.ndarray]:
    assert(ind+1 < len(cores))
    new_cores = [G for G in cores]
    new_cores[ind], new_cores[ind+1] = _left_orthogonalize_pair_of_cores(
        cores[ind],
        cores[ind+1],
        orthogonalize_with_svd=orthogonalize_with_svd,
    )
    return tuple(new_cores)


def right_orthogonalize_one_core_in_tt(
        ind: int,
        cores: typ.Sequence[jnp.ndarray],
        orthogonalize_with_svd: bool = True,
) -> typ.Tuple[jnp.ndarray]:
    assert(0 <= ind - 1)
    new_cores = [G for G in cores]
    new_cores[ind-1], new_cores[ind] = _right_orthogonalize_pair_of_cores(
        cores[ind-1],
        cores[ind],
        orthogonalize_with_svd=orthogonalize_with_svd,
    )
    return tuple(new_cores)


def left_orthogonalize_tt(
        cores: typ.Sequence[jnp.ndarray],
        orthogonalize_with_svd: bool = True,
):
    for ii in range(len(cores)-1):
        cores = left_orthogonalize_one_core_in_tt(
            ii,
            cores,
            orthogonalize_with_svd=orthogonalize_with_svd,
        )
    return cores


def right_orthogonalize_tt(
        cores: typ.Sequence[jnp.ndarray],
        orthogonalize_with_svd: bool = True,
):
    return tt_reverse(left_orthogonalize_tt(
        tt_reverse(cores),
        orthogonalize_with_svd=orthogonalize_with_svd,
    ))


########    Add/subtract/scale    ########

def tt_add(
        coresA: typ.Sequence[jnp.ndarray],
        coresB: typ.Sequence[jnp.ndarray],
) -> typ.Tuple[jnp.ndarray,...]:
    assert(tt_get_shape(coresA) == tt_get_shape(coresB))
    coresAB = []

    GAB = jnp.concatenate([coresA[0], coresB[0]], axis=2)
    coresAB.append(GAB)

    for GA, GB in zip(coresA[1:-1], coresB[1:-1]):
        Z12 = jnp.zeros((GA.shape[0], GA.shape[1], GB.shape[2]))
        Z21 = jnp.zeros((GB.shape[0], GA.shape[1], GA.shape[2]))
        GAB = jnp.concatenate([jnp.concatenate([GA, Z12], axis=2),
                               jnp.concatenate([Z21, GB], axis=2)], axis=0)
        coresAB.append(GAB)

    GAB = jnp.concatenate([coresA[-1], coresB[-1]], axis=0)
    coresAB.append(GAB)
    return tuple(coresAB)


def tt_scale(
        s,
        cores: typ.Sequence[jnp.ndarray],
) -> typ.Tuple[jnp.ndarray,...]:
    return tuple(cores[:-1]) + (s*cores[-1],)


def tt_sub(
        coresA: typ.Sequence[jnp.ndarray],
        coresB: typ.Sequence[jnp.ndarray],
) -> typ.Tuple[jnp.ndarray,...]:
    return tt_add(coresA, tt_scale(-1.0, coresB))


########    Zipper matrices    ########

def tt_zipper_left_to_right(
        coresA: typ.Sequence[jnp.ndarray],
        coresB: typ.Sequence[jnp.ndarray],
) -> typ.Tuple[jnp.ndarray,...]: # zipper_matrices. len=num_cores+1
    zipper_matrices = [jnp.array([[1.0]])]
    for GA, GB in zip(coresA, coresB):
        Z_prev = zipper_matrices[-1]
        Z = jnp.einsum('ij,iak,jal->kl', Z_prev, GA, GB)
        zipper_matrices.append(Z)
    return tuple(zipper_matrices)


def tt_zipper_right_to_left(
        coresA: typ.Sequence[jnp.ndarray],
        coresB: typ.Sequence[jnp.ndarray],
) -> typ.Tuple[jnp.ndarray,...]: # zipper_matrices. len=num_cores+1
    return tt_zipper_left_to_right(tt_reverse(coresA), tt_reverse(coresB))[::-1]


########    Inner product / Norm    ########

def tt_inner_product(
        coresA: typ.Sequence[jnp.ndarray],
        coresB: typ.Sequence[jnp.ndarray],
):
    return tt_zipper_left_to_right(coresA, coresB)[-1][0,0]


def tt_norm(
        cores: typ.Sequence[jnp.ndarray],
):
    return jnp.linalg.norm(left_orthogonalize_tt(cores)[-1])


########    Zeros    ########

def tt_zeros(
        shape: typ.Sequence[int],
        ranks: typ.Sequence[int],
) -> typ.Tuple[jnp.ndarray,...]:
    num_cores = len(shape)
    assert(len(ranks) == num_cores+1)
    return tuple([jnp.zeros((ranks[ii], shape[ii], ranks[ii+1])) for ii in range(num_cores)])


########    Pad rank    ########

def tt_pad_rank(
        cores: typ.Sequence[jnp.ndarray],
        new_ranks: typ.Sequence[int],
) -> typ.Tuple[jnp.ndarray]:
    num_cores = len(cores)
    old_ranks = tt_get_ranks(cores)
    assert(len(old_ranks) == len(new_ranks))
    delta_ranks = [r_new - r_old for r_new, r_old in zip(new_ranks, old_ranks)]

    new_cores = []
    for ii in range(num_cores):
        new_cores.append(jnp.pad(cores[ii], ((0,delta_ranks[ii]), (0,0), (0,delta_ranks[ii+1]))))
    return tuple(new_cores)


def tt_pad_rank_and_shape(
        cores: typ.Sequence[jnp.ndarray],
        new_ranks: typ.Sequence[int],
        new_shape: typ.Sequence[int],
) -> typ.Tuple[jnp.ndarray]:
    num_cores = len(cores)
    old_ranks = tt_get_ranks(cores)
    old_shape = tt_get_shape(cores)
    assert(len(old_ranks) == len(new_ranks))
    assert(len(old_shape) == len(new_shape))
    delta_ranks = [r_new - r_old for r_new, r_old in zip(new_ranks, old_ranks)]
    delta_shape = [n_new - n_old for n_new, n_old in zip(new_shape, old_shape)]

    new_cores = []
    for ii in range(num_cores):
        new_cores.append(jnp.pad(cores[ii], ((0,delta_ranks[ii]), (0,delta_shape[ii]), (0,delta_ranks[ii+1]))))
    return tuple(new_cores)


########    TT-SVD    ########

def tt_svd(
        cores: typ.Sequence[typ.Union[jnp.ndarray, np.ndarray]], # elm_shapes=[(1,n1,r1), (r1,n2,r2), ..., (r(k-1),nk,1)]
        max_mid_ranks: typ.Sequence[int] = None, # len=k-1, i.e., Correct: (r1, ..., r_{k-1}), Incorrect: (1,r1, ..., r_{k-1},1)
        rtol: float = None,
        atol: float = None,
        use_numpy: bool = False,
) -> typ.Tuple[
    typ.Tuple[jnp.ndarray,...], # new_cores, elm_shapes=[(r0',n1,r1'), (r1',n2,r2'), ..., (r(k-1)',nk,rk')]
    typ.Tuple[jnp.ndarray,...], # singular_values_of_unfoldings, elm_shapes=[(r0',), (r1',), ..., (rk',)]
]:
    if use_numpy:
        xnp = np
    else:
        xnp = jnp

    if max_mid_ranks is not None:
        assert(len(max_mid_ranks) == len(cores)-1)
    rtol = 0.0 if rtol is None else rtol
    atol = 0.0 if atol is None else atol

    singular_values_of_unfoldings = []
    new_cores = list(right_orthogonalize_tt(cores))
    singular_values_of_unfoldings.append(xnp.array([xnp.linalg.norm(new_cores[0])]))
    for ii in range(len(cores)-1):
        A = new_cores[ii]
        B = new_cores[ii+1]
        rL, n1, rM = A.shape
        _, n2, rR = B.shape
        assert (B.shape[0] == rM)
        U, ss, Vt = xnp.linalg.svd(A.reshape((rL * n1, rM)), full_matrices=False)
        if max_mid_ranks is not None:
            ss = ss[:max_mid_ranks[ii]]

        tol = xnp.maximum(ss[0] * rtol, atol)

        rM_new = xnp.sum(ss >= tol)
        U = U[:, :rM_new]
        ss = ss[:rM_new]
        Vt = Vt[:rM_new, :]

        singular_values_of_unfoldings.append(ss)
        new_cores[ii] = U.reshape((rL, n1, rM_new))
        new_cores[ii+1] = xnp.einsum('ij,jak->iak', xnp.diag(ss) @ Vt, B)

    singular_values_of_unfoldings.append(xnp.array([xnp.linalg.norm(new_cores[-1])]))

    return tuple(new_cores), tuple(singular_values_of_unfoldings)


def tt_svd_dense(
        T: jnp.ndarray,
        max_mid_ranks: typ.Sequence[int] = None, # len=k-1, i.e., Correct: (r1, ..., r_{k-1}), Incorrect: (1,r1, ..., r_{k-1},1)
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[jnp.ndarray,...], # tt_cores
    typ.Tuple[jnp.ndarray,...], # singular values of unfoldings
]:
    nn = T.shape
    rtol = 0.0 if rtol is None else rtol
    atol = 0.0 if atol is None else atol

    X = T.reshape((1,) + T.shape)
    singular_values_of_unfoldings = []
    cores = []
    for ii in range(len(nn)-1):
        rL = X.shape[0]
        U, ss, Vt = jnp.linalg.svd(X.reshape((rL * nn[ii], -1)), full_matrices=False)

        if max_mid_ranks is not None:
            ss = ss[:max_mid_ranks[ii]]

        tol = jnp.maximum(ss[0] * rtol, atol)

        rR = jnp.sum(ss >= tol)
        U = U[:, :rR]
        ss = ss[:rR]
        Vt = Vt[:rR, :]

        singular_values_of_unfoldings.append(ss)
        cores.append(U.reshape((rL, nn[ii], rR)))
        X = ss.reshape((-1,1)) * Vt
    cores.append(X.reshape(X.shape + (1,)))

    return tuple(cores), tuple(singular_values_of_unfoldings)


def tt_unfolding_condition_numbers(
        cores: typ.Sequence[jnp.ndarray],
) -> jnp.array: # len=num_cores+1
    all_ss = tt_svd(cores)[1]
    return jnp.array([jnp.max(ss) / jnp.min(ss) for ss in all_ss])


#

@jax.jit
def canonical_to_tensor_train(
        factors: typ.Sequence[jnp.ndarray], # canonical factors, len=dim, ith_elm_shape=(num_terms, ni)
) -> typ.Tuple[jnp.array,...]: # cores
    num_terms, n0 = factors[0].shape
    G0 = factors[0].swapaxes(0,1).reshape((1, n0, num_terms))
    cores_list = [G0]
    for F in factors[1:-1]:
        G = jax.lax.map(jnp.diag, F.T).swapaxes(0,1)
        cores_list.append(G)
    nl = factors[-1].shape[1]
    G_last = factors[-1].reshape((num_terms, nl, 1))
    cores_list.append(G_last)
    return tuple(cores_list)


def tt_relative_fro_error(
        cores:      typ.Sequence[jnp.ndarray],
        true_cores: typ.Sequence[jnp.ndarray],
):
    return tt_norm(tt_sub(true_cores, cores)) / tt_norm(true_cores)


def tt_save(filename, cores: typ.Sequence[jnp.ndarray]):
    np.savez(filename, *cores)


def load_array_sequence(filename) -> typ.Tuple[jnp.ndarray]:
    d = np.load(filename, allow_pickle=False)
    arrays = []
    for ii in range(len(d)):
        arrays.append(jnp.array(d['arr_'+str(ii)]))
    return tuple(arrays)


tt_load = load_array_sequence


############################################################################
########################    Uniform Tensor train    ########################
############################################################################

def utt_reverse(
        uniform_cores: jnp.ndarray, # shape=(num_cores, r, N, r)
) -> jnp.ndarray: # shape=(num_cores, r, N, r)
    return uniform_cores[::-1, :, :, :].swapaxes(1, 3)


def pack_uniform_tensor_train(
        cores: typ.Sequence[jnp.ndarray], # elm_shapes=[(1,N0,r1), (r1,N1,r2), (r2,N1,r3), ..., (rk,Nk,1)]
) -> jnp.ndarray: # uniform_cores,   shape=(num_cores, r, N, r)
    '''Forms a uniform tensor train from a list of 3-tensor cores, padding by zero as needed.'''
    tt_check_correctness(cores)

    N = np.max([G.shape[1] for G in cores])
    r = np.max(tt_get_ranks(cores))

    padded_cores_list = []
    for G in cores:
        rL, N0, rR = G.shape
        pad = [(0, r - rL, 0), (0, N - N0, 0), (0, r - rR, 0)]
        padded_core = jax.lax.pad(G, 0.0, pad)
        padded_cores_list.append(padded_core)

    uniform_cores = jnp.stack(padded_cores_list)
    return uniform_cores


def unpack_uniform_tensor_train(
        uniform_cores:  jnp.ndarray, # shape=(d, r, N, r)
        ranks:          typ.Sequence[int], # len=d+1, (r1, r2, ..., r(d+1))
        shape:          typ.Sequence[int], # len=d, (n1, n2, ..., nd)
) -> typ.Tuple[jnp.ndarray,...]: # cores, len=d, elm_shape=(ri, ni, r(i+1))
    d, r, N, r = uniform_cores.shape
    cores_list = []
    for ii, G in enumerate(uniform_cores):
        rL = ranks[ii]
        n = shape[ii]
        rR = ranks[ii+1]
        pad = [(0, int(rL - r), 0), (0, int(n - N), 0), (0, int(rR - r), 0)]
        core = jax.lax.pad(G, 0.0, pad)
        cores_list.append(core)
    return tuple(cores_list)
