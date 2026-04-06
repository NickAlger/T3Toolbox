



# # # # BAD DON'T USE


###############    UNIFORM TUCKER TENSOR TRAIN    ###############

def pack_matrices(
        MM: typ.Sequence[jnp.ndarray], # elm_shapes=[(n0, N0), (n1, N1), ..., (nk,Nk)]
) -> jnp.ndarray: # uniform_cores,   shape=(num_cores, n, N)
    for M in MM:
        assert(len(M.shape) == 2)

    n = np.max([M.shape[0] for M in MM])
    N = np.max([M.shape[1] for M in MM])

    padded_MM_list = []
    for M in MM:
        n0, N0 = M.shape
        pad = [(0, n - n0, 0), (0, N - N0, 0)]
        padded_M = jax.lax.pad(M, 0.0, pad)
        padded_MM_list.append(padded_M)

    uniform_MM = jnp.stack(padded_MM_list)
    return uniform_MM

def unpack_matrices(
        uniform_MM:     jnp.ndarray, # shape=(d, n, N)
        nn:             typ.Sequence[int], # len=d, (n1, ..., nd)
        NN:             typ.Sequence[int], # len=d, (N1, ..., Nd)
) -> typ.Tuple[jnp.ndarray,...]: # cores, len=d, elm_shape=(ni, Ni)
    d, n, N = uniform_MM.shape
    MM_list = []
    for ii, M in enumerate(uniform_MM):
        n0 = int(nn[ii])
        N0 = int(NN[ii])
        pad = [(0, n0 - n, 0), (0, N0 - N, 0)]
        M0 = jax.lax.pad(M, 0.0, pad)
        MM_list.append(M0)
    return tuple(MM_list)


UniformT3Variations = typ.Tuple[
    jnp.ndarray,  # basis_variations, shape=(d, n, N)
    jnp.ndarray,  # tt_variations, shape=(d, r, n, r)
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UniformT3TangentSpace:
    orthogonal_basis_cores:     jnp.ndarray # shape=(d, n, N)
    left_orthogonal_tt_cores:   jnp.ndarray # shape=(d, r, n, r)
    right_orthogonal_tt_cores:  jnp.ndarray # shape=(d, r, n, r)
    up_orthogonal_tt_cores:     jnp.ndarray # shape=(d, r, n, r)
    original_shape:         typ.Tuple[int, ...] # len=d
    original_tucker_ranks:  typ.Tuple[int, ...]  # len=d
    original_tt_ranks:      typ.Tuple[int, ...]  # len=d+1

    def __post_init__(me):
        assert(me.orthogonal_basis_cores.shape == (me.d, me.n, me.N))
        assert(me.left_orthogonal_tt_cores.shape == (me.d, me.r, me.n, me.r))
        assert(me.right_orthogonal_tt_cores.shape == (me.d, me.r, me.n, me.r))
        assert(me.up_orthogonal_tt_cores.shape == (me.d, me.r, me.n, me.r))
        assert(len(me.original_shape) == me.d)
        assert(len(me.original_tucker_ranks) == me.d)
        assert(len(me.original_tt_ranks) == me.d+1)

    @ft.cached_property
    def d(me):
        return me.orthogonal_basis_cores.shape[0]

    @ft.cached_property
    def n(me):
        return me.orthogonal_basis_cores.shape[1]

    @ft.cached_property
    def r(me):
        return me.left_orthogonal_tt_cores.shape[1]

    @ft.cached_property
    def N(me):
        return me.orthogonal_basis_cores.shape[2]

    @staticmethod
    def from_nonuniform(TS: T3TangentSpace) -> 'UniformT3TangentSpace':
        orthogonal_basis_cores = pack_matrices(TS.orthogonal_basis_cores)
        left_orthogonal_tt_cores = pack_uniform_tensor_train(TS.left_orthogonal_tt_cores)
        right_orthogonal_tt_cores = pack_uniform_tensor_train(TS.right_orthogonal_tt_cores)
        up_orthogonal_tt_cores = pack_uniform_tensor_train(TS.up_orthogonal_tt_cores)

        return UniformT3TangentSpace(
            orthogonal_basis_cores,
            left_orthogonal_tt_cores,
            right_orthogonal_tt_cores,
            up_orthogonal_tt_cores,
            TS.shape, TS.tucker_ranks, TS.tt_ranks,
        )

    def to_nonuniform(me) -> T3TangentSpace:
        orthogonal_basis_cores = unpack_matrices(
            me.orthogonal_basis_cores, me.original_tucker_ranks, me.original_shape,
        )
        left_orthogonal_tt_cores = unpack_uniform_tensor_train(
            me.left_orthogonal_tt_cores, me.original_tt_ranks, me.original_tucker_ranks,
        )
        right_orthogonal_tt_cores = unpack_uniform_tensor_train(
            me.right_orthogonal_tt_cores, me.original_tt_ranks, me.original_tucker_ranks,
        )
        up_orthogonal_tt_cores = unpack_uniform_tensor_train(
            me.up_orthogonal_tt_cores, me.original_tt_ranks, me.original_tucker_ranks,
        )
        return T3TangentSpace(
            orthogonal_basis_cores,
            left_orthogonal_tt_cores,
            right_orthogonal_tt_cores,
            up_orthogonal_tt_cores,
        )

    def pack_uniform_variations(
            me,
            u: T3Variations,
    ) -> UniformT3Variations:
        assert(len(u[1]) == me.d)
        for ii, dU in enumerate(u[1]):
            ni = me.original_tucker_ranks[ii]
            rL = me.original_tt_ranks[ii]
            rR = me.original_tt_ranks[ii+1]
            assert(dU.shape == (rL, ni, rR))

        assert(len(u[0]) == me.d)
        for ii, dB in enumerate(u[0]):
            ni = me.original_tucker_ranks[ii]
            Ni = me.original_shape[ii]
            assert(dB.shape == (ni, Ni))

        uniform_dB = pack_matrices(u[0])
        uniform_dU = pack_uniform_tensor_train(u[1])

        return uniform_dB, uniform_dU

    def unpack_uniform_variations(
            me,
            U: UniformT3Variations,
    ) -> T3Variations:
        basis_variations = unpack_matrices(U[0], me.original_tucker_ranks, me.original_shape)
        tt_variations = unpack_uniform_tensor_train(U[1], me.original_tt_ranks, me.original_tucker_ranks)
        return basis_variations, tt_variations

    @ft.cached_property
    def data(me):
        return (
            me.orthogonal_basis_cores,
            me.left_orthogonal_tt_cores,
            me.right_orthogonal_tt_cores,
            me.up_orthogonal_tt_cores,
            me.original_shape,
            me.original_tucker_ranks,
            me.original_tt_ranks,
        )

    @ft.cached_property
    def static_data(me):
        return (
            me.original_shape,
            me.original_tucker_ranks,
            me.original_tt_ranks,
        )

    @ft.cached_property
    def traced_data(me):
        return (
            me.orthogonal_basis_cores,
            me.left_orthogonal_tt_cores,
            me.right_orthogonal_tt_cores,
            me.up_orthogonal_tt_cores,
        )

    @staticmethod
    def from_static_and_traced_data(
        static_data, traced_data,
    ):
        data = traced_data + static_data
        return UniformT3TangentSpace(*data)

    def tree_flatten(me):
        return (me.data, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def ut3_orthogonal_gauge_projection(
        U: UniformT3Variations,
        UTS: UniformT3TangentSpace,
) -> UniformT3Variations:
    '''Makes variations left-perpendicular by orthogonally projecting away the parallel components.
    Changes tangent vector.
    dV_L -> (I - P_L P_L^T) dV_L
    '''
    basis_variations, tt_variations = U

    d, r, n, _ = tt_variations.shape
    N = basis_variations.shape[-1]
    assert(tt_variations.shape == (d, r, n, r))
    assert(basis_variations.shape == (d, n, N))
    assert(UTS.N == N)
    assert(UTS.d == d)
    assert(UTS.r == r)
    assert(UTS.n == n)

    gauged_tt_variations_list = []
    for ii in range(d-1):
        P = UTS.left_orthogonal_tt_cores[ii, :, :, :]
        dV = tt_variations[ii,:,:,:]
        dV2 = dV - jnp.einsum('iaj,jk->iak', P, jnp.einsum('iaj,iak->jk', P, dV))
        gauged_tt_variations_list.append(dV2)
    gauged_tt_variations_list.append(tt_variations[-1,:,:,:])

    gauged_tt_variations = jnp.stack(gauged_tt_variations_list)

    gauged_basis_variations_list = []
    for ii in range(d):
        B = UTS.orthogonal_basis_cores[ii, :, :]
        dB = basis_variations[ii,:,:]
        dB2 = dB - (dB @ B.T) @ B
        gauged_basis_variations_list.append(dB2)

    gauged_basis_variations = jnp.stack(gauged_basis_variations_list)

    return gauged_basis_variations, gauged_tt_variations



@jax.jit
def ut3_orthogonal_gauge_projection_using_map(
        U: UniformT3Variations,
        orthogonal_basis_cores,
        left_orthogonal_tt_cores,
) -> UniformT3Variations:
    '''Makes variations left-perpendicular by orthogonally projecting away the parallel components.
    Changes tangent vector.
    dV_L -> (I - P_L P_L^T) dV_L
    '''
    basis_variations, tt_variations = U

    first_gauged_tt_variations = jax.lax.map(
        lambda P_dV: P_dV[1] - jnp.einsum('iaj,jk->iak', P_dV[0], jnp.einsum('iaj,iak->jk', P_dV[0], P_dV[1])),
        (left_orthogonal_tt_cores[:-1], tt_variations[:-1]),
    )
    last_gauged_tt_variation = tt_variations[-1,:,:,:]
    gauged_tt_variations = jnp.concatenate(
        [first_gauged_tt_variations, last_gauged_tt_variation.reshape((1,) + last_gauged_tt_variation.shape)],
        axis=0
    )

    gauged_basis_variations = jax.lax.map(
        lambda B_dB: B_dB[1] - (B_dB[1] @ B_dB[0].T) @ B_dB[0], (orthogonal_basis_cores, basis_variations),
    )

    return gauged_basis_variations, gauged_tt_variations


def get_t3_tangent_space_dim(
        shape: typ.Sequence[int], # len=d
        tucker_ranks: typ.Sequence[int], # len=d
        tt_ranks: typ.Sequence[int], # len=d+1
) -> int:
    num_cores = len(shape)
    assert(len(tucker_ranks) == num_cores)
    assert(len(tt_ranks) == num_cores+1)
    manifold_dim: int = 0
    for ii in range(num_cores):
        n = tucker_ranks[ii]
        rL = tt_ranks[ii]
        rR = tt_ranks[ii+1]
        if ii == num_cores-1:
            manifold_dim += rL * n * rR
        else:
            manifold_dim += (rL * n - rR) * rR

    for ii in range(num_cores):
        n = tucker_ranks[ii]
        N = shape[ii]
        manifold_dim += (N - n) * n

    return manifold_dim
