# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform as ut3
import t3toolbox.t3svd as t3svd
import t3toolbox.base_variation_format as bvf
# from t3toolbox.common import jnp, NDArray, numpy_scan, jax_scan
import t3toolbox.common as common
from t3toolbox.common import *


__all__ = [
    'uniform_tangent_to_uniform_t3',
    'uniform_tangent_to_dense',
    # 'ut3_retract',
    # 'ut3_project_dense_tensor_onto_tangent_space',
    # 'ut3_orthogonal_gauge_projection',
]


############################################################
########    Uniform Tucker Tensor Train Manifold    ########
############################################################

def uniform_tangent_to_uniform_t3(
        variations: ut3.UniformT3Variation,
        base: ut3.UniformT3Base,
        bv_masks: ut3.UniformBVEdgeWeights,
        include_shift: bool = False,
        use_jax: bool = False,
) -> typ.Tuple[
    ut3.UniformTuckerTensorTrain,
    ut3.UniformEdgeWeights, # masks
]:
    '''Rank 2r Tucker tensor train representation of tangent vector:
            u(x,y,z,w) = ([dG1(B x) L1(B x)]) ([R2(B y)        0]) ([R3(B z)        0]) ([R4(B w) ])
                         (                  ) ([dG2(B y) L2(B y)]) ([dG3(B z) L3(B z)]) ([dG4(B w)])
                         (         +        ) (         +        ) (        +         ) (    +     )
                         ([O1(dB x)       0]) ([0              0]) ([0              0]) ([0       ])
                         (                  ) ([O2(dB y)       0]) ([O3(dB z)       0]) ([O4(dB w)])

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.uniform as ut3
    >>> import t3toolbox.uniform_manifold as utm
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.t3svd as t3svd
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (5,3,2,4)))
    >>> p, _, _ = t3svd.t3_svd(p)
    >>> base, dummy_var = orth.orthogonal_representations(p)
    >>> v = t3m.tangent_randn(base)
    >>> uniform_v, uniform_base, bv_mask = ut3.bv_to_ubv(v, base)
    >>> dense_v = t3m.tangent_to_dense(v, base)
    >>> x_ut3, masks_ut3 = utm.uniform_tangent_to_uniform_t3(uniform_v, uniform_base, bv_mask)
    >>> dense_uniform_v = ut3.ut3_to_dense(x_ut3, masks_ut3)
    >>> print(np.linalg.norm(dense_v - dense_uniform_v))
    4.72221182491572e-14
    '''
    xnp, _, _ = get_backend(True, use_jax)

    #

    tucker_variations, tt_variations = variations

    up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base

    d, N, nU, nO, rL, rR = ut3.get_uniform_base_structure(base)

    # d, n, N = tucker_variations.shape
    # r = tt_variations.shape[1]

    tangent_tucker_cores = xnp.concatenate([up_tucker_cores, tucker_variations], axis=1)

    dG = tt_variations
    O = outer_tt_cores
    L = left_tt_cores
    R = right_tt_cores
    # Z = xnp.zeros((d, r, n, r))
    Z000 = xnp.zeros((d, rL, nU, rL))
    Z001 = xnp.zeros((d, rL, nU, rR))
    Z010 = xnp.zeros((d, rL, nO, rL))
    Z011 = xnp.zeros((d, rL, nO, rR))
    Z100 = xnp.zeros((d, rR, nU, rL))
    Z101 = xnp.zeros((d, rR, nU, rR))
    Z110 = xnp.zeros((d, rR, nO, rL))
    Z111 = xnp.zeros((d, rR, nO, rR))

    first_tangent_tt_core = xnp.concatenate([
        xnp.concatenate([
            xnp.concatenate([dG[:1],    L[:1]], axis=3),
            xnp.concatenate([Z100[:1],  Z101[:1]], axis=3),
        ], axis=1),
        xnp.concatenate([
            xnp.concatenate([O[:1],     Z011[:1]], axis=3),
            xnp.concatenate([Z110[:1],  Z111[:1]], axis=3),
        ], axis=1)
    ], axis=2)

    mid_tangent_tt_cores = xnp.concatenate([
        xnp.concatenate([
            xnp.concatenate([R[1:-1],   Z001[1:-1]], axis=3),
            xnp.concatenate([dG[1:-1],  L[1:-1]], axis=3),
        ], axis=1),
        xnp.concatenate([
            xnp.concatenate([Z010[1:-1],    Z011[1:-1]], axis=3),
            xnp.concatenate([O[1:-1],       Z111[1:-1]], axis=3),
        ], axis=1)
    ], axis=2)

    if include_shift:
        last_tangent_tt_core = xnp.concatenate([
            xnp.concatenate([
                xnp.concatenate([R[-1:],            Z001[-1:]], axis=3),
                xnp.concatenate([L[-1:] + dG[-1:],  Z101[-1:]], axis=3),
            ], axis=1),
            xnp.concatenate([
                xnp.concatenate([Z010[-1:], Z011[-1:]], axis=3),
                xnp.concatenate([O[-1:],    Z111[-1:]], axis=3),
            ], axis=1)
        ], axis=2)
    else:
        last_tangent_tt_core = xnp.concatenate([
            xnp.concatenate([
                xnp.concatenate([R[-1:],  Z001[-1:]], axis=3),
                xnp.concatenate([dG[-1:], Z101[-1:]], axis=3),
            ], axis=1),
            xnp.concatenate([
                xnp.concatenate([Z010[-1:], Z011[-1:]], axis=3),
                xnp.concatenate([O[-1:],    Z111[-1:]], axis=3),
            ], axis=1)
        ], axis=2)

    tangent_tt_cores = xnp.concatenate(
        [first_tangent_tt_core, mid_tangent_tt_cores, last_tangent_tt_core],
        axis=0
    )

    shape_mask, up_mask, outer_mask, left_mask, right_mask = bv_masks

    left_mask_extended = xnp.concatenate([left_mask, xnp.ones((1,rL), dtype=bool)], axis=0) # len=d+1
    right_mask_extended = xnp.concatenate([xnp.ones((1,rR), dtype=bool), right_mask], axis=0) # len=d+1

    tucker_mask = xnp.concatenate([up_mask, outer_mask], axis=1)
    tt_mask = xnp.concatenate([left_mask_extended, right_mask_extended], axis=1)

    return (tangent_tucker_cores, tangent_tt_cores), (shape_mask, tucker_mask, tt_mask)


def uniform_tangent_to_dense(
        variations: ut3.UniformT3Variation,
        base: ut3.UniformT3Base,
        masks: ut3.UniformBVEdgeWeights,
        include_shift: bool = False,
        use_jax: bool = False,
) -> NDArray:
    """Convert uniform Tangent vector to dense tensor.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.uniform as ut3
    >>> import t3toolbox.uniform_manifold as utm
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.t3svd as t3svd
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (5,3,2,4)))
    >>> p, _, _ = t3svd.t3_svd(p)
    >>> base, dummy_var = orth.orthogonal_representations(p)
    >>> v = t3m.tangent_randn(base)
    >>> uniform_v, uniform_base, bv_mask = ut3.bv_to_ubv(v, base)
    >>> dense_v = t3m.tangent_to_dense(v, base)
    >>> dense_uniform_v = utm.uniform_tangent_to_dense(uniform_v, uniform_base, bv_mask)
    >>> print(np.linalg.norm(dense_v - dense_uniform_v))
    4.72221182491572e-14
    """
    xnp, _, _ = get_backend(True, use_jax=use_jax)

    x_ut3, masks_ut3 = uniform_tangent_to_uniform_t3(variations, base, masks, include_shift=include_shift, use_jax=use_jax)
    x_dense = ut3.ut3_to_dense(x_ut3, masks_ut3)
    return x_dense


if False:
    def ut3_attached_tangent_vector_to_ut3(
            variations: typ.Tuple[
                jnp.ndarray,  # basis_variations, shape=(d, n, N)
                jnp.ndarray,  # tt_variations, shape=(d, r, n, r)
            ],
            orthogonal_basis_cores: jnp.ndarray,  # shape=(d, n, N)
            left_orthogonal_tt_cores: jnp.ndarray,  # shape=(d, r, n, r)
            right_orthogonal_tt_cores: jnp.ndarray,  # shape=(d, r, n, r)
            up_orthogonal_tt_cores: jnp.ndarray,  # shape=(d, r, n, r)
    ) -> typ.Tuple[
        jnp.ndarray,  # tangent_basis_cores, shape=(d, 2n, N)
        jnp.ndarray,  # tangent_tt_cores, shape=(2, 2r, 2n, 2r)
    ]:
        '''Rank 2r Tucker tensor train representation of *attached* tangent vector:
                u(x,y,z,w) = ([dU1(B x) P1(B x)]) ([Q2(B y)        0]) ([Q3(B z)        0]) ([Q4(B w) ])
                             (                  ) ([dU2(B y) P2(B y)]) ([dU3(B z) P3(B z)]) ([P4(B w) + dU4(B w)])
                             (         +        ) (         +        ) (        +         ) (    +     )
                             ([R1(dB x)       0]) ([0              0]) ([0              0]) ([0       ])
                             (                  ) ([R2(dB y)       0]) ([R3(dB z)       0]) ([R4(dB w)])
        '''
        basis_variations, tt_variations = variations
        # (orthogonal_basis_cores,
        #  left_orthogonal_tt_cores,
        #  right_orthogonal_tt_cores,
        #  up_orthogonal_tt_cores,
        #  _, _,
        #  ) = base_representations

        d, n, N = basis_variations.shape
        r = tt_variations.shape[1]

        tangent_basis_cores = jnp.concatenate([orthogonal_basis_cores, basis_variations], axis=1)

        dU = tt_variations
        R = up_orthogonal_tt_cores
        P = left_orthogonal_tt_cores
        Q = right_orthogonal_tt_cores
        Z = jnp.zeros((d, r, n, r))

        first_tangent_tt_core = jnp.concatenate([
            jnp.concatenate([
                jnp.concatenate([dU[:1], P[:1]], axis=3),
                jnp.concatenate([Z[:1], Z[:1]], axis=3),
            ], axis=1),
            jnp.concatenate([
                jnp.concatenate([R[:1], Z[:1]], axis=3),
                jnp.concatenate([Z[:1], Z[:1]], axis=3),
            ], axis=1)
        ], axis=2)

        mid_tangent_tt_cores = jnp.concatenate([
            jnp.concatenate([
                jnp.concatenate([Q[1:-1], Z[1:-1]], axis=3),
                jnp.concatenate([dU[1:-1], P[1:-1]], axis=3),
            ], axis=1),
            jnp.concatenate([
                jnp.concatenate([Z[1:-1], Z[1:-1]], axis=3),
                jnp.concatenate([R[1:-1], Z[1:-1]], axis=3),
            ], axis=1)
        ], axis=2)

        last_tangent_tt_core = jnp.concatenate([
            jnp.concatenate([
                jnp.concatenate([Q[-1:], Z[-1:]], axis=3),
                jnp.concatenate([P[-1:] + dU[-1:], Z[-1:]], axis=3),
            ], axis=1),
            jnp.concatenate([
                jnp.concatenate([Z[-1:], Z[-1:]], axis=3),
                jnp.concatenate([R[-1:], Z[-1:]], axis=3),
            ], axis=1)
        ], axis=2)

        tangent_tt_cores = jnp.concatenate(
            [first_tangent_tt_core, mid_tangent_tt_cores, last_tangent_tt_core],
            axis=0
        )

        return tangent_basis_cores, tangent_tt_cores


    def ut3_retract(
            variations: typ.Tuple[
                jnp.ndarray, # basis_variations, shape=(d, n, N)
                jnp.ndarray, # tt_variations, shape=(d, r, n, r)
            ],
            orthogonal_basis_cores:     jnp.ndarray, # shape=(d, n, N)
            left_orthogonal_tt_cores:   jnp.ndarray, # shape=(d, r, n, r)
            right_orthogonal_tt_cores:  jnp.ndarray, # shape=(d, r, n, r)
            up_orthogonal_tt_cores:     jnp.ndarray, # shape=(d, r, n, r)
            doubled_rank_masks: typ.Tuple[
                jnp.ndarray,  # shape_mask, shape=(d, N)
                jnp.ndarray,  # tucker_cores_mask, shape=(d, 2*n)
                jnp.ndarray,  # tt_cores_mask, shape=(d+1, 2*r)
            ],  # use to specify ranks
    ) -> typ.Tuple[
        jnp.ndarray, # retracted_basis_cores, shape=(d, n, N)
        jnp.ndarray, # retracted_tt_cores, shape=(d, r, n, r)
    ]:
        basis_variations, tt_variations = variations

        d, n, N = basis_variations.shape
        r = tt_variations.shape[1]

        X = ut3_attached_tangent_vector_to_ut3(
            variations,
            orthogonal_basis_cores,
            left_orthogonal_tt_cores,
            right_orthogonal_tt_cores,
            up_orthogonal_tt_cores,
        )

        (basis_cores0, tt_cores0), _, _ = ut3_svd_masked(
            X,
            doubled_rank_masks,
            # basis_ranks, tt_ranks,
        )

        basis_cores = basis_cores0[:, :n, :]
        tt_cores = tt_cores0[:, :r, :n, :r]
        return basis_cores, tt_cores


    def ut3_project_dense_tensor_onto_tangent_space(
            T:                          jnp.ndarray, # shape=(N, N, ..., N)
            orthogonal_basis_cores:     jnp.ndarray, # shape=(d, n, N)
            left_orthogonal_tt_cores:   jnp.ndarray, # shape=(d, r, n, r)
            right_orthogonal_tt_cores:  jnp.ndarray, # shape=(d, r, n, r)
            up_orthogonal_tt_cores:     jnp.ndarray, # shape=(d, r, n, r)
    ):
        '''Very expensive, probably only useful for debugging other functions'''
        d, n, N = orthogonal_basis_cores.shape
        r = left_orthogonal_tt_cores.shape[1]

        XX = []
        for ii in range(d):
            X = T.reshape((1,) + T.shape + (1,))
            X = jnp.pad(
                X,
                [(0, r-1)] + [(0, 0)]*d + [(0, r-1)],
            )
            for jj in range(ii):
                B = orthogonal_basis_cores[jj]
                P = left_orthogonal_tt_cores[jj]
                X = jnp.einsum('ac,ec...->ea...', B, X)
                X = jnp.einsum('axb,ax...->b...', P, X)

            for jj in range(d-1, ii, -1):
                B = orthogonal_basis_cores[jj]
                Q = right_orthogonal_tt_cores[jj]
                X = jnp.einsum('bc,...ce->...be', B, X)
                X = jnp.einsum('...xb,axb->...a', X, Q)

            XX.append(X)

        BB_tilde = []
        GG_tilde = []
        for ii in range(d):
            X = XX[ii]
            B = orthogonal_basis_cores[ii]
            S = up_orthogonal_tt_cores[ii]
            G_tilde = jnp.einsum('aob,ko->akb', X, B)
            B_tilde = jnp.einsum('aob,akb->ko', X, S)
            BB_tilde.append(B_tilde)
            GG_tilde.append(G_tilde)

        ungauged_basis_variations = jnp.stack(BB_tilde)
        ungauged_tt_variations = jnp.stack(GG_tilde)
        ungauged_variations = (ungauged_basis_variations, ungauged_tt_variations)

        variations = ut3_orthogonal_gauge_projection_using_map(
            ungauged_variations,
            orthogonal_basis_cores,
            left_orthogonal_tt_cores,
        )
        return variations






if False:
    # # #


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


    def ut3_orthogonal_gauge_projection_for_loop(
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





