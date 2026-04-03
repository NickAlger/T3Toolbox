import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import functools as ft
import typing as typ

from .tt_basic_operations import *
from .t3_operations import *


__all__ = [
    'T3TangentSpace',
    'T3Variations',
    'UniformT3TangentSpace',
    'UniformT3Variations',
    'pack_matrices',
    'unpack_matrices',
    'ut3_orthogonal_gauge_projection',
    'get_t3_tangent_space_dim',
    'ut3_orthogonal_gauge_projection_using_map',
]

####################################################################
#################    Tensor train tangent space   ##################
####################################################################

T3Variations = typ.Tuple[
    typ.Tuple[jnp.ndarray,...],  # basis_variations
    typ.Tuple[jnp.ndarray,...],  # tt_variations
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class T3TangentSpace:
    '''Representation of a Tucker tensor train tangent vector.

    Example for 4-tensor:
        tt_variations               = (dV1, dV2, dV3, dV4)
        basis_variations            = (dB1, dB2, dB3, dB4)

        left_orthogonal_tt_cores    = (P1,   P2,  P3,  P4)
        right_orthogonal_tt_cores   = (Q1,   Q2,  Q3,  Q4)
        up_orthogonal_tt_cores      = (R1,   R2,  R3,  R4)
        orthogonal_basis_cores      = (B1,   B2,  B3,  B4)

        Tangent vector v:
        v(x,y,z,w) = (dV1(B1 x) + R1(dB1 x)) . Q2(B2 y) . Q3(B3 z) . Q4(B4 w)
                   + P1(B1 x) . (dV2(B2 y) + R2(dB2 y)) . Q3(B3 z) . Q4(B4 w)
                   + P1(B1 x) . P2(B2 y) . (dV3(B3 z) + R3(dB3 z)) . Q4(B4 w)
                   + P1(B1 x) . P2(B2 y) . P3(B3 z) . (dV4(B4 w) + R4(dB4 w))
    '''
    orthogonal_basis_cores:     typ.Tuple[jnp.ndarray, ...] # B_xo B_yo = I_xy    B.shape = (n, N)
    left_orthogonal_tt_cores:   typ.Tuple[jnp.ndarray, ...] # P_iax P_iay = I_xy, P.shape = (rL, n, rR)
    right_orthogonal_tt_cores:  typ.Tuple[jnp.ndarray, ...] # Q_xaj Q_yaj = I_xy  Q.shape = (rL, n, rR)
    up_orthogonal_tt_cores:     typ.Tuple[jnp.ndarray, ...] # R_ixj R_iyj = I_xy  R.shape = (rL, n, rR)


    def __post_init__(me):
        assert(me.tt_ranks == tt_get_ranks(me.left_orthogonal_tt_cores))
        assert(me.tt_ranks == tt_get_ranks(me.right_orthogonal_tt_cores))
        assert(me.tt_ranks == tt_get_ranks(me.up_orthogonal_tt_cores))

        assert(me.tucker_ranks == tt_get_shape(me.left_orthogonal_tt_cores))
        assert(me.tucker_ranks == tt_get_shape(me.right_orthogonal_tt_cores))
        assert(me.tucker_ranks == tt_get_shape(me.up_orthogonal_tt_cores))

        for n, N, B in zip(me.tucker_ranks, me.shape, me.orthogonal_basis_cores):
            assert(B.shape == (n, N))

    @ft.cached_property
    def shape(me) -> typ.Tuple[int,...]:
        return tuple([int(B.shape[1]) for B in me.orthogonal_basis_cores])

    @ft.cached_property
    def tucker_ranks(me) -> typ.Tuple[int,...]:
        return tuple([int(B.shape[0]) for B in me.orthogonal_basis_cores])

    @ft.cached_property
    def tt_ranks(me) -> typ.Tuple[int,...]:
        return tt_get_ranks(me.left_orthogonal_tt_cores)

    @ft.cached_property
    def num_cores(me) -> int:
        return len(me.shape)

    @staticmethod
    def make(
            x: TuckerTensorTrain,
    ) -> 'T3TangentSpace':
        t3_check_correctness(x)
        num_cores = len(t3_get_shape(x))

        left_orthogonal_t3, _, _ = t3_svd(x)

        orthogonal_basis_cores, left_orthogonal_tt_cores = left_orthogonal_t3

        # Sweep right to left
        right_orthogonal_tt_cores = list(left_orthogonal_tt_cores)
        up_orthogonal_tt_cores = [None] * num_cores
        for ii in range(num_cores-1,-1,-1):
            A0_a_i_b = right_orthogonal_tt_cores[ii-1]
            B0_b_j_c = right_orthogonal_tt_cores[ii]

            U_a_x_b, _, _ = outer_svd_3tensor(B0_b_j_c)
            up_orthogonal_tt_cores[ii] = U_a_x_b

            if ii > 0:
                U_b_x, ss_x, B_x_j_c = right_svd_3tensor(B0_b_j_c)
                A_a_i_x = jnp.tensordot(A0_a_i_b, U_b_x * ss_x.reshape((1, -1)), axes=1)

                right_orthogonal_tt_cores[ii-1] = A_a_i_x
                right_orthogonal_tt_cores[ii] = B_x_j_c

        return T3TangentSpace(
            orthogonal_basis_cores,
            left_orthogonal_tt_cores,
            tuple(right_orthogonal_tt_cores),
            tuple(up_orthogonal_tt_cores),
        )

    def _check_consistency_with_other_ttt_cores(
            me,
            other_t3_cores: TuckerTensorTrain,
    ):
        t3_check_correctness(other_t3_cores)

        other_basis_cores, other_tt_cores = other_t3_cores

        other_tucker_ranks = tt_get_shape(other_tt_cores)
        if (me.tucker_ranks != other_tucker_ranks):
            raise RuntimeError('tucker ranks='+str(me.tucker_ranks)+', other tt shape='+str(other_tucker_ranks))

        other_tt_ranks = tt_get_ranks(other_tt_cores)
        if (me.tt_ranks != other_tt_ranks):
            raise RuntimeError('tt ranks='+str(me.tt_ranks)+', other tt ranks='+str(other_tt_ranks))

        other_shape = tuple([B.shape[1] for B in other_basis_cores])
        if (me.shape != other_shape):
            raise RuntimeError('shape='+str(me.shape)+', other shape='+str(other_shape))

    def add_tangent_vectors(
            me,
            u: T3Variations,
            v: T3Variations,
    ) -> T3Variations: # u + v
        me._check_consistency_with_other_ttt_cores(u)
        me._check_consistency_with_other_ttt_cores(v)
        return (
            tuple([dB_u + dB_v for dB_u, dB_v in zip(u[0], v[0])]),
            tuple([dG_u + dG_v for dG_u, dG_v in zip(u[1], v[1])]),
        )

    def subtract_tangent_vectors(
            me,
            u: T3Variations,
            v: T3Variations,
    ) -> T3Variations: # u - v
        me._check_consistency_with_other_ttt_cores(u)
        me._check_consistency_with_other_ttt_cores(v)
        return (
            tuple([dB_u - dB_v for dB_u, dB_v in zip(u[0], v[0])]),
            tuple([dG_u - dG_v for dG_u, dG_v in zip(u[1], v[1])]),
        )

    def scale_tangent_vector(
            me,
            scaling_factor,
            u: T3Variations,
    ) -> T3Variations: # s * u
        me._check_consistency_with_other_ttt_cores(u)
        return (
            tuple([scaling_factor * dB_u for dB_u in u[0]]),
            tuple([scaling_factor * dG_u for dG_u in u[1]]),
        )

    def gauged_inner_product(
            me,
            u_gauged: T3Variations, # Must satisfy TTT Gauge condition!
            v_gauged: T3Variations, # Must satisfy TTT Gauge condition!
    ): # (u_gauged, v_gauged)_HS, Hilbert-Schmidt inner product inherited from ambient space
        me._check_consistency_with_other_ttt_cores(u_gauged)
        me._check_consistency_with_other_ttt_cores(v_gauged)
        return (
            jnp.sum(jnp.array([jnp.sum(dB_u * dB_v) for dB_u, dB_v in zip(u_gauged[0], v_gauged[0])])) +
            jnp.sum(jnp.array([jnp.sum(dG_u * dG_v) for dG_u, dG_v in zip(u_gauged[1], v_gauged[1])]))
        )

    def gauged_norm(
            me,
            u_gauged: T3Variations, # Must satisfy TTT Gauge condition!
    ):
        me._check_consistency_with_other_ttt_cores(u_gauged)
        return jnp.sqrt(me.gauged_inner_product(u_gauged, u_gauged))

    def tangent_vector_to_tucker_tensor_train(
            me,
            u: T3Variations,
    ) -> TuckerTensorTrain:
        '''Rank 2r Tucker tensor train representation of tangent vector:
                u(x,y,z,w) = ([dU1(B x) P1(B x)]) ([Q2(B y)        0]) ([Q3(B z)        0]) ([Q4(B w) ])
                             (                  ) ([dU2(B y) P2(B y)]) ([dU3(B z) P3(B z)]) ([dU4(B w)])
                             (         +        ) (         +        ) (        +         ) (    +     )
                             ([R1(dB x)       0]) ([0              0]) ([0              0]) ([0       ])
                             (                  ) ([R2(dB y)       0]) ([R3(dB z)       0]) ([R4(dB w)])
        '''
        me._check_consistency_with_other_ttt_cores(u)
        basis_variations, tt_variations = u

        basis_cores = []
        for B, dB in zip(me.orthogonal_basis_cores, basis_variations):
            B2 = jnp.concatenate([B, dB], axis=0)
            basis_cores.append(B2)

        tt_cores = []

        dU = tt_variations[0]
        R = me.up_orthogonal_tt_cores[0]
        P = me.left_orthogonal_tt_cores[0]
        Z = jnp.zeros(P.shape)
        G_top = jnp.concatenate([dU, P], axis=2)
        G_bot = jnp.concatenate([R, Z], axis=2)
        G = jnp.concatenate([G_top, G_bot], axis=1)
        tt_cores.append(G)

        for ii in range(1, me.num_cores-1):
            P = me.left_orthogonal_tt_cores[ii]
            Q = me.right_orthogonal_tt_cores[ii]
            R = me.up_orthogonal_tt_cores[ii]
            dU = tt_variations[ii]
            Z001 = jnp.zeros((Q.shape[0], Q.shape[1], P.shape[2]))
            Z100 = jnp.zeros(Q.shape)
            Z101 = Z001
            Z111 = jnp.zeros(P.shape)
            G_top = jnp.concatenate([
                jnp.concatenate([Q, Z001], axis=2),
                jnp.concatenate([dU, P], axis=2)
            ], axis=0)
            G_bot = jnp.concatenate([
                jnp.concatenate([Z100, Z101], axis=2),
                jnp.concatenate([R, Z111], axis=2)
            ], axis=0)
            G = jnp.concatenate([G_top, G_bot], axis=1)
            tt_cores.append(G)

        dU = tt_variations[-1]
        Q = me.right_orthogonal_tt_cores[-1]
        R = me.up_orthogonal_tt_cores[-1]
        Z = jnp.zeros(Q.shape)
        G_top = jnp.concatenate([Q, dU], axis=0)
        G_bot = jnp.concatenate([Z, R], axis=0)
        G = jnp.concatenate([G_top, G_bot], axis=1)
        tt_cores.append(G)

        return tuple(basis_cores), tuple(tt_cores)

    def attached_tangent_vector_to_tucker_tensor_train(
            me,
            u: T3Variations,
    ) -> TuckerTensorTrain:
        '''Rank 2r tensor train representation of shifted tangent vector p + v,
        where the tail of the vector is attached at the base point on the manifold.

        Same as unshifted tangent vector v, except last core modified as follows:
        [Q4(B w) ]                  [Q4(B w)           ]
        [dU4(B w)]                  [P4(B w) + dU4(B w)]
            +             ->            +
        [0       ]                  [0                 ]
        [R4(dB w)]                  [R4(dB w)          ]
        '''
        me._check_consistency_with_other_ttt_cores(u)
        basis_variations, tt_variations = u
        basis_cores, tt_cores = me.tangent_vector_to_tucker_tensor_train(u)

        dU = tt_variations[-1]
        P = me.left_orthogonal_tt_cores[-1]
        Q = me.right_orthogonal_tt_cores[-1]
        R = me.up_orthogonal_tt_cores[-1]
        Z = jnp.zeros(Q.shape)
        G_top = jnp.concatenate([Q, P + dU], axis=0)
        G_bot = jnp.concatenate([Z, R], axis=0)
        G = jnp.concatenate([G_top, G_bot], axis=1)

        tt_cores = tt_cores[:-1] + (G,)

        return (basis_cores, tt_cores)

    def get_tt_term(
            me,
            u: T3Variations,
            ii: int
    ) -> TuckerTensorTrain:
        me._check_consistency_with_other_ttt_cores(u)
        term_tt_cores = (
                me.left_orthogonal_tt_cores[:ii] +
                (u[1][ii],) +
                me.right_orthogonal_tt_cores[ii+1:]
        )
        term_basis_cores = me.orthogonal_basis_cores
        return (term_basis_cores, term_tt_cores)

    def get_basis_term(
            me,
            u: T3Variations,
            ii: int
    ) -> TuckerTensorTrain:
        me._check_consistency_with_other_ttt_cores(u)
        term_tt_cores = (
                me.left_orthogonal_tt_cores[:ii] +
                (me.up_orthogonal_tt_cores[ii],) +
                me.right_orthogonal_tt_cores[ii+1:]
        )
        term_basis_cores = (
                me.orthogonal_basis_cores[:ii] +
                (u[0][ii],) +
                me.orthogonal_basis_cores[ii+1:]
        )
        return (term_basis_cores, term_tt_cores)

    def tangent_vector_to_dense(
            me,
            u: T3Variations,
    ) -> jnp.ndarray:
        T = jnp.zeros(me.shape)
        for ii in range(me.num_cores):
            T = T + t3_to_dense(me.get_tt_term(u, ii))
            T = T + t3_to_dense(me.get_basis_term(u, ii))
        return T

    def attached_tangent_vector_to_dense(
            me,
            u: T3Variations,
    ) -> jnp.ndarray:
        return t3_to_dense((me.orthogonal_basis_cores, me.left_orthogonal_tt_cores)) + me.tangent_vector_to_dense(u)

    def orthogonal_gauge_projection(
            me,
            u: T3Variations,
    ) -> T3Variations:
        '''Makes variations left-perpendicular by orthogonally projecting away the parallel components.
        Changes tangent vector.
        1) Make basis variations left-perpendicular to corresponding basis cores (all basis cores).
        2) Make tt variations left-perpendicular to corresponding tt cores (except last tt cores).
        '''
        me._check_consistency_with_other_ttt_cores(u)
        basis_variations, tt_variations = u

        new_tt_variations = []
        for dV, P in zip(tt_variations[:-1], me.left_orthogonal_tt_cores[:-1]):
            dV2 = dV - jnp.einsum('iaj,jk->iak', P, jnp.einsum('iaj,iak->jk', P, dV))
            new_tt_variations.append(dV2)
        new_tt_variations.append(tt_variations[-1])

        new_basis_variations = []
        for dB, B in zip(basis_variations, me.orthogonal_basis_cores):
            dB2 = dB - (dB @ B.T) @ B
            new_basis_variations.append(dB2)

        return tuple(new_basis_variations), tuple(new_tt_variations)

    def oblique_gauge_projection(
            me,
            u: T3Variations,
    ) -> T3Variations:
        '''Makes variations left-perpendicular.
        Preserves tangent vector.
        1) Make basis variations left-perpendicular by pushing remainder onto tt variations
        2) Make tt variations left-perpendicular by standard sweeping tt method
        '''
        me._check_consistency_with_other_ttt_cores(u)
        basis_variations, tt_variations = u

        tt_variations = list(tt_variations)
        basis_variations = list(basis_variations)

        # Make basis variations left-perpendicular
        for ii in range(me.num_cores):
            B_io = me.orthogonal_basis_cores[ii]
            dB_jo = basis_variations[ii]
            R_aib = me.up_orthogonal_tt_cores[ii]
            dG_ajb = tt_variations[ii]

            X_ji = dB_jo @ B_io.T
            dB_parallel_jo = X_ji @ B_io
            dB2_jo = dB_jo - dB_parallel_jo # dB_perp
            # dG2_ajb = dG_ajb + jnp.einsum('aib,ji->ajb', R_aib, X_ji)
            dG2_ajb = dG_ajb + jnp.einsum('aib,ij->ajb', R_aib, X_ji) # <-- Why is this correct?

            tt_variations[ii] = dG2_ajb
            basis_variations[ii] = dB2_jo

        # Make tt cores left-perpendicular
        for ii in range(me.num_cores-1):
            dV1 = tt_variations[ii]
            dV2 = tt_variations[ii+1]

            P1 = me.left_orthogonal_tt_cores[ii]
            Q2 = me.right_orthogonal_tt_cores[ii+1]
            X = jnp.einsum('iaj,iak->jk', P1, dV1)
            new_dV1 = dV1 - jnp.einsum('iaj,jk->iak', P1, X)
            new_dV2 = dV2 + jnp.einsum('jk,kbl->jbl', X, Q2)

            tt_variations[ii] = new_dV1
            tt_variations[ii+1] = new_dV2

        return tuple(basis_variations), tuple(tt_variations)

    def project_ttt_onto_tangent_space(
            me,
            x: TuckerTensorTrain,
    ) -> T3Variations:
        other_basis_cores, other_tt_cores = x

        other_tt_cores2 = []
        for G_other, B_other, B in zip(other_tt_cores, other_basis_cores, me.orthogonal_basis_cores):
            G_other2 = jnp.einsum('aib,ix->axb', G_other, B_other @ B.T)
            other_tt_cores2.append(G_other2)

        zipper_left2right = tt_zipper_left_to_right(other_tt_cores2, me.left_orthogonal_tt_cores)[:-1]
        zipper_right2left = tt_zipper_right_to_left(other_tt_cores2, me.right_orthogonal_tt_cores)[1:]

        ungauged_tt_variations = []
        ungauged_basis_variations = []
        for ZL_ax, ZR_by, G_aib, B_io, R0_xjy, B0_jo in zip(
                zipper_left2right, zipper_right2left,
                other_tt_cores, other_basis_cores,
                me.up_orthogonal_tt_cores, me.orthogonal_basis_cores,
        ):
            X_xiy = jnp.einsum('ax,aib,by->xiy', ZL_ax, G_aib, ZR_by)
            dG_xjy = jnp.einsum('xiy,ij->xjy', X_xiy, B_io @ B0_jo.T)
            M_ij = jnp.einsum('xiy,xjy->ij', X_xiy, R0_xjy)
            dB_jo = jnp.einsum('ij,io->jo', M_ij, B_io)

            ungauged_tt_variations.append(dG_xjy)
            ungauged_basis_variations.append(dB_jo)

        ungauged_u = (ungauged_basis_variations, ungauged_tt_variations)
        gauged_u = me.orthogonal_gauge_projection(ungauged_u)
        return gauged_u

    @jax.jit
    def retract(
            me,
            u: T3Variations,
    ) -> TuckerTensorTrain: # retracted Tucker tensor train
        me._check_consistency_with_other_ttt_cores(u)
        x = me.attached_tangent_vector_to_tucker_tensor_train(u)
        retracted_x, _, _ = t3_svd(
            x,
            forced_tt_ranks=me.tt_ranks[1:-1],
            forced_basis_ranks=me.tucker_ranks,
        )
        return retracted_x

    @ft.cached_property
    def data(me):
        return (
            me.orthogonal_basis_cores,
            me.left_orthogonal_tt_cores,
            me.right_orthogonal_tt_cores,
            me.up_orthogonal_tt_cores,
        )

    def tree_flatten(me):
        return (me.data, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


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
