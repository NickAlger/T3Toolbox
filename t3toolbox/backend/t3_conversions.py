# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Conversions between a (ragged) Tucker tensor train and other representations
(dense tensor, flat vector, plain tensor train, canonical/CP factors)."""
import math
import numpy as np
import typing as typ

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.tt_operations as tt_operations
from t3toolbox.backend.common import *
from t3toolbox.backend.t3_operations import t3_absorb_tucker_into_tt, t3_broadcast_to_common_stack, t3_core_shapes

__all__ = [
    't3_to_dense',
    't3_to_dense_chain',
    't3_to_vector',
    't3_from_vector',
    't3_to_tensor_train',
    't3_from_tensor_train',
    't3_from_canonical',
]


def t3_to_dense(
        x:            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        squash_tails: bool = True,
) -> NDArray:  # shape = stack_shape + (N0,...,N(d-1)); +leading/trailing TT-rank axes if squash_tails=False
    """Fully contract a Tucker tensor train to create a dense tensor.
    """
    tucker_cores, tt_cores = x

    # Cores may carry different (but broadcastable) leading stack axes (e.g. a tangent term mixes a
    # V+G-stacked variation core with G-stacked frame cores); broadcast to the common stack so the
    # reshape-based contraction sees one uniform stack_shape. No-op for a uniform-stack T3.
    tucker_cores, tt_cores = t3_broadcast_to_common_stack(tucker_cores, tt_cores)
    return t3_to_dense_chain(tucker_cores, tt_cores, squash_tails)


def t3_to_dense_chain(
        tucker_cores: typ.Union[
            typ.Sequence[NDArray],  # ragged:  len=d, elm_shape=stack_shape+(ni, Ni)
            NDArray,                # uniform: shape=(d,)+stack_shape+(ni, Ni)
        ],
        tt_cores: typ.Union[
            typ.Sequence[NDArray],  # ragged:  len=d, elm_shape=stack_shape+(ri, ni, r(i+1))
            NDArray,                # uniform: shape=(d,)+stack_shape+(ri, ni, r(i+1))
        ],
        squash_tails: bool = True,
) -> NDArray:  # stack_shape + (N0,...,N(d-1)); +leading/trailing TT-rank axes if squash_tails=False
    """Chain-contract (Tucker-absorbed) TT cores into a dense tensor -- the representation-agnostic core
    of :py:func:`t3_to_dense`.

    Works on a ragged core tuple *or* a uniform supercore array: it only zips/indexes the cores and uses
    a leading ``'...'`` for the stack, so a supercore's leading mode axis is consumed by iteration just
    like a tuple. Callers handle the representation-specific pre/post steps (ragged: broadcast to a common
    stack; uniform: mask, then static prefix-slice to the real shape).
    """
    use_jax = tree_contains_jax((tucker_cores, tt_cores))
    xnp, _, _ = get_backend(False, use_jax)

    vs = tucker_cores[0].shape[:-2]  # stack_shape

    big_tt_cores = t3_absorb_tucker_into_tt(tucker_cores, tt_cores)

    T = big_tt_cores[0]
    for G in big_tt_cores[1:]:
        ts = T.shape[len(vs):-1]
        cs = (T.shape[-1],)
        T_a_b_c_xyz_r = T.reshape(vs + (math.prod(ts),) + cs)

        ts2 = G.shape[-2:]
        G_a_b_c_r_lm = G.reshape(vs + cs + (math.prod(ts2),))
        T_a_b_c_xyzlm = T_a_b_c_xyz_r @ G_a_b_c_r_lm
        T = T_a_b_c_xyzlm.reshape(vs + ts + ts2)

    if squash_tails:
        mu_L = xnp.ones(big_tt_cores[0].shape[-3])
        mu_R = xnp.ones(big_tt_cores[-1].shape[-1])

        T = xnp.tensordot(T, mu_R, axes=1)
        T = xnp.tensordot(mu_L, T, axes=((0,), (len(vs),)))

    return T


def t3_to_vector(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
) -> NDArray: # shape=(x_size,)
    """Converts T3 to a 1D vector containing all of the core entries.
    """
    xnp, _, _ = get_backend(False, tree_contains_jax(x))

    x_flats = []
    for B in x[0]:
        x_flats.append(B.reshape(-1))
    for G in x[1]:
        x_flats.append(G.reshape(-1))

    return xnp.concatenate(x_flats)


def t3_from_vector(
        x_flat:       NDArray,                 # shape=(x_size,), all core entries flattened
        shape:        typ.Sequence[int],       # len=d, the tensor mode sizes (N0,...,N(d-1))
        tucker_ranks: typ.Sequence[int],       # len=d
        tt_ranks:     typ.Sequence[int],       # len=d+1
        stack_shape:  typ.Sequence[int] = (),  # leading batch axes
) -> typ.Tuple[
    typ.Sequence[NDArray],  # tucker_cores. len=d, elm_shape=stack_shape+(ni,Ni)
    typ.Sequence[NDArray],  # tt_cores.     len=d, elm_shape=stack_shape+(rLi,ni,rR(i+1))
]:
    """Constructs a T3 from a 1D vector containing the core entries
    """
    tucker_core_shapes, tt_core_shapes = t3_core_shapes(
        shape, tucker_ranks, tt_ranks, stack_shape=stack_shape,
    )

    start = 0
    tucker_cores = []
    for B_shape in tucker_core_shapes:
        stop = start + math.prod(B_shape)
        B = x_flat[start:stop].copy().reshape(B_shape)
        tucker_cores.append(B)
        start = stop

    tt_cores = []
    for G_shape in tt_core_shapes:
        stop = start + math.prod(G_shape)
        B = x_flat[start:stop].copy().reshape(G_shape)
        tt_cores.append(B)
        start = stop

    return tuple(tucker_cores), tuple(tt_cores)


def t3_to_tensor_train(
        x: typ.Tuple[
            typ.Tuple[NDArray, ...],  # tucker_cores. len=d, elm_shape=stack_shape+(ni,Ni)
            typ.Tuple[NDArray, ...],  # tt_cores.     len=d, elm_shape=stack_shape+(rLi,ni,rR(i+1))
        ],
) -> typ.Tuple[NDArray, ...]:  # tt_cores (Tucker absorbed). len=d, elm_shape=stack_shape+(rLi,Ni,rR(i+1))
    """Convert TuckerTensorTrain to tensor train by contracting Tucker bases with TT cores.
    """
    use_jax = any([is_jax_ndarray(c) for c in x[0]]) or any([is_jax_ndarray(c) for c in x[1]])
    xnp, _, _ = get_backend(False, use_jax)

    return tuple(
        xnp.einsum('...aib,...io->...aob', G, B)
        for G, B in zip(x[1], x[0])
        )


def t3_from_tensor_train(
        tt_cores: typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rLi,Ni,rR(i+1))
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores. len=d, identity bases, elm_shape=stack_shape+(Ni,Ni)
    typ.Tuple[NDArray, ...],  # tt_cores.     len=d, elm_shape=stack_shape+(rLi,Ni,rR(i+1))
]:
    """Convert tensor train into Tucker tensor train by using identity matrices for Tucker bases.
    """
    use_jax = any(is_jax_ndarray(G) for G in tt_cores)
    xnp, _, _ = get_backend(False, use_jax)

    shape = tuple(G.shape[-2] for G in tt_cores)
    stack_shape = tt_cores[0].shape[:-3]

    tucker_cores = tuple(
        xnp.tensordot(xnp.ones(stack_shape), xnp.eye(N), axes=[(), ()]) for N in shape
    )
    return tucker_cores, tuple(tt_cores)


def t3_from_canonical(
        factors: typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(canonical_rank,Ni)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores. len=d, elm_shape=stack_shape+(canonical_rank,Ni)
    typ.Tuple[NDArray, ...],  # tt_cores.     len=d, superdiagonal, elm_shape=stack_shape+(cr,cr,cr)
]:
    """Constructs Tucker tensor train from Canonical decomposition.
    """
    use_jax = any([is_jax_ndarray(F) for F in factors])
    xnp, _, _ = get_backend(False, use_jax)

    #
    shape = tuple(F.shape[-1] for F in factors)
    n = factors[0].shape[-2] # canonical_rank
    ss = factors[0].shape[:-2] # stack_shape

    I = xnp.eye(n)
    I3 = xnp.einsum('ij,jk,ki->ijk', I, I, I) # 3D tensor with ones on the superdiagonal and zeros elsewhere
    G = xnp.tensordot(xnp.ones(ss), I3, axes=[(),()])

    tt_cores = tuple(G for _ in range(len(shape)))
    tucker_cores = tuple(factors)
    return tucker_cores, tt_cores
