# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ
import math

import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.stacking as stacking
from t3toolbox.backend.common import *

__all__ = [
    'to_dense',
    'squash_tt_tails',
    'reverse_tt',
    'change_tucker_core_shapes',
    'change_tt_core_shapes',
    't3_unstack',
    't3_core_shapes',
    't3_to_vector',
    't3_from_vector',
    't3_zeros',
    't3_corewise_randn',
    't3_ones',
    't3_from_canonical',
    't3_from_tensor_train',
    't3_sum',
]


def to_dense(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tt_cores, tucker_cores)
        squash_tails: bool = True,
) -> NDArray:
    """Fully contract a Tucker tensor train to create a dense tensor.
    """
    use_jax = any([is_jax_ndarray(c) for c in x[0]]) or any([is_jax_ndarray(c) for c in x[1]])
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x
    vs = tucker_cores[0].shape[:-2] # stack_shape

    big_tt_cores = [xnp.einsum('...iaj,...ab->...ibj', G, U) for G, U in zip(tt_cores, tucker_cores)]

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


def squash_tt_tails(
        tt_cores: typ.Sequence[NDArray],
) -> typ.Tuple[NDArray]:
    """Make leading and trailing TT ranks equal to 1 (r0=rd=1), without changing tensor being represented.
    """
    use_jax = any([is_jax_ndarray(G) for G in tt_cores])
    xnp, _, _ = get_backend(False, use_jax)

    #
    G0 = tt_cores[0]
    G0 = xnp.einsum('az,...aib->...zib', xnp.ones((G0.shape[-3],1)), G0)

    Gf = tt_cores[-1]
    Gf = xnp.einsum('...aib,bz->...aiz', Gf, xnp.ones((Gf.shape[-1],1)))

    return (G0,) + tuple(tt_cores[1:-1]) + (Gf,)


def reverse_tt(
        tt_cores: typ.Sequence[NDArray],
) -> typ.Tuple[NDArray,...]:
    """Reverse a tensor train (no Tucker).
    """
    return tuple(G.swapaxes(-3, -1) for G in tt_cores[::-1])


def change_tucker_core_shapes(
        tucker_cores: typ.Sequence[NDArray],
        new_shape: typ.Sequence[int], # len=d
        new_tucker_ranks: typ.Sequence[int], # len=d
) -> typ.Tuple[NDArray,...]:
    """Increase/decrease Tucker and/or TT ranks for TT cores using zero padding/truncation.
    """
    use_jax = is_jax_ndarray(tucker_cores[0])
    xnp, xmap, _ = get_backend(False, use_jax)

    #
    old_shape = [B.shape[-1] for B in tucker_cores]
    old_tucker_ranks = [B.shape[-2] for B in tucker_cores]

    num_cores = len(tucker_cores)
    stack_shape = tucker_cores[0].shape[:-2]

    delta_shape         = [N_new - N_old for N_new, N_old in zip(new_shape, old_shape)]
    delta_tucker_ranks  = [n_new - n_old for n_new, n_old in zip(new_tucker_ranks, old_tucker_ranks)]

    new_tucker_cores = []
    for ii in range(num_cores):
        stack_pad = ((0,0),)*len(stack_shape)
        pad = stack_pad + (
            (0,delta_tucker_ranks[ii]),
            (0,delta_shape[ii]),
        )
        # new_B = xnp.pad(tucker_cores[ii], pad)
        new_B = linalg.pad_or_truncate(tucker_cores[ii], pad)
        new_tucker_cores.append(new_B)

    return tuple(new_tucker_cores)


def change_tt_core_shapes(
        tt_cores: typ.Sequence[NDArray],
        new_tucker_ranks: typ.Sequence[int], # len=d
        new_tt_ranks: typ.Sequence[int], # len=d+1
) -> typ.Tuple[NDArray,...]:
    """Increase/decrease Tucker and/or TT ranks for TT cores using zero padding/truncation.
    """
    use_jax = is_jax_ndarray(tt_cores[0])
    xnp, xmap, _ = get_backend(False, use_jax)

    #
    old_tucker_ranks = [G.shape[-2] for G in tt_cores]
    old_tt_ranks = [G.shape[-3] for G in tt_cores] + [tt_cores[-1].shape[-1]]

    num_cores = len(tt_cores)
    stack_shape = tt_cores[0].shape[:-3]

    delta_tucker_ranks  = [n_new - n_old for n_new, n_old in zip(new_tucker_ranks, old_tucker_ranks)]
    delta_tt_ranks      = [r_new - r_old for r_new, r_old in zip(new_tt_ranks, old_tt_ranks)]

    new_tt_cores = []
    for ii in range(num_cores):
        stack_pad = ((0,0),)*len(stack_shape)
        pad = stack_pad + (
            (0,delta_tt_ranks[ii]),
            (0,delta_tucker_ranks[ii]),
            (0,delta_tt_ranks[ii+1]),
        )
        # new_G = xnp.pad(tt_cores[ii], pad)
        new_G = linalg.pad_or_truncate(tt_cores[ii], pad)
        new_tt_cores.append(new_G)

    return tuple(new_tt_cores)


def t3_stack(
        xx, # array-like structure of nested tuples containing Tucker tensor trains
        use_jax: bool = False,
) -> typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]]:  # (stacked_tucker_cores, stacked_tt_cores)
    xnp,_,_ = get_backend(False, use_jax)

    num_stacking_axes = stacking.tree_depth(xx) - 2
    stacking_axes = tuple(range(num_stacking_axes))
    stacked_xx = stacking.stack(xx, stacking_axes, use_jax=use_jax)
    return stacked_xx


def t3_unstack(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
): # returns an array-like structure of nested tuples containing Tucker tensor trains
    """Given multiple stacked T3s, this unstacks them
    into an array-like structure of nested tuples with the same "shape" as the stacking shape.
    """
    num_stacking_axes = len(stacking.get_first_leaf(x).shape) - 2 # shape=stacking_shape + (ni,Ni)
    stacking_axes = tuple(range(num_stacking_axes))
    x_unstacked = stacking.unstack(x, stacking_axes)
    return x_unstacked


# def t3_sum_stack(
#         x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
# ) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (summed_tucker_cores, summed_tt_cores)
#     """If this object contains multiple stacked T3s, this sums them.
#     """
#     num_stacking_axes = len(x[0][0].shape) - 2
#     axes = tuple(range(num_stacking_axes))
#     return stacking.sum_leafs_along_axes(x, axes=axes)


def t3_core_shapes(
        shape: typ.Sequence[int],
        tucker_ranks: typ.Sequence[int],
        tt_ranks: typ.Sequence[int],
        stack_shape: typ.Sequence[int] = (),
) -> typ.Tuple[
    typ.Tuple[int,...], # tucker_core_shapes
    typ.Tuple[int,...], # tt_core_shapes
]:
    """Determines the shapes of the T3 cores based on the ranks.
    """
    vs = tuple(stack_shape)
    tucker_core_shapes = []
    for n, N in zip(tucker_ranks, shape):
        tucker_core_shapes.append(vs+(n,N))

    tt_core_shapes = []
    for rL, n, rR in zip(tt_ranks[:-1], tucker_ranks, tt_ranks[1:]):
        tt_core_shapes.append(vs+(rL,n,rR))

    return tuple(tucker_core_shapes), tuple(tt_core_shapes)


def t3_to_vector(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        use_jax: bool=False,
) -> NDArray: # shape=(x_size,)
    """Converts T3 to a 1D vector containing all of the core entries.
    """
    xnp, _, _ = get_backend(False, use_jax)

    x_flats = []
    for B in x[0]:
        x_flats.append(B.reshape(-1))
    for G in x[1]:
        x_flats.append(G.reshape(-1))

    return xnp.concatenate(x_flats)


def t3_from_vector(
        x_flat: NDArray,
        shape: typ.Sequence[int],
        tucker_ranks: typ.Sequence[int],
        tt_ranks: typ.Sequence[int],
        stack_shape: typ.Sequence[int] = (),
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]: # (tucker_cores, tt_cores)
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


def t3_corewise_randn(
        shape:                  typ.Tuple[int, ...],
        tucker_ranks:           typ.Tuple[int, ...],
        tt_ranks:               typ.Tuple[int, ...],
        stack_shape:    typ.Tuple[int, ...] = (),
        use_jax: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]: # (tucker_cores, tt_cores)
    """Construct a Tucker tensor train with random cores.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    d = len(tucker_ranks)
    vs = stack_shape

    tt_cores = []
    for ii in range(d):
        shape_G = vs + (tt_ranks[ii], tucker_ranks[ii], tt_ranks[ii+1])
        G = randn(*shape_G, use_jax=use_jax)
        tt_cores.append(G)

    tucker_cores = []
    for ii in range(d):
        shape_B = vs + (tucker_ranks[ii], shape[ii])
        B = randn(*shape_B, use_jax=use_jax)
        tucker_cores.append(B)

    return tuple(tucker_cores), tuple(tt_cores)


def t3_zeros(
        shape:                  typ.Tuple[int,...],
        tucker_ranks:           typ.Tuple[int,...],
        tt_ranks:               typ.Tuple[int,...],
        stack_shape:    typ.Tuple[int,...] = (),
        use_jax: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]: # (tucker_cores, tt_cores)
    """Construct a Tucker tensor train of zeros.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    vs = stack_shape

    tt_cores = tuple([xnp.zeros(vs+(tt_ranks[ii], tucker_ranks[ii], tt_ranks[ii+1])) for ii in range(len(tucker_ranks))])
    tucker_cores = tuple([xnp.zeros(vs+(n, N)) for n, N  in zip(tucker_ranks, shape)])
    return tucker_cores, tt_cores


def t3_ones(
        shape:                  typ.Tuple[int,...],
        stack_shape:    typ.Tuple[int,...] = (),
        use_jax: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]: # (tucker_cores, tt_cores)
    """Construct the rank-1 Tucker tensor train representing a tensor full of ones.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    vs = stack_shape

    tt_cores = tuple([xnp.ones(vs+(1, 1, 1)) for ii in range(len(shape))])
    tucker_cores = tuple([xnp.ones(vs+(1, N)) for N  in shape])
    return tucker_cores, tt_cores


def wt3_squash_tails(
        x, # weighted Tucker tensor train
        use_jax: bool = False,
):
    """Reduce the first and last dimensions of the first and last tt cores to 1.
    """
    xnp, _, _ = get_backend(False, use_jax=use_jax)

    x0, w = x
    tucker_cores, tt_cores = x0
    tucker_weights, tt_weights = w

    stack_shape = tucker_weights[0].shape[:-1]

    first_G = xnp.einsum('...aib,...a->...aib', tt_cores[0], tt_weights[0])
    first_G = first_G.sum(axis=-3, keepdims=True)
    first_wtt = xnp.ones(stack_shape + (1,))

    mid_G = tt_cores[1:-1]
    mid_wtt = tt_weights[1:-1]

    last_G = xnp.einsum('...aib,...b->...aib', tt_cores[-1], tt_weights[-1])
    last_G = last_G.sum(axis=-1, keepdims=True)
    last_wtt = xnp.ones(stack_shape + (1,))

    tt_cores = (first_G,) + mid_G + (last_G,)
    tt_weights = (first_wtt,) + mid_wtt + (last_wtt,)

    x0 = (tucker_cores, tt_cores)
    w = (tucker_weights, tt_weights)
    return (x0, w)


def t3_from_canonical(
        factors: typ.Sequence[NDArray], # elm_shape = stack_shape + (canonical_rank, Ni)
) -> typ.Tuple[
    typ.Tuple[NDArray,...], # tucker_cores
    typ.Tuple[NDArray,...], # tt_cores
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


def t3_from_tensor_train(
        tt_cores: typ.Sequence[NDArray], # elm_shape=stack_shape+(ri, N, r(i+1))
) -> typ.Tuple[
    typ.Tuple[NDArray,...], # tucker_cores
    typ.Tuple[NDArray,...], # tt_cores
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


def t3_to_tensor_train(
        x: typ.Tuple[
            typ.Tuple[NDArray,...], # tucker_cores
            typ.Tuple[NDArray,...], # tt_cores
        ],
) -> typ.Tuple[NDArray,...]: # tt_cores
    """Convert TuckerTensorTrain to tensor train by contracting Tucker bases with TT cores.
    """
    use_jax = any([is_jax_ndarray(c) for c in x[0]]) or any([is_jax_ndarray(c) for c in x[1]])
    xnp, _, _ = get_backend(False, use_jax)

    return tuple(
        xnp.einsum('...aib,...io->...aob', G, B)
        for G, B in zip(x[1], x[0])
        )


def t3_sum(
        x: typ.Tuple[
            typ.Tuple[NDArray,...], # tucker_cores
            typ.Tuple[NDArray,...], # tt_cores
        ],
        axis=None,
):
    """Sum over axes of TuckerTensorTrain.
    """
    tucker_cores, tt_cores = x
    d = len(tucker_cores)

    use_jax = any([is_jax_ndarray(c) for c in tucker_cores]) or any([is_jax_ndarray(c) for c in tt_cores])
    xnp, _, _ = get_backend(False, use_jax)

    if axis is None:
        axis = list(range(d))
    elif not isinstance(axis, typ.Sequence):
        axis = [axis]
    else:
        axis = list(axis)

    for ii, ax in enumerate(axis):
        if ax < 0:
            ax = d + ax
            axis[ii] = ax
        assert(0 <= ax)
        assert(ax < d)

    axis = sorted(list(set(axis))) # remove duplicates

    S = (tuple(tucker_cores), tuple(tt_cores))
    while len(axis) > 0:
        ax = axis[-1]
        axis = axis[:-1]

        B, G = S[0][ax], S[1][ax]

        M = xnp.einsum('...aib,...io->...ab', G, B)

        if len(S[0]) == 1:
            S = M.sum(axis=(-2,-1))
        else:
            left_tucker,    right_tucker    = list(S[0][:ax]), list(S[0][ax+1:])
            left_tt,        right_tt        = list(S[1][:ax]), list(S[1][ax+1:])

            if ax == 0:
                right_tt[0] = xnp.einsum('...ab,...bic->...aic', M, right_tt[0])
            else:
                left_tt[-1] = xnp.einsum('...aib,...bc->...aic', left_tt[-1], M)

            S = (tuple(left_tucker + right_tucker), tuple(left_tt + right_tt))

    return S
