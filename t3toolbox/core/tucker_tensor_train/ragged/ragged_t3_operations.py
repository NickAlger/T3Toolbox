import numpy as np
import typing as typ

from t3toolbox.common import *

__all__ = [
    'to_dense',
    'squash_tt_tails',
    'reverse_tt',
    'change_tucker_core_shapes',
    'change_tt_core_shapes',
    'absorb_edge_weights_into_t3',
    't3_unstack',
    't3_core_shapes',
    't3_to_vector',
    't3_from_vector',
    't3_sum_stack',
]


def to_dense(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tt_cores, tucker_cores)
        squash_tails: bool = True,
        use_jax: bool = False,
) -> NDArray:
    """Fully contract a Tucker tensor train to create a dense tensor.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x
    vs = tucker_cores[0].shape[:-2] # stack_shape

    big_tt_cores = [xnp.einsum('...iaj,...ab->...ibj', G, U) for G, U in zip(tt_cores, tucker_cores)]

    T = big_tt_cores[0]
    for G in big_tt_cores[1:]:
        ts = T.shape[len(vs):-1]
        cs = (T.shape[-1],)
        T_a_b_c_xyz_r = T.reshape(vs + (xnp.prod(ts, dtype=int),) + cs)

        ts2 = G.shape[-2:]
        G_a_b_c_r_lm = G.reshape(vs + cs + (xnp.prod(ts2, dtype=int),))
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
        use_jax: bool = False,
) -> typ.Tuple[NDArray]:
    """Make leading and trailing TT ranks equal to 1 (r0=rd=1), without changing tensor being represented.
    """
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
    return tuple([G.swapaxes(-3, -1) for G in tt_cores[::-1]])


def change_tucker_core_shapes(
        tucker_cores: typ.Sequence[NDArray],
        new_shape: typ.Sequence[int], # len=d
        new_tucker_ranks: typ.Sequence[int], # len=d
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]:
    """Increase Tucker and/or TT ranks for TT cores using zero padding.
    """
    xnp, _, _ = get_backend(False, use_jax)

    old_shape = [B.shape[1] for B in tucker_cores]
    old_tucker_ranks = [B.shape[0] for B in tucker_cores]

    num_cores = len(tucker_cores)

    delta_shape         = [N_new - N_old for N_new, N_old in zip(new_shape, old_shape)]
    delta_tucker_ranks  = [n_new - n_old for n_new, n_old in zip(new_tucker_ranks, old_tucker_ranks)]

    new_tucker_cores = []
    for ii in range(num_cores):
        new_tucker_cores.append(xnp.pad(
            tucker_cores[ii],
            (
                (0,delta_tucker_ranks[ii]),
                (0,delta_shape[ii]),
            ),
        ))

    return tuple(new_tucker_cores)


def change_tt_core_shapes(
        tt_cores: typ.Sequence[NDArray],
        new_tucker_ranks: typ.Sequence[int], # len=d
        new_tt_ranks: typ.Sequence[int], # len=d+1
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]:
    """Increase Tucker and/or TT ranks for TT cores using zero padding.
    """
    xnp, _, _ = get_backend(False, use_jax)

    old_tucker_ranks = [G.shape[1] for G in tt_cores]
    old_tt_ranks = [G.shape[0] for G in tt_cores] + [tt_cores[-1].shape[2]]

    num_cores = len(tt_cores)

    delta_tucker_ranks  = [n_new - n_old for n_new, n_old in zip(new_tucker_ranks, old_tucker_ranks)]
    delta_tt_ranks      = [r_new - r_old for r_new, r_old in zip(new_tt_ranks, old_tt_ranks)]

    new_tt_cores = []
    for ii in range(num_cores):
        new_tt_cores.append(xnp.pad(
            tt_cores[ii],
            (
                (0,delta_tt_ranks[ii]),
                (0,delta_tucker_ranks[ii]),
                (0,delta_tt_ranks[ii+1]),
            ),
        ))

    return tuple(new_tt_cores)


def absorb_edge_weights_into_t3(
        x0: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        weights: typ.Tuple[
            typ.Sequence[NDArray],  # shape_weights, len=d, elm_shape=(Ni,)
            typ.Sequence[NDArray],  # tucker_weights, len=d, elm_shape=(ni,)
            typ.Sequence[NDArray],  # tt_weights, len=d+1, elm_shape=(ri,)
        ],
        use_jax: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:
    """Contract each edge weight into a neighboring core.

    Tensor network diagram illustrating groupings::

             ____     ____     ________
            /    \   /    \   /        \
        1---w---G0---w---G1---w---G2---w---1
                |        |        |
              / w      / w      / w
              | |      | |      | |
              | B0     | B1     | B2
              | |      | |      | |
              \ w      \ w      \ w
                |        |        |

    """
    is_uniform = not isinstance(x0[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    tucker_cores0, tt_cores0 = x0
    shape_weights, tucker_weights, tt_weights = weights

    (tucker_cores,) = xmap(
        lambda tw_B_sw: (xnp.einsum('i,io,o->io', tw_B_sw[0], tw_B_sw[1], tw_B_sw[2]),),
        (tucker_weights, tucker_cores0, shape_weights)
    )
    (first_tt_cores,) = xmap(
        lambda lw_G: (xnp.einsum('i,iaj->iaj', lw_G[0], lw_G[1]),),
        (tt_weights[:-2], tt_cores0[:-1])
    )
    Gf = xnp.einsum('i,iaj,j->iaj', tt_weights[-2], tt_cores0[-1], tt_weights[-1])
    tt_cores = tuple(first_tt_cores) + (Gf,)

    return tucker_cores, tt_cores


def t3_stack(
        xx, # array-like structure of nested tuples containing Tucker tensor trains
        use_jax: bool = False,
) -> typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]]:  # (stacked_tucker_cores, stacked_tt_cores)
    xnp,_,_ = get_backend(False, use_jax)

    if is_ndarray(xx[0][0]):
        return xx

    xx = [t3_stack(x, use_jax=use_jax) for x in xx]
    x0 = xx[0]
    tucker_cores0, _ = x0
    num_cores = len(tucker_cores0)
    BBB = []
    GGG = []
    for ii in range(num_cores):
        BBi = []
        GGi = []
        for x in xx:
            Bi = x[0][ii]
            Gi = x[1][ii]
            BBi.append(Bi)
            GGi.append(Gi)
        BBi = xnp.stack(BBi)
        GGi = xnp.stack(GGi)
        BBB.append(BBi)
        GGG.append(GGi)

    return tuple(BBB), tuple(GGG)


def t3_unstack(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
): # returns an array-like structure of nested tuples containing Tucker tensor trains
    """Given multiple stacked T3s, this unstacks them
    into an array-like structure of nested tuples with the same "shape" as the stacking shape.
    """
    tucker_cores, tt_cores = x
    stack_shape = tucker_cores[0].shape[:-2]

    if not stack_shape:
        return x

    n = tucker_cores[0].shape[0]
    unstacked_x = []
    for ii in range(n):
        BB = tuple([B[ii] for B in tucker_cores])
        GG = tuple([G[ii] for G in tt_cores])
        xi = (BB, GG)
        unstacked_x.append(t3_unstack(xi))

    return tuple(unstacked_x)


def t3_sum_stack(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        use_jax: bool=False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (summed_tucker_cores, summed_tt_cores)
    """If this object contains multiple stacked T3s, this sums them.
    """
    xnp, _, _ = get_backend(False, use_jax=use_jax)
    tucker_cores, tt_cores = x
    vsv = tucker_cores[0].shape[:-2]
    N_vsv = np.prod(vsv, dtype=int)

    summed_tucker_cores = []
    for B in tucker_cores:
        B_sum = xnp.sum(B.reshape((N_vsv,) + B.shape[-2:]), axis=0)
        summed_tucker_cores.append(B_sum)

    summed_tt_cores = []
    for G in tt_cores:
        G_sum = xnp.sum(G.reshape((N_vsv,) + G.shape[-3:]), axis=0)
        summed_tt_cores.append(G_sum)

    return tuple(summed_tucker_cores), tuple(summed_tt_cores)


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
):
    """Constructs a T3 from a 1D vector containing the core entries
    """
    tucker_core_shapes, tt_core_shapes = t3_core_shapes(
        shape, tucker_ranks, tt_ranks, stack_shape=stack_shape,
    )

    start = 0
    tucker_cores = []
    for B_shape in tucker_core_shapes:
        stop = start + np.prod(B_shape, dtype=int)
        B = x_flat[start:stop].copy().reshape(B_shape)
        tucker_cores.append(B)
        start = stop

    tt_cores = []
    for G_shape in tt_core_shapes:
        stop = start + np.prod(G_shape, dtype=int)
        B = x_flat[start:stop].copy().reshape(G_shape)
        tt_cores.append(B)
        start = stop

    return tuple(tucker_cores), tuple(tt_cores)







