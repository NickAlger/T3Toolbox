# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Structural operations on ragged t3 data.

Squash tails, segment/concatenate, stack/unstack, broadcast-to-common-stack, core-shape
utilities. ``wt3_squash_tails`` is an unexported PARKED weighted-layer copy (pending the
post-1.0 weighted redesign) -- leave it alone.
"""
import numpy as np
import typing as typ
import math

import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.stacking as stacking
from t3toolbox.backend.common import *
from t3toolbox.backend.tt_operations import tt_reverse, tt_squash_tails, tt_change_core_shapes

__all__ = [
    't3_absorb_tucker_into_tt',
    't3_broadcast_to_common_stack',
    't3_squash_tails',
    't3_segment',
    't3_concatenate',
    'tucker_change_core_shapes',
    't3_unstack',
    't3_stack',
    't3_core_shapes',
    't3_sum',
    't3_absorb_weights',
    't3_weights_consistent',
]


def t3_broadcast_to_common_stack(
        tucker_cores: typ.Sequence[NDArray],  # each shape = stack_i + (n, N)
        tt_cores:     typ.Sequence[NDArray],  # each shape = stack_i + (rL, n, rR)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores, each shape = stack + (n, N)
    typ.Tuple[NDArray, ...],  # tt_cores,     each shape = stack + (rL, n, rR)
]:
    """Broadcast every core of a T3 up to the common (broadcast) stack of all its cores.

    Cores may carry different but broadcastable leading stack axes -- e.g. a single-core-replacement
    term (``fv_to_t3``) or a tangent term mixes a V+G-stacked variation core with G-stacked frame
    cores (the shared base point replicated over the tangent stack V). Returns the cores all stacked
    at the common ``np.broadcast_shapes`` stack, so the result is a valid uniform-stack T3. A no-op
    when every core already shares one stack.
    """
    xnp, _, _ = get_backend(False, tree_contains_jax((tucker_cores, tt_cores)))
    vs = np.broadcast_shapes(                       # common stack_shape
        *(B.shape[:-2] for B in tucker_cores),
        *(G.shape[:-3] for G in tt_cores),
    )
    new_tucker_cores = tuple(xnp.broadcast_to(B, vs + B.shape[-2:]) for B in tucker_cores)
    new_tt_cores     = tuple(xnp.broadcast_to(G, vs + G.shape[-3:]) for G in tt_cores)
    return new_tucker_cores, new_tt_cores


def t3_absorb_tucker_into_tt(
        tucker_cores: typ.Union[
            typ.Sequence[NDArray],  # ragged:  len=d, elm_shape=stack_shape+(ni, Ni)
            NDArray,                # uniform: shape=(d,)+stack_shape+(ni, Ni)
        ],
        tt_cores: typ.Union[
            typ.Sequence[NDArray],  # ragged:  len=d, elm_shape=stack_shape+(ri, ni, r(i+1))
            NDArray,                # uniform: shape=(d,)+stack_shape+(ri, ni, r(i+1))
        ],
) -> typ.Union[
    typ.Tuple[NDArray, ...],  # ragged:  big TT cores, elm_shape=stack_shape+(ri, Ni, r(i+1))
    NDArray,                  # uniform: big TT supercore, shape=(d,)+stack_shape+(r, N, r)
]:
    """Absorb each Tucker core into its TT core, replacing the mode (``n``) leg with the physical (``N``)
    leg: ``big_tt[...,a,o,b] = sum_n tt[...,a,n,b] * tucker[...,n,o]``.

    Representation-agnostic: a single batched einsum over ``(d,)+stack`` for a uniform supercore (the
    vectorization win), a per-core list-comp for ragged tuples. The opening step of both :py:func:`t3_to_dense`
    (`t3_to_dense_chain`) and the inner-product/norm zipper.
    """
    is_uniform = is_ndarray(tucker_cores)
    use_jax = tree_contains_jax((tucker_cores, tt_cores))
    xnp, _, _ = get_backend(is_uniform, use_jax)

    if is_uniform:
        return xnp.einsum('d...anb,d...no->d...aob', tt_cores, tucker_cores)
    return tuple(xnp.einsum('...anb,...no->...aob', G, U) for G, U in zip(tt_cores, tucker_cores))


def t3_squash_tails(
        data: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores. len=d, elm_shape=stack_shape+(ni,Ni)
            typ.Sequence[NDArray],  # tt_cores.     len=d, elm_shape=stack_shape+(rLi,ni,rR(i+1))
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores, unchanged
    typ.Tuple[NDArray, ...],  # tt_cores with r0=rd=1
]:
    """Collapse the leading/trailing TT bonds of a T3 to one, without changing the represented tensor.

    The T3-data-level twin of :py:func:`~t3toolbox.backend.ut3_operations.ut3_squash_tails`
    (the Tucker cores are untouched; the TT chain goes through
    :py:func:`~t3toolbox.backend.tt_operations.tt_squash_tails`).
    """
    tucker_cores, tt_cores = data
    return tuple(tucker_cores), tt_squash_tails(tt_cores)


def t3_segment(
        data: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores. len=d
            typ.Sequence[NDArray],  # tt_cores.     len=d
        ],
        start: typ.Optional[int] = None,  # Python slice start (None -> 0; negatives wrap)
        stop:  typ.Optional[int] = None,  # Python slice stop  (None -> d; negatives wrap)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores[start:stop]
    typ.Tuple[NDArray, ...],  # tt_cores[start:stop]
]:
    """Contiguous mode-segment of a T3: slice both core families over modes ``start:stop``.

    ``start``/``stop`` follow Python slice semantics (``None`` -> the ends; negatives wrap), with a
    length >= 1 guard. Inverse of :py:func:`t3_concatenate`.
    """
    tucker_cores, tt_cores = data
    d = len(tucker_cores)
    if start is None:
        start = 0
    if stop is None:
        stop = d
    if start < 0:
        start = d + start
    if stop < 0:
        stop = d + stop
    if stop <= start:
        raise ValueError(
            "Attempted to extract segment with length < 1.\n"
            + str(start) + ' = start >= stop = ' + str(stop)
        )
    return tuple(tucker_cores[start:stop]), tuple(tt_cores[start:stop])


def t3_concatenate(
        xx: typ.Sequence[
            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]
        ],  # T3 data tuples, each (tucker_cores, tt_cores); TT ranks must match at each seam
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores (concatenated over modes)
    typ.Tuple[NDArray, ...],  # tt_cores
]:
    """Concatenate a sequence of T3 segments into one T3 (inverse of :py:func:`t3_segment`).

    At each seam the TT ranks must match -- the trailing TT rank of one segment equals the leading
    TT rank of the next -- otherwise a ``ValueError`` is raised. The join is structural (core-tuple
    concatenation); no re-orthogonalization.
    """
    if len(xx) < 1:
        raise ValueError(
            'Empty TuckerTensorTrain not supported.\n'
            + str(len(xx)) + ' = len(xx)'
        )
    elif len(xx) == 1:
        return xx[0]
    elif len(xx) == 2:
        (x_tucker, x_tt), (y_tucker, y_tt) = xx[0], xx[1]
        if x_tt[-1].shape[-1] != y_tt[0].shape[-3]:
            raise ValueError(
                'First and last TT-ranks inconsistent for concatenation.\n'
                + str(x_tt[-1].shape[-1]) + ' = x.tt_ranks[-1] != y.tt_ranks[0] = '
                + str(y_tt[0].shape[-3])
            )
        return tuple(x_tucker) + tuple(y_tucker), tuple(x_tt) + tuple(y_tt)
    else:
        return t3_concatenate([t3_concatenate(xx[:2])] + list(xx[2:]))


def tucker_change_core_shapes(
        tucker_cores:     typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(ni,Ni)
        new_shape:        typ.Sequence[int],      # len=d
        new_tucker_ranks: typ.Sequence[int],      # len=d
) -> typ.Tuple[NDArray, ...]:  # resized tucker_cores. len=d, elm_shape=stack_shape+(new_ni,new_Ni)
    """Increase/decrease Tucker and/or TT ranks for TT cores using zero padding/truncation.
    """
    use_jax = tree_contains_jax(tucker_cores)
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


def t3_stack(
        xx,  # array-like structure of nested tuples containing Tucker tensor trains (stack tree)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # stacked_tucker_cores. elm_shape=stack_shape+(ni,Ni)
    typ.Tuple[NDArray, ...],  # stacked_tt_cores.     elm_shape=stack_shape+(rLi,ni,rR(i+1))
]:
    num_stacking_axes = stacking.tree_depth(xx) - 2
    stacking_axes = tuple(range(num_stacking_axes))
    stacked_xx = stacking.stack(xx, stacking_axes)
    return stacked_xx


def t3_unstack(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores), stacked
):  # -> array-like structure of nested tuples containing Tucker tensor trains (shape = stack_shape)
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
        shape:        typ.Sequence[int],       # len=d, the tensor mode sizes (N0,...,N(d-1))
        tucker_ranks: typ.Sequence[int],       # len=d
        tt_ranks:     typ.Sequence[int],       # len=d+1
        stack_shape:  typ.Sequence[int] = (),  # leading batch axes
) -> typ.Tuple[
    typ.Tuple[int, ...],  # tucker_core_shapes. len=d, each = stack_shape+(ni,Ni)
    typ.Tuple[int, ...],  # tt_core_shapes.     len=d, each = stack_shape+(rLi,ni,rR(i+1))
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


def wt3_squash_tails(
        x, # weighted Tucker tensor train
):
    """Reduce the first and last dimensions of the first and last tt cores to 1.

    PARKED weighted-layer copy (unexported; kept pending the post-1.0 weighted redesign).
    Calling it emits a UserWarning.
    """
    import warnings
    warnings.warn(
        "wt3_squash_tails is PARKED weighted-layer code: untested and scheduled for redesign. "
        "Results may be wrong.",
        UserWarning, stacklevel=2,
    )
    xnp, _, _ = get_backend(False)

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


def t3_sum(
        x: typ.Tuple[
            typ.Tuple[NDArray, ...],  # tucker_cores. len=d, elm_shape=stack_shape+(ni,Ni)
            typ.Tuple[NDArray, ...],  # tt_cores.     len=d, elm_shape=stack_shape+(rLi,ni,rR(i+1))
        ],
        axis: typ.Union[int, typ.Sequence[int], None] = None,  # modes to sum (None -> all); negatives wrap
):  # -> T3 data tuple over the remaining modes, or a scalar NDArray (shape=stack_shape) if all modes summed
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


def t3_absorb_weights(
        x0:      typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        weights: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_weights, tt_weights)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # weighted tucker_cores, len=d,   elm_shape=stack_shape+(ni, Ni)
    typ.Tuple[NDArray, ...],  # weighted tt_cores,     len=d,   elm_shape=stack_shape+(ri, ni, r(i+1))
]:
    """Contract diagonal edge weights into a Tucker tensor train's cores (shape-preserving).

    ``weights = (tucker_weights, tt_weights)`` with ``tucker_weights`` len=d, elm_shape
    ``stack_shape+(ni,)`` and ``tt_weights`` len=d+1, elm_shape ``stack_shape+(ri,)`` -- one diagonal
    (stored as its vector) per internal edge. The result is a plain ``(tucker_cores, tt_cores)`` whose
    dense value is the fully-weighted network. Side-convention (library-decided):

    - **Tucker weights → the Tucker cores** (the rank leg): ``'...i,...io->...io'``.
    - **TT bond weights leftward**: bond ``r(k+1)`` into its left-neighbour core ``G_k``'s right leg;
      the leftmost boundary bond ``r0`` (no left neighbour) is absorbed **rightward** into ``G_0``'s left
      leg. Each of the d+1 bonds is absorbed exactly once.

    Stacking rides the leading ``'...'`` (weights share the cores' ``C`` stack). jax-ness is inferred.
    """
    xnp, _, _ = get_backend(False, tree_contains_jax((x0, weights)))
    tucker_cores0, tt_cores0 = x0
    tucker_weights, tt_weights = weights

    tucker_cores = tuple(xnp.einsum('...i,...io->...io', w, B)
                         for w, B in zip(tucker_weights, tucker_cores0))

    tt_cores = []
    for k, G in enumerate(tt_cores0):
        Gk = xnp.einsum('...iaj,...j->...iaj', G, tt_weights[k + 1])   # bond r(k+1) leftward into G_k's right leg
        if k == 0:
            Gk = xnp.einsum('...i,...iaj->...iaj', tt_weights[0], Gk)  # boundary r0 rightward into G_0's left leg
        tt_cores.append(Gk)

    return tucker_cores, tuple(tt_cores)


def t3_weights_consistent(
        x0:      typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        weights: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_weights, tt_weights)
) -> bool:                                                                 # True iff shape-consistent
    """True iff the weight tuple's lengths, per-edge ranks, and stack_shape match the T3 ``x0``
    (a non-raising structural shape predicate)."""
    tucker_cores, tt_cores = x0
    tucker_weights, tt_weights = weights
    d = len(tucker_cores)
    if len(tucker_weights) != d or len(tt_weights) != d + 1:
        return False
    stack = tucker_cores[0].shape[:-2]
    tucker_ranks = tuple(B.shape[-2] for B in tucker_cores)
    tt_ranks = tuple(G.shape[-3] for G in tt_cores) + (tt_cores[-1].shape[-1],)
    for w, n in zip(tucker_weights, tucker_ranks):
        if w.shape != stack + (n,):
            return False
    for w, r in zip(tt_weights, tt_ranks):
        if w.shape != stack + (r,):
            return False
    return True
