# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Operations on bare tensor-train chains (the ``tt`` family).

A "tt chain" is the TT core sequence alone -- no Tucker cores, no masks: ragged (a ``len=d``
tuple of ``stack_shape+(rLi,ni,rR(i+1))`` arrays) or uniform (one ``(d,)+stack_shape+(r,n,r)``
supercore). Every public function here is polymorphic over the two (inferred via
``is_ndarray``), following the shared-sweep precedent of ``tt_orthogonalization``; the ragged
name doubles as the polymorphic name, so there is no separate ``utt_`` twin.
"""
import numpy as np
import typing as typ

from t3toolbox.backend.common import *
import t3toolbox.backend.linalg as linalg

__all__ = [
    'tt_reverse',
    'tt_squash_tails',
    'tt_change_core_shapes',
    'tt_zipper_left_to_right',
    'tt_zipper_right_to_left',
]


def tt_reverse(
        tt_cores: typ.Union[
            typ.Sequence[NDArray],  # ragged. len=d, elm_shape=stack_shape+(rLi,ni,rR(i+1))
            NDArray,                # uniform. shape=(d,)+stack_shape+(r,n,r)
        ],
) -> typ.Union[
    typ.Tuple[NDArray, ...],  # reversed cores. len=d, elm_shape=stack_shape+(rR(i+1),ni,rLi)
    NDArray,                  # reversed supercore. shape=(d,)+stack_shape+(r,n,r)
]:
    """Reverse a tensor-train chain: reverse the mode order and swap the two bond axes of each core.
    """
    if is_ndarray(tt_cores):
        return tt_cores[::-1].swapaxes(-3, -1)
    return tuple(G.swapaxes(-3, -1) for G in tt_cores[::-1])


def tt_squash_tails(
        tt_cores: typ.Union[
            typ.Sequence[NDArray],  # ragged. len=d, elm_shape=stack_shape+(rLi,ni,rR(i+1))
            NDArray,                # uniform. shape=(d,)+stack_shape+(r,n,r)
        ],
) -> typ.Union[
    typ.Tuple[NDArray, ...],  # ragged: r0=rd=1. len=d, elm_shape=stack_shape+(1,n0,r1),...,(r(d-1),n(d-1),1)
    NDArray,                  # uniform: same shape, boundary bond summed into slot 0, rest zeroed
]:
    """Collapse the leading and trailing TT bonds to one without changing the represented tensor.

    Ragged cores shrink to boundary bond 1; a uniform supercore keeps its shape (the bond is
    summed into slot 0 and the remaining slots zeroed, so the boundary rank *mask* becomes 1).
    """
    if is_ndarray(tt_cores):
        return _tt_squash_tails_uniform(tt_cores)
    return _tt_squash_tails_ragged(tt_cores)


def _tt_squash_tails_ragged(
        tt_cores: typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rLi,ni,rR(i+1))
) -> typ.Tuple[NDArray, ...]:  # tt_cores with r0=rd=1. len=d, elm_shape=stack_shape+(1,n0,r1),...,(r(d-1),n(d-1),1)
    """Make leading and trailing TT ranks equal to 1 (r0=rd=1), without changing tensor being represented.
    """
    use_jax = any([is_jax_ndarray(G) for G in tt_cores])
    xnp, _, _ = get_backend(False, use_jax)

    #
    tt_cores = tuple(tt_cores)

    G0 = tt_cores[0]
    G0 = xnp.einsum('az,...aib->...zib', xnp.ones((G0.shape[-3],1)), G0)

    tt_cores = (G0,) + tt_cores[1:]

    Gf = tt_cores[-1]
    Gf = xnp.einsum('...aib,bz->...aiz', Gf, xnp.ones((Gf.shape[-1],1)))

    tt_cores = tt_cores[:-1] + (Gf,)

    return tt_cores


def _tt_squash_tails_uniform(
        tt_supercore: NDArray,  # shape=(d,)+stack_shape+(r,n,r)
) -> NDArray:                   # shape=(d,)+stack_shape+(r,n,r), leading/trailing bond summed into slot 0
    """Make the leading bond of the first TT core and the trailing bond of the last collapse to one,
    by summing them into slot 0 (and zeroing the rest), so the represented tensor is unchanged.
    """
    use_jax = is_jax_ndarray(tt_supercore)
    xnp, _, _ = get_backend(True, use_jax)

    stack_shape = tt_supercore.shape[1:-3]
    n = tt_supercore.shape[-2]
    r = tt_supercore.shape[-1]

    def squash_left(G):                                               # (1,)+stack+(r,n,r)
        return xnp.concatenate([
            xnp.sum(G, axis=-3, keepdims=True),                       # (1,)+stack+(1,n,r)
            xnp.zeros((1,) + stack_shape + (r - 1, n, r)),
        ], axis=-3)

    def squash_right(G):                                              # (1,)+stack+(r,n,r)
        return xnp.concatenate([
            xnp.sum(G, axis=-1, keepdims=True),                       # (1,)+stack+(r,n,1)
            xnp.zeros((1,) + stack_shape + (r, n, r - 1)),
        ], axis=-1)

    if tt_supercore.shape[0] == 1:
        # d = 1: the first and last core are the SAME core -- squash both bonds of it (the ragged twin does
        # this naturally); concatenating [first, middle, last] would duplicate it into a 2-core supercore.
        return squash_right(squash_left(tt_supercore))

    new_G0 = squash_left(tt_supercore[:1])
    GG_mid = tt_supercore[1:-1]
    new_Gf = squash_right(tt_supercore[-1:])
    return xnp.concatenate([new_G0, GG_mid, new_Gf], axis=0)


def tt_change_core_shapes(
        tt_cores:         typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rLi,ni,rR(i+1))
        new_tucker_ranks: typ.Sequence[int],      # len=d
        new_tt_ranks:     typ.Sequence[int],      # len=d+1
) -> typ.Tuple[NDArray, ...]:  # resized tt_cores. len=d, elm_shape=stack_shape+(new_rLi,new_ni,new_rR(i+1))
    """Increase/decrease Tucker and/or TT ranks for TT cores using zero padding/truncation.
    """
    use_jax = tree_contains_jax(tt_cores)
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


def _tt_zipper_step(
        Z:     NDArray,                      # carry: stack_shape+(rAi, rBi)
        GA_GB: typ.Tuple[NDArray, NDArray],  # (A_i, B_i): stack+(rAi,ni,rA(i+1)) ; stack+(rBi,ni,rB(i+1))
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (Z,))
    '''One core of the sweep of :py:func:`tt_zipper_left_to_right`. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((Z, GA_GB)))   # only xnp; it ignores the flag
    GA, GB = GA_GB
    Z_next = xnp.einsum('...ij,...iak,...jal->...kl', Z, GA, GB)
    return Z_next, (Z,)


def tt_zipper_left_to_right(
        coresA:     typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rAi, ni, rA(i+1))
        coresB:     typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rBi, ni, rB(i+1))
) -> typ.Tuple[NDArray, ...]:  # zipper matrices, len=d+1, elm_shape=stack_shape+(rAi, rBi)
    """Accumulate left-to-right the partial contractions of two TT chains sharing tensor indices.

    Returns d+1 matrices Z_i; Z_0 is the (left-boundary) ones matrix and Z_(i+1) contracts Z_i with
    cores A_i, B_i. Stack-aware. **Polymorphic over the representation** (the scan reuse rule): a ragged
    core tuple gives a ragged ``xscan`` and a tuple of d+1 matrices; a uniform TT *supercore*
    (``(d,)+stack+(rA, n, rB)``, detected as a bare ndarray) gives the uniform ``xscan`` over the leading
    mode axis -- still returned as a tuple of d+1 matrices (the caller stacks if it wants a supercore).
    """
    is_uniform = is_ndarray(coresA)   # uniform: ONE supercore; ragged: a tuple of cores
    use_jax = tree_contains_jax((coresA, coresB))
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    ss = coresA[0].shape[:-3]
    Z0 = xnp.ones(ss + (coresA[0].shape[-3], coresB[0].shape[-3]))
    Zf, (ZZ_first,) = xscan(_tt_zipper_step, Z0, (coresA, coresB))
    return tuple(ZZ_first) + (Zf,)


def tt_zipper_right_to_left(
        coresA:     typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rAi, ni, rA(i+1))
        coresB:     typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rBi, ni, rB(i+1))
) -> typ.Tuple[NDArray, ...]:  # zipper matrices, len=d+1, elm_shape=stack_shape+(rA(i+1), rB(i+1))
    """As :py:func:`tt_zipper_left_to_right`, accumulating right-to-left. Polymorphic via
    :py:func:`tt_reverse` (ragged core tuple or uniform supercore)."""
    rev = tt_zipper_left_to_right(tt_reverse(coresA), tt_reverse(coresB))
    return rev[::-1]
