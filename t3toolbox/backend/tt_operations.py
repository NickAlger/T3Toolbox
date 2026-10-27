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
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.linalg as linalg

__all__ = [
    'tt_reverse',
    'tt_squash_tails',
    'tt_change_core_shapes',
    'tt_zipper_left_to_right',
    'tt_zipper_right_to_left',
    'tt_zipper_tangent_left_to_right',
    'tt_zipper_tangent_right_to_left',
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


def _tt_zipper_tangent_step(
        Zdot:   NDArray,                    # carry: K+C+(rAi, rBi)
        data:   typ.Tuple[NDArray, NDArray, NDArray,
                          NDArray, NDArray],  # (Z_i, A_i, B_i, Adot_i, Bdot_i): C ; C ; C ; K+C ; K+C
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:  # (next carry, (Zdot,))
    '''One core of :py:func:`tt_zipper_tangent_left_to_right`: the product rule of
    ``Z_(i+1) = Z_i A_i B_i`` (carry, A-direction and B-direction terms). Closure-free scan body --
    ``docs/contributor/scan_body_principles.md`` (``contract`` dispatches on its own operands).'''
    Z, A, B, Adot, Bdot = data
    Zdot_next = (contractions.contract('KCij,Ciak,Cjal->KCkl', Zdot, A, B)
                 + contractions.contract('Cij,KCiak,Cjal->KCkl', Z, Adot, B)
                 + contractions.contract('Cij,Ciak,KCjal->KCkl', Z, A, Bdot))
    return Zdot_next, (Zdot,)


def tt_zipper_tangent_left_to_right(
        coresA:     typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=C+(rAi, ni, rA(i+1))
        coresB:     typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=C+(rBi, ni, rB(i+1))
        coresA_dot: typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=K+C+(rAi, ni, rA(i+1))
        coresB_dot: typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=K+C+(rBi, ni, rB(i+1))

        zippers:    typ.Optional[typ.Sequence[NDArray]] = None,  # len=d+1, elm_shape=C+(rAi, rBi); None -> computed
) -> typ.Tuple[NDArray, ...]:  # zipper tangents, len=d+1, elm_shape=K+C+(rAi, rBi); the first is 0
    """The directional derivative of :py:func:`tt_zipper_left_to_right` along both chains.

    With ``Z = tt_zipper_left_to_right(A, B)`` (``Z_0`` the boundary ones, ``Z_(i+1) = Z_i A_i B_i``), this
    returns ``Zdot_i = d/ds Z_i(A + s Adot, B + s Bdot)`` at ``s = 0``: ``Zdot_0 = 0`` and, by the product
    rule, ``Zdot_(i+1) = Zdot_i A_i B_i + Z_i Adot_i B_i + Z_i A_i Bdot_i`` -- one scan, three grouped terms
    per core. It is the "exactly one core replaced" companion of a zipper that a two-hole (second
    directional derivative) sweep carries beside the one-hole environment.

    Batching: the chains and the zippers carry ``C``; the two direction chains carry the same ``K + C``
    (a stack of directions at one point), which the tangents inherit. ``zippers`` may be passed when the
    caller already holds them (a precomputed sweep); ``None`` computes them.

    **Polymorphic over the representation**, like its parent: a ragged core tuple or a uniform TT
    supercore (``(d,)+stack+(r, n, r)``). Named like its parent too -- the chain helpers keep descriptive
    names (a cataloged exception, ``docs/naming_conventions.md``).

    Examples
    --------
    The tangent agrees with a central difference of the zipper (to the difference's ``O(h^2)``):

    >>> import numpy as np
    >>> import t3toolbox.backend.tt_operations as tto
    >>> np.random.seed(0)
    >>> shapes_A, shapes_B = [(1, 3, 2), (2, 4, 3), (3, 2, 1)], [(1, 3, 3), (3, 4, 2), (2, 2, 1)]
    >>> A = tuple(np.random.randn(*s) for s in shapes_A)
    >>> B = tuple(np.random.randn(*s) for s in shapes_B)
    >>> Ad = tuple(np.random.randn(5, *s) for s in shapes_A)              # K = (5,) directions
    >>> Bd = tuple(np.random.randn(5, *s) for s in shapes_B)
    >>> Zdot = tto.tt_zipper_tangent_left_to_right(A, B, Ad, Bd)
    >>> print([z.shape for z in Zdot])
    [(5, 1, 1), (5, 2, 3), (5, 3, 2), (5, 1, 1)]
    >>> h = 1e-3
    >>> zipper_at = lambda t: tto.tt_zipper_left_to_right([a + t * ad[0] for a, ad in zip(A, Ad)],
    ...                                                   [b + t * bd[0] for b, bd in zip(B, Bd)])
    >>> print(all(np.allclose(zd[0], (p - m) / (2 * h))
    ...           for zd, p, m in zip(Zdot, zipper_at(h), zipper_at(-h))))
    True
    """
    is_uniform = is_ndarray(coresA)   # uniform: ONE supercore; ragged: a tuple of cores
    use_jax = tree_contains_jax((coresA, coresB, coresA_dot, coresB_dot, zippers))
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    if zippers is None:
        zippers = tt_zipper_left_to_right(coresA, coresB)
    d = len(coresA)
    zippers_in = xnp.stack(tuple(zippers[:d])) if is_uniform else tuple(zippers[:d])

    ss = np.broadcast_shapes(coresA_dot[0].shape[:-3], coresB_dot[0].shape[:-3])   # K + C
    Zdot0 = xnp.zeros(ss + (coresA[0].shape[-3], coresB[0].shape[-3]))
    Zdot_f, (ZZdot_first,) = xscan(_tt_zipper_tangent_step, Zdot0,
                                   (zippers_in, coresA, coresB, coresA_dot, coresB_dot))
    return tuple(ZZdot_first) + (Zdot_f,)


def tt_zipper_tangent_right_to_left(
        coresA:     typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=C+(rAi, ni, rA(i+1))
        coresB:     typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=C+(rBi, ni, rB(i+1))
        coresA_dot: typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=K+C+(rAi, ni, rA(i+1))
        coresB_dot: typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=K+C+(rBi, ni, rB(i+1))

        zippers:    typ.Optional[typ.Sequence[NDArray]] = None,  # len=d+1, elm_shape=C+(rA(i+1), rB(i+1)); None -> computed
) -> typ.Tuple[NDArray, ...]:  # zipper tangents, len=d+1, elm_shape=K+C+(rA(i+1), rB(i+1)); the last is 0
    """As :py:func:`tt_zipper_tangent_left_to_right`, for :py:func:`tt_zipper_right_to_left` (``zippers``
    are that function's output). Polymorphic via :py:func:`tt_reverse`."""
    rev = tt_reverse
    rev_zippers = None if zippers is None else tuple(zippers)[::-1]
    out = tt_zipper_tangent_left_to_right(rev(coresA), rev(coresB), rev(coresA_dot), rev(coresB_dot),
                                          zippers=rev_zippers)
    return out[::-1]
