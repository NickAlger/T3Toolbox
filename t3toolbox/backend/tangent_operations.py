# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
from __future__ import annotations

import typing as typ
import numpy as np

import t3toolbox.backend.bv_conversions as bv_conversions
import t3toolbox.backend.t3_operations as ragged_operations
import t3toolbox.backend.t3_svd as ragged_t3svd
import t3toolbox.backend.stacking as stacking
from t3toolbox.backend.common import *

__all__ = [
    'tangent_to_dense',
    'tangent_to_t3',
    'orthogonal_gauge_projection',
    'oblique_gauge_projection',
    'tt_zipper_left_to_right',
    'tt_zipper_right_to_left',
    'project_t3_onto_tangent_space',
    'unstack_tangent_stack',
    'stack_tangent_stack',
    'unstack_base_stack',
    'stack_base_stack',
    'gauge_residual',
    'retract',
]


def tangent_to_dense(
        basis:      typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        variations: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations
            typ.Sequence[NDArray],  # tt_variations
        ],
        include_shift:  bool = False,  # False: tangent vector v. True: base point + v.
) -> NDArray:  # dense tangent vector. shape=stack_shape+(N0,...,N(d-1))
    """Form the dense tensor represented by a basis-variations tangent vector.

    The tangent vector is the sum of the 2d single-core-replacement terms (one per Tucker hole
    and one per TT hole). This is stack-aware: leading stack axes ride along through ``bv_to_t3``
    and ``to_dense``.
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations

    num_cores = len(tucker_variations)

    terms = [bv_conversions.bv_to_t3((False, ii), basis, variations) for ii in range(num_cores)]
    terms += [bv_conversions.bv_to_t3((True, ii), basis, variations) for ii in range(num_cores)]

    V = ragged_operations.to_dense(terms[0])
    for term in terms[1:]:
        V = V + ragged_operations.to_dense(term)

    if include_shift:
        P = ragged_operations.to_dense((up_tucker_cores, left_tt_cores))
        V = P + V

    return V


def orthogonal_gauge_projection(
        basis:      typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        variations: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations
            typ.Sequence[NDArray],  # tt_variations
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # gauged_tucker_variations
    typ.Tuple[NDArray, ...],  # gauged_tt_variations
]:
    """Project the variations onto the gauge-satisfying subspace (orthogonal projection).

    Changes the represented tangent vector. The result satisfies, for an orthogonal basis,
    ``U_i V_i^T = 0`` (all i) and ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` (i = 0..d-2).
    Stack-aware. Ragged path only (uniform deferred).

    Gauge conditions (48)-(49), Appendix A.3, of Alger et al. (2026), "Tucker Tensor Train
    Taylor Series" (arXiv:2603.21141).
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations

    use_jax = tree_contains_jax((basis, variations))
    xnp, _, _ = get_backend(False, use_jax)

    # TT variations: remove the component parallel to the left cores (all but the last)
    new_tt_variations = []
    for dV, L in zip(tt_variations[:-1], left_tt_cores[:-1]):
        parallel = xnp.einsum('...iaj,...jk->...iak', L, xnp.einsum('...iaj,...iak->...jk', L, dV))
        new_tt_variations.append(dV - parallel)
    new_tt_variations.append(tt_variations[-1])

    # Tucker variations: remove the component parallel to the up cores
    new_tucker_variations = []
    for dB, U in zip(tucker_variations, up_tucker_cores):
        parallel = xnp.einsum('...jk,...ko->...jo', xnp.einsum('...jo,...ko->...jk', dB, U), U)
        new_tucker_variations.append(dB - parallel)

    return tuple(new_tucker_variations), tuple(new_tt_variations)


def oblique_gauge_projection(
        basis:      typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        variations: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations
            typ.Sequence[NDArray],  # tt_variations
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # gauged_tucker_variations
    typ.Tuple[NDArray, ...],  # gauged_tt_variations
]:
    """Project the variations onto the gauge-satisfying subspace while preserving the tangent vector.

    Generalizes Holtz, Rohwedder & Schneider (2012), "On manifolds of tensors of fixed TT-rank".
    The Tucker perturbation is made perpendicular to U (compensating through the down/outer cores),
    then the TT variations are made left-perpendicular (compensating through the right cores).
    Stack-aware. Ragged path only (uniform deferred).

    Enforces the gauge conditions (48)-(49), Appendix A.3, of Alger et al. (2026), "Tucker Tensor
    Train Taylor Series" (arXiv:2603.21141).
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations

    use_jax = tree_contains_jax((basis, variations))
    xnp, _, _ = get_backend(False, use_jax)

    num_cores = len(tucker_variations)
    tucker_vars = list(tucker_variations)
    tt_vars = list(tt_variations)

    # Make Tucker variations perpendicular to U; compensate through the down/outer cores.
    for ii in range(num_cores):
        U = up_tucker_cores[ii]
        dB = tucker_vars[ii]
        O = down_tt_cores[ii]
        dG = tt_vars[ii]

        X_ji = xnp.einsum('...jo,...io->...ji', dB, U)
        dB_parallel = xnp.einsum('...ji,...io->...jo', X_ji, U)
        tucker_vars[ii] = dB - dB_parallel
        tt_vars[ii] = dG + xnp.einsum('...aib,...ij->...ajb', O, X_ji)

    # Make TT variations left-perpendicular; compensate through the right cores.
    for ii in range(num_cores - 1):
        L = left_tt_cores[ii]
        R = right_tt_cores[ii + 1]
        dG1 = tt_vars[ii]
        X = xnp.einsum('...iaj,...iak->...jk', L, dG1)
        tt_vars[ii] = dG1 - xnp.einsum('...iaj,...jk->...iak', L, X)
        tt_vars[ii + 1] = tt_vars[ii + 1] + xnp.einsum('...jk,...kbl->...jbl', X, R)

    return tuple(tucker_vars), tuple(tt_vars)


def tangent_to_t3(
        basis:      typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        variations: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations
            typ.Sequence[NDArray],  # tt_variations
        ],
        include_shift:  bool = False,  # False: tangent vector v. True: base point + v.
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores (doubled Tucker ranks)
    typ.Tuple[NDArray, ...],  # tt_cores     (doubled TT ranks)
]:
    """Doubled-rank Tucker tensor train representing a basis-variations tangent vector.

    The Tucker cores become ``[U_i; V_i]`` (stacked along the Tucker-rank axis); the TT cores form
    the standard block-bidiagonal embedding. With ``include_shift=True`` the base point is folded
    into the last TT core so the result represents ``base point + v``. Stack-aware.

    Equations (50)-(53) and Figure 20, Appendix A.3.1, of Alger et al. (2026),
    "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations

    use_jax = tree_contains_jax((basis, variations))
    xnp, _, _ = get_backend(False, use_jax)

    # The output is a single uniformly-stacked doubled-rank T3 (one per (v, g) pair). Its full stack
    # V+G comes from the variations; the base cores carry only G (the shared base point), so broadcast
    # every base-derived core up to V+G -- replicating the base point over the tangent stack V -- so
    # each concatenated doubled-rank core is uniformly stacked. No-op when V=() (the plain G-stack).
    ss = tucker_variations[0].shape[:-2]  # stack_shape = V + G
    bcast2 = lambda C: xnp.broadcast_to(C, ss + C.shape[-2:])  # tucker-shaped base core (..., n, N)
    bcast3 = lambda C: xnp.broadcast_to(C, ss + C.shape[-3:])  # tt-shaped base core (..., rL, n, rR)
    up_tucker_cores = [bcast2(U) for U in up_tucker_cores]
    down_tt_cores   = [bcast3(O) for O in down_tt_cores]
    left_tt_cores   = [bcast3(L) for L in left_tt_cores]
    right_tt_cores  = [bcast3(R) for R in right_tt_cores]

    num_cores = len(up_tucker_cores)

    # Tucker cores: [U_i ; V_i] stacked along the Tucker-rank axis.
    x_tucker_cores = [xnp.concatenate([U, V], axis=-2) for U, V in zip(up_tucker_cores, tucker_variations)]

    if num_cores == 1:
        H = tt_variations[0]
        O = down_tt_cores[0]
        if include_shift:
            H = left_tt_cores[0] + H
        G = xnp.concatenate([H, O], axis=-2)
        return (x_tucker_cores[0],), (G,)

    x_tt_cores = []

    # First TT core.
    dU = tt_variations[0]
    O = down_tt_cores[0]
    L = left_tt_cores[0]
    Z = xnp.zeros(ss + (O.shape[-3], O.shape[-2], L.shape[-1]))
    G_top = xnp.concatenate([dU, L], axis=-1)
    G_bot = xnp.concatenate([O, Z], axis=-1)
    G = xnp.concatenate([G_top, G_bot], axis=-2)
    x_tt_cores.append(G)

    # Middle TT cores.
    for ii in range(1, num_cores - 1):
        L = left_tt_cores[ii]
        R = right_tt_cores[ii]
        O = down_tt_cores[ii]
        dU = tt_variations[ii]
        Z001 = xnp.zeros(ss + (R.shape[-3], dU.shape[-2], L.shape[-1]))
        Z100 = xnp.zeros(ss + (R.shape[-3], O.shape[-2], R.shape[-1]))
        Z101 = xnp.zeros(ss + (R.shape[-3], O.shape[-2], L.shape[-1]))
        Z111 = xnp.zeros(ss + (L.shape[-3], O.shape[-2], L.shape[-1]))
        G_top = xnp.concatenate([
            xnp.concatenate([R, Z001], axis=-1),
            xnp.concatenate([dU, L], axis=-1),
        ], axis=-3)
        G_bot = xnp.concatenate([
            xnp.concatenate([Z100, Z101], axis=-1),
            xnp.concatenate([O, Z111], axis=-1),
        ], axis=-3)
        G = xnp.concatenate([G_top, G_bot], axis=-2)
        x_tt_cores.append(G)

    # Last TT core.
    dU = tt_variations[-1]
    R = right_tt_cores[-1]
    O = down_tt_cores[-1]
    Z = xnp.zeros(ss + (R.shape[-3], O.shape[-2], R.shape[-1]))
    if include_shift:
        Lf = left_tt_cores[-1]
        G_top = xnp.concatenate([R, Lf + dU], axis=-3)
    else:
        G_top = xnp.concatenate([R, dU], axis=-3)
    G_bot = xnp.concatenate([Z, O], axis=-3)
    G = xnp.concatenate([G_top, G_bot], axis=-2)
    x_tt_cores.append(G)

    return tuple(x_tucker_cores), tuple(x_tt_cores)


def tt_zipper_left_to_right(
        coresA:     typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rAi, ni, rA(i+1))
        coresB:     typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rBi, ni, rB(i+1))
) -> typ.Tuple[NDArray, ...]:  # zipper matrices, len=d+1, elm_shape=stack_shape+(rAi, rBi)
    """Accumulate left-to-right the partial contractions of two TT chains sharing tensor indices.

    Returns d+1 matrices Z_i; Z_0 is the (left-boundary) ones matrix and Z_(i+1) contracts Z_i with
    cores A_i, B_i. Stack-aware.
    """
    use_jax = tree_contains_jax((coresA, coresB))
    xnp, _, xscan = get_backend(False, use_jax)

    def _func(Z, GA_GB):
        GA, GB = GA_GB
        Z_next = xnp.einsum('...ij,...iak,...jal->...kl', Z, GA, GB)
        return Z_next, (Z,)

    ss = coresA[0].shape[:-3]
    Z0 = xnp.ones(ss + (coresA[0].shape[-3], coresB[0].shape[-3]))
    Zf, (ZZ_first,) = xscan(_func, Z0, (coresA, coresB))
    return tuple(ZZ_first) + (Zf,)


def tt_zipper_right_to_left(
        coresA:     typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rAi, ni, rA(i+1))
        coresB:     typ.Sequence[NDArray],  # len=d, elm_shape=stack_shape+(rBi, ni, rB(i+1))
) -> typ.Tuple[NDArray, ...]:  # zipper matrices, len=d+1, elm_shape=stack_shape+(rA(i+1), rB(i+1))
    """As :py:func:`tt_zipper_left_to_right`, accumulating right-to-left."""
    rev = tt_zipper_left_to_right(
        ragged_operations.reverse_tt(coresA), ragged_operations.reverse_tt(coresB),
    )
    return rev[::-1]


def project_t3_onto_tangent_space(
        basis:      typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        x:          typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores
            typ.Sequence[NDArray],  # tt_cores
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # gauged tucker_variations
    typ.Tuple[NDArray, ...],  # gauged tt_variations
]:
    """Orthogonal projection of a Tucker tensor train onto the tangent space at an orthogonal base.

    Returns gauged variations representing the orthogonal projection of ``x`` *directly* onto the
    tangent space (a linear subspace); it does not subtract the base point. The base must be an
    orthogonal, minimal-rank representation. Stack-aware. Ragged path only (uniform deferred).
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    outer_tt_cores = down_tt_cores

    other_tucker_cores, other_tt_cores = x
    other_tt_cores = ragged_operations.squash_tt_tails(other_tt_cores)

    use_jax = tree_contains_jax((basis, x))
    xnp, xmap, _ = get_backend(False, use_jax)

    # Re-express the other T3's TT cores in the base's up-Tucker basis.
    def _func1(args):
        G_other, B_other, U = args
        BU = xnp.einsum('...iz,...xz->...ix', B_other, U)
        G_other2 = xnp.einsum('...aib,...ix->...axb', G_other, BU)
        return (G_other2,)

    (other_tt_cores2,) = xmap(_func1, (other_tt_cores, other_tucker_cores, up_tucker_cores))

    zipper_left2right = tt_zipper_left_to_right(other_tt_cores2[:-1], left_tt_cores[:-1])
    zipper_right2left = tt_zipper_right_to_left(other_tt_cores2[1:], right_tt_cores[1:])

    def _func2(args):
        ZL, ZR, G, B, O, U = args
        env = xnp.einsum('...ax,...aib,...by->...xiy', ZL, G, ZR)
        BU = xnp.einsum('...io,...jo->...ij', B, U)
        dG = xnp.einsum('...xiy,...ij->...xjy', env, BU)
        M = xnp.einsum('...xiy,...xjy->...ij', env, O)
        dB = xnp.einsum('...ij,...io->...jo', M, B)
        return dG, dB

    ungauged_tt_variations, ungauged_tucker_variations = xmap(
        _func2,
        (zipper_left2right, zipper_right2left, other_tt_cores, other_tucker_cores, outer_tt_cores, up_tucker_cores),
    )

    return orthogonal_gauge_projection(
        basis, (ungauged_tucker_variations, ungauged_tt_variations),
    )


def _tangent_stack_split(
        basis,       # (UU, DD, LL, RR), each core stack = G
        variations,  # (VV, HH), each core stack = V + G
) -> typ.Tuple[int, int]:  # (|V|, |G|)
    """Recover the tangent-stack / base-stack split from a (basis, variations) data pair.

    The base cores carry only the base stack G; the variation cores carry the full stack V + G (the
    extra tangent stack V outermost). So |G| comes from a base core and |V| is the remainder.
    """
    n_base = len(basis[0][0].shape) - 2       # |G|   (up core: stack + (nU, N))
    n_full = len(variations[0][0].shape) - 2  # |V+G| (tucker variation: stack + (nD, N))
    return n_full - n_base, n_base


def _pair_base_leaves(basis_tree, variations_tree, n_base):
    """Pair a basis-data tree and a variations-data tree (same G-shaped outer structure, n_base axes
    deep) leaf-by-leaf into one G-shaped tree of ``(basis_data, variations_data)`` pairs.

    Not a :py:func:`stacking.tree_zip`: the data-tuple leaves are themselves sequences, so tree_zip
    would recurse into the cores. We stop at the known base-stack depth ``n_base`` instead.
    """
    if n_base == 0:
        return (basis_tree, variations_tree)  # both are single data tuples -> one pair
    return tuple(_pair_base_leaves(b, v, n_base - 1)
                 for b, v in zip(basis_tree, variations_tree))


def _unpair_base_leaves(paired_tree, n_base):
    """Inverse of :py:func:`_pair_base_leaves`: split a G-shaped tree of ``(basis_data,
    variations_data)`` pairs back into a ``(basis_tree, variations_tree)``.
    """
    if n_base == 0:
        return paired_tree  # already a single (basis_data, variations_data) pair
    split = [_unpair_base_leaves(p, n_base - 1) for p in paired_tree]
    return tuple(s[0] for s in split), tuple(s[1] for s in split)


def unstack_tangent_stack(
        basis,       # (UU, DD, LL, RR), each core stack = G
        variations,  # (VV, HH), each core stack = V + G
):  # -> array-like tree (shape V) of variations-data tuples (each stack = G)
    """Peel the tangent stack V off the variations, returning a V-shaped tree of variation-data.

    The base point is shared across V, so the base cores are untouched (the caller pairs the same
    base with every leaf). Inverse of :py:func:`stack_tangent_stack`.
    """
    n_tangent, _ = _tangent_stack_split(basis, variations)
    return stacking.unstack(variations, axes=tuple(range(n_tangent)))


def stack_tangent_stack(
        variations_tree,  # array-like tree (shape V) of variations-data tuples (each stack = G)
):  # -> variations-data tuple (stack = V + G)
    """Stack a V-shaped tree of variation-data over the tangent stack V (outermost).

    Inverse of :py:func:`unstack_tangent_stack`.
    """
    return stacking.basic_ragged_stack(variations_tree)


def unstack_base_stack(
        basis,       # (UU, DD, LL, RR), each core stack = G
        variations,  # (VV, HH), each core stack = V + G
):  # -> array-like tree (shape G) of (basis_data, variations_data) pairs
    """Peel the base stack G off both the basis and the variations, returning a G-shaped tree whose
    leaves are ``(basis_data, variations_data)`` pairs -- one single-base-point tangent per leaf.

    Each basis-data leaf has stack () (a single base point); each variations-data leaf has stack V.
    The base stack is the *inner* part of the variation stack (V + G), so it is peeled from the
    interior axes of the variation cores. The basis and variation leaves are paired for you (a plain
    :py:func:`stacking.tree_zip` cannot do it -- it would recurse into the data-tuple leaves -- so a
    backend user would otherwise have to hand-roll a depth-aware zip). Inverse of
    :py:func:`stack_base_stack`.
    """
    n_tangent, n_base = _tangent_stack_split(basis, variations)
    basis_tree = stacking.unstack(basis, axes=tuple(range(n_base)))
    variations_tree = stacking.unstack(variations, axes=tuple(range(n_tangent, n_tangent + n_base)))
    return _pair_base_leaves(basis_tree, variations_tree, n_base)


def stack_base_stack(
        paired_tree,  # array-like tree (shape G) of (basis_data, variations_data) pairs
):  # -> (
    #        basis-data,       # stack = G
    #        variations-data,  # stack = V + G
    #    )
    """Stack a G-shaped tree of ``(basis_data, variations_data)`` pairs over the base stack G.

    The base stack is placed *innermost* (the variation stack becomes V + G), matching the base-inner
    convention. Takes exactly the paired-tree layout that :py:func:`unstack_base_stack` produces (its
    inverse), so a backend user round-trips without splitting the pairs by hand.
    """
    n_base = stacking.tree_depth(paired_tree) - 3   # |G|; a (basis_data, variations_data) leaf is 3 levels deep
    basis_tree, variations_tree = _unpair_base_leaves(paired_tree, n_base)
    basis = stacking.basic_ragged_stack(basis_tree)                      # G at leading -> stack = G
    n_tangent = len(stacking.get_first_leaf(variations_tree).shape) - 2  # |V| (a tucker variation leaf)
    variations = stacking.stack(variations_tree, axes=tuple(range(n_tangent, n_tangent + n_base)))
    return basis, variations


def gauge_residual(
        basis: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        variations: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations
            typ.Sequence[NDArray],  # tt_variations
        ],
) -> float:
    '''Max violation of the gauge conditions for a tangent vector (over the whole stack).

    The gauged tangent space requires each tucker variation orthogonal to its up-core, and each
    left-interior tt variation orthogonal to its left-core (see :py:func:`orthogonal_gauge_projection`).
    Returns the max absolute gauge inner product; a caller thresholds it (``<= atol``).
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations
    xnp, _, _ = get_backend(False, tree_contains_jax((basis, variations)))
    devs = []
    for U, V in zip(up_tucker_cores, tucker_variations):
        g = xnp.einsum('...ia,...ja->...ij', U, V)
        devs.append(xnp.max(xnp.abs(g)))
    for L, H in zip(left_tt_cores[:-1], tt_variations[:-1]):
        g = xnp.einsum('...abi,...abj->...ij', L, H)
        devs.append(xnp.max(xnp.abs(g)))
    return xnp.max(xnp.stack(devs))


def retract(
        basis: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        variations: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations
            typ.Sequence[NDArray],  # tt_variations
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores (retracted T3, base-point ranks)
    typ.Tuple[NDArray, ...],  # tt_cores
]:
    '''Retract a basis-variations tangent vector onto the fixed-rank manifold.

    Forms the shifted doubled-rank embedding (base point + v) via :py:func:`tangent_to_t3`
    (``include_shift=True``) and truncates it back to the **base point's own ranks** -- the Tucker
    ``up`` ranks and ``left`` TT ranks read off the basis cores -- with the implicit T3-SVD, yielding
    a point on the manifold of the base point's ranks.

    The truncation is the implicit T3-SVD (Algorithm 10) of Alger et al. (2026), "Tucker Tensor
    Train Taylor Series" (arXiv:2603.21141).
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    shifted = tangent_to_t3(basis, variations, include_shift=True)
    up_ranks = tuple(U.shape[-2] for U in up_tucker_cores)
    left_ranks = tuple(L.shape[-3] for L in left_tt_cores) + (left_tt_cores[-1].shape[-1],)
    retracted, _, _ = ragged_t3svd.t3svd(shifted, max_tucker_ranks=up_ranks, max_tt_ranks=left_ranks)
    return retracted
