# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
from __future__ import annotations

import typing as typ
import numpy as np

import t3toolbox.backend.bv_conversions as bv_conversions
import t3toolbox.backend.t3_operations as ragged_operations
from t3toolbox.backend.common import *

__all__ = [
    'tangent_to_dense',
    'tangent_to_t3',
    'orthogonal_gauge_projection',
    'oblique_gauge_projection',
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
        use_jax:        bool = False,
) -> NDArray:  # dense tangent vector. shape=stack_shape+(N0,...,N(d-1))
    """Form the dense tensor represented by a basis-variations tangent vector.

    The tangent vector is the sum of the 2d single-core-replacement terms (one per Tucker hole
    and one per TT hole). This is stack-aware: leading stack axes ride along through ``bv_to_t3``
    and ``to_dense``.
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations

    num_cores = len(tucker_variations)

    terms = [bv_conversions.bv_to_t3((False, ii), basis, variations, use_jax=use_jax) for ii in range(num_cores)]
    terms += [bv_conversions.bv_to_t3((True, ii), basis, variations, use_jax=use_jax) for ii in range(num_cores)]

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
        use_jax:    bool = False,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # gauged_tucker_variations
    typ.Tuple[NDArray, ...],  # gauged_tt_variations
]:
    """Project the variations onto the gauge-satisfying subspace (orthogonal projection).

    Changes the represented tangent vector. The result satisfies, for an orthogonal basis,
    ``U_i V_i^T = 0`` (all i) and ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` (i = 0..d-2).
    Stack-aware. Ragged path only (uniform deferred).
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations

    use_jax = use_jax or tree_contains_jax((basis, variations))
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
        use_jax:    bool = False,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # gauged_tucker_variations
    typ.Tuple[NDArray, ...],  # gauged_tt_variations
]:
    """Project the variations onto the gauge-satisfying subspace while preserving the tangent vector.

    Generalizes Holtz, Rohwedder & Schneider (2012), "On manifolds of tensors of fixed TT-rank".
    The Tucker perturbation is made perpendicular to U (compensating through the down/outer cores),
    then the TT variations are made left-perpendicular (compensating through the right cores).
    Stack-aware. Ragged path only (uniform deferred).
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations

    use_jax = use_jax or tree_contains_jax((basis, variations))
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
        use_jax:        bool = False,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores (doubled Tucker ranks)
    typ.Tuple[NDArray, ...],  # tt_cores     (doubled TT ranks)
]:
    """Doubled-rank Tucker tensor train representing a basis-variations tangent vector.

    The Tucker cores become ``[U_i; V_i]`` (stacked along the Tucker-rank axis); the TT cores form
    the standard block-bidiagonal embedding. With ``include_shift=True`` the base point is folded
    into the last TT core so the result represents ``base point + v``. Stack-aware.
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    tucker_variations, tt_variations = variations

    use_jax = use_jax or tree_contains_jax((basis, variations))
    xnp, _, _ = get_backend(False, use_jax)

    ss = up_tucker_cores[0].shape[:-2]  # stack_shape
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
