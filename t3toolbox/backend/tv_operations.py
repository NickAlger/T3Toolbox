# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Tangent-variations operations on raw (frame, variations) data.

``tv_to_t3`` (the tangent SUM over variation terms -- vs ``fv_conversions.fv_to_t3``'s single
term), ``tv_to_dense``, the orthogonal/oblique gauge projections, the tangent-space projections
of t3/dense operands, ``tv_retract``, ``tv_gauge_residual``, and the tangent/frame stack
converters.
"""
from __future__ import annotations

import math
import typing as typ
import numpy as np

import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.t3_operations as ragged_operations
import t3toolbox.backend.ut3_operations as uniform_operations
import t3toolbox.backend.t3_svd as ragged_t3svd
import t3toolbox.backend.sharing as sharing_module
import t3toolbox.backend.stacking as stacking
from t3toolbox.backend.common import *
from t3toolbox.backend.tt_operations import tt_reverse, tt_zipper_left_to_right, tt_zipper_right_to_left

__all__ = [
    'tv_to_dense',
    'tv_to_t3',
    'tv_orthogonal_gauge_projection',
    'tv_oblique_gauge_projection',
    'tv_project_t3_onto_tangent_space',
    'tv_project_dense_onto_tangent_space',
    'tv_unstack_tangent_stack',
    'tv_stack_tangent_stack',
    'tv_unstack_frame_stack',
    'tv_stack_frame_stack',
    'tv_gauge_residual',
    'tv_retract',
]


def tv_to_dense(
        frame:      typ.Tuple[
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
    """Form the dense tensor represented by a frame-variations tangent vector.

    The tangent vector is the sum of the 2d single-core-replacement terms (one per Tucker hole
    and one per TT hole). This is stack-aware: leading stack axes ride along through ``fv_to_t3``
    and ``to_dense``.
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    tucker_variations, tt_variations = variations

    num_cores = len(tucker_variations)

    terms = [fv_conversions.fv_to_t3((False, ii), frame, variations) for ii in range(num_cores)]
    terms += [fv_conversions.fv_to_t3((True, ii), frame, variations) for ii in range(num_cores)]

    V = t3_conversions.t3_to_dense(terms[0])
    for term in terms[1:]:
        V = V + t3_conversions.t3_to_dense(term)

    if include_shift:
        P = t3_conversions.t3_to_dense((up_tucker_cores, left_tt_cores))
        V = P + V

    return V


def tv_orthogonal_gauge_projection(
        frame:      typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        variations: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations
            typ.Sequence[NDArray],  # tt_variations
        ],
        shared_data: typ.Optional['sharing_module.T3SharedFrameData'] = None,  # tied post-pass (SF-T3)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # gauged_tucker_variations
    typ.Tuple[NDArray, ...],  # gauged_tt_variations
]:
    """Project the variations onto the gauge-satisfying subspace (orthogonal projection).

    Changes the represented tangent vector. The result satisfies, for an orthogonal frame,
    ``U_i V_i^T = 0`` (all i) and ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` (i = 0..d-2).
    Stack-aware. Ragged path only (uniform deferred).

    With ``shared_data`` (the frame's SF-T3 companion,
    :py:func:`~t3toolbox.backend.sharing.fv_shared_frame_data`), the gauge projection is
    followed by the tied post-pass
    (:py:func:`~t3toolbox.backend.sharing.fv_share_tucker_variations`) -- the composition is
    the orthogonal projection onto the TIED gauged tangent subspace, since the tied subspace is
    contained in the gauged one. Default ``None``: unchanged behavior.

    Gauge conditions (48)-(49), Appendix A.3, of Alger et al. (2026), "Tucker Tensor Train
    Taylor Series" (arXiv:2603.21141).
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    tucker_variations, tt_variations = variations

    use_jax = tree_contains_jax((frame, variations))
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

    gauged = (tuple(new_tucker_variations), tuple(new_tt_variations))
    if shared_data is not None:
        gauged = sharing_module.fv_share_tucker_variations(gauged, shared_data)
    return gauged


def tv_oblique_gauge_projection(
        frame:      typ.Tuple[
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
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    tucker_variations, tt_variations = variations

    use_jax = tree_contains_jax((frame, variations))
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


def tv_to_t3(
        frame:      typ.Tuple[
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
        shared_data: typ.Optional['sharing_module.T3SharedFrameData'] = None,  # tied embedding (SF-T3)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores (doubled Tucker ranks)
    typ.Tuple[NDArray, ...],  # tt_cores     (doubled TT ranks)
]:
    """Doubled-rank Tucker tensor train representing a frame-variations tangent vector.

    The Tucker cores become ``[U_i; V_i]`` (stacked along the Tucker-rank axis); the TT cores form
    the standard block-bidiagonal embedding. With ``include_shift=True`` the base point is folded
    into the last TT core so the result represents ``base point + v``. Stack-aware.

    With ``shared_data`` (the frame's SF-T3 companion), the embedding is built TIED: per
    nontrivial group, the common gauged ambient direction ``Udot`` is recovered from the (tied)
    coordinates (:py:func:`~t3toolbox.backend.sharing.fv_tied_ambient_directions`) and takes the
    ``V_i`` slot at every group mode -- ONE array per group, so the doubled factors
    ``[U_g; Udot]`` are exactly equal across the group -- while the paired core block becomes
    the companion's center core ``H_i`` (the identity ``S_i``-absorbed-``O_i == H_i`` makes this
    an exact rewrite of each Tucker term). The naive ``[U_g; V_i]`` embedding is NOT tied (the
    ``V_i = S_i^T Udot`` differ across a group in value, and in shape when the ``nD_i``
    differ). Cf. the SF-ETT tangent representation, Molozhavenko & Rakhuba (2026) Sec. 5.2.

    Equations (50)-(53) and Figure 20, Appendix A.3.1, of Alger et al. (2026),
    "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    tucker_variations, tt_variations = variations

    if shared_data is not None and sharing_module.nontrivial_groups(shared_data.groups):
        udots = sharing_module.fv_tied_ambient_directions(variations, shared_data)
        tucker_variations = list(tucker_variations)
        down_tt_cores = list(down_tt_cores)
        for gi, group in enumerate(sharing_module.nontrivial_groups(shared_data.groups)):
            for jj, ii in enumerate(group):
                tucker_variations[ii] = udots[gi]                    # ONE array per group
                down_tt_cores[ii] = shared_data.centers[gi][jj]      # H_i replaces O_i (exact)
        tucker_variations = tuple(tucker_variations)
        down_tt_cores = tuple(down_tt_cores)

    use_jax = tree_contains_jax((frame, variations))
    xnp, _, _ = get_backend(False, use_jax)

    # The output is a single uniformly-stacked doubled-rank T3 (one per (v, g) pair). Its full stack
    # V+G comes from the variations; the frame cores carry only G (the shared base point), so broadcast
    # every frame-derived core up to V+G -- replicating the base point over the tangent stack V -- so
    # each concatenated doubled-rank core is uniformly stacked. No-op when V=() (the plain G-stack).
    ss = tucker_variations[0].shape[:-2]  # stack_shape = V + G
    bcast2 = lambda C: xnp.broadcast_to(C, ss + C.shape[-2:])  # tucker-shaped frame core (..., n, N)
    bcast3 = lambda C: xnp.broadcast_to(C, ss + C.shape[-3:])  # tt-shaped frame core (..., rL, n, rR)
    up_tucker_cores = [bcast2(U) for U in up_tucker_cores]
    down_tt_cores   = [bcast3(O) for O in down_tt_cores]
    left_tt_cores   = [bcast3(L) for L in left_tt_cores]
    right_tt_cores  = [bcast3(R) for R in right_tt_cores]

    num_cores = len(up_tucker_cores)

    # Tucker cores: [U_i ; V_i] stacked along the Tucker-rank axis.
    x_tucker_cores = [xnp.concatenate([U, V], axis=-2) for U, V in zip(up_tucker_cores, tucker_variations)]
    if shared_data is not None:
        # the group inputs are identical objects, so the concatenations are value-equal;
        # assign ONE array per group (structural tie, never floating-point agreement)
        for group in sharing_module.nontrivial_groups(shared_data.groups):
            for ii in group[1:]:
                x_tucker_cores[ii] = x_tucker_cores[group[0]]

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


def tv_project_t3_onto_tangent_space(
        frame:      typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        x:          typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores
            typ.Sequence[NDArray],  # tt_cores
        ],
        shared_data: typ.Optional['sharing_module.T3SharedFrameData'] = None,  # tied post-pass (SF-T3)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # gauged tucker_variations
    typ.Tuple[NDArray, ...],  # gauged tt_variations
]:
    """Orthogonal projection of a Tucker tensor train onto the tangent space at an orthogonal frame.

    Returns gauged variations representing the orthogonal projection of ``x`` *directly* onto the
    tangent space (a linear subspace); it does not subtract the base point. The frame must be an
    orthogonal representation (minimal rank is *not* required). Stack-aware. Ragged path only (uniform
    deferred).
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    outer_tt_cores = down_tt_cores

    other_tucker_cores, other_tt_cores = x
    other_tt_cores = tt_operations.tt_squash_tails(other_tt_cores)

    use_jax = tree_contains_jax((frame, x))
    xnp, xmap, _ = get_backend(False, use_jax)

    # Re-express the other T3's TT cores in the frame's up-Tucker basis.
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

    return tv_orthogonal_gauge_projection(
        frame, (ungauged_tucker_variations, ungauged_tt_variations), shared_data=shared_data,
    )


def tv_project_dense_onto_tangent_space(
        frame:  typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        Z:      NDArray,  # dense ambient tensor. shape = stack_shape + (N0, ..., N(d-1))
        shared_data: typ.Optional['sharing_module.T3SharedFrameData'] = None,  # tied post-pass (SF-T3)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # gauged tucker_variations
    typ.Tuple[NDArray, ...],  # gauged tt_variations
]:
    """Orthogonal projection of a *dense* tensor onto the tangent space at an orthogonal frame.

    Contraction-only: contracts ``Z`` directly against the frame's orthonormal frames -- no SVD and no
    large intermediate Tucker tensor train (unlike densifying ``Z`` with the T3-SVD first). Returns
    gauged variations representing the orthogonal projection of ``Z`` *directly* onto the tangent space
    (a linear subspace); it does not subtract the base point. Stack-aware (leading axes beyond the
    ``d`` tensor modes are a stack). Ragged path only (uniform deferred).

    Requires an **orthogonal** frame: the canonical conditions (U row-orthonormal, L/R left/right-
    canonical, O outer-orthonormal) make each surrounding frame an isometry -- so a bare contraction
    yields the orthogonal-projection coefficient -- and make the gauged single-core directions mutually
    orthogonal. A *minimal-rank* frame is **not** required.

    Algorithm. For each mode ``i``, reduce ``Z`` over every *other* mode against the frame chains -- the
    left interface ``(U, L)`` over modes ``< i`` and the right interface ``(U, R)`` over modes ``> i``
    -- leaving the single mode ``x_i`` open, giving the shared environment ``core_env_i`` of shape
    ``(r_i, N_i, r_{i+1})``. Both variations at ``i`` read off it: ``dG_i = <U_i, core_env_i>`` (the TT
    variation) and ``dU_i = <O_i, core_env_i>`` (the Tucker variation; ``O`` = the outer/down cores). A
    single left sweep builds the left-reduced environments; each slot finishes with a right reduction.
    Finally :py:func:`tv_orthogonal_gauge_projection` orthogonalizes the ``2d`` directions so the sum of
    their per-direction projections equals the projection onto the tangent space.
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    d = len(up_tucker_cores)

    use_jax = tree_contains_jax((frame, Z))
    xnp, _, _ = get_backend(False, use_jax)

    ns = Z.ndim - d                                    # number of leading stack axes
    stack = tuple(Z.shape[:ns])                        # stack_shape
    Ns = tuple(U.shape[-1] for U in up_tucker_cores)   # mode dimensions N_i

    # Left sweep: EL[i] is Z with modes 0..i-1 absorbed into the left bond r_i (modes i..d-1 still
    # open), shape stack + (r_i, N_i, N_{i+1}, ..., N_{d-1}).
    EL = [Z.reshape(stack + (1,) + Ns)]
    for ii in range(d - 1):
        cur = EL[ii]
        r_i = cur.shape[ns]
        cur = cur.reshape(stack + (r_i, Ns[ii], math.prod(Ns[ii + 1:])))
        cur = xnp.einsum('...axm,...nx->...anm', cur, up_tucker_cores[ii])  # absorb N_i into the Tucker rank
        cur = xnp.einsum('...anm,...anb->...bm', cur, left_tt_cores[ii])    # ... then into the left bond
        EL.append(cur.reshape(stack + (left_tt_cores[ii].shape[-1],) + Ns[ii + 1:]))

    # Each slot: right-reduce EL[i]'s remaining modes (i+1..d-1) into the right bond r_{i+1}, giving the
    # shared core_env_i, then read off both variations.
    ungauged_tucker_variations, ungauged_tt_variations = [], []
    for ii in range(d):
        cur = EL[ii]
        r_i, N_i = cur.shape[ns], cur.shape[ns + 1]
        keep = r_i * N_i
        cur = cur.reshape(stack + (keep,) + Ns[ii + 1:] + (1,))  # trailing 1 = right boundary bond r_d
        for jj in range(d - 1, ii, -1):
            lead = math.prod(cur.shape[ns:-2])
            N_j, b = cur.shape[-2], cur.shape[-1]
            cur = cur.reshape(stack + (lead, N_j, b))
            cur = xnp.einsum('...mxb,...nx->...mnb', cur, up_tucker_cores[jj])  # absorb N_j into the Tucker rank
            cur = xnp.einsum('...mnb,...anb->...ma', cur, right_tt_cores[jj])   # ... then into the right bond
            cur = cur.reshape(stack + (keep,) + Ns[ii + 1:jj] + (right_tt_cores[jj].shape[-3],))
        core_env = cur.reshape(stack + (r_i, N_i, cur.shape[-1]))  # (r_i, N_i, r_{i+1})
        ungauged_tt_variations.append(xnp.einsum('...axb,...nx->...anb', core_env, up_tucker_cores[ii]))
        ungauged_tucker_variations.append(xnp.einsum('...axb,...anb->...nx', core_env, down_tt_cores[ii]))

    return tv_orthogonal_gauge_projection(
        frame, (tuple(ungauged_tucker_variations), tuple(ungauged_tt_variations)),
        shared_data=shared_data,
    )


def _tangent_stack_split(
        frame,       # (UU, DD, LL, RR), each core stack = G
        variations,  # (VV, HH), each core stack = V + G
) -> typ.Tuple[int, int]:  # (|V|, |G|)
    """Recover the tangent-stack / frame-stack split from a (frame, variations) data pair.

    The frame cores carry only the frame stack G; the variation cores carry the full stack V + G (the
    extra tangent stack V outermost). So |G| comes from a frame core and |V| is the remainder.
    """
    n_frame = len(frame[0][0].shape) - 2       # |G|   (up core: stack + (nU, N))
    n_full = len(variations[0][0].shape) - 2  # |V+G| (tucker variation: stack + (nD, N))
    return n_full - n_frame, n_frame


def _pair_frame_leaves(
        frame_tree,        # G-shaped tree of frame-data tuples,      n_frame levels deep
        variations_tree,   # G-shaped tree of variations-data tuples, n_frame levels deep
        n_frame:     int,   # frame-stack depth |G|
):  # -> G-shaped tree of (frame_data, variations_data) pairs
    """Pair a frame-data tree and a variations-data tree (same G-shaped outer structure, n_frame axes
    deep) leaf-by-leaf into one G-shaped tree of ``(frame_data, variations_data)`` pairs.

    Not a :py:func:`stacking.tree_zip`: the data-tuple leaves are themselves sequences, so tree_zip
    would recurse into the cores. We stop at the known frame-stack depth ``n_frame`` instead.
    """
    if n_frame == 0:
        return (frame_tree, variations_tree)  # both are single data tuples -> one pair
    return tuple(_pair_frame_leaves(b, v, n_frame - 1)
                 for b, v in zip(frame_tree, variations_tree))


def _unpair_frame_leaves(
        paired_tree,       # G-shaped tree of (frame_data, variations_data) pairs
        n_frame:     int,   # frame-stack depth |G|
):  # -> (frame_tree, variations_tree), each G-shaped
    """Inverse of :py:func:`_pair_frame_leaves`: split a G-shaped tree of ``(frame_data,
    variations_data)`` pairs back into a ``(frame_tree, variations_tree)``.
    """
    if n_frame == 0:
        return paired_tree  # already a single (frame_data, variations_data) pair
    split = [_unpair_frame_leaves(p, n_frame - 1) for p in paired_tree]
    return tuple(s[0] for s in split), tuple(s[1] for s in split)


def tv_unstack_tangent_stack(
        frame,       # (UU, DD, LL, RR), each core stack = G
        variations,  # (VV, HH), each core stack = V + G
):  # -> array-like tree (shape V) of variations-data tuples (each stack = G)
    """Peel the tangent stack V off the variations, returning a V-shaped tree of variation-data.

    The base point is shared across V, so the frame cores are untouched (the caller pairs the same
    frame with every leaf). Inverse of :py:func:`tv_stack_tangent_stack`.
    """
    n_tangent, _ = _tangent_stack_split(frame, variations)
    return stacking.unstack(variations, axes=tuple(range(n_tangent)))


def tv_stack_tangent_stack(
        variations_tree,  # array-like tree (shape V) of variations-data tuples (each stack = G)
):  # -> variations-data tuple (stack = V + G)
    """Stack a V-shaped tree of variation-data over the tangent stack V (outermost).

    Inverse of :py:func:`tv_unstack_tangent_stack`.
    """
    return stacking.basic_ragged_stack(variations_tree)


def tv_unstack_frame_stack(
        frame,       # (UU, DD, LL, RR), each core stack = G
        variations,  # (VV, HH), each core stack = V + G
):  # -> array-like tree (shape G) of (frame_data, variations_data) pairs
    """Peel the frame stack G off both the frame and the variations, returning a G-shaped tree whose
    leaves are ``(frame_data, variations_data)`` pairs -- one single-base-point tangent per leaf.

    Each frame-data leaf has stack () (a single base point); each variations-data leaf has stack V.
    The frame stack is the *inner* part of the variation stack (V + G), so it is peeled from the
    interior axes of the variation cores. The frame and variation leaves are paired for you (a plain
    :py:func:`stacking.tree_zip` cannot do it -- it would recurse into the data-tuple leaves -- so a
    backend user would otherwise have to hand-roll a depth-aware zip). Inverse of
    :py:func:`tv_stack_frame_stack`.
    """
    n_tangent, n_frame = _tangent_stack_split(frame, variations)
    frame_tree = stacking.unstack(frame, axes=tuple(range(n_frame)))
    variations_tree = stacking.unstack(variations, axes=tuple(range(n_tangent, n_tangent + n_frame)))
    return _pair_frame_leaves(frame_tree, variations_tree, n_frame)


def tv_stack_frame_stack(
        paired_tree,  # array-like tree (shape G) of (frame_data, variations_data) pairs
):  # -> (
    #        frame-data,       # stack = G
    #        variations-data,  # stack = V + G
    #    )
    """Stack a G-shaped tree of ``(frame_data, variations_data)`` pairs over the frame stack G.

    The frame stack is placed *innermost* (the variation stack becomes V + G), matching the base-inner
    convention. Takes exactly the paired-tree layout that :py:func:`tv_unstack_frame_stack` produces (its
    inverse), so a backend user round-trips without splitting the pairs by hand.
    """
    n_frame = stacking.tree_depth(paired_tree) - 3   # |G|; a (frame_data, variations_data) leaf is 3 levels deep
    frame_tree, variations_tree = _unpair_frame_leaves(paired_tree, n_frame)
    frame = stacking.basic_ragged_stack(frame_tree)                      # G at leading -> stack = G
    n_tangent = len(stacking.get_first_leaf(variations_tree).shape) - 2  # |V| (a tucker variation leaf)
    variations = stacking.stack(variations_tree, axes=tuple(range(n_tangent, n_tangent + n_frame)))
    return frame, variations


def tv_gauge_residual(
        frame: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        variations: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations
            typ.Sequence[NDArray],  # tt_variations
        ],
) -> NDArray:  # shape = variation stack_shape (K+C); per stack element (scalar/0-d when unstacked)
    '''Max violation of the gauge conditions for a tangent vector, **per stack element**.

    The gauged tangent space requires each tucker variation orthogonal to its up-core, and each
    left-interior tt variation orthogonal to its left-core (see :py:func:`tv_orthogonal_gauge_projection`).
    Returns the max absolute gauge inner product reduced over the **non-stack** axes (shape = the variation
    stack ``K+C``); a caller thresholds it (``<= atol``).
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    tucker_variations, tt_variations = variations
    xnp, _, _ = get_backend(False, tree_contains_jax((frame, variations)))
    devs = []
    for U, V in zip(up_tucker_cores, tucker_variations):
        g = xnp.einsum('...ia,...ja->...ij', U, V)
        devs.append(xnp.max(xnp.abs(g), axis=(-2, -1)))   # keep stack, max over the gauge-gram axes
    for L, H in zip(left_tt_cores[:-1], tt_variations[:-1]):
        g = xnp.einsum('...abi,...abj->...ij', L, H)
        devs.append(xnp.max(xnp.abs(g), axis=(-2, -1)))
    return xnp.max(xnp.stack(devs), axis=0)   # max over the checks, keep stack_shape


def tv_retract(
        frame: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
        variations: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations
            typ.Sequence[NDArray],  # tt_variations
        ],
        shared_data: typ.Optional['sharing_module.T3SharedFrameData'] = None,  # tied retraction (SF-T3)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores (retracted T3, base-point ranks)
    typ.Tuple[NDArray, ...],  # tt_cores
]:
    '''Retract a frame-variations tangent vector onto the fixed-rank manifold.

    Forms the shifted doubled-rank embedding (base point + v) via :py:func:`tv_to_t3`
    (``include_shift=True``) and truncates it back to the **base point's own ranks** -- the Tucker
    ``up`` ranks and ``left`` TT ranks read off the frame cores -- with the implicit T3-SVD, yielding
    a point on the manifold of the base point's ranks.

    With ``shared_data`` (the frame's SF-T3 companion), the embedding is built TIED
    (:py:func:`tv_to_t3` with ``shared_data``) and truncated by the GROUPED T3-SVD, so the
    retracted point's group factors are exactly one shared array per group -- the shared
    manifold's retraction.

    The truncation is the implicit T3-SVD (Algorithm 10) of Alger et al. (2026), "Tucker Tensor
    Train Taylor Series" (arXiv:2603.21141).
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    shifted = tv_to_t3(frame, variations, include_shift=True, shared_data=shared_data)
    up_ranks = tuple(U.shape[-2] for U in up_tucker_cores)
    left_ranks = tuple(L.shape[-3] for L in left_tt_cores) + (left_tt_cores[-1].shape[-1],)
    sharing_labels = (None if shared_data is None
                      else sharing_module.groups_to_labels(shared_data.groups))
    retracted, _, _ = ragged_t3svd.t3svd(shifted, max_tucker_ranks=up_ranks, max_tt_ranks=left_ranks,
                                         sharing=sharing_labels)
    return retracted
