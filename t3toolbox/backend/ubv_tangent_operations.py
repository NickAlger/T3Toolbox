# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Stateless tangent-stack reshuffles for the uniform tangent layer (UT3Tangent), uniform-fix 3b-1b.

The uniform mirror of the ragged ``tangent_operations`` stack/unstack helpers, for the ``.data`` layout
``(*supercores, shape, masks)`` with a leading mode index ``d`` and the stack at axes ``1 ..``. A uniform
tangent's variations carry the full ``K + C`` stack (tangent stack ``K`` outermost, base stack ``C``
inner); the basis carries only ``C``. These functions split that stack into a Python tree of per-element
objects and recombine -- a tree<->array conversion, NOT an axis permutation of the supercores. They are
the backend the UT3Tangent ``unstack_tangents`` / ``unstack_basis`` / ``stack_tangents`` / ``stack_basis``
/ ``sum_tangents`` methods delegate to.

Varying ranks across the base ``C`` stack are first-class here (the rank-sweep use case): the per-element
masks may differ, and ``stack_base_stack`` stacks frames of different ranks into one batch (the masks just
ride along). Uniform rank is required only across ``K`` (one shared base = one tangent space), which
``stack_tangent_stack`` gets for free (the leaves share a frame). See
``docs/uniform_ranks_and_varieties.md``.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ubv_operations as ubv_operations
import t3toolbox.backend.ut3_svd as ut3_svd
from t3toolbox.backend.common import *

__all__ = [
    'tangent_to_ut3',
    'retract',
    'unstack_tangent_stack',
    'stack_tangent_stack',
    'unstack_base_stack',
    'stack_base_stack',
    'sum_tangent_stack',
]


def _tangent_stack_split(
        basis_data,       # (up, down, left, right, shape, masks), each supercore stack = C
        variations_data,  # (tkv, ttv, shape, masks),             each supercore stack = K + C
) -> typ.Tuple[int, int]:  # (|K|, |C|)
    """Recover the tangent / base stack split (|K|, |C|) from a (basis, variations) ``.data`` pair.

    The frame supercores carry only the base stack ``C`` (up: ``(d,)+C+(nU, N)``); the variation supercores
    carry the full ``K + C`` (tucker variation: ``(d,)+K+C+(nD, N)``). So ``|C|`` comes from a frame core
    and ``|K|`` is the remainder.
    """
    n_base = basis_data[0].ndim - 3       # |C|   (up supercore: (d,) + C + (nU, N))
    n_full = variations_data[0].ndim - 3  # |K+C| (tucker variation: (d,) + K + C + (nD, N))
    return n_full - n_base, n_base


def _pair_leaves(basis_tree, variations_tree, n_base):  # mirror tangent_operations._pair_base_leaves
    """Pair a basis-data tree and a variations-data tree (same ``C``-shaped outer structure, ``n_base``
    levels deep) leaf-by-leaf into one tree of ``(basis_data, variations_data)`` pairs. NOT a
    :py:func:`stacking.tree_zip`: the data-tuple leaves are themselves sequences (and carry the int-tuple
    ``shape``), so a generic zip would recurse into them -- we stop at the known base depth ``n_base``."""
    if n_base == 0:
        return (basis_tree, variations_tree)  # both are single .data tuples -> one pair
    return tuple(_pair_leaves(b, v, n_base - 1) for b, v in zip(basis_tree, variations_tree))


def _unpair_leaves(paired_tree, n_base):  # inverse of _pair_leaves
    """Split a ``C``-shaped tree of ``(basis_data, variations_data)`` pairs back into
    ``(basis_tree, variations_tree)``."""
    if n_base == 0:
        return paired_tree  # already a single (basis_data, variations_data) pair
    split = [_unpair_leaves(p, n_base - 1) for p in paired_tree]
    return tuple(s[0] for s in split), tuple(s[1] for s in split)


def _depth_to_pair(paired_tree) -> int:  # |C|: nesting levels above a (basis_data, variations_data) leaf
    """Count the nesting depth of a ``C``-shaped tree of ``(basis_data, variations_data)`` pairs without
    recursing into the data tuples (whose int-tuple ``shape`` would fool :py:func:`stacking.tree_depth`).
    A leaf pair is reached when ``node[0][0]`` -- the first frame supercore of ``basis_data`` -- is an
    ndarray; until then ``node[0]`` is a subtree."""
    depth, node = 0, paired_tree
    while not is_ndarray(node[0][0]):
        node, depth = node[0], depth + 1
    return depth


def unstack_tangent_stack(
        basis_data,       # frame .data,      supercore stack = C
        variations_data,  # variations .data, supercore stack = K + C
):  # -> array-like tree (shape K) of variations .data tuples (each stack = C)
    """Peel the tangent stack ``K`` off the variations, returning a ``K``-shaped tree of variation-``.data``.

    The base is shared across ``K``, so the frame is untouched (the caller pairs the same base with every
    leaf). Inverse of :py:func:`stack_tangent_stack`."""
    n_tangent, _ = _tangent_stack_split(basis_data, variations_data)
    return ubv_operations.ubv_unstack_axes(variations_data, 2, range(1, 1 + n_tangent))


def stack_tangent_stack(
        variations_tree,  # array-like tree (shape K) of variations .data tuples (each stack = C)
):  # -> variations .data tuple (stack = K + C)
    """Stack a ``K``-shaped tree of variation-``.data`` over the tangent stack ``K`` (outermost; ``C`` stays
    inner). Inverse of :py:func:`unstack_tangent_stack`. The leaves share one frame, so ranks are uniform
    across ``K`` (the masks, constant along ``K``, just replicate)."""
    return ubv_operations.ubv_stack(variations_tree, 2)


def unstack_base_stack(
        basis_data,       # frame .data,      supercore stack = C
        variations_data,  # variations .data, supercore stack = K + C
):  # -> array-like tree (shape C) of (basis_data, variations_data) pairs
    """Peel the base stack ``C`` off both the frame and the variations, returning a ``C``-shaped tree whose
    leaves are ``(basis_data, variations_data)`` pairs -- one single-base-point tangent per leaf.

    Each frame-``.data`` leaf has stack ``()`` (a single base); each variations-``.data`` leaf has stack
    ``K``. The base stack is the *inner* part of the ``K + C`` variation stack, so it is peeled from the
    interior axes ``1+|K| .. 1+|K|+|C|`` of the variation supercores; the frame's whole stack is ``C``. The
    leaves are paired for you (a plain :py:func:`stacking.tree_zip` would recurse into the data tuples).
    Inverse of :py:func:`stack_base_stack`."""
    n_tangent, n_base = _tangent_stack_split(basis_data, variations_data)
    basis_tree = ubv_operations.ubv_unstack(basis_data, 4)  # frame stack is all C
    variations_tree = ubv_operations.ubv_unstack_axes(variations_data, 2,
                                                      range(1 + n_tangent, 1 + n_tangent + n_base))
    return _pair_leaves(basis_tree, variations_tree, n_base)


def stack_base_stack(
        paired_tree,  # array-like tree (shape C) of (basis_data, variations_data) pairs
):  # -> (basis_data [stack C], variations_data [stack K + C])
    """Stack a ``C``-shaped tree of ``(basis_data, variations_data)`` pairs over the base stack ``C``.

    The base stack is placed *innermost* (the variation stack becomes ``K + C``), matching the base-inner
    convention. Frames of DIFFERENT ranks stack into one batch (varying-``C`` -- the per-element masks just
    ride along); the shared requirement is only matching padded dims and tangent stack ``K``. Takes exactly
    the layout :py:func:`unstack_base_stack` produces (its inverse)."""
    n_base = _depth_to_pair(paired_tree)
    basis_tree, variations_tree = _unpair_leaves(paired_tree, n_base)
    basis_data = ubv_operations.ubv_stack(basis_tree, 4)                 # C at axes 1.. (frame stack = C)
    n_tangent = ubv_operations._first_data_leaf(variations_tree)[0].ndim - 3  # |K| of a variations leaf
    variations_data = ubv_operations.ubv_stack_axes(variations_tree, 2, axes_start=1 + n_tangent)  # C after K
    return basis_data, variations_data


def sum_tangent_stack(
        variations_data,         # variations .data, supercore stack = K + C
        n_tangent:   int,        # |K|
        axis:        typ.Optional[int] = None,  # 0-based index WITHIN K (None = the whole tangent stack)
):  # -> variations .data with the summed K axes removed (stack = C, or K-with-one-axis-removed)
    """Sum the variations over the tangent stack ``K`` (a batch of tangents at one base -> their sum;
    corewise == the tangent sum, by linearity). The base stack ``C`` is preserved.

    The supercores sum via ``xnp``; the masks **OR** over the same axes (host ``np``). Because a ``K`` stack
    shares one base, its masks are constant along ``K``, so the OR is a no-op (the summed tangent carries
    the base's gauge masks) -- but it is the correct reduction in general. ``axis`` indexes within ``K``."""
    tkv, ttv, shape, masks = variations_data
    xnp, _, _ = get_backend(True, tree_contains_jax((tkv, ttv)))

    k_axes = tuple(range(1, 1 + n_tangent)) if axis is None else (1 + axis,)
    new_tkv = xnp.sum(tkv, axis=k_axes)
    new_ttv = xnp.sum(ttv, axis=k_axes)
    new_masks = tuple(np.any(m, axis=k_axes) for m in masks)   # host np: OR the real slots over K
    return new_tkv, new_ttv, shape, new_masks


def tangent_to_ut3(
        basis_data,       # UT3Basis .data:      (up, down, left, right, shape, (4 masks)),  supercore stack = C
        variations_data,  # UT3Variations .data: (tkv, ttv, shape, (4 masks)),               supercore stack = K + C
        include_shift: bool = False,  # False: tangent vector v. True: base point + v.
):  # -> doubled-rank UniformTuckerTensorTrain .data: (tucker_supercore, tt_supercore, shape, (tucker_mask, tt_mask))
    """Doubled-rank uniform Tucker tensor train representing a uniform basis-variations tangent vector.

    The uniform mirror of :py:func:`tangent_operations.tangent_to_t3` (equations (50)-(53) / Figure 20,
    Appendix A.3.1 of Alger et al. 2026). The Tucker supercore becomes ``[U ; V]`` (concat along the
    Tucker-rank axis); the TT supercore is the block-bidiagonal embedding, uniform-padded to bonds
    ``rL+rR`` for every core with the **base-inner ``[R, L]`` bond order** (mirroring the ragged build).
    The doubled rank masks are concatenations of the existing masks (the **#1 trap**: the appended boundary
    slots are FULL ``ones`` -- the supercore is zero there, so to_dense's mask-then-contract is unaffected):
    ``tucker_mask = concat([up, down])``; ``tt_mask = concat([right_ext, left_ext])`` with
    ``left_ext = [var_left, ones]`` and ``right_ext = [ones, var_right]``.

    Stack-aware: the variation supercores carry ``K + C``; the base supercores (stack ``C``) are broadcast
    up to ``K + C`` (mirror ragged ``bcast``), and the masks (host numpy, carrying ``K + C`` already from
    the variations) are concatenated on the host. With ``include_shift=True`` the base point is folded into
    the last core (``base point + v``)."""
    up_sc, down_sc, left_sc, right_sc, shape, _base_masks = basis_data
    tkv, ttv, _shape_v, var_masks = variations_data
    var_up_mask, var_down_mask, var_left_mask, var_right_mask = var_masks

    use_jax = tree_contains_jax((up_sc, down_sc, left_sc, right_sc, tkv, ttv))
    xnp, _, _ = get_backend(True, use_jax)

    d  = up_sc.shape[0]
    nU = up_sc.shape[-2]; N = up_sc.shape[-1]; nD = down_sc.shape[-2]
    rL = left_sc.shape[-1]; rR = right_sc.shape[-1]
    base_stack = up_sc.shape[1:-2]                 # C
    ss = tkv.shape[1:-2]                           # K + C (the variation/output stack)
    n_K = len(ss) - len(base_stack)

    def bcast(sc):  # base supercore (d,)+C+(core) -> (d,)+K+C+(core): insert |K| size-1 axes after d
        return xnp.broadcast_to(sc.reshape(sc.shape[:1] + (1,) * n_K + sc.shape[1:]),
                                sc.shape[:1] + tuple(ss[:n_K]) + sc.shape[1:])

    U = bcast(up_sc); O = bcast(down_sc); L = bcast(left_sc); R = bcast(right_sc)
    Z = lambda nd, a, b, c: xnp.zeros((nd,) + ss + (a, b, c))   # nd cores of a (a, b, c) zero block

    # ---- Tucker supercore: [U ; V] along the Tucker-rank axis (-2). Fully vectorized over d. ----
    tucker_supercore = xnp.concatenate([U, tkv], axis=-2)       # (d,)+ss+(nU+nD, N)

    # ---- TT supercore: first / mid / last cores, [R, L] bond order, padded to (rR+rL, nU+nD, rR+rL). ----
    def first_core():     # left bond is the boundary (the L-block); pad the R-block (rR rows) with zeros
        Gtop = xnp.concatenate([ttv[:1], L[:1]], axis=-1)                       # (rL, nU, rR+rL)
        Gbot = xnp.concatenate([O[:1], Z(1, rL, nD, rL)], axis=-1)             # (rL, nD, rR+rL)
        G = xnp.concatenate([Gtop, Gbot], axis=-2)                            # (rL, nU+nD, rR+rL)
        return xnp.concatenate([Z(1, rR, nU + nD, rR + rL), G], axis=-3)       # (rR+rL, nU+nD, rR+rL)

    def mid_cores():      # the full block-bidiagonal core (ragged-middle), vectorized over [1:-1]
        nd = d - 2
        Gtop = xnp.concatenate([
            xnp.concatenate([R[1:-1], Z(nd, rR, nU, rL)], axis=-1),            # [R, Z001] -> (rR, nU, rR+rL)
            xnp.concatenate([ttv[1:-1], L[1:-1]], axis=-1),                    # [dU, L]   -> (rL, nU, rR+rL)
        ], axis=-3)                                                            # (rR+rL, nU, rR+rL)
        Gbot = xnp.concatenate([
            xnp.concatenate([Z(nd, rR, nD, rR), Z(nd, rR, nD, rL)], axis=-1),  # (rR, nD, rR+rL)
            xnp.concatenate([O[1:-1], Z(nd, rL, nD, rL)], axis=-1),            # [O, Z111] -> (rL, nD, rR+rL)
        ], axis=-3)                                                            # (rR+rL, nD, rR+rL)
        return xnp.concatenate([Gtop, Gbot], axis=-2)                         # (rR+rL, nU+nD, rR+rL)

    def last_core():      # right bond is the boundary (the R-block); pad the L-block (rL cols) with zeros
        dG = ttv[-1:]
        if include_shift:  # fold the base point's last core in: add its rank-1 right boundary (col 0) to dU
            Lf_boundary = xnp.concatenate(
                [L[-1:][..., :1], Z(1, rL, nU, rR - 1)], axis=-1)              # (rL, nU, rR), boundary in col 0
            dG = dG + Lf_boundary
        Gtop = xnp.concatenate([R[-1:], dG], axis=-3)                          # [R, dU] -> (rR+rL, nU, rR)
        Gbot = xnp.concatenate([Z(1, rR, nD, rR), O[-1:]], axis=-3)            # (rR+rL, nD, rR)
        G = xnp.concatenate([Gtop, Gbot], axis=-2)                            # (rR+rL, nU+nD, rR)
        return xnp.concatenate([G, Z(1, rR + rL, nU + nD, rL)], axis=-1)       # (rR+rL, nU+nD, rR+rL)

    if d < 2:
        # d==1 is degenerate: the single core's two boundaries don't double into a square (rR+rL) bond,
        # so it does not fit the square-bond uniform tt_supercore (r, n, r) when rL != rR. A single-mode T3
        # is a plain Tucker decomposition; support it separately if a real use case appears.
        raise NotImplementedError(
            'tangent_to_ut3 requires d >= 2 (the d == 1 single-mode case does not fit the square-bond '
            'uniform doubled-rank format).')
    blocks = [first_core()] + ([mid_cores()] if d > 2 else []) + [last_core()]
    tt_supercore = xnp.concatenate(blocks, axis=0)                             # (d,)+ss+(rR+rL, nU+nD, rR+rL)

    # ---- Doubled rank masks (HOST numpy; carry K+C from the variation masks). The order respects the
    # block structure (eqs 50-53): Tucker axis [U | dU] = [up | down]; the TT bond is [Q | P] = [R | L], so
    # the right-chain mask sits in the Q block and the left-chain mask in the P block -- the unique
    # concatenation where each diagonal mask multiplies the core block it belongs to. At the two boundary
    # bonds the global "1" sits in the CONTENT block (P at bond 0, Q at bond d); the opposite (free) block
    # has no core, hence rank 0 -- appended as zeros (not phantom ones), so the doubled ranks stay honest. ----
    tucker_mask = np.concatenate([var_up_mask, var_down_mask], axis=-1)        # (d,)+ss+(nU+nD,)  [U | dU]
    left_ext  = np.concatenate([var_left_mask,  np.zeros((1,) + ss + (rL,), bool)], axis=0)  # (d+1,)+ss+(rL,): P-part
    right_ext = np.concatenate([np.zeros((1,) + ss + (rR,), bool), var_right_mask], axis=0)  # (d+1,)+ss+(rR,): Q-part
    tt_mask = np.concatenate([right_ext, left_ext], axis=-1)                   # (d+1,)+ss+(rR+rL,)  [Q | P] order

    return tucker_supercore, tt_supercore, shape, (tucker_mask, tt_mask)


def retract(
        basis_data,       # UT3Basis .data:      supercore stack = C
        variations_data,  # UT3Variations .data: supercore stack = K + C
):  # -> retracted UniformTuckerTensorTrain .data (at the BASE point's ranks; stack = K + C)
    """Retract a uniform basis-variations tangent vector onto the fixed-rank manifold.

    Forms the shifted doubled-rank embedding ``base point + v`` (:py:func:`tangent_to_ut3` with
    ``include_shift=True``) and truncates it back to the **base point's own ranks** -- the Tucker ``up``
    ranks and ``left`` TT ranks read off the base masks -- via the mask-truncated uniform T3-SVD. The output
    is a UT3 at the base padded dims (``ut3svd`` truncates by max rank to a fixed shape, so no extra slice
    is needed), one retracted point per stack element. The uniform mirror of
    :py:func:`tangent_operations.retract` (the implicit T3-SVD / Algorithm 10, Alger et al. 2026).

    **Varying ranks across ``C``** work for free: the per-``C`` base ranks are the per-element truncation
    targets. **The ``K`` (tangent) stack:** the base ranks have stack ``C`` while the shifted UT3 has stack
    ``K + C``, so the base ranks are broadcast over ``K`` (the ``K`` tangents share the base, hence the same
    truncation targets)."""
    doubled = tangent_to_ut3(basis_data, variations_data, include_shift=True)   # .data, stack K + C
    ss = doubled[0].shape[1:-2]                       # K + C (the shifted UT3 stack)
    C = basis_data[0].shape[1:-2]                     # C (the base stack)
    n_K = len(ss) - len(C)

    up_mask, _down_mask, basis_left_mask, _basis_right_mask = basis_data[5]
    up_ranks   = up_mask.sum(axis=-1)                 # (d,)   + C, HOST int
    left_ranks = basis_left_mask.sum(axis=-1)         # (d+1,) + C, HOST int

    def bcast_over_K(ranks):  # (L,)+C -> (L,)+K+C: the K tangents share the base point's truncation targets
        return np.broadcast_to(ranks.reshape(ranks.shape[:1] + (1,) * n_K + ranks.shape[1:]),
                               ranks.shape[:1] + ss)

    new_data, _ss_tucker, _ss_tt = ut3_svd.ut3svd(
        doubled, max_tucker_ranks=bcast_over_K(up_ranks), max_tt_ranks=bcast_over_K(left_ranks))
    return new_data
