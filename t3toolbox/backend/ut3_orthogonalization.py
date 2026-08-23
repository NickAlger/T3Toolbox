# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Orthogonalization for uniform supercores -- SVD-based so the masks stay a deterministic prefix.

Supercore-level sweeps, their core-level mirrors, and ``ut3_orthogonality_residual``. Why SVD
rather than QR is load-bearing: ``docs/contributor/uniform_svd_prefix_orthogonalization.md``.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.tt_orthogonalization as orth
from t3toolbox.backend.common import *

__all__ = [
    'down_orthogonalize_tucker_supercores',
    'up_orthogonalize_tt_supercores',
    'ut3_down_orthogonalize_tucker_cores',
    'ut3_up_orthogonalize_tt_cores',
    'ut3_left_orthogonalize_tt_cores',
    'ut3_right_orthogonalize_tt_cores',
    'ut3_orthogonality_residual',
]

# A uniform-T3 .data tuple: (tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask)).
# `shape` is a static int tuple; the two rank masks are HOST bool, static structure (numpy, never
# traced); the supercores are xnp data.
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]]


def ut3_orthogonality_residual(
        data: UT3Data,
        side: str,  # 'left' or 'right'
) -> NDArray:  # shape = stack_shape; max abs deviation of the masked cores from orthonormality, per element
    '''Uniform analog of :py:func:`t3_orthogonality_residual`: non-enforcing check of left/right-orthogonal
    form (Tucker supercores down-orthogonal AND TT supercores left/right-orthogonal), **per stack element**.

    Compares each masked supercore's Gram against ``diag(mask)`` (the masked rows/cols are zero, so the
    identity is restricted to the real block). The boundary TT core (last for left, first for right) is the
    center remainder and is not checked. Reduced over the **non-stack** axes (the leading mode index ``d``
    and the two gram axes), so the result has shape ``stack_shape``.
    '''
    side = side.lower()
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    tucker_sc, tt_sc = ut3_masking.ut3_apply_masks(data)  # guards: masks must be host (not traced)
    tucker_mask, tt_mask = data[3]                             # HOST bool rank masks (constant operands below)
    n, r = tucker_sc.shape[-2], tt_sc.shape[-1]

    Mt = xnp.einsum('...io,...jo->...ij', tucker_sc, tucker_sc)          # (d,)+stack+(n,n)
    resid = xnp.max(xnp.abs(Mt - xnp.eye(n) * tucker_mask[..., None, :]), axis=(0, -2, -1))  # -> stack_shape
    if tt_sc.shape[0] == 1:
        # d = 1: no interior bond, the single TT core is the center remainder -- only the Tucker term applies
        # (a max over the empty interior would raise). Mirrors the ragged twin's loop over an empty tuple.
        return resid

    interior = tt_mask[1:-1]                                              # interior bonds 1..d-1
    if side == 'left':                                                   # modes 0..d-2, right bonds
        M = xnp.einsum('...aib,...aic->...bc', tt_sc[:-1], tt_sc[:-1])
    elif side == 'right':                                               # modes 1..d-1, left bonds
        M = xnp.einsum('...aib,...cib->...ac', tt_sc[1:], tt_sc[1:])
    else:
        raise ValueError("side must be 'left' or 'right'; got %r" % (side,))
    resid_tt = xnp.max(xnp.abs(M - xnp.eye(r) * interior[..., None, :]), axis=(0, -2, -1))   # -> stack_shape
    return xnp.maximum(resid, resid_tt)

# Each function re-masks on entry; the SVD remainder R = ss.Vt has ss=0 in padded slots, so no garbage
# propagates. Ranks shrink to the structural minimum the SVD produces, and the masks are recomputed to
# match (minimal-for-free). See dev/archive/uniform_port_plan.md (slice 2).
#
# All rank recurrences / mask builders below use np (host), NOT xnp: masks are static structure (a jax
# mask is a tracer under jit -> leaks into aux_data). The mask `np.*` is intentional; see
# docs/contributor/uniform_pytree_composition.md. Only the supercore SVDs go through xnp.


def _left_orthogonalized_tt_ranks(tt_ranks, tucker_ranks):  # HOST int (d+1,)+stack ; L->R recurrence
    d = tucker_ranks.shape[0]
    new = [tt_ranks[0]]
    for i in range(d - 1):
        new.append(np.minimum(new[i] * tucker_ranks[i], tt_ranks[i + 1]))
    new.append(tt_ranks[d])
    return np.stack(new)


def _right_orthogonalized_tt_ranks(tt_ranks, tucker_ranks):  # HOST int (d+1,)+stack ; R->L recurrence
    d = tucker_ranks.shape[0]
    new = [None] * (d + 1)
    new[d] = tt_ranks[d]
    for i in range(d - 1, 0, -1):
        new[i] = np.minimum(tucker_ranks[i] * new[i + 1], tt_ranks[i])
    new[0] = tt_ranks[0]
    return np.stack(new)


def down_orthogonalize_tucker_supercores(
        tucker_supercore: NDArray,  # shape=(d,)+stack+(n,N) (assumed masked)
        tt_supercore:     NDArray,  # shape=(d,)+stack+(r,n,r)
) -> typ.Tuple[NDArray, NDArray]:   # (new_tucker, new_tt) with n-axis -> min(N,n)
    """Bare batched Tucker SVD on supercores (assumes masked input): rows orthonormal over the mode
    index, remainder pushed into the TT cores. The SVD core of :py:func:`ut3_down_orthogonalize_tucker_cores`;
    also reused by the T3-SVD sweep (which manages its own truncation masks)."""
    use_jax = tree_contains_jax((tucker_supercore, tt_supercore))
    xnp, _, _ = get_backend(True, use_jax)
    U_o_x, ss, WT_x_i = xnp.linalg.svd(tucker_supercore.swapaxes(-2, -1), full_matrices=False)
    R_x_i = xnp.einsum('...x,...xi->...xi', ss, WT_x_i)
    new_tt = xnp.einsum('...aib,...xi->...axb', tt_supercore, R_x_i)
    new_tk = U_o_x.swapaxes(-1, -2)
    return new_tk, new_tt


def ut3_down_orthogonalize_tucker_cores(data: UT3Data) -> UT3Data:
    """Orthogonalize the Tucker cores (rows orthonormal over the mode index), pushing the remainder up
    into the TT cores. Core-local -> one batched SVD over ``(d,)+stack``. Tucker rank -> min(shape, rank)."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    mtk, mtt = ut3_masking.ut3_apply_masks(data)
    shape = data[2]                                 # static int tuple
    tkm, ttm = data[3]                              # HOST bool rank masks

    new_tk, new_tt = down_orthogonalize_tucker_supercores(mtk, mtt)

    # masks/ranks on the host (np), supercores via xnp -- see the module note above.
    stack = mtk.shape[1:-2]
    shape_arr = np.asarray(shape).reshape((mtk.shape[0],) + (1,) * len(stack))
    new_tucker_ranks = np.minimum(tkm.sum(axis=-1), shape_arr)
    new_tkm = prefix_mask(new_tucker_ranks, new_tk.shape[-2])
    return new_tk, new_tt, shape, (new_tkm, ttm)


def up_orthogonalize_tt_supercores(
        tucker_supercore: NDArray,  # (d,)+stack+(n,N) (assumed masked)
        tt_supercore:     NDArray,  # (d,)+stack+(rL, n, rR)
) -> typ.Tuple[NDArray, NDArray]:   # (new_tucker = variations V, new_tt = down-orthogonal O)
    """Bare batched TT-up SVD on supercores (assumes masked input): the TT mode index becomes orthonormal
    over the bonds, the remainder pushed down into the Tucker core. The SVD core of
    :py:func:`ut3_up_orthogonalize_tt_cores`; also reused by the orthogonal-representation sweep
    (``fv_conversions.t3_orthogonal_representations`` via ``ufv_conversions``), which manages its own ranks/masks afterward."""
    use_jax = tree_contains_jax((tucker_supercore, tt_supercore))
    xnp, _, _ = get_backend(True, use_jax)
    d = tt_supercore.shape[0]
    stack = tt_supercore.shape[1:-3]
    rL, n, rR = tt_supercore.shape[-3:]
    H_ab_i = tt_supercore.swapaxes(-1, -2).reshape((d,) + stack + (rL * rR, n))
    O_ab_x, ss, WT_x_i = xnp.linalg.svd(H_ab_i, full_matrices=False)
    x = ss.shape[-1]
    new_tt = O_ab_x.reshape((d,) + stack + (rL, rR, x)).swapaxes(-1, -2)   # (d,)+stack+(rL,x,rR)
    C_x_i = xnp.einsum('...x,...xi->...xi', ss, WT_x_i)
    new_tk = xnp.einsum('...xi,...io->...xo', C_x_i, tucker_supercore)     # (d,)+stack+(x,N)
    return new_tk, new_tt


def ut3_up_orthogonalize_tt_cores(data: UT3Data) -> UT3Data:
    """Up-orthogonalize the TT cores (mode index orthonormal over the bonds), pushing the remainder down
    into the Tucker cores. Core-local -> one batched SVD. Tucker rank -> min(rank, rL*rR)."""
    mtk, mtt = ut3_masking.ut3_apply_masks(data)
    shape = data[2]                                 # static int tuple
    tkm, ttm = data[3]                              # HOST bool rank masks

    new_tk, new_tt = up_orthogonalize_tt_supercores(mtk, mtt)

    tt_ranks = ttm.sum(axis=-1)  # HOST int (d+1,)+stack
    new_tucker_ranks = np.minimum(tkm.sum(axis=-1), tt_ranks[:-1] * tt_ranks[1:])
    new_tkm = prefix_mask(new_tucker_ranks, new_tt.shape[-2])
    return new_tk, new_tt, shape, (new_tkm, ttm)


def ut3_left_orthogonalize_tt_cores(data: UT3Data) -> UT3Data:
    """Left-orthogonalize the TT cores (shared polymorphic sweep). Bond ranks: L->R recurrence."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    _, mtt = ut3_masking.ut3_apply_masks(data)
    tkm, ttm = data[3]                              # HOST bool rank masks
    new_tt = orth.tt_left_orthogonalize(mtt)
    new_tt_ranks = _left_orthogonalized_tt_ranks(ttm.sum(axis=-1), tkm.sum(axis=-1))
    new_ttm = prefix_mask(new_tt_ranks, new_tt.shape[-1])
    return data[0], new_tt, data[2], (tkm, new_ttm)


def ut3_right_orthogonalize_tt_cores(data: UT3Data) -> UT3Data:
    """Right-orthogonalize the TT cores (shared polymorphic sweep). Bond ranks: R->L recurrence."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    _, mtt = ut3_masking.ut3_apply_masks(data)
    tkm, ttm = data[3]                              # HOST bool rank masks
    new_tt = orth.tt_right_orthogonalize(mtt)
    new_tt_ranks = _right_orthogonalized_tt_ranks(ttm.sum(axis=-1), tkm.sum(axis=-1))
    new_ttm = prefix_mask(new_tt_ranks, new_tt.shape[-1])
    return data[0], new_tt, data[2], (tkm, new_ttm)
