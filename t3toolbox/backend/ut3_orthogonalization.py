# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.orthogonalization as orth
from t3toolbox.backend.common import *

__all__ = [
    'down_orthogonalize_tucker_cores',
    'up_orthogonalize_tt_cores',
    'left_orthogonalize_tt_cores',
    'right_orthogonalize_tt_cores',
]

# A uniform-T3 .data tuple: (tucker_supercore, tt_supercore, (shape_mask, tucker_edge_mask, tt_edge_mask)).
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[NDArray, NDArray, NDArray]]

# Each function re-masks on entry; the SVD remainder R = ss.Vt has ss=0 in padded slots, so no garbage
# propagates. Ranks shrink to the structural minimum the SVD produces, and the masks are recomputed to
# match (minimal-for-free). See docs/uniform_port_plan.md (slice 2).


def _prefix_mask(ranks: NDArray, pad: int, xnp) -> NDArray:  # ranks -> bool prefix mask of width pad
    return xnp.arange(pad) < ranks[..., None]


def _left_orthogonalized_tt_ranks(tt_ranks, tucker_ranks, xnp):  # (d+1,)+stack ; L->R recurrence
    d = tucker_ranks.shape[0]
    new = [tt_ranks[0]]
    for i in range(d - 1):
        new.append(xnp.minimum(new[i] * tucker_ranks[i], tt_ranks[i + 1]))
    new.append(tt_ranks[d])
    return xnp.stack(new)


def _right_orthogonalized_tt_ranks(tt_ranks, tucker_ranks, xnp):  # (d+1,)+stack ; R->L recurrence
    d = tucker_ranks.shape[0]
    new = [None] * (d + 1)
    new[d] = tt_ranks[d]
    for i in range(d - 1, 0, -1):
        new[i] = xnp.minimum(tucker_ranks[i] * new[i + 1], tt_ranks[i])
    new[0] = tt_ranks[0]
    return xnp.stack(new)


def down_orthogonalize_tucker_cores(data: UT3Data) -> UT3Data:
    """Orthogonalize the Tucker cores (rows orthonormal over the mode index), pushing the remainder up
    into the TT cores. Core-local -> one batched SVD over ``(d,)+stack``. Tucker rank -> min(shape, rank)."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    mtk, mtt = ut3_masking.apply_masks_to_cores(data)
    sm, tkm, ttm = data[2]

    U_o_x, ss, WT_x_i = xnp.linalg.svd(mtk.swapaxes(-2, -1), full_matrices=False)
    R_x_i = xnp.einsum('...x,...xi->...xi', ss, WT_x_i)
    new_tt = xnp.einsum('...aib,...xi->...axb', mtt, R_x_i)
    new_tk = U_o_x.swapaxes(-1, -2)

    stack = mtk.shape[1:-2]
    shape_arr = sm.sum(axis=-1).reshape((mtk.shape[0],) + (1,) * len(stack))
    new_tucker_ranks = xnp.minimum(tkm.sum(axis=-1), shape_arr)
    new_tkm = _prefix_mask(new_tucker_ranks, new_tk.shape[-2], xnp)
    return new_tk, new_tt, (sm, new_tkm, ttm)


def up_orthogonalize_tt_cores(data: UT3Data) -> UT3Data:
    """Up-orthogonalize the TT cores (mode index orthonormal over the bonds), pushing the remainder down
    into the Tucker cores. Core-local -> one batched SVD. Tucker rank -> min(rank, rL*rR)."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    mtk, mtt = ut3_masking.apply_masks_to_cores(data)
    sm, tkm, ttm = data[2]

    d = mtt.shape[0]
    stack = mtt.shape[1:-3]
    rL, n, rR = mtt.shape[-3:]
    H_ab_i = mtt.swapaxes(-1, -2).reshape((d,) + stack + (rL * rR, n))
    O_ab_x, ss, WT_x_i = xnp.linalg.svd(H_ab_i, full_matrices=False)
    x = ss.shape[-1]
    new_tt = O_ab_x.reshape((d,) + stack + (rL, rR, x)).swapaxes(-1, -2)   # (d,)+stack+(rL,x,rR)
    C_x_i = xnp.einsum('...x,...xi->...xi', ss, WT_x_i)
    new_tk = xnp.einsum('...xi,...io->...xo', C_x_i, mtk)                  # (d,)+stack+(x,N)

    tt_ranks = ttm.sum(axis=-1)  # (d+1,)+stack
    new_tucker_ranks = xnp.minimum(tkm.sum(axis=-1), tt_ranks[:-1] * tt_ranks[1:])
    new_tkm = _prefix_mask(new_tucker_ranks, new_tt.shape[-2], xnp)
    return new_tk, new_tt, (sm, new_tkm, ttm)


def left_orthogonalize_tt_cores(data: UT3Data) -> UT3Data:
    """Left-orthogonalize the TT cores (shared polymorphic sweep). Bond ranks: L->R recurrence."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    _, mtt = ut3_masking.apply_masks_to_cores(data)
    sm, tkm, ttm = data[2]
    new_tt = orth.left_orthogonalize_tt_cores(mtt)
    new_tt_ranks = _left_orthogonalized_tt_ranks(ttm.sum(axis=-1), tkm.sum(axis=-1), xnp)
    new_ttm = _prefix_mask(new_tt_ranks, new_tt.shape[-1], xnp)
    return data[0], new_tt, (sm, tkm, new_ttm)


def right_orthogonalize_tt_cores(data: UT3Data) -> UT3Data:
    """Right-orthogonalize the TT cores (shared polymorphic sweep). Bond ranks: R->L recurrence."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    _, mtt = ut3_masking.apply_masks_to_cores(data)
    sm, tkm, ttm = data[2]
    new_tt = orth.right_orthogonalize_tt_cores(mtt)
    new_tt_ranks = _right_orthogonalized_tt_ranks(ttm.sum(axis=-1), tkm.sum(axis=-1), xnp)
    new_ttm = _prefix_mask(new_tt_ranks, new_tt.shape[-1], xnp)
    return data[0], new_tt, (sm, tkm, new_ttm)
