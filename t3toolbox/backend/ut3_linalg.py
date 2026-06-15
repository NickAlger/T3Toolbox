# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ
import math

import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ut3_operations as ut3_operations
import t3toolbox.backend.t3_operations as t3_operations
from t3toolbox.backend.common import *

__all__ = [
    'ut3_add',
    'ut3_sum_stack',
    'ut3_inner_product',
    'ut3_norm_orthogonalized',
]

# A uniform-T3 .data tuple: (tucker_supercore, tt_supercore, (shape_mask, tucker_edge_mask, tt_edge_mask)).
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[NDArray, NDArray, NDArray]]


def ut3_add(x: UT3Data, y: UT3Data) -> UT3Data:  # z = x + y (ranks add; NOT squashed)
    """Add two uniform Tucker tensor trains (direct sum): concatenate Tucker supercores along the rank
    axis, block-diagonalize the TT supercores, and concatenate the masks (``shape`` via OR). Vectorized
    over ``(d,)+stack`` (the ``xnp.block`` acts on the last 3 axes). ``x``,``y`` need not share padded
    ``n``/``r``; only ``N``, ``d``, ``stack_shape`` must match (the frontend enforces that).
    """
    use_jax = tree_contains_jax((x[:2], y[:2]))
    xnp, _, _ = get_backend(True, use_jax)

    tk_x, tt_x, (sm_x, tkm_x, ttm_x) = x
    tk_y, tt_y, (sm_y, tkm_y, ttm_y) = y

    d = tk_x.shape[0]
    stack = tk_x.shape[1:-2]
    rx, nx = tt_x.shape[-1], tt_x.shape[-2]
    ry, ny = tt_y.shape[-1], tt_y.shape[-2]

    z_tk = xnp.concatenate([tk_x, tk_y], axis=-2)

    Z = lambda a, b, c: xnp.zeros((d,) + stack + (a, b, c))
    z_tt = xnp.block([
        [[tt_x,            Z(rx, nx, ry)], [Z(rx, ny, rx), Z(rx, ny, ry)]],
        [[Z(ry, nx, rx),   Z(ry, nx, ry)], [Z(ry, ny, rx), tt_y]],
    ])

    z_sm = xnp.logical_or(sm_x, sm_y)                       # same ambient shape -> either
    z_tkm = xnp.concatenate([tkm_x, tkm_y], axis=-1)        # Tucker ranks add
    z_ttm = xnp.concatenate([ttm_x, ttm_y], axis=-1)        # TT ranks add
    return z_tk, z_tt, (z_sm, z_tkm, z_ttm)


def ut3_sum_stack(x: UT3Data) -> UT3Data:  # sum over ALL stack axes -> unstacked UT3 (NOT squashed)
    """Sum the represented dense tensors over the whole stack (the genuine tensor sum, not a corewise
    sum): fold the stack ``S`` into the Tucker rank (merge) and into the TT ranks (block-diagonal over
    all three TT axes via three identities), and likewise reshape the masks. The frontend then squashes
    the tails, which performs the summation. ``S``-fold generalization of :py:func:`ut3_add`.
    """
    use_jax = tree_contains_jax(x[:2])
    xnp, _, _ = get_backend(True, use_jax)

    tk, tt, (sm, tkm, ttm) = x
    d = tk.shape[0]
    stack = tk.shape[1:-2]
    if len(stack) == 0:
        return x
    S = math.prod(stack)

    n, N = tk.shape[-2:]
    rL, nt, rR = tt.shape[-3:]
    I_S = xnp.eye(S)

    new_tk = tk.reshape((d, S, n, N)).reshape((d, S * n, N))                       # merge S into Tucker rank
    tt_dSaib = tt.reshape((d, S, rL, nt, rR))
    tt_block = xnp.einsum('dsaib,sx,sy,sz->dxayizb', tt_dSaib, I_S, I_S, I_S)      # block-diagonal in s
    new_tt = tt_block.reshape((d, S * rL, S * nt, S * rR))

    new_tkm = tkm.reshape((d, S * n))                                              # masks reshape the same way
    new_ttm = ttm.reshape((d + 1, S * rR))
    return new_tk, new_tt, (sm, new_tkm, new_ttm)


def ut3_inner_product(x: UT3Data, y: UT3Data) -> NDArray:  # HS inner product, shape=stack_shape
    """Hilbert-Schmidt inner product of two uniform Tucker tensor trains. Masks (zero padding) and
    squashes both, absorbs the Tucker cores into the TT cores, then zippers the two trains to a scalar
    via a scan over the modes. Orthogonalization (for stability) is the frontend's job, applied first.
    """
    use_jax = tree_contains_jax((x[:2], y[:2]))
    xnp, _, xscan = get_backend(True, use_jax)

    mtk_x, mtt_x = ut3_masking.apply_masks_to_cores(x)
    mtk_y, mtt_y = ut3_masking.apply_masks_to_cores(y)
    mtt_x = ut3_operations.uniform_squash_tt_tails(mtt_x)
    mtt_y = ut3_operations.uniform_squash_tt_tails(mtt_y)

    big_x = t3_operations.absorb_tucker_into_tt(mtk_x, mtt_x)   # (d,)+stack+(rL, N, rR)
    big_y = t3_operations.absorb_tucker_into_tt(mtk_y, mtt_y)

    stack_shape = mtk_x.shape[1:-2]
    rx = mtt_x.shape[-1]
    ry = mtt_y.shape[-1]

    def _push(M_ab, G_x_y):
        Gx_aob, Gy_cod = G_x_y
        M_cd = xnp.einsum('...ab,...aoc,...bod->...cd', M_ab, Gx_aob, Gy_cod)
        return M_cd, (0,)

    M0 = xnp.ones(stack_shape + (rx, ry))
    Mf, _ = xscan(_push, M0, (big_x, big_y))
    return xnp.einsum('...ab->...', Mf)


def ut3_norm_orthogonalized(x: UT3Data) -> NDArray:  # HS norm, shape=stack_shape
    """Hilbert-Schmidt norm of an already-left-orthogonalized uniform T3 (the frontend left-orthogonalizes
    first): masks + squashes, then the norm is the Frobenius norm of the last TT core (all others are
    orthonormal). Mirrors the ragged ``t3_norm`` fast path.
    """
    use_jax = tree_contains_jax(x[:2])
    xnp, _, _ = get_backend(True, use_jax)

    _, mtt = ut3_masking.apply_masks_to_cores(x)
    mtt = ut3_operations.uniform_squash_tt_tails(mtt)

    Gf = mtt[-1].sum(axis=-1)                 # last TT core, trailing bond summed -> stack+(r,n)
    norm_sq = (Gf * Gf).sum(axis=(-2, -1))    # over (r, n); keep the stack
    return xnp.sqrt(xnp.abs(norm_sq))
