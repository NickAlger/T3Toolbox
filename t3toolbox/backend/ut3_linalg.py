# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Linear algebra on uniform supercores (dense-tensor semantics) with mask bookkeeping.

``ut3_scale``/``ut3_add``/``ut3_sum_stack``/``ut3_inner_product``/``ut3_norm_orthogonalized``, plus the
weighted-layer ``ut3_weighted_norm``/``ut3_weighted_inner`` (uniform twins of ``t3_weighted_norm``/
``t3_weighted_inner``). Output masks follow the rank recurrences (add = mask concatenation) on the host
(``np``), while the supercores flow through ``xnp`` (``docs/uniform_masks_vs_ranks.md``).
"""
import numpy as np
import typing as typ
import math

import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ut3_operations as ut3_operations
import t3toolbox.backend.ut3_orthogonalization as ut3_orthogonalization
import t3toolbox.backend.t3_operations as t3_operations
from t3toolbox.backend.common import *

__all__ = [
    'ut3_scale',
    'ut3_add',
    'ut3_sum_stack',
    'ut3_inner_product',
    'ut3_norm_orthogonalized',
    'ut3_weighted_norm',
    'ut3_weighted_inner',
]

# A uniform-T3 .data tuple: (tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask)).
# `shape` is a static int tuple; the two rank masks are HOST bool, static structure (numpy, never
# traced); the supercores are xnp data.
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]]


def ut3_scale(x: UT3Data, s) -> UT3Data:  # z = s * x
    """Scale a uniform Tucker tensor train by a scalar (scales the last Tucker supercore slice; shape and
    masks unchanged)."""
    use_jax = tree_contains_jax(x[:2])
    xnp, _, _ = get_backend(True, use_jax)
    tk, tt, shape, masks = x
    scaled = xnp.concatenate([tk[:-1], s * tk[-1:]], axis=0)
    return scaled, tt, shape, masks


def ut3_add(x: UT3Data, y: UT3Data) -> UT3Data:  # z = x + y (ranks add; NOT squashed)
    """Add two uniform Tucker tensor trains (direct sum): concatenate Tucker supercores along the rank
    axis, block-diagonalize the TT supercores, and concatenate the masks (``shape`` via OR). Vectorized
    over ``(d,)+stack`` (the ``xnp.block`` acts on the last 3 axes). ``x``,``y`` need not share padded
    ``n``/``r``; only ``N``, ``d``, ``stack_shape`` must match (the frontend enforces that).
    """
    use_jax = tree_contains_jax((x[:2], y[:2]))
    xnp, _, _ = get_backend(True, use_jax)

    tk_x, tt_x, shape, (tkm_x, ttm_x) = x
    tk_y, tt_y, _,     (tkm_y, ttm_y) = y
    require_concrete_masks(tkm_x, ttm_x, tkm_y, ttm_y)  # masks are host, not traced

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

    # masks via np (host): + concatenates the rank masks -- the closure under +. The physical `shape` is
    # shared (the frontend enforces x.shape == y.shape), so it passes straight through.
    # np, not xnp: masks are static structure (a jax mask is a tracer under jit). See the module note.
    z_tkm = np.concatenate([tkm_x, tkm_y], axis=-1)         # Tucker ranks add
    z_ttm = np.concatenate([ttm_x, ttm_y], axis=-1)         # TT ranks add
    return z_tk, z_tt, shape, (z_tkm, z_ttm)


def ut3_sum_stack(x: UT3Data) -> UT3Data:  # sum over ALL stack axes -> unstacked UT3 (NOT squashed)
    """Sum the represented dense tensors over the whole stack (the genuine tensor sum, not a corewise
    sum): fold the stack ``S`` into the Tucker rank (merge) and into the TT ranks (block-diagonal over
    all three TT axes via three identities), and likewise reshape the masks. The frontend then squashes
    the tails, which performs the summation. ``S``-fold generalization of :py:func:`ut3_add`.
    """
    use_jax = tree_contains_jax(x[:2])
    xnp, _, _ = get_backend(True, use_jax)

    tk, tt, shape, (tkm, ttm) = x
    require_concrete_masks(tkm, ttm)  # masks are host, not traced
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
    return new_tk, new_tt, shape, (new_tkm, new_ttm)


def ut3_inner_product(x: UT3Data, y: UT3Data) -> NDArray:  # HS inner product, shape=stack_shape
    """Hilbert-Schmidt inner product of two uniform Tucker tensor trains. Masks (zero padding) and
    squashes both, absorbs the Tucker cores into the TT cores, then zippers the two trains to a scalar
    via a scan over the modes. Orthogonalization (for stability) is the frontend's job, applied first.
    """
    use_jax = tree_contains_jax((x[:2], y[:2]))
    xnp, _, xscan = get_backend(True, use_jax)

    mtk_x, mtt_x = ut3_masking.ut3_apply_masks(x)
    mtk_y, mtt_y = ut3_masking.ut3_apply_masks(y)
    mtt_x = tt_operations.tt_squash_tails(mtt_x)
    mtt_y = tt_operations.tt_squash_tails(mtt_y)

    big_x = t3_operations.t3_absorb_tucker_into_tt(mtk_x, mtt_x)   # (d,)+stack+(rL, N, rR)
    big_y = t3_operations.t3_absorb_tucker_into_tt(mtk_y, mtt_y)

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

    _, mtt = ut3_masking.ut3_apply_masks(x)
    mtt = tt_operations.tt_squash_tails(mtt)

    Gf = mtt[-1].sum(axis=-1)                 # last TT core, trailing bond summed -> stack+(r,n)
    norm_sq = (Gf * Gf).sum(axis=(-2, -1))    # over (r, n); keep the stack
    return xnp.sqrt(xnp.abs(norm_sq))


def _ut3_left_orthogonalized(data: UT3Data) -> UT3Data:  # down-orth the Tucker, then left-orth the TT
    """The two-step chain the norm fast path assumes: down-orthogonalize the Tucker supercores, then
    left-orthogonalize the TT supercores. Order matters, and neither step alone suffices.

    (Asymmetry worth knowing: the ragged backend's ``t3_norm``/``t3_inner_product`` orthogonalize
    internally behind a ``use_orthogonalization`` flag, but the uniform backend exposes only the
    already-orthogonalized fast path ``ut3_norm_orthogonalized`` and leaves this composition to the
    frontend -- so there is no ``ut3_norm``/``ut3_inner`` twin to delegate to here. That gap is an
    unfinished port rather than a design decision, and filling it means promoting this helper; it is
    **wanted eventually, low priority** -- logged in
    ``docs/contributor/deferred_and_rejected.md``. This helper keeps the chain in one place meanwhile.)
    """
    return ut3_orthogonalization.ut3_left_orthogonalize_tt_cores(
        ut3_orthogonalization.ut3_down_orthogonalize_tucker_cores(data))


def ut3_weighted_norm(
        x:       UT3Data,                              # (tucker_supercore, tt_supercore, shape, masks)
        weights: ut3_operations.UT3WeightsData,        # (tucker_weight_supercore, tt_weight_supercore, masks)
        use_orthogonalization: bool = True,            # for numerical stability
) -> NDArray:                                          # weighted HS norm, shape=stack_shape
    """Weighted Hilbert-Schmidt norm of a uniform Tucker tensor train -- the norm of the fully-weighted
    network, ``norm(absorb(x, weights))``. Uniform twin of ``t3_weighted_norm``.

    The plain norm **squares** the inserted diagonals, so ``weights = 1/sigma`` penalises by
    ``1/sigma^2``. Absorbing breaks any orthogonality ``x`` had (that is what the weights do), so the
    orthogonalization here runs on the *weighted* train, as in ragged.

    Weighting does not mask, but the norm does: the reduction is the existing plain uniform norm, which
    masks its own input on entry -- so the garbage padding ``absorb`` passes through is zeroed there,
    where reductions are, not here (``dev/uniform_weighting_design.md`` §2).

    **Precondition:** ``weights``' masks must equal ``x``'s masks
    (:py:func:`~t3toolbox.backend.ut3_operations.ut3_weights_consistent`); the frontend enforces it.
    """
    weighted = ut3_operations.ut3_absorb_weights(x, weights)
    if use_orthogonalization:
        return ut3_norm_orthogonalized(_ut3_left_orthogonalized(weighted))
    xnp, _, _ = get_backend(True, tree_contains_jax(weighted[:2]))
    return xnp.sqrt(xnp.abs(ut3_inner_product(weighted, weighted)))


def ut3_weighted_inner(
        x_A:       UT3Data,                        # (tucker_supercore, tt_supercore, shape, masks) of A
        weights_A: ut3_operations.UT3WeightsData,  # weights of A
        x_B:       UT3Data,                        # (tucker_supercore, tt_supercore, shape, masks) of B
        weights_B: ut3_operations.UT3WeightsData,  # weights of B
        use_orthogonalization: bool = True,        # for numerical stability
) -> NDArray:                                      # weighted HS inner, shape=stack_shape
    """Weighted Hilbert-Schmidt inner product ``<absorb(A, weights_A), absorb(B, weights_B)>`` of two
    weighted uniform Tucker tensor trains. Uniform twin of ``t3_weighted_inner``.

    A and B must share physical ``shape`` (the same ambient space); their ranks, masks and weights may
    differ from each other. Each operand's weights must match *its own* object's masks
    (:py:func:`~t3toolbox.backend.ut3_operations.ut3_weights_consistent`); the frontend enforces it.
    """
    weighted_A = ut3_operations.ut3_absorb_weights(x_A, weights_A)
    weighted_B = ut3_operations.ut3_absorb_weights(x_B, weights_B)
    if use_orthogonalization:
        weighted_A = _ut3_left_orthogonalized(weighted_A)
        weighted_B = _ut3_left_orthogonalized(weighted_B)
    return ut3_inner_product(weighted_A, weighted_B)
