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

if jax_available:          # the custom_jvp rules below need jax itself, not only jnp
    import jax

__all__ = [
    'ut3_scale',
    'ut3_add',
    'ut3_sum_stack',
    'ut3_inner_product',
    'ut3_norm_orthogonalized',
    'ut3_norm',
    'ut3_inner',
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


def _ut3_inner_product_step(
        M_ab:   NDArray,                      # carry: stack_shape+(rx_i, ry_i)
        G_x_y:  typ.Tuple[NDArray, NDArray],  # (Gx, Gy) for one mode, each stack+(rL, N, rR)
) -> typ.Tuple[NDArray, typ.Tuple[int]]:      # (next carry, (0,) -- only the terminal carry is used)
    '''One mode of the zipper of :py:func:`ut3_inner_product`. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((M_ab, G_x_y)))   # only xnp; it ignores the flag
    Gx_aob, Gy_cod = G_x_y
    M_cd = xnp.einsum('...ab,...aoc,...bod->...cd', M_ab, Gx_aob, Gy_cod)
    return M_cd, (0,)


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

    M0 = xnp.ones(stack_shape + (rx, ry))
    Mf, _ = xscan(_ut3_inner_product_step, M0, (big_x, big_y))
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


# --------------------------------------------------------------------------------------------------
# ut3_norm / ut3_inner: the orthogonalize-then-reduce twins of the ragged t3_norm / t3_inner_product, with
# an autodiff rule that keeps the SVD out of the derivative.
#
# The VALUE goes through the SVD-based left-orthogonalization (precise: the norm of an orthogonal chain is
# the norm of its last core). Differentiating THROUGH that path is not an option: jax's SVD JVP carries
# 1/(s_i^2 - s_j^2) terms, and a padded uniform train has several exactly-zero singular values, so every
# gradient came out NaN (review S11). The derivative is instead the exact multilinear rule
#     d<T(x), T(x)> = 2 <T(x), dT>,   d<T(x), T(y)> = <dT_x, T(y)> + <T(x), dT_y>,
# with dT the directional derivative of the represented tensor along the core perturbation, obtained by
# jax.jvp of the zipper inner product with the ORTHOGONALIZED other side held fixed: the zipper is plain
# einsums (exact derivative), one operand is orthonormal (well conditioned), and the SVD only ever sees
# primals. Eigenvalue-sensitivity-based derivatives are future work.
# --------------------------------------------------------------------------------------------------
def _norm_sq_value(x: UT3Data):                  # ‖T(x)‖², precise path
    n = ut3_norm_orthogonalized(_ut3_left_orthogonalized(x))
    return n * n


def _inner_value(x: UT3Data, y: UT3Data):        # <T(x), T(y)>, both orthogonalized first
    return ut3_inner_product(_ut3_left_orthogonalized(x), _ut3_left_orthogonalized(y))


def _norm_sq_jax(shape, masks):
    """``(tk, tt) -> ‖T‖²`` as a jax custom_jvp closing over the static structure (shape ints, host masks)."""
    @jax.custom_jvp
    def f(tk, tt):
        return _norm_sq_value((tk, tt, shape, masks))

    @f.defjvp
    def f_jvp(primals, tangents):
        tk, tt = primals
        dtk, dtt = tangents
        xo = _ut3_left_orthogonalized((tk, tt, shape, masks))                 # primals only: no SVD derivative
        _, d_inner = jax.jvp(lambda a, b: ut3_inner_product(xo, (a, b, shape, masks)), (tk, tt), (dtk, dtt))
        return _norm_sq_value((tk, tt, shape, masks)), 2.0 * d_inner
    return f


def _inner_jax(shape_x, masks_x, shape_y, masks_y):
    @jax.custom_jvp
    def f(tkx, ttx, tky, tty):
        return _inner_value((tkx, ttx, shape_x, masks_x), (tky, tty, shape_y, masks_y))

    @f.defjvp
    def f_jvp(primals, tangents):
        tkx, ttx, tky, tty = primals
        dtkx, dttx, dtky, dtty = tangents
        xo = _ut3_left_orthogonalized((tkx, ttx, shape_x, masks_x))
        yo = _ut3_left_orthogonalized((tky, tty, shape_y, masks_y))
        _, d_y = jax.jvp(lambda a, b: ut3_inner_product(xo, (a, b, shape_y, masks_y)), (tky, tty), (dtky, dtty))
        _, d_x = jax.jvp(lambda a, b: ut3_inner_product((a, b, shape_x, masks_x), yo), (tkx, ttx), (dtkx, dttx))
        return ut3_inner_product(xo, yo), d_x + d_y
    return f


def ut3_norm(
        x:  UT3Data,                                  # (tucker_supercore, tt_supercore, shape, masks)
        use_orthogonalization: bool = True,           # True (default): orthogonalize first -- the stable path
) -> NDArray:                                         # HS norm ‖T(x)‖, shape=stack_shape
    """Hilbert-Schmidt norm of a uniform Tucker tensor train -- the twin of the ragged ``t3_norm``.
    ``use_orthogonalization=True`` (default) left-orthogonalizes first and reads the last core's norm
    (numerically stable; the zipper alternative accumulates roundoff along the chain); its jax derivative is
    the exact multilinear rule above, so ``jax.grad`` through it is finite on any padded train.
    ``False`` is the raw zipper ``sqrt(<x, x>)`` (cheaper, less stable; differentiable by plain autodiff)."""
    tk, tt, shape, masks = x
    xnp, _, _ = get_backend(True, tree_contains_jax((tk, tt)))
    if not use_orthogonalization:
        return xnp.sqrt(xnp.abs(ut3_inner_product(x, x)))
    if jax_available and tree_contains_jax((tk, tt)):
        return xnp.sqrt(xnp.abs(_norm_sq_jax(shape, masks)(tk, tt)))
    return xnp.sqrt(xnp.abs(_norm_sq_value(x)))


def ut3_inner(
        x:  UT3Data,                                  # (tucker_supercore, tt_supercore, shape, masks)
        y:  UT3Data,                                  # same physical shape; ranks/masks/padding may differ
        use_orthogonalization: bool = True,           # True (default): orthogonalize both first (stable)
) -> NDArray:                                         # HS inner product <T(x), T(y)>, shape=stack_shape
    """Hilbert-Schmidt inner product of two uniform Tucker tensor trains -- the twin of the ragged
    ``t3_inner_product``. See :py:func:`ut3_norm` for the orthogonalization / autodiff story."""
    if not use_orthogonalization:
        return ut3_inner_product(x, y)
    if jax_available and tree_contains_jax((x[:2], y[:2])):
        return _inner_jax(x[2], x[3], y[2], y[3])(x[0], x[1], y[0], y[1])
    return _inner_value(x, y)


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
    where reductions are, not here (``docs/contributor/weighted_internals.md`` §2).

    **Precondition:** ``weights``' masks must equal ``x``'s masks
    (:py:func:`~t3toolbox.backend.ut3_operations.ut3_weights_consistent`); the frontend enforces it.
    """
    weighted = ut3_operations.ut3_absorb_weights(x, weights)
    return ut3_norm(weighted, use_orthogonalization)


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
    return ut3_inner(weighted_A, weighted_B, use_orthogonalization)
