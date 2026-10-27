# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Linear algebra on ragged t3 data, defined w.r.t. the REPRESENTED dense tensor.

``t3_add``/``t3_scale``/``t3_inner_product``/``t3_norm``/``t3_mult`` and the ``t3m_*`` forms
simulate operations on the dense tensor, using the cores only as a computational device (add
concatenates ranks; nothing here is corewise). ``t3_sum_stack`` is the rank-growing tensor sum
over a stack -- not the rank-preserving corewise core sum. The exception, marked by its name, is the
inner product's derivatives in the cores: ``t3_inner_product_corewise_transpose`` (the gradient of
``theta -> <z, Phi(theta)>``) and ``t3_inner_product_corewise_hessian[_from_sweep]`` (its Hessian applied to
directions, the exact Riemannian Hessian's T3-T3 term), with ``t3_precompute_inner_product_sweep``.
"""
import numpy as np
import typing as typ
import math

from t3toolbox.backend.t3_operations import tt_squash_tails
import t3toolbox.backend.t3_constructors as t3_constructors
import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.t3_orthogonalization as ragged_orth
import t3toolbox.backend.t3_operations as t3_ops
import t3toolbox.backend.t3_svd as ragged_t3svd
import t3toolbox.backend.tt_orthogonalization as orth
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.ranks as ranks
from t3toolbox.backend.common import *

__all__ = [
    't3_add',
    't3_sum_stack',
    't3_scale',
    't3_inner_product',
    't3_norm',
    't3_mult',
    't3m_form_then_round',
    't3m_inplace_fused',
    't3m_swap',
    't3_plus_scalar',
    't3_weighted_norm',
    't3_weighted_inner',
    't3_inner_product_corewise_transpose',
    't3_precompute_inner_product_sweep',
    't3_inner_product_corewise_hessian',
    't3_inner_product_corewise_hessian_from_sweep',
]


def t3_add(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # x_plus_y tucker_cores
    typ.Tuple[NDArray, ...],  # x_plus_y tt_cores
]:
    """Add Tucker tensor trains x and y, yielding a Tucker tensor train with summed ranks.
    """
    use_jax = tree_contains_jax((x, y))
    xnp, xmap, _ = get_backend(False, use_jax)

    #
    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    if vsx != vsy:
        raise ValueError('stack_shape mismatch: %s vs %s' % (vsx, vsy))

    tucker_cores_z = [xnp.concatenate([Bx, By], axis=-2) for Bx, By in zip(tucker_cores_x, tucker_cores_y)]

    tt_cores_z = []

    for Gx, Gy in zip(tt_cores_x, tt_cores_y):
        G000 = Gx
        G001 = xnp.zeros(vsx + (Gx.shape[-3], Gx.shape[-2], Gy.shape[-1]))
        G010 = xnp.zeros(vsx + (Gx.shape[-3], Gy.shape[-2], Gx.shape[-1]))
        G011 = xnp.zeros(vsx + (Gx.shape[-3], Gy.shape[-2], Gy.shape[-1]))
        G100 = xnp.zeros(vsx + (Gy.shape[-3], Gx.shape[-2], Gx.shape[-1]))
        G101 = xnp.zeros(vsx + (Gy.shape[-3], Gx.shape[-2], Gy.shape[-1]))
        G110 = xnp.zeros(vsx + (Gy.shape[-3], Gy.shape[-2], Gx.shape[-1]))
        G111 = Gy
        Gz = xnp.block([[[G000, G001], [G010, G011]], [[G100, G101], [G110, G111]]])
        tt_cores_z.append(Gz)

    return tuple(tucker_cores_z), tuple(tt_cores_z)


def t3_sum_stack(
        x:      typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        axis:   typ.Union[int, typ.Sequence[int], None] = None,  # stack axes to sum over (None: all)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # summed_tucker_cores
    typ.Tuple[NDArray, ...],  # summed_tt_cores
]:
    """Sum the dense tensors represented by a stacked Tucker tensor train over stack axes.

    This is the genuine tensor sum (summing the represented dense tensors), NOT a corewise sum
    of the core arrays. The summed-over stack axes are removed; any remaining stack axes are kept.

    Ranks grow: summing over stack axes whose sizes multiply to S multiplies every Tucker and TT
    rank by S. This is the S-fold generalization of t3_add (which is the S=2 case): the stack is
    folded into the Tucker ranks (by merging) and into the TT ranks (block-diagonally), then the
    leading and trailing TT tails are squashed, which performs the sum.
    """
    tucker_cores, tt_cores = x

    use_jax = tree_contains_jax(x)
    xnp, _, _ = get_backend(False, use_jax)

    #
    stack_shape = tucker_cores[0].shape[:-2]
    m = len(stack_shape)

    if axis is None:
        summed_axes = list(range(m))
    elif isinstance(axis, np.ndarray):                     # ndarray axis (0-d or 1-d) works too (R2-13)
        summed_axes = [int(a) for a in np.atleast_1d(axis)]
    elif not isinstance(axis, typ.Sequence):
        summed_axes = [int(axis)]
    else:
        summed_axes = [int(a) for a in axis]

    summed_axes = sorted(set((ax + m) if ax < 0 else ax for ax in summed_axes))
    for ax in summed_axes:
        if not (0 <= ax < m):
            raise ValueError('stack axis %d out of range for %d stack axes' % (ax, m))

    if len(summed_axes) == 0: # nothing to sum over
        return tuple(B.copy() for B in tucker_cores), tuple(G.copy() for G in tt_cores)

    kept_axes = [k for k in range(m) if k not in summed_axes]
    KS = tuple(stack_shape[k] for k in kept_axes) # kept stack shape
    S = math.prod([stack_shape[a] for a in summed_axes]) # total size of summed stack axes

    I_ss = xnp.eye(S)

    def _gather_summed_axes(core): # core -> KS + (S,) + core_own_axes
        own_axes = list(range(m, core.ndim))
        core = xnp.transpose(core, kept_axes + summed_axes + own_axes)
        own_shape = core.shape[len(kept_axes) + len(summed_axes):]
        return core.reshape(KS + (S,) + own_shape)

    summed_tucker_cores = []
    for B in tucker_cores:
        B_KSsio = _gather_summed_axes(B) # KS + (S, ni, Ni)
        ni, Ni = B_KSsio.shape[-2:]
        B_new = B_KSsio.reshape(KS + (S * ni, Ni)) # merge S into Tucker rank
        summed_tucker_cores.append(B_new)

    summed_tt_cores = []
    for G in tt_cores:
        G_KSsaib = _gather_summed_axes(G) # KS + (S, rLi, ni, rRi)
        rLi, ni, rRi = G_KSsaib.shape[-3:]
        G_block = xnp.einsum('...saib,sx,sy,sz->...xayizb', G_KSsaib, I_ss, I_ss, I_ss) # block diagonal in s
        G_new = G_block.reshape(KS + (S * rLi, S * ni, S * rRi))
        summed_tt_cores.append(G_new)

    summed_tt_cores = tt_squash_tails(tuple(summed_tt_cores)) # ones-contraction at the tails performs the sum
    return tuple(summed_tucker_cores), summed_tt_cores


def t3_scale(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        s,  # scalar
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # scaled tucker_cores (x * s)
    typ.Tuple[NDArray, ...],  # tt_cores (unchanged)
]:
    """Multipy a Tucker tensor train by a scaling factor.
    """
    tucker_cores, tt_cores = x

    scaled_tucker_cores = [B.copy() for B in tucker_cores]
    scaled_tucker_cores[-1] = scaled_tucker_cores[-1] * s

    copied_tt_cores = [G.copy() for G in tt_cores]

    return tuple(scaled_tucker_cores), tuple(copied_tt_cores)


def t3_inner_product(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores_y, tt_cores_y)
        use_orthogonalization: bool = True,  # for numerical stability
) -> NDArray:  # HS inner product, shape=stack_shape (scalar if unstacked)
    """Compute Hilbert-Schmidt inner product of two Tucker tensor trains.
    """
    use_jax = any([is_jax_ndarray(c) for c in tuple(x[0]) + tuple(x[1]) + tuple(y[0]) + tuple(y[1])])   # list/tuple mix ok (R2-13)
    xnp, _, _ = get_backend(False, use_jax)

    #
    x = (x[0], tt_squash_tails(x[1]))
    y = (y[0], tt_squash_tails(y[1]))

    if use_orthogonalization:
        x = ragged_orth.t3_left_orthogonalize(x)
        y = ragged_orth.t3_left_orthogonalize(y)

    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    if vsx != vsy:
        raise ValueError('stack_shape mismatch: %s vs %s' % (vsx, vsy))

    r0_x = tt_cores_x[0].shape[-3]
    r0_y = tt_cores_y[0].shape[-3]
    stack_shape = x[0][0].shape[:-2]

    M_sp = xnp.ones(stack_shape + (r0_x, r0_y))
    for Bx_ai, Gx_sat, By_bi, Gy_pbq in zip(tucker_cores_x, tt_cores_x, tucker_cores_y, tt_cores_y):
        tmp_ab = xnp.einsum('...ai,...bi->...ab', Bx_ai, By_bi)
        tmp_sbt = xnp.einsum('...sat,...ab->...sbt', Gx_sat, tmp_ab)
        tmp_pbt = xnp.einsum('...sp,...sbt->...pbt', M_sp, tmp_sbt)
        tmp_tq = xnp.einsum('...pbt,...pbq->...tq', tmp_pbt, Gy_pbq)
        M_sp = tmp_tq

    rd_x = tt_cores_x[-1].shape[-1]
    rd_y = tt_cores_y[-1].shape[-1]

    result = xnp.einsum('...tq,t,q', M_sp, xnp.ones(rd_x), xnp.ones(rd_y))
    return result


def t3_norm(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        use_orthogonalization: bool = True,  # for numerical stability
) -> NDArray:  # HS norm, shape=stack_shape (scalar if unstacked)
    """Compute Hilbert-Schmidt norm of a Tucker tensor train.
    """
    use_jax = any([is_jax_ndarray(B) for B in x[0]] + [is_jax_ndarray(G) for G in x[1]])
    xnp, _, _ = get_backend(False, use_jax)

    #
    x = (x[0], tt_squash_tails(x[1]))
    if use_orthogonalization:
        x = ragged_orth.t3_left_orthogonalize(x)
        Gf = x[1][-1].sum(axis=-1)
        norm_sq = (Gf*Gf).sum(axis=(-2,-1)) # Don't sum over stacked axes
    else:
        norm_sq = t3_inner_product(x, x, use_orthogonalization=False)   # honor the flag: the raw zipper

    return xnp.sqrt(xnp.abs(norm_sq))


def t3_mult(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # x_times_y tucker_cores
    typ.Tuple[NDArray, ...],  # x_times_y tt_cores
]:
    """Pointwise multiply Tucker tensor trains x and y, yielding a Tucker tensor train with multiplied ranks.

    This is the conventional "dumb" algorithm which does not do intermediate rank truncation.
    Ideally, we should also implement the newer "TTM" algorithm at some point.
    """
    use_jax = tree_contains_jax((x, y))
    xnp, xmap, _ = get_backend(False, use_jax)

    #
    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    if vsx != vsy:
        raise ValueError('stack_shape mismatch: %s vs %s' % (vsx, vsy))

    tucker_cores_xy = []
    for Bx, By in zip(tucker_cores_x, tucker_cores_y):
        nx, Nx = Bx.shape[-2:]
        ny, Ny = By.shape[-2:]
        Bxy0 = xnp.einsum('...io,...jo->...ijo', Bx, By)
        Bxy = Bxy0.reshape(vsx + (nx*ny, Nx))
        tucker_cores_xy.append(Bxy)
    tucker_cores_xy = tuple(tucker_cores_xy)

    tt_cores_xy = []
    for Gx, Gy in zip(tt_cores_x, tt_cores_y):
        rLx, nx, rRx = Gx.shape[-3:]
        rLy, ny, rRy = Gy.shape[-3:]
        Gxy0 = xnp.einsum('...aib,...ujv->...auijbv', Gx, Gy)
        Gxy = Gxy0.reshape(vsx + (rLx*rLy, nx*ny, rRx*rRy))
        tt_cores_xy.append(Gxy)
    tt_cores_xy = tuple(tt_cores_xy)

    return tucker_cores_xy, tt_cores_xy


def t3m_form_then_round(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores_y, tt_cores_y)

        max_tucker_ranks:   typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d
        max_tt_ranks:       typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d+1

        rtol:               typ.Optional[float] = None,  # requires unstacked (enforced by the frontend)
        atol:               typ.Optional[float] = None,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # x_times_y tucker_cores
    typ.Tuple[NDArray, ...],  # x_times_y tt_cores
]:
    '''Elementwise product ``x ⊙ y`` -- method (a): form the full product, then round.

    Forms the full Khatri-Rao/Kronecker product (:py:func:`t3_mult`, multiplied ranks) and, if any
    truncation is requested, rounds it with :py:func:`t3svd`; with no truncation it returns the exact
    full product directly. The forming step is embarrassingly parallel (no sweep) but materializes the
    whole product -- see ``docs/ttm_t3m_ht_note.tex`` for the cost trade-off vs the fused / swap methods.
    Stack-aware with max-rank truncation; ``rtol``/``atol`` require unstacked.
    '''
    product = t3_mult(x, y)
    if max_tucker_ranks is None and max_tt_ranks is None and rtol is None and atol is None:
        return product
    rounded, _, _ = ragged_t3svd.t3svd(
        product, max_tt_ranks=max_tt_ranks, max_tucker_ranks=max_tucker_ranks, rtol=rtol, atol=atol)
    return rounded


def t3m_inplace_fused(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores_y, tt_cores_y)

        max_tucker_ranks:   typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d
        max_tt_ranks:       typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d+1

        rtol:               typ.Optional[float] = None,  # requires unstacked (enforced by the frontend)
        atol:               typ.Optional[float] = None,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # x_times_y tucker_cores
    typ.Tuple[NDArray, ...],  # x_times_y tt_cores
]:
    '''Elementwise product ``x ⊙ y`` -- method (b): a fused left-to-right sweep that truncates as it
    goes, never materializing the full product (the workhorse; see ``docs/ttm_t3m_ht_note.tex``).

    Right-orthogonalizes the two central TTs separately (the Kronecker of right-canonical cores is
    right-canonical, so the product's central TT is right-canonical without being formed), then sweeps
    left to right carrying the *separate* ``(r_x, r_y)`` bonds. At each mode it builds the product core
    on the fly, then does the **joint** per-site T3 truncation (the Tucker rank, weighted by the
    canonical environment, then the TT bond) -- both truncations are optimal because the right side is
    right-canonical. ``O(d·r⁴)``, memory ``O(r̃·n²·r²)`` (one site). Stack-aware with max-rank
    truncation; ``rtol``/``atol`` require unstacked. No truncation requested -> the exact full product.
    '''
    if max_tucker_ranks is None and max_tt_ranks is None and rtol is None and atol is None:
        return t3_mult(x, y)

    Ux, Gx = x
    Uy, Gy = y
    d = len(Ux)
    use_jax = tree_contains_jax((x, y))
    xnp, _, _ = get_backend(False, use_jax)
    mtr = ranks.normalize_max_ranks(max_tucker_ranks, d)
    mrr = ranks.normalize_max_ranks(max_tt_ranks, d + 1)

    # A truncating product canonicalizes the boundary bonds to 1 (docs/t3m_methods.md): the sweep below
    # never SVDs the last bond, so an unsquashed rd on either operand would survive as rd_x*rd_y, uncapped.
    Gx = tt_operations.tt_squash_tails(Gx)
    Gy = tt_operations.tt_squash_tails(Gy)

    # Right-orthogonalize each central TT (the product's central TT is then implicitly right-canonical).
    Gx = orth.tt_right_orthogonalize(Gx)
    Gy = orth.tt_right_orthogonalize(Gy)

    stack = tuple(Ux[0].shape[:-2])
    out_tucker = []
    out_tt = []
    carry = xnp.ones(stack + (1, 1, 1))  # left center: [center_bond, r_x, r_y], all 1 at the boundary
    for ii in range(d):
        Gxi, Gyi, Uxi, Uyi = Gx[ii], Gy[ii], Ux[ii], Uy[ii]
        t = carry.shape[-3]
        nA, cA = Gxi.shape[-2], Gxi.shape[-1]
        nB, cB = Gyi.shape[-2], Gyi.shape[-1]
        N = Uxi.shape[-1]
        P, C2 = nA * nB, cA * cB

        # Product core at this site: contract the carry with the two central cores.
        site = xnp.einsum('...tab,...anc,...bme->...tnmce', carry, Gxi, Gyi)  # [t, nA, nB, cA, cB]
        site = site.reshape(stack + (t, P, C2))                               # [t, n^2, r^2]

        # Tucker factor W = U_x ⊙ U_y; orthonormalize (full SVD, no truncation): W = Uw diag(sw) Vt.
        W = xnp.einsum('...nx,...mx->...nmx', Uxi, Uyi).reshape(stack + (P, N))
        Uw, sw, Vt = linalg.truncated_svd(W)
        Utilde_full = Vt                                   # row-orthonormal Tucker factor [k, N]
        Mw = xnp.einsum('...pk,...k->...pk', Uw, sw)       # remainder [n^2, k] -> into the site
        site = xnp.einsum('...tpc,...pk->...tkc', site, Mw)  # [t, k, r^2]
        k = site.shape[-2]

        # Joint Tucker truncation (weighted by the canonical environment t and r^2): SVD over the
        # tucker leg.  site -> [k, t*r^2].
        env = xnp.moveaxis(site, -2, -3).reshape(stack + (k, t * C2))
        Qt, st, Vt2 = linalg.truncated_svd(env, max_rank=mtr[ii], rtol=rtol, atol=atol)
        ntil = Qt.shape[-1]
        out_tucker.append(xnp.einsum('...kr,...kN->...rN', Qt, Utilde_full))   # [ntil, N]
        site = (xnp.einsum('...r,...rc->...rc', st, Vt2)                       # [ntil, t*r^2]
                .reshape(stack + (ntil, t, C2)))
        site = xnp.moveaxis(site, -3, -2)                                      # [t, ntil, r^2]

        if ii < d - 1:
            # Joint TT-bond truncation: SVD over (t, ntil) vs r^2.
            A = site.reshape(stack + (t * ntil, C2))
            Ac, sc, Vc = linalg.truncated_svd(A, max_rank=mrr[ii + 1], rtol=rtol, atol=atol)
            rnew = Ac.shape[-1]
            out_tt.append(Ac.reshape(stack + (t, ntil, rnew)))
            carry = (xnp.einsum('...r,...rc->...rc', sc, Vc)
                     .reshape(stack + (rnew, cA, cB)))                          # next carry [rnew, r_x, r_y]
        else:
            out_tt.append(site.reshape(stack + (t, ntil, C2)))  # C2 == 1 at the right boundary

    return tuple(out_tucker), tuple(out_tt)


def _t3m_swap_pair(
        eL,                     # [G, U, mode]: G=stack+(a, nL, b), U=stack+(nL, NL)
        eR,                     # [G, U, mode]: G=stack+(b, nR, c), U=stack+(nR, NR)
        push:       str,        # 'left'|'right' -- which side the orthogonality center lands on
        max_rank,
        rtol,
        atol,
):
    '''Swap two adjacent T3M chain elements via a centered truncated SVD.

    The two central cores share their middle bond; contracting and re-splitting with the
    physical legs grouped the other way swaps the legs (and their Tucker factors / mode
    labels ride along). ``push`` controls where the singular values -- hence the
    orthogonality center -- land, so the caller keeps the center co-located with the swap.
    '''
    GL, UL, mL = eL
    GR, UR, mR = eR
    xnp, _, _ = get_backend(False, is_jax_ndarray(GL) or is_jax_ndarray(GR))

    stack = GL.shape[:-3]
    a, nL, b = GL.shape[-3:]
    _, nR, c = GR.shape[-3:]
    T = xnp.einsum('...anb,...bmc->...anmc', GL, GR)            # stack+(a, nL, nR, c)
    T = xnp.moveaxis(T, -3, -2)                                 # stack+(a, nR, nL, c)
    M = T.reshape(stack + (a * nR, nL * c))
    U, s, Vt = linalg.truncated_svd(M, max_rank=max_rank, rtol=rtol, atol=atol)
    rp = s.shape[-1]
    if push == 'right':
        newGL = U.reshape(stack + (a, nR, rp))                  # left-orth, physical nR
        newGR = (s[..., :, None] * Vt).reshape(stack + (rp, nL, c))
    else:                                                       # push left -> center at left core
        newGL = (U * s[..., None, :]).reshape(stack + (a, nR, rp))
        newGR = Vt.reshape(stack + (rp, nL, c))                 # right-orth, physical nL
    return [newGL, UR, mR], [newGR, UL, mL]                     # Tucker factors / modes swap


def _t3m_contract_pair(
        eL,                     # [G, U, mode]: a_i. G=stack+(aL, nA, b), U=stack+(nA, N)
        eR,                     # [G, U, mode]: b_i. G=stack+(b, nB, c),  U=stack+(nB, N)
        max_tucker_rank,
        rtol,
        atol,
):
    '''Merge a same-mode pair (a_i, b_i) into the product core m_i.

    COPY-merges the Tucker factors (row-wise Khatri-Rao) and fuses the central cores, then
    applies the joint per-site Tucker truncation (the leg ``nA*nB`` weighted by the
    canonical environment of both bonds, as ``t3svd`` does). With the center at the pair the
    environment is orthonormal, so the truncation is optimal; m_i is left as the new center.
    '''
    Ga, Ua, mi = eL
    Gb, Ub, _  = eR
    xnp, _, _ = get_backend(False, is_jax_ndarray(Ga) or is_jax_ndarray(Gb))

    stack = Ga.shape[:-3]
    aL, nA, b = Ga.shape[-3:]
    _, nB, c = Gb.shape[-3:]
    N = Ua.shape[-1]
    P = nA * nB
    site = xnp.einsum('...anb,...bmc->...anmc', Ga, Gb).reshape(stack + (aL, P, c))
    W = xnp.einsum('...nx,...mx->...nmx', Ua, Ub).reshape(stack + (P, N))   # Khatri-Rao Tucker factor
    Uw, sw, Vt = linalg.truncated_svd(W)                       # lossless orthonormalize of W
    Utilde_full = Vt                                           # [k, N]
    Mw = Uw * sw[..., None, :]                                 # [P, k] -> folded into the site
    site = xnp.einsum('...apc,...pk->...akc', site, Mw)        # [aL, k, c]
    k = site.shape[-2]
    env = xnp.moveaxis(site, -2, -3).reshape(stack + (k, aL * c))           # weighted env (both bonds orth)
    Qt, st, Vt2 = linalg.truncated_svd(env, max_rank=max_tucker_rank, rtol=rtol, atol=atol)
    ntil = st.shape[-1]
    Utilde = xnp.einsum('...kr,...kN->...rN', Qt, Utilde_full)              # [ntil, N] new Tucker factor
    site = (st[..., :, None] * Vt2).reshape(stack + (ntil, aL, c))
    site = xnp.moveaxis(site, -3, -2)                          # [aL, ntil, c] = m_i central core (center)
    return [site, Utilde, mi]


def _t3m_move_center(
        chain,                  # list of [G, U, mode]
        frm:    int,
        to:     int,
) -> int:
    '''Move the orthogonality center from index ``frm`` to ``to`` with no truncation.

    Re-gauges the central cores in between (left/right SVD-pair steps), leaving the cores it
    crosses orthogonal toward the new center. Cheap relative to the swaps; keeps every later
    truncation centered.
    '''
    if to > frm:
        for p in range(frm, to):
            nGL, nGR, _ = linalg.left_svd_pair(chain[p][0], chain[p + 1][0])
            chain[p][0], chain[p + 1][0] = nGL, nGR
    elif to < frm:
        for p in range(frm, to, -1):
            nGL, nGR, _ = linalg.right_svd_pair(chain[p - 1][0], chain[p][0])
            chain[p - 1][0], chain[p][0] = nGL, nGR
    return to


def t3m_swap(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores_y, tt_cores_y)

        max_tucker_ranks:   typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d
        max_tt_ranks:       typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d+1

        rtol:               typ.Optional[float] = None,  # requires unstacked (enforced by the frontend)
        atol:               typ.Optional[float] = None,

        oversample:         float = 1,  # >= 1; intermediate rank/tol relaxation factor (see docs/ttm_t3m_ht_note.tex)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # x_times_y tucker_cores
    typ.Tuple[NDArray, ...],  # x_times_y tt_cores
]:
    '''Elementwise product ``x ⊙ y`` -- method (c): the swap-based TTM generalization, for
    ``r ≫ d`` (huge TT bonds, few modes); ``O(d²·r³)`` compute, ``O(r̃²)`` memory.

    Concatenates the two central TTs into a length-``2d`` chain (the second reversed so
    matching modes meet in ``d(d-1)/2`` swaps), keeps each core's Tucker factor riding on its
    leg, then for each mode (``d-1`` down to ``0``) bubbles the matching pair adjacent via
    **gauge-centered truncating swaps** and merges them (Khatri-Rao Tucker factor + joint
    weighted-Tucker truncation). The orthogonality center is tracked explicitly so every
    swap/merge SVD is locally optimal; the Tucker factors are down-orthogonalized up front.

    Because the merged Tucker leg of the last-merged mode is full-rank while its bonds are set
    (a tension absent from pure-TT TTM -- see ``docs/ttm_t3m_ht_note.tex``), in-process
    truncation is only quasi-optimal. ``oversample = k`` resolves this: intermediate ranks are
    capped at ``k×`` the target (tolerances at ``/k``) purely to bound memory, and a single
    ``t3svd`` cleanup at the exact targets does the decisive truncation. ``k=1`` (default) skips
    the cleanup (aggressive corner; with ``rtol`` the uncleaned per-step tolerances accumulate to
    ~``d·rtol`` overall -- use ``oversample>1`` for tighter tolerance-based quality); ``k≈3``
    recovers ``t3svd`` quality; a per-position
    ``max_tt_ranks`` sequence also triggers the cleanup (the swaps cannot honor per-position
    bonds). Stack-aware with max-rank truncation; ``rtol``/``atol`` require unstacked. No
    truncation requested -> the exact full product.
    '''
    if max_tucker_ranks is None and max_tt_ranks is None and rtol is None and atol is None:
        return t3_mult(x, y)
    if oversample < 1:
        raise ValueError('oversample must be >= 1, got %r' % (oversample,))

    Ux, Gx = x
    Uy, Gy = y
    d = len(Ux)

    mtr = ranks.normalize_max_ranks(max_tucker_ranks, d)        # exact targets
    mrr = ranks.normalize_max_ranks(max_tt_ranks, d + 1)
    osc = lambda c: None if c is None else int(math.ceil(oversample * c))
    mtr_in = tuple(osc(c) for c in mtr)
    nn = [r for r in mrr if r is not None]
    mrr_in = osc(max(nn)) if nn else None                      # uniform in-process bond cap
    rtol_in = None if rtol is None else rtol / oversample
    atol_in = None if atol is None else atol / oversample

    # A truncating product canonicalizes the boundary bonds to 1 (docs/t3m_methods.md): the seam
    # contraction pairs the two operands' trailing bonds against each other, which is only the
    # represented-tensor product when both are 1.
    Gx = tt_operations.tt_squash_tails(Gx)
    Gy = tt_operations.tt_squash_tails(Gy)

    # Leaves -> root (fold Tucker weight into the central TTs), then build/gauge the chain.
    Ux, Gx = ragged_orth.t3_down_orthogonalize_tucker_cores((Ux, Gx))
    Uy, Gy = ragged_orth.t3_down_orthogonalize_tucker_cores((Uy, Gy))
    Gx_o = orth.tt_left_orthogonalize(list(Gx))
    Gy_o = orth.tt_right_orthogonalize(tt_operations.tt_reverse(list(Gy)))
    Uy_rev = list(Uy[::-1])
    chain = [[Gx_o[i], Ux[i], i] for i in range(d)] \
          + [[Gy_o[j], Uy_rev[j], d - 1 - j] for j in range(d)]
    center = d - 1                                              # seam pair = (d-1, d)

    for i in range(d - 1, -1, -1):
        if i < d - 1:
            center = _t3m_move_center(chain, center, d - 1)     # to right end of merged block
            for p in range(d - 1, i, -1):                       # bubble b_i left to idx i+1
                chain[p], chain[p + 1] = _t3m_swap_pair(
                    chain[p], chain[p + 1], 'left', mrr_in, rtol_in, atol_in)
                center = p
        chain = chain[:i] + [_t3m_contract_pair(
            chain[i], chain[i + 1], mtr_in[i], rtol_in, atol_in)] + chain[i + 2:]
        center = i

    tucker_cores = tuple(e[1] for e in chain)
    tt_cores = tuple(e[0] for e in chain)

    # per-position bond caps (a list, tuple OR ndarray -- np.ndim, not typ.Sequence, which an ndarray is not)
    need_cleanup = (oversample > 1) or (max_tt_ranks is not None and np.ndim(max_tt_ranks) > 0)
    if need_cleanup:
        rounded, _, _ = ragged_t3svd.t3svd(
            (tucker_cores, tt_cores),
            max_tt_ranks=max_tt_ranks, max_tucker_ranks=max_tucker_ranks, rtol=rtol, atol=atol)
        return rounded
    return tucker_cores, tt_cores


def t3_plus_scalar(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        s,  # scalar
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores of x + s
    typ.Tuple[NDArray, ...],  # tt_cores of x + s
]:
    """Add a scalar to a Tucker tensor train: ``(x + s).to_dense() == x.to_dense() + s``."""
    use_jax = tree_contains_jax(x)

    x_shape = tuple(B.shape[-1] for B in x[0])
    x_stack_shape = x[0][0].shape[:-2]

    y0 = t3_constructors.t3_ones(x_shape, x_stack_shape, use_jax=use_jax)
    y = t3_scale(y0, s)
    xs = t3_add(x, y)
    return xs










def t3_weighted_norm(
        x0:      typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        weights: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_weights len=d, tt_weights len=d+1)
        use_orthogonalization: bool = True,                                # for numerical stability
) -> NDArray:                                                              # weighted HS norm, shape=stack_shape
    """Weighted Hilbert-Schmidt norm of a Tucker tensor train: ``t3_norm(absorb(x0, weights))`` -- the norm
    of the fully-weighted network. The plain norm squares the inserted diagonals (so ``diag(1/sigma)``
    penalises by ``1/sigma^2``)."""
    return t3_norm(t3_ops.t3_absorb_weights(x0, weights), use_orthogonalization=use_orthogonalization)


def t3_weighted_inner(
        x0_A:      typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores) of A
        weights_A: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # weights of A
        x0_B:      typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores) of B
        weights_B: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # weights of B
        use_orthogonalization: bool = True,                                  # for numerical stability
) -> NDArray:                                                                # weighted HS inner, shape=stack_shape
    """Weighted Hilbert-Schmidt inner product ``<absorb(A), absorb(B)>`` of two weighted Tucker tensor
    trains. A and B must share physical shape (same ambient space); ranks/weights may differ."""
    return t3_inner_product(t3_ops.t3_absorb_weights(x0_A, weights_A),
                            t3_ops.t3_absorb_weights(x0_B, weights_B),
                            use_orthogonalization=use_orthogonalization)


# --------------------------------------------------------------------------------------------------
# The inner product <z, Phi(theta)> as a function of the cores: its gradient and Hessian
# (the T3-T3 term of the exact Riemannian Hessian)
# --------------------------------------------------------------------------------------------------
def _inner_product_rebasis_step(
        x: typ.Tuple[NDArray, NDArray, NDArray],   # (G_z, B_z, U): C+(rZi,nZi,rZ(i+1)) ; C+(nZi,Ni) ; C+(nUi,Ni)
) -> typ.Tuple[NDArray]:                           # (G_z2,): C+(rZi,nUi,rZ(i+1))
    """Re-express one TT core of ``z`` in the frame's up-Tucker basis, for
    :py:func:`_inner_product_frame_transpose`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``."""
    G_other, B_other, U = x
    xnp, _, _ = get_backend(False, tree_contains_jax(x))
    BU = xnp.einsum('...iz,...xz->...ix', B_other, U)
    G_other2 = xnp.einsum('...aib,...ix->...axb', G_other, BU)
    return (G_other2,)


def _inner_product_variations_step(
        x: typ.Tuple[NDArray, NDArray, NDArray,
                     NDArray, NDArray, NDArray],   # (ZL, ZR, G_z, B_z, O, U)
) -> typ.Tuple[
    NDArray,  # dG. ungauged tt variation,     C+(rLi,nUi,rR(i+1))
    NDArray,  # dB. ungauged tucker variation, C+(nOi,Ni)
]:
    """One core of :py:func:`_inner_product_frame_transpose`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``."""
    ZL, ZR, G, B, O, U = x
    xnp, _, _ = get_backend(False, tree_contains_jax(x))
    env = xnp.einsum('...ax,...aib,...by->...xiy', ZL, G, ZR)
    BU = xnp.einsum('...io,...jo->...ij', B, U)
    dG = xnp.einsum('...xiy,...ij->...xjy', env, BU)
    M = xnp.einsum('...xiy,...xjy->...ij', env, O)
    dB = xnp.einsum('...ij,...io->...jo', M, B)
    return dG, dB


def _inner_product_frame_transpose(
        frame:  typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores. len=d, elm_shape=C+(nUi, Ni)
            typ.Sequence[NDArray],  # down_tt_cores.   len=d, elm_shape=C+(rLi, nDi, rR(i+1))
            typ.Sequence[NDArray],  # left_tt_cores.   len=d, elm_shape=C+(rLi, nUi, rL(i+1))
            typ.Sequence[NDArray],  # right_tt_cores.  len=d, elm_shape=C+(rRi, nUi, rR(i+1))
        ],
        z:      typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores of z. len=d, elm_shape=K+C+(nZi, Ni); K may be empty
            typ.Sequence[NDArray],  # tt_cores of z.     len=d, elm_shape=K+C+(rZi, nZi, rZ(i+1))
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],        # UNGAUGED tucker_variations. len=d, elm_shape=K+C+(nDi, Ni)
    typ.Tuple[NDArray, ...],        # UNGAUGED tt_variations.     len=d, elm_shape=K+C+(rLi, nUi, rR(i+1))
]:
    """The frame transpose of the inner product with ``z``: ``E^T z``, the coordinates of ``<z, term>``
    against every single-core-replacement term of the frame (``E`` maps variations to the tangent sum).

    Gauged, at an orthogonal frame, this is the orthogonal tangent-space projection of ``z``
    (:py:func:`~t3toolbox.backend.tv_operations.tv_project_t3_onto_tangent_space`); at the corewise
    frame ``(U, G, G, G)`` it is the core gradient of ``<z, Phi(theta)>``
    (:py:func:`t3_inner_product_corewise_transpose`). Algorithm: re-express ``z``'s TT cores in the frame's
    up basis, zip them against the left/right chains, and contract each one-hole environment with the
    core's two variation slots. ``z`` may carry an extra leading stack ``K`` over the frame's ``C`` (one
    projection per stacked ``z``): the ``'...'`` einsums broadcast the frame over it.
    """
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    outer_tt_cores = down_tt_cores

    other_tucker_cores, other_tt_cores = z
    other_tt_cores = tt_operations.tt_squash_tails(other_tt_cores)

    use_jax = tree_contains_jax((frame, z))
    xnp, xmap, _ = get_backend(False, use_jax)

    # Re-express the other T3's TT cores in the frame's up-Tucker basis.
    (other_tt_cores2,) = xmap(
        _inner_product_rebasis_step, (other_tt_cores, other_tucker_cores, up_tucker_cores),
    )

    if len(other_tt_cores2) == 1:
        # d = 1: no chain on either side of the single core -- the environments are the boundary ones
        # matrices (an empty zipper cannot read the stack/bond sizes off an empty tuple).
        ss = other_tt_cores2[0].shape[:-3]
        zipper_left2right = (xnp.ones(ss + (other_tt_cores2[0].shape[-3], left_tt_cores[0].shape[-3])),)
        zipper_right2left = (xnp.ones(ss + (other_tt_cores2[0].shape[-1], right_tt_cores[0].shape[-1])),)
    else:
        zipper_left2right = tt_operations.tt_zipper_left_to_right(other_tt_cores2[:-1], left_tt_cores[:-1])
        zipper_right2left = tt_operations.tt_zipper_right_to_left(other_tt_cores2[1:], right_tt_cores[1:])

    ungauged_tt_variations, ungauged_tucker_variations = xmap(
        _inner_product_variations_step,
        (zipper_left2right, zipper_right2left, other_tt_cores, other_tucker_cores, outer_tt_cores, up_tucker_cores),
    )
    return ungauged_tucker_variations, ungauged_tt_variations


def t3_inner_product_corewise_transpose(
        z:          typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores of z. len=d, elm_shape=C+(nZi, Ni)
            typ.Sequence[NDArray],  # tt_cores of z.     len=d, elm_shape=C+(rZi, nZi, rZ(i+1))
        ],
        core_pair:  typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores of theta. len=d, elm_shape=C+(ni, Ni)
            typ.Sequence[NDArray],  # tt_cores of theta.     len=d, elm_shape=C+(ri, ni, r(i+1))
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],        # tucker gradients. len=d, elm_shape=C+(ni, Ni)
    typ.Tuple[NDArray, ...],        # tt gradients.     len=d, elm_shape=C+(ri, ni, r(i+1))
]:
    """The gradient of ``theta -> <z, Phi(theta)>`` with respect to the raw cores ``theta = core_pair``.

    The corewise transpose of the inner product: the frame transpose of ``<z, .>`` at the corewise frame
    ``(U, G, G, G)`` -- the Section 6.3 substitution ``(P, Q, O) -> G`` of Alger et al. (2026), "Tucker
    Tensor Train Taylor Series" (arXiv:2603.21141), applied to the T3-to-T3 tangent projection. The
    result has the shapes of ``theta``'s cores (a raw tuple, a gradient -- not a tensor). In the exact
    Riemannian Hessian it is the one-hole of the term ``-<grad f, Phi(theta)>``, whose derivative
    :py:func:`t3_inner_product_corewise_hessian` is. Stack ``C`` on both operands.

    Examples
    --------
    Every core enters ``Phi`` exactly once, so ``<z, Phi(theta)>`` is homogeneous of degree ``2d`` in the
    cores and Euler's identity pins the gradient: ``<grad, theta> = 2d <z, Phi(theta)>``.

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.backend.t3_constructors as t3k
    >>> import t3toolbox.backend.t3_linalg as t3la
    >>> np.random.seed(0)
    >>> theta = t3k.t3_corewise_randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
    >>> z = t3k.t3_corewise_randn((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))
    >>> grad = t3la.t3_inner_product_corewise_transpose(z, theta)
    >>> print([g.shape for g in grad[0]], [g.shape for g in grad[1]])
    [(2, 5), (3, 6), (2, 7)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]
    >>> print(bool(np.isclose(cw.corewise_dot(grad, theta), 6 * t3la.t3_inner_product(z, theta))))
    True
    """
    return _inner_product_frame_transpose(fv_conversions.t3_corewise_frame(core_pair), z)


def t3_precompute_inner_product_sweep(
        z:          typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores of z. len=d, elm_shape=C+(nZi, Ni)
            typ.Sequence[NDArray],  # tt_cores of z.     len=d, elm_shape=C+(rZi, nZi, rZ(i+1))
        ],
        core_pair:  typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores of theta. len=d, elm_shape=C+(ni, Ni)
            typ.Sequence[NDArray],  # tt_cores of theta.     len=d, elm_shape=C+(ri, ni, r(i+1))
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],        # BU.  len=d, elm_shape=C+(nZi, ni):          z's Tucker cores against theta's
    typ.Tuple[NDArray, ...],        # G2.  len=d, elm_shape=C+(rZi, ni, rZ(i+1)): z's TT cores in theta's basis
    typ.Tuple[NDArray, ...],        # ZL.  len=d, elm_shape=C+(rZi, ri):          left environments
    typ.Tuple[NDArray, ...],        # ZR.  len=d, elm_shape=C+(rZ(i+1), r(i+1)):  right environments
    typ.Tuple[NDArray, ...],        # env. len=d, elm_shape=C+(ri, nZi, r(i+1)):  one-hole environments
]:
    """The ``(z, theta)``-dependent part of :py:func:`t3_inner_product_corewise_hessian`, computed once.

    ``BU_i = B_z,i U_i^T``; ``G2_i = G_z,i x_2 BU_i`` (``z``'s TT cores re-expressed in ``theta``'s Tucker
    basis, tails squashed); ``ZL`` / ``ZR`` the left-to-right / right-to-left zippers of ``G2`` against
    ``theta``'s TT chain (``ZL_i`` is everything left of core ``i``, ``ZR_i`` everything right of it); and
    ``env_i = ZL_i G_z,i ZR_i``, the one-hole environment of core ``i``. These are exactly the
    intermediates of the gradient :py:func:`t3_inner_product_corewise_transpose`
    (``dG_i = env_i x_2 BU_i``, ``dU_i = <env_i, G_i>^T B_z,i``); the Hessian differentiates each of them
    along a direction. In a Newton-CG inner solve ``z`` and ``theta`` are fixed while the direction varies,
    so this is the per-frame sweep and only the dotted terms are per matvec. Stack ``C``.

    **Polymorphic** over ragged core tuples and uniform supercores (``(d,)+C+(...)``: ``d``-prefixed grouped
    contractions, the zippers' uniform scan; each returned family is then one supercore). Mask-free by
    contract: a uniform caller masks ``z`` and ``theta`` first
    (:py:func:`~t3toolbox.backend.ut3_linalg.ut3_precompute_inner_product_sweep`).
    """
    z_tucker, z_tt = z
    z_tt = tt_operations.tt_squash_tails(z_tt)
    tucker_cores, tt_cores = core_pair
    is_uniform = is_ndarray(tucker_cores)          # uniform: supercores, d-prefixed contractions
    xnp, _, _ = get_backend(is_uniform, tree_contains_jax((z, core_pair)))
    d = len(tucker_cores)

    if is_uniform:
        BU = contractions.contract('dCzn,dCun->dCzu', z_tucker, tucker_cores)
        G2 = contractions.contract('dCazb,dCzu->dCaub', z_tt, BU)
    else:
        BU = tuple(contractions.contract('Czn,Cun->Czu', z_tucker[ii], tucker_cores[ii]) for ii in range(d))
        G2 = tuple(contractions.contract('Cazb,Czu->Caub', z_tt[ii], BU[ii]) for ii in range(d))
    if d == 1:      # no chain on either side: the boundary ones environments (as in the gradient)
        ss = G2[0].shape[:-3]
        ZL = (xnp.ones(ss + (G2[0].shape[-3], tt_cores[0].shape[-3])),)
        ZR = (xnp.ones(ss + (G2[0].shape[-1], tt_cores[0].shape[-1])),)
    else:
        ZL = tt_operations.tt_zipper_left_to_right(G2[:-1], tt_cores[:-1])
        ZR = tt_operations.tt_zipper_right_to_left(G2[1:], tt_cores[1:])
    if is_uniform:
        ZL, ZR = xnp.stack(ZL, axis=0), xnp.stack(ZR, axis=0)
        env = contractions.contract('dCxa,dCxzc,dCcb->dCazb', ZL, z_tt, ZR)
    else:
        env = tuple(contractions.contract('Cxa,Cxzc,Ccb->Cazb', ZL[ii], z_tt[ii], ZR[ii]) for ii in range(d))
    return BU, G2, ZL, ZR, env


def t3_inner_product_corewise_hessian(
        z:                  typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores of z. len=d, elm_shape=C+(nZi, Ni)
            typ.Sequence[NDArray],  # tt_cores of z.     len=d, elm_shape=C+(rZi, nZi, rZ(i+1))
        ],
        core_pair:          typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores of theta. len=d, elm_shape=C+(ni, Ni)
            typ.Sequence[NDArray],  # tt_cores of theta.     len=d, elm_shape=C+(ri, ni, r(i+1))
        ],
        core_directions:    typ.Tuple[
            typ.Sequence[NDArray],  # Tucker directions theta_hat. len=d, elm_shape=K+C+(ni, Ni)
            typ.Sequence[NDArray],  # TT directions theta_hat.     len=d, elm_shape=K+C+(ri, ni, r(i+1))
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],        # Tucker block of K_z theta_hat. len=d, elm_shape=K+C+(ni, Ni)
    typ.Tuple[NDArray, ...],        # TT block of K_z theta_hat.     len=d, elm_shape=K+C+(ri, ni, r(i+1))
]:
    """The core-coordinate Hessian of ``theta -> <z, Phi(theta)>`` applied to ``theta_hat``:

    ``K_z theta_hat = d/ds grad_theta <z, Phi(theta + s theta_hat)> |_{s=0}``,

    the derivative of :py:func:`t3_inner_product_corewise_transpose` along the directions. A **two-hole**
    sweep: every environment of the gradient gains its "exactly one core replaced by its direction"
    companion (the tangent zippers :py:func:`~t3toolbox.backend.tt_operations.tt_zipper_tangent_left_to_right`
    / ``_right_to_left``, and ``BU``, ``G2`` along the Tucker directions), and each output block takes the
    product rule. Exact (no finite differences), symmetric (``<a, K_z b> = <b, K_z a>``: it is a Hessian),
    linear in ``z``, and of the cost of a few gradients per direction.

    In the exact Riemannian Hessian it is the tangential-curvature term: with ``z = grad f`` (a
    doubled-rank T3) and ``theta_hat`` a lifted tangent, ``K_{grad f}`` is what turns the ambient
    second-order term ``<A^T r, D^2 Phi>`` into its normal part (the Weingarten map; Absil, Mahony &
    Trumpf (2013), "An extrinsic look at the Riemannian Hessian", via the second-fundamental-form identity
    ``<W(xi, N), eta> = <N, D^2 Phi[xi_hat, eta_hat]>``). This one-shot form computes the per-``(z, theta)``
    sweep itself; reuse it across directions with :py:func:`t3_precompute_inner_product_sweep` +
    :py:func:`t3_inner_product_corewise_hessian_from_sweep`.

    Batching: ``z`` and ``theta`` carry ``C``; the directions (and the result) ``K + C`` -- one
    Hessian-vector product per direction at the shared point.

    Examples
    --------
    The Hessian is the derivative of the gradient (a polynomial along the line, so a central difference
    agrees to ``O(h^2)``), and it is symmetric:

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.backend.t3_constructors as t3k
    >>> import t3toolbox.backend.t3_linalg as t3la
    >>> np.random.seed(0)
    >>> theta = t3k.t3_corewise_randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
    >>> z = t3k.t3_corewise_randn((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))
    >>> a, b = (t3k.t3_corewise_randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1)) for _ in range(2))
    >>> Ka = t3la.t3_inner_product_corewise_hessian(z, theta, a)
    >>> h = 1e-4
    >>> shift = lambda s: cw.corewise_add(theta, cw.corewise_scale(a, s))
    >>> fd = cw.corewise_scale(cw.corewise_sub(t3la.t3_inner_product_corewise_transpose(z, shift(h)),
    ...                                        t3la.t3_inner_product_corewise_transpose(z, shift(-h))), 0.5 / h)
    >>> print(bool(cw.corewise_norm(cw.corewise_sub(Ka, fd)) < 1e-6 * cw.corewise_norm(Ka)))
    True
    >>> Kb = t3la.t3_inner_product_corewise_hessian(z, theta, b)
    >>> print(bool(np.isclose(cw.corewise_dot(b, Ka), cw.corewise_dot(a, Kb))))
    True
    """
    sweep = t3_precompute_inner_product_sweep(z, core_pair)
    return t3_inner_product_corewise_hessian_from_sweep(z, core_pair, core_directions, sweep)


def t3_inner_product_corewise_hessian_from_sweep(
        z:                  typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores of z. len=d, elm_shape=C+(nZi, Ni)
            typ.Sequence[NDArray],  # tt_cores of z.     len=d, elm_shape=C+(rZi, nZi, rZ(i+1))
        ],
        core_pair:          typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores of theta. len=d, elm_shape=C+(ni, Ni)
            typ.Sequence[NDArray],  # tt_cores of theta.     len=d, elm_shape=C+(ri, ni, r(i+1))
        ],
        core_directions:    typ.Tuple[
            typ.Sequence[NDArray],  # Tucker directions theta_hat. len=d, elm_shape=K+C+(ni, Ni)
            typ.Sequence[NDArray],  # TT directions theta_hat.     len=d, elm_shape=K+C+(ri, ni, r(i+1))
        ],
        sweep:              typ.Tuple[typ.Tuple[NDArray, ...], ...],  # t3_precompute_inner_product_sweep(z, core_pair)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],        # Tucker block of K_z theta_hat. len=d, elm_shape=K+C+(ni, Ni)
    typ.Tuple[NDArray, ...],        # TT block of K_z theta_hat.     len=d, elm_shape=K+C+(ri, ni, r(i+1))
]:
    """:py:func:`t3_inner_product_corewise_hessian` with the ``(z, theta)`` sweep precomputed
    (:py:func:`t3_precompute_inner_product_sweep`): only the direction-dependent (dotted) terms run here.

    Per core ``i``, with dots the derivatives along ``theta_hat``:
    ``BUdot_i = B_z,i Udot_i^T``, ``G2dot_i = G_z,i x_2 BUdot_i``; the tangent zippers ``ZLdot``, ``ZRdot``
    of ``(G2, G)`` along ``(G2dot, Gdot)``; ``envdot_i = ZLdot_i G_z,i ZR_i + ZL_i G_z,i ZRdot_i``; then
    ``(K theta_hat)_{G_i} = envdot_i x_2 BU_i + env_i x_2 BUdot_i`` and
    ``(K theta_hat)_{U_i} = (<envdot_i, G_i> + <env_i, Gdot_i>)^T B_z,i``.

    **Polymorphic** like the sweep (uniform supercores in, supercores out, every core in one ``d``-prefixed
    contraction). Mask-free by contract: a second derivative multiplies a direction's padded slot into real
    slots, so a uniform caller masks ``z``, ``theta`` AND ``theta_hat`` first and re-masks the output
    (:py:func:`~t3toolbox.backend.ut3_linalg.ut3_inner_product_corewise_hessian_from_sweep`).
    """
    BU, G2, ZL, ZR, env = sweep
    z_tucker, z_tt = z
    z_tt = tt_operations.tt_squash_tails(z_tt)
    tucker_cores, tt_cores = core_pair
    tucker_dirs, tt_dirs = core_directions
    is_uniform = is_ndarray(tucker_cores)          # uniform: supercores, d-prefixed contractions
    xnp, _, _ = get_backend(is_uniform, tree_contains_jax((z, core_pair, core_directions)))
    d = len(tucker_cores)

    if is_uniform:
        BUd = contractions.contract('dCzn,dKCun->dKCzu', z_tucker, tucker_dirs)
        G2d = contractions.contract('dCazb,dKCzu->dKCaub', z_tt, BUd)
    else:
        BUd = tuple(contractions.contract('Czn,KCun->KCzu', z_tucker[ii], tucker_dirs[ii]) for ii in range(d))
        G2d = tuple(contractions.contract('Cazb,KCzu->KCaub', z_tt[ii], BUd[ii]) for ii in range(d))
    if d == 1:      # no chain: the boundary environments do not move
        ss = tucker_dirs[0].shape[:-2]
        ZLd = (xnp.zeros(ss + ZL[0].shape[-2:]),)
        ZRd = (xnp.zeros(ss + ZR[0].shape[-2:]),)
    else:
        ZLd = tt_operations.tt_zipper_tangent_left_to_right(G2[:-1], tt_cores[:-1], G2d[:-1], tt_dirs[:-1],
                                                            zippers=ZL)
        ZRd = tt_operations.tt_zipper_tangent_right_to_left(G2[1:], tt_cores[1:], G2d[1:], tt_dirs[1:],
                                                            zippers=ZR)

    if is_uniform:                                 # the same product rule, every core at once
        ZLd, ZRd = xnp.stack(ZLd, axis=0), xnp.stack(ZRd, axis=0)
        envd = (contractions.contract('dKCxa,dCxzc,dCcb->dKCazb', ZLd, z_tt, ZR)
                + contractions.contract('dCxa,dCxzc,dKCcb->dKCazb', ZL, z_tt, ZRd))
        tt_out = (contractions.contract('dKCazb,dCzu->dKCaub', envd, BU)
                  + contractions.contract('dCazb,dKCzu->dKCaub', env, BUd))
        Md = (contractions.contract('dKCazb,dCaub->dKCzu', envd, tt_cores)
              + contractions.contract('dCazb,dKCaub->dKCzu', env, tt_dirs))
        return contractions.contract('dKCzu,dCzn->dKCun', Md, z_tucker), tt_out

    tucker_out, tt_out = [], []
    for ii in range(d):
        envd = (contractions.contract('KCxa,Cxzc,Ccb->KCazb', ZLd[ii], z_tt[ii], ZR[ii])
                + contractions.contract('Cxa,Cxzc,KCcb->KCazb', ZL[ii], z_tt[ii], ZRd[ii]))
        tt_out.append(contractions.contract('KCazb,Czu->KCaub', envd, BU[ii])
                      + contractions.contract('Cazb,KCzu->KCaub', env[ii], BUd[ii]))
        Md = (contractions.contract('KCazb,Caub->KCzu', envd, tt_cores[ii])
              + contractions.contract('Cazb,KCaub->KCzu', env[ii], tt_dirs[ii]))
        tucker_out.append(contractions.contract('KCzu,Czn->KCun', Md, z_tucker[ii]))
    return tuple(tucker_out), tuple(tt_out)
