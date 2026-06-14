# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ
import math

from t3toolbox.backend.t3_operations import squash_tt_tails
import t3toolbox.backend.t3_orthogonalization as ragged_orth
import t3toolbox.backend.t3_operations as t3_ops
import t3toolbox.backend.t3_svd as ragged_t3svd
import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.ranks as ranks
from t3toolbox.backend.common import *

__all__ = [
    't3_add',
    't3_sum_stack',
    't3_scale',
    't3_inner_product_t3',
    't3_norm',
    't3_mult',
    't3m_form_then_round',
    't3m_inplace_fused',
    't3m_swap',
    't3_plus_scalar',
]


def t3_add(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
) -> typ.Tuple[typ.Tuple[NDArray], typ.Tuple[NDArray]]: # (x_plus_y_tucker_cores, x_plus_y_tt_cores)
    """Add Tucker tensor trains x and y, yielding a Tucker tensor train with summed ranks.
    """
    use_jax = (is_jax_ndarray(x) or is_jax_ndarray(y))
    xnp, xmap, _ = get_backend(False, use_jax)

    #
    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    assert(vsx == vsy)

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
        x:          typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        axis        = None, # stack axis, or sequence of stack axes, to sum over. None: sum over all stack axes
) -> typ.Tuple[
    typ.Tuple[NDArray,...], # summed_tucker_cores
    typ.Tuple[NDArray,...], # summed_tt_cores
]: # (summed_tucker_cores, summed_tt_cores)
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
    elif not isinstance(axis, typ.Sequence):
        summed_axes = [axis]
    else:
        summed_axes = list(axis)

    summed_axes = sorted(set((ax + m) if ax < 0 else ax for ax in summed_axes))
    for ax in summed_axes:
        assert(0 <= ax < m)

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

    summed_tt_cores = squash_tt_tails(tuple(summed_tt_cores)) # ones-contraction at the tails performs the sum
    return tuple(summed_tucker_cores), summed_tt_cores


def t3_scale(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        s,  # scalar
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]: # x*s
    """Multipy a Tucker tensor train by a scaling factor.
    """
    tucker_cores, tt_cores = x

    scaled_tucker_cores = [B.copy() for B in tucker_cores]
    scaled_tucker_cores[-1] = scaled_tucker_cores[-1] * s

    copied_tt_cores = [G.copy() for G in tt_cores]

    return tuple(scaled_tucker_cores), tuple(copied_tt_cores)


def t3_inner_product_t3(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        use_orthogonalization: bool = True, # for numerical stability
):
    """Compute Hilbert-Schmidt inner product of two Tucker tensor trains.
    """
    use_jax = any([is_jax_ndarray(c) for c in x[0] + x[1] + y[0] + y[1]])
    xnp, _, _ = get_backend(False, use_jax)

    #
    x = (x[0], squash_tt_tails(x[1]))
    y = (y[0], squash_tt_tails(y[1]))

    if use_orthogonalization:
        x = ragged_orth.left_orthogonalize_t3(x)
        y = ragged_orth.left_orthogonalize_t3(y)

    tucker_cores_x, tt_cores_x = x
    tucker_cores_y, tt_cores_y = y

    vsx = tucker_cores_x[0].shape[:-2] # vectorization shape for x
    vsy = tucker_cores_y[0].shape[:-2] # vectorization shape for y
    assert(vsx == vsy)

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
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        use_orthogonalization: bool = True, # for numerical stability
):
    """Compute Hilbert-Schmidt norm of a Tucker tensor train.
    """
    use_jax = any([is_jax_ndarray(B) for B in x[0]] + [is_jax_ndarray(G) for G in x[1]])
    xnp, _, _ = get_backend(False, use_jax)

    #
    x = (x[0], squash_tt_tails(x[1]))
    if use_orthogonalization:
        x = ragged_orth.left_orthogonalize_t3(x)
        Gf = x[1][-1].sum(axis=-1)
        norm_sq = (Gf*Gf).sum(axis=(-2,-1)) # Don't sum over stacked axes
    else:
        norm_sq = t3_inner_product_t3(x, x)

    return xnp.sqrt(xnp.abs(norm_sq))


def t3_mult(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
) -> typ.Tuple[typ.Tuple[NDArray], typ.Tuple[NDArray]]: # (x_times_y_tucker_cores, x_times_y_tt_cores)
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
    assert(vsx == vsy)

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
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
        max_tucker_ranks=None,  # int | Sequence[int] | None
        max_tt_ranks=None,      # int | Sequence[int] | None
        rtol=None,              # float | None  (requires unstacked; enforced by the frontend)
        atol=None,              # float | None
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # x_times_y tucker_cores
    typ.Tuple[NDArray, ...],  # x_times_y tt_cores
]:
    '''Elementwise product ``x ⊙ y`` -- method (a): form the full product, then round.

    Forms the full Khatri-Rao/Kronecker product (:py:func:`t3_mult`, multiplied ranks) and, if any
    truncation is requested, rounds it with :py:func:`t3svd`; with no truncation it returns the exact
    full product directly. The forming step is embarrassingly parallel (no sweep) but materializes the
    whole product -- see ``docs/t3m_plan.md`` for the cost trade-off vs the fused / swap methods.
    Stack-aware with max-rank truncation; ``rtol``/``atol`` require unstacked.
    '''
    product = t3_mult(x, y)
    if max_tucker_ranks is None and max_tt_ranks is None and rtol is None and atol is None:
        return product
    rounded, _, _ = ragged_t3svd.t3svd(
        product, max_tt_ranks=max_tt_ranks, max_tucker_ranks=max_tucker_ranks, rtol=rtol, atol=atol)
    return rounded


def t3m_inplace_fused(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
        max_tucker_ranks=None,  # int | Sequence[int] | None
        max_tt_ranks=None,      # int | Sequence[int] | None
        rtol=None,              # float | None  (requires unstacked; enforced by the frontend)
        atol=None,              # float | None
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # x_times_y tucker_cores
    typ.Tuple[NDArray, ...],  # x_times_y tt_cores
]:
    '''Elementwise product ``x ⊙ y`` -- method (b): a fused left-to-right sweep that truncates as it
    goes, never materializing the full product (the workhorse; see ``docs/t3m_plan.md``).

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

    # Right-orthogonalize each central TT (the product's central TT is then implicitly right-canonical).
    Gx = orth.right_orthogonalize_tt_cores(Gx)
    Gy = orth.right_orthogonalize_tt_cores(Gy)

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
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_x, tt_cores_x)
        y: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores_y, tt_cores_y)
        max_tucker_ranks=None,  # int | Sequence[int] | None
        max_tt_ranks=None,      # int | Sequence[int] | None
        rtol=None,              # float | None  (requires unstacked; enforced by the frontend)
        atol=None,              # float | None
        oversample=1,           # numeric >= 1; intermediate rank/tol relaxation factor (see docs/t3m_swap_plan.md)
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
    the cleanup (aggressive corner); ``k≈3`` recovers ``t3svd`` quality; a per-position
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

    # Leaves -> root (fold Tucker weight into the central TTs), then build/gauge the chain.
    Ux, Gx = ragged_orth.down_orthogonalize_tucker_cores((Ux, Gx))
    Uy, Gy = ragged_orth.down_orthogonalize_tucker_cores((Uy, Gy))
    Gx_o = orth.left_orthogonalize_tt_cores(list(Gx))
    Gy_o = orth.right_orthogonalize_tt_cores(t3_ops.reverse_tt(list(Gy)))
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

    need_cleanup = (oversample > 1) or (
        isinstance(max_tt_ranks, typ.Sequence) and not isinstance(max_tt_ranks, (int, np.integer)))
    if need_cleanup:
        rounded, _, _ = ragged_t3svd.t3svd(
            (tucker_cores, tt_cores),
            max_tt_ranks=max_tt_ranks, max_tucker_ranks=max_tucker_ranks, rtol=rtol, atol=atol)
        return rounded
    return tucker_cores, tt_cores


def t3_plus_scalar(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],
        s,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:
    use_jax = is_jax_ndarray(x)

    x_shape = tuple(B.shape[-1] for B in x[0])
    x_stack_shape = x[0][0].shape[:-2]

    y0 = t3_ops.t3_ones(x_shape, x_stack_shape)
    y = t3_scale(y0, s)
    xs = t3_add(x, y)
    return xs








