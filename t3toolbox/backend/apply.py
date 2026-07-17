# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The ``apply`` sampling type: contract a T3 with vectors in ALL modes -> one scalar per sample.

Holds the t3 + tv ops, the ambient/tangent/corewise transposes, and the frame sweeps for applies.
Specializes the general probing machinery (containment probe ⊃ apply ⊃ entries; this module
imports from ``probing``, never the reverse). Costs and the role in Riemannian least-squares:
``docs/entries_apply_probe.md``.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.contractions as contractions
from t3toolbox.backend.common import *
import math
import t3toolbox.backend.probing as probing
from t3toolbox.backend.probing import compute_xi, compute_mu, compute_dxi, compute_sigma_hat, _sigma_step

__all__ = [
    't3_apply',
    't3_apply_ambient_transpose',
    'tv_apply',
    'tv_apply_transpose',
    'tv_precompute_apply_frame_sweep',
    'tv_apply_jacobian_from_sweep',
    'tv_apply_transpose_from_sweep',
    't3_apply_corewise_transpose',
]

def t3_apply(
        x: typ.Union[
            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
            typ.Tuple[NDArray, NDArray], # (tucker_supercore, tt_supercore)
        ],
        vecs: typ.Union[
            typ.Sequence[NDArray],  # len=d, elm_shape=vsw+(Ni,), ragged
            NDArray, # shape=(d,) + vsw +(Ni,), uniform
        ],
) -> NDArray:
    '''Contract a Tucker tensor train with vectors in all indices.
    '''
    use_jax = tree_contains_jax((x, vecs))
    is_uniform = is_ndarray(x[0])  # supercore -> real lax.scan over the mode axis (like entries)
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    #
    tucker_cores, tt_cores = x

    #

    vsc = tucker_cores[0].shape[:-2] # core/frame stack C (the batch of T3s)
    vsw = vecs[0].shape[:-1]         # vec stack W (the probe-like vectors), base-inner: W outer, C inner

    def _func(mu_WCa, v_B_G):
        v_Wo, B_Cpo, G_Capb = v_B_G
        mu_WCb = contractions.contract('WCa,Caib,Wo,Cio->WCb', 
            mu_WCa, G_Capb, v_Wo, B_Cpo,
        )
        return mu_WCb, (0,)

    mu_WCa = xnp.ones(vsw + vsc + (tt_cores[0].shape[-3],))   # W + C
    v_B_G = (vecs, tucker_cores, tt_cores)
    mu_WCz, _ = xscan(_func, mu_WCa, v_B_G)

    result = xnp.sum(mu_WCz, axis=-1)
    return result


def t3_apply_ambient_transpose(
        c:                  NDArray,                # residual, shape=W+C
        ww:                 typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)
        sum_over_probes:    bool = False,           # True: W becomes the CP rank (ambient J^T r)
) -> typ.Sequence[NDArray]:  # canonical (CP) factors. len=d, ith elm_shape=stack_shape+(R, Ni)
    '''Ambient transpose of :py:func:`t3_apply`: back-project ``c`` into CP factors.

    The *ambient* adjoint -- the transpose of ``apply`` as a linear map on the **full tensor space**
    (``X -> ( <X, w0^W (x) ... (x) w_{d-1}^W> )_W``). Frame-free; the back-projection
    ``c * (w0 (x) ... (x) w_{d-1})`` is rank-1, whose natural representation is a **canonical (CP)
    decomposition** (``apply`` consumes one vector per mode; its adjoint emits one scaled vector per
    mode). This is distinct from the *corewise* transpose (gradient w.r.t. a base point's cores) and
    the *tangent* transpose (Riemannian gradient) -- see ``docs/transposes.md`` for the full taxonomy.

    - ``sum_over_probes=False`` (primary): ``W`` is a passthrough stacking axis -- a ``W (+ C)`` stack
      of rank-1 CP tensors (CP rank ``R=1``).
    - ``sum_over_probes=True``: ``W`` becomes the CP **rank** -- one rank-``|W|`` CP tensor
      ``sum_W c_W * (w0^W (x) ...)`` (the ambient ``J^T r``). Cheap as CP (``O(d |W| N)``, the shared
      rank index stays implicit); the ``|W|^2`` cost of a *dense* Tucker tensor train is incurred only
      if you convert with ``t3_conversions.t3_from_canonical``.

    Returns the CP ``factors`` (``c`` folded into the first), in the layout
    ``t3_conversions.t3_from_canonical`` consumes.
    '''
    use_jax = tree_contains_jax((c, ww))
    xnp, _, _ = get_backend(False, use_jax)
    c = xnp.asarray(c)

    nW = ww[0].ndim - 1     # probe stack rank (ww[i] is W + (Ni,))
    W  = ww[0].shape[:nW]   # probe stack
    C  = c.shape[nW:]       # frame stack (c is W + C)
    nC = len(C)

    if sum_over_probes:
        # canonical rank |W|, stack C: w_i flattened over W into the rank axis, broadcast over C;
        # c folded into F_0 as  F_0[C, s, n] = c_flat[s, C] * w0_flat[s, n].
        m = int(np.prod(W, dtype=int))                       # |W|
        c_flat = xnp.moveaxis(c.reshape((m,) + C), 0, nC)    # (m,) + C  ->  C + (m,)
        factors = []
        for i, w in enumerate(ww):
            w_flat = w.reshape((1,) * nC + (m, w.shape[-1]))  # broadcastable to C + (m, Ni)
            if i == 0:
                factors.append(c_flat[..., None] * w_flat)               # C + (m, N0)
            else:
                factors.append(w_flat * xnp.ones(C + (1, 1)))            # materialize C + (m, Ni)
    else:
        # canonical rank 1, stack W + C: c folded into F_0 as  F_0[W, C, 0, n] = c[W, C] * w0[W, n].
        c_exp = c.reshape(W + C + (1, 1))                    # W + C + (1, 1)
        factors = []
        for i, w in enumerate(ww):
            w_exp = w.reshape(W + (1,) * nC + (1, w.shape[-1]))  # broadcastable to W + C + (1, Ni)
            if i == 0:
                factors.append(c_exp * w_exp)                            # W + C + (1, N0)
            else:
                factors.append(w_exp * xnp.ones(C + (1, 1)))             # materialize W + C + (1, Ni)

    return tuple(factors)


def tv_apply(
        ww:         typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],
        frame:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # frame order = T3Frame.data = (up, down, left, right)
) -> NDArray:                               # the scalar apply(v, ww), one per stack element; shape = W + K + C
    '''Apply a tangent vector in all modes: contract the dense tangent with ``ww`` in every index.

    The all-modes special case of probing -- a single left-to-right pass (mu-hat via P, then the
    perturbation sigma via Q), contracted at the terminal bond. No right (nu) / central (eta) sweeps,
    no per-mode assembly. See Section 6.2.2 (Algorithms 6-7) of Alger et al. (2026).

    See Also
    --------
    tv_entries
    tv_probe
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    var_tucker_cores, var_tt_cores = variation

    xis  = compute_xi(up_tucker_cores, ww)       # xi-hat_i  = U_i^T w_i
    dxis = compute_dxi(var_tucker_cores, ww)     # delta-xi_i = dU_i^T w_i
    mus  = compute_mu(left_tt_cores, xis)        # frame left sweep via P

    return _apply_from_xis(xis, dxis, mus, right_tt_cores, down_tt_cores, var_tt_cores)


def tv_apply_transpose(
        c:          NDArray,                # residual, shape = W + C (one per probe-set, per base point)
        ww:         typ.Sequence[NDArray],  # the apply vectors, len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray],
                              typ.Sequence[NDArray], typ.Sequence[NDArray]],   # (up, down, left, right)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (dU_tildes, dG_tildes) = T3Variations.data
    '''Apply the transpose of :py:func:`tv_apply` -- back-project a residual ``c`` into a tangent.

    The adjoint of the (linear-in-the-variation) all-modes apply. Needs only the frame sweep
    (xi-hat, mu-hat, nu-hat, eta-hat) and a single-term scatter assembly (it skips the adjoint
    perturbation sweep that tv_probe_transpose runs). With ``sum_over_probes=False`` the probe
    stack W becomes the output tangent stack; with ``True`` it is summed (the ``J^T r`` back-projection).

    See Also
    --------
    tv_apply
    tv_entries_transpose
    '''
    frame_sweep = tv_precompute_apply_frame_sweep(frame, ww)
    return tv_apply_transpose_from_sweep(c, ww, frame, frame_sweep, sum_over_probes)


def tv_precompute_apply_frame_sweep(
        frame:   typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # frame order = T3Frame.data = (up, down, left, right)
        ww:     typ.Sequence[NDArray],      # apply vectors, len=d, elm_shape=W+(Ni,)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xis. len=d, elm_shape=W+C+(nUi,)
    typ.Sequence[NDArray],  # mus. len=d, elm_shape=W+C+(rLi,)
]:                                          # lean frame sweep -- (xis, mus) only
    '''The all-modes apply **frame sweep** (lean): the frame edge variables ``(xi-hat, mu-hat)`` that
    depend only on the frame frame and the apply vectors ``ww`` -- NOT on the tangent or residual.
    Computing them is the expensive, ``W``-scaled part of the apply Jacobian; precomputed **once per
    frame** and reused across every ``J`` / ``Jᵀ`` of an inner solve (the reuse hook for ``fitting.py``).

    **Lean ``(xis, mus)`` only** (not the right ``nu`` / down ``eta`` sweeps): the all-modes apply
    forward AND its adjoint-state transpose use only ``(xi, mu)`` -- the transpose recomputes the right
    context as ``sigma_hat`` from the residual rather than storing ``nu``/``eta``, halving the
    ``W``-scaling memory (apply on the manifold, §6.2.2 of Alger et al. (2026)). Probe, which leaves a
    mode free, needs the full sweep -- :py:func:`tv_precompute_probe_frame_sweep`.

    See Also
    --------
    tv_apply_jacobian_from_sweep
    tv_apply_transpose_from_sweep
    tv_precompute_probe_frame_sweep
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    xis = compute_xi(up_tucker_cores, ww)
    mus = compute_mu(left_tt_cores, xis)
    return xis, mus


def tv_apply_jacobian_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],
        ww:         typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,) -- for the variation's dxis
        frame:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d  (unused; for a uniform call signature)
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d  (unused)
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # frame order = T3Frame.data = (up, down, left, right)
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray],          # xis
            typ.Sequence[NDArray],          # mus
        ],                                  # = tv_precompute_apply_frame_sweep(frame, ww)  (lean)
) -> NDArray:                               # the scalar apply(v, ww), one per stack element; shape = W + K + C
    '''Forward all-modes apply of a tangent vector reusing a precomputed frame sweep -- the bare ``𝒥`` with
    the frame edge variables injected. Equivalent to :py:func:`tv_apply`, but it takes the lean
    ``(xis, mus)`` from ``frame_sweep`` instead of recomputing them (the reuse hook for ``fitting.py``).
    Only the variation-dependent ``dxis`` is computed here. No gauge projector ``Π``.

    See Also
    --------
    tv_precompute_apply_frame_sweep
    tv_apply
    '''
    var_tucker_cores, var_tt_cores = variation
    _, down_tt_cores, _, right_tt_cores = frame
    xis, mus = frame_sweep
    dxis = compute_dxi(var_tucker_cores, ww)     # variation-dependent; not part of the frame sweep
    return _apply_from_xis(xis, dxis, mus, right_tt_cores, down_tt_cores, var_tt_cores)


def tv_apply_transpose_from_sweep(
        c:          NDArray,                # residual, shape = W + K + C (K optional)
        ww:         typ.Sequence[NDArray],  # apply vectors (one-hot e_index for entries), len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d  (unused; uniform call signature)
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d  (unused)
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # frame order = T3Frame.data = (up, down, left, right)
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray],          # xis
            typ.Sequence[NDArray],          # mus
        ],                                  # = tv_precompute_apply_frame_sweep(frame, ww)  (lean: no nu/eta)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (dU_tildes, dG_tildes) = T3Variations.data
    '''Transpose of the all-modes apply reusing a precomputed frame sweep -- the bare ``𝒥ᵀ``, by the
    **adjoint-state** method (the scalar residual ``c`` seeds one reverse ``sigma_hat`` sweep; see
    :py:func:`_apply_transpose_adjoint`). Takes the **lean** ``(xis, mus)`` sweep + the frame cores ``O,
    Q`` (it recomputes the right context rather than storing ``nu``/``eta`` -- half the memory). Reuse
    hook for ``fitting.py`` (one frame sweep feeds the forward and this transpose). Full ``W + K + C``
    (the residual ``c`` may carry the tangent stack ``K``). No gauge projector ``Π``.

    See Also
    --------
    tv_precompute_apply_frame_sweep
    tv_apply_transpose
    '''
    _, down_tt_cores, _, right_tt_cores = frame
    xis, mus = frame_sweep
    return _apply_transpose_adjoint(c, ww, xis, mus, down_tt_cores, right_tt_cores, sum_over_probes)


def t3_apply_corewise_transpose(
        c:          NDArray,                # residual, shape=W+C
        ww:         typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)
        core_pair:  typ.Tuple[
            typ.Sequence[NDArray],          # tucker_cores, len=d, elm_shape=C+(ni,Ni)
            typ.Sequence[NDArray],          # tt_cores,     len=d, elm_shape=C+(ri,ni,r(i+1))
        ],
        sum_over_probes: bool = False,      # True: sum the apply stack W (the gradient J^T r)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker-core gradients, same shapes as tucker_cores
    typ.Tuple[NDArray, ...],  # tt-core gradients,     same shapes as tt_cores
]:
    '''Corewise (non-manifold) transpose of :py:func:`t3_apply`: gradient of the
    measurement w.r.t. the cores of the frame ``core_pair``, treated as independent variables.

    The adjoint of the *core parametrization* ``cores -> apply(X(cores), ww)`` at the base point -- the
    gradient a core-wise optimizer (Adam, L-BFGS) needs. Returns gradients shaped exactly like
    ``(tucker_cores, tt_cores)`` -- a gradient, NOT a tensor (so no ``|W|`` blow-up: the apply stack
    collapses into the fixed-size cores). Distinct from the *ambient* transpose (a free CP tensor) and
    the *tangent* transpose (a Riemannian tangent); see ``docs/transposes.md``.

    Implemented by the Section 6.3 ("corewise simplification") substitution into the tangent transpose:
    feed the frame's own cores in place of the orthogonal frames (``P, Q, O -> G_i``), with ``U_i`` no
    longer required orthogonal -- i.e. :py:func:`tv_apply_transpose` at frame ``(U, G, G, G)``. No
    orthogonality is required. ``sum_over_probes=True`` sums the apply stack ``W`` (the gradient
    ``J^T r``); ``False`` keeps ``W`` as a stack (one core-gradient set per probe).

    Math reference: Section 6.3, Alger et al. (2026), "Tucker Tensor Train Taylor Series"
    (arXiv:2603.21141).
    '''
    tucker_cores, tt_cores = core_pair
    return tv_apply_transpose(
        c, ww, (tucker_cores, tt_cores, tt_cores, tt_cores), sum_over_probes=sum_over_probes,
    )


def _apply_from_xis(xis, dxis, mus, right_tt_cores, down_tt_cores, var_tt_cores):
    '''Run the perturbation sigma sweep (via Q) to its TERMINAL carry and contract the final bond.

    Shared tail of tv_apply and tv_entries (they differ only in how xis/dxis are formed).
    '''
    use_jax = tree_contains_jax((xis, dxis, mus, right_tt_cores))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    def _func(sigma, x):
        Q, O, dG, xi, dxi, mu = x
        return _sigma_step(sigma, Q, O, dG, xi, dxi, mu), (0,)

    # carry sigma is W+K+C; take the leading stack from dxis (carries K), not xis (W+C only).
    rR0 = right_tt_cores[0].shape[-3]
    init = xnp.zeros(dxis[0].shape[:-1] + (rR0,))
    sigma_terminal, _ = xscan(_func, init, (right_tt_cores, down_tt_cores, var_tt_cores, xis, dxis, mus))
    return xnp.sum(sigma_terminal, axis=-1)   # contract the terminal bond -> W + K + C


def _apply_transpose_adjoint(c, ww, xis, mus, down_tt_cores, right_tt_cores, sum_over_probes):
    '''Adjoint-state assembly shared by apply/tv_entries_transpose_from_sweep -- the **low-memory,
    K-aware** transpose (replaces the old scatter). The scalar residual ``c`` seeds one reverse
    ``sigma_hat`` sweep (recomputing the right context, so ``nu``/``eta`` are never stored); then

        dxi-hat_k  = mu-hat_{k-1} . O_k . sigma-hat_k                 # over the down mode nO
        dG-tilde_k = mu-hat_{k-1} (x) xi-hat_k (x) sigma-hat_k        # over (rL, nU, rR)
        dU-tilde_k = dxi-hat_k (x) w_k                               # over (nO, N)

    Uses only the lean frame sweep ``(xis, mus)`` (half the memory of the ``(xis, mus, nus, etas)`` the
    scatter stored -- the ``W``-scaling ``nu``/``eta`` are gone), at the cost of the ``sigma_hat`` sweep
    per transpose. Full ``W + K + C``: the residual ``c`` may carry the tangent stack ``K`` (the output
    space of a ``K``-stacked forward ``tv_apply``), which rides into the variation gradient -- the
    capability the scatter lacked. ``sum_over_probes=True`` sums the probe stack ``W`` (the ``J^T r``
    back-projection); ``False`` keeps it as the output tangent stack. ``K``/``C`` always kept.'''
    is_uniform = is_ndarray(mus)
    sigma_hats = compute_sigma_hat(right_tt_cores, xis, c)               # polymorphic reverse sweep

    if is_uniform:
        # d-prefixed WKC (3b-6a/c), vectorized over the core index d (NOT a per-core loop -> no jit
        # unroll). ww is the packed apply/one-hot probe supercore (d,)+W+(N,) -> n_probe = len(W). The
        # ragged loop below is the oracle.
        n_probe = ww.ndim - 2
        dxi_hats = contractions.contract('dWCa,dCaib,dWKCb->dWKCi', mus, down_tt_cores, sigma_hats)
        if sum_over_probes:
            dG_tildes = contractions.contract('dWCa,dWCi,dWKCb->dKCaib', mus, xis, sigma_hats, len_W=n_probe)
            dU_tildes = contractions.contract('dWo,dWKCa->dKCao', ww, dxi_hats)
        else:
            dG_tildes = contractions.contract('dWCa,dWCi,dWKCb->dWKCaib', mus, xis, sigma_hats, len_W=n_probe)
            dU_tildes = contractions.contract('dWo,dWKCa->dWKCao', ww, dxi_hats)
        return dU_tildes, dG_tildes

    n_probe = ww[0].ndim - 1
    dxi_hats = tuple(contractions.contract('WCa,Caib,WKCb->WKCi', mu, O, sh)        # dxi_hat = mu . O . sigma_hat
                     for mu, O, sh in zip(mus, down_tt_cores, sigma_hats))
    if sum_over_probes:
        dG_tildes = tuple(contractions.contract('WCa,WCi,WKCb->KCaib', mu, xi, sh, len_W=n_probe)
                          for mu, xi, sh in zip(mus, xis, sigma_hats))
        dU_tildes = tuple(contractions.contract('Wo,WKCa->KCao', w, dxh) for w, dxh in zip(ww, dxi_hats))
    else:
        dG_tildes = tuple(contractions.contract('WCa,WCi,WKCb->WKCaib', mu, xi, sh, len_W=n_probe)
                          for mu, xi, sh in zip(mus, xis, sigma_hats))
        dU_tildes = tuple(contractions.contract('Wo,WKCa->WKCao', w, dxh) for w, dxh in zip(ww, dxi_hats))
    return dU_tildes, dG_tildes   # (var_tucker, var_tt) = T3Variations.data
