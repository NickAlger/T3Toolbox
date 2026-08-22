# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The ``entries`` sampling type: evaluate individual T3 entries (one-hot probes) -> scalars.

Holds the t3 + tv ops, the ambient/tangent/corewise transposes, and the frame sweeps for entry
sampling. The most-special case of probe ⊃ apply ⊃ entries; imports the general machinery from
``probing``, never the reverse. See ``docs/entries_apply_probe.md``.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.apply as apply
from t3toolbox.backend.common import *
import math
import t3toolbox.backend.probing as probing
from t3toolbox.backend.probing import compute_mu
from t3toolbox.backend.apply import _apply_from_xis, _apply_transpose_adjoint

__all__ = [
    't3_entries',
    't3_entries_ambient_transpose',
    'tv_entries',
    'tv_entries_transpose',
    'tv_precompute_entries_frame_sweep',
    'tv_entries_jacobian_from_sweep',
    'tv_entries_transpose_from_sweep',
    't3_entries_corewise_transpose',
]


def _t3_entries_step(
        mu_WCa:  NDArray,                                # carry: W+C+(rLi,)
        ind_B_G: typ.Tuple[NDArray, NDArray, NDArray],   # (ind, B, G): int W ; C+(nUi,Ni) ; C+(rLi,nUi,rL(i+1))
) -> typ.Tuple[NDArray, typ.Tuple[int]]:   # (next carry, (0,) -- nothing emitted per mode)
    '''One mode of the all-modes fiber contraction of :py:func:`t3_entries`. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((mu_WCa, ind_B_G)))   # only xnp; it ignores the flag
    ind, B_Cpo, G_Capb = ind_B_G
    n_idx = ind.ndim                                           # this mode's index carries exactly the W axes
    xi_CpW = B_Cpo[..., ind]                                   # C + (p,) + W (index batch trails)
    xi_WCp = xnp.moveaxis(                                     # -> W + C + (p,) = WCi
        xi_CpW, tuple(range(-n_idx, 0)), tuple(range(n_idx)),
    )

    mu_WCb = contractions.contract('WCa,Caib,WCi->WCb', 
        mu_WCa, G_Capb, xi_WCp,
    )

    return mu_WCb, (0,)


def t3_entries(
        x: typ.Union[
            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
            typ.Tuple[NDArray, NDArray], # (tucker_supercore, tt_supercore)
        ],
        index: NDArray, # dtype=int, shape=(d,)+vsw. (or convertible to int array of this shape)
) -> NDArray: # shape=vsw+vsc (W + C, base-inner)
    '''Compute entries of a Tucker tensor train.
    '''
    use_jax = tree_contains_jax((x, index))
    is_uniform = is_ndarray(x[0])
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    #
    index = xnp.array(index)

    tucker_cores, tt_cores = x
    vsc = x[0][0].shape[:-2]
    index = xnp.array(index)

    vsw = index.shape[1:]    # index stack W (base-inner: W outer, C inner)

    mu_WCa = xnp.ones(vsw + vsc + (tt_cores[0].shape[-3],))   # W + C
    ind_B_G = (index, tucker_cores, tt_cores)
    mu_WCz, _ = xscan(_t3_entries_step, mu_WCa, ind_B_G)

    result = xnp.sum(mu_WCz, axis=-1)
    return result


def t3_entries_ambient_transpose(
        c:                  NDArray,                # residual, shape=W+C
        index:              NDArray,                # int, shape=(d,)+W
        shape:              typ.Sequence[int],      # ambient dims (N0,...,N(d-1)) -- to size the one-hots
        sum_over_probes:    bool = False,           # True: W becomes the CP rank (scatter-adds collisions)
) -> typ.Sequence[NDArray]:  # canonical (CP) factors. len=d, ith elm_shape=stack_shape+(R, Ni)
    '''Ambient transpose of :py:func:`t3_entries`: scatter ``c`` at ``index`` into CP factors.

    The ``entries`` counterpart of :py:func:`t3_apply_ambient_transpose` -- identical
    with the apply vectors replaced by the unit vectors ``e_{index_k}``, so the CP factors are one-hots
    and the back-projection is ``c * e_{idx_0} (x) ... (x) e_{idx_{d-1}}``. ``sum_over_probes=True``
    makes ``W`` the CP rank (scatter-adding colliding indices -- the ``J^T r`` for entry sampling).
    ``shape`` supplies the ambient dims, which (unlike the apply case, where ``ww`` carries them) the
    residual and index alone do not determine. Returns CP factors (see the apply version).
    '''
    use_jax = tree_contains_jax((c, index))
    xnp, _, _ = get_backend(False, use_jax)
    index = xnp.array(index)
    # one-hot CP factors e_{index_i}, elm_shape = W + (Ni,). eye(N)[index] follows numpy index semantics
    # (a negative index wraps), exactly like the forward t3_entries and the tangent/corewise transposes
    # (_onehot_vectors); the former `arange(N) == index` scatter matched nothing for a negative index and
    # silently returned zero factors.
    ww = tuple(xnp.eye(N)[index[i]] for i, N in enumerate(shape))
    return apply.t3_apply_ambient_transpose(c, ww, sum_over_probes=sum_over_probes)


def tv_entries(
        index:      NDArray,                # int, shape=(d,)+W (index stack W)
        variation:  typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (var_tucker, var_tt)
        frame:       typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray],
                              typ.Sequence[NDArray], typ.Sequence[NDArray]],   # (up, down, left, right)
) -> NDArray:                               # entries of the dense tangent at ``index``; shape = W + K + C
    '''Extract entries of the dense tangent at ``index`` (= apply with unit vectors, by slicing).

    Identical to :py:func:`tv_apply` except the up-index edge variables come from slicing the
    Tucker-core fibers (``U_i[..., index_i]`` and ``dU_i[..., index_i]``) rather than contracting with
    vectors -- so there is no contraction with unit basis vectors and no ``N`` factor.

    See Also
    --------
    tv_apply
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    var_tucker_cores, var_tt_cores = variation

    xis  = _entry_xis(up_tucker_cores, index)     # xi-hat_i  = U_i[..., index_i]   (fiber slice)
    dxis = _entry_xis(var_tucker_cores, index)    # delta-xi_i = dU_i[..., index_i] (fiber slice)
    mus  = compute_mu(left_tt_cores, xis)

    return _apply_from_xis(xis, dxis, mus, right_tt_cores, down_tt_cores, var_tt_cores)


def tv_entries_transpose(
        c:          NDArray,                # residual, shape = W + C
        index:      NDArray,                # int, shape=(d,)+W (the indices whose entries c weights)
        frame:       typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray],
                              typ.Sequence[NDArray], typ.Sequence[NDArray]],   # (up, down, left, right)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (dU_tildes, dG_tildes)
    '''Apply the transpose of :py:func:`tv_entries` -- scatter a residual ``c`` at ``index`` into a tangent.

    Identical to :py:func:`tv_apply_transpose` with the up-index ``xi-hat`` from fiber slicing and
    the apply vectors replaced by the unit vectors ``e_{index_k}`` (so the ``dU-tilde`` outer product
    *is* the entry scatter).

    See Also
    --------
    tv_entries
    tv_apply_transpose
    '''
    frame_sweep = tv_precompute_entries_frame_sweep(frame, index)
    return tv_entries_transpose_from_sweep(c, index, frame, frame_sweep, sum_over_probes)


def tv_precompute_entries_frame_sweep(
        frame:   typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # frame order = T3Frame.data = (up, down, left, right)
        index:  NDArray,                    # int, shape=(d,)+W -- the grid points
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xis. len=d, elm_shape=W+C+(nUi,) -- the FIBER-SLICED seed (not contracted)
    typ.Sequence[NDArray],  # mus. len=d, elm_shape=W+C+(rLi,)
]:                                          # lean frame sweep -- (xis, mus) only (entries seed)
    '''The all-modes **entries** frame sweep (lean): identical to :py:func:`tv_precompute_apply_frame_sweep`
    but the ``xi-hat`` seed comes from slicing the Tucker-core fibers at ``index`` (``_entry_xis``)
    instead of contracting with probe vectors. Like apply, entries uses the adjoint-state transpose, so
    only ``(xis, mus)`` are needed (no ``nu``/``eta``). Reused by the entries forward/transpose (the
    reuse hook for ``fitting.py``).

    See Also
    --------
    tv_precompute_apply_frame_sweep
    tv_entries_jacobian_from_sweep
    tv_entries_transpose_from_sweep
    '''
    up_tucker_cores, _, left_tt_cores, _ = frame
    xis = _entry_xis(up_tucker_cores, index)
    mus = compute_mu(left_tt_cores, xis)
    return xis, mus


def tv_entries_jacobian_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],
        index:      NDArray,                # int, shape=(d,)+W -- for the variation's fiber-sliced dxis
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q); uses Q (right) and O (down)
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = tv_precompute_entries_frame_sweep(frame, index)  (lean)
) -> NDArray:                               # entries of the dense tangent at ``index``; shape = W + K + C
    '''Forward all-modes entries of a tangent vector reusing a precomputed frame sweep -- the bare ``𝒥``
    (entries) with the frame edge variables injected. Equivalent to :py:func:`tv_entries`, but takes
    the lean ``(xis, mus)`` from ``frame_sweep``; only the fiber-sliced ``dxis`` is computed here. No
    gauge projector ``Π``.'''
    var_tucker_cores, var_tt_cores = variation
    _, down_tt_cores, _, right_tt_cores = frame
    xis, mus = frame_sweep
    dxis = _entry_xis(var_tucker_cores, index)    # fiber slice; variation-dependent, not in the frame sweep
    return _apply_from_xis(xis, dxis, mus, right_tt_cores, down_tt_cores, var_tt_cores)


def tv_entries_transpose_from_sweep(
        c:          NDArray,                # residual, shape = W + C (or W + K + C)
        index:      NDArray,                # int, shape=(d,)+W -- the indices c weights (-> one-hot vectors)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q); uses U (one-hot), O, Q
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = tv_precompute_entries_frame_sweep(frame, index)  (lean: no nu/eta)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (dU_tildes, dG_tildes) = T3Variations.data
    '''Transpose of the all-modes entries reusing a precomputed frame sweep -- the bare ``𝒥ᵀ`` (entries),
    by the **adjoint-state** method (see :py:func:`_apply_transpose_adjoint`). Takes the **lean**
    ``(xis, mus)`` sweep (the reuse hook for ``fitting.py``). Identical to
    :py:func:`tv_apply_transpose_from_sweep` with the one-hot vectors ``e_{index}`` as the apply vectors.
    Full ``W + K + C``; no gauge projector ``Π``.'''
    up_tucker_cores, down_tt_cores, _, right_tt_cores = frame
    ww = _onehot_vectors(index, up_tucker_cores)
    xis, mus = frame_sweep
    return _apply_transpose_adjoint(c, ww, xis, mus, down_tt_cores, right_tt_cores, sum_over_probes)


def t3_entries_corewise_transpose(
        c:          NDArray,                # residual, shape=W+C
        index:      NDArray,                # int, shape=(d,)+W
        core_pair:  typ.Tuple[
            typ.Sequence[NDArray],          # tucker_cores, len=d, elm_shape=C+(ni,Ni)
            typ.Sequence[NDArray],          # tt_cores,     len=d, elm_shape=C+(ri,ni,r(i+1))
        ],
        sum_over_probes: bool = False,      # True: sum the apply stack W (scatter-adds collisions)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker-core gradients, same shapes as tucker_cores
    typ.Tuple[NDArray, ...],  # tt-core gradients,     same shapes as tt_cores
]:
    '''Corewise (non-manifold) transpose of :py:func:`t3_entries`: gradient of the
    sampled entries w.r.t. the frame's cores.

    The ``entries`` counterpart of :py:func:`t3_apply_corewise_transpose` -- the Section 6.3 substitution
    into :py:func:`tv_entries_transpose`. Needs no ambient ``shape`` argument: the dims come from
    the frame ``tucker_cores``. ``sum_over_probes=True`` scatter-adds colliding indices (the gradient
    ``J^T r``).
    '''
    return tv_entries_transpose(
        c, index, fv_conversions.t3_corewise_frame(core_pair), sum_over_probes=sum_over_probes,
    )


def _entry_xis(tucker_cores, index):
    '''Up-index edge variables by slicing Tucker-core fibers at ``index`` (no contraction).

    tucker_cores[i].shape = C + (p_i, Ni); index is an int array of shape (d,) + W (index stack W).
    Returns xis with elm_shape = W + C + (p_i,) -- the same layout compute_xi produces, so the
    downstream mu/sigma sweeps are identical to apply.
    '''
    use_jax = tree_contains_jax((tucker_cores,))
    is_uniform = is_ndarray(tucker_cores)
    xnp, _, _ = get_backend(is_uniform, use_jax)
    index = xnp.array(index)

    if is_uniform:
        # Vectorized fiber slice over the core index d: a single advanced-indexing gather, NOT a per-core
        # Python loop (which would unroll under jit) and NOT a one-hot contraction (which would add the N
        # factor entries exists to avoid). The supercore is (d,)+C+(p,N); advanced indices on axis 0 (d)
        # and axis -1 (N) broadcast to (d,)+W, the C/p slice rides between -> (d,)+W+C+(p,), matching the
        # ragged W+C+(p,) layout. index_i < Ni <= N stays in the real region.
        D = tucker_cores.shape[0]
        n_W = index.ndim - 1
        d_idx = xnp.arange(D).reshape((D,) + (1,) * n_W)          # (d,) + (1,)*W, broadcasts with index
        return tucker_cores[d_idx, ..., index]                   # (d,) + W + C + (p,)

    n_idx = len(index.shape[1:])                                  # number of index-stack (W) axes
    xis = []
    for i, B in enumerate(tucker_cores):
        xi_CpW = B[..., index[i]]                                 # C + (p,) + W (index batch trails)
        xi_WCp = xnp.moveaxis(xi_CpW, tuple(range(-n_idx, 0)), tuple(range(n_idx)))  # -> W + C + (p,)
        xis.append(xi_WCp)
    return tuple(xis)


def _onehot_vectors(index, up_tucker_cores):
    '''Unit vectors e_{index_k} (shape W + (Nk,)) -- the "apply vectors" whose adjoint is the entry
    scatter, so that tv_entries_transpose is tv_apply_transpose with these one-hot vectors.'''
    use_jax = tree_contains_jax((index, up_tucker_cores))
    is_uniform = is_ndarray(up_tucker_cores)
    xnp, _, _ = get_backend(is_uniform, use_jax)
    index = xnp.array(index)
    if is_uniform:
        # packed one-hots (d,)+W+(N,): eye(N) indexed by the (d,)+W index array. N is the padded mode dim;
        # each real index_i < Ni <= N puts the 1 in the real region (the padding stays zero -> contracts
        # to zero against the masked supercore).
        N = up_tucker_cores.shape[-1]
        return xnp.eye(N)[index]
    return tuple(xnp.eye(B.shape[-1])[index[i]] for i, B in enumerate(up_tucker_cores))
