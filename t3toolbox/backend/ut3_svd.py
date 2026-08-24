# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform T3-SVD: minimal-rank reduction of ut3 data with prefix-mask output.

``ut3svd`` / ``ut3svd_supercores`` / ``ut3_rank_adjustment_sweep`` -- SVD-based so the output
masks are a deterministic prefix (``docs/contributor/uniform_svd_prefix_orthogonalization.md``).
"""
import numpy as np
import typing as typ

import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.tt_orthogonalization as orth
import t3toolbox.backend.ut3_orthogonalization as ut3_orthogonalization
import t3toolbox.backend.ut3_operations as ut3_operations
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.sharing as sharing_module
from t3toolbox.backend.common import *

__all__ = [
    'ut3svd',
    'ut3svd_supercores',
    'ut3_rank_adjustment_sweep',
]

# .data[2] is the static int-tuple shape; .data[3] = (tucker_edge_mask, tt_edge_mask) are HOST bool,
# static structure (numpy, never traced); the supercores are xnp.
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]]


def _cap_ranks(current, spec, length):  # current: HOST int (length,)+stack ; spec: None|int|seq|array
    """Cap current real ranks by a max-rank spec (None = no cap; per-position None entries allowed).

    np (host): ranks are mask (structure) metadata, computed on the host (a jax rank under jit is a
    tracer -> shape-extraction / mask leaks). See docs/contributor/uniform_pytree_composition.md.
    """
    if spec is None:
        return current
    spec = ranks.normalize_max_ranks(spec, length)  # tuple, entries int|array|None
    capped = [current[i] if spec[i] is None else np.minimum(current[i], spec[i]) for i in range(length)]
    return np.stack(capped)


def ut3svd(
        data:             UT3Data,
        max_tucker_ranks: typ.Union[int, typ.Sequence[int], NDArray, None] = None,  # scalar / len=d / (d,)+stack
        max_tt_ranks:     typ.Union[int, typ.Sequence[int], NDArray, None] = None,  # scalar / len=d+1 / (d+1,)+stack
        assume_orthogonal: bool = False,
        sharing:          typ.Optional[typ.Sequence] = None,  # len=d, static; one hashable group label per mode (None = unshared)
) -> typ.Tuple[
    UT3Data,  # new_x (left-orthogonal; the raw-sweep ranks -- NOT necessarily minimal under truncation)
    NDArray,  # Tucker singular values, shape=(d,)+stack+(n',)
    NDArray,  # TT singular values,     shape=(d+1,)+stack+(r',)
]:
    """Mask-truncated T3-SVD of a uniform Tucker tensor train -- the basic algorithm, matching ragged
    ``t3svd`` on real parts. Always **left-orthogonal**; under truncation **not** necessarily minimal.

    Truncation is by **max rank only** (no rtol/atol -- those would make data-dependent shapes): a single
    left-to-right sweep, shrinking the padded supercore to the raw-sweep content ranks
    (`compute_raw_sweep_ranks`). It does **not** re-tune to minimal ranks -- use
    :py:func:`ut3_rank_adjustment_sweep`. Per-stack-element ``max_*_ranks`` arrays are allowed (the
    variety / rank sweep). ``assume_orthogonal=True`` skips the orthogonalization, asserting the input is
    already right-orthogonal (not enforced). See ``docs/t3svd_minimal_ranks.md``.

    ``sharing`` (SF-T3 grouped truncation, matching ragged ``t3svd(sharing=)`` on real parts): the
    input's masked factors must be tied within groups (the frontend checks in safe mode; ranks/caps
    equal within groups is structural, checked here). ``sharing=None`` and all-singleton partitions
    dispatch to the literal unshared sweep above (bit-identical); a real group runs the two-phase
    grouped sweep (:py:func:`_ut3svd_shared_supercores`), with ONE group rank mask assigned to every
    group mode, and the reported spectra masked to the FINAL ranks (the ragged two-phase's trimming;
    group modes carry the group spectrum ``s_g``, ``sqrt(k)``-inflated -- see
    :py:class:`~t3toolbox.backend.sharing.SharedFrameData`).
    """
    masked_tucker, masked_tt = ut3_masking.ut3_apply_masks(data)   # guards: masks must be host
    shape = data[2]                                                     # static int tuple
    tucker_mask, tt_mask = data[3]                                      # HOST bool rank masks

    d = masked_tucker.shape[0]
    n, r = masked_tucker.shape[-2], masked_tt.shape[-1]

    groups = None
    if sharing is not None:
        all_groups = sharing_module.validate_sharing(sharing, shape)
        if sharing_module.nontrivial_groups(all_groups):
            groups = all_groups
        # trivial partition -> the literal unshared sweep below (bit-identical)

    # All rank/mask computation is np (host) -- masks are static structure (a jax rank/mask is a tracer
    # under jit); only the SVD sweep below touches the supercores via xnp. See docs/contributor/uniform_pytree_composition.md.
    capped_tucker = _cap_ranks(tucker_mask.sum(axis=-1), max_tucker_ranks, d)
    capped_tt = _cap_ranks(tt_mask.sum(axis=-1), max_tt_ranks, d + 1)

    cap_masks = ut3_masking.ut3_make_masks(capped_tucker, capped_tt, n, r)
    if groups is None:
        (out_tucker, out_tt), ss_tucker, ss_tt = ut3svd_supercores(
            (masked_tucker, masked_tt), cap_masks, skip_orthogonalization=assume_orthogonal,
            content=(shape, tucker_mask, tt_mask))
        raw_tucker, raw_tt = ranks.compute_raw_sweep_ranks(
            shape, tucker_mask.sum(axis=-1), tt_mask.sum(axis=-1), capped_tucker, capped_tt)
    else:
        (out_tucker, out_tt), ss_tucker, ss_tt = _ut3svd_shared_supercores(
            (masked_tucker, masked_tt), cap_masks, groups, skip_orthogonalization=assume_orthogonal,
            content=(shape, tucker_mask, tt_mask))
        # the grouped recurrence also validates within-group equality of the input ranks AND the caps
        raw_tucker, raw_tt = ranks.compute_raw_sweep_ranks(
            shape, tucker_mask.sum(axis=-1), tt_mask.sum(axis=-1), capped_tucker, capped_tt,
            sharing=sharing)

    # Shrink the (left-orthogonal) result to the raw-sweep content ranks -- the actual ranks the SVDs
    # kept (= what the masks would be once the loose padding is removed; possibly non-minimal).
    n2 = int(np.max(raw_tucker))
    r2 = int(np.max(raw_tt))
    out_tucker = out_tucker[..., :n2, :]
    out_tt = out_tt[..., :r2, :n2, :r2]
    raw_masks = ut3_masking.ut3_make_masks(raw_tucker, raw_tt, n, r)
    new_masks = (raw_masks[0][..., :n2], raw_masks[1][..., :r2])
    ss_tucker, ss_tt = ss_tucker[..., :n2], ss_tt[..., :r2]
    if groups is not None:
        # trim the reported spectra to the FINAL ranks (the ragged two-phase trims its phase-1 bond
        # spectra to the final bond dims; masking is the uniform trim)
        ss_tucker = ss_tucker * new_masks[0]
        ss_tt = ss_tt * new_masks[1]
    return (out_tucker, out_tt, shape, new_masks), ss_tucker, ss_tt


def ut3_rank_adjustment_sweep(
        data: UT3Data,
        direction: str = 'right_to_left',  # 'right_to_left' | 'left_to_right'
        sharing: typ.Optional[typ.Sequence] = None,  # len=d, static; one hashable group label per mode (None = unshared)
) -> UT3Data:
    """A single lossless directional sweep that drops structurally-redundant ranks -- the uniform analog
    of ragged ``rank_adjustment_sweep`` (the minimization step; :py:func:`ut3svd` does not minimize).

    ``'right_to_left'`` returns a **right-orthogonal** result; ``'left_to_right'`` a **left-orthogonal**
    one. It reaches the minimal ranks **only if the input is already orthogonal in the opposite
    direction** -- a :py:func:`ut3svd` result is left-orthogonal, so ``'right_to_left'`` minimizes it.
    (The precondition is required here: unlike ragged, the static masks shrink to the minimal ranks, so a
    wrong-direction call on a non-oppositely-orthogonal input is lossy. Compose both directions for a
    minimal result in a chosen gauge.)

    With ``sharing``, the reduction is the grouped one (SHARED minimal ranks -- the group ceiling, so a
    shared rank exceeding a single mode's local ceiling is kept, where the per-mode reduction would clip
    it and untie the group) and the sweep is the grouped two-phase (tied factors stay tied, one array's
    worth of content at every group mode). The partition is remapped through the mode reversal.
    """
    if direction == 'right_to_left':
        rev_sharing = None if sharing is None else tuple(reversed(tuple(sharing)))
        return ut3_operations.ut3_reverse(
            _reduce_left_to_right(ut3_operations.ut3_reverse(data), sharing=rev_sharing))
    elif direction == 'left_to_right':
        return _reduce_left_to_right(data, sharing=sharing)
    raise ValueError("direction must be 'left_to_right' or 'right_to_left'; got %r" % (direction,))


def _reduce_left_to_right(
        data: UT3Data,
        sharing: typ.Optional[typ.Sequence] = None,  # len=d, static; group labels (None = unshared)
) -> UT3Data:
    """Single left-to-right reduction to minimal ranks, skipping orthogonalization (assumes the input is
    right-orthogonal). Lossless when that precondition holds. Shrinks the padded supercore to minimal
    (with ``sharing``: to the SHARED minimal ranks, via the grouped two-phase sweep)."""
    masked_tucker, masked_tt = ut3_masking.ut3_apply_masks(data)   # guards: masks must be host
    shape = data[2]                                                     # static int tuple
    tucker_mask, tt_mask = data[3]                                      # HOST bool rank masks
    n, r = masked_tucker.shape[-2], masked_tt.shape[-1]

    groups = None
    if sharing is not None:
        all_groups = sharing_module.validate_sharing(sharing, shape)
        if sharing_module.nontrivial_groups(all_groups):
            groups = all_groups

    # ranks/masks on the host (np); compute_minimal_ranks defaults to numpy. See docs/contributor/uniform_pytree_composition.md.
    min_tucker, min_tt = ranks.compute_minimal_ranks(shape, tucker_mask.sum(axis=-1), tt_mask.sum(axis=-1),
                                                     sharing=sharing)
    min_masks = ut3_masking.ut3_make_masks(min_tucker, min_tt, n, r)
    if groups is None:
        (out_tucker, out_tt), _, _ = ut3svd_supercores(
            (masked_tucker, masked_tt), min_masks, skip_orthogonalization=True,
            content=(shape, tucker_mask, tt_mask))
    else:
        (out_tucker, out_tt), _, _ = _ut3svd_shared_supercores(
            (masked_tucker, masked_tt), min_masks, groups, skip_orthogonalization=True,
            content=(shape, tucker_mask, tt_mask))

    n2 = int(np.max(min_tucker))
    r2 = int(np.max(min_tt))
    out_tucker = out_tucker[..., :n2, :]
    out_tt = out_tt[..., :r2, :n2, :r2]
    new_masks = (min_masks[0][..., :n2], min_masks[1][..., :r2])
    return (out_tucker, out_tt, shape, new_masks)


def _ut3svd_step(
        carry: NDArray,        # Y: stack_shape+(r,r), the running bond factor
        x:     typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (B, G, frame_mask, tt_mask_i), one mode
) -> typ.Tuple[
        NDArray,               # Y_next: stack_shape+(r,r)
        typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (new_B, new_G, ss_frame, ss_tt)
]:
    '''One mode of the uniform T3-SVD sweep of :py:func:`ut3svd_supercores`. Closure-free scan
    body -- ``docs/contributor/scan_body_principles.md``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, x)))
    Y = carry  # (r, r)
    B, G, frame_mask, tt_mask_i = x
    stack_shape = carry.shape[:-2]
    r = carry.shape[-1]
    n = G.shape[-2]                                    # before the einsum below rebinds G

    G = xnp.einsum('...ij,...jak->...iak', Y, G)
    M = G.swapaxes(-2, -1).reshape(stack_shape + (r * r, n))
    U, ss_frame, Vt = xnp.linalg.svd(M, full_matrices=False)
    nb = ss_frame.shape[-1]
    U = xnp.concatenate([U, xnp.zeros(stack_shape + (r * r, n - nb))], axis=-1)
    ss_frame = xnp.concatenate([ss_frame, xnp.zeros(stack_shape + (n - nb,))], axis=-1)
    Vt = xnp.concatenate([Vt, xnp.zeros(stack_shape + (n - nb, n))], axis=-2)
    U = U * frame_mask.reshape(stack_shape + (1, -1))
    ss_frame = ss_frame * frame_mask
    Vt = Vt * frame_mask.reshape(stack_shape + (-1, 1))

    new_B = xnp.einsum('...ij,...jk->...ik', Vt, B)

    M = xnp.einsum('...ij,...j->...ij', U, ss_frame).reshape(
        stack_shape + (r, r, n)).swapaxes(-1, -2).reshape(stack_shape + (r * n, r))
    U, ss_tt, Vt = xnp.linalg.svd(M, full_matrices=False)
    U = U * tt_mask_i.reshape(stack_shape + (1, -1))
    ss_tt = ss_tt * tt_mask_i
    Vt = Vt * tt_mask_i.reshape(stack_shape + (-1, 1))

    new_G = U.reshape(stack_shape + (r, n, r))
    Y_next = xnp.einsum('...i,...ij->...ij', ss_tt, Vt)
    return Y_next, (new_B, new_G, ss_frame, ss_tt)


def _ut3svd_step_pad_safe(
        carry: NDArray,        # Y: stack_shape+(r,r), the running bond factor
        x:     typ.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray],
               # (B, G, frame_mask, tt_mask_i, row1, col1, row2, col2) -- one mode; the last four are
               # the pad-safe content masks (HOST, from the raw-sweep recurrence)
) -> typ.Tuple[
        NDArray,               # Y_next: stack_shape+(r,r)
        typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (new_B, new_G, ss_frame, ss_tt)
]:
    """Pad-safe twin of :py:func:`_ut3svd_step` (review S1b): both kept-basis SVDs go through
    :py:func:`~t3toolbox.backend.linalg.pad_safe_svd`, so at a numerically rank-deficient point the
    sigma~0 completion columns stay OFF the padded slots -- which matters when the output cores
    become a FRAME (``ut3svd_orthogonal_representations`` / ``already_left_orthogonal=True``).
    Closure-free scan body; masks ride the scan ``xs`` as HOST arrays (constants under jit)."""
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, x)))
    Y = carry  # (r, r)
    B, G, frame_mask, tt_mask_i, row1, col1, row2, col2 = x
    stack_shape = carry.shape[:-2]
    r = carry.shape[-1]
    n = G.shape[-2]                                    # before the einsum below rebinds G

    G = xnp.einsum('...ij,...jak->...iak', Y, G)
    M = G.swapaxes(-2, -1).reshape(stack_shape + (r * r, n))
    U, ss_frame, Vt = linalg.pad_safe_svd(M, row1, col1)         # K = min(r*r, n) columns
    nb = ss_frame.shape[-1]
    U = xnp.concatenate([U, xnp.zeros(stack_shape + (r * r, n - nb))], axis=-1)
    ss_frame = xnp.concatenate([ss_frame, xnp.zeros(stack_shape + (n - nb,))], axis=-1)
    Vt = xnp.concatenate([Vt, xnp.zeros(stack_shape + (n - nb, n))], axis=-2)
    U = U * frame_mask.reshape(stack_shape + (1, -1))
    ss_frame = ss_frame * frame_mask
    Vt = Vt * frame_mask.reshape(stack_shape + (-1, 1))

    new_B = xnp.einsum('...ij,...jk->...ik', Vt, B)

    M = xnp.einsum('...ij,...j->...ij', U, ss_frame).reshape(
        stack_shape + (r, r, n)).swapaxes(-1, -2).reshape(stack_shape + (r * n, r))
    U, ss_tt, Vt = linalg.pad_safe_svd(M, row2, col2)
    U = U * tt_mask_i.reshape(stack_shape + (1, -1))
    ss_tt = ss_tt * tt_mask_i
    Vt = Vt * tt_mask_i.reshape(stack_shape + (-1, 1))

    new_G = U.reshape(stack_shape + (r, n, r))
    Y_next = xnp.einsum('...i,...ij->...ij', ss_tt, Vt)
    return Y_next, (new_B, new_G, ss_frame, ss_tt)


def _raw_sweep_step_masks(
        shape,          # static int tuple, len=d
        tkm,            # HOST bool (d,)+stack+(n,)   -- input content masks (post entry-masking)
        ttm_sq,         # HOST bool (d+1,)+stack+(r,) -- input content masks, POST-SQUASH boundaries
        cap_tkm,        # HOST bool (d,)+stack+(n,)   -- the truncation cap masks
        cap_ttm,        # HOST bool (d+1,)+stack+(r,)
):
    """Per-step pad-safe masks for the (unshared) ut3svd scan: the raw-sweep rank recurrence
    (:py:func:`~t3toolbox.backend.ranks.compute_raw_sweep_ranks`), evaluated stepwise on the host.
    Returns stacked (d, ...) arrays (row1, col1, row2, col2) plus the pre-scan content ranks
    (n0, rr) the pre-orthogonalization needs."""
    d = tkm.shape[0]
    n_w = tkm.shape[-1]
    r_w = ttm_sq.shape[-1]
    shape_col = np.asarray(shape).reshape((d,) + (1,) * (tkm.ndim - 2))
    n0 = np.minimum(tkm.sum(axis=-1), shape_col)               # after down-orthogonalization
    rr = ttm_sq.sum(axis=-1).copy()                            # after right-orthogonalization
    for ii in range(d - 1, 0, -1):
        rr[ii] = np.minimum(rr[ii], n0[ii] * rr[ii + 1])
    cap_n = cap_tkm.sum(axis=-1)
    cap_r = cap_ttm.sum(axis=-1)
    rin = rr[0]                                                # evolving capped left bond
    row1 = []; col1 = []; row2 = []; col2 = []
    for ii in range(d):
        lm = prefix_mask(rin, r_w)
        rm = prefix_mask(rr[ii + 1], r_w)
        k2 = lm[..., :, None] & rm[..., None, :]
        row1.append(k2.reshape(k2.shape[:-2] + (-1,)))
        col1.append(prefix_mask(n0[ii], n_w))
        nS = np.minimum(np.minimum(n0[ii], rin * rr[ii + 1]), cap_n[ii])
        k2 = lm[..., :, None] & prefix_mask(nS, n_w)[..., None, :]
        row2.append(k2.reshape(k2.shape[:-2] + (-1,)))
        col2.append(rm)
        rin = np.minimum(np.minimum(rin * nS, rr[ii + 1]), cap_r[ii + 1])
    return ((np.stack(row1), np.stack(col1), np.stack(row2), np.stack(col2)), n0, rr)


def ut3svd_supercores(
        cores: typ.Tuple[
            NDArray,  # tucker_supercore (assumed masked)
            NDArray,  # tt_supercore
        ],
        rank_truncation_masks: typ.Tuple[
            NDArray,  # tucker_edge_mask -- prefix truncation masks
            NDArray,  # tt_edge_mask
        ],
        squash_tails_first: bool = True,
        skip_orthogonalization: bool = False,  # assume input already right-orthogonal (Tucker down + TT right)

        content: typ.Optional[typ.Tuple[typ.Sequence[int], NDArray, NDArray]] = None,
                 # (shape, tucker_edge_mask, tt_edge_mask) of the INPUT -- HOST. Given -> every
                 # kept-basis SVD (pre-orthogonalization + scan) is PAD-SAFE (review S1b).
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],  # (tucker_supercore, tt_supercore) at the INPUT padded (n, r)
    NDArray,  # frame_singular_values, shape=(d,)+stack+(n,)
    NDArray,  # tt_singular_values,    shape=(d+1,)+stack+(r,)
]:
    """The T3-SVD sweep: orthogonalize, then a left-to-right scan that SVDs each Tucker/TT edge, pads the
    factors back to the padded size, and multiplies by the prefix truncation masks. Operates at the input
    padded ``(n, r)``; :py:func:`ut3svd` builds the masks and shrinks afterward.

    ``skip_orthogonalization=True`` assumes the input is already right-orthogonal (Tucker down-orthogonal,
    TT right-orthogonal -- the gauge the L->R scan needs) and skips the orthogonalization passes. Silently
    wrong if the input is not in that form (not checked).
    """
    use_jax = tree_contains_jax(cores)
    xnp, _, xscan = get_backend(True, use_jax)

    frame_supercore, tt_supercore = cores
    frame_masks, tt_masks = rank_truncation_masks
    require_concrete_masks(frame_masks, tt_masks)  # masks (constant operands) are host

    if squash_tails_first:
        tt_supercore = tt_operations.tt_squash_tails(tt_supercore)

    d = frame_supercore.shape[0]
    stack_shape = frame_supercore.shape[1:-2]
    n, N = frame_supercore.shape[-2:]
    r = tt_supercore.shape[-1]

    step_masks = None
    if content is not None:
        shape_c, tkm_c, ttm_c = content
        stack_c = tkm_c.shape[1:-1]
        one_r = prefix_mask(np.ones((1,) + stack_c, dtype=int), r)     # squash -> boundary rank 1
        ttm_sq = np.concatenate([one_r, ttm_c[1:-1], one_r], axis=0) if squash_tails_first else ttm_c
        step_masks, n0_c, rr_c = _raw_sweep_step_masks(shape_c, tkm_c, ttm_sq, frame_masks, tt_masks)

    if not skip_orthogonalization:
        if content is None:
            frame_supercore, tt_supercore = ut3_orthogonalization.down_orthogonalize_tucker_supercores(
                frame_supercore, tt_supercore)
            tt_supercore = orth.tt_right_orthogonalize(tt_supercore)
        else:
            shape_col = np.asarray(shape_c).reshape((d,) + (1,) * len(stack_c))
            frame_supercore, tt_supercore = ut3_orthogonalization.down_orthogonalize_tucker_supercores(
                frame_supercore, tt_supercore,
                row_mask=prefix_mask(shape_col, N), col_mask=tkm_c,
                out_mask=prefix_mask(n0_c, min(n, N)))
            n0_masks = prefix_mask(n0_c, tt_supercore.shape[-2])
            bond_in = prefix_mask(ttm_sq.sum(axis=-1), r)
            tt_supercore = orth.tt_right_orthogonalize(
                tt_supercore,
                pad_masks=ut3_orthogonalization._tt_left_sweep_pad_masks(
                    bond_in[::-1], n0_masks[::-1], rr_c[::-1]))

    # keep everything the same shape, for consistency with masks
    n2 = frame_supercore.shape[-2]
    frame_supercore = xnp.concatenate([frame_supercore, xnp.zeros((d,) + stack_shape + (n - n2, N))], axis=-2)
    tt_supercore = xnp.concatenate([tt_supercore, xnp.zeros((d,) + stack_shape + (r, n - n2, r))], axis=-2)

    _, ss_tt00, _ = xnp.linalg.svd(tt_supercore[0].reshape(stack_shape + (r, n * r)), full_matrices=False)
    ss_tt0 = xnp.concatenate([ss_tt00, xnp.zeros(stack_shape + (r - ss_tt00.shape[-1],))], axis=-1)
    ss_tt0 = ss_tt0 * tt_masks[0]

    Y0 = xnp.eye(r)
    if stack_shape:
        Y0 = xnp.tensordot(xnp.ones(stack_shape), Y0, axes=[(), ()])

    if step_masks is None:
        Yf, (new_frame_cores, new_tt_cores, frame_singular_values, tt_singular_values0) = xscan(
            _ut3svd_step, Y0, (frame_supercore, tt_supercore, frame_masks, tt_masks[1:]))
    else:
        Yf, (new_frame_cores, new_tt_cores, frame_singular_values, tt_singular_values0) = xscan(
            _ut3svd_step_pad_safe, Y0,
            (frame_supercore, tt_supercore, frame_masks, tt_masks[1:]) + step_masks)

    G_last = xnp.einsum('d...iaj,...jk->d...iak', new_tt_cores[-1:], Yf)
    new_tt_cores = xnp.concatenate([new_tt_cores[:-1], G_last], axis=0)

    tt_singular_values = xnp.concatenate([ss_tt0.reshape((1,) + stack_shape + (r,)), tt_singular_values0], axis=0)
    return (new_frame_cores, new_tt_cores), frame_singular_values, tt_singular_values


def _ut3svd_shared_step(
        carry: NDArray,                      # Y: stack_shape+(r,r), the running bond factor
        x:     typ.Tuple[NDArray, NDArray],  # (G, tt_mask_i), one mode
) -> typ.Tuple[
        NDArray,                             # Y_next: stack_shape+(r,r)
        typ.Tuple[NDArray, NDArray],         # (new_G, ss)
]:
    '''One mode of the TT-bond rounding scan of :py:func:`_ut3svd_shared_supercores` -- :py:func:`_ut3svd_step`
    without the Tucker steps. Closure-free scan body -- ``docs/contributor/scan_body_principles.md``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, x)))
    Y = carry  # (r, r)
    G, tt_mask_i = x
    stack_shape = carry.shape[:-2]
    r = carry.shape[-1]
    n = G.shape[-2]                                    # before the einsum below rebinds G

    G = xnp.einsum('...ij,...jak->...iak', Y, G)
    M = G.reshape(stack_shape + (r * n, r))
    U, ss, Vt = xnp.linalg.svd(M, full_matrices=False)   # thin: exactly r columns (r <= r*n)
    U = U * tt_mask_i.reshape(stack_shape + (1, -1))
    ss = ss * tt_mask_i
    Vt = Vt * tt_mask_i.reshape(stack_shape + (-1, 1))

    new_G = U.reshape(stack_shape + (r, n, r))
    Y_next = xnp.einsum('...i,...ij->...ij', ss, Vt)
    return Y_next, (new_G, ss)


def _ut3svd_shared_step_pad_safe(
        carry: NDArray,                      # Y: stack_shape+(r,r), the running bond factor
        x:     typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (G, tt_mask_i, row1, col1), one mode
) -> typ.Tuple[
        NDArray,                             # Y_next: stack_shape+(r,r)
        typ.Tuple[NDArray, NDArray],         # (new_G, ss)
]:
    """Pad-safe twin of :py:func:`_ut3svd_shared_step` (review S1b): the bond-rounding SVD goes
    through :py:func:`~t3toolbox.backend.linalg.pad_safe_svd`. Closure-free scan body; masks ride
    the scan ``xs`` as HOST arrays."""
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, x)))
    Y = carry  # (r, r)
    G, tt_mask_i, row1, col1 = x
    stack_shape = carry.shape[:-2]
    r = carry.shape[-1]
    n = G.shape[-2]                                    # before the einsum below rebinds G

    G = xnp.einsum('...ij,...jak->...iak', Y, G)
    M = G.reshape(stack_shape + (r * n, r))
    U, ss, Vt = linalg.pad_safe_svd(M, row1, col1)       # thin: exactly r columns (r <= r*n)
    U = U * tt_mask_i.reshape(stack_shape + (1, -1))
    ss = ss * tt_mask_i
    Vt = Vt * tt_mask_i.reshape(stack_shape + (-1, 1))

    new_G = U.reshape(stack_shape + (r, n, r))
    Y_next = xnp.einsum('...i,...ij->...ij', ss, Vt)
    return Y_next, (new_G, ss)


def _ut3svd_shared_supercores(
        cores: typ.Tuple[
            NDArray,  # tucker_supercore (assumed masked; factors tied within groups on real content)
            NDArray,  # tt_supercore
        ],
        rank_truncation_masks: typ.Tuple[
            NDArray,  # tucker_edge_mask -- prefix cap masks; equal within groups (validated upstream)
            NDArray,  # tt_edge_mask
        ],
        groups: typ.Tuple[typ.Tuple[int, ...], ...],  # static; canonical partition with >= 1 real group
        skip_orthogonalization: bool = False,  # assume input already right-orthogonal (Tucker down + TT right)

        content: typ.Optional[typ.Tuple[typ.Sequence[int], NDArray, NDArray]] = None,
                 # (shape, tucker_edge_mask, tt_edge_mask) of the INPUT -- HOST. Given -> every
                 # kept-basis SVD (pre-orthogonalization + all four phases) is PAD-SAFE (review S1b).
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],  # (tucker_supercore, tt_supercore) at the INPUT padded (n, r)
    NDArray,  # frame_singular_values, shape=(d,)+stack+(n,); group modes carry the group spectrum s_g
    NDArray,  # tt_singular_values,    shape=(d+1,)+stack+(r,)
]:
    """The grouped (SF-T3) uniform T3-SVD sweep -- the two-phase rounding of ragged ``_t3svd_shared``
    on supercores (Molozhavenko & Rakhuba 2026, Algorithm 1): (1) a TT-bond rounding scan with the
    Tucker steps deliberately SKIPPED (skipping is what keeps the tied factors tied -- a per-mode
    rotation would untie them); (2) one lossless polymorphic right sweep collects every mode's center
    of the SAME TT-rounded tensor; (3) all Tucker truncations at once -- each group truncates ONE SVD
    of its statically-gathered concatenation ``[W2_i1 | ... | W2_ik]`` (zero-padded columns contribute
    zero singular values, so the fixed padded widths are harmless), applying the rotation to the shared
    factor once per group; (4) a lossless left re-orthogonalization restores the left-orthogonal output
    contract. Operates at the input padded ``(n, r)``: masks multiply, nothing is sliced --
    :py:func:`ut3svd` builds the masks and shrinks afterward (same conventions as
    :py:func:`ut3svd_supercores`).
    """
    use_jax = tree_contains_jax(cores)
    xnp, _, xscan = get_backend(True, use_jax)

    frame_supercore, tt_supercore = cores
    frame_masks, tt_masks = rank_truncation_masks
    require_concrete_masks(frame_masks, tt_masks)  # masks (constant operands) are host

    tt_supercore = tt_operations.tt_squash_tails(tt_supercore)

    d = frame_supercore.shape[0]
    stack_shape = frame_supercore.shape[1:-2]
    n, N = frame_supercore.shape[-2:]
    r = tt_supercore.shape[-1]

    # -- pad-safe content recurrences (HOST; review S1b), phase by phase --
    ph = None
    if content is not None:
        shape_c, tkm_c, ttm_c = content
        stack_c = tkm_c.shape[1:-1]
        one_r = prefix_mask(np.ones((1,) + stack_c, dtype=int), r)     # squash -> boundary rank 1
        ttm_sq = np.concatenate([one_r, ttm_c[1:-1], one_r], axis=0)
        shape_col = np.asarray(shape_c).reshape((d,) + (1,) * len(stack_c))
        n0 = np.minimum(tkm_c.sum(axis=-1), shape_col)                 # after down-orth
        rr = ttm_sq.sum(axis=-1).copy()                                # after right-orth
        for ii in range(d - 1, 0, -1):
            rr[ii] = np.minimum(rr[ii], n0[ii] * rr[ii + 1])
        n0m = prefix_mask(n0, n)
        cap_r = tt_masks.sum(axis=-1)
        rin = list(rr)                                                 # phase 1: capped bond rounding
        for ii in range(d - 1):
            rin[ii + 1] = np.minimum(np.minimum(rin[ii + 1], rin[ii] * n0[ii]), cap_r[ii + 1])
        row1 = []; col1 = []
        for ii in range(d):
            k2 = prefix_mask(rin[ii], r)[..., :, None] & n0m[ii][..., None, :]
            row1.append(k2.reshape(k2.shape[:-2] + (-1,)))
            col1.append(prefix_mask(rr[ii + 1], r))
        r2 = list(rin)                                                 # phase 2: lossless right sweep
        for ii in range(d - 1, 0, -1):
            r2[ii] = np.minimum(rin[ii], n0[ii] * r2[ii + 1])
        cap_n = frame_masks.sum(axis=-1)
        n3 = [None] * d                                                # phase 3: group truncations
        for group in groups:
            cols_g = sum(r2[ii] * r2[ii + 1] for ii in group)
            n_g = np.minimum(np.minimum(n0[group[0]], cols_g), cap_n[group[0]])
            for ii in group:
                n3[ii] = n_g
        r4 = list(r2)                                                  # phase 4: lossless left sweep
        for ii in range(d - 1):
            r4[ii + 1] = np.minimum(r4[ii + 1], r4[ii] * n3[ii])
        ph = dict(n0=n0, n0m=n0m, rr=rr, rin=np.stack(rin), r2=np.stack(r2),
                  n3=np.stack(n3), r4=np.stack(r4),
                  row1=np.stack(row1), col1=np.stack(col1), ttm_sq=ttm_sq, shape_col=shape_col)

    if not skip_orthogonalization:
        if ph is None:
            frame_supercore, tt_supercore = ut3_orthogonalization.down_orthogonalize_tucker_supercores(
                frame_supercore, tt_supercore)
            tt_supercore = orth.tt_right_orthogonalize(tt_supercore)
        else:
            frame_supercore, tt_supercore = ut3_orthogonalization.down_orthogonalize_tucker_supercores(
                frame_supercore, tt_supercore,
                row_mask=prefix_mask(ph['shape_col'], N), col_mask=tkm_c,
                out_mask=prefix_mask(ph['n0'], min(n, N)))
            n0m_now = prefix_mask(ph['n0'], tt_supercore.shape[-2])
            bond_in = prefix_mask(ph['ttm_sq'].sum(axis=-1), r)
            tt_supercore = orth.tt_right_orthogonalize(
                tt_supercore,
                pad_masks=ut3_orthogonalization._tt_left_sweep_pad_masks(
                    bond_in[::-1], n0m_now[::-1], ph['rr'][::-1]))

    # keep everything the same shape, for consistency with masks
    n2 = frame_supercore.shape[-2]
    frame_supercore = xnp.concatenate([frame_supercore, xnp.zeros((d,) + stack_shape + (n - n2, N))], axis=-2)
    tt_supercore = xnp.concatenate([tt_supercore, xnp.zeros((d,) + stack_shape + (r, n - n2, r))], axis=-2)

    # tie insurance (the ragged re-assign, as a static gather): copy the group reference factor to
    # every group mode -- exact on tied input, roundoff repair otherwise
    ref_index = np.arange(d)
    for group in groups:
        for jj in group:
            ref_index[jj] = group[0]
    frame_supercore = frame_supercore[np.asarray(ref_index)]

    # boundary spectrum of the right-orthogonalized tensor's first core (as the ragged sweep reports)
    _, ss_tt00, _ = xnp.linalg.svd(tt_supercore[0].reshape(stack_shape + (r, n * r)), full_matrices=False)
    ss_tt0 = xnp.concatenate([ss_tt00, xnp.zeros(stack_shape + (r - ss_tt00.shape[-1],))], axis=-1)
    ss_tt0 = ss_tt0 * tt_masks[0]

    # ---- phase 1: TT-bond rounding scan (Tucker steps deliberately skipped) ----
    Y0 = xnp.eye(r)
    if stack_shape:
        Y0 = xnp.tensordot(xnp.ones(stack_shape), Y0, axes=[(), ()])
    if ph is None:
        Yf, (rounded_tt, ss_tt_scan) = xscan(_ut3svd_shared_step, Y0, (tt_supercore, tt_masks[1:]))
    else:
        Yf, (rounded_tt, ss_tt_scan) = xscan(
            _ut3svd_shared_step_pad_safe, Y0,
            (tt_supercore, tt_masks[1:], ph['row1'], ph['col1']))
    G_last = xnp.einsum('d...iaj,...jk->d...iak', rounded_tt[-1:], Yf)
    rounded_tt = xnp.concatenate([rounded_tt[:-1], G_last], axis=0)

    # ---- phase 2: collect every mode's center of the TT-rounded tensor (lossless) ----
    if ph is None:
        right_tt, HH = orth.tt_right_orthogonalize(rounded_tt, return_variation_cores=True)
    else:
        rin_m = prefix_mask(ph['rin'], r)
        right_tt, HH = orth.tt_right_orthogonalize(
            rounded_tt, return_variation_cores=True,
            pad_masks=ut3_orthogonalization._tt_left_sweep_pad_masks(
                rin_m[::-1], ph['n0m'][::-1], ph['r2'][::-1]))

    # ---- phase 3: all Tucker truncations at once, per group (static gathers, no segment sums) ----
    new_tucker_modes = [None] * d
    new_tt_modes = [None] * d
    ss_tucker_modes = [None] * d
    for group in groups:
        mats = [HH[ii].swapaxes(-3, -2).reshape(stack_shape + (n, r * r)) for ii in group]
        M = xnp.concatenate(mats, axis=-1)      # the group concatenation [W2_i1 | ... | W2_ik]
        if ph is None:
            Y, ss, _ = xnp.linalg.svd(M, full_matrices=False)
        else:
            col_blocks = []
            for ii in group:                     # per-mode kron of the phase-2 bond masks
                k2 = (prefix_mask(ph['r2'][ii], r)[..., :, None]
                      & prefix_mask(ph['r2'][ii + 1], r)[..., None, :])
                col_blocks.append(k2.reshape(k2.shape[:-2] + (-1,)))
            Y, ss, _ = linalg.pad_safe_svd(
                M, ph['n0m'][group[0]], np.concatenate(col_blocks, axis=-1))
        q = ss.shape[-1]
        Y = xnp.concatenate([Y, xnp.zeros(stack_shape + (n, n - q))], axis=-1)
        ss = xnp.concatenate([ss, xnp.zeros(stack_shape + (n - q,))], axis=-1)
        group_mask = frame_masks[group[0]]      # the cap mask; equal at every group mode (validated)
        Y = Y * group_mask.reshape(stack_shape + (1, -1))
        ss = ss * group_mask
        B_shared = xnp.einsum('...ux,...uo->...xo', Y, frame_supercore[group[0]])
        for ii in group:
            new_tucker_modes[ii] = B_shared     # ONE shared factor content at every group mode
            new_tt_modes[ii] = xnp.einsum('...aub,...ux->...axb', right_tt[ii], Y)
            ss_tucker_modes[ii] = ss            # the group spectrum s_g, reported at every group mode
    new_frame_cores = xnp.stack(new_tucker_modes, axis=0)
    new_tt_cores = xnp.stack(new_tt_modes, axis=0)
    frame_singular_values = xnp.stack(ss_tucker_modes, axis=0)

    # ---- phase 4: restore the left-orthogonal output contract (lossless) ----
    if ph is None:
        new_tt_cores = orth.tt_left_orthogonalize(new_tt_cores)
    else:
        new_tt_cores = orth.tt_left_orthogonalize(
            new_tt_cores,
            pad_masks=ut3_orthogonalization._tt_left_sweep_pad_masks(
                prefix_mask(ph['r2'], r), prefix_mask(ph['n3'], n), ph['r4']))

    # boundary norm at the right edge (of the FINAL tensor), as the ragged sweep reports it
    _, ss_last0, _ = xnp.linalg.svd(new_tt_cores[-1].reshape(stack_shape + (r * n, r)), full_matrices=False)
    ss_last = ss_last0 * tt_masks[-1]

    tt_singular_values = xnp.concatenate(
        [ss_tt0.reshape((1,) + stack_shape + (r,)),
         ss_tt_scan[:-1],
         ss_last.reshape((1,) + stack_shape + (r,))], axis=0)
    return (new_frame_cores, new_tt_cores), frame_singular_values, tt_singular_values
