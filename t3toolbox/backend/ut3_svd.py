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
    :py:class:`~t3toolbox.backend.sharing.T3SharedFrameData`).
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
            (masked_tucker, masked_tt), cap_masks, skip_orthogonalization=assume_orthogonal)
        raw_tucker, raw_tt = ranks.compute_raw_sweep_ranks(
            shape, tucker_mask.sum(axis=-1), tt_mask.sum(axis=-1), capped_tucker, capped_tt)
    else:
        (out_tucker, out_tt), ss_tucker, ss_tt = _ut3svd_shared_supercores(
            (masked_tucker, masked_tt), cap_masks, groups, skip_orthogonalization=assume_orthogonal)
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
            (masked_tucker, masked_tt), min_masks, skip_orthogonalization=True)
    else:
        (out_tucker, out_tt), _, _ = _ut3svd_shared_supercores(
            (masked_tucker, masked_tt), min_masks, groups, skip_orthogonalization=True)

    n2 = int(np.max(min_tucker))
    r2 = int(np.max(min_tt))
    out_tucker = out_tucker[..., :n2, :]
    out_tt = out_tt[..., :r2, :n2, :r2]
    new_masks = (min_masks[0][..., :n2], min_masks[1][..., :r2])
    return (out_tucker, out_tt, shape, new_masks)


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

    if not skip_orthogonalization:
        frame_supercore, tt_supercore = ut3_orthogonalization.down_orthogonalize_tucker_supercores(
            frame_supercore, tt_supercore)
        tt_supercore = orth.tt_right_orthogonalize(tt_supercore)

    # keep everything the same shape, for consistency with masks
    n2 = frame_supercore.shape[-2]
    frame_supercore = xnp.concatenate([frame_supercore, xnp.zeros((d,) + stack_shape + (n - n2, N))], axis=-2)
    tt_supercore = xnp.concatenate([tt_supercore, xnp.zeros((d,) + stack_shape + (r, n - n2, r))], axis=-2)

    _, ss_tt00, _ = xnp.linalg.svd(tt_supercore[0].reshape(stack_shape + (r, n * r)), full_matrices=False)
    ss_tt0 = xnp.concatenate([ss_tt00, xnp.zeros(stack_shape + (r - ss_tt00.shape[-1],))], axis=-1)
    ss_tt0 = ss_tt0 * tt_masks[0]

    def _step(carry, x):
        Y = carry  # (r, r)
        B, G, frame_mask, tt_mask_i = x

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

    Y0 = xnp.eye(r)
    if stack_shape:
        Y0 = xnp.tensordot(xnp.ones(stack_shape), Y0, axes=[(), ()])

    Yf, (new_frame_cores, new_tt_cores, frame_singular_values, tt_singular_values0) = xscan(
        _step, Y0, (frame_supercore, tt_supercore, frame_masks, tt_masks[1:]))

    G_last = xnp.einsum('d...iaj,...jk->d...iak', new_tt_cores[-1:], Yf)
    new_tt_cores = xnp.concatenate([new_tt_cores[:-1], G_last], axis=0)

    tt_singular_values = xnp.concatenate([ss_tt0.reshape((1,) + stack_shape + (r,)), tt_singular_values0], axis=0)
    return (new_frame_cores, new_tt_cores), frame_singular_values, tt_singular_values


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

    if not skip_orthogonalization:
        frame_supercore, tt_supercore = ut3_orthogonalization.down_orthogonalize_tucker_supercores(
            frame_supercore, tt_supercore)
        tt_supercore = orth.tt_right_orthogonalize(tt_supercore)

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
    def _tt_step(carry, x):
        Y = carry  # (r, r)
        G, tt_mask_i = x

        G = xnp.einsum('...ij,...jak->...iak', Y, G)
        M = G.reshape(stack_shape + (r * n, r))
        U, ss, Vt = xnp.linalg.svd(M, full_matrices=False)   # thin: exactly r columns (r <= r*n)
        U = U * tt_mask_i.reshape(stack_shape + (1, -1))
        ss = ss * tt_mask_i
        Vt = Vt * tt_mask_i.reshape(stack_shape + (-1, 1))

        new_G = U.reshape(stack_shape + (r, n, r))
        Y_next = xnp.einsum('...i,...ij->...ij', ss, Vt)
        return Y_next, (new_G, ss)

    Y0 = xnp.eye(r)
    if stack_shape:
        Y0 = xnp.tensordot(xnp.ones(stack_shape), Y0, axes=[(), ()])
    Yf, (rounded_tt, ss_tt_scan) = xscan(_tt_step, Y0, (tt_supercore, tt_masks[1:]))
    G_last = xnp.einsum('d...iaj,...jk->d...iak', rounded_tt[-1:], Yf)
    rounded_tt = xnp.concatenate([rounded_tt[:-1], G_last], axis=0)

    # ---- phase 2: collect every mode's center of the TT-rounded tensor (lossless) ----
    right_tt, HH = orth.tt_right_orthogonalize(rounded_tt, return_variation_cores=True)

    # ---- phase 3: all Tucker truncations at once, per group (static gathers, no segment sums) ----
    new_tucker_modes = [None] * d
    new_tt_modes = [None] * d
    ss_tucker_modes = [None] * d
    for group in groups:
        mats = [HH[ii].swapaxes(-3, -2).reshape(stack_shape + (n, r * r)) for ii in group]
        M = xnp.concatenate(mats, axis=-1)      # the group concatenation [W2_i1 | ... | W2_ik]
        Y, ss, _ = xnp.linalg.svd(M, full_matrices=False)
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
    new_tt_cores = orth.tt_left_orthogonalize(new_tt_cores)

    # boundary norm at the right edge (of the FINAL tensor), as the ragged sweep reports it
    _, ss_last0, _ = xnp.linalg.svd(new_tt_cores[-1].reshape(stack_shape + (r * n, r)), full_matrices=False)
    ss_last = ss_last0 * tt_masks[-1]

    tt_singular_values = xnp.concatenate(
        [ss_tt0.reshape((1,) + stack_shape + (r,)),
         ss_tt_scan[:-1],
         ss_last.reshape((1,) + stack_shape + (r,))], axis=0)
    return (new_frame_cores, new_tt_cores), frame_singular_values, tt_singular_values
