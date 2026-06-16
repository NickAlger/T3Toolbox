# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.ut3_orthogonalization as ut3_orthogonalization
import t3toolbox.backend.ut3_operations as ut3_operations
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ranks as ranks
from t3toolbox.backend.common import *

__all__ = [
    'ut3svd',
    'uniform_t3_svd',
]

UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[NDArray, NDArray, NDArray]]


def _cap_ranks(current, spec, length, xnp):  # current: (length,)+stack ; spec: None|int|seq|array
    """Cap current real ranks by a max-rank spec (None = no cap; per-position None entries allowed)."""
    if spec is None:
        return current
    spec = ranks.normalize_max_ranks(spec, length)  # tuple, entries int|array|None
    capped = [current[i] if spec[i] is None else xnp.minimum(current[i], spec[i]) for i in range(length)]
    return xnp.stack(capped)


def _reverse_max_ranks(spec):  # reverse a max-rank spec along the mode axis (None/scalar unchanged)
    if spec is None or isinstance(spec, (int, np.integer)):
        return spec
    return spec[::-1]


def ut3svd(
        data:             UT3Data,
        max_tucker_ranks: typ.Union[int, typ.Sequence[int], NDArray, None] = None,  # scalar / len=d / (d,)+stack
        max_tt_ranks:     typ.Union[int, typ.Sequence[int], NDArray, None] = None,  # scalar / len=d+1 / (d+1,)+stack
        minimize_ranks:   bool = True,
        assume_orthogonal: str = None,
) -> typ.Tuple[
    UT3Data,  # new_x (minimal structural ranks of the capped target, or the capped ranks if not minimizing)
    NDArray,  # Tucker singular values, shape=(d,)+stack+(n',)
    NDArray,  # TT singular values,     shape=(d+1,)+stack+(r',)
]:
    """Mask-truncated T3-SVD of a uniform Tucker tensor train. Matches ragged ``t3svd`` on real parts.

    Truncation is by **max rank only** (no rtol/atol -- those would make data-dependent shapes). The
    sweep truncates to the **capped** ranks (``min(current, max)``), keeping the full Tucker rank through
    each bond SVD -- the best approximation (ragged ``t3svd`` "option a"). With ``minimize_ranks=True``
    (default) a second sweep re-tightens to the minimal **structural** ranks of the capped target
    (`compute_minimal_ranks`), dropping ranks a hard cap orphaned (lossless -- the cap-result's structural
    minimum IS the minimal masks); ``minimize_ranks=False`` skips that and keeps the capped ranks (same
    tensor, possibly non-minimal). The padded supercore shrinks to whichever it kept. Per-stack-element
    ``max_*_ranks`` arrays are allowed (the variety / rank sweep). See ``docs/t3svd_minimal_ranks.md``.

    ``assume_orthogonal`` (``None``/``'left'``/``'right'``) skips the initial orthogonalization when the
    input is already in that orthogonal form (as in ragged ``t3svd``): ``'right'`` skips it; ``'left'``
    reverses to a right-orthogonal T3, sweeps, and reverses back (R->L truncation). **Not enforced**.

    (Slice A: the re-tighten re-runs the full sweep; Slice B replaces it with a cheaper R->L pass.)
    """
    assume_orthogonal = ranks.normalize_assume_orthogonal(assume_orthogonal)
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    # 'left'-orthogonal input: reverse -> right-orthogonal, run the 'right' path, reverse back.
    if assume_orthogonal == 'left':
        rev = ut3_operations.ut3_reverse(data)
        rev_result, ss_tucker, ss_tt = ut3svd(
            rev, _reverse_max_ranks(max_tucker_ranks), _reverse_max_ranks(max_tt_ranks),
            minimize_ranks=minimize_ranks, assume_orthogonal='right')
        return ut3_operations.ut3_reverse(rev_result), ss_tucker[::-1], ss_tt[::-1]

    masked_tucker, masked_tt = ut3_masking.apply_masks_to_cores(data)
    shape_mask, tucker_mask, tt_mask = data[2]

    d, N = masked_tucker.shape[0], masked_tucker.shape[-1]
    n, r = masked_tucker.shape[-2], masked_tt.shape[-1]
    shape = tuple(int(m.sum()) for m in shape_mask)

    capped_tucker = _cap_ranks(tucker_mask.sum(axis=-1), max_tucker_ranks, d, xnp)
    capped_tt = _cap_ranks(tt_mask.sum(axis=-1), max_tt_ranks, d + 1, xnp)

    # Truncate to the CAPPED ranks (not the minimal ranks): the bond SVD sees the full Tucker rank ->
    # the best approximation (ragged "option a"). The singular values from this sweep are the truncation
    # singular values we report.
    cap_masks = ut3_masking.make_uniform_masks(shape, capped_tucker, capped_tt, N, n, r, use_jax=use_jax)
    (out_tucker, out_tt), ss_tucker, ss_tt = uniform_t3_svd(
        (masked_tucker, masked_tt), cap_masks, skip_orthogonalization=(assume_orthogonal == 'right'))

    keep_tucker, keep_tt, keep_masks = capped_tucker, capped_tt, cap_masks
    if minimize_ranks:
        # Re-tighten to minimal ranks: re-run the sweep with the minimal masks. Lossless (the cap-result's
        # structural minimum IS the minimal masks), drops only orphaned ranks. Re-run singular values are
        # discarded -- we keep the truncation singular values from the cap sweep, sliced to minimal.
        min_tucker, min_tt = ranks.compute_minimal_ranks(shape, capped_tucker, capped_tt)
        min_masks = ut3_masking.make_uniform_masks(shape, min_tucker, min_tt, N, n, r, use_jax=use_jax)
        (out_tucker, out_tt), _, _ = uniform_t3_svd((out_tucker, out_tt), min_masks)
        keep_tucker, keep_tt, keep_masks = min_tucker, min_tt, min_masks

    # shrink the padded supercore to the kept ranks
    n2 = int(xnp.max(keep_tucker))
    r2 = int(xnp.max(keep_tt))
    out_tucker = out_tucker[..., :n2, :]
    out_tt = out_tt[..., :r2, :n2, :r2]
    _, keep_tkm, keep_ttm = keep_masks
    new_masks = (shape_mask, keep_tkm[..., :n2], keep_ttm[..., :r2])
    return (out_tucker, out_tt, new_masks), ss_tucker[..., :n2], ss_tt[..., :r2]


def uniform_t3_svd(
        cores: typ.Tuple[
            NDArray,  # tucker_supercore (assumed masked)
            NDArray,  # tt_supercore
        ],
        rank_truncation_masks: typ.Tuple[
            NDArray,  # shape_mask (unused: the SVD does not truncate physical dims)
            NDArray,  # tucker_edge_mask -- prefix truncation masks
            NDArray,  # tt_edge_mask
        ],
        squash_tails_first: bool = True,
        skip_orthogonalization: bool = False,  # assume input already right-orthogonal (Tucker down + TT right)
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],  # (tucker_supercore, tt_supercore) at the INPUT padded (n, r)
    NDArray,  # basis_singular_values, shape=(d,)+stack+(n,)
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

    basis_supercore, tt_supercore = cores
    _, basis_masks, tt_masks = rank_truncation_masks

    if squash_tails_first:
        tt_supercore = ut3_operations.uniform_squash_tt_tails(tt_supercore)

    d = basis_supercore.shape[0]
    stack_shape = basis_supercore.shape[1:-2]
    n, N = basis_supercore.shape[-2:]
    r = tt_supercore.shape[-1]

    if not skip_orthogonalization:
        basis_supercore, tt_supercore = ut3_orthogonalization.down_orthogonalize_tucker_supercores(
            basis_supercore, tt_supercore)
        tt_supercore = orth.right_orthogonalize_tt_cores(tt_supercore)

    # keep everything the same shape, for consistency with masks
    n2 = basis_supercore.shape[-2]
    basis_supercore = xnp.concatenate([basis_supercore, xnp.zeros((d,) + stack_shape + (n - n2, N))], axis=-2)
    tt_supercore = xnp.concatenate([tt_supercore, xnp.zeros((d,) + stack_shape + (r, n - n2, r))], axis=-2)

    _, ss_tt00, _ = xnp.linalg.svd(tt_supercore[0].reshape(stack_shape + (r, n * r)), full_matrices=False)
    ss_tt0 = xnp.concatenate([ss_tt00, xnp.zeros(stack_shape + (r - ss_tt00.shape[-1],))], axis=-1)
    ss_tt0 = ss_tt0 * tt_masks[0]

    def _step(carry, x):
        Y = carry  # (r, r)
        B, G, basis_mask, tt_mask_i = x

        G = xnp.einsum('...ij,...jak->...iak', Y, G)
        M = G.swapaxes(-2, -1).reshape(stack_shape + (r * r, n))
        U, ss_basis, Vt = xnp.linalg.svd(M, full_matrices=False)
        nb = ss_basis.shape[-1]
        U = xnp.concatenate([U, xnp.zeros(stack_shape + (r * r, n - nb))], axis=-1)
        ss_basis = xnp.concatenate([ss_basis, xnp.zeros(stack_shape + (n - nb,))], axis=-1)
        Vt = xnp.concatenate([Vt, xnp.zeros(stack_shape + (n - nb, n))], axis=-2)
        U = U * basis_mask.reshape(stack_shape + (1, -1))
        ss_basis = ss_basis * basis_mask
        Vt = Vt * basis_mask.reshape(stack_shape + (-1, 1))

        new_B = xnp.einsum('...ij,...jk->...ik', Vt, B)

        M = xnp.einsum('...ij,...j->...ij', U, ss_basis).reshape(
            stack_shape + (r, r, n)).swapaxes(-1, -2).reshape(stack_shape + (r * n, r))
        U, ss_tt, Vt = xnp.linalg.svd(M, full_matrices=False)
        U = U * tt_mask_i.reshape(stack_shape + (1, -1))
        ss_tt = ss_tt * tt_mask_i
        Vt = Vt * tt_mask_i.reshape(stack_shape + (-1, 1))

        new_G = U.reshape(stack_shape + (r, n, r))
        Y_next = xnp.einsum('...i,...ij->...ij', ss_tt, Vt)
        return Y_next, (new_B, new_G, ss_basis, ss_tt)

    Y0 = xnp.eye(r)
    if stack_shape:
        Y0 = xnp.tensordot(xnp.ones(stack_shape), Y0, axes=[(), ()])

    Yf, (new_basis_cores, new_tt_cores, basis_singular_values, tt_singular_values0) = xscan(
        _step, Y0, (basis_supercore, tt_supercore, basis_masks, tt_masks[1:]))

    G_last = xnp.einsum('d...iaj,...jk->d...iak', new_tt_cores[-1:], Yf)
    new_tt_cores = xnp.concatenate([new_tt_cores[:-1], G_last], axis=0)

    tt_singular_values = xnp.concatenate([ss_tt0.reshape((1,) + stack_shape + (r,)), tt_singular_values0], axis=0)
    return (new_basis_cores, new_tt_cores), basis_singular_values, tt_singular_values
