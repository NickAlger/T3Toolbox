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


def ut3svd(
        data:             UT3Data,
        max_tucker_ranks: typ.Union[int, typ.Sequence[int], NDArray, None] = None,  # scalar / len=d / (d,)+stack
        max_tt_ranks:     typ.Union[int, typ.Sequence[int], NDArray, None] = None,  # scalar / len=d+1 / (d+1,)+stack
) -> typ.Tuple[
    UT3Data,  # new_x (ranks shrunk to the minimal structural ranks of the capped target)
    NDArray,  # Tucker singular values, shape=(d,)+stack+(n',)
    NDArray,  # TT singular values,     shape=(d+1,)+stack+(r',)
]:
    """Mask-truncated T3-SVD of a uniform Tucker tensor train.

    Truncation is by **max rank only** (no rtol/atol -- those would make data-dependent shapes). The
    output ranks are the minimal **structural** ranks of the capped target (``min(current, max)`` then
    `compute_minimal_ranks`), and the padded supercore shrinks to those (minimal-for-free). Per-stack-
    element ``max_*_ranks`` arrays are allowed (the variety / rank sweep). Matches ``t3svd`` on real parts.
    """
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    masked_tucker, masked_tt = ut3_masking.apply_masks_to_cores(data)
    shape_mask, tucker_mask, tt_mask = data[2]

    d, N = masked_tucker.shape[0], masked_tucker.shape[-1]
    n, r = masked_tucker.shape[-2], masked_tt.shape[-1]
    shape = tuple(int(m.sum()) for m in shape_mask)

    capped_tucker = _cap_ranks(tucker_mask.sum(axis=-1), max_tucker_ranks, d, xnp)
    capped_tt = _cap_ranks(tt_mask.sum(axis=-1), max_tt_ranks, d + 1, xnp)
    new_tucker_ranks, new_tt_ranks = ranks.compute_minimal_ranks(shape, capped_tucker, capped_tt)

    trunc_masks = ut3_masking.make_uniform_masks(shape, new_tucker_ranks, new_tt_ranks, N, n, r, use_jax=use_jax)
    (out_tucker, out_tt), ss_tucker, ss_tt = uniform_t3_svd((masked_tucker, masked_tt), trunc_masks)

    # shrink the padded supercore to the minimal structural ranks the SVD produced
    n2 = int(xnp.max(new_tucker_ranks))
    r2 = int(xnp.max(new_tt_ranks))
    out_tucker = out_tucker[..., :n2, :]
    out_tt = out_tt[..., :r2, :n2, :r2]
    _, out_tkm, out_ttm = trunc_masks
    new_masks = (shape_mask, out_tkm[..., :n2], out_ttm[..., :r2])
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
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],  # (tucker_supercore, tt_supercore) at the INPUT padded (n, r)
    NDArray,  # basis_singular_values, shape=(d,)+stack+(n,)
    NDArray,  # tt_singular_values,    shape=(d+1,)+stack+(r,)
]:
    """The T3-SVD sweep: orthogonalize, then a left-to-right scan that SVDs each Tucker/TT edge, pads the
    factors back to the padded size, and multiplies by the prefix truncation masks. Operates at the input
    padded ``(n, r)``; :py:func:`ut3svd` builds the masks and shrinks afterward.
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
