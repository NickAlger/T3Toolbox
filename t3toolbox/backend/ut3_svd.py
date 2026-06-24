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
    'ut3_rank_adjustment_sweep',
]

# .data[2] is the static int-tuple shape; .data[3] = (tucker_edge_mask, tt_edge_mask) are HOST bool,
# static structure (numpy, never traced); the supercores are xnp.
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]]


def _cap_ranks(current, spec, length):  # current: HOST int (length,)+stack ; spec: None|int|seq|array
    """Cap current real ranks by a max-rank spec (None = no cap; per-position None entries allowed).

    np (host): ranks are mask (structure) metadata, computed on the host (a jax rank under jit is a
    tracer -> shape-extraction / mask leaks). See docs/uniform_pytree_composition.md.
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
    """
    masked_tucker, masked_tt = ut3_masking.apply_masks_to_cores(data)   # guards: masks must be host
    shape = data[2]                                                     # static int tuple
    tucker_mask, tt_mask = data[3]                                      # HOST bool rank masks

    d = masked_tucker.shape[0]
    n, r = masked_tucker.shape[-2], masked_tt.shape[-1]

    # All rank/mask computation is np (host) -- masks are static structure (a jax rank/mask is a tracer
    # under jit); only the SVD sweep below touches the supercores via xnp. See docs/uniform_pytree_composition.md.
    capped_tucker = _cap_ranks(tucker_mask.sum(axis=-1), max_tucker_ranks, d)
    capped_tt = _cap_ranks(tt_mask.sum(axis=-1), max_tt_ranks, d + 1)

    cap_masks = ut3_masking.make_uniform_masks(capped_tucker, capped_tt, n, r)
    (out_tucker, out_tt), ss_tucker, ss_tt = uniform_t3_svd(
        (masked_tucker, masked_tt), cap_masks, skip_orthogonalization=assume_orthogonal)

    # Shrink the (left-orthogonal) result to the raw-sweep content ranks -- the actual ranks the SVDs
    # kept (= what the masks would be once the loose padding is removed; possibly non-minimal).
    raw_tucker, raw_tt = ranks.compute_raw_sweep_ranks(
        shape, tucker_mask.sum(axis=-1), tt_mask.sum(axis=-1), capped_tucker, capped_tt)
    n2 = int(np.max(raw_tucker))
    r2 = int(np.max(raw_tt))
    out_tucker = out_tucker[..., :n2, :]
    out_tt = out_tt[..., :r2, :n2, :r2]
    raw_masks = ut3_masking.make_uniform_masks(raw_tucker, raw_tt, n, r)
    new_masks = (raw_masks[0][..., :n2], raw_masks[1][..., :r2])
    return (out_tucker, out_tt, shape, new_masks), ss_tucker[..., :n2], ss_tt[..., :r2]


def ut3_rank_adjustment_sweep(
        data: UT3Data,
        direction: str = 'right_to_left',  # 'right_to_left' | 'left_to_right'
) -> UT3Data:
    """A single lossless directional sweep that drops structurally-redundant ranks -- the uniform analog
    of ragged ``rank_adjustment_sweep`` (the minimization step; :py:func:`ut3svd` does not minimize).

    ``'right_to_left'`` returns a **right-orthogonal** result; ``'left_to_right'`` a **left-orthogonal**
    one. It reaches the minimal ranks **only if the input is already orthogonal in the opposite
    direction** -- a :py:func:`ut3svd` result is left-orthogonal, so ``'right_to_left'`` minimizes it.
    (The precondition is required here: unlike ragged, the static masks shrink to the minimal ranks, so a
    wrong-direction call on a non-oppositely-orthogonal input is lossy. Compose both directions for a
    minimal result in a chosen gauge.)
    """
    if direction == 'right_to_left':
        return ut3_operations.ut3_reverse(_reduce_left_to_right(ut3_operations.ut3_reverse(data)))
    elif direction == 'left_to_right':
        return _reduce_left_to_right(data)
    raise ValueError("direction must be 'left_to_right' or 'right_to_left'; got %r" % (direction,))


def _reduce_left_to_right(data: UT3Data) -> UT3Data:
    """Single left-to-right reduction to minimal ranks, skipping orthogonalization (assumes the input is
    right-orthogonal). Lossless when that precondition holds. Shrinks the padded supercore to minimal."""
    masked_tucker, masked_tt = ut3_masking.apply_masks_to_cores(data)   # guards: masks must be host
    shape = data[2]                                                     # static int tuple
    tucker_mask, tt_mask = data[3]                                      # HOST bool rank masks
    n, r = masked_tucker.shape[-2], masked_tt.shape[-1]

    # ranks/masks on the host (np); compute_minimal_ranks defaults to numpy. See docs/uniform_pytree_composition.md.
    min_tucker, min_tt = ranks.compute_minimal_ranks(shape, tucker_mask.sum(axis=-1), tt_mask.sum(axis=-1))
    min_masks = ut3_masking.make_uniform_masks(min_tucker, min_tt, n, r)
    (out_tucker, out_tt), _, _ = uniform_t3_svd(
        (masked_tucker, masked_tt), min_masks, skip_orthogonalization=True)

    n2 = int(np.max(min_tucker))
    r2 = int(np.max(min_tt))
    out_tucker = out_tucker[..., :n2, :]
    out_tt = out_tt[..., :r2, :n2, :r2]
    new_masks = (min_masks[0][..., :n2], min_masks[1][..., :r2])
    return (out_tucker, out_tt, shape, new_masks)


def uniform_t3_svd(
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
    basis_masks, tt_masks = rank_truncation_masks
    ut3_masking.require_concrete_masks(basis_masks, tt_masks)  # masks (constant operands) are host

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
