# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Raw-data (frame, variations) conversions for the ragged layer.

``t3_orthogonal_representations`` factors t3 data into the orthogonal frame + gauged variations
(data-level twin of the frontend function); ``fv_to_t3`` reconstructs the SINGLE term selected
by a frame/variations pairing -- not the tangent sum (that is ``tv_operations.tv_to_t3``;
same-looking near-twins, different math).
"""
import numpy as np
import typing as typ

from t3toolbox.backend.common import *
import t3toolbox.backend.tt_orthogonalization as orth
import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.t3_orthogonalization as ragged_orth
import t3toolbox.backend.ut3_orthogonalization as uniform_orth

from t3toolbox.backend.common import NDArray, is_ndarray, get_backend

__all__ = [
    'fv_to_t3',
    't3_orthogonal_representations',
    't3_corewise_frame',
]


def t3_corewise_frame(
        x_cores:  typ.Tuple,   # (tucker_cores, tt_cores)
) -> typ.Tuple:                # (U, O, P, Q) = (U, G, G, G); the raw cores ARE the frame
    '''The **corewise** frame of a T3: the Section 6.3 substitution ``(P, Q, O) -> G``.

    The counterpart of :py:func:`t3_orthogonal_representations` -- the two ways to make a frame from a
    point. Where the orthogonal representation builds an orthonormal frame with a gauge, this one just
    re-labels the cores, which is what makes the corewise geometry the over-parametrized Euclidean one
    (no gauge, additive retraction). Trivial to write and easy to write in the wrong slot order, so it
    is named here rather than open-coded.'''
    tucker_cores, tt_cores = x_cores
    return (tucker_cores, tt_cores, tt_cores, tt_cores)


def fv_to_t3(
        index: typ.Tuple[
            bool, # If True, use TT coordinate. If False, use Tucker coordinate
            int, # index of coordinate
        ],
        frame: typ.Union[
            typ.Tuple[
                typ.Tuple[NDArray, ...],  # up_tucker_cores
                typ.Tuple[NDArray, ...],  # down_tt_cores
                typ.Tuple[NDArray, ...],  # left_tt_cores
                typ.Tuple[NDArray, ...],  # right_tt_cores
            ], # ragged
            typ.Tuple[
                NDArray,  # up_tucker_supercore
                NDArray,  # down_tucker_supercore
                NDArray,  # left_tt_supercore
                NDArray,  # right_tucker_supercore
            ], # uniform
        ],
        variations: typ.Union[
            typ.Tuple[
                typ.Tuple[NDArray, ...],  # tucker_variations
                typ.Tuple[NDArray, ...],  # tt_variations
            ], # ragged
            typ.Tuple[
                NDArray,  # tucker_variations_supercore
                NDArray,  # tt_variations_supercore
            ], # uniform
        ],
) -> typ.Union[
    typ.Tuple[
        typ.Tuple[NDArray,...], # tucker_cores
        typ.Tuple[NDArray,...], # tt_cores
    ], # ragged
    typ.Tuple[
        NDArray, # tucker_supercore
        NDArray, # tt_supercore
    ], # uniform
]:
    '''Convert ith frame-variation representation to TuckerTensorTrain.
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    tucker_variations, tt_variations = variations

    is_uniform = is_ndarray(up_tucker_cores)
    xnp, _, _ = get_backend(True, tree_contains_jax((frame, variations)))

    use_tt_coord, ii = index

    if use_tt_coord:
        x_tucker_cores = up_tucker_cores

        LL = left_tt_cores[:ii]
        H = tt_variations[ii]
        RR = right_tt_cores[ii+1:]
        if is_uniform:
            x_tt_cores = xnp.concatenate([LL, H.reshape((1,)+H.shape), RR])
        else:
            x_tt_cores = tuple(LL) + (H,) + tuple(RR)
    else:
        left_UU = up_tucker_cores[:ii]
        V = tucker_variations[ii]
        right_UU = up_tucker_cores[ii+1:]
        if is_uniform:
            x_tucker_cores = xnp.concatenate([left_UU, V.reshape((1,)+V.shape), right_UU])
        else:
            x_tucker_cores = tuple(left_UU) + (V,) + tuple(right_UU)

        LL = left_tt_cores[:ii]
        D = down_tt_cores[ii]
        RR = right_tt_cores[ii+1:]
        if is_uniform:
            x_tt_cores = xnp.concatenate([LL, D.reshape((1,)+D.shape), RR])
        else:
            x_tt_cores = tuple(LL) + (D,) + tuple(RR)

    return x_tucker_cores, x_tt_cores


def t3_orthogonal_representations(
        x: typ.Union[
            typ.Tuple[
                typ.Tuple[NDArray,...], # tucker_cores
                typ.Tuple[NDArray,...], # tt_cores
            ], # ragged
            typ.Tuple[
                NDArray, # tucker_supercore
                NDArray, # tt_supercore
            ], # uniform
        ],
        already_left_orthogonal: bool = False,
        squash_tails: bool = True,

        uniform_masks: typ.Optional[typ.Tuple[typ.Sequence[int], NDArray, NDArray]] = None,
                       # uniform only: (shape, tucker_edge_mask, tt_edge_mask) -- HOST static ints +
                       # bool rank masks of the input .data. Given -> every SVD in the sweep is
                       # PAD-SAFE (linalg.pad_safe_svd) with the mask recurrences threaded per step.
) -> typ.Union[
    typ.Tuple[
        typ.Tuple[
            typ.Tuple[NDArray,...], # up_tucker_cores
            typ.Tuple[NDArray, ...],  # down_tt_cores
            typ.Tuple[NDArray,...], # left_tt_cores
            typ.Tuple[NDArray,...], # right_tt_cores
        ],
        typ.Tuple[
            typ.Tuple[NDArray,...], # tucker_variations
            typ.Tuple[NDArray,...], # tt_variations
        ],
    ], # ragged
    typ.Tuple[
        typ.Tuple[
            NDArray,  # up_tucker_supercore
            NDArray,  # down_tucker_supercore
            NDArray,  # left_tt_supercore
            NDArray,  # right_tucker_supercore
        ],
        typ.Tuple[
            NDArray,  # tucker_variations_supercore
            NDArray,  # tt_variations_supercore
        ],
    ],  # uniform
]:
    '''Construct frame-variation representations of TuckerTensorTrain with orthogonal frame.

    Sweeping orthogonalization (Algorithm 11) producing the representations (45)-(46), Appendix A.3,
    of Alger et al. (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141). NOTE: the
    left/right sweep order here differs from Algorithm 11 (left-then-right vs the paper's
    right-then-left); the resulting orthogonal representations are equivalent.
    '''
    is_uniform = is_ndarray(x[0])

    if uniform_masks is not None and not is_uniform:
        raise ValueError('uniform_masks is only meaningful for uniform (supercore) input')

    if is_uniform:
        # uniform path operates on bare (masked) supercores -- the (n,N)/(rL,n,rR) arrays, not .data.
        up_orthogonalize_tucker_cores = lambda x: uniform_orth.down_orthogonalize_tucker_supercores(*x)
        down_orthogonalize_tt_cores = lambda x: uniform_orth.up_orthogonalize_tt_supercores(*x)
    else:
        up_orthogonalize_tucker_cores = ragged_orth.t3_down_orthogonalize_tucker_cores
        down_orthogonalize_tt_cores = ragged_orth.t3_up_orthogonalize_tt_cores

    if squash_tails:
        # tt_squash_tails is polymorphic over the representation (ragged core tuple / supercore)
        x = (x[0], tt_operations.tt_squash_tails(x[1]))

    if uniform_masks is not None:
        # The pad-safe uniform sweep (review S1b): every SVD below goes through pad_safe_svd, with
        # the per-site masks precomputed HERE (host numpy) by the same recurrences the frame masks
        # are built from (ranks.compute_orthogonal_representation_ranks; see that docstring for the
        # order). Each site also zeroes its don't-care completion columns (out masks), so the chain
        # stays bitwise-clean and every downstream pad-safe SVD sees genuinely zero padding.
        shape_u, tkm_u, ttm_u = uniform_masks
        d_u = tkm_u.shape[0]
        stack_u = tkm_u.shape[1:-1]
        r_u = ttm_u.shape[-1]
        N_u = x[0].shape[-1]
        ttm_sq = ttm_u
        if squash_tails:                       # squashing makes the boundary bond real rank exactly 1
            one_r = prefix_mask(np.ones((1,) + stack_u, dtype=int), r_u)
            ttm_sq = np.concatenate([one_r, ttm_u[1:-1], one_r], axis=0)
        shape_col = np.asarray(shape_u).reshape((d_u,) + (1,) * len(stack_u))
        up_r = np.minimum(tkm_u.sum(axis=-1), shape_col)               # 1. Tucker down-orth
        bond_r = ttm_sq.sum(axis=-1)
        left_r = np.array(np.broadcast_to(bond_r, bond_r.shape), dtype=int)
        for i in range(d_u - 1):                                       # 2. left sweep, original bonds
            left_r[i + 1] = np.minimum(left_r[i] * up_r[i], bond_r[i + 1])
        right_r = np.array(left_r, copy=True)
        for i in range(d_u - 1, 0, -1):                                # 3. right sweep, left chain
            right_r[i] = np.minimum(left_r[i], up_r[i] * right_r[i + 1])
        down_r = np.minimum(up_r, left_r[:-1] * right_r[1:])           # 4. the down step

    if not already_left_orthogonal:
        if uniform_masks is None:
            # Orthogonalize Tucker cores upward to get up_tt_cores U
            up_tucker_cores, tt_cores = up_orthogonalize_tucker_cores(x)

            # Sweep left-to-right, generating left orthogonal tt_cores L
            left_tt_cores = orth.tt_left_orthogonalize(tt_cores)
        else:
            up_tucker_cores, tt_cores = uniform_orth.down_orthogonalize_tucker_supercores(
                x[0], x[1],
                row_mask=prefix_mask(shape_col, N_u), col_mask=tkm_u,
                out_mask=prefix_mask(up_r, min(tkm_u.shape[-1], N_u)))
            up_masks = prefix_mask(up_r, up_tucker_cores.shape[-2])
            left_tt_cores = orth.tt_left_orthogonalize(
                tt_cores, pad_masks=uniform_orth._tt_left_sweep_pad_masks(ttm_sq, up_masks, left_r))
    else:
        up_tucker_cores, left_tt_cores = x

    if uniform_masks is not None:
        up_masks = prefix_mask(up_r, up_tucker_cores.shape[-2])

    # Sweep right-to-left, generating tt_variations H, and right orthogonal tt_cores R
    if uniform_masks is None:
        right_tt_cores, tt_variations = orth.tt_right_orthogonalize(
            left_tt_cores, return_variation_cores=True,
        )
    else:
        left_masks = prefix_mask(left_r, left_tt_cores.shape[-1])
        right_tt_cores, tt_variations = orth.tt_right_orthogonalize(   # executed as a LEFT sweep on
            left_tt_cores, return_variation_cores=True,                # the reversed chain
            pad_masks=uniform_orth._tt_left_sweep_pad_masks(
                left_masks[::-1], up_masks[::-1], right_r[::-1]),
        )

    # Orthogonalize TT cores downward to get outer_tt_cores O and tucker_variations V
    if uniform_masks is None:
        tucker_variations, down_tt_cores = down_orthogonalize_tt_cores(
            (up_tucker_cores, tt_variations),
        )
    else:
        r_now = tt_variations.shape[-1]
        kron_ab = (prefix_mask(left_r[:-1], r_now)[..., :, None]
                   & prefix_mask(right_r[1:], r_now)[..., None, :])    # (d,)+stack+(rL,rR)
        tucker_variations, down_tt_cores = uniform_orth.up_orthogonalize_tt_supercores(
            up_tucker_cores, tt_variations,
            row_mask=kron_ab.reshape(kron_ab.shape[:-2] + (-1,)), col_mask=up_masks,
            out_mask=prefix_mask(down_r, min(r_now * r_now, up_tucker_cores.shape[-2])))

    frame = (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
    variation = (tucker_variations, tt_variations)
    return frame, variation
