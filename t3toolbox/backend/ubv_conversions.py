# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.orthogonal_representations as orth_reps
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.ubv_masking as ubv_masking
import t3toolbox.backend.ut3_masking as ut3_masking
from t3toolbox.backend.common import *

__all__ = [
    'ut3_orthogonal_representations',
    'ut3basis_to_t3basis',
]


def ut3_orthogonal_representations(
        data: typ.Tuple[
            NDArray,                          # tucker_supercore
            NDArray,                          # tt_supercore
            typ.Tuple[int, ...],              # shape
            typ.Tuple[NDArray, NDArray],      # (tucker_edge_mask, tt_edge_mask) -- the plain-UT3 rank masks
        ],
        already_left_orthogonal: bool = False,
        squash:                  bool = True,
) -> typ.Tuple[
    typ.Tuple[                                # frame .data:
        NDArray, NDArray, NDArray, NDArray,   #   up_sc, down_sc, left_sc, right_sc
        typ.Tuple[int, ...],                  #   shape
        typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (up, down, basis_left, basis_right) masks
    ],
    typ.Tuple[                                # variations .data:
        NDArray, NDArray,                     #   tucker_var_sc, tt_var_sc
        typ.Tuple[int, ...],                  #   shape
        typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (variations up, down, left, right) masks
    ],
]:
    '''Orthogonal (frame, variations) representation of a uniform Tucker tensor train, on raw ``.data``.

    Backend twin of the frontend ``ut3_orthogonal_representations`` (which wraps this into the OO
    ``UT3Basis`` / ``UT3Variations``). Takes a plain ``UniformTuckerTensorTrain.data`` and returns the
    **frame** and **variation** ``.data`` tuples (supercores + ``shape`` + the rank masks).

    WHY THIS IS A BACKEND FUNCTION (and not something to open-code): the output frame masks are **prefix**
    masks built from the orthogonal-representation *ranks* (``make_basis_masks`` = ``arange < rank``) --
    they assert the real orthonormal content sits in the **upper-left** ``[0, rank)`` slots of each
    supercore. That is correct ONLY because the orthogonalization is **SVD-based**: the SVD sorts content
    by singular value into the leading slots, with zeros / orthonormal completion trailing. A QR-based
    orthogonalization would scatter the real content across non-prefix positions and these masks would be
    WRONG -- see ``docs/uniform_svd_prefix_orthogonalization.md``. Building the masks any other way (e.g.
    from raw supercore magnitudes) is the easy mistake this function exists to prevent.

    The frame masks come from the orthogonal-representation ranks; the variation masks reuse the up/down
    masks and the basis left/right masks shifted by one (a variation occupies one TT slot, not a boundary
    edge -- hence ``left[:-1]`` / ``right[1:]``).
    '''
    tk_sc, tt_sc, shape, (tkm, ttm) = data
    masked_tk, masked_tt = ut3_masking.apply_masks_to_cores(data)   # zero the garbage before the SVD sweep

    # orth_reps.orthogonal_representations is polymorphic (accepts uniform supercores) and SVD-based.
    (uc, dc, lc, rc), (tkv, ttv) = orth_reps.orthogonal_representations(
        (masked_tk, masked_tt), already_left_orthogonal=already_left_orthogonal, squash=squash)

    up_ranks, down_ranks, left_ranks, right_ranks = ranks.compute_orthogonal_representation_ranks(
        shape, tkm.sum(axis=-1), ttm.sum(axis=-1))

    nU, nD, rL, rR = uc.shape[-2], dc.shape[-2], lc.shape[-1], rc.shape[-1]
    um, dm, lm, rm = ubv_masking.make_basis_masks(up_ranks, down_ranks, left_ranks, right_ranks, nU, nD, rL, rR)

    frame_data     = (uc, dc, lc, rc, shape, (um, dm, lm, rm))
    variation_data = (tkv, ttv, shape, (um, dm, lm[:-1], rm[1:]))
    return frame_data, variation_data


def ut3basis_to_t3basis(
        x: typ.Tuple[
            NDArray,                          # up_tucker_supercore
            NDArray,                          # down_tt_supercore
            NDArray,                          # left_tt_supercore
            NDArray,                          # right_tt_supercore
            typ.Tuple[int, ...],              # shape
            typ.Tuple[                        # (up_mask, down_mask, basis_left_mask, basis_right_mask)
                NDArray, NDArray, NDArray, NDArray,
            ],
        ],
) -> typ.Union[
    typ.Tuple[typ.Tuple[NDArray, ...], ...],  # (up_cores, down_cores, left_cores, right_cores), if unstacked
    typ.Tuple,                                # else a nested tree (shaped like stack_shape) of those
]:
    '''Convert a uniform UT3Basis ``.data`` to ragged ``T3Basis`` core-tuples (or a nested tree, if stacked).

    The physical mode dims are a contiguous prefix, so they slice ``[:Ni]`` (from the ``shape`` ints, no
    argwhere); only the *rank* masks scatter, so they are extracted with ``np.argwhere`` (HOST numpy --
    masks are host). The supercores may be jax; advanced-indexing them with the host int indices is fine.
    '''
    (up_supercore, down_supercore, left_supercore, right_supercore,
     shape, (up_mask, down_mask, basis_left_mask, basis_right_mask)) = x
    stack_shape = up_supercore.shape[1:-2]
    d = up_supercore.shape[0]

    if not stack_shape:  # unstacked -> one ragged (up, down, left, right) core set
        up_cores, down_cores, left_cores, right_cores = [], [], [], []
        for ind in range(d):
            up_inds   = np.argwhere(up_mask[ind]).reshape(-1)
            down_inds = np.argwhere(down_mask[ind]).reshape(-1)
            left_a    = np.argwhere(basis_left_mask[ind]).reshape(-1)
            left_b    = np.argwhere(basis_left_mask[ind + 1]).reshape(-1)
            right_a   = np.argwhere(basis_right_mask[ind]).reshape(-1)
            right_b   = np.argwhere(basis_right_mask[ind + 1]).reshape(-1)
            Ni = shape[ind]

            up_cores.append(   up_supercore[ind][up_inds, :][:, :Ni])
            down_cores.append( down_supercore[ind][left_a, :, :][:, down_inds, :][:, :, right_b])
            left_cores.append( left_supercore[ind][left_a, :, :][:, up_inds, :][:, :, left_b])
            right_cores.append(right_supercore[ind][right_a, :, :][:, up_inds, :][:, :, right_b])

        return tuple(up_cores), tuple(down_cores), tuple(left_cores), tuple(right_cores)

    all_T3Bs = []
    for ii in range(up_supercore.shape[1]):
        xi = (
            up_supercore[:, ii], down_supercore[:, ii], left_supercore[:, ii], right_supercore[:, ii],
            shape,
            (up_mask[:, ii], down_mask[:, ii], basis_left_mask[:, ii], basis_right_mask[:, ii]),
        )
        all_T3Bs.append(ut3basis_to_t3basis(xi))
    return tuple(all_T3Bs)
