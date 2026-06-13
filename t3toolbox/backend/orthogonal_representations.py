# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.t3_operations as ragged_operations
import t3toolbox.backend.ut3_operations as uniform_operations
import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.t3_orthogonalization as ragged_orth
import t3toolbox.backend.ut3_orthogonalization as uniform_orth
from t3toolbox.backend.common import *

__all__ = [
    'orthogonal_representations',
    'basis_orthogonality_residual',
    'basis_consistency_residual',
]


def orthogonal_representations(
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
        squash: bool = True,
) -> typ.Union[
    typ.Tuple[
        typ.Tuple[
            typ.Tuple[NDArray,...], # up_tucker_cores
            typ.Tuple[NDArray, ...],  # down_tucker_cores
            typ.Tuple[NDArray,...], # left_tt_cores
            typ.Tuple[NDArray,...], # right_tucker_cores
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
    '''Construct base-variation representations of TuckerTensorTrain with orthogonal base.

    Sweeping orthogonalization (Algorithm 11) producing the representations (45)-(46), Appendix A.3,
    of Alger et al. (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141). NOTE: the
    left/right sweep order here differs from Algorithm 11 (left-then-right vs the paper's
    right-then-left); the resulting orthogonal representations are equivalent.
    '''
    is_uniform = is_ndarray(x[0])

    if is_uniform:
        squash_tails = lambda tk, tt: (tk, uniform_operations.uniform_squash_tt_tails(tt))
        up_orthogonalize_tucker_cores = lambda x, **kwargs: uniform_orth.up_orthogonalize_uniform_tucker_cores(*x, **kwargs)
        down_orthogonalize_tt_cores = lambda x, **kwargs: uniform_orth.down_orthogonalize_uniform_tt_cores(*x, **kwargs)
    else:
        squash_tails = lambda tk, tt: (tk, ragged_operations.squash_tt_tails(tt))
        up_orthogonalize_tucker_cores = ragged_orth.down_orthogonalize_tucker_cores
        down_orthogonalize_tt_cores = ragged_orth.up_orthogonalize_tt_cores

    if squash:
        x = squash_tails(*x)

    if not already_left_orthogonal:
        # Orthogonalize Tucker cores upward to get up_tt_cores U
        up_tucker_cores, tt_cores = up_orthogonalize_tucker_cores(x)

        # Sweep left-to-right, generating left orthogonal tt_cores L
        left_tt_cores = orth.left_orthogonalize_tt_cores(tt_cores)
    else:
        up_tucker_cores, left_tt_cores = x

    # Sweep right-to-left, generating tt_variations H, and right orthogonal tt_cores R
    right_tt_cores, tt_variations = orth.right_orthogonalize_tt_cores(
        left_tt_cores, return_variation_cores=True,
    )

    # Orthogonalize TT cores downward to get outer_tt_cores O and tucker_variations V
    tucker_variations, down_tt_cores = down_orthogonalize_tt_cores(
        (up_tucker_cores, tt_variations),
    )

    base = (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
    variation = (tucker_variations, tt_variations)
    return base, variation


def basis_orthogonality_residual(
        basis: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
) -> float:
    '''Max deviation from orthogonality of the four basis core families (over the whole stack).

    Checks each stacked block's gram against the identity:
        - up_tucker U_i (all i), outer/down D_i (all i),
        - left L_i (i=0..d-2), right R_i (i=1..d-1).
    The last left core and first right core are boundary remainders and are not checked. Returns the
    max absolute deviation; a caller thresholds it (``<= atol``) for a boolean orthogonality test.
    '''
    UU, DD, LL, RR = basis
    d = len(UU)

    def _dev(gram, n):
        return float(np.max(np.abs(np.asarray(gram) - np.eye(n))))

    resid = 0.0
    for ii in range(d):
        U = np.asarray(UU[ii])
        D = np.asarray(DD[ii])
        resid = max(resid, _dev(np.einsum('...io,...jo->...ij', U, U), U.shape[-2]))
        resid = max(resid, _dev(np.einsum('...iaj,...ibj->...ab', D, D), D.shape[-2]))
    for ii in range(d - 1):
        L = np.asarray(LL[ii])
        resid = max(resid, _dev(np.einsum('...iaj,...iak->...jk', L, L), L.shape[-1]))
    for ii in range(1, d):
        R = np.asarray(RR[ii])
        resid = max(resid, _dev(np.einsum('...iaj,...kaj->...ik', R, R), R.shape[-3]))
    return resid


def basis_consistency_residual(
        basis: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores
            typ.Sequence[NDArray],  # down_tt_cores
            typ.Sequence[NDArray],  # left_tt_cores
            typ.Sequence[NDArray],  # right_tt_cores
        ],
) -> float:
    '''Relative Frobenius mismatch between the left- and right-canonical reconstructions of the base
    point (``up`` over ``left`` vs ``up`` over ``right``).

    Returns ``||left - right|| / max(1, ||right||)`` over the dense tensors; a caller thresholds it
    (``<= rtol``) for a boolean consistency test. EXPENSIVE -- densifies both reconstructions.
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = basis
    left = to_numpy(ragged_operations.to_dense((up_tucker_cores, left_tt_cores)))
    right = to_numpy(ragged_operations.to_dense((up_tucker_cores, right_tt_cores)))
    return float(np.linalg.norm(left - right) / max(1.0, np.linalg.norm(right)))

