# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.t3_operations as ragged_ops
import t3toolbox.backend.t3_orthogonalization as ragged_orth
import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.ranks as ranks
from t3toolbox.backend.common import *

__all__ = [
    't3svd',
]

def t3svd(
        x: typ.Tuple[
            typ.Tuple[NDArray,...], # tucker_cores
            typ.Tuple[NDArray,...], # tt_cores
        ],
        max_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        max_tucker_ranks:   typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
        minimize_ranks: bool = True,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[NDArray, ...],  # new_tucker_cores
        typ.Tuple[NDArray, ...],  # new_tt_cores
    ],
    typ.Tuple[NDArray,...], # Tucker singular values, len=d
    typ.Tuple[NDArray,...], # TT singular values, len=d+1
]:
    '''Compute (truncated) T3-SVD of TuckerTensorTrain.

    Implicit T3-SVD (Algorithm 10), Appendix A.2, of Alger et al. (2026), "Tucker Tensor Train
    Taylor Series" (arXiv:2603.21141).

    With ``minimize_ranks=True`` (default) the result has structurally minimal ranks: a hard rank cap
    can orphan a Tucker rank / bond (see :py:func:`_shrink_to_minimal_ranks`), which is then re-tightened
    away losslessly. ``minimize_ranks=False`` skips that re-tighten (an extra SVD sweep) and returns the
    raw sweep output -- the SAME represented tensor, but with the possibly-redundant ranks left in. The
    sweep is significant for large, lightly-compressed problems rounded repeatedly (e.g. ODE/iterative
    solvers); it is the caller's trade-off. See ``docs/t3svd_minimal_ranks.md``.
    '''
    num_cores = len(x[0])

    # Accept scalar or per-position max ranks (None entry = no cap at that position).
    max_tucker_ranks = ranks.normalize_max_ranks(max_tucker_ranks, num_cores)
    max_tt_ranks = ranks.normalize_max_ranks(max_tt_ranks, num_cores + 1)

    # print('0. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
    # print('0. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

    # make leading and trailing TT-ranks equal to 1
    x = (x[0], ragged_ops.squash_tt_tails(x[1]))

    # print('1. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
    # print('1. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

    # Orthogonalize Tucker matrices
    x = ragged_orth.down_orthogonalize_tucker_cores(x)

    # print('2. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
    # print('2. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

    # Right orthogonalize
    x = (x[0], orth.right_orthogonalize_tt_cores(x[1]))

    # print('3. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
    # print('3. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

    G0 = x[1][0]
    _, ss_first, _ = linalg.right_svd(G0)

    # Sweep left to right computing SVDS
    all_ss_tucker = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        max_rank = max_tucker_ranks[ii]
        # SVD inbetween TT core and Tucker core
        x, ss_tucker = ragged_orth.down_svd_tt_core(
            x, ii,
            max_rank=max_rank, rtol=rtol, atol=atol,
        )
        all_ss_tucker.append(ss_tucker)

        # print('4. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
        # print('4. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

        if ii < num_cores-1:
            max_rank = max_tt_ranks[ii+1]
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = ragged_orth.left_svd_tt_core(
                x, ii,
                max_rank=max_rank, rtol=rtol, atol=atol,
            )

            # print('5. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
            # print('5. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

        else:
            Gf = x[1][-1]
            _, ss_tt, _ = linalg.left_svd(Gf)
        all_ss_tt.append(ss_tt)

    # print('6. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
    # print('6. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

    # Re-tighten to minimal ranks. A hard rank cap can force a bond below its structural value
    # rL*n; that retroactively orphans the Tucker rank (or neighbouring bond) the sweep already
    # fixed against the pre-cap neighbour, leaving the result NON-minimal. The no-cap / tolerance
    # path is already minimal, so this is a no-op there. See docs/t3svd_minimal_ranks.md.
    # minimize_ranks=False skips it entirely (zero overhead; ranks may be non-minimal).
    if minimize_ranks:
        x, all_ss_tucker, all_ss_tt = _shrink_to_minimal_ranks(x, all_ss_tucker, all_ss_tt)

    return x, tuple(all_ss_tucker), tuple(all_ss_tt)


def _shrink_to_minimal_ranks(
        x: typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]],  # (tucker_cores, tt_cores)
        all_ss_tucker: typ.Sequence[NDArray],  # len=d,   Tucker singular values
        all_ss_tt:     typ.Sequence[NDArray],  # len=d+1, TT singular values
) -> typ.Tuple[
    typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]],  # re-tightened (tucker_cores, tt_cores)
    typ.Tuple[NDArray, ...],  # singular values sliced to the surviving Tucker ranks
    typ.Tuple[NDArray, ...],  # singular values sliced to the surviving TT ranks
]:
    '''Losslessly drop ranks left above the structural minimum by truncation (see :py:func:`t3svd`).

    A hard rank cap can shrink a bond below ``rL*n``, which orphans the Tucker rank / neighbouring
    bond the left-to-right sweep already fixed against the un-capped neighbour. A structural
    **right-to-left** re-tightening sweep (re-SVD each Tucker edge, then each bond, with no cap)
    removes exactly those orphaned directions: each edge unfolding has only as many rows as the
    surviving structure allows, so the SVD keeps the structural rank and discards no real content
    -- the represented tensor is unchanged. Already-minimal input is returned untouched, so the
    no-truncation and ``rtol``/``atol`` paths pay nothing.

    The returned singular values are the sweep's truncation singular values sliced to the surviving
    (minimal) ranks.
    '''
    num_cores = len(x[0])
    shape      = tuple(B.shape[-1] for B in x[0])
    cur_tucker = tuple(B.shape[-2] for B in x[0])
    cur_tt     = tuple([x[1][0].shape[-3]] + [G.shape[-1] for G in x[1]])

    min_tucker, min_tt = ranks.compute_minimal_ranks(shape, cur_tucker, cur_tt)
    min_tucker = tuple(int(n) for n in min_tucker)
    min_tt     = tuple(int(r) for r in min_tt)
    if min_tucker == cur_tucker and min_tt == cur_tt:
        return x, tuple(all_ss_tucker), tuple(all_ss_tt)

    for ii in range(num_cores - 1, -1, -1):
        x, _ = ragged_orth.down_svd_tt_core(x, ii)   # re-tighten Tucker rank n_ii against rL*rR
        if ii > 0:
            x, _ = ragged_orth.right_svd_tt_core(x, ii)  # re-tighten bond r_ii against n_ii*rR

    new_ss_tucker = tuple(s[..., :min_tucker[ii]] for ii, s in enumerate(all_ss_tucker))
    new_ss_tt     = tuple(s[..., :min_tt[ii]]     for ii, s in enumerate(all_ss_tt))
    return x, new_ss_tucker, new_ss_tt

