# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.t3_operations as ragged_ops
import t3toolbox.backend.t3_orthogonalization as ragged_orth
import t3toolbox.backend.tt_orthogonalization as orth
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.ranks as ranks
from t3toolbox.backend.common import *

__all__ = [
    't3svd',
    'rank_adjustment_sweep',
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
        assume_orthogonal: bool = False,
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
    Taylor Series" (arXiv:2603.21141) -- the basic algorithm, analogous to Oseledets' TT-SVD.

    Orthogonalize, then a single left-to-right truncating sweep. The result is **always left-orthogonal**.
    It is **not** re-tuned to minimal ranks: a hard rank cap can leave a Tucker rank / bond above its
    structural minimum (non-minimal), exactly as the paper's algorithm does. To reduce to minimal ranks,
    follow with :py:func:`rank_adjustment_sweep` -- the output is left-orthogonal, so
    ``rank_adjustment_sweep(x, 'right_to_left')`` minimizes it; check with
    ``TuckerTensorTrain.has_minimal_ranks``. See ``docs/t3svd_minimal_ranks.md``.

    ``assume_orthogonal=True`` skips the initial orthogonalization, asserting the input is already
    **right-orthogonal** (Tucker down-orthogonal + TT right-orthogonal -- the form the L->R sweep needs).
    **Not enforced** (verify with ``TuckerTensorTrain.is_right_orthogonal``). A left-orthogonal input must
    be reversed by the caller (a left-orthogonal T3 reversed is right-orthogonal).
    '''
    num_cores = len(x[0])

    # Accept scalar or per-position max ranks (None entry = no cap at that position).
    max_tucker_ranks = ranks.normalize_max_ranks(max_tucker_ranks, num_cores)
    max_tt_ranks = ranks.normalize_max_ranks(max_tt_ranks, num_cores + 1)

    # make leading and trailing TT-ranks equal to 1 (no-op when already 1, i.e. for orthogonal input)
    x = (x[0], tt_operations.tt_squash_tails(x[1]))

    # Orthogonalize (Tucker down-orthogonal, TT right-orthogonal) -- skipped if asserted right-orthogonal
    if not assume_orthogonal:
        x = ragged_orth.down_orthogonalize_tucker_cores(x)
        x = (x[0], orth.tt_right_orthogonalize(x[1]))

    G0 = x[1][0]
    _, ss_first, _ = linalg.right_svd(G0)

    # Single left-to-right truncating sweep (-> left-orthogonal). No minimal-rank re-tuning.
    all_ss_tucker = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        x, ss_tucker = ragged_orth.down_svd_tt_core(  # SVD between TT core and Tucker core
            x, ii, max_rank=max_tucker_ranks[ii], rtol=rtol, atol=atol)
        all_ss_tucker.append(ss_tucker)

        if ii < num_cores-1:
            x, ss_tt = ragged_orth.left_svd_tt_core(   # SVD between ith and (i+1)th TT core
                x, ii, max_rank=max_tt_ranks[ii+1], rtol=rtol, atol=atol)
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = linalg.left_svd(Gf)
        all_ss_tt.append(ss_tt)

    return x, tuple(all_ss_tucker), tuple(all_ss_tt)


def rank_adjustment_sweep(
        x: typ.Tuple[
            typ.Tuple[NDArray, ...],  # tucker_cores
            typ.Tuple[NDArray, ...],  # tt_cores
        ],
        direction: str = 'right_to_left',  # 'right_to_left' | 'left_to_right'
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores
    typ.Tuple[NDArray, ...],  # tt_cores
]:
    '''A single lossless directional sweep that drops structurally-redundant ranks (re-SVD each Tucker
    edge and TT bond with **no cap**). The represented tensor is unchanged.

    ``'right_to_left'`` produces a **right-orthogonal** result; ``'left_to_right'`` a **left-orthogonal**
    one. A single sweep reaches the minimal ranks **only if the input is already orthogonal in the
    opposite direction** -- e.g. a :py:func:`t3svd` result is left-orthogonal, so
    ``rank_adjustment_sweep(result, 'right_to_left')`` minimizes it (verify with ``has_minimal_ranks``).
    On a general input it is a partial reduction; compose both directions for guaranteed minimal ranks.
    '''
    x = (x[0], tt_operations.tt_squash_tails(x[1]))
    num_cores = len(x[0])
    if direction == 'right_to_left':
        for ii in range(num_cores - 1, -1, -1):
            x, _ = ragged_orth.down_svd_tt_core(x, ii)        # Tucker: drop n above rL*rR
            if ii > 0:
                x, _ = ragged_orth.right_svd_tt_core(x, ii)   # bond: drop r above n*rR (push left)
    elif direction == 'left_to_right':
        for ii in range(num_cores):
            x, _ = ragged_orth.down_svd_tt_core(x, ii)        # Tucker
            if ii < num_cores - 1:
                x, _ = ragged_orth.left_svd_tt_core(x, ii)    # bond (push right)
    else:
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'; got %r" % (direction,))
    return x

