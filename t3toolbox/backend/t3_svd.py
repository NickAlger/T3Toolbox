# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import math
import numpy as np
import typing as typ

import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.t3_operations as ragged_ops
import t3toolbox.backend.t3_orthogonalization as ragged_orth
import t3toolbox.backend.tt_orthogonalization as orth
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.common as common
from t3toolbox.backend.common import *

__all__ = [
    't3svd',
    't3_rank_adjustment_sweep',
    'dense_tucker_svd',
    'dense_ttsvd',
    'dense_t3svd',
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
    follow with :py:func:`t3_rank_adjustment_sweep` -- the output is left-orthogonal, so
    ``t3_rank_adjustment_sweep(x, 'right_to_left')`` minimizes it; check with
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
        x = ragged_orth.t3_down_orthogonalize_tucker_cores(x)
        x = (x[0], orth.tt_right_orthogonalize(x[1]))

    G0 = x[1][0]
    _, ss_first, _ = linalg.right_svd(G0)

    # Single left-to-right truncating sweep (-> left-orthogonal). No minimal-rank re-tuning.
    all_ss_tucker = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        x, ss_tucker = ragged_orth.t3_down_svd_tt_core(  # SVD between TT core and Tucker core
            x, ii, max_rank=max_tucker_ranks[ii], rtol=rtol, atol=atol)
        all_ss_tucker.append(ss_tucker)

        if ii < num_cores-1:
            x, ss_tt = ragged_orth.t3_left_svd_tt_core(   # SVD between ith and (i+1)th TT core
                x, ii, max_rank=max_tt_ranks[ii+1], rtol=rtol, atol=atol)
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = linalg.left_svd(Gf)
        all_ss_tt.append(ss_tt)

    return x, tuple(all_ss_tucker), tuple(all_ss_tt)


def t3_rank_adjustment_sweep(
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
    ``t3_rank_adjustment_sweep(result, 'right_to_left')`` minimizes it (verify with ``has_minimal_ranks``).
    On a general input it is a partial reduction; compose both directions for guaranteed minimal ranks.
    '''
    x = (x[0], tt_operations.tt_squash_tails(x[1]))
    num_cores = len(x[0])
    if direction == 'right_to_left':
        for ii in range(num_cores - 1, -1, -1):
            x, _ = ragged_orth.t3_down_svd_tt_core(x, ii)        # Tucker: drop n above rL*rR
            if ii > 0:
                x, _ = ragged_orth.t3_right_svd_tt_core(x, ii)   # bond: drop r above n*rR (push left)
    elif direction == 'left_to_right':
        for ii in range(num_cores):
            x, _ = ragged_orth.t3_down_svd_tt_core(x, ii)        # Tucker
            if ii < num_cores - 1:
                x, _ = ragged_orth.t3_left_svd_tt_core(x, ii)    # bond (push right)
    else:
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'; got %r" % (direction,))
    return x


def dense_tucker_svd(
        T: common.NDArray, # shape=(N1, N2, .., Nd)
        min_ranks:  typ.Sequence[int] = None, # len=d
        max_ranks:  typ.Sequence[int] = None,  # len=d
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[common.NDArray,...], # Tucker bases, ith_elm_shape=(ni, Ni)
        common.NDArray, # Tucker core, shape=(n1,n2,...,nd)
    ],
    typ.Tuple[common.NDArray,...], # singular values of matricizations
]:
    '''Compute Tucker decomposition and matricization singular values for dense tensor.

    Parameters
    ----------
    T: NDArray
        The dense tensor. shape=(N1, ..., Nd)
    min_ranks: typ.Sequence[int]
        Minimum Tucker ranks for truncation. len=d
    max_ranks: typ.Sequence[int]
        Maximum Tucker ranks for truncation. len=d
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[typ.Tuple[NDArray,...],NDArray]
        Tucker decomposition (tucker_bases, tucker_core). tucker_bases[ii].shape=(ni,Ni). tucker_core.shape=(n1,...,nd)
    typ.Tuple[NDArray,...]
        Singular values of matricizations

    See Also
    --------
    truncated_svd
    tt_svd_dense
    t3_svd_dense
    t3_svd

    Examples
    --------
    No truncation -- a lossless Tucker decomposition. The factors reconstruct ``T``, and
    ``bases[ii].shape = (ni, Ni)`` (the small rank ``ni`` first), one singular-value vector per mode:

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_svd as t3_svd
    >>> np.random.seed(0)
    >>> T = np.random.randn(5, 6, 7)
    >>> (bases, core), ss = t3_svd.dense_tucker_svd(T)
    >>> print(core.shape, [B.shape for B in bases])
    (5, 6, 7) [(5, 5), (6, 6), (7, 7)]
    >>> T2 = np.einsum('abc,ai,bj,ck->ijk', core, bases[0], bases[1], bases[2])
    >>> print(np.allclose(T, T2))                          # exact reconstruction
    True

    The mode-``i`` singular values ARE the singular values of the mode-``i`` matricization (shown for
    mode 0; the other modes are analogous):

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_svd as t3_svd
    >>> np.random.seed(0)
    >>> T = np.random.randn(5, 6, 7)
    >>> _, ss = t3_svd.dense_tucker_svd(T)
    >>> dense_svals = np.linalg.svd(T.reshape(5, 6 * 7), compute_uv=False)   # mode-0 matricization
    >>> print(np.allclose(ss[0], dense_svals[:len(ss[0])]))
    True

    Truncation -- a smooth tensor has gradually decaying matricization spectra, so ``rtol`` truncates
    meaningfully (a sharp random spectrum would not):

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_svd as t3_svd
    >>> i, j, k = np.ogrid[1:9, 1:9, 1:9]
    >>> T = 1.0 / (i + j + k)                              # graded-spectrum tensor
    >>> (bases_f, _), ss_full = t3_svd.dense_tucker_svd(T)         # full (untruncated) spectra
    >>> (bases, core), _ = t3_svd.dense_tucker_svd(T, rtol=1e-3)   # truncate at rtol
    >>> print(tuple(B.shape[0] for B in bases_f), '->', tuple(B.shape[0] for B in bases))
    (8, 8, 8) -> (3, 3, 3)
    >>> T2 = np.einsum('abc,ai,bj,ck->ijk', core, bases[0], bases[1], bases[2])
    >>> ranks = tuple(B.shape[0] for B in bases)
    >>> dropped_sq = sum(float(np.sum(s[r:]**2)) for s, r in zip(ss_full, ranks))
    >>> print(bool(np.linalg.norm(T - T2) <= np.sqrt(dropped_sq)))  # accuracy bound [Oseledets]
    True
    '''
    bases = []
    singular_values_of_matricizations = []
    C = T
    for ii in range(len(T.shape)):
        C_swap = C.swapaxes(ii,0)
        old_shape_swap = C_swap.shape

        min_rank = None if min_ranks is None else min_ranks[ii]
        max_rank = None if max_ranks is None else max_ranks[ii]

        C_swap_mat = C_swap.reshape((old_shape_swap[0], -1))
        U, ss, Vt = linalg.truncated_svd(
            C_swap_mat, min_rank, max_rank, rtol, atol,
        )
        rM_new = len(ss)

        singular_values_of_matricizations.append(ss)
        bases.append(U.T)
        C_swap = (ss.reshape((-1,1)) * Vt).reshape((rM_new,) + old_shape_swap[1:])
        C = C_swap.swapaxes(0, ii)

    return (tuple(bases), C), tuple(singular_values_of_matricizations)


def dense_ttsvd(
        T: common.NDArray,  # shape=(N0,...,N(d-1))
        min_ranks:  typ.Sequence[int] = None, # len=d+1
        max_ranks:  typ.Sequence[int] = None,  # len=d+1
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[common.NDArray,...], # tt_cores
    typ.Tuple[common.NDArray,...], # singular values of unfoldings
]:
    '''Compute tensor train (TT) decomposition and unfolding singular values for dense tensor.

    Parameters
    ----------
    T: NDArray
        The dense tensor. shape=(N1, ..., Nd)
    min_ranks: typ.Sequence[int]
        Minimum TT-ranks for truncation. len=d+1. e.g., (1,3,3,3,1)
    max_ranks: typ.Sequence[int]
        Maximum TT-ranks for truncation. len=d+1. e.g., (1,5,5,5,1)
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        TT cores. len=d. elm_shape=(ri, ni, r(i+1))
    typ.Tuple[NDArray,...]
        Singular values of unfoldings. len=d+1. elm_shape=(ri,)

    See Also
    --------
    truncated_svd
    dense_tucker_svd
    t3_svd_dense
    t3_svd

    Examples
    --------
    No truncation -- a lossless TT decomposition. The cores reconstruct ``T``; there are ``d`` cores
    and ``d+1`` singular-value vectors (the two boundary entries are both just ``||T||``):

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_svd as t3_svd
    >>> np.random.seed(0)
    >>> T = np.random.randn(5, 6, 7)
    >>> cores, ss = t3_svd.dense_ttsvd(T)
    >>> print([G.shape for G in cores])               # cores[i].shape = (ri, ni, r(i+1))
    [(1, 5, 5), (5, 6, 7), (7, 7, 1)]
    >>> T2 = np.einsum('aib,bjc,ckd->ijk', cores[0], cores[1], cores[2])
    >>> print(np.allclose(T, T2))                     # exact reconstruction
    True
    >>> print(len(ss), np.allclose(ss[0], np.linalg.norm(T)), np.allclose(ss[-1], np.linalg.norm(T)))
    4 True True

    The internal singular values ARE the singular values of the matrix unfoldings (shown for the first
    unfolding; the others are analogous):

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_svd as t3_svd
    >>> np.random.seed(0)
    >>> T = np.random.randn(5, 6, 7)
    >>> _, ss = t3_svd.dense_ttsvd(T)
    >>> dense_svals = np.linalg.svd(T.reshape(5, 6 * 7), compute_uv=False)   # first unfolding
    >>> print(np.allclose(ss[1], dense_svals[:len(ss[1])]))
    True

    Truncation -- a smooth tensor has gradually decaying unfolding spectra, so ``rtol`` truncates
    meaningfully (a sharp random spectrum would not):

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_svd as t3_svd
    >>> i, j, k = np.ogrid[1:9, 1:9, 1:9]
    >>> T = 1.0 / (i + j + k)                          # graded-spectrum tensor
    >>> cores_f, ss_full = t3_svd.dense_ttsvd(T)           # full (untruncated) spectra
    >>> cores, ss = t3_svd.dense_ttsvd(T, rtol=1e-3)       # truncate at rtol
    >>> full_ranks = tuple(G.shape[0] for G in cores_f) + (1,)
    >>> tt_ranks = tuple(G.shape[0] for G in cores) + (1,)
    >>> print(full_ranks, '->', tt_ranks)
    (1, 8, 8, 1) -> (1, 3, 3, 1)
    >>> T2 = np.einsum('aib,bjc,ckd->ijk', cores[0], cores[1], cores[2])
    >>> dropped_sq = sum(float(np.sum(s[r:]**2)) for s, r in zip(ss_full[1:-1], tt_ranks[1:-1]))
    >>> print(bool(np.linalg.norm(T - T2) <= np.sqrt(dropped_sq)))  # accuracy bound [Oseledets]
    True
    '''
    use_jax = common.is_jax_ndarray(T)
    xnp, xmap, xscan = common.get_backend(False, use_jax)

    #
    nn = T.shape

    X = T.reshape((1,) + T.shape)
    singular_values_of_unfoldings = []
    cores = []
    for ii in range(len(nn)-1):
        rL = X.shape[0]

        min_rank = None if min_ranks is None else min_ranks[ii+1]
        max_rank = None if max_ranks is None else max_ranks[ii+1]

        U, ss, Vt = linalg.truncated_svd(
            X.reshape((rL * nn[ii], -1)), min_rank, max_rank, rtol, atol,
        )
        rR = len(ss)

        singular_values_of_unfoldings.append(ss)
        cores.append(U.reshape((rL, nn[ii], rR)))
        X = ss.reshape((-1,1)) * Vt
    cores.append(X.reshape(X.shape + (1,)))

    norm_T_vec = xnp.array([xnp.linalg.norm(T)])
    singular_values_of_unfoldings = [norm_T_vec,] + singular_values_of_unfoldings + [norm_T_vec,]

    return tuple(cores), tuple(singular_values_of_unfoldings)


def dense_t3svd(
        T: common.NDArray, # shape=stack_shape+(N0, .., N(d-1))
        stack_shape: typ.Sequence[int] = (),
        max_tucker_ranks:  typ.Sequence[int] = None,  # len=d
        max_tt_ranks:  typ.Sequence[int] = None,  # len=d+1
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[common.NDArray,...], # tucker_cores
        typ.Tuple[common.NDArray,...], # tt_cores
    ], # Approximation of T by Tucker tensor train
    typ.Tuple[common.NDArray,...], # Tucker singular values, len=d
    typ.Tuple[common.NDArray,...], # TT singular values, len=d+1
]:
    '''Compute TuckerTensorTrain and edge singular values for dense tensor.

    Examples
    --------
    No truncation -- a lossless T3 (Tucker + tensor-train) decomposition. Each Tucker basis ``B`` is
    contracted into its TT core ``G`` to rebuild the dense tensor:

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_svd as t3_svd
    >>> np.random.seed(0)
    >>> T = np.random.randn(5, 6, 7)
    >>> (tucker_cores, tt_cores), ss_tucker, ss_tt = t3_svd.dense_t3svd(T)
    >>> print([B.shape for B in tucker_cores], [G.shape for G in tt_cores])
    [(5, 5), (6, 6), (7, 7)] [(1, 5, 5), (5, 6, 7), (7, 7, 1)]
    >>> GG_big = [np.einsum('io,aib->aob', B, G) for B, G in zip(tucker_cores, tt_cores)]
    >>> T2 = np.einsum('aib,bjc,ckd->ijk', *GG_big)
    >>> print(np.allclose(T, T2))                      # exact reconstruction
    True
    >>> print(len(ss_tucker), len(ss_tt))              # d Tucker spectra, d+1 TT spectra
    3 4

    Stacked -- a leading ``stack_shape`` rides along on every core; the decomposition is vectorized
    over the stack:

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_svd as t3_svd
    >>> np.random.seed(0)
    >>> T = np.random.randn(2, 3, 5, 6, 7)
    >>> (tucker_cores, tt_cores), _, _ = t3_svd.dense_t3svd(T, stack_shape=(2, 3))
    >>> print([B.shape for B in tucker_cores])         # stack_shape=(2,3) prefixes each core
    [(2, 3, 5, 5), (2, 3, 6, 6), (2, 3, 7, 7)]
    >>> GG_big = [np.einsum('...io,...aib->...aob', B, G) for B, G in zip(tucker_cores, tt_cores)]
    >>> T2 = np.einsum('...aib,...bjc,...ckd->...ijk', *GG_big)
    >>> print(np.allclose(T, T2))
    True

    Truncation -- a smooth tensor has gradually decaying spectra, so ``rtol`` truncates meaningfully
    (a sharp random spectrum would not):

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_svd as t3_svd
    >>> i, j, k = np.ogrid[1:9, 1:9, 1:9]
    >>> T = 1.0 / (i + j + k)                          # graded-spectrum tensor
    >>> (tk_f, tt_f), ss_tk_full, ss_tt_full = t3_svd.dense_t3svd(T)        # full spectra
    >>> (tk, tt), _, _ = t3_svd.dense_t3svd(T, rtol=1e-3)                   # truncate at rtol
    >>> tucker_ranks = tuple(B.shape[0] for B in tk)
    >>> tt_ranks = tuple(G.shape[0] for G in tt) + (1,)
    >>> print(tuple(B.shape[0] for B in tk_f), '->', tucker_ranks)         # Tucker ranks drop
    (8, 8, 8) -> (3, 3, 3)
    >>> print(tuple(G.shape[0] for G in tt_f) + (1,), '->', tt_ranks)      # TT ranks drop
    (1, 8, 8, 1) -> (1, 3, 3, 1)
    >>> GG_big = [np.einsum('io,aib->aob', B, G) for B, G in zip(tk, tt)]
    >>> T2 = np.einsum('aib,bjc,ckd->ijk', *GG_big)
    >>> dropped_sq = (sum(float(np.sum(s[r:]**2)) for s, r in zip(ss_tt_full, tt_ranks))
    ...             + sum(float(np.sum(s[r:]**2)) for s, r in zip(ss_tk_full, tucker_ranks)))
    >>> print(bool(np.linalg.norm(T - T2) <= np.sqrt(dropped_sq)))  # accuracy bound [Oseledets]
    True

    Tolerances need a single (unstacked) tensor -- ``rtol``/``atol`` with a non-empty ``stack_shape``
    raise, since different slices could truncate to different ranks (use ``max_*_ranks`` instead):

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_svd as t3_svd
    >>> np.random.seed(0)
    >>> T = np.random.randn(2, 3, 5, 6, 7)
    >>> t3_svd.dense_t3svd(T, stack_shape=(2, 3), rtol=1e-3)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError
    '''
    use_jax = common.is_jax_ndarray(T)
    xnp, xmap, xscan = common.get_backend(False, use_jax)

    #
    shape = T.shape[len(stack_shape):]

    # Accept scalar or per-position max ranks (None entry = no cap at that position).
    max_tucker_ranks    = ranks.normalize_max_ranks(max_tucker_ranks, len(shape))
    max_tt_ranks        = ranks.normalize_max_ranks(max_tt_ranks, len(shape)+1)

    ss_tt0 = xnp.linalg.norm(T.reshape(stack_shape+(-1,)), axis=-1).reshape(stack_shape + (1,))

    max_tt_ranks = list(max_tt_ranks)[1:]
    max_tucker_ranks = list(max_tucker_ranks)

    T = T.reshape(stack_shape + (1,) + shape)

    tucker_cores = []
    tt_cores = []
    ss_tucker = []
    ss_tt = [ss_tt0]
    while len(T.shape) > len(stack_shape)+1:
        rL = T.shape[len(stack_shape)]
        N = T.shape[len(stack_shape)+1]
        mm = T.shape[len(stack_shape)+2:]
        M = math.prod(mm)
        A = T.reshape(stack_shape + (rL, N, M)).swapaxes(-3, -2)
        A = A.reshape(stack_shape+(N, rL*M))

        U, ss, Vt = linalg.truncated_svd(A, max_rank=max_tucker_ranks[0], rtol=rtol, atol=atol)
        max_tucker_ranks = max_tucker_ranks[1:]
        n = ss.shape[-1]

        tucker_cores.append(U.swapaxes(-2,-1).copy())
        ss_tucker.append(ss)

        T = xnp.einsum(
            '...n,...nx->...nx', ss, Vt
        ).reshape(stack_shape + (n, rL, M)).swapaxes(-3, -2) # shape=stack_shape+(rL, n, M)

        A = T.reshape(stack_shape + (rL*n, M))
        U, ss, Vt = linalg.truncated_svd(A, max_rank=max_tt_ranks[0], rtol=rtol, atol=atol)
        max_tt_ranks = max_tt_ranks[1:]
        rR = ss.shape[-1]

        G = U.reshape(stack_shape + (rL, n, rR))
        tt_cores.append(G)
        ss_tt.append(ss)

        T = xnp.einsum('...r,...rx->...rx', ss, Vt).reshape(stack_shape + (rR,) + mm)

    Gf = tt_cores[-1]

    Gf = xnp.einsum('...aib,...b->...aib', Gf, T)
    tt_cores[-1] = Gf

    return (tuple(tucker_cores), tuple(tt_cores)), tuple(ss_tucker), tuple(ss_tt)
