# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Rank bookkeeping: minimal/continuation ranks, manifold dimension, rank-spec normalization.

Mostly cheap structural integer arithmetic -- bare ``minimal_ranks`` means the structural notion,
per the naming convention; the SVD-based ``numerically_minimal`` checkers live with their
operations -- plus edge condition-number diagnostics.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.linalg as linalg
from t3toolbox.backend.common import *

__all__ = [
    'compute_minimal_ranks',
    'compute_continuation_ranks',
    'edge_condition_numbers',
    'compute_raw_sweep_ranks',
    'compute_orthogonal_representation_ranks',
    'compute_manifold_dim',
    'frame_has_minimal_ranks',
    'normalize_max_ranks',
]


def normalize_max_ranks(
        spec,            # None | int | Sequence[int or None]
        length: int,     # d for Tucker ranks, d+1 for TT ranks
) -> typ.Tuple:          # length-`length` tuple of (int or None); None entry = no cap at that position
    '''Normalize a max-rank specification to a per-position tuple.

    ``None`` -> no cap anywhere; a scalar caps every position uniformly; a sequence is per-position
    (length-checked). Shared by :py:func:`t3svd` and the elementwise-multiply (``t3m``) backends so a
    scalar like ``max_tt_ranks=4`` works the same everywhere.
    '''
    if spec is None:
        return (None,) * length
    if isinstance(spec, (int, np.integer)):
        return (int(spec),) * length
    spec = tuple(spec)
    if len(spec) != length:
        raise ValueError(
            'max-rank sequence has length %d, expected %d' % (len(spec), length))
    return spec


def compute_minimal_ranks(
        shape: typ.Sequence[int], # (N0, ..., N(d-1))
        tucker_ranks: typ.Union[
            typ.Sequence[int], # (n0,...,n(d-1))
            NDArray, # dtype=int, shape=(d,) + stack_shape
        ],
        tt_ranks: typ.Union[
            typ.Sequence[int], # (r0,...,rd)
            NDArray, # dtype=int, shape=(d+1,) + stack_shape
        ],
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Union[
        typ.Tuple[int,...],  # (n0',...,n(d-1)')
        NDArray,  # dtype=int, shape=(d,) + stack_shape
    ], # new_tucker_ranks
    typ.Union[
        typ.Tuple[int,...],  # (r0',...,rd')
        NDArray,  # dtype=int, shape=(d+1,) + stack_shape
    ], # new_tt_ranks
]:
    '''Find minimal ranks for a generic Tucker tensor train with a given structure.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    is_sequence: bool = False
    if isinstance(tucker_ranks, typ.Sequence):
        is_sequence = True

    tucker_ranks = xnp.array(tucker_ranks)
    tt_ranks = xnp.array(tt_ranks)

    d = len(shape)
    assert(len(tucker_ranks) == d)
    assert(len(tt_ranks) == d+1)

    new_tucker_ranks   = list(tucker_ranks)
    new_tt_ranks       = list(tt_ranks)

    for ii in range(d):
        new_tucker_ranks[ii] = xnp.minimum(new_tucker_ranks[ii], shape[ii])

    new_tt_ranks[-1] = xnp.ones(tt_ranks.shape[1:], dtype=int)
    for ii in range(d-1, 0, -1):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        new_tt_ranks[ii] = np.minimum(rL, n*rR)

    new_tt_ranks[0] = xnp.ones(tt_ranks.shape[1:], dtype=int)
    for ii in range(d):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        n = np.minimum(n, rL*rR)
        rR = np.minimum(rR, rL*n)
        new_tucker_ranks[ii] = n
        new_tt_ranks[ii+1] = rR

    if is_sequence:
        new_tucker_ranks = tuple(int(n) for n in new_tucker_ranks)
        new_tt_ranks = tuple(int(r) for r in new_tt_ranks)
    else:
        new_tucker_ranks = xnp.array(new_tucker_ranks)
        new_tt_ranks = xnp.array(new_tt_ranks)

    return new_tucker_ranks, new_tt_ranks


def _edge_condition_number(
        singular_values: NDArray,  # (k,), nonnegative, descending
) -> float:                        # sigma_1 / sigma_k, with degenerate-edge conventions (see below)
    s = np.asarray(singular_values, dtype=float)
    if s.size == 0:
        return 1.0
    s_max = float(s.max())
    s_min = float(s.min())
    if s_max <= 0.0:   # entire edge ~ zero (e.g. the zero tensor): trivial, treat as well conditioned
        return 1.0
    if s_min <= 0.0:   # rank-deficient edge (sigma_1 > 0, sigma_k ~ 0): worst-conditioned, never grow it
        return float(np.inf)
    return s_max / s_min


def edge_condition_numbers(
        tucker_singular_values: typ.Sequence[NDArray],  # len=d,   elm_shape=(n_i,), descending
        tt_singular_values:     typ.Sequence[NDArray],  # len=d+1, elm_shape=(r_i,), descending
) -> typ.Tuple[
    typ.Tuple[float, ...],  # len=d,   Tucker edge condition numbers (kappa^Tucker)
    typ.Tuple[float, ...],  # len=d+1, TT edge condition numbers (kappa^TT); length-1 boundary bonds = 1.0
]:
    '''Edge condition numbers ``kappa_i = sigma_{i,1} / sigma_{i,r_i}`` of the matrix unfoldings.

    Section 5.4.1 of Alger et al. (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141): the
    ratio of the largest to the smallest *retained* singular value on each Tucker matricization and each
    TT unfolding, as returned by the implicit T3-SVD (:py:meth:`TuckerTensorTrain.t3svd`). The two
    length-1 boundary TT bonds give ``1.0``. Degenerate-edge conventions keep the value defined on any
    iterate (including the zero tensor): an all-zero edge -> ``1.0``; a rank-deficient edge -> ``+inf``.
    '''
    kappa_tucker = tuple(_edge_condition_number(s) for s in tucker_singular_values)
    kappa_tt     = tuple(_edge_condition_number(s) for s in tt_singular_values)
    return kappa_tucker, kappa_tt


def _grow_capped_edges(
        shape:        typ.Sequence[int],  # (N0, ..., N(d-1))
        tucker_ranks: typ.Tuple[int,...], # (n_i),  current
        tt_ranks:     typ.Tuple[int,...], # (r_i),  current
        candidates:   typ.Sequence[typ.Tuple[float, int, int]],  # (kappa, kind, index); kind 0=Tucker mode, 1=TT bond
        n_chunk:      int,
        max_grow:     int,                # >= 1
) -> typ.Tuple[
    typ.Tuple[int,...],  # new_tucker_ranks
    typ.Tuple[int,...],  # new_tt_ranks
]:
    '''Grow up to ``max_grow`` of the candidate edges, best-conditioned (smallest kappa) first, taking
    each only if it survives useless-rank removal -- so a structurally-capped edge is skipped and the
    next candidate tried. Ties break deterministically (kappa, then Tucker before TT, then index).
    '''
    tucker = list(tucker_ranks)
    tt = list(tt_ranks)
    grown = 0
    for _, kind, index in sorted(candidates):   # ascending (kappa, kind, index)
        if grown >= max_grow:
            break
        trial_tucker = list(tucker)
        trial_tt = list(tt)
        if kind == 0:
            trial_tucker[index] += n_chunk
        else:
            trial_tt[index] += n_chunk
        clean_tucker, clean_tt = compute_minimal_ranks(shape, tuple(trial_tucker), tuple(trial_tt))
        if (tuple(clean_tucker), tuple(clean_tt)) != (tuple(tucker), tuple(tt)):
            tucker, tt = list(clean_tucker), list(clean_tt)   # this edge actually grew -> keep it
            grown += 1
    return tuple(tucker), tuple(tt)


def compute_continuation_ranks(
        shape:                  typ.Sequence[int],      # (N0, ..., N(d-1))
        tucker_singular_values: typ.Sequence[NDArray],  # len=d,   elm_shape=(n_i,), descending
        tt_singular_values:     typ.Sequence[NDArray],  # len=d+1, elm_shape=(r_i,), descending
        tau:                    float = 10.0,           # grow edge i only if kappa_i < kappa_max / tau  (tau > 1)
        n_chunk:                int   = 1,              # rank increment added to each grown edge
        kappa_guard:            float = 1e12,           # absolute safety cap: never grow an edge with kappa_i >= this
        max_grow:               typ.Optional[int] = None,  # cap on #edges grown per call (None = all eligible)
) -> typ.Tuple[
    typ.Tuple[int, ...],  # (n0', ..., n(d-1)')  new Tucker ranks
    typ.Tuple[int, ...],  # (r0', ..., rd')      new TT ranks
]:
    '''Rank-continuation update: choose new ranks from the current iterate's unfolding singular values.

    Section 5.4.1 ("Choosing the new ranks") of Alger et al. (2026), "Tucker Tensor Train Taylor Series"
    (arXiv:2603.21141). Grow only the *well-conditioned* edges -- those a factor ``tau`` below the
    worst-conditioned edge -- so the increases bring all edges toward comparable conditioning::

        n_i' = n_i + n_chunk  if kappa^Tucker_i < kappa_max / tau  else n_i
        r_i' = r_i + n_chunk  if kappa^TT_i     < kappa_max / tau  else r_i

    with ``kappa_max`` the largest (finite) edge condition number (:py:func:`edge_condition_numbers`).
    The current ranks ``n_i = len(tucker_singular_values[i])`` and ``r_i = len(tt_singular_values[i])``
    are read off the provided singular values, so pass the structural-rank T3-SVD output (``t3svd()``
    with no truncation); a truncated SVD instead grows from the numerical rank. The boundary TT bonds
    ``r_0, r_d`` are never grown. Proposed ranks are then de-degenerated by
    :py:func:`compute_minimal_ranks` (the paper's "useless-rank removal" -- shape and ranks only, no
    linear algebra); if that leaves the ranks unchanged (every edge already comparably conditioned, or
    all increases removed) every (below-guard) rank is bumped by ``n_chunk`` and re-cleaned, so
    continuation makes progress unless the structure is already maximal.

    **Absolute conditioning guard** (``kappa_guard``, a numerical safety net distinct from the relative
    ``tau`` rule): an edge is grown only if ``kappa_i < kappa_guard`` as well -- so an edge that is well
    conditioned *relative to a catastrophic edge* but ill conditioned *in absolute terms* is still
    frozen. If **no** edge is below the guard (every growable edge is extremely ill conditioned or
    rank-deficient), nothing grows and the returned ranks **equal the input ranks** -- the caller's
    signal to stop the whole continuation. The default ``1e12`` is large: it should fire only on genuine
    near-degeneracy, never during a well-behaved fit. (Note "ranks unchanged" is also returned when the
    structure is already maximal; both mean "stop". The caller can call :py:func:`edge_condition_numbers`
    to tell the two apart for reporting.)

    **Edges per round** (``max_grow``): ``None`` (default) grows *every* eligible edge at once -- the
    Section 5.4.1 rule. An integer ``k`` grows only the ``k`` **best-conditioned** eligible edges that
    survive useless-rank removal (greedy, smallest condition number first, skipping a structurally-capped
    edge for the next candidate). ``max_grow=1`` is **one edge at a time** (the ``tau -> infinity``
    intent, made robust against the spectrum-dependence and structural caps that make a large ``tau``
    unreliable); pair it with ``tau=1.0`` to grow the single best-conditioned edge each round regardless
    of the conditioning spread. The uniform-bump fallback is **not** capped by ``max_grow``, so
    continuation can still escape a degenerate start (e.g. all-ones, where no *single* edge can grow --
    a Tucker rank and its neighboring bond must grow together).

    The paper uses ``tau = 10.0`` and typically ``n_chunk = 1``. Pure host arithmetic on ranks
    (structure), hence numpy-only -- a between-solves decision, never inside a jit trace.
    '''
    assert(max_grow is None or max_grow >= 1)
    d = len(shape)
    tucker_ranks = tuple(int(np.asarray(s).size) for s in tucker_singular_values)  # (n_i),  len d
    tt_ranks     = tuple(int(np.asarray(s).size) for s in tt_singular_values)      # (r_i),  len d+1
    assert(len(tucker_ranks) == d)
    assert(len(tt_ranks) == d + 1)

    kappa_tucker, kappa_tt = edge_condition_numbers(tucker_singular_values, tt_singular_values)
    finite_kappas = [k for k in (kappa_tucker + kappa_tt) if np.isfinite(k)]
    kappa_max = max(finite_kappas) if finite_kappas else 1.0   # +inf edges never grow, never set the max
    threshold = kappa_max / tau

    # An edge is eligible only if it is well conditioned relative to the worst edge (the Section 5.4.1
    # rule) AND below the absolute safety guard; boundary TT bonds (i in {0, d}) are structural, never grow.
    grow_tucker = tuple(kappa_tucker[i] < threshold and kappa_tucker[i] < kappa_guard
                        for i in range(d))
    grow_tt = tuple((0 < i < d) and kappa_tt[i] < threshold and kappa_tt[i] < kappa_guard
                    for i in range(d + 1))

    if max_grow is None:
        proposed_tucker = tuple(n + n_chunk if grow_tucker[i] else n for i, n in enumerate(tucker_ranks))
        proposed_tt     = tuple(r + n_chunk if grow_tt[i]     else r for i, r in enumerate(tt_ranks))
        new_tucker, new_tt = compute_minimal_ranks(shape, proposed_tucker, proposed_tt)  # useless-rank removal
    else:
        eligible = ([(kappa_tucker[i], 0, i) for i in range(d) if grow_tucker[i]]
                  + [(kappa_tt[i], 1, i)     for i in range(d + 1) if grow_tt[i]])
        new_tucker, new_tt = _grow_capped_edges(shape, tucker_ranks, tt_ranks, eligible, n_chunk, max_grow)

    if tuple(new_tucker) == tucker_ranks and tuple(new_tt) == tt_ranks:
        # Nothing grew by the rule -> uniform-bump fallback over every below-guard edge (NOT capped by
        # max_grow: this is the "edges comparably conditioned / get off the ground" escape, and from a
        # degenerate start no single edge can grow). If no edge is below the guard the bump is empty ->
        # ranks return unchanged -> the caller stops (the safety stop). Boundary bonds are held fixed.
        bump_tucker = tuple(n + n_chunk if kappa_tucker[i] < kappa_guard else n
                            for i, n in enumerate(tucker_ranks))
        bump_tt = tuple(r + n_chunk if (0 < i < d and kappa_tt[i] < kappa_guard) else r
                        for i, r in enumerate(tt_ranks))
        new_tucker, new_tt = compute_minimal_ranks(shape, bump_tucker, bump_tt)

    return new_tucker, new_tt


def compute_raw_sweep_ranks(
        shape:        typ.Sequence[int],  # (N0, ..., N(d-1))
        tucker_ranks,                     # current Tucker ranks: seq (n0,...) or array (d,)+stack
        tt_ranks,                         # current TT ranks:     seq (r0,...) or array (d+1,)+stack
        cap_tucker_ranks,                 # min(current, max) Tucker ranks, same form as tucker_ranks
        cap_tt_ranks,                     # min(current, max) TT ranks,     same form as tt_ranks
        use_jax: bool = False,
) -> typ.Tuple:                           # (raw_tucker_ranks, raw_tt_ranks), same form as inputs
    '''Ranks the T3-SVD sweep produces under hard rank caps -- i.e. the ranks ``t3svd`` returns (it does
    not minimize; see :py:func:`rank_adjustment_sweep`). The sweep is down-orthogonalize,
    right-orthogonalize, then a left-to-right pass that caps each Tucker/TT edge: at each mode the SVD
    keeps ``min(structural rank, cap)``, so a downstream cap can leave an upstream rank above the
    structural minimum (non-minimal -- see :py:func:`compute_minimal_ranks`). The caps enter the forward
    pass via the pre-capped ranks. (Used by uniform ``ut3svd`` to shrink the padded supercore to the
    actual content ranks.)
    '''
    xnp, _, _ = get_backend(False, use_jax)

    is_sequence = isinstance(tucker_ranks, typ.Sequence)
    tucker_ranks = xnp.array(tucker_ranks)
    tt_ranks = xnp.array(tt_ranks)
    cap_tucker_ranks = xnp.array(cap_tucker_ranks)
    cap_tt_ranks = xnp.array(cap_tt_ranks)

    d = len(shape)
    n = list(tucker_ranks)
    r = list(tt_ranks)

    for ii in range(d):
        n[ii] = xnp.minimum(n[ii], shape[ii])              # down-orthogonalize: n_i <= N_i

    r[0] = xnp.ones(tt_ranks.shape[1:], dtype=int)
    r[-1] = xnp.ones(tt_ranks.shape[1:], dtype=int)
    for ii in range(d - 1, 0, -1):                          # right-orthogonalize: r_i <- min(r_i, n_i*r_{i+1})
        r[ii] = xnp.minimum(r[ii], n[ii] * r[ii + 1])

    for ii in range(d):                                     # L->R sweep, each edge capped
        n[ii] = xnp.minimum(xnp.minimum(n[ii], r[ii] * r[ii + 1]), cap_tucker_ranks[ii])
        r[ii + 1] = xnp.minimum(xnp.minimum(r[ii] * n[ii], r[ii + 1]), cap_tt_ranks[ii + 1])

    if is_sequence:
        return tuple(int(v) for v in n), tuple(int(v) for v in r)
    return xnp.array(n), xnp.array(r)


def compute_orthogonal_representation_ranks(
        shape: typ.Sequence[int], # (N0, ..., N(d-1))
        tucker_ranks: typ.Union[
            typ.Sequence[int], # (n0,...,n(d-1))
            NDArray, # dtype=int, shape=(d,) + stack_shape
        ],
        tt_ranks: typ.Union[
            typ.Sequence[int], # (r0,...,rd)
            NDArray, # dtype=int, shape=(d+1,) + stack_shape
        ],
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Union[
        typ.Tuple[int,...],  # (nU0,...,nU(d-1))
        NDArray,  # dtype=int, shape=(d,) + stack_shape
    ], # up_tucker_ranks
    typ.Union[
        typ.Tuple[int, ...],  # (nD0',...,nD(d-1)')
        NDArray,  # dtype=int, shape=(d,) + stack_shape
    ],  # down_tucker_ranks
    typ.Union[
        typ.Tuple[int,...],  # (rL0',...,rLd')
        NDArray,  # dtype=int, shape=(d+1,) + stack_shape
    ], # left_tt_ranks
    typ.Union[
        typ.Tuple[int, ...],  # (rR0',...,rRd')
        NDArray,  # dtype=int, shape=(d+1,) + stack_shape
    ],  # right_tt_ranks
]:
    '''Find ranks that would be produced by sweeping orthogonalization, except without actually doing it.
    '''
    xnp, _, _ = get_backend(False, use_jax)

    is_sequence: bool = False
    if isinstance(tucker_ranks, typ.Sequence):
        is_sequence = True

    tucker_ranks = xnp.array(tucker_ranks)
    tt_ranks = xnp.array(tt_ranks)

    d = len(shape)
    assert(len(tucker_ranks) == d)
    assert(len(tt_ranks) == d+1)

    stack_shape = tt_ranks.shape[1:]

    up_ranks    = list(tucker_ranks)
    right_ranks = list(tt_ranks)

    for ii in range(d):
        up_ranks[ii] = xnp.minimum(up_ranks[ii], shape[ii])

    right_ranks[-1] = xnp.ones(stack_shape, dtype=int)
    for ii in range(d-1, 0, -1):
        n   = up_ranks[ii]
        rL  = tt_ranks[ii]
        rR  = right_ranks[ii+1]

        right_ranks[ii] = np.minimum(rL, n*rR)

    left_ranks = right_ranks.copy()

    left_ranks[0] = xnp.ones(stack_shape, dtype=int)
    for ii in range(d):
        n   = up_ranks[ii]
        rL  = left_ranks[ii]
        rR  = right_ranks[ii+1]

        left_ranks[ii+1] = np.minimum(rL*n, rR)

    down_ranks = up_ranks.copy()

    for ii in range(d):
        n   = up_ranks[ii]
        rL  = left_ranks[ii]
        rR  = right_ranks[ii+1]

        down_ranks[ii] = np.minimum(n, rL*rR)

    if is_sequence:
        up_ranks = tuple(int(n) for n in up_ranks)
        left_ranks = tuple(int(r) for r in left_ranks)
        right_ranks = tuple(int(r) for r in right_ranks)
        down_ranks = tuple(int(r) for r in down_ranks)
    else:
        up_ranks = xnp.array(up_ranks)
        left_ranks = xnp.array(left_ranks)
        right_ranks = xnp.array(right_ranks)
        down_ranks = xnp.array(down_ranks)

    return up_ranks, down_ranks, left_ranks, right_ranks


def compute_manifold_dim(
        shape:          typ.Sequence[int],  # (N0, ..., N(d-1))
        tucker_ranks:   typ.Sequence[int],  # (n0, ..., n(d-1))
        tt_ranks:       typ.Sequence[int],  # (r0, ..., rd)
) -> int:
    '''Dimension of the fixed-rank Tucker tensor train manifold for the given structure.

    Computed from the structurally-minimal ranks (gauge already quotiented), so this is the true
    tangent-space dimension for a minimal-rank base point.
    '''
    min_tucker_ranks, min_tt_ranks = compute_minimal_ranks(shape, tucker_ranks, tt_ranks)

    num_cores = len(shape)
    manifold_dim: int = 0
    for ii in range(num_cores):
        n  = min_tucker_ranks[ii]
        rL = min_tt_ranks[ii]
        rR = min_tt_ranks[ii + 1]
        if ii == num_cores - 1:
            manifold_dim += rL * n * rR
        else:
            manifold_dim += (rL * n - rR) * rR

    for ii in range(num_cores):
        n = min_tucker_ranks[ii]
        N = shape[ii]
        manifold_dim += (N - n) * n

    return int(manifold_dim)


def frame_has_minimal_ranks(
        shape:          typ.Sequence[int],
        up_ranks:       typ.Sequence[int],
        down_ranks:     typ.Sequence[int],
        left_ranks:     typ.Sequence[int],
        right_ranks:    typ.Sequence[int],
) -> bool:
    '''True if a T3Frame with these (redundant) ranks is structurally minimal.

    Requires the left/right and up/down rank stores to agree, and the up/left ranks to equal the
    minimal ranks for the shape.
    '''
    if tuple(left_ranks) != tuple(right_ranks):
        return False
    if tuple(up_ranks) != tuple(down_ranks):
        return False
    min_tucker_ranks, min_tt_ranks = compute_minimal_ranks(shape, up_ranks, left_ranks)
    return (tuple(int(n) for n in min_tucker_ranks) == tuple(int(n) for n in up_ranks)
            and tuple(int(r) for r in min_tt_ranks) == tuple(int(r) for r in left_ranks))