# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.linalg as linalg
from t3toolbox.backend.common import *

__all__ = [
    'compute_minimal_ranks',
    'compute_orthogonal_representation_ranks',
    'compute_manifold_dim',
    'basis_has_minimal_ranks',
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


def basis_has_minimal_ranks(
        shape:          typ.Sequence[int],
        up_ranks:       typ.Sequence[int],
        down_ranks:     typ.Sequence[int],
        left_ranks:     typ.Sequence[int],
        right_ranks:    typ.Sequence[int],
) -> bool:
    '''True if a T3Basis with these (redundant) ranks is structurally minimal.

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