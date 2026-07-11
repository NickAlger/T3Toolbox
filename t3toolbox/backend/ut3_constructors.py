# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Constructors and file IO for uniform Tucker tensor trains (UT3), on the raw ``.data`` tuple.

``ut3_zeros`` / ``ut3_ones`` / ``ut3_corewise_randn`` build the padded supercores + masks **directly** (the
uniform feature ragged round-tripping cannot express: ranks may vary per stack element -- the
determinantal variety, ``docs/uniform_ranks_and_varieties.md``). ``ut3_save`` / ``ut3_load`` share
``common.save_core_families`` (2 supercores + 2 rank masks + the shape ints).

There are deliberately **no** ``ut3_from_canonical`` / ``ut3_from_tensor_train`` / ``ut3_to_tensor_train``
round-trips: they would take *ragged* CP/TT data and round-trip through ``TuckerTensorTrain``, which is
ambiguous (ragged vs uniform input) and trivially composable from the existing ragged ops +
``UniformTuckerTensorTrain.from_t3`` / ``.to_t3``. Be explicit at the boundary instead.

Following the layer-wide rule (``docs/uniform_pytree_composition.md``): **supercores (data) ->
``xnp``/``use_jax``; masks (structure) -> ``np`` (host)**. The pure constructors keep a ``use_jax``
flag for the supercores (there is no array input to infer from). ``ut3_load`` keeps ``use_jax`` for the
supercores but always returns numpy (host) bool masks.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.ut3_masking as ut3_masking
from t3toolbox.backend.common import *
import t3toolbox.backend.common as common

__all__ = [
    'ut3_zeros',
    'ut3_ones',
    'ut3_corewise_randn',
    'ut3_save',
    'ut3_load',
]

# .data[2] is the static int-tuple shape; .data[3] = (tucker_edge_mask, tt_edge_mask) are HOST bool,
# static structure (numpy, never traced); the supercores are xnp.
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]]


def _broadcast_ranks(
        ranks_spec,                       # int | Sequence[int] (len=length, per-stack uniform) | array (length,)+stack
        length:      int,                 # d for Tucker ranks, d+1 for TT ranks
        stack_shape: typ.Tuple[int, ...],
) -> NDArray:                             # HOST int, shape=(length,)+stack_shape
    """Normalize a rank spec to a HOST int rank array of shape ``(length,)+stack_shape``.

    Accepts a single int (every mode/stack element the same), a ``len=length`` sequence (uniform across
    the stack -- broadcast), or a full ``(length,)+stack_shape`` array (the **variety**: ranks varying
    per stack element). ``np`` (host): ranks are mask (structure) metadata, never supercore data.
    """
    target = (length,) + stack_shape
    arr = np.asarray(ranks_spec, dtype=int)
    if arr.ndim == 0:                                   # scalar -> fill every position
        return np.full(target, int(arr))
    if arr.shape == (length,):                          # per-mode, uniform across the stack
        return np.broadcast_to(arr.reshape((length,) + (1,) * len(stack_shape)), target).copy()
    if arr.shape == target:                             # already the full (variety) array
        return arr.copy()
    raise ValueError(
        'rank spec has shape %s, expected a scalar, (%d,), or %s' % (arr.shape, length, target))


def _ut3_constant(
        fill,                                          # 'zeros' | 'ones' | 'randn'
        shape:        typ.Sequence[int],               # (N0,...,N(d-1))
        tucker_ranks,                                  # int | len-d seq | (d,)+stack array
        tt_ranks,                                      # int | len-(d+1) seq | (d+1,)+stack array
        stack_shape:  typ.Tuple[int, ...] = (),
        use_jax:      bool = False,                    # constructor: chooses the SUPERCORE type
) -> UT3Data:
    """Build a UT3 ``.data`` tuple with the supercores filled by ``fill`` (masked in padded regions).

    Direct construction: the padded sizes are ``N=max(shape)``, ``n=max(tucker_ranks)``,
    ``r=max(tt_ranks)`` (over modes AND the stack -- so per-stack-element ranks set a common pad). The
    supercores are built with ``xnp`` per ``use_jax`` then masked to zero the padding; the masks are
    built numpy (host).
    """
    xnp, _, _ = get_backend(True, use_jax)
    d = len(shape)
    stack_shape = tuple(stack_shape)

    tucker_ranks = _broadcast_ranks(tucker_ranks, d, stack_shape)       # HOST int (d,)+stack
    tt_ranks     = _broadcast_ranks(tt_ranks, d + 1, stack_shape)       # HOST int (d+1,)+stack

    N = int(max(shape))
    n = int(np.max(tucker_ranks))
    r = int(np.max(tt_ranks))

    tucker_shape = (d,) + stack_shape + (n, N)
    tt_shape     = (d,) + stack_shape + (r, n, r)
    if fill == 'zeros':
        tucker_supercore, tt_supercore = xnp.zeros(tucker_shape), xnp.zeros(tt_shape)
    elif fill == 'ones':
        tucker_supercore, tt_supercore = xnp.ones(tucker_shape), xnp.ones(tt_shape)
    elif fill == 'randn':
        tucker_supercore = common.randn(*tucker_shape, use_jax=use_jax)
        tt_supercore     = common.randn(*tt_shape, use_jax=use_jax)
    else:
        raise ValueError('unknown fill %r' % (fill,))

    shape = tuple(int(Ni) for Ni in shape)  # static int tuple (N0,...,N(d-1))
    masks = ut3_masking.ut3_make_masks(tucker_ranks, tt_ranks, n, r)  # HOST bool rank masks, static
    # Mask the padded ("garbage") regions to zero so the represented tensor is exactly the fill value.
    masked_tucker, masked_tt = ut3_masking.ut3_apply_masks((tucker_supercore, tt_supercore, shape, masks))
    return masked_tucker, masked_tt, shape, masks


def ut3_zeros(
        shape:        typ.Sequence[int],               # (N0,...,N(d-1))
        tucker_ranks=None,                             # int | len-d seq | (d,)+stack array;   None -> all 1
        tt_ranks=None,                                 # int | len-(d+1) seq | (d+1,)+stack array; None -> all 1
        stack_shape:  typ.Tuple[int, ...] = (),
        use_jax:      bool = False,                    # constructor: chooses the SUPERCORE type
) -> UT3Data:
    """Uniform Tucker tensor train of zeros. ``tucker_ranks``/``tt_ranks`` may vary per stack element."""
    d = len(shape)
    tucker_ranks = 1 if tucker_ranks is None else tucker_ranks
    tt_ranks     = 1 if tt_ranks is None else tt_ranks
    return _ut3_constant('zeros', shape, tucker_ranks, tt_ranks, stack_shape, use_jax=use_jax)


def ut3_ones(
        shape:        typ.Sequence[int],               # (N0,...,N(d-1))
        stack_shape:  typ.Tuple[int, ...] = (),
        use_jax:      bool = False,                    # constructor: chooses the SUPERCORE type
) -> UT3Data:
    """Rank-1 uniform Tucker tensor train representing a tensor full of ones (every real entry == 1)."""
    return _ut3_constant('ones', shape, 1, 1, stack_shape, use_jax=use_jax)


def ut3_corewise_randn(
        shape:        typ.Sequence[int],               # (N0,...,N(d-1))
        tucker_ranks,                                  # int | len-d seq | (d,)+stack array
        tt_ranks,                                      # int | len-(d+1) seq | (d+1,)+stack array
        stack_shape:  typ.Tuple[int, ...] = (),
        use_jax:      bool = False,                    # constructor: chooses the SUPERCORE type
) -> UT3Data:
    """Uniform Tucker tensor train with random N(0,1) supercores (padded regions masked to zero).

    ``tucker_ranks``/``tt_ranks`` may vary per stack element (the variety) -- a full ``(d,)+stack`` /
    ``(d+1,)+stack`` array sets per-element ranks while keeping one padded supercore shape.
    """
    return _ut3_constant('randn', shape, tucker_ranks, tt_ranks, stack_shape, use_jax=use_jax)


def ut3_save(
        file,         # path or open file object to write the .npz to
        data: UT3Data,
) -> None:
    """Save a uniform Tucker tensor train (2 supercores + 2 rank masks + the shape ints) to a ``.npz``.

    Shares :py:func:`~t3toolbox.backend.common.save_core_families`: family 0 is the supercores, family 1
    is the (numpy, host) rank masks, family 2 is the static ``shape`` as a 1-element int array.
    :py:func:`ut3_load` regroups them. ``np.savez`` stores the boolean masks with their dtype, so
    ``ut3_load`` recovers numpy bool masks.
    """
    tucker_supercore, tt_supercore, shape, masks = data
    common.save_core_families(file, (
        (tucker_supercore, tt_supercore),
        tuple(masks),
        (np.asarray(shape, dtype=int),),
    ))


def ut3_load(
        file,                  # path or open file object to read the .npz from
        use_jax: bool = False, # chooses the SUPERCORE type; masks always come back numpy (host) bool
) -> UT3Data:
    """Load a uniform Tucker tensor train from a ``.npz`` file written by :py:func:`ut3_save`.

    The supercores follow ``use_jax``; the masks stay **numpy (host) bool** regardless -- a jax mask is a
    tracer under jit and breaks the layer (``docs/uniform_pytree_composition.md``). ``np.load`` returns
    the masks with their saved bool dtype; we only convert the supercores.
    """
    (supercores, masks, shape_family) = common.load_core_families(file)
    tucker_supercore, tt_supercore = supercores
    if use_jax:
        tucker_supercore = common.to_jax(tucker_supercore)
        tt_supercore     = common.to_jax(tt_supercore)
    # Masks: numpy (host), boolean. np.load already gives numpy; ensure bool dtype (saved as bool).
    masks = tuple(np.asarray(m, dtype=bool) for m in masks)
    shape = tuple(int(x) for x in shape_family[0])  # static int tuple (saved as a 1-element int array)
    return tucker_supercore, tt_supercore, shape, masks
