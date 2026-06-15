# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.probing as probing
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.common import *

__all__ = [
    'ut3_entries',
    'ut3_apply',
    'ut3_probe',
    'ut3_full_sum',
]

# All re-mask first (so the padded "garbage" contributes nothing), then call the SHARED, already-
# polymorphic sampling backends (entries/apply/probing) on the masked supercores. Physical vectors are
# zero-padded to N (pack) and probe results sliced back to the real shape (unpack); index entries need
# no packing (the real block is the prefix, so an index in [0,Ni) hits the right slot).

UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[NDArray, NDArray, NDArray]]


def ut3_entries(
        data:  UT3Data,
        index: NDArray,  # dtype=int, shape=(d,)+idx_stack
) -> NDArray:            # shape=idx_stack+stack_shape
    """Compute entries of a uniform Tucker tensor train (shares ``entries.tucker_tensor_train_entries``)."""
    masked = ut3_masking.apply_masks_to_cores(data)
    return entries.tucker_tensor_train_entries(masked, index)


def ut3_apply(
        data: UT3Data,
        vecs: typ.Sequence[NDArray],  # len=d, ith elm_shape=vec_stack+(Ni,)
) -> NDArray:                         # shape=vec_stack+stack_shape
    """Contract a uniform Tucker tensor train with vectors in all modes (shares
    ``apply.tucker_tensor_train_apply``). Vectors are zero-padded to ``N``."""
    masked = ut3_masking.apply_masks_to_cores(data)
    packed = ut3_operations.pack_vectors(vecs, masked[0].shape[-1])
    return apply.tucker_tensor_train_apply(masked, packed)


def ut3_probe(
        ww:   typ.Sequence[NDArray],  # len=d, ith elm_shape=W+(Ni,)
        data: UT3Data,
) -> typ.Tuple[NDArray, ...]:         # len=d, ith elm_shape=W+(Ni,)
    """Probe a uniform Tucker tensor train (contract all-but-one mode, for each mode; shares
    ``probing.probe_t3``). Vectors padded to ``N``; results sliced back to the real shape."""
    masked = ut3_masking.apply_masks_to_cores(data)
    packed = ut3_operations.pack_vectors(ww, masked[0].shape[-1])
    zz = probing.probe_t3(packed, masked)                       # packed, shape=(d,)+W+(N)
    shape = [int(m.sum()) for m in data[2][0]]                  # data[2][0] = shape_mask
    return ut3_operations.unpack_vectors(zz, shape)


def ut3_full_sum(
        data: UT3Data,
) -> NDArray:  # shape=stack_shape (scalar if unstacked) -- sum over ALL physical modes
    """Sum the represented dense tensor over all physical modes: ``apply`` with all-ones vectors (the
    masked supercore's padding is zero, so ones over ``N`` sum only the real ``Ni``)."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)
    masked = ut3_masking.apply_masks_to_cores(data)
    d, N = masked[0].shape[0], masked[0].shape[-1]
    return apply.tucker_tensor_train_apply(masked, xnp.ones((d, N)))
