# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Constructors for (ragged) Tucker tensor trains from a structure spec (zeros, ones,
corewise-iid Gaussian). Mirrors ``ut3_constructors``."""
import math
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    't3_zeros',
    't3_ones',
    't3_corewise_randn',
]


def t3_zeros(
        shape:        typ.Tuple[int, ...],       # len=d, the tensor mode sizes (N0,...,N(d-1))
        tucker_ranks: typ.Tuple[int, ...],       # len=d
        tt_ranks:     typ.Tuple[int, ...],       # len=d+1
        stack_shape:  typ.Tuple[int, ...] = (),  # leading batch axes

        use_jax:      bool = False,  # constructor: no array inputs, so the flag chooses the output type
) -> typ.Tuple[
    typ.Sequence[NDArray],  # tucker_cores. len=d, elm_shape=stack_shape+(ni,Ni)
    typ.Sequence[NDArray],  # tt_cores.     len=d, elm_shape=stack_shape+(rLi,ni,rR(i+1))
]:
    """Construct a Tucker tensor train of zeros.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    vs = stack_shape

    tt_cores = tuple([xnp.zeros(vs+(tt_ranks[ii], tucker_ranks[ii], tt_ranks[ii+1])) for ii in range(len(tucker_ranks))])
    tucker_cores = tuple([xnp.zeros(vs+(n, N)) for n, N  in zip(tucker_ranks, shape)])
    return tucker_cores, tt_cores


def t3_ones(
        shape:        typ.Tuple[int, ...],       # len=d, the tensor mode sizes (N0,...,N(d-1))
        stack_shape:  typ.Tuple[int, ...] = (),  # leading batch axes

        use_jax:      bool = False,  # constructor: no array inputs, so the flag chooses the output type
) -> typ.Tuple[
    typ.Sequence[NDArray],  # tucker_cores. len=d, rank-1, elm_shape=stack_shape+(1,Ni)
    typ.Sequence[NDArray],  # tt_cores.     len=d, rank-1, elm_shape=stack_shape+(1,1,1)
]:
    """Construct the rank-1 Tucker tensor train representing a tensor full of ones.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    vs = stack_shape

    tt_cores = tuple([xnp.ones(vs+(1, 1, 1)) for ii in range(len(shape))])
    tucker_cores = tuple([xnp.ones(vs+(1, N)) for N  in shape])
    return tucker_cores, tt_cores


def t3_corewise_randn(
        shape:        typ.Tuple[int, ...],       # len=d, the tensor mode sizes (N0,...,N(d-1))
        tucker_ranks: typ.Tuple[int, ...],       # len=d
        tt_ranks:     typ.Tuple[int, ...],       # len=d+1
        stack_shape:  typ.Tuple[int, ...] = (),  # leading batch axes

        use_jax:      bool = False,  # constructor: no array inputs, so the flag chooses the output type
) -> typ.Tuple[
    typ.Sequence[NDArray],  # tucker_cores. len=d, elm_shape=stack_shape+(ni,Ni)
    typ.Sequence[NDArray],  # tt_cores.     len=d, elm_shape=stack_shape+(rLi,ni,rR(i+1))
]:
    """Construct a Tucker tensor train with random cores.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    d = len(tucker_ranks)
    vs = stack_shape

    tt_cores = []
    for ii in range(d):
        shape_G = vs + (tt_ranks[ii], tucker_ranks[ii], tt_ranks[ii+1])
        G = randn(*shape_G, use_jax=use_jax)
        tt_cores.append(G)

    tucker_cores = []
    for ii in range(d):
        shape_B = vs + (tucker_ranks[ii], shape[ii])
        B = randn(*shape_B, use_jax=use_jax)
        tucker_cores.append(B)

    return tuple(tucker_cores), tuple(tt_cores)
