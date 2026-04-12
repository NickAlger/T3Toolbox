# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

__all__ = [
    'numpy_scan',
]


Carry = typ.TypeVar('Carry')
ScanX = typ.TypeVar('ScanX')
ScanY = typ.TypeVar('ScanY')
NDArray = typ.TypeVar('NDArray')
StackedNDArray = typ.TypeVar('StackedNDArray')


def ragged_scan(
        f: typ.Callable[[Carry, typ.Tuple[ScanX]], typ.Tuple[Carry, typ.Sequence[ScanY]]],
        init: Carry,
        xs: typ.Sequence[typ.Sequence[ScanX]], # elements all have length L
) -> typ.Tuple[
    Carry,
    typ.Tuple[typ.Tuple[ScanY, ...], ...], # elements all have length L
]:
    """Similar to jax.lax.scan, except for ragged-sized arrays
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html

    """
    length = len(xs[0])
    carry = init

    ys_list = [[] for _ in range(length)]
    for ii in range(length):
        x = tuple([x[ii] for x in xs])
        carry, y = f(carry, x)
        for l, elm in zip(ys_list, y):
            l.append(elm)

    return carry, tuple([tuple(y) for y in ys_list])


def numpy_scan(
        f: typ.Callable[[Carry, NDArray], typ.Tuple[Carry, NDArray]],
        init: Carry,
        xs: typ.Sequence[StackedNDArray], # xs[ii].shape[0] = L
) -> typ.Tuple[
    Carry,
    typ.Tuple[StackedNDArray, ...], # ith_elm.shape[0] = L
]:
    xs_list = [list(x) for x in xs]
    carry, ys_list = ragged_scan(f, init, xs_list)
    ys = tuple([np.stack(y) for y in ys_list])
    return carry, ys



