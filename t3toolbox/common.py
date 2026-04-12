# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

__all__ = [
    'ragged_scan',
    'numpy_scan',
    'ragged_map',
    # 'numpy_map',
]


CarryType = typ.TypeVar('CarryType')
InputType = typ.TypeVar('InputType')
OutputType = typ.TypeVar('OutputType')
NDArray = typ.TypeVar('NDArray')
StackedNDArray = typ.TypeVar('StackedNDArray')


def ragged_scan(
        f: typ.Callable[[CarryType, typ.Tuple[InputType]], typ.Tuple[CarryType, typ.Sequence[OutputType]]],
        init: CarryType,
        xs: typ.Sequence[typ.Sequence[InputType]], # len=k, elements all have length L
) -> typ.Tuple[
    CarryType,
    typ.Tuple[typ.Tuple[OutputType, ...], ...], # len=k, elements all have length L
]:
    """Similar to jax.lax.scan, except for ragged-sized arrays
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html

    """
    length = len(xs[0])
    carry = init

    ys_list = []
    for ii in range(length):
        x = tuple([x[ii] for x in xs])
        carry, y = f(carry, x)

        if ii==0:
            ys_list = [[] for _ in range(len(y))]

        for l, elm in zip(ys_list, y):
            l.append(elm)

    return carry, tuple([tuple(y) for y in ys_list])


def numpy_scan(
        f: typ.Callable[[CarryType, NDArray], typ.Tuple[CarryType, NDArray]],
        init: CarryType,
        xs: typ.Sequence[StackedNDArray], # xs[ii].shape[0] = L
) -> typ.Tuple[
    CarryType,
    typ.Tuple[StackedNDArray, ...], # ith_elm.shape[0] = L
]:
    xs_list = [list(x) for x in xs]
    carry, ys_list = ragged_scan(f, init, xs_list)
    ys = tuple([np.stack(y) for y in ys_list])
    return carry, ys


def ragged_map(
        f: typ.Callable[[typ.Tuple[InputType]], typ.Tuple[typ.Sequence[OutputType]]],
        xs: typ.Sequence[typ.Sequence[InputType]], # len(xs[0])=len(xs[1])=...=L
) -> typ.Tuple[
    typ.Tuple[typ.Tuple[OutputType, ...], ...], # elements all have length L
]:
    length = len(xs[0])

    ys_list = []
    for ii in range(length):
        x = tuple([elm[ii] for elm in xs])
        y = f(x)

        if ii==0:
            ys_list = [[] for _ in range(len(y))]

        for l, elm in zip(ys_list, y):
            l.append(elm)

    return tuple([tuple(y) for y in ys_list])

