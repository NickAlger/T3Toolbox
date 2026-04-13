# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ


has_jax = False
try:
    import jax.numpy as jnp
    import jax
    has_jax = True
    NDArray = typ.Union[np.ndarray, jnp.ndarray]
    jax_scan = jax.lax.scan
    jax_map = jax.lax.map
except ImportError:
    NDArray = np.ndarray


__all__ = [
    'NDArray',
    #
    'ragged_scan',
    'numpy_scan',
    'jax_scan',
    #
    'ragged_map',
    'numpy_map',
    'jax_map',
    #
    'get_backend',
]

#


CarryType = typ.TypeVar('CarryType')

def ragged_scan(
        f: typ.Callable[
            [CarryType,
             typ.Sequence[NDArray],   # len=num_inputs
             ],
            typ.Tuple[
                CarryType,
                typ.Sequence[NDArray],   # len=num_outputs
            ],
        ],
        init: CarryType,
        xs: typ.Sequence[
            typ.Union[
                typ.Sequence[NDArray], # len=scan_length
                NDArray, # shape[0]=scan_length
            ]
        ], # len=num_inputs
) -> typ.Tuple[
    CarryType,
    typ.Tuple[
        typ.Tuple[NDArray, ...], # len=scan_length
        ...
    ],  # len=num_outputs,
]:
    """Similar to jax.lax.scan, except for ragged-sized arrays
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html

    """
    scan_length = len(xs[0])
    carry = init

    ys_list = []
    for ii in range(scan_length):
        x = tuple([x[ii] for x in xs])
        carry, y = f(carry, x)

        if ii==0:
            ys_list = [[] for _ in range(len(y))]

        for l, elm in zip(ys_list, y):
            l.append(elm)

    return carry, tuple([tuple(y) for y in ys_list])


def numpy_scan(
        f: typ.Callable[
            [CarryType,
             typ.Sequence[NDArray],  # len=num_inputs
             ],
            typ.Tuple[
                CarryType,
                typ.Sequence[NDArray],  # len=num_outputs
            ],
        ],
        init: CarryType,
        xs: typ.Sequence[
            typ.Union[
                typ.Sequence[NDArray],  # len=scan_length
                NDArray,  # shape[0]=scan_length
            ]
        ],  # len=num_inputs
) -> typ.Tuple[
    CarryType,
    typ.Tuple[
        NDArray, # shape[0]=scan_length
        ...
    ],  # len=num_outputs,
]:
    """Similar to jax.lax.scan, except returns numpy arrays instead of jax arrays.
    """
    xs_list = [list(x) for x in xs]
    carry, ys_list = ragged_scan(f, init, xs_list)
    ys = tuple([np.stack(y) for y in ys_list])
    return carry, ys


def ragged_map(
        f: typ.Callable[
            [CarryType,
             typ.Sequence[NDArray],  # len=num_inputs
             ],
            typ.Tuple[
                CarryType,
                typ.Sequence[NDArray],  # len=num_outputs
            ],
        ],
        xs: typ.Sequence[
            typ.Union[
                typ.Sequence[NDArray],  # len=map_length
                NDArray,  # shape[0]=map_length
            ]
        ],  # len=num_inputs
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # len=map_length
    ...
]:  # len=num_outputs
    map_length = len(xs[0])

    ys_list = []
    for ii in range(map_length):
        x = tuple([elm[ii] for elm in xs])
        y = f(x)

        if ii==0:
            ys_list = [[] for _ in range(len(y))]

        for l, elm in zip(ys_list, y):
            l.append(elm)

    return tuple([tuple(y) for y in ys_list])


def numpy_map(
        f: typ.Callable[
            [CarryType,
             typ.Sequence[NDArray],  # len=num_inputs
             ],
            typ.Tuple[
                CarryType,
                typ.Sequence[NDArray],  # len=num_outputs
            ],
        ],
        xs: typ.Sequence[
            typ.Union[
                typ.Sequence[NDArray],  # len=map_length
                NDArray,  # shape[0]=map_length
            ]
        ],  # len=num_inputs
) -> typ.Tuple[
    NDArray,  # shape[0]=map_length
    ...
]:  # len=num_outputs,
    xs_list = [list(x) for x in xs]
    ys_list = ragged_map(f, xs_list)
    ys = tuple([np.stack(y) for y in ys_list])
    return ys


if not has_jax:
    jax_scan = numpy_scan
    jax_map = numpy_map


def get_backend(
        is_ragged: bool,
        use_jax: bool,
):
    if is_ragged:
        xmap = ragged_map
        xscan = ragged_scan
    else:
        if use_jax:
            xmap = jax_map
            xscan = jax_scan
        else:
            xmap = numpy_map
            xscan = numpy_scan

    if use_jax:
        xnp = jnp
    else:
        xnp = np

    return xnp, xmap, xscan

