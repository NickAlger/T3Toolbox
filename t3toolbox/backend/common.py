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
except ImportError:
    print('Unable to import Jax. Using numpy instead.')

NDArray = np.ndarray
if has_jax:
    NDArray = typ.Union[np.ndarray, jnp.ndarray]

is_ndarray = lambda x: isinstance(x, np.ndarray)
if has_jax:
    is_ndarray = lambda x: (isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray))


is_jax_ndarray = lambda x: False
if has_jax:
    is_jax_ndarray = lambda x: isinstance(x, jnp.ndarray)


is_numpy_ndarray = lambda x: isinstance(x, np.ndarray)


def is_boolean_ndarray(x):
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.bool_)
    else:
        return False

if has_jax:
    def is_boolean_ndarray(x):
        if isinstance(x, jnp.ndarray):
            return jnp.issubdtype(x.dtype, jnp.bool_)
        elif isinstance(x, np.ndarray):
            return np.issubdtype(x.dtype, np.bool_)
        else:
            return False

to_jax = lambda x: np.array(x)
if has_jax:
    to_jax = lambda x: jnp.array(x)

to_numpy = lambda x: np.array(x)


__all__ = [
    'has_jax',
    #
    'NDArray',
    'is_ndarray',
    'is_boolean_ndarray',
    'is_jax_ndarray',
    'is_numpy_ndarray',
    'to_jax',
    'to_numpy',
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
    'xcat',
    'xappend',
    'xprepend',
    #
    'randn',
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
    print('RAGGED SCAN')
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
    print('NUMPY SCAN(')
    xs_list = [list(x) for x in xs]
    carry, ys_list = ragged_scan(f, init, xs_list)
    ys = tuple([np.stack(y) for y in ys_list])
    print(')')
    return carry, ys


def ragged_map(
        f: typ.Callable[
            [
                typ.Sequence[NDArray],  # len=num_inputs
             ],
            typ.Tuple[
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
    print('RAGGED MAP')
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
    print('NUMPY MAP(')
    xs_list = [list(x) for x in xs]
    ys_list = ragged_map(f, xs_list)
    ys = tuple([np.stack(y) for y in ys_list])
    print(')')
    return ys


jax_scan = numpy_scan
jax_map = numpy_map
if has_jax:
    jax_scan = jax.lax.scan
    jax_map = jax.lax.map


def get_backend(
        is_uniform: bool,
        use_jax: bool,
):
    if is_uniform:
        if use_jax:
            xmap = jax_map
            xscan = jax_scan
        else:
            xmap = numpy_map
            xscan = numpy_scan
    else:
        xmap = ragged_map
        xscan = ragged_scan

    if use_jax:
        xnp = jnp
    else:
        xnp = np

    return xnp, xmap, xscan


def xcat(
        x: typ.Union[NDArray, typ.Sequence],
        y: typ.Union[NDArray, typ.Sequence],
) -> typ.Union[NDArray, typ.Tuple]:
    """Concatenate arrays or sequences.
    """
    if is_ndarray(x):
        assert(is_ndarray(y))
        if is_jax_ndarray(x) or is_jax_ndarray(y):
            return jnp.concatenate([x, y], axis=0)
        else:
            return np.concatenate([x, y], axis=0)

    assert(isinstance(x, typ.Sequence) and isinstance(y, typ.Sequence))

    if len(x) == 0:
        return y
    elif len(y) == 0:
        return x
    else:
        return tuple(x) + tuple(y)


def xappend(
        S: typ.Union[NDArray, typ.Sequence],
        x,
) -> typ.Union[NDArray, typ.Tuple]:
    """Append slice to array or element to sequence
    """
    if is_ndarray(S):
        assert(is_ndarray(x))
        if is_jax_ndarray(S) or is_jax_ndarray(x):
            return jnp.concatenate([S, x.reshape((1,)+x.shape)], axis=0)
        else:
            return np.concatenate([S, x.reshape((1,)+x.shape)], axis=0)

    assert(isinstance(S, typ.Sequence))

    if len(S) == 0:
        return (x,)
    else:
        return tuple(S) + (x,)


def xprepend(
        x,
        S: typ.Union[NDArray, typ.Sequence],
) -> typ.Union[NDArray, typ.Tuple]:
    """Prepend slice to array or element to sequence
    """
    if is_ndarray(S):
        assert(is_ndarray(x))
        if is_jax_ndarray(S) or is_jax_ndarray(x):
            return jnp.concatenate([x.reshape((1,)+x.shape), S], axis=0)
        else:
            return np.concatenate([x.reshape((1,)+x.shape), S], axis=0)

    assert(isinstance(S, typ.Sequence))

    if len(S) == 0:
        return (x,)
    else:
        return (x,) + tuple(S)



def randn(*args, use_jax: bool):
    if use_jax:
        return jnp.array(np.random.randn(*args)) # should convert this to pure jax
    else:
        return np.random.randn(*args)


