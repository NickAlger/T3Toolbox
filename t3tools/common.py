# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np

__all__ = [
    # 'jnp',
    # 'NDArray',
    'numpy_scan',
    # 'jax_scan',
    #
    #
]


def numpy_scan(f, init, xs, length=None):
    """Numpy version of jax.lax.scan.
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
    Generated with help from AI.
    """
    if length is None:
        def get_length(tree):
            if isinstance(tree, (list, tuple)): return get_length(tree[0])
            return len(tree)

        length = get_length(xs)

    def get_slice(tree, i):
        if isinstance(tree, tuple):
            return tuple(get_slice(x, i) for x in tree)
        if isinstance(tree, list):
            return [get_slice(x, i) for x in tree]
        return tree[i]

    carry = init
    ys_list = []

    for i in range(length):
        current_xs = get_slice(xs, i)
        carry, y = f(carry, current_xs)
        ys_list.append(y)

    if isinstance(ys_list[0], (tuple, list)):
        return carry, tuple(np.stack([step[i] for step in ys_list]) for i in range(len(ys_list[0])))

    return carry, np.stack(ys_list)


# try:
#     import jax.numpy as jnp
#     import jax
#     jax_scan = jax.lax.scan
# except ImportError:
#     print('jax import failed. Defaulting to numpy.')
#     jnp = np
#     jax_scan = numpy_scan
#
# NDArray = typ.Union[np.ndarray, jnp.ndarray]





