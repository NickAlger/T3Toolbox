# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import typing as typ
import numpy as np

from t3toolbox.backend.common import *

__all__ = [
    'NDArrayTree',
    'corewise_add',
    'corewise_sub',
    'corewise_scale',
    'corewise_zeros_like',
    'corewise_map',
    'corewise_stack_scale',
    'corewise_neg',
    'corewise_sum',
    'corewise_stack_sum',
    'corewise_dot',
    'corewise_stack_dot',
    'corewise_norm',
    'corewise_stack_norm',
    'corewise_err',
    'corewise_relerr',
    'corewise_logical_not',
]


###############################################
########    Corewise linear algebra    ########
###############################################

NDArrayTree = typ.Union[int, float, NDArray, typ.List['NDArrayTree'], typ.Tuple['NDArrayTree',...]]


def corewise_add(
        X:  NDArrayTree,  # any nested tree of ints/floats/arrays
        Y:  NDArrayTree,  # same tree structure as X; leaves broadcast-compatible
) -> NDArrayTree:         # X+Y. same tree structure as X
    '''Add nested objects, X,Y -> X+Y.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> Y = (2*np.ones(3), (3, (), -np.ones(2)))
    >>> print(cw.corewise_add(X, Y))
    (array([3., 3., 3.]), (4, (), array([0., 0.])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        assert(isinstance(Y, list) or isinstance(Y, tuple))
        assert(len(X) == len(Y))
        return tuple([corewise_add(x, y) for x, y in zip(X, Y)])
    else:
        return X + Y


def corewise_sub(
        X:  NDArrayTree,  # any nested tree of ints/floats/arrays
        Y:  NDArrayTree,  # same tree structure as X; leaves broadcast-compatible
) -> NDArrayTree:         # X-Y. same tree structure as X
    '''Subtract nested objects, X,Y -> X-Y.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> Y = (2*np.ones(3), (3, (), -np.ones(2)))
    >>> print(cw.corewise_sub(X, Y))
    (array([-1., -1., -1.]), (-2, (), array([2., 2.])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        assert(isinstance(Y, list) or isinstance(Y, tuple))
        assert(len(X) == len(Y))
        return tuple([corewise_sub(x, y) for x, y in zip(X, Y)])
    else:
        return X - Y


def corewise_scale(
        X:  NDArrayTree,  # any nested tree of ints/floats/arrays
        s,                # scalar multiplier
) -> NDArrayTree:         # s*X. same tree structure as X
    '''Scale nested objects, X,s -> s*X.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> print(cw.corewise_scale(X, 1.5))
    (array([1.5, 1.5, 1.5]), (1.5, (), array([1.5, 1.5])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        return tuple([corewise_scale(x, s) for x in X])
    else:
        return s*X


def corewise_zeros_like(
        X:  NDArrayTree,  # any nested tree of ints/floats/arrays
) -> NDArrayTree:         # zeros with X's tree structure and leaf shapes (numpy/jax inferred from X)
    '''Tree of zeros matching ``X``'s structure and leaf shapes/backend (``= corewise_scale(X, 0)``).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones(3), (1.0, (), np.ones(2)))
    >>> print(cw.corewise_zeros_like(X))
    (array([0., 0., 0.]), (0.0, (), array([0., 0.])))
    '''
    return corewise_scale(X, 0)


def corewise_map(
        f,                  # callable applied to matching leaves: f(leaf_X0, leaf_X1, ...) -> leaf
        *Xs:  NDArrayTree,  # one or more identically-structured trees
) -> NDArrayTree:           # tree of f-results, same structure as the inputs
    '''Apply ``f`` elementwise over the leaves of one or more identically-structured trees (the general
    tree map the corewise ops are special cases of; used e.g. for an Adam moment update over the cores).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones(2), (3.0,))
    >>> print(cw.corewise_map(lambda x: x + 1, X))
    (array([2., 2.]), (4.0,))
    >>> print(cw.corewise_map(lambda a, b: a * b, X, X))
    (array([1., 1.]), (9.0,))
    '''
    if isinstance(Xs[0], list) or isinstance(Xs[0], tuple):
        return tuple([corewise_map(f, *xs) for xs in zip(*Xs)])
    return f(*Xs)


def corewise_neg(X: NDArrayTree) -> NDArrayTree:
    '''Negate nested objects, X -> -X.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> print(cw.corewise_neg(X))
    (array([-1., -1., -1.]), (-1, (), array([-1., -1.])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        return tuple([corewise_neg(x) for x in X])
    else:
        return -X


def corewise_sum(
        X:      NDArrayTree,                              # any nested tree of ints/floats/arrays
        axis:   typ.Union[int, typ.Sequence[int], None] = None,  # axis/axes summed in every leaf (None -> all)
) -> NDArrayTree:                                        # same tree structure as X, summed axes removed
    '''Sum each array in a nested object along the given axis or axes, X -> sum(X, axis).

    The same axis or axes are summed in every leaf array, leaving the tree structure intact.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones((2,3)), (np.ones((2,4)),))
    >>> print(cw.corewise_sum(X, axis=0))
    (array([2., 2., 2.]), (array([2., 2., 2., 2.]),))
    '''
    use_jax = tree_contains_jax(X)
    xnp, _, _ = get_backend(False, use_jax)

    if isinstance(X, list) or isinstance(X, tuple):
        return tuple([corewise_sum(x, axis=axis) for x in X])
    else:
        return xnp.sum(X, axis=axis)


def corewise_dot(
        X:  NDArrayTree,  # any nested tree of ints/floats/arrays
        Y:  NDArrayTree,  # same tree structure as X; matching leaf shapes
) -> NDArray:  # X.Y, a scalar (collapses EVERY axis, stacks included)
    '''Dot product of nested objects, X,Y -> X.Y.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> Y = (2*np.ones(3), (3, (), -np.ones(2)))
    >>> print(cw.corewise_dot(X, Y))
    7.0
    '''
    use_jax = tree_contains_jax((X, Y))
    xnp, _, _ = get_backend(False, use_jax)

    if isinstance(X, list) or isinstance(X, tuple):
        assert(isinstance(Y, list) or isinstance(Y, tuple))
        assert(len(X) == len(Y))
        return xnp.sum(xnp.array([corewise_dot(x, y) for x, y in zip(X, Y)]))
    else:
        return xnp.sum(X * Y)


def corewise_stack_dot(
        X:          NDArrayTree,  # tree; each leaf shape = stack_shape + (core dims)
        Y:          NDArrayTree,  # same tree structure / leaf shapes as X
        n_stack:    int,          # number of leading stack axes kept (contract the rest)
) -> NDArray:  # array of shape = common stack_shape (scalar when n_stack==0)
    '''Like corewise_dot, but vectorized over the leading ``n_stack`` (stack) axes.

    Each leaf is contracted over its trailing (non-stack) axes only, keeping the leading ``n_stack``
    axes, and the per-leaf results are summed. Returns an array of shape equal to the common leading
    stack shape (a scalar when ``n_stack == 0``, matching :py:func:`corewise_dot`).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones((2, 3)), np.ones((2, 4, 5)))   # stack=(2,), then core axes
    >>> Y = (2*np.ones((2, 3)), np.ones((2, 4, 5)))
    >>> print(cw.corewise_stack_dot(X, Y, 1))       # per-stack-slice: 2*3 + 4*5 = 26
    [26. 26.]
    '''
    use_jax = tree_contains_jax((X, Y))
    xnp, _, _ = get_backend(False, use_jax)

    if isinstance(X, list) or isinstance(X, tuple):
        assert(isinstance(Y, list) or isinstance(Y, tuple))
        assert(len(X) == len(Y))
        terms = [corewise_stack_dot(x, y, n_stack) for x, y in zip(X, Y)]
        out = terms[0]
        for term in terms[1:]:
            out = out + term
        return out
    else:
        return xnp.sum(X * Y, axis=tuple(range(n_stack, xnp.ndim(X))))


def corewise_stack_scale(
        X:  NDArrayTree,  # tree; each leaf shape = stack_shape + (core dims)
        s,                # per-stack-slice factor, shape = stack_shape (scalar ndim 0 = uniform)
) -> NDArrayTree:         # same tree structure as X, each leaf scaled
    '''Scale each leaf by a per-stack-slice factor ``s``, broadcasting ``s`` over each leaf's trailing
    (non-stack) axes.

    ``s`` has shape equal to the leading stack shape; each leaf has shape ``stack_shape + (core dims)``.
    A scalar ``s`` (ndim 0) scales every entry uniformly, matching :py:func:`corewise_scale`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones((2, 3)), np.ones((2, 4, 5)))   # stack=(2,), then core axes
    >>> out = cw.corewise_stack_scale(X, np.array([10., 100.]))
    >>> print(out[0][:, 0], out[1][:, 0, 0])
    [ 10. 100.] [ 10. 100.]
    '''
    use_jax = tree_contains_jax((X, s))
    xnp, _, _ = get_backend(False, use_jax)
    s = xnp.asarray(s)
    k = s.ndim
    def go(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return tuple([go(xi) for xi in x])
        return x * s.reshape(s.shape + (1,) * (xnp.ndim(x) - k))
    return go(X)


def corewise_stack_sum(
        X:          NDArrayTree,  # tree; each leaf shape = stack_shape + (core dims)
        axis,                     # stack axis/axes (None -> all; negatives wrap rel. to n_stack)
        n_stack:    int,          # number of leading stack axes axis is normalized against
) -> NDArrayTree:                 # same tree structure as X, summed stack axes removed
    '''Sum each leaf over stack axes, vectorized.

    Normalizes ``axis`` against the ``n_stack`` leading (stack) axes (``None`` -> all of them; negative
    axes wrap relative to ``n_stack``), then :py:func:`corewise_sum` over those axes.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones((2, 3, 4)),)   # stack=(2, 3), then a core axis
    >>> print(cw.corewise_stack_sum(X, None, 2)[0].shape)   # sum both stack axes
    (4,)
    '''
    if axis is None:
        stack_axes = tuple(range(n_stack))
    elif not isinstance(axis, (tuple, list)):
        stack_axes = ((axis + n_stack) if axis < 0 else axis,)
    else:
        stack_axes = tuple((ax + n_stack) if ax < 0 else ax for ax in axis)
    return corewise_sum(X, axis=stack_axes)


def corewise_norm(
        X,  # any nested tree of ints/floats/arrays
) -> NDArray:  # ||X||, a scalar (collapses EVERY axis, stacks included)
    '''Norm of nested objects, X -> ||X||

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.array([1,2,3]), (4, (), np.array([5,6])))
    >>> print(cw.corewise_norm(X))
    9.539392014169456
    >>> print(np.sqrt(1**2 + 2**2 + 3**2 + 4**2 + 5**2 + 6**2))
    9.539392014169456
    '''
    use_jax = tree_contains_jax(X)
    xnp, _, _ = get_backend(False, use_jax)
    norm_sq = corewise_dot(X, X)
    return xnp.sqrt(xnp.abs(norm_sq))


def corewise_stack_norm(
        X:          NDArrayTree,  # tree; each leaf shape = stack_shape + (core dims)
        n_stack:    int,          # number of leading stack axes kept (one norm per stack slice)
) -> NDArray:  # array of shape = common stack_shape (scalar when n_stack==0)
    '''Like :py:func:`corewise_norm`, but vectorized over the leading ``n_stack`` (stack) axes.

    Returns an array of shape equal to the common leading stack shape -- one norm per stack slice
    (a scalar when ``n_stack == 0``, matching :py:func:`corewise_norm`). It is the ``sqrt`` of the
    per-slice :py:func:`corewise_stack_dot` of ``X`` with itself. Use this (not
    :py:func:`corewise_norm`, which collapses *every* axis including the stack) for a vectorized norm.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones((2, 3)), np.ones((2, 4, 5)))   # stack=(2,), then core axes
    >>> print(cw.corewise_stack_norm(X, 1))         # sqrt(3 + 20) per slice
    [4.79583152 4.79583152]
    '''
    use_jax = tree_contains_jax(X)
    xnp, _, _ = get_backend(False, use_jax)
    return xnp.sqrt(xnp.abs(corewise_stack_dot(X, X, n_stack)))


def corewise_err(
        X_true,  # reference tree of ints/floats/arrays
        X,       # same tree structure as X_true
) -> NDArray:  # ||X_true - X||, a scalar
    return corewise_norm(corewise_sub(X_true, X))


def corewise_relerr(
        X_true,  # reference tree of ints/floats/arrays
        X,       # same tree structure as X_true
) -> NDArray:  # ||X_true - X|| / ||X_true||, a scalar
    return corewise_err(X_true, X) / corewise_norm(X_true)


def corewise_logical_not(X: NDArrayTree) -> NDArrayTree:
    '''Perform logical not operation on nested objects

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.array([True, False, False]), (True, (), np.array([False, True, False])))
    >>> print(cw.corewise_logical_not(X))
    (array([False,  True,  True]), (np.False_, (), array([ True, False,  True])))
    '''
    use_jax = tree_contains_jax(X)
    xnp, _, _ = get_backend(False, use_jax)

    if isinstance(X, list) or isinstance(X, tuple):
        if not X:
            return ()
        else:
            return tuple([corewise_logical_not(x) for x in X])
    else:
        return xnp.logical_not(X)

