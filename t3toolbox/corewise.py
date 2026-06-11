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
    'corewise_neg',
    'corewise_sum',
    'corewise_dot',
    'corewise_stack_dot',
    'corewise_norm',
    'corewise_err',
    'corewise_relerr',
    'corewise_logical_not',
]


###############################################
########    Corewise linear algebra    ########
###############################################

NDArrayTree = typ.Union[int, float, NDArray, typ.List['NDArrayTree'], typ.Tuple['NDArrayTree',...]]


def corewise_add(X: NDArrayTree, Y: NDArrayTree) -> NDArrayTree:
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


def corewise_sub(X: NDArrayTree, Y: NDArrayTree) -> NDArrayTree:
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


def corewise_scale(X: NDArrayTree, s) -> NDArrayTree:
    '''Scale nested objects, X,s -> s*X.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> print(cw.corewise_scale(X, 1.5))
    (array([-1., -1., -1.]), (-2, (), array([2., 2.])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        return tuple([corewise_scale(x, s) for x in X])
    else:
        return s*X


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


def corewise_sum(X: NDArrayTree, axis=None, use_jax: bool=False) -> NDArrayTree:
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
    xnp, _, _ = get_backend(False, use_jax)

    if isinstance(X, list) or isinstance(X, tuple):
        return tuple([corewise_sum(x, axis=axis, use_jax=use_jax) for x in X])
    else:
        return xnp.sum(X, axis=axis)


def corewise_dot(X: NDArrayTree, Y: NDArrayTree, use_jax: bool=False):
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
    xnp, _, _ = get_backend(False, use_jax)

    if isinstance(X, list) or isinstance(X, tuple):
        assert(isinstance(Y, list) or isinstance(Y, tuple))
        assert(len(X) == len(Y))
        return xnp.sum(xnp.array([corewise_dot(x, y) for x, y in zip(X, Y)]))
    else:
        return xnp.sum(X * Y)


def corewise_stack_dot(X: NDArrayTree, Y: NDArrayTree, n_stack: int, use_jax: bool=False):
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
    xnp, _, _ = get_backend(False, use_jax)

    if isinstance(X, list) or isinstance(X, tuple):
        assert(isinstance(Y, list) or isinstance(Y, tuple))
        assert(len(X) == len(Y))
        terms = [corewise_stack_dot(x, y, n_stack, use_jax=use_jax) for x, y in zip(X, Y)]
        out = terms[0]
        for term in terms[1:]:
            out = out + term
        return out
    else:
        return xnp.sum(X * Y, axis=tuple(range(n_stack, xnp.ndim(X))))


def corewise_norm(X, use_jax: bool=False):
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
    xnp, _, _ = get_backend(False, use_jax)
    norm_sq = corewise_dot(X, X)
    return xnp.sqrt(xnp.abs(norm_sq))


def corewise_err(X_true, X, use_jax: bool=False):
    xnp, _, _ = get_backend(False, use_jax)
    return corewise_norm(corewise_sub(X_true, X), use_jax=use_jax)


def corewise_relerr(X_true, X, use_jax:bool = False):
    xnp, _, _ = get_backend(False, use_jax)
    return corewise_err(X_true, X) / corewise_norm(X_true)


def corewise_logical_not(X: NDArrayTree, use_jax: bool=False) -> NDArrayTree:
    '''Perform logical not operation on nested objects

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> X = (np.array([True, False, False]), (True, (), np.array([False, True, False])))
    >>> print(cw.corewise_logical_not(X))
    (array([False,  True,  True]), (False, (), array([ True, False,  True])))
    '''
    xnp, _, _ = get_backend(False, use_jax)

    if isinstance(X, list) or isinstance(X, tuple):
        if not X:
            return ()
        else:
            return tuple([corewise_logical_not(x) for x in X])
    else:
        return xnp.logical_not(X)

