# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import typing as typ
import numpy as np

from t3toolbox.common import *

__all__ = [
    'NDArrayTree',
    'corewise_add',
    'corewise_sub',
    'corewise_scale',
    'corewise_neg',
    'corewise_dot',
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
    return xnp.sqrt(corewise_dot(X, X))


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

