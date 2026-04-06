# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ
import t3tools.tucker_tensor_train as t3

try:
    import jax.numpy as jnp
except:
    print('jax import failed in tucker_tensor_train. Defaulting to numpy.')
    jnp = np

NDArray = typ.Union[np.ndarray, jnp.ndarray]


__all__ = [
    'corewise_add',
    'corewise_scale',
    'corewise_neg',
    'corewise_sub',
    'corewise_dot',
    'corewise_norm',
]


def corewise_add(X, Y):
    '''Add nested objects, X,Y -> X+Y.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
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


def corewise_sub(X, Y):
    '''Subtract nested objects, X,Y -> X-Y.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
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


def corewise_scale(X, s):
    '''Scale nested objects, X,s -> s*X.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> print(cw.corewise_scale(X, 1.5))
    (array([-1., -1., -1.]), (-2, (), array([2., 2.])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        return tuple([corewise_scale(x, s) for x in X])
    else:
        return s*X


def corewise_neg(X):
    '''Negate nested objects, X -> -X.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> print(cw.corewise_neg(X))
    (array([-1., -1., -1.]), (-1, (), array([-1., -1.])))
    '''
    if isinstance(X, list) or isinstance(X, tuple):
        return tuple([corewise_neg(x) for x in X])
    else:
        return -X


def corewise_dot(X, Y, use_jax: bool = False):
    '''Dot product of nested objects, X,Y -> X.Y.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> Y = (2*np.ones(3), (3, (), -np.ones(2)))
    >>> print(cw.corewise_dot(X, Y))
    7.0
    '''
    xnp = jnp if use_jax else np
    if isinstance(X, list) or isinstance(X, tuple):
        assert(isinstance(Y, list) or isinstance(Y, tuple))
        assert(len(X) == len(Y))
        return xnp.sum(xnp.array([corewise_dot(x, y) for x, y in zip(X, Y)]))
    else:
        return xnp.sum(X * Y)


def corewise_norm(X, use_jax: bool = False):
    '''Norm of nested objects, X -> ||X||

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.corewise as cw
    >>> X = (np.ones(3), (1, (), np.ones(2)))
    >>> print(cw.corewise_norm(X))
    2.449489742783178
    >>> print(np.sqrt(3 + 1 + 2))
    2.449489742783178
    '''
    xnp = jnp if use_jax else np
    return xnp.sqrt(corewise_dot(X, X, use_jax=use_jax))

