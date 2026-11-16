"""Draft of the documentation example for the derivative-mode adapter (target decision 2.3).

The library works in Taylor coefficients.  A user whose data are derivatives converts them, by hand with the
three standalone functions below or by passing a keyword to a fitting function, which calls the same functions.
The names here and the keyword's spelling (`jet_form='derivatives'` in this draft) are placeholders.

The maintainer asked for a doctest that shows what happens to the data in each case (2026-11-15).  This is a
draft of it, with the three functions prototyped so that the session runs.  The example uses the apply kind,
whose data have the shape T + W and whose weight has the shape T.

    >>> import numpy as np
    >>> order = 3

    The derivatives of 1 / (1 - s) at s = 0 are 0!, 1!, 2!, 3!.  Its Taylor coefficients are all 1.

    >>> data_derivatives = np.array([[1.0], [1.0], [2.0], [6.0]])       # shape (order+1, W) with W = 1
    >>> taylor_from_derivatives(data_derivatives)[:, 0]
    array([1., 1., 1., 1.])
    >>> derivatives_from_taylor(taylor_from_derivatives(data_derivatives))[:, 0]
    array([1., 1., 2., 6.])

    Case 1.  Taylor data (the default).  The fit uses the data and the weight as given; no weight means ones.

    Case 2.  Derivative data and a weight chosen for derivatives.  With the keyword, the fit uses

    >>> weight_derivatives = np.array([1.0, 10.0, 10.0, 10.0])          # shape (order+1,)
    >>> taylor_from_derivatives(data_derivatives)[:, 0]
    array([1., 1., 1., 1.])
    >>> taylor_weight_from_derivative_weight(weight_derivatives)
    array([ 1., 10., 20., 60.])

    The data are divided by t! and the weight is multiplied by it: a weight goes the opposite way.

    Case 3.  Derivative data and NO weight.  With the keyword, "no weight" means ones on the DERIVATIVES, so the
    fit uses the weight t! on the Taylor coefficients, not ones:

    >>> taylor_weight_from_derivative_weight(np.ones(order + 1))
    array([1., 1., 2., 6.])

    In every case the objective is the one the user wrote down.  For a model whose Taylor coefficients are z:

    >>> z = np.array([[1.0], [0.9], [1.2], [0.8]])
    >>> objective = lambda weight, prediction, data: 0.5 * float(np.sum((weight[:, None] * (prediction - data)) ** 2))
    >>> in_derivatives = objective(weight_derivatives, derivatives_from_taylor(z), data_derivatives)
    >>> in_taylor = objective(taylor_weight_from_derivative_weight(weight_derivatives), z,
    ...                       taylor_from_derivatives(data_derivatives))
    >>> bool(np.isclose(in_derivatives, in_taylor))
    True

    Without the keyword the same derivative data would be read as Taylor coefficients, which is a different
    problem: the order-3 target would be 6 where it should be 1.

Usage:  python3 -I adapter_doctest_draft.py
"""
import math

import numpy as np


def _factorials(n_orders: int) -> np.ndarray:        # shape=(n_orders,): 0!, 1!, ..., (n_orders-1)!
    return np.array([float(math.factorial(t)) for t in range(n_orders)])


def taylor_from_derivatives(
        jets: np.ndarray,       # derivatives,         shape=(order+1,)+...  (the order axis leads)
) -> np.ndarray:                # Taylor coefficients, same shape: slot t divided by t!
    return jets / _factorials(jets.shape[0]).reshape((-1,) + (1,) * (jets.ndim - 1))


def derivatives_from_taylor(
        jets: np.ndarray,       # Taylor coefficients, shape=(order+1,)+...  (the order axis leads)
) -> np.ndarray:                # derivatives,         same shape: slot t multiplied by t!
    return jets * _factorials(jets.shape[0]).reshape((-1,) + (1,) * (jets.ndim - 1))


def taylor_weight_from_derivative_weight(
        weight: np.ndarray,     # a weight on derivative residuals,     shape=...+(order+1,)  (the order axis is last)
) -> np.ndarray:                # the same objective's weight on Taylor residuals, same shape: slot t multiplied by t!
    return weight * _factorials(weight.shape[-1])


if __name__ == '__main__':
    import doctest
    result = doctest.testmod()
    print('the doctest of the adapter draft: %d examples, %d failures' % (result.attempted, result.failed))
