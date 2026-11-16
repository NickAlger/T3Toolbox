"""Draft of the documentation example for derivative data (target decisions 2.2 and 2.3, as amended on 2026-11-16).

The library works in Taylor coefficients, everywhere.  A user whose data are derivatives converts them before
fitting, with the standalone functions below.  No fitting function, oracle or sampling method has a keyword for
this (the keyword of the first draft was removed on 2026-11-16).  The names here are placeholders.

The maintainer asked for a doctest that shows what happens to the data in each case (2026-11-15).  This is a
draft of it, with the functions prototyped so that the session runs.  The example uses the apply kind, whose
data have the shape T + W and whose weight has the shape T.

    >>> import numpy as np
    >>> order = 3

    The derivatives of 1 / (1 - s) at s = 0 are 0!, 1!, 2!, 3!.  Its Taylor coefficients are all 1.

    >>> data_derivatives = np.array([[1.0], [1.0], [2.0], [6.0]])       # shape (order+1, W) with W = 1
    >>> taylor_from_derivatives(data_derivatives)[:, 0]
    array([1., 1., 1., 1.])
    >>> derivatives_from_taylor(taylor_from_derivatives(data_derivatives))[:, 0]
    array([1., 1., 2., 6.])

    Case 1.  Taylor data.  Nothing to do: the fit uses the data and the weight as given, and no weight means ones.

    Case 2.  Derivative data and a weight chosen for derivatives.  One call prepares both:

    >>> weight_derivatives = np.array([1.0, 10.0, 10.0, 10.0])          # shape (order+1,)
    >>> data, weight, _ = prepare_derivative_data(data_derivatives, weight_derivatives)
    >>> data[:, 0]
    array([1., 1., 1., 1.])
    >>> weight
    array([ 1., 10., 20., 60.])

    The data are divided by t! and the weight is multiplied by it: a weight goes the opposite way.

    Case 3.  Derivative data and NO weight.  The trap: ones on the DERIVATIVES are not ones on the Taylor
    coefficients.  The same call returns the weight t!, which the user then passes to the fit:

    >>> data, weight, _ = prepare_derivative_data(data_derivatives)
    >>> weight
    array([1., 1., 2., 6.])

    Validation data given to the same call are converted like the data:

    >>> _, _, validation = prepare_derivative_data(data_derivatives, validation=2.0 * data_derivatives)
    >>> validation[:, 0]
    array([2., 2., 2., 2.])

    In every case the objective is the one the user wrote down.  For a model whose Taylor coefficients are z:

    >>> z = np.array([[1.0], [0.9], [1.2], [0.8]])
    >>> objective = lambda weight, prediction, data: 0.5 * float(np.sum((weight[:, None] * (prediction - data)) ** 2))
    >>> in_derivatives = objective(weight_derivatives, derivatives_from_taylor(z), data_derivatives)
    >>> data, weight, _ = prepare_derivative_data(data_derivatives, weight_derivatives)
    >>> bool(np.isclose(in_derivatives, objective(weight, z, data)))
    True

    Derivative data passed to a fit unconverted are read as Taylor coefficients, which is a different problem:
    the order-3 target would be 6 where it should be 1.  The library cannot detect this.

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


def prepare_derivative_data(
        data:        np.ndarray,                    # derivatives, shape=(order+1,)+...
        weight:      np.ndarray = None,             # a weight on derivative residuals, shape=...+(order+1,); None: ones
        validation:  np.ndarray = None,             # derivatives, shape=(order+1,)+...; None: no validation data
):                                                  # -> (Taylor data, Taylor weight, Taylor validation data or None)
    if weight is None:
        weight = np.ones(data.shape[0])             # ones on the derivatives, so t! on the Taylor coefficients
    return (taylor_from_derivatives(data), taylor_weight_from_derivative_weight(weight),
            None if validation is None else taylor_from_derivatives(validation))


if __name__ == '__main__':
    import doctest
    result = doctest.testmod()
    print('the doctest of the derivative data draft: %d examples, %d failures' % (result.attempted, result.failed))
