# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Jets of the sampling outputs: the dual jet, derivative data, and the checkers of a sample, its data and weight.

A **jet** here is what a sampling function returns at ``order >= 1``: the Taylor coefficients of its output along
the line ``w + s p``, the coefficient of ``s^t`` in slot ``t`` of a leading order axis ``T = (order + 1,)``. At
``order = 0`` there is no order axis, ``T = ()``. An object "of the type of ``z``" is one output of a sampling
kind: for ``'probe'`` a ``len=d`` tuple or list, one array of shape ``T + W + (Ni,)`` per mode; for ``'apply'``
and ``'entries'`` one array of shape ``T + W``. The data ``y`` of a fit, the residual ``r = z - y`` and a weight's
blocks follow that type.

**``kind`` says which type an object is of**, in every function that works on one; the object itself is never
looked at to decide it. For ``'probe'`` the functions that compute (:py:func:`jet_to_dual_jet`,
:py:func:`dual_jet_to_jet`, :py:func:`derivatives_to_taylor`, :py:func:`taylor_to_derivatives`,
:py:func:`prepare_derivative_data`) go over the modes along the first axis of the object and return a tuple, one
array per mode. So a tuple, a list, and an array packed for the uniform layout, ``(d,) + T + W + ... + (N,)``,
are all taken mode by mode, and the packed array gives what ``tuple(packed)`` gives (``xnp.stack`` packs the
result again). For ``'apply'`` and ``'entries'`` the object is one array whose first axis is the order axis. An
object of the other type than ``kind`` says is not guarded against: a tuple given with ``kind='apply'`` is made
into one array, which fails where the modes differ in size and otherwise reads the mode axis as the order axis.
The checkers take the ragged form only (a tuple or a list for the probe kind).

Four groups of functions:

- **The input jets.** A sampling function with a jet shape takes input jets, the Taylor coefficients of its input
  vectors as polynomials in the jet variables. :py:func:`affine_jet` makes the input jet of a point moved along
  directions, ``w + s_1 p_1 + ... + s_m p_m`` (the line ``w + s p`` with one), and :py:func:`index_to_one_hot` the
  one-hot vectors of a grid point, the apply kind's sample of an entry.

- **The dual jet.** :py:func:`jet_to_dual_jet` makes the seed of a transpose from a residual: the weights
  (``weight**2``, slot by slot), then the turn of the order axis. :py:func:`dual_jet_to_jet` is its inverse.
- **Derivative data.** The library works in Taylor coefficients, everywhere. A user whose data are derivatives
  converts them before fitting: :py:func:`derivatives_to_taylor`, :py:func:`taylor_to_derivatives`,
  :py:func:`derivative_weight_to_taylor_weight`, and :py:func:`prepare_derivative_data`, which does the data, the
  validation data and the weight in one call. No fitting function, oracle or sampling method has a keyword for
  derivative data.
- **Checkers.** :py:func:`check_sample`, :py:func:`check_data` and :py:func:`check_weight` raise a ``ValueError``
  that names what is wrong with the form or the shape of a sample, its data or its weight;
  :py:func:`check_perturbation_vectors` and :py:func:`check_perturbation_index` check the directions ``pp``
  against the inputs. They are checkers, not guarantees: no backend function calls them. A backend function given
  a wrong shape computes a wrong answer or fails deep inside; call the checker first if you want the error.
  Shapes are static, so every checker also runs under ``jax.jit``.

**The weight** of a fit has exactly the shape of the mode and order axes of the data: ``(d,) + T`` for
``'probe'`` and ``T`` for ``'apply'`` and ``'entries'``. There is no rule for a bare vector and no broadcast of a
mode dimension of 1. The misfit is ``1/2 sum (weight * r)**2`` over modes, orders and samples, on Taylor
coefficient residuals. **With no weight given (``None``) the weight is ones**: the plain sum of squares of the
Taylor coefficient residuals. That default may change in a later release, if a more robust choice that just
works is found; pass a weight to be independent of it.

Examples
--------
Derivative data. The example uses the apply kind, whose data have the shape ``T + W`` and whose weight has the
shape ``T``.

>>> import numpy as np
>>> import t3toolbox.backend.jets as jets
>>> order = 3

The derivatives of ``1 / (1 - s)`` at ``s = 0`` are ``0!, 1!, 2!, 3!``. Its Taylor coefficients are all 1.

>>> y_derivatives = np.array([[1.0], [1.0], [2.0], [6.0]])              # shape T + W, with W = (1,)
>>> print(jets.derivatives_to_taylor('apply', order, y_derivatives)[:, 0].tolist())
[1.0, 1.0, 1.0, 1.0]
>>> y_back = jets.taylor_to_derivatives('apply', order, jets.derivatives_to_taylor('apply', order, y_derivatives))
>>> print(y_back[:, 0].tolist())
[1.0, 1.0, 2.0, 6.0]

Case 1. Taylor data. Nothing to do: the fit uses the data and the weight as given, and no weight means ones.

Case 2. Derivative data and a weight chosen for derivatives. One call prepares both:

>>> weight_derivatives = np.array([1.0, 10.0, 10.0, 10.0])              # shape T
>>> y, weight, _ = jets.prepare_derivative_data('apply', order, y_derivatives, weight_derivatives)
>>> print(y[:, 0].tolist())
[1.0, 1.0, 1.0, 1.0]
>>> print(weight.tolist())
[1.0, 10.0, 20.0, 60.0]

The data are divided by ``t!`` and the weight is multiplied by it: a weight goes the opposite way.

Case 3. Derivative data and NO weight. The trap: ones on the DERIVATIVES are not ones on the Taylor
coefficients. The same call returns the weight ``t!``, which you then pass to the fit:

>>> y, weight, _ = jets.prepare_derivative_data('apply', order, y_derivatives)
>>> print(weight.tolist())
[1.0, 1.0, 2.0, 6.0]

Validation data given to the same call are converted like the data:

>>> _, _, validation = jets.prepare_derivative_data('apply', order, y_derivatives, validation=2.0 * y_derivatives)
>>> print(validation[:, 0].tolist())
[2.0, 2.0, 2.0, 2.0]

In every case the objective is the one you wrote down. For a model whose Taylor coefficients are ``z``:

>>> z = np.array([[1.0], [0.9], [1.2], [0.8]])
>>> objective = lambda weight, prediction, data: 0.5 * float(np.sum((weight[:, None] * (prediction - data)) ** 2))
>>> in_derivatives = objective(weight_derivatives, jets.taylor_to_derivatives('apply', order, z), y_derivatives)
>>> y, weight, _ = jets.prepare_derivative_data('apply', order, y_derivatives, weight_derivatives)
>>> print(bool(np.isclose(in_derivatives, objective(weight, z, y))))
True

Derivative data passed to a fit unconverted are read as Taylor coefficients, which is a different problem: the
order-3 target would be 6 where it should be 1. The library cannot detect this.

The probe kind has one array per mode, ``T + W + (Ni,)``, and its weight has a mode axis in front of the order
axis, ``(d,) + T``. With no weight, every mode gets ``t!``:

>>> np.random.seed(0)
>>> y_derivatives = [np.random.randn(4, 5, N) for N in (6, 7)]          # d = 2 modes, T = (4,), W = (5,)
>>> y, weight, _ = jets.prepare_derivative_data('probe', order, y_derivatives)
>>> print([y_i.shape for y_i in y])
[(4, 5, 6), (4, 5, 7)]
>>> print(weight.tolist())
[[1.0, 1.0, 2.0, 6.0], [1.0, 1.0, 2.0, 6.0]]
>>> print(bool(np.allclose(y[1][3], y_derivatives[1][3] / 6.0)))        # mode 1, order 3: divided by 3!
True

A weight per mode and order, chosen for derivatives, is converted slot by slot along its order axis:

>>> weight_derivatives = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
>>> _, weight, _ = jets.prepare_derivative_data('probe', order, y_derivatives, weight_derivatives)
>>> print(weight.tolist())
[[1.0, 1.0, 2.0, 6.0], [2.0, 2.0, 4.0, 12.0]]
"""
import itertools
import math
import typing as typ

import numpy as np

from t3toolbox.backend.common import *

__all__ = [
    # The dual jet
    'jet_to_dual_jet',
    'dual_jet_to_jet',
    # Derivative data
    'derivatives_to_taylor',
    'taylor_to_derivatives',
    'derivative_weight_to_taylor_weight',
    'prepare_derivative_data',
    # The input jets
    'affine_jet',
    'index_to_one_hot',
    # The checkers of a sample, its data and its weight
    'check_perturbation_vectors',
    'check_perturbation_index',
    'check_sample',
    'check_data',
    'check_weight',
]

_KINDS = ('probe', 'apply', 'entries')


#########################################
########    Private helpers    ##########
#########################################

def _require_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError("unknown sampling kind %r: expected 'probe', 'apply' or 'entries'" % (kind,))


def _lowest_precision_dtype(
        xnp,                        # numpy or jax.numpy: the module the result will be made in
        *arrays: typ.Any,
) -> typ.Optional[np.dtype]:
    '''The floating dtype of lowest precision among ``arrays`` (the absorbing rule: a result is no more precise
    than its inputs), or None if none of them is a floating array. Objects without a dtype do not count. On jax
    the result is no more precise than jax's own floating dtype either (float32 unless x64 is on), so a numpy
    float64 array that meets a jax array is never asked of jax in a dtype jax does not have.'''
    dtypes = [np.dtype(a.dtype) for a in arrays
              if hasattr(a, 'dtype') and np.issubdtype(np.dtype(a.dtype), np.floating)]
    if dtypes and xnp is not np:
        dtypes.append(np.dtype(xnp.result_type(float)))
    return min(dtypes, key=lambda dtype: dtype.itemsize) if dtypes else None


def _arrays(kind: str, value: typ.Any) -> typ.List[typ.Any]:
    '''The arrays of an object of the type of ``z``, as ``kind`` says: for ``'probe'`` one per mode, taken along
    the first axis of the object (the elements of a tuple or a list, or the slices of a packed array); for
    ``'apply'`` and ``'entries'`` the object itself, one array.'''
    return [value[i] for i in range(len(value))] if kind == 'probe' else [value]


def _of_type_of_z(kind: str, arrays: typ.List[typ.Any]) -> typ.Any:
    '''The inverse of :py:func:`_arrays`: a tuple, one array per mode, for ``'probe'``; the one array otherwise.'''
    return tuple(arrays) if kind == 'probe' else arrays[0]


def _scale_orders(
        kind:     str,              # 'probe' | 'apply' | 'entries'
        jets:     typ.Any,          # of the type of z: probe one array T+X per mode; apply/entries one array T+X
        order:    int,
        divide:   bool,             # True: slot t divided by t!; False: multiplied
        xnp,                        # numpy or jax.numpy, chosen once over all the inputs of the call
) -> typ.Any:                       # of the type of z, same shapes (probe: a tuple)
    '''Slot ``t`` of every order axis divided or multiplied by ``t!``. Nothing at order 0.'''
    out = []
    for jet_TX in _arrays(kind, jets):
        jet_TX = xnp.asarray(jet_TX)
        if order:
            factorials = [float(math.factorial(t)) for t in range(order + 1)]
            factorials_T = xnp.asarray(factorials, dtype=_lowest_precision_dtype(xnp, jet_TX))
            factorials_T = factorials_T.reshape((order + 1,) + (1,) * (jet_TX.ndim - 1))
            jet_TX = xnp.asarray(jet_TX / factorials_T if divide else jet_TX * factorials_T)
        out.append(jet_TX)
    return _of_type_of_z(kind, out)


def _taylor_weight(
        weight:  typ.Any,           # a weight on derivative residuals: probe (d,)+T, apply/entries T
        order:   int,
        xnp,
) -> NDArray:                       # same shape: slot t of the last axis multiplied by t!
    weight_xT = xnp.asarray(weight)
    if order == 0:
        return weight_xT
    factorials = [float(math.factorial(t)) for t in range(order + 1)]
    return weight_xT * xnp.asarray(factorials, dtype=_lowest_precision_dtype(xnp, weight_xT))


def _squared_weights(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        jet:     typ.Any,                   # of the type of z (a jet or a dual jet)
        weight:  typ.Optional[typ.Any],     # probe (d,)+T; apply/entries T; None = ones
        xnp,
) -> typ.List[typ.Optional[NDArray]]:       # per array of jet: weight**2, shape=T+(1,)*len(X), or None for no weight
    '''``weight**2`` for each array of an object of the type of ``z``, shaped to act on it slot by slot, in the
    dtype the product or the quotient is computed in (the lower floating precision of the array and the weight).
    For ``'probe'`` row ``i`` of the weight is mode ``i``'s. This is the divisor of :py:func:`dual_jet_to_jet`,
    exactly as it is computed: a weight whose square underflows in that dtype gives a zero here.'''
    out = []
    for i, jet_TX in enumerate(_arrays(kind, jet)):
        if weight is None:
            out.append(None)
            continue
        weight_T = weight[i] if kind == 'probe' else weight
        omega_T = xnp.asarray(weight_T, dtype=_lowest_precision_dtype(xnp, jet_TX, weight_T))
        out.append((omega_T * omega_T).reshape(tuple(omega_T.shape) + (1,) * (len(np.shape(jet_TX)) - omega_T.ndim)))
    return out


####################################
########    The dual jet    ########
####################################

def jet_to_dual_jet(
        kind:    str,                      # 'probe' | 'apply' | 'entries': which type the jet is of
        order:   int,                      # highest Taylor order. T=(order+1,) if order else ()
        jet:     typ.Any,                  # of the type of z: probe len=d, elm_shape=T+W+[K]+C+(Ni,); apply/entries T+W+[K]+C
        weight:  typ.Optional[NDArray],    # probe (d,)+T; apply/entries T; None = ones
) -> typ.Any:                              # the dual jet, same shapes: weight**2 * jet, then the order axis turned. probe: a tuple
    """Make a jet into a dual jet: the weights, then the turn of the order axis.

    A residual ``r`` is a jet, in forward order. The seed of a transpose is a dual jet, the functional
    ``a -> sum_t weight[t]**2 * r[t] * a[t]`` on jets. It is stored turned,

        ``dual_jet[t] = weight[order - t]**2 * jet[order - t]``,

    so that its pairing with a jet ``a`` is the coefficient of ``s^order`` of the product of the two as
    polynomials in ``s``: ``sum_t dual_jet[order - t] * a[t]``. The weight enters squared, since the misfit is
    ``1/2 sum (weight * r)**2``. At ``order = 0`` there is nothing to turn: the dual jet is ``weight**2 * jet``.

    **``kind`` says which type the jet is of**; the object is not looked at to decide it. For ``'probe'`` the
    jet is one array per mode: the function goes over the modes along the first axis of ``jet``, row ``i`` of
    the weight acts on mode ``i``, and the result is a tuple, one array per mode. So a tuple, a list, and an
    array packed for the uniform layout, ``(d,) + T + W + ... + (N,)``, are all taken mode by mode, and the
    packed array gives what ``tuple(packed)`` gives. For ``'apply'`` and ``'entries'`` the jet is one array,
    its first axis the order axis. A tuple given with ``kind='apply'`` is not of the apply type: it is made into
    one array, which fails where the modes have different sizes and otherwise reads the mode axis as the order
    axis (an error from broadcasting, or a wrong result); nothing here checks for it.

    The result is on jax if any array of the jet or the weight is a jax array (then every array of the result
    is), and has the lower floating precision of the jet and the weight.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: int
        The highest Taylor order.
    jet: of the type of z
        probe: ``len=d``, ``elm_shape=T+W+[K]+C+(Ni,)``. apply and entries: ``shape=T+W+[K]+C``.
    weight: NDArray or None
        probe: ``shape=(d,)+T``. apply and entries: ``shape=T``. ``None`` is ones, the library's default
        weight (a default that may change in a later release). The argument is required: pass ``None`` for
        no weight. No other shape is accepted (:py:func:`check_weight` tests it; this function does not).

    Returns
    -------
    of the type of z
        The dual jet, with the shapes of ``jet``. probe: a tuple, one array per mode.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds.

    See Also
    --------
    dual_jet_to_jet
    check_weight

    Examples
    --------
    One sample of the apply kind at order 2, so the jet has shape ``T + W = (3, 1)``:

    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> r = np.array([[1.0], [2.0], [3.0]])
    >>> weight = np.array([1.0, 2.0, 10.0])
    >>> dual = jets.jet_to_dual_jet('apply', 2, r, weight)
    >>> print(dual[:, 0].tolist())                          # weight**2 * r is (1, 8, 300); then turned
    [300.0, 8.0, 1.0]

    The pairing with a jet ``a`` is the top coefficient of the product, and equals the weighted inner product:

    >>> a = np.array([[0.5], [-1.0], [2.0]])
    >>> top_coefficient = sum(dual[2 - t] * a[t] for t in range(3))
    >>> print(bool(np.allclose(top_coefficient, np.sum(weight[:, None] ** 2 * r * a, axis=0))))
    True

    For the probe kind the jet is one array per mode and the weight has a row per mode:

    >>> np.random.seed(0)
    >>> r = [np.random.randn(3, 4, N) for N in (5, 6)]      # d = 2, T = (3,), W = (4,)
    >>> weight = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    >>> dual = jets.jet_to_dual_jet('probe', 2, r, weight)
    >>> print(type(dual).__name__, [dual_i.shape for dual_i in dual])
    tuple [(3, 4, 5), (3, 4, 6)]
    >>> print(bool(np.allclose(dual[1][0], 9.0 * r[1][2])))  # mode 1: slot 0 holds order 2, weight 3
    True

    A probe jet packed for the uniform layout, one array ``(d,) + T + W + (N,)``, is taken mode by mode as
    well, and comes back as a tuple (``np.stack`` packs it again):

    >>> packed = np.random.randn(2, 3, 4, 6)                # d = 2, T = (3,), W = (4,), N = 6
    >>> dual = jets.jet_to_dual_jet('probe', 2, packed, weight)
    >>> print(type(dual).__name__, [dual_i.shape for dual_i in dual])
    tuple [(3, 4, 6), (3, 4, 6)]
    >>> by_mode = jets.jet_to_dual_jet('probe', 2, tuple(packed), weight)
    >>> print(all(bool(np.array_equal(a, b)) for a, b in zip(dual, by_mode)))
    True
    >>> print(bool(np.allclose(np.stack(dual)[1, 0], 9.0 * packed[1, 2])))
    True
    """
    _require_kind(kind)
    xnp, _, _ = get_backend(False, tree_contains_jax((jet, weight)))        # once, over all the inputs
    out = []
    for jet_TX, weight_squared_T in zip(_arrays(kind, jet), _squared_weights(kind, jet, weight, xnp)):
        weighted_TX = xnp.asarray(jet_TX, dtype=_lowest_precision_dtype(xnp, jet_TX, weight))
        if weight_squared_T is not None:
            weighted_TX = weight_squared_T * weighted_TX
        out.append(xnp.asarray(weighted_TX[::-1] if order else weighted_TX))
    return _of_type_of_z(kind, out)


def dual_jet_to_jet(
        kind:      str,                    # 'probe' | 'apply' | 'entries': which type the dual jet is of
        order:     int,                    # highest Taylor order. T=(order+1,) if order else ()
        dual_jet:  typ.Any,                # same type and shapes as jet_to_dual_jet's result
        weight:    typ.Optional[NDArray],  # probe (d,)+T; apply/entries T; None = ones
) -> typ.Any:                              # the jet: the order axis turned back, then divided by weight**2. probe: a tuple
    """Make a dual jet into a jet: the inverse of :py:func:`jet_to_dual_jet`.

    The order axis is turned back and each slot is divided by ``weight**2``:

        ``jet[t] = dual_jet[order - t] / weight[t]**2``.

    **Where a weight is zero this divides by zero**: the result is ``inf`` or ``nan`` there, and nothing is
    raised or warned, on numpy and on jax alike. The dual jet of a zero weight keeps nothing of the jet, so there
    is nothing to give back. The divisor is ``weight**2`` in the precision of the computation (the lower
    floating precision of the dual jet and the weight), so a weight whose square underflows there divides by
    zero too. Test the weight first if that matters (the frontend function of the same name,
    :py:func:`t3toolbox.fitting.dual_jet_to_jet`, raises in safe mode where that divisor is zero).

    ``kind`` says which type the dual jet is of, as for :py:func:`jet_to_dual_jet`: for ``'probe'`` one array
    per mode, taken along the first axis (a tuple, a list, or an array packed for the uniform layout), and a
    tuple is returned; for ``'apply'`` and ``'entries'`` one array, its first axis the order axis.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: int
        The highest Taylor order.
    dual_jet: of the type of z
        probe: ``len=d``, ``elm_shape=T+W+[K]+C+(Ni,)``. apply and entries: ``shape=T+W+[K]+C``.
    weight: NDArray or None
        probe: ``shape=(d,)+T``. apply and entries: ``shape=T``. ``None`` is ones, the library's default
        weight (a default that may change in a later release). The argument is required: pass ``None`` for
        no weight.

    Returns
    -------
    of the type of z
        The jet, with the shapes of ``dual_jet``. probe: a tuple, one array per mode.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds.

    See Also
    --------
    jet_to_dual_jet

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> np.random.seed(0)
    >>> r = np.random.randn(3, 4)                           # apply: T + W = (3, 4)
    >>> weight = np.array([1.0, 2.0, 10.0])
    >>> dual = jets.jet_to_dual_jet('apply', 2, r, weight)
    >>> print(bool(np.allclose(jets.dual_jet_to_jet('apply', 2, dual, weight), r)))
    True

    A zero weight divides by zero, without an error:

    >>> back = jets.dual_jet_to_jet('apply', 2, dual, np.array([1.0, 0.0, 10.0]))
    >>> print(np.isfinite(back).all(axis=1).tolist())       # order 1 is lost
    [True, False, True]
    """
    _require_kind(kind)
    use_jax = tree_contains_jax((dual_jet, weight))                         # once, over all the inputs
    xnp, _, _ = get_backend(False, use_jax)
    out = []
    for dual_jet_TX, weight_squared_T in zip(_arrays(kind, dual_jet), _squared_weights(kind, dual_jet, weight, xnp)):
        weighted_TX = xnp.asarray(dual_jet_TX, dtype=_lowest_precision_dtype(xnp, dual_jet_TX, weight))
        weighted_TX = weighted_TX[::-1] if order else weighted_TX
        if weight_squared_T is None:
            jet_TX = weighted_TX
        elif use_jax:
            jet_TX = weighted_TX / weight_squared_T
        else:
            with np.errstate(divide='ignore', invalid='ignore'):    # a zero divisor: inf or nan, silently, as on jax
                jet_TX = weighted_TX / weight_squared_T
        out.append(xnp.asarray(jet_TX))
    return _of_type_of_z(kind, out)


#######################################
########    Derivative data    ########
#######################################

def derivatives_to_taylor(
        kind:   str,                       # 'probe' | 'apply' | 'entries': which type the jets are of
        order:  int,                       # highest order. T=(order+1,) if order else ()
        jets:   typ.Any,                   # derivatives: probe len=d, elm_shape=T+W+(Ni,); apply/entries T+W
) -> typ.Any:                              # Taylor coefficients, same shapes: slot t divided by t!. probe: a tuple
    """Derivatives to Taylor coefficients: slot ``t`` of the order axis divided by ``t!``.

    The library holds a jet as Taylor coefficients, the coefficient of ``s^t`` in slot ``t``. The ``t``-th
    derivative at ``s = 0`` is ``t!`` times it. Data measured as derivatives are converted with this function
    before they are given to a fit (or with :py:func:`prepare_derivative_data`, which also takes care of the
    weight). At ``order = 0`` nothing changes.

    ``kind`` says which type the jets are of. For ``'probe'`` one array per mode, taken along the first axis of
    ``jets`` (a tuple, a list, or an array packed for the uniform layout, ``(d,) + T + ...``), and a tuple is
    returned; for ``'apply'`` and ``'entries'`` one array, its first axis the order axis. A tuple given with
    ``kind='apply'`` is not of the apply type (see :py:func:`jet_to_dual_jet`); nothing here checks for it.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: int
        The highest order.
    jets: of the type of z
        Derivatives. probe: ``len=d``, ``elm_shape=T+W+(Ni,)``. apply and entries: ``shape=T+W``. Any axes
        after the order axis are carried along.

    Returns
    -------
    of the type of z
        Taylor coefficients, with the shapes of ``jets``. probe: a tuple, one array per mode.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds.

    See Also
    --------
    taylor_to_derivatives
    derivative_weight_to_taylor_weight
    prepare_derivative_data

    Examples
    --------
    The derivatives of ``exp(2 s)`` at ``s = 0`` are ``1, 2, 4, 8``; its Taylor coefficients are ``2**t / t!``:

    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> derivatives = np.array([[1.0], [2.0], [4.0], [8.0]])        # apply: T + W = (4, 1)
    >>> taylor = jets.derivatives_to_taylor('apply', 3, derivatives)
    >>> print(bool(np.allclose(taylor[:, 0], [1.0, 2.0, 2.0, 8.0 / 6.0])))
    True

    The probe kind: one array per mode in, a tuple out.

    >>> np.random.seed(0)
    >>> derivatives = [np.random.randn(4, 5, N) for N in (6, 7)]    # d = 2, T = (4,), W = (5,)
    >>> taylor = jets.derivatives_to_taylor('probe', 3, derivatives)
    >>> print(type(taylor).__name__, [t_i.shape for t_i in taylor])
    tuple [(4, 5, 6), (4, 5, 7)]
    >>> print(bool(np.allclose(taylor[1][3], derivatives[1][3] / 6.0)))
    True
    """
    _require_kind(kind)
    xnp, _, _ = get_backend(False, tree_contains_jax(jets))                 # once, over all the inputs
    return _scale_orders(kind, jets, order, True, xnp)


def taylor_to_derivatives(
        kind:   str,                       # 'probe' | 'apply' | 'entries': which type the jets are of
        order:  int,                       # highest order. T=(order+1,) if order else ()
        jets:   typ.Any,                   # Taylor coefficients: probe len=d, elm_shape=T+W+(Ni,); apply/entries T+W
) -> typ.Any:                              # derivatives, same shapes: slot t multiplied by t!. probe: a tuple
    """Taylor coefficients to derivatives: slot ``t`` of the order axis multiplied by ``t!``.

    The reverse of :py:func:`derivatives_to_taylor`, for a user who wants an output of the library (which is
    always in Taylor coefficients) as derivatives. ``kind`` says which type the jets are of, in the same way:
    for ``'probe'`` one array per mode, taken along the first axis (a tuple, a list, or a packed array), a
    tuple returned; for ``'apply'`` and ``'entries'`` one array, its first axis the order axis.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: int
        The highest order.
    jets: of the type of z
        Taylor coefficients. probe: ``len=d``, ``elm_shape=T+W+(Ni,)``. apply and entries: ``shape=T+W``.

    Returns
    -------
    of the type of z
        Derivatives, with the shapes of ``jets``. probe: a tuple, one array per mode.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds.

    See Also
    --------
    derivatives_to_taylor

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> taylor = np.ones((4, 1))                                    # 1 / (1 - s): every coefficient is 1
    >>> print(jets.taylor_to_derivatives('apply', 3, taylor)[:, 0].tolist())
    [1.0, 1.0, 2.0, 6.0]
    """
    _require_kind(kind)
    xnp, _, _ = get_backend(False, tree_contains_jax(jets))                 # once, over all the inputs
    return _scale_orders(kind, jets, order, False, xnp)


def derivative_weight_to_taylor_weight(
        weight:  NDArray,                  # a weight on derivative residuals: probe (d,)+T, apply/entries T
        order:   int,                      # highest order. T=(order+1,) if order else ()
) -> NDArray:                              # same shape: slot t multiplied by t!
    """The Taylor weight that gives the misfit of a weight on derivative residuals: slot ``t`` times ``t!``.

    A derivative residual is ``t!`` times the Taylor residual, so ``weight[t] * (derivative residual)`` equals
    ``(weight[t] * t!) * (Taylor residual)``. The weight goes the opposite way to the data, which are divided
    by ``t!``, and it is the weight itself that is multiplied, not its square. The order axis is the last axis
    of the weight for every kind. At ``order = 0`` nothing changes.

    Parameters
    ----------
    weight: NDArray
        A weight on derivative residuals. probe: ``shape=(d,)+T``. apply and entries: ``shape=T``.
    order: int
        The highest order.

    Returns
    -------
    NDArray
        The weight on Taylor coefficient residuals, same shape.

    See Also
    --------
    prepare_derivative_data

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> print(jets.derivative_weight_to_taylor_weight(np.array([1.0, 10.0, 10.0, 10.0]), 3).tolist())
    [1.0, 10.0, 20.0, 60.0]
    >>> probe_weight = np.ones((2, 4))                              # d = 2 modes
    >>> print(jets.derivative_weight_to_taylor_weight(probe_weight, 3).tolist())
    [[1.0, 1.0, 2.0, 6.0], [1.0, 1.0, 2.0, 6.0]]
    """
    xnp, _, _ = get_backend(False, tree_contains_jax(weight))
    return _taylor_weight(weight, order, xnp)


def prepare_derivative_data(
        kind:        str,                              # 'probe' | 'apply' | 'entries'
        order:       int,                              # highest order. T=(order+1,) if order else ()
        y:           typ.Any,                          # derivatives, of the type of z: probe len=d, elm_shape=T+W+(Ni,); apply/entries T+W
        weight:      typ.Optional[NDArray] = None,     # on derivative residuals: probe (d,)+T, apply/entries T; None = ones on the derivatives
        validation:  typ.Optional[typ.Any] = None,     # validation derivatives, of the type of z, or None
) -> typ.Tuple[
        typ.Any,                                       # y in Taylor coefficients, same type and shapes
        NDArray,                                       # the Taylor weight: probe (d,)+T, apply/entries T. t! per slot when weight is None
        typ.Optional[typ.Any],                         # validation in Taylor coefficients, or None
]:
    """Prepare derivative data for a fit: the data, the validation data and the weight, in one call.

    The fit works in Taylor coefficients and its default weight is ones on the Taylor residuals. Ones on the
    DERIVATIVE residuals are not that: they are the weight ``t!`` on the Taylor residuals. So this function
    always returns a weight, and with ``weight=None`` it returns ``t!`` per slot (for the probe kind, in every
    mode's row). Pass the returned weight to the fit; with it, the objective of the fit is the weighted sum of
    squares of the derivative residuals that you had in mind.

    The outputs of the fitted tensor stay in Taylor coefficients; :py:func:`taylor_to_derivatives` converts
    them back.

    For ``'probe'`` the data are one array per mode, taken along the first axis of ``y`` (a tuple, a list, or
    an array packed for the uniform layout), and a tuple is returned. The result is on jax if any array given
    is a jax array.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: int
        The highest order. At ``order = 0`` there is nothing to convert and the weight returned is ones.
    y: of the type of z
        The data, as derivatives. probe: ``len=d``, ``elm_shape=T+W+(Ni,)``. apply and entries: ``shape=T+W``.
    weight: NDArray or None
        A weight on the derivative residuals. probe: ``shape=(d,)+T``. apply and entries: ``shape=T``.
        ``None`` is ones on the derivatives, which is NOT the fit's own default: a fit given no weight uses
        ones on the Taylor coefficients (a default that may change in a later release), so always pass the
        weight this function returns.
    validation: of the type of z, or None
        Validation data, as derivatives.

    Returns
    -------
    y_taylor: of the type of z
        The data in Taylor coefficients.
    weight_taylor: NDArray
        The weight on Taylor coefficient residuals, in the shape of a weight of ``kind``.
    validation_taylor: of the type of z, or None
        The validation data in Taylor coefficients, or ``None`` if none were given.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds.

    See Also
    --------
    derivatives_to_taylor
    derivative_weight_to_taylor_weight

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> y_derivatives = np.array([[1.0], [1.0], [2.0], [6.0]])      # apply: T + W = (4, 1)
    >>> y, weight, validation = jets.prepare_derivative_data('apply', 3, y_derivatives)
    >>> print(y[:, 0].tolist(), weight.tolist(), validation)
    [1.0, 1.0, 1.0, 1.0] [1.0, 1.0, 2.0, 6.0] None

    The module's docstring walks through the cases: a weight chosen for derivatives, no weight, validation
    data, and the probe kind.
    """
    _require_kind(kind)
    xnp, _, _ = get_backend(False, tree_contains_jax((y, weight, validation)))     # once, over all the inputs
    y_taylor = _scale_orders(kind, y, order, True, xnp)
    if weight is None:
        first_TX = y_taylor[0] if kind == 'probe' else y_taylor
        T = (order + 1,) if order else ()
        weight = xnp.ones(((len(y_taylor),) if kind == 'probe' else ()) + T,
                          dtype=_lowest_precision_dtype(xnp, first_TX))
    weight_taylor = _taylor_weight(weight, order, xnp)
    validation_taylor = None if validation is None else _scale_orders(kind, validation, order, True, xnp)
    return y_taylor, weight_taylor, validation_taylor


#############################################
########    The input jets    ###############
#############################################

def _affine_jet_of_one_array(
        w:          NDArray,                    # the point: shape=X
        directions: typ.Sequence[NDArray],      # m of them, each shape=X
        axis:       int,                        # where the m jet axes go
        xnp,
) -> NDArray:                                   # shape=X[:axis]+(2,)*m+X[axis:]
    m = len(directions)
    dtype = xnp.result_type(w, *directions)
    zero = xnp.zeros(w.shape, dtype=dtype)      # zeros made, not 0 * w: a slot of no term is exactly zero
    slots = []
    for k in itertools.product(range(2), repeat=m):                 # the multi-indices of the box, row-major
        slots.append(w if sum(k) == 0 else (directions[k.index(1)] if sum(k) == 1 else zero))
    stacked = xnp.stack([xnp.asarray(slot, dtype=dtype) for slot in slots], axis=axis)
    return xnp.reshape(stacked, w.shape[:axis] + (2,) * m + w.shape[axis:])


def affine_jet(
        ww:             typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+(Ni,); or packed (d,)+W+(N,)
        *directions:    typ.Union[typ.Sequence[NDArray], NDArray],  # m of them, each of the form and shapes of ww
) -> typ.Union[typ.Tuple[NDArray, ...], NDArray]:   # len=d, elm_shape=(2,)*m+W+(Ni,); or packed (d,)+(2,)*m+W+(N,)
    """The input jet of the point ``ww`` moved along ``m`` directions: ``w + s_1 p_1 + ... + s_m p_m``.

    A sampling function with a jet shape ``order`` takes input jets: per mode an array whose ``m = len(order)``
    leading jet axes hold the Taylor coefficients of the input vector as a polynomial in ``s_1 ... s_m``. The input
    of degree at most one in each variable, ``w + s_1 p_1 + ... + s_m p_m``, has two slots per jet axis: slot
    ``(0, ..., 0)`` is ``w``, slot ``e_j`` (one in axis ``j``, zero elsewhere) is ``p_j``, and every other slot is
    zero (with two directions, slot ``(1, 1)``, the coefficient of ``s_1 s_2``). The line ``w + s p`` is
    ``affine_jet(ww, pp)``, of jet sizes ``(2,)``, used with a jet shape ``(n,)``. An input jet of another kind (a
    curve, with more slots) is built directly; nothing else is needed.

    It returns the form it is given: for a tuple or a list, a tuple, one array per mode; for vectors packed for the
    uniform layout, one array ``(d,) + W + (N,)``, the packed array ``(d,) + (2,) * m + W + (N,)``. With no
    direction it returns ``ww`` as it is (``m = 0``: no jets). The dtype is the one the point and the directions
    promote to. It checks only what it needs to stack: each direction has the form and the shapes of ``ww``.

    Parameters
    ----------
    ww: typ.Union[typ.Sequence[NDArray], NDArray]
        The point. len=d, elm_shape=W+(Ni,); or packed, shape=(d,)+W+(N,)
    *directions: typ.Union[typ.Sequence[NDArray], NDArray]
        The directions ``p_1 ... p_m``, each of the form and the shapes of ``ww``.

    Returns
    -------
    typ.Union[typ.Tuple[NDArray, ...], NDArray]
        The input jet. len=d, elm_shape=(2,)*m+W+(Ni,); or packed, shape=(d,)+(2,)*m+W+(N,)

    Raises
    ------
    ValueError
        If a direction does not have the form of ``ww`` (a sequence of as many arrays, or one array) or the shape
        of ``ww`` in some mode.

    See Also
    --------
    index_to_one_hot
    t3toolbox.backend.probing.dense_probe

    Examples
    --------
    A line in one variable: two slots, the point and the direction.

    >>> import numpy as np
    >>> from t3toolbox.backend.jets import affine_jet
    >>> np.random.seed(0)
    >>> ww = [np.random.randn(5, N) for N in (3, 4)]          # d = 2, W = (5,)
    >>> pp = [np.random.randn(5, N) for N in (3, 4)]
    >>> line = affine_jet(ww, pp)
    >>> print([x.shape for x in line])                        # (2,) + W + (Ni,)
    [(2, 5, 3), (2, 5, 4)]
    >>> print(bool(np.array_equal(line[1][0], ww[1])), bool(np.array_equal(line[1][1], pp[1])))
    True True

    Two directions: jet sizes ``(2, 2)``, and the slot of ``s_1 s_2`` is zero.

    >>> qq = [np.random.randn(5, N) for N in (3, 4)]
    >>> plane = affine_jet(ww, pp, qq)
    >>> print(plane[0].shape)
    (2, 2, 5, 3)
    >>> print(bool(np.array_equal(plane[0][0, 1], qq[0])), bool(np.all(plane[0][1, 1] == 0)))
    True True

    Packed vectors give a packed jet, the jet axes after the mode axis:

    >>> packed = np.stack([np.random.randn(5, 3) for _ in range(2)])     # (d,) + W + (N,)
    >>> print(affine_jet(packed, packed).shape)
    (2, 2, 5, 3)
    """
    m = len(directions)
    if m == 0:
        return ww
    xnp, _, _ = get_backend(True, tree_contains_jax((ww, directions)))
    if is_ndarray(ww):
        for j, p in enumerate(directions):
            if not is_ndarray(p) or tuple(p.shape) != tuple(ww.shape):
                raise ValueError('affine_jet: direction %d must be packed like ww, an array of shape %s; got %s'
                                 % (j, tuple(ww.shape), tuple(p.shape) if is_ndarray(p) else type(p).__name__))
        return _affine_jet_of_one_array(ww, directions, 1, xnp)
    d = len(ww)
    for j, p in enumerate(directions):
        if is_ndarray(p) or len(p) != d:
            raise ValueError('affine_jet: direction %d must be a sequence of %d arrays, as ww is' % (j, d))
        for i, (w_i, p_i) in enumerate(zip(ww, p)):
            if tuple(p_i.shape) != tuple(w_i.shape):
                raise ValueError('affine_jet: direction %d has the shape %s in mode %d, where ww has %s'
                                 % (j, tuple(p_i.shape), i, tuple(w_i.shape)))
    return tuple(_affine_jet_of_one_array(ww[i], [p[i] for p in directions], 0, xnp) for i in range(d))


def index_to_one_hot(
        index:  NDArray,                # int, shape=(d,)+W
        shape:  typ.Sequence[int],      # the mode sizes N_i, len=d
) -> typ.Tuple[NDArray, ...]:           # len=d, elm_shape=W+(Ni,): e_index, in the default float dtype
    """The one-hot vectors of a grid point: ``e_index[i]`` in each mode ``i``.

    ``entries`` at ``index`` is ``apply`` at these vectors. The entries kind has no jets; the jets of a tensor
    along a curve of input vectors that starts at the grid point are the apply kind's at an input jet whose slot
    ``0`` is these vectors, for example ``affine_jet(index_to_one_hot(index, shape), pp)`` for the line
    ``e_index + s pp``. It is also the sample of a tensor with some discrete modes beside continuous ones.

    The vectors are numpy for a numpy index and jax for a jax index (a traced one included), in the library's
    default floating dtype (float64 on numpy; on jax float32 unless x64 is on); there is no ``dtype`` argument, and
    the sampling functions lower it to the cores' precision. Index semantics are numpy's indexing, as in
    :py:func:`~t3toolbox.backend.entries.dense_entries`: an index in ``[-N_i, 0)`` wraps, and a numpy index
    outside ``[-N_i, N_i)`` raises an ``IndexError``. A jax index (a traced one included) cannot be checked, and an
    out-of-range one gives a zero vector.

    Parameters
    ----------
    index: NDArray
        The grid points, one row per mode. dtype=int, shape=(d,)+W
    shape: typ.Sequence[int]
        The mode sizes ``N_i``. len=d

    Returns
    -------
    typ.Tuple[NDArray, ...]
        The one-hot vectors. len=d, elm_shape=W+(Ni,)

    Raises
    ------
    ValueError
        If ``index`` is not of an integer dtype, or has not one row per mode.
    IndexError
        If ``index`` is a numpy array and an index of mode ``i`` is outside ``[-N_i, N_i)``.

    See Also
    --------
    affine_jet
    t3toolbox.backend.entries.dense_entries

    Examples
    --------
    >>> import numpy as np
    >>> from t3toolbox.backend.jets import index_to_one_hot
    >>> index = np.array([[0, 2], [1, -1]])                   # (d,) + W: d = 2, W = (2,)
    >>> e = index_to_one_hot(index, (3, 4))
    >>> print([x.shape for x in e], e[0].dtype)
    [(2, 3), (2, 4)] float64
    >>> print(e[0].tolist(), e[1].tolist())
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

    An index outside the grid has no one-hot vector (structural gotcha):

    >>> index_to_one_hot(np.array([[0, 3], [1, 0]]), (3, 4))      # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    IndexError: index_to_one_hot: mode 0 has the index or indices [3], outside [-3, 3)
    """
    use_jax = tree_contains_jax(index)
    xnp, _, _ = get_backend(True, use_jax)
    if not is_ndarray(index):
        index = np.asarray(index)
    if not np.issubdtype(index.dtype, np.integer):
        raise ValueError('index_to_one_hot: the index is of an integer dtype; got %s' % (index.dtype,))
    if index.ndim < 1 or index.shape[0] != len(shape):
        raise ValueError('index_to_one_hot: the index has one row per mode, %d; got the shape %s'
                         % (len(shape), tuple(index.shape)))
    if not use_jax:                             # a concrete numpy index: checked as numpy's indexing checks it
        for i, N in enumerate(shape):
            outside = (index[i] < -N) | (index[i] >= N)
            if outside.any():
                raise IndexError('index_to_one_hot: mode %d has the index or indices %s, outside [%d, %d)'
                                 % (i, sorted(set(index[i][outside].tolist()))[:5], -N, N))
    default_float = xnp.zeros(()).dtype
    vectors = []
    for i, N in enumerate(shape):
        index_i = xnp.where(index[i] < 0, index[i] + N, index[i])
        vectors.append((index_i[..., None] == xnp.arange(N)).astype(default_float))
    return tuple(vectors)


######################################################################
########    The checkers of a sample, its data and its weight    #####
######################################################################

def check_perturbation_vectors(
        ww: typ.Sequence[NDArray],  # points X,       len=d, elm_shape=W+(Ni,)
        pp: typ.Sequence[NDArray],  # perturbation P, len=d, elm_shape=W+(Ni,)
) -> None:
    '''Structural check (hard error): the perturbation ``P`` (``pp``) shares the probe stack ``W`` and
    mode dims of the points ``X`` (``ww``) -- each sample pairs a point with a direction. Used by the
    derivative-probe/apply frontends; shapes are static, so this is jit-safe.'''
    for i, (w, p) in enumerate(zip(ww, pp)):
        if np.shape(w) != np.shape(p):
            raise ValueError(
                "perturbation P must match the shape (probe stack W and mode dim) of the points X; "
                "mode %d: P %s vs X %s" % (i, np.shape(p), np.shape(w)))


def check_perturbation_index(
        index: NDArray,                          # grid points, int, shape=(d,)+W
        pp:    typ.Sequence[NDArray],            # perturbation P, len=d, elm_shape=W+(Ni,)
        shape: typ.Optional[typ.Sequence[int]]   # ambient mode dims (Ni), len=d -- if given, also check P's mode dim
             = None,
) -> None:
    '''Structural check (hard error): the perturbation ``P`` shares the probe stack ``W`` of the grid
    points ``index`` (shape ``(d,)+W``), and -- when ``shape`` is given (the ambient mode dims ``Ni``,
    which the integer ``index`` does not carry) -- ``P``'s mode dim matches it. Used by the
    derivative-entries frontends; jit-safe.'''
    iW = tuple(np.shape(index)[1:])
    for i, p in enumerate(pp):
        if tuple(np.shape(p)[:-1]) != iW:
            raise ValueError(
                "perturbation P's probe stack W must match index's; mode %d: P %s vs index %s"
                % (i, tuple(np.shape(p)[:-1]), iW))
        if shape is not None and np.shape(p)[-1] != shape[i]:
            raise ValueError(
                "perturbation P's mode dim must match the ambient dim Ni; mode %d: P has %d vs Ni=%d"
                % (i, np.shape(p)[-1], shape[i]))


def _require_order(order: int) -> None:
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError('order must be an integer >= 0; got %r' % (order,))


def _require_vectors(
        vectors:  typ.Any,                   # expected: len=d, elm_shape=W+(Ni,)
        shape:    typ.Sequence[int],         # the mode sizes N_i, len=d
        name:     str,                       # 'ww' or 'pp', for the message
) -> typ.Tuple[int, ...]:                    # W, read from the first vector
    '''A tuple or a list of ``d`` arrays, the ``i``-th of shape ``W + (N_i,)``, with one ``W`` for all.'''
    d = len(shape)
    if not isinstance(vectors, (tuple, list)):
        raise ValueError('%s must be a tuple or a list of %d arrays, one per mode, the i-th of shape W+(Ni,); '
                         'got a %s' % (name, d, type(vectors).__name__))
    if len(vectors) != d:
        raise ValueError('%s must have one array per mode: expected %d, got %d' % (name, d, len(vectors)))
    for i, vector in enumerate(vectors):
        if isinstance(vector, (tuple, list)) or len(np.shape(vector)) < 1:
            raise ValueError('%s[%d] must be one array of shape W+(N_%d,) = W+(%d,); got a %s'
                             % (name, i, i, shape[i], type(vector).__name__))
    W = tuple(np.shape(vectors[0])[:-1])
    for i, vector in enumerate(vectors):
        if tuple(np.shape(vector)) != W + (shape[i],):
            raise ValueError('%s[%d] has shape %s; expected W+(N_%d,) = %s (W is read from %s[0])'
                             % (name, i, tuple(np.shape(vector)), i, W + (shape[i],), name))
    return W


def _sample_parts(
        kind:    str,
        order:   int,
        sample:  typ.Any,
) -> typ.Tuple[
        typ.Any,                             # ww or index
        typ.Optional[typ.Any],               # pp, or None at order 0
]:
    '''Split a sample by the order: ``ww`` or ``index`` at order 0, the pair with ``pp`` above.'''
    first_name = 'index' if kind == 'entries' else 'ww'
    if order == 0:
        return sample, None
    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        raise ValueError('at order %d the sample of %r must be the pair (%s, pp); got %s'
                         % (order, kind, first_name,
                            'a %s of length %d' % (type(sample).__name__, len(sample))
                            if isinstance(sample, (tuple, list)) else 'a %s' % type(sample).__name__))
    return sample[0], sample[1]


def _sample_stack(
        kind:   str,
        first:  typ.Any,                     # ww or index, already checked
) -> typ.Tuple[int, ...]:                    # W
    return tuple(np.shape(first)[1:]) if kind == 'entries' else tuple(np.shape(first[0])[:-1])


def check_sample(
        kind:    str,                      # 'probe' | 'apply' | 'entries'
        order:   int,                      # highest Taylor order
        sample:  typ.Any,                  # ww | index | (ww, pp) | (index, pp), as order says
        shape:   typ.Sequence[int],        # the mode sizes N_i, len=d
) -> None:                                 # the form for the order; d vectors of W+(Ni,), or index (d,)+W; pp as ww
    """Check the form and the shapes of a sample against the mode sizes; raise a ``ValueError`` if wrong.

    What a good sample is:

    - at ``order = 0`` it is the inputs alone: ``ww`` for ``'probe'`` and ``'apply'``, ``index`` for
      ``'entries'``. At ``order >= 1`` it is the pair ``(ww, pp)`` or ``(index, pp)``. The order decides, never
      the type;
    - ``ww`` is a tuple or a list of ``d`` arrays, the ``i``-th of shape ``W + (N_i,)``, with the same sample
      stack ``W`` for every mode;
    - ``index`` is one integer array of shape ``(d,) + W``;
    - ``pp`` is like ``ww``, with the same ``W`` (through :py:func:`check_perturbation_vectors` and
      :py:func:`check_perturbation_index`).

    This is the ragged form of a sample. Vectors packed for the uniform layout, one array ``(d,) + W + (N,)``,
    are not accepted: check before packing. The values are not looked at (an index out of range is not caught).
    Shapes are static, so the check also runs under ``jax.jit``.

    This is a checker, not a guarantee: no backend function calls it.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: int
        The highest Taylor order.
    sample
        ``ww`` or ``index`` at order 0; ``(ww, pp)`` or ``(index, pp)`` at order 1 and above.
    shape: Sequence[int]
        The mode sizes ``(N_0, ..., N_(d-1))`` of the tensor the sample is for.

    Raises
    ------
    ValueError
        Naming what is wrong.

    See Also
    --------
    check_data
    check_weight

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> np.random.seed(0)
    >>> shape = (4, 5, 6)
    >>> ww = [np.random.randn(7, N) for N in shape]             # W = (7,)
    >>> pp = [np.random.randn(7, N) for N in shape]
    >>> jets.check_sample('probe', 0, ww, shape)                # good: returns nothing
    >>> jets.check_sample('apply', 2, (ww, pp), shape)
    >>> jets.check_sample('entries', 0, np.zeros((3, 7), dtype=int), shape)

    A sample of the wrong form for the order, and a vector of the wrong mode size:

    >>> jets.check_sample('apply', 2, ww, shape)                # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: at order 2 the sample of 'apply' must be the pair (ww, pp); got a list of length 3
    >>> jets.check_sample('probe', 0, ww[:2] + [np.zeros((7, 9))], shape)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: ww[2] has shape (7, 9); expected W+(N_2,) = (7, 6) (W is read from ww[0])
    """
    _require_kind(kind)
    _require_order(order)
    d = len(shape)
    first, pp = _sample_parts(kind, order, sample)
    first_name = 'index' if kind == 'entries' else 'ww'
    if order >= 1 and pp is None:
        raise ValueError('at order %d the sample of %r needs the directions pp: it is the pair (%s, pp); got '
                         '(%s, None)' % (order, kind, first_name, first_name))
    try:
        if kind == 'entries':
            if isinstance(first, (tuple, list)):
                raise ValueError('index must be one integer array of shape (d,)+W with d=%d; got a %s'
                                 % (d, type(first).__name__))
            index_shape = tuple(np.shape(first))
            if len(index_shape) < 1 or index_shape[0] != d:
                raise ValueError('index must have shape (d,)+W with d=%d; got shape %s' % (d, index_shape))
            if not np.issubdtype(np.dtype(first.dtype), np.integer):
                raise ValueError('index must be an integer array; got dtype %s' % (np.dtype(first.dtype),))
            W = index_shape[1:]
        else:
            W = _require_vectors(first, shape, 'ww')
    except ValueError as error:
        # name the likely cause where the inputs look like the sample of the other form
        if order == 0 and isinstance(sample, (tuple, list)) and len(sample) == 2 and isinstance(sample[1], (tuple, list)):
            raise ValueError('%s. At order 0 the sample is the inputs %s alone, with no pp; this sample looks '
                             'like the pair (%s, pp)' % (error, first_name, first_name)) from None
        if order >= 1 and kind != 'entries' and not isinstance(first, (tuple, list)):
            raise ValueError('%s. At order %d the sample is the pair (ww, pp); this sample looks like ww alone'
                             % (error, order)) from None
        raise
    if pp is not None:
        if _require_vectors(pp, shape, 'pp') != W:
            raise ValueError("pp's sample stack W must be that of the inputs: pp[0] has shape %s, the inputs "
                             "have W = %s" % (tuple(np.shape(pp[0])), W))
        if kind == 'entries':
            check_perturbation_index(first, pp, shape)
        else:
            check_perturbation_vectors(first, pp)


def check_data(
        kind:    str,                      # 'probe' | 'apply' | 'entries'
        order:   int,                      # highest Taylor order. T=(order+1,) if order else ()
        sample:  typ.Any,                  # ww | index | (ww, pp) | (index, pp), as order says; a good sample
        y:       typ.Any,                  # the data: probe len=d, elm_shape=T+W+(Ni,); apply/entries T+W
) -> None:
    """Check the form and the shapes of the data ``y`` against a sample; raise a ``ValueError`` if wrong.

    The data are of the type of ``z``: for ``'probe'`` a tuple or a list of ``d`` arrays, the ``i``-th of shape
    ``T + W + (N_i,)``; for ``'apply'`` and ``'entries'`` one array of shape ``T + W``. ``T`` is ``(order + 1,)``,
    or ``()`` at order 0. ``W``, ``d`` and the mode sizes are read from the sample, which is taken to be good:
    call :py:func:`check_sample` on it first. The shapes are exact: the data carry no stack of base points.
    This is the ragged type: probe data packed for the uniform layout into one array are refused.

    This is a checker, not a guarantee: no backend function calls it.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: int
        The highest Taylor order.
    sample
        A good sample for ``kind`` and ``order`` (see :py:func:`check_sample`).
    y: of the type of z
        The data.

    Raises
    ------
    ValueError
        Naming what is wrong.

    See Also
    --------
    check_sample
    check_weight

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> np.random.seed(0)
    >>> ww = [np.random.randn(7, N) for N in (4, 5, 6)]         # W = (7,)
    >>> pp = [np.random.randn(7, N) for N in (4, 5, 6)]
    >>> jets.check_data('apply', 2, (ww, pp), np.zeros((3, 7)))                         # T + W: good
    >>> jets.check_data('probe', 2, (ww, pp), [np.zeros((3, 7, N)) for N in (4, 5, 6)])

    Data without the order axis, at order 2:

    >>> jets.check_data('apply', 2, (ww, pp), np.zeros(7))      # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: the data y has shape (7,); expected T+W = (3, 7) (T = (order+1,) = (3,), W = (7,))
    """
    _require_kind(kind)
    _require_order(order)
    first, _ = _sample_parts(kind, order, sample)
    W = _sample_stack(kind, first)
    T = (order + 1,) if order else ()
    describe = '(T = %s, W = %s)' % ('(order+1,) = %s' % (T,) if order else '()', W)
    if kind == 'probe':
        d = len(first)
        if not isinstance(y, (tuple, list)):
            raise ValueError('the data y of the probe kind must be a tuple or a list of %d arrays, one per mode, '
                             'the i-th of shape T+W+(Ni,); got a %s' % (d, type(y).__name__))
        if len(y) != d:
            raise ValueError('the data y of the probe kind must have one array per mode: expected %d, got %d'
                             % (d, len(y)))
        for i, y_TWo in enumerate(y):
            expected = T + W + (np.shape(first[i])[-1],)
            if tuple(np.shape(y_TWo)) != expected:
                raise ValueError('the data y[%d] has shape %s; expected T+W+(N_%d,) = %s %s'
                                 % (i, tuple(np.shape(y_TWo)), i, expected, describe))
    elif isinstance(y, (tuple, list)) or tuple(np.shape(y)) != T + W:
        got = ('a %s of length %d' % (type(y).__name__, len(y)) if isinstance(y, (tuple, list))
               else 'shape %s' % (tuple(np.shape(y)),))
        raise ValueError('the data y has %s; expected T+W = %s %s' % (got, T + W, describe))


def check_weight(
        kind:    str,                      # 'probe' | 'apply' | 'entries'
        order:   int,                      # highest Taylor order. T=(order+1,) if order else ()
        weight:  typ.Optional[NDArray],    # None, or exactly (d,)+T (probe) or T (apply, entries)
        d:       int,                      # the number of modes
) -> None:
    """Check the shape of a weight; raise a ``ValueError`` if it is not exactly the shape for the kind.

    The weight of a fit has the shape of the mode and order axes of the data and no other: ``(d,) + T`` for
    ``'probe'`` and ``T`` for ``'apply'`` and ``'entries'``, with ``T = (order + 1,)``, or ``()`` at order 0.
    ``None`` is good and means ones, the library's default weight (a default that may change in a later
    release). There is no rule for a bare vector, and a dimension of 1 is not broadcast: a weight per order for
    the probe kind is written out per mode, ``np.tile(per_order, (d, 1))``, and a weight per mode is written
    out per order, ``np.tile(per_mode[:, None], (1, order + 1))``.

    A shape check cannot see what the axes mean: when ``d == order + 1`` a probe weight given transposed, with
    the order axis first, has the right shape and passes. The mode axis is the first.

    This is a checker, not a guarantee: no backend function calls it.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: int
        The highest Taylor order.
    weight: NDArray or None
        The weight.
    d: int
        The number of modes.

    Raises
    ------
    ValueError
        Naming the shape found and the shape expected.

    See Also
    --------
    check_sample
    check_data

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> jets.check_weight('probe', 2, np.ones((3, 3)), d=3)     # (d,) + T: good
    >>> jets.check_weight('apply', 2, np.ones(3), d=3)          # T: good
    >>> jets.check_weight('probe', 0, np.ones(3), d=3)          # (d,) at order 0: good
    >>> jets.check_weight('apply', 2, None, d=3)                # None is ones

    A weight per order, given as a bare vector to the probe kind, is not accepted. Write it out per mode:

    >>> per_order = np.array([1.0, 0.5, 0.25])
    >>> jets.check_weight('probe', 2, per_order, d=4)           # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: the weight of the probe kind has shape (3,); expected exactly (d,)+T = (4, 3) (no bare vector, and no dimension of 1 to broadcast: write the weight out in full)
    >>> jets.check_weight('probe', 2, np.tile(per_order, (4, 1)), d=4)
    """
    _require_kind(kind)
    _require_order(order)
    if weight is None:
        return
    T = (order + 1,) if order else ()
    expected = ((d,) if kind == 'probe' else ()) + T
    if tuple(np.shape(weight)) != expected:
        raise ValueError('the weight of the %s kind has shape %s; expected exactly %s = %s (no bare vector, and '
                         'no dimension of 1 to broadcast: write the weight out in full)'
                         % (kind, tuple(np.shape(weight)), '(d,)+T' if kind == 'probe' else 'T', expected))
