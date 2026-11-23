# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Jets of the sampling outputs: the dual jet, derivative data, the input jets, and the checkers of a sample, its data
and its weight.

A **jet** here is what a sampling function returns with a jet shape ``order = (n_1, ..., n_m)``: the Taylor
coefficients of its output as a polynomial in the jet variables ``s_1 ... s_m``, on the box ``F = (n_1 + 1, ...,
n_m + 1)``, the coefficient of ``s_1^k_1 ... s_m^k_m`` at the multi-index ``k`` of ``m`` leading jet axes. With no
jets, ``order = ()``, there are no jet axes, ``F = ()``. The jet shape is a tuple, always: an integer is not one and
raises. An object "of the type of ``z``" is one output of a sampling kind: for ``'probe'`` a ``len=d`` tuple or list,
one array of shape ``F + W + C + (Ni,)`` per mode; for ``'apply'`` one array of shape ``F + W + C``; for
``'entries'``, which has no jets, one array of shape ``W + C`` (``C`` the model's stack of base points, ``()`` for an
unstacked model). The data ``y`` of a fit, the residual ``r = z - y`` and the weight follow that type.

**``kind`` says which type an object is of**, in every function that works on one; the object itself is never
looked at to decide it. For ``'probe'`` the functions that compute (:py:func:`jet_to_dual_jet`,
:py:func:`dual_jet_to_jet`, :py:func:`derivatives_to_taylor`, :py:func:`taylor_to_derivatives`,
:py:func:`prepare_derivative_data`) go over the modes along the first axis of the object and return a tuple, one
array per mode. So a tuple, a list, and an array packed for the uniform layout, ``(d,) + F + W + ... + (N,)``,
are all taken mode by mode, and the packed array gives what ``tuple(packed)`` gives (``xnp.stack`` packs the
result again). For ``'apply'`` and ``'entries'`` the object is one array whose leading axes are the jet axes. An
object of the other type than ``kind`` says is not guarded against: a tuple given with ``kind='apply'`` is made
into one array, which fails where the modes differ in size and otherwise reads the mode axis as a jet axis. The
entries kind takes ``order = ()`` only (its sample, an integer index, cannot carry jets): any other jet shape
raises, and with ``()`` the functions of this module return its data as they are.

Five groups of functions:

- **The data metric.** :py:func:`data_metric` is ``M: Y* -> Y``, ``gamma * omega**2 * r`` elementwise: what a
  transpose is applied to for the gradient of a weighted fit. Its weight is of the type of ``z`` (each axis 1 or
  ``z``'s size); :py:func:`blocks_to_weight` makes one from a value per block.
- **The dual jet,** the engine's form of an element of ``Y``. :py:func:`jet_to_dual_jet` makes the engine's seed
  from a residual: the data metric at ``gamma = 1`` (``weight**2``, entry by entry), then the turn, a flip of every
  jet axis.
  :py:func:`dual_jet_to_jet` is its inverse. They are for the engine's ``w_tilde`` and its held tilde variables,
  never for a transpose.
- **Derivative data.** The library works in Taylor coefficients, everywhere. A user whose data are derivatives
  converts them before fitting: :py:func:`derivatives_to_taylor`, :py:func:`taylor_to_derivatives`,
  :py:func:`derivative_weight_to_taylor_weight`, and :py:func:`prepare_derivative_data`, which does the data, the
  validation data and the weight in one call. The mixed derivative ``d^k / ds^k`` of ``s^k`` is
  ``k_1! ... k_m!``, the factor of slot ``k``. No fitting function, oracle or sampling method has a keyword for
  derivative data.
- **The input jets.** A sampling function with a jet shape takes input jets, the Taylor coefficients of its input
  vectors as polynomials in the jet variables. :py:func:`affine_jet` makes the input jet of a point moved along
  directions, ``w + s_1 p_1 + ... + s_m p_m`` (the line ``w + s p`` with one), and :py:func:`index_to_one_hot` the
  one-hot vectors of a grid point, the apply kind's sample of an entry.
- **Checkers.** :py:func:`check_sample`, :py:func:`check_data` and :py:func:`check_weight` raise a ``ValueError``
  that names what is wrong with the form or the shape of a sample, its data or its weight. They are checkers, not
  guarantees: no backend function calls them. A backend function given a wrong shape computes a wrong answer or
  fails deep inside; call the checker first if you want the error. Shapes are static, so every checker also runs
  under ``jax.jit``.

**The weight** ``omega`` of a fit, like the data ``y``, is of the type of ``z``: each array of the rank of ``z``'s
array, each axis of size 1 or ``z``'s size (``F``, ``W``, the stack ``C``, the mode edge), never the stack of
directions ``K``. A size of 1 is the same value along that axis: a weight per slot has size 1 off the jet axes, a
weight per sample size 1 on them. :py:func:`blocks_to_weight` makes one from a value per block (per mode and slot,
per slot, or one number), the form older code used. The misfit is ``1/2 sum omega**2 r**2`` over every entry, on Taylor
coefficient residuals (:py:func:`data_metric`). **With no weight given (``None``) the weight is ones**: weights of 1.0
in the natural coordinates of the data, the plain sum of squares of the Taylor coefficient residuals.

**The data coordinates are Taylor coefficients, everywhere.** Data measured as derivatives are converted once, at the
boundary, data and weight together (:py:func:`prepare_derivative_data`); a weight computed from the data is computed
after the data are converted (the recipe is in :py:func:`prepare_derivative_data`).

Examples
--------
Derivative data. The example uses the apply kind with one jet variable, whose data have the shape ``F + W``.

>>> import numpy as np
>>> import t3toolbox.backend.jets as jets
>>> order = (3,)

The derivatives of ``1 / (1 - s)`` at ``s = 0`` are ``0!, 1!, 2!, 3!``. Its Taylor coefficients are all 1.

>>> y_derivatives = np.array([[1.0], [1.0], [2.0], [6.0]])              # shape F + W, with W = (1,)
>>> print(jets.derivatives_to_taylor('apply', order, y_derivatives)[:, 0].tolist())
[1.0, 1.0, 1.0, 1.0]
>>> y_back = jets.taylor_to_derivatives('apply', order, jets.derivatives_to_taylor('apply', order, y_derivatives))
>>> print(y_back[:, 0].tolist())
[1.0, 1.0, 2.0, 6.0]

Case 1. Taylor data. Nothing to do: the fit uses the data and the weight as given, and no weight means ones.

Case 2. Derivative data and a weight chosen for derivatives, here one per slot (``F + (1,)``, the same for every
sample). One call prepares both:

>>> weight_derivatives = np.array([1.0, 10.0, 10.0, 10.0]).reshape(4, 1)
>>> y, weight, _ = jets.prepare_derivative_data('apply', order, y_derivatives, weight_derivatives)
>>> print(y[:, 0].tolist())
[1.0, 1.0, 1.0, 1.0]
>>> print(weight[:, 0].tolist())
[1.0, 10.0, 20.0, 60.0]

The data are divided by ``t!`` and the weight is multiplied by it: a weight goes the opposite way.

Case 3. Derivative data and NO weight. The trap: ones on the DERIVATIVES are not ones on the Taylor
coefficients. The same call returns the weight ``t!``, which you then pass to the fit:

>>> y, weight, _ = jets.prepare_derivative_data('apply', order, y_derivatives)
>>> print(weight.shape, weight[:, 0].tolist())
(4, 1) [1.0, 1.0, 2.0, 6.0]

Validation data given to the same call are converted like the data:

>>> _, _, validation = jets.prepare_derivative_data('apply', order, y_derivatives, validation=2.0 * y_derivatives)
>>> print(validation[:, 0].tolist())
[2.0, 2.0, 2.0, 2.0]

In every case the objective is the one you wrote down. For a model whose Taylor coefficients are ``z``:

>>> z = np.array([[1.0], [0.9], [1.2], [0.8]])
>>> objective = lambda weight, prediction, data: 0.5 * float(np.sum((weight * (prediction - data)) ** 2))
>>> in_derivatives = objective(weight_derivatives, jets.taylor_to_derivatives('apply', order, z), y_derivatives)
>>> y, weight, _ = jets.prepare_derivative_data('apply', order, y_derivatives, weight_derivatives)
>>> print(bool(np.isclose(in_derivatives, objective(weight, z, y))))
True

Derivative data passed to a fit unconverted are read as Taylor coefficients, which is a different problem: the
order-3 target would be 6 where it should be 1. The library cannot detect this.

In two jet variables, at the jet shape ``(1, 2)``, slot ``(k_1, k_2)`` is divided by ``k_1! k_2!``: the mixed
derivative of ``s_1^k_1 s_2^k_2``. With no weight, every slot gets that factor:

>>> y_derivatives = np.ones((2, 3, 5))                                  # apply: F + W = (2, 3) + (5,)
>>> y, weight, _ = jets.prepare_derivative_data('apply', (1, 2), y_derivatives)
>>> print(y[:, :, 0].tolist())
[[1.0, 1.0, 0.5], [1.0, 1.0, 0.5]]
>>> print(weight[..., 0].tolist())
[[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]]

The probe kind has one array per mode, ``F + W + (Ni,)``. With no weight the weight returned is one mode-leading
array, ``(d,) + F + (1,) + (1,)``, the factorials in every mode:

>>> np.random.seed(0)
>>> y_derivatives = [np.random.randn(4, 5, N) for N in (6, 7)]          # d = 2 modes, F = (4,), W = (5,)
>>> y, weight, _ = jets.prepare_derivative_data('probe', (3,), y_derivatives)
>>> print([y_i.shape for y_i in y], weight.shape)
[(4, 5, 6), (4, 5, 7)] (2, 4, 1, 1)
>>> print(weight[:, :, 0, 0].tolist())
[[1.0, 1.0, 2.0, 6.0], [1.0, 1.0, 2.0, 6.0]]
>>> print(bool(np.allclose(y[1][3], y_derivatives[1][3] / 6.0)))        # mode 1, slot 3: divided by 3!
True

A weight per mode and slot, chosen for derivatives, is made from its blocks and converted slot by slot:

>>> blocks = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
>>> weight_derivatives = jets.blocks_to_weight('probe', (3,), blocks, y_derivatives)
>>> _, weight, _ = jets.prepare_derivative_data('probe', (3,), y_derivatives, weight_derivatives)
>>> print(weight[:, :, 0, 0].tolist())
[[1.0, 1.0, 2.0, 6.0], [2.0, 2.0, 4.0, 12.0]]
"""
import itertools
import math
import typing as typ

import numpy as np

import t3toolbox.backend.contractions as contractions
from t3toolbox.backend.common import *

__all__ = [
    # The data metric
    'data_metric',
    'blocks_to_weight',
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


def _kind_and_order(
        kind:   str,
        order:  typ.Any,                # the jet shape as given
        name:   str,                    # the public function, for the messages
) -> typ.Tuple[int, ...]:               # the jet shape, read: a tuple of Python ints
    '''The kind checked, and the jet shape read by the library's one reader (``contractions._read_order``: a tuple
    only, of nonnegative integers). The entries kind has no jets: any jet shape but ``()`` raises.'''
    _require_kind(kind)
    order = contractions._read_order(order, name)
    if kind == 'entries' and order != ():
        raise ValueError("%s: the entries kind has no jets: its order is (), got %r" % (name, order))
    return order


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


def _factorials(
        order:  typ.Tuple[int, ...],    # the jet shape
) -> np.ndarray:                        # HOST float64, shape=F: slot k holds k_1! ... k_m!
    '''The factor of slot ``k`` between a mixed derivative and a Taylor coefficient: ``d^k / ds^k`` of ``s^k`` is
    ``k_1! ... k_m!``. A 0-dimensional one with no jets.'''
    factors = np.ones(tuple(n + 1 for n in order))
    for j, n in enumerate(order):
        along_j = np.array([float(math.factorial(t)) for t in range(n + 1)])
        factors = factors * along_j.reshape((1,) * j + (n + 1,) + (1,) * (len(order) - j - 1))
    return factors


def _scale_slots(
        kind:     str,              # 'probe' | 'apply' | 'entries'
        jets:     typ.Any,          # of the type of z: probe one array F+X per mode; apply F+X; entries X
        order:    typ.Tuple[int, ...],
        divide:   bool,             # True: slot k divided by k_1! ... k_m!; False: multiplied
        xnp,                        # numpy or jax.numpy, chosen once over all the inputs of the call
) -> typ.Any:                       # of the type of z, same shapes (probe: a tuple)
    '''Slot ``k`` of the jet axes of every array divided or multiplied by ``k_1! ... k_m!``. Nothing with no jets.'''
    out = []
    for jet_FX in _arrays(kind, jets):
        jet_FX = xnp.asarray(jet_FX)
        if order:
            factors_F = xnp.asarray(_factorials(order), dtype=_lowest_precision_dtype(xnp, jet_FX))
            factors_F = factors_F.reshape(factors_F.shape + (1,) * (jet_FX.ndim - len(order)))
            jet_FX = xnp.asarray(jet_FX / factors_F if divide else jet_FX * factors_F)
        out.append(jet_FX)
    return _of_type_of_z(kind, out)


def _taylor_weight(
        weight:  typ.Any,           # one array of a weight of the type of z: jet axes after `lead` axes, each 1 or F's
        order:   typ.Tuple[int, ...],
        lead:    int,               # 1 for a mode-leading probe array, else 0
        xnp,
) -> NDArray:                       # slot k of the jet axes multiplied by k_1! ... k_m!; a size-1 jet axis comes back full
    weight_X = xnp.asarray(weight)
    if not order:
        return weight_X
    m = len(order)
    if weight_X.ndim < lead + m:
        raise ValueError('derivative_weight_to_taylor_weight: a weight of rank %d has no %d jet axes after its %d '
                         'leading axes; it is of the type of z, its jet axes first (after the mode axis of a '
                         'mode-leading probe array)' % (weight_X.ndim, m, lead))
    factors_F = xnp.asarray(_factorials(order), dtype=_lowest_precision_dtype(xnp, weight_X))
    return weight_X * factors_F.reshape((1,) * lead + factors_F.shape + (1,) * (weight_X.ndim - lead - m))


def _flipped(
        array_FX:  NDArray,         # m leading jet axes
        m:         int,
) -> NDArray:                       # every jet axis flipped: slot k and slot order - k change places
    return array_FX[(slice(None, None, -1),) * m]


def _mode_weights(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        weight:  typ.Optional[typ.Any],     # omega, of the type of z: probe len=d or a mode-leading array; else one array
        d:       int,                       # the number of arrays of the value it weighs (the modes for probe, else 1)
        name:    str,                       # the public function, for the messages
) -> typ.List[typ.Optional[typ.Any]]:       # one weight array (or None) per array of the value
    '''The weight's array for each array of a value of the type of ``z``, read as :py:func:`data_metric` reads a ragged
    pair: for ``'probe'`` a tuple or a list of exactly ``d`` arrays, or one array whose first axis is the mode, of
    length exactly ``d`` (a mode axis of 1 is not broadcast); for ``'apply'`` and ``'entries'`` one array.'''
    if weight is None:
        return [None] * d
    if kind != 'probe':
        if isinstance(weight, (tuple, list)):
            raise ValueError('%s: the weight of the %s kind is one array, of the type of z; got a %s'
                             % (name, kind, type(weight).__name__))
        return [weight]
    if isinstance(weight, (tuple, list)):
        if len(weight) != d:
            raise ValueError('%s: the value has %d arrays (modes) but the weight has %d' % (name, d, len(weight)))
        return list(weight)
    if np.ndim(weight) < 1 or np.shape(weight)[0] != d:
        raise ValueError('%s: a weight of the probe kind given as one array has the modes along its first axis, '
                         'exactly %d of them (a mode axis of 1 is not broadcast), got shape %s'
                         % (name, d, tuple(np.shape(weight))))
    return [weight[i] for i in range(d)]


####################################
########    The dual jet    ########
####################################

def jet_to_dual_jet(
        kind:    str,                      # 'probe' | 'apply' | 'entries': which type the jet is of
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        jet:     typ.Any,                  # Y*, of the type of z: probe len=d (or packed), elm_shape=F+W+[K]+C+(Ni,); apply F+W+[K]+C; entries W+[K]+C
        weight:  typ.Optional[typ.Any],    # omega, of the type of z (each axis 1 or the jet's size, no K); None = ones
) -> typ.Any:                              # Y, turned: data_metric(jet, weight, None), every jet axis flipped. probe: a tuple
    """Make a jet into a dual jet: the data metric at ``gamma = 1``, then the turn, a flip of every jet axis.

    **Its type is** ``Y* -> Y`` **in the engine's turned coordinates**: ``flip(data_metric(jet, weight, None))``, mode by
    mode. With ``weight=None`` it reads as the turn alone, ``Y -> Y`` turned: an element of ``Y`` in slot coordinates
    (what :py:func:`data_metric` returns) into the engine's form. **It is for**
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`' ``w_tilde`` **and its held tilde variables,
    never for a transpose**: a transpose takes an element of ``Y`` in slot coordinates, unturned, and turns it itself.

    A residual ``r`` is a jet, in forward order. The engine's adjoint sweeps are seeded by a dual jet, the functional
    ``a -> sum_k weight[k]**2 * r[k] * a[k]`` on jets (``k`` a multi-index of the box ``F``, the sum also over the
    samples and the mode edge). It is stored turned,

        ``dual_jet[k] = weight[order - k]**2 * jet[order - k]``,

    so that its pairing with a jet ``a`` is the coefficient of the corner ``s_1^n_1 ... s_m^n_m`` of the product of
    the two as polynomials in ``s``: ``sum_k dual_jet[order - k] * a[k]``. The box keeps this property, which a
    truncation by total degree would not. With no jets there is nothing to turn: the dual jet is ``weight**2 * jet``.

    **The weight is of the type of** ``z``, as :py:func:`data_metric` takes it: the rank of the jet's array, each axis
    1 or its size (:py:func:`blocks_to_weight` makes one from a value per block). A jet with a stack of directions
    ``K`` has no weight of its rank, so it raises when a weight is given; with ``weight=None`` any jet is turned.

    **``kind`` says which type the jet is of**; the object is not looked at to decide it. For ``'probe'`` the jet is
    one array per mode: the function goes over the modes along the first axis of ``jet`` and of the weight (a tuple,
    a list, or one array whose first axis is the mode, of length exactly ``d``), and the result is a tuple. So a
    packed probe jet ``(d,) + F + W + ... + (N,)`` gives what ``tuple(packed)`` gives. For ``'apply'`` and
    ``'entries'`` the jet is one array, its leading axes the jet axes.

    The result is on jax if any array of the jet or the weight is a jax array (then every array of the result
    is), and has the lower floating precision of the jet and the weight (the weight cast, then squared).

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    jet: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        probe: ``len=d``, ``elm_shape=F+W+[K]+C+(Ni,)``. apply: ``shape=F+W+[K]+C``. entries: ``shape=W+[K]+C``.
    weight: tuple of NDArray or NDArray, or None
        ``omega``, of the type of ``z``: each array of the jet's rank, each axis 1 or the jet's size. ``None`` is ones.
        The argument is required: pass ``None`` for no weight.

    Returns
    -------
    tuple of NDArray or NDArray
        The dual jet, with the shapes of ``jet``. probe: a tuple, one array per mode.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds, the entries kind is given a jet shape but ``()``, or the weight is
        not of the type of the jet (another rank, an axis neither 1 nor the jet's size, another mode count).
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    dual_jet_to_jet
    data_metric
    blocks_to_weight

    Examples
    --------
    One sample of the apply kind at the jet shape ``(2,)``, so the jet has shape ``F + W = (3, 1)``, with a weight per
    slot, ``F + (1,)``:

    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> r = np.array([[1.0], [2.0], [3.0]])
    >>> weight = np.array([[1.0], [2.0], [10.0]])
    >>> dual = jets.jet_to_dual_jet('apply', (2,), r, weight)
    >>> print(dual[:, 0].tolist())                          # weight**2 * r is (1, 8, 300); then turned
    [300.0, 8.0, 1.0]

    The pairing with a jet ``a`` is the corner coefficient of the product, and equals the weighted inner product:

    >>> a = np.array([[0.5], [-1.0], [2.0]])
    >>> corner = sum(dual[2 - t] * a[t] for t in range(3))
    >>> print(bool(np.allclose(corner, np.sum(weight ** 2 * r * a, axis=0))))
    True

    In two jet variables the turn flips both jet axes, and the pairing is the corner ``s_1^1 s_2^2`` of the
    product, a sum over the slots ``k`` of the box; here the weight is per entry:

    >>> np.random.seed(0)
    >>> r, a, weight = np.random.randn(2, 3, 4), np.random.randn(2, 3, 4), 0.5 + np.random.rand(2, 3, 4)
    >>> dual = jets.jet_to_dual_jet('apply', (1, 2), r, weight)            # F + W = (2, 3) + (4,)
    >>> print(bool(np.allclose(dual[0, 0], weight[1, 2] ** 2 * r[1, 2])))   # slot (0, 0) holds the corner
    True
    >>> corner = sum(dual[1 - k1, 2 - k2] * a[k1, k2] for k1 in range(2) for k2 in range(3))
    >>> print(bool(np.allclose(corner, np.sum(weight ** 2 * r * a, axis=(0, 1)))))
    True

    For the probe kind the jet is one array per mode, and so is the weight; one per mode and slot, from its blocks:

    >>> r = [np.random.randn(3, 4, N) for N in (5, 6)]      # d = 2, F = (3,), W = (4,)
    >>> weight = jets.blocks_to_weight('probe', (2,), np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]]), r)
    >>> print(weight.shape)                                 # (d,) + F + W + (edge,), size 1 off the blocks
    (2, 3, 1, 1)
    >>> dual = jets.jet_to_dual_jet('probe', (2,), r, weight)
    >>> print(type(dual).__name__, [dual_i.shape for dual_i in dual])
    tuple [(3, 4, 5), (3, 4, 6)]
    >>> print(bool(np.allclose(dual[1][0], 9.0 * r[1][2])))  # mode 1: slot 0 holds slot 2, weight 3
    True

    A weight in the old block form, one value per mode and slot, is not of the type of ``z``:

    >>> jets.jet_to_dual_jet('probe', (2,), r, np.ones((2, 3)))    # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: jet_to_dual_jet: the weight of mode 0 has rank 1, but the array it weighs has rank 3 (shape (3, 4, 5)) ...
    """
    order = _kind_and_order(kind, order, 'jet_to_dual_jet')
    xnp, _, _ = get_backend(False, tree_contains_jax((jet, weight)))        # once, over all the inputs
    arrays = _arrays(kind, jet)
    out = []
    for i, (jet_FX, omega_FX) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays), 'jet_to_dual_jet'))):
        jet_FX = xnp.asarray(jet_FX)
        if omega_FX is not None:
            _check_pair(jet_FX, omega_FX, (' of mode %d' % i) if kind == 'probe' else '', 'jet_to_dual_jet')
        out.append(xnp.asarray(_flipped(data_metric(jet_FX, omega_FX, None), len(order))))
    return _of_type_of_z(kind, out)


def dual_jet_to_jet(
        kind:      str,                    # 'probe' | 'apply' | 'entries': which type the dual jet is of
        order:     typ.Tuple[int, ...],    # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        dual_jet:  typ.Any,                # Y, turned: same type and shapes as jet_to_dual_jet's result
        weight:    typ.Optional[typ.Any],  # omega, of the type of z (each axis 1 or the dual jet's size); None = ones
) -> typ.Any:                              # Y*: every jet axis flipped back, then divided by weight**2. probe: a tuple
    """Make a dual jet into a jet: the inverse of :py:func:`jet_to_dual_jet`.

    **Its type is** ``Y -> Y*`` **from the engine's turned coordinates**, the inverse of the data metric at
    ``gamma = 1`` after the turn back; with ``weight=None`` it is the turn back alone, the engine's form of an element
    of ``Y`` into slot coordinates. Like :py:func:`jet_to_dual_jet` it is for the engine's ``w_tilde`` and its held
    tilde variables, never for a transpose.

    Every jet axis is flipped back and each entry is divided by ``weight**2``:

        ``jet[k] = dual_jet[order - k] / weight[k]**2``.

    The weight is of the type of ``z``, read as :py:func:`jet_to_dual_jet` reads it. **Where a weight is zero this
    divides by zero**: the result is ``inf`` or ``nan`` there, and nothing is raised or warned, on numpy and on jax
    alike. The dual jet of a zero weight keeps nothing of the jet, so there is nothing to give back. The divisor is
    ``weight**2`` in the precision of the computation (the lower floating precision of the dual jet and the weight,
    the weight cast, then squared), so a weight whose square underflows there divides by zero too.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    dual_jet: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        probe: ``len=d``, ``elm_shape=F+W+[K]+C+(Ni,)``. apply: ``shape=F+W+[K]+C``. entries: ``shape=W+[K]+C``.
    weight: tuple of NDArray or NDArray, or None
        ``omega``, of the type of ``z``. ``None`` is ones. The argument is required: pass ``None`` for no weight.

    Returns
    -------
    tuple of NDArray or NDArray
        The jet, with the shapes of ``dual_jet``. probe: a tuple, one array per mode.

    Raises
    ------
    ValueError
        As :py:func:`jet_to_dual_jet`.
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    jet_to_dual_jet

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> np.random.seed(0)
    >>> r = np.random.randn(3, 4)                           # apply: F + W = (3, 4)
    >>> weight = np.array([[1.0], [2.0], [10.0]])           # per slot
    >>> dual = jets.jet_to_dual_jet('apply', (2,), r, weight)
    >>> print(bool(np.allclose(jets.dual_jet_to_jet('apply', (2,), dual, weight), r)))
    True

    A zero weight divides by zero, without an error:

    >>> back = jets.dual_jet_to_jet('apply', (2,), dual, np.array([[1.0], [0.0], [10.0]]))
    >>> print(np.isfinite(back).all(axis=1).tolist())       # slot 1 is lost
    [True, False, True]
    """
    order = _kind_and_order(kind, order, 'dual_jet_to_jet')
    use_jax = tree_contains_jax((dual_jet, weight))                         # once, over all the inputs
    xnp, _, _ = get_backend(False, use_jax)
    arrays = _arrays(kind, dual_jet)
    out = []
    for i, (dual_FX, omega_FX) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays), 'dual_jet_to_jet'))):
        dual_FX = xnp.asarray(dual_FX)
        jet_FX = _flipped(xnp.asarray(dual_FX, dtype=_lowest_precision_dtype(xnp, dual_FX, omega_FX)), len(order))
        if omega_FX is not None:
            _check_pair(jet_FX, omega_FX, (' of mode %d' % i) if kind == 'probe' else '', 'dual_jet_to_jet')
            omega_squared_FX = _omega_squared(jet_FX, omega_FX, xnp)
            if use_jax:
                jet_FX = jet_FX / omega_squared_FX
            else:
                with np.errstate(divide='ignore', invalid='ignore'):    # a zero divisor: inf or nan, silently, as on jax
                    jet_FX = jet_FX / omega_squared_FX
        out.append(xnp.asarray(jet_FX))
    return _of_type_of_z(kind, out)


######################################
########    The data metric    #######
######################################

def _omega_squared(
        r_X:      typ.Any,                  # one array of r
        omega_X:  typ.Any,                  # its weight: the same rank, each axis 1 or r's size
        xnp,
) -> NDArray:                               # omega**2, in the lower floating precision of the pair
    '''``omega * omega``, formed in the lower floating precision of ``r`` and ``omega`` (the absorbing rule): the core
    of :py:func:`data_metric`, the one place the squared weight of the full form is made.'''
    omega_X = xnp.asarray(omega_X, dtype=_lowest_precision_dtype(xnp, r_X, omega_X))
    return omega_X * omega_X


def _check_pair(
        r_X:      typ.Any,
        omega_X:  typ.Any,
        where:    str,                      # which array, for the message
        name:     str = 'data_metric',      # the public function, for the message
) -> None:
    '''The structural rule of a pair: equal rank, and each axis of the weight 1 or the array's size, so the product
    has exactly the array's shape. Static, so it runs under ``jax.jit``.'''
    r_shape, omega_shape = tuple(np.shape(r_X)), tuple(np.shape(omega_X))
    if len(omega_shape) != len(r_shape):
        raise ValueError('%s: the weight%s has rank %d, but the array it weighs has rank %d (shape %s); a '
                         'weight is of the type of z, each axis 1 or its size, and never carries the stack of '
                         'directions K' % (name, where, len(omega_shape), len(r_shape), r_shape))
    for axis, (a, b) in enumerate(zip(omega_shape, r_shape)):
        if a not in (1, b):
            raise ValueError('%s: axis %d of the weight%s has size %d, but the array it weighs has size %d '
                             'there (shapes %s and %s); each axis of a weight is 1 or the size of the array'
                             % (name, axis, where, a, b, omega_shape, r_shape))


def data_metric(
        r:                typ.Any,                  # Y*, of the type of z, no K: probe len=d (or packed (d,)+...), elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:           typ.Optional[typ.Any],    # omega, of the type of z: each axis 1 or r's size; None = ones
        minibatch_scale:  typ.Optional[typ.Any],    # gamma: a number or a 0-d array (traced under jit); None = 1
) -> typ.Any:                                       # M r: Y, gamma * omega**2 * r elementwise, r's structure (a tuple for a tuple or a list)
    """The data metric ``M: Y* -> Y``: ``gamma * omega**2 * r``, elementwise.

    The misfit of a least-squares fit is ``f = 1/2 <M r, r>`` with ``r = z - y``, and its gradient is ``J^T (M r)``.
    ``r`` is a measurement, of the type of ``z`` (an element of ``Y*``); ``M r`` is an element of ``Y``, the space the
    transposes take, paired with ``z`` slot by slot. So the gradient of a weighted fit is a transpose applied to
    this function's result, ``t3_apply_corewise_transpose(x, ww, data_metric(r, weight, minibatch_scale), order)``
    for instance; a transpose applies no weight of its own. On a batch the method
    :py:meth:`~t3toolbox.backend.fitting.LeastSquaresBatch.data_metric` passes the batch's own weight and scale.

    **The weight is of the type of** ``z``: the same structure and the same rank as ``r``, each axis of size 1 or
    ``r``'s size (a size of 1 means the same along that axis: a weight per slot, per sample, per entry, ...). It
    enters squared, and a zero drops an entry. :py:func:`blocks_to_weight` makes a weight of this form from one value
    per block (per mode and slot for probe, per slot for apply, one number for entries). ``minibatch_scale`` is the
    scale ``gamma`` of a drawn batch, the number of samples of the full batch over the number drawn, so that a
    drawn batch's misfit estimates the full one's. Both are required: ``None`` written out means ones, or a scale
    of 1.

    **How the arrays pair.** ``r`` of the probe kind is ``d`` arrays, a tuple or a list (one per mode); its weight is
    a tuple or a list of ``d`` arrays, or one array whose first axis is the mode, of length exactly ``d`` (a mode
    axis of 1 is not broadcast). An array ``r`` (apply, entries, or a probe packed for the uniform layout,
    ``(d,) + F + W + C + (N,)``) pairs with an array weight, elementwise. A packed ``r`` with a ragged weight raises:
    pack the weight with the data (:py:func:`~t3toolbox.backend.uniform_fitting.pack_batch`). Against a packed
    ``r`` a packed weight with a mode axis of 1 is broadcast over the modes: the checkers take the ragged form, and
    the packing builds the packed weight from it. Values in a packed ``r``'s padding come back multiplied by
    ``omega**2``: masking them is the engine's business, not this function's.

    **What raises** (structural, so also under ``jax.jit``): a weight of another rank than its array, which includes
    an ``r`` with a stack of directions ``K`` (products with a stack of directions go through the oracle); an axis
    of the weight that is neither 1 nor ``r``'s size (a weight never changes ``r``'s shape); a mode count that does
    not match; a packed ``r`` with a ragged weight; a scale that is not a scalar. With ``weight=None`` there is no
    rank to compare, and the result is ``gamma * r`` for any ``r``, a stack of directions included.

    **Coordinates.** Data are Taylor coefficients, everywhere in the library: data given as derivatives are
    converted once, the data and the weight together, by :py:func:`prepare_derivative_data`. A weight computed from
    the data is computed after the conversion, on the Taylor data, for instance
    ``blocks_to_weight(kind, order, equalizing_weight(data_block_norms(kind, order, y)), y)``, and then needs no
    conversion. An element of ``Y`` (this function's result) held in derivative coordinates converts the other way
    than a measurement: slot ``k`` of a measurement in derivatives is ``k_1! ... k_m!`` times its Taylor coefficient,
    so the pairing makes slot ``k`` of an element of ``Y`` in Taylor coordinates ``k_1! ... k_m!`` times its value in
    derivatives, and it converts with :py:func:`taylor_to_derivatives`, the converter of the opposite name.

    The result has the lower floating precision of ``r`` and the weight (the scale is cast to it, and does not
    raise it), and is on jax if any input is.

    Parameters
    ----------
    r: tuple of NDArray or NDArray
        An element of ``Y*``, of the type of ``z``: probe ``len=d``, ``elm_shape=F+W+C+(Ni,)`` (or packed); apply
        ``shape=F+W+C``; entries ``shape=W+C``.
    weight: tuple of NDArray, NDArray or None
        ``omega``, of the type of ``z``: each array of ``r``'s rank, each axis 1 or ``r``'s size. ``None`` is ones.
    minibatch_scale: float, 0-dimensional NDArray or None
        ``gamma``: 1 on a full batch, the number of samples over the number drawn on a drawn batch. ``None`` is 1.

    Returns
    -------
    tuple of NDArray or NDArray
        ``gamma * omega**2 * r``, an element of ``Y`` with ``r``'s shapes: a tuple for a tuple or a list.

    Raises
    ------
    ValueError
        For the structural errors listed above.

    See Also
    --------
    blocks_to_weight
    t3toolbox.backend.fitting.LeastSquaresBatch.data_metric

    Examples
    --------
    One number per sample (apply, ``F + W = (3,) + (4,)``), a weight per slot (``F + (1,)``) and a drawn batch's
    scale:

    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> r = np.ones((3, 4))
    >>> weight = np.array([[1.0], [2.0], [3.0]])                         # F + (1,): one value per slot
    >>> print(jets.data_metric(r, weight, 2.0)[:, 0].tolist())          # 2 * weight**2 * r
    [2.0, 8.0, 18.0]
    >>> print(jets.data_metric(r, None, None).tolist() == r.tolist())   # no weight, no scale: r itself
    True

    The weight from one value per block, and the probe kind, one array per mode:

    >>> rng = np.random.default_rng(0)
    >>> y = [rng.standard_normal((3, 4, N)) for N in (5, 6)]           # d = 2, F + W + (Ni,)
    >>> weight = jets.blocks_to_weight('probe', (2,), np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]]), y)
    >>> print(weight.shape)                                             # (d,) + F + (1,) for W + (1,) for the edge
    (2, 3, 1, 1)
    >>> c = jets.data_metric(y, weight, None)
    >>> print(type(c).__name__, bool(np.allclose(c[1][2], 9.0 * y[1][2])))
    tuple True

    A residual with a stack of directions has no weight of its rank:

    >>> jets.data_metric(np.ones((2, 3, 4)), np.ones((3, 1)), None)    # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: data_metric: the weight has rank 2, but the array it weighs has rank 3 ...

    The gradient of a weighted fit: a transpose applied to the result.

    >>> import t3toolbox.backend.apply as apply
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(7, N) for N in (4, 5, 6)]                # W = (7,), no jets
    >>> y = np.random.randn(7)
    >>> weight = np.random.rand(7)                                      # one weight per sample
    >>> r = apply.t3_apply(x, ww) - y
    >>> g = apply.t3_apply_corewise_transpose(x, ww, jets.data_metric(r, weight, None))
    >>> dx = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> f = lambda t: 0.5 * float(np.sum(weight**2 * (apply.t3_apply((tuple(a + t * b for a, b in zip(x[0], dx[0])),
    ...                                                                  tuple(a + t * b for a, b in zip(x[1], dx[1]))),
    ...                                                                 ww) - y)**2))
    >>> slope = sum(float(np.sum(gi * di)) for gs, ds in zip(g, dx) for gi, di in zip(gs, ds))
    >>> print(bool(np.isclose(slope, (f(1e-6) - f(-1e-6)) / 2e-6, rtol=1e-6)))
    True
    """
    xnp, _, _ = get_backend(False, tree_contains_jax((r, weight, minibatch_scale)))   # once, over all the inputs
    if minibatch_scale is not None and np.ndim(minibatch_scale) != 0:
        raise ValueError('data_metric: the minibatch scale is one number (a scalar or a 0-d array), got shape %s; a '
                         'weight per sample or per stack element belongs in the weight'
                         % (np.shape(minibatch_scale),))
    ragged = isinstance(r, (tuple, list))
    if ragged:
        arrays = list(r)
        if weight is None:
            weights = [None] * len(arrays)
        elif isinstance(weight, (tuple, list)):
            if len(weight) != len(arrays):
                raise ValueError('data_metric: r has %d arrays (modes) but the weight has %d'
                                 % (len(arrays), len(weight)))
            weights = list(weight)
        else:
            if np.ndim(weight) < 1 or np.shape(weight)[0] != len(arrays):
                raise ValueError('data_metric: r has %d arrays (modes); a weight given as one array has them along '
                                 'its first axis, exactly %d of them (a mode axis of 1 is not broadcast), got shape %s'
                                 % (len(arrays), len(arrays), np.shape(weight)))
            weights = [weight[i] for i in range(len(arrays))]
    else:
        if isinstance(weight, (tuple, list)):
            raise ValueError('data_metric: an array r (packed for the uniform layout, or of the apply or entries kind) '
                             'pairs with an array weight; a ragged weight with a packed r is a layout mix: pack the '
                             'weight with the data (uniform_fitting.pack_batch)')
        arrays, weights = [r], [weight]
    out = []
    for i, (r_X, omega_X) in enumerate(zip(arrays, weights)):
        dtype = _lowest_precision_dtype(xnp, r_X, omega_X)
        if dtype is None:
            dtype = np.dtype(xnp.result_type(float))
        r_X = xnp.asarray(r_X, dtype=dtype)
        if omega_X is not None:
            _check_pair(r_X, omega_X, (' of mode %d' % i) if ragged else '')
            r_X = _omega_squared(r_X, omega_X, xnp) * r_X
        if minibatch_scale is not None:
            r_X = xnp.asarray(minibatch_scale, dtype=r_X.dtype) * r_X
        out.append(xnp.asarray(r_X))                    # a 0-d product is an array, not a numpy scalar
    return tuple(out) if ragged else out[0]


def blocks_to_weight(
        kind:    str,                               # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],               # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        blocks:  typ.Any,                           # one value per block: probe (d,)+F; apply F; entries ()
        y:       typ.Any,                           # Y*, of the type of z: only its rank is read (C included)
) -> NDArray:                                       # omega: probe (d,)+F+(1,)*(len W+len C)+(1,); apply F+(1,)*(len W+len C); entries (1,)*(len W+len C)
    """A weight of the type of ``z`` from one value per block: the blocks on the jet axes, size 1 on every other.

    A block is the data of one mode at one slot of the box ``F`` (probe), of one slot (apply), or all of the data
    (entries). Per-block quantities such as :py:func:`~t3toolbox.backend.data_preparation.equalizing_weight` of
    :py:func:`~t3toolbox.backend.data_preparation.data_block_norms`, a noise level per mode, or a weight per slot
    have that shape: ``(d,) + F``, ``F`` or ``()``. This places them on the leading jet axes of a weight shaped like
    the data ``y``, with size 1 on the sample axes ``W``, the stack ``C`` and (probe) the mode edge, which is the
    form :py:func:`data_metric` takes. For the probe kind the result is one array whose first axis is the mode (the
    modes differ only in their edge, which is 1 here). Only the rank of ``y`` is read: ragged or packed, with or
    without the stack ``C``. The blocks keep their dtype and their backend.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    blocks: NDArray
        probe ``shape=(d,)+F``; apply ``shape=F``; entries ``shape=()``.
    y: tuple of NDArray or NDArray
        Of the type of ``z``: probe ``len=d``, ``elm_shape=F+W+C+(Ni,)`` (or packed ``(d,)+F+W+C+(N,)``); apply
        ``shape=F+W+C``; entries ``shape=W+C``.

    Returns
    -------
    NDArray
        probe ``shape=(d,)+F+(1,)*(len W+len C)+(1,)``; apply ``shape=F+(1,)*(len W+len C)``; entries
        ``shape=(1,)*(len W+len C)``.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds, the entries kind is given a jet shape, the blocks do not have the
        block shape, or ``y`` has too low a rank (or, ragged, modes of different ranks).
    TypeError
        If ``order`` is not a tuple of integers.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> y = np.zeros((3, 7, 2))                                         # apply: F + W + C = (3,) + (7,) + (2,)
    >>> weight = jets.blocks_to_weight('apply', (2,), np.array([1.0, 2.0, 3.0]), y)
    >>> print(weight.shape, weight[:, 0, 0].tolist())
    (3, 1, 1) [1.0, 2.0, 3.0]
    >>> print(jets.blocks_to_weight('entries', (), 0.5, np.zeros(7)).shape)
    (1,)
    """
    order = _kind_and_order(kind, order, 'blocks_to_weight')
    m = len(order)
    box = tuple(n + 1 for n in order)
    if kind == 'probe':
        if isinstance(y, (tuple, list)):
            ranks = {int(np.ndim(y_i)) for y_i in y}
            if len(ranks) != 1:
                raise ValueError('blocks_to_weight: the modes of y have different ranks %s' % sorted(ranks))
            d, n_stack = len(y), ranks.pop() - m - 1
        else:
            d, n_stack = int(np.shape(y)[0]), int(np.ndim(y)) - 1 - m - 1
        expected, tail = (d,) + box, (1,) * max(n_stack, 0) + (1,)
    elif kind == 'apply':
        n_stack = int(np.ndim(y)) - m
        expected, tail = box, (1,) * max(n_stack, 0)
    else:
        n_stack = int(np.ndim(y))
        expected, tail = (), (1,) * n_stack
    if n_stack < 0:
        raise ValueError('blocks_to_weight: y has too low a rank for the %s kind at the jet shape %s' % (kind, order))
    if tuple(np.shape(blocks)) != expected:
        raise ValueError('blocks_to_weight: the blocks of the %s kind at the jet shape %s have shape %s, got %s'
                         % (kind, order, expected, tuple(np.shape(blocks))))
    xnp, _, _ = get_backend(False, tree_contains_jax(blocks))
    return xnp.reshape(xnp.asarray(blocks), expected + tail)


#######################################
########    Derivative data    ########
#######################################

def derivatives_to_taylor(
        kind:   str,                       # 'probe' | 'apply' | 'entries': which type the jets are of
        order:  typ.Tuple[int, ...],       # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        jets:   typ.Any,                   # derivatives: probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W
) -> typ.Any:                              # Taylor coefficients, same shapes: slot k divided by k_1! ... k_m!. probe: a tuple
    """Derivatives to Taylor coefficients: slot ``k`` of the jet axes divided by ``k_1! ... k_m!``.

    The library holds a jet as Taylor coefficients, the coefficient of ``s_1^k_1 ... s_m^k_m`` in slot ``k``. The
    mixed derivative ``d^k / ds^k`` at ``s = 0`` is ``k_1! ... k_m!`` times it. Data measured as derivatives are
    converted with this function before they are given to a fit (or with :py:func:`prepare_derivative_data`, which
    also takes care of the weight). With no jets nothing changes (the entries kind: its data as they are).

    This converts a measurement, an element of ``Y*`` (data, a residual, an output). An element of ``Y`` (what
    :py:func:`data_metric` returns, what a transpose takes) converts the contragredient way: its pairing with a
    measurement must not change, so slot ``k`` is multiplied where a measurement's is divided, and the conversion of an
    element of ``Y`` from derivative to Taylor coordinates is :py:func:`taylor_to_derivatives`.

    ``kind`` says which type the jets are of. For ``'probe'`` one array per mode, taken along the first axis of
    ``jets`` (a tuple, a list, or an array packed for the uniform layout, ``(d,) + F + ...``), and a tuple is
    returned; for ``'apply'`` and ``'entries'`` one array, its leading axes the jet axes. A tuple given with
    ``kind='apply'`` is not of the apply type (see :py:func:`jet_to_dual_jet`); nothing here checks for it.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    jets: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        Derivatives. probe: ``len=d``, ``elm_shape=F+W+(Ni,)``. apply: ``shape=F+W``. entries: ``shape=W``. Any
        axes after the jet axes are carried along.

    Returns
    -------
    tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        Taylor coefficients, with the shapes of ``jets``. probe: a tuple, one array per mode.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds, or the entries kind is given a jet shape but ``()``.
    TypeError
        If ``order`` is not a tuple of integers.

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
    >>> derivatives = np.array([[1.0], [2.0], [4.0], [8.0]])        # apply: F + W = (4, 1)
    >>> taylor = jets.derivatives_to_taylor('apply', (3,), derivatives)
    >>> print(bool(np.allclose(taylor[:, 0], [1.0, 2.0, 2.0, 8.0 / 6.0])))
    True

    In two variables: the mixed derivatives of ``exp(2 s_1 + 3 s_2)`` are ``2**k_1 3**k_2``, its Taylor
    coefficients ``2**k_1 3**k_2 / (k_1! k_2!)``:

    >>> derivatives = np.array([[2.0 ** k1 * 3.0 ** k2 for k2 in range(3)] for k1 in range(2)])  # F = (2, 3)
    >>> taylor = jets.derivatives_to_taylor('apply', (1, 2), derivatives)
    >>> print(bool(np.allclose(taylor[1, 2], 2.0 * 9.0 / 2.0)))
    True

    The probe kind: one array per mode in, a tuple out.

    >>> np.random.seed(0)
    >>> derivatives = [np.random.randn(4, 5, N) for N in (6, 7)]    # d = 2, F = (4,), W = (5,)
    >>> taylor = jets.derivatives_to_taylor('probe', (3,), derivatives)
    >>> print(type(taylor).__name__, [t_i.shape for t_i in taylor])
    tuple [(4, 5, 6), (4, 5, 7)]
    >>> print(bool(np.allclose(taylor[1][3], derivatives[1][3] / 6.0)))
    True
    """
    order = _kind_and_order(kind, order, 'derivatives_to_taylor')
    xnp, _, _ = get_backend(False, tree_contains_jax(jets))                 # once, over all the inputs
    return _scale_slots(kind, jets, order, True, xnp)


def taylor_to_derivatives(
        kind:   str,                       # 'probe' | 'apply' | 'entries': which type the jets are of
        order:  typ.Tuple[int, ...],       # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        jets:   typ.Any,                   # Taylor coefficients: probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W
) -> typ.Any:                              # derivatives, same shapes: slot k multiplied by k_1! ... k_m!. probe: a tuple
    """Taylor coefficients to derivatives: slot ``k`` of the jet axes multiplied by ``k_1! ... k_m!``.

    The reverse of :py:func:`derivatives_to_taylor`, for a user who wants an output of the library (which is
    always in Taylor coefficients) as mixed derivatives. On an element of ``Y`` (the dual side, :py:func:`data_metric`'s
    result) it is the conversion the other way, from derivative to Taylor coordinates (see
    :py:func:`derivatives_to_taylor`). ``kind`` says which type the jets are of, in the same way:
    for ``'probe'`` one array per mode, taken along the first axis (a tuple, a list, or a packed array), a tuple
    returned; for ``'apply'`` and ``'entries'`` one array, its leading axes the jet axes.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    jets: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        Taylor coefficients. probe: ``len=d``, ``elm_shape=F+W+(Ni,)``. apply: ``shape=F+W``. entries:
        ``shape=W``.

    Returns
    -------
    tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        Derivatives, with the shapes of ``jets``. probe: a tuple, one array per mode.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds, or the entries kind is given a jet shape but ``()``.
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    derivatives_to_taylor

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> taylor = np.ones((4, 1))                                    # 1 / (1 - s): every coefficient is 1
    >>> print(jets.taylor_to_derivatives('apply', (3,), taylor)[:, 0].tolist())
    [1.0, 1.0, 2.0, 6.0]
    """
    order = _kind_and_order(kind, order, 'taylor_to_derivatives')
    xnp, _, _ = get_backend(False, tree_contains_jax(jets))                 # once, over all the inputs
    return _scale_slots(kind, jets, order, False, xnp)


def derivative_weight_to_taylor_weight(
        kind:    str,                      # 'probe' | 'apply' | 'entries': which type the weight is of
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        weight:  typ.Any,                  # omega on derivative residuals, of the type of z: probe len=d or (d,)+...; else one array
) -> typ.Any:                              # omega on Taylor residuals, in the form given: slot k times k_1! ... k_m!
    """The Taylor weight that gives the misfit of a weight on derivative residuals: slot ``k`` times
    ``k_1! ... k_m!``.

    A derivative residual is ``k_1! ... k_m!`` times the Taylor residual, so ``weight[k] * (derivative residual)``
    equals ``(weight[k] * k_1! ... k_m!) * (Taylor residual)``. The weight goes the opposite way to the data, which
    are divided by the factorials, and it is the weight itself that is multiplied, not its square.

    The weight is of the type of ``z`` (each axis 1 or the data's size), so its jet axes are its **leading** ``m`` axes:
    of each array of a tuple or a list (probe), after the mode axis of a mode-leading probe array, and of the one array
    of apply. A jet axis of size 1 (a weight constant over the slots) comes back with the full ``F``: the factorials
    differ by slot. The weight comes back in the form it was given (a tuple for a tuple or a list). With no jets
    nothing changes. A weight of too low a rank for its jet axes raises; nothing else is checked here
    (:py:func:`prepare_derivative_data` checks the weight against the data).

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    weight: tuple of NDArray or NDArray
        A weight on derivative residuals, of the type of ``z``.

    Returns
    -------
    tuple of NDArray or NDArray
        The weight on Taylor coefficient residuals, in the form given.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds, the entries kind is given a jet shape but ``()``, or a weight's
        array has fewer axes than its jet axes need.
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    prepare_derivative_data
    blocks_to_weight

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> weight = np.array([1.0, 10.0, 10.0, 10.0]).reshape(4, 1)      # apply, per slot: F + W = (4,) + (1,)
    >>> print(jets.derivative_weight_to_taylor_weight('apply', (3,), weight)[:, 0].tolist())
    [1.0, 10.0, 20.0, 60.0]

    A weight constant over the slots (a jet axis of 1) comes back per slot:

    >>> print(jets.derivative_weight_to_taylor_weight('apply', (3,), np.ones((1, 5))).shape)
    (4, 5)

    The probe kind, a mode-leading array ``(d,) + F + W + (edge,)``: the jet axis is after the mode axis.

    >>> probe_weight = np.ones((2, 4, 1, 1))                           # d = 2 modes, F = (4,)
    >>> print(jets.derivative_weight_to_taylor_weight('probe', (3,), probe_weight)[:, :, 0, 0].tolist())
    [[1.0, 1.0, 2.0, 6.0], [1.0, 1.0, 2.0, 6.0]]
    >>> print(jets.derivative_weight_to_taylor_weight('apply', (1, 2), np.ones((2, 3, 1)))[..., 0].tolist())  # k_1! k_2!
    [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]]
    """
    order = _kind_and_order(kind, order, 'derivative_weight_to_taylor_weight')
    xnp, _, _ = get_backend(False, tree_contains_jax(weight))
    return _taylor_weights(kind, order, weight, xnp)


def _taylor_weights(
        kind:    str,
        order:   typ.Tuple[int, ...],       # read
        weight:  typ.Any,                   # omega on derivative residuals, of the type of z
        xnp,                                # chosen once by the caller, over all the inputs of its call
) -> typ.Any:                               # omega on Taylor residuals, in the form given
    if kind == 'probe' and isinstance(weight, (tuple, list)):
        return tuple(_taylor_weight(omega, order, 0, xnp) for omega in weight)
    return _taylor_weight(weight, order, 1 if kind == 'probe' else 0, xnp)


def prepare_derivative_data(
        kind:        str,                              # 'probe' | 'apply' | 'entries'
        order:       typ.Tuple[int, ...],              # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        y:           typ.Any,                          # derivatives, of the type of z: probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:      typ.Optional[typ.Any] = None,     # omega on derivative residuals, of the type of z; None = ones on the derivatives
        validation:  typ.Optional[typ.Any] = None,     # validation derivatives, of the type of z, or None
) -> typ.Tuple[
        typ.Any,                                       # y in Taylor coefficients, same type and shapes
        typ.Any,                                       # the Taylor weight, of the type of z: k_1! ... k_m! per slot when weight is None
        typ.Optional[typ.Any],                         # validation in Taylor coefficients, or None
]:
    """Prepare derivative data for a fit: the data, the validation data and the weight, in one call.

    **The library's data coordinates are Taylor coefficients, everywhere.** Data measured as derivatives are converted
    once, at the boundary, data and weight together, by this function, and every stored copy (the data, the validation
    data, anything compared with a model's output) stays in Taylor coordinates. The outputs of the fitted tensor are
    Taylor coefficients too; :py:func:`taylor_to_derivatives` converts them back.

    **The weight** is of the type of ``z`` (each axis 1 or the data's size; :py:func:`blocks_to_weight` makes one from a
    value per block) and is checked against ``y`` here. A weight on the derivative residuals becomes the weight on the
    Taylor residuals that gives the same misfit (:py:func:`derivative_weight_to_taylor_weight`: slot ``k`` times
    ``k_1! ... k_m!``). With ``weight=None`` the default is ones in the data's own coordinates, the derivatives, so the
    weight returned is ``k_1! ... k_m!`` per slot, in the form of :py:func:`blocks_to_weight` (size 1 off the jet axes).
    That default is rarely what data with decaying jets want: their high slots then dominate.

    **The recipe for a weight computed from the data** (an equalizing weight, say): convert the data first, then
    compute the weight from the Taylor data; it then needs no conversion::

        y_taylor, _, _ = prepare_derivative_data(kind, order, y_derivatives)
        weight = blocks_to_weight(kind, order, equalizing_weight(data_block_norms(kind, order, y_taylor)), y_taylor)

    Taylor coordinates widen the spread across slots by up to ``n!`` against derivatives, so an equalizing weight's
    relative floor depends on the coordinates; rescale the direction first, then equalize.

    For ``'probe'`` the data are one array per mode, taken along the first axis of ``y`` (a tuple, a list, or an array
    packed for the uniform layout), and a tuple is returned. The entries kind has no jets: its data are returned as
    they are, with the weight given, or a weight of ones. The result is on jax if any array given is a jax array.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. With no jets, ``()``, there is nothing to convert and the weight returned is ones.
    y: tuple of NDArray or NDArray
        Of the type of ``z``: the data, as derivatives. probe: ``len=d``, ``elm_shape=F+W+C+(Ni,)``. apply:
        ``shape=F+W+C``. entries: ``shape=W+C``.
    weight: tuple of NDArray or NDArray, or None
        A weight on the derivative residuals, of the type of ``z``. ``None`` is ones on the derivatives.
    validation: tuple of NDArray or NDArray, or None
        Validation data, as derivatives, of the type of ``z``.

    Returns
    -------
    y_taylor: tuple of NDArray or NDArray
        The data in Taylor coefficients.
    weight_taylor: tuple of NDArray or NDArray
        The weight on Taylor coefficient residuals, of the type of ``z``: in the form given, or with ``weight=None`` one
        array (mode-leading for probe) from :py:func:`blocks_to_weight`.
    validation_taylor: tuple of NDArray or NDArray, or None
        The validation data in Taylor coefficients, or ``None`` if none were given.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds, the entries kind is given a jet shape but ``()``, or the weight is not
        of the type of ``y`` (another rank, an axis neither 1 nor ``y``'s size, another mode count).
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    derivatives_to_taylor
    derivative_weight_to_taylor_weight
    blocks_to_weight

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> y_derivatives = np.array([[1.0], [1.0], [2.0], [6.0]])      # apply: F + W = (4, 1)
    >>> y, weight, validation = jets.prepare_derivative_data('apply', (3,), y_derivatives)
    >>> print(y[:, 0].tolist(), weight[:, 0].tolist(), validation)
    [1.0, 1.0, 1.0, 1.0] [1.0, 1.0, 2.0, 6.0] None

    The module's docstring walks through the cases: a weight chosen for derivatives, no weight, validation
    data, two jet variables, and the probe kind.
    """
    order = _kind_and_order(kind, order, 'prepare_derivative_data')
    xnp, _, _ = get_backend(False, tree_contains_jax((y, weight, validation)))     # once, over all the inputs
    y_taylor = _scale_slots(kind, y, order, True, xnp)
    arrays = _arrays(kind, y_taylor)
    if weight is None:
        dtype = _lowest_precision_dtype(xnp, arrays[0])
        factors_F = xnp.asarray(_factorials(order), dtype=np.dtype(xnp.result_type(float)) if dtype is None else dtype)
        blocks = xnp.broadcast_to(factors_F, (len(arrays),) + factors_F.shape) if kind == 'probe' else factors_F
        weight_taylor = blocks_to_weight(kind, order, blocks, y_taylor)
    else:
        for i, (y_X, omega_X) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays),
                                                                     'prepare_derivative_data'))):
            _check_pair(y_X, omega_X, (' of mode %d' % i) if kind == 'probe' else '', 'prepare_derivative_data')
        weight_taylor = _taylor_weights(kind, order, weight, xnp)
    validation_taylor = None if validation is None else _scale_slots(kind, validation, order, True, xnp)
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

def _require_input_jets(
        ww:     typ.Any,                     # expected: len=d, elm_shape=F_in_i+W+(Ni,)
        shape:  typ.Sequence[int],           # the mode sizes N_i, len=d
        m:      int,                         # the number of jet variables
) -> typ.Tuple[int, ...]:                    # W, read from the first input after its m jet axes
    '''A tuple or a list of ``d`` arrays, the ``i``-th of shape ``F_in_i + W + (N_i,)`` with ``m`` jet axes of at
    least one slot each, and one ``W`` for all.'''
    d = len(shape)
    form = 'F_in_i+W+(Ni,), with %d leading jet axes' % m if m else 'W+(Ni,)'
    if not isinstance(ww, (tuple, list)):
        raise ValueError('ww must be a tuple or a list of %d arrays, one per mode, the i-th of shape %s; got a %s'
                         % (d, form, type(ww).__name__))
    if len(ww) != d:
        raise ValueError('ww must have one array per mode: expected %d, got %d' % (d, len(ww)))
    for i, w in enumerate(ww):
        if isinstance(w, (tuple, list)):
            raise ValueError('ww[%d] must be one array of shape %s; got a %s' % (i, form, type(w).__name__))
        if len(np.shape(w)) < m + 1:
            raise ValueError('ww[%d] has shape %s: too few axes for the jet shape, which has %d jet axes, before the '
                             'sample axes W and the mode axis' % (i, tuple(np.shape(w)), m))
        if m and min(np.shape(w)[:m]) < 1:
            raise ValueError('ww[%d] has the jet sizes %s: every jet axis has at least one slot'
                             % (i, tuple(np.shape(w)[:m])))
    W = tuple(np.shape(ww[0])[m:-1])
    for i, w in enumerate(ww):
        if tuple(np.shape(w)[m:-1]) != W:
            raise ValueError('ww[%d] has the sample stack W = %s once its %d jet axes are read; ww[0] has W = %s. '
                             'All modes share one W (is the jet shape right?)'
                             % (i, tuple(np.shape(w)[m:-1]), m, W))
        if np.shape(w)[-1] != shape[i]:
            raise ValueError('ww[%d] has the mode size %d; expected N_%d = %d' % (i, np.shape(w)[-1], i, shape[i]))
    return W


def check_sample(
        kind:    str,                      # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); entries: ()
        sample:  typ.Any,                  # ww (probe, apply: input jets) | index (entries)
        shape:   typ.Sequence[int],        # the mode sizes N_i, len=d
) -> None:                                 # d arrays F_in_i+W+(Ni,), each with len(order) jet axes of size >= 1 and
                                           #   one W for all; or index (d,)+W, integer
    """Check the form and the shapes of a sample against the jet shape and the mode sizes; raise a ``ValueError``
    if wrong.

    What a good sample is:

    - for ``'probe'`` and ``'apply'``, the input jets ``ww``: a tuple or a list of ``d`` arrays, the ``i``-th of
      shape ``F_in_i + W + (N_i,)``. Its first ``m = len(order)`` axes are the jet axes, of sizes of its own, each
      at least 1 (an input with fewer slots than the box is zero beyond them; one with more is not read beyond
      the box); then the sample stack ``W``, the same for every mode; then the mode axis, of size ``N_i``;
    - for ``'entries'``, one integer array ``index`` of shape ``(d,) + W``, with ``order = ()``: the entries kind
      has no jets.

    **What a shape cannot say.** How many leading axes of ``ww[i]`` are jet axes is not in its shape: ``W`` has any
    number of axes. This function reads the first ``len(order)`` axes as jet axes and the axes between them and the
    last as ``W``, and requires one ``W`` for all modes. A sample of the wrong form for the jet shape is caught when
    the modes' ``W`` then disagree or an input has too few axes; but an input jet given with ``order = ()`` (a
    forgotten jet shape), or a sample whose every mode has the same jet sizes, can read as one more axis of ``W``
    and pass; and a jet shape with one variable too many reads an axis of ``W`` as a jet axis (``(1, 1)`` given for a
    line), and passes too. The jet shape is never inferred from the sample; say it.

    This is the ragged form of a sample. Vectors packed for the uniform layout, one array ``(d,) + F_in + W + (N,)``,
    are not accepted: check before packing. The values are not looked at (an index out of range is not caught).
    Shapes are static, so the check also runs under ``jax.jit``.

    This is a checker, not a guarantee: no backend function calls it.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    sample
        The input jets ``ww``, or ``index`` for the entries kind.
    shape: Sequence[int]
        The mode sizes ``(N_0, ..., N_(d-1))`` of the tensor the sample is for.

    Raises
    ------
    ValueError
        Naming what is wrong; and for a jet shape given to the entries kind.
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    check_data
    check_weight
    affine_jet

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> np.random.seed(0)
    >>> shape = (4, 5, 6)
    >>> ww = [np.random.randn(7, N) for N in shape]             # W = (7,)
    >>> pp = [np.random.randn(7, N) for N in shape]
    >>> jets.check_sample('probe', (), ww, shape)               # good: returns nothing
    >>> jets.check_sample('apply', (2,), jets.affine_jet(ww, pp), shape)       # input jets of the line ww + s pp
    >>> jets.check_sample('entries', (), np.zeros((3, 7), dtype=int), shape)

    Inputs with too few axes for the jet shape, and modes of different jet sizes read with too few jet axes:

    >>> jets.check_sample('apply', (1, 2), ww, shape)          # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: ww[0] has shape (7, 4): too few axes for the jet shape, which has 2 jet axes, before ...
    >>> mixed = [np.random.randn(2, 7, 4), np.random.randn(3, 7, 5), np.random.randn(1, 7, 6)]  # F_in: (2,), (3,), (1,)
    >>> jets.check_sample('probe', (2,), mixed, shape)          # good: one jet axis, sizes of their own
    >>> jets.check_sample('probe', (), mixed, shape)            # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: ww[1] has the sample stack W = (3, 7) once its 0 jet axes are read; ww[0] has W = (2, 7). ...

    The trap: input jets that all have the same jet sizes, given with no jet shape, pass, their jet axis read as
    one more sample axis:

    >>> jets.check_sample('probe', (), jets.affine_jet(ww, pp), shape)        # passes: W read as (2, 7)
    """
    order = _kind_and_order(kind, order, 'check_sample')
    d = len(shape)
    if kind == 'entries':
        if isinstance(sample, (tuple, list)):
            raise ValueError('index must be one integer array of shape (d,)+W with d=%d; got a %s'
                             % (d, type(sample).__name__))
        index_shape = tuple(np.shape(sample))
        if len(index_shape) < 1 or index_shape[0] != d:
            raise ValueError('index must have shape (d,)+W with d=%d; got shape %s' % (d, index_shape))
        if not np.issubdtype(np.dtype(sample.dtype), np.integer):
            raise ValueError('index must be an integer array; got dtype %s' % (np.dtype(sample.dtype),))
        return
    _require_input_jets(sample, shape, len(order))


def _check_of_type_of_z(
        kind:         str,
        order:        typ.Tuple[int, ...],  # read
        sample:       typ.Any,              # a good sample
        value:        typ.Any,              # y or a weight: probe len=d or a mode-leading array; else one array
        stack_shape:  typ.Tuple[int, ...],  # C
        what:         str,                  # 'the data y' or 'the weight', for the messages
) -> None:
    '''The rule of a data-space object, against ``z``'s shape read from the sample and the stack: for ``'probe'`` exactly
    ``d`` arrays (a tuple, a list, or one array whose first axis is the mode, of length exactly ``d``), each of shape
    ``F + W + C + (N_i,)`` up to axes of size 1; for ``'apply'`` one array ``F + W + C``, for ``'entries'`` ``W + C``.'''
    m = len(order)
    F = tuple(n + 1 for n in order)
    C = tuple(int(n) for n in stack_shape)
    W = tuple(np.shape(sample)[1:]) if kind == 'entries' else tuple(np.shape(sample[0])[m:-1])
    describe = '(F = %s, W = %s, C = %s)' % (F, W, C)
    if kind == 'probe':
        d = len(sample)
        if isinstance(value, (tuple, list)):
            count = len(value)
        else:
            count = int(np.shape(value)[0]) if np.ndim(value) >= 1 else None
        if count != d:
            raise ValueError('%s of the probe kind has exactly d = %d arrays, one per mode: a tuple, a list, or one '
                             'array whose first axis is the mode, of length d (a mode axis of 1 is not broadcast); got '
                             '%s' % (what, d, 'shape %s' % (tuple(np.shape(value)),) if count is None or not
                                     isinstance(value, (tuple, list)) else '%d arrays' % count))
        pairs = [(' (mode %d)' % i, value[i], F + W + C + (int(np.shape(sample[i])[-1]),), 'F+W+C+(N_%d,)' % i)
                 for i in range(d)]
    else:
        if isinstance(value, (tuple, list)):
            raise ValueError('%s of the %s kind is one array, of the type of z; got a %s of length %d'
                             % (what, kind, type(value).__name__, len(value)))
        pairs = [('', value, (F + W + C) if kind == 'apply' else (W + C), 'F+W+C' if kind == 'apply' else 'W+C')]
    for where, array, expected, form in pairs:
        got = tuple(np.shape(array))
        if len(got) != len(expected) or any(a not in (1, b) for a, b in zip(got, expected)):
            raise ValueError('%s of the %s kind%s has shape %s; it is of the type of z, whose array there has shape %s '
                             '= %s %s: the same rank, each axis 1 or that size, never the stack of directions K'
                             % (what, kind, where, got, form, expected, describe))


def check_data(
        kind:         str,                      # 'probe' | 'apply' | 'entries'
        order:        typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        sample:       typ.Any,                  # ww (input jets) | index, a good sample
        y:            typ.Any,                  # Y*, of the type of z: probe len=d (or (d,)+...), elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        stack_shape:  typ.Tuple[int, ...] = (), # C, the model's stack of base points; () for an unstacked model
) -> None:
    """Check the form and the shapes of the data ``y`` against a sample and the model's stack; raise a ``ValueError``
    if wrong.

    The data are of the type of ``z``, on the full box ``F = (n_1 + 1, ..., n_m + 1)`` whatever the input jet sizes,
    and with the model's stack ``C``: for ``'probe'`` ``d`` arrays (a tuple, a list, or one array whose first axis is
    the mode, of length exactly ``d``), the ``i``-th of shape ``F + W + C + (N_i,)``; for ``'apply'`` one array of shape
    ``F + W + C``; for ``'entries'`` one array of shape ``W + C``. **Each axis has that size or 1**, a size of 1 meaning
    the same datum along the axis: data shared by every model of a stack have size-1 ``C`` axes, data constant over
    the samples a size-1 ``W`` axis. The rank is exact, so data of a stacked model checked without its
    ``stack_shape`` fail by rank. Data measured only up to a total degree are given in the box, with weight zero on the
    slots not measured. ``W``, ``d`` and the mode sizes are read from the sample, which is taken to be good: call
    :py:func:`check_sample` on it first. Probe data packed for the uniform layout pass only where every ``N_i`` is the
    padded ``N`` (the checkers take the unpadded form).

    **What a shape cannot say** (the trap of :py:func:`check_sample`). ``W`` is read off the sample with the jet
    shape given, which is never inferred. With a forgotten jet shape (``order = ()``), input jets read as one more
    axis of ``W``, and data whose leading axes happen to have the input jet sizes pass as data with no jets (the
    last example). Axes given in another order pass when their sizes coincide. Say the jet shape.

    This is a checker, not a guarantee: no backend function calls it (the engine checks the same rule on entry).

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    sample
        A good sample for ``kind`` and ``order`` (see :py:func:`check_sample`).
    y: tuple of NDArray or NDArray
        The data, of the type of ``z``.
    stack_shape: tuple of int
        The model's stack of base points ``C``; ``()`` (the default) for an unstacked model.

    Raises
    ------
    ValueError
        Naming what is wrong; and for a jet shape given to the entries kind.
    TypeError
        If ``order`` is not a tuple of integers.

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
    >>> line = jets.affine_jet(ww, pp)                          # F_in = (2,)
    >>> jets.check_data('apply', (2,), line, np.zeros((3, 7)))                         # F + W: good
    >>> jets.check_data('probe', (2,), line, [np.zeros((3, 7, N)) for N in (4, 5, 6)])
    >>> jets.check_data('apply', (2,), line, np.zeros((3, 7, 2)), stack_shape=(2,))    # F + W + C, C = (2,)
    >>> jets.check_data('apply', (2,), line, np.zeros((3, 1, 1)), stack_shape=(2,))    # one datum per slot: good

    Data on the input's jet sizes, not on the full box:

    >>> jets.check_data('apply', (2,), line, np.zeros((2, 7)))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: the data y of the apply kind has shape (2, 7); it is of the type of z, ...

    Data of a stacked model, checked without its stack, fail by rank:

    >>> jets.check_data('apply', (2,), line, np.zeros((3, 7, 2)))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: the data y of the apply kind has shape (3, 7, 2); it is of the type of z, ...

    The trap: the jets of the line at ``(1,)``, checked with the jet shape forgotten, pass as data with no jets
    (``W`` read as ``(2, 7)``):

    >>> jets.check_data('apply', (), line, np.zeros((2, 7)))   # passes: it should have been order=(1,)
    """
    order = _kind_and_order(kind, order, 'check_data')
    _check_of_type_of_z(kind, order, sample, y, stack_shape, 'the data y')


def check_weight(
        kind:         str,                      # 'probe' | 'apply' | 'entries'
        order:        typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        sample:       typ.Any,                  # ww (input jets) | index, a good sample
        weight:       typ.Optional[typ.Any],    # omega, of the type of z (each axis 1 or z's size), or None
        stack_shape:  typ.Tuple[int, ...] = (), # C, the model's stack of base points; () for an unstacked model
) -> None:
    """Check the form and the shapes of a weight against a sample and the model's stack; raise a ``ValueError`` if
    wrong.

    The weight ``omega`` is of the type of ``z``, by the rule of :py:func:`check_data`: for ``'probe'`` ``d`` arrays (a
    tuple, a list, or one array whose first axis is the mode, of length exactly ``d``), each of the shape of ``z``'s
    array, ``F + W + C + (N_i,)``, up to axes of size 1; for ``'apply'`` ``F + W + C``; for ``'entries'`` ``W + C``. A
    size of 1 is the same weight along that axis: a weight per slot has size 1 on ``W``, ``C`` and the edge, a weight
    per sample size 1 on the jet axes. :py:func:`blocks_to_weight` makes one from a value per block (per mode and slot,
    per slot, or one number), the form older code used, which is refused here unless it coincides with the full form
    (no samples axes, no stack). ``None`` is good and means ones. The weight never carries the stack of directions
    ``K``.

    A shape check cannot see what the axes mean: axes given in another order pass when their sizes coincide. And the
    jet shape is taken as given, never inferred (the trap of :py:func:`check_sample`): with a forgotten jet shape the
    weight is checked against the wrong ``z``, and caught only if it then has the wrong shape.

    This is a checker, not a guarantee: no backend function calls it (the engine checks the same rule on entry).

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    sample
        A good sample for ``kind`` and ``order`` (see :py:func:`check_sample`).
    weight: tuple of NDArray or NDArray, or None
        The weight.
    stack_shape: tuple of int
        The model's stack of base points ``C``; ``()`` (the default) for an unstacked model.

    Raises
    ------
    ValueError
        Naming the shape found and the shape expected; and for a jet shape given to the entries kind.
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    check_sample
    check_data
    blocks_to_weight

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> np.random.seed(0)
    >>> ww = [np.random.randn(7, N) for N in (4, 5, 6)]         # W = (7,)
    >>> line = jets.affine_jet(ww, [np.random.randn(7, N) for N in (4, 5, 6)])
    >>> y = [np.zeros((3, 7, N)) for N in (4, 5, 6)]            # probe at (2,): F + W + (N_i,)
    >>> jets.check_weight('probe', (2,), line, [np.ones((3, 7, N)) for N in (4, 5, 6)])   # per entry: good
    >>> jets.check_weight('probe', (2,), line, np.ones((3, 1, 7, 1)))                     # per sample: good
    >>> jets.check_weight('apply', (2,), line, None)                                     # None is ones

    A weight per mode and slot, from its blocks:

    >>> per_mode_and_slot = np.array([[1.0, 0.5, 0.25]] * 3)
    >>> jets.check_weight('probe', (2,), line, jets.blocks_to_weight('probe', (2,), per_mode_and_slot, y))

    The blocks themselves are not of the type of ``z``:

    >>> jets.check_weight('probe', (2,), line, per_mode_and_slot)    # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: the weight of the probe kind (mode 0) has shape (3,); it is of the type of z, ...
    """
    order = _kind_and_order(kind, order, 'check_weight')
    if weight is None:
        return
    _check_of_type_of_z(kind, order, sample, weight, stack_shape, 'the weight')
