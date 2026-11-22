# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Preparing the data of a fit: block norms, the equalizing weight, and two rescalings.

A fit takes the data as they are given and, with no weight, weighs every Taylor coefficient residual by one.
Badly scaled data make a badly conditioned fit. The functions here are what a user calls **before** fitting, when
the batch is made; no fitting function has a keyword for any of them. They are pure, and work on the ragged form
of a sample and its data (for the probe kind a tuple or a list, one array per mode; not arrays packed for the
uniform layout). Each chooses numpy or jax once, over all the arrays it is given: if any is a jax array, every
array it computes is. The jet shape ``order = (n_1, ..., n_m)`` is a tuple (an integer raises), the samples are
input jets (``F_in_i + W + (Ni,)``, ``m`` leading jet axes of sizes of their own), and the data have the full jet
sizes ``F = (n_1 + 1, ..., n_m + 1)``; the entries kind has no jets.

Three kinds of preparation, in the order in which they are usually applied:

- **Rescale the coordinates** (:py:func:`rescale_coordinates`). Coordinate ``j`` of mode ``i`` of every input
  vector is multiplied by ``scales[i][j]``. This **changes the unknowns**: the tensor fitted to the rescaled
  sample is the original one seen in the new coordinates. :py:func:`t3_unscale_coordinates` (and
  :py:func:`ut3_unscale_coordinates` on the uniform layout) gives the fitted tensor back in the original
  coordinates. For the probe kind the data of a mode live in that mode's coordinates, so they are divided by the
  scales, which also changes the loss. :py:func:`check_coordinate_scales` checks the form of the scales; the
  functions that take them do not.
- **Rescale the direction** (:py:func:`rescale_direction`). A new jet variable ``s_j = length_j s'_j``: slot ``k`` of
  every input jet and of the data is multiplied by ``length_1^k_1 ... length_m^k_m`` (for the line ``w + s p``,
  the direction times the length and slot ``t`` of the data times ``length**t``). The tensor that fits is the
  same; only the loss changes (the slots are weighed differently).
- **Equalize** (:py:func:`data_block_norms`, then :py:func:`equalizing_weight`). The norm of each block of the
  data (one block per mode and slot for the probe kind, per slot for the others), and a weight that is the
  filtered inverse of those norms, so that every block counts about equally in the misfit. Only the loss changes.

The block norms are also the diagnostic: norms that fall or grow by a constant factor from one order to the next
say that the directions are too short or too long. Rescale the lengths first and equalize after: the floor of the
equalizing weight is relative to the largest block, so it can only tell a block that is zero from a block that is
small if the blocks are not spread over many decades for a reason of scale.

Examples
--------
A tensor seen through badly scaled coordinates, sampled by probes along lines, at the jet shape ``(2,)``, with
directions that are too short.

>>> import numpy as np
>>> import t3toolbox.backend.data_preparation as prep
>>> import t3toolbox.backend.probing as probing
>>> from t3toolbox.backend.jets import affine_jet
>>> np.random.seed(0)
>>> shape, order, W = (4, 5, 6), (2,), 8
>>> T = np.random.randn(*shape)                                          # a dense tensor, for the data
>>> ww = [np.random.randn(W, N) for N in shape]
>>> pp = [1e-2 * np.random.randn(W, N) for N in shape]                   # directions 100 times too short
>>> line = affine_jet(ww, pp)                                            # the input jets: F_in = (2,)
>>> y = probing.dense_probe(T, line, order)                              # len=d, elm_shape=F+W+(Ni,)

The norm of each block, one row per mode and one column per slot, falls by about 100 per slot:

>>> norms = prep.data_block_norms('probe', order, y)
>>> print(norms.shape)                                                   # the shape of a probe weight, (d,) + F
(3, 3)
>>> print(bool(np.all(norms[:, 0] > 20 * norms[:, 1])), bool(np.all(norms[:, 1] > 20 * norms[:, 2])))
True True

Lengthen the directions. The data change with them, slot ``t`` by ``length**t``, so they are still the data of
the same tensor:

>>> line2, y2 = prep.rescale_direction('probe', order, line, y, 100.0)
>>> print(bool(np.allclose(line2[1][1], 100.0 * pp[1])), bool(np.allclose(y2[0], probing.dense_probe(T, line2, order)[0])))
True True

The equalizing weight is the inverse of the block norms, so every block of the weighted data has norm one:

>>> weight = prep.equalizing_weight(prep.data_block_norms('probe', order, y2))
>>> print(weight.shape)
(3, 3)
>>> print(bool(np.allclose(weight * prep.data_block_norms('probe', order, y2), 1.0)))
True

Rescale the coordinates: the inputs are multiplied by the scales, and the probe data divided by them.

>>> scales = [10.0 ** np.random.randn(N) for N in shape]                 # len=d, elm_shape=(Ni,)
>>> line3, y3 = prep.rescale_coordinates('probe', order, line2, y2, scales)
>>> print(bool(np.allclose(line3[1], line2[1] * scales[1])), bool(np.allclose(y3[1], y2[1] / scales[1])))
True True

A tensor fitted to ``line3`` and ``y3`` lives in the rescaled coordinates; :py:func:`t3_unscale_coordinates`
gives it back in the original ones (its docstring has the example).
"""
import typing as typ

import numpy as np

import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.jets import _kind_and_order, _lowest_precision_dtype, _require_kind
from t3toolbox.backend.common import *

__all__ = [
    'data_block_norms',
    'equalizing_weight',
    'rescale_direction',
    'rescale_coordinates',
    't3_unscale_coordinates',
    'ut3_unscale_coordinates',
    'check_coordinate_scales',
]


def _scaled(
        array_Xo:  typ.Any,             # shape=X+(Ni,)
        scale_o:   typ.Any,             # shape=(Ni,)
        xnp,                            # numpy or jax.numpy, chosen once over all the inputs of the call
        divide:    bool = False,        # True: divided by the scale; False: multiplied
) -> NDArray:                           # shape=X+(Ni,), the last axis scaled entry by entry
    '''The last axis of an array multiplied (or divided) by one scale per coordinate, in the lower floating
    precision of the two.'''
    dtype = _lowest_precision_dtype(xnp, array_Xo, scale_o)
    array_Xo, scale_o = xnp.asarray(array_Xo, dtype=dtype), xnp.asarray(scale_o, dtype=dtype)
    return array_Xo / scale_o if divide else array_Xo * scale_o


def _block_norms_F(
        y_FX:   typ.Any,                # shape=F+X
        m:      int,                    # the number of jet axes
        xnp,
) -> NDArray:                           # shape=F: the norm of each slot of the jet axes
    y_FX = xnp.asarray(y_FX)
    axes = tuple(range(m, y_FX.ndim))
    return xnp.asarray(xnp.sqrt(xnp.sum(y_FX * y_FX, axis=axes)))      # an array with no jets too, not a numpy scalar


def data_block_norms(
        kind:   str,                       # 'probe' | 'apply' | 'entries'
        order:  typ.Tuple[int, ...],       # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        y:      typ.Any,                   # the data: probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W
) -> NDArray:                              # the shape of the weight: probe (d,)+F; apply F; entries ()
    """The norm of each block of the data, in the shape of the weight.

    A block is what one entry of the weight multiplies: for ``'probe'`` the data of one mode at one slot of the box
    ``F`` (so the result has shape ``(d,) + F``), for ``'apply'`` the data at one slot (shape ``F``), and for
    ``'entries'``, which has no jets, all the data (shape ``()``). The norm is the root of the sum of squares over
    the samples (and, for probe, over the mode's coordinates). With no jets the result has shape ``(d,)`` for probe
    and is a 0-dimensional array for the others. The data are of the ragged type (for ``'probe'`` a tuple or a
    list, one array per mode), not an array packed for the uniform layout.

    The norms feed :py:func:`equalizing_weight`, and they are the diagnostic of the length of the directions
    (see the module's docstring).

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    y: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        The data. probe: ``len=d``, ``elm_shape=F+W+(Ni,)``. apply: ``shape=F+W``. entries: ``shape=W``.

    Returns
    -------
    NDArray
        The block norms. probe: ``shape=(d,)+F``. apply: ``shape=F``. entries: ``shape=()``.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three kinds, or the entries kind is given a jet shape but ``()``.
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    equalizing_weight

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.data_preparation as prep
    >>> y = np.array([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])          # apply at (2,): F + W = (3, 2)
    >>> print(bool(np.allclose(prep.data_block_norms('apply', (2,), y), [5.0, 0.5, 0.0])))
    True
    >>> np.random.seed(0)
    >>> y = [np.random.randn(3, 7, N) for N in (4, 5)]              # probe: d = 2, F = (3,), W = (7,)
    >>> norms = prep.data_block_norms('probe', (2,), y)
    >>> print(norms.shape)
    (2, 3)
    >>> print(bool(np.isclose(norms[1, 2], np.linalg.norm(y[1][2]))))
    True
    >>> print(prep.data_block_norms('apply', (1, 2), np.ones((2, 3, 4))).shape)    # two jet variables: F
    (2, 3)
    >>> print(prep.data_block_norms('probe', (), [y_i[0] for y_i in y]).shape)     # no jets: (d,)
    (2,)
    """
    order = _kind_and_order(kind, order, 'data_block_norms')
    xnp, _, _ = get_backend(False, tree_contains_jax(y))                    # once, over all the inputs
    if kind == 'probe':
        return xnp.stack([_block_norms_F(y_FWo, len(order), xnp) for y_FWo in y])
    return _block_norms_F(y, len(order), xnp)


def equalizing_weight(
        block_norms:  NDArray,             # from data_block_norms: probe (d,)+F; apply F; entries ()
        floor:        float = 1e-6,        # relative to the largest block
) -> NDArray:                              # same shape: 1 / max(x, floor * max(x))
    """The equalizing weight: the filtered inverse of the block norms, ``1 / max(x, floor * max(x))``.

    With this weight every block of the data whose norm is above the floor has weighted norm one, so every
    mode and order counts about equally in the misfit. The floor is relative to the largest block. A block
    below it, a block that is exactly zero included, gets the floor weight ``1 / (floor * max(x))`` and no
    special case: to leave a block out of the fit, set its entry of the returned weight to zero.

    Rescale the directions and the coordinates first, and equalize after: norms that span many decades because
    of a bad length scale would otherwise reach the floor without being zero.

    If every block is zero the floor is zero as well and the result is ``inf`` in every entry, with no error
    and no warning, on numpy and on jax alike. The result is an array of the module of the input, also for a
    0-dimensional input (the block norm of the apply kind with no jets, or of the entries kind).

    Parameters
    ----------
    block_norms: NDArray
        The block norms, from :py:func:`data_block_norms`.
    floor: float
        The smallest norm that is inverted as it is, as a fraction of the largest block's norm.

    Returns
    -------
    NDArray
        The weight, with the shape of ``block_norms``.

    See Also
    --------
    data_block_norms

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.data_preparation as prep
    >>> norms = np.array([5.0, 0.5, 0.0])
    >>> weight = prep.equalizing_weight(norms)                      # the zero block gets 1 / (1e-6 * 5)
    >>> print(bool(np.allclose(weight, [0.2, 2.0, 2e5])))
    True
    >>> print(prep.equalizing_weight(norms, floor=0.5).tolist())    # floor 2.5: the block 0.5 is below it too
    [0.2, 0.4, 0.4]
    """
    use_jax = tree_contains_jax(block_norms)
    xnp, _, _ = get_backend(False, use_jax)
    norms = xnp.asarray(block_norms)
    filtered = xnp.maximum(norms, floor * xnp.max(norms))
    if use_jax:
        return 1.0 / filtered
    with np.errstate(divide='ignore'):                      # all blocks zero: inf, silently, as on jax
        return np.asarray(1.0 / filtered)                   # an array for a 0-dimensional input too


def _slot_factors(
        lengths:  typ.Sequence[typ.Any],    # len=m: one length per jet variable (numbers or 0-d arrays, maybe traced)
        sizes:    typ.Tuple[int, ...],      # the jet sizes of the array to scale, len=m
        dtype:    typ.Any,
        xnp,
) -> NDArray:                               # shape=sizes: slot k holds length_1**k_1 ... length_m**k_m
    m = len(sizes)
    factors = xnp.ones(sizes, dtype=dtype)
    for j, size in enumerate(sizes):
        along = xnp.asarray(lengths[j], dtype=dtype) ** xnp.arange(size, dtype=dtype)
        factors = factors * along.reshape((1,) * j + (size,) + (1,) * (m - j - 1))
    return factors


def _scale_jet(
        jet_PX:   typ.Any,                  # shape=P+X: m leading jet axes of sizes of its own
        lengths:  typ.Sequence[typ.Any],    # len=m
        xnp,
) -> NDArray:                               # shape=P+X: slot k multiplied by length_1**k_1 ... length_m**k_m
    jet_PX = xnp.asarray(jet_PX)
    m = len(lengths)
    factors_P = _slot_factors(lengths, tuple(jet_PX.shape[:m]), _lowest_precision_dtype(xnp, jet_PX), xnp)
    return factors_P.reshape(factors_P.shape + (1,) * (jet_PX.ndim - m)) * jet_PX


def rescale_direction(
        kind:    str,                      # 'probe' | 'apply'
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m), at least one jet variable
        sample:  typ.Any,                  # ww, the input jets: len=d, elm_shape=F_in_i+W+(Ni,)
        y:       typ.Any,                  # of the type of z: probe len=d, elm_shape=F+W+(Ni,); apply F+W
        length:  typ.Any,                  # a number or a 0-d array, or one per jet variable (len=m): slot k of the
                                           #   input jets and of y times length_1**k_1 ... length_m**k_m   (loss-only)
) -> typ.Tuple[
        typ.Any,                           # the sample: the input jets rescaled, a tuple
        typ.Any,                           # y, same type and shapes, rescaled the same way
]:
    """Rescale the jet variables of a sample by a length, and the data with them. Changes the loss only.

    A jet is the Taylor expansion of the output in the jet variables ``s_1 ... s_m``. A new variable
    ``s_j = length_j s'_j`` multiplies the coefficient of ``s^k`` by ``length_1**k_1 ... length_m**k_m``, in the input
    jets and in the outputs alike. So slot ``k`` of every input jet and of the data is multiplied by that factor, and
    the result is the data of the same tensor at the new sample: the tensor that fits is unchanged, and nothing has
    to be undone afterwards. What changes is the loss, because the slots are weighed differently. For the line
    ``w + s p`` (``affine_jet(ww, pp)``) this is the direction times ``length`` and slot ``t`` of the data times
    ``length**t``.

    ``length`` is one number (or a 0-dimensional array), the same for every jet variable, or one per jet variable:
    a sequence or a 1-dimensional array of ``m`` lengths. A length may be a jax scalar computed from the data, and
    it may be traced under ``jax.jit``. If the inputs, the data or the length are on jax, the inputs and the data
    returned are. The sample and the data are of the ragged type (for ``'probe'`` a tuple or a list, one array per
    mode), not arrays packed for the uniform layout. The entries kind has no jets, so no direction, and is not
    accepted.

    Parameters
    ----------
    kind: str
        ``'probe'`` or ``'apply'``.
    order: tuple of int
        The jet shape, with at least one jet variable (with none a sample has no direction).
    sample
        The input jets ``ww``.
    y: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        The data, in Taylor coefficients or in derivatives (slot ``k`` scales the same way in both).
    length: float, 0-dimensional NDArray, or ``m`` of them
        The factor on each jet variable.

    Returns
    -------
    sample
        The input jets for the new variables: a tuple, one array per mode.
    y: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        The data for the new sample.

    Raises
    ------
    ValueError
        If ``kind`` is not ``'probe'`` or ``'apply'``, the jet shape has no jet variable, or ``length`` has neither
        one value nor ``m`` of them.
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    rescale_coordinates
    data_block_norms

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.data_preparation as prep
    >>> import t3toolbox.backend.apply as apply
    >>> from t3toolbox.backend.jets import affine_jet
    >>> np.random.seed(0)
    >>> shape, order = (4, 5, 6), (2,)
    >>> T = np.random.randn(*shape)
    >>> ww = [np.random.randn(7, N) for N in shape]
    >>> pp = [np.random.randn(7, N) for N in shape]
    >>> line = affine_jet(ww, pp)                                   # the input jets of ww + s pp
    >>> y = apply.dense_apply(T, line, order)                       # F + W = (3, 7)
    >>> line2, y2 = prep.rescale_direction('apply', order, line, y, 10.0)
    >>> print(bool(np.allclose(line2[0][1], 10.0 * pp[0])), bool(np.allclose(y2[2], 100.0 * y[2])))
    True True
    >>> print(bool(np.allclose(y2, apply.dense_apply(T, line2, order))))    # the data of the same tensor
    True

    Two jet variables, one length each: slot ``(k_1, k_2)`` times ``2**k_1 * 10**k_2``.

    >>> qq = [np.random.randn(7, N) for N in shape]
    >>> plane = affine_jet(ww, pp, qq)
    >>> y = apply.dense_apply(T, plane, (1, 2))                     # F + W = (2, 3, 7)
    >>> plane2, y2 = prep.rescale_direction('apply', (1, 2), plane, y, (2.0, 10.0))
    >>> print(bool(np.allclose(y2[1, 2], 2.0 * 100.0 * y[1, 2])), bool(np.allclose(y2, apply.dense_apply(T, plane2, (1, 2)))))
    True True
    """
    order = _kind_and_order(kind, order, 'rescale_direction')
    if kind == 'entries':
        raise ValueError("rescale_direction is for 'probe' and 'apply': the entries kind has no jets, so no direction")
    m = len(order)
    if m == 0:
        raise ValueError('rescale_direction needs a jet shape with at least one jet variable: with no jets a sample '
                         'has no direction')
    if np.ndim(length) == 0:
        lengths = [length] * m
    elif np.ndim(length) == 1 and len(length) == m:
        lengths = [length[j] for j in range(m)]
    else:
        raise ValueError('rescale_direction: length is one number, or one per jet variable (%d); got shape %s'
                         % (m, np.shape(length)))
    xnp, _, _ = get_backend(False, tree_contains_jax((sample, y, length)))  # once, over all the inputs
    scaled_sample = tuple(_scale_jet(w_PWo, lengths, xnp) for w_PWo in sample)
    if kind == 'probe':
        return scaled_sample, tuple(_scale_jet(y_FWo, lengths, xnp) for y_FWo in y)
    return scaled_sample, _scale_jet(y, lengths, xnp)


def rescale_coordinates(
        kind:    str,                      # 'probe' | 'apply'
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        sample:  typ.Any,                  # ww, the input jets: len=d, elm_shape=F_in_i+W+(Ni,)
        y:       typ.Any,                  # of the type of z: probe len=d, elm_shape=F+W+(Ni,); apply F+W
        scales:  typ.Sequence[NDArray],    # len=d, elm_shape=(Ni,): coordinate j of mode i times scales[i][j]
) -> typ.Tuple[
        typ.Any,                           # the sample: every slot of every input jet times its mode's scales
        typ.Any,                           # y: for probe, mode i's data divided by scales[i]; for apply, as given
]:
    """Rescale the coordinates of each mode: the inputs times ``scales``, the probe data divided by them.

    Coordinate ``j`` of mode ``i`` of every slot of every input jet is multiplied by ``scales[i][j]`` (the map is
    linear in the input, so the input jet of the rescaled inputs is the input jet rescaled, slot by slot). This
    **changes the unknowns**. A tensor ``x'`` fitted to the rescaled sample is the wanted tensor ``x`` seen in the
    new coordinates: ``x'`` applied to ``scales * w`` is ``x`` applied to ``w``. After the fit,
    :py:func:`t3_unscale_coordinates` (or :py:func:`ut3_unscale_coordinates`, or the ``unscale_coordinates`` method
    of the tensor) turns ``x'`` into ``x``, with the same ``scales``.

    The data follow. For ``'apply'`` the output is a number per sample and does not change. For ``'probe'`` the
    data of mode ``i`` are a vector in mode ``i``'s coordinates, and they are divided by ``scales[i]``. So for
    the probe kind the rescaling is not neutral for the loss: the components of each mode's residual are
    reweighed.

    The entries kind has no coordinates to rescale (its inputs are grid points), so it is not accepted.

    The sample and the data are of the ragged type (for ``'probe'`` a tuple or a list, one array per mode), not
    arrays packed for the uniform layout. **``scales`` is not checked here.** A list that is too short silently
    drops the modes it does not cover (the sample returned has fewer modes), and a scale of length 1 is
    broadcast over its mode; a zero scale divides the probe data by zero. :py:func:`check_coordinate_scales`
    checks the form; call it first if you want the error.

    Parameters
    ----------
    kind: str
        ``'probe'`` or ``'apply'``.
    order: tuple of int
        The jet shape (``()`` for no jets).
    sample
        The input jets ``ww``.
    y: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        The data. probe: ``len=d``, ``elm_shape=F+W+(Ni,)``. apply: ``shape=F+W``.
    scales: Sequence[NDArray]
        One positive scale per coordinate: ``len=d``, ``elm_shape=(Ni,)``.

    Returns
    -------
    sample
        The rescaled input jets, a tuple.
    y: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        The data for the rescaled sample.

    Raises
    ------
    ValueError
        If ``kind`` is not ``'probe'`` or ``'apply'``.
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    check_coordinate_scales
    t3_unscale_coordinates
    ut3_unscale_coordinates
    rescale_direction

    Examples
    --------
    The data of a tensor ``x`` in the original coordinates are, after rescaling, the data of the tensor whose
    Tucker cores are those of ``x`` divided by the scales, at the rescaled sample:

    >>> import numpy as np
    >>> import t3toolbox.backend.data_preparation as prep
    >>> import t3toolbox.backend.probing as probing
    >>> import t3toolbox.backend.t3_constructors as t3_constructors
    >>> import t3toolbox.backend.t3_conversions as t3_conversions
    >>> np.random.seed(0)
    >>> shape = (4, 5, 6)
    >>> x = t3_constructors.t3_corewise_randn(shape, (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(7, N) for N in shape]
    >>> y = probing.dense_probe(t3_conversions.t3_to_dense(x), ww)
    >>> scales = [10.0 ** np.random.randn(N) for N in shape]
    >>> ww2, y2 = prep.rescale_coordinates('probe', (), ww, y, scales)
    >>> x2 = (tuple(B / s for B, s in zip(x[0], scales)), x[1])     # x in the new coordinates
    >>> y_of_x2 = probing.dense_probe(t3_conversions.t3_to_dense(x2), ww2)
    >>> print(all(bool(np.allclose(a, b)) for a, b in zip(y2, y_of_x2)))
    True
    >>> x_back = prep.t3_unscale_coordinates(x2, scales)            # and back to the original coordinates
    >>> print(bool(np.allclose(x_back[0][1], x[0][1])))
    True
    """
    _kind_and_order(kind, order, 'rescale_coordinates')
    if kind == 'entries':
        raise ValueError("rescale_coordinates is for 'probe' and 'apply': the entries kind takes grid points, "
                         "which have no coordinates to rescale")
    xnp, _, _ = get_backend(False, tree_contains_jax((sample, y, scales)))  # once, over all the inputs
    scaled_sample = tuple(_scaled(w_PWo, s_o, xnp) for w_PWo, s_o in zip(sample, scales))
    if kind == 'probe':
        return scaled_sample, tuple(_scaled(y_FWo, s_o, xnp, divide=True) for y_FWo, s_o in zip(y, scales))
    return scaled_sample, y


def t3_unscale_coordinates(
        x:       typ.Tuple[
                     typ.Sequence[NDArray],     # tucker_cores, len=d, elm_shape=C+(ni,Ni)
                     typ.Sequence[NDArray],     # tt_cores,     len=d, elm_shape=C+(ri,ni,r(i+1))
                 ],                             # fitted in rescaled coordinates
        scales:  typ.Sequence[NDArray],         # len=d, elm_shape=(Ni,): as given to rescale_coordinates
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                # tucker_cores, same shapes: column j of core i times scales[i][j]
        typ.Tuple[NDArray, ...],                # tt_cores, as given
]:
    """A Tucker tensor train fitted in rescaled coordinates, in the original coordinates.

    The inverse of :py:func:`rescale_coordinates` on the fitted tensor. If ``x'`` was fitted to a sample whose
    coordinates were multiplied by ``scales``, the tensor in the original coordinates is
    ``x[j_0, ..., j_(d-1)] = x'[j_0, ..., j_(d-1)] * scales[0][j_0] * ... * scales[d-1][j_(d-1)]``: applied to
    the original inputs it gives what ``x'`` gives at the rescaled ones. Only the Tucker cores change: column
    ``j`` of core ``i`` is multiplied by ``scales[i][j]``. The ranks stay.

    **``scales`` is not checked here.** A list that is too short silently drops the Tucker cores it does not
    cover, and a scale of length 1 is broadcast over its mode. :py:func:`check_coordinate_scales` checks the
    form (the ``unscale_coordinates`` method of the tensor calls it).

    Parameters
    ----------
    x: tuple
        ``(tucker_cores, tt_cores)``: the tensor fitted in the rescaled coordinates. ``tucker_cores``:
        ``len=d``, ``elm_shape=C+(ni,Ni)``.
    scales: Sequence[NDArray]
        The scales that were given to :py:func:`rescale_coordinates`: ``len=d``, ``elm_shape=(Ni,)``.

    Returns
    -------
    tuple
        ``(tucker_cores, tt_cores)``: the same tensor as a function of the original coordinates.

    See Also
    --------
    check_coordinate_scales
    rescale_coordinates
    ut3_unscale_coordinates

    Examples
    --------
    The tensor in the original coordinates, applied to the original inputs, gives what the fitted tensor gives
    at the rescaled inputs:

    >>> import numpy as np
    >>> import t3toolbox.backend.data_preparation as prep
    >>> import t3toolbox.backend.apply as apply
    >>> import t3toolbox.backend.t3_constructors as t3_constructors
    >>> import t3toolbox.backend.t3_conversions as t3_conversions
    >>> np.random.seed(0)
    >>> shape = (4, 5, 6)
    >>> x_fitted = t3_constructors.t3_corewise_randn(shape, (2, 3, 2), (1, 2, 2, 1))
    >>> scales = [10.0 ** np.random.randn(N) for N in shape]
    >>> ww = [np.random.randn(7, N) for N in shape]                 # inputs in the original coordinates
    >>> ww_rescaled = [w * s for w, s in zip(ww, scales)]
    >>> x = prep.t3_unscale_coordinates(x_fitted, scales)
    >>> at_original = apply.dense_apply(t3_conversions.t3_to_dense(x), ww)
    >>> at_rescaled = apply.dense_apply(t3_conversions.t3_to_dense(x_fitted), ww_rescaled)
    >>> print(bool(np.allclose(at_original, at_rescaled)))
    True
    >>> print([B.shape for B in x[0]] == [B.shape for B in x_fitted[0]])    # the ranks stay
    True
    """
    tucker_cores, tt_cores = x
    xnp, _, _ = get_backend(False, tree_contains_jax((tucker_cores, scales)))   # once, over the arrays that meet
    return tuple(_scaled(B_Cio, s_o, xnp) for B_Cio, s_o in zip(tucker_cores, scales)), tuple(tt_cores)


def ut3_unscale_coordinates(
        x:       typ.Tuple[
                     NDArray,                           # tucker_supercore, shape=(d,)+C+(n,N)
                     NDArray,                           # tt_supercore,     shape=(d,)+C+(r,n,r)
                     typ.Tuple[int, ...],               # shape, len=d: the mode sizes Ni
                     typ.Tuple[NDArray, NDArray],       # (tucker_edge_mask, tt_edge_mask), host bool
                 ],                                     # fitted in rescaled coordinates
        scales:  typ.Union[typ.Sequence[NDArray], NDArray],     # len=d, elm_shape=(Ni,); or packed, shape=(d,N)
) -> typ.Tuple[
        NDArray,                                        # tucker_supercore, shape=(d,)+C+(n,N): the last axis scaled
        NDArray,                                        # tt_supercore, as given
        typ.Tuple[int, ...],                            # shape, as given
        typ.Tuple[NDArray, NDArray],                    # masks, as given
]:
    """A uniform Tucker tensor train fitted in rescaled coordinates, in the original coordinates.

    The uniform twin of :py:func:`t3_unscale_coordinates`: the last axis of the Tucker supercore is multiplied by
    the scales of its mode, at all modes at once. ``scales`` is the ragged sequence that was given to
    :py:func:`rescale_coordinates`, or the same packed to ``(d, N)``.

    The ranks do not change, so the masks, the shape and the TT supercore are returned as given.

    What holds in the padding. Beyond each mode's size ``Ni`` the scales are set to zero (from ``shape``,
    whatever a packed ``scales`` holds there), so the padding of the mode axis comes out zero whatever the
    supercore held, for ragged and for packed scales alike. The padding of the rank axis is multiplied like the
    rest: zero stays zero and garbage stays garbage, as the uniform contract allows. The real entries do not
    depend on any padding, of the supercore or of the packed scales.

    **``scales`` is not checked here.** Ragged scales of a wrong count or length fail when they are packed, or
    are packed wrongly; :py:func:`check_coordinate_scales` checks the ragged form (the ``unscale_coordinates``
    method of the tensor calls it).

    Parameters
    ----------
    x: tuple
        The ``.data`` of the uniform tensor fitted in the rescaled coordinates:
        ``(tucker_supercore, tt_supercore, shape, masks)``.
    scales: Sequence[NDArray] or NDArray
        ``len=d``, ``elm_shape=(Ni,)``; or one packed array of shape ``(d, N)``.

    Returns
    -------
    tuple
        ``(tucker_supercore, tt_supercore, shape, masks)``: the same tensor as a function of the original
        coordinates.

    See Also
    --------
    check_coordinate_scales
    t3_unscale_coordinates
    rescale_coordinates

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.data_preparation as prep
    >>> import t3toolbox.backend.t3_constructors as t3_constructors
    >>> import t3toolbox.backend.t3_conversions as t3_conversions
    >>> import t3toolbox.backend.ut3_conversions as ut3_conversions
    >>> np.random.seed(0)
    >>> shape = (4, 5, 6)
    >>> x = t3_constructors.t3_corewise_randn(shape, (2, 3, 2), (1, 2, 2, 1))
    >>> scales = [10.0 ** np.random.randn(N) for N in shape]
    >>> ux = ut3_conversions.t3_to_ut3(x)                           # the uniform .data of the same tensor
    >>> ux_original = prep.ut3_unscale_coordinates(ux, scales)
    >>> print(ux_original[0].shape, ux_original[2])                 # (d, n, N), and the mode sizes
    (3, 3, 6) (4, 5, 6)
    >>> x_original = prep.t3_unscale_coordinates(x, scales)         # the ragged function, for comparison
    >>> dense = ut3_conversions.ut3_to_dense(ux_original)
    >>> print(bool(np.allclose(dense, t3_conversions.t3_to_dense(x_original))))
    True
    """
    tucker_supercore, tt_supercore, shape, masks = x
    xnp, _, _ = get_backend(True, tree_contains_jax((tucker_supercore, scales)))
    d, N = tucker_supercore.shape[0], tucker_supercore.shape[-1]
    scales_dN = ut3_operations.pack_if_ragged(scales, N)
    dtype = _lowest_precision_dtype(xnp, tucker_supercore, scales_dN)
    mode_mask_dN = np.arange(N)[None, :] < np.asarray(shape)[:, None]                   # host: the real coordinates
    scales_dN = xnp.asarray(scales_dN, dtype=dtype) * mode_mask_dN                      # zero beyond each Ni
    scales_dCnN = scales_dN.reshape((d,) + (1,) * (tucker_supercore.ndim - 2) + (N,))
    return xnp.asarray(tucker_supercore, dtype=dtype) * scales_dCnN, tt_supercore, shape, masks


def check_coordinate_scales(
        scales:  typ.Sequence[NDArray],    # expected: len=d, elm_shape=(Ni,)
        shape:   typ.Sequence[int],        # the mode sizes N_i, len=d
) -> None:
    """Check the form of coordinate scales against the mode sizes; raise a ``ValueError`` if wrong.

    Good ``scales`` are a tuple or a list with one array per mode, the ``i``-th of shape ``(N_i,)``: one scale
    per coordinate. This is the form :py:func:`rescale_coordinates`, :py:func:`t3_unscale_coordinates` and
    :py:func:`ut3_unscale_coordinates` take. Refused: a wrong count of arrays, an array of the wrong length for
    its mode, an array that is not a vector (a number per mode, or a 2-dimensional array), and anything that is
    not a tuple or a list (scales packed to one array ``(d, N)`` included: check before packing).

    This is a checker, not a guarantee: those three functions do not call it. What they do with wrong
    ``scales``, silently: a list that is too short drops the modes it does not cover (the result has fewer
    modes); a scale of length 1, or a number, is broadcast over all the coordinates of its mode. The
    ``unscale_coordinates`` methods of the two tensor classes call this checker.

    Only the form is checked, never a value (shapes are static, so the check also runs under ``jax.jit``).
    A scale should be positive. A zero scale is not refused: :py:func:`rescale_coordinates` then zeroes that
    coordinate of every input and, for the probe kind, divides the data there by zero; and
    :py:func:`t3_unscale_coordinates` multiplies that column of the Tucker core by zero, so every entry of the
    tensor with that coordinate becomes zero. An infinite or ``nan`` scale goes through the arithmetic in the
    same way.

    Parameters
    ----------
    scales: Sequence[NDArray]
        One scale per coordinate: ``len=d``, ``elm_shape=(Ni,)``.
    shape: Sequence[int]
        The mode sizes ``(N_0, ..., N_(d-1))`` of the tensor.

    Raises
    ------
    ValueError
        Naming what is wrong.

    See Also
    --------
    rescale_coordinates
    t3_unscale_coordinates
    ut3_unscale_coordinates

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.data_preparation as prep
    >>> shape = (4, 5, 6)
    >>> scales = [np.ones(N) for N in shape]
    >>> prep.check_coordinate_scales(scales, shape)                 # good: returns nothing

    One number per mode is not the form (it would be broadcast without an error by the functions that take
    ``scales``):

    >>> prep.check_coordinate_scales([np.ones(4), np.ones(1), np.ones(6)], shape)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: scales[1] has shape (1,); expected (N_1,) = (5,): one scale per coordinate of mode 1
    >>> prep.check_coordinate_scales(scales[:2], shape)             # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: scales must be a tuple or a list with one array per mode (len=d=3, elm_shape=(Ni,)); got length 2
    """
    d = len(shape)
    if not isinstance(scales, (tuple, list)):
        raise ValueError('scales must be a tuple or a list with one array per mode (len=d=%d, elm_shape=(Ni,)); '
                         'got a %s' % (d, type(scales).__name__))
    if len(scales) != d:
        raise ValueError('scales must be a tuple or a list with one array per mode (len=d=%d, elm_shape=(Ni,)); '
                         'got length %d' % (d, len(scales)))
    for i, (scale, N) in enumerate(zip(scales, shape)):
        if isinstance(scale, (tuple, list)) or tuple(np.shape(scale)) != (N,):
            got = 'a %s' % type(scale).__name__ if isinstance(scale, (tuple, list)) else 'shape %s' % (tuple(np.shape(scale)),)
            raise ValueError('scales[%d] has %s; expected (N_%d,) = (%d,): one scale per coordinate of mode %d'
                             % (i, got, i, N, i))
