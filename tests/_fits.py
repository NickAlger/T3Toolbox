# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Test-only: a fit for the backend optimizers, built from plain specs (graded messages, P16).

A backend fit is a geometry, an oracle, its batch and an optional regularizer. :class:`Fit` holds the four and
forwards each call to the backend function of the same name, every argument by keyword after the four, so a test reads
``fit.newton_cg(x0, max_newton=8)`` for ``newton_cg(geom, oracle, batch, x0, regularizer=..., max_newton=8)``. It is
a convenience of the tests only: the library has no such object (the optimizers take the four), and the real call
forms are pinned by the module doctests, by ``tests/backend/test_optimizers.py``'s own tests of the head, and by
``tests/test_dispatch.py``.

The builders:
- :func:`fit` -- a plain fit (or a fit of Taylor data at a jet shape ``order``): ``LeastSquaresOracle(kind, order)``
  and ``LeastSquaresBatch(sample, y, weight)``;
- :func:`derivative_fit` -- a fit of derivative data along the line ``w + s p`` (what the old derivative kinds fitted):
  the sample ``affine_jet(ww, pp)`` (entries: the apply kind at ``affine_jet(index_to_one_hot(index, shape), pp)``),
  the data and the weight through ``prepare_derivative_data``, so the objective is the old kinds' exactly
  (``1/2 sum_t |omega_t (d_t(x) - y_t)|^2`` on the derivatives);

  both take the weight as **one value per block** (probe ``(d,)+F``, apply ``F``, entries ``()``: the form the tests
  were written in, a weight per mode and slot) and place it on the data with ``jets.blocks_to_weight``, which reads
  the rank of ``y`` (its stack ``C`` included), so the batch carries a weight of the type of ``z`` (data-metric plan
  §2; M2 part 4). A test that wants another full-form weight (per sample, per entry) builds its batch itself;
- :func:`shared_over` -- the same fit for a model stacked over ``C`` whose data are shared by every element: ``y`` and
  the weight with size-1 ``C`` axes (a stacked model's data carry ``C``; the elements' models use the fit as built);
  :func:`at_element` the inverse, a stacked fit's data at one element of ``C``;
- :func:`uniform` -- the same fit on the uniform layout: the geometry ``from_point`` of the packed start, the batch
  ``pack_batch``-ed, and the start's bare supercore pair.
"""
import dataclasses
import typing as typ

import numpy as np

import t3toolbox.backend.optimizers as opt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.jets as jets
import t3toolbox.backend.uniform_fitting as uf
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch


class Fit(typ.NamedTuple):
    geom:         typ.Any
    oracle:       typ.Any
    batch:        typ.Any
    regularizer:  typ.Any = None

    def local_model(self, x, **kw):
        return opt.local_model(self.geom, self.oracle, self.batch, x, regularizer=self.regularizer, **kw)

    def objective(self, x, batch=None):
        return opt.point_objective(self.geom, self.oracle, self.batch if batch is None else batch, x,
                                   regularizer=self.regularizer)

    def newton_cg(self, x0, **kw):
        return opt.newton_cg(self.geom, self.oracle, self.batch, x0, regularizer=self.regularizer, **kw)

    def gradient_descent(self, x0, **kw):
        return opt.gradient_descent(self.geom, self.oracle, self.batch, x0, regularizer=self.regularizer, **kw)

    def mc_sgd(self, x0, rng, batch_size, **kw):
        return opt.mc_sgd(self.geom, self.oracle, self.batch, x0, rng, batch_size, regularizer=self.regularizer, **kw)

    def adam(self, x0, rng, batch_size, **kw):
        return opt.adam(self.geom, self.oracle, self.batch, x0, rng, batch_size, regularizer=self.regularizer, **kw)

    def padded_start(self, x0, x_old, **kw):
        return opt.padded_start(self.geom, self.oracle, self.batch, x0, x_old, **kw)

    def with_(self, **fields):                       # a copy with fields replaced (a regularizer, a batch)
        return self._replace(**fields)


def fit(
        geom,
        kind:         str,                           # 'probe' | 'apply' | 'entries'
        sample,                                      # ww (input jets at `order`) | index
        y,                                           # of the type of z
        regularizer=None,
        weight=None,                                 # one value per block: probe (d,)+F, apply F, entries (); None = ones
        order:        typ.Tuple[int, ...] = (),
) -> Fit:
    order = tuple(order)
    omega = None if weight is None else jets.blocks_to_weight(kind, order, np.asarray(weight), y)
    return Fit(geom, LeastSquaresOracle(kind, order), LeastSquaresBatch(sample, y, omega), regularizer)


def derivative_fit(
        geom,
        kind:         str,                           # 'probe' | 'apply' | 'entries' (entries: the apply kind at one-hots)
        order:        int,                           # the derivative order n: the jet shape (n,)
        sample,                                      # ww | index
        pp,                                          # the line's direction
        derivatives,                                 # derivative data: slot t the t-th derivative along the line
        regularizer=None,
        weight=None,                                 # one value per block on the derivative residuals: probe (d, n+1),
                                                     #   else (n+1,); None = ones
        shape=None,                                  # entries: the mode sizes (for the one-hot vectors)
) -> Fit:
    o = (int(order),)
    if kind == 'entries':
        kind, sample = 'apply', jets.index_to_one_hot(sample, shape)
    omega = None if weight is None else jets.blocks_to_weight(kind, o, np.asarray(weight), derivatives)
    y, w, _ = jets.prepare_derivative_data(kind, o, derivatives, omega)
    return Fit(geom, LeastSquaresOracle(kind, o), LeastSquaresBatch(jets.affine_jet(sample, pp), y, w), regularizer)


def shared_over(
        fit:  Fit,                                   # a fit whose y (and weight) have no stack C
        C:    typ.Tuple[int, ...],                   # the model's stack of base points
) -> Fit:                                            # the same fit for a model stacked over C, its data shared
    """The fit for a model stacked over ``C`` whose data are shared by every element: ``y`` and the weight with
    ``len(C)`` axes of size 1 in the place of ``C`` (data-metric plan §2: data shared by a stack have size-1 ``C``
    axes). ``C`` is the last group before the probe kind's mode edge, and the last of apply's and entries' arrays,
    whatever the form (a ragged tuple, one mode-leading array, packed). ``C = ()`` returns the fit unchanged."""
    if not C:
        return fit
    probe = fit.oracle.kind == 'probe'

    def placed(a):
        a = np.asarray(a) if not hasattr(a, 'shape') else a
        at = a.ndim - 1 if probe else a.ndim
        return a.reshape(a.shape[:at] + (1,) * len(C) + a.shape[at:])

    def each(value):
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            return [placed(a) for a in value]
        return placed(value)
    return fit._replace(batch=dataclasses.replace(fit.batch, y=each(fit.batch.y), weight=each(fit.batch.weight)))


def at_element(
        fit:  Fit,                                   # a fit for a model stacked over C (y and the weight carry C)
        cc:   typ.Tuple[int, ...],                   # an index into C
) -> Fit:                                            # the fit of element cc: y and the weight at cc, C dropped
    """The fit of one element of a stacked model: ``y`` and the weight at the element ``cc`` of their ``C`` axes (an
    axis of size 1 is the same for every element, read at 0), ``C`` dropped. The inverse of :func:`shared_over`, and
    the per-element fit of data that differ by element. ``C`` is located as there."""
    probe = fit.oracle.kind == 'probe'
    n = len(cc)

    def picked(a):
        a = np.asarray(a) if not hasattr(a, 'shape') else a
        start = (a.ndim - 1 if probe else a.ndim) - n
        at = tuple(c if a.shape[start + k] > 1 else 0 for k, c in enumerate(cc))
        return a[(slice(None),) * start + at]

    def each(value):
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            return [picked(a) for a in value]
        return picked(value)
    return fit._replace(batch=dataclasses.replace(fit.batch, y=each(fit.batch.y), weight=each(fit.batch.weight)))


def uniform(
        ragged:      Fit,                            # the fit on the ragged layout (its geometry names the uniform one)
        ux,                                          # the UniformTuckerTensorTrain start (minimal ranks)
        sharing=None,
) -> typ.Tuple[Fit, typ.Tuple]:                      # (the uniform fit, the start's bare supercore pair)
    manifold = isinstance(ragged.geom, bgeo.ManifoldGeometryOps)
    cls = bgeo.UniformManifoldGeometryOps if manifold else bgeo.UniformCorewiseGeometryOps
    geom = cls.from_point(ux.data, sharing)
    batch = uf.pack_batch(ragged.oracle.kind, ragged.oracle.order, ragged.batch, ux.N)
    return Fit(geom, ragged.oracle, batch, ragged.regularizer), (ux.data[0], ux.data[1])


def old_vocabulary(
        kind:    str,                 # a 2026.5.0 kind name: 'probe', ..., 'probe_derivatives', ...
        sample,                       # ww / index, or (ww, pp) / (index, pp)
        data,                         # the data (derivatives for a derivative kind)
        shape,                        # the mode sizes
        order=None,                   # derivative kinds: the highest order
        weight=None,                  # plain probe (d,); a derivative kind: (n+1,) per order, or probe (d, n+1)
):                                    # (oracle, batch)
    """(oracle, batch) of a fit stated in the 2026.5.0 vocabulary, built with the public helpers: a plain kind at the
    jet shape ``()``; a derivative kind along the lines ``affine_jet(ww, pp)`` (entries: the apply kind at the one-hot
    jet) with ``prepare_derivative_data``; a 1-D per-order weight of a probe fit broadcast over the modes. (Test-only;
    independent of ``optimizers``' translation of the old call form.)"""
    if kind.endswith('_derivatives'):
        base, (first, pp) = kind[:-len('_derivatives')], sample
        w = None if weight is None else np.asarray(weight, dtype=float)
        if w is not None and base == 'probe' and w.ndim == 1:
            w = np.broadcast_to(w, (len(shape), order + 1)).copy()
        fit = derivative_fit(None, base, order, first, pp, data, weight=w, shape=shape)
    else:
        fit = globals()['fit'](None, kind, sample, data,
                               weight=None if weight is None else np.asarray(weight, dtype=float))
    return fit.oracle, fit.batch


def old_model(
        kind:      str,               # the 2026.5.0 factory's kind: 'apply' -> apply_model, ...
        geometry,                     # the frontend geometry
        x,                            # the point (ragged or uniform; unstacked, or data shared over its stack)
        sample,
        residual,                     # the residual r = z(x) - y in the old form (derivative kinds: derivatives)
        order=None,
        weight=None,
        **kwargs,                     # regularizer, hessian, curvature_damping, frame
):                                    # the frontend LocalModel whose residual is `residual`
    """The 2026.5.0 factory ``<kind>_model(geometry, x, sample, residual, ...)`` in the new form: the data
    ``y = z(x) - residual`` (``z`` from the ragged point; derivative kinds through the derivative data), then
    ``fitting.local_model``. The model's residual is ``residual`` to rounding. The old factories took one ``y`` for a
    stacked model, so for a stacked ``x`` the residual must be ``z(x) - y`` with one ``y`` for every element; another
    raises (it would fit another residual than the one stated). That one ``y`` (and the weight) is then given size-1
    ``C`` axes (:func:`shared_over`), the new form of data shared by the stack."""
    import t3toolbox.fitting as fitting
    import t3toolbox.uniform_tucker_tensor_train as ut3
    import _derivative_data as dd
    rx = x.to_t3() if isinstance(x, ut3.UniformTuckerTensorTrain) else x
    if kind.endswith('_derivatives'):
        base, (first, pp) = kind[:-len('_derivatives')], sample
        z = dd.forward(rx, base, first, pp, order)
    else:
        z = rx.entries(sample) if kind == 'entries' else (rx.probe(sample) if kind == 'probe' else rx.apply(sample))
    stack = len(rx.stack_shape)
    if isinstance(z, (list, tuple)):
        y = [np.asarray(zi) - np.asarray(ri) for zi, ri in zip(z, residual)]
        if stack:
            y = [_one_over_the_stack(yi, (slice(None),) * (yi.ndim - 1 - stack), stack) for yi in y]
    else:
        y = np.asarray(z) - np.asarray(residual)
        if stack:
            y = _one_over_the_stack(y, (Ellipsis,), stack)
    oracle, batch = old_vocabulary(kind, sample, y, tuple(rx.shape), order, weight)
    shared = shared_over(Fit(None, oracle, batch), tuple(rx.stack_shape))
    return fitting.local_model(geometry, oracle, shared.batch, x, **kwargs)


def _one_over_the_stack(y, lead, stack):
    """``y``'s element 0 over the stack axes (the last ``stack`` axes after ``lead``), after checking every element
    has the same ``y`` (to rounding)."""
    first = y[lead + (0,) * stack]
    if not np.allclose(y, first[lead + (None,) * stack], rtol=1e-9, atol=1e-9 * max(1.0, float(np.max(np.abs(y))))):
        raise ValueError('old_model: a stacked point needs one y for every element (the residual z(x_c) - y); this '
                         'residual gives a different y per element')
    return first


def old_factory(kind: str):
    """The 2026.5.0 factory ``fitting.<kind>_model`` with its exact signature, in the new form (:py:func:`old_model`):
    ``apply_model`` / ``entries_model(geometry, x, sample, residual, regularizer, hessian, curvature_damping)``,
    ``probe_model(geometry, x, ww, residual, weight, ...)``, ``<kind>_derivatives_model(geometry, x, first, pp,
    order, residual, weight, ...)``."""
    if kind.endswith('_derivatives'):
        def factory(geometry, x, first, pp, order, residual, weight=None, regularizer=None, hessian='gauss_newton',
                    curvature_damping=None):
            return old_model(kind, geometry, x, (first, pp), residual, order, weight, regularizer=regularizer,
                             hessian=hessian, curvature_damping=curvature_damping)
    elif kind == 'probe':
        def factory(geometry, x, ww, residual, weight=None, regularizer=None, hessian='gauss_newton',
                    curvature_damping=None):
            return old_model(kind, geometry, x, ww, residual, None, weight, regularizer=regularizer, hessian=hessian,
                             curvature_damping=curvature_damping)
    else:
        def factory(geometry, x, sample, residual, regularizer=None, hessian='gauss_newton', curvature_damping=None):
            return old_model(kind, geometry, x, sample, residual, regularizer=regularizer, hessian=hessian,
                             curvature_damping=curvature_damping)
    return factory
