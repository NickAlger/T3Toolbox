# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Test-only: derivative data from the merged sampling methods (graded messages, P13b).

The ``*_derivatives*`` methods are gone. Their values are made here from the merged methods along the line
``w + s p``:
- ``probe(affine_jet(ww, pp), (n,))`` gives the Taylor coefficients;
- ``taylor_to_derivatives`` gives the derivatives (slot ``t`` times ``t!``).

The entries kind is the apply kind at the one-hot jet. The old transposes took their residual in the derivatives'
convention, ``sum_t D_t^T c_t`` with ``D_t = t! T_t``, so they are the merged transposes of the residual with slot
``t`` multiplied by ``t!``: ``taylor_to_derivatives`` again. A packed (uniform) vector keeps its form.

This serves the tests that still fit with the old call form of the frontend optimizers (until P16) and the tests of
the old kinds (until P20). The plan's prescription, section 5.3, P13. ``chunk_size`` is accepted and ignored: the
engine does not chunk.
"""
import math

import numpy as np

import t3toolbox.backend.jets as jets
import t3toolbox.backend.ut3_operations as ut3o


def _factorials(n):
    return np.array([math.factorial(t) for t in range(n + 1)], dtype=float)


def _scale(kind, n, value):
    """Slot ``t`` of the leading jet axis multiplied by ``t!``. A ragged probe vector keeps the form of a tuple; a
    packed one, ``(d,) + (n+1,) + ...``, keeps the form of an array."""
    if kind == 'probe' and ut3o.is_packed(value):
        f = _factorials(n).reshape((1, n + 1) + (1,) * (np.ndim(value) - 2))
        return value * f
    return jets.taylor_to_derivatives(kind, (n,), value)


def _jet(kind, sample, pp, shape):
    """The input jet of the line, and the kind the merged method is called with (entries: apply at the one-hot)."""
    if kind == 'entries':
        return 'apply', jets.affine_jet(jets.index_to_one_hot(sample, shape), pp)
    return kind, jets.affine_jet(sample, pp)


def forward(obj, kind, sample, pp, order):
    """``obj.<kind>_derivatives(sample, pp, order)``, for any of the four classes."""
    method, jet = _jet(kind, sample, pp, obj.shape)
    return _scale(method, order, getattr(obj, method)(jet, (order,)))


def tangent_transpose(cls, kind, residual, sample, pp, frame, order, sum_over_probes=False, chunk_size=None):
    """``cls.<kind>_derivatives_transpose(residual, sample, pp, frame, order, sum_over_probes)``,
    ``cls`` the ``T3Tangent`` or the ``UT3Tangent`` class."""
    method, jet = _jet(kind, sample, pp, frame.shape)
    return getattr(cls, method + '_transpose')(frame, jet, _scale(method, order, residual), (order,),
                                               sum_over_samples=sum_over_probes)


def corewise_transpose(x, kind, residual, sample, pp, order, sum_over_probes=False, chunk_size=None):
    """``x.<kind>_corewise_derivatives_transpose(residual, sample, pp, order, sum_over_probes)``."""
    method, jet = _jet(kind, sample, pp, x.shape)
    return getattr(x, method + '_corewise_transpose')(jet, _scale(method, order, residual), (order,),
                                                      sum_over_samples=sum_over_probes)
