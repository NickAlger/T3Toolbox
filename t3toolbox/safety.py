# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
'''Safe / unsafe mode: numerical **precondition** checks gated by an ambient safety tolerance.

The library distinguishes two kinds of guard (see ``docs/safe_unsafe_mode_plan.md`` and
``docs/numerical_contract_catalog.md``):

- **consistency / well-formedness** checks (shapes, ranks, ``check_bv_pair``) -- always run, in both
  modes, at construction. These are *not* governed by this module.
- **numerical preconditions** (same-frame, orthogonal base, gauged variations, structurally-minimal ranks)
  -- the operation is numerically wrong without them, but a violation is not *malformedness*. These run
  **only in safe mode**.

Safe vs unsafe is an ambient ``safety_rtol`` (a :py:class:`contextvars.ContextVar`): a float is **safe
mode** at that tolerance; ``None`` is **unsafe mode**. The default is **safe**. Numerical checks are
**eager-only** -- skipped under a jax trace (you cannot branch on a tracer, and jit is for performance,
which is unsafe by definition). The contract: validate eagerly in safe mode, then jit (effectively unsafe)
for speed.

This module is the **mechanism** only; the per-op precondition checks are wired at the call sites that own
the relevant ``T3*`` objects (see the catalog).

Examples
--------
>>> import numpy as np
>>> import t3toolbox.safety as safety
>>> print(safety.current_safety_rtol())                 # default: safe
1e-09
>>> with safety.unsafe():
...     print(safety.current_safety_rtol())             # None == unsafe
...     print(safety.checks_active(np.zeros(3)))         # checks are off
None
False
>>> print(safety.current_safety_rtol())                 # restored on exit
1e-09

``frames_equal`` is the honest "same tangent space" test -- it accepts value-equal-but-different-object
frames (a jit round-trip), unlike object identity:

>>> a = (np.ones((2, 3)),)
>>> b = (np.ones((2, 3)),)                                # value-equal, different objects
>>> print(a is b, safety.frames_equal(a, b))
False True
>>> print(safety.frames_equal(a, (np.zeros((2, 3)),)))   # genuinely different
False
'''

import contextlib
import contextvars

import numpy as np

from t3toolbox.backend.common import has_jax

__all__ = [
    'DEFAULT_SAFETY_RTOL',
    'safe',
    'unsafe',
    'current_safety_rtol',
    'set_default_safety_rtol',
    'is_tracing',
    'checks_active',
    'require',
    'frames_equal',
    'frames_equal_or_skip',
]

DEFAULT_SAFETY_RTOL = 1e-9                                       # default mode is SAFE (None == unsafe)

_safety_rtol = contextvars.ContextVar('t3_safety_rtol', default=DEFAULT_SAFETY_RTOL)


def current_safety_rtol():
    '''The ambient safety tolerance: a float (safe mode) or ``None`` (unsafe mode).'''
    return _safety_rtol.get()


def set_default_safety_rtol(rtol):
    '''Set the safety tolerance for the current context (``None`` = unsafe). Prefer the :py:func:`safe` /
    :py:func:`unsafe` context managers for scoped changes; use this for a script-level default.'''
    _safety_rtol.set(rtol)


@contextlib.contextmanager
def safe(rtol=DEFAULT_SAFETY_RTOL):
    '''Context manager: run numerical precondition checks within the block, at tolerance ``rtol``.'''
    token = _safety_rtol.set(rtol)
    try:
        yield
    finally:
        _safety_rtol.reset(token)


@contextlib.contextmanager
def unsafe():
    '''Context manager: skip numerical precondition checks within the block (performance / jit).'''
    token = _safety_rtol.set(None)
    try:
        yield
    finally:
        _safety_rtol.reset(token)


def is_tracing(*arrays):
    '''True if any argument is a jax tracer (i.e. we are inside a jax transform: jit / grad / vmap).'''
    if not has_jax:
        return False
    import jax
    return any(isinstance(a, jax.core.Tracer) for a in arrays)


def checks_active(*witness_arrays):
    '''True iff numerical precondition checks should run here: **safe mode AND not under a jax trace.**

    Pass a few representative arrays from the operands as ``witness_arrays`` so tracing is detected; a
    check site computes its (possibly expensive) numerical condition only when this returns True::

        if safety.checks_active(*basis.up_tucker_cores):
            safety.require(basis.is_orthogonal(atol=safety.current_safety_rtol()), 'basis not orthogonal')
    '''
    return current_safety_rtol() is not None and not is_tracing(*witness_arrays)


def require(condition, message):
    '''Raise ``ValueError(message)`` if ``condition`` is falsy. Call only inside a :py:func:`checks_active`
    guard, so the (possibly expensive) condition is not computed in unsafe mode / under a trace.'''
    if not condition:
        raise ValueError(message)


def frames_equal(data1, data2, rtol=None):
    '''Numerical equality of two frames given as nested core tuples (e.g. ``T3Basis.data``).

    The honest "same tangent space" test: two frames are the same iff their cores are equal. It accepts
    the value-equal-but-different-object frames a jit round-trip produces, while rejecting a genuinely
    different base. ``rtol`` defaults to the ambient :py:func:`current_safety_rtol`.'''
    if rtol is None:
        rtol = current_safety_rtol() or DEFAULT_SAFETY_RTOL
    flat1, flat2 = _flatten_arrays(data1), _flatten_arrays(data2)
    if len(flat1) != len(flat2):
        return False
    return all(a.shape == b.shape and np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=0.0)
               for a, b in zip(flat1, flat2))


def frames_equal_or_skip(data1, data2):
    '''The convenience guard for "same frame": ``True`` when checks are inactive (unsafe / under a trace)
    or when the frames are numerically equal -- so a call site reads::

        if not (b1 is b2 or safety.frames_equal_or_skip(b1.data, b2.data)):
            raise ValueError('tangents are in different tangent spaces')

    The ``b1 is b2`` identity is the O(1) fast path; this only runs (and only matters) when the objects
    differ but may still represent the same frame (the jit round-trip).'''
    if current_safety_rtol() is None:
        return True
    flat1, flat2 = _flatten_arrays(data1), _flatten_arrays(data2)
    if is_tracing(*flat1, *flat2):
        return True
    if len(flat1) != len(flat2):
        return False
    rtol = current_safety_rtol()
    return all(a.shape == b.shape and np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=0.0)
               for a, b in zip(flat1, flat2))


def _flatten_arrays(tree):
    out = []
    if isinstance(tree, (tuple, list)):
        for sub in tree:
            out.extend(_flatten_arrays(sub))
    else:
        out.append(tree)
    return out
