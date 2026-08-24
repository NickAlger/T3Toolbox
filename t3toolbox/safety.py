# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
'''Safe / unsafe mode: numerical **precondition** checks gated by an ambient safety tolerance.

The library distinguishes (see ``docs/contributor/numerical_contract_catalog.md``):

- **consistency / well-formedness** checks (shapes, ranks, ``check_fv_pair``) -- always run, in both
  modes, at construction. These are *not* governed by this module.
- **preconditions** -- the operation is numerically wrong without them, but a violation is not
  *malformedness*. These run **only in safe mode**, and every one of them is **numerical**
  (same-frame, orthogonal frame, gauged variations, tied factors): compared at a jax-aware tolerance,
  and **eager-only** (a tracer cannot be branched on, so they skip under a jax trace).

**Minimal ranks are NOT among them** -- not in either flavour. Minimal rank was measured, op by op,
against a dense oracle and is a correctness precondition for *nothing*: the tangent machinery needs only
per-core orthonormality plus the gauge conditions, and an orthonormal frame carries excess rank harmlessly
as slack between its four rank stores (``up``/``down``, ``left``/``right``). Enforcing it would reject
legitimate frames -- including every frame :py:meth:`T3Frame.random_orthogonal` returns for an
over-ranked request. ``has_minimal_ranks`` (structural) and ``has_numerically_minimal_ranks`` remain as
**diagnostics**; see ``docs/numerical_contracts.md`` for the verdict and ``docs/frame_variations.md`` for
why excess rank is inert.

Safe vs unsafe is an ambient setting (a :py:class:`contextvars.ContextVar`): a
:py:class:`SafetyTolerances` pair is **safe mode**; ``None`` is **unsafe mode**. The default is **safe**.

**Two tolerances, numpy vs jax.** The library mixes numpy and jax arrays deliberately (jax for autodiff
prototyping, *not* jit), and jax runs float32 by default -- so its residuals are far looser than numpy's
float64 (e.g. an orthogonality residual of ~1e-7 vs ~1e-15). A single tolerance would false-fail every jax
input. So a precondition check picks ``rtol_jax`` when ``tree_contains_jax(inputs)`` else ``rtol_numpy``,
using the same numpy/jax dispatch as the rest of the codebase. (If you enable ``jax_enable_x64`` you can
tighten ``rtol_jax`` via :py:func:`safe`.)

Numerical checks are **eager-only** -- skipped under a jax trace (you cannot branch on a tracer, and jit is
for performance, which is unsafe by definition). The contract: validate eagerly in safe mode, then jit
(effectively unsafe) for speed. This module is the **mechanism** only; the per-op checks are wired at the
call sites that own the relevant ``T3*`` objects (see the catalog).

Examples
--------
>>> import numpy as np
>>> import t3toolbox.safety as safety
>>> print(safety.current_safety())                          # default: safe (numpy, jax tolerances)
SafetyTolerances(rtol_numpy=1e-09, rtol_jax=1e-05)
>>> print(safety.effective_rtol((np.zeros(3),)))            # numpy input -> numpy rtol
1e-09
>>> with safety.unsafe():
...     print(safety.current_safety(), safety.checks_active(np.zeros(3)))
None False
>>> print(safety.current_safety() is None)                  # restored on exit
False

``frames_equal`` is the honest "same tangent space" test -- it accepts value-equal-but-different-object
frames (a jit round-trip), unlike object identity:

>>> a = (np.ones((2, 3)),)
>>> b = (np.ones((2, 3)),)                                   # value-equal, different objects
>>> print(a is b, safety.frames_equal(a, b))
False True
>>> print(safety.frames_equal(a, (np.zeros((2, 3)),)))      # genuinely different
False
'''

import contextlib
import contextvars
import typing as typ

import numpy as np

from t3toolbox.backend.common import jax_available, tree_contains_jax

__all__ = [
    'SafetyTolerances',
    'DEFAULT_RTOL_NUMPY',
    'DEFAULT_RTOL_JAX',
    'safe',
    'unsafe',
    'current_safety',
    'set_default_safety',
    'effective_rtol',
    'comparison_rtol',
    'is_tracing',
    'checks_active',
    'require',
    'frames_equal',
    'frames_equal_or_skip',
]

DEFAULT_RTOL_NUMPY = 1e-9        # numpy float64
DEFAULT_RTOL_JAX = 1e-5          # jax defaults to float32 (no x64) -> far looser residuals


class SafetyTolerances(typ.NamedTuple):
    '''The two precondition-check tolerances; the one used is chosen by ``tree_contains_jax(inputs)``.'''
    rtol_numpy: float
    rtol_jax: float


_DEFAULT = SafetyTolerances(DEFAULT_RTOL_NUMPY, DEFAULT_RTOL_JAX)
_safety = contextvars.ContextVar('t3_safety', default=_DEFAULT)   # a SafetyTolerances (safe) or None (unsafe)


def current_safety():
    '''The ambient :py:class:`SafetyTolerances` (safe mode) or ``None`` (unsafe mode).'''
    return _safety.get()


def set_default_safety(rtol_numpy=DEFAULT_RTOL_NUMPY, rtol_jax=DEFAULT_RTOL_JAX):
    '''Set the safety tolerances for the current context. Prefer the :py:func:`safe` / :py:func:`unsafe`
    context managers for scoped changes; use this for a script-level default.

    ``set_default_safety(None, None)`` makes **unsafe** the default (the only script-level way to do so;
    ``unsafe()`` is a context manager). Otherwise both tolerances must be positive numbers -- a lone
    ``None`` is rejected here rather than failing inside the first precondition check.

    Scope: a ``contextvars`` variable, so the setting applies to the **current context** -- the calling
    thread, or the current asyncio task and the tasks it spawns afterwards. Threads started before or
    independently see the module default (safe); set it in each thread, or use the context managers.'''
    if rtol_numpy is None and rtol_jax is None:
        _safety.set(None)                       # the ContextVar's own None-means-unsafe convention
        return
    for name, v in (('rtol_numpy', rtol_numpy), ('rtol_jax', rtol_jax)):
        if v is None or not (float(v) > 0):
            raise ValueError('set_default_safety: %s must be a positive number, or BOTH tolerances None for '
                             'unsafe mode; got rtol_numpy=%r, rtol_jax=%r' % (name, rtol_numpy, rtol_jax))
    _safety.set(SafetyTolerances(rtol_numpy, rtol_jax))


@contextlib.contextmanager
def safe(rtol_numpy=DEFAULT_RTOL_NUMPY, rtol_jax=DEFAULT_RTOL_JAX):
    '''Context manager: run numerical precondition checks within the block, at these tolerances.'''
    token = _safety.set(SafetyTolerances(rtol_numpy, rtol_jax))
    try:
        yield
    finally:
        _safety.reset(token)


@contextlib.contextmanager
def unsafe():
    '''Context manager: skip numerical precondition checks within the block (performance / jit).'''
    token = _safety.set(None)
    try:
        yield
    finally:
        _safety.reset(token)


def effective_rtol(*inputs):
    '''The tolerance to use for a check on these inputs (``rtol_jax`` if any input is a jax array, else
    ``rtol_numpy``), or ``None`` in unsafe mode. Pass the operand data trees, e.g. ``effective_rtol(frame.data)``.'''
    tols = _safety.get()
    if tols is None:
        return None
    return tols.rtol_jax if tree_contains_jax(inputs) else tols.rtol_numpy


def comparison_rtol(*inputs):
    '''The default tolerance for the named equality checks (``numerically_equal`` and friends): the
    ambient jax-aware tolerance, falling back to the defaults in unsafe mode. A pure comparison is
    mode-agnostic (like :py:func:`frames_equal`), so unlike :py:func:`effective_rtol` this never
    returns ``None``.'''
    tols = _safety.get() or _DEFAULT
    return tols.rtol_jax if tree_contains_jax(inputs) else tols.rtol_numpy


_trace_probe = None  # a committed jax array; lazily built (fallback trace detector, see _inside_jax_trace)


def _inside_jax_trace():
    '''True iff we are inside a **jit** trace, independent of the inputs (the committed-array probe).

    Needed because a numerical check can operate on **closed-over concrete** arrays (not tracers): inside
    a jit trace, even a jnp op on a committed array yields a (constant) tracer, so ``bool(...)`` on the
    result would fail. Inspecting only the passed arrays (``is_tracing``'s fast path) misses this. The
    probe: an op on a committed array is a tracer iff we are jit-tracing.

    Deliberately jit-only (review H6-12): under ``grad``/``vmap`` with concrete operands the probe returns
    False and the checks simply run -- harmless, since concrete operands mean ``bool(...)`` works, and the
    checks pass or fail on their merits. (The old primary detector, ``jax.core.trace_state_clean``, does
    not exist in jax 0.10; it cost a raise/catch per check and promised any-transform detection the
    fallback never delivered.)'''
    if not jax_available:
        return False
    import jax
    global _trace_probe
    try:
        if _trace_probe is None:
            _trace_probe = jax.numpy.zeros(1)
        return isinstance(_trace_probe + _trace_probe, jax.core.Tracer)
    except Exception:
        return False


def is_tracing(*arrays):
    '''True if we are inside a jax transform: any argument is a tracer (catches jit / grad / vmap
    operands), **or** we are inside a jit trace with only closed-over concrete operands (the probe --
    see :py:func:`_inside_jax_trace`).'''
    if not jax_available:
        return False
    import jax
    if any(isinstance(a, jax.core.Tracer) for a in arrays):
        return True
    return _inside_jax_trace()


def checks_active(*inputs):
    '''True iff numerical precondition checks should run here: **safe mode AND not under a jax trace.**

    Pass the operand arrays/trees so tracing is detected; a check site computes its (possibly expensive)
    numerical condition only when this returns True::

        if safety.checks_active(frame.data):
            safety.require(frame.is_orthogonal(atol=safety.effective_rtol(frame.data)).all(), 'frame not orthogonal')
    '''
    return _safety.get() is not None and not is_tracing(*_flatten_arrays(inputs))


def require(condition, message):
    '''Raise ``ValueError(message)`` if ``condition`` is falsy. Call only inside a :py:func:`checks_active`
    guard, so the (possibly expensive) condition is not computed in unsafe mode / under a trace.'''
    if not condition:
        raise ValueError(message)


def frames_equal(data1, data2, rtol=None):
    '''Numerical equality of two frames given as nested core tuples (e.g. ``T3Frame.data``).

    The honest "same tangent space" test: two frames are the same iff their cores are equal. It accepts
    the value-equal-but-different-object frames a jit round-trip produces, while rejecting a genuinely
    different frame. Contract: callers pass **real content only** -- a padded representation (the
    uniform layer) masks its supercores first (``ufv_apply_frame_masks``), so don't-care padding never
    participates; this function itself is representation-agnostic and mask-ignorant.
    ``rtol`` defaults to the ambient jax-aware tolerance (falling back to the defaults if
    unsafe -- this is a pure comparison, mode-agnostic).'''
    if rtol is None:
        tols = _safety.get() or _DEFAULT
        rtol = tols.rtol_jax if tree_contains_jax((data1, data2)) else tols.rtol_numpy
    return _arrays_allclose(_flatten_arrays(data1), _flatten_arrays(data2), rtol)


def frames_equal_or_skip(data1, data2):
    '''The convenience guard for "same frame": ``True`` when checks are inactive (unsafe / under a trace)
    or when the frames are numerically equal (jax-aware tolerance) -- so a call site reads::

        if not (b1 is b2 or safety.frames_equal_or_skip(b1.data, b2.data)):
            raise ValueError('tangents are in different tangent spaces')

    The ``b1 is b2`` identity is the O(1) fast path; this only runs (and only matters) when the objects
    differ but may still represent the same frame (the jit round-trip).'''
    tols = _safety.get()
    if tols is None:
        return True                                              # unsafe -> skip
    flat1, flat2 = _flatten_arrays(data1), _flatten_arrays(data2)
    if is_tracing(*flat1, *flat2):
        return True                                              # under a trace -> skip
    rtol = tols.rtol_jax if tree_contains_jax((data1, data2)) else tols.rtol_numpy
    return _arrays_allclose(flat1, flat2, rtol)


def _arrays_allclose(flat1, flat2, rtol):
    if len(flat1) != len(flat2):
        return False
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
