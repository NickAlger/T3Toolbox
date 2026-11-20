# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Settings shared by every test file.

**``subTest`` labels must be builtins.** CI's floor leg runs pytest-xdist with pytest-subtests on numpy 1.x, which
sends each subtest's ``msg`` and keyword labels to the controller through execnet. execnet serializes by exact type,
and only the builtin types below, so a numpy scalar (``numpy.int64``, ``numpy.bool_``, ``numpy.float64``), an array
or a dtype as a label fails the test there, and only there: neither pytest 9 nor a run without xdist shows it. It bit
the 2026.5.0 release and three tests of the graded-messages arc. This guard makes the mistake fail everywhere, at
the ``subTest`` call, with a message that says what to write instead (``repr(value)``).
"""

import unittest

_SENDABLE = (type(None), bool, int, float, complex, str, bytes)     # execnet's save_<type> methods, by exact type
_CONTAINERS = (tuple, list, set, frozenset)


def _unsendable(
        value,                      # a subTest label, or a part of one
):                                  # -> the qualified type name of its first part execnet cannot send, or None
    kind = type(value)
    if kind in _SENDABLE:
        return None
    if kind in _CONTAINERS:
        return next((bad for bad in map(_unsendable, value) if bad is not None), None)
    if kind is dict:
        return next((bad for bad in map(_unsendable, [*value.keys(), *value.values()]) if bad is not None), None)
    return kind.__module__ + '.' + kind.__qualname__


_unchecked_subTest = unittest.TestCase.subTest


def _subTest_with_builtin_labels(self, *args, **params):
    for name, value in [('msg', a) for a in args] + list(params.items()):
        bad = _unsendable(value)
        if bad is not None:
            raise TypeError(
                f"subTest label {name}={value!r} holds a {bad}: CI's floor leg (pytest-xdist on numpy 1.x) cannot "
                f"send it, and the test fails there only. Use builtins, e.g. {name}=repr(...) "
                f"(tests/conftest.py).")
    return _unchecked_subTest(self, *args, **params)


unittest.TestCase.subTest = _subTest_with_builtin_labels
