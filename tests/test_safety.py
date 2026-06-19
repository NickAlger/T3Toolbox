'''Tests for the safe/unsafe-mode mechanism (``t3toolbox/safety.py``).

The mechanism only -- no per-op preconditions are wired yet (those are S3-S5). Checks: the ambient
``safety_rtol`` mode + context managers (scoped, nesting, restore), ``frames_equal`` (accepts
value-equal-different-object, rejects genuine difference), and the eager-only behaviour (checks skip in
unsafe mode and under a jax trace).
'''

import unittest

import numpy as np

import t3toolbox.safety as safety

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TestSafetyMode(unittest.TestCase):
    def test_default_is_safe(self):
        self.assertEqual(safety.current_safety_rtol(), safety.DEFAULT_SAFETY_RTOL)
        self.assertIsNotNone(safety.current_safety_rtol())

    def test_context_managers_scope_and_restore(self):
        self.assertIsNotNone(safety.current_safety_rtol())
        with safety.unsafe():
            self.assertIsNone(safety.current_safety_rtol())
            with safety.safe(rtol=1e-12):                       # nesting
                self.assertEqual(safety.current_safety_rtol(), 1e-12)
            self.assertIsNone(safety.current_safety_rtol())     # inner restored
        self.assertEqual(safety.current_safety_rtol(), safety.DEFAULT_SAFETY_RTOL)  # outer restored

    def test_checks_active(self):
        a = np.zeros(3)
        self.assertTrue(safety.checks_active(a))                # default safe, eager
        with safety.unsafe():
            self.assertFalse(safety.checks_active(a))
        self.assertTrue(safety.checks_active(a))

    def test_require(self):
        safety.require(True, 'should not raise')
        with self.assertRaises(ValueError):
            safety.require(False, 'boom')


class TestFramesEqual(unittest.TestCase):
    def test_value_equal_different_object(self):
        a = (np.ones((2, 3)), (np.arange(6.0).reshape(2, 3),))
        b = (np.ones((2, 3)), (np.arange(6.0).reshape(2, 3),))   # value-equal, different objects
        self.assertIsNot(a, b)
        self.assertTrue(safety.frames_equal(a, b))               # the jit round-trip case -> accepted

    def test_genuinely_different(self):
        a = (np.ones((2, 3)),)
        self.assertFalse(safety.frames_equal(a, (np.zeros((2, 3)),)))

    def test_different_shapes(self):
        self.assertFalse(safety.frames_equal((np.ones((2, 3)),), (np.ones((2, 4)),)))

    def test_or_skip_unsafe(self):
        a = (np.ones((2, 3)),)
        b = (np.zeros((2, 3)),)                                   # different
        self.assertFalse(safety.frames_equal_or_skip(a, b))      # safe -> the real (False) result
        with safety.unsafe():
            self.assertTrue(safety.frames_equal_or_skip(a, b))   # unsafe -> skipped (passes)


@unittest.skipUnless(HAS_JAX, 'jax not available')
class TestEagerOnlyUnderJit(unittest.TestCase):
    def test_is_tracing(self):
        self.assertFalse(safety.is_tracing(np.zeros(3)))         # eager numpy
        captured = {}
        def f(x):
            captured['tracing'] = safety.is_tracing(x)           # x is a tracer inside jit
            return x
        jax.jit(f)(jnp.ones(3))
        self.assertTrue(captured['tracing'])

    def test_checks_skip_under_trace(self):
        # checks_active is False under a trace even in safe mode -> precondition checks are no-ops in jit
        out = {}
        def f(x):
            out['active'] = safety.checks_active(x)
            out['or_skip'] = safety.frames_equal_or_skip((x,), (jnp.zeros(3),))  # would be False eagerly
            return x
        jax.jit(f)(jnp.ones(3))
        self.assertFalse(out['active'])                          # safe mode, but tracing -> inactive
        self.assertTrue(out['or_skip'])                          # skipped under trace -> passes


if __name__ == '__main__':
    unittest.main()
