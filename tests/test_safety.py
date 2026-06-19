'''Tests for the safe/unsafe-mode mechanism (``t3toolbox/safety.py``).

The mechanism only -- no per-op preconditions are wired yet (those are S3-S5). Checks: the ambient mode +
context managers (scoped, nesting, restore), the two-tolerance numpy/jax picking (``effective_rtol`` via
``tree_contains_jax``), ``frames_equal`` (accepts value-equal-different-object, rejects genuine
difference), and the eager-only behaviour (checks skip in unsafe mode and under a jax trace).
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
        s = safety.current_safety()
        self.assertEqual(s, safety.SafetyTolerances(safety.DEFAULT_RTOL_NUMPY, safety.DEFAULT_RTOL_JAX))
        self.assertIsNotNone(s)

    def test_context_managers_scope_and_restore(self):
        self.assertIsNotNone(safety.current_safety())
        with safety.unsafe():
            self.assertIsNone(safety.current_safety())
            with safety.safe(rtol_numpy=1e-12, rtol_jax=1e-6):     # nesting
                self.assertEqual(safety.current_safety().rtol_numpy, 1e-12)
                self.assertEqual(safety.current_safety().rtol_jax, 1e-6)
            self.assertIsNone(safety.current_safety())             # inner restored
        self.assertIsNotNone(safety.current_safety())              # outer restored

    def test_effective_rtol_numpy(self):
        self.assertEqual(safety.effective_rtol((np.zeros(3),)), safety.DEFAULT_RTOL_NUMPY)
        with safety.unsafe():
            self.assertIsNone(safety.effective_rtol((np.zeros(3),)))

    def test_checks_active(self):
        a = np.zeros(3)
        self.assertTrue(safety.checks_active(a))                  # default safe, eager
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
        b = (np.ones((2, 3)), (np.arange(6.0).reshape(2, 3),))    # value-equal, different objects
        self.assertIsNot(a, b)
        self.assertTrue(safety.frames_equal(a, b))                # the jit round-trip case -> accepted

    def test_genuinely_different(self):
        self.assertFalse(safety.frames_equal((np.ones((2, 3)),), (np.zeros((2, 3)),)))

    def test_different_shapes(self):
        self.assertFalse(safety.frames_equal((np.ones((2, 3)),), (np.ones((2, 4)),)))

    def test_or_skip_unsafe(self):
        a, b = (np.ones((2, 3)),), (np.zeros((2, 3)),)            # different
        self.assertFalse(safety.frames_equal_or_skip(a, b))      # safe -> the real (False) result
        with safety.unsafe():
            self.assertTrue(safety.frames_equal_or_skip(a, b))   # unsafe -> skipped (passes)


@unittest.skipUnless(HAS_JAX, 'jax not available')
class TestJaxTolerance(unittest.TestCase):
    def test_effective_rtol_picks_jax(self):
        self.assertEqual(safety.effective_rtol((np.zeros(3),)), safety.DEFAULT_RTOL_NUMPY)
        self.assertEqual(safety.effective_rtol((jnp.zeros(3),)), safety.DEFAULT_RTOL_JAX)  # jax -> looser

    def test_frames_equal_jax_uses_loose_tol(self):
        # two jax frames differing by ~1e-7 (float32-scale noise): rejected by the numpy tol, accepted by jax
        a = (jnp.ones((2, 3)),)
        b = (jnp.ones((2, 3)) + 1e-7,)
        self.assertTrue(safety.frames_equal(a, b))                      # jax-aware tol (1e-5) -> equal
        self.assertFalse(safety.frames_equal(a, b, rtol=safety.DEFAULT_RTOL_NUMPY))  # numpy tol -> different


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
        out = {}
        def f(x):
            out['active'] = safety.checks_active(x)
            out['or_skip'] = safety.frames_equal_or_skip((x,), (jnp.zeros(3),))  # would be False eagerly
            return x
        jax.jit(f)(jnp.ones(3))
        self.assertFalse(out['active'])                          # safe mode, but tracing -> inactive
        self.assertTrue(out['or_skip'])                          # skipped under trace -> passes

    def test_checks_skip_under_trace_closed_over_concrete(self):
        # the subtle case: the CHECKED operand is a closed-over CONCRETE array (not a tracer), but we are
        # globally inside a trace -- a jnp op then still yields a (constant) tracer, so checks must skip.
        const = jnp.ones(3)                                      # committed concrete array, closed over
        out = {}
        def f(y):                                                # y is the only traced arg
            out['tracing'] = safety.is_tracing(const)            # const is concrete, yet we ARE tracing
            out['active'] = safety.checks_active(const)
            return y
        jax.jit(f)(jnp.zeros(2))
        self.assertTrue(out['tracing'])                          # global-trace detection, not input-tracer
        self.assertFalse(out['active'])                          # so the check correctly skips
        self.assertFalse(safety.is_tracing(const))               # eager again: not tracing


if __name__ == '__main__':
    unittest.main()
