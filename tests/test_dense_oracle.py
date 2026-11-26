# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``tests/oracle_sweeps/oracle.py``, its full form (the data-metric arc, M2 part 1), pinned before anything is tested
against it.

The full form: the data ``y`` and the weight ``omega`` are of the type of ``z`` (each axis 1 or ``z``'s size, read by
position), in Taylor coordinates. Three lines of evidence, none of which calls the library:

1. ``expand`` against an explicit loop over every entry, and its refusals;
2. ``S_taylor`` along a line against ``S`` evaluated on the line: ``sum_t c_t s^t == S(main + s pp)`` at points that
   are not the interpolation's nodes;
3. ``misfit_full`` against an explicit loop over every entry of ``z`` (the weight and the data read at index 0 on
   their size-1 axes), for a weight per entry, per sample, per slot, per mode, one number, and none; and against the
   old form's ``misfit`` of derivative data, ``omega_t = w_t t!`` (a derivative is ``t!`` times its Taylor coefficient).

numpy only.
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oracle_sweeps'))
import oracle                                   # noqa: E402

SHAPE = (3, 4, 2)
W = (5,)


def at(a, index):
    """``a`` at ``index``, reading index 0 on every size-1 axis of ``a``: the full form's rule, by hand."""
    return a[tuple(0 if n == 1 else k for n, k in zip(a.shape, index))]


def problem(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(SHAPE)
    ww = [rng.standard_normal(W + (n,)) for n in SHAPE]
    pp = [rng.standard_normal(W + (n,)) for n in SHAPE]
    index = np.stack([rng.integers(0, n, size=W) for n in SHAPE])
    return rng, X, ww, pp, index


def sample_of(kind, ww, pp, index, order):
    main = index if kind == 'entries' else ww
    return main if order == () else (main, pp)


def pattern_shape(shape, full_axes):
    return tuple(n if a in full_axes else 1 for a, n in enumerate(shape))


class TestExpand(unittest.TestCase):

    def test_expand_writes_out_the_size_one_axes(self):
        rng = np.random.default_rng(0)
        for shape, small in (((3, 2, 4), (3, 1, 4)), ((3, 2, 4), (1, 1, 1)), ((2, 2), (2, 1)), ((2, 2), (1, 2)),
                             ((), ())):
            with self.subTest(shape=shape, small=small):
                a = rng.standard_normal(small)
                got = oracle.expand(a, shape)
                self.assertEqual(got.shape, shape)
                for index in np.ndindex(*shape):
                    self.assertEqual(got[index], at(a, index))

    def test_expand_refuses_another_rank_or_size(self):
        for small, shape in (((3,), (3, 2)), ((3, 2), (3,)), ((2, 2), (3, 2)), ((3, 2), (3, 4))):
            with self.subTest(small=small, shape=shape):
                with self.assertRaises(ValueError):
                    oracle.expand(np.ones(small), shape)


class TestTaylor(unittest.TestCase):

    def test_the_taylor_coefficients_sum_to_the_output_on_the_line(self):
        _, X, ww, pp, index = problem(1)
        onehots = oracle.onehots(index, SHAPE)
        for kind in ('probe', 'apply', 'entries'):
            for n in (1, 2, 3):
                main = onehots if kind == 'entries' else ww
                coefficients = oracle.S_taylor(kind, X, sample_of(kind, ww, pp, index, (n,)), (n,))
                for s in (0.37, -1.3):
                    with self.subTest(kind=kind, n=n, s=s):
                        on_line = oracle.S('probe' if kind == 'probe' else 'apply', X,
                                           [w + s * p for w, p in zip(main, pp)])
                        for c, value in zip(coefficients if kind == 'probe' else [coefficients],
                                            on_line if kind == 'probe' else [on_line]):
                            total = sum(c[t] * s ** t for t in range(n + 1))
                            # the degree is d = 3 in s: from n = 3 on the sum is the whole polynomial; below it, the
                            # truncation, whose remainder is the higher coefficients
                            if n >= 3:
                                self.assertTrue(np.allclose(total, value, rtol=1e-12, atol=1e-12))
                        if n == 3:
                            continue
                        full = oracle.S_taylor(kind, X, sample_of(kind, ww, pp, index, (3,)), (3,))
                        for c, c3 in zip(coefficients if kind == 'probe' else [coefficients],
                                         full if kind == 'probe' else [full]):
                            self.assertTrue(np.allclose(c, c3[:n + 1], rtol=1e-12, atol=1e-12))

    def test_order_empty_is_the_plain_output(self):
        _, X, ww, pp, index = problem(2)
        for kind in ('probe', 'apply', 'entries'):
            with self.subTest(kind=kind):
                sample = index if kind == 'entries' else ww
                a, b = oracle.S_taylor(kind, X, sample), oracle.S(kind, X, sample)
                for u, v in zip(a if kind == 'probe' else [a], b if kind == 'probe' else [b]):
                    self.assertTrue(np.array_equal(u, v))


class TestMisfitFull(unittest.TestCase):

    def loop(self, kind, z, y, weight):
        """``1/2 sum omega**2 (z - y)**2`` by an explicit loop over every entry of ``z``."""
        zs = list(z) if kind == 'probe' else [z]
        ys = [y[i] for i in range(len(zs))] if kind == 'probe' else [y]
        ws = [None] * len(zs) if weight is None else ([weight[i] for i in range(len(zs))] if kind == 'probe'
                                                        else [weight])
        total = 0.0
        for z_, y_, w_ in zip(zs, ys, ws):
            y_ = np.asarray(y_)
            for index in np.ndindex(*z_.shape):
                omega = 1.0 if w_ is None else float(at(np.asarray(w_), index))
                total += 0.5 * omega ** 2 * (z_[index] - at(y_, index)) ** 2
        return total

    def test_against_the_loop_for_every_pattern(self):
        rng, X, ww, pp, index = problem(3)
        for kind in ('probe', 'apply', 'entries'):
            for order in (((), (2,)) if kind != 'entries' else ((),)):
                sample = sample_of(kind, ww, pp, index, order)
                z = oracle.S_taylor(kind, X, sample, order)
                shapes = [a.shape for a in z] if kind == 'probe' else [z.shape]
                m = len(order)
                patterns = {'per entry': lambda sh: set(range(len(sh))),
                            'per sample': lambda sh: set(range(m, m + len(W))),
                            'per slot': lambda sh: set(range(m)),
                            'per mode, or one number': lambda sh: set(),
                            'the edge and the first slot axis': lambda sh: {len(sh) - 1} | set(range(min(m, 1)))}
                for name, full_axes in patterns.items():
                    for y_constant in (False, True):
                        with self.subTest(kind=kind, order=order, weight=name, y_constant_in_W=y_constant):
                            weight = [0.5 + rng.random(pattern_shape(sh, full_axes(sh))) for sh in shapes]
                            y_axes = lambda sh: set(range(len(sh))) - (set(range(m, m + len(W))) if y_constant
                                                                        else set())
                            y = [rng.standard_normal(pattern_shape(sh, y_axes(sh))) for sh in shapes]
                            if kind != 'probe':
                                weight, y = weight[0], y[0]
                            expected = self.loop(kind, z, y, weight)
                            got = oracle.misfit_full(kind, X, sample, y, weight, order)
                            self.assertLessEqual(abs(got - expected), 1e-12 * max(abs(expected), 1.0))
                            if kind == 'probe' and name != 'per entry':     # every mode the same shape: one array
                                if len({w.shape for w in weight}) == 1:
                                    stacked = oracle.misfit_full(kind, X, sample, y, np.stack(weight), order)
                                    self.assertEqual(stacked, got)
                with self.subTest(kind=kind, order=order, weight='None'):
                    y = [rng.standard_normal(sh) for sh in shapes]
                    y = y if kind == 'probe' else y[0]
                    self.assertLessEqual(abs(oracle.misfit_full(kind, X, sample, y, None, order)
                                             - self.loop(kind, z, y, None)), 1e-12)

    def test_a_block_weight_is_refused(self):
        _, X, ww, pp, index = problem(4)
        for kind, weight in (('probe', np.ones((3, 3))), ('apply', np.ones(3)), ('probe', [np.ones(3)] * 3)):
            with self.subTest(kind=kind, shape=repr(np.shape(weight))):
                z = oracle.S_taylor(kind, X, (ww, pp), (2,))
                with self.assertRaises(ValueError):
                    oracle.misfit_full(kind, X, (ww, pp), z, weight, (2,))
        with self.assertRaises(ValueError):                         # a mode count other than d
            oracle.misfit_full('probe', X, ww, oracle.S('probe', X, ww), [np.ones((1, 1))] * 2)

    def test_the_old_derivative_form_is_omega_t_times_t_factorial(self):
        rng, X, ww, pp, index = problem(5)
        n = 2
        factorials = np.array([math.factorial(t) for t in range(n + 1)], dtype=float)
        for kind in ('probe', 'apply', 'entries'):
            with self.subTest(kind=kind):
                sample = sample_of(kind, ww, pp, index, (n,))
                derivatives = oracle.S(kind + '_derivatives', X, sample, n)
                y_d = ([a + rng.standard_normal(a.shape) for a in derivatives] if kind == 'probe'
                       else derivatives + rng.standard_normal(derivatives.shape))
                w = 0.5 + rng.random(n + 1)                     # the old per-order weight, on derivatives
                old = oracle.misfit(kind + '_derivatives', X, sample, y_d, w, n)
                per_slot = lambda a, c: c.reshape((n + 1,) + (1,) * (np.ndim(a) - 1))
                y_t = ([a / per_slot(a, factorials) for a in y_d] if kind == 'probe'
                       else y_d / per_slot(y_d, factorials))
                omega = w * factorials
                weight = ([per_slot(a, omega) for a in y_d] if kind == 'probe' else per_slot(y_d, omega))
                new = oracle.misfit_full('apply' if kind == 'entries' else kind, X,
                                         (oracle.onehots(index, SHAPE), pp) if kind == 'entries' else sample,
                                         y_t, weight, (n,))
                self.assertLessEqual(abs(new - old), 1e-12 * old)


if __name__ == '__main__':
    unittest.main()
