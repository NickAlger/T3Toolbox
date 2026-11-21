# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The frontend sampling methods (graded messages, P13a): the merged signatures on the four classes and the six new
corewise methods of the two tensor-train classes.

- **each method is its backend function** with ``self.data`` (or the frame's), at ``order = ()`` and ``(2,)``;
- **the frontend's checks:** ``jets.check_sample`` through every method (the wrong number of modes, a mismatched
  ``W``, too few axes for the jet shape, an integer ``order``, a non-integer index); a packed sample on the uniform
  classes is the engine's; a direction is a pair of core sequences (or two supercores) and anything else raises a
  ``TypeError``; a nested-list index is converted;
- **the new corewise methods:** the Hessian is symmetric, and is the derivative of the corewise transpose along the
  direction, by a polynomial-exact quotient (``J^T c`` is a polynomial of degree at most ``2d - 1`` in the cores
  along a line, so a fit on ``2d`` points is exact to rounding); the uniform ones mirror the sample's form and read
  no padding;
- **the sum flag:** ``False`` keeps ``W``, and summed over it is the default;
- **a guard on the test design:** at ``d = 3`` the probe corewise and ambient transposes with ``ww`` and the residual
  swapped give different numbers, so no call-site test here rests on a symmetric problem.

numpy only (jax dispatch is ``tests/test_dispatch.py``'s).
"""
import inspect
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.jets as jets
import t3toolbox.backend.ut3_operations as ut3o
from t3toolbox.backend import apply as bapply
from t3toolbox.backend import entries as bentries
from t3toolbox.backend import probing

STRUCT = ((4, 5, 3), (2, 3, 2), (1, 2, 2, 1))
W = (3,)
KINDS = ('probe', 'apply', 'entries')
MODULES = {'probe': probing, 'apply': bapply, 'entries': bentries}


def leaves(x):
    if isinstance(x, (tuple, list)):
        return [a for item in x for a in leaves(item)]
    return [np.asarray(x)]


def relerr(a, b):
    num = np.sqrt(sum(np.sum((p - q) ** 2) for p, q in zip(leaves(a), leaves(b))))
    return float(num / max(np.sqrt(sum(np.sum(q ** 2) for q in leaves(b))), 1e-300))


def dot(a, b):
    return float(sum(np.vdot(p, q) for p, q in zip(leaves(a), leaves(b))))


def sample_of(kind, rng, order=(), shape=STRUCT[0]):
    if kind == 'entries':
        return np.stack([rng.integers(0, N, size=W) for N in shape])
    ww = [rng.standard_normal(W + (N,)) for N in shape]
    return ww if order == () else list(jets.affine_jet(ww, [rng.standard_normal(W + (N,)) for N in shape]))


def residual_like(kind, rng, z):
    return [rng.standard_normal(np.shape(a)) for a in z] if kind == 'probe' else rng.standard_normal(np.shape(z))


def okw(kind, order):
    return {} if kind == 'entries' else {'order': order}


class Fixture(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(31)
        np.random.seed(31)
        self.x = t3.TuckerTensorTrain.randn(*STRUCT)
        self.y = t3.TuckerTensorTrain.randn(*STRUCT)                    # a direction of the cores' shapes
        self.frame = t3m.MANIFOLD.frame(self.x)
        self.v = t3m.COREWISE.randn(self.frame)
        self.xu = ut3.UniformTuckerTensorTrain.from_t3(self.x, N=6, n=4, r=4)
        self.yu = ut3.UniformTuckerTensorTrain.from_t3(self.y, N=6, n=4, r=4)
        self.vu = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(self.xu))


class TestEachMethodIsItsBackendFunction(Fixture):

    def test_the_tensor_train(self):
        x, y = self.x, self.y
        for kind in KINDS:
            m = MODULES[kind]
            for order in ([(), (2,)] if kind != 'entries' else [()]):
                with self.subTest(kind=kind, order=order):
                    s, kw = sample_of(kind, self.rng, order), okw(kind, order)
                    z = getattr(x, kind)(s, **kw)
                    self.assertEqual(relerr(z, getattr(m, 't3_%s' % kind)(x.data, s, **kw)), 0.0)
                    c = residual_like(kind, self.rng, z)
                    pairs = [
                        (getattr(x, '%s_corewise_tangent' % kind)(s, y.data, **kw),
                         getattr(m, 't3_%s_corewise_tangent' % kind)(x.data, s, y.data, **kw)),
                        (getattr(x, '%s_corewise_transpose' % kind)(s, residual=c, **kw),
                         getattr(m, 't3_%s_corewise_transpose' % kind)(x.data, s, c, **kw)),
                        (getattr(x, '%s_corewise_hessian' % kind)(s, residual=c, core_directions=y.data, **kw),
                         getattr(m, 't3_%s_corewise_hessian' % kind)(x.data, s, c, y.data, **kw)),
                    ]
                    for got, ref in pairs:
                        self.assertEqual(relerr(got, ref), 0.0)

    def test_the_tangent(self):
        v, fd = self.v, self.frame.data
        for kind in KINDS:
            m = MODULES[kind]
            for order in ([(), (2,)] if kind != 'entries' else [()]):
                with self.subTest(kind=kind, order=order):
                    s, kw = sample_of(kind, self.rng, order), okw(kind, order)
                    Jp = getattr(v, kind)(s, **kw)
                    self.assertEqual(relerr(Jp, getattr(m, 'tv_%s' % kind)(fd, v.variations.data, s, **kw)), 0.0)
                    c = residual_like(kind, self.rng, Jp)
                    t = getattr(t3m.T3Tangent, '%s_transpose' % kind)(self.frame, s, residual=c, **kw)
                    self.assertIsInstance(t, t3m.T3Tangent)
                    self.assertEqual(relerr(t.variations.data, getattr(m, 'tv_%s_transpose' % kind)(fd, s, c, **kw)), 0.0)

    def test_the_uniform_classes(self):
        """Against the ragged methods on the real parts, with a ragged and a packed sample (probe mirrors)."""
        xu, yu, x, y = self.xu, self.yu, self.x, self.y
        for kind in KINDS:
            for order in ([(), (2,)] if kind != 'entries' else [()]):
                s, kw = sample_of(kind, self.rng, order), okw(kind, order)
                forms = [('ragged', s)] + ([] if kind == 'entries' else
                                           [('packed', ut3o.pack_vectors(s, 6, len_J=len(order)))])
                for form, su in forms:
                    with self.subTest(kind=kind, order=order, sample=form):
                        for name, args in (('', ()), ('_corewise_tangent', (None,))):
                            ref = getattr(x, kind + name)(s, *([y.data] if args else []), **kw)
                            got = getattr(xu, kind + name)(su, *([yu.data[:2]] if args else []), **kw)
                            if kind == 'probe' and form == 'packed':
                                self.assertTrue(ut3o.is_packed(got))
                                got = ut3o.unpack_vectors(got, x.shape)
                            elif kind == 'probe':
                                self.assertIsInstance(got, tuple)
                            self.assertLess(relerr(got, ref), 1e-12)


class TestTheChecks(Fixture):

    def methods(self, kind, order=()):
        """Every frontend method of the kind that takes a sample, as a function of the sample."""
        x, xu, v, vu, fr = self.x, self.xu, self.v, self.vu, self.frame
        kw = okw(kind, order)
        c = np.zeros(W) if kind != 'probe' else [np.zeros(W + (N,)) for N in STRUCT[0]]
        T = t3m.T3Tangent if True else None
        out = {
            'TT.%s' % kind: lambda s: getattr(x, kind)(s, **kw),
            'TT.%s_corewise_tangent' % kind: lambda s: getattr(x, kind + '_corewise_tangent')(s, self.y.data, **kw),
            'TT.%s_corewise_transpose' % kind: lambda s: getattr(x, kind + '_corewise_transpose')(s, residual=c, **kw),
            'TT.%s_corewise_hessian' % kind: lambda s: getattr(x, kind + '_corewise_hessian')(
                s, residual=c, core_directions=self.y.data, **kw),
            'T3Tangent.%s' % kind: lambda s: getattr(v, kind)(s, **kw),
            'T3Tangent.%s_transpose' % kind: lambda s: getattr(T, kind + '_transpose')(fr, s, residual=c, **kw),
            'UT3.%s' % kind: lambda s: getattr(xu, kind)(s, **kw),
            'UT3Tangent.%s' % kind: lambda s: getattr(vu, kind)(s, **kw),
        }
        return out

    def test_a_malformed_sample_raises(self):
        for kind in ('probe', 'apply'):
            good = sample_of(kind, self.rng)
            bad = {
                'd - 1 modes': good[:2],
                'a mismatched W': [good[0], good[1][:2], good[2]],
                'too few axes for the jet shape': good,           # with order (1, 1): two jet axes, W, the mode
            }
            for label, s in bad.items():
                order = (1, 1) if label.startswith('too few') else ()
                for name, call in self.methods(kind, order).items():
                    with self.subTest(case=label, method=name):
                        with self.assertRaises(ValueError):
                            call(s)
        idx = sample_of('entries', self.rng)
        for label, s in {'d - 1 rows': idx[:2], 'a float index': idx.astype(float)}.items():
            for name, call in self.methods('entries').items():
                with self.subTest(case=label, method=name):
                    with self.assertRaises(ValueError):
                        call(s)

    def test_an_integer_order_raises_a_type_error(self):
        ww = sample_of('probe', self.rng)
        for call in (lambda: self.x.probe(ww, 2), lambda: self.x.apply(ww, 2), lambda: self.v.probe(ww, 2),
                     lambda: self.xu.probe(ww, 2)):
            with self.assertRaises(TypeError):
                call()

    def test_entries_takes_no_order_and_converts_a_nested_list(self):
        for owner in (t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain, t3m.T3Tangent, ut3m.UT3Tangent):
            for name in ('entries', 'entries_transpose', 'entries_corewise_transpose', 'entries_corewise_tangent',
                         'entries_corewise_hessian', 'entries_ambient_transpose'):
                if hasattr(owner, name):
                    with self.subTest(owner=owner.__name__, method=name):
                        self.assertNotIn('order', inspect.signature(getattr(owner, name)).parameters)
        idx = sample_of('entries', self.rng)
        as_list = idx.tolist()
        c = self.rng.standard_normal(W)
        pairs = [
            (self.x.entries(as_list), self.x.entries(idx)),
            (self.xu.entries(as_list), self.xu.entries(idx)),
            (self.v.entries(as_list), self.v.entries(idx)),
            (self.vu.entries(as_list), self.vu.entries(idx)),
            (self.x.entries_corewise_transpose(as_list, residual=c), self.x.entries_corewise_transpose(idx, residual=c)),
            (self.x.entries_corewise_tangent(as_list, self.y.data), self.x.entries_corewise_tangent(idx, self.y.data)),
            (t3.TuckerTensorTrain.entries_ambient_transpose(as_list, self.x.shape, residual=c),
             t3.TuckerTensorTrain.entries_ambient_transpose(idx, self.x.shape, residual=c)),
            (t3m.T3Tangent.entries_transpose(self.frame, as_list, residual=c).variations.data,
             t3m.T3Tangent.entries_transpose(self.frame, idx, residual=c).variations.data),
        ]
        for k, (a, b) in enumerate(pairs):
            with self.subTest(pair=k):
                self.assertEqual(relerr(a, b), 0.0)

    def test_a_direction_is_a_pair_of_core_sequences(self):
        ww = sample_of('probe', self.rng)
        c = [np.zeros(W + (N,)) for N in STRUCT[0]]
        wrong = {'TuckerTensorTrain': self.y, 'UniformTuckerTensorTrain': self.yu, 'T3Tangent': self.v,
                 'three parts': self.y.data + (self.y.data[0],), 'a pair of arrays': (np.zeros(3), np.zeros(3))}
        for label, d in wrong.items():
            for name, call in (('tangent', lambda: self.x.probe_corewise_tangent(ww, d)),
                               ('hessian', lambda: self.x.probe_corewise_hessian(ww, residual=c, core_directions=d))):
                with self.subTest(case=label, method=name):
                    with self.assertRaisesRegex(TypeError, r'pass y\.data'):
                        call()
        uwrong = {'UniformTuckerTensorTrain': self.yu, 'its whole .data': self.yu.data,
                  'TuckerTensorTrain': self.y, 'ragged core tuples': self.y.data}
        for label, d in uwrong.items():
            with self.subTest(case=label, method='uniform tangent'):
                with self.assertRaisesRegex(TypeError, r'pass y\.data\[:2\]'):
                    self.xu.probe_corewise_tangent(ww, d)
        self.x.probe_corewise_tangent(ww, list(self.y.data))                  # a list pair is fine
        self.xu.probe_corewise_tangent(ww, self.yu.data[:2])


class TestTheCorewiseHessianMethods(Fixture):

    def test_symmetric(self):
        """``<y, H z> = <z, H y>``."""
        np.random.seed(32)
        z = t3.TuckerTensorTrain.randn(*STRUCT)
        for kind in KINDS:
            with self.subTest(kind=kind):
                s = sample_of(kind, self.rng)
                c = residual_like(kind, self.rng, getattr(self.x, kind)(s))
                H = lambda d: getattr(self.x, kind + '_corewise_hessian')(s, residual=c, core_directions=d)
                self.assertLess(abs(dot(self.y.data, H(z.data)) - dot(z.data, H(self.y.data))),
                                1e-12 * max(abs(dot(self.y.data, H(z.data))), 1.0))

    def test_the_derivative_of_the_transpose_by_an_exact_polynomial_fit(self):
        """``g(t) = J^T c`` at ``x + t y`` is a polynomial of degree at most ``2d - 1`` in ``t``; a fit on ``2d``
        points is exact to rounding, and its slope at 0 is ``H y``."""
        d = len(STRUCT[0])
        ts = np.linspace(-0.5, 0.5, 2 * d)
        for kind in KINDS:
            for order in ([(), (2,)] if kind != 'entries' else [()]):
                with self.subTest(kind=kind, order=order):
                    s, kw = sample_of(kind, self.rng, order), okw(kind, order)
                    c = residual_like(kind, self.rng, getattr(self.x, kind)(s, **kw))
                    at = lambda t: t3.TuckerTensorTrain(*[tuple(a + t * b for a, b in zip(p, q))
                                                         for p, q in zip(self.x.data, self.y.data)])
                    G = [leaves(getattr(at(t), kind + '_corewise_transpose')(s, residual=c, **kw)) for t in ts]
                    slope = []
                    for j in range(len(G[0])):
                        stack = np.stack([g[j].ravel() for g in G])                    # (2d, size)
                        coef = np.polynomial.polynomial.polyfit(ts, stack, 2 * d - 1)  # (2d, size)
                        slope.append(coef[1].reshape(G[0][j].shape))
                    H = getattr(self.x, kind + '_corewise_hessian')(s, residual=c, core_directions=self.y.data, **kw)
                    self.assertLess(relerr(H, slope), 1e-9)


class TestTheSumFlag(Fixture):

    def test_unsummed_sums_to_the_default(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                s = sample_of(kind, self.rng)
                c = residual_like(kind, self.rng, getattr(self.x, kind)(s))
                calls = [          # (the call, the axis of W: 0 on core tuples, 1 on supercores, after d)
                    (lambda f: getattr(self.x, kind + '_corewise_transpose')(s, residual=c, sum_over_samples=f), 0),
                    (lambda f: getattr(self.x, kind + '_corewise_hessian')(s, residual=c, core_directions=self.y.data,
                                                                          sum_over_samples=f), 0),
                    (lambda f: getattr(t3m.T3Tangent, kind + '_transpose')(self.frame, s, residual=c,
                                                                           sum_over_samples=f).variations.data, 0),
                    (lambda f: getattr(self.xu, kind + '_corewise_transpose')(s, residual=c, sum_over_samples=f), 1),
                ]
                for k, (call, axis) in enumerate(calls):
                    kept, summed = leaves(call(False)), leaves(call(True))
                    self.assertTrue(all(a.shape[axis:axis + len(W)] == W for a in kept), k)
                    self.assertLess(relerr([a.sum(axis) for a in kept], summed), 1e-13, k)


class TestNoSymmetricProblem(Fixture):

    def test_swapping_the_sample_and_the_residual_changes_the_result(self):
        """At ``d = 3`` the probe corewise and ambient transposes are not symmetric in ``ww`` and the residual: a
        call with the two swapped (an old positional call) gives different numbers."""
        ww, zt = sample_of('probe', self.rng), sample_of('probe', self.rng)
        right = self.x.probe_corewise_transpose(ww, residual=zt)
        swapped = self.x.probe_corewise_transpose(zt, residual=ww)
        self.assertGreater(relerr(swapped, right), 1e-2)
        right = t3.TuckerTensorTrain.probe_ambient_transpose(ww, residual=zt, sum_over_samples=True)
        swapped = t3.TuckerTensorTrain.probe_ambient_transpose(zt, residual=ww, sum_over_samples=True)
        dense = lambda f: t3.TuckerTensorTrain.from_canonical(f).to_dense()
        self.assertGreater(relerr(dense(swapped), dense(right)), 1e-2)


if __name__ == '__main__':
    unittest.main()
