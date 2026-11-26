# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The transposes typed (the data-metric arc, M4): their argument is ``c``, an element of ``Y``.

- **The signatures**: each of the 42 transposes and second-order transposes (backend and frontend, ragged and uniform)
  takes ``c`` and no ``residual``.
- **The laws** (``tests/_typing_laws.py``) on all 42: the input law ``w_i -> Delta_i w_i`` (``c`` moves as an
  element of ``Y``: ``Delta_i`` on a probe edge, nothing for apply, ``prod_i Delta_i[n_i]`` for entries; the output,
  ``X*``, by ``Delta_i`` on the mode axis; the ambient output, ``E*``, by ``Delta_0 (x) ... (x) Delta_(d-1)``), and the
  jet law ``s_j -> ell_j s_j`` on the 24 with jets (``c`` by ``ell^-k``, the output fixed), at ``(2,)`` and ``(1, 1)``.
  Homogeneity tests: necessary, not sufficient; the adjoint and dense tests stay the primary ones.
- **The gradient of a weighted fit on a drawn batch**: ``<J^T(batch.data_metric(r)), x_dot>`` against a central
  difference of the dense misfit ``1/2 gamma sum omega**2 r**2`` (``tests/oracle_sweeps/oracle.py``), one core at a
  time (the misfit is then quadratic in ``t`` and the central difference exact to rounding), every kind.

The moved frames are not orthonormal, so the frontend calls run under ``safety.unsafe()`` (the transposes are
metric-free). numpy only (jax dispatch is ``tests/test_dispatch.py``'s).
"""
import inspect
import os
import sys
import unittest

import numpy as np

import t3toolbox.safety as safety
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.backend.apply as bapply
import t3toolbox.backend.entries as bentries
import t3toolbox.backend.probing as bprobing
import t3toolbox.backend.ut3_sampling as ut3s
import t3toolbox.backend.utv_sampling as utvs
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch

from _typing_laws import JetLaw, InputLaw, assert_law

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oracle_sweeps'))
import oracle as dense_oracle                    # noqa: E402  the dense misfit (misfit_full) and S_taylor

SHAPE, TUCKER, TT = (4, 5, 3), (2, 3, 2), (1, 2, 2, 1)
W = (3,)
KINDS = ('probe', 'apply', 'entries')
BACKEND = {'probe': bprobing, 'apply': bapply, 'entries': bentries}
JET_ORDERS = {(2,): (0.7,), (1, 1): (0.7, 1.9)}

# the 42, by owner: the backend functions by name, the frontend methods by class
FORTY_TWO = (
    [(BACKEND[k], 'tv_%s_transpose' % k) for k in KINDS]
    + [(BACKEND[k], 't3_%s_corewise_transpose' % k) for k in KINDS]
    + [(BACKEND[k], 't3_%s_corewise_hessian' % k) for k in KINDS]
    + [(BACKEND[k], 't3_%s_ambient_transpose' % k) for k in KINDS]
    + [(ut3s, 'ut3_%s_corewise_transpose' % k) for k in KINDS]
    + [(ut3s, 'ut3_%s_corewise_hessian' % k) for k in KINDS]
    + [(utvs, 'utv_%s_transpose' % k) for k in KINDS]
    + [(t3.TuckerTensorTrain, '%s_%s' % (k, f)) for k in KINDS
       for f in ('ambient_transpose', 'corewise_transpose', 'corewise_hessian')]
    + [(ut3.UniformTuckerTensorTrain, '%s_%s' % (k, f)) for k in KINDS for f in ('corewise_transpose', 'corewise_hessian')]
    + [(t3m.T3Tangent, '%s_transpose' % k) for k in KINDS]
    + [(ut3m.UT3Tangent, '%s_transpose' % k) for k in KINDS])


def cp_dense(factors):
    """The dense tensor of CP factors ``len=d, elm_shape=stack+(R, Ni)``: ``stack + (N0, ..., N(d-1))``."""
    d = len(factors)
    letters = 'abcdefgh'[:d]
    spec = ','.join('...r' + l for l in letters) + '->...' + letters
    return np.einsum(spec, *[np.asarray(f) for f in factors])


class TestTheSignatures(unittest.TestCase):

    def test_each_of_the_42_takes_c(self):
        self.assertEqual(len(FORTY_TWO), 42)
        self.assertEqual(len({(id(owner), name) for owner, name in FORTY_TWO}), 42)
        for owner, name in FORTY_TWO:
            with self.subTest(owner=getattr(owner, '__name__', str(owner)), name=name):
                params = inspect.signature(getattr(owner, name)).parameters
                self.assertIn('c', params)
                self.assertNotIn('residual', params)

    def test_the_old_keyword_is_refused(self):
        """No shim: ``residual=`` is an unknown keyword (positional calls are unaffected)."""
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        ww = [np.ones((2, n)) for n in SHAPE]
        with self.assertRaises(TypeError):
            x.apply_corewise_transpose(ww, residual=np.ones(2))
        with self.assertRaises(TypeError):
            bapply.tv_apply_transpose(t3m.MANIFOLD.frame(x).data, ww, residual=np.ones(2))


class Fixture(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(17)
        np.random.seed(17)
        self.x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        self.frame = t3m.MANIFOLD.frame(self.x).data
        self.dirs = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        self.law = InputLaw([0.5 + 1.5 * self.rng.random(n) for n in SHAPE])

    def sample(self, kind, order):
        if kind == 'entries':
            return np.stack([self.rng.integers(0, n, size=W) for n in SHAPE])
        return [self.rng.standard_normal((2,) * len(order) + W + (n,)) for n in SHAPE]

    def c(self, kind, order):
        F = tuple(n + 1 for n in order)
        if kind == 'probe':
            return [self.rng.standard_normal(F + W + (n,)) for n in SHAPE]
        return self.rng.standard_normal((F + W) if kind == 'apply' else W)


def _uframe(frame):
    return ubv.UT3Frame.from_t3frame(bvf.T3Frame(*frame))


def _upoint(x):
    return ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain(*x))


def families(kind, order):
    """The 42 as callables ``f(base, sample, c, dirs)`` with plain-array outputs, and how each moves under the input law:
    ``(name, base ('frame' | 'point' | None), f, output mover name)``. ``order`` only for the kinds with jets."""
    o = {} if kind == 'entries' else {'order': order}
    m = BACKEND[kind]
    T, UT = t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain
    out = [
        ('tv', 'frame', lambda b, s, c, dd: getattr(m, 'tv_%s_transpose' % kind)(b, s, c, **o), 'variations'),
        ('t3 corewise transpose', 'point',
         lambda b, s, c, dd: getattr(m, 't3_%s_corewise_transpose' % kind)(b, s, c, **o), 'tucker'),
        ('t3 corewise hessian', 'point',
         lambda b, s, c, dd: getattr(m, 't3_%s_corewise_hessian' % kind)(b, s, c, dd, **o), 'tucker'),
        ('utv', 'frame', lambda b, s, c, dd: getattr(utvs, 'utv_%s_transpose' % kind)(_uframe(b).data, s, c, **o)[:2],
         'packed'),
        ('ut3 corewise transpose', 'point',
         lambda b, s, c, dd: getattr(ut3s, 'ut3_%s_corewise_transpose' % kind)(_upoint(b).data, s, c, **o), 'packed'),
        ('ut3 corewise hessian', 'point',
         lambda b, s, c, dd: getattr(ut3s, 'ut3_%s_corewise_hessian' % kind)(_upoint(b).data, s, c,
                                                                            _upoint(dd).data[:2], **o), 'packed'),
        ('T3Tangent', 'frame',
         lambda b, s, c, dd: getattr(t3m.T3Tangent, '%s_transpose' % kind)(bvf.T3Frame(*b), s, c, **o).variations.data,
         'variations'),
        ('UT3Tangent', 'frame',
         lambda b, s, c, dd: getattr(ut3m.UT3Tangent, '%s_transpose' % kind)(_uframe(b), s, c, **o).variations.data[:2],
         'packed'),
        ('TT corewise transpose', 'point',
         lambda b, s, c, dd: getattr(T(*b), '%s_corewise_transpose' % kind)(s, c, **o), 'tucker'),
        ('TT corewise hessian', 'point',
         lambda b, s, c, dd: getattr(T(*b), '%s_corewise_hessian' % kind)(s, c, dd, **o), 'tucker'),
        ('UTT corewise transpose', 'point',
         lambda b, s, c, dd: getattr(_upoint(b), '%s_corewise_transpose' % kind)(s, c, **o), 'packed'),
        ('UTT corewise hessian', 'point',
         lambda b, s, c, dd: getattr(_upoint(b), '%s_corewise_hessian' % kind)(s, c, _upoint(dd).data[:2], **o),
         'packed'),
    ]
    if kind == 'entries':
        out += [('t3 ambient', None,
                 lambda b, s, c, dd: cp_dense(bentries.t3_entries_ambient_transpose(s, SHAPE, c)), 'ambient'),
                ('TT ambient', None,
                 lambda b, s, c, dd: cp_dense(T.entries_ambient_transpose(s, SHAPE, c)), 'ambient')]
    else:
        out += [('t3 ambient', None,
                 lambda b, s, c, dd: cp_dense(getattr(m, 't3_%s_ambient_transpose' % kind)(s, c)), 'ambient'),
                ('TT ambient', None,
                 lambda b, s, c, dd: cp_dense(getattr(T, '%s_ambient_transpose' % kind)(s, c)), 'ambient')]
    return out


class TestTheInputLaw(Fixture):
    """``w_i -> Delta_i w_i`` with every represented value fixed: each of the 42 commutes with it."""

    def test_all_42(self):
        law, n = self.law, 0
        for kind in KINDS:
            order = () if kind == 'entries' else (2,)
            for name, base, f, mover in families(kind, order):
                ambient = mover == 'ambient'
                s = self.sample(kind, () if ambient else order)
                c = self.c(kind, () if ambient else order)
                b = {'frame': self.frame, 'point': self.x.data, None: None}[base]
                moved_b = {'frame': law.frame, 'point': law.tucker, None: lambda v: None}[base](b)
                moved_s = s if kind == 'entries' else law.inputs(s)
                moved_c = (law.entries_dual(c, s) if kind == 'entries' else law.V(c) if kind == 'probe' else c)
                moved_dirs = law.tucker(self.dirs)
                move_out = {'variations': law.variations_dual,
                            'tucker': lambda out: (law.tucker_dual(out[0]), out[1]),
                            'packed': lambda out: (law.packed_mode_dual(out[0]), out[1]),
                            'ambient': law.ambient}[mover]
                with self.subTest(kind=kind, function=name):
                    with safety.unsafe():
                        assert_law(self, f, (b, s, c, self.dirs), (moved_b, moved_s, moved_c, moved_dirs), move_out,
                                   rtol=1e-11)
                    n += 1
        self.assertEqual(n, 42)


class TestTheJetLaw(Fixture):
    """``s_j -> ell_j s_j``: the inputs by ``ell^k``, ``c`` (an element of ``P*``) by ``ell^-k``, the output fixed; on
    the 24 with jets, at ``(2,)`` and ``(1, 1)``."""

    def test_the_24_with_jets(self):
        n = 0
        for order, ell in JET_ORDERS.items():
            law = JetLaw(order, ell)
            for kind in ('probe', 'apply'):
                for name, base, f, mover in families(kind, order):
                    if mover == 'ambient':
                        continue                                     # no jets: the law is vacuous
                    s, c = self.sample(kind, order), self.c(kind, order)
                    b = self.frame if base == 'frame' else self.x.data
                    with self.subTest(order=order, kind=kind, function=name):
                        with safety.unsafe():
                            assert_law(self, f, (b, s, c, self.dirs), (b, law.inputs(s), law.P_star(c), self.dirs),
                                       lambda out: out, rtol=1e-11)
                        n += 1
        self.assertEqual(n, 2 * 24)


class TestTheGradientOfAWeightedFit(unittest.TestCase):
    """On a drawn batch (``take``, scale ``gamma = n / b``) with a weight per entry, zeros included: the corewise
    transpose of ``batch.data_metric(r)`` is the gradient of the dense misfit ``1/2 gamma sum omega**2 r**2``. Checked
    along a direction in one core at a time, where the misfit is quadratic in ``t`` (the tensor is linear in each
    core), so the central difference is exact to rounding."""

    def test_every_kind(self):
        rng = np.random.default_rng(23)
        np.random.seed(23)
        Wn, indices = (6,), np.array([0, 2, 5])
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        for kind, order in (('probe', (2,)), ('apply', (2,)), ('entries', ())):
            F = tuple(n + 1 for n in order)
            if kind == 'entries':
                sample = np.stack([rng.integers(0, n, size=Wn) for n in SHAPE])
            else:
                sample = [np.stack([rng.standard_normal(Wn + (n,)), rng.standard_normal(Wn + (n,))]) for n in SHAPE]
            shapes = [F + Wn + (n,) for n in SHAPE] if kind == 'probe' else [(F + Wn) if kind == 'apply' else Wn]
            y = [rng.standard_normal(s) for s in shapes]
            weight = [0.5 + rng.random(s) for s in shapes]
            for w in weight:
                w.reshape(-1)[::5] = 0.0
            y, weight = (y, weight) if kind == 'probe' else (y[0], weight[0])
            oracle = LeastSquaresOracle()
            drawn = LeastSquaresBatch(kind, order, sample, y, weight=weight).take(indices)
            gamma = float(drawn.minibatch_scale)
            forward = (lambda T: T.entries(drawn.sample)) if kind == 'entries' else \
                (lambda T: getattr(T, kind)(drawn.sample, order))
            z = forward(x)
            r = [zi - yi for zi, yi in zip(z, drawn.y)] if kind == 'probe' else z - drawn.y
            o = {} if kind == 'entries' else {'order': order}
            gB, gG = getattr(x, '%s_corewise_transpose' % kind)(drawn.sample, drawn.data_metric(r), **o)
            dense_sample = drawn.sample if kind == 'entries' else ([s[0] for s in drawn.sample],
                                                                     [s[1] for s in drawn.sample])

            def misfit(cores):
                X = t3.TuckerTensorTrain(*cores).to_dense()
                return gamma * dense_oracle.misfit_full(kind, X, dense_sample, drawn.y, drawn.weight, order)

            self.assertEqual(gamma, 2.0)
            for family, blocks in ((0, gB), (1, gG)):
                for i in range(len(SHAPE)):
                    with self.subTest(kind=kind, core=('tucker', 'tt')[family] + ' %d' % i):
                        direction = rng.standard_normal(np.shape(x.data[family][i]))
                        h = 1e-2

                        def moved(t):
                            cores = [list(x.data[0]), list(x.data[1])]
                            cores[family][i] = cores[family][i] + t * direction
                            return tuple(cores[0]), tuple(cores[1])
                        slope = (misfit(moved(h)) - misfit(moved(-h))) / (2 * h)
                        pairing = float(np.sum(np.asarray(blocks[i]) * direction))
                        self.assertLess(abs(slope - pairing), 1e-9 * max(abs(pairing), 1.0))


if __name__ == '__main__':
    unittest.main()
