# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The least-squares oracle and its batch (graded messages, P14), and the engine's two additions for it.

- **the engine:**
  - ``computed_names`` (P14b, in place of P14's ``keep=``) is exactly the public names of the rules the engine runs,
    for every kind, form, layout, route (``Vc``, a held ``Jp``, held ``r`` and ``mu``) and the oracle's want patterns;
    it raises as the engine does; the oracle holds ``HOLD`` intersected with it, with the values of a fresh call;
  - ``frame_form`` reads the four forms of a frame, including the count of four that means two things.
- **``query``:**
  - each output against the engine, on a manifold frame and a corewise point, ragged and uniform (``pack_batch``),
    at ``()``, ``(2,)`` and ``(1, 1)`` (entries ``()``);
  - ``y_block_sumsq`` and ``f_resolution`` by their formulas;
  - ``held`` exactly the hold-list names the call computed.
- **The direction on a corewise point:** ``core_directions``, ``variations`` renamed, both the same (and a rebuilt
  tuple of the same arrays), and both different (raises).
- **The minibatch scale (O-21):**
  - the mean over every subset of the batch size of the drawn ``f`` and ``g`` is the full one, exactly;
  - on a drawn batch ``r``, ``Jp`` and the block sums are plain, and ``f = 1/2 <r, r>``, ``<Jp, r> = <p, g>`` and
    ``Jp_norm_sq = <Jp, Jp>`` hold in the batch's inner product;
  - ``f_resolution`` is from the unscaled values, scaled once.
- **The batch:** ``take`` (scale, indices, jets skipped, packed), ``check_batch`` on each wrong part, the pytree round
  trip.
- **The oracle:**
  - hash and equality by value;
  - ``order`` required for every kind (O-44);
  - entries with jets refused (O-46);
  - the bridge against the old derivative kinds through the conversions.

numpy only, except the pytree round trip (jax dispatch is ``tests/test_dispatch.py``'s).
"""
import dataclasses as dc
import itertools
import unittest.mock
import math
import os
import sys
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.jets as jets
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.ut3_operations as ut3o
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
from t3toolbox.backend.uniform_fitting import pack_batch

SHAPE, RANKS = (4, 5, 3), ((2, 3, 2), (1, 2, 2, 1))
HOLD = ('xi', 'mu', 'nu', 'eta', 'r')
ORDERS = {'probe': [(), (2,), (1, 1)], 'apply': [(), (2,), (1, 1)], 'entries': [()]}
WANT = ('f', 'g', 'JtJp', 'Jp_norm_sq', 'Jp', 'r', 'r_block_sumsq', 'Kp')


def leaves(x):
    if isinstance(x, (tuple, list)):
        return [a for item in x for a in leaves(item)]
    return [np.asarray(x)]


def relerr(a, b):
    a, b = leaves(a), leaves(b)
    assert len(a) == len(b), (len(a), len(b))
    num = math.sqrt(sum(float(np.sum((p - q) ** 2)) for p, q in zip(a, b)))
    return num / max(math.sqrt(sum(float(np.sum(q ** 2)) for q in b)), 1e-300)


def jet_sample(kind, rng, order, n=6):
    if kind == 'entries':
        return np.stack([rng.integers(0, N, size=n) for N in SHAPE])
    sizes = {(): [()] * 3, (2,): [(3,), (2,), (1,)], (1, 1): [(2, 1), (1, 2), (2, 2)]}[order]
    return [rng.standard_normal(F_in + (n, N)) for F_in, N in zip(sizes, SHAPE)]


def weight_of(kind, rng, order):
    F = tuple(k + 1 for k in order)
    if kind == 'entries':
        return np.asarray(0.5 + rng.random())
    return 0.5 + rng.random(((3,) if kind == 'probe' else ()) + F)


def data_of(kind, target, sample, order):
    if kind == 'entries':
        return target.entries(sample)
    return getattr(target, kind)(sample, order)


class Fixture(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(41)
        np.random.seed(41)
        self.x, self.target = (t3.TuckerTensorTrain.randn(SHAPE, *RANKS) for _ in range(2))
        self.frame = t3m.MANIFOLD.frame(self.x)
        self.v = t3m.COREWISE.randn(self.frame).variations.data
        self.dirs = t3.TuckerTensorTrain.randn(SHAPE, *RANKS).data          # raw directions at (D, L) or (B, G)

    def batch(self, kind, order, scale=None):
        s = jet_sample(kind, self.rng, order)
        return LeastSquaresBatch(s, data_of(kind, self.target, s, order), weight_of(kind, self.rng, order), scale)


class TestTheEngineAdditions(Fixture):

    def runs(self, kind, frame, sample, want, **args):
        """The public names of the rules the engine runs for this call (its steps, counted)."""
        ran = []
        original = sv._run
        def counting(rule, *a, **k):
            ran.append(rule.name)
            return original(rule, *a, **k)
        with unittest.mock.patch.object(sv, '_run', side_effect=counting):
            sv.sampling_variables(kind, frame, sample, want, **args)
        return frozenset(name for name in ran if not name.startswith('_'))

    def test_computed_names_is_what_the_engine_runs(self):
        xu = ut3.UniformTuckerTensorTrain.from_t3(self.x, N=6, n=4, r=4)
        uf = ut3m.UNIFORM_MANIFOLD.frame(xu)
        uv = ut3m.UNIFORM_COREWISE.randn(uf).variations.data[:2]
        ud = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain(*self.dirs), N=6, n=4, r=4).data[:2]
        layouts = {
            ('ragged', 'manifold'): (self.frame.data, dict(variations=self.v, core_directions=self.dirs)),
            ('ragged', 'corewise'): (self.x.data, dict(core_directions=self.dirs)),
            ('uniform', 'manifold'): (uf.data, dict(variations=uv, core_directions=ud)),
            ('uniform', 'corewise'): (xu.data, dict(core_directions=ud)),
        }
        patterns = [('f',), ('f', 'g'), ('z',), ('JtJp',), ('Jp_norm_sq',), ('Kp',), ('g', 'JtJp', 'Kp'), ('r', 'f')]
        n = 0
        for kind in ('probe', 'apply', 'entries'):
            order = ORDERS[kind][-1]
            b = self.batch(kind, order)
            for (layout, call), (frame, dirs) in layouts.items():
                for want in patterns:
                    args = dict(order=order, y=b.y, weight=b.weight, **dirs)
                    variants = [('plain', args)]
                    if kind == 'probe' and call == 'manifold' and layout == 'ragged' and 'z' in want:
                        variants.append(('Vc', dict(args, Vc=self.frame.data[0])))      # a z route through Vc
                    if 'JtJp' in want:                                               # JtJp from a held Jp
                        (Jp,) = sv.sampling_variables(kind, frame, b.sample, ('Jp',), **args)
                        no_dirs = {k: v for k, v in args.items() if k not in ('variations', 'core_directions')}
                        variants.append(('held Jp', dict(no_dirs, held=sv.HeldVariables(Jp=Jp))))
                    (r, mu) = sv.sampling_variables(kind, frame, b.sample, ('r', 'mu'), **args)
                    variants.append(('held r, mu', dict(args, held=sv.HeldVariables(r=r, mu=mu))))
                    for label, a in variants:
                        if 'Kp' in want and label == 'held Jp':
                            continue
                        with self.subTest(kind=kind, layout=layout, call=call, want=want, variant=label):
                            names = sv.computed_names(kind, frame, b.sample, want, **a)
                            self.assertIsInstance(names, frozenset)
                            self.assertEqual(names, self.runs(kind, frame, b.sample, want, **a))
                            held = a.get('held')
                            if held is not None:
                                self.assertFalse(names & set(held.keys()))               # held: reused, not computed
                            self.assertFalse(any(name.startswith('_') for name in names))
                            n += 1
        self.assertGreater(n, 150)

    def test_computed_names_raises_as_the_engine(self):
        fr, point = self.frame.data, self.x.data
        b = self.batch('probe', ())
        cases = [('probe', fr, ('nonsense',), {}), ('probe', fr, ('f',), {}),             # unknown name; no y
                 ('probe', point, ('JtJp',), dict(variations=self.v)),                    # variations on a point
                 ('entries', fr, ('z',), dict(order=(1,))), ('nonsense', fr, ('z',), {}),
                 ('probe', fr[:3], ('z',), {}), ('probe', fr, 'z', {})]
        for kind, frame, want, args in cases:
            with self.subTest(kind=kind, want=want, args=sorted(args)):
                with self.assertRaises(Exception) as engine:
                    sv.sampling_variables(kind, frame, b.sample, want, **args)
                with self.assertRaises(type(engine.exception)) as query:
                    sv.computed_names(kind, frame, None, want, **args)
                self.assertEqual(str(query.exception), str(engine.exception))

    def test_the_oracle_holds_what_is_free(self):
        for kind in ('probe', 'apply', 'entries'):
            order = ORDERS[kind][-1]
            oracle = LeastSquaresOracle(kind, order)
            b = self.batch(kind, order)
            for want in [('f',), ('f', 'g'), ('JtJp',), ('r', 'f')]:
                with self.subTest(kind=kind, want=want):
                    d = dict(variations=self.v) if 'JtJp' in want else {}
                    ran = []
                    original = sv._run
                    def counting(rule, *a, **k):
                        ran.append(rule.name)
                        return original(rule, *a, **k)
                    with unittest.mock.patch.object(sv, '_run', side_effect=counting):
                        _, held = oracle.query(self.frame.data, b, want, **d)
                    args = dict(order=order, y=b.y, weight=b.weight, **d)
                    free = sv.computed_names(kind, self.frame.data, b.sample, want, **args)
                    # holding adds no work: the oracle runs exactly what want alone runs (nothing computed to be held)
                    self.assertEqual(frozenset(n for n in ran if not n.startswith('_')), free)
                    self.assertEqual(set(held.keys()), set(HOLD) & free)
                    fresh = sv.sampling_variables(kind, self.frame.data, b.sample, held.keys(), **args)
                    for name, value in zip(held.keys(), fresh):
                        self.assertEqual(relerr(held[name], value), 0.0, name)
                    # a later query at the same frame and batch: the held passed in comes back, plus what is new
                    _, held2 = oracle.query(self.frame.data, b, ('g',), held=held)
                    for name in held.keys():
                        self.assertIs(held2[name], held[name])
                    self.assertEqual(set(held2.keys()) - set(held.keys()),
                                     set(HOLD) & sv.computed_names(kind, self.frame.data, b.sample, ('g',),
                                                                   order=order, y=b.y, weight=b.weight, held=held))

    def test_frame_shape(self):
        """The mode sizes for the four forms: from the Tucker cores on ragged, from the static shape on uniform (not
        the padded width N of the supercores), stacked too; bad structures raise as frame_form does."""
        np.random.seed(42)
        for C in [(), (2,)]:
            x = t3.TuckerTensorTrain.randn(SHAPE, *RANKS, stack_shape=C)
            xu = ut3.UniformTuckerTensorTrain.from_t3(x, N=7, n=4, r=4)
            forms = {'ragged frame': t3m.MANIFOLD.frame(x).data, 'ragged point': x.data,
                     'uniform frame': ut3m.UNIFORM_MANIFOLD.frame(xu).data, 'uniform point': xu.data}
            for label, frame in forms.items():
                with self.subTest(C=C, form=label):
                    shape = sv.frame_shape(frame)
                    self.assertEqual(shape, SHAPE)
                    self.assertTrue(all(type(n) is int for n in shape))
        for bad in ((self.x.data[0],), np.zeros(3), ()):
            with self.assertRaises(ValueError):
                sv.frame_shape(bad)

    def test_frame_form(self):
        xu = ut3.UniformTuckerTensorTrain.from_t3(self.x)
        uf = ut3m.UNIFORM_MANIFOLD.frame(xu)
        cases = {'ragged frame': (self.frame.data, ('ragged', 'manifold')),
                 'ragged point': (self.x.data, ('ragged', 'corewise')),
                 'uniform frame': (uf.data, ('uniform', 'manifold')),
                 'uniform point': (xu.data, ('uniform', 'corewise'))}
        for label, (frame, form) in cases.items():
            with self.subTest(case=label):
                self.assertEqual(sv.frame_form(frame), form)
        self.assertEqual(len(self.frame.data), len(xu.data))                   # four elements, two meanings
        for bad in ((self.x.data[0],), self.x.data + (self.x.data[0],), np.zeros(3), ()):
            with self.assertRaises(ValueError):
                sv.frame_form(bad)


class TestQuery(Fixture):

    def test_each_output_against_the_engine(self):
        point = self.x.data
        for kind in ('probe', 'apply', 'entries'):
            for order in ORDERS[kind]:
                oracle = LeastSquaresOracle(kind, order)
                b = self.batch(kind, order)
                for call, frame in (('manifold', self.frame.data), ('corewise', point)):
                    with self.subTest(kind=kind, order=order, call=call):
                        # a manifold frame takes both forms; a point the raw one only
                        d = dict(core_directions=self.dirs, **({'variations': self.v} if call == 'manifold' else {}))
                        got, held = oracle.query(frame, b, WANT, **d)
                        ref = sv.sampling_variables(kind, frame, b.sample, WANT, order=order, y=b.y, weight=b.weight,
                                                    **d)
                        for name, a, c in zip(WANT, got, ref):
                            self.assertEqual(relerr(a, c), 0.0, name)
                        self.assertTrue(set(held.keys()) <= set(HOLD))
                        free = sv.computed_names(kind, frame, b.sample, WANT, order=order, y=b.y, weight=b.weight,
                                                 **d)
                        self.assertEqual(set(held.keys()), free & set(HOLD))
                        (ybs, fres, f), _ = oracle.query(frame, b, ('y_block_sumsq', 'f_resolution', 'f'))
                        ybs_ref = np.square(bfit.data_preparation.data_block_norms(kind, order, b.y))
                        self.assertLess(relerr(ybs, ybs_ref), 1e-14)
                        wy = math.sqrt(float(np.sum(np.asarray(b.weight) ** 2 * ybs_ref)))
                        eps = np.finfo(float).eps
                        self.assertLess(abs(float(fres) - eps * (wy * math.sqrt(2 * float(f)) + float(f))),
                                        1e-12 * float(fres))

    def test_the_uniform_layout_after_pack_batch(self):
        xu = ut3.UniformTuckerTensorTrain.from_t3(self.x, N=6, n=4, r=4)
        for kind in ('probe', 'apply', 'entries'):
            for order in ORDERS[kind]:
                oracle = LeastSquaresOracle(kind, order)
                b = self.batch(kind, order)
                pb = pack_batch(kind, order, b, 6)
                if kind != 'entries':                    # garbage in the mode padding of every packed form (round 1)
                    sample = np.array(pb.sample)
                    for i, N in enumerate(SHAPE):
                        sample[i, ..., N:] = 1e3
                    y = pb.y
                    if kind == 'probe':
                        y = np.array(pb.y)
                        for i, N in enumerate(SHAPE):
                            y[i, ..., N:] = 1e3
                    pb = dc.replace(pb, sample=sample, y=y)
                with self.subTest(kind=kind, order=order):
                    if kind != 'entries':
                        self.assertTrue(ut3o.is_packed(pb.sample))
                    want = ('f', 'r_block_sumsq', 'y_block_sumsq', 'f_resolution', 'g')
                    rag, _ = oracle.query(self.x.data, b, want)
                    uni, held = oracle.query(xu.data, pb, want)
                    for name, a, c in zip(want[:4], uni[:4], rag[:4]):
                        self.assertLess(relerr(a, c), 1e-12, name)
                    gB, gG = uni[4]
                    for i, (B, G) in enumerate(zip(*rag[4])):                   # the real block, per core
                        self.assertLess(relerr(gB[i][:B.shape[0], :B.shape[1]], B), 1e-12)
                        self.assertLess(relerr(gG[i][:G.shape[0], :G.shape[1], :G.shape[2]], G), 1e-12)
                    self.assertTrue(set(held.keys()) <= set(HOLD))


class TestTheResolutionDtype(Fixture):

    def test_float32_inputs_give_float32_eps(self):
        """``f_resolution``'s ``eps`` is the result dtype's: float32 inputs give a float32 ``f`` and float32's eps (round
        1: a float64 eps would make the floor 2**29 times too fine)."""
        b = self.batch('apply', (2,))
        f32 = lambda t: tuple(f32(a) for a in t) if isinstance(t, (tuple, list)) else np.asarray(t, np.float32)
        frame = f32(self.frame.data)
        b32 = LeastSquaresBatch(f32(b.sample), f32(b.y), f32(b.weight))
        (f, fres, ybs), _ = LeastSquaresOracle('apply', (2,)).query(frame, b32, ('f', 'f_resolution', 'y_block_sumsq'))
        self.assertEqual(np.asarray(f).dtype, np.float32)
        wy = math.sqrt(float(np.sum(np.asarray(b32.weight, float) ** 2 * np.asarray(ybs, float))))
        expected = float(np.finfo(np.float32).eps) * (wy * math.sqrt(2 * float(f)) + float(f))
        self.assertLess(abs(float(fres) - expected), 1e-5 * expected)


class TestTheCorewiseDirection(Fixture):

    def test_the_three_cases(self):
        oracle = LeastSquaresOracle('apply', ())
        b = self.batch('apply', ())
        point = self.x.data
        ref, _ = oracle.query(point, b, ('JtJp', 'Jp', 'Kp'), core_directions=self.dirs)
        renamed, _ = oracle.query(point, b, ('JtJp', 'Jp', 'Kp'), variations=self.dirs)
        self.assertEqual(relerr(renamed, ref), 0.0)
        same, _ = oracle.query(point, b, ('JtJp', 'Jp', 'Kp'), variations=self.dirs, core_directions=self.dirs)
        self.assertEqual(relerr(same, ref), 0.0)
        rebuilt = (tuple(self.dirs[0]), list(self.dirs[1]))                     # a rebuilt container, the same arrays
        same, _ = oracle.query(point, b, ('JtJp', 'Jp', 'Kp'), variations=rebuilt, core_directions=self.dirs)
        self.assertEqual(relerr(same, ref), 0.0)
        other = (tuple(a.copy() for a in self.dirs[0]), self.dirs[1])           # equal values, other arrays
        with self.assertRaisesRegex(ValueError, 'one form of a direction'):
            oracle.query(point, b, ('JtJp',), variations=other, core_directions=self.dirs)
        # on a manifold frame both forms go through as given
        (JtJp, Kp), _ = oracle.query(self.frame.data, b, ('JtJp', 'Kp'), variations=self.v, core_directions=self.dirs)
        ref = sv.sampling_variables('apply', self.frame.data, b.sample, ('JtJp', 'Kp'), y=b.y, weight=b.weight,
                                    variations=self.v, core_directions=self.dirs)
        self.assertEqual(relerr((JtJp, Kp), ref), 0.0)


class TestTheMinibatchScale(Fixture):

    def test_unbiased_over_every_subset(self):
        for kind in ('probe', 'apply', 'entries'):
            order = ORDERS[kind][-1]
            oracle = LeastSquaresOracle(kind, order)
            full = self.batch(kind, order)
            (f, g), _ = oracle.query(self.frame.data, full, ('f', 'g'))
            n, b = oracle.n_measurements(full), 2
            fs, gs = [], []
            for subset in itertools.combinations(range(n), b):
                (fb, gb), _ = oracle.query(self.frame.data, oracle.take(full, np.array(subset)), ('f', 'g'))
                fs.append(float(fb))
                gs.append(leaves(gb))
            with self.subTest(kind=kind):
                self.assertLess(abs(np.mean(fs) - float(f)) / float(f), 1e-13)
                self.assertLess(relerr([np.mean([g_[j] for g_ in gs], axis=0) for j in range(len(gs[0]))], leaves(g)),
                                1e-13)

    def test_a_drawn_batch(self):
        for kind in ('probe', 'apply', 'entries'):
            for order in ORDERS[kind]:
                oracle = LeastSquaresOracle(kind, order)
                full = self.batch(kind, order)
                sub = oracle.take(full, np.array([1, 4]))
                s = sub.minibatch_scale
                want = ('f', 'g', 'r', 'Jp', 'Jp_norm_sq', 'JtJp', 'Kp', 'r_block_sumsq', 'y_block_sumsq',
                        'f_resolution')
                got, _ = oracle.query(self.frame.data, sub, want, variations=self.v, core_directions=self.dirs)
                plain, _ = oracle.query(self.frame.data, dc.replace(sub, minibatch_scale=None), want,
                                        variations=self.v, core_directions=self.dirs)
                values, unscaled = dict(zip(want, got)), dict(zip(want, plain))
                with self.subTest(kind=kind, order=order):
                    self.assertEqual(s, 3.0)
                    for name in ('r', 'Jp', 'r_block_sumsq', 'y_block_sumsq'):          # plain
                        self.assertEqual(relerr(values[name], unscaled[name]), 0.0, name)
                    for name in ('f', 'g', 'Jp_norm_sq', 'JtJp', 'Kp', 'f_resolution'):  # times the scale, once
                        self.assertLess(relerr(values[name], [s * a for a in leaves(unscaled[name])]), 1e-15, name)
                    w = np.asarray(sub.weight)
                    def inner(a, c):                                              # scale * <a, w^2 c>
                        if kind == 'probe':
                            return s * sum(float(np.sum(w[i].reshape(w[i].shape + (1,) * (a[i].ndim - w[i].ndim)) ** 2
                                                        * a[i] * c[i])) for i in range(3))
                        return s * float(np.sum(w.reshape(w.shape + (1,) * (np.ndim(a) - w.ndim)) ** 2 * a * c))
                    f, r, Jp = float(values['f']), values['r'], values['Jp']
                    pg = sum(float(np.sum(p * q)) for P, Q in zip(self.v, values['g']) for p, q in zip(P, Q))
                    self.assertLess(abs(f - 0.5 * inner(r, r)), 1e-13 * f)
                    self.assertLess(abs(inner(Jp, r) - pg), 1e-12 * max(abs(pg), 1.0))
                    Jn = float(values['Jp_norm_sq'])
                    self.assertLess(abs(Jn - inner(Jp, Jp)), 1e-12 * Jn)
                    # f_resolution from the UNSCALED values, then the scale once (not s**1.5)
                    fb = f / s
                    wy = math.sqrt(float(np.sum(w ** 2 * values['y_block_sumsq'])))
                    expected = s * np.finfo(float).eps * (wy * math.sqrt(2 * fb) + fb)
                    self.assertLess(abs(float(values['f_resolution']) - expected), 1e-12 * expected)


class TestTheBatch(Fixture):

    def test_take(self):
        for kind in ('probe', 'apply', 'entries'):
            for order in ORDERS[kind]:
                oracle = LeastSquaresOracle(kind, order)
                full = self.batch(kind, order)
                idx = np.array([5, 0, 2])
                sub = oracle.take(full, idx)
                with self.subTest(kind=kind, order=order):
                    self.assertEqual(oracle.n_measurements(full), 6)
                    self.assertEqual(oracle.n_measurements(sub), 3)
                    self.assertEqual(sub.minibatch_scale, 2.0)
                    self.assertIs(sub.drawn_indices, idx)
                    self.assertEqual(oracle.take(sub, np.array([0])).minibatch_scale, 6.0)     # nested: multiplied
                    m = len(order)
                    if kind == 'entries':
                        self.assertTrue(np.array_equal(sub.sample, full.sample[:, idx]))
                        self.assertTrue(np.array_equal(sub.y, full.y[idx]))
                    else:
                        for a, c in zip(sub.sample, full.sample):
                            self.assertTrue(np.array_equal(a, np.take(c, idx, axis=m)))
                        ys = sub.y if kind == 'probe' else [sub.y]
                        yf = full.y if kind == 'probe' else [full.y]
                        for a, c in zip(ys, yf):
                            self.assertTrue(np.array_equal(a, np.take(c, idx, axis=m)))
                        packed = pack_batch(kind, order, full, 6)
                        psub = oracle.take(packed, idx)
                        self.assertEqual(oracle.n_measurements(packed), 6)
                        self.assertTrue(np.array_equal(psub.sample, pack_batch(kind, order, sub, 6).sample))
                        if kind == 'probe':
                            self.assertTrue(np.array_equal(psub.y, pack_batch(kind, order, sub, 6).y))
                    # a multi-axis W is flattened
                    if kind == 'apply' and order == ():
                        b2 = LeastSquaresBatch([a.reshape(2, 3, -1) for a in full.sample], full.y.reshape(2, 3))
                        t = oracle.take(b2, np.array([4, 1]))
                        self.assertTrue(np.array_equal(t.y, full.y[[4, 1]]))

    def test_check_batch(self):
        for kind in ('probe', 'apply', 'entries'):
            order = ORDERS[kind][-1]
            oracle = LeastSquaresOracle(kind, order)
            good = self.batch(kind, order)
            oracle.check_batch(good, SHAPE)
            bad = {'weight': dc.replace(good, weight=np.ones(7))}
            if kind == 'entries':
                bad['index rows'] = dc.replace(good, sample=good.sample[:2])
                bad['y W'] = dc.replace(good, y=good.y[:3])
            else:
                bad['modes'] = dc.replace(good, sample=good.sample[:2])
                bad['W'] = dc.replace(good, sample=[good.sample[0], good.sample[1][..., :3, :], good.sample[2]])
                bad['mode size'] = dc.replace(good, sample=[good.sample[0], good.sample[1][..., :2], good.sample[2]])
                bad['y'] = dc.replace(good, y=(good.y[:2] if kind == 'probe' else good.y[..., :3]))
            for label, b in bad.items():
                with self.subTest(kind=kind, case=label):
                    with self.assertRaises(ValueError):
                        oracle.check_batch(b, SHAPE)

    def test_the_pytree_round_trip(self):
        try:
            import jax
        except ImportError:
            self.skipTest('jax is not installed')
        b = self.batch('probe', (2,), scale=2.0)
        b = dc.replace(b, drawn_indices=np.array([1, 2]))
        leaves_, treedef = jax.tree_util.tree_flatten(b)
        back = jax.tree_util.tree_unflatten(treedef, leaves_)
        self.assertIsInstance(back, LeastSquaresBatch)
        for name in ('sample', 'y', 'weight', 'minibatch_scale', 'drawn_indices'):
            self.assertEqual(relerr(getattr(back, name), getattr(b, name)), 0.0)
        none = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(LeastSquaresBatch(b.sample))))
        self.assertIsNone(none.y)


class TestTheOracle(unittest.TestCase):

    def test_value_hash_and_equality(self):
        a, b = LeastSquaresOracle('probe', (2,)), LeastSquaresOracle('probe', (2,))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, LeastSquaresOracle('probe', (1,)))
        self.assertNotEqual(a, LeastSquaresOracle('apply', (2,)))

    def test_order_is_required_and_entries_has_no_jets(self):
        for kind in ('probe', 'apply', 'entries'):
            with self.subTest(kind=kind):
                with self.assertRaises(TypeError):
                    LeastSquaresOracle(kind)
                with self.assertRaises(TypeError):
                    LeastSquaresOracle(kind, 2)
        LeastSquaresOracle('entries', ())
        with self.assertRaisesRegex(ValueError, 'no jets'):
            LeastSquaresOracle('entries', (1,))
        with self.assertRaises(ValueError):
            LeastSquaresOracle('nonsense', ())
        self.assertEqual(set(LeastSquaresOracle.optional_outputs),
                         {'Kp', 'Jp', 'r', 'r_block_sumsq', 'y_block_sumsq', 'f_resolution'})

    def test_reachable_from_the_frontend(self):
        import t3toolbox.fitting as fitting
        self.assertIs(fitting.LeastSquaresOracle, LeastSquaresOracle)
        self.assertIs(fitting.LeastSquaresBatch, LeastSquaresBatch)
        self.assertIn('LeastSquaresOracle', fitting.__all__)
        self.assertIn('LeastSquaresBatch', fitting.__all__)


class TestTheBridgeToTheOldKinds(Fixture):
    """Against the old derivative kinds through the conversions: their sample ``(ww, pp)`` is the input jet
    ``affine_jet(ww, pp)``, their data the derivatives (``taylor_to_derivatives``), their weight a weight on derivative
    residuals (``derivative_weight_to_taylor_weight``). ``f``, ``g``, ``JtJp`` and ``Kp`` agree."""

    def test_against_the_old_kinds(self):
        n = 2
        frame = self.frame.data
        base = (frame[0], frame[2])
        ww, pp = ([self.rng.standard_normal((6, N)) for N in SHAPE] for _ in range(2))
        index = np.stack([self.rng.integers(0, N, size=6) for N in SHAPE])
        F = (n + 1,)
        cases = {
            'probe': (bfit.ProbeDerivativesKind, (ww, pp), jets.affine_jet(ww, pp), 'probe',
                      0.5 + self.rng.random((3,) + F)),
            'apply': (bfit.ApplyDerivativesKind, (ww, pp), jets.affine_jet(ww, pp), 'apply', 0.5 + self.rng.random(F)),
            'entries': (bfit.EntriesDerivativesKind, (index, pp),
                        jets.affine_jet(jets.index_to_one_hot(index, SHAPE), pp), 'apply', 0.5 + self.rng.random(F)),
        }
        for name, (Kind, old_sample, jet, new_kind, w_old) in cases.items():
            with self.subTest(kind=name):
                kind = Kind(order=n, weight=w_old)
                y_taylor = getattr(self.target, new_kind)(jet, (n,))
                y_old = jets.taylor_to_derivatives(new_kind, (n,), y_taylor)
                r_old = jets.taylor_to_derivatives(new_kind, (n,), getattr(self.x, new_kind)(jet, (n,)))
                r_old = [a - b for a, b in zip(r_old, y_old)] if new_kind == 'probe' else r_old - y_old
                sweep = kind.precompute(frame, old_sample)
                oracle = LeastSquaresOracle(new_kind, (n,))
                batch = LeastSquaresBatch(jet, y_taylor, jets.derivative_weight_to_taylor_weight(w_old, (n,)))
                (f, g, JtJp, Kp), _ = oracle.query(frame, batch, ('f', 'g', 'JtJp', 'Kp'), variations=self.v,
                                                    core_directions=self.dirs)
                w2 = kind._apply_weight(r_old, 1)
                f_old = 0.5 * sum(float(np.sum(a ** 2)) for a in leaves(w2))
                self.assertLess(abs(float(f) - f_old) / f_old, 1e-12)
                self.assertLess(relerr(g, kind.transpose(r_old, old_sample, frame, sweep)), 1e-12)
                Jp_old = kind.forward(self.v, old_sample, frame, sweep)
                self.assertLess(relerr(JtJp, kind.transpose(Jp_old, old_sample, frame, sweep)), 1e-12)
                Kp_old = kind.corewise_hessian(self.dirs, r_old, old_sample, frame, sweep, base)
                self.assertLess(relerr(Kp, Kp_old), 1e-12)


if __name__ == '__main__':
    unittest.main()
