# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The types of the sampling surface (the data-metric arc, M5): the transformation laws (``tests/_typing_laws.py``) on
the functions no earlier slice tested, and on the type column of the engine's variable table.

- **The forwards and the tangents** (`TestTheForwardsAndTangents`), 36: ``t3_*``, ``ut3_*``, ``TuckerTensorTrain`` and
  ``UniformTuckerTensorTrain`` ``.probe/.apply/.entries`` (``z: Y*``); ``tv_*``, ``utv_*``, ``*_corewise_tangent``
  (backend, both layouts, and the two frontend classes), ``T3Tangent`` and ``UT3Tangent`` ``.probe/.apply/.entries``
  (``X -> Y*``). The input law on all 36 (the point and the directions as ``X``, the frame's down store by
  ``Delta^-1``, the output as a measurement); the jet law on the 24 with jets.
- **The dense functions** (`TestTheDenseFunctions`): ``E x inputs -> Y*``, ``T`` moved as ``E``.
- **The jets and data-preparation helpers** (`TestTheHelpers`): ``affine_jet`` is the jet law's own definition; the
  Taylor/derivative converters move ``P`` to ``P`` and pair contragrediently (a ``Y`` element converts the other way:
  the pairing is invariant, which tells ``k!`` from ``1/k!``); the weight converter keeps ``f``; ``data_block_norms``
  is ``P`` per block, ``equalizing_weight`` ``P`` to ``P*``, and the recipe's weight leaves ``f`` invariant.
  ``index_to_one_hot`` and the checkers move nothing (an index does not move), so they have no law to test.
- **The oracle's outputs** (`TestTheOracle`) on a drawn batch (``gamma = 2``): ``r``, ``Jp`` in ``Y*``; ``g``,
  ``JtJp``, ``Kp`` in ``X*``; ``f``, ``Jp_norm_sq``, ``f_resolution`` and the block sums in ``R``, invariant.
- **The engine's type column** (`TestTheEngineTypeColumn`): read from the module docstring and checked name by name,
  under both laws, so that the column cannot drift from what the engine computes.

Homogeneity tests: necessary, not sufficient; the dense oracles stay the primary tests. The moved frames are not
orthonormal, so the frontend calls run under ``safety.unsafe()``. numpy only (jax dispatch is ``test_dispatch``'s).
"""
import itertools
import re
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
import t3toolbox.backend.jets as jets
import t3toolbox.backend.data_preparation as prep
import t3toolbox.backend.sampling_variables as sv
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch

from _typing_laws import JetLaw, InputLaw, assert_law, law_error

SHAPE, TUCKER, TT = (4, 5, 3), (2, 3, 2), (1, 2, 2, 1)
W = (3,)
KINDS = ('probe', 'apply', 'entries')
BACKEND = {'probe': bprobing, 'apply': bapply, 'entries': bentries}
JET_ORDERS = {(2,): (0.7,), (1, 1): (0.7, 1.9)}
FACTORIAL_ORDERS = {(2,): (0.7,), (2, 1): (0.7, 1.9)}        # some k! != 1: the converters are not the identity


def box(order):
    return tuple(n + 1 for n in order)


def _uframe(frame):
    return ubv.UT3Frame.from_t3frame(bvf.T3Frame(*frame))


def _uvariations(variations):
    return ubv.UT3Variations.from_t3variations(bvf.T3Variations(*variations))


def _upoint(x):
    return ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain(*x))


def _o(kind, order):
    return {} if kind == 'entries' else {'order': order}


def forwards_and_tangents(kind, order):
    """The 36 as ``(name, base, direction, f(base, sample, direction))``: ``base`` 'point' | 'frame', ``direction``
    None | 'core' | 'variations'; each returns plain arrays (``z`` or ``J x_dot``, ``Y*``)."""
    m, o = BACKEND[kind], _o(kind, order)
    T, UT = t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain
    return [
        ('t3_%s' % kind, 'point', None, lambda b, s, p: getattr(m, 't3_%s' % kind)(b, s, **o)),
        ('ut3_%s' % kind, 'point', None, lambda b, s, p: getattr(ut3s, 'ut3_%s' % kind)(_upoint(b).data, s, **o)),
        ('TuckerTensorTrain.%s' % kind, 'point', None, lambda b, s, p: getattr(T(*b), kind)(s, **o)),
        ('UniformTuckerTensorTrain.%s' % kind, 'point', None, lambda b, s, p: getattr(_upoint(b), kind)(s, **o)),
        ('tv_%s' % kind, 'frame', 'variations', lambda b, s, p: getattr(m, 'tv_%s' % kind)(b, p, s, **o)),
        ('utv_%s' % kind, 'frame', 'variations',
         lambda b, s, p: getattr(utvs, 'utv_%s' % kind)(_uframe(b).data, _uvariations(p).data, s, **o)),
        ('t3_%s_corewise_tangent' % kind, 'point', 'core',
         lambda b, s, p: getattr(m, 't3_%s_corewise_tangent' % kind)(b, s, p, **o)),
        ('ut3_%s_corewise_tangent' % kind, 'point', 'core',
         lambda b, s, p: getattr(ut3s, 'ut3_%s_corewise_tangent' % kind)(_upoint(b).data, s, _upoint(p).data[:2], **o)),
        ('T3Tangent.%s' % kind, 'frame', 'variations',
         lambda b, s, p: getattr(t3m.T3Tangent(bvf.T3Frame(*b), bvf.T3Variations(*p)), kind)(s, **o)),
        ('UT3Tangent.%s' % kind, 'frame', 'variations',
         lambda b, s, p: getattr(ut3m.UT3Tangent(_uframe(b), _uvariations(p)), kind)(s, **o)),
        ('TuckerTensorTrain.%s_corewise_tangent' % kind, 'point', 'core',
         lambda b, s, p: getattr(T(*b), '%s_corewise_tangent' % kind)(s, p, **o)),
        ('UniformTuckerTensorTrain.%s_corewise_tangent' % kind, 'point', 'core',
         lambda b, s, p: getattr(_upoint(b), '%s_corewise_tangent' % kind)(s, _upoint(p).data[:2], **o)),
    ]


class Fixture(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(31)
        np.random.seed(31)
        self.x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        self.frame = t3m.MANIFOLD.frame(self.x).data
        zero = t3m.T3Tangent.zeros(t3m.MANIFOLD.frame(self.x)).variations.data
        self.variations = (tuple(self.rng.standard_normal(V.shape) for V in zero[0]),
                           tuple(self.rng.standard_normal(H.shape) for H in zero[1]))
        self.core = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        self.law = InputLaw([0.5 + 1.5 * self.rng.random(n) for n in SHAPE])

    def sample(self, kind, order):
        if kind == 'entries':
            return np.stack([self.rng.integers(0, n, size=W) for n in SHAPE])
        return [self.rng.standard_normal((2,) * len(order) + W + (n,)) for n in SHAPE]

    def measurement(self, kind, index=None):
        """The input law of ``Y*`` for the kind: probe ``Delta_i^-1`` on the edge, apply fixed, entries by the index."""
        if kind == 'probe':
            return self.law.V_star
        if kind == 'apply':
            return lambda z: z
        return lambda z: self.law.entries(z, index)


class TestTheForwardsAndTangents(Fixture):

    def test_the_input_law_on_all_36(self):
        law, n = self.law, 0
        for kind in KINDS:
            order = () if kind == 'entries' else (2,)
            for name, base, direction, f in forwards_and_tangents(kind, order):
                s = self.sample(kind, order)
                b = self.frame if base == 'frame' else self.x.data
                moved_b = law.frame(b) if base == 'frame' else law.tucker(b)
                p = {'variations': self.variations, 'core': self.core, None: None}[direction]
                moved_p = {'variations': law.variations, 'core': law.tucker, None: lambda v: None}[direction](p)
                moved_s = s if kind == 'entries' else law.inputs(s)
                with self.subTest(kind=kind, function=name):
                    with safety.unsafe():
                        assert_law(self, f, (b, s, p), (moved_b, moved_s, moved_p), self.measurement(kind, s),
                                   rtol=1e-11)
                n += 1
        self.assertEqual(n, 36)

    def test_the_jet_law_on_the_24_with_jets(self):
        n = 0
        for order, ell in JET_ORDERS.items():
            law = JetLaw(order, ell)
            for kind in ('probe', 'apply'):
                for name, base, direction, f in forwards_and_tangents(kind, order):
                    s = self.sample(kind, order)
                    b = self.frame if base == 'frame' else self.x.data
                    p = {'variations': self.variations, 'core': self.core, None: None}[direction]
                    with self.subTest(order=order, kind=kind, function=name):
                        with safety.unsafe():
                            assert_law(self, f, (b, s, p), (b, law.inputs(s), p), law.P, rtol=1e-11)
                    n += 1
        self.assertEqual(n, 2 * 24)


class TestTheDenseFunctions(Fixture):
    """``dense_*``: ``E x inputs -> Y*``, linear in ``E``."""

    def test_the_input_law(self):
        T = self.x.to_dense()
        for kind in KINDS:
            order = () if kind == 'entries' else (1,)
            s = self.sample(kind, order)
            f = (lambda T, s: bentries.dense_entries(T, s)) if kind == 'entries' else \
                (lambda T, s, kind=kind: getattr(BACKEND[kind], 'dense_%s' % kind)(T, s, order))
            moved_s = s if kind == 'entries' else self.law.inputs(s)
            with self.subTest(kind=kind):
                assert_law(self, f, (T, s), (self.law.dense(T), moved_s), self.measurement(kind, s), rtol=1e-12)

    def test_the_jet_law(self):
        T = self.x.to_dense()
        for (order, ell), kind in itertools.product(JET_ORDERS.items(), ('probe', 'apply')):
            law = JetLaw(order, ell)
            s = self.sample(kind, order)
            f = lambda s, kind=kind, order=order: getattr(BACKEND[kind], 'dense_%s' % kind)(T, s, order)
            with self.subTest(kind=kind, order=order):
                assert_law(self, f, (s,), (law.inputs(s),), law.P, rtol=1e-12)


class TestTheHelpers(Fixture):

    def test_affine_jet_is_the_jet_law(self):
        """``s_j -> ell_j s_j`` scales direction ``j`` by ``ell_j``: the line's jet moves by ``P``; and ``V_i`` in,
        ``P_in_i (x) V_i`` out under the input law."""
        law = JetLaw((1, 1), (0.7, 1.9))
        ww, pp, qq = ([self.rng.standard_normal(W + (n,)) for n in SHAPE] for _ in range(3))
        assert_law(self, lambda p, q: jets.affine_jet(ww, p, q), (pp, qq),
                   ([0.7 * v for v in pp], [1.9 * v for v in qq]), law.inputs, rtol=0.0)
        assert_law(self, lambda w, p: jets.affine_jet(w, p), (ww, pp), (self.law.inputs(ww), self.law.inputs(pp)),
                   self.law.inputs, rtol=1e-15)

    def test_the_taylor_converters(self):
        """Each moves ``P`` to ``P`` (and ``P*`` to ``P*``) under the jet law and ``V_i*`` to ``V_i*`` under the input
        law; and the pairing of ``Y`` with ``Y*`` is kept when a ``Y*`` element converts one way and a ``Y`` element
        the other (the contragredient of the signature comments)."""
        for kind, (order, ell) in itertools.product(('probe', 'apply'), FACTORIAL_ORDERS.items()):
            law, F = JetLaw(order, ell), box(order)
            shapes = [F + W + (n,) for n in SHAPE] if kind == 'probe' else [F + W]
            pick = (lambda a: a) if kind == 'probe' else (lambda a: a[0])
            r, c = pick([self.rng.standard_normal(s) for s in shapes]), pick([self.rng.standard_normal(s) for s in shapes])
            to_t = lambda v: jets.derivatives_to_taylor(kind, order, v)
            to_d = lambda v: jets.taylor_to_derivatives(kind, order, v)
            with self.subTest(kind=kind, order=order):
                for convert in (to_t, to_d):
                    assert_law(self, convert, (r,), (law.P(r),), law.P, rtol=1e-14)
                    assert_law(self, convert, (r,), (law.P_star(r),), law.P_star, rtol=1e-14)
                    if kind == 'probe':
                        assert_law(self, convert, (r,), (self.law.V_star(r),), self.law.V_star, rtol=1e-14)
                pair = lambda a, b: sum(float(np.sum(np.asarray(u) * np.asarray(v)))
                                        for u, v in zip(*(([a], [b]) if kind == 'apply' else (a, b))))
                # r Taylor -> derivative (Y*); c derivative -> Taylor (Y): <c_t, r_t> = <c_d, r_d>
                self.assertAlmostEqual(pair(to_d(c), r), pair(c, to_d(r)), places=12)
                # c Taylor -> derivative by derivatives_to_taylor (Y): <c_d, r_d> = <c_t, r_t>
                self.assertAlmostEqual(pair(to_t(c), to_d(r)), pair(c, r), places=12)
                self.assertGreater(abs(pair(to_t(c), r) - pair(c, r)), 1e-3)          # the wrong way is not

    def test_the_weight_converter_and_prepare_derivative_data(self):
        """The weight moves by ``P*``; the converted weight on Taylor residuals gives the derivative misfit; and
        ``prepare_derivative_data`` moves ``y`` and the validation data by ``P``, the weight by ``P*``."""
        for kind, (order, ell) in itertools.product(('probe', 'apply'), FACTORIAL_ORDERS.items()):
            law, F = JetLaw(order, ell), box(order)
            shapes = [F + W + (n,) for n in SHAPE] if kind == 'probe' else [F + W]
            pick = (lambda a: a) if kind == 'probe' else (lambda a: a[0])
            y, y_val, r = (pick([self.rng.standard_normal(s) for s in shapes]) for _ in range(3))
            omega = pick([0.5 + self.rng.random(s) for s in shapes])
            convert = lambda w: jets.derivative_weight_to_taylor_weight(kind, order, w)
            with self.subTest(kind=kind, order=order):
                assert_law(self, convert, (omega,), (law.P_star(omega),), law.P_star, rtol=1e-14)
                r_d = jets.taylor_to_derivatives(kind, order, r)
                prod = lambda a, b: [np.asarray(u) * np.asarray(v) for u, v in
                                     (zip(a, b) if kind == 'probe' else [(a, b)])]
                self.assertAlmostEqual(sum(float(np.sum(q)) for q in prod(r, jets.data_metric(r, convert(omega), None))),
                                       sum(float(np.sum(q)) for q in prod(r_d, jets.data_metric(r_d, omega, None))),
                                       places=10)
                f = lambda y, w, v: jets.prepare_derivative_data(kind, order, y, w, v)
                assert_law(self, f, (y, omega, y_val), (law.P(y), law.P_star(omega), law.P(y_val)),
                           lambda out: (law.P(out[0]), law.P_star(out[1]), law.P(out[2])), rtol=1e-14)

    def test_block_norms_equalizing_and_the_recipe(self):
        """``data_block_norms``: ``P`` per block; ``equalizing_weight``: ``P`` per block to ``P*`` per block (no block
        at the floor); so the recipe's weight moves by ``P*`` and the misfit ``<r, M r>`` is invariant."""
        for kind, (order, ell) in itertools.product(('probe', 'apply'), JET_ORDERS.items()):
            law, F = JetLaw(order, ell), box(order)
            lead = 1 if kind == 'probe' else 0
            shapes = [F + W + (n,) for n in SHAPE] if kind == 'probe' else [F + W]
            pick = (lambda a: a) if kind == 'probe' else (lambda a: a[0])
            y, r = (pick([self.rng.standard_normal(s) for s in shapes]) for _ in range(2))
            norms = lambda y: prep.data_block_norms(kind, order, y)
            equalize = lambda b: prep.equalizing_weight(b, floor=1e-12)
            recipe = lambda y: jets.blocks_to_weight(kind, order, equalize(norms(y)), y)
            with self.subTest(kind=kind, order=order):
                assert_law(self, norms, (y,), (law.P(y),), lambda b: law.P(b, lead=lead), rtol=1e-14)
                b = norms(y)
                assert_law(self, equalize, (b,), (law.P(b, lead=lead),), lambda w: law.P_star(w, lead=lead),
                           rtol=1e-14)
                assert_law(self, recipe, (y,), (law.P(y),), lambda w: law.P_star(w, lead=lead), rtol=1e-14)
                misfit = lambda y, r: sum(float(np.sum(np.asarray(u) * np.asarray(v))) for u, v in
                                          (zip(r, jets.data_metric(r, recipe(y), None)) if kind == 'probe'
                                           else [(r, jets.data_metric(r, recipe(y), None))]))
                self.assertLessEqual(law_error(misfit, (y, r), (law.P(y), law.P(r)), lambda f: f), 1e-13)


class TestTheOracle(Fixture):
    """The oracle's outputs on a drawn batch (``gamma = 2``), each by its type: ``r``, ``Jp`` in ``Y*``; ``g``,
    ``JtJp``, ``Kp`` in ``X*`` (the ``V`` / ``D`` / ``B`` blocks by ``Delta_i``, the TT blocks fixed); ``f``,
    ``Jp_norm_sq``, ``f_resolution``, ``r_block_sumsq``, ``y_block_sumsq`` in ``R`` (invariant)."""

    WANT = ('f', 'g', 'JtJp', 'Jp_norm_sq', 'Kp', 'Jp', 'r', 'r_block_sumsq', 'y_block_sumsq', 'f_resolution')
    INDICES = np.array([0, 2, 3])

    def data(self, kind, order, Wn):
        F = box(order)
        shapes = [F + Wn + (n,) for n in SHAPE] if kind == 'probe' else [(F + Wn) if kind == 'apply' else Wn]
        y = [self.rng.standard_normal(s) for s in shapes]
        weight = [0.5 + self.rng.random(s) for s in shapes]          # per entry: a full edge for probe
        return (y, weight) if kind == 'probe' else (y[0], weight[0])

    def query(self, kind, order, base, sample, y, weight, directions):
        oracle = LeastSquaresOracle(kind, order)
        drawn = oracle.take(LeastSquaresBatch(sample, y, weight), self.INDICES)
        self.assertEqual(float(drawn.minibatch_scale), 2.0)
        values, _ = oracle.query(base, drawn, self.WANT, **directions)
        return values

    def full_sample(self, kind, order):
        """The batch's sample before the draw, ``W = (6,)``."""
        if kind == 'entries':
            return np.stack([self.rng.integers(0, n, size=(6,)) for n in SHAPE])
        return [self.rng.standard_normal((2,) * len(order) + (6,) + (n,)) for n in SHAPE]

    def directions(self, base_name):
        dot = (tuple(self.rng.standard_normal(D.shape) for D in self.frame[0]),
               tuple(self.rng.standard_normal(L.shape) for L in self.frame[2])) if base_name == 'frame' else self.core
        if base_name == 'frame':
            return dict(variations=self.variations, core_directions=dot)
        return dict(core_directions=dot)

    @staticmethod
    def moved_directions(law, directions):
        out = {}
        for key, value in directions.items():
            out[key] = law.variations(value) if key == 'variations' else law.tucker(value)
        return out

    def test_the_input_law(self):
        law, blocks = self.law, lambda pair: (self.law.tucker_dual(pair[0]), tuple(pair[1]))
        for kind, base_name in itertools.product(KINDS, ('frame', 'point')):
            order = () if kind == 'entries' else (2,)
            sample = self.full_sample(kind, order)
            y, weight = self.data(kind, order, (6,))
            base = self.frame if base_name == 'frame' else self.x.data
            moved_base = law.frame(base) if base_name == 'frame' else law.tucker(base)
            directions = self.directions(base_name)
            drawn_index = None if kind != 'entries' else sample[:, self.INDICES]
            meas = self.measurement(kind, drawn_index)
            dual = law.V if kind == 'probe' else ((lambda c: c) if kind == 'apply' else
                                                   (lambda c: law.entries_dual(c, sample)))
            moved_y = law.V_star(y) if kind == 'probe' else (y if kind == 'apply' else law.entries(y, sample))
            moved_weight = dual(weight)

            def move_out(out):
                f, g, JtJp, Jp_norm_sq, Kp, Jp, r, rb, yb, fres = out
                return f, blocks(g), blocks(JtJp), Jp_norm_sq, blocks(Kp), meas(Jp), meas(r), rb, yb, fres
            with self.subTest(kind=kind, base=base_name):
                with safety.unsafe():
                    assert_law(self, lambda b, s, y, w, dd: self.query(kind, order, b, s, y, w, dd),
                               (base, sample, y, weight, directions),
                               (moved_base, sample if kind == 'entries' else law.inputs(sample), moved_y, moved_weight,
                                self.moved_directions(law, directions)), move_out, rtol=1e-11)

    def test_the_jet_law(self):
        for (order, ell), kind, base_name in itertools.product(JET_ORDERS.items(), ('probe', 'apply'),
                                                               ('frame', 'point')):
            law = JetLaw(order, ell)
            sample = self.full_sample(kind, order)
            y, weight = self.data(kind, order, (6,))
            base = self.frame if base_name == 'frame' else self.x.data
            directions = self.directions(base_name)

            def move_out(out):
                f, g, JtJp, Jp_norm_sq, Kp, Jp, r, rb, yb, fres = out
                return f, g, JtJp, Jp_norm_sq, Kp, law.P(Jp), law.P(r), rb, yb, fres
            with self.subTest(kind=kind, order=order, base=base_name):
                assert_law(self, lambda s, y, w: self.query(kind, order, base, s, y, w, directions),
                           (sample, y, weight), (law.inputs(sample), law.P(y), law.P_star(weight)), move_out,
                           rtol=1e-11)


def type_column():
    """The engine's variable table, read from the module docstring: ``{name: (index, size, type)}``."""
    lines = sv.__doc__.splitlines()
    start = next(n for n, line in enumerate(lines) if line.startswith('| Variable | Index | Jet sizes | Size | Type |'))
    out = {}
    for line in lines[start + 2:]:
        if not line.startswith('|'):
            break
        name, index, _, size, kind = [cell.strip() for cell in line.strip().strip('|').split('|')]
        out[re.findall(r'`([^`]+)`', name)[0]] = (index, size, kind)
    return out


class TestTheEngineTypeColumn(Fixture):
    """The last column of the engine's variable table, name by name: under the jet law a ``P`` slot ``k`` scales as
    ``ell^k``, a turned one (``Y turned``, ``P* turned``) as ``ell^-(n-k)``, a block (``X*``) not at all; under the
    input law the edge variables (``P``, ``P* turned``) are fixed, ``Y*`` moves as a measurement, ``Y turned`` as an
    element of ``Y`` and an ``X*`` block on the mode axis (size ``(., N_i)``) by ``Delta_i``. On probe and apply (an
    entries index does not move with the input law, so its edge variables do: the column's types are kind-free, its
    input law is not), on a manifold frame and on a corewise point."""

    VOCABULARY = {'P', 'Y*', 'Y turned', 'P* turned', 'X*'}

    def test_the_column_is_complete_and_its_vocabulary(self):
        column = type_column()
        self.assertEqual(set(column), set(sv._SHAPES))
        for name, (index, size, kind) in column.items():
            with self.subTest(name=name):
                self.assertIn(kind, self.VOCABULARY)
                self.assertEqual(index == 'mode', kind in ('Y*', 'Y turned'))       # the data space is on the modes
                self.assertEqual(size.startswith('('), kind == 'X*')                  # a block has a core's shape
                self.assertEqual('tilde' in name and kind != 'X*', kind.endswith('turned'))

    def engine(self, kind, base_name, order, names):
        directions = dict(variations=self.variations,
                          core_directions=(tuple(self.rng.standard_normal(D.shape) for D in self.frame[0]),
                                           tuple(self.rng.standard_normal(L.shape) for L in self.frame[2]))) \
            if base_name == 'frame' else dict(core_directions=self.core)

        def run(base, sample, y, weight, dd):
            return sv.sampling_variables(kind, base, sample, names, order=order, y=y, weight=weight, **dd)
        return run, directions

    def public(self, kind, base_name):
        graph = sv._GRAPHS[kind, '_manifold' if base_name == 'frame' else '_corewise']
        column = type_column()
        return tuple(n for n in sv._public(graph) if n in column)

    def test_the_jet_law(self):
        column = type_column()
        for (order, ell), kind, base_name in itertools.product(JET_ORDERS.items(), ('probe', 'apply'),
                                                               ('frame', 'point')):
            law, F = JetLaw(order, ell), box(order)
            names = self.public(kind, base_name)
            run, dd = self.engine(kind, base_name, order, names)
            sample = self.sample(kind, order)
            y = [self.rng.standard_normal(F + W + (n,)) for n in SHAPE] if kind == 'probe' else \
                self.rng.standard_normal(F + W)
            weight = jax_free_like(self.rng, y)
            base = self.frame if base_name == 'frame' else self.x.data
            out = run(base, sample, y, weight, dd)
            moved = run(base, law.inputs(sample), law.P(y), law.P_star(weight), dd)
            mover = {'P': law.P, 'Y*': law.P, 'Y turned': law.P_star_turned, 'P* turned': law.P_star_turned,
                     'X*': lambda v: v}
            for name, a, b in zip(names, out, moved):
                with self.subTest(kind=kind, order=order, base=base_name, name=name, type=column[name][2]):
                    self.assertLessEqual(law_error(lambda v: v, (a,), (b,), mover[column[name][2]]), 1e-11)

    def test_the_input_law(self):
        column, law = type_column(), self.law
        for order, kind, base_name in itertools.product(((), (2,)), ('probe', 'apply'), ('frame', 'point')):
            F = box(order)
            names = self.public(kind, base_name)
            run, dd = self.engine(kind, base_name, order, names)
            sample = self.sample(kind, order)
            y = [self.rng.standard_normal(F + W + (n,)) for n in SHAPE] if kind == 'probe' else \
                self.rng.standard_normal(F + W)
            weight = jax_free_like(self.rng, y)
            meas = law.V_star if kind == 'probe' else (lambda v: v)
            dual = law.V if kind == 'probe' else (lambda v: v)
            base = self.frame if base_name == 'frame' else self.x.data
            moved_base = law.frame(base) if base_name == 'frame' else law.tucker(base)
            moved_dd = TestTheOracle.moved_directions(law, dd)
            out = run(base, sample, y, weight, dd)
            moved = run(moved_base, law.inputs(sample), meas(y), dual(weight), moved_dd)
            for name, a, b in zip(names, out, moved):
                index, size, kind_of = column[name]
                mover = {'P': lambda v: v, 'P* turned': lambda v: v, 'Y*': meas, 'Y turned': dual,
                         'X*': law.tucker_dual if size.endswith('N_i)') else (lambda v: v)}[kind_of]
                with self.subTest(kind=kind, order=order, base=base_name, name=name, type=kind_of):
                    self.assertLessEqual(law_error(lambda v: v, (a,), (b,), mover), 1e-11)

    def test_the_outputs_named_in_the_prose(self):
        """``f``, ``Jp_norm_sq``, ``r_block_sumsq`` (``R``) and ``g``, ``JtJp``, ``Kp`` (pairs of ``X*`` blocks),
        typed in the prose after the table, under the jet law."""
        names = ('f', 'Jp_norm_sq', 'r_block_sumsq', 'g', 'JtJp', 'Kp')
        for (order, ell), kind in itertools.product(JET_ORDERS.items(), ('probe', 'apply')):
            law, F = JetLaw(order, ell), box(order)
            run, dd = self.engine(kind, 'point', order, names)
            sample = self.sample(kind, order)
            y = [self.rng.standard_normal(F + W + (n,)) for n in SHAPE] if kind == 'probe' else \
                self.rng.standard_normal(F + W)
            weight = jax_free_like(self.rng, y)
            with self.subTest(kind=kind, order=order):
                assert_law(self, lambda s, y, w: run(self.x.data, s, y, w, dd), (sample, y, weight),
                           (law.inputs(sample), law.P(y), law.P_star(weight)), lambda v: v, rtol=1e-11)


def jax_free_like(rng, y):
    """A positive weight per entry, of ``y``'s structure."""
    if isinstance(y, (list, tuple)):
        return [0.5 + rng.random(np.shape(a)) for a in y]
    return 0.5 + rng.random(np.shape(y))


if __name__ == '__main__':
    unittest.main()
