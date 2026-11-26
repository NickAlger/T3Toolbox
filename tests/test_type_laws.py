# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The transformation laws (``tests/_typing_laws.py``) on the data metric, its weight, the dual-jet converters and the
library's own instances of the laws, the two rescalings: graded messages, the data-metric sub-arc, M1.

1. **The data metric** (`TestDataMetric`): ``M: Y* -> Y`` maps the ``P`` law to the ``P*`` law, with the weight moved
   by its own (``P*``); and, for probe, ``P (x) V_i*`` to ``P* (x) V_i`` with a full-edge weight moved by ``Delta_i``.
   The scale does not move. ``blocks_to_weight`` maps ``P*`` blocks to a ``P*`` weight; the batch's method follows
   the law with its weight.
2. **The converters** (`TestConverters`): ``jet_to_dual_jet`` is ``Y* -> Y`` turned (``P`` with the weight moved, to
   ``P*`` turned), and with no weight ``Y -> Y`` turned (``P*`` to ``P*`` turned); ``dual_jet_to_jet`` the other way.
3. **The rescalings are the laws** (`TestRescalings`): ``rescale_direction`` is the jet law on the sample and the
   data, and the forwards at its sample are the data's law; ``rescale_coordinates`` and ``t3_unscale_coordinates``
   are the input law, and the probe forward at the moved point and sample is the measurement's law.
4. **Negative controls** (`TestTheLawsCatch`): the faults the laws are for fail them at ``O(1)``: no square, no flip,
   a flip on one axis, a jet read as a dual jet; ``c`` read as a measurement (the input law); and the helper refuses
   the vacuous settings.

The laws are homogeneity tests, necessary and not sufficient; the dense oracles stay the primary tests. numpy only.
"""
import itertools
import unittest

import numpy as np

import t3toolbox.backend.jets as jets
import t3toolbox.backend.apply as bapply
import t3toolbox.backend.probing as probing
import t3toolbox.backend.data_preparation as prep
from t3toolbox.backend.t3_constructors import t3_corewise_randn
from t3toolbox.backend.fitting import LeastSquaresBatch

from _typing_laws import JetLaw, InputLaw, assert_law, law_error

SHAPE = (4, 5, 6)
W, C = (3,), (2,)
ORDERS = {(2,): (0.7,), (1, 1): (0.7, 1.9), (2, 1): (0.7, 1.9)}


def box(order):
    return tuple(n + 1 for n in order)


class TestDataMetric(unittest.TestCase):

    def test_the_jet_law(self):
        rng = np.random.default_rng(0)
        for (order, ell), gamma in itertools.product(ORDERS.items(), (None, 2.5)):
            law, F = JetLaw(order, ell), box(order)
            r_probe = tuple(rng.standard_normal(F + W + C + (n,)) for n in SHAPE)
            omega_probe = tuple(0.5 + rng.random(F + (1,) * len(W) + C + (n,)) for n in SHAPE)
            r_apply, omega_apply = rng.standard_normal(F + W + C), 0.5 + rng.random(F + W + (1,))
            metric = lambda r, omega: jets.data_metric(r, omega, gamma)
            for kind, r, omega in (('probe', r_probe, omega_probe), ('apply', r_apply, omega_apply)):
                with self.subTest(kind=kind, order=order, gamma=repr(gamma)):
                    assert_law(self, metric, (r, omega), (law.P(r), law.P_star(omega)), law.P_star)
                    assert_law(self, lambda r: jets.data_metric(r, None, gamma), (law.P_star(r),),
                               (law.P_star(law.P_star(r)),), law.P_star)     # no weight: M is gamma, P* to P*

    def test_the_input_law(self):
        rng = np.random.default_rng(1)
        law = InputLaw([0.5 + 1.5 * rng.random(n) for n in SHAPE])
        for order in ((), (2,), (1, 1)):
            F = box(order)
            r = tuple(rng.standard_normal(F + W + C + (n,)) for n in SHAPE)
            omega = tuple(0.5 + rng.random(F + W + (1,) + (n,)) for n in SHAPE)   # a full edge
            with self.subTest(order=order):
                assert_law(self, lambda r, omega: jets.data_metric(r, omega, 3.0), (r, omega),
                           (law.V_star(r), law.V(omega)), law.V)

    def test_blocks_to_weight_and_the_batch_method(self):
        rng = np.random.default_rng(2)
        for order, ell in ORDERS.items():
            law, F = JetLaw(order, ell), box(order)
            y_probe = tuple(rng.standard_normal(F + W + C + (n,)) for n in SHAPE)
            y_apply = rng.standard_normal(F + W + C)
            for kind, y, blocks, lead in (('probe', y_probe, 0.5 + rng.random((len(SHAPE),) + F), 1),
                                          ('apply', y_apply, 0.5 + rng.random(F), 0)):
                with self.subTest(kind=kind, order=order, function='blocks_to_weight'):
                    assert_law(self, lambda b: jets.blocks_to_weight(kind, order, b, y), (blocks,),
                               (law.P_star(blocks, lead=lead),), lambda w: law.P_star(w, lead=lead))
                weight = jets.blocks_to_weight(kind, order, blocks, y)
                method = lambda r, w: LeastSquaresBatch(kind, order, None, y, weight=w,
                                                        minibatch_scale=0.5).data_metric(r)
                with self.subTest(kind=kind, order=order, function='LeastSquaresBatch.data_metric'):
                    assert_law(self, method, (y, weight), (law.P(y), law.P_star(weight, lead=lead)), law.P_star)


class TestConverters(unittest.TestCase):

    def test_jet_to_dual_jet_and_back(self):
        rng = np.random.default_rng(3)
        for order, ell in ORDERS.items():
            law, F = JetLaw(order, ell), box(order)
            for kind, lead in (('probe', 1), ('apply', 0)):
                r = (tuple(rng.standard_normal(F + W + (n,)) for n in SHAPE) if kind == 'probe'
                     else rng.standard_normal(F + W))
                blocks = 0.5 + rng.random(((len(SHAPE),) if kind == 'probe' else ()) + F)     # per block
                weight = jets.blocks_to_weight(kind, order, blocks, r)           # of the type of z (M2 part 3)
                to_dual = lambda r, w: jets.jet_to_dual_jet(kind, order, r, w)
                to_jet = lambda c, w: jets.dual_jet_to_jet(kind, order, c, w)
                with self.subTest(kind=kind, order=order, reading='Y* -> Y turned'):
                    assert_law(self, to_dual, (r, weight), (law.P(r), law.P_star(weight, lead=lead)),
                               law.P_star_turned)
                    turned = to_dual(r, weight)
                    assert_law(self, to_jet, (turned, weight),
                               (law.P_star_turned(turned), law.P_star(weight, lead=lead)), law.P)
                with self.subTest(kind=kind, order=order, reading='Y -> Y turned (no weight)'):
                    assert_law(self, lambda c: to_dual(c, None), (r,), (law.P_star(r),), law.P_star_turned)
                    assert_law(self, lambda c: to_jet(c, None), (r,), (law.P_star_turned(r),), law.P_star)


class TestRescalings(unittest.TestCase):

    def setUp(self):
        np.random.seed(4)
        self.x = t3_corewise_randn(SHAPE, (2, 3, 2), (1, 2, 2, 1))

    def test_rescale_direction_is_the_jet_law(self):
        rng = np.random.default_rng(5)
        for order, ell in ORDERS.items():
            law, F = JetLaw(order, ell), box(order)
            ww, pp, qq = ([rng.standard_normal(W + (n,)) for n in SHAPE] for _ in range(3))
            sample = jets.affine_jet(ww, pp) if len(order) == 1 else jets.affine_jet(ww, pp, qq)
            for kind, forward in (('probe', probing.t3_probe), ('apply', bapply.t3_apply)):
                z = forward(self.x, sample, order=order)
                moved_sample, moved_y = prep.rescale_direction(kind, order, sample, z, ell)
                with self.subTest(kind=kind, order=order):
                    for a, b in zip(moved_sample, law.inputs(sample)):        # the sample: the jet law
                        self.assertLessEqual(np.linalg.norm(a - b) / np.linalg.norm(b), 1e-14)
                    self.assertLessEqual(law_error(lambda y: y, (z,), (moved_y,), law.P), 1e-14)
                    assert_law(self, lambda s: forward(self.x, s, order=order), (sample,), (moved_sample,), law.P)

    def test_rescale_coordinates_and_unscale_are_the_input_law(self):
        rng = np.random.default_rng(6)
        law = InputLaw([0.5 + 1.5 * rng.random(n) for n in SHAPE])
        for order in ((), (2,), (1, 1)):
            ww, pp, qq = ([rng.standard_normal(W + (n,)) for n in SHAPE] for _ in range(3))
            sample = (ww if not order else jets.affine_jet(ww, pp) if len(order) == 1 else jets.affine_jet(ww, pp, qq))
            for kind, forward, move_z in (('probe', probing.t3_probe, law.V_star), ('apply', bapply.t3_apply, lambda z: z)):
                z = forward(self.x, sample, order=order)
                moved_sample, moved_y = prep.rescale_coordinates(kind, order, sample, z, law.deltas)
                with self.subTest(kind=kind, order=order):
                    self.assertLessEqual(law_error(lambda s: s, (sample,), (moved_sample,), law.inputs), 1e-14)
                    self.assertLessEqual(law_error(lambda y: y, (z,), (moved_y,), move_z), 1e-14)
                    assert_law(self, lambda x, s: forward(x, s, order=order), (self.x, sample),
                               (law.tucker(self.x), moved_sample), move_z)
        back = prep.t3_unscale_coordinates(law.tucker(self.x), law.deltas)
        self.assertLessEqual(law_error(lambda x: x, (self.x,), (back,), lambda x: x), 1e-14)

    def test_the_entries_and_the_transpose_laws_of_the_helper(self):
        """The helper's own entries and ``X*`` laws against the library: the entries forward at the moved point is the
        entries law; the apply corewise transpose of an unmoved ``c`` (apply ``c`` does not move) is ``X*`` on the
        Tucker blocks."""
        rng = np.random.default_rng(7)
        law = InputLaw([0.5 + 1.5 * rng.random(n) for n in SHAPE])
        import t3toolbox.backend.entries as bentries
        index = np.stack([rng.integers(0, n, size=W) for n in SHAPE])
        assert_law(self, lambda x: bentries.t3_entries(x, index), (self.x,), (law.tucker(self.x),),
                   lambda z: law.entries(z, index))
        ww = [rng.standard_normal(W + (n,)) for n in SHAPE]
        c = rng.standard_normal(W)
        transpose = lambda x, s: bapply.t3_apply_corewise_transpose(x, s, c)[0]
        assert_law(self, transpose, (self.x, ww), (law.tucker(self.x), law.inputs(ww)), law.tucker_dual)


class TestTheLawsCatch(unittest.TestCase):
    """Each fault fails its law at O(1): the laws are worth what they catch."""

    FAULT = 1e-3

    def test_the_jet_law_catches_a_missing_square_a_missing_flip_a_one_axis_flip_and_a_swap(self):
        rng = np.random.default_rng(8)
        order, ell = (1, 1), (0.7, 1.9)
        law, F = JetLaw(order, ell), box(order)
        r, omega = rng.standard_normal(F + W), 0.5 + rng.random(F + (1,))
        blocks = 0.5 + rng.random(F)
        no_square = lambda r, w: w * r
        self.assertGreater(law_error(no_square, (r, omega), (law.P(r), law.P_star(omega)), law.P_star), self.FAULT)
        no_flip = lambda r, w: jets.data_metric(r, jets.blocks_to_weight('apply', order, w, r), None)
        self.assertGreater(law_error(no_flip, (r, blocks), (law.P(r), law.P_star(blocks)), law.P_star_turned),
                           self.FAULT)
        one_axis = lambda r, w: no_flip(r, w)[::-1]                             # one jet axis flipped of the two
        self.assertGreater(law_error(one_axis, (r, blocks), (law.P(r), law.P_star(blocks)), law.P_star_turned),
                           self.FAULT)
        swapped = lambda r, w: jets.jet_to_dual_jet('apply', order, r,       # a dual jet read as a jet
                                                    jets.blocks_to_weight('apply', order, w, r))
        self.assertGreater(law_error(swapped, (r, blocks), (law.P(r), law.P_star(blocks)), law.P), self.FAULT)

    def test_the_input_law_catches_c_read_as_a_measurement(self):
        rng = np.random.default_rng(9)
        law = InputLaw([0.5 + 1.5 * rng.random(n) for n in SHAPE])
        r = tuple(rng.standard_normal(W + (n,)) for n in SHAPE)
        omega = tuple(0.5 + rng.random(W + (n,)) for n in SHAPE)
        metric = lambda r, omega: jets.data_metric(r, omega, None)
        self.assertGreater(law_error(metric, (r, omega), (law.V_star(r), law.V(omega)), law.V_star), self.FAULT)
        # a probe "transpose" that weighs c as a measurement (by Delta^-1, as V_i*) instead of as V_i
        np.random.seed(10)
        x = t3_corewise_randn(SHAPE, (2, 3, 2), (1, 2, 2, 1))
        ww = [rng.standard_normal(W + (n,)) for n in SHAPE]
        transpose = lambda x, s, c: probing.t3_probe_corewise_transpose(x, s, c)[0]
        assert_law(self, transpose, (x, ww, r), (law.tucker(x), law.inputs(ww), law.V(r)), law.tucker_dual)
        self.assertGreater(law_error(transpose, (x, ww, r), (law.tucker(x), law.inputs(ww), law.V_star(r)),
                                     law.tucker_dual), self.FAULT)

    def test_the_helper_refuses_vacuous_settings(self):
        for order, ell in (((2,), (1.0,)), ((1, 1), (0.7, 0.7)), ((0, 0), (0.7, 1.9)), ((2,), (0.7, 1.9))):
            with self.subTest(order=order, ell=ell):
                with self.assertRaises(ValueError):
                    JetLaw(order, ell)


if __name__ == '__main__':
    unittest.main()
