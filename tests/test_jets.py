# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.jets``: the dual jet, derivative data and the checkers, in the jet shape; and their reach from
``t3toolbox.fitting`` and ``t3toolbox``.

The jet shape ``order = (n_1, ..., n_m)`` is a tuple; the box is ``F = (n_1 + 1, ..., n_m + 1)``; the entries kind
has no jets. The jet shapes tested: ``()``, ``(1,)``, ``(3,)``, ``(1, 2)`` and ``(2, 0)`` (a second variable of
order 0).

1. **The dual jet** (`TestDualJet`): ``jet_to_dual_jet`` against its definition written out slot by slot (a flip of
   every jet axis); the round trip through ``dual_jet_to_jet``; **the pairing identity** at ``m = 1`` and
   ``m = 2``, with the corner coefficient of the product taken here by explicit loops over multi-indices (and at
   ``m = 1`` by ``numpy.polynomial``), routes the function does not take; the backend inverse at a zero weight (it
   divides, and raises and warns nothing).
2. **The frontend has no dual-jet converter** (M1, 9.2): the engine's turned form is the backend's. The data metric
   and its weight are tested in ``test_data_metric.py``.
3. **Derivative data** (`TestDerivativeData`): the conversions against mixed partial derivatives of a known
   polynomial in two variables (``numpy.polynomial.polyder`` along each axis), against an analytic function, and
   against derivatives of the dense tensor's outputs; the weight conversion and ``prepare_derivative_data``
   reproducing the objective written on derivatives; the weight ``k_1! ... k_m!`` when none is given, for apply and
   for probe with its mode axis; the entries kind with ``()`` returns its data as they are, and with a jet shape
   raises.
4. **The checkers** (`TestCheckers`): each passes every good input and raises on each wrong form; the trap of a
   forgotten jet shape; no backend module calls one; the validators of ``pp`` are gone.
5. **The reach from the frontend** (`TestFrontendReach`): the names of ``t3toolbox.fitting`` and of ``t3toolbox``.

Numerical comparisons are to a tolerance, structural ones exact. numpy only, by the house convention; the functions
under jit are in ``test_dispatch.py``.
"""
import ast
import inspect
import itertools
import math
import pathlib
import unittest
import warnings

import numpy as np

import t3toolbox
import t3toolbox.fitting as fitting
import t3toolbox.backend.apply as bapply
import t3toolbox.backend.jets as jets
import t3toolbox.backend.probing as probing

TOL = 1e-13
SHAPE = (4, 5, 6)           # the mode sizes N_i
D = len(SHAPE)
JET_SHAPES = ((), (1,), (3,), (1, 2), (2, 0))


def relerr(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b))


def F_of(order):
    return tuple(n + 1 for n in order)


def corner_minus(order, k):
    """The multi-index ``order - k``: where slot ``k`` of a jet goes when every jet axis is flipped."""
    return tuple(n - k_j for n, k_j in zip(order, k))


def random_jet(rng, kind, order, W, extra=()):
    """An object of the type of z: probe len=d, elm_shape=F+W+extra+(Ni,); apply F+W+extra; entries W+extra."""
    if kind == 'probe':
        return [rng.standard_normal(F_of(order) + W + extra + (N,)) for N in SHAPE]
    return rng.standard_normal(F_of(order) + W + extra)


def random_weight(rng, kind, order, W=(), extra=()):
    """A weight of the type of z, one per entry (data-metric arc): probe a list, one array F+W+extra+(Ni,) per mode;
    apply F+W+extra; entries W+extra."""
    if kind == 'probe':
        return [0.5 + rng.random(F_of(order) + W + extra + (N,)) for N in SHAPE]
    return 0.5 + rng.random(F_of(order) + W + extra)


def per_mode(kind, value):
    """The arrays of an object of the type of z, as a list (one per mode for probe, one otherwise)."""
    return list(value) if kind == 'probe' else [value]


def mode_weights(kind, weight):
    """The weight of each array of an object of the type of z: its mode's array for probe, the weight otherwise."""
    return [weight[i] for i in range(len(weight))] if kind == 'probe' else [weight]


def mapped(kind, function, value):
    """``function`` applied to each array of an object of the type of z, in its form (a list for probe)."""
    return [function(a) for a in value] if kind == 'probe' else function(value)


def factorials(order):
    """``k_1! ... k_m!`` on the box, written out slot by slot."""
    out = np.zeros(F_of(order))
    for k in np.ndindex(*F_of(order)):
        out[k] = math.prod(math.factorial(k_j) for k_j in k)
    return out


CASES = [(kind, order, W, extra)
         for kind in ('probe', 'apply', 'entries')
         for order in (JET_SHAPES if kind != 'entries' else ((),))
         for W, extra in (((7,), ()), ((), ()), ((2, 3), ()), ((7,), (2, 3)))]     # extra: the stacks K + C


class TestDualJet(unittest.TestCase):

    def test_the_dual_jet_is_the_weights_then_the_turn(self):
        """``dual_jet[k] = weight[order - k]**2 * jet[order - k]``, slot by slot, every jet axis flipped; with no
        jets, ``weight**2 * jet``."""
        rng = np.random.default_rng(0)
        for kind, order, W, extra in CASES:
            jet, weight = random_jet(rng, kind, order, W, extra), random_weight(rng, kind, order, W, extra)
            dual = jets.jet_to_dual_jet(kind, order, jet, weight)
            with self.subTest(kind=kind, order=order, W=W, extra=extra):
                self.assertIsInstance(dual, tuple if kind == 'probe' else (np.ndarray, np.generic))   # W = (): a numpy scalar
                for jet_FX, dual_FX, omega_F in zip(per_mode(kind, jet), per_mode(kind, dual), mode_weights(kind, weight)):
                    self.assertEqual(dual_FX.shape, jet_FX.shape)
                    for k in np.ndindex(*F_of(order)):
                        flipped = corner_minus(order, k)
                        self.assertLessEqual(relerr(dual_FX[k], omega_F[flipped] ** 2 * jet_FX[flipped]), TOL)

    def test_no_weight_is_ones(self):
        rng = np.random.default_rng(1)
        for kind, order, W, extra in CASES:
            jet = random_jet(rng, kind, order, W, extra)
            ones = mapped(kind, np.ones_like, jet)
            with self.subTest(kind=kind, order=order, W=W, extra=extra):
                for a, b, c in zip(per_mode(kind, jets.jet_to_dual_jet(kind, order, jet, None)),
                                   per_mode(kind, jets.jet_to_dual_jet(kind, order, jet, ones)),
                                   per_mode(kind, jets.dual_jet_to_jet(kind, order, jet, None))):
                    self.assertTrue(np.array_equal(a, b))
                    self.assertTrue(np.array_equal(a, c))        # with no weight both directions are the turn

    def test_the_round_trip(self):
        """``dual_jet_to_jet(jet_to_dual_jet(x)) == x`` for a nonzero weight, and the other way round."""
        rng = np.random.default_rng(2)
        for kind, order, W, extra in CASES:
            jet, weight = random_jet(rng, kind, order, W, extra), random_weight(rng, kind, order, W, extra)
            there_and_back = jets.dual_jet_to_jet(kind, order, jets.jet_to_dual_jet(kind, order, jet, weight), weight)
            back_and_there = jets.jet_to_dual_jet(kind, order, jets.dual_jet_to_jet(kind, order, jet, weight), weight)
            with self.subTest(kind=kind, order=order, W=W, extra=extra):
                for x_FX, a_FX, b_FX in zip(per_mode(kind, jet), per_mode(kind, there_and_back),
                                            per_mode(kind, back_and_there)):
                    self.assertLessEqual(relerr(a_FX, x_FX), TOL)
                    self.assertLessEqual(relerr(b_FX, x_FX), TOL)
        # on a packed probe array too
        packed, weight = rng.standard_normal((D, 2, 3, 5, 6)), 0.5 + rng.random((D, 2, 3, 1, 6))   # constant in W
        back = jets.dual_jet_to_jet('probe', (1, 2), jets.jet_to_dual_jet('probe', (1, 2), packed, weight), weight)
        self.assertLessEqual(relerr(np.stack(back), packed), TOL)

    def test_the_pairing_identity(self):
        """``sum_k weight[k]**2 a[k] b[k]`` is the coefficient of the corner ``s^order`` of ``jet_to_dual_jet(a) * b``
        as polynomials in ``s``, at ``m = 1`` and ``m = 2``. The product is computed here by explicit loops over the
        pairs of multi-indices of the two (every pair, no truncation), and its corner read off; at ``m = 1`` also
        by ``numpy.polynomial``, one entry at a time."""
        rng = np.random.default_rng(3)
        for kind in ('probe', 'apply'):
            for order in ((1,), (2,), (5,), (1, 2), (2, 3), (2, 0)):
                a, b = random_jet(rng, kind, order, (3,)), random_jet(rng, kind, order, (3,))
                weight = random_weight(rng, kind, order, (3,))
                dual = jets.jet_to_dual_jet(kind, order, a, weight)
                m, F = len(order), F_of(order)
                with self.subTest(kind=kind, order=order):
                    for dual_FX, a_FX, b_FX, omega_F in zip(per_mode(kind, dual), per_mode(kind, a), per_mode(kind, b),
                                                            mode_weights(kind, weight)):
                        corner = 0.0
                        for k in np.ndindex(*F):                    # the product's coefficient at the sum k + l
                            for l in np.ndindex(*F):
                                if tuple(k_j + l_j for k_j, l_j in zip(k, l)) == tuple(order):
                                    corner = corner + dual_FX[k] * b_FX[l]
                        weighted = sum(omega_F[k] ** 2 * a_FX[k] * b_FX[k] for k in np.ndindex(*F))
                        self.assertLessEqual(relerr(corner, weighted), TOL)
                        if m == 1:
                            n = order[0]
                            flat = lambda x_FX: x_FX.reshape(n + 1, -1)
                            top = np.array([np.polynomial.polynomial.polymul(p, q)[n]
                                            for p, q in zip(flat(dual_FX).T, flat(b_FX).T)])
                            self.assertLessEqual(relerr(top.reshape(weighted.shape), weighted), TOL)
        # no jets: the pairing is the plain product
        a, b, weight = rng.standard_normal(3), rng.standard_normal(3), np.asarray([1.5])
        self.assertLessEqual(relerr(jets.jet_to_dual_jet('apply', (), a, weight) * b, 1.5 ** 2 * a * b), TOL)

    def test_the_weight_enters_squared_and_every_jet_axis_is_turned(self):
        """Wrong readings of the definition are told apart from the right one: a plain weight, no turn, and a turn of
        one jet axis only."""
        jet, weight = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 5.0])
        self.assertEqual(jets.jet_to_dual_jet('apply', (2,), jet, weight).tolist(), [75.0, 18.0, 4.0])  # (4, 18, 75), turned
        jet2 = np.arange(1.0, 7.0).reshape(2, 3)
        self.assertEqual(jets.jet_to_dual_jet('apply', (1, 2), jet2, None).tolist(), [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])

    def test_the_backend_inverse_divides_by_a_zero_weight_and_raises_nothing(self):
        rng = np.random.default_rng(4)
        for kind in ('probe', 'apply'):
            for order, zero_slot in (((2,), (1,)), ((1, 2), (0, 1))):
                jet, weight = random_jet(rng, kind, order, (5,)), random_weight(rng, kind, order, (5,))
                for omega in (weight if kind == 'probe' else [weight]):
                    omega[zero_slot] = 0.0                                # one slot, in every mode
                with self.subTest(kind=kind, order=order):
                    with warnings.catch_warnings():
                        warnings.simplefilter('error')                    # a warning would be an error here
                        back = jets.dual_jet_to_jet(kind, order, jet, weight)
                        all_zero = jets.dual_jet_to_jet(kind, order, jet, mapped(kind, np.zeros_like, weight))
                    reference = jets.dual_jet_to_jet(kind, order, jet,
                                                     mapped(kind, lambda w: np.where(w == 0.0, 1.0, w), weight))
                    for back_FX, zero_FX, reference_FX in zip(per_mode(kind, back), per_mode(kind, all_zero),
                                                              per_mode(kind, reference)):
                        self.assertFalse(np.isfinite(back_FX[zero_slot]).any())         # the slot of the zero weight
                        for k in np.ndindex(*F_of(order)):                                 # the others untouched
                            if k != zero_slot:
                                self.assertTrue(np.array_equal(back_FX[k], reference_FX[k]))
                        self.assertFalse(np.isfinite(zero_FX).any())
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertTrue(np.isnan(jets.dual_jet_to_jet('apply', (), np.zeros(3), np.zeros(1))).all())    # 0 / 0

    def test_a_converter_refuses_a_weight_not_of_the_type_of_the_jet(self):
        """The converters read the weight as :py:func:`data_metric` does (the data-metric arc): a weight of another
        rank (the old block form; or against a jet with a stack of directions ``K``), an axis neither 1 nor the jet's
        size, or another mode count raises a ``ValueError``; with no weight any jet is turned, ``K`` included. They
        check the pairing with the jet, not the sample: that is ``check_weight``'s."""
        rng = np.random.default_rng(5)
        jet = rng.standard_normal((3, 4))                                       # apply at (2,): F + W
        with_K = rng.standard_normal((3, 4, 2))                                 # F + W + K
        probe = random_jet(rng, 'probe', (2,), (4,))
        bad = {'the block form F': ('apply', jet, np.ones(3)),
               'a jet with K': ('apply', with_K, np.ones((3, 4))),
               'an axis neither 1 nor its size': ('apply', jet, np.ones((3, 2))),
               'a tuple for apply': ('apply', jet, (np.ones((3, 4)),)),
               'the probe block form (d,) + F': ('probe', probe, np.ones((D, 3))),
               'd - 1 arrays': ('probe', probe, [np.ones((3, 4, 1))] * (D - 1)),
               'a mode axis of 1': ('probe', probe, np.ones((1, 3, 4, 1)))}
        for label, (kind, value, weight) in bad.items():
            for function in (jets.jet_to_dual_jet, jets.dual_jet_to_jet):
                with self.subTest(weight=label, function=function.__name__):
                    with self.assertRaisesRegex(ValueError, '^%s: ' % function.__name__):     # its own message
                        function(kind, (2,), value, weight)
        for function in (jets.jet_to_dual_jet, jets.dual_jet_to_jet):
            self.assertEqual(function('apply', (2,), with_K, None).shape, with_K.shape)
        with self.assertRaises(ValueError):
            jets.prepare_derivative_data('apply', (2,), jet, np.ones(3))

    def test_kind_decides_the_type_and_a_packed_probe_array_is_taken_mode_by_mode(self):
        """``kind`` says which type the object is of; the object is not looked at. With ``kind='probe'`` the
        functions go over the modes along the first axis, so a list, a tuple and an array packed for the uniform
        layout, ``(d,) + F + W + (N,)``, give the same result, returned as a tuple, one array per mode. The values
        are checked against the definition written out on the packed array, at ``(3,)`` and ``(1, 2)``."""
        rng = np.random.default_rng(13)
        d, W, N = 3, (5,), 6
        for order in ((3,), (1, 2)):
            m, F = len(order), F_of(order)
            packed = rng.standard_normal((d,) + F + W + (N,))
            weight = 0.5 + rng.random((d,) + F + (1, 1))         # per mode and slot, mode-leading
            ragged = [packed[i] for i in range(d)]
            fact = factorials(order).reshape((1,) + F + (1, 1))
            w2 = weight ** 2
            flip = (slice(None),) + (slice(None, None, -1),) * m
            wanted = {'jet_to_dual_jet': (w2 * packed)[flip],                  # the weights, then the turn, per mode
                      'dual_jet_to_jet': packed[flip] / w2,
                      'derivatives_to_taylor': packed / fact,
                      'taylor_to_derivatives': packed * fact}
            calls = {'jet_to_dual_jet': lambda x: jets.jet_to_dual_jet('probe', order, x, weight),
                     'dual_jet_to_jet': lambda x: jets.dual_jet_to_jet('probe', order, x, weight),
                     'derivatives_to_taylor': lambda x: jets.derivatives_to_taylor('probe', order, x),
                     'taylor_to_derivatives': lambda x: jets.taylor_to_derivatives('probe', order, x)}
            for name, call in calls.items():
                from_packed, from_tuple, from_list = call(packed), call(tuple(packed)), call(ragged)
                with self.subTest(function=name, order=order):
                    for out in (from_packed, from_tuple, from_list):
                        self.assertEqual((type(out), len(out)), (tuple, d))            # a tuple, one array per mode
                    for a, b, c in zip(from_packed, from_tuple, from_list):
                        self.assertEqual(a.shape, F + W + (N,))
                        self.assertTrue(np.array_equal(a, b) and np.array_equal(a, c))
                    self.assertLessEqual(relerr(np.stack(from_packed), wanted[name]), TOL)
            # prepare_derivative_data follows: a packed array with kind='probe' gives the tuple, and the factorials
            y, taylor_weight, validation = jets.prepare_derivative_data('probe', order, packed, validation=packed)
            self.assertEqual((type(y), len(y), type(validation), taylor_weight.shape), (tuple, d, tuple, (d,) + F + (1, 1)))
            self.assertLessEqual(relerr(np.stack(y), wanted['derivatives_to_taylor']), TOL)
        # one mode: the probe type is still a tuple, of one array (the type comes from kind, not from the count)
        one = rng.standard_normal((4,) + W + (N,))
        for out in (jets.jet_to_dual_jet('probe', (3,), [one], 0.5 + rng.random((1, 4, 1, 1))), jets.dual_jet_to_jet('probe', (3,), (one,), None),
                    jets.derivatives_to_taylor('probe', (3,), one[None]), jets.taylor_to_derivatives('probe', (3,), [one])):
            self.assertEqual((type(out), len(out), out[0].shape), (tuple, 1, one.shape))
        self.assertEqual(type(jets.derivatives_to_taylor('apply', (3,), one)), np.ndarray)

    def test_an_object_of_the_other_type_than_kind_says_is_not_guarded_against(self):
        """What the docstrings say happens, pinned: a tuple given with ``kind='apply'`` is made into one array.
        Where the modes differ in size that fails; where they do not, the mode axis is read as the jet axis: an
        error from broadcasting, or a wrong result with no error. And an unknown kind raises."""
        rng = np.random.default_rng(15)
        order = (2,)
        ragged = [rng.standard_normal((3, 5, N)) for N in SHAPE]                    # the probe type, modes of sizes 4, 5, 6
        same = [rng.standard_normal((3, 5, 6)) for _ in range(4)]                   # four modes of one size
        weight = 0.5 + rng.random((3, 1, 1))
        for call in (lambda x: jets.jet_to_dual_jet('apply', order, x, weight),
                     lambda x: jets.dual_jet_to_jet('apply', order, x, None),
                     lambda x: jets.derivatives_to_taylor('apply', order, x),
                     lambda x: jets.taylor_to_derivatives('apply', order, x)):
            with self.assertRaises(ValueError):                                     # the modes cannot be stacked
                call(tuple(ragged))
        with self.assertRaises(ValueError):                                         # 4 modes against F = (3,)
            jets.derivatives_to_taylor('apply', order, tuple(same))
        with self.assertRaises(ValueError):
            jets.jet_to_dual_jet('apply', order, tuple(same), weight)
        wrong = jets.jet_to_dual_jet('apply', order, tuple(same), None)             # no error: the MODES are turned
        self.assertEqual(type(wrong), np.ndarray)
        self.assertTrue(np.array_equal(wrong, np.stack(same)[::-1]))
        for function, args in ((jets.jet_to_dual_jet, (same[0], weight)), (jets.dual_jet_to_jet, (same[0], weight)),
                               (jets.derivatives_to_taylor, (same[0],)), (jets.taylor_to_derivatives, (same[0],))):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, 'unknown sampling kind'):
                    function('probes', order, *args)

    def test_the_jet_shape_is_a_tuple_and_the_entries_kind_has_none(self):
        """Every function of the module that takes a jet shape reads it with the library's one reader: an integer,
        a list or a negative entry raises; a numpy integer is accepted. The entries kind takes ``()`` only (O-47:
        its data then come back as they are)."""
        rng = np.random.default_rng(16)
        y, weight = rng.standard_normal((3, 4)), 0.5 + rng.random((3, 1))
        calls = {'jet_to_dual_jet': lambda kind, order: jets.jet_to_dual_jet(kind, order, y, None),
                 'dual_jet_to_jet': lambda kind, order: jets.dual_jet_to_jet(kind, order, y, None),
                 'derivatives_to_taylor': lambda kind, order: jets.derivatives_to_taylor(kind, order, y),
                 'taylor_to_derivatives': lambda kind, order: jets.taylor_to_derivatives(kind, order, y),
                 'prepare_derivative_data': lambda kind, order: jets.prepare_derivative_data(kind, order, y),
                 'derivative_weight_to_taylor_weight': lambda kind, order: jets.derivative_weight_to_taylor_weight(kind, order, weight)}
        for name, call in calls.items():
            with self.subTest(function=name):
                for bad in (2, [2]):
                    with self.assertRaisesRegex(TypeError, 'order is the jet shape'):
                        call('apply', bad)
                with self.assertRaises(ValueError):
                    call('apply', (-1,))
                first = lambda value: value[0] if isinstance(value, tuple) and name == 'prepare_derivative_data' else value
                self.assertTrue(np.array_equal(first(call('apply', (np.int64(2),))), first(call('apply', (2,)))))
                with self.assertRaisesRegex(ValueError, 'the entries kind has no jets'):
                    call('entries', (2,))
        # the entries kind with no jets: its data as they are, and a weight of ones
        index_data = rng.standard_normal((5,))
        for function in (jets.derivatives_to_taylor, jets.taylor_to_derivatives):
            self.assertTrue(np.array_equal(function('entries', (), index_data), index_data))
        self.assertTrue(np.array_equal(jets.jet_to_dual_jet('entries', (), index_data, np.asarray([2.0])), 4.0 * index_data))
        y_taylor, weight_taylor, _ = jets.prepare_derivative_data('entries', (), index_data)
        self.assertTrue(np.array_equal(y_taylor, index_data))
        self.assertEqual((weight_taylor.shape, float(weight_taylor[0])), ((1,), 1.0))      # one number, of the type of z

    def test_the_four_functions_take_kind_then_order(self):
        """The signatures: ``(kind, order, ...)``, as every other function of the two modules that takes a kind;
        ``weight`` is required (pass ``None`` for no weight)."""
        wanted = {'jet_to_dual_jet': ['kind', 'order', 'jet', 'weight'],
                  'dual_jet_to_jet': ['kind', 'order', 'dual_jet', 'weight'],
                  'derivatives_to_taylor': ['kind', 'order', 'jets'],
                  'taylor_to_derivatives': ['kind', 'order', 'jets'],
                  'derivative_weight_to_taylor_weight': ['kind', 'order', 'weight']}    # gains kind (M2)
        for name, parameters in wanted.items():
            signature = inspect.signature(getattr(jets, name))
            with self.subTest(function=name):
                self.assertEqual(list(signature.parameters), parameters)
                self.assertTrue(all(p.default is inspect.Parameter.empty for p in signature.parameters.values()))

    def test_a_zero_dimensional_input_gives_an_array(self):
        """The apply kind with no jets and no sample stack: numpy in gives a numpy array out, not a numpy scalar."""
        for out in (jets.jet_to_dual_jet('apply', (), np.asarray(2.0), np.asarray(3.0)), jets.jet_to_dual_jet('apply', (), 2.0, 3.0),
                    jets.dual_jet_to_jet('apply', (), np.asarray(2.0), np.asarray(3.0)), jets.dual_jet_to_jet('apply', (), np.asarray(2.0), None),
                    jets.derivatives_to_taylor('apply', (), np.asarray(2.0)), jets.taylor_to_derivatives('apply', (), np.asarray(2.0)),
                    jets.derivative_weight_to_taylor_weight('apply', (), np.asarray(2.0))):
            self.assertIs(type(out), np.ndarray)
            self.assertEqual(out.shape, ())
        self.assertEqual(float(jets.jet_to_dual_jet('apply', (), np.asarray(2.0), np.asarray(3.0))), 18.0)

    def test_the_result_has_the_lower_precision_of_the_jet_and_the_weight(self):
        jet64, weight64 = np.ones((2, 3, 4)), np.ones((2, 3, 1))
        for function in (jets.jet_to_dual_jet, jets.dual_jet_to_jet):
            for jet_dtype, weight_dtype in itertools.product((np.float32, np.float64), repeat=2):
                expected = np.float32 if np.float32 in (jet_dtype, weight_dtype) else np.float64
                with self.subTest(function=function.__name__, jet=jet_dtype.__name__, weight=weight_dtype.__name__):
                    out = function('apply', (1, 2), jet64.astype(jet_dtype), weight64.astype(weight_dtype))
                    self.assertEqual(out.dtype, expected)
                    self.assertEqual(function('apply', (1, 2), jet64.astype(jet_dtype), None).dtype, jet_dtype)


def interpolated_derivatives(values_of, order, n_points):
    """The derivatives at ``s = 0``, orders ``0 .. order``, of the polynomial through ``values_of(s)`` at
    ``n_points`` Chebyshev points: fitted and differentiated by ``numpy.polynomial``. shape=(order+1,)+X."""
    nodes = np.cos(np.pi * (np.arange(n_points) + 0.5) / n_points)
    values_mX = np.stack([values_of(s) for s in nodes])
    flat_mx = values_mX.reshape(n_points, -1)
    out_Tx = np.zeros((order + 1, flat_mx.shape[1]))
    for k in range(flat_mx.shape[1]):
        polynomial = np.polynomial.Polynomial(np.polynomial.polynomial.polyfit(nodes, flat_mx[:, k], n_points - 1))
        for t in range(order + 1):
            out_Tx[t, k] = polynomial.deriv(t)(0.0) if t else polynomial(0.0)
    return out_Tx.reshape((order + 1,) + values_mX.shape[1:])


class TestDerivativeData(unittest.TestCase):

    def test_an_analytic_function(self):
        """``exp(a s)``: derivatives ``a**t``, Taylor coefficients ``a**t / t!``; and ``exp(a s_1 + b s_2)``: mixed
        derivatives ``a**k_1 b**k_2``, Taylor coefficients ``a**k_1 b**k_2 / (k_1! k_2!)``."""
        a, b = 1.7, -0.6
        derivatives = np.array([[a ** t] for t in range(7)])
        taylor = np.array([[a ** t / math.factorial(t)] for t in range(7)])
        self.assertLessEqual(relerr(jets.derivatives_to_taylor('apply', (6,), derivatives), taylor), TOL)
        self.assertLessEqual(relerr(jets.taylor_to_derivatives('apply', (6,), taylor), derivatives), TOL)
        derivatives = np.array([[a ** k1 * b ** k2 for k2 in range(5)] for k1 in range(4)])
        taylor = np.array([[a ** k1 * b ** k2 / (math.factorial(k1) * math.factorial(k2)) for k2 in range(5)]
                           for k1 in range(4)])
        self.assertLessEqual(relerr(jets.derivatives_to_taylor('apply', (3, 4), derivatives), taylor), TOL)
        self.assertLessEqual(relerr(jets.taylor_to_derivatives('apply', (3, 4), taylor), derivatives), TOL)

    def test_against_mixed_partial_derivatives_of_a_polynomial_in_two_variables(self):
        """A known polynomial ``p(s_1, s_2) = sum_k c[k] s_1^k_1 s_2^k_2`` (a matrix of coefficients per sample):
        its mixed partial derivatives at 0 are taken here by ``numpy.polynomial.polynomial.polyder`` along each axis
        in turn, and ``taylor_to_derivatives`` of the coefficients must be them; ``derivatives_to_taylor`` the
        reverse. For apply and for probe."""
        rng = np.random.default_rng(17)
        order = (3, 2)
        F = F_of(order)
        for kind in ('apply', 'probe'):
            coefficients = random_jet(rng, kind, order, (2,))
            with self.subTest(kind=kind):
                for c_FX, derivatives_FX, taylor_FX in zip(
                        per_mode(kind, coefficients), per_mode(kind, jets.taylor_to_derivatives(kind, order, coefficients)),
                        per_mode(kind, jets.derivatives_to_taylor(kind, order,
                                                                  jets.taylor_to_derivatives(kind, order, coefficients)))):
                    for k in np.ndindex(*F):
                        mixed = np.polynomial.polynomial.polyder(c_FX, m=k[0], axis=0)
                        mixed = np.polynomial.polynomial.polyder(mixed, m=k[1], axis=1)
                        self.assertLessEqual(relerr(derivatives_FX[k], mixed[0, 0]), TOL)
                    self.assertLessEqual(relerr(taylor_FX, c_FX), TOL)

    def test_against_derivatives_of_the_dense_tensor_s_outputs(self):
        """The library's jets are Taylor coefficients (``dense_probe``, ``dense_apply``). The derivatives of the
        same outputs along ``ww + s pp`` are taken here from an interpolating polynomial, by ``numpy.polynomial``;
        ``derivatives_to_taylor`` of them must be the library's jets, and ``taylor_to_derivatives`` the reverse."""
        rng = np.random.default_rng(6)
        T = rng.standard_normal(SHAPE)
        ww = [rng.standard_normal((3, N)) for N in SHAPE]
        pp = [rng.standard_normal((3, N)) for N in SHAPE]
        along = lambda s: [w + s * p for w, p in zip(ww, pp)]
        order = (3,)                                                # the outputs have degree d = 3 in s (probe: 2)
        apply_derivatives = interpolated_derivatives(lambda s: bapply.dense_apply(T, along(s)), 3, D + 2)
        apply_taylor = bapply.dense_apply(T, jets.affine_jet(ww, pp), order)
        self.assertLessEqual(relerr(jets.derivatives_to_taylor('apply', order, apply_derivatives), apply_taylor), 1e-11)
        self.assertLessEqual(relerr(jets.taylor_to_derivatives('apply', order, apply_taylor), apply_derivatives), 1e-11)
        probe_taylor = probing.dense_probe(T, jets.affine_jet(ww, pp), order)
        probe_derivatives = [interpolated_derivatives(lambda s: probing.dense_probe(T, along(s))[i], 3, D + 2)
                             for i in range(D)]
        converted = jets.derivatives_to_taylor('probe', order, probe_derivatives)
        self.assertEqual(type(converted), tuple)
        for i in range(D):
            self.assertLessEqual(relerr(converted[i][:3], probe_taylor[i][:3]), 1e-11)
            self.assertLessEqual(np.linalg.norm(converted[i][3]), 1e-11 * np.linalg.norm(probe_taylor[i][0]))
            self.assertLessEqual(relerr(jets.taylor_to_derivatives('probe', order, probe_taylor)[i][:3],
                                        probe_derivatives[i][:3]), 1e-11)

    def test_slot_by_slot_and_the_round_trip(self):
        rng = np.random.default_rng(7)
        for kind, order, W, extra in CASES:
            jet = random_jet(rng, kind, order, W, extra)
            taylor = jets.derivatives_to_taylor(kind, order, jet)
            derivatives = jets.taylor_to_derivatives(kind, order, jet)
            back = jets.taylor_to_derivatives(kind, order, taylor)
            with self.subTest(kind=kind, order=order, W=W, extra=extra):
                self.assertIsInstance(taylor, tuple if kind == 'probe' else (np.ndarray, np.generic))
                for x_FX, taylor_FX, derivatives_FX, back_FX in zip(per_mode(kind, jet), per_mode(kind, taylor),
                                                                   per_mode(kind, derivatives), per_mode(kind, back)):
                    self.assertEqual(taylor_FX.shape, x_FX.shape)
                    self.assertLessEqual(relerr(back_FX, x_FX), TOL)
                    if order == ():
                        self.assertTrue(np.array_equal(taylor_FX, x_FX))
                        self.assertTrue(np.array_equal(derivatives_FX, x_FX))
                    for k in (np.ndindex(*F_of(order)) if order else ()):
                        factor = math.prod(math.factorial(k_j) for k_j in k)
                        self.assertLessEqual(relerr(taylor_FX[k], x_FX[k] / factor), TOL)
                        self.assertLessEqual(relerr(derivatives_FX[k], x_FX[k] * factor), TOL)

    def test_the_weight_goes_the_opposite_way_on_its_jet_axes(self):
        """The factorials by hand, on the LEADING jet axes of a weight of the type of z (after the mode axis of a
        mode-leading probe array), entry by entry: per entry, per sample (jet axes of 1, which come back full), and a
        mode-leading probe array; the sizes of ``F``, ``W`` and the edge distinct, and equal (``F = W = (3,)`` against
        an edge of 3), where a factor on the wrong axis cannot fail by shape. The form comes back as given."""
        rng = np.random.default_rng(8)
        for kind in ('probe', 'apply', 'entries'):
            for order in (JET_SHAPES if kind != 'entries' else ((),)) + (((2,),) if kind != 'entries' else ()):
                for W in ((5,), (3,)):
                    F, m = F_of(order), len(order)
                    edges = (3, 3, 3) if W == (3,) else SHAPE
                    per_entry = ([0.5 + rng.random(F + W + (N,)) for N in edges] if kind == 'probe'
                                 else 0.5 + rng.random(F + W))
                    per_sample = mapped(kind, lambda w: w[(slice(0, 1),) * m], per_entry)
                    forms = [('per entry', per_entry), ('per sample', per_sample)]
                    if kind == 'probe' and W == (3,):
                        forms.append(('mode-leading', np.stack(per_entry)))
                    for label, weight in forms:
                        converted = jets.derivative_weight_to_taylor_weight(kind, order, weight)
                        with self.subTest(kind=kind, order=order, W=W, form=label):
                            self.assertEqual(type(converted), tuple if isinstance(weight, list) else np.ndarray)
                            for got, given in zip(mode_weights(kind, converted), mode_weights(kind, weight)):
                                given = np.asarray(given)
                                self.assertEqual(got.shape, F + given.shape[m:])        # a jet axis of 1 comes back full
                                for index in np.ndindex(*got.shape):
                                    k = index[:m]
                                    factor = math.prod(math.factorial(k_j) for k_j in k)
                                    source = tuple(0 if n == 1 else i for n, i in zip(given.shape, index))
                                    self.assertEqual(got[index], given[source] * factor)

    def test_the_objective_written_on_derivatives_is_reproduced(self):
        """The draft's case 2, for apply and probe at ``(1,)``, ``(3,)`` and ``(1, 2)``: ``1/2 sum (weight *
        (derivative residual))**2`` equals the fit's objective ``1/2 sum (Taylor weight * (Taylor residual))**2``
        after ``prepare_derivative_data``. With no weight given: the plain sum of squares of the derivative
        residuals."""
        rng = np.random.default_rng(9)

        def objective(kind, order, weight, z, y):   # the misfit of the library: 1/2 sum (weight * (z - y))**2
            return 0.5 * sum(float(np.sum((np.asarray(omega_FX) * (z_FX - y_FX)) ** 2))     # positionally: z's type
                             for omega_FX, z_FX, y_FX in zip(mode_weights(kind, weight), per_mode(kind, z),
                                                            per_mode(kind, y)))

        for kind in ('probe', 'apply'):
            for order in ((1,), (3,), (1, 2)):
                z_taylor = random_jet(rng, kind, order, (5,))               # the model's output, as the library gives it
                z_derivatives = jets.taylor_to_derivatives(kind, order, z_taylor)
                y_derivatives = random_jet(rng, kind, order, (5,))
                weight_derivatives = random_weight(rng, kind, order, (5,))
                ones = mapped(kind, np.ones_like, weight_derivatives)
                with self.subTest(kind=kind, order=order):
                    wanted = objective(kind, order, weight_derivatives, z_derivatives, y_derivatives)
                    y, weight, validation = jets.prepare_derivative_data(kind, order, y_derivatives, weight_derivatives)
                    self.assertIsNone(validation)
                    self.assertLessEqual(abs(objective(kind, order, weight, z_taylor, y) - wanted), 1e-12 * wanted)
                    # the trap: no weight means ones on the DERIVATIVES
                    wanted = objective(kind, order, ones, z_derivatives, y_derivatives)
                    y, weight, _ = jets.prepare_derivative_data(kind, order, y_derivatives)
                    self.assertLessEqual(abs(objective(kind, order, weight, z_taylor, y) - wanted), 1e-12 * wanted)
                    # and it is NOT the objective with ones on the Taylor coefficients (the test can tell them
                    # apart where some factorial is not 1)
                    if max(order) >= 2:
                        self.assertGreater(abs(objective(kind, order, ones, z_taylor, y) - wanted), 1e-3 * wanted)

    def test_no_weight_returns_the_factorials(self):
        """``k_1! ... k_m!`` per slot, in the form of ``blocks_to_weight`` (size 1 off the jet axes): for apply ``F +
        (1,)``; for probe one mode-leading array ``(d,) + F + (1,) + (1,)``, every mode's factorials. A weight of the
        kind (``check_weight``)."""
        rng = np.random.default_rng(10)
        for order in ((1,), (2,), (5,), (1, 2), (2, 0)):
            for kind in ('probe', 'apply'):
                y = random_jet(rng, kind, order, (4,))
                y_taylor, weight, validation = jets.prepare_derivative_data(kind, order, y)
                with self.subTest(kind=kind, order=order):
                    if kind == 'probe':
                        expected = np.stack([factorials(order)] * D).reshape((D,) + F_of(order) + (1, 1))
                    else:
                        expected = factorials(order).reshape(F_of(order) + (1,))
                    self.assertEqual(weight.shape, expected.shape)
                    self.assertEqual(weight.tolist(), expected.tolist())
                    self.assertEqual(weight.dtype, np.float64)
                    sample = [rng.standard_normal(F_of(order) + (4, N)) for N in SHAPE]
                    jets.check_weight(kind, order, sample, weight)               # it is a weight of the kind
                    self.assertIsNone(validation)
                    for a, b in zip(per_mode(kind, y_taylor), per_mode(kind, jets.derivatives_to_taylor(kind, order, y))):
                        self.assertTrue(np.array_equal(a, b))
        # no jets: nothing to convert, and the weight is ones, of the type of z
        y, weight, _ = jets.prepare_derivative_data('probe', (), random_jet(rng, 'probe', (), (4,)))
        self.assertEqual((weight.shape, weight.ravel().tolist()), ((D, 1, 1), [1.0] * D))
        y, weight, _ = jets.prepare_derivative_data('apply', (), random_jet(rng, 'apply', (), (4,)))
        self.assertEqual((weight.shape, float(weight[0])), ((1,), 1.0))

    def test_a_given_weight_and_validation_data_are_converted(self):
        rng = np.random.default_rng(11)
        for kind, order in (('probe', (3,)), ('apply', (3,)), ('probe', (1, 2)), ('apply', (1, 2)), ('entries', ())):
            y, validation = random_jet(rng, kind, order, (4,)), random_jet(rng, kind, order, (2,))
            weight = random_weight(rng, kind, order, (4,))
            y_taylor, weight_taylor, validation_taylor = jets.prepare_derivative_data(kind, order, y, weight, validation)
            with self.subTest(kind=kind, order=order):
                for a, b in zip(mode_weights(kind, weight_taylor),
                                mode_weights(kind, jets.derivative_weight_to_taylor_weight(kind, order, weight))):
                    self.assertTrue(np.array_equal(a, b))
                for a, b in zip(per_mode(kind, validation_taylor),
                                per_mode(kind, jets.derivatives_to_taylor(kind, order, validation))):
                    self.assertTrue(np.array_equal(a, b))
                for a, b in zip(per_mode(kind, y_taylor), per_mode(kind, jets.derivatives_to_taylor(kind, order, y))):
                    self.assertTrue(np.array_equal(a, b))

    def test_the_dtype_is_kept_and_an_unknown_kind_raises(self):
        y32 = np.ones((4, 3), dtype=np.float32)
        y, weight, validation = jets.prepare_derivative_data('apply', (3,), y32, validation=y32)
        self.assertEqual((y.dtype, weight.dtype, validation.dtype), (np.float32,) * 3)
        self.assertEqual(jets.taylor_to_derivatives('apply', (3,), y32).dtype, np.float32)
        self.assertEqual(jets.derivative_weight_to_taylor_weight('apply', (3,), np.ones((4, 1), dtype=np.float32)).dtype,
                         np.float32)
        self.assertEqual(jets.taylor_to_derivatives('apply', (1, 2), np.ones((2, 3), dtype=np.float32)).dtype, np.float32)
        with self.assertRaisesRegex(ValueError, 'unknown sampling kind'):
            jets.prepare_derivative_data('probe_derivatives', (3,), y32)


class TestCheckers(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(12)
        self.rng = rng
        self.W = (7,)
        self.ww = [rng.standard_normal(self.W + (N,)) for N in SHAPE]
        self.pp = [rng.standard_normal(self.W + (N,)) for N in SHAPE]
        self.index = np.stack([rng.integers(0, N, size=self.W) for N in SHAPE])

    def good_sample(self, kind, order, W=None, sizes=None):
        """Input jets of the given jet sizes per mode (default: the box), or an index."""
        W = self.W if W is None else W
        rng = self.rng
        if kind == 'entries':
            return np.stack([rng.integers(0, N, size=W) for N in SHAPE]) if W else np.array([1, 2, 3])
        sizes = [F_of(order)] * D if sizes is None else sizes
        return [rng.standard_normal(tuple(s) + W + (N,)) for s, N in zip(sizes, SHAPE)]

    def good_data(self, kind, order, W=None):
        W = self.W if W is None else W
        if kind == 'probe':
            return [np.zeros(F_of(order) + W + (N,)) for N in SHAPE]
        return np.zeros(F_of(order) + W)

    def assert_raises(self, pattern, function, *args):
        with self.assertRaisesRegex(ValueError, pattern):
            function(*args)

    # ------------------------------------------------------------------------------------------ check_sample
    def test_check_sample_passes_every_good_sample(self):
        for kind in ('probe', 'apply', 'entries'):
            for order in (JET_SHAPES if kind != 'entries' else ((),)):
                for W in ((7,), (), (2, 3)):
                    with self.subTest(kind=kind, order=order, W=W):
                        self.assertIsNone(jets.check_sample(kind, order, self.good_sample(kind, order, W), SHAPE))
        mixed = [(3, 2), (1, 1), (2, 3)]                                    # input jet sizes of their own, per mode
        self.assertIsNone(jets.check_sample('probe', (1, 2), self.good_sample('probe', (1, 2), sizes=mixed), SHAPE))
        self.assertIsNone(jets.check_sample('apply', (2,), jets.affine_jet(self.ww, self.pp), SHAPE))
        self.assertIsNone(jets.check_sample('apply', (), tuple(self.ww), list(SHAPE)))      # a tuple, and a list of sizes
        self.assertIsNone(jets.check_sample('entries', (), self.index.astype(np.int32), SHAPE))
        self.assertIsNone(jets.check_sample('apply', (np.int64(2),), jets.affine_jet(self.ww, self.pp), SHAPE))

    def test_check_sample_raises_on_each_wrong_form_of_the_input_jets(self):
        ww = self.ww
        line = list(jets.affine_jet(self.ww, self.pp))
        for kind in ('probe', 'apply'):
            with self.subTest(kind=kind):
                self.assert_raises('one array per mode: expected 3, got 2', jets.check_sample, kind, (), ww[:2], SHAPE)
                self.assert_raises('one array per mode: expected 3, got 4', jets.check_sample, kind, (), ww + ww[:1], SHAPE)
                wrong_size = ww[:2] + [np.zeros((7, 9))]
                self.assert_raises(r'ww\[2\] has the mode size 9; expected N_2 = 6', jets.check_sample, kind, (), wrong_size, SHAPE)
                wrong_stack = [ww[0], np.zeros((8, 5)), ww[2]]
                self.assert_raises(r'ww\[1\] has the sample stack W = \(8,\)', jets.check_sample, kind, (), wrong_stack, SHAPE)
                packed = np.zeros((3, 7, 6))                                    # the uniform layout's packed form
                self.assert_raises('ww must be a tuple or a list', jets.check_sample, kind, (), packed, SHAPE)
                self.assert_raises(r'ww\[0\] has shape \(\): too few axes', jets.check_sample, kind, (), [1.0, ww[1], ww[2]], SHAPE)
                self.assert_raises(r'ww\[0\] must be one array', jets.check_sample, kind, (), [ww[:1], ww[1], ww[2]], SHAPE)
                # too few axes for the jet shape
                self.assert_raises(r'ww\[0\] has shape \(7, 4\): too few axes for the jet shape, which has 2',
                                   jets.check_sample, kind, (1, 2), ww, SHAPE)
                # modes whose W disagree once the jet axes are read: different jet sizes, read with too few jet axes
                mixed = self.good_sample(kind, (2,), sizes=[(3,), (2,), (3,)])
                self.assert_raises(r'ww\[1\] has the sample stack W = \(2, 7\) once its 0 jet axes are read; ww\[0\] has '
                                   r'W = \(3, 7\)', jets.check_sample, kind, (), mixed, SHAPE)
                # a jet axis of size 0
                self.assert_raises(r'ww\[1\] has the jet sizes \(0,\)', jets.check_sample, kind, (2,),
                                   [line[0], line[1][:0], line[2]], SHAPE)
                # the old pair (ww, pp) is not a sample
                self.assert_raises('one array per mode: expected 3, got 2', jets.check_sample, kind, (1,),
                                   (self.ww, self.pp), SHAPE)

    def test_the_trap_of_a_forgotten_jet_shape(self):
        """What the docstrings state: a shape cannot say how many leading axes are jet axes. Input jets that all have
        the same jet sizes, given with no jet shape, pass, their jet axis read as one more sample axis; read with two
        jet axes, a line's second axis is read as a jet axis too, and they pass."""
        line = jets.affine_jet(self.ww, self.pp)                        # (2, 7, N_i)
        self.assertIsNone(jets.check_sample('probe', (), line, SHAPE))
        self.assertIsNone(jets.check_sample('probe', (1, 2), line, SHAPE))
        for checker in (jets.check_sample, jets.check_data, jets.check_weight):
            self.assertIn('forgotten jet shape', checker.__doc__)
        y = np.zeros((2, 7))                                            # the jets of the line at (1,), for apply
        self.assertIsNone(jets.check_data('apply', (1,), line, y))
        self.assertIsNone(jets.check_data('apply', (), line, y))        # and with no jet shape: W read as (2, 7)

    def test_check_sample_raises_on_each_wrong_form_of_the_index(self):
        index = self.index
        self.assert_raises(r'index must have shape \(d,\)\+W with d=3', jets.check_sample, 'entries', (), index[:2], SHAPE)
        self.assert_raises(r'index must have shape \(d,\)\+W', jets.check_sample, 'entries', (), np.asarray(3), SHAPE)
        self.assert_raises('index must be an integer array', jets.check_sample, 'entries', (), index.astype(float), SHAPE)
        self.assert_raises('the entries kind has no jets', jets.check_sample, 'entries', (2,), index, SHAPE)
        self.assert_raises('index must be one integer array', jets.check_sample, 'entries', (), (index, self.pp), SHAPE)
        self.assert_raises('index must be one integer array', jets.check_sample, 'entries', (), index.tolist(), SHAPE)

    def test_the_checkers_raise_on_an_unknown_kind_and_a_bad_jet_shape(self):
        for function, args in ((jets.check_sample, (self.ww, SHAPE)), (jets.check_data, (self.ww, np.zeros(7))),
                               (jets.check_weight, (None, D))):
            with self.subTest(function=function.__name__):
                self.assert_raises('unknown sampling kind', function, 'apply_derivatives', (), *args)
                self.assert_raises('every entry of the jet shape is >= 0', function, 'apply', (-1,), *args)
                for bad in (0, 2, [1], (1.0,), (True,)):
                    with self.assertRaises(TypeError):
                        function('apply', bad, *args)

    # ------------------------------------------------------------------------------------------ check_data
    def test_check_data_passes_every_good_y(self):
        for kind in ('probe', 'apply', 'entries'):
            for order in (JET_SHAPES if kind != 'entries' else ((),)):
                for W in ((7,), (), (2, 3)):
                    with self.subTest(kind=kind, order=order, W=W):
                        sample = self.good_sample(kind, order, W)
                        self.assertIsNone(jets.check_data(kind, order, sample, self.good_data(kind, order, W)))
        self.assertIsNone(jets.check_data('probe', (), self.ww, tuple(self.good_data('probe', ()))))
        # the data are on the full box whatever the input jet sizes
        mixed = self.good_sample('probe', (1, 2), sizes=[(3, 2), (1, 1), (2, 3)])
        self.assertIsNone(jets.check_data('probe', (1, 2), mixed, self.good_data('probe', (1, 2))))

    def test_check_data_raises_on_each_wrong_y(self):
        """The data-space rule (data-metric arc): exactly ``d`` arrays for probe (a tuple, a list, or a mode-leading
        array), each of ``z``'s rank, each axis 1 or ``z``'s size; ``C`` only with ``stack_shape``."""
        line = jets.affine_jet(self.ww, self.pp)                        # F_in = (2,)
        order = (2,)
        good = self.good_data('probe', order)
        self.assert_raises('exactly d = 3 arrays.*got 2 arrays', jets.check_data, 'probe', order, line, good[:2])
        self.assert_raises(r'exactly d = 3 arrays.*got shape \(1, 3, 7, 4\)', jets.check_data, 'probe', order, line,
                           np.zeros((1, 3, 7, 4)))                                       # a mode axis of 1
        self.assert_raises(r'\(mode 0\) has shape \(3, 7, 6\).*\(3, 7, 4\)', jets.check_data, 'probe', order, line,
                           np.zeros((3, 3, 7, 6)))                                       # a padded edge, mode-leading
        wrong_size = good[:2] + [np.zeros((3, 7, 9))]
        self.assert_raises(r'\(mode 2\) has shape \(3, 7, 9\); .* F\+W\+C\+\(N_2,\) = \(3, 7, 6\)',
                           jets.check_data, 'probe', order, line, wrong_size)
        self.assert_raises(r'\(mode 0\) has shape \(7, 4\); .* = \(3, 7, 4\)',
                           jets.check_data, 'probe', order, line, self.good_data('probe', ()))      # no jet axis
        self.assert_raises(r'\(mode 0\) has shape \(2, 7, 4\); .* = \(3, 7, 4\)',                    # the input's
                           jets.check_data, 'probe', order, line, [np.zeros((2, 7, N)) for N in SHAPE])  # sizes, not F
        self.assert_raises(r'\(mode 0\) has shape \(3, 7, 4\); .* = \(7, 4\)',
                           jets.check_data, 'probe', (), self.ww, good)                  # a jet axis with no jets
        wrong_stack = [np.zeros((3, 8, N)) for N in SHAPE]
        self.assert_raises(r'\(mode 0\) has shape \(3, 8, 4\)', jets.check_data, 'probe', order, line, wrong_stack)
        with_C = [np.zeros((3, 7, 2, N)) for N in SHAPE]                            # a stack of base points
        self.assert_raises(r'\(mode 0\) has shape \(3, 7, 2, 4\)', jets.check_data, 'probe', order, line, with_C)
        self.assertIsNone(jets.check_data('probe', order, line, with_C, stack_shape=(2,)))
        self.assert_raises(r'\(mode 0\) has shape \(3, 7, 2, 4\)', jets.check_data, 'probe', order, line, with_C, (3,))
        plane = jets.affine_jet(self.ww, self.pp, self.pp)
        self.assert_raises(r'has shape \(2, 7\); .* F\+W\+C = \(2, 3, 7\)',
                           jets.check_data, 'apply', (1, 2), plane, np.zeros((2, 7)))            # not of the full box
        self.assert_raises(r'has shape \(7,\); .* = \(3, 7\)', jets.check_data, 'apply', order, line, np.zeros(7))
        self.assert_raises(r'has shape \(3, 7\); .* = \(7,\)', jets.check_data, 'apply', (), self.ww, np.zeros((3, 7)))
        self.assert_raises(r'has shape \(3, 8\)', jets.check_data, 'apply', order, line, np.zeros((3, 8)))
        self.assert_raises('is one array, of the type of z; got a list of length 3', jets.check_data, 'apply', order,
                           line, good)
        self.assert_raises(r'has shape \(3, 7\); .* W\+C = \(7,\)', jets.check_data, 'entries', (), self.index,
                           np.zeros((3, 7)))
        self.assert_raises('the entries kind has no jets', jets.check_data, 'entries', order, self.index, np.zeros(7))

    def test_an_object_larger_than_z_where_z_has_a_size_of_one_raises(self):
        """Where ``z``'s axis is 1 (a one-sample batch; the jet axis at ``(0,)``) an object with 3 there raises: a size of
        1 is the object's to broadcast, never ``z``'s."""
        rng = self.rng
        one = [rng.standard_normal((1, N)) for N in SHAPE]                           # W = (1,)
        clipped = [rng.standard_normal((1,) + self.W + (N,)) for N in SHAPE]         # order (0,): F = (1,)
        for label, order, sample, shape in (('W = (1,)', (), one, (3,)), ('order (0,)', (0,), clipped, (3,) + self.W)):
            for kind in ('probe', 'apply'):
                z = [np.zeros(shape + (N,)) for N in SHAPE] if kind == 'probe' else np.zeros(shape)
                good = mapped(kind, lambda a: np.zeros(tuple(1 if n == 3 else n for n in a.shape)), z)
                with self.subTest(case=label, kind=kind):
                    self.assertIsNone(jets.check_data(kind, order, sample, good))
                    self.assertIsNone(jets.check_weight(kind, order, sample, good))
                    self.assert_raises('has shape', jets.check_data, kind, order, sample, z)
                    self.assert_raises('has shape', jets.check_weight, kind, order, sample, z)

    def test_check_data_passes_size_1_axes_and_a_mode_leading_array(self):
        """A size of 1 is the same datum along the axis: data constant over the samples, shared by a stack, per slot;
        a probe mode-leading array whose edge is 1 (or every ``N_i``)."""
        line = jets.affine_jet(self.ww, self.pp)
        order = (2,)
        for label, kind, y, C in (('apply, constant in W', 'apply', np.zeros((3, 1)), ()),
                                  ('apply, per slot, shared by C', 'apply', np.zeros((3, 1, 1)), (2,)),
                                  ('apply, per element of C', 'apply', np.zeros((3, 7, 2)), (2,)),
                                  ('probe, constant over the edge', 'probe', [np.zeros((3, 7, 1))] * D, ()),
                                  ('probe, mode-leading, edge 1', 'probe', np.zeros((D, 3, 1, 1)), ()),
                                  ('entries, one number', 'entries', np.zeros((1,)), ())):
            with self.subTest(case=label):
                sample = self.index if kind == 'entries' else line
                self.assertIsNone(jets.check_data(kind, () if kind == 'entries' else order, sample, y, C))

    # ------------------------------------------------------------------------------------------ check_weight
    def test_check_weight_passes_the_full_form_and_none(self):
        """Every form of the type of z: ``z``'s shape, every axis of 1, the blocks placed by ``blocks_to_weight``, a
        mode-leading probe array, and a stack ``C``; and ``None``."""
        rng = self.rng
        for kind in ('probe', 'apply', 'entries'):
            for order in (JET_SHAPES if kind != 'entries' else ((),)):
                for C in ((), (2,)):
                    sample = self.good_sample(kind, order)
                    z = [np.zeros(F_of(order) + self.W + C + (N,)) for N in SHAPE] if kind == 'probe' else \
                        np.zeros((F_of(order) if kind == 'apply' else ()) + self.W + C)
                    blocks = 0.5 + rng.random(((D,) if kind == 'probe' else ()) + (F_of(order) if kind != 'entries'
                                                                                   else ()))
                    forms = {'z\'s shape': mapped(kind, np.ones_like, z),
                             'every axis 1': mapped(kind, lambda a: np.ones((1,) * a.ndim), z),
                             'from its blocks': jets.blocks_to_weight(kind, order, blocks, z)}
                    if kind == 'probe':
                        forms['mode-leading, per sample'] = np.ones((D,) + (1,) * len(order) + self.W + C + (1,))
                    for label, weight in forms.items():
                        with self.subTest(kind=kind, order=order, C=C, form=label):
                            self.assertIsNone(jets.check_weight(kind, order, sample, weight, C))
                    self.assertIsNone(jets.check_weight(kind, order, sample, None, C))

    def test_check_weight_raises_on_every_other_shape(self):
        """The old table (``d = 3``, ``W = (7,)``, ``n = 3``, ``C = ()``), re-ruled by the data-space rule: two cases
        become legal (a full form with a ``W`` axis of 1), the other 17 raise, by the mode count or by shape (rank,
        or a size that is neither 1 nor ``z``'s)."""
        n = 3
        cases = [
            ('probe', (n,), np.ones(n + 1), 'a bare per-slot vector', 'count'),
            ('probe', (n,), np.ones((1, n + 1)), 'a mode dimension of 1', 'count'),
            ('probe', (n,), np.ones((D, 1)), 'a jet dimension of 1', 'shape'),
            ('probe', (n,), np.ones(D), 'a bare per-mode vector', 'shape'),
            ('probe', (n,), np.ones((D + 1, n + 1)), 'more rows than modes', 'count'),
            ('probe', (n,), np.ones((n + 1, D)), 'the two axes exchanged', 'count'),
            ('probe', (1, 2), np.ones((D, 3)), 'a weight of (d, n + 1) given for m = 2', 'shape'),
            ('probe', (1, 2), np.ones((D, 3, 2)), 'the jet axes exchanged', 'shape'),
            ('probe', (), np.ones((D, 1)), 'no jets, with a jet axis', 'shape'),
            ('probe', (), np.ones(1), 'no jets, a mode dimension of 1', 'count'),
            ('probe', (), np.ones(()), 'no jets, a number', 'count'),
            ('apply', (n,), np.ones((1, n + 1)), 'a mode dimension of 1', 'shape'),
            ('apply', (n,), np.ones((D, n + 1)), 'a mode axis', 'shape'),
            ('apply', (n,), np.ones(1), 'a jet dimension of 1', 'shape'),
            ('apply', (n,), np.ones(()), 'a number with jets', 'shape'),
            ('apply', (1, 2), np.ones(3), 'one jet axis for two', 'shape'),
            ('apply', (), np.ones(1), 'a vector with no jets: W of 1, legal', None),
            ('entries', (), np.ones(1), 'a vector: W of 1, legal', None),
            ('entries', (), np.ones((1, n + 1)), 'a mode dimension of 1', 'shape'),
        ]
        self.assertEqual(sum(reason is not None for *_, reason in cases), 17)
        for kind, order, weight, what, reason in cases:
            with self.subTest(kind=kind, order=order, what=what):
                sample = self.good_sample(kind, order)
                if reason is None:
                    self.assertIsNone(jets.check_weight(kind, order, sample, weight))
                    continue
                read = weight.shape[1:] if kind == 'probe' else weight.shape      # probe: mode 0's slice
                shape = str(read).replace('(', r'\(').replace(')', r'\)')
                pattern = ('exactly d = 3 arrays' if reason == 'count' else
                           r'the weight of the %s kind(?: \(mode 0\))? has shape %s; it is of the type of z' % (kind, shape))
                self.assert_raises(pattern, jets.check_weight, kind, order, sample, weight)
        self.assert_raises('the entries kind has no jets', jets.check_weight, 'entries', (n,), self.index, np.ones(n + 1))

    def test_check_weight_cannot_see_a_transposed_square_weight(self):
        """The limit the docstring states: a weight whose axes are given in another order passes when their sizes
        coincide (apply at ``(2,)`` with ``W = (3,)``: ``F + W = (3, 3)``)."""
        sample = [self.rng.standard_normal((3, 3, N)) for N in SHAPE]
        weight = np.arange(1.0, 10.0).reshape(3, 3)
        self.assertIsNone(jets.check_weight('apply', (2,), sample, weight))
        self.assertIsNone(jets.check_weight('apply', (2,), sample, weight.T))
        self.assertIn('another order', jets.check_weight.__doc__)

    # ------------------------------------------------------------------------------------------ the tier
    def test_no_backend_module_calls_a_checker(self):
        """Checkers, not guarantees: outside ``jets`` itself, no backend module refers to one of the checkers, except
        ``LeastSquaresOracle.check_batch`` in ``fitting.py``, itself a checker built from them (the plan's 2.2, O-29;
        P14); and no backend module calls ``check_batch``."""
        names = {'check_sample', 'check_data', 'check_weight', 'check_coordinate_scales'}
        backend = pathlib.Path(t3toolbox.__file__).resolve().parent / 'backend'
        files = sorted(backend.glob('*.py'))
        self.assertGreater(len(files), 30)
        for path in files:
            tree = ast.parse(path.read_text(), filename=str(path))
            used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} \
                 | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} \
                 | {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
            with self.subTest(module=path.name):
                if path.name in ('jets.py', 'data_preparation.py'):          # where they are defined
                    calls = {node.func.id for node in ast.walk(tree)
                             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
                    self.assertEqual(calls & names, set())                   # not even their own modules call them
                elif path.name == 'fitting.py':                              # only inside check_batch
                    batch_checker = [node for node in ast.walk(tree)
                                     if isinstance(node, ast.FunctionDef) and node.name == 'check_batch']
                    inner = {id(n) for f in batch_checker for n in ast.walk(f)}
                    inside = {n.attr for n in ast.walk(tree) if id(n) in inner and isinstance(n, ast.Attribute)}
                    outside = {n.attr for n in ast.walk(tree) if id(n) not in inner and isinstance(n, ast.Attribute)} \
                        | {n.id for n in ast.walk(tree) if id(n) not in inner and isinstance(n, ast.Name)}
                    self.assertEqual(outside & names, set())
                    self.assertEqual(inside & names, {'check_sample', 'check_data', 'check_weight'})
                else:
                    self.assertEqual(used & names, set())
                # and nothing in the backend calls check_batch: a checker, called by the frontend only (round 1)
                calls = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', None)
                         for node in ast.walk(tree) if isinstance(node, ast.Call)}
                self.assertNotIn('check_batch', calls)

    def test_the_validators_of_pp_are_gone(self):
        """There is no ``pp``: the two validators P3 copied from ``sampling_derivatives`` are not in ``jets`` (the
        originals stay in ``sampling_derivatives``, called by the old frontend methods, until P20)."""
        for name in ('check_perturbation_vectors', 'check_perturbation_index'):
            self.assertNotIn(name, jets.__all__)
            self.assertFalse(hasattr(jets, name), name)

    def test_every_public_function_has_a_doctest(self):
        for name in jets.__all__:
            with self.subTest(name=name):
                self.assertIn('>>> ', getattr(jets, name).__doc__)
        self.assertIn('>>> ', jets.__doc__)


class TestFrontendReach(unittest.TestCase):

    def test_the_names_of_t3toolbox_fitting(self):
        """The preparation functions and the jet helpers are reachable from ``t3toolbox.fitting`` under their
        backend names. Twelve are the backend functions themselves. One is the frontend's own function, with the
        backend's name and parameters, which makes a check and calls the backend one: ``rescale_coordinates`` (the
        form of the scales, always). The checkers are not re-exported, and neither are the dual-jet converters: the
        engine's turned form is the backend's (M1, 9.2), so no frontend name takes or returns it."""
        import t3toolbox.backend.data_preparation as data_preparation
        from_jets = ('derivatives_to_taylor', 'taylor_to_derivatives', 'derivative_weight_to_taylor_weight',
                     'prepare_derivative_data', 'data_metric', 'blocks_to_weight', 'affine_jet', 'index_to_one_hot')
        from_preparation = ('data_block_norms', 'equalizing_weight', 'rescale_direction', 't3_unscale_coordinates')
        for module, names in ((jets, from_jets), (data_preparation, from_preparation)):
            for name in names:
                with self.subTest(name=name):
                    self.assertIs(getattr(fitting, name), getattr(module, name))
                    self.assertIn(name, fitting.__all__)
        for module, name in ((data_preparation, 'rescale_coordinates'),):
            with self.subTest(own=name):
                own, backend = getattr(fitting, name), getattr(module, name)
                self.assertIsNot(own, backend)
                self.assertEqual(own.__module__, 't3toolbox.fitting')
                self.assertEqual(list(inspect.signature(own).parameters), list(inspect.signature(backend).parameters))
                self.assertIn(name, fitting.__all__)
        for name in ('jet_to_dual_jet', 'dual_jet_to_jet',
                     'check_sample', 'check_data', 'check_weight', 'check_coordinate_scales',
                     'check_perturbation_vectors', 'check_perturbation_index', 'ut3_unscale_coordinates'):
            self.assertNotIn(name, fitting.__all__)
            self.assertFalse(hasattr(fitting, name), name)
        self.assertEqual(len(fitting.__all__), len(set(fitting.__all__)))

    def test_the_input_jet_helpers_are_reachable_from_t3toolbox(self):
        """O-37: ``affine_jet`` and ``index_to_one_hot`` from the top level too, the backend functions themselves."""
        for name in ('affine_jet', 'index_to_one_hot'):
            with self.subTest(name=name):
                self.assertIs(getattr(t3toolbox, name), getattr(jets, name))
                self.assertIn(name, t3toolbox.__all__)
        self.assertEqual(len(t3toolbox.__all__), len(set(t3toolbox.__all__)))


if __name__ == '__main__':
    unittest.main()
