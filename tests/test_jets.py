# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.jets``: the dual jet, derivative data and the checkers; and their reach from ``t3toolbox.fitting``.

1. **The dual jet** (`TestDualJet`): ``jet_to_dual_jet`` against its definition written out slot by slot; the
   round trip through ``dual_jet_to_jet``; the pairing identity, with the top coefficient of the product taken
   by ``numpy.polynomial`` (a route the function does not take); the backend inverse at a zero weight (it
   divides, and raises and warns nothing).
2. **The frontend ``dual_jet_to_jet``** (`TestFrontendDualJetToJet`): raises in safe mode where a weight is
   zero, divides in ``safety.unsafe()`` and under ``jax.jit``, and gives the backend's numbers wherever both run.
3. **Derivative data** (`TestDerivativeData`): the conversions against derivatives taken by ``numpy.polynomial``
   from an interpolating polynomial of the dense tensor's outputs; the weight conversion and
   ``prepare_derivative_data`` reproducing the objective written on derivatives; the weight ``t!`` when none is
   given, for apply and for probe with its mode axis.
4. **The checkers** (`TestCheckers`): each passes every good input and raises on each wrong form; no backend
   module calls one; the two validators of ``pp`` are the ones of ``sampling_derivatives``, copied.
5. **The reach from the frontend** (`TestFrontendReach`): the names of ``t3toolbox.fitting``.

Numerical comparisons are to a tolerance, structural ones exact. numpy only, by the house convention, except the
jit cases of the frontend ``dual_jet_to_jet``; the functions under jit are in ``test_dispatch.py``.
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
import t3toolbox.safety as safety
import t3toolbox.backend.apply as bapply
import t3toolbox.backend.jets as jets
import t3toolbox.backend.probing as probing
import t3toolbox.backend.sampling_derivatives as sampling_derivatives

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

TOL = 1e-13
SHAPE = (4, 5, 6)           # the mode sizes N_i
D = len(SHAPE)


def relerr(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b))


def T_of(order):
    return (order + 1,) if order else ()


def random_jet(rng, kind, order, W, extra=()):
    """An object of the type of z: probe len=d, elm_shape=T+W+extra+(Ni,); apply and entries T+W+extra."""
    if kind == 'probe':
        return [rng.standard_normal(T_of(order) + W + extra + (N,)) for N in SHAPE]
    return rng.standard_normal(T_of(order) + W + extra)


def random_weight(rng, kind, order):
    return 0.5 + rng.random(((D,) if kind == 'probe' else ()) + T_of(order))


def per_mode(kind, value):
    """The arrays of an object of the type of z, as a list (one per mode for probe, one otherwise)."""
    return list(value) if kind == 'probe' else [value]


def mode_weights(kind, weight):
    """The weight of each array of an object of the type of z: a row per mode for probe, the weight otherwise."""
    return [weight[i] for i in range(D)] if kind == 'probe' else [weight]


CASES = [(kind, order, W, extra)
         for kind in ('probe', 'apply', 'entries')
         for order in (0, 1, 3)
         for W, extra in (((7,), ()), ((), ()), ((2, 3), ()), ((7,), (2, 3)))]     # extra: the stacks K + C


class TestDualJet(unittest.TestCase):

    def test_the_dual_jet_is_the_weights_then_the_turn(self):
        """``dual_jet[t] = weight[order - t]**2 * jet[order - t]``, slot by slot; at order 0, ``weight**2 * jet``."""
        rng = np.random.default_rng(0)
        for kind, order, W, extra in CASES:
            jet, weight = random_jet(rng, kind, order, W, extra), random_weight(rng, kind, order)
            dual = jets.jet_to_dual_jet(kind, order, jet, weight)
            with self.subTest(kind=kind, order=order, W=W, extra=extra):
                self.assertIsInstance(dual, tuple if kind == 'probe' else (np.ndarray, np.generic))   # W = (): a numpy scalar
                for jet_TX, dual_TX, omega_T in zip(per_mode(kind, jet), per_mode(kind, dual), mode_weights(kind, weight)):
                    self.assertEqual(dual_TX.shape, jet_TX.shape)
                    if order == 0:
                        self.assertLessEqual(relerr(dual_TX, float(omega_T) ** 2 * jet_TX), TOL)
                    for t in range(order + 1 if order else 0):
                        self.assertLessEqual(relerr(dual_TX[t], omega_T[order - t] ** 2 * jet_TX[order - t]), TOL)

    def test_no_weight_is_ones(self):
        rng = np.random.default_rng(1)
        for kind, order, W, extra in CASES:
            jet = random_jet(rng, kind, order, W, extra)
            ones = np.ones(((D,) if kind == 'probe' else ()) + T_of(order))
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
            jet, weight = random_jet(rng, kind, order, W, extra), random_weight(rng, kind, order)
            there_and_back = jets.dual_jet_to_jet(kind, order, jets.jet_to_dual_jet(kind, order, jet, weight), weight)
            back_and_there = jets.jet_to_dual_jet(kind, order, jets.dual_jet_to_jet(kind, order, jet, weight), weight)
            with self.subTest(kind=kind, order=order, W=W, extra=extra):
                for x_TX, a_TX, b_TX in zip(per_mode(kind, jet), per_mode(kind, there_and_back),
                                            per_mode(kind, back_and_there)):
                    self.assertLessEqual(relerr(a_TX, x_TX), TOL)
                    self.assertLessEqual(relerr(b_TX, x_TX), TOL)

    def test_the_pairing_identity(self):
        """``sum_t weight[t]**2 a[t] b[t]`` is the coefficient of ``s^order`` of ``jet_to_dual_jet(a) * b`` as
        polynomials in ``s``. The product is taken by ``numpy.polynomial``, one entry at a time."""
        rng = np.random.default_rng(3)
        for kind in ('probe', 'apply'):
            for order in (1, 2, 5):
                a, b = random_jet(rng, kind, order, (3,)), random_jet(rng, kind, order, (3,))
                weight = random_weight(rng, kind, order)
                dual = jets.jet_to_dual_jet(kind, order, a, weight)
                with self.subTest(kind=kind, order=order):
                    for dual_TX, a_TX, b_TX, omega_T in zip(per_mode(kind, dual), per_mode(kind, a), per_mode(kind, b),
                                                            mode_weights(kind, weight)):
                        flat = lambda x_TX: x_TX.reshape(order + 1, -1)
                        top = np.array([np.polynomial.polynomial.polymul(p, q)[order]
                                        for p, q in zip(flat(dual_TX).T, flat(b_TX).T)])
                        weighted = np.einsum('t,tx,tx->x', omega_T ** 2, flat(a_TX), flat(b_TX))
                        self.assertLessEqual(relerr(top, weighted), TOL)
        # order 0: the pairing is the plain product
        a, b, weight = rng.standard_normal(3), rng.standard_normal(3), np.asarray(1.5)
        self.assertLessEqual(relerr(jets.jet_to_dual_jet('apply', 0, a, weight) * b, 1.5 ** 2 * a * b), TOL)

    def test_the_weight_enters_squared_and_the_turn_matters(self):
        """Two wrong readings of the definition are told apart from the right one: a plain weight, and no turn."""
        jet, weight = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 5.0])
        dual = jets.jet_to_dual_jet('apply', 2, jet, weight)
        self.assertEqual(dual.tolist(), [75.0, 18.0, 4.0])          # (4*1, 9*2, 25*3), turned

    def test_the_backend_inverse_divides_by_a_zero_weight_and_raises_nothing(self):
        rng = np.random.default_rng(4)
        for kind in ('probe', 'apply'):
            order = 2
            jet, weight = random_jet(rng, kind, order, (5,)), random_weight(rng, kind, order)
            weight[..., 1] = 0.0                                    # order 1, in every mode
            with self.subTest(kind=kind):
                with warnings.catch_warnings():
                    warnings.simplefilter('error')                  # a warning would be an error here
                    back = jets.dual_jet_to_jet(kind, order, jet, weight)
                    all_zero = jets.dual_jet_to_jet(kind, order, jet, np.zeros_like(weight))
                nonzero = np.where(weight == 0.0, 1.0, weight)
                reference = jets.dual_jet_to_jet(kind, order, jet, nonzero)
                for back_TX, zero_TX, reference_TX in zip(per_mode(kind, back), per_mode(kind, all_zero),
                                                          per_mode(kind, reference)):
                    self.assertFalse(np.isfinite(back_TX[1]).any())             # the slot of the zero weight
                    self.assertTrue(np.array_equal(back_TX[[0, 2]], reference_TX[[0, 2]]))   # the others untouched
                    self.assertFalse(np.isfinite(zero_TX).any())
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertTrue(np.isnan(jets.dual_jet_to_jet('apply', 0, np.zeros(3), np.zeros(()))).all())    # 0 / 0, order 0

    def test_a_backend_function_given_a_bad_weight_checks_nothing(self):
        """Checkers, not guarantees: a weight ``check_weight`` rejects is used as it broadcasts, with no error."""
        jet = np.arange(12.0).reshape(3, 4)
        bad = np.array([[1.0], [2.0], [3.0]])                       # shape (3, 1), where T = (3,) is expected
        with self.assertRaises(ValueError):
            jets.check_weight('apply', 2, bad, D)
        self.assertTrue(np.array_equal(jets.jet_to_dual_jet('apply', 2, jet, bad), jets.jet_to_dual_jet('apply', 2, jet, bad[:, 0])))

    def test_kind_decides_the_type_and_a_packed_probe_array_is_taken_mode_by_mode(self):
        """``kind`` says which type the object is of; the object is not looked at. With ``kind='probe'`` the
        functions go over the modes along the first axis, so a list, a tuple and an array packed for the uniform
        layout, ``(d,) + T + W + (N,)``, give the same result, returned as a tuple, one array per mode. The values
        are checked against the definition written out on the packed array."""
        rng = np.random.default_rng(13)
        d, order, W, N = 3, 3, (5,), 6
        packed = rng.standard_normal((d, order + 1) + W + (N,))
        weight = 0.5 + rng.random((d, order + 1))
        ragged = [packed[i] for i in range(d)]
        factorials = np.array([math.factorial(t) for t in range(order + 1)], dtype=float)
        w2 = weight[:, :, None, None] ** 2
        wanted = {'jet_to_dual_jet': (w2 * packed)[:, ::-1],                  # the weights, then the turn, per mode
                  'dual_jet_to_jet': packed[:, ::-1] / w2,
                  'derivatives_to_taylor': packed / factorials[None, :, None, None],
                  'taylor_to_derivatives': packed * factorials[None, :, None, None]}
        calls = {'jet_to_dual_jet': lambda x: jets.jet_to_dual_jet('probe', order, x, weight),
                 'dual_jet_to_jet': lambda x: jets.dual_jet_to_jet('probe', order, x, weight),
                 'derivatives_to_taylor': lambda x: jets.derivatives_to_taylor('probe', order, x),
                 'taylor_to_derivatives': lambda x: jets.taylor_to_derivatives('probe', order, x)}
        for name, call in calls.items():
            from_packed, from_tuple, from_list = call(packed), call(tuple(packed)), call(ragged)
            with self.subTest(function=name):
                for out in (from_packed, from_tuple, from_list):
                    self.assertEqual((type(out), len(out)), (tuple, d))            # a tuple, one array per mode
                for a, b, c in zip(from_packed, from_tuple, from_list):
                    self.assertEqual(a.shape, (order + 1,) + W + (N,))
                    self.assertTrue(np.array_equal(a, b) and np.array_equal(a, c))
                self.assertLessEqual(relerr(np.stack(from_packed), wanted[name]), TOL)
        self.assertLessEqual(relerr(np.stack(jets.jet_to_dual_jet('probe', order, packed, None)), packed[:, ::-1]), TOL)
        # one mode: the probe type is still a tuple, of one array (the type comes from kind, not from the count)
        one = rng.standard_normal((order + 1,) + W + (N,))
        for out in (jets.jet_to_dual_jet('probe', order, [one], weight[:1]), jets.dual_jet_to_jet('probe', order, (one,), None),
                    jets.derivatives_to_taylor('probe', order, one[None]), jets.taylor_to_derivatives('probe', order, [one])):
            self.assertEqual((type(out), len(out), out[0].shape), (tuple, 1, one.shape))
        self.assertEqual(type(jets.derivatives_to_taylor('apply', order, one)), np.ndarray)
        # prepare_derivative_data follows: a packed array with kind='probe' gives the tuple, and t! per mode
        y, taylor_weight, validation = jets.prepare_derivative_data('probe', order, packed, validation=packed)
        self.assertEqual((type(y), len(y), type(validation), taylor_weight.shape), (tuple, d, tuple, (d, order + 1)))
        self.assertLessEqual(relerr(np.stack(y), wanted['derivatives_to_taylor']), TOL)
        # the frontend function too
        for a, b in zip(fitting.dual_jet_to_jet('probe', order, packed, weight), calls['dual_jet_to_jet'](ragged)):
            self.assertTrue(np.array_equal(a, b))

    def test_an_object_of_the_other_type_than_kind_says_is_not_guarded_against(self):
        """What the docstrings say happens, pinned: a tuple given with ``kind='apply'`` is made into one array.
        Where the modes differ in size that fails; where they do not, the mode axis is read as the order axis: an
        error from broadcasting, or a wrong result with no error. And an unknown kind raises."""
        rng = np.random.default_rng(15)
        order = 2
        ragged = [rng.standard_normal((order + 1, 5, N)) for N in SHAPE]            # the probe type, modes of sizes 4, 5, 6
        same = [rng.standard_normal((order + 1, 5, 6)) for _ in range(4)]           # four modes of one size
        weight = 0.5 + rng.random(order + 1)
        for call in (lambda x: jets.jet_to_dual_jet('apply', order, x, weight),
                     lambda x: jets.dual_jet_to_jet('apply', order, x, None),
                     lambda x: jets.derivatives_to_taylor('apply', order, x),
                     lambda x: jets.taylor_to_derivatives('entries', order, x)):
            with self.assertRaises(ValueError):                                     # the modes cannot be stacked
                call(tuple(ragged))
        with self.assertRaises(ValueError):                                         # 4 modes against order + 1 = 3
            jets.derivatives_to_taylor('apply', order, tuple(same))
        with self.assertRaises(ValueError):
            jets.jet_to_dual_jet('apply', order, tuple(same), weight)
        wrong = jets.jet_to_dual_jet('apply', order, tuple(same), None)             # no error: the MODES are turned
        self.assertEqual(type(wrong), np.ndarray)
        self.assertTrue(np.array_equal(wrong, np.stack(same)[::-1]))
        for function, args in ((jets.jet_to_dual_jet, (same[0], weight)), (jets.dual_jet_to_jet, (same[0], weight)),
                               (jets.derivatives_to_taylor, (same[0],)), (jets.taylor_to_derivatives, (same[0],)),
                               (fitting.dual_jet_to_jet, (same[0], weight))):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, 'unknown sampling kind'):
                    function('probes', order, *args)

    def test_the_four_functions_take_kind_then_order(self):
        """The signatures: ``(kind, order, ...)``, as every other function of the two modules that takes a kind;
        ``weight`` is required (pass ``None`` for no weight)."""
        wanted = {'jet_to_dual_jet': ['kind', 'order', 'jet', 'weight'],
                  'dual_jet_to_jet': ['kind', 'order', 'dual_jet', 'weight'],
                  'derivatives_to_taylor': ['kind', 'order', 'jets'],
                  'taylor_to_derivatives': ['kind', 'order', 'jets'],
                  'derivative_weight_to_taylor_weight': ['weight', 'order']}        # unchanged: no kind
        for name, parameters in wanted.items():
            signature = inspect.signature(getattr(jets, name))
            with self.subTest(function=name):
                self.assertEqual(list(signature.parameters), parameters)
                self.assertTrue(all(p.default is inspect.Parameter.empty for p in signature.parameters.values()))

    def test_a_zero_dimensional_input_gives_an_array(self):
        """The apply kind at order 0 with no sample stack: numpy in gives a numpy array out, not a numpy scalar."""
        for out in (jets.jet_to_dual_jet('apply', 0, np.asarray(2.0), np.asarray(3.0)), jets.jet_to_dual_jet('apply', 0, 2.0, 3.0),
                    jets.dual_jet_to_jet('apply', 0, np.asarray(2.0), np.asarray(3.0)), jets.dual_jet_to_jet('apply', 0, np.asarray(2.0), None),
                    jets.derivatives_to_taylor('apply', 0, np.asarray(2.0)), jets.taylor_to_derivatives('apply', 0, np.asarray(2.0)),
                    jets.derivative_weight_to_taylor_weight(np.asarray(2.0), 0)):
            self.assertIs(type(out), np.ndarray)
            self.assertEqual(out.shape, ())
        self.assertEqual(float(jets.jet_to_dual_jet('apply', 0, np.asarray(2.0), np.asarray(3.0))), 18.0)

    def test_the_result_has_the_lower_precision_of_the_jet_and_the_weight(self):
        jet64, weight64 = np.ones((3, 4)), np.ones(3)
        for function in (jets.jet_to_dual_jet, jets.dual_jet_to_jet):
            for jet_dtype, weight_dtype in itertools.product((np.float32, np.float64), repeat=2):
                expected = np.float32 if np.float32 in (jet_dtype, weight_dtype) else np.float64
                with self.subTest(function=function.__name__, jet=jet_dtype.__name__, weight=weight_dtype.__name__):
                    out = function('apply', 2, jet64.astype(jet_dtype), weight64.astype(weight_dtype))
                    self.assertEqual(out.dtype, expected)
                    self.assertEqual(function('apply', 2, jet64.astype(jet_dtype), None).dtype, jet_dtype)


class TestFrontendDualJetToJet(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(5)
        self.order = 2
        self.cases = []
        for kind in ('probe', 'apply'):
            dual, weight = random_jet(rng, kind, self.order, (5,)), random_weight(rng, kind, self.order)
            zero_weight = weight.copy()
            zero_weight[..., 1] = 0.0
            self.cases.append((kind, dual, weight, zero_weight))

    def test_it_is_the_frontend_s_own_function(self):
        self.assertIsNot(fitting.dual_jet_to_jet, jets.dual_jet_to_jet)
        self.assertEqual(fitting.dual_jet_to_jet.__module__, 't3toolbox.fitting')
        self.assertEqual(list(inspect.signature(fitting.dual_jet_to_jet).parameters),
                         list(inspect.signature(jets.dual_jet_to_jet).parameters))

    def test_a_nonzero_weight_gives_the_backend_s_numbers_in_both_modes(self):
        for kind, dual, weight, _ in self.cases:
            for w in (weight, None):
                reference = jets.dual_jet_to_jet(kind, self.order, dual, w)
                safe = fitting.dual_jet_to_jet(kind, self.order, dual, w)
                with safety.unsafe():
                    unsafe = fitting.dual_jet_to_jet(kind, self.order, dual, w)
                with self.subTest(kind=kind, weight=w is not None):
                    for a, b, c in zip(per_mode(kind, reference), per_mode(kind, safe), per_mode(kind, unsafe)):
                        self.assertTrue(np.array_equal(a, b))
                        self.assertTrue(np.array_equal(a, c))

    def test_a_zero_weight_raises_in_safe_mode(self):
        for kind, dual, _, zero_weight in self.cases:
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, 'squared weight is zero'):
                    fitting.dual_jet_to_jet(kind, self.order, dual, zero_weight)
        with self.assertRaisesRegex(ValueError, 'zero in 1 entry'):
            fitting.dual_jet_to_jet('apply', 0, np.ones(3), np.zeros(()))            # order 0: the weight is one number

    def test_a_small_nonzero_weight_passes_in_safe_mode(self):
        """The check is exact, with no tolerance: a small weight is a legitimate weight. ``1e-8`` (and ``1e-150``,
        whose square is still a float64) passes in safe mode and gives the backend's numbers."""
        for kind, dual, weight, _ in self.cases:
            for small in (1e-8, 1e-150):
                small_weight = weight.copy()
                small_weight[..., 1] = small
                reference = jets.dual_jet_to_jet(kind, self.order, dual, small_weight)
                safe = fitting.dual_jet_to_jet(kind, self.order, dual, small_weight)
                with self.subTest(kind=kind, small=small):
                    for a, b in zip(per_mode(kind, reference), per_mode(kind, safe)):
                        self.assertTrue(np.isfinite(b).all())
                        self.assertTrue(np.array_equal(a, b))

    def test_a_weight_whose_square_underflows_is_refused_like_a_zero(self):
        """The check is on the divisor as the backend computes it, ``weight**2`` in the precision of the
        computation: a float32 weight of ``1e-30``, a float64 weight of ``1e-30`` against a float32 dual jet
        (the weight is cast down first), and ``1e-200`` in float64. Each divides to ``inf`` in the backend."""
        dual64 = np.ones((3, 4))
        cases = (('a float32 weight', dual64.astype(np.float32), np.array([1.0, 1e-30, 2.0], dtype=np.float32)),
                 ('a float64 weight, a float32 dual jet', dual64.astype(np.float32), np.array([1.0, 1e-30, 2.0])),
                 ('1e-200 in float64', dual64, np.array([1.0, 1e-200, 2.0])))
        for label, dual, weight in cases:
            with self.subTest(case=label):
                self.assertTrue((weight != 0).all())                                # no entry of the weight is zero
                self.assertFalse(np.isfinite(jets.dual_jet_to_jet('apply', 2, dual, weight)[1]).any())
                with self.assertRaisesRegex(ValueError, 'squared weight is zero in 1 entry'):
                    fitting.dual_jet_to_jet('apply', 2, dual, weight)
                with safety.unsafe():
                    self.assertFalse(np.isfinite(fitting.dual_jet_to_jet('apply', 2, dual, weight)[1]).any())
        # the same weight against a float64 dual jet has a nonzero square, and passes
        self.assertTrue(np.isfinite(fitting.dual_jet_to_jet('apply', 2, dual64, np.array([1.0, 1e-30, 2.0]))).all())
        # probe: one underflowing entry in one mode's row
        dual = [np.ones((3, 4, N), dtype=np.float32) for N in SHAPE]
        weight = np.ones((D, 3), dtype=np.float32)
        weight[2, 0] = 1e-30
        with self.assertRaisesRegex(ValueError, 'squared weight is zero in 1 entry'):
            fitting.dual_jet_to_jet('probe', 2, dual, weight)

    def test_a_zero_weight_divides_in_unsafe_mode(self):
        for kind, dual, _, zero_weight in self.cases:
            reference = jets.dual_jet_to_jet(kind, self.order, dual, zero_weight)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                with safety.unsafe():
                    unsafe = fitting.dual_jet_to_jet(kind, self.order, dual, zero_weight)
            with self.subTest(kind=kind):
                for a, b in zip(per_mode(kind, reference), per_mode(kind, unsafe)):
                    self.assertFalse(np.isfinite(b[1]).any())
                    self.assertTrue(np.array_equal(a, b, equal_nan=True))
        self.assertIsNotNone(safety.current_safety())                       # safe mode is back

    @unittest.skipUnless(HAS_JAX, 'jax not available')
    def test_a_zero_weight_divides_under_jit(self):
        """Under a trace the check is skipped (a tracer cannot be branched on): with the weight traced, and with
        the weight a concrete array closed over. The numbers are the backend's on numpy."""
        function = lambda dual, weight: fitting.dual_jet_to_jet(kind, self.order, dual, weight)
        for kind, dual, weight, zero_weight in self.cases:
            jdual = [jnp.asarray(x) for x in dual] if kind == 'probe' else jnp.asarray(dual)
            for w in (weight, zero_weight):
                reference = jets.dual_jet_to_jet(kind, self.order, dual, w)
                traced = jax.jit(function)(jdual, jnp.asarray(w))
                closed_over = jax.jit(lambda dual: fitting.dual_jet_to_jet(kind, self.order, dual, w))(jdual)
                with self.subTest(kind=kind, zero=w is zero_weight):
                    for a, b, c in zip(per_mode(kind, reference), per_mode(kind, traced), per_mode(kind, closed_over)):
                        self.assertEqual(np.isfinite(a).tolist(), np.isfinite(np.asarray(b)).tolist())
                        self.assertEqual(np.isfinite(a).tolist(), np.isfinite(np.asarray(c)).tolist())
                        finite = np.isfinite(a)
                        tol = 1e-5 if np.asarray(b).dtype == np.float32 else TOL
                        self.assertLessEqual(relerr(np.asarray(b)[finite], a[finite]), tol)
                        self.assertLessEqual(relerr(np.asarray(c)[finite], a[finite]), tol)
            # eager jax arrays are concrete: the check runs
            with self.subTest(kind=kind, eager=True):
                with self.assertRaisesRegex(ValueError, 'squared weight is zero'):
                    function(jdual, jnp.asarray(zero_weight))


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
        """``exp(a s)``: derivatives ``a**t``, Taylor coefficients ``a**t / t!``."""
        a, order = 1.7, 6
        derivatives = np.array([[a ** t] for t in range(order + 1)])
        taylor = np.array([[a ** t / math.factorial(t)] for t in range(order + 1)])
        self.assertLessEqual(relerr(jets.derivatives_to_taylor('apply', order, derivatives), taylor), TOL)
        self.assertLessEqual(relerr(jets.taylor_to_derivatives('apply', order, taylor), derivatives), TOL)

    def test_against_derivatives_of_the_dense_tensor_s_outputs(self):
        """The library's jets are Taylor coefficients (``dense_probe``, ``dense_apply``). The derivatives of the
        same outputs along ``ww + s pp`` are taken here from an interpolating polynomial, by ``numpy.polynomial``;
        ``derivatives_to_taylor`` of them must be the library's jets, and ``taylor_to_derivatives`` the reverse."""
        rng = np.random.default_rng(6)
        T = rng.standard_normal(SHAPE)
        ww = [rng.standard_normal((3, N)) for N in SHAPE]
        pp = [rng.standard_normal((3, N)) for N in SHAPE]
        along = lambda s: [w + s * p for w, p in zip(ww, pp)]
        order = 3                                                   # the outputs have degree d = 3 in s (probe: 2)
        apply_derivatives = interpolated_derivatives(lambda s: bapply.dense_apply(T, along(s)), order, D + 2)
        apply_taylor = bapply.dense_apply(T, jets.affine_jet(ww, pp), (order,))
        self.assertLessEqual(relerr(jets.derivatives_to_taylor('apply', order, apply_derivatives), apply_taylor), 1e-11)
        self.assertLessEqual(relerr(jets.taylor_to_derivatives('apply', order, apply_taylor), apply_derivatives), 1e-11)
        probe_taylor = probing.dense_probe(T, jets.affine_jet(ww, pp), (order,))
        probe_derivatives = [interpolated_derivatives(lambda s: probing.dense_probe(T, along(s))[i], order, D + 2)
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
                for x_TX, taylor_TX, derivatives_TX, back_TX in zip(per_mode(kind, jet), per_mode(kind, taylor),
                                                                   per_mode(kind, derivatives), per_mode(kind, back)):
                    self.assertEqual(taylor_TX.shape, x_TX.shape)
                    self.assertLessEqual(relerr(back_TX, x_TX), TOL)
                    if order == 0:
                        self.assertTrue(np.array_equal(taylor_TX, x_TX))
                        self.assertTrue(np.array_equal(derivatives_TX, x_TX))
                    for t in range(order + 1 if order else 0):
                        self.assertLessEqual(relerr(taylor_TX[t], x_TX[t] / math.factorial(t)), TOL)
                        self.assertLessEqual(relerr(derivatives_TX[t], x_TX[t] * math.factorial(t)), TOL)

    def test_the_weight_goes_the_opposite_way_on_its_last_axis(self):
        rng = np.random.default_rng(8)
        for kind in ('probe', 'apply', 'entries'):
            for order in (0, 1, 4):
                weight = random_weight(rng, kind, order)
                converted = jets.derivative_weight_to_taylor_weight(weight, order)
                with self.subTest(kind=kind, order=order):
                    self.assertEqual(converted.shape, weight.shape)
                    if order == 0:
                        self.assertTrue(np.array_equal(converted, weight))
                    for t in range(order + 1 if order else 0):
                        self.assertLessEqual(relerr(converted[..., t], weight[..., t] * math.factorial(t)), TOL)

    def test_the_objective_written_on_derivatives_is_reproduced(self):
        """The draft's case 2, for every kind: ``1/2 sum (weight * (derivative residual))**2`` equals the fit's
        objective ``1/2 sum (Taylor weight * (Taylor residual))**2`` after ``prepare_derivative_data``. With
        no weight given: the plain sum of squares of the derivative residuals."""
        rng = np.random.default_rng(9)

        def objective(kind, weight, z, y):          # the misfit of the library: 1/2 sum (weight * (z - y))**2
            return 0.5 * sum(float(np.sum((omega_T.reshape((-1,) + (1,) * (z_TX.ndim - 1)) * (z_TX - y_TX)) ** 2))
                             for omega_T, z_TX, y_TX in zip(mode_weights(kind, weight), per_mode(kind, z),
                                                            per_mode(kind, y)))

        for kind in ('probe', 'apply', 'entries'):
            for order in (1, 3):
                z_taylor = random_jet(rng, kind, order, (5,))               # the model's output, as the library gives it
                z_derivatives = jets.taylor_to_derivatives(kind, order, z_taylor)
                y_derivatives = random_jet(rng, kind, order, (5,))
                weight_derivatives = random_weight(rng, kind, order)
                ones = np.ones_like(weight_derivatives)
                with self.subTest(kind=kind, order=order):
                    wanted = objective(kind, weight_derivatives, z_derivatives, y_derivatives)
                    y, weight, validation = jets.prepare_derivative_data(kind, order, y_derivatives, weight_derivatives)
                    self.assertIsNone(validation)
                    self.assertLessEqual(abs(objective(kind, weight, z_taylor, y) - wanted), 1e-12 * wanted)
                    # the trap: no weight means ones on the DERIVATIVES
                    wanted = objective(kind, ones, z_derivatives, y_derivatives)
                    y, weight, _ = jets.prepare_derivative_data(kind, order, y_derivatives)
                    self.assertLessEqual(abs(objective(kind, weight, z_taylor, y) - wanted), 1e-12 * wanted)
                    # and it is NOT the objective with ones on the Taylor coefficients (the test can tell them
                    # apart; at order 1 the two coincide, 0! = 1! = 1)
                    if order >= 2:
                        self.assertGreater(abs(objective(kind, ones, z_taylor, y) - wanted), 1e-3 * wanted)

    def test_no_weight_returns_t_factorial(self):
        """For apply and entries ``(order+1,)``; for probe ``(d, order+1)``, every mode's row ``t!``."""
        rng = np.random.default_rng(10)
        for order in (1, 2, 5):
            factorials = [float(math.factorial(t)) for t in range(order + 1)]
            for kind in ('probe', 'apply', 'entries'):
                y = random_jet(rng, kind, order, (4,))
                y_taylor, weight, validation = jets.prepare_derivative_data(kind, order, y)
                with self.subTest(kind=kind, order=order):
                    self.assertEqual(weight.tolist(), [factorials] * D if kind == 'probe' else factorials)
                    self.assertEqual(weight.dtype, np.float64)
                    jets.check_weight(kind, order, weight, D)            # it is a weight of the kind
                    self.assertIsNone(validation)
                    for a, b in zip(per_mode(kind, y_taylor), per_mode(kind, jets.derivatives_to_taylor(kind, order, y))):
                        self.assertTrue(np.array_equal(a, b))
        # order 0: nothing to convert, and the weight is ones in the shape of a weight
        y, weight, _ = jets.prepare_derivative_data('probe', 0, random_jet(rng, 'probe', 0, (4,)))
        self.assertEqual(weight.tolist(), [1.0] * D)
        y, weight, _ = jets.prepare_derivative_data('apply', 0, random_jet(rng, 'apply', 0, (4,)))
        self.assertEqual((weight.shape, float(weight)), ((), 1.0))

    def test_a_given_weight_and_validation_data_are_converted(self):
        rng = np.random.default_rng(11)
        order = 3
        for kind in ('probe', 'apply', 'entries'):
            y, validation = random_jet(rng, kind, order, (4,)), random_jet(rng, kind, order, (2,))
            weight = random_weight(rng, kind, order)
            y_taylor, weight_taylor, validation_taylor = jets.prepare_derivative_data(kind, order, y, weight, validation)
            with self.subTest(kind=kind):
                self.assertTrue(np.array_equal(weight_taylor, jets.derivative_weight_to_taylor_weight(weight, order)))
                for a, b in zip(per_mode(kind, validation_taylor),
                                per_mode(kind, jets.derivatives_to_taylor(kind, order, validation))):
                    self.assertTrue(np.array_equal(a, b))
                for a, b in zip(per_mode(kind, y_taylor), per_mode(kind, jets.derivatives_to_taylor(kind, order, y))):
                    self.assertTrue(np.array_equal(a, b))

    def test_the_dtype_is_kept_and_an_unknown_kind_raises(self):
        y32 = np.ones((4, 3), dtype=np.float32)
        y, weight, validation = jets.prepare_derivative_data('apply', 3, y32, validation=y32)
        self.assertEqual((y.dtype, weight.dtype, validation.dtype), (np.float32,) * 3)
        self.assertEqual(jets.taylor_to_derivatives('apply', 3, y32).dtype, np.float32)
        self.assertEqual(jets.derivative_weight_to_taylor_weight(np.ones(4, dtype=np.float32), 3).dtype, np.float32)
        with self.assertRaisesRegex(ValueError, 'unknown sampling kind'):
            jets.prepare_derivative_data('probe_derivatives', 3, y32)


class TestCheckers(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(12)
        self.rng = rng
        self.W = (7,)
        self.ww = [rng.standard_normal(self.W + (N,)) for N in SHAPE]
        self.pp = [rng.standard_normal(self.W + (N,)) for N in SHAPE]
        self.index = np.stack([rng.integers(0, N, size=self.W) for N in SHAPE])

    def good_sample(self, kind, order, W=None):
        W = self.W if W is None else W
        rng = self.rng
        ww = [rng.standard_normal(W + (N,)) for N in SHAPE]
        pp = tuple(rng.standard_normal(W + (N,)) for N in SHAPE)
        index = np.stack([rng.integers(0, N, size=W) for N in SHAPE]) if W else np.array([1, 2, 3])
        first = index if kind == 'entries' else ww
        return first if order == 0 else (first, pp)

    def good_data(self, kind, order, W=None):
        W = self.W if W is None else W
        if kind == 'probe':
            return [np.zeros(T_of(order) + W + (N,)) for N in SHAPE]
        return np.zeros(T_of(order) + W)

    def assert_raises(self, pattern, function, *args):
        with self.assertRaisesRegex(ValueError, pattern):
            function(*args)

    # ------------------------------------------------------------------------------------------ check_sample
    def test_check_sample_passes_every_good_sample(self):
        for kind in ('probe', 'apply', 'entries'):
            for order in (0, 1, 3):
                for W in ((7,), (), (2, 3)):
                    with self.subTest(kind=kind, order=order, W=W):
                        self.assertIsNone(jets.check_sample(kind, order, self.good_sample(kind, order, W), SHAPE))
        self.assertIsNone(jets.check_sample('apply', 0, tuple(self.ww), list(SHAPE)))       # a tuple, and a list of sizes
        self.assertIsNone(jets.check_sample('entries', 1, [self.index, self.pp], SHAPE))    # the pair as a list
        self.assertIsNone(jets.check_sample('entries', 0, self.index.astype(np.int32), SHAPE))

    def test_check_sample_raises_on_each_wrong_form_of_the_vectors(self):
        ww, pp = self.ww, self.pp
        for kind in ('probe', 'apply'):
            with self.subTest(kind=kind):
                self.assert_raises('one array per mode: expected 3, got 2', jets.check_sample, kind, 0, ww[:2], SHAPE)
                self.assert_raises('one array per mode: expected 3, got 4', jets.check_sample, kind, 0, ww + ww[:1], SHAPE)
                wrong_size = ww[:2] + [np.zeros((7, 9))]
                self.assert_raises(r'ww\[2\] has shape \(7, 9\)', jets.check_sample, kind, 0, wrong_size, SHAPE)
                wrong_stack = [ww[0], np.zeros((8, 5)), ww[2]]
                self.assert_raises(r'ww\[1\] has shape \(8, 5\)', jets.check_sample, kind, 0, wrong_stack, SHAPE)
                packed = np.zeros((3, 7, 6))                                    # the uniform layout's packed form
                self.assert_raises('ww must be a tuple or a list', jets.check_sample, kind, 0, packed, SHAPE)
                self.assert_raises(r'ww\[0\] must be one array', jets.check_sample, kind, 0, [1.0, ww[1], ww[2]], SHAPE)
                # the order decides the form
                self.assert_raises(r'must be the pair \(ww, pp\)', jets.check_sample, kind, 2, ww, SHAPE)
                self.assert_raises(r'must be the pair \(ww, pp\)', jets.check_sample, kind, 2, (ww, pp, pp), SHAPE)
                self.assert_raises('one array per mode: expected 3, got 2', jets.check_sample, kind, 0, (ww, pp), SHAPE)
                # pp unlike ww
                self.assert_raises('pp must have one array per mode', jets.check_sample, kind, 2, (ww, pp[:2]), SHAPE)
                unlike = [pp[0], np.zeros((7, 9)), pp[2]]
                self.assert_raises(r'pp\[1\] has shape \(7, 9\)', jets.check_sample, kind, 2, (ww, unlike), SHAPE)
                other_stack = [np.zeros((8, N)) for N in SHAPE]
                self.assert_raises("pp's sample stack W", jets.check_sample, kind, 2, (ww, other_stack), SHAPE)
                self.assert_raises('pp must be a tuple or a list', jets.check_sample, kind, 2, (ww, np.zeros((3, 7, 6))), SHAPE)
        # with two modes a pair at order 0 has the right count; its elements are not arrays
        two = (4, 5)
        self.assert_raises(r'ww\[0\] must be one array', jets.check_sample, 'apply', 0, (ww[:2], pp[:2]), two)

    def test_check_sample_refuses_a_missing_pp(self):
        """At order 1 and above the sample is a pair and ``pp`` is required: ``(ww, None)`` (left from an order-0
        code path) is refused, with a message that says so."""
        for kind, first, name in (('probe', self.ww, 'ww'), ('apply', self.ww, 'ww'), ('entries', self.index, 'index')):
            for order in (1, 3):
                with self.subTest(kind=kind, order=order):
                    self.assert_raises(r'at order %d the sample of %r needs the directions pp: it is the pair '
                                       r'\(%s, pp\); got \(%s, None\)' % (order, kind, name, name),
                                       jets.check_sample, kind, order, (first, None), SHAPE)
                    self.assert_raises('needs the directions pp', jets.check_sample, kind, order, [first, None], SHAPE)

    def test_check_sample_names_the_cause_when_the_form_is_that_of_the_other_order(self):
        ww, pp, index = self.ww, self.pp, self.index
        at_order_0 = r'At order 0 the sample is the inputs %s alone, with no pp; this sample looks like the pair'
        self.assert_raises('expected 3, got 2. ' + at_order_0 % 'ww', jets.check_sample, 'apply', 0, (ww, pp), SHAPE)
        self.assert_raises('got a tuple. ' + at_order_0 % 'index', jets.check_sample, 'entries', 0, (index, pp), SHAPE)
        self.assert_raises('got a list. ' + at_order_0 % 'ww', jets.check_sample, 'probe', 0, (ww[:2], pp[:2]), (4, 5))
        # two modes: ww alone at order 1 has the length of a pair
        self.assert_raises(r'got a ndarray. At order 1 the sample is the pair \(ww, pp\); this sample looks like ww alone',
                           jets.check_sample, 'apply', 1, ww[:2], (4, 5))
        # a wrong sample that does not look like the other form gets no such hint
        with self.assertRaises(ValueError) as caught:
            jets.check_sample('apply', 0, ww[:2], SHAPE)
        self.assertNotIn('looks like', str(caught.exception))

    def test_check_sample_raises_on_each_wrong_form_of_the_index(self):
        index, pp = self.index, self.pp
        self.assert_raises(r'index must have shape \(d,\)\+W with d=3', jets.check_sample, 'entries', 0, index[:2], SHAPE)
        self.assert_raises(r'index must have shape \(d,\)\+W', jets.check_sample, 'entries', 0, np.asarray(3), SHAPE)
        self.assert_raises('index must be an integer array', jets.check_sample, 'entries', 0, index.astype(float), SHAPE)
        self.assert_raises(r'must be the pair \(index, pp\)', jets.check_sample, 'entries', 2, index, SHAPE)
        self.assert_raises('index must be one integer array', jets.check_sample, 'entries', 0, (index, pp), SHAPE)
        self.assert_raises('index must be one integer array', jets.check_sample, 'entries', 0, index.tolist(), SHAPE)
        self.assert_raises('pp must have one array per mode', jets.check_sample, 'entries', 2, (index, pp[:2]), SHAPE)
        other_stack = [np.zeros((8, N)) for N in SHAPE]
        self.assert_raises("pp's sample stack W", jets.check_sample, 'entries', 2, (index, other_stack), SHAPE)
        unlike = [pp[0], np.zeros((7, 9)), pp[2]]
        self.assert_raises(r'pp\[1\] has shape \(7, 9\)', jets.check_sample, 'entries', 2, (index, unlike), SHAPE)

    def test_the_checkers_raise_on_an_unknown_kind_and_a_bad_order(self):
        for function, args in ((jets.check_sample, (self.ww, SHAPE)), (jets.check_data, (self.ww, np.zeros(7))),
                               (jets.check_weight, (None, D))):
            with self.subTest(function=function.__name__):
                self.assert_raises('unknown sampling kind', function, 'apply_derivatives', 0, *args)
                self.assert_raises('order must be an integer >= 0', function, 'apply', -1, *args)
                self.assert_raises('order must be an integer >= 0', function, 'apply', 1.0, *args)
                self.assert_raises('order must be an integer >= 0', function, 'apply', True, *args)

    # ------------------------------------------------------------------------------------------ check_data
    def test_check_data_passes_every_good_y(self):
        for kind in ('probe', 'apply', 'entries'):
            for order in (0, 1, 3):
                for W in ((7,), (), (2, 3)):
                    with self.subTest(kind=kind, order=order, W=W):
                        sample = self.good_sample(kind, order, W)
                        self.assertIsNone(jets.check_data(kind, order, sample, self.good_data(kind, order, W)))
        self.assertIsNone(jets.check_data('probe', 0, self.ww, tuple(self.good_data('probe', 0))))

    def test_check_data_raises_on_each_wrong_y(self):
        ww, pp, index = self.ww, self.pp, self.index
        good = self.good_data('probe', 2)
        sample = (ww, pp)
        self.assert_raises('one array per mode: expected 3, got 2', jets.check_data, 'probe', 2, sample, good[:2])
        self.assert_raises('must be a tuple or a list of 3 arrays', jets.check_data, 'probe', 2, sample, np.zeros((3, 3, 7, 6)))
        wrong_size = good[:2] + [np.zeros((3, 7, 9))]
        self.assert_raises(r'y\[2\] has shape \(3, 7, 9\); expected T\+W\+\(N_2,\) = \(3, 7, 6\)',
                           jets.check_data, 'probe', 2, sample, wrong_size)
        no_order_axis = self.good_data('probe', 0)
        self.assert_raises(r'y\[0\] has shape \(7, 4\); expected T\+W\+\(N_0,\) = \(3, 7, 4\)',
                           jets.check_data, 'probe', 2, sample, no_order_axis)
        self.assert_raises(r'y\[0\] has shape \(3, 7, 4\); expected T\+W\+\(N_0,\) = \(7, 4\)',
                           jets.check_data, 'probe', 0, ww, good)                    # an order axis at order 0
        wrong_order = [np.zeros((4, 7, N)) for N in SHAPE]
        self.assert_raises(r'y\[0\] has shape \(4, 7, 4\)', jets.check_data, 'probe', 2, sample, wrong_order)
        wrong_stack = [np.zeros((3, 8, N)) for N in SHAPE]
        self.assert_raises(r'y\[0\] has shape \(3, 8, 4\)', jets.check_data, 'probe', 2, sample, wrong_stack)
        with_C = [np.zeros((3, 7, 2, N)) for N in SHAPE]                            # a stack of base points
        self.assert_raises(r'y\[0\] has shape \(3, 7, 2, 4\)', jets.check_data, 'probe', 2, sample, with_C)
        for kind, sample0, sample2 in (('apply', ww, (ww, pp)), ('entries', index, (index, pp))):
            with self.subTest(kind=kind):
                self.assert_raises(r'y has shape \(7,\); expected T\+W = \(3, 7\)', jets.check_data, kind, 2, sample2, np.zeros(7))
                self.assert_raises(r'y has shape \(3, 7\); expected T\+W = \(7,\)', jets.check_data, kind, 0, sample0, np.zeros((3, 7)))
                self.assert_raises(r'y has shape \(3, 8\)', jets.check_data, kind, 2, sample2, np.zeros((3, 8)))
                self.assert_raises(r'y has shape \(3, 7, 2\)', jets.check_data, kind, 2, sample2, np.zeros((3, 7, 2)))
                self.assert_raises('y has a list of length 3', jets.check_data, kind, 2, sample2, good)
                self.assert_raises(r'must be the pair', jets.check_data, kind, 2, sample0, np.zeros((3, 7)))

    # ------------------------------------------------------------------------------------------ check_weight
    def test_check_weight_passes_the_exact_shape_and_none(self):
        for kind in ('probe', 'apply', 'entries'):
            for order in (0, 1, 3):
                shape = ((D,) if kind == 'probe' else ()) + T_of(order)
                with self.subTest(kind=kind, order=order):
                    self.assertIsNone(jets.check_weight(kind, order, np.ones(shape), D))
                    self.assertIsNone(jets.check_weight(kind, order, None, D))
        self.assertIsNone(jets.check_weight('apply', 0, 2.0, D))                    # a number is a 0-d weight
        self.assertIsNone(jets.check_weight('probe', 1, [[1.0, 2.0]] * D, D))       # nested lists have a shape

    def test_check_weight_raises_on_every_other_shape(self):
        order = 3
        cases = [
            # the old forms (2026.5.0): a bare vector meant "per order", a mode dimension of 1 was broadcast
            ('probe', order, np.ones(order + 1), 'a bare per-order vector'),
            ('probe', order, np.ones((1, order + 1)), 'a mode dimension of 1'),
            ('probe', order, np.ones((D, 1)), 'an order dimension of 1'),
            ('probe', order, np.ones(D), 'a bare per-mode vector at order 3'),
            ('probe', order, np.ones((D + 1, order + 1)), 'more rows than modes'),
            ('probe', order, np.ones((order + 1, D)), 'the two axes exchanged'),
            ('probe', 0, np.ones((D, 1)), 'plain probe with an order axis'),
            ('probe', 0, np.ones(1), 'plain probe, a mode dimension of 1'),
            ('probe', 0, np.ones(()), 'plain probe, a number'),
            ('apply', order, np.ones((1, order + 1)), 'a mode dimension of 1'),
            ('apply', order, np.ones((D, order + 1)), 'a mode axis'),
            ('apply', order, np.ones(1), 'an order dimension of 1'),
            ('apply', order, np.ones(()), 'a number at order 3'),
            ('apply', 0, np.ones(1), 'a vector at order 0'),
            ('entries', order, np.ones((1, order + 1)), 'a mode dimension of 1'),
            ('entries', order, np.ones(order), 'one slot short'),
        ]
        for kind, order_, weight, what in cases:
            with self.subTest(kind=kind, order=order_, what=what):
                expected = ((D,) if kind == 'probe' else ()) + T_of(order_)
                pattern = 'has shape %s; expected exactly .* = %s' % tuple(
                    str(s).replace('(', r'\(').replace(')', r'\)') for s in (weight.shape, expected))
                self.assert_raises(pattern, jets.check_weight, kind, order_, weight, D)

    def test_check_weight_cannot_see_a_transposed_square_weight(self):
        """The limit the docstring states: with ``d == order + 1`` a probe weight with its axes exchanged has
        the right shape, and passes."""
        weight = np.arange(1.0, 10.0).reshape(D, D)
        self.assertIsNone(jets.check_weight('probe', D - 1, weight, D))
        self.assertIsNone(jets.check_weight('probe', D - 1, weight.T, D))
        self.assertIn('transposed', jets.check_weight.__doc__)

    # ------------------------------------------------------------------------------------------ the tier
    def test_no_backend_module_calls_a_checker(self):
        """Checkers, not guarantees: outside ``jets`` itself, no backend module refers to one of the three new
        checkers (the two validators of ``pp`` are called only by ``check_sample`` and, until P20, the old ones
        of ``sampling_derivatives`` stand beside them)."""
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
                    if path.name == 'jets.py':
                        self.assertTrue({'check_perturbation_vectors', 'check_perturbation_index'} <= calls)
                else:
                    self.assertEqual(used & names, set())

    def test_the_two_validators_of_pp_are_copies_of_the_old_ones(self):
        """Copied unchanged from ``sampling_derivatives`` (which goes in P20, and this test with it)."""
        for name in ('check_perturbation_vectors', 'check_perturbation_index'):
            with self.subTest(name=name):
                self.assertEqual(inspect.getsource(getattr(jets, name)),
                                 inspect.getsource(getattr(sampling_derivatives, name)))
        self.assertIsNone(jets.check_perturbation_vectors(self.ww, self.pp))
        self.assertIsNone(jets.check_perturbation_index(self.index, self.pp, SHAPE))
        self.assert_raises('perturbation P must match', jets.check_perturbation_vectors, self.ww, self.pp[::-1])
        self.assert_raises("P's probe stack W must match", jets.check_perturbation_index, self.index[:, :3], self.pp)
        self.assert_raises("P's mode dim must match", jets.check_perturbation_index, self.index, self.pp, (4, 5, 9))

    def test_every_public_function_has_a_doctest(self):
        for name in jets.__all__:
            with self.subTest(name=name):
                if name.startswith('check_perturbation'):
                    continue                                                 # copied unchanged, with their docstrings
                self.assertIn('>>> ', getattr(jets, name).__doc__)
        self.assertIn('>>> ', jets.__doc__)


class TestFrontendReach(unittest.TestCase):

    def test_the_names_of_t3toolbox_fitting(self):
        """The preparation functions and the jet helpers are reachable from ``t3toolbox.fitting`` under their
        backend names. Nine are the backend functions themselves. Two are the frontend's own functions, with the
        backend's name and parameters, which make a check and call the backend one: ``dual_jet_to_jet`` (the
        zero divisor, in safe mode) and ``rescale_coordinates`` (the form of the scales, always). The checkers are
        not re-exported."""
        import t3toolbox.backend.data_preparation as data_preparation
        from_jets = ('derivatives_to_taylor', 'taylor_to_derivatives', 'derivative_weight_to_taylor_weight',
                     'prepare_derivative_data', 'jet_to_dual_jet')
        from_preparation = ('data_block_norms', 'equalizing_weight', 'rescale_direction', 't3_unscale_coordinates')
        for module, names in ((jets, from_jets), (data_preparation, from_preparation)):
            for name in names:
                with self.subTest(name=name):
                    self.assertIs(getattr(fitting, name), getattr(module, name))
                    self.assertIn(name, fitting.__all__)
        for module, name in ((jets, 'dual_jet_to_jet'), (data_preparation, 'rescale_coordinates')):
            with self.subTest(own=name):
                own, backend = getattr(fitting, name), getattr(module, name)
                self.assertIsNot(own, backend)
                self.assertEqual(own.__module__, 't3toolbox.fitting')
                self.assertEqual(list(inspect.signature(own).parameters), list(inspect.signature(backend).parameters))
                self.assertIn(name, fitting.__all__)
        for name in ('check_sample', 'check_data', 'check_weight', 'check_coordinate_scales',
                     'check_perturbation_vectors', 'check_perturbation_index', 'ut3_unscale_coordinates'):
            self.assertNotIn(name, fitting.__all__)
            self.assertFalse(hasattr(fitting, name), name)
        self.assertEqual(len(fitting.__all__), len(set(fitting.__all__)))


if __name__ == '__main__':
    unittest.main()
