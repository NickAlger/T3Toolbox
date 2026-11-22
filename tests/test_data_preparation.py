# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.data_preparation``, and ``unscale_coordinates`` on the two tensor classes.

The jet shape ``order = (n_1, ..., n_m)`` is a tuple, the box ``F = (n_1 + 1, ...)``, and a sample of the probe and
apply kinds is its input jets ``ww`` (``F_in_i + W + (Ni,)``); the entries kind has no jets.

1. **The block norms and the equalizing weight** (`TestEqualizing`): the norms against ``numpy.linalg.norm``
   block by block, in the shape of the weight for every jet shape (``check_weight`` accepts them); the weight
   against ``1 / max(x, floor * max(x))``, the floor weight on a zero block, and that the floor is relative.
2. **Rescaling the direction** (`TestRescaleDirection`): the rescaled data are the data of the SAME dense
   tensor at the rescaled input jets (the dense functions), so the unknowns do not change; at ``m = 1`` and
   ``m = 2``, with one length and with one per jet variable; slot ``k`` times ``length_1**k_1 ... length_m**k_m``.
3. **Rescaling the coordinates** (`TestRescaleCoordinates`): the rescaled data are the data of the tensor in
   the new coordinates; ``t3_unscale_coordinates`` then gives back the same probe values in the original
   coordinates, at an input jet; the method equals the function.
4. **The uniform twin** (`TestUniformUnscaleCoordinates`), by the three prongs of
   ``docs/contributor/testing_strategy.md``: equal to ragged on real parts over the ``(C, force_pad)`` matrix and
   a stack of varying ranks; the output masks exactly the input's; garbage in every padded slot of the input.

Numerical comparisons are to a tolerance, structural ones exact. numpy only, by the house convention; the
functions under jit are in ``test_dispatch.py``.
"""
import math
import unittest
import unittest.mock
import warnings

import numpy as np

import t3toolbox.fitting as fitting
import t3toolbox.safety as safety
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.apply as bapply
import t3toolbox.backend.data_preparation as prep
import t3toolbox.backend.entries as bentries
import t3toolbox.backend.jets as jets
import t3toolbox.backend.probing as probing
import t3toolbox.backend.ut3_conversions as ut3_conversions

TOL = 1e-12
SHAPE = (4, 5, 6)           # the mode sizes N_i
D = len(SHAPE)
STRUCTURE = (SHAPE, (2, 3, 2), (1, 2, 2, 1))
JET_SHAPES = ((), (1,), (3,), (1, 2))


def relerr(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b))


def F_of(order):
    return tuple(n + 1 for n in order)


def per_mode(kind, value):
    return list(value) if kind == 'probe' else [value]


def dense_sample(kind, T, sample, order):
    """The data of the dense tensor ``T`` at a sample, of the type of z, through the dense functions."""
    if kind == 'entries':
        return bentries.dense_entries(T, sample)
    return {'probe': probing.dense_probe, 'apply': bapply.dense_apply}[kind](T, sample, order)


def random_sample(rng, kind, order, W=(7,), sizes=None):
    """Input jets of the given jet sizes per mode (default: the box ``F``), or an index for the entries kind."""
    if kind == 'entries':
        return np.stack([rng.integers(0, N, size=W) for N in SHAPE])
    sizes = [F_of(order)] * D if sizes is None else sizes
    return [rng.standard_normal(tuple(s) + W + (N,)) for s, N in zip(sizes, SHAPE)]


def random_data(rng, kind, order, W=(7,)):
    if kind == 'probe':
        return [rng.standard_normal(F_of(order) + W + (N,)) for N in SHAPE]
    return rng.standard_normal(F_of(order) + W)


def slot_factor(lengths, k):
    """``length_1**k_1 ... length_m**k_m``, written out."""
    return math.prod(length ** k_j for length, k_j in zip(lengths, k))


class TestEqualizing(unittest.TestCase):

    def test_the_block_norms_have_the_shape_of_the_weight(self):
        rng = np.random.default_rng(0)
        for kind in ('probe', 'apply', 'entries'):
            for order in (JET_SHAPES if kind != 'entries' else ((),)):
                for W in ((7,), (), (2, 3)):
                    y = random_data(rng, kind, order, W)
                    norms = prep.data_block_norms(kind, order, y)
                    with self.subTest(kind=kind, order=order, W=W):
                        self.assertEqual(norms.shape, ((D,) if kind == 'probe' else ()) + F_of(order))
                        jets.check_weight(kind, order, norms, D)             # the checker of a weight accepts it
                        jets.check_weight(kind, order, prep.equalizing_weight(norms), D)

    def test_the_block_norms_are_the_norms_of_the_blocks(self):
        rng = np.random.default_rng(1)
        for kind in ('probe', 'apply', 'entries'):
            for order in (JET_SHAPES if kind != 'entries' else ((),)):
                y = random_data(rng, kind, order, (2, 3))
                norms = prep.data_block_norms(kind, order, y)
                rows = [norms[i] for i in range(D)] if kind == 'probe' else [norms]
                with self.subTest(kind=kind, order=order):
                    for norms_F, y_FX in zip(rows, per_mode(kind, y)):
                        for k in np.ndindex(*F_of(order)):                  # with no jets, the one block ()
                            self.assertLessEqual(abs(float(norms_F[k]) - np.linalg.norm(y_FX[k])),
                                                 TOL * np.linalg.norm(y_FX[k]))
        with self.assertRaisesRegex(ValueError, 'unknown sampling kind'):
            prep.data_block_norms('probe_derivatives', (), y)
        with self.assertRaisesRegex(ValueError, 'the entries kind has no jets'):
            prep.data_block_norms('entries', (1,), np.zeros((2, 7)))
        with self.assertRaisesRegex(TypeError, 'order is the jet shape'):
            prep.data_block_norms('apply', 1, np.zeros((2, 7)))

    def test_the_equalizing_weight_is_the_filtered_inverse(self):
        rng = np.random.default_rng(2)
        norms = 10.0 ** rng.uniform(-9.0, 2.0, size=(D, 4))                 # blocks over eleven decades
        for floor in (1e-6, 1e-3, 0.5):
            weight = prep.equalizing_weight(norms, floor) if floor != 1e-6 else prep.equalizing_weight(norms)
            cut = floor * norms.max()
            with self.subTest(floor=floor):
                self.assertEqual(weight.shape, norms.shape)
                self.assertTrue((norms < cut).any() and (norms > cut).any())            # both branches are met
                for x, w in zip(norms.ravel(), weight.ravel()):
                    self.assertLessEqual(abs(w - 1.0 / max(x, cut)), TOL / max(x, cut))
                above = norms >= cut
                self.assertLessEqual(np.abs(weight[above] * norms[above] - 1.0).max(), TOL)     # equalized
                self.assertLessEqual(np.abs(weight[~above] * cut - 1.0).max(), TOL)             # the floor weight

    def test_a_zero_block_gets_the_floor_weight(self):
        norms = np.array([[5.0, 0.5, 0.0], [2.0, 0.0, 1e-9]])
        weight = prep.equalizing_weight(norms)
        self.assertTrue(np.isfinite(weight).all())
        self.assertLessEqual(np.abs(weight[norms == 0.0] * (1e-6 * 5.0) - 1.0).max(), TOL)      # 1 / (floor * max)
        self.assertLessEqual(abs(weight[1, 2] * (1e-6 * 5.0) - 1.0), TOL)                       # below the floor too
        self.assertLessEqual(np.abs(weight[norms > 1e-3] * norms[norms > 1e-3] - 1.0).max(), TOL)
        weight = prep.equalizing_weight(norms, floor=0.25)
        self.assertLessEqual(np.abs(weight - 1.0 / np.array([[5.0, 1.25, 1.25], [2.0, 1.25, 1.25]])).max(), TOL)

    def test_the_floor_is_relative_to_the_largest_block(self):
        """Equalizing does not depend on the units of the data: ``weight(c x) = weight(x) / c``."""
        norms = np.array([5.0, 0.5, 0.0, 1e-9])
        for c in (1e-8, 1e8):
            self.assertLessEqual(relerr(prep.equalizing_weight(c * norms), prep.equalizing_weight(norms) / c), TOL)

    def test_all_zero_norms_give_infinity_silently_and_a_number_gives_an_array(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')                                  # a warning would be an error here
            weight = prep.equalizing_weight(np.zeros((D, 3)))
            single = prep.equalizing_weight(np.zeros(()))
        self.assertEqual(weight.shape, (D, 3))
        self.assertTrue(np.isposinf(weight).all())
        self.assertTrue(type(single) is np.ndarray and single.shape == () and np.isposinf(single))
        # a 0-dimensional input (the apply kind with no jets) gives a 0-dimensional array, not a numpy scalar
        weight = prep.equalizing_weight(prep.data_block_norms('apply', (), np.array([3.0, 4.0])))
        self.assertTrue(type(weight) is np.ndarray and weight.shape == ())
        self.assertLessEqual(abs(float(weight) - 0.2), TOL)

    def test_the_weighted_blocks_of_real_data_have_norm_one(self):
        rng = np.random.default_rng(3)
        T = rng.standard_normal(SHAPE)
        ww = [rng.standard_normal((7, N)) for N in SHAPE]
        pp = [1e-2 * rng.standard_normal((7, N)) for N in SHAPE]
        qq = [1e-1 * rng.standard_normal((7, N)) for N in SHAPE]
        for order, sample in (((2,), jets.affine_jet(ww, pp)), ((1, 1), jets.affine_jet(ww, pp, qq))):
            y = probing.dense_probe(T, sample, order)
            weight = prep.equalizing_weight(prep.data_block_norms('probe', order, y))
            weighted = [weight[i].reshape(F_of(order) + (1, 1)) * y[i] for i in range(D)]
            with self.subTest(order=order):
                self.assertLessEqual(np.abs(prep.data_block_norms('probe', order, weighted) - 1.0).max(), TOL)


class TestRescaleDirection(unittest.TestCase):

    def cases(self, rng):
        """(order, input jets, lengths as given, the length per jet variable): a line, a plane and input jets of
        sizes of their own (one length, and one per jet variable)."""
        ww, pp, qq = ([rng.standard_normal((7, N)) for N in SHAPE] for _ in range(3))
        line, plane = jets.affine_jet(ww, pp), jets.affine_jet(ww, pp, qq)
        mixed = random_sample(rng, 'probe', (1, 2), sizes=[(2, 3), (1, 2), (2, 1)])
        return [((1,), line, 100.0, (100.0,)), ((3,), line, 0.3, (0.3,)), ((3,), random_sample(rng, 'apply', (3,)), 2.0, (2.0,)),
                ((1, 2), plane, 10.0, (10.0, 10.0)), ((1, 2), plane, (2.0, 0.1), (2.0, 0.1)),
                ((2, 1), plane, np.array([0.5, 3.0]), (0.5, 3.0)), ((1, 2), mixed, [3.0, 0.5], (3.0, 0.5))]

    def test_the_rescaled_data_are_the_data_of_the_same_tensor(self):
        """The new variables ``s_j = length_j s'_j``: slot ``k`` of each input jet and of the data times
        ``length_1**k_1 ... length_m**k_m``, and the data are those of the same dense tensor at the new input jets."""
        rng = np.random.default_rng(4)
        T = rng.standard_normal(SHAPE)
        for kind in ('probe', 'apply'):
            for order, sample, length, lengths in self.cases(rng):
                y = dense_sample(kind, T, sample, order)
                sample_scaled, y_scaled = prep.rescale_direction(kind, order, sample, y, length)
                with self.subTest(kind=kind, order=order, length=str(length)):
                    self.assertEqual((type(sample_scaled), len(sample_scaled)), (tuple, D))
                    for w_scaled, w in zip(sample_scaled, sample):
                        self.assertEqual(w_scaled.shape, w.shape)
                        for k in np.ndindex(*w.shape[:len(order)]):           # the input jet's own sizes
                            self.assertLessEqual(np.linalg.norm(w_scaled[k] - slot_factor(lengths, k) * w[k]),
                                                 TOL * slot_factor(lengths, k) * np.linalg.norm(w[k]))
                    reference = dense_sample(kind, T, sample_scaled, order)
                    self.assertEqual(type(y_scaled), tuple if kind == 'probe' else np.ndarray)
                    for a_FX, b_FX, y_FX in zip(per_mode(kind, y_scaled), per_mode(kind, reference), per_mode(kind, y)):
                        self.assertLessEqual(relerr(a_FX, b_FX), TOL)
                        self.assertEqual(a_FX.shape, y_FX.shape)
                        for k in np.ndindex(*F_of(order)):    # (a probe's slot of total degree above d - 1 is zero)
                            self.assertLessEqual(np.linalg.norm(a_FX[k] - slot_factor(lengths, k) * y_FX[k]),
                                                 TOL * slot_factor(lengths, k) * np.linalg.norm(y_FX))

    def test_the_block_norms_scale_by_the_length_to_the_multi_index(self):
        rng = np.random.default_rng(5)
        for order, lengths in (((3,), 7.0), ((1, 2), (7.0, 0.5))):
            sample, y = random_sample(rng, 'probe', order), random_data(rng, 'probe', order)
            _, y_scaled = prep.rescale_direction('probe', order, sample, y, lengths)
            before, after = prep.data_block_norms('probe', order, y), prep.data_block_norms('probe', order, y_scaled)
            factor = np.ones(F_of(order))
            for k in np.ndindex(*F_of(order)):
                factor[k] = slot_factor(np.broadcast_to(lengths, (len(order),)), k)
            with self.subTest(order=order):
                self.assertLessEqual(relerr(after, before * factor), TOL)

    def test_the_length_may_be_a_zero_dimensional_array_or_a_sequence(self):
        rng = np.random.default_rng(14)
        sample, y = random_sample(rng, 'apply', (2,)), random_data(rng, 'apply', (2,))
        sample_a, y_a = prep.rescale_direction('apply', (2,), sample, y, 3.0)
        for length in (np.asarray(3.0), [3.0], (3.0,), np.array([3.0])):
            sample_b, y_b = prep.rescale_direction('apply', (2,), sample, y, length)
            with self.subTest(length=repr(length)):
                self.assertTrue(np.array_equal(y_a, y_b) and all(np.array_equal(a, b) for a, b in zip(sample_a, sample_b)))
        sample_c, y_c = prep.rescale_direction('apply', (2,), sample, y.astype(np.float32), np.asarray(3.0))
        self.assertEqual((y_c.dtype, sample_c[0].dtype), (np.float32, np.float64))      # each in its own array's dtype
        for bad in ((2.0, 3.0), np.ones((1, 1)), [[2.0]]):
            with self.subTest(bad=repr(bad)):
                with self.assertRaisesRegex(ValueError, 'length is one number, or one per jet variable'):
                    prep.rescale_direction('apply', (2,), sample, y, bad)
        with self.assertRaisesRegex(ValueError, 'length is one number, or one per jet variable'):
            prep.rescale_direction('apply', (1, 2), random_sample(rng, 'apply', (1, 2)),
                                   random_data(rng, 'apply', (1, 2)), (2.0, 3.0, 4.0))

    def test_no_jets_entries_an_unknown_kind_and_an_integer_order_raise(self):
        rng = np.random.default_rng(6)
        with self.assertRaisesRegex(ValueError, 'at least one jet variable'):
            prep.rescale_direction('apply', (), random_sample(rng, 'apply', ()), np.zeros(7), 2.0)
        with self.assertRaisesRegex(ValueError, 'the entries kind has no jets'):
            prep.rescale_direction('entries', (), random_sample(rng, 'entries', ()), np.zeros(7), 2.0)
        with self.assertRaisesRegex(ValueError, 'unknown sampling kind'):
            prep.rescale_direction('apply_derivatives', (1,), random_sample(rng, 'apply', (1,)), np.zeros((2, 7)), 2.0)
        with self.assertRaisesRegex(TypeError, 'order is the jet shape'):
            prep.rescale_direction('apply', 1, random_sample(rng, 'apply', (1,)), np.zeros((2, 7)), 2.0)

    def test_the_dtype_is_kept(self):
        rng = np.random.default_rng(7)
        sample32 = [w.astype(np.float32) for w in random_sample(rng, 'apply', (2,))]
        sample_scaled, y_scaled = prep.rescale_direction('apply', (2,), sample32, np.ones((3, 7), dtype=np.float32), 3.0)
        self.assertEqual({w.dtype for w in sample_scaled} | {y_scaled.dtype}, {np.dtype(np.float32)})


class TestRescaleCoordinates(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.rng = np.random.default_rng(8)
        self.x = t3.TuckerTensorTrain.randn(*STRUCTURE)
        self.scales = [10.0 ** self.rng.uniform(-2.0, 2.0, size=N) for N in SHAPE]      # one scale per coordinate
        # the same tensor seen in the rescaled coordinates: what a fit to the rescaled sample finds
        self.x_rescaled = t3.TuckerTensorTrain(tuple(B / s for B, s in zip(self.x.tucker_cores, self.scales)),
                                               self.x.tt_cores)

    def test_the_rescaled_data_are_the_data_of_the_tensor_in_the_new_coordinates(self):
        rng = self.rng
        dense, dense_rescaled = self.x.to_dense(), self.x_rescaled.to_dense()
        for kind in ('probe', 'apply'):
            for order in ((), (2,), (1, 2)):
                sample = random_sample(rng, kind, order)
                y = dense_sample(kind, dense, sample, order)
                sample_rescaled, y_rescaled = prep.rescale_coordinates(kind, order, sample, y, self.scales)
                with self.subTest(kind=kind, order=order):
                    self.assertEqual((type(sample_rescaled), len(sample_rescaled)), (tuple, D))
                    for i in range(D):                                          # every slot of every input jet
                        self.assertLessEqual(relerr(sample_rescaled[i], sample[i] * self.scales[i]), TOL)
                    reference = dense_sample(kind, dense_rescaled, sample_rescaled, order)
                    for a_FX, b_FX in zip(per_mode(kind, y_rescaled), per_mode(kind, reference)):
                        self.assertLessEqual(relerr(a_FX, b_FX), 1e-11)
                    if kind == 'apply':
                        self.assertIs(y_rescaled, y)                                    # a number per sample: unchanged
                    else:
                        for i in range(D):
                            self.assertLessEqual(relerr(y_rescaled[i], y[i] / self.scales[i]), TOL)

    def test_rescale_then_unscale_gives_back_the_probe_values_in_the_original_coordinates(self):
        """The round trip of a fit: rescale the sample and the data; the tensor that fits them, unscaled, has the
        original data at the original sample, an input jet included."""
        rng = self.rng
        for order in ((), (2,), (1, 2)):
            sample = random_sample(rng, 'probe', order)
            y = dense_sample('probe', self.x.to_dense(), sample, order)
            sample_rescaled, y_rescaled = prep.rescale_coordinates('probe', order, sample, y, self.scales)
            # x_rescaled fits the rescaled problem exactly (checked through the frontend's own probe with no jets) ...
            if order == ():
                for a, b in zip(self.x_rescaled.probe(list(sample_rescaled)), y_rescaled):
                    self.assertLessEqual(relerr(a, b), 1e-11)
            for a, b in zip(dense_sample('probe', self.x_rescaled.to_dense(), sample_rescaled, order), y_rescaled):
                self.assertLessEqual(relerr(a, b), 1e-11)
            # ... and unscaled it has the original probe values at the original sample
            unscaled = prep.t3_unscale_coordinates(self.x_rescaled.data, self.scales)
            back = dense_sample('probe', t3.TuckerTensorTrain(*unscaled).to_dense(), sample, order)
            with self.subTest(order=order):
                for a_FX, y_FX in zip(back, y):
                    self.assertLessEqual(relerr(a_FX, y_FX), 1e-11)
                self.assertLessEqual(relerr(t3.TuckerTensorTrain(*unscaled).to_dense(), self.x.to_dense()), 1e-11)

    def test_unscale_is_the_dense_tensor_scaled_along_every_mode(self):
        for C in ((), (2,), (2, 3)):
            x = t3.TuckerTensorTrain.randn(*STRUCTURE, stack_shape=C)
            unscaled = prep.t3_unscale_coordinates(x.data, self.scales)
            s0, s1, s2 = self.scales
            reference = x.to_dense() * s0[:, None, None] * s1[None, :, None] * s2[None, None, :]
            with self.subTest(C=C):
                self.assertEqual((type(unscaled[0]), type(unscaled[1])), (tuple, tuple))
                self.assertEqual([B.shape for B in unscaled[0]], [B.shape for B in x.tucker_cores])     # the ranks stay
                self.assertTrue(all(a is b for a, b in zip(unscaled[1], x.tt_cores)))                   # TT cores as given
                self.assertLessEqual(relerr(t3.TuckerTensorTrain(*unscaled).to_dense(), reference), TOL)

    def test_the_method_equals_the_function(self):
        for C in ((), (2,)):
            x = t3.TuckerTensorTrain.randn(*STRUCTURE, stack_shape=C)
            from_method = x.unscale_coordinates(self.scales)
            from_function = prep.t3_unscale_coordinates(x.data, self.scales)
            with self.subTest(C=C):
                self.assertIsInstance(from_method, t3.TuckerTensorTrain)
                self.assertEqual(from_method.structure, x.structure)
                for a, b in zip(from_method.tucker_cores + from_method.tt_cores, from_function[0] + from_function[1]):
                    self.assertTrue(np.array_equal(a, b))

    def wrong_scales(self):
        """(wrong scales, the message's pattern): each wrong form of ``scales`` for the mode sizes ``SHAPE``."""
        s = self.scales
        return (
            (s[:2], 'one array per mode .*got length 2'),                                # a wrong count
            (s + s[:1], 'one array per mode .*got length 4'),
            (np.ones(3), 'one array per mode .*got a ndarray'),                          # one number per mode
            (np.ones((3, 6)), 'one array per mode .*got a ndarray'),                     # packed to (d, N)
            ([s[0], np.ones(1), s[2]], r'scales\[1\] has shape \(1,\); expected \(N_1,\) = \(5,\)'),   # a length of 1
            ([s[0], np.ones(6), s[2]], r'scales\[1\] has shape \(6,\)'),                # a wrong length for the mode
            ([s[0], 2.0, s[2]], r'scales\[1\] has shape \(\)'),                         # a wrong rank: a number
            ([s[0], np.ones((1, 5)), s[2]], r'scales\[1\] has shape \(1, 5\)'),         # a wrong rank: two axes
            ([s[0], [1.0] * 5, s[2]], r'scales\[1\] has a list'),                       # a list in place of an array
        )

    def test_check_coordinate_scales_passes_good_scales_and_raises_on_each_wrong_form(self):
        s = self.scales
        self.assertIsNone(prep.check_coordinate_scales(s, SHAPE))
        self.assertIsNone(prep.check_coordinate_scales(tuple(s), list(SHAPE)))
        self.assertIsNone(prep.check_coordinate_scales([x.astype(np.float32) for x in s], SHAPE))
        self.assertIsNone(prep.check_coordinate_scales([np.ones(2)], (2,)))               # one mode
        for bad, pattern in self.wrong_scales():
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    prep.check_coordinate_scales(bad, SHAPE)

    def test_check_coordinate_scales_looks_at_no_value(self):
        """Form only: the backend raises on no numerical condition. A zero, a negative, an infinite and a nan
        scale pass, and do what the docstring says: a zero scale zeroes that coordinate's slice of the tensor."""
        s = [x.copy() for x in self.scales]
        s[0][1], s[1][0], s[2][2], s[2][3] = 0.0, -1.0, np.inf, np.nan
        self.assertIsNone(prep.check_coordinate_scales(s, SHAPE))
        zero_scale = [x.copy() for x in self.scales]
        zero_scale[0][1] = 0.0
        dense = t3.TuckerTensorTrain(*prep.t3_unscale_coordinates(self.x.data, zero_scale)).to_dense()
        self.assertFalse(dense[1].any())
        self.assertTrue(dense[0].any())

    def test_the_backend_functions_do_not_check_scales(self):
        """Checkers, not guarantees. What wrong ``scales`` silently do, as the docstrings say: a short list drops
        a mode, a scale of length 1 is broadcast."""
        rng, s = self.rng, self.scales
        ww = random_sample(rng, 'apply', ())
        short, _ = prep.rescale_coordinates('apply', (), ww, np.zeros(7), s[:2])
        self.assertEqual(len(short), 2)                                             # a mode is gone, no error
        self.assertEqual(len(prep.t3_unscale_coordinates(self.x.data, s[:2])[0]), 2)
        broadcast = [s[0], np.array([3.0]), s[2]]
        with self.assertRaises(ValueError):
            prep.check_coordinate_scales(broadcast, SHAPE)
        scaled, _ = prep.rescale_coordinates('apply', (), ww, np.zeros(7), broadcast)
        self.assertLessEqual(relerr(scaled[1], 3.0 * ww[1]), TOL)
        self.assertLessEqual(relerr(prep.t3_unscale_coordinates(self.x.data, broadcast)[0][1],
                                    3.0 * self.x.tucker_cores[1]), TOL)

    def test_the_frontend_rescale_coordinates_checks_the_scales_and_gives_the_backend_s_numbers(self):
        """``fitting.rescale_coordinates`` is the frontend's own function: it raises on each wrong form of
        ``scales`` (a wrong count, a wrong length for a mode, a length of 1, ...), where the backend function drops
        a mode or broadcasts; the check is structural, so it is made in unsafe mode too; on good input it returns
        what the backend function returns."""
        rng = self.rng
        for kind in ('probe', 'apply'):
            for order in ((), (2,), (1, 2)):
                sample, y = random_sample(rng, kind, order), random_data(rng, kind, order)
                with self.subTest(kind=kind, order=order):
                    for bad, pattern in self.wrong_scales():
                        with self.assertRaisesRegex(ValueError, pattern):
                            fitting.rescale_coordinates(kind, order, sample, y, bad)
                        with safety.unsafe():                                       # structural: both modes
                            with self.assertRaisesRegex(ValueError, pattern):
                                fitting.rescale_coordinates(kind, order, sample, y, bad)
                    reference = prep.rescale_coordinates(kind, order, sample, y, self.scales)
                    for mode in ('safe', 'unsafe'):
                        if mode == 'safe':
                            out = fitting.rescale_coordinates(kind, order, sample, y, self.scales)
                        else:
                            with safety.unsafe():
                                out = fitting.rescale_coordinates(kind, order, sample, y, tuple(self.scales))
                        flat = lambda value: sum((flat(v) for v in value), []) if isinstance(value, (tuple, list)) else [value]
                        self.assertEqual(len(flat(out)), len(flat(reference)))
                        for a, b in zip(flat(out), flat(reference)):
                            self.assertTrue(np.array_equal(a, b))
                    if kind == 'apply':
                        self.assertIs(out[1], y)
        # the difference: on the backend function a short list and a length-1 scale give no error
        ww = random_sample(rng, 'apply', ())
        s = self.scales
        for bad in (s[:2], [s[0], np.ones(1), s[2]]):
            prep.rescale_coordinates('apply', (), ww, np.zeros(7), bad)
        # the mode sizes are read off the sample: scales right for another tensor are refused
        with self.assertRaisesRegex(ValueError, r'scales\[0\] has shape \(5,\); expected \(N_0,\) = \(4,\)'):
            fitting.rescale_coordinates('apply', (), ww, np.zeros(7), [np.ones(5), np.ones(5), np.ones(6)])
        # a kind the backend function refuses: its error, not the checker's
        with self.assertRaisesRegex(ValueError, "is for 'probe' and 'apply'"):
            fitting.rescale_coordinates('entries', (), np.zeros((D, 7), dtype=int), np.zeros(7), s[:2])
        with self.assertRaisesRegex(ValueError, 'unknown sampling kind'):
            fitting.rescale_coordinates('probes', (), ww, np.zeros(7), s)
        # it goes through the backend checker
        with unittest.mock.patch.object(prep, 'check_coordinate_scales', side_effect=ValueError('the checker')):
            with self.assertRaisesRegex(ValueError, 'the checker'):
                fitting.rescale_coordinates('apply', (), ww, np.zeros(7), s)

    def test_the_methods_raise_on_scales_of_the_wrong_form(self):
        """Structural, so always an error, on both classes, through the backend checker: one vector of ``Ni``
        scales per mode (O-35)."""
        s = self.scales
        for x in (self.x, ut3.UniformTuckerTensorTrain.from_t3(self.x)):
            with self.subTest(layout=type(x).__name__):
                for bad, pattern in self.wrong_scales():
                    with self.assertRaisesRegex(ValueError, pattern):
                        x.unscale_coordinates(bad)
                self.assertIsNotNone(x.unscale_coordinates(tuple(s)))
                with unittest.mock.patch.object(prep, 'check_coordinate_scales', side_effect=ValueError('the checker')):
                    with self.assertRaisesRegex(ValueError, 'the checker'):     # the method calls the backend checker
                        x.unscale_coordinates(s)

    def test_entries_an_unknown_kind_and_an_integer_order_raise(self):
        index = np.zeros((D, 7), dtype=int)
        with self.assertRaisesRegex(ValueError, "is for 'probe' and 'apply'"):
            prep.rescale_coordinates('entries', (), index, np.zeros(7), self.scales)
        with self.assertRaisesRegex(ValueError, 'unknown sampling kind'):
            prep.rescale_coordinates('probes', (), index, np.zeros(7), self.scales)
        with self.assertRaisesRegex(ValueError, 'the entries kind has no jets'):
            prep.rescale_coordinates('entries', (1,), index, np.zeros(7), self.scales)
        with self.assertRaisesRegex(TypeError, 'order is the jet shape'):
            prep.rescale_coordinates('probe', 0, random_sample(self.rng, 'probe', ()), np.zeros(7), self.scales)
        with self.assertRaisesRegex(TypeError, 'order is the jet shape'):
            fitting.rescale_coordinates('probe', 0, random_sample(self.rng, 'probe', ()), np.zeros(7), self.scales)

    def test_the_result_has_the_lower_precision_of_the_arrays_that_meet(self):
        ww = [np.ones((7, N), dtype=np.float32) for N in SHAPE]
        y = [np.ones((7, N)) for N in SHAPE]
        ww_rescaled, y_rescaled = prep.rescale_coordinates('probe', (), ww, y, self.scales)       # scales: float64
        self.assertEqual({w.dtype for w in ww_rescaled}, {np.dtype(np.float32)})
        self.assertEqual({a.dtype for a in y_rescaled}, {np.dtype(np.float64)})
        scales32 = [s.astype(np.float32) for s in self.scales]
        unscaled = prep.t3_unscale_coordinates(self.x.data, scales32)
        self.assertEqual({B.dtype for B in unscaled[0]}, {np.dtype(np.float32)})


# The `(C, force_pad)` matrix of `docs/contributor/testing_strategy.md`, as in `tests/test_uniform_tucker_tensor_train.py`.
_CONFIGS = [((), False), ((2,), False), ((), True), ((2,), True), ((2, 3), False), ((2, 3), True)]
_PAD = dict(N=9, n=5, r=5)                      # forced-larger padding: every core gets a padded region
_STRUCT_ASYM = ((5, 7, 6), (2, 3, 2), (1, 2, 2, 1))


def _real_indicators(data):
    """1.0 on the real entries of the two supercores and 0.0 on every padded slot: the rank padding (from the
    masks) and the mode padding beyond each ``Ni`` (from ``shape``). Derived here from the masks and the shape."""
    tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask) = data
    d, N = tucker_supercore.shape[0], tucker_supercore.shape[-1]
    mode_mask = np.arange(N)[None, :] < np.asarray(shape)[:, None]                              # (d, N)
    mode_mask = mode_mask.reshape((d,) + (1,) * (tucker_supercore.ndim - 3) + (1, N))
    tucker_real = tucker_edge_mask[..., :, None] & mode_mask
    tt_real = tt_edge_mask[:-1][..., :, None, None] & tucker_edge_mask[..., None, :, None] & tt_edge_mask[1:][..., None, None, :]
    return tucker_real.astype(float), tt_real.astype(float)


def _corrupt(data, rng, scale=1e3):
    """Garbage in EVERY padded slot of both supercores; the real entries untouched."""
    tucker_real, tt_real = _real_indicators(data)
    tucker_supercore, tt_supercore, shape, masks = data
    return (tucker_supercore + scale * (1.0 - tucker_real) * rng.standard_normal(tucker_supercore.shape),
            tt_supercore + scale * (1.0 - tt_real) * rng.standard_normal(tt_supercore.shape), shape, masks)


class TestUniformUnscaleCoordinates(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.rng = np.random.default_rng(9)
        self.scales = [10.0 ** self.rng.uniform(-2.0, 2.0, size=N) for N in _STRUCT_ASYM[0]]

    def cases(self):
        """(label, the ragged tensors of the stack as a flat list, the uniform .data)."""
        for C, force_pad in _CONFIGS:
            x = t3.TuckerTensorTrain.randn(*_STRUCT_ASYM, stack_shape=C)
            data = ut3_conversions.t3_to_ut3(x.data, **(_PAD if force_pad else {}))
            yield 'C=%s, force_pad=%s' % (C, force_pad), x, data
        # a stack whose elements have different ranks (the determinantal variety)
        xa = t3.TuckerTensorTrain.randn((5, 7, 6), (2, 3, 2), (1, 2, 2, 1))
        xb = t3.TuckerTensorTrain.randn((5, 7, 6), (4, 2, 3), (1, 3, 2, 1))
        stacked = ut3.UniformTuckerTensorTrain.stack([ut3.UniformTuckerTensorTrain.from_t3(x, N=8, n=4, r=3)
                                                      for x in (xa, xb)])
        yield 'varying ranks', [xa, xb], stacked.data

    def ragged_dense(self, x):
        """The dense tensor of the ragged twin: ``t3_unscale_coordinates`` on the ragged cores."""
        if isinstance(x, list):
            return np.stack([t3.TuckerTensorTrain(*prep.t3_unscale_coordinates(xi.data, self.scales)).to_dense() for xi in x])
        return t3.TuckerTensorTrain(*prep.t3_unscale_coordinates(x.data, self.scales)).to_dense()

    def test_the_corruption_helper_reaches_only_the_padding(self):
        """The garbage must not touch a real entry (else the garbage test below proves nothing), and must change
        what an UNMASKED read of the supercores sees."""
        for label, x, data in self.cases():
            corrupted = _corrupt(data, self.rng)
            with self.subTest(case=label):
                self.assertLessEqual(relerr(ut3_conversions.ut3_to_dense(corrupted), ut3_conversions.ut3_to_dense(data)), TOL)
                tucker_real, tt_real = _real_indicators(data)
                self.assertTrue(np.array_equal(corrupted[0] * tucker_real, data[0] * tucker_real))
                if (tucker_real == 0.0).any():
                    self.assertFalse(np.array_equal(corrupted[0], data[0]))

    def test_equal_to_ragged_on_real_parts_with_exact_masks_and_garbage_in_the_padding(self):
        for label, x, data in self.cases():
            reference = self.ragged_dense(x)
            for garbage in (False, True):
                given = _corrupt(data, self.rng) if garbage else data
                out = prep.ut3_unscale_coordinates(given, self.scales)
                with self.subTest(case=label, garbage=garbage):
                    # prong 1: the represented tensor is the ragged one
                    self.assertLessEqual(relerr(ut3_conversions.ut3_to_dense(out), reference), TOL)
                    # ... and so are the real entries of the Tucker supercore, entry by entry
                    tucker_real, _ = _real_indicators(data)
                    d, N = data[0].shape[0], data[0].shape[-1]
                    packed = np.zeros((d, N))
                    for i, s in enumerate(self.scales):
                        packed[i, :len(s)] = s
                    wanted = given[0] * packed.reshape((d,) + (1,) * (data[0].ndim - 2) + (N,))
                    self.assertLessEqual(relerr(out[0] * tucker_real, wanted * tucker_real), TOL)
                    # prong 2: exact output structure. The ranks do not change, so the masks and the shape are
                    # the input's, and the masks are host numpy bool arrays
                    self.assertEqual(out[2], data[2])
                    for mask_out, mask_in in zip(out[3], data[3]):
                        self.assertTrue(type(mask_out) is np.ndarray and mask_out.dtype == bool)
                        self.assertTrue(np.array_equal(mask_out, mask_in))
                    self.assertEqual((out[0].shape, out[1].shape), (data[0].shape, data[1].shape))
                    self.assertIs(out[1], given[1])                                     # the TT supercore as given
                    # the mode padding beyond each Ni comes out zero, whatever it held
                    for i, Ni in enumerate(data[2]):
                        self.assertFalse(out[0][i, ..., Ni:].any())

    def test_packed_scales_are_the_ragged_ones(self):
        """``scales`` packed to ``(d, N)`` gives the result of the ragged sequence on the real parts, also with
        garbage beyond each ``Ni`` in the packed scales."""
        for label, x, data in self.cases():
            d, N = data[0].shape[0], data[0].shape[-1]
            packed = self.rng.standard_normal((d, N)) * 1e3                             # garbage everywhere ...
            for i, s in enumerate(self.scales):
                packed[i, :len(s)] = s                                                  # ... but on the real coordinates
            for garbage in (False, True):
                given = _corrupt(data, self.rng) if garbage else data
                out = prep.ut3_unscale_coordinates(given, packed)
                with self.subTest(case=label, garbage=garbage):
                    self.assertLessEqual(relerr(ut3_conversions.ut3_to_dense(out), self.ragged_dense(x)), TOL)
                    # the docstring's statement: the mode padding comes out zero for packed scales too, whatever
                    # the supercore and the packed scales held there; within each Ni every entry, rank padding
                    # included, is the input's times the scale of its coordinate
                    for i, Ni in enumerate(data[2]):
                        self.assertFalse(out[0][i, ..., Ni:].any())
                        self.assertLessEqual(relerr(out[0][i, ..., :Ni], given[0][i, ..., :Ni] * packed[i, :Ni]), TOL)

    def test_the_method_equals_the_function_and_the_ragged_method(self):
        for C, force_pad in _CONFIGS:
            x = t3.TuckerTensorTrain.randn(*_STRUCT_ASYM, stack_shape=C)
            ux = ut3.UniformTuckerTensorTrain.from_t3(x, **(_PAD if force_pad else {}))
            from_method = ux.unscale_coordinates(self.scales)
            from_function = prep.ut3_unscale_coordinates(ux.data, self.scales)
            with self.subTest(C=C, force_pad=force_pad):
                self.assertIsInstance(from_method, ut3.UniformTuckerTensorTrain)
                self.assertTrue(np.array_equal(from_method.tucker_supercore, from_function[0]))
                self.assertTrue(np.array_equal(from_method.tt_supercore, from_function[1]))
                self.assertEqual(from_method.shape, ux.shape)
                for a, b in zip(from_method.masks.data, ux.masks.data):
                    self.assertTrue(np.array_equal(a, b))
                self.assertLessEqual(relerr(from_method.to_dense(), x.unscale_coordinates(self.scales).to_dense()), TOL)


class TestPublicNames(unittest.TestCase):
    """The module's public names, exact. A name dropped from ``__all__`` loses its documentation page with no
    other test failing (that happened to ``ut3_unscale_coordinates`` in the review of this slice)."""

    def test_the_public_names_are_these_seven(self):
        self.assertEqual(
            tuple(prep.__all__),
            (
                'data_block_norms',
                'equalizing_weight',
                'rescale_direction',
                'rescale_coordinates',
                't3_unscale_coordinates',
                'ut3_unscale_coordinates',
                'check_coordinate_scales',
            ),
        )

    def test_every_public_name_is_a_function_of_the_module(self):
        for name in prep.__all__:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(prep, name)))
                self.assertEqual(getattr(prep, name).__module__, prep.__name__)


if __name__ == '__main__':
    unittest.main()
