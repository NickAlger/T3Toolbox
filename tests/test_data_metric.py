# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The data metric ``M: Y* -> Y`` (``backend.jets.data_metric``), its weight from blocks (``blocks_to_weight``) and the
batch's method (``LeastSquaresBatch.data_metric``): graded messages, the data-metric sub-arc, M1.

1. **Values** (`TestValues`): against an explicit loop over every index, ``gamma * omega[k]**2 * r[k]`` with the
   weight read at index 0 on its axes of size 1 (no broadcasting in the oracle). Probe ragged (the weight a tuple, or
   one array whose first axis is the mode), probe packed, apply, entries; the stack ``C`` empty or not; the sizes of
   ``F``, ``W``, ``C`` distinct, and all equal (where a transposed axis cannot fail by shape); weights per slot, per
   sample, per entry, per mode, with a ``C`` of 1; the scale ``None``, a number, a 0-d array.
2. **What raises, and what does not** (`TestStructure`): a weight of another rank (a residual with a stack of
   directions ``K``); with no weight, ``gamma r`` for any ``r``, ``K`` included; a weight axis that is neither 1 nor
   ``r``'s size (a weight never grows ``r``); a ragged ``r`` with a weight of another mode count, or a mode axis of 1;
   a packed ``r`` with a ragged weight; a scale that is not a scalar; a missing argument. The structure of the result.
3. **The precision** (`TestPrecision`): the lower floating precision of ``r`` and the weight; a float64 scale does not
   raise a float32 result.
4. **``blocks_to_weight``** (`TestBlocksToWeight`): the placement against arrays filled by loops, ragged and packed,
   with and without ``C``; through ``data_metric`` with no scale it gives the old block weighting
   (``_squared_weights``, the divisor and factor of the dual-jet converters) **bitwise**, the bridge of M2. With a
   scale the two agree only to rounding (``(gamma omega**2) r`` against a scale applied to the sums), which is not
   tested here. Wrong blocks raise.
5. **The batch method** (`TestBatchMethod`): ``data_metric(r, batch.weight, batch.minibatch_scale)``; a batch whose
   weight is one value per block raises by rank (until the batch takes the full form).

numpy only, by the house convention; jax and jit are in ``test_dispatch.py``.
"""
import inspect
import itertools
import unittest

import numpy as np

import t3toolbox.backend.jets as jets
from t3toolbox.backend.fitting import LeastSquaresBatch

TOL = 1e-15


def metric_loop(r, omega, gamma):
    """The definition, by an explicit loop: ``gamma * omega[k]**2 * r[k]`` with omega read at 0 on its size-1 axes."""
    r = np.asarray(r, dtype=float)
    out = np.empty(r.shape)
    for idx in np.ndindex(*r.shape):
        if omega is None:
            w2 = 1.0
        else:
            w = np.asarray(omega, dtype=float)
            w2 = w[tuple(0 if w.shape[a] == 1 else idx[a] for a in range(r.ndim))] ** 2
        out[idx] = (1.0 if gamma is None else float(gamma)) * w2 * r[idx]
    return out


def relerr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


# (m-tuple F, W, C, N per mode): sizes distinct, and all equal
SIZES = {'distinct': ((3,), (4,), (2,), (5, 6, 7)),
         'equal': ((2,), (2,), (2,), (2, 2, 2)),
         'no stack': ((2, 3), (4,), (), (5, 6, 7))}


def weight_forms(rng, F, W, C, edge):
    """Full-form weights of one array of shape F+W+C+edge: per slot, per sample, per entry, per mode (all 1), with a
    C of 1."""
    m, w, c = len(F), len(W), len(C)
    forms = {'per slot': F + (1,) * (w + c) + (1,) * len(edge),
             'per sample': (1,) * m + W + (1,) * c + (1,) * len(edge),
             'per entry': F + W + C + edge,
             'one number': (1,) * (m + w + c + len(edge)),
             'C of 1': F + W + (1,) * c + edge}
    return {name: 0.5 + rng.random(shape) for name, shape in forms.items()}


class TestValues(unittest.TestCase):

    def test_apply_and_entries(self):
        rng = np.random.default_rng(1)
        for (label, (F, W, C, _)), gamma in itertools.product(SIZES.items(), (None, 2.5, np.asarray(0.25))):
            r_apply = rng.standard_normal(F + W + C)
            for name, omega in weight_forms(rng, F, W, C, ()).items():
                with self.subTest(kind='apply', sizes=label, weight=name, gamma=repr(gamma)):
                    self.assertLessEqual(relerr(jets.data_metric(r_apply, omega, gamma),
                                                metric_loop(r_apply, omega, gamma)), TOL)
            r_entries = rng.standard_normal(W + C)
            for name, omega in weight_forms(rng, (), W, C, ()).items():
                with self.subTest(kind='entries', sizes=label, weight=name, gamma=repr(gamma)):
                    self.assertLessEqual(relerr(jets.data_metric(r_entries, omega, gamma),
                                                metric_loop(r_entries, omega, gamma)), TOL)

    def test_probe_ragged_with_a_tuple_and_with_an_array_weight(self):
        rng = np.random.default_rng(2)
        for (label, (F, W, C, N)), gamma in itertools.product(SIZES.items(), (None, 2.5)):
            r = tuple(rng.standard_normal(F + W + C + (n,)) for n in N)
            per_mode = [weight_forms(rng, F, W, C, (n,)) for n in N]
            for name in per_mode[0]:
                omega = tuple(forms[name] for forms in per_mode)
                with self.subTest(sizes=label, weight=name + ' (a tuple)', gamma=repr(gamma)):
                    got = jets.data_metric(r, omega, gamma)
                    self.assertIs(type(got), tuple)
                    for got_i, r_i, w_i in zip(got, r, omega):
                        self.assertLessEqual(relerr(got_i, metric_loop(r_i, w_i, gamma)), TOL)
            # one array, the mode first: the edge 1 (the modes differ in it), and per mode and slot
            omega = 0.5 + rng.random((len(N),) + F + (1,) * (len(W) + len(C)) + (1,))
            with self.subTest(sizes=label, weight='a d-leading array', gamma=repr(gamma)):
                got = jets.data_metric(list(r), omega, gamma)              # a list in gives a tuple
                self.assertIs(type(got), tuple)
                for i, (got_i, r_i) in enumerate(zip(got, r)):
                    self.assertLessEqual(relerr(got_i, metric_loop(r_i, omega[i], gamma)), TOL)

    def test_probe_packed(self):
        """A packed ``r`` pairs with a packed weight elementwise, its padding included (masking is the engine's)."""
        rng = np.random.default_rng(3)
        for label, (F, W, C, N) in SIZES.items():
            d, n_pad = len(N), max(N)
            r = rng.standard_normal((d,) + F + W + C + (n_pad,))
            for name, omega in {'full edge': 0.5 + rng.random((d,) + F + W + C + (n_pad,)),
                                'per mode and slot': 0.5 + rng.random((d,) + F + (1,) * (len(W) + len(C) + 1)),
                                'a mode axis of 1 (broadcast: the documented limit)':
                                    0.5 + rng.random((1,) + F + (1,) * (len(W) + len(C)) + (n_pad,))}.items():
                with self.subTest(sizes=label, weight=name):
                    got = jets.data_metric(r, omega, 3.0)
                    self.assertIs(type(got), np.ndarray)
                    self.assertLessEqual(relerr(got, metric_loop(r, omega, 3.0)), TOL)


class TestStructure(unittest.TestCase):

    def test_a_weight_of_another_rank_raises_and_no_weight_takes_any_r(self):
        rng = np.random.default_rng(4)
        F, W, K, C = (3,), (4,), (2,), (2,)
        r_K = rng.standard_normal(F + W + K + C)                       # apply, with a stack of directions
        omega = 0.5 + rng.random(F + W + C)
        with self.assertRaisesRegex(ValueError, 'rank'):
            jets.data_metric(r_K, omega, None)
        with self.assertRaisesRegex(ValueError, 'rank'):
            jets.data_metric(tuple(rng.standard_normal(F + W + K + C + (n,)) for n in (5, 6)),
                             tuple(0.5 + rng.random(F + W + C + (n,)) for n in (5, 6)), None)
        self.assertTrue(np.array_equal(jets.data_metric(r_K, None, 2.0), 2.0 * r_K))
        self.assertTrue(np.array_equal(jets.data_metric(r_K, None, None), r_K))

    def test_a_weight_never_grows_r(self):
        """An ``r`` of one sample and a weight of four: equal rank, and broadcasting would return four samples."""
        rng = np.random.default_rng(5)
        r = rng.standard_normal((3, 1, 2))                             # apply: F + W + C, W = (1,)
        with self.assertRaisesRegex(ValueError, 'axis 1'):
            jets.data_metric(r, 0.5 + rng.random((3, 4, 2)), None)
        with self.assertRaisesRegex(ValueError, 'axis 2'):           # a C of 1 against a weight with a C of 2
            jets.data_metric(rng.standard_normal((3, 4, 1)), np.ones((1, 1, 2)), None)
        with self.assertRaisesRegex(ValueError, 'mode 1'):           # probe: named by its mode
            jets.data_metric((np.ones((3, 4, 5)), np.ones((3, 1, 6))), (np.ones((3, 1, 5)), np.ones((3, 4, 6))), None)

    def test_the_mode_count_of_a_ragged_r_is_exact(self):
        rng = np.random.default_rng(6)
        r = tuple(rng.standard_normal((3, 4, n)) for n in (5, 6, 7))
        for label, omega in (('a tuple of 2', (np.ones((3, 1, 1)),) * 2),
                             ('an array of 2 modes', np.ones((2, 3, 1, 1))),
                             ('an array with a mode axis of 1', np.ones((1, 3, 1, 1))),
                             ('a 0-d array', np.asarray(2.0))):
            with self.subTest(weight=label):
                with self.assertRaisesRegex(ValueError, 'modes'):
                    jets.data_metric(r, omega, None)

    def test_a_packed_r_with_a_ragged_weight_raises(self):
        with self.assertRaisesRegex(ValueError, 'pack_batch'):
            jets.data_metric(np.ones((3, 2, 4, 7)), tuple(np.ones((2, 1, 1)) for _ in range(3)), None)

    def test_the_scale_is_a_scalar(self):
        for scale in (np.ones(4), np.ones((1,)), [2.0]):
            with self.subTest(scale=repr(scale)):
                with self.assertRaisesRegex(ValueError, 'scale'):
                    jets.data_metric(np.ones((3, 4)), None, scale)

    def test_both_factors_are_required(self):
        parameters = inspect.signature(jets.data_metric).parameters
        self.assertEqual(['r', 'weight', 'minibatch_scale'], list(parameters))
        self.assertTrue(all(p.default is inspect.Parameter.empty for p in parameters.values()))
        with self.assertRaises(TypeError):
            jets.data_metric(np.ones(3), None)

    def test_the_result_has_r_s_shapes_and_structure(self):
        rng = np.random.default_rng(7)
        r = [rng.standard_normal((2, 3, n)) for n in (4, 5)]
        out = jets.data_metric(r, (np.ones((2, 1, 1)), np.ones((2, 3, 5))), 1.5)
        self.assertEqual((type(out), [o.shape for o in out]), (tuple, [(2, 3, 4), (2, 3, 5)]))
        self.assertIs(type(jets.data_metric(np.asarray(2.0), np.asarray(3.0), None)), np.ndarray)


class TestPrecision(unittest.TestCase):

    def test_the_lower_precision_and_a_scale_that_does_not_raise_it(self):
        r64, omega64 = np.ones((3, 4)), np.full((3, 1), 2.0)
        for r_dtype, w_dtype in itertools.product((np.float32, np.float64), repeat=2):
            expected = np.float32 if np.float32 in (r_dtype, w_dtype) else np.float64
            for scale in (None, 2.0, np.asarray(2.0), np.asarray(2.0, dtype=np.float32)):
                with self.subTest(r=r_dtype.__name__, weight=w_dtype.__name__, scale=repr(scale)):
                    out = jets.data_metric(r64.astype(r_dtype), omega64.astype(w_dtype), scale)
                    self.assertEqual(out.dtype, expected)
            self.assertEqual(jets.data_metric(r64.astype(r_dtype), None, np.asarray(2.0)).dtype, r_dtype)


class TestBlocksToWeight(unittest.TestCase):

    CASES = (((2,), (4,), (2,)), ((1, 2), (3,), ()), ((), (5,), (2, 3)))      # (order, W, C)
    N = (4, 5, 6)

    def _y(self, rng, kind, order, W, C, packed=False):
        F = tuple(n + 1 for n in order)
        if kind == 'probe':
            if packed:
                return rng.standard_normal((len(self.N),) + F + W + C + (max(self.N),))
            return tuple(rng.standard_normal(F + W + C + (n,)) for n in self.N)
        return rng.standard_normal(F + W + C) if kind == 'apply' else rng.standard_normal(W + C)

    def _blocks(self, rng, kind, order):
        F = tuple(n + 1 for n in order)
        return 0.5 + rng.random(((len(self.N),) + F) if kind == 'probe' else F if kind == 'apply' else ())

    def test_the_placement_against_a_loop(self):
        rng = np.random.default_rng(8)
        for (order, W, C), kind, packed in itertools.product(self.CASES, ('probe', 'apply', 'entries'), (False, True)):
            if (kind == 'entries' and order) or (packed and kind != 'probe'):
                continue
            m = len(order)
            y, blocks = self._y(rng, kind, order, W, C, packed), self._blocks(rng, kind, order)
            with self.subTest(kind=kind, order=order, W=W, C=C, packed=packed):
                weight = jets.blocks_to_weight(kind, order, blocks, y)
                arrays = (list(y) if kind == 'probe' else [y])
                for i, y_i in enumerate(arrays):
                    w_i = weight[i] if kind == 'probe' else weight
                    self.assertEqual(w_i.ndim, np.ndim(y_i))
                    expected = np.empty(np.shape(y_i))
                    for idx in np.ndindex(*np.shape(y_i)):             # the block of the slot, on every entry
                        block = blocks[i] if kind == 'probe' else blocks
                        expected[idx] = block[idx[:m]] if m else block
                    self.assertTrue(np.array_equal(np.broadcast_to(w_i, np.shape(y_i)), expected))
                    self.assertTrue(all(a == 1 for a in w_i.shape[m:]))  # size 1 off the jet axes

    def test_through_the_metric_it_is_the_old_block_weighting_bitwise(self):
        """With no scale, ``data_metric(r, blocks_to_weight(...), None)`` is the old block weighting exactly, the seed
        ``jet_to_dual_jet`` makes before its turn: the same cast to the lower precision, the same square, one
        multiply. In every pairing of float32 and float64 (a square formed in the higher precision and cast after
        differs in the last bit). (With a scale the two agree only to rounding.)"""
        rng = np.random.default_rng(9)
        dtypes = ((np.float64, np.float64), (np.float32, np.float64), (np.float64, np.float32), (np.float32, np.float32))
        for (order, W, C), kind, packed, (r_dtype, w_dtype) in itertools.product(
                self.CASES, ('probe', 'apply', 'entries'), (False, True), dtypes):
            if (kind == 'entries' and order) or (packed and kind != 'probe'):
                continue
            r, blocks = self._y(rng, kind, order, W, C, packed), self._blocks(rng, kind, order).astype(w_dtype)
            r = tuple(r_i.astype(r_dtype) for r_i in r) if isinstance(r, tuple) else r.astype(r_dtype)
            with self.subTest(kind=kind, order=order, W=W, C=C, packed=packed, r=r_dtype.__name__,
                              weight=w_dtype.__name__):
                new = jets.data_metric(r, jets.blocks_to_weight(kind, order, blocks, r), None)
                old = jets.jet_to_dual_jet(kind, order, r, blocks)                   # the weights, then the turn
                new = list(new) if kind == 'probe' else [new]
                old = [jets._flipped(o, len(order)) for o in (old if kind == 'probe' else [old])]   # the turn undone
                for a, b in zip(new, old):
                    self.assertEqual(a.dtype, b.dtype)
                    self.assertTrue(np.array_equal(a, b))

    def test_wrong_blocks_raise(self):
        rng = np.random.default_rng(10)
        y = tuple(rng.standard_normal((3, 4, n)) for n in self.N)
        for label, call in (('probe blocks of the apply shape', lambda: jets.blocks_to_weight('probe', (2,), np.ones(3), y)),
                            ('probe blocks of another mode count', lambda: jets.blocks_to_weight('probe', (2,), np.ones((2, 3)), y)),
                            ('apply blocks of another F', lambda: jets.blocks_to_weight('apply', (2,), np.ones(2), y[0])),
                            ('entries blocks not one number', lambda: jets.blocks_to_weight('entries', (), np.ones(1), y[0])),
                            ('y of too low a rank', lambda: jets.blocks_to_weight('apply', (1, 2), np.ones((2, 3)), np.ones(2))),
                            ('ragged modes of different ranks',
                             lambda: jets.blocks_to_weight('probe', (2,), np.ones((2, 3)), (np.ones((3, 4, 5)), np.ones((3, 5))))),
                            ('entries with jets', lambda: jets.blocks_to_weight('entries', (2,), np.ones(3), y[0]))):
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    call()


class TestBatchMethod(unittest.TestCase):

    def test_it_is_data_metric_with_the_batch_s_weight_and_scale(self):
        rng = np.random.default_rng(11)
        y = tuple(rng.standard_normal((3, 4, n)) for n in (5, 6))
        r = tuple(rng.standard_normal((3, 4, n)) for n in (5, 6))
        weight = jets.blocks_to_weight('probe', (2,), 0.5 + rng.random((2, 3)), y)
        for scale in (None, 2.0):
            batch = LeastSquaresBatch(None, y, weight, scale)
            with self.subTest(scale=repr(scale)):
                got, wanted = batch.data_metric(r), jets.data_metric(r, weight, scale)
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(got, wanted)))
                for a, r_i, w_i in zip(got, r, weight):
                    self.assertLessEqual(relerr(a, metric_loop(r_i, w_i, scale)), TOL)

    def test_a_batch_with_a_weight_per_block_raises_by_rank(self):
        """Until the batch takes the full form, a weight of one value per block is not of the type of ``z``: it raises,
        and is never read as a full-form weight. (The forms coincide only with no sample or stack axes.)"""
        rng = np.random.default_rng(12)
        y_apply, y_probe = rng.standard_normal((3, 4)), tuple(rng.standard_normal((3, 4, n)) for n in (5, 6))
        for label, batch, r in (('apply', LeastSquaresBatch(None, y_apply, 0.5 + rng.random(3)), y_apply),
                                ('probe', LeastSquaresBatch(None, y_probe, 0.5 + rng.random((2, 3))), y_probe),
                                ('entries', LeastSquaresBatch(None, y_apply[0], np.asarray(0.5)), y_apply[0])):
            with self.subTest(kind=label):
                with self.assertRaisesRegex(ValueError, 'rank'):
                    batch.data_metric(r)


if __name__ == '__main__':
    unittest.main()
