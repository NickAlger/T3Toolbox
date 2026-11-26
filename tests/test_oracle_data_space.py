# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The least-squares oracle's batch on the data-space rule (the data-metric arc, M2 part 4): ``y`` and the weight are
of the type of ``z``, each axis 1 or ``z``'s size, with the model's stack ``C``.

- the batch's ``check(shape, stack_shape)`` passes the stack to both checkers;
- ``take`` draws ``y`` and the weight against explicit index loops (a weight full on one ``W`` axis only, one constant
  over ``W``; probe ragged, mode-leading and packed);
- ``pack_batch`` packs the weight with ``y`` by one rule, against loops (the join across the modes, an edge of 1 kept or
  broadcast, never padded); finite garbage in the packed padding of ``y`` and the weight changes nothing;
- ``f_resolution`` is ``eps (sqrt(<y, M y>) sqrt(2 f) + |f|)`` with ``<y, M y>`` summed over ``z``'s shape, per element of
  ``C``, against a dense sum (``y`` and the weight written out by ``np.repeat``);
- the named oracles of the M2 design note, at the oracle: **a sample with weight zero and a nonzero residual gives the
  ``f``, ``g`` and ``JtJp`` of the batch without it**; **per-element stacked data give the separate fits**; **the drawn
  ``f`` and ``g`` are unbiased**, by exact enumeration of every subset, with a weight full on one ``W`` axis only and a
  ``y`` constant over ``W``, against the dense misfit (``tests/oracle_sweeps/oracle.py``) and a central difference of it.

Garbage in a padding is finite by the house rule (``pack_vectors``: masking multiplies, so a NaN would stay NaN; that is
not a bug). numpy only (jax dispatch is ``tests/test_dispatch.py``'s).
"""
import dataclasses as dc
import itertools
import math
import os
import sys
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.jets as jets
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
from t3toolbox.backend.uniform_fitting import pack_batch, pack_data

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oracle_sweeps'))
import oracle as dense_oracle                    # noqa: E402  the dense oracle: expand (np.repeat), misfit_full, S_taylor

SHAPE, RANKS = (4, 5, 3), ((2, 3, 2), (1, 2, 2, 1))
NPACK = 6
ORDERS = {'probe': [(), (1,)], 'apply': [(), (1,)], 'entries': [()]}


def leaves(x):
    if isinstance(x, (tuple, list)):
        return [a for item in x for a in leaves(item)]
    return [np.asarray(x)]


def relerr(a, b):
    a, b = leaves(a), leaves(b)
    assert len(a) == len(b), (len(a), len(b))
    num = math.sqrt(sum(float(np.sum((p - q) ** 2)) for p, q in zip(a, b)))
    return num / max(math.sqrt(sum(float(np.sum(q ** 2)) for q in b)), 1e-300)


def z_shapes(kind, order, W, C):
    """The shapes of z's arrays: probe d of them, F+W+C+(N_i,); apply F+W+C; entries W+C."""
    F = tuple(n + 1 for n in order)
    if kind == 'probe':
        return [F + W + C + (N,) for N in SHAPE]
    return [(F + W + C) if kind == 'apply' else (W + C)]


def sample_of(kind, rng, order, W):
    """Input jets with len(order) jet axes of 2 slots (a line), or an index (d,)+W."""
    if kind == 'entries':
        return np.stack([rng.integers(0, N, size=W) for N in SHAPE])
    return [rng.standard_normal((2,) * len(order) + W + (N,)) for N in SHAPE]


def of_type(kind, arrays):
    return list(arrays) if kind == 'probe' else arrays[0]


def patterned(rng, shapes, keeps, positive=False):
    """One array per shape, of size 1 on the axes not kept (a data-space object of the type of z)."""
    out = []
    for shape, keep in zip(shapes, keeps):
        sizes = tuple(n if k else 1 for n, k in zip(shape, keep))
        out.append(0.5 + rng.random(sizes) if positive else rng.standard_normal(sizes))
    return out


def element(kind, value, axis, c):
    """The arrays of a data-space object at element c of its stack axis (an index; size 1 is the same element)."""
    def at(a):
        a = np.asarray(a)
        return np.take(a, c if a.shape[axis] > 1 else 0, axis=axis)
    if value is None:
        return None
    return [at(a) for a in value] if kind == 'probe' else at(value)


def tree_at(value, c):
    """A frame or a point at element c of its stack (every array's first axis)."""
    if isinstance(value, (tuple, list)):
        return tuple(tree_at(v, c) for v in value)
    return np.asarray(value)[c]


class Base(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(29)
        np.random.seed(29)
        self.x, self.target = (t3.TuckerTensorTrain.randn(SHAPE, *RANKS) for _ in range(2))

    def data(self, kind, target, sample, order):
        if kind == 'entries':
            return target.entries(sample)
        return getattr(target, kind)(sample, order)


class TestCheckBatch(Base):

    def test_the_stack_reaches_both_checkers(self):
        """``stack_shape`` is ``C`` for the data and for the weight: a shared ``y`` (size-1 ``C``) and a per-element
        weight pass with it; each of them, full in ``C``, fails without it (by rank), and so does a ``C`` axis of
        another size."""
        C = (2,)
        for kind in ('probe', 'apply', 'entries'):
            order = ORDERS[kind][-1]
            m = len(order)
            W = (3,)
            sample = sample_of(kind, self.rng, order, W)
            full = z_shapes(kind, order, W, C)
            shared_y = of_type(kind, patterned(self.rng, full, [[True] * (m + 1) + [False] + [True] * (len(s) - m - 2)
                                                                for s in full]))
            per_element = of_type(kind, patterned(self.rng, full, [[True] * len(s) for s in full], positive=True))
            unstacked = of_type(kind, patterned(self.rng, z_shapes(kind, order, W, ()),
                                                [[True] * len(s) for s in z_shapes(kind, order, W, ())]))
            oracle = LeastSquaresOracle()
            with self.subTest(kind=kind, case='good'):
                LeastSquaresBatch(kind, order, sample, shared_y, weight=per_element).check(SHAPE, stack_shape=C)
                LeastSquaresBatch(kind, order, sample, per_element, weight=None).check(SHAPE, stack_shape=C)
                LeastSquaresBatch(kind, order, sample, unstacked, weight=None).check(SHAPE)   # default (): unstacked
            three = of_type(kind, patterned(self.rng, z_shapes(kind, order, W, (3,)),
                                            [[True] * len(s) for s in z_shapes(kind, order, W, (3,))], positive=True))
            bad = {'y full in C, no stack_shape':
                       (LeastSquaresBatch(kind, order, sample, per_element, weight=None), ()),
                   'weight full in C, no stack_shape':
                       (LeastSquaresBatch(kind, order, sample, unstacked, weight=per_element), ()),
                   'a stacked y against C = ()':
                       (LeastSquaresBatch(kind, order, sample, shared_y, weight=None), ()),
                   'y with C = 3 against C = 2':
                       (LeastSquaresBatch(kind, order, sample, three, weight=None), C),
                   'weight with C = 3 against C = 2':
                       (LeastSquaresBatch(kind, order, sample, shared_y, weight=three), C),
                   'an unstacked weight against C = 2':
                       (LeastSquaresBatch(kind, order, sample, shared_y, weight=unstacked), C)}
            for label, (batch, stack) in bad.items():
                with self.subTest(kind=kind, case=label):
                    with self.assertRaisesRegex(ValueError, 'of the type of z'):
                        batch.check(SHAPE, stack_shape=stack)


class TestTake(Base):

    W = (2, 3)
    W_KEEPS = [(True, False), (False, True), (False, False), (True, True)]   # full on W's axis 0 only, ..., all full

    def expected(self, array, start, indices):
        """``array`` drawn by an explicit loop: element t of the drawn axis is the sample at flat index indices[t]
        (read at 0 on a W axis of size 1); a W all of size 1 becomes one axis of 1."""
        a = np.asarray(array)
        n_w = len(self.W)
        sizes = a.shape[start:start + n_w]
        lead, tail = (slice(None),) * start, a.shape[start + n_w:]
        if all(s == 1 for s in sizes):
            return a.reshape(a.shape[:start] + (1,) + tail)
        out = []
        for k in indices:
            at = tuple(i if s > 1 else 0 for i, s in zip(np.unravel_index(int(k), self.W), sizes))
            out.append(a[lead + at])
        return np.stack(out, axis=start)

    def test_y_and_the_weight_against_index_loops(self):
        indices = np.array([4, 0, 5, 1])
        for kind in ('probe', 'apply', 'entries'):
            for order in ORDERS[kind]:
                m = len(order)
                sample = sample_of(kind, self.rng, order, self.W)
                full = z_shapes(kind, order, self.W, ())
                oracle = LeastSquaresOracle()
                for wk in self.W_KEEPS:
                    for yk in ((True, True), (False, False)):          # y full, or constant over W
                        rest = lambda s, keepW: [bool(self.rng.integers(2)) for _ in range(m)] + list(keepW) + \
                            [bool(self.rng.integers(2)) for _ in range(len(s) - m - len(self.W))]
                        y = patterned(self.rng, full, [rest(s, yk) for s in full])
                        w = patterned(self.rng, full, [rest(s, wk) for s in full], positive=True)
                        forms = [('ragged', of_type(kind, y), of_type(kind, w), m)]
                        if kind == 'probe':
                            forms.append(('packed', pack_data(kind, order, y, NPACK), pack_data(kind, order, w, NPACK),
                                          1 + m))
                        for form, yv, wv, start in forms:
                            with self.subTest(kind=kind, order=order, weight_W=wk, y_W=yk, form=form):
                                sub = LeastSquaresBatch(kind, order, sample, yv, weight=wv).take(indices)
                                for got, given in ((sub.y, yv), (sub.weight, wv)):
                                    if form == 'ragged' and kind == 'probe':
                                        self.assertIsInstance(got, list)
                                        pairs = zip(got, given)
                                    else:
                                        pairs = [(got, given)]
                                    for g, a in pairs:
                                        e = self.expected(a, start, indices)
                                        self.assertEqual(np.shape(g), e.shape)
                                        self.assertTrue(np.array_equal(g, e))

    def test_a_mode_leading_weight_is_drawn_after_its_mode_axis(self):
        """A probe weight given as one unpacked array, mode first (all its edges 1), per sample on W's axis 1."""
        order = (1,)
        sample = sample_of('probe', self.rng, order, self.W)
        w = 0.5 + self.rng.random((3, 2, 1, 3, 1))                 # (d,) + F + W + (edge,), full on W's axis 1
        sub = LeastSquaresBatch('probe', order, sample, None, weight=w).take(np.array([5, 2]))
        self.assertEqual(sub.weight.shape, (3, 2, 2, 1))
        self.assertTrue(np.array_equal(sub.weight, self.expected(w, 2, [5, 2])))

    def test_an_object_without_the_W_axes_raises(self):
        """A weight that is not of the type of ``z`` (one value per block: its W axes are missing) raises at the draw,
        instead of being read with its jet axes in W's place."""
        sample = sample_of('apply', self.rng, (1,), self.W)
        oracle = LeastSquaresOracle()
        for label, w in (('blocks', np.ones(2)), ('a W of another size', np.ones((2, 4, 3)))):
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, 'not of the type of z'):
                    LeastSquaresBatch('apply', (1,), sample, None, weight=w).take(np.array([0, 1]))


class TestPacking(Base):

    def expected_packed(self, arrays):
        """Packed by loops: each axis of F+W+C the join over the modes; the edge 1 if every mode's is, else N, a full
        edge zero-padded and an edge of 1 repeated to N."""
        rank = arrays[0].ndim
        body = tuple(max(a.shape[k] for a in arrays) for k in range(rank - 1))
        all_one = all(a.shape[-1] == 1 for a in arrays)
        out = np.zeros((len(arrays),) + body + ((1,) if all_one else (NPACK,)))
        for i, a in enumerate(arrays):
            e = a.shape[-1]
            target = body + (e,)
            A = dense_oracle.expand(a, target)
            if all_one or e > 1:
                out[(i,) + (Ellipsis,) + (slice(0, e),)] = A
            else:
                out[i] = np.repeat(A, NPACK, axis=-1)
        return out

    def test_pack_batch_packs_the_weight_by_the_rule_of_y(self):
        order, W, C = (1,), (3,), (2,)
        sample = sample_of('probe', self.rng, order, W)
        full = z_shapes('probe', order, W, C)
        cases = {
            'per entry': [[True] * 5] * 3,
            'per slot (edge 1)': [[True, False, False, False]] * 3,
            'mixed edges and size-1 axes': [[True, True, False, True], [False, True, True, False],
                                            [True, False, True, False]],
        }
        for label, keeps in cases.items():
            keeps = [list(k) + [True] * (len(s) - len(k)) if len(k) < len(s) else list(k)[:len(s)]
                     for k, s in zip(keeps, full)]
            w = patterned(self.rng, full, keeps, positive=True)
            y = patterned(self.rng, full, keeps[::-1])
            with self.subTest(case=label):
                packed = pack_batch(LeastSquaresBatch('probe', order, sample, y, weight=w, minibatch_scale=2.0,
                                                      drawn_indices=np.array([0])), NPACK)
                self.assertTrue(np.array_equal(packed.weight, self.expected_packed(w)))
                self.assertTrue(np.array_equal(packed.y, self.expected_packed(y)))
                self.assertEqual(packed.minibatch_scale, 2.0)
                again = pack_batch(packed, NPACK)               # a packed object is kept as it is
                self.assertIs(again.weight, packed.weight)
                self.assertIs(again.y, packed.y)
        for kind in ('apply', 'entries'):                                       # no mode edge: not packed
            order = ORDERS[kind][-1]
            s = sample_of(kind, self.rng, order, W)
            (shape,) = z_shapes(kind, order, W, C)
            (y,) = patterned(self.rng, [shape], [[True] * len(shape)])
            (w,) = patterned(self.rng, [shape], [[True] * len(shape)], positive=True)
            with self.subTest(kind=kind):
                packed = pack_batch(LeastSquaresBatch(kind, order, s, y, weight=w), NPACK)
                self.assertIs(packed.weight, w)
                self.assertIs(packed.y, y)

    def test_garbage_in_the_packed_padding_changes_nothing(self):
        """Uniform (packed, then finite garbage written into the padding of the sample, ``y`` and the weight) against
        ragged: ``f``, ``f_resolution``, ``r_block_sumsq`` and every real block of ``g``."""
        order, W = (1,), (3,)
        xu = ut3.UniformTuckerTensorTrain.from_t3(self.x, N=NPACK, n=4, r=4)
        sample = sample_of('probe', self.rng, order, W)
        y = self.data('probe', self.target, sample, order)
        full = z_shapes('probe', order, W, ())
        weights = {'per entry': patterned(self.rng, full, [[True] * len(s) for s in full], positive=True),
                   'edge 1, per sample': patterned(self.rng, full, [[False, True, False]] * 3, positive=True),
                   'mixed edges': patterned(self.rng, full, [[True, True, True], [True, False, False],
                                                              [False, True, True]], positive=True)}
        want = ('f', 'f_resolution', 'r_block_sumsq', 'g')
        oracle = LeastSquaresOracle()
        for label, w in weights.items():
            batch = LeastSquaresBatch('probe', order, sample, y, weight=w)
            pb = pack_batch(batch, NPACK)
            garbage = {}
            for name in ('sample', 'y', 'weight'):
                a = np.array(getattr(pb, name))
                if a.shape[-1] == NPACK:
                    for i, N in enumerate(SHAPE):
                        a[i, ..., N:] = 1e3 * (1 + i)
                garbage[name] = a
            pb = dc.replace(pb, **garbage)
            with self.subTest(weight=label):
                rag, _ = oracle.query(self.x.data, batch, want)
                uni, _ = oracle.query(xu.data, pb, want)
                for name, a, c in zip(want[:3], uni[:3], rag[:3]):
                    self.assertLess(relerr(a, c), 1e-12, name)
                gB, gG = uni[3]
                for i, (B, G) in enumerate(zip(*rag[3])):
                    self.assertLess(relerr(gB[i][:B.shape[0], :B.shape[1]], B), 1e-12)
                    self.assertLess(relerr(gG[i][:G.shape[0], :G.shape[1], :G.shape[2]], G), 1e-12)


class TestFResolution(Base):

    def test_against_a_dense_sum_over_zs_shape(self):
        """``<y, M y>`` summed over every entry of ``z`` per element of ``C`` (``y`` and the weight written out to
        ``z``'s shape by ``np.repeat``): ``y`` shared by the stack or constant over the samples counts once per entry
        of ``z``, as ``f`` does. On a manifold frame, a corewise point, and packed on the uniform layout."""
        W = (3,)
        eps = np.finfo(float).eps
        n = 0
        for kind in ('probe', 'apply', 'entries'):
            for order in ORDERS[kind]:
                for C in [(), (2,)]:
                    m = len(order)
                    x = t3.TuckerTensorTrain.randn(SHAPE, *RANKS, stack_shape=C)
                    xu = ut3.UniformTuckerTensorTrain.from_t3(x, N=NPACK, n=4, r=4)
                    sample = sample_of(kind, self.rng, order, W)
                    full = z_shapes(kind, order, W, C)
                    oracle = LeastSquaresOracle()
                    for trial in range(4):
                        ky = [[bool(self.rng.integers(2)) for _ in s] for s in full]
                        kw = [[bool(self.rng.integers(2)) for _ in s] for s in full]
                        if trial == 0:                                        # shared by the stack, constant over W
                            ky = [[True] * m + [False] * (len(s) - m - (kind == 'probe')) + [True] * (kind == 'probe')
                                  for s in full]
                        y = of_type(kind, patterned(self.rng, full, ky))
                        w = None if trial == 1 else of_type(kind, patterned(self.rng, full, kw, positive=True))
                        batch = LeastSquaresBatch(kind, order, sample, y, weight=w)
                        expected_yMy = np.zeros(C)
                        ys = y if kind == 'probe' else [y]
                        ws = [None] * len(ys) if w is None else (w if kind == 'probe' else [w])
                        for c in np.ndindex(*C):
                            total = 0.0
                            for y_i, w_i, s in zip(ys, ws, full):
                                Y = dense_oracle.expand(y_i, s)
                                O = np.ones(s) if w_i is None else dense_oracle.expand(w_i, s)
                                at = (slice(None),) * (m + len(W)) + c
                                total += float(np.sum((O * O * Y * Y)[at]))
                            expected_yMy[c] = total
                        calls = [('frame', t3m.MANIFOLD.frame(x).data, batch), ('point', x.data, batch),
                                 ('uniform point', xu.data, pack_batch(batch, NPACK))]
                        for label, frame, b in calls:
                            with self.subTest(kind=kind, order=order, C=C, trial=trial, call=label):
                                (f, fres), _ = oracle.query(frame, b, ('f', 'f_resolution'))
                                self.assertEqual(np.shape(fres), C)
                                f = np.asarray(f)
                                ref = eps * (np.sqrt(expected_yMy) * np.sqrt(2 * f) + np.abs(f))
                                self.assertLess(float(np.max(np.abs(np.asarray(fres) - ref) / ref)), 1e-12)
                                n += 1
        self.assertEqual(n, 5 * 2 * 4 * 3)

    def test_it_needs_y_even_with_a_held_r(self):
        """``f`` can come from a held ``r`` with no ``y``; ``<y, M y>`` cannot: a ``ValueError`` saying so."""
        sample = sample_of('apply', self.rng, (), (4,))
        batch = LeastSquaresBatch('apply', (), sample, self.data('apply', self.target, sample, ()))
        oracle = LeastSquaresOracle()
        (f,), held = oracle.query(self.x.data, batch, ('f',))
        (f_held,), _ = oracle.query(self.x.data, dc.replace(batch, y=None), ('f',), held=held)
        self.assertEqual(float(f_held), float(f))
        with self.assertRaisesRegex(ValueError, "f_resolution .* needs the batch's data y"):
            oracle.query(self.x.data, dc.replace(batch, y=None), ('f', 'f_resolution'), held=held)


class TestAZeroWeightDeletesASample(Base):

    def test_f_g_and_JtJp_are_those_of_the_batch_without_it(self):
        """A per-sample weight with a zero at sample ``j``, whose residual is not zero: ``f``, ``g`` and ``JtJp`` equal
        those of the batch with sample ``j`` deleted; with weight 1 there, ``f`` is visibly another. Three kinds, on a
        manifold frame (ragged) and the uniform frame (packed)."""
        W, j = (5,), 2
        frame = t3m.MANIFOLD.frame(self.x)
        v = t3m.COREWISE.randn(frame).variations.data
        xu = ut3.UniformTuckerTensorTrain.from_t3(self.x, N=NPACK, n=4, r=4)
        uframe = ut3m.UNIFORM_MANIFOLD.frame(xu)
        uv = ut3m.UNIFORM_COREWISE.randn(uframe).variations.data[:2]
        want = ('f', 'g', 'JtJp')
        for kind in ('probe', 'apply', 'entries'):
            for order in ORDERS[kind]:
                m = len(order)
                sample = sample_of(kind, self.rng, order, W)
                y = self.data(kind, self.target, sample, order)
                full = z_shapes(kind, order, W, ())
                keep = [[False] * m + [True] + [False] * (len(s) - m - 1) for s in full]      # per sample
                w = patterned(self.rng, full, keep, positive=True)
                for a in w:
                    a[(slice(None),) * m + (j,)] = 0.0
                weight = of_type(kind, w)
                rest = np.array([k for k in range(W[0]) if k != j])
                cut = lambda a, start: np.take(a, rest, axis=start)
                if kind == 'entries':
                    sample_cut = sample[:, rest]
                else:
                    sample_cut = [cut(a, m) for a in sample]
                y_cut = [cut(a, m) for a in y] if kind == 'probe' else cut(y, m)
                w_cut = [cut(a, m) for a in w] if kind == 'probe' else cut(weight, m)
                with_zero = LeastSquaresBatch(kind, order, sample, y, weight=weight)
                deleted = LeastSquaresBatch(kind, order, sample_cut, y_cut, weight=w_cut)
                oracle = LeastSquaresOracle()
                # the residual at j is not zero (the deletion is visible)
                (r,), _ = oracle.query(frame.data, with_zero, ('r',))
                r_j = [np.take(a, j, axis=m) for a in r] if kind == 'probe' else np.take(r, j, axis=m)
                self.assertGreater(min(float(np.max(np.abs(a))) for a in leaves(r_j)), 1e-3)
                calls = [('ragged', frame.data, v, with_zero, deleted),
                         ('uniform', uframe.data, uv, pack_batch(with_zero, NPACK),
                          pack_batch(deleted, NPACK))]
                for label, fr, dirs, a_batch, b_batch in calls:
                    with self.subTest(kind=kind, order=order, layout=label):
                        got, _ = oracle.query(fr, a_batch, want, variations=dirs)
                        ref, _ = oracle.query(fr, b_batch, want, variations=dirs)
                        for name, p, q in zip(want, got, ref):
                            self.assertLess(relerr(p, q), 1e-13, name)
                        w_one = [a.copy() for a in w]
                        for a in w_one:
                            a[(slice(None),) * m + (j,)] = 1.0
                        one = LeastSquaresBatch(kind, order, sample, y, weight=of_type(kind, w_one))
                        one = one if label == 'ragged' else pack_batch(one, NPACK)
                        (f_one,), _ = oracle.query(fr, one, ('f',))
                        term_j = 0.5 * sum(float(np.sum(a ** 2)) for a in leaves(r_j))      # weight 1 at j
                        self.assertGreater(term_j, 1e-6 * float(got[0]))
                        self.assertLess(abs(float(f_one) - float(got[0]) - term_j), 1e-9 * term_j)


class TestStackedDataAreSeparateFits(Base):

    def test_each_element_is_its_own_fit(self):
        """``C = (2,)``, ``y`` and the weight full in ``C`` and different per element: ``f``, ``g``, ``JtJp`` and
        ``f_resolution`` element by element equal the unstacked queries at that element's frame and data; another
        element's data give another ``f``. Ragged on the stacked manifold frame; uniform on the stacked point against
        the ragged point."""
        C, W = (2,), (4,)
        x = t3.TuckerTensorTrain.randn(SHAPE, *RANKS, stack_shape=C)
        target = t3.TuckerTensorTrain.randn(SHAPE, *RANKS, stack_shape=C)
        frame = t3m.MANIFOLD.frame(x).data
        v = t3m.COREWISE.randn(t3m.MANIFOLD.frame(x)).variations.data
        xu = ut3.UniformTuckerTensorTrain.from_t3(x, N=NPACK, n=4, r=4)
        want = ('f', 'g', 'JtJp', 'f_resolution')
        for kind in ('probe', 'apply', 'entries'):
            for order in ORDERS[kind]:
                m = len(order)
                axis = m + len(W)                                       # the C axis of each array
                sample = sample_of(kind, self.rng, order, W)
                y = self.data(kind, target, sample, order)              # F+W+C(+(N_i,)): one dataset per element
                full = z_shapes(kind, order, W, C)
                weight = of_type(kind, patterned(self.rng, full, [[True] * len(s) for s in full], positive=True))
                oracle = LeastSquaresOracle()
                stacked, _ = oracle.query(frame, LeastSquaresBatch(kind, order, sample, y, weight=weight), want,
                                          variations=v)
                f_elements = []
                for c in range(C[0]):
                    with self.subTest(kind=kind, order=order, element=c):
                        b_c = LeastSquaresBatch(kind, order, sample, element(kind, y, axis, c),
                                                weight=element(kind, weight, axis, c))
                        single, _ = oracle.query(tree_at(frame, c), b_c, want, variations=tree_at(v, c))
                        f_elements.append(float(single[0]))
                        self.assertLess(abs(float(stacked[0][c]) - float(single[0])), 1e-13 * float(single[0]))
                        self.assertLess(relerr(tree_at(stacked[1], c), single[1]), 1e-13)
                        self.assertLess(relerr(tree_at(stacked[2], c), single[2]), 1e-13)
                        self.assertLess(abs(float(stacked[3][c]) - float(single[3])), 1e-13 * float(single[3]))
                        # the other element's data give another f (a mix-up of the elements would show)
                        other = LeastSquaresBatch(kind, order, sample, element(kind, y, axis, 1 - c),
                                                  weight=element(kind, weight, axis, 1 - c))
                        (f_other,), _ = oracle.query(tree_at(frame, c), other, ('f',))
                        self.assertGreater(abs(float(f_other) - float(single[0])), 1e-3 * float(single[0]))
                with self.subTest(kind=kind, order=order, layout='uniform'):
                    batch = LeastSquaresBatch(kind, order, sample, y, weight=weight)
                    (f_u, g_u, fres_u), _ = oracle.query(xu.data, pack_batch(batch, NPACK),
                                                         ('f', 'g', 'f_resolution'))
                    (f_r, g_r, fres_r), _ = oracle.query(x.data, batch, ('f', 'g', 'f_resolution'))
                    self.assertLess(relerr(f_u, f_r), 1e-12)
                    self.assertLess(relerr(fres_u, fres_r), 1e-12)
                    for i, (B, G) in enumerate(zip(*g_r)):
                        self.assertLess(relerr(g_u[0][i][..., :B.shape[-2], :B.shape[-1]], B), 1e-12)
                        self.assertLess(relerr(g_u[1][i][..., :G.shape[-3], :G.shape[-2], :G.shape[-1]], G), 1e-12)


class TestUnbiasedOverEverySubset(Base):

    def test_with_a_partly_broadcast_weight(self):
        """``W = (2, 3)``, the weight full on ``W``'s axis 0 only (and per slot), ``y`` constant over ``W`` or full:
        over every subset of two of the six samples (``take``), the mean of the drawn ``f`` and ``g`` is the full
        batch's, and the full ``f`` is the dense misfit (``misfit_full``) while ``<g, p>`` is its central difference
        along a direction ``p`` in one Tucker core (exact: the misfit is quadratic along it). At the corewise point,
        whose ``g`` is the gradient in the cores."""
        W = (2, 3)
        n_w = len(W)
        for kind in ('probe', 'apply', 'entries'):
            for order in ORDERS[kind]:
                m = len(order)
                sample = sample_of(kind, self.rng, order, W)
                pp = None if not order else [self.rng.standard_normal(W + (N,)) for N in SHAPE]
                if order:
                    main = [s[0] for s in sample]                          # the line main + s pp, as input jets
                    sample = jets.affine_jet(main, pp)
                    dense_sample = (main, pp)
                else:
                    dense_sample = sample
                full = z_shapes(kind, order, W, ())
                w = of_type(kind, patterned(self.rng, full, [[True] * m + [True, False] + [False] * (kind == 'probe')
                                                             for _ in full], positive=True))
                z_target = self.data(kind, self.target, sample, order)
                ys = {'y full': z_target,
                      'y constant over W': of_type(kind, patterned(self.rng, full,
                                                                   [[True] * m + [False] * n_w + [True] * (kind == 'probe')
                                                                    for _ in full]))}
                oracle = LeastSquaresOracle()
                X = self.x.to_dense()
                for label, y in ys.items():
                    with self.subTest(kind=kind, order=order, y=label):
                        batch = LeastSquaresBatch(kind, order, sample, y, weight=w)
                        (f, g), _ = oracle.query(self.x.data, batch, ('f', 'g'))
                        f_dense = dense_oracle.misfit_full(kind, X, dense_sample, y, w, order)
                        self.assertLess(abs(float(f) - f_dense), 1e-12 * f_dense)
                        # <g, p> along one Tucker core: (f(1) - f(-1)) / 2, the misfit quadratic in t
                        dB = self.rng.standard_normal(self.x.data[0][1].shape)
                        def f_at(t):
                            B = list(self.x.data[0])
                            B[1] = B[1] + t * dB
                            Xt = t3.TuckerTensorTrain(tuple(B), self.x.data[1]).to_dense()
                            return dense_oracle.misfit_full(kind, Xt, dense_sample, y, w, order)
                        self.assertLess(abs(float(np.sum(g[0][1] * dB)) - (f_at(1.0) - f_at(-1.0)) / 2),
                                        1e-10 * max(f_dense, 1.0))
                        fs, gs = [], []
                        for subset in itertools.combinations(range(math.prod(W)), 2):
                            (fb, gb), _ = oracle.query(self.x.data, batch.take(np.array(subset)), ('f', 'g'))
                            fs.append(float(fb))
                            gs.append(leaves(gb))
                        self.assertEqual(len(fs), 15)
                        self.assertLess(abs(np.mean(fs) - float(f)) / float(f), 1e-13)
                        mean_g = [np.mean([g_[k] for g_ in gs], axis=0) for k in range(len(gs[0]))]
                        self.assertLess(relerr(mean_g, leaves(g)), 1e-13)


if __name__ == '__main__':
    unittest.main()
