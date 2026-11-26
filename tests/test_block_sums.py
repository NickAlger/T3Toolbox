# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The block sums on the data metric (the data-metric arc, M3): ``r_block_sumsq`` and ``y_block_sumsq`` are
``<r, M r>`` and ``<y, M y>`` per block, over ``z``'s shape, the scale inside ``M``.

- **Against dense sums**: the engine's ``r_block_sumsq`` and the oracle's two sums (manifold frame, corewise point,
  uniform packed with finite garbage in the padding of ``y`` and the weight, edges of 1 and ``N``) against loops over
  the blocks of ``omega * (z - y)`` and ``omega * y`` written out to ``z``'s shape by ``np.repeat``
  (``tests/oracle_sweeps/oracle.py``: ``weighted_residual_full``, ``weighted_full``, ``block_sumsq_full``), with ``z``
  per stack element from the reference module (``sampling_definitions``, the dense definitions); every kind, the jet
  shapes ``()``, ``(2,)``, ``(1, 1)``; weights per entry with zeros, per sample with a zero, constant per block, none;
  ``y`` full, shared by the stack, shared by the samples, and (probe) constant over the mode edge.
- **Data shared by the stack or the samples** count once per entry of ``z``: ``y_block_sumsq`` is ``|C|`` (or ``|W|``)
  times one copy's (the display's ratio was off by ``sqrt |C|`` before M3).
- **No weight is the sum of squares**: ``weight=None``, full-form ones and ``blocks_to_weight`` of ones give ``Σ r²``
  bitwise; ``y_block_sumsq`` unweighted on a full ``y`` is ``data_block_norms(y)**2``.
- **The scale**: on a drawn batch both sums are ``n / b`` times the dense sums over the drawn samples, and their ratio
  is the plain subset's; the law ``1/2 Σ_b r_block_sumsq = Σ_C f`` (two routes of the engine, a consistency check).

Garbage in a padding is finite by the house rule (``pack_vectors``: masking multiplies). numpy only (jax dispatch is
``tests/test_dispatch.py``'s).
"""
import dataclasses as dc
import math
import os
import sys
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.sampling_definitions as sdef
import t3toolbox.backend.data_preparation as data_preparation
from t3toolbox.backend.jets import blocks_to_weight
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
from t3toolbox.backend.uniform_fitting import pack_batch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oracle_sweeps'))
import oracle as dense_oracle                    # noqa: E402  expand (np.repeat), weighted_full, block_sumsq_full

SHAPE, RANKS = (4, 5, 3), ((2, 3, 2), (1, 2, 2, 1))
NPACK = 6
W = (3,)
CASES = [('probe', ()), ('probe', (2,)), ('probe', (1, 1)), ('apply', ()), ('apply', (2,)), ('apply', (1, 1)),
         ('entries', ())]


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
    F = tuple(n + 1 for n in order)
    if kind == 'probe':
        return [F + W + C + (N,) for N in SHAPE]
    return [(F + W + C) if kind == 'apply' else (W + C)]


def sample_of(kind, rng, order, W):
    """Input jets of 2 slots on every jet axis, or an index (d,)+W."""
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


def at_element(kind, value, axis, c):
    """The arrays of a data-space object at element c of its stack axis (size 1: the same element)."""
    def at(a):
        a = np.asarray(a)
        return np.take(a, c if a.shape[axis] > 1 else 0, axis=axis)
    if value is None:
        return None
    return [at(a) for a in value] if kind == 'probe' else at(value)


def tree_at(value, c):
    if isinstance(value, (tuple, list)):
        return tuple(tree_at(v, c) for v in value)
    return np.asarray(value)[c]


def y_forms(kind, order, C):
    """Which axes of y are full (True) or 1: full; shared by the stack; shared by the samples; (probe) the edge 1."""
    full = z_shapes(kind, order, W, C)
    m, nW, nC = len(order), len(W), len(C)
    def keep(jets, w, c, edge):
        return [[jets] * m + [w] * nW + [c] * nC + ([edge] if kind == 'probe' else []) for _ in full]
    forms = {'full': keep(True, True, True, True), 'shared by W': keep(True, False, True, True)}
    if C:
        forms['shared by C'] = keep(True, True, False, True)
    if kind == 'probe':
        forms['edge 1'] = keep(True, True, True, False)
    return forms


def weight_forms(rng, kind, order, C):
    """None; per entry with zeros; per sample with a zero (W full, the rest 1); constant per block."""
    full = z_shapes(kind, order, W, C)
    m = len(order)
    per_entry = patterned(rng, full, [[True] * len(s) for s in full], positive=True)
    for a in per_entry:
        a.reshape(-1)[::4] = 0.0
    per_sample = patterned(rng, full, [[False] * m + [True] * len(W) + [False] * (len(s) - m - len(W)) for s in full],
                           positive=True)
    for a in per_sample:
        a[(slice(None),) * m + (1,)] = 0.0
    blocks = 0.5 + rng.random(((len(SHAPE),) if kind == 'probe' else ()) + tuple(n + 1 for n in order))
    y_like = of_type(kind, [np.zeros(s) for s in full])
    return {'none': None, 'per entry, zeros': of_type(kind, per_entry), 'per sample, a zero': of_type(kind, per_sample),
            'per block': blocks_to_weight(kind, order, blocks, y_like)}


def dense_sums(kind, order, frame_at, sample, y, weight, C):
    """The two block sums by the dense oracle: summed over the stack elements of loops over the blocks of
    ``omega * (z_c - y_c)`` and ``omega * y_c``, each written out to ``z_c``'s shape by ``np.repeat``."""
    m = len(order)
    axis = m + len(W)                                      # the stack axis C (one axis here) of every data array
    r_sums = y_sums = 0.0
    for c in (np.ndindex(*C) if C else [()]):
        z_c = sdef.z(kind, frame_at(c), sample, order=order)
        z_c = list(z_c) if kind == 'probe' else z_c
        y_c = at_element(kind, y, axis, c[0]) if C else y
        w_c = at_element(kind, weight, axis, c[0]) if C else weight
        r_sums = r_sums + dense_oracle.block_sumsq_full(kind, dense_oracle.weighted_residual_full(kind, z_c, y_c, w_c),
                                                        m)
        y_sums = y_sums + dense_oracle.block_sumsq_full(kind, dense_oracle.weighted_full(kind, z_c, y_c, w_c), m)
    return r_sums, y_sums


def with_garbage(pb):
    """A packed probe batch with finite garbage in the padding of every array whose edge is N."""
    garbage = {}
    for name in ('sample', 'y', 'weight'):
        value = getattr(pb, name)
        if value is None:
            continue
        a = np.array(value)
        if a.shape[-1] == NPACK:
            for i, N in enumerate(SHAPE):
                a[i, ..., N:] = 1e3 * (1 + i)
        garbage[name] = a
    return dc.replace(pb, **garbage)


class TestAgainstDenseSums(unittest.TestCase):

    def test_engine_and_oracle(self):
        rng = np.random.default_rng(31)
        np.random.seed(31)
        n = 0
        for kind, order in CASES:
            for C in [(), (2,)]:
                x = t3.TuckerTensorTrain.randn(SHAPE, *RANKS, stack_shape=C)
                frame = t3m.MANIFOLD.frame(x).data
                xu = ut3.UniformTuckerTensorTrain.from_t3(x, N=NPACK, n=4, r=4)
                sample = sample_of(kind, rng, order, W)
                full = z_shapes(kind, order, W, C)
                oracle = LeastSquaresOracle()
                for y_label, keep in y_forms(kind, order, C).items():
                    y = of_type(kind, patterned(rng, full, keep))
                    for w_label, weight in weight_forms(rng, kind, order, C).items():
                        r_ref, y_ref = dense_sums(kind, order, lambda c: tree_at(frame, c[0]) if C else frame,
                                                  sample, y, weight, C)
                        batch = LeastSquaresBatch(kind, order, sample, y, weight=weight)
                        pb = pack_batch(batch, NPACK)
                        if kind == 'probe':
                            pb = with_garbage(pb)
                        calls = [('frame', frame, batch), ('point', x.data, batch), ('uniform', xu.data, pb)]
                        with self.subTest(kind=kind, order=order, C=C, y=y_label, weight=w_label):
                            (engine,) = sv.sampling_variables(kind, frame, sample, ('r_block_sumsq',), order=order,
                                                              y=y, weight=weight)
                            self.assertEqual(np.shape(engine), np.shape(r_ref))
                            self.assertLess(relerr(engine, r_ref), 1e-12, 'engine')
                            for label, fr, b in calls:
                                (rbs, ybs, f), _ = oracle.query(fr, b, ('r_block_sumsq', 'y_block_sumsq', 'f'))
                                self.assertEqual(np.shape(rbs), np.shape(r_ref), label)
                                self.assertEqual(np.shape(ybs), np.shape(y_ref), label)
                                self.assertLess(relerr(rbs, r_ref), 1e-12, label)
                                self.assertLess(relerr(ybs, y_ref), 1e-12, label)
                                # the law, two routes of the engine: 1/2 sum of the blocks is the misfit summed over C
                                self.assertLess(abs(0.5 * float(np.sum(rbs)) - float(np.sum(f))),
                                                1e-12 * max(float(np.sum(f)), 1e-300), label)
                            n += 1
        self.assertEqual(n, sum(4 * (len(y_forms(k, o, C))) for k, o in CASES for C in [(), (2,)]))


class TestSharedData(unittest.TestCase):
    """A ``y`` shared by the stack (or by the samples) counts once per entry of ``z``: ``y_block_sumsq`` is ``|C|``
    (``|W|``) times one copy's. Summed over ``y``'s own shape it would count once, and the display's ratio
    ``sqrt(r_block_sumsq / y_block_sumsq)`` would be ``sqrt |C|`` (``sqrt |W|``) too large."""

    def test_shared_by_the_stack_and_by_the_samples(self):
        rng = np.random.default_rng(5)
        np.random.seed(5)
        C, Wb = (3,), (4,)
        x = t3.TuckerTensorTrain.randn(SHAPE, *RANKS, stack_shape=C)
        frame = t3m.MANIFOLD.frame(x).data
        for kind, order in [('probe', (1,)), ('apply', (1,)), ('entries', ())]:
            m = len(order)
            sample = sample_of(kind, rng, order, Wb)
            F = tuple(n + 1 for n in order)
            oracle = LeastSquaresOracle()
            arrays = len(SHAPE) if kind == 'probe' else 1
            for label, counted, shape_of in [
                    ('stack', C[0], lambda N: F + Wb + (1,) + ((N,) if kind == 'probe' else ())),
                    ('samples', Wb[0], lambda N: F + (1,) + C + ((N,) if kind == 'probe' else ()))]:
                copies = [rng.standard_normal(shape_of(N)) for N in SHAPE[:arrays]]
                y = of_type(kind, copies)
                one_copy = [np.sum(a ** 2, axis=tuple(range(m, a.ndim))) for a in copies]
                one_copy = np.stack(one_copy) if kind == 'probe' else one_copy[0]
                with self.subTest(kind=kind, shared_by=label):
                    (ybs,), _ = oracle.query(frame, LeastSquaresBatch(kind, order, sample, y, weight=None),
                                             ('y_block_sumsq',))
                    self.assertLess(relerr(ybs, counted * one_copy), 1e-13)


class TestNoWeight(unittest.TestCase):

    def test_none_ones_and_blocks_of_ones_are_the_sum_of_squares(self):
        rng = np.random.default_rng(7)
        np.random.seed(7)
        for kind, order in CASES:
            for C in [(), (2,)]:
                x = t3.TuckerTensorTrain.randn(SHAPE, *RANKS, stack_shape=C)
                frame = t3m.MANIFOLD.frame(x).data
                sample = sample_of(kind, rng, order, W)
                full = z_shapes(kind, order, W, C)
                y = of_type(kind, [rng.standard_normal(s) for s in full])
                m = len(order)
                (r,) = sv.sampling_variables(kind, frame, sample, ('r',), order=order, y=y)
                rs = list(r) if kind == 'probe' else [r]
                plain = [np.sum(r_i * r_i, axis=tuple(range(m, r_i.ndim))) for r_i in rs]
                plain = np.stack(plain) if kind == 'probe' else plain[0]
                blocks = np.ones(((len(SHAPE),) if kind == 'probe' else ()) + tuple(n + 1 for n in order))
                ones = {'None': None, 'ones': of_type(kind, [np.ones(s) for s in full]),
                        'blocks of ones': blocks_to_weight(kind, order, blocks, y)}
                for label, weight in ones.items():
                    with self.subTest(kind=kind, order=order, C=C, weight=label):
                        (got,) = sv.sampling_variables(kind, frame, sample, ('r_block_sumsq',), order=order, y=y,
                                                       weight=weight)
                        self.assertEqual(np.asarray(got).tobytes(), np.asarray(plain).tobytes())
                if not C:
                    with self.subTest(kind=kind, order=order, name='y_block_sumsq'):
                        (ybs,), _ = LeastSquaresOracle().query(frame, LeastSquaresBatch(kind, order, sample, y,
                                                                                        weight=None),
                                                                          ('y_block_sumsq',))
                        ref = data_preparation.data_block_norms(kind, order, y) ** 2
                        self.assertLess(relerr(ybs, ref), 1e-14)


class TestTheScale(unittest.TestCase):

    def test_a_drawn_batch(self):
        """On a draw of ``b`` of ``n`` samples both sums are ``n / b`` times the dense sums over the drawn samples (the
        scale inside ``M``), so their ratio is the plain subset's."""
        rng = np.random.default_rng(11)
        np.random.seed(11)
        Wn, indices = (6,), np.array([1, 4])
        x = t3.TuckerTensorTrain.randn(SHAPE, *RANKS)
        frame = t3m.MANIFOLD.frame(x).data
        for kind, order in [('probe', (1,)), ('apply', (2,)), ('entries', ())]:
            sample = sample_of(kind, rng, order, Wn)
            full = z_shapes(kind, order, Wn, ())
            y = of_type(kind, [rng.standard_normal(s) for s in full])
            m = len(order)
            weight = of_type(kind, patterned(rng, full, [[False] * m + [True] + [False] * (len(s) - m - 1)
                                                         for s in full], positive=True))
            oracle = LeastSquaresOracle()
            drawn = LeastSquaresBatch(kind, order, sample, y, weight=weight).take(indices)
            pick = lambda a: np.take(np.asarray(a), indices, axis=m)           # W is the axis after the jets
            if kind == 'entries':
                sub_sample = np.asarray(sample)[:, indices]
            else:
                sub_sample = [np.take(s, indices, axis=m) for s in sample]
            sub_y = [pick(a) for a in y] if kind == 'probe' else pick(y)
            sub_w = [pick(a) for a in weight] if kind == 'probe' else pick(weight)
            z = sdef.z(kind, frame, sub_sample, order=order)
            z = list(z) if kind == 'probe' else z
            r_ref = dense_oracle.block_sumsq_full(kind, dense_oracle.weighted_residual_full(kind, z, sub_y, sub_w), m)
            y_ref = dense_oracle.block_sumsq_full(kind, dense_oracle.weighted_full(kind, z, sub_y, sub_w), m)
            scale = 6 / 2
            with self.subTest(kind=kind):
                self.assertEqual(float(drawn.minibatch_scale), scale)
                (rbs, ybs), _ = oracle.query(frame, drawn, ('r_block_sumsq', 'y_block_sumsq'))
                self.assertLess(relerr(rbs, scale * r_ref), 1e-12)
                self.assertLess(relerr(ybs, scale * y_ref), 1e-12)
                np.testing.assert_allclose(np.sqrt(np.asarray(rbs) / np.asarray(ybs)), np.sqrt(r_ref / y_ref),
                                           rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
