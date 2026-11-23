# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The engine on data-space objects of the type of ``z`` (the data-metric arc, M2 part 2).

``y`` and the weight ``omega`` are of the type of ``z``: each array of ``z``'s rank, each axis 1 or ``z``'s size,
never the stack of directions ``K``. The matrices of ``test_sampling_variables``, ``_stacks`` and ``_uniform`` check
the values against the reference module (part 1, pinned against brute force) and the unstacked engine; here are the
named oracles of the plan, each independent of the engine's weighting:

0. **Against the reference, over the weight patterns** (`TestAgainstTheReference`): on the reference's problems
   (``small``, where at the curve ``F = W = N_1 = (3,)``; ``tails``, ``W = (2, 2)``; ``W = ()``), the three kinds and
   four jet cases, the weight per entry, per sample, per slot, one per mode, and on the edge and the first jet axis,
   ``y`` full or constant in ``W``, a stack of two directions ``K``: ``f``, ``r``, ``w_tilde``, ``g`` and, element by
   element of ``K``, ``Jp``, ``Jp_norm_sq``, ``JtJp`` and ``Kp``.

1. **The structural check** (`TestTheCheck`): every other form raises a ``ValueError``; in particular a ``y`` without
   ``C`` on a stacked model with ``W`` of the size of ``C``, which plain ``z - y`` would right-align along ``C`` with
   no error. The accepted forms (size-1 axes, a mode-leading array) pass.
2. **A sample with ``omega = 0`` is the batch without it** (`TestAZeroWeightDeletesASample`): ``f``, ``g``,
   ``Jp_norm_sq`` and ``JtJp`` equal those of the batch with that sample deleted, the sample's residual being nonzero
   (its deletion is visible: with weight 1 it changes ``f``); three kinds, ragged and uniform.
3. **Per-element stacked data are separate fits** (`TestStackedDataAreSeparateFits`): on a stack ``C = (2,)``, ``y``
   and ``omega`` different per element give, element by element, the unstacked call at that element's data.
4. **The uniform layout with an edge of 1** (`TestTheUniformEdgeOfOne`): a probe ``y`` and ``omega`` constant over the
   mode edge give the ragged values; the ``y`` mask is what keeps ``y`` off the padding of ``r`` (correction b). Edges
   of 1 mixed with full ones are broadcast, not zero-padded; and a weight whose padding is too large to square is
   masked before it is squared.
5. **The laws on the engine** (`TestTheLaws`): the jet law and the input law (``tests/_typing_laws.py``) on ``f``,
   ``g``, ``r``, ``w_tilde``, ``Jp``, ``Jp_norm_sq``, ``JtJp`` and ``Kp``, with the weight moved by its own law.

Garbage in the padding is finite garbage (the masks multiply; a NaN survives ``0 * NaN``, the house rule of
``docs/contributor/testing_strategy.md``). numpy only.
"""
import os
import sys
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.uniform_frame_variations_format as ufv
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.sampling_definitions as sdef
from t3toolbox.backend.jets import affine_jet, blocks_to_weight

from _typing_laws import JetLaw, InputLaw, assert_law

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampling_definitions as defs        # the reference's problems and jet matrix (part 1)

SHAPE, TUCKER, TT = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1)
d = len(SHAPE)
TOL = 1e-11


def leaves(value):
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in leaves(item)]
    return [np.asarray(value)]


def close(test, a, b, tol=TOL):
    a, b = leaves(a), leaves(b)
    test.assertEqual(len(a), len(b))
    for u, v in zip(a, b):
        test.assertEqual(u.shape, v.shape)
        test.assertLessEqual(float(np.linalg.norm(u - v)), tol * max(float(np.linalg.norm(v)), 1.0))


def sample_of(rng, kind, order, W):
    if kind == 'entries':
        return np.stack([rng.integers(0, n, size=W) for n in SHAPE])
    ww = [rng.standard_normal(W + (n,)) for n in SHAPE]
    if not order:
        return ww
    return list(affine_jet(ww, *([rng.standard_normal(W + (n,)) for n in SHAPE] for _ in order)))


def z_shapes(kind, order, W, C=()):
    F = tuple(n + 1 for n in order)
    if kind == 'probe':
        return [F + W + C + (n,) for n in SHAPE]
    return [F + W + C] if kind == 'apply' else [W + C]


def of_kind(kind, arrays):
    return list(arrays) if kind == 'probe' else arrays[0]


class TestAgainstTheReference(unittest.TestCase):

    PATTERNS = ('per entry', 'per sample', 'per slot', 'one per mode', 'edge and first jet axis')
    PLAIN, WITH_K = ('f', 'r', 'w_tilde', 'g'), ('Jp', 'Jp_norm_sq', 'JtJp', 'Kp')

    @staticmethod
    def full_axes(pattern, kind, m, n_W, ndim):
        """The axes on which a weight of ``pattern`` has ``z``'s size (every other axis is 1)."""
        if pattern == 'per entry':
            return set(range(ndim))
        if pattern == 'per sample':
            return set(range(m, m + n_W))
        if pattern == 'per slot':
            return set(range(m))
        if pattern == 'one per mode':
            return set()
        return ({ndim - 1} if kind == 'probe' else set()) | ({0} if m else set())

    @staticmethod
    def at_k(name, value, k, lead):
        """Element ``k`` of the stack of directions: after the ``lead`` axes (jets and ``W``) of ``Jp``, first for
        ``Jp_norm_sq`` and the blocks of ``JtJp`` and ``Kp``."""
        if name == 'Jp_norm_sq':
            return np.asarray(value)[k]
        if name in ('JtJp', 'Kp'):
            return tuple(tuple(block[k] for block in family) for family in value)
        return [np.take(array, k, axis=lead) for array in leaves(value)]

    def test_the_engine_is_the_reference(self):
        problems = (defs.PROBLEMS[0], defs.PROBLEMS[1], defs.PROBLEMS[3])     # small, tails (W = (2, 2)), W = ()
        jet_cases = (defs.NO_JETS, defs.LINE_3, defs.PLANE, defs.CURVE)
        count = 0
        for labels, P, kind, case in defs.cases(problems=problems, jet_cases=jet_cases):
            rng = np.random.default_rng(100 + count)
            sample = P.sample(kind, case)
            m, n_W = case.m, len(P.W)
            V2 = [np.stack([V, rng.standard_normal(V.shape)]) for V in P.V]               # K = (2,)
            H2 = [np.stack([H, rng.standard_normal(H.shape)]) for H in P.H]
            D2 = [np.stack([B, rng.standard_normal(B.shape)]) for B in P.D_dot]
            L2 = [np.stack([G, rng.standard_normal(G.shape)]) for G in P.L_dot]
            shapes = [np.shape(a) for a in leaves(sdef.z(kind, P.frame, sample, None, case.order))]
            for j, pattern in enumerate(self.PATTERNS):
                count += 1
                y_constant = j % 2 == 1                     # alternately: the data constant in W
                axes = self.full_axes(pattern, kind, m, n_W, len(shapes[0]))
                weight = [0.5 + rng.random(tuple(n if a in axes else 1 for a, n in enumerate(sh))) for sh in shapes]
                y_axes = set(range(len(shapes[0]))) - (set(range(m, m + n_W)) if y_constant else set())
                y = [rng.standard_normal(tuple(n if a in y_axes else 1 for a, n in enumerate(sh))) for sh in shapes]
                as_array = kind == 'probe' and len({w.shape for w in weight}) == 1 and j % 2 == 0
                w_arg = np.stack(weight) if as_array else of_kind(kind, weight)
                y_arg = of_kind(kind, y)
                with self.subTest(pattern=pattern, y_constant_in_W=y_constant, **labels):
                    got = dict(zip(self.PLAIN + self.WITH_K, sv.sampling_variables(
                        kind, P.frame, sample, self.PLAIN + self.WITH_K, order=case.order, y=y_arg, weight=w_arg,
                        variations=(V2, H2), core_directions=(D2, L2))))
                    arguments = dict(y=y_arg, weight=w_arg)
                    for name in self.PLAIN:
                        expected = getattr(sdef, name)(kind, P.frame, sample, None, case.order, **arguments)
                        for a, b in zip(leaves(got[name]), leaves(expected)):
                            self.assertLessEqual(defs.relative_error(a, b), defs.TOL, name)
                    for k in range(2):
                        directions = dict(variations=([V[k] for V in V2], [H[k] for H in H2]),
                                          core_directions=([B[k] for B in D2], [G[k] for G in L2]))
                        for name in self.WITH_K:
                            expected = getattr(sdef, name)(kind, P.frame, sample, None, case.order, **arguments,
                                                           **directions)
                            for a, b in zip(leaves(self.at_k(name, got[name], k, m + n_W)), leaves(expected)):
                                self.assertLessEqual(defs.relative_error(a, b), defs.TOL, (name, k))


class TestTheCheck(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.rng = np.random.default_rng(0)
        self.x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT, stack_shape=(2,))
        self.frame = t3m.MANIFOLD.frame(self.x).data

    def call(self, kind, sample, **arguments):
        return sv.sampling_variables(kind, self.frame, sample, ('f',), **arguments)

    def test_a_y_without_C_on_a_stack_raises_where_plain_broadcasting_would_not(self):
        """``W = C = (2,)`` with no jets: a ``y`` of shape ``W + (N_i,)`` against ``z``'s ``W + C + (N_i,)``. Plain
        ``z - y`` right-aligns it, reading ``y``'s sample axis along ``C``, with no error (shown); the check raises."""
        W = (2,)
        for kind in ('probe', 'apply', 'entries'):
            with self.subTest(kind=kind):
                sample = sample_of(self.rng, kind, (), W)
                y = of_kind(kind, [self.rng.standard_normal(W + ((n,) if kind == 'probe' else ())) for n in
                                   (SHAPE if kind == 'probe' else (0,))])
                z_shape = z_shapes(kind, (), W, (2,))[0]
                self.assertEqual(np.broadcast_shapes(z_shape, np.shape(leaves(y)[0])), z_shape)   # silent
                with self.assertRaises(ValueError):
                    self.call(kind, sample, y=y)
                shared = of_kind(kind, [np.expand_dims(a, axis=len(W)) for a in leaves(y)])     # size-1 C: legal
                self.call(kind, sample, y=shared)

    def test_other_forms_raise(self):
        W, order = (3,), (2,)
        F = (3,)
        sample = sample_of(self.rng, 'probe', order, W)
        y = [self.rng.standard_normal(F + W + (1,) + (n,)) for n in SHAPE]
        good = blocks_to_weight('probe', order, 0.5 + self.rng.random((d,) + F), y)       # (d,)+F+(1,1)+(1,)
        bad = {'one per block, (d,) + F': 0.5 + self.rng.random((d,) + F),
               'with a stack of directions K': np.expand_dims(good, 3),
               'an axis neither 1 nor its size': np.ones((d,) + F + (2, 1, 1)),
               'a mode axis of 1': good[:1],
               'd - 1 arrays': list(good[:d - 1])}
        for label, weight in bad.items():
            with self.subTest(weight=label):
                with self.assertRaises(ValueError):
                    self.call('probe', sample, order=order, y=y, weight=weight)
        self.call('probe', sample, order=order, y=y, weight=good)
        self.call('probe', sample, order=order, y=y, weight=list(good))
        with self.subTest(y='a tuple for apply'):
            with self.assertRaises(ValueError):
                self.call('apply', sample, order=order, y=(self.rng.standard_normal(F + W + (2,)),))
        with self.subTest(y='larger than z where z is 1'):           # a one-sample batch, a y of three samples
            one = sample_of(self.rng, 'apply', (), (1,))
            with self.assertRaises(ValueError):
                self.call('apply', one, y=self.rng.standard_normal((3, 2)))

    def test_a_packed_object_has_an_edge_of_1_or_N(self):
        ux = ut3.UniformTuckerTensorTrain.from_t3(self.x)
        frame = ut3m.UNIFORM_MANIFOLD.frame(ux).data
        N = max(SHAPE)
        W = (3,)
        sample = sample_of(self.rng, 'probe', (), W)
        for edge, raises in ((1, False), (N, False), (N - 1, True)):
            with self.subTest(edge=edge):
                y = self.rng.standard_normal((d,) + W + (2,) + (edge,))
                if raises:
                    with self.assertRaises(ValueError):
                        sv.sampling_variables('probe', frame, sample, ('f',), y=y)
                else:
                    sv.sampling_variables('probe', frame, sample, ('f',), y=y)


class TestAZeroWeightDeletesASample(unittest.TestCase):

    def test_f_g_and_the_gauss_newton_terms(self):
        rng = np.random.default_rng(1)
        np.random.seed(1)
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        frame, variations = t3m.MANIFOLD.frame(x).data, None
        V = [rng.standard_normal(np.shape(U)[-2:-1] + (n,)) for U, n in zip(frame[1], SHAPE)]
        H = [rng.standard_normal(np.shape(L)[:-1] + np.shape(R)[-1:]) for L, R in zip(frame[2], frame[3])]
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        ufr = ut3m.UNIFORM_MANIFOLD.frame(ux).data
        names = ('f', 'g', 'Jp_norm_sq', 'JtJp')
        W, drop = (5,), 2
        keep = [j for j in range(W[0]) if j != drop]
        for kind in ('probe', 'apply', 'entries'):
            for order in (((), (2,)) if kind != 'entries' else ((),)):
                m = len(order)
                sample = sample_of(rng, kind, order, W)
                y = of_kind(kind, [rng.standard_normal(s) for s in z_shapes(kind, order, W)])
                per_sample = [0.5 + rng.random(tuple(n if a == m else 1 for a, n in enumerate(s)))
                              for s in z_shapes(kind, order, W)]                     # a weight per sample
                zeroed = [w.copy() for w in per_sample]
                for w in zeroed:
                    w[(slice(None),) * m + (drop,)] = 0.0
                take = lambda v, axis: np.take(np.asarray(v), keep, axis=axis)
                sample_without = (take(sample, 1) if kind == 'entries' else [take(w, m) for w in sample])
                y_without = of_kind(kind, [take(a, m) for a in leaves(y)])
                weight_without = of_kind(kind, [take(w, m) for w in per_sample])
                layouts = [('ragged', frame, (V, H))]
                if kind == 'probe':
                    pad = ufr[0].shape[-1]
                    Vu = np.stack([np.pad(v, ((0, ufr[1].shape[-2] - v.shape[0]), (0, pad - v.shape[1]))) for v in V])
                    Hu = np.stack([np.pad(h, [(0, a - b) for a, b in zip(ufr[2].shape[1:], h.shape)]) for h in H])
                    layouts.append(('uniform', ufr, (Vu, Hu)))
                for layout, fr, direction in layouts:
                    with self.subTest(kind=kind, order=repr(order), layout=layout):
                        got = sv.sampling_variables(kind, fr, sample, names, order=order, y=y,
                                                    weight=of_kind(kind, zeroed), variations=direction)
                        without = sv.sampling_variables(kind, fr, sample_without, names, order=order, y=y_without,
                                                        weight=weight_without, variations=direction)
                        close(self, got, without)
                        (f_kept,) = sv.sampling_variables(kind, fr, sample, ('f',), order=order, y=y,
                                                          weight=of_kind(kind, per_sample))
                        self.assertGreater(abs(float(f_kept) - float(got[0])), 1e-3 * abs(float(got[0])))


class TestStackedDataAreSeparateFits(unittest.TestCase):

    def test_element_by_element(self):
        rng = np.random.default_rng(2)
        np.random.seed(2)
        C, W = (2,), (3,)
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT, stack_shape=C)
        frame = t3m.MANIFOLD.frame(x).data
        names = ('f', 'r', 'g', 'w_tilde')
        for kind in ('probe', 'apply', 'entries'):
            for order in (((), (1,)) if kind != 'entries' else ((),)):
                m = len(order)
                sample = sample_of(rng, kind, order, W)
                shapes = z_shapes(kind, order, W, C)
                y = of_kind(kind, [rng.standard_normal(s) for s in shapes])
                weight = of_kind(kind, [0.5 + rng.random(s) for s in shapes])
                with self.subTest(kind=kind, order=repr(order)):
                    stacked = dict(zip(names, sv.sampling_variables(kind, frame, sample, names, order=order, y=y,
                                                                    weight=weight)))
                    lead = m + len(W)
                    for c in range(C[0]):
                        at = lambda v: of_kind(kind, [a[(slice(None),) * lead + (c,)] for a in leaves(v)])
                        frame_c = tuple(tuple(core[c] for core in store) for store in frame)
                        alone = dict(zip(names, sv.sampling_variables(kind, frame_c, sample, names, order=order,
                                                                      y=at(y), weight=at(weight))))
                        close(self, np.asarray(stacked['f'])[c], alone['f'])
                        close(self, at(stacked['r']), alone['r'])
                        close(self, at(stacked['w_tilde']), alone['w_tilde'])
                        close(self, [b[c] for b in leaves(stacked['g'])], alone['g'])
                    other = dict(zip(names, sv.sampling_variables(kind, tuple(tuple(core[1] for core in store)
                                                                              for store in frame), sample, names,
                                                                  order=order, y=of_kind(kind, [a[(slice(None),) *
                                                                  lead + (0,)] for a in leaves(y)]), weight=None)))
                    self.assertGreater(abs(float(other['f']) - float(np.asarray(stacked['f'])[1])), 1e-6)


class TestTheUniformEdgeOfOne(unittest.TestCase):
    """The probe ``y`` and weight on the uniform layout, against the ragged layout on the same frame (the uniform frame
    packs the ragged one, so ``g`` is in the same gauge and compares block by block)."""

    NAMES = ('f', 'g', 'r_block_sumsq')

    def setUp(self):
        self.rng = np.random.default_rng(3)
        np.random.seed(3)
        self.C, self.W = (2,), (3,)
        x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT, stack_shape=self.C)
        ragged_frame = t3m.MANIFOLD.frame(x)
        self.frame, self.ufr = ragged_frame.data, ufv.UT3Frame.from_t3frame(ragged_frame).data

    def assert_same(self, uniform, ragged):
        close(self, uniform[0], ragged[0])
        close(self, uniform[2], ragged[2])
        for a, b in zip(uniform[1][0], ragged[1][0]):                                     # the V blocks
            close(self, a[..., :b.shape[-2], :b.shape[-1]], b)
        for a, b in zip(uniform[1][1], ragged[1][1]):                                     # the H blocks
            close(self, a[..., :b.shape[-3], :b.shape[-2], :b.shape[-1]], b)

    def test_y_and_the_weight_constant_over_the_edge(self):
        """Every mode's edge 1, given ragged or packed: the packed object keeps the edge 1, and the ``y`` mask
        broadcasts it to ``N`` with zero padding (without it ``-y`` would sit on the padding of ``r``)."""
        rng, C, W = self.rng, self.C, self.W
        for order in ((), (2,)):
            sample = sample_of(rng, 'probe', order, W)
            F = tuple(n + 1 for n in order)
            y = [rng.standard_normal(F + W + C + (1,)) for _ in SHAPE]
            for weight_label, weight in (('edge 1', [0.5 + rng.random(F + (1,) + C + (1,)) for _ in SHAPE]),
                                         ('none', None)):
                ragged = sv.sampling_variables('probe', self.frame, sample, self.NAMES, order=order, y=y,
                                               weight=weight)
                for form, (y_u, w_u) in (('ragged in', (y, weight)),
                                         ('packed in', (np.stack(y), None if weight is None else np.stack(weight)))):
                    with self.subTest(order=repr(order), weight=weight_label, form=form):
                        self.assert_same(sv.sampling_variables('probe', self.ufr, sample, self.NAMES, order=order,
                                                               y=y_u, weight=w_u), ragged)

    def test_mixed_edges(self):
        """Given ragged, the edge 1 in some modes and ``N_i`` in others: the packed edge is ``N``, the arrays of edge
        1 broadcast over it (never zero-padded, which would put their value on entry 0 only)."""
        rng, C, W = self.rng, self.C, self.W
        for order in ((), (2,)):
            sample = sample_of(rng, 'probe', order, W)
            F = tuple(n + 1 for n in order)
            y = [rng.standard_normal(F + W + C + ((1,) if i == 0 else (n,))) for i, n in enumerate(SHAPE)]
            weight = [0.5 + rng.random(F + (1,) + C + ((1,) if i == 1 else (n,))) for i, n in enumerate(SHAPE)]
            with self.subTest(order=repr(order)):
                ragged = sv.sampling_variables('probe', self.frame, sample, self.NAMES, order=order, y=y,
                                               weight=weight)
                self.assert_same(sv.sampling_variables('probe', self.ufr, sample, self.NAMES, order=order, y=y,
                                                       weight=weight), ragged)

    def test_modes_with_different_size_1_axes(self):
        """Given ragged, the modes differ in their size-1 axes of ``F + W + C`` as well as in their edges: per slot with
        a full edge, per sample and element of ``C`` with an edge of 1, and per entry. The packed body is the join
        across the modes (each axis full if any mode has it full), so every mode is broadcast to it."""
        rng, C, W = self.rng, self.C, self.W
        order = (2,)
        sample = sample_of(rng, 'probe', order, W)
        F = (3,)
        patterns = (lambda n: F + (1,) + (1,) + (n,),                  # per slot, a full edge
                    lambda n: (1,) + W + C + (1,),                      # per sample and element, an edge of 1
                    lambda n: F + W + C + (n,))                         # per entry
        for label, rotate in (('slot, sample, entry', 0), ('sample, entry, slot', 1), ('entry, slot, sample', 2)):
            y = [rng.standard_normal(patterns[(i + rotate) % d](n)) for i, n in enumerate(SHAPE)]
            weight = [0.5 + rng.random(patterns[(i + rotate + 1) % d](n)) for i, n in enumerate(SHAPE)]   # not y's
            with self.subTest(patterns=label):
                ragged = sv.sampling_variables('probe', self.frame, sample, self.NAMES, order=order, y=y,
                                               weight=weight)
                self.assert_same(sv.sampling_variables('probe', self.ufr, sample, self.NAMES, order=order, y=y,
                                                       weight=weight), ragged)

    def test_a_weight_whose_padding_would_overflow_when_squared(self):
        """A packed weight with an edge of ``N`` whose padding holds a finite number too large to square (``1e200``:
        its square is ``inf``, and ``inf`` times the zero padding of ``r`` is NaN). The engine masks the weight on
        entry, so the values are finite and the ragged layout's. (Finite garbage, the house rule: a NaN or an
        ``inf`` in the padding survives any mask, which multiplies.)"""
        rng, C, W = self.rng, self.C, self.W
        N = max(SHAPE)
        for order in ((), (2,)):
            sample = sample_of(rng, 'probe', order, W)
            F = tuple(n + 1 for n in order)
            y = [rng.standard_normal(F + W + C + (n,)) for n in SHAPE]
            weight = [0.5 + rng.random(F + W + (1,) + (n,)) for n in SHAPE]
            packed = np.full((d,) + F + W + (1,) + (N,), 1e200)
            for i, n in enumerate(SHAPE):
                packed[i, ..., :n] = weight[i]
            with self.subTest(order=repr(order)):
                ragged = sv.sampling_variables('probe', self.frame, sample, self.NAMES, order=order, y=y,
                                               weight=weight)
                uniform = sv.sampling_variables('probe', self.ufr, sample, self.NAMES, order=order, y=y,
                                                weight=packed)
                self.assertTrue(all(np.all(np.isfinite(leaf)) for leaf in leaves(uniform)))
                self.assert_same(uniform, ragged)


class TestTheLaws(unittest.TestCase):
    """The jet law (``s_j -> ell_j s_j``) and the input law (``w_i -> Delta_i w_i``) on the engine's outputs at a
    corewise point, with ``y`` moved as a measurement and the weight by its own law."""

    NAMES = ('f', 'r', 'w_tilde', 'g', 'Jp', 'Jp_norm_sq', 'JtJp', 'Kp')

    def setUp(self):
        np.random.seed(4)
        self.rng = np.random.default_rng(4)
        self.x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT).data
        self.cd = ([self.rng.standard_normal(B.shape) for B in self.x[0]],
                   [self.rng.standard_normal(G.shape) for G in self.x[1]])

    def run_engine(self, kind, order):
        def function(x, sample, y, weight, cd):
            return sv.sampling_variables(kind, x, sample, self.NAMES, order=order, y=y, weight=weight,
                                         core_directions=cd)
        return function

    def test_the_jet_law(self):
        for kind in ('probe', 'apply'):
            for order, ell in (((2,), (0.7,)), ((1, 1), (0.7, 1.9)), ((2, 1), (0.7, 1.9))):
                with self.subTest(kind=kind, order=repr(order)):
                    law = JetLaw(order, ell)
                    sample = sample_of(self.rng, kind, order, (3,))
                    shapes = z_shapes(kind, order, (3,))
                    y = of_kind(kind, [self.rng.standard_normal(s) for s in shapes])
                    weight = of_kind(kind, [0.5 + self.rng.random(s) for s in shapes])

                    def move_out(out):
                        f, r, w_tilde, g, Jp, Jp_norm_sq, JtJp, Kp = out
                        return (f, law.P(r), law.P_star_turned(w_tilde), g, law.P(Jp), Jp_norm_sq, JtJp, Kp)
                    assert_law(self, self.run_engine(kind, order), (self.x, sample, y, weight, self.cd),
                               (self.x, law.inputs(sample), law.P(y), law.P_star(weight), self.cd), move_out,
                               rtol=1e-11)

    def test_the_input_law(self):
        for kind in ('probe', 'apply'):
            for order in ((), (2,)):
                with self.subTest(kind=kind, order=repr(order)):
                    law = InputLaw([0.5 + 1.5 * self.rng.random(n) for n in SHAPE])
                    sample = sample_of(self.rng, kind, order, (3,))
                    shapes = z_shapes(kind, order, (3,))
                    y = of_kind(kind, [self.rng.standard_normal(s) for s in shapes])
                    weight = of_kind(kind, [0.5 + self.rng.random(s) for s in shapes])    # a full edge (probe)
                    moved_cd = (law.tucker((self.cd[0], ()))[0], self.cd[1])
                    measurement = law.V_star if kind == 'probe' else (lambda v: v)
                    dual = law.V if kind == 'probe' else (lambda v: v)
                    blocks = lambda pair: (law.tucker_dual(pair[0]), pair[1])

                    def move_out(out):
                        f, r, w_tilde, g, Jp, Jp_norm_sq, JtJp, Kp = out
                        return (f, measurement(r), dual(w_tilde), blocks(g), measurement(Jp), Jp_norm_sq,
                                blocks(JtJp), blocks(Kp))
                    assert_law(self, self.run_engine(kind, order), (self.x, sample, y, weight, self.cd),
                               (law.tucker(self.x), law.inputs(sample), measurement(y), dual(weight), moved_cd),
                               move_out, rtol=1e-11)


if __name__ == '__main__':
    unittest.main()
