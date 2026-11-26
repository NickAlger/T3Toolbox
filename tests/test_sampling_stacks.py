# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_variables`` with stacks: ``W`` of any number of axes, ``K`` on the directions, ``C`` on every
core, ``sum_over_samples=False``, the stack check, and the dtype rule.

1. **The stack matrix** (`TestTheStackMatrix`): the stack configurations of ``docs/contributor/testing_strategy.md``
   (no stack, ``C``, ``K``, ``K + C``, ``C = (2, 3)``, ``K = (2, 3)``; forced padding is the uniform layout's) on a
   covering design: every configuration with every ``W`` in ``()``, ``(3,)``, ``(2, 2)`` once per kind, the jet shapes
   ``()``, ``(2,)``, ``(1, 1)`` assigned cyclically (entries at ``()``), the call type alternating. Every name of the
   call, element ``(k, c)`` by element, against the UNSTACKED engine (itself checked against the reference, P5 to P9b)
   at frame element ``c``, direction ``k`` and ``W`` flattened to one axis, so a multi-axis ``W`` is checked against a
   single-axis one; and one maximal case per kind, every group multi-axis at once.
2. **Held stacked variables** (`TestHeldStacked`): a held ``mu`` and ``r`` with ``C`` and a multi-axis ``W`` give the
   recomputed outputs, bit for bit.
3. **Unsummed blocks** (`TestUnsummed`): with ``sum_over_samples=False`` the blocks are ``W + [K] + C`` and the core,
   summed over ``W`` they are the summed blocks, and the block of one sample (the first, the last, one off the
   diagonal of a multi-axis ``W``) is the call on that sample alone; on a multi-axis ``K``, a multi-axis ``W``
   (``(2, 3)`` for entries), ``C`` and entries.
4. **The reference anchor** (`TestTheReferenceAnchor`): the probe kind at ``(2,)`` and ``W = (2, 2)`` with
   ``C = (2, 3)``, with ``K = (2, 3)``, and the maximal case, element by element against the reference module.
5. **The stack check** (`TestTheStackCheck`, removable with ``_check_stacks``): a size-1 ``W`` or ``C`` beside a larger
   one, a missing ``C``, on both call types, raise naming the operand; a consistent ``C = (1,)`` passes.
6. **The dtype rule** (`TestTheDtypeMatrix`): every output on numpy and jax, float64 and float32 (jax float64 in
   ``jax_in_double_precision``), the mixed cases of the plan's P10, held variables of another precision, an integer
   index and entries' one-hot ``'w'``.

numpy only, but for the jax column of 6.
"""
import itertools
import math
import os
import sys
import unittest

import numpy as np

import t3toolbox.backend.jets as jets
import t3toolbox.backend.sampling_definitions as sdef
import t3toolbox.backend.sampling_variables as sv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampling_definitions as defs

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

TOL = 1e-12
N, nD, nU, rL, rR = (4, 5, 3), (2, 3, 2), (3, 2, 3), (1, 2, 3, 2), (1, 3, 2, 3)     # tails at bond d, L and R differ
D_SITES = 3
STACKS = (((), ()), ((2,), ()), ((), (3,)), ((2,), (3,)), ((2, 3), ()), ((), (2, 3)))   # (C, K)
SAMPLE_STACKS = ((), (3,), (2, 2))
ORDERS = {'probe': ((), (2,), (1, 1)), 'apply': ((), (2,), (1, 1)), 'entries': ((),)}
F_IN = {(): (), (2,): (2,), (1, 1): (2, 2)}                                         # a line; two directions
BLOCKS = ('V_tilde', 'H_tilde', 'V_tilde_dot', 'H_tilde_dot')
PAIRS = {'g': ('V_tilde', 'H_tilde'), 'JtJp': ('V_tilde', 'H_tilde'), 'Kp': ('V_tilde_dot', 'H_tilde_dot')}


class Case:
    """A stacked call: a frame with ``C``, directions with ``K + C``, a sample with ``W``, data and a weight."""

    def __init__(self, kind, C, K, W, order, call, seed):
        rng = np.random.default_rng(seed)
        self.kind, self.C, self.K, self.W, self.order, self.call = kind, C, K, W, order, call
        self.m, self.box = len(order), tuple(n + 1 for n in order)
        d = D_SITES
        self.frame = (tuple(rng.standard_normal(C + (nD[i], N[i])) for i in range(d)),
                      tuple(rng.standard_normal(C + (rL[i], nU[i], rR[i + 1])) for i in range(d)),
                      tuple(rng.standard_normal(C + (rL[i], nD[i], rL[i + 1])) for i in range(d)),
                      tuple(rng.standard_normal(C + (rR[i], nD[i], rR[i + 1])) for i in range(d)))
        self.variations = (tuple(rng.standard_normal(K + C + (nU[i], N[i])) for i in range(d)),
                           tuple(rng.standard_normal(K + C + (rL[i], nD[i], rR[i + 1])) for i in range(d)))
        self.core_directions = (tuple(rng.standard_normal(K + C + (nD[i], N[i])) for i in range(d)),
                                tuple(rng.standard_normal(K + C + (rL[i], nD[i], rL[i + 1])) for i in range(d)))
        # y and the weight of the type of z (the data-metric rule): one per entry and per element of the stack C
        if kind == 'entries':
            self.sample = np.stack([rng.integers(0, 2, size=W) for _ in range(d)])      # repeated indices
            self.y = rng.standard_normal(W + C)
            self.weight = 0.5 + rng.random(W + C)
        else:
            self.sample = [rng.standard_normal(F_IN[order] + W + (N[i],)) for i in range(d)]
            if kind == 'probe':
                self.y = [rng.standard_normal(self.box + W + C + (N[i],)) for i in range(d)]
                self.weight = [0.5 + rng.random(self.box + W + C + (N[i],)) for i in range(d)]
            else:
                self.y = rng.standard_normal(self.box + W + C)
                self.weight = 0.5 + rng.random(self.box + W + C)

    @property
    def at(self):
        return self.frame if self.call == '_manifold' else (self.frame[0], self.frame[2])

    def arguments(self, k=None, c=None, flat=False):
        """The call's keyword arguments, at direction ``k`` and frame element ``c`` when given, ``W`` flattened."""
        pick = lambda pair: tuple(tuple(core[k + c] for core in cores) for cores in pair) if k is not None else pair
        y, weight = (self.y, self.weight) if c is None else (self.at_element(self.y, c), self.at_element(self.weight, c))
        arguments = dict(order=self.order, y=self.flat(y, 'y') if flat else y,
                         weight=self.flat(weight, 'y') if flat else weight, core_directions=pick(self.core_directions))
        if self.call == '_manifold':
            arguments['variations'] = pick(self.variations)
        return arguments

    def at_element(self, value, c):
        """Element ``c`` of a data-space object of the type of ``z``: its ``C`` axes (after ``F + W``) taken."""
        where = (slice(None),) * (self.m + len(self.W)) + tuple(c)
        return [v[where] for v in value] if self.kind == 'probe' else value[where]

    def element_at(self, c):
        stores = self.at
        return tuple(tuple(core[c] for core in store) for store in stores)

    def flat(self, value, what):
        """``W`` flattened to one axis: in the sample, the index or the data."""
        size = (math.prod(self.W),)
        if what == 'index':
            return value.reshape((D_SITES,) + size)
        if what == 'y':
            if self.kind == 'probe':
                return [y.reshape(self.box + size + y.shape[-1:]) for y in value]
            return value.reshape(self.box + size)
        return [w.reshape(F_IN[self.order] + size + w.shape[-1:]) for w in value]

    def names(self):
        return tuple(name for name in sv._names(self.kind, self.call) if name != 'w' or self.kind == 'entries')

    def with_K(self):
        return {rule.name for rule in sv._GRAPHS[self.kind, self.call] if rule.carries_K}

    def run(self, names=None, **extra):
        names = self.names() if names is None else names
        return sv.sampling_variables(self.kind, self.at, self.sample, names, **dict(self.arguments(), **extra))

    def oracle(self, k, c, names):
        """The unstacked engine at direction ``k``, frame element ``c``, ``W`` flattened to one axis."""
        sample = self.flat(self.sample, 'index' if self.kind == 'entries' else 'w')
        return sv.sampling_variables(self.kind, self.element_at(c), sample, names,
                                     **self.arguments(k, c, flat=True))

    def element(self, name, value, k, c, keep_W=False):
        """Element ``(k, c)`` of a stacked variable, its ``W`` flattened to one axis."""
        k_ = k if name in self.with_K() else ()
        if name in PAIRS:
            return tuple(tuple(self.block(block, k_, c, keep_W) for block in family) for family in value)
        if name in BLOCKS:
            return tuple(self.block(block, k_, c, keep_W) for block in value)
        if name == 'f':
            return np.asarray(value)[c]
        if name == 'Jp_norm_sq':
            return np.asarray(value)[k + c]
        if name == 'w':                                         # entries' one-hot vectors: W + (N,)
            return tuple(e.reshape((math.prod(self.W),) + e.shape[-1:]) for e in value)
        arrays = value if isinstance(value, tuple) else (value,)
        out = []
        for array in arrays:
            lead = self.m + len(self.W)
            taken = np.asarray(array)[(slice(None),) * lead + k_ + c]
            out.append(taken.reshape(taken.shape[:self.m] + (math.prod(self.W),) + taken.shape[lead:]))
        return tuple(out) if isinstance(value, tuple) else out[0]

    def block(self, block, k, c, keep_W):
        if keep_W:
            taken = np.asarray(block)[(slice(None),) * len(self.W) + k + c]
            return taken.reshape((math.prod(self.W),) + taken.shape[len(self.W):])
        return np.asarray(block)[k + c]


def leaves(value):
    """The arrays of a value as returned (numpy or jax), not converted."""
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in leaves(item)]
    return [value]


def arrays_of(value):
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in arrays_of(item)]
    return [np.asarray(value)]


def covering_cases():
    """Every stack configuration with every ``W`` once per kind; the jet shapes cyclic; the call type alternating."""
    cases = []
    for kind in ('probe', 'apply', 'entries'):
        for (s, (C, K)), (w, W) in itertools.product(enumerate(STACKS), enumerate(SAMPLE_STACKS)):
            order = ORDERS[kind][(s + w) % len(ORDERS[kind])]
            call = '_manifold' if (s + w) % 2 == 0 else '_corewise'
            cases.append(Case(kind, C, K, W, order, call, seed=100 * s + 10 * w))
    return cases


def maximal_cases():
    """Every group multi-axis at once: ``K = (2, 3)``, ``C = (2, 3)``, ``W = (2, 2)``, two jet axes."""
    return [Case(kind, (2, 3), (2, 3), (2, 2), ORDERS[kind][-1], '_manifold', seed=7) for kind in ORDERS]


class TestTheStackMatrix(unittest.TestCase):

    def check(self, case):
        names = case.names()
        got = dict(zip(names, case.run(names)))
        oracles = {}
        for k in (tuple(np.ndindex(*case.K)) or ((),)):
            for c in (tuple(np.ndindex(*case.C)) or ((),)):
                oracles[k, c] = dict(zip(names, case.oracle(k, c, names)))
        for name in names:
            with self.subTest(kind=case.kind, call=case.call, C=repr(case.C), K=repr(case.K), W=repr(case.W),
                              order=repr(case.order), name=name):
                for (k, c), oracle in oracles.items():
                    if name == 'r_block_sumsq':         # a sum over C too: the oracles' sum
                        expected = sum(oracles[k, c2][name] for c2 in (tuple(np.ndindex(*case.C)) or ((),)))
                        self.assertLessEqual(defs.relative_error(got[name], expected), TOL)
                        break
                    value = case.element(name, got[name], k, c)
                    self.assertEqual(len(arrays_of(value)), len(arrays_of(oracle[name])))
                    for a, b in zip(arrays_of(value), arrays_of(oracle[name])):
                        self.assertEqual(a.shape, b.shape)
                        self.assertLessEqual(defs.relative_error(a, b), TOL)

    def test_the_covering_design(self):
        cases = covering_cases()
        self.assertEqual(len(cases), 54)
        self.assertIn(('probe', (2, 2), (1, 1)), {(case.kind, case.W, case.order) for case in cases})
        self.assertIn(('apply', (2, 2), (1, 1)), {(case.kind, case.W, case.order) for case in cases})
        for case in cases:
            self.check(case)

    def test_the_maximal_case_of_each_kind(self):
        for case in maximal_cases():
            self.check(case)


class TestHeldStacked(unittest.TestCase):

    def test_held_mu_and_r_with_stacks(self):
        case = Case('probe', (2, 3), (2,), (2, 2), (1, 1), '_manifold', seed=11)
        names = ('z', 'g', 'Kp', 'JtJp')
        mu, r = case.run(('mu', 'r'))
        recomputed = case.run(names)
        held = case.run(names, held=sv.HeldVariables(mu=mu, r=r))
        for name, a, b in zip(names, recomputed, held):
            with self.subTest(name=name):
                for x, y in zip(arrays_of(a), arrays_of(b)):
                    self.assertTrue(np.array_equal(x, y))
        with self.assertRaisesRegex(ValueError, 'the held mu'):             # a held mu without its C
            case.run(('z',), held=sv.HeldVariables(mu=tuple(m[:, :, :, :, 0, 0] for m in mu)))


class TestUnsummed(unittest.TestCase):

    def test_one_block_per_sample(self):
        cases = (Case('probe', (2,), (), (3,), (2,), '_manifold', seed=21),
                 Case('probe', (), (2, 3), (3,), (1, 1), '_manifold', seed=22),
                 Case('apply', (), (3,), (2, 2), (2,), '_corewise', seed=23),
                 Case('entries', (2,), (2,), (2, 3), (), '_manifold', seed=24))
        for case in cases:
            names = ('g', 'Kp', 'JtJp') if case.call == '_manifold' else ('g', 'Kp')
            summed = dict(zip(names, case.run(names)))
            kept = dict(zip(names, case.run(names, sum_over_samples=False)))
            for name in names:
                with self.subTest(kind=case.kind, K=repr(case.K), W=repr(case.W), name=name):
                    k_ = case.K if name != 'g' else ()
                    for a, b in zip(arrays_of(kept[name]), arrays_of(summed[name])):
                        self.assertEqual(a.shape, case.W + b.shape)                     # W + [K] + C + core
                        self.assertEqual(b.shape[:len(k_) + len(case.C)], k_ + case.C)
                        self.assertLessEqual(defs.relative_error(a.sum(axis=tuple(range(len(case.W)))), b), TOL)
            off_diagonal = tuple(min(j, n - 1) for j, n in enumerate(case.W))         # (0, 1): not on the diagonal
            for w in ((0,) * len(case.W), off_diagonal, tuple(n - 1 for n in case.W)):  # the block of one sample
                one = Case.__new__(Case)
                one.__dict__.update(case.__dict__)
                if case.kind == 'entries':
                    one.sample = case.sample[(slice(None),) + w]
                    one.y, one.weight = case.y[w], case.weight[w]
                else:
                    one.sample = [s[(slice(None),) * case.m + w] for s in case.sample]
                    at_w = lambda v: ([a[(slice(None),) * case.m + w] for a in v] if case.kind == 'probe'
                                      else v[(slice(None),) * case.m + w])
                    one.y, one.weight = at_w(case.y), at_w(case.weight)
                one.W = ()
                alone = dict(zip(names, one.run(names)))
                for name in names:
                    with self.subTest(kind=case.kind, sample=repr(w), name=name):
                        for a, b in zip(arrays_of(kept[name]), arrays_of(alone[name])):
                            self.assertLessEqual(defs.relative_error(a[w], b), TOL)


class TestTheReferenceAnchor(unittest.TestCase):
    """The probe kind at ``(2,)``, ``W = (2, 2)``, against the reference module (which has ``W`` and no other stack),
    element by element: ``C = (2, 3)``; ``K = (2, 3)``; and every group multi-axis, on the outputs."""

    def test_against_the_reference(self):
        anchors = ((Case('probe', (2, 3), (), (2, 2), (2,), '_manifold', seed=31), None),
                   (Case('probe', (), (2, 3), (2, 2), (2,), '_manifold', seed=32), None),
                   (Case('probe', (2, 3), (2, 3), (2, 2), (2,), '_manifold', seed=33),
                    ('z', 'r', 'f', 'g', 'Jp', 'Jp_norm_sq', 'JtJp', 'Kp')))
        for case, names in anchors:
            names = tuple(name for name in case.names() if name != 'w') if names is None else names
            got = dict(zip(names, case.run(names)))
            for k in (tuple(np.ndindex(*case.K)) or ((),)):
                for c in (tuple(np.ndindex(*case.C)) or ((),)):
                    arguments = case.arguments(k, c)
                    frame = case.element_at(c)
                    y_c, weight_c = case.at_element(case.y, c), case.at_element(case.weight, c)   # element c's data
                    for name in names:
                        if name == 'r_block_sumsq':
                            continue                    # a sum over C: the matrix checks it against the engine
                        expected = getattr(sdef, name)('probe', frame, case.sample, None, case.order,
                                                       y=y_c, full_weight=weight_c,
                                                       variations=arguments['variations'],
                                                       core_directions=arguments['core_directions'])
                        k_ = k if name in case.with_K() else ()
                        value = got[name]
                        if name in PAIRS:
                            value = tuple(tuple(np.asarray(b)[k_ + c] for b in family) for family in value)
                        elif name in BLOCKS:
                            value = tuple(np.asarray(b)[k_ + c] for b in value)
                        elif name == 'f':
                            value = np.asarray(value)[c]
                        elif name == 'Jp_norm_sq':
                            value = np.asarray(value)[k + c]
                        else:
                            lead = case.m + len(case.W)
                            value = tuple(np.asarray(a)[(slice(None),) * lead + k_ + c] for a in value)
                        with self.subTest(C=repr(case.C), K=repr(case.K), k=repr(k), c=repr(c), name=name):
                            for a, b in zip(arrays_of(value), arrays_of(expected)):
                                self.assertLessEqual(defs.relative_error(a, b), 1e-10)


class TestTheStackCheck(unittest.TestCase):
    """``_check_stacks``: removable with it (O-18, pending the maintainer)."""

    def test_a_stray_stack_raises(self):
        case = Case('probe', (2,), (), (3,), (2,), '_manifold', seed=41)
        D, U, L, R = case.frame
        bad_w = list(case.sample)
        bad_w[2] = bad_w[2][:, :1]                                         # W = (1,) beside (3,)
        with self.assertRaisesRegex(ValueError, r'w\[2\] has shape \(2, 1, 3\), where W = \(3,\)'):
            sv.sampling_variables('probe', case.frame, bad_w, ('z',), order=(2,))
        bad_U = list(U)
        bad_U[1] = U[1][:1]                                                # C = (1,) beside (2,)
        with self.assertRaisesRegex(ValueError, r'U\[1\] has shape \(1, 2, 2, 2\), where C = \(2,\)'):
            sv.sampling_variables('probe', (D, bad_U, L, R), case.sample, ('z',), order=(2,))
        Vc = [np.zeros((2,) + (nU[i], N[i])) for i in range(D_SITES)]
        Vc[0] = Vc[0][0]                                                   # no C at all
        with self.assertRaisesRegex(ValueError, r'Vc\[0\] has shape \(3, 4\), where C = \(2,\)'):
            sv.sampling_variables('probe', case.frame, case.sample, ('z',), order=(2,), Vc=Vc)
        bad_G = list(L)
        bad_G[2] = L[2][:1]
        with self.assertRaisesRegex(ValueError, r'L\[2\] has shape \(1, 3, 2, 2\), where C = \(2,\)'):
            sv.sampling_variables('apply', (D, bad_G), case.sample, ('z',), order=(2,))

    def test_consistent_stacks_pass(self):
        case = Case('probe', (1,), (), (1,), (), '_manifold', seed=42)       # C = (1,) and W = (1,) everywhere
        (z,) = case.run(('z',))
        self.assertEqual(z[0].shape, (1, 1, 4))


def dtype_inputs(dtype, to_jax=False):
    """A frame, input jets, data and directions in ``dtype`` (jax arrays when ``to_jax``). On jax with no jets: eager
    jax runs one dispatch per contraction, and the dtype does not depend on the jet shape."""
    case = Case('probe', (), (2,), (3,), () if to_jax else (2,), '_manifold', seed=51)
    cast = (lambda a: jnp.asarray(np.asarray(a, dtype=dtype))) if to_jax else (lambda a: np.asarray(a, dtype=dtype))
    tree = lambda value: tuple(tree(item) for item in value) if isinstance(value, (tuple, list)) else cast(value)
    return case, tree


class TestTheDtypeMatrix(unittest.TestCase):

    NAMES = ('z', 'r', 'f', 'g', 'Jp', 'Jp_norm_sq', 'JtJp', 'Kp', 'mu')

    def outputs(self, frame, sample, case, tree, names=None, jit=False):
        """Every output of a call wanting ``names`` (all of ``NAMES``); under ``jax.jit`` when ``jit`` (eager jax
        dispatches every contraction on its own, ten times slower here, and the dtype rule is the same)."""
        names = self.NAMES if names is None else names
        arrays = (tree(case.y), tree(case.weight), tree(case.variations), tree(case.core_directions))
        call = lambda frame, sample, y, weight, variations, core_directions: sv.sampling_variables(
            'probe', frame, sample, names, order=case.order, y=y, weight=weight, variations=variations,
            core_directions=core_directions)
        return leaves((jax.jit(call) if jit else call)(frame, sample, *arrays))

    def backends(self):
        yield 'numpy float64', np.float64, False
        yield 'numpy float32', np.float32, False
        if HAS_JAX:
            yield 'jax float32', np.float32, True
            yield 'jax float64', np.float64, True

    def context(self, label):
        import contextlib
        return defs.jax_in_double_precision() if label == 'jax float64' else contextlib.nullcontext()

    def test_every_output_in_the_inputs_dtype(self):
        for label, dtype, to_jax in self.backends():
            with self.context(label):
                case, tree = dtype_inputs(dtype, to_jax)
                outs = self.outputs(tree(case.frame), tree(case.sample), case, tree, jit=to_jax)
                with self.subTest(backend=label):
                    for a in outs:
                        self.assertEqual(np.dtype(a.dtype), np.dtype(dtype))
                        self.assertEqual(HAS_JAX and isinstance(a, jax.Array), to_jax)

    def test_mixed_inputs(self):
        """float32 with float64 gives float32; jax with numpy gives jax; a float64 ``y`` nothing wanted reads does not
        lower or raise anything."""
        case, tree32 = dtype_inputs(np.float32)
        _, tree64 = dtype_inputs(np.float64)
        outs = self.outputs(tree32(case.frame), tree64(case.sample), case, tree64)
        self.assertEqual({np.dtype(a.dtype) for a in outs}, {np.dtype(np.float32)})
        (z,) = sv.sampling_variables('probe', tree32(case.frame), tree32(case.sample), ('z',), order=(2,),
                                     y=tree64(case.y))
        self.assertEqual(z[0].dtype, np.float32)
        if HAS_JAX:
            case, tree_jax = dtype_inputs(np.float32, to_jax=True)
            outs = self.outputs(tree_jax(case.frame), tree32(case.sample), case, tree32, names=('z', 'f', 'mu'))
            self.assertTrue(all(isinstance(a, jax.Array) and a.dtype == np.float32 for a in outs))

    def test_held_variables_of_another_precision(self):
        """A held float64 variable a row reads, in a float32 call, gives float32 (it is counted and cast); one
        neither read nor wanted does not change the dtype; a jax one, unread, still makes the call jax."""
        case, tree32 = dtype_inputs(np.float32)
        _, tree64 = dtype_inputs(np.float64)
        frame, sample = tree32(case.frame), tree32(case.sample)
        (mu64,) = sv.sampling_variables('probe', tree64(case.frame), tree64(case.sample), ('mu',), order=(2,))
        z, mu = sv.sampling_variables('probe', frame, sample, ('z', 'mu'), order=(2,), held=sv.HeldVariables(mu=mu64))
        self.assertEqual((z[0].dtype, mu[0].dtype), (np.dtype(np.float32),) * 2)
        (xi,) = sv.sampling_variables('probe', frame, sample, ('xi',), order=(2,), held=sv.HeldVariables(mu=mu64))
        self.assertEqual(xi[0].dtype, np.float32)
        (xi64,) = sv.sampling_variables('probe', tree64(case.frame), tree64(case.sample), ('xi',), order=(2,),
                                        held=sv.HeldVariables(mu=tuple(np.asarray(m, np.float32) for m in mu64)))
        self.assertEqual(xi64[0].dtype, np.float64)
        if HAS_JAX:
            held = sv.HeldVariables(mu=tuple(jnp.asarray(m) for m in mu64))
            (xi,) = sv.sampling_variables('probe', frame, sample, ('xi',), order=(2,), held=held)
            self.assertTrue(isinstance(xi[0], jax.Array))

    def test_the_index_and_the_one_hot_vectors(self):
        """An integer index does not count; entries' one-hot ``'w'`` is in the result dtype; the one-hot vectors of
        ``index_to_one_hot`` given to apply count like any float array."""
        import t3toolbox.backend.jets as jets
        case = Case('entries', (), (), (3,), (), '_manifold', seed=52)
        frame32 = tuple(tuple(np.asarray(core, np.float32) for core in store) for store in case.frame)
        for index_dtype in (np.int32, np.int64):
            w, z = sv.sampling_variables('entries', frame32, case.sample.astype(index_dtype), ('w', 'z'))
            with self.subTest(index=np.dtype(index_dtype).name):
                self.assertEqual((w[0].dtype, z.dtype), (np.dtype(np.float32),) * 2)
        one_hot = jets.index_to_one_hot(case.sample, N)                     # float64
        (z,) = sv.sampling_variables('apply', frame32, one_hot, ('z',))
        self.assertEqual(z.dtype, np.float32)
        (z,) = sv.sampling_variables('apply', case.frame, tuple(e.astype(np.float32) for e in one_hot), ('z',))
        self.assertEqual(z.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
