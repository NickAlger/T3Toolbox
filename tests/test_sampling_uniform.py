# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_variables`` on the uniform layout: supercores, masks, packed samples.

1. **Equivalence** (`TestEquivalence`): every name of a uniform call, element ``(k, c)`` by element and site by site,
   against the ragged engine on that element's ragged frame, directions, sample and data, on a covering design: the
   nine stack configurations (``_CONFIGS`` of the uniform tests: no stack, ``C``, ``K``, ``K + C``, forced-larger
   padding, forced padding with ``C``, ``C = (2, 3)``, ``K = (2, 3)``; and ranks varying across ``C``) with the four jet
   cases (``()``, a line at ``(2,)``, ``(1, 1)``, modes of different input jet sizes at ``(1, 2)``) once each for probe
   and apply, entries on every configuration at ``()``, the call type alternating; every input carries garbage in every
   masked slot (the padding of every core, of ``V``, ``H``, ``Vc``, ``D_dot``, ``L_dot``, of the mode axis ``N`` of the
   sample and of ``y``); and every output's padding is exactly zero, the real block read off the RAGGED result's shape
   (non-circular). Plus the maximal case per kind, with ``sum_over_samples=False`` on one.
2. **The packed jet zeros are data** (`TestPackedJets`): a nonzero written into a jet slot that packing filled changes
   the result as the ragged call with that slot added to the input jet; ``pack_vectors`` with ``len_P``.
3. **Held variables** (`TestHeld`): held variables with garbage padding give the clean result (``f`` included).
4. **The index** (`TestTheIndex`): a numpy index out of ``[-N_i, N_i)`` raises as on ragged; a negative one counts
   from ``N_i``; entries' one-hot ``'w'`` has the padded length.
5. **``Kp`` at TT slack** (`TestKpAtTTSlack`): a frame whose masks are a strict subset of its point's; garbage in the
   directions' padding leaves the real blocks alone, and with the directions' mask removed the test turns red (a
   permanent negative control); likewise the exit mask.
6. **The stack check's uniform branch** (`TestTheUniformStackCheck`): a wrong ``d``, a wrong ``N``, a size-1 ``C`` on
   one supercore, each alone.

numpy only; the jit and sharding cases are in ``test_dispatch.py`` and ``test_contractions_sharding.py``.
"""
import itertools
import math
import os
import sys
import unittest
import unittest.mock

import numpy as np

import t3toolbox.backend.jets as jets
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.common import prefix_mask

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampling_definitions as defs

TOL = 1e-12
GARBAGE = 1e3
D_SITES = 3
N, nD, nU, rL, rR = (4, 5, 3), (2, 3, 2), (3, 2, 3), (2, 2, 3, 2), (1, 3, 2, 3)     # tails at both ends of L
SMALLER = dict(nD=(1, 2, 2), nU=(2, 2, 1), rL=(2, 1, 2, 2), rR=(1, 2, 2, 3))          # another element's ranks
CONFIGS = (((), (), False, False), ((2,), (), False, False), ((), (3,), False, False), ((2,), (3,), False, False),
           ((), (), True, False), ((2,), (), True, False), ((2, 3), (), False, False), ((), (2, 3), False, False),
           ((3,), (), False, True))                                                 # (C, K, force_pad, varying)
JET_CASES = (((), None), ((2,), (2,)), ((1, 1), (2, 2)), ((1, 2), ((3, 2), (1, 1), (2, 3))))   # (order, F_in)
BLOCKS = ('V_tilde', 'H_tilde', 'V_tilde_dot', 'H_tilde_dot')
BLOCK_PAIRS = {'g': ('V_tilde', 'H_tilde'), 'JtJp': ('V_tilde', 'H_tilde'), 'Kp': ('V_tilde_dot', 'H_tilde_dot')}


def ranks_of(c, varying):
    """The ranks of frame element ``c`` (the second and later elements of a varying stack have smaller ones)."""
    if varying and any(c):
        return SMALLER['nD'], SMALLER['nU'], SMALLER['rL'], SMALLER['rR']
    return nD, nU, rL, rR


def core_shapes(ranks):
    """The ragged core shapes of a frame with these ranks, and of its two forms of a direction."""
    n_D, n_U, r_L, r_R = ranks
    d = D_SITES
    return dict(D=[(n_D[i], N[i]) for i in range(d)], U=[(r_L[i], n_U[i], r_R[i + 1]) for i in range(d)],
                L=[(r_L[i], n_D[i], r_L[i + 1]) for i in range(d)], R=[(r_R[i], n_D[i], r_R[i + 1]) for i in range(d)],
                V=[(n_U[i], N[i]) for i in range(d)], H=[(r_L[i], n_D[i], r_R[i + 1]) for i in range(d)],
                D_dot=[(n_D[i], N[i]) for i in range(d)], L_dot=[(r_L[i], n_D[i], r_L[i + 1]) for i in range(d)],
                Vc=[(n_U[i], N[i]) for i in range(d)])


class UniformCase:
    """A uniform call and, per element ``(k, c)``, its ragged twin: random cores of each element's ranks, padded (tight,
    or one more on every axis), the padding filled with garbage."""

    def __init__(self, kind, C, K, force_pad, varying, order, F_in, call, W=(3,), seed=0):
        rng = np.random.default_rng(seed)
        self.kind, self.C, self.K, self.order, self.call, self.W = kind, C, K, order, call, W
        self.m, self.box = len(order), tuple(n + 1 for n in order)
        d = D_SITES
        self.elements = tuple(np.ndindex(*C)) or ((),)
        self.directions_k = tuple(np.ndindex(*K)) or ((),)
        extra = 1 if force_pad else 0
        pads = dict(D=(max(nD) + extra, max(N) + extra), U=(max(rL) + extra, max(nU) + extra, max(rR) + extra),
                    L=(max(rL) + extra, max(nD) + extra, max(rL) + extra),
                    R=(max(rR) + extra, max(nD) + extra, max(rR) + extra))
        pads.update(V=(pads['U'][1], pads['D'][1]), H=(pads['L'][0], pads['D'][0], pads['R'][0]), D_dot=pads['D'],
                    L_dot=pads['L'], Vc=(pads['U'][1], pads['D'][1]))
        self.Np = pads['D'][1]
        # the ragged cores of every element, and the supercores with garbage in their padding
        self.ragged = {}
        for c in self.elements:
            shapes = core_shapes(ranks_of(c, varying))
            self.ragged[c] = {name: [rng.standard_normal(s) for s in shapes[name]]
                              for name in ('D', 'U', 'L', 'R', 'Vc')}
            for k in self.directions_k:
                self.ragged[k, c] = {name: [rng.standard_normal(s) for s in shapes[name]]
                                     for name in ('V', 'H', 'D_dot', 'L_dot')}
        self.supercores = {}
        for name in ('D', 'U', 'L', 'R', 'Vc'):
            self.supercores[name] = self.packed(lambda k, c: self.ragged[c][name], pads[name], (), rng)
        for name in ('V', 'H', 'D_dot', 'L_dot'):
            self.supercores[name] = self.packed(lambda k, c: self.ragged[k, c][name], pads[name], K, rng)
        # the masks, from the ragged ranks
        def family(size_of, length, pad):
            ranks = np.zeros((length,) + C, dtype=int)
            for c in self.elements:
                ranks[(slice(None),) + c] = size_of(ranks_of(c, varying))
            return prefix_mask(ranks, pad)
        self.masks = (family(lambda r: r[0], d, pads['D'][0]), family(lambda r: r[1], d, pads['U'][1]),
                      family(lambda r: r[2], d + 1, pads['L'][0]), family(lambda r: r[3], d + 1, pads['R'][0]))
        # the sample, the data and the weight; y and the weight of the type of z, one per entry and per element of
        # the stack C (the data-metric rule): each element's ragged twin gets its own slice
        if kind == 'entries':
            self.index = np.stack([rng.integers(-N[i], N[i], size=W) for i in range(d)])
            self.ragged_sample, self.sample = self.index, self.index
            self.y, self.weight = rng.standard_normal(W + C), 0.5 + rng.random(W + C)
            self.ragged_weight = self.weight
        else:
            sizes = [(F_in if F_in is None or isinstance(F_in[0], int) else F_in[i]) or () for i in range(d)]
            self.ragged_sample = [rng.standard_normal(tuple(sizes[i]) + W + (N[i],)) for i in range(d)]
            packed = np.asarray(ut3_operations.pack_if_ragged(self.ragged_sample, self.Np, len_P=self.m))
            garbage = GARBAGE * rng.standard_normal(packed.shape)
            garbage[..., :min(N)] = 0.0
            for i in range(d):
                packed[i, ..., N[i]:] = garbage[i, ..., N[i]:]                    # the mode padding only
            self.sample = packed
            if kind == 'probe':                         # packed, finite garbage in the mode padding of both
                self.ragged_y = [rng.standard_normal(self.box + W + C + (N[i],)) for i in range(d)]
                self.ragged_weight = [0.5 + rng.random(self.box + W + C + (N[i],)) for i in range(d)]
                self.y, self.weight = (self.garbage_padded(ragged, rng) for ragged in (self.ragged_y,
                                                                                       self.ragged_weight))
            else:
                self.ragged_y = self.y = rng.standard_normal(self.box + W + C)
                self.ragged_weight = self.weight = 0.5 + rng.random(self.box + W + C)
        if kind == 'entries':
            self.ragged_y = self.y

    def garbage_padded(self, ragged, rng):
        """A probe data-space object packed to ``Np``, finite garbage in the mode padding (masked by the engine)."""
        packed = np.asarray(ut3_operations.pack_if_ragged(ragged, self.Np))
        for i in range(D_SITES):
            packed[i, ..., N[i]:] = GARBAGE * rng.standard_normal(packed[i, ..., N[i]:].shape)
        return packed

    def at_element(self, value, c):
        """Element ``c`` of a data-space object of the type of ``z``: its ``C`` axes (after ``F + W``) taken."""
        where = (slice(None),) * (self.m + len(self.W)) + tuple(c)
        if self.kind == 'probe':
            return [v[where] for v in value]
        return value[(slice(None),) * len(self.W) + tuple(c)] if self.kind == 'entries' else value[where]

    def packed(self, ragged_of, pad, K, rng):
        """A supercore ``(d,) + K + C + pad`` of the ragged cores, garbage in every slot they do not fill."""
        out = GARBAGE * rng.standard_normal((D_SITES,) + K + self.C + pad)
        for k in (tuple(np.ndindex(*K)) or ((),)):
            for c in self.elements:
                for i, core in enumerate(ragged_of(k, c)):
                    out[(i,) + k + c + tuple(slice(0, s) for s in core.shape)] = core
        return out

    @property
    def frame(self):
        s = self.supercores
        if self.call == '_manifold':
            return (s['D'], s['U'], s['L'], s['R'], N, self.masks)
        return (s['D'], s['L'], N, (self.masks[0], self.masks[2]))

    def ragged_frame(self, c):
        r = self.ragged[c]
        return (r['D'], r['U'], r['L'], r['R']) if self.call == '_manifold' else (r['D'], r['L'])

    def arguments(self):
        s = self.supercores
        arguments = dict(order=self.order, y=self.y, weight=self.weight, core_directions=(s['D_dot'], s['L_dot']))
        if self.call == '_manifold':
            arguments.update(variations=(s['V'], s['H']), Vc=s['Vc'])
        return arguments

    def ragged_arguments(self, k, c):
        r = self.ragged[k, c]
        arguments = dict(order=self.order, y=self.at_element(self.ragged_y, c),
                         weight=self.at_element(self.ragged_weight, c), core_directions=(r['D_dot'], r['L_dot']))
        if self.call == '_manifold':
            arguments.update(variations=(r['V'], r['H']), Vc=self.ragged[c]['Vc'])
        return arguments

    def names(self):
        return tuple(name for name in sv._names(self.kind, self.call) if name != 'w' or self.kind == 'entries')

    def with_K(self):
        return {rule.name for rule in sv._GRAPHS[self.kind, self.call] if rule.carries_K}


def check_real_and_zero(test, uniform, ragged, label):
    """``uniform`` (padded) has ``ragged`` in its leading block, read off ``ragged``'s shape, and exactly zero
    elsewhere."""
    uniform = np.asarray(uniform)
    ragged = np.asarray(ragged)
    test.assertEqual(uniform.ndim, ragged.ndim, label)
    real = tuple(slice(0, s) for s in ragged.shape)
    test.assertLessEqual(defs.relative_error(uniform[real], ragged), TOL, label)
    rest = uniform.copy()
    rest[real] = 0.0
    test.assertEqual(float(np.abs(rest).max(initial=0.0)), 0.0, label + ': the padding is not zero')


def compare(test, case, name, value, ragged_by_element, keep_W=False):
    """Compare a uniform output with the ragged ones, element by element and site by site."""
    with_K = case.with_K()
    lead_P = case.m
    if name == 'w':                                 # entries' one-hot vectors: (d,) + W + (N,), no stack
        for i, want in enumerate(next(iter(ragged_by_element.values()))):
            check_real_and_zero(test, np.asarray(value)[i], want, 'w %d' % i)
        return
    for (k, c), ragged in ragged_by_element.items():
        k_ = k if name in with_K else ()
        label = '%s k=%r c=%r' % (name, k, c)
        lead = (slice(None),) + ((slice(None),) * len(case.W) if keep_W else ())   # a block: (d,) + [W] + [K] + C
        if name in BLOCK_PAIRS:
            for family, (got, want) in enumerate(zip(value, ragged)):
                block = np.asarray(got)[lead + k_ + c]
                for i in range(D_SITES):
                    check_real_and_zero(test, block[i], want[i], label + ' family %d site %d' % (family, i))
        elif name in BLOCKS:
            block = np.asarray(value)[lead + k_ + c]
            for i in range(D_SITES):
                check_real_and_zero(test, block[i], ragged[i], label + ' site %d' % i)
        elif name in ('f',):
            test.assertLessEqual(defs.relative_error(np.asarray(value)[c], ragged), TOL, label)
        elif name == 'Jp_norm_sq':
            test.assertLessEqual(defs.relative_error(np.asarray(value)[k + c], ragged), TOL, label)
        elif name == 'r_block_sumsq':
            continue                                # a sum over C: below
        elif isinstance(ragged, tuple):             # an indexed variable: (lead,) + P + W + [K] + C + (size,)
            array = np.asarray(value)
            taken = array[(slice(None),) * (1 + lead_P + len(case.W)) + k_ + c]
            for i in range(len(ragged)):
                check_real_and_zero(test, taken[i], ragged[i], label + ' index %d' % i)
        else:                                       # the apply and entries kinds' one array: P + W + [K] + C
            array = np.asarray(value)
            taken = array[(slice(None),) * (lead_P + len(case.W)) + k_ + c]
            test.assertLessEqual(defs.relative_error(taken, ragged), TOL, label)
    if name == 'r_block_sumsq':
        k0 = next(iter(ragged_by_element))[0]
        expected = sum(ragged_by_element[k0, c] for c in case.elements)
        test.assertLessEqual(defs.relative_error(value, expected), TOL, name)


def run_and_compare(test, case, sum_over_samples=True):
    names = case.names()
    got = dict(zip(names, sv.sampling_variables(case.kind, case.frame, case.sample, names,
                                                sum_over_samples=sum_over_samples, **case.arguments())))
    ragged = {}
    for k in case.directions_k:
        for c in case.elements:
            ragged[k, c] = dict(zip(names, sv.sampling_variables(case.kind, case.ragged_frame(c), case.ragged_sample,
                                                                 names, sum_over_samples=sum_over_samples,
                                                                 **case.ragged_arguments(k, c))))
    for name in names:
        with test.subTest(kind=case.kind, call=case.call, C=repr(case.C), K=repr(case.K), order=repr(case.order),
                          W=repr(case.W), name=name):
            compare(test, case, name, got[name], {key: value[name] for key, value in ragged.items()},
                    keep_W=not sum_over_samples)


def covering_cases():
    cases = []
    for kind in ('probe', 'apply'):
        for (s, config), (j, (order, F_in)) in itertools.product(enumerate(CONFIGS), enumerate(JET_CASES)):
            call = '_manifold' if (s + j) % 2 == 0 else '_corewise'
            cases.append(UniformCase(kind, *config, order, F_in, call, seed=100 * s + 10 * j))
    for s, config in enumerate(CONFIGS):
        cases.append(UniformCase('entries', *config, (), None, '_manifold' if s % 2 == 0 else '_corewise', seed=7 + s))
    return cases


class TestEquivalence(unittest.TestCase):

    def test_the_covering_design(self):
        cases = covering_cases()
        self.assertEqual(len(cases), 81)
        for case in cases:
            run_and_compare(self, case)

    def test_the_maximal_case_of_each_kind(self):
        """Forced padding, ranks varying across ``C = (2, 3)``... as far as the generator varies them (every element
        but the first smaller), ``K = (2, 3)``, ``W = (2, 2)``, two jet axes; unsummed on the apply kind."""
        for kind, order, F_in in (('probe', (1, 1), (2, 2)), ('apply', (1, 1), (2, 2)), ('entries', (), None)):
            case = UniformCase(kind, (2, 3), (2, 3), True, True, order, F_in, '_manifold', W=(2, 2), seed=99)
            run_and_compare(self, case, sum_over_samples=kind != 'apply')


class TestPackedJets(unittest.TestCase):

    def test_a_packing_zero_is_a_coefficient(self):
        """At ``(1, 2)`` with modes of input jet sizes ``(3, 2)``, ``(1, 1)``, ``(2, 3)``: writing ``s`` into the slot
        ``(1, 1)`` of mode 1, which packing filled with zero, gives the ragged call whose mode 1 has that slot."""
        case = UniformCase('probe', (), (), False, False, (1, 2), ((3, 2), (1, 1), (2, 3)), '_manifold', seed=5)
        sample = case.sample.copy()
        rng = np.random.default_rng(6)
        slot = rng.standard_normal(sample[1, 1, 1].shape)
        sample[1, 1, 1] = slot
        names = ('z', 'g')
        got = sv.sampling_variables('probe', case.frame, sample, names, **case.arguments())
        w1 = np.zeros((2, 2) + case.W + (N[1],))
        w1[0, 0] = case.ragged_sample[1][0, 0]
        w1[1, 1] = slot[..., :N[1]]
        ragged_sample = [case.ragged_sample[0], w1, case.ragged_sample[2]]
        want = sv.sampling_variables('probe', case.ragged_frame(()), ragged_sample, names,
                                     **case.ragged_arguments((), ()))
        compare(self, case, 'z', got[0], {((), ()): want[0]})
        compare(self, case, 'g', got[1], {((), ()): want[1]})
        unchanged = sv.sampling_variables('probe', case.frame, case.sample, ('z',), **case.arguments())[0]
        self.assertGreater(defs.relative_error(np.asarray(got[0]), np.asarray(unchanged)), 1e-3)

    def test_a_ragged_sample_is_packed_by_the_engine(self):
        """A uniform call given the ragged input jets of modes of different jet sizes packs them itself (zero
        coefficients on the jet axes, zero mode padding): the same as the call given them packed."""
        case = UniformCase('probe', (2,), (), True, False, (1, 2), ((3, 2), (1, 1), (2, 3)), '_corewise', seed=9)
        names = ('z', 'g', 'xi')
        packed = np.asarray(ut3_operations.pack_if_ragged(case.ragged_sample, case.Np, len_P=2))
        given_packed = sv.sampling_variables('probe', case.frame, packed, names, **case.arguments())
        given_ragged = sv.sampling_variables('probe', case.frame, case.ragged_sample, names, **case.arguments())
        for a, b in zip(defs_arrays(given_ragged), defs_arrays(given_packed)):
            self.assertTrue(np.array_equal(a, b))

    def test_pack_vectors_with_jet_axes(self):
        rng = np.random.default_rng(8)
        modes = [rng.standard_normal((3, 2, 4, 5)), rng.standard_normal((1, 1, 4, 3)),
                 rng.standard_normal((2, 3, 4, 2))]
        packed = ut3_operations.pack_vectors(modes, len_P=2)
        self.assertEqual(packed.shape, (3, 3, 3, 4, 5))
        for i, v in enumerate(modes):
            real = (i,) + tuple(slice(0, s) for s in v.shape)
            self.assertTrue(np.array_equal(packed[real], v))
            rest = packed[i].copy()
            rest[tuple(slice(0, s) for s in v.shape)] = 0.0
            self.assertFalse(rest.any())
        self.assertIs(ut3_operations.pack_if_ragged(packed, len_P=2), packed)
        old = [rng.standard_normal((4, 5)), rng.standard_normal((4, 3))]
        self.assertTrue(np.array_equal(ut3_operations.pack_vectors(old), ut3_operations.pack_vectors(old, len_P=0)))
        for len_P in (0, 1):
            with self.subTest(len_P=len_P):
                with self.assertRaisesRegex(ValueError, 'every vector must have the stack shape of the first'):
                    ut3_operations.pack_vectors([np.ones((2, 4, 5)), np.ones((2, 3, 3))], len_P=len_P)


class TestHeld(unittest.TestCase):

    def test_held_variables_with_garbage_padding(self):
        """Held ``mu``, ``r`` and ``z`` with garbage in their padding give the clean outputs, ``f`` included."""
        case = UniformCase('probe', (2,), (), True, False, (2,), (2,), '_manifold', seed=12)
        names = ('f', 'g', 'z')
        clean = sv.sampling_variables('probe', case.frame, case.sample, names, **case.arguments())
        mu, r, z = sv.sampling_variables('probe', case.frame, case.sample, ('mu', 'r', 'z'), **case.arguments())
        rng = np.random.default_rng(13)

        def dirty(array, real_sizes):               # garbage beyond each index's real size on the last axis
            array = np.array(array)
            for i, size in enumerate(real_sizes):
                array[i, ..., size:] = GARBAGE * rng.standard_normal(array[i, ..., size:].shape)
            return array
        for held in (sv.HeldVariables(mu=dirty(mu, rL)), sv.HeldVariables(r=dirty(r, N)),
                     sv.HeldVariables(z=dirty(z, N))):
            got = sv.sampling_variables('probe', case.frame, case.sample, names, held=held, **case.arguments())
            with self.subTest(held=held.keys()[0]):
                for a, b in zip(defs_arrays(got), defs_arrays(clean)):
                    self.assertLessEqual(defs.relative_error(a, b), TOL)


def defs_arrays(value):
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in defs_arrays(item)]
    return [np.asarray(value)]


class TestTheIndex(unittest.TestCase):

    def test_out_of_range_raises_as_on_ragged(self):
        case = UniformCase('entries', (), (), True, False, (), None, '_manifold', seed=14)
        for bad in (N[1], -N[1] - 1):
            index = case.index.copy()
            index[1, 0] = bad
            with self.subTest(index=bad):
                with self.assertRaises(IndexError):
                    sv.sampling_variables('entries', case.ragged_frame(()), index, ('z',))
                with self.assertRaisesRegex(IndexError, 'out of range for mode 1 of size %d' % N[1]):
                    sv.sampling_variables('entries', case.frame, index, ('z',))

    def test_the_index_has_one_row_per_mode_on_both_layouts(self):
        """A leading 1 would broadcast to every mode on uniform (``T[i, i, i]``), a leading 4 be silently cut on
        ragged: the length ``d`` is checked on both layouts."""
        case = UniformCase('entries', (), (), True, False, (), None, '_manifold', seed=18)
        for rows in (1, 2, 4):
            index = np.resize(case.index, (rows,) + case.index.shape[1:])
            for layout, frame in (('ragged', case.ragged_frame(())), ('uniform', case.frame)):
                with self.subTest(rows=rows, layout=layout):
                    with self.assertRaisesRegex(ValueError, r'one row per mode, shape \(d,\) \+ W with d = 3'):
                        sv.sampling_variables('entries', frame, index, ('z',))

    def test_one_hot_w_has_the_padded_length(self):
        case = UniformCase('entries', (), (), True, False, (), None, '_corewise', seed=15)
        (w,) = sv.sampling_variables('entries', case.frame, case.index, ('w',))
        self.assertIsInstance(w, np.ndarray)                                # packed, not a tuple of d vectors
        self.assertEqual(np.shape(w), (D_SITES,) + case.W + (case.Np,))
        expected = jets.index_to_one_hot(case.index, N)
        for i in range(D_SITES):
            check_real_and_zero(self, w[i], expected[i], 'w %d' % i)


class TestKpAtTTSlack(unittest.TestCase):
    """At TT slack the frame's ``D`` and ``L`` masks are a strict subset of its point's, so a direction from the point
    has values in slots that are padding for the frame (here: garbage, which includes them). The directions take the
    FRAME's masks: then ``Kp`` and the dot variables it reads equal ragged, with zero padding. The masks are layered
    (every input masked on entry, every output on exit; the real blocks are right as long as every OTHER operand is
    masked), so each permanent negative control removes one entry family AND the exit mask: without the directions'
    masks the frame's padding of ``mu_dot`` is not zero; without the sample's mask on ``N``, that of ``Kp``."""

    NAMES = ('Kp', 'mu_dot', 'nu_L_tilde_dot')

    def setUp(self):
        self.case = UniformCase('probe', (2,), (), True, True, (2,), (2,), '_manifold', seed=16)

    def check(self):
        got = sv.sampling_variables('probe', self.case.frame, self.case.sample, self.NAMES, **self.case.arguments())
        for name, value in zip(self.NAMES, got):
            ragged = {((), c): sv.sampling_variables('probe', self.case.ragged_frame(c), self.case.ragged_sample,
                                                     (name,), **self.case.ragged_arguments((), c))[0]
                      for c in self.case.elements}
            compare(self, self.case, name, value, ragged)

    def test_the_directions_take_the_frames_masks(self):
        self.check()

    def test_without_the_directions_and_exit_masks_the_padding_is_not_zero(self):
        """The directions' masks are the first layer: with them alone (no exit mask) everything is right; without them
        and without the exit mask, the directions' values in the frame's padding reach the padding of ``mu_dot``."""
        identity = lambda name, value, setting: value
        with unittest.mock.patch.object(sv, '_mask_variable', identity):
            self.check()                                                    # the entry masks alone suffice
            table = dict(sv._UNIFORM_MASKS, D_dot=(), L_dot=())
            with unittest.mock.patch.object(sv, '_UNIFORM_MASKS', table):
                with self.assertRaisesRegex(AssertionError, 'the padding is not zero'):
                    self.check()

    def test_without_the_sample_and_exit_masks_the_padding_is_not_zero(self):
        """The masks are layered: the exit mask is the second layer behind the sample's mask on ``N`` (with the sample
        masked, the ``V`` readouts' padded columns are already zero). Without both, the garbage in the sample's mode
        padding reaches the padding of ``Kp``."""
        table = dict(sv._UNIFORM_MASKS, w=())
        with unittest.mock.patch.object(sv, '_UNIFORM_MASKS', table):
            self.check()                                                    # the exit mask alone suffices
            with unittest.mock.patch.object(sv, '_mask_variable', lambda name, value, setting: value):
                with self.assertRaisesRegex(AssertionError, 'the padding is not zero'):
                    self.check()


class TestTheUniformStackCheck(unittest.TestCase):
    """The uniform branch of ``_check_stacks`` (removable with it): each family alone."""

    def setUp(self):
        self.case = UniformCase('probe', (2,), (), False, False, (), None, '_manifold', seed=17)

    def test_a_wrong_d_or_N_of_the_packed_sample(self):
        sample = self.case.sample
        with self.assertRaisesRegex(ValueError, "the packed sample must have the frame's d and N"):
            sv.sampling_variables('probe', self.case.frame, sample[:2], ('z',))
        with self.assertRaisesRegex(ValueError, "the packed sample must have the frame's d and N"):
            sv.sampling_variables('probe', self.case.frame, sample[..., :1], ('z',))

    def test_a_size_1_C_on_one_supercore(self):
        D, U, L, R, shape, masks = self.case.frame
        with self.assertRaisesRegex(ValueError, r'U\[0\] has shape'):
            sv.sampling_variables('probe', (D, U[:, :1], L, R, shape, masks), self.case.sample, ('z',))

    def test_a_mask_of_another_C(self):
        D, U, L, R, shape, (down, up, left, right) = self.case.frame
        with self.assertRaisesRegex(ValueError, 'the up mask has shape'):
            sv.sampling_variables('probe', (D, U, L, R, shape, (down, up[:, :1], left, right)), self.case.sample,
                                  ('z',))


if __name__ == '__main__':
    unittest.main()
