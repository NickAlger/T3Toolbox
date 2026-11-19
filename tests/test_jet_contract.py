# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``contractions.jet_contract`` and ``contractions.GroupLengths``.

``jet_contract`` is ``contract`` with the jet group ``J`` multiplying as truncated polynomials in ``m`` jet
variables: each operand with ``J`` is a polynomial with jet sizes of its own (zero beyond them), and the result
is the product on the box ``order + 1``, or its corner coefficient when the output has no ``J``. Lines of
evidence, each independent of the others:

1. **The reference** (`TestAgainstReference`): every one of the 29 strings of the sampling terms against a
   reference that uses neither ``contract`` nor ``jet_contract`` (``_jet_contract_cases.reference``: every
   stack flattened to one axis, explicit loops over the multi-indices of every jet operand), at ``m = 0, 1, 2``
   and on a few strings at ``m = 3``; with the library's operand shapes and with random jet sizes per operand
   (below, at and above the box, and 1); ``W``, ``K``, ``C`` empty and not; both forms over the orders. And at
   ``m = 1`` against ``numpy.polynomial``, an arithmetic that shares nothing with either. Tolerance tests.
2. **The cases of section 2.4 of the plan** (`TestTheCases`): slots beyond the box are not read; an operand
   with fewer slots equals it padded with zeros; a jet axis of size 1; a box or a corner beyond every operand;
   the pairing the dual jets rest on.
3. **The two forms** (`TestTheTwoForms`): ``'scan'`` and ``'loop'`` agree to rounding wherever a full partner is
   taken one coefficient at a time (``m = 1``), at orders 1 to 16 in float64 and float32; elsewhere, and at
   ``m = 2``, they make the same contractions and give the same arrays.
4. **Sensitivity** (`TestSensitivity`): a perturbation of ONE slot of ONE operand moves the result away from
   the reference by far more than the tolerance of (1), for every slot that can reach the result, and moves it
   not at all for every slot that cannot. So (1) would fail if any coefficient were dropped or misplaced.
5. **Strictness** (`TestStrictness`): a stray or a missing axis on any operand raises, wherever it is; a wrong
   record, a wrong jet shape, an empty jet axis raise.
6. **Structure** (`TestStructure`): exact. How many ``contract`` calls are made (one at ``order = ()``, on the
   string as written); the dtype; numpy in, numpy out; the errors of a malformed string or call.
7. `TestGroupLengths`: the record is a value-hashed static.
8. **The order inside a step** (`TestOrderInsideAStep`): which two operands one step's contraction pairs
   first, read off the path each backend uses. It holds an EXPECTED FAILURE, a marker for Part 2 (row 5.5, the
   contraction order): on numpy most product steps contract the carried jet with the core first, which is the
   costly order.

The numerical tests are numpy-only, by the house convention (`TestOrderInsideAStep` reads jax's contraction
path where jax is installed; it compares no numbers); jax dispatch (jit, jax in gives jax out, the scan is one
``lax.scan``) is in ``test_dispatch.py``, sharding in ``test_contractions_sharding.py``, and the strings
``jet_contract`` hands to ``contract`` in ``test_contractions_interpreter.py``.
"""
import dataclasses as dc
import itertools
import os
import sys
import unittest
import unittest.mock

import numpy as np

import t3toolbox.backend.common as common
import t3toolbox.backend.contractions as contractions
from t3toolbox.backend.contractions import GroupLengths, jet_contract

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _jet_contract_cases as cases     # the 29 strings, their operand shapes and their reference: test-only

W_SHAPES = ((), (3,), (2, 2))
K_SHAPES = ((), (2,), (2, 2))
C_SHAPES = ((), (2,))
FORMS = ('scan', 'loop')

TOL = 1e-12                             # float64: against the reference, and the two forms against each other
TOL_32 = 1e-5                           # float32

ORDER_INSIDE_A_STEP = 4                 # TestOrderInsideAStep's order, as P1 pinned it

RANK_MESSAGE = r"jet_contract\('[A-Za-z,]+->[A-Za-z]*'\): operand %d \(term '[A-Za-z]+'\) has ndim \d+, but"


def lens_of(order, shape_of):
    return GroupLengths(J=len(order), W=len(shape_of['W']), K=len(shape_of['K']), C=len(shape_of['C']))


def stack_matrix(string):
    """(W, K, C) for one string: every combination, with K only where the string has a direction."""
    Ks = K_SHAPES if 'K' in string else ((),)
    return itertools.product(W_SHAPES, Ks, C_SHAPES)


def random_sizes(rng, string, order):
    """Jet sizes per operand drawn below, at and above the box, 1 included."""
    return {p: tuple(int(rng.integers(1, n + 4)) for n in order) for p in cases.jet_operands(string)}


class TestTheStrings(unittest.TestCase):
    """The table of strings is the one the terms have: 29 strings in seven families, over the groups J, W, K, C
    and no others."""

    def test_29_strings_in_seven_families(self):
        self.assertEqual({'xi': 5, 'z': 3, 'mu': 4, 'nu': 4, 'eta': 4, 'V': 5, 'H': 4},
                         {family: len(strings) for family, strings in cases.STRINGS.items()})
        strings = [string for _, string in cases.ALL_STRINGS]
        self.assertEqual(29, len(strings))
        self.assertEqual(29, len(set(strings)))
        self.assertEqual(set('JWKC'), {ch for s in strings for ch in s if ch.isupper()})
        self.assertEqual(12, sum(cases.is_product(s) for s in strings))         # mu, nu, eta
        self.assertEqual(9, sum(cases.is_readout(s) for s in strings))          # V, H
        self.assertEqual(8, sum(len(cases.jet_operands(s)) == 1 and not cases.is_readout(s) for s in strings))
        for s in strings:                                                       # J leads every term that has it
            for term in cases.input_terms(s) + [cases.output_term(s)]:
                self.assertTrue('J' not in term or term[0] == 'J', s)


class TestAgainstReference(unittest.TestCase):

    def _check(self, string, arrays, order, shape_of, forms=FORMS, out_order=None):
        out_order = order if out_order is None else out_order
        ref = cases.reference(string, arrays, out_order, shape_of)
        for form in forms:
            got = jet_contract(string, *arrays, lens=lens_of(out_order, shape_of), order=out_order, over_orders=form)
            self.assertEqual(ref.shape, got.shape)                              # structural: exact
            self.assertLess(cases.relerr(got, ref), TOL)

    def test_every_string_without_jets(self):
        rng = np.random.default_rng(0)
        for family, string in cases.ALL_STRINGS:
            for W, K, C in stack_matrix(string):
                with self.subTest(string=string, W=W, K=K, C=C):
                    shape_of = cases.shape_of_stacks(string, W, K, C)
                    self._check(string, cases.build(string, {}, shape_of, rng), (), shape_of)

    def test_every_string_one_variable_library_shapes(self):
        rng = np.random.default_rng(1)
        n_checked = 0
        for order in ((1,), (3,)):
            for family, string in cases.ALL_STRINGS:
                for label, sizes, out_order in cases.library_shapes(family, string, order):
                    for W, K, C in stack_matrix(string):
                        with self.subTest(string=string, order=order, shapes=label, W=W, K=K, C=C):
                            shape_of = cases.shape_of_stacks(string, W, K, C)
                            arrays = cases.build(string, sizes, shape_of, rng)
                            self._check(string, arrays, order, shape_of, out_order=out_order)
                            n_checked += 1
        self.assertGreater(n_checked, 400)

    def test_every_string_random_jet_sizes(self):
        """At m = 1 and m = 2, each jet operand with sizes of its own, below, at and above the box, and 1."""
        rng = np.random.default_rng(2)
        for order in ((1,), (3,), (1, 2), (2, 0)):
            for family, string in cases.ALL_STRINGS:
                for trial in range(3):
                    sizes = random_sizes(rng, string, order)
                    with self.subTest(string=string, order=order, sizes=sizes):
                        shape_of = cases.shape_of_stacks(string, (2,), (2,), (2,))
                        self._check(string, cases.build(string, sizes, shape_of, rng), order, shape_of)

    def test_every_string_two_variables_library_shapes(self):
        rng = np.random.default_rng(3)
        for order in ((1, 2), (2, 0)):
            for family, string in cases.ALL_STRINGS:
                for label, sizes, out_order in cases.library_shapes(family, string, order):
                    for W, K, C in (((), (), ()), ((3,), (2,), (2,)), ((2, 2), (2, 2), (2,))):
                        with self.subTest(string=string, order=order, shapes=label, W=W, K=K, C=C):
                            shape_of = cases.shape_of_stacks(string, W, K, C)
                            arrays = cases.build(string, sizes, shape_of, rng)
                            self._check(string, arrays, order, shape_of, out_order=out_order)

    def test_three_variables_on_a_few_strings(self):
        rng = np.random.default_rng(4)
        strings = ['JWo,KCio->JWKCi', 'JWCa,KCaib,JWCi->JWKCb', 'Caib,JWKCi,JWCb->JWKCa', 'JWKCi,JWo->KCio',
                   'JWCa,JWCi,JWKCb->KCaib']
        for order in ((1, 0, 2), (1, 1, 1)):
            for string in strings:
                for sizes in ({p: cases.box_of(order) for p in cases.jet_operands(string)},
                              random_sizes(rng, string, order)):
                    with self.subTest(string=string, order=order, sizes=sizes):
                        shape_of = cases.shape_of_stacks(string, (2,), (2,), (2,))
                        self._check(string, cases.build(string, sizes, shape_of, rng), order, shape_of)

    def test_whitespace_in_the_string_is_allowed(self):
        rng = np.random.default_rng(5)
        string = 'JWCa,Caib,JWCi->JWCb'
        shape_of = dict(W=(3,), K=(), C=(2,))
        arrays = cases.build(string, {0: (4,), 2: (2,)}, shape_of, rng)
        lens = lens_of((3,), shape_of)
        plain = jet_contract(string, *arrays, lens=lens, order=(3,), over_orders='loop')
        spaced = jet_contract('JWCa, Caib, JWCi -> JWCb', *arrays, lens=lens, order=(3,), over_orders='loop')
        self.assertTrue(np.array_equal(plain, spaced))

    def test_beyond_the_29(self):
        """Strings the library does not have, each against the reference under both forms: a single jet read out
        at its corner; a product of three jets (two partners: under 'scan' the first full one is scanned and the
        other looped, the carrier starting as far up as the looped one took); readouts of a pair, a leaf and a
        further full jet, and of four full jets (under 'scan' the first full jet beyond the pair is scanned);
        at m = 1, 2, in float64 and float32 (the zeros the start shift puts in front of the carried jet must be
        of its dtype)."""
        rng = np.random.default_rng(6)
        shape_of = dict(W=(3,), K=(), C=())
        strings = ['JWCi,Cio->WCo', 'JWCa,JWCi,JWCb->JWCaib', 'JWa,Caib,JWi,JWb->JWC',
                   'JWCa,JWCi,JWCb,JWCc->Caibc', 'JWCa,JWCb,JWCc,JWCj->Cabcj']
        line = lambda order: tuple(min(2, n + 1) for n in order)
        for order in ((1,), (2,), (4,), (1, 2)):
            for string in strings:
                jets = cases.jet_operands(string)
                shapes = [{p: cases.box_of(order) for p in jets},
                          {p: (line(order) if p == jets[-1] else cases.box_of(order)) for p in jets}]
                for sizes in shapes:
                    arrays = cases.build(string, sizes, shape_of, rng)
                    for dtype, tol in ((np.float64, TOL), (np.float32, TOL_32)):
                        typed = [x.astype(dtype) for x in arrays]
                        ref = cases.reference(string, typed, order, shape_of)
                        for form in FORMS:
                            with self.subTest(string=string, order=order, sizes=sizes, dtype=np.dtype(dtype).name,
                                              form=form):
                                got = jet_contract(string, *typed, lens=lens_of(order, shape_of), order=order,
                                                   over_orders=form)
                                self.assertEqual(dtype, got.dtype)
                                self.assertEqual(ref.shape, got.shape)
                                self.assertLess(cases.relerr(got, ref), tol)

    def test_against_numpy_polynomial(self):
        """At m = 1, on strings whose operands are scalar polynomials (every group but J empty): the product
        truncated to the box equals ``numpy.polynomial.polynomial.polymul`` truncated, and the readout equals
        its coefficient at the corner. ``polymul`` shares nothing with ``jet_contract`` or the reference."""
        P = np.polynomial.polynomial
        rng = np.random.default_rng(7)
        lens = GroupLengths(J=1, W=0, K=0, C=0)      # W of zero axes: each operand is one polynomial
        for n in (0, 1, 2, 5, 9):
            for sizes in ((n + 1, n + 1), (2, n + 1), (n + 3, 1), (1, 1), (n + 1, n + 1, n + 1), (3, n + 2, 2)):
                polys = [rng.standard_normal(size) for size in sizes]
                full = polys[0]
                for q in polys[1:]:
                    full = P.polymul(full, q)
                full = np.concatenate([full, np.zeros(max(0, n + 1 - full.shape[0]))])
                string = ','.join(['JW'] * len(polys))
                for form in FORMS:
                    with self.subTest(order=n, sizes=sizes, form=form):
                        got = jet_contract(string + '->JW', *polys, lens=lens, order=(n,), over_orders=form)
                        self.assertEqual((n + 1,), got.shape)
                        self.assertLess(cases.relerr(got, full[:n + 1]), TOL)
                        top = jet_contract(string + '->', *polys, lens=lens, order=(n,), over_orders=form)
                        self.assertEqual((), top.shape)
                        self.assertLess(abs(float(top) - full[n]), TOL * max(1.0, abs(full[n])))


class TestTheCases(unittest.TestCase):
    """The easy cases to get wrong (plan, section 2.4), each stated as a test."""

    def test_slots_beyond_the_box_are_not_read(self):
        """NaN in every slot beyond the box, on every jet operand: the result is bit for bit the result with those
        slots cut off. In a product the index of a contribution is the sum of the operands' indices, so a slot
        beyond the box reaches nothing; in a readout one beyond the corner reaches nothing either."""
        rng = np.random.default_rng(0)
        shape_of = dict(W=(3,), K=(2,), C=(2,))
        for order in ((2,), (1, 2)):
            box = cases.box_of(order)
            for family, string in cases.ALL_STRINGS:
                shape_of_here = cases.shape_of_stacks(string, **shape_of)
                for label, sizes, out_order in cases.library_shapes(family, string, order):
                    out_box = cases.box_of(out_order)
                    big = {p: tuple(s + 2 for s in sizes[p]) for p in sizes}
                    arrays = cases.build(string, big, shape_of_here, rng)
                    cut = list(arrays)
                    for p in sizes:                         # NaN beyond the output's box, and the arrays cut to it
                        inside = tuple(slice(0, f) for f in out_box)
                        cut[p] = arrays[p][inside].copy()
                        poisoned = np.full_like(arrays[p], np.nan)
                        poisoned[inside] = arrays[p][inside]
                        arrays[p] = poisoned
                    lens = lens_of(out_order, shape_of_here)
                    for form in FORMS:
                        with self.subTest(string=string, order=order, shapes=label, form=form):
                            got = jet_contract(string, *arrays, lens=lens, order=out_order, over_orders=form)
                            want = jet_contract(string, *cut, lens=lens, order=out_order, over_orders=form)
                            self.assertFalse(np.isnan(got).any())
                            self.assertTrue(np.array_equal(got, want))
            self.assertEqual(len(box), len(order))

    def test_an_operand_with_fewer_slots_is_the_same_operand_padded_with_zeros(self):
        rng = np.random.default_rng(1)
        shape_of = dict(W=(3,), K=(2,), C=(2,))
        for order in ((3,), (1, 2)):
            for family, string in cases.ALL_STRINGS:
                shape_of_here = cases.shape_of_stacks(string, **shape_of)
                sizes = random_sizes(rng, string, order)
                sizes = {p: tuple(min(s, n + 1) for s, n in zip(size, order)) for p, size in sizes.items()}
                arrays = cases.build(string, sizes, shape_of_here, rng)
                padded = list(arrays)
                for p in sizes:
                    widths = [(0, n + 1 - s) for s, n in zip(sizes[p], order)] + [(0, 0)] * (arrays[p].ndim - len(order))
                    padded[p] = np.pad(arrays[p], widths)
                lens = lens_of(order, shape_of_here)
                for form in FORMS:
                    with self.subTest(string=string, order=order, sizes=sizes, form=form):
                        got = jet_contract(string, *arrays, lens=lens, order=order, over_orders=form)
                        want = jet_contract(string, *padded, lens=lens, order=order, over_orders=form)
                        self.assertLess(cases.relerr(got, want), TOL)

    def test_a_jet_axis_of_size_one(self):
        """An axis of size 1 is a variable the operand does not depend on: it counts in the rank, and it is folded
        against a larger axis of the other operand in a readout without being broadcast along it. Against the
        reference, and against the same operand padded with zeros."""
        rng = np.random.default_rng(2)
        shape_of = dict(W=(3,), K=(), C=(2,))
        order = (2, 3)
        for string, sizes in (('JWCa,JWCi,JWCb->Caib', {0: (3, 1), 1: (1, 4), 2: (3, 4)}),
                              ('JWCi,JWCo->Cio', {0: (3, 4), 1: (1, 4)}),
                              ('JWCi,JWCo->Cio', {0: (1, 1), 1: (3, 4)}),
                              ('JWCa,Caib,JWCi->JWCb', {0: (3, 1), 2: (1, 4)}),
                              ('JWCi,Cio->JWCo', {0: (1, 1)})):
            arrays = cases.build(string, sizes, shape_of, rng)
            ref = cases.reference(string, arrays, order, shape_of)
            padded = [np.pad(x, [(0, 3 - x.shape[0]), (0, 4 - x.shape[1])] + [(0, 0)] * (x.ndim - 2))
                      if p in sizes else x for p, x in enumerate(arrays)]
            for form in FORMS:
                with self.subTest(string=string, sizes=sizes, form=form):
                    got = jet_contract(string, *arrays, lens=lens_of(order, shape_of), order=order, over_orders=form)
                    self.assertEqual(ref.shape, got.shape)
                    self.assertLess(cases.relerr(got, ref), TOL)
                    again = jet_contract(string, *padded, lens=lens_of(order, shape_of), order=order, over_orders=form)
                    self.assertLess(cases.relerr(got, again), TOL)

    def test_a_box_or_a_corner_beyond_every_operand(self):
        """Well defined, though the engine never asks for it: a product's box larger than every operand's has
        zeros where no product reaches; a corner no product reaches is zeros, of the output's shape, the
        operands' dtype and numpy type, for a pair and for a lone operand alike."""
        rng = np.random.default_rng(3)
        shape_of = dict(W=(3,), K=(), C=(2,))
        for string, sizes, order in (('JWCa,Caib,JWCi->JWCb', {0: (2,), 2: (2,)}, (5,)),
                                     ('JWCi,Cio->JWCo', {0: (2, 1)}, (3, 2)),
                                     ('JWCa,JWCi,JWCb->Caib', {0: (2,), 1: (2,), 2: (2,)}, (6,)),
                                     ('JWCi,JWCo->Cio', {0: (2, 2), 1: (1, 3)}, (2, 4)),
                                     ('JWCi,Cio->WCo', {0: (2,)}, (2,)),
                                     ('JWCi,Cio->WCo', {0: (3, 1)}, (2, 1))):
            for dtype in (np.float64, np.float32):
                arrays = [x.astype(dtype) for x in cases.build(string, sizes, shape_of, rng)]
                ref = cases.reference(string, arrays, order, shape_of)
                for form in FORMS:
                    with self.subTest(string=string, sizes=sizes, order=order, form=form, dtype=np.dtype(dtype).name):
                        got = jet_contract(string, *arrays, lens=lens_of(order, shape_of), order=order, over_orders=form)
                        self.assertIs(np.ndarray, type(got))
                        self.assertEqual(dtype, got.dtype)
                        self.assertEqual(ref.shape, got.shape)
                        if cases.is_readout(string):
                            self.assertFalse(np.any(got))                   # nothing reaches the corner: zeros
                        else:
                            self.assertLess(cases.relerr(got, ref), TOL_32)
                            for j in range(len(order)):     # exact zeros beyond the reach of every product, axis by axis
                                cut = [slice(None)] * len(order)
                                cut[j] = slice(sum(sizes[p][j] - 1 for p in sizes) + 1, None)
                                self.assertFalse(np.any(got[tuple(cut)]))

    def test_the_pairing_the_dual_jets_rest_on(self):
        """The corner of ``a`` times ``omega^2 b`` flipped on every jet axis is ``sum_k omega_k^2 a_k b_k``: the
        pairing of a jet with a dual jet stored flipped (plan 2.4, row 2.4), at m = 1 and m = 2."""
        rng = np.random.default_rng(4)
        for order in ((3,), (2, 3), (1, 0)):
            m, box = len(order), cases.box_of(order)
            lens = GroupLengths(J=m, W=1, K=0, C=0)
            a, b = rng.standard_normal(box + (5, 2)), rng.standard_normal(box + (5, 2))
            omega = rng.standard_normal(box)
            dual = (omega ** 2).reshape(box + (1, 1)) * b
            dual = dual[(slice(None, None, -1),) * m]
            want = float(np.sum((omega ** 2).reshape(box + (1, 1)) * a * b))
            for form in FORMS:
                with self.subTest(order=order, form=form):
                    got = jet_contract('JWi,JWi->', a, dual, lens=lens, order=order, over_orders=form)
                    self.assertLess(abs(float(got) - want), TOL * max(1.0, abs(want)))


class TestTheTwoForms(unittest.TestCase):
    """'scan' (the carried jet shifted one slot per step) against 'loop' (slices)."""

    def test_forms_agree_wherever_a_full_partner_is_taken_one_coefficient_at_a_time(self):
        """m = 1: the 12 products of two full jets (in the mu and nu strings, with a full leaf) and the 4
        readouts of three full jets (H with a full leaf), at orders 1 to 16."""
        rng = np.random.default_rng(0)
        shape_of_all = dict(W=(3,), K=(2,), C=(2,))
        followers = [(family, string) for family, string in cases.ALL_STRINGS if family in ('mu', 'nu', 'eta', 'H')]
        self.assertEqual(16, len(followers))
        for dtype, tol in ((np.float64, TOL), (np.float32, TOL_32)):
            for n in range(1, 17):
                order = (n,)
                for family, string in followers:
                    with self.subTest(string=string, order=order, dtype=np.dtype(dtype).name):
                        shape_of = cases.shape_of_stacks(string, **shape_of_all)
                        sizes = {p: (n + 1,) for p in cases.jet_operands(string)}
                        arrays = [x.astype(dtype) for x in cases.build(string, sizes, shape_of, rng)]
                        ref = cases.reference(string, arrays, order, shape_of)
                        lens = lens_of(order, shape_of)
                        scan = jet_contract(string, *arrays, lens=lens, order=order, over_orders='scan')
                        loop = jet_contract(string, *arrays, lens=lens, order=order, over_orders='loop')
                        self.assertEqual(scan.dtype, dtype)
                        self.assertEqual(loop.dtype, dtype)
                        self.assertLess(cases.relerr(scan, ref), tol)
                        self.assertLess(cases.relerr(loop, ref), tol)
                        self.assertLess(cases.relerr(scan, np.asarray(loop, dtype=np.float64)), tol)

    def test_the_form_changes_nothing_where_no_full_partner_is_taken(self):
        """At orders of 2 and above with the library's shapes: a ride, a partner of two slots, a readout with no
        third full jet make the same contractions on the same slices under both forms, bit for bit. (At order 1
        a leaf of two slots is as large as the box, and is a full partner.)"""
        rng = np.random.default_rng(1)
        n_checked = 0
        for family, string in cases.ALL_STRINGS:
            for label, sizes, out_order in cases.library_shapes(family, string, (3,)):
                if label == 'full leaf' or family == 'eta':     # a full partner: the forms differ (the test above)
                    continue
                with self.subTest(string=string, shapes=label):
                    shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=(2,))
                    arrays = cases.build(string, sizes, shape_of, rng)
                    lens = lens_of(out_order, shape_of)
                    self.assertTrue(np.array_equal(
                        jet_contract(string, *arrays, lens=lens, order=out_order, over_orders='scan'),
                        jet_contract(string, *arrays, lens=lens, order=out_order, over_orders='loop')))
                    n_checked += 1
        self.assertEqual(25, n_checked)     # every string but the 4 eta strings, once, at its line or leaf shape

    def test_with_two_variables_both_forms_run_the_same_loop(self):
        """m = 2: 'scan' takes the loop (Part 2 may scan over several jet axes), so the arrays are equal, exactly."""
        rng = np.random.default_rng(2)
        for order in ((1, 2), (2, 2)):
            for family, string in cases.ALL_STRINGS:
                for label, sizes, out_order in cases.library_shapes(family, string, order):
                    with self.subTest(string=string, order=order, shapes=label):
                        shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=(2,))
                        arrays = cases.build(string, sizes, shape_of, rng)
                        lens = lens_of(out_order, shape_of)
                        self.assertTrue(np.array_equal(
                            jet_contract(string, *arrays, lens=lens, order=out_order, over_orders='scan'),
                            jet_contract(string, *arrays, lens=lens, order=out_order, over_orders='loop')))


class TestSensitivity(unittest.TestCase):
    """The reference tests have teeth: a relative perturbation of 1e-6 of ONE jet slot of ONE operand (the whole
    operand where it has no jet axes) leaves the reference test's tolerance behind by orders of magnitude.
    Every slot of every operand of every string is tried, with the library's shapes, at m = 1 and m = 2, under
    both forms, so no coefficient can be dropped, doubled or put in the wrong slot without the reference test
    failing.

    A slot that cannot reach the result must not move it at all: one beyond the box, or, in a readout, one whose
    complement to the corner the other operands cannot supply (in the V strings with an input of two slots, the
    low slots of the full jet)."""

    @staticmethod
    def _is_read(string, sizes, p, slot, order):
        """Does slot `slot` of operand p reach the result? Derived from the definition of the truncated product,
        not from the code."""
        if any(k > n for k, n in zip(slot, order)):
            return False                            # beyond the box
        if not cases.is_readout(string):
            return True                             # it meets every other operand's slot 0
        others = [q for q in cases.jet_operands(string) if q != p]
        reach = [sum(sizes[q][j] - 1 for q in others) for j in range(len(order))]
        return all(n - k <= r for n, k, r in zip(order, slot, reach))

    def test_a_perturbation_of_any_slot_of_any_operand_is_caught(self):
        rng = np.random.default_rng(0)
        n_not_read = 0
        for order in ((3,), (1, 2)):
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=(2,))
                for label, sizes, out_order in cases.library_shapes(family, string, order):
                    arrays = cases.build(string, sizes, shape_of, rng)
                    ref = cases.reference(string, arrays, out_order, shape_of)
                    lens = lens_of(out_order, shape_of)
                    for p, term in enumerate(cases.input_terms(string)):
                        slots = list(np.ndindex(*sizes[p])) if 'J' in term else [None]
                        for slot in slots:
                            perturbed = [x.copy() for x in arrays]
                            target = perturbed[p][slot] if slot is not None else perturbed[p]
                            target *= 1.0 + 1e-6                        # in place: a view of the operand
                            is_read = slot is None or self._is_read(string, sizes, p, slot, out_order)
                            for form in FORMS:
                                with self.subTest(string=string, order=order, shapes=label, operand=p, slot=slot,
                                                  form=form, is_read=is_read):
                                    got = jet_contract(string, *perturbed, lens=lens, order=out_order, over_orders=form)
                                    if is_read:
                                        self.assertGreater(cases.relerr(got, ref), 1e4 * TOL)
                                    else:
                                        unperturbed = jet_contract(string, *arrays, lens=lens, order=out_order,
                                                                   over_orders=form)
                                        self.assertTrue(np.array_equal(got, unperturbed))
                                        n_not_read += 1
        # not read, in the two V strings with an input of two slots, slots of the full jet the input cannot
        # complete to the corner: at (3,) slots 0 and 1 (2 strings x 2 slots x 2 forms), at (1, 2) the slots
        # (0, 0) and (1, 0) (2 x 2 x 2). Nowhere else.
        self.assertEqual(8 + 8, n_not_read)


class TestStrictness(unittest.TestCase):
    """Every call passes the whole record of lengths, so every group of every string is pinned: an operand with
    one axis too many or too few raises, wherever the axis is. (With only the lengths a string strictly needs, a
    stray axis beside W or K would be read as one more axis of that stack.)"""

    def test_a_stray_or_a_missing_axis_on_any_operand_raises(self):
        rng = np.random.default_rng(0)
        n_checked = 0
        for order in ((), (3,), (1, 2)):
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(2, 2), K=(2,), C=(2,))
                label, sizes, out_order = cases.library_shapes(family, string, order)[0] if order else ('', {}, ())
                arrays = cases.build(string, sizes, shape_of, rng)
                lens = lens_of(out_order, shape_of)
                for form in FORMS:
                    jet_contract(string, *arrays, lens=lens, order=out_order, over_orders=form)     # the good call
                    for p in range(len(arrays)):
                        bad_operands = [np.expand_dims(arrays[p], axis) for axis in range(arrays[p].ndim + 1)]
                        bad_operands += [np.take(arrays[p], 0, axis) for axis in range(arrays[p].ndim)]
                        for bad in bad_operands:
                            with self.subTest(string=string, order=order, form=form, operand=p, shape=bad.shape):
                                with self.assertRaisesRegex(ValueError, RANK_MESSAGE % p):  # one message, every order
                                    jet_contract(string, *arrays[:p], bad, *arrays[p + 1:], lens=lens, order=out_order,
                                                 over_orders=form)
                                n_checked += 1
        self.assertGreater(n_checked, 4000)

    def test_a_wrong_record_raises(self):
        """A record wrong in W, K or C: the ranks no longer match. A record wrong in J with the jet shape of the
        same length: the ranks no longer match. A record whose J disagrees with the length of the jet shape: the
        jet shape's error."""
        rng = np.random.default_rng(0)
        string = 'JWKCo,Cio->JWKCi'         # W and K stand together on the only operand that has them
        shape_of = dict(W=(2, 2), K=(2,), C=(2,))
        order = (3,)
        arrays = cases.build(string, {0: (4,)}, shape_of, rng)
        good = lens_of(order, shape_of)
        jet_contract(string, *arrays, lens=good, order=order, over_orders='loop')
        for field in ('W', 'K', 'C'):
            for delta in (-1, 1):
                value = getattr(good, field) + delta
                if value < 0:
                    continue
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, 'ndim'):
                        jet_contract(string, *arrays, lens=dc.replace(good, **{field: value}), order=order,
                                     over_orders='loop')
        for J, wrong_order in ((0, ()), (2, (3, 0))):
            with self.subTest(J=J, order=wrong_order):
                with self.assertRaisesRegex(ValueError, 'ndim'):
                    jet_contract(string, *arrays, lens=dc.replace(good, J=J), order=wrong_order, over_orders='loop')
            with self.subTest(J=J, order=order):
                with self.assertRaisesRegex(ValueError, 'entries, but lens.J'):
                    jet_contract(string, *arrays, lens=dc.replace(good, J=J), order=order, over_orders='loop')
        # W and K traded one axis for the other: the same total, so the same ranks. Not an error, and the
        # same contraction (the split inside a run of co-traveling groups is unobservable).
        traded = dc.replace(good, W=good.W - 1, K=good.K + 1)
        self.assertTrue(np.array_equal(jet_contract(string, *arrays, lens=traded, order=order, over_orders='loop'),
                                       jet_contract(string, *arrays, lens=good, order=order, over_orders='loop')))

    def test_the_jet_shape_is_a_tuple_of_nonnegative_integers(self):
        """``order`` is a tuple (an integer or a list raises, before any cache lookup); its entries are Python ints
        or numpy integer scalars, normalized to Python ints, and never a bool, a float or negative."""
        rng = np.random.default_rng(1)
        string = 'JWCa,Caib,JWCi->JWCb'
        shape_of = dict(W=(3,), K=(), C=())
        arrays = cases.build(string, {0: (4,), 2: (2,)}, shape_of, rng)
        lens = lens_of((3,), shape_of)
        want = jet_contract(string, *arrays, lens=lens, order=(3,), over_orders='loop')
        for good in ((np.int64(3),), (np.int32(3),), (np.uint8(3),)):
            with self.subTest(order=good):
                self.assertTrue(np.array_equal(want, jet_contract(string, *arrays, lens=lens, order=good,
                                                                  over_orders='loop')))
        for bad, error in ((3, TypeError), ([3], TypeError), (np.array([3]), TypeError), ((3.0,), TypeError),
                           ((True,), TypeError), ((np.bool_(True),), TypeError), ((-1,), ValueError),
                           ((None,), TypeError), ((3, 1), ValueError), ((), ValueError)):
            with self.subTest(order=bad):
                with self.assertRaises(error):
                    jet_contract(string, *arrays, lens=lens, order=bad, over_orders='loop')
        with self.assertRaisesRegex(TypeError, 'order is the jet shape'):
            jet_contract(string, *arrays, lens=lens, order=3, over_orders='loop')
        # the normalized jet shape, Python ints only, is what keys the cached choice of roles
        recorded, original = [], contractions._jet_roles
        with unittest.mock.patch.object(contractions, '_jet_roles',
                                        lambda *args: recorded.append(args) or original(*args)):
            jet_contract(string, *arrays, lens=lens, order=(np.int64(3),), over_orders='loop')
        (args,) = recorded
        self.assertEqual((3,), args[2])
        self.assertIs(int, type(args[2][0]))

    def test_a_jet_axis_of_size_zero_raises(self):
        lens = GroupLengths(J=2, W=1, K=0, C=0)
        mu, G, xi = np.ones((2, 3, 5, 2)), np.ones((2, 3, 4)), np.ones((2, 0, 5, 3))
        for form in FORMS:
            with self.assertRaisesRegex(ValueError, r"operand 2 \(term 'JWCi'\) has the jet sizes \(2, 0\)"):
                jet_contract('JWCa,Caib,JWCi->JWCb', mu, G, xi, lens=lens, order=(1, 2), over_orders=form)


class TestStructure(unittest.TestCase):
    """Exact tests: counts of calls, dtypes, types, errors."""

    def _counted(self, string, arrays, lens, order, form):
        with unittest.mock.patch.object(contractions, 'contract', wraps=contractions.contract) as spy:
            got = jet_contract(string, *arrays, lens=lens, order=order, over_orders=form)
        return got, spy

    def test_no_jets_is_exactly_one_contract_call_on_the_string_as_written(self):
        rng = np.random.default_rng(0)
        for family, string in cases.ALL_STRINGS:
            for W, K, C in stack_matrix(string):
                shape_of = cases.shape_of_stacks(string, W, K, C)
                arrays = cases.build(string, {}, shape_of, rng)
                lens = lens_of((), shape_of)
                for form in FORMS:
                    with self.subTest(string=string, W=W, K=K, C=C, form=form):
                        got, spy = self._counted(string, arrays, lens, (), form)
                        self.assertEqual(1, spy.call_count)
                        (args, kwargs), = [(c.args, c.kwargs) for c in spy.call_args_list]
                        self.assertEqual(string, args[0])
                        self.assertTrue(all(a is b for a, b in zip(args[1:], arrays)))    # the operands, untouched
                        groups = {ch for ch in string if ch.isupper()}
                        self.assertEqual({'len_' + g for g in groups}, set(kwargs))       # every group it names
                        self.assertEqual(0, kwargs['len_J'])
                        self.assertTrue(np.array_equal(got, contractions.contract(string, *arrays, **kwargs)))

    def test_one_variable_with_the_library_shapes_makes_the_contractions_of_today(self):
        """With one jet variable, a line input and a full order, as P1 made them: one for a ride, two for a product
        with a leaf of two slots, order + 1 for two full jets (under both forms: on numpy a scan is a Python loop of
        one call per coefficient), and no more for a readout than P1 made (the V strings with an input: one, where
        P1 made two). At order 1 a leaf of two slots is as large as the box; the counts are the same."""
        rng = np.random.default_rng(1)
        p1 = {('xi', 'line input'): lambda n: 1, ('xi', 'full'): lambda n: 1, ('z', 'full'): lambda n: 1,
              ('mu', 'leaf of a line'): lambda n: 2, ('mu', 'full leaf'): lambda n: n + 1,
              ('nu', 'leaf of a line'): lambda n: 2, ('nu', 'full leaf'): lambda n: n + 1,
              ('eta', 'full'): lambda n: n + 1,
              ('V', 'line input'): lambda n: 2, ('V', 'full'): lambda n: 1,
              ('H', 'leaf of a line'): lambda n: 2, ('H', 'full leaf'): lambda n: n + 1}
        now = dict(p1)
        now[('V', 'line input')] = lambda n: 1
        for n in (1, 2, 3, 6):
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=())
                for label, sizes, out_order in cases.library_shapes(family, string, (n,)):
                    arrays = cases.build(string, sizes, shape_of, rng)
                    for form in FORMS:
                        with self.subTest(string=string, order=n, shapes=label, form=form):
                            _, spy = self._counted(string, arrays, lens_of(out_order, shape_of), out_order, form)
                            self.assertEqual(now[family, label](n), spy.call_count)
                            self.assertLessEqual(spy.call_count, p1[family, label](n))
                            self.assertEqual(cases.n_contractions(string, sizes, out_order, form), spy.call_count)

    def test_the_number_of_contractions_follows_the_rule(self):
        """Any jet sizes, m = 1 and m = 2: one per combination of the looped operands' multi-indices that lies in
        the box (in a readout, that leaves the pair a nonempty range); a scanned partner's coefficients one each on
        numpy; one for a corner nothing reaches. The rule is ``_jet_contract_cases.n_contractions``, a regression
        pin of the rule as built (the independent pin of the counts is the test above, P1's table)."""
        rng = np.random.default_rng(2)
        strings = [string for _, string in cases.ALL_STRINGS] + ['JWCa,JWCi,JWCb->JWCaib', 'JWCa,JWCi,JWCb,JWCc->Caibc']
        for order in ((1,), (3,), (1, 2)):
            for string in strings:
                for trial in range(4):
                    sizes = random_sizes(rng, string, order)
                    shape_of = cases.shape_of_stacks(string, W=(2,), K=(2,), C=())
                    arrays = cases.build(string, sizes, shape_of, rng)
                    for form in FORMS:
                        with self.subTest(string=string, order=order, sizes=sizes, form=form):
                            _, spy = self._counted(string, arrays, lens_of(order, shape_of), order, form)
                            self.assertEqual(cases.n_contractions(string, sizes, order, form), spy.call_count)

    def test_which_partner_scan_scans(self):
        """Under 'scan', with one jet variable, a partner as large as the box once clipped to it (larger ones
        included) is scanned: one scan per combination of the looped operands. A smaller one never is, and with two
        jet variables nothing is. Under 'loop' nothing is. (The counts of contractions cannot see it: on numpy a
        scan makes one call per coefficient, as the loop does.)"""
        rng = np.random.default_rng(8)
        shape_of = dict(W=(2,), K=(), C=())
        for string, sizes, order, n_scans in (
                ('JWCa,Caib,JWCb->JWCi', {0: (4,), 2: (4,)}, (3,), 1),         # a full partner
                ('JWCa,Caib,JWCb->JWCi', {0: (6,), 2: (5,)}, (3,), 1),         # larger than the box: sliced, then full
                ('JWCa,Caib,JWCb->JWCi', {0: (4,), 2: (3,)}, (3,), 0),         # smaller: looped
                ('JWCa,Caib,JWCb->JWCi', {0: (2, 3), 2: (2, 3)}, (1, 2), 0),   # two jet variables: looped
                ('JWCa,JWCi,JWCb->Caib', {0: (4,), 1: (2,), 2: (4,)}, (3,), 0),  # a readout with no third full jet
                ('JWCa,JWCi,JWCb->Caib', {0: (4,), 1: (5,), 2: (4,)}, (3,), 1),  # ... and with one
                ('JWCa,JWCi,JWCb->JWCaib', {0: (4,), 1: (4,), 2: (3,)}, (3,), 3),  # two partners: one looped
                ('JWCa,JWCi,JWCb->JWCaib', {0: (4,), 1: (3,), 2: (4,)}, (3,), 3)):  #   (3 slots), the other scanned
            arrays = cases.build(string, sizes, shape_of, rng)
            for form in FORMS:
                with self.subTest(string=string, sizes=sizes, order=order, form=form):
                    with unittest.mock.patch.object(contractions, '_jet_scan', wraps=contractions._jet_scan) as spy:
                        jet_contract(string, *arrays, lens=lens_of(order, shape_of), order=order, over_orders=form)
                    self.assertEqual(n_scans if form == 'scan' else 0, spy.call_count)

    def test_no_contraction_names_a_pair_of_orders(self):
        """No dense order-by-order form: every string ``contract`` is handed is the literal with J deleted from
        some input terms (so J at most once per term, the output's as written), with len_J = m; no array carries
        two sets of jet axes, and no lowercase letter is added (no table over the orders)."""
        rng = np.random.default_rng(3)
        for order in ((3,), (1, 2)):
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=())
                for label, sizes, out_order in cases.library_shapes(family, string, order):
                    arrays = cases.build(string, sizes, shape_of, rng)
                    for form in FORMS:
                        _, spy = self._counted(string, arrays, lens_of(out_order, shape_of), out_order, form)
                        for call in spy.call_args_list:
                            made = call.args[0]
                            with self.subTest(string=string, order=order, shapes=label, form=form, made=made):
                                self.assertEqual(cases.output_term(string), cases.output_term(made))
                                for term, made_term in zip(cases.input_terms(string), cases.input_terms(made)):
                                    self.assertIn(made_term, (term, term.replace('J', '')))
                                self.assertEqual(len(order), call.kwargs.get('len_J', len(order)))
                                lowercase = lambda s: {ch for ch in s if ch.islower()}
                                self.assertEqual(lowercase(string), lowercase(made))    # no table over the orders

    def test_float32_stays_float32(self):
        rng = np.random.default_rng(4)
        for order in ((), (1,), (3,), (1, 2)):
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=(2,))
                for label, sizes, out_order in (cases.library_shapes(family, string, order) if order else [('', {}, ())]):
                    arrays = [x.astype(np.float32) for x in cases.build(string, sizes, shape_of, rng)]
                    ref = cases.reference(string, arrays, out_order, shape_of)
                    for form in FORMS:
                        with self.subTest(string=string, order=order, shapes=label, form=form):
                            got = jet_contract(string, *arrays, lens=lens_of(out_order, shape_of), order=out_order,
                                               over_orders=form)
                            self.assertEqual(np.float32, got.dtype)
                            self.assertLess(cases.relerr(got, ref), TOL_32)

    def test_numpy_in_numpy_out(self):
        rng = np.random.default_rng(5)
        for order in ((), (1,), (3,), (1, 2)):
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=())
                for label, sizes, out_order in (cases.library_shapes(family, string, order) if order else [('', {}, ())]):
                    arrays = cases.build(string, sizes, shape_of, rng)
                    for form in FORMS:
                        with self.subTest(string=string, order=order, shapes=label, form=form):
                            got = jet_contract(string, *arrays, lens=lens_of(out_order, shape_of), order=out_order,
                                               over_orders=form)
                            self.assertIs(np.ndarray, type(got))

    def test_the_operands_are_not_changed(self):
        rng = np.random.default_rng(6)
        for order in ((3,), (1, 2)):
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=())
                for label, sizes, out_order in cases.library_shapes(family, string, order):
                    arrays = cases.build(string, sizes, shape_of, rng)
                    copies = [x.copy() for x in arrays]
                    for form in FORMS:
                        jet_contract(string, *arrays, lens=lens_of(out_order, shape_of), order=out_order,
                                     over_orders=form)
                    self.assertTrue(all(np.array_equal(x, c) for x, c in zip(arrays, copies)), string)

    def test_over_orders_and_order_have_no_default_and_over_orders_two_values(self):
        lens0, lens1 = GroupLengths(J=0, W=1, K=0, C=0), GroupLengths(J=1, W=1, K=0, C=0)
        eta, D = np.ones((3, 4)), np.ones((4, 5))
        for lens, x, order in ((lens0, eta, ()), (lens1, np.ones((2, 3, 4)), (1,))):   # with no jets too
            for bad in ('dense', 'Scan', None, True):
                with self.subTest(J=lens.J, over_orders=bad):
                    with self.assertRaisesRegex(ValueError, 'over_orders'):
                        jet_contract('JWCi,Cio->JWCo', x, D, lens=lens, order=order, over_orders=bad)
            with self.assertRaises(TypeError):
                jet_contract('JWCi,Cio->JWCo', x, D, lens=lens, order=order)            # no default
            with self.assertRaises(TypeError):
                jet_contract('JWCi,Cio->JWCo', x, D, lens=lens, over_orders='loop')     # no default for the jet shape
            with self.assertRaises(TypeError):
                jet_contract('JWCi,Cio->JWCo', x, D, order=order, over_orders='loop')   # nor for the record
            with self.assertRaises(TypeError):
                jet_contract('JWCi,Cio->JWCo', x, D, lens, order, 'loop')               # keyword-only

    def test_lens_must_be_the_record(self):
        eta, D = np.ones((3, 4)), np.ones((4, 5))
        for bad in ((0, 1, 0, 0), dict(J=0, W=1, K=0, C=0), None):
            with self.subTest(lens=bad):
                with self.assertRaisesRegex(TypeError, 'GroupLengths'):
                    jet_contract('JWCi,Cio->JWCo', eta, D, lens=bad, order=(), over_orders='loop')

    def test_malformed_strings_raise_at_every_order(self):
        """Whether a string is a jet_contract string is decided from the string alone, so a call site is wrong at
        every order or at none (`order = ()` included)."""
        one = np.ones(())
        malformed = [
            ('WJCa,Caib,JWCi->JWCb', 'lead'),               # J does not lead an operand's term
            ('JWCa,Caib,JWCi->WJCb', 'lead'),               # ... or the output's
            ('WCa,Caib,WCi->WCb', 'no operand has'),        # no J at all: a contract string
            ('WCi,Cio->', 'no operand has'),                # ... a readout of no jet
            ('WCa,Caib->JWCb', 'do not appear'),            # J in the output and on no operand (contract's parser)
            ('TWCa,Caib,AWCi->TWCb', 'not groups'),         # the groups of P1 are gone
            ('JWCa,Caib,AWCi->JWCb', 'not groups'),
            ('TWCi,Cio->TWCo', 'not groups'),
            ('JWGa,Gab->JWGb', 'not groups'),               # a group the record does not count
            ('SWCi,SCio->SWCo', 'not groups'),              # S like any other: there is no site group,
            ('JSWCi,SCio->JSWCo', 'not groups'),            #     after the jet group
            ('SJWCi,SCio->SJWCo', 'not groups'),            #     or before it
            ('JWCa,Caib->JWCb,', 'invalid'),                # the grouped-string parser's own errors pass through
            ('JWCa,Caib', "'->'"),
            ('J,JWCi->JWCi', 'J alone'),                    # a term of J alone: no term left to take a coefficient
        ]
        for J in (0, 1, 2):
            lens = GroupLengths(J=J, W=0, K=0, C=0)
            for string, message in malformed:
                with self.subTest(string=string, J=J):
                    with self.assertRaisesRegex(ValueError, message):
                        jet_contract(string, *([one] * len(string.split('->')[0].split(','))),
                                     lens=lens, order=(1,) * J, over_orders='loop')

    def test_a_product_of_three_jets_is_a_product(self):
        """'JWCa,JWCi,JWCb->JWCa' (three jets and J in the output; P1 refused it) is a truncated product like any
        other: against the reference."""
        rng = np.random.default_rng(7)
        string = 'JWCa,JWCi,JWCb->JWCa'
        shape_of = dict(W=(3,), K=(), C=())
        self.assertTrue(cases.is_product(string))
        for order in ((3,), (1, 2)):
            arrays = cases.build(string, {p: cases.box_of(order) for p in range(3)}, shape_of, rng)
            ref = cases.reference(string, arrays, order, shape_of)
            for form in FORMS:
                with self.subTest(order=order, form=form):
                    got = jet_contract(string, *arrays, lens=lens_of(order, shape_of), order=order, over_orders=form)
                    self.assertLess(cases.relerr(got, ref), TOL)

    def test_operand_count_mismatch_raises(self):
        lens = GroupLengths(J=1, W=1, K=0, C=0)
        with self.assertRaisesRegex(ValueError, 'operand'):
            jet_contract('JWCa,Caib,JWCi->JWCb', np.ones((4, 3, 2)), np.ones((2, 3, 4)), lens=lens, order=(3,),
                         over_orders='loop')


class TestGroupLengths(unittest.TestCase):
    """The record is a static: frozen, with value equality and a value hash (it keys the memoized scan body, and
    is jax static data)."""

    def test_fields_in_order(self):
        self.assertEqual(['J', 'W', 'K', 'C'], [f.name for f in dc.fields(GroupLengths)])
        lens = GroupLengths(1, 2, 3, 4)
        self.assertEqual((1, 2, 3, 4), (lens.J, lens.W, lens.K, lens.C))
        with self.assertRaises(TypeError):
            GroupLengths(J=1, W=2, K=0)                         # no defaults: every count is said
        with self.assertRaises(TypeError):
            GroupLengths(S=0, J=1, W=2, K=0, C=0)               # there is no site group
        with self.assertRaises(TypeError):
            GroupLengths(T=1, W=2, K=0, C=0)                    # nor the order group of P1

    def test_value_equality_and_hash(self):
        a, b = GroupLengths(J=2, W=2, K=0, C=1), GroupLengths(J=2, W=2, K=0, C=1)
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(1, len({a, b}))
        for field in ('J', 'W', 'K', 'C'):
            self.assertNotEqual(a, dc.replace(a, **{field: getattr(a, field) + 1}))

    def test_frozen(self):
        with self.assertRaises(dc.FrozenInstanceError):
            GroupLengths(J=1, W=2, K=0, C=1).W = 3

    def test_counts_are_nonnegative_ints_and_J_any_count(self):
        for J in (0, 1, 2, 3, 7):
            self.assertEqual(J, GroupLengths(J=J, W=0, K=0, C=0).J)
        for bad in (dict(W=-1), dict(J=-1), dict(W=1.0), dict(W=None), dict(C=True), dict(K=(2,)),
                    dict(W=np.int64(2)), dict(J=np.int64(1))):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    GroupLengths(**{**dict(J=1, W=2, K=0, C=1), **bad})

    def test_an_equal_record_gets_the_same_scan_body(self):
        """The scan body comes from a factory memoized on the statics it reads (the step string, the record, the
        carried and the scanned positions): an equal record built anew is the same key, so the body is ONE object
        and an eager jax scan finds its compiled program again. What the body closes over is static: strings,
        ints and a dict of ints, nothing else."""
        for step, carrier, partner in (('JWCa,Caib,WCb->JWCi', 0, 2), ('JWCa,WCi,JWCb->Caib', 0, 1)):
            a, b = GroupLengths(J=1, W=2, K=0, C=1), GroupLengths(J=1, W=2, K=0, C=1)
            body = contractions._jet_scan_body(step, a, carrier, partner)
            self.assertIs(body, contractions._jet_scan_body(step, b, carrier, partner))
            self.assertIsNot(body, contractions._jet_scan_body(step, dc.replace(a, W=1), carrier, partner))
            cells = [cell.cell_contents for cell in body.__closure__]
            self.assertEqual(5, len(cells))                     # the step string, its lengths, three positions
            for value in cells:
                with self.subTest(step=step, cell=value):
                    self.assertIn(type(value), (str, int, dict))
                    if type(value) is dict:
                        self.assertTrue(all(type(k) is str and type(v) is int for k, v in value.items()))

    def test_the_scan_body_key_holds_every_static_it_reads(self):
        """The step string does not always say which operand is scanned: in a readout of four jets the pair keeps
        J and both other operands lose it, so the same step string serves a scan of either. Calls in turn that
        scan the third and then the fourth operand (by their sizes) must each get their own body: checked by
        value against the reference, in both orders of the calls. Likewise a product whose carrier moves."""
        rng = np.random.default_rng(0)
        shape_of = dict(W=(3,), K=(), C=())
        lens = lens_of((3,), shape_of)
        for string, variants in (('JWCa,JWCb,JWCc,JWCj->Cabcj', ({0: (4,), 1: (4,), 2: (4,), 3: (2,)},
                                                                 {0: (4,), 1: (4,), 2: (2,), 3: (4,)})),
                                 ('Caib,JWCi,JWCb->JWCa', ({1: (4,), 2: (4,)}, {1: (4,), 2: (2,)}, {1: (2,), 2: (4,)}))):
            for ordering in (variants, variants[::-1]):
                contractions._jet_scan_body.cache_clear()
                for sizes in ordering:
                    arrays = cases.build(string, sizes, shape_of, rng)
                    with self.subTest(string=string, sizes=sizes):
                        got = jet_contract(string, *arrays, lens=lens, order=(3,), over_orders='scan')
                        self.assertLess(cases.relerr(got, cases.reference(string, arrays, (3,), shape_of)), TOL)


class TestOrderInsideAStep(unittest.TestCase):
    """Which two operands ONE step's contraction pairs first. Not a test of values: the result is the same in any
    order. It is a test of cost, and a marker.

    A product step contracts three things: the carried jet with all its slots, the operands with no jet group (a
    core), and one coefficient of the partner. The cheap order pairs the coefficient with the core first (the
    expensive contraction, done once), then meets every slot of the carried jet. Pairing the carried jet with the
    core first does the expensive contraction on every slot, at every step. A readout step should fold its two
    jets along the jet axis first.

    ``jet_contract`` leaves the order to ``contract``: one ``jnp.einsum`` with its default optimizer on jax, the
    forced greedy pairwise path on numpy. What each does is read here from the calls themselves: the strings are
    the ones ``jet_contract`` derives (spied, not copied), the roles of the operands are read off the literal and
    the derived string, and the path is the one handed to ``np.einsum`` or the one ``jnp.einsum_path`` reports for
    the same default ``jnp.einsum`` uses.

    The cases are the 12 product strings with the library's operand shapes at order 4, ``ORDER``: in the 8 ``mu``
    and ``nu`` strings a leaf of two slots and a full leaf, in the 4 ``eta`` strings two full jets, 20 cases. The
    carrier is the operand with the most slots, the first among equals: in the ``nu`` strings with a full leaf the
    leaf (first in the string) carries. At order 1 a leaf of two slots ties with a full jet of two slots, so in the
    4 ``nu`` strings with a leaf of two slots the leaf carries too, and all 20 cases are costly on numpy (16 of 20
    at order 4; recounted with ``contractions._pairwise_path`` for P4a).
    """

    ORDER = ORDER_INSIDE_A_STEP
    SIZES = dict(a=12, i=6, b=12, o=20)                     # representative: bonds larger than the mode rank
    STACKS = dict(W=(200,), K=(2,), C=(2,))
    # jnp.einsum's default `optimize`, by jax version: 'auto' (jax 0.10, the current CI leg's twin) and 'optimal'
    # (jax 0.4.30, the CI floor). A third value is seen, not silently followed.
    JAX_DEFAULT_OPTIMIZE = ('auto', 'optimal')

    PRODUCT_CASES = [(family, string, label, sizes) for family, string in cases.ALL_STRINGS if cases.is_product(string)
                     for label, sizes, _ in cases.library_shapes(family, string, (ORDER_INSIDE_A_STEP,))]
    READOUT_CASES = [(family, string, label, sizes) for family, string in cases.ALL_STRINGS if family == 'H'
                     for label, sizes, _ in cases.library_shapes(family, string, (ORDER_INSIDE_A_STEP,))]

    def _first_pairs(self, string, sizes, xnp):
        """For every 3-operand contraction ``jet_contract(string, ...)`` makes on backend ``xnp`` in which the
        carried jets have ALL the box's slots:
        ``(the derived string, the positions paired first, the carried positions, the no-jet positions)``.

        The carried positions are the terms of the derived string that still have ``J``; the no-jet positions are
        the terms with no jet group in the literal; the remaining one is a coefficient.

        Only the steps with every slot are looked at, because jax's path depends on the sizes: under ``'loop'``
        the later steps contract a short slice of the carried jet (down to one slot), where another order is as
        cheap or cheaper, and jax takes it. The step with every slot is the first of ``'loop'``, and it is every
        step of ``'scan'``, whose carried jet keeps all its slots."""
        shape_of = cases.shape_of_stacks(string, **self.STACKS)
        with unittest.mock.patch.dict(cases.SIZE, self.SIZES):
            arrays = cases.build(string, sizes, shape_of, np.random.default_rng(0))
        arrays = [xnp.asarray(x) for x in arrays]
        lens = GroupLengths(J=1, W=1, K=len(shape_of['K']), C=1)
        with unittest.mock.patch.object(contractions, 'contract', wraps=contractions.contract) as contract_spy, \
                unittest.mock.patch.object(xnp, 'einsum', wraps=xnp.einsum) as einsum_spy:
            jet_contract(string, *arrays, lens=lens, order=(self.ORDER,), over_orders='loop')
        self.assertEqual(contract_spy.call_count, einsum_spy.call_count)       # one einsum per contraction
        literal_terms = cases.input_terms(string)
        no_jet = frozenset(p for p, t in enumerate(literal_terms) if 'J' not in t)
        steps = []
        for contract_call, einsum_call in zip(contract_spy.call_args_list, einsum_spy.call_args_list):
            made = contract_call.args[0]
            self.assertEqual(3, len(cases.input_terms(made)))
            if xnp is np:
                path = einsum_call.kwargs['optimize']               # the path `contract` forces on numpy
            else:
                self.assertNotIn('optimize', einsum_call.kwargs)    # jax: `jnp.einsum`'s own default
                path = xnp.einsum_path(*einsum_call.args, optimize=self._jax_default_optimize())[0]
            first = next(step for step in path if not isinstance(step, str))    # numpy's path leads with a label
            carried = frozenset(p for p, t in enumerate(cases.input_terms(made)) if 'J' in t)
            if all(contract_call.args[1 + p].shape[0] == self.ORDER + 1 for p in carried):
                steps.append((made, frozenset(int(p) for p in first), carried, no_jet))
        self.assertGreater(len(steps), 0)
        return steps

    @staticmethod
    def _jax_default_optimize():
        import inspect
        import jax.numpy as jnp
        return inspect.signature(jnp.einsum).parameters['optimize'].default

    @unittest.skipUnless(common.jax_available, 'jax not available')
    def test_jax_default_optimizer_is_one_read_here(self):
        # recorded so that a change of jax's default is seen, not silently followed
        self.assertIn(self._jax_default_optimize(), self.JAX_DEFAULT_OPTIMIZE)

    @unittest.skipUnless(common.jax_available, 'jax not available')
    def test_on_jax_a_product_step_pairs_the_coefficient_with_the_core_first(self):
        import jax.numpy as jnp
        self.assertEqual(20, len(self.PRODUCT_CASES))
        for family, string, label, sizes in self.PRODUCT_CASES:
            for made, first, carried, no_jet in self._first_pairs(string, sizes, jnp):
                with self.subTest(string=string, shapes=label, made=made):
                    self.assertEqual(1, len(carried))
                    self.assertFalse(first & carried, 'the carried jet is contracted first')
                    self.assertEqual(1, len(first & no_jet))        # the coefficient with the core

    def test_on_numpy_a_readout_step_folds_its_two_jets_first(self):
        self.assertEqual(8, len(self.READOUT_CASES))
        for family, string, label, sizes in self.READOUT_CASES:
            for made, first, carried, no_jet in self._first_pairs(string, sizes, np):
                with self.subTest(string=string, shapes=label, made=made):
                    self.assertEqual(2, len(carried))               # the pair, each with its jet axis
                    self.assertEqual(carried, first)

    @unittest.skipUnless(common.jax_available, 'jax not available')
    def test_on_jax_a_readout_step_folds_its_two_jets_first(self):
        import jax.numpy as jnp
        for family, string, label, sizes in self.READOUT_CASES:
            for made, first, carried, no_jet in self._first_pairs(string, sizes, jnp):
                with self.subTest(string=string, shapes=label, made=made):
                    self.assertEqual(2, len(carried))
                    self.assertEqual(carried, first)

    def test_on_numpy_what_a_product_step_pairs_first_today(self):
        """A plain, passing test of what numpy does today, so that the helper behind the expected failure below is
        exercised on numpy product strings by a test that can fail (``expectedFailure`` accepts ANY failure, a
        broken helper included).

        Every product case has exactly one step in which the carried jet has all its slots, and its first pair is
        two of its three operands. For the four ``nu`` strings with a leaf of two slots the first pair is the core
        with the coefficient: the cheap order, by the accident of the operand order. For the other 16 cases it is
        the carried jet with the core. The count 16 against 4 is pinned here: when Part 2 (row 5.5) fixes the order
        it becomes 0 against 20, and this test is then updated together with the marker below."""
        cheap_today = [(string, 'leaf of a line') for string in cases.STRINGS['nu']]
        costly, cheap = [], []
        for family, string, label, sizes in self.PRODUCT_CASES:
            steps = self._first_pairs(string, sizes, np)
            with self.subTest(string=string, shapes=label):
                self.assertEqual(1, len(steps))                     # 'loop': only the first step has every slot
                (made, first, carried, no_jet), = steps
                self.assertEqual(1, len(carried))
                self.assertEqual(1, len(no_jet))
                self.assertFalse(carried & no_jet)
                self.assertEqual(2, len(first))
                self.assertTrue(first <= {0, 1, 2})
                if (string, label) in cheap_today:
                    self.assertFalse(first & carried)               # the coefficient ...
                    self.assertEqual(no_jet, first & no_jet)        # ... with the core
                else:
                    self.assertEqual(carried | no_jet, first)       # the carried jet with the core
            (costly if first & carried and first & no_jet else cheap).append((string, label))
        self.assertEqual(16, len(costly))
        self.assertEqual(sorted(cheap_today), sorted(cheap))

    @unittest.expectedFailure
    def test_on_numpy_no_product_step_pairs_the_carried_jet_with_the_core_first(self):
        """AN EXPECTED FAILURE, kept as a marker for PART 2 (row 5.5, the contraction order).

        What it asserts is what should hold and does not hold today: on numpy, no product step contracts the
        carried jet with an operand that has no jet group first. Today ``contract`` forces ``_pairwise_path`` on
        numpy; a product step is a three-way tie there, the tie goes to the first two operands, and for 16 of the
        20 product cases (every ``mu`` and ``eta`` case, and the ``nu`` strings with a full leaf) those are the
        carried jet and the core. So the expensive contraction is done on every slot of the carried jet at every
        step. The values are right; only the cost is wrong. The maintainer deferred the fix to Part 2 (2026-11-17),
        with the instruction that it be marked so that it is not forgotten. This test is that mark: it is reported
        as an expected failure in every run.

        WHEN THIS TEST STARTS PASSING (the suite then reports an unexpected success, which is a failure): the order
        was fixed. Remove the ``expectedFailure`` decorator, so that the test guards the fix, update the counts of
        ``test_on_numpy_what_a_product_step_pairs_first_today`` (which fails at the same moment), and remove the
        PART 2 (row 5.5) notes in ``backend/contractions.py``.
        """
        offenders = []
        for family, string, label, sizes in self.PRODUCT_CASES:
            for made, first, carried, no_jet in self._first_pairs(string, sizes, np):
                if first & carried and first & no_jet:
                    offenders.append(made)
        self.assertEqual([], sorted(set(offenders)))


if __name__ == '__main__':
    unittest.main()
