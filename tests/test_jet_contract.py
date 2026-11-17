# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``contractions.jet_contract`` and ``contractions.GroupLengths``.

``jet_contract`` is ``contract`` with the order groups ``T`` (a full jet) and ``A`` (an affine variable)
multiplying as truncated polynomials. Lines of evidence, each independent of the others:

1. **The reference** (`TestAgainstReference`): every one of the 41 strings of the sampling terms against
   a reference that uses neither ``contract`` nor ``jet_contract`` (``_jet_contract_cases.reference``:
   every stack flattened to one axis, explicit loops over pairs of orders), at orders 0, 1 and 3, with
   ``W``, ``K``, ``C`` empty and not, under both forms over the orders. A tolerance test.
2. **The two forms** (`TestTheTwoForms`): ``'scan'`` and ``'loop'`` against each other and against the
   reference wherever a full jet is taken one coefficient at a time (the 12 products of two full jets and
   the 4 readouts of three), at orders 1 to 16, in float64 and float32. They agree to rounding, not bit
   for bit, so this is a tolerance test.
3. **Sensitivity** (`TestSensitivity`): a perturbation of ONE slot of ONE operand moves the result away
   from the reference by far more than the tolerance of (1), for every string, operand and slot. So
   (1) would fail if any coefficient were dropped or misplaced.
4. **Strictness** (`TestStrictness`): a stray or a missing axis on any operand raises, wherever it is.
5. **Structure** (`TestStructure`): exact. How many ``contract`` calls are made (one at order 0, and
   the same string); the dtype; numpy in, numpy out; the errors of a malformed string or call.
6. `TestGroupLengths`: the record is a value-hashed static.
7. **The order inside a step** (`TestOrderInsideAStep`): which two operands one step's contraction
   pairs first, read off the path each backend uses. It holds an EXPECTED FAILURE, a marker for Part 2
   (row 5.5, the contraction order): on numpy most product steps contract the carried jet with the core
   first, which is the costly order.

The numerical tests are numpy-only, by the house convention (`TestOrderInsideAStep` reads jax's
contraction path where jax is installed; it compares no numbers); jax dispatch (jit, jax in gives jax out,
the scan is one ``lax.scan``) is in ``test_dispatch.py``, sharding in ``test_contractions_sharding.py``,
and the strings ``jet_contract`` hands to ``contract`` in ``test_contractions_interpreter.py``.
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
import _jet_contract_cases as cases     # the 41 strings and their reference: test-only

ORDERS = (0, 1, 3)
W_SHAPES = ((), (3,), (2, 2))
K_SHAPES = ((), (2,), (2, 2))
C_SHAPES = ((), (2,))
FORMS = ('scan', 'loop')

TOL = 1e-12                             # float64: against the reference, and the two forms against each other
TOL_32 = 1e-5                           # float32

RANK_MESSAGE = r"jet_contract\('[A-Za-z,]+->[A-Za-z]*'\): operand %d \(term '[A-Za-z]+'\) has ndim \d+, but"


def lens_of(order, shape_of):
    return GroupLengths(T=1 if order else 0, W=len(shape_of['W']), K=len(shape_of['K']), C=len(shape_of['C']))


def stack_matrix(string):
    """(W, K, C) for one string: every combination, with K only where the string has a direction."""
    Ks = K_SHAPES if 'K' in string else ((),)
    return itertools.product(W_SHAPES, Ks, C_SHAPES)


class TestTheStrings(unittest.TestCase):
    """The table of strings is the one the terms have: 41 strings in seven families, over the groups
    T, A, W, K, C and no others."""

    def test_41_strings_in_seven_families(self):
        self.assertEqual({'xi': 5, 'z': 3, 'mu': 8, 'nu': 8, 'eta': 4, 'V': 5, 'H': 8},
                         {family: len(strings) for family, strings in cases.STRINGS.items()})
        strings = [string for _, string in cases.ALL_STRINGS]
        self.assertEqual(41, len(strings))
        self.assertEqual(41, len(set(strings)))
        self.assertEqual(set('TAWKC'), {ch for s in strings for ch in s if ch.isupper()})
        self.assertEqual(12, sum(cases.is_full_product(s) for s in strings))
        self.assertEqual(20, sum(cases.is_product(s) for s in strings))
        # where `over_orders` is about something: the 12 products of two full jets, and the 4 readouts
        # of three ('H' with a full leaf)
        self.assertEqual(16, sum(cases.follows_the_form(s) for s in strings))
        self.assertEqual(4, sum(cases.follows_the_form(s) and not cases.is_full_product(s) for s in strings))


class TestAgainstReference(unittest.TestCase):

    def test_every_string_every_stack_both_forms(self):
        rng = np.random.default_rng(0)
        n_checked = 0
        for order in ORDERS:
            for family, string in cases.ALL_STRINGS:
                for W, K, C in stack_matrix(string):
                    shape_of = cases.shape_of_stacks(string, W, K, C)
                    arrays, operands = cases.build(string, order, shape_of, rng)
                    ref = cases.reference(family, string, operands, order, shape_of)
                    for form in FORMS:
                        with self.subTest(string=string, order=order, W=W, K=K, C=C, form=form):
                            got = jet_contract(string, *arrays, lens=lens_of(order, shape_of), over_orders=form)
                            self.assertEqual(ref.shape, got.shape)                      # structural: exact
                            self.assertLess(cases.relerr(got, ref), TOL)
                            n_checked += 1
        self.assertEqual(n_checked, 2 * len(ORDERS) * sum(len(list(stack_matrix(s))) for _, s in cases.ALL_STRINGS))

    def test_whitespace_in_the_string_is_allowed(self):
        rng = np.random.default_rng(1)
        string = 'TWCa,Caib,AWCi->TWCb'
        shape_of = dict(W=(3,), K=(), C=(2,))
        arrays, _ = cases.build(string, 3, shape_of, rng)
        lens = lens_of(3, shape_of)
        plain = jet_contract(string, *arrays, lens=lens, over_orders='loop')
        spaced = jet_contract('TWCa, Caib, AWCi -> TWCb', *arrays, lens=lens, over_orders='loop')
        self.assertTrue(np.array_equal(plain, spaced))

    def test_a_single_jet_and_no_order_in_the_output_is_its_top_coefficient(self):
        # beyond the 41: the rule "no order group in the output: the top coefficient" with one operand
        rng = np.random.default_rng(2)
        eta, D = rng.standard_normal((4, 3, 5)), rng.standard_normal((5, 6))
        got = jet_contract('TWCi,Cio->WCo', eta, D, lens=GroupLengths(T=1, W=1, K=0, C=0), over_orders='loop')
        self.assertLess(cases.relerr(got, np.einsum('wi,io->wo', eta[3], D)), TOL)


    def test_more_jets_than_the_41_have(self):
        """Beyond the 41: a readout with a pair, an affine operand AND a further full jet, and one of four
        full jets. The operand that follows the form is the first full jet beyond the pair; the others are
        taken one coefficient at a time by slices under both forms (under 'scan' the carried jet then
        starts shifted by the orders already taken). Against explicit loops over the orders, in float64
        and in float32: the start shift puts zeros in front of the carried jet, and they must be of the
        jet's dtype (zeros of the default dtype would turn a float32 result into float64)."""
        rng = np.random.default_rng(3)
        lens = GroupLengths(T=1, W=1, K=0, C=0)
        for order in (1, 2, 4):
            n = order + 1
            a64, b64, c64 = rng.standard_normal((n, 3, 2)), rng.standard_normal((n, 3, 4)), rng.standard_normal((n, 3, 5))
            x64, e64 = rng.standard_normal((2, 3, 3)), rng.standard_normal((n, 3, 2))
            for dtype, tol in ((np.float64, TOL), (np.float32, TOL_32)):
                a, b, c, x, e = (v.astype(dtype) for v in (a64, b64, c64, x64, e64))
                a_, b_, c_, x_, e_ = (v.astype(np.float64) for v in (a, b, c, x, e))        # the reference: float64
                ref_affine = sum(np.einsum('wa,wi,wb,wc->aibc', a_[i], x_[j], b_[k], c_[order - i - j - k])
                                 for i in range(n) for j in range(2) for k in range(n) if i + j + k <= order)
                ref_full = sum(np.einsum('wa,wb,wc,wj->abcj', a_[i], b_[k], c_[l], e_[order - i - k - l])
                               for i in range(n) for k in range(n) for l in range(n) if i + k + l <= order)
                for form in FORMS:
                    with self.subTest(order=order, form=form, dtype=np.dtype(dtype).name):
                        got = jet_contract('TWCa,AWCi,TWCb,TWCc->Caibc', a, x, b, c, lens=lens, over_orders=form)
                        self.assertEqual(dtype, got.dtype)
                        self.assertLess(cases.relerr(got, ref_affine), tol)
                        got = jet_contract('TWCa,TWCb,TWCc,TWCj->Cabcj', a, b, c, e, lens=lens, over_orders=form)
                        self.assertEqual(dtype, got.dtype)
                        self.assertLess(cases.relerr(got, ref_full), tol)


class TestTheTwoForms(unittest.TestCase):
    """'scan' (one jet carried and shifted one slot per step) against 'loop' (slices)."""

    def test_forms_agree_wherever_a_full_jet_is_taken_one_coefficient_at_a_time(self):
        # the 12 products of two full jets and the 4 readouts of three: 'scan' scans, 'loop' slices
        rng = np.random.default_rng(0)
        shape_of_all = dict(W=(3,), K=(2,), C=(2,))
        products = [(family, string) for family, string in cases.ALL_STRINGS if cases.follows_the_form(string)]
        self.assertEqual(16, len(products))
        for dtype, tol in ((np.float64, TOL), (np.float32, TOL_32)):
            for order in range(1, 17):
                for family, string in products:
                    with self.subTest(string=string, order=order, dtype=np.dtype(dtype).name):
                        shape_of = cases.shape_of_stacks(string, **shape_of_all)
                        arrays, operands = cases.build(string, order, shape_of, rng)
                        arrays = [x.astype(dtype) for x in arrays]
                        operands = [(x, groups, core) for x, (_, groups, core) in zip(arrays, operands)]
                        ref = cases.reference(family, string, operands, order, shape_of)
                        lens = lens_of(order, shape_of)
                        scan = jet_contract(string, *arrays, lens=lens, over_orders='scan')
                        loop = jet_contract(string, *arrays, lens=lens, over_orders='loop')
                        self.assertEqual(scan.dtype, dtype)
                        self.assertEqual(loop.dtype, dtype)
                        self.assertLess(cases.relerr(scan, ref), tol)
                        self.assertLess(cases.relerr(loop, ref), tol)
                        self.assertLess(cases.relerr(scan, np.asarray(loop, dtype=np.float64)), tol)

    def test_the_form_changes_nothing_where_no_full_jet_is_taken_one_coefficient_at_a_time(self):
        # a ride, an affine partner, and a readout with no third full jet make the same contractions on the
        # same slices under both forms: bit for bit
        rng = np.random.default_rng(0)
        for family, string in cases.ALL_STRINGS:
            if cases.follows_the_form(string):
                continue
            with self.subTest(string=string):
                shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=(2,))
                arrays, _ = cases.build(string, 3, shape_of, rng)
                lens = lens_of(3, shape_of)
                self.assertTrue(np.array_equal(jet_contract(string, *arrays, lens=lens, over_orders='scan'),
                                               jet_contract(string, *arrays, lens=lens, over_orders='loop')))


class TestSensitivity(unittest.TestCase):
    """The reference test has teeth: a relative perturbation of 1e-6 of ONE order slot of ONE operand (the
    whole operand where it has no order axis) leaves the reference test's tolerance behind by orders of
    magnitude. Every slot of every operand of every string is tried, under both forms, so no coefficient
    can be dropped, doubled or put in the wrong slot without the reference test failing.

    One kind of slot does not enter a result, and must not: in a top coefficient (no order group in the
    output) the orders of the operands sum to the top, so a low slot of a full jet whose partners cannot
    make up the rest (an affine partner reaches order 1 only) is not read. There the result must not
    move at all."""

    @staticmethod
    def _is_read(string, p, slot, order):
        """Does coefficient `slot` of operand `p` enter the result? (Derived from the definition of the
        truncated product, not from the code.)"""
        terms = cases.input_terms(string)
        if 'T' in cases.output_term(string) or cases.n_order_operands(string) == 1:
            return True                         # a product keeps every coefficient (it meets the other's slot 0)
        reach = sum(order if 'T' in t else 1 for q, t in enumerate(terms) if q != p and ('T' in t or 'A' in t))
        return order - slot <= reach            # the others must supply order - slot

    def test_a_perturbation_of_any_slot_of_any_operand_is_caught(self):
        rng = np.random.default_rng(0)
        order = 3
        caught = {family: 0 for family in cases.STRINGS}
        n_not_read = 0
        for family, string in cases.ALL_STRINGS:
            shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=(2,))
            arrays, operands = cases.build(string, order, shape_of, rng)
            ref = cases.reference(family, string, operands, order, shape_of)
            lens = lens_of(order, shape_of)
            for p, term in enumerate(cases.input_terms(string)):
                has_order = 'T' in term or 'A' in term
                for slot in (range(arrays[p].shape[0]) if has_order else (None,)):
                    perturbed = [x.copy() for x in arrays]
                    target = perturbed[p][slot] if has_order else perturbed[p]
                    target *= 1.0 + 1e-6                                # in place: a view of the operand
                    is_read = (not has_order) or self._is_read(string, p, slot, order)
                    for form in FORMS:
                        with self.subTest(string=string, operand=p, slot=slot, form=form, is_read=is_read):
                            got = jet_contract(string, *perturbed, lens=lens, over_orders=form)
                            if is_read:
                                self.assertGreater(cases.relerr(got, ref), 1e4 * TOL)
                                caught[family] += 1
                            else:
                                unperturbed = jet_contract(string, *arrays, lens=lens, over_orders=form)
                                self.assertTrue(np.array_equal(got, unperturbed))
                                n_not_read += 1
        self.assertTrue(all(n > 0 for n in caught.values()), caught)   # every family was tried
        # the slots that are not read: slots 0 and 1 of the full jet in the two 'V' strings with an input
        # (2 strings x 2 slots x 2 forms); in 'H' with an affine leaf the two full jets reach each other
        self.assertEqual(8, n_not_read)


class TestStrictness(unittest.TestCase):
    """Every call passes the whole record of lengths, so every group of every string is pinned: an
    operand with one axis too many or too few raises, wherever the axis is. (With only the lengths a
    string strictly needs, a stray axis beside W or K is read as one more axis of that stack.)"""

    def test_a_stray_or_a_missing_axis_on_any_operand_raises(self):
        rng = np.random.default_rng(0)
        n_checked = 0
        for order in (0, 3):
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(2, 2), K=(2,), C=(2,))
                arrays, _ = cases.build(string, order, shape_of, rng)
                lens = lens_of(order, shape_of)
                for form in FORMS:
                    jet_contract(string, *arrays, lens=lens, over_orders=form)      # the good call passes
                    for p in range(len(arrays)):
                        bad_operands = [np.expand_dims(arrays[p], axis) for axis in range(arrays[p].ndim + 1)]
                        bad_operands += [np.take(arrays[p], 0, axis) for axis in range(arrays[p].ndim)]
                        for bad in bad_operands:
                            with self.subTest(string=string, order=order, form=form, operand=p, shape=bad.shape):
                                with self.assertRaisesRegex(ValueError, RANK_MESSAGE % p):  # one message, every order
                                    jet_contract(string, *arrays[:p], bad, *arrays[p + 1:], lens=lens, over_orders=form)
                                n_checked += 1
        self.assertGreater(n_checked, 3000)

    def test_a_wrong_record_raises(self):
        rng = np.random.default_rng(0)
        string = 'TWKCo,Cio->TWKCi'         # W and K stand together on the only operand that has them
        shape_of = dict(W=(2, 2), K=(2,), C=(2,))
        arrays, _ = cases.build(string, 3, shape_of, rng)
        good = lens_of(3, shape_of)
        jet_contract(string, *arrays, lens=good, over_orders='loop')
        for field in ('T', 'W', 'K', 'C'):
            for delta in (-1, 1):
                value = getattr(good, field) + delta
                if value < 0 or (field == 'T' and value > 1):
                    continue
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, 'ndim'):
                        jet_contract(string, *arrays, lens=dc.replace(good, **{field: value}), over_orders='loop')
        # W and K traded one axis for the other: the same total, so the same ranks. Not an error, and the
        # same contraction (the split inside a run of co-traveling groups is unobservable).
        traded = dc.replace(good, W=good.W - 1, K=good.K + 1)
        self.assertTrue(np.array_equal(jet_contract(string, *arrays, lens=traded, over_orders='loop'),
                                       jet_contract(string, *arrays, lens=good, over_orders='loop')))

    def test_order_axes_of_the_wrong_length_raise(self):
        """Where there is an order axis (lens.T == 1) every full jet has the same order+1 >= 2 slots and an
        affine variable 2: one check, with one message, whether the axis rides, multiplies or is read out."""
        rng = np.random.default_rng(0)
        lens = GroupLengths(T=1, W=1, K=0, C=0)
        mu, G = rng.standard_normal((4, 3, 2)), rng.standard_normal((2, 3, 4))
        xi, xi_full, nu = rng.standard_normal((2, 3, 3)), rng.standard_normal((4, 3, 3)), rng.standard_normal((4, 3, 4))
        D = rng.standard_normal((3, 5))
        bad_calls = [
            ('a product: an affine operand with three slots', 'TWCa,Caib,AWCi->TWCb', (mu, G, xi_full[:3]), 2),
            ('a product: two full jets of different orders', 'TWCa,Caib,TWCi->TWCb', (mu, G, xi_full[:3]), 2),
            ('a product: a full jet with one slot', 'TWCa,Caib,AWCi->TWCb', (mu[:1], G, xi), 0),
            ('a readout: full jets of different orders', 'TWCa,TWCi,TWCb->Caib', (mu, xi_full, nu[:2]), 2),
            ('a readout: a full jet with one slot', 'TWCa,AWCi,TWCb->Caib', (mu[:1], xi, nu[:1]), 0),
            ('a ride: a full jet with one slot', 'TWCi,Cio->TWCo', (xi_full[:1], D), 0),
            ('a ride: an affine variable with three slots', 'AWCi,Cio->AWCo', (xi_full[:3], D), 0),
            ('a ride: an affine variable with one slot', 'AWCi,Cio->AWCo', (xi[:1], D), 0),
        ]
        for form in FORMS:
            for what, string, operands, p in bad_calls:
                with self.subTest(case=what, form=form):
                    with self.assertRaisesRegex(ValueError, r"operand %d \(term '\w+'\) has \d slot\(s\) on its order axis" % p):
                        jet_contract(string, *operands, lens=lens, over_orders=form)
            jet_contract('TWCa,Caib,AWCi->TWCb', mu, G, xi, lens=lens, over_orders=form)      # the good calls pass
            jet_contract('TWCa,Caib,TWCi->TWCb', mu, G, xi_full, lens=lens, over_orders=form)
            jet_contract('TWCi,Cio->TWCo', xi_full, D, lens=lens, over_orders=form)
            jet_contract('AWCi,Cio->AWCo', xi, D, lens=lens, over_orders=form)
        # at order 0 there is no order axis to check: the same strings, one array less in rank
        lens0 = GroupLengths(T=0, W=1, K=0, C=0)
        jet_contract('TWCi,Cio->TWCo', xi_full[0], D, lens=lens0, over_orders='loop')


class TestStructure(unittest.TestCase):
    """Exact tests: counts of calls, dtypes, types, errors."""

    def _counted(self, string, arrays, lens, form):
        with unittest.mock.patch.object(contractions, 'contract', wraps=contractions.contract) as spy:
            got = jet_contract(string, *arrays, lens=lens, over_orders=form)
        return got, spy

    def test_order_zero_is_exactly_one_contract_call_on_the_string_as_written(self):
        rng = np.random.default_rng(0)
        for family, string in cases.ALL_STRINGS:
            for W, K, C in stack_matrix(string):
                shape_of = cases.shape_of_stacks(string, W, K, C)
                arrays, _ = cases.build(string, 0, shape_of, rng)
                lens = lens_of(0, shape_of)
                for form in FORMS:
                    with self.subTest(string=string, W=W, K=K, C=C, form=form):
                        got, spy = self._counted(string, arrays, lens, form)
                        self.assertEqual(1, spy.call_count)
                        (args, kwargs), = [(c.args, c.kwargs) for c in spy.call_args_list]
                        self.assertEqual(string, args[0])
                        self.assertTrue(all(a is b for a, b in zip(args[1:], arrays)))    # the operands, untouched
                        groups = {ch for ch in string if ch.isupper()}
                        self.assertEqual({'len_' + g for g in groups}, set(kwargs))       # every group it names
                        self.assertTrue(np.array_equal(got, contractions.contract(string, *arrays, **kwargs)))

    def test_number_of_contractions_above_order_zero(self):
        """A ride: 1. An affine partner: 2. Two full jets: order+1, under both forms. A readout: 1 for two
        full jets, times the coefficients of the third operand that are in range."""
        rng = np.random.default_rng(0)
        for order in (1, 3):
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=())
                arrays, _ = cases.build(string, order, shape_of, rng)
                for form in FORMS:
                    with self.subTest(string=string, order=order, form=form):
                        _, spy = self._counted(string, arrays, lens_of(order, shape_of), form)
                        self.assertEqual(cases.n_contractions(string, order), spy.call_count)
        # the counts themselves, spelled out at order 3 for one string of each kind
        for string, n in (('TWCo,Cio->TWCi', 1), ('AWo,Cio->AWCi', 1), ('TWCa,Caib,AWCi->TWCb', 2),
                          ('TWCa,Caib,TWCi->TWCb', 4), ('TWCi,AWo->Cio', 2), ('TWCi,TWCo->Cio', 1),
                          ('TWCa,AWCi,TWCb->Caib', 2), ('TWCa,TWCi,TWCb->Caib', 4)):
            self.assertEqual(n, cases.n_contractions(string, 3), string)

    def test_no_contraction_names_a_pair_of_orders(self):
        """No dense order-by-order form: every contraction that is made has at most one order axis among its
        operands' letters and at most one in its output (an (order+1)^2 intermediate would need two)."""
        rng = np.random.default_rng(0)
        for family, string in cases.ALL_STRINGS:
            shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=())
            arrays, _ = cases.build(string, 3, shape_of, rng)
            for form in FORMS:
                _, spy = self._counted(string, arrays, lens_of(3, shape_of), form)
                for call in spy.call_args_list:
                    made = call.args[0]
                    with self.subTest(string=string, form=form, made=made):
                        for term in made.replace('->', ',').split(','):                 # one order axis per array
                            self.assertLessEqual(sum(ch in 'TA' for ch in term), 1)
                        self.assertEqual(1, call.kwargs.get('len_T', 1))
                        self.assertEqual(1, call.kwargs.get('len_A', 1))
                        self.assertFalse(any(ch in 'trs' for ch in made))               # no table over the orders

    def test_float32_stays_float32(self):
        rng = np.random.default_rng(0)
        for order in ORDERS:
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=(2,))
                arrays, operands = cases.build(string, order, shape_of, rng)
                arrays = [x.astype(np.float32) for x in arrays]
                operands = [(x, groups, core) for x, (_, groups, core) in zip(arrays, operands)]
                ref = cases.reference(family, string, operands, order, shape_of)
                for form in FORMS:
                    with self.subTest(string=string, order=order, form=form):
                        got = jet_contract(string, *arrays, lens=lens_of(order, shape_of), over_orders=form)
                        self.assertEqual(np.float32, got.dtype)
                        self.assertLess(cases.relerr(got, ref), TOL_32)

    def test_numpy_in_numpy_out(self):
        rng = np.random.default_rng(0)
        for order in ORDERS:
            for family, string in cases.ALL_STRINGS:
                shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=())
                arrays, _ = cases.build(string, order, shape_of, rng)
                for form in FORMS:
                    with self.subTest(string=string, order=order, form=form):
                        got = jet_contract(string, *arrays, lens=lens_of(order, shape_of), over_orders=form)
                        self.assertIs(np.ndarray, type(got))

    def test_the_operands_are_not_changed(self):
        rng = np.random.default_rng(0)
        for family, string in cases.ALL_STRINGS:
            shape_of = cases.shape_of_stacks(string, W=(3,), K=(2,), C=())
            arrays, _ = cases.build(string, 3, shape_of, rng)
            copies = [x.copy() for x in arrays]
            for form in FORMS:
                jet_contract(string, *arrays, lens=lens_of(3, shape_of), over_orders=form)
            self.assertTrue(all(np.array_equal(x, c) for x, c in zip(arrays, copies)), string)

    def test_over_orders_has_no_default_and_only_two_values(self):
        lens0, lens1 = GroupLengths(T=0, W=1, K=0, C=0), GroupLengths(T=1, W=1, K=0, C=0)
        eta, D = np.ones((3, 4)), np.ones((4, 5))
        for lens, x in ((lens0, eta), (lens1, np.ones((2, 3, 4)))):         # at order 0 too, where it is not used
            for bad in ('dense', 'Scan', None, True):
                with self.subTest(T=lens.T, over_orders=bad):
                    with self.assertRaisesRegex(ValueError, 'over_orders'):
                        jet_contract('TWCi,Cio->TWCo', x, D, lens=lens, over_orders=bad)
            with self.assertRaises(TypeError):
                jet_contract('TWCi,Cio->TWCo', x, D, lens=lens)                 # no default
            with self.assertRaises(TypeError):
                jet_contract('TWCi,Cio->TWCo', x, D, over_orders='loop')        # no default for the record either
            with self.assertRaises(TypeError):
                jet_contract('TWCi,Cio->TWCo', x, D, lens, 'loop')              # keyword-only: no positional slip

    def test_lens_must_be_the_record(self):
        eta, D = np.ones((3, 4)), np.ones((4, 5))
        for bad in ((0, 1, 0, 0), dict(T=0, W=1, K=0, C=0), None):
            with self.subTest(lens=bad):
                with self.assertRaisesRegex(TypeError, 'GroupLengths'):
                    jet_contract('TWCi,Cio->TWCo', eta, D, lens=bad, over_orders='loop')

    def test_malformed_strings_raise_at_every_order(self):
        """Whether a string is a jet_contract string is decided from the string alone, so a call site is
        wrong at every order or at none."""
        one = np.ones(())
        cases_ = [
            ('WTCa,Caib,AWCi->TWCb', 'lead'),               # the order group does not lead an operand's term
            ('TWCa,Caib,AWCi->WTCb', 'lead'),               # ... or the output's
            ('TAWCa,Caib->TWCb', 'both'),                   # T and A in one term
            ('TWCa,TWCi,TWCb->TWCa', 'two operands'),       # three jets and T in the output
            ('TWCa,Caib,AWCi->AWCb', 'cannot have A'),      # a product of jets is a full jet
            ('AWCa,AWCi->Cai', 'needs an operand with T'),  # a top coefficient with nothing saying the order
            ('TWGa,Gab->TWGb', 'not groups'),               # a group the record does not count
            ('SWCi,SCio->SWCo', 'not groups'),              # ... S like any other: there is no site group,
            ('TSWCi,SCio->TSWCo', 'not groups'),            #     after the order group
            ('STWCi,SCio->STWCo', 'not groups'),            #     or before it
            ('TWCa,Caib->TWCb,', 'invalid'),                # the grouped-string parser's own errors pass through
            ('TWCa,Caib', "'->'"),
        ]
        for T in (0, 1):
            lens = GroupLengths(T=T, W=0, K=0, C=0)
            for string, message in cases_:
                with self.subTest(string=string, T=T):
                    with self.assertRaisesRegex(ValueError, message):
                        jet_contract(string, *([one] * len(string.split('->')[0].split(','))),
                                     lens=lens, over_orders='loop')

    def test_operand_count_mismatch_raises(self):
        lens = GroupLengths(T=1, W=1, K=0, C=0)
        with self.assertRaisesRegex(ValueError, 'operand'):
            jet_contract('TWCa,Caib,AWCi->TWCb', np.ones((4, 3, 2)), np.ones((2, 3, 4)), lens=lens, over_orders='loop')


class TestGroupLengths(unittest.TestCase):
    """The record is a static: frozen, with value equality and a value hash (it keys the memoized scan
    body, and is jax static data)."""

    def test_fields_in_order(self):
        self.assertEqual(['T', 'W', 'K', 'C'], [f.name for f in dc.fields(GroupLengths)])
        lens = GroupLengths(1, 2, 3, 4)
        self.assertEqual((1, 2, 3, 4), (lens.T, lens.W, lens.K, lens.C))
        with self.assertRaises(TypeError):
            GroupLengths(T=1, W=2, K=0)                         # no defaults: every count is said
        with self.assertRaises(TypeError):
            GroupLengths(S=0, T=1, W=2, K=0, C=0)               # there is no site group

    def test_value_equality_and_hash(self):
        a, b = GroupLengths(T=1, W=2, K=0, C=1), GroupLengths(T=1, W=2, K=0, C=1)
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(1, len({a, b}))
        for field in ('W', 'K', 'C'):
            self.assertNotEqual(a, dc.replace(a, **{field: getattr(a, field) + 1}))
        self.assertNotEqual(a, dc.replace(a, T=0))

    def test_frozen(self):
        with self.assertRaises(dc.FrozenInstanceError):
            GroupLengths(T=1, W=2, K=0, C=1).W = 3

    def test_counts_are_nonnegative_ints_and_T_is_zero_or_one(self):
        for bad in (dict(W=-1), dict(W=1.0), dict(W=None), dict(C=True), dict(K=(2,)), dict(T=2), dict(W=np.int64(2))):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    GroupLengths(**{**dict(T=1, W=2, K=0, C=1), **bad})

    def test_an_equal_record_gets_the_same_scan_body(self):
        """The scan body comes from a factory memoized on (string, record): an equal record built anew is
        the same key, so the body is ONE object and an eager jax scan finds its compiled program again.
        What the body closes over is static: strings, ints and a dict of ints, nothing else. (That no
        ARRAY reaches it other than through the carry and the scanned input is what the factory's
        signature guarantees: it is handed a string and the record, never an operand.)"""
        for string in ('TWCa,Caib,TWCb->TWCi', 'TWCa,TWCi,TWCb->Caib'):     # a product, and a readout of three
            a, b = GroupLengths(T=1, W=2, K=0, C=1), GroupLengths(T=1, W=2, K=0, C=1)
            body = contractions._jet_scan_body(string, a)
            self.assertIs(body, contractions._jet_scan_body(string, b))
            self.assertIsNot(body, contractions._jet_scan_body(string, dc.replace(a, W=1)))
            cells = [cell.cell_contents for cell in body.__closure__]
            self.assertEqual(5, len(cells))                     # the step string, its lengths, three positions
            for value in cells:
                with self.subTest(string=string, cell=value):
                    self.assertIn(type(value), (str, int, dict))
                    if type(value) is dict:
                        self.assertTrue(all(type(k) is str and type(v) is int for k, v in value.items()))


class TestOrderInsideAStep(unittest.TestCase):
    """Which two operands ONE step's contraction pairs first. Not a test of values: the result is the same
    in any order. It is a test of cost, and a marker.

    A product step contracts three things: the carried jet with all its ``order+1`` slots, the operands
    with no order group (a core), and one coefficient of the partner. The cheap order pairs the
    coefficient with the core first (the expensive contraction, done once), then meets every slot of
    the carried jet. Pairing the carried jet with the core first does the expensive contraction on every
    slot, at every step. A readout step should fold its two full jets along the order axis first.

    ``jet_contract`` leaves the order to ``contract``: one ``jnp.einsum`` with its default optimizer on
    jax, the forced greedy pairwise path on numpy. What each does is read here from the calls themselves:
    the strings are the ones ``jet_contract`` derives (spied, not copied), the roles of the operands are
    read off the literal and the derived string, and the path is the one handed to ``np.einsum`` or the
    one ``jnp.einsum_path`` reports for the same default ``jnp.einsum`` uses.
    """

    ORDER = 4
    SIZES = dict(a=12, i=6, b=12, o=20)                     # representative: bonds larger than the mode rank
    STACKS = dict(W=(200,), K=(2,), C=(2,))

    PRODUCTS = [string for _, string in cases.ALL_STRINGS if cases.is_product(string)]
    READOUTS = [string for family, string in cases.ALL_STRINGS if family == 'H']    # a pair and a third operand

    def _first_pairs(self, string, xnp):
        """For every 3-operand contraction ``jet_contract(string, ...)`` makes on backend ``xnp`` in which the
        carried jets have ALL their slots:
        ``(the derived string, the positions paired first, the carried positions, the no-order positions)``.

        The carried positions are the terms of the derived string that still have ``T``; the no-order
        positions are the terms with no order group in the literal; the remaining one is a coefficient.

        Only the steps with every slot are looked at, because jax's path depends on the sizes: under
        ``'loop'`` the later steps contract a short slice of the carried jet (down to one slot), where
        another order is as cheap or cheaper, and jax takes it. The step with every slot is the first of
        ``'loop'``, and it is every step of ``'scan'``, whose carried jet keeps all its slots."""
        shape_of = cases.shape_of_stacks(string, **self.STACKS)
        with unittest.mock.patch.dict(cases.SIZE, self.SIZES):
            arrays, _ = cases.build(string, self.ORDER, shape_of, np.random.default_rng(0))
        arrays = [xnp.asarray(x) for x in arrays]
        lens = GroupLengths(T=1, W=1, K=len(shape_of['K']), C=1)
        with unittest.mock.patch.object(contractions, 'contract', wraps=contractions.contract) as contract_spy, \
                unittest.mock.patch.object(xnp, 'einsum', wraps=xnp.einsum) as einsum_spy:
            jet_contract(string, *arrays, lens=lens, over_orders='loop')
        self.assertEqual(contract_spy.call_count, einsum_spy.call_count)       # one einsum per contraction
        literal_terms = cases.input_terms(string)
        no_order = frozenset(p for p, t in enumerate(literal_terms) if not ('T' in t or 'A' in t))
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
            carried = frozenset(p for p, t in enumerate(cases.input_terms(made)) if 'T' in t)
            if all(einsum_call.args[1 + p].shape[0] == self.ORDER + 1 for p in carried):
                steps.append((made, frozenset(int(p) for p in first), carried, no_order))
        self.assertGreater(len(steps), 0)
        return steps

    @staticmethod
    def _jax_default_optimize():
        import inspect
        import jax.numpy as jnp
        return inspect.signature(jnp.einsum).parameters['optimize'].default

    @unittest.skipUnless(common.jax_available, 'jax not available')
    def test_jax_default_optimizer_is_the_one_read_here(self):
        # the value is recorded so that a change of jax's default is seen, not silently followed
        self.assertEqual('auto', self._jax_default_optimize())

    @unittest.skipUnless(common.jax_available, 'jax not available')
    def test_on_jax_a_product_step_pairs_the_coefficient_with_the_core_first(self):
        import jax.numpy as jnp
        self.assertEqual(20, len(self.PRODUCTS))
        for string in self.PRODUCTS:
            for made, first, carried, no_order in self._first_pairs(string, jnp):
                with self.subTest(string=string, made=made):
                    self.assertEqual(1, len(carried))
                    self.assertFalse(first & carried, 'the carried jet is contracted first')
                    self.assertEqual(1, len(first & no_order))      # the coefficient with the core

    def test_on_numpy_a_readout_step_folds_its_two_full_jets_first(self):
        self.assertEqual(8, len(self.READOUTS))
        for string in self.READOUTS:
            for made, first, carried, no_order in self._first_pairs(string, np):
                with self.subTest(string=string, made=made):
                    self.assertEqual(2, len(carried))               # the pair, each with its order axis
                    self.assertEqual(carried, first)

    @unittest.skipUnless(common.jax_available, 'jax not available')
    def test_on_jax_a_readout_step_folds_its_two_full_jets_first(self):
        import jax.numpy as jnp
        for string in self.READOUTS:
            for made, first, carried, no_order in self._first_pairs(string, jnp):
                with self.subTest(string=string, made=made):
                    self.assertEqual(2, len(carried))
                    self.assertEqual(carried, first)

    def test_on_numpy_what_a_product_step_pairs_first_today(self):
        """A plain, passing test of what numpy does today, so that the helper behind the expected failure
        below is exercised on numpy product strings by a test that can fail (``expectedFailure`` accepts
        ANY failure, a broken helper included).

        Every product string has exactly one step in which the carried jet has all its slots, and its
        first pair is two of its three operands. For the four ``nu`` strings with an affine leaf the
        first pair is the core with the coefficient: the cheap order, by the accident of the operand
        order. For the other 16 it is the carried jet with the core. The count 16 against 4 is pinned
        here: when Part 2 (row 5.5) fixes the order it becomes 0 against 20, and this test is then
        updated together with the marker below."""
        cheap_today = [string for string in cases.STRINGS['nu'] if 'A' in string]       # an affine leaf
        self.assertEqual(4, len(cheap_today))
        costly, cheap = [], []
        for string in self.PRODUCTS:
            steps = self._first_pairs(string, np)
            with self.subTest(string=string):
                self.assertEqual(1, len(steps))                     # 'loop': only the first step has every slot
                (made, first, carried, no_order), = steps
                self.assertEqual(3, len(cases.input_terms(made)))
                self.assertEqual(1, len(carried))
                self.assertEqual(1, len(no_order))
                self.assertFalse(carried & no_order)
                self.assertEqual(2, len(first))
                self.assertTrue(first <= {0, 1, 2})
                if string in cheap_today:
                    self.assertFalse(first & carried)               # the coefficient ...
                    self.assertEqual(no_order, first & no_order)    # ... with the core
                else:
                    self.assertEqual(carried | no_order, first)     # the carried jet with the core
            (costly if first & carried and first & no_order else cheap).append(string)
        self.assertEqual(16, len(costly))
        self.assertEqual(sorted(cheap_today), sorted(cheap))

    @unittest.expectedFailure
    def test_on_numpy_no_product_step_pairs_the_carried_jet_with_the_core_first(self):
        """AN EXPECTED FAILURE, kept as a marker for PART 2 (row 5.5, the contraction order).

        What it asserts is what should hold and does not hold today: on numpy, no product step contracts
        the carried jet with an operand that has no order group first. Today ``contract`` forces
        ``_pairwise_path`` on numpy; a product step is a three-way tie there, the tie goes to the first
        two operands, and for 16 of the 20 product strings (every ``mu`` and ``eta`` string, and the
        ``nu`` strings of two full jets) those are the carried jet and the core. So the expensive
        contraction is done on every slot of the carried jet at every step. The values are right; only
        the cost is wrong. The maintainer deferred the fix to Part 2 (2026-11-17), with the instruction
        that it be marked so that it is not forgotten. This test is that mark: it is reported as an
        expected failure in every run.

        WHEN THIS TEST STARTS PASSING (the suite then reports an unexpected success, which is a
        failure): the order was fixed. Remove the ``expectedFailure`` decorator, so that the test
        guards the fix, update the counts of ``test_on_numpy_what_a_product_step_pairs_first_today``
        (which fails at the same moment), and remove the PART 2 (row 5.5) notes in
        ``backend/contractions.py``.
        """
        offenders = []
        for string in self.PRODUCTS:
            for made, first, carried, no_order in self._first_pairs(string, np):
                if first & carried and first & no_order:
                    offenders.append(made)
        self.assertEqual([], sorted(set(offenders)))


if __name__ == '__main__':
    unittest.main()
