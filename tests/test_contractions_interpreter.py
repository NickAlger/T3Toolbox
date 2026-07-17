# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The standing test suite for the grouped-einsum interpreter ``contractions.contract``.

The interpreter is a single implementation serving every grouped contraction in the library, so its
suite carries the weight the per-function oracles used to. Independent lines of evidence:

1. **The definitional loop oracle** (`TestVocabularyOracle`): a grouped contraction MEANS "the
   single-axis contraction mapped over every group index tuple (summing groups absent from the
   output)". The oracle executes that definition directly -- slice every operand, einsum the
   lowercase-only subscript, accumulate -- sharing no mechanism with the interpreter (which solves
   ndim equations and expands letters). It runs over the FULL VOCABULARY: every subscript literal
   AST-scanned from the library source (self-maintaining -- a new call site is covered the moment
   it is written) plus the frozen historical table below.
2. **The frozen identifiability contract** (`TestSupplementContract`): for every vocabulary string,
   whether ``contract`` demands a ``len_<G>`` supplement -- pinned as DATA. Seeded 2026-07-17 from
   the hand analysis encoded in the (since deleted) named contractions' n_probe/n_frame signatures,
   which the rank solve was verified to reproduce exactly. Guards the solver and the co-travel
   merge against silently accepting an underdetermined string (the silent-reinterpretation class).
3. **Call-site consistency** (`TestCallSiteConsistency`): every literal ``contract(...)`` call site
   in the library supplies a sufficient supplement set for its string, checked from an AST scan --
   a future call site with a missing/typo'd ``len_*`` fails here before it ever runs.
4. **Split invariance** (`TestSplitInvariance`): for co-traveling group runs the split is
   unobservable -- all valid ``len_*`` splits (and no supplement at all) must give the IDENTICAL
   expanded einsum, hence bitwise-equal results.
5. Generic strings beyond the library, and the error contract.

(Sharding -- every sub-axis of every group free -- is compiled evidence and lives in
``test_contractions_sharding.py``. jax/jit dispatch is in ``test_dispatch.py``. The consumer-level
ground truths -- dense ``np.einsum`` references that never touch the interpreter -- are the
library-wide anchor; see ``docs/contributor/testing_strategy.md``.)

Numpy-only, per the house convention.
"""
import ast
import itertools
import pathlib
import re
import unittest

import numpy as np

import t3toolbox
from t3toolbox.backend.contractions import contract

# single-axis sizes, shared across all operands of a contraction (same letter = same size)
AXIS_SIZE = {'d': 2, 't': 3, 'r': 3, 's': 2, 'u': 3, 'k': 2, 'a': 2, 'i': 3, 'b': 4, 'o': 5,
             'j': 2, 'c': 3}

# (W_shape, K_shape, C_shape): empty / single / multi-axis blocks, where the bookkeeping bites
SHAPE_MATRIX = [
    ((), (), ()),
    ((5,), (2,), (3,)),
    ((), (2,), (3,)),
    ((5,), (), (3,)),
    ((5,), (2,), ()),
    ((2, 3), (2,), (2, 2)),
    ((2, 3), (2, 2), (2, 3)),
]

# The frozen identifiability table: subscripts -> the supplement block `contract` must demand
# ('' = fully determined from the operand ndims). Frozen 2026-07-17: the 104 named-contraction
# entries carry the hand analysis from their n_probe (-> 'W') / n_frame (-> 'C') signatures (the
# rank solve was verified to reproduce every one before the functions were deleted); the remaining
# entries are the lean-jet scan-step strings, whose requirements were hand-derived at extraction.
# A NEW library subscript must be added here (TestVocabularyCompleteness enforces it) -- decide its
# expected supplement by hand first; the test then pins the solver against your analysis.
HISTORICAL = {
    'CWa,Caib,CiW->CWb': '',
    'CWa,Caib,Wo,Cio->CWb': '',
    'Caib,kWCi->kWCab': '',
    'Caib,sWCi->sWCab': '',
    'Caib,sWKCi->sWKCab': '',
    'Cio,Wo->WCi': '',
    'KCaib,sWCi->sWKCab': 'C',
    'WCa,Caib->WCib': '',
    'WCa,Caib,WCb->WCi': '',
    'WCa,Caib,WCi->WCb': '',
    'WCa,Caib,WKCb->WKCi': '',
    'WCa,Caib,WKCi->WKCb': '',
    'WCa,Caib,Wo,Cio->WCb': '',
    'WCa,KCaib->WKCib': 'C',
    'WCa,KCaib,WCb->WKCi': 'C',
    'WCa,KCaib,WCi->WKCb': 'C',
    'WCa,WCi,WKCb->KCaib': 'W',
    'WCa,WCi,WKCb->WKCaib': 'W',
    'WCi,Cio->WCo': '',
    'WCi,KCio->WKCo': 'C',
    'WCi,WCa,WCj->Ciaj': 'W',
    'WCi,WCa,WKCj->KCiaj': 'W',
    'WCi,WCa,WKCj->WKCiaj': 'W',
    'WCi,WKCa,WCj->KCiaj': 'W',
    'WCi,WKCa,WCj->WKCiaj': 'W',
    'WCib,sWCb->sWCi': '',
    'WCib,sWKCb->sWKCi': 'C',
    'WCib,tWKCi->tWKCb': 'C',
    'WCo,Cio->WCi': '',
    'WCo,WCa->Cao': 'W',
    'WKCa,Caib->WKCib': '',
    'WKCa,Caib,WCb->WKCi': '',
    'WKCa,Caib,WCi->WKCb': '',
    'WKCi,Cio->WKCo': '',
    'WKCi,WCa,WCj->KCiaj': 'W',
    'WKCi,WCa,WCj->WKCiaj': 'W',
    'WKCib,sWCb->sWKCi': 'C',
    'WKCo,WCa->KCao': 'W',
    'WKCo,WCa->WKCao': 'W',
    'Wa,Caib,Wi->WCb': '',
    'Wo,WCa->Cao': '',
    'Wo,WKCa->KCao': '',
    'Wo,WKCa->WKCao': '',
    'dCio,dWo->dWCi': '',
    'dWCa,dCaib,dWCb->dWCi': '',
    'dWCa,dCaib,dWKCb->dWKCi': '',
    'dWCa,dKCaib,dWCb->dWKCi': 'C',
    'dWCa,dWCi,dWKCb->dKCaib': 'W',
    'dWCa,dWCi,dWKCb->dWKCaib': 'W',
    'dWCi,dCio->dWCo': '',
    'dWCi,dKCio->dWKCo': 'C',
    'dWCi,dWCa,dWKCj->dKCiaj': 'W',
    'dWCi,dWCa,dWKCj->dWKCiaj': 'W',
    'dWCi,dWKCa,dWCj->dKCiaj': 'W',
    'dWCi,dWKCa,dWCj->dWKCiaj': 'W',
    'dWCo,dCio->dWCi': '',
    'dWKCa,dCaib,dWCb->dWKCi': '',
    'dWKCi,dCio->dWKCo': '',
    'dWKCi,dWCa,dWCj->dKCiaj': 'W',
    'dWKCi,dWCa,dWCj->dWKCiaj': 'W',
    'dWKCo,dWCa->dKCao': 'W',
    'dWKCo,dWCa->dWKCao': 'W',
    'dWo,dWKCa->dKCao': '',
    'dWo,dWKCa->dWKCao': '',
    'dtWCa,dtWKCo->dKCao': 'W',
    'dtWCa,dtWKCo->dWKCao': 'W',
    'dtWCi,dCio->dtWCo': '',
    'dtWCi,dKCio->dtWKCo': 'C',
    'dtWCo,dCio->dtWCi': '',
    'dtWKCi,dCio->dtWKCo': '',
    'dtWKCo,dCio->dtWKCi': '',
    'duWKCa,duWo->dKCao': '',
    'duWKCa,duWo->dWKCao': '',
    'ksWKCa,kWCab->sWKCb': 'C',
    'sWKCa,WCab->sWKCb': 'C',
    'stWCa,sWCab->tWCb': '',
    'stWCa,sWKCab->tWKCb': 'C',
    'stWKCa,sWCab->tWKCb': 'C',
    'tWCa,WCab->tWCb': '',
    'tWCa,WKCab->tWKCb': 'C',
    'tWCa,tWKCo->KCao': 'W',
    'tWCa,tWKCo->WKCao': 'W',
    'tWCi,Cio->tWCo': '',
    'tWCi,KCio->tWKCo': 'C',
    'tWCo,Cio->tWCi': '',
    'tWKCa,WCab->tWKCb': 'C',
    'tWKCi,Cio->tWKCo': '',
    'tWKCo,Cio->tWKCi': '',
    'trs,drWCa,dCaib,dsWCb->dtWCi': '',
    'trs,drWCa,dCaib,dsWKCb->dtWKCi': '',
    'trs,drWCa,dCaib,dtWKCb->dsWKCi': '',
    'trs,drWCa,dKCaib,dsWCb->dtWKCi': 'C',
    'trs,drWCa,dsWCi,dtWKCb->dKCaib': 'W',
    'trs,drWCa,dsWCi,dtWKCb->dWKCaib': 'W',
    'trs,drWCa,dtWKCi,dsWCb->dKCaib': 'W',
    'trs,drWCa,dtWKCi,dsWCb->dWKCaib': 'W',
    'trs,drWKCa,dCaib,dsWCb->dtWKCi': '',
    'trs,dtWKCa,dCaib,dsWCb->drWKCi': '',
    'trs,dtWKCa,drWCi,dsWCb->dKCaib': 'W',
    'trs,dtWKCa,drWCi,dsWCb->dWKCaib': 'W',
    'trs,rWCa,Caib,sWCb->tWCi': '',
    'trs,rWCa,Caib,sWCi->tWCb': '',
    'trs,rWCa,Caib,sWKCb->tWKCi': '',
    'trs,rWCa,Caib,sWKCi->tWKCb': '',
    'trs,rWCa,Caib,tWKCb->sWKCi': '',
    'trs,rWCa,Caib,tWKCi->sWKCb': '',
    'trs,rWCa,KCaib,sWCb->tWKCi': 'C',
    'trs,rWCa,KCaib,sWCi->tWKCb': 'C',
    'trs,rWCa,sWCi,tWKCb->KCaib': 'W',
    'trs,rWCa,sWCi,tWKCb->WKCaib': 'W',
    'trs,rWCa,tWKCi,sWCb->KCaib': 'W',
    'trs,rWCa,tWKCi,sWCb->WKCaib': 'W',
    'trs,rWKCa,Caib,sWCb->tWKCi': '',
    'trs,rWKCa,Caib,sWCi->tWKCb': '',
    'trs,tWKCa,Caib,rWCi->sWKCb': '',
    'trs,tWKCa,Caib,sWCb->rWKCi': '',
    'trs,tWKCa,rWCi,sWCb->KCaib': 'W',
    'trs,tWKCa,rWCi,sWCb->WKCaib': 'W',
    'ts,sWCi->tWCi': '',
    'ts,sWKCi->tWKCi': '',
    'ts,tWKCb->sWKCb': '',
    'uWKCa,uWo->KCao': '',
    'uWKCa,uWo->WKCao': '',
}

SUBS_RE = re.compile(r'^[A-Za-z]+(,[A-Za-z]+)*->[A-Za-z]*$')
LIBRARY_ROOT = pathlib.Path(t3toolbox.__file__).parent


def scan_vocabulary():
    """Every grouped-subscript string literal in the library source (whitespace-stripped).
    Self-maintaining: a new call site's string is swept into the oracle the moment it is written,
    whether it appears inline in the call or assigned to a variable first."""
    vocab = set()
    for path in sorted(LIBRARY_ROOT.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value.replace(' ', '')
                if SUBS_RE.match(s) and any(ch.isupper() for ch in s):
                    vocab.add(s)
    return vocab


def scan_literal_call_sites():
    """(path, lineno, subscripts, supplied len-letters) for every ``contract(<string literal>, ...)``
    call in the library. Calls whose first argument is a variable (the keep-W/sum-W selection
    sites) are not listed here -- their strings are still swept by ``scan_vocabulary``."""
    sites = []
    for path in sorted(LIBRARY_ROOT.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else None)
            if fname != 'contract' or not node.args:
                continue
            first = node.args[0]
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                continue
            lens = frozenset(kw.arg[4] for kw in node.keywords
                             if kw.arg and kw.arg.startswith('len_'))
            sites.append((str(path.relative_to(LIBRARY_ROOT)), node.lineno,
                          first.value.replace(' ', ''), lens))
    return sites


def groups_of(subscripts):
    return sorted({ch for ch in subscripts if ch.isupper()})


def build_operands(subscripts, shapes, rng):
    """Random operands for a grouped subscripts string; ``shapes`` maps group letter -> shape."""
    ops = []
    for term in subscripts.split('->')[0].split(','):
        shape = ()
        for ch in term:
            shape += shapes[ch] if ch.isupper() else (AXIS_SIZE[ch],)
        ops.append(rng.standard_normal(shape))
    return ops


def all_lens(subscripts, shapes):
    """Supplements for every group -- always a sufficient (and verified-redundant) set."""
    return {'len_' + g: len(shapes[g]) for g in groups_of(subscripts)}


def loop_oracle(subscripts, operands, group_shapes):
    """The definitional reference: the lowercase-only contraction mapped over every group index
    tuple, accumulating (+=) over groups absent from the output."""
    s = subscripts.replace(' ', '')
    lhs, out = s.split('->')
    terms = lhs.split(',')

    size = {}
    for t, op in zip(terms, operands):
        pos = 0
        for ch in t:
            if ch.isupper():
                pos += len(group_shapes[ch])
            else:
                size[ch] = op.shape[pos]
                pos += 1

    groups = sorted({ch for t in terms + [out] for ch in t if ch.isupper()})
    base = (','.join(''.join(c for c in t if c.islower()) for t in terms)
            + '->' + ''.join(c for c in out if c.islower()))
    out_shape = ()
    for ch in out:
        out_shape += group_shapes[ch] if ch.isupper() else (size[ch],)

    result = np.zeros(out_shape)
    for assign in itertools.product(*[list(np.ndindex(*group_shapes[g])) for g in groups]):
        gidx = dict(zip(groups, assign))
        sliced = []
        for t, op in zip(terms, operands):
            index = ()
            for ch in t:
                index += gidx[ch] if ch.isupper() else (slice(None),)
            sliced.append(op[index])
        oidx = ()
        for ch in out:
            oidx += gidx[ch] if ch.isupper() else (slice(None),)
        result[oidx] += np.einsum(base, *sliced)
    return result


class TestVocabularyCompleteness(unittest.TestCase):
    """Every scanned library subscript must have a frozen expectation in HISTORICAL. This is the
    reverse of a hand inventory: the AST scan drives coverage (it cannot under-count), and this
    test only forces each new string's identifiability to be recorded deliberately."""

    def test_every_library_subscript_is_frozen(self):
        vocab = scan_vocabulary()
        self.assertGreater(len(vocab), 100, 'the AST scan found suspiciously few subscripts')
        missing = vocab - set(HISTORICAL)
        self.assertFalse(
            missing,
            'library subscripts with no frozen identifiability expectation (add each to '
            'HISTORICAL in this file, deciding its required supplement BY HAND first): %s'
            % sorted(missing))


class TestVocabularyOracle(unittest.TestCase):
    """contract(...) == the definition, executed directly, over the FULL vocabulary x the full
    block-shape matrix -- through both the all-supplements path and the inference path."""

    def test_definitional_oracle_full_vocabulary(self):
        rng = np.random.default_rng(0)
        for subs in sorted(set(HISTORICAL) | scan_vocabulary()):
            req = HISTORICAL.get(subs, None)
            for W, K, C in SHAPE_MATRIX:
                with self.subTest(subs=subs, W=W, K=K, C=C):
                    shapes = {'W': W, 'K': K, 'C': C}
                    for g in groups_of(subs):
                        shapes.setdefault(g, (2,))
                    ops = build_operands(subs, shapes, rng)
                    ref = loop_oracle(subs, ops, shapes)

                    got = contract(subs, *ops, **all_lens(subs, shapes))
                    self.assertEqual(got.shape, ref.shape)
                    self.assertTrue(np.allclose(got, ref, rtol=1e-10, atol=1e-12))

                    if req is not None:     # the inference path: only the required supplement
                        lens = {'len_' + req: len(shapes[req])} if req else {}
                        got2 = contract(subs, *ops, **lens)
                        self.assertTrue(np.array_equal(np.asarray(got), np.asarray(got2)))


class TestSupplementContract(unittest.TestCase):
    """The identifiability contract, pinned as data: contract demands a supplement exactly for the
    strings HISTORICAL says are underdetermined -- and is satisfied by exactly the recorded block.
    Guards the rank solver and the co-travel merge against silently accepting an underdetermined
    string (a wrong-but-plausible arbitrary split -- the silent-reinterpretation bug class)."""

    def test_demands_match_the_frozen_analysis(self):
        rng = np.random.default_rng(0)
        shapes = {'W': (2, 3), 'K': (2, 2), 'C': (2, 3)}    # generic multi-axis point (ndim-only)
        for subs, req in HISTORICAL.items():
            with self.subTest(subs=subs, req=req):
                ops = build_operands(subs, shapes, rng)
                if req:
                    with self.assertRaisesRegex(ValueError, 'do not determine'):
                        contract(subs, *ops)
                    contract(subs, *ops, **{'len_' + req: len(shapes[req])})
                else:
                    contract(subs, *ops)


class TestCallSiteConsistency(unittest.TestCase):
    """Every literal contract(...) call site in the library supplies a SUFFICIENT supplement set:
    the call must succeed with exactly the len_* letters the site passes (on generic multi-axis
    operands). A new call site with a missing or unknown len_* fails here before it ever runs."""

    def test_every_literal_call_site_supplies_enough(self):
        rng = np.random.default_rng(0)
        sites = scan_literal_call_sites()
        self.assertGreater(len(sites), 90, 'the AST scan found suspiciously few call sites')
        shapes = {'W': (2, 3), 'K': (2, 2), 'C': (2, 3)}
        for path, lineno, subs, lens in sites:
            with self.subTest(site='%s:%d' % (path, lineno), subs=subs, lens=sorted(lens)):
                self.assertTrue(set(lens) <= set(groups_of(subs)),
                                'call site supplies len_* for a letter not in its subscripts')
                ops = build_operands(subs, shapes, rng)
                contract(subs, *ops, **{'len_' + g: len(shapes[g]) for g in lens})


class TestSplitInvariance(unittest.TestCase):
    """For co-traveling runs the boundary is unobservable: every valid split (and no supplement at
    all) must yield the IDENTICAL expanded einsum, hence bitwise-equal results."""

    # (subscripts, the co-traveling run, shapes for groups outside the run)
    CASES = [
        ('WKCi,Cio->WKCo', 'WK', {'C': (2,)}),
        ('tWKCo,Cio->tWKCi', 'WK', {'C': (2,)}),
        ('Caib,sWKCi->sWKCab', 'WK', {'C': (2,)}),
        ('dWo,dWKCa->dKCao', 'KC', {'W': (2,)}),
        ('ts,sWKCi->tWKCi', 'WKC', {}),
    ]

    def test_all_valid_splits_agree_bitwise(self):
        rng = np.random.default_rng(0)
        run_shape = (2, 3)      # the merged run's axes; splits assign them to the run's members
        for subs, run, other in self.CASES:
            with self.subTest(subs=subs, run=run):
                shapes = dict(other)
                shapes[run[0]] = run_shape                  # build operands once: run leads with
                for g in run[1:]:                           # both axes, later members empty
                    shapes[g] = ()
                ops = build_operands(subs, shapes, rng)
                base = np.asarray(contract(subs, *ops))     # no supplement: the merge infers
                n_splits = 0
                for cut in itertools.combinations(range(len(run_shape) + 1), len(run) - 1):
                    bounds = (0,) + cut + (len(run_shape),)
                    lens = {'len_' + g: bounds[k + 1] - bounds[k] for k, g in enumerate(run)}
                    got = np.asarray(contract(subs, *ops, **lens))
                    self.assertTrue(np.array_equal(base, got),
                                    'split %s changed the result' % (lens,))
                    n_splits += 1
                self.assertGreaterEqual(n_splits, 3)


class TestGenericStrings(unittest.TestCase):
    """Strings with no library counterpart: the dialect is general, not a T3 special case."""

    def test_generic_strings_vs_loop_oracle(self):
        rng = np.random.default_rng(1)
        cases = [
            ('WCa->Ca', {'len_W': 1}, {'W': (4,), 'C': (2,)}),                 # sum a lone group
            ('Wa, Kb -> WKab', {}, {'W': (2, 2), 'K': (3,)}),                  # pure outer product
            ('Ga,Gb,Gc->Gabc', {}, {'G': (2, 3)}),                             # shared 3-operand batch
            ('WCo,WCa->Cao', {'len_W': 2}, {'W': (2, 2), 'C': (3,)}),          # the n_probe archetype
        ]
        for subs, lens, shapes in cases:
            with self.subTest(subs=subs):
                s = subs.replace(' ', '')
                ops = build_operands(s, shapes, rng)
                got = contract(subs, *ops, **lens)
                ref = loop_oracle(s, ops, shapes)
                self.assertEqual(got.shape, ref.shape)
                self.assertTrue(np.allclose(got, ref, rtol=1e-10, atol=1e-12))


class TestErrorsAndValidation(unittest.TestCase):

    def test_underdetermined_names_the_missing_lengths(self):
        with self.assertRaisesRegex(ValueError, r'len_W'):
            contract('WCo,WCa->Cao', np.ones((5, 2, 3)), np.ones((5, 2, 6)))

    def test_underdetermined_is_decided_from_the_string_not_the_shapes(self):
        # ndim-0 prefixes would pin W=C=() instance-wise; the interpreter must still demand len_*
        with self.assertRaisesRegex(ValueError, 'do not determine'):
            contract('WCo,WCa->Cao', np.ones(3), np.ones(6))
        got = contract('WCo,WCa->Cao', np.ones(3), np.ones(6), len_W=0)
        self.assertEqual(got.shape, (6, 3))

    def test_redundant_supplement_verified(self):
        A, x = np.ones((2, 4, 3)), np.ones((5, 2, 3))
        ok = contract('Cio,WCo->WCi', A, x, len_C=1)        # redundant but consistent
        self.assertEqual(ok.shape, (5, 2, 4))
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            contract('Cio,WCo->WCi', A, x, len_C=2)         # redundant and wrong

    def test_group_shape_mismatch_raises(self):
        # same flattened size, transposed axes: silent garbage under a flatten, an error here
        with self.assertRaises(ValueError):
            contract('Wa,Wb->Wab', np.ones((2, 3, 4)), np.ones((3, 2, 5)), len_W=2)

    def test_ndim_inconsistent_with_terms_raises(self):
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            contract('trs,rWCa->tWCa', np.ones((3, 3)), np.ones((3, 5, 2, 4)))  # trs must be 3-d
        with self.assertRaisesRegex(ValueError, 'ndim'):
            contract('ij,jk->ik', np.ones((3, 4, 2)), np.ones((4, 5)))          # no groups branch

    def test_malformed_subscripts_raise(self):
        for bad in ['WCo,WCa', '...a,ab->b', 'W!a,ab->b', 'WWa,b->ab', 'Wa,,b->ab', 'Wa,b->Wc']:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    contract(bad, np.ones((2, 2)), np.ones((2, 2)))

    def test_operand_count_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, 'operand'):
            contract('Wa,Wb->Wab', np.ones((2, 2)))

    def test_bad_keyword_arguments_raise(self):
        with self.assertRaisesRegex(TypeError, 'len_'):
            contract('Wa->Wa', np.ones((2, 2)), n_probe=1)
        with self.assertRaisesRegex(ValueError, 'does not appear'):
            contract('Wa->Wa', np.ones((2, 2)), len_K=1)
        with self.assertRaises(TypeError):
            contract('Wa->Wa', np.ones((2, 2)), len_W=1.5)

    def test_negative_solved_length_raises(self):
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            contract('Wab,c->Wabc', np.ones(1), np.ones(3))     # forces len_W = -1


class TestMigrationCrossChecks(unittest.TestCase):
    """TEMPORARY -- delete this class together with the named contractions (interpreter slice 4).

    While the 104 named contractions still exist they are an independent second implementation:
    (a) the frozen HISTORICAL table must match their live signatures (the table's provenance), and
    (b) contract must equal each of them over the block-shape matrix (the differential sweep that
    transferred their verification history to the interpreter)."""

    @staticmethod
    def _named_contractions():
        import inspect
        import t3toolbox.backend.contractions as ctr
        out = []
        for name, fn in inspect.getmembers(ctr, inspect.isfunction):
            if name.startswith('_') or fn.__module__ != ctr.__name__ or '_to_' not in name:
                continue
            out.append((name, fn))
        return out

    @staticmethod
    def _to_subscripts(name):
        lhs, out = name.rsplit('_to_', 1)
        return ','.join(lhs.split('_')) + '->' + out

    def test_frozen_table_matches_live_signatures(self):
        import inspect
        fns = self._named_contractions()
        self.assertGreater(len(fns), 90)
        for name, fn in fns:
            with self.subTest(name=name):
                params = inspect.signature(fn).parameters
                expected = 'W' if 'n_probe' in params else ('C' if 'n_frame' in params else '')
                self.assertEqual(HISTORICAL[self._to_subscripts(name)], expected)

    def test_differential_vs_named(self):
        import inspect
        rng = np.random.default_rng(0)
        for name, fn in self._named_contractions():
            subs = self._to_subscripts(name)
            params = inspect.signature(fn).parameters
            for W, K, C in SHAPE_MATRIX:
                with self.subTest(name=name, W=W, K=K, C=C):
                    shapes = {'W': W, 'K': K, 'C': C}
                    ops = build_operands(subs, shapes, rng)
                    kwargs_old, kwargs_new = {}, {}
                    if 'n_probe' in params:
                        kwargs_old['n_probe'] = len(W)
                        kwargs_new['len_W'] = len(W)
                    if 'n_frame' in params:
                        kwargs_old['n_frame'] = len(C)
                        kwargs_new['len_C'] = len(C)
                    expected = fn(*ops, **kwargs_old)
                    got = contract(subs, *ops, **kwargs_new)
                    self.assertEqual(got.shape, expected.shape)
                    self.assertTrue(np.allclose(got, expected, rtol=1e-10, atol=1e-12))


if __name__ == '__main__':
    unittest.main()
