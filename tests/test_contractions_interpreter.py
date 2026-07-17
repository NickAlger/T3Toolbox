# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Tests for the grouped-einsum interpreter ``contractions.contract``.

Three independent lines of evidence, plus the error contract:

1. **The definitional loop oracle** (`TestLoopOracle`): a grouped contraction MEANS "the single-axis
   contraction mapped over every group index (summing groups absent from the output)". The oracle
   executes that definition directly -- loop over the product of all group index tuples, slice every
   operand, einsum the lowercase-only subscript, accumulate. It shares no mechanism with the
   interpreter (which solves ndim equations and expands letters), so it checks the semantics.
2. **The differential sweep vs the named contractions** (`TestDifferentialVsNamed`): every public
   named contraction in the module is one `contract(...)` call; sweep all of them against the
   interpreter over an empty/single/multi-axis block-shape matrix. The hand-written flatten-based
   bodies (each with its own oracle test) serve as the independent reference implementation.
3. **The supplement analysis** (`TestSupplementAnalysis`): the interpreter demands a ``len_<G>``
   argument exactly when the subscripts cannot pin the group sizes -- a rank condition. The named
   contractions encode the SAME analysis, done by hand: they take ``n_probe``/``n_frame`` exactly
   when needed. Assert the two analyses agree on all ~100 names.

Numpy-only, per the house convention (jax dispatch is covered in ``test_dispatch.py``).
"""
import inspect
import itertools
import unittest

import numpy as np

import t3toolbox.backend.contractions as contractions
from t3toolbox.backend.contractions import contract

# single-axis sizes, shared across all operands of a contraction (same letter = same size)
AXIS_SIZE = {'d': 2, 't': 3, 'r': 3, 's': 2, 'u': 3, 'a': 2, 'i': 3, 'b': 4, 'o': 5, 'j': 2, 'c': 3}

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


def named_contractions():
    """Every public named contraction, enumerated from the module's own functions (not __all__),
    mirroring test_contractions_sharding."""
    out = []
    for name, fn in inspect.getmembers(contractions, inspect.isfunction):
        if name.startswith('_') or fn.__module__ != contractions.__name__ or '_to_' not in name:
            continue
        out.append((name, fn))
    return out


def to_subscripts(name):
    """Transliterate a contraction NAME to the grouped-einsum string: 'WCo_WCa_to_Cao' -> 'WCo,WCa->Cao'."""
    lhs, out = name.rsplit('_to_', 1)
    return ','.join(lhs.split('_')) + '->' + out


def build_operands(fn, W, K, C, rng):
    """Random operands for a named contraction, shaped from its parameter-name tokens."""
    tokens = [p for p in inspect.signature(fn).parameters if p not in ('n_probe', 'n_frame')]
    block = {'W': W, 'K': K, 'C': C}
    ops = []
    for tok in tokens:
        shape = ()
        for ch in tok:
            shape += block[ch] if ch.isupper() else (AXIS_SIZE[ch],)
        ops.append(rng.standard_normal(shape))
    return ops


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


class TestSupplementAnalysis(unittest.TestCase):
    """The interpreter's rank analysis reproduces the hand analysis encoded in the named
    contractions' signatures: it demands a supplement exactly when the name carries
    n_probe/n_frame, and the mapped supplement (n_probe -> len_W, n_frame -> len_C) suffices."""

    def test_needed_supplements_match_handwritten_signatures(self):
        rng = np.random.default_rng(0)
        W, K, C = (2, 3), (2, 2), (2, 3)    # generic multi-axis point (analysis is ndim-only)
        n_checked = 0
        for name, fn in named_contractions():
            with self.subTest(name=name):
                params = inspect.signature(fn).parameters
                needs_old = {p for p in params if p in ('n_probe', 'n_frame')}
                ops = build_operands(fn, W, K, C, rng)
                subs = to_subscripts(name)
                if needs_old:
                    with self.assertRaisesRegex(ValueError, 'do not determine'):
                        contract(subs, *ops)
                lens = {}
                if 'n_probe' in needs_old:
                    lens['len_W'] = len(W)
                if 'n_frame' in needs_old:
                    lens['len_C'] = len(C)
                contract(subs, *ops, **lens)    # must succeed with exactly the mapped supplements
                n_checked += 1
        self.assertGreater(n_checked, 90, 'the sweep found suspiciously few contractions')


class TestDifferentialVsNamed(unittest.TestCase):
    """contract(transliterated name) == the hand-written named contraction, over the full
    block-shape matrix. The flatten-based bodies are the independent reference implementation."""

    def test_all_named_contractions_all_shapes(self):
        rng = np.random.default_rng(0)
        for name, fn in named_contractions():
            subs = to_subscripts(name)
            params = inspect.signature(fn).parameters
            for W, K, C in SHAPE_MATRIX:
                with self.subTest(name=name, W=W, K=K, C=C):
                    ops = build_operands(fn, W, K, C, rng)
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


class TestLoopOracle(unittest.TestCase):
    """contract(...) == the definition, executed directly (map the single-axis contraction over
    every group index, summing groups absent from the output)."""

    ORACLE_MATRIX = [
        ((), (), ()),
        ((2,), (2,), (2,)),
        ((2, 3), (2, 2), (2, 2)),
    ]

    def test_all_named_subscripts_vs_loop_oracle(self):
        rng = np.random.default_rng(0)
        for name, fn in named_contractions():
            subs = to_subscripts(name)
            for W, K, C in self.ORACLE_MATRIX:
                with self.subTest(name=name, W=W, K=K, C=C):
                    ops = build_operands(fn, W, K, C, rng)
                    lens = {}
                    if 'n_probe' in inspect.signature(fn).parameters:
                        lens['len_W'] = len(W)
                    if 'n_frame' in inspect.signature(fn).parameters:
                        lens['len_C'] = len(C)
                    got = contract(subs, *ops, **lens)
                    ref = loop_oracle(subs, ops, {'W': W, 'K': K, 'C': C})
                    self.assertEqual(got.shape, ref.shape)
                    self.assertTrue(np.allclose(got, ref, rtol=1e-10, atol=1e-12))

    def test_generic_strings_beyond_the_library(self):
        # strings with no named-contraction counterpart: sum over a lone group; two riders; a
        # group appearing on every operand; whitespace tolerated.
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
                terms = s.split('->')[0].split(',')
                ops = []
                for t in terms:
                    shape = ()
                    for ch in t:
                        shape += shapes[ch] if ch.isupper() else (AXIS_SIZE[ch],)
                    ops.append(rng.standard_normal(shape))
                got = contract(subs, *ops, **lens)
                ref = loop_oracle(subs, ops, shapes)
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


if __name__ == '__main__':
    unittest.main()
