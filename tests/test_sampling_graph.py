# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_variables``: the declared graph, the closure, and what the engine checks. Structural.

1. **The graph as data** (`TestGraph`): the docstring's table equals the graph row for row, and the shape table
   equals ``_SHAPES``; every step name is a function of ``sampling_steps``; the three namespaces of ``reads`` are
   disjoint and cover every read; the slots of ``HeldVariables`` are exactly the computed variables some other row
   reads; the table of bond offsets; the row type's ``carries_K``.
2. **The closure** (`TestClosure`): the earlier applicable row of a name wins; a held variable does not change the
   route; asking for one variable runs only the steps the graph says it reads, counted per step with a hook on the
   private step module (the counts derived here by hand from the graph).
3. **The sweeps' ends** (`TestEnds`): the start value of a sweep is sized by the end bond of its own store, on a
   frame whose ``L`` and ``R`` end bonds differ.
4. **What the engine checks** (`TestChecks`): an unknown kind or name; an entries call with jets; a jet shape that
   is not a tuple; every missing argument; a held variable of a wrong form or shape; a frame of neither form.

numpy only.
"""
import os
import re
import sys
import unittest
import unittest.mock

import numpy as np

import t3toolbox.backend.sampling_steps as sampling_steps
import t3toolbox.backend.sampling_variables as sv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampling_definitions as defs
import test_sampling_variables as tsv


def doc_table(header):
    """The rows of the module docstring's table whose header line starts with ``header``, each a list of cells."""
    lines = sv.__doc__.splitlines()
    start = next(n for n, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith('|'):
            break
        rows.append([cell.strip() for cell in line.strip().strip('|').split('|')])
    return rows


def names_of(cell):
    """The names in a cell of backquoted names: '`D`, `w`' -> ('D', 'w')."""
    return tuple(re.findall(r'`([^`]+)`', cell))


class TestGraph(unittest.TestCase):

    def test_the_docstring_table_is_the_graph_row_for_row(self):
        rows = doc_table('| Variable | Driver | Step | Reads | Needs | K |')
        self.assertEqual(len(rows), len(sv._PROBE_GRAPH))
        for row, rule in zip(rows, sv._PROBE_GRAPH):
            with self.subTest(row=' | '.join(row)):
                name, driver, step, reads, needs, K = row
                self.assertEqual(names_of(name), (rule.name,))
                self.assertEqual((driver, names_of(step)), (rule.step[0], (rule.step[1],)))
                self.assertEqual(names_of(reads), rule.reads)
                self.assertEqual(frozenset(names_of(needs)), rule.needs)
                self.assertEqual(K, 'yes' if rule.carries_K else 'no')

    def test_the_docstring_shape_table_is_the_shape_table(self):
        rows = doc_table('| Variable | Index | Jet sizes | Size |')
        self.assertEqual([names_of(row[0])[0] for row in rows], list(sv._SHAPES))
        for row in rows:
            name = names_of(row[0])[0]
            index, jets, size = sv._SHAPES[name]
            with self.subTest(name=name):
                self.assertEqual(row[1], index)
                self.assertEqual(row[2], 'min(J_in_i, F)' if jets == 'leaf' else 'F')
                self.assertEqual(row[3], size + '_i')

    def test_every_step_name_is_a_function_of_the_step_module(self):
        for rule in sv._PROBE_GRAPH:
            driver, step_name = rule.step
            with self.subTest(name=rule.name, step=step_name):
                self.assertIn(driver, sv._DRIVERS)
                self.assertTrue(callable(getattr(sampling_steps, step_name, None)))
                self.assertEqual(getattr(sampling_steps, step_name).__module__, sampling_steps.__name__)
        self.assertEqual(sampling_steps.__all__, [])

    def test_the_three_namespaces_of_reads_are_disjoint_and_cover_every_read(self):
        made = set(sv._made(sv._PROBE_GRAPH))
        self.assertFalse(made & sv._STORES)
        self.assertFalse(made & sv._ARGUMENTS)
        self.assertFalse(sv._STORES & sv._ARGUMENTS)
        for rule in sv._PROBE_GRAPH:
            with self.subTest(name=rule.name, reads=rule.reads):
                self.assertTrue(set(rule.reads) <= made | sv._STORES | sv._ARGUMENTS)
                self.assertTrue(rule.needs <= sv._ARGUMENTS | {'variations', 'core_directions', '_manifold',
                                                               '_corewise'})

    def test_the_held_slots_are_the_computed_variables_another_row_reads(self):
        made = set(sv._made(sv._PROBE_GRAPH))
        read_by_another = {name for rule in sv._PROBE_GRAPH for name in rule.reads if name != rule.name} & made
        self.assertEqual(set(sv.HeldVariables.slots()), read_by_another)
        self.assertNotIn('w', sv.HeldVariables.slots())                     # an argument, not a slot

    def test_the_bond_offsets_and_the_shape_table(self):
        """A site op reads the left of bond ``i`` (``mu``) and the right of bond ``i + 1`` (``nu``); every bond
        variable is made by a sweep, and every sweep makes a bond variable whose letter has an offset."""
        self.assertEqual(sv._BOND_OFFSET, {'mu': 0, 'nu': 1})
        for rule in sv._PROBE_GRAPH:
            with self.subTest(name=rule.name):
                is_sweep = rule.step[0] in ('left', 'right')
                if rule.name in sv._SHAPES:
                    self.assertEqual(sv._SHAPES[rule.name][0] == 'bond', is_sweep)
                if is_sweep:
                    self.assertIn(rule.name.split('_')[0], sv._BOND_OFFSET)
                    self.assertIn(rule.reads[0], sv._STORES)                # the store it sweeps through
                    self.assertEqual(rule.step[0], 'left' if rule.name.startswith('mu') else 'right')
        problem = defs.small()
        mu, nu, eta = sv.sampling_variables('probe', problem.frame, problem.ww, ('mu', 'nu', 'eta'))
        U = problem.frame[1]
        for i in range(problem.d):                                         # eta[i] from mu[i] and nu[i+1]
            expected = np.einsum('Wa,aib,Wb->Wi', mu[i], U[i], nu[i + 1])
            self.assertLessEqual(defs.relative_error(eta[i], expected), 1e-12)

    def test_carries_K_is_a_field_of_every_row(self):
        self.assertIn('carries_K', sv.Rule._fields)
        self.assertEqual({rule.carries_K for rule in sv._PROBE_GRAPH}, {False})     # no direction yet


class CountingSteps:
    """Every step of ``sampling_steps`` wrapped by a mock that counts its calls."""

    def __enter__(self):
        self.patches = {}
        for rule in sv._PROBE_GRAPH:
            name = rule.step[1]
            if name not in self.patches:
                patch = unittest.mock.patch.object(sampling_steps, name, wraps=getattr(sampling_steps, name))
                self.patches[name] = (patch, patch.start())
        return self

    def counts(self):
        return {name: mock.call_count for name, (_, mock) in self.patches.items() if mock.call_count}

    def __exit__(self, *args):
        for patch, _ in self.patches.values():
            patch.stop()
        return False


class TestClosure(unittest.TestCase):

    def setUp(self):
        self.problem = defs.small()
        self.d = self.problem.d
        self.y, self.weight = self.problem.data('probe', defs.LINE_1)
        self.sample = self.problem.input_jets(defs.LINE_1)
        self.frame, self.Vc = tsv.consistent_frame(seed=3)
        rng = np.random.default_rng(4)
        self.ww = [rng.standard_normal((2, D.shape[-1])) for D in self.frame[0]]

    def test_the_earlier_applicable_row_wins(self):
        """For every name with two rows or more, a call whose facts satisfy every row takes the first."""
        rows = {}
        for rule in sv._PROBE_GRAPH:
            rows.setdefault(rule.name, []).append(rule)
        several = {name: rules for name, rules in rows.items() if len(rules) > 1}
        self.assertEqual(set(several), {'z'})
        for name, rules in several.items():
            every_fact = frozenset().union(*(rule.needs for rule in rules)) | {'_manifold'}
            chosen = [rule for rule in sv._closure('probe', (name,), frozenset(), every_fact) if rule.name == name]
            self.assertEqual(chosen, [rules[0]])
            fewer = frozenset({'_manifold'})
            chosen = [rule for rule in sv._closure('probe', (name,), frozenset(), fewer) if rule.name == name]
            self.assertEqual(chosen, [rules[-1]])

    def test_a_held_variable_does_not_change_the_route(self):
        """With ``Vc`` given and ``eta_D`` held, ``z`` still takes the route of ``Vc``, and ``eta`` is computed."""
        held = frozenset({'eta_D'})
        rules = sv._closure('probe', ('z',), held, frozenset({'Vc', '_manifold'}))
        self.assertEqual([rule.name for rule in rules], ['xi', 'mu', 'nu', 'eta', 'z'])
        self.assertEqual(rules[-1].reads, ('eta', 'Vc'))
        (eta_D,) = sv.sampling_variables('probe', self.frame, self.ww, ('eta_D',))
        with CountingSteps() as steps:
            sv.sampling_variables('probe', self.frame, self.ww, ('z',), Vc=self.Vc,
                                  held=sv.HeldVariables(eta_D=eta_D))
        d = len(self.frame[0])
        self.assertEqual(steps.counts(), {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d, 'z_step': d})

    def test_one_variable_runs_only_the_steps_it_reads(self):
        """The step counts, derived by hand from the graph: a site step runs ``d`` times, a sweep ``d`` times (its
        start value is not a step), a whole step once. ``nu_step`` makes both ``nu`` and ``nu_L``."""
        d = self.d
        expected = {
            ('xi',):            {'xi_step': d},
            ('mu',):            {'xi_step': d, 'mu_step': d},
            ('nu',):            {'xi_step': d, 'nu_step': d},
            ('nu_L',):          {'xi_step': d, 'nu_step': d},
            ('nu', 'nu_L'):     {'xi_step': d, 'nu_step': 2 * d},
            ('eta',):           {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d},
            ('eta_D',):         {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d},
            ('eta', 'eta_D'):   {'xi_step': d, 'mu_step': d, 'nu_step': 2 * d, 'eta_step': 2 * d},
            ('z',):             {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d, 'z_step': d},
            ('r',):             {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d, 'z_step': d, 'r_step': 1},
            ('f',):             {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d, 'z_step': d, 'r_step': 1,
                                 'f_step': 1},
            ('r_block_sumsq', 'f'): {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d, 'z_step': d,
                                     'r_step': 1, 'f_step': 1, 'r_block_sumsq_step': 1},
            ('w',):             {},
            ('z', 'z', 'mu'):   {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d, 'z_step': d},
        }
        for want, counts in expected.items():
            with self.subTest(want=want):
                with CountingSteps() as steps:
                    sv.sampling_variables('probe', self.problem.frame, self.sample, want, order=(1,), y=self.y,
                                          weight=self.weight)
                self.assertEqual(steps.counts(), counts)
        d = len(self.frame[0])                          # with Vc: z through eta, no sweep of nu_L
        with CountingSteps() as steps:
            sv.sampling_variables('probe', self.frame, self.ww, ('z',), Vc=self.Vc)
        self.assertEqual(steps.counts(), {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d, 'z_step': d})
        with CountingSteps() as steps:                  # held mu: no mu sweep; xi still read by nu
            (mu,) = sv.sampling_variables('probe', self.frame, self.ww, ('mu',))
            sv.sampling_variables('probe', self.frame, self.ww, ('eta',), held=sv.HeldVariables(mu=mu))
        self.assertEqual(steps.counts(), {'xi_step': 2 * d, 'mu_step': d, 'nu_step': d, 'eta_step': d})


class TestEnds(unittest.TestCase):

    def test_a_sweep_starts_from_the_end_bond_of_its_own_store(self):
        """On a frame whose ``L`` and ``R`` end bonds differ at both ends: ``mu[0]`` has the ``L`` store's bond 0,
        ``nu[d]`` the ``R`` store's bond ``d`` and ``nu_L[d]`` the ``L`` store's; each start is ones in slot
        ``(0, ..., 0)`` and zero in every other slot."""
        problem = tsv.tails_one_axis()
        D, U, L, R = problem.frame
        self.assertNotEqual(L[-1].shape[-1], R[-1].shape[-1])
        self.assertNotEqual(L[0].shape[0], R[0].shape[0])
        for case in (defs.NO_JETS, defs.PLANE):
            mu, nu, nu_L = sv.sampling_variables('probe', problem.frame, problem.input_jets(case),
                                                 ('mu', 'nu', 'nu_L'), order=case.order)
            with self.subTest(jets=case.label):
                for start, size in ((mu[0], L[0].shape[0]), (nu[-1], R[-1].shape[-1]), (nu_L[-1], L[-1].shape[-1])):
                    expected = np.zeros(case.box + problem.W + (size,))
                    expected[(0,) * case.m] = 1.0
                    self.assertTrue(np.array_equal(start, expected))


class TestChecks(unittest.TestCase):

    def setUp(self):
        self.problem = defs.small()
        self.frame, self.ww = self.problem.frame, self.problem.ww

    def test_an_unknown_kind_or_name_raises(self):
        with self.assertRaisesRegex(ValueError, 'unknown sampling kind'):
            sv.sampling_variables('probes', self.frame, self.ww, ('z',))
        with self.assertRaisesRegex(ValueError, "unknown variable 'g', 'zz' for the probe kind"):
            sv.sampling_variables('probe', self.frame, self.ww, ('z', 'g', 'zz'))
        with self.assertRaisesRegex(TypeError, 'want is a sequence of names'):
            sv.sampling_variables('probe', self.frame, self.ww, 'z')
        for kind in ('apply', 'entries'):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(NotImplementedError, 'not built yet'):
                    sv.sampling_variables(kind, self.frame, self.ww, ('z',))

    def test_the_jet_shape(self):
        with self.assertRaisesRegex(ValueError, 'the entries kind has no jets'):
            sv.sampling_variables('entries', self.frame, self.problem.index, ('z',), order=(1,))
        for bad in (1, [1], None):
            with self.subTest(order=repr(bad)):
                with self.assertRaisesRegex(TypeError, 'order is the jet shape'):
                    sv.sampling_variables('probe', self.frame, self.ww, ('z',), order=bad)

    def test_every_missing_argument_raises(self):
        for want, pattern in ((('r',), "'r' needs the argument y, which was not given$"),
                              (('f',), r"'r' needs the argument y, which was not given \(needed through f -> r\)"),
                              (('r_block_sumsq',), r'needed through r_block_sumsq -> r')):
            with self.subTest(want=want):
                with self.assertRaisesRegex(ValueError, pattern):
                    sv.sampling_variables('probe', self.frame, self.ww, want)
        (r,) = sv.sampling_variables('probe', self.frame, self.ww, ('r',), y=self.problem.data('probe', defs.NO_JETS)[0])
        sv.sampling_variables('probe', self.frame, self.ww, ('f',), held=sv.HeldVariables(r=r))     # a held r will do

    def test_a_held_variable_of_a_wrong_form_raises(self):
        mu, z = sv.sampling_variables('probe', self.frame, self.ww, ('mu', 'z'))
        bad = {
            'one array too few':   (sv.HeldVariables(mu=mu[:-1]), r"the held 'mu' must be a tuple of 4 arrays; got a "
                                                                   r"tuple of length 3"),
            'one array, not a tuple': (sv.HeldVariables(mu=mu[0]), r"the held 'mu' must be a tuple of 4 arrays; got "
                                                                   r"ndarray"),
            'a wrong bond size':   (sv.HeldVariables(mu=mu[:1] + (np.zeros((3, 5)),) + mu[2:]),
                                    r'the held mu\[1\] has shape \(3, 5\); expected \(3, 2\)'),
            'a jet axis too many': (sv.HeldVariables(z=tuple(z_i[None] for z_i in z)),
                                    r'the held z\[0\] has shape \(1, 3, 4\); expected \(3, 4\)'),
            'nu where nu_L goes':  (sv.HeldVariables(nu_L=sv.sampling_variables('probe', self.frame, self.ww, ('nu',))[0]),
                                    r'the held nu_L\[1\] has shape \(3, 3\); expected \(3, 2\)'),
        }
        for label, (held, pattern) in bad.items():
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, pattern):
                    sv.sampling_variables('probe', self.frame, self.ww, ('w',), held=held)     # checked though unread
        # every filled slot is checked, not only the first, and a slot that nothing wanted reads is checked too: a
        # good mu and a wrong nu, wanting eta_D (which reads mu and nu_L, not nu)
        bad_nu = sv.sampling_variables('probe', self.frame, self.ww, ('nu_L',))[0]       # rL sizes where rR go
        with self.assertRaisesRegex(ValueError, r'the held nu\[1\] has shape \(3, 2\); expected \(3, 3\)'):
            sv.sampling_variables('probe', self.frame, self.ww, ('eta_D',), held=sv.HeldVariables(mu=mu, nu=bad_nu))
        with self.assertRaisesRegex(TypeError, 'held is a HeldVariables or None'):
            sv.sampling_variables('probe', self.frame, self.ww, ('z',), held={'mu': mu})

    def test_a_frame_of_neither_form(self):
        with self.assertRaisesRegex(ValueError, 'a ragged frame has four stores'):
            sv.sampling_variables('probe', self.frame[:3], self.ww, ('z',))
        with self.assertRaisesRegex(NotImplementedError, 'a corewise point'):
            sv.sampling_variables('probe', (self.frame[0], self.frame[2]), self.ww, ('z',))
        with self.assertRaisesRegex(NotImplementedError, 'the uniform layout'):
            sv.sampling_variables('probe', (np.zeros((3, 2, 2)),) * 6, self.ww, ('z',))


if __name__ == '__main__':
    unittest.main()
