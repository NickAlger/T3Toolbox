# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_variables``: the declared graph, the closure, and what the engine checks. Structural.

1. **The graph as data** (`TestGraph`): the docstring's table equals the graph row for row (the probe kind's, and the
   apply kind's own rows), and the shape table equals ``_SHAPES``; every graph is made of the shared blocks and its
   kind's rows, and depends only on what it makes, the stores and the arguments; one table of the paths of ``JtJp``,
   the twins of a start node; the starts and the routes of the apply kind; the entries kind derived from apply's rows
   by its table of steps; every step name is a function of ``sampling_steps``; the three namespaces of ``reads`` are
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
        rows = doc_table('| Variable | Driver | Step | Reads | Needs | K | Start |')
        self.assertEqual(len(rows), len(sv._PROBE_GRAPH))
        for row, rule in zip(rows, sv._PROBE_GRAPH):
            with self.subTest(row=' | '.join(row)):
                name, driver, step, reads, needs, K, start = row
                self.assertEqual(start or None, rule.start)
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
                self.assertEqual(row[2], {'leaf': 'min(J_in_i, F)', 'F': 'F', 'none': 'none'}[jets])
                self.assertEqual(row[3], {'V': '(nU_i, N_i)', 'H': '(rL_i, nD_i, rR_(i+1))', 'VD': '(nD_i, N_i)',
                                          'HL': '(rL_i, nD_i, rL_(i+1))'}.get(size, size + '_i'))

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

    def test_the_held_slots_are_the_public_computed_variables(self):
        """Every public computed variable of the graphs (row 6.4: every variable can be wanted and held); not the
        sample ``w``, an argument, and not a private node."""
        public = [name for name in sv._made(sv._PROBE_GRAPH) if not name.startswith('_')]
        public += [name for name in sv._made(sv._PROBE_COREWISE_GRAPH) if not name.startswith('_') and name not in public]
        self.assertEqual(sv.HeldVariables.slots(), tuple(public))
        self.assertEqual(public[:len(sv._public(sv._PROBE_GRAPH))], list(sv._public(sv._PROBE_GRAPH)))
        self.assertEqual(public[len(sv._public(sv._PROBE_GRAPH)):], ['nu_dot', 'eta_dot', 'nu_tilde_dot',
                                                                    'eta_tilde_dot'])   # a corewise point's own
        self.assertNotIn('w', sv.HeldVariables.slots())                     # an argument, not a slot
        self.assertNotIn('_eta_tilde_top', sv.HeldVariables.slots())        # a private node
        self.assertIn('_eta_tilde_top', sv._made(sv._PROBE_GRAPH))

    def test_every_sweep_declares_its_start(self):
        """Each row declares its start (not chosen by grade): the plain sweeps start at ones, the probe kind's tilde
        sweeps at zeros, and a row of another driver declares none."""
        expected = {'mu': 'ones', 'nu': 'ones', 'nu_L': 'ones', 'mu_tilde': 'zeros', 'nu_tilde': 'zeros',
                    'mu_R_dot': 'zeros', '_frame.nu_L_dot': 'zeros', '_Jp.mu_tilde': 'zeros', '_Jp.nu_tilde': 'zeros',
                    'nu_L_tilde': 'zeros', 'mu_dot': 'zeros', '_raw.nu_L_dot': 'zeros', 'mu_tilde_dot': 'zeros',
                    'nu_L_tilde_dot': 'zeros'}
        for rule in sv._PROBE_GRAPH:
            with self.subTest(name=rule.name):
                self.assertEqual(rule.start, expected.get(rule.name))
                if rule.start is not None:
                    self.assertIn(rule.start, sv._STARTS)
        problem = defs.small()
        mu_tilde, nu_tilde = sv.sampling_variables('probe', problem.frame, problem.input_jets(defs.LINE_1),
                                                   ('mu_tilde', 'nu_tilde'), order=(1,),
                                                   y=problem.data('probe', defs.LINE_1)[0])
        self.assertFalse(mu_tilde[0].any() or nu_tilde[-1].any())          # the starts: zero
        self.assertTrue(mu_tilde[1].any() and nu_tilde[-2].any())

    def test_the_bond_offsets_and_the_shape_table(self):
        """A site op reads the left of bond ``i`` (``mu``) and the right of bond ``i + 1`` (``nu``); every bond
        variable is made by a sweep (or is a route over one), and every sweep makes a bond variable whose letter, read
        after a private prefix by ``_base``, has an offset."""
        self.assertEqual(sv._BOND_OFFSET, {'mu': 0, 'nu': 1})
        self.assertEqual([sv._base(n) for n in ('_Jp.mu_tilde', '_frame.nu_L_dot', '_eta_tilde_top', 'mu_R_dot')],
                         ['mu_tilde', 'nu_L_dot', 'eta_tilde_top', 'mu_R_dot'])
        self.assertEqual([sv._letter(n) for n in ('_Jp.mu_tilde', '_frame.nu_L_dot', '_Jp.xi_tilde', 'eta_D_dot')],
                         ['mu', 'nu', 'xi', 'eta'])
        rows = {rule.name: rule for rule in sv._PROBE_GRAPH}
        for rule in sv._PROBE_GRAPH:
            with self.subTest(name=rule.name):
                is_sweep = rule.step[0] in ('left', 'right')
                is_bond = sv._SHAPES[sv._table_name(rule.name)][0] == 'bond' if rule.name not in (
                    'f', 'r_block_sumsq', 'g', 'Jp_norm_sq', 'JtJp', 'Kp') else False
                if rule.step == ('whole', 'identity_step'):                 # a route over a private node
                    self.assertEqual(is_bond, rows[rule.reads[0]].step[0] in ('left', 'right'))
                else:
                    self.assertEqual(is_bond, is_sweep)
                if is_sweep:
                    self.assertIn(sv._letter(rule.name), sv._BOND_OFFSET)
                    self.assertIn(rule.reads[0], sv._STORES)                # the store it sweeps through
                    self.assertEqual(rule.step[0], 'left' if sv._letter(rule.name) == 'mu' else 'right')
        problem = defs.small()
        mu, nu, eta = sv.sampling_variables('probe', problem.frame, problem.ww, ('mu', 'nu', 'eta'))
        U = problem.frame[1]
        for i in range(problem.d):                                         # eta[i] from mu[i] and nu[i+1]
            expected = np.einsum('Wa,aib,Wb->Wi', mu[i], U[i], nu[i + 1])
            self.assertLessEqual(defs.relative_error(eta[i], expected), 1e-12)

    def test_the_direction_operands_and_their_masks(self):
        """P11's attach point: every core-shaped argument a row reads (the two forms of a direction and the Tucker
        centers) is in ``_DIRECTION_MASKS``, mapped to the frame's variation masks or to a store's own masks."""
        read = {name for graph in sv._GRAPHS.values() for rule in graph for name in sv._deps(rule)}
        self.assertEqual(set(sv._DIRECTION_MASKS), read & sv._ARGUMENTS - {'w', 'index', 'y', 'weight'})
        self.assertEqual(sv._DIRECTION_MASKS, {'V': 'variations', 'H': 'variations', 'Vc': 'variations',
                                               'D_dot': 'D', 'L_dot': 'L'})

    def test_carries_K_is_a_field_of_every_row(self):
        self.assertIn('carries_K', sv.Rule._fields)
        with_K = {rule.name for rule in sv._PROBE_GRAPH if rule.carries_K}
        self.assertEqual(with_K, {'xi_U_dot', 'mu_R_dot', '_frame.nu_L_dot', 'nu_L_dot', '_frame.eta_D_dot',
                                  'eta_D_dot', 'Jp', 'Jp_norm_sq', 'JtJp', 'xi_dot', 'mu_dot', '_raw.nu_L_dot',
                                  '_raw.eta_D_dot', 'xi_tilde_dot', 'mu_tilde_dot', 'nu_L_tilde_dot', 'eta_D_tilde_dot',
                                  '_eta_D_tilde_dot_top', 'V_tilde_dot', 'H_tilde_dot', 'Kp'}
                         | {sv._twin(n) for n in sv._JP_PATHS['probe', '_manifold']})

    def test_every_row_on_the_path_of_JtJp_has_its_twin(self):
        """The adjoint seeded by ``Jp``: every row on the residual's path from ``w_tilde`` to ``V_tilde``, ``H_tilde``
        and the short node has a twin named ``'_Jp.' + name.lstrip('_')``, with the same step and start, ``carries_K``,
        each read on the path renamed and ``r`` read as ``Jp``; ``JtJp`` is the twin of ``g``. The full ``eta_tilde``
        and ``g`` have no ``'_Jp.'`` twin."""
        rows = {rule.name: rule for rule in sv._PROBE_GRAPH}
        path = sv._JP_PATHS['probe', '_manifold']
        self.assertEqual(path, ('w_tilde', 'xi_tilde', 'mu_tilde', 'nu_tilde', '_eta_tilde_top', 'V_tilde', 'H_tilde'))
        rename = lambda read: sv._twin(read) if read in path else ('Jp' if read == 'r' else read)
        for name in path + ('g',):
            base = rows[name]
            twin = rows['JtJp' if name == 'g' else sv._twin(name)]
            with self.subTest(name=name):
                self.assertEqual(twin.name, 'JtJp' if name == 'g' else '_Jp.' + name.lstrip('_'))
                # JtJp alone also needs the direction itself, so that a call without it names JtJp (P7's review)
                needs = base.needs | {'variations'} if name == 'g' else base.needs
                self.assertEqual((twin.step, twin.start, twin.needs, twin.carries_K),
                                 (base.step, base.start, needs, True))
                self.assertEqual(twin.reads, tuple(map(rename, base.reads)))
                self.assertFalse(base.carries_K)
        self.assertNotIn('_Jp.eta_tilde', rows)
        self.assertNotIn('_Jp.g', rows)
        twins = [name for name in rows if name.startswith('_Jp.')]
        self.assertEqual(len(twins), len(path))

    def test_the_apply_docstring_table_is_its_own_rows(self):
        """The apply kind's table lists the rows of its manifold graph that are not in the shared blocks, row for
        row, its twins included; a start node is written by its name."""
        shared = set(sv._PLAIN_ROWS + sv._RESIDUAL_ROWS + sv._DOT_ROWS + sv._DOT_SUM_ROWS)
        own = [rule for rule in sv._GRAPHS['apply', '_manifold'] if rule not in shared]
        rows = doc_table("| Apply's own row | Driver | Step | Reads | Needs | K | Start |")
        self.assertEqual(len(rows), len(own))
        for row, rule in zip(rows, own):
            with self.subTest(row=' | '.join(row)):
                name, driver, step, reads, needs, K, start = row
                self.assertEqual((names_of(name), driver, names_of(step)),
                                 ((rule.name,), rule.step[0], (rule.step[1],)))
                self.assertEqual((names_of(reads), frozenset(names_of(needs))), (rule.reads, rule.needs))
                self.assertEqual((K, start or None), ('yes' if rule.carries_K else 'no', rule.start))

    def test_every_graph_is_made_of_the_blocks(self):
        """The probe and apply rows are the shared blocks and each kind's own, in order; every graph's reads and
        start nodes are made by it, a store or an argument; every step is a function of the step module; every
        sweep's start is a named value or a node the graph makes."""
        self.assertEqual(sv._PROBE_ROWS, sv._PLAIN_ROWS + sv._PROBE_Z_ROWS + sv._RESIDUAL_ROWS + sv._PROBE_ADJOINT_ROWS
                         + sv._DOT_ROWS + sv._PROBE_JP_ROWS + sv._DOT_SUM_ROWS + sv._PROBE_CURVATURE_ROWS)
        self.assertEqual(sv._APPLY_ROWS, sv._PLAIN_ROWS + sv._APPLY_Z_ROWS + sv._RESIDUAL_ROWS + sv._APPLY_ADJOINT_ROWS
                         + sv._DOT_ROWS + sv._APPLY_JP_ROWS + sv._DOT_SUM_ROWS + sv._APPLY_CURVATURE_ROWS)
        self.assertEqual(sv._ENTRIES_ROWS, sv._ENTRIES_OWN_ROWS + sv._entries(sv._APPLY_ROWS))
        self.assertEqual([rule.name for rule in sv._ENTRIES_OWN_ROWS], ['w'])
        self.assertEqual(set(sv._GRAPHS), {(kind, call) for kind in ('probe', 'apply', 'entries')
                                           for call in ('_manifold', '_corewise')})
        for key, graph in sv._GRAPHS.items():
            made = set(sv._made(graph))
            stores = sv._STORES if key[1] == '_manifold' else {'D', 'L'}
            for rule in graph:
                with self.subTest(graph=repr(key), name=rule.name):
                    self.assertTrue(set(sv._deps(rule)) <= made | stores | sv._ARGUMENTS)
                    self.assertTrue(callable(getattr(sampling_steps, rule.step[1], None)))
                    if rule.step[0] in ('left', 'right'):
                        self.assertTrue(rule.start in sv._STARTS or rule.start in made)
                    else:
                        self.assertIsNone(rule.start)

    def test_one_table_of_the_paths_of_JtJp(self):
        """``_JP_PATHS`` is keyed like ``_GRAPHS``; each path's names are made by its graph, and each graph has one
        twin per name of its path and ``JtJp``; the twin of a sweep that starts from a node on the path starts from
        that node's twin."""
        self.assertEqual(set(sv._JP_PATHS), set(sv._GRAPHS))
        for key, graph in sv._GRAPHS.items():
            path = sv._JP_PATHS[key]
            rows = {}
            for rule in graph:
                rows.setdefault(rule.name, rule)
            with self.subTest(graph=repr(key)):
                self.assertTrue(set(path) <= set(rows))
                self.assertEqual({name for name in rows if name.startswith('_Jp.')}, {sv._twin(n) for n in path})
                self.assertIn('JtJp', rows)
                for name in path:
                    start = rows[name].start
                    if start is not None and start not in sv._STARTS:
                        self.assertEqual(rows[sv._twin(name)].start, sv._twin(start))
        with self.assertRaisesRegex(RuntimeError, 'the path of the adjoint seeded by Jp of the apply kind'):
            sv._manifold(tuple(rule for rule in sv._APPLY_ROWS if rule.name != 'H_tilde'), 'apply')

    def test_the_starts_of_the_apply_kind(self):
        """The apply kind's right tilde sweeps start from the seed, a node; its tilde-dot sweep from zeros."""
        starts = {rule.name: rule.start for rule in sv._GRAPHS['apply', '_manifold'] if rule.start is not None}
        self.assertEqual(starts, {'mu': 'ones', 'nu': 'ones', 'nu_L': 'ones', 'nu_tilde': 'w_tilde',
                                  'nu_L_tilde': 'w_tilde', 'mu_R_dot': 'zeros', '_frame.nu_L_dot': 'zeros',
                                  'mu_dot': 'zeros', '_raw.nu_L_dot': 'zeros', 'nu_L_tilde_dot': 'zeros',
                                  '_Jp.nu_tilde': '_Jp.w_tilde'})

    def test_the_earlier_applicable_row_wins_on_apply(self):
        rows = {}
        for rule in sv._GRAPHS['apply', '_manifold']:
            if '_corewise' not in rule.needs:
                rows.setdefault(rule.name, []).append(rule)
        several = {name: rules for name, rules in rows.items() if len(rules) > 1}
        self.assertEqual(set(several), {'nu_L_dot', 'eta_D_dot', 'Jp'})
        for name, rules in several.items():
            with self.subTest(name=name):
                every_fact = frozenset().union(*(rule.needs for rule in rules)) | {'_manifold'}
                chosen = [rule for rule in sv._closure('apply', (name,), frozenset(), every_fact) if rule.name == name]
                self.assertEqual(chosen, [rules[0]])
        self.assertEqual([rule.reads for rule in rows['Jp']], [('mu_R_dot',), ('mu_dot',)])

    def test_a_zero_name_is_not_made(self):
        for kind in ('apply', 'entries'):
            for call in ('_manifold', '_corewise'):
                self.assertFalse(sv._ZERO_NAMES[kind] & set(sv._made(sv._GRAPHS[kind, call])))

    def test_the_entries_kind_is_derived_by_the_table_of_steps(self):
        """The docstring's table is ``_ENTRIES_STEPS``; every entries row is the apply row with its step from the
        table, ``w`` read as ``index`` (and ``D`` added for the scatter's mode size), a short node read as its full
        node; the short nodes are gone; the derivation raises on a row that reads the input with no step in the
        table."""
        rows = doc_table('| A step of apply | Its step for entries | What it does |')
        self.assertEqual({names_of(a)[0]: names_of(b)[0] for a, b, _ in rows}, sv._ENTRIES_STEPS)
        apply_rows = [rule for rule in sv._APPLY_ROWS if rule.name not in sv._ENTRIES_FULL]
        derived = sv._entries(sv._APPLY_ROWS)
        self.assertEqual(len(apply_rows), len(derived))
        for a, e in zip(apply_rows, derived):
            with self.subTest(name=a.name, step=a.step[1]):
                reads = tuple('index' if r == 'w' else sv._ENTRIES_FULL.get(r, r) for r in a.reads)
                if 'w' in a.reads:
                    self.assertEqual(e.step, (a.step[0], sv._ENTRIES_STEPS[a.step[1]]))
                    reads += ('D',) if e.step[1] == 'V_scatter_step' else ()
                else:
                    self.assertEqual(e.step, a.step)
                self.assertEqual((e.name, e.reads, e.needs, e.carries_K, e.start),
                                 (a.name, reads, a.needs, a.carries_K, a.start))
        for graph in (sv._GRAPHS['entries', '_manifold'], sv._GRAPHS['entries', '_corewise']):
            self.assertFalse({dep for rule in graph for dep in sv._deps(rule)} & ({'w'} | set(sv._ENTRIES_FULL)))
            self.assertFalse({rule.name for rule in graph} & set(sv._ENTRIES_FULL))
        unknown = sv._APPLY_ROWS + (sv.Rule('_leaf2', ('site', 'z_step'), ('D', 'w'), sv._NONE, False),)
        with self.assertRaisesRegex(RuntimeError, "the entries kind has no step for 'z_step'"):
            sv._entries(unknown)


class CountingSteps:
    """Every step of ``sampling_steps`` wrapped by a mock that counts its calls."""

    def __enter__(self):
        self.patches = {}
        for rule in (rule for graph in sv._GRAPHS.values() for rule in graph):
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
        """For every name with two rows or more on a manifold call, a call whose facts satisfy every row takes the
        first; a call with only the last row's facts takes the last. (The routes needing ``_corewise`` are never taken
        on a manifold call.)"""
        rows = {}
        for rule in sv._PROBE_GRAPH:
            if '_corewise' not in rule.needs:
                rows.setdefault(rule.name, []).append(rule)
        several = {name: rules for name, rules in rows.items() if len(rules) > 1}
        self.assertEqual(set(several), {'z', 'nu_L_dot', 'eta_D_dot', 'Jp'})
        for name, rules in several.items():
            with self.subTest(name=name):
                every_fact = frozenset().union(*(rule.needs for rule in rules)) | {'_manifold'}
                chosen = [rule for rule in sv._closure('probe', (name,), frozenset(), every_fact) if rule.name == name]
                self.assertEqual(chosen, [rules[0]])
                fewer = rules[-1].needs | {'_manifold'}
                chosen = [rule for rule in sv._closure('probe', (name,), frozenset(), fewer) if rule.name == name]
                self.assertEqual(chosen, [rules[-1]])
        self.assertEqual({rule.name for rule in sv._PROBE_GRAPH if '_corewise' in rule.needs}, {'V_tilde', 'H_tilde'})

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
        # the tilde grade: the seed reads r (through z, by D: nu_L and eta_D); nu (through R) is swept for nu_tilde
        seed = {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d, 'z_step': d, 'r_step': 1, 'w_tilde_step': 1}
        plus = lambda base, **more: dict(base, **{k: base.get(k, 0) + v for k, v in more.items()})
        sweeps = plus(seed, xi_tilde_step=d, mu_tilde_step=d, nu_tilde_step=d, nu_step=d)
        expected.update({
            ('w_tilde',):           seed,
            ('xi_tilde',):          plus(seed, xi_tilde_step=d),
            ('mu_tilde',):          plus(seed, xi_tilde_step=d, mu_tilde_step=d),
            ('nu_tilde',):          plus(seed, xi_tilde_step=d, nu_tilde_step=d, nu_step=d),
            ('H_tilde',):           plus(sweeps, H_tilde_step=d),
            ('eta_tilde',):         plus(sweeps, eta_tilde_step=d),
            ('V_tilde',):           plus(sweeps, eta_step=d, eta_tilde_top_step=d, V_tilde_step=d),
            ('g',):                 plus(sweeps, eta_step=d, eta_tilde_top_step=d, V_tilde_step=d, H_tilde_step=d,
                                         g_step=1),
            ('g', 'eta_tilde'):     plus(sweeps, eta_step=d, eta_tilde_top_step=d, V_tilde_step=d, H_tilde_step=d,
                                         g_step=1, eta_tilde_step=d),
        })
        # the dot grade (the frame's form) and JtJp: Jp reads the plain mu, nu (through R) and eta (through U), not
        # nu_L or eta_D; the Jp-seeded adjoint runs each tilde step once more, with no z and no r
        plain = {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d}
        dots = plus(plain, xi_dot_step=d, mu_dot_step=d, nu_dot_step=d, eta_dot_step=d, Jp_step=d)
        jp_adjoint = dict(w_tilde_step=1, xi_tilde_step=d, mu_tilde_step=d, nu_tilde_step=d, eta_tilde_top_step=d,
                          V_tilde_step=d, H_tilde_step=d, g_step=1)
        with_directions = {
            ('xi_U_dot',):          {'xi_dot_step': d},
            ('nu_L_dot',):          {'xi_step': d, 'nu_step': d, 'xi_dot_step': d, 'nu_dot_step': d, 'identity_step': 1},
            ('eta_D_dot',):         plus(plain, eta_step=-d, xi_dot_step=d, mu_dot_step=d, nu_dot_step=d,
                                         eta_dot_step=d, identity_step=1),
            ('Jp',):                dots,
            ('Jp_norm_sq',):        plus(dots, Jp_norm_sq_step=1),
            ('JtJp',):              plus(dots, **jp_adjoint),
            ('g', 'JtJp'):          {'xi_step': d, 'mu_step': d, 'nu_step': 2 * d, 'eta_step': 2 * d, 'z_step': d,
                                     'r_step': 1, 'w_tilde_step': 2, 'xi_tilde_step': 2 * d, 'mu_tilde_step': 2 * d,
                                     'nu_tilde_step': 2 * d, 'eta_tilde_top_step': 2 * d, 'V_tilde_step': 2 * d,
                                     'H_tilde_step': 2 * d, 'g_step': 2, 'xi_dot_step': d, 'mu_dot_step': d,
                                     'nu_dot_step': d, 'eta_dot_step': d, 'Jp_step': d},
        }
        variations = (self.problem.V, self.problem.H)
        for want, counts in with_directions.items():
            with self.subTest(want=want, directions=True):
                with CountingSteps() as steps:
                    sv.sampling_variables('probe', self.problem.frame, self.sample, want, order=(1,), y=self.y,
                                          weight=self.weight, variations=variations)
                self.assertEqual(steps.counts(), {k: v for k, v in counts.items() if v})
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
        with self.assertRaisesRegex(ValueError, "unknown variable 'Hp', 'zz' for the probe kind"):
            sv.sampling_variables('probe', self.frame, self.ww, ('z', 'Hp', 'zz'))
        for private in ('_Jp.mu_tilde', '_frame.nu_L_dot'):
            with self.assertRaisesRegex(ValueError, 'unknown variable %r' % private):
                sv.sampling_variables('probe', self.frame, self.ww, (private,))
        with self.assertRaisesRegex(ValueError, "unknown variable '_eta_tilde_top'"):     # a private node
            sv.sampling_variables('probe', self.frame, self.ww, ('_eta_tilde_top',))
        with self.assertRaisesRegex(TypeError, 'want is a sequence of names'):
            sv.sampling_variables('probe', self.frame, self.ww, 'z')

    def test_the_jet_shape(self):
        with self.assertRaisesRegex(ValueError, 'the entries kind has no jets'):
            sv.sampling_variables('entries', self.frame, self.problem.index, ('z',), order=(1,))
        for bad in (1, [1], None):
            with self.subTest(order=repr(bad)):
                with self.assertRaisesRegex(TypeError, 'order is the jet shape'):
                    sv.sampling_variables('probe', self.frame, self.ww, ('z',), order=bad)

    def test_every_missing_argument_raises(self):
        for want in (('Jp',), ('JtJp',), ('nu_L_dot',), ('Jp_norm_sq',)):
            with self.subTest(want=want, missing='variations'):
                with self.assertRaisesRegex(ValueError, "needs the argument (core_directions or )?variations, which "
                                                        "was not given"):
                    sv.sampling_variables('probe', self.frame, self.ww, want)
        with self.subTest(want='JtJp', message='names JtJp, not a chain of private nodes'):
            with self.assertRaisesRegex(ValueError, "'JtJp' needs the argument variations, which was not given$"):
                sv.sampling_variables('probe', self.frame, self.ww, ('JtJp',))
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
            'one array too few':   (sv.HeldVariables(mu=mu[:-1]), r"the held mu must be a tuple of 4; got a tuple of "
                                                                   r"length 3"),
            'one array, not a tuple': (sv.HeldVariables(mu=mu[0]), r"the held mu must be a tuple of 4; got ndarray"),
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
        # the slots that are not a tuple over an index: f (one array, shape C), r_block_sumsq ((d,) + F), g (a pair)
        y = self.problem.data('probe', defs.NO_JETS)[0]
        f, blocks, g = sv.sampling_variables('probe', self.frame, self.ww, ('f', 'r_block_sumsq', 'g'), y=y)
        sv.sampling_variables('probe', self.frame, self.ww, ('w',), held=sv.HeldVariables(f=f, r_block_sumsq=blocks, g=g))
        for label, held, pattern in (
                ('f as a tuple', sv.HeldVariables(f=(f,)), r'the held f has shape tuple; expected \(\)'),
                ('f with an axis', sv.HeldVariables(f=f[None]), r'the held f has shape \(1,\); expected \(\)'),
                ('blocks of a jet axis', sv.HeldVariables(r_block_sumsq=blocks[:, None]),
                 r'the held r_block_sumsq has shape \(3, 1\); expected \(3,\)'),
                ('g one family', sv.HeldVariables(g=g[0]), r'the held g must be a tuple of 2; got a tuple of length 3'),
                ('g a block short', sv.HeldVariables(g=(g[0], g[1][:2])),
                 r'the held g\[1\] must be a tuple of 3; got a tuple of length 2'),
                ('g a block of V where H goes', sv.HeldVariables(g=(g[0], g[0])), r'the held g\[1\]\[0\] has shape')):
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, pattern):
                    sv.sampling_variables('probe', self.frame, self.ww, ('w',), held=held)
        with self.assertRaisesRegex(TypeError, 'held is a HeldVariables or None'):
            sv.sampling_variables('probe', self.frame, self.ww, ('z',), held={'mu': mu})

    def test_a_frame_of_neither_form(self):
        with self.assertRaisesRegex(ValueError, 'a ragged frame has four stores'):
            sv.sampling_variables('probe', self.frame[:3], self.ww, ('z',))
        with self.assertRaisesRegex(ValueError, 'a corewise point has one form of a direction, the raw one: pass it as '
                                                'core_directions'):
            sv.sampling_variables('probe', (self.frame[0], self.frame[2]), self.ww, ('z',),
                                  variations=(self.problem.V, self.problem.H))
        for count in (3, 5):                                    # a uniform frame has six elements, a point four
            with self.subTest(uniform=count):
                with self.assertRaisesRegex(ValueError, 'got %d on the uniform layout' % count):
                    sv.sampling_variables('probe', (np.zeros((3, 2, 2)),) * count, self.ww, ('z',))


if __name__ == '__main__':
    unittest.main()
