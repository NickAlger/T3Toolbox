# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_variables`` on a corewise point ``(B, G)``: one basis on each edge.

1. **By definition** (`TestByDefinition`): every name a corewise call accepts against the reference module on the
   frame ``(B, G, G, G)``, through the table of ``SAMPLE_VARIABLES.md`` section 9 (``nu`` is the reference's ``nu_L``,
   ``eta_dot`` its ``eta_D_dot``, ...), with ``K = ()`` and, element by element, ``K = (2,)``; and ``JtJp`` equals ``g``
   with ``Jp`` held as ``r``.
2. **Against the manifold call** (`TestAgainstTheManifoldCall`): at the frame's base point ``(D, L)``, the corewise
   ``nu`` is the manifold ``nu_L``, ``eta`` is ``eta_D``, and ``z``, ``mu`` and ``Kp`` are the manifold call's.
3. **The derived graph** (`TestTheDerivedGraph`): the rename table (in the docstring too); no corewise row reads ``U``,
   ``R``, ``V``, ``H``, ``Vc`` or a frame node, or needs ``variations``; the two import-time assertions (a rename that
   would land on a surviving name raises; every public name is reachable); the names a corewise call accepts are
   section 9's, and a name with a basis is unknown there; ``variations`` raises.
4. **Each sweep once** (`TestEachSweepOnce`): the step counts of ``('z', 'g')`` and ``('Kp',)``, derived by hand.
5. **Held variables** (`TestHeld`): a held ``nu`` of the corewise sizes passes, one of the manifold ``nu``'s sizes
   (the ``R`` cores) raises.

numpy only.
"""
import os
import re
import sys
import unittest

import numpy as np

import t3toolbox.backend.sampling_definitions as sdef
import t3toolbox.backend.sampling_variables as sv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampling_definitions as defs
import test_sampling_graph as tsg
import test_sampling_variables as tsv

TOL = 1e-10
SECTION_9 = {'nu': 'nu_L', 'eta': 'eta_D', 'nu_tilde': 'nu_L_tilde', 'eta_tilde': 'eta_D_tilde', 'nu_dot': 'nu_L_dot',
             'eta_dot': 'eta_D_dot', 'nu_tilde_dot': 'nu_L_tilde_dot', 'eta_tilde_dot': 'eta_D_tilde_dot'}
NAMES = ('xi', 'mu', 'nu', 'eta', 'z', 'r', 'f', 'r_block_sumsq', 'w_tilde', 'xi_tilde', 'mu_tilde', 'nu_tilde',
         'eta_tilde', 'V_tilde', 'H_tilde', 'g', 'xi_dot', 'mu_dot', 'nu_dot', 'eta_dot', 'Jp', 'Jp_norm_sq',
         'xi_tilde_dot', 'mu_tilde_dot', 'nu_tilde_dot', 'eta_tilde_dot', 'V_tilde_dot', 'H_tilde_dot', 'Kp', 'JtJp')
WITH_K = ('xi_dot', 'mu_dot', 'nu_dot', 'eta_dot', 'Jp', 'Jp_norm_sq', 'xi_tilde_dot', 'mu_tilde_dot', 'nu_tilde_dot',
          'eta_tilde_dot', 'V_tilde_dot', 'H_tilde_dot', 'Kp', 'JtJp')


def corewise(problem, case, names, core_directions=None, **extra):
    y, weight = problem.data('probe', case)
    core_directions = (problem.D_dot, problem.L_dot) if core_directions is None else core_directions
    return sv.sampling_variables('probe', (problem.D, problem.L), problem.input_jets(case), names, order=case.order,
                                 y=y, weight=weight, core_directions=core_directions, **extra)


def reference(problem, case, name, core_directions):
    """The reference on the frame ``(B, G, G, G)``: the frame's form of a direction there is its raw form."""
    y, weight = problem.data('probe', case)
    B, G = problem.D, problem.L
    return getattr(sdef, SECTION_9.get(name, name))('probe', (B, G, G, G), problem.input_jets(case), None, case.order,
                                                     y=y, weight=weight, variations=core_directions,
                                                     core_directions=core_directions)


def element(name, value, k, lead):
    if name not in WITH_K:
        return value
    return tsv.element(name, value, k, lead)


class TestByDefinition(unittest.TestCase):

    def test_every_name_against_the_reference(self):
        self.assertEqual(sv._public(sv._PROBE_COREWISE_GRAPH), NAMES)
        for problem_name, make in tsv.PROBLEMS:
            problem = make()
            for label, core_directions in (('K=()', (problem.D_dot, problem.L_dot)),
                                           ('K=(2,)', tsv.stacked_core_directions(problem))):
                stacked = label != 'K=()'
                for case in defs.JET_CASES:
                    got = dict(zip(NAMES, corewise(problem, case, NAMES, core_directions)))
                    lead = case.m + len(problem.W)
                    for k in (range(2) if stacked else (0,)):
                        direction = tsv.direction(core_directions, k) if stacked else core_directions
                        for name in NAMES:
                            expected = reference(problem, case, name, direction)
                            value = element(name, got[name], k, lead) if stacked else got[name]
                            with self.subTest(problem=problem_name, K=label, jets=case.label, k=k, name=name):
                                for a, b in zip(tsv.arrays_of(value), tsv.arrays_of(expected)):
                                    self.assertLessEqual(defs.relative_error(a, b), TOL)

    def test_JtJp_is_g_with_Jp_held_as_r(self):
        """The corewise twins are derived through the rename: ``JtJp[k]`` is the corewise ``g`` with ``Jp[k]`` held
        as ``r``, per element of a stack of two directions."""
        for problem_name, make in tsv.PROBLEMS:
            problem = make()
            core_directions = tsv.stacked_core_directions(problem)
            for case in defs.JET_CASES:
                Jp, JtJp = corewise(problem, case, ('Jp', 'JtJp'), core_directions)
                lead = case.m + len(problem.W)
                for k in range(2):
                    _, weight = problem.data('probe', case)
                    (g,) = sv.sampling_variables('probe', (problem.D, problem.L), problem.input_jets(case), ('g',),
                                                 order=case.order, weight=weight,
                                                 held=sv.HeldVariables(r=tsv.element('Jp', Jp, k, lead)))
                    with self.subTest(problem=problem_name, jets=case.label, k=k):
                        for a, b in zip(tsv.arrays_of(g), tsv.arrays_of(tsv.element('JtJp', JtJp, k, lead))):
                            self.assertLessEqual(defs.relative_error(a, b), TOL)


class TestAgainstTheManifoldCall(unittest.TestCase):

    def test_at_the_base_point(self):
        """At ``(D, L)`` the corewise point is the base point of the frame ``(D, U, L, R)``: the corewise ``nu`` is the
        manifold ``nu_L``, ``eta`` is ``eta_D``; ``mu``, ``z`` and ``Kp`` (a raw direction at the base point, in both)
        are the same."""
        pairs = (('nu', 'nu_L'), ('eta', 'eta_D'), ('mu', 'mu'), ('z', 'z'), ('Kp', 'Kp'), ('nu_dot', 'nu_L_dot'),
                 ('eta_tilde', 'eta_D_tilde'))
        for problem_name, make in tsv.PROBLEMS:
            problem = make()
            for case in defs.JET_CASES:
                y, weight = problem.data('probe', case)
                sample = problem.input_jets(case)
                arguments = dict(order=case.order, y=y, weight=weight, core_directions=(problem.D_dot, problem.L_dot))
                got = sv.sampling_variables('probe', (problem.D, problem.L), sample, [a for a, _ in pairs], **arguments)
                manifold = sv.sampling_variables('probe', problem.frame, sample, [b for _, b in pairs], **arguments)
                for (name, _), a, b in zip(pairs, got, manifold):
                    with self.subTest(problem=problem_name, jets=case.label, name=name):
                        for x, y_ in zip(tsv.arrays_of(a), tsv.arrays_of(b)):
                            self.assertLessEqual(defs.relative_error(x, y_), TOL)


class TestTheDerivedGraph(unittest.TestCase):

    def test_the_rename_table(self):
        rows = tsg.doc_table('| On a manifold frame | On a corewise point |')
        self.assertEqual({tsg.names_of(a)[0]: tsg.names_of(b)[0] for a, b in rows}, sv._COREWISE_RENAME)
        made = set(sv._made(sv._PROBE_COREWISE_GRAPH))
        for rule in sv._PROBE_COREWISE_GRAPH:
            with self.subTest(name=rule.name):
                self.assertFalse(set(rule.reads) & {'U', 'R', 'V', 'H', 'Vc'})
                self.assertFalse(any(read.startswith('_frame.') for read in rule.reads))
                self.assertNotIn('variations', rule.needs)
                self.assertNotIn('_manifold', rule.needs)
                self.assertNotIn(rule.name, sv._COREWISE_RENAME)                     # every name renamed
                self.assertTrue(set(rule.reads) <= made | {'D', 'L'} | sv._ARGUMENTS)
        self.assertEqual({r.name for r in sv._PROBE_COREWISE_GRAPH if r.name.startswith('_Jp.')},
                         {sv._twin(n) for n in sv._JP_PATHS['probe', '_corewise']})
        jtjp = [r for r in sv._PROBE_COREWISE_GRAPH if r.name == 'JtJp']
        self.assertEqual([r.needs for r in jtjp], [frozenset({'core_directions'})])

    def test_the_import_time_assertions(self):
        """A rename that would land on a name a surviving row keeps raises; every public name is reachable with
        ``core_directions`` and data (asserted at import, and again here)."""
        colliding = sv._PROBE_ROWS + (sv.Rule('nu', ('right', 'nu_step'), ('L', 'xi'), sv._NONE, False, 'ones'),)
        with self.assertRaisesRegex(RuntimeError, "the corewise rename of 'nu_L' lands on the surviving 'nu'"):
            sv._corewise(colliding, 'probe')
        # no surviving probe row needs variations today; a later one (a route reading only D, L and data) must come
        # out needing core_directions, the one form of a direction at a corewise point
        later = sv._PROBE_ROWS + (sv.Rule('later', ('whole', 'identity_step'), ('z',), sv._VAR, True),)
        self.assertEqual([r.needs for r in sv._corewise(later, 'probe') if r.name == 'later'], [sv._COR])
        for name in sv._public(sv._PROBE_COREWISE_GRAPH):
            with self.subTest(name=name):
                sv._closure('probe', (name,), frozenset(), frozenset({'_corewise', 'y', 'core_directions'}))

    def test_a_name_with_a_basis_is_unknown_and_variations_raises(self):
        problem = defs.small()
        for name in ('nu_L', 'eta_D', 'nu_L_dot', 'eta_D_tilde_dot', 'xi_U_dot', 'mu_R_dot', 'nu_L_tilde'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, r"unknown variable %r for the probe kind on a corewise call"
                                            % name):
                    sv.sampling_variables('probe', (problem.D, problem.L), problem.ww, (name,))
        with self.assertRaisesRegex(ValueError, 'a corewise point has one form of a direction'):
            sv.sampling_variables('probe', (problem.D, problem.L), problem.ww, ('z',),
                                  variations=(problem.D_dot, problem.L_dot))


class TestEachSweepOnce(unittest.TestCase):

    def test_the_step_counts(self):
        """Derived by hand from the corewise graph: one sweep of each family (``nu_step`` makes the one ``nu``)."""
        problem = defs.small()
        d = problem.d
        plain = {'xi_step': d, 'mu_step': d, 'nu_step': d, 'eta_step': d, 'z_step': d, 'r_step': 1}
        adjoint = dict(plain, w_tilde_step=1, xi_tilde_step=d, mu_tilde_step=d, nu_tilde_step=d)
        expected = {
            ('z', 'g'): dict(adjoint, eta_tilde_top_step=d, V_tilde_step=d, H_tilde_step=d, g_step=1),
            ('Kp',): dict(adjoint, xi_dot_step=d, mu_dot_step=d, nu_dot_step=d, eta_dot_step=d, xi_tilde_dot_step=d,
                          mu_tilde_dot_step=d, nu_tilde_dot_step=d, eta_tilde_dot_top_step=d, V_tilde_dot_step=d,
                          H_tilde_dot_step=d, g_step=1),
        }
        for want, counts in expected.items():
            with self.subTest(want=want):
                with tsg.CountingSteps() as steps:
                    corewise(problem, defs.LINE_1, want)
                self.assertEqual(steps.counts(), counts)


class TestHeld(unittest.TestCase):

    def test_a_held_nu_has_the_corewise_sizes(self):
        """On a frame whose ``L`` and ``R`` bonds differ, the corewise ``nu`` has the ``L`` (= ``G``) sizes: one of
        those passes, and the manifold ``nu`` (the ``R`` sizes) raises."""
        problem = tsv.tails_one_axis()
        nu_L, nu = sv.sampling_variables('probe', problem.frame, problem.ww, ('nu_L', 'nu'))
        (eta,) = sv.sampling_variables('probe', (problem.D, problem.L), problem.ww, ('eta',),
                                       held=sv.HeldVariables(nu=nu_L))
        (eta_D,) = sv.sampling_variables('probe', problem.frame, problem.ww, ('eta_D',))
        for a, b in zip(eta, eta_D):
            self.assertLessEqual(defs.relative_error(a, b), TOL)
        with self.assertRaisesRegex(ValueError, re.escape('the held nu[0] has shape (3, 3); expected (3, 2)')):
            sv.sampling_variables('probe', (problem.D, problem.L), problem.ww, ('eta',), held=sv.HeldVariables(nu=nu))


if __name__ == '__main__':
    unittest.main()
