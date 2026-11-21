# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_variables`` for the apply kind: one number per sample and slot, the seed on bond ``d``.

1. **By definition** (`TestByDefinition`): every name an apply call accepts against the reference module, on a
   manifold frame and on a corewise point ``(B, G)`` (through the table of ``SAMPLE_VARIABLES.md`` section 9), with
   ``K = ()`` and, element by element, ``K = (2,)``, over the jet matrix and on three problems, one with end bonds that
   differ between the ``L`` and the ``R`` cores.
2. **The names** (`TestNames`): per kind (all three) and call type, the names the engine accepts against the reference
   module's, the difference stated exactly; asking for a zero name raises, saying it is zero.
3. **The seed on bond d** (`TestTheSeedOnBondD`): ``nu_tilde`` and ``nu_L_tilde`` are the jet products of the seed
   and ``nu``, ``nu_L``; their start is the seed times the closing ones; ``z`` and ``Jp`` close ``mu[d]`` and
   ``mu_R_dot[d]`` with the ones of their own end bonds (``rL_d != rR_d`` there).
4. **The identities** (`TestIdentities`): per element of a stack of two directions, ``<Jp, omega^2 r> = <p, g>``,
   ``<p, JtJp> = Jp_norm_sq`` and ``JtJp`` equal to ``g`` with ``Jp`` held as ``r``, on both call types; the corewise
   call at ``(D, L)`` against the manifold call. (The multiplier rows of ``mu_tilde`` and ``xi_tilde`` do not hold for
   apply: ``SAMPLE_VARIABLES.md`` section 8.)
5. **The start node** (`TestTheStartNode`): ``_deps`` lists a start node; the twin and the corewise rename carry it; a
   held seed is counted for the dtype and read as the start.
6. **Each sweep once** (`TestEachSweepOnce`): the step counts of ``('z', 'g')`` (no plain ``nu`` or ``eta`` sweep),
   ``('Kp',)`` and ``('JtJp',)``, derived by hand; a ``Vc`` is read by nothing.

numpy only.
"""
import os
import sys
import unittest

import numpy as np

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.sampling_definitions as sdef
import t3toolbox.backend.sampling_variables as sv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampling_definitions as defs
import test_sampling_graph as tsg
import test_sampling_variables as tsv
import test_sampling_corewise as tsc

TOL = 1e-10
ZERO = ('xi_tilde', 'mu_tilde', 'xi_tilde_dot', 'mu_tilde_dot')
FRAME_ONLY = ('nu', 'eta', 'nu_tilde', 'eta_tilde', 'xi_U_dot', 'mu_R_dot')   # the R and U bases: no corewise name
WITH_K = ('xi_U_dot', 'mu_R_dot', 'nu_L_dot', 'eta_D_dot', 'Jp', 'Jp_norm_sq', 'JtJp', 'xi_dot', 'mu_dot',
          'nu_L_tilde_dot', 'eta_D_tilde_dot', 'V_tilde_dot', 'H_tilde_dot', 'Kp',
          'nu_dot', 'eta_dot', 'nu_tilde_dot', 'eta_tilde_dot')


def names(call):
    return tuple(name for name in sv._names('apply', call) if name != 'w')


def element(name, value, k, lead):
    """Element ``k`` of the stack of directions: the apply kind's ``Jp`` is one array, its ``K`` after ``lead``."""
    if name not in WITH_K:
        return value
    if name == 'Jp':
        return np.take(value, k, axis=lead)
    return tsv.element(name, value, k, lead)


def directions(problem, stacked):
    if stacked:
        return tsv.stacked_directions(problem), tsv.stacked_core_directions(problem)
    return (problem.V, problem.H), (problem.D_dot, problem.L_dot)


def apply_call(problem, case, want, at=None, **arguments):
    y, weight = problem.data('apply', case)
    return sv.sampling_variables('apply', problem.frame if at is None else at, problem.input_jets(case), want,
                                 order=case.order, **dict(dict(y=y, weight=weight), **arguments))


class TestByDefinition(unittest.TestCase):

    def check(self, call):
        for problem_name, make in tsv.PROBLEMS:
            problem = make()
            for stacked in (False, True):
                variations, core_directions = directions(problem, stacked)
                for case in defs.JET_CASES:
                    y, weight = problem.data('apply', case)
                    if call == '_manifold':
                        got = apply_call(problem, case, names(call), variations=variations,
                                         core_directions=core_directions)
                    else:
                        got = apply_call(problem, case, names(call), at=(problem.D, problem.L),
                                         core_directions=core_directions)
                    lead = case.m + len(problem.W)
                    for k in (range(2) if stacked else (None,)):
                        v = tsv.direction(variations, k) if stacked else variations
                        c = tsv.direction(core_directions, k) if stacked else core_directions
                        for name, value in zip(names(call), got):
                            if call == '_manifold':
                                expected = getattr(sdef, name)('apply', problem.frame, problem.input_jets(case), None,
                                                               case.order, y=y, weight=weight, variations=v,
                                                               core_directions=c)
                            else:
                                B, G = problem.D, problem.L
                                expected = getattr(sdef, tsc.SECTION_9.get(name, name))(
                                    'apply', (B, G, G, G), problem.input_jets(case), None, case.order, y=y,
                                    weight=weight, variations=c, core_directions=c)
                            value = element(name, value, k, lead) if stacked else value
                            with self.subTest(problem=problem_name, stacked=stacked, jets=case.label, k=k, name=name):
                                self.assertEqual(len(tsv.arrays_of(value)), len(tsv.arrays_of(expected)))
                                for a, b in zip(tsv.arrays_of(value), tsv.arrays_of(expected)):
                                    self.assertLessEqual(defs.relative_error(a, b), TOL)

    def test_every_name_on_a_manifold_frame(self):
        self.check('_manifold')

    def test_every_name_on_a_corewise_point(self):
        self.check('_corewise')


class TestNames(unittest.TestCase):

    def test_the_names_against_the_reference(self):
        """The reference module defines every name for every kind (the zero ones as zeros). A manifold call accepts
        exactly those, but for apply and entries the four zero names; a corewise call their basis-free spellings (section 9), but
        the names in the frame's other bases."""
        reference = set(sdef.__all__)
        renamed = {b: a for a, b in tsc.SECTION_9.items()}
        for kind in ('probe', 'apply', 'entries'):
            zero = set(ZERO) if kind != 'probe' else set()
            with self.subTest(kind=kind, call='manifold'):
                self.assertEqual(set(sv._names(kind, '_manifold')), reference - zero)
            with self.subTest(kind=kind, call='corewise'):
                expected = {renamed.get(name, name) for name in reference - zero - set(FRAME_ONLY)}
                self.assertEqual(set(sv._names(kind, '_corewise')), expected)
        self.assertEqual(sv._ZERO_NAMES, {'apply': frozenset(ZERO), 'entries': frozenset(ZERO)})

    def test_a_zero_name_raises_saying_it_is_zero(self):
        problem = defs.small()
        for at, call in ((problem.frame, 'manifold'), ((problem.D, problem.L), 'corewise')):
            for name in ZERO:
                with self.subTest(call=call, name=name):
                    with self.assertRaisesRegex(ValueError, r"unknown variable %r for the apply kind on a %s call "
                                                            r"\(%r is zero for the apply kind: the seed sits on the "
                                                            r"ones that close bond d" % (name, call, name)):
                        sv.sampling_variables('apply', at, problem.ww, (name,))
        with self.assertRaisesRegex(ValueError, r"unknown variable 'zz' for the apply kind on a manifold call; "):
            sv.sampling_variables('apply', problem.frame, problem.ww, ('zz',))


class TestTheSeedOnBondD(unittest.TestCase):

    def test_nu_tilde_is_the_seed_times_nu(self):
        """``nu_tilde = w_tilde * nu`` and ``nu_L_tilde = w_tilde * nu_L``, the truncated jet product (the seed
        multiplies the ones at bond ``d``, and the recurrence is linear in its carry); the start, bond ``d``, is the
        seed times the ones of the end bond."""
        for problem_name, make in tsv.PROBLEMS:
            problem = make()
            for case in defs.JET_CASES:
                nu_tilde, nu_L_tilde, nu, nu_L, w_tilde = apply_call(
                    problem, case, ('nu_tilde', 'nu_L_tilde', 'nu', 'nu_L', 'w_tilde'))
                lens = contractions.GroupLengths(J=case.m, W=len(problem.W), K=0, C=0)
                for tilde, plain in ((nu_tilde, nu), (nu_L_tilde, nu_L)):
                    with self.subTest(problem=problem_name, jets=case.label):
                        for a, b in zip(tilde, plain):
                            product = contractions.jet_contract('JW,JWa->JWa', w_tilde, b, lens=lens,
                                                                order=case.order, over_orders='loop')
                            self.assertLessEqual(defs.relative_error(a, product), 1e-12)
                        self.assertTrue(np.array_equal(tilde[-1], w_tilde[..., None] * np.ones(plain[-1].shape[-1])))

    def test_z_and_Jp_close_their_own_end_bond(self):
        """With ``rL_d != rR_d``: ``z`` is ``mu[d]`` summed over its ``rL_d`` axis, ``Jp`` is ``mu_R_dot[d]`` summed
        over its ``rR_d`` axis (and the raw route ``mu_dot[d]`` over ``rL_d``)."""
        problem = tsv.tails_one_axis()
        D, U, L, R = problem.frame
        self.assertNotEqual(L[-1].shape[-1], R[-1].shape[-1])
        for case in (defs.NO_JETS, defs.LINE_3):
            z, mu, Jp, mu_R_dot = apply_call(problem, case, ('z', 'mu', 'Jp', 'mu_R_dot'),
                                             variations=(problem.V, problem.H))
            Jp_raw, mu_dot = apply_call(problem, case, ('Jp', 'mu_dot'), core_directions=(problem.D_dot, problem.L_dot))
            with self.subTest(jets=case.label):
                self.assertEqual((mu[-1].shape[-1], mu_R_dot[-1].shape[-1]), (L[-1].shape[-1], R[-1].shape[-1]))
                self.assertTrue(np.array_equal(z, mu[-1].sum(axis=-1)))
                self.assertTrue(np.array_equal(Jp, mu_R_dot[-1].sum(axis=-1)))
                self.assertTrue(np.array_equal(Jp_raw, mu_dot[-1].sum(axis=-1)))
                expected = sdef.Jp('apply', problem.frame, problem.input_jets(case), None, case.order,
                                   variations=(problem.V, problem.H))
                self.assertLessEqual(defs.relative_error(Jp, expected), TOL)


class TestIdentities(unittest.TestCase):

    def test_the_direction_per_element(self):
        for problem_name, make in tsv.PROBLEMS:
            problem = make()
            variations, core_directions = directions(problem, True)
            for call, at, arguments, p_of in (
                    ('manifold', problem.frame, dict(variations=variations), lambda k: tsv.direction(variations, k)),
                    ('corewise', (problem.D, problem.L), dict(core_directions=core_directions),
                     lambda k: tsv.direction(core_directions, k))):
                for case in defs.JET_CASES:
                    y, weight = problem.data('apply', case)
                    Jp, JtJp, norm_sq, r, g = apply_call(problem, case, ('Jp', 'JtJp', 'Jp_norm_sq', 'r', 'g'), at=at,
                                                         **arguments)
                    lead = case.m + len(problem.W)
                    weight_2 = np.asarray(weight).reshape(case.box + (1,) * (r.ndim - case.m)) ** 2
                    pair = lambda blocks, p: sum(float(np.sum(a * b)) for a, b in zip(blocks[0] + blocks[1],
                                                                                       tuple(p[0]) + tuple(p[1])))
                    for k in range(2):
                        Jp_k = np.take(Jp, k, axis=lead)
                        JtJp_k = tsv.element('JtJp', JtJp, k, lead)
                        with self.subTest(problem=problem_name, call=call, jets=case.label, k=k):
                            adjoint = float(np.sum(weight_2 * Jp_k * r))
                            self.assertLessEqual(abs(adjoint - pair(g, p_of(k))), 1e-12 * max(abs(adjoint), 1.0))
                            self.assertLessEqual(abs(pair(JtJp_k, p_of(k)) - float(norm_sq[k])),
                                                 1e-12 * max(abs(float(norm_sq[k])), 1.0))
                            (g_of_Jp,) = sv.sampling_variables('apply', at, problem.input_jets(case), ('g',),
                                                               order=case.order, weight=weight,
                                                               held=sv.HeldVariables(r=Jp_k))
                            for a, b in zip(tsv.arrays_of(g_of_Jp), tsv.arrays_of(JtJp_k)):
                                self.assertLessEqual(defs.relative_error(a, b), 1e-12)

    def test_the_corewise_call_at_the_base_point(self):
        pairs = (('nu', 'nu_L'), ('eta', 'eta_D'), ('mu', 'mu'), ('z', 'z'), ('Kp', 'Kp'), ('nu_tilde', 'nu_L_tilde'),
                 ('eta_tilde', 'eta_D_tilde'), ('nu_tilde_dot', 'nu_L_tilde_dot'), ('Jp', 'Jp'))
        for problem_name, make in tsv.PROBLEMS:
            problem = make()
            for case in defs.JET_CASES:
                arguments = dict(core_directions=(problem.D_dot, problem.L_dot))
                got = apply_call(problem, case, [a for a, _ in pairs], at=(problem.D, problem.L), **arguments)
                manifold = apply_call(problem, case, [b for _, b in pairs], **arguments)
                for (name, _), a, b in zip(pairs, got, manifold):
                    with self.subTest(problem=problem_name, jets=case.label, name=name):
                        for x, y_ in zip(tsv.arrays_of(a), tsv.arrays_of(b)):
                            self.assertLessEqual(defs.relative_error(x, y_), 1e-12)


class TestTheStartNode(unittest.TestCase):

    def test_deps_and_the_renames_carry_the_start(self):
        rows = {(rule.name, rule.needs): rule for rule in sv._GRAPHS['apply', '_manifold']}
        nu_tilde = rows['nu_tilde', frozenset()]
        self.assertEqual((nu_tilde.reads, nu_tilde.start), (('R', 'xi'), 'w_tilde'))
        self.assertEqual(sv._deps(nu_tilde), ('R', 'xi', 'w_tilde'))
        self.assertEqual(sv._deps(rows['mu', frozenset()]), ('L', 'xi'))         # 'ones' is not a node
        twin = rows['_Jp.nu_tilde', frozenset()]
        self.assertEqual((twin.start, twin.carries_K, sv._deps(twin)),
                         ('_Jp.w_tilde', True, ('R', 'xi', '_Jp.w_tilde')))
        corewise = {rule.name: rule for rule in sv._GRAPHS['apply', '_corewise']}
        self.assertEqual((corewise['nu_tilde'].reads, corewise['nu_tilde'].start), (('L', 'xi'), 'w_tilde'))
        self.assertEqual(corewise['_Jp.nu_tilde'].start, '_Jp.w_tilde')
        names_ = [rule.name for rule in sv._closure('apply', ('nu_tilde',), frozenset(), frozenset({'y', '_manifold'}))]
        self.assertEqual(names_, ['xi', 'mu', 'z', 'r', 'w_tilde', 'nu_tilde'])        # the seed, through the start
        # apply's own start node is neither dropped nor renamed on a corewise point; later rows whose start node is
        # (a sweep starting from the U-basis eta, one from nu_L_tilde) must be dropped, and renamed with it
        later = sv._APPLY_ROWS + (sv.Rule('_from_eta', ('right', 'nu_step'), ('L', 'xi'), sv._NONE, False, 'eta'),
                                  sv.Rule('_from_nu_L_tilde', ('right', 'nu_step'), ('L', 'xi'), sv._NONE, False,
                                          'nu_L_tilde'))
        derived = {rule.name: rule for rule in sv._corewise(later, 'apply')}
        self.assertNotIn('_from_eta', derived)
        self.assertEqual(derived['_from_nu_L_tilde'].start, 'nu_tilde')

    def test_a_held_seed_is_the_start_and_counts_for_the_dtype(self):
        problem = defs.small()
        case = defs.LINE_3
        (w_tilde,) = apply_call(problem, case, ('w_tilde',))
        held = sv.HeldVariables(w_tilde=(2.0 * w_tilde).astype(np.float32))
        nu_tilde, nu = sv.sampling_variables('apply', problem.frame, problem.input_jets(case), ('nu_tilde', 'nu'),
                                             order=case.order, held=held)
        (expected,) = apply_call(problem, case, ('nu_tilde',))
        self.assertEqual(nu_tilde[0].dtype, np.float32)
        self.assertEqual(nu[0].dtype, np.float32)
        for a, b in zip(nu_tilde, expected):
            self.assertLessEqual(defs.relative_error(a, 2.0 * b), 1e-5)
        with self.assertRaisesRegex(ValueError, "'r' needs the argument y, which was not given"):     # no held seed
            sv.sampling_variables('apply', problem.frame, problem.input_jets(case), ('nu_tilde',), order=case.order)

    def test_a_held_z_has_the_apply_shape(self):
        problem = defs.small()
        case = defs.LINE_1
        (z,) = apply_call(problem, case, ('z',))
        self.assertEqual(z.shape, case.box + problem.W)
        (r,) = apply_call(problem, case, ('r',), held=sv.HeldVariables(z=z))
        self.assertLessEqual(defs.relative_error(r, z - problem.data('apply', case)[0]), 1e-15)
        expected = repr(case.box + problem.W).replace('(', r'\(').replace(')', r'\)')
        with self.assertRaisesRegex(ValueError, 'the held z has shape tuple; expected ' + expected):  # probe's form
            apply_call(problem, case, ('r',), held=sv.HeldVariables(z=(z,)))
        with self.assertRaisesRegex(ValueError, r'the held z has shape \(2, 5\); expected ' + expected):
            apply_call(problem, case, ('r',), held=sv.HeldVariables(z=np.zeros((2, 5))))


class TestEachSweepOnce(unittest.TestCase):

    def test_the_step_counts(self):
        """Derived by hand from the apply graph: the output and the seed read only ``xi`` and ``mu``; the adjoint
        sweeps ``nu_step`` once from the seed; there is no plain ``nu`` or ``eta`` sweep."""
        problem = defs.small()
        d = problem.d
        seed = {'xi_step': d, 'mu_step': d, 'close_step': 1, 'r_step': 1, 'w_tilde_step': 1}
        adjoint = dict(nu_step=d, eta_top_step=d, V_top_step=d, H_readout_step=d, g_step=1)
        expected = {
            ('manifold', ('z',)):      {'xi_step': d, 'mu_step': d, 'close_step': 1},
            ('manifold', ('z', 'g')):  dict(seed, **adjoint),
            ('corewise', ('z', 'g')):  dict(seed, **adjoint),
            ('manifold', ('Kp',)):     dict(seed, nu_step=d, xi_dot_step=d, mu_dot_step=d, nu_dot_step=d,
                                            eta_dot_top_step=d, V_top_step=d, H_tilde_step=d, g_step=1),
            ('manifold', ('JtJp',)):   {'xi_step': d, 'mu_step': d, 'xi_dot_step': d, 'mu_dot_step': d,
                                        'close_step': 1, 'w_tilde_step': 1, **adjoint},
        }
        for (call, want), counts in expected.items():
            at = problem.frame if call == 'manifold' else (problem.D, problem.L)
            arguments = dict(core_directions=(problem.D_dot, problem.L_dot))
            if call == 'manifold' and want == ('JtJp',):
                arguments = dict(variations=(problem.V, problem.H))
            with self.subTest(call=call, want=want):
                with tsg.CountingSteps() as steps:
                    apply_call(problem, defs.LINE_1, want, at=at, **arguments)
                self.assertEqual(steps.counts(), counts)

    def test_Vc_is_read_by_nothing(self):
        frame, Vc = tsv.consistent_frame(seed=5)
        rng = np.random.default_rng(6)
        ww = [rng.standard_normal((3, D.shape[-1])) for D in frame[0]]
        with tsg.CountingSteps() as steps:
            (z,) = sv.sampling_variables('apply', frame, ww, ('z',), Vc=Vc)
        (z0,) = sv.sampling_variables('apply', frame, ww, ('z',))
        self.assertTrue(np.array_equal(z, z0))
        self.assertEqual(steps.counts(), {'xi_step': 3, 'mu_step': 3, 'close_step': 1})


if __name__ == '__main__':
    unittest.main()
