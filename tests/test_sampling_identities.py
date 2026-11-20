# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_variables``: identities between calls of the engine, with no reference.

1. **No jets is slot zero** (`TestSlotZero`): every variable at ``order = ()`` equals slot ``(0, ..., 0)`` of the
   same variable at any jet shape, at the input jets whose slot ``(0, ..., 0)`` is the sample.
2. **One variable of two** (`TestRestriction`): a call at ``(n_1, n_2)`` along two directions, restricted to the
   slots ``(k, 0)``, is the call at ``(n_1,)`` along the first direction.
3. **Held variables** (`TestHeld`): a held variable gives the result of a recomputed one, bit for bit; a held
   variable deliberately changed changes what reads it, as the recurrence says (it is used, not recomputed).
4. **The two routes of ``z``** (`TestTwoRoutesOfZ`): ``z`` through ``D`` (the sweep of ``nu_L`` and ``eta_D``)
   equals ``z`` with the Tucker centers ``Vc`` (``eta`` through the frame), on a frame whose stores and centers
   agree.

numpy only, by the house convention. Comparisons to ``1e-12`` relative where two routes compute one number, exact
where they share every operation.
"""
import os
import sys
import unittest

import numpy as np

import t3toolbox.backend.jets as jets
import t3toolbox.backend.sampling_variables as sv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampling_definitions as defs            # the problems and the jet matrix of the reference's tests
import test_sampling_variables as tsv               # the engine's problems, and a frame with its centers

TOL = 1e-12
INDEXED = ('xi', 'mu', 'nu', 'nu_L', 'eta', 'eta_D', 'z', 'r')


class TestSlotZero(unittest.TestCase):

    def test_no_jets_is_slot_zero_of_any_jet_shape(self):
        for problem_name, make in tsv.PROBLEMS:
            problem = make()
            for case in defs.WITH_JETS:
                y, _ = problem.data('probe', case)
                zero = (0,) * case.m
                with_jets = sv.sampling_variables('probe', problem.frame, problem.input_jets(case),
                                                  INDEXED + ('r_block_sumsq',), order=case.order, y=y)
                without = sv.sampling_variables('probe', problem.frame, problem.ww, INDEXED + ('r_block_sumsq',),
                                                y=[y_i[zero] for y_i in y])
                for name, a, b in zip(INDEXED, with_jets, without):
                    with self.subTest(problem=problem_name, jets=case.label, name=name):
                        for a_i, b_i in zip(a, b):
                            self.assertLessEqual(defs.relative_error(a_i[zero], b_i), TOL)
                with self.subTest(problem=problem_name, jets=case.label, name='r_block_sumsq'):
                    self.assertLessEqual(defs.relative_error(with_jets[-1][(slice(None),) + zero], without[-1]), TOL)


class TestRestriction(unittest.TestCase):

    def test_two_variables_restricted_to_the_first(self):
        """At ``(n_1, n_2)`` along ``ww + s_1 pp + s_2 qq``, the slots ``(k, 0)`` are the call at ``(n_1,)`` along
        ``ww + s pp``: the second variable at zero."""
        for problem_name, make in tsv.PROBLEMS:
            problem = make()
            for n1, n2 in ((1, 2), (3, 1), (2, 0)):
                plane = sv.sampling_variables('probe', problem.frame, jets.affine_jet(problem.ww, problem.pp,
                                                                                      problem.qq), INDEXED[:-1],
                                              order=(n1, n2))
                line = sv.sampling_variables('probe', problem.frame, jets.affine_jet(problem.ww, problem.pp),
                                             INDEXED[:-1], order=(n1,))
                for name, a, b in zip(INDEXED, plane, line):
                    with self.subTest(problem=problem_name, order=(n1, n2), name=name):
                        for a_i, b_i in zip(a, b):
                            self.assertLessEqual(defs.relative_error(a_i[:, 0], b_i), TOL)


class TestHeld(unittest.TestCase):

    def setUp(self):
        self.problem = tsv.tails_one_axis()
        self.case = defs.MIXED_MODES
        self.y, self.weight = self.problem.data('probe', self.case)
        self.sample = self.problem.input_jets(self.case)

    def call(self, names, **extra):
        return sv.sampling_variables('probe', self.problem.frame, self.sample, names, order=self.case.order,
                                     y=self.y, weight=self.weight, **extra)

    def test_a_held_variable_gives_the_recomputed_result(self):
        """Each slot held in turn, everything wanted: the same arrays, bit for bit (the held value is the one a
        recomputation makes, and every later operation is the same)."""
        names = INDEXED + ('f', 'r_block_sumsq')
        everything = dict(zip(names, self.call(names)))
        for slot in sv.HeldVariables.slots():
            held = sv.HeldVariables(**{slot: everything[slot]})
            again = dict(zip(names, self.call(names, held=held)))
            with self.subTest(held=slot):
                for name in names:
                    for a, b in zip(tsv.arrays_of(again[name]), tsv.arrays_of(everything[name])):
                        self.assertTrue(np.array_equal(a, b))

    def test_a_held_variable_is_used(self):
        """``mu`` held doubled doubles ``eta`` and ``eta_D`` (each linear in ``mu``) and leaves ``nu`` alone;
        ``z`` held doubled doubles ``r + y``; ``r`` held makes ``f`` with no ``y`` at all."""
        mu, nu, eta, eta_D, z, r, f = self.call(('mu', 'nu', 'eta', 'eta_D', 'z', 'r', 'f'))
        doubled = sv.HeldVariables(mu=tuple(2.0 * m for m in mu))
        nu2, eta2, eta_D2 = self.call(('nu', 'eta', 'eta_D'), held=doubled)
        for a, b in zip(eta2 + eta_D2, eta + eta_D):
            self.assertLessEqual(defs.relative_error(a, 2.0 * b), TOL)
        for a, b in zip(nu2, nu):
            self.assertTrue(np.array_equal(a, b))
        (r2,) = self.call(('r',), held=sv.HeldVariables(z=tuple(2.0 * z_i for z_i in z)))
        for r2_i, z_i, y_i in zip(r2, z, self.y):
            self.assertLessEqual(defs.relative_error(r2_i, 2.0 * z_i - y_i), TOL)
        (f2,) = sv.sampling_variables('probe', self.problem.frame, self.sample, ('f',), order=self.case.order,
                                      weight=self.weight, held=sv.HeldVariables(r=r))
        self.assertEqual(float(f2), float(f))


class TestTwoRoutesOfZ(unittest.TestCase):

    def test_z_through_D_equals_z_with_the_centers(self):
        frame, Vc = tsv.consistent_frame(seed=11)
        rng = np.random.default_rng(12)
        ww = [rng.standard_normal((3, D.shape[-1])) for D in frame[0]]
        pp = [rng.standard_normal((3, D.shape[-1])) for D in frame[0]]
        qq = [rng.standard_normal((3, D.shape[-1])) for D in frame[0]]
        for order, sample in (((), ww), ((3,), jets.affine_jet(ww, pp)), ((1, 2), jets.affine_jet(ww, pp, qq))):
            (through_D,) = sv.sampling_variables('probe', frame, sample, ('z',), order=order)
            (with_Vc,) = sv.sampling_variables('probe', frame, sample, ('z',), order=order, Vc=Vc)
            with self.subTest(order=order):
                for a, b in zip(with_Vc, through_D):
                    self.assertLessEqual(defs.relative_error(a, b), TOL)


if __name__ == '__main__':
    unittest.main()
