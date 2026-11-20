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
4. **The gradient** (`TestGradient`): ``g`` paired with a direction is the derivative of ``f`` along it, by a
   central difference (exact to rounding: ``f`` is quadratic along the direction).
5. **The direction** (`TestTheDirection`): per element of a stack of two directions, the adjoint identity
   ``<Jp, omega^2 r> = <p, g>``, ``<p, JtJp> = Jp_norm_sq``, and ``JtJp`` equal to ``g`` with ``Jp`` held as ``r``.
6. **The two forms of a direction** (`TestTheTwoFormsOfADirection`): along the undamped lift of a gauged ``(V, H)``
   the raw and frame ``nu_L_dot``, ``eta_D_dot`` and ``Jp`` are equal; with both directions, the public names are
   the raw form and ``Jp`` the frame's (O-12).
7. **The two routes of ``z``** (`TestTwoRoutesOfZ`): ``z`` through ``D`` (the sweep of ``nu_L`` and ``eta_D``)
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
import t3toolbox.backend.sampling_definitions as sdef
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
                                     y=self.y, weight=self.weight, variations=tsv.stacked_directions(self.problem),
                                     core_directions=tsv.stacked_core_directions(self.problem), **extra)

    def test_a_held_variable_gives_the_recomputed_result(self):
        """Each slot held in turn, everything wanted: the same arrays, bit for bit (the held value is the one a
        recomputation makes, and every later operation is the same)."""
        names = sv._public(sv._PROBE_GRAPH)                     # the slots of a manifold call
        everything = dict(zip(names, self.call(names)))
        for slot in names:
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


    def test_a_held_public_dot_never_feeds_Jp(self):
        """``Jp`` reads the private frame nodes, so a held ``nu_L_dot`` or ``eta_D_dot`` (which in P8 may be the raw
        form) is a leaf for its public name only: ``Jp`` is unchanged, bit for bit, and the public name comes back
        as held."""
        Jp, nu_L_dot, eta_D_dot = self.call(('Jp', 'nu_L_dot', 'eta_D_dot'))
        changed = sv.HeldVariables(nu_L_dot=tuple(2.0 * a for a in nu_L_dot),
                                   eta_D_dot=tuple(2.0 * a for a in eta_D_dot))
        Jp2, eta_D_dot2 = self.call(('Jp', 'eta_D_dot'), held=changed)
        for a, b in zip(Jp2, Jp):
            self.assertTrue(np.array_equal(a, b))
        for a, b in zip(eta_D_dot2, eta_D_dot):
            self.assertTrue(np.array_equal(a, 2.0 * b))


class TestGradient(unittest.TestCase):

    def test_g_is_the_derivative_of_f(self):
        """Along a direction ``p = (V, H)`` at the frame, ``f`` is a quadratic in ``t`` (``z`` moves along the
        reference's ``Jp``, linearly), so the central difference of ``f`` is exact to rounding: it equals ``<g, p>``.
        A random nonuniform weight, every jet case."""
        problem = tsv.tails_one_axis()
        for case in defs.JET_CASES:
            y, weight = problem.data('probe', case)
            sample = problem.input_jets(case)
            g, z = sv.sampling_variables('probe', problem.frame, sample, ('g', 'z'), order=case.order, y=y,
                                         weight=weight)
            Jp = sdef.Jp('probe', problem.frame, sample, None, case.order, variations=(problem.V, problem.H))
            weights = [np.asarray(weight[i]).reshape(case.box + (1,) * (z[i].ndim - case.m)) for i in range(problem.d)]
            f = lambda t: 0.5 * sum(float(np.sum((w_i * (z_i + t * p_i - y_i)) ** 2))
                                    for w_i, z_i, p_i, y_i in zip(weights, z, Jp, y))
            h = 1e-3
            difference = (f(h) - f(-h)) / (2 * h)
            paired = sum(float(np.sum(a * b)) for a, b in zip(g[0] + g[1], tuple(problem.V) + tuple(problem.H)))
            with self.subTest(jets=case.label):
                self.assertLessEqual(abs(difference - paired), 1e-9 * max(abs(paired), 1.0))


class TestTheDirection(unittest.TestCase):
    """The identities of the dot grade and of the adjoint seeded by ``Jp``, per element of a stack of two directions,
    to rounding: ``<Jp, omega^2 r> = <p, g>`` (the adjoint identity; ``<., .>`` sums every slot, sample and entry);
    ``<p, JtJp> = Jp_norm_sq``; and ``JtJp`` is ``g`` with ``Jp`` held as ``r`` (no data)."""

    def test_the_identities_per_direction(self):
        for problem_name, make in tsv.PROBLEMS:
            problem = make()
            variations = tsv.stacked_directions(problem)
            for case in defs.JET_CASES:
                y, weight = problem.data('probe', case)
                sample = problem.input_jets(case)
                Jp, JtJp, norm_sq, r, g = sv.sampling_variables(
                    'probe', problem.frame, sample, ('Jp', 'JtJp', 'Jp_norm_sq', 'r', 'g'), order=case.order, y=y,
                    weight=weight, variations=variations)
                lead = case.m + len(problem.W)
                weights = [np.asarray(weight[i]).reshape(case.box + (1,) * (r[i].ndim - case.m))
                           for i in range(problem.d)]
                pair = lambda blocks, p: sum(float(np.sum(a * b)) for a, b in zip(blocks[0] + blocks[1],
                                                                                   tuple(p[0]) + tuple(p[1])))
                for k in range(2):
                    p = tsv.direction(variations, k)
                    Jp_k = tsv.element('Jp', Jp, k, lead)
                    JtJp_k = tsv.element('JtJp', JtJp, k, lead)
                    with self.subTest(problem=problem_name, jets=case.label, k=k):
                        adjoint = sum(float(np.sum(w_i ** 2 * a * b)) for w_i, a, b in zip(weights, Jp_k, r))
                        self.assertLessEqual(abs(adjoint - pair(g, p)), TOL * max(abs(adjoint), 1.0))
                        self.assertLessEqual(abs(pair(JtJp_k, p) - float(norm_sq[k])),
                                             TOL * max(abs(float(norm_sq[k])), 1.0))
                        (g_of_Jp,) = sv.sampling_variables('probe', problem.frame, sample, ('g',), order=case.order,
                                                           weight=weight, held=sv.HeldVariables(r=Jp_k))
                        for a, b in zip(tsv.arrays_of(g_of_Jp), tsv.arrays_of(JtJp_k)):
                            self.assertLessEqual(defs.relative_error(a, b), TOL)


class TestTheTwoFormsOfADirection(unittest.TestCase):
    """The raw form of a direction is the frame's form along its lift (``subtree_gauge_check.py``): on a frame the
    library builds, with a gauged ``(V, H)`` and its undamped lift ``(D_dot, L_dot)``, the raw and the frame's
    ``nu_L_dot``, ``eta_D_dot`` and ``Jp`` are the same arrays, to rounding. And O-12: with both directions given, the
    public ``nu_L_dot`` and ``eta_D_dot`` are the raw form and ``Jp`` the frame's."""

    def test_along_the_lift_and_with_both(self):
        import t3toolbox.backend.fv_operations as fvo
        import t3toolbox.backend.tv_operations as tvo
        frame, _ = tsv.consistent_frame(seed=21)
        np.random.seed(22)
        v = tvo.tv_orthogonal_gauge_projection(frame, fvo.fv_variations_randn(fvo.fv_variation_shapes(frame), ()))
        lift = tuple(tuple(c) for c in tvo.tv_corewise_lift(frame, v, fvo.fv_lift_factors(frame, 0.0)))
        variations = tuple(tuple(c) for c in v)
        rng = np.random.default_rng(23)
        ww = [rng.standard_normal((3, D.shape[-1])) for D in frame[0]]
        pp = [rng.standard_normal((3, D.shape[-1])) for D in frame[0]]
        names = ('nu_L_dot', 'eta_D_dot', 'Jp')
        for order, sample in (((), ww), ((3,), jets.affine_jet(ww, pp))):
            frame_form = sv.sampling_variables('probe', frame, sample, names, order=order, variations=variations)
            raw_form = sv.sampling_variables('probe', frame, sample, names, order=order, core_directions=lift)
            for name, a, b in zip(names, frame_form, raw_form):
                with self.subTest(order=order, name=name):
                    for x, y in zip(a, b):
                        self.assertLessEqual(defs.relative_error(x, y), TOL)
        # O-12, on arbitrary directions (where the two forms differ): the public names raw, Jp the frame's
        problem = tsv.tails_one_axis()
        raw = sv.sampling_variables('probe', problem.frame, problem.ww, names,
                                    core_directions=(problem.D_dot, problem.L_dot))
        framed = sv.sampling_variables('probe', problem.frame, problem.ww, names, variations=(problem.V, problem.H))
        both = sv.sampling_variables('probe', problem.frame, problem.ww, names, variations=(problem.V, problem.H),
                                     core_directions=(problem.D_dot, problem.L_dot))
        for name, a, b in zip(names, both, [raw[0], raw[1], framed[2]]):
            with self.subTest(both=True, name=name):
                for x, y in zip(a, b):
                    self.assertTrue(np.array_equal(x, y))
        self.assertGreater(defs.relative_error(raw[0][1], framed[0][1]), 1e-3)           # the forms differ here


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
