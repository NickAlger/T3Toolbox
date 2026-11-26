# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_variables``: every variable the engine makes, against its definition.

1. **By definition** (`TestByDefinition`): every name the engine accepts against the reference module
   (``sampling_definitions``, one dense contraction per variable), with a random weight and data, over the jet
   matrix of ``test_sampling_definitions.py`` (no jets; a line at ``(1,)`` and ``(3,)``; two directions at
   ``(1, 2)``; modes of input jet sizes ``(3, 2)``, ``(1, 1)``, ``(2, 3)`` at ``(1, 2)``; a curve of three slots
   at ``(2,)``; modes of sizes ``(1,)``, ``(4,)``, ``(2,)`` at ``(2,)``) and on three problems, one with end bonds
   that differ between the ``L`` and the ``R`` cores.
2. **The declared jet sizes** (`TestDeclaredJetSizes`): every variable the engine returns has the shape of the
   module's table, the leaves the input's jet sizes clipped to ``F``; every ``jet_contract`` call of a step is
   given the jet shape of its output from the table (the calls recorded through ``contractions.jet_contract``).
3. **The direction stack** (`TestTheDirectionStack`): the dot variables, ``Jp``, ``Jp_norm_sq`` and ``JtJp`` with
   ``K = (2,)``, element by element against the reference; the structural check that every ``V[i]`` and ``H[i]``
   shares the ``K + C`` that ``V[0]`` gives, and that a held ``Jp`` has the call's ``K``.
   **The generic twin** (`TestAgainstTheGenericTwin`): the plain and tilde variables against
   ``_generic_twin.py``, which shares no code with the engine and forms ``V_tilde`` from the full ``eta_tilde``.
   **The short form** (`TestTheShortForm`): the private node ``_eta_tilde_top`` is the slice of the full
   ``eta_tilde``, to rounding; ``g`` does not depend on wanting ``eta_tilde``.
4. **The result dtype** (`TestResultDtype`): the lowest floating precision among the arrays the call reads, in its
   simplest cases (the matrix is P10's).
5. **The safety mode** (`TestNoSafetyMode`): the numbers and the errors are the same in safe mode, in
   ``safety.unsafe()`` and under ``jax.jit``.

Numerical comparisons are to a tolerance, structural ones exact. numpy only, by the house convention, except the
jit case of 5; the engine under jit is in ``test_dispatch.py`` too.
"""
import ast
import contextlib
import os
import pathlib
import sys
import unittest
import unittest.mock

import numpy as np

import t3toolbox.safety as safety
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.jets as jets
import t3toolbox.backend.sampling_definitions as sdef
import t3toolbox.backend.sampling_steps as sampling_steps
import t3toolbox.backend.sampling_variables as sv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampling_definitions as defs        # the problems and the jet matrix of the reference's tests
import _generic_twin as generic_twin            # the prototype's one program over a graded algebra: test-only

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

TOL = 1e-10
NAMES = ('w', 'xi', 'mu', 'nu', 'nu_L', 'eta', 'eta_D', 'z', 'r', 'f', 'r_block_sumsq',
         'w_tilde', 'xi_tilde', 'mu_tilde', 'nu_tilde', 'eta_tilde', 'V_tilde', 'H_tilde', 'g',
         'xi_U_dot', 'mu_R_dot', 'nu_L_dot', 'eta_D_dot', 'Jp', 'Jp_norm_sq', 'JtJp',
         'xi_dot', 'mu_dot', 'nu_L_tilde', 'eta_D_tilde', 'xi_tilde_dot', 'mu_tilde_dot', 'nu_L_tilde_dot',
         'eta_D_tilde_dot', 'V_tilde_dot', 'H_tilde_dot', 'Kp')
DOT_NAMES = ('xi_U_dot', 'mu_R_dot', 'nu_L_dot', 'eta_D_dot', 'Jp', 'Jp_norm_sq', 'JtJp',
             'xi_dot', 'mu_dot', 'xi_tilde_dot', 'mu_tilde_dot', 'nu_L_tilde_dot', 'eta_D_tilde_dot', 'V_tilde_dot',
             'H_tilde_dot', 'Kp')


def tails_one_axis():
    """End bonds larger than one, and different between the L and the R cores at both ends; one sample axis."""
    return defs.Problem(5, (3, 4, 3), (2, 2, 3), (3, 2, 2), (2, 2, 3, 2), (3, 3, 2, 4), (3,))


PROBLEMS = (('small', defs.small), ('tails, L and R ends differ', tails_one_axis), ('one site', defs.one_site))


def engine(problem, case, names, **extra):
    """The engine on the problem's frame and the case's input jets, with data, a weight and both directions."""
    y, weight = problem.data('probe', case)
    arguments = dict(order=case.order, y=y, weight=weight, variations=(problem.V, problem.H),
                     core_directions=(problem.D_dot, problem.L_dot))
    arguments.update(extra)
    return sv.sampling_variables('probe', problem.frame, problem.input_jets(case), names, **arguments)


def reference(problem, case, name, variations=None, core_directions=None):
    y, _ = problem.data('probe', case)
    block = problem.block_weight('probe', case)             # the reference takes the blocks themselves
    variations = (problem.V, problem.H) if variations is None else variations
    core_directions = (problem.D_dot, problem.L_dot) if core_directions is None else core_directions
    return getattr(sdef, name)('probe', problem.frame, problem.input_jets(case), None, case.order, y=y,
                               block_weight=block, variations=variations, core_directions=core_directions)


def stacked_directions(problem, seed=0):
    """A stack of two directions in the frame's form, K = (2,): the problem's own, and a second one."""
    rng = np.random.default_rng(seed)
    return ([np.stack([V, rng.standard_normal(V.shape)]) for V in problem.V],
            [np.stack([H, rng.standard_normal(H.shape)]) for H in problem.H])


def stacked_core_directions(problem, seed=1):
    """A stack of two directions in the raw form, K = (2,): the problem's own, and a second one."""
    rng = np.random.default_rng(seed)
    return ([np.stack([B, rng.standard_normal(B.shape)]) for B in problem.D_dot],
            [np.stack([G, rng.standard_normal(G.shape)]) for G in problem.L_dot])


def direction(variations, k):
    """Direction ``k`` of a stack of directions."""
    return [V[k] for V in variations[0]], [H[k] for H in variations[1]]


def element(name, value, k, lead):
    """Element ``k`` of the stack of directions of a variable: its ``K`` axis is after the ``lead`` axes (jets and
    ``W``) of an indexed variable, first for ``Jp_norm_sq`` and the blocks of ``JtJp``."""
    if name == 'Jp_norm_sq':
        return np.asarray(value)[k]
    if name in ('JtJp', 'Kp'):
        return tuple(tuple(block[k] for block in family) for family in value)
    if name in ('V_tilde_dot', 'H_tilde_dot'):
        return tuple(block[k] for block in value)
    return tuple(np.take(array, k, axis=lead) for array in value)


def arrays_of(value):
    """The arrays of a value, in order: an array, a tuple over an index, or ``g``'s pair of tuples."""
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in arrays_of(item)]
    return [value]


class TestByDefinition(unittest.TestCase):

    def test_every_name_against_the_reference(self):
        """Every name, every index, against the reference, to ``1e-10`` relative (``max(|expected|, 1)``)."""
        for problem_name, make in PROBLEMS:
            problem = make()
            for case in defs.JET_CASES:
                got = dict(zip(NAMES, engine(problem, case, NAMES)))
                for name in NAMES:
                    expected = reference(problem, case, name)
                    with self.subTest(problem=problem_name, jets=case.label, name=name):
                        if name != 'w':                                 # w: the sample as given, a list here
                            self.assertEqual(type(got[name]) is tuple, isinstance(expected, tuple))
                        self.assertEqual(len(arrays_of(got[name])), len(arrays_of(expected)))
                        for a, b in zip(arrays_of(got[name]), arrays_of(expected)):
                            self.assertLessEqual(defs.relative_error(a, b), TOL)

    def test_w_is_the_sample_as_given(self):
        problem = defs.small()
        sample = problem.input_jets(defs.LINE_3)
        (w,) = sv.sampling_variables('probe', problem.frame, sample, ('w',), order=(3,))
        self.assertIs(w, sample)
        w, z = sv.sampling_variables('probe', problem.frame, sample, ('w', 'z'), order=(3,))   # w read by xi
        self.assertIs(w, sample)
        w32 = [w_i.astype(np.float32) for w_i in sample]
        w, z = sv.sampling_variables('probe', problem.frame, w32, ('w', 'z'), order=(3,))      # not cast
        self.assertIs(w, w32)

    def test_z_through_the_tucker_centers_against_the_reference(self):
        """With ``Vc`` given, ``z`` is ``z_term(eta[i], Vc[i])``: on a frame whose stores and centers agree it is the
        reference's ``z`` (the reference contracts the dense tensor of ``(D, L)``)."""
        frame, Vc = consistent_frame(seed=7)
        rng = np.random.default_rng(8)
        ww = [rng.standard_normal((3, D.shape[-1])) for D in frame[0]]
        pp = [rng.standard_normal((3, D.shape[-1])) for D in frame[0]]
        line = list(jets.affine_jet(ww, pp))
        for order, sample in (((), ww), ((2,), line)):
            (z,) = sv.sampling_variables('probe', frame, sample, ('z',), order=order, Vc=Vc)
            expected = sdef.z('probe', frame, sample, None, order)
            with self.subTest(order=order):
                for a, b in zip(z, expected):
                    self.assertLessEqual(defs.relative_error(a, b), TOL)


def consistent_frame(seed):
    """A frame of a random Tucker tensor train, its stores consistent, and its Tucker centers ``Vc``."""
    import t3toolbox.backend.fv_conversions as fv_conversions
    import t3toolbox.backend.t3_constructors as t3_constructors
    np.random.seed(seed)
    x = t3_constructors.t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    frame, (Vc, Hc) = fv_conversions.t3_orthogonal_representations(x)
    return tuple(tuple(store) for store in frame), tuple(Vc)


class TestTheDirectionStack(unittest.TestCase):
    """The dot variables, ``Jp``, ``Jp_norm_sq`` and ``JtJp`` with a stack of two directions, ``K = (2,)``: element
    ``k`` of each is the reference at direction ``k`` (the reference has no ``K``). And the structural check at entry:
    every ``V[i]`` and ``H[i]`` has the ``K + C`` that ``V[0]`` gives."""

    def test_every_dot_name_with_two_directions(self):
        for problem_name, make in PROBLEMS:
            problem = make()
            variations, core_directions = stacked_directions(problem), stacked_core_directions(problem)
            for case in defs.JET_CASES:
                got = dict(zip(DOT_NAMES, engine(problem, case, DOT_NAMES, variations=variations,
                                                 core_directions=core_directions)))
                lead = case.m + len(problem.W)
                for k in range(2):
                    for name in DOT_NAMES:
                        expected = reference(problem, case, name, variations=direction(variations, k),
                                             core_directions=direction(core_directions, k))
                        with self.subTest(problem=problem_name, jets=case.label, k=k, name=name):
                            for a, b in zip(arrays_of(element(name, got[name], k, lead)), arrays_of(expected)):
                                self.assertLessEqual(defs.relative_error(a, b), TOL)

    def test_the_variations_share_one_stack_of_directions(self):
        """A size-1 ``K`` on one ``H[i]`` would broadcast silently in a contraction; a wrong count, or another
        ``C``, are caught at entry too, naming the operand."""
        problem = defs.small()
        V, H = stacked_directions(problem)
        bad = {
            'a size-1 K on H[1]':      (V, [H[0], H[1][:1], H[2]], r'H\[1\] has shape \(1, 2, 3, 2\)'),
            'no K on V[2]':            ([V[0], V[1], V[2][0]], H, r'V\[2\] has shape \(3, 5\)'),
            'one K axis more on H[0]': (V, [H[0][None]] + H[1:], r'H\[0\] has shape \(1, 2, 1, 2, 3\)'),
            'a C on V[1]':             ([V[0], V[1][:, None], V[2]], H, r'V\[1\] has shape \(2, 1, 2, 3\)'),
        }
        for label, (V_, H_, pattern) in bad.items():
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, 'the directions must share one stack of directions K'):
                    sv.sampling_variables('probe', problem.frame, problem.ww, ('Jp',), variations=(V_, H_))
                with self.assertRaisesRegex(ValueError, pattern):
                    sv.sampling_variables('probe', problem.frame, problem.ww, ('z',), variations=(V_, H_))
        # the two forms of a direction share the one K of the call
        D_dot, L_dot = stacked_core_directions(problem)
        bad = {
            'raw K of another size':  ((V, H), ([B[:1] for B in D_dot], [G[:1] for G in L_dot]),
                                       r'D_dot\[0\] has shape \(1, 2, 4\), where V\[0\] gives K = \(2,\)'),
            'a size-1 K on L_dot[2]': ((V, H), (D_dot, L_dot[:2] + [L_dot[2][:1]]), r'L_dot\[2\] has shape \(1, '),
            'no K on D_dot[0]':       (None, ([D_dot[0][0]] + D_dot[1:], L_dot), r'D_dot\[1\] has shape \(2, 3, 3\), '
                                                                                 r'where D_dot\[0\] gives K = \(\)'),
        }
        for label, (variations, core_directions, pattern) in bad.items():
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, pattern):
                    sv.sampling_variables('probe', problem.frame, problem.ww, ('z',), variations=variations,
                                          core_directions=core_directions)
        # a held Jp must have the call's K (P12: its K is read and compared with the direction's at entry)
        (Jp,) = sv.sampling_variables('probe', problem.frame, problem.ww, ('Jp',), variations=(V, H))
        with self.assertRaisesRegex(ValueError, r'the held Jp has the stack of directions K = \(\), where the '
                                                r'directions give K = \(2,\)'):
            sv.sampling_variables('probe', problem.frame, problem.ww, ('Jp_norm_sq',), variations=(V, H),
                                  held=sv.HeldVariables(Jp=tuple(a[:, 0] for a in Jp)))


class TestAgainstTheGenericTwin(unittest.TestCase):
    """The generic twin (``_generic_twin.py``): the prototype's one message-passing program over a graded algebra,
    sharing no code with the engine. It forms ``V_tilde`` from the FULL ``eta_tilde``, so it checks the engine's
    short form the long way round, per jet axis on the mixed-modes case."""

    NAMES = ('z', 'r', 'w_tilde', 'mu', 'nu', 'eta', 'xi_tilde', 'mu_tilde', 'nu_tilde', 'eta_tilde', 'V_tilde',
             'H_tilde')

    def test_the_twin_and_the_engine_agree(self):
        for problem_name, make in PROBLEMS[:2]:
            problem = make()
            for case in (defs.NO_JETS, defs.LINE_3, defs.PLANE, defs.MIXED_MODES, defs.MIXED_SIZES):
                y, weight = problem.data('probe', case)
                sample = problem.input_jets(case)
                twin = generic_twin.twin(problem.frame, sample, case.order, y, weight)
                got = dict(zip(self.NAMES, engine(problem, case, self.NAMES)))
                for name in self.NAMES:
                    with self.subTest(problem=problem_name, jets=case.label, name=name):
                        for a, b in zip(arrays_of(got[name]), arrays_of(twin[name])):
                            self.assertLessEqual(defs.relative_error(a, b), TOL)
                (xi,) = engine(problem, case, ('xi',))                    # the leaves: the twin lifts them to F
                with self.subTest(problem=problem_name, jets=case.label, name='xi'):
                    for a, b in zip(xi, twin['xi']):
                        self.assertLessEqual(defs.relative_error(a, b[tuple(slice(0, n) for n in a.shape[:case.m])]),
                                             TOL)

    def test_the_twin_imports_nothing_of_the_library(self):
        tree = ast.parse(pathlib.Path(generic_twin.__file__).read_text())
        imported = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(imported, {'itertools', 'numpy'})


class TestTheShortForm(unittest.TestCase):
    """The private node ``_eta_tilde_top`` (the slots of ``eta_tilde`` that ``V_tilde`` reads) equals the slice
    ``eta_tilde[i][c - a + 1 : c + 1]`` of the full one on every jet axis, to rounding; and ``V_tilde`` reads the
    short node whether ``eta_tilde`` is wanted or not: the route is pinned exactly, the value of ``g`` to rounding (the
    maintainer's rule of 2026-11-20: the same value is equal to rounding)."""

    def test_the_short_node_is_the_slice_of_the_full_eta_tilde(self):
        for problem_name, make in PROBLEMS:
            problem = make()
            for case in defs.JET_CASES:
                tops = []
                step = sampling_steps.eta_tilde_top_step
                with unittest.mock.patch.object(sampling_steps, 'eta_tilde_top_step',
                                                side_effect=lambda *a, **k: tops.append(step(*a, **k)) or tops[-1]):
                    (eta_tilde,) = engine(problem, case, ('eta_tilde', 'g'))[:1]
                sample = problem.input_jets(case)
                with self.subTest(problem=problem_name, jets=case.label):
                    self.assertEqual(len(tops), problem.d)
                    for i, (top, full) in enumerate(zip(tops, eta_tilde)):
                        a = tuple(min(s_, f) for s_, f in zip(sample[i].shape[:case.m], case.box))
                        self.assertEqual(top.shape[:case.m], a)
                        sliced = full[tuple(slice(f - a_k, f) for f, a_k in zip(case.box, a))]
                        self.assertLessEqual(defs.relative_error(top, sliced), 1e-13)

    def test_g_does_not_depend_on_wanting_eta_tilde(self):
        problem = tails_one_axis()
        step = sampling_steps.eta_tilde_top_step
        for case in (defs.LINE_3, defs.MIXED_MODES):
            calls = {}
            for want in (('g',), ('g', 'eta_tilde')):
                calls[want] = []
                with unittest.mock.patch.object(sampling_steps, 'eta_tilde_top_step',
                                                side_effect=lambda *a, **k: calls[want].append(1) or step(*a, **k)):
                    calls[want + ('result',)] = engine(problem, case, want)[0]
            with self.subTest(jets=case.label):
                # the route, exactly: the short node is made once per site, wanted eta_tilde or not
                self.assertEqual(len(calls[('g',)]), problem.d)
                self.assertEqual(len(calls[('g', 'eta_tilde')]), problem.d)
                # the value, to rounding
                for a, b in zip(arrays_of(calls[('g', 'result')]), arrays_of(calls[('g', 'eta_tilde', 'result')])):
                    self.assertLessEqual(defs.relative_error(a, b), 1e-13)


class TestDeclaredJetSizes(unittest.TestCase):

    def test_the_shapes_are_those_of_the_table(self):
        """Every indexed variable has, at every index, the shape ``_shape`` gives from the table ``_SHAPES``:
        ``P + W + C + (size,)``, ``P`` of the input's jet sizes clipped to ``F`` for the leaves and of ``F`` otherwise."""
        names = tuple(name for name in sv._SHAPES if name in sv._public(sv._PROBE_GRAPH))
        for problem_name, make in PROBLEMS:
            problem = make()
            for case in defs.JET_CASES:
                sample = problem.input_jets(case)
                setting = sv._read_setting('probe', dict(zip('DULR', problem.frame)), sample, case.order,
                                           (problem.V, problem.H), sv._PROBE_GRAPH, (problem.D_dot, problem.L_dot))
                got = dict(zip(names, engine(problem, case, names)))
                for name in names:
                    with self.subTest(problem=problem_name, jets=case.label, name=name):
                        self.assertEqual(len(got[name]), sv._count(name, setting))
                        for i, array in enumerate(got[name]):
                            self.assertEqual(array.shape, sv._shape(name, i, setting))
                            jets = array.shape[:case.m]
                            if sv._SHAPES[name][1] == 'none':           # a block: C + the core's shape
                                self.assertEqual(array.ndim, 2 if name.startswith('V_') else 3)
                            elif sv._SHAPES[name][1] == 'leaf':         # xi, xi_U_dot
                                self.assertEqual(jets, tuple(min(a, f) for a, f in
                                                             zip(sample[i].shape[:case.m], case.box)))
                            else:
                                self.assertEqual(jets, case.box)
                f, blocks, g, norm_sq, JtJp, Kp = engine(problem, case, ('f', 'r_block_sumsq', 'g', 'Jp_norm_sq', 'JtJp',
                                                                         'Kp'))
                with self.subTest(problem=problem_name, jets=case.label, name='f, r_block_sumsq, g, Jp_norm_sq, JtJp'):
                    self.assertEqual(np.shape(f), ())
                    self.assertEqual(np.shape(norm_sq), ())
                    self.assertEqual(blocks.shape, (problem.d,) + case.box)
                    self.assertEqual(sv._expected('JtJp', setting), (tuple(a.shape for a in JtJp[0]),
                                                                     tuple(a.shape for a in JtJp[1])))
                    self.assertEqual(sv._expected('g', setting), (tuple(a.shape for a in g[0]),
                                                                  tuple(a.shape for a in g[1])))
                    self.assertEqual(sv._expected('Kp', setting), (tuple(a.shape for a in Kp[0]),
                                                                   tuple(a.shape for a in Kp[1])))

    def test_every_jet_contract_call_is_given_the_output_jet_shape_of_the_table(self):
        """The calls are recorded through ``contractions.jet_contract``. The leaf string ``'PWo,Cio->PWCi'`` is
        given the input's jet sizes clipped to ``F``, minus one; every other string the call's jet shape, and its
        result has the full jet sizes ``F``."""
        problem = defs.small()
        for case in defs.JET_CASES:
            sample = problem.input_jets(case)
            calls = []

            def recording(subscripts, *operands, lens, order, over_orders):
                out = jet_contract(subscripts, *operands, lens=lens, order=order, over_orders=over_orders)
                calls.append((subscripts, operands, lens, order, over_orders, out.shape))
                return out

            jet_contract = contractions.jet_contract
            with unittest.mock.patch.object(contractions, 'jet_contract', side_effect=recording):
                engine(problem, case, ('z', 'eta', 'f'))
            with self.subTest(jets=case.label):
                self.assertEqual(len(calls), 7 * problem.d)         # xi, mu, nu, nu_L, eta, eta_D, z: d each
                leaves = [call for call in calls if call[0] == 'PWo,Cio->PWCi']
                self.assertEqual(len(leaves), problem.d)
                for i, (_, operands, lens, order, form, shape) in enumerate(leaves):
                    clipped = tuple(min(a, f) for a, f in zip(sample[i].shape[:case.m], case.box))
                    self.assertEqual(order, tuple(a - 1 for a in clipped))
                    self.assertEqual(shape[:case.m], clipped)
                for subscripts, _, lens, order, form, shape in calls:
                    self.assertEqual((lens.P, lens.K, form), (case.m, 0, 'loop'))
                    if subscripts != 'PWo,Cio->PWCi':
                        self.assertEqual(order, case.order)
                        self.assertEqual(shape[:case.m], case.box)


class TestResultDtype(unittest.TestCase):
    """The rule of the plan's section 2.3, step 4, in its simplest cases (the full matrix is P10's): the result is in
    the lowest floating precision among the arrays the computation reads; an argument nothing wanted reads does not
    count."""

    def test_float32_anywhere_read_gives_float32(self):
        problem = defs.small()
        case = defs.LINE_1
        y, weight = problem.data('probe', case)
        sample = problem.input_jets(case)
        frame32 = tuple(tuple(G.astype(np.float32) for G in store) for store in problem.frame)
        sample32 = [w.astype(np.float32) for w in sample]
        y32 = [y_i.astype(np.float32) for y_i in y]
        for label, frame, sample_, y_, expected in (('all float64', problem.frame, sample, y, np.float64),
                                                    ('float32 frame', frame32, sample, y, np.float32),
                                                    ('float32 sample', problem.frame, sample32, y, np.float32),
                                                    ('float32 data', problem.frame, sample, y32, np.float32)):
            z, f = sv.sampling_variables('probe', frame, sample_, ('z', 'f'), order=case.order, y=y_,
                                         weight=weight)
            with self.subTest(case=label):
                self.assertEqual({z_i.dtype for z_i in z}, {np.dtype(expected)})   # one dtype for the call
                self.assertEqual(f.dtype, np.dtype(expected))
        (z,) = sv.sampling_variables('probe', problem.frame, sample, ('z',), order=case.order, y=y32)
        self.assertEqual({z_i.dtype for z_i in z}, {np.dtype(np.float64)})     # y is not read for z


    def test_a_held_variable_read_or_wanted_counts_and_is_cast(self):
        """Every output of a call is in the result dtype; the counted inputs include every held variable that a row
        reads or that is wanted. A float64 held ``mu``, wanted and read, among float32 inputs comes back float32, and
        the rows that read it use the cast value; a float64 held ``mu`` neither read nor wanted changes nothing; a
        float32 held ``mu`` that is only wanted makes a float64 call float32."""
        problem = defs.small()
        frame32 = tuple(tuple(G.astype(np.float32) for G in store) for store in problem.frame)
        ww32 = [w.astype(np.float32) for w in problem.ww]
        (mu64,) = sv.sampling_variables('probe', problem.frame, problem.ww, ('mu',))
        mu32 = tuple(m.astype(np.float32) for m in mu64)
        mu, eta = sv.sampling_variables('probe', frame32, ww32, ('mu', 'eta'), held=sv.HeldVariables(mu=mu64))
        self.assertEqual({a.dtype for a in mu + eta}, {np.dtype(np.float32)})
        for a, b in zip(mu, mu32):
            self.assertTrue(np.array_equal(a, b))
        (eta_cast,) = sv.sampling_variables('probe', frame32, ww32, ('eta',), held=sv.HeldVariables(mu=mu32))
        for a, b in zip(eta, eta_cast):
            self.assertTrue(np.array_equal(a, b))                       # the rows read the cast value
        (nu,) = sv.sampling_variables('probe', frame32, ww32, ('nu',), held=sv.HeldVariables(mu=mu64))
        self.assertEqual({a.dtype for a in nu}, {np.dtype(np.float32)})   # neither read nor wanted: not counted
        (nu,) = sv.sampling_variables('probe', problem.frame, problem.ww, ('nu',), held=sv.HeldVariables(mu=mu32))
        self.assertEqual({a.dtype for a in nu}, {np.dtype(np.float64)})
        nu, mu = sv.sampling_variables('probe', problem.frame, problem.ww, ('nu', 'mu'),
                                       held=sv.HeldVariables(mu=mu32))
        self.assertEqual({a.dtype for a in nu + mu}, {np.dtype(np.float32)})   # wanted: an output, counted


class TestNoSafetyMode(unittest.TestCase):

    def test_the_same_numbers_and_errors_in_safe_and_unsafe_mode(self):
        problem = defs.small()
        case = defs.PLANE
        safe = engine(problem, case, NAMES[1:])
        with safety.unsafe():
            unsafe = engine(problem, case, NAMES[1:])
        for name, a, b in zip(NAMES[1:], safe, unsafe):
            with self.subTest(name=name):
                for x, y in zip(arrays_of(a), arrays_of(b)):
                    self.assertTrue(np.array_equal(x, y))
        for mode in ('safe', 'unsafe'):
            with self.subTest(mode=mode):
                with (safety.unsafe() if mode == 'unsafe' else contextlib.nullcontext()):
                    with self.assertRaisesRegex(ValueError, "'r' needs the argument y"):
                        sv.sampling_variables('probe', problem.frame, problem.ww, ('f',))
                    with self.assertRaisesRegex(ValueError, "the held mu must be a tuple of 4; got"):
                        sv.sampling_variables('probe', problem.frame, problem.ww, ('eta',),
                                              held=sv.HeldVariables(mu=problem.ww))

    @unittest.skipUnless(HAS_JAX, 'jax not available')
    def test_the_same_numbers_under_jit(self):
        """Under ``jax.jit`` the numbers are the eager numpy ones, to the precision of jax's float (float32 unless
        x64 is on), and no check is skipped or added: a missing argument raises while the function is traced."""
        problem = defs.small()
        case = defs.LINE_3
        y, weight = problem.data('probe', case)
        sample = problem.input_jets(case)
        names = NAMES[1:]
        variations, core_directions = stacked_directions(problem), stacked_core_directions(problem)
        eager = sv.sampling_variables('probe', problem.frame, sample, names, order=case.order, y=y, weight=weight,
                                      variations=variations, core_directions=core_directions)
        jitted = jax.jit(lambda frame, sample, y, weight, variations, core_directions: sv.sampling_variables(
            'probe', frame, sample, names, order=case.order, y=y, weight=weight, variations=variations,
            core_directions=core_directions))(problem.frame, sample, y, weight, variations, core_directions)
        tol = 1e-12 if jnp.zeros(()).dtype == jnp.float64 else 1e-5
        for name, a, b in zip(names, eager, jitted):
            with self.subTest(name=name):
                for x, j in zip(arrays_of(a), arrays_of(b)):
                    self.assertTrue(isinstance(j, jax.Array))
                    self.assertLessEqual(defs.relative_error(np.asarray(j), x), tol)
        with self.assertRaisesRegex(ValueError, "'r' needs the argument y"):
            jax.jit(lambda frame, sample: sv.sampling_variables('probe', frame, sample, ('f',), order=case.order))(
                problem.frame, sample)


if __name__ == '__main__':
    unittest.main()
