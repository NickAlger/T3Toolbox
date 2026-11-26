# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_variables`` for the entries kind: the apply kind at the unit vectors ``e_index``, no jets.

1. **By definition** (`TestByDefinition`): every name an entries call accepts against the reference module, on a
   manifold frame and on a corewise point ``(B, G)`` (through ``SAMPLE_VARIABLES.md`` section 9), with ``K = ()`` and,
   element by element, ``K = (2,)``, on three problems; their grid points repeat an index in some mode.
2. **Against apply** (`TestAgainstApply`): every name, on both call types, equals the apply kind's at
   ``index_to_one_hot(index, shape)``, to tolerance (a gather and a scatter against products).
3. **The scatter** (`TestTheScatter`): ``V_scatter_step`` against the one-hot product with duplicate indices in a mode
   (they add up); with ``keep_W`` one block per sample, which sum to the summed block.
4. **The index** (`TestTheIndex`): it is neither counted for the dtype nor cast; ``'w'`` returns it as given.
5. **The identities** (`TestIdentities`): per element of a stack of two directions, ``<Jp, omega^2 r> = <p, g>``,
   ``<p, JtJp> = Jp_norm_sq`` and ``JtJp`` equal to ``g`` with ``Jp`` held as ``r``, on both call types.
6. **Each sweep once** (`TestEachSweepOnce`): the step counts of ``('z', 'g')`` and ``('Kp',)``, derived by hand.

numpy only.
"""
import os
import sys
import unittest

import numpy as np

import t3toolbox.backend.jets as jets
import t3toolbox.backend.sampling_definitions as sdef
import t3toolbox.backend.sampling_steps as sampling_steps
import t3toolbox.backend.sampling_variables as sv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampling_apply as tsa
import test_sampling_corewise as tsc
import test_sampling_definitions as defs
import test_sampling_graph as tsg
import test_sampling_variables as tsv

TOL = 1e-10
CASE = defs.NO_JETS


def names(call):
    return tuple(name for name in sv._names('entries', call) if name != 'w')


def at_of(problem, call):
    return problem.frame if call == '_manifold' else (problem.D, problem.L)


def directions_of(problem, call, stacked):
    variations, core_directions = tsa.directions(problem, stacked)
    if call == '_manifold':
        return dict(variations=variations, core_directions=core_directions)
    return dict(core_directions=core_directions)


class TestByDefinition(unittest.TestCase):

    def test_the_problems_repeat_an_index(self):
        """The scatter's duplicates are exercised by the problems themselves."""
        problem = tsv.tails_one_axis()
        self.assertTrue(any(len(set(row.tolist())) < len(row) for row in problem.index))

    def test_every_name(self):
        for call in ('_manifold', '_corewise'):
            for problem_name, make in tsv.PROBLEMS:
                problem = make()
                y, weight = problem.data('entries', CASE)
                full = problem.full_weight('entries', CASE)         # the reference's weight: of the type of z
                for stacked in (False, True):
                    arguments = directions_of(problem, call, stacked)
                    got = sv.sampling_variables('entries', at_of(problem, call), problem.index, names(call), y=y,
                                                weight=weight, **arguments)
                    lead = len(problem.W)
                    for k in (range(2) if stacked else (None,)):
                        pick = (lambda pair: tsv.direction(pair, k)) if stacked else (lambda pair: pair)
                        c = pick(arguments['core_directions'])
                        v = pick(arguments['variations']) if call == '_manifold' else c
                        for name, value in zip(names(call), got):
                            if call == '_manifold':
                                expected = getattr(sdef, name)('entries', problem.frame, problem.index, None, (), y=y,
                                                               weight=full, variations=v, core_directions=c)
                            else:
                                B, G = problem.D, problem.L
                                expected = getattr(sdef, tsc.SECTION_9.get(name, name))(
                                    'entries', (B, G, G, G), problem.index, None, (), y=y, weight=full,
                                    variations=c, core_directions=c)
                            value = tsa.element(name, value, k, lead) if stacked else value
                            with self.subTest(call=call, problem=problem_name, stacked=stacked, k=k, name=name):
                                self.assertEqual(len(tsv.arrays_of(value)), len(tsv.arrays_of(expected)))
                                for a, b in zip(tsv.arrays_of(value), tsv.arrays_of(expected)):
                                    self.assertLessEqual(defs.relative_error(a, b), TOL)


class TestAgainstApply(unittest.TestCase):

    def test_entries_is_apply_at_the_unit_vectors(self):
        for call in ('_manifold', '_corewise'):
            for problem_name, make in tsv.PROBLEMS:
                problem = make()
                y, weight = problem.data('entries', CASE)
                arguments = directions_of(problem, call, True)
                at = at_of(problem, call)
                got = sv.sampling_variables('entries', at, problem.index, names(call), y=y, weight=weight, **arguments)
                one_hot = jets.index_to_one_hot(problem.index, problem.shape)
                expected = sv.sampling_variables('apply', at, one_hot, names(call), y=y, weight=weight, **arguments)
                for name, a, b in zip(names(call), got, expected):
                    with self.subTest(call=call, problem=problem_name, name=name):
                        for x, z in zip(tsv.arrays_of(a), tsv.arrays_of(b)):
                            self.assertLessEqual(defs.relative_error(x, z), 1e-12)


class TestTheScatter(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        self.index = np.array([[1, 1, 3, 1], [0, 2, 2, 0]])               # W = (2, 4): 1 thrice, then 2, 0 twice
        self.eta = rng.standard_normal((2, 4, 2, 3, 5))                   # W + K + C + (n,), K = (2,), C = (3,)
        self.D = rng.standard_normal((3, 5, 6))                           # C + (n, N), N = 6
        self.lens = sv.contractions.GroupLengths(P=0, W=2, K=1, C=1)

    def test_duplicates_add_up(self):
        one_hot = np.eye(6)[self.index]                                   # W + (N,)
        expected = np.einsum('wvkcn,wvo->kcno', self.eta, one_hot)
        got = sampling_steps.V_scatter_step(self.eta, self.index, self.D, lens=self.lens, order=(), form='loop')
        self.assertEqual(got.shape, (2, 3, 5, 6))
        self.assertLessEqual(defs.relative_error(got, expected), 1e-14)
        last = np.zeros((2, 3, 5, 6))                                     # what a plain += would give: one of each
        for w in np.ndindex(2, 4):
            last[..., self.index[w]] = self.eta[w]
        self.assertGreater(defs.relative_error(last, expected), 0.1)

    def test_the_mode_size_is_that_of_D(self):
        """The block's last axis is ``D``'s mode size ``N``, read off ``D`` even where the Tucker rank ``n`` is larger."""
        rng = np.random.default_rng(8)
        eta, index, D = rng.standard_normal((4, 5)), np.array([1, 0, 1, 1]), rng.standard_normal((5, 2))
        lens = sv.contractions.GroupLengths(P=0, W=1, K=0, C=0)
        got = sampling_steps.V_scatter_step(eta, index, D, lens=lens, order=(), form='loop')
        self.assertEqual(got.shape, (5, 2))
        self.assertLessEqual(defs.relative_error(got, np.einsum('wn,wo->no', eta, np.eye(2)[index])), 1e-14)

    def test_keep_W_is_one_block_per_sample(self):
        kept = sampling_steps.V_scatter_step(self.eta, self.index, self.D, lens=self.lens, order=(), form='loop',
                                             keep_W=True)
        self.assertEqual(kept.shape, (2, 4, 2, 3, 5, 6))
        expected = np.einsum('wvkcn,wvo->wvkcno', self.eta, np.eye(6)[self.index])
        self.assertTrue(np.array_equal(kept, expected))
        summed = sampling_steps.V_scatter_step(self.eta, self.index, self.D, lens=self.lens, order=(), form='loop')
        self.assertLessEqual(defs.relative_error(kept.sum(axis=(0, 1)), summed), 1e-14)

    def test_the_gather(self):
        B = self.D[None] + np.zeros((2, 1, 1, 1))                         # K + C + (n, N)
        got = sampling_steps.xi_gather_step(B, self.index[0], lens=self.lens, order=(), form='loop')
        self.assertEqual(got.shape, (4, 2, 3, 5))                         # W + K + C + (n,), W = (4,)
        for w in range(4):
            self.assertTrue(np.array_equal(got[w], B[..., self.index[0, w]]))


class TestTheIndex(unittest.TestCase):

    def test_the_index_is_neither_counted_nor_cast(self):
        """A float32 frame and an int64 index give float32; a float64 frame and an int32 index float64."""
        problem = defs.small()
        for frame_dtype, index_dtype in ((np.float32, np.int64), (np.float64, np.int32)):
            frame = tuple(tuple(core.astype(frame_dtype) for core in store) for store in problem.frame)
            index = problem.index.astype(index_dtype)
            z, xi = sv.sampling_variables('entries', frame, index, ('z', 'xi'))
            with self.subTest(frame=np.dtype(frame_dtype).name, index=np.dtype(index_dtype).name):
                self.assertEqual((z.dtype, xi[0].dtype), (np.dtype(frame_dtype),) * 2)

    def test_w_is_the_one_hot_vectors(self):
        """``'w'`` on an entries call is its definition, the one-hot vectors ``e_index`` (``index_to_one_hot``), in
        the result dtype, also when it is the only name wanted, on both call types; it is a row of the graph, run only
        when wanted; a numpy index out of range raises the gather's ``IndexError``. On probe and apply ``'w'`` is the
        sample as given."""
        problem = defs.small()
        expected = jets.index_to_one_hot(problem.index, problem.shape)
        for call in ('_manifold', '_corewise'):
            for frame_dtype in (np.float32, np.float64):
                at = tuple(tuple(core.astype(frame_dtype) for core in store) for store in at_of(problem, call))
                for want in (('w',), ('w', 'z')):
                    got = sv.sampling_variables('entries', at, problem.index, want)[0]
                    with self.subTest(call=call, frame=np.dtype(frame_dtype).name, want=want):
                        self.assertEqual(len(got), problem.d)
                        for a, b in zip(got, expected):
                            self.assertEqual(a.dtype, np.dtype(frame_dtype))
                            self.assertTrue(np.array_equal(a, b.astype(frame_dtype)))
        with tsg.CountingSteps() as steps:
            sv.sampling_variables('entries', problem.frame, problem.index, ('z',))
        self.assertNotIn('w_one_hot_step', steps.counts())
        with tsg.CountingSteps() as steps:
            sv.sampling_variables('entries', problem.frame, problem.index, ('w',))
        self.assertEqual(steps.counts(), {'w_one_hot_step': 1})
        bad = problem.index.copy()
        bad[1, 0] = problem.shape[1]
        with self.assertRaises(IndexError):
            sv.sampling_variables('entries', problem.frame, bad, ('w',))
        ww = problem.ww
        self.assertIs(sv.sampling_variables('apply', problem.frame, ww, ('w',))[0], ww)

    def test_a_zero_name_raises(self):
        problem = defs.small()
        with self.assertRaisesRegex(ValueError, r"unknown variable 'xi_tilde' for the entries kind on a corewise "
                                                r"call \('xi_tilde' is zero for the entries kind"):
            sv.sampling_variables('entries', (problem.D, problem.L), problem.index, ('xi_tilde',))


class TestIdentities(unittest.TestCase):

    def test_the_direction_per_element(self):
        for call in ('_manifold', '_corewise'):
            for problem_name, make in tsv.PROBLEMS:
                problem = make()
                y, weight = problem.data('entries', CASE)
                arguments = directions_of(problem, call, True)
                p_stack = arguments['variations'] if call == '_manifold' else arguments['core_directions']
                at = at_of(problem, call)
                Jp, JtJp, norm_sq, r, g = sv.sampling_variables(
                    'entries', at, problem.index, ('Jp', 'JtJp', 'Jp_norm_sq', 'r', 'g'), y=y, weight=weight,
                    **arguments)
                lead = len(problem.W)
                pair = lambda blocks, p: sum(float(np.sum(a * b)) for a, b in zip(blocks[0] + blocks[1],
                                                                                   tuple(p[0]) + tuple(p[1])))
                for k in range(2):
                    p = tsv.direction(p_stack, k)
                    Jp_k, JtJp_k = np.take(Jp, k, axis=lead), tsv.element('JtJp', JtJp, k, lead)
                    with self.subTest(call=call, problem=problem_name, k=k):
                        adjoint = float(np.sum(weight ** 2 * Jp_k * r))
                        self.assertLessEqual(abs(adjoint - pair(g, p)), 1e-12 * max(abs(adjoint), 1.0))
                        self.assertLessEqual(abs(pair(JtJp_k, p) - float(norm_sq[k])),
                                             1e-12 * max(abs(float(norm_sq[k])), 1.0))
                        (g_of_Jp,) = sv.sampling_variables('entries', at, problem.index, ('g',), weight=weight,
                                                           held=sv.HeldVariables(r=Jp_k))
                        for a, b in zip(tsv.arrays_of(g_of_Jp), tsv.arrays_of(JtJp_k)):
                            self.assertLessEqual(defs.relative_error(a, b), 1e-12)


class TestEachSweepOnce(unittest.TestCase):

    def test_the_step_counts(self):
        """Derived by hand from the entries graph: the leaves are gathers, the blocks of a Tucker core's shape
        scatters, and the trunk's tilde is the full ``eta_step`` (no short node at ``order = ()``)."""
        problem = defs.small()
        d = problem.d
        seed = {'xi_gather_step': d, 'mu_step': d, 'close_step': 1, 'r_step': 1, 'w_tilde_step': 1}
        expected = {
            ('_manifold', ('z', 'g')): dict(seed, nu_step=d, eta_step=d, V_scatter_step=d, H_readout_step=d, g_step=1),
            ('_corewise', ('z', 'g')): dict(seed, nu_step=d, eta_step=d, V_scatter_step=d, H_readout_step=d, g_step=1),
            ('_manifold', ('Kp',)): dict(seed, xi_gather_step=2 * d, nu_step=d, mu_dot_step=d, nu_dot_step=d,
                                         eta_dot_step=d, V_scatter_step=d, H_tilde_step=d, g_step=1),
        }
        y, weight = problem.data('entries', CASE)
        for (call, want), counts in expected.items():
            with self.subTest(call=call, want=want):
                with tsg.CountingSteps() as steps:
                    sv.sampling_variables('entries', at_of(problem, call), problem.index, want, y=y, weight=weight,
                                          core_directions=(problem.D_dot, problem.L_dot))
                self.assertEqual(steps.counts(), counts)


if __name__ == '__main__':
    unittest.main()
