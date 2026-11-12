# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Tests for the padded start: its working-rank rule and its optimizer seam (the aligned-padded-frame plan's T19 and
T14).

T19: ``ranks.oversampled_ranks``, the one home of the working-rank rule ("``+p`` on every Tucker rank and interior
bond, structurally minimal, never below the input"), on integer sequences (Python-int tuples out) and on
per-stack-element ``(d,)+C`` / ``(d+1,)+C`` rank arrays (each element its own sequence call).

T14, the seam (both layers unless stated):

- **(a)** ``require_frame_structure`` through ``Problem.local_model(frame=)``: one rejection per family with every
  other byte identical (each ragged store's shapes; each uniform variation-mask family; each boundary entry;
  ``shape``; a padded width); it accepts ``geom.frame(x)`` and the aligned frame; a geometry without the member
  raises.
- **(b)** ``newton_cg(padded_from=)`` is ``newton_cg(start_frame=padded_start(...)[0])``.
- **(c)** The G-bit: without a new keyword the ``stats`` keys and history keys are unchanged; the new keys only with
  the new keywords. (The before/after equality of a seeded run is recorded in the N7 handoff.)
- **(d)** ``'cone'``: deterministic; independent of ``x0``'s values; ``cone_max_passes=0`` equals ``'zero'``;
  ``padded_max_passes=0`` starts from the hint-free padded frame with no projection; zero passes with an unread
  option raises (newton_cg, gradient_descent, padded_start, one option at a time; the seed allowed); the structural
  errors (both keywords, a group's option without its group, ``cone_*`` with ``g0norm_newton``, corewise, a
  geometry without the member, a vanishing explicit cone); ``'cone'`` with ``g0norm_newton`` runs no loop;
  ``gradient_descent``'s cone.
- **(e)** ``padded_start`` on ``probe_derivatives`` at d = 4: the uniform loop against the ragged one, pass by pass.
- **(f)** A shared problem, both layers: one pass against the dense oracle with the TIED tangent projection, from
  each layer's own old frame and start (they differ in span at a shared point: a basis choice at singular-value
  ties).
- **(g1)** At a full-rank point and a gauge-rotated frame of it, the exact model passes the FD criterion and matches
  the default frame's quadratic; **(g2)** exact Newton from a padded start is finite and descends.
- **(h)** A rank-0 and a partially-zero-rank ``padded_from`` through the members give the empty frame's start; a
  stack mixing zero and nonzero old points raises.

(T14 (i), the uniform ``use_jit`` paths, is in ``tests/test_dispatch.py``, where x64 is on.) numpy-only.
"""
import itertools
import os
import sys
import unittest
import unittest.mock
import warnings

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.shared_geometry as sgm
import t3toolbox.optimizers as topt
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.padded_frames as padded_frames
import t3toolbox.backend.uniform_padded_frames as uniform_padded_frames
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.ufv_conversions as ufc
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.sharing as sharing_module
import t3toolbox.corewise as cw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _padded_frame_oracles as oracles
import _uniform_padded_frame_testing as testing


def random_minimal_structure(rng, d, sharing=None):
    """A random structurally minimal (shape, tucker_ranks, tt_ranks), shared-minimal under ``sharing``."""
    shape = tuple(int(N) for N in rng.integers(1, 8, size=d))
    if sharing is not None:                                   # a group shares one mode size
        first = {}
        shape = tuple(first.setdefault(g, N) for g, N in zip(sharing, shape))
    tucker = [int(n) for n in rng.integers(1, 7, size=d)]
    if sharing is not None:
        first = {}
        tucker = [first.setdefault(g, n) for g, n in zip(sharing, tucker)]
    tt = (1,) + tuple(int(r) for r in rng.integers(1, 7, size=d - 1)) + (1,)
    return (shape,) + ranks.compute_minimal_ranks(shape, tuple(tucker), tt, sharing=sharing)


class TestOversampledRanks(unittest.TestCase):

    def test_p0_returns_a_minimal_input_unchanged(self):
        rng = np.random.default_rng(0)
        for trial in range(200):
            d = int(rng.integers(1, 6))
            shape, tucker, tt = random_minimal_structure(rng, d)
            with self.subTest(trial=trial, shape=shape, tucker=tucker, tt=tt):
                self.assertEqual(ranks.oversampled_ranks(shape, tucker, tt, 0), (tucker, tt))

    def test_minimal_input_gives_the_reduced_plus_p_ranks(self):
        """For a minimal input the maximum is a no-op (the reduction is monotone): the output is
        ``compute_minimal_ranks`` of the ``+p`` ranks, is itself minimal, and is never below the input."""
        rng = np.random.default_rng(1)
        for trial, p in itertools.product(range(100), (1, 2, 5)):
            d = int(rng.integers(1, 6))
            shape, tucker, tt = random_minimal_structure(rng, d)
            with self.subTest(trial=trial, p=p, shape=shape, tucker=tucker, tt=tt):
                out_tucker, out_tt = ranks.oversampled_ranks(shape, tucker, tt, p)
                plus = (tuple(n + p for n in tucker), (1,) + tuple(r + p for r in tt[1:-1]) + (1,))
                self.assertEqual((out_tucker, out_tt), ranks.compute_minimal_ranks(shape, *plus))
                self.assertEqual(ranks.compute_minimal_ranks(shape, out_tucker, out_tt), (out_tucker, out_tt))
                self.assertTrue(all(a >= b for a, b in zip(out_tucker + out_tt, tucker + tt)))
                self.assertTrue(all(type(r) is int for r in out_tucker + out_tt))

    def test_hand_worked_values(self):
        cases = [
            # (shape, tucker, tt, p) -> expected; the doctest's cases plus the saturated / d = 1 corners
            (((3, 10, 4), (2, 3, 2), (1, 2, 2, 1), 1), ((3, 4, 3), (1, 3, 3, 1))),
            (((3, 10, 4), (2, 3, 2), (1, 2, 2, 1), 5), ((3, 8, 4), (1, 3, 4, 1))),
            (((3, 3), (3, 3), (1, 3, 1), 4), ((3, 3), (1, 3, 1))),           # saturated: nothing to add
            (((5,), (1,), (1, 1), 3), ((1,), (1, 1))),                       # d = 1: the Tucker rank is capped
            (((5,), (2,), (1, 1), 3), ((2,), (1, 1))),                       #   by the bonds' product 1; a
        ]                                                                    #   non-minimal 2 is kept
        for (shape, tucker, tt, p), expected in cases:
            with self.subTest(shape=shape, tucker=tucker, tt=tt, p=p):
                self.assertEqual(ranks.oversampled_ranks(shape, tucker, tt, p), expected)

    def test_never_below_a_non_minimal_input(self):
        """A non-minimal input keeps its slack (elementwise, every store), so the working structure never
        undercuts the target -- including slack the +p reduction would remove."""
        rng = np.random.default_rng(2)
        for trial, p in itertools.product(range(100), (0, 1, 3)):
            d = int(rng.integers(1, 6))
            shape = tuple(int(N) for N in rng.integers(1, 6, size=d))
            tucker = tuple(int(n) for n in rng.integers(1, 9, size=d))
            tt = (1,) + tuple(int(r) for r in rng.integers(1, 9, size=d - 1)) + (1,)
            with self.subTest(trial=trial, p=p, shape=shape, tucker=tucker, tt=tt):
                out_tucker, out_tt = ranks.oversampled_ranks(shape, tucker, tt, p)
                self.assertTrue(all(a >= b for a, b in zip(out_tucker + out_tt, tucker + tt)))
                lo_tucker, lo_tt = ranks.compute_minimal_ranks(shape, tucker, tt)
                self.assertTrue(all(a >= b for a, b in zip(out_tucker + out_tt, lo_tucker + lo_tt)))
        self.assertEqual(ranks.oversampled_ranks((3, 10, 4), (5, 3, 2), (1, 2, 2, 1), 1),
                         ((5, 4, 3), (1, 3, 3, 1)))                          # Tucker 5 > N0 = 3 survives

    def test_sharing_a_group_keeps_one_rank(self):
        rng = np.random.default_rng(3)
        for trial, p in itertools.product(range(100), (0, 1, 4)):
            d = int(rng.integers(2, 6))
            sharing = tuple(int(g) for g in rng.integers(0, max(1, d - 1), size=d))
            shape, tucker, tt = random_minimal_structure(rng, d, sharing)
            with self.subTest(trial=trial, p=p, sharing=sharing, shape=shape, tucker=tucker, tt=tt):
                out_tucker, out_tt = ranks.oversampled_ranks(shape, tucker, tt, p, sharing=sharing)
                for g in set(sharing):
                    self.assertEqual(len({out_tucker[i] for i in range(d) if sharing[i] == g}), 1)
                self.assertEqual(ranks.compute_minimal_ranks(shape, out_tucker, out_tt, sharing=sharing),
                                 (out_tucker, out_tt))                       # shared-minimal
                self.assertTrue(all(a >= b for a, b in zip(out_tucker + out_tt, tucker + tt)))
                if p == 0:
                    self.assertEqual((out_tucker, out_tt), (tucker, tt))
        # the shared notion, not the unshared one: the tied rank may exceed a member's bond product
        self.assertEqual(ranks.oversampled_ranks((8, 3, 8), (2, 1, 2), (1, 1, 1, 1), 1, sharing=('a', 'b', 'a')),
                         ((3, 2, 3), (1, 2, 2, 1)))

    def test_structural_errors(self):
        args = ((3, 10, 4), (2, 3, 2), (1, 2, 2, 1))
        for p in (-1, 1.0, True, None):
            with self.subTest(p=p), self.assertRaisesRegex(ValueError, 'non-negative int'):
                ranks.oversampled_ranks(*args, p)
        with self.assertRaisesRegex(ValueError, 'len\\(tucker_ranks\\)'):
            ranks.oversampled_ranks((3, 10, 4), (2, 3), (1, 2, 2, 1), 1)
        with self.assertRaisesRegex(ValueError, 'len\\(tt_ranks\\)'):
            ranks.oversampled_ranks((3, 10, 4), (2, 3, 2), (1, 2, 1), 1)
        with self.assertRaises(TypeError):                                   # integer tuples in
            ranks.oversampled_ranks((3, 10, 4), (2.0, 3, 2), (1, 2, 2, 1), 1)
        with self.assertRaisesRegex(ValueError, 'equal within a sharing group'):
            ranks.oversampled_ranks((3, 10, 3), (2, 3, 1), (1, 2, 2, 1), 1, sharing=(0, 1, 0))

    def test_arrays_are_per_element_calls(self):
        """T19: (d,)+C / (d+1,)+C integer arrays give HOST integer arrays of the same shapes, each stack element
        equal to the sequence call on that element; with and without sharing."""
        rng = np.random.default_rng(4)
        for trial, C, p in itertools.product(range(20), [(), (3,), (2, 2)], (0, 1, 3)):
            d = int(rng.integers(1, 6))
            sharing = None if trial % 2 else tuple(int(g) for g in rng.integers(0, max(1, d - 1), size=d))
            shape = tuple(int(N) for N in rng.integers(1, 8, size=d))
            tucker = rng.integers(1, 9, size=(d,) + C)
            tt = rng.integers(1, 9, size=(d + 1,) + C)
            tt[0], tt[-1] = 1, 1
            if sharing is not None:                               # a group shares one mode size and one rank
                first = {}
                for i, g in enumerate(sharing):
                    j = first.setdefault(g, i)
                    tucker[i] = tucker[j]
                shape = tuple(shape[first[g]] for g in sharing)
            with self.subTest(trial=trial, C=C, p=p, sharing=sharing, shape=shape):
                out_tucker, out_tt = ranks.oversampled_ranks(shape, tucker, tt, p, sharing=sharing)
                for out, inp in ((out_tucker, tucker), (out_tt, tt)):
                    self.assertIsInstance(out, np.ndarray)
                    self.assertTrue(np.issubdtype(out.dtype, np.integer))
                    self.assertEqual(out.shape, inp.shape)
                for idx in np.ndindex(*C):
                    at = (slice(None),) + idx
                    want = ranks.oversampled_ranks(shape, tuple(int(n) for n in tucker[at]),
                                                   tuple(int(r) for r in tt[at]), p, sharing=sharing)
                    self.assertEqual((tuple(out_tucker[at].tolist()), tuple(out_tt[at].tolist())), want)

    def test_array_structural_errors(self):
        shape = (3, 10, 4)
        tucker = np.array([[2, 2], [3, 5], [2, 1]])
        tt = np.array([[1, 1], [2, 2], [2, 1], [1, 1]])
        ranks.oversampled_ranks(shape, tucker, tt, 1)                       # the good input
        with self.assertRaisesRegex(ValueError, 'share one stack'):
            ranks.oversampled_ranks(shape, tucker, tt[:, :1], 1)
        with self.assertRaisesRegex(ValueError, 'share one stack'):        # a sequence beside a stacked array
            ranks.oversampled_ranks(shape, (2, 3, 2), tt, 1)
        with self.assertRaises(TypeError):                                  # integer arrays in
            ranks.oversampled_ranks(shape, tucker.astype(float), tt, 1)
        with self.assertRaisesRegex(ValueError, 'len\\(tucker_ranks\\)'):
            ranks.oversampled_ranks(shape, tucker[:2], tt, 1)
        with self.assertRaisesRegex(ValueError, 'len\\(tt_ranks\\)'):
            ranks.oversampled_ranks(shape, tucker, tt[:3], 1)
        with self.assertRaisesRegex(ValueError, 'len\\(tt_ranks\\)'):       # d = 1: checked before the boundary
            ranks.oversampled_ranks((7,), np.array([[2, 3]]), np.array([[1, 1]]), 1)   # splice could lengthen it
        with self.assertRaisesRegex(ValueError, 'non-negative int'):
            ranks.oversampled_ranks(shape, tucker, tt, -1)



# ------------------------------------------------------------------------------------------------------------------
# T14: the optimizer seam
# ------------------------------------------------------------------------------------------------------------------
SHAPE, OLD, TARGET = (5, 6, 5, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 4, 4, 3), (1, 3, 4, 3, 1))


def fit_case(layer, shape=SHAPE, old=OLD, target=TARGET, seed=0, kind='probe', sharing=None, M=80):
    """(setup, the old point in the layer's backend form, the frontend x_old, x0): a probe (or probe-derivative)
    fit of a random point at the target ranks, x0 = x_old zero-padded to them. ``layer``: 'ragged' | 'uniform'."""
    np.random.seed(seed)
    ww = [np.random.randn(M, n) for n in shape]
    pp = [np.random.randn(M, n) for n in shape]
    truth = t3.TuckerTensorTrain.randn(shape, *target)
    x_old = t3.TuckerTensorTrain.randn(shape, *old)
    if sharing is not None:
        truth = truth.share(sharing, max_tucker_ranks=target[0], max_tt_ranks=target[1])
        x_old = x_old.share(sharing, max_tucker_ranks=old[0], max_tt_ranks=old[1])
    sample, data = (ww, truth.probe(ww)) if kind == 'probe' else ((ww, pp), truth.probe_derivatives(ww, pp, 1))
    kw = {} if kind == 'probe' else {'order': 1}
    x0 = x_old.resize(shape, *target)
    if layer == 'ragged':
        geometry = t3m.MANIFOLD if sharing is None else sgm.shared(t3m.MANIFOLD, sharing)
        return topt.least_squares_problem(geometry, kind, sample, data, x0, **kw), x_old.data, x_old, x0
    ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)
    geometry = ut3m.UNIFORM_MANIFOLD if sharing is None else sgm.shared(ut3m.UNIFORM_MANIFOLD, sharing)
    setup = topt.least_squares_problem(geometry, kind, sample, data, ux0, **kw)
    return setup, ut3.UniformTuckerTensorTrain.from_t3(x_old, N=ux0.N).data, x_old, x0


def ragged_view(layer, frame):
    """A frame of either layer as a ragged frame (the uniform read-back)."""
    return frame if layer == 'ragged' else testing.ragged(tuple(np.asarray(a) for a in frame[:4]) + tuple(frame[4:]))


def frame_distance(Fa, Fb):
    return max(oracles.distances(oracles.dense_frames(Fa), oracles.dense_frames(Fb)).values())


LAYERS = ('ragged', 'uniform')


class TestFrameStructure(unittest.TestCase):
    """(a)."""

    def test_accepts_the_frame_and_the_aligned_frame(self):
        for layer in LAYERS:
            with self.subTest(layer=layer):
                setup, old, _, _ = fit_case(layer)
                problem, x = setup.problem, setup.x0_cores
                problem.local_model(x, frame=problem.geom.frame(x))
                problem.local_model(x, frame=bopt.padded_start(problem, x, old, max_passes=2)[0])

    def test_ragged_rejections(self):
        setup, _, _, _ = fit_case('ragged')
        problem, x = setup.problem, setup.x0_cores
        F = problem.geom.frame(x)
        for k, store in enumerate(('down', 'up', 'left', 'right')):
            with self.subTest(store=store):
                bad = list(F)
                cores = list(bad[k])
                cores[1] = np.zeros(cores[1].shape[:-1] + (cores[1].shape[-1] + 1,))
                bad[k] = tuple(cores)
                with self.assertRaisesRegex(ValueError, 'the %s cores have shapes' % store):
                    problem.local_model(x, frame=tuple(bad))

    def test_uniform_rejections(self):
        setup, old, _, _ = fit_case('uniform')
        problem, x = setup.problem, setup.x0_cores
        F = problem.geom.frame(x)
        D, U, L, R, shape, masks = F
        for k, store in enumerate(('down', 'up', 'left', 'right')):
            with self.subTest(family=store):
                m = [np.array(a) for a in masks]
                i = 1                                                          # an interior mode / bond
                m[k][i][..., -1] = ~m[k][i][..., -1]
                with self.assertRaisesRegex(ValueError, 'its %s masks differ' % store):
                    problem.local_model(x, frame=(D, U, L, R, shape, tuple(m)))
        for k, (store, bond) in enumerate((('left', -1), ('right', 0))):
            with self.subTest(boundary=store):
                m = [np.array(a) for a in masks]
                m[2 + k][bond][..., -1] = True                                 # a second real slot on the boundary
                with self.assertRaisesRegex(ValueError, 'its %s boundary bond mask' % store):
                    problem.local_model(x, frame=(D, U, L, R, shape, tuple(m)))
        with self.subTest(part='shape'), self.assertRaisesRegex(ValueError, 'shape'):
            problem.local_model(x, frame=(D, U, L, R, tuple(shape[:-1]) + (shape[-1] - 1,), masks))
        with self.subTest(part='a padded width'), self.assertRaisesRegex(ValueError, 'supercore widths'):
            problem.local_model(x, frame=(D, np.concatenate([U, U[..., :1, :]], axis=-2), L, R, shape, masks))

    def test_a_geometry_without_the_member(self):
        setup = topt.least_squares_problem(t3m.COREWISE, 'probe', *fit_case('ragged')[0].problem.sample[:0] or (
            [np.ones((3, n)) for n in SHAPE], [np.ones(3)] * 4), t3.TuckerTensorTrain.randn(SHAPE, *TARGET))
        with self.assertRaisesRegex(ValueError, 'require_frame_structure'):
            setup.problem.local_model(setup.x0_cores, frame=bgeo.ManifoldGeometryOps().frame(setup.x0_cores))


class TestNewtonKeywords(unittest.TestCase):
    """(b), (c), (d)."""

    def test_padded_from_is_the_padded_start(self):
        for layer in LAYERS:
            with self.subTest(layer=layer):
                setup, old, _, _ = fit_case(layer)
                x_a, a = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=4, padded_from=old)
                frame, movements = bopt.padded_start(setup.problem, setup.x0_cores, old)
                x_b, b = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=4, start_frame=frame)
                self.assertEqual(a['losses'], b['losses'])
                self.assertEqual((a['start_frame'], b['start_frame']), ('padded', 'given'))
                self.assertEqual(len(a['start_movements']), int(np.sum(~np.isnan(np.asarray(movements, float)))))
                self.assertNotIn('start_movements', b)

    def test_the_stats_keys(self):
        """The G-bit: the keys of stats and of the history rows without the new keywords; the new keys only with them."""
        setup, old, _, _ = fit_case('ragged')
        _, plain = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=3)
        self.assertEqual(set(plain), {'losses', 'newton_steps', 'newton_accepted', 'stop_reason', 'history'})
        keys = set(plain['history'][0])
        _, padded = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=3, padded_from=old)
        _, cone = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=3, g0_reference='cone')
        self.assertEqual(set(padded) - set(plain), {'start_frame', 'start_movements'})
        self.assertEqual(set(cone) - set(plain), {'cone_movements'})
        self.assertEqual(set(padded['history'][0]), keys)
        self.assertEqual(cone['history'][0]['g0_reference'], 'cone')
        _, gd = bopt.gradient_descent(setup.problem, setup.x0_cores, n_iter=2)
        self.assertEqual(set(gd), {'losses', 'n_iter', 'line_search_failed'})

    def test_the_cone(self):
        """Deterministic; independent of x0's values (its structure only); cone_max_passes=0 is the misfit gradient
        norm at the zero tensor's hint-free start (D-41: no longer 'zero', whose frame has coordinate TT cores);
        gradient_descent's cone is newton_cg's."""
        for layer in LAYERS:
            with self.subTest(layer=layer):
                setup, _, _, _ = fit_case(layer)
                problem, x = setup.problem, setup.x0_cores
                g0 = lambda x_, **kw: bopt.newton_cg(problem, x_, max_newton=1, **kw)[1]['history'][0]['g0norm']
                cone = g0(x, g0_reference='cone')
                self.assertEqual(cone, g0(x, g0_reference='cone'))
                self.assertEqual(cone, g0(cw.corewise_scale(x, -2.0), g0_reference='cone'))
                F0 = bopt.padded_start(problem, x, None, max_passes=0)[0]
                r0 = cw.corewise_sub(problem.kind.point_forward(cw.corewise_zeros_like(x), problem.sample), problem.data)
                g = bopt._misfit_gradient_at(F0, (problem, r0))
                hint_free = float(problem.geom.with_frame(F0).inner(g, g)) ** 0.5
                self.assertLessEqual(abs(g0(x, g0_reference='cone', cone_max_passes=0) - hint_free), 1e-12 * hint_free)
                self.assertGreater(cone, hint_free)                            # the aligned frame sees more gradient
                _, gd = bopt.gradient_descent(problem, x, n_iter=1, g0_reference='cone')
                _, nc = bopt.newton_cg(problem, x, max_newton=1, g0_reference='cone')
                self.assertEqual(gd['cone_movements'], nc['cone_movements'])

    def test_zero_padded_passes_start_from_the_hint_free_frame(self):
        for layer in LAYERS:
            with self.subTest(layer=layer):
                setup, old, x_old, x0 = fit_case(layer)
                seen = []
                with unittest.mock.patch.object(bopt, '_misfit_gradient_at', side_effect=AssertionError('projected')):
                    bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=1, padded_from=old, padded_max_passes=0,
                                   padded_seed=3, callback=lambda info: seen.append(info.lm.frame))
                if layer == 'ragged':
                    want = padded_frames.fv_padded_frame(fvc.t3_orthogonal_representations(old)[0], *TARGET, seed=3)
                else:
                    widths = tuple(m.shape[-1] for m in setup.problem.geom.var_masks)
                    want = uniform_padded_frames.ufv_compact_frame(uniform_padded_frames.ufv_padded_frame(
                        ufc.ut3_orthogonal_representations(old)[0], *TARGET, seed=3)[0], widths)
                got = seen[0]
                for a, b in zip(got[:4], want[:4]):
                    for x_, y_ in (zip(a, b) if layer == 'ragged' else ((a, b),)):
                        self.assertLessEqual(float(np.max(np.abs(np.asarray(x_) - y_), initial=0.0)), 1e-14)

    def test_zero_passes_with_an_unread_option(self):
        setup, old, _, _ = fit_case('ragged')
        problem, x = setup.problem, setup.x0_cores
        for option, value in (('oversampling', 3), ('oversampled_passes', 1), ('tol', 0.1), ('alternate', False)):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, 'max_passes=0'):
                    bopt.padded_start(problem, x, old, max_passes=0, **{option: value})
                with self.assertRaisesRegex(ValueError, 'padded_max_passes=0'):
                    bopt.newton_cg(problem, x, padded_from=old, padded_max_passes=0, **{'padded_' + option: value})
                with self.assertRaisesRegex(ValueError, 'cone_max_passes=0'):
                    bopt.newton_cg(problem, x, g0_reference='cone', cone_max_passes=0, **{'cone_' + option: value})
                with self.assertRaisesRegex(ValueError, 'cone_max_passes=0'):
                    bopt.gradient_descent(problem, x, g0_reference='cone', cone_max_passes=0,
                                          **{'cone_' + option: value})
        bopt.padded_start(problem, x, old, max_passes=0, seed=4)              # the seed is read
        bopt.newton_cg(problem, x, max_newton=1, padded_from=old, padded_max_passes=0, padded_seed=4)
        bopt.newton_cg(problem, x, max_newton=1, g0_reference='cone', cone_max_passes=0, cone_seed=4)

    def test_structural_errors(self):
        setup, old, _, _ = fit_case('ragged')
        problem, x = setup.problem, setup.x0_cores
        frame = problem.geom.frame(x)
        cases = ((dict(start_frame=frame, padded_from=old), 'not both'),
                 (dict(padded_tol=0.1), 'without padded_from'),
                 (dict(cone_tol=0.1), "cone reference is not computed"),
                 (dict(g0_reference='zero', cone_seed=1), "cone reference is not computed"),
                 (dict(g0_reference='cone', g0norm_newton=1.0, cone_tol=0.1), "cone reference is not computed"))
        for kw, message in cases:
            with self.subTest(**{k: repr(v)[:20] for k, v in kw.items()}), self.assertRaisesRegex(ValueError, message):
                bopt.newton_cg(problem, x, max_newton=1, **kw)
        with self.assertRaisesRegex(ValueError, "cone reference is not computed"):
            bopt.gradient_descent(problem, x, n_iter=1, cone_tol=0.1)
        corewise = topt.least_squares_problem(t3m.COREWISE, 'probe', problem.sample, problem.data,
                                              t3.TuckerTensorTrain(*x))
        for kw, message in ((dict(g0_reference='cone'), 'aligned_padded_frame member'),
                            (dict(padded_from=old), 'aligned_padded_frame member'),
                            (dict(start_frame=frame), 'require_frame_structure')):
            with self.subTest(corewise=list(kw)), self.assertRaisesRegex(ValueError, message):
                bopt.newton_cg(corewise.problem, corewise.x0_cores, max_newton=1, **kw)
        zero_data = topt.least_squares_problem(t3m.MANIFOLD, 'probe', problem.sample,
                                               [np.zeros_like(y) for y in problem.data], t3.TuckerTensorTrain(*x))
        with self.assertRaisesRegex(ValueError, 'the cone reference is 0'):
            bopt.newton_cg(zero_data.problem, zero_data.x0_cores, max_newton=1, g0_reference='cone')

    def test_cone_with_g0norm_newton_runs_no_loop(self):
        setup, _, _, _ = fit_case('ragged')
        with unittest.mock.patch.object(bopt, '_misfit_gradient_at', side_effect=AssertionError('the loop ran')):
            _, stats = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=1, g0_reference='cone',
                                      g0norm_newton=2.0)
        self.assertEqual(stats['history'][0]['g0_reference'], 'override')
        self.assertNotIn('cone_movements', stats)


class TestSeamNumerics(unittest.TestCase):
    """(e), (f), (g1), (g2), (h)."""

    def test_probe_derivatives_uniform_against_ragged(self):
        """(e): the uniform kinds at block-layout frames, d = 4, against the ragged loop. The two layers' old frames
        agree up to signs, and the Gaussian seeds (D-41) are not sign-covariant, so the starts and the passes differ;
        the aligned frames agree (the fixed point is the gradient's): spans to 1e-9 at tol 1e-10, to 0.2 at the
        default stop."""
        (rs, rold, _, _), (us, uold, _, _) = (fit_case(layer, kind='probe_derivatives') for layer in LAYERS)
        for tol, bound in ((1e-10, 1e-9), (None, 0.2)):
            with self.subTest(tol=tol):
                kw = {} if tol is None else {'tol': tol, 'max_passes': 300}
                Fr, mr = bopt.padded_start(rs.problem, rs.x0_cores, rold, **kw)
                Fu, mu = bopt.padded_start(us.problem, us.x0_cores, uold, **kw)
                if tol is not None:                                           # both stopped on tol, not the cap
                    self.assertLess(max(len(mr), len(bopt._movement_record(mu))), 300)
                self.assertLessEqual(frame_distance(ragged_view('uniform', Fu), Fr), bound)

    def shared_one_pass(self, layer):
        """One pass (p = 0) of the shared problem's padded start against the dense pass from the SAME layer's old frame
        and hint-free start, with the hint projected onto the TIED tangent space (the shared geometry's projection).
        Each layer's own: at a shared point the Tucker factors are orthonormal (singular values all 1), and the two
        layers' orthogonal representations then choose different bases of the same subspaces, so their hint-free
        starts differ in span."""
        shape, sharing = (5, 6, 6, 5), (0, 1, 1, 0)
        groups = sharing_module.validate_sharing(sharing, shape)
        stores = ranks.compute_orthogonal_representation_ranks(shape, *TARGET)
        setup, old, x_old, x0 = fit_case(layer, shape=shape, sharing=sharing, M=60)
        F, movements = bopt.padded_start(setup.problem, setup.x0_cores, old, oversampling=0, max_passes=1)
        if layer == 'ragged':
            old_frame = fvc.t3_orthogonal_representations(old)[0]
            start = padded_frames.fv_padded_frame(old_frame, *TARGET, sharing=sharing)
        else:
            old_frame = ragged_view(layer, ufc.ut3_orthogonal_representations(old)[0])
            start = ragged_view(layer, bopt.padded_start(setup.problem, setup.x0_cores, old, max_passes=0)[0])
        ragged_problem = fit_case('ragged', shape=shape, sharing=sharing, M=60)[0].problem   # the same fit
        residual = [a - b for a, b in zip(x0.probe(ragged_problem.sample), ragged_problem.data)]
        G = dense_probe_gradient(ragged_problem.sample, residual)
        start_dense = oracles.dense_frames(start)
        dense, move = oracles.dense_pass(oracles.dense_frames(old_frame), start_dense,
                                     dense_tied_project(start_dense, G, groups, x0.to_dense()), stores,
                                     ('D', 'U', 'L', 'R'), groups)
        self.assertLessEqual(max(oracles.distances(oracles.dense_frames(ragged_view(layer, F)), dense).values()), 1e-9)
        self.assertLessEqual(abs(bopt._movement_record(movements)[0] - move), 1e-9)

    def test_shared_one_pass(self):
        """(f), both layers: the tied projection at a padded frame is the dense tied projection."""
        for layer in LAYERS:
            with self.subTest(layer=layer):
                self.shared_one_pass(layer)

    def test_the_exact_model_at_a_rotated_frame(self):
        """(g1): at a full-rank point, a gauge-rotated frame (another core representation of the same tensor)
        passes the FD criterion along its own retraction and matches the default frame's quadratic."""
        np.random.seed(5)
        shape, rk = (4, 5, 6), ((2, 2, 2), (1, 2, 2, 1))
        ww = [np.random.randn(20, n) for n in shape]
        pp = [np.random.randn(20, n) for n in shape]
        data = bfit.ProbeDerivativesKind(order=1).point_forward(t3.TuckerTensorTrain.randn(shape, *rk).data, (ww, pp))
        problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.ProbeDerivativesKind(order=1),
                                             (ww, pp), data)
        x = t3.TuckerTensorTrain.randn(shape, *rk).data
        Q = [np.linalg.qr(np.random.randn(n, n))[0] for n in rk[0]]
        x_rot = (tuple(q.T @ B for q, B in zip(Q, x[0])), tuple(np.einsum('anb,nm->amb', G, q) for G, q in zip(x[1], Q)))
        F = problem.geom.frame(x_rot)
        default, rotated = problem.local_model(x, hessian='exact'), problem.local_model(x, frame=F, hessian='exact')
        import t3toolbox.backend.tv_operations as tvo
        S = np.random.randn(*shape)
        p, p_rot = tvo.tv_project_dense_onto_tangent_space(default.frame, S), tvo.tv_project_dense_onto_tangent_space(F, S)
        q, q_rot = float(default.quadratic(p)), float(rotated.quadratic(p_rot))
        self.assertLessEqual(abs(q - q_rot), 1e-10 * abs(q))
        scale = 1.0 / float(problem.geom.inner(p_rot, p_rot)) ** 0.5
        f = lambda t: float(problem.objective(rotated.retract(cw.corewise_scale(p_rot, t * scale))))
        fd = (f(1e-3) - 2 * f(0.0) + f(-1e-3)) / 1e-6
        self.assertLessEqual(abs(q_rot * scale ** 2 - fd), 1e-4 * abs(fd))

    def test_exact_newton_from_a_padded_start(self):
        """(g2): finite, and the objective decreases."""
        setup, old, _, _ = fit_case('ragged', kind='probe_derivatives')
        frame, _ = bopt.padded_start(setup.problem, setup.x0_cores, old)
        _, stats = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=6, hessian='exact', start_frame=frame)
        self.assertTrue(all(np.isfinite(stats['losses'])))
        self.assertLess(stats['losses'][-1], stats['losses'][0])

    def test_zero_rank_old_points(self):
        """(h): a rank-0 and a partially-zero-rank padded_from (Tucker (0,2,2), TT (1,0,2,1)) through the members:
        the empty frame's start, equal to the zero tensor's; a stack mixing zero and nonzero old points raises."""
        shape, target = (4, 5, 6), ((2, 2, 2), (1, 2, 2, 1))
        np.random.seed(2)
        project = lambda frame, hint: hint                                     # unused: zero passes
        olds = {'rank 0': ((0, 0, 0), (1, 0, 0, 1)), 'partially zero': ((0, 2, 2), (1, 0, 2, 1))}
        x0 = t3.TuckerTensorTrain.zeros(shape, *target)
        ragged_geom = bgeo.ManifoldGeometryOps()
        want = ragged_geom.aligned_padded_frame(x0.data, None, project, None, max_passes=0)[0]
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)
        ugeom = bgeo.UniformManifoldGeometryOps.from_point(ux0.data)
        uwant = ugeom.aligned_padded_frame((ux0.data[0], ux0.data[1]), None, project, None, max_passes=0)[0]
        for name, ranks_ in olds.items():
            with self.subTest(old=name):
                x_old = t3.TuckerTensorTrain.zeros(shape, *ranks_)
                got = ragged_geom.aligned_padded_frame(x0.data, x_old.data, project, None, max_passes=0)[0]
                for a, b in zip(got, want):
                    self.assertTrue(all(np.array_equal(x_, y_) for x_, y_ in zip(a, b)))
                uold = (np.zeros((3, 2, 6)), np.zeros((3, 2, 2, 2)), shape,
                        (np.arange(2) < np.array(ranks_[0])[:, None], np.arange(2) < np.array(ranks_[1])[:, None]))
                ugot = ugeom.aligned_padded_frame((ux0.data[0], ux0.data[1]), uold, project, None, max_passes=0)[0]
                for a, b in zip(ugot[:4], uwant[:4]):
                    self.assertLessEqual(float(np.max(np.abs(a - b), initial=0.0)), 1e-14)
        mixed = (np.zeros((3, 2, 2, 6)), np.zeros((3, 2, 2, 2, 2)), shape,
                 (np.stack([np.arange(2) < np.array(r[0])[:, None] for r in (olds['rank 0'], target)], axis=1),
                  np.stack([np.arange(2) < np.array(r[1])[:, None] for r in (olds['rank 0'], target)], axis=1)))
        with self.assertRaisesRegex(ValueError, 'mixes structurally zero and nonzero'):
            ugeom.aligned_padded_frame((ux0.data[0], ux0.data[1]), mixed, project, None, max_passes=0)


def dense_probe_gradient(ww, residual):
    """The ambient misfit gradient of a probe fit: sum_w sum_m residual[m][w, i_m] prod_(j != m) ww[j][w, i_j]."""
    d = len(ww)
    letters = 'abcdefgh'[:d]
    G = 0.0
    for m in range(d):
        operands = [residual[m]] + [ww[j] for j in range(d) if j != m]
        subs = ['w' + letters[m]] + ['w' + letters[j] for j in range(d) if j != m]
        G = G + np.einsum(','.join(subs) + '->' + letters, *operands)
    return G


def dense_tied_project(dense, S, groups, X):
    """The orthogonal projection of the dense S onto the TIED tangent space at the dense frames D of the point X: the
    TT directions as they are, and per group one shared Tucker variation dU moving every member's factor,
    sum_(i in g) Y_i x_i dU with Y_i = X x_i U_i^T (the untied down-frame Tucker columns are replaced)."""
    import math
    D, phi, psi, U = dense
    N = S.shape
    d = len(N)
    cols, tucker = [], {}
    for i in range(d):
        nx, ny = math.prod(N[:i]), math.prod(N[i + 1:])
        cols.append(np.einsum('xa,in,yb->xiyanb', phi[i], D[i], psi[i + 1]).reshape(math.prod(N), -1))
        Y = np.einsum('xiy,in->xny', X.reshape(nx, N[i], ny), D[i])
        tucker[i] = np.einsum('ij,xny->xiyjn', np.eye(N[i]), Y).reshape(math.prod(N), -1)
    for group in groups:
        cols.append(sum(tucker[i] for i in group))
    Q, s, _ = np.linalg.svd(np.concatenate(cols, axis=1), full_matrices=False)
    Q = Q[:, s > 1e-9 * s[0]]
    return (Q @ (Q.T @ S.ravel())).reshape(S.shape)


if __name__ == '__main__':
    unittest.main()
