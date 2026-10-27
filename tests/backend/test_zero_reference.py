# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""T26: the zero reference (``g0_reference``) and the canonical zero-point frame behind it.

Two halves. The **frame** (``fv_zero_point_frame`` / ``ufv_zero_point_frame``): valid (orthonormal, at the
zero tensor, a working gauge projection) at d = 1, 3, 4 and shared; the recipe's three properties (rank
independence, nesting, a pinned function of ``(seed, k_g, n_g, N_g)``); the same real blocks on both layers;
and no collapse on data with no energy on the leading coordinates. The **resolution** in ``newton_cg`` /
``gradient_descent``: ``None`` means the geometry's own reference, the explicit overrides win (and skip the
computation), a vanishing reference falls back with a warning by default and raises when asked for
explicitly, and the reference -- hence a seeded fit -- is identical on the ragged and uniform layers.
numpy-only (jax dispatch: ``tests/test_dispatch.py``)."""
import unittest
import warnings

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.shared_geometry as sg
import t3toolbox.optimizers as topt
import t3toolbox.corewise as cw
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.ufv_operations as ufo
import t3toolbox.backend.ufv_conversions as ufc
import t3toolbox.backend.ufv_masking as ufm
import t3toolbox.backend.t3_conversions as t3c

# (label, shape, tucker ranks, tt ranks, sharing)
_VALIDITY_CASES = [
    ('d=1',    (6,),         (3,),         (1, 1),          None),
    ('d=3',    (5, 6, 7),    (2, 3, 2),    (1, 2, 2, 1),    None),
    ('d=4',    (6, 6, 6, 6), (3, 3, 3, 3), (1, 3, 4, 3, 1), None),
    ('shared', (6, 6, 5),    (3, 3, 2),    (1, 3, 2, 1),    (0, 0, 1)),
]


def _ragged_geometry(shape, sharing):
    geom = bgeo.ManifoldGeometryOps()
    return geom if sharing is None else geom.with_sharing(sharing, shape)


def _template(shape, tucker_ranks, tt_ranks, sharing=None):
    """A frame with these ranks: the structure a fit at these ranks linearizes with."""
    return _ragged_geometry(shape, sharing).frame(t3.TuckerTensorTrain.zeros(shape, tucker_ranks, tt_ranks).data)


def _uniform_template(shape, tucker_ranks, tt_ranks, sharing=None):
    ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.zeros(shape, tucker_ranks, tt_ranks))
    geom = bgeo.UniformManifoldGeometryOps.from_point(ux.data, sharing)
    return geom, geom.frame((ux.data[0], ux.data[1]))


def _unit_rows(rng, W, shape):
    ww = [rng.standard_normal((W, N)) for N in shape]
    return [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]


def _tied(x, groups):
    """``x`` with each group's Tucker factors replaced by the group's first (an exactly tied point)."""
    tk = list(x.tucker_cores)
    for group in groups:
        for i in group[1:]:
            tk[i] = tk[group[0]]
    return t3.TuckerTensorTrain(tuple(tk), x.tt_cores)


def _max_abs_diff(a, b):
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(a, b))


class TestZeroPointFrame(unittest.TestCase):
    """The canonical frame itself (C.2): validity, the recipe, the two layers, the structural errors."""

    def setUp(self):
        np.random.seed(0)

    def test_valid_frame_at_the_zero_tensor(self):
        """(c): orthonormal, base point exactly zero (both reconstructions), the template's core shapes, a
        gauge projection that is idempotent there -- and, shared, one factor per group (exactly tied)."""
        for label, shape, tk, tt, sharing in _VALIDITY_CASES:
            with self.subTest(case=label):
                geom = _ragged_geometry(shape, sharing)
                template = _template(shape, tk, tt, sharing)
                F0 = geom.zero_point_frame(template)
                self.assertEqual([[c.shape for c in fam] for fam in F0], [[c.shape for c in fam] for fam in template])
                self.assertLessEqual(float(fvo.fv_frame_orthogonality_residual(F0)), 1e-14)
                self.assertEqual(float(np.abs(t3c.t3_to_dense((F0[0], F0[2]))).max()), 0.0)   # (U, L): left
                self.assertEqual(float(np.abs(t3c.t3_to_dense((F0[0], F0[3]))).max()), 0.0)   # (U, R): right
                aux = geom.precompute(F0)
                v = fvo.fv_variations_randn(fvo.fv_variation_shapes(F0))
                p1 = geom.project(F0, v, aux=aux)
                p2 = geom.project(F0, p1, aux=aux)
                scale = float(cw.corewise_norm(p1))
                self.assertGreater(scale, 0.0)
                self.assertLessEqual(float(cw.corewise_norm(cw.corewise_sub(p2, p1))), 1e-12 * scale)
                if sharing is not None:
                    self.assertTrue(np.array_equal(F0[0][0], F0[0][1]))      # the group (0, 1) is tied

    def test_rank_independence(self):
        """(d): each group draws from its own keyed stream, so a mode's factor is bit-identical when ANOTHER
        mode's rank -- or size -- changes."""
        F1 = fvo.fv_zero_point_frame(_template((8, 8, 8), (2, 3, 3), (1, 2, 3, 1)))
        F2 = fvo.fv_zero_point_frame(_template((8, 8, 8), (3, 3, 3), (1, 3, 3, 1)))
        F3 = fvo.fv_zero_point_frame(_template((11, 8, 8), (5, 3, 3), (1, 5, 3, 1)))
        self.assertTrue(np.array_equal(F1[0][2], F2[0][2]))                  # mode 0's rank changed
        self.assertTrue(np.array_equal(F1[0][2], F3[0][2]))                  # mode 0's rank AND size changed
        self.assertTrue(np.array_equal(F1[0][1], F3[0][1]))

    def test_nested_across_levels(self):
        """(d): QR column j depends only on draw rows <= j, so the rank-r factor is the leading r rows of the
        rank-(r+1) factor -- the reference frames nest across a continuation ladder."""
        frames = [fvo.fv_zero_point_frame(_template((8, 8, 8), (r, r, r), (1, r, r, 1))) for r in range(1, 7)]
        for Fa, Fb in zip(frames, frames[1:]):
            for i in range(3):
                r = Fa[0][i].shape[0]
                self.assertLessEqual(float(np.max(np.abs(Fb[0][i][:r] - Fa[0][i]))), 1e-15)

    def test_pinned_draw(self):
        """(d): the factor is a fixed function of (seed, k_g, n_g, N_g) -- a frozen small case (mode 1:
        k = 1, n = 2, N = 4, seed 0). Not bitwise across BLAS builds: the sign-fixed QR is unique, and
        reproducible to ~1e-15. Another seed is another factor."""
        pinned = np.array([[-0.20427947008759073, -0.9621057198447386, 0.04067747018222295, -0.1759767750651908],
                           [-0.3043238880778042, -0.07880558055316438, -0.7210720921240522, 0.6174396242376488]])
        template = _template((3, 4, 3), (2, 2, 2), (1, 2, 2, 1))
        self.assertTrue(np.allclose(fvo.fv_zero_point_frame(template)[0][1], pinned, rtol=0.0, atol=1e-12))
        self.assertEqual(fvo.ZERO_FRAME_SEED, 0)
        self.assertFalse(np.allclose(fvo.fv_zero_point_frame(template, seed=1)[0][1], pinned, atol=1e-3))

    def test_group_key_is_its_first_mode(self):
        """A group draws with the key of its first mode: the tied factor is the one that mode gets unshared."""
        template = _template((6, 6, 5), (3, 3, 2), (1, 3, 2, 1), (0, 0, 1))
        shared = fvo.fv_zero_point_frame(template, ((0, 1), (2,)))[0]
        unshared = fvo.fv_zero_point_frame(template)[0]
        self.assertTrue(np.array_equal(shared[0], unshared[0]))
        self.assertTrue(np.array_equal(shared[1], unshared[0]))
        self.assertFalse(np.array_equal(shared[1], unshared[1]))
        self.assertTrue(np.array_equal(shared[2], unshared[2]))

    def test_uniform_twin_has_the_ragged_real_blocks(self):
        """(b): the uniform frame's real blocks ARE the ragged construction's (so the reference is
        layer-independent); its padding is zero; its masks are the template's."""
        for label, shape, tk, tt, sharing in _VALIDITY_CASES:
            with self.subTest(case=label):
                ugeom, utemplate = _uniform_template(shape, tk, tt, sharing)
                UF0 = ugeom.zero_point_frame(utemplate)
                self.assertIs(UF0[5], utemplate[5])
                self.assertEqual([a.shape for a in UF0[:4]], [a.shape for a in utemplate[:4]])
                ragged = _ragged_geometry(shape, sharing).zero_point_frame(_template(shape, tk, tt, sharing))
                unpacked = ufc.ut3frame_to_t3frame(UF0)
                for fam_u, fam_r in zip(unpacked, ragged):
                    self.assertLessEqual(_max_abs_diff(fam_u, fam_r), 1e-15)
                for sc, masked in zip(UF0[:4], ufm.ufv_apply_frame_masks(UF0)):
                    self.assertTrue(np.array_equal(sc, masked))              # the padding is exactly zero
                self.assertLessEqual(float(ufo.ufv_frame_orthogonality_residual(UF0)), 1e-14)

    def test_structural_errors(self):
        """A stacked template is not supported (the optimizers' unstacked contract), on either layer; a
        sharing group needs one factor shape."""
        stacked = t3m.MANIFOLD.frame(t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1),
                                                                stack_shape=(2,))).data
        with self.assertRaisesRegex(NotImplementedError, '_require_unstacked'):
            fvo.fv_zero_point_frame(stacked)
        ux = ut3.UniformTuckerTensorTrain.from_t3(
            t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1), stack_shape=(2,)))
        uframe = ufc.ut3_orthogonal_representations(ux.data)[0]
        with self.assertRaisesRegex(NotImplementedError, '_require_unstacked'):
            ufo.ufv_zero_point_frame(uframe)
        with self.assertRaises(ValueError):                                 # modes 0 and 1: ranks 2 vs 3
            fvo.fv_zero_point_frame(_template((6, 6, 5), (2, 3, 2), (1, 3, 2, 1)), ((0, 1), (2,)))


class TestNonCollapse(unittest.TestCase):
    """(e), F32: a coordinate-aligned completion gives a reference of EXACTLY zero when the data have no
    energy on its leading coordinates; the seeded Haar completion does not."""

    def test_trailing_supported_data(self):
        rng = np.random.default_rng(1)
        shape, tk, tt, W = (6, 7, 8), (2, 3, 2), (1, 2, 2, 1), 60
        ww = [rng.standard_normal((W, N)) for N in shape]
        ww = [np.concatenate([np.zeros((W, n)), w[:, n:]], axis=1) for w, n in zip(ww, tk)]
        data = rng.standard_normal(W)
        x0 = t3.TuckerTensorTrain.zeros(shape, tk, tt)
        problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
        g = problem.local_model(x0.data).gradient                            # at the start's LAPACK frame
        self.assertEqual(float(problem.geom.inner(g, g)), 0.0)               # the exact-zero trap
        with warnings.catch_warnings():
            warnings.simplefilter('error')                                   # no vanishing-reference warning
            _, stats = bopt.newton_cg(problem, x0.data, max_newton=1)
        row = stats['history'][0]
        self.assertEqual(row['g0_reference'], 'zero')
        self.assertGreater(row['g0norm'], 1.0)


class _CountingGeometry:
    """A duck-typed geometry wrapper that counts ``zero_point_frame`` calls (everything else delegates)."""

    def __init__(self, base):
        self.base = base
        self.calls = 0

    def zero_point_frame(self, frame):
        self.calls += 1
        return self.base.zero_point_frame(frame)

    def __getattr__(self, name):
        return getattr(self.base, name)


class _NineMemberGeometry:
    """A custom geometry with exactly the nine required members -- no optional ``zero_point_frame``."""

    def __init__(self, base):
        self._base = base

    def frame(self, x_cores):                       return self._base.frame(x_cores)
    def base_point(self, frame):                    return self._base.base_point(frame)
    def stack_shape(self, x_cores):                 return self._base.stack_shape(x_cores)
    def precompute(self, frame):                    return self._base.precompute(frame)
    def project(self, frame, variations, aux=None): return self._base.project(frame, variations, aux)
    def retract(self, frame, variations, aux=None): return self._base.retract(frame, variations, aux)
    def inner(self, a, b):                          return self._base.inner(a, b)
    def point_norm_sq(self, x_cores):               return self._base.point_norm_sq(x_cores)
    def point_tangent(self, frame):                 return self._base.point_tangent(frame)


class TestReferenceResolution(unittest.TestCase):
    """The 5-step resolution of C.15, on both optimizers that use a relative stop."""

    def setUp(self):
        np.random.seed(3)
        rng = np.random.default_rng(3)
        self.shape, self.tk, self.tt = (5, 6, 7), (2, 2, 2), (1, 2, 2, 1)
        self.A = t3.TuckerTensorTrain.randn(self.shape, self.tk, self.tt)
        self.ww = _unit_rows(rng, 60, self.shape)
        self.data = np.asarray(self.A.apply(self.ww))
        self.x0 = t3.TuckerTensorTrain.randn(self.shape, self.tk, self.tt)   # nonzero: valid on corewise too

    def _problem(self, geom, data=None):
        return bopt.least_squares_problem(geom, bfit.APPLY, self.ww, self.data if data is None else data)

    def test_none_is_the_geometrys_own(self):
        """(a): None -> 'zero' on the manifold geometries (ragged, uniform, shared), 'warm_start' on the
        corewise ones; an explicit 'zero' where there is no zero-point frame is a structural error."""
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(self.x0)
        spec = (0, 0, 1)                                                     # a genuinely shared point
        xs = _tied(t3.TuckerTensorTrain.randn((6, 6, 5), (2, 2, 2), (1, 2, 2, 1)), ((0, 1),))
        wws = _unit_rows(np.random.default_rng(4), 60, (6, 6, 5))
        ds = np.asarray(_tied(t3.TuckerTensorTrain.randn((6, 6, 5), (2, 2, 2), (1, 2, 2, 1)), ((0, 1),)).apply(wws))
        uxs = ut3.UniformTuckerTensorTrain.from_t3(xs)
        cases = [('MANIFOLD', t3m.MANIFOLD, self.ww, self.data, self.x0, 'zero'),
                 ('COREWISE', t3m.COREWISE, self.ww, self.data, self.x0, 'warm_start'),
                 ('UNIFORM_MANIFOLD', ut3m.UNIFORM_MANIFOLD, self.ww, self.data, ux0, 'zero'),
                 ('UNIFORM_COREWISE', ut3m.UNIFORM_COREWISE, self.ww, self.data, ux0, 'warm_start'),
                 ('shared MANIFOLD', sg.shared(t3m.MANIFOLD, spec), wws, ds, xs, 'zero'),
                 ('shared COREWISE', sg.shared(t3m.COREWISE, spec), wws, ds, xs, 'warm_start'),
                 ('shared UNIFORM_MANIFOLD', sg.shared(ut3m.UNIFORM_MANIFOLD, spec), wws, ds, uxs, 'zero')]
        for label, geometry, ww, data, x0, expected in cases:
            with self.subTest(geometry=label):
                _, stats = topt.newton_cg(geometry, 'apply', ww, data, x0, max_newton=1)
                self.assertEqual(stats['history'][0]['g0_reference'], expected)
                _, stats = topt.newton_cg(geometry, 'apply', ww, data, x0, max_newton=1,
                                          g0_reference='Warm_Start')         # case-insensitive
                self.assertEqual(stats['history'][0]['g0_reference'], 'warm_start')
                if expected == 'warm_start':
                    with self.assertRaisesRegex(ValueError, 'zero-point frame'):
                        topt.newton_cg(geometry, 'apply', ww, data, x0, max_newton=1, g0_reference='zero')
                    with self.assertRaisesRegex(ValueError, 'zero-point frame'):
                        topt.gradient_descent(geometry, 'apply', ww, data, x0, n_iter=1, g0_reference='zero')
        with self.assertRaisesRegex(ValueError, 'g0_reference must be'):
            topt.newton_cg(t3m.MANIFOLD, 'apply', self.ww, self.data, self.x0, g0_reference='start')

    def test_custom_geometry_without_the_member(self):
        """(a), m34: a custom geometry with only the nine required members runs unchanged -- the default is
        the warm start, bit-identical to an explicit 'warm_start' on the geometry it wraps -- and an explicit
        'zero' names the missing member."""
        custom = self._problem(_NineMemberGeometry(bgeo.ManifoldGeometryOps()))
        _, st_custom = bopt.newton_cg(custom, self.x0.data, max_newton=4)
        _, st_base = bopt.newton_cg(self._problem(bgeo.ManifoldGeometryOps()), self.x0.data, max_newton=4,
                                    g0_reference='warm_start')
        self.assertEqual(st_custom['losses'], st_base['losses'])
        self.assertEqual(st_custom['history'][0]['g0_reference'], 'warm_start')
        with self.assertRaisesRegex(ValueError, 'zero_point_frame'):
            bopt.newton_cg(custom, self.x0.data, max_newton=1, g0_reference='zero')
        with self.assertRaisesRegex(ValueError, 'zero_point_frame'):
            bopt.gradient_descent(custom, self.x0.data, n_iter=1, g0_reference='zero')

    def test_overrides_win_and_skip_the_computation(self):
        """(f), m32: g0norm_newton alone sets both references and never asks for the zero-point frame;
        g0norm_cg alone sets only CG and does not skip; the reference is computed once per run."""
        def run(**kw):
            geom = _CountingGeometry(bgeo.ManifoldGeometryOps())
            _, stats = bopt.newton_cg(self._problem(geom), self.x0.data, max_newton=4, **kw)
            return geom.calls, stats['history']

        calls, h = run()
        self.assertEqual(calls, 1)                                           # once, at iteration 0
        zero_ref = h[0]['g0norm']
        self.assertTrue(all(row['g0norm'] == zero_ref and row['g0_reference'] == 'zero' for row in h))
        calls, h = run(g0norm_newton=5.0)
        self.assertEqual(calls, 0)
        self.assertEqual((h[0]['g0_reference'], h[0]['cg_reference'], h[0]['g0norm']), ('override', 'override', 5.0))
        self.assertAlmostEqual(h[0]['forcing_eta'], min(0.5, (h[0]['gnorm'] / 5.0) ** 0.5), places=12)
        calls, h = run(g0norm_cg=5.0)
        self.assertEqual(calls, 1)
        self.assertEqual((h[0]['g0_reference'], h[0]['cg_reference'], h[0]['g0norm']), ('zero', 'override', zero_ref))
        self.assertAlmostEqual(h[0]['forcing_eta'], min(0.5, (h[0]['gnorm'] / 5.0) ** 0.5), places=12)
        calls, h = run(g0_reference='warm_start')
        self.assertEqual(calls, 0)
        # gradient_descent: the same resolution, once per run
        geom = _CountingGeometry(bgeo.ManifoldGeometryOps())
        bopt.gradient_descent(self._problem(geom), self.x0.data, n_iter=3)
        self.assertEqual(geom.calls, 1)
        geom = _CountingGeometry(bgeo.ManifoldGeometryOps())
        bopt.gradient_descent(self._problem(geom), self.x0.data, n_iter=3, g0_reference='warm_start')
        self.assertEqual(geom.calls, 0)

    def test_vanishing_reference(self):
        """(g), m33: all-zero data -- the zero-tensor gradient is 0. By default: a RuntimeWarning and the
        warm start (the fit from a nonzero start still converges); an explicit 'zero' raises."""
        zeros = np.zeros_like(self.data)
        problem = self._problem(bgeo.ManifoldGeometryOps(), zeros)
        with self.assertWarnsRegex(RuntimeWarning, 'vanishes'):
            _, stats = bopt.newton_cg(problem, self.x0.data)
        self.assertEqual(stats['history'][0]['g0_reference'], 'warm_start')
        self.assertEqual(stats['stop_reason'], 'gradient')
        self.assertLess(stats['losses'][-1], 1e-12 * stats['losses'][0])
        with self.assertRaisesRegex(ValueError, 'reference is 0'):
            bopt.newton_cg(problem, self.x0.data, g0_reference='zero')
        with self.assertWarnsRegex(RuntimeWarning, 'vanishes'):
            bopt.gradient_descent(problem, self.x0.data, n_iter=2)
        with self.assertRaisesRegex(ValueError, 'reference is 0'):
            bopt.gradient_descent(problem, self.x0.data, n_iter=2, g0_reference='zero')
        with self.assertRaises(ValueError):                                  # an override does not rescue 0
            bopt.newton_cg(problem, self.x0.data, g0norm_newton=0.0)

    def test_warm_start_is_the_2026_2_reference(self):
        """(h): 'warm_start' -- and the corewise default -- is today's `‖g(x0)‖`, bitwise."""
        for geom, kw in ((bgeo.ManifoldGeometryOps(), dict(g0_reference='warm_start')),
                         (bgeo.CorewiseGeometryOps(), {})):
            with self.subTest(geometry=type(geom).__name__):
                problem = self._problem(geom)
                g = problem.local_model(self.x0.data).gradient
                g0 = float(problem.geom.inner(g, g)) ** 0.5                   # the 2026.2.0 computed_g0norm
                _, stats = bopt.newton_cg(problem, self.x0.data, max_newton=3, **kw)
                for row in stats['history']:
                    self.assertEqual(row['g0norm'], g0)
                    self.assertEqual(row['g0_reference'], 'warm_start')

    def test_zero_reference_is_the_hand_computed_one(self):
        """The recorded zero reference is ‖Π 𝒥ᵀ(S(0) − data)‖ at the canonical frame of x0's structure --
        independent of x0's VALUES (only its ranks matter), and the same from a zero start."""
        problem = self._problem(bgeo.ManifoldGeometryOps())
        geom, kind = problem.geom, problem.kind
        F0 = geom.zero_point_frame(problem.local_model(self.x0.data).frame)
        g = geom.project(F0, kind.transpose(-self.data, self.ww, F0, kind.precompute(F0, self.ww)))
        expected = float(geom.inner(g, g)) ** 0.5
        for x0 in (self.x0, t3.TuckerTensorTrain.zeros(self.shape, self.tk, self.tt),
                   t3.TuckerTensorTrain.randn(self.shape, self.tk, self.tt)):
            _, stats = bopt.newton_cg(problem, x0.data, max_newton=1)
            self.assertAlmostEqual(stats['history'][0]['g0norm'], expected, delta=1e-13 * expected)

    def test_regularizer_does_not_enter_the_reference(self):
        """(i): ρ's gradient vanishes at the zero tensor, so the regularized reference is the unregularized one."""
        plain = bopt.newton_cg(self._problem(bgeo.ManifoldGeometryOps()), self.x0.data, max_newton=1)[1]
        reg = bopt.newton_cg(bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, self.ww,
                                                        self.data, regularizer=bopt.IdentityRegularizer(0.7)),
                             self.x0.data, max_newton=1)[1]
        self.assertEqual(reg['history'][0]['g0_reference'], 'zero')
        self.assertAlmostEqual(reg['history'][0]['g0norm'], plain['history'][0]['g0norm'],
                               delta=1e-14 * plain['history'][0]['g0norm'])


class TestLayerIndependence(unittest.TestCase):
    """(b), P1c: the canonical frame has the same real blocks on both layers, so the reference is the same
    number, and a seeded ragged fit and its uniform twin keep tracking each other (the O2 contract)."""

    def setUp(self):
        np.random.seed(5)

    def _case(self, kind, shared):
        rng = np.random.default_rng(5)
        shape = (6, 6, 5) if shared else (5, 6, 7)
        tk, tt = (3, 3, 2), (1, 3, 2, 1)
        sharing = (0, 0, 1) if shared else None
        A = t3.TuckerTensorTrain.randn(shape, tk, tt)
        x = t3.TuckerTensorTrain.randn(shape, tk, tt)                      # a NONZERO start (see §4.11)
        if shared:
            A, x = _tied(A, ((0, 1),)), _tied(x, ((0, 1),))
        ww = _unit_rows(rng, 40, shape)
        if kind == 'apply':
            return x, sharing, ww, np.asarray(A.apply(ww)), {}
        if kind == 'probe':
            return x, sharing, ww, [np.asarray(z) for z in A.probe(ww)], {}
        pp = _unit_rows(rng, 40, shape)
        omega = np.array([1.0, 0.5, 0.25])
        jets = [np.asarray(z) for z in A.probe_derivatives(ww, pp, 2)]
        return x, sharing, (ww, pp), jets, dict(order=2, weight=omega)

    def _geometries(self, sharing):
        if sharing is None:
            return t3m.MANIFOLD, ut3m.UNIFORM_MANIFOLD
        return sg.shared(t3m.MANIFOLD, sharing), sg.shared(ut3m.UNIFORM_MANIFOLD, sharing)

    def test_reference_is_layer_independent(self):
        for kind in ('apply', 'probe', 'probe_derivatives'):
            for shared in (False, True):
                with self.subTest(kind=kind, shared=shared):
                    x, sharing, sample, data, kw = self._case(kind, shared)
                    rg, ug = self._geometries(sharing)
                    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                    r = topt.newton_cg(rg, kind, sample, data, x, max_newton=1, **kw)[1]['history'][0]
                    u = topt.newton_cg(ug, kind, sample, data, ux, max_newton=1, **kw)[1]['history'][0]
                    self.assertEqual((r['g0_reference'], u['g0_reference']), ('zero', 'zero'))
                    self.assertAlmostEqual(u['g0norm'], r['g0norm'], delta=1e-14 * r['g0norm'])

    def test_seeded_pairs_track(self):
        """The pinned pair check: a ragged and a uniform newton_cg from the same nonzero start have equal
        `newton_losses` to 1e-7 (the O2 threshold; observed worst over the 44 O2 pairs 3.5e-12, F37)."""
        for kind, shared in (('apply', False), ('probe_derivatives', False), ('apply', True), ('probe', True)):
            with self.subTest(kind=kind, shared=shared):
                x, sharing, sample, data, kw = self._case(kind, shared)
                rg, ug = self._geometries(sharing)
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                lr = np.array(topt.newton_cg(rg, kind, sample, data, x, max_newton=6, **kw)[1]['losses'])
                lu = np.array(topt.newton_cg(ug, kind, sample, data, ux, max_newton=6, **kw)[1]['losses'])
                self.assertEqual(len(lr), len(lu))     # same stop iteration on both layers, not just a common prefix
                m = len(lr)
                self.assertGreater(m, 1)
                self.assertLessEqual(float(np.linalg.norm(lr[:m] - lu[:m]) / np.linalg.norm(lr[:m])), 1e-7)


if __name__ == '__main__':
    unittest.main()
