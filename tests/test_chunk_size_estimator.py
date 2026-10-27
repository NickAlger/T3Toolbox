# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The eager chunk_size estimators (sampling_derivatives.estimate_chunk_size / max_chunk_size_within).

These size the probe-transpose gradient assembly from the problem shapes, measuring the assembly's
per-W-row scratch via XLA's memory_analysis (jax-only). We check the two policies against that same
measured cost: BALANCED keeps the assembly peak near the resident edge-jet floor; BUDGET keeps it under
an absolute cap. Plus sharding (n_shards divides W), the frame stack C (which multiplies the assembly but
not an absolute budget, so it shrinks the BUDGET chunk by prod(C) and the BALANCED one barely at all),
and the dense fallback for small problems.

The measured per-row cost is the COMPILER's, not a constant of the math: XLA releases pack the chunk loop
differently. On jaxlib 0.4.30 (the CI floor) buffer assignment fits the assemblies' chunk scratch into the
space the earlier W-sized stages free, so one more probe costs about one jet slice, d*(order+1)*r*itemsize
(384 B at the in-situ shape below); on current releases it costs the whole assembly row (5664 B there). Both
match the resident memory the program really allocates (checked against RSS, 2026-10-27). So these tests
pin the POLICIES against whatever slope is measured -- never a slope, a slope ratio, or a budget sized for
one compiler's slope (each of those broke the floor leg once).
"""
import math
import unittest
from unittest import mock
import numpy as np

import t3toolbox.backend.sampling_derivatives as pd

try:
    import jax  # the estimator compiles to measure; skip the module if jax is absent
    _HAVE_JAX = True
except ImportError:
    _HAVE_JAX = False


def _per_row(ms, tr, tt, order, K=1):
    return pd._assembly_per_row_bytes(tuple(ms), tuple(tr), tuple(tt), order, K, np.dtype('float32'))


@unittest.skipUnless(_HAVE_JAX, "estimator measures via jax memory_analysis")
class TestChunkSizeEstimator(unittest.TestCase):
    MS, TR, TT, ORDER = (128,) * 6, (128,) * 6, (128,) * 7, 5

    def test_balanced_tracks_the_jet_floor(self):
        # the balanced chunk is the largest whose assembly peak does not exceed the resident jets -- or all of W
        # when even that fits (here on every measured compiler: floor / per-row-cost is ~1.7 W on current XLA and
        # ~8 W on jaxlib 0.4.30, and that ratio is W-independent, so no W makes the balance bind)
        per_row = _per_row(self.MS, self.TR, self.TT, self.ORDER)
        for W in (3000, 32000):
            with self.subTest(W=W):
                cs = pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, W)
                self.assertTrue(1 <= cs <= W)
                floor = pd._jet_floor_bytes(self.MS, self.TR, self.TT, self.ORDER, W, 1, 4)
                self.assertLessEqual(cs * per_row, floor)              # never taller than the floor
                if cs < W:
                    self.assertGreater((cs + 1) * per_row, floor)      # and maximal (not needlessly tiny)
        # ... so pin the binding branch with a stand-in cost 4x the jets' per row: a chunk of ~W/4, maximal
        W = 32000
        floor = pd._jet_floor_bytes(self.MS, self.TR, self.TT, self.ORDER, W, 1, 4)
        with mock.patch.object(pd, '_assembly_per_row_bytes', return_value=4.0 * floor / W):
            cs = pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, W)
        self.assertEqual(cs, W // 4)

    def test_budget_is_respected(self):
        per_row = _per_row(self.MS, self.TR, self.TT, self.ORDER)
        W = 32000
        for target in (1e9, 4e9, 16e9):
            with self.subTest(target=target):
                cs = pd.max_chunk_size_within(self.MS, self.TR, self.TT, self.ORDER, W, target)
                self.assertLessEqual(cs * per_row, target)                    # within budget
                if cs < W:
                    self.assertGreater((cs + 1) * per_row, target - per_row)  # and near-maximal
                else:
                    self.assertLessEqual(W * per_row, target)                 # ... or the whole W fits (16e9 here)

    def test_sharding_sizes_the_local_shard(self):
        # n_shards divides |W|, so the per-device chunk shrinks ~proportionally
        c1 = pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, 32000, n_shards=1)
        c4 = pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, 32000, n_shards=4)
        self.assertEqual(round(c1 / c4), 4)

    def test_small_problem_is_dense(self):
        # when the whole thing already fits the balance, chunk == W_local (the transpose then runs dense)
        cs = pd.estimate_chunk_size((10, 11, 12), (5, 6, 4), (1, 2, 3, 1), 2, 50)
        self.assertEqual(cs, 50)

    def test_per_row_cost_is_independent_of_x64(self):
        # dtype=float32 means float32 whatever the global jax_enable_x64 says: the measured program must not
        # be promoted to float64 by default-dtype scan inits (it was, so a full suite -- where test_dispatch
        # turns x64 on -- saw different answers from this file run alone). Bypass the cache: measure twice.
        ms, tr, tt, order, f32 = (10, 11, 12), (5, 6, 4), (1, 2, 3, 1), 2, np.dtype('float32')
        before = jax.config.read('jax_enable_x64')
        got = {}
        try:
            for x64 in (False, True):
                jax.config.update('jax_enable_x64', x64)
                got[x64] = (pd._assembly_per_row_bytes.__wrapped__(ms, tr, tt, order, 1, f32),
                            pd._two_hole_assembly_per_row_bytes.__wrapped__(ms, tr, tt, order, 1, f32))
        finally:
            jax.config.update('jax_enable_x64', before)
        self.assertEqual(got[False], got[True])

    def test_returns_plain_int(self):
        cs = pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, 5000)
        self.assertIsInstance(cs, int)

    def test_frame_stack_shrinks_the_budget_chunk(self):
        # Every edge-variable jet carries the frame stack C, so the transpose's chunk costs nearly prod(C) times
        # more (not exactly: the probe vectors ww/pp carry no C) -- while an absolute target_bytes does not scale
        # at all. The budget policy must therefore return a ~prod(C)-times smaller chunk (measured for C = (4,)
        # here: 3.4x on current XLA; 4.0x on jaxlib 0.4.30, whose chunk cost is jet slices only, all carrying C);
        # ignoring C is how it would hand back a chunk that OOMs. W is large so the budget, not |W|, binds on
        # every compiler (the free chunk is ~2800 on current XLA but ~19400 on jaxlib 0.4.30).
        ms, tr, tt, order, W, target = (12,) * 3, (6,) * 3, (1, 6, 6, 1), 2, 10 ** 6, 2.0 ** 22   # free chunk < W
        free = pd.max_chunk_size_within(ms, tr, tt, order, W, target)
        for stack_shape in ((2,), (4,), (2, 2)):
            with self.subTest(stack_shape=stack_shape):
                got = pd.max_chunk_size_within(ms, tr, tt, order, W, target, stack_shape=stack_shape)
                self.assertLess(got, free)
                c = math.prod(stack_shape)
                self.assertGreater(free / got, 0.6 * c)             # shrinks ~prod(C) ...
                self.assertLessEqual(free / got, 1.05 * c)          # ... and no more
                per_row = pd._assembly_per_row_bytes(ms, tr, tt, order, 1, np.dtype('float32'),
                                                     tuple(stack_shape))
                self.assertLessEqual(got * per_row, target)          # still within the real budget

    def test_frame_stack_leaves_the_default_untouched(self):
        # stack_shape=() must reproduce the pre-parameter answers exactly (every existing call site)
        for W in (3000, 32000):
            with self.subTest(W=W):
                self.assertEqual(pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, W),
                                 pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, W,
                                                        stack_shape=()))

    def test_balanced_is_only_mildly_sensitive_to_the_frame_stack(self):
        # the jet floor it balances against scales with C too, so the two sides largely cancel -- but not
        # exactly, because ww/pp carry no C. Shrinking, yet nowhere near the prod(C) of the budget policy.
        ms, tr, tt, order, W = (12,) * 3, (6,) * 3, (1, 6, 6, 1), 2, 8192
        c1 = pd.estimate_chunk_size(ms, tr, tt, order, W)
        c8 = pd.estimate_chunk_size(ms, tr, tt, order, W, stack_shape=(8,))
        self.assertLessEqual(c8, c1)
        self.assertGreater(c8 * 8, c1)          # far less than the 8x a pure prod(C) scaling would give

    def test_hessian_also_bounds_the_two_hole(self):
        # An exact-Hessian fit runs the probe-derivative two-hole with the same chunk_size. hessian='exact' must
        # keep BOTH programs within their jet floors (balanced) / within the cap (budget), sizing by whichever
        # costs more per row; 'gauss_newton' (the default) is unchanged. Which one that is, is the compiler's: the
        # two-hole's dotted companions make it ~2.0x the transpose here on current XLA (why the argument exists),
        # but ~1.0x on jaxlib 0.4.30, whose chunk cost is one jet slice in both programs -- so pin the policy, not
        # the ratio. (Balanced, both clamp at |W| here: the in-situ per-row costs are below the resident jets'.)
        ms, tr, tt, order, W = (24,) * 4, (16,) * 4, (1, 16, 16, 16, 1), 3, 32000
        f32 = np.dtype('float32')
        gn = pd._assembly_per_row_bytes(ms, tr, tt, order, 1, f32)
        th = pd._two_hole_assembly_per_row_bytes(ms, tr, tt, order, 1, f32)
        floor_gn = pd._jet_floor_bytes(ms, tr, tt, order, W, 1, 4)
        floor_th = pd._two_hole_jet_floor_bytes(ms, tr, tt, order, W, 1, 4)
        plain = pd.estimate_chunk_size(ms, tr, tt, order, W)
        self.assertEqual(plain, pd.estimate_chunk_size(ms, tr, tt, order, W, hessian='gauss_newton'))
        self.assertEqual(plain, pd.estimate_chunk_size(ms, tr, tt, order, W, hessian='Gauss_Newton'))
        cs = pd.estimate_chunk_size(ms, tr, tt, order, W, hessian='exact')
        self.assertEqual(cs, pd.estimate_chunk_size(ms, tr, tt, order, W, hessian='EXACT'))   # case-insensitive
        self.assertEqual(cs, max(1, min(W, int(floor_gn // gn), int(floor_th // th))))      # the smaller balance
        self.assertLessEqual(cs, plain)
        self.assertLessEqual(cs * th, floor_th)
        self.assertLessEqual(cs * gn, floor_gn)
        W, target = 10 ** 6, 2.0 ** 26                         # the budget, not |W|, binds on every compiler
        free = pd.max_chunk_size_within(ms, tr, tt, order, W, target)
        self.assertEqual(free, pd.max_chunk_size_within(ms, tr, tt, order, W, target, hessian='gauss_newton'))
        capped = pd.max_chunk_size_within(ms, tr, tt, order, W, target, hessian='exact')
        worst = max(gn, th)
        self.assertEqual(capped, int(target // worst))         # sized by the costlier program (binds where th > gn)
        self.assertLessEqual(capped, free)
        self.assertLessEqual(capped * worst, target)
        self.assertGreater((capped + 1) * worst, target)       # and maximal

    def test_hessian_mode_is_the_solver_vocabulary(self):
        # The keyword means the solver's string mode everywhere ('gauss_newton' | 'exact'), so a pass-through
        # from newton_cg is right by construction; a bool (whose 'gauss_newton'-is-truthy reading would hand
        # every GN fit the smaller two-hole chunk), a typo or None is a structural error, in both estimators.
        args = ((24,) * 4, (16,) * 4, (1, 16, 16, 16, 1), 3, 32000)
        for bad in (True, False, None, 1, 'newton', 'gn', ''):
            with self.subTest(hessian=bad):
                with self.assertRaisesRegex(ValueError, "hessian must be 'gauss_newton' or 'exact'"):
                    pd.estimate_chunk_size(*args, hessian=bad)
                with self.assertRaisesRegex(ValueError, "hessian must be 'gauss_newton' or 'exact'"):
                    pd.max_chunk_size_within(*args, 2.0 ** 26, hessian=bad)


@unittest.skipUnless(_HAVE_JAX, "the in-situ check compiles via jax")
class TestBudgetHoldsInTheCompiledProgram(unittest.TestCase):
    """The budget policy's promise checked against the compiled programs the chunk size drives, not against the
    estimator's own per-row number (every test above compares the estimate with itself): at the chunk
    ``max_chunk_size_within`` returns, the chunk-dependent scratch of the real chunked programs -- the one-shot
    transpose, the transpose from a precomputed sweep (the fitting path), and for ``hessian='exact'`` the two-hole
    -- stays within ``target_bytes``. A Tucker-dominated shape: where a per-kernel measurement (inputs as
    parameters, so the chunk's slices of them unseen) under-estimated the running program by ~1.2-1.5x.

    Within ``PACKING_RTOL``: XLA's heap-packed scratch is affine in the chunk size only up to its buffer packing,
    so the slope measured here (over ``[c0, cs]``) and the estimator's (over ``_SLOPE_CHUNKS``) differ by ~0.1%
    -- measured +0.08% on jaxlib 0.4.30 and -0.14% on current XLA for the transpose -- and the estimator returns
    the MAXIMAL chunk under its slope, so a hard ``<=`` flips on that noise alone. 1% is ~10x the packing noise
    and far below the overshoot this test exists to catch."""

    PACKING_RTOL = 0.01

    def test_chunk_dependent_peak_within_the_budget(self):
        S = lambda *s: jax.ShapeDtypeStruct(s, np.float32)
        temp = lambda f, args: jax.jit(f).lower(*args).compile().memory_analysis().temp_size_in_bytes
        d, n, r, N, order, target, c0 = 3, 4, 8, 64, 3, 2.0 ** 21, 8
        ms, tr, tt = (N,) * d, (n,) * d, (1,) + (r,) * (d - 1) + (1,)
        frame = (S(d, n, N), S(d, r, n, r), S(d, r, n, r), S(d, r, n, r))           # uniform: every bond at r
        theta, dirs = (S(d, n, N), S(d, r, n, r)), (S(d, n, N), S(d, r, n, r))
        for hessian in ('gauss_newton', 'exact'):
            cs = pd.max_chunk_size_within(ms, tr, tt, order, 10 ** 6, target, hessian=hessian)
            W = 8 * cs                                     # a multiple of cs and of c0: no remainder chunk
            z, ww, pp = S(d, order + 1, W, N), S(d, W, N), S(d, W, N)
            sweep = (S(d, 2, W, n), S(d, order + 1, W, r), S(d, order + 1, W, r), S(d, order + 1, W, n))
            programs = {
                'transpose': (lambda c: lambda a, b, e, f: pd.tv_probe_derivatives_transpose(
                    a, b, e, f, order, sum_over_probes=True, chunk_size=c), (z, ww, pp, frame)),
                'transpose_from_sweep': (lambda c: lambda a, b, e, f, sw: pd.tv_probe_transpose_derivatives_from_sweep(
                    a, b, e, f, sw, order, sum_over_probes=True, chunk_size=c), (z, ww, pp, frame, sweep)),
            }
            if hessian == 'exact':
                programs = {'two_hole': (lambda c: lambda a, b, e, th, dd: pd.t3_probe_corewise_derivatives_hessian(
                    a, b, e, th, dd, order, chunk_size=c), (z, ww, pp, theta, dirs))}
            for name, (make, args) in programs.items():
                with self.subTest(hessian=hessian, program=name, chunk_size=cs):
                    self.assertGreater(cs, 4 * c0)                 # the budget binds (no clamp at |W|)
                    slope = (temp(make(cs), args) - temp(make(c0), args)) / (cs - c0)   # bytes per probe in the chunk
                    self.assertLessEqual(cs * slope, (1 + self.PACKING_RTOL) * target)   # the chunk's share fits


@unittest.skipUnless(_HAVE_JAX, "the 'auto' resolver estimates via jax")
class TestFittingChunkSizeWiring(unittest.TestCase):
    """chunk_size threads through the optimizer layer: 'auto' resolves to the estimator for a uniform
    probe_derivatives fit (None for ragged / non-probe), and chunking is exact through the optimizer."""

    STRUCT = ((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))

    def _probes(self, W, seed=0):
        rng = np.random.default_rng(seed)
        ww = [rng.standard_normal((W, n)) for n in self.STRUCT[0]]
        pp = [rng.standard_normal((W, n)) for n in self.STRUCT[0]]
        return ww, pp

    def test_resolver(self):
        import t3toolbox.optimizers as opt
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        xu = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*self.STRUCT))
        xr = t3.TuckerTensorTrain.randn(*self.STRUCT)
        s = self._probes(300)
        self.assertIsInstance(opt._resolve_chunk_size('auto', 'probe_derivatives', xu, s, 2), int)  # uniform -> int
        self.assertIsNone(opt._resolve_chunk_size('auto', 'probe_derivatives', xr, s, 2))            # ragged -> None
        self.assertIsNone(opt._resolve_chunk_size('auto', 'apply_derivatives', xu, s, 2))            # non-probe -> None
        self.assertEqual(opt._resolve_chunk_size(7, 'probe_derivatives', xu, s, 2), 7)               # int passthrough
        self.assertEqual(opt._resolve_chunk_size('auto', 'probe_derivatives', xu, s, 2, batch=40),   # minibatch caps W
                         opt._resolve_chunk_size('auto', 'probe_derivatives', xu, self._probes(40), 2))
        # The solver's hessian mode reaches the estimator, so an exact fit's chunk also bounds the two-hole. The two
        # ANSWERS cannot be told apart through the resolver: it sizes one CG direction (n_tangent = 1), where the
        # measured balance admits all of W in both modes -- jet floor / per-row cost >= 2.0 for the transpose and
        # the two-hole alike over 12 measured shapes (d = 3-5, order 0-6, ranks to 36, N to 1000; e.g. this STRUCT
        # at order 2: 2.89 GN, 2.55 exact) -- so both return W = 40000 here. The pass-through is pinned directly
        # instead: the forwarded mode (a spy), the estimator's own validation reached through the resolver, and
        # equality with a direct estimator call.
        from unittest import mock
        big = self._probes(40000)
        with mock.patch.object(pd, 'estimate_chunk_size', wraps=pd.estimate_chunk_size) as spy:
            exact = opt._resolve_chunk_size('auto', 'probe_derivatives', xu, big, 2, hessian='exact')
            gn = opt._resolve_chunk_size('auto', 'probe_derivatives', xu, big, 2)
        self.assertEqual([c.kwargs['hessian'] for c in spy.call_args_list], ['exact', 'gauss_newton'])
        with self.assertRaisesRegex(ValueError, "hessian must be 'gauss_newton' or 'exact'"):
            opt._resolve_chunk_size('auto', 'probe_derivatives', xu, big, 2, hessian='newton')
        d, nU, r = 3, int(xu.tucker_supercore.shape[-2]), int(xu.tt_supercore.shape[-1])
        for mode, got in (('exact', exact), ('gauss_newton', gn)):
            self.assertEqual(got, pd.estimate_chunk_size(self.STRUCT[0], (nU,) * d, (r,) * (d + 1), 2, 40000,
                                                         dtype=xu.tucker_supercore.dtype, hessian=mode))
        self.assertIsNone(opt._resolve_chunk_size('auto', 'probe_derivatives', xr, big, 2, hessian='exact'))  # ragged

    def test_newton_cg_exact_across_chunk_size(self):
        # chunk_size is an exact reorganization -> the fit is identical up to float rounding, for any
        # chunk_size. (Relative tolerance: the add-reducer differs from the dense sum only at ~1e-12.)
        import t3toolbox.optimizers as opt
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_manifold as ut3m
        np.random.seed(0)                     # TuckerTensorTrain.randn uses the global RNG; pin it
        ww, pp = self._probes(200)
        xtrue = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*self.STRUCT))
        data = xtrue.probe_derivatives(ww, pp, 2)
        x0 = ut3.UniformTuckerTensorTrain.zeros(*self.STRUCT)
        losses = []
        for cs in ('auto', 4, None):          # 4 forces chunking (W=200 > 4); all must agree (exact)
            _, stats = opt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', (ww, pp), data, x0,
                                     order=2, chunk_size=cs, max_newton=2)
            losses.append(stats['losses'][-1])
        ref = losses[2]                       # None = the dense assembly
        for l in losses[:2]:
            self.assertLessEqual(abs(l - ref), 1e-9 * abs(ref) + 1e-12)


if __name__ == '__main__':
    unittest.main()
