# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The eager chunk_size estimators (sampling_derivatives.estimate_chunk_size / max_chunk_size_within).

These size the probe-transpose gradient assembly from the problem shapes, measuring the assembly's
per-W-row scratch via XLA's memory_analysis (jax-only). We check the two policies against that same
measured cost: BALANCED keeps the assembly peak near the resident edge-jet floor; BUDGET keeps it under
an absolute cap. Plus sharding (n_shards divides W) and the dense fallback for small problems.
"""
import unittest
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
        # the balanced chunk makes the assembly peak comparable to (and not exceeding) the resident jets
        for W in (3000, 32000):
            cs = pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, W)
            self.assertTrue(1 <= cs <= W)
            floor = pd._jet_floor_bytes(self.MS, self.TR, self.TT, self.ORDER, W, 1, 4)
            peak = cs * _per_row(self.MS, self.TR, self.TT, self.ORDER)
            self.assertLessEqual(peak, floor)                 # never taller than the floor
            self.assertGreater(peak, 0.5 * floor)             # but comparable (not needlessly tiny)

    def test_budget_is_respected(self):
        per_row = _per_row(self.MS, self.TR, self.TT, self.ORDER)
        for target in (1e9, 4e9, 16e9):
            cs = pd.max_chunk_size_within(self.MS, self.TR, self.TT, self.ORDER, 32000, target)
            self.assertLessEqual(cs * per_row, target)                    # within budget
            self.assertGreater((cs + 1) * per_row, target - per_row)      # and near-maximal

    def test_sharding_sizes_the_local_shard(self):
        # n_shards divides |W|, so the per-device chunk shrinks ~proportionally
        c1 = pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, 32000, n_shards=1)
        c4 = pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, 32000, n_shards=4)
        self.assertEqual(round(c1 / c4), 4)

    def test_small_problem_is_dense(self):
        # when the whole thing already fits the balance, chunk == W_local (the transpose then runs dense)
        cs = pd.estimate_chunk_size((10, 11, 12), (5, 6, 4), (1, 2, 3, 1), 2, 50)
        self.assertEqual(cs, 50)

    def test_returns_plain_int(self):
        cs = pd.estimate_chunk_size(self.MS, self.TR, self.TT, self.ORDER, 5000)
        self.assertIsInstance(cs, int)


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
