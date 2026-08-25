# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The frame-sweep REUSE contract (Phase D of the 2026-08-22 review): the ``*_from_sweep`` forms fed a
``tv_precompute_*`` sweep must equal the direct one-shot ops -- the hook pair the ``Kind`` layer's inner
CG relies on, previously only indirectly tested. Ragged: all 12 jacobian/transpose hooks (plain + jets).
Uniform: the 6 plain hooks checked per stack element against the RAGGED twins at the sliced frames
(non-circular -- the uniform jets hooks are behaviorally covered through the kinds in
``tests/backend/test_uniform_fitting.py``)."""
import unittest
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.sampling_derivatives as pd

_STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))


def _relerr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _pair_relerr(A, B):
    return max(max(_relerr(x, y) for x, y in zip(A[0], B[0])),
               max(_relerr(x, y) for x, y in zip(A[1], B[1])))


class TestRaggedFromSweepReuse(unittest.TestCase):
    W, K, C, ORDER = (2,), (2,), (2,), 2

    def setUp(self):
        np.random.seed(0)
        shapes = _STRUCT[0]
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=self.C)
        self.frame = t3m.MANIFOLD.frame(x).data
        self.var = t3m.COREWISE.randn(t3m.MANIFOLD.frame(x), stack_shape=self.K).variations.data
        self.ww = [np.random.randn(*(self.W + (N,))) for N in shapes]
        self.pp = [np.random.randn(*(self.W + (N,))) for N in shapes]
        self.index = np.stack([np.random.randint(0, N, size=self.W) for N in shapes], axis=0)

    def test_probe(self):
        sweep = probing.tv_precompute_probe_frame_sweep(self.frame, self.ww)
        z_direct = probing.tv_probe(self.ww, self.var, self.frame)
        z_sweep = probing.tv_probe_jacobian_from_sweep(self.var, self.ww, self.frame, sweep)
        for a, b in zip(z_sweep, z_direct):
            self.assertLessEqual(_relerr(a, b), 1e-13)
        zt = [np.random.randn(*np.asarray(zi).shape) for zi in z_direct]
        for sop in (True, False):
            with self.subTest(sum_over_probes=sop):
                g_direct = probing.tv_probe_transpose(zt, self.ww, self.frame, sum_over_probes=sop)
                g_sweep = probing.tv_probe_transpose_from_sweep(zt, self.ww, self.frame, sweep,
                                                                sum_over_probes=sop)
                self.assertLessEqual(_pair_relerr(g_sweep, g_direct), 1e-13)
        # jets
        sweep_j = pd.tv_precompute_probe_frame_sweep_jets(self.frame, self.ww, self.pp, self.ORDER)
        zj_direct = pd.tv_probe_derivatives(self.ww, self.pp, self.var, self.frame, self.ORDER)
        zj_sweep = pd.tv_probe_jacobian_derivatives_from_sweep(self.var, self.ww, self.pp,
                                                               self.frame, sweep_j, self.ORDER)
        for a, b in zip(zj_sweep, zj_direct):
            self.assertLessEqual(_relerr(a, b), 1e-13)
        r = [np.random.randn(*np.asarray(zi).shape) for zi in zj_direct]
        for sop in (True, False):
            with self.subTest(jets_sum_over_probes=sop):
                gj_direct = pd.tv_probe_derivatives_transpose(r, self.ww, self.pp, self.frame,
                                                              self.ORDER, sum_over_probes=sop)
                gj_sweep = pd.tv_probe_transpose_derivatives_from_sweep(
                    r, self.ww, self.pp, self.frame, sweep_j, self.ORDER, sum_over_probes=sop)
                self.assertLessEqual(_pair_relerr(gj_sweep, gj_direct), 1e-13)

    def test_apply(self):
        sweep = apply.tv_precompute_apply_frame_sweep(self.frame, self.ww)
        y_direct = apply.tv_apply(self.ww, self.var, self.frame)
        y_sweep = apply.tv_apply_jacobian_from_sweep(self.var, self.ww, self.frame, sweep)
        self.assertLessEqual(_relerr(y_sweep, y_direct), 1e-13)
        c = np.random.randn(*np.asarray(y_direct).shape)
        for sop in (True, False):
            with self.subTest(sum_over_probes=sop):
                g_direct = apply.tv_apply_transpose(c, self.ww, self.frame, sum_over_probes=sop)
                g_sweep = apply.tv_apply_transpose_from_sweep(c, self.ww, self.frame, sweep,
                                                              sum_over_probes=sop)
                self.assertLessEqual(_pair_relerr(g_sweep, g_direct), 1e-13)
        sweep_j = pd.tv_precompute_apply_frame_sweep_jets(self.frame, self.ww, self.pp, self.ORDER)
        yj_direct = pd.tv_apply_derivatives(self.ww, self.pp, self.var, self.frame, self.ORDER)
        yj_sweep = pd.tv_apply_jacobian_derivatives_from_sweep(self.var, self.ww, self.pp,
                                                               self.frame, sweep_j, self.ORDER)
        self.assertLessEqual(_relerr(yj_sweep, yj_direct), 1e-13)
        cj = np.random.randn(*np.asarray(yj_direct).shape)
        for sop in (True, False):
            with self.subTest(jets_sum_over_probes=sop):
                gj_direct = pd.tv_apply_derivatives_transpose(cj, self.ww, self.pp, self.frame,
                                                              self.ORDER, sum_over_probes=sop)
                gj_sweep = pd.tv_apply_transpose_derivatives_from_sweep(
                    cj, self.ww, self.pp, self.frame, sweep_j, self.ORDER, sum_over_probes=sop)
                self.assertLessEqual(_pair_relerr(gj_sweep, gj_direct), 1e-13)

    def test_entries(self):
        sweep = entries.tv_precompute_entries_frame_sweep(self.frame, self.index)
        e_direct = entries.tv_entries(self.index, self.var, self.frame)
        e_sweep = entries.tv_entries_jacobian_from_sweep(self.var, self.index, self.frame, sweep)
        self.assertLessEqual(_relerr(e_sweep, e_direct), 1e-13)
        c = np.random.randn(*np.asarray(e_direct).shape)
        for sop in (True, False):
            with self.subTest(sum_over_probes=sop):
                g_direct = entries.tv_entries_transpose(c, self.index, self.frame, sum_over_probes=sop)
                g_sweep = entries.tv_entries_transpose_from_sweep(c, self.index, self.frame, sweep,
                                                                  sum_over_probes=sop)
                self.assertLessEqual(_pair_relerr(g_sweep, g_direct), 1e-13)
        sweep_j = pd.tv_precompute_entries_frame_sweep_jets(self.frame, self.index, self.pp, self.ORDER)
        ej_direct = pd.tv_entries_derivatives(self.index, self.pp, self.var, self.frame, self.ORDER)
        ej_sweep = pd.tv_entries_jacobian_derivatives_from_sweep(self.var, self.index, self.pp,
                                                                 self.frame, sweep_j, self.ORDER)
        self.assertLessEqual(_relerr(ej_sweep, ej_direct), 1e-13)
        cj = np.random.randn(*np.asarray(ej_direct).shape)
        for sop in (True, False):
            with self.subTest(jets_sum_over_probes=sop):
                gj_direct = pd.tv_entries_derivatives_transpose(cj, self.index, self.pp, self.frame,
                                                                self.ORDER, sum_over_probes=sop)
                gj_sweep = pd.tv_entries_transpose_derivatives_from_sweep(
                    cj, self.index, self.pp, self.frame, sweep_j, self.ORDER, sum_over_probes=sop)
                self.assertLessEqual(_pair_relerr(gj_sweep, gj_direct), 1e-13)


class TestUniformPlainFromSweepVsRagged(unittest.TestCase):
    """The 6 plain uniform hooks, per (k, c) stack element against the RAGGED twins at the
    ``to_t3frame``-sliced frames (the non-circular cross-representation contract)."""
    W, K, C = (2,), (2,), (2,)

    def setUp(self):
        np.random.seed(0)
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_frame_variations_format as ubv
        import t3toolbox.uniform_manifold as ut3m
        import t3toolbox.backend.utv_sampling as utvs
        self.ubv, self.ut3m, self.utvs = ubv, ut3m, utvs
        shapes = _STRUCT[0]
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=self.C)
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        self.frame_u = ut3m.UNIFORM_MANIFOLD.frame(ux)
        self.v = ut3m.UNIFORM_MANIFOLD.randn(self.frame_u, stack_shape=self.K)
        self.frames_r = self.frame_u.to_t3frame()
        self.ww = [np.random.randn(*(self.W + (N,))) for N in shapes]
        self.index = np.stack([np.random.randint(0, N, size=self.W) for N in shapes], axis=0)

    def _leaves(self):    # (k, c) -> (ragged frame, ragged variations data)
        out = {}
        for k, tk_ in enumerate(self.v.unstack_tangents()):
            for c, leaf in enumerate(tk_.unstack_frame()):
                out[(k, c)] = (self.frames_r[c], leaf.variations.to_t3variations())
        return out

    def _bare_to_dense(self, dU, dG):
        """Wrap a bare (dU, dG) variation supercore pair (stack K+C) in the K-broadcast gauge masks of
        ``self.v.variations`` -> dense tangent."""
        V = self.ubv.UT3Variations(np.asarray(dU), np.asarray(dG), self.frame_u.shape,
                                   self.v.variations.masks)
        return np.asarray(self.ut3m.UT3Tangent(self.frame_u, V).to_dense())

    def test_jacobians_per_element(self):
        d = len(_STRUCT[0])
        s_probe = self.utvs.utv_precompute_probe_frame_sweep(self.frame_u.data, self.ww)
        s_apply = self.utvs.utv_precompute_apply_frame_sweep(self.frame_u.data, self.ww)
        s_entries = self.utvs.utv_precompute_entries_frame_sweep(self.frame_u.data, self.index)
        z_u = self.utvs.utv_probe_jacobian_from_sweep(self.v.variations.data, s_probe)   # packed
        y_u = np.asarray(self.utvs.utv_apply_jacobian_from_sweep(self.v.variations.data, s_apply))
        e_u = np.asarray(self.utvs.utv_entries_jacobian_from_sweep(self.v.variations.data, s_entries))
        for (k, c), (fr, vr) in self._leaves().items():
            z_r = probing.tv_probe(self.ww, vr.data, fr.data)
            for i in range(d):
                with self.subTest(op='probe', k=k, c=c, i=i):
                    got = np.asarray(z_u[i])[..., k, c, :_STRUCT[0][i]]
                    self.assertLessEqual(_relerr(got, np.asarray(z_r[i])), 1e-10)
            with self.subTest(op='apply', k=k, c=c):
                self.assertLessEqual(_relerr(y_u[..., k, c],
                                             apply.tv_apply(self.ww, vr.data, fr.data)), 1e-10)
            with self.subTest(op='entries', k=k, c=c):
                self.assertLessEqual(_relerr(e_u[..., k, c],
                                             entries.tv_entries(self.index, vr.data, fr.data)), 1e-10)

    def test_transposes_per_element(self):
        shapes = _STRUCT[0]
        s_probe = self.utvs.utv_precompute_probe_frame_sweep(self.frame_u.data, self.ww)
        s_apply = self.utvs.utv_precompute_apply_frame_sweep(self.frame_u.data, self.ww)
        s_entries = self.utvs.utv_precompute_entries_frame_sweep(self.frame_u.data, self.index)
        zt = [np.random.randn(*(self.W + self.K + self.C + (N,))) for N in shapes]
        c_res = np.random.randn(*(self.W + self.K + self.C))
        dU, dG = self.utvs.utv_probe_transpose_from_sweep(zt, s_probe, sum_over_probes=True)
        dense_u = self._bare_to_dense(dU, dG)
        for (k, c), (fr, vr) in self._leaves().items():
            zt_leaf = [z[(Ellipsis, k, c, slice(None))] for z in zt]
            g_r = probing.tv_probe_transpose(zt_leaf, self.ww, fr.data, sum_over_probes=True)
            d_r = np.asarray(t3m.T3Tangent(fr, bvf.T3Variations(*g_r)).to_dense())
            with self.subTest(op='probe', k=k, c=c):
                self.assertLessEqual(_relerr(dense_u[k, c], d_r), 1e-10)
        dU, dG = self.utvs.utv_apply_transpose_from_sweep(c_res, s_apply, sum_over_probes=True)
        dense_u = self._bare_to_dense(dU, dG)
        for (k, c), (fr, vr) in self._leaves().items():
            g_r = apply.tv_apply_transpose(c_res[..., k, c], self.ww, fr.data, sum_over_probes=True)
            d_r = np.asarray(t3m.T3Tangent(fr, bvf.T3Variations(*g_r)).to_dense())
            with self.subTest(op='apply', k=k, c=c):
                self.assertLessEqual(_relerr(dense_u[k, c], d_r), 1e-10)
        dU, dG = self.utvs.utv_entries_transpose_from_sweep(c_res, s_entries, sum_over_probes=True)
        dense_u = self._bare_to_dense(dU, dG)
        for (k, c), (fr, vr) in self._leaves().items():
            g_r = entries.tv_entries_transpose(c_res[..., k, c], self.index, fr.data,
                                               sum_over_probes=True)
            d_r = np.asarray(t3m.T3Tangent(fr, bvf.T3Variations(*g_r)).to_dense())
            with self.subTest(op='entries', k=k, c=c):
                self.assertLessEqual(_relerr(dense_u[k, c], d_r), 1e-10)
