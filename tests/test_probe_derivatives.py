# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
import itertools
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.probing as t3p
import t3toolbox.backend.probe_derivatives as pd

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm


class TestProbeDerivatives(unittest.TestCase):
    def check_relerr(self, expected, actual):
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        denom = norm(expected)
        if denom == 0.0:
            self.assertLessEqual(norm(actual), tol)
        else:
            self.assertLessEqual(norm(actual - expected) / denom, tol)

    def test_probe_derivatives_match_dense(self):
        # y_i^(k) = d^k/ds^k y_i(X + s P)|_0 matches the exact multilinear subset-expansion oracle,
        # for every sample of the W stack (unstacked is W=()).
        STRUCTS = [
            ((4, 5),         (2, 3),       (1, 2, 1)),
            ((4, 5, 6),      (2, 3, 2),    (1, 2, 2, 1)),
            ((5, 4, 6, 5),   (2, 3, 2, 3), (1, 2, 3, 2, 1)),
        ]
        for STRUCT in STRUCTS:
            for W in [(), (3,), (2, 2)]:
                for ORDER in [0, 1, 2, 3, 4]:
                    with self.subTest(STRUCT=STRUCT, W=W, ORDER=ORDER):
                        shapes = STRUCT[0]
                        d = len(shapes)
                        x = t3.TuckerTensorTrain.randn(*STRUCT)
                        T = x.to_dense()
                        ww = [np.random.randn(*(W + (N,))) for N in shapes]
                        pp = [np.random.randn(*(W + (N,))) for N in shapes]

                        z_jets = pd.probe_derivatives_t3(ww, pp, x.data, ORDER)

                        self.assertEqual(len(z_jets), d)
                        for i in range(d):
                            self.assertEqual(np.asarray(z_jets[i]).shape, (ORDER + 1,) + W + (shapes[i],))

                        # check every sample against the unstacked dense oracle
                        for sample in itertools.product(*[range(n) for n in W]):
                            sel = (slice(None),) + sample   # (order, *sample) index into a z_jet
                            ww_s = [w[sample] for w in ww]
                            pp_s = [p[sample] for p in pp]
                            z_dense = pd.probe_derivatives_dense(ww_s, pp_s, T, ORDER)
                            for i in range(d):
                                for k in range(ORDER + 1):
                                    self.check_relerr(z_dense[i][k], np.asarray(z_jets[i])[sel][k])

    def test_base_core_stack(self):
        # base/core stack C (a batch of T3s) alongside the sample stack S, base-inner S+C. Each base
        # T3 is probed by the same S samples; validate every (sample, base) element vs the oracle.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        shapes = STRUCT[0]
        d = len(shapes)
        ORDER = 3
        for S, C in [((), (2,)), ((3,), (2,)), ((2,), (2, 2))]:
            with self.subTest(S=S, C=C):
                x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                T = x.to_dense()                                  # shape C + (N1..Nd)
                ww = [np.random.randn(*(S + (N,))) for N in shapes]
                pp = [np.random.randn(*(S + (N,))) for N in shapes]

                z_jets = pd.probe_derivatives_t3(ww, pp, x.data, ORDER)
                for i in range(d):
                    self.assertEqual(np.asarray(z_jets[i]).shape, (ORDER + 1,) + S + C + (shapes[i],))

                for s_idx in itertools.product(*[range(n) for n in S]):
                    ww_s = [w[s_idx] for w in ww]
                    pp_s = [p[s_idx] for p in pp]
                    for c_idx in itertools.product(*[range(n) for n in C]):
                        z_dense = pd.probe_derivatives_dense(ww_s, pp_s, T[c_idx], ORDER)
                        sel = (slice(None),) + s_idx + c_idx       # (order, *S, *C)
                        for i in range(d):
                            for k in range(ORDER + 1):
                                self.check_relerr(z_dense[i][k], np.asarray(z_jets[i])[sel][k])

    def test_tangent_derivatives_match_dense(self):
        # Riemannian forward: symmetric derivatives of a tangent vector's probing map. The densified
        # tangent is a dense tensor, so the oracle is the same multilinear subset expansion.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        shapes = STRUCT[0]
        d = len(shapes)
        for S, C in [((), ()), ((3,), ()), ((2,), (2,))]:
            for ORDER in [0, 1, 2, 3]:
                with self.subTest(S=S, C=C, ORDER=ORDER):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    v = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
                    Vd = v.to_dense()                                 # shape C + (N1..Nd)
                    ww = [np.random.randn(*(S + (N,))) for N in shapes]
                    pp = [np.random.randn(*(S + (N,))) for N in shapes]

                    z_jets = pd.probe_tangent_derivatives(ww, pp, v.variations.data, v.basis.data, ORDER)
                    for i in range(d):
                        self.assertEqual(np.asarray(z_jets[i]).shape, (ORDER + 1,) + S + C + (shapes[i],))

                    for s_idx in itertools.product(*[range(n) for n in S]):
                        ww_s = [w[s_idx] for w in ww]
                        pp_s = [p[s_idx] for p in pp]
                        for c_idx in itertools.product(*[range(n) for n in C]):
                            z_dense = pd.probe_derivatives_dense(ww_s, pp_s, Vd[c_idx], ORDER)
                            sel = (slice(None),) + s_idx + c_idx
                            for i in range(d):
                                for k in range(ORDER + 1):
                                    self.check_relerr(z_dense[i][k], np.asarray(z_jets[i])[sel][k])

    def test_tangent_transpose_adjoint_identity(self):
        # J^T via the jet-ified adjoint-state Lagrangian: <r, J v> = <J^T r, v>, with the sample stack
        # S and base stack C (base-inner); sum_over_probes keeps/sums S consistently, C always kept.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        shapes = STRUCT[0]
        d = len(shapes)
        for S, C in [((), ()), ((3,), ()), ((2,), (2,)), ((), (2, 2))]:
            for K in [0, 1, 2, 3]:
                with self.subTest(S=S, C=C, K=K):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    v = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
                    dU_v, dG_v = v.variations.data
                    ww = [np.random.randn(*(S + (N,))) for N in shapes]
                    pp = [np.random.randn(*(S + (N,))) for N in shapes]

                    z_jets = pd.probe_tangent_derivatives(ww, pp, v.variations.data, v.basis.data, K)
                    r = [np.random.randn(*np.asarray(zi).shape) for zi in z_jets]

                    dU, dG = pd.probe_tangent_derivatives_transpose(
                        r, ww, pp, v.basis.data, K, sum_over_probes=True)
                    lhs = sum(np.sum(r[i] * np.asarray(z_jets[i])) for i in range(d))
                    rhs = (sum(np.sum(np.asarray(dU[i]) * dU_v[i]) for i in range(d))
                           + sum(np.sum(np.asarray(dG[i]) * dG_v[i]) for i in range(d)))
                    self.assertLessEqual(abs(lhs - rhs) / abs(lhs), tol)

                    # sum_over_probes=False keeps S; summing those axes recovers the True result
                    if S:
                        dU0, dG0 = pd.probe_tangent_derivatives_transpose(
                            r, ww, pp, v.basis.data, K, sum_over_probes=False)
                        ax = tuple(range(len(S)))
                        for a, b in zip(dU0, dU):
                            self.check_relerr(np.asarray(b), np.sum(np.asarray(a), axis=ax))
                        for a, b in zip(dG0, dG):
                            self.check_relerr(np.asarray(b), np.sum(np.asarray(a), axis=ax))

    def test_tangent_order_zero_is_plain_tangent_probe(self):
        # The 0-th derivative jet of the Riemannian map is exactly probe_tangent.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        x = t3.TuckerTensorTrain.randn(*STRUCT)
        base, _ = bvf.t3_orthogonal_representations(x)
        v = t3m.T3Tangent.randn(base, apply_gauge_projection=False)
        ww = [np.random.randn(N) for N in STRUCT[0]]
        pp = [np.random.randn(N) for N in STRUCT[0]]

        z_jets = pd.probe_tangent_derivatives(ww, pp, v.variations.data, v.basis.data, 3)
        z_probe = t3p.probe_tangent(ww, v.variations.data, v.basis.data)
        for zj, zp in zip(z_jets, z_probe):
            self.check_relerr(zp, np.asarray(zj)[0])

    def test_order_zero_is_plain_probe(self):
        # The 0-th derivative jet is exactly the ordinary probe.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        x = t3.TuckerTensorTrain.randn(*STRUCT)
        ww = [np.random.randn(N) for N in STRUCT[0]]
        pp = [np.random.randn(N) for N in STRUCT[0]]

        z_jets = pd.probe_derivatives_t3(ww, pp, x.data, 3)
        z_probe = t3p.probe_t3(ww, x.data)
        for zj, zp in zip(z_jets, z_probe):
            self.check_relerr(zp, zj[0])

    def test_first_derivative_matches_finite_difference(self):
        # An independent check on order 1: directional derivative vs central finite difference.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        x = t3.TuckerTensorTrain.randn(*STRUCT)
        ww = [np.random.randn(N) for N in STRUCT[0]]
        pp = [np.random.randn(N) for N in STRUCT[0]]

        z_jets = pd.probe_derivatives_t3(ww, pp, x.data, 1)
        s = 1e-6
        z_plus  = t3p.probe_t3([w + s * p for w, p in zip(ww, pp)], x.data)
        z_minus = t3p.probe_t3([w - s * p for w, p in zip(ww, pp)], x.data)
        for i in range(len(STRUCT[0])):
            fd = (np.asarray(z_plus[i]) - np.asarray(z_minus[i])) / (2 * s)
            self.assertLessEqual(norm(fd - np.asarray(z_jets[i][1])) / norm(fd), 1e-6)

    def _all_modes_check(self, y, oracle, W, K, C, order):
        # y: (order+1,)+W+K+C scalar-jets; oracle(w_idx,k_idx,c_idx) -> (order+1,) dense reference.
        for w_idx in itertools.product(*[range(n) for n in W]):
            for k_idx in itertools.product(*[range(n) for n in K]):
                for c_idx in itertools.product(*[range(n) for n in C]):
                    got = np.asarray(y)[(slice(None),) + w_idx + k_idx + c_idx]
                    self.check_relerr(oracle(w_idx, k_idx, c_idx), got)

    def test_tangent_derivatives_K_stacked(self):
        # probe_tangent_derivatives with a tangent stack K (a batch of tangents sharing one base):
        # full W + K + C, base-inner, matching the dense oracle on each (W,K,C) element.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        shapes = STRUCT[0]
        d = len(shapes)
        for W, K, C in [((), (3,), ()), ((2,), (3,), ()), ((2,), (3,), (2,)), ((2, 2), (2,), (2,))]:
            for ORDER in [0, 1, 3]:
                with self.subTest(W=W, K=K, C=C, ORDER=ORDER):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    v = t3m.T3Tangent.randn(base, stack_shape=K, apply_gauge_projection=False)
                    Vd = v.to_dense()                              # K + C + (N...)
                    ww = [np.random.randn(*(W + (N,))) for N in shapes]
                    pp = [np.random.randn(*(W + (N,))) for N in shapes]
                    z_jets = pd.probe_tangent_derivatives(ww, pp, v.variations.data, v.basis.data, ORDER)
                    for i in range(d):
                        self.assertEqual(np.asarray(z_jets[i]).shape, (ORDER + 1,) + W + K + C + (shapes[i],))
                    for w_idx in itertools.product(*[range(n) for n in W]):
                        ww_s = [a[w_idx] for a in ww]
                        pp_s = [a[w_idx] for a in pp]
                        for k_idx in itertools.product(*[range(n) for n in K]):
                            for c_idx in itertools.product(*[range(n) for n in C]):
                                z_dense = pd.probe_derivatives_dense(ww_s, pp_s, Vd[k_idx + c_idx], ORDER)
                                sel = (slice(None),) + w_idx + k_idx + c_idx
                                for i in range(d):
                                    for k in range(ORDER + 1):
                                        self.check_relerr(z_dense[i][k], np.asarray(z_jets[i])[sel][k])

    def test_tangent_transpose_K_stacked(self):
        # K-stacked transpose: the adjoint identity <r, J v> = <J^T r, v> with a tangent stack K, plus
        # the sum_over_probes False/True consistency (summing the W axes of False recovers True).
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        shapes = STRUCT[0]
        d = len(shapes)
        for W, K, C in [((2,), (3,), ()), ((), (3,), ()), ((2,), (3,), (2,)), ((2, 2), (2,), (2,))]:
            for ORDER in [0, 1, 3]:
                with self.subTest(W=W, K=K, C=C, ORDER=ORDER):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    v = t3m.T3Tangent.randn(base, stack_shape=K, apply_gauge_projection=False)
                    dU_v, dG_v = v.variations.data
                    ww = [np.random.randn(*(W + (N,))) for N in shapes]
                    pp = [np.random.randn(*(W + (N,))) for N in shapes]

                    Jv = pd.probe_tangent_derivatives(ww, pp, v.variations.data, v.basis.data, ORDER)
                    r = [np.random.randn(*np.asarray(z).shape) for z in Jv]

                    dU, dG = pd.probe_tangent_derivatives_transpose(
                        r, ww, pp, v.basis.data, ORDER, sum_over_probes=True)
                    lhs = sum(np.sum(r[i] * np.asarray(Jv[i])) for i in range(d))
                    rhs = (sum(np.sum(np.asarray(dU[i]) * dU_v[i]) for i in range(d))
                           + sum(np.sum(np.asarray(dG[i]) * dG_v[i]) for i in range(d)))
                    self.assertLessEqual(abs(lhs - rhs) / abs(lhs), tol)

                    if W:                                  # summing the kept-W result recovers the summed one
                        dU0, dG0 = pd.probe_tangent_derivatives_transpose(
                            r, ww, pp, v.basis.data, ORDER, sum_over_probes=False)
                        ax = tuple(range(len(W)))
                        for a, b in zip(dU0, dU):
                            self.check_relerr(np.asarray(b), np.sum(np.asarray(a), axis=ax))
                        for a, b in zip(dG0, dG):
                            self.check_relerr(np.asarray(b), np.sum(np.asarray(a), axis=ax))

    def test_apply_derivatives_match_dense(self):
        # apply derivatives (all modes contracted): Euclidean (plain T3, W+C) and Riemannian (tangent,
        # W+K+C) vs the all-modes dense subset-expansion oracle.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        shapes = STRUCT[0]
        for W, K, C in [((), (), ()), ((2,), (), ()), ((2,), (3,), ()), ((2,), (3,), (2,))]:
            for ORDER in [0, 1, 3, 4]:
                with self.subTest(W=W, K=K, C=C, ORDER=ORDER):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                    ww = [np.random.randn(*(W + (N,))) for N in shapes]
                    pp = [np.random.randn(*(W + (N,))) for N in shapes]

                    if not K:                                      # Euclidean plain-T3 apply
                        T = x.to_dense()
                        y = pd.apply_derivatives_t3(ww, pp, x.data, ORDER)
                        self.assertEqual(y.shape, (ORDER + 1,) + W + C)
                        self._all_modes_check(
                            y, lambda w, k, c: pd.apply_derivatives_dense([a[w] for a in ww], [a[w] for a in pp], T[c], ORDER),
                            W, K, C, ORDER)

                    base, _ = bvf.t3_orthogonal_representations(x)  # Riemannian tangent apply
                    v = t3m.T3Tangent.randn(base, stack_shape=K, apply_gauge_projection=False)
                    Vd = v.to_dense()
                    yv = pd.apply_tangent_derivatives(ww, pp, v.variations.data, v.basis.data, ORDER)
                    self.assertEqual(yv.shape, (ORDER + 1,) + W + K + C)
                    self._all_modes_check(
                        yv, lambda w, k, c: pd.apply_derivatives_dense([a[w] for a in ww], [a[w] for a in pp], Vd[k + c], ORDER),
                        W, K, C, ORDER)
                    self.check_relerr(t3p.apply_tangent(ww, v.variations.data, v.basis.data), yv[0])

    def test_entries_derivatives_match_dense(self):
        # entries derivatives (all modes, one-hot base + general P): Euclidean and Riemannian vs oracle.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        shapes = STRUCT[0]
        d = len(shapes)
        for W, K, C in [((), (), ()), ((2,), (), ()), ((2,), (3,), ()), ((2,), (3,), (2,))]:
            for ORDER in [0, 1, 3]:
                with self.subTest(W=W, K=K, C=C, ORDER=ORDER):
                    x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                    index = np.stack([np.random.randint(0, N, size=W) for N in shapes], axis=0)  # (d,)+W
                    pp = [np.random.randn(*(W + (N,))) for N in shapes]
                    idx_s = lambda w: [int(index[(j,) + w]) for j in range(d)]

                    if not K:                                      # Euclidean
                        T = x.to_dense()
                        y = pd.entries_derivatives_t3(index, pp, x.data, ORDER)
                        self.assertEqual(y.shape, (ORDER + 1,) + W + C)
                        self._all_modes_check(
                            y, lambda w, k, c: pd.entries_derivatives_dense(idx_s(w), [a[w] for a in pp], T[c], ORDER),
                            W, K, C, ORDER)

                    base, _ = bvf.t3_orthogonal_representations(x)  # Riemannian
                    v = t3m.T3Tangent.randn(base, stack_shape=K, apply_gauge_projection=False)
                    Vd = v.to_dense()
                    yv = pd.entries_tangent_derivatives(index, pp, v.variations.data, v.basis.data, ORDER)
                    self.assertEqual(yv.shape, (ORDER + 1,) + W + K + C)
                    self._all_modes_check(
                        yv, lambda w, k, c: pd.entries_derivatives_dense(idx_s(w), [a[w] for a in pp], Vd[k + c], ORDER),
                        W, K, C, ORDER)
                    self.check_relerr(t3p.entries_tangent(index, v.variations.data, v.basis.data), yv[0])

    def test_apply_entries_transpose_adjoint_identity(self):
        # adjoint identity <c, J v> = <J^T c, v> for the all-modes apply/entries derivative transposes
        # (adjoint-state seeded sweep), with the tangent stack K; plus sum_over_probes consistency.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        shapes = STRUCT[0]
        d = len(shapes)
        for kind in ['apply', 'entries']:
            for W, K, C in [((2,), (3,), ()), ((), (3,), ()), ((2,), (3,), (2,)), ((2, 2), (2,), (2,))]:
                for ORDER in [0, 1, 3]:
                    with self.subTest(kind=kind, W=W, K=K, C=C, ORDER=ORDER):
                        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
                        base, _ = bvf.t3_orthogonal_representations(x)
                        v = t3m.T3Tangent.randn(base, stack_shape=K, apply_gauge_projection=False)
                        dU_v, dG_v = v.variations.data
                        pp = [np.random.randn(*(W + (N,))) for N in shapes]
                        if kind == 'apply':
                            ww = [np.random.randn(*(W + (N,))) for N in shapes]
                            Jv = pd.apply_tangent_derivatives(ww, pp, v.variations.data, v.basis.data, ORDER)
                            T = lambda cc, sop: pd.apply_tangent_derivatives_transpose(
                                cc, ww, pp, v.basis.data, ORDER, sum_over_probes=sop)
                        else:
                            index = np.stack([np.random.randint(0, N, size=W) for N in shapes], axis=0)
                            Jv = pd.entries_tangent_derivatives(index, pp, v.variations.data, v.basis.data, ORDER)
                            T = lambda cc, sop: pd.entries_tangent_derivatives_transpose(
                                cc, index, pp, v.basis.data, ORDER, sum_over_probes=sop)
                        c = np.random.randn(*np.asarray(Jv).shape)

                        dU, dG = T(c, True)
                        lhs = float(np.sum(c * np.asarray(Jv)))
                        rhs = (sum(np.sum(np.asarray(dU[i]) * dU_v[i]) for i in range(d))
                               + sum(np.sum(np.asarray(dG[i]) * dG_v[i]) for i in range(d)))
                        self.assertLessEqual(abs(lhs - rhs) / abs(lhs), tol)

                        if W:
                            dU0, dG0 = T(c, False)
                            ax = tuple(range(len(W)))
                            for a, b in zip(dU0, dU):
                                self.check_relerr(np.asarray(b), np.sum(np.asarray(a), axis=ax))
                            for a, b in zip(dG0, dG):
                                self.check_relerr(np.asarray(b), np.sum(np.asarray(a), axis=ax))

    def test_high_order_vanishes(self):
        # y_i depends on d-1 vectors, so symmetric derivatives above order d-1 are exactly zero.
        STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        d = len(STRUCT[0])
        x = t3.TuckerTensorTrain.randn(*STRUCT)
        ww = [np.random.randn(N) for N in STRUCT[0]]
        pp = [np.random.randn(N) for N in STRUCT[0]]

        z_jets = pd.probe_derivatives_t3(ww, pp, x.data, d + 1)
        for i in range(d):
            for k in range(d, d + 2):
                self.assertLessEqual(norm(np.asarray(z_jets[i][k])), tol)


if __name__ == "__main__":
    unittest.main()
