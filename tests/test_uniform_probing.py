# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Tests for uniform tangent probing (UT3Tangent probe / apply / entries), uniform-fix slice 3b-6.

The bare Riemannian Jacobian 𝒥 (3b-6b: forward) -- and, once landed, 𝒥ᵀ (3b-6c) -- on the uniform
(supercore + mask) tangent layer, verified per stack element against the ragged T3Tangent (the
equivalence contract). The hard, historically-untrusted part is the W (probe) / K (tangent) / C (base)
multi-block stacking, so every op is swept over the full _CONFIGS matrix (incl. K, multi-axis) and the
varying-C rank sweep. The underlying d-prefixed WKC contractions are unit-tested in
tests/backend/test_contractions.py (3b-6a)."""
import unittest
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_basis_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m

_STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))   # (shape, tucker_ranks, tt_ranks)
# (base stack C, tangent stack K): unstacked / C / K / K+C / multi-axis C / multi-axis K
_CONFIGS = [((), ()), ((2,), ()), ((), (3,)), ((2,), (3,)), ((2, 3), ()), ((), (2, 3))]

# varying-C rank sweep: two models of DIFFERENT base ranks padded to common dims (so they stack on one C)
_HETERO = [((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)), ((4, 5, 6), (3, 3, 2), (1, 1, 2, 1))]
_HETERO_PAD = dict(N=6, nU=4, nD=4, rL=3, rR=3)


def _uniform_tangent(C=(), K=(), seed=0):
    """A random uniform tangent at an orthogonal frame (forward 𝒥 is linear in the variation, so the
    corewise -- ungauged -- random tangent is a fine test input; gauge is irrelevant to 𝒥)."""
    np.random.seed(seed)
    x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=C)
    base = ut3m.UNIFORM_MANIFOLD.base(ut3.UniformTuckerTensorTrain.from_t3(x))
    return ut3m.UNIFORM_COREWISE.randn(base, stack_shape=K)


def _varying_C_tangent(seed=0):
    np.random.seed(seed)
    us = []
    for s in _HETERO:
        rb, rv = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(*s))
        ub = ubv.UT3Basis.from_t3basis(rb, **_HETERO_PAD)
        uv = ubv.UT3Variations.from_t3variations(rv, **_HETERO_PAD)
        us.append(ut3m.UT3Tangent(ub, uv))
    return ut3m.UT3Tangent.stack_basis(us)


def _full_unstack(v):
    """Fully unstack a UT3Tangent into a FLAT list of single-element tangents, K-major then C (the order
    of the W+K+C row-major flattening of a probe/apply/entries output)."""
    if not v.stack_shape:
        return [v]
    sub = v.unstack_tangents() if v.tangent_stack_shape else v.unstack_basis()
    out = []
    for leaf in ut3m._flatten_tangents(sub):
        out.extend(_full_unstack(leaf))
    return out


def _probe_vectors(W, seed=1):
    np.random.seed(seed)
    return [np.random.randn(*(tuple(W) + (N,))) for N in _STRUCT[0]]


def _index(W):
    base = np.array([1, 2, 3])                      # one valid multi-index (< every Ni)
    if not W:
        return base
    return np.broadcast_to(base[:, None], (len(base),) + tuple(W)) + np.zeros((1,) + tuple(W), int)


class TestUT3TangentForward(unittest.TestCase):
    """3b-6b: the bare forward Jacobian 𝒥 (probe / apply / entries on UT3Tangent), per stack element vs
    the ragged T3Tangent over the _CONFIGS matrix + varying-C, with the probe stack W = (2,)."""

    def _leaves(self, v):
        return [leaf.to_t3tangent() for leaf in _full_unstack(v)]

    def _check_probe(self, v, ww, W):
        zz = v.probe(ww)
        nW = int(np.prod(W)) if W else 1
        leaves = self._leaves(v)
        for i, leaf in enumerate(leaves):
            rzz = leaf.probe(ww)
            for m in range(len(rzz)):
                Nm = rzz[m].shape[-1]
                u = np.asarray(zz[m]).reshape((nW, -1, Nm))[:, i, :]
                self.assertTrue(np.allclose(u, np.asarray(rzz[m]), atol=1e-9),
                                msg='probe mode %d, element %d' % (m, i))

    def _check_apply(self, v, ww, W):
        aa = np.asarray(v.apply(ww))
        nW = int(np.prod(W)) if W else 1
        for i, leaf in enumerate(self._leaves(v)):
            u = aa.reshape((nW, -1))[:, i]
            self.assertTrue(np.allclose(u, np.asarray(leaf.apply(ww)), atol=1e-9), msg='apply element %d' % i)

    def _check_entries(self, v, index, W):
        ee = np.asarray(v.entries(index))
        nW = int(np.prod(W)) if W else 1
        for i, leaf in enumerate(self._leaves(v)):
            u = ee.reshape((nW, -1))[:, i]
            self.assertTrue(np.allclose(u, np.asarray(leaf.entries(index)), atol=1e-9),
                            msg='entries element %d' % i)

    def test_probe(self):
        W = (2,)
        ww = _probe_vectors(W)
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                self._check_probe(_uniform_tangent(C, K), ww, W)
        with self.subTest('varying_C'):
            self._check_probe(_varying_C_tangent(), ww, W)

    def test_apply(self):
        W = (2,)
        ww = _probe_vectors(W)
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                self._check_apply(_uniform_tangent(C, K), ww, W)
        with self.subTest('varying_C'):
            self._check_apply(_varying_C_tangent(), ww, W)

    def test_entries(self):
        W = (2,)
        index = _index(W)
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                self._check_entries(_uniform_tangent(C, K), index, W)
        with self.subTest('varying_C'):
            self._check_entries(_varying_C_tangent(), index, W)

    def test_no_probe_stack(self):
        # W = () (a single probe / index, no probe stack) -- apply & entries return a scalar per K+C element
        ww = _probe_vectors(())
        index = _index(())
        for C, K in [((), ()), ((2,), ()), ((), (3,)), ((2,), (3,))]:
            with self.subTest(C=C, K=K):
                v = _uniform_tangent(C, K)
                self._check_apply(v, ww, ())
                self._check_entries(v, index, ())
                self._check_probe(v, ww, ())


class TestUT3TangentTranspose(unittest.TestCase):
    """3b-6c: the bare transpose Jacobian 𝒥ᵀ (probe / apply / entries transpose) -- the adjoint identity
    ``<r, 𝒥V> = <𝒥ᵀr, V>`` (the defining property), the ``Σ_W(sum=False) == sum=True`` relation, and a
    per-element comparison to the ragged 𝒥ᵀ. Probe stack W = (2,)."""

    def setUp(self):
        np.random.seed(2)

    def _meas_dot_sumW(self, r, y):
        # measurement inner product summed over the probe stack W (axis 0), keeping K+C: for probe r/y are
        # d vectors (sum the mode too); for apply/entries r/y are single arrays shape W+K+C.
        if isinstance(y, (tuple, list)):
            return sum((np.asarray(a) * np.asarray(b)).sum(axis=(0, -1)) for a, b in zip(r, y))
        return (np.asarray(r) * np.asarray(y)).sum(axis=0)

    def _adjoint(self, v, forward, transpose):
        y = forward()                                          # 𝒥V
        r = [np.random.randn(*np.asarray(a).shape) for a in y] if isinstance(y, (tuple, list)) \
            else np.random.randn(*np.asarray(y).shape)
        JTr = transpose(r)                                     # 𝒥ᵀr (sum_over_probes=True -> tangent stack K)
        lhs = self._meas_dot_sumW(r, y)
        rhs = np.asarray(JTr.corewise_inner(v))
        self.assertTrue(np.allclose(np.asarray(lhs), rhs, atol=1e-9))

    def test_probe_transpose_adjoint(self):
        ww = _probe_vectors((2,))
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                v = _uniform_tangent(C, K)
                self._adjoint(v, lambda: v.probe(ww),
                              lambda r: ut3m.UT3Tangent.probe_transpose(r, ww, v.basis, sum_over_probes=True))
        with self.subTest('varying_C'):
            v = _varying_C_tangent()
            self._adjoint(v, lambda: v.probe(ww),
                          lambda r: ut3m.UT3Tangent.probe_transpose(r, ww, v.basis, sum_over_probes=True))

    def test_apply_transpose_adjoint(self):
        ww = _probe_vectors((2,))
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                v = _uniform_tangent(C, K)
                self._adjoint(v, lambda: v.apply(ww),
                              lambda r: ut3m.UT3Tangent.apply_transpose(r, ww, v.basis, sum_over_probes=True))
        with self.subTest('varying_C'):
            v = _varying_C_tangent()
            self._adjoint(v, lambda: v.apply(ww),
                          lambda r: ut3m.UT3Tangent.apply_transpose(r, ww, v.basis, sum_over_probes=True))

    def test_entries_transpose_adjoint(self):
        index = _index((2,))
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                v = _uniform_tangent(C, K)
                self._adjoint(v, lambda: v.entries(index),
                              lambda r: ut3m.UT3Tangent.entries_transpose(r, index, v.basis, sum_over_probes=True))
        with self.subTest('varying_C'):
            v = _varying_C_tangent()
            self._adjoint(v, lambda: v.entries(index),
                          lambda r: ut3m.UT3Tangent.entries_transpose(r, index, v.basis, sum_over_probes=True))

    def test_sum_over_probes_is_W_sum(self):
        # sum_over_probes=True == Σ_W (sum_over_probes=False): the kept-W result summed over its W axis
        ww = _probe_vectors((2,))
        for C, K in [((), ()), ((2,), ()), ((), (3,)), ((2,), (3,))]:
            with self.subTest(C=C, K=K):
                v = _uniform_tangent(C, K)
                zz = v.probe(ww)
                r = [np.random.randn(*z.shape) for z in zz]
                kept = ut3m.UT3Tangent.probe_transpose(r, ww, v.basis, sum_over_probes=False)   # stack W+K+C
                summ = ut3m.UT3Tangent.probe_transpose(r, ww, v.basis, sum_over_probes=True)     # stack K+C
                for k, s in zip(kept.variations.supercores, summ.variations.supercores):
                    self.assertTrue(np.allclose(k.sum(axis=1), s, atol=1e-9))   # sum the W axis (axis 1)

    def test_transpose_masks_are_gauge(self):
        # the transpose output carries the basis's gauge masks (broadcast over the new tangent stack)
        ww = _probe_vectors((2,))
        v = _uniform_tangent((), ())
        r = [np.random.randn(*z.shape) for z in v.probe(ww)]
        JTr = ut3m.UT3Tangent.probe_transpose(r, ww, v.basis, sum_over_probes=True)
        gauge = ubv.UT3Variations._variation_masks_of(v.basis)
        for got, exp in zip(JTr.variations.masks.data, gauge.data):
            self.assertTrue(np.array_equal(got, exp))


class TestUT3CorewiseTranspose(unittest.TestCase):
    """3b-6c: the corewise (non-manifold) sampling transposes on UniformTuckerTensorTrain -- the Section 6.3
    (P,Q,O)->G substitution. The raw gradient supercores match the ragged TuckerTensorTrain corewise
    gradients per stack element (in the real/masked region), across base stacks; sum_over_probes=True (the
    residual c is shape W+C, shared by both layers)."""

    def setUp(self):
        np.random.seed(3)

    def _xs(self, C):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=C)
        return x, ut3.UniformTuckerTensorTrain.from_t3(x)

    def _cmp(self, du, dg, rdu, rdg):
        # uniform grad supercores (d,)+C+(p,N) / (d,)+C+(rL,nU,rR); compare the real region to the ragged
        for i in range(len(_STRUCT[0])):
            self.assertTrue(np.allclose(
                np.asarray(du)[i][..., :rdu[i].shape[-2], :rdu[i].shape[-1]], np.asarray(rdu[i]), atol=1e-9))
            self.assertTrue(np.allclose(
                np.asarray(dg)[i][..., :rdg[i].shape[-3], :rdg[i].shape[-2], :rdg[i].shape[-1]],
                np.asarray(rdg[i]), atol=1e-9))

    def test_apply_corewise(self):
        ww = _probe_vectors((2,))
        for C in [(), (2,)]:
            with self.subTest(C=C):
                x, xu = self._xs(C)
                c = np.random.randn(*((2,) + C))                       # residual W+C
                self._cmp(*xu.apply_corewise_transpose(c, ww, sum_over_probes=True),
                          *x.apply_corewise_transpose(c, ww, sum_over_probes=True))

    def test_entries_corewise(self):
        index = _index((2,))
        for C in [(), (2,)]:
            with self.subTest(C=C):
                x, xu = self._xs(C)
                c = np.random.randn(*((2,) + C))
                self._cmp(*xu.entries_corewise_transpose(c, index, sum_over_probes=True),
                          *x.entries_corewise_transpose(c, index, sum_over_probes=True))

    def test_probe_corewise(self):
        ww = _probe_vectors((2,))
        for C in [(), (2,)]:
            with self.subTest(C=C):
                x, xu = self._xs(C)
                zt = [np.random.randn(*((2,) + C + (N,))) for N in _STRUCT[0]]
                self._cmp(*xu.probe_corewise_transpose(zt, ww, sum_over_probes=True),
                          *x.probe_corewise_transpose(zt, ww, sum_over_probes=True))


if __name__ == '__main__':
    unittest.main()
