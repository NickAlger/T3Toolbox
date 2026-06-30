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


if __name__ == '__main__':
    unittest.main()
