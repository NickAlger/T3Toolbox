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

# forced padding STRICTLY above the real max ranks of _STRUCT (nU=3, rL=rR=2, N=6) so EVERY core has a
# padded region (the default from_t3 pads to max(ranks), leaving the max-rank cores clean) -- 3b-6d (E)
_PAD_T3 = dict(N=8, n=5, r=4)


def _uniform_tangent(C=(), K=(), force_pad=False, seed=0):
    """A random uniform tangent at an orthogonal frame (forward 𝒥 is linear in the variation, so the
    corewise -- ungauged -- random tangent is a fine test input; gauge is irrelevant to 𝒥)."""
    np.random.seed(seed)
    x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=C)
    xu = ut3.UniformTuckerTensorTrain.from_t3(x, **_PAD_T3) if force_pad \
        else ut3.UniformTuckerTensorTrain.from_t3(x)
    return ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.base(xu), stack_shape=K)


def _corrupt(obj, scale=1e3):
    """Add ``scale`` * garbage to ``obj``'s masked-out (padding) region; the real region is unchanged.
    A correct (mask-once) probing op must be UNAFFECTED. ``obj``: UT3Basis / UT3Variations / UniformT3."""
    scs = obj.supercores
    ind = type(obj)(*([np.ones_like(s) for s in scs] + [obj.shape, obj.masks])).apply_masks().supercores
    new = [sc + scale * (1.0 - i) for sc, i in zip(scs, ind)]
    return type(obj)(*(new + [obj.shape, obj.masks]))


def _corrupt_tangent(v, scale=1e3):
    return ut3m.UT3Tangent(_corrupt(v.basis, scale), _corrupt(v.variations, scale))


def _prefix(ranks, size):  # int ranks -> boolean prefix mask of width `size` (canonical form)
    return np.arange(size) < np.asarray(ranks)[..., None]


def _bc_over_K(m, K):  # (d,)+C+(size,) -> (d,)+K+C+(size,)
    return np.broadcast_to(m.reshape(m.shape[:1] + (1,) * len(K) + m.shape[1:]),
                           m.shape[:1] + tuple(K) + m.shape[1:])


def _expected_gauge_masks(basis, K_new=()):
    """The variation gauge masks built INDEPENDENTLY from the base ranks + the gauge rule (prefix; the
    left/right boundary-shifted [:-1] / [1:]), broadcast over the new tangent stack -- a different
    derivation than the impl's slice of the stored masks, so the comparison catches a boundary slip."""
    up = _prefix(np.asarray(basis.up_ranks), basis.nU)
    down = _prefix(np.asarray(basis.down_ranks), basis.nD)
    left = _prefix(np.asarray(basis.left_ranks)[:-1], basis.rL)
    right = _prefix(np.asarray(basis.right_ranks)[1:], basis.rR)
    return tuple(_bc_over_K(m, K_new) for m in (up, down, left, right))


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


_ORDER = 2   # highest derivative order for the jet tests (>=2 exercises the binomial convolution)


def _pert_vectors(W, seed=3):
    np.random.seed(seed)
    return [np.random.randn(*(tuple(W) + (N,))) for N in _STRUCT[0]]


class TestUT3TangentForwardDerivatives(unittest.TestCase):
    """3b-6'b: the forward Jacobian DERIVATIVES 𝒥 (probe/apply/entries_derivatives on UT3Tangent), per
    stack element vs the ragged T3Tangent over the _CONFIGS matrix + varying-C. The output carries a
    leading derivative-order axis (order + W + K + C [+ Ni]); order 0 == the plain forward sample."""

    def _leaves(self, v):
        return [leaf.to_t3tangent() for leaf in _full_unstack(v)]

    def _check_probe(self, v, ww, pp, W):
        zz = v.probe_derivatives(ww, pp, _ORDER)
        nW = int(np.prod(W)) if W else 1
        O = _ORDER + 1
        for i, leaf in enumerate(self._leaves(v)):
            rzz = leaf.probe_derivatives(ww, pp, _ORDER)
            for m in range(len(rzz)):
                Nm = rzz[m].shape[-1]
                u = np.asarray(zz[m]).reshape((O, nW, -1, Nm))[:, :, i, :]
                self.assertTrue(np.allclose(u, np.asarray(rzz[m]).reshape((O, nW, Nm)), atol=1e-8),
                                msg='probe_deriv mode %d, element %d' % (m, i))
            # order 0 == plain probe
            z0 = leaf.probe(ww)
            for m in range(len(rzz)):
                self.assertTrue(np.allclose(np.asarray(rzz[m])[0], np.asarray(z0[m]), atol=1e-9))

    def _check_apply(self, v, ww, pp, W):
        aa = np.asarray(v.apply_derivatives(ww, pp, _ORDER))
        nW, O = (int(np.prod(W)) if W else 1), _ORDER + 1
        for i, leaf in enumerate(self._leaves(v)):
            u = aa.reshape((O, nW, -1))[:, :, i]
            self.assertTrue(np.allclose(u, np.asarray(leaf.apply_derivatives(ww, pp, _ORDER)).reshape((O, nW)),
                                        atol=1e-8), msg='apply_deriv element %d' % i)

    def _check_entries(self, v, index, pp, W):
        ee = np.asarray(v.entries_derivatives(index, pp, _ORDER))
        nW, O = (int(np.prod(W)) if W else 1), _ORDER + 1
        for i, leaf in enumerate(self._leaves(v)):
            u = ee.reshape((O, nW, -1))[:, :, i]
            self.assertTrue(np.allclose(u, np.asarray(leaf.entries_derivatives(index, pp, _ORDER)).reshape((O, nW)),
                                        atol=1e-8), msg='entries_deriv element %d' % i)

    def test_probe(self):
        W = (2,); ww, pp = _probe_vectors(W), _pert_vectors(W)
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                self._check_probe(_uniform_tangent(C, K), ww, pp, W)
        with self.subTest('varying_C'):
            self._check_probe(_varying_C_tangent(), ww, pp, W)

    def test_apply(self):
        W = (2,); ww, pp = _probe_vectors(W), _pert_vectors(W)
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                self._check_apply(_uniform_tangent(C, K), ww, pp, W)
        with self.subTest('varying_C'):
            self._check_apply(_varying_C_tangent(), ww, pp, W)

    def test_entries(self):
        W = (2,); index, pp = _index(W), _pert_vectors(W)
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                self._check_entries(_uniform_tangent(C, K), index, pp, W)
        with self.subTest('varying_C'):
            self._check_entries(_varying_C_tangent(), index, pp, W)

    def test_no_probe_stack(self):
        ww, pp, index = _probe_vectors(()), _pert_vectors(()), _index(())
        for C, K in [((), ()), ((2,), ()), ((), (3,)), ((2,), (3,))]:
            with self.subTest(C=C, K=K):
                v = _uniform_tangent(C, K)
                self._check_apply(v, ww, pp, ())
                self._check_entries(v, index, pp, ())
                self._check_probe(v, ww, pp, ())


class TestUniformT3PlainDerivatives(unittest.TestCase):
    """3b-6'b: the PLAIN (non-tangent) forward derivatives (UniformTuckerTensorTrain.{probe,apply,entries}
    _derivatives) vs the ragged TuckerTensorTrain, per base-stack (C) element. Output order + W + C [+ Ni]."""

    def _models(self, C, force_pad=False):
        np.random.seed(5)
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=C)
        xu = ut3.UniformTuckerTensorTrain.from_t3(x, **_PAD_T3) if force_pad \
            else ut3.UniformTuckerTensorTrain.from_t3(x)
        return x, xu

    def test_probe(self):
        W = (2,); ww, pp = _probe_vectors(W), _pert_vectors(W)
        for C in [(), (2,), (2, 3)]:
            for fp in [False, True]:
                with self.subTest(C=C, force_pad=fp):
                    x, xu = self._models(C, fp)
                    zr, zu = x.probe_derivatives(ww, pp, _ORDER), xu.probe_derivatives(ww, pp, _ORDER)
                    for m in range(len(zr)):
                        self.assertTrue(np.allclose(np.asarray(zu[m]), np.asarray(zr[m]), atol=1e-9))

    def test_apply_entries(self):
        W = (2,); ww, pp, index = _probe_vectors(W), _pert_vectors(W), _index(W)
        for C in [(), (2,), (2, 3)]:
            for fp in [False, True]:
                with self.subTest(C=C, force_pad=fp):
                    x, xu = self._models(C, fp)
                    self.assertTrue(np.allclose(np.asarray(xu.apply_derivatives(ww, pp, _ORDER)),
                                                np.asarray(x.apply_derivatives(ww, pp, _ORDER)), atol=1e-9))
                    self.assertTrue(np.allclose(np.asarray(xu.entries_derivatives(index, pp, _ORDER)),
                                                np.asarray(x.entries_derivatives(index, pp, _ORDER)), atol=1e-9))


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


class TestUT3TangentTransposeDerivatives(unittest.TestCase):
    """3b-6'c: the transpose DERIVATIVES 𝒥ᵀ (probe/apply/entries_derivatives_transpose on UT3Tangent). The
    residual jets carry the order axis; the transpose sums it into a SINGLE variation gradient. Verified by
    the adjoint identity ``<r, 𝒥V> = <𝒥ᵀr, V>`` (the measurement dot now sums the order axis too) over the
    _CONFIGS matrix + varying-C, plus the Σ_W(sum=False)==sum=True relation. Probe stack W = (2,)."""

    def setUp(self):
        np.random.seed(2)

    def _meas_dot(self, r, y, nW):
        # measurement inner product summed over the ORDER axis (0) + the probe stack W (1..1+nW), keeping
        # K+C: probe r/y are d vectors (also sum the mode -1); apply/entries r/y are single arrays.
        sax = tuple(range(1 + nW))
        if isinstance(y, (tuple, list)):
            return sum((np.asarray(a) * np.asarray(b)).sum(axis=sax + (-1,)) for a, b in zip(r, y))
        return (np.asarray(r) * np.asarray(y)).sum(axis=sax)

    def _adjoint(self, v, forward, transpose, nW=1):
        y = forward()                                          # 𝒥V (order-stacked)
        r = [np.random.randn(*np.asarray(a).shape) for a in y] if isinstance(y, (tuple, list)) \
            else np.random.randn(*np.asarray(y).shape)
        JTr = transpose(r)                                     # 𝒥ᵀr (sum_over_probes=True -> tangent stack K)
        lhs = self._meas_dot(r, y, nW)
        rhs = np.asarray(JTr.corewise_inner(v))
        self.assertTrue(np.allclose(np.asarray(lhs), rhs, atol=1e-8))

    def test_probe_transpose_adjoint(self):
        ww, pp = _probe_vectors((2,)), _pert_vectors((2,))
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                v = _uniform_tangent(C, K)
                self._adjoint(v, lambda: v.probe_derivatives(ww, pp, _ORDER),
                              lambda r: ut3m.UT3Tangent.probe_derivatives_transpose(r, ww, pp, v.basis, _ORDER, sum_over_probes=True))
        with self.subTest('varying_C'):
            v = _varying_C_tangent()
            self._adjoint(v, lambda: v.probe_derivatives(ww, pp, _ORDER),
                          lambda r: ut3m.UT3Tangent.probe_derivatives_transpose(r, ww, pp, v.basis, _ORDER, sum_over_probes=True))

    def test_apply_transpose_adjoint(self):
        ww, pp = _probe_vectors((2,)), _pert_vectors((2,))
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                v = _uniform_tangent(C, K)
                self._adjoint(v, lambda: v.apply_derivatives(ww, pp, _ORDER),
                              lambda r: ut3m.UT3Tangent.apply_derivatives_transpose(r, ww, pp, v.basis, _ORDER, sum_over_probes=True))
        with self.subTest('varying_C'):
            v = _varying_C_tangent()
            self._adjoint(v, lambda: v.apply_derivatives(ww, pp, _ORDER),
                          lambda r: ut3m.UT3Tangent.apply_derivatives_transpose(r, ww, pp, v.basis, _ORDER, sum_over_probes=True))

    def test_entries_transpose_adjoint(self):
        index, pp = _index((2,)), _pert_vectors((2,))
        for C, K in _CONFIGS:
            with self.subTest(C=C, K=K):
                v = _uniform_tangent(C, K)
                self._adjoint(v, lambda: v.entries_derivatives(index, pp, _ORDER),
                              lambda r: ut3m.UT3Tangent.entries_derivatives_transpose(r, index, pp, v.basis, _ORDER, sum_over_probes=True))
        with self.subTest('varying_C'):
            v = _varying_C_tangent()
            self._adjoint(v, lambda: v.entries_derivatives(index, pp, _ORDER),
                          lambda r: ut3m.UT3Tangent.entries_derivatives_transpose(r, index, pp, v.basis, _ORDER, sum_over_probes=True))

    def test_sum_over_probes_is_W_sum(self):
        # sum_over_probes=True == Σ_W (sum_over_probes=False): the kept-W gradient summed over its W axis
        ww, pp = _probe_vectors((2,)), _pert_vectors((2,))
        for C, K in [((), ()), ((2,), ()), ((), (3,)), ((2,), (3,))]:
            with self.subTest(C=C, K=K):
                v = _uniform_tangent(C, K)
                r = [np.random.randn(*z.shape) for z in v.probe_derivatives(ww, pp, _ORDER)]
                kept = ut3m.UT3Tangent.probe_derivatives_transpose(r, ww, pp, v.basis, _ORDER, sum_over_probes=False)
                summ = ut3m.UT3Tangent.probe_derivatives_transpose(r, ww, pp, v.basis, _ORDER, sum_over_probes=True)
                for k, s in zip(kept.variations.supercores, summ.variations.supercores):
                    self.assertTrue(np.allclose(k.sum(axis=1), s, atol=1e-8))   # sum the W axis (axis 1)


class TestUT3CorewiseTransposeDerivatives(unittest.TestCase):
    """3b-6'c: the corewise (non-manifold) DERIVATIVE transposes on UniformTuckerTensorTrain (the §6.3
    substitution into the tangent derivative transpose). Raw gradient supercores match the ragged per stack
    element (real region), across base stacks; sum_over_probes=True (residual jets shape (order+1)+W+C[+Ni])."""

    def setUp(self):
        np.random.seed(3)

    def _xs(self, C):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=C)
        return x, ut3.UniformTuckerTensorTrain.from_t3(x)

    def _cmp(self, du, dg, rdu, rdg):  # compare the real region of the uniform grad supercores to the ragged
        for i in range(len(_STRUCT[0])):
            self.assertTrue(np.allclose(
                np.asarray(du)[i][..., :rdu[i].shape[-2], :rdu[i].shape[-1]], np.asarray(rdu[i]), atol=1e-8))
            self.assertTrue(np.allclose(
                np.asarray(dg)[i][..., :rdg[i].shape[-3], :rdg[i].shape[-2], :rdg[i].shape[-1]],
                np.asarray(rdg[i]), atol=1e-8))

    def test_apply_corewise(self):
        ww, pp = _probe_vectors((2,)), _pert_vectors((2,))
        for C in [(), (2,)]:
            with self.subTest(C=C):
                x, xu = self._xs(C)
                c = np.random.randn(*((_ORDER + 1, 2) + C))            # residual jet (order+1)+W+C
                self._cmp(*xu.apply_corewise_derivatives_transpose(c, ww, pp, _ORDER, sum_over_probes=True),
                          *x.apply_corewise_derivatives_transpose(c, ww, pp, _ORDER, sum_over_probes=True))

    def test_entries_corewise(self):
        index, pp = _index((2,)), _pert_vectors((2,))
        for C in [(), (2,)]:
            with self.subTest(C=C):
                x, xu = self._xs(C)
                c = np.random.randn(*((_ORDER + 1, 2) + C))
                self._cmp(*xu.entries_corewise_derivatives_transpose(c, index, pp, _ORDER, sum_over_probes=True),
                          *x.entries_corewise_derivatives_transpose(c, index, pp, _ORDER, sum_over_probes=True))

    def test_probe_corewise(self):
        ww, pp = _probe_vectors((2,)), _pert_vectors((2,))
        for C in [(), (2,)]:
            with self.subTest(C=C):
                x, xu = self._xs(C)
                zt = [np.random.randn(*((_ORDER + 1, 2) + C + (N,))) for N in _STRUCT[0]]  # residual jets
                self._cmp(*xu.probe_corewise_derivatives_transpose(zt, ww, pp, _ORDER, sum_over_probes=True),
                          *x.probe_corewise_derivatives_transpose(zt, ww, pp, _ORDER, sum_over_probes=True))


class TestUT3ProbingHardening(unittest.TestCase):
    """3b-6d: mask-strict + garbage-robust hardening of the uniform probing path (per
    docs/testing_strategy.md). Dense/numerical tests on clean padding are blind to too-permissive masks;
    these close that with (A) garbage-padded inputs -- mask-once must make every op's output UNCHANGED
    (clean == dirty, since the garbage contracts to zero) -- and (B) exact transpose output masks derived
    independently from the base ranks. Forced padding (E) exercises masking on every core."""

    def setUp(self):
        np.random.seed(4)

    @staticmethod
    def _equal_supercores(a, b):
        return all(np.allclose(np.asarray(x), np.asarray(y), atol=1e-9) for x, y in zip(a, b))

    def test_forward_garbage_robust(self):
        ww = _probe_vectors((2,))
        index = _index((2,))
        for C, K, fp in [((), (), False), ((2,), (3,), False), ((), (), True), ((2, 3), (), False)]:
            with self.subTest(C=C, K=K, fp=fp):
                v = _uniform_tangent(C, K, force_pad=fp)
                d = _corrupt_tangent(v)
                self.assertTrue(self._equal_supercores(v.probe(ww), d.probe(ww)))
                self.assertTrue(np.allclose(np.asarray(v.apply(ww)), np.asarray(d.apply(ww)), atol=1e-9))
                self.assertTrue(np.allclose(np.asarray(v.entries(index)), np.asarray(d.entries(index)), atol=1e-9))

    def test_transpose_garbage_robust(self):
        ww = _probe_vectors((2,))
        index = _index((2,))
        for C, K, fp in [((), (), False), ((2,), (3,), False), ((), (), True)]:
            with self.subTest(C=C, K=K, fp=fp):
                v = _uniform_tangent(C, K, force_pad=fp)
                bd = _corrupt(v.basis)                                    # corrupt the basis padding only
                r = [np.random.randn(*z.shape) for z in v.probe(ww)]
                c = np.random.randn(*np.asarray(v.apply(ww)).shape)
                pairs = [(ut3m.UT3Tangent.probe_transpose, (r, ww)),
                         (ut3m.UT3Tangent.apply_transpose, (c, ww)),
                         (ut3m.UT3Tangent.entries_transpose, (c, index))]
                for op, args in pairs:
                    clean = op(*args, v.basis, sum_over_probes=True)
                    dirty = op(*args, bd, sum_over_probes=True)
                    self.assertTrue(self._equal_supercores(clean.variations.supercores, dirty.variations.supercores))
                    for ma, mb in zip(clean.variations.masks.data, dirty.variations.masks.data):
                        self.assertTrue(np.array_equal(ma, mb))

    def test_transpose_exact_masks(self):
        ww = _probe_vectors((2,))
        index = _index((2,))
        for C, K, fp in [((), (), False), ((2,), (3,), False), ((), (), True), ((2, 3), (), False), ((), (2, 3), False)]:
            with self.subTest(C=C, K=K, fp=fp):
                v = _uniform_tangent(C, K, force_pad=fp)
                r = [np.random.randn(*z.shape) for z in v.probe(ww)]
                c = np.random.randn(*np.asarray(v.apply(ww)).shape)
                for sop in (True, False):
                    K_new = tuple(K) if sop else (2,) + tuple(K)         # sum: K ; keep: W + K (W = (2,))
                    exp = _expected_gauge_masks(v.basis, K_new)
                    outs = [ut3m.UT3Tangent.probe_transpose(r, ww, v.basis, sum_over_probes=sop),
                            ut3m.UT3Tangent.apply_transpose(c, ww, v.basis, sum_over_probes=sop),
                            ut3m.UT3Tangent.entries_transpose(c, index, v.basis, sum_over_probes=sop)]
                    for JT in outs:
                        for got, e in zip(JT.variations.masks.data, exp):
                            self.assertTrue(np.array_equal(got, e), msg='sop=%s' % sop)

    def test_corewise_garbage_robust(self):
        ww = _probe_vectors((2,))
        index = _index((2,))
        for C, fp in [((), False), ((2,), False), ((), True)]:
            with self.subTest(C=C, fp=fp):
                x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=C)
                xu = ut3.UniformTuckerTensorTrain.from_t3(x, **_PAD_T3) if fp \
                    else ut3.UniformTuckerTensorTrain.from_t3(x)
                xd = _corrupt(xu)
                c = np.random.randn(*((2,) + C))
                zt = [np.random.randn(*((2,) + C + (N,))) for N in _STRUCT[0]]
                self.assertTrue(self._equal_supercores(xu.apply_corewise_transpose(c, ww, sum_over_probes=True),
                                                       xd.apply_corewise_transpose(c, ww, sum_over_probes=True)))
                self.assertTrue(self._equal_supercores(xu.entries_corewise_transpose(c, index, sum_over_probes=True),
                                                       xd.entries_corewise_transpose(c, index, sum_over_probes=True)))
                self.assertTrue(self._equal_supercores(xu.probe_corewise_transpose(zt, ww, sum_over_probes=True),
                                                       xd.probe_corewise_transpose(zt, ww, sum_over_probes=True)))


class TestUT3DerivativeHardening(unittest.TestCase):
    """3b-6'd: mask-strict + garbage-robust hardening of the uniform DERIVATIVE probing path (per
    docs/testing_strategy.md), the jet twin of TestUT3ProbingHardening. Same three guards -- (A)
    garbage-padded inputs leave every derivative op UNCHANGED, (B) exact transpose output masks (the
    derivative gradient carries no order axis, so its masks are the plain gauge masks over K_new), (C)
    forced padding on every core. The perturbation pp is a second garbage-robustness surface."""

    def setUp(self):
        np.random.seed(4)

    @staticmethod
    def _equal_supercores(a, b):
        return all(np.allclose(np.asarray(x), np.asarray(y), atol=1e-8) for x, y in zip(a, b))

    def test_forward_garbage_robust(self):
        ww, pp, index = _probe_vectors((2,)), _pert_vectors((2,)), _index((2,))
        for C, K, fp in [((), (), False), ((2,), (3,), False), ((), (), True), ((2, 3), (), False)]:
            with self.subTest(C=C, K=K, fp=fp):
                v = _uniform_tangent(C, K, force_pad=fp)
                d = _corrupt_tangent(v)
                self.assertTrue(self._equal_supercores(v.probe_derivatives(ww, pp, _ORDER),
                                                       d.probe_derivatives(ww, pp, _ORDER)))
                self.assertTrue(np.allclose(np.asarray(v.apply_derivatives(ww, pp, _ORDER)),
                                            np.asarray(d.apply_derivatives(ww, pp, _ORDER)), atol=1e-8))
                self.assertTrue(np.allclose(np.asarray(v.entries_derivatives(index, pp, _ORDER)),
                                            np.asarray(d.entries_derivatives(index, pp, _ORDER)), atol=1e-8))

    def test_transpose_garbage_robust(self):
        ww, pp, index = _probe_vectors((2,)), _pert_vectors((2,)), _index((2,))
        for C, K, fp in [((), (), False), ((2,), (3,), False), ((), (), True)]:
            with self.subTest(C=C, K=K, fp=fp):
                v = _uniform_tangent(C, K, force_pad=fp)
                bd = _corrupt(v.basis)                                    # corrupt the basis padding only
                r = [np.random.randn(*z.shape) for z in v.probe_derivatives(ww, pp, _ORDER)]
                c = np.random.randn(*np.asarray(v.apply_derivatives(ww, pp, _ORDER)).shape)
                T = ut3m.UT3Tangent
                pairs = [(T.probe_derivatives_transpose, (r, ww, pp)),
                         (T.apply_derivatives_transpose, (c, ww, pp)),
                         (T.entries_derivatives_transpose, (c, index, pp))]
                for op, args in pairs:
                    clean = op(*args, v.basis, _ORDER, sum_over_probes=True)
                    dirty = op(*args, bd, _ORDER, sum_over_probes=True)
                    self.assertTrue(self._equal_supercores(clean.variations.supercores, dirty.variations.supercores))
                    for ma, mb in zip(clean.variations.masks.data, dirty.variations.masks.data):
                        self.assertTrue(np.array_equal(ma, mb))

    def test_transpose_exact_masks(self):
        ww, pp, index = _probe_vectors((2,)), _pert_vectors((2,)), _index((2,))
        for C, K, fp in [((), (), False), ((2,), (3,), False), ((), (), True), ((2, 3), (), False), ((), (2, 3), False)]:
            with self.subTest(C=C, K=K, fp=fp):
                v = _uniform_tangent(C, K, force_pad=fp)
                r = [np.random.randn(*z.shape) for z in v.probe_derivatives(ww, pp, _ORDER)]
                c = np.random.randn(*np.asarray(v.apply_derivatives(ww, pp, _ORDER)).shape)
                T = ut3m.UT3Tangent
                for sop in (True, False):
                    K_new = tuple(K) if sop else (2,) + tuple(K)         # the gradient has no order axis
                    exp = _expected_gauge_masks(v.basis, K_new)
                    outs = [T.probe_derivatives_transpose(r, ww, pp, v.basis, _ORDER, sum_over_probes=sop),
                            T.apply_derivatives_transpose(c, ww, pp, v.basis, _ORDER, sum_over_probes=sop),
                            T.entries_derivatives_transpose(c, index, pp, v.basis, _ORDER, sum_over_probes=sop)]
                    for JT in outs:
                        for got, e in zip(JT.variations.masks.data, exp):
                            self.assertTrue(np.array_equal(got, e), msg='sop=%s' % sop)

    def test_corewise_garbage_robust(self):
        ww, pp, index = _probe_vectors((2,)), _pert_vectors((2,)), _index((2,))
        for C, fp in [((), False), ((2,), False), ((), True)]:
            with self.subTest(C=C, fp=fp):
                x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=C)
                xu = ut3.UniformTuckerTensorTrain.from_t3(x, **_PAD_T3) if fp \
                    else ut3.UniformTuckerTensorTrain.from_t3(x)
                xd = _corrupt(xu)
                c = np.random.randn(*((_ORDER + 1, 2) + C))
                zt = [np.random.randn(*((_ORDER + 1, 2) + C + (N,))) for N in _STRUCT[0]]
                self.assertTrue(self._equal_supercores(
                    xu.apply_corewise_derivatives_transpose(c, ww, pp, _ORDER, sum_over_probes=True),
                    xd.apply_corewise_derivatives_transpose(c, ww, pp, _ORDER, sum_over_probes=True)))
                self.assertTrue(self._equal_supercores(
                    xu.entries_corewise_derivatives_transpose(c, index, pp, _ORDER, sum_over_probes=True),
                    xd.entries_corewise_derivatives_transpose(c, index, pp, _ORDER, sum_over_probes=True)))
                self.assertTrue(self._equal_supercores(
                    xu.probe_corewise_derivatives_transpose(zt, ww, pp, _ORDER, sum_over_probes=True),
                    xd.probe_corewise_derivatives_transpose(zt, ww, pp, _ORDER, sum_over_probes=True)))


if __name__ == '__main__':
    unittest.main()
