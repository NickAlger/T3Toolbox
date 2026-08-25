# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Direct tests for the tangent-variations stack converters and the oblique gauge projection (Phase D
of the 2026-08-22 review, finding R4-15; promoted from ``repros/R4/r4_02_tv_stacks.py`` and
``r4_03_oblique.py``): the four ``tv_{stack,unstack}_{tangent,frame}_stack`` converters at multi-axis
``K``/``C`` stacks, the frontend ``T3Tangent.stack_tangents``/``stack_frame`` round-trips, and the
projection properties of ``tv_oblique_gauge_projection``."""
import unittest
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
import t3toolbox.backend.tv_operations as tv

_STRUCT = ((5, 6, 7), (3, 2, 4), (1, 2, 3, 1))               # asymmetric everywhere
_NONMIN = ((6, 6, 6), (4, 4, 4), (1, 2, 2, 1))               # over-ranked Tucker: nU != nD slack


def _slice_tree(data, idx):
    if isinstance(data, (tuple, list)):
        return tuple(_slice_tree(c, idx) for c in data)
    return data[idx]


def _leaf(tree, idx):
    for i in idx:
        tree = tree[i]
    return tree


def _tree_relerr(a, b):
    return float(cw.corewise_norm(cw.corewise_sub(a, b))) / max(float(cw.corewise_norm(b)), 1e-300)


class TestStackConverters(unittest.TestCase):
    """The four converters at the R4-15 combo ``K=(2, 3)``, ``C=(2, 2)`` (multi-axis both), plus the
    single-axis and empty edges, each leaf checked against a manual slice."""

    def setUp(self):
        np.random.seed(0)

    def _fv(self, K, C):
        x = t3.TuckerTensorTrain.randn(*_STRUCT, stack_shape=C)
        frame = t3m.MANIFOLD.frame(x)
        v = t3m.COREWISE.randn(frame, stack_shape=K)
        return frame, v

    def test_tangent_stack_roundtrip_and_leaves(self):
        for K, C in [((2, 3), (2, 2)), ((2,), ()), ((2, 3), ())]:
            with self.subTest(K=K, C=C):
                frame, v = self._fv(K, C)
                fd, vd = frame.data, v.variations.data
                tree = tv.tv_unstack_tangent_stack(fd, vd)       # K-shaped tree of variations (stack C)
                for kk in np.ndindex(*K):
                    self.assertLess(_tree_relerr(_slice_tree(vd, kk), _leaf(tree, kk)), 1e-15)
                self.assertLess(_tree_relerr(vd, tv.tv_stack_tangent_stack(tree)), 1e-15)
                # frontend round-trip
                v2 = t3m.T3Tangent.stack_tangents(v.unstack_tangents())
                self.assertLess(_tree_relerr(vd, v2.variations.data), 1e-15)

    def test_frame_stack_roundtrip_and_leaves(self):
        for K, C in [((2, 3), (2, 2)), ((), (3,)), ((2,), (2, 2))]:
            with self.subTest(K=K, C=C):
                frame, v = self._fv(K, C)
                fd, vd = frame.data, v.variations.data
                tree = tv.tv_unstack_frame_stack(fd, vd)         # C-shaped tree of (frame, vars) pairs
                for cc in np.ndindex(*C):
                    fl, vl = _leaf(tree, cc)
                    self.assertLess(_tree_relerr(_slice_tree(fd, cc), fl), 1e-15)
                    # the variations leaf is the K-stacked slice at c (C peeled from the INNER stack)
                    ref_vl = tuple(tuple(
                        np.moveaxis(c_, list(range(len(K), len(K) + len(C))), list(range(len(C))))[cc]
                        for c_ in fam) for fam in vd)
                    self.assertLess(_tree_relerr(ref_vl, vl), 1e-15)
                fd2, vd2 = tv.tv_stack_frame_stack(tree)
                self.assertLess(_tree_relerr(fd, fd2), 1e-15)
                self.assertLess(_tree_relerr(vd, vd2), 1e-15)
                # frontend round-trip
                v3 = t3m.T3Tangent.stack_frame(v.unstack_frame())
                self.assertLess(_tree_relerr(vd, v3.variations.data), 1e-15)
                self.assertLess(_tree_relerr(fd, v3.frame.data), 1e-15)

    def test_frame_stack_empty_C_is_identity(self):
        frame, v = self._fv((2,), ())
        fd, vd = frame.data, v.variations.data
        fl, vl = tv.tv_unstack_frame_stack(fd, vd)               # C=(): a single (frame, vars) pair
        self.assertLess(_tree_relerr(fd, fl) + _tree_relerr(vd, vl), 1e-15)
        fd2, vd2 = tv.tv_stack_frame_stack((fl, vl))
        self.assertLess(_tree_relerr(fd, fd2) + _tree_relerr(vd, vd2), 1e-15)
        self.assertIsInstance(v.unstack_frame(), t3m.T3Tangent)  # frontend: C=() unstack is the tangent


class TestObliqueGaugeProjection(unittest.TestCase):
    """R4-15 / ``r4_03``: ``tv_oblique_gauge_projection`` is a projection onto the gauged subspace
    ALONG the vertical (pure-gauge) directions: it preserves the represented tangent, lands gauged,
    is idempotent, annihilates vertical variations, and fixes already-gauged input."""

    def setUp(self):
        np.random.seed(1)

    def test_projection_properties(self):
        for struct in [_STRUCT, _NONMIN]:
            for K, C in [((), ()), ((2,), ()), ((), (3,)), ((2,), (3,))]:
                with self.subTest(struct=struct, K=K, C=C):
                    shape, tr, ttr = struct
                    d = len(shape)
                    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
                    frame = t3m.MANIFOLD.frame(x)
                    v = t3m.COREWISE.randn(frame, stack_shape=K)
                    fd, vd = frame.data, v.variations.data
                    ob = tv.tv_oblique_gauge_projection(fd, vd)
                    dense = tv.tv_to_dense(fd, vd)
                    self.assertLess(float(np.linalg.norm(tv.tv_to_dense(fd, ob) - dense)),
                                    1e-9 * np.linalg.norm(dense))                       # preserves
                    self.assertLess(float(np.max(tv.tv_gauge_residual(fd, ob))), 1e-9)  # gauged
                    self.assertLess(_tree_relerr(ob, tv.tv_oblique_gauge_projection(fd, ob)), 1e-9)
                    og = tv.tv_orthogonal_gauge_projection(fd, vd)
                    self.assertLess(_tree_relerr(og, tv.tv_oblique_gauge_projection(fd, og)), 1e-9)
                    # a vertical (pure-gauge) variation represents zero and is annihilated
                    U, O, L, R = fd
                    VV = [np.zeros_like(c) for c in vd[0]]
                    HH = [np.zeros_like(c) for c in vd[1]]
                    for i in range(d):
                        X = np.random.randn(*(K + C + (O[i].shape[-2], U[i].shape[-2])))
                        VV[i] = VV[i] + np.einsum('...ji,...io->...jo', X, U[i])
                        HH[i] = HH[i] - np.einsum('...aib,...ij->...ajb', O[i], X)
                    for i in range(d - 1):
                        Y = np.random.randn(*(K + C + (L[i].shape[-1], R[i + 1].shape[-3])))
                        HH[i] = HH[i] + np.einsum('...iaj,...jk->...iak', L[i], Y)
                        HH[i + 1] = HH[i + 1] - np.einsum('...jk,...kbl->...jbl', Y, R[i + 1])
                    gd = (tuple(VV), tuple(HH))
                    gnorm = float(cw.corewise_norm(gd))
                    self.assertLess(float(np.max(np.abs(tv.tv_to_dense(fd, gd)))), 1e-9 * max(gnorm, 1.0))
                    self.assertLess(float(cw.corewise_norm(tv.tv_oblique_gauge_projection(fd, gd))),
                                    1e-9 * gnorm)
