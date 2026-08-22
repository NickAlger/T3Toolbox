# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
"""Tests for the weighted tensor-network layer (S1: T3Weights + absorb + from_t3svd + is_consistent).

Correctness is against a dense ground-truth hand-einsum with the weights inserted, checked across
structures x stack_shapes (including non-trivial, non-square stacks -- what actually exposes an axis
mistake), plus the algebraic identities (all-ones absorb = identity; absorb W then 1/W = x)."""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw


# minimal-rank structures so t3svd preserves ranks (from_t3svd stays consistent with x)
STRUCTURES = [
    ((6, 7, 8), (2, 2, 2), (1, 2, 2, 1)),
    ((5, 6, 7, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1)),
    ((9,), (1,), (1, 1)),   # d=1 edge case (the single core absorbs both boundary bonds); minimal
]
STACKS = [(), (4,), (2, 3)]


def rand_weights(x, rng):
    """A random T3Weights shape-consistent with x."""
    ss = x.stack_shape
    return t3.T3Weights(tuple(rng.standard_normal(ss + (n,)) for n in x.tucker_ranks),
                        tuple(rng.standard_normal(ss + (r,)) for r in x.tt_ranks))


def hand_weighted_dense(x, W):
    """Dense value of the weighted network via an independent hand-einsum (weights inserted on edges)."""
    d = x.d
    tk, tt = x.data
    tw, ttw = W.data
    pool = iter('abcdefghijklmnopqrstuvwxyz')
    out = [next(pool) for _ in range(d)]        # output modes
    rank = [next(pool) for _ in range(d)]       # Tucker ranks (shared: tucker core, tt core, tucker weight)
    bond = [next(pool) for _ in range(d + 1)]   # TT bonds (shared: adjacent tt cores, tt weight)
    terms, ops = [], []
    for k in range(d):
        terms.append('...' + rank[k] + out[k]); ops.append(tk[k])
    for k in range(d):
        terms.append('...' + bond[k] + rank[k] + bond[k + 1]); ops.append(tt[k])
    for k in range(d):
        terms.append('...' + rank[k]); ops.append(tw[k])
    for k in range(d + 1):
        terms.append('...' + bond[k]); ops.append(ttw[k])
    return np.einsum(','.join(terms) + '->...' + ''.join(out), *ops)


def hadamard_cores(xA, xB):
    """Hadamard (elementwise-product) combine of two T3s' UNWEIGHTED cores (physical output shared,
    internal legs Kronecker) -- the core partner of T3Weights.kronecker, done here in the test."""
    ss = xA.stack_shape
    Uc = tuple(np.einsum('...ix,...jx->...ijx', a, b).reshape(ss + (a.shape[-2] * b.shape[-2], a.shape[-1]))
               for a, b in zip(xA.data[0], xB.data[0]))
    Gc = tuple(np.einsum('...aib,...cjd->...acijbd', a, b).reshape(
                   ss + (a.shape[-3] * b.shape[-3], a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1]))
               for a, b in zip(xA.data[1], xB.data[1]))
    return t3.TuckerTensorTrain(Uc, Gc)


class TestT3Weights(unittest.TestCase):
    def test_absorb_dense_oracle(self):
        """absorb_weights(x, W).to_dense() == the weights-inserted hand-einsum, across structures x stacks."""
        rng = np.random.default_rng(0)
        for struct in STRUCTURES:
            for ss in STACKS:
                with self.subTest(struct=struct, stack=ss):
                    x = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss)
                    W = rand_weights(x, rng)
                    xw = t3.t3_absorb_weights(x, W)
                    self.assertEqual(xw.ranks, x.ranks)                       # shape-preserving
                    ref = hand_weighted_dense(x, W)
                    self.assertLess(np.linalg.norm(xw.to_dense() - ref) / max(np.linalg.norm(ref), 1e-30), 1e-12)

    def test_absorb_identities(self):
        """all-ones weights absorb to x; absorb W then 1/W recovers x (edge cancellation)."""
        rng = np.random.default_rng(1)
        for struct in STRUCTURES:
            for ss in STACKS:
                with self.subTest(struct=struct, stack=ss):
                    x = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss)
                    ones = t3.T3Weights(tuple(np.ones(ss + (n,)) for n in x.tucker_ranks),
                                        tuple(np.ones(ss + (r,)) for r in x.tt_ranks))
                    self.assertLess(cw.corewise_norm(cw.corewise_sub(t3.t3_absorb_weights(x, ones).data, x.data)), 1e-12)
                    W = rand_weights(x, rng)
                    back = t3.t3_absorb_weights(t3.t3_absorb_weights(x, W), W.reciprocal())
                    self.assertLess(cw.corewise_norm(cw.corewise_sub(back.data, x.data)), 1e-9)

    def test_from_t3svd(self):
        """from_t3svd returns the (nonnegative) singular values, shape-consistent with a minimal x."""
        for struct in STRUCTURES:
            for ss in [(), (2,)]:
                with self.subTest(struct=struct, stack=ss):
                    x = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss)
                    W = t3.T3Weights.from_t3svd(x)
                    self.assertEqual((W.tucker_ranks, W.tt_ranks), x.ranks)
                    self.assertTrue(W.is_consistent_with(x))
                    for w in W.tucker_weights + W.tt_weights:
                        self.assertTrue(np.all(w >= -1e-12))                  # singular values are >= 0

    def test_is_consistent_with(self):
        """is_consistent_with: True for a matching weight; False for wrong rank / length / stack_shape."""
        rng = np.random.default_rng(2)
        x = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1), stack_shape=(3,))
        self.assertTrue(rand_weights(x, rng).is_consistent_with(x))
        bad_rank = t3.T3Weights(tuple(rng.standard_normal((3,) + (n,)) for n in (2, 2, 2)),
                                tuple(rng.standard_normal((3,) + (r,)) for r in (1, 2, 3, 1)))  # tt bond 3 != 2
        self.assertFalse(bad_rank.is_consistent_with(x))
        bad_stack = t3.T3Weights(tuple(rng.standard_normal((5,) + (n,)) for n in (2, 2, 2)),
                                 tuple(rng.standard_normal((5,) + (r,)) for r in (1, 2, 2, 1)))  # stack 5 != 3
        self.assertFalse(bad_stack.is_consistent_with(x))

    def test_validate_raises(self):
        """Structural inconsistency raises (wrong tt length; ragged stack_shape)."""
        with self.assertRaises(ValueError):
            t3.T3Weights((np.ones((2,)),), (np.ones((1,)),))                  # tt len 1 != d+1=2
        with self.assertRaises(ValueError):
            t3.T3Weights((np.ones((3, 2)),), (np.ones((3, 1)), np.ones((4, 1))))  # ragged stack (3 vs 4)

    def test_weighted_norm_inner(self):
        """weighted_norm/weighted_inner match the dense (weights-inserted) norm/inner, over structures x stacks."""
        rng = np.random.default_rng(4)
        for struct in STRUCTURES:
            for ss in STACKS:
                with self.subTest(struct=struct, stack=ss):
                    xA = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss); WA = rand_weights(xA, rng)
                    xB = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss); WB = rand_weights(xB, rng)
                    dA, dB = hand_weighted_dense(xA, WA), hand_weighted_dense(xB, WB)
                    ax = tuple(range(len(ss), dA.ndim))          # non-stack (mode) axes -> reduce to stack_shape
                    ref_norm = np.sqrt((dA ** 2).sum(axis=ax))
                    ref_inner = (dA * dB).sum(axis=ax)
                    self.assertLess(np.abs(np.asarray(t3.t3_weighted_norm(xA, WA)) - ref_norm).max(),
                                    1e-10 * (ref_norm.max() + 1))
                    self.assertLess(np.abs(np.asarray(t3.t3_weighted_inner(xA, WA, xB, WB)) - ref_inner).max(),
                                    1e-10 * (np.abs(ref_inner).max() + 1))

    def test_concatenate(self):
        """concatenate: ranks add; values are the per-edge last-axis concatenation (the '+' combine)."""
        rng = np.random.default_rng(6)
        x = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1), stack_shape=(2, 3))
        WA, WB = rand_weights(x, rng), rand_weights(x, rng)
        C = WA.concatenate(WB)
        self.assertEqual(C.tucker_ranks, tuple(2 * n for n in x.tucker_ranks))
        self.assertEqual(C.tt_ranks, tuple(2 * r for r in x.tt_ranks))
        for a, b, c in zip(WA.tucker_weights + WA.tt_weights, WB.tucker_weights + WB.tt_weights,
                           C.tucker_weights + C.tt_weights):
            self.assertTrue(np.allclose(c, np.concatenate([a, b], axis=-1)))

    def test_kronecker_hadamard(self):
        """kronecker: ranks multiply, and it IS the weight of the Hadamard product -- absorb(kron cores,
        kron weights).to_dense() == elementwise product of the two represented tensors (verifies A-major)."""
        rng = np.random.default_rng(5)
        for struct in STRUCTURES:
            for ss in [(), (2,)]:
                with self.subTest(struct=struct, stack=ss):
                    xA = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss); WA = rand_weights(xA, rng)
                    xB = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss); WB = rand_weights(xB, rng)
                    dA, dB = hand_weighted_dense(xA, WA), hand_weighted_dense(xB, WB)
                    WC = WA.kronecker(WB)
                    self.assertEqual(WC.tucker_ranks, tuple(a * b for a, b in zip(xA.tucker_ranks, xB.tucker_ranks)))
                    dC = t3.t3_absorb_weights(hadamard_cores(xA, xB), WC).to_dense()
                    self.assertLess(np.linalg.norm(dC - dA * dB) / max(np.linalg.norm(dA * dB), 1e-30), 1e-11)

    def test_structural_ops(self):
        """reverse / stack / unstack round-trips; reverse mirrors TuckerTensorTrain.reverse."""
        rng = np.random.default_rng(3)
        x = t3.TuckerTensorTrain.randn((5, 6, 7, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1), stack_shape=(2, 3))
        W = rand_weights(x, rng)
        self.assertEqual(W.reverse().tt_ranks, x.tt_ranks[::-1])
        self.assertEqual(W.reverse().reverse().tucker_ranks, W.tucker_ranks)
        Wr = t3.T3Weights.stack(W.unstack())
        for a, b in zip(Wr.tucker_weights + Wr.tt_weights, W.tucker_weights + W.tt_weights):
            self.assertTrue(np.allclose(a, b))


def make_tangent(struct, C, K):
    x = t3.TuckerTensorTrain.randn(*struct, stack_shape=C)
    frame, _ = bvf.t3_orthogonal_representations(x)
    return t3m.COREWISE.randn(frame, stack_shape=K)   # variations stack = K + C


def rand_frame_weights(v, rng, stack_shape=None):
    """A random metric at v's frame. A T3FrameWeights is FRAME-LIKE: it carries the frame stack C, NOT
    the variations' K + C (one metric per base point, shared by the K tangents there). Pass
    ``stack_shape`` to override (the tests use it to build a deliberately non-frame-like weight)."""
    V, H = v.variations.data
    ss = v.frame.stack_shape if stack_shape is None else stack_shape   # C
    d = len(V)
    return bvf.T3FrameWeights(
        tuple(rng.standard_normal(ss + (H[i].shape[-2],)) for i in range(d)),   # up   <- nU
        tuple(rng.standard_normal(ss + (V[i].shape[-2],)) for i in range(d)),   # down <- nD
        tuple(rng.standard_normal(ss + (H[i].shape[-3],)) for i in range(d)),   # left <- rL
        tuple(rng.standard_normal(ss + (H[i].shape[-1],)) for i in range(d)))   # right<- rR


def hand_frame_metric_sq(v, W):
    """Independent hand computation of the weighted tangent metric ||v||^2_W = sum of weighted Frobenius^2.

    Does its own C -> K + C broadcast (numpy right-alignment on the reshaped weight), so it checks the
    library's broadcast rather than borrowing it."""
    V, H = v.variations.data
    ss = v.variations.stack_shape        # K + C
    wss = W.stack_shape                  # C (frame-like); broadcasts over the K axes by right-alignment
    ns, d = len(ss), len(V)
    up, dn, lf, rt = W.data

    def wfro2(arr, wts):
        w = arr
        for k, wt in enumerate(wts):
            shp = wss + tuple(wt.shape[-1] if j == k else 1 for j in range(arr.ndim - ns))
            w = w * wt.reshape(shp)
        return (w * w).sum(axis=tuple(range(ns, arr.ndim)))

    return (sum(wfro2(V[i], [dn[i]]) for i in range(d))
            + sum(wfro2(H[i], [lf[i], up[i], rt[i]]) for i in range(d)))


CK_STACKS = [((), ()), ((), (2,)), ((2,), ()), ((2,), (2, 3))]   # (C, K)


class TestT3FrameWeights(unittest.TestCase):
    def test_weighted_metric(self):
        """T3Tangent.weighted_norm matches the hand metric; all-ones weights == corewise_norm; over
        structures x (C,K) stacks (including a non-trivial K=(2,3))."""
        rng = np.random.default_rng(0)
        for struct in STRUCTURES[:2]:                 # d=3 and d=4 (skip d=1: no interior)
            for C, K in CK_STACKS:
                with self.subTest(struct=struct, C=C, K=K):
                    v = make_tangent(struct, C, K)
                    W = rand_frame_weights(v, rng)
                    self.assertTrue(W.is_consistent_with(v))
                    ref = np.sqrt(np.asarray(hand_frame_metric_sq(v, W)))
                    self.assertLess(np.abs(np.asarray(v.weighted_norm(W)) - ref).max(), 1e-10 * (ref.max() + 1))
                    ones = bvf.T3FrameWeights(*[tuple(np.ones_like(w) for w in fam) for fam in W.data])
                    self.assertLess(np.abs(np.asarray(v.weighted_norm(ones)) - np.asarray(v.corewise_norm())).max(), 1e-11)

    def test_weighted_inner(self):
        """weighted_inner(self) == weighted_norm^2; symmetric; matches the metric."""
        rng = np.random.default_rng(1)
        for struct in STRUCTURES[:2]:
            for C, K in [((), ()), ((2,), (2,))]:
                with self.subTest(struct=struct, C=C, K=K):
                    v = make_tangent(struct, C, K)
                    W = rand_frame_weights(v, rng)
                    self.assertLess(np.abs(np.asarray(v.weighted_inner(v, W))
                                           - np.asarray(v.weighted_norm(W)) ** 2).max(), 1e-9)
                    w2 = t3m.COREWISE.randn(v.frame, stack_shape=K)  # another tangent at the SAME frame
                    a = np.asarray(v.weighted_inner(w2, W)); b = np.asarray(w2.weighted_inner(v, W))
                    self.assertLess(np.abs(a - b).max(), 1e-9)      # symmetry

    def test_reciprocal_identity(self):
        """down->V, up/left/right->H legs undo under reciprocal: absorb then reciprocal-absorb recovers v."""
        import t3toolbox.backend.fv_operations as fv
        rng = np.random.default_rng(2)
        v = make_tangent(STRUCTURES[0], (), (2,))
        W = rand_frame_weights(v, rng)
        wvar = fv.fv_absorb_weights(v.variations.data, W.data)
        back = fv.fv_absorb_weights(wvar, W.reciprocal().data)
        self.assertLess(cw.corewise_norm(cw.corewise_sub(back, v.variations.data)), 1e-9)

    def test_consistent_and_validate(self):
        """is_consistent_with rejects a wrong rank; validate rejects a wrong family length / ragged stack."""
        rng = np.random.default_rng(3)
        v = make_tangent(STRUCTURES[0], (), ())
        W = rand_frame_weights(v, rng)
        bad = bvf.T3FrameWeights(W.up_weights, W.down_weights, W.left_weights,
                                 tuple(np.ones(r + 1) for r in W.right_ranks))   # perturb right ranks
        self.assertFalse(bad.is_consistent_with(v))
        with self.assertRaises(ValueError):
            bvf.T3FrameWeights((np.ones(2),), (np.ones(2),), (np.ones(1),), (np.ones(2), np.ones(1)))  # ragged lengths

    def test_frame_like_stack(self):
        """A T3FrameWeights is FRAME-LIKE: a C-stacked metric pairs with a K+C tangent (the K tangents at
        one frame share the one metric) and gives the same numbers as an explicitly K-tiled weight. This is
        the canonical Grasedyck-Kramer shape -- from_t3weights(from_t3svd(x)) is C-stacked -- and was the
        case the pre-S0 predicate wrongly rejected."""
        import t3toolbox.backend.fv_operations as fv
        rng = np.random.default_rng(11)
        C, K = (3,), (5,)
        v = make_tangent(STRUCTURES[0], C, K)
        self.assertEqual(v.frame.stack_shape, C)
        self.assertEqual(v.variations.stack_shape, K + C)

        W = rand_frame_weights(v, rng)                    # stack C -- frame-like
        self.assertEqual(W.stack_shape, C)
        self.assertTrue(W.is_consistent_with(v))          # C is the trailing part of K + C

        # Tiling the metric over K must change nothing: the leading '...' broadcast IS the tiling.
        # (Reference via the backend, which is blind to the frame -- the frontend now rejects a K+C weight.)
        W_tiled = bvf.T3FrameWeights(*[tuple(np.broadcast_to(w, K + w.shape).copy() for w in fam)
                                       for fam in W.data])
        got = np.asarray(v.weighted_norm(W))
        ref = np.asarray(fv.fv_weighted_norm(v.variations.data, W_tiled.data, len(K + C)))
        self.assertEqual(got.shape, K + C)
        self.assertLess(np.abs(got - ref).max(), 1e-12 * (ref.max() + 1))

        # The GK metric really is C-stacked, and really does pair with a K-stack of tangents.
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0], stack_shape=C)
        frame, _ = bvf.t3_orthogonal_representations(x)
        gk = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(x)).reciprocal()
        self.assertEqual(gk.stack_shape, C)
        self.assertEqual(np.asarray(t3m.COREWISE.randn(frame, stack_shape=K).weighted_norm(gk)).shape, K + C)

    def test_tangent_rejects_non_frame_stack(self):
        """check_fw_pair rejects a weight whose stack is not the frame's C: a K+C weight would silently
        weight the K tangents at ONE frame with K DIFFERENT metrics. The backend predicate is blind to the
        frame and still accepts it (it reads as C_w = K+C -- that many base points, one tangent each), so a
        raw absorption is legitimate; only the tangent has enough information to reject. Two levels, both
        deliberate."""
        rng = np.random.default_rng(12)
        C, K = (3,), (5,)
        v = make_tangent(STRUCTURES[0], C, K)

        W_bad = rand_frame_weights(v, rng, stack_shape=K + C)   # NOT frame-like
        self.assertTrue(W_bad.is_consistent_with(v))            # blind predicate: a legitimate absorption
        for name, op in (('weighted_norm',  lambda: v.weighted_norm(W_bad)),
                         ('weighted_inner', lambda: v.weighted_inner(v, W_bad)),
                         ('absorb_weights', lambda: v.absorb_weights(W_bad))):
            with self.subTest(op=name), self.assertRaises(ValueError):
                op()

        # A stack that is not even a trailing part of the variation stack fails BOTH levels.
        W_worse = rand_frame_weights(v, rng, stack_shape=K)      # (5,) is not a suffix of (5, 3)
        self.assertFalse(W_worse.is_consistent_with(v))
        with self.assertRaises(ValueError):
            v.weighted_norm(W_worse)

    def test_backend_norm_inner_match_frontend(self):
        """The backend fv_weighted_norm / fv_weighted_inner equal the T3Tangent methods."""
        import t3toolbox.backend.fv_operations as fv
        rng = np.random.default_rng(7)
        v = make_tangent(STRUCTURES[0], (), (2,))
        w = t3m.COREWISE.randn(v.frame, stack_shape=(2,))
        W = rand_frame_weights(v, rng)
        ns = len(v.variations.stack_shape)
        self.assertTrue(np.allclose(np.asarray(fv.fv_weighted_norm(v.variations.data, W.data, ns)),
                                    np.asarray(v.weighted_norm(W))))
        self.assertTrue(np.allclose(np.asarray(fv.fv_weighted_inner(v.variations.data, w.variations.data, W.data, ns)),
                                    np.asarray(v.weighted_inner(w, W))))

    def test_absorb_weights_frontend(self):
        """standalone absorb_weights -> T3Variations; T3Tangent.absorb_weights -> T3Tangent (same frame,
        weighted variations); corewise_norm of the result equals weighted_norm; the result is not gauged."""
        rng = np.random.default_rng(8)
        v = make_tangent(STRUCTURES[1], (), (2,))
        W = rand_frame_weights(v, rng)
        wv_s = bvf.fv_absorb_weights(v.variations, W)
        vw = v.absorb_weights(W)
        self.assertIsInstance(wv_s, bvf.T3Variations)
        self.assertIsInstance(vw, t3m.T3Tangent)
        self.assertIs(vw.frame, v.frame)                          # same frame (untouched)
        self.assertTrue(all(np.array_equal(a, b)                  # method's variations == standalone
                            for fa, fb in zip(wv_s.data, vw.variations.data) for a, b in zip(fa, fb)))
        self.assertTrue(np.allclose(np.asarray(vw.corewise_norm()), np.asarray(v.weighted_norm(W))))

    def test_from_t3weights(self):
        """from_t3weights: up=down=tucker, left=tt[:-1], right=tt[1:]; consistent with a minimal tangent."""
        for struct in STRUCTURES[:2]:
            with self.subTest(struct=struct):
                x = t3.TuckerTensorTrain.randn(*struct)
                Wt = t3.T3Weights.from_t3svd(x)
                Wf = bvf.T3FrameWeights.from_t3weights(Wt)
                for a, b in zip(Wf.up_weights, Wt.tucker_weights):     self.assertTrue(np.array_equal(a, b))
                for a, b in zip(Wf.down_weights, Wt.tucker_weights):   self.assertTrue(np.array_equal(a, b))
                for a, b in zip(Wf.left_weights, Wt.tt_weights[:-1]):  self.assertTrue(np.array_equal(a, b))
                for a, b in zip(Wf.right_weights, Wt.tt_weights[1:]):  self.assertTrue(np.array_equal(a, b))
                frame, _ = bvf.t3_orthogonal_representations(x)        # nD==nU for minimal x
                self.assertTrue(Wf.is_consistent_with(t3m.COREWISE.randn(frame)))

    def test_concat_kron_reverse(self):
        """concatenate ranks add; kronecker ranks multiply; reverse swaps left<->right and reverses."""
        rng = np.random.default_rng(4)
        v = make_tangent(STRUCTURES[1], (2,), (2,))
        W = rand_frame_weights(v, rng)
        self.assertEqual(W.concatenate(W).up_ranks, tuple(2 * n for n in W.up_ranks))
        self.assertEqual(W.kronecker(W).left_ranks, tuple(n * n for n in W.left_ranks))
        self.assertEqual(W.reverse().left_ranks, W.right_ranks[::-1])   # left<->right swap + reverse
        Wr = bvf.T3FrameWeights.stack(W.unstack())
        self.assertTrue(all(np.allclose(a, b) for fa, fb in zip(Wr.data, W.data) for a, b in zip(fa, fb)))


class TestT3svdFrameGauge(unittest.TestCase):
    """Review 2026-08-22 (S14): the frame a singular-value metric pairs with must carry the singular basis.
    t3svd_orthogonal_representations keeps the T3-SVD gauge exactly; the default sweep does not."""

    def test_frame_is_in_the_t3svd_gauge(self):
        rng = np.random.default_rng(0)
        for structure in STRUCTURES:
            for C in STACKS:
                with self.subTest(structure=structure, stack=C):
                    np.random.seed(1)
                    x = t3.TuckerTensorTrain.randn(*structure, stack_shape=C)
                    frame, variations, sigma = bvf.t3svd_orthogonal_representations(x)
                    xs, st, stt = x.t3svd()
                    for U, Ux in zip(frame.up_tucker_cores, xs.tucker_cores):
                        self.assertLess(float(np.max(np.abs(np.asarray(U) - np.asarray(Ux)))), 1e-12)
                    self.assertTrue(np.all(frame.is_orthogonal()))
                    self.assertLess(float(np.max(np.abs(np.asarray(frame.to_dense()) - np.asarray(x.to_dense())))),
                                    1e-9 * (1 + float(np.max(np.abs(np.asarray(x.to_dense()))))))
                    W = bvf.T3FrameWeights.from_t3weights(sigma)
                    self.assertTrue(W.is_consistent_with(variations))
                    self.assertTrue(all(np.allclose(a, b) for a, b in zip(sigma.tucker_weights, st)))
                    # the default sweep (no flag) is NOT in that gauge -- the bug the helper exists for
                    if max(structure[1]) > 1:
                        default_frame, _ = bvf.t3_orthogonal_representations(xs)
                        dev = max(float(np.max(np.abs(np.abs(np.einsum('...ai,...bi->...ab', np.asarray(U), np.asarray(Ux)))
                                                         - np.eye(U.shape[-2]))))
                                  for U, Ux in zip(default_frame.up_tucker_cores, xs.tucker_cores) if U.shape[-2] > 1)
                        self.assertGreater(dev, 1e-3)

    def test_uniform_twin_matches_ragged(self):
        import t3toolbox.uniform_tucker_tensor_train as ut3
        import t3toolbox.uniform_frame_variations_format as ubvf
        for structure in STRUCTURES[:2]:
            for C in STACKS:
                with self.subTest(structure=structure, stack=C):
                    np.random.seed(2)
                    x = t3.TuckerTensorTrain.randn(*structure, stack_shape=C)
                    frame, _, sigma = bvf.t3svd_orthogonal_representations(x)
                    uframe, uvar, usigma = ubvf.ut3svd_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
                    self.assertTrue(ubvf.UT3FrameWeights.from_ut3weights(usigma).is_consistent_with(uvar))
                    self.assertTrue(np.all(uframe.is_orthogonal()))
                    self.assertLess(float(np.max(np.abs(np.asarray(uframe.to_dense()) - np.asarray(frame.to_dense())))),
                                    1e-9 * (1 + float(np.max(np.abs(np.asarray(x.to_dense()))))))
                    if not C:                                     # to_t3frame() is a tree when stacked
                        rf = uframe.to_t3frame()
                        for U, Ur in zip(rf.up_tucker_cores, frame.up_tucker_cores):
                            self.assertLess(float(np.max(np.abs(np.abs(np.asarray(U)) - np.abs(np.asarray(Ur))))), 1e-10)


if __name__ == "__main__":
    unittest.main()
