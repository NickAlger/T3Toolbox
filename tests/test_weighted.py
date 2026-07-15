# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
"""Tests for the weighted tensor-network layer (S1: T3Weights + absorb + from_t3svd + is_consistent).

Correctness is against a dense ground-truth hand-einsum with the weights inserted, checked across
structures x stack_shapes (including non-trivial, non-square stacks -- what actually exposes an axis
mistake), plus the algebraic identities (all-ones absorb = identity; absorb W then 1/W = x)."""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
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
                    xw = t3.absorb_weights(x, W)
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
                    self.assertLess(cw.corewise_norm(cw.corewise_sub(t3.absorb_weights(x, ones).data, x.data)), 1e-12)
                    W = rand_weights(x, rng)
                    back = t3.absorb_weights(t3.absorb_weights(x, W), W.reciprocal())
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
                    self.assertLess(np.abs(np.asarray(t3.weighted_norm(xA, WA)) - ref_norm).max(),
                                    1e-10 * (ref_norm.max() + 1))
                    self.assertLess(np.abs(np.asarray(t3.weighted_inner(xA, WA, xB, WB)) - ref_inner).max(),
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
                    dC = t3.absorb_weights(hadamard_cores(xA, xB), WC).to_dense()
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


if __name__ == "__main__":
    unittest.main()
